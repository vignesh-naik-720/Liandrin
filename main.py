
from fastapi import FastAPI, Request, UploadFile, File, Path, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, List, Any, Type
import logging
from pathlib import Path as PathLib
from uuid import uuid4
import json
import asyncio
import time
import base64
import os
from dotenv import load_dotenv, set_key
import config
from services import stt, llm, tts
from schemas import TTSRequest
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
chat_histories: Dict[str, List[Dict[str, Any]]] = {}

BASE_DIR = PathLib(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
FALLBACK_AUDIO_PATH = BASE_DIR / "static" / "fallback.mp3"


if not ENV_PATH.exists():
    ENV_PATH.touch()


load_dotenv(dotenv_path=ENV_PATH, override=True)


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/set_keys")
async def set_keys(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")

        
        env_vars = {
            "MURF_API_KEY": data.get("murf"),
            "ASSEMBLYAI_API_KEY": data.get("assembly"),
            "GEMINI_API_KEY": data.get("gemini"),
            "NEWS_API_KEY": data.get("news"),
            "SERP_API_KEY": data.get("serp")
        }

        
        for key, value in env_vars.items():
            if value:
                set_key(str(ENV_PATH), key, value)
                os.environ[key] = value  
                setattr(config, key, value)  

        load_dotenv(dotenv_path=ENV_PATH, override=True)

        logging.info(f"API keys set and saved for session {session_id}")
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        logging.error(f"Failed to set API keys: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to set keys"})



@app.post("/agent/chat/{session_id}")
async def agent_chat(
    session_id: str = Path(...),
    audio_file: UploadFile = File(...),
):
    if not all([config.GEMINI_API_KEY, config.ASSEMBLYAI_API_KEY, config.MURF_API_KEY]):
        return FileResponse(FALLBACK_AUDIO_PATH, media_type="audio/mpeg", headers={"X-Error": "true"})
    try:
        user_text = stt.transcribe_audio(audio_file)
        history = chat_histories.get(session_id, [])
        
        history.append({"role": "user", "text": user_text, "ts": time.time()})
        llm_resp, updated = llm.get_llm_response(user_text, history)
        chat_histories[session_id] = updated
        audio_url = tts.convert_text_to_speech(llm_resp)
        if audio_url:
            return JSONResponse(content={"audio_url": audio_url})
        raise Exception("TTS failed")
    except Exception as e:
        logging.error(f"Error in session {session_id}: {e}")
        return FileResponse(FALLBACK_AUDIO_PATH, media_type="audio/mpeg", headers={"X-Error": "true"})


@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    try:
        audio_url = tts.convert_text_to_speech(request.text, request.voiceId)
        return JSONResponse(content={"audio_url": audio_url}) if audio_url else JSONResponse(status_code=500, content={"error": "No audio URL"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS failed: {e}"})


@app.get("/voices")
async def get_voices():
    try:
        return JSONResponse(content={"voices": tts.get_available_voices()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to fetch voices: {e}"})


async def llm_stream_wrapper(prompt: str, session_history: List[Dict[str, Any]] = None):
    session_history = session_history or []
    try:
        if hasattr(llm, "stream_llm_response"):
            func = llm.stream_llm_response
            try:
                gen = func(prompt, session_history)
            except TypeError:
                gen = func(prompt)

            if hasattr(gen, "__aiter__"):
                async for chunk in gen:
                    yield chunk
            else:
                for chunk in gen:
                    yield chunk
        else:
            if hasattr(llm, "get_llm_response"):
                resp = llm.get_llm_response(prompt, session_history)
                text = resp[0] if isinstance(resp, tuple) else resp
                if text:
                    yield text
            else:
                yield "[no LLM available]"
    except Exception:
        logging.exception("llm_stream_wrapper error")
        yield "[llm error]"


async def llm_tts_pipeline(text: str, websocket: WebSocket, session_id: str = None):
    logging.info(f"[pipeline] start pipeline for session={session_id} text: {text!r}")
    tts_queue = asyncio.Queue()
    collected_chunks = []

    if session_id:
        chat_histories.setdefault(session_id, []).append(
            {"role": "user", "text": text, "ts": time.time()}
        )

    async def llm_worker():
        try:
            async for chunk in llm_stream_wrapper(text, chat_histories.get(session_id, [])):
                if chunk:
                    try:
                        await websocket.send_json({"type": "llm_response_text", "text": chunk})
                    except Exception:
                        logging.exception("[pipeline] failed sending llm_response_text")

                    await tts_queue.put(chunk)
                    collected_chunks.append(chunk)
        except Exception:
            logging.exception("[pipeline] llm_worker error")
        finally:
            await tts_queue.put(None)
            logging.info("[pipeline] llm_worker finished")

    async def tts_worker():
        chunk_count = 0
        try:
            while True:
                chunk = await tts_queue.get()
                if chunk is None:
                    break
                try:
                    audio_bytes = tts.speak(chunk)
                    if audio_bytes:
                        chunk_count += 1
                        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "chunk_index": chunk_count,
                            "audio": b64_audio,
                            "is_final": False
                        })
                        logging.info(f"[pipeline] sent audio chunk #{chunk_count} (size={len(audio_bytes)} bytes)")
                except Exception:
                    logging.exception("[pipeline] TTS error for chunk")
                finally:
                    tts_queue.task_done()
        except Exception:
            logging.exception("[pipeline] tts_worker top-level error")
        finally:
            try:
                await websocket.send_json({
                    "type": "audio_complete",
                    "message": "Audio streaming completed",
                    "total_chunks": chunk_count
                })
                logging.info(f"[pipeline] audio_complete sent, total_chunks={chunk_count}")
            except Exception:
                logging.exception("[pipeline] failed sending audio_complete")

    await asyncio.gather(
        asyncio.create_task(llm_worker()),
        asyncio.create_task(tts_worker())
    )
    logging.info("[pipeline] finished all tasks")

    try:
        full_response = "".join(collected_chunks).strip()
        if session_id and full_response:
            chat_histories.setdefault(session_id, []).append(
                {"role": "assistant", "text": full_response, "ts": time.time()}
            )
            logging.info(f"[pipeline] saved assistant message to chat_histories[{session_id}]")
    except Exception:
        logging.exception("[pipeline] failed saving chat history")


@app.websocket("/ws")
async def websocket_audio_streaming(websocket: WebSocket):
    await websocket.accept()

    if not config.ASSEMBLYAI_API_KEY:
        await websocket.send_text(json.dumps({"type": "error", "message": "AssemblyAI API key not configured"}))
        await websocket.close(code=1000)
        return
    if not config.GEMINI_API_KEY:
        await websocket.send_text(json.dumps({"type": "error", "message": "Gemini API key not configured"}))
        await websocket.close(code=1000)
        return
    if not config.MURF_API_KEY:
        await websocket.send_text(json.dumps({"type": "error", "message": "Murf API key not configured"}))
        await websocket.close(code=1000)
        return

    queue: asyncio.Queue = asyncio.Queue()
    processed_turns = set()
    last_turn_time = 0.0
    main_loop = asyncio.get_running_loop()
    scheduled_futures = set()

    client = StreamingClient(
        StreamingClientOptions(api_key=config.ASSEMBLYAI_API_KEY, api_host="streaming.assemblyai.com")
    )

    session_id: str | None = None

    def on_turn(self: Type[StreamingClient], event: TurnEvent):
        nonlocal processed_turns, last_turn_time
        text = (event.transcript or "").strip()
        current_time = time.time()

        if text:
            main_loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "transcription",
                "text": text,
                "is_final": event.end_of_turn,
                "end_of_turn": event.end_of_turn
            })

        if event.end_of_turn:
            main_loop.call_soon_threadsafe(queue.put_nowait, {"type": "turn_end", "message": "User stopped talking"})
            normalized = " ".join(text.lower().split())
            if text and len(text) > 3 and normalized not in processed_turns and (current_time - last_turn_time) > 1.5:
                processed_turns.add(normalized)
                last_turn_time = current_time
                try:
                    if session_id:
                        chat_histories.setdefault(session_id, []).append(
                            {"role": "user", "text": text, "ts": time.time()}
                        )

                    future = asyncio.run_coroutine_threadsafe(llm_tts_pipeline(text, websocket, session_id), main_loop)
                    scheduled_futures.add(future)
                    logging.info("[ws] scheduled llm_tts_pipeline")
                except Exception:
                    logging.exception("LLM scheduling error")
                    main_loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": "LLM scheduling error"})

    client.on(StreamingEvents.Turn, on_turn)
    sender = asyncio.create_task(send_loop(queue, websocket))

    try:
        client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
                enable_extra_session_information=True
            )
        )
        await websocket.send_text(json.dumps({"type": "status", "message": "Connected to transcription service"}))

        with open(UPLOADS_DIR / f"streamed_{uuid4().hex}.pcm", "wb") as f:
            while True:
                msg = await websocket.receive()
                if isinstance(msg, bytes):
                    f.write(msg)
                    client.stream(msg)
                elif isinstance(msg, dict):
                    if "bytes" in msg and msg["bytes"]:
                        f.write(msg["bytes"])
                        client.stream(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        text_msg = msg["text"]
                        if text_msg == "EOF":
                            break
                        try:
                            parsed = json.loads(text_msg)
                            if isinstance(parsed, dict) and parsed.get("type") == "session":
                                session_id = parsed.get("session_id")
                                logging.info(f"[ws] session_id set from client: {session_id}")
                                chat_histories.setdefault(session_id, chat_histories.get(session_id, []))
                        except Exception:
                            pass
                elif isinstance(msg, str):
                    if msg == "EOF":
                        break
                    try:
                        parsed = json.loads(msg)
                        if isinstance(parsed, dict) and parsed.get("type") == "session":
                            session_id = parsed.get("session_id")
                            logging.info(f"[ws] session_id set from client: {session_id}")
                            chat_histories.setdefault(session_id, chat_histories.get(session_id, []))
                    except Exception:
                        pass
                else:
                    break

    except WebSocketDisconnect:
        logging.info("Client disconnected")
    except Exception as e:
        logging.exception(f"WebSocket error: {e}")
    finally:
        sender.cancel()
        for fut in list(scheduled_futures):
            try:
                if not fut.done():
                    fut.cancel()
            except Exception:
                pass
        scheduled_futures.clear()
        try:
            client.disconnect(terminate=True)
        except Exception:
            pass


async def send_loop(queue: asyncio.Queue, websocket: WebSocket):
    while True:
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=0.1)
            await websocket.send_text(json.dumps(msg))
            queue.task_done()
        except asyncio.TimeoutError:
            continue
        except Exception:
            logging.exception("Failed sending message in send_loop")
            break
    logging.info("Send loop ended")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
