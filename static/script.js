
document.addEventListener("DOMContentLoaded", async () => {
  const urlParams = new URLSearchParams(window.location.search);
  let sessionId = urlParams.get("session_id");
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    window.history.replaceState({}, "", `?session_id=${sessionId}`);
  }

  let audioContext = null;
  let source = null;
  let processor = null;
  let isRecording = false;
  let socket = null;

  let audioQueue = [];
  let isPlaying = false;
  const BUFFER_SIZE = 1;

  const recordBtn = document.getElementById("recordBtn");
  const cancelBtn = document.getElementById("cancelBtn");
  const statusText = document.getElementById("statusText");
  const statusDot = document.getElementById("statusDot");
  const transcriptBox = document.getElementById("transcript");
  const responseBox = document.getElementById("llmResponse");

  
  const modal = document.getElementById("apiKeyModal");
  const saveKeysBtn = document.getElementById("saveKeysBtn");
  const murfKey = document.getElementById("murfKey");
  const assemblyKey = document.getElementById("assemblyKey");
  const geminiKey = document.getElementById("geminiKey");
  const newsKey = document.getElementById("newsKey");
  const serpKey = document.getElementById("serpKey");

  let llmBuffer = "";
  let llmStarted = false;
  let userKeysSet = false;

  const setStatus = (text, active = false, error = false) => {
    statusText.textContent = text;
    statusDot.style.backgroundColor = error ? "red" : active ? "green" : "gray";
  };

  const normalizeBase64 = (b64) => {
    if (!b64) return "";
    let s = b64.replace(/\s+/g, "");
    const pad = s.length % 4;
    if (pad) s += "=".repeat(4 - pad);
    return s;
  };

  const playNextInQueue = () => {
    if (audioQueue.length === 0) {
      isPlaying = false;
      return;
    }
    isPlaying = true;
    const buffer = audioQueue.shift();
    const src = audioContext.createBufferSource();
    src.buffer = buffer;
    src.connect(audioContext.destination);
    src.onended = playNextInQueue;
    src.start();
  };

  
  saveKeysBtn.addEventListener("click", async () => {
    const keys = {
      murf: murfKey.value.trim(),
      assembly: assemblyKey.value.trim(),
      gemini: geminiKey.value.trim(),
      news: newsKey.value.trim(),
      serp: serpKey.value.trim(),
      session_id: sessionId
    };

    if (!keys.murf || !keys.assembly || !keys.gemini || !keys.news || !keys.serp) {
      alert("All keys are required!");
      return;
    }

    try {
      const res = await fetch("/set_keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(keys)
      });
      if (res.ok) {
        userKeysSet = true;
        modal.style.display = "none";
        console.log("✅ API keys saved, assistant ready.");
      } else {
        alert("Failed to save keys.");
      }
    } catch (err) {
      console.error("Error sending keys:", err);
      alert("Error saving keys.");
    }
  });

  
  const startRecording = async () => {
    if (!userKeysSet) {
      alert("Please enter your API keys before starting.");
      modal.style.display = "flex"; 
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      alert("Audio recording not supported in this browser.");
      return;
    }

    isRecording = true;
    recordBtn.classList.add("recording");
    setStatus("Streaming audio... Click to stop.", true);

    transcriptBox.innerHTML = `<p class="placeholder">Listening...</p>`;

    llmBuffer = "";
    llmStarted = false;
    audioQueue = [];
    isPlaying = false;

    try {
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);

      socket.onopen = async () => {
        setStatus("Connected. Speak now!", true);
        try {
          socket.send(JSON.stringify({ type: "session", session_id: sessionId }));
        } catch (err) {
          console.warn("Failed sending session message:", err);
        }

        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

          audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
          source = audioContext.createMediaStreamSource(stream);
          processor = audioContext.createScriptProcessor(4096, 1, 1);

          processor.onaudioprocess = (event) => {
            const inputData = event.inputBuffer.getChannelData(0);
            const pcmData = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
              const sample = Math.max(-1, Math.min(1, inputData[i]));
              pcmData[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
            }
            if (socket && socket.readyState === WebSocket.OPEN) {
              socket.send(pcmData.buffer);
            }
          };

          source.connect(processor);
          processor.connect(audioContext.destination);
          recordBtn.mediaStream = stream;
        } catch (micError) {
          alert("Mic access denied.");
          stopRecording(true);
        }
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "status") {
            setStatus(data.message, true);
          }
          if (data.type === "transcription") {
            if (data.is_final) {
              transcriptBox.innerHTML = `<p class="final-transcript">${data.text}</p>`;
              setStatus("Turn completed. Processing response...");
              llmStarted = false;
            } else {
              transcriptBox.innerHTML = `<p class="interim-transcript">${data.text}</p>`;
            }
          }
          if (
            data.type === "llm_response_text" ||
            data.type === "llm_response" ||
            data.type === "response.text.delta"
          ) {
            const chunk = data.text || data.delta || data.message || "";
            if (chunk) {
              if (!llmStarted) {
                responseBox.innerHTML = `<p class="placeholder">Thinking...</p>`;
                llmBuffer = "";
                llmStarted = true;
              }
              llmBuffer += chunk;
              responseBox.innerHTML = `<p class="llm">${llmBuffer}</p>`;
            }
          }
          if (data.type === "audio_chunk") {
            const b64 = normalizeBase64(data.audio || data.audio_data || "");
            if (b64) {
              const binary = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer;
              audioContext.decodeAudioData(binary).then((buffer) => {
                audioQueue.push(buffer);
                if (!isPlaying && audioQueue.length >= BUFFER_SIZE) {
                  playNextInQueue();
                }
              }).catch((err) => {
                console.error("[Client] decodeAudioData error:", err);
              });
            }
          }
          if (data.type === "audio_complete") {
            setStatus("AI response completed. Continue speaking or stop recording.");
            if (!isPlaying && audioQueue.length > 0) {
              setTimeout(() => {
                if (!isPlaying) playNextInQueue();
              }, 300);
            }
          }
          if (data.type === "error") {
            console.error("Error:", data.message);
            setStatus(`Error: ${data.message}`, false, true);
          }
        } catch (err) {
          console.error("Error handling socket message:", err, event.data);
        }
      };

      socket.onclose = () => setStatus("Idle");
      socket.onerror = () => setStatus("Connection error", false, true);
    } catch (err) {
      alert("Failed to start recording session.");
      stopRecording(true);
    }
  };

  const stopRecording = (error = false) => {
    if (!isRecording) return;
    isRecording = false;
    recordBtn.classList.remove("recording");
    setStatus(error ? "Error" : "Idle", false, error);
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (audioContext) audioContext.close();
    if (recordBtn.mediaStream) {
      recordBtn.mediaStream.getTracks().forEach((track) => track.stop());
      recordBtn.mediaStream = null;
    }
    if (socket && socket.readyState === WebSocket.OPEN) {
      try {
        socket.send("EOF");
      } catch (e) {}
      socket.close();
    }
    socket = null;
  };

  recordBtn.addEventListener("click", () => {
    if (isRecording) stopRecording();
    else startRecording();
  });

  cancelBtn.addEventListener("click", () => {
    stopRecording();
    transcriptBox.innerHTML = `<p class="placeholder">Your speech will appear here...</p>`;
    responseBox.innerHTML = `<p class="placeholder">Assistant response will appear here...</p>`;
    llmBuffer = "";
    llmStarted = false;
  });

  window.addEventListener("beforeunload", () => {
    if (isRecording) stopRecording();
  });

  if (!userKeysSet) {
    modal.style.display = "flex";
  }
});
