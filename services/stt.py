
import assemblyai as aai
from fastapi import UploadFile
import config 

def transcribe_audio(audio_file: UploadFile) -> str:
    """Transcribes audio to text using AssemblyAI."""

    if not config.ASSEMBLYAI_API_KEY:
        raise Exception("AssemblyAI API key not set. Please provide it via /set_keys.")

    
    aai.settings.api_key = config.ASSEMBLYAI_API_KEY

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file.file)

    if transcript.status == aai.TranscriptStatus.error or not transcript.text:
        raise Exception(f"Transcription failed: {transcript.error or 'No speech detected'}")

    return transcript.text
