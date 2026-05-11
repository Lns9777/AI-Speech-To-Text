import io
import wave
from typing import Literal

from config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="AI Text to Speech")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None

SpeakingStyle = Literal["natural", "storytelling", "news_reporter", "poem_reader"]

LANGUAGE_NAMES: dict[str, str] = {
    "en-US": "English (United States)",
    "en-IN": "English (India)",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "ml-IN": "Malayalam",
    "kn-IN": "Kannada",
    "mr-IN": "Marathi",
}

STYLE_PROMPTS: dict[str, str] = {
    "natural": (
        "Read this in a realistic, natural conversational voice. "
        "Use clear pronunciation, subtle emotion, comfortable pacing, and human-like pauses. "
        "Avoid sounding robotic, flat, or overly dramatic."
    ),
    "storytelling": (
        "Read this like an engaging audiobook storyteller. "
        "Use warm expression, gentle characterful emphasis, vivid pacing, and natural pauses. "
        "Let the delivery feel immersive and human without becoming theatrical."
    ),
    "news_reporter": (
        "Read this like a professional news reporter. "
        "Use a confident, crisp broadcast cadence, neutral authority, clear articulation, "
        "and measured emphasis on important facts. Keep it natural and credible."
    ),
    "poem_reader": (
        "Read this like a thoughtful poetry reader. "
        "Use a calm, lyrical cadence, expressive pauses at line breaks and punctuation, "
        "soft emotional shading, and a natural human tone."
    ),
}
DEFAULT_SPEAKING_STYLE = (
    Settings.DEFAULT_SPEAKING_STYLE
    if Settings.DEFAULT_SPEAKING_STYLE in STYLE_PROMPTS
    else "natural"
)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    voice: str = Field(default=Settings.DEFAULT_VOICE, min_length=1, max_length=40)
    language_code: str = Field(default=Settings.DEFAULT_LANGUAGE_CODE, min_length=2, max_length=12)
    speaking_style: SpeakingStyle = Field(default=DEFAULT_SPEAKING_STYLE)


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 24000) -> bytes:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return wav_buffer.getvalue()


def extract_audio_bytes(response) -> bytes:
    for candidate in response.candidates or []:
        if not candidate.content:
            continue
        for part in candidate.content.parts or []:
            if part.inline_data and part.inline_data.data:
                return part.inline_data.data
    raise ValueError("Gemini did not return audio data.")


def get_client():
    global client
    if not Settings.GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing in .env.")
    if client is None:
        client = genai.Client(api_key=Settings.GEMINI_API_KEY)
    return client


def build_tts_prompt(text: str, speaking_style: str, language_code: str) -> str:
    style_prompt = STYLE_PROMPTS.get(speaking_style, STYLE_PROMPTS["natural"])
    language_name = LANGUAGE_NAMES.get(language_code, language_code)
    return (
        f"{style_prompt}\n\n"
        f"Speak in {language_name}. "
        "If the source text is written in another language, translate it naturally into "
        f"{language_name} before speaking. Preserve the original meaning, names, numbers, "
        "and tone. Do not say these instructions aloud.\n\n"
        f"Source text:\n{text}"
    )


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/api/tts")
async def text_to_speech(payload: TTSRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please enter text to convert to speech.")

    try:
        response = await get_client().aio.models.generate_content(
            model=Settings.TTS_MODEL,
            contents=build_tts_prompt(text, payload.speaking_style, payload.language_code),
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    language_code=payload.language_code,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=payload.voice,
                        )
                    ),
                ),
            ),
        )
        wav_bytes = pcm_to_wav_bytes(extract_audio_bytes(response))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Audio generation failed: {exc}") from exc

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="generated-speech.wav"'},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000, reload=True)
