import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TTS_MODEL = os.getenv("TTS_MODEL", "gemini-2.5-flash-preview-tts")
    # TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.1-flash-live-preview")
    DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "Kore")
    DEFAULT_LANGUAGE_CODE = os.getenv("DEFAULT_LANGUAGE_CODE", "en-US")
    DEFAULT_SPEAKING_STYLE = os.getenv("DEFAULT_SPEAKING_STYLE", "natural")
