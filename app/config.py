import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "BOA TTS Service"

    # "local" runs the OmniVoice model on the GPU; "gemini" calls the Gemini
    # API; "openrouter" calls Gemini TTS through OpenRouter
    tts_provider: Literal["local", "gemini", "openrouter"] = "local"

    hf_home: str = "/models/hf_cache"
    model_cache_dir: str = "/models/cache"

    tts_cuda_device: int = 0
    tts_warmup_text: str = (
        "ሰላም፣ ይህ የድምፅ ማሞቂያ ሙከራ ነው።"
    )

    amharic_tts_model: str = "gheero-Leyu/amharic-omnivoice-tts"
    amharic_voice_reference_path: str = str(
        Path(__file__).resolve().parents[1]
        / "assets/voices/amharic_reference_voice.wav"
    )
    amharic_voice_reference_text_path: str = str(
        Path(__file__).resolve().parents[1]
        / "assets/voices/amharic_reference_voice.txt"
    )
    amharic_tts_chunk_chars: int = 140
    amharic_tts_chunk_pause_ms: int = 120

    gemini_api_key: str = ""
    gemini_tts_model: str = "gemini-3.8-flash-tts"
    # Prebuilt voice name (e.g. "Kore") or a cloned voice ID (voice_...)
    gemini_tts_voice: str = "Kore"
    # Empty lets Gemini detect the language from the text
    gemini_tts_language_code: str = ""

    openrouter_api_key: str = ""
    openrouter_tts_model: str = "google/gemini-3.8-flash-tts"
    # Prebuilt voice name only; cloned voices are not available via OpenRouter
    openrouter_tts_voice: str = "Kore"
    # Upstream calls in flight at once; extra requests wait for a free slot
    openrouter_max_concurrency: int = 4
    # Each attempt is cut off after this long, then retried
    openrouter_attempt_timeout_seconds: int = 60
    # Extra attempts after a timeout, 408, 429 or 5xx
    openrouter_max_retries: int = 2

    # Default delivery directions (tone, pace) for the gemini and openrouter
    # providers, e.g. "calm and clear"; a request's `instructions` override it
    tts_instructions: str = ""

    request_timeout_seconds: int = 300

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
os.environ.setdefault("HF_HOME", settings.hf_home)
