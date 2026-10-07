from __future__ import annotations

import io
import logging
import re
import wave

from google import genai
from google.genai import types

from app.config import settings
from app.services.base import BaseTTSService

logger = logging.getLogger(__name__)

# Custom voices from `voices.create`: stored (voice_...) or client-held (voicekey_...)
_CUSTOM_VOICE_PREFIXES = ("voice_", "voicekey_")
_DEFAULT_PCM_RATE = 24000


class GeminiTTSService(BaseTTSService):
    """Speech synthesis through the Gemini API.

    `GEMINI_TTS_VOICE` is either a prebuilt voice name (e.g. "Kore") or the
    ID of a cloned voice created with `scripts/create_gemini_voice.py`.
    """

    def __init__(self) -> None:
        self.client: genai.Client | None = None
        self.ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when TTS_PROVIDER=gemini")
        if not settings.gemini_tts_voice:
            raise RuntimeError("GEMINI_TTS_VOICE is required when TTS_PROVIDER=gemini")

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=settings.request_timeout_seconds * 1000
            ),
        )
        self.ready = True
        logger.info(
            "Gemini TTS ready (model %s, voice %s)",
            settings.gemini_tts_model,
            settings.gemini_tts_voice,
        )

    # ------------------------------------------------------------------
    # Public API (satisfies BaseTTSService)
    # ------------------------------------------------------------------

    def generate(self, text: str) -> bytes:
        if not self.ready or self.client is None:
            raise RuntimeError("Gemini TTS client is not initialized")

        # No normalize_text_for_tts: it reads numbers digit by digit, which
        # the local model needs but Gemini handles natively.
        text = text.strip()
        if not text:
            return b""

        response = self.client.models.generate_content(
            model=settings.gemini_tts_model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=self._speech_config(),
            ),
        )
        return self._extract_wav(response)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _speech_config(self) -> types.SpeechConfig:
        voice = settings.gemini_tts_voice
        if voice.startswith(_CUSTOM_VOICE_PREFIXES):
            voice_config = types.VoiceConfig(voice=voice)
        else:
            voice_config = types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        return types.SpeechConfig(
            voice_config=voice_config,
            language_code=settings.gemini_tts_language_code or None,
        )

    @staticmethod
    def _extract_wav(response: types.GenerateContentResponse) -> bytes:
        for candidate in response.candidates or []:
            for part in (candidate.content.parts if candidate.content else None) or []:
                blob = part.inline_data
                if blob and blob.data:
                    return _as_wav(blob.data, blob.mime_type or "")

        reason = response.candidates[0].finish_reason if response.candidates else None
        raise RuntimeError(f"Gemini returned no audio (finish reason: {reason})")


def _as_wav(data: bytes, mime_type: str) -> bytes:
    """Return WAV bytes, wrapping raw PCM (audio/L16;rate=...) in a WAV header."""
    if data.startswith(b"RIFF"):
        return data

    match = re.search(r"rate=(\d+)", mime_type)
    rate = int(match.group(1)) if match else _DEFAULT_PCM_RATE
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # signed 16-bit little-endian
        wav.setframerate(rate)
        wav.writeframes(data)
    return buffer.getvalue()
