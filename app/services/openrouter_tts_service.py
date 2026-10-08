from __future__ import annotations

import logging

import httpx

from app.config import settings
from app.services.base import BaseTTSService
from app.utils.audio import pcm_to_wav

logger = logging.getLogger(__name__)

_SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"


class OpenRouterTTSService(BaseTTSService):
    """Speech synthesis through OpenRouter's /audio/speech endpoint.

    `OPENROUTER_TTS_VOICE` must be a prebuilt voice name (e.g. "Kore"); cloned
    Gemini voice IDs only exist in the Google project that created them.
    """

    def __init__(self) -> None:
        self.client: httpx.Client | None = None
        self.ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when TTS_PROVIDER=openrouter"
            )
        if not settings.openrouter_tts_voice:
            raise RuntimeError(
                "OPENROUTER_TTS_VOICE is required when TTS_PROVIDER=openrouter"
            )

        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=settings.request_timeout_seconds,
        )
        self.ready = True
        logger.info(
            "OpenRouter TTS ready (model %s, voice %s)",
            settings.openrouter_tts_model,
            settings.openrouter_tts_voice,
        )

    # ------------------------------------------------------------------
    # Public API (satisfies BaseTTSService)
    # ------------------------------------------------------------------

    def generate(self, text: str, instructions: str | None = None) -> bytes:
        if not self.ready or self.client is None:
            raise RuntimeError("OpenRouter TTS client is not initialized")

        # No normalize_text_for_tts: Gemini reads numbers natively.
        text = text.strip()
        if not text:
            return b""

        payload = {
            "model": settings.openrouter_tts_model,
            "input": text,
            "voice": settings.openrouter_tts_voice,
            # Gemini TTS on OpenRouter rejects anything but raw PCM
            "response_format": "pcm",
        }
        if style := instructions or settings.tts_instructions:
            payload["instructions"] = style

        response = self.client.post(_SPEECH_URL, json=payload)
        if response.is_error:
            raise RuntimeError(
                f"OpenRouter TTS failed ({response.status_code}): {response.text[:500]}"
            )
        if not response.content:
            raise RuntimeError("OpenRouter returned no audio")

        return pcm_to_wav(response.content, response.headers.get("content-type", ""))
