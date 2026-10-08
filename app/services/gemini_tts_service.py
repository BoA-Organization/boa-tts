from __future__ import annotations

import base64
import logging
from collections.abc import Iterator

from google import genai
from google.genai import types

from app.config import settings
from app.services.base import BaseTTSService
from app.utils.audio import pcm_to_wav

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 24000


class GeminiTTSService(BaseTTSService):
    """Speech synthesis through the Gemini API.

    Uses the Interactions API, the one that takes a speech style
    (`instructions`); its audio stream is collected into one clip.

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
        # The SDK builds the Interactions client on first access (~1.5 s);
        # do it now rather than on the first request.
        _ = self.client.interactions
        self.ready = True
        logger.info(
            "Gemini TTS ready (model %s, voice %s)",
            settings.gemini_tts_model,
            settings.gemini_tts_voice,
        )

    # ------------------------------------------------------------------
    # Public API (satisfies BaseTTSService)
    # ------------------------------------------------------------------

    def generate(self, text: str, instructions: str | None = None) -> bytes:
        if not self.ready or self.client is None:
            raise RuntimeError("Gemini TTS client is not initialized")

        # No normalize_text_for_tts: it reads numbers digit by digit, which
        # the local model needs but Gemini handles natively.
        text = text.strip()
        if not text:
            return b""

        speech_config = {"voice": settings.gemini_tts_voice}
        if settings.gemini_tts_language_code:
            speech_config["language"] = settings.gemini_tts_language_code

        content = {"type": "text", "text": text}
        if style := instructions or settings.tts_instructions:
            content["annotations"] = [{"type": "speech_metadata", "style": style}]

        events = self.client.interactions.create(
            model=settings.gemini_tts_model,
            input=[{"type": "user_input", "content": [content]}],
            response_format={
                "type": "audio",
                "mime_type": "audio/l16",
                "sample_rate": _SAMPLE_RATE,
            },
            generation_config={"speech_config": [speech_config]},
            stream=True,
        )
        chunks = list(_audio_chunks(events))
        if not chunks:
            raise RuntimeError("Gemini returned no audio")
        sample_rate = chunks[0][0]
        return pcm_to_wav(b"".join(data for _, data in chunks), f"rate={sample_rate}")


def _audio_chunks(events) -> Iterator[tuple[int, bytes]]:
    """Yield (sample rate, PCM) for each audio delta; raise on an error event."""
    try:
        for event in events:
            if event.event_type == "error":
                error = event.error
                raise RuntimeError(
                    f"Gemini TTS stream failed: {error.message if error else event}"
                )
            if event.event_type != "step.delta" or event.delta.type != "audio":
                continue
            if event.delta.data:
                rate = event.delta.sample_rate or _SAMPLE_RATE
                yield rate, base64.b64decode(event.delta.data)
    finally:
        events.close()
