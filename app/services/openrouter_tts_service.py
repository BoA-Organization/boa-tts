from __future__ import annotations

import logging
import random
import threading
import time

import httpx

from app.config import settings
from app.services.base import BaseTTSService
from app.utils.audio import pcm_to_wav

logger = logging.getLogger(__name__)

_SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"

# Statuses worth another attempt: timeouts, rate limits, upstream failures
# (524 is Cloudflare's "origin timed out")
_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504, 524}
_MAX_BACKOFF_SECONDS = 10.0


class OpenRouterTTSService(BaseTTSService):
    """Speech synthesis through OpenRouter's /audio/speech endpoint.

    `OPENROUTER_TTS_VOICE` must be a prebuilt voice name (e.g. "Kore"); cloned
    Gemini voice IDs only exist in the Google project that created them.
    """

    def __init__(self) -> None:
        self.client: httpx.Client | None = None
        self.slots: threading.BoundedSemaphore | None = None
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
            timeout=httpx.Timeout(
                settings.openrouter_attempt_timeout_seconds, connect=10.0
            ),
        )
        self.slots = threading.BoundedSemaphore(settings.openrouter_max_concurrency)
        self.ready = True
        logger.info(
            "OpenRouter TTS ready (model %s, voice %s, concurrency %d)",
            settings.openrouter_tts_model,
            settings.openrouter_tts_voice,
            settings.openrouter_max_concurrency,
        )

    # ------------------------------------------------------------------
    # Public API (satisfies BaseTTSService)
    # ------------------------------------------------------------------

    def generate(self, text: str, instructions: str | None = None) -> bytes:
        if not self.ready or self.client is None or self.slots is None:
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

        with self.slots:
            response = self._post_with_retries(payload)
        if not response.content:
            raise RuntimeError("OpenRouter returned no audio")

        return pcm_to_wav(response.content, response.headers.get("content-type", ""))

    def _post_with_retries(self, payload: dict) -> httpx.Response:
        """POST the request, retrying timeouts and transient upstream errors."""
        attempts = settings.openrouter_max_retries + 1
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            retry_after: str | None = None
            try:
                response = self.client.post(_SPEECH_URL, json=payload)
            except httpx.TransportError as exc:  # includes timeouts
                failure = f"{type(exc).__name__}: {exc}"
            else:
                if not response.is_error:
                    logger.info(
                        "OpenRouter TTS took %.1fs (attempt %d)",
                        time.monotonic() - started,
                        attempt,
                    )
                    return response
                failure = f"{response.status_code}: {response.text[:500]}"
                if response.status_code not in _RETRYABLE_STATUSES:
                    raise RuntimeError(f"OpenRouter TTS failed ({failure})")
                retry_after = response.headers.get("retry-after")

            if attempt == attempts:
                raise RuntimeError(
                    f"OpenRouter TTS failed after {attempts} attempts ({failure})"
                )
            delay = _backoff_seconds(attempt, retry_after)
            logger.warning(
                "OpenRouter TTS attempt %d/%d failed after %.1fs (%s); "
                "retrying in %.1fs",
                attempt,
                attempts,
                time.monotonic() - started,
                failure,
                delay,
            )
            time.sleep(delay)
        raise AssertionError("unreachable")


def _backoff_seconds(attempt: int, retry_after: str | None) -> float:
    """Honor a numeric Retry-After, else exponential backoff with jitter."""
    if retry_after and retry_after.isdigit():
        return min(float(retry_after), _MAX_BACKOFF_SECONDS)
    return min(2 ** (attempt - 1), _MAX_BACKOFF_SECONDS) + random.uniform(0, 1)  # noqa: S311
