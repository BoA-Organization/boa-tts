from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.utils.audio import wav_to_pcm


class BaseTTSService(ABC):
    """Abstract base class for all language-specific TTS services."""

    ready: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """Load the model and any required assets. Called once at startup."""

    @abstractmethod
    def generate(self, text: str, instructions: str | None = None) -> bytes:
        """Synthesize speech from text and return raw WAV bytes.

        `instructions` are delivery directions (tone, pace, emotion) for
        providers that support them; others ignore them.
        """

    def stream(
        self, text: str, instructions: str | None = None
    ) -> Iterator[tuple[int, bytes]]:
        """Yield (sample rate, signed 16-bit mono PCM) chunks as they are made.

        Yields nothing for text that produces no audio. This default sends the
        whole `generate` clip as one chunk; providers that synthesize
        incrementally override it.
        """
        wav_bytes = self.generate(text, instructions)
        if wav_bytes:
            yield wav_to_pcm(wav_bytes)
