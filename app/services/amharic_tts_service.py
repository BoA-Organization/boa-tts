from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from pydub import AudioSegment

from app.config import settings
from app.services.base import BaseTTSService
from app.utils.audio import export_wav, trim_leading_trailing_silence

logger = logging.getLogger(__name__)


class AmharicTTSService(BaseTTSService):
    def __init__(self) -> None:
        self.model: OmniVoice | None = None
        self.voice_clone_prompt = None
        self.ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        logger.info("Loading Amharic TTS: %s", settings.amharic_tts_model)

        device = self._require_cuda()
        self.model = OmniVoice.from_pretrained(
            settings.amharic_tts_model,
            device_map={"": device},
            dtype=torch.float16,
            attn_implementation="eager",
        )
        self._assert_cuda_resident()

        ref_wav = Path(settings.amharic_voice_reference_path)
        ref_txt = Path(settings.amharic_voice_reference_text_path)

        if not ref_wav.exists():
            raise RuntimeError(f"Reference voice WAV not found: {ref_wav}")
        if not ref_txt.exists():
            raise RuntimeError(f"Reference voice transcript not found: {ref_txt}")

        ref_text = ref_txt.read_text(encoding="utf-8").strip()
        if not ref_text:
            raise RuntimeError(f"Reference voice transcript is empty: {ref_txt}")

        logger.info("Creating voice clone prompt from %s", ref_wav.name)
        self.voice_clone_prompt = self.model.create_voice_clone_prompt(
            ref_audio=str(ref_wav),
            ref_text=ref_text,
        )

        logger.info("Warming up Amharic TTS on %s", device)
        self._generate_chunk(settings.tts_warmup_text)
        torch.cuda.synchronize(settings.tts_cuda_device)
        self._assert_cuda_resident()

        self.ready = True
        logger.info(
            "Amharic TTS ready on %s (allocated %.2f GiB, reserved %.2f GiB)",
            device,
            torch.cuda.memory_allocated(settings.tts_cuda_device) / 1024**3,
            torch.cuda.memory_reserved(settings.tts_cuda_device) / 1024**3,
        )

    # ------------------------------------------------------------------
    # Public API (satisfies BaseTTSService)
    # ------------------------------------------------------------------

    def generate(self, text: str, instructions: str | None = None) -> bytes:
        # OmniVoice has no delivery directions; instructions are ignored.
        if not self.ready:
            raise RuntimeError("TTS model is not loaded")

        from app.utils.text_normalizer import normalize_text_for_tts

        safe_text = normalize_text_for_tts(text, "am")
        if not safe_text:
            return b""

        chunks = self._split_chunks(safe_text)
        if not chunks:
            return b""

        if len(chunks) == 1:
            return self._generate_chunk(chunks[0])

        return self._stitch_chunks(chunks)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_cuda(self) -> str:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required, but PyTorch cannot access an NVIDIA GPU"
            )
        if not 0 <= settings.tts_cuda_device < torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {settings.tts_cuda_device} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.set_device(settings.tts_cuda_device)
        return f"cuda:{settings.tts_cuda_device}"

    def _assert_cuda_resident(self) -> None:
        if self.model is None:
            raise RuntimeError("TTS model is not loaded")

        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            non_cuda = {
                name: device
                for name, device in device_map.items()
                if not str(device).startswith("cuda")
                and not isinstance(device, int)
            }
            if non_cuda:
                raise RuntimeError(
                    f"Model contains CPU/disk-offloaded layers: {non_cuda}"
                )

        modules = [self.model]
        modules.extend(
            value
            for value in vars(self.model).values()
            if isinstance(value, torch.nn.Module)
        )
        tensors = [
            tensor
            for module in modules
            if isinstance(module, torch.nn.Module)
            for tensor in (*module.parameters(), *module.buffers())
        ]
        if not tensors:
            raise RuntimeError("Could not verify model tensor placement")

        non_cuda_devices = sorted(
            {str(tensor.device) for tensor in tensors if not tensor.is_cuda}
        )
        if non_cuda_devices:
            raise RuntimeError(
                f"Model tensors are not fully resident on CUDA: {non_cuda_devices}"
            )

    def _split_chunks(self, text: str) -> list[str]:
        max_chars = max(40, settings.amharic_tts_chunk_chars)
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        sentence_parts = re.split(r"(?<=[።፧!?])\s+|\n+", normalized)
        chunks: list[str] = []
        current = ""
        for part in [p.strip() for p in sentence_parts if p.strip()]:
            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = ""
            if len(part) <= max_chars:
                current = part
                continue
            for idx in range(0, len(part), max_chars):
                subchunk = part[idx : idx + max_chars].strip()
                if subchunk:
                    chunks.append(subchunk)
        if current:
            chunks.append(current)
        return chunks

    def _generate_chunk(self, text: str) -> bytes:
        config = OmniVoiceGenerationConfig(
            num_step=32,
            guidance_scale=2.0,
            postprocess_output=True,
        )
        with torch.inference_mode():
            audio = self.model.generate(
                text=text,
                language="Amharic",
                voice_clone_prompt=self.voice_clone_prompt,
                generation_config=config,
            )
        buffer = io.BytesIO()
        sf.write(buffer, audio[0], 24000, format="WAV")
        return buffer.getvalue()

    def _stitch_chunks(self, chunks: list[str]) -> bytes:
        pause = AudioSegment.silent(
            duration=max(0, settings.amharic_tts_chunk_pause_ms)
        )
        stitched = AudioSegment.empty()

        for i, chunk in enumerate(chunks):
            logger.info("TTS chunk %d/%d: %s", i + 1, len(chunks), chunk[:40])
            chunk_bytes = self._generate_chunk(chunk)
            audio = AudioSegment.from_file(io.BytesIO(chunk_bytes), format="wav")
            audio = trim_leading_trailing_silence(audio)
            if i > 0 and len(pause) > 0:
                stitched += pause
            stitched += audio

        return export_wav(trim_leading_trailing_silence(stitched))
