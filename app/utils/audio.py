import io
import logging
import re
import subprocess
import wave

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# Gemini TTS emits 24 kHz mono signed 16-bit PCM
_DEFAULT_PCM_RATE = 24000


def bytes_to_audio_segment(audio_bytes: bytes) -> AudioSegment:
    return AudioSegment.from_file(io.BytesIO(audio_bytes))


def trim_leading_trailing_silence(
    audio: AudioSegment,
    *,
    min_silence_len: int = 300,
    silence_thresh: float | None = None,
    keep_leading_ms: int = 20,
    keep_trailing_ms: int = 50,
) -> AudioSegment:
    if len(audio) == 0:
        return audio

    if silence_thresh is None:
        silence_thresh = audio.dBFS - 16

    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )
    if not ranges:
        return audio

    start_ms = max(0, ranges[0][0] - keep_leading_ms)
    end_ms = min(len(audio), ranges[-1][1] + keep_trailing_ms)
    if end_ms <= start_ms:
        return audio
    return audio[start_ms:end_ms]


def speed_adjust_wav(audio: AudioSegment, speed: float) -> AudioSegment:
    if speed == 1.0:
        return audio

    in_buf = io.BytesIO()
    audio.export(in_buf, format="wav")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-filter:a",
            f"atempo={speed}",
            "-f",
            "wav",
            "pipe:1",
        ],
        input=in_buf.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        logging.getLogger(__name__).warning(
            "ffmpeg atempo failed; returning unmodified audio (rc=%s, err=%s)",
            proc.returncode,
            proc.stderr.decode("utf-8", errors="ignore")[:500],
        )
        return audio
    return AudioSegment.from_file(io.BytesIO(proc.stdout), format="wav")


def export_wav(audio: AudioSegment) -> bytes:
    out_buf = io.BytesIO()
    audio.export(out_buf, format="wav")
    return out_buf.getvalue()


def pcm_to_wav(data: bytes, mime_type: str) -> bytes:
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
