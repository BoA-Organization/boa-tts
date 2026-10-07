"""Create a cloned Gemini voice from the reference recording.

Gemini needs, from the person whose voice is cloned:
  - 15-20 seconds of natural speech (AMHARIC_VOICE_REFERENCE_PATH by default)
  - a recording of them reading Google's consent phrase word for word, e.g.
    "I am the owner of this voice and I consent to Google using this voice
    to create a synthetic voice model."

Both are converted to 24 kHz mono 16-bit WAV before upload. The voice is
stored by Google and its ID is printed; set it as GEMINI_TTS_VOICE.

Usage:
    uv run python -m scripts.create_gemini_voice --consent consent.wav
"""

from __future__ import annotations

import argparse
import base64
import io

from google import genai
from pydub import AudioSegment

from app.config import settings


def _to_wav_b64(path: str) -> str:
    audio = (
        AudioSegment.from_file(path)
        .set_frame_rate(24000)
        .set_channels(1)
        .set_sample_width(2)
    )
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--consent", required=True, help="Recording of the consent phrase"
    )
    parser.add_argument(
        "--source",
        default=settings.amharic_voice_reference_path,
        help="Voice sample to clone (default: AMHARIC_VOICE_REFERENCE_PATH)",
    )
    parser.add_argument("--name", default="BOA Amharic", help="Display name")
    args = parser.parse_args()

    if not settings.gemini_api_key:
        parser.error("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=settings.gemini_api_key)
    voice = client.voices.create(
        store=True,
        voice={
            "model": settings.gemini_tts_model,
            "type": "replicated",
            "display_name": args.name,
            "replicated": {
                "source_audio": {
                    "mime_type": "audio/wav",
                    "data": _to_wav_b64(args.source),
                },
                "consent_audio": {
                    "mime_type": "audio/wav",
                    "data": _to_wav_b64(args.consent),
                },
            },
        },
    )
    print(f"Created voice {voice.id}")
    print(f"Set GEMINI_TTS_VOICE={voice.id}")


if __name__ == "__main__":
    main()
