import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Literal, Self

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, model_validator

from app.config import settings
from app.services import registry  # noqa: F401 — triggers __init__ registration
from app.services import registry as svc_registry
from app.services.base import BaseTTSService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    text: str
    # Delivery directions for this clip (tone, pace, emotion), e.g. "warm and
    # inviting". Overrides TTS_INSTRUCTIONS; the local provider ignores it.
    instructions: str | None = None
    # "wav" answers JSON with a base64 WAV clip. "pcm" answers raw signed
    # 16-bit mono PCM, with its rate in the X-Sample-Rate header.
    format: Literal["wav", "pcm"] = "wav"
    # Send the PCM as it is synthesized instead of once the clip is done
    stream: bool = False

    @model_validator(mode="after")
    def _stream_needs_pcm(self) -> Self:
        if self.stream and self.format != "pcm":
            raise ValueError("stream=true requires format=pcm")
        return self


class TTSResponse(BaseModel):
    audio_base64: str
    media_type: str = "audio/wav"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s", settings.app_name)
    languages = svc_registry.all_languages()
    for lang in languages:
        service = svc_registry.get(lang)
        logger.info("Initializing TTS service for language: %s", lang)
        await asyncio.to_thread(service.initialize)
        logger.info("TTS service ready for language: %s", lang)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan, docs_url=None)


@app.get("/docs", include_in_schema=False)
def swagger_ui() -> HTMLResponse:
    """Serve Swagger UI with a browser-native player for generated speech."""
    swagger = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{settings.app_name} - Swagger UI",
    )
    languages = json.dumps(svc_registry.all_languages())
    player = f"""
    <section id="tts-player" style="max-width: 1460px; margin: 18px auto;
      padding: 18px; font-family: sans-serif; border: 1px solid #d8dde3;
      border-radius: 6px; box-sizing: border-box;">
      <h2 style="margin-top: 0;">Try TTS &amp; listen</h2>
      <form id="tts-player-form">
        <label>Language
          <select id="tts-language" style="margin: 0 12px 10px 6px;"></select>
        </label>
        <label style="display: block; margin-bottom: 10px;">Text
          <textarea id="tts-text" required rows="3" style="display: block; width: 100%;
            margin-top: 5px; box-sizing: border-box;"></textarea>
        </label>
        <label style="display: block; margin-bottom: 10px;">Instructions (optional)
          <input id="tts-instructions" placeholder="e.g. warm and inviting"
            style="display: block; width: 100%; margin-top: 5px; box-sizing: border-box;">
        </label>
        <button type="submit" style="padding: 8px 18px; cursor: pointer;">
          Generate audio
        </button>
        <span id="tts-status" role="status" style="margin-left: 10px;"></span>
      </form>
      <audio id="tts-audio" controls
        style="display: none; width: 100%; margin-top: 14px;"></audio>
    </section>
    <script>
      const languages = {languages};
      const languageSelect = document.getElementById('tts-language');
      for (const language of languages) {{
        languageSelect.add(new Option(language, language));
      }}

      let currentAudioUrl;
      const playerForm = document.getElementById('tts-player-form');
      playerForm.addEventListener('submit', async (event) => {{
        event.preventDefault();
        const status = document.getElementById('tts-status');
        const audio = document.getElementById('tts-audio');
        status.textContent = 'Generating...';
        audio.style.display = 'none';

        try {{
          const response = await fetch(`/tts/${{encodeURIComponent(languageSelect.value)}}`, {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{
              text: document.getElementById('tts-text').value,
              instructions: document.getElementById('tts-instructions').value || null,
            }}),
          }});
          if (!response.ok) {{
            let message = `Request failed (${{response.status}})`;
            try {{
              const error = await response.json();
              message = error.detail || message;
            }} catch (_) {{}}
            throw new Error(message);
          }}

          const result = await response.json();
          const binary = atob(result.audio_base64);
          const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
          if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
          currentAudioUrl = URL.createObjectURL(
            new Blob([bytes], {{type: result.media_type}})
          );
          audio.src = currentAudioUrl;
          audio.style.display = 'block';
          status.textContent = 'Ready to play';
          audio.load();
        }} catch (error) {{
          status.textContent = error.message;
        }}
      }});
    </script>
    """
    html = swagger.body.decode("utf-8").replace(
        '<div id="swagger-ui">', player + '<div id="swagger-ui">', 1
    )
    return HTMLResponse(html)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    statuses = {
        lang: svc_registry.get(lang).ready for lang in svc_registry.all_languages()
    }
    if not all(statuses.values()):
        raise HTTPException(status_code=503, detail=statuses)
    return statuses


@app.post("/tts/{lang}", response_model=TTSResponse)
async def tts(lang: str, payload: TTSRequest) -> TTSResponse | Response:
    try:
        service = svc_registry.get(lang)
    except KeyError as kerr:
        raise HTTPException(
            status_code=404,
            detail=f"No TTS service for language: '{lang}'. "
            f"Available: {svc_registry.all_languages()}",
        ) from kerr

    if not service.ready:
        raise HTTPException(
            status_code=503, detail=f"TTS service for '{lang}' is not ready"
        )

    if payload.format == "pcm":
        return await _pcm_response(lang, service, payload)

    try:
        wav_bytes = await asyncio.to_thread(
            service.generate, payload.text, payload.instructions
        )
    except Exception as exc:
        logger.exception("TTS failed for language: %s", lang)
        raise HTTPException(status_code=500, detail="TTS processing failed") from exc

    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Input produced no audio")

    return TTSResponse(audio_base64=base64.b64encode(wav_bytes).decode("ascii"))


async def _pcm_response(
    lang: str, service: BaseTTSService, payload: TTSRequest
) -> Response:
    """Raw PCM, streamed or whole.

    The first chunk is synthesized before responding, so a failure up to then
    still gets an error status.
    """
    chunks = service.stream(payload.text, payload.instructions)
    first = await _next_chunk(lang, chunks)
    if first is None:
        raise HTTPException(status_code=400, detail="Input produced no audio")
    rate, data = first
    headers = {
        "X-Sample-Rate": str(rate),
        "X-Sample-Format": "s16le",
        "X-Channels": "1",
    }

    if payload.stream:
        return StreamingResponse(
            _pcm_body(lang, data, chunks),
            media_type="application/octet-stream",
            headers=headers,
        )
    pcm = bytearray(data)
    while (chunk := await _next_chunk(lang, chunks)) is not None:
        pcm += chunk[1]
    return Response(bytes(pcm), media_type="application/octet-stream", headers=headers)


async def _next_chunk(
    lang: str, chunks: Iterator[tuple[int, bytes]]
) -> tuple[int, bytes] | None:
    try:
        return await asyncio.to_thread(next, chunks, None)
    except Exception as exc:
        logger.exception("TTS failed for language: %s", lang)
        raise HTTPException(status_code=500, detail="TTS processing failed") from exc


async def _pcm_body(
    lang: str, first: bytes, chunks: Iterator[tuple[int, bytes]]
) -> AsyncIterator[bytes]:
    try:
        yield first
        while (chunk := await asyncio.to_thread(next, chunks, None)) is not None:
            yield chunk[1]
    except Exception:
        # The status is already sent; all we can do is end the audio early
        logger.exception("TTS stream failed for language: %s", lang)
    finally:
        # Frees the provider's upstream connection if the client hung up.
        # ValueError: a worker thread is still inside it; it closes on its own.
        with contextlib.suppress(ValueError):
            chunks.close()
