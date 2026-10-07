"""
Service registry wiring.

TTS_PROVIDER selects the backend: "local" (OmniVoice on the GPU) or "gemini".

To add a new language:
  1. Create app/services/<lang>_tts_service.py with a subclass of BaseTTSService.
  2. Import it here and call registry.register("<lang_code>", <YourService>()).
"""

from app.config import settings
from app.services import registry

if settings.tts_provider == "gemini":
    from app.services.gemini_tts_service import GeminiTTSService

    registry.register("am", GeminiTTSService())
else:
    from app.services.amharic_tts_service import AmharicTTSService

    registry.register("am", AmharicTTSService())
