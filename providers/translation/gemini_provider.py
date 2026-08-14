from __future__ import annotations

import json

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from config.settings import get_settings
from core.domain import Transcript, TranslatedSegment, TranslatedTranscript
from providers.errors import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from providers.translation.base import TranslationProvider

_SYSTEM_PROMPT = (
    "You are a professional subtitle/dubbing translator. Translate each "
    "numbered line into {target_language}, preserving tone, register, and "
    "conversational flow across lines from the same speaker. Return ONLY a "
    'JSON object: {{"translations": ["...", "..."]}} with one string per '
    "input line, in the same order. Do not merge, split, or omit lines."
)


class GeminiTranslationProvider(TranslationProvider):
    def __init__(self, model: str = "gemini-3.5-flash"):
        settings = get_settings()
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = model
        self._timeout_ms = settings.provider_timeout_seconds * 1000

    async def translate(self, transcript: Transcript, target_language: str) -> TranslatedTranscript:
        lines = [s.text for s in transcript.segments]
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(lines))
        prompt = _SYSTEM_PROMPT.format(target_language=target_language) + "\n\n" + numbered

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    http_options=types.HttpOptions(timeout=self._timeout_ms),
                ),
            )
        except ClientError as e:
            # google-genai raises ClientError for 4xx — inspect status to
            # distinguish auth (401/403) from rate limit (429).
            status = getattr(e, "code", None)
            if status in (401, 403):
                raise ProviderAuthError(str(e)) from e
            if status == 429:
                raise ProviderRateLimitError(str(e)) from e
            raise
        except ServerError as e:
            raise ProviderUnavailableError(str(e)) from e
        except TimeoutError as e:
            raise ProviderTimeoutError(str(e)) from e
        except APIError as e:
            raise ProviderUnavailableError(str(e)) from e

        if response.text is None:
            raise ValueError("Gemini returned an empty response")

        parsed = json.loads(response.text)
        translations = parsed.get("translations") if isinstance(parsed, dict) else parsed
        if not isinstance(translations, list):
            raise ValueError("Translation response must be a JSON list of strings")

        if len(translations) != len(transcript.segments):
            raise ValueError(
                f"Translation count mismatch: got {len(translations)}, "
                f"expected {len(transcript.segments)}"
            )

        segments = [
            TranslatedSegment(
                speaker_id=s.speaker_id, start_ms=s.start_ms, end_ms=s.end_ms,
                source_text=s.text, translated_text=t, target_language=target_language,
            )
            for s, t in zip(transcript.segments, translations)
        ]
        return TranslatedTranscript(segments=segments, target_language=target_language)