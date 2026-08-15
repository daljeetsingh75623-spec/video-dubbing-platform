from __future__ import annotations

import json
import re

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from config.settings import get_settings
from core.domain import Transcript, TranslatedSegment, TranslatedTranscript
from providers.errors import (
    ProviderAuthError,
    ProviderError,
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
    "input line, in the same order. Do not merge, split, or omit lines. "
    "Do not wrap the JSON in markdown code fences or add any other text."
)


def _extract_json(text: str) -> str:
    """
    Models occasionally wrap the JSON in ```json ... ``` fences or append a
    sentence of prose after it. Strip any fence, then decode exactly the
    first complete JSON value via raw_decode (so a stray epilogue can't
    break the parse). If that value is valid but doesn't look like our
    payload, keep scanning for a better one.
    """
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1)

    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            start = text.find("[", search_from)
        if start == -1:
            raise ValueError("No JSON object/array found in model response")
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            search_from = start + 1
            continue
        if isinstance(value, (list, dict)):
            return text[start : start + end]
        search_from = start + 1


class GeminiTranslationProvider(TranslationProvider):
    def __init__(self, model: str = "gemini-3.5-flash"):
        settings = get_settings()
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = model
        self._timeout_ms = settings.provider_timeout_seconds * 1000
        self._batch_size = settings.translation_batch_size

    async def translate(self, transcript: Transcript, target_language: str) -> TranslatedTranscript:
        # Translate in batches so each response stays well under the model's
        # output-token limit — a single huge transcript can get truncated
        # mid-JSON, which no parser can recover from.
        translations: list[str] = []
        for start in range(0, len(transcript.segments), self._batch_size):
            batch = transcript.segments[start : start + self._batch_size]
            translations.extend(await self._translate_batch(batch, target_language))

        segments = [
            TranslatedSegment(
                speaker_id=s.speaker_id, start_ms=s.start_ms, end_ms=s.end_ms,
                source_text=s.text, translated_text=t, target_language=target_language,
            )
            for s, t in zip(transcript.segments, translations)
        ]
        return TranslatedTranscript(segments=segments, target_language=target_language)

    async def _translate_batch(
        self,
        batch: list[Transcript],
        target_language: str,
    ) -> list[str]:
        numbered = "\n".join(f"{i+1}. {s.text}" for i, s in enumerate(batch))
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
            raise ProviderError("Gemini returned an empty response")

        try:
            parsed = json.loads(_extract_json(response.text))
        except (json.JSONDecodeError, ValueError) as e:
            raise ProviderError(
                f"Gemini returned invalid JSON: {e}. Response: {response.text[:500]}"
            ) from e
        translations = parsed.get("translations") if isinstance(parsed, dict) else parsed
        if not isinstance(translations, list):
            raise ProviderError("Translation response must be a JSON list of strings")

        if len(translations) != len(batch):
            raise ProviderError(
                f"Translation count mismatch: got {len(translations)}, "
                f"expected {len(batch)}"
            )
        return translations