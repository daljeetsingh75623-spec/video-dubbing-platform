from __future__ import annotations

import asyncio

from core.domain import Transcript, TranslatedSegment, TranslatedTranscript
from providers.translation.base import TranslationProvider


class StubTranslationProvider(TranslationProvider):
    """Passes text through with a bracketed tag instead of calling an LLM."""

    async def translate(
        self,
        transcript: Transcript,
        target_language: str,
    ) -> TranslatedTranscript:
        await asyncio.sleep(0)
        segments = [
            TranslatedSegment(
                speaker_id=s.speaker_id,
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                source_text=s.text,
                translated_text=f"[{target_language}] {s.text}",
                target_language=target_language,
            )
            for s in transcript.segments
        ]
        return TranslatedTranscript(segments=segments, target_language=target_language)