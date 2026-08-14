from __future__ import annotations

from abc import ABC, abstractmethod

from core.domain import Transcript, TranslatedTranscript


class TranslationProvider(ABC):
    """
    Translates a full transcript into a target language while preserving
    per-segment speaker mapping and ordering, so tone/context can be
    inferred from surrounding segments rather than translated line-by-line
    in isolation.
    """

    @abstractmethod
    async def translate(
        self,
        transcript: Transcript,
        target_language: str,
    ) -> TranslatedTranscript:
        """
        Args:
            transcript: source-language transcript with speaker attribution.
            target_language: ISO 639-1 code, e.g. "es", "hi", "fr".

        Returns:
            TranslatedTranscript with one TranslatedSegment per source segment,
            in the same order, so downstream stages (TTS, sync) can zip them
            against the original timing.
        """
        raise NotImplementedError