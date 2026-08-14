from __future__ import annotations

from abc import ABC, abstractmethod

from core.domain import DiarizationResult, Transcript


class STTProvider(ABC):
    """
    Speech-to-text: produces a timestamped, optionally speaker-attributed
    transcript from an audio file.
    """

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
        diarization: DiarizationResult | None = None,
        language_hint: str | None = None,
    ) -> Transcript:
        """
        Args:
            audio_path: path/key (in configured storage) to the audio file.
            diarization: optional prior diarization result, used to attribute
                transcript segments to speaker_ids where the STT engine
                itself doesn't do speaker attribution.
            language_hint: optional ISO 639-1 code; if omitted, implementations
                must auto-detect language.

        Returns:
            Transcript with timestamped segments.
        """
        raise NotImplementedError