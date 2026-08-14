from __future__ import annotations

from abc import ABC, abstractmethod

from core.domain import DiarizationResult


class DiarizationProvider(ABC):
    """
    Identifies speakers and their time boundaries in an audio track.

    Contract: implementations must assign a STABLE speaker_id for the same
    physical speaker across the entire audio, even across non-contiguous
    segments (i.e. speaker identity must not drift or reset mid-file).
    """

    @abstractmethod
    async def diarize(self, audio_path: str) -> DiarizationResult:
        """
        Args:
            audio_path: path/key (in configured storage) to a mono/stereo
                audio file extracted from the source video.

        Returns:
            DiarizationResult with speaker-labeled time segments.

        Raises:
            ProviderTimeoutError, ProviderError (see providers/errors.py)
        """
        raise NotImplementedError