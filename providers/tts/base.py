from __future__ import annotations

from abc import ABC, abstractmethod

from core.domain import SynthesisResult, TranslatedTranscript


class TTSProvider(ABC):
    """
    Synthesizes speech audio for each translated segment, preserving speaker
    identity/voice-consistency across segments belonging to the same
    speaker_id (via voice cloning/reference audio where the provider
    supports it, or a stable voice-id mapping otherwise).
    """

    @abstractmethod
    async def synthesize(
        self,
        translated: TranslatedTranscript,
        speaker_reference_audio: dict[str, str] | None = None,
    ) -> SynthesisResult:
        """
        Args:
            translated: translated transcript segments to voice.
            speaker_reference_audio: optional map of speaker_id -> path/key
                of a short reference audio clip of that speaker's original
                voice, used for voice cloning where supported.

        Returns:
            SynthesisResult with one synthesized audio chunk per segment.
        """
        raise NotImplementedError