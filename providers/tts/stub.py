from __future__ import annotations

import asyncio

from core.domain import SynthesisResult, SynthesizedSegment, TranslatedTranscript
from providers.tts.base import TTSProvider


class StubTTSProvider(TTSProvider):
    """
    Doesn't synthesize real audio — records a fake output path per segment so
    downstream AV-sync/packaging code can be developed and tested without a
    real TTS engine.
    """

    async def synthesize(
        self,
        translated: TranslatedTranscript,
        speaker_reference_audio: dict[str, str] | None = None,
    ) -> SynthesisResult:
        await asyncio.sleep(0)
        segments = [
            SynthesizedSegment(
                speaker_id=s.speaker_id,
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                audio_path=f"stub-audio/{s.speaker_id}_{s.start_ms}_{s.end_ms}.wav",
                target_language=s.target_language,
            )
            for s in translated.segments
        ]
        return SynthesisResult(segments=segments)