from __future__ import annotations

import asyncio

from core.domain import DiarizationResult, Transcript, TranscriptSegment
from providers.stt.base import STTProvider


class StubSTTProvider(STTProvider):
    """Emits one placeholder text segment per diarized speaker turn."""

    async def transcribe(
        self,
        audio_path: str,
        diarization: DiarizationResult | None = None,
        language_hint: str | None = None,
    ) -> Transcript:
        await asyncio.sleep(0)
        segments: list[TranscriptSegment] = []
        if diarization and diarization.segments:
            for seg in diarization.segments:
                segments.append(
                    TranscriptSegment(
                        speaker_id=seg.speaker_id,
                        start_ms=seg.start_ms,
                        end_ms=seg.end_ms,
                        text=f"[stub transcript for {seg.speaker_id} "
                             f"{seg.start_ms}-{seg.end_ms}ms]",
                        language=language_hint or "en",
                        confidence=0.99,
                    )
                )
        else:
            segments.append(
                TranscriptSegment(
                    speaker_id=None,
                    start_ms=0,
                    end_ms=5_000,
                    text="[stub transcript, no diarization provided]",
                    language=language_hint or "en",
                    confidence=0.99,
                )
            )
        return Transcript(segments=segments, source_language=language_hint or "en")