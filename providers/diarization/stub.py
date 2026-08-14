from __future__ import annotations

import asyncio

from core.domain import DiarizationResult, SpeakerSegment
from providers.diarization.base import DiarizationProvider


class StubDiarizationProvider(DiarizationProvider):
    """
    Deterministic fake: splits the (assumed) audio into two alternating
    3-second speaker turns for a fixed fake duration. Good enough to exercise
    every downstream stage without a real model or GPU.
    """

    def __init__(self, fake_duration_ms: int = 30_000, turn_ms: int = 3_000):
        self.fake_duration_ms = fake_duration_ms
        self.turn_ms = turn_ms

    async def diarize(self, audio_path: str) -> DiarizationResult:
        await asyncio.sleep(0)
        segments: list[SpeakerSegment] = []
        t = 0
        i = 0
        while t < self.fake_duration_ms:
            speaker = f"SPEAKER_{i % 2:02d}"
            end = min(t + self.turn_ms, self.fake_duration_ms)
            segments.append(SpeakerSegment(speaker_id=speaker, start_ms=t, end_ms=end))
            t = end
            i += 1
        return DiarizationResult(segments=segments, num_speakers=2)