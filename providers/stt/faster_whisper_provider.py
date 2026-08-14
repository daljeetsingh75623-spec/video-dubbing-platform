from __future__ import annotations

import asyncio

from faster_whisper import WhisperModel

from core.domain import DiarizationResult, Transcript, TranscriptSegment
from providers.stt.base import STTProvider
from storage.factory import get_storage_backend


class FasterWhisperProvider(STTProvider):
    """
    Local Whisper inference — no API key, no per-request network call.
    Trade-off vs. an API-based provider: slower on CPU-only machines, but
    zero marginal cost and no rate limits, which matters for a take-home
    demo you'll be running repeatedly.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        # Loaded once per worker process, not per request — model load is
        # the expensive part (~seconds), inference itself is fast after that.
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    async def transcribe(
        self,
        audio_path: str,
        diarization: DiarizationResult | None = None,
        language_hint: str | None = None,
    ) -> Transcript:
        storage = get_storage_backend()
        local_path = f"/tmp/{audio_path.replace('/', '_')}"
        await storage.download(audio_path, local_path)

        segments_iter, info = await asyncio.to_thread(
            self._model.transcribe, local_path, language=language_hint, word_timestamps=False,
        )

        segments: list[TranscriptSegment] = []
        for seg in segments_iter:
            speaker_id = self._attribute_speaker(seg.start, seg.end, diarization)
            segments.append(TranscriptSegment(
                speaker_id=speaker_id,
                start_ms=int(seg.start * 1000),
                end_ms=int(seg.end * 1000),
                text=seg.text.strip(),
                language=info.language,
                confidence=seg.avg_logprob,
            ))

        return Transcript(segments=segments, source_language=info.language)

    @staticmethod
    def _attribute_speaker(start_s: float, end_s: float, diarization: DiarizationResult | None) -> str | None:
        """Assign the diarized speaker whose segment has the most time overlap."""
        if not diarization or not diarization.segments:
            return None
        start_ms, end_ms = start_s * 1000, end_s * 1000
        best, best_overlap = None, 0
        for d in diarization.segments:
            overlap = min(end_ms, d.end_ms) - max(start_ms, d.start_ms)
            if overlap > best_overlap:
                best, best_overlap = d.speaker_id, overlap
        return best