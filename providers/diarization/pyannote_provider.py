from __future__ import annotations

import asyncio
from typing import Any

from pyannote.audio import Pipeline

from config.settings import get_settings
from core.domain import DiarizationResult, SpeakerSegment
from providers.diarization.base import DiarizationProvider
from storage.factory import get_storage_backend


class PyannoteDiarizationProvider(DiarizationProvider):
    """
    Local diarization via pyannote.audio. Loaded once per worker process —
    the pipeline download/init is the expensive part, inference is fast
    after that. CPU works but is slow; set device="cuda" if a GPU is
    available in the worker container.
    """

    def __init__(self, device: str = "cpu"):
        settings = get_settings()
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=settings.huggingface_token,
        )
        if pipeline is None:
            raise RuntimeError("Failed to load pyannote speaker-diarization pipeline")
        self._pipeline = pipeline
        if device == "cuda":
            import torch
            self._pipeline.to(torch.device("cuda"))

    async def diarize(self, audio_path: str) -> DiarizationResult:
        storage = get_storage_backend()
        local_path = f"/tmp/{audio_path.replace('/', '_')}"
        await storage.download(audio_path, local_path)

        annotation: Any = await asyncio.to_thread(self._pipeline, local_path)

        # pyannote assigns labels like "SPEAKER_00" already, but per-file —
        # remap through a stable, deterministic ordering (first-appearance)
        # so speaker_id stays consistent with the rest of the domain model.
        label_map: dict[str, str] = {}
        segments: list[SpeakerSegment] = []
        for turn, _, label in annotation.itertracks(yield_label=True):
            if label not in label_map:
                label_map[label] = f"SPEAKER_{len(label_map):02d}"
            segments.append(SpeakerSegment(
                speaker_id=label_map[label],
                start_ms=int(turn.start * 1000),
                end_ms=int(turn.end * 1000),
            ))

        return DiarizationResult(segments=segments, num_speakers=len(label_map))