from __future__ import annotations

import asyncio
import inspect
from typing import Any

from pyannote.audio import Pipeline

from config.settings import get_settings
from core.domain import DiarizationResult, SpeakerSegment
from providers.diarization.base import DiarizationProvider
from storage.factory import get_storage_backend


def _from_pretrained_kwargs(token: str | None) -> dict[str, Any]:
    """Pass the HF token through the kwarg name this pyannote version uses.

    pyannote.audio 3.x names it ``use_auth_token``; 4.x renamed it to
    ``token``. Sniff the signature so the same code runs on either.
    """
    if not token:
        return {}
    params = inspect.signature(Pipeline.from_pretrained).parameters
    key = "token" if "token" in params else "use_auth_token"
    return {key: token}


class PyannoteDiarizationProvider(DiarizationProvider):
    """
    Local diarization via pyannote.audio. Loaded once per worker process —
    the pipeline download/init is the expensive part, inference is fast
    after that. CPU works but is slow; set device="cuda" if a GPU is
    available in the worker container.
    """

    def __init__(self, device: str = "cpu"):
        settings = get_settings()
        self._min_speakers = settings.diarization_min_speakers
        self._max_speakers = settings.diarization_max_speakers
        if self._min_speakers and self._max_speakers and self._min_speakers > self._max_speakers:
            raise ValueError("diarization_min_speakers cannot exceed diarization_max_speakers")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            **_from_pretrained_kwargs(settings.huggingface_token),
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

        kwargs = {}
        if self._min_speakers is not None:
            kwargs["min_speakers"] = self._min_speakers
        if self._max_speakers is not None:
            kwargs["max_speakers"] = self._max_speakers

        # pyannote.audio 4.x returns a DiarizeOutput dataclass; the speaker
        # diarization Annotation is on the .speaker_diarization field.
        output: Any = await asyncio.to_thread(self._pipeline, local_path, **kwargs)
        annotation = output.speaker_diarization

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