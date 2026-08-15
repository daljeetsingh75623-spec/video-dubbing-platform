from __future__ import annotations

import asyncio
import tempfile
import uuid
import wave

from core.domain import SynthesisResult, SynthesizedSegment, TranslatedTranscript
from providers.tts.base import TTSProvider
from storage.factory import get_storage_backend


class StubTTSProvider(TTSProvider):
    """
    Deterministic fake TTS for offline/dev runs: generates a real (silent)
    WAV chunk per segment, uploads it to storage under `synthesized/<uuid>.wav`
    and returns the storage key, exactly like a real provider would.

    This lets the whole pipeline (including AV sync and packaging) run
    end-to-end without a real TTS engine — the audio is real but empty, so
    dubbed output is silence. Swap in edge_tts / gtts / a cloning provider in
    config for actual speech.
    """

    SAMPLE_RATE = 22_050

    def __init__(self) -> None:
        self._storage = get_storage_backend()

    async def _write_silence_wav(self, tmp_path: str, duration_ms: int) -> None:
        """Stdlib-only silent mono 16-bit WAV so no ffmpeg dependency is needed."""
        n_frames = int(duration_ms / 1000 * self.SAMPLE_RATE)
        with wave.open(tmp_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.SAMPLE_RATE)
            w.writeframes(b"\x00\x00" * n_frames)

    async def synthesize(
        self,
        translated: TranslatedTranscript,
        speaker_reference_audio: dict[str, str] | None = None,
    ) -> SynthesisResult:
        segments: list[SynthesizedSegment] = []
        for s in translated.segments:
            duration_ms = max(int(s.end_ms - s.start_ms), 1)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            await self._write_silence_wav(tmp_path, duration_ms)
            key = f"synthesized/{uuid.uuid4()}.wav"
            await self._storage.upload(tmp_path, key)
            segments.append(SynthesizedSegment(
                speaker_id=s.speaker_id, start_ms=s.start_ms, end_ms=s.end_ms,
                audio_path=key, target_language=s.target_language,
            ))
        return SynthesisResult(segments=segments)
