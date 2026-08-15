from __future__ import annotations

import asyncio
import tempfile
import uuid

from gtts import gTTS

from core.domain import SynthesisResult, SynthesizedSegment, TranslatedTranscript
from providers.tts.base import TTSProvider
from storage.factory import get_storage_backend


class GTTSProvider(TTSProvider):
    """
    Uses Google Translate's free neural TTS voices via the `gTTS` package —
    no API key, no cost. Quality is lower than Edge/ElevenLabs but the
    endpoint is generally not geo-blocked, which makes it a reliable
    fallback for a demo.
    """

    def __init__(self):
        self._storage = get_storage_backend()

    async def synthesize(
        self,
        translated: TranslatedTranscript,
        speaker_reference_audio: dict[str, str] | None = None,
    ) -> SynthesisResult:
        segments = []
        for s in translated.segments:
            tts = gTTS(text=s.translated_text, lang=s.target_language, slow=False)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            await asyncio.to_thread(tts.save, tmp_path)

            key = f"synthesized/{uuid.uuid4()}.mp3"
            await self._storage.upload(tmp_path, key)

            segments.append(SynthesizedSegment(
                speaker_id=s.speaker_id, start_ms=s.start_ms, end_ms=s.end_ms,
                audio_path=key, target_language=s.target_language,
            ))

        return SynthesisResult(segments=segments)
