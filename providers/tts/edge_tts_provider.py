from __future__ import annotations

import tempfile
import uuid

import edge_tts
import structlog

from core.domain import SynthesisResult, SynthesizedSegment, TranslatedTranscript
from providers.tts.base import TTSProvider
from storage.factory import get_storage_backend

log = structlog.get_logger()

# Distinct Edge neural voices per language. Each speaker (by speaker index)
# is assigned a voice from this pool, so every speaker sounds different up to
# the pool size before voices cycle. Full list: `edge-tts --list-voices`.
_VOICE_POOL: dict[str, list[str]] = {
    "es": [
        "es-ES-AlvaroNeural",
        "es-ES-ElviraNeural",
        "es-MX-JorgeNeural",
        "es-MX-DaliaNeural",
        "es-AR-TomasNeural",
        "es-AR-ElenaNeural",
    ],
    "fr": [
        "fr-FR-HenriNeural",
        "fr-FR-DeniseNeural",
        "fr-CA-AntoineNeural",
        "fr-CA-JeanNeural",
        "fr-CA-SylvieNeural",
        "fr-FR-VivienneNeural",
    ],
    "hi": [
        "hi-IN-MadhurNeural",
        "hi-IN-SwaraNeural",
        "hi-IN-NeerjaNeural",
    ],
    "de": [
        "de-DE-ConradNeural",
        "de-DE-KatjaNeural",
        "de-DE-FlorianNeural",
        "de-DE-AmelieNeural",
        "de-AT-JonasNeural",
        "de-CH-JanNeural",
    ],
    "en": [
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-US-ChristopherNeural",
        "en-US-MichelleNeural",
        "en-GB-RyanNeural",
        "en-GB-SoniaNeural",
        "en-AU-WilliamNeural",
        "en-AU-NatashaNeural",
    ],
    "ja": [
        "ja-JP-KeitaNeural",
        "ja-JP-NanamiNeural",
        "ja-JP-ShioriNeural",
    ],
    "pt": [
        "pt-BR-AntonioNeural",
        "pt-BR-FranciscaNeural",
        "pt-PT-FernandaNeural",
        "pt-PT-DuarteNeural",
    ],
}


class EdgeTTSProvider(TTSProvider):
    """
    Uses Microsoft Edge's free neural TTS voices via the unofficial
    `edge-tts` package — no API key, no cost. Each speaker is mapped to a
    distinct voice from the language's pool (stable per speaker_id), which
    preserves speaker identity across segments. Real voice cloning (with
    reference audio) would need ElevenLabs/Coqui.
    """

    def __init__(self):
        self._storage = get_storage_backend()

    def _voices_for(self, target_language: str) -> list[str]:
        return _VOICE_POOL.get(target_language) or _VOICE_POOL["en"]

    def _voice_for(self, speaker_id: str | None, target_language: str) -> str:
        voices = self._voices_for(target_language)
        if speaker_id is None:
            return voices[0]
        # stable per-speaker index from labels like "SPEAKER_00"
        idx = int(speaker_id.split("_")[-1]) if "_" in speaker_id and speaker_id.split("_")[-1].isdigit() else 0
        return voices[idx % len(voices)]

    async def _synthesize_chunk(self, text: str, voice: str, fallback_voice: str, tmp_path: str) -> str:
        """Synthesize one chunk; returns the voice that was actually used so
        the segment's voice_id is always accurate (not the assigned voice)."""
        try:
            await edge_tts.Communicate(text, voice).save(tmp_path)
            return voice
        except Exception:
            # The assigned voice may have been removed upstream — fall back to
            # the language's default rather than failing the whole job.
            if voice == fallback_voice:
                raise
            await edge_tts.Communicate(text, fallback_voice).save(tmp_path)
            return fallback_voice

    async def synthesize(
        self,
        translated: TranslatedTranscript,
        speaker_reference_audio: dict[str, str] | None = None,
    ) -> SynthesisResult:
        segments = []
        for s in translated.segments:
            voices = self._voices_for(s.target_language)
            voice = self._voice_for(s.speaker_id, s.target_language)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            voice = await self._synthesize_chunk(s.translated_text, voice, voices[0], tmp_path)

            key = f"synthesized/{uuid.uuid4()}.mp3"
            await self._storage.upload(tmp_path, key)

            log.info(
                "tts_segment",
                speaker_id=s.speaker_id,
                voice=voice,
                target_language=s.target_language,
                start_ms=s.start_ms,
            )
            segments.append(SynthesizedSegment(
                speaker_id=s.speaker_id, start_ms=s.start_ms, end_ms=s.end_ms,
                audio_path=key, target_language=s.target_language, voice_id=voice,
            ))

        return SynthesisResult(segments=segments)
