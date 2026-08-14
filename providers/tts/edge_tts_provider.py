from __future__ import annotations

import tempfile
import uuid

import edge_tts

from core.domain import SynthesisResult, SynthesizedSegment, TranslatedTranscript
from providers.tts.base import TTSProvider
from storage.factory import get_storage_backend

# Maps a target language code to a default Edge neural voice. Extend as
# needed — full voice list: `edge-tts --list-voices`.
_VOICE_MAP: dict[str, str] = {
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "hi": "hi-IN-MadhurNeural",
    "de": "de-DE-ConradNeural",
    "en": "en-US-GuyNeural",
    "ja": "ja-JP-KeitaNeural",
    "pt": "pt-BR-AntonioNeural",
}

# Alternate voice per speaker_id parity, so at least 2 speakers sound
# distinct even without real voice cloning — cheap approximation of
# "preserve speaker identity" for a free-tier TTS engine.
_VOICE_MAP_ALT: dict[str, str] = {
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "hi": "hi-IN-SwaraNeural",
    "de": "de-DE-KatjaNeural",
    "en": "en-US-JennyNeural",
    "ja": "ja-JP-NanamiNeural",
    "pt": "pt-BR-FranciscaNeural",
}


class EdgeTTSProvider(TTSProvider):
    """
    Uses Microsoft Edge's free neural TTS voices via the unofficial
    `edge-tts` package — no API key, no cost. Voice consistency per speaker
    is approximated by alternating between two voices per language (real
    voice cloning would need ElevenLabs/Coqui with reference audio).
    """

    def __init__(self):
        self._storage = get_storage_backend()

    def _voice_for(self, speaker_id: str | None, target_language: str) -> str:
        lang = target_language if target_language in _VOICE_MAP else "en"
        if speaker_id is None:
            return _VOICE_MAP[lang]
        # stable alternation: even-indexed speaker labels get primary voice
        idx = int(speaker_id.split("_")[-1]) if "_" in speaker_id and speaker_id.split("_")[-1].isdigit() else 0
        return _VOICE_MAP[lang] if idx % 2 == 0 else _VOICE_MAP_ALT[lang]

    async def synthesize(
        self,
        translated: TranslatedTranscript,
        speaker_reference_audio: dict[str, str] | None = None,
    ) -> SynthesisResult:
        segments = []
        for s in translated.segments:
            voice = self._voice_for(s.speaker_id, s.target_language)
            communicate = edge_tts.Communicate(s.translated_text, voice)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            await communicate.save(tmp_path)

            key = f"synthesized/{uuid.uuid4()}.mp3"
            await self._storage.upload(tmp_path, key)

            segments.append(SynthesizedSegment(
                speaker_id=s.speaker_id, start_ms=s.start_ms, end_ms=s.end_ms,
                audio_path=key, target_language=s.target_language,
            ))

        return SynthesisResult(segments=segments)