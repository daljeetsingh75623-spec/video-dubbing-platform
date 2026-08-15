from __future__ import annotations

import json

import pytest

from providers.factory import ProviderFactory
from providers.translation.gemini_provider import _extract_json
from providers.tts.edge_tts_provider import EdgeTTSProvider


@pytest.mark.asyncio
async def test_stub_diarization_produces_alternating_speakers():
    provider = ProviderFactory.get_diarization("stub")
    result = await provider.diarize("fake/audio.wav")
    assert result.num_speakers == 2
    assert len(result.segments) > 0
    assert result.segments[0].speaker_id == "SPEAKER_00"
    assert result.segments[1].speaker_id == "SPEAKER_01"


@pytest.mark.asyncio
async def test_stub_pipeline_end_to_end():
    """Diarize -> transcribe -> translate -> synthesize, all with stubs."""
    diarization_provider = ProviderFactory.get_diarization("stub")
    stt_provider = ProviderFactory.get_stt("stub")
    translation_provider = ProviderFactory.get_translation("stub")
    tts_provider = ProviderFactory.get_tts("stub")

    diarization = await diarization_provider.diarize("fake/audio.wav")
    transcript = await stt_provider.transcribe("fake/audio.wav", diarization=diarization)
    assert len(transcript.segments) == len(diarization.segments)

    translated = await translation_provider.translate(transcript, target_language="es")
    assert translated.target_language == "es"
    assert all(seg.translated_text.startswith("[es]") for seg in translated.segments)

    synthesis = await tts_provider.synthesize(translated)
    assert len(synthesis.segments) == len(translated.segments)
    assert all(seg.audio_path.endswith(".wav") for seg in synthesis.segments)


def test_factory_raises_on_unknown_provider():
    with pytest.raises(ValueError, match="Unknown STT provider"):
        ProviderFactory.get_stt("not_a_real_provider")


def test_factory_lists_available_providers_in_error():
    with pytest.raises(ValueError, match="stub"):
        ProviderFactory.get_translation("nonexistent")


def test_extract_json_parses_plain_response():
    raw = '{"translations": ["hola", "mundo"]}'
    assert json.loads(_extract_json(raw)) == {"translations": ["hola", "mundo"]}


def test_extract_json_strips_markdown_fence():
    raw = '```json\n{"translations": ["hola", "mundo"]}\n```'
    assert json.loads(_extract_json(raw)) == {"translations": ["hola", "mundo"]}


def test_extract_json_strips_prose_prefix_and_suffix():
    raw = 'Here is the result:\n{"translations": ["hola", "mundo"]}\nHope that helps!'
    assert json.loads(_extract_json(raw)) == {"translations": ["hola", "mundo"]}


def test_extract_json_ignores_trailing_prose_with_braces():
    # Naive first-{/last-} slicing would span into the trailing prose and fail
    # with "Expecting ',' delimiter"; raw_decode must stop at the first value.
    raw = '{"translations": ["hola", "mundo"]} Hope this helps {thanks}'
    assert json.loads(_extract_json(raw)) == {"translations": ["hola", "mundo"]}


def test_extract_json_supports_top_level_array():
    raw = '```json\n["hola", "mundo"]\n```'
    assert json.loads(_extract_json(raw)) == ["hola", "mundo"]


def test_extract_json_raises_when_no_json_present():
    with pytest.raises(ValueError, match="No JSON"):
        _extract_json("I could not translate these lines.")


def test_edge_voice_is_distinct_per_speaker():
    provider = EdgeTTSProvider()
    voices = {provider._voice_for(f"SPEAKER_{i:02d}", "en") for i in range(4)}
    assert len(voices) == 4  # four different speakers -> four different voices


def test_edge_voice_is_stable_per_speaker():
    provider = EdgeTTSProvider()
    assert provider._voice_for("SPEAKER_02", "es") == provider._voice_for("SPEAKER_02", "es")
    assert provider._voice_for("SPEAKER_00", "en") != provider._voice_for("SPEAKER_01", "en")


def test_edge_voice_falls_back_for_unknown_language():
    provider = EdgeTTSProvider()
    assert provider._voice_for("SPEAKER_00", "xx") == provider._voice_for("SPEAKER_00", "en")


def test_edge_none_speaker_uses_default_voice():
    provider = EdgeTTSProvider()
    assert provider._voice_for(None, "en") == "en-US-GuyNeural"


@pytest.mark.asyncio
async def test_edge_synthesize_maps_distinct_voices_per_speaker(monkeypatch):
    from core.domain import TranslatedSegment, TranslatedTranscript

    class _FakeStorage:
        async def upload(self, local_path, key):
            return key

    provider = object.__new__(EdgeTTSProvider)
    provider._storage = _FakeStorage()

    async def _fake_synth(text, voice, fallback_voice, tmp_path):
        return voice

    monkeypatch.setattr(provider, "_synthesize_chunk", _fake_synth)

    transcript = TranslatedTranscript(
        target_language="en",
        segments=[
            TranslatedSegment("SPEAKER_00", 0, 1000, "hi", "hola", "es"),
            TranslatedSegment("SPEAKER_01", 1000, 2000, "bye", "adios", "es"),
            TranslatedSegment("SPEAKER_00", 2000, 3000, "again", "otra vez", "es"),
        ],
    )
    result = await provider.synthesize(transcript)
    voices_by_speaker = {seg.speaker_id: seg.voice_id for seg in result.segments}
    assert voices_by_speaker["SPEAKER_00"] != voices_by_speaker["SPEAKER_01"]
    # stable within a speaker
    assert result.segments[0].voice_id == result.segments[2].voice_id