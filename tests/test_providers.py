from __future__ import annotations

import pytest

from providers.factory import ProviderFactory


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