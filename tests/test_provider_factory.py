"""ProviderFactory: the switching mechanism behind config-driven model
selection. Proves that swapping a provider = changing a config string, never
application code."""
from __future__ import annotations

import pytest

from providers.diarization.pyannote_provider import PyannoteDiarizationProvider
from providers.diarization.stub import StubDiarizationProvider
from providers.factory import (
    ProviderFactory,
    _DIARIZATION_REGISTRY,
    _STT_REGISTRY,
    _TRANSLATION_REGISTRY,
    _TTS_REGISTRY,
)
from providers.stt.base import STTProvider
from providers.stt.faster_whisper_provider import FasterWhisperProvider
from providers.stt.stub import StubSTTProvider
from providers.translation.base import TranslationProvider
from providers.translation.gemini_provider import GeminiTranslationProvider
from providers.translation.stub import StubTranslationProvider
from providers.tts.base import TTSProvider
from providers.tts.edge_tts_provider import EdgeTTSProvider
from providers.tts.stub import StubTTSProvider

_gtts = pytest.importorskip("gtts", reason="gtts is not installed")
from providers.tts.gtts_provider import GTTSProvider  # noqa: E402


def test_registry_maps_every_name_to_its_class():
    """Switching is a lookup: every registered name resolves (without
    instantiating — heavy providers load models in __init__) to the exact
    implementation class."""
    assert _STT_REGISTRY["stub"]() is StubSTTProvider
    assert _STT_REGISTRY["faster_whisper"]() is FasterWhisperProvider

    assert _TRANSLATION_REGISTRY["stub"]() is StubTranslationProvider
    assert _TRANSLATION_REGISTRY["gemini"]() is GeminiTranslationProvider

    assert _TTS_REGISTRY["stub"]() is StubTTSProvider
    assert _TTS_REGISTRY["edge_tts"]() is EdgeTTSProvider
    assert _TTS_REGISTRY["gtts"]() is GTTSProvider

    assert _DIARIZATION_REGISTRY["stub"]() is StubDiarizationProvider
    assert _DIARIZATION_REGISTRY["pyannote"]() is PyannoteDiarizationProvider


def test_factory_instantiates_lightweight_providers():
    stt = ProviderFactory.get_stt("stub")
    assert isinstance(stt, STTProvider)
    assert isinstance(stt, StubSTTProvider)

    translation = ProviderFactory.get_translation("stub")
    assert isinstance(translation, TranslationProvider)
    assert isinstance(translation, StubTranslationProvider)

    assert isinstance(ProviderFactory.get_tts("edge_tts"), EdgeTTSProvider)
    assert isinstance(ProviderFactory.get_tts("gtts"), GTTSProvider)
    assert isinstance(ProviderFactory.get_tts("stub"), TTSProvider)
    assert isinstance(ProviderFactory.get_diarization("stub"), StubDiarizationProvider)


def test_unknown_provider_raises_with_available_list():
    for stage_call, available in (
        (lambda: ProviderFactory.get_stt("deepgram"), "faster_whisper, stub"),
        (lambda: ProviderFactory.get_translation("claude"), "gemini, stub"),
        (lambda: ProviderFactory.get_tts("elevenlabs"), "edge_tts, gtts, stub"),
        (lambda: ProviderFactory.get_diarization("azure"), "pyannote, stub"),
    ):
        with pytest.raises(ValueError, match="Unknown .* provider"):
            stage_call()
        with pytest.raises(ValueError, match=available):
            stage_call()


def test_unimportable_provider_raises_friendly_error(monkeypatch):
    def _broken_loader():
        raise ImportError("No module named 'missing_dep'")

    monkeypatch.setitem(_STT_REGISTRY, "broken", _broken_loader)
    with pytest.raises(ValueError, match="dependency installed"):
        ProviderFactory.get_stt("broken")


def test_config_drives_provider_selection_without_code_changes(monkeypatch, tmp_path):
    """This mirrors exactly what workers/tasks.py does: read the provider name
    off config/job, hand it to the factory. Changing the setting below swaps
    the model — nothing in the pipeline code changes."""
    from config.settings import Settings

    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLOUD_CONFIG_URL", raising=False)
    monkeypatch.delenv("SECRETS_DIR", raising=False)

    s = Settings(
        stt_provider="faster_whisper",
        translation_provider="gemini",
        tts_provider="edge_tts",
        diarization_provider="pyannote",
    )

    assert _STT_REGISTRY[s.stt_provider]() is FasterWhisperProvider
    assert _TRANSLATION_REGISTRY[s.translation_provider]() is GeminiTranslationProvider
    assert _TTS_REGISTRY[s.tts_provider]() is EdgeTTSProvider
    assert _DIARIZATION_REGISTRY[s.diarization_provider]() is PyannoteDiarizationProvider

    # Flipping the same fields to another provider is a pure config change.
    s2 = Settings(
        stt_provider="stub",
        translation_provider="stub",
        tts_provider="gtts",
        diarization_provider="stub",
    )
    assert _STT_REGISTRY[s2.stt_provider]() is StubSTTProvider
    assert _TRANSLATION_REGISTRY[s2.translation_provider]() is StubTranslationProvider
    assert _TTS_REGISTRY[s2.tts_provider]() is GTTSProvider
    assert _DIARIZATION_REGISTRY[s2.diarization_provider]() is StubDiarizationProvider
