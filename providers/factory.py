"""
ProviderFactory: the single place that knows how to turn a config string
(e.g. "whisper", "elevenlabs", "stub") into a live provider instance.

Adding a new provider = write the class + register it in the relevant
registry dict below. Nothing else in the codebase needs to change — routers,
Celery tasks, and tests all depend on the *base class* interfaces, never on
concrete provider classes directly.

Providers are imported lazily so a missing optional dependency (pyannote,
faster-whisper, ...) only breaks that provider when it is actually used,
never the app itself at import time.
"""
from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from providers.diarization.base import DiarizationProvider
from providers.stt.base import STTProvider
from providers.translation.base import TranslationProvider
from providers.tts.base import TTSProvider


def _lazy(module_name: str, attr: str) -> Callable[[], type]:
    def _load():
        return getattr(import_module(module_name), attr)

    return _load


_STT_REGISTRY: dict[str, Callable[[], type[STTProvider]]] = {
    "stub": _lazy("providers.stt.stub", "StubSTTProvider"),
    "faster_whisper": _lazy("providers.stt.faster_whisper_provider", "FasterWhisperProvider"),
}

_TRANSLATION_REGISTRY: dict[str, Callable[[], type[TranslationProvider]]] = {
    "stub": _lazy("providers.translation.stub", "StubTranslationProvider"),
    "gemini": _lazy("providers.translation.gemini_provider", "GeminiTranslationProvider"),
}

_TTS_REGISTRY: dict[str, Callable[[], type[TTSProvider]]] = {
    "stub": _lazy("providers.tts.stub", "StubTTSProvider"),
    "edge_tts": _lazy("providers.tts.edge_tts_provider", "EdgeTTSProvider"),
}

_DIARIZATION_REGISTRY: dict[str, Callable[[], type[DiarizationProvider]]] = {
    "stub": _lazy("providers.diarization.stub", "StubDiarizationProvider"),
    "pyannote": _lazy("providers.diarization.pyannote_provider", "PyannoteDiarizationProvider"),
}


class ProviderFactory:
    """Instantiates configured provider implementations by name."""

    @staticmethod
    def get_stt(name: str, **kwargs) -> STTProvider:
        return ProviderFactory._resolve(_STT_REGISTRY, name, "STT")(**kwargs)

    @staticmethod
    def get_translation(name: str, **kwargs) -> TranslationProvider:
        return ProviderFactory._resolve(_TRANSLATION_REGISTRY, name, "translation")(**kwargs)

    @staticmethod
    def get_tts(name: str, **kwargs) -> TTSProvider:
        return ProviderFactory._resolve(_TTS_REGISTRY, name, "TTS")(**kwargs)

    @staticmethod
    def get_diarization(name: str, **kwargs) -> DiarizationProvider:
        return ProviderFactory._resolve(_DIARIZATION_REGISTRY, name, "diarization")(**kwargs)

    @staticmethod
    def _resolve(registry: dict[str, Callable[[], type]], name: str, stage: str):
        try:
            loader = registry[name]
        except KeyError as e:
            available = ", ".join(sorted(registry)) or "(none registered)"
            raise ValueError(
                f"Unknown {stage} provider '{name}'. Available: {available}"
            ) from e
        try:
            return loader()
        except ImportError as e:
            raise ValueError(
                f"Provider '{name}' ({stage}) could not be imported — is its "
                f"dependency installed? {e}"
            ) from e
