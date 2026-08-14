"""
ProviderFactory: the single place that knows how to turn a config string
(e.g. "whisper", "elevenlabs", "stub") into a live provider instance.

Adding a new provider = write the class + register it in the relevant
registry dict below. Nothing else in the codebase needs to change — routers,
Celery tasks, and tests all depend on the *base class* interfaces, never on
concrete provider classes directly.
"""
from __future__ import annotations

from providers.diarization.base import DiarizationProvider
from providers.diarization.stub import StubDiarizationProvider
from providers.stt.base import STTProvider
from providers.stt.stub import StubSTTProvider
from providers.translation.base import TranslationProvider
from providers.translation.stub import StubTranslationProvider
from providers.tts.base import TTSProvider
from providers.tts.stub import StubTTSProvider

_STT_REGISTRY: dict[str, type[STTProvider]] = {
    "stub": StubSTTProvider,
    # "faster_whisper": FasterWhisperProvider,
    # "openai_whisper_api": OpenAIWhisperAPIProvider,
}

_TRANSLATION_REGISTRY: dict[str, type[TranslationProvider]] = {
    "stub": StubTranslationProvider,
    # "openai": OpenAITranslationProvider,
    # "claude": ClaudeTranslationProvider,
    # "gemini": GeminiTranslationProvider,
}

_TTS_REGISTRY: dict[str, type[TTSProvider]] = {
    "stub": StubTTSProvider,
    # "elevenlabs": ElevenLabsTTSProvider,
    # "azure_speech": AzureSpeechTTSProvider,
}

_DIARIZATION_REGISTRY: dict[str, type[DiarizationProvider]] = {
    "stub": StubDiarizationProvider,
    # "pyannote": PyannoteDiarizationProvider,
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
    def _resolve(registry: dict, name: str, stage: str):
        try:
            return registry[name]
        except KeyError as e:
            available = ", ".join(sorted(registry)) or "(none registered)"
            raise ValueError(
                f"Unknown {stage} provider '{name}'. Available: {available}"
            ) from e