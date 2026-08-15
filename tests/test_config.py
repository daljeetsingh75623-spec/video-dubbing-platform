from __future__ import annotations

from config.settings import Settings


def test_defaults_load_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
    monkeypatch.chdir(tmp_path)  # no .env file present here either
    s = Settings()
    assert s.stt_provider == "stub"
    assert s.max_upload_size_mb == 500
    assert "mp4" in s.allowed_video_formats


def test_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STT_PROVIDER", "faster_whisper")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "999")
    s = Settings()
    assert s.stt_provider == "faster_whisper"
    assert s.max_upload_size_mb == 999


def test_provider_fields_default_to_stub(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
    monkeypatch.chdir(tmp_path)
    for var in ("STT_PROVIDER", "TRANSLATION_PROVIDER", "TTS_PROVIDER", "DIARIZATION_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.stt_provider == "stub"
    assert s.translation_provider == "stub"
    assert s.tts_provider == "stub"
    assert s.diarization_provider == "stub"