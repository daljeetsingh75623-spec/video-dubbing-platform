from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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


def test_secrets_dir_overrides_yaml(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "max_upload_size_mb").write_text("888")
    (secrets / "gemini_api_key").write_text("sekrit")

    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SECRETS_DIR", str(secrets))
    monkeypatch.delenv("CLOUD_CONFIG_URL", raising=False)

    s = Settings()
    assert s.max_upload_size_mb == 888
    assert s.gemini_api_key == "sekrit"


class _ConfigHandler(BaseHTTPRequestHandler):
    config_blob = ""

    def do_GET(self):
        body = self.config_blob.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/yaml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _ConfigServer:
    """Tiny HTTP server serving a fixed YAML blob — stands in for a cloud
    configuration service (AppConfig agent, Consul, Vault, ...)."""

    def __init__(self, yaml_blob: str):
        handler = type("Handler", (_ConfigHandler,), {"config_blob": yaml_blob})
        self._server = HTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/config"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()


def test_cloud_config_overrides_yaml(monkeypatch, tmp_path):
    with _ConfigServer("max_upload_size_mb: 777\nmax_concurrent_uploads: 42\n") as server:
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CLOUD_CONFIG_URL", server.url)
        monkeypatch.delenv("SECRETS_DIR", raising=False)

        s = Settings()
        assert s.max_upload_size_mb == 777
        assert s.max_concurrent_uploads == 42
        # A key absent from the cloud blob still comes from the YAML default.
        assert "mp4" in s.allowed_video_formats


def test_precedence_env_over_secrets_over_cloud(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "max_upload_size_mb").write_text("888")
    with _ConfigServer("max_upload_size_mb: 777\n") as server:
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SECRETS_DIR", str(secrets))
        monkeypatch.setenv("CLOUD_CONFIG_URL", server.url)

        s = Settings()
        assert s.max_upload_size_mb == 888  # secrets beats cloud

        monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "999")
        s2 = Settings()
        assert s2.max_upload_size_mb == 999  # env beats everything


def test_cloud_config_unavailable_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLOUD_CONFIG_URL", "http://127.0.0.1:1/config")  # connection refused
    monkeypatch.setenv("CLOUD_CONFIG_TIMEOUT_SECONDS", "0.5")
    monkeypatch.delenv("SECRETS_DIR", raising=False)

    s = Settings()
    assert s.max_upload_size_mb == 500  # defaults still apply
