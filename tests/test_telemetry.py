"""init_otel must be safe in every configuration: disabled, unreachable
collector, or console-only. A broken collector must never take the API down."""
from __future__ import annotations

import fastapi
from fastapi.testclient import TestClient
from opentelemetry import trace

from core.telemetry import init_otel


def _shutdown_provider():
    """Tear down the process-global tracer provider so a test's provider (and
    its background export thread / fake exporter) can't leak into later tests."""
    provider = trace.get_tracer_provider()
    try:
        provider.shutdown()
    except Exception:
        pass


def _settings_with(monkeypatch, tmp_path, **overrides):
    from config.settings import Settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
    monkeypatch.delenv("CLOUD_CONFIG_URL", raising=False)
    monkeypatch.delenv("SECRETS_DIR", raising=False)
    return Settings(**overrides)


def _make_app():
    return fastapi.FastAPI()


def test_disabled_is_a_noop(monkeypatch, tmp_path):
    fake = _settings_with(monkeypatch, tmp_path, otel_enabled=False)

    called = []
    monkeypatch.setattr("core.telemetry.get_settings", lambda: fake)
    monkeypatch.setattr(
        "core.telemetry.FastAPIInstrumentor.instrument_app",
        lambda app, tracer_provider=None: called.append(1),
    )
    monkeypatch.setattr(
        "core.telemetry.OTLPSpanExporter",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not build exporter")),
    )

    init_otel(_make_app())
    assert called == []


def test_unreachable_endpoint_never_breaks_requests(monkeypatch, tmp_path):
    fake = _settings_with(
        monkeypatch, tmp_path, otel_enabled=True, otel_exporter_endpoint="http://127.0.0.1:9999"
    )
    monkeypatch.setattr("core.telemetry.get_settings", lambda: fake)

    app = _make_app()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    init_otel(app)

    try:
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
    finally:
        _shutdown_provider()


class _ConsoleExporter:
    """Fake ConsoleSpanExporter with the SDK exporter lifecycle methods."""

    def export(self, spans):
        return None

    def shutdown(self):
        pass


def test_no_endpoint_uses_console_exporter(monkeypatch, tmp_path):
    fake = _settings_with(monkeypatch, tmp_path, otel_enabled=True, otel_exporter_endpoint=None)

    used = []
    monkeypatch.setattr("core.telemetry.get_settings", lambda: fake)
    monkeypatch.setattr(
        "core.telemetry.ConsoleSpanExporter",
        lambda: used.append("console") or _ConsoleExporter(),
    )

    try:
        init_otel(_make_app())
    finally:
        _shutdown_provider()
    assert used == ["console"]
