"""
OpenTelemetry bootstrap.

Config gate: `otel_enabled`. When enabled, the FastAPI app is instrumented
and spans are exported to the configured OTLP endpoint (gRPC). If no
endpoint is set, spans go to the console so tracing still works out of the
box during local dev. Guarding everything here means a misconfigured/missing
collector never takes the API down — exporters fail asynchronously.
"""
from __future__ import annotations

import structlog
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from config.settings import get_settings

log = structlog.get_logger()


def init_otel(app: FastAPI) -> None:
    settings = get_settings()
    if not settings.otel_enabled:
        return

    resource = Resource.create({SERVICE_NAME: "video-dubbing-api"})
    provider = TracerProvider(resource=resource)

    if settings.otel_exporter_endpoint:
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_endpoint,
            insecure=True,
        )
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

    log.info(
        "otel_initialized",
        exporter="otlp" if settings.otel_exporter_endpoint else "console",
        endpoint=settings.otel_exporter_endpoint,
    )
