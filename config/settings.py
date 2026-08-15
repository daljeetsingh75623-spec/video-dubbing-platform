"""
Configuration system.

Precedence (highest wins): constructor kwargs > environment variables >
.env file > secrets directory (path set by SECRETS_DIR) > cloud config
service (URL set by CLOUD_CONFIG_URL) > YAML config file (path set by
CONFIG_FILE env var, defaults to config/default.yaml) > field defaults.

Nothing is hardcoded in application code — every operational limit below can
be overridden through any of the four channels (env vars, config files,
secrets management, cloud config services).
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SecretsSettingsSource,
    SettingsConfigDict,
)

log = structlog.get_logger()


class YamlConfigSource(PydanticBaseSettingsSource):
    """Loads a YAML file into a flat dict pydantic-settings can consume."""

    def __init__(self, path: Path, settings_cls: type[BaseSettings]):
        super().__init__(settings_cls)
        self.path = path
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with open(self.path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def __call__(self) -> dict[str, Any]:
        return self._data

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False


class CloudConfigSource(PydanticBaseSettingsSource):
    """Loads config from any HTTP(S) endpoint returning YAML or JSON.

    This is the "cloud configuration services" channel: point CLOUD_CONFIG_URL
    at the URL a cloud config service exposes for this app (AWS AppConfig via
    the config-extension agent, Consul, Vault agent, Spring Cloud Config, a
    custom config microservice, ...) and every instance picks up the same
    values. A missing or unreachable service is *not* fatal — it is logged and
    treated as "no data", so startup and lower-precedence sources are never
    blocked.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        url: str | None,
        timeout: float,
    ):
        super().__init__(settings_cls)
        self.url = url
        self.timeout = timeout
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.url:
            return {}
        try:
            req = urllib.request.Request(
                self.url,
                headers={"Accept": "application/yaml, application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - never take the service down
            log.warning("cloud_config_unavailable", url=self.url, error=str(exc))
            return {}
        try:
            data = yaml.safe_load(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("cloud_config_invalid", url=self.url, error=str(exc))
            return {}
        if not isinstance(data, dict):
            log.warning("cloud_config_not_a_dict", url=self.url)
            return {}
        return data

    def __call__(self) -> dict[str, Any]:
        return self._data

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # --- Database ---
    database_url: str = Field(default="postgresql+asyncpg://postgres:postgres@postgres:5432/dubbing")

    # --- Redis / Celery ---
    redis_url: str = Field(default="redis://redis:6379/0")
    celery_broker_url: str = Field(default="redis://redis:6379/0")
    celery_result_backend: str = Field(default="redis://redis:6379/1")

    # --- Storage ---
    storage_backend: str = Field(default="local")   # "local" | "s3"
    storage_local_path: str = Field(default="/data/storage")
    # Scratch dir used by the pipeline for intermediate files (downloaded
    # source, audio chunks, composite track). Defaults to the platform temp dir.
    work_dir: str = Field(default_factory=tempfile.gettempdir)
    s3_endpoint_url: str | None = Field(default="http://minio:9000")
    s3_bucket: str = Field(default="video-dubbing")
    s3_access_key: str | None = Field(default="minioadmin")
    s3_secret_key: str | None = Field(default="minioadmin")
    s3_region: str = Field(default="us-east-1")

    # --- Upload / processing limits ---
    max_upload_size_mb: int = Field(default=500)
    max_video_duration_seconds: int = Field(default=600)
    # str | list[str]: env/dotenv/secrets values that aren't valid JSON keep the
    # raw string (union with str disables strict JSON decoding) and the
    # validator below splits a comma-separated value; JSON lists and YAML lists
    # pass straight through.
    allowed_video_formats: str | list[str] = Field(default_factory=lambda: ["mp4", "mov", "avi", "mkv"])
    max_concurrent_uploads: int = Field(default=10)
    max_concurrent_processing_jobs: int = Field(default=5)
    processing_timeout_seconds: int = Field(default=1800)
    retry_count: int = Field(default=3)
    queue_max_length: int = Field(default=1000)

    @field_validator("allowed_video_formats", mode="before")
    @classmethod
    def _split_formats(cls, v):
        """Accept a comma-separated string ("mp4,mov,avi") from any config
        channel (env var, dotenv, secrets dir, cloud config) in addition to a
        JSON/YAML list — pydantic-settings can only parse JSON lists from
        env-style sources otherwise."""
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v

    # Jobs stuck in an in-flight stage (diarizing..packaging) or queued without
    # being picked up for longer than this are marked failed by the
    # recover_stale_jobs beat task (see workers/tasks.py).
    stale_job_timeout_seconds: int = Field(default=3600)
    stale_job_recovery_interval_seconds: int = Field(default=60)

    # --- Provider selection ---
    stt_provider: str = Field(default="stub")
    translation_provider: str = Field(default="stub")
    tts_provider: str = Field(default="stub")
    diarization_provider: str = Field(default="stub")
    huggingface_token: str | None = Field(default=None)
    # Force a speaker-count range on diarization (e.g. min_speakers=2 for a
    # known 2-person conversation that pyannote sometimes merges into one).
    diarization_min_speakers: int | None = Field(default=None)
    diarization_max_speakers: int | None = Field(default=None)

    # --- Provider credentials ---
    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    gemini_api_key: str | None = Field(default=None)
    deepgram_api_key: str | None = Field(default=None)
    elevenlabs_api_key: str | None = Field(default=None)
    azure_speech_key: str | None = Field(default=None)
    azure_speech_region: str | None = Field(default=None)

    # --- Provider-level resilience ---
    provider_timeout_seconds: int = Field(default=60)
    provider_retry_count: int = Field(default=2)
    # Max transcript segments sent per translation call — keeps responses well
    # under the model's output-token limit so JSON never gets truncated.
    translation_batch_size: int = Field(default=25)

    # --- Security ---
    api_key_auth_enabled: bool = Field(default=False)
    api_key: str | None = Field(default=None)
    rate_limit_per_minute: int = Field(default=60)

    # --- Observability ---
    otel_enabled: bool = Field(default=True)
    otel_exporter_endpoint: str | None = Field(default=None)

    # --- Config channels ---
    # Directory whose files are read as settings values (filename == field
    # name, e.g. "secrets/huggingface_token"). Used for Docker secrets /
    # Kubernetes mounted secrets.
    secrets_dir: str | None = Field(default=None)
    # Optional URL of a cloud configuration service returning a YAML/JSON
    # blob of settings (see CloudConfigSource above).
    cloud_config_url: str | None = Field(default=None)
    cloud_config_timeout_seconds: float = Field(default=3.0)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        config_file = Path(os.environ.get("CONFIG_FILE", "config/default.yaml"))
        yaml_source = YamlConfigSource(config_file, settings_cls)

        secrets_dir = os.environ.get("SECRETS_DIR")
        secrets_source = SecretsSettingsSource(
            settings_cls, secrets_dir=secrets_dir or settings_cls.model_config.get("secrets_dir")
        )

        cloud_source = CloudConfigSource(
            settings_cls,
            url=os.environ.get("CLOUD_CONFIG_URL"),
            timeout=float(os.environ.get("CLOUD_CONFIG_TIMEOUT_SECONDS", "3.0")),
        )

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            secrets_source,
            cloud_source,
            yaml_source,
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call this everywhere instead of instantiating Settings() directly."""
    return Settings()