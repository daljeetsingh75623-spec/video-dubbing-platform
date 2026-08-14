"""
Configuration system.

Precedence (highest wins): environment variables > YAML config file (path set
by CONFIG_FILE env var, defaults to config/default.yaml) > field defaults
defined below.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class YamlConfigSource:
    """Loads a YAML file into a flat dict pydantic-settings can consume."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with open(self.path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return {}
        return data


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
    s3_endpoint_url: str | None = Field(default="http://minio:9000")
    s3_bucket: str = Field(default="video-dubbing")
    s3_access_key: str | None = Field(default="minioadmin")
    s3_secret_key: str | None = Field(default="minioadmin")
    s3_region: str = Field(default="us-east-1")

    # --- Upload / processing limits ---
    max_upload_size_mb: int = Field(default=500)
    max_video_duration_seconds: int = Field(default=600)
    allowed_video_formats: list[str] = Field(default_factory=lambda: ["mp4", "mov", "avi", "mkv"])
    max_concurrent_uploads: int = Field(default=10)
    max_concurrent_processing_jobs: int = Field(default=5)
    processing_timeout_seconds: int = Field(default=1800)
    retry_count: int = Field(default=3)
    queue_max_length: int = Field(default=1000)

    # --- Provider selection ---
    stt_provider: str = Field(default="stub")
    translation_provider: str = Field(default="stub")
    tts_provider: str = Field(default="stub")
    diarization_provider: str = Field(default="stub")

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

    # --- Security ---
    api_key_auth_enabled: bool = Field(default=False)
    api_key: str | None = Field(default=None)
    rate_limit_per_minute: int = Field(default=60)

    # --- Observability ---
    otel_enabled: bool = Field(default=True)
    otel_exporter_endpoint: str | None = Field(default=None)

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
        yaml_source = YamlConfigSource(config_file)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            lambda: yaml_source(),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call this everywhere instead of instantiating Settings() directly."""
    return Settings()