from __future__ import annotations

from functools import lru_cache

from config.settings import get_settings
from storage.base import StorageBackend
from storage.local import LocalStorageBackend
from storage.s3 import S3StorageBackend


@lru_cache
def get_storage_backend() -> StorageBackend:
    settings = get_settings()
    if settings.storage_backend == "local":
        return LocalStorageBackend(base_path=settings.storage_local_path)
    if settings.storage_backend == "s3":
        return S3StorageBackend(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            region=settings.s3_region,
        )
    raise ValueError(f"Unknown storage_backend: {settings.storage_backend}")