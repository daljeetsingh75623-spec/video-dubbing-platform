from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """
    Storage abstraction so the same application code works against local
    filesystem (dev) or S3/MinIO (prod) purely via config swap.
    """

    @abstractmethod
    async def upload(self, local_path: str | Path, key: str) -> str:
        """Upload a local file to `key`. Returns the storage key."""
        raise NotImplementedError

    @abstractmethod
    async def download(self, key: str, local_path: str | Path) -> str:
        """Download `key` to a local path. Returns the local path."""
        raise NotImplementedError

    @abstractmethod
    async def get_url(self, key: str, expires_seconds: int = 3600) -> str:
        """Return a (pre-signed, if applicable) URL to fetch `key`."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def exists(self, key: str) -> bool:
        raise NotImplementedError