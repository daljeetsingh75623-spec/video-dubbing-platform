from __future__ import annotations

import asyncio
from pathlib import Path

import boto3
from botocore.client import Config as BotoConfig

from storage.base import StorageBackend


class S3StorageBackend(StorageBackend):
    """
    Works against real AWS S3 or MinIO — same client, MinIO just needs
    `endpoint_url` set.
    """

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
    ):
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=BotoConfig(signature_version="s3v4"),
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        existing = [b["Name"] for b in self._client.list_buckets().get("Buckets", [])]
        if self.bucket not in existing:
            self._client.create_bucket(Bucket=self.bucket)

    async def upload(self, local_path: str | Path, key: str) -> str:
        await asyncio.to_thread(self._client.upload_file, str(local_path), self.bucket, key)
        return key

    async def download(self, key: str, local_path: str | Path) -> str:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._client.download_file, self.bucket, key, str(local_path))
        return str(local_path)

    async def get_url(self, key: str, expires_seconds: int = 3600) -> str:
        return await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._client.delete_object, Bucket=self.bucket, Key=key)

    async def exists(self, key: str) -> bool:
        try:
            await asyncio.to_thread(self._client.head_object, Bucket=self.bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False