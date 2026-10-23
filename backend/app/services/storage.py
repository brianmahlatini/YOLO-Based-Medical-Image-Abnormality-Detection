from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ObjectStorage(Protocol):
    def put_bytes(self, key: str, payload: bytes, content_type: str) -> str:
        ...


class LocalObjectStorage:
    def __init__(self, root: Path = Path("data/uploads")) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, payload: bytes, content_type: str) -> str:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)


class S3ObjectStorage:
    def __init__(self, bucket: str, endpoint_url: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Install boto3 to enable S3-compatible storage.") from exc

        self.bucket = bucket
        self.client = boto3.client("s3", endpoint_url=endpoint_url)

    def put_bytes(self, key: str, payload: bytes, content_type: str) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=payload, ContentType=content_type)
        return f"s3://{self.bucket}/{key}"
