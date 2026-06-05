from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import ArtifactRef, StageName, utc_now_iso


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_stage(stage: StageName | str) -> str:
    return StageName(stage).value if not isinstance(stage, StageName) else stage.value


class PipelineArtifactStore:
    async def write_jsonl(
        self,
        *,
        run_id: str,
        stage: StageName | str,
        name: str,
        rows: list[dict[str, Any]],
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        raise NotImplementedError

    async def read_jsonl(self, uri: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def write_json(
        self,
        *,
        run_id: str,
        stage: StageName | str,
        name: str,
        data: dict[str, Any],
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        raise NotImplementedError

    async def read_json(self, uri: str) -> dict[str, Any]:
        raise NotImplementedError

    async def write_manifest(
        self,
        *,
        run_id: str,
        stage: StageName | str,
        name: str,
        manifest: dict[str, Any],
    ) -> ArtifactRef:
        manifest.setdefault("created_at", utc_now_iso())
        return await self.write_json(
            run_id=run_id,
            stage=stage,
            name=name,
            data=manifest,
            kind="manifest",
            metadata={"stage": _normalize_stage(stage)},
        )


class LocalPipelineArtifactStore(PipelineArtifactStore):
    def __init__(self, root_dir: str | os.PathLike[str] | None = None) -> None:
        self.root_dir = Path(
            root_dir or os.getenv("PIPELINE_ARTIFACT_LOCAL_DIR", "/tmp/lightrag_pipeline")
        )

    def _path(self, run_id: str, stage: StageName | str, name: str) -> Path:
        safe_name = name.replace("/", "_")
        return self.root_dir / run_id / _normalize_stage(stage) / safe_name

    async def _write_bytes(self, path: Path, data: bytes) -> None:
        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        await asyncio.to_thread(write)

    async def _read_bytes(self, path: Path) -> bytes:
        return await asyncio.to_thread(path.read_bytes)

    async def write_jsonl(
        self,
        *,
        run_id: str,
        stage: StageName | str,
        name: str,
        rows: list[dict[str, Any]],
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        payload = "".join(_json_dumps(row) + "\n" for row in rows).encode("utf-8")
        path = self._path(run_id, stage, name)
        await self._write_bytes(path, payload)
        return ArtifactRef(
            uri=path.as_uri(),
            kind=kind,
            row_count=len(rows),
            checksum=_checksum(payload),
            metadata=metadata or {},
        )

    async def read_jsonl(self, uri: str) -> list[dict[str, Any]]:
        parsed = urlparse(uri)
        path = Path(parsed.path if parsed.scheme == "file" else uri)
        payload = await self._read_bytes(path)
        rows = []
        for line in payload.decode("utf-8").split("\n"):
            if line.strip():
                rows.append(json.loads(line))
        return rows

    async def write_json(
        self,
        *,
        run_id: str,
        stage: StageName | str,
        name: str,
        data: dict[str, Any],
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        payload = (_json_dumps(data) + "\n").encode("utf-8")
        path = self._path(run_id, stage, name)
        await self._write_bytes(path, payload)
        return ArtifactRef(
            uri=path.as_uri(),
            kind=kind,
            row_count=1,
            checksum=_checksum(payload),
            metadata=metadata or {},
        )

    async def read_json(self, uri: str) -> dict[str, Any]:
        parsed = urlparse(uri)
        path = Path(parsed.path if parsed.scheme == "file" else uri)
        payload = await self._read_bytes(path)
        return json.loads(payload.decode("utf-8"))


class S3PipelineArtifactStore(PipelineArtifactStore):
    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        prefix: str | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL")
        self.access_key = (
            access_key
            or os.getenv("S3_ACCESS_KEY")
            or os.getenv("S3_ACCESS_KEY_ID")
        )
        self.secret_key = (
            secret_key
            or os.getenv("S3_SECRET_KEY")
            or os.getenv("S3_SECRET_ACCESS_KEY")
        )
        self.bucket = bucket or os.getenv("PIPELINE_ARTIFACT_BUCKET") or os.getenv(
            "S3_BUCKET", "lightrag-markdown"
        )
        self.prefix = (prefix or os.getenv("PIPELINE_ARTIFACT_PREFIX", "pipeline_tmp")).strip("/")

        missing = [
            name
            for name, value in {
                "S3_ENDPOINT_URL": self.endpoint_url,
                "S3_ACCESS_KEY": self.access_key,
                "S3_SECRET_KEY": self.secret_key,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing S3 artifact configuration: {', '.join(missing)}")

    def _key(self, run_id: str, stage: StageName | str, name: str) -> str:
        safe_name = name.replace("/", "_")
        return f"{self.prefix}/{run_id}/{_normalize_stage(stage)}/{safe_name}"

    async def _client(self):
        import aioboto3

        session = aioboto3.Session()
        return session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    async def _put(self, key: str, payload: bytes) -> None:
        async with await self._client() as s3:
            await s3.put_object(Bucket=self.bucket, Key=key, Body=payload)

    async def _get(self, uri: str) -> bytes:
        from botocore.exceptions import ClientError

        parsed = urlparse(uri)
        bucket = parsed.netloc or self.bucket
        key = parsed.path.lstrip("/")

        max_retries = int(os.getenv("PIPELINE_ARTIFACT_S3_RETRY_ATTEMPTS", "3"))
        base_delay = float(os.getenv("PIPELINE_ARTIFACT_S3_RETRY_DELAY", "2.0"))

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                async with await self._client() as s3:
                    response = await s3.get_object(Bucket=bucket, Key=key)
                    async with response["Body"] as stream:
                        return await stream.read()
            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code", "")
                # NoSuchKey is not retryable
                if error_code == "NoSuchKey":
                    raise
                last_exc = exc
            except Exception as exc:
                last_exc = exc

            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                from lightrag.utils import logger as _logger
                _logger.warning(
                    "[Pipeline] S3 get retry attempt=%s/%s delay=%.1fs uri=%s error=%s",
                    attempt, max_retries, delay, uri, last_exc,
                )
                await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    async def write_jsonl(
        self,
        *,
        run_id: str,
        stage: StageName | str,
        name: str,
        rows: list[dict[str, Any]],
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        payload = "".join(_json_dumps(row) + "\n" for row in rows).encode("utf-8")
        key = self._key(run_id, stage, name)
        await self._put(key, payload)
        return ArtifactRef(
            uri=f"s3://{self.bucket}/{key}",
            kind=kind,
            row_count=len(rows),
            checksum=_checksum(payload),
            metadata=metadata or {},
        )

    async def read_jsonl(self, uri: str) -> list[dict[str, Any]]:
        payload = await self._get(uri)
        rows = []
        for line in payload.decode("utf-8").split("\n"):
            if line.strip():
                rows.append(json.loads(line))
        return rows

    async def write_json(
        self,
        *,
        run_id: str,
        stage: StageName | str,
        name: str,
        data: dict[str, Any],
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        payload = (_json_dumps(data) + "\n").encode("utf-8")
        key = self._key(run_id, stage, name)
        await self._put(key, payload)
        return ArtifactRef(
            uri=f"s3://{self.bucket}/{key}",
            kind=kind,
            row_count=1,
            checksum=_checksum(payload),
            metadata=metadata or {},
        )

    async def read_json(self, uri: str) -> dict[str, Any]:
        payload = await self._get(uri)
        return json.loads(payload.decode("utf-8"))


def create_artifact_store_from_env() -> PipelineArtifactStore:
    backend = os.getenv("PIPELINE_ARTIFACT_BACKEND", "s3").lower()
    if backend == "local":
        return LocalPipelineArtifactStore()
    return S3PipelineArtifactStore()
