import os
import json
import base64
from dataclasses import dataclass
from typing import Any, final

import pipmaster as pm
if not pm.is_installed("aioboto3"):
    pm.install("aioboto3")
import aioboto3
from botocore.exceptions import ClientError

from lightrag.base import BaseKVStorage
from lightrag.utils import logger
from lightrag.namespace import NameSpace, is_namespace

@final
@dataclass
class S3KVStorage(BaseKVStorage):
    def __post_init__(self):
        self.endpoint_url = os.environ.get("S3_ENDPOINT_URL")
        self.access_key = os.environ.get("S3_ACCESS_KEY")
        self.secret_key = os.environ.get("S3_SECRET_KEY")
        self.bucket = os.environ.get("S3_BUCKET", "lightrag-markdown")
        if not self.endpoint_url or not self.access_key or not self.secret_key:
            raise ValueError("S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY must be set in environment")
        
        self.session = aioboto3.Session()

    def _get_client(self):
        # We need to yield the client so it can be used with `async with`
        return self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    async def initialize(self):
        """Initialize storage by checking/creating bucket"""
        async with self._get_client() as s3:
            try:
                await s3.head_bucket(Bucket=self.bucket)
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    try:
                        await s3.create_bucket(Bucket=self.bucket)
                        logger.info(f"[{self.workspace}] Created S3 bucket: {self.bucket}")
                    except Exception as ce:
                        logger.error(f"[{self.workspace}] Failed to create bucket: {ce}")
                        raise
                else:
                    raise

    def _get_key(self, doc_id: str) -> str:
        # Include workspace in the path for data isolation. If workspace is empty, use default.
        ws = self.workspace if self.workspace else "default"
        is_full_docs = is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS)
        if is_full_docs:
            return f"{self.namespace}/{ws}/{doc_id}.md"
        else:
            return f"{self.namespace}/{ws}/{doc_id}.json"

    async def index_done_callback(self) -> None:
        pass

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        key = self._get_key(id)
        is_full_docs = is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS)
        async with self._get_client() as s3:
            try:
                response = await s3.get_object(Bucket=self.bucket, Key=key)
                async with response['Body'] as stream:
                    data = await stream.read()
                    if is_full_docs:
                        metadata = response.get('Metadata', {})
                        # Some metadata may be base64-encoded if it contained non-ASCII chars
                        fp = metadata.get('file_path', '')
                        if metadata.get('is_b64', 'false') == 'true':
                            try:
                                fp = base64.b64decode(fp).decode("utf-8")
                            except:
                                pass
                        result = {
                            "content": data.decode("utf-8"),
                            "file_path": fp
                        }
                    else:
                        result = json.loads(data.decode("utf-8"))
                    result["_id"] = id
                    return result
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    return None
                raise

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        results = []
        for doc_id in ids:
            results.append(await self.get_by_id(doc_id))
        return results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        missing = set()
        for doc_id in keys:
            if await self.get_by_id(doc_id) is None:
                missing.add(doc_id)
        return missing

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        
        is_full_docs = is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS)
        async with self._get_client() as s3:
            for k, v in data.items():
                key = self._get_key(k)
                if is_full_docs:
                    content_bytes = v["content"].encode("utf-8")
                    content_type = "text/markdown"
                    # Handle non-ASCII in metadata by base64 encoding it
                    fp = v.get("file_path", "")
                    fp_b64 = base64.b64encode(fp.encode("utf-8")).decode("ascii")
                    metadata_to_save = {"file_path": fp_b64, "is_b64": "true"}
                else:
                    content_bytes = json.dumps(v).encode("utf-8")
                    content_type = "application/json"
                    metadata_to_save = {}

                try:
                    await s3.put_object(
                        Bucket=self.bucket,
                        Key=key,
                        Body=content_bytes,
                        ContentType=content_type,
                        Metadata=metadata_to_save
                    )
                except Exception as e:
                    logger.error(f"[{self.workspace}] Failed to upsert to S3: {e}")
                    raise

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
            
        async with self._get_client() as s3:
            objects_to_delete = [{'Key': self._get_key(doc_id)} for doc_id in ids]
            
            # S3 can delete up to 1000 objects per request
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i:i + 1000]
                try:
                    await s3.delete_objects(
                        Bucket=self.bucket,
                        Delete={'Objects': batch, 'Quiet': True}
                    )
                except Exception as e:
                    logger.error(f"[{self.workspace}] Failed to delete from S3: {e}")

    async def is_empty(self) -> bool:
        ws = self.workspace if self.workspace else "default"
        prefix = f"{self.namespace}/{ws}/"
        async with self._get_client() as s3:
            response = await s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1)
            return response.get('KeyCount', 0) == 0

    async def drop(self) -> dict[str, str]:
        ws = self.workspace if self.workspace else "default"
        prefix = f"{self.namespace}/{ws}/"
        try:
            async with self._get_client() as s3:
                paginator = s3.get_paginator('list_objects_v2')
                async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                    if 'Contents' in page:
                        objects = [{'Key': obj['Key']} for obj in page['Contents']]
                        for i in range(0, len(objects), 1000):
                            batch = objects[i:i + 1000]
                            await s3.delete_objects(Bucket=self.bucket, Delete={'Objects': batch, 'Quiet': True})
            logger.info(f"[{self.workspace}] Dropped all data in {prefix} from S3")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Failed to drop data from S3: {e}")
            return {"status": "error", "message": str(e)}
