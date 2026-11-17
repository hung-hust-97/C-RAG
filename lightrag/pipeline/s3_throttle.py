import asyncio
import os
import time

_CHUNK = 1 << 20  # 1 MiB
_bucket = None
_loaded = False


class _TokenBucket:
    def __init__(self, bytes_per_sec: float) -> None:
        self.rate = bytes_per_sec
        self.tokens = bytes_per_sec  # allow ~1s burst
        self.last = time.monotonic()

    async def consume(self, n: int) -> None:
        now = time.monotonic()
        self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)
        self.last = now
        self.tokens -= n  # negative = debt, subsequent read must wait
        if self.tokens < 0:
            await asyncio.sleep(-self.tokens / self.rate)


def _get_bucket():
    global _bucket, _loaded
    if not _loaded:
        mbps = float(os.getenv("S3_READ_MAX_MBPS") or 0)
        _bucket = _TokenBucket(mbps * 1_000_000 / 8) if mbps > 0 else None
        _loaded = True
    return _bucket


async def _read_chunk(stream, n: int) -> bytes:
    content = getattr(stream, "content", None)  # aiohttp.ClientResponse
    if content is not None:
        return await content.read(n)
    return await stream.read(n)  # aiobotocore StreamingBody


async def read_body(stream) -> bytes:
    bucket = _get_bucket()
    if bucket is None:
        return await stream.read()
    parts = []
    while chunk := await _read_chunk(stream, _CHUNK):
        parts.append(chunk)
        await bucket.consume(len(chunk))
    return b"".join(parts)

