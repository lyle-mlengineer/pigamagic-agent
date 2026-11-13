"""Blob storage: content-addressed, immutable, S3-compatible (design.md §9).

Three rules govern this module, and all three exist to protect the user's
photographs:

1. **Content-addressed.** Every object is keyed by the SHA-256 of its bytes. Two
   identical generations occupy one object, and a re-run is free.
2. **Immutable.** `put` never overwrites. Writing the same key twice is a no-op,
   not a mutation. A delivered photo can therefore never change under the user —
   an edit creates a *new* object with lineage back to the parent.
3. **Portable.** The same `BlobStore` protocol covers a local directory and any
   S3-compatible service. The pipeline depends only on the protocol, so a laptop
   and Backblaze B2 run identical code.

Why boto3 rather than aiobotocore: boto3 is already installed and battle-tested,
and `asyncio.to_thread` keeps the event loop free. Adding a second AWS SDK for
marginal async purity is not a trade worth making.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.agent.config import AgentSettings, get_settings

CONTENT_TYPE_EXTENSIONS: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "application/json": "json",
    "text/plain": "txt",
}


class StorageError(Exception):
    """A blob operation failed. Never leaks an SDK exception to the caller."""


# --------------------------------------------------------------------------
# Value types
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredBlob:
    """A persisted object, addressed by content."""

    key: str
    sha256: str
    size: int
    content_type: str
    bucket: str = ""
    url: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sha256": self.sha256,
            "size": self.size,
            "content_type": self.content_type,
            "bucket": self.bucket,
            "url": self.url,
        }


@dataclass(frozen=True, slots=True)
class BlobRef:
    """Everything needed to fetch a blob back."""

    key: str
    sha256: str = ""


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


@runtime_checkable
class BlobStore(Protocol):
    """Persistence for byte blobs. All methods are async.

    Implementations must be safe for concurrent use and must never overwrite an
    existing key with different content.
    """

    name: str

    async def put(
        self, content: bytes, *, content_type: str = "application/octet-stream", prefix: str = ""
    ) -> StoredBlob: ...

    async def get(self, key: str) -> bytes: ...

    async def exists(self, key: str) -> bool: ...

    async def delete(self, key: str) -> bool: ...

    async def list(self, prefix: str = "") -> list[StoredBlob]: ...


# --------------------------------------------------------------------------
# Key building
# --------------------------------------------------------------------------


def content_key(content: bytes, *, content_type: str, prefix: str = "") -> str:
    """Build a content-addressed key.

    Layout: `<prefix>/<aa>/<bb>/<sha256>.<ext>` — a two-level fan-out keeps any
    single directory small, which matters for local stores on Windows.
    """
    digest = hashlib.sha256(content).hexdigest()
    extension = CONTENT_TYPE_EXTENSIONS.get(content_type, "bin")
    parts = [part.strip("/") for part in (prefix,) if part and part.strip("/")]
    parts.extend([digest[:2], digest[2:4], f"{digest}.{extension}"])
    return "/".join(parts)


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# --------------------------------------------------------------------------
# Local filesystem store
# --------------------------------------------------------------------------


@dataclass
class LocalBlobStore:
    """Filesystem-backed store, for development and tests.

    Writes are atomic: content lands in a temp file and is then renamed into
    place, so a crash mid-write cannot leave a truncated photo behind. Rename is
    also what makes "never overwrite" enforceable.
    """

    root: str | Path = "./var/media"
    name: str = "local"
    base_url: str = ""

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def _path(self, key: str) -> Path:
        # Reject traversal explicitly; a key from a URL must never escape root.
        safe = key.replace("\\", "/").strip("/")
        if not safe or ".." in Path(safe).parts:
            raise StorageError(f"Refusing unsafe storage key: {key!r}")
        return self.root / safe

    async def put(
        self, content: bytes, *, content_type: str = "application/octet-stream", prefix: str = ""
    ) -> StoredBlob:
        if not content:
            raise StorageError("Refusing to store empty content.")

        key = content_key(content, content_type=content_type, prefix=prefix)
        target = self._path(key)

        def _write() -> bool:
            if target.exists():
                return False  # immutable: existing object wins
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, temp = tempfile.mkstemp(dir=str(target.parent), suffix=".part")
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(content)
                os.replace(temp, target)
            except Exception:
                Path(temp).unlink(missing_ok=True)
                raise
            return True

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Local write failed for {key}: {exc}") from exc

        return StoredBlob(
            key=key,
            sha256=sha256_hex(content),
            size=len(content),
            content_type=content_type,
            bucket=str(self.root),
            url=f"{self.base_url.rstrip('/')}/{key}" if self.base_url else "",
        )

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise StorageError(f"No blob at {key}") from exc
        except OSError as exc:
            raise StorageError(f"Local read failed for {key}: {exc}") from exc

    async def exists(self, key: str) -> bool:
        try:
            return await asyncio.to_thread(self._path(key).exists)
        except StorageError:
            return False

    async def delete(self, key: str) -> bool:
        path = self._path(key)
        try:
            await asyncio.to_thread(path.unlink)
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise StorageError(f"Local delete failed for {key}: {exc}") from exc

    async def list(self, prefix: str = "") -> list[StoredBlob]:
        def _walk() -> list[StoredBlob]:
            base = self._path(prefix) if prefix else self.root
            if not base.exists():
                return []
            found: list[StoredBlob] = []
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.suffix == ".part":
                    continue
                relative = path.relative_to(self.root).as_posix()
                found.append(
                    StoredBlob(
                        key=relative,
                        sha256=path.stem,
                        size=path.stat().st_size,
                        content_type=_guess_content_type(path.suffix),
                        bucket=str(self.root),
                    )
                )
            return found

        return await asyncio.to_thread(_walk)

    def url_for(self, key: str) -> str:
        return f"{self.base_url.rstrip('/')}/{key}" if self.base_url else ""

    def wipe(self) -> None:
        """Delete everything. Tests and local resets only."""
        if self.root.exists():
            shutil.rmtree(self.root)


def _guess_content_type(suffix: str) -> str:
    lowered = suffix.lower().lstrip(".")
    for content_type, extension in CONTENT_TYPE_EXTENSIONS.items():
        if extension == lowered:
            return content_type
    return "application/octet-stream"


# --------------------------------------------------------------------------
# S3-compatible store (AWS S3, Backblaze B2, Cloudflare R2, MinIO, Wasabi)
# --------------------------------------------------------------------------


@dataclass
class S3BlobStore:
    """Any S3-compatible object store.

    Backblaze B2 works here unchanged: point `endpoint_url` at
    `https://s3.<region>.backblazeb2.com`, use the *application key ID* as the
    access key and the *application key* as the secret. The B2 master key is not
    accepted by the S3 API, and region must match the bucket's region.

    `addressing_style="path"` is required by B2 and MinIO; virtual-hosted style
    breaks on some non-AWS providers.
    """

    bucket: str
    endpoint_url: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = ""
    prefix: str = ""
    name: str = "s3"
    addressing_style: str = "path"
    public_base_url: str = ""
    client: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.bucket:
            raise StorageError("An S3 bucket name is required.")

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            import boto3  # noqa: PLC0415
            from botocore.config import Config  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise StorageError(
                "S3 storage needs boto3. Install it with: pip install boto3"
            ) from exc

        if not self.access_key_id or not self.secret_access_key:
            raise StorageError(
                f"{self.name} storage requires an access key id and secret access key."
            )

        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": self.addressing_style},
            retries={"max_attempts": 3, "mode": "standard"},
        )
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url or None,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region or None,
            config=config,
        )
        return self.client

    def _key(self, key: str) -> str:
        safe = key.replace("\\", "/").strip("/")
        if not safe or ".." in Path(safe).parts:
            raise StorageError(f"Refusing unsafe storage key: {key!r}")
        return f"{self.prefix.strip('/')}/{safe}" if self.prefix.strip("/") else safe

    def _strip(self, key: str) -> str:
        marker = f"{self.prefix.strip('/')}/" if self.prefix.strip("/") else ""
        return key[len(marker) :] if marker and key.startswith(marker) else key

    async def put(
        self, content: bytes, *, content_type: str = "application/octet-stream", prefix: str = ""
    ) -> StoredBlob:
        if not content:
            raise StorageError("Refusing to store empty content.")

        key = content_key(content, content_type=content_type, prefix=prefix)
        full_key = self._key(key)
        digest = sha256_hex(content)

        def _upload() -> None:
            client = self._client()
            # Immutability: skip if the object is already there.
            try:
                existing = client.head_object(Bucket=self.bucket, Key=full_key)
                if int(existing.get("ContentLength", -1)) == len(content):
                    return
            except Exception:  # noqa: BLE001 - missing object is the normal path
                pass
            client.put_object(
                Bucket=self.bucket,
                Key=full_key,
                Body=content,
                ContentType=content_type,
                Metadata={"sha256": digest},
            )

        try:
            await asyncio.to_thread(_upload)
        except StorageError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"{self.name} upload failed for {key}: {exc}") from exc

        return StoredBlob(
            key=key,
            sha256=digest,
            size=len(content),
            content_type=content_type,
            bucket=self.bucket,
            url=self.url_for(key),
        )

    async def get(self, key: str) -> bytes:
        def _download() -> bytes:
            client = self._client()
            try:
                response = client.get_object(Bucket=self.bucket, Key=self._key(key))
                return response["Body"].read()
            except Exception as exc:  # noqa: BLE001
                raise StorageError(f"{self.name} read failed for {key}: {exc}") from exc

        return await asyncio.to_thread(_download)

    async def exists(self, key: str) -> bool:
        def _head() -> bool:
            client = self._client()
            try:
                client.head_object(Bucket=self.bucket, Key=self._key(key))
                return True
            except Exception:  # noqa: BLE001
                return False

        try:
            return await asyncio.to_thread(_head)
        except StorageError:
            return False

    async def delete(self, key: str) -> bool:
        def _delete() -> bool:
            client = self._client()
            try:
                client.delete_object(Bucket=self.bucket, Key=self._key(key))
                return True
            except Exception as exc:  # noqa: BLE001
                raise StorageError(f"{self.name} delete failed for {key}: {exc}") from exc

        return await asyncio.to_thread(_delete)

    async def list(self, prefix: str = "") -> list[StoredBlob]:
        full_prefix = self._key(prefix) if prefix else (self.prefix.strip("/") or "")

        def _list() -> list[StoredBlob]:
            client = self._client()
            found: list[StoredBlob] = []
            try:
                paginator = client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                    for item in page.get("Contents", []) or []:
                        key = self._strip(item["Key"])
                        found.append(
                            StoredBlob(
                                key=key,
                                sha256=Path(key).stem,
                                size=int(item.get("Size", 0)),
                                content_type=_guess_content_type(Path(key).suffix),
                                bucket=self.bucket,
                                url=self.url_for(key),
                            )
                        )
            except Exception as exc:  # noqa: BLE001
                raise StorageError(f"{self.name} list failed: {exc}") from exc
            return found

        return await asyncio.to_thread(_list)

    def url_for(self, key: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url.rstrip('/')}/{self._key(key)}"
        return ""


@dataclass
class BackblazeB2Store(S3BlobStore):
    """Backblaze B2 via its S3-compatible API.

    A thin specialisation: B2 differs from generic S3 only in defaults.

    - endpoint is derived from the region: `https://s3.<region>.backblazeb2.com`
    - path addressing is mandatory
    - the **application key ID** is the access key; the **application key** is
      the secret. The account master key is rejected by the S3 API.
    - an account lives in exactly one region
    """

    name: str = "backblaze-b2"
    addressing_style: str = "path"

    def __post_init__(self) -> None:
        if not self.region:
            raise StorageError(
                "Backblaze B2 requires a region, e.g. us-west-004. Find it in the "
                "bucket's endpoint URL (s3.us-west-004.backblazeb2.com)."
            )
        if not self.endpoint_url:
            self.endpoint_url = b2_endpoint(self.region)
        super().__post_init__()


def b2_endpoint(region: str) -> str:
    return f"https://s3.{region.strip()}.backblazeb2.com"


def b2_region_from_endpoint(endpoint: str) -> str:
    """Extract the region from a B2 endpoint URL, so users can paste one value."""
    text = endpoint.strip().rstrip("/")
    text = text.replace("https://", "").replace("http://", "")
    if text.startswith("s3.") and ".backblazeb2.com" in text:
        return text[len("s3.") : text.index(".backblazeb2.com")]
    return ""


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def build_blob_store(settings: AgentSettings | None = None, *, name: str | None = None) -> BlobStore:
    """Construct the configured store.

    Defaults to the local store so nothing needs credentials to run.
    """
    active = settings or get_settings()
    chosen = (name or getattr(active, "storage_provider", "local") or "local").strip().casefold()

    if chosen in {"local", "file", "filesystem", "files"}:
        return LocalBlobStore(
            root=active.media_dir,
            base_url=getattr(active, "media_base_url", "") or "",
        )

    if chosen in {"backblaze", "b2", "backblaze-b2"}:
        region = getattr(active, "storage_region", "") or b2_region_from_endpoint(
            getattr(active, "storage_endpoint", "")
        )
        if not region:
            raise StorageError(
                "Backblaze B2 needs a region. Set AGENT_STORAGE_REGION=us-west-004 "
                "or paste the endpoint into AGENT_STORAGE_ENDPOINT."
            )
        return BackblazeB2Store(
            bucket=getattr(active, "storage_bucket", ""),
            region=region,
            access_key_id=getattr(active, "storage_access_key_id", ""),
            secret_access_key=getattr(active, "storage_secret_access_key", ""),
            prefix=getattr(active, "storage_prefix", ""),
            public_base_url=getattr(active, "storage_public_base_url", ""),
        )

    if chosen in {"s3", "r2", "minio", "wasabi", "spaces", "aws"}:
        return S3BlobStore(
            bucket=getattr(active, "storage_bucket", ""),
            endpoint_url=getattr(active, "storage_endpoint", ""),
            access_key_id=getattr(active, "storage_access_key_id", ""),
            secret_access_key=getattr(active, "storage_secret_access_key", ""),
            region=getattr(active, "storage_region", ""),
            prefix=getattr(active, "storage_prefix", ""),
            public_base_url=getattr(active, "storage_public_base_url", ""),
            name=chosen,
        )

    raise StorageError(
        f"Unknown storage provider {chosen!r}. Choose local, s3 or backblaze."
    )


def write_manifest(store: LocalBlobStore, manifest: dict[str, Any], name: str = "manifest.json") -> Path:
    """Write a JSON manifest next to a local store's blobs (developer convenience)."""
    path = Path(store.root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BackblazeB2Store",
    "BlobRef",
    "BlobStore",
    "CONTENT_TYPE_EXTENSIONS",
    "LocalBlobStore",
    "S3BlobStore",
    "StorageError",
    "StoredBlob",
    "b2_endpoint",
    "b2_region_from_endpoint",
    "build_blob_store",
    "content_key",
    "sha256_hex",
    "utc_now",
    "write_manifest",
]
