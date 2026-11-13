"""Generic HTTP provider scaffolding shared by real adapters (design.md §8.4).

Concrete adapters should contain **only** vendor specifics: how to build a
request, how to read a response, and what the vendor charges. Everything else —
retry policy, timeouts, error mapping, budget enforcement, offline refusal —
lives here exactly once, so three adapters cannot drift into three different
retry behaviours.

Key rule from the design: the pipeline must never see a raw transport
exception. Every failure leaves this module as `ProviderError` (retryable or
not) or `ProviderNotConfigured`.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

import httpx

from app.agent.config import AgentSettings, get_settings
from app.agent.providers.base import (
    CostEstimate,
    OfflineModeError,
    ProviderError,
    ProviderNotConfigured,
)

# HTTP status codes worth retrying. 429 is a rate limit; 5xx is the vendor's
# problem. Everything else (400, 401, 403, 404, 422) will never succeed on
# retry and must not burn budget.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 8.0


class BudgetExceeded(ProviderError):
    """A call would cost more than the remaining budget. Never retried."""

    def __init__(self, message: str, *, spent_usd: float = 0.0, limit_usd: float = 0.0) -> None:
        super().__init__(message, retryable=False)
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd


@dataclass
class BudgetLedger:
    """Tracks spend across a run so a loop cannot quietly burn money.

    The pipeline creates one per shoot and passes it to adapters. `reserve`
    is called *before* the network request, so a failed call that the vendor
    still bills (many do) is not invisible.
    """

    limit_usd: float = 0.0
    spent_usd: float = 0.0
    calls: int = 0

    def __post_init__(self) -> None:
        if self.limit_usd < 0:
            raise ValueError("limit_usd must be >= 0")
        if self.spent_usd < 0:
            raise ValueError("spent_usd must be >= 0")

    @property
    def remaining_usd(self) -> float:
        if self.limit_usd <= 0:
            # Unlimited; remaining is meaningless but must be usable.
            return float("inf")
        return max(0.0, self.limit_usd - self.spent_usd)

    def check(self, estimate: CostEstimate) -> None:
        """Raise if `estimate` would exceed the limit. Does not spend."""
        if self.limit_usd <= 0:
            return
        if self.spent_usd + estimate.usd > self.limit_usd:
            raise BudgetExceeded(
                f"Call would cost ${estimate.usd:.4f} but only "
                f"${self.remaining_usd:.4f} of ${self.limit_usd:.2f} remains.",
                spent_usd=self.spent_usd,
                limit_usd=self.limit_usd,
            )

    def spend(self, amount: CostEstimate) -> None:
        self.check(amount)
        self.spent_usd += amount.usd
        self.calls += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit_usd": self.limit_usd,
            "spent_usd": round(self.spent_usd, 6),
            "calls": self.calls,
        }


def encode_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode_b64(text: str) -> bytes:
    """Decode base64 that may or may not carry a data-URL prefix."""
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    return base64.b64decode(text, validate=False)


def sniff_image_type(data: bytes, fallback: str = "image/png") -> str:
    """Identify an image from its magic bytes, ignoring vendor claims.

    Vendors occasionally label PNG bytes `image/jpeg`. `GeneratedImage` rejects
    mismatched content types, so trusting the header would turn a cosmetic
    vendor bug into a hard failure.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def image_dimensions(data: bytes) -> tuple[int, int]:
    """Best-effort width/height from image headers.

    Uses Pillow when available (already a project dependency); falls back to
    parsing the PNG IHDR chunk so the adapter still works without it.
    """
    try:
        from io import BytesIO

        from PIL import Image  # noqa: PLC0415

        with Image.open(BytesIO(data)) as image:
            return image.width, image.height
    except Exception:  # pragma: no cover - Pillow present in practice
        pass

    # PNG: width/height are big-endian uint32 at offsets 16 and 20.
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return 1024, 1024


def is_retryable_exception(exc: BaseException) -> bool:
    """Transport-level failures that are worth another attempt."""
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))


@dataclass
class HttpProvider:
    """Shared behaviour for adapters that speak HTTP.

    Subclasses set `name` and implement the vendor call. They inherit retry,
    offline refusal, budget accounting and error typing.
    """

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    settings: AgentSettings = field(default_factory=get_settings)
    budget: BudgetLedger | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    _client: httpx.AsyncClient | None = field(default=None, repr=False, compare=False)

    # ClassVar so no __init__ parameter is generated: subclasses override it
    # with a plain attribute, and the value cannot be shadowed by a default.
    name: ClassVar[str] = "http"

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        # Inherit from settings unless the caller overrode the dataclass default.
        if self.max_attempts == DEFAULT_MAX_ATTEMPTS:
            self.max_attempts = int(getattr(self.settings, "provider_max_attempts", self.max_attempts))
        if self.timeout_s == DEFAULT_TIMEOUT_S:
            self.timeout_s = float(getattr(self.settings, "provider_timeout_s", self.timeout_s))
        if self.budget is None:
            limit = float(getattr(self.settings, "image_budget_usd", 0.0) or 0.0)
            self.budget = BudgetLedger(limit_usd=limit)

    # -- configuration ---------------------------------------------------

    def require_credentials(self) -> None:
        if not self.api_key:
            raise ProviderNotConfigured(
                f"{self.name} requires an API key. Set the provider's key in the "
                f"environment, or use the fixture provider."
            )

    def guard_offline(self) -> None:
        """Refuse to make outbound calls when offline mode is on.

        This is a hard guarantee, not a nicety: tests and CI must never be able
        to spend money by accident.
        """
        if getattr(self.settings, "offline", False):
            raise OfflineModeError(f"{self.name} refused a call: offline mode is enabled.")

    def resolve_model(self, override: str | None) -> str:
        chosen = override or self.model or ""
        if not chosen:
            raise ProviderNotConfigured(f"{self.name} has no model configured.")
        return chosen

    # -- HTTP ------------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> HttpProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        files: Any = None,
        data: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        estimate: CostEstimate | None = None,
    ) -> Any:
        """Perform a request, retrying only when retrying could help.

        The budget is charged after a successful response, because several
        vendors bill on success only. A failed-then-retried call is charged once
        per successful response, which is the honest accounting.
        """
        self.guard_offline()
        self.require_credentials()
        assert self.budget is not None
        if estimate is not None:
            self.budget.check(estimate)

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._http().request(
                    method,
                    url,
                    headers=dict(headers or {}),
                    json=json_body,
                    content=content,
                    files=files,
                    data=data,
                    params=params,
                )
            except Exception as exc:  # noqa: BLE001 - narrowed immediately
                if not is_retryable_exception(exc):
                    raise ProviderError(
                        f"{self.name} transport error: {exc}", provider=self.name, retryable=False
                    ) from exc
                last_error = exc
                if attempt == self.max_attempts:
                    break
                await asyncio.sleep(min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S))
                continue

            if response.status_code >= 400:
                detail = _short_error(response)
                if response.status_code in RETRYABLE_STATUS and attempt < self.max_attempts:
                    await asyncio.sleep(min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S))
                    continue
                raise ProviderError(
                    f"{self.name} returned {response.status_code}: {detail}",
                    provider=self.name,
                    retryable=response.status_code in RETRYABLE_STATUS,
                    status=response.status_code,
                )

            if estimate is not None:
                self.budget.spend(estimate)

            if not response.content:
                return None
            try:
                return response.json()
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"{self.name} returned non-JSON content.", provider=self.name, retryable=False
                ) from exc

        raise ProviderError(
            f"{self.name} failed after {self.max_attempts} attempts: {last_error}",
            provider=self.name,
            retryable=True,
        )

    async def request_bytes(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        """Fetch raw bytes (generated images usually arrive as a URL)."""
        self.guard_offline()
        try:
            response = await self._http().get(url, headers=dict(headers or {}))
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"{self.name} could not fetch result bytes: {exc}",
                provider=self.name,
                retryable=is_retryable_exception(exc),
            ) from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name} result fetch returned {response.status_code}",
                provider=self.name,
                retryable=response.status_code in RETRYABLE_STATUS,
                status=response.status_code,
            )
        return response.content


def _short_error(response: httpx.Response) -> str:
    """Extract a short, non-secret error message from a vendor response."""
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return response.text[:200] if response.text else "(no body)"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)[:200]
        if error:
            return str(error)[:200]
        if "message" in payload:
            return str(payload["message"])[:200]
    return json.dumps(payload)[:200]


def require_positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_MAX_S",
    "BudgetExceeded",
    "BudgetLedger",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_TIMEOUT_S",
    "HttpProvider",
    "RETRYABLE_STATUS",
    "decode_b64",
    "encode_b64",
    "image_dimensions",
    "is_retryable_exception",
    "require_positive",
    "sniff_image_type",
]
