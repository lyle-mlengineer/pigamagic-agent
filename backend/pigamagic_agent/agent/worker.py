"""Job queue and worker: claim, run, ack — with leases that survive a crash.

The API only enqueues; a separate process does the expensive work (design.md
§"Where the AI runs"). That split creates two failure modes this module exists
to fix:

1. **Worker crash mid-job.** Nobody acks, and without a lease the job is lost
   forever — a user's credits are held and nothing is ever produced. So a claim
   carries a deadline: an expired claim returns the job to `queued` with
   `attempt += 1` (§16).

2. **Duplicate processing.** A worker that stalls (GC pause, slow provider) can
   wake up after its lease expired and a second worker took the job. Both would
   then write results. So every claim carries a **lease token** derived from the
   attempt number, and `ack`/`nack` only succeed for the current token. A stale
   worker's ack is rejected rather than silently overwriting newer state — this
   is fencing, and it is the part naive queues get wrong.

The queue never decides *what* a job does: `run_job` is injected, so the worker
is testable without providers and the real body is `start_run` from `run.py`.

Clocks are injected (`now`) so lease expiry is testable without sleeping.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Literal, Protocol, Sequence

from .hashing import stable_digest

QUEUE_VERSION = "queue-v1"

JobStatus = Literal["queued", "claimed", "done", "failed", "dead"]


class QueueError(Exception):
    """Base class for queue problems."""


class ClaimLost(QueueError):
    """The caller no longer owns the job: its lease expired or was replaced.

    Raised instead of writing, so a stale worker cannot overwrite results that a
    newer owner already produced.
    """


class JobNotFound(QueueError):
    """No job with that id."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Job:
    """One unit of background work. `payload` is opaque to the queue."""

    job_id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = "queued"
    attempt: int = 0
    #: Monotonic count of claims ever issued, so a lease token is unique even
    #: across retries of the same attempt number.
    claims: int = 0
    lease_token: str = ""
    visible_at: datetime | None = None
    error: str = ""
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None
    #: Set once the job is terminal; a done/failed job is never re-queued.
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.status in ("done", "failed", "dead")

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "status": self.status,
            "attempt": self.attempt,
            "claims": self.claims,
            "visible_at": None if self.visible_at is None else self.visible_at.isoformat(),
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "finished_at": None if self.finished_at is None else self.finished_at.isoformat(),
            "result": dict(self.result),
        }


def lease_token_for(job_id: str, attempt: int, claims: int) -> str:
    """Fencing token for one claim. Changes on every claim, deterministic."""
    return "lease_" + stable_digest(QUEUE_VERSION, job_id, str(attempt), str(claims))[:20]


class JobQueue(Protocol):
    """Minimal queue surface. Redis/Postgres implementations must match it."""

    def enqueue(self, job: Job) -> Job: ...

    def claim(self, *, worker_id: str, lease_seconds: float) -> Job | None: ...

    def ack(self, job_id: str, *, lease_token: str, result: dict[str, Any] | None = None) -> Job: ...

    def nack(self, job_id: str, *, lease_token: str, error: str, retry: bool = True) -> Job: ...

    def get(self, job_id: str) -> Job: ...

    def release_expired(self) -> list[Job]: ...


class InMemoryJobQueue:
    """A correct-by-construction in-memory queue.

    `claim` is atomic with respect to other coroutines because it contains no
    `await`: the check and the state change happen in one uninterrupted step.
    That is what makes "no job is processed twice concurrently" true here, and
    it is the property the Postgres version must reproduce with
    `SELECT ... FOR UPDATE SKIP LOCKED`.
    """

    def __init__(
        self,
        *,
        now: Callable[[], datetime] | None = None,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._now = now or _utcnow
        self.max_attempts = max(1, max_attempts)
        #: Delay before a nacked job becomes claimable again. Without it a
        #: drain loop re-claims a failing job immediately, burning every attempt
        #: in milliseconds and denying a transient outage any chance to clear.
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    # -- writes --------------------------------------------------------------

    def enqueue(self, job: Job) -> Job:
        if job.status != "queued":
            raise QueueError(f"only queued jobs can be enqueued (got {job.status!r})")
        existing = self._jobs.get(job.job_id)
        if existing is not None:
            # Idempotent by job id: re-enqueueing the same id must not create a
            # second unit of work.
            return existing
        job.updated_at = self._now()
        self._jobs[job.job_id] = job
        return job

    def claim(self, *, worker_id: str, lease_seconds: float) -> Job | None:
        """Take the oldest visible job and lease it. Always leases *something*
        new, never a job another live lease holds.

        Deliberately contains no `await`, so with a single-threaded event loop
        the claim is atomic.
        """
        now = self._now()
        self.release_expired(now=now)

        candidates = [
            job
            for job in self._jobs.values()
            if job.status == "queued" and (job.visible_at is None or job.visible_at <= now)
        ]
        if not candidates:
            return None

        job = min(candidates, key=lambda item: (item.created_at, item.job_id))
        job.claims += 1
        job.status = "claimed"
        job.lease_token = lease_token_for(job.job_id, job.attempt, job.claims)
        job.visible_at = now + timedelta(seconds=max(1.0, lease_seconds))
        job.updated_at = now
        job.payload.setdefault("worker_id", worker_id)
        return job

    def ack(self, job_id: str, *, lease_token: str, result: dict[str, Any] | None = None) -> Job:
        job = self._require(job_id)
        self._require_lease(job, lease_token)
        now = self._now()
        job.status = "done"
        job.result = dict(result or {})
        job.lease_token = ""
        job.visible_at = None
        job.error = ""
        job.updated_at = now
        job.finished_at = now
        return job

    def nack(self, job_id: str, *, lease_token: str, error: str, retry: bool = True) -> Job:
        job = self._require(job_id)
        self._require_lease(job, lease_token)
        now = self._now()
        job.error = error
        job.lease_token = ""
        job.visible_at = None
        job.updated_at = now
        job.attempt += 1

        if not retry:
            job.status = "failed"
            job.finished_at = now
        elif job.attempt >= self.max_attempts:
            # Out of attempts: park it as `dead` rather than looping forever.
            job.status = "dead"
            job.finished_at = now
        else:
            job.status = "queued"
            if self.retry_backoff_seconds:
                job.visible_at = now + timedelta(seconds=self.retry_backoff_seconds)
        return job

    def release_expired(self, *, now: datetime | None = None) -> list[Job]:
        """Return jobs whose lease lapsed to `queued` with `attempt += 1`.

        This is the crash-recovery path (§16): a worker that died mid-job leaves
        a claimed job with no ack, and only expiry gets it back into the queue.
        """
        current = now or self._now()
        recovered: list[Job] = []
        for job in self._jobs.values():
            if job.status != "claimed":
                continue
            if job.visible_at is None or job.visible_at > current:
                continue
            job.attempt += 1
            job.lease_token = ""
            job.visible_at = None
            job.updated_at = current
            job.error = job.error or "lease expired; worker presumed crashed"
            if job.attempt >= self.max_attempts:
                job.status = "dead"
                job.finished_at = current
            else:
                job.status = "queued"
            recovered.append(job)
        return recovered

    # -- reads ---------------------------------------------------------------

    def get(self, job_id: str) -> Job:
        return self._require(job_id)

    def all_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda item: (item.created_at, item.job_id))

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for job in self._jobs.values():
            tally[job.status] = tally.get(job.status, 0) + 1
        return tally

    # -- internals -----------------------------------------------------------

    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(f"no job {job_id!r}")
        return job

    def _require_lease(self, job: Job, lease_token: str) -> None:
        if job.status != "claimed":
            raise ClaimLost(
                f"job {job.job_id!r} is {job.status!r}, not claimed; "
                f"this worker no longer owns it."
            )
        if not lease_token or job.lease_token != lease_token:
            raise ClaimLost(
                f"lease token for job {job.job_id!r} is stale; another worker "
                f"owns this job now. Discard this attempt's result."
            )


#: What a worker does with a claimed job. Returns a result dict for the ack.
JobRunner = Callable[[Job], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class WorkerStats:
    claimed: int = 0
    succeeded: int = 0
    failed: int = 0
    lost: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "claimed": self.claimed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "lost": self.lost,
        }


class Worker:
    """Claims one job at a time and runs it under a lease.

    Deliberately single-job-at-a-time: concurrency is a deployment concern
    (run more workers), and serial execution keeps "no job processed twice"
    trivially true within one worker.
    """

    def __init__(
        self,
        queue: InMemoryJobQueue,
        run_job: JobRunner,
        *,
        worker_id: str = "worker-1",
        lease_seconds: float = 30.0,
        stop_when_idle: bool = True,
    ) -> None:
        self.queue = queue
        self.run_job = run_job
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.stop_when_idle = stop_when_idle
        self.stats = WorkerStats()

    async def _claim_and_run_once(self) -> bool:
        job = self.queue.claim(worker_id=self.worker_id, lease_seconds=self.lease_seconds)
        if job is None:
            return False
        self.stats.claimed += 1
        token = job.lease_token

        try:
            result = await self.run_job(job)
        except Exception as exc:  # noqa: BLE001 - a job failure must not kill the worker
            try:
                self.queue.nack(job.job_id, lease_token=token, error=f"{type(exc).__name__}: {exc}")
                self.stats.failed += 1
            except ClaimLost:
                self.stats.lost += 1
            return True

        try:
            self.queue.ack(job.job_id, lease_token=token, result=result or {})
            self.stats.succeeded += 1
        except ClaimLost:
            # The lease expired and someone else owns the job. Discarding is the
            # only correct move: writing would clobber newer state.
            self.stats.lost += 1
        return True

    async def run_forever(self, *, poll_seconds: float = 0.05, max_iterations: int | None = None) -> WorkerStats:
        """Drain the queue. Stops when idle (if configured) or at max_iterations."""
        iterations = 0
        while True:
            if max_iterations is not None and iterations >= max_iterations:
                break
            iterations += 1
            worked = await self._claim_and_run_once()
            if not worked:
                if self.stop_when_idle:
                    break
                await asyncio.sleep(poll_seconds)
        return self.stats


def enqueue_run_job(
    queue: InMemoryJobQueue,
    *,
    job_id: str,
    shoot_id: str,
    account_id: str,
    brief_name: str = "",
    shot_count: int = 0,
    extra: dict[str, Any] | None = None,
) -> Job:
    """Convenience constructor for the job the API enqueues on POST /generate."""
    payload: dict[str, Any] = {
        "shoot_id": shoot_id,
        "account_id": account_id,
        "brief_name": brief_name,
        "shot_count": shot_count,
    }
    payload.update(extra or {})
    return queue.enqueue(Job(job_id=job_id, kind="generate_run", payload=payload))


__all__ = [
    "ClaimLost",
    "InMemoryJobQueue",
    "Job",
    "JobNotFound",
    "JobQueue",
    "JobRunner",
    "JobStatus",
    "QUEUE_VERSION",
    "QueueError",
    "Worker",
    "WorkerStats",
    "enqueue_run_job",
    "lease_token_for",
]
