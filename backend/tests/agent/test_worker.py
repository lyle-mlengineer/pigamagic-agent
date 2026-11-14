"""Step 12 gates: lease, crash recovery, and no double processing.

The two assertions steps.md names explicitly:

* killing the worker mid-job returns the job to `queued` with `attempt += 1`
* no job is processed twice concurrently

Time is injected, so lease expiry is tested without sleeping.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.agent.worker import (
    ClaimLost,
    InMemoryJobQueue,
    Job,
    JobNotFound,
    QueueError,
    Worker,
    enqueue_run_job,
    lease_token_for,
)

BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    """A controllable clock, so lease expiry needs no real waiting."""

    def __init__(self) -> None:
        self.current = BASE

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


def _job(job_id: str = "job_1", **payload) -> Job:
    return Job(job_id=job_id, kind="generate_run", payload=dict(payload))


# --------------------------------------------------------------------------
# Enqueue and claim
# --------------------------------------------------------------------------


def test_enqueue_is_idempotent_by_job_id():
    queue = InMemoryJobQueue()
    first = queue.enqueue(_job("job_1"))
    second = queue.enqueue(_job("job_1"))
    assert first is second
    assert len(queue.all_jobs()) == 1


def test_enqueue_rejects_a_non_queued_job():
    queue = InMemoryJobQueue()
    job = _job("job_1")
    job.status = "done"
    with pytest.raises(QueueError):
        queue.enqueue(job)


def test_claim_returns_none_on_an_empty_queue():
    queue = InMemoryJobQueue()
    assert queue.claim(worker_id="w1", lease_seconds=30) is None


def test_claim_marks_the_job_and_issues_a_lease():
    queue = InMemoryJobQueue()
    queue.enqueue(_job("job_1"))
    claimed = queue.claim(worker_id="w1", lease_seconds=30)

    assert claimed is not None
    assert claimed.status == "claimed"
    assert claimed.attempt == 0
    assert claimed.lease_token
    assert claimed.payload["worker_id"] == "w1"


def test_claim_is_oldest_first():
    clock = Clock()
    queue = InMemoryJobQueue(now=clock)
    queue.enqueue(_job("job_old"))
    clock.advance(1)
    queue.enqueue(_job("job_new"))

    assert queue.claim(worker_id="w1", lease_seconds=30).job_id == "job_old"


def test_second_claim_does_not_take_a_live_lease():
    queue = InMemoryJobQueue()
    queue.enqueue(_job("job_1"))

    first = queue.claim(worker_id="w1", lease_seconds=30)
    second = queue.claim(worker_id="w2", lease_seconds=30)

    assert first is not None
    # The gate: no job is processed twice concurrently.
    assert second is None


def test_no_job_is_processed_twice_concurrently():
    """Many workers racing one job: exactly one wins."""
    queue = InMemoryJobQueue()
    queue.enqueue(_job("job_1"))

    winners = [
        queue.claim(worker_id=f"w{index}", lease_seconds=30)
        for index in range(20)
    ]
    assert sum(1 for winner in winners if winner is not None) == 1


# --------------------------------------------------------------------------
# Ack / nack and fencing
# --------------------------------------------------------------------------


def test_ack_completes_the_job():
    queue = InMemoryJobQueue()
    queue.enqueue(_job("job_1"))
    claimed = queue.claim(worker_id="w1", lease_seconds=30)

    done = queue.ack("job_1", lease_token=claimed.lease_token, result={"photos": 3})
    assert done.status == "done"
    assert done.terminal
    assert done.result == {"photos": 3}
    assert done.finished_at is not None


def test_ack_with_a_stale_token_is_rejected():
    queue = InMemoryJobQueue()
    queue.enqueue(_job("job_1"))
    claimed = queue.claim(worker_id="w1", lease_seconds=30)

    with pytest.raises(ClaimLost):
        queue.ack("job_1", lease_token="lease_forged")

    # And the job is untouched.
    assert queue.get("job_1").status == "claimed"
    assert claimed.lease_token == queue.get("job_1").lease_token


def test_nack_requeues_and_bumps_the_attempt():
    queue = InMemoryJobQueue()
    queue.enqueue(_job("job_1"))
    claimed = queue.claim(worker_id="w1", lease_seconds=30)

    job = queue.nack("job_1", lease_token=claimed.lease_token, error="boom")
    assert job.status == "queued"
    assert job.attempt == 1
    assert job.error == "boom"


def test_nack_without_retry_fails_the_job():
    queue = InMemoryJobQueue()
    queue.enqueue(_job("job_1"))
    claimed = queue.claim(worker_id="w1", lease_seconds=30)

    job = queue.nack("job_1", lease_token=claimed.lease_token, error="fatal", retry=False)
    assert job.status == "failed"
    assert job.terminal


def test_job_goes_dead_after_max_attempts():
    queue = InMemoryJobQueue(max_attempts=2)
    queue.enqueue(_job("job_1"))

    for _ in range(2):
        claimed = queue.claim(worker_id="w1", lease_seconds=30)
        queue.nack("job_1", lease_token=claimed.lease_token, error="boom")

    job = queue.get("job_1")
    assert job.status == "dead"
    assert job.terminal


def test_unknown_job_raises():
    queue = InMemoryJobQueue()
    with pytest.raises(JobNotFound):
        queue.ack("nope", lease_token="x")


# --------------------------------------------------------------------------
# The crash-recovery gate from steps.md
# --------------------------------------------------------------------------


def test_crashed_worker_returns_the_job_with_a_bumped_attempt():
    """The gate: kill the worker mid-job -> back to `queued`, `attempt += 1`."""
    clock = Clock()
    queue = InMemoryJobQueue(now=clock, max_attempts=5)
    queue.enqueue(_job("job_1"))

    # Worker claims the job, then "dies" without acking.
    claimed = queue.claim(worker_id="w1", lease_seconds=30)
    assert claimed.status == "claimed"

    # Lease still live: not recovered yet.
    assert queue.release_expired() == []
    assert queue.get("job_1").status == "claimed"

    # Lease lapses.
    clock.advance(31)
    recovered = queue.release_expired()

    assert len(recovered) == 1
    job = queue.get("job_1")
    assert job.status == "queued"
    assert job.attempt == 1
    assert "lease expired" in job.error
    assert job.lease_token == ""

    # And it is claimable again.
    again = queue.claim(worker_id="w2", lease_seconds=30)
    assert again is not None
    assert again.job_id == "job_1"
    assert again.attempt == 1


def test_expired_lease_is_recovered_automatically_on_the_next_claim():
    clock = Clock()
    queue = InMemoryJobQueue(now=clock, max_attempts=5)
    queue.enqueue(_job("job_1"))

    queue.claim(worker_id="w1", lease_seconds=30)
    clock.advance(31)

    # A new worker claims without anyone explicitly calling release_expired.
    reclaimed = queue.claim(worker_id="w2", lease_seconds=30)
    assert reclaimed is not None
    assert reclaimed.attempt == 1


def test_stale_worker_cannot_ack_after_its_lease_expired():
    """Fencing: the original worker must not clobber the new owner's work."""
    clock = Clock()
    queue = InMemoryJobQueue(now=clock, max_attempts=5)
    queue.enqueue(_job("job_1"))

    original = queue.claim(worker_id="w1", lease_seconds=30)
    original_token = original.lease_token

    clock.advance(31)
    replacement = queue.claim(worker_id="w2", lease_seconds=30)
    assert replacement is not None
    assert replacement.lease_token != original_token

    # The slow worker finally finishes and tries to record its result.
    with pytest.raises(ClaimLost):
        queue.ack("job_1", lease_token=original_token, result={"photos": 1})

    # The current owner still can.
    queue.ack("job_1", lease_token=replacement.lease_token, result={"photos": 9})
    assert queue.get("job_1").result == {"photos": 9}


def test_a_crash_does_not_lose_the_job_after_max_attempts():
    clock = Clock()
    queue = InMemoryJobQueue(now=clock, max_attempts=1)
    queue.enqueue(_job("job_1"))

    queue.claim(worker_id="w1", lease_seconds=30)
    clock.advance(31)
    queue.release_expired()

    # Out of attempts: parked as dead, never silently dropped.
    job = queue.get("job_1")
    assert job.status == "dead"
    assert job.terminal


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_runs_a_job_and_acks():
    queue = InMemoryJobQueue()
    queue.enqueue(_job("job_1"))
    seen: list[str] = []

    async def run_job(job: Job) -> dict[str, object]:
        seen.append(job.job_id)
        return {"ok": True}

    worker = Worker(queue, run_job, worker_id="w1")
    stats = await worker.run_forever()

    assert seen == ["job_1"]
    assert stats.claimed == 1
    assert stats.succeeded == 1
    job = queue.get("job_1")
    assert job.status == "done"
    assert job.result == {"ok": True}


@pytest.mark.asyncio
async def test_worker_nacks_a_raising_job_and_keeps_going():
    queue = InMemoryJobQueue(max_attempts=5)
    queue.enqueue(_job("job_1"))

    async def run_job(job: Job) -> dict[str, object]:
        raise RuntimeError("provider exploded")

    worker = Worker(queue, run_job, worker_id="w1")
    await worker.run_forever(max_iterations=1)

    job = queue.get("job_1")
    assert job.status == "queued"
    assert job.attempt == 1
    assert "provider exploded" in job.error
    assert worker.stats.failed == 1


@pytest.mark.asyncio
async def test_worker_discards_result_when_it_loses_the_lease():
    clock = Clock()
    queue = InMemoryJobQueue(now=clock, max_attempts=5)
    queue.enqueue(_job("job_1"))

    async def slow_job(job: Job) -> dict[str, object]:
        # Simulate a stall long enough for the lease to lapse and a new worker
        # to take over.
        clock.advance(60)
        queue.release_expired()
        queue.claim(worker_id="w2", lease_seconds=30)
        return {"photos": 1}

    worker = Worker(queue, slow_job, worker_id="w1")
    await worker.run_forever(max_iterations=1)

    # The slow worker's result was discarded, not written.
    assert worker.stats.lost == 1
    assert queue.get("job_1").result == {}
    assert queue.get("job_1").status == "claimed"


@pytest.mark.asyncio
async def test_worker_processes_each_job_exactly_once():
    queue = InMemoryJobQueue()
    for index in range(5):
        queue.enqueue(_job(f"job_{index}"))
    processed: list[str] = []

    async def run_job(job: Job) -> dict[str, object]:
        processed.append(job.job_id)
        return {}

    worker = Worker(queue, run_job, worker_id="w1")
    await worker.run_forever()

    assert sorted(processed) == [f"job_{i}" for i in range(5)]
    assert len(processed) == len(set(processed))
    assert all(job.status == "done" for job in queue.all_jobs())


@pytest.mark.asyncio
async def test_two_workers_never_run_the_same_job():
    queue = InMemoryJobQueue()
    for index in range(4):
        queue.enqueue(_job(f"job_{index}"))
    processed: list[str] = []

    async def run_job(job: Job) -> dict[str, object]:
        processed.append(job.job_id)
        await asyncio.sleep(0)
        return {}

    workers = [Worker(queue, run_job, worker_id=f"w{i}") for i in range(3)]
    await asyncio.gather(*(worker.run_forever() for worker in workers))

    assert sorted(processed) == [f"job_{i}" for i in range(4)]
    assert len(processed) == len(set(processed))


@pytest.mark.asyncio
async def test_worker_reports_stats():
    queue = InMemoryJobQueue(max_attempts=5)
    queue.enqueue(_job("job_ok"))
    queue.enqueue(_job("job_bad"))

    async def run_job(job: Job) -> dict[str, object]:
        if job.job_id == "job_bad":
            raise ValueError("nope")
        return {}

    # One iteration per claim: the good job succeeds, the bad one is retried
    # (that retry loop is bounded by max_attempts and covered separately).
    worker = Worker(queue, run_job, worker_id="w1")
    stats = await worker.run_forever(max_iterations=2)

    assert stats.claimed == 2
    assert stats.succeeded == 1
    assert stats.failed == 1
    assert stats.as_dict() == {"claimed": 2, "succeeded": 1, "failed": 1, "lost": 0}


@pytest.mark.asyncio
async def test_a_failing_job_is_retried_until_it_dies():
    queue = InMemoryJobQueue(max_attempts=3)
    queue.enqueue(_job("job_bad"))

    async def run_job(job: Job) -> dict[str, object]:
        raise ValueError("always fails")

    worker = Worker(queue, run_job, worker_id="w1")
    await worker.run_forever()

    job = queue.get("job_bad")
    assert job.status == "dead"
    assert job.attempt == 3
    assert worker.stats.failed == 3


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_enqueue_run_job_builds_a_run_payload():
    queue = InMemoryJobQueue()
    job = enqueue_run_job(
        queue,
        job_id="job_run_1",
        shoot_id="shoot_1",
        account_id="acct_1",
        brief_name="wedding",
        shot_count=12,
    )
    assert job.kind == "generate_run"
    assert job.payload["shoot_id"] == "shoot_1"
    assert job.payload["shot_count"] == 12


def test_lease_token_changes_per_claim():
    assert lease_token_for("job_1", 0, 1) != lease_token_for("job_1", 0, 2)
    assert lease_token_for("job_1", 0, 1) != lease_token_for("job_1", 1, 1)
    # Deterministic, so a token can be recomputed for an audit.
    assert lease_token_for("job_1", 0, 1) == lease_token_for("job_1", 0, 1)


def test_counts_and_ordering():
    clock = Clock()
    queue = InMemoryJobQueue(now=clock)
    queue.enqueue(_job("job_b"))
    clock.advance(1)
    queue.enqueue(_job("job_a"))
    # FIFO by enqueue time, not by id.
    assert [job.job_id for job in queue.all_jobs()] == ["job_b", "job_a"]
    assert queue.counts() == {"queued": 2}


def test_retry_backoff_delays_visibility():
    clock = Clock()
    queue = InMemoryJobQueue(now=clock, max_attempts=5, retry_backoff_seconds=10)
    queue.enqueue(_job("job_1"))

    claimed = queue.claim(worker_id="w1", lease_seconds=30)
    queue.nack("job_1", lease_token=claimed.lease_token, error="boom")

    # Queued, but not yet claimable: a drain loop must not hammer it.
    assert queue.get("job_1").status == "queued"
    assert queue.claim(worker_id="w1", lease_seconds=30) is None

    clock.advance(11)
    assert queue.claim(worker_id="w1", lease_seconds=30) is not None
