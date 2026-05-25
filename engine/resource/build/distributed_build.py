"""Distributed build coordinator — manages build job distribution."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class JobState(Enum):
    """State of a build job."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(slots=True)
class BuildJob:
    """A single build job."""

    job_id: int
    asset_path: str
    state: JobState = JobState.PENDING
    worker_id: Optional[str] = None
    result: Any = None


@dataclass(slots=True)
class BuildWorker:
    """A build worker node."""

    worker_id: str
    capacity: int
    current_jobs: int = 0

    @property
    def is_available(self) -> bool:
        return self.current_jobs < self.capacity


class DistributedBuildCoordinator:
    """Coordinates build job distribution across workers."""

    __slots__ = ("_workers", "_jobs", "_next_job_id")

    def __init__(self) -> None:
        self._workers: dict[str, BuildWorker] = {}
        self._jobs: dict[int, BuildJob] = {}
        self._next_job_id: int = 1

    def add_worker(self, worker: BuildWorker) -> None:
        """Register a worker."""
        self._workers[worker.worker_id] = worker

    def submit_job(self, asset_path: str) -> BuildJob:
        """Submit a new build job."""
        job = BuildJob(job_id=self._next_job_id, asset_path=asset_path)
        self._jobs[job.job_id] = job
        self._next_job_id += 1
        return job

    def assign_jobs(self) -> int:
        """Assign pending jobs to available workers. Returns count assigned."""
        assigned = 0
        pending = [j for j in self._jobs.values() if j.state == JobState.PENDING]
        available = [w for w in self._workers.values() if w.is_available]

        worker_idx = 0
        for job in pending:
            if worker_idx >= len(available):
                break
            worker = available[worker_idx]
            job.state = JobState.ASSIGNED
            job.worker_id = worker.worker_id
            worker.current_jobs += 1
            assigned += 1
            if not worker.is_available:
                worker_idx += 1

        return assigned

    def start_job(self, job_id: int) -> None:
        """Mark a job as running."""
        job = self._jobs[job_id]
        job.state = JobState.RUNNING

    def complete_job(self, job_id: int, result: Any) -> None:
        """Mark a job as complete with a result."""
        job = self._jobs[job_id]
        job.state = JobState.COMPLETE
        job.result = result
        if job.worker_id and job.worker_id in self._workers:
            self._workers[job.worker_id].current_jobs -= 1

    def fail_job(self, job_id: int, result: Any = None) -> None:
        """Mark a job as failed."""
        job = self._jobs[job_id]
        job.state = JobState.FAILED
        job.result = result
        if job.worker_id and job.worker_id in self._workers:
            self._workers[job.worker_id].current_jobs -= 1

    def get_progress(self) -> dict[str, int]:
        """Return counts per job state."""
        counts: dict[str, int] = {s.value: 0 for s in JobState}
        for job in self._jobs.values():
            counts[job.state.value] += 1
        return counts

    def get_job(self, job_id: int) -> Optional[BuildJob]:
        return self._jobs.get(job_id)
