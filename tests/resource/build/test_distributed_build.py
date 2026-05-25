"""Tests for the distributed build coordinator."""
from engine.resource.build.distributed_build import (
    BuildJob,
    BuildWorker,
    DistributedBuildCoordinator,
    JobState,
)


class TestBuildWorker:
    def test_availability(self) -> None:
        w = BuildWorker(worker_id="w1", capacity=2, current_jobs=0)
        assert w.is_available is True
        w.current_jobs = 2
        assert w.is_available is False


class TestDistributedBuildCoordinator:
    def test_submit_job(self) -> None:
        c = DistributedBuildCoordinator()
        job = c.submit_job("tex.png")
        assert job.state == JobState.PENDING
        assert job.job_id == 1

    def test_sequential_job_ids(self) -> None:
        c = DistributedBuildCoordinator()
        j1 = c.submit_job("a")
        j2 = c.submit_job("b")
        assert j2.job_id == j1.job_id + 1

    def test_assign_jobs(self) -> None:
        c = DistributedBuildCoordinator()
        c.add_worker(BuildWorker("w1", capacity=2))
        c.submit_job("a.png")
        c.submit_job("b.png")
        assigned = c.assign_jobs()
        assert assigned == 2
        assert c.get_job(1).state == JobState.ASSIGNED
        assert c.get_job(1).worker_id == "w1"

    def test_no_workers_no_assignment(self) -> None:
        c = DistributedBuildCoordinator()
        c.submit_job("a.png")
        assert c.assign_jobs() == 0

    def test_complete_job(self) -> None:
        c = DistributedBuildCoordinator()
        c.add_worker(BuildWorker("w1", capacity=1))
        job = c.submit_job("x.bin")
        c.assign_jobs()
        c.start_job(job.job_id)
        assert c.get_job(job.job_id).state == JobState.RUNNING
        c.complete_job(job.job_id, result=b"cooked")
        assert c.get_job(job.job_id).state == JobState.COMPLETE
        assert c.get_job(job.job_id).result == b"cooked"

    def test_fail_job(self) -> None:
        c = DistributedBuildCoordinator()
        c.add_worker(BuildWorker("w1", capacity=1))
        job = c.submit_job("bad.bin")
        c.assign_jobs()
        c.fail_job(job.job_id, result="error msg")
        assert c.get_job(job.job_id).state == JobState.FAILED

    def test_get_progress(self) -> None:
        c = DistributedBuildCoordinator()
        c.add_worker(BuildWorker("w1", capacity=5))
        c.submit_job("a")
        c.submit_job("b")
        c.submit_job("c")
        c.assign_jobs()
        c.complete_job(1, "done")
        progress = c.get_progress()
        assert progress["complete"] == 1
        assert progress["assigned"] == 2
        assert progress["pending"] == 0

    def test_worker_capacity_respected(self) -> None:
        c = DistributedBuildCoordinator()
        c.add_worker(BuildWorker("w1", capacity=1))
        c.submit_job("a")
        c.submit_job("b")
        assigned = c.assign_jobs()
        assert assigned == 1
        # second job stays pending
        assert c.get_job(2).state == JobState.PENDING
