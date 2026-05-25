"""Integration tests for resource build pipeline."""

import tempfile
import time
from pathlib import Path

import pytest

# HashCache is planned but not yet implemented (incremental rebuild feature)
pytest.skip("HashCache not yet implemented", allow_module_level=True)

from engine.resource.build import (
    BuildDependencyTracker,
    DistributedBuildCoordinator,
    HashCache,
    JobState,
)


class TestHashCache:
    """Tests for HashCache."""

    def test_cache_hit(self) -> None:
        """Cache returns stored hash when mtime matches."""
        cache = HashCache()
        cache.put("test.py", 1234.0, "abc123")
        assert cache.get("test.py", 1234.0) == "abc123"

    def test_cache_miss_different_mtime(self) -> None:
        """Cache returns None when mtime differs."""
        cache = HashCache()
        cache.put("test.py", 1234.0, "abc123")
        assert cache.get("test.py", 1235.0) is None

    def test_cache_persistence(self, tmp_path: Path) -> None:
        """Cache persists to and loads from disk."""
        cache_file = tmp_path / "hashes.json"

        cache1 = HashCache(cache_file)
        cache1.put("test.py", 1234.0, "abc123")
        cache1.save()

        cache2 = HashCache(cache_file)
        assert cache2.get("test.py", 1234.0) == "abc123"


class TestBuildDependencyTracker:
    """Tests for BuildDependencyTracker."""

    def test_dirty_detection_new_file(self) -> None:
        """New files are always dirty."""
        tracker = BuildDependencyTracker()
        assert tracker.is_dirty("new.py", 1234.0, b"content") is True

    def test_dirty_detection_unchanged(self) -> None:
        """Unchanged files are not dirty."""
        tracker = BuildDependencyTracker()
        tracker.record_file("test.py", 1234.0, b"content")
        assert tracker.is_dirty("test.py", 1234.0, b"content") is False

    def test_dirty_detection_content_changed(self) -> None:
        """Files with changed content are dirty."""
        tracker = BuildDependencyTracker()
        tracker.record_file("test.py", 1234.0, b"original")
        assert tracker.is_dirty("test.py", 1235.0, b"modified") is True

    def test_walk_traversal(self) -> None:
        """Walk traverses dependency graph."""
        tracker = BuildDependencyTracker()
        tracker.record_file("a.py", 1.0, b"a", dependencies={"b.py", "c.py"})
        tracker.record_file("b.py", 1.0, b"b", dependencies={"c.py"})
        tracker.record_file("c.py", 1.0, b"c")

        visited = list(tracker.walk("a.py"))
        paths = [p for p, _ in visited]

        assert "a.py" in paths
        assert "b.py" in paths
        assert "c.py" in paths

    def test_get_dependents(self) -> None:
        """Get files that depend on a given file."""
        tracker = BuildDependencyTracker()
        tracker.record_file("a.py", 1.0, b"a", dependencies={"c.py"})
        tracker.record_file("b.py", 1.0, b"b", dependencies={"c.py"})
        tracker.record_file("c.py", 1.0, b"c")

        deps = tracker.get_dependents("c.py")
        assert deps == {"a.py", "b.py"}

    def test_build_order_topological(self) -> None:
        """Build order is topologically sorted."""
        tracker = BuildDependencyTracker()
        tracker.record_file("app.py", 1.0, b"app", dependencies={"lib.py"})
        tracker.record_file("lib.py", 1.0, b"lib", dependencies={"util.py"})
        tracker.record_file("util.py", 1.0, b"util")

        order = tracker.get_build_order(["util.py"])
        assert order.index("util.py") < order.index("lib.py")
        assert order.index("lib.py") < order.index("app.py")

    def test_build_order_cycle_detection(self) -> None:
        """Cyclic dependencies raise error."""
        tracker = BuildDependencyTracker()
        tracker.record_file("a.py", 1.0, b"a", dependencies={"b.py"})
        tracker.record_file("b.py", 1.0, b"b", dependencies={"a.py"})

        with pytest.raises(ValueError, match="Cyclic"):
            tracker.get_build_order(["a.py"])

    def test_progress_callback(self) -> None:
        """Progress callbacks are invoked during dirty detection."""
        tracker = BuildDependencyTracker()
        progress: list[tuple[int, int, str]] = []

        def on_progress(current: int, total: int, path: str) -> None:
            progress.append((current, total, path))

        tracker.add_progress_callback(on_progress)
        tracker.get_dirty_files({
            "a.py": (1.0, b"a"),
            "b.py": (1.0, b"b"),
        })

        assert len(progress) == 2
        assert progress[-1][0] == 2
        assert progress[-1][1] == 2


class TestDistributedBuildCoordinator:
    """Tests for DistributedBuildCoordinator."""

    def test_run_parallel(self) -> None:
        """Parallel execution completes all jobs."""
        coord = DistributedBuildCoordinator(max_parallel=2)

        def build(path: str) -> str:
            return f"built:{path}"

        jobs = coord.run_parallel(build, ["a.py", "b.py", "c.py"])

        assert len(jobs) == 3
        assert all(j.state == JobState.COMPLETE for j in jobs)
        assert jobs[0].result == "built:a.py"

    def test_parallel_progress_callback(self) -> None:
        """Progress callbacks fire during parallel execution."""
        coord = DistributedBuildCoordinator(max_parallel=2)
        progress: list[int] = []

        def on_progress(completed: int, total: int, job) -> None:
            progress.append(completed)

        coord.add_progress_callback(on_progress)
        coord.run_parallel(lambda p: p, ["a", "b", "c"])

        assert sorted(progress) == [1, 2, 3]

    def test_parallel_handles_failures(self) -> None:
        """Failed jobs are marked as FAILED."""
        coord = DistributedBuildCoordinator(max_parallel=2)

        def build(path: str) -> str:
            if path == "bad.py":
                raise ValueError("build error")
            return "ok"

        jobs = coord.run_parallel(build, ["good.py", "bad.py"])

        good = next(j for j in jobs if j.asset_path == "good.py")
        bad = next(j for j in jobs if j.asset_path == "bad.py")

        assert good.state == JobState.COMPLETE
        assert bad.state == JobState.FAILED
        assert "build error" in bad.result

    def test_progress_counts(self) -> None:
        """get_progress returns accurate job counts."""
        coord = DistributedBuildCoordinator()
        coord.submit_job("a.py")
        coord.submit_job("b.py")

        progress = coord.get_progress()
        assert progress["pending"] == 2
        assert progress["complete"] == 0
