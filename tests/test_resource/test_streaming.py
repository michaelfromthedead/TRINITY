"""Integration tests for resource streaming subsystem."""

import pytest

from engine.resource.streaming import (
    StreamManager,
    StreamPriority,
    StreamRequest,
    StreamState,
    StreamType,
)


class TestStreamPriority:
    """Tests for StreamPriority enum."""

    def test_priority_ordering(self) -> None:
        """CRITICAL < HIGH < NORMAL < LOW < BACKGROUND."""
        assert StreamPriority.CRITICAL < StreamPriority.HIGH
        assert StreamPriority.HIGH < StreamPriority.NORMAL
        assert StreamPriority.NORMAL < StreamPriority.LOW
        assert StreamPriority.LOW < StreamPriority.BACKGROUND

    def test_priority_values(self) -> None:
        """Priority values are 0-4."""
        assert StreamPriority.CRITICAL == 0
        assert StreamPriority.HIGH == 1
        assert StreamPriority.NORMAL == 2
        assert StreamPriority.LOW == 3
        assert StreamPriority.BACKGROUND == 4


class TestStreamManager:
    """Tests for StreamManager."""

    def test_priority_queue_ordering(self) -> None:
        """Higher priority requests are processed first."""
        mgr = StreamManager(bytes_per_frame=100)
        low = mgr.request_stream("low.png", StreamPriority.LOW)
        low.bytes_total = 1_000_000
        high = mgr.request_stream("high.png", StreamPriority.HIGH)
        high.bytes_total = 1_000_000
        critical = mgr.request_stream("critical.png", StreamPriority.CRITICAL)
        critical.bytes_total = 1_000_000

        mgr.update()
        active = mgr.get_active_streams()
        active_ids = [r.request_id for r in active]

        assert critical.request_id in active_ids
        assert high.request_id in active_ids

    def test_cancellation_immediate(self) -> None:
        """Cancel immediately removes request from active."""
        mgr = StreamManager()
        req = mgr.request_stream("test.png", StreamPriority.NORMAL)
        req.bytes_total = 10_000_000

        mgr.update()
        assert req.state == StreamState.ACTIVE

        result = mgr.cancel(req.request_id)
        assert result is True
        assert req.state == StreamState.CANCELLED
        assert req.request_id not in [r.request_id for r in mgr.get_active_streams()]

    def test_cancellation_idempotent(self) -> None:
        """Cancelling twice returns False on second call."""
        mgr = StreamManager(bytes_per_frame=100)
        req = mgr.request_stream("test.png")
        req.bytes_total = 1_000_000
        mgr.update()
        assert mgr.cancel(req.request_id) is True
        assert mgr.cancel(req.request_id) is False

    def test_bandwidth_throttling(self) -> None:
        """Bandwidth limit prevents all bytes loading in single update."""
        mgr = StreamManager(bytes_per_frame=1024)
        req = mgr.request_stream("big.png")
        req.bytes_total = 10_000

        mgr.update()
        assert req.bytes_loaded == 1024
        assert req.state == StreamState.ACTIVE

        mgr.update()
        assert req.bytes_loaded == 2048

    def test_memory_pressure_callback(self) -> None:
        """Pressure callbacks are invoked on memory changes."""
        mgr = StreamManager(bytes_per_frame=1000, memory_limit=5000)
        notifications: list[tuple[int, int]] = []

        def on_pressure(used: int, limit: int) -> None:
            notifications.append((used, limit))

        mgr.add_pressure_callback(on_pressure)

        req = mgr.request_stream("test.png")
        req.bytes_total = 2000

        mgr.update()
        assert len(notifications) >= 1
        used, limit = notifications[-1]
        assert used == 1000
        assert limit == 5000

    def test_memory_release(self) -> None:
        """release_completed frees memory from finished streams."""
        mgr = StreamManager(bytes_per_frame=10_000)
        req = mgr.request_stream("small.png")
        req.bytes_total = 1000

        mgr.update()
        assert req.state == StreamState.COMPLETE

        used_before, _ = mgr.get_memory_usage()
        freed = mgr.release_completed()

        assert freed == 1000
        used_after, _ = mgr.get_memory_usage()
        assert used_after == used_before - 1000

    def test_stream_types(self) -> None:
        """All stream types are supported."""
        mgr = StreamManager()
        for stype in StreamType:
            req = mgr.request_stream(f"test_{stype.value}", stream_type=stype)
            assert req.stream_type == stype


class TestStreamRequest:
    """Tests for StreamRequest dataclass."""

    def test_unique_ids(self) -> None:
        """Each request gets a unique ID."""
        mgr = StreamManager()
        r1 = mgr.request_stream("a.png")
        r2 = mgr.request_stream("b.png")
        r3 = mgr.request_stream("c.png")
        assert len({r1.request_id, r2.request_id, r3.request_id}) == 3

    def test_default_state_pending(self) -> None:
        """New requests start in PENDING state."""
        mgr = StreamManager()
        req = mgr.request_stream("test.png")
        assert req.state == StreamState.PENDING

    def test_request_ordering(self) -> None:
        """Requests sort by (priority, id)."""
        mgr = StreamManager()
        r1 = mgr.request_stream("a.png", StreamPriority.LOW)
        r2 = mgr.request_stream("b.png", StreamPriority.HIGH)
        r3 = mgr.request_stream("c.png", StreamPriority.LOW)

        assert r2 < r1
        assert r2 < r3
        assert r1 < r3
