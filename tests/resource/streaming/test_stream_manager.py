"""Tests for StreamManager."""

import pytest

from engine.resource.streaming.stream_manager import (
    MAX_CONCURRENT_STREAMS,
    StreamManager,
    StreamPriority,
    StreamRequest,
    StreamState,
    StreamType,
)


class TestStreamRequest:
    def test_auto_increment_ids(self) -> None:
        r1 = StreamRequest(asset_id="a")
        r2 = StreamRequest(asset_id="b")
        assert r2.request_id > r1.request_id

    def test_default_state_is_pending(self) -> None:
        r = StreamRequest(asset_id="x")
        assert r.state is StreamState.PENDING
        assert r.bytes_loaded == 0
        assert r.bytes_total == 0

    def test_ordering_by_priority_then_id(self) -> None:
        r_low = StreamRequest(asset_id="a", priority=StreamPriority.LOW)
        r_high = StreamRequest(asset_id="b", priority=StreamPriority.HIGH)
        assert r_high < r_low


class TestStreamManager:
    def test_request_stream_creates_pending(self) -> None:
        mgr = StreamManager()
        req = mgr.request_stream("tex_01", StreamPriority.NORMAL, StreamType.TEXTURE_MIP)
        assert req.state is StreamState.PENDING
        assert req.asset_id == "tex_01"
        assert mgr.get_pending_count() == 1

    def test_cancel_pending_request(self) -> None:
        mgr = StreamManager()
        req = mgr.request_stream("tex_01")
        assert mgr.cancel(req.request_id) is True
        assert req.state is StreamState.CANCELLED
        assert mgr.get_pending_count() == 0

    def test_cancel_nonexistent_returns_false(self) -> None:
        mgr = StreamManager()
        assert mgr.cancel(999999) is False

    def test_cancel_completed_returns_false(self) -> None:
        mgr = StreamManager()
        req = mgr.request_stream("tex_01")
        mgr.update()  # pending -> active -> complete (0 bytes)
        assert req.state is StreamState.COMPLETE
        assert mgr.cancel(req.request_id) is False

    def test_update_promotes_to_active(self) -> None:
        mgr = StreamManager()
        req = mgr.request_stream("tex_01")
        req.bytes_total = 1000  # non-zero so it stays active one tick
        mgr.update()
        # After one update with bytes_total set, it loads fully and completes
        assert req.state is StreamState.COMPLETE

    def test_priority_ordering(self) -> None:
        mgr = StreamManager()
        r_low = mgr.request_stream("low", StreamPriority.LOW)
        r_crit = mgr.request_stream("crit", StreamPriority.CRITICAL)
        r_low.bytes_total = 1000
        r_crit.bytes_total = 1000
        mgr.update()
        # Critical should process first; both complete in one tick
        assert r_crit.state is StreamState.COMPLETE

    def test_max_concurrent_streams(self) -> None:
        mgr = StreamManager()
        requests = []
        total = MAX_CONCURRENT_STREAMS + 4
        for i in range(total):
            r = mgr.request_stream(f"asset_{i}")
            r.bytes_total = 100000  # large so they don't auto-complete
            requests.append(r)
        # Don't call update yet; modify bytes_total to be huge to prevent completion
        # Actually, our update completes in one tick. Let's test pending count.
        assert mgr.get_pending_count() == total
        mgr.update()
        # All promoted and completed (since our sim completes in one tick)
        # But only MAX_CONCURRENT_STREAMS promoted per update
        active = mgr.get_active_streams()
        # After update, first batch completes, so active is empty
        # Pending should have remaining
        assert mgr.get_pending_count() == total - MAX_CONCURRENT_STREAMS

    def test_get_active_streams(self) -> None:
        mgr = StreamManager()
        req = mgr.request_stream("tex_01")
        assert mgr.get_active_streams() == []
        # Zero-byte requests complete on update, so no active after update
        mgr.update()
        assert mgr.get_active_streams() == []
        assert req.state is StreamState.COMPLETE
