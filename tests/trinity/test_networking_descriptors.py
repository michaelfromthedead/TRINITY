"""
Tests for networking descriptors: NetworkedDescriptor, InterpolatedDescriptor,
PredictedDescriptor, ThrottledNetworkDescriptor.

Verifies:
- Network queue behavior and authority validation
- Interpolation modes and snapshot buffering
- Prediction history and rollback
- Network update throttling and flush
"""
import pytest
import time
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from trinity.descriptors.networking import (
    NetworkedDescriptor,
    InterpolatedDescriptor,
    PredictedDescriptor,
    ThrottledNetworkDescriptor,
    get_network_queue,
    clear_network_queue,
    pop_network_updates,
)


class TestNetworkedDescriptor:
    """Test NetworkedDescriptor queues updates and validates authority."""

    def test_queues_update_on_change(self):
        """Value change should add entry to network queue."""
        class Foo:
            pos = NetworkedDescriptor(field_type=float)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 10.0
        f.pos = 20.0
        queue = get_network_queue(f)
        assert len(queue) == 2
        assert queue[0]["value"] == 10.0
        assert queue[1]["value"] == 20.0

    def test_no_queue_on_same_value(self):
        """Setting the same value should not queue an update."""
        class Foo:
            pos = NetworkedDescriptor(field_type=float)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 10.0
        clear_network_queue(f)
        f.pos = 10.0
        queue = get_network_queue(f)
        assert len(queue) == 0

    def test_invalid_authority_raises(self):
        """Invalid authority value should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid authority"):
            NetworkedDescriptor(field_type=float, authority="invalid")

    def test_valid_authorities_accepted(self):
        """Valid authority values should be accepted."""
        for auth in ["server", "client", "owner"]:
            desc = NetworkedDescriptor(field_type=float, authority=auth)
            assert desc._authority == auth

    def test_negative_update_frequency_raises(self):
        """Negative update_frequency should raise ValueError."""
        with pytest.raises(ValueError, match="update_frequency must be >= 0"):
            NetworkedDescriptor(field_type=float, update_frequency=-1)

    def test_unhashable_type_comparison(self):
        """Unhashable types should use identity comparison."""
        class Foo:
            data = NetworkedDescriptor(field_type=list)
        Foo.data.__set_name__(Foo, 'data')
        f = Foo()
        list1 = [1, 2, 3]
        f.data = list1
        clear_network_queue(f)
        f.data = list1  # Same identity
        queue = get_network_queue(f)
        assert len(queue) == 0

    def test_pop_network_updates(self):
        """pop_network_updates should return and clear queue."""
        class Foo:
            pos = NetworkedDescriptor(field_type=float)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 10.0
        f.pos = 20.0
        updates = pop_network_updates(f)
        assert len(updates) == 2
        assert len(get_network_queue(f)) == 0

    def test_priority_stored_in_queue(self):
        """Network priority should be stored in queued updates."""
        class Foo:
            pos = NetworkedDescriptor(field_type=float, priority=5)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 10.0
        queue = get_network_queue(f)
        assert queue[0]["priority"] == 5


class TestInterpolatedDescriptor:
    """Test InterpolatedDescriptor stores snapshots and supports interpolation modes."""

    def test_linear_interpolation(self):
        """Linear mode should interpolate between two snapshots."""
        class Foo:
            pos = InterpolatedDescriptor(field_type=float, mode="linear")
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 0.0
        f.pos = 10.0
        result = Foo.pos.get_interpolated(f, t=0.5)
        assert result == pytest.approx(5.0)

    def test_linear_interpolation_edges(self):
        """Linear interpolation at t=0 and t=1 should return endpoints."""
        class Foo:
            pos = InterpolatedDescriptor(field_type=float, mode="linear")
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 0.0
        f.pos = 10.0
        assert Foo.pos.get_interpolated(f, t=0.0) == pytest.approx(0.0)
        assert Foo.pos.get_interpolated(f, t=1.0) == pytest.approx(10.0)

    def test_hermite_interpolation(self):
        """Hermite mode should provide smooth interpolation."""
        class Foo:
            pos = InterpolatedDescriptor(field_type=float, mode="hermite")
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 0.0
        f.pos = 10.0
        # Hermite at t=0.5 should be close to midpoint
        result = Foo.pos.get_interpolated(f, t=0.5)
        assert isinstance(result, float)
        assert 4.0 < result < 6.0  # Should be near midpoint

    def test_hermite_interpolation_edges(self):
        """Hermite at t=0 and t=1 should return endpoints."""
        class Foo:
            pos = InterpolatedDescriptor(field_type=float, mode="hermite")
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 0.0
        f.pos = 10.0
        assert Foo.pos.get_interpolated(f, t=0.0) == pytest.approx(0.0)
        assert Foo.pos.get_interpolated(f, t=1.0) == pytest.approx(10.0)

    def test_invalid_mode_raises(self):
        """An invalid interpolation mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid interpolation mode"):
            InterpolatedDescriptor(field_type=float, mode="cubic_invalid")

    def test_get_returns_latest_value(self):
        """A regular get should return the most recent value set."""
        class Foo:
            pos = InterpolatedDescriptor(field_type=float, mode="linear")
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 7.5
        assert f.pos == pytest.approx(7.5)

    def test_buffer_limit_enforced(self):
        """Buffer should not exceed INTERPOLATION_BUFFER_SIZE."""
        class Foo:
            pos = InterpolatedDescriptor(field_type=float, mode="linear")
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        for i in range(10):
            f.pos = float(i)
        buf = getattr(f, '_interp_buffer_pos', [])
        assert len(buf) <= 3

    def test_empty_buffer_returns_none(self):
        """get_interpolated on empty buffer should return None."""
        class Foo:
            pos = InterpolatedDescriptor(field_type=float, mode="linear")
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        result = Foo.pos.get_interpolated(f, t=0.5)
        assert result is None

    def test_single_value_returns_that_value(self):
        """get_interpolated with only one value should return that value."""
        class Foo:
            pos = InterpolatedDescriptor(field_type=float, mode="linear")
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 42.0
        result = Foo.pos.get_interpolated(f, t=0.5)
        assert result == pytest.approx(42.0)

    def test_non_numeric_fallback(self):
        """Non-numeric values should fall back to latest value."""
        class Foo:
            name = InterpolatedDescriptor(field_type=str, mode="linear")
        Foo.name.__set_name__(Foo, 'name')
        f = Foo()
        f.name = "Alice"
        f.name = "Bob"
        result = Foo.name.get_interpolated(f, t=0.5)
        assert result == "Bob"  # Should return b, not interpolate


class TestPredictedDescriptor:
    """Test PredictedDescriptor maintains a history and supports rollback."""

    def test_history_accumulates(self):
        """Setting values should accumulate history entries."""
        class Foo:
            pos = PredictedDescriptor(field_type=float, max_history=10)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 1.0
        f.pos = 2.0
        f.pos = 3.0
        history = Foo.pos.get_history(f)
        assert len(history) == 3
        assert history == [1.0, 2.0, 3.0]

    def test_rollback(self):
        """Rollback should restore the previous value and trim history."""
        class Foo:
            pos = PredictedDescriptor(field_type=float, max_history=10)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 1.0
        f.pos = 2.0
        f.pos = 3.0
        result = Foo.pos.rollback(f, frames=1)
        assert result == pytest.approx(2.0)
        assert f.pos == pytest.approx(2.0)
        history = Foo.pos.get_history(f)
        assert len(history) == 2

    def test_rollback_multiple_frames(self):
        """Rollback multiple frames should work correctly."""
        class Foo:
            pos = PredictedDescriptor(field_type=float, max_history=10)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        for i in range(5):
            f.pos = float(i)
        result = Foo.pos.rollback(f, frames=3)
        assert result == pytest.approx(1.0)
        assert f.pos == pytest.approx(1.0)

    def test_max_history_enforced(self):
        """History should not exceed max_history entries."""
        class Foo:
            pos = PredictedDescriptor(field_type=float, max_history=3)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        for i in range(10):
            f.pos = float(i)
        history = Foo.pos.get_history(f)
        assert len(history) == 3
        assert history == [7.0, 8.0, 9.0]

    def test_invalid_max_history_raises(self):
        """max_history of 0 or negative should raise ValueError."""
        with pytest.raises(ValueError, match="max_history must be > 0"):
            PredictedDescriptor(field_type=float, max_history=0)
        with pytest.raises(ValueError, match="max_history must be > 0"):
            PredictedDescriptor(field_type=float, max_history=-1)

    def test_rollback_empty_history_returns_none(self):
        """Rollback with no history should return None."""
        class Foo:
            pos = PredictedDescriptor(field_type=float, max_history=10)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        result = Foo.pos.rollback(f)
        assert result is None

    def test_rollback_insufficient_history(self):
        """Rollback beyond history should return None."""
        class Foo:
            pos = PredictedDescriptor(field_type=float, max_history=10)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 1.0
        result = Foo.pos.rollback(f, frames=5)
        assert result is None


class TestThrottledNetworkDescriptor:
    """Test ThrottledNetworkDescriptor limits network update frequency."""

    def test_allows_first_update(self):
        """The first set should always be allowed through."""
        class Foo:
            pos = ThrottledNetworkDescriptor(field_type=float, max_updates_per_second=10.0)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 1.0
        assert f.pos == pytest.approx(1.0)
        assert not Foo.pos.has_pending(f)

    def test_throttles_rapid_updates(self):
        """Rapid updates within min_interval should be throttled."""
        class Foo:
            pos = ThrottledNetworkDescriptor(field_type=float, max_updates_per_second=10.0)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 1.0
        f.pos = 2.0  # Should be throttled (queued but not sent)
        f.pos = 3.0  # Should be throttled
        # The stored value should still update locally
        assert f.pos == pytest.approx(3.0)
        # But network queue should reflect throttling
        assert Foo.pos.has_pending(f)

    def test_flush(self):
        """Flush should force-send any pending throttled update."""
        class Foo:
            pos = ThrottledNetworkDescriptor(field_type=float, max_updates_per_second=10.0)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 1.0
        f.pos = 2.0
        assert Foo.pos.has_pending(f)
        Foo.pos.flush(f)
        assert not Foo.pos.has_pending(f)

    def test_allows_after_interval(self):
        """After waiting the interval, updates should be allowed again."""
        class Foo:
            pos = ThrottledNetworkDescriptor(field_type=float, max_updates_per_second=20.0)
        Foo.pos.__set_name__(Foo, 'pos')
        f = Foo()
        f.pos = 1.0
        time.sleep(0.06)
        f.pos = 2.0
        # After the interval, the update should go through without throttling
        assert not Foo.pos.has_pending(f)

    def test_invalid_max_updates_per_second_raises(self):
        """max_updates_per_second <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_updates_per_second must be > 0"):
            ThrottledNetworkDescriptor(field_type=float, max_updates_per_second=0.0)
        with pytest.raises(ValueError, match="max_updates_per_second must be > 0"):
            ThrottledNetworkDescriptor(field_type=float, max_updates_per_second=-1.0)

    def test_min_interval_calculated_correctly(self):
        """min_interval should be 1/max_updates_per_second."""
        desc = ThrottledNetworkDescriptor(field_type=float, max_updates_per_second=20.0)
        assert desc._min_interval == pytest.approx(0.05)

    def test_multiple_fields_independent_throttling(self):
        """Different fields should have independent throttle timers."""
        class Foo:
            x = ThrottledNetworkDescriptor(field_type=float, max_updates_per_second=10.0)
            y = ThrottledNetworkDescriptor(field_type=float, max_updates_per_second=10.0)
        Foo.x.__set_name__(Foo, 'x')
        Foo.y.__set_name__(Foo, 'y')
        f = Foo()
        f.x = 1.0
        f.y = 1.0
        f.x = 2.0  # Should be throttled
        f.y = 2.0  # Should be throttled
        # Both should have pending updates
        assert Foo.x.has_pending(f)
        assert Foo.y.has_pending(f)
