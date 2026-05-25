"""
Tests for debug descriptors: ProfiledDescriptor, LoggedDescriptor, WatchedDescriptor.

Verifies:
- Timing recording on get/set
- Statistics aggregation
- Log capture for get and set operations
- Watch callback triggering with conditions
"""
import pytest
import time
from trinity.descriptors.debug import ProfiledDescriptor, LoggedDescriptor, WatchedDescriptor


class TestProfiledDescriptor:
    """Test ProfiledDescriptor records timing information for get/set operations."""

    def test_records_timing(self):
        """Get and set operations should record timing samples."""
        class Foo:
            value = ProfiledDescriptor(field_type=int, max_samples=100)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 42
        _ = f.value
        stats = Foo.value.get_stats(f)
        assert stats["set"]["count"] == 1
        assert stats["get"]["count"] == 1
        assert stats["set"]["avg_ns"] > 0
        assert stats["get"]["avg_ns"] > 0

    def test_get_stats(self):
        """Stats should include count, min, max, and average timings."""
        class Foo:
            value = ProfiledDescriptor(field_type=int, max_samples=100)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        for i in range(5):
            f.value = i
            _ = f.value
        stats = Foo.value.get_stats(f)
        assert stats["set"]["count"] == 5
        assert stats["get"]["count"] == 5
        assert "avg_ns" in stats["set"]
        assert "avg_ns" in stats["get"]

    def test_max_samples_enforced(self):
        """Recorded samples should not exceed max_samples."""
        class Foo:
            value = ProfiledDescriptor(field_type=int, max_samples=3)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        for i in range(10):
            f.value = i
        stats = Foo.value.get_stats(f)
        assert stats["set"]["count"] == 3  # Should be capped at max_samples

    def test_metadata(self):
        """Metadata should include profiling configuration."""
        class Foo:
            value = ProfiledDescriptor(field_type=int, max_samples=50)
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["descriptor_id"] == "profiled"
        assert meta["max_samples"] == 50

    def test_empty_stats_before_access(self):
        """Stats should handle zero samples gracefully."""
        class Foo:
            value = ProfiledDescriptor(field_type=int, max_samples=10)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        stats = Foo.value.get_stats(f)
        assert stats["get"]["count"] == 0
        assert stats["get"]["avg_ns"] == 0
        assert stats["set"]["count"] == 0
        assert stats["set"]["avg_ns"] == 0


class TestLoggedDescriptor:
    """Test LoggedDescriptor captures log entries for get/set operations."""

    def test_logs_set(self, caplog):
        """Setting a value should produce a log entry."""
        import logging
        class Foo:
            value = LoggedDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        with caplog.at_level(logging.DEBUG):
            f.value = 42
        assert "42" in caplog.text

    def test_logs_get(self, caplog):
        """Getting a value should produce a log entry."""
        import logging
        class Foo:
            value = LoggedDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        with caplog.at_level(logging.DEBUG):
            _ = f.value
        assert "10" in caplog.text

    def test_log_includes_field_name(self, caplog):
        """Log entries should include the field name."""
        import logging
        class Foo:
            value = LoggedDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        with caplog.at_level(logging.DEBUG):
            f.value = 5
        assert "value" in caplog.text

    def test_metadata(self):
        """Metadata should identify this as a logged descriptor."""
        class Foo:
            value = LoggedDescriptor(field_type=int, log_level="INFO")
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["descriptor_id"] == "logged"
        assert meta["log_level"] == "INFO"

    def test_invalid_log_level_defaults_to_debug(self):
        """Invalid log level should default to DEBUG without error."""
        import logging
        class Foo:
            value = LoggedDescriptor(field_type=int, log_level="INVALID_LEVEL")
        Foo.value.__set_name__(Foo, 'value')
        # getattr with default returns logging.DEBUG for invalid levels
        assert Foo.value._log_level == logging.DEBUG


class TestWatchedDescriptor:
    """Test WatchedDescriptor triggers callbacks on value changes."""

    def test_callback_triggered_with_condition(self):
        """Callback should fire when condition is met."""
        triggered = []

        def on_watch(obj, name, value):
            triggered.append(value)

        class Foo:
            value = WatchedDescriptor(
                field_type=int,
                condition=lambda v: v > 50,
                callback=on_watch,
            )
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10  # condition not met
        f.value = 60  # condition met
        assert 60 in triggered
        assert 10 not in triggered

    def test_no_trigger_without_condition(self):
        """Without a condition, callback should never fire."""
        triggered = []

        class Foo:
            value = WatchedDescriptor(
                field_type=int,
                callback=lambda o, n, v: triggered.append(v),
            )
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        f.value = 20
        assert len(triggered) == 0

    def test_callback_receives_object(self):
        """Callback should receive the owning object as first argument."""
        received_obj = []

        def on_watch(obj, name, value):
            received_obj.append(obj)

        class Foo:
            value = WatchedDescriptor(
                field_type=int,
                condition=lambda v: True,
                callback=on_watch,
            )
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 1
        assert received_obj[-1] is f

    def test_metadata(self):
        """Metadata should identify this as a watched descriptor."""
        class Foo:
            value = WatchedDescriptor(field_type=int, condition=lambda v: v > 0, callback=lambda o, n, v: None)
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["descriptor_id"] == "watched"
        assert meta["has_condition"] is True
        assert meta["has_callback"] is True

    def test_condition_always_false_never_triggers(self):
        """Condition that always returns False should never trigger callback."""
        triggered = []

        class Foo:
            value = WatchedDescriptor(
                field_type=int,
                condition=lambda v: False,
                callback=lambda o, n, v: triggered.append(v),
            )
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        f.value = 100
        f.value = 1000
        assert len(triggered) == 0
