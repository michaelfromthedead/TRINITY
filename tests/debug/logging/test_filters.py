"""
Comprehensive tests for log filters.

Tests cover:
- LevelFilter minimum level filtering
- CategoryFilter include/exclude modes
- KeywordFilter string and regex matching
- CompositeFilter AND/OR logic
- NegateFilter inversion
- RateLimitFilter rate limiting
- SamplingFilter random sampling
- CallbackFilter custom predicates
- FieldFilter structured field filtering
"""

import pytest
import sys
import time
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

sys.path.insert(0, "/home/user/dev/AI_GAME_ENGINE")

from engine.debug.logging.logger import LogLevel, LogCategory, LogEntry
from engine.debug.logging.filters import (
    LogFilter,
    LevelFilter,
    CategoryFilter,
    KeywordFilter,
    CompositeFilter,
    NegateFilter,
    RateLimitFilter,
    SamplingFilter,
    CallbackFilter,
    FieldFilter,
)


def make_entry(
    message: str = "Test message",
    level: LogLevel = LogLevel.INFO,
    category: LogCategory = LogCategory.LogEngine,
    logger_name: str = "Test",
    fields: dict = None,
) -> LogEntry:
    """Helper to create test log entries."""
    return LogEntry(
        timestamp=datetime.now(timezone.utc),
        level=level,
        category=category,
        message=message,
        logger_name=logger_name,
        fields=fields or {},
    )


class TestLevelFilter:
    """Tests for LevelFilter."""

    def test_filter_below_min_level(self):
        """Verify entries below min level are rejected."""
        filter = LevelFilter(LogLevel.WARNING)

        assert not filter.should_log(make_entry(level=LogLevel.DEBUG))
        assert not filter.should_log(make_entry(level=LogLevel.INFO))

    def test_allow_at_min_level(self):
        """Verify entries at min level are allowed."""
        filter = LevelFilter(LogLevel.WARNING)

        assert filter.should_log(make_entry(level=LogLevel.WARNING))

    def test_allow_above_min_level(self):
        """Verify entries above min level are allowed."""
        filter = LevelFilter(LogLevel.WARNING)

        assert filter.should_log(make_entry(level=LogLevel.ERROR))
        assert filter.should_log(make_entry(level=LogLevel.FATAL))

    def test_verbose_level_allows_all(self):
        """Verify VERBOSE min level allows all."""
        filter = LevelFilter(LogLevel.VERBOSE)

        for level in LogLevel:
            assert filter.should_log(make_entry(level=level))

    def test_fatal_level_only_fatal(self):
        """Verify FATAL min level only allows FATAL."""
        filter = LevelFilter(LogLevel.FATAL)

        for level in LogLevel:
            if level == LogLevel.FATAL:
                assert filter.should_log(make_entry(level=level))
            else:
                assert not filter.should_log(make_entry(level=level))

    def test_repr(self):
        """Verify repr is informative."""
        filter = LevelFilter(LogLevel.WARNING)
        assert "WARNING" in repr(filter)


class TestCategoryFilter:
    """Tests for CategoryFilter."""

    def test_include_mode(self):
        """Verify include mode only allows specified categories."""
        filter = CategoryFilter(
            include=[LogCategory.LogRendering, LogCategory.LogPhysics]
        )

        assert filter.should_log(make_entry(category=LogCategory.LogRendering))
        assert filter.should_log(make_entry(category=LogCategory.LogPhysics))
        assert not filter.should_log(make_entry(category=LogCategory.LogNetwork))

    def test_exclude_mode(self):
        """Verify exclude mode blocks specified categories."""
        filter = CategoryFilter(
            exclude=[LogCategory.LogInput, LogCategory.LogUI]
        )

        assert filter.should_log(make_entry(category=LogCategory.LogEngine))
        assert not filter.should_log(make_entry(category=LogCategory.LogInput))
        assert not filter.should_log(make_entry(category=LogCategory.LogUI))

    def test_empty_filter_allows_all(self):
        """Verify empty filter allows all categories."""
        filter = CategoryFilter()

        for category in LogCategory:
            assert filter.should_log(make_entry(category=category))

    def test_both_include_exclude_raises(self):
        """Verify specifying both include and exclude raises."""
        with pytest.raises(ValueError):
            CategoryFilter(
                include=[LogCategory.LogEngine],
                exclude=[LogCategory.LogInput],
            )

    def test_add_include(self):
        """Verify categories can be added to include set."""
        filter = CategoryFilter(include=[LogCategory.LogEngine])
        filter.add_include(LogCategory.LogNetwork)

        assert filter.should_log(make_entry(category=LogCategory.LogNetwork))

    def test_add_exclude(self):
        """Verify categories can be added to exclude set."""
        filter = CategoryFilter(exclude=[LogCategory.LogInput])
        filter.add_exclude(LogCategory.LogUI)

        assert not filter.should_log(make_entry(category=LogCategory.LogUI))

    def test_add_include_with_exclude_set_raises(self):
        """Verify adding include when exclude is set raises."""
        filter = CategoryFilter(exclude=[LogCategory.LogInput])

        with pytest.raises(ValueError):
            filter.add_include(LogCategory.LogEngine)

    def test_repr(self):
        """Verify repr is informative."""
        filter = CategoryFilter(include=[LogCategory.LogEngine])
        assert "include" in repr(filter)


class TestKeywordFilter:
    """Tests for KeywordFilter."""

    def test_include_keyword(self):
        """Verify include keywords must be present."""
        filter = KeywordFilter(include=["error", "failed"])

        assert filter.should_log(make_entry("Connection error occurred"))
        assert filter.should_log(make_entry("Operation failed"))
        assert not filter.should_log(make_entry("Success!"))

    def test_exclude_keyword(self):
        """Verify exclude keywords are blocked."""
        filter = KeywordFilter(exclude=["debug", "trace"])

        assert filter.should_log(make_entry("Important message"))
        assert not filter.should_log(make_entry("debug: checking values"))
        assert not filter.should_log(make_entry("trace: function called"))

    def test_case_insensitive_default(self):
        """Verify matching is case-insensitive by default."""
        filter = KeywordFilter(include=["error"])

        assert filter.should_log(make_entry("ERROR occurred"))
        assert filter.should_log(make_entry("Error found"))
        assert filter.should_log(make_entry("error detected"))

    def test_case_sensitive_option(self):
        """Verify case-sensitive matching when enabled."""
        filter = KeywordFilter(include=["Error"], case_sensitive=True)

        assert filter.should_log(make_entry("Error found"))
        assert not filter.should_log(make_entry("ERROR occurred"))
        assert not filter.should_log(make_entry("error detected"))

    def test_include_regex(self):
        """Verify regex pattern matching."""
        filter = KeywordFilter(include_regex=[r"player_\d+"])

        assert filter.should_log(make_entry("player_123 joined"))
        assert filter.should_log(make_entry("player_456 left"))
        assert not filter.should_log(make_entry("player joined"))

    def test_exclude_regex(self):
        """Verify regex exclusion patterns."""
        filter = KeywordFilter(exclude_regex=[r"heartbeat_\d+"])

        assert filter.should_log(make_entry("Game started"))
        assert not filter.should_log(make_entry("heartbeat_100"))

    def test_search_fields_option(self):
        """Verify searching in fields when enabled."""
        filter = KeywordFilter(include=["admin"], search_fields=True)

        assert filter.should_log(make_entry(
            "User action",
            fields={"role": "admin"},
        ))
        assert not filter.should_log(make_entry(
            "User action",
            fields={"role": "player"},
        ))

    def test_multiple_conditions(self):
        """Verify multiple include/exclude conditions."""
        filter = KeywordFilter(
            include=["error", "warning"],
            exclude=["ignore"],
        )

        assert filter.should_log(make_entry("error occurred"))
        assert filter.should_log(make_entry("warning issued"))
        assert not filter.should_log(make_entry("ignore this error"))
        assert not filter.should_log(make_entry("success"))


class TestCompositeFilter:
    """Tests for CompositeFilter."""

    def test_and_mode(self):
        """Verify AND mode requires all filters to pass."""
        filter = CompositeFilter([
            LevelFilter(LogLevel.WARNING),
            CategoryFilter(include=[LogCategory.LogNetwork]),
        ], mode="and")

        # Both conditions met
        assert filter.should_log(make_entry(
            level=LogLevel.WARNING,
            category=LogCategory.LogNetwork,
        ))

        # Only level met
        assert not filter.should_log(make_entry(
            level=LogLevel.WARNING,
            category=LogCategory.LogEngine,
        ))

        # Only category met
        assert not filter.should_log(make_entry(
            level=LogLevel.INFO,
            category=LogCategory.LogNetwork,
        ))

    def test_or_mode(self):
        """Verify OR mode passes if any filter passes."""
        filter = CompositeFilter([
            LevelFilter(LogLevel.ERROR),
            CategoryFilter(include=[LogCategory.LogNetwork]),
        ], mode="or")

        # Both conditions met
        assert filter.should_log(make_entry(
            level=LogLevel.ERROR,
            category=LogCategory.LogNetwork,
        ))

        # Only level met
        assert filter.should_log(make_entry(
            level=LogLevel.ERROR,
            category=LogCategory.LogEngine,
        ))

        # Only category met
        assert filter.should_log(make_entry(
            level=LogLevel.INFO,
            category=LogCategory.LogNetwork,
        ))

        # Neither met
        assert not filter.should_log(make_entry(
            level=LogLevel.INFO,
            category=LogCategory.LogEngine,
        ))

    def test_invalid_mode_raises(self):
        """Verify invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            CompositeFilter([], mode="invalid")

    def test_empty_filter_allows_all(self):
        """Verify empty composite allows all."""
        filter = CompositeFilter([], mode="and")
        assert filter.should_log(make_entry())

    def test_add_filter(self):
        """Verify filters can be added dynamically."""
        filter = CompositeFilter([], mode="and")
        filter.add_filter(LevelFilter(LogLevel.WARNING))

        assert filter.should_log(make_entry(level=LogLevel.WARNING))
        assert not filter.should_log(make_entry(level=LogLevel.INFO))

    def test_operator_and(self):
        """Verify & operator combines with AND."""
        f1 = LevelFilter(LogLevel.WARNING)
        f2 = CategoryFilter(include=[LogCategory.LogNetwork])

        combined = f1 & f2

        assert isinstance(combined, CompositeFilter)
        assert combined.mode == "and"

    def test_operator_or(self):
        """Verify | operator combines with OR."""
        f1 = LevelFilter(LogLevel.ERROR)
        f2 = CategoryFilter(include=[LogCategory.LogNetwork])

        combined = f1 | f2

        assert isinstance(combined, CompositeFilter)
        assert combined.mode == "or"


class TestNegateFilter:
    """Tests for NegateFilter."""

    def test_negation(self):
        """Verify filter results are inverted."""
        level_filter = LevelFilter(LogLevel.WARNING)
        negate_filter = NegateFilter(level_filter)

        # Original would pass, negated should fail
        assert not negate_filter.should_log(make_entry(level=LogLevel.ERROR))

        # Original would fail, negated should pass
        assert negate_filter.should_log(make_entry(level=LogLevel.INFO))

    def test_operator_invert(self):
        """Verify ~ operator creates NegateFilter."""
        level_filter = LevelFilter(LogLevel.ERROR)
        negated = ~level_filter

        assert isinstance(negated, NegateFilter)
        assert negated.should_log(make_entry(level=LogLevel.INFO))

    def test_double_negation(self):
        """Verify double negation restores original behavior."""
        original = LevelFilter(LogLevel.WARNING)
        double_negated = ~~original

        entry = make_entry(level=LogLevel.ERROR)
        assert double_negated.should_log(entry) == original.should_log(entry)


class TestRateLimitFilter:
    """Tests for RateLimitFilter."""

    def test_within_rate_limit(self):
        """Verify entries within rate limit pass."""
        filter = RateLimitFilter(max_entries=5, window_seconds=1.0)

        for _ in range(5):
            assert filter.should_log(make_entry())

    def test_exceeds_rate_limit(self):
        """Verify entries exceeding rate limit are blocked."""
        filter = RateLimitFilter(max_entries=3, window_seconds=1.0)

        for _ in range(3):
            assert filter.should_log(make_entry())

        # 4th should be blocked
        assert not filter.should_log(make_entry())

    def test_rate_limit_window_reset(self):
        """Verify rate limit resets after window expires."""
        filter = RateLimitFilter(max_entries=2, window_seconds=0.1)

        # Fill the limit
        assert filter.should_log(make_entry())
        assert filter.should_log(make_entry())
        assert not filter.should_log(make_entry())

        # Wait for window to expire
        time.sleep(0.15)

        # Should be allowed again
        assert filter.should_log(make_entry())

    def test_per_category_rate_limit(self):
        """Verify per-category rate limiting."""
        filter = RateLimitFilter(
            max_entries=2,
            window_seconds=1.0,
            per_category=True,
        )

        # Fill limit for LogEngine
        assert filter.should_log(make_entry(category=LogCategory.LogEngine))
        assert filter.should_log(make_entry(category=LogCategory.LogEngine))
        assert not filter.should_log(make_entry(category=LogCategory.LogEngine))

        # LogNetwork should have its own limit
        assert filter.should_log(make_entry(category=LogCategory.LogNetwork))

    def test_per_logger_rate_limit(self):
        """Verify per-logger rate limiting."""
        filter = RateLimitFilter(
            max_entries=2,
            window_seconds=1.0,
            per_logger=True,
        )

        # Fill limit for Logger1
        assert filter.should_log(make_entry(logger_name="Logger1"))
        assert filter.should_log(make_entry(logger_name="Logger1"))
        assert not filter.should_log(make_entry(logger_name="Logger1"))

        # Logger2 should have its own limit
        assert filter.should_log(make_entry(logger_name="Logger2"))


class TestSamplingFilter:
    """Tests for SamplingFilter."""

    def test_sample_rate_validation(self):
        """Verify invalid sample rates raise ValueError."""
        with pytest.raises(ValueError):
            SamplingFilter(sample_rate=-0.1)

        with pytest.raises(ValueError):
            SamplingFilter(sample_rate=1.5)

    def test_zero_sample_rate_blocks_all(self):
        """Verify 0.0 sample rate blocks all entries."""
        filter = SamplingFilter(sample_rate=0.0)

        for _ in range(100):
            assert not filter.should_log(make_entry())

    def test_full_sample_rate_allows_all(self):
        """Verify 1.0 sample rate allows all entries."""
        filter = SamplingFilter(sample_rate=1.0)

        for _ in range(100):
            assert filter.should_log(make_entry())

    def test_partial_sample_rate(self):
        """Verify partial sample rate approximately correct."""
        filter = SamplingFilter(sample_rate=0.5, seed=42)

        passed = sum(1 for _ in range(1000) if filter.should_log(make_entry()))

        # Should be roughly 50% (with some tolerance)
        assert 400 < passed < 600


class TestCallbackFilter:
    """Tests for CallbackFilter."""

    def test_callback_accepts(self):
        """Verify callback that returns True allows entries."""
        filter = CallbackFilter(lambda entry: True)

        assert filter.should_log(make_entry())

    def test_callback_rejects(self):
        """Verify callback that returns False rejects entries."""
        filter = CallbackFilter(lambda entry: False)

        assert not filter.should_log(make_entry())

    def test_callback_with_logic(self):
        """Verify callback can implement custom logic."""
        def custom_filter(entry):
            return entry.fields.get("priority", 0) > 5

        filter = CallbackFilter(custom_filter)

        assert filter.should_log(make_entry(fields={"priority": 10}))
        assert not filter.should_log(make_entry(fields={"priority": 3}))

    def test_callback_repr(self):
        """Verify repr includes callback name."""
        def my_filter(entry):
            return True

        filter = CallbackFilter(my_filter)
        assert "my_filter" in repr(filter)


class TestFieldFilter:
    """Tests for FieldFilter."""

    def test_field_exists_and_matches(self):
        """Verify field matching works."""
        filter = FieldFilter("player_id", lambda v: v == 123)

        assert filter.should_log(make_entry(fields={"player_id": 123}))
        assert not filter.should_log(make_entry(fields={"player_id": 456}))

    def test_field_missing_with_require(self):
        """Verify missing field is rejected when required."""
        filter = FieldFilter("player_id", lambda v: True, require_field=True)

        assert not filter.should_log(make_entry(fields={}))

    def test_field_missing_without_require(self):
        """Verify missing field is allowed when not required."""
        filter = FieldFilter("player_id", lambda v: True, require_field=False)

        assert filter.should_log(make_entry(fields={}))

    def test_field_numeric_comparison(self):
        """Verify numeric field comparisons work."""
        filter = FieldFilter("latency_ms", lambda v: v > 100)

        assert filter.should_log(make_entry(fields={"latency_ms": 150}))
        assert not filter.should_log(make_entry(fields={"latency_ms": 50}))

    def test_field_string_comparison(self):
        """Verify string field comparisons work."""
        filter = FieldFilter("status", lambda v: v == "error")

        assert filter.should_log(make_entry(fields={"status": "error"}))
        assert not filter.should_log(make_entry(fields={"status": "success"}))


class TestFilterThreadSafety:
    """Tests for filter thread safety."""

    def test_rate_limit_thread_safety(self):
        """Verify RateLimitFilter is thread-safe."""
        filter = RateLimitFilter(max_entries=50, window_seconds=1.0)
        results = []
        lock = threading.Lock()

        def check_filter():
            for _ in range(20):
                result = filter.should_log(make_entry())
                with lock:
                    results.append(result)

        threads = [
            threading.Thread(target=check_filter)
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have processed all without errors
        assert len(results) == 100
        # Exactly 50 should have passed
        assert sum(results) == 50

    def test_sampling_filter_thread_safety(self):
        """Verify SamplingFilter is thread-safe."""
        filter = SamplingFilter(sample_rate=0.5, seed=42)
        results = []
        errors = []
        lock = threading.Lock()

        def check_filter():
            try:
                for _ in range(100):
                    result = filter.should_log(make_entry())
                    with lock:
                        results.append(result)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [
            threading.Thread(target=check_filter)
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors during concurrent access
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Should have processed all entries
        assert len(results) == 500
        # Sampling rate should be approximately 50% (with tolerance for randomness)
        passed = sum(results)
        assert 175 < passed < 325, f"Sampling rate out of expected range: {passed}/500"
