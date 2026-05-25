"""Tests for log filtering.

Tests all filter types: level, category, pattern, rate limit, etc.
"""

import pytest
import time
from datetime import datetime

from engine.tooling.logging.log_system import LogMessage, LogLevel, LogCategory
from engine.tooling.logging.log_filter import (
    LogFilter,
    LevelFilter,
    CategoryFilter,
    PatternFilter,
    RateLimitFilter,
    SamplingFilter,
    DeduplicationFilter,
    CompositeFilter,
    CallbackFilter,
    FilterAction,
)


@pytest.fixture
def info_message():
    """Create an info level message."""
    return LogMessage(
        level=LogLevel.INFO,
        category=LogCategory.ENGINE,
        message="Test message"
    )


@pytest.fixture
def error_message():
    """Create an error level message."""
    return LogMessage(
        level=LogLevel.ERROR,
        category=LogCategory.ENGINE,
        message="Error occurred"
    )


class TestLevelFilter:
    """Tests for LevelFilter."""

    def test_min_level(self, info_message):
        filter = LevelFilter(min_level=LogLevel.WARNING)

        assert filter.filter(info_message) == FilterAction.DROP

        info_message.level = LogLevel.WARNING
        assert filter.filter(info_message) == FilterAction.PASS

    def test_max_level(self, info_message):
        filter = LevelFilter(max_level=LogLevel.WARNING)

        assert filter.filter(info_message) == FilterAction.PASS

        info_message.level = LogLevel.ERROR
        assert filter.filter(info_message) == FilterAction.DROP

    def test_level_range(self, info_message):
        filter = LevelFilter(min_level=LogLevel.INFO, max_level=LogLevel.WARNING)

        info_message.level = LogLevel.DEBUG
        assert filter.filter(info_message) == FilterAction.DROP

        info_message.level = LogLevel.INFO
        assert filter.filter(info_message) == FilterAction.PASS

        info_message.level = LogLevel.WARNING
        assert filter.filter(info_message) == FilterAction.PASS

        info_message.level = LogLevel.ERROR
        assert filter.filter(info_message) == FilterAction.DROP

    def test_disabled_filter(self, info_message):
        filter = LevelFilter(min_level=LogLevel.ERROR)
        filter.enabled = False

        assert filter.filter(info_message) == FilterAction.PASS

    def test_level_properties(self):
        filter = LevelFilter(min_level=LogLevel.INFO, max_level=LogLevel.ERROR)
        assert filter.min_level == LogLevel.INFO
        assert filter.max_level == LogLevel.ERROR

        filter.min_level = LogLevel.DEBUG
        assert filter.min_level == LogLevel.DEBUG


class TestCategoryFilter:
    """Tests for CategoryFilter."""

    def test_include_filter(self, info_message):
        filter = CategoryFilter(include={LogCategory.ENGINE, LogCategory.RENDER})

        assert filter.filter(info_message) == FilterAction.PASS

        info_message.category = LogCategory.AUDIO
        assert filter.filter(info_message) == FilterAction.DROP

    def test_exclude_filter(self, info_message):
        filter = CategoryFilter(exclude={LogCategory.AUDIO, LogCategory.NETWORK})

        assert filter.filter(info_message) == FilterAction.PASS

        info_message.category = LogCategory.AUDIO
        assert filter.filter(info_message) == FilterAction.DROP

    def test_include_and_exclude(self, info_message):
        filter = CategoryFilter(
            include={LogCategory.ENGINE, LogCategory.RENDER, LogCategory.AUDIO},
            exclude={LogCategory.AUDIO}
        )

        assert filter.filter(info_message) == FilterAction.PASS

        info_message.category = LogCategory.AUDIO
        assert filter.filter(info_message) == FilterAction.DROP

    def test_include_method(self, info_message):
        filter = CategoryFilter()
        filter.include(LogCategory.ENGINE)

        assert filter.filter(info_message) == FilterAction.PASS

        info_message.category = LogCategory.RENDER
        assert filter.filter(info_message) == FilterAction.DROP

    def test_exclude_method(self, info_message):
        filter = CategoryFilter()
        filter.exclude(LogCategory.ENGINE)

        assert filter.filter(info_message) == FilterAction.DROP

        info_message.category = LogCategory.RENDER
        assert filter.filter(info_message) == FilterAction.PASS


class TestPatternFilter:
    """Tests for PatternFilter."""

    def test_basic_match(self, info_message):
        filter = PatternFilter(pattern="error", invert=True)

        assert filter.filter(info_message) == FilterAction.PASS

        info_message.message = "An error occurred"
        assert filter.filter(info_message) == FilterAction.DROP

    def test_keep_matching(self, info_message):
        filter = PatternFilter(pattern="test", invert=False)

        assert filter.filter(info_message) == FilterAction.PASS

        info_message.message = "Other message"
        assert filter.filter(info_message) == FilterAction.DROP

    def test_case_insensitive(self, info_message):
        filter = PatternFilter(pattern="TEST", ignore_case=True)
        assert filter.filter(info_message) == FilterAction.PASS

    def test_regex_pattern(self, info_message):
        filter = PatternFilter(pattern=r"\d+")

        info_message.message = "Message 123"
        assert filter.filter(info_message) == FilterAction.PASS

        info_message.message = "No numbers"
        assert filter.filter(info_message) == FilterAction.DROP

    def test_match_file(self, info_message):
        filter = PatternFilter(pattern="physics", match_field="file")

        info_message.file = "physics_system.py"
        assert filter.filter(info_message) == FilterAction.PASS

        info_message.file = "audio_system.py"
        assert filter.filter(info_message) == FilterAction.DROP

    def test_match_function(self, info_message):
        filter = PatternFilter(pattern="update", match_field="function")

        info_message.function = "update_entities"
        assert filter.filter(info_message) == FilterAction.PASS

        info_message.function = "render_frame"
        assert filter.filter(info_message) == FilterAction.DROP

    def test_match_all(self, info_message):
        filter = PatternFilter(pattern="physics", match_field="all")

        info_message.file = "physics.py"
        assert filter.filter(info_message) == FilterAction.PASS

        info_message.file = None
        info_message.function = "physics_update"
        assert filter.filter(info_message) == FilterAction.PASS


class TestRateLimitFilter:
    """Tests for RateLimitFilter."""

    def test_allows_under_limit(self, info_message):
        filter = RateLimitFilter(max_count=10, window_seconds=1.0)

        for _ in range(10):
            assert filter.filter(info_message) == FilterAction.PASS

    def test_blocks_over_limit(self, info_message):
        filter = RateLimitFilter(max_count=5, window_seconds=1.0)

        for _ in range(5):
            filter.filter(info_message)

        assert filter.filter(info_message) == FilterAction.DROP

    def test_resets_after_window(self, info_message):
        filter = RateLimitFilter(max_count=2, window_seconds=0.05)

        filter.filter(info_message)
        filter.filter(info_message)
        assert filter.filter(info_message) == FilterAction.DROP

        time.sleep(0.1)

        assert filter.filter(info_message) == FilterAction.PASS

    def test_separate_keys(self, info_message, error_message):
        filter = RateLimitFilter(max_count=2, window_seconds=1.0)

        filter.filter(info_message)
        filter.filter(info_message)

        # Different category/level should have separate limit
        assert filter.filter(error_message) == FilterAction.PASS


class TestSamplingFilter:
    """Tests for SamplingFilter."""

    def test_sample_rate_1(self, info_message):
        filter = SamplingFilter(sample_rate=1.0)

        for _ in range(100):
            assert filter.filter(info_message) == FilterAction.PASS

    def test_sample_rate_0(self, info_message):
        filter = SamplingFilter(sample_rate=0.0)

        for _ in range(100):
            assert filter.filter(info_message) == FilterAction.DROP

    def test_sample_rate_half(self, info_message):
        filter = SamplingFilter(sample_rate=0.5)

        passed = sum(
            1 for _ in range(100)
            if filter.filter(info_message) == FilterAction.PASS
        )

        # Should be roughly 50%, but deterministic
        assert 40 <= passed <= 60


class TestDeduplicationFilter:
    """Tests for DeduplicationFilter."""

    def test_allows_unique(self, info_message):
        filter = DeduplicationFilter(window_size=100)

        info_message.message = "Message 1"
        assert filter.filter(info_message) == FilterAction.PASS

        info_message.message = "Message 2"
        assert filter.filter(info_message) == FilterAction.PASS

    def test_blocks_duplicate(self, info_message):
        filter = DeduplicationFilter()

        assert filter.filter(info_message) == FilterAction.PASS
        assert filter.filter(info_message) == FilterAction.DROP

    def test_allows_after_window(self, info_message):
        filter = DeduplicationFilter(window_size=2)

        assert filter.filter(info_message) == FilterAction.PASS

        # Fill window with other messages
        info_message.message = "Other 1"
        filter.filter(info_message)
        info_message.message = "Other 2"
        filter.filter(info_message)

        # Original should be allowed again
        info_message.message = "Test message"
        assert filter.filter(info_message) == FilterAction.PASS


class TestCompositeFilter:
    """Tests for CompositeFilter."""

    def test_and_mode(self, info_message):
        level_filter = LevelFilter(min_level=LogLevel.INFO)
        category_filter = CategoryFilter(include={LogCategory.ENGINE})

        composite = CompositeFilter([level_filter, category_filter], mode="and")

        assert composite.filter(info_message) == FilterAction.PASS

        info_message.level = LogLevel.DEBUG
        assert composite.filter(info_message) == FilterAction.DROP

        info_message.level = LogLevel.INFO
        info_message.category = LogCategory.AUDIO
        assert composite.filter(info_message) == FilterAction.DROP

    def test_or_mode(self, info_message):
        level_filter = LevelFilter(min_level=LogLevel.ERROR)
        pattern_filter = PatternFilter(pattern="important")

        composite = CompositeFilter([level_filter, pattern_filter], mode="or")

        info_message.message = "important update"
        assert composite.filter(info_message) == FilterAction.PASS

        info_message.message = "regular message"
        info_message.level = LogLevel.ERROR
        assert composite.filter(info_message) == FilterAction.PASS

        info_message.message = "regular message"
        info_message.level = LogLevel.INFO
        assert composite.filter(info_message) == FilterAction.DROP

    def test_add_filter(self, info_message):
        composite = CompositeFilter(mode="and")
        composite.add_filter(LevelFilter(min_level=LogLevel.WARNING))

        assert composite.filter(info_message) == FilterAction.DROP

    def test_remove_filter(self, info_message):
        level_filter = LevelFilter(min_level=LogLevel.WARNING)
        composite = CompositeFilter([level_filter])

        composite.remove_filter(level_filter)
        assert composite.filter(info_message) == FilterAction.PASS

    def test_empty_composite(self, info_message):
        composite = CompositeFilter()
        assert composite.filter(info_message) == FilterAction.PASS


class TestCallbackFilter:
    """Tests for CallbackFilter."""

    def test_callback_pass(self, info_message):
        def callback(msg):
            return FilterAction.PASS

        filter = CallbackFilter(callback)
        assert filter.filter(info_message) == FilterAction.PASS

    def test_callback_drop(self, info_message):
        def callback(msg):
            return FilterAction.DROP

        filter = CallbackFilter(callback)
        assert filter.filter(info_message) == FilterAction.DROP

    def test_callback_bool_return(self, info_message):
        def callback(msg):
            return msg.level >= LogLevel.WARNING

        filter = CallbackFilter(callback)
        assert filter.filter(info_message) == FilterAction.DROP

        info_message.level = LogLevel.WARNING
        assert filter.filter(info_message) == FilterAction.PASS

    def test_callback_exception(self, info_message):
        def bad_callback(msg):
            raise Exception("Callback error")

        filter = CallbackFilter(bad_callback)
        # Should pass on exception
        assert filter.filter(info_message) == FilterAction.PASS


class TestFilterDisabled:
    """Tests for disabled filters."""

    def test_disabled_level_filter(self, info_message):
        filter = LevelFilter(min_level=LogLevel.FATAL)
        filter.enabled = False
        assert filter.filter(info_message) == FilterAction.PASS

    def test_disabled_category_filter(self, info_message):
        filter = CategoryFilter(include={LogCategory.AUDIO})
        filter.enabled = False
        assert filter.filter(info_message) == FilterAction.PASS

    def test_disabled_pattern_filter(self, info_message):
        filter = PatternFilter(pattern="nomatch")
        filter.enabled = False
        assert filter.filter(info_message) == FilterAction.PASS

    def test_disabled_rate_limit_filter(self, info_message):
        filter = RateLimitFilter(max_count=0)
        filter.enabled = False
        assert filter.filter(info_message) == FilterAction.PASS
