"""
Tests for Tier 53: BRIDGES_CACHING decorators.
"""

import pytest

from trinity.decorators.bridges_caching import (
    VALID_BACKOFF_STRATEGIES,
    VALID_CACHE_SCOPES,
    VALID_DIFF_STRATEGIES,
    VALID_FLUSH_MODES,
    VALID_LAZY_INIT_MODES,
    VALID_NOTIFY_MODES,
    VALID_THROTTLE_SCOPES,
    async_load,
    batch,
    cached,
    diff,
    lazy,
    observable,
    priority,
    retry,
    throttle_network,
)
from trinity.decorators.registry import Tier, registry


# ============================================================================
# @cached
# ============================================================================


class TestCached:
    """Test @cached decorator."""

    def test_basic_cached(self):
        @cached(scope="global")
        class MyCache:
            pass

        assert MyCache._cached is True
        assert MyCache._cached_scope == "global"
        assert MyCache._cached_ttl is None
        assert MyCache._cached_max_size is None

    def test_cached_with_ttl_and_max_size(self):
        @cached(ttl=5.0, max_size=1000, scope="entity")
        class EntityCache:
            pass

        assert EntityCache._cached_ttl == 5.0
        assert EntityCache._cached_max_size == 1000
        assert EntityCache._cached_scope == "entity"

    def test_cached_frame_scope(self):
        @cached(scope="frame")
        class FrameCache:
            pass

        assert FrameCache._cached_scope == "frame"

    def test_cached_invalid_scope(self):
        with pytest.raises(ValueError, match="Invalid scope"):
            @cached(scope="invalid")
            class Bad:
                pass

    def test_cached_invalid_ttl(self):
        with pytest.raises(ValueError, match="ttl must be > 0"):
            @cached(ttl=-1)
            class Bad:
                pass

    def test_cached_invalid_max_size(self):
        with pytest.raises(ValueError, match="max_size must be > 0"):
            @cached(max_size=0)
            class Bad:
                pass


# ============================================================================
# @lazy
# ============================================================================


class TestLazy:
    """Test @lazy decorator."""

    def test_basic_lazy(self):
        @lazy()
        class MyLazy:
            pass

        assert MyLazy._lazy is True
        assert MyLazy._lazy_init_on == "first_access"
        assert MyLazy._lazy_thread_safe is True
        assert MyLazy._lazy_fallback is None

    def test_lazy_first_frame(self):
        @lazy(init_on="first_frame", thread_safe=False, fallback="default")
        class FrameLazy:
            pass

        assert FrameLazy._lazy_init_on == "first_frame"
        assert FrameLazy._lazy_thread_safe is False
        assert FrameLazy._lazy_fallback == "default"

    def test_lazy_explicit(self):
        @lazy(init_on="explicit")
        class ExplicitLazy:
            pass

        assert ExplicitLazy._lazy_init_on == "explicit"

    def test_lazy_invalid_init_on(self):
        with pytest.raises(ValueError, match="Invalid init_on"):
            @lazy(init_on="never")
            class Bad:
                pass


# ============================================================================
# @batch
# ============================================================================


class TestBatch:
    """Test @batch decorator."""

    def test_basic_batch(self):
        @batch()
        class MyBatch:
            pass

        assert MyBatch._batch is True
        assert MyBatch._batch_max_size == 64
        assert MyBatch._batch_flush_on == "frame_end"
        assert MyBatch._batch_timeout_ms is None
        assert MyBatch._batch_coalesce is False

    def test_batch_with_params(self):
        @batch(max_size=100, flush_on="full", timeout_ms=500, coalesce=True)
        class FullBatch:
            pass

        assert FullBatch._batch_max_size == 100
        assert FullBatch._batch_flush_on == "full"
        assert FullBatch._batch_timeout_ms == 500
        assert FullBatch._batch_coalesce is True

    def test_batch_timeout_flush(self):
        @batch(flush_on="timeout", timeout_ms=1000)
        class TimeoutBatch:
            pass

        assert TimeoutBatch._batch_flush_on == "timeout"

    def test_batch_invalid_max_size(self):
        with pytest.raises(ValueError, match="max_size must be > 0"):
            @batch(max_size=0)
            class Bad:
                pass

    def test_batch_invalid_flush_on(self):
        with pytest.raises(ValueError, match="Invalid flush_on"):
            @batch(flush_on="never")
            class Bad:
                pass

    def test_batch_invalid_timeout_ms(self):
        with pytest.raises(ValueError, match="timeout_ms must be > 0"):
            @batch(timeout_ms=-1)
            class Bad:
                pass


# ============================================================================
# @async_load
# ============================================================================


class TestAsyncLoad:
    """Test @async_load decorator."""

    def test_basic_async_load(self):
        @async_load()
        class MyLoader:
            pass

        assert MyLoader._async_load is True
        assert MyLoader._async_load_priority == 0
        assert MyLoader._async_load_timeout_ms is None
        assert MyLoader._async_load_fallback is None

    def test_async_load_with_params(self):
        @async_load(priority=10, timeout_ms=5000, fallback="placeholder")
        class PriorityLoader:
            pass

        assert PriorityLoader._async_load_priority == 10
        assert PriorityLoader._async_load_timeout_ms == 5000
        assert PriorityLoader._async_load_fallback == "placeholder"

    def test_async_load_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout_ms must be > 0"):
            @async_load(timeout_ms=-100)
            class Bad:
                pass


# ============================================================================
# @diff
# ============================================================================


class TestDiff:
    """Test @diff decorator."""

    def test_basic_diff(self):
        @diff()
        class MyDiff:
            pass

        assert MyDiff._diff is True
        assert MyDiff._diff_strategy == "shallow"
        assert MyDiff._diff_include_fields is None
        assert MyDiff._diff_exclude_fields is None

    def test_diff_deep(self):
        @diff(strategy="deep")
        class DeepDiff:
            pass

        assert DeepDiff._diff_strategy == "deep"

    def test_diff_structural(self):
        @diff(strategy="structural", include_fields=["slots", "items"])
        class StructDiff:
            pass

        assert StructDiff._diff_strategy == "structural"
        assert StructDiff._diff_include_fields == ["slots", "items"]

    def test_diff_custom(self):
        def my_differ(old, new):
            return old != new

        @diff(strategy="custom", custom_differ=my_differ)
        class CustomDiff:
            pass

        assert CustomDiff._diff_strategy == "custom"

    def test_diff_invalid_strategy(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            @diff(strategy="magic")
            class Bad:
                pass

    def test_diff_include_exclude_conflict(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            @diff(include_fields=["a"], exclude_fields=["b"])
            class Bad:
                pass

    def test_diff_custom_without_differ(self):
        with pytest.raises(ValueError, match="custom_differ is required"):
            @diff(strategy="custom")
            class Bad:
                pass


# ============================================================================
# @priority
# ============================================================================


class TestPriority:
    """Test @priority decorator."""

    def test_basic_priority(self):
        @priority()
        class MyPriority:
            pass

        assert MyPriority._priority is True
        assert MyPriority._priority_value == 0
        assert MyPriority._priority_queue == "default"
        assert MyPriority._priority_higher_first is True

    def test_priority_with_params(self):
        @priority(value=100, queue="damage", higher_first=False)
        class DamagePriority:
            pass

        assert DamagePriority._priority_value == 100
        assert DamagePriority._priority_queue == "damage"
        assert DamagePriority._priority_higher_first is False

    def test_priority_invalid_queue(self):
        with pytest.raises(ValueError, match="queue must be a non-empty string"):
            @priority(queue="")
            class Bad:
                pass


# ============================================================================
# @retry
# ============================================================================


class TestRetry:
    """Test @retry decorator."""

    def test_basic_retry(self):
        @retry()
        class MyRetry:
            pass

        assert MyRetry._retry is True
        assert MyRetry._retry_max_attempts == 3
        assert MyRetry._retry_backoff == "exponential"
        assert MyRetry._retry_base_delay_ms == 100
        assert MyRetry._retry_max_delay_ms == 10000

    def test_retry_with_params(self):
        @retry(max_attempts=5, backoff="linear", base_delay_ms=200, max_delay_ms=5000)
        class LinearRetry:
            pass

        assert LinearRetry._retry_max_attempts == 5
        assert LinearRetry._retry_backoff == "linear"
        assert LinearRetry._retry_base_delay_ms == 200
        assert LinearRetry._retry_max_delay_ms == 5000

    def test_retry_no_backoff(self):
        @retry(backoff="none")
        class NoBackoff:
            pass

        assert NoBackoff._retry_backoff == "none"

    def test_retry_invalid_max_attempts(self):
        with pytest.raises(ValueError, match="max_attempts must be > 0"):
            @retry(max_attempts=0)
            class Bad:
                pass

    def test_retry_invalid_backoff(self):
        with pytest.raises(ValueError, match="Invalid backoff"):
            @retry(backoff="random")
            class Bad:
                pass

    def test_retry_invalid_base_delay(self):
        with pytest.raises(ValueError, match="base_delay_ms must be > 0"):
            @retry(base_delay_ms=0)
            class Bad:
                pass

    def test_retry_invalid_max_delay(self):
        with pytest.raises(ValueError, match="max_delay_ms must be > 0"):
            @retry(max_delay_ms=-1)
            class Bad:
                pass

    def test_retry_base_greater_than_max(self):
        with pytest.raises(ValueError, match="base_delay_ms.*must be <= max_delay_ms"):
            @retry(base_delay_ms=5000, max_delay_ms=100)
            class Bad:
                pass


# ============================================================================
# @throttle_network
# ============================================================================


class TestThrottleNetwork:
    """Test @throttle_network decorator."""

    def test_basic_throttle(self):
        @throttle_network()
        class MyThrottle:
            pass

        assert MyThrottle._throttle_network is True
        assert MyThrottle._throttle_network_max_ups == 20.0
        assert MyThrottle._throttle_network_priority_decay == 0.9
        assert MyThrottle._throttle_network_burst_allowance == 3
        assert MyThrottle._throttle_network_per == "entity"

    def test_throttle_with_params(self):
        @throttle_network(
            max_updates_per_second=60.0,
            priority_decay=1.0,
            burst_allowance=5,
            per="global",
        )
        class GlobalThrottle:
            pass

        assert GlobalThrottle._throttle_network_max_ups == 60.0
        assert GlobalThrottle._throttle_network_priority_decay == 1.0
        assert GlobalThrottle._throttle_network_burst_allowance == 5
        assert GlobalThrottle._throttle_network_per == "global"

    def test_throttle_component_scope(self):
        @throttle_network(per="component")
        class CompThrottle:
            pass

        assert CompThrottle._throttle_network_per == "component"

    def test_throttle_invalid_max_ups(self):
        with pytest.raises(ValueError, match="max_updates_per_second must be > 0"):
            @throttle_network(max_updates_per_second=0)
            class Bad:
                pass

    def test_throttle_invalid_priority_decay(self):
        with pytest.raises(ValueError, match="priority_decay must be between 0 and 1"):
            @throttle_network(priority_decay=1.5)
            class Bad:
                pass

    def test_throttle_invalid_burst(self):
        with pytest.raises(ValueError, match="burst_allowance must be >= 0"):
            @throttle_network(burst_allowance=-1)
            class Bad:
                pass

    def test_throttle_invalid_per(self):
        with pytest.raises(ValueError, match="Invalid per"):
            @throttle_network(per="world")
            class Bad:
                pass


# ============================================================================
# @observable
# ============================================================================


class TestObservable:
    """Test @observable decorator."""

    def test_basic_observable(self):
        @observable()
        class MyObs:
            pass

        assert MyObs._observable is True
        assert MyObs._observable_notify == "sync"
        assert MyObs._observable_batch_delay_ms == 16.0

    def test_observable_deferred(self):
        @observable(notify="deferred")
        class DeferredObs:
            pass

        assert DeferredObs._observable_notify == "deferred"

    def test_observable_batched(self):
        @observable(notify="batched", batch_delay_ms=32.0)
        class BatchedObs:
            pass

        assert BatchedObs._observable_notify == "batched"
        assert BatchedObs._observable_batch_delay_ms == 32.0

    def test_observable_invalid_notify(self):
        with pytest.raises(ValueError, match="Invalid notify"):
            @observable(notify="immediate")
            class Bad:
                pass

    def test_observable_invalid_batch_delay(self):
        with pytest.raises(ValueError, match="batch_delay_ms must be > 0"):
            @observable(batch_delay_ms=0)
            class Bad:
                pass


# ============================================================================
# Registry
# ============================================================================


class TestRegistry:
    """Test registry integration."""

    def test_all_decorators_registered(self):
        names = [
            "cached", "lazy", "batch", "async_load", "diff",
            "priority", "retry", "throttle_network", "observable",
        ]
        for name in names:
            assert name in registry._decorators, f"{name} not registered"

    def test_all_in_bridges_caching_tier(self):
        specs = registry._by_tier[Tier.BRIDGES_CACHING]
        names = {s.name for s in specs}
        expected = {
            "cached", "lazy", "batch", "async_load", "diff",
            "priority", "retry", "throttle_network", "observable",
        }
        assert expected.issubset(names)

    def test_tier_value(self):
        assert Tier.BRIDGES_CACHING == 53


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    """Test valid constant frozensets."""

    def test_cache_scopes(self):
        assert VALID_CACHE_SCOPES == frozenset({"global", "entity", "frame"})

    def test_lazy_init_modes(self):
        assert VALID_LAZY_INIT_MODES == frozenset({"first_access", "first_frame", "explicit"})

    def test_flush_modes(self):
        assert VALID_FLUSH_MODES == frozenset({"full", "frame_end", "explicit", "timeout"})

    def test_diff_strategies(self):
        assert VALID_DIFF_STRATEGIES == frozenset({"shallow", "deep", "structural", "custom"})

    def test_backoff_strategies(self):
        assert VALID_BACKOFF_STRATEGIES == frozenset({"none", "linear", "exponential"})

    def test_throttle_scopes(self):
        assert VALID_THROTTLE_SCOPES == frozenset({"entity", "component", "global"})

    def test_notify_modes(self):
        assert VALID_NOTIFY_MODES == frozenset({"sync", "deferred", "batched"})
