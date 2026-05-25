"""Tests for advanced composite stacks added to composite.py."""
import pytest
from trinity.decorators.stacks import Stack
from trinity.decorators.builtin_stacks.composite import (
    resilient_cloud_service,
    reactive_ui_component,
    smart_query_cache,
    streaming_asset_loader,
    optimized_network_sync,
    saveable_game_state,
    observable_game_event,
)


def _expand(s: Stack) -> list[str]:
    """Return decorator names for a stack."""
    return s.expand()


def _has_decorator(s: Stack, name: str) -> bool:
    """Check if a stack contains a decorator with the given name."""
    return name in _expand(s)


# ---------------------------------------------------------------------------
# resilient_cloud_service
# ---------------------------------------------------------------------------

class TestResilientCloudService:
    """resilient_cloud_service — retry + cached + cloud resilience."""

    def test_default_returns_stack(self):
        s = resilient_cloud_service()
        assert isinstance(s, Stack), "Should return a Stack instance"

    def test_exact_decorator_count(self):
        s = resilient_cloud_service()
        # production_component(5) + stack(retry, async_load, cached) = 8
        assert len(s.decorators) == 8, (
            f"Expected 8 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_contains_resolvable_decorators(self):
        """Verify decorators whose names resolve (track_changes, component)."""
        s = resilient_cloud_service()
        names = _expand(s)
        assert "track_changes" in names, (
            f"Should include track_changes, got {names}"
        )
        assert "component" in names, (
            f"Should include component, got {names}"
        )

    def test_custom_params_preserve_structure(self):
        s = resilient_cloud_service(max_attempts=10, cache_ttl=600.0)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 8, (
            "Custom params should not change decorator count"
        )

    def test_all_custom_params(self):
        s = resilient_cloud_service(
            max_attempts=10, cache_ttl=600.0, timeout_ms=5000, pool_size=64
        )
        assert len(s.decorators) == 8

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            resilient_cloud_service(nonexistent_param=42)


# ---------------------------------------------------------------------------
# reactive_ui_component
# ---------------------------------------------------------------------------

class TestReactiveUiComponent:
    """reactive_ui_component — observable + diff for reactive UI."""

    def test_default_returns_stack(self):
        s = reactive_ui_component()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = reactive_ui_component()
        # production_component(5) + stack(observable, diff, lazy, serializable) = 9
        assert len(s.decorators) == 9, (
            f"Expected 9 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_contains_resolvable_decorators(self):
        s = reactive_ui_component()
        names = _expand(s)
        assert "track_changes" in names, (
            f"Should include track_changes, got {names}"
        )
        assert "component" in names, (
            f"Should include component, got {names}"
        )

    def test_custom_params(self):
        s = reactive_ui_component(batch_delay_ms=16.0)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 9

    def test_custom_pool_size(self):
        s = reactive_ui_component(pool_size=512)
        assert len(s.decorators) == 9

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            reactive_ui_component(nonexistent=True)


# ---------------------------------------------------------------------------
# smart_query_cache
# ---------------------------------------------------------------------------

class TestSmartQueryCache:
    """smart_query_cache — cached + batch + priority for query optimization."""

    def test_default_returns_stack(self):
        s = smart_query_cache()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = smart_query_cache()
        # stack(cached, batch, priority) + profiled_dev(3) = 6
        assert len(s.decorators) == 6, (
            f"Expected 6 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_custom_max_size(self):
        # NOTE: param is max_size, not cache_size
        s = smart_query_cache(max_size=1000)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 6

    def test_custom_ttl(self):
        s = smart_query_cache(ttl=10.0)
        assert len(s.decorators) == 6

    def test_custom_batch_size(self):
        s = smart_query_cache(batch_size=128)
        assert len(s.decorators) == 6

    def test_wrong_param_name_cache_size_raises(self):
        """cache_size is NOT a valid param; max_size is correct."""
        with pytest.raises(TypeError):
            smart_query_cache(cache_size=1000)


# ---------------------------------------------------------------------------
# streaming_asset_loader
# ---------------------------------------------------------------------------

class TestStreamingAssetLoader:
    """streaming_asset_loader — streamable + async_load + cached."""

    def test_default_returns_stack(self):
        s = streaming_asset_loader()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = streaming_asset_loader()
        # 8 decorators: streamable, loading_priority, unloadable,
        #   async_load, lazy, priority, batch, cached
        assert len(s.decorators) == 8, (
            f"Expected 8 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_custom_params(self):
        s = streaming_asset_loader(cache_size=500)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 8

    def test_custom_timeout(self):
        s = streaming_asset_loader(timeout_ms=10000)
        assert len(s.decorators) == 8

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            streaming_asset_loader(nonexistent=True)


# ---------------------------------------------------------------------------
# optimized_network_sync
# ---------------------------------------------------------------------------

class TestOptimizedNetworkSync:
    """optimized_network_sync — throttle_network + diff + batch."""

    def test_default_returns_stack(self):
        s = optimized_network_sync()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = optimized_network_sync()
        assert len(s.decorators) == 3, (
            f"Expected 3 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_custom_params(self):
        s = optimized_network_sync(max_updates_per_second=20.0, batch_size=256)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 3

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            optimized_network_sync(nonexistent=True)


# ---------------------------------------------------------------------------
# saveable_game_state
# ---------------------------------------------------------------------------

class TestSaveableGameState:
    """saveable_game_state — encrypted + retry + diff for persistence."""

    def test_default_returns_stack(self):
        s = saveable_game_state()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = saveable_game_state()
        assert len(s.decorators) == 5, (
            f"Expected 5 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_custom_params(self):
        s = saveable_game_state(version=2, max_retry=5)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 5

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            saveable_game_state(nonexistent=True)


# ---------------------------------------------------------------------------
# observable_game_event
# ---------------------------------------------------------------------------

class TestObservableGameEvent:
    """observable_game_event — event + observable + priority."""

    def test_default_returns_stack(self):
        s = observable_game_event()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = observable_game_event()
        assert len(s.decorators) == 4, (
            f"Expected 4 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_custom_params(self):
        s = observable_game_event(priority_value=20, batch_size=512)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 4

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            observable_game_event(nonexistent=True)


# ---------------------------------------------------------------------------
# Cross-stack composition
# ---------------------------------------------------------------------------

class TestStackComposition:
    """Verify that advanced composite stacks can be combined with +."""

    def test_two_composites_add(self):
        a = resilient_cloud_service()
        b = smart_query_cache()
        combined = a + b
        assert isinstance(combined, Stack)
        assert len(combined.decorators) == len(a.decorators) + len(b.decorators), (
            "Combined stack should have sum of decorator counts"
        )

    def test_add_non_stack_raises(self):
        s = resilient_cloud_service()
        result = s.__add__(42)
        assert result is NotImplemented
