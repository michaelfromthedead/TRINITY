"""
Comprehensive tests for the ScreenStack class and related components.

Tests cover:
- Stack operations (push, pop, replace, clear)
- Screen cache management
- History tracking
- Modal and overlay screens
- Factory registration
- Event callbacks
- Edge cases and error handling
"""

from __future__ import annotations

import pytest
from typing import List, Optional, Any
from unittest.mock import Mock, MagicMock, patch
import time

from engine.ui.screens.screen import Screen, ScreenParams, ScreenResult, ScreenState
from engine.ui.screens.screen_stack import (
    ScreenStack,
    ScreenCache,
    StackOperation,
    HistoryEntry,
    ScreenFactory,
    StackEventCallback,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


class MockScreen(Screen):
    """Mock screen implementation for testing."""

    def __init__(self, name: str = "mock_screen") -> None:
        super().__init__(name)
        self.lifecycle_calls: List[str] = []

    def on_create(self) -> None:
        self.lifecycle_calls.append("on_create")

    def on_destroy(self) -> None:
        self.lifecycle_calls.append("on_destroy")

    def on_enter(self) -> None:
        self.lifecycle_calls.append("on_enter")

    def on_exit(self) -> None:
        self.lifecycle_calls.append("on_exit")

    def on_pause(self) -> None:
        self.lifecycle_calls.append("on_pause")

    def on_resume(self) -> None:
        self.lifecycle_calls.append("on_resume")


@pytest.fixture
def stack() -> ScreenStack:
    """Create a fresh screen stack for each test."""
    return ScreenStack()


@pytest.fixture
def screen() -> MockScreen:
    """Create a mock screen for testing."""
    return MockScreen("test_screen")


@pytest.fixture
def populated_stack(stack: ScreenStack) -> ScreenStack:
    """Create a stack with registered factories."""
    stack.register_factory("main_menu", lambda n, p: MockScreen("main_menu"))
    stack.register_factory("settings", lambda n, p: MockScreen("settings"))
    stack.register_factory("game", lambda n, p: MockScreen("game"))
    return stack


# =============================================================================
# SCREEN CACHE TESTS
# =============================================================================


class TestScreenCache:
    """Tests for ScreenCache class."""

    def test_cache_init(self) -> None:
        """Cache should initialize with correct defaults."""
        cache = ScreenCache()
        assert cache.enabled is True
        assert cache.max_size == 10
        assert cache.size == 0

    def test_cache_init_with_max_size(self) -> None:
        """Cache should accept custom max size."""
        cache = ScreenCache(max_size=5)
        assert cache.max_size == 5

    def test_cache_put_and_get(self) -> None:
        """Should be able to store and retrieve screens."""
        cache = ScreenCache()
        screen = MockScreen("test")

        cache.put(screen)
        assert cache.get("test") is screen

    def test_cache_get_nonexistent(self) -> None:
        """Getting nonexistent screen should return None."""
        cache = ScreenCache()
        assert cache.get("nonexistent") is None

    def test_cache_contains(self) -> None:
        """Should correctly check for screen existence."""
        cache = ScreenCache()
        screen = MockScreen("test")

        cache.put(screen)
        assert cache.contains("test") is True
        assert cache.contains("other") is False

    def test_cache_remove(self) -> None:
        """Should be able to remove screens."""
        cache = ScreenCache()
        screen = MockScreen("test")

        cache.put(screen)
        removed = cache.remove("test")

        assert removed is screen
        assert cache.contains("test") is False

    def test_cache_remove_nonexistent(self) -> None:
        """Removing nonexistent screen should return None."""
        cache = ScreenCache()
        assert cache.remove("nonexistent") is None

    def test_cache_clear(self) -> None:
        """Should be able to clear all cached screens."""
        cache = ScreenCache()
        cache.put(MockScreen("s1"))
        cache.put(MockScreen("s2"))

        cache.clear()

        assert cache.size == 0
        assert cache.get("s1") is None
        assert cache.get("s2") is None

    def test_cache_lru_eviction(self) -> None:
        """Should evict least recently used screens when full."""
        cache = ScreenCache(max_size=2)

        cache.put(MockScreen("s1"))
        cache.put(MockScreen("s2"))
        cache.put(MockScreen("s3"))

        # s1 should have been evicted
        assert cache.get("s1") is None
        assert cache.get("s2") is not None
        assert cache.get("s3") is not None

    def test_cache_lru_access_updates_order(self) -> None:
        """Accessing screen should update its LRU position."""
        cache = ScreenCache(max_size=2)

        cache.put(MockScreen("s1"))
        cache.put(MockScreen("s2"))

        # Access s1 to make it recently used
        cache.get("s1")

        cache.put(MockScreen("s3"))

        # s2 should have been evicted (was least recently used)
        assert cache.get("s1") is not None
        assert cache.get("s2") is None
        assert cache.get("s3") is not None

    def test_cache_disabled(self) -> None:
        """Disabled cache should not store screens."""
        cache = ScreenCache()
        cache.enabled = False

        cache.put(MockScreen("test"))
        assert cache.get("test") is None

    def test_disabling_cache_clears_it(self) -> None:
        """Disabling cache should clear existing entries."""
        cache = ScreenCache()
        cache.put(MockScreen("test"))

        cache.enabled = False

        assert cache.size == 0

    def test_cache_max_size_setter(self) -> None:
        """Setting max_size should evict if needed."""
        cache = ScreenCache(max_size=5)

        for i in range(5):
            cache.put(MockScreen(f"s{i}"))

        cache.max_size = 2

        assert cache.size == 2

    def test_cache_max_size_minimum(self) -> None:
        """Max size should be at least 1."""
        cache = ScreenCache()
        cache.max_size = 0
        assert cache.max_size == 1


# =============================================================================
# STACK INITIALIZATION TESTS
# =============================================================================


class TestStackInitialization:
    """Tests for screen stack initialization."""

    def test_stack_starts_empty(self, stack: ScreenStack) -> None:
        """Stack should start empty."""
        assert stack.is_empty is True
        assert stack.count == 0

    def test_stack_has_no_top(self, stack: ScreenStack) -> None:
        """Empty stack should have no top."""
        assert stack.top is None

    def test_stack_has_no_bottom(self, stack: ScreenStack) -> None:
        """Empty stack should have no bottom."""
        assert stack.bottom is None

    def test_stack_not_transitioning(self, stack: ScreenStack) -> None:
        """Stack should not be transitioning initially."""
        assert stack.is_transitioning is False

    def test_stack_has_cache(self, stack: ScreenStack) -> None:
        """Stack should have a cache."""
        assert stack.cache is not None

    def test_stack_history_enabled_by_default(self, stack: ScreenStack) -> None:
        """History tracking should be enabled by default."""
        assert stack.history_enabled is True


# =============================================================================
# FACTORY REGISTRATION TESTS
# =============================================================================


class TestFactoryRegistration:
    """Tests for screen factory registration."""

    def test_register_factory(self, stack: ScreenStack) -> None:
        """Should register factory for screen creation."""
        factory = Mock(return_value=MockScreen("test"))
        stack.register_factory("test", factory)

        assert stack.has_factory("test") is True

    def test_unregister_factory(self, stack: ScreenStack) -> None:
        """Should be able to unregister factories."""
        stack.register_factory("test", Mock())

        result = stack.unregister_factory("test")

        assert result is True
        assert stack.has_factory("test") is False

    def test_unregister_nonexistent_factory(self, stack: ScreenStack) -> None:
        """Unregistering nonexistent factory should return False."""
        result = stack.unregister_factory("nonexistent")
        assert result is False

    def test_get_registered_names(self, stack: ScreenStack) -> None:
        """Should get list of registered screen names."""
        stack.register_factory("screen1", Mock())
        stack.register_factory("screen2", Mock())

        names = stack.get_registered_names()

        assert "screen1" in names
        assert "screen2" in names


# =============================================================================
# PUSH OPERATION TESTS
# =============================================================================


class TestPushOperation:
    """Tests for push operation."""

    def test_push_screen_by_name(self, populated_stack: ScreenStack) -> None:
        """Should push screen by name."""
        screen = populated_stack.push("main_menu")

        assert screen is not None
        assert populated_stack.count == 1
        assert populated_stack.top is screen

    def test_push_screen_instance(self, stack: ScreenStack) -> None:
        """Should push screen instance directly."""
        screen = MockScreen("direct")

        result = stack.push(screen)

        assert result is screen
        assert stack.count == 1

    def test_push_with_params(self, populated_stack: ScreenStack) -> None:
        """Should pass params to pushed screen."""
        params = ScreenParams(data={"level": 5})

        screen = populated_stack.push("game", params=params)

        assert screen is not None
        assert screen.params.get("level") == 5

    def test_push_returns_none_for_unknown_screen(self, stack: ScreenStack) -> None:
        """Pushing unknown screen name should return None."""
        result = stack.push("unknown")
        assert result is None

    def test_push_triggers_enter(self, populated_stack: ScreenStack) -> None:
        """Push should trigger enter lifecycle on new screen."""
        screen = populated_stack.push("main_menu")

        assert screen is not None
        assert "on_enter" in screen.lifecycle_calls

    def test_push_pauses_previous_screen(self, populated_stack: ScreenStack) -> None:
        """Push should pause the previous top screen."""
        first = populated_stack.push("main_menu")
        second = populated_stack.push("settings")

        assert first is not None
        assert "on_pause" in first.lifecycle_calls

    def test_push_sets_stack_reference(self, populated_stack: ScreenStack) -> None:
        """Push should set stack reference on screen."""
        screen = populated_stack.push("main_menu")

        assert screen is not None
        assert screen.stack is populated_stack

    def test_push_prevents_duplicates_by_default(self, populated_stack: ScreenStack) -> None:
        """Push should prevent duplicate screens by default."""
        populated_stack.push("main_menu")
        second = populated_stack.push("main_menu")

        assert second is None
        assert populated_stack.count == 1

    def test_push_allows_duplicates_when_enabled(self, populated_stack: ScreenStack) -> None:
        """Push should allow duplicates when enabled."""
        populated_stack.allow_duplicate_screens = True

        populated_stack.push("main_menu")
        second = populated_stack.push("main_menu")

        assert second is not None
        assert populated_stack.count == 2

    def test_push_records_history(self, populated_stack: ScreenStack) -> None:
        """Push should record in history."""
        populated_stack.push("main_menu")

        assert len(populated_stack.history) == 1
        assert populated_stack.history[0].screen_name == "main_menu"
        assert populated_stack.history[0].operation == StackOperation.PUSH

    def test_push_uses_cached_screen(self, populated_stack: ScreenStack) -> None:
        """Push should use cached screen if available."""
        # Push and pop to cache a screen
        screen = populated_stack.push("main_menu")
        populated_stack.pop()

        # Push again - should get cached screen
        cached = populated_stack.push("main_menu")

        assert cached is screen

    def test_push_bypasses_cache_when_disabled(self, populated_stack: ScreenStack) -> None:
        """Push should create new screen when cache disabled."""
        screen = populated_stack.push("main_menu")
        populated_stack.pop()

        # Push with use_cache=False
        new_screen = populated_stack.push("main_menu", use_cache=False)

        assert new_screen is not screen


# =============================================================================
# POP OPERATION TESTS
# =============================================================================


class TestPopOperation:
    """Tests for pop operation."""

    def test_pop_empty_stack(self, stack: ScreenStack) -> None:
        """Popping empty stack should return None."""
        result = stack.pop()
        assert result is None

    def test_pop_returns_top_screen(self, populated_stack: ScreenStack) -> None:
        """Pop should return the top screen."""
        pushed = populated_stack.push("main_menu")
        popped = populated_stack.pop()

        assert popped is pushed

    def test_pop_removes_screen(self, populated_stack: ScreenStack) -> None:
        """Pop should remove screen from stack."""
        populated_stack.push("main_menu")
        populated_stack.pop()

        assert populated_stack.count == 0

    def test_pop_triggers_exit(self, populated_stack: ScreenStack) -> None:
        """Pop should trigger exit lifecycle on screen."""
        screen = populated_stack.push("main_menu")
        populated_stack.pop()

        assert screen is not None
        assert "on_exit" in screen.lifecycle_calls

    def test_pop_resumes_previous_screen(self, populated_stack: ScreenStack) -> None:
        """Pop should resume the previous screen."""
        first = populated_stack.push("main_menu")
        populated_stack.push("settings")

        assert first is not None
        # Force first to be paused
        first._state = ScreenState.PAUSED

        populated_stack.pop()

        assert "on_resume" in first.lifecycle_calls

    def test_pop_with_result(self, populated_stack: ScreenStack) -> None:
        """Pop should set result on the popped screen."""
        screen = populated_stack.push("main_menu")
        result = ScreenResult(success=True, data={"answer": 42})

        populated_stack.pop(result)

        assert screen is not None
        assert screen.result is not None
        assert screen.result.get("answer") == 42

    def test_pop_records_history(self, populated_stack: ScreenStack) -> None:
        """Pop should record in history."""
        populated_stack.push("main_menu")
        populated_stack.pop()

        # Should have PUSH and POP entries
        assert len(populated_stack.history) == 2
        assert populated_stack.history[-1].operation == StackOperation.POP

    def test_pop_caches_screen(self, populated_stack: ScreenStack) -> None:
        """Pop should cache the screen."""
        screen = populated_stack.push("main_menu")
        populated_stack.pop()

        assert populated_stack.cache.contains("main_menu") is True


# =============================================================================
# REPLACE OPERATION TESTS
# =============================================================================


class TestReplaceOperation:
    """Tests for replace operation."""

    def test_replace_on_empty_stack_pushes(self, populated_stack: ScreenStack) -> None:
        """Replace on empty stack should just push."""
        screen = populated_stack.replace("main_menu")

        assert screen is not None
        assert populated_stack.count == 1

    def test_replace_replaces_top_screen(self, populated_stack: ScreenStack) -> None:
        """Replace should replace the top screen."""
        old = populated_stack.push("main_menu")
        new = populated_stack.replace("settings")

        assert new is not None
        assert populated_stack.count == 1
        assert populated_stack.top is new
        assert populated_stack.top is not old

    def test_replace_triggers_exit_on_old_screen(self, populated_stack: ScreenStack) -> None:
        """Replace should trigger exit on old screen."""
        old = populated_stack.push("main_menu")
        populated_stack.replace("settings")

        assert old is not None
        assert "on_exit" in old.lifecycle_calls

    def test_replace_triggers_enter_on_new_screen(self, populated_stack: ScreenStack) -> None:
        """Replace should trigger enter on new screen."""
        populated_stack.push("main_menu")
        new = populated_stack.replace("settings")

        assert new is not None
        assert "on_enter" in new.lifecycle_calls

    def test_replace_with_params(self, populated_stack: ScreenStack) -> None:
        """Replace should pass params to new screen."""
        populated_stack.push("main_menu")
        params = ScreenParams(data={"tab": "audio"})

        new = populated_stack.replace("settings", params=params)

        assert new is not None
        assert new.params.get("tab") == "audio"

    def test_replace_records_history(self, populated_stack: ScreenStack) -> None:
        """Replace should record in history."""
        populated_stack.push("main_menu")
        populated_stack.replace("settings")

        assert any(e.operation == StackOperation.REPLACE for e in populated_stack.history)

    def test_replace_caches_old_screen(self, populated_stack: ScreenStack) -> None:
        """Replace should cache the old screen."""
        populated_stack.push("main_menu")
        populated_stack.replace("settings")

        assert populated_stack.cache.contains("main_menu") is True


# =============================================================================
# CLEAR OPERATION TESTS
# =============================================================================


class TestClearOperation:
    """Tests for clear operation."""

    def test_clear_empty_stack(self, stack: ScreenStack) -> None:
        """Clear on empty stack should return empty list."""
        result = stack.clear()
        assert result == []

    def test_clear_removes_all_screens(self, populated_stack: ScreenStack) -> None:
        """Clear should remove all screens."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")
        populated_stack.push("game")

        cleared = populated_stack.clear()

        assert len(cleared) == 3
        assert populated_stack.count == 0
        assert populated_stack.is_empty is True

    def test_clear_triggers_exit_on_all_screens(self, populated_stack: ScreenStack) -> None:
        """Clear should trigger exit on all screens."""
        s1 = populated_stack.push("main_menu")
        s2 = populated_stack.push("settings")

        populated_stack.clear()

        assert s1 is not None and "on_exit" in s1.lifecycle_calls
        assert s2 is not None and "on_exit" in s2.lifecycle_calls

    def test_clear_records_history(self, populated_stack: ScreenStack) -> None:
        """Clear should record in history."""
        populated_stack.push("main_menu")
        populated_stack.clear()

        assert any(e.operation == StackOperation.CLEAR for e in populated_stack.history)

    def test_clear_caches_all_screens(self, populated_stack: ScreenStack) -> None:
        """Clear should cache all screens."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")

        populated_stack.clear()

        assert populated_stack.cache.contains("main_menu") is True
        assert populated_stack.cache.contains("settings") is True


# =============================================================================
# POP_TO OPERATION TESTS
# =============================================================================


class TestPopToOperation:
    """Tests for pop_to operation."""

    def test_pop_to_target_screen(self, populated_stack: ScreenStack) -> None:
        """pop_to should pop screens until target is on top."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")
        populated_stack.push("game")

        popped = populated_stack.pop_to("main_menu")

        assert len(popped) == 2
        assert populated_stack.top is not None
        assert populated_stack.top.name == "main_menu"

    def test_pop_to_nonexistent_target(self, populated_stack: ScreenStack) -> None:
        """pop_to should return empty list for nonexistent target."""
        populated_stack.push("main_menu")

        popped = populated_stack.pop_to("nonexistent")

        assert popped == []
        assert populated_stack.count == 1

    def test_pop_to_resumes_target(self, populated_stack: ScreenStack) -> None:
        """pop_to should resume the target screen."""
        target = populated_stack.push("main_menu")
        populated_stack.push("settings")

        assert target is not None
        target._state = ScreenState.PAUSED

        populated_stack.pop_to("main_menu")

        assert "on_resume" in target.lifecycle_calls


# =============================================================================
# POP_TO_ROOT OPERATION TESTS
# =============================================================================


class TestPopToRootOperation:
    """Tests for pop_to_root operation."""

    def test_pop_to_root(self, populated_stack: ScreenStack) -> None:
        """pop_to_root should pop all but the bottom screen."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")
        populated_stack.push("game")

        popped = populated_stack.pop_to_root()

        assert len(popped) == 2
        assert populated_stack.count == 1
        assert populated_stack.top is not None
        assert populated_stack.top.name == "main_menu"

    def test_pop_to_root_with_single_screen(self, populated_stack: ScreenStack) -> None:
        """pop_to_root should do nothing with single screen."""
        populated_stack.push("main_menu")

        popped = populated_stack.pop_to_root()

        assert popped == []
        assert populated_stack.count == 1


# =============================================================================
# SWAP OPERATION TESTS
# =============================================================================


class TestSwapOperation:
    """Tests for swap operation."""

    def test_swap_top_two_screens(self, populated_stack: ScreenStack) -> None:
        """Swap should swap the top two screens."""
        populated_stack.push("main_menu")
        s2 = populated_stack.push("settings")

        result = populated_stack.swap()

        assert result is True
        assert populated_stack.top is not None
        assert populated_stack.top.name == "main_menu"
        assert populated_stack.get(0) is not None
        assert populated_stack.get(0).name == "settings"

    def test_swap_with_less_than_two_screens(self, populated_stack: ScreenStack) -> None:
        """Swap should fail with less than two screens."""
        populated_stack.push("main_menu")

        result = populated_stack.swap()

        assert result is False

    def test_swap_pauses_old_top(self, populated_stack: ScreenStack) -> None:
        """Swap should pause the old top screen."""
        populated_stack.push("main_menu")
        old_top = populated_stack.push("settings")

        # Set old_top to active for pause to work
        old_top._state = ScreenState.ACTIVE

        populated_stack.swap()

        assert old_top is not None
        assert "on_pause" in old_top.lifecycle_calls

    def test_swap_resumes_new_top(self, populated_stack: ScreenStack) -> None:
        """Swap should resume the new top screen."""
        new_top = populated_stack.push("main_menu")
        populated_stack.push("settings")

        assert new_top is not None
        new_top._state = ScreenState.PAUSED

        populated_stack.swap()

        assert "on_resume" in new_top.lifecycle_calls


# =============================================================================
# QUERY OPERATION TESTS
# =============================================================================


class TestQueryOperations:
    """Tests for query operations."""

    def test_get_by_index(self, populated_stack: ScreenStack) -> None:
        """Should get screen by index."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")

        assert populated_stack.get(0) is not None
        assert populated_stack.get(0).name == "main_menu"
        assert populated_stack.get(1) is not None
        assert populated_stack.get(1).name == "settings"

    def test_get_invalid_index(self, populated_stack: ScreenStack) -> None:
        """Should return None for invalid index."""
        populated_stack.push("main_menu")

        assert populated_stack.get(-1) is None
        assert populated_stack.get(100) is None

    def test_get_by_name(self, populated_stack: ScreenStack) -> None:
        """Should get screen by name."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")

        screen = populated_stack.get_by_name("main_menu")

        assert screen is not None
        assert screen.name == "main_menu"

    def test_get_by_name_returns_topmost(self, populated_stack: ScreenStack) -> None:
        """get_by_name should return topmost screen with that name."""
        populated_stack.allow_duplicate_screens = True
        first = populated_stack.push("main_menu")
        second = populated_stack.push("main_menu")

        result = populated_stack.get_by_name("main_menu")

        assert result is second

    def test_contains(self, populated_stack: ScreenStack) -> None:
        """Should check if screen is in stack."""
        populated_stack.push("main_menu")

        assert populated_stack.contains("main_menu") is True
        assert populated_stack.contains("settings") is False

    def test_index_of(self, populated_stack: ScreenStack) -> None:
        """Should get index of screen by name."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")

        assert populated_stack.index_of("main_menu") == 0
        assert populated_stack.index_of("settings") == 1
        assert populated_stack.index_of("unknown") == -1

    def test_get_screens_above(self, populated_stack: ScreenStack) -> None:
        """Should get screens above a named screen."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")
        populated_stack.push("game")

        above = populated_stack.get_screens_above("main_menu")

        assert len(above) == 2
        assert above[0].name == "settings"
        assert above[1].name == "game"

    def test_get_screens_below(self, populated_stack: ScreenStack) -> None:
        """Should get screens below a named screen."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")
        populated_stack.push("game")

        below = populated_stack.get_screens_below("game")

        assert len(below) == 2
        assert below[0].name == "main_menu"
        assert below[1].name == "settings"


# =============================================================================
# MODAL AND OVERLAY TESTS
# =============================================================================


class TestModalAndOverlay:
    """Tests for modal and overlay screen operations."""

    def test_push_modal_sets_flags(self, populated_stack: ScreenStack) -> None:
        """push_modal should set modal-specific flags."""
        screen = populated_stack.push_modal("settings")

        assert screen is not None
        assert screen.is_modal is True
        assert screen.blocks_input is True
        assert screen.is_overlay is True
        assert screen.pause_below is False

    def test_push_overlay_sets_flags(self, populated_stack: ScreenStack) -> None:
        """push_overlay should set overlay-specific flags."""
        screen = populated_stack.push_overlay("settings")

        assert screen is not None
        assert screen.is_modal is False
        assert screen.blocks_input is False
        assert screen.is_overlay is True
        assert screen.pause_below is False


# =============================================================================
# EVENT CALLBACK TESTS
# =============================================================================


class TestEventCallbacks:
    """Tests for stack event callbacks."""

    def test_add_event_callback(self, populated_stack: ScreenStack) -> None:
        """Should invoke event callbacks on operations."""
        callback = Mock()
        populated_stack.add_event_callback(callback)

        populated_stack.push("main_menu")

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] is populated_stack
        assert args[1] == StackOperation.PUSH

    def test_remove_event_callback(self, populated_stack: ScreenStack) -> None:
        """Should be able to remove event callbacks."""
        callback = Mock()
        populated_stack.add_event_callback(callback)
        result = populated_stack.remove_event_callback(callback)

        assert result is True

        populated_stack.push("main_menu")
        callback.assert_not_called()

    def test_remove_nonexistent_callback(self, stack: ScreenStack) -> None:
        """Removing nonexistent callback should return False."""
        callback = Mock()
        result = stack.remove_event_callback(callback)
        assert result is False


# =============================================================================
# HISTORY TESTS
# =============================================================================


class TestHistory:
    """Tests for navigation history tracking."""

    def test_history_limit(self, populated_stack: ScreenStack) -> None:
        """History should respect the limit."""
        populated_stack.history_limit = 3

        for i in range(10):
            populated_stack.push(f"screen{i % 3}")
            populated_stack.pop()

        assert len(populated_stack.history) <= 3

    def test_history_disabled(self, populated_stack: ScreenStack) -> None:
        """History should not be recorded when disabled."""
        populated_stack.history_enabled = False

        populated_stack.push("main_menu")

        assert len(populated_stack.history) == 0

    def test_clear_history(self, populated_stack: ScreenStack) -> None:
        """Should be able to clear history."""
        populated_stack.push("main_menu")
        populated_stack.clear_history()

        assert len(populated_stack.history) == 0

    def test_get_history_entry(self, populated_stack: ScreenStack) -> None:
        """Should get history entry by index."""
        populated_stack.push("main_menu")

        entry = populated_stack.get_history_entry(0)

        assert entry is not None
        assert entry.screen_name == "main_menu"

    def test_get_last_history_entry(self, populated_stack: ScreenStack) -> None:
        """Should get most recent history entry."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")

        entry = populated_stack.get_last_history_entry()

        assert entry is not None
        assert entry.screen_name == "settings"


# =============================================================================
# BACK NAVIGATION TESTS
# =============================================================================


class TestBackNavigation:
    """Tests for back navigation."""

    def test_back_on_empty_stack(self, stack: ScreenStack) -> None:
        """Back on empty stack should return False."""
        result = stack.back()
        assert result is False

    def test_back_pops_screen(self, populated_stack: ScreenStack) -> None:
        """Back should pop the top screen."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")

        result = populated_stack.back()

        assert result is True
        assert populated_stack.count == 1

    def test_back_respects_can_go_back(self, populated_stack: ScreenStack) -> None:
        """Back should respect can_go_back flag."""
        screen = populated_stack.push("main_menu")
        assert screen is not None
        screen.can_go_back = False

        result = populated_stack.back()

        assert result is False
        assert populated_stack.count == 1


# =============================================================================
# UPDATE TESTS
# =============================================================================


class TestStackUpdate:
    """Tests for stack update method."""

    def test_update_active_screens(self, populated_stack: ScreenStack) -> None:
        """Update should update active screens."""
        screen = populated_stack.push("main_menu")
        assert screen is not None
        screen._state = ScreenState.ACTIVE

        populated_stack.update(0.016)

        # MockScreen doesn't track update calls, but we verify no errors


# =============================================================================
# STRING REPRESENTATION TESTS
# =============================================================================


class TestStringRepresentation:
    """Tests for string representation."""

    def test_repr_empty(self, stack: ScreenStack) -> None:
        """__repr__ should work for empty stack."""
        repr_str = repr(stack)
        assert "ScreenStack" in repr_str

    def test_str_empty(self, stack: ScreenStack) -> None:
        """__str__ should indicate empty stack."""
        str_str = str(stack)
        assert "empty" in str_str

    def test_str_with_screens(self, populated_stack: ScreenStack) -> None:
        """__str__ should show screen names."""
        populated_stack.push("main_menu")
        populated_stack.push("settings")

        str_str = str(populated_stack)

        assert "main_menu" in str_str
        assert "settings" in str_str
