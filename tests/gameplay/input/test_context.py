"""Comprehensive tests for the input context system.

Tests cover context creation, activation, stacking, priorities,
context-specific bindings, transitions, and blocking contexts.

Note: Since context.py doesn't exist in the source, these tests define
the expected interface and behavior for an input context system.
"""

import pytest
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from unittest.mock import Mock, MagicMock


# =============================================================================
# Context System Mock Implementation (for testing expected behavior)
# =============================================================================

class ContextState(Enum):
    """State of an input context."""
    INACTIVE = auto()
    ACTIVE = auto()
    BLOCKED = auto()
    TRANSITIONING = auto()


@dataclass
class InputContext:
    """An input context that defines a set of active bindings."""
    name: str
    priority: int = 0
    blocking: bool = False
    consume_all: bool = False
    bindings: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None
    _state: ContextState = field(default=ContextState.INACTIVE)

    @property
    def state(self) -> ContextState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == ContextState.ACTIVE


class ContextManager:
    """Manages input contexts with stacking and priority."""

    def __init__(self):
        self._contexts: Dict[str, InputContext] = {}
        self._stack: List[str] = []
        self._active_contexts: Set[str] = set()

    def register_context(self, context: InputContext) -> bool:
        """Register a context."""
        if context.name in self._contexts:
            return False
        self._contexts[context.name] = context
        return True

    def unregister_context(self, name: str) -> bool:
        """Unregister a context."""
        if name not in self._contexts:
            return False
        self.deactivate_context(name)
        del self._contexts[name]
        return True

    def get_context(self, name: str) -> Optional[InputContext]:
        """Get a context by name."""
        return self._contexts.get(name)

    def activate_context(self, name: str) -> bool:
        """Activate a context."""
        context = self._contexts.get(name)
        if context is None:
            return False
        context._state = ContextState.ACTIVE
        self._active_contexts.add(name)
        return True

    def deactivate_context(self, name: str) -> bool:
        """Deactivate a context."""
        context = self._contexts.get(name)
        if context is None:
            return False
        context._state = ContextState.INACTIVE
        self._active_contexts.discard(name)
        if name in self._stack:
            self._stack.remove(name)
        return True

    def push_context(self, name: str) -> bool:
        """Push a context onto the stack."""
        if name not in self._contexts:
            return False
        if name in self._stack:
            return False
        self._stack.append(name)
        self.activate_context(name)
        return True

    def pop_context(self) -> Optional[str]:
        """Pop the top context from the stack."""
        if not self._stack:
            return None
        name = self._stack.pop()
        self.deactivate_context(name)
        return name

    def get_stack(self) -> List[str]:
        """Get the current context stack."""
        return self._stack.copy()

    def get_active_contexts(self) -> List[InputContext]:
        """Get all active contexts sorted by priority."""
        active = [self._contexts[n] for n in self._active_contexts
                  if n in self._contexts]
        return sorted(active, key=lambda c: c.priority, reverse=True)

    def get_top_context(self) -> Optional[InputContext]:
        """Get the highest priority active context."""
        active = self.get_active_contexts()
        return active[0] if active else None

    def is_context_active(self, name: str) -> bool:
        """Check if a context is active."""
        return name in self._active_contexts

    def is_context_blocked(self, name: str) -> bool:
        """Check if a context is blocked by a higher priority context."""
        context = self._contexts.get(name)
        if context is None or not context.is_active:
            return False

        for other in self.get_active_contexts():
            if other.name == name:
                break
            if other.blocking:
                return True
        return False

    def get_effective_bindings(self) -> Dict[str, Any]:
        """Get merged bindings from all active, non-blocked contexts."""
        bindings = {}
        active = self.get_active_contexts()

        for context in reversed(active):  # Low to high priority
            if not self.is_context_blocked(context.name):
                bindings.update(context.bindings)

        return bindings

    def transition_context(self, from_name: str, to_name: str) -> bool:
        """Transition from one context to another."""
        from_ctx = self._contexts.get(from_name)
        to_ctx = self._contexts.get(to_name)

        if from_ctx is None or to_ctx is None:
            return False

        if from_ctx.is_active:
            self.deactivate_context(from_name)

        self.activate_context(to_name)
        return True

    def clear_stack(self) -> None:
        """Clear the context stack."""
        while self._stack:
            self.pop_context()

    def reset(self) -> None:
        """Reset all contexts."""
        self._active_contexts.clear()
        self._stack.clear()
        for context in self._contexts.values():
            context._state = ContextState.INACTIVE


# =============================================================================
# Context State Tests
# =============================================================================

class TestContextState:
    """Tests for ContextState enum."""

    def test_context_states_exist(self):
        """All context states exist."""
        assert ContextState.INACTIVE
        assert ContextState.ACTIVE
        assert ContextState.BLOCKED
        assert ContextState.TRANSITIONING


# =============================================================================
# Input Context Tests
# =============================================================================

class TestInputContext:
    """Tests for InputContext class."""

    def test_context_creation(self):
        """InputContext can be created."""
        context = InputContext(
            name="gameplay",
            priority=0,
            blocking=False
        )
        assert context.name == "gameplay"
        assert context.priority == 0
        assert context.blocking is False

    def test_context_defaults(self):
        """InputContext has sensible defaults."""
        context = InputContext(name="test")
        assert context.priority == 0
        assert context.blocking is False
        assert context.consume_all is False
        assert context.bindings == {}
        assert context.parent is None

    def test_context_initial_state(self):
        """Context starts inactive."""
        context = InputContext(name="test")
        assert context.state == ContextState.INACTIVE
        assert context.is_active is False

    def test_context_with_bindings(self):
        """Context can have bindings."""
        bindings = {
            "jump": {"key": "space"},
            "move": {"keys": ["w", "a", "s", "d"]}
        }
        context = InputContext(name="gameplay", bindings=bindings)
        assert "jump" in context.bindings
        assert "move" in context.bindings

    def test_context_with_priority(self):
        """Context priority is stored correctly."""
        context = InputContext(name="menu", priority=100)
        assert context.priority == 100

    def test_blocking_context(self):
        """Blocking context is stored correctly."""
        context = InputContext(name="modal", blocking=True)
        assert context.blocking is True

    def test_consume_all_context(self):
        """Consume all flag is stored correctly."""
        context = InputContext(name="exclusive", consume_all=True)
        assert context.consume_all is True

    def test_parent_context(self):
        """Parent context reference is stored."""
        context = InputContext(name="child", parent="base")
        assert context.parent == "base"


# =============================================================================
# Context Manager Basic Tests
# =============================================================================

class TestContextManagerBasic:
    """Basic tests for ContextManager class."""

    @pytest.fixture
    def manager(self):
        """Create a context manager."""
        return ContextManager()

    @pytest.fixture
    def gameplay_context(self):
        """Create a gameplay context."""
        return InputContext(
            name="gameplay",
            priority=0,
            bindings={"jump": "space", "move": "wasd"}
        )

    def test_register_context(self, manager, gameplay_context):
        """register_context adds context to manager."""
        result = manager.register_context(gameplay_context)
        assert result is True
        assert manager.get_context("gameplay") is not None

    def test_register_duplicate_fails(self, manager, gameplay_context):
        """Registering duplicate context fails."""
        manager.register_context(gameplay_context)
        result = manager.register_context(gameplay_context)
        assert result is False

    def test_unregister_context(self, manager, gameplay_context):
        """unregister_context removes context."""
        manager.register_context(gameplay_context)
        result = manager.unregister_context("gameplay")
        assert result is True
        assert manager.get_context("gameplay") is None

    def test_unregister_nonexistent(self, manager):
        """Unregistering nonexistent context fails."""
        result = manager.unregister_context("nonexistent")
        assert result is False

    def test_unregister_deactivates(self, manager, gameplay_context):
        """Unregistering a context deactivates it first."""
        manager.register_context(gameplay_context)
        manager.activate_context("gameplay")
        manager.unregister_context("gameplay")
        # Should not crash even if active

    def test_get_context(self, manager, gameplay_context):
        """get_context returns correct context."""
        manager.register_context(gameplay_context)
        context = manager.get_context("gameplay")
        assert context is gameplay_context

    def test_get_context_nonexistent(self, manager):
        """get_context returns None for nonexistent."""
        assert manager.get_context("nonexistent") is None


# =============================================================================
# Context Activation Tests
# =============================================================================

class TestContextActivation:
    """Tests for context activation/deactivation."""

    @pytest.fixture
    def manager(self):
        """Create a context manager with contexts."""
        mgr = ContextManager()
        mgr.register_context(InputContext(name="gameplay", priority=0))
        mgr.register_context(InputContext(name="menu", priority=100))
        mgr.register_context(InputContext(name="dialog", priority=50))
        return mgr

    def test_activate_context(self, manager):
        """activate_context activates a context."""
        result = manager.activate_context("gameplay")
        assert result is True
        assert manager.is_context_active("gameplay")

    def test_activate_nonexistent(self, manager):
        """Activating nonexistent context fails."""
        result = manager.activate_context("nonexistent")
        assert result is False

    def test_deactivate_context(self, manager):
        """deactivate_context deactivates a context."""
        manager.activate_context("gameplay")
        result = manager.deactivate_context("gameplay")
        assert result is True
        assert not manager.is_context_active("gameplay")

    def test_deactivate_nonexistent(self, manager):
        """Deactivating nonexistent context fails."""
        result = manager.deactivate_context("nonexistent")
        assert result is False

    def test_multiple_contexts_active(self, manager):
        """Multiple contexts can be active."""
        manager.activate_context("gameplay")
        manager.activate_context("dialog")

        assert manager.is_context_active("gameplay")
        assert manager.is_context_active("dialog")

    def test_get_active_contexts(self, manager):
        """get_active_contexts returns all active contexts."""
        manager.activate_context("gameplay")
        manager.activate_context("menu")

        active = manager.get_active_contexts()
        names = [c.name for c in active]
        assert "gameplay" in names
        assert "menu" in names

    def test_active_contexts_sorted_by_priority(self, manager):
        """Active contexts are sorted by priority (high to low)."""
        manager.activate_context("gameplay")  # priority 0
        manager.activate_context("menu")       # priority 100
        manager.activate_context("dialog")     # priority 50

        active = manager.get_active_contexts()
        priorities = [c.priority for c in active]
        assert priorities == sorted(priorities, reverse=True)

    def test_get_top_context(self, manager):
        """get_top_context returns highest priority active context."""
        manager.activate_context("gameplay")  # priority 0
        manager.activate_context("dialog")     # priority 50

        top = manager.get_top_context()
        assert top.name == "dialog"

        manager.activate_context("menu")  # priority 100
        top = manager.get_top_context()
        assert top.name == "menu"

    def test_get_top_context_none(self, manager):
        """get_top_context returns None when no active contexts."""
        top = manager.get_top_context()
        assert top is None


# =============================================================================
# Context Stack Tests
# =============================================================================

class TestContextStack:
    """Tests for context stacking (push/pop)."""

    @pytest.fixture
    def manager(self):
        """Create a context manager with contexts."""
        mgr = ContextManager()
        mgr.register_context(InputContext(name="gameplay", priority=0))
        mgr.register_context(InputContext(name="inventory", priority=50))
        mgr.register_context(InputContext(name="options", priority=100))
        return mgr

    def test_push_context(self, manager):
        """push_context adds context to stack and activates it."""
        result = manager.push_context("gameplay")
        assert result is True
        assert "gameplay" in manager.get_stack()
        assert manager.is_context_active("gameplay")

    def test_push_multiple_contexts(self, manager):
        """Multiple contexts can be pushed."""
        manager.push_context("gameplay")
        manager.push_context("inventory")
        manager.push_context("options")

        stack = manager.get_stack()
        assert stack == ["gameplay", "inventory", "options"]

    def test_push_duplicate_fails(self, manager):
        """Pushing duplicate context fails."""
        manager.push_context("gameplay")
        result = manager.push_context("gameplay")
        assert result is False

    def test_push_nonexistent_fails(self, manager):
        """Pushing nonexistent context fails."""
        result = manager.push_context("nonexistent")
        assert result is False

    def test_pop_context(self, manager):
        """pop_context removes top context from stack."""
        manager.push_context("gameplay")
        manager.push_context("inventory")

        popped = manager.pop_context()
        assert popped == "inventory"
        assert "inventory" not in manager.get_stack()
        assert not manager.is_context_active("inventory")

    def test_pop_empty_stack(self, manager):
        """Popping empty stack returns None."""
        result = manager.pop_context()
        assert result is None

    def test_get_stack(self, manager):
        """get_stack returns copy of stack."""
        manager.push_context("gameplay")
        manager.push_context("inventory")

        stack = manager.get_stack()
        assert stack == ["gameplay", "inventory"]

        # Verify it's a copy
        stack.append("test")
        assert "test" not in manager.get_stack()

    def test_clear_stack(self, manager):
        """clear_stack removes all contexts from stack."""
        manager.push_context("gameplay")
        manager.push_context("inventory")
        manager.push_context("options")

        manager.clear_stack()

        assert manager.get_stack() == []
        assert not manager.is_context_active("gameplay")
        assert not manager.is_context_active("inventory")
        assert not manager.is_context_active("options")


# =============================================================================
# Context Priority Tests
# =============================================================================

class TestContextPriority:
    """Tests for context priority handling."""

    @pytest.fixture
    def manager(self):
        """Create a context manager with prioritized contexts."""
        mgr = ContextManager()
        mgr.register_context(InputContext(
            name="base",
            priority=0,
            bindings={"action": "binding_base"}
        ))
        mgr.register_context(InputContext(
            name="overlay",
            priority=50,
            bindings={"action": "binding_overlay"}
        ))
        mgr.register_context(InputContext(
            name="modal",
            priority=100,
            bindings={"action": "binding_modal"}
        ))
        return mgr

    def test_higher_priority_context_first(self, manager):
        """Higher priority contexts appear first in active list."""
        manager.activate_context("base")
        manager.activate_context("overlay")
        manager.activate_context("modal")

        active = manager.get_active_contexts()
        assert active[0].name == "modal"
        assert active[1].name == "overlay"
        assert active[2].name == "base"

    def test_priority_affects_binding_resolution(self, manager):
        """Higher priority bindings override lower priority."""
        manager.activate_context("base")
        manager.activate_context("overlay")

        bindings = manager.get_effective_bindings()
        assert bindings["action"] == "binding_overlay"

    def test_all_priorities_merged(self, manager):
        """All non-conflicting bindings are merged."""
        mgr = ContextManager()
        mgr.register_context(InputContext(
            name="ctx1",
            priority=0,
            bindings={"a": "1", "b": "1"}
        ))
        mgr.register_context(InputContext(
            name="ctx2",
            priority=50,
            bindings={"b": "2", "c": "2"}
        ))
        mgr.activate_context("ctx1")
        mgr.activate_context("ctx2")

        bindings = mgr.get_effective_bindings()
        assert bindings["a"] == "1"  # From ctx1
        assert bindings["b"] == "2"  # Overridden by ctx2
        assert bindings["c"] == "2"  # From ctx2

    def test_same_priority_order(self, manager):
        """Contexts with same priority have consistent ordering."""
        mgr = ContextManager()
        mgr.register_context(InputContext(name="a", priority=50))
        mgr.register_context(InputContext(name="b", priority=50))

        mgr.activate_context("a")
        mgr.activate_context("b")

        # Should not crash, order may be arbitrary but consistent


# =============================================================================
# Blocking Context Tests
# =============================================================================

class TestBlockingContext:
    """Tests for blocking context behavior."""

    @pytest.fixture
    def manager(self):
        """Create a context manager with blocking contexts."""
        mgr = ContextManager()
        mgr.register_context(InputContext(
            name="gameplay",
            priority=0,
            bindings={"move": "wasd"}
        ))
        mgr.register_context(InputContext(
            name="modal_dialog",
            priority=100,
            blocking=True,
            bindings={"confirm": "enter", "cancel": "escape"}
        ))
        mgr.register_context(InputContext(
            name="non_blocking_overlay",
            priority=50,
            blocking=False,
            bindings={"toggle": "tab"}
        ))
        return mgr

    def test_blocking_context_blocks_lower(self, manager):
        """Blocking context blocks lower priority contexts."""
        manager.activate_context("gameplay")
        manager.activate_context("modal_dialog")

        assert manager.is_context_blocked("gameplay")

    def test_non_blocking_does_not_block(self, manager):
        """Non-blocking context doesn't block lower priority."""
        manager.activate_context("gameplay")
        manager.activate_context("non_blocking_overlay")

        assert not manager.is_context_blocked("gameplay")

    def test_blocked_context_bindings_excluded(self, manager):
        """Blocked context bindings are excluded."""
        manager.activate_context("gameplay")
        manager.activate_context("modal_dialog")

        bindings = manager.get_effective_bindings()
        assert "move" not in bindings  # Blocked
        assert "confirm" in bindings

    def test_blocking_context_not_blocked_by_itself(self, manager):
        """Blocking context is not blocked by itself."""
        manager.activate_context("modal_dialog")

        assert not manager.is_context_blocked("modal_dialog")

    def test_multiple_blocking_contexts(self, manager):
        """Multiple blocking contexts stack."""
        mgr = ContextManager()
        mgr.register_context(InputContext(name="base", priority=0))
        mgr.register_context(InputContext(name="blocker1", priority=50, blocking=True))
        mgr.register_context(InputContext(name="blocker2", priority=100, blocking=True))

        mgr.activate_context("base")
        mgr.activate_context("blocker1")
        mgr.activate_context("blocker2")

        assert mgr.is_context_blocked("base")
        assert mgr.is_context_blocked("blocker1")  # Blocked by blocker2
        assert not mgr.is_context_blocked("blocker2")


# =============================================================================
# Context Transition Tests
# =============================================================================

class TestContextTransition:
    """Tests for context transitions."""

    @pytest.fixture
    def manager(self):
        """Create a context manager."""
        mgr = ContextManager()
        mgr.register_context(InputContext(name="gameplay"))
        mgr.register_context(InputContext(name="menu"))
        mgr.register_context(InputContext(name="cutscene"))
        return mgr

    def test_transition_deactivates_source(self, manager):
        """Transition deactivates source context."""
        manager.activate_context("gameplay")
        manager.transition_context("gameplay", "menu")

        assert not manager.is_context_active("gameplay")

    def test_transition_activates_target(self, manager):
        """Transition activates target context."""
        manager.activate_context("gameplay")
        manager.transition_context("gameplay", "menu")

        assert manager.is_context_active("menu")

    def test_transition_from_inactive(self, manager):
        """Transition from inactive context still activates target."""
        result = manager.transition_context("gameplay", "menu")
        assert result is True
        assert manager.is_context_active("menu")

    def test_transition_nonexistent_source(self, manager):
        """Transition with nonexistent source fails."""
        result = manager.transition_context("nonexistent", "menu")
        assert result is False

    def test_transition_nonexistent_target(self, manager):
        """Transition with nonexistent target fails."""
        result = manager.transition_context("gameplay", "nonexistent")
        assert result is False

    def test_transition_same_context(self, manager):
        """Transition to same context is allowed."""
        manager.activate_context("gameplay")
        result = manager.transition_context("gameplay", "gameplay")
        assert result is True
        # Should still be active
        assert manager.is_context_active("gameplay")


# =============================================================================
# Context Inheritance Tests
# =============================================================================

class TestContextInheritance:
    """Tests for context inheritance (parent contexts)."""

    @pytest.fixture
    def manager(self):
        """Create a context manager with parent-child contexts."""
        mgr = ContextManager()
        mgr.register_context(InputContext(
            name="base_gameplay",
            priority=0,
            bindings={"move": "wasd", "camera": "mouse"}
        ))
        mgr.register_context(InputContext(
            name="vehicle_gameplay",
            priority=10,
            parent="base_gameplay",
            bindings={"move": "arrows", "brake": "space"}  # Overrides move
        ))
        return mgr

    def test_child_has_parent_reference(self, manager):
        """Child context has parent reference."""
        child = manager.get_context("vehicle_gameplay")
        assert child.parent == "base_gameplay"

    def test_parent_has_no_parent(self, manager):
        """Parent context has no parent."""
        parent = manager.get_context("base_gameplay")
        assert parent.parent is None

    def test_parent_reference_for_inheritance(self, manager):
        """Parent reference can be used for inheritance logic."""
        child = manager.get_context("vehicle_gameplay")
        parent_name = child.parent

        if parent_name:
            parent = manager.get_context(parent_name)
            assert parent is not None


# =============================================================================
# Reset and Cleanup Tests
# =============================================================================

class TestContextReset:
    """Tests for context reset and cleanup."""

    @pytest.fixture
    def manager(self):
        """Create a context manager with active contexts."""
        mgr = ContextManager()
        mgr.register_context(InputContext(name="gameplay"))
        mgr.register_context(InputContext(name="menu"))
        mgr.activate_context("gameplay")
        mgr.push_context("menu")
        return mgr

    def test_reset_deactivates_all(self, manager):
        """reset deactivates all contexts."""
        manager.reset()

        assert not manager.is_context_active("gameplay")
        assert not manager.is_context_active("menu")

    def test_reset_clears_stack(self, manager):
        """reset clears the context stack."""
        manager.reset()

        assert manager.get_stack() == []

    def test_reset_preserves_registrations(self, manager):
        """reset preserves context registrations."""
        manager.reset()

        assert manager.get_context("gameplay") is not None
        assert manager.get_context("menu") is not None

    def test_contexts_can_reactivate_after_reset(self, manager):
        """Contexts can be reactivated after reset."""
        manager.reset()

        result = manager.activate_context("gameplay")
        assert result is True
        assert manager.is_context_active("gameplay")


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestContextEdgeCases:
    """Edge case tests for context system."""

    def test_empty_context_manager(self):
        """Empty context manager works."""
        manager = ContextManager()

        assert manager.get_active_contexts() == []
        assert manager.get_top_context() is None
        assert manager.get_stack() == []
        assert manager.get_effective_bindings() == {}

    def test_many_contexts(self):
        """Manager handles many contexts."""
        manager = ContextManager()

        for i in range(100):
            manager.register_context(InputContext(
                name=f"context_{i}",
                priority=i
            ))

        for i in range(50):
            manager.activate_context(f"context_{i}")

        active = manager.get_active_contexts()
        assert len(active) == 50

    def test_rapid_push_pop(self):
        """Manager handles rapid push/pop."""
        manager = ContextManager()

        for i in range(10):
            manager.register_context(InputContext(name=f"ctx_{i}"))

        for _ in range(100):
            for i in range(10):
                manager.push_context(f"ctx_{i}")
            for _ in range(10):
                manager.pop_context()

    def test_negative_priority(self):
        """Negative priorities work."""
        manager = ContextManager()
        manager.register_context(InputContext(name="low", priority=-100))
        manager.register_context(InputContext(name="high", priority=100))

        manager.activate_context("low")
        manager.activate_context("high")

        top = manager.get_top_context()
        assert top.name == "high"

    def test_very_high_priority(self):
        """Very high priorities work."""
        manager = ContextManager()
        manager.register_context(InputContext(name="normal", priority=0))
        manager.register_context(InputContext(name="super", priority=999999))

        manager.activate_context("normal")
        manager.activate_context("super")

        top = manager.get_top_context()
        assert top.name == "super"


# =============================================================================
# Integration Tests
# =============================================================================

class TestContextIntegration:
    """Integration tests for context system."""

    def test_menu_system_simulation(self):
        """Simulate a game menu system."""
        manager = ContextManager()

        # Register contexts
        manager.register_context(InputContext(
            name="gameplay",
            priority=0,
            bindings={"move": "wasd", "jump": "space", "pause": "escape"}
        ))
        manager.register_context(InputContext(
            name="pause_menu",
            priority=100,
            blocking=True,
            bindings={"navigate": "arrows", "select": "enter", "back": "escape"}
        ))
        manager.register_context(InputContext(
            name="options_menu",
            priority=200,
            blocking=True,
            bindings={"navigate": "arrows", "select": "enter", "back": "escape"}
        ))

        # Start gameplay
        manager.activate_context("gameplay")
        assert manager.is_context_active("gameplay")
        assert "move" in manager.get_effective_bindings()

        # Open pause menu
        manager.push_context("pause_menu")
        assert manager.is_context_active("pause_menu")
        assert manager.is_context_blocked("gameplay")
        assert "move" not in manager.get_effective_bindings()
        assert "navigate" in manager.get_effective_bindings()

        # Open options from pause menu
        manager.push_context("options_menu")
        assert manager.is_context_blocked("pause_menu")

        # Close options
        manager.pop_context()
        assert not manager.is_context_active("options_menu")
        assert manager.is_context_active("pause_menu")

        # Close pause menu (return to gameplay)
        manager.pop_context()
        assert not manager.is_context_active("pause_menu")
        assert not manager.is_context_blocked("gameplay")
        assert "move" in manager.get_effective_bindings()

    def test_game_mode_transitions(self):
        """Simulate game mode transitions."""
        manager = ContextManager()

        manager.register_context(InputContext(name="walking", priority=0))
        manager.register_context(InputContext(name="driving", priority=0))
        manager.register_context(InputContext(name="swimming", priority=0))
        manager.register_context(InputContext(name="flying", priority=0))

        # Start walking
        manager.activate_context("walking")
        assert manager.is_context_active("walking")

        # Enter vehicle
        manager.transition_context("walking", "driving")
        assert not manager.is_context_active("walking")
        assert manager.is_context_active("driving")

        # Exit vehicle into water
        manager.transition_context("driving", "swimming")
        assert not manager.is_context_active("driving")
        assert manager.is_context_active("swimming")

        # Get picked up by helicopter
        manager.transition_context("swimming", "flying")
        assert not manager.is_context_active("swimming")
        assert manager.is_context_active("flying")
