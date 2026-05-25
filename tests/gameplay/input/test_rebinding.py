"""Comprehensive tests for the input rebinding system.

Tests cover runtime rebinding, rebinding validation, persistence,
conflict handling, reset to defaults, and UI integration.

Note: Since the rebinding system is part of the broader input system,
these tests define the expected interface for a rebinding system.
"""

import pytest
import json
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from unittest.mock import Mock, MagicMock, patch


# =============================================================================
# Rebinding System Mock Implementation (for testing expected behavior)
# =============================================================================

class ConflictResolution(Enum):
    """How to handle binding conflicts."""
    REJECT = auto()      # Reject the rebind
    SWAP = auto()        # Swap bindings
    DUPLICATE = auto()   # Allow duplicate bindings
    UNBIND = auto()      # Unbind the conflicting action


class RebindResult(Enum):
    """Result of a rebind operation."""
    SUCCESS = auto()
    CONFLICT = auto()
    RESERVED = auto()
    INVALID = auto()
    CANCELLED = auto()


@dataclass
class RebindEvent:
    """Event for rebind operations."""
    action_name: str
    old_binding: Optional[str]
    new_binding: str
    result: RebindResult
    conflict_action: Optional[str] = None


@dataclass
class BindingProfile:
    """A saved set of bindings."""
    name: str
    bindings: Dict[str, List[str]]
    description: str = ""
    is_default: bool = False


class RebindingSystem:
    """Manages input rebinding."""

    # Reserved bindings that cannot be changed
    RESERVED_BINDINGS = frozenset({"escape"})

    def __init__(self, conflict_resolution: ConflictResolution = ConflictResolution.REJECT):
        self._bindings: Dict[str, List[str]] = {}
        self._defaults: Dict[str, List[str]] = {}
        self._conflict_resolution = conflict_resolution
        self._reserved = set(self.RESERVED_BINDINGS)
        self._max_bindings_per_action = 4
        self._profiles: Dict[str, BindingProfile] = {}
        self._current_profile: Optional[str] = None
        self._listeners: List[callable] = []

        # Rebinding mode state
        self._rebind_mode = False
        self._rebind_action: Optional[str] = None
        self._rebind_slot: int = 0

    @property
    def conflict_resolution(self) -> ConflictResolution:
        return self._conflict_resolution

    @conflict_resolution.setter
    def conflict_resolution(self, value: ConflictResolution):
        self._conflict_resolution = value

    @property
    def is_rebind_mode_active(self) -> bool:
        return self._rebind_mode

    def register_action(self, action_name: str, default_bindings: List[str]) -> bool:
        """Register an action with default bindings."""
        if action_name in self._bindings:
            return False

        self._defaults[action_name] = default_bindings.copy()
        self._bindings[action_name] = default_bindings.copy()
        return True

    def unregister_action(self, action_name: str) -> bool:
        """Unregister an action."""
        if action_name not in self._bindings:
            return False

        del self._bindings[action_name]
        del self._defaults[action_name]
        return True

    def get_bindings(self, action_name: str) -> List[str]:
        """Get current bindings for an action."""
        return self._bindings.get(action_name, []).copy()

    def get_default_bindings(self, action_name: str) -> List[str]:
        """Get default bindings for an action."""
        return self._defaults.get(action_name, []).copy()

    def get_all_bindings(self) -> Dict[str, List[str]]:
        """Get all current bindings."""
        return {k: v.copy() for k, v in self._bindings.items()}

    def get_action_for_binding(self, binding: str) -> Optional[str]:
        """Get action that uses a binding."""
        for action, bindings in self._bindings.items():
            if binding in bindings:
                return action
        return None

    def rebind(
        self,
        action_name: str,
        new_binding: str,
        slot: int = 0
    ) -> Tuple[RebindResult, Optional[str]]:
        """
        Rebind an action to a new input.

        Returns:
            Tuple of (result, conflicting_action_name or None)
        """
        if action_name not in self._bindings:
            return (RebindResult.INVALID, None)

        # Check reserved
        if new_binding in self._reserved:
            return (RebindResult.RESERVED, None)

        # Check for conflicts
        conflict_action = self.get_action_for_binding(new_binding)
        if conflict_action and conflict_action != action_name:
            return self._handle_conflict(action_name, new_binding, slot, conflict_action)

        # Perform rebind
        return self._perform_rebind(action_name, new_binding, slot)

    def _handle_conflict(
        self,
        action_name: str,
        new_binding: str,
        slot: int,
        conflict_action: str
    ) -> Tuple[RebindResult, Optional[str]]:
        """Handle a binding conflict."""
        if self._conflict_resolution == ConflictResolution.REJECT:
            return (RebindResult.CONFLICT, conflict_action)

        elif self._conflict_resolution == ConflictResolution.SWAP:
            # Get old binding from target action
            old_binding = None
            bindings = self._bindings[action_name]
            if slot < len(bindings):
                old_binding = bindings[slot]

            # Remove new binding from conflict action
            self._bindings[conflict_action].remove(new_binding)

            # Add old binding to conflict action (if we had one)
            if old_binding:
                self._bindings[conflict_action].append(old_binding)

            return self._perform_rebind(action_name, new_binding, slot)

        elif self._conflict_resolution == ConflictResolution.UNBIND:
            # Remove from conflict action
            self._bindings[conflict_action].remove(new_binding)
            return self._perform_rebind(action_name, new_binding, slot)

        elif self._conflict_resolution == ConflictResolution.DUPLICATE:
            # Allow duplicate
            return self._perform_rebind(action_name, new_binding, slot)

        return (RebindResult.INVALID, None)

    def _perform_rebind(
        self,
        action_name: str,
        new_binding: str,
        slot: int
    ) -> Tuple[RebindResult, Optional[str]]:
        """Perform the actual rebind."""
        bindings = self._bindings[action_name]

        # Validate slot
        if slot < 0 or slot > self._max_bindings_per_action:
            return (RebindResult.INVALID, None)

        # Remove binding if already bound to this action
        if new_binding in bindings:
            bindings.remove(new_binding)

        # Extend list if needed
        while len(bindings) <= slot:
            bindings.append("")

        # Set the binding
        old_binding = bindings[slot] if slot < len(bindings) else None
        bindings[slot] = new_binding

        # Clean up empty slots
        while bindings and bindings[-1] == "":
            bindings.pop()

        # Notify listeners
        event = RebindEvent(
            action_name=action_name,
            old_binding=old_binding,
            new_binding=new_binding,
            result=RebindResult.SUCCESS
        )
        self._notify_listeners(event)

        return (RebindResult.SUCCESS, None)

    def unbind(self, action_name: str, slot: int = -1) -> bool:
        """
        Unbind an action.

        If slot is -1, unbind all. Otherwise unbind specific slot.
        """
        if action_name not in self._bindings:
            return False

        bindings = self._bindings[action_name]

        if slot == -1:
            # Unbind all
            old_bindings = bindings.copy()
            bindings.clear()
            for old in old_bindings:
                self._notify_listeners(RebindEvent(
                    action_name=action_name,
                    old_binding=old,
                    new_binding="",
                    result=RebindResult.SUCCESS
                ))
        else:
            if slot < 0 or slot >= len(bindings):
                return False
            old = bindings[slot]
            bindings.pop(slot)
            self._notify_listeners(RebindEvent(
                action_name=action_name,
                old_binding=old,
                new_binding="",
                result=RebindResult.SUCCESS
            ))

        return True

    def add_binding(self, action_name: str, binding: str) -> Tuple[RebindResult, Optional[str]]:
        """Add a binding without replacing existing ones."""
        if action_name not in self._bindings:
            return (RebindResult.INVALID, None)

        bindings = self._bindings[action_name]
        if len(bindings) >= self._max_bindings_per_action:
            return (RebindResult.INVALID, None)

        # Use next available slot
        slot = len(bindings)
        return self.rebind(action_name, binding, slot)

    def reset_action(self, action_name: str) -> bool:
        """Reset an action to default bindings."""
        if action_name not in self._bindings:
            return False

        self._bindings[action_name] = self._defaults[action_name].copy()
        return True

    def reset_all(self) -> None:
        """Reset all bindings to defaults."""
        for action_name in self._bindings:
            self._bindings[action_name] = self._defaults[action_name].copy()

    def is_binding_reserved(self, binding: str) -> bool:
        """Check if a binding is reserved."""
        return binding in self._reserved

    def add_reserved_binding(self, binding: str) -> None:
        """Add a reserved binding."""
        self._reserved.add(binding)

    def remove_reserved_binding(self, binding: str) -> bool:
        """Remove a reserved binding."""
        if binding in self.RESERVED_BINDINGS:
            return False  # Cannot remove built-in reserved
        self._reserved.discard(binding)
        return True

    def is_binding_valid(self, binding: str) -> bool:
        """Check if a binding string is valid."""
        if not binding or not isinstance(binding, str):
            return False
        if binding.strip() == "":
            return False
        return True

    def get_conflicts(self, binding: str) -> List[str]:
        """Get all actions that conflict with a binding."""
        conflicts = []
        for action, bindings in self._bindings.items():
            if binding in bindings:
                conflicts.append(action)
        return conflicts

    # ==========================================================================
    # Rebind Mode (for UI)
    # ==========================================================================

    def start_rebind_mode(self, action_name: str, slot: int = 0) -> bool:
        """Start listening for a new binding."""
        if action_name not in self._bindings:
            return False

        self._rebind_mode = True
        self._rebind_action = action_name
        self._rebind_slot = slot
        return True

    def cancel_rebind_mode(self) -> None:
        """Cancel rebind mode."""
        self._rebind_mode = False
        self._rebind_action = None
        self._rebind_slot = 0

    def submit_rebind(self, binding: str) -> Tuple[RebindResult, Optional[str]]:
        """Submit the captured input for rebinding."""
        if not self._rebind_mode:
            return (RebindResult.CANCELLED, None)

        result = self.rebind(self._rebind_action, binding, self._rebind_slot)
        self.cancel_rebind_mode()
        return result

    # ==========================================================================
    # Profiles
    # ==========================================================================

    def save_profile(self, name: str, description: str = "") -> bool:
        """Save current bindings as a profile."""
        profile = BindingProfile(
            name=name,
            bindings=self.get_all_bindings(),
            description=description
        )
        self._profiles[name] = profile
        return True

    def load_profile(self, name: str) -> bool:
        """Load a binding profile."""
        if name not in self._profiles:
            return False

        profile = self._profiles[name]
        for action, bindings in profile.bindings.items():
            if action in self._bindings:
                self._bindings[action] = bindings.copy()

        self._current_profile = name
        return True

    def delete_profile(self, name: str) -> bool:
        """Delete a binding profile."""
        if name not in self._profiles:
            return False
        if self._profiles[name].is_default:
            return False  # Cannot delete default

        del self._profiles[name]
        if self._current_profile == name:
            self._current_profile = None
        return True

    def get_profiles(self) -> List[str]:
        """Get list of profile names."""
        return list(self._profiles.keys())

    def get_current_profile(self) -> Optional[str]:
        """Get name of current profile."""
        return self._current_profile

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def export_bindings(self) -> str:
        """Export bindings as JSON."""
        return json.dumps(self._bindings)

    def import_bindings(self, data: str) -> bool:
        """Import bindings from JSON."""
        try:
            bindings = json.loads(data)
            if not isinstance(bindings, dict):
                return False

            for action, binding_list in bindings.items():
                if action in self._bindings:
                    self._bindings[action] = binding_list

            return True
        except (json.JSONDecodeError, TypeError):
            return False

    # ==========================================================================
    # Listeners
    # ==========================================================================

    def add_listener(self, callback: callable) -> None:
        """Add a rebind event listener."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: callable) -> None:
        """Remove a rebind event listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self, event: RebindEvent) -> None:
        """Notify all listeners of a rebind event."""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass


# =============================================================================
# Basic Rebinding Tests
# =============================================================================

class TestRebindingBasic:
    """Basic tests for RebindingSystem."""

    @pytest.fixture
    def system(self):
        """Create a rebinding system."""
        return RebindingSystem()

    def test_register_action(self, system):
        """register_action adds action."""
        result = system.register_action("jump", ["space"])
        assert result is True
        assert system.get_bindings("jump") == ["space"]

    def test_register_duplicate_fails(self, system):
        """Registering duplicate action fails."""
        system.register_action("jump", ["space"])
        result = system.register_action("jump", ["w"])
        assert result is False

    def test_unregister_action(self, system):
        """unregister_action removes action."""
        system.register_action("jump", ["space"])
        result = system.unregister_action("jump")
        assert result is True
        assert system.get_bindings("jump") == []

    def test_unregister_nonexistent(self, system):
        """Unregistering nonexistent action fails."""
        result = system.unregister_action("nonexistent")
        assert result is False

    def test_get_bindings(self, system):
        """get_bindings returns current bindings."""
        system.register_action("jump", ["space", "w"])
        bindings = system.get_bindings("jump")
        assert "space" in bindings
        assert "w" in bindings

    def test_get_bindings_returns_copy(self, system):
        """get_bindings returns a copy."""
        system.register_action("jump", ["space"])
        bindings = system.get_bindings("jump")
        bindings.append("modified")
        assert "modified" not in system.get_bindings("jump")

    def test_get_default_bindings(self, system):
        """get_default_bindings returns original bindings."""
        system.register_action("jump", ["space"])
        system.rebind("jump", "w", 0)
        defaults = system.get_default_bindings("jump")
        assert defaults == ["space"]

    def test_get_all_bindings(self, system):
        """get_all_bindings returns all bindings."""
        system.register_action("jump", ["space"])
        system.register_action("crouch", ["ctrl"])
        all_bindings = system.get_all_bindings()
        assert "jump" in all_bindings
        assert "crouch" in all_bindings

    def test_get_action_for_binding(self, system):
        """get_action_for_binding finds action."""
        system.register_action("jump", ["space"])
        action = system.get_action_for_binding("space")
        assert action == "jump"

    def test_get_action_for_binding_none(self, system):
        """get_action_for_binding returns None for unbound."""
        action = system.get_action_for_binding("unbound")
        assert action is None


# =============================================================================
# Runtime Rebinding Tests
# =============================================================================

class TestRuntimeRebinding:
    """Tests for runtime rebinding."""

    @pytest.fixture
    def system(self):
        """Create a system with actions."""
        sys = RebindingSystem()
        sys.register_action("jump", ["space"])
        sys.register_action("crouch", ["ctrl"])
        sys.register_action("move_up", ["w"])
        return sys

    def test_rebind_action(self, system):
        """rebind changes action binding."""
        result, conflict = system.rebind("jump", "enter", 0)
        assert result == RebindResult.SUCCESS
        assert "enter" in system.get_bindings("jump")
        assert "space" not in system.get_bindings("jump")

    def test_rebind_nonexistent_action(self, system):
        """Rebinding nonexistent action fails."""
        result, _ = system.rebind("nonexistent", "space", 0)
        assert result == RebindResult.INVALID

    def test_rebind_to_different_slot(self, system):
        """Rebind to different slot works."""
        result, _ = system.rebind("jump", "enter", 1)
        assert result == RebindResult.SUCCESS
        bindings = system.get_bindings("jump")
        assert "space" in bindings
        assert "enter" in bindings

    def test_rebind_same_binding(self, system):
        """Rebinding to same binding works."""
        result, _ = system.rebind("jump", "space", 0)
        assert result == RebindResult.SUCCESS

    def test_rebind_preserves_other_bindings(self, system):
        """Rebinding one slot preserves others."""
        system.rebind("jump", "w", 1)
        system.rebind("jump", "enter", 0)
        bindings = system.get_bindings("jump")
        assert "enter" in bindings
        assert "w" in bindings


# =============================================================================
# Rebinding Validation Tests
# =============================================================================

class TestRebindingValidation:
    """Tests for rebinding validation."""

    @pytest.fixture
    def system(self):
        """Create a rebinding system."""
        sys = RebindingSystem()
        sys.register_action("jump", ["space"])
        return sys

    def test_reserved_binding_rejected(self, system):
        """Reserved bindings are rejected."""
        result, _ = system.rebind("jump", "escape", 0)
        assert result == RebindResult.RESERVED

    def test_is_binding_reserved(self, system):
        """is_binding_reserved returns correct value."""
        assert system.is_binding_reserved("escape") is True
        assert system.is_binding_reserved("space") is False

    def test_add_reserved_binding(self, system):
        """Can add new reserved bindings."""
        system.add_reserved_binding("f12")
        assert system.is_binding_reserved("f12") is True

    def test_remove_reserved_binding(self, system):
        """Can remove user-added reserved bindings."""
        system.add_reserved_binding("f12")
        result = system.remove_reserved_binding("f12")
        assert result is True
        assert system.is_binding_reserved("f12") is False

    def test_cannot_remove_builtin_reserved(self, system):
        """Cannot remove built-in reserved bindings."""
        result = system.remove_reserved_binding("escape")
        assert result is False
        assert system.is_binding_reserved("escape") is True

    def test_is_binding_valid(self, system):
        """is_binding_valid validates binding strings."""
        assert system.is_binding_valid("space") is True
        assert system.is_binding_valid("") is False
        assert system.is_binding_valid(None) is False
        assert system.is_binding_valid("   ") is False

    def test_invalid_slot_rejected(self, system):
        """Invalid slot numbers are rejected."""
        result, _ = system.rebind("jump", "enter", -1)
        assert result == RebindResult.INVALID

        result, _ = system.rebind("jump", "enter", 100)
        assert result == RebindResult.INVALID


# =============================================================================
# Conflict Handling Tests
# =============================================================================

class TestConflictHandling:
    """Tests for binding conflict handling."""

    @pytest.fixture
    def system_reject(self):
        """Create system with REJECT conflict resolution."""
        sys = RebindingSystem(ConflictResolution.REJECT)
        sys.register_action("jump", ["space"])
        sys.register_action("interact", ["e"])
        return sys

    @pytest.fixture
    def system_swap(self):
        """Create system with SWAP conflict resolution."""
        sys = RebindingSystem(ConflictResolution.SWAP)
        sys.register_action("jump", ["space"])
        sys.register_action("interact", ["e"])
        return sys

    @pytest.fixture
    def system_unbind(self):
        """Create system with UNBIND conflict resolution."""
        sys = RebindingSystem(ConflictResolution.UNBIND)
        sys.register_action("jump", ["space"])
        sys.register_action("interact", ["e"])
        return sys

    @pytest.fixture
    def system_duplicate(self):
        """Create system with DUPLICATE conflict resolution."""
        sys = RebindingSystem(ConflictResolution.DUPLICATE)
        sys.register_action("jump", ["space"])
        sys.register_action("interact", ["e"])
        return sys

    def test_reject_conflict(self, system_reject):
        """REJECT mode rejects conflicting bindings."""
        result, conflict = system_reject.rebind("interact", "space", 0)
        assert result == RebindResult.CONFLICT
        assert conflict == "jump"
        # Bindings unchanged
        assert system_reject.get_bindings("jump") == ["space"]
        assert system_reject.get_bindings("interact") == ["e"]

    def test_swap_conflict(self, system_swap):
        """SWAP mode swaps bindings."""
        system_swap.rebind("interact", "space", 0)

        # jump should now have 'e', interact should have 'space'
        assert "space" in system_swap.get_bindings("interact")
        assert "e" in system_swap.get_bindings("jump")

    def test_unbind_conflict(self, system_unbind):
        """UNBIND mode removes from conflicting action."""
        system_unbind.rebind("interact", "space", 0)

        assert "space" in system_unbind.get_bindings("interact")
        assert "space" not in system_unbind.get_bindings("jump")

    def test_duplicate_conflict(self, system_duplicate):
        """DUPLICATE mode allows same binding on multiple actions."""
        system_duplicate.rebind("interact", "space", 0)

        assert "space" in system_duplicate.get_bindings("interact")
        assert "space" in system_duplicate.get_bindings("jump")

    def test_get_conflicts(self, system_reject):
        """get_conflicts returns all conflicting actions."""
        conflicts = system_reject.get_conflicts("space")
        assert "jump" in conflicts

    def test_no_conflict_same_action(self, system_reject):
        """No conflict when rebinding to same action's binding."""
        # Moving binding within same action
        system_reject.register_action("test", ["a", "b"])
        result, _ = system_reject.rebind("test", "a", 1)
        assert result == RebindResult.SUCCESS

    def test_conflict_resolution_property(self, system_reject):
        """conflict_resolution property can be changed."""
        system_reject.conflict_resolution = ConflictResolution.SWAP
        assert system_reject.conflict_resolution == ConflictResolution.SWAP


# =============================================================================
# Reset to Defaults Tests
# =============================================================================

class TestResetToDefaults:
    """Tests for resetting bindings to defaults."""

    @pytest.fixture
    def system(self):
        """Create system with modified bindings."""
        sys = RebindingSystem()
        sys.register_action("jump", ["space"])
        sys.register_action("crouch", ["ctrl"])
        sys.rebind("jump", "w", 0)
        sys.rebind("crouch", "c", 0)
        return sys

    def test_reset_action(self, system):
        """reset_action restores default bindings."""
        system.reset_action("jump")
        assert system.get_bindings("jump") == ["space"]

    def test_reset_action_nonexistent(self, system):
        """reset_action on nonexistent action fails."""
        result = system.reset_action("nonexistent")
        assert result is False

    def test_reset_all(self, system):
        """reset_all restores all defaults."""
        system.reset_all()
        assert system.get_bindings("jump") == ["space"]
        assert system.get_bindings("crouch") == ["ctrl"]

    def test_reset_preserves_registrations(self, system):
        """reset_all preserves action registrations."""
        system.reset_all()
        assert system.get_bindings("jump") is not None
        assert system.get_bindings("crouch") is not None


# =============================================================================
# Unbind Tests
# =============================================================================

class TestUnbind:
    """Tests for unbinding actions."""

    @pytest.fixture
    def system(self):
        """Create system with multi-bind action."""
        sys = RebindingSystem()
        sys.register_action("jump", ["space", "w", "up"])
        return sys

    def test_unbind_specific_slot(self, system):
        """unbind removes specific slot."""
        result = system.unbind("jump", 1)
        assert result is True
        bindings = system.get_bindings("jump")
        assert "w" not in bindings
        assert "space" in bindings
        assert "up" in bindings

    def test_unbind_all(self, system):
        """unbind with slot=-1 removes all."""
        result = system.unbind("jump", -1)
        assert result is True
        assert system.get_bindings("jump") == []

    def test_unbind_invalid_slot(self, system):
        """unbind with invalid slot fails."""
        result = system.unbind("jump", 10)
        assert result is False

    def test_unbind_nonexistent_action(self, system):
        """unbind nonexistent action fails."""
        result = system.unbind("nonexistent", 0)
        assert result is False


# =============================================================================
# Add Binding Tests
# =============================================================================

class TestAddBinding:
    """Tests for adding bindings."""

    @pytest.fixture
    def system(self):
        """Create system with single binding."""
        sys = RebindingSystem()
        sys.register_action("jump", ["space"])
        return sys

    def test_add_binding(self, system):
        """add_binding adds without replacing."""
        result, _ = system.add_binding("jump", "w")
        assert result == RebindResult.SUCCESS
        bindings = system.get_bindings("jump")
        assert "space" in bindings
        assert "w" in bindings

    def test_add_binding_max_limit(self, system):
        """Cannot exceed max bindings per action."""
        system.add_binding("jump", "w")
        system.add_binding("jump", "up")
        system.add_binding("jump", "enter")

        result, _ = system.add_binding("jump", "tab")
        assert result == RebindResult.INVALID

    def test_add_binding_nonexistent(self, system):
        """add_binding to nonexistent action fails."""
        result, _ = system.add_binding("nonexistent", "space")
        assert result == RebindResult.INVALID


# =============================================================================
# Rebind Mode Tests
# =============================================================================

class TestRebindMode:
    """Tests for rebind mode (UI integration)."""

    @pytest.fixture
    def system(self):
        """Create system with actions."""
        sys = RebindingSystem()
        sys.register_action("jump", ["space"])
        return sys

    def test_start_rebind_mode(self, system):
        """start_rebind_mode activates mode."""
        result = system.start_rebind_mode("jump", 0)
        assert result is True
        assert system.is_rebind_mode_active is True

    def test_start_rebind_mode_nonexistent(self, system):
        """start_rebind_mode for nonexistent action fails."""
        result = system.start_rebind_mode("nonexistent", 0)
        assert result is False

    def test_cancel_rebind_mode(self, system):
        """cancel_rebind_mode deactivates mode."""
        system.start_rebind_mode("jump", 0)
        system.cancel_rebind_mode()
        assert system.is_rebind_mode_active is False

    def test_submit_rebind(self, system):
        """submit_rebind performs the rebind."""
        system.start_rebind_mode("jump", 0)
        result, _ = system.submit_rebind("enter")

        assert result == RebindResult.SUCCESS
        assert system.is_rebind_mode_active is False
        assert "enter" in system.get_bindings("jump")

    def test_submit_rebind_not_active(self, system):
        """submit_rebind without active mode fails."""
        result, _ = system.submit_rebind("enter")
        assert result == RebindResult.CANCELLED


# =============================================================================
# Profile Tests
# =============================================================================

class TestProfiles:
    """Tests for binding profiles."""

    @pytest.fixture
    def system(self):
        """Create system with actions."""
        sys = RebindingSystem()
        sys.register_action("jump", ["space"])
        sys.register_action("crouch", ["ctrl"])
        return sys

    def test_save_profile(self, system):
        """save_profile saves current bindings."""
        result = system.save_profile("custom", "My custom bindings")
        assert result is True
        assert "custom" in system.get_profiles()

    def test_load_profile(self, system):
        """load_profile restores saved bindings."""
        system.rebind("jump", "w", 0)
        system.save_profile("modified")

        system.reset_all()
        result = system.load_profile("modified")

        assert result is True
        assert "w" in system.get_bindings("jump")

    def test_load_profile_nonexistent(self, system):
        """load_profile for nonexistent profile fails."""
        result = system.load_profile("nonexistent")
        assert result is False

    def test_delete_profile(self, system):
        """delete_profile removes profile."""
        system.save_profile("temp")
        result = system.delete_profile("temp")
        assert result is True
        assert "temp" not in system.get_profiles()

    def test_delete_profile_nonexistent(self, system):
        """delete_profile for nonexistent profile fails."""
        result = system.delete_profile("nonexistent")
        assert result is False

    def test_get_current_profile(self, system):
        """get_current_profile returns loaded profile."""
        system.save_profile("test")
        system.load_profile("test")
        assert system.get_current_profile() == "test"


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for binding serialization."""

    @pytest.fixture
    def system(self):
        """Create system with actions."""
        sys = RebindingSystem()
        sys.register_action("jump", ["space"])
        sys.register_action("crouch", ["ctrl"])
        return sys

    def test_export_bindings(self, system):
        """export_bindings returns JSON string."""
        data = system.export_bindings()
        parsed = json.loads(data)
        assert "jump" in parsed
        assert "crouch" in parsed

    def test_import_bindings(self, system):
        """import_bindings loads from JSON."""
        data = json.dumps({"jump": ["w"], "crouch": ["c"]})
        result = system.import_bindings(data)

        assert result is True
        assert system.get_bindings("jump") == ["w"]
        assert system.get_bindings("crouch") == ["c"]

    def test_import_bindings_invalid_json(self, system):
        """import_bindings handles invalid JSON."""
        result = system.import_bindings("not json")
        assert result is False

    def test_import_bindings_wrong_type(self, system):
        """import_bindings handles wrong data type."""
        result = system.import_bindings('["list", "not", "dict"]')
        assert result is False

    def test_import_preserves_unmentioned(self, system):
        """import_bindings only changes mentioned actions."""
        system.register_action("other", ["o"])
        data = json.dumps({"jump": ["w"]})
        system.import_bindings(data)

        # Other action unchanged
        assert system.get_bindings("other") == ["o"]

    def test_export_import_roundtrip(self, system):
        """Export and import preserve bindings."""
        system.rebind("jump", "enter", 0)
        data = system.export_bindings()

        system.reset_all()
        system.import_bindings(data)

        assert "enter" in system.get_bindings("jump")


# =============================================================================
# Listener Tests
# =============================================================================

class TestListeners:
    """Tests for rebind event listeners."""

    @pytest.fixture
    def system(self):
        """Create system with actions."""
        sys = RebindingSystem()
        sys.register_action("jump", ["space"])
        return sys

    def test_add_listener(self, system):
        """add_listener registers callback."""
        events = []
        system.add_listener(lambda e: events.append(e))

        system.rebind("jump", "enter", 0)

        assert len(events) == 1
        assert events[0].action_name == "jump"

    def test_remove_listener(self, system):
        """remove_listener unregisters callback."""
        events = []
        callback = lambda e: events.append(e)

        system.add_listener(callback)
        system.remove_listener(callback)
        system.rebind("jump", "enter", 0)

        assert len(events) == 0

    def test_listener_receives_event_data(self, system):
        """Listener receives complete event data."""
        events = []
        system.add_listener(lambda e: events.append(e))

        system.rebind("jump", "enter", 0)

        event = events[0]
        assert event.action_name == "jump"
        assert event.old_binding == "space"
        assert event.new_binding == "enter"
        assert event.result == RebindResult.SUCCESS

    def test_listener_exception_handled(self, system):
        """Listener exception doesn't break system."""
        def bad_listener(e):
            raise ValueError("Test error")

        system.add_listener(bad_listener)

        # Should not raise
        system.rebind("jump", "enter", 0)


# =============================================================================
# Integration Tests
# =============================================================================

class TestRebindingIntegration:
    """Integration tests for rebinding system."""

    def test_options_menu_workflow(self):
        """Simulate options menu rebinding workflow."""
        system = RebindingSystem(ConflictResolution.REJECT)

        # Register game actions
        system.register_action("move_forward", ["w"])
        system.register_action("move_back", ["s"])
        system.register_action("jump", ["space"])
        system.register_action("crouch", ["ctrl"])
        system.register_action("interact", ["e"])

        # User wants to rebind jump to 'e'
        result, conflict = system.rebind("jump", "e", 0)
        assert result == RebindResult.CONFLICT
        assert conflict == "interact"

        # User chooses to swap
        system.conflict_resolution = ConflictResolution.SWAP
        result, _ = system.rebind("jump", "e", 0)
        assert result == RebindResult.SUCCESS

        # jump now has 'e', interact has 'space'
        assert "e" in system.get_bindings("jump")
        assert "space" in system.get_bindings("interact")

    def test_controller_preset_workflow(self):
        """Simulate controller preset selection."""
        system = RebindingSystem()

        system.register_action("jump", ["button_a"])
        system.register_action("attack", ["button_b"])
        system.register_action("block", ["button_x"])
        system.register_action("special", ["button_y"])

        # Save default
        system.save_profile("default")

        # Create alternate preset
        system.rebind("jump", "button_b", 0)
        system.rebind("attack", "button_a", 0)
        system.save_profile("alternate")

        # Switch between presets
        system.load_profile("default")
        assert "button_a" in system.get_bindings("jump")

        system.load_profile("alternate")
        assert "button_b" in system.get_bindings("jump")

    def test_rebind_mode_ui_flow(self):
        """Simulate UI rebind flow."""
        system = RebindingSystem()
        system.register_action("jump", ["space"])

        # User clicks "rebind" button
        system.start_rebind_mode("jump", 0)
        assert system.is_rebind_mode_active

        # User presses a key
        result, _ = system.submit_rebind("enter")

        assert result == RebindResult.SUCCESS
        assert not system.is_rebind_mode_active
        assert "enter" in system.get_bindings("jump")

    def test_rebind_mode_cancel_flow(self):
        """Simulate cancelling a rebind."""
        system = RebindingSystem()
        system.register_action("jump", ["space"])

        system.start_rebind_mode("jump", 0)

        # User presses escape (cancel)
        system.cancel_rebind_mode()

        assert not system.is_rebind_mode_active
        assert system.get_bindings("jump") == ["space"]  # Unchanged
