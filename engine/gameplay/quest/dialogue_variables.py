"""
Dialogue Variable System.

Provides variable storage and management with three scopes:
- Local: Per-conversation variables, cleared when dialogue ends
- Global: Persistent variables shared across all dialogues
- Quest: Variables linked to quest state
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Set, Union

from .constants import (
    DEFAULT_VARIABLE_NAMESPACE,
    MAX_VARIABLE_NAME_LENGTH,
    MAX_VARIABLE_VALUE_LENGTH,
    VariableScope,
)

# Type alias for variable values
VariableValue = Union[str, int, float, bool, list, dict, None]


# =============================================================================
# Variable Name Validation
# =============================================================================

# Valid variable name pattern: alphanumeric, underscore, dot for namespacing
VARIABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


def validate_variable_name(name: str) -> bool:
    """
    Validate a variable name.

    Args:
        name: The variable name to validate.

    Returns:
        True if valid, raises ValueError otherwise.

    Raises:
        ValueError: If the name is invalid.
    """
    if not name:
        raise ValueError("Variable name cannot be empty")

    if len(name) > MAX_VARIABLE_NAME_LENGTH:
        raise ValueError(
            f"Variable name too long: {len(name)} > {MAX_VARIABLE_NAME_LENGTH}"
        )

    if not VARIABLE_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid variable name '{name}': must start with letter or underscore, "
            "contain only alphanumeric, underscore, or dot characters"
        )

    return True


def validate_variable_value(value: VariableValue) -> bool:
    """
    Validate a variable value.

    Args:
        value: The value to validate.

    Returns:
        True if valid, raises ValueError otherwise.

    Raises:
        ValueError: If the value is invalid.
    """
    if isinstance(value, str) and len(value) > MAX_VARIABLE_VALUE_LENGTH:
        raise ValueError(
            f"Variable value too long: {len(value)} > {MAX_VARIABLE_VALUE_LENGTH}"
        )

    # Validate nested structures
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("Dictionary keys must be strings")
            validate_variable_value(v)
    elif isinstance(value, list):
        for item in value:
            validate_variable_value(item)

    return True


# =============================================================================
# Variable Change Event
# =============================================================================

@dataclass
class VariableChangeEvent:
    """Event fired when a variable changes."""
    name: str
    old_value: VariableValue
    new_value: VariableValue
    scope: VariableScope
    namespace: str = DEFAULT_VARIABLE_NAMESPACE


# =============================================================================
# Variable Store Base Class
# =============================================================================

class VariableStore:
    """
    Base class for variable storage.

    Provides core functionality for storing, retrieving, and managing
    dialogue variables with change tracking and observers.
    """

    def __init__(
        self,
        scope: VariableScope,
        namespace: str = DEFAULT_VARIABLE_NAMESPACE
    ):
        """
        Initialize the variable store.

        Args:
            scope: The scope of variables in this store.
            namespace: The namespace for organizing variables.
        """
        self._scope = scope
        self._namespace = namespace
        self._variables: Dict[str, VariableValue] = {}
        self._defaults: Dict[str, VariableValue] = {}
        self._observers: Set[Callable[[VariableChangeEvent], None]] = set()
        self._change_history: list[VariableChangeEvent] = []
        self._track_history: bool = False

    @property
    def scope(self) -> VariableScope:
        """Get the variable scope."""
        return self._scope

    @property
    def namespace(self) -> str:
        """Get the namespace."""
        return self._namespace

    def set(self, name: str, value: VariableValue) -> None:
        """
        Set a variable value.

        Args:
            name: The variable name.
            value: The value to set.

        Raises:
            ValueError: If name or value is invalid.
        """
        validate_variable_name(name)
        validate_variable_value(value)

        old_value = self._variables.get(name)

        # Deep copy mutable values
        if isinstance(value, (list, dict)):
            value = copy.deepcopy(value)

        self._variables[name] = value

        # Fire change event
        event = VariableChangeEvent(
            name=name,
            old_value=old_value,
            new_value=value,
            scope=self._scope,
            namespace=self._namespace
        )

        if self._track_history:
            self._change_history.append(event)

        for observer in self._observers:
            observer(event)

    def get(self, name: str, default: VariableValue = None) -> VariableValue:
        """
        Get a variable value.

        Args:
            name: The variable name.
            default: Default value if not found.

        Returns:
            The variable value or default.
        """
        value = self._variables.get(name, self._defaults.get(name, default))

        # Return deep copy of mutable values
        if isinstance(value, (list, dict)):
            return copy.deepcopy(value)

        return value

    def has(self, name: str) -> bool:
        """
        Check if a variable exists.

        Args:
            name: The variable name.

        Returns:
            True if the variable exists.
        """
        return name in self._variables or name in self._defaults

    def delete(self, name: str) -> bool:
        """
        Delete a variable.

        Args:
            name: The variable name.

        Returns:
            True if deleted, False if not found.
        """
        if name in self._variables:
            old_value = self._variables.pop(name)

            event = VariableChangeEvent(
                name=name,
                old_value=old_value,
                new_value=None,
                scope=self._scope,
                namespace=self._namespace
            )

            if self._track_history:
                self._change_history.append(event)

            for observer in self._observers:
                observer(event)

            return True

        return False

    def clear(self) -> None:
        """Clear all variables (not defaults)."""
        old_vars = self._variables.copy()
        self._variables.clear()

        for name, old_value in old_vars.items():
            event = VariableChangeEvent(
                name=name,
                old_value=old_value,
                new_value=None,
                scope=self._scope,
                namespace=self._namespace
            )

            if self._track_history:
                self._change_history.append(event)

            for observer in self._observers:
                observer(event)

    def set_default(self, name: str, value: VariableValue) -> None:
        """
        Set a default value for a variable.

        Args:
            name: The variable name.
            value: The default value.
        """
        validate_variable_name(name)
        validate_variable_value(value)

        if isinstance(value, (list, dict)):
            value = copy.deepcopy(value)

        self._defaults[name] = value

    def reset_to_default(self, name: str) -> bool:
        """
        Reset a variable to its default value.

        Args:
            name: The variable name.

        Returns:
            True if reset, False if no default exists.
        """
        if name in self._defaults:
            self.set(name, self._defaults[name])
            return True
        return False

    def reset_all_to_defaults(self) -> None:
        """Reset all variables to their default values."""
        self._variables.clear()
        for name, value in self._defaults.items():
            self.set(name, value)

    def add_observer(
        self,
        callback: Callable[[VariableChangeEvent], None]
    ) -> None:
        """
        Add an observer for variable changes.

        Args:
            callback: Function called on changes.
        """
        self._observers.add(callback)

    def remove_observer(
        self,
        callback: Callable[[VariableChangeEvent], None]
    ) -> bool:
        """
        Remove an observer.

        Args:
            callback: The callback to remove.

        Returns:
            True if removed, False if not found.
        """
        if callback in self._observers:
            self._observers.discard(callback)
            return True
        return False

    def enable_history(self, enabled: bool = True) -> None:
        """Enable or disable change history tracking."""
        self._track_history = enabled
        if not enabled:
            self._change_history.clear()

    def get_history(self) -> list[VariableChangeEvent]:
        """Get the change history."""
        return self._change_history.copy()

    def clear_history(self) -> None:
        """Clear the change history."""
        self._change_history.clear()

    def keys(self) -> Iterator[str]:
        """Iterate over variable names."""
        seen = set()
        for name in self._variables:
            seen.add(name)
            yield name
        for name in self._defaults:
            if name not in seen:
                yield name

    def items(self) -> Iterator[tuple[str, VariableValue]]:
        """Iterate over name-value pairs."""
        for name in self.keys():
            yield name, self.get(name)

    def __len__(self) -> int:
        """Get number of variables."""
        return len(set(self._variables.keys()) | set(self._defaults.keys()))

    def __contains__(self, name: str) -> bool:
        """Check if variable exists."""
        return self.has(name)

    def __getitem__(self, name: str) -> VariableValue:
        """Get variable value."""
        if not self.has(name):
            raise KeyError(name)
        return self.get(name)

    def __setitem__(self, name: str, value: VariableValue) -> None:
        """Set variable value."""
        self.set(name, value)

    def to_dict(self) -> Dict[str, VariableValue]:
        """
        Export variables to dictionary.

        Returns:
            Dictionary of all variables.
        """
        result = {}
        for name in self.keys():
            result[name] = self.get(name)
        return result

    def from_dict(self, data: Dict[str, VariableValue]) -> None:
        """
        Import variables from dictionary.

        Args:
            data: Dictionary of variables to import.
        """
        for name, value in data.items():
            self.set(name, value)


# =============================================================================
# Local Variable Store
# =============================================================================

class LocalVariableStore(VariableStore):
    """
    Local variable store for per-conversation variables.

    Variables are automatically cleared when the dialogue ends.
    """

    def __init__(self, namespace: str = DEFAULT_VARIABLE_NAMESPACE):
        """Initialize local variable store."""
        super().__init__(VariableScope.LOCAL, namespace)
        self._conversation_id: Optional[str] = None

    @property
    def conversation_id(self) -> Optional[str]:
        """Get the current conversation ID."""
        return self._conversation_id

    def start_conversation(self, conversation_id: str) -> None:
        """
        Start a new conversation, clearing existing variables.

        Args:
            conversation_id: The conversation identifier.
        """
        self.clear()
        self._conversation_id = conversation_id

    def end_conversation(self) -> None:
        """End the current conversation and clear variables."""
        self.clear()
        self._conversation_id = None


# =============================================================================
# Global Variable Store
# =============================================================================

class GlobalVariableStore(VariableStore):
    """
    Global variable store for persistent variables.

    Variables persist across all dialogues and game sessions.
    """

    def __init__(self, namespace: str = DEFAULT_VARIABLE_NAMESPACE):
        """Initialize global variable store."""
        super().__init__(VariableScope.GLOBAL, namespace)

    def increment(self, name: str, amount: Union[int, float] = 1) -> VariableValue:
        """
        Increment a numeric variable.

        Args:
            name: The variable name.
            amount: Amount to increment by.

        Returns:
            The new value.

        Raises:
            TypeError: If variable is not numeric.
        """
        current = self.get(name, 0)

        if not isinstance(current, (int, float)):
            raise TypeError(f"Cannot increment non-numeric variable: {name}")

        new_value = current + amount
        self.set(name, new_value)
        return new_value

    def decrement(self, name: str, amount: Union[int, float] = 1) -> VariableValue:
        """
        Decrement a numeric variable.

        Args:
            name: The variable name.
            amount: Amount to decrement by.

        Returns:
            The new value.
        """
        return self.increment(name, -amount)

    def toggle(self, name: str) -> bool:
        """
        Toggle a boolean variable.

        Args:
            name: The variable name.

        Returns:
            The new value.
        """
        current = self.get(name, False)
        new_value = not bool(current)
        self.set(name, new_value)
        return new_value

    def append_to_list(self, name: str, value: VariableValue) -> list:
        """
        Append a value to a list variable.

        Args:
            name: The variable name.
            value: The value to append.

        Returns:
            The updated list.
        """
        current = self.get(name, [])

        if not isinstance(current, list):
            current = [current]

        current.append(value)
        self.set(name, current)
        return current

    def remove_from_list(self, name: str, value: VariableValue) -> bool:
        """
        Remove a value from a list variable.

        Args:
            name: The variable name.
            value: The value to remove.

        Returns:
            True if removed, False if not found.
        """
        current = self.get(name, [])

        if not isinstance(current, list):
            return False

        if value in current:
            current.remove(value)
            self.set(name, current)
            return True

        return False


# =============================================================================
# Quest Variable Store
# =============================================================================

@dataclass
class QuestBinding:
    """Binding between a variable and a quest."""
    quest_id: str
    variable_name: str
    sync_mode: str = "bidirectional"  # "read", "write", "bidirectional"


class QuestVariableStore(VariableStore):
    """
    Quest-linked variable store.

    Variables are synchronized with quest state and persist
    as long as the quest exists.
    """

    def __init__(self, namespace: str = DEFAULT_VARIABLE_NAMESPACE):
        """Initialize quest variable store."""
        super().__init__(VariableScope.QUEST, namespace)
        self._quest_bindings: Dict[str, QuestBinding] = {}
        self._quest_states: Dict[str, Dict[str, VariableValue]] = {}

    def bind_to_quest(
        self,
        variable_name: str,
        quest_id: str,
        sync_mode: str = "bidirectional"
    ) -> None:
        """
        Bind a variable to a quest.

        Args:
            variable_name: The variable to bind.
            quest_id: The quest to bind to.
            sync_mode: How to sync ("read", "write", "bidirectional").
        """
        validate_variable_name(variable_name)

        if sync_mode not in ("read", "write", "bidirectional"):
            raise ValueError(f"Invalid sync_mode: {sync_mode}")

        self._quest_bindings[variable_name] = QuestBinding(
            quest_id=quest_id,
            variable_name=variable_name,
            sync_mode=sync_mode
        )

        # Initialize from quest state if exists
        if quest_id in self._quest_states:
            quest_state = self._quest_states[quest_id]
            if variable_name in quest_state:
                self._variables[variable_name] = quest_state[variable_name]

    def unbind_from_quest(self, variable_name: str) -> bool:
        """
        Remove a variable binding from a quest.

        Args:
            variable_name: The variable to unbind.

        Returns:
            True if unbound, False if not found.
        """
        if variable_name in self._quest_bindings:
            del self._quest_bindings[variable_name]
            return True
        return False

    def get_binding(self, variable_name: str) -> Optional[QuestBinding]:
        """
        Get the quest binding for a variable.

        Args:
            variable_name: The variable name.

        Returns:
            The binding or None if not bound.
        """
        return self._quest_bindings.get(variable_name)

    def set_quest_state(
        self,
        quest_id: str,
        state: Dict[str, VariableValue]
    ) -> None:
        """
        Set the state for a quest.

        Args:
            quest_id: The quest identifier.
            state: The quest state dictionary.
        """
        self._quest_states[quest_id] = copy.deepcopy(state)

        # Update bound variables
        for var_name, binding in self._quest_bindings.items():
            if binding.quest_id == quest_id and var_name in state:
                if binding.sync_mode in ("read", "bidirectional"):
                    self._variables[var_name] = state[var_name]

    def get_quest_state(self, quest_id: str) -> Dict[str, VariableValue]:
        """
        Get the state for a quest.

        Args:
            quest_id: The quest identifier.

        Returns:
            The quest state dictionary.
        """
        return copy.deepcopy(self._quest_states.get(quest_id, {}))

    def sync_to_quest(self, variable_name: str) -> bool:
        """
        Sync a variable value to its bound quest.

        Args:
            variable_name: The variable name.

        Returns:
            True if synced, False if not bound.
        """
        binding = self._quest_bindings.get(variable_name)
        if not binding:
            return False

        if binding.sync_mode in ("write", "bidirectional"):
            if binding.quest_id not in self._quest_states:
                self._quest_states[binding.quest_id] = {}

            self._quest_states[binding.quest_id][variable_name] = self.get(
                variable_name
            )
            return True

        return False

    def sync_all_to_quest(self, quest_id: str) -> int:
        """
        Sync all variables bound to a quest.

        Args:
            quest_id: The quest identifier.

        Returns:
            Number of variables synced.
        """
        count = 0
        for var_name, binding in self._quest_bindings.items():
            if binding.quest_id == quest_id:
                if self.sync_to_quest(var_name):
                    count += 1
        return count

    def clear_quest(self, quest_id: str) -> None:
        """
        Clear all data for a quest.

        Args:
            quest_id: The quest identifier.
        """
        # Remove quest state
        self._quest_states.pop(quest_id, None)

        # Remove bindings for this quest
        to_remove = [
            name for name, binding in self._quest_bindings.items()
            if binding.quest_id == quest_id
        ]
        for name in to_remove:
            del self._quest_bindings[name]


# =============================================================================
# Variable Manager
# =============================================================================

class VariableManager:
    """
    Central manager for all dialogue variables.

    Coordinates between local, global, and quest variable stores.
    """

    def __init__(self, namespace: str = DEFAULT_VARIABLE_NAMESPACE):
        """
        Initialize the variable manager.

        Args:
            namespace: The default namespace for variables.
        """
        self._namespace = namespace
        self._local = LocalVariableStore(namespace)
        self._global = GlobalVariableStore(namespace)
        self._quest = QuestVariableStore(namespace)

    @property
    def local(self) -> LocalVariableStore:
        """Get the local variable store."""
        return self._local

    @property
    def global_store(self) -> GlobalVariableStore:
        """Get the global variable store."""
        return self._global

    @property
    def quest(self) -> QuestVariableStore:
        """Get the quest variable store."""
        return self._quest

    def get(
        self,
        name: str,
        scope: Optional[VariableScope] = None,
        default: VariableValue = None
    ) -> VariableValue:
        """
        Get a variable from the appropriate scope.

        Args:
            name: The variable name.
            scope: The scope to search (None = search all).
            default: Default value if not found.

        Returns:
            The variable value.
        """
        if scope == VariableScope.LOCAL:
            return self._local.get(name, default)
        elif scope == VariableScope.GLOBAL:
            return self._global.get(name, default)
        elif scope == VariableScope.QUEST:
            return self._quest.get(name, default)
        else:
            # Search in order: local, quest, global
            if self._local.has(name):
                return self._local.get(name)
            if self._quest.has(name):
                return self._quest.get(name)
            if self._global.has(name):
                return self._global.get(name)
            return default

    def set(
        self,
        name: str,
        value: VariableValue,
        scope: VariableScope = VariableScope.LOCAL
    ) -> None:
        """
        Set a variable in the specified scope.

        Args:
            name: The variable name.
            value: The value to set.
            scope: The scope to set in.
        """
        if scope == VariableScope.LOCAL:
            self._local.set(name, value)
        elif scope == VariableScope.GLOBAL:
            self._global.set(name, value)
        elif scope == VariableScope.QUEST:
            self._quest.set(name, value)

    def has(
        self,
        name: str,
        scope: Optional[VariableScope] = None
    ) -> bool:
        """
        Check if a variable exists.

        Args:
            name: The variable name.
            scope: The scope to check (None = any scope).

        Returns:
            True if the variable exists.
        """
        if scope == VariableScope.LOCAL:
            return self._local.has(name)
        elif scope == VariableScope.GLOBAL:
            return self._global.has(name)
        elif scope == VariableScope.QUEST:
            return self._quest.has(name)
        else:
            return (
                self._local.has(name) or
                self._global.has(name) or
                self._quest.has(name)
            )

    def delete(
        self,
        name: str,
        scope: Optional[VariableScope] = None
    ) -> bool:
        """
        Delete a variable.

        Args:
            name: The variable name.
            scope: The scope to delete from (None = all scopes).

        Returns:
            True if any variable was deleted.
        """
        deleted = False

        if scope is None or scope == VariableScope.LOCAL:
            deleted = self._local.delete(name) or deleted
        if scope is None or scope == VariableScope.GLOBAL:
            deleted = self._global.delete(name) or deleted
        if scope is None or scope == VariableScope.QUEST:
            deleted = self._quest.delete(name) or deleted

        return deleted

    def start_conversation(self, conversation_id: str) -> None:
        """
        Start a new conversation.

        Args:
            conversation_id: The conversation identifier.
        """
        self._local.start_conversation(conversation_id)

    def end_conversation(self) -> None:
        """End the current conversation."""
        self._local.end_conversation()

    def export_state(self) -> Dict[str, Any]:
        """
        Export all variable state.

        Returns:
            Dictionary with all variable state.
        """
        return {
            "local": self._local.to_dict(),
            "global": self._global.to_dict(),
            "quest": self._quest.to_dict(),
            "quest_states": {
                qid: self._quest.get_quest_state(qid)
                for qid in self._quest._quest_states
            },
            "quest_bindings": {
                name: {
                    "quest_id": binding.quest_id,
                    "sync_mode": binding.sync_mode
                }
                for name, binding in self._quest._quest_bindings.items()
            }
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """
        Import variable state.

        Args:
            state: Dictionary with variable state.
        """
        if "local" in state:
            self._local.from_dict(state["local"])

        if "global" in state:
            self._global.from_dict(state["global"])

        if "quest" in state:
            self._quest.from_dict(state["quest"])

        if "quest_states" in state:
            for qid, qstate in state["quest_states"].items():
                self._quest.set_quest_state(qid, qstate)

        if "quest_bindings" in state:
            for name, binding_data in state["quest_bindings"].items():
                self._quest.bind_to_quest(
                    name,
                    binding_data["quest_id"],
                    binding_data.get("sync_mode", "bidirectional")
                )
