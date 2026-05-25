"""
Commands - Command pattern for undoable editor actions.

Integrates with Foundation's Tracker for transaction-based undo/redo.

Provides:
- Base Command class with execute/undo interface
- CommandManager for managing command history
- Concrete commands: Transform, Create, Delete, Property, Reparent
- Command batching for atomic multi-operations
- Composite commands for grouping related commands
"""
from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar

from engine.tooling.editor.app_shell import editor, reloadable

T = TypeVar('T')


@editor(category="Commands")
@reloadable()
class Command(ABC):
    """Base class for undoable commands."""
    __slots__ = ("name", "description", "_executed", "timestamp")

    def __init__(self, name: str = "", description: str = ""):
        self.name = name or self.__class__.__name__
        self.description = description
        self._executed: bool = False
        self.timestamp: float = 0.0

    @property
    def executed(self) -> bool:
        """Check if command has been executed."""
        return self._executed

    @abstractmethod
    def execute(self) -> bool:
        """Execute the command. Returns True if successful."""
        pass

    @abstractmethod
    def undo(self) -> bool:
        """Undo the command. Returns True if successful."""
        pass

    def redo(self) -> bool:
        """Redo the command. Default is to re-execute."""
        return self.execute()

    def can_merge(self, other: "Command") -> bool:
        """Check if this command can be merged with another."""
        return False

    def merge(self, other: "Command") -> Optional["Command"]:
        """Merge this command with another. Returns merged command or None."""
        return None


@editor(category="Commands")
@reloadable()
class TransformCommand(Command):
    """Command for transforming object position/rotation/scale."""
    __slots__ = ("target", "transform_type", "old_value", "new_value",
                 "_target_ref")

    def __init__(self, target: Any, transform_type: str,
                 old_value: tuple, new_value: tuple, name: str = ""):
        super().__init__(name or f"Transform {transform_type}")
        self.target = target
        self._target_ref = weakref.ref(target) if target else None
        self.transform_type = transform_type
        self.old_value = old_value
        self.new_value = new_value

    def _get_target(self) -> Optional[Any]:
        """Get target from weak reference."""
        return self._target_ref() if self._target_ref else None

    def execute(self) -> bool:
        target = self._get_target()
        if target is None:
            return False

        if self.transform_type == "position":
            if hasattr(target, "set_position"):
                target.set_position(*self.new_value)
            elif hasattr(target, "position"):
                target.position = self.new_value
            else:
                return False
        elif self.transform_type == "rotation":
            if hasattr(target, "set_rotation"):
                target.set_rotation(*self.new_value)
            elif hasattr(target, "rotation"):
                target.rotation = self.new_value
            else:
                return False
        elif self.transform_type == "scale":
            if hasattr(target, "set_scale"):
                target.set_scale(*self.new_value)
            elif hasattr(target, "scale"):
                target.scale = self.new_value
            else:
                return False
        else:
            return False

        self._executed = True
        return True

    def undo(self) -> bool:
        target = self._get_target()
        if target is None:
            return False

        if self.transform_type == "position":
            if hasattr(target, "set_position"):
                target.set_position(*self.old_value)
            elif hasattr(target, "position"):
                target.position = self.old_value
        elif self.transform_type == "rotation":
            if hasattr(target, "set_rotation"):
                target.set_rotation(*self.old_value)
            elif hasattr(target, "rotation"):
                target.rotation = self.old_value
        elif self.transform_type == "scale":
            if hasattr(target, "set_scale"):
                target.set_scale(*self.old_value)
            elif hasattr(target, "scale"):
                target.scale = self.old_value

        self._executed = False
        return True

    def can_merge(self, other: "Command") -> bool:
        """Merge consecutive transforms of same type on same target."""
        if not isinstance(other, TransformCommand):
            return False
        other_target = other._get_target()
        return (self._get_target() is other_target and
                self.transform_type == other.transform_type)

    def merge(self, other: "Command") -> Optional["Command"]:
        if self.can_merge(other) and isinstance(other, TransformCommand):
            # Keep original old_value, use new new_value
            return TransformCommand(
                self._get_target(),
                self.transform_type,
                self.old_value,
                other.new_value,
                self.name
            )
        return None


@editor(category="Commands")
@reloadable()
class CreateCommand(Command):
    """Command for creating objects."""
    __slots__ = ("object_type", "creation_args", "created_object",
                 "parent", "_scene_ref")

    def __init__(self, object_type: type, creation_args: dict = None,
                 parent: Any = None, scene: Any = None, name: str = ""):
        super().__init__(name or f"Create {object_type.__name__}")
        self.object_type = object_type
        self.creation_args = creation_args or {}
        self.created_object: Any = None
        self.parent = parent
        self._scene_ref = weakref.ref(scene) if scene else None

    def execute(self) -> bool:
        try:
            self.created_object = self.object_type(**self.creation_args)

            # Add to scene if scene reference exists
            scene = self._scene_ref() if self._scene_ref else None
            if scene and hasattr(scene, "add"):
                scene.add(self.created_object, self.parent)

            self._executed = True
            return True
        except Exception:
            return False

    def undo(self) -> bool:
        if self.created_object is None:
            return False

        scene = self._scene_ref() if self._scene_ref else None
        if scene and hasattr(scene, "remove"):
            scene.remove(self.created_object)

        self._executed = False
        return True


@editor(category="Commands")
@reloadable()
class DeleteCommand(Command):
    """Command for deleting objects."""
    __slots__ = ("targets", "_target_data", "_scene_ref")

    def __init__(self, targets: list[Any], scene: Any = None, name: str = ""):
        super().__init__(name or f"Delete {len(targets)} objects")
        self.targets = list(targets)
        self._target_data: list[dict] = []  # Store data for undo
        self._scene_ref = weakref.ref(scene) if scene else None

    def execute(self) -> bool:
        scene = self._scene_ref() if self._scene_ref else None

        # Store object data for undo
        self._target_data = []
        for target in self.targets:
            data = {
                "object": target,
                "parent": getattr(target, "parent", None),
            }
            # Store transform if available
            if hasattr(target, "position"):
                data["position"] = target.position
            if hasattr(target, "rotation"):
                data["rotation"] = target.rotation
            if hasattr(target, "scale"):
                data["scale"] = target.scale
            self._target_data.append(data)

            # Remove from scene
            if scene and hasattr(scene, "remove"):
                scene.remove(target)

        self._executed = True
        return True

    def undo(self) -> bool:
        scene = self._scene_ref() if self._scene_ref else None

        for data in self._target_data:
            obj = data["object"]
            # Restore to scene
            if scene and hasattr(scene, "add"):
                scene.add(obj, data.get("parent"))

            # Restore transform
            if "position" in data and hasattr(obj, "position"):
                obj.position = data["position"]
            if "rotation" in data and hasattr(obj, "rotation"):
                obj.rotation = data["rotation"]
            if "scale" in data and hasattr(obj, "scale"):
                obj.scale = data["scale"]

        self._executed = False
        return True


@editor(category="Commands")
@reloadable()
class PropertyCommand(Command, Generic[T]):
    """Command for changing a single property value."""
    __slots__ = ("target", "property_name", "old_value", "new_value",
                 "_target_ref")

    def __init__(self, target: Any, property_name: str,
                 old_value: T, new_value: T, name: str = ""):
        super().__init__(name or f"Set {property_name}")
        self.target = target
        self._target_ref = weakref.ref(target) if target else None
        self.property_name = property_name
        self.old_value = old_value
        self.new_value = new_value

    def _get_target(self) -> Optional[Any]:
        return self._target_ref() if self._target_ref else None

    def execute(self) -> bool:
        target = self._get_target()
        if target is None:
            return False

        try:
            setattr(target, self.property_name, self.new_value)
            self._executed = True
            return True
        except Exception:
            return False

    def undo(self) -> bool:
        target = self._get_target()
        if target is None:
            return False

        try:
            setattr(target, self.property_name, self.old_value)
            self._executed = False
            return True
        except Exception:
            return False

    def can_merge(self, other: "Command") -> bool:
        if not isinstance(other, PropertyCommand):
            return False
        other_target = other._get_target()
        return (self._get_target() is other_target and
                self.property_name == other.property_name)

    def merge(self, other: "Command") -> Optional["Command"]:
        if self.can_merge(other) and isinstance(other, PropertyCommand):
            return PropertyCommand(
                self._get_target(),
                self.property_name,
                self.old_value,
                other.new_value,
                self.name
            )
        return None


@editor(category="Commands")
@reloadable()
class ReparentCommand(Command):
    """Command for changing object parent."""
    __slots__ = ("target", "old_parent", "new_parent", "_target_ref",
                 "_old_parent_ref", "_new_parent_ref", "_old_local_transform")

    def __init__(self, target: Any, old_parent: Any, new_parent: Any,
                 name: str = ""):
        super().__init__(name or "Reparent")
        self.target = target
        self._target_ref = weakref.ref(target) if target else None
        self.old_parent = old_parent
        self._old_parent_ref = weakref.ref(old_parent) if old_parent else None
        self.new_parent = new_parent
        self._new_parent_ref = weakref.ref(new_parent) if new_parent else None
        self._old_local_transform: Optional[tuple] = None

    def execute(self) -> bool:
        target = self._target_ref() if self._target_ref else None
        if target is None:
            return False

        new_parent = self._new_parent_ref() if self._new_parent_ref else None

        # Store old local transform
        if hasattr(target, "local_position"):
            self._old_local_transform = (
                getattr(target, "local_position", (0, 0, 0)),
                getattr(target, "local_rotation", (0, 0, 0)),
                getattr(target, "local_scale", (1, 1, 1)),
            )

        # Set new parent
        if hasattr(target, "set_parent"):
            target.set_parent(new_parent)
        elif hasattr(target, "parent"):
            target.parent = new_parent

        self._executed = True
        return True

    def undo(self) -> bool:
        target = self._target_ref() if self._target_ref else None
        if target is None:
            return False

        old_parent = self._old_parent_ref() if self._old_parent_ref else None

        # Restore parent
        if hasattr(target, "set_parent"):
            target.set_parent(old_parent)
        elif hasattr(target, "parent"):
            target.parent = old_parent

        # Restore local transform
        if self._old_local_transform:
            pos, rot, scale = self._old_local_transform
            if hasattr(target, "local_position"):
                target.local_position = pos
            if hasattr(target, "local_rotation"):
                target.local_rotation = rot
            if hasattr(target, "local_scale"):
                target.local_scale = scale

        self._executed = False
        return True


@editor(category="Commands")
@reloadable()
class CompositeCommand(Command):
    """Command composed of multiple sub-commands."""
    __slots__ = ("_commands", "_execute_in_order")

    def __init__(self, commands: list[Command] = None, name: str = "",
                 execute_in_order: bool = True):
        super().__init__(name or "Composite Command")
        self._commands = list(commands) if commands else []
        self._execute_in_order = execute_in_order

    @property
    def commands(self) -> list[Command]:
        """Get sub-commands."""
        return list(self._commands)

    def add(self, command: Command) -> None:
        """Add a sub-command."""
        self._commands.append(command)

    def remove(self, command: Command) -> bool:
        """Remove a sub-command."""
        if command in self._commands:
            self._commands.remove(command)
            return True
        return False

    def execute(self) -> bool:
        success = True
        executed = []

        for cmd in self._commands:
            if cmd.execute():
                executed.append(cmd)
            else:
                success = False
                # Rollback already executed commands
                for exec_cmd in reversed(executed):
                    exec_cmd.undo()
                break

        self._executed = success
        return success

    def undo(self) -> bool:
        # Undo in reverse order
        success = True
        for cmd in reversed(self._commands):
            if cmd.executed and not cmd.undo():
                success = False

        self._executed = False
        return success


@editor(category="Commands")
@reloadable()
class CommandBatch:
    """Batches multiple commands into a single undoable operation."""
    __slots__ = ("name", "_commands", "_manager_ref", "_started")

    def __init__(self, name: str, manager: "CommandManager"):
        self.name = name
        self._commands: list[Command] = []
        self._manager_ref = weakref.ref(manager)
        self._started = False

    def __enter__(self) -> "CommandBatch":
        self._started = True
        manager = self._manager_ref()
        if manager:
            manager._begin_batch(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        manager = self._manager_ref()
        if manager:
            if exc_type is None:
                manager._end_batch()
            else:
                manager._cancel_batch()
        self._started = False
        return False

    def add(self, command: Command) -> None:
        """Add a command to the batch."""
        if self._started:
            self._commands.append(command)


@editor(category="Commands")
@reloadable(preserve=["_undo_stack", "_redo_stack"])
class CommandManager:
    """Manages command execution and undo/redo history."""
    __slots__ = ("_undo_stack", "_redo_stack", "_max_history",
                 "_current_batch", "_tracker_ref", "on_execute",
                 "on_undo", "on_redo", "merge_timeout", "__weakref__")

    def __init__(self, max_history: int = 100):
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []
        self._max_history = max_history
        self._current_batch: Optional[CommandBatch] = None
        self._tracker_ref: Any = None
        self.on_execute: Optional[Callable[[Command], None]] = None
        self.on_undo: Optional[Callable[[Command], None]] = None
        self.on_redo: Optional[Callable[[Command], None]] = None
        self.merge_timeout: float = 1.0  # Seconds between commands for merge

    def set_tracker(self, tracker: Any) -> None:
        """Set the Foundation Tracker for integration."""
        self._tracker_ref = tracker

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    @property
    def undo_stack(self) -> list[Command]:
        """Get the undo stack (copy)."""
        return list(self._undo_stack)

    @property
    def redo_stack(self) -> list[Command]:
        """Get the redo stack (copy)."""
        return list(self._redo_stack)

    def execute(self, command: Command) -> bool:
        """Execute a command and add to history."""
        import time

        if self._current_batch:
            self._current_batch.add(command)
            return command.execute()

        # Try to merge with previous command
        if self._undo_stack:
            last_cmd = self._undo_stack[-1]
            time_diff = time.time() - last_cmd.timestamp
            if time_diff < self.merge_timeout and last_cmd.can_merge(command):
                merged = last_cmd.merge(command)
                if merged:
                    self._undo_stack[-1] = merged
                    merged.timestamp = time.time()
                    merged._executed = True
                    if command.execute():
                        return True

        # Normal execution
        if command.execute():
            command.timestamp = time.time()
            self._undo_stack.append(command)
            self._redo_stack.clear()

            # Enforce history limit
            while len(self._undo_stack) > self._max_history:
                self._undo_stack.pop(0)

            if self.on_execute:
                self.on_execute(command)

            # Integrate with Foundation Tracker if available
            if self._tracker_ref:
                tracker = self._tracker_ref
                if hasattr(tracker, "mark_dirty"):
                    # Mark affected objects as dirty
                    pass

            return True
        return False

    def undo(self) -> bool:
        """Undo the last command."""
        if not self._undo_stack:
            return False

        command = self._undo_stack.pop()
        if command.undo():
            self._redo_stack.append(command)
            if self.on_undo:
                self.on_undo(command)
            return True

        # Failed to undo, put it back
        self._undo_stack.append(command)
        return False

    def redo(self) -> bool:
        """Redo the last undone command."""
        if not self._redo_stack:
            return False

        command = self._redo_stack.pop()
        if command.redo():
            self._undo_stack.append(command)
            if self.on_redo:
                self.on_redo(command)
            return True

        # Failed to redo, put it back
        self._redo_stack.append(command)
        return False

    def undo_to(self, command: Command) -> int:
        """Undo to a specific command in history. Returns count undone."""
        count = 0
        while self._undo_stack and self._undo_stack[-1] != command:
            if self.undo():
                count += 1
            else:
                break
        return count

    def redo_to(self, command: Command) -> int:
        """Redo to a specific command. Returns count redone."""
        count = 0
        while self._redo_stack:
            if self._redo_stack[-1] == command:
                if self.redo():
                    count += 1
                break
            elif self.redo():
                count += 1
            else:
                break
        return count

    def clear(self) -> None:
        """Clear all history."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def batch(self, name: str) -> CommandBatch:
        """Create a command batch context manager."""
        return CommandBatch(name, self)

    def _begin_batch(self, batch: CommandBatch) -> None:
        """Begin a command batch (internal)."""
        self._current_batch = batch

    def _end_batch(self) -> None:
        """End a command batch (internal)."""
        if self._current_batch and self._current_batch._commands:
            composite = CompositeCommand(
                self._current_batch._commands,
                self._current_batch.name
            )
            composite._executed = all(c.executed for c in self._current_batch._commands)
            self._undo_stack.append(composite)
            self._redo_stack.clear()

            while len(self._undo_stack) > self._max_history:
                self._undo_stack.pop(0)

        self._current_batch = None

    def _cancel_batch(self) -> None:
        """Cancel a command batch (internal)."""
        if self._current_batch:
            # Undo all executed commands in reverse
            for cmd in reversed(self._current_batch._commands):
                if cmd.executed:
                    cmd.undo()
        self._current_batch = None

    def get_undo_description(self) -> Optional[str]:
        """Get description of the next undo action."""
        if self._undo_stack:
            cmd = self._undo_stack[-1]
            return cmd.description or cmd.name
        return None

    def get_redo_description(self) -> Optional[str]:
        """Get description of the next redo action."""
        if self._redo_stack:
            cmd = self._redo_stack[-1]
            return cmd.description or cmd.name
        return None
