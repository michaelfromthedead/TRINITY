"""
Command Pattern - Reversible action implementation for undo/redo.

Provides a command pattern implementation where each action is encapsulated
as an object that knows how to execute and reverse itself.
"""
from __future__ import annotations

import copy
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

from foundation import mirror


T = TypeVar("T")


class Command(ABC):
    """
    Abstract base class for undoable commands.

    Commands encapsulate actions that can be executed and reversed.
    They track the state needed to undo the operation.
    """

    def __init__(self, name: str = ""):
        """
        Initialize the command.

        Args:
            name: Human-readable name for the command.
        """
        self._name = name
        self._executed = False
        self._metadata: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Get the command name."""
        return self._name or self.__class__.__name__

    @property
    def executed(self) -> bool:
        """Check if command has been executed."""
        return self._executed

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get command metadata."""
        return self._metadata

    @abstractmethod
    def execute(self) -> bool:
        """
        Execute the command.

        Returns:
            True if execution succeeded.
        """
        pass

    @abstractmethod
    def unexecute(self) -> bool:
        """
        Reverse the command (undo).

        Returns:
            True if reversal succeeded.
        """
        pass

    def can_execute(self) -> bool:
        """Check if command can be executed."""
        return not self._executed

    def can_unexecute(self) -> bool:
        """Check if command can be unexecuted."""
        return self._executed

    def merge_with(self, other: "Command") -> Optional["Command"]:
        """
        Try to merge with another command.

        Override in subclasses to enable command merging.

        Args:
            other: Command to merge with.

        Returns:
            Merged command or None if cannot merge.
        """
        return None


class CompositeCommand(Command):
    """
    A command composed of multiple sub-commands.

    Executes all sub-commands in order, and undoes them in reverse order.
    """

    def __init__(self, name: str = "", commands: Optional[List[Command]] = None):
        """
        Initialize the composite command.

        Args:
            name: Name for the composite.
            commands: Initial list of commands.
        """
        super().__init__(name)
        self._commands: List[Command] = commands or []

    @property
    def commands(self) -> List[Command]:
        """Get the list of sub-commands."""
        return self._commands

    @property
    def command_count(self) -> int:
        """Number of sub-commands."""
        return len(self._commands)

    def add(self, command: Command) -> None:
        """Add a command to the composite."""
        self._commands.append(command)

    def execute(self) -> bool:
        """Execute all sub-commands in order."""
        if self._executed:
            return False

        executed = []
        try:
            for cmd in self._commands:
                if cmd.execute():
                    executed.append(cmd)
                else:
                    # Rollback on failure
                    for ec in reversed(executed):
                        ec.unexecute()
                    return False

            self._executed = True
            return True
        except Exception:
            # Rollback on exception
            for ec in reversed(executed):
                try:
                    ec.unexecute()
                except Exception:
                    pass
            return False

    def unexecute(self) -> bool:
        """Unexecute all sub-commands in reverse order."""
        if not self._executed:
            return False

        unexecuted = []
        try:
            for cmd in reversed(self._commands):
                if cmd.unexecute():
                    unexecuted.append(cmd)
                else:
                    # Re-execute on failure
                    for uc in reversed(unexecuted):
                        uc.execute()
                    return False

            self._executed = False
            return True
        except Exception:
            # Re-execute on exception
            for uc in reversed(unexecuted):
                try:
                    uc.execute()
                except Exception:
                    pass
            return False


class SetFieldCommand(Command):
    """Command to set a field value on an object."""

    def __init__(
        self,
        obj: Any,
        field_name: str,
        new_value: Any,
        name: str = "",
    ):
        """
        Initialize the set field command.

        Args:
            obj: Target object.
            field_name: Name of field to set.
            new_value: New value for the field.
            name: Optional command name.
        """
        super().__init__(name or f"Set {field_name}")
        self._obj_ref = weakref.ref(obj)
        self._field_name = field_name
        self._new_value = new_value
        self._old_value: Any = None

    @property
    def target(self) -> Optional[Any]:
        """Get the target object."""
        return self._obj_ref()

    @property
    def field_name(self) -> str:
        """Get the field name."""
        return self._field_name

    @property
    def old_value(self) -> Any:
        """Get the previous value."""
        return self._old_value

    @property
    def new_value(self) -> Any:
        """Get the new value."""
        return self._new_value

    def execute(self) -> bool:
        obj = self._obj_ref()
        if obj is None:
            return False

        try:
            # Store old value
            m = mirror(obj)
            self._old_value = m.get(self._field_name)

            # Set new value
            m.set(self._field_name, self._new_value)

            self._executed = True
            return True
        except (AttributeError, KeyError):
            return False

    def unexecute(self) -> bool:
        obj = self._obj_ref()
        if obj is None:
            return False

        try:
            m = mirror(obj)
            m.set(self._field_name, self._old_value)

            self._executed = False
            return True
        except (AttributeError, KeyError):
            return False

    def merge_with(self, other: Command) -> Optional[Command]:
        """Merge consecutive field sets on same object/field."""
        if not isinstance(other, SetFieldCommand):
            return None

        if (
            self._obj_ref() is other._obj_ref()
            and self._field_name == other._field_name
        ):
            # Create merged command with original old value
            merged = SetFieldCommand(
                obj=self._obj_ref(),
                field_name=self._field_name,
                new_value=other._new_value,
                name=self._name,
            )
            merged._old_value = self._old_value
            return merged

        return None


class CallMethodCommand(Command):
    """Command to call a method with undo/redo methods."""

    def __init__(
        self,
        obj: Any,
        do_method: str,
        undo_method: str,
        do_args: tuple = (),
        do_kwargs: Optional[Dict[str, Any]] = None,
        undo_args: tuple = (),
        undo_kwargs: Optional[Dict[str, Any]] = None,
        name: str = "",
    ):
        """
        Initialize the method call command.

        Args:
            obj: Target object.
            do_method: Method to call for execute.
            undo_method: Method to call for undo.
            do_args: Positional args for do method.
            do_kwargs: Keyword args for do method.
            undo_args: Positional args for undo method.
            undo_kwargs: Keyword args for undo method.
            name: Optional command name.
        """
        super().__init__(name or f"Call {do_method}")
        self._obj_ref = weakref.ref(obj)
        self._do_method = do_method
        self._undo_method = undo_method
        self._do_args = do_args
        self._do_kwargs = do_kwargs or {}
        self._undo_args = undo_args
        self._undo_kwargs = undo_kwargs or {}
        self._result: Any = None

    @property
    def result(self) -> Any:
        """Get the result of the last execute call."""
        return self._result

    def execute(self) -> bool:
        obj = self._obj_ref()
        if obj is None:
            return False

        method = getattr(obj, self._do_method, None)
        if not callable(method):
            return False

        try:
            self._result = method(*self._do_args, **self._do_kwargs)
            self._executed = True
            return True
        except Exception:
            return False

    def unexecute(self) -> bool:
        obj = self._obj_ref()
        if obj is None:
            return False

        method = getattr(obj, self._undo_method, None)
        if not callable(method):
            return False

        try:
            method(*self._undo_args, **self._undo_kwargs)
            self._executed = False
            return True
        except Exception:
            return False


class CreateObjectCommand(Command, Generic[T]):
    """Command to create an object with automatic cleanup on undo."""

    def __init__(
        self,
        factory: Callable[[], T],
        destroyer: Callable[[T], None],
        name: str = "Create Object",
    ):
        """
        Initialize the create object command.

        Args:
            factory: Function to create the object.
            destroyer: Function to destroy/cleanup the object.
            name: Command name.
        """
        super().__init__(name)
        self._factory = factory
        self._destroyer = destroyer
        self._object: Optional[T] = None

    @property
    def created_object(self) -> Optional[T]:
        """Get the created object."""
        return self._object

    def execute(self) -> bool:
        try:
            self._object = self._factory()
            self._executed = True
            return True
        except Exception:
            return False

    def unexecute(self) -> bool:
        if self._object is None:
            return False

        try:
            self._destroyer(self._object)
            self._object = None
            self._executed = False
            return True
        except Exception:
            return False


class DeleteObjectCommand(Command, Generic[T]):
    """Command to delete an object with restoration on undo."""

    def __init__(
        self,
        obj: T,
        deleter: Callable[[T], None],
        restorer: Callable[[T], None],
        name: str = "Delete Object",
    ):
        """
        Initialize the delete object command.

        Args:
            obj: Object to delete.
            deleter: Function to delete the object.
            restorer: Function to restore the object.
            name: Command name.
        """
        super().__init__(name)
        self._obj_ref = weakref.ref(obj)
        self._obj_backup: Optional[T] = None
        self._deleter = deleter
        self._restorer = restorer

    @property
    def deleted_object(self) -> Optional[T]:
        """Get the (backed up) deleted object."""
        return self._obj_backup

    def execute(self) -> bool:
        obj = self._obj_ref()
        if obj is None:
            return False

        try:
            # Backup before delete
            self._obj_backup = copy.deepcopy(obj)
            self._deleter(obj)
            self._executed = True
            return True
        except Exception:
            return False

    def unexecute(self) -> bool:
        if self._obj_backup is None:
            return False

        try:
            self._restorer(self._obj_backup)
            self._executed = False
            return True
        except Exception:
            return False


class CommandFactory:
    """
    Factory for creating common command types.

    Provides convenience methods for creating commands without
    directly instantiating command classes.
    """

    @staticmethod
    def set_field(
        obj: Any,
        field_name: str,
        value: Any,
        name: str = "",
    ) -> SetFieldCommand:
        """Create a SetFieldCommand."""
        return SetFieldCommand(obj, field_name, value, name)

    @staticmethod
    def set_fields(
        obj: Any,
        fields: Dict[str, Any],
        name: str = "Set Multiple Fields",
    ) -> CompositeCommand:
        """Create a composite command to set multiple fields."""
        composite = CompositeCommand(name)
        for field_name, value in fields.items():
            composite.add(SetFieldCommand(obj, field_name, value))
        return composite

    @staticmethod
    def call_method(
        obj: Any,
        do_method: str,
        undo_method: str,
        **kwargs: Any,
    ) -> CallMethodCommand:
        """Create a CallMethodCommand."""
        return CallMethodCommand(
            obj,
            do_method,
            undo_method,
            do_kwargs=kwargs.get("do_kwargs", {}),
            undo_kwargs=kwargs.get("undo_kwargs", {}),
            name=kwargs.get("name", ""),
        )

    @staticmethod
    def create_object(
        factory: Callable[[], T],
        destroyer: Callable[[T], None],
        name: str = "Create Object",
    ) -> CreateObjectCommand[T]:
        """Create a CreateObjectCommand."""
        return CreateObjectCommand(factory, destroyer, name)

    @staticmethod
    def delete_object(
        obj: T,
        deleter: Callable[[T], None],
        restorer: Callable[[T], None],
        name: str = "Delete Object",
    ) -> DeleteObjectCommand[T]:
        """Create a DeleteObjectCommand."""
        return DeleteObjectCommand(obj, deleter, restorer, name)


__all__ = [
    "Command",
    "CompositeCommand",
    "SetFieldCommand",
    "CallMethodCommand",
    "CreateObjectCommand",
    "DeleteObjectCommand",
    "CommandFactory",
]
