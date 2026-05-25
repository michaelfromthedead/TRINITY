"""
Transaction - Atomic operations with commit/rollback support.

Provides transaction grouping for multiple changes that should be
undone/redone as a single unit.
"""
from __future__ import annotations

import contextlib
import functools
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

from engine.tooling.undo.command_pattern import Command, CompositeCommand


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class TransactionState(Enum):
    """State of a transaction."""
    PENDING = auto()
    ACTIVE = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()
    FAILED = auto()


@dataclass
class Transaction:
    """
    A transaction grouping multiple commands.

    Transactions ensure atomicity - either all commands succeed,
    or all are rolled back.
    """

    name: str
    commands: List[Command] = field(default_factory=list)
    state: TransactionState = TransactionState.PENDING
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent: Optional["Transaction"] = None

    @property
    def is_active(self) -> bool:
        """Check if transaction is active."""
        return self.state == TransactionState.ACTIVE

    @property
    def is_complete(self) -> bool:
        """Check if transaction is complete (committed or rolled back)."""
        return self.state in {
            TransactionState.COMMITTED,
            TransactionState.ROLLED_BACK,
        }

    @property
    def command_count(self) -> int:
        """Number of commands in this transaction."""
        return len(self.commands)

    def add_command(self, command: Command) -> None:
        """Add a command to the transaction."""
        if not self.is_active:
            raise RuntimeError("Cannot add command to inactive transaction")
        self.commands.append(command)

    def begin(self) -> None:
        """Begin the transaction."""
        if self.state != TransactionState.PENDING:
            raise RuntimeError("Transaction already started")
        self.state = TransactionState.ACTIVE
        self.timestamp = time.time()

    def commit(self) -> bool:
        """
        Commit the transaction.

        Executes all commands. If any fails, rolls back.

        Returns:
            True if all commands succeeded.
        """
        if not self.is_active:
            raise RuntimeError("Cannot commit inactive transaction")

        executed = []
        try:
            for cmd in self.commands:
                if cmd.execute():
                    executed.append(cmd)
                else:
                    # Rollback executed commands
                    for ec in reversed(executed):
                        ec.unexecute()
                    self.state = TransactionState.FAILED
                    return False

            self.state = TransactionState.COMMITTED
            return True

        except Exception:
            # Rollback on exception
            for ec in reversed(executed):
                try:
                    ec.unexecute()
                except Exception:
                    pass
            self.state = TransactionState.FAILED
            return False

    def rollback(self) -> bool:
        """
        Rollback the transaction.

        Undoes all executed commands.

        Returns:
            True if rollback succeeded.
        """
        if self.state == TransactionState.PENDING:
            self.state = TransactionState.ROLLED_BACK
            return True

        if not self.is_active and self.state != TransactionState.COMMITTED:
            raise RuntimeError("Cannot rollback inactive transaction")

        try:
            for cmd in reversed(self.commands):
                if cmd.executed:
                    cmd.unexecute()

            self.state = TransactionState.ROLLED_BACK
            return True

        except Exception:
            self.state = TransactionState.FAILED
            return False

    def to_command(self) -> CompositeCommand:
        """Convert transaction to a composite command."""
        return CompositeCommand(self.name, list(self.commands))


class TransactionManager:
    """
    Manages transactions with support for nesting and savepoints.

    Features:
    - Nested transactions
    - Savepoints for partial rollback
    - Automatic rollback on exception
    - Thread-safe operation
    """

    def __init__(self):
        """Initialize the transaction manager."""
        self._lock = threading.RLock()
        self._current: Optional[Transaction] = None
        self._stack: List[Transaction] = []
        self._committed: List[Transaction] = []
        self._savepoints: Dict[str, int] = {}

    @property
    def in_transaction(self) -> bool:
        """Check if there's an active transaction."""
        return self._current is not None and self._current.is_active

    @property
    def current_transaction(self) -> Optional[Transaction]:
        """Get the current transaction."""
        return self._current

    @property
    def nesting_level(self) -> int:
        """Get the current nesting level."""
        return len(self._stack)

    def begin(self, name: str = "Transaction") -> Transaction:
        """
        Begin a new transaction.

        Args:
            name: Name of the transaction.

        Returns:
            The new Transaction.
        """
        with self._lock:
            txn = Transaction(name=name, parent=self._current)

            if self._current:
                self._stack.append(self._current)

            self._current = txn
            txn.begin()

            return txn

    def commit(self) -> bool:
        """
        Commit the current transaction.

        Returns:
            True if commit succeeded.
        """
        with self._lock:
            if not self._current:
                raise RuntimeError("No active transaction")

            result = self._current.commit()

            if result:
                self._committed.append(self._current)

                # Pop parent if nested
                if self._stack:
                    parent = self._stack.pop()
                    # Add nested commands to parent
                    parent.commands.extend(self._current.commands)
                    self._current = parent
                else:
                    self._current = None

            return result

    def rollback(self) -> bool:
        """
        Rollback the current transaction.

        Returns:
            True if rollback succeeded.
        """
        with self._lock:
            if not self._current:
                raise RuntimeError("No active transaction")

            result = self._current.rollback()

            # Pop parent if nested
            if self._stack:
                self._current = self._stack.pop()
            else:
                self._current = None

            return result

    def savepoint(self, name: str) -> str:
        """
        Create a savepoint in the current transaction.

        Args:
            name: Name of the savepoint.

        Returns:
            The savepoint name.
        """
        with self._lock:
            if not self._current:
                raise RuntimeError("No active transaction")

            self._savepoints[name] = len(self._current.commands)
            return name

    def rollback_to_savepoint(self, name: str) -> bool:
        """
        Rollback to a savepoint.

        Args:
            name: Name of the savepoint.

        Returns:
            True if rollback succeeded.
        """
        with self._lock:
            if not self._current:
                raise RuntimeError("No active transaction")

            if name not in self._savepoints:
                raise ValueError(f"Unknown savepoint: {name}")

            index = self._savepoints[name]

            # Undo commands added after savepoint
            for cmd in reversed(self._current.commands[index:]):
                if cmd.executed:
                    cmd.unexecute()

            self._current.commands = self._current.commands[:index]

            # Remove savepoints after this one
            to_remove = [
                sp for sp, idx in self._savepoints.items()
                if idx >= index
            ]
            for sp in to_remove:
                del self._savepoints[sp]

            return True

    def release_savepoint(self, name: str) -> None:
        """Release a savepoint (no longer needed)."""
        with self._lock:
            self._savepoints.pop(name, None)

    def add_command(self, command: Command) -> None:
        """
        Add a command to the current transaction.

        Args:
            command: Command to add.
        """
        with self._lock:
            if not self._current:
                raise RuntimeError("No active transaction")
            self._current.add_command(command)

    @contextlib.contextmanager
    def transaction(self, name: str = "Transaction") -> Generator[Transaction, None, None]:
        """
        Context manager for transactions.

        Automatically commits on success, rolls back on exception.

        Args:
            name: Transaction name.

        Yields:
            The Transaction object.
        """
        txn = self.begin(name)
        try:
            yield txn
            self.commit()
        except Exception:
            self.rollback()
            raise

    def clear(self) -> None:
        """Clear all transaction state."""
        with self._lock:
            self._current = None
            self._stack.clear()
            self._committed.clear()
            self._savepoints.clear()


# Global transaction manager
_transaction_manager: Optional[TransactionManager] = None


def get_transaction_manager() -> TransactionManager:
    """Get the global TransactionManager instance."""
    global _transaction_manager
    if _transaction_manager is None:
        _transaction_manager = TransactionManager()
    return _transaction_manager


def atomic(name: str = "Atomic Operation") -> Callable[[F], F]:
    """
    Decorator to wrap a function in a transaction.

    Args:
        name: Transaction name.

    Example:
        @atomic("Update Player")
        def update_player(player, health, position):
            player.health = health
            player.position = position
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_transaction_manager()
            with manager.transaction(name):
                return fn(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


__all__ = [
    "TransactionState",
    "Transaction",
    "TransactionManager",
    "get_transaction_manager",
    "atomic",
]
