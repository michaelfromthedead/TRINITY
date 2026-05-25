"""
Undo/Redo System - Transaction-based undo/redo for the AI Game Engine tooling layer.

This module provides:
- UndoSystem with integration to Foundation's Tracker
- Command pattern implementation for reversible actions
- Transaction grouping for atomic operations
- Undo history visualization with branching support
- Dirty tracking per-document/scene

Usage:
    from engine.tooling.undo import (
        UndoSystem,
        Command,
        Transaction,
        HistoryView,
        DirtyTracker,
    )
"""

from engine.tooling.undo.undo_system import (
    UndoSystem,
    UndoRedoError,
    UndoStackEmpty,
    RedoStackEmpty,
    UndoSystemConfig,
)

from engine.tooling.undo.command_pattern import (
    Command,
    CompositeCommand,
    SetFieldCommand,
    CallMethodCommand,
    CreateObjectCommand,
    DeleteObjectCommand,
    CommandFactory,
)

from engine.tooling.undo.transaction import (
    Transaction,
    TransactionState,
    TransactionManager,
    atomic,
)

from engine.tooling.undo.history_view import (
    HistoryView,
    HistoryNode,
    HistoryBranch,
    HistoryNavigator,
)

from engine.tooling.undo.dirty_tracking import (
    DirtyTracker,
    DirtyState,
    DocumentDirtyTracker,
    SavePromptResult,
)

__all__ = [
    # Undo system
    "UndoSystem",
    "UndoRedoError",
    "UndoStackEmpty",
    "RedoStackEmpty",
    "UndoSystemConfig",
    # Command pattern
    "Command",
    "CompositeCommand",
    "SetFieldCommand",
    "CallMethodCommand",
    "CreateObjectCommand",
    "DeleteObjectCommand",
    "CommandFactory",
    # Transactions
    "Transaction",
    "TransactionState",
    "TransactionManager",
    "atomic",
    # History view
    "HistoryView",
    "HistoryNode",
    "HistoryBranch",
    "HistoryNavigator",
    # Dirty tracking
    "DirtyTracker",
    "DirtyState",
    "DocumentDirtyTracker",
    "SavePromptResult",
]
