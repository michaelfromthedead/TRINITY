"""TRINITY Collaboration Module.

Provides CRDT (Conflict-free Replicated Data Types) for real-time
collaborative scene editing with automatic conflict resolution.

Components:
- CRDT types: VectorClock, LWWRegister, GCounter, PNCounter, ORSet, LWWMap
- CRDTDocument: Scene/entity document with multiple CRDT fields
- CollaborationServer: Central authority maintaining world state
- CollaborationClient: Client-side stub for collaboration
- ServerOperationLog: Persistent operation history
- SyncProtocol: Push/pull synchronization protocol
- Presence: Soft locking and real-time presence tracking
"""

from .crdt import (
    # Vector Clock
    VectorClock,

    # CRDT Types
    LWWRegister,
    GCounter,
    PNCounter,
    ORSet,
    LWWMap,

    # Document
    CRDTDocument,

    # Operations
    CRDTOperation,
    OperationType,
    OperationLog,

    # Exceptions
    CRDTError,
    CausalityViolation,
    MergeConflict,
)

from .server import (
    # Server
    CollaborationServer,
    AsyncCollaborationServer,

    # Client
    CollaborationClient,

    # Session
    ClientSession,
    SessionState,

    # Operation Log
    ServerOperationLog,

    # Undo/Redo
    UndoManager,
    UndoEntry,

    # Sync Protocol
    SyncProtocol,
    SyncMessage,
    SyncMessageType,

    # Exceptions
    ServerError,
    SessionError,
    SyncError,
    PersistenceError,
    DocumentNotFoundError,
    ClientNotConnectedError,
)

from .presence import (
    # Data Structures
    CursorPosition,
    Selection,

    # Soft Lock
    SoftLock,
    LockType,
    LockPriority,
    LockConflictNotification,

    # Lock Registry
    LockRegistry,

    # Presence
    PresenceInfo,
    PresenceStatus,
    PresenceManager,

    # Collaborative Session
    CollaborativeSession,

    # Exceptions
    PresenceError,
    LockError,
    LockNotFound,
    LockConflict,
    LockExpired,
    UserNotFound,
)

__all__ = [
    # Vector Clock
    "VectorClock",

    # CRDT Types
    "LWWRegister",
    "GCounter",
    "PNCounter",
    "ORSet",
    "LWWMap",

    # Document
    "CRDTDocument",

    # Operations
    "CRDTOperation",
    "OperationType",
    "OperationLog",

    # CRDT Exceptions
    "CRDTError",
    "CausalityViolation",
    "MergeConflict",

    # Server
    "CollaborationServer",
    "AsyncCollaborationServer",

    # Client
    "CollaborationClient",

    # Session
    "ClientSession",
    "SessionState",

    # Operation Log
    "ServerOperationLog",

    # Undo/Redo
    "UndoManager",
    "UndoEntry",

    # Sync Protocol
    "SyncProtocol",
    "SyncMessage",
    "SyncMessageType",

    # Server Exceptions
    "ServerError",
    "SessionError",
    "SyncError",
    "PersistenceError",
    "DocumentNotFoundError",
    "ClientNotConnectedError",

    # Presence - Data Structures
    "CursorPosition",
    "Selection",

    # Presence - Soft Lock
    "SoftLock",
    "LockType",
    "LockPriority",
    "LockConflictNotification",

    # Presence - Lock Registry
    "LockRegistry",

    # Presence - Presence Tracking
    "PresenceInfo",
    "PresenceStatus",
    "PresenceManager",

    # Presence - Collaborative Session
    "CollaborativeSession",

    # Presence - Exceptions
    "PresenceError",
    "LockError",
    "LockNotFound",
    "LockConflict",
    "LockExpired",
    "UserNotFound",
]
