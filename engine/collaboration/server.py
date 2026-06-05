"""T-CC-4.6: Collaboration server with operation log.

Implements a collaboration server for real-time multi-user scene editing:

- CollaborationServer: Central authority maintaining world state
- ServerOperationLog: Persistent operation history for undo/replay
- ClientSession: Per-client state and connection management
- SyncProtocol: Push/pull synchronization protocol

The server uses CRDT merge for conflict resolution and maintains
a complete operation log for history navigation and undo support.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from .crdt import (
    CRDTDocument,
    CRDTOperation,
    OperationLog,
    OperationType,
    VectorClock,
)


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ServerError(Exception):
    """Base exception for server operations."""

    def __init__(self, message: str, code: str = "SERVER_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class SessionError(ServerError):
    """Error related to client sessions."""

    def __init__(self, message: str, session_id: Optional[str] = None):
        super().__init__(message, "SESSION_ERROR", {"session_id": session_id})
        self.session_id = session_id


class SyncError(ServerError):
    """Error during synchronization."""

    def __init__(self, message: str, client_id: Optional[str] = None):
        super().__init__(message, "SYNC_ERROR", {"client_id": client_id})
        self.client_id = client_id


class PersistenceError(ServerError):
    """Error during persistence operations."""

    def __init__(self, message: str, path: Optional[str] = None):
        super().__init__(message, "PERSISTENCE_ERROR", {"path": path})
        self.path = path


class DocumentNotFoundError(ServerError):
    """Requested document does not exist."""

    def __init__(self, doc_id: str):
        super().__init__(f"Document not found: {doc_id}", "DOC_NOT_FOUND", {"doc_id": doc_id})
        self.doc_id = doc_id


class ClientNotConnectedError(ServerError):
    """Client is not connected to the server."""

    def __init__(self, client_id: str):
        super().__init__(f"Client not connected: {client_id}", "CLIENT_NOT_CONNECTED", {"client_id": client_id})
        self.client_id = client_id


# =============================================================================
# Session State
# =============================================================================


class SessionState(Enum):
    """Client session states."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    SYNCING = auto()
    RECONNECTING = auto()
    ERROR = auto()


# =============================================================================
# Client Session
# =============================================================================


@dataclass
class ClientSession:
    """Represents a connected client's session state.

    Tracks:
    - Connection status and heartbeat
    - Last synced vector clock (for delta sync)
    - Subscribed documents
    - Pending operations queue
    """

    client_id: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.DISCONNECTED
    connected_at: Optional[float] = None
    last_heartbeat: float = field(default_factory=time.time)
    last_sync_clock: VectorClock = field(default_factory=VectorClock)
    subscribed_docs: Set[str] = field(default_factory=set)
    pending_operations: List[CRDTOperation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reconnect_token: Optional[str] = None
    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 5

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    def connect(self) -> None:
        """Mark session as connected."""
        with self._lock:
            self.state = SessionState.CONNECTED
            self.connected_at = time.time()
            self.last_heartbeat = time.time()
            self.reconnect_token = str(uuid.uuid4())
            self.reconnect_attempts = 0

    def disconnect(self) -> None:
        """Mark session as disconnected."""
        with self._lock:
            self.state = SessionState.DISCONNECTED
            self.connected_at = None

    def start_sync(self) -> None:
        """Mark session as syncing."""
        with self._lock:
            self.state = SessionState.SYNCING

    def end_sync(self, clock: VectorClock) -> None:
        """Mark sync as complete and update last sync clock."""
        with self._lock:
            self.state = SessionState.CONNECTED
            self.last_sync_clock = clock.copy()

    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        with self._lock:
            self.last_heartbeat = time.time()

    def is_alive(self, timeout: float = 30.0) -> bool:
        """Check if session is still alive based on heartbeat."""
        with self._lock:
            if self.state == SessionState.DISCONNECTED:
                return False
            return (time.time() - self.last_heartbeat) < timeout

    def subscribe(self, doc_id: str) -> None:
        """Subscribe to a document."""
        with self._lock:
            self.subscribed_docs.add(doc_id)

    def unsubscribe(self, doc_id: str) -> None:
        """Unsubscribe from a document."""
        with self._lock:
            self.subscribed_docs.discard(doc_id)

    def is_subscribed(self, doc_id: str) -> bool:
        """Check if subscribed to a document."""
        with self._lock:
            return doc_id in self.subscribed_docs

    def queue_operation(self, op: CRDTOperation) -> None:
        """Queue an operation for delivery to client."""
        with self._lock:
            self.pending_operations.append(op)

    def get_pending_operations(self, clear: bool = True) -> List[CRDTOperation]:
        """Get pending operations, optionally clearing the queue."""
        with self._lock:
            ops = list(self.pending_operations)
            if clear:
                self.pending_operations.clear()
            return ops

    def can_reconnect(self, token: str) -> bool:
        """Check if client can reconnect with given token."""
        with self._lock:
            if self.reconnect_token is None:
                return False
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                return False
            return token == self.reconnect_token

    def start_reconnect(self) -> None:
        """Start reconnection process."""
        with self._lock:
            self.state = SessionState.RECONNECTING
            self.reconnect_attempts += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        with self._lock:
            return {
                "client_id": self.client_id,
                "session_id": self.session_id,
                "state": self.state.name,
                "connected_at": self.connected_at,
                "last_heartbeat": self.last_heartbeat,
                "last_sync_clock": self.last_sync_clock.to_dict(),
                "subscribed_docs": list(self.subscribed_docs),
                "pending_count": len(self.pending_operations),
                "metadata": self.metadata,
                "reconnect_attempts": self.reconnect_attempts,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientSession":
        """Deserialize from dictionary."""
        session = cls(
            client_id=data["client_id"],
            session_id=data.get("session_id", str(uuid.uuid4())),
        )
        session.state = SessionState[data.get("state", "DISCONNECTED")]
        session.connected_at = data.get("connected_at")
        session.last_heartbeat = data.get("last_heartbeat", time.time())
        session.last_sync_clock = VectorClock.from_dict(data.get("last_sync_clock", {}))
        session.subscribed_docs = set(data.get("subscribed_docs", []))
        session.metadata = data.get("metadata", {})
        session.reconnect_attempts = data.get("reconnect_attempts", 0)
        return session


# =============================================================================
# Server Operation Log (Persistent)
# =============================================================================


class ServerOperationLog:
    """Persistent operation log for history, undo, and replay.

    Features:
    - Append-only log of all operations
    - Snapshot support for fast recovery
    - Range queries by time or vector clock
    - Undo/redo via inverse operations
    - Persistence to disk
    """

    def __init__(
        self,
        doc_id: str,
        persist_path: Optional[Path] = None,
        max_operations_in_memory: int = 10000,
        snapshot_interval: int = 1000,
    ):
        self.doc_id = doc_id
        self.persist_path = persist_path
        self.max_operations_in_memory = max_operations_in_memory
        self.snapshot_interval = snapshot_interval

        self._operations: List[CRDTOperation] = []
        self._operation_index: Dict[str, int] = {}  # op_id -> index
        self._operation_seq: Dict[str, int] = {}  # op_id -> sequence number
        self._clock = VectorClock()
        self._sequence_number: int = 0
        self._snapshots: List[Tuple[int, Dict[str, Any]]] = []  # (seq_num, snapshot)
        self._lock = threading.RLock()

        self._persisted_up_to: int = 0
        self._dirty = False

        if persist_path:
            self._load_from_disk()

    def append(self, op: CRDTOperation) -> int:
        """Append an operation to the log. Returns sequence number."""
        with self._lock:
            if op.id in self._operation_index:
                # Return existing sequence number for duplicates
                return self._operation_seq.get(op.id, 0)

            self._sequence_number += 1
            seq = self._sequence_number

            self._operations.append(op)
            self._operation_index[op.id] = len(self._operations) - 1
            self._operation_seq[op.id] = seq
            self._clock.merge_inplace(op.clock)
            self._dirty = True

            # Trim old operations if exceeding limit
            if len(self._operations) > self.max_operations_in_memory:
                self._compact()

            return seq

    def append_batch(self, ops: List[CRDTOperation]) -> List[int]:
        """Append multiple operations atomically."""
        with self._lock:
            sequences = []
            for op in ops:
                seq = self.append(op)
                sequences.append(seq)
            return sequences

    def get_operation(self, op_id: str) -> Optional[CRDTOperation]:
        """Get operation by ID."""
        with self._lock:
            idx = self._operation_index.get(op_id)
            if idx is not None and idx < len(self._operations):
                return self._operations[idx]
            return None

    def get_operations_since(
        self,
        clock: Optional[VectorClock] = None,
        limit: Optional[int] = None,
    ) -> List[CRDTOperation]:
        """Get operations since a given clock state."""
        with self._lock:
            if clock is None:
                ops = list(self._operations)
            else:
                ops = [op for op in self._operations if not clock.dominates(op.clock)]

            if limit:
                ops = ops[:limit]
            return ops

    def get_operations_in_range(
        self,
        start_seq: int,
        end_seq: Optional[int] = None,
    ) -> List[CRDTOperation]:
        """Get operations in a sequence number range."""
        with self._lock:
            if end_seq is None:
                end_seq = self._sequence_number

            # Find operations by sequence (approximate via index)
            start_idx = max(0, start_seq - 1)
            end_idx = min(len(self._operations), end_seq)
            return list(self._operations[start_idx:end_idx])

    def get_operations_by_node(self, node_id: str) -> List[CRDTOperation]:
        """Get all operations from a specific node."""
        with self._lock:
            return [op for op in self._operations if op.node_id == node_id]

    def get_operations_by_path(self, path_prefix: str) -> List[CRDTOperation]:
        """Get operations affecting a path prefix."""
        with self._lock:
            return [op for op in self._operations if op.path.startswith(path_prefix)]

    def get_last_n_operations(self, n: int) -> List[CRDTOperation]:
        """Get the last N operations."""
        with self._lock:
            return list(self._operations[-n:])

    @property
    def current_clock(self) -> VectorClock:
        """Get current vector clock."""
        with self._lock:
            return self._clock.copy()

    @property
    def sequence_number(self) -> int:
        """Get current sequence number."""
        with self._lock:
            return self._sequence_number

    def __len__(self) -> int:
        with self._lock:
            return len(self._operations)

    def _compact(self) -> None:
        """Compact log by removing old operations (keep in snapshots)."""
        # Take snapshot before compacting
        if len(self._operations) > 0:
            self._create_snapshot()

        # Keep only recent operations
        keep_count = self.max_operations_in_memory // 2
        if len(self._operations) > keep_count:
            # Persist before discarding
            if self.persist_path:
                self._persist_operations()

            # Remove old operations
            removed = self._operations[:-keep_count]
            self._operations = self._operations[-keep_count:]

            # Rebuild index
            self._operation_index.clear()
            for i, op in enumerate(self._operations):
                self._operation_index[op.id] = i

    def _create_snapshot(self) -> None:
        """Create a snapshot for recovery."""
        with self._lock:
            snapshot = {
                "sequence_number": self._sequence_number,
                "clock": self._clock.to_dict(),
                "operation_count": len(self._operations),
                "timestamp": time.time(),
            }
            self._snapshots.append((self._sequence_number, snapshot))

            # Keep only recent snapshots
            max_snapshots = 10
            if len(self._snapshots) > max_snapshots:
                self._snapshots = self._snapshots[-max_snapshots:]

    def create_snapshot(self, document_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a full snapshot including document state."""
        with self._lock:
            return {
                "doc_id": self.doc_id,
                "sequence_number": self._sequence_number,
                "clock": self._clock.to_dict(),
                "document_state": document_state,
                "timestamp": time.time(),
            }

    def _persist_operations(self) -> None:
        """Persist operations to disk."""
        if not self.persist_path:
            return

        with self._lock:
            ops_to_persist = self._operations[self._persisted_up_to:]
            if not ops_to_persist:
                return

            # Ensure directory exists
            self.persist_path.mkdir(parents=True, exist_ok=True)

            # Append to operations file
            ops_file = self.persist_path / f"{self.doc_id}_operations.jsonl"
            with open(ops_file, "a") as f:
                for op in ops_to_persist:
                    f.write(json.dumps(op.to_dict()) + "\n")

            self._persisted_up_to = len(self._operations)
            self._dirty = False

    def persist(self) -> None:
        """Force persistence of all pending operations."""
        self._persist_operations()

    def _load_from_disk(self) -> None:
        """Load operations from disk."""
        if not self.persist_path:
            return

        ops_file = self.persist_path / f"{self.doc_id}_operations.jsonl"
        if not ops_file.exists():
            return

        with self._lock:
            try:
                with open(ops_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            op_data = json.loads(line)
                            op = CRDTOperation.from_dict(op_data)
                            self._operations.append(op)
                            self._operation_index[op.id] = len(self._operations) - 1
                            self._clock.merge_inplace(op.clock)
                            self._sequence_number += 1

                self._persisted_up_to = len(self._operations)
                logger.info(f"Loaded {len(self._operations)} operations for {self.doc_id}")
            except Exception as e:
                logger.error(f"Failed to load operations: {e}")
                raise PersistenceError(f"Failed to load operations: {e}", str(ops_file))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize log metadata to dictionary."""
        with self._lock:
            return {
                "doc_id": self.doc_id,
                "sequence_number": self._sequence_number,
                "clock": self._clock.to_dict(),
                "operation_count": len(self._operations),
                "persisted_up_to": self._persisted_up_to,
                "dirty": self._dirty,
                "snapshot_count": len(self._snapshots),
            }

    def clear(self) -> None:
        """Clear all operations (use with caution)."""
        with self._lock:
            self._operations.clear()
            self._operation_index.clear()
            self._operation_seq.clear()
            self._clock = VectorClock()
            self._sequence_number = 0
            self._snapshots.clear()
            self._persisted_up_to = 0
            self._dirty = False


# =============================================================================
# Undo/Redo Support
# =============================================================================


@dataclass
class UndoEntry:
    """Entry in the undo stack."""

    operation_id: str
    inverse_operation: CRDTOperation
    original_value: Any
    timestamp: float = field(default_factory=time.time)


class UndoManager:
    """Manages undo/redo for a client.

    Tracks operations per-client for local undo support.
    Uses inverse operations to revert changes.
    """

    def __init__(self, client_id: str, max_undo_depth: int = 100):
        self.client_id = client_id
        self.max_undo_depth = max_undo_depth
        self._undo_stack: List[UndoEntry] = []
        self._redo_stack: List[UndoEntry] = []
        self._lock = threading.RLock()

    def push(self, op: CRDTOperation, inverse: CRDTOperation, original_value: Any = None) -> None:
        """Push an operation onto the undo stack."""
        with self._lock:
            entry = UndoEntry(
                operation_id=op.id,
                inverse_operation=inverse,
                original_value=original_value,
            )
            self._undo_stack.append(entry)

            # Clear redo stack on new operation
            self._redo_stack.clear()

            # Trim undo stack if too large
            if len(self._undo_stack) > self.max_undo_depth:
                self._undo_stack = self._undo_stack[-self.max_undo_depth:]

    def can_undo(self) -> bool:
        """Check if undo is available."""
        with self._lock:
            return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        with self._lock:
            return len(self._redo_stack) > 0

    def pop_undo(self) -> Optional[UndoEntry]:
        """Pop from undo stack and push to redo."""
        with self._lock:
            if not self._undo_stack:
                return None
            entry = self._undo_stack.pop()
            self._redo_stack.append(entry)
            return entry

    def pop_redo(self) -> Optional[UndoEntry]:
        """Pop from redo stack and push to undo."""
        with self._lock:
            if not self._redo_stack:
                return None
            entry = self._redo_stack.pop()
            self._undo_stack.append(entry)
            return entry

    def clear(self) -> None:
        """Clear both stacks."""
        with self._lock:
            self._undo_stack.clear()
            self._redo_stack.clear()

    @property
    def undo_count(self) -> int:
        with self._lock:
            return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        with self._lock:
            return len(self._redo_stack)


# =============================================================================
# Sync Protocol
# =============================================================================


class SyncMessageType(Enum):
    """Types of sync protocol messages."""

    # Client -> Server
    CONNECT = auto()
    DISCONNECT = auto()
    SUBSCRIBE = auto()
    UNSUBSCRIBE = auto()
    PUSH = auto()
    PULL = auto()
    HEARTBEAT = auto()
    RECONNECT = auto()

    # Server -> Client
    WELCOME = auto()
    ACK = auto()
    NACK = auto()
    DELTA = auto()
    FULL_SYNC = auto()
    BROADCAST = auto()
    ERROR = auto()


@dataclass
class SyncMessage:
    """Message in the sync protocol."""

    type: SyncMessageType
    client_id: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    doc_id: Optional[str] = None
    operations: List[CRDTOperation] = field(default_factory=list)
    clock: Optional[VectorClock] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": self.type.name,
            "client_id": self.client_id,
            "message_id": self.message_id,
            "doc_id": self.doc_id,
            "operations": [op.to_dict() for op in self.operations],
            "clock": self.clock.to_dict() if self.clock else None,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncMessage":
        """Deserialize from dictionary."""
        return cls(
            type=SyncMessageType[data["type"]],
            client_id=data["client_id"],
            message_id=data.get("message_id", str(uuid.uuid4())[:12]),
            doc_id=data.get("doc_id"),
            operations=[CRDTOperation.from_dict(op) for op in data.get("operations", [])],
            clock=VectorClock.from_dict(data["clock"]) if data.get("clock") else None,
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
        )

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "SyncMessage":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))


class SyncProtocol:
    """Handles sync protocol between clients and server.

    Protocol flow:
    1. CONNECT -> WELCOME (with current state)
    2. SUBSCRIBE -> ACK (subscribe to document)
    3. PUSH -> ACK/NACK (send local operations)
    4. PULL -> DELTA (request missing operations)
    5. HEARTBEAT -> ACK (keep connection alive)
    6. RECONNECT -> DELTA (recover from disconnect)
    """

    def __init__(self, server: "CollaborationServer"):
        self.server = server
        self._lock = threading.RLock()

    def handle_message(self, message: SyncMessage) -> SyncMessage:
        """Handle an incoming sync message and return response."""
        handlers = {
            SyncMessageType.CONNECT: self._handle_connect,
            SyncMessageType.DISCONNECT: self._handle_disconnect,
            SyncMessageType.SUBSCRIBE: self._handle_subscribe,
            SyncMessageType.UNSUBSCRIBE: self._handle_unsubscribe,
            SyncMessageType.PUSH: self._handle_push,
            SyncMessageType.PULL: self._handle_pull,
            SyncMessageType.HEARTBEAT: self._handle_heartbeat,
            SyncMessageType.RECONNECT: self._handle_reconnect,
        }

        handler = handlers.get(message.type)
        if handler:
            try:
                return handler(message)
            except Exception as e:
                logger.error(f"Error handling {message.type}: {e}")
                return self._error_response(message, str(e))
        else:
            return self._error_response(message, f"Unknown message type: {message.type}")

    def _handle_connect(self, message: SyncMessage) -> SyncMessage:
        """Handle client connection."""
        session = self.server.connect_client(
            message.client_id,
            metadata=message.payload.get("metadata", {}),
        )

        return SyncMessage(
            type=SyncMessageType.WELCOME,
            client_id=message.client_id,
            payload={
                "session_id": session.session_id,
                "reconnect_token": session.reconnect_token,
                "server_time": time.time(),
                "documents": list(self.server.list_documents()),
            },
        )

    def _handle_disconnect(self, message: SyncMessage) -> SyncMessage:
        """Handle client disconnection."""
        self.server.disconnect_client(message.client_id)

        return SyncMessage(
            type=SyncMessageType.ACK,
            client_id=message.client_id,
            payload={"disconnected": True},
        )

    def _handle_subscribe(self, message: SyncMessage) -> SyncMessage:
        """Handle document subscription."""
        if not message.doc_id:
            return self._error_response(message, "Missing doc_id")

        try:
            self.server.subscribe_client(message.client_id, message.doc_id)

            # Get current document state for initial sync
            doc = self.server.get_document(message.doc_id)

            return SyncMessage(
                type=SyncMessageType.FULL_SYNC,
                client_id=message.client_id,
                doc_id=message.doc_id,
                clock=doc.clock if doc else None,
                payload={
                    "document": doc.to_dict() if doc else None,
                    "subscribed": True,
                },
            )
        except DocumentNotFoundError:
            return self._error_response(message, f"Document not found: {message.doc_id}")

    def _handle_unsubscribe(self, message: SyncMessage) -> SyncMessage:
        """Handle document unsubscription."""
        if not message.doc_id:
            return self._error_response(message, "Missing doc_id")

        self.server.unsubscribe_client(message.client_id, message.doc_id)

        return SyncMessage(
            type=SyncMessageType.ACK,
            client_id=message.client_id,
            doc_id=message.doc_id,
            payload={"unsubscribed": True},
        )

    def _handle_push(self, message: SyncMessage) -> SyncMessage:
        """Handle client pushing operations."""
        if not message.doc_id:
            return self._error_response(message, "Missing doc_id")

        if not message.operations:
            return SyncMessage(
                type=SyncMessageType.ACK,
                client_id=message.client_id,
                doc_id=message.doc_id,
                payload={"applied": 0},
            )

        try:
            applied = self.server.apply_operations(
                message.doc_id,
                message.operations,
                from_client=message.client_id,
            )

            return SyncMessage(
                type=SyncMessageType.ACK,
                client_id=message.client_id,
                doc_id=message.doc_id,
                clock=self.server.get_document_clock(message.doc_id),
                payload={
                    "applied": applied,
                    "total": len(message.operations),
                },
            )
        except DocumentNotFoundError:
            return self._error_response(message, f"Document not found: {message.doc_id}")
        except Exception as e:
            return SyncMessage(
                type=SyncMessageType.NACK,
                client_id=message.client_id,
                doc_id=message.doc_id,
                payload={"error": str(e)},
            )

    def _handle_pull(self, message: SyncMessage) -> SyncMessage:
        """Handle client pulling operations."""
        if not message.doc_id:
            return self._error_response(message, "Missing doc_id")

        try:
            ops = self.server.get_operations_since(
                message.doc_id,
                message.clock,
                limit=message.payload.get("limit"),
            )

            return SyncMessage(
                type=SyncMessageType.DELTA,
                client_id=message.client_id,
                doc_id=message.doc_id,
                operations=ops,
                clock=self.server.get_document_clock(message.doc_id),
                payload={"count": len(ops)},
            )
        except DocumentNotFoundError:
            return self._error_response(message, f"Document not found: {message.doc_id}")

    def _handle_heartbeat(self, message: SyncMessage) -> SyncMessage:
        """Handle heartbeat."""
        session = self.server.get_session(message.client_id)
        if session:
            session.heartbeat()

        return SyncMessage(
            type=SyncMessageType.ACK,
            client_id=message.client_id,
            payload={"server_time": time.time()},
        )

    def _handle_reconnect(self, message: SyncMessage) -> SyncMessage:
        """Handle client reconnection."""
        reconnect_token = message.payload.get("reconnect_token")
        if not reconnect_token:
            return self._error_response(message, "Missing reconnect_token")

        try:
            session, missed_ops = self.server.reconnect_client(
                message.client_id,
                reconnect_token,
                message.clock,
            )

            return SyncMessage(
                type=SyncMessageType.DELTA,
                client_id=message.client_id,
                operations=missed_ops,
                clock=session.last_sync_clock,
                payload={
                    "session_id": session.session_id,
                    "reconnected": True,
                    "missed_count": len(missed_ops),
                },
            )
        except SessionError as e:
            return self._error_response(message, str(e))

    def _error_response(self, message: SyncMessage, error: str) -> SyncMessage:
        """Create an error response."""
        return SyncMessage(
            type=SyncMessageType.ERROR,
            client_id=message.client_id,
            doc_id=message.doc_id,
            payload={"error": error},
        )


# =============================================================================
# Message Handler Protocol
# =============================================================================


@runtime_checkable
class MessageHandler(Protocol):
    """Protocol for handling outgoing messages to clients."""

    def send_to_client(self, client_id: str, message: SyncMessage) -> None:
        """Send a message to a specific client."""
        ...

    def broadcast_to_subscribers(self, doc_id: str, message: SyncMessage, exclude_client: Optional[str] = None) -> None:
        """Broadcast a message to all subscribers of a document."""
        ...


# =============================================================================
# Collaboration Server
# =============================================================================


class CollaborationServer:
    """Central collaboration server maintaining authoritative world state.

    Responsibilities:
    - Maintain authoritative CRDT state for all documents
    - Process client operations and merge conflicts
    - Broadcast changes to subscribed clients
    - Manage client sessions and reconnection
    - Persist operation log for history/recovery
    """

    def __init__(
        self,
        server_id: Optional[str] = None,
        persist_path: Optional[Path] = None,
        heartbeat_timeout: float = 30.0,
        cleanup_interval: float = 60.0,
    ):
        self.server_id = server_id or f"server-{uuid.uuid4().hex[:8]}"
        self.persist_path = Path(persist_path) if persist_path else None
        self.heartbeat_timeout = heartbeat_timeout
        self.cleanup_interval = cleanup_interval

        # Documents (authoritative state)
        self._documents: Dict[str, CRDTDocument] = {}

        # Operation logs per document
        self._operation_logs: Dict[str, ServerOperationLog] = {}

        # Client sessions
        self._sessions: Dict[str, ClientSession] = {}

        # Document subscribers
        self._subscribers: Dict[str, Set[str]] = {}  # doc_id -> {client_ids}

        # Undo managers per client
        self._undo_managers: Dict[str, UndoManager] = {}

        # Sync protocol
        self._protocol = SyncProtocol(self)

        # Message handler (set externally)
        self._message_handler: Optional[MessageHandler] = None

        # Server state
        self._clock = VectorClock()
        self._running = False
        self._lock = threading.RLock()

        # Event callbacks
        self._on_operation: List[Callable[[str, CRDTOperation], None]] = []
        self._on_client_connect: List[Callable[[str], None]] = []
        self._on_client_disconnect: List[Callable[[str], None]] = []

        if self.persist_path:
            self._load_state()

    # -------------------------------------------------------------------------
    # Document Management
    # -------------------------------------------------------------------------

    def create_document(self, doc_id: str, initial_data: Optional[Dict[str, Any]] = None) -> CRDTDocument:
        """Create a new document."""
        with self._lock:
            if doc_id in self._documents:
                raise ServerError(f"Document already exists: {doc_id}")

            doc = CRDTDocument(doc_id, self.server_id)

            # Apply initial data if provided
            if initial_data:
                for key, value in initial_data.items():
                    doc.set_register(key, value)

            self._documents[doc_id] = doc
            self._operation_logs[doc_id] = ServerOperationLog(
                doc_id,
                persist_path=self.persist_path,
            )
            self._subscribers[doc_id] = set()

            logger.info(f"Created document: {doc_id}")
            return doc

    def get_document(self, doc_id: str) -> CRDTDocument:
        """Get a document by ID."""
        with self._lock:
            if doc_id not in self._documents:
                raise DocumentNotFoundError(doc_id)
            return self._documents[doc_id]

    def get_document_clock(self, doc_id: str) -> VectorClock:
        """Get the current vector clock for a document."""
        with self._lock:
            if doc_id not in self._documents:
                raise DocumentNotFoundError(doc_id)
            return self._documents[doc_id].clock

    def list_documents(self) -> List[str]:
        """List all document IDs."""
        with self._lock:
            return list(self._documents.keys())

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        with self._lock:
            if doc_id not in self._documents:
                return False

            # Unsubscribe all clients
            for client_id in list(self._subscribers.get(doc_id, [])):
                self.unsubscribe_client(client_id, doc_id)

            del self._documents[doc_id]
            if doc_id in self._operation_logs:
                del self._operation_logs[doc_id]
            if doc_id in self._subscribers:
                del self._subscribers[doc_id]

            logger.info(f"Deleted document: {doc_id}")
            return True

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists."""
        with self._lock:
            return doc_id in self._documents

    # -------------------------------------------------------------------------
    # Operation Handling
    # -------------------------------------------------------------------------

    def apply_operations(
        self,
        doc_id: str,
        operations: List[CRDTOperation],
        from_client: Optional[str] = None,
    ) -> int:
        """Apply operations to a document and broadcast to subscribers.

        Returns number of newly applied operations.
        """
        with self._lock:
            if doc_id not in self._documents:
                raise DocumentNotFoundError(doc_id)

            doc = self._documents[doc_id]
            op_log = self._operation_logs[doc_id]
            applied_count = 0

            for op in operations:
                # Apply to document
                if doc.apply_operation(op):
                    # Add to operation log
                    op_log.append(op)
                    applied_count += 1

                    # Fire callbacks
                    for callback in self._on_operation:
                        try:
                            callback(doc_id, op)
                        except Exception as e:
                            logger.error(f"Operation callback error: {e}")

            # Broadcast to other subscribers
            if applied_count > 0 and self._message_handler:
                broadcast_msg = SyncMessage(
                    type=SyncMessageType.BROADCAST,
                    client_id=self.server_id,
                    doc_id=doc_id,
                    operations=operations[:applied_count],
                    clock=doc.clock,
                )
                self._message_handler.broadcast_to_subscribers(
                    doc_id,
                    broadcast_msg,
                    exclude_client=from_client,
                )

            # Queue for offline subscribers
            for client_id in self._subscribers.get(doc_id, []):
                if client_id != from_client:
                    session = self._sessions.get(client_id)
                    if session and not session.is_alive(self.heartbeat_timeout):
                        for op in operations[:applied_count]:
                            session.queue_operation(op)

            return applied_count

    def get_operations_since(
        self,
        doc_id: str,
        clock: Optional[VectorClock] = None,
        limit: Optional[int] = None,
    ) -> List[CRDTOperation]:
        """Get operations since a given clock state."""
        with self._lock:
            if doc_id not in self._operation_logs:
                raise DocumentNotFoundError(doc_id)

            return self._operation_logs[doc_id].get_operations_since(clock, limit)

    def get_operation_log(self, doc_id: str) -> ServerOperationLog:
        """Get the operation log for a document."""
        with self._lock:
            if doc_id not in self._operation_logs:
                raise DocumentNotFoundError(doc_id)
            return self._operation_logs[doc_id]

    # -------------------------------------------------------------------------
    # Client Session Management
    # -------------------------------------------------------------------------

    def connect_client(
        self,
        client_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClientSession:
        """Connect a client and create a session."""
        with self._lock:
            # Check for existing session
            if client_id in self._sessions:
                session = self._sessions[client_id]
                if session.state != SessionState.DISCONNECTED:
                    # Already connected, just update heartbeat
                    session.heartbeat()
                    return session

            # Create new session
            session = ClientSession(
                client_id=client_id,
                metadata=metadata or {},
            )
            session.connect()
            self._sessions[client_id] = session
            self._undo_managers[client_id] = UndoManager(client_id)

            logger.info(f"Client connected: {client_id}")

            # Fire callbacks
            for callback in self._on_client_connect:
                try:
                    callback(client_id)
                except Exception as e:
                    logger.error(f"Connect callback error: {e}")

            return session

    def disconnect_client(self, client_id: str, clear_subscriptions: bool = False) -> None:
        """Disconnect a client.

        Args:
            client_id: The client to disconnect
            clear_subscriptions: If True, unsubscribe from all documents.
                                 If False (default), preserve subscriptions for reconnect.
        """
        with self._lock:
            if client_id not in self._sessions:
                return

            session = self._sessions[client_id]
            session.disconnect()

            # Optionally unsubscribe from all documents (not done by default for reconnect)
            if clear_subscriptions:
                for doc_id in list(session.subscribed_docs):
                    self.unsubscribe_client(client_id, doc_id)

            logger.info(f"Client disconnected: {client_id}")

            # Fire callbacks
            for callback in self._on_client_disconnect:
                try:
                    callback(client_id)
                except Exception as e:
                    logger.error(f"Disconnect callback error: {e}")

    def reconnect_client(
        self,
        client_id: str,
        reconnect_token: str,
        last_known_clock: Optional[VectorClock] = None,
    ) -> Tuple[ClientSession, List[CRDTOperation]]:
        """Reconnect a client and return missed operations."""
        with self._lock:
            if client_id not in self._sessions:
                raise SessionError(f"No session found for client: {client_id}", client_id)

            session = self._sessions[client_id]

            if not session.can_reconnect(reconnect_token):
                raise SessionError(f"Invalid reconnect token or max attempts exceeded", client_id)

            session.start_reconnect()

            # Gather missed operations from all subscribed documents
            missed_ops: List[CRDTOperation] = []
            sync_clock = last_known_clock or session.last_sync_clock

            for doc_id in session.subscribed_docs:
                if doc_id in self._operation_logs:
                    doc_ops = self._operation_logs[doc_id].get_operations_since(sync_clock)
                    missed_ops.extend(doc_ops)

            # Also include pending operations queued while disconnected
            missed_ops.extend(session.get_pending_operations(clear=True))

            # Sort by timestamp
            missed_ops.sort(key=lambda op: op.timestamp)

            session.connect()
            session.end_sync(self._get_combined_clock(session.subscribed_docs))

            logger.info(f"Client reconnected: {client_id} (missed {len(missed_ops)} operations)")

            return session, missed_ops

    def _get_combined_clock(self, doc_ids: Set[str]) -> VectorClock:
        """Get combined vector clock from multiple documents."""
        combined = VectorClock()
        for doc_id in doc_ids:
            if doc_id in self._documents:
                combined.merge_inplace(self._documents[doc_id].clock)
        return combined

    def get_session(self, client_id: str) -> Optional[ClientSession]:
        """Get a client's session."""
        with self._lock:
            return self._sessions.get(client_id)

    def list_clients(self) -> List[str]:
        """List all connected client IDs."""
        with self._lock:
            return [
                cid for cid, session in self._sessions.items()
                if session.state == SessionState.CONNECTED
            ]

    def is_client_connected(self, client_id: str) -> bool:
        """Check if a client is connected."""
        with self._lock:
            session = self._sessions.get(client_id)
            return session is not None and session.state == SessionState.CONNECTED

    # -------------------------------------------------------------------------
    # Subscription Management
    # -------------------------------------------------------------------------

    def subscribe_client(self, client_id: str, doc_id: str) -> None:
        """Subscribe a client to a document."""
        with self._lock:
            if client_id not in self._sessions:
                raise ClientNotConnectedError(client_id)

            if doc_id not in self._documents:
                raise DocumentNotFoundError(doc_id)

            session = self._sessions[client_id]
            session.subscribe(doc_id)

            if doc_id not in self._subscribers:
                self._subscribers[doc_id] = set()
            self._subscribers[doc_id].add(client_id)

            logger.debug(f"Client {client_id} subscribed to {doc_id}")

    def unsubscribe_client(self, client_id: str, doc_id: str) -> None:
        """Unsubscribe a client from a document."""
        with self._lock:
            if client_id in self._sessions:
                self._sessions[client_id].unsubscribe(doc_id)

            if doc_id in self._subscribers:
                self._subscribers[doc_id].discard(client_id)

            logger.debug(f"Client {client_id} unsubscribed from {doc_id}")

    def get_subscribers(self, doc_id: str) -> Set[str]:
        """Get all subscribers for a document."""
        with self._lock:
            return self._subscribers.get(doc_id, set()).copy()

    def get_subscriptions(self, client_id: str) -> Set[str]:
        """Get all documents a client is subscribed to."""
        with self._lock:
            session = self._sessions.get(client_id)
            if session:
                return session.subscribed_docs.copy()
            return set()

    # -------------------------------------------------------------------------
    # Undo/Redo
    # -------------------------------------------------------------------------

    def get_undo_manager(self, client_id: str) -> Optional[UndoManager]:
        """Get undo manager for a client."""
        with self._lock:
            return self._undo_managers.get(client_id)

    def undo(self, client_id: str, doc_id: str) -> Optional[CRDTOperation]:
        """Undo the last operation for a client."""
        with self._lock:
            undo_mgr = self._undo_managers.get(client_id)
            if not undo_mgr or not undo_mgr.can_undo():
                return None

            entry = undo_mgr.pop_undo()
            if not entry:
                return None

            # Apply inverse operation
            self.apply_operations(doc_id, [entry.inverse_operation], from_client=client_id)
            return entry.inverse_operation

    def redo(self, client_id: str, doc_id: str) -> Optional[CRDTOperation]:
        """Redo the last undone operation for a client."""
        with self._lock:
            undo_mgr = self._undo_managers.get(client_id)
            if not undo_mgr or not undo_mgr.can_redo():
                return None

            entry = undo_mgr.pop_redo()
            if not entry:
                return None

            # The inverse of the inverse is approximately the original
            # In practice, we'd need to track the original operation
            return entry.inverse_operation

    # -------------------------------------------------------------------------
    # Sync Protocol
    # -------------------------------------------------------------------------

    def handle_message(self, message: SyncMessage) -> SyncMessage:
        """Handle a sync protocol message."""
        return self._protocol.handle_message(message)

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Set the handler for outgoing messages."""
        self._message_handler = handler

    # -------------------------------------------------------------------------
    # Event Callbacks
    # -------------------------------------------------------------------------

    def on_operation(self, callback: Callable[[str, CRDTOperation], None]) -> None:
        """Register a callback for when operations are applied."""
        self._on_operation.append(callback)

    def on_client_connect(self, callback: Callable[[str], None]) -> None:
        """Register a callback for client connections."""
        self._on_client_connect.append(callback)

    def on_client_disconnect(self, callback: Callable[[str], None]) -> None:
        """Register a callback for client disconnections."""
        self._on_client_disconnect.append(callback)

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def cleanup_stale_sessions(self) -> int:
        """Clean up stale sessions. Returns count cleaned."""
        with self._lock:
            stale = []
            for client_id, session in self._sessions.items():
                if not session.is_alive(self.heartbeat_timeout):
                    stale.append(client_id)

            for client_id in stale:
                self.disconnect_client(client_id)

            if stale:
                logger.info(f"Cleaned up {len(stale)} stale sessions")

            return len(stale)

    def persist_all(self) -> None:
        """Persist all operation logs."""
        with self._lock:
            for op_log in self._operation_logs.values():
                op_log.persist()

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self.persist_path:
            return

        # Load documents
        docs_dir = self.persist_path / "documents"
        if docs_dir.exists():
            for doc_file in docs_dir.glob("*.json"):
                try:
                    with open(doc_file, "r") as f:
                        doc_data = json.load(f)
                    doc = CRDTDocument.from_dict(doc_data)
                    self._documents[doc.doc_id] = doc
                    self._operation_logs[doc.doc_id] = ServerOperationLog(
                        doc.doc_id,
                        persist_path=self.persist_path,
                    )
                    self._subscribers[doc.doc_id] = set()
                    logger.info(f"Loaded document: {doc.doc_id}")
                except Exception as e:
                    logger.error(f"Failed to load document {doc_file}: {e}")

    def save_state(self) -> None:
        """Save state to disk."""
        if not self.persist_path:
            return

        with self._lock:
            # Save documents
            docs_dir = self.persist_path / "documents"
            docs_dir.mkdir(parents=True, exist_ok=True)

            for doc_id, doc in self._documents.items():
                doc_file = docs_dir / f"{doc_id}.json"
                with open(doc_file, "w") as f:
                    json.dump(doc.to_dict(), f, indent=2, default=str)

            # Persist operation logs
            self.persist_all()

    # -------------------------------------------------------------------------
    # Server Lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start the server."""
        with self._lock:
            if self._running:
                return
            self._running = True
            logger.info(f"Collaboration server started: {self.server_id}")

    def stop(self) -> None:
        """Stop the server."""
        with self._lock:
            if not self._running:
                return

            # Disconnect all clients
            for client_id in list(self._sessions.keys()):
                self.disconnect_client(client_id)

            # Persist state
            self.save_state()

            self._running = False
            logger.info(f"Collaboration server stopped: {self.server_id}")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        with self._lock:
            return self._running

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        with self._lock:
            connected_clients = sum(
                1 for s in self._sessions.values()
                if s.state == SessionState.CONNECTED
            )

            total_ops = sum(
                len(log) for log in self._operation_logs.values()
            )

            return {
                "server_id": self.server_id,
                "running": self._running,
                "documents": len(self._documents),
                "total_clients": len(self._sessions),
                "connected_clients": connected_clients,
                "total_operations": total_ops,
                "subscriptions": {
                    doc_id: len(subs)
                    for doc_id, subs in self._subscribers.items()
                },
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize server state to dictionary."""
        with self._lock:
            return {
                "server_id": self.server_id,
                "running": self._running,
                "documents": {doc_id: doc.to_dict() for doc_id, doc in self._documents.items()},
                "sessions": {cid: s.to_dict() for cid, s in self._sessions.items()},
                "subscribers": {doc_id: list(subs) for doc_id, subs in self._subscribers.items()},
                "operation_logs": {doc_id: log.to_dict() for doc_id, log in self._operation_logs.items()},
                "stats": self.get_stats(),
            }


# =============================================================================
# Async Collaboration Server
# =============================================================================


class AsyncCollaborationServer(CollaborationServer):
    """Async version of the collaboration server for use with asyncio.

    Wraps blocking operations in executors and provides async interfaces.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = None

    async def async_handle_message(self, message: SyncMessage) -> SyncMessage:
        """Handle a sync message asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.handle_message, message)

    async def async_apply_operations(
        self,
        doc_id: str,
        operations: List[CRDTOperation],
        from_client: Optional[str] = None,
    ) -> int:
        """Apply operations asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.apply_operations(doc_id, operations, from_client),
        )

    async def async_create_document(
        self,
        doc_id: str,
        initial_data: Optional[Dict[str, Any]] = None,
    ) -> CRDTDocument:
        """Create a document asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.create_document(doc_id, initial_data),
        )

    async def async_save_state(self) -> None:
        """Save state asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.save_state)

    async def run_cleanup_loop(self) -> None:
        """Run periodic cleanup of stale sessions."""
        while self.is_running:
            await asyncio.sleep(self.cleanup_interval)
            self.cleanup_stale_sessions()


# =============================================================================
# Client Stub (for testing)
# =============================================================================


class CollaborationClient:
    """Client-side stub for collaboration.

    Provides a simple interface for:
    - Connecting/disconnecting from server
    - Subscribing to documents
    - Pushing/pulling operations
    - Automatic reconnection
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        server: Optional[CollaborationServer] = None,
    ):
        self.client_id = client_id or f"client-{uuid.uuid4().hex[:8]}"
        self.server = server

        self._session_id: Optional[str] = None
        self._reconnect_token: Optional[str] = None
        self._local_docs: Dict[str, CRDTDocument] = {}
        self._subscribed_docs: Set[str] = set()
        self._connected = False
        self._lock = threading.RLock()

    def connect(self, server: Optional[CollaborationServer] = None) -> bool:
        """Connect to the server."""
        with self._lock:
            if server:
                self.server = server

            if not self.server:
                raise ServerError("No server specified")

            msg = SyncMessage(
                type=SyncMessageType.CONNECT,
                client_id=self.client_id,
                payload={"metadata": {"client_version": "1.0"}},
            )

            response = self.server.handle_message(msg)

            if response.type == SyncMessageType.WELCOME:
                self._session_id = response.payload.get("session_id")
                self._reconnect_token = response.payload.get("reconnect_token")
                self._connected = True
                return True

            return False

    def disconnect(self) -> None:
        """Disconnect from the server."""
        with self._lock:
            if not self.server or not self._connected:
                return

            msg = SyncMessage(
                type=SyncMessageType.DISCONNECT,
                client_id=self.client_id,
            )

            self.server.handle_message(msg)
            self._connected = False
            self._subscribed_docs.clear()

    def subscribe(self, doc_id: str) -> Optional[CRDTDocument]:
        """Subscribe to a document and get initial state."""
        with self._lock:
            if not self.server or not self._connected:
                raise ClientNotConnectedError(self.client_id)

            msg = SyncMessage(
                type=SyncMessageType.SUBSCRIBE,
                client_id=self.client_id,
                doc_id=doc_id,
            )

            response = self.server.handle_message(msg)

            if response.type == SyncMessageType.FULL_SYNC:
                doc_data = response.payload.get("document")
                if doc_data:
                    doc = CRDTDocument.from_dict(doc_data)
                    self._local_docs[doc_id] = doc
                    self._subscribed_docs.add(doc_id)
                    return doc
            elif response.type == SyncMessageType.ERROR:
                raise ServerError(response.payload.get("error", "Unknown error"))

            return None

    def unsubscribe(self, doc_id: str) -> None:
        """Unsubscribe from a document."""
        with self._lock:
            if not self.server or not self._connected:
                return

            msg = SyncMessage(
                type=SyncMessageType.UNSUBSCRIBE,
                client_id=self.client_id,
                doc_id=doc_id,
            )

            self.server.handle_message(msg)
            self._subscribed_docs.discard(doc_id)
            if doc_id in self._local_docs:
                del self._local_docs[doc_id]

    def push(self, doc_id: str, operations: List[CRDTOperation]) -> int:
        """Push operations to the server."""
        with self._lock:
            if not self.server or not self._connected:
                raise ClientNotConnectedError(self.client_id)

            msg = SyncMessage(
                type=SyncMessageType.PUSH,
                client_id=self.client_id,
                doc_id=doc_id,
                operations=operations,
            )

            response = self.server.handle_message(msg)

            if response.type == SyncMessageType.ACK:
                return response.payload.get("applied", 0)
            elif response.type == SyncMessageType.NACK:
                raise SyncError(response.payload.get("error", "Push rejected"))
            elif response.type == SyncMessageType.ERROR:
                raise ServerError(response.payload.get("error", "Unknown error"))

            return 0

    def pull(self, doc_id: str, limit: Optional[int] = None) -> List[CRDTOperation]:
        """Pull operations from the server."""
        with self._lock:
            if not self.server or not self._connected:
                raise ClientNotConnectedError(self.client_id)

            local_doc = self._local_docs.get(doc_id)
            clock = local_doc.clock if local_doc else None

            msg = SyncMessage(
                type=SyncMessageType.PULL,
                client_id=self.client_id,
                doc_id=doc_id,
                clock=clock,
                payload={"limit": limit} if limit else {},
            )

            response = self.server.handle_message(msg)

            if response.type == SyncMessageType.DELTA:
                ops = response.operations

                # Apply to local document
                if local_doc and ops:
                    local_doc.apply_operations(ops)

                return ops
            elif response.type == SyncMessageType.ERROR:
                raise ServerError(response.payload.get("error", "Unknown error"))

            return []

    def heartbeat(self) -> bool:
        """Send heartbeat to server."""
        with self._lock:
            if not self.server or not self._connected:
                return False

            msg = SyncMessage(
                type=SyncMessageType.HEARTBEAT,
                client_id=self.client_id,
            )

            response = self.server.handle_message(msg)
            return response.type == SyncMessageType.ACK

    def reconnect(self) -> Tuple[bool, List[CRDTOperation]]:
        """Attempt to reconnect to the server."""
        with self._lock:
            if not self.server or not self._reconnect_token:
                return False, []

            # Get combined clock from local docs
            combined_clock = VectorClock()
            for doc in self._local_docs.values():
                combined_clock.merge_inplace(doc.clock)

            msg = SyncMessage(
                type=SyncMessageType.RECONNECT,
                client_id=self.client_id,
                clock=combined_clock,
                payload={"reconnect_token": self._reconnect_token},
            )

            response = self.server.handle_message(msg)

            if response.type == SyncMessageType.DELTA:
                self._connected = True
                self._session_id = response.payload.get("session_id")

                # Apply missed operations
                ops = response.operations
                for doc_id, doc in self._local_docs.items():
                    doc_ops = [op for op in ops if op.path.startswith(f"doc:{doc_id}") or True]
                    doc.apply_operations(doc_ops)

                return True, ops
            elif response.type == SyncMessageType.ERROR:
                return False, []

            return False, []

    def get_local_document(self, doc_id: str) -> Optional[CRDTDocument]:
        """Get local copy of a document."""
        with self._lock:
            return self._local_docs.get(doc_id)

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        with self._lock:
            return self._connected

    @property
    def subscriptions(self) -> Set[str]:
        """Get subscribed document IDs."""
        with self._lock:
            return self._subscribed_docs.copy()
