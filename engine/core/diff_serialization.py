"""T-CC-2.8: Diff-based serialization for undo/network delta.

Builds on DiffWriter/DiffReader from serialization_formats.py to provide:
- DiffSerializer: Compare two snapshots and produce minimal delta
- DiffApplier: Apply delta to restore state
- UndoStack: Efficient undo/redo using deltas (not full snapshots)
- NetworkDelta: Wire-efficient state sync packets
"""
from __future__ import annotations

import copy
import hashlib
import json
import struct
import threading
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, IntEnum, auto
from io import BytesIO
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from .serialization import (
    SchemaInfo,
    SchemaVersion,
    Serializable,
    SerializationContext,
    SerializationError,
    SerializationFormat,
    _serialize_value,
    serializable,
)
from .serialization_formats import (
    DiffEntry,
    DiffPatch,
    DiffReader,
    DiffWriter,
    compute_diff,
    apply_diff,
)


T = TypeVar('T')


class DiffOperation(IntEnum):
    """Operations in a diff entry."""
    ADD = 1
    REMOVE = 2
    REPLACE = 3
    MOVE = 4  # For array reordering
    COPY = 5  # For duplicating values


@dataclass
class DiffMeta:
    """Metadata for a diff operation."""
    timestamp: float = field(default_factory=time.time)
    author: Optional[str] = None
    description: Optional[str] = None
    source_hash: Optional[str] = None
    target_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "author": self.author,
            "description": self.description,
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiffMeta":
        return cls(
            timestamp=data.get("timestamp", time.time()),
            author=data.get("author"),
            description=data.get("description"),
            source_hash=data.get("source_hash"),
            target_hash=data.get("target_hash"),
        )


@dataclass
class SerializedDiff:
    """A complete serialized diff with metadata."""
    entries: List[DiffEntry] = field(default_factory=list)
    meta: DiffMeta = field(default_factory=DiffMeta)
    compressed: bool = False
    _raw_bytes: Optional[bytes] = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[DiffEntry]:
        return iter(self.entries)

    def is_empty(self) -> bool:
        """Check if diff has no changes."""
        return len(self.entries) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "meta": self.meta.to_dict(),
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerializedDiff":
        return cls(
            entries=[DiffEntry.from_dict(e) for e in data.get("entries", [])],
            meta=DiffMeta.from_dict(data.get("meta", {})),
            compressed=data.get("compressed", False),
        )

    def to_bytes(self, compress: bool = True) -> bytes:
        """Serialize diff to bytes."""
        json_data = json.dumps(self.to_dict()).encode('utf-8')

        if compress and len(json_data) > 64:
            compressed = zlib.compress(json_data)
            if len(compressed) < len(json_data):
                # Header: 1 byte flag + 4 bytes original length
                header = struct.pack('<BI', 1, len(json_data))
                return header + compressed

        # Uncompressed: 1 byte flag
        return struct.pack('<B', 0) + json_data

    @classmethod
    def from_bytes(cls, data: bytes) -> "SerializedDiff":
        """Deserialize diff from bytes."""
        flag = struct.unpack('<B', data[:1])[0]

        if flag == 1:
            # Compressed
            original_len = struct.unpack('<I', data[1:5])[0]
            json_data = zlib.decompress(data[5:])
        else:
            json_data = data[1:]

        return cls.from_dict(json.loads(json_data.decode('utf-8')))

    def size_bytes(self) -> int:
        """Get size in bytes."""
        if self._raw_bytes is not None:
            return len(self._raw_bytes)
        return len(self.to_bytes())

    def invert(self) -> "SerializedDiff":
        """Create an inverted diff (for undo)."""
        inverted_entries = []
        for entry in reversed(self.entries):
            if entry.operation == "add":
                inverted_entries.append(DiffEntry(
                    path=entry.path,
                    operation="remove",
                    old_value=entry.new_value,
                ))
            elif entry.operation == "remove":
                inverted_entries.append(DiffEntry(
                    path=entry.path,
                    operation="add",
                    new_value=entry.old_value,
                ))
            elif entry.operation == "replace":
                inverted_entries.append(DiffEntry(
                    path=entry.path,
                    operation="replace",
                    old_value=entry.new_value,
                    new_value=entry.old_value,
                ))

        return SerializedDiff(
            entries=inverted_entries,
            meta=DiffMeta(
                timestamp=time.time(),
                author=self.meta.author,
                description=f"Invert: {self.meta.description}" if self.meta.description else "Inverted",
                source_hash=self.meta.target_hash,
                target_hash=self.meta.source_hash,
            ),
        )


def _compute_hash(obj: Any) -> str:
    """Compute a hash for any object."""
    if hasattr(obj, 'serialize'):
        ctx = SerializationContext(include_schema=False)
        data = obj.serialize(ctx)
    elif isinstance(obj, dict):
        data = obj
    else:
        data = str(obj)

    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode()).hexdigest()[:12]


def _flatten_object(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested object to path -> value mapping."""
    result = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("__"):
                continue
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict) and not any(k.startswith("__") for k in v.keys()):
                result.update(_flatten_object(v, path))
            elif isinstance(v, list):
                result.update(_flatten_object(v, path))
            else:
                result[path] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                result.update(_flatten_object(v, path))
            else:
                result[path] = v
    else:
        if prefix:
            result[prefix] = obj

    return result


def _unflatten_object(flat: Dict[str, Any]) -> Any:
    """Unflatten path -> value mapping back to nested object."""
    if not flat:
        return {}

    result: Dict[str, Any] = {}

    for path, value in flat.items():
        parts = _parse_path(path)
        _set_nested_value(result, parts, value)

    return result


def _parse_path(path: str) -> List[Union[str, int]]:
    """Parse a dot-notation path with array indices."""
    if path == "$" or not path:
        return []

    parts: List[Union[str, int]] = []
    current = ""
    i = 0

    while i < len(path):
        char = path[i]
        if char == ".":
            if current:
                parts.append(current)
                current = ""
        elif char == "[":
            if current:
                parts.append(current)
                current = ""
            j = i + 1
            while j < len(path) and path[j] != "]":
                j += 1
            parts.append(int(path[i+1:j]))
            i = j
        else:
            current += char
        i += 1

    if current:
        parts.append(current)

    return parts


def _set_nested_value(obj: Dict[str, Any], parts: List[Union[str, int]], value: Any) -> None:
    """Set a nested value in a dict/list structure."""
    if not parts:
        return

    current = obj
    for i, part in enumerate(parts[:-1]):
        next_part = parts[i + 1]

        if isinstance(part, int):
            while len(current) <= part:
                current.append(None)
            if current[part] is None:
                current[part] = [] if isinstance(next_part, int) else {}
            current = current[part]
        else:
            if part not in current:
                current[part] = [] if isinstance(next_part, int) else {}
            current = current[part]

    last_part = parts[-1]
    if isinstance(last_part, int):
        while len(current) <= last_part:
            current.append(None)
        current[last_part] = value
    else:
        current[last_part] = value


class DiffSerializer:
    """Compares two snapshots and produces a minimal delta.

    Works with any @serializable object or dict-like structure.
    Uses structural comparison for optimal diff size.
    """

    def __init__(
        self,
        include_metadata: bool = True,
        author: Optional[str] = None,
        array_diff_threshold: int = 10,
    ):
        """Initialize diff serializer.

        Args:
            include_metadata: Include timestamps and hashes
            author: Author name for diff metadata
            array_diff_threshold: Max array size for element-wise diff
        """
        self._include_metadata = include_metadata
        self._author = author
        self._array_diff_threshold = array_diff_threshold

    def compute(
        self,
        old_state: Any,
        new_state: Any,
        description: Optional[str] = None,
    ) -> SerializedDiff:
        """Compute diff between old and new state.

        Args:
            old_state: Previous state (dict or @serializable)
            new_state: New state (dict or @serializable)
            description: Optional description of the change

        Returns:
            SerializedDiff containing minimal delta
        """
        # Convert to dict representations
        old_data = self._to_dict(old_state)
        new_data = self._to_dict(new_state)

        # Flatten for comparison
        old_flat = _flatten_object(old_data)
        new_flat = _flatten_object(new_data)

        entries = []

        # Find additions and changes
        for path, new_val in new_flat.items():
            old_val = old_flat.get(path)
            if old_val is None and path not in old_flat:
                entries.append(DiffEntry(
                    path=path,
                    operation="add",
                    new_value=new_val,
                ))
            elif old_val != new_val:
                entries.append(DiffEntry(
                    path=path,
                    operation="replace",
                    old_value=old_val,
                    new_value=new_val,
                ))

        # Find removals
        for path, old_val in old_flat.items():
            if path not in new_flat:
                entries.append(DiffEntry(
                    path=path,
                    operation="remove",
                    old_value=old_val,
                ))

        # Build metadata
        meta = DiffMeta(
            timestamp=time.time() if self._include_metadata else 0,
            author=self._author,
            description=description,
            source_hash=_compute_hash(old_data) if self._include_metadata else None,
            target_hash=_compute_hash(new_data) if self._include_metadata else None,
        )

        return SerializedDiff(entries=entries, meta=meta)

    def compute_incremental(
        self,
        old_state: Any,
        new_state: Any,
        changed_paths: Optional[Set[str]] = None,
    ) -> SerializedDiff:
        """Compute diff only for specified changed paths (optimization).

        If changed_paths is provided, only those paths are compared.
        This is useful when you know which fields changed.
        """
        if changed_paths is None:
            return self.compute(old_state, new_state)

        old_data = self._to_dict(old_state)
        new_data = self._to_dict(new_state)

        old_flat = _flatten_object(old_data)
        new_flat = _flatten_object(new_data)

        entries = []

        for path in changed_paths:
            old_val = old_flat.get(path)
            new_val = new_flat.get(path)

            if old_val is None and path not in old_flat:
                if new_val is not None or path in new_flat:
                    entries.append(DiffEntry(
                        path=path,
                        operation="add",
                        new_value=new_val,
                    ))
            elif new_val is None and path not in new_flat:
                entries.append(DiffEntry(
                    path=path,
                    operation="remove",
                    old_value=old_val,
                ))
            elif old_val != new_val:
                entries.append(DiffEntry(
                    path=path,
                    operation="replace",
                    old_value=old_val,
                    new_value=new_val,
                ))

        return SerializedDiff(
            entries=entries,
            meta=DiffMeta(author=self._author),
        )

    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert object to dict representation."""
        if obj is None:
            return {}
        if hasattr(obj, 'serialize'):
            ctx = SerializationContext(include_schema=False)
            return obj.serialize(ctx)
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            result = {}
            for f in fields(obj):
                result[f.name] = _serialize_value(getattr(obj, f.name), SerializationContext())
            return result
        return {"$value": obj}


class DiffApplier:
    """Applies a delta to restore state.

    Can apply forward (for redo) or inverted (for undo) diffs.
    """

    def __init__(self, validate: bool = True):
        """Initialize diff applier.

        Args:
            validate: Validate old_values match before applying
        """
        self._validate = validate

    def apply(
        self,
        base_state: Any,
        diff: SerializedDiff,
        target_type: Optional[Type[T]] = None,
    ) -> Union[Dict[str, Any], T]:
        """Apply diff to base state.

        Args:
            base_state: State to apply diff to
            diff: Diff to apply
            target_type: Optional type to deserialize result into

        Returns:
            New state with diff applied
        """
        # Convert base to dict
        if hasattr(base_state, 'serialize'):
            ctx = SerializationContext(include_schema=False)
            data = base_state.serialize(ctx)
        elif isinstance(base_state, dict):
            data = copy.deepcopy(base_state)
        else:
            data = {"$value": base_state}

        # Apply each entry
        for entry in diff.entries:
            data = self._apply_entry(data, entry)

        # Convert back to target type if requested
        if target_type is not None and hasattr(target_type, 'deserialize'):
            return target_type.deserialize(data)

        return data

    def apply_inverted(
        self,
        base_state: Any,
        diff: SerializedDiff,
        target_type: Optional[Type[T]] = None,
    ) -> Union[Dict[str, Any], T]:
        """Apply inverted diff (undo operation).

        Args:
            base_state: State to apply inverted diff to
            diff: Diff to invert and apply
            target_type: Optional type to deserialize result into

        Returns:
            State with diff undone
        """
        inverted = diff.invert()
        return self.apply(base_state, inverted, target_type)

    def _apply_entry(self, data: Dict[str, Any], entry: DiffEntry) -> Dict[str, Any]:
        """Apply a single diff entry."""
        path_parts = _parse_path(entry.path)

        if not path_parts:
            # Root level change
            if entry.operation == "replace":
                return entry.new_value if isinstance(entry.new_value, dict) else {"$value": entry.new_value}
            elif entry.operation == "add":
                return entry.new_value if isinstance(entry.new_value, dict) else {"$value": entry.new_value}
            elif entry.operation == "remove":
                return {}
            return data

        # Navigate to parent and apply change
        return self._set_nested(data, path_parts, entry)

    def _set_nested(
        self,
        data: Any,
        path_parts: List[Union[str, int]],
        entry: DiffEntry,
    ) -> Any:
        """Set a nested value based on diff entry."""
        if not path_parts:
            if entry.operation == "remove":
                return None
            return entry.new_value

        # Make a copy to avoid mutating original
        if isinstance(data, dict):
            data = dict(data)
            key = str(path_parts[0])

            if len(path_parts) == 1:
                if entry.operation == "remove":
                    data.pop(key, None)
                else:
                    # Validate old value if enabled
                    if self._validate and entry.operation == "replace":
                        current = data.get(key)
                        if current != entry.old_value:
                            # Allow mismatch for robustness, but could raise
                            pass
                    data[key] = entry.new_value
            else:
                if key in data:
                    data[key] = self._set_nested(data[key], path_parts[1:], entry)
                elif entry.operation == "add":
                    # Create intermediate structure
                    data[key] = self._create_path(path_parts[1:], entry.new_value)
            return data

        elif isinstance(data, list):
            data = list(data)
            idx = path_parts[0]
            if not isinstance(idx, int):
                return data

            # Extend if needed
            while len(data) <= idx:
                data.append(None)

            if len(path_parts) == 1:
                if entry.operation == "remove":
                    if idx < len(data):
                        data.pop(idx)
                else:
                    data[idx] = entry.new_value
            else:
                if data[idx] is None:
                    data[idx] = {} if isinstance(path_parts[1], str) else []
                data[idx] = self._set_nested(data[idx], path_parts[1:], entry)
            return data

        return data

    def _create_path(self, path_parts: List[Union[str, int]], value: Any) -> Any:
        """Create nested structure for remaining path parts."""
        if not path_parts:
            return value

        key = path_parts[0]
        if isinstance(key, int):
            result: List[Any] = [None] * (key + 1)
            result[key] = self._create_path(path_parts[1:], value)
            return result
        else:
            return {str(key): self._create_path(path_parts[1:], value)}


@dataclass
class UndoEntry:
    """Entry in the undo stack."""
    forward_diff: SerializedDiff
    description: str
    timestamp: float = field(default_factory=time.time)

    def size_bytes(self) -> int:
        """Get memory size of this entry."""
        return self.forward_diff.size_bytes()


class UndoStack:
    """Undo/redo stack using diff-based serialization.

    Stores deltas instead of full snapshots for memory efficiency.
    Supports limited undo depth to bound memory usage.
    """

    def __init__(
        self,
        max_depth: int = 100,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB default
        merge_timeout: float = 0.5,  # Merge actions within 500ms
    ):
        """Initialize undo stack.

        Args:
            max_depth: Maximum number of undo steps
            max_bytes: Maximum memory usage in bytes
            merge_timeout: Time window for merging consecutive edits
        """
        self._max_depth = max_depth
        self._max_bytes = max_bytes
        self._merge_timeout = merge_timeout

        self._undo_stack: List[UndoEntry] = []
        self._redo_stack: List[UndoEntry] = []
        self._current_bytes = 0
        self._lock = threading.Lock()

        self._serializer = DiffSerializer()
        self._applier = DiffApplier()

        self._last_action_time = 0.0
        self._last_action_path: Optional[str] = None

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    @property
    def undo_depth(self) -> int:
        """Get current undo depth."""
        return len(self._undo_stack)

    @property
    def redo_depth(self) -> int:
        """Get current redo depth."""
        return len(self._redo_stack)

    @property
    def memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self._current_bytes

    def push(
        self,
        old_state: Any,
        new_state: Any,
        description: str = "",
        merge_path: Optional[str] = None,
    ) -> None:
        """Push a state change onto the undo stack.

        Args:
            old_state: State before the change
            new_state: State after the change
            description: Human-readable description
            merge_path: Path for merging consecutive changes
        """
        with self._lock:
            # Compute diff
            diff = self._serializer.compute(old_state, new_state, description)

            if diff.is_empty():
                return  # No actual change

            # Check for merge with previous action
            current_time = time.time()
            should_merge = (
                merge_path is not None
                and merge_path == self._last_action_path
                and (current_time - self._last_action_time) < self._merge_timeout
                and len(self._undo_stack) > 0
            )

            if should_merge:
                # Merge with previous entry
                prev_entry = self._undo_stack[-1]
                merged_entries = prev_entry.forward_diff.entries + diff.entries
                merged_diff = SerializedDiff(
                    entries=merged_entries,
                    meta=DiffMeta(
                        timestamp=prev_entry.timestamp,
                        description=f"{prev_entry.description} (merged)",
                    ),
                )
                self._current_bytes -= prev_entry.size_bytes()
                self._undo_stack[-1] = UndoEntry(
                    forward_diff=merged_diff,
                    description=prev_entry.description,
                    timestamp=prev_entry.timestamp,
                )
                self._current_bytes += self._undo_stack[-1].size_bytes()
            else:
                # Push new entry
                entry = UndoEntry(
                    forward_diff=diff,
                    description=description,
                )
                self._undo_stack.append(entry)
                self._current_bytes += entry.size_bytes()

            # Clear redo stack on new action
            for entry in self._redo_stack:
                self._current_bytes -= entry.size_bytes()
            self._redo_stack.clear()

            self._last_action_time = current_time
            self._last_action_path = merge_path

            # Enforce limits
            self._enforce_limits()

    def undo(self, current_state: Any, target_type: Optional[Type[T]] = None) -> Tuple[Any, str]:
        """Undo the last action.

        Args:
            current_state: Current state to apply undo to
            target_type: Optional type for result

        Returns:
            Tuple of (new_state, description of undone action)

        Raises:
            IndexError: If undo stack is empty
        """
        with self._lock:
            if not self._undo_stack:
                raise IndexError("Nothing to undo")

            entry = self._undo_stack.pop()
            self._current_bytes -= entry.size_bytes()

            # Apply inverted diff
            new_state = self._applier.apply_inverted(
                current_state,
                entry.forward_diff,
                target_type,
            )

            # Push to redo stack
            self._redo_stack.append(entry)
            self._current_bytes += entry.size_bytes()

            return new_state, entry.description

    def redo(self, current_state: Any, target_type: Optional[Type[T]] = None) -> Tuple[Any, str]:
        """Redo the last undone action.

        Args:
            current_state: Current state to apply redo to
            target_type: Optional type for result

        Returns:
            Tuple of (new_state, description of redone action)

        Raises:
            IndexError: If redo stack is empty
        """
        with self._lock:
            if not self._redo_stack:
                raise IndexError("Nothing to redo")

            entry = self._redo_stack.pop()
            self._current_bytes -= entry.size_bytes()

            # Apply forward diff
            new_state = self._applier.apply(
                current_state,
                entry.forward_diff,
                target_type,
            )

            # Push to undo stack
            self._undo_stack.append(entry)
            self._current_bytes += entry.size_bytes()

            return new_state, entry.description

    def peek_undo(self) -> Optional[str]:
        """Peek at the description of the next undo action."""
        with self._lock:
            if self._undo_stack:
                return self._undo_stack[-1].description
            return None

    def peek_redo(self) -> Optional[str]:
        """Peek at the description of the next redo action."""
        with self._lock:
            if self._redo_stack:
                return self._redo_stack[-1].description
            return None

    def get_undo_history(self, limit: int = 10) -> List[str]:
        """Get descriptions of recent undo actions."""
        with self._lock:
            return [e.description for e in reversed(self._undo_stack[-limit:])]

    def get_redo_history(self, limit: int = 10) -> List[str]:
        """Get descriptions of recent redo actions."""
        with self._lock:
            return [e.description for e in reversed(self._redo_stack[-limit:])]

    def clear(self) -> None:
        """Clear all undo/redo history."""
        with self._lock:
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._current_bytes = 0
            self._last_action_time = 0.0
            self._last_action_path = None

    def _enforce_limits(self) -> None:
        """Enforce depth and memory limits."""
        # Enforce depth limit
        while len(self._undo_stack) > self._max_depth:
            entry = self._undo_stack.pop(0)
            self._current_bytes -= entry.size_bytes()

        # Enforce memory limit
        while self._current_bytes > self._max_bytes and len(self._undo_stack) > 1:
            entry = self._undo_stack.pop(0)
            self._current_bytes -= entry.size_bytes()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the undo stack."""
        with self._lock:
            return {
                "undo_depth": len(self._undo_stack),
                "redo_depth": len(self._redo_stack),
                "memory_bytes": self._current_bytes,
                "max_depth": self._max_depth,
                "max_bytes": self._max_bytes,
            }


class NetworkDeltaFlags(IntEnum):
    """Flags for network delta packets."""
    NONE = 0
    COMPRESSED = 1
    HAS_SEQ = 2
    HAS_ACK = 4
    RELIABLE = 8
    ORDERED = 16


@dataclass
class NetworkDelta:
    """Efficient state sync packet for network transmission.

    Optimized for wire efficiency with:
    - Field-level granularity
    - Sequence numbers for ordering
    - Acknowledgment support
    - Compression for large deltas
    """

    entity_id: str
    sequence: int
    entries: List[DiffEntry] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    ack_sequence: Optional[int] = None
    flags: int = NetworkDeltaFlags.NONE

    # Wire format magic bytes
    MAGIC = b'NDLT'
    VERSION = 1

    def __len__(self) -> int:
        return len(self.entries)

    def is_empty(self) -> bool:
        """Check if delta has no changes."""
        return len(self.entries) == 0

    def add_change(self, path: str, old_value: Any, new_value: Any) -> None:
        """Add a field change to the delta."""
        if old_value is None and new_value is not None:
            self.entries.append(DiffEntry(
                path=path,
                operation="add",
                new_value=new_value,
            ))
        elif old_value is not None and new_value is None:
            self.entries.append(DiffEntry(
                path=path,
                operation="remove",
                old_value=old_value,
            ))
        elif old_value != new_value:
            self.entries.append(DiffEntry(
                path=path,
                operation="replace",
                old_value=old_value,
                new_value=new_value,
            ))

    def merge(self, other: "NetworkDelta") -> "NetworkDelta":
        """Merge another delta into this one."""
        if other.entity_id != self.entity_id:
            raise ValueError("Cannot merge deltas for different entities")

        # Combine entries, newer overwrites older for same path
        path_map = {e.path: e for e in self.entries}
        for entry in other.entries:
            path_map[entry.path] = entry

        return NetworkDelta(
            entity_id=self.entity_id,
            sequence=max(self.sequence, other.sequence),
            entries=list(path_map.values()),
            timestamp=max(self.timestamp, other.timestamp),
            ack_sequence=other.ack_sequence or self.ack_sequence,
            flags=self.flags | other.flags,
        )

    def to_bytes(self, compress: bool = True) -> bytes:
        """Serialize to wire format.

        Wire format:
        - Magic (4 bytes): 'NDLT'
        - Version (1 byte)
        - Flags (1 byte)
        - Entity ID length (2 bytes) + Entity ID
        - Sequence (4 bytes)
        - Ack sequence (4 bytes, if HAS_ACK)
        - Entry count (2 bytes)
        - Entries (variable)
        """
        buffer = BytesIO()

        # Header
        buffer.write(self.MAGIC)
        buffer.write(struct.pack('<B', self.VERSION))

        flags = self.flags
        if compress and len(self.entries) > 5:
            flags |= NetworkDeltaFlags.COMPRESSED

        buffer.write(struct.pack('<B', flags))

        # Entity ID
        entity_bytes = self.entity_id.encode('utf-8')
        buffer.write(struct.pack('<H', len(entity_bytes)))
        buffer.write(entity_bytes)

        # Sequence
        buffer.write(struct.pack('<I', self.sequence))

        # Ack sequence
        if flags & NetworkDeltaFlags.HAS_ACK:
            buffer.write(struct.pack('<I', self.ack_sequence or 0))

        # Entries
        entries_json = json.dumps([e.to_dict() for e in self.entries]).encode('utf-8')

        if flags & NetworkDeltaFlags.COMPRESSED:
            entries_data = zlib.compress(entries_json)
        else:
            entries_data = entries_json

        buffer.write(struct.pack('<I', len(entries_data)))
        buffer.write(entries_data)

        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "NetworkDelta":
        """Deserialize from wire format."""
        buffer = BytesIO(data)

        # Validate magic
        magic = buffer.read(4)
        if magic != cls.MAGIC:
            raise SerializationError(f"Invalid network delta magic: {magic!r}")

        version = struct.unpack('<B', buffer.read(1))[0]
        if version > cls.VERSION:
            raise SerializationError(f"Unsupported network delta version: {version}")

        flags = struct.unpack('<B', buffer.read(1))[0]

        # Entity ID
        entity_len = struct.unpack('<H', buffer.read(2))[0]
        entity_id = buffer.read(entity_len).decode('utf-8')

        # Sequence
        sequence = struct.unpack('<I', buffer.read(4))[0]

        # Ack sequence
        ack_sequence = None
        if flags & NetworkDeltaFlags.HAS_ACK:
            ack_sequence = struct.unpack('<I', buffer.read(4))[0]

        # Entries
        entries_len = struct.unpack('<I', buffer.read(4))[0]
        entries_data = buffer.read(entries_len)

        if flags & NetworkDeltaFlags.COMPRESSED:
            entries_json = zlib.decompress(entries_data)
        else:
            entries_json = entries_data

        entries_list = json.loads(entries_json.decode('utf-8'))
        entries = [DiffEntry.from_dict(e) for e in entries_list]

        return cls(
            entity_id=entity_id,
            sequence=sequence,
            entries=entries,
            ack_sequence=ack_sequence,
            flags=flags,
        )

    def size_bytes(self) -> int:
        """Get wire size in bytes."""
        return len(self.to_bytes())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "sequence": self.sequence,
            "entries": [e.to_dict() for e in self.entries],
            "timestamp": self.timestamp,
            "ack_sequence": self.ack_sequence,
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkDelta":
        """Create from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            sequence=data["sequence"],
            entries=[DiffEntry.from_dict(e) for e in data.get("entries", [])],
            timestamp=data.get("timestamp", time.time()),
            ack_sequence=data.get("ack_sequence"),
            flags=data.get("flags", NetworkDeltaFlags.NONE),
        )


class NetworkDeltaBuilder:
    """Builder for creating network deltas efficiently.

    Tracks field changes and creates minimal delta packets.
    """

    def __init__(
        self,
        entity_id: str,
        include_old_values: bool = True,
    ):
        """Initialize builder.

        Args:
            entity_id: Entity this delta is for
            include_old_values: Include old values (for validation/conflict detection)
        """
        self._entity_id = entity_id
        self._include_old_values = include_old_values
        self._sequence = 0
        self._pending_changes: Dict[str, Tuple[Any, Any]] = {}  # path -> (old, new)
        self._lock = threading.Lock()

    def track_change(self, path: str, old_value: Any, new_value: Any) -> None:
        """Track a field change.

        Args:
            path: Field path (e.g., "position.x")
            old_value: Previous value
            new_value: New value
        """
        with self._lock:
            if path in self._pending_changes:
                # Merge: keep original old, update new
                orig_old, _ = self._pending_changes[path]
                self._pending_changes[path] = (orig_old, new_value)
            else:
                self._pending_changes[path] = (old_value, new_value)

    def build(self, reliable: bool = False, ordered: bool = False) -> NetworkDelta:
        """Build a network delta from tracked changes.

        Args:
            reliable: Mark as requiring acknowledgment
            ordered: Mark as requiring ordering

        Returns:
            NetworkDelta with all tracked changes
        """
        with self._lock:
            entries = []

            for path, (old_val, new_val) in self._pending_changes.items():
                if old_val == new_val:
                    continue

                if old_val is None:
                    entries.append(DiffEntry(
                        path=path,
                        operation="add",
                        new_value=new_val,
                    ))
                elif new_val is None:
                    entries.append(DiffEntry(
                        path=path,
                        operation="remove",
                        old_value=old_val if self._include_old_values else None,
                    ))
                else:
                    entries.append(DiffEntry(
                        path=path,
                        operation="replace",
                        old_value=old_val if self._include_old_values else None,
                        new_value=new_val,
                    ))

            flags = NetworkDeltaFlags.NONE
            if reliable:
                flags |= NetworkDeltaFlags.RELIABLE
            if ordered:
                flags |= NetworkDeltaFlags.ORDERED

            self._sequence += 1

            delta = NetworkDelta(
                entity_id=self._entity_id,
                sequence=self._sequence,
                entries=entries,
                flags=flags,
            )

            self._pending_changes.clear()

            return delta

    def has_changes(self) -> bool:
        """Check if there are pending changes."""
        with self._lock:
            return any(old != new for old, new in self._pending_changes.values())

    def clear(self) -> None:
        """Clear pending changes without building."""
        with self._lock:
            self._pending_changes.clear()


class NetworkDeltaAccumulator:
    """Accumulates network deltas for batching.

    Useful for reducing network traffic by combining multiple
    small deltas into larger batched updates.
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_age_ms: float = 100.0,
    ):
        """Initialize accumulator.

        Args:
            max_entries: Max entries before auto-flush
            max_age_ms: Max age in milliseconds before auto-flush
        """
        self._max_entries = max_entries
        self._max_age_ms = max_age_ms

        self._deltas: Dict[str, NetworkDelta] = {}  # entity_id -> merged delta
        self._first_add_time: Optional[float] = None
        self._lock = threading.Lock()

    def add(self, delta: NetworkDelta) -> Optional[Dict[str, NetworkDelta]]:
        """Add a delta to the accumulator.

        Args:
            delta: Delta to add

        Returns:
            Accumulated deltas if flush was triggered, None otherwise
        """
        with self._lock:
            if self._first_add_time is None:
                self._first_add_time = time.time()

            # Merge with existing delta for this entity
            if delta.entity_id in self._deltas:
                self._deltas[delta.entity_id] = self._deltas[delta.entity_id].merge(delta)
            else:
                self._deltas[delta.entity_id] = delta

            # Check if we should flush
            total_entries = sum(len(d) for d in self._deltas.values())
            age_ms = (time.time() - self._first_add_time) * 1000

            if total_entries >= self._max_entries or age_ms >= self._max_age_ms:
                return self._flush()

            return None

    def flush(self) -> Dict[str, NetworkDelta]:
        """Force flush all accumulated deltas."""
        with self._lock:
            return self._flush()

    def _flush(self) -> Dict[str, NetworkDelta]:
        """Internal flush (must hold lock)."""
        result = self._deltas
        self._deltas = {}
        self._first_add_time = None
        return result

    def is_empty(self) -> bool:
        """Check if accumulator is empty."""
        with self._lock:
            return len(self._deltas) == 0


def create_diff_serializer(
    include_metadata: bool = True,
    author: Optional[str] = None,
) -> DiffSerializer:
    """Factory function to create a DiffSerializer."""
    return DiffSerializer(include_metadata=include_metadata, author=author)


def create_diff_applier(validate: bool = True) -> DiffApplier:
    """Factory function to create a DiffApplier."""
    return DiffApplier(validate=validate)


def create_undo_stack(
    max_depth: int = 100,
    max_bytes: int = 10 * 1024 * 1024,
) -> UndoStack:
    """Factory function to create an UndoStack."""
    return UndoStack(max_depth=max_depth, max_bytes=max_bytes)


def compute_state_diff(
    old_state: Any,
    new_state: Any,
    description: Optional[str] = None,
) -> SerializedDiff:
    """Convenience function to compute diff between two states."""
    serializer = DiffSerializer()
    return serializer.compute(old_state, new_state, description)


def apply_state_diff(
    base_state: Any,
    diff: SerializedDiff,
    target_type: Optional[Type[T]] = None,
) -> Union[Dict[str, Any], T]:
    """Convenience function to apply diff to state."""
    applier = DiffApplier()
    return applier.apply(base_state, diff, target_type)
