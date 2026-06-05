"""T-CC-4.4: CRDT/OT merge for scene edits.

Implements Conflict-free Replicated Data Types (CRDTs) for collaborative
scene editing with automatic conflict resolution:

- VectorClock: Causality tracking across multiple editors
- LWWRegister: Last-Writer-Wins for scalar values (position, rotation, etc.)
- GCounter: Grow-only counter (add operations only)
- PNCounter: Positive-Negative counter (increment/decrement)
- ORSet: Observed-Remove Set (add/remove with conflict resolution)
- LWWMap: Last-Writer-Wins Map (component properties)
- CRDTDocument: Scene/entity document wrapping multiple CRDTs
- OperationLog: Sync operations with server
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)


# =============================================================================
# Exceptions
# =============================================================================


class CRDTError(Exception):
    """Base exception for CRDT operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CausalityViolation(CRDTError):
    """Raised when an operation violates causality ordering."""

    def __init__(
        self,
        message: str,
        expected_clock: Optional["VectorClock"] = None,
        actual_clock: Optional["VectorClock"] = None,
    ):
        super().__init__(
            message,
            {
                "expected": expected_clock.to_dict() if expected_clock else None,
                "actual": actual_clock.to_dict() if actual_clock else None,
            },
        )
        self.expected_clock = expected_clock
        self.actual_clock = actual_clock


class MergeConflict(CRDTError):
    """Raised when merge cannot be resolved automatically."""

    def __init__(
        self,
        message: str,
        local_value: Any = None,
        remote_value: Any = None,
    ):
        super().__init__(
            message,
            {
                "local": str(local_value),
                "remote": str(remote_value),
            },
        )
        self.local_value = local_value
        self.remote_value = remote_value


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


# =============================================================================
# Vector Clock
# =============================================================================


class VectorClock:
    """Vector clock for causality tracking across multiple editors.

    Each node/editor has its own counter. The vector clock captures
    happens-before relationships between operations.

    Thread-safe implementation.
    """

    __slots__ = ("_clocks", "_lock")

    def __init__(self, clocks: Optional[Dict[str, int]] = None):
        """Initialize vector clock.

        Args:
            clocks: Initial clock values {node_id: counter}
        """
        self._clocks: Dict[str, int] = dict(clocks) if clocks else {}
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"VectorClock({self._clocks})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        with self._lock:
            return self._clocks == other._clocks

    def __hash__(self) -> int:
        with self._lock:
            return hash(frozenset(self._clocks.items()))

    def get(self, node_id: str) -> int:
        """Get counter value for a node."""
        with self._lock:
            return self._clocks.get(node_id, 0)

    def set(self, node_id: str, value: int) -> None:
        """Set counter value for a node."""
        with self._lock:
            self._clocks[node_id] = value

    def increment(self, node_id: str) -> int:
        """Increment counter for a node. Returns new value."""
        with self._lock:
            current = self._clocks.get(node_id, 0)
            self._clocks[node_id] = current + 1
            return self._clocks[node_id]

    def tick(self, node_id: str) -> "VectorClock":
        """Create new clock with incremented counter for node."""
        with self._lock:
            new_clock = VectorClock(self._clocks.copy())
            new_clock.increment(node_id)
            return new_clock

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Merge with another clock (element-wise max)."""
        with self._lock:
            all_nodes = set(self._clocks.keys()) | set(other._clocks.keys())
            merged = {}
            for node in all_nodes:
                merged[node] = max(self.get(node), other.get(node))
            return VectorClock(merged)

    def merge_inplace(self, other: "VectorClock") -> None:
        """Merge with another clock in place."""
        with self._lock:
            for node_id, counter in other._clocks.items():
                self._clocks[node_id] = max(self._clocks.get(node_id, 0), counter)

    def happens_before(self, other: "VectorClock") -> bool:
        """Check if this clock happens-before other (strict causal ordering)."""
        with self._lock:
            # self < other iff all(self[i] <= other[i]) and exists i: self[i] < other[i]
            all_nodes = set(self._clocks.keys()) | set(other._clocks.keys())
            at_least_one_less = False
            for node in all_nodes:
                self_val = self.get(node)
                other_val = other.get(node)
                if self_val > other_val:
                    return False
                if self_val < other_val:
                    at_least_one_less = True
            return at_least_one_less

    def concurrent_with(self, other: "VectorClock") -> bool:
        """Check if clocks are concurrent (neither happens-before the other)."""
        return not self.happens_before(other) and not other.happens_before(self)

    def dominates(self, other: "VectorClock") -> bool:
        """Check if this clock dominates other (>=)."""
        with self._lock:
            all_nodes = set(self._clocks.keys()) | set(other._clocks.keys())
            for node in all_nodes:
                if self.get(node) < other.get(node):
                    return False
            return True

    def is_empty(self) -> bool:
        """Check if clock is empty (all zeros or no entries)."""
        with self._lock:
            return all(v == 0 for v in self._clocks.values())

    def nodes(self) -> FrozenSet[str]:
        """Get all node IDs in this clock."""
        with self._lock:
            return frozenset(self._clocks.keys())

    def copy(self) -> "VectorClock":
        """Create a copy of this clock."""
        with self._lock:
            return VectorClock(self._clocks.copy())

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        with self._lock:
            return self._clocks.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "VectorClock":
        """Create from dictionary."""
        return cls(data)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "VectorClock":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# CRDT Base Protocol
# =============================================================================


@runtime_checkable
class CRDTType(Protocol):
    """Protocol for all CRDT types."""

    def merge(self, other: "CRDTType") -> "CRDTType":
        """Merge with another CRDT of the same type."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CRDTType":
        """Deserialize from dictionary."""
        ...


# =============================================================================
# LWW Register (Last-Writer-Wins)
# =============================================================================


@dataclass
class LWWRegister(Generic[T]):
    """Last-Writer-Wins Register for scalar values.

    Uses timestamps to resolve concurrent writes. The write with the
    highest timestamp wins. Ties are broken by node_id.

    Use cases:
    - Entity position, rotation, scale
    - Component scalar properties
    - Any single-valued property
    """

    value: T
    timestamp: float = field(default_factory=time.time)
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    def get(self) -> T:
        """Get current value."""
        with self._lock:
            return self.value

    def set(self, value: T, timestamp: Optional[float] = None, node_id: Optional[str] = None) -> bool:
        """Set value if timestamp is newer. Returns True if value was updated."""
        with self._lock:
            ts = timestamp if timestamp is not None else time.time()
            nid = node_id if node_id is not None else self.node_id

            # Compare timestamps, break ties with node_id
            if ts > self.timestamp or (ts == self.timestamp and nid > self.node_id):
                self.value = value
                self.timestamp = ts
                self.node_id = nid
                return True
            return False

    def merge(self, other: "LWWRegister[T]") -> "LWWRegister[T]":
        """Merge with another register. Returns new register with winning value."""
        with self._lock:
            if other.timestamp > self.timestamp:
                return LWWRegister(other.value, other.timestamp, other.node_id)
            elif other.timestamp == self.timestamp:
                # Tie-breaker: higher node_id wins
                if other.node_id > self.node_id:
                    return LWWRegister(other.value, other.timestamp, other.node_id)
            return LWWRegister(self.value, self.timestamp, self.node_id)

    def merge_inplace(self, other: "LWWRegister[T]") -> bool:
        """Merge in place. Returns True if value changed."""
        with self._lock:
            return self.set(other.value, other.timestamp, other.node_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        with self._lock:
            return {
                "value": self.value,
                "timestamp": self.timestamp,
                "node_id": self.node_id,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LWWRegister":
        """Deserialize from dictionary."""
        reg = cls(
            value=data["value"],
            timestamp=data.get("timestamp", 0.0),
            node_id=data.get("node_id", ""),
        )
        return reg

    def copy(self) -> "LWWRegister[T]":
        """Create a copy."""
        with self._lock:
            reg = LWWRegister(
                value=copy.deepcopy(self.value),
                timestamp=self.timestamp,
                node_id=self.node_id,
            )
            return reg


# =============================================================================
# G-Counter (Grow-only Counter)
# =============================================================================


@dataclass
class GCounter:
    """Grow-only Counter CRDT.

    Each node has its own counter that can only be incremented.
    The total value is the sum of all counters.

    Use cases:
    - View counts
    - Total operations performed
    - Any monotonically increasing count
    """

    _counters: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    def value(self) -> int:
        """Get total count (sum of all node counters)."""
        with self._lock:
            return sum(self._counters.values())

    def increment(self, node_id: str, amount: int = 1) -> int:
        """Increment counter for a node. Returns new node-local value."""
        if amount < 0:
            raise CRDTError("GCounter can only be incremented (use PNCounter for decrements)")
        with self._lock:
            current = self._counters.get(node_id, 0)
            self._counters[node_id] = current + amount
            return self._counters[node_id]

    def get_node_count(self, node_id: str) -> int:
        """Get counter value for a specific node."""
        with self._lock:
            return self._counters.get(node_id, 0)

    def merge(self, other: "GCounter") -> "GCounter":
        """Merge with another counter (element-wise max)."""
        with self._lock:
            all_nodes = set(self._counters.keys()) | set(other._counters.keys())
            merged = GCounter()
            for node in all_nodes:
                merged._counters[node] = max(
                    self._counters.get(node, 0),
                    other._counters.get(node, 0),
                )
            return merged

    def merge_inplace(self, other: "GCounter") -> None:
        """Merge in place."""
        with self._lock:
            for node_id, count in other._counters.items():
                self._counters[node_id] = max(self._counters.get(node_id, 0), count)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        with self._lock:
            return {"counters": self._counters.copy()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GCounter":
        """Deserialize from dictionary."""
        counter = cls()
        counter._counters = dict(data.get("counters", {}))
        return counter

    def copy(self) -> "GCounter":
        """Create a copy."""
        with self._lock:
            counter = GCounter()
            counter._counters = self._counters.copy()
            return counter


# =============================================================================
# PN-Counter (Positive-Negative Counter)
# =============================================================================


@dataclass
class PNCounter:
    """Positive-Negative Counter CRDT.

    Combines two G-Counters: one for increments, one for decrements.
    Value is increments - decrements.

    Use cases:
    - Like/dislike counts
    - Inventory quantities
    - Any value that can increase or decrease
    """

    _positive: GCounter = field(default_factory=GCounter)
    _negative: GCounter = field(default_factory=GCounter)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    def value(self) -> int:
        """Get current value (positive - negative)."""
        with self._lock:
            return self._positive.value() - self._negative.value()

    def increment(self, node_id: str, amount: int = 1) -> int:
        """Increment counter. Returns new value."""
        if amount < 0:
            return self.decrement(node_id, -amount)
        with self._lock:
            self._positive.increment(node_id, amount)
            return self.value()

    def decrement(self, node_id: str, amount: int = 1) -> int:
        """Decrement counter. Returns new value."""
        if amount < 0:
            return self.increment(node_id, -amount)
        with self._lock:
            self._negative.increment(node_id, amount)
            return self.value()

    def merge(self, other: "PNCounter") -> "PNCounter":
        """Merge with another counter."""
        with self._lock:
            result = PNCounter()
            result._positive = self._positive.merge(other._positive)
            result._negative = self._negative.merge(other._negative)
            return result

    def merge_inplace(self, other: "PNCounter") -> None:
        """Merge in place."""
        with self._lock:
            self._positive.merge_inplace(other._positive)
            self._negative.merge_inplace(other._negative)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        with self._lock:
            return {
                "positive": self._positive.to_dict(),
                "negative": self._negative.to_dict(),
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PNCounter":
        """Deserialize from dictionary."""
        counter = cls()
        counter._positive = GCounter.from_dict(data.get("positive", {}))
        counter._negative = GCounter.from_dict(data.get("negative", {}))
        return counter

    def copy(self) -> "PNCounter":
        """Create a copy."""
        with self._lock:
            counter = PNCounter()
            counter._positive = self._positive.copy()
            counter._negative = self._negative.copy()
            return counter


# =============================================================================
# OR-Set (Observed-Remove Set)
# =============================================================================


@dataclass
class ORSetEntry(Generic[T]):
    """Entry in an OR-Set with unique tag."""

    value: T
    tag: str  # Unique identifier for this add operation

    def __hash__(self) -> int:
        return hash(self.tag)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ORSetEntry):
            return NotImplemented
        return self.tag == other.tag


class ORSet(Generic[T]):
    """Observed-Remove Set CRDT.

    Supports add and remove operations with automatic conflict resolution.
    Each add generates a unique tag; remove only removes observed tags.

    Add-wins semantics: concurrent add and remove of the same element
    results in the element being present.

    Use cases:
    - Entity component lists
    - Selection sets
    - Tag collections
    """

    def __init__(self) -> None:
        self._entries: Dict[str, Set[str]] = {}  # value_key -> {tags}
        self._tombstones: Dict[str, Set[str]] = {}  # value_key -> {removed_tags}
        self._value_map: Dict[str, T] = {}  # value_key -> actual value
        self._lock = threading.RLock()

    def _value_key(self, value: T) -> str:
        """Generate a stable key for a value."""
        if isinstance(value, (str, int, float, bool)):
            return f"{type(value).__name__}:{value}"
        return f"{type(value).__name__}:{hash(str(value))}"

    def add(self, value: T, node_id: str) -> str:
        """Add an element. Returns the unique tag for this add."""
        with self._lock:
            tag = f"{node_id}:{uuid.uuid4().hex[:8]}:{time.time()}"
            vkey = self._value_key(value)

            if vkey not in self._entries:
                self._entries[vkey] = set()
            self._entries[vkey].add(tag)
            self._value_map[vkey] = value

            return tag

    def remove(self, value: T) -> Set[str]:
        """Remove an element (all observed tags). Returns removed tags."""
        with self._lock:
            vkey = self._value_key(value)

            if vkey not in self._entries:
                return set()

            tags = self._entries[vkey].copy()

            # Move tags to tombstones
            if vkey not in self._tombstones:
                self._tombstones[vkey] = set()
            self._tombstones[vkey].update(tags)

            # Clear entries
            self._entries[vkey].clear()

            return tags

    def contains(self, value: T) -> bool:
        """Check if element is in the set."""
        with self._lock:
            vkey = self._value_key(value)
            return vkey in self._entries and len(self._entries[vkey]) > 0

    def __contains__(self, value: T) -> bool:
        return self.contains(value)

    def elements(self) -> Set[T]:
        """Get all elements in the set."""
        with self._lock:
            result = set()
            for vkey, tags in self._entries.items():
                if tags and vkey in self._value_map:
                    result.add(self._value_map[vkey])
            return result

    def __iter__(self) -> Iterator[T]:
        return iter(self.elements())

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for tags in self._entries.values() if tags)

    def merge(self, other: "ORSet[T]") -> "ORSet[T]":
        """Merge with another OR-Set."""
        with self._lock:
            result = ORSet()

            # Merge value maps
            result._value_map = {**self._value_map, **other._value_map}

            # Merge tombstones
            all_vkeys = set(self._tombstones.keys()) | set(other._tombstones.keys())
            for vkey in all_vkeys:
                result._tombstones[vkey] = (
                    self._tombstones.get(vkey, set()) | other._tombstones.get(vkey, set())
                )

            # Merge entries, excluding tombstoned tags
            all_entry_keys = set(self._entries.keys()) | set(other._entries.keys())
            for vkey in all_entry_keys:
                self_tags = self._entries.get(vkey, set())
                other_tags = other._entries.get(vkey, set())
                merged_tags = self_tags | other_tags
                # Remove tombstoned tags
                tombstones = result._tombstones.get(vkey, set())
                live_tags = merged_tags - tombstones
                if live_tags:
                    result._entries[vkey] = live_tags

            return result

    def merge_inplace(self, other: "ORSet[T]") -> None:
        """Merge in place."""
        with self._lock:
            # Merge value maps
            self._value_map.update(other._value_map)

            # Merge tombstones
            for vkey, tags in other._tombstones.items():
                if vkey not in self._tombstones:
                    self._tombstones[vkey] = set()
                self._tombstones[vkey].update(tags)

            # Merge entries
            for vkey, tags in other._entries.items():
                if vkey not in self._entries:
                    self._entries[vkey] = set()
                self._entries[vkey].update(tags)

            # Remove tombstoned tags from entries
            for vkey in self._entries:
                tombstones = self._tombstones.get(vkey, set())
                self._entries[vkey] -= tombstones

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        with self._lock:
            return {
                "entries": {k: list(v) for k, v in self._entries.items()},
                "tombstones": {k: list(v) for k, v in self._tombstones.items()},
                "values": {k: v for k, v in self._value_map.items()},
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ORSet":
        """Deserialize from dictionary."""
        orset = cls()
        orset._entries = {k: set(v) for k, v in data.get("entries", {}).items()}
        orset._tombstones = {k: set(v) for k, v in data.get("tombstones", {}).items()}
        orset._value_map = dict(data.get("values", {}))
        return orset

    def copy(self) -> "ORSet[T]":
        """Create a copy."""
        with self._lock:
            orset = ORSet()
            orset._entries = {k: v.copy() for k, v in self._entries.items()}
            orset._tombstones = {k: v.copy() for k, v in self._tombstones.items()}
            orset._value_map = self._value_map.copy()
            return orset


# =============================================================================
# LWW-Map (Last-Writer-Wins Map)
# =============================================================================


class LWWMap(Generic[K, V]):
    """Last-Writer-Wins Map CRDT.

    Each key maps to an LWWRegister. Supports add, update, and remove.

    Use cases:
    - Component properties
    - Entity metadata
    - Configuration dictionaries
    """

    def __init__(self) -> None:
        self._entries: Dict[K, LWWRegister[Optional[V]]] = {}
        self._lock = threading.RLock()

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get value for key."""
        with self._lock:
            if key not in self._entries:
                return default
            value = self._entries[key].get()
            return value if value is not None else default

    def set(
        self,
        key: K,
        value: V,
        timestamp: Optional[float] = None,
        node_id: Optional[str] = None,
    ) -> bool:
        """Set value for key. Returns True if value was updated."""
        with self._lock:
            ts = timestamp if timestamp is not None else time.time()
            nid = node_id if node_id is not None else str(uuid.uuid4())[:8]

            if key not in self._entries:
                self._entries[key] = LWWRegister(value, ts, nid)
                return True
            return self._entries[key].set(value, ts, nid)

    def remove(
        self,
        key: K,
        timestamp: Optional[float] = None,
        node_id: Optional[str] = None,
    ) -> bool:
        """Remove key (set to None tombstone). Returns True if removed."""
        with self._lock:
            ts = timestamp if timestamp is not None else time.time()
            nid = node_id if node_id is not None else str(uuid.uuid4())[:8]

            if key not in self._entries:
                self._entries[key] = LWWRegister(None, ts, nid)
                return True
            return self._entries[key].set(None, ts, nid)

    def __getitem__(self, key: K) -> V:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value)

    def __delitem__(self, key: K) -> None:
        self.remove(key)

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._entries and self._entries[key].get() is not None

    def keys(self) -> Set[K]:
        """Get all keys with non-None values."""
        with self._lock:
            return {k for k, v in self._entries.items() if v.get() is not None}

    def values(self) -> List[V]:
        """Get all non-None values."""
        with self._lock:
            return [v.get() for v in self._entries.values() if v.get() is not None]

    def items(self) -> List[Tuple[K, V]]:
        """Get all key-value pairs with non-None values."""
        with self._lock:
            return [(k, v.get()) for k, v in self._entries.items() if v.get() is not None]

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for v in self._entries.values() if v.get() is not None)

    def __iter__(self) -> Iterator[K]:
        return iter(self.keys())

    def merge(self, other: "LWWMap[K, V]") -> "LWWMap[K, V]":
        """Merge with another map."""
        with self._lock:
            result = LWWMap()

            all_keys = set(self._entries.keys()) | set(other._entries.keys())
            for key in all_keys:
                if key in self._entries and key in other._entries:
                    result._entries[key] = self._entries[key].merge(other._entries[key])
                elif key in self._entries:
                    result._entries[key] = self._entries[key].copy()
                else:
                    result._entries[key] = other._entries[key].copy()

            return result

    def merge_inplace(self, other: "LWWMap[K, V]") -> None:
        """Merge in place."""
        with self._lock:
            for key, reg in other._entries.items():
                if key not in self._entries:
                    self._entries[key] = reg.copy()
                else:
                    self._entries[key].merge_inplace(reg)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        with self._lock:
            return {
                "entries": {
                    str(k): v.to_dict() for k, v in self._entries.items()
                }
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LWWMap":
        """Deserialize from dictionary."""
        lwwmap = cls()
        for k, v in data.get("entries", {}).items():
            lwwmap._entries[k] = LWWRegister.from_dict(v)
        return lwwmap

    def copy(self) -> "LWWMap[K, V]":
        """Create a copy."""
        with self._lock:
            lwwmap = LWWMap()
            lwwmap._entries = {k: v.copy() for k, v in self._entries.items()}
            return lwwmap


# =============================================================================
# Operation Types and Log
# =============================================================================


class OperationType(Enum):
    """Types of CRDT operations."""

    # Register operations
    SET = auto()

    # Counter operations
    INCREMENT = auto()
    DECREMENT = auto()

    # Set operations
    ADD = auto()
    REMOVE = auto()

    # Map operations
    MAP_SET = auto()
    MAP_REMOVE = auto()

    # Document operations
    DOC_UPDATE = auto()
    DOC_CREATE_FIELD = auto()
    DOC_DELETE_FIELD = auto()


@dataclass
class CRDTOperation:
    """A CRDT operation for synchronization.

    Operations are idempotent and commutative for the same CRDT type.
    """

    id: str
    type: OperationType
    path: str  # JSON path to the CRDT field
    value: Any
    timestamp: float
    node_id: str
    clock: VectorClock
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"{self.node_id}:{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type.name,
            "path": self.path,
            "value": self.value,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "clock": self.clock.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CRDTOperation":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            type=OperationType[data["type"]],
            path=data["path"],
            value=data["value"],
            timestamp=data["timestamp"],
            node_id=data["node_id"],
            clock=VectorClock.from_dict(data["clock"]),
            metadata=data.get("metadata", {}),
        )

    def happens_before(self, other: "CRDTOperation") -> bool:
        """Check if this operation happens-before another."""
        return self.clock.happens_before(other.clock)

    def concurrent_with(self, other: "CRDTOperation") -> bool:
        """Check if operations are concurrent."""
        return self.clock.concurrent_with(other.clock)


class OperationLog:
    """Log of operations for server synchronization.

    Maintains a log of operations that can be:
    - Sent to server for persistence
    - Received from server and applied locally
    - Used for conflict resolution
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._operations: List[CRDTOperation] = []
        self._applied_ids: Set[str] = set()
        self._clock = VectorClock()
        self._lock = threading.RLock()

    def create_operation(
        self,
        op_type: OperationType,
        path: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CRDTOperation:
        """Create a new operation and add to log."""
        with self._lock:
            self._clock.increment(self.node_id)

            op = CRDTOperation(
                id="",  # Will be generated in __post_init__
                type=op_type,
                path=path,
                value=value,
                timestamp=time.time(),
                node_id=self.node_id,
                clock=self._clock.copy(),
                metadata=metadata or {},
            )

            self._operations.append(op)
            self._applied_ids.add(op.id)

            return op

    def add_operation(self, op: CRDTOperation) -> bool:
        """Add an external operation (from sync). Returns True if new."""
        with self._lock:
            if op.id in self._applied_ids:
                return False

            self._operations.append(op)
            self._applied_ids.add(op.id)
            self._clock.merge_inplace(op.clock)

            return True

    def get_operations(
        self,
        since_clock: Optional[VectorClock] = None,
        limit: Optional[int] = None,
    ) -> List[CRDTOperation]:
        """Get operations, optionally filtered by clock and limit."""
        with self._lock:
            if since_clock is None:
                ops = self._operations
            else:
                ops = [op for op in self._operations if not since_clock.dominates(op.clock)]

            if limit:
                ops = ops[:limit]

            return list(ops)

    def get_pending(self) -> List[CRDTOperation]:
        """Get operations that haven't been acknowledged by server."""
        # In a full implementation, this would track server acks
        return self.get_operations()

    def clear_before(self, clock: VectorClock) -> int:
        """Clear operations that happened-before the given clock. Returns count cleared."""
        with self._lock:
            old_len = len(self._operations)
            self._operations = [
                op for op in self._operations
                if not op.clock.happens_before(clock) or op.clock == clock
            ]
            return old_len - len(self._operations)

    @property
    def current_clock(self) -> VectorClock:
        """Get current vector clock."""
        with self._lock:
            return self._clock.copy()

    def __len__(self) -> int:
        with self._lock:
            return len(self._operations)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        with self._lock:
            return {
                "node_id": self.node_id,
                "operations": [op.to_dict() for op in self._operations],
                "clock": self._clock.to_dict(),
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperationLog":
        """Deserialize from dictionary."""
        log = cls(data["node_id"])
        log._clock = VectorClock.from_dict(data.get("clock", {}))
        for op_data in data.get("operations", []):
            op = CRDTOperation.from_dict(op_data)
            log._operations.append(op)
            log._applied_ids.add(op.id)
        return log


# =============================================================================
# CRDT Document
# =============================================================================


@dataclass
class CRDTField:
    """Metadata about a CRDT field in a document."""

    name: str
    crdt_type: str  # "lww_register", "gcounter", "pncounter", "orset", "lwwmap"
    path: str


class CRDTDocument:
    """A document containing multiple CRDT fields.

    Represents a scene or entity with collaborative editing support.
    Each field is a CRDT that can be independently merged.
    """

    def __init__(
        self,
        doc_id: str,
        node_id: Optional[str] = None,
    ):
        self.doc_id = doc_id
        self.node_id = node_id or str(uuid.uuid4())[:8]

        # CRDT fields
        self._registers: Dict[str, LWWRegister] = {}
        self._counters: Dict[str, Union[GCounter, PNCounter]] = {}
        self._sets: Dict[str, ORSet] = {}
        self._maps: Dict[str, LWWMap] = {}

        # Metadata
        self._clock = VectorClock()
        self._operation_log = OperationLog(self.node_id)
        self._field_metadata: Dict[str, CRDTField] = {}

        self._lock = threading.RLock()
        self._created_at = time.time()
        self._updated_at = time.time()

    # -------------------------------------------------------------------------
    # Register operations
    # -------------------------------------------------------------------------

    def set_register(self, name: str, value: Any) -> CRDTOperation:
        """Set a scalar value using LWW semantics."""
        with self._lock:
            ts = time.time()
            self._updated_at = ts

            if name not in self._registers:
                self._registers[name] = LWWRegister(value, ts, self.node_id)
                self._field_metadata[name] = CRDTField(name, "lww_register", f"registers.{name}")
            else:
                self._registers[name].set(value, ts, self.node_id)

            self._clock.increment(self.node_id)

            return self._operation_log.create_operation(
                OperationType.SET,
                f"registers.{name}",
                {"value": value, "timestamp": ts, "node_id": self.node_id},
            )

    def get_register(self, name: str, default: Any = None) -> Any:
        """Get a scalar value."""
        with self._lock:
            if name not in self._registers:
                return default
            return self._registers[name].get()

    # -------------------------------------------------------------------------
    # Counter operations
    # -------------------------------------------------------------------------

    def increment_counter(self, name: str, amount: int = 1, pn: bool = False) -> CRDTOperation:
        """Increment a counter."""
        with self._lock:
            self._updated_at = time.time()

            if name not in self._counters:
                self._counters[name] = PNCounter() if pn else GCounter()
                ctype = "pncounter" if pn else "gcounter"
                self._field_metadata[name] = CRDTField(name, ctype, f"counters.{name}")

            counter = self._counters[name]
            if isinstance(counter, PNCounter):
                counter.increment(self.node_id, amount)
            else:
                counter.increment(self.node_id, abs(amount))

            self._clock.increment(self.node_id)

            return self._operation_log.create_operation(
                OperationType.INCREMENT,
                f"counters.{name}",
                {"amount": amount, "node_id": self.node_id},
            )

    def decrement_counter(self, name: str, amount: int = 1) -> CRDTOperation:
        """Decrement a PN-Counter."""
        with self._lock:
            self._updated_at = time.time()

            if name not in self._counters:
                self._counters[name] = PNCounter()
                self._field_metadata[name] = CRDTField(name, "pncounter", f"counters.{name}")

            counter = self._counters[name]
            if not isinstance(counter, PNCounter):
                raise CRDTError(f"Counter '{name}' is a GCounter and cannot be decremented")

            counter.decrement(self.node_id, amount)
            self._clock.increment(self.node_id)

            return self._operation_log.create_operation(
                OperationType.DECREMENT,
                f"counters.{name}",
                {"amount": amount, "node_id": self.node_id},
            )

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        with self._lock:
            if name not in self._counters:
                return 0
            return self._counters[name].value()

    # -------------------------------------------------------------------------
    # Set operations
    # -------------------------------------------------------------------------

    def add_to_set(self, name: str, value: Any) -> CRDTOperation:
        """Add value to an OR-Set."""
        with self._lock:
            self._updated_at = time.time()

            if name not in self._sets:
                self._sets[name] = ORSet()
                self._field_metadata[name] = CRDTField(name, "orset", f"sets.{name}")

            tag = self._sets[name].add(value, self.node_id)
            self._clock.increment(self.node_id)

            return self._operation_log.create_operation(
                OperationType.ADD,
                f"sets.{name}",
                {"value": value, "tag": tag, "node_id": self.node_id},
            )

    def remove_from_set(self, name: str, value: Any) -> CRDTOperation:
        """Remove value from an OR-Set."""
        with self._lock:
            self._updated_at = time.time()

            if name not in self._sets:
                self._sets[name] = ORSet()
                self._field_metadata[name] = CRDTField(name, "orset", f"sets.{name}")

            tags = self._sets[name].remove(value)
            self._clock.increment(self.node_id)

            return self._operation_log.create_operation(
                OperationType.REMOVE,
                f"sets.{name}",
                {"value": value, "tags": list(tags), "node_id": self.node_id},
            )

    def get_set(self, name: str) -> Set[Any]:
        """Get set elements."""
        with self._lock:
            if name not in self._sets:
                return set()
            return self._sets[name].elements()

    def set_contains(self, name: str, value: Any) -> bool:
        """Check if set contains value."""
        with self._lock:
            if name not in self._sets:
                return False
            return value in self._sets[name]

    # -------------------------------------------------------------------------
    # Map operations
    # -------------------------------------------------------------------------

    def set_map_value(self, name: str, key: Any, value: Any) -> CRDTOperation:
        """Set a value in an LWW-Map."""
        with self._lock:
            ts = time.time()
            self._updated_at = ts

            if name not in self._maps:
                self._maps[name] = LWWMap()
                self._field_metadata[name] = CRDTField(name, "lwwmap", f"maps.{name}")

            self._maps[name].set(key, value, ts, self.node_id)
            self._clock.increment(self.node_id)

            return self._operation_log.create_operation(
                OperationType.MAP_SET,
                f"maps.{name}.{key}",
                {"key": key, "value": value, "timestamp": ts, "node_id": self.node_id},
            )

    def remove_map_value(self, name: str, key: Any) -> CRDTOperation:
        """Remove a value from an LWW-Map."""
        with self._lock:
            ts = time.time()
            self._updated_at = ts

            if name not in self._maps:
                self._maps[name] = LWWMap()
                self._field_metadata[name] = CRDTField(name, "lwwmap", f"maps.{name}")

            self._maps[name].remove(key, ts, self.node_id)
            self._clock.increment(self.node_id)

            return self._operation_log.create_operation(
                OperationType.MAP_REMOVE,
                f"maps.{name}.{key}",
                {"key": key, "timestamp": ts, "node_id": self.node_id},
            )

    def get_map_value(self, name: str, key: Any, default: Any = None) -> Any:
        """Get a value from an LWW-Map."""
        with self._lock:
            if name not in self._maps:
                return default
            return self._maps[name].get(key, default)

    def get_map(self, name: str) -> Dict[Any, Any]:
        """Get all key-value pairs from an LWW-Map."""
        with self._lock:
            if name not in self._maps:
                return {}
            return dict(self._maps[name].items())

    # -------------------------------------------------------------------------
    # Merge operations
    # -------------------------------------------------------------------------

    def merge(self, other: "CRDTDocument") -> "CRDTDocument":
        """Merge with another document. Returns new merged document."""
        with self._lock:
            result = CRDTDocument(self.doc_id, self.node_id)

            # Merge registers
            all_reg_keys = set(self._registers.keys()) | set(other._registers.keys())
            for key in all_reg_keys:
                if key in self._registers and key in other._registers:
                    result._registers[key] = self._registers[key].merge(other._registers[key])
                elif key in self._registers:
                    result._registers[key] = self._registers[key].copy()
                else:
                    result._registers[key] = other._registers[key].copy()

            # Merge counters
            all_counter_keys = set(self._counters.keys()) | set(other._counters.keys())
            for key in all_counter_keys:
                if key in self._counters and key in other._counters:
                    result._counters[key] = self._counters[key].merge(other._counters[key])
                elif key in self._counters:
                    result._counters[key] = self._counters[key].copy()
                else:
                    result._counters[key] = other._counters[key].copy()

            # Merge sets
            all_set_keys = set(self._sets.keys()) | set(other._sets.keys())
            for key in all_set_keys:
                if key in self._sets and key in other._sets:
                    result._sets[key] = self._sets[key].merge(other._sets[key])
                elif key in self._sets:
                    result._sets[key] = self._sets[key].copy()
                else:
                    result._sets[key] = other._sets[key].copy()

            # Merge maps
            all_map_keys = set(self._maps.keys()) | set(other._maps.keys())
            for key in all_map_keys:
                if key in self._maps and key in other._maps:
                    result._maps[key] = self._maps[key].merge(other._maps[key])
                elif key in self._maps:
                    result._maps[key] = self._maps[key].copy()
                else:
                    result._maps[key] = other._maps[key].copy()

            # Merge clocks
            result._clock = self._clock.merge(other._clock)

            # Merge field metadata
            result._field_metadata = {**self._field_metadata, **other._field_metadata}

            # Merge operation logs
            for op in other._operation_log.get_operations():
                result._operation_log.add_operation(op)
            for op in self._operation_log.get_operations():
                result._operation_log.add_operation(op)

            result._updated_at = max(self._updated_at, other._updated_at)
            result._created_at = min(self._created_at, other._created_at)

            return result

    def merge_inplace(self, other: "CRDTDocument") -> None:
        """Merge another document into this one."""
        with self._lock:
            # Merge registers
            for key, reg in other._registers.items():
                if key not in self._registers:
                    self._registers[key] = reg.copy()
                else:
                    self._registers[key].merge_inplace(reg)

            # Merge counters
            for key, counter in other._counters.items():
                if key not in self._counters:
                    self._counters[key] = counter.copy()
                else:
                    self._counters[key].merge_inplace(counter)

            # Merge sets
            for key, orset in other._sets.items():
                if key not in self._sets:
                    self._sets[key] = orset.copy()
                else:
                    self._sets[key].merge_inplace(orset)

            # Merge maps
            for key, lwwmap in other._maps.items():
                if key not in self._maps:
                    self._maps[key] = lwwmap.copy()
                else:
                    self._maps[key].merge_inplace(lwwmap)

            # Merge clock
            self._clock.merge_inplace(other._clock)

            # Merge metadata
            self._field_metadata.update(other._field_metadata)

            # Merge operations
            for op in other._operation_log.get_operations():
                self._operation_log.add_operation(op)

            self._updated_at = max(self._updated_at, other._updated_at)

    def apply_operation(self, op: CRDTOperation) -> bool:
        """Apply a single operation. Returns True if applied (not duplicate)."""
        with self._lock:
            if not self._operation_log.add_operation(op):
                return False  # Already applied

            parts = op.path.split(".")
            if len(parts) < 2:
                return False

            category = parts[0]
            field_name = parts[1]

            if category == "registers" and op.type == OperationType.SET:
                if field_name not in self._registers:
                    self._registers[field_name] = LWWRegister(
                        op.value["value"],
                        op.value["timestamp"],
                        op.value["node_id"],
                    )
                else:
                    self._registers[field_name].set(
                        op.value["value"],
                        op.value["timestamp"],
                        op.value["node_id"],
                    )

            elif category == "counters":
                if field_name not in self._counters:
                    self._counters[field_name] = (
                        PNCounter() if op.type == OperationType.DECREMENT else GCounter()
                    )
                counter = self._counters[field_name]
                if op.type == OperationType.INCREMENT:
                    if isinstance(counter, GCounter):
                        counter.increment(op.value["node_id"], op.value["amount"])
                    else:
                        counter.increment(op.value["node_id"], op.value["amount"])
                elif op.type == OperationType.DECREMENT and isinstance(counter, PNCounter):
                    counter.decrement(op.value["node_id"], op.value["amount"])

            elif category == "sets":
                if field_name not in self._sets:
                    self._sets[field_name] = ORSet()
                orset = self._sets[field_name]
                if op.type == OperationType.ADD:
                    orset.add(op.value["value"], op.value["node_id"])
                elif op.type == OperationType.REMOVE:
                    orset.remove(op.value["value"])

            elif category == "maps" and len(parts) >= 3:
                key = parts[2]
                if field_name not in self._maps:
                    self._maps[field_name] = LWWMap()
                lwwmap = self._maps[field_name]
                if op.type == OperationType.MAP_SET:
                    lwwmap.set(key, op.value["value"], op.value["timestamp"], op.value["node_id"])
                elif op.type == OperationType.MAP_REMOVE:
                    lwwmap.remove(key, op.value["timestamp"], op.value["node_id"])

            self._clock.merge_inplace(op.clock)
            self._updated_at = max(self._updated_at, op.timestamp)

            return True

    def apply_operations(self, ops: List[CRDTOperation]) -> int:
        """Apply multiple operations. Returns count of newly applied."""
        count = 0
        for op in ops:
            if self.apply_operation(op):
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        with self._lock:
            return {
                "doc_id": self.doc_id,
                "node_id": self.node_id,
                "registers": {k: v.to_dict() for k, v in self._registers.items()},
                "counters": {k: v.to_dict() for k, v in self._counters.items()},
                "sets": {k: v.to_dict() for k, v in self._sets.items()},
                "maps": {k: v.to_dict() for k, v in self._maps.items()},
                "clock": self._clock.to_dict(),
                "operation_log": self._operation_log.to_dict(),
                "field_metadata": {
                    k: {"name": v.name, "crdt_type": v.crdt_type, "path": v.path}
                    for k, v in self._field_metadata.items()
                },
                "created_at": self._created_at,
                "updated_at": self._updated_at,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CRDTDocument":
        """Deserialize from dictionary."""
        doc = cls(data["doc_id"], data.get("node_id"))

        # Deserialize registers
        for k, v in data.get("registers", {}).items():
            doc._registers[k] = LWWRegister.from_dict(v)

        # Deserialize counters
        for k, v in data.get("counters", {}).items():
            if "positive" in v:
                doc._counters[k] = PNCounter.from_dict(v)
            else:
                doc._counters[k] = GCounter.from_dict(v)

        # Deserialize sets
        for k, v in data.get("sets", {}).items():
            doc._sets[k] = ORSet.from_dict(v)

        # Deserialize maps
        for k, v in data.get("maps", {}).items():
            doc._maps[k] = LWWMap.from_dict(v)

        # Deserialize clock
        doc._clock = VectorClock.from_dict(data.get("clock", {}))

        # Deserialize operation log
        if "operation_log" in data:
            doc._operation_log = OperationLog.from_dict(data["operation_log"])

        # Deserialize field metadata
        for k, v in data.get("field_metadata", {}).items():
            doc._field_metadata[k] = CRDTField(v["name"], v["crdt_type"], v["path"])

        doc._created_at = data.get("created_at", time.time())
        doc._updated_at = data.get("updated_at", time.time())

        return doc

    def to_json(self, pretty: bool = False) -> str:
        """Serialize to JSON string."""
        data = self.to_dict()
        if pretty:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "CRDTDocument":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    @property
    def clock(self) -> VectorClock:
        """Get current vector clock."""
        with self._lock:
            return self._clock.copy()

    @property
    def operation_log(self) -> OperationLog:
        """Get operation log."""
        return self._operation_log

    def get_pending_operations(self) -> List[CRDTOperation]:
        """Get operations pending sync to server."""
        return self._operation_log.get_pending()

    def get_operations_since(self, clock: VectorClock) -> List[CRDTOperation]:
        """Get operations since a given clock state."""
        return self._operation_log.get_operations(since_clock=clock)

    def field_names(self) -> Set[str]:
        """Get all field names."""
        with self._lock:
            return (
                set(self._registers.keys())
                | set(self._counters.keys())
                | set(self._sets.keys())
                | set(self._maps.keys())
            )

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"CRDTDocument(id={self.doc_id}, "
                f"registers={len(self._registers)}, "
                f"counters={len(self._counters)}, "
                f"sets={len(self._sets)}, "
                f"maps={len(self._maps)})"
            )
