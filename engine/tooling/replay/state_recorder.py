"""
State Recorder - Record game state snapshots at intervals.

Captures periodic full state snapshots and delta-compressed intermediate
states for efficient seeking and determinism verification.
"""

from __future__ import annotations

import copy
import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Iterator
import json


class CompressionMethod(Enum):
    """Compression methods for state data."""
    NONE = auto()
    ZLIB = auto()
    ZLIB_FAST = auto()  # Level 1
    ZLIB_BEST = auto()  # Level 9
    LZ4 = auto()  # Requires external library
    DELTA = auto()  # Delta encoding only


@dataclass(slots=True)
class StateSnapshot:
    """A full game state snapshot."""
    frame: int
    timestamp: float
    state_data: dict[str, Any]
    checksum: str
    size_bytes: int
    is_keyframe: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def serialize(self, compression: CompressionMethod = CompressionMethod.ZLIB) -> bytes:
        """Serialize snapshot to bytes with compression.

        Args:
            compression: Compression method to use

        Returns:
            Serialized bytes
        """
        # Serialize state data to JSON
        state_json = json.dumps(self.state_data, separators=(',', ':')).encode('utf-8')
        metadata_json = json.dumps(self.metadata, separators=(',', ':')).encode('utf-8')

        # Compress if requested
        compressed_state = self._compress(state_json, compression)

        # Pack header: frame(4), timestamp(8), is_keyframe(1), compression(1)
        header = struct.pack(
            '<IdBB',
            self.frame,
            self.timestamp,
            1 if self.is_keyframe else 0,
            compression.value
        )

        # Pack checksum (32 bytes for SHA-256, or zeros if empty)
        if self.checksum and len(self.checksum) == 64:
            checksum_bytes = bytes.fromhex(self.checksum)
        else:
            checksum_bytes = b'\x00' * 32

        # Pack data lengths and data
        return (
            header +
            checksum_bytes +
            struct.pack('<I', len(compressed_state)) +
            compressed_state +
            struct.pack('<I', len(metadata_json)) +
            metadata_json
        )

    @classmethod
    def deserialize(cls, data: bytes, offset: int = 0) -> tuple['StateSnapshot', int]:
        """Deserialize snapshot from bytes.

        Args:
            data: Byte buffer
            offset: Starting offset

        Returns:
            Tuple of (StateSnapshot, bytes consumed)
        """
        start_offset = offset

        # Unpack header
        header_size = struct.calcsize('<IdBB')
        frame, timestamp, is_keyframe, compression_val = struct.unpack(
            '<IdBB', data[offset:offset + header_size]
        )
        offset += header_size

        # Read checksum
        checksum = data[offset:offset + 32].hex()
        offset += 32

        # Read compressed state
        state_len = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        compressed_state = data[offset:offset + state_len]
        offset += state_len

        # Read metadata
        metadata_len = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        metadata_json = data[offset:offset + metadata_len]
        offset += metadata_len

        # Decompress and parse
        compression = CompressionMethod(compression_val)
        state_json = cls._decompress(compressed_state, compression)
        state_data = json.loads(state_json.decode('utf-8'))
        metadata = json.loads(metadata_json.decode('utf-8'))

        return cls(
            frame=frame,
            timestamp=timestamp,
            state_data=state_data,
            checksum=checksum,
            size_bytes=offset - start_offset,
            is_keyframe=bool(is_keyframe),
            metadata=metadata
        ), offset

    @staticmethod
    def _compress(data: bytes, method: CompressionMethod) -> bytes:
        """Compress data using specified method."""
        if method == CompressionMethod.NONE:
            return data
        elif method == CompressionMethod.ZLIB:
            return zlib.compress(data, level=6)
        elif method == CompressionMethod.ZLIB_FAST:
            return zlib.compress(data, level=1)
        elif method == CompressionMethod.ZLIB_BEST:
            return zlib.compress(data, level=9)
        else:
            return zlib.compress(data, level=6)  # Default fallback

    @staticmethod
    def _decompress(data: bytes, method: CompressionMethod) -> bytes:
        """Decompress data using specified method."""
        if method == CompressionMethod.NONE:
            return data
        elif method in (CompressionMethod.ZLIB, CompressionMethod.ZLIB_FAST,
                       CompressionMethod.ZLIB_BEST):
            return zlib.decompress(data)
        else:
            return zlib.decompress(data)  # Default fallback

    def compute_checksum(self) -> str:
        """Compute checksum of state data.

        Returns:
            SHA-256 hash of state data
        """
        state_json = json.dumps(self.state_data, sort_keys=True).encode('utf-8')
        return hashlib.sha256(state_json).hexdigest()

    def verify_checksum(self) -> bool:
        """Verify state data matches checksum.

        Returns:
            True if checksum matches
        """
        return self.compute_checksum() == self.checksum


@dataclass(slots=True)
class StateDelta:
    """A delta-encoded state change between snapshots."""
    from_frame: int
    to_frame: int
    timestamp: float
    changes: list[tuple[str, Any, Any]]  # (path, old_value, new_value)
    size_bytes: int

    def apply(self, base_state: dict[str, Any]) -> dict[str, Any]:
        """Apply delta to base state.

        Args:
            base_state: State to apply delta to

        Returns:
            New state with delta applied
        """
        result = copy.deepcopy(base_state)

        for path, _, new_value in self.changes:
            self._set_path(result, path, new_value)

        return result

    def reverse(self, current_state: dict[str, Any]) -> dict[str, Any]:
        """Reverse delta to get previous state.

        Args:
            current_state: Current state

        Returns:
            Previous state with delta reversed
        """
        result = copy.deepcopy(current_state)

        for path, old_value, _ in self.changes:
            self._set_path(result, path, old_value)

        return result

    @staticmethod
    def _set_path(obj: dict, path: str, value: Any) -> None:
        """Set value at dotted path in nested dict."""
        parts = path.split('.')
        for part in parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = obj.setdefault(part, {})

        final = parts[-1]
        if final.isdigit():
            obj[int(final)] = value
        else:
            obj[final] = value

    def serialize(self) -> bytes:
        """Serialize delta to bytes."""
        # Serialize changes to JSON
        changes_json = json.dumps(self.changes, separators=(',', ':')).encode('utf-8')
        compressed = zlib.compress(changes_json)

        # Pack header
        header = struct.pack('<IId', self.from_frame, self.to_frame, self.timestamp)

        return header + struct.pack('<I', len(compressed)) + compressed

    @classmethod
    def deserialize(cls, data: bytes, offset: int = 0) -> tuple['StateDelta', int]:
        """Deserialize delta from bytes."""
        start_offset = offset

        # Unpack header
        header_size = struct.calcsize('<IId')
        from_frame, to_frame, timestamp = struct.unpack(
            '<IId', data[offset:offset + header_size]
        )
        offset += header_size

        # Read compressed changes
        changes_len = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        compressed = data[offset:offset + changes_len]
        offset += changes_len

        # Decompress and parse
        changes_json = zlib.decompress(compressed)
        changes = json.loads(changes_json.decode('utf-8'))

        return cls(
            from_frame=from_frame,
            to_frame=to_frame,
            timestamp=timestamp,
            changes=[tuple(c) for c in changes],
            size_bytes=offset - start_offset
        ), offset


@dataclass
class StateRecordingConfig:
    """Configuration for state recording."""
    # Snapshot intervals
    keyframe_interval: int = 60  # Frames between full snapshots
    delta_interval: int = 1  # Frames between delta snapshots

    # Compression
    compression: CompressionMethod = CompressionMethod.ZLIB

    # State filtering
    state_filter: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None

    # Paths to exclude from recording
    excluded_paths: set[str] = field(default_factory=set)

    # Paths to always include (overrides exclusions)
    included_paths: set[str] = field(default_factory=set)

    # Size limits
    max_state_size: int = 10 * 1024 * 1024  # 10 MB
    max_snapshots: int = 10000

    # Delta compression
    enable_delta_compression: bool = True
    delta_threshold: float = 0.1  # Min change ratio to store delta

    # Verification
    compute_checksums: bool = True


class StateRecorder:
    """Records game state snapshots for replay.

    Captures periodic full state snapshots (keyframes) and delta-compressed
    intermediate states for efficient seeking and determinism verification.
    """
    __slots__ = (
        '_config', '_snapshots', '_deltas', '_is_recording',
        '_current_frame', '_last_state', '_last_keyframe_state',
        '_last_keyframe_frame', '_stats'
    )

    def __init__(self, config: Optional[StateRecordingConfig] = None):
        """Initialize the state recorder.

        Args:
            config: Recording configuration
        """
        self._config = config or StateRecordingConfig()
        self._snapshots: list[StateSnapshot] = []
        self._deltas: list[StateDelta] = []
        self._is_recording = False
        self._current_frame = 0
        self._last_state: Optional[dict[str, Any]] = None
        self._last_keyframe_state: Optional[dict[str, Any]] = None
        self._last_keyframe_frame = 0
        self._stats = {
            'total_snapshots': 0,
            'total_deltas': 0,
            'total_bytes': 0,
            'compression_ratio': 1.0
        }

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    @property
    def snapshot_count(self) -> int:
        """Get number of snapshots."""
        return len(self._snapshots)

    @property
    def delta_count(self) -> int:
        """Get number of deltas."""
        return len(self._deltas)

    @property
    def current_frame(self) -> int:
        """Get current frame number."""
        return self._current_frame

    @property
    def stats(self) -> dict[str, Any]:
        """Get recording statistics."""
        return self._stats.copy()

    def start(self) -> None:
        """Start recording state."""
        if self._is_recording:
            return

        self._is_recording = True
        self._current_frame = 0
        self._last_state = None
        self._last_keyframe_state = None
        self._last_keyframe_frame = 0
        self._snapshots.clear()
        self._deltas.clear()
        self._stats = {
            'total_snapshots': 0,
            'total_deltas': 0,
            'total_bytes': 0,
            'compression_ratio': 1.0
        }

    def stop(self) -> tuple[list[StateSnapshot], list[StateDelta]]:
        """Stop recording and return all recorded data.

        Returns:
            Tuple of (snapshots, deltas)
        """
        self._is_recording = False
        return list(self._snapshots), list(self._deltas)

    def record_state(
        self,
        state: dict[str, Any],
        timestamp: float,
        force_keyframe: bool = False,
        metadata: Optional[dict[str, Any]] = None
    ) -> Optional[StateSnapshot | StateDelta]:
        """Record current game state.

        Args:
            state: Current game state dictionary
            timestamp: Current timestamp
            force_keyframe: Force a keyframe snapshot
            metadata: Optional metadata for snapshot

        Returns:
            The recorded snapshot or delta, or None if not recording
        """
        if not self._is_recording:
            return None

        # Apply state filter if configured
        filtered_state = self._filter_state(state)

        # Check if we should create a keyframe
        should_keyframe = (
            force_keyframe or
            self._last_keyframe_state is None or
            (self._current_frame - self._last_keyframe_frame) >= self._config.keyframe_interval
        )

        if should_keyframe:
            result = self._create_keyframe(filtered_state, timestamp, metadata)
        elif self._config.enable_delta_compression:
            result = self._create_delta(filtered_state, timestamp)
        else:
            result = None

        self._last_state = filtered_state
        self._current_frame += 1

        return result

    def get_state_at_frame(self, frame: int) -> Optional[dict[str, Any]]:
        """Get game state at specific frame.

        Uses keyframes and deltas to reconstruct state efficiently.

        Args:
            frame: Target frame number

        Returns:
            Reconstructed state at frame, or None if not available
        """
        # Find nearest keyframe before target frame
        keyframe = self._find_nearest_keyframe(frame)
        if keyframe is None:
            return None

        # Start with keyframe state
        state = copy.deepcopy(keyframe.state_data)

        # Apply deltas up to target frame
        for delta in self._deltas:
            if delta.from_frame >= keyframe.frame and delta.to_frame <= frame:
                state = delta.apply(state)

        return state

    def get_nearest_keyframe(self, frame: int) -> Optional[StateSnapshot]:
        """Get the nearest keyframe at or before the specified frame.

        Args:
            frame: Target frame number

        Returns:
            Nearest keyframe snapshot, or None if not available
        """
        return self._find_nearest_keyframe(frame)

    def get_snapshots_in_range(
        self,
        start_frame: int,
        end_frame: int
    ) -> list[StateSnapshot]:
        """Get all snapshots within frame range.

        Args:
            start_frame: Start frame (inclusive)
            end_frame: End frame (inclusive)

        Returns:
            List of snapshots in range
        """
        return [
            s for s in self._snapshots
            if start_frame <= s.frame <= end_frame
        ]

    def get_deltas_in_range(
        self,
        start_frame: int,
        end_frame: int
    ) -> list[StateDelta]:
        """Get all deltas within frame range.

        Args:
            start_frame: Start frame (inclusive)
            end_frame: End frame (inclusive)

        Returns:
            List of deltas in range
        """
        return [
            d for d in self._deltas
            if start_frame <= d.from_frame and d.to_frame <= end_frame
        ]

    def iter_snapshots(self) -> Iterator[StateSnapshot]:
        """Iterate over all snapshots.

        Yields:
            State snapshots in order
        """
        yield from self._snapshots

    def iter_deltas(self) -> Iterator[StateDelta]:
        """Iterate over all deltas.

        Yields:
            State deltas in order
        """
        yield from self._deltas

    def clear(self) -> None:
        """Clear all recorded data."""
        self._snapshots.clear()
        self._deltas.clear()
        self._last_state = None
        self._last_keyframe_state = None
        self._last_keyframe_frame = 0
        self._current_frame = 0

    def serialize(self) -> bytes:
        """Serialize all recorded data to bytes.

        Returns:
            Serialized byte representation
        """
        # Serialize snapshots
        snapshot_parts = []
        for snapshot in self._snapshots:
            snapshot_parts.append(snapshot.serialize(self._config.compression))

        # Serialize deltas
        delta_parts = []
        for delta in self._deltas:
            delta_parts.append(delta.serialize())

        # Pack counts and data
        snapshot_data = b''.join(snapshot_parts)
        delta_data = b''.join(delta_parts)

        header = struct.pack('<II', len(self._snapshots), len(self._deltas))
        return (
            header +
            struct.pack('<I', len(snapshot_data)) + snapshot_data +
            struct.pack('<I', len(delta_data)) + delta_data
        )

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        config: Optional[StateRecordingConfig] = None
    ) -> 'StateRecorder':
        """Deserialize recorder from bytes.

        Args:
            data: Serialized byte data
            config: Optional configuration

        Returns:
            StateRecorder with deserialized data
        """
        recorder = cls(config)
        offset = 0

        # Read counts
        snapshot_count, delta_count = struct.unpack('<II', data[offset:offset + 8])
        offset += 8

        # Read snapshot data
        snapshot_data_len = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        snapshot_offset = 0
        snapshot_data = data[offset:offset + snapshot_data_len]
        offset += snapshot_data_len

        for _ in range(snapshot_count):
            snapshot, snapshot_offset = StateSnapshot.deserialize(
                snapshot_data, snapshot_offset
            )
            recorder._snapshots.append(snapshot)

        # Read delta data
        delta_data_len = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        delta_offset = 0
        delta_data = data[offset:offset + delta_data_len]

        for _ in range(delta_count):
            delta, delta_offset = StateDelta.deserialize(delta_data, delta_offset)
            recorder._deltas.append(delta)

        return recorder

    def _filter_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Filter state based on configuration."""
        if self._config.state_filter:
            state = self._config.state_filter(state)

        # Apply path exclusions
        if self._config.excluded_paths:
            state = self._exclude_paths(state, self._config.excluded_paths)

        return state

    def _exclude_paths(
        self,
        state: dict[str, Any],
        paths: set[str]
    ) -> dict[str, Any]:
        """Exclude specified paths from state."""
        result = copy.deepcopy(state)

        for path in paths:
            self._remove_path(result, path)

        return result

    def _remove_path(self, obj: dict, path: str) -> None:
        """Remove a path from nested dict."""
        parts = path.split('.')
        for part in parts[:-1]:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return

        if isinstance(obj, dict):
            obj.pop(parts[-1], None)

    def _create_keyframe(
        self,
        state: dict[str, Any],
        timestamp: float,
        metadata: Optional[dict[str, Any]] = None
    ) -> StateSnapshot:
        """Create a keyframe snapshot."""
        # Compute checksum
        state_json = json.dumps(state, sort_keys=True).encode('utf-8')
        checksum = hashlib.sha256(state_json).hexdigest() if self._config.compute_checksums else ''

        snapshot = StateSnapshot(
            frame=self._current_frame,
            timestamp=timestamp,
            state_data=copy.deepcopy(state),
            checksum=checksum,
            size_bytes=len(state_json),
            is_keyframe=True,
            metadata=metadata or {}
        )

        self._snapshots.append(snapshot)
        self._last_keyframe_state = copy.deepcopy(state)
        self._last_keyframe_frame = self._current_frame
        self._stats['total_snapshots'] += 1
        self._stats['total_bytes'] += snapshot.size_bytes

        return snapshot

    def _create_delta(
        self,
        state: dict[str, Any],
        timestamp: float
    ) -> Optional[StateDelta]:
        """Create a delta from previous state."""
        if self._last_state is None:
            return None

        # Compute differences
        changes = self._compute_changes(self._last_state, state)

        if not changes:
            return None

        delta = StateDelta(
            from_frame=self._current_frame - 1,
            to_frame=self._current_frame,
            timestamp=timestamp,
            changes=changes,
            size_bytes=0  # Will be computed on serialization
        )

        self._deltas.append(delta)
        self._stats['total_deltas'] += 1

        return delta

    def _compute_changes(
        self,
        old_state: dict[str, Any],
        new_state: dict[str, Any],
        prefix: str = ''
    ) -> list[tuple[str, Any, Any]]:
        """Compute changes between two states."""
        changes = []

        all_keys = set(old_state.keys()) | set(new_state.keys())

        for key in all_keys:
            path = f"{prefix}.{key}" if prefix else key
            old_val = old_state.get(key)
            new_val = new_state.get(key)

            if old_val != new_val:
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    # Recurse into nested dicts
                    changes.extend(self._compute_changes(old_val, new_val, path))
                else:
                    changes.append((path, old_val, new_val))

        return changes

    def _find_nearest_keyframe(self, frame: int) -> Optional[StateSnapshot]:
        """Find the nearest keyframe at or before the specified frame."""
        nearest = None

        for snapshot in self._snapshots:
            if snapshot.is_keyframe and snapshot.frame <= frame:
                if nearest is None or snapshot.frame > nearest.frame:
                    nearest = snapshot

        return nearest
