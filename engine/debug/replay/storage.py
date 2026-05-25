"""Efficient replay storage with compression and delta encoding.

This module provides storage utilities for replay data, including
compression, keyframe storage, and delta encoding for efficient storage.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TypeVar

from engine.debug.replay.recorder import InputRecord, StateSnapshot


T = TypeVar("T")


class CompressionLevel:
    """Compression level constants."""
    NONE = 0
    FAST = 1
    BALANCED = 6
    BEST = 9


@dataclass
class DeltaData:
    """Represents changes between two state snapshots.

    Attributes:
        added: Keys that were added
        removed: Keys that were removed
        modified: Keys whose values changed (contains new values)
    """
    added: dict[str, Any] = field(default_factory=dict)
    removed: set[str] = field(default_factory=set)
    modified: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Check if delta contains no changes."""
        return not self.added and not self.removed and not self.modified

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "added": self.added,
            "removed": list(self.removed),
            "modified": self.modified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeltaData:
        """Create from dictionary."""
        return cls(
            added=data.get("added", {}),
            removed=set(data.get("removed", [])),
            modified=data.get("modified", {}),
        )


class DeltaEncoder:
    """Encodes and decodes state deltas for efficient storage.

    Delta encoding stores only the differences between consecutive
    state snapshots, significantly reducing storage requirements
    for games with large state that changes incrementally.

    Example:
        encoder = DeltaEncoder()

        # Compute delta
        delta = encoder.encode_delta(old_state, new_state)

        # Reconstruct state from delta
        reconstructed = encoder.apply_delta(old_state, delta)
    """

    @staticmethod
    def encode_delta(
        old_state: dict[str, Any],
        new_state: dict[str, Any],
    ) -> DeltaData:
        """Compute the difference between two state dictionaries.

        Args:
            old_state: Previous state
            new_state: Current state

        Returns:
            Delta representing the changes
        """
        old_keys = set(old_state.keys())
        new_keys = set(new_state.keys())

        delta = DeltaData()
        delta.added = {k: new_state[k] for k in new_keys - old_keys}
        delta.removed = old_keys - new_keys
        delta.modified = {
            k: new_state[k]
            for k in old_keys & new_keys
            if old_state[k] != new_state[k]
        }

        return delta

    @staticmethod
    def apply_delta(base_state: dict[str, Any], delta: DeltaData) -> dict[str, Any]:
        """Apply a delta to reconstruct the new state.

        Args:
            base_state: State to apply delta to
            delta: Changes to apply

        Returns:
            Reconstructed state after applying delta
        """
        result = dict(base_state)

        # Remove deleted keys
        for key in delta.removed:
            result.pop(key, None)

        # Add new keys
        result.update(delta.added)

        # Update modified keys
        result.update(delta.modified)

        return result


class ReplayStorage:
    """Efficient storage for replay data with compression and keyframes.

    ReplayStorage handles saving and loading replay data efficiently by:
    - Compressing data using zlib
    - Using keyframes at regular intervals for fast seeking
    - Delta encoding between keyframes for space efficiency

    Example:
        storage = ReplayStorage()

        # Compress and save
        storage.save_replay(path, inputs, snapshots, keyframe_interval=60)

        # Load
        inputs, snapshots, keyframes = storage.load_replay(path)

        # Seek to tick using nearest keyframe
        state = storage.get_state_at_tick(120, keyframes)
    """

    def __init__(
        self,
        compression_level: int = CompressionLevel.BALANCED,
    ) -> None:
        """Initialize replay storage.

        Args:
            compression_level: zlib compression level (0-9)
        """
        self._compression_level = compression_level
        self._delta_encoder = DeltaEncoder()

    def compress(self, data: bytes) -> bytes:
        """Compress binary data.

        Args:
            data: Data to compress

        Returns:
            Compressed data
        """
        if self._compression_level == CompressionLevel.NONE:
            return data
        return zlib.compress(data, self._compression_level)

    def decompress(self, data: bytes) -> bytes:
        """Decompress binary data.

        Args:
            data: Compressed data

        Returns:
            Decompressed data
        """
        try:
            return zlib.decompress(data)
        except zlib.error:
            # Data may not be compressed
            return data

    def compress_object(self, obj: Any) -> bytes:
        """Pickle and compress an object.

        Args:
            obj: Object to compress

        Returns:
            Compressed pickled data
        """
        pickled = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return self.compress(pickled)

    def decompress_object(self, data: bytes) -> Any:
        """Decompress and unpickle an object.

        Args:
            data: Compressed pickled data

        Returns:
            Reconstructed object
        """
        decompressed = self.decompress(data)
        return pickle.loads(decompressed)

    def save_with_keyframes(
        self,
        path: Path | str,
        inputs: list[InputRecord],
        snapshots: list[StateSnapshot],
        keyframe_interval: int = 60,
    ) -> None:
        """Save replay with keyframe-based storage.

        Keyframes store complete state snapshots at regular intervals.
        Between keyframes, only deltas are stored. This allows efficient
        seeking (jump to keyframe, replay deltas) while saving space.

        Args:
            path: Output file path
            inputs: List of input records
            snapshots: List of state snapshots
            keyframe_interval: Ticks between keyframes
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Identify keyframe ticks
        if not snapshots:
            keyframe_ticks = set()
        else:
            first_tick = snapshots[0].tick
            last_tick = snapshots[-1].tick
            keyframe_ticks = set(
                range(first_tick, last_tick + 1, keyframe_interval)
            )
            # Always include first and last
            keyframe_ticks.add(first_tick)
            keyframe_ticks.add(last_tick)

        # Process snapshots: keyframes get full state, others get delta
        processed_snapshots = []
        last_keyframe_state: dict[str, Any] | None = None

        for snapshot in snapshots:
            if snapshot.tick in keyframe_ticks:
                # Full keyframe
                processed_snapshots.append({
                    "tick": snapshot.tick,
                    "timestamp": snapshot.timestamp,
                    "is_keyframe": True,
                    "state_data": snapshot.state_data,
                })
                last_keyframe_state = snapshot.state_data
            elif last_keyframe_state is not None:
                # Delta from last keyframe
                delta = self._delta_encoder.encode_delta(
                    last_keyframe_state,
                    snapshot.state_data,
                )
                processed_snapshots.append({
                    "tick": snapshot.tick,
                    "timestamp": snapshot.timestamp,
                    "is_keyframe": False,
                    "delta": delta.to_dict(),
                })

        # Build output data
        output = {
            "type": "keyframe_replay",
            "version": 2,
            "compression": self._compression_level,
            "keyframe_interval": keyframe_interval,
            "inputs": [i.to_dict() for i in inputs],
            "snapshots": processed_snapshots,
        }

        # Compress and save
        compressed = self.compress_object(output)
        with open(path, "wb") as f:
            f.write(compressed)

    def load_with_keyframes(
        self,
        path: Path | str,
    ) -> tuple[list[InputRecord], list[StateSnapshot], dict[int, StateSnapshot]]:
        """Load replay with keyframe support.

        Returns inputs, full snapshots (with deltas applied), and a
        dictionary of keyframe snapshots for fast seeking.

        Args:
            path: Replay file path

        Returns:
            Tuple of (inputs, snapshots, keyframes_dict)
        """
        path = Path(path)
        with open(path, "rb") as f:
            compressed = f.read()

        data = self.decompress_object(compressed)

        if data.get("type") != "keyframe_replay":
            raise ValueError(f"Invalid replay format: expected keyframe_replay")

        inputs = [InputRecord.from_dict(i) for i in data.get("inputs", [])]

        # Reconstruct snapshots from keyframes and deltas
        snapshots = []
        keyframes: dict[int, StateSnapshot] = {}
        last_keyframe_state: dict[str, Any] | None = None

        for snap_data in data.get("snapshots", []):
            tick = snap_data["tick"]
            timestamp = snap_data["timestamp"]

            if snap_data.get("is_keyframe", False):
                # Full keyframe
                state_data = snap_data["state_data"]
                last_keyframe_state = state_data
                snapshot = StateSnapshot(tick=tick, state_data=state_data, timestamp=timestamp)
                keyframes[tick] = snapshot
            else:
                # Delta - reconstruct from last keyframe
                if last_keyframe_state is None:
                    continue
                delta = DeltaData.from_dict(snap_data["delta"])
                state_data = self._delta_encoder.apply_delta(
                    last_keyframe_state,
                    delta,
                )
                snapshot = StateSnapshot(tick=tick, state_data=state_data, timestamp=timestamp)

            snapshots.append(snapshot)

        return inputs, snapshots, keyframes

    def get_state_at_tick(
        self,
        tick: int,
        snapshots: list[StateSnapshot],
        keyframes: dict[int, StateSnapshot],
    ) -> StateSnapshot | None:
        """Efficiently get state at a specific tick using keyframes.

        Args:
            tick: Target tick
            snapshots: All snapshots
            keyframes: Keyframe lookup dictionary

        Returns:
            Snapshot at or before tick, or None
        """
        # Find exact match
        for s in snapshots:
            if s.tick == tick:
                return s

        # Find nearest keyframe at or before tick
        nearest_keyframe_tick = None
        for kf_tick in keyframes:
            if kf_tick <= tick:
                if nearest_keyframe_tick is None or kf_tick > nearest_keyframe_tick:
                    nearest_keyframe_tick = kf_tick

        if nearest_keyframe_tick is None:
            return None

        return keyframes[nearest_keyframe_tick]

    def compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum of data.

        Args:
            data: Data to checksum

        Returns:
            Hex-encoded checksum
        """
        return hashlib.sha256(data).hexdigest()

    def save_replay(
        self,
        path: Path | str,
        inputs: list[InputRecord],
        snapshots: list[StateSnapshot],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save replay with simple compression (no keyframes).

        Args:
            path: Output file path
            inputs: Input records
            snapshots: State snapshots
            metadata: Optional metadata to include
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "type": "simple_replay",
            "version": 1,
            "compression": self._compression_level,
            "metadata": metadata or {},
            "inputs": [i.to_dict() for i in inputs],
            "snapshots": [s.to_dict() for s in snapshots],
        }

        compressed = self.compress_object(output)

        # Add checksum
        checksum = self.compute_checksum(compressed)
        with open(path, "wb") as f:
            # Write checksum as header (64 bytes hex)
            f.write(checksum.encode("ascii"))
            f.write(compressed)

    def load_replay(
        self,
        path: Path | str,
        verify_checksum: bool = True,
    ) -> tuple[list[InputRecord], list[StateSnapshot], dict[str, Any]]:
        """Load a simple replay file.

        Args:
            path: Replay file path
            verify_checksum: Whether to verify data integrity

        Returns:
            Tuple of (inputs, snapshots, metadata)

        Raises:
            ValueError: If checksum verification fails
        """
        path = Path(path)
        with open(path, "rb") as f:
            # Read checksum header
            stored_checksum = f.read(64).decode("ascii")
            compressed = f.read()

        if verify_checksum:
            actual_checksum = self.compute_checksum(compressed)
            if stored_checksum != actual_checksum:
                raise ValueError(f"Checksum mismatch: replay file may be corrupted")

        data = self.decompress_object(compressed)

        if data.get("type") != "simple_replay":
            raise ValueError(f"Invalid replay format")

        inputs = [InputRecord.from_dict(i) for i in data.get("inputs", [])]
        snapshots = [StateSnapshot.from_dict(s) for s in data.get("snapshots", [])]
        metadata = data.get("metadata", {})

        return inputs, snapshots, metadata

    def get_compressed_size(self, data: Any) -> int:
        """Get the compressed size of data without saving.

        Args:
            data: Data to measure

        Returns:
            Compressed size in bytes
        """
        return len(self.compress_object(data))

    def get_compression_ratio(self, data: Any) -> float:
        """Calculate the compression ratio.

        Args:
            data: Data to measure

        Returns:
            Ratio of compressed to original size (lower is better)
        """
        original = len(pickle.dumps(data))
        compressed = self.get_compressed_size(data)
        return compressed / original if original > 0 else 1.0


class ContentAddressedStorage:
    """Content-addressed storage for deduplicated replay data.

    Stores data chunks by their content hash, enabling deduplication
    of repeated state patterns. Useful for games with lots of static
    or repeated state.

    Example:
        cas = ContentAddressedStorage(base_path)

        # Store data
        hash_id = cas.store(state_data)

        # Retrieve data
        data = cas.retrieve(hash_id)
    """

    def __init__(self, base_path: Path | str) -> None:
        """Initialize content-addressed storage.

        Args:
            base_path: Base directory for storage
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._compression_level = CompressionLevel.BALANCED

    def _get_path(self, content_hash: str) -> Path:
        """Get file path for a content hash.

        Uses two-level directory structure to avoid too many files
        in a single directory.

        Args:
            content_hash: Content hash

        Returns:
            File path
        """
        return self._base_path / content_hash[:2] / content_hash[2:]

    def store(self, data: Any) -> str:
        """Store data and return its content hash.

        If data with the same hash already exists, this is a no-op.

        Args:
            data: Data to store

        Returns:
            Content hash identifier
        """
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(serialized, self._compression_level)
        content_hash = hashlib.sha256(compressed).hexdigest()

        path = self._get_path(content_hash)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(compressed)

        return content_hash

    def retrieve(self, content_hash: str) -> Any:
        """Retrieve data by its content hash.

        Args:
            content_hash: Content hash identifier

        Returns:
            Stored data

        Raises:
            KeyError: If content not found
        """
        path = self._get_path(content_hash)
        if not path.exists():
            raise KeyError(f"Content not found: {content_hash}")

        with open(path, "rb") as f:
            compressed = f.read()

        serialized = zlib.decompress(compressed)
        return pickle.loads(serialized)

    def exists(self, content_hash: str) -> bool:
        """Check if content exists.

        Args:
            content_hash: Content hash identifier

        Returns:
            True if content exists
        """
        return self._get_path(content_hash).exists()

    def delete(self, content_hash: str) -> bool:
        """Delete content by hash.

        Args:
            content_hash: Content hash identifier

        Returns:
            True if content was deleted, False if not found
        """
        path = self._get_path(content_hash)
        if path.exists():
            path.unlink()
            # Clean up empty parent directory
            try:
                path.parent.rmdir()
            except OSError:
                pass
            return True
        return False

    def list_all(self) -> list[str]:
        """List all stored content hashes.

        Returns:
            List of content hashes
        """
        hashes = []
        for subdir in self._base_path.iterdir():
            if subdir.is_dir():
                for file in subdir.iterdir():
                    if file.is_file():
                        hashes.append(subdir.name + file.name)
        return hashes

    def get_total_size(self) -> int:
        """Get total storage size in bytes.

        Returns:
            Total size
        """
        total = 0
        for subdir in self._base_path.iterdir():
            if subdir.is_dir():
                for file in subdir.iterdir():
                    if file.is_file():
                        total += file.stat().st_size
        return total
