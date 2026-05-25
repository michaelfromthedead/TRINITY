"""
Delta compression for network state synchronization.

Provides efficient encoding of state changes by only transmitting
differences from a known baseline state.
"""

from __future__ import annotations

import hashlib
import logging
import struct
import zlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar

from .bit_packer import BitReader, BitWriter
from ..config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class DeltaFieldType(IntEnum):
    """Types of fields for delta encoding."""
    INT8 = 1
    INT16 = 2
    INT32 = 3
    INT64 = 4
    UINT8 = 5
    UINT16 = 6
    UINT32 = 7
    UINT64 = 8
    FLOAT32 = 9
    FLOAT64 = 10
    BOOL = 11
    BYTES = 12
    STRING = 13


@dataclass
class DeltaFieldDescriptor:
    """Describes a field for delta encoding."""
    name: str
    field_type: DeltaFieldType
    index: int
    max_length: int = 255  # For bytes/string fields


@dataclass
class DeltaSchema:
    """Schema for a delta-encodable structure."""
    name: str
    version: int
    fields: List[DeltaFieldDescriptor]

    def __post_init__(self):
        self._field_map = {f.name: f for f in self.fields}
        self._index_map = {f.index: f for f in self.fields}

    def get_field(self, name: str) -> Optional[DeltaFieldDescriptor]:
        return self._field_map.get(name)

    def get_field_by_index(self, index: int) -> Optional[DeltaFieldDescriptor]:
        return self._index_map.get(index)


@dataclass
class DeltaBaseline:
    """
    Represents a baseline state for delta encoding.

    Attributes:
        sequence: Sequence number of this baseline.
        state: The baseline state data.
        hash: Hash of the state for verification.
        timestamp: When this baseline was created.
    """
    sequence: int
    state: Dict[str, Any]
    hash: bytes
    timestamp: float = 0.0

    @staticmethod
    def compute_hash(state: Dict[str, Any]) -> bytes:
        """Compute hash of state dictionary."""
        # Sort keys for deterministic hashing
        sorted_items = sorted(state.items())
        content = str(sorted_items).encode('utf-8')
        return hashlib.md5(content).digest()[:DEFAULT_CONFIG.BASELINE_HASH_SIZE]


class DeltaEncoder:
    """
    Encodes state changes as deltas from a baseline.

    Supports:
    - Tracking multiple baselines for different acknowledgment scenarios
    - Automatic baseline management with ACK handling
    - Bit-level delta encoding for efficiency
    - Optional compression for large deltas

    Example:
        encoder = DeltaEncoder()

        # Set initial baseline
        baseline = {'x': 0.0, 'y': 0.0, 'health': 100}
        encoder.set_baseline(0, baseline)

        # Encode delta
        current = {'x': 10.5, 'y': 0.0, 'health': 95}
        delta = encoder.encode_delta(current, baseline_seq=0)

        # Decode delta
        decoded = encoder.decode_delta(delta, baseline_seq=0)
    """

    def __init__(
        self,
        schema: Optional[DeltaSchema] = None,
        max_baselines: int = DEFAULT_CONFIG.MAX_BASELINES,
        compress_threshold: int = DEFAULT_CONFIG.DELTA_COMPRESS_THRESHOLD
    ) -> None:
        """
        Initialize the delta encoder.

        Args:
            schema: Optional schema for typed encoding.
            max_baselines: Maximum number of baselines to track.
            compress_threshold: Compress deltas larger than this (bytes).
        """
        self._schema = schema
        self._max_baselines = max_baselines
        self._compress_threshold = compress_threshold
        self._baselines: Dict[int, DeltaBaseline] = {}
        self._latest_sequence = -1

    @property
    def latest_sequence(self) -> int:
        """Get the latest baseline sequence number."""
        return self._latest_sequence

    def set_baseline(
        self,
        sequence: int,
        state: Dict[str, Any],
        timestamp: float = 0.0
    ) -> DeltaBaseline:
        """
        Set a new baseline state.

        Args:
            sequence: Sequence number for this baseline.
            state: The state dictionary.
            timestamp: Optional timestamp.

        Returns:
            The created baseline.
        """
        # Clean up old baselines if at capacity
        if len(self._baselines) >= self._max_baselines:
            # Remove oldest
            oldest_seq = min(self._baselines.keys())
            del self._baselines[oldest_seq]

        baseline = DeltaBaseline(
            sequence=sequence,
            state=dict(state),  # Copy
            hash=DeltaBaseline.compute_hash(state),
            timestamp=timestamp
        )
        self._baselines[sequence] = baseline

        if sequence > self._latest_sequence:
            self._latest_sequence = sequence

        return baseline

    def get_baseline(self, sequence: int) -> Optional[DeltaBaseline]:
        """Get a baseline by sequence number."""
        return self._baselines.get(sequence)

    def acknowledge_baseline(self, sequence: int) -> None:
        """
        Acknowledge receipt of a baseline.

        Removes all baselines older than the acknowledged one.

        Args:
            sequence: The acknowledged sequence number.
        """
        to_remove = [seq for seq in self._baselines if seq < sequence]
        for seq in to_remove:
            del self._baselines[seq]

    def encode_delta(
        self,
        current_state: Dict[str, Any],
        baseline_seq: int
    ) -> bytes:
        """
        Encode the difference between current state and a baseline.

        Args:
            current_state: The current state to encode.
            baseline_seq: Sequence number of the baseline to diff against.

        Returns:
            Encoded delta bytes.

        Raises:
            KeyError: If baseline_seq doesn't exist.
        """
        baseline = self._baselines.get(baseline_seq)
        if baseline is None:
            raise KeyError(f"No baseline with sequence {baseline_seq}")

        return self._encode_state_delta(current_state, baseline.state)

    def decode_delta(
        self,
        delta_bytes: bytes,
        baseline_seq: int
    ) -> Dict[str, Any]:
        """
        Decode a delta and apply it to a baseline.

        Args:
            delta_bytes: The encoded delta.
            baseline_seq: Sequence number of the baseline.

        Returns:
            The reconstructed current state.

        Raises:
            KeyError: If baseline_seq doesn't exist.
        """
        baseline = self._baselines.get(baseline_seq)
        if baseline is None:
            raise KeyError(f"No baseline with sequence {baseline_seq}")

        return self._decode_state_delta(delta_bytes, baseline.state)

    def _encode_state_delta(
        self,
        current: Dict[str, Any],
        baseline: Dict[str, Any]
    ) -> bytes:
        """Encode delta between two state dictionaries."""
        writer = BitWriter()

        # Collect changed fields
        changed_fields: List[Tuple[str, Any]] = []

        # Check for modified or new fields
        for key, value in current.items():
            if key not in baseline or baseline[key] != value:
                changed_fields.append((key, value))

        # Check for removed fields
        removed_fields: List[str] = []
        for key in baseline:
            if key not in current:
                removed_fields.append(key)

        # Write header: number of changed fields and removed fields
        writer.write_bits(len(changed_fields), 16)
        writer.write_bits(len(removed_fields), 16)

        # Write changed fields
        for key, value in changed_fields:
            self._write_field(writer, key, value)

        # Write removed field keys
        for key in removed_fields:
            self._write_string(writer, key)

        # Get raw data
        raw_data = writer.to_bytes()

        # Compress if above threshold
        if len(raw_data) > self._compress_threshold:
            compressed = zlib.compress(raw_data, level=DEFAULT_CONFIG.COMPRESSION_LEVEL)
            if len(compressed) < len(raw_data):
                # Prepend compression flag
                return b'\x01' + compressed

        return b'\x00' + raw_data

    def _decode_state_delta(
        self,
        delta_bytes: bytes,
        baseline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decode delta and apply to baseline."""
        if not delta_bytes:
            return dict(baseline)

        # Check compression flag
        is_compressed = delta_bytes[0] == 0x01
        data = delta_bytes[1:]

        if is_compressed:
            data = zlib.decompress(data)

        reader = BitReader(data)

        # Start with baseline copy
        result = dict(baseline)

        # Read number of changed and removed fields
        num_changed = reader.read_bits(16)
        num_removed = reader.read_bits(16)

        # Read changed fields
        for _ in range(num_changed):
            key, value = self._read_field(reader)
            result[key] = value

        # Read removed fields
        for _ in range(num_removed):
            key = self._read_string(reader)
            result.pop(key, None)

        return result

    def _write_field(self, writer: BitWriter, key: str, value: Any) -> None:
        """Write a field (key + type + value) to the writer."""
        # Write key
        self._write_string(writer, key)

        # Determine type and write
        if isinstance(value, bool):
            writer.write_bits(DeltaFieldType.BOOL, 4)
            writer.write_bool(value)
        elif isinstance(value, int):
            if -128 <= value <= 127:
                writer.write_bits(DeltaFieldType.INT8, 4)
                writer.write_bits(value & 0xFF, 8)
            elif -32768 <= value <= 32767:
                writer.write_bits(DeltaFieldType.INT16, 4)
                writer.write_bits(value & 0xFFFF, 16)
            elif -2147483648 <= value <= 2147483647:
                writer.write_bits(DeltaFieldType.INT32, 4)
                writer.write_bits(value & 0xFFFFFFFF, 32)
            else:
                writer.write_bits(DeltaFieldType.INT64, 4)
                writer.write_bits(value & 0xFFFFFFFFFFFFFFFF, 64)
        elif isinstance(value, float):
            writer.write_bits(DeltaFieldType.FLOAT32, 4)
            packed = struct.pack('!f', value)
            for byte in packed:
                writer.write_bits(byte, 8)
        elif isinstance(value, bytes):
            writer.write_bits(DeltaFieldType.BYTES, 4)
            writer.write_bits(len(value), 16)
            for byte in value:
                writer.write_bits(byte, 8)
        elif isinstance(value, str):
            writer.write_bits(DeltaFieldType.STRING, 4)
            self._write_string(writer, value)
        else:
            # Fallback: serialize as string
            writer.write_bits(DeltaFieldType.STRING, 4)
            self._write_string(writer, str(value))

    def _read_field(self, reader: BitReader) -> Tuple[str, Any]:
        """Read a field from the reader."""
        key = self._read_string(reader)
        field_type = DeltaFieldType(reader.read_bits(4))

        if field_type == DeltaFieldType.BOOL:
            value = reader.read_bool()
        elif field_type == DeltaFieldType.INT8:
            raw = reader.read_bits(8)
            value = raw if raw < 128 else raw - 256
        elif field_type == DeltaFieldType.INT16:
            raw = reader.read_bits(16)
            value = raw if raw < 32768 else raw - 65536
        elif field_type == DeltaFieldType.INT32:
            raw = reader.read_bits(32)
            value = raw if raw < 2147483648 else raw - 4294967296
        elif field_type == DeltaFieldType.INT64:
            raw = reader.read_bits(64)
            value = raw if raw < 9223372036854775808 else raw - 18446744073709551616
        elif field_type == DeltaFieldType.FLOAT32:
            packed = bytes([reader.read_bits(8) for _ in range(4)])
            value = struct.unpack('!f', packed)[0]
        elif field_type == DeltaFieldType.FLOAT64:
            packed = bytes([reader.read_bits(8) for _ in range(8)])
            value = struct.unpack('!d', packed)[0]
        elif field_type == DeltaFieldType.BYTES:
            length = reader.read_bits(16)
            value = bytes([reader.read_bits(8) for _ in range(length)])
        elif field_type == DeltaFieldType.STRING:
            value = self._read_string(reader)
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        return key, value

    def _write_string(self, writer: BitWriter, value: str) -> None:
        """Write a length-prefixed string."""
        encoded = value.encode('utf-8')
        writer.write_bits(len(encoded), 16)
        for byte in encoded:
            writer.write_bits(byte, 8)

    def _read_string(self, reader: BitReader) -> str:
        """Read a length-prefixed string."""
        length = reader.read_bits(16)
        data = bytes([reader.read_bits(8) for _ in range(length)])
        return data.decode('utf-8')

    def encode_full_state(self, state: Dict[str, Any]) -> bytes:
        """
        Encode a full state without delta compression.

        Used for initial synchronization or when no baseline exists.

        Args:
            state: The state to encode.

        Returns:
            Encoded bytes.
        """
        # Encode as delta from empty state
        return self._encode_state_delta(state, {})

    def decode_full_state(self, data: bytes) -> Dict[str, Any]:
        """
        Decode a full state.

        Args:
            data: Encoded state bytes.

        Returns:
            The decoded state dictionary.
        """
        return self._decode_state_delta(data, {})

    def clear_baselines(self) -> None:
        """Clear all stored baselines."""
        self._baselines.clear()
        self._latest_sequence = -1

    def get_baseline_count(self) -> int:
        """Get the number of stored baselines."""
        return len(self._baselines)


class SnapshotDeltaEncoder:
    """
    Specialized delta encoder for game state snapshots.

    Optimized for encoding multiple entity states with efficient
    change detection and serialization.
    """

    def __init__(self, encoder: Optional[DeltaEncoder] = None) -> None:
        """
        Initialize snapshot encoder.

        Args:
            encoder: Optional underlying delta encoder.
        """
        self._encoder = encoder or DeltaEncoder()
        self._entity_baselines: Dict[int, Dict[str, Any]] = {}

    def encode_snapshot(
        self,
        entities: Dict[int, Dict[str, Any]],
        baseline_entities: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> bytes:
        """
        Encode a snapshot of entity states.

        Args:
            entities: Current entity states (entity_id -> state).
            baseline_entities: Previous entity states for delta encoding.

        Returns:
            Encoded snapshot bytes.
        """
        writer = BitWriter()

        baseline = baseline_entities or {}

        # Find added, modified, and removed entities
        current_ids = set(entities.keys())
        baseline_ids = set(baseline.keys())

        added_ids = current_ids - baseline_ids
        removed_ids = baseline_ids - current_ids
        common_ids = current_ids & baseline_ids

        # Filter common to only changed
        changed_ids = {
            eid for eid in common_ids
            if entities[eid] != baseline.get(eid, {})
        }

        # Write counts
        writer.write_bits(len(added_ids), 16)
        writer.write_bits(len(changed_ids), 16)
        writer.write_bits(len(removed_ids), 16)

        # Write added entities (full state)
        for eid in sorted(added_ids):
            writer.write_bits(eid, 32)
            full_data = self._encoder.encode_full_state(entities[eid])
            writer.write_bits(len(full_data), 16)
            writer.write_bytes(full_data)

        # Write changed entities (delta)
        for eid in sorted(changed_ids):
            writer.write_bits(eid, 32)
            delta_data = self._encoder._encode_state_delta(
                entities[eid],
                baseline[eid]
            )
            writer.write_bits(len(delta_data), 16)
            writer.write_bytes(delta_data)

        # Write removed entity IDs
        for eid in sorted(removed_ids):
            writer.write_bits(eid, 32)

        return writer.to_bytes()

    def decode_snapshot(
        self,
        data: bytes,
        baseline_entities: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Decode a snapshot.

        Args:
            data: Encoded snapshot bytes.
            baseline_entities: Previous entity states.

        Returns:
            Decoded entity states.
        """
        reader = BitReader(data)
        baseline = baseline_entities or {}

        # Start with copy of baseline
        result = {eid: dict(state) for eid, state in baseline.items()}

        # Read counts
        num_added = reader.read_bits(16)
        num_changed = reader.read_bits(16)
        num_removed = reader.read_bits(16)

        # Read added entities
        for _ in range(num_added):
            eid = reader.read_bits(32)
            data_len = reader.read_bits(16)
            entity_data = reader.read_bytes(data_len)
            result[eid] = self._encoder.decode_full_state(entity_data)

        # Read changed entities
        for _ in range(num_changed):
            eid = reader.read_bits(32)
            data_len = reader.read_bits(16)
            delta_data = reader.read_bytes(data_len)
            result[eid] = self._encoder._decode_state_delta(
                delta_data,
                baseline.get(eid, {})
            )

        # Remove entities
        for _ in range(num_removed):
            eid = reader.read_bits(32)
            result.pop(eid, None)

        return result
