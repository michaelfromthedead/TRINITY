"""
Replay File - Replay file format with compression and metadata.

Provides a standardized file format for storing replays including
header with metadata, input stream, and state keyframes.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, BinaryIO, Optional

from .input_recorder import RecordedInput, InputRecorder
from .state_recorder import StateSnapshot, StateDelta, StateRecorder, CompressionMethod


# File format magic number
REPLAY_MAGIC = b'RPLY'
REPLAY_VERSION = 1


class ReplayFileFormat(Enum):
    """Supported replay file formats."""
    BINARY = auto()  # Native binary format
    COMPRESSED = auto()  # Compressed binary
    JSON = auto()  # Human-readable JSON (debug only)


class ReplayFileError(Exception):
    """Exception raised for replay file errors."""
    pass


@dataclass
class ReplayMetadata:
    """Metadata stored with replay files."""
    # Recording info
    game_name: str = ""
    game_version: str = ""
    map_name: str = ""
    game_mode: str = ""

    # Session info
    session_id: str = ""
    player_name: str = ""
    player_id: str = ""

    # Timing
    recorded_at: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    total_frames: int = 0

    # Statistics
    input_count: int = 0
    snapshot_count: int = 0
    delta_count: int = 0

    # Verification
    checksum: str = ""
    input_hash: str = ""

    # Custom data
    custom: dict[str, Any] = field(default_factory=dict)

    # Tags for filtering
    tags: list[str] = field(default_factory=list)

    # Result info (for competitive games)
    result: Optional[str] = None  # "win", "loss", "draw", etc.
    score: Optional[int] = None
    opponent: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'game_name': self.game_name,
            'game_version': self.game_version,
            'map_name': self.map_name,
            'game_mode': self.game_mode,
            'session_id': self.session_id,
            'player_name': self.player_name,
            'player_id': self.player_id,
            'recorded_at': self.recorded_at.isoformat(),
            'duration': self.duration,
            'total_frames': self.total_frames,
            'input_count': self.input_count,
            'snapshot_count': self.snapshot_count,
            'delta_count': self.delta_count,
            'checksum': self.checksum,
            'input_hash': self.input_hash,
            'custom': self.custom,
            'tags': self.tags,
            'result': self.result,
            'score': self.score,
            'opponent': self.opponent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ReplayMetadata':
        """Create from dictionary."""
        recorded_at = data.get('recorded_at')
        if isinstance(recorded_at, str):
            recorded_at = datetime.fromisoformat(recorded_at)
        elif recorded_at is None:
            recorded_at = datetime.now()

        return cls(
            game_name=data.get('game_name', ''),
            game_version=data.get('game_version', ''),
            map_name=data.get('map_name', ''),
            game_mode=data.get('game_mode', ''),
            session_id=data.get('session_id', ''),
            player_name=data.get('player_name', ''),
            player_id=data.get('player_id', ''),
            recorded_at=recorded_at,
            duration=data.get('duration', 0.0),
            total_frames=data.get('total_frames', 0),
            input_count=data.get('input_count', 0),
            snapshot_count=data.get('snapshot_count', 0),
            delta_count=data.get('delta_count', 0),
            checksum=data.get('checksum', ''),
            input_hash=data.get('input_hash', ''),
            custom=data.get('custom', {}),
            tags=data.get('tags', []),
            result=data.get('result'),
            score=data.get('score'),
            opponent=data.get('opponent'),
        )


@dataclass
class ReplayHeader:
    """Header section of replay file."""
    magic: bytes = REPLAY_MAGIC
    version: int = REPLAY_VERSION
    format: ReplayFileFormat = ReplayFileFormat.COMPRESSED
    compression: CompressionMethod = CompressionMethod.ZLIB

    # Section offsets
    metadata_offset: int = 0
    inputs_offset: int = 0
    snapshots_offset: int = 0
    deltas_offset: int = 0

    # Section sizes
    metadata_size: int = 0
    inputs_size: int = 0
    snapshots_size: int = 0
    deltas_size: int = 0

    # File integrity
    header_checksum: str = ""
    file_size: int = 0

    def serialize(self) -> bytes:
        """Serialize header to bytes."""
        # Pack fixed header
        data = struct.pack(
            '<4sIBB',
            self.magic,
            self.version,
            self.format.value,
            self.compression.value
        )

        # Pack offsets
        data += struct.pack(
            '<IIII',
            self.metadata_offset,
            self.inputs_offset,
            self.snapshots_offset,
            self.deltas_offset
        )

        # Pack sizes
        data += struct.pack(
            '<IIII',
            self.metadata_size,
            self.inputs_size,
            self.snapshots_size,
            self.deltas_size
        )

        # Pack file size and checksum
        data += struct.pack('<Q', self.file_size)
        checksum_bytes = bytes.fromhex(self.header_checksum) if self.header_checksum else b'\x00' * 32
        data += checksum_bytes

        return data

    @classmethod
    def deserialize(cls, data: bytes) -> 'ReplayHeader':
        """Deserialize header from bytes."""
        offset = 0

        # Read fixed header
        magic, version, format_val, compression_val = struct.unpack(
            '<4sIBB', data[offset:offset + 10]
        )
        offset += 10

        if magic != REPLAY_MAGIC:
            raise ReplayFileError(f"Invalid replay file magic: {magic}")

        # Read offsets
        meta_off, inputs_off, snaps_off, deltas_off = struct.unpack(
            '<IIII', data[offset:offset + 16]
        )
        offset += 16

        # Read sizes
        meta_size, inputs_size, snaps_size, deltas_size = struct.unpack(
            '<IIII', data[offset:offset + 16]
        )
        offset += 16

        # Read file size and checksum
        file_size = struct.unpack('<Q', data[offset:offset + 8])[0]
        offset += 8
        checksum = data[offset:offset + 32].hex()

        return cls(
            magic=magic,
            version=version,
            format=ReplayFileFormat(format_val),
            compression=CompressionMethod(compression_val),
            metadata_offset=meta_off,
            inputs_offset=inputs_off,
            snapshots_offset=snaps_off,
            deltas_offset=deltas_off,
            metadata_size=meta_size,
            inputs_size=inputs_size,
            snapshots_size=snaps_size,
            deltas_size=deltas_size,
            header_checksum=checksum,
            file_size=file_size
        )

    @staticmethod
    def header_size() -> int:
        """Get size of serialized header."""
        return 10 + 16 + 16 + 8 + 32  # 82 bytes


class ReplayFile:
    """Handles replay file reading and writing.

    Provides a standardized file format for storing replays including
    header with metadata, input stream, and state keyframes.
    """
    __slots__ = (
        '_path', '_header', '_metadata', '_inputs',
        '_snapshots', '_deltas', '_is_loaded', '_compression'
    )

    def __init__(self, path: Optional[str | Path] = None):
        """Initialize replay file.

        Args:
            path: Optional file path
        """
        self._path = Path(path) if path else None
        self._header = ReplayHeader()
        self._metadata = ReplayMetadata()
        self._inputs: list[RecordedInput] = []
        self._snapshots: list[StateSnapshot] = []
        self._deltas: list[StateDelta] = []
        self._is_loaded = False
        self._compression = CompressionMethod.ZLIB

    @property
    def path(self) -> Optional[Path]:
        """Get file path."""
        return self._path

    @property
    def header(self) -> ReplayHeader:
        """Get replay header."""
        return self._header

    @property
    def metadata(self) -> ReplayMetadata:
        """Get replay metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: ReplayMetadata) -> None:
        """Set replay metadata."""
        self._metadata = value

    @property
    def inputs(self) -> list[RecordedInput]:
        """Get recorded inputs."""
        return self._inputs

    @property
    def snapshots(self) -> list[StateSnapshot]:
        """Get state snapshots."""
        return self._snapshots

    @property
    def deltas(self) -> list[StateDelta]:
        """Get state deltas."""
        return self._deltas

    @property
    def is_loaded(self) -> bool:
        """Check if file is loaded."""
        return self._is_loaded

    @property
    def compression(self) -> CompressionMethod:
        """Get compression method."""
        return self._compression

    @compression.setter
    def compression(self, value: CompressionMethod) -> None:
        """Set compression method."""
        self._compression = value

    def set_data(
        self,
        inputs: list[RecordedInput],
        snapshots: list[StateSnapshot],
        deltas: Optional[list[StateDelta]] = None,
        metadata: Optional[ReplayMetadata] = None
    ) -> None:
        """Set replay data.

        Args:
            inputs: List of recorded inputs
            snapshots: List of state snapshots
            deltas: Optional list of state deltas
            metadata: Optional metadata
        """
        self._inputs = inputs
        self._snapshots = snapshots
        self._deltas = deltas or []

        if metadata:
            self._metadata = metadata

        # Update metadata statistics
        self._metadata.input_count = len(self._inputs)
        self._metadata.snapshot_count = len(self._snapshots)
        self._metadata.delta_count = len(self._deltas)

        if self._inputs:
            self._metadata.total_frames = max(i.frame for i in self._inputs)
            self._metadata.duration = max(i.timestamp for i in self._inputs)

        if self._snapshots:
            self._metadata.total_frames = max(
                self._metadata.total_frames,
                max(s.frame for s in self._snapshots)
            )
            self._metadata.duration = max(
                self._metadata.duration,
                max(s.timestamp for s in self._snapshots)
            )

    def save(
        self,
        path: Optional[str | Path] = None,
        format: ReplayFileFormat = ReplayFileFormat.COMPRESSED
    ) -> int:
        """Save replay to file.

        Args:
            path: File path (uses stored path if not provided)
            format: File format to use

        Returns:
            Number of bytes written

        Raises:
            ReplayFileError: If save fails
        """
        save_path = Path(path) if path else self._path
        if not save_path:
            raise ReplayFileError("No file path specified")

        # Serialize sections
        metadata_bytes = self._serialize_metadata()
        inputs_bytes = self._serialize_inputs()
        snapshots_bytes = self._serialize_snapshots()
        deltas_bytes = self._serialize_deltas()

        # Compress if requested
        if format == ReplayFileFormat.COMPRESSED:
            inputs_bytes = self._compress(inputs_bytes)
            snapshots_bytes = self._compress(snapshots_bytes)
            deltas_bytes = self._compress(deltas_bytes)

        # Calculate offsets
        header_size = ReplayHeader.header_size()
        metadata_offset = header_size
        inputs_offset = metadata_offset + len(metadata_bytes)
        snapshots_offset = inputs_offset + len(inputs_bytes)
        deltas_offset = snapshots_offset + len(snapshots_bytes)
        total_size = deltas_offset + len(deltas_bytes)

        # Build header
        self._header = ReplayHeader(
            format=format,
            compression=self._compression,
            metadata_offset=metadata_offset,
            inputs_offset=inputs_offset,
            snapshots_offset=snapshots_offset,
            deltas_offset=deltas_offset,
            metadata_size=len(metadata_bytes),
            inputs_size=len(inputs_bytes),
            snapshots_size=len(snapshots_bytes),
            deltas_size=len(deltas_bytes),
            file_size=total_size
        )

        # Calculate header checksum
        all_data = metadata_bytes + inputs_bytes + snapshots_bytes + deltas_bytes
        self._header.header_checksum = hashlib.sha256(all_data).hexdigest()

        # Write file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(self._header.serialize())
            f.write(metadata_bytes)
            f.write(inputs_bytes)
            f.write(snapshots_bytes)
            f.write(deltas_bytes)

        self._path = save_path
        return total_size

    def load(self, path: Optional[str | Path] = None) -> None:
        """Load replay from file.

        Args:
            path: File path (uses stored path if not provided)

        Raises:
            ReplayFileError: If load fails
        """
        load_path = Path(path) if path else self._path
        if not load_path:
            raise ReplayFileError("No file path specified")

        if not load_path.exists():
            raise ReplayFileError(f"File not found: {load_path}")

        with open(load_path, 'rb') as f:
            # Read header
            header_data = f.read(ReplayHeader.header_size())
            self._header = ReplayHeader.deserialize(header_data)

            # Read metadata
            f.seek(self._header.metadata_offset)
            metadata_bytes = f.read(self._header.metadata_size)
            self._metadata = self._deserialize_metadata(metadata_bytes)

            # Read inputs
            f.seek(self._header.inputs_offset)
            inputs_bytes = f.read(self._header.inputs_size)
            if self._header.format == ReplayFileFormat.COMPRESSED:
                inputs_bytes = self._decompress(inputs_bytes)
            self._inputs = self._deserialize_inputs(inputs_bytes)

            # Read snapshots
            f.seek(self._header.snapshots_offset)
            snapshots_bytes = f.read(self._header.snapshots_size)
            if self._header.format == ReplayFileFormat.COMPRESSED:
                snapshots_bytes = self._decompress(snapshots_bytes)
            self._snapshots = self._deserialize_snapshots(snapshots_bytes)

            # Read deltas
            f.seek(self._header.deltas_offset)
            deltas_bytes = f.read(self._header.deltas_size)
            if self._header.format == ReplayFileFormat.COMPRESSED:
                deltas_bytes = self._decompress(deltas_bytes)
            self._deltas = self._deserialize_deltas(deltas_bytes)

        self._path = load_path
        self._is_loaded = True
        self._compression = self._header.compression

    def load_metadata_only(self, path: Optional[str | Path] = None) -> ReplayMetadata:
        """Load only metadata from file (faster for browsing).

        Args:
            path: File path

        Returns:
            Replay metadata
        """
        load_path = Path(path) if path else self._path
        if not load_path:
            raise ReplayFileError("No file path specified")

        with open(load_path, 'rb') as f:
            # Read header
            header_data = f.read(ReplayHeader.header_size())
            header = ReplayHeader.deserialize(header_data)

            # Read metadata only
            f.seek(header.metadata_offset)
            metadata_bytes = f.read(header.metadata_size)
            return self._deserialize_metadata(metadata_bytes)

    def verify_integrity(self) -> bool:
        """Verify file integrity using checksum.

        Returns:
            True if file is valid
        """
        if not self._is_loaded:
            return False

        # Re-serialize and compute checksum
        metadata_bytes = self._serialize_metadata()
        inputs_bytes = self._serialize_inputs()
        snapshots_bytes = self._serialize_snapshots()
        deltas_bytes = self._serialize_deltas()

        if self._header.format == ReplayFileFormat.COMPRESSED:
            inputs_bytes = self._compress(inputs_bytes)
            snapshots_bytes = self._compress(snapshots_bytes)
            deltas_bytes = self._compress(deltas_bytes)

        all_data = metadata_bytes + inputs_bytes + snapshots_bytes + deltas_bytes
        computed_checksum = hashlib.sha256(all_data).hexdigest()

        return computed_checksum == self._header.header_checksum

    def export_json(self, path: str | Path) -> None:
        """Export replay to JSON format (for debugging).

        Args:
            path: Output file path
        """
        data = {
            'metadata': self._metadata.to_dict(),
            'inputs': [
                {
                    'type': inp.input_type.name,
                    'timestamp': inp.timestamp,
                    'frame': inp.frame,
                    'device_id': inp.device_id,
                    'data': inp.data
                }
                for inp in self._inputs
            ],
            'snapshots': [
                {
                    'frame': snap.frame,
                    'timestamp': snap.timestamp,
                    'is_keyframe': snap.is_keyframe,
                    'state': snap.state_data,
                    'metadata': snap.metadata
                }
                for snap in self._snapshots
            ],
            'deltas': [
                {
                    'from_frame': delta.from_frame,
                    'to_frame': delta.to_frame,
                    'timestamp': delta.timestamp,
                    'changes': delta.changes
                }
                for delta in self._deltas
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def from_recorders(
        cls,
        input_recorder: InputRecorder,
        state_recorder: StateRecorder,
        metadata: Optional[ReplayMetadata] = None
    ) -> 'ReplayFile':
        """Create replay file from recorders.

        Args:
            input_recorder: Input recorder with data
            state_recorder: State recorder with data
            metadata: Optional metadata

        Returns:
            ReplayFile instance with data
        """
        replay = cls()
        inputs = list(input_recorder.iter_inputs())
        snapshots = list(state_recorder.iter_snapshots())
        deltas = list(state_recorder.iter_deltas())
        replay.set_data(inputs, snapshots, deltas, metadata)

        # Set input hash for determinism verification
        replay.metadata.input_hash = input_recorder.input_hash

        return replay

    def get_file_size(self) -> int:
        """Get file size in bytes.

        Returns:
            File size, or 0 if not saved
        """
        if self._path and self._path.exists():
            return self._path.stat().st_size
        return self._header.file_size

    def _serialize_metadata(self) -> bytes:
        """Serialize metadata to bytes."""
        return json.dumps(self._metadata.to_dict()).encode('utf-8')

    def _deserialize_metadata(self, data: bytes) -> ReplayMetadata:
        """Deserialize metadata from bytes."""
        return ReplayMetadata.from_dict(json.loads(data.decode('utf-8')))

    def _serialize_inputs(self) -> bytes:
        """Serialize inputs to bytes."""
        parts = []
        for inp in self._inputs:
            parts.append(inp.serialize())
        count = struct.pack('<I', len(self._inputs))
        return count + b''.join(parts)

    def _deserialize_inputs(self, data: bytes) -> list[RecordedInput]:
        """Deserialize inputs from bytes."""
        if not data:
            return []
        count = struct.unpack('<I', data[:4])[0]
        offset = 4
        inputs = []
        for _ in range(count):
            inp, offset = RecordedInput.deserialize(data, offset)
            inputs.append(inp)
        return inputs

    def _serialize_snapshots(self) -> bytes:
        """Serialize snapshots to bytes."""
        parts = []
        for snap in self._snapshots:
            parts.append(snap.serialize(self._compression))
        count = struct.pack('<I', len(self._snapshots))
        return count + b''.join(parts)

    def _deserialize_snapshots(self, data: bytes) -> list[StateSnapshot]:
        """Deserialize snapshots from bytes."""
        if not data:
            return []
        count = struct.unpack('<I', data[:4])[0]
        offset = 4
        snapshots = []
        for _ in range(count):
            snap, offset = StateSnapshot.deserialize(data, offset)
            snapshots.append(snap)
        return snapshots

    def _serialize_deltas(self) -> bytes:
        """Serialize deltas to bytes."""
        parts = []
        for delta in self._deltas:
            parts.append(delta.serialize())
        count = struct.pack('<I', len(self._deltas))
        return count + b''.join(parts)

    def _deserialize_deltas(self, data: bytes) -> list[StateDelta]:
        """Deserialize deltas from bytes."""
        if not data:
            return []
        count = struct.unpack('<I', data[:4])[0]
        offset = 4
        deltas = []
        for _ in range(count):
            delta, offset = StateDelta.deserialize(data, offset)
            deltas.append(delta)
        return deltas

    def _compress(self, data: bytes) -> bytes:
        """Compress data using configured method."""
        if self._compression == CompressionMethod.NONE:
            return data
        elif self._compression == CompressionMethod.ZLIB:
            return zlib.compress(data, level=6)
        elif self._compression == CompressionMethod.ZLIB_FAST:
            return zlib.compress(data, level=1)
        elif self._compression == CompressionMethod.ZLIB_BEST:
            return zlib.compress(data, level=9)
        return zlib.compress(data)

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data using configured method."""
        if self._compression == CompressionMethod.NONE:
            return data
        return zlib.decompress(data)
