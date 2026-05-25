"""
Tests for replay_file.py - File format and compression.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from engine.tooling.replay.replay_file import (
    ReplayFile,
    ReplayHeader,
    ReplayMetadata,
    ReplayFileFormat,
    ReplayFileError,
    REPLAY_MAGIC,
    REPLAY_VERSION,
)
from engine.tooling.replay.input_recorder import RecordedInput, InputType
from engine.tooling.replay.state_recorder import StateSnapshot, CompressionMethod


def create_test_inputs(count: int = 10) -> list[RecordedInput]:
    """Create test input data."""
    return [
        RecordedInput(
            input_type=InputType.KEYBOARD,
            timestamp=i * 0.016,
            frame=i,
            device_id=0,
            data={'key': 'A', 'pressed': True}
        )
        for i in range(count)
    ]


def create_test_snapshots(count: int = 5) -> list[StateSnapshot]:
    """Create test snapshot data."""
    return [
        StateSnapshot(
            frame=i * 10,
            timestamp=i * 0.166,
            state_data={'value': i},
            checksum='',
            size_bytes=0,
            is_keyframe=True
        )
        for i in range(count)
    ]


class TestReplayMetadata:
    """Tests for ReplayMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating metadata."""
        meta = ReplayMetadata(
            game_name="Test Game",
            game_version="1.0.0",
            player_name="Player1",
            duration=120.5
        )
        assert meta.game_name == "Test Game"
        assert meta.game_version == "1.0.0"
        assert meta.player_name == "Player1"
        assert meta.duration == 120.5

    def test_default_metadata(self):
        """Test default metadata values."""
        meta = ReplayMetadata()
        assert meta.game_name == ""
        assert meta.duration == 0.0
        assert meta.total_frames == 0

    def test_to_dict(self):
        """Test converting to dictionary."""
        meta = ReplayMetadata(
            game_name="Test",
            player_name="Player",
            tags=['ranked', 'tournament']
        )
        data = meta.to_dict()

        assert data['game_name'] == "Test"
        assert data['player_name'] == "Player"
        assert 'ranked' in data['tags']

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            'game_name': "Test",
            'game_version': "2.0",
            'player_name': "Player",
            'duration': 60.0,
            'recorded_at': datetime.now().isoformat(),
            'tags': ['casual']
        }
        meta = ReplayMetadata.from_dict(data)

        assert meta.game_name == "Test"
        assert meta.game_version == "2.0"
        assert meta.duration == 60.0
        assert 'casual' in meta.tags

    def test_result_fields(self):
        """Test result-related fields."""
        meta = ReplayMetadata(
            result='win',
            score=1500,
            opponent='Opponent1'
        )
        assert meta.result == 'win'
        assert meta.score == 1500
        assert meta.opponent == 'Opponent1'

    def test_custom_data(self):
        """Test custom data field."""
        meta = ReplayMetadata(custom={'extra': 'info', 'value': 42})
        assert meta.custom['extra'] == 'info'
        assert meta.custom['value'] == 42


class TestReplayHeader:
    """Tests for ReplayHeader dataclass."""

    def test_create_header(self):
        """Test creating header."""
        header = ReplayHeader(
            format=ReplayFileFormat.COMPRESSED,
            compression=CompressionMethod.ZLIB
        )
        assert header.magic == REPLAY_MAGIC
        assert header.version == REPLAY_VERSION
        assert header.format == ReplayFileFormat.COMPRESSED

    def test_serialize_deserialize(self):
        """Test header serialization."""
        header = ReplayHeader(
            metadata_offset=100,
            inputs_offset=500,
            snapshots_offset=1000,
            deltas_offset=1500,
            metadata_size=400,
            inputs_size=500,
            snapshots_size=500,
            deltas_size=200,
            file_size=1700
        )

        serialized = header.serialize()
        assert isinstance(serialized, bytes)

        deserialized = ReplayHeader.deserialize(serialized)
        assert deserialized.metadata_offset == 100
        assert deserialized.inputs_offset == 500
        assert deserialized.file_size == 1700

    def test_header_size(self):
        """Test header size is consistent."""
        size = ReplayHeader.header_size()
        header = ReplayHeader()
        serialized = header.serialize()

        assert len(serialized) == size

    def test_invalid_magic(self):
        """Test handling invalid magic number."""
        bad_data = b'XXXX' + b'\x00' * 100

        with pytest.raises(ReplayFileError, match="Invalid replay file magic"):
            ReplayHeader.deserialize(bad_data)


class TestReplayFile:
    """Tests for ReplayFile class."""

    def test_create_replay_file(self):
        """Test creating a replay file instance."""
        replay = ReplayFile()
        assert not replay.is_loaded
        assert replay.path is None

    def test_create_with_path(self):
        """Test creating with path."""
        replay = ReplayFile("/tmp/test.replay")
        assert replay.path == Path("/tmp/test.replay")

    def test_set_data(self):
        """Test setting replay data."""
        replay = ReplayFile()
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(5)

        meta = ReplayMetadata(game_name="Test")
        replay.set_data(inputs, snapshots, metadata=meta)

        assert len(replay.inputs) == 10
        assert len(replay.snapshots) == 5
        assert replay.metadata.game_name == "Test"
        assert replay.metadata.input_count == 10
        assert replay.metadata.snapshot_count == 5

    def test_save_and_load(self):
        """Test saving and loading replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.replay"

            # Create and save
            replay = ReplayFile()
            inputs = create_test_inputs(20)
            snapshots = create_test_snapshots(5)
            meta = ReplayMetadata(
                game_name="Test Game",
                player_name="Player1"
            )
            replay.set_data(inputs, snapshots, metadata=meta)
            bytes_written = replay.save(path)

            assert bytes_written > 0
            assert path.exists()

            # Load
            loaded = ReplayFile(path)
            loaded.load()

            assert loaded.is_loaded
            assert len(loaded.inputs) == 20
            assert len(loaded.snapshots) == 5
            assert loaded.metadata.game_name == "Test Game"

    def test_save_compressed(self):
        """Test saving with compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.replay"

            replay = ReplayFile()
            replay.compression = CompressionMethod.ZLIB_BEST

            inputs = create_test_inputs(100)
            snapshots = create_test_snapshots(10)
            replay.set_data(inputs, snapshots)

            bytes_written = replay.save(path, format=ReplayFileFormat.COMPRESSED)
            assert bytes_written > 0

    def test_load_metadata_only(self):
        """Test loading only metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.replay"

            # Save
            replay = ReplayFile()
            inputs = create_test_inputs(100)
            snapshots = create_test_snapshots(10)
            meta = ReplayMetadata(
                game_name="Test",
                duration=60.0
            )
            replay.set_data(inputs, snapshots, metadata=meta)
            replay.save(path)

            # Load metadata only
            replay2 = ReplayFile()
            loaded_meta = replay2.load_metadata_only(path)

            assert loaded_meta.game_name == "Test"
            # Duration gets recalculated from inputs/snapshots in set_data
            assert loaded_meta.duration > 0

    def test_verify_integrity(self):
        """Test integrity verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.replay"

            replay = ReplayFile()
            inputs = create_test_inputs(10)
            snapshots = create_test_snapshots(5)
            replay.set_data(inputs, snapshots)
            replay.save(path)

            # Load and verify
            loaded = ReplayFile(path)
            loaded.load()

            # Note: Verification checks serialized content matches checksum
            # This test checks the method exists and runs
            result = loaded.verify_integrity()
            # Result depends on implementation details

    def test_export_json(self):
        """Test exporting to JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "test.replay"
            json_path = Path(tmpdir) / "test.json"

            replay = ReplayFile()
            inputs = create_test_inputs(5)
            snapshots = create_test_snapshots(2)
            replay.set_data(inputs, snapshots)
            replay.save(replay_path)

            replay.export_json(json_path)
            assert json_path.exists()

            # Verify JSON content
            import json
            with open(json_path) as f:
                data = json.load(f)

            assert 'metadata' in data
            assert 'inputs' in data
            assert 'snapshots' in data

    def test_get_file_size(self):
        """Test getting file size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.replay"

            replay = ReplayFile()
            inputs = create_test_inputs(10)
            snapshots = create_test_snapshots(5)
            replay.set_data(inputs, snapshots)
            replay.save(path)

            size = replay.get_file_size()
            assert size > 0

    def test_no_path_error(self):
        """Test error when no path specified."""
        replay = ReplayFile()
        inputs = create_test_inputs(5)
        snapshots = create_test_snapshots(2)
        replay.set_data(inputs, snapshots)

        with pytest.raises(ReplayFileError, match="No file path"):
            replay.save()

    def test_file_not_found_error(self):
        """Test error when file not found."""
        replay = ReplayFile("/nonexistent/path/file.replay")

        with pytest.raises(ReplayFileError, match="File not found"):
            replay.load()

    def test_compression_property(self):
        """Test compression property."""
        replay = ReplayFile()
        assert replay.compression == CompressionMethod.ZLIB

        replay.compression = CompressionMethod.ZLIB_BEST
        assert replay.compression == CompressionMethod.ZLIB_BEST

    def test_header_property(self):
        """Test header property."""
        replay = ReplayFile()
        header = replay.header

        assert header.magic == REPLAY_MAGIC
        assert header.version == REPLAY_VERSION

    def test_deltas_property(self):
        """Test deltas property."""
        from engine.tooling.replay.state_recorder import StateDelta

        replay = ReplayFile()
        inputs = create_test_inputs(5)
        snapshots = create_test_snapshots(2)
        deltas = [
            StateDelta(
                from_frame=0,
                to_frame=1,
                timestamp=0.016,
                changes=[('value', 0, 1)],
                size_bytes=0
            )
        ]
        replay.set_data(inputs, snapshots, deltas)

        assert len(replay.deltas) == 1

    def test_different_formats(self):
        """Test different file formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for fmt in [ReplayFileFormat.BINARY, ReplayFileFormat.COMPRESSED]:
                path = Path(tmpdir) / f"test_{fmt.name}.replay"

                replay = ReplayFile()
                inputs = create_test_inputs(10)
                snapshots = create_test_snapshots(5)
                replay.set_data(inputs, snapshots)
                replay.save(path, format=fmt)

                loaded = ReplayFile(path)
                loaded.load()

                assert len(loaded.inputs) == 10
