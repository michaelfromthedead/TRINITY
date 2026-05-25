"""
Tests for state_recorder.py - State snapshots and delta compression.
"""

import pytest

from engine.tooling.replay.state_recorder import (
    StateRecorder,
    StateSnapshot,
    StateDelta,
    StateRecordingConfig,
    CompressionMethod,
)


class TestStateSnapshot:
    """Tests for StateSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a state snapshot."""
        snapshot = StateSnapshot(
            frame=100,
            timestamp=1.67,
            state_data={'player': {'x': 10.0, 'y': 20.0}},
            checksum='abc123',
            size_bytes=256,
            is_keyframe=True
        )
        assert snapshot.frame == 100
        assert snapshot.timestamp == 1.67
        assert snapshot.is_keyframe is True
        assert snapshot.state_data['player']['x'] == 10.0

    def test_serialize_deserialize(self):
        """Test snapshot serialization."""
        snapshot = StateSnapshot(
            frame=50,
            timestamp=0.83,
            state_data={
                'player': {'position': [1.0, 2.0, 3.0], 'health': 100},
                'score': 500
            },
            checksum='def456',
            size_bytes=0,
            is_keyframe=True,
            metadata={'level': 'test'}
        )

        serialized = snapshot.serialize(CompressionMethod.ZLIB)
        assert isinstance(serialized, bytes)

        deserialized, offset = StateSnapshot.deserialize(serialized)
        assert deserialized.frame == 50
        assert deserialized.timestamp == 0.83
        assert deserialized.state_data['player']['health'] == 100
        assert deserialized.metadata['level'] == 'test'

    def test_compute_checksum(self):
        """Test checksum computation."""
        snapshot = StateSnapshot(
            frame=0,
            timestamp=0.0,
            state_data={'value': 42},
            checksum='',
            size_bytes=0
        )
        checksum = snapshot.compute_checksum()
        assert len(checksum) == 64  # SHA-256 hex

    def test_verify_checksum(self):
        """Test checksum verification."""
        state_data = {'value': 42}
        snapshot = StateSnapshot(
            frame=0,
            timestamp=0.0,
            state_data=state_data,
            checksum='',
            size_bytes=0
        )
        snapshot = StateSnapshot(
            frame=0,
            timestamp=0.0,
            state_data=state_data,
            checksum=snapshot.compute_checksum(),
            size_bytes=0
        )
        assert snapshot.verify_checksum()


class TestStateDelta:
    """Tests for StateDelta dataclass."""

    def test_create_delta(self):
        """Test creating a state delta."""
        delta = StateDelta(
            from_frame=10,
            to_frame=11,
            timestamp=0.183,
            changes=[
                ('player.x', 10.0, 11.0),
                ('player.y', 20.0, 22.0),
            ],
            size_bytes=64
        )
        assert delta.from_frame == 10
        assert delta.to_frame == 11
        assert len(delta.changes) == 2

    def test_apply_delta(self):
        """Test applying delta to base state."""
        base_state = {
            'player': {'x': 10.0, 'y': 20.0},
            'score': 100
        }
        delta = StateDelta(
            from_frame=0,
            to_frame=1,
            timestamp=0.0,
            changes=[
                ('player.x', 10.0, 15.0),
                ('score', 100, 150),
            ],
            size_bytes=0
        )

        new_state = delta.apply(base_state)
        assert new_state['player']['x'] == 15.0
        assert new_state['score'] == 150
        assert new_state['player']['y'] == 20.0  # Unchanged

    def test_reverse_delta(self):
        """Test reversing delta to get previous state."""
        current_state = {
            'player': {'x': 15.0, 'y': 20.0},
            'score': 150
        }
        delta = StateDelta(
            from_frame=0,
            to_frame=1,
            timestamp=0.0,
            changes=[
                ('player.x', 10.0, 15.0),
                ('score', 100, 150),
            ],
            size_bytes=0
        )

        prev_state = delta.reverse(current_state)
        assert prev_state['player']['x'] == 10.0
        assert prev_state['score'] == 100

    def test_serialize_deserialize_delta(self):
        """Test delta serialization."""
        delta = StateDelta(
            from_frame=5,
            to_frame=6,
            timestamp=0.1,
            changes=[('value', 1, 2)],
            size_bytes=0
        )

        serialized = delta.serialize()
        deserialized, offset = StateDelta.deserialize(serialized)

        assert deserialized.from_frame == 5
        assert deserialized.to_frame == 6
        assert len(deserialized.changes) == 1


class TestStateRecordingConfig:
    """Tests for StateRecordingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = StateRecordingConfig()
        assert config.keyframe_interval == 60
        assert config.delta_interval == 1
        assert config.compression == CompressionMethod.ZLIB
        assert config.enable_delta_compression is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = StateRecordingConfig(
            keyframe_interval=120,
            compression=CompressionMethod.ZLIB_BEST
        )
        assert config.keyframe_interval == 120
        assert config.compression == CompressionMethod.ZLIB_BEST


class TestStateRecorder:
    """Tests for StateRecorder class."""

    def test_create_recorder(self):
        """Test creating a state recorder."""
        recorder = StateRecorder()
        assert not recorder.is_recording
        assert recorder.snapshot_count == 0
        assert recorder.delta_count == 0

    def test_start_stop_recording(self):
        """Test starting and stopping recording."""
        recorder = StateRecorder()
        recorder.start()
        assert recorder.is_recording

        snapshots, deltas = recorder.stop()
        assert not recorder.is_recording
        assert isinstance(snapshots, list)
        assert isinstance(deltas, list)

    def test_record_first_state_creates_keyframe(self):
        """Test that first state creates keyframe."""
        recorder = StateRecorder()
        recorder.start()

        result = recorder.record_state(
            state={'value': 1},
            timestamp=0.0
        )

        assert result is not None
        assert isinstance(result, StateSnapshot)
        assert result.is_keyframe is True

        recorder.stop()

    def test_record_state_at_interval_creates_keyframe(self):
        """Test keyframe creation at intervals."""
        config = StateRecordingConfig(keyframe_interval=5)
        recorder = StateRecorder(config)
        recorder.start()

        # Record initial keyframe
        recorder.record_state({'value': 0}, timestamp=0.0)

        # Record deltas
        for i in range(1, 5):
            recorder.record_state({'value': i}, timestamp=i * 0.016)

        # This should create a new keyframe (frame 5)
        result = recorder.record_state({'value': 5}, timestamp=5 * 0.016)
        assert isinstance(result, StateSnapshot)
        assert result.is_keyframe is True

        recorder.stop()

    def test_force_keyframe(self):
        """Test forcing a keyframe."""
        recorder = StateRecorder()
        recorder.start()

        recorder.record_state({'value': 0}, timestamp=0.0)
        recorder.record_state({'value': 1}, timestamp=0.016)

        result = recorder.record_state(
            {'value': 2},
            timestamp=0.032,
            force_keyframe=True
        )

        assert isinstance(result, StateSnapshot)
        assert result.is_keyframe is True

        recorder.stop()

    def test_record_state_with_metadata(self):
        """Test recording state with metadata."""
        recorder = StateRecorder()
        recorder.start()

        result = recorder.record_state(
            state={'value': 1},
            timestamp=0.0,
            metadata={'checkpoint': 'start'}
        )

        assert result.metadata['checkpoint'] == 'start'
        recorder.stop()

    def test_get_state_at_frame(self):
        """Test retrieving state at specific frame."""
        recorder = StateRecorder()
        recorder.start()

        recorder.record_state({'x': 0}, timestamp=0.0)
        recorder.record_state({'x': 10}, timestamp=0.016)
        recorder.record_state({'x': 20}, timestamp=0.032)

        state = recorder.get_state_at_frame(0)
        assert state is not None
        assert state['x'] == 0

        recorder.stop()

    def test_get_nearest_keyframe(self):
        """Test getting nearest keyframe."""
        config = StateRecordingConfig(keyframe_interval=3)
        recorder = StateRecorder(config)
        recorder.start()

        # Record frames 0-5
        for i in range(6):
            recorder.record_state({'frame': i}, timestamp=i * 0.016)

        # Frame 3 should be a keyframe
        keyframe = recorder.get_nearest_keyframe(4)
        assert keyframe is not None
        assert keyframe.frame == 3

        recorder.stop()

    def test_get_snapshots_in_range(self):
        """Test getting snapshots in frame range."""
        config = StateRecordingConfig(keyframe_interval=2)
        recorder = StateRecorder(config)
        recorder.start()

        for i in range(6):
            recorder.record_state({'frame': i}, timestamp=i * 0.016)

        snapshots = recorder.get_snapshots_in_range(1, 4)
        assert len(snapshots) > 0
        for s in snapshots:
            assert 1 <= s.frame <= 4

        recorder.stop()

    def test_delta_compression(self):
        """Test delta compression between states."""
        config = StateRecordingConfig(
            keyframe_interval=100,
            enable_delta_compression=True
        )
        recorder = StateRecorder(config)
        recorder.start()

        recorder.record_state({'x': 0, 'y': 0}, timestamp=0.0)
        result = recorder.record_state({'x': 1, 'y': 0}, timestamp=0.016)

        # Should create a delta (not keyframe)
        assert isinstance(result, StateDelta)
        assert result is not None

        recorder.stop()

    def test_state_filter(self):
        """Test state filtering via config."""
        def filter_func(state):
            # Remove 'internal' key
            return {k: v for k, v in state.items() if k != 'internal'}

        config = StateRecordingConfig(state_filter=filter_func)
        recorder = StateRecorder(config)
        recorder.start()

        result = recorder.record_state(
            {'value': 1, 'internal': 'secret'},
            timestamp=0.0
        )

        assert 'internal' not in result.state_data
        assert result.state_data['value'] == 1

        recorder.stop()

    def test_excluded_paths(self):
        """Test path exclusion."""
        config = StateRecordingConfig(excluded_paths={'debug', 'temp'})
        recorder = StateRecorder(config)
        recorder.start()

        result = recorder.record_state(
            {'value': 1, 'debug': 'info', 'temp': 'data'},
            timestamp=0.0
        )

        assert 'debug' not in result.state_data
        assert 'temp' not in result.state_data
        assert result.state_data['value'] == 1

        recorder.stop()

    def test_clear(self):
        """Test clearing recorded data."""
        recorder = StateRecorder()
        recorder.start()

        recorder.record_state({'value': 1}, timestamp=0.0)
        recorder.record_state({'value': 2}, timestamp=0.016)

        assert recorder.snapshot_count > 0

        recorder.clear()
        assert recorder.snapshot_count == 0
        assert recorder.delta_count == 0
        assert recorder.current_frame == 0

        recorder.stop()

    def test_iter_snapshots(self):
        """Test iterating over snapshots."""
        recorder = StateRecorder()
        recorder.start()

        recorder.record_state({'frame': 0}, timestamp=0.0)
        recorder.record_state({'frame': 1}, timestamp=0.016, force_keyframe=True)

        snapshots = list(recorder.iter_snapshots())
        assert len(snapshots) == 2

        recorder.stop()

    def test_serialize_deserialize(self):
        """Test serialization and deserialization."""
        recorder = StateRecorder()
        recorder.start()

        recorder.record_state({'x': 10, 'y': 20}, timestamp=0.0)
        recorder.record_state({'x': 15, 'y': 25}, timestamp=0.016)

        serialized = recorder.serialize()
        assert isinstance(serialized, bytes)

        new_recorder = StateRecorder.deserialize(serialized)
        assert new_recorder.snapshot_count == recorder.snapshot_count

        recorder.stop()

    def test_stats_tracking(self):
        """Test statistics tracking."""
        recorder = StateRecorder()
        recorder.start()

        recorder.record_state({'value': 1}, timestamp=0.0)
        recorder.record_state({'value': 2}, timestamp=0.016)

        stats = recorder.stats
        assert stats['total_snapshots'] > 0
        assert 'total_bytes' in stats

        recorder.stop()

    def test_compression_methods(self):
        """Test different compression methods."""
        for method in [CompressionMethod.NONE, CompressionMethod.ZLIB,
                      CompressionMethod.ZLIB_FAST, CompressionMethod.ZLIB_BEST]:
            config = StateRecordingConfig(compression=method)
            recorder = StateRecorder(config)
            recorder.start()

            recorder.record_state({'large_data': 'x' * 1000}, timestamp=0.0)
            snapshots, _ = recorder.stop()

            assert len(snapshots) > 0
