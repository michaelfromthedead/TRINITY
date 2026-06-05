"""
Whitebox tests for music_timing.py - Musical timing system.
"""

import pytest
import time
import threading
import math
from engine.audio.adaptive.music_timing import (
    BeatSubdivision,
    TimeSignature,
    SyncPoint,
    BeatInfo,
    BeatGrid,
    MusicClock,
    SyncPointManager,
)
from engine.audio.adaptive.config import (
    DEFAULT_BPM,
    MIN_BPM,
    MAX_BPM,
    DEFAULT_TIME_SIGNATURE,
    GRID_SUBDIVISIONS,
)


class TestTimeSignature:
    """Tests for TimeSignature dataclass."""

    def test_create_time_signature_4_4(self):
        """Create standard 4/4 time signature."""
        ts = TimeSignature(4, 4)
        assert ts.beats_per_bar == 4
        assert ts.beat_unit == 4

    def test_create_time_signature_3_4(self):
        """Create 3/4 waltz time signature."""
        ts = TimeSignature(3, 4)
        assert ts.beats_per_bar == 3
        assert ts.beat_unit == 4

    def test_create_time_signature_6_8(self):
        """Create 6/8 compound time signature."""
        ts = TimeSignature(6, 8)
        assert ts.beats_per_bar == 6
        assert ts.beat_unit == 8

    def test_invalid_beats_per_bar(self):
        """Reject zero or negative beats_per_bar."""
        with pytest.raises(ValueError, match="beats_per_bar must be positive"):
            TimeSignature(0, 4)

    def test_invalid_beat_unit_not_power_of_2(self):
        """Reject beat_unit that is not a power of 2."""
        with pytest.raises(ValueError, match="beat_unit must be a power of 2"):
            TimeSignature(4, 3)

    def test_from_tuple(self):
        """Create TimeSignature from tuple."""
        ts = TimeSignature.from_tuple((3, 4))
        assert ts.beats_per_bar == 3
        assert ts.beat_unit == 4

    def test_to_tuple(self):
        """Convert TimeSignature to tuple."""
        ts = TimeSignature(6, 8)
        assert ts.to_tuple() == (6, 8)

    def test_time_signature_frozen(self):
        """TimeSignature should be immutable (frozen)."""
        ts = TimeSignature(4, 4)
        with pytest.raises(Exception):
            ts.beats_per_bar = 3


class TestSyncPoint:
    """Tests for SyncPoint dataclass."""

    def test_create_sync_point(self):
        """Create a sync point."""
        sp = SyncPoint(name="intro_end", beat=8.0, bar=2, time_ms=4000.0)
        assert sp.name == "intro_end"
        assert sp.beat == 8.0
        assert sp.bar == 2
        assert sp.time_ms == 4000.0
        assert sp.metadata == {}

    def test_sync_point_with_metadata(self):
        """Create sync point with metadata."""
        sp = SyncPoint(
            name="boss_trigger",
            beat=16.0,
            bar=4,
            time_ms=8000.0,
            metadata={"boss_id": "dragon", "intensity": 0.9},
        )
        assert sp.metadata["boss_id"] == "dragon"
        assert sp.metadata["intensity"] == 0.9

    def test_sync_point_equality(self):
        """Sync points with same name and bar are equal."""
        sp1 = SyncPoint("marker", 4.0, 1, 2000.0)
        sp2 = SyncPoint("marker", 4.0, 1, 2000.0)
        sp3 = SyncPoint("marker", 4.0, 2, 4000.0)
        assert sp1 == sp2
        assert sp1 != sp3

    def test_sync_point_hashable(self):
        """Sync points should be hashable for set/dict usage."""
        sp = SyncPoint("test", 4.0, 1, 2000.0)
        hash_val = hash(sp)
        assert isinstance(hash_val, int)


class TestBeatInfo:
    """Tests for BeatInfo dataclass."""

    def test_create_beat_info(self):
        """Create BeatInfo instance."""
        bi = BeatInfo(
            beat_in_bar=2,
            bar=3,
            total_beats=14,
            subdivision=8,
            time_ms=7000.0,
            progress_in_beat=0.5,
            progress_in_bar=0.75,
        )
        assert bi.beat_in_bar == 2
        assert bi.bar == 3
        assert bi.total_beats == 14
        assert bi.subdivision == 8
        assert bi.time_ms == 7000.0
        assert bi.progress_in_beat == 0.5
        assert bi.progress_in_bar == 0.75


class TestBeatGrid:
    """Tests for BeatGrid class."""

    def test_create_beat_grid_defaults(self):
        """Create beat grid with default values."""
        grid = BeatGrid()
        assert grid.bpm == DEFAULT_BPM
        assert grid.time_signature == TimeSignature.from_tuple(DEFAULT_TIME_SIGNATURE)

    def test_create_beat_grid_custom_bpm(self):
        """Create beat grid with custom BPM."""
        grid = BeatGrid(bpm=140.0)
        assert grid.bpm == 140.0

    def test_bpm_too_low(self):
        """Reject BPM below minimum."""
        with pytest.raises(ValueError, match="BPM must be between"):
            BeatGrid(bpm=20.0)

    def test_bpm_too_high(self):
        """Reject BPM above maximum."""
        with pytest.raises(ValueError, match="BPM must be between"):
            BeatGrid(bpm=400.0)

    def test_set_bpm(self):
        """Change BPM after creation."""
        grid = BeatGrid()
        grid.bpm = 180.0
        assert grid.bpm == 180.0

    def test_set_bpm_invalid(self):
        """Reject invalid BPM when setting."""
        grid = BeatGrid()
        with pytest.raises(ValueError):
            grid.bpm = 10.0

    def test_beat_duration_ms(self):
        """Beat duration calculated correctly."""
        grid = BeatGrid(bpm=120.0)
        # At 120 BPM, one beat = 500ms
        assert grid.beat_duration_ms == 500.0

    def test_bar_duration_ms(self):
        """Bar duration calculated correctly for 4/4."""
        grid = BeatGrid(bpm=120.0, time_signature=TimeSignature(4, 4))
        # At 120 BPM, one bar = 4 * 500ms = 2000ms
        assert grid.bar_duration_ms == 2000.0

    def test_subdivision_duration_ms(self):
        """Subdivision duration calculated correctly."""
        grid = BeatGrid(bpm=120.0, subdivisions=16)
        # At 120 BPM, one subdivision = 500ms / 16 = 31.25ms
        assert grid.subdivision_duration_ms == pytest.approx(31.25)

    def test_time_to_beat(self):
        """Convert time to beat position."""
        grid = BeatGrid(bpm=120.0)
        assert grid.time_to_beat(500.0) == 1.0
        assert grid.time_to_beat(1000.0) == 2.0
        assert grid.time_to_beat(250.0) == 0.5

    def test_beat_to_time(self):
        """Convert beat position to time."""
        grid = BeatGrid(bpm=120.0)
        assert grid.beat_to_time(1.0) == 500.0
        assert grid.beat_to_time(4.0) == 2000.0

    def test_time_to_bar(self):
        """Convert time to bar and beat within bar."""
        grid = BeatGrid(bpm=120.0, time_signature=TimeSignature(4, 4))
        bar, beat = grid.time_to_bar(2500.0)
        assert bar == 1
        assert beat == pytest.approx(1.0)

    def test_bar_to_time(self):
        """Convert bar and beat to time."""
        grid = BeatGrid(bpm=120.0, time_signature=TimeSignature(4, 4))
        assert grid.bar_to_time(2) == 4000.0
        assert grid.bar_to_time(1, 2.0) == 3000.0

    def test_quantize_to_beat(self):
        """Quantize time to nearest beat."""
        grid = BeatGrid(bpm=120.0)
        assert grid.quantize_to_beat(480.0) == 500.0  # Rounds to beat 1
        assert grid.quantize_to_beat(520.0) == 500.0  # Rounds to beat 1
        assert grid.quantize_to_beat(780.0) == 1000.0  # Rounds to beat 2

    def test_quantize_to_bar(self):
        """Quantize time to nearest bar."""
        grid = BeatGrid(bpm=120.0, time_signature=TimeSignature(4, 4))
        assert grid.quantize_to_bar(500.0) == 0.0  # Before midpoint, rounds down
        assert grid.quantize_to_bar(1500.0) == 2000.0  # After midpoint, rounds up

    def test_quantize_to_subdivision(self):
        """Quantize time to nearest subdivision."""
        grid = BeatGrid(bpm=120.0, subdivisions=16)
        # Subdivision = 31.25ms
        assert grid.quantize_to_subdivision(30.0) == pytest.approx(31.25)
        assert grid.quantize_to_subdivision(60.0) == pytest.approx(62.5)

    def test_next_beat(self):
        """Get time of next beat."""
        grid = BeatGrid(bpm=120.0)
        assert grid.next_beat(100.0) == 500.0
        assert grid.next_beat(500.0) == 1000.0  # Exactly on beat, next is +1
        assert grid.next_beat(600.0) == 1000.0

    def test_next_bar(self):
        """Get time of next bar."""
        grid = BeatGrid(bpm=120.0, time_signature=TimeSignature(4, 4))
        assert grid.next_bar(100.0) == 2000.0
        assert grid.next_bar(2000.0) == 4000.0  # Exactly on bar, next bar

    def test_get_beat_info(self):
        """Get detailed beat information."""
        grid = BeatGrid(bpm=120.0, time_signature=TimeSignature(4, 4), subdivisions=16)
        info = grid.get_beat_info(2750.0)  # In bar 1, beat 1, past halfway
        assert info.bar == 1
        assert info.beat_in_bar == 1
        assert info.total_beats == 5  # 4 beats in bar 0, beat 1 in bar 1
        assert 0 <= info.progress_in_beat <= 1
        assert 0 <= info.progress_in_bar <= 1


class TestMusicClock:
    """Tests for MusicClock class."""

    def test_create_music_clock_defaults(self):
        """Create music clock with default values."""
        clock = MusicClock()
        assert clock.bpm == DEFAULT_BPM
        assert clock.is_running is False

    def test_create_music_clock_custom(self):
        """Create music clock with custom values."""
        clock = MusicClock(bpm=180.0, time_signature=TimeSignature(3, 4))
        assert clock.bpm == 180.0
        assert clock.time_signature.beats_per_bar == 3

    def test_start_stop_clock(self):
        """Start and stop the music clock."""
        clock = MusicClock()
        assert clock.is_running is False
        clock.start()
        assert clock.is_running is True
        clock.stop()
        assert clock.is_running is False

    def test_pause_resume_clock(self):
        """Pause and resume the music clock."""
        clock = MusicClock()
        clock.start()
        time.sleep(0.05)
        clock.pause()
        assert clock.is_running is False
        pause_time = clock.get_time_ms()
        time.sleep(0.05)
        # Time should not advance while paused
        assert clock.get_time_ms() == pytest.approx(pause_time, abs=5)
        clock.resume()
        assert clock.is_running is True

    def test_get_time_ms(self):
        """Get current time advances while running."""
        clock = MusicClock()
        clock.start()
        time.sleep(0.05)
        elapsed = clock.get_time_ms()
        assert elapsed > 0
        clock.stop()

    def test_seek(self):
        """Seek to specific time."""
        clock = MusicClock()
        clock.start()
        clock.seek(5000.0)
        assert clock.get_time_ms() == pytest.approx(5000.0, abs=50)
        clock.stop()

    def test_seek_to_bar(self):
        """Seek to specific bar and beat."""
        clock = MusicClock(bpm=120.0)
        clock.start()
        clock.seek_to_bar(2, 1.0)
        # Bar 2, beat 1 = 4 beats * 500ms + 500ms = 4500ms
        assert clock.get_time_ms() == pytest.approx(4500.0, abs=50)
        clock.stop()

    def test_get_beat_info(self):
        """Get beat info from clock."""
        clock = MusicClock(bpm=120.0)
        clock.start()
        clock.seek(3000.0)
        info = clock.get_beat_info()
        assert info.bar == 1
        assert info.beat_in_bar == 2
        clock.stop()

    def test_get_current_beat(self):
        """Get current beat position."""
        clock = MusicClock(bpm=120.0)
        clock.start()
        clock.seek(1500.0)
        assert clock.get_current_beat() == pytest.approx(3.0, abs=0.1)
        clock.stop()

    def test_get_current_bar(self):
        """Get current bar number."""
        clock = MusicClock(bpm=120.0, time_signature=TimeSignature(4, 4))
        clock.start()
        clock.seek(5000.0)  # After bar 2
        assert clock.get_current_bar() == 2
        clock.stop()

    def test_time_until_next_beat(self):
        """Get time until next beat."""
        clock = MusicClock(bpm=120.0)
        clock.start()
        clock.seek(400.0)
        # Next beat at 500ms, so ~100ms remaining
        remaining = clock.time_until_next_beat()
        assert 50 <= remaining <= 150
        clock.stop()

    def test_time_until_next_bar(self):
        """Get time until next bar."""
        clock = MusicClock(bpm=120.0, time_signature=TimeSignature(4, 4))
        clock.start()
        clock.seek(500.0)  # Beat 1 of bar 0
        remaining = clock.time_until_next_bar()
        # Next bar at 2000ms, so ~1500ms remaining
        assert 1400 <= remaining <= 1600
        clock.stop()

    def test_is_on_beat(self):
        """Check if on beat boundary."""
        clock = MusicClock(bpm=120.0)
        clock.start()
        clock.seek(500.0)
        assert clock.is_on_beat(tolerance_ms=10)
        clock.seek(520.0)
        assert not clock.is_on_beat(tolerance_ms=10)
        clock.stop()

    def test_is_on_bar(self):
        """Check if on bar boundary."""
        clock = MusicClock(bpm=120.0, time_signature=TimeSignature(4, 4))
        clock.start()
        clock.seek(2000.0)
        assert clock.is_on_bar(tolerance_ms=10)
        clock.seek(2500.0)
        assert not clock.is_on_bar(tolerance_ms=10)
        clock.stop()

    def test_set_bpm_updates_grid(self):
        """Setting BPM updates the beat grid."""
        clock = MusicClock(bpm=120.0)
        clock.bpm = 180.0
        assert clock.bpm == 180.0
        assert clock.grid.bpm == 180.0

    def test_clock_grid_property(self):
        """Access beat grid through clock."""
        clock = MusicClock(bpm=120.0)
        assert clock.grid.beat_duration_ms == 500.0


class TestSyncPointManager:
    """Tests for SyncPointManager class."""

    def test_create_sync_point_manager(self):
        """Create sync point manager."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        assert manager.get_all_sync_points() == []

    def test_add_sync_point(self):
        """Add a sync point."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        sp = manager.add_sync_point("intro_end", bar=4, beat=0.0)
        assert sp.name == "intro_end"
        assert sp.bar == 4
        assert sp.time_ms == pytest.approx(8000.0)

    def test_add_sync_point_with_metadata(self):
        """Add sync point with metadata."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        sp = manager.add_sync_point(
            "boss",
            bar=8,
            beat=0.0,
            metadata={"type": "boss_intro"},
        )
        assert sp.metadata["type"] == "boss_intro"

    def test_remove_sync_point(self):
        """Remove a sync point."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        manager.add_sync_point("test", bar=2)
        assert manager.remove_sync_point("test") is True
        assert manager.get_sync_point("test") is None

    def test_remove_nonexistent_sync_point(self):
        """Removing nonexistent sync point returns False."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        assert manager.remove_sync_point("nonexistent") is False

    def test_get_sync_point(self):
        """Get sync point by name."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        manager.add_sync_point("marker", bar=3)
        sp = manager.get_sync_point("marker")
        assert sp is not None
        assert sp.name == "marker"

    def test_get_all_sync_points_sorted(self):
        """Get all sync points sorted by time."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        manager.add_sync_point("late", bar=8)
        manager.add_sync_point("early", bar=2)
        manager.add_sync_point("mid", bar=5)
        points = manager.get_all_sync_points()
        assert len(points) == 3
        assert points[0].name == "early"
        assert points[1].name == "mid"
        assert points[2].name == "late"

    def test_get_next_sync_point(self):
        """Get next sync point after a time."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        manager.add_sync_point("sp1", bar=2)  # 4000ms
        manager.add_sync_point("sp2", bar=4)  # 8000ms
        manager.add_sync_point("sp3", bar=6)  # 12000ms

        sp = manager.get_next_sync_point(5000.0)
        assert sp.name == "sp2"

    def test_get_next_sync_point_with_filter(self):
        """Get next sync point with name filter."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        manager.add_sync_point("marker_1", bar=2)  # 4000ms
        manager.add_sync_point("event_1", bar=4)   # 8000ms
        manager.add_sync_point("marker_2", bar=6)  # 12000ms

        # Looking for next sync point after 5000ms with marker prefix
        sp = manager.get_next_sync_point(5000.0, name_filter="marker")
        assert sp.name == "marker_2"

    def test_get_sync_points_in_range(self):
        """Get sync points within a time range."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        manager.add_sync_point("sp1", bar=1)  # 2000ms
        manager.add_sync_point("sp2", bar=3)  # 6000ms
        manager.add_sync_point("sp3", bar=5)  # 10000ms

        points = manager.get_sync_points_in_range(1500.0, 7000.0)
        assert len(points) == 2
        assert points[0].name == "sp1"
        assert points[1].name == "sp2"

    def test_find_nearest_sync_point(self):
        """Find nearest sync point within tolerance."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        manager.add_sync_point("sp1", bar=2)  # 4000ms

        sp = manager.find_nearest_sync_point(4005.0, tolerance_ms=10.0)
        assert sp.name == "sp1"

        sp = manager.find_nearest_sync_point(4100.0, tolerance_ms=10.0)
        assert sp is None

    def test_clear_sync_points(self):
        """Clear all sync points."""
        grid = BeatGrid(bpm=120.0)
        manager = SyncPointManager(grid)
        manager.add_sync_point("sp1", bar=1)
        manager.add_sync_point("sp2", bar=2)
        manager.clear()
        assert manager.get_all_sync_points() == []


class TestBeatSubdivision:
    """Tests for BeatSubdivision enum."""

    def test_subdivision_values(self):
        """Subdivision values should be correct."""
        assert BeatSubdivision.WHOLE.value == 1
        assert BeatSubdivision.HALF.value == 2
        assert BeatSubdivision.QUARTER.value == 4
        assert BeatSubdivision.EIGHTH.value == 8
        assert BeatSubdivision.SIXTEENTH.value == 16
        assert BeatSubdivision.THIRTY_SECOND.value == 32

    def test_triplet_subdivisions(self):
        """Triplet subdivisions should be correct."""
        assert BeatSubdivision.TRIPLET_QUARTER.value == 3
        assert BeatSubdivision.TRIPLET_EIGHTH.value == 6
        assert BeatSubdivision.TRIPLET_SIXTEENTH.value == 12
