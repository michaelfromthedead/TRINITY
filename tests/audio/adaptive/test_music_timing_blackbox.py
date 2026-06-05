"""Blackbox tests for music_timing.py -- MusicClock and BeatGrid.

BLACKBOX coverage plan:
  - BeatSubdivision enum values
  - TimeSignature dataclass
  - SyncPoint dataclass
  - BeatInfo dataclass
  - BeatGrid initialization
  - BeatGrid quantization methods
  - BeatGrid time/beat conversion
  - MusicClock initialization
  - MusicClock start/stop/pause
  - MusicClock BPM and time signature

Total: 25+ tests
"""

from __future__ import annotations

import pytest
import math
from typing import Optional
from unittest.mock import MagicMock


class TestBeatSubdivision:
    """Tests for BeatSubdivision enumeration."""

    def test_whole_subdivision_exists(self):
        """BeatSubdivision should have WHOLE value."""
        from engine.audio.adaptive.music_timing import BeatSubdivision

        assert hasattr(BeatSubdivision, 'WHOLE')
        assert BeatSubdivision.WHOLE.value == 1

    def test_quarter_subdivision_exists(self):
        """BeatSubdivision should have QUARTER value."""
        from engine.audio.adaptive.music_timing import BeatSubdivision

        assert hasattr(BeatSubdivision, 'QUARTER')
        assert BeatSubdivision.QUARTER.value == 4

    def test_eighth_subdivision_exists(self):
        """BeatSubdivision should have EIGHTH value."""
        from engine.audio.adaptive.music_timing import BeatSubdivision

        assert hasattr(BeatSubdivision, 'EIGHTH')
        assert BeatSubdivision.EIGHTH.value == 8

    def test_sixteenth_subdivision_exists(self):
        """BeatSubdivision should have SIXTEENTH value."""
        from engine.audio.adaptive.music_timing import BeatSubdivision

        assert hasattr(BeatSubdivision, 'SIXTEENTH')
        assert BeatSubdivision.SIXTEENTH.value == 16


class TestTimeSignature:
    """Tests for TimeSignature dataclass."""

    def test_create_time_signature(self):
        """Should create TimeSignature with beats and unit."""
        from engine.audio.adaptive.music_timing import TimeSignature

        sig = TimeSignature(beats_per_bar=4, beat_unit=4)

        assert sig.beats_per_bar == 4
        assert sig.beat_unit == 4

    def test_time_signature_from_tuple(self):
        """from_tuple should create TimeSignature from tuple."""
        from engine.audio.adaptive.music_timing import TimeSignature

        sig = TimeSignature.from_tuple((3, 4))

        assert sig.beats_per_bar == 3
        assert sig.beat_unit == 4

    def test_time_signature_to_tuple(self):
        """to_tuple should convert to tuple."""
        from engine.audio.adaptive.music_timing import TimeSignature

        sig = TimeSignature(beats_per_bar=6, beat_unit=8)
        result = sig.to_tuple()

        assert result == (6, 8)

    def test_invalid_beats_per_bar(self):
        """Should reject invalid beats_per_bar."""
        from engine.audio.adaptive.music_timing import TimeSignature

        with pytest.raises(ValueError):
            TimeSignature(beats_per_bar=0, beat_unit=4)

    def test_invalid_beat_unit(self):
        """Should reject non-power-of-2 beat_unit."""
        from engine.audio.adaptive.music_timing import TimeSignature

        with pytest.raises(ValueError):
            TimeSignature(beats_per_bar=4, beat_unit=3)


class TestSyncPoint:
    """Tests for SyncPoint dataclass."""

    def test_create_sync_point(self):
        """Should create SyncPoint with required fields."""
        from engine.audio.adaptive.music_timing import SyncPoint

        point = SyncPoint(
            name="verse_start",
            beat=0.0,
            bar=8,
            time_ms=8000.0
        )

        assert point.name == "verse_start"
        assert point.bar == 8
        assert point.time_ms == 8000.0

    def test_sync_point_equality(self):
        """SyncPoints with same name and bar should be equal."""
        from engine.audio.adaptive.music_timing import SyncPoint

        p1 = SyncPoint(name="test", beat=0.0, bar=4, time_ms=4000.0)
        p2 = SyncPoint(name="test", beat=0.0, bar=4, time_ms=4000.0)

        assert p1 == p2

    def test_sync_point_metadata(self):
        """SyncPoint should accept metadata."""
        from engine.audio.adaptive.music_timing import SyncPoint

        point = SyncPoint(
            name="marker",
            beat=0.0,
            bar=0,
            time_ms=0.0,
            metadata={"type": "intro"}
        )

        assert point.metadata["type"] == "intro"


class TestBeatInfo:
    """Tests for BeatInfo dataclass."""

    def test_create_beat_info(self):
        """Should create BeatInfo with all fields."""
        from engine.audio.adaptive.music_timing import BeatInfo

        info = BeatInfo(
            beat_in_bar=2,
            bar=4,
            total_beats=18,
            subdivision=0,
            time_ms=9000.0,
            progress_in_beat=0.0,
            progress_in_bar=0.5
        )

        assert info.beat_in_bar == 2
        assert info.bar == 4
        assert info.total_beats == 18


class TestBeatGridInitialization:
    """Tests for BeatGrid construction."""

    def test_default_initialization(self):
        """BeatGrid should initialize with defaults."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid()
        assert grid is not None
        assert grid.bpm > 0

    def test_initialization_with_bpm(self):
        """BeatGrid should accept BPM parameter."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=120)
        assert grid.bpm == 120

    def test_initialization_with_time_signature(self):
        """BeatGrid should accept time signature."""
        from engine.audio.adaptive.music_timing import BeatGrid, TimeSignature

        sig = TimeSignature(beats_per_bar=3, beat_unit=4)
        grid = BeatGrid(bpm=120, time_signature=sig)

        assert grid.time_signature.beats_per_bar == 3

    def test_invalid_bpm_rejected(self):
        """BeatGrid should reject invalid BPM."""
        from engine.audio.adaptive.music_timing import BeatGrid

        with pytest.raises(ValueError):
            BeatGrid(bpm=0)


class TestBeatGridProperties:
    """Tests for BeatGrid properties."""

    def test_beat_duration_ms(self):
        """beat_duration_ms should be correct for BPM."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=120)  # 120 BPM = 500ms per beat

        assert abs(grid.beat_duration_ms - 500.0) < 0.1

    def test_bar_duration_ms(self):
        """bar_duration_ms should be correct for time signature."""
        from engine.audio.adaptive.music_timing import BeatGrid, TimeSignature

        grid = BeatGrid(
            bpm=120,
            time_signature=TimeSignature(beats_per_bar=4, beat_unit=4)
        )

        # 4 beats * 500ms = 2000ms
        assert abs(grid.bar_duration_ms - 2000.0) < 0.1

    def test_bpm_setter(self):
        """Setting BPM should update durations."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=120)
        grid.bpm = 60  # 1000ms per beat

        assert abs(grid.beat_duration_ms - 1000.0) < 0.1


class TestBeatGridConversion:
    """Tests for BeatGrid time/beat conversion."""

    def test_time_to_beat(self):
        """time_to_beat should convert time to beats."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=120)  # 500ms per beat

        beat = grid.time_to_beat(1000.0)  # 1 second = 2 beats
        assert abs(beat - 2.0) < 0.001

    def test_beat_to_time(self):
        """beat_to_time should convert beats to time."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=120)

        time = grid.beat_to_time(4.0)  # 4 beats = 2 seconds
        assert abs(time - 2000.0) < 0.1

    def test_conversion_round_trip(self):
        """Time -> Beat -> Time should be consistent."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=120)
        original = 1500.0

        beat = grid.time_to_beat(original)
        converted = grid.beat_to_time(beat)

        assert abs(converted - original) < 0.001

    def test_time_to_bar(self):
        """time_to_bar should return bar and beat."""
        from engine.audio.adaptive.music_timing import BeatGrid, TimeSignature

        grid = BeatGrid(
            bpm=120,
            time_signature=TimeSignature(beats_per_bar=4, beat_unit=4)
        )

        bar, beat = grid.time_to_bar(5000.0)  # 5 seconds = 10 beats

        assert bar == 2  # Bar 2 (0-indexed)
        assert abs(beat - 2.0) < 0.001  # Beat 2 in bar


class TestBeatGridQuantization:
    """Tests for BeatGrid quantization methods."""

    def test_quantize_to_beat(self):
        """quantize_to_beat should snap to nearest beat."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=120)  # 500ms per beat

        # 600ms should quantize to beat 1 (500ms)
        quantized = grid.quantize_to_beat(600.0)
        assert abs(quantized - 500.0) < 0.1

        # 900ms should quantize to beat 2 (1000ms)
        quantized = grid.quantize_to_beat(900.0)
        assert abs(quantized - 1000.0) < 0.1

    def test_quantize_to_bar(self):
        """quantize_to_bar should snap to nearest bar."""
        from engine.audio.adaptive.music_timing import BeatGrid, TimeSignature

        grid = BeatGrid(
            bpm=120,
            time_signature=TimeSignature(beats_per_bar=4, beat_unit=4)
        )

        # 2500ms should quantize to bar 1 (2000ms)
        quantized = grid.quantize_to_bar(2500.0)
        assert abs(quantized - 2000.0) < 0.1

    def test_next_beat(self):
        """next_beat should return time of next beat."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=120)

        # At 300ms, next beat is at 500ms
        next_b = grid.next_beat(300.0)
        assert abs(next_b - 500.0) < 0.1

    def test_next_bar(self):
        """next_bar should return time of next bar."""
        from engine.audio.adaptive.music_timing import BeatGrid, TimeSignature

        grid = BeatGrid(
            bpm=120,
            time_signature=TimeSignature(beats_per_bar=4, beat_unit=4)
        )

        # At 500ms, next bar is at 2000ms
        next_b = grid.next_bar(500.0)
        assert abs(next_b - 2000.0) < 0.1


class TestBeatGridBeatInfo:
    """Tests for BeatGrid beat info."""

    def test_get_beat_info(self):
        """get_beat_info should return detailed beat info."""
        from engine.audio.adaptive.music_timing import BeatGrid, TimeSignature

        grid = BeatGrid(
            bpm=120,
            time_signature=TimeSignature(beats_per_bar=4, beat_unit=4)
        )

        info = grid.get_beat_info(2500.0)  # 5 beats

        assert info.bar == 1
        assert info.beat_in_bar == 1
        assert info.total_beats == 5


class TestMusicClockInitialization:
    """Tests for MusicClock construction."""

    def test_default_initialization(self):
        """MusicClock should initialize with defaults."""
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock()
        assert clock is not None
        assert clock.bpm > 0

    def test_initialization_with_bpm(self):
        """MusicClock should accept BPM parameter."""
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=140)
        assert clock.bpm == 140

    def test_initialization_with_time_signature(self):
        """MusicClock should accept time signature."""
        from engine.audio.adaptive.music_timing import MusicClock, TimeSignature

        sig = TimeSignature(beats_per_bar=6, beat_unit=8)
        clock = MusicClock(bpm=120, time_signature=sig)

        assert clock.time_signature.beats_per_bar == 6


class TestMusicClockPlayback:
    """Tests for MusicClock playback control."""

    def test_start_clock(self):
        """start should begin clock playback."""
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        clock.start()

        assert clock.is_running is True

    def test_stop_clock(self):
        """stop should halt clock playback."""
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        clock.start()
        clock.stop()

        assert clock.is_running is False

    def test_pause_clock(self):
        """pause should suspend clock."""
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        clock.start()
        clock.pause()

        assert clock.is_running is False

    def test_resume_clock(self):
        """resume should continue from pause."""
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        clock.start()
        clock.pause()
        clock.start()  # Resume

        assert clock.is_running is True


class TestMusicClockProperties:
    """Tests for MusicClock properties."""

    def test_bpm_property(self):
        """bpm property should return current BPM."""
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=130)
        assert clock.bpm == 130

    def test_bpm_setter(self):
        """Setting BPM should update clock."""
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        clock.bpm = 140

        assert clock.bpm == 140

    def test_grid_property(self):
        """grid property should return BeatGrid."""
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        grid = clock.grid

        assert grid is not None
        assert grid.bpm == 120


class TestEdgeCases:
    """Edge case tests for music timing."""

    def test_very_fast_bpm(self):
        """Should handle very fast BPM."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=300)
        assert grid.bpm == 300
        assert grid.beat_duration_ms > 0

    def test_very_slow_bpm(self):
        """Should handle slow BPM at the minimum allowed."""
        from engine.audio.adaptive.music_timing import BeatGrid
        from engine.audio.adaptive.config import MIN_BPM

        grid = BeatGrid(bpm=MIN_BPM)  # Use minimum allowed BPM
        assert grid.bpm == MIN_BPM
        assert grid.beat_duration_ms == 60000.0 / MIN_BPM

    def test_zero_time(self):
        """Zero time should convert to zero beats."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=120)

        assert grid.time_to_beat(0.0) == 0.0
        bar, beat = grid.time_to_bar(0.0)
        assert bar == 0
        assert beat == 0.0

    def test_negative_time(self):
        """Negative time should produce negative beats."""
        from engine.audio.adaptive.music_timing import BeatGrid

        grid = BeatGrid(bpm=120)

        beat = grid.time_to_beat(-500.0)
        assert beat < 0
