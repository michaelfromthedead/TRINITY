"""Whitebox tests for Engine Audio Core: VirtualVoiceTracker module."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from engine.audio.core.audio_source import AudioSource
from engine.audio.core.config import (
    AudioCategory,
    VIRTUAL_VOICE_FORCE_PROMOTE_GRACE_MS,
    VIRTUAL_VOICE_MAX_TIME_SECONDS,
    VIRTUAL_VOICE_URGENCY_PRIORITY_WEIGHT,
    VIRTUAL_VOICE_URGENCY_RISE_WEIGHT,
    VIRTUAL_VOICE_URGENCY_TIME_WEIGHT,
    VoiceState,
)
from engine.audio.core.virtual_voice import VirtualVoiceState, VirtualVoiceTracker


# ============================================================================
# Helpers
# ============================================================================

def make_source(
    source_id: str = "test",
    priority: int = 50,
    pitch: float = 1.0,
    category: AudioCategory = AudioCategory.SFX,
    sample_rate: int = 48000,
) -> AudioSource:
    """Factory helper for AudioSource with common defaults."""
    src = AudioSource()
    src.id = source_id
    src.priority = priority
    src.pitch = pitch
    src.category = category
    if sample_rate:
        src.clip = MagicMock()
        src.clip.sample_rate = sample_rate
    return src


# ============================================================================
# TestVirtualVoiceState
# ============================================================================

class TestVirtualVoiceState:
    """VirtualVoiceState dataclass construction."""

    def test_requires_voice_id_and_source_id(self):
        """VirtualVoiceState requires voice_id and source_id as positional args."""
        state = VirtualVoiceState(voice_id=1, source_id="test")
        assert state.voice_id == 1
        assert state.source_id == "test"

    def test_default_values(self):
        """Default field values match expectations."""
        state = VirtualVoiceState(voice_id=5, source_id="snd")
        assert state.position_samples == 0
        assert state.position_at_virtualization == 0
        assert state.virtual_start_time == 0.0
        assert state.last_update_time == 0.0
        assert state.accumulated_virtual_time == 0.0
        assert state.priority_at_virtualization == 50
        assert state.current_priority == 50
        assert state.peak_priority == 50
        assert state.urgency_score == 0.0
        assert state.category == AudioCategory.SFX
        assert state.force_promote is False

    def test_explicit_values(self):
        """Explicit values override defaults."""
        state = VirtualVoiceState(
            voice_id=10,
            source_id="test2",
            position_samples=44100,
            accumulated_virtual_time=5.0,
            priority_at_virtualization=80,
            current_priority=90,
            category=AudioCategory.MUSIC,
            force_promote=True,
        )
        assert state.voice_id == 10
        assert state.source_id == "test2"
        assert state.position_samples == 44100
        assert state.accumulated_virtual_time == 5.0
        assert state.priority_at_virtualization == 80
        assert state.current_priority == 90
        assert state.category == AudioCategory.MUSIC
        assert state.force_promote is True


# ============================================================================
# TestTrackerInit
# ============================================================================

class TestTrackerInit:
    """VirtualVoiceTracker initial state."""

    def test_initial_state(self):
        tracker = VirtualVoiceTracker()
        assert tracker._states == {}
        assert tracker.total_virtualized == 0
        assert tracker.total_promoted == 0
        assert tracker.total_force_promoted == 0
        assert tracker.peak_virtual_count == 0
        assert tracker.virtual_count == 0


# ============================================================================
# TestTrackVirtualization
# ============================================================================

class TestTrackVirtualization:
    """track_virtualization() stores state snapshot."""

    @patch("time.time", return_value=1000.0)
    def test_tracks_state(self, _mock_time):
        tracker = VirtualVoiceTracker()
        source = make_source(source_id="src1", priority=80)

        tracker.track_virtualization(voice_id=1, source=source, position_samples=44100)

        state = tracker._states[1]
        assert state.voice_id == 1
        assert state.source_id == "src1"
        assert state.position_samples == 44100
        assert state.position_at_virtualization == 44100
        assert state.virtual_start_time == 1000.0
        assert state.last_update_time == 1000.0
        assert state.accumulated_virtual_time == 0.0
        assert state.priority_at_virtualization == 80
        assert state.current_priority == 80
        assert state.peak_priority == 80
        assert state.urgency_score == 0.0
        assert state.category == AudioCategory.SFX
        assert state.force_promote is False

    @patch("time.time", return_value=500.0)
    def test_increments_total_virtualized(self, _mock_time):
        tracker = VirtualVoiceTracker()
        source = make_source()

        tracker.track_virtualization(voice_id=1, source=source, position_samples=0)
        tracker.track_virtualization(voice_id=2, source=source, position_samples=0)

        assert tracker.total_virtualized == 2

    @patch("time.time", return_value=500.0)
    def test_increments_peak_virtual_count(self, _mock_time):
        tracker = VirtualVoiceTracker()

        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        assert tracker.peak_virtual_count == 1

        tracker.track_virtualization(voice_id=2, source=make_source(), position_samples=0)
        assert tracker.peak_virtual_count == 2

    @patch("time.time", return_value=500.0)
    def test_virtual_count_property(self, _mock_time):
        tracker = VirtualVoiceTracker()

        assert tracker.virtual_count == 0

        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        assert tracker.virtual_count == 1

        tracker.track_virtualization(voice_id=2, source=make_source(), position_samples=0)
        assert tracker.virtual_count == 2

    @patch("time.time", return_value=500.0)
    def test_overwrite_same_voice_id(self, _mock_time):
        """Track same voice_id overwrites previous state."""
        tracker = VirtualVoiceTracker()
        source1 = make_source(source_id="src1", priority=50)
        source2 = make_source(source_id="src2", priority=90)

        tracker.track_virtualization(voice_id=1, source=source1, position_samples=100)
        tracker.track_virtualization(voice_id=1, source=source2, position_samples=200)

        state = tracker._states[1]
        assert state.source_id == "src2"
        assert state.position_samples == 200
        assert state.priority_at_virtualization == 90
        assert tracker.virtual_count == 1

    @patch("time.time", return_value=500.0)
    def test_peak_virtual_count_never_decreases(self, _mock_time):
        tracker = VirtualVoiceTracker()

        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        tracker.track_virtualization(voice_id=2, source=make_source(), position_samples=0)
        assert tracker.peak_virtual_count == 2

        tracker.on_promoted(1)
        assert tracker.virtual_count == 1
        assert tracker.peak_virtual_count == 2


# ============================================================================
# TestUpdate
# ============================================================================

class TestUpdate:
    """update() advances time and recomputes urgency."""

    def test_early_return_on_zero_delta(self):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)

        with patch("time.time") as mt:
            mt.side_effect = [100.0, 100.0]
            tracker.update(delta_time=0.0, source_lookup={1: make_source()})
            # state should be unchanged
            state = tracker._states[1]
            assert state.accumulated_virtual_time == 0.0

    def test_early_return_on_negative_delta(self):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)

        with patch("time.time") as mt:
            mt.side_effect = [100.0, 100.0]
            tracker.update(delta_time=-1.0, source_lookup={1: make_source()})
            state = tracker._states[1]
            assert state.accumulated_virtual_time == 0.0

    def test_advances_accumulated_time(self):
        """Uses real wall-clock time for accumulated virtual time."""
        tracker = VirtualVoiceTracker()

        with patch("time.time") as mt:
            mt.side_effect = [100.0]  # registration time
            tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)

        with patch("time.time") as mt:
            mt.side_effect = [102.0]  # now=102 means 2s elapsed
            tracker.update(delta_time=1.0, source_lookup={1: make_source()})

        state = tracker._states[1]
        assert state.accumulated_virtual_time == pytest.approx(2.0, abs=0.001)

    def test_removes_orphaned_sources(self):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        tracker.track_virtualization(voice_id=2, source=make_source(), position_samples=0)

        with patch("time.time") as mt:
            mt.side_effect = [100.0, 101.0]
            tracker.update(delta_time=1.0, source_lookup={1: make_source()})

        assert 1 in tracker._states
        assert 2 not in tracker._states

    def test_advances_position_samples(self):
        tracker = VirtualVoiceTracker()

        with patch("time.time") as mt:
            mt.side_effect = [100.0, 100.0]
            tracker.track_virtualization(voice_id=1, source=make_source(pitch=1.0), position_samples=0)

        with patch("time.time") as mt:
            source = make_source(pitch=1.0)
            mt.side_effect = [100.0, 101.0]
            # delta_time=0.5s, pitch=1.0, sample_rate=48000
            # 0.5 * 1000 = 500ms, 500 * (48000/1000) * 1.0 = 24000 samples
            tracker.update(delta_time=0.5, source_lookup={1: source})

        state = tracker._states[1]
        assert state.position_samples == 24000  # 0.5s * 48000

    def test_tracks_priority_changes(self):
        tracker = VirtualVoiceTracker()

        with patch("time.time") as mt:
            mt.side_effect = [100.0, 100.0]
            tracker.track_virtualization(voice_id=1, source=make_source(priority=50), position_samples=0)

        with patch("time.time") as mt:
            changed_source = make_source(priority=90)
            mt.side_effect = [100.0, 101.0]
            tracker.update(delta_time=1.0, source_lookup={1: changed_source})

        state = tracker._states[1]
        assert state.current_priority == 90
        assert state.peak_priority == 90

    def test_recomputes_urgency(self):
        tracker = VirtualVoiceTracker()

        with patch("time.time") as mt:
            mt.side_effect = [100.0]
            tracker.track_virtualization(voice_id=1, source=make_source(priority=50), position_samples=0)

        with patch("time.time") as mt:
            # 2 seconds of wall-clock time
            mt.side_effect = [102.0]
            changed_source = make_source(priority=80)
            tracker.update(delta_time=1.0, source_lookup={1: changed_source})

        state = tracker._states[1]
        # priority_score = 80/100 = 0.8
        # time_score = (2.0/10.0)^2 = 0.04
        # rise_score = (80-50)/100 = 0.3
        # urgency = 2.0*0.8 + 1.0*0.04 + 1.5*0.3 = 1.6 + 0.04 + 0.45 = 2.09
        assert state.urgency_score == pytest.approx(2.09, abs=0.01)


# ============================================================================
# TestComputeUrgency
# ============================================================================

class TestComputeUrgency:
    """_compute_urgency() static method."""

    def test_force_promote_returns_infinity(self):
        state = VirtualVoiceState(voice_id=1, source_id="test", force_promote=True)
        assert VirtualVoiceTracker._compute_urgency(state) == float("inf")

    def test_zero_time_zero_priority(self):
        state = VirtualVoiceState(
            voice_id=1,
            source_id="test",
            accumulated_virtual_time=0.0,
            current_priority=0,
            priority_at_virtualization=0,
            peak_priority=0,
        )
        # priority_score=0, time_score=0, rise_score=0
        expected = 0.0
        assert VirtualVoiceTracker._compute_urgency(state) == expected

    def test_max_urgency(self):
        state = VirtualVoiceState(
            voice_id=1,
            source_id="test",
            accumulated_virtual_time=10.0,
            current_priority=100,
            priority_at_virtualization=0,
            peak_priority=100,
        )
        # priority_score=1.0, time_score=1.0, rise_score=1.0
        expected = 2.0 * 1.0 + 1.0 * 1.0 + 1.5 * 1.0
        assert VirtualVoiceTracker._compute_urgency(state) == pytest.approx(expected)

    def test_time_score_is_quadratic(self):
        state_half = VirtualVoiceState(
            voice_id=1,
            source_id="test",
            accumulated_virtual_time=5.0,
            current_priority=0,
        )
        state_full = VirtualVoiceState(
            voice_id=2,
            source_id="test",
            accumulated_virtual_time=10.0,
            current_priority=0,
        )
        half = VirtualVoiceTracker._compute_urgency(state_half)
        full = VirtualVoiceTracker._compute_urgency(state_full)
        # 5/10=0.5 -> 0.25, 10/10=1.0 -> 1.0
        # half is only the time component: 1.0*0.25=0.25
        # full is 1.0*1.0=1.0
        assert half < full
        assert half == pytest.approx(0.25)
        assert full == pytest.approx(1.0)

    def test_priority_rise_clamped(self):
        state = VirtualVoiceState(
            voice_id=1,
            source_id="test",
            accumulated_virtual_time=0.0,
            current_priority=100,
            priority_at_virtualization=0,
            peak_priority=200,  # cannot exceed 100 for rise component
        )
        # rise is clamped to 1.0
        result = VirtualVoiceTracker._compute_urgency(state)
        assert result == pytest.approx(2.0 * 1.0 + 0.0 + 1.5 * 1.0)


# ============================================================================
# TestGetPromotionCandidates
# ============================================================================

class TestGetPromotionCandidates:
    """get_promotion_candidates() ordering."""

    def test_empty_returns_empty_list(self):
        tracker = VirtualVoiceTracker()
        assert tracker.get_promotion_candidates(5) == []

    def test_force_promote_first(self):
        tracker = VirtualVoiceTracker()

        with patch("time.time", return_value=100.0):
            normal_source = make_source(source_id="normal", priority=50)
            force_source = make_source(source_id="force", priority=30)
            tracker.track_virtualization(
                voice_id=1, source=normal_source, position_samples=0
            )
            tracker.track_virtualization(
                voice_id=2, source=force_source, position_samples=0
            )
            # Manually set force_promote on voice 2
            tracker._states[2].force_promote = True

        candidates = tracker.get_promotion_candidates(10)
        assert candidates[0].voice_id == 2  # force_promote first
        assert candidates[1].voice_id == 1

    def test_ordered_by_urgency_descending(self):
        tracker = VirtualVoiceTracker()

        with patch("time.time", return_value=100.0):
            high = make_source(source_id="high", priority=90)
            low = make_source(source_id="low", priority=10)
            tracker.track_virtualization(
                voice_id=1, source=make_source(priority=50), position_samples=0
            )
            tracker.track_virtualization(
                voice_id=2, source=make_source(priority=50), position_samples=0
            )
            # Manually set different urgency
            tracker._states[1].urgency_score = 2.0
            tracker._states[2].urgency_score = 1.0

        candidates = tracker.get_promotion_candidates(10)
        assert candidates[0].voice_id == 1
        assert candidates[1].voice_id == 2

    def test_respects_max_count(self):
        tracker = VirtualVoiceTracker()

        with patch("time.time", return_value=100.0):
            for i in range(5):
                tracker.track_virtualization(
                    voice_id=i, source=make_source(), position_samples=0
                )

        candidates = tracker.get_promotion_candidates(3)
        assert len(candidates) == 3

    def test_force_promote_sorted_by_accumulated_time(self):
        tracker = VirtualVoiceTracker()

        with patch("time.time", return_value=100.0):
            tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
            tracker.track_virtualization(voice_id=2, source=make_source(), position_samples=0)
            tracker._states[1].force_promote = True
            tracker._states[1].accumulated_virtual_time = 15.0
            tracker._states[2].force_promote = True
            tracker._states[2].accumulated_virtual_time = 20.0

        candidates = tracker.get_promotion_candidates(10)
        assert candidates[0].voice_id == 2  # longer accumulated time first
        assert candidates[1].voice_id == 1


# ============================================================================
# TestPromotedAndReleased
# ============================================================================

class TestPromotedAndReleased:
    """on_promoted() and on_released() lifecycle."""

    @patch("time.time", return_value=100.0)
    def test_on_promoted_removes_state(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        assert tracker.virtual_count == 1

        tracker.on_promoted(voice_id=1)
        assert tracker.virtual_count == 0

    @patch("time.time", return_value=100.0)
    def test_on_promoted_increments_total_promoted(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)

        tracker.on_promoted(voice_id=1)
        assert tracker.total_promoted == 1

    @patch("time.time", return_value=100.0)
    def test_on_released_removes_state(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        assert tracker.virtual_count == 1

        tracker.on_released(voice_id=1)
        assert tracker.virtual_count == 0

    @patch("time.time", return_value=100.0)
    def test_on_released_does_not_increment_total_promoted(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)

        tracker.on_released(voice_id=1)
        assert tracker.total_promoted == 0

    @patch("time.time", return_value=100.0)
    def test_ids_not_found_are_noop(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.on_promoted(999)  # should not raise
        tracker.on_released(999)  # should not raise
        assert tracker.virtual_count == 0

    # --- total_force_promoted tests (M3 verification) ---

    @patch("time.time", return_value=100.0)
    def test_on_promoted_increments_total_force_promoted_when_force_promote(self, _mt):
        """M3: on_promoted increments total_force_promoted when state has force_promote=True."""
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        tracker._states[1].force_promote = True

        assert tracker.total_force_promoted == 0
        tracker.on_promoted(voice_id=1)
        assert tracker.total_force_promoted == 1

    @patch("time.time", return_value=100.0)
    def test_on_promoted_does_not_increment_force_promoted_when_not_force(self, _mt):
        """M3: on_promoted does NOT increment total_force_promoted when force_promote=False."""
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        assert not tracker._states[1].force_promote

        tracker.on_promoted(voice_id=1)
        assert tracker.total_force_promoted == 0

    @patch("time.time", return_value=100.0)
    def test_on_promoted_increments_force_promoted_after_force_promote_flag_set(self, _mt):
        """M3: Voice that crosses the max-time threshold gets total_force_promoted on promote."""
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)

        # Trigger force_promote via max-time exceeded
        from engine.audio.core.config import VIRTUAL_VOICE_MAX_TIME_SECONDS
        tracker._states[1].accumulated_virtual_time = VIRTUAL_VOICE_MAX_TIME_SECONDS + 0.001
        tracker._states[1].force_promote = True

        assert tracker.total_force_promoted == 0
        tracker.on_promoted(voice_id=1)
        assert tracker.total_force_promoted == 1

    @patch("time.time", return_value=100.0)
    def test_multiple_promotions_accumulate_force_promoted(self, _mt):
        """M3: Three promotions -- two force, one normal -- yields total_force_promoted=2."""
        tracker = VirtualVoiceTracker()

        for i in range(3):
            tracker.track_virtualization(
                voice_id=i, source=make_source(), position_samples=0
            )

        tracker._states[0].force_promote = True  # force
        tracker._states[1].force_promote = False  # normal
        tracker._states[2].force_promote = True  # force

        tracker.on_promoted(0)
        tracker.on_promoted(1)
        tracker.on_promoted(2)

        assert tracker.total_promoted == 3
        assert tracker.total_force_promoted == 2


# ============================================================================
# TestGetStats
# ============================================================================

class TestGetStats:
    """get_stats() output."""

    @patch("time.time", return_value=100.0)
    def test_empty_stats(self, _mt):
        tracker = VirtualVoiceTracker()
        stats = tracker.get_stats()
        assert stats["total_virtualized"] == 0
        assert stats["total_promoted"] == 0
        assert stats["total_force_promoted"] == 0
        assert stats["currently_virtual"] == 0
        assert stats["peak_virtual_count"] == 0
        assert stats["per_voice"] == []

    @patch("time.time", return_value=100.0)
    def test_after_tracking(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)

        stats = tracker.get_stats()
        assert stats["total_virtualized"] == 1
        assert stats["currently_virtual"] == 1
        assert stats["peak_virtual_count"] == 1
        assert len(stats["per_voice"]) == 1

    @patch("time.time", return_value=100.0)
    def test_per_voice_fields(self, _mt):
        tracker = VirtualVoiceTracker()
        source = make_source(source_id="unique_id", priority=75, category=AudioCategory.MUSIC)
        tracker.track_virtualization(voice_id=5, source=source, position_samples=44100)

        stats = tracker.get_stats()
        entry = stats["per_voice"][0]
        assert entry["voice_id"] == 5
        assert entry["source_id"] == "unique_id"
        assert entry["priority"] == 75
        assert entry["category"] == "MUSIC"
        assert entry["force_promote"] is False
        assert isinstance(entry["urgency"], float)

    @patch("time.time", return_value=100.0)
    def test_after_promotion(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        tracker.on_promoted(voice_id=1)

        stats = tracker.get_stats()
        assert stats["total_virtualized"] == 1
        assert stats["total_promoted"] == 1
        assert stats["currently_virtual"] == 0

    @patch("time.time", return_value=100.0)
    def test_sorted_by_urgency_descending(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        tracker.track_virtualization(voice_id=2, source=make_source(), position_samples=0)
        tracker._states[1].urgency_score = 5.0
        tracker._states[2].urgency_score = 1.0

        stats = tracker.get_stats()
        assert stats["per_voice"][0]["voice_id"] == 1
        assert stats["per_voice"][1]["voice_id"] == 2


# ============================================================================
# TestGetState
# ============================================================================

class TestGetState:
    """get_state() lookup."""

    @patch("time.time", return_value=100.0)
    def test_returns_state_for_tracked_voice(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=42, source=make_source(), position_samples=0)

        state = tracker.get_state(42)
        assert state is not None
        assert state.voice_id == 42

    @patch("time.time", return_value=100.0)
    def test_returns_none_for_unknown_voice(self, _mt):
        tracker = VirtualVoiceTracker()
        assert tracker.get_state(999) is None

    @patch("time.time", return_value=100.0)
    def test_returns_none_after_promotion(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        tracker.on_promoted(1)
        assert tracker.get_state(1) is None


# ============================================================================
# TestEdgeCases
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_update_with_empty_states(self):
        tracker = VirtualVoiceTracker()
        # Should not raise
        tracker.update(delta_time=1.0, source_lookup={})
        assert tracker.virtual_count == 0

    def test_get_promotion_candidates_with_empty_states(self):
        tracker = VirtualVoiceTracker()
        assert tracker.get_promotion_candidates(10) == []

    @patch("time.time", return_value=100.0)
    def test_force_promote_triggers_at_max_time(self, _mt):
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)

        # Set accumulated time just below threshold
        tracker._states[1].accumulated_virtual_time = VIRTUAL_VOICE_MAX_TIME_SECONDS - 0.001
        assert not tracker._states[1].force_promote

        # cross-threshold via update (time.time advanced by 0.002s)
        with patch("time.time") as mt:
            mt.side_effect = [100.002]
            tracker.update(delta_time=0.001, source_lookup={1: make_source()})

        assert tracker._states[1].force_promote

    def test_update_does_not_change_state_with_no_source_lookup(self):
        tracker = VirtualVoiceTracker()

        with patch("time.time") as mt:
            mt.side_effect = [100.0, 100.0]
            tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)

        with patch("time.time") as mt:
            mt.side_effect = [100.0, 101.0]
            tracker.update(delta_time=1.0, source_lookup={})  # voice 1 not in lookup

        assert 1 not in tracker._states  # removed as orphaned

    @patch("time.time", return_value=100.0)
    def test_multiple_track_and_promote_cycle(self, _mt):
        tracker = VirtualVoiceTracker()

        for i in range(5):
            tracker.track_virtualization(
                voice_id=i, source=make_source(), position_samples=0
            )

        assert tracker.virtual_count == 5
        assert tracker.total_virtualized == 5
        assert tracker.peak_virtual_count == 5

        for i in range(5):
            tracker.on_promoted(i)

        assert tracker.virtual_count == 0
        assert tracker.total_promoted == 5
        assert tracker.peak_virtual_count == 5

    @patch("time.time", return_value=100.0)
    def test_force_promoted_count_tracks_manually_set(self, _mt):
        """total_force_promoted is not auto-incremented; force_promote is a flag."""
        tracker = VirtualVoiceTracker()
        tracker.track_virtualization(voice_id=1, source=make_source(), position_samples=0)
        tracker._states[1].force_promote = True

        # The tracker doesn't auto-increment force_promoted count
        # It's tracked externally by the VoiceManager
        assert tracker.total_force_promoted == 0

    @patch("time.time", return_value=100.0)
    def test_update_recomputes_urgency(self, _mt):
        tracker = VirtualVoiceTracker()

        # Track with time=100
        with patch("time.time") as mt:
            mt.side_effect = [100.0]
            tracker.track_virtualization(voice_id=1, source=make_source(priority=50), position_samples=0)

        # Update with time=102 (2s wall-clock elapsed)
        with patch("time.time") as mt:
            mt.side_effect = [102.0]
            changed_source = make_source(priority=80)
            tracker.update(delta_time=1.0, source_lookup={1: changed_source})

        state = tracker._states[1]
        # priority_score = 80/100 = 0.8
        # time_score = (2.0/10.0)^2 = 0.04
        # rise_score = (80-50)/100 = 0.3
        # urgency = 2.0*0.8 + 1.0*0.04 + 1.5*0.3 = 1.6 + 0.04 + 0.45 = 2.09
        assert state.urgency_score == pytest.approx(2.09, abs=0.01)
