"""Blackbox tests for music_transition.py -- transitions and crossfades.

BLACKBOX coverage plan:
  - TransitionState enum values
  - TransitionConfig dataclass
  - TransitionRequest dataclass
  - TransitionProgress dataclass
  - MusicTransition initialization
  - MusicTransition state management
  - TransitionManager initialization
  - TransitionManager request handling
  - Fade curve types
  - Beat-aligned transitions
  - Bar-aligned transitions

Total: 25+ tests
"""

from __future__ import annotations

import pytest
import math
from typing import Optional
from unittest.mock import MagicMock


class TestTransitionState:
    """Tests for TransitionState enumeration."""

    def test_idle_state_exists(self):
        """TransitionState should have IDLE state."""
        from engine.audio.adaptive.music_transition import TransitionState

        assert hasattr(TransitionState, 'IDLE')

    def test_pending_state_exists(self):
        """TransitionState should have PENDING state."""
        from engine.audio.adaptive.music_transition import TransitionState

        assert hasattr(TransitionState, 'PENDING')

    def test_active_state_exists(self):
        """TransitionState should have ACTIVE state."""
        from engine.audio.adaptive.music_transition import TransitionState

        assert hasattr(TransitionState, 'ACTIVE')

    def test_completing_state_exists(self):
        """TransitionState should have COMPLETING state."""
        from engine.audio.adaptive.music_transition import TransitionState

        assert hasattr(TransitionState, 'COMPLETING')

    def test_completed_state_exists(self):
        """TransitionState should have COMPLETED state."""
        from engine.audio.adaptive.music_transition import TransitionState

        assert hasattr(TransitionState, 'COMPLETED')

    def test_cancelled_state_exists(self):
        """TransitionState should have CANCELLED state."""
        from engine.audio.adaptive.music_transition import TransitionState

        assert hasattr(TransitionState, 'CANCELLED')


class TestTransitionConfig:
    """Tests for TransitionConfig dataclass."""

    def test_create_transition_config(self):
        """Should create TransitionConfig with defaults."""
        from engine.audio.adaptive.music_transition import TransitionConfig

        config = TransitionConfig()
        assert config is not None

    def test_config_transition_type(self):
        """TransitionConfig should have transition_type."""
        from engine.audio.adaptive.music_transition import TransitionConfig
        from engine.audio.adaptive.config import TRANSITION_CROSSFADE

        config = TransitionConfig(transition_type=TRANSITION_CROSSFADE)
        assert config.transition_type == TRANSITION_CROSSFADE

    def test_config_duration(self):
        """TransitionConfig should accept duration_ms."""
        from engine.audio.adaptive.music_transition import TransitionConfig

        config = TransitionConfig(duration_ms=2000.0)
        assert config.duration_ms == 2000.0

    def test_config_fade_curve(self):
        """TransitionConfig should accept fade_curve."""
        from engine.audio.adaptive.music_transition import TransitionConfig
        from engine.audio.adaptive.config import FADE_CURVE_LINEAR

        config = TransitionConfig(fade_curve=FADE_CURVE_LINEAR)
        assert config.fade_curve == FADE_CURVE_LINEAR

    def test_config_stinger_id(self):
        """TransitionConfig should accept stinger_id."""
        from engine.audio.adaptive.music_transition import TransitionConfig
        from engine.audio.adaptive.config import TRANSITION_STINGER

        config = TransitionConfig(
            transition_type=TRANSITION_STINGER,
            stinger_id="impact_hit"
        )
        assert config.stinger_id == "impact_hit"

    def test_config_quantize_options(self):
        """TransitionConfig should accept quantize options."""
        from engine.audio.adaptive.music_transition import TransitionConfig

        config = TransitionConfig(
            quantize_to_beat=True,
            quantize_to_bar=False
        )
        assert config.quantize_to_beat is True
        assert config.quantize_to_bar is False

    def test_invalid_transition_type_rejected(self):
        """TransitionConfig should reject invalid type."""
        from engine.audio.adaptive.music_transition import TransitionConfig

        with pytest.raises(ValueError):
            TransitionConfig(transition_type="invalid_type")

    def test_duration_too_short_rejected(self):
        """TransitionConfig should reject too short duration."""
        from engine.audio.adaptive.music_transition import TransitionConfig

        with pytest.raises(ValueError):
            TransitionConfig(duration_ms=1.0)  # Too short


class TestTransitionRequest:
    """Tests for TransitionRequest dataclass."""

    def test_create_transition_request(self):
        """Should create TransitionRequest."""
        from engine.audio.adaptive.music_transition import (
            TransitionRequest,
            TransitionConfig,
        )

        config = TransitionConfig()
        request = TransitionRequest(
            request_id=1,
            config=config,
            destination_id="combat"
        )

        assert request.request_id == 1
        assert request.destination_id == "combat"

    def test_request_source_id(self):
        """TransitionRequest should accept source_id."""
        from engine.audio.adaptive.music_transition import (
            TransitionRequest,
            TransitionConfig,
        )

        request = TransitionRequest(
            request_id=1,
            config=TransitionConfig(),
            source_id="exploration",
            destination_id="combat"
        )

        assert request.source_id == "exploration"

    def test_request_priority(self):
        """TransitionRequest should accept priority."""
        from engine.audio.adaptive.music_transition import (
            TransitionRequest,
            TransitionConfig,
        )

        request = TransitionRequest(
            request_id=1,
            config=TransitionConfig(),
            destination_id="combat",
            priority=10
        )

        assert request.priority == 10


class TestTransitionProgress:
    """Tests for TransitionProgress dataclass."""

    def test_create_transition_progress(self):
        """Should create TransitionProgress."""
        from engine.audio.adaptive.music_transition import (
            TransitionProgress,
            TransitionRequest,
            TransitionConfig,
            TransitionState,
        )

        request = TransitionRequest(
            request_id=1,
            config=TransitionConfig(),
            destination_id="combat"
        )
        progress = TransitionProgress(request=request)

        assert progress.state == TransitionState.IDLE
        assert progress.progress == 0.0

    def test_progress_volumes(self):
        """TransitionProgress should track source/dest volumes."""
        from engine.audio.adaptive.music_transition import (
            TransitionProgress,
            TransitionRequest,
            TransitionConfig,
        )

        request = TransitionRequest(
            request_id=1,
            config=TransitionConfig(),
            destination_id="combat"
        )
        progress = TransitionProgress(
            request=request,
            source_volume=0.5,
            destination_volume=0.5
        )

        assert progress.source_volume == 0.5
        assert progress.destination_volume == 0.5


class TestMusicTransitionInitialization:
    """Tests for MusicTransition construction."""

    def test_create_music_transition(self):
        """Should create MusicTransition with request."""
        from engine.audio.adaptive.music_transition import (
            MusicTransition,
            TransitionRequest,
            TransitionConfig,
        )

        request = TransitionRequest(
            request_id=1,
            config=TransitionConfig(),
            destination_id="combat"
        )
        transition = MusicTransition(request=request)

        assert transition is not None

    def test_transition_with_clock(self):
        """MusicTransition should accept clock parameter."""
        from engine.audio.adaptive.music_transition import (
            MusicTransition,
            TransitionRequest,
            TransitionConfig,
        )
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        request = TransitionRequest(
            request_id=1,
            config=TransitionConfig(),
            destination_id="combat"
        )
        transition = MusicTransition(request=request, clock=clock)

        assert transition is not None


class TestMusicTransitionProperties:
    """Tests for MusicTransition properties."""

    def test_progress_snapshot(self):
        """get_progress_snapshot should return progress info."""
        from engine.audio.adaptive.music_transition import (
            MusicTransition,
            TransitionRequest,
            TransitionConfig,
        )

        request = TransitionRequest(
            request_id=42,
            config=TransitionConfig(),
            destination_id="combat"
        )
        transition = MusicTransition(request=request)

        snapshot = transition.get_progress_snapshot()
        assert snapshot.request.request_id == 42

    def test_state_property(self):
        """state property should return current state."""
        from engine.audio.adaptive.music_transition import (
            MusicTransition,
            TransitionRequest,
            TransitionConfig,
            TransitionState,
        )

        request = TransitionRequest(
            request_id=1,
            config=TransitionConfig(),
            destination_id="combat"
        )
        transition = MusicTransition(request=request)

        assert transition.state == TransitionState.IDLE


class TestMusicTransitionLifecycle:
    """Tests for MusicTransition lifecycle."""

    def test_start_transition(self):
        """start should begin the transition."""
        from engine.audio.adaptive.music_transition import (
            MusicTransition,
            TransitionRequest,
            TransitionConfig,
            TransitionState,
        )

        request = TransitionRequest(
            request_id=1,
            config=TransitionConfig(),
            destination_id="combat"
        )
        transition = MusicTransition(request=request)
        transition.start()

        assert transition.state in (TransitionState.ACTIVE, TransitionState.PENDING)

    def test_cancel_transition(self):
        """cancel should abort the transition."""
        from engine.audio.adaptive.music_transition import (
            MusicTransition,
            TransitionRequest,
            TransitionConfig,
            TransitionState,
        )

        request = TransitionRequest(
            request_id=1,
            config=TransitionConfig(),
            destination_id="combat"
        )
        transition = MusicTransition(request=request)
        transition.start()
        transition.cancel()

        assert transition.state == TransitionState.CANCELLED


class TestTransitionManager:
    """Tests for TransitionManager."""

    def test_create_transition_manager(self):
        """Should create TransitionManager with required clock."""
        from engine.audio.adaptive.music_transition import TransitionManager
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = TransitionManager(clock=clock)
        assert manager is not None

    def test_manager_with_clock(self):
        """TransitionManager should accept clock."""
        from engine.audio.adaptive.music_transition import TransitionManager
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = TransitionManager(clock=clock)

        assert manager is not None

    def test_request_transition(self):
        """request_transition should queue a transition."""
        from engine.audio.adaptive.music_transition import TransitionManager
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = TransitionManager(clock=clock)

        request_id = manager.request_transition(
            destination_id="combat"
        )

        assert request_id is not None


class TestFadeCurves:
    """Tests for fade curve functionality."""

    def test_linear_curve_exists(self):
        """Linear fade curve should be available."""
        from engine.audio.adaptive.config import FADE_CURVE_LINEAR

        assert FADE_CURVE_LINEAR is not None

    def test_equal_power_curve_exists(self):
        """Equal power fade curve should be available."""
        from engine.audio.adaptive.config import FADE_CURVE_EQUAL_POWER

        assert FADE_CURVE_EQUAL_POWER is not None

    def test_s_curve_exists(self):
        """S-curve fade should be available."""
        from engine.audio.adaptive.config import FADE_CURVE_S_CURVE

        assert FADE_CURVE_S_CURVE is not None


class TestEdgeCases:
    """Edge case tests for transitions."""

    def test_zero_progress_volumes(self):
        """At zero progress, source should be full."""
        from engine.audio.adaptive.music_transition import (
            TransitionProgress,
            TransitionRequest,
            TransitionConfig,
        )

        request = TransitionRequest(
            request_id=1,
            config=TransitionConfig(),
            destination_id="combat"
        )
        progress = TransitionProgress(request=request, progress=0.0)

        assert progress.source_volume == 1.0
        assert progress.destination_volume == 0.0

    def test_rapid_transition_requests(self):
        """Should handle rapid transition requests."""
        from engine.audio.adaptive.music_transition import TransitionManager
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = TransitionManager(clock=clock)

        for i in range(10):
            manager.request_transition(
                destination_id=f"state_{i}"
            )

        # Should not crash
        assert True

    def test_very_long_transition(self):
        """Should handle very long transition duration."""
        from engine.audio.adaptive.music_transition import TransitionConfig

        config = TransitionConfig(duration_ms=60000.0)  # 1 minute

        assert config.duration_ms == 60000.0
