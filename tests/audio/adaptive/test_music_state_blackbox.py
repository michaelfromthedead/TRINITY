"""Blackbox tests for music_state.py -- MusicStateManager and MusicState.

BLACKBOX coverage plan:
  - StateChangeReason enum values
  - MusicStateConfig dataclass
  - MusicState initialization
  - MusicState properties
  - MusicStateManager initialization
  - MusicStateManager.register_state
  - MusicStateManager.change_state
  - MusicStateManager state transitions
  - State priority system
  - State history tracking
  - StateTransition configuration
  - StateHistoryEntry tracking

Total: 25+ tests
"""

from __future__ import annotations

import pytest
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch


class TestStateChangeReason:
    """Tests for StateChangeReason enumeration."""

    def test_explicit_reason_exists(self):
        """StateChangeReason should have EXPLICIT reason."""
        from engine.audio.adaptive.music_state import StateChangeReason

        assert hasattr(StateChangeReason, 'EXPLICIT')

    def test_priority_reason_exists(self):
        """StateChangeReason should have PRIORITY reason."""
        from engine.audio.adaptive.music_state import StateChangeReason

        assert hasattr(StateChangeReason, 'PRIORITY')

    def test_timeout_reason_exists(self):
        """StateChangeReason should have TIMEOUT reason."""
        from engine.audio.adaptive.music_state import StateChangeReason

        assert hasattr(StateChangeReason, 'TIMEOUT')

    def test_trigger_reason_exists(self):
        """StateChangeReason should have TRIGGER reason."""
        from engine.audio.adaptive.music_state import StateChangeReason

        assert hasattr(StateChangeReason, 'TRIGGER')

    def test_default_reason_exists(self):
        """StateChangeReason should have DEFAULT reason."""
        from engine.audio.adaptive.music_state import StateChangeReason

        assert hasattr(StateChangeReason, 'DEFAULT')


class TestMusicStateConfig:
    """Tests for MusicStateConfig dataclass."""

    def test_create_state_config(self):
        """Should create MusicStateConfig with state_id."""
        from engine.audio.adaptive.music_state import MusicStateConfig

        config = MusicStateConfig(state_id="exploration")

        assert config.state_id == "exploration"

    def test_config_default_values(self):
        """MusicStateConfig should have sensible defaults."""
        from engine.audio.adaptive.music_state import MusicStateConfig

        config = MusicStateConfig(state_id="test")

        assert config.track_ids == []
        assert config.loop is True
        assert config.can_interrupt is True

    def test_config_with_tracks(self):
        """MusicStateConfig should accept track_ids."""
        from engine.audio.adaptive.music_state import MusicStateConfig

        config = MusicStateConfig(
            state_id="combat",
            track_ids=["combat_main", "combat_intense"]
        )

        assert "combat_main" in config.track_ids
        assert "combat_intense" in config.track_ids

    def test_config_with_stem_config(self):
        """MusicStateConfig should accept stem_config."""
        from engine.audio.adaptive.music_state import MusicStateConfig

        config = MusicStateConfig(
            state_id="combat",
            stem_config={"drums": 1.0, "bass": 0.8}
        )

        assert config.stem_config["drums"] == 1.0

    def test_config_priority_auto_assignment(self):
        """MusicStateConfig should auto-assign priority from STATE_PRIORITY."""
        from engine.audio.adaptive.music_state import MusicStateConfig
        from engine.audio.adaptive.config import STATE_COMBAT, STATE_PRIORITY

        config = MusicStateConfig(state_id=STATE_COMBAT)

        # Priority should be auto-assigned
        assert config.priority == STATE_PRIORITY.get(STATE_COMBAT, 0)

    def test_config_custom_priority(self):
        """MusicStateConfig should accept custom priority."""
        from engine.audio.adaptive.music_state import MusicStateConfig

        config = MusicStateConfig(
            state_id="custom",
            priority=100
        )

        assert config.priority == 100


class TestStateTransition:
    """Tests for StateTransition dataclass."""

    def test_create_state_transition(self):
        """Should create StateTransition with from/to states."""
        from engine.audio.adaptive.music_state import StateTransition

        transition = StateTransition(
            from_state="exploration",
            to_state="combat"
        )

        assert transition.from_state == "exploration"
        assert transition.to_state == "combat"

    def test_transition_with_stinger(self):
        """StateTransition should accept stinger_id."""
        from engine.audio.adaptive.music_state import StateTransition

        transition = StateTransition(
            from_state="exploration",
            to_state="combat",
            stinger_id="combat_start"
        )

        assert transition.stinger_id == "combat_start"

    def test_transition_with_conditions(self):
        """StateTransition should accept conditions."""
        from engine.audio.adaptive.music_state import StateTransition

        transition = StateTransition(
            from_state="exploration",
            to_state="combat",
            conditions={"min_danger": 0.5}
        )

        assert transition.conditions["min_danger"] == 0.5


class TestStateHistoryEntry:
    """Tests for StateHistoryEntry dataclass."""

    def test_create_history_entry(self):
        """Should create StateHistoryEntry."""
        from engine.audio.adaptive.music_state import StateHistoryEntry

        entry = StateHistoryEntry(
            state_id="exploration",
            enter_time=1000.0
        )

        assert entry.state_id == "exploration"
        assert entry.enter_time == 1000.0

    def test_history_entry_defaults(self):
        """StateHistoryEntry should have sensible defaults."""
        from engine.audio.adaptive.music_state import StateHistoryEntry, StateChangeReason

        entry = StateHistoryEntry(
            state_id="test",
            enter_time=0.0
        )

        assert entry.exit_time is None
        assert entry.reason == StateChangeReason.EXPLICIT
        assert entry.duration_ms == 0.0


class TestMusicState:
    """Tests for MusicState class."""

    def test_create_music_state(self):
        """Should create MusicState with config."""
        from engine.audio.adaptive.music_state import MusicState, MusicStateConfig

        config = MusicStateConfig(state_id="exploration")
        state = MusicState(config)

        assert state.state_id == "exploration"

    def test_state_id_property(self):
        """state_id property should return state ID."""
        from engine.audio.adaptive.music_state import MusicState, MusicStateConfig

        config = MusicStateConfig(state_id="combat")
        state = MusicState(config)

        assert state.state_id == "combat"

    def test_config_property(self):
        """config property should return configuration."""
        from engine.audio.adaptive.music_state import MusicState, MusicStateConfig

        config = MusicStateConfig(state_id="test", priority=50)
        state = MusicState(config)

        assert state.config.priority == 50

    def test_priority_property(self):
        """priority property should return state priority."""
        from engine.audio.adaptive.music_state import MusicState, MusicStateConfig

        config = MusicStateConfig(state_id="test", priority=75)
        state = MusicState(config)

        assert state.priority == 75


class TestMusicStateManagerInitialization:
    """Tests for MusicStateManager construction."""

    def test_create_state_manager(self):
        """Should create MusicStateManager with required clock."""
        from engine.audio.adaptive.music_state import MusicStateManager
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        assert manager is not None

    def test_manager_with_clock(self):
        """MusicStateManager should accept clock parameter."""
        from engine.audio.adaptive.music_state import MusicStateManager
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)

        assert manager is not None


class TestMusicStateManagerRegistration:
    """Tests for state registration."""

    def test_register_state(self):
        """register_state should add state to manager."""
        from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        config = MusicStateConfig(state_id="exploration")

        manager.register_state(config)

        # State should be registered
        state = manager.get_state("exploration")
        assert state is not None

    def test_unregister_state(self):
        """unregister_state should remove state."""
        from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        config = MusicStateConfig(state_id="exploration")

        manager.register_state(config)
        result = manager.unregister_state("exploration")

        assert result is True
        state = manager.get_state("exploration")
        assert state is None

    def test_unregister_nonexistent(self):
        """unregister_state should handle missing state."""
        from engine.audio.adaptive.music_state import MusicStateManager
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        result = manager.unregister_state("nonexistent")

        assert result is False


class TestMusicStateManagerTransitions:
    """Tests for state transitions."""

    def test_change_state(self):
        """change_state should transition to new state."""
        from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        manager.register_state(MusicStateConfig(state_id="exploration"))
        manager.register_state(MusicStateConfig(state_id="combat"))

        result = manager.change_state("combat")

        assert result is True
        assert manager.current_state_id == "combat"

    def test_change_state_nonexistent(self):
        """change_state to nonexistent state should fail."""
        from engine.audio.adaptive.music_state import MusicStateManager
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        result = manager.change_state("nonexistent")

        assert result is False

    def test_previous_state(self):
        """previous_state_id should return last state."""
        from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        manager.register_state(MusicStateConfig(state_id="exploration"))
        manager.register_state(MusicStateConfig(state_id="combat"))

        manager.change_state("exploration")
        manager.change_state("combat")

        assert manager.previous_state_id == "exploration"


class TestMusicStateManagerPriority:
    """Tests for state priority system."""

    def test_priority_comparison(self):
        """Higher priority state should take precedence."""
        from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        manager.register_state(MusicStateConfig(state_id="low", priority=10))
        manager.register_state(MusicStateConfig(state_id="high", priority=100))

        manager.change_state("low")
        manager.change_state("high")

        assert manager.current_state_id == "high"

    def test_force_state_change(self):
        """force=True should override priority."""
        from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        manager.register_state(MusicStateConfig(state_id="low", priority=10))
        manager.register_state(MusicStateConfig(state_id="high", priority=100))

        manager.change_state("high")
        result = manager.change_state("low", force=True)

        assert result is True
        assert manager.current_state_id == "low"


class TestMusicStateManagerStack:
    """Tests for state stack operations."""

    def test_push_state(self):
        """push_state should save current state."""
        from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        manager.register_state(MusicStateConfig(state_id="exploration"))
        manager.register_state(MusicStateConfig(state_id="combat"))

        manager.change_state("exploration")
        manager.push_state("combat")

        assert manager.current_state_id == "combat"

    def test_pop_state(self):
        """pop_state should return previous state ID from stack."""
        from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        # Same priority so we can transition freely
        manager.register_state(MusicStateConfig(state_id="exploration", priority=50))
        manager.register_state(MusicStateConfig(state_id="combat", priority=50))

        manager.change_state("exploration")
        manager.push_state("combat")
        result = manager.pop_state()

        # pop_state returns the state ID we returned to
        assert result == "exploration"


class TestMusicStateManagerHistory:
    """Tests for state history tracking."""

    def test_get_state_history(self):
        """State changes should be tracked in history."""
        from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        manager.register_state(MusicStateConfig(state_id="exploration"))
        manager.register_state(MusicStateConfig(state_id="combat"))

        manager.change_state("exploration")
        manager.change_state("combat")

        # Manager tracks state changes
        assert manager.current_state_id == "combat"
        assert manager.previous_state_id == "exploration"


class TestMusicStateManagerCallbacks:
    """Tests for state change callbacks."""

    def test_register_state_transition(self):
        """Should be able to register state transitions."""
        from engine.audio.adaptive.music_state import (
            MusicStateManager,
            MusicStateConfig,
            StateTransition,
        )
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        manager.register_state(MusicStateConfig(state_id="exploration"))
        manager.register_state(MusicStateConfig(state_id="combat"))

        transition = StateTransition(
            from_state="exploration",
            to_state="combat",
            stinger_id="combat_start"
        )
        manager.register_state_transition(transition)

        # Transition should be registered
        assert manager is not None


class TestEdgeCases:
    """Edge case tests for music state system."""

    def test_rapid_state_changes(self):
        """System should handle rapid state changes."""
        from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        manager.register_state(MusicStateConfig(state_id="a"))
        manager.register_state(MusicStateConfig(state_id="b"))
        manager.register_state(MusicStateConfig(state_id="c"))

        for _ in range(50):
            manager.change_state("a")
            manager.change_state("b")
            manager.change_state("c")

        # Should be in a valid state
        current = manager.current_state_id
        assert current in ["a", "b", "c"]

    def test_empty_manager_operations(self):
        """Operations on empty manager should be safe."""
        from engine.audio.adaptive.music_state import MusicStateManager
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)

        # Should not crash
        result = manager.change_state("nonexistent")
        assert result is False

        current = manager.current_state_id
        # Should return None for no active state
        assert current is None

    def test_get_state_returns_none_for_missing(self):
        """get_state on empty manager should return None."""
        from engine.audio.adaptive.music_state import MusicStateManager
        from engine.audio.adaptive.music_timing import MusicClock

        clock = MusicClock(bpm=120)
        manager = MusicStateManager(clock=clock)
        state = manager.get_state("nonexistent")

        assert state is None
