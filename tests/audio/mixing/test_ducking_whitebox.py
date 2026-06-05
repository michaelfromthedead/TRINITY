"""Whitebox tests for the Ducking system.

Tests internal implementation of:
- DuckEnvelope state machine (IDLE/ATTACKING/HOLDING/RELEASING)
- DuckConfig validation
- DuckingInstance level tracking
- DuckingManager multi-duck coordination
- Attack/hold/release timing
- dB/linear conversions for duck amounts
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from engine.audio.mixing.config import (
    DIALOGUE_DUCK_AMOUNT_DB,
    DUCK_ATTACK_MS,
    DUCK_HOLD_MS,
    DUCK_RELEASE_MS,
    DUCK_THRESHOLD_DB,
    EVENT_DUCK_AMOUNT_DB,
    EVENT_DUCK_ATTACK_MS,
    EVENT_DUCK_HOLD_MS,
    FOCUS_DUCK_AMOUNT_DB,
    FOCUS_DUCK_ATTACK_MS,
    FOCUS_DUCK_HOLD_MS,
    FOCUS_DUCK_RELEASE_MS,
    db_to_linear,
    linear_to_db,
)
from engine.audio.mixing.ducking import (
    DuckConfig,
    DuckEnvelope,
    DuckingInstance,
    DuckingManager,
    DuckState,
    DuckType,
)
from engine.audio.mixing.mix_bus import BusType, MixBus


# =============================================================================
# DuckType Tests
# =============================================================================


class TestDuckType:
    """Test DuckType enum."""

    def test_duck_types_exist(self):
        """All duck types exist."""
        assert DuckType.DIALOGUE.value == "dialogue"
        assert DuckType.EVENT.value == "event"
        assert DuckType.FOCUS.value == "focus"
        assert DuckType.CUSTOM.value == "custom"


# =============================================================================
# DuckState Tests
# =============================================================================


class TestDuckState:
    """Test DuckState enum."""

    def test_duck_states_exist(self):
        """All duck states exist."""
        assert DuckState.IDLE.value == "idle"
        assert DuckState.ATTACKING.value == "attacking"
        assert DuckState.HOLDING.value == "holding"
        assert DuckState.RELEASING.value == "releasing"


# =============================================================================
# DuckEnvelope Tests
# =============================================================================


class TestDuckEnvelope:
    """Test DuckEnvelope state machine."""

    def test_default_values(self):
        """DuckEnvelope has correct defaults."""
        env = DuckEnvelope()
        assert env.attack_ms == DUCK_ATTACK_MS
        assert env.hold_ms == DUCK_HOLD_MS
        assert env.release_ms == DUCK_RELEASE_MS
        assert env.state == DuckState.IDLE
        assert env.current_amount == 0.0
        assert env.target_amount == 0.0

    def test_trigger_starts_attack(self):
        """Trigger transitions to ATTACKING state."""
        env = DuckEnvelope()
        env.trigger(1.0)

        assert env.state == DuckState.ATTACKING
        assert env.target_amount == 1.0

    def test_trigger_clamped(self):
        """Trigger amount is clamped to [0, 1]."""
        env = DuckEnvelope()

        env.trigger(1.5)
        assert env.target_amount == 1.0

        env.reset()
        env.trigger(-0.5)
        assert env.target_amount == 0.0

    def test_trigger_from_releasing(self):
        """Trigger while releasing restarts attack."""
        env = DuckEnvelope()
        env.trigger(1.0)

        # Simulate getting to releasing state
        env.state = DuckState.RELEASING
        env.current_amount = 0.5

        env.trigger(1.0)

        assert env.state == DuckState.ATTACKING

    def test_release_transitions_to_holding(self):
        """Release while attacking/holding starts hold timer."""
        env = DuckEnvelope()
        env.trigger(1.0)

        env.release()

        assert env.state == DuckState.HOLDING
        assert env.hold_end_time > 0

    def test_update_attack_phase(self):
        """Update during attack phase increases amount."""
        env = DuckEnvelope(attack_ms=100.0)
        env.trigger(1.0)

        # Small time step
        env.update(0.01)

        assert env.current_amount > 0.0
        assert env.current_amount < 1.0

    def test_update_attack_completes(self):
        """Attack phase completes and transitions to holding."""
        env = DuckEnvelope(attack_ms=10.0)
        env.trigger(1.0)

        # Large time step to complete attack
        env.update(0.1)

        assert env.current_amount == 1.0
        assert env.state == DuckState.HOLDING

    def test_update_hold_phase(self):
        """Hold phase waits before transitioning to release."""
        env = DuckEnvelope(attack_ms=0.0, hold_ms=100.0)
        env.trigger(1.0)

        env.update(0.001)  # Complete attack instantly

        # Should be in holding
        assert env.state == DuckState.HOLDING

        # Small update shouldn't transition yet
        env.update(0.01)
        assert env.state == DuckState.HOLDING

    def test_update_release_phase(self):
        """Release phase decreases amount."""
        env = DuckEnvelope(attack_ms=0.0, hold_ms=0.0, release_ms=100.0)
        env.trigger(1.0)

        env.update(0.001)  # Complete attack
        env.update(0.001)  # Complete hold (0ms)

        # Should be releasing
        assert env.state == DuckState.RELEASING

        env.update(0.01)
        assert env.current_amount < 1.0

    def test_update_release_completes(self):
        """Release phase completes and transitions to idle."""
        env = DuckEnvelope(attack_ms=0.0, hold_ms=0.0, release_ms=10.0)
        env.trigger(1.0)

        env.update(0.001)  # Complete attack
        env.update(0.001)  # Complete hold

        # Large update to complete release
        env.update(0.5)

        assert env.current_amount == 0.0
        assert env.state == DuckState.IDLE

    def test_update_zero_attack_time(self):
        """Zero attack time jumps to target instantly."""
        env = DuckEnvelope(attack_ms=0.0)
        env.trigger(1.0)

        env.update(0.001)

        assert env.current_amount == 1.0

    def test_update_zero_release_time(self):
        """Zero release time jumps to zero instantly."""
        env = DuckEnvelope(attack_ms=0.0, hold_ms=0.0, release_ms=0.0)
        env.trigger(1.0)

        env.update(0.001)  # Attack
        env.update(0.001)  # Hold
        env.update(0.001)  # Release

        assert env.current_amount == 0.0

    def test_reset(self):
        """Reset returns envelope to idle."""
        env = DuckEnvelope()
        env.trigger(1.0)
        env.update(0.01)

        env.reset()

        assert env.state == DuckState.IDLE
        assert env.current_amount == 0.0
        assert env.target_amount == 0.0

    def test_is_active_when_ducking(self):
        """is_active returns True when ducking."""
        env = DuckEnvelope()

        assert env.is_active() is False

        env.trigger(1.0)
        assert env.is_active() is True

    def test_is_active_with_residual_amount(self):
        """is_active True even in IDLE with residual amount."""
        env = DuckEnvelope()
        env.current_amount = 0.1
        env.state = DuckState.IDLE

        assert env.is_active() is True


# =============================================================================
# DuckConfig Tests
# =============================================================================


class TestDuckConfig:
    """Test DuckConfig dataclass."""

    def test_default_values(self):
        """DuckConfig has correct defaults."""
        config = DuckConfig()
        assert config.name == ""
        assert config.duck_type == DuckType.DIALOGUE
        assert config.source_bus is None
        assert config.target_buses == []
        assert config.amount_db == DIALOGUE_DUCK_AMOUNT_DB
        assert config.threshold_db == DUCK_THRESHOLD_DB
        assert config.attack_ms == DUCK_ATTACK_MS
        assert config.hold_ms == DUCK_HOLD_MS
        assert config.release_ms == DUCK_RELEASE_MS
        assert config.enabled is True
        assert config.priority == 100

    def test_amount_linear_property(self):
        """amount_linear converts dB to linear."""
        config = DuckConfig(amount_db=-6.0)
        assert config.amount_linear == pytest.approx(0.5012, rel=1e-3)

        config = DuckConfig(amount_db=-12.0)
        assert config.amount_linear == pytest.approx(0.251, rel=1e-2)

    def test_copy(self):
        """copy creates independent copy."""
        source = MixBus("vo", BusType.CATEGORY)
        target = MixBus("music", BusType.CATEGORY)

        config = DuckConfig(
            name="test_duck",
            duck_type=DuckType.EVENT,
            source_bus=source,
            target_buses=[target],
            amount_db=-9.0,
            priority=200,
        )

        copy = config.copy()

        assert copy.name == "test_duck"
        assert copy.duck_type == DuckType.EVENT
        assert copy.source_bus is source
        assert target in copy.target_buses
        assert copy.amount_db == -9.0
        assert copy.priority == 200

        # Modify copy's list
        copy.target_buses.append(MixBus("sfx", BusType.CATEGORY))
        assert len(config.target_buses) == 1


# =============================================================================
# DuckingInstance Tests
# =============================================================================


class TestDuckingInstance:
    """Test DuckingInstance."""

    def test_initialization(self):
        """DuckingInstance initializes from config."""
        source = MixBus("vo", BusType.CATEGORY)
        target = MixBus("music", BusType.CATEGORY)

        config = DuckConfig(
            name="dialogue_duck",
            source_bus=source,
            target_buses=[target],
            amount_db=-12.0,
            attack_ms=50.0,
            hold_ms=100.0,
            release_ms=500.0,
        )

        instance = DuckingInstance(config)

        assert instance.config.name == "dialogue_duck"
        assert instance.envelope.attack_ms == 50.0
        assert instance.envelope.hold_ms == 100.0
        assert instance.envelope.release_ms == 500.0

    def test_is_active(self):
        """is_active delegates to envelope."""
        config = DuckConfig()
        instance = DuckingInstance(config)

        assert instance.is_active is False

        instance.trigger()
        assert instance.is_active is True

    def test_current_duck_db(self):
        """current_duck_db returns scaled duck amount."""
        config = DuckConfig(amount_db=-12.0)
        instance = DuckingInstance(config)
        instance._envelope.current_amount = 0.5

        # -12.0 * 0.5 = -6.0
        assert instance.current_duck_db == pytest.approx(-6.0, rel=1e-6)

    def test_current_duck_linear(self):
        """current_duck_linear converts duck to multiplier."""
        config = DuckConfig(amount_db=-12.0)
        instance = DuckingInstance(config)
        instance._envelope.current_amount = 0.0

        # No duck = 1.0
        assert instance.current_duck_linear == 1.0

        instance._envelope.current_amount = 1.0
        # Full duck at -12dB
        assert instance.current_duck_linear == pytest.approx(0.251, rel=1e-2)

    def test_set_source_level_triggers_duck(self):
        """set_source_level triggers ducking above threshold."""
        config = DuckConfig(threshold_db=-20.0)
        instance = DuckingInstance(config)

        # Below threshold - no duck
        instance.set_source_level(-30.0)
        assert instance.envelope.state == DuckState.IDLE

        # Above threshold - triggers duck
        instance.set_source_level(-10.0)
        assert instance.envelope.state == DuckState.ATTACKING

    def test_set_source_level_releases_duck(self):
        """set_source_level releases duck below threshold."""
        config = DuckConfig(threshold_db=-20.0)
        instance = DuckingInstance(config)

        # Trigger
        instance.set_source_level(-10.0)
        assert instance.envelope.state == DuckState.ATTACKING

        # Drop below threshold
        instance.set_source_level(-30.0)
        assert instance.envelope.state == DuckState.HOLDING

    def test_set_source_level_disabled(self):
        """set_source_level does nothing when disabled."""
        config = DuckConfig(threshold_db=-20.0, enabled=False)
        instance = DuckingInstance(config)

        instance.set_source_level(-10.0)
        assert instance.envelope.state == DuckState.IDLE

    def test_manual_trigger(self):
        """Manual trigger activates ducking."""
        config = DuckConfig()
        instance = DuckingInstance(config)

        instance.trigger(0.5)

        assert instance.envelope.state == DuckState.ATTACKING
        assert instance.envelope.target_amount == 0.5

    def test_manual_release(self):
        """Manual release starts release phase."""
        config = DuckConfig()
        instance = DuckingInstance(config)

        instance.trigger()
        instance.release()

        assert instance.envelope.state == DuckState.HOLDING

    def test_update(self):
        """Update advances envelope and returns duck linear."""
        config = DuckConfig(attack_ms=10.0)
        instance = DuckingInstance(config)

        instance.trigger()

        result = instance.update(0.1)

        assert result < 1.0  # Some ducking applied
        assert result > 0.0

    def test_apply_to_targets(self):
        """apply_to_targets returns multipliers for target buses."""
        target1 = MixBus("music", BusType.CATEGORY)
        target2 = MixBus("sfx", BusType.CATEGORY)

        config = DuckConfig(
            target_buses=[target1, target2],
            amount_db=-12.0,
        )
        instance = DuckingInstance(config)
        instance._envelope.current_amount = 1.0

        result = instance.apply_to_targets()

        assert target1.id in result
        assert target2.id in result
        assert result[target1.id] == pytest.approx(0.251, rel=1e-2)
        assert result[target2.id] == pytest.approx(0.251, rel=1e-2)

    def test_reset(self):
        """Reset clears instance state."""
        config = DuckConfig()
        instance = DuckingInstance(config)

        instance.trigger()
        instance.update(0.01)

        instance.reset()

        assert instance.is_active is False
        assert instance.envelope.state == DuckState.IDLE


# =============================================================================
# DuckingManager Tests
# =============================================================================


class TestDuckingManager:
    """Test DuckingManager."""

    def test_create_duck(self):
        """Create ducking instance."""
        manager = DuckingManager()

        config = DuckConfig(name="test_duck")
        instance = manager.create_duck(config)

        assert instance is not None
        assert instance.config.name == "test_duck"

    def test_remove_duck(self):
        """Remove ducking instance."""
        manager = DuckingManager()

        config = DuckConfig(name="test_duck")
        instance = manager.create_duck(config)

        result = manager.remove_duck(config.id)

        assert result is True
        assert manager.get_duck(config.id) is None

    def test_remove_duck_not_found(self):
        """remove_duck returns False if not found."""
        manager = DuckingManager()

        result = manager.remove_duck("nonexistent")
        assert result is False

    def test_get_duck(self):
        """Get ducking instance by ID."""
        manager = DuckingManager()

        config = DuckConfig(name="test_duck")
        instance = manager.create_duck(config)

        found = manager.get_duck(config.id)

        assert found is instance

    def test_get_duck_not_found(self):
        """get_duck returns None if not found."""
        manager = DuckingManager()

        result = manager.get_duck("nonexistent")
        assert result is None

    def test_get_ducks_by_type(self):
        """Get ducks filtered by type."""
        manager = DuckingManager()

        dialogue = DuckConfig(name="dialogue", duck_type=DuckType.DIALOGUE)
        event = DuckConfig(name="event", duck_type=DuckType.EVENT)

        manager.create_duck(dialogue)
        manager.create_duck(event)

        dialogue_ducks = manager.get_ducks_by_type(DuckType.DIALOGUE)
        event_ducks = manager.get_ducks_by_type(DuckType.EVENT)

        assert len(dialogue_ducks) == 1
        assert len(event_ducks) == 1
        assert dialogue_ducks[0].config.name == "dialogue"

    def test_get_ducks_for_target(self):
        """Get ducks affecting a target bus."""
        manager = DuckingManager()
        target = MixBus("music", BusType.CATEGORY)

        config1 = DuckConfig(target_buses=[target])
        config2 = DuckConfig(target_buses=[])

        manager.create_duck(config1)
        manager.create_duck(config2)

        ducks = manager.get_ducks_for_target(target)

        assert len(ducks) == 1

    def test_create_dialogue_duck(self):
        """Create preset dialogue ducking."""
        manager = DuckingManager()
        source = MixBus("vo", BusType.CATEGORY)
        targets = [MixBus("music", BusType.CATEGORY)]

        instance = manager.create_dialogue_duck(source, targets, amount_db=-12.0)

        assert instance.config.duck_type == DuckType.DIALOGUE
        assert instance.config.amount_db == -12.0
        assert instance.config.priority == 200

    def test_create_event_duck(self):
        """Create preset event ducking."""
        manager = DuckingManager()
        targets = [MixBus("music", BusType.CATEGORY)]

        instance = manager.create_event_duck(targets, amount_db=-6.0)

        assert instance.config.duck_type == DuckType.EVENT
        assert instance.config.amount_db == -6.0
        assert instance.config.priority == 250

    def test_create_focus_duck(self):
        """Create preset focus ducking."""
        manager = DuckingManager()
        targets = [MixBus("ambient", BusType.CATEGORY)]

        instance = manager.create_focus_duck(targets, amount_db=-9.0)

        assert instance.config.duck_type == DuckType.FOCUS
        assert instance.config.amount_db == -9.0

    def test_update_accumulates_duck_amounts(self):
        """Update accumulates duck amounts per bus."""
        manager = DuckingManager()
        target = MixBus("music", BusType.CATEGORY)

        config = DuckConfig(target_buses=[target], amount_db=-12.0)
        instance = manager.create_duck(config)

        instance.trigger()
        manager.update(0.1)

        amount = manager.get_duck_amount(target)
        assert amount < 1.0  # Some ducking applied

    def test_update_multiple_ducks_stack(self):
        """Multiple ducks on same target stack (multiply)."""
        manager = DuckingManager()
        target = MixBus("music", BusType.CATEGORY)

        config1 = DuckConfig(target_buses=[target], amount_db=-6.0)
        config2 = DuckConfig(target_buses=[target], amount_db=-6.0)

        inst1 = manager.create_duck(config1)
        inst2 = manager.create_duck(config2)

        inst1.trigger()
        inst2.trigger()

        # Fast forward both to full duck
        inst1._envelope.current_amount = 1.0
        inst2._envelope.current_amount = 1.0

        manager.update(0.001)

        amount = manager.get_duck_amount(target)
        # -6dB = ~0.5 linear, stacking takes minimum
        single_duck = db_to_linear(-6.0)
        # Amount should be <= single_duck (implementation uses min for stacking)
        assert amount <= single_duck

    def test_update_disabled_ducks_ignored(self):
        """Disabled ducks don't contribute to duck amount."""
        manager = DuckingManager()
        target = MixBus("music", BusType.CATEGORY)

        config = DuckConfig(target_buses=[target], amount_db=-12.0, enabled=False)
        instance = manager.create_duck(config)

        instance.trigger()
        manager.update(0.1)

        amount = manager.get_duck_amount(target)
        assert amount == 1.0  # No ducking

    def test_get_duck_amount(self):
        """Get duck amount for a bus."""
        manager = DuckingManager()
        target = MixBus("music", BusType.CATEGORY)

        # No ducking configured
        amount = manager.get_duck_amount(target)
        assert amount == 1.0

    def test_get_duck_amount_db(self):
        """Get duck amount in dB."""
        manager = DuckingManager()
        target = MixBus("music", BusType.CATEGORY)

        config = DuckConfig(target_buses=[target], amount_db=-12.0)
        instance = manager.create_duck(config)
        instance.trigger()
        instance._envelope.current_amount = 1.0

        manager.update(0.001)

        amount_db = manager.get_duck_amount_db(target)
        assert amount_db == pytest.approx(-12.0, rel=0.1)

    def test_trigger_event_duck(self):
        """Trigger all event ducks."""
        manager = DuckingManager()
        targets = [MixBus("music", BusType.CATEGORY)]

        instance = manager.create_event_duck(targets)

        manager.trigger_event_duck(500.0)

        assert instance.envelope.state == DuckState.ATTACKING

    def test_trigger_focus_duck(self):
        """Trigger all focus ducks."""
        manager = DuckingManager()
        targets = [MixBus("ambient", BusType.CATEGORY)]

        instance = manager.create_focus_duck(targets)

        manager.trigger_focus_duck()

        assert instance.envelope.state == DuckState.ATTACKING

    def test_release_focus_duck(self):
        """Release all focus ducks."""
        manager = DuckingManager()
        targets = [MixBus("ambient", BusType.CATEGORY)]

        instance = manager.create_focus_duck(targets)
        instance.trigger()

        manager.release_focus_duck()

        assert instance.envelope.state == DuckState.HOLDING

    def test_analyze_source_levels(self):
        """Analyze source levels triggers automatic ducking."""
        manager = DuckingManager()
        source = MixBus("vo", BusType.CATEGORY)
        target = MixBus("music", BusType.CATEGORY)

        config = DuckConfig(
            source_bus=source,
            target_buses=[target],
            threshold_db=-20.0,
        )
        instance = manager.create_duck(config)

        # Source is loud
        manager.analyze_source_levels({source.id: -10.0})

        assert instance.envelope.state == DuckState.ATTACKING

    def test_on_duck_change_callback(self):
        """Callback called on duck amount changes."""
        manager = DuckingManager()
        target = MixBus("music", BusType.CATEGORY)
        callback = MagicMock()

        manager.on_duck_change(callback)

        config = DuckConfig(target_buses=[target], amount_db=-12.0)
        instance = manager.create_duck(config)
        instance.trigger()
        instance._envelope.current_amount = 1.0

        manager.update(0.001)

        callback.assert_called()

    def test_remove_callback(self):
        """Remove callback stops notifications."""
        manager = DuckingManager()
        callback = MagicMock()

        manager.on_duck_change(callback)
        result = manager.remove_callback(callback)

        assert result is True

    def test_reset_all(self):
        """Reset all instances."""
        manager = DuckingManager()
        target = MixBus("music", BusType.CATEGORY)

        config = DuckConfig(target_buses=[target])
        instance = manager.create_duck(config)
        instance.trigger()

        manager.reset_all()

        assert instance.is_active is False

    def test_clear(self):
        """Clear all instances."""
        manager = DuckingManager()

        config = DuckConfig()
        manager.create_duck(config)

        manager.clear()

        assert len(manager.get_ducks_by_type(DuckType.DIALOGUE)) == 0

    def test_get_state(self):
        """Get ducking state for debugging."""
        manager = DuckingManager()
        target = MixBus("music", BusType.CATEGORY)

        config = DuckConfig(name="test_duck", target_buses=[target])
        manager.create_duck(config)

        state = manager.get_state()

        assert "instances" in state
        assert "bus_amounts" in state
        assert len(state["instances"]) == 1

    def test_repr(self):
        """repr shows useful info."""
        manager = DuckingManager()

        config = DuckConfig()
        instance = manager.create_duck(config)
        instance.trigger()

        repr_str = repr(manager)

        assert "DuckingManager" in repr_str
        assert "instances=" in repr_str
        assert "active=" in repr_str
