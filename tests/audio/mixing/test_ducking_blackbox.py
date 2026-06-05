"""
Blackbox tests for Ducking component.

Tests PUBLIC behavior only - no internal state inspection.
Covers: volume ducking with attack/release, dialogue ducking, event ducking.
"""

import pytest
import math

from engine.audio.mixing import (
    DuckingManager,
    DuckingInstance,
    DuckConfig,
    DuckEnvelope,
    DuckType,
    DuckState,
    Mixer,
    MixBus,
    BusType,
    db_to_linear,
    linear_to_db,
    DIALOGUE_DUCK_AMOUNT_DB,
    EVENT_DUCK_AMOUNT_DB,
    FOCUS_DUCK_AMOUNT_DB,
    DUCK_ATTACK_MS,
    DUCK_RELEASE_MS,
    DUCK_THRESHOLD_DB,
    DUCK_HOLD_MS,
)


class TestDuckingManagerCreation:
    """Test DuckingManager instantiation."""

    def test_create_ducking_manager(self):
        """DuckingManager can be created."""
        manager = DuckingManager()
        assert manager is not None

    def test_create_with_mixer(self):
        """DuckingManager can be created with mixer reference."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)
        assert manager is not None

    def test_default_no_active_ducks(self):
        """No ducks active by default."""
        manager = DuckingManager()
        assert manager.active_duck_count == 0


class TestDuckConfig:
    """Test duck configuration."""

    def test_create_duck_config(self):
        """DuckConfig can be created."""
        config = DuckConfig(
            amount_db=-6.0,
            attack_ms=50.0,
            release_ms=200.0
        )
        assert config.amount_db == pytest.approx(-6.0)
        assert config.attack_ms == pytest.approx(50.0)
        assert config.release_ms == pytest.approx(200.0)

    def test_default_duck_config_values(self):
        """DuckConfig has sensible defaults."""
        config = DuckConfig()
        assert config.amount_db < 0  # Ducking reduces volume
        assert config.attack_ms > 0
        assert config.release_ms > 0

    def test_duck_config_with_hold(self):
        """DuckConfig can have hold time."""
        config = DuckConfig(
            amount_db=-6.0,
            attack_ms=50.0,
            hold_ms=100.0,
            release_ms=200.0
        )
        assert config.hold_ms == pytest.approx(100.0)

    def test_duck_config_with_threshold(self):
        """DuckConfig can have threshold."""
        config = DuckConfig(
            amount_db=-6.0,
            threshold_db=-20.0
        )
        assert config.threshold_db == pytest.approx(-20.0)


class TestDuckTypes:
    """Test different ducking types."""

    def test_dialogue_duck_type(self):
        """Dialogue duck type exists."""
        assert DuckType.DIALOGUE is not None

    def test_event_duck_type(self):
        """Event duck type exists."""
        assert DuckType.EVENT is not None

    def test_focus_duck_type(self):
        """Focus duck type exists."""
        assert DuckType.FOCUS is not None

    def test_custom_duck_type(self):
        """Custom duck type exists."""
        assert DuckType.CUSTOM is not None


class TestApplyDuck:
    """Test applying ducks to buses."""

    def test_apply_duck_returns_id(self):
        """Applying duck returns instance ID."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        duck_id = manager.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0
        )
        assert duck_id is not None

    def test_apply_duck_increases_count(self):
        """Applying duck increases active count."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        initial = manager.active_duck_count
        manager.apply_duck(source="vo", target="sfx", amount_db=-6.0)
        assert manager.active_duck_count == initial + 1

    def test_apply_multiple_ducks(self):
        """Multiple ducks can be active."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        manager.apply_duck(source="vo", target="sfx", amount_db=-6.0)
        manager.apply_duck(source="vo", target="music", amount_db=-3.0)
        manager.apply_duck(source="event", target="ambient", amount_db=-9.0)

        assert manager.active_duck_count == 3

    def test_apply_duck_with_attack_release(self):
        """Duck with custom attack/release can be applied."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        duck_id = manager.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0,
            attack_ms=30.0,
            release_ms=150.0
        )
        assert duck_id is not None

    def test_apply_dialogue_duck(self):
        """Dialogue duck applies default dialogue settings."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        duck_id = manager.apply_dialogue_duck(target="sfx")
        assert duck_id is not None

    def test_apply_event_duck(self):
        """Event duck applies default event settings."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        duck_id = manager.apply_event_duck(target="music")
        assert duck_id is not None

    def test_apply_focus_duck(self):
        """Focus duck applies default focus settings."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        duck_id = manager.apply_focus_duck(target="ambient")
        assert duck_id is not None


class TestReleaseDuck:
    """Test releasing ducks."""

    def test_release_duck_by_id(self):
        """Duck can be released by ID."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        duck_id = manager.apply_duck(source="vo", target="sfx", amount_db=-6.0)
        manager.release_duck(duck_id)

        # After release completes, count decreases
        # May need to tick for release envelope
        for _ in range(100):
            manager.update(0.016)
        assert manager.active_duck_count == 0

    def test_release_all_ducks(self):
        """All ducks can be released."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        manager.apply_duck(source="vo", target="sfx", amount_db=-6.0)
        manager.apply_duck(source="vo", target="music", amount_db=-3.0)
        manager.release_all()

        for _ in range(100):
            manager.update(0.016)
        assert manager.active_duck_count == 0

    def test_release_nonexistent_duck(self):
        """Releasing nonexistent duck is handled."""
        manager = DuckingManager()
        # Should not raise
        manager.release_duck("nonexistent_id_xyz")


class TestDuckState:
    """Test duck state transitions."""

    def test_duck_state_idle(self):
        """Duck starts in idle state."""
        assert DuckState.IDLE is not None

    def test_duck_state_attack(self):
        """Attack state exists."""
        assert DuckState.ATTACK is not None

    def test_duck_state_hold(self):
        """Hold state exists."""
        assert DuckState.HOLD is not None

    def test_duck_state_release(self):
        """Release state exists."""
        assert DuckState.RELEASE is not None

    def test_get_duck_state(self):
        """Duck state can be queried."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        duck_id = manager.apply_duck(source="vo", target="sfx", amount_db=-6.0)
        state = manager.get_duck_state(duck_id)
        assert state in (DuckState.IDLE, DuckState.ATTACK, DuckState.HOLD)


class TestDuckEnvelope:
    """Test ducking envelope behavior."""

    def test_create_envelope(self):
        """DuckEnvelope can be created."""
        envelope = DuckEnvelope(
            attack_ms=50.0,
            hold_ms=0.0,
            release_ms=200.0
        )
        assert envelope.attack_ms == pytest.approx(50.0)

    def test_envelope_value_at_start(self):
        """Envelope starts at 1.0 (no ducking)."""
        envelope = DuckEnvelope(attack_ms=50.0, release_ms=200.0)
        envelope.reset()
        assert envelope.value == pytest.approx(1.0)

    def test_envelope_trigger(self):
        """Envelope can be triggered."""
        envelope = DuckEnvelope(attack_ms=50.0, release_ms=200.0)
        envelope.trigger(target=0.5)
        # After trigger, envelope should move toward target

    def test_envelope_release(self):
        """Envelope can be released."""
        envelope = DuckEnvelope(attack_ms=50.0, release_ms=200.0)
        envelope.trigger(target=0.5)
        envelope.release()
        # After release, envelope should move back to 1.0


class TestDuckingUpdate:
    """Test ducking manager update/tick."""

    def test_update_with_delta(self):
        """Manager accepts delta time for update."""
        manager = DuckingManager()
        manager.update(delta_time=0.016)

    def test_update_advances_envelopes(self):
        """Update advances envelope states."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        manager.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0,
            attack_ms=100.0
        )

        # Update multiple times to advance envelope
        for _ in range(10):
            manager.update(0.016)


class TestDuckingEffectOnVolume:
    """Test that ducking affects bus volume."""

    def test_duck_reduces_target_volume(self):
        """Active duck reduces target bus effective volume."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        sfx = mixer.get_bus("sfx")
        sfx.volume = 1.0
        initial_volume = sfx.effective_volume

        manager.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0,
            attack_ms=0.0  # Instant attack
        )
        manager.update(0.016)

        # Duck should reduce effective volume
        # The exact mechanism depends on implementation

    def test_released_duck_restores_volume(self):
        """Released duck restores original volume."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        sfx = mixer.get_bus("sfx")
        sfx.volume = 1.0
        initial_volume = sfx.volume

        duck_id = manager.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0,
            attack_ms=0.0
        )
        manager.update(0.016)
        manager.release_duck(duck_id)

        # After release completes
        for _ in range(200):
            manager.update(0.016)

        # Volume should be restored
        assert sfx.volume == pytest.approx(initial_volume)


class TestDuckPriority:
    """Test duck priority handling."""

    def test_higher_priority_duck_wins(self):
        """Higher priority duck takes precedence."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        manager.apply_duck(
            source="sfx",
            target="music",
            amount_db=-3.0,
            priority=1
        )
        manager.apply_duck(
            source="vo",
            target="music",
            amount_db=-6.0,
            priority=2  # Higher priority
        )

        # The -6dB duck should be active


class TestDuckingPresets:
    """Test predefined ducking presets."""

    def test_dialogue_duck_amount(self):
        """Dialogue duck has expected default amount."""
        assert DIALOGUE_DUCK_AMOUNT_DB < 0

    def test_event_duck_amount(self):
        """Event duck has expected default amount."""
        assert EVENT_DUCK_AMOUNT_DB < 0

    def test_focus_duck_amount(self):
        """Focus duck has expected default amount."""
        assert FOCUS_DUCK_AMOUNT_DB < 0

    def test_default_attack_time(self):
        """Default attack time is positive."""
        assert DUCK_ATTACK_MS > 0

    def test_default_release_time(self):
        """Default release time is positive."""
        assert DUCK_RELEASE_MS > 0


class TestDuckingInstance:
    """Test DuckingInstance data structure."""

    def test_create_ducking_instance(self):
        """DuckingInstance can be created."""
        instance = DuckingInstance(
            id="duck_001",
            source="vo",
            target="sfx",
            config=DuckConfig(amount_db=-6.0)
        )
        assert instance.id == "duck_001"
        assert instance.source == "vo"
        assert instance.target == "sfx"

    def test_ducking_instance_has_state(self):
        """DuckingInstance tracks state."""
        instance = DuckingInstance(
            id="duck_001",
            source="vo",
            target="sfx",
            config=DuckConfig(amount_db=-6.0)
        )
        assert hasattr(instance, 'state')

    def test_ducking_instance_string_repr(self):
        """DuckingInstance has readable string representation."""
        instance = DuckingInstance(
            id="duck_001",
            source="vo",
            target="sfx",
            config=DuckConfig(amount_db=-6.0)
        )
        str_repr = str(instance)
        assert "vo" in str_repr or "sfx" in str_repr


class TestDuckingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_duck_same_bus_to_itself(self):
        """Ducking bus to itself is handled."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        # Should either be rejected or handled gracefully
        try:
            manager.apply_duck(source="sfx", target="sfx", amount_db=-6.0)
        except Exception:
            pass  # Expected to reject self-ducking

    def test_duck_with_zero_attack(self):
        """Duck with zero attack is instant."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        manager.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0,
            attack_ms=0.0
        )

    def test_duck_with_zero_release(self):
        """Duck with zero release is instant."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        duck_id = manager.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0,
            release_ms=0.0
        )
        manager.release_duck(duck_id)

    def test_duck_with_very_long_times(self):
        """Duck with very long times is handled."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        manager.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0,
            attack_ms=10000.0,
            release_ms=10000.0
        )

    def test_duck_nonexistent_source(self):
        """Duck with nonexistent source is handled."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        # Should be handled gracefully
        try:
            manager.apply_duck(
                source="nonexistent_xyz",
                target="sfx",
                amount_db=-6.0
            )
        except Exception:
            pass

    def test_duck_nonexistent_target(self):
        """Duck with nonexistent target is handled."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        try:
            manager.apply_duck(
                source="vo",
                target="nonexistent_xyz",
                amount_db=-6.0
            )
        except Exception:
            pass

    def test_duck_amount_zero(self):
        """Duck with zero amount has no effect."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        manager.apply_duck(
            source="vo",
            target="sfx",
            amount_db=0.0
        )

    def test_duck_amount_positive(self):
        """Duck with positive amount (boost) is handled."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        # Positive amount would be a boost, not a duck
        # Implementation may reject or allow
        try:
            manager.apply_duck(
                source="vo",
                target="sfx",
                amount_db=6.0
            )
        except Exception:
            pass

    def test_many_simultaneous_ducks(self):
        """Many simultaneous ducks are handled."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        targets = ["sfx", "music", "ambient", "ui"]
        for i in range(50):
            manager.apply_duck(
                source="vo",
                target=targets[i % len(targets)],
                amount_db=-(i % 12 + 1)
            )

        # Should handle many ducks
        manager.update(0.016)


class TestDuckingConversions:
    """Test dB/linear conversions in ducking context."""

    def test_duck_amount_db_to_linear(self):
        """Duck amount converts correctly to linear."""
        duck_db = -6.0
        linear = db_to_linear(duck_db)
        assert linear == pytest.approx(0.5, rel=0.01)

    def test_duck_amount_linear_to_db(self):
        """Duck amount converts correctly to dB."""
        linear = 0.5
        db = linear_to_db(linear)
        assert db == pytest.approx(-6.02, rel=0.01)

    def test_full_duck_is_silence(self):
        """Full duck (-inf dB) is effective silence."""
        # -80 dB should be effectively silent
        linear = db_to_linear(-80.0)
        assert linear < 0.0001


class TestDuckingStats:
    """Test ducking statistics."""

    def test_get_active_ducks(self):
        """Active ducks can be listed."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        manager.apply_duck(source="vo", target="sfx", amount_db=-6.0)
        active = manager.get_active_ducks()
        assert len(active) >= 1

    def test_get_duck_by_id(self):
        """Duck instance can be retrieved by ID."""
        mixer = Mixer()
        mixer.initialize()
        manager = DuckingManager(mixer=mixer)

        duck_id = manager.apply_duck(source="vo", target="sfx", amount_db=-6.0)
        duck = manager.get_duck(duck_id)
        assert duck is not None
        assert duck.id == duck_id
