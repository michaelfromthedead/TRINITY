"""Whitebox tests for Sidechain Compression.

Tests internal implementation of:
- SidechainCompressor envelope following
- Gain reduction calculations with soft knee
- Attack/release timing
- SidechainManager multi-compressor coordination
- Key input level analysis
- Wet/dry mix
"""

import time
from unittest.mock import MagicMock

import pytest

from engine.audio.mixing.config import (
    MIN_VOLUME_DB,
    SIDECHAIN_ATTACK_MS,
    SIDECHAIN_KNEE_DB,
    SIDECHAIN_MAKEUP_GAIN_DB,
    SIDECHAIN_RATIO,
    SIDECHAIN_RELEASE_MS,
    SIDECHAIN_THRESHOLD_DB,
    db_to_linear,
    linear_to_db,
)
from engine.audio.mixing.mix_bus import BusType, MixBus
from engine.audio.mixing.sidechain import (
    CompressorState,
    SidechainCompressor,
    SidechainConfig,
    SidechainManager,
)


# =============================================================================
# CompressorState Tests
# =============================================================================


class TestCompressorState:
    """Test CompressorState enum."""

    def test_compressor_states_exist(self):
        """All compressor states exist."""
        assert CompressorState.IDLE.value == "idle"
        assert CompressorState.ATTACKING.value == "attacking"
        assert CompressorState.COMPRESSING.value == "compressing"
        assert CompressorState.RELEASING.value == "releasing"


# =============================================================================
# SidechainConfig Tests
# =============================================================================


class TestSidechainConfig:
    """Test SidechainConfig dataclass."""

    def test_default_values(self):
        """SidechainConfig has correct defaults."""
        config = SidechainConfig()
        assert config.name == ""
        assert config.key_bus is None
        assert config.target_bus is None
        assert config.threshold_db == SIDECHAIN_THRESHOLD_DB
        assert config.ratio == SIDECHAIN_RATIO
        assert config.attack_ms == SIDECHAIN_ATTACK_MS
        assert config.release_ms == SIDECHAIN_RELEASE_MS
        assert config.knee_db == SIDECHAIN_KNEE_DB
        assert config.makeup_gain_db == SIDECHAIN_MAKEUP_GAIN_DB
        assert config.enabled is True
        assert config.mix == 1.0
        assert config.id is not None

    def test_unique_ids(self):
        """Each SidechainConfig gets unique ID."""
        config1 = SidechainConfig()
        config2 = SidechainConfig()
        assert config1.id != config2.id

    def test_copy(self):
        """copy creates independent copy."""
        key = MixBus("vo", BusType.CATEGORY)
        target = MixBus("music", BusType.CATEGORY)

        config = SidechainConfig(
            name="vo_to_music",
            key_bus=key,
            target_bus=target,
            threshold_db=-15.0,
            ratio=8.0,
            attack_ms=5.0,
            release_ms=50.0,
            knee_db=3.0,
            makeup_gain_db=2.0,
            enabled=False,
            mix=0.5,
        )

        copy = config.copy()

        assert copy.id == config.id
        assert copy.name == "vo_to_music"
        assert copy.key_bus is key
        assert copy.target_bus is target
        assert copy.threshold_db == -15.0
        assert copy.ratio == 8.0
        assert copy.attack_ms == 5.0
        assert copy.release_ms == 50.0
        assert copy.knee_db == 3.0
        assert copy.makeup_gain_db == 2.0
        assert copy.enabled is False
        assert copy.mix == 0.5


# =============================================================================
# SidechainCompressor Basic Tests
# =============================================================================


class TestSidechainCompressorBasic:
    """Test SidechainCompressor basic functionality."""

    def test_initialization(self):
        """Compressor initializes from config."""
        key = MixBus("vo", BusType.CATEGORY)
        target = MixBus("music", BusType.CATEGORY)

        config = SidechainConfig(
            name="test",
            key_bus=key,
            target_bus=target,
            threshold_db=-20.0,
            ratio=4.0,
        )

        comp = SidechainCompressor(config)

        assert comp.config.name == "test"
        assert comp.config.threshold_db == -20.0
        assert comp.config.ratio == 4.0
        assert comp.state == CompressorState.IDLE
        assert comp.gain_reduction_db == 0.0

    def test_gain_reduction_linear_property(self):
        """gain_reduction_linear converts dB to linear."""
        config = SidechainConfig()
        comp = SidechainCompressor(config)

        # No reduction
        assert comp.gain_reduction_linear == 1.0

        # Set internal reduction
        comp._current_gain_reduction_db = -6.0
        assert comp.gain_reduction_linear == pytest.approx(0.5012, rel=1e-3)

    def test_output_gain_linear_property(self):
        """output_gain_linear includes makeup gain."""
        config = SidechainConfig(makeup_gain_db=3.0)
        comp = SidechainCompressor(config)

        comp._current_gain_reduction_db = -6.0

        # -6dB reduction + 3dB makeup = -3dB total
        expected = db_to_linear(-3.0)
        assert comp.output_gain_linear == pytest.approx(expected, rel=1e-3)

    def test_is_compressing(self):
        """is_compressing returns True when active."""
        config = SidechainConfig()
        comp = SidechainCompressor(config)

        assert comp.is_compressing is False

        comp._current_gain_reduction_db = -3.0
        assert comp.is_compressing is True

    def test_set_key_level(self):
        """set_key_level stores key input level."""
        config = SidechainConfig()
        comp = SidechainCompressor(config)

        comp.set_key_level(-10.0)
        assert comp._key_level_db == -10.0


# =============================================================================
# SidechainCompressor Gain Reduction Tests
# =============================================================================


class TestSidechainCompressorGainReduction:
    """Test SidechainCompressor gain reduction calculations."""

    def test_no_reduction_below_threshold(self):
        """No reduction when key below threshold."""
        config = SidechainConfig(threshold_db=-20.0, ratio=4.0, knee_db=0.0)
        comp = SidechainCompressor(config)

        # Below threshold
        reduction = comp._calculate_gain_reduction(-30.0)
        assert reduction == 0.0

    def test_reduction_above_threshold_hard_knee(self):
        """Reduction above threshold with hard knee."""
        config = SidechainConfig(threshold_db=-20.0, ratio=4.0, knee_db=0.0)
        comp = SidechainCompressor(config)

        # 8dB above threshold
        reduction = comp._calculate_gain_reduction(-12.0)

        # With 4:1 ratio, 8dB overshoot becomes 2dB output
        # Reduction = 8 * (1 - 1/4) = 8 * 0.75 = 6dB
        expected = -6.0
        assert reduction == pytest.approx(expected, rel=0.1)

    def test_reduction_with_soft_knee(self):
        """Reduction with soft knee is gradual."""
        config = SidechainConfig(threshold_db=-20.0, ratio=4.0, knee_db=6.0)
        comp = SidechainCompressor(config)

        # At knee start (-23dB = threshold - knee/2)
        reduction_start = comp._calculate_gain_reduction(-23.0)
        assert reduction_start == 0.0

        # In knee region
        reduction_mid = comp._calculate_gain_reduction(-20.0)
        assert reduction_mid < 0.0  # Some reduction

        # Above knee end (-17dB = threshold + knee/2)
        reduction_above = comp._calculate_gain_reduction(-10.0)
        assert reduction_above < reduction_mid  # More reduction

    def test_no_reduction_with_unity_ratio(self):
        """No reduction with 1:1 ratio."""
        config = SidechainConfig(threshold_db=-20.0, ratio=1.0)
        comp = SidechainCompressor(config)

        reduction = comp._calculate_gain_reduction(-10.0)
        assert reduction == 0.0


# =============================================================================
# SidechainCompressor Update Tests
# =============================================================================


class TestSidechainCompressorUpdate:
    """Test SidechainCompressor update/envelope logic."""

    def test_update_disabled_returns_unity(self):
        """Update when disabled returns 1.0."""
        config = SidechainConfig(enabled=False)
        comp = SidechainCompressor(config)

        comp.set_key_level(-10.0)
        gain = comp.update(0.1)

        assert gain == 1.0
        assert comp.gain_reduction_db == 0.0

    def test_update_attack_phase(self):
        """Update with key above threshold enters attack."""
        config = SidechainConfig(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=100.0,
            knee_db=0.0,
        )
        comp = SidechainCompressor(config)

        comp.set_key_level(-10.0)  # 10dB above threshold
        comp.update(0.01)

        assert comp.state == CompressorState.ATTACKING
        assert comp.gain_reduction_db < 0.0

    def test_update_attack_progresses(self):
        """Attack phase progresses over time."""
        config = SidechainConfig(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=100.0,
            knee_db=0.0,
        )
        comp = SidechainCompressor(config)

        comp.set_key_level(-10.0)

        # First update
        comp.update(0.01)
        reduction1 = comp.gain_reduction_db

        # Second update
        comp.update(0.01)
        reduction2 = comp.gain_reduction_db

        # More reduction as attack progresses
        assert reduction2 < reduction1

    def test_update_release_phase(self):
        """Update with key below threshold enters release."""
        config = SidechainConfig(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=0.0,  # Instant attack
            release_ms=100.0,
            knee_db=0.0,
        )
        comp = SidechainCompressor(config)

        # Build up compression
        comp.set_key_level(-10.0)
        comp.update(0.1)

        # Key drops below threshold
        comp.set_key_level(-30.0)
        comp.update(0.01)

        assert comp.state == CompressorState.RELEASING

    def test_update_release_progresses(self):
        """Release phase reduces gain reduction over time."""
        config = SidechainConfig(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=0.0,
            release_ms=100.0,
            knee_db=0.0,
        )
        comp = SidechainCompressor(config)

        # Full compression
        comp.set_key_level(-10.0)
        comp.update(0.1)
        full_reduction = comp.gain_reduction_db

        # Start release
        comp.set_key_level(-30.0)
        comp.update(0.05)

        # Should have less reduction
        assert comp.gain_reduction_db > full_reduction

    def test_update_zero_attack_time(self):
        """Zero attack time applies compression instantly."""
        config = SidechainConfig(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=0.0,
            knee_db=0.0,
        )
        comp = SidechainCompressor(config)

        comp.set_key_level(-10.0)
        comp.update(0.001)

        # Should be at target reduction
        expected_reduction = comp._calculate_gain_reduction(-10.0)
        assert comp.gain_reduction_db == pytest.approx(expected_reduction, rel=0.1)

    def test_update_zero_release_time(self):
        """Zero release time removes compression instantly."""
        config = SidechainConfig(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=0.0,
            release_ms=0.0,
            knee_db=0.0,
        )
        comp = SidechainCompressor(config)

        # Full compression
        comp.set_key_level(-10.0)
        comp.update(0.1)

        # Release
        comp.set_key_level(-30.0)
        comp.update(0.001)

        # Should be fully released
        assert comp.gain_reduction_db == pytest.approx(0.0, abs=0.1)

    def test_update_applies_mix(self):
        """Update applies wet/dry mix."""
        config = SidechainConfig(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=0.0,
            knee_db=0.0,
            mix=0.5,  # 50% wet
        )
        comp = SidechainCompressor(config)

        comp.set_key_level(-10.0)
        gain = comp.update(0.1)

        # With 50% mix, effect is halved
        full_reduction = comp._calculate_gain_reduction(-10.0)
        expected = db_to_linear(full_reduction * 0.5)
        assert gain == pytest.approx(expected, rel=0.1)

    def test_reset(self):
        """Reset clears compressor state."""
        config = SidechainConfig()
        comp = SidechainCompressor(config)

        comp.set_key_level(-10.0)
        comp.update(0.1)

        comp.reset()

        assert comp.state == CompressorState.IDLE
        assert comp.gain_reduction_db == 0.0
        assert comp._key_level_db == MIN_VOLUME_DB

    def test_get_stats(self):
        """get_stats returns compressor info."""
        config = SidechainConfig()
        comp = SidechainCompressor(config)

        comp.set_key_level(-10.0)
        comp.update(0.1)

        stats = comp.get_stats()

        assert "state" in stats
        assert "key_level_db" in stats
        assert "gain_reduction_db" in stats
        assert "target_reduction_db" in stats
        assert "is_compressing" in stats


# =============================================================================
# SidechainManager Tests
# =============================================================================


class TestSidechainManager:
    """Test SidechainManager."""

    def test_create_compressor(self):
        """Create sidechain compressor."""
        manager = SidechainManager()

        config = SidechainConfig(name="test_comp")
        comp = manager.create_compressor(config)

        assert comp is not None
        assert comp.config.name == "test_comp"

    def test_remove_compressor(self):
        """Remove compressor."""
        manager = SidechainManager()

        config = SidechainConfig(name="test")
        manager.create_compressor(config)

        result = manager.remove_compressor(config.id)

        assert result is True
        assert manager.get_compressor(config.id) is None

    def test_remove_compressor_not_found(self):
        """remove_compressor returns False if not found."""
        manager = SidechainManager()

        result = manager.remove_compressor("nonexistent")
        assert result is False

    def test_get_compressor(self):
        """Get compressor by ID."""
        manager = SidechainManager()

        config = SidechainConfig(name="test")
        comp = manager.create_compressor(config)

        found = manager.get_compressor(config.id)

        assert found is comp

    def test_get_compressor_not_found(self):
        """get_compressor returns None if not found."""
        manager = SidechainManager()

        result = manager.get_compressor("nonexistent")
        assert result is None

    def test_get_compressors_for_target(self):
        """Get compressors affecting a target bus."""
        manager = SidechainManager()
        target = MixBus("music", BusType.CATEGORY)
        other = MixBus("sfx", BusType.CATEGORY)

        config1 = SidechainConfig(target_bus=target)
        config2 = SidechainConfig(target_bus=other)

        manager.create_compressor(config1)
        manager.create_compressor(config2)

        comps = manager.get_compressors_for_target(target)

        assert len(comps) == 1
        assert comps[0].config.target_bus is target

    def test_get_compressors_with_key(self):
        """Get compressors using a key input."""
        manager = SidechainManager()
        key = MixBus("vo", BusType.CATEGORY)
        other = MixBus("sfx", BusType.CATEGORY)

        config1 = SidechainConfig(key_bus=key)
        config2 = SidechainConfig(key_bus=other)

        manager.create_compressor(config1)
        manager.create_compressor(config2)

        comps = manager.get_compressors_with_key(key)

        assert len(comps) == 1
        assert comps[0].config.key_bus is key


# =============================================================================
# SidechainManager Update Tests
# =============================================================================


class TestSidechainManagerUpdate:
    """Test SidechainManager update logic."""

    def test_update_updates_all_compressors(self):
        """Update updates all enabled compressors."""
        manager = SidechainManager()
        target1 = MixBus("music", BusType.CATEGORY)
        target2 = MixBus("ambient", BusType.CATEGORY)

        config1 = SidechainConfig(
            target_bus=target1,
            threshold_db=-20.0,
        )
        config2 = SidechainConfig(
            target_bus=target2,
            threshold_db=-20.0,
        )

        comp1 = manager.create_compressor(config1)
        comp2 = manager.create_compressor(config2)

        comp1.set_key_level(-10.0)
        comp2.set_key_level(-10.0)

        manager.update(0.1)

        # Both should have gain tracked
        assert manager.get_gain(target1) < 1.0
        assert manager.get_gain(target2) < 1.0

    def test_update_disabled_compressors_ignored(self):
        """Update ignores disabled compressors."""
        manager = SidechainManager()
        target = MixBus("music", BusType.CATEGORY)

        config = SidechainConfig(
            target_bus=target,
            threshold_db=-20.0,
            enabled=False,
        )
        comp = manager.create_compressor(config)
        comp.set_key_level(-10.0)

        manager.update(0.1)

        # Should not be compressed
        assert manager.get_gain(target) == 1.0

    def test_update_stacks_multiple_compressors(self):
        """Multiple compressors on same target stack gains."""
        manager = SidechainManager()
        target = MixBus("music", BusType.CATEGORY)
        key1 = MixBus("vo", BusType.CATEGORY)
        key2 = MixBus("sfx", BusType.CATEGORY)

        config1 = SidechainConfig(
            key_bus=key1,
            target_bus=target,
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=0.0,
        )
        config2 = SidechainConfig(
            key_bus=key2,
            target_bus=target,
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=0.0,
        )

        comp1 = manager.create_compressor(config1)
        comp2 = manager.create_compressor(config2)

        comp1.set_key_level(-10.0)
        comp2.set_key_level(-10.0)

        manager.update(0.1)

        # Both should contribute, resulting in more compression
        gain = manager.get_gain(target)
        assert gain < 1.0

    def test_analyze_key_levels(self):
        """Analyze key levels updates compressors."""
        manager = SidechainManager()
        key = MixBus("vo", BusType.CATEGORY)
        target = MixBus("music", BusType.CATEGORY)

        config = SidechainConfig(
            key_bus=key,
            target_bus=target,
            threshold_db=-20.0,
            attack_ms=0.0,
        )
        manager.create_compressor(config)

        manager.analyze_key_levels({key.id: -10.0})
        manager.update(0.1)

        assert manager.get_gain(target) < 1.0


# =============================================================================
# SidechainManager Gain Queries Tests
# =============================================================================


class TestSidechainManagerGainQueries:
    """Test SidechainManager gain query methods."""

    def test_get_gain(self):
        """Get compression gain for a bus."""
        manager = SidechainManager()
        target = MixBus("music", BusType.CATEGORY)

        config = SidechainConfig(
            target_bus=target,
            threshold_db=-20.0,
            attack_ms=0.0,
        )
        comp = manager.create_compressor(config)
        comp.set_key_level(-10.0)

        manager.update(0.1)

        gain = manager.get_gain(target)
        assert gain < 1.0

    def test_get_gain_no_compression(self):
        """get_gain returns 1.0 if no compression."""
        manager = SidechainManager()
        target = MixBus("music", BusType.CATEGORY)

        gain = manager.get_gain(target)
        assert gain == 1.0

    def test_get_gain_db(self):
        """Get compression gain in dB."""
        manager = SidechainManager()
        target = MixBus("music", BusType.CATEGORY)

        config = SidechainConfig(
            target_bus=target,
            threshold_db=-20.0,
            attack_ms=0.0,
        )
        comp = manager.create_compressor(config)
        comp.set_key_level(-10.0)

        manager.update(0.1)

        gain_db = manager.get_gain_db(target)
        assert gain_db < 0.0  # Negative dB = compression


# =============================================================================
# SidechainManager Callback Tests
# =============================================================================


class TestSidechainManagerCallbacks:
    """Test SidechainManager callbacks."""

    def test_on_compression_change(self):
        """Callback called on compression changes."""
        manager = SidechainManager()
        target = MixBus("music", BusType.CATEGORY)

        callback = MagicMock()
        manager.on_compression_change(callback)

        config = SidechainConfig(
            target_bus=target,
            threshold_db=-20.0,
            attack_ms=0.0,
        )
        comp = manager.create_compressor(config)
        comp.set_key_level(-10.0)

        manager.update(0.1)

        callback.assert_called()

    def test_remove_callback(self):
        """Remove compression change callback."""
        manager = SidechainManager()

        callback = MagicMock()
        manager.on_compression_change(callback)

        result = manager.remove_callback(callback)

        assert result is True

    def test_remove_callback_not_found(self):
        """remove_callback returns False if not found."""
        manager = SidechainManager()

        callback = MagicMock()
        result = manager.remove_callback(callback)

        assert result is False


# =============================================================================
# SidechainManager State Management Tests
# =============================================================================


class TestSidechainManagerState:
    """Test SidechainManager state management."""

    def test_reset_all(self):
        """Reset all compressors."""
        manager = SidechainManager()
        target = MixBus("music", BusType.CATEGORY)

        config = SidechainConfig(
            target_bus=target,
            threshold_db=-20.0,
            attack_ms=0.0,
        )
        comp = manager.create_compressor(config)
        comp.set_key_level(-10.0)
        manager.update(0.1)

        manager.reset_all()

        assert manager.get_gain(target) == 1.0
        assert comp.state == CompressorState.IDLE

    def test_clear(self):
        """Clear all compressors."""
        manager = SidechainManager()

        config = SidechainConfig(name="test")
        manager.create_compressor(config)

        manager.clear()

        assert manager.get_compressor(config.id) is None

    def test_get_state(self):
        """Get sidechain state for debugging."""
        manager = SidechainManager()
        target = MixBus("music", BusType.CATEGORY)

        config = SidechainConfig(name="test", target_bus=target)
        manager.create_compressor(config)

        state = manager.get_state()

        assert "compressors" in state
        assert "bus_gains" in state
        assert config.id in state["compressors"]

    def test_repr(self):
        """repr shows useful info."""
        manager = SidechainManager()
        target = MixBus("music", BusType.CATEGORY)

        config = SidechainConfig(
            target_bus=target,
            threshold_db=-20.0,
            attack_ms=0.0,
        )
        comp = manager.create_compressor(config)
        comp.set_key_level(-10.0)
        manager.update(0.1)

        repr_str = repr(manager)

        assert "SidechainManager" in repr_str
        assert "compressors=" in repr_str
        assert "active=" in repr_str
