"""Whitebox tests for Engine Audio Mixing: sidechain_bridge module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from engine.audio.mixing.config import (
    SIDECHAIN_ATTACK_MS,
    SIDECHAIN_KNEE_DB,
    SIDECHAIN_MAKEUP_GAIN_DB,
    SIDECHAIN_RATIO,
    SIDECHAIN_RELEASE_MS,
    SIDECHAIN_THRESHOLD_DB,
)
from engine.audio.mixing.mix_bus import MixBus
from engine.audio.mixing.sidechain import SidechainCompressor, SidechainConfig, SidechainManager
from engine.audio.mixing.sidechain_bridge import (
    SidechainBridgeConfig,
    apply_sidechain,
    create_compressor_config,
    extract_sidechain_config,
    has_sidechain,
    register_sidechain,
)


# ============================================================================
# TestSidechainBridgeConfig
# ============================================================================

class TestSidechainBridgeConfig:
    """SidechainBridgeConfig dataclass construction and default values."""

    def test_defaults(self):
        """Default config uses system-wide sidechain defaults."""
        cfg = SidechainBridgeConfig()
        assert cfg.source_bus == ""
        assert cfg.attack == SIDECHAIN_ATTACK_MS / 1000.0
        assert cfg.release == SIDECHAIN_RELEASE_MS / 1000.0
        assert cfg.ratio == SIDECHAIN_RATIO
        assert cfg.threshold_db == SIDECHAIN_THRESHOLD_DB
        assert cfg.knee_db == SIDECHAIN_KNEE_DB
        assert cfg.makeup_gain_db == SIDECHAIN_MAKEUP_GAIN_DB
        assert cfg.mix == 1.0

    def test_explicit_values(self):
        """Explicit constructor values are stored."""
        cfg = SidechainBridgeConfig(
            source_bus="kick",
            attack=0.005,
            release=0.2,
            ratio=6.0,
            threshold_db=-24.0,
            knee_db=3.0,
            makeup_gain_db=2.0,
            mix=0.8,
        )
        assert cfg.source_bus == "kick"
        assert cfg.attack == 0.005
        assert cfg.release == 0.2
        assert cfg.ratio == 6.0
        assert cfg.threshold_db == -24.0
        assert cfg.knee_db == 3.0
        assert cfg.makeup_gain_db == 2.0
        assert cfg.mix == 0.8

    def test_dataclass_attributes(self):
        """SidechainBridgeConfig has all expected attributes."""
        cfg = SidechainBridgeConfig()
        assert hasattr(cfg, "source_bus")
        assert hasattr(cfg, "attack")
        assert hasattr(cfg, "release")
        assert hasattr(cfg, "ratio")
        assert hasattr(cfg, "threshold_db")
        assert hasattr(cfg, "knee_db")
        assert hasattr(cfg, "makeup_gain_db")
        assert hasattr(cfg, "mix")


# ============================================================================
# TestExtractSidechainConfig
# ============================================================================

class TestExtractSidechainConfig:
    """extract_sidechain_config() reads decorator attrs or returns defaults."""

    def test_decorated_class(self):
        """Extract returns config matching decorator attrs."""

        class Sidechained:
            _sidechain = True
            _sidechain_source_bus = "kick"
            _sidechain_attack = 0.005
            _sidechain_release = 0.2
            _sidechain_ratio = 6.0

        cfg = extract_sidechain_config(Sidechained)
        assert cfg.source_bus == "kick"
        assert cfg.attack == 0.005
        assert cfg.release == 0.2
        assert cfg.ratio == 6.0

    def test_undecorated_class(self):
        """Extract returns defaults for undecorated classes."""

        class Plain:
            pass

        cfg = extract_sidechain_config(Plain)
        assert cfg.source_bus == ""
        assert cfg.attack == SIDECHAIN_ATTACK_MS / 1000.0
        assert cfg.release == SIDECHAIN_RELEASE_MS / 1000.0
        assert cfg.ratio == SIDECHAIN_RATIO

    def test_partial_attrs(self):
        """Missing sub-attrs fall back to defaults."""

        class Partial:
            _sidechain = True
            # no _sidechain_source_bus etc.

        cfg = extract_sidechain_config(Partial)
        assert cfg.source_bus == ""
        assert cfg.attack == SIDECHAIN_ATTACK_MS / 1000.0
        assert cfg.release == SIDECHAIN_RELEASE_MS / 1000.0
        assert cfg.ratio == SIDECHAIN_RATIO

    def test_returns_sidechain_bridge_config(self):
        """Extract always returns a SidechainBridgeConfig instance."""

        class Any:
            pass

        cfg = extract_sidechain_config(Any)
        assert isinstance(cfg, SidechainBridgeConfig)

    def test_decorated_class_all_params(self):
        """Extract returns full config when all attrs present."""

        class Full:
            _sidechain = True
            _sidechain_source_bus = "kick"
            _sidechain_attack = 0.003
            _sidechain_release = 0.15
            _sidechain_ratio = 8.0
            _sidechain_threshold_db = -30.0
            _sidechain_knee_db = 4.0
            _sidechain_makeup_gain_db = 1.5
            _sidechain_mix = 0.9

        cfg = extract_sidechain_config(Full)
        assert cfg.source_bus == "kick"
        assert cfg.attack == 0.003
        assert cfg.release == 0.15
        assert cfg.ratio == 8.0
        assert cfg.threshold_db == -30.0
        assert cfg.knee_db == 4.0
        assert cfg.makeup_gain_db == 1.5
        assert cfg.mix == 0.9

    def test_undecorated_supports_all_defaults(self):
        """Undecorated class yields defaults for every field."""

        class Plain:
            pass

        cfg = extract_sidechain_config(Plain)
        # Full matrix of defaults being verified
        assert cfg.source_bus == ""
        assert abs(cfg.attack - SIDECHAIN_ATTACK_MS / 1000.0) < 1e-9
        assert abs(cfg.release - SIDECHAIN_RELEASE_MS / 1000.0) < 1e-9
        assert abs(cfg.ratio - SIDECHAIN_RATIO) < 1e-9
        assert abs(cfg.threshold_db - SIDECHAIN_THRESHOLD_DB) < 1e-9
        assert abs(cfg.knee_db - SIDECHAIN_KNEE_DB) < 1e-9
        assert abs(cfg.makeup_gain_db - SIDECHAIN_MAKEUP_GAIN_DB) < 1e-9
        assert abs(cfg.mix - 1.0) < 1e-9


# ============================================================================
# TestHasSidechain
# ============================================================================

class TestHasSidechain:
    """has_sidechain() detection."""

    def test_decorated_returns_true(self):
        class Decorated:
            _sidechain = True

        assert has_sidechain(Decorated) is True

    def test_undecorated_returns_false(self):
        class Plain:
            pass

        assert has_sidechain(Plain) is False

    def test_explicit_false(self):
        class Explicit:
            _sidechain = False

        assert has_sidechain(Explicit) is False

    def test_falsy_value_zero(self):
        class ZeroVal:
            _sidechain = 0

        assert has_sidechain(ZeroVal) is False

    def test_truthy_non_bool(self):
        class StrVal:
            _sidechain = "yes"

        assert has_sidechain(StrVal) is True


# ============================================================================
# TestCreateCompressorConfig
# ============================================================================

class TestCreateCompressorConfig:
    """create_compressor_config() produces a SidechainConfig from bridge config."""

    @pytest.fixture
    def key_bus(self):
        bus = MagicMock(spec=MixBus)
        bus.name = "kick"
        bus.id = "bus_kick"
        return bus

    @pytest.fixture
    def target_bus(self):
        bus = MagicMock(spec=MixBus)
        bus.name = "bass"
        bus.id = "bus_bass"
        return bus

    def test_creates_config_with_resolved_buses(self, key_bus, target_bus):
        bridge = SidechainBridgeConfig(
            source_bus="kick",
            attack=0.005,
            release=0.2,
            ratio=6.0,
            threshold_db=-24.0,
            knee_db=3.0,
            makeup_gain_db=2.0,
            mix=0.8,
        )
        sc_config = create_compressor_config(bridge, key_bus, target_bus, name="kick->bass")

        assert sc_config.key_bus is key_bus
        assert sc_config.target_bus is target_bus
        assert sc_config.threshold_db == -24.0
        assert sc_config.ratio == 6.0
        assert sc_config.attack_ms == 5.0  # 0.005 * 1000
        assert sc_config.release_ms == 200.0  # 0.2 * 1000
        assert sc_config.knee_db == 3.0
        assert sc_config.makeup_gain_db == 2.0
        assert sc_config.mix == 0.8
        assert sc_config.enabled is True
        assert "kick" in sc_config.name and "bass" in sc_config.name

    def test_default_bridge_config(self, key_bus, target_bus):
        bridge = SidechainBridgeConfig(source_bus="kick")
        sc_config = create_compressor_config(bridge, key_bus, target_bus)

        assert sc_config.threshold_db == SIDECHAIN_THRESHOLD_DB
        assert sc_config.ratio == SIDECHAIN_RATIO
        assert sc_config.attack_ms == SIDECHAIN_ATTACK_MS
        assert sc_config.release_ms == SIDECHAIN_RELEASE_MS
        assert sc_config.knee_db == SIDECHAIN_KNEE_DB
        assert sc_config.mix == 1.0

    def test_returns_sidechain_config_instance(self, key_bus, target_bus):
        bridge = SidechainBridgeConfig(source_bus="kick")
        sc_config = create_compressor_config(bridge, key_bus, target_bus)
        assert isinstance(sc_config, SidechainConfig)

    def test_default_name_generated(self, key_bus, target_bus):
        bridge = SidechainBridgeConfig(source_bus="kick")
        sc_config = create_compressor_config(bridge, key_bus, target_bus)
        assert "sidechain:" in sc_config.name
        assert "kick" in sc_config.name
        assert "bass" in sc_config.name

    def test_custom_name(self, key_bus, target_bus):
        bridge = SidechainBridgeConfig(source_bus="kick")
        sc_config = create_compressor_config(bridge, key_bus, target_bus, name="custom-name")
        assert sc_config.name == "custom-name"


# ============================================================================
# TestRegisterSidechain
# ============================================================================

class TestRegisterSidechain:
    """register_sidechain() creates compressor via SidechainManager."""

    @pytest.fixture
    def sidechain_manager(self):
        mgr = MagicMock(spec=SidechainManager)
        return mgr

    @pytest.fixture
    def key_bus(self):
        bus = MagicMock(spec=MixBus)
        bus.name = "kick"
        bus.id = "bus_kick"
        return bus

    @pytest.fixture
    def target_bus(self):
        bus = MagicMock(spec=MixBus)
        bus.name = "bass"
        bus.id = "bus_bass"
        return bus

    def test_creates_compressor_with_config(self, sidechain_manager, key_bus, target_bus):
        bridge = SidechainBridgeConfig(source_bus="kick", ratio=8.0)
        compressor = register_sidechain(sidechain_manager, bridge, key_bus, target_bus)

        sidechain_manager.create_compressor.assert_called_once()
        config_arg = sidechain_manager.create_compressor.call_args[0][0]
        assert isinstance(config_arg, SidechainConfig)
        assert config_arg.ratio == 8.0
        assert config_arg.key_bus is key_bus
        assert config_arg.target_bus is target_bus

    def test_returns_compressor(self, sidechain_manager, key_bus, target_bus):
        fake_compressor = MagicMock(spec=SidechainCompressor)
        sidechain_manager.create_compressor.return_value = fake_compressor

        bridge = SidechainBridgeConfig(source_bus="kick")
        result = register_sidechain(sidechain_manager, bridge, key_bus, target_bus)
        assert result is fake_compressor

    def test_custom_name_passed_through(self, sidechain_manager, key_bus, target_bus):
        bridge = SidechainBridgeConfig(source_bus="kick")
        register_sidechain(sidechain_manager, bridge, key_bus, target_bus, name="my-comp")
        config_arg = sidechain_manager.create_compressor.call_args[0][0]
        assert config_arg.name == "my-comp"

    def test_empty_source_bus_still_registers(self, sidechain_manager, key_bus, target_bus):
        """Even with empty source_bus, register_sidechain creates a compressor."""
        bridge = SidechainBridgeConfig()  # source_bus=""
        register_sidechain(sidechain_manager, bridge, key_bus, target_bus)
        sidechain_manager.create_compressor.assert_called_once()


# ============================================================================
# TestApplySidechain
# ============================================================================

class TestApplySidechain:
    """apply_sidechain() combines extract + register in one call."""

    @pytest.fixture
    def sidechain_manager(self):
        mgr = MagicMock(spec=SidechainManager)
        return mgr

    @pytest.fixture
    def key_bus(self):
        bus = MagicMock(spec=MixBus)
        bus.name = "kick"
        bus.id = "bus_kick"
        return bus

    @pytest.fixture
    def target_bus(self):
        bus = MagicMock(spec=MixBus)
        bus.name = "bass"
        bus.id = "bus_bass"
        return bus

    def test_decorated_component_creates_compressor(self, sidechain_manager, key_bus, target_bus):
        class Sidechained:
            _sidechain = True
            _sidechain_source_bus = "kick"
            _sidechain_attack = 0.005
            _sidechain_release = 0.2
            _sidechain_ratio = 6.0

        compressor = apply_sidechain(sidechain_manager, Sidechained, key_bus, target_bus)
        sidechain_manager.create_compressor.assert_called_once()
        config_arg = sidechain_manager.create_compressor.call_args[0][0]
        assert config_arg.ratio == 6.0
        assert compressor is not None

    def test_undecorated_component_returns_none(self, sidechain_manager, key_bus, target_bus):
        class Plain:
            pass

        result = apply_sidechain(sidechain_manager, Plain, key_bus, target_bus)
        assert result is None
        sidechain_manager.create_compressor.assert_not_called()

    def test_decorated_empty_source_bus_returns_none(self, sidechain_manager, key_bus, target_bus):
        class EmptySource:
            _sidechain = True
            _sidechain_source_bus = ""
            _sidechain_attack = 0.01
            _sidechain_release = 0.1
            _sidechain_ratio = 4.0

        result = apply_sidechain(sidechain_manager, EmptySource, key_bus, target_bus)
        assert result is None
        sidechain_manager.create_compressor.assert_not_called()

    def test_custom_name_passed_to_compressor(self, sidechain_manager, key_bus, target_bus):
        class Named:
            _sidechain = True
            _sidechain_source_bus = "kick"
            _sidechain_attack = 0.01
            _sidechain_release = 0.1
            _sidechain_ratio = 4.0

        apply_sidechain(sidechain_manager, Named, key_bus, target_bus, name="kick-to-bass")
        config_arg = sidechain_manager.create_compressor.call_args[0][0]
        assert config_arg.name == "kick-to-bass"

    def test_maps_attack_release_to_ms(self, sidechain_manager, key_bus, target_bus):
        class Mapped:
            _sidechain = True
            _sidechain_source_bus = "kick"
            _sidechain_attack = 0.003
            _sidechain_release = 0.15
            _sidechain_ratio = 4.0

        apply_sidechain(sidechain_manager, Mapped, key_bus, target_bus)
        config_arg = sidechain_manager.create_compressor.call_args[0][0]
        assert config_arg.attack_ms == 3.0  # 0.003 * 1000
        assert config_arg.release_ms == 150.0  # 0.15 * 1000

    def test_real_sidechain_manager_integration(self):
        """Integration-style: real SidechainManager creates a real compressor."""
        mgr = SidechainManager()
        key_bus = MagicMock(spec=MixBus)
        key_bus.name = "kick"
        key_bus.id = "bus_kick"
        target_bus = MagicMock(spec=MixBus)
        target_bus.name = "bass"
        target_bus.id = "bus_bass"

        class Sidechained:
            _sidechain = True
            _sidechain_source_bus = "kick"
            _sidechain_attack = 0.005
            _sidechain_release = 0.2
            _sidechain_ratio = 6.0

        compressor = apply_sidechain(mgr, Sidechained, key_bus, target_bus, name="int-test")
        assert compressor is not None
        assert isinstance(compressor, SidechainCompressor)
        assert compressor.config.ratio == 6.0
        assert compressor.config.attack_ms == 5.0
        assert compressor.config.release_ms == 200.0
        assert compressor.config.key_bus is key_bus
        assert compressor.config.target_bus is target_bus

        # Verify it's registered in the manager
        retrieved = mgr.get_compressor(compressor.config.id)
        assert retrieved is compressor

    def test_real_sidechain_manager_multiple_compressors(self):
        """Real SidechainManager supports multiple independent compressors."""
        mgr = SidechainManager()
        kick = MagicMock(spec=MixBus, name="kick")
        kick.name = "kick"
        kick.id = "bus_kick"
        bass = MagicMock(spec=MixBus, name="bass")
        bass.name = "bass"
        bass.id = "bus_bass"
        snare = MagicMock(spec=MixBus, name="snare")
        snare.name = "snare"
        snare.id = "bus_snare"
        pad = MagicMock(spec=MixBus, name="pad")
        pad.name = "pad"
        pad.id = "bus_pad"

        class KickToBass:
            _sidechain = True
            _sidechain_source_bus = "kick"
            _sidechain_attack = 0.005
            _sidechain_release = 0.2
            _sidechain_ratio = 6.0

        class SnareToPad:
            _sidechain = True
            _sidechain_source_bus = "snare"
            _sidechain_attack = 0.003
            _sidechain_release = 0.15
            _sidechain_ratio = 4.0

        c1 = apply_sidechain(mgr, KickToBass, kick, bass, name="kick->bass")
        c2 = apply_sidechain(mgr, SnareToPad, snare, pad, name="snare->pad")

        assert c1 is not None
        assert c2 is not None
        assert c1.config.ratio == 6.0
        assert c2.config.ratio == 4.0

        assert mgr.get_compressor(c1.config.id) is c1
        assert mgr.get_compressor(c2.config.id) is c2

        # Each returns the correct target
        assert mgr.get_compressors_for_target(bass) == [c1]
        assert mgr.get_compressors_for_target(pad) == [c2]


# ============================================================================
# Whitebox: SidechainBridgeConfig edge cases
# ============================================================================

class TestSidechainBridgeConfigEdgeCases:
    """Boundary and edge-case parameter values."""

    def test_zero_attack(self):
        """Zero attack time is accepted at the bridge level."""
        cfg = SidechainBridgeConfig(attack=0.0)
        assert cfg.attack == 0.0

    def test_zero_release(self):
        """Zero release time is accepted."""
        cfg = SidechainBridgeConfig(release=0.0)
        assert cfg.release == 0.0

    def test_zero_ratio(self):
        """Zero ratio is stored (validation is in the compressor)."""
        cfg = SidechainBridgeConfig(ratio=0.0)
        assert cfg.ratio == 0.0

    def test_negative_attack(self):
        """Negative attack is stored (domain validation downstream)."""
        cfg = SidechainBridgeConfig(attack=-1.0)
        assert cfg.attack == -1.0

    def test_extreme_threshold_positive(self):
        """Large positive threshold is stored."""
        cfg = SidechainBridgeConfig(threshold_db=100.0)
        assert cfg.threshold_db == 100.0

    def test_extreme_threshold_negative(self):
        """Large negative threshold is stored."""
        cfg = SidechainBridgeConfig(threshold_db=-100.0)
        assert cfg.threshold_db == -100.0

    def test_mix_at_lower_bound(self):
        """mix=0.0 means no compression."""
        cfg = SidechainBridgeConfig(mix=0.0)
        assert cfg.mix == 0.0

    def test_mix_at_upper_bound(self):
        """mix=1.0 means full compression."""
        cfg = SidechainBridgeConfig(mix=1.0)
        assert cfg.mix == 1.0

    def test_mix_beyond_bounds(self):
        """mix outside [0,1] is stored (downstream DSP validates)."""
        cfg = SidechainBridgeConfig(mix=1.5)
        assert cfg.mix == 1.5
        cfg = SidechainBridgeConfig(mix=-0.5)
        assert cfg.mix == -0.5


# ============================================================================
# Whitebox: extract_sidechain_config internal paths
# ============================================================================

class TestExtractSidechainConfigWhitebox:
    """Deep path coverage of extract_sidechain_config internals."""

    def test_non_class_int_argument(self):
        """Passing builtin type returns default config (no _sidechain attr)."""
        cfg = extract_sidechain_config(int)
        assert isinstance(cfg, SidechainBridgeConfig)
        assert cfg.source_bus == ""

    def test_sidechain_set_to_none(self):
        """_sidechain=None is falsy, returns defaults."""

        class WithNone:
            _sidechain = None

        cfg = extract_sidechain_config(WithNone)
        assert cfg.source_bus == ""

    def test_sidechain_truthy_int_one(self):
        """_sidechain=1 is truthy, reads other attrs."""

        class WithOne:
            _sidechain = 1
            _sidechain_source_bus = "kick"
            _sidechain_ratio = 6.0

        cfg = extract_sidechain_config(WithOne)
        assert cfg.source_bus == "kick"
        assert cfg.ratio == 6.0

    def test_decorated_all_getattr_fallbacks(self):
        """When _sidechain=True but no other attrs, every field uses default."""

        class BarelyDecorated:
            _sidechain = True

        cfg = extract_sidechain_config(BarelyDecorated)
        assert cfg.source_bus == ""
        assert cfg.attack == SIDECHAIN_ATTACK_MS / 1000.0
        assert cfg.release == SIDECHAIN_RELEASE_MS / 1000.0
        assert cfg.ratio == SIDECHAIN_RATIO
        assert cfg.threshold_db == SIDECHAIN_THRESHOLD_DB
        assert cfg.knee_db == SIDECHAIN_KNEE_DB
        assert cfg.makeup_gain_db == SIDECHAIN_MAKEUP_GAIN_DB
        assert cfg.mix == 1.0

    def test_explicit_empty_source_bus_in_decorated(self):
        """_sidechain=True with source_bus='' returns empty source_bus."""

        class EmptySource:
            _sidechain = True
            _sidechain_source_bus = ""

        cfg = extract_sidechain_config(EmptySource)
        assert cfg.source_bus == ""

    def test_sidechain_false_ignores_all_other_attrs(self):
        """_sidechain=False returns all-defaults even if other attrs exist."""

        class FalseButHasAttrs:
            _sidechain = False
            _sidechain_source_bus = "kick"
            _sidechain_ratio = 6.0

        cfg = extract_sidechain_config(FalseButHasAttrs)
        assert cfg.source_bus == ""
        assert cfg.ratio == SIDECHAIN_RATIO

    def test_inherited_sidechain_from_parent(self):
        """Subclass inherits _sidechain attrs from parent."""

        class Parent:
            _sidechain = True
            _sidechain_source_bus = "kick"
            _sidechain_ratio = 8.0

        class Child(Parent):
            pass

        cfg = extract_sidechain_config(Child)
        assert cfg.source_bus == "kick"
        assert cfg.ratio == 8.0

    def test_child_overrides_parent_attr(self):
        """Subclass can selectively override parent sidechain attrs."""

        class Parent:
            _sidechain = True
            _sidechain_source_bus = "kick"
            _sidechain_ratio = 4.0

        class Child(Parent):
            _sidechain_ratio = 10.0

        cfg = extract_sidechain_config(Child)
        assert cfg.source_bus == "kick"
        assert cfg.ratio == 10.0


# ============================================================================
# Whitebox: create_compressor_config internal paths
# ============================================================================

class TestCreateCompressorConfigWhitebox:
    """Deep path coverage of create_compressor_config internals."""

    @pytest.fixture
    def bridge(self):
        return SidechainBridgeConfig(source_bus="kick", attack=0.005, release=0.2, ratio=6.0)

    @pytest.fixture
    def key_bus(self):
        bus = MagicMock(spec=MixBus)
        bus.name = "kick"
        bus.id = "bus_kick"
        return bus

    @pytest.fixture
    def target_bus(self):
        bus = MagicMock(spec=MixBus)
        bus.name = "bass"
        bus.id = "bus_bass"
        return bus

    def test_none_key_bus_raises_in_default_name(self):
        """key_bus=None causes AttributeError when generating default name."""
        bridge = SidechainBridgeConfig(source_bus="kick")
        target = MagicMock(spec=MixBus)
        target.name = "bass"
        with pytest.raises(AttributeError):
            create_compressor_config(bridge, None, target)

    def test_none_target_bus_raises_in_default_name(self):
        """target_bus=None causes AttributeError in default name generation."""
        bridge = SidechainBridgeConfig(source_bus="kick")
        key = MagicMock(spec=MixBus)
        key.name = "kick"
        with pytest.raises(AttributeError):
            create_compressor_config(bridge, key, None)

    def test_attack_converts_seconds_to_ms(self, key_bus, target_bus):
        """attack (seconds) * 1000.0 produces attack_ms."""
        bridge = SidechainBridgeConfig(source_bus="kick", attack=0.00325)
        sc = create_compressor_config(bridge, key_bus, target_bus)
        assert sc.attack_ms == pytest.approx(3.25)

    def test_release_converts_seconds_to_ms(self, key_bus, target_bus):
        """release (seconds) * 1000.0 produces release_ms."""
        bridge = SidechainBridgeConfig(source_bus="kick", release=0.123)
        sc = create_compressor_config(bridge, key_bus, target_bus)
        assert sc.release_ms == pytest.approx(123.0)

    def test_enabled_field_hardcoded_true(self, key_bus, target_bus):
        """enabled is always True regardless of bridge config."""
        bridge = SidechainBridgeConfig(source_bus="kick")
        sc = create_compressor_config(bridge, key_bus, target_bus)
        assert sc.enabled is True

    def test_name_empty_triggers_default_pattern(self, key_bus, target_bus):
        """name='' triggers auto-generated 'sidechain:key->target'."""
        bridge = SidechainBridgeConfig(source_bus="kick")
        sc = create_compressor_config(bridge, key_bus, target_bus, name="")
        assert sc.name == "sidechain:kick->bass"

    def test_makeup_gain_propagated(self, key_bus, target_bus):
        """makeup_gain_db passes through from bridge config."""
        bridge = SidechainBridgeConfig(source_bus="kick", makeup_gain_db=3.5)
        sc = create_compressor_config(bridge, key_bus, target_bus)
        assert sc.makeup_gain_db == 3.5

    def test_knee_db_propagated(self, key_bus, target_bus):
        """knee_db passes through from bridge config."""
        bridge = SidechainBridgeConfig(source_bus="kick", knee_db=9.0)
        sc = create_compressor_config(bridge, key_bus, target_bus)
        assert sc.knee_db == 9.0

    def test_threshold_db_propagated(self, key_bus, target_bus):
        """threshold_db passes through from bridge config."""
        bridge = SidechainBridgeConfig(source_bus="kick", threshold_db=-30.0)
        sc = create_compressor_config(bridge, key_bus, target_bus)
        assert sc.threshold_db == -30.0


# ============================================================================
# Whitebox: register_sidechain internal delegation
# ============================================================================

class TestRegisterSidechainWhitebox:
    """Whitebox verification of register_sidechain internal call graph."""

    def test_delegates_to_create_compressor_config(self):
        """register_sidechain calls create_compressor_config internally."""
        mgr = MagicMock(spec=SidechainManager)
        mgr.create_compressor = MagicMock(return_value=MagicMock(spec=SidechainCompressor))
        bridge = SidechainBridgeConfig(source_bus="kick", ratio=5.0)
        key = MagicMock(spec=MixBus)
        key.name = "kick"
        target = MagicMock(spec=MixBus)
        target.name = "bass"

        result = register_sidechain(mgr, bridge, key, target, name="test")

        mgr.create_compressor.assert_called_once()
        config_arg = mgr.create_compressor.call_args[0][0]
        assert isinstance(config_arg, SidechainConfig)
        assert config_arg.ratio == 5.0

    def test_errors_propagate_from_manager(self):
        """Exceptions from SidechainManager.create_compressor propagate."""
        mgr = MagicMock(spec=SidechainManager)
        mgr.create_compressor = MagicMock(side_effect=RuntimeError("manager full"))
        bridge = SidechainBridgeConfig(source_bus="kick")
        key = MagicMock(spec=MixBus)
        key.name = "kick"
        target = MagicMock(spec=MixBus)
        target.name = "bass"

        with pytest.raises(RuntimeError, match="manager full"):
            register_sidechain(mgr, bridge, key, target)

    def test_compressor_id_is_uuid4(self):
        """Created compressor gets a valid UUID4 string as its id."""
        mgr = SidechainManager()
        bridge = SidechainBridgeConfig(source_bus="kick", attack=0.005)
        key = MagicMock(spec=MixBus)
        key.name = "kick"
        target = MagicMock(spec=MixBus)
        target.name = "bass"

        compressor = register_sidechain(mgr, bridge, key, target)
        assert len(compressor.config.id) == 36
        assert compressor.config.id.count("-") == 4


# ============================================================================
# Whitebox: apply_sidechain internal call verification
# ============================================================================

class TestApplySidechainWhitebox:
    """Whitebox verification of apply_sidechain's internal call graph."""

    @pytest.fixture
    def sidechain_manager(self):
        return MagicMock(spec=SidechainManager)

    @pytest.fixture
    def key_bus(self):
        bus = MagicMock(spec=MixBus)
        bus.name = "kick"
        bus.id = "bus_kick"
        return bus

    @pytest.fixture
    def target_bus(self):
        bus = MagicMock(spec=MixBus)
        bus.name = "bass"
        bus.id = "bus_bass"
        return bus

    def test_extract_config_called_internally(self, sidechain_manager, key_bus, target_bus):
        """apply_sidechain calls extract_sidechain_config on the component."""
        class Decorated:
            _sidechain = True
            _sidechain_source_bus = "kick"

        with patch(
            "engine.audio.mixing.sidechain_bridge.extract_sidechain_config",
            wraps=extract_sidechain_config,
        ) as mock_extract:
            apply_sidechain(sidechain_manager, Decorated, key_bus, target_bus)
            mock_extract.assert_called_once_with(Decorated)

    def test_register_not_called_when_no_source(self, sidechain_manager, key_bus, target_bus):
        """register_sidechain is NOT called when no source_bus (whitebox: guard at L247)."""
        class NoSource:
            _sidechain = True

        with patch(
            "engine.audio.mixing.sidechain_bridge.register_sidechain",
        ) as mock_register:
            result = apply_sidechain(sidechain_manager, NoSource, key_bus, target_bus)
            assert result is None
            mock_register.assert_not_called()

    def test_register_called_when_source_set(self, sidechain_manager, key_bus, target_bus):
        """register_sidechain IS called when source_bus is non-empty."""
        class WithSource:
            _sidechain = True
            _sidechain_source_bus = "kick"

        with patch(
            "engine.audio.mixing.sidechain_bridge.register_sidechain",
        ) as mock_register:
            apply_sidechain(sidechain_manager, WithSource, key_bus, target_bus)
            mock_register.assert_called_once()

    def test_whitespace_source_bus_passes_guard(self, sidechain_manager, key_bus, target_bus):
        """source_bus=' ' (whitespace) is truthy and passes the guard."""
        class WhitespaceSource:
            _sidechain = True
            _sidechain_source_bus = " "

        with patch(
            "engine.audio.mixing.sidechain_bridge.register_sidechain",
        ) as mock_register:
            apply_sidechain(sidechain_manager, WhitespaceSource, key_bus, target_bus)
            mock_register.assert_called_once()

    def test_extract_config_passed_to_register(self, sidechain_manager, key_bus, target_bus):
        """The SidechainBridgeConfig from extract is forwarded to register_sidechain."""
        class Custom:
            _sidechain = True
            _sidechain_source_bus = "snare"
            _sidechain_ratio = 10.0

        with patch(
            "engine.audio.mixing.sidechain_bridge.register_sidechain",
        ) as mock_register:
            apply_sidechain(sidechain_manager, Custom, key_bus, target_bus)
            mock_register.assert_called_once()
            _args, _kwargs = mock_register.call_args
            passed_config = _args[1]
            assert passed_config.source_bus == "snare"
            assert passed_config.ratio == 10.0

    def test_name_parameter_forwards_to_register(self, sidechain_manager, key_bus, target_bus):
        """The name parameter is forwarded to register_sidechain."""
        class NamedComp:
            _sidechain = True
            _sidechain_source_bus = "kick"

        with patch(
            "engine.audio.mixing.sidechain_bridge.register_sidechain",
        ) as mock_register:
            apply_sidechain(sidechain_manager, NamedComp, key_bus, target_bus, name="kick-to-bass")
            _args, _kwargs = mock_register.call_args
            assert _kwargs.get("name") == "kick-to-bass"


# ============================================================================
# Whitebox: Module-level API integrity
# ============================================================================

class TestModuleAPI:
    """Module-level __all__ and import integrity."""

    def test_all_exports_match_definitions(self):
        """__all__ contains exactly the 6 public API symbols."""
        from engine.audio.mixing import sidechain_bridge as mod

        expected = [
            "SidechainBridgeConfig",
            "extract_sidechain_config",
            "has_sidechain",
            "create_compressor_config",
            "register_sidechain",
            "apply_sidechain",
        ]
        assert mod.__all__ == expected

    def test_all_symbols_defined_in_module(self):
        """Every name in __all__ is actually defined in the module."""
        import engine.audio.mixing.sidechain_bridge as mod

        for name in mod.__all__:
            assert hasattr(mod, name), (
                f"__all__ contains {name} but module has no such attribute"
            )

    def test_undecorated_apply_returns_none_via_guard(self):
        """apply_sidechain returns None when source_bus is falsy (guard at L247)."""
        mgr = MagicMock(spec=SidechainManager)

        class FalsySource:
            _sidechain = True
            _sidechain_source_bus = ""

        key = MagicMock(spec=MixBus)
        key.name = "kick"
        target = MagicMock(spec=MixBus)
        target.name = "bass"

        result = apply_sidechain(mgr, FalsySource, key, target)
        assert result is None
        mgr.create_compressor.assert_not_called()
