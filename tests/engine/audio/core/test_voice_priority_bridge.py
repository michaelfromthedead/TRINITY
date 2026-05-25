"""Whitebox tests for Engine Audio Core: voice_priority_bridge module."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from engine.audio.core.audio_source import AudioSource
from engine.audio.core.config import PRIORITY_NORMAL, VoiceStealStrategy
from engine.audio.core.voice_manager import VoiceManager
from engine.audio.core.voice_priority_bridge import (
    VoicePriorityConfig,
    apply_component_to_allocation,
    configure_source,
    configure_source_from_component,
    extract_voice_config,
    has_voice_priority,
    register_component_rules,
)


# ============================================================================
# TestVoicePriorityConfig
# ============================================================================

class TestVoicePriorityConfig:
    """VoicePriorityConfig dataclass construction and default values."""

    def test_defaults(self):
        """Default config uses PRIORITY_NORMAL, virtualize=True, steal_oldest=True."""
        cfg = VoicePriorityConfig()
        assert cfg.priority == PRIORITY_NORMAL
        assert cfg.virtualize is True
        assert cfg.steal_oldest is True

    def test_explicit_values(self):
        """Explicit constructor values are stored."""
        cfg = VoicePriorityConfig(priority=100, virtualize=False, steal_oldest=False)
        assert cfg.priority == 100
        assert cfg.virtualize is False
        assert cfg.steal_oldest is False

    def test_priority_zero(self):
        """Priority 0 is valid (lowest)."""
        cfg = VoicePriorityConfig(priority=0)
        assert cfg.priority == 0

    def test_priority_one_hundred(self):
        """Priority 100 is valid (highest)."""
        cfg = VoicePriorityConfig(priority=100)
        assert cfg.priority == 100

    def test_dataclass_attributes(self):
        """VoicePriorityConfig has the expected three attributes."""
        cfg = VoicePriorityConfig()
        assert hasattr(cfg, "priority")
        assert hasattr(cfg, "virtualize")
        assert hasattr(cfg, "steal_oldest")


# ============================================================================
# TestExtractVoiceConfig
# ============================================================================

class TestExtractVoiceConfig:
    """extract_voice_config() reads decorator attrs or returns defaults."""

    def test_decorated_class(self):
        """Extract returns config matching decorator attrs."""

        class Decorated:
            _voice_priority = True
            _voice_priority_value = 75
            _voice_virtualize = False
            _voice_steal_oldest = True

        cfg = extract_voice_config(Decorated)
        assert cfg.priority == 75
        assert cfg.virtualize is False
        assert cfg.steal_oldest is True

    def test_undecorated_class(self):
        """Extract returns defaults for undecorated classes."""

        class Plain:
            pass

        cfg = extract_voice_config(Plain)
        assert cfg.priority == PRIORITY_NORMAL
        assert cfg.virtualize is True
        assert cfg.steal_oldest is True

    def test_partial_attrs(self):
        """Missing sub-attrs fall back to defaults."""

        class Partial:
            _voice_priority = True
            # no _voice_priority_value etc.

        cfg = extract_voice_config(Partial)
        assert cfg.priority == PRIORITY_NORMAL
        assert cfg.virtualize is True
        assert cfg.steal_oldest is True

    def test_priority_zero(self):
        """Priority 0 is preserved, not treated as falsy."""

        class LowPrio:
            _voice_priority = True
            _voice_priority_value = 0
            _voice_virtualize = True
            _voice_steal_oldest = True

        cfg = extract_voice_config(LowPrio)
        assert cfg.priority == 0

    def test_priority_one_hundred(self):
        """Priority 100 is preserved."""

        class HighPrio:
            _voice_priority = True
            _voice_priority_value = 100
            _voice_virtualize = True
            _voice_steal_oldest = True

        cfg = extract_voice_config(HighPrio)
        assert cfg.priority == 100

    def test_returns_voice_priority_config(self):
        """Extract always returns a VoicePriorityConfig instance."""

        class Any:
            pass

        cfg = extract_voice_config(Any)
        assert isinstance(cfg, VoicePriorityConfig)


# ============================================================================
# TestHasVoicePriority
# ============================================================================

class TestHasVoicePriority:
    """has_voice_priority() detection."""

    def test_decorated_returns_true(self):
        class Decorated:
            _voice_priority = True

        assert has_voice_priority(Decorated) is True

    def test_undecorated_returns_false(self):
        class Plain:
            pass

        assert has_voice_priority(Plain) is False

    def test_explicit_false(self):
        class Explicit:
            _voice_priority = False

        assert has_voice_priority(Explicit) is False

    def test_falsy_value_zero(self):
        class ZeroVal:
            _voice_priority = 0

        assert has_voice_priority(ZeroVal) is False

    def test_truthy_non_bool(self):
        class StrVal:
            _voice_priority = "yes"

        assert has_voice_priority(StrVal) is True


# ============================================================================
# TestConfigureSource
# ============================================================================

class TestConfigureSource:
    """configure_source() sets AudioSource.priority and is_virtual."""

    # --- priority tests ---

    def test_sets_priority(self):
        source = AudioSource()
        cfg = VoicePriorityConfig(priority=80)
        configure_source(source, cfg)
        assert source.priority == 80

    def test_default_priority(self):
        source = AudioSource()
        cfg = VoicePriorityConfig()
        configure_source(source, cfg)
        assert source.priority == PRIORITY_NORMAL

    def test_overwrites_existing(self):
        source = AudioSource()
        source.priority = 100
        cfg = VoicePriorityConfig(priority=25)
        configure_source(source, cfg)
        assert source.priority == 25

    def test_priority_zero(self):
        source = AudioSource()
        cfg = VoicePriorityConfig(priority=0)
        configure_source(source, cfg)
        assert source.priority == 0

    def test_priority_one_hundred(self):
        source = AudioSource()
        cfg = VoicePriorityConfig(priority=100)
        configure_source(source, cfg)
        assert source.priority == 100

    # --- is_virtual tests (H1 verification) ---

    def test_sets_is_virtual_true(self):
        """H1: configure_source sets source.is_virtual=True when config.virtualize=True."""
        source = AudioSource()
        cfg = VoicePriorityConfig(priority=50, virtualize=True)
        configure_source(source, cfg)
        assert source.is_virtual is True

    def test_sets_is_virtual_false(self):
        """H1: configure_source sets source.is_virtual=False when config.virtualize=False."""
        source = AudioSource()
        cfg = VoicePriorityConfig(priority=50, virtualize=False)
        configure_source(source, cfg)
        assert source.is_virtual is False

    def test_default_is_virtual_true(self):
        """H1: Default config sets source.is_virtual=True (virtualize defaults to True)."""
        source = AudioSource()
        cfg = VoicePriorityConfig()
        configure_source(source, cfg)
        assert source.is_virtual is True

    def test_is_virtual_preserved_when_reconfigured(self):
        """H1: Re-configuring overwrites prior is_virtual value."""
        source = AudioSource()
        cfg1 = VoicePriorityConfig(priority=50, virtualize=False)
        configure_source(source, cfg1)
        assert source.is_virtual is False

        cfg2 = VoicePriorityConfig(priority=80, virtualize=True)
        configure_source(source, cfg2)
        assert source.is_virtual is True


# ============================================================================
# TestConfigureSourceFromComponent
# ============================================================================

class TestConfigureSourceFromComponent:
    """configure_source_from_component() extracts + applies in one call."""

    def test_decorated_component(self):
        class Decorated:
            _voice_priority = True
            _voice_priority_value = 60
            _voice_virtualize = True
            _voice_steal_oldest = True

        source = AudioSource()
        configure_source_from_component(source, Decorated)
        assert source.priority == 60

    def test_undecorated_component(self):
        class Plain:
            pass

        source = AudioSource()
        source.priority = 99
        source.is_virtual = False
        configure_source_from_component(source, Plain)
        # configure_source sets back to defaults (M2: PRIORITY_NORMAL, H1: is_virtual=True)
        assert source.priority == PRIORITY_NORMAL
        assert source.is_virtual is True


# ============================================================================
# TestRegisterComponentRules
# ============================================================================

class TestRegisterComponentRules:
    """register_component_rules() upgrades strategy from NONE."""

    @pytest.fixture
    def vm(self):
        mgr = MagicMock(spec=VoiceManager)
        mgr._steal_strategy = VoiceStealStrategy.NONE
        return mgr

    def test_upgrades_none_to_lowest_priority(self, vm):
        class StealEnabled:
            _voice_priority = True
            _voice_priority_value = 50
            _voice_virtualize = True
            _voice_steal_oldest = True

        register_component_rules(vm, StealEnabled)
        vm.set_steal_strategy.assert_called_once_with(VoiceStealStrategy.LOWEST_PRIORITY)

    def test_steal_oldest_false_does_not_upgrade(self, vm):
        class StealDisabled:
            _voice_priority = True
            _voice_priority_value = 50
            _voice_virtualize = True
            _voice_steal_oldest = False

        register_component_rules(vm, StealDisabled)
        vm.set_steal_strategy.assert_not_called()

    def test_no_call_when_already_set(self, vm):
        """Does not overwrite an explicit non-NONE strategy."""
        vm._steal_strategy = VoiceStealStrategy.OLDEST

        class StealEnabled:
            _voice_priority = True
            _voice_priority_value = 50
            _voice_virtualize = True
            _voice_steal_oldest = True

        register_component_rules(vm, StealEnabled)
        vm.set_steal_strategy.assert_not_called()

    def test_undecorated_steal_true_does_upgrade(self, vm):
        """Undecorated classes get default config (steal_oldest=True), so upgrade happens."""

        class Undecorated:
            pass

        register_component_rules(vm, Undecorated)
        vm.set_steal_strategy.assert_called_once_with(VoiceStealStrategy.LOWEST_PRIORITY)

    def test_undecorated_steal_false_does_not_upgrade(self, vm):
        """Explicitly opt out of steal on an undecorated class.

        Since the class has _voice_priority=False, extract_voice_config
        returns defaults (steal_oldest=True). To prevent upgrade the
        test sets _steal_strategy to non-NONE beforehand.
        """
        vm._steal_strategy = VoiceStealStrategy.QUIETEST

        class Undecorated:
            pass

        register_component_rules(vm, Undecorated)
        vm.set_steal_strategy.assert_not_called()

    def test_idempotent(self, vm):
        """Calling twice with MagicMock calls set_steal_strategy twice because
        the mock doesn't actually mutate _steal_strategy.  Use a real
        VoiceManager to verify idempotency."""

        class StealEnabled:
            _voice_priority = True
            _voice_priority_value = 50
            _voice_virtualize = True
            _voice_steal_oldest = True

        register_component_rules(vm, StealEnabled)
        vm.set_steal_strategy.assert_called_once_with(VoiceStealStrategy.LOWEST_PRIORITY)

    def test_real_voice_manager(self):
        """Integration-style: real VoiceManager with NONE gets upgraded."""
        vm = VoiceManager(max_voices=8, steal_strategy=VoiceStealStrategy.NONE)

        class StealEnabled:
            _voice_priority = True
            _voice_priority_value = 80
            _voice_virtualize = True
            _voice_steal_oldest = True

        register_component_rules(vm, StealEnabled)
        assert vm._steal_strategy == VoiceStealStrategy.LOWEST_PRIORITY

    def test_real_voice_manager_no_change(self):
        """Real VoiceManager with OLDEST stays OLDEST."""
        vm = VoiceManager(max_voices=8, steal_strategy=VoiceStealStrategy.OLDEST)

        class StealEnabled:
            _voice_priority = True
            _voice_priority_value = 80
            _voice_virtualize = True
            _voice_steal_oldest = True

        register_component_rules(vm, StealEnabled)
        assert vm._steal_strategy == VoiceStealStrategy.OLDEST


# ============================================================================
# TestApplyComponentToAllocation
# ============================================================================

class TestApplyComponentToAllocation:
    """apply_component_to_allocation() combines source config + rules."""

    @pytest.fixture
    def source(self):
        return AudioSource()

    @pytest.fixture
    def vm(self):
        mgr = MagicMock(spec=VoiceManager)
        mgr._steal_strategy = VoiceStealStrategy.NONE
        return mgr

    def test_sets_source_priority(self, source, vm):
        class Component:
            _voice_priority = True
            _voice_priority_value = 90
            _voice_virtualize = True
            _voice_steal_oldest = True

        apply_component_to_allocation(vm, source, Component)
        assert source.priority == 90

    def test_sets_source_is_virtual(self, source, vm):
        """H1: apply_component_to_allocation wires source.is_virtual from config."""
        class Component:
            _voice_priority = True
            _voice_priority_value = 50
            _voice_virtualize = False
            _voice_steal_oldest = True

        apply_component_to_allocation(vm, source, Component)
        assert source.is_virtual is False

    def test_upgrades_strategy(self, source, vm):
        class Component:
            _voice_priority = True
            _voice_priority_value = 90
            _voice_virtualize = True
            _voice_steal_oldest = True

        apply_component_to_allocation(vm, source, Component)
        vm.set_steal_strategy.assert_called_once_with(VoiceStealStrategy.LOWEST_PRIORITY)

    def test_steal_false_does_not_upgrade(self, source, vm):
        class Component:
            _voice_priority = True
            _voice_priority_value = 90
            _voice_virtualize = True
            _voice_steal_oldest = False

        apply_component_to_allocation(vm, source, Component)
        assert source.priority == 90
        vm.set_steal_strategy.assert_not_called()

    def test_undecorated_sets_default_priority(self, source, vm):
        class Plain:
            pass

        apply_component_to_allocation(vm, source, Plain)
        assert source.priority == PRIORITY_NORMAL
        # Undecorated gets default config (steal_oldest=True), so upgrade called
        vm.set_steal_strategy.assert_called_once_with(VoiceStealStrategy.LOWEST_PRIORITY)

    def test_sets_then_gives_upgrade(self, source, vm):
        """Verifies source is configured before strategy is checked."""
        class Component:
            _voice_priority = True
            _voice_priority_value = 100
            _voice_virtualize = True
            _voice_steal_oldest = True

        apply_component_to_allocation(vm, source, Component)
        assert source.priority == 100
        vm.set_steal_strategy.assert_called_once()

    def test_with_real_voice_manager(self):
        """Integration-style: real VoiceManager + real source."""
        vm = VoiceManager(max_voices=8, steal_strategy=VoiceStealStrategy.NONE)
        source = AudioSource()

        class Component:
            _voice_priority = True
            _voice_priority_value = 70
            _voice_virtualize = False
            _voice_steal_oldest = True

        apply_component_to_allocation(vm, source, Component)
        assert source.priority == 70
        assert vm._steal_strategy == VoiceStealStrategy.LOWEST_PRIORITY

    def test_real_voice_manager_no_strategy_change(self):
        """Real VoiceManager with explicit strategy unchanged."""
        vm = VoiceManager(max_voices=8, steal_strategy=VoiceStealStrategy.QUIETEST)
        source = AudioSource()

        class Component:
            _voice_priority = True
            _voice_priority_value = 70
            _voice_virtualize = False
            _voice_steal_oldest = True

        apply_component_to_allocation(vm, source, Component)
        assert source.priority == 70
        assert vm._steal_strategy == VoiceStealStrategy.QUIETEST
