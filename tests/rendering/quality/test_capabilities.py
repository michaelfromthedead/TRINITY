"""Tests for QualityCapabilities trait (T-CC-0.4)."""

import pytest

from trinity.types import QualityTier

from engine.rendering.quality.capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    QualityCapabilities,
    QualityCapabilitiesRegistry,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class TestTierFeatureSet:
    """Test TierFeatureSet dataclass."""

    def test_default_empty(self):
        """Test default feature set is empty."""
        fs = TierFeatureSet(tier=QualityTier.HIGH)
        assert len(fs.enabled_features) == 0
        assert len(fs.disabled_features) == 0

    def test_has_feature_true(self):
        """Test feature detection when enabled."""
        fs = TierFeatureSet(
            tier=QualityTier.HIGH,
            enabled_features=frozenset({"shadows", "gi", "reflections"}),
        )
        assert fs.has_feature("shadows") is True
        assert fs.has_feature("gi") is True

    def test_has_feature_false(self):
        """Test feature detection when not enabled."""
        fs = TierFeatureSet(
            tier=QualityTier.LOW,
            enabled_features=frozenset({"shadows"}),
        )
        assert fs.has_feature("gi") is False

    def test_get_param_exists(self):
        """Test getting existing parameter."""
        fs = TierFeatureSet(
            tier=QualityTier.MEDIUM,
            parameters={"light_count": 64, "shadow_cascades": 2},
        )
        assert fs.get_param("light_count") == 64
        assert fs.get_param("shadow_cascades") == 2

    def test_get_param_default(self):
        """Test getting non-existent parameter returns default."""
        fs = TierFeatureSet(tier=QualityTier.LOW)
        assert fs.get_param("light_count", 8) == 8
        assert fs.get_param("missing") is None


class TestTierBudget:
    """Test TierBudget dataclass."""

    def test_default_values(self):
        """Test default budget values."""
        budget = TierBudget(tier=QualityTier.HIGH)
        assert budget.gpu_time_ms == 2.0
        assert budget.memory_mb == 128
        assert budget.draw_calls == 100

    def test_exceeds_budget_gpu_time(self):
        """Test exceeds budget detection for GPU time."""
        budget = TierBudget(tier=QualityTier.HIGH, gpu_time_ms=5.0)
        assert budget.exceeds_budget(gpu_time_ms=6.0) is True
        assert budget.exceeds_budget(gpu_time_ms=4.0) is False

    def test_exceeds_budget_memory(self):
        """Test exceeds budget detection for memory."""
        budget = TierBudget(tier=QualityTier.HIGH, memory_mb=256)
        assert budget.exceeds_budget(memory_mb=300) is True
        assert budget.exceeds_budget(memory_mb=200) is False

    def test_exceeds_budget_combined(self):
        """Test combined budget check."""
        budget = TierBudget(
            tier=QualityTier.HIGH,
            gpu_time_ms=5.0,
            memory_mb=256,
            draw_calls=1000,
        )
        # All within budget
        assert budget.exceeds_budget(gpu_time_ms=3.0, memory_mb=128, draw_calls=500) is False
        # One exceeds
        assert budget.exceeds_budget(gpu_time_ms=3.0, memory_mb=300, draw_calls=500) is True


class TestTierResolution:
    """Test TierResolution dataclass."""

    def test_default_values(self):
        """Test default resolution values."""
        res = TierResolution(tier=QualityTier.HIGH)
        assert res.render_scale == 1.0
        assert res.shadow_resolution == 1024

    def test_scaled_resolution_native(self):
        """Test scaled resolution at native scale."""
        res = TierResolution(tier=QualityTier.ULTRA, render_scale=1.0)
        assert res.scaled_resolution(1920, 1080) == (1920, 1080)

    def test_scaled_resolution_half(self):
        """Test scaled resolution at half scale."""
        res = TierResolution(tier=QualityTier.LOW, render_scale=0.5)
        assert res.scaled_resolution(1920, 1080) == (960, 540)

    def test_scaled_resolution_minimum(self):
        """Test scaled resolution never goes below 1."""
        res = TierResolution(tier=QualityTier.LOW, render_scale=0.001)
        w, h = res.scaled_resolution(100, 100)
        assert w >= 1
        assert h >= 1


class TestFallbackChain:
    """Test FallbackChain dataclass."""

    def test_primary_only(self):
        """Test chain with only primary."""
        chain = FallbackChain(primary="ray_traced_shadows")
        assert chain.get_fallback(QualityTier.ULTRA) == "ray_traced_shadows"
        assert chain.get_fallback(QualityTier.LOW) == "ray_traced_shadows"

    def test_with_fallbacks(self):
        """Test chain with fallbacks."""
        chain = FallbackChain(
            primary="ray_traced_shadows",
            fallbacks=("vsm_shadows", "pcf_shadows", "hard_shadows"),
        )
        # ULTRA (3) gets primary
        assert chain.get_fallback(QualityTier.ULTRA) == "ray_traced_shadows"
        # HIGH (2) gets first fallback
        assert chain.get_fallback(QualityTier.HIGH) == "vsm_shadows"
        # MEDIUM (1) gets second fallback
        assert chain.get_fallback(QualityTier.MEDIUM) == "pcf_shadows"
        # LOW (0) gets last fallback
        assert chain.get_fallback(QualityTier.LOW) == "hard_shadows"


class MockSubsystemCapabilities(BaseQualityCapabilities):
    """Mock subsystem for testing."""

    @property
    def subsystem_name(self) -> str:
        return "mock_subsystem"

    def _init_tier_configs(self) -> None:
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({"basic_lighting"}),
                parameters={"light_count": 8},
            ),
        )
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({"basic_lighting", "shadows", "gi"}),
                parameters={"light_count": 256},
            ),
        )
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=1.0, memory_mb=64),
        )
        self._set_budget(
            QualityTier.HIGH,
            TierBudget(tier=QualityTier.HIGH, gpu_time_ms=4.0, memory_mb=512),
        )
        self._set_resolution(
            QualityTier.LOW,
            TierResolution(tier=QualityTier.LOW, render_scale=0.5, shadow_resolution=512),
        )
        self._set_resolution(
            QualityTier.HIGH,
            TierResolution(tier=QualityTier.HIGH, render_scale=1.0, shadow_resolution=2048),
        )
        self._set_fallback(
            "shadows",
            FallbackChain(primary="ray_traced", fallbacks=("vsm", "pcf")),
        )


class TestBaseQualityCapabilities:
    """Test BaseQualityCapabilities abstract base class."""

    def test_subsystem_name(self):
        """Test subsystem name property."""
        caps = MockSubsystemCapabilities()
        assert caps.subsystem_name == "mock_subsystem"

    def test_get_features_defined_tier(self):
        """Test getting features for defined tier."""
        caps = MockSubsystemCapabilities()
        features = caps.get_features(QualityTier.HIGH)
        assert features.has_feature("shadows")
        assert features.has_feature("gi")
        assert features.get_param("light_count") == 256

    def test_get_features_fallback(self):
        """Test features fall back to lower tier."""
        caps = MockSubsystemCapabilities()
        # MEDIUM not defined, should fall back to LOW
        features = caps.get_features(QualityTier.MEDIUM)
        assert features.tier == QualityTier.LOW
        assert features.get_param("light_count") == 8

    def test_get_budget_defined(self):
        """Test getting budget for defined tier."""
        caps = MockSubsystemCapabilities()
        budget = caps.get_budget(QualityTier.HIGH)
        assert budget.gpu_time_ms == 4.0
        assert budget.memory_mb == 512

    def test_get_resolution_defined(self):
        """Test getting resolution for defined tier."""
        caps = MockSubsystemCapabilities()
        res = caps.get_resolution(QualityTier.HIGH)
        assert res.render_scale == 1.0
        assert res.shadow_resolution == 2048

    def test_get_fallback_chain(self):
        """Test getting fallback chain."""
        caps = MockSubsystemCapabilities()
        chain = caps.get_fallback_chain("shadows")
        assert chain is not None
        assert chain.primary == "ray_traced"

    def test_get_fallback_chain_missing(self):
        """Test getting non-existent fallback chain."""
        caps = MockSubsystemCapabilities()
        chain = caps.get_fallback_chain("nonexistent")
        assert chain is None

    def test_supports_tier(self):
        """Test tier support check."""
        caps = MockSubsystemCapabilities()
        assert caps.supports_tier(QualityTier.LOW) is True
        assert caps.supports_tier(QualityTier.HIGH) is True
        assert caps.supports_tier(QualityTier.MEDIUM) is False

    def test_list_features(self):
        """Test listing enabled features."""
        caps = MockSubsystemCapabilities()
        features = caps.list_features(QualityTier.HIGH)
        assert "shadows" in features
        assert "gi" in features
        assert "basic_lighting" in features

    def test_compare_tiers(self):
        """Test comparing features between tiers."""
        caps = MockSubsystemCapabilities()
        diff = caps.compare_tiers(QualityTier.LOW, QualityTier.HIGH)
        assert "shadows" in diff["added"]
        assert "gi" in diff["added"]


class TestQualityCapabilitiesProtocol:
    """Test QualityCapabilities protocol compliance."""

    def test_mock_is_quality_capabilities(self):
        """Test mock implements protocol."""
        caps = MockSubsystemCapabilities()
        assert isinstance(caps, QualityCapabilities)


class TestQualityCapabilitiesRegistry:
    """Test QualityCapabilitiesRegistry singleton."""

    def setup_method(self):
        """Reset registry before each test."""
        QualityCapabilitiesRegistry.reset()

    def test_singleton(self):
        """Test registry is singleton."""
        reg1 = QualityCapabilitiesRegistry()
        reg2 = QualityCapabilitiesRegistry()
        assert reg1 is reg2

    def test_register_and_get(self):
        """Test registering and retrieving capabilities."""
        registry = QualityCapabilitiesRegistry()
        caps = MockSubsystemCapabilities()
        registry.register(caps)
        retrieved = registry.get("mock_subsystem")
        assert retrieved is caps

    def test_get_nonexistent(self):
        """Test getting non-existent subsystem."""
        registry = QualityCapabilitiesRegistry()
        assert registry.get("nonexistent") is None

    def test_unregister(self):
        """Test unregistering subsystem."""
        registry = QualityCapabilitiesRegistry()
        caps = MockSubsystemCapabilities()
        registry.register(caps)
        registry.unregister("mock_subsystem")
        assert registry.get("mock_subsystem") is None

    def test_list_subsystems(self):
        """Test listing registered subsystems."""
        registry = QualityCapabilitiesRegistry()
        caps = MockSubsystemCapabilities()
        registry.register(caps)
        subsystems = registry.list_subsystems()
        assert "mock_subsystem" in subsystems

    def test_get_all(self):
        """Test getting all registered capabilities."""
        registry = QualityCapabilitiesRegistry()
        caps = MockSubsystemCapabilities()
        registry.register(caps)
        all_caps = registry.get_all()
        assert "mock_subsystem" in all_caps

    def test_get_features_for_tier(self):
        """Test getting all features for a tier."""
        registry = QualityCapabilitiesRegistry()
        caps = MockSubsystemCapabilities()
        registry.register(caps)
        features = registry.get_features_for_tier(QualityTier.HIGH)
        assert "mock_subsystem" in features
        assert features["mock_subsystem"].has_feature("shadows")

    def test_get_total_budget(self):
        """Test getting combined budget across subsystems."""
        registry = QualityCapabilitiesRegistry()
        caps = MockSubsystemCapabilities()
        registry.register(caps)
        total = registry.get_total_budget(QualityTier.HIGH)
        assert total.gpu_time_ms == 4.0
        assert total.memory_mb == 512

    def test_reset(self):
        """Test reset clears registry."""
        registry = QualityCapabilitiesRegistry()
        caps = MockSubsystemCapabilities()
        registry.register(caps)
        QualityCapabilitiesRegistry.reset()
        new_registry = QualityCapabilitiesRegistry()
        assert len(new_registry.list_subsystems()) == 0
