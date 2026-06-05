"""Tests for T-CC-1.1: Quality tier integration for Lighting subsystem."""
import pytest
from trinity.types import QualityTier
from engine.rendering.lighting.tier_integration import (
    LightCullingMode,
    LightingFeature,
    ClusterConfig,
    LightingTierConfig,
    LightBudgetState,
    TierChangeListener,
    LightingTierManager,
    create_low_tier_config,
    create_medium_tier_config,
    create_high_tier_config,
    create_ultra_tier_config,
    TIER_CONFIGS,
    get_tier_for_light_count,
    estimate_lighting_memory,
)


class TestLightCullingMode:
    """Tests for LightCullingMode enum."""

    def test_all_modes_exist(self):
        modes = [
            LightCullingMode.NONE,
            LightCullingMode.PER_OBJECT,
            LightCullingMode.CLUSTERED,
            LightCullingMode.TILED,
            LightCullingMode.TILED_CLUSTERED,
            LightCullingMode.GPU_DRIVEN,
        ]
        assert len(modes) == 6


class TestLightingFeature:
    """Tests for LightingFeature enum."""

    def test_all_features_exist(self):
        features = list(LightingFeature)
        assert LightingFeature.DIRECTIONAL_LIGHT in features
        assert LightingFeature.VOLUMETRIC_LIGHTING in features


class TestClusterConfig:
    """Tests for ClusterConfig dataclass."""

    def test_default_values(self):
        cfg = ClusterConfig()
        assert cfg.tiles_x == 16
        assert cfg.tiles_y == 8
        assert cfg.slices_z == 24
        assert cfg.use_exponential_depth is True

    def test_custom_values(self):
        cfg = ClusterConfig(tiles_x=32, tiles_y=16, slices_z=48)
        assert cfg.tiles_x == 32
        assert cfg.tiles_y == 16
        assert cfg.slices_z == 48


class TestLightingTierConfig:
    """Tests for LightingTierConfig dataclass."""

    def test_uses_clustering_false_for_per_object(self):
        cfg = LightingTierConfig(
            tier=QualityTier.LOW,
            max_lights=8,
            max_point_lights=4,
            max_spot_lights=4,
            culling_mode=LightCullingMode.PER_OBJECT,
        )
        assert cfg.uses_clustering is False

    def test_uses_clustering_true_for_clustered(self):
        cfg = LightingTierConfig(
            tier=QualityTier.MEDIUM,
            max_lights=64,
            max_point_lights=32,
            max_spot_lights=32,
            culling_mode=LightCullingMode.CLUSTERED,
        )
        assert cfg.uses_clustering is True

    def test_uses_clustering_true_for_gpu_driven(self):
        cfg = LightingTierConfig(
            tier=QualityTier.ULTRA,
            max_lights=-1,
            max_point_lights=-1,
            max_spot_lights=-1,
            culling_mode=LightCullingMode.GPU_DRIVEN,
        )
        assert cfg.uses_clustering is True

    def test_uses_deferred_false_without_feature(self):
        cfg = create_low_tier_config()
        assert cfg.uses_deferred is False

    def test_uses_deferred_true_with_feature(self):
        cfg = create_high_tier_config()
        assert cfg.uses_deferred is True

    def test_supports_area_lights(self):
        low = create_low_tier_config()
        high = create_high_tier_config()
        assert low.supports_area_lights is False
        assert high.supports_area_lights is True


class TestTierConfigs:
    """Tests for tier config factory functions."""

    def test_low_tier_config(self):
        cfg = create_low_tier_config()
        assert cfg.tier == QualityTier.LOW
        assert cfg.max_lights == 8
        assert cfg.max_point_lights == 4
        assert cfg.max_spot_lights == 4
        assert cfg.culling_mode == LightCullingMode.PER_OBJECT
        assert LightingFeature.FORWARD_SHADING in cfg.enabled_features
        assert cfg.cluster_config is None

    def test_medium_tier_config(self):
        cfg = create_medium_tier_config()
        assert cfg.tier == QualityTier.MEDIUM
        assert cfg.max_lights == 64
        assert cfg.culling_mode == LightCullingMode.CLUSTERED
        assert cfg.cluster_config is not None
        assert cfg.cluster_config.tiles_x == 16

    def test_high_tier_config(self):
        cfg = create_high_tier_config()
        assert cfg.tier == QualityTier.HIGH
        assert cfg.max_lights == 256
        assert cfg.max_area_lights == 32
        assert cfg.culling_mode == LightCullingMode.TILED_CLUSTERED
        assert LightingFeature.AREA_LIGHTS in cfg.enabled_features

    def test_ultra_tier_config(self):
        cfg = create_ultra_tier_config()
        assert cfg.tier == QualityTier.ULTRA
        assert cfg.max_lights == -1  # Unlimited
        assert cfg.culling_mode == LightCullingMode.GPU_DRIVEN
        assert LightingFeature.VOLUMETRIC_LIGHTING in cfg.enabled_features
        assert cfg.volumetric_samples == 64

    def test_all_tiers_in_dict(self):
        assert QualityTier.LOW in TIER_CONFIGS
        assert QualityTier.MEDIUM in TIER_CONFIGS
        assert QualityTier.HIGH in TIER_CONFIGS
        assert QualityTier.ULTRA in TIER_CONFIGS


class TestLightBudgetState:
    """Tests for LightBudgetState tracking."""

    def test_initial_values(self):
        state = LightBudgetState()
        assert state.point_lights_used == 0
        assert state.spot_lights_used == 0
        assert state.area_lights_used == 0

    def test_total_lights(self):
        state = LightBudgetState(point_lights_used=5, spot_lights_used=3, area_lights_used=2)
        assert state.total_lights == 10

    def test_reset(self):
        state = LightBudgetState(point_lights_used=5, spot_lights_used=3, area_lights_used=2)
        state.reset()
        assert state.total_lights == 0


class TestTierChangeListener:
    """Tests for tier change listener protocol."""

    def test_listener_receives_notification(self):
        class TestListener(TierChangeListener):
            def __init__(self):
                self.changes = []

            def on_tier_changed(self, old_tier, new_tier, config):
                self.changes.append((old_tier, new_tier))

        mgr = LightingTierManager(QualityTier.LOW)
        listener = TestListener()
        mgr.add_listener(listener)
        mgr.set_tier(QualityTier.HIGH)

        assert len(listener.changes) == 1
        assert listener.changes[0] == (QualityTier.LOW, QualityTier.HIGH)


class TestLightingTierManager:
    """Tests for LightingTierManager."""

    def test_default_tier(self):
        mgr = LightingTierManager()
        assert mgr.current_tier == QualityTier.MEDIUM

    def test_custom_initial_tier(self):
        mgr = LightingTierManager(QualityTier.HIGH)
        assert mgr.current_tier == QualityTier.HIGH

    def test_set_tier(self):
        mgr = LightingTierManager(QualityTier.LOW)
        mgr.set_tier(QualityTier.ULTRA)
        assert mgr.current_tier == QualityTier.ULTRA
        assert mgr.config.tier == QualityTier.ULTRA

    def test_set_same_tier_no_op(self):
        mgr = LightingTierManager(QualityTier.MEDIUM)
        changes = []
        listener = type("L", (), {"on_tier_changed": lambda s, o, n, c: changes.append(1)})()
        mgr.add_listener(listener)
        mgr.set_tier(QualityTier.MEDIUM)  # Same tier
        assert len(changes) == 0

    def test_config_property(self):
        mgr = LightingTierManager(QualityTier.LOW)
        assert mgr.config.max_lights == 8

    def test_is_feature_enabled(self):
        mgr = LightingTierManager(QualityTier.LOW)
        assert mgr.is_feature_enabled(LightingFeature.FORWARD_SHADING)
        assert not mgr.is_feature_enabled(LightingFeature.DEFERRED_SHADING)

    def test_can_add_light_within_budget(self):
        mgr = LightingTierManager(QualityTier.LOW)
        assert mgr.can_add_light("point") is True

    def test_can_add_light_exceeds_budget(self):
        mgr = LightingTierManager(QualityTier.LOW)
        # Register all 4 point lights
        for _ in range(4):
            mgr.register_light("point")
        assert mgr.can_add_light("point") is False

    def test_can_add_area_light_low_tier(self):
        mgr = LightingTierManager(QualityTier.LOW)
        assert mgr.can_add_light("area") is False  # Not supported

    def test_can_add_area_light_high_tier(self):
        mgr = LightingTierManager(QualityTier.HIGH)
        assert mgr.can_add_light("area") is True

    def test_register_light(self):
        mgr = LightingTierManager(QualityTier.MEDIUM)
        assert mgr.register_light("point") is True
        assert mgr.budget_state.point_lights_used == 1

    def test_register_light_over_budget(self):
        mgr = LightingTierManager(QualityTier.LOW)
        for _ in range(4):
            mgr.register_light("point")
        assert mgr.register_light("point") is False

    def test_unregister_light(self):
        mgr = LightingTierManager(QualityTier.MEDIUM)
        mgr.register_light("spot")
        assert mgr.budget_state.spot_lights_used == 1
        mgr.unregister_light("spot")
        assert mgr.budget_state.spot_lights_used == 0

    def test_unregister_light_underflow(self):
        mgr = LightingTierManager(QualityTier.MEDIUM)
        mgr.unregister_light("point")  # Should not go negative
        assert mgr.budget_state.point_lights_used == 0

    def test_begin_frame_resets_budget(self):
        mgr = LightingTierManager(QualityTier.MEDIUM)
        mgr.register_light("point")
        mgr.register_light("spot")
        mgr.begin_frame()
        assert mgr.budget_state.total_lights == 0


class TestOverrides:
    """Tests for feature overrides."""

    def test_override_feature_enable(self):
        mgr = LightingTierManager(QualityTier.LOW)
        assert not mgr.is_feature_enabled(LightingFeature.VOLUMETRIC_LIGHTING)
        mgr.override_feature(LightingFeature.VOLUMETRIC_LIGHTING, True)
        assert mgr.is_feature_enabled(LightingFeature.VOLUMETRIC_LIGHTING)

    def test_override_feature_disable(self):
        mgr = LightingTierManager(QualityTier.ULTRA)
        assert mgr.is_feature_enabled(LightingFeature.VOLUMETRIC_LIGHTING)
        mgr.override_feature(LightingFeature.VOLUMETRIC_LIGHTING, False)
        assert not mgr.is_feature_enabled(LightingFeature.VOLUMETRIC_LIGHTING)

    def test_clear_overrides(self):
        mgr = LightingTierManager(QualityTier.LOW)
        mgr.override_feature(LightingFeature.VOLUMETRIC_LIGHTING, True)
        mgr.clear_overrides()
        assert not mgr.is_feature_enabled(LightingFeature.VOLUMETRIC_LIGHTING)


class TestListeners:
    """Tests for listener management."""

    def test_add_listener(self):
        mgr = LightingTierManager()
        listener = TierChangeListener()
        mgr.add_listener(listener)
        assert listener in mgr._listeners

    def test_remove_listener(self):
        mgr = LightingTierManager()
        listener = TierChangeListener()
        mgr.add_listener(listener)
        mgr.remove_listener(listener)
        assert listener not in mgr._listeners

    def test_duplicate_listener_not_added(self):
        mgr = LightingTierManager()
        listener = TierChangeListener()
        mgr.add_listener(listener)
        mgr.add_listener(listener)
        assert mgr._listeners.count(listener) == 1


class TestClusterConfig:
    """Tests for cluster configuration access."""

    def test_get_cluster_config_clustered(self):
        mgr = LightingTierManager(QualityTier.MEDIUM)
        cfg = mgr.get_cluster_config()
        assert cfg is not None
        assert cfg.tiles_x == 16

    def test_get_cluster_config_not_clustered(self):
        mgr = LightingTierManager(QualityTier.LOW)
        cfg = mgr.get_cluster_config()
        assert cfg is None

    def test_get_culling_mode(self):
        mgr = LightingTierManager(QualityTier.LOW)
        assert mgr.get_culling_mode() == LightCullingMode.PER_OBJECT


class TestResourceBudgets:
    """Tests for resource budget accessors."""

    def test_get_shadow_resolution(self):
        low = LightingTierManager(QualityTier.LOW)
        high = LightingTierManager(QualityTier.HIGH)
        assert low.get_shadow_resolution() == 512
        assert high.get_shadow_resolution() == 2048

    def test_get_gpu_budget_ms(self):
        low = LightingTierManager(QualityTier.LOW)
        assert low.get_gpu_budget_ms() == 1.0

    def test_get_memory_budget_mb(self):
        ultra = LightingTierManager(QualityTier.ULTRA)
        assert ultra.get_memory_budget_mb() == 256

    def test_get_volumetric_samples(self):
        low = LightingTierManager(QualityTier.LOW)
        ultra = LightingTierManager(QualityTier.ULTRA)
        assert low.get_volumetric_samples() == 0
        assert ultra.get_volumetric_samples() == 64


class TestStatusDict:
    """Tests for status dictionary generation."""

    def test_get_status_dict(self):
        mgr = LightingTierManager(QualityTier.MEDIUM)
        mgr.register_light("point")
        mgr.register_light("spot")

        status = mgr.get_status_dict()
        assert status["tier"] == "MEDIUM"
        assert status["max_lights"] == 64
        assert status["culling_mode"] == "CLUSTERED"
        assert status["lights_used"] == 2
        assert status["point_lights"] == 1
        assert status["spot_lights"] == 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_tier_for_light_count_low(self):
        assert get_tier_for_light_count(1) == QualityTier.LOW
        assert get_tier_for_light_count(8) == QualityTier.LOW

    def test_get_tier_for_light_count_medium(self):
        assert get_tier_for_light_count(9) == QualityTier.MEDIUM
        assert get_tier_for_light_count(64) == QualityTier.MEDIUM

    def test_get_tier_for_light_count_high(self):
        assert get_tier_for_light_count(65) == QualityTier.HIGH
        assert get_tier_for_light_count(256) == QualityTier.HIGH

    def test_get_tier_for_light_count_ultra(self):
        assert get_tier_for_light_count(257) == QualityTier.ULTRA
        assert get_tier_for_light_count(10000) == QualityTier.ULTRA


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_estimate_low_tier_memory(self):
        cfg = create_low_tier_config()
        memory = estimate_lighting_memory(cfg, 1280, 720)
        assert memory > 0
        assert memory < 100 * 1024 * 1024  # Under 100MB

    def test_estimate_high_tier_memory(self):
        cfg = create_high_tier_config()
        memory = estimate_lighting_memory(cfg, 1920, 1080)
        # High tier with deferred needs G-buffer
        assert memory > estimate_lighting_memory(create_low_tier_config(), 1920, 1080)

    def test_estimate_includes_gbuffer_for_deferred(self):
        cfg = create_high_tier_config()  # Uses deferred
        memory_hd = estimate_lighting_memory(cfg, 1920, 1080)
        memory_4k = estimate_lighting_memory(cfg, 3840, 2160)
        # 4K should require more memory for G-buffer
        assert memory_4k > memory_hd


class TestIntegration:
    """Integration tests for tier management."""

    def test_full_tier_progression(self):
        """Test progressing through all tiers."""
        mgr = LightingTierManager(QualityTier.LOW)

        tiers = [QualityTier.MEDIUM, QualityTier.HIGH, QualityTier.ULTRA]
        expected_lights = [64, 256, -1]

        for tier, expected in zip(tiers, expected_lights):
            mgr.set_tier(tier)
            assert mgr.config.max_lights == expected

    def test_frame_lifecycle(self):
        """Test typical frame lifecycle."""
        mgr = LightingTierManager(QualityTier.MEDIUM)

        # Begin frame
        mgr.begin_frame()
        assert mgr.budget_state.total_lights == 0

        # Register lights during frame
        for _ in range(10):
            mgr.register_light("point")
        for _ in range(5):
            mgr.register_light("spot")

        assert mgr.budget_state.total_lights == 15

        # Next frame resets
        mgr.begin_frame()
        assert mgr.budget_state.total_lights == 0

    def test_tier_change_mid_frame(self):
        """Test tier change resets budget."""
        mgr = LightingTierManager(QualityTier.LOW)
        mgr.register_light("point")
        mgr.register_light("point")

        mgr.set_tier(QualityTier.HIGH)
        # Budget should be reset
        assert mgr.budget_state.total_lights == 0
        # Can now add many more lights
        for _ in range(100):
            assert mgr.register_light("point") is True

    def test_unlimited_lights_ultra_tier(self):
        """Test that Ultra tier has no practical limit."""
        mgr = LightingTierManager(QualityTier.ULTRA)

        # Should be able to add thousands of lights
        for _ in range(1000):
            assert mgr.register_light("point") is True
            assert mgr.register_light("spot") is True
            assert mgr.register_light("area") is True

        assert mgr.budget_state.total_lights == 3000
