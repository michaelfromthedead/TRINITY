"""Tests for T-CC-1.4: Quality tier integration for Shadows subsystem."""
import pytest
from trinity.types import QualityTier
from engine.rendering.shadows.tier_integration import (
    ShadowFilterMethod,
    ShadowFeature,
    CascadeConfig,
    PCFConfig,
    VSMConfig,
    PCSSConfig,
    ContactShadowConfig,
    RTShadowConfig,
    ShadowTierConfig,
    ShadowUsageStats,
    TierChangeListener,
    ShadowTierManager,
    create_low_tier_config,
    create_medium_tier_config,
    create_high_tier_config,
    create_ultra_tier_config,
    TIER_CONFIGS,
    get_tier_for_resolution,
    get_tier_for_filter,
    estimate_shadow_memory,
    calculate_cascade_splits,
)


class TestEnums:
    """Tests for enum definitions."""

    def test_shadow_filter_methods(self):
        methods = [
            ShadowFilterMethod.HARD,
            ShadowFilterMethod.PCF,
            ShadowFilterMethod.VSM,
            ShadowFilterMethod.PCSS,
            ShadowFilterMethod.RT,
        ]
        assert len(methods) == 5

    def test_shadow_features(self):
        features = list(ShadowFeature)
        assert ShadowFeature.DIRECTIONAL_SHADOW in features
        assert ShadowFeature.RAY_TRACED_SHADOWS in features


class TestCascadeConfig:
    """Tests for CascadeConfig."""

    def test_default_values(self):
        cfg = CascadeConfig()
        assert cfg.count == 4
        assert cfg.split_lambda == 0.5

    def test_custom_values(self):
        cfg = CascadeConfig(count=2, split_lambda=0.7)
        assert cfg.count == 2


class TestPCFConfig:
    """Tests for PCFConfig."""

    def test_default_values(self):
        cfg = PCFConfig()
        assert cfg.samples == 9
        assert cfg.use_rotated_poisson is True


class TestVSMConfig:
    """Tests for VSMConfig."""

    def test_default_values(self):
        cfg = VSMConfig()
        assert cfg.blur_samples == 5
        assert cfg.light_bleed_reduction == 0.2


class TestPCSSConfig:
    """Tests for PCSSConfig."""

    def test_default_values(self):
        cfg = PCSSConfig()
        assert cfg.blocker_samples == 16
        assert cfg.pcf_samples == 32


class TestContactShadowConfig:
    """Tests for ContactShadowConfig."""

    def test_default_disabled(self):
        cfg = ContactShadowConfig()
        assert cfg.enabled is False

    def test_enabled_with_steps(self):
        cfg = ContactShadowConfig(enabled=True, steps=32)
        assert cfg.steps == 32


class TestRTShadowConfig:
    """Tests for RTShadowConfig."""

    def test_default_disabled(self):
        cfg = RTShadowConfig()
        assert cfg.enabled is False

    def test_enabled_with_rays(self):
        cfg = RTShadowConfig(enabled=True, rays_per_pixel=4)
        assert cfg.rays_per_pixel == 4


class TestShadowTierConfig:
    """Tests for ShadowTierConfig."""

    def test_uses_cascades_false(self):
        cfg = create_low_tier_config()
        assert cfg.uses_cascades is False

    def test_uses_cascades_true(self):
        cfg = create_medium_tier_config()
        assert cfg.uses_cascades is True

    def test_uses_ray_tracing_false(self):
        cfg = create_high_tier_config()
        assert cfg.uses_ray_tracing is False

    def test_uses_ray_tracing_true(self):
        cfg = create_ultra_tier_config()
        assert cfg.uses_ray_tracing is True

    def test_supports_soft_shadows(self):
        pcf = create_medium_tier_config()
        ultra = create_ultra_tier_config()
        assert pcf.supports_soft_shadows is False
        assert ultra.supports_soft_shadows is True


class TestTierConfigs:
    """Tests for tier config factory functions."""

    def test_low_tier_config(self):
        cfg = create_low_tier_config()
        assert cfg.tier == QualityTier.LOW
        assert cfg.resolution == 512
        assert cfg.filter_method == ShadowFilterMethod.PCF
        assert cfg.cascade_config.count == 1
        assert cfg.pcf_config.samples == 4

    def test_medium_tier_config(self):
        cfg = create_medium_tier_config()
        assert cfg.tier == QualityTier.MEDIUM
        assert cfg.resolution == 1024
        assert cfg.cascade_config.count == 2
        assert ShadowFeature.CASCADED_SHADOWS in cfg.enabled_features

    def test_high_tier_config(self):
        cfg = create_high_tier_config()
        assert cfg.tier == QualityTier.HIGH
        assert cfg.resolution == 2048
        assert cfg.filter_method == ShadowFilterMethod.VSM
        assert cfg.cascade_config.count == 4
        assert cfg.contact_config.enabled is True

    def test_ultra_tier_config(self):
        cfg = create_ultra_tier_config()
        assert cfg.tier == QualityTier.ULTRA
        assert cfg.resolution == 4096
        assert cfg.filter_method == ShadowFilterMethod.RT
        assert cfg.rt_config.enabled is True
        assert ShadowFeature.RAY_TRACED_SHADOWS in cfg.enabled_features

    def test_all_tiers_in_dict(self):
        for tier in QualityTier:
            assert tier in TIER_CONFIGS

    def test_tier_resolutions_increase(self):
        low = create_low_tier_config()
        medium = create_medium_tier_config()
        high = create_high_tier_config()
        ultra = create_ultra_tier_config()
        assert low.resolution < medium.resolution
        assert medium.resolution < high.resolution
        assert high.resolution < ultra.resolution


class TestShadowUsageStats:
    """Tests for shadow usage statistics."""

    def test_initial_values(self):
        stats = ShadowUsageStats()
        assert stats.active_spot_shadows == 0
        assert stats.active_point_shadows == 0


class TestTierChangeListener:
    """Tests for tier change listener."""

    def test_listener_receives_notification(self):
        class TestListener(TierChangeListener):
            def __init__(self):
                self.changes = []

            def on_tier_changed(self, old_tier, new_tier, config):
                self.changes.append((old_tier, new_tier))

        mgr = ShadowTierManager(QualityTier.LOW)
        listener = TestListener()
        mgr.add_listener(listener)
        mgr.set_tier(QualityTier.HIGH)

        assert len(listener.changes) == 1
        assert listener.changes[0] == (QualityTier.LOW, QualityTier.HIGH)


class TestShadowTierManager:
    """Tests for ShadowTierManager."""

    def test_default_tier(self):
        mgr = ShadowTierManager()
        assert mgr.current_tier == QualityTier.MEDIUM

    def test_custom_initial_tier(self):
        mgr = ShadowTierManager(QualityTier.HIGH)
        assert mgr.current_tier == QualityTier.HIGH

    def test_set_tier(self):
        mgr = ShadowTierManager(QualityTier.LOW)
        mgr.set_tier(QualityTier.ULTRA)
        assert mgr.current_tier == QualityTier.ULTRA

    def test_set_same_tier_no_op(self):
        mgr = ShadowTierManager(QualityTier.MEDIUM)
        changes = []
        listener = type("L", (), {"on_tier_changed": lambda s, o, n, c: changes.append(1)})()
        mgr.add_listener(listener)
        mgr.set_tier(QualityTier.MEDIUM)
        assert len(changes) == 0

    def test_is_feature_enabled(self):
        mgr = ShadowTierManager(QualityTier.LOW)
        assert mgr.is_feature_enabled(ShadowFeature.DIRECTIONAL_SHADOW)
        assert not mgr.is_feature_enabled(ShadowFeature.CASCADED_SHADOWS)


class TestResolutionAndFilter:
    """Tests for resolution and filter method."""

    def test_get_resolution(self):
        low = ShadowTierManager(QualityTier.LOW)
        high = ShadowTierManager(QualityTier.HIGH)
        assert low.get_resolution() == 512
        assert high.get_resolution() == 2048

    def test_get_filter_method(self):
        low = ShadowTierManager(QualityTier.LOW)
        high = ShadowTierManager(QualityTier.HIGH)
        assert low.get_filter_method() == ShadowFilterMethod.PCF
        assert high.get_filter_method() == ShadowFilterMethod.VSM

    def test_rt_fallback_when_unavailable(self):
        mgr = ShadowTierManager(QualityTier.ULTRA)
        mgr.set_rt_available(False)
        assert mgr.get_filter_method() == ShadowFilterMethod.PCSS

    def test_rt_used_when_available(self):
        mgr = ShadowTierManager(QualityTier.ULTRA)
        mgr.set_rt_available(True)
        assert mgr.get_filter_method() == ShadowFilterMethod.RT


class TestCascadeAccess:
    """Tests for cascade configuration access."""

    def test_get_cascade_config(self):
        mgr = ShadowTierManager(QualityTier.HIGH)
        cascade = mgr.get_cascade_config()
        assert cascade.count == 4

    def test_get_cascade_count_with_cascades(self):
        mgr = ShadowTierManager(QualityTier.HIGH)
        assert mgr.get_cascade_count() == 4

    def test_get_cascade_count_without_cascades(self):
        mgr = ShadowTierManager(QualityTier.LOW)
        assert mgr.get_cascade_count() == 1


class TestShadowCasterManagement:
    """Tests for shadow caster registration."""

    def test_can_add_spot_shadow(self):
        mgr = ShadowTierManager(QualityTier.LOW)
        assert mgr.can_add_spot_shadow() is True

    def test_can_add_spot_shadow_at_limit(self):
        mgr = ShadowTierManager(QualityTier.LOW)  # 2 max
        mgr.register_spot_shadow()
        mgr.register_spot_shadow()
        assert mgr.can_add_spot_shadow() is False

    def test_can_add_point_shadow_low_tier(self):
        mgr = ShadowTierManager(QualityTier.LOW)  # 0 max
        assert mgr.can_add_point_shadow() is False

    def test_can_add_point_shadow_medium_tier(self):
        mgr = ShadowTierManager(QualityTier.MEDIUM)  # 1 max
        assert mgr.can_add_point_shadow() is True

    def test_register_spot_shadow(self):
        mgr = ShadowTierManager(QualityTier.MEDIUM)
        assert mgr.register_spot_shadow() is True
        assert mgr.usage_stats.active_spot_shadows == 1

    def test_register_point_shadow(self):
        mgr = ShadowTierManager(QualityTier.HIGH)
        assert mgr.register_point_shadow() is True
        assert mgr.usage_stats.active_point_shadows == 1

    def test_unregister_spot_shadow(self):
        mgr = ShadowTierManager(QualityTier.MEDIUM)
        mgr.register_spot_shadow()
        mgr.unregister_spot_shadow()
        assert mgr.usage_stats.active_spot_shadows == 0


class TestFrameLifecycle:
    """Tests for frame lifecycle."""

    def test_begin_frame_resets_cascade_renders(self):
        mgr = ShadowTierManager(QualityTier.HIGH)
        mgr.record_cascade_render()
        mgr.record_cascade_render()
        mgr.begin_frame()
        assert mgr.usage_stats.cascade_renders_this_frame == 0

    def test_record_cascade_render(self):
        mgr = ShadowTierManager(QualityTier.HIGH)
        mgr.record_cascade_render()
        assert mgr.usage_stats.cascade_renders_this_frame == 1


class TestFeatureOverrides:
    """Tests for feature overrides."""

    def test_override_feature_enable(self):
        mgr = ShadowTierManager(QualityTier.LOW)
        assert not mgr.is_feature_enabled(ShadowFeature.CONTACT_SHADOWS)
        mgr.override_feature(ShadowFeature.CONTACT_SHADOWS, True)
        assert mgr.is_feature_enabled(ShadowFeature.CONTACT_SHADOWS)

    def test_override_feature_disable(self):
        mgr = ShadowTierManager(QualityTier.HIGH)
        assert mgr.is_feature_enabled(ShadowFeature.CONTACT_SHADOWS)
        mgr.override_feature(ShadowFeature.CONTACT_SHADOWS, False)
        assert not mgr.is_feature_enabled(ShadowFeature.CONTACT_SHADOWS)

    def test_clear_overrides(self):
        mgr = ShadowTierManager(QualityTier.LOW)
        mgr.override_feature(ShadowFeature.CONTACT_SHADOWS, True)
        mgr.clear_overrides()
        assert not mgr.is_feature_enabled(ShadowFeature.CONTACT_SHADOWS)


class TestFallbackFilter:
    """Tests for fallback filter selection."""

    def test_select_fallback_filter_ultra_with_rt(self):
        mgr = ShadowTierManager(QualityTier.ULTRA)
        mgr.set_rt_available(True)
        assert mgr.select_fallback_filter() == ShadowFilterMethod.RT

    def test_select_fallback_filter_ultra_without_rt(self):
        mgr = ShadowTierManager(QualityTier.ULTRA)
        mgr.set_rt_available(False)
        assert mgr.select_fallback_filter() == ShadowFilterMethod.PCSS

    def test_select_fallback_filter_high(self):
        mgr = ShadowTierManager(QualityTier.HIGH)
        assert mgr.select_fallback_filter() == ShadowFilterMethod.VSM

    def test_select_fallback_filter_low(self):
        mgr = ShadowTierManager(QualityTier.LOW)
        assert mgr.select_fallback_filter() == ShadowFilterMethod.PCF


class TestStatusDict:
    """Tests for status dictionary."""

    def test_get_status_dict(self):
        mgr = ShadowTierManager(QualityTier.HIGH)
        mgr.register_spot_shadow()
        mgr.set_rt_available(False)

        status = mgr.get_status_dict()
        assert status["tier"] == "HIGH"
        assert status["resolution"] == 2048
        assert status["filter_method"] == "VSM"
        assert status["cascade_count"] == 4
        assert status["uses_cascades"] is True
        assert status["contact_shadows_enabled"] is True
        assert status["active_spot_shadows"] == 1
        assert status["rt_available"] is False


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_tier_for_resolution_512(self):
        assert get_tier_for_resolution(512) == QualityTier.LOW

    def test_get_tier_for_resolution_1024(self):
        assert get_tier_for_resolution(1024) == QualityTier.MEDIUM

    def test_get_tier_for_resolution_2048(self):
        assert get_tier_for_resolution(2048) == QualityTier.HIGH

    def test_get_tier_for_resolution_4096(self):
        assert get_tier_for_resolution(4096) == QualityTier.ULTRA

    def test_get_tier_for_filter_pcf(self):
        assert get_tier_for_filter(ShadowFilterMethod.PCF) == QualityTier.LOW

    def test_get_tier_for_filter_vsm(self):
        assert get_tier_for_filter(ShadowFilterMethod.VSM) == QualityTier.HIGH

    def test_get_tier_for_filter_rt(self):
        assert get_tier_for_filter(ShadowFilterMethod.RT) == QualityTier.ULTRA


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_estimate_low_tier_memory(self):
        cfg = create_low_tier_config()
        memory = estimate_shadow_memory(cfg)
        assert memory > 0

    def test_estimate_high_tier_memory(self):
        cfg = create_high_tier_config()
        memory_high = estimate_shadow_memory(cfg)
        memory_low = estimate_shadow_memory(create_low_tier_config())
        assert memory_high > memory_low

    def test_estimate_includes_rt_buffers(self):
        cfg = create_ultra_tier_config()
        memory = estimate_shadow_memory(cfg)
        # RT config adds significant buffer
        assert memory > estimate_shadow_memory(create_high_tier_config())


class TestCascadeSplits:
    """Tests for cascade split calculation."""

    def test_cascade_splits_count(self):
        splits = calculate_cascade_splits(0.1, 100.0, 4)
        assert len(splits) == 5  # count + 1 for far plane

    def test_cascade_splits_bounds(self):
        splits = calculate_cascade_splits(0.1, 100.0, 4)
        assert splits[0] == pytest.approx(0.1, rel=0.1)
        assert splits[-1] == pytest.approx(100.0, rel=0.1)

    def test_cascade_splits_monotonic(self):
        splits = calculate_cascade_splits(0.1, 100.0, 4)
        for i in range(len(splits) - 1):
            assert splits[i] < splits[i + 1]


class TestIntegration:
    """Integration tests."""

    def test_full_tier_progression(self):
        mgr = ShadowTierManager(QualityTier.LOW)
        for tier in [QualityTier.MEDIUM, QualityTier.HIGH, QualityTier.ULTRA]:
            mgr.set_tier(tier)
            assert mgr.current_tier == tier

    def test_shadow_caster_lifecycle(self):
        mgr = ShadowTierManager(QualityTier.HIGH)  # 8 spot, 4 point

        for _ in range(8):
            assert mgr.register_spot_shadow() is True
        assert mgr.register_spot_shadow() is False

        for _ in range(4):
            assert mgr.register_point_shadow() is True
        assert mgr.register_point_shadow() is False

        mgr.unregister_spot_shadow()
        assert mgr.register_spot_shadow() is True

    def test_tier_change_resets_usage(self):
        mgr = ShadowTierManager(QualityTier.HIGH)
        mgr.register_spot_shadow()
        mgr.set_tier(QualityTier.ULTRA)
        assert mgr.usage_stats.active_spot_shadows == 0
