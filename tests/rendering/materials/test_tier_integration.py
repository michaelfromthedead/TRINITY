"""Tests for T-CC-1.3: Quality tier integration for Materials subsystem."""
import pytest
from trinity.types import QualityTier
from engine.rendering.materials.tier_integration import (
    MaterialFeature,
    TextureFilterMode,
    TextureConfig,
    SSSConfig,
    ParallaxConfig,
    MaterialTierConfig,
    VariantUsageStats,
    TierChangeListener,
    MaterialTierManager,
    create_low_tier_config,
    create_medium_tier_config,
    create_high_tier_config,
    create_ultra_tier_config,
    TIER_CONFIGS,
    get_tier_for_features,
    get_tier_for_variant_count,
    estimate_material_memory,
    create_shader_permutation_key,
)


class TestEnums:
    """Tests for enum definitions."""

    def test_material_feature_count(self):
        features = list(MaterialFeature)
        assert len(features) == 15

    def test_texture_filter_modes(self):
        assert TextureFilterMode.NEAREST in TextureFilterMode
        assert TextureFilterMode.BILINEAR in TextureFilterMode
        assert TextureFilterMode.TRILINEAR in TextureFilterMode


class TestTextureConfig:
    """Tests for TextureConfig."""

    def test_default_values(self):
        cfg = TextureConfig()
        assert cfg.max_size == 1024
        assert cfg.lod_bias == 0.0
        assert cfg.filter_mode == TextureFilterMode.TRILINEAR
        assert cfg.anisotropic_level == 4

    def test_custom_values(self):
        cfg = TextureConfig(max_size=512, lod_bias=2.0, anisotropic_level=1)
        assert cfg.max_size == 512
        assert cfg.lod_bias == 2.0


class TestSSSConfig:
    """Tests for SSSConfig."""

    def test_default_disabled(self):
        cfg = SSSConfig()
        assert cfg.enabled is False

    def test_enabled_with_samples(self):
        cfg = SSSConfig(enabled=True, samples=16)
        assert cfg.samples == 16


class TestParallaxConfig:
    """Tests for ParallaxConfig."""

    def test_default_disabled(self):
        cfg = ParallaxConfig()
        assert cfg.enabled is False

    def test_pom_method(self):
        cfg = ParallaxConfig(enabled=True, steps=32, method="pom")
        assert cfg.method == "pom"


class TestMaterialTierConfig:
    """Tests for MaterialTierConfig."""

    def test_supports_advanced_shading_false(self):
        cfg = create_low_tier_config()
        assert cfg.supports_advanced_shading is False

    def test_supports_advanced_shading_true(self):
        cfg = create_high_tier_config()
        assert cfg.supports_advanced_shading is True

    def test_feature_count(self):
        cfg = create_low_tier_config()
        assert cfg.feature_count == 3

    def test_unlimited_variants_false(self):
        cfg = create_high_tier_config()
        assert cfg.unlimited_variants is False

    def test_unlimited_variants_true(self):
        cfg = create_ultra_tier_config()
        assert cfg.unlimited_variants is True


class TestTierConfigs:
    """Tests for tier config factory functions."""

    def test_low_tier_config(self):
        cfg = create_low_tier_config()
        assert cfg.tier == QualityTier.LOW
        assert cfg.max_variants == 1
        assert MaterialFeature.BASE_COLOR in cfg.enabled_features
        assert MaterialFeature.NORMAL_MAPPING in cfg.enabled_features
        assert cfg.texture_config.max_size == 512
        assert cfg.sss_config.enabled is False

    def test_low_tier_no_sss(self):
        cfg = create_low_tier_config()
        assert MaterialFeature.SUBSURFACE_SCATTERING not in cfg.enabled_features

    def test_medium_tier_config(self):
        cfg = create_medium_tier_config()
        assert cfg.tier == QualityTier.MEDIUM
        assert cfg.max_variants == 3
        assert MaterialFeature.AMBIENT_OCCLUSION in cfg.enabled_features
        assert MaterialFeature.EMISSIVE in cfg.enabled_features
        assert cfg.texture_config.max_size == 1024

    def test_high_tier_config(self):
        cfg = create_high_tier_config()
        assert cfg.tier == QualityTier.HIGH
        assert cfg.max_variants == 10
        assert MaterialFeature.SUBSURFACE_SCATTERING in cfg.enabled_features
        assert MaterialFeature.CLEAR_COAT in cfg.enabled_features
        assert MaterialFeature.ANISOTROPY in cfg.enabled_features
        assert cfg.sss_config.enabled is True
        assert cfg.parallax_config.enabled is True

    def test_ultra_tier_config(self):
        cfg = create_ultra_tier_config()
        assert cfg.tier == QualityTier.ULTRA
        assert cfg.max_variants == -1  # Unlimited
        assert MaterialFeature.SHEEN in cfg.enabled_features
        assert MaterialFeature.TRANSMISSION in cfg.enabled_features
        assert MaterialFeature.IRIDESCENCE in cfg.enabled_features
        assert MaterialFeature.TESSELLATION in cfg.enabled_features
        assert cfg.sss_config.samples == 16

    def test_all_tiers_in_dict(self):
        for tier in QualityTier:
            assert tier in TIER_CONFIGS

    def test_tier_texture_sizes_increase(self):
        low = create_low_tier_config()
        medium = create_medium_tier_config()
        high = create_high_tier_config()
        ultra = create_ultra_tier_config()
        assert low.texture_config.max_size < medium.texture_config.max_size
        assert medium.texture_config.max_size < high.texture_config.max_size
        assert high.texture_config.max_size < ultra.texture_config.max_size

    def test_tier_variant_counts_increase(self):
        low = create_low_tier_config()
        medium = create_medium_tier_config()
        high = create_high_tier_config()
        assert low.max_variants < medium.max_variants
        assert medium.max_variants < high.max_variants


class TestVariantUsageStats:
    """Tests for variant usage statistics."""

    def test_initial_values(self):
        stats = VariantUsageStats()
        assert stats.active_variants == 0
        assert stats.total_materials == 0

    def test_variant_breakdown(self):
        stats = VariantUsageStats(variant_breakdown={"standard": 5, "skin": 2})
        assert stats.variant_breakdown["standard"] == 5


class TestTierChangeListener:
    """Tests for tier change listener."""

    def test_listener_receives_notification(self):
        class TestListener(TierChangeListener):
            def __init__(self):
                self.changes = []

            def on_tier_changed(self, old_tier, new_tier, config):
                self.changes.append((old_tier, new_tier))

        mgr = MaterialTierManager(QualityTier.LOW)
        listener = TestListener()
        mgr.add_listener(listener)
        mgr.set_tier(QualityTier.HIGH)

        assert len(listener.changes) == 1
        assert listener.changes[0] == (QualityTier.LOW, QualityTier.HIGH)


class TestMaterialTierManager:
    """Tests for MaterialTierManager."""

    def test_default_tier(self):
        mgr = MaterialTierManager()
        assert mgr.current_tier == QualityTier.MEDIUM

    def test_custom_initial_tier(self):
        mgr = MaterialTierManager(QualityTier.HIGH)
        assert mgr.current_tier == QualityTier.HIGH

    def test_set_tier(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        mgr.set_tier(QualityTier.ULTRA)
        assert mgr.current_tier == QualityTier.ULTRA

    def test_set_same_tier_no_op(self):
        mgr = MaterialTierManager(QualityTier.MEDIUM)
        changes = []
        listener = type("L", (), {"on_tier_changed": lambda s, o, n, c: changes.append(1)})()
        mgr.add_listener(listener)
        mgr.set_tier(QualityTier.MEDIUM)
        assert len(changes) == 0

    def test_is_feature_enabled(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        assert mgr.is_feature_enabled(MaterialFeature.BASE_COLOR)
        assert not mgr.is_feature_enabled(MaterialFeature.SUBSURFACE_SCATTERING)

    def test_get_max_variants(self):
        low = MaterialTierManager(QualityTier.LOW)
        high = MaterialTierManager(QualityTier.HIGH)
        assert low.get_max_variants() == 1
        assert high.get_max_variants() == 10


class TestVariantManagement:
    """Tests for variant registration."""

    def test_can_add_variant_within_limit(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        assert mgr.can_add_variant() is True

    def test_can_add_variant_at_limit(self):
        mgr = MaterialTierManager(QualityTier.LOW)  # 1 variant max
        mgr.register_variant("standard")
        assert mgr.can_add_variant() is False

    def test_can_add_variant_unlimited(self):
        mgr = MaterialTierManager(QualityTier.ULTRA)
        for i in range(100):
            assert mgr.register_variant(f"variant_{i}") is True

    def test_register_variant(self):
        mgr = MaterialTierManager(QualityTier.MEDIUM)
        assert mgr.register_variant("standard") is True
        assert mgr.usage_stats.active_variants == 1

    def test_register_variant_over_limit(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        mgr.register_variant("standard")
        assert mgr.register_variant("skin") is False

    def test_unregister_variant(self):
        mgr = MaterialTierManager(QualityTier.MEDIUM)
        mgr.register_variant("standard")
        mgr.unregister_variant("standard")
        assert mgr.usage_stats.active_variants == 0

    def test_unregister_variant_underflow(self):
        mgr = MaterialTierManager(QualityTier.MEDIUM)
        mgr.unregister_variant("nonexistent")
        assert mgr.usage_stats.active_variants == 0


class TestMaterialManagement:
    """Tests for material registration."""

    def test_register_material(self):
        mgr = MaterialTierManager()
        mgr.register_material()
        assert mgr.usage_stats.total_materials == 1

    def test_unregister_material(self):
        mgr = MaterialTierManager()
        mgr.register_material()
        mgr.unregister_material()
        assert mgr.usage_stats.total_materials == 0


class TestConfigAccessors:
    """Tests for configuration accessors."""

    def test_get_texture_config(self):
        mgr = MaterialTierManager(QualityTier.HIGH)
        tex = mgr.get_texture_config()
        assert tex.max_size == 2048

    def test_get_sss_config(self):
        mgr = MaterialTierManager(QualityTier.HIGH)
        sss = mgr.get_sss_config()
        assert sss.enabled is True
        assert sss.samples == 8

    def test_get_parallax_config(self):
        mgr = MaterialTierManager(QualityTier.HIGH)
        px = mgr.get_parallax_config()
        assert px.enabled is True

    def test_get_max_texture_size(self):
        low = MaterialTierManager(QualityTier.LOW)
        assert low.get_max_texture_size() == 512

    def test_get_lod_bias(self):
        low = MaterialTierManager(QualityTier.LOW)
        assert low.get_lod_bias() == 2.0

    def test_get_anisotropic_level(self):
        ultra = MaterialTierManager(QualityTier.ULTRA)
        assert ultra.get_anisotropic_level() == 16


class TestFeatureOverrides:
    """Tests for feature overrides."""

    def test_override_feature_enable(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        assert not mgr.is_feature_enabled(MaterialFeature.SUBSURFACE_SCATTERING)
        mgr.override_feature(MaterialFeature.SUBSURFACE_SCATTERING, True)
        assert mgr.is_feature_enabled(MaterialFeature.SUBSURFACE_SCATTERING)

    def test_override_feature_disable(self):
        mgr = MaterialTierManager(QualityTier.HIGH)
        assert mgr.is_feature_enabled(MaterialFeature.SUBSURFACE_SCATTERING)
        mgr.override_feature(MaterialFeature.SUBSURFACE_SCATTERING, False)
        assert not mgr.is_feature_enabled(MaterialFeature.SUBSURFACE_SCATTERING)

    def test_clear_overrides(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        mgr.override_feature(MaterialFeature.SUBSURFACE_SCATTERING, True)
        mgr.clear_overrides()
        assert not mgr.is_feature_enabled(MaterialFeature.SUBSURFACE_SCATTERING)


class TestRequiredFeatures:
    """Tests for required features lookup."""

    def test_get_required_features_standard(self):
        mgr = MaterialTierManager(QualityTier.MEDIUM)
        features = mgr.get_required_features("standard")
        assert MaterialFeature.BASE_COLOR in features
        assert MaterialFeature.NORMAL_MAPPING in features

    def test_get_required_features_skin_low_tier(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        features = mgr.get_required_features("skin")
        # SSS not available on LOW tier
        assert MaterialFeature.SUBSURFACE_SCATTERING not in features

    def test_get_required_features_skin_high_tier(self):
        mgr = MaterialTierManager(QualityTier.HIGH)
        features = mgr.get_required_features("skin")
        assert MaterialFeature.SUBSURFACE_SCATTERING in features


class TestFallbackVariants:
    """Tests for fallback variant selection."""

    def test_select_fallback_variant_standard(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        variant = mgr.select_fallback_variant("standard")
        assert variant == "standard"

    def test_select_fallback_variant_skin_low(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        variant = mgr.select_fallback_variant("skin")
        # Falls back to skin_simple since SSS not available
        assert variant in ("skin_simple", "standard")

    def test_select_fallback_variant_skin_high(self):
        mgr = MaterialTierManager(QualityTier.HIGH)
        variant = mgr.select_fallback_variant("skin")
        assert variant == "skin_sss"  # Full SSS available


class TestStatusDict:
    """Tests for status dictionary."""

    def test_get_status_dict(self):
        mgr = MaterialTierManager(QualityTier.HIGH)
        mgr.register_variant("standard")
        mgr.register_material()

        status = mgr.get_status_dict()
        assert status["tier"] == "HIGH"
        assert status["max_variants"] == 10
        assert status["active_variants"] == 1
        assert status["total_materials"] == 1
        assert status["supports_advanced_shading"] is True
        assert status["sss_enabled"] is True


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_tier_for_features_basic(self):
        features = {MaterialFeature.BASE_COLOR, MaterialFeature.NORMAL_MAPPING}
        tier = get_tier_for_features(features)
        assert tier == QualityTier.LOW

    def test_get_tier_for_features_sss(self):
        features = {MaterialFeature.SUBSURFACE_SCATTERING}
        tier = get_tier_for_features(features)
        assert tier == QualityTier.HIGH

    def test_get_tier_for_features_ultra(self):
        features = {MaterialFeature.IRIDESCENCE, MaterialFeature.TESSELLATION}
        tier = get_tier_for_features(features)
        assert tier == QualityTier.ULTRA

    def test_get_tier_for_variant_count_1(self):
        assert get_tier_for_variant_count(1) == QualityTier.LOW

    def test_get_tier_for_variant_count_3(self):
        assert get_tier_for_variant_count(3) == QualityTier.MEDIUM

    def test_get_tier_for_variant_count_10(self):
        assert get_tier_for_variant_count(10) == QualityTier.HIGH

    def test_get_tier_for_variant_count_unlimited(self):
        assert get_tier_for_variant_count(100) == QualityTier.ULTRA


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_estimate_low_tier_memory(self):
        cfg = create_low_tier_config()
        memory = estimate_material_memory(cfg, 100)
        assert memory > 0

    def test_estimate_high_tier_memory(self):
        cfg = create_high_tier_config()
        memory_high = estimate_material_memory(cfg, 100)
        memory_low = estimate_material_memory(create_low_tier_config(), 100)
        assert memory_high > memory_low

    def test_estimate_scales_with_materials(self):
        cfg = create_medium_tier_config()
        memory_10 = estimate_material_memory(cfg, 10)
        memory_100 = estimate_material_memory(cfg, 100)
        assert memory_100 > memory_10


class TestShaderPermutationKey:
    """Tests for shader permutation key generation."""

    def test_empty_features(self):
        key = create_shader_permutation_key(set())
        assert key == 0

    def test_single_feature(self):
        key = create_shader_permutation_key({MaterialFeature.BASE_COLOR})
        assert key == 1

    def test_multiple_features(self):
        key1 = create_shader_permutation_key({MaterialFeature.BASE_COLOR})
        key2 = create_shader_permutation_key({MaterialFeature.BASE_COLOR, MaterialFeature.NORMAL_MAPPING})
        assert key2 > key1
        assert key2 == key1 | 2  # Normal mapping is bit 1

    def test_unique_keys(self):
        keys = set()
        for feature in MaterialFeature:
            key = create_shader_permutation_key({feature})
            assert key not in keys
            keys.add(key)


class TestIntegration:
    """Integration tests."""

    def test_full_tier_progression(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        for tier in [QualityTier.MEDIUM, QualityTier.HIGH, QualityTier.ULTRA]:
            mgr.set_tier(tier)
            assert mgr.current_tier == tier

    def test_variant_lifecycle(self):
        mgr = MaterialTierManager(QualityTier.MEDIUM)  # 3 variants

        assert mgr.register_variant("standard") is True
        assert mgr.register_variant("skin") is True
        assert mgr.register_variant("glass") is True
        assert mgr.register_variant("metal") is False  # At limit

        mgr.unregister_variant("skin")
        assert mgr.register_variant("metal") is True

    def test_tier_change_preserves_stats(self):
        mgr = MaterialTierManager(QualityTier.LOW)
        mgr.register_material()
        mgr.register_material()

        mgr.set_tier(QualityTier.HIGH)
        assert mgr.usage_stats.total_materials == 2
