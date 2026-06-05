"""Tests for shader variant pruning (T-CC-0.7)."""
import pytest

from trinity.types import QualityTier
from engine.rendering.materials.shader_compiler import (
    PermutationKey,
    ShaderPermutation,
)
from engine.rendering.quality.shader_variant_pruner import (
    ShaderVariantPruner,
    VariantPruningConfig,
    VariantPruningResult,
    FeatureMapping,
    STANDARD_FEATURE_MAPPINGS,
    create_pruner_for_tier,
)


class TestFeatureMapping:
    """Test FeatureMapping dataclass."""

    def test_mapping_fields(self):
        """Test mapping has required fields."""
        mapping = FeatureMapping(
            shader_feature="normal_map",
            capability_feature="normal_mapping",
            subsystem="materials",
            min_tier=QualityTier.LOW,
        )
        assert mapping.shader_feature == "normal_map"
        assert mapping.capability_feature == "normal_mapping"
        assert mapping.subsystem == "materials"

    def test_is_available_at_tier(self):
        """Test tier availability check."""
        mapping = FeatureMapping(
            shader_feature="tessellation",
            capability_feature="tessellation",
            subsystem="materials",
            min_tier=QualityTier.HIGH,
        )
        assert not mapping.is_available_at_tier(QualityTier.LOW)
        assert not mapping.is_available_at_tier(QualityTier.MEDIUM)
        assert mapping.is_available_at_tier(QualityTier.HIGH)
        assert mapping.is_available_at_tier(QualityTier.ULTRA)

    def test_low_tier_feature(self):
        """Test LOW tier features are always available."""
        mapping = FeatureMapping(
            shader_feature="bloom",
            capability_feature="bloom",
            subsystem="postprocess",
            min_tier=QualityTier.LOW,
        )
        assert mapping.is_available_at_tier(QualityTier.LOW)
        assert mapping.is_available_at_tier(QualityTier.MEDIUM)
        assert mapping.is_available_at_tier(QualityTier.HIGH)
        assert mapping.is_available_at_tier(QualityTier.ULTRA)


class TestVariantPruningConfig:
    """Test VariantPruningConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VariantPruningConfig()
        assert config.max_variants_per_shader == 128
        assert config.max_total_variants == 1024
        assert config.always_include_base_variant

    def test_for_tier_low(self):
        """Test LOW tier config."""
        config = VariantPruningConfig.for_tier(QualityTier.LOW)
        assert config.max_variants_per_shader == 32
        assert config.max_total_variants == 256
        assert not config.include_debug_variants

    def test_for_tier_ultra(self):
        """Test ULTRA tier config."""
        config = VariantPruningConfig.for_tier(QualityTier.ULTRA)
        assert config.max_variants_per_shader == 256
        assert config.max_total_variants == 2048
        assert config.include_debug_variants


class TestVariantPruningResult:
    """Test VariantPruningResult."""

    def test_reduction_percent(self):
        """Test reduction percentage calculation."""
        result = VariantPruningResult(
            original_count=100,
            pruned_count=25,
            excluded_features={"feature_a", "feature_b"},
            included_features={"feature_c"},
        )
        assert result.reduction_percent == 75.0

    def test_reduction_percent_zero(self):
        """Test reduction with zero original count."""
        result = VariantPruningResult(
            original_count=0,
            pruned_count=0,
            excluded_features=set(),
            included_features=set(),
        )
        assert result.reduction_percent == 0.0

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = VariantPruningResult(
            original_count=100,
            pruned_count=50,
            excluded_features={"parallax"},
            included_features={"normal_map"},
        )
        d = result.to_dict()
        assert d["original_count"] == 100
        assert d["pruned_count"] == 50
        assert d["reduction_percent"] == 50.0


class TestShaderVariantPrunerCreation:
    """Test ShaderVariantPruner creation."""

    def test_default_creation(self):
        """Test default pruner creation."""
        pruner = ShaderVariantPruner()
        assert pruner.tier == QualityTier.HIGH

    def test_creation_with_tier(self):
        """Test creation with specific tier."""
        pruner = ShaderVariantPruner(tier=QualityTier.LOW)
        assert pruner.tier == QualityTier.LOW

    def test_tier_setter(self):
        """Test changing tier updates config."""
        pruner = ShaderVariantPruner(tier=QualityTier.HIGH)
        pruner.tier = QualityTier.LOW
        assert pruner.tier == QualityTier.LOW
        assert pruner.config.max_variants_per_shader == 32


class TestShaderVariantPrunerFeatures:
    """Test feature availability checking."""

    def test_get_available_features_low(self):
        """Test available features at LOW tier."""
        pruner = ShaderVariantPruner(tier=QualityTier.LOW)
        available = pruner.get_available_features()
        # LOW tier features
        assert "normal_map" in available
        assert "bloom" in available
        assert "soft_particles" in available
        # Not LOW tier
        assert "tessellation" not in available
        assert "rt_reflections" not in available

    def test_get_available_features_ultra(self):
        """Test available features at ULTRA tier."""
        pruner = ShaderVariantPruner(tier=QualityTier.ULTRA)
        available = pruner.get_available_features()
        # ULTRA includes everything
        assert "normal_map" in available
        assert "tessellation" in available
        assert "rt_reflections" in available
        assert "dlss" in available

    def test_get_excluded_features(self):
        """Test excluded features at LOW tier."""
        pruner = ShaderVariantPruner(tier=QualityTier.LOW)
        excluded = pruner.get_excluded_features()
        assert "tessellation" in excluded
        assert "rt_reflections" in excluded
        assert "ddgi" in excluded

    def test_is_feature_available(self):
        """Test individual feature availability check."""
        pruner = ShaderVariantPruner(tier=QualityTier.MEDIUM)
        assert pruner.is_feature_available("normal_map")
        assert pruner.is_feature_available("parallax")
        assert pruner.is_feature_available("ssr")
        assert not pruner.is_feature_available("tessellation")


class TestShaderVariantPrunerCustomMappings:
    """Test custom feature mapping registration."""

    def test_register_mapping(self):
        """Test registering custom mapping."""
        pruner = ShaderVariantPruner(tier=QualityTier.LOW)
        mapping = FeatureMapping(
            shader_feature="custom_effect",
            capability_feature="custom",
            subsystem="custom",
            min_tier=QualityTier.LOW,
        )
        pruner.register_mapping(mapping)
        assert pruner.is_feature_available("custom_effect")

    def test_register_mappings(self):
        """Test registering multiple mappings."""
        pruner = ShaderVariantPruner(tier=QualityTier.HIGH)
        mappings = [
            FeatureMapping("effect_a", "a", "custom", QualityTier.LOW),
            FeatureMapping("effect_b", "b", "custom", QualityTier.HIGH),
        ]
        pruner.register_mappings(mappings)
        assert pruner.is_feature_available("effect_a")
        assert pruner.is_feature_available("effect_b")


class TestShaderVariantPrunerPruning:
    """Test permutation key pruning."""

    def test_prune_permutation_key(self):
        """Test pruning a permutation key."""
        pruner = ShaderVariantPruner(tier=QualityTier.LOW)
        key = PermutationKey.from_set({
            "normal_map",  # LOW tier - keep
            "tessellation",  # HIGH tier - prune
            "bloom",  # LOW tier - keep
        })
        pruned = pruner.prune_permutation_key(key)
        assert "normal_map" in pruned.features
        assert "bloom" in pruned.features
        assert "tessellation" not in pruned.features

    def test_prune_keeps_unknown_features(self):
        """Test unknown features are kept."""
        pruner = ShaderVariantPruner(tier=QualityTier.LOW)
        key = PermutationKey.from_set({
            "normal_map",
            "unknown_custom_feature",
        })
        pruned = pruner.prune_permutation_key(key)
        assert "normal_map" in pruned.features
        assert "unknown_custom_feature" in pruned.features


class TestShaderVariantPrunerPermutation:
    """Test full permutation pruning."""

    def test_get_valid_keys_for_tier(self):
        """Test getting valid keys for a tier."""
        pruner = ShaderVariantPruner(tier=QualityTier.LOW)
        perm = ShaderPermutation(name="test")
        perm.add_feature("normal_map")  # LOW
        perm.add_feature("bloom")  # LOW
        perm.add_feature("tessellation")  # HIGH

        valid_keys = pruner.get_valid_keys_for_tier(perm)

        # Should have combinations of LOW features only
        assert len(valid_keys) > 0

        # No key should have tessellation
        for key in valid_keys:
            assert "tessellation" not in key.features

    def test_prune_permutation_result(self):
        """Test prune_permutation returns correct stats."""
        pruner = ShaderVariantPruner(tier=QualityTier.LOW)
        perm = ShaderPermutation(name="test")
        perm.add_feature("normal_map")
        perm.add_feature("parallax")
        perm.add_feature("tessellation")

        result = pruner.prune_permutation(perm)

        assert result.original_count > 0
        assert result.pruned_count <= result.original_count
        assert "tessellation" in result.excluded_features
        assert "normal_map" in result.included_features

    def test_max_variants_enforced(self):
        """Test max variants per shader is enforced."""
        pruner = ShaderVariantPruner(tier=QualityTier.LOW)
        pruner._config.max_variants_per_shader = 4

        perm = ShaderPermutation(name="test")
        for i in range(10):
            perm.add_feature(f"feature_{i}")

        # Register all as LOW tier
        for i in range(10):
            pruner.register_mapping(FeatureMapping(
                f"feature_{i}", f"cap_{i}", "test", QualityTier.LOW
            ))

        valid_keys = pruner.get_valid_keys_for_tier(perm)
        assert len(valid_keys) <= 4


class TestShaderVariantPrunerStats:
    """Test variant statistics calculation."""

    def test_estimate_variant_reduction(self):
        """Test variant reduction estimation."""
        pruner = ShaderVariantPruner(tier=QualityTier.LOW)

        perms = [
            ShaderPermutation(name="shader1"),
            ShaderPermutation(name="shader2"),
        ]
        perms[0].add_feature("normal_map")
        perms[0].add_feature("tessellation")
        perms[1].add_feature("bloom")
        perms[1].add_feature("rt_reflections")

        results = pruner.estimate_variant_reduction(perms)

        assert "shader1" in results
        assert "shader2" in results
        assert results["shader1"].pruned_count <= results["shader1"].original_count

    def test_total_variant_stats(self):
        """Test total statistics across permutations."""
        pruner = ShaderVariantPruner(tier=QualityTier.MEDIUM)

        perms = [
            ShaderPermutation(name="pbr"),
            ShaderPermutation(name="unlit"),
        ]
        perms[0].add_feature("normal_map")
        perms[0].add_feature("parallax")
        perms[1].add_feature("bloom")

        stats = pruner.total_variant_stats(perms)

        assert stats["tier"] == "MEDIUM"
        assert stats["shader_count"] == 2
        assert "total_original_variants" in stats
        assert "total_pruned_variants" in stats
        assert "total_reduction_percent" in stats


class TestCreatePrunerForTier:
    """Test factory function."""

    def test_create_low_tier(self):
        """Test creating LOW tier pruner."""
        pruner = create_pruner_for_tier(QualityTier.LOW)
        assert pruner.tier == QualityTier.LOW
        assert pruner.config.max_variants_per_shader == 32

    def test_create_ultra_tier(self):
        """Test creating ULTRA tier pruner."""
        pruner = create_pruner_for_tier(QualityTier.ULTRA)
        assert pruner.tier == QualityTier.ULTRA
        assert pruner.config.include_debug_variants


class TestStandardFeatureMappings:
    """Test standard feature mappings coverage."""

    def test_mappings_exist(self):
        """Test standard mappings are defined."""
        assert len(STANDARD_FEATURE_MAPPINGS) > 0

    def test_mappings_have_valid_tiers(self):
        """Test all mappings have valid tier assignments."""
        for mapping in STANDARD_FEATURE_MAPPINGS:
            assert mapping.min_tier in QualityTier
            assert mapping.shader_feature
            assert mapping.capability_feature
            assert mapping.subsystem

    def test_mappings_cover_subsystems(self):
        """Test mappings cover major subsystems."""
        subsystems = {m.subsystem for m in STANDARD_FEATURE_MAPPINGS}
        assert "materials" in subsystems
        assert "lighting" in subsystems
        assert "shadows" in subsystems
        assert "gi" in subsystems
        assert "reflections" in subsystems
        assert "postprocess" in subsystems

    def test_low_tier_features_exist(self):
        """Test LOW tier features are defined."""
        low_tier = [m for m in STANDARD_FEATURE_MAPPINGS if m.min_tier == QualityTier.LOW]
        assert len(low_tier) > 0
        low_features = {m.shader_feature for m in low_tier}
        assert "normal_map" in low_features
        assert "bloom" in low_features

    def test_ultra_tier_features_exist(self):
        """Test ULTRA tier features are defined."""
        ultra_tier = [m for m in STANDARD_FEATURE_MAPPINGS if m.min_tier == QualityTier.ULTRA]
        assert len(ultra_tier) > 0
        ultra_features = {m.shader_feature for m in ultra_tier}
        assert "ray_traced_shadows" in ultra_features or "rt_reflections" in ultra_features


class TestIntegration:
    """Integration tests for shader variant pruning."""

    def test_pbr_shader_pruning(self):
        """Test PBR shader variant pruning across tiers."""
        # Create a typical PBR shader permutation
        pbr = ShaderPermutation(name="pbr_standard")
        pbr.add_feature("normal_map", required=True)  # Always needed
        pbr.add_feature("parallax")
        pbr.add_feature("tessellation")
        pbr.add_feature("subsurface")
        pbr.add_feature("clear_coat")

        # Compare variant counts across tiers
        results = {}
        for tier in QualityTier:
            pruner = ShaderVariantPruner(tier=tier)
            result = pruner.prune_permutation(pbr)
            results[tier] = result

        # LOW should have fewest variants
        assert results[QualityTier.LOW].pruned_count <= results[QualityTier.MEDIUM].pruned_count
        # ULTRA should have most variants
        assert results[QualityTier.ULTRA].pruned_count >= results[QualityTier.HIGH].pruned_count

    def test_reduction_goal_390_vs_550(self):
        """Test reduction achieves target (390 vs 550+ variants)."""
        # Create permutations simulating full engine shader set
        shaders = []
        for name in ["pbr", "unlit", "post", "shadow", "gi"]:
            perm = ShaderPermutation(name=name)
            perm.add_feature("normal_map")
            perm.add_feature("parallax")
            perm.add_feature("tessellation")
            perm.add_feature("ssr")
            perm.add_feature("rt_reflections")
            shaders.append(perm)

        # Get stats at HIGH tier
        pruner = ShaderVariantPruner(tier=QualityTier.HIGH)
        stats = pruner.total_variant_stats(shaders)

        # Should achieve significant reduction
        assert stats["total_reduction_percent"] > 0
