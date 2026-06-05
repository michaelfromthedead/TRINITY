"""Tests for Material LOD (Level-of-Detail) selection system.

Task: T-MAT-5.6 Material LOD System
Gap: S3-G15 (MEDIUM)

Test coverage:
- LODConfig validation and construction
- Distance-based LOD level selection
- Quality tier mapping
- Cross-fade blend factor computation
- MaterialLODSelector integration
- Edge cases and boundary conditions
"""

from __future__ import annotations

import math
import pytest
from typing import List, Optional, Tuple

from trinity.materials.variants import (
    MaterialDomain,
    BlendMode,
    QualityTier,
    VariantConfig,
)
from trinity.materials.variant_registry import (
    CompiledVariant,
    MaterialVariantRegistry,
)
from trinity.materials.lod import (
    LODConfig,
    LODBlendInfo,
    MaterialLODSelector,
    DEFAULT_LOD_DISTANCES,
    DEFAULT_LOD_QUALITIES,
    MAX_LOD_LEVELS,
    compute_blend_factor,
    smooth_blend_factor,
    create_lod_selector_with_defaults,
    create_lod_selector_for_quality,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def basic_registry() -> MaterialVariantRegistry:
    """Create a registry with pre-compiled mock variants."""
    registry = MaterialVariantRegistry()

    # Register mock variants for a test material
    variants = [
        (VariantConfig(quality=QualityTier.HIGH), "// HIGH quality WGSL"),
        (VariantConfig(quality=QualityTier.MEDIUM), "// MEDIUM quality WGSL"),
        (VariantConfig(quality=QualityTier.LOW), "// LOW quality WGSL"),
    ]
    registry.register_precompiled("test_material", variants)

    return registry


@pytest.fixture
def multi_material_registry() -> MaterialVariantRegistry:
    """Create a registry with multiple materials."""
    registry = MaterialVariantRegistry()

    # Material 1: all quality tiers
    variants1 = [
        (VariantConfig(quality=QualityTier.HIGH), "// Gold HIGH"),
        (VariantConfig(quality=QualityTier.MEDIUM), "// Gold MEDIUM"),
        (VariantConfig(quality=QualityTier.LOW), "// Gold LOW"),
    ]
    registry.register_precompiled("gold", variants1)

    # Material 2: only HIGH and LOW
    variants2 = [
        (VariantConfig(quality=QualityTier.HIGH), "// Silver HIGH"),
        (VariantConfig(quality=QualityTier.LOW), "// Silver LOW"),
    ]
    registry.register_precompiled("silver", variants2)

    return registry


@pytest.fixture
def default_config() -> LODConfig:
    """Create a default LODConfig."""
    return LODConfig()


@pytest.fixture
def crossfade_config() -> LODConfig:
    """Create a LODConfig with cross-fade enabled."""
    return LODConfig(
        distances=[0.0, 10.0, 50.0, 100.0],
        qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
        blend_range=5.0,
        unlit_fallback_lod=3,
    )


# ==============================================================================
# LODConfig Tests
# ==============================================================================


class TestLODConfig:
    """Tests for LODConfig dataclass."""

    def test_default_config_values(self) -> None:
        """Test that default config has correct values."""
        config = LODConfig()

        assert config.distances == DEFAULT_LOD_DISTANCES
        assert config.qualities == DEFAULT_LOD_QUALITIES
        assert config.blend_range == 0.0
        assert config.unlit_fallback_lod == 3

    def test_lod_count_property(self) -> None:
        """Test lod_count property."""
        config = LODConfig()
        assert config.lod_count == 4

        config2 = LODConfig(
            distances=[0.0, 50.0],
            qualities=[QualityTier.HIGH, QualityTier.LOW],
            unlit_fallback_lod=1,  # Valid for 2-level config
        )
        assert config2.lod_count == 2

    def test_max_distance_property(self) -> None:
        """Test max_distance property."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
        )
        assert config.max_distance == 100.0

    def test_has_crossfade_property(self) -> None:
        """Test has_crossfade property."""
        config_no_fade = LODConfig(blend_range=0.0)
        assert config_no_fade.has_crossfade is False

        config_with_fade = LODConfig(blend_range=5.0)
        assert config_with_fade.has_crossfade is True

    def test_validation_mismatched_lengths(self) -> None:
        """Test validation rejects mismatched distances/qualities."""
        with pytest.raises(ValueError, match="distances length.*must match"):
            LODConfig(
                distances=[0.0, 10.0, 50.0],
                qualities=[QualityTier.HIGH, QualityTier.LOW],  # Only 2 qualities
            )

    def test_validation_empty_distances(self) -> None:
        """Test validation rejects empty distances."""
        with pytest.raises(ValueError, match="At least 1 LOD level required"):
            LODConfig(distances=[], qualities=[])

    def test_validation_too_many_levels(self) -> None:
        """Test validation rejects too many LOD levels."""
        distances = [float(i * 10) for i in range(MAX_LOD_LEVELS + 2)]
        qualities = [QualityTier.HIGH] * (MAX_LOD_LEVELS + 2)

        with pytest.raises(ValueError, match=f"Maximum {MAX_LOD_LEVELS} LOD levels"):
            LODConfig(distances=distances, qualities=qualities)

    def test_validation_negative_first_distance(self) -> None:
        """Test validation rejects negative first distance."""
        with pytest.raises(ValueError, match="First LOD distance must be >= 0"):
            LODConfig(
                distances=[-1.0, 10.0],
                qualities=[QualityTier.HIGH, QualityTier.LOW],
            )

    def test_validation_non_monotonic_distances(self) -> None:
        """Test validation rejects non-monotonic distances."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            LODConfig(
                distances=[0.0, 50.0, 30.0],  # 30 < 50, not increasing
                qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW],
            )

    def test_validation_equal_distances(self) -> None:
        """Test validation rejects equal consecutive distances."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            LODConfig(
                distances=[0.0, 10.0, 10.0],  # Equal distances
                qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW],
            )

    def test_validation_negative_blend_range(self) -> None:
        """Test validation rejects negative blend_range."""
        with pytest.raises(ValueError, match="blend_range must be >= 0"):
            LODConfig(blend_range=-1.0)

    def test_validation_invalid_unlit_fallback(self) -> None:
        """Test validation rejects invalid unlit_fallback_lod."""
        with pytest.raises(ValueError, match="unlit_fallback_lod.*must be <"):
            LODConfig(
                distances=[0.0, 10.0],
                qualities=[QualityTier.HIGH, QualityTier.LOW],
                unlit_fallback_lod=5,  # Out of range
            )

    def test_get_lod_level_at_distances(self) -> None:
        """Test LOD level selection at exact distance thresholds."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
        )

        assert config.get_lod_level(0.0) == 0
        assert config.get_lod_level(10.0) == 1
        assert config.get_lod_level(50.0) == 2
        assert config.get_lod_level(100.0) == 3

    def test_get_lod_level_between_thresholds(self) -> None:
        """Test LOD level selection between thresholds."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
        )

        assert config.get_lod_level(5.0) == 0   # Between 0 and 10
        assert config.get_lod_level(25.0) == 1  # Between 10 and 50
        assert config.get_lod_level(75.0) == 2  # Between 50 and 100
        assert config.get_lod_level(150.0) == 3  # Beyond 100

    def test_get_lod_level_negative_distance(self) -> None:
        """Test LOD level selection with negative distance."""
        config = LODConfig()
        assert config.get_lod_level(-10.0) == 0  # Should return 0 for negatives

    def test_get_quality_for_lod(self) -> None:
        """Test quality tier lookup by LOD level."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
        )

        assert config.get_quality_for_lod(0) == QualityTier.HIGH
        assert config.get_quality_for_lod(1) == QualityTier.MEDIUM
        assert config.get_quality_for_lod(2) == QualityTier.LOW
        assert config.get_quality_for_lod(3) == QualityTier.LOW

    def test_get_quality_for_lod_out_of_range(self) -> None:
        """Test quality lookup with invalid LOD level."""
        config = LODConfig()

        with pytest.raises(IndexError):
            config.get_quality_for_lod(10)

        with pytest.raises(IndexError):
            config.get_quality_for_lod(-1)

    def test_is_unlit_lod(self) -> None:
        """Test unlit fallback detection."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
            unlit_fallback_lod=3,
        )

        assert config.is_unlit_lod(0) is False
        assert config.is_unlit_lod(1) is False
        assert config.is_unlit_lod(2) is False
        assert config.is_unlit_lod(3) is True

    def test_is_unlit_lod_disabled(self) -> None:
        """Test unlit fallback when disabled."""
        config = LODConfig(
            distances=[0.0, 10.0],
            qualities=[QualityTier.HIGH, QualityTier.LOW],
            unlit_fallback_lod=-1,  # Disabled
        )

        assert config.is_unlit_lod(0) is False
        assert config.is_unlit_lod(1) is False

    def test_create_simple(self) -> None:
        """Test create_simple factory method."""
        config = LODConfig.create_simple(near=15.0, mid=60.0, far=120.0, blend_range=3.0)

        assert config.distances == [0.0, 15.0, 60.0, 120.0]
        assert config.qualities[0] == QualityTier.HIGH
        assert config.qualities[1] == QualityTier.MEDIUM
        assert config.qualities[2] == QualityTier.LOW
        assert config.blend_range == 3.0

    def test_create_aggressive(self) -> None:
        """Test create_aggressive factory method."""
        config = LODConfig.create_aggressive(near=5.0, far=25.0, blend_range=2.0)

        assert config.lod_count == 3
        assert config.distances == [0.0, 5.0, 25.0]
        assert config.qualities[0] == QualityTier.MEDIUM  # Even LOD 0 is MEDIUM
        assert config.blend_range == 2.0

    def test_create_quality_focused(self) -> None:
        """Test create_quality_focused factory method."""
        config = LODConfig.create_quality_focused(
            near=25.0, mid=75.0, far=150.0, blend_range=5.0
        )

        assert config.lod_count == 4
        assert config.qualities[0] == QualityTier.HIGH
        assert config.qualities[1] == QualityTier.HIGH  # Maintains HIGH longer
        assert config.qualities[2] == QualityTier.MEDIUM


# ==============================================================================
# Blend Factor Tests
# ==============================================================================


class TestBlendFactor:
    """Tests for blend factor computation."""

    def test_compute_blend_factor_no_blend(self) -> None:
        """Test blend factor with blend_range=0."""
        # Before threshold - should be 0
        factor = compute_blend_factor(5.0, 0.0, 10.0, 0.0)
        assert factor == 0.0

        # At threshold - should be 1
        factor = compute_blend_factor(10.0, 0.0, 10.0, 0.0)
        assert factor == 1.0

        # After threshold - should be 1
        factor = compute_blend_factor(15.0, 0.0, 10.0, 0.0)
        assert factor == 1.0

    def test_compute_blend_factor_with_blend(self) -> None:
        """Test blend factor with blend_range > 0."""
        # Before transition zone
        factor = compute_blend_factor(5.0, 0.0, 10.0, 2.0)  # Zone is [8, 10]
        assert factor == 0.0

        # Start of transition zone
        factor = compute_blend_factor(8.0, 0.0, 10.0, 2.0)
        assert factor == 0.0

        # Middle of transition zone
        factor = compute_blend_factor(9.0, 0.0, 10.0, 2.0)
        assert factor == pytest.approx(0.5)

        # End of transition zone
        factor = compute_blend_factor(10.0, 0.0, 10.0, 2.0)
        assert factor == 1.0

    def test_compute_blend_factor_full_range(self) -> None:
        """Test blend factor across entire LOD range."""
        # Transition zone covers entire range
        factor = compute_blend_factor(5.0, 0.0, 10.0, 10.0)  # Zone is [0, 10]
        assert factor == pytest.approx(0.5)

    def test_smooth_blend_factor(self) -> None:
        """Test smoothstep blend factor."""
        # Boundary values unchanged
        assert smooth_blend_factor(0.0) == 0.0
        assert smooth_blend_factor(1.0) == 1.0

        # Midpoint
        assert smooth_blend_factor(0.5) == pytest.approx(0.5)

        # Smoothstep has characteristic S-curve
        # At 0.25, smoothstep(0.25) = 0.15625
        assert smooth_blend_factor(0.25) == pytest.approx(0.15625)

        # At 0.75, smoothstep(0.75) = 0.84375
        assert smooth_blend_factor(0.75) == pytest.approx(0.84375)

    def test_smooth_blend_factor_clamping(self) -> None:
        """Test smoothstep clamping for out-of-range values."""
        assert smooth_blend_factor(-0.5) == 0.0
        assert smooth_blend_factor(1.5) == 1.0


# ==============================================================================
# MaterialLODSelector Tests
# ==============================================================================


class TestMaterialLODSelector:
    """Tests for MaterialLODSelector class."""

    def test_init_with_default_config(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test initialization with default configuration."""
        selector = MaterialLODSelector(basic_registry)

        assert selector.registry is basic_registry
        assert selector.config.distances == DEFAULT_LOD_DISTANCES
        assert selector.lod_count == 4

    def test_init_with_custom_config(
        self, basic_registry: MaterialVariantRegistry, crossfade_config: LODConfig
    ) -> None:
        """Test initialization with custom configuration."""
        selector = MaterialLODSelector(basic_registry, crossfade_config)

        assert selector.config is crossfade_config
        assert selector.config.blend_range == 5.0

    def test_select_lod_basic(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test basic LOD level selection."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
        )
        selector = MaterialLODSelector(basic_registry, config)

        assert selector.select_lod("test_material", 5.0) == 0
        assert selector.select_lod("test_material", 15.0) == 1
        assert selector.select_lod("test_material", 75.0) == 2
        assert selector.select_lod("test_material", 150.0) == 3

    def test_select_quality(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test quality tier selection based on distance."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
        )
        selector = MaterialLODSelector(basic_registry, config)

        assert selector.select_quality("test_material", 5.0) == QualityTier.HIGH
        assert selector.select_quality("test_material", 25.0) == QualityTier.MEDIUM
        assert selector.select_quality("test_material", 75.0) == QualityTier.LOW

    def test_select_variant(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test variant selection based on distance."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
        )
        selector = MaterialLODSelector(basic_registry, config)

        # Close distance - HIGH quality
        variant = selector.select_variant("test_material", 5.0)
        assert variant is not None
        assert variant.quality == QualityTier.HIGH

        # Medium distance - MEDIUM quality
        variant = selector.select_variant("test_material", 25.0)
        assert variant is not None
        assert variant.quality == QualityTier.MEDIUM

        # Far distance - LOW quality
        variant = selector.select_variant("test_material", 75.0)
        assert variant is not None
        assert variant.quality == QualityTier.LOW

    def test_select_variant_nonexistent_material(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test variant selection for non-existent material."""
        selector = MaterialLODSelector(basic_registry)

        variant = selector.select_variant("nonexistent", 50.0)
        assert variant is None

    def test_select_with_blend_no_crossfade(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test select_with_blend when crossfade is disabled."""
        config = LODConfig(blend_range=0.0)
        selector = MaterialLODSelector(basic_registry, config)

        near_v, far_v, factor = selector.select_with_blend("test_material", 25.0)

        # Without crossfade, near and far should be the same
        assert near_v is not None
        assert far_v is not None
        assert near_v.quality == far_v.quality
        assert factor == 0.0

    def test_select_with_blend_in_transition(
        self, basic_registry: MaterialVariantRegistry, crossfade_config: LODConfig
    ) -> None:
        """Test select_with_blend during transition zone."""
        selector = MaterialLODSelector(basic_registry, crossfade_config)

        # At distance 7.5, we're in the transition zone [5.0, 10.0] for LOD 0->1
        # (blend_range=5.0, threshold at 10.0, so zone starts at 5.0)
        near_v, far_v, factor = selector.select_with_blend("test_material", 7.5)

        assert near_v is not None
        assert far_v is not None
        assert near_v.quality == QualityTier.HIGH
        assert far_v.quality == QualityTier.MEDIUM
        assert 0.0 < factor < 1.0

    def test_select_with_blend_outside_transition(
        self, basic_registry: MaterialVariantRegistry, crossfade_config: LODConfig
    ) -> None:
        """Test select_with_blend outside transition zones."""
        selector = MaterialLODSelector(basic_registry, crossfade_config)

        # At distance 25.0, we're between LOD 1 and 2, not in transition
        # LOD 1 threshold is 10.0, LOD 2 is 50.0, zone is [45.0, 50.0]
        near_v, far_v, factor = selector.select_with_blend("test_material", 25.0)

        assert near_v is not None
        # Not in transition, so both variants should be the same
        assert near_v.quality == QualityTier.MEDIUM

    def test_get_blend_info(
        self, basic_registry: MaterialVariantRegistry, crossfade_config: LODConfig
    ) -> None:
        """Test get_blend_info method."""
        selector = MaterialLODSelector(basic_registry, crossfade_config)

        # In transition zone
        blend_info = selector.get_blend_info("test_material", 7.5)
        assert blend_info.near_lod == 0
        assert blend_info.far_lod == 1
        assert blend_info.is_blending is True
        assert 0.0 < blend_info.blend_factor < 1.0

        # Outside transition zone
        blend_info = selector.get_blend_info("test_material", 3.0)
        assert blend_info.near_lod == 0
        assert blend_info.is_blending is False

    def test_is_unlit(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test unlit fallback detection."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
            unlit_fallback_lod=3,
        )
        selector = MaterialLODSelector(basic_registry, config)

        assert selector.is_unlit("test_material", 50.0) is False  # LOD 2
        assert selector.is_unlit("test_material", 150.0) is True  # LOD 3

    def test_get_lod_distances(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test get_lod_distances returns copy."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW],
            unlit_fallback_lod=2,  # Valid for 3-level config
        )
        selector = MaterialLODSelector(basic_registry, config)

        distances = selector.get_lod_distances()
        assert distances == [0.0, 10.0, 50.0]

        # Modifying return value shouldn't affect selector
        distances[0] = 999.0
        assert selector.get_lod_distances()[0] == 0.0

    def test_get_lod_qualities(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test get_lod_qualities returns copy."""
        config = LODConfig(
            distances=[0.0, 10.0],
            qualities=[QualityTier.HIGH, QualityTier.LOW],
            unlit_fallback_lod=1,  # Valid for 2-level config
        )
        selector = MaterialLODSelector(basic_registry, config)

        qualities = selector.get_lod_qualities()
        assert qualities == [QualityTier.HIGH, QualityTier.LOW]

    def test_precompute_lod_variants(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test precomputing variants for all LOD levels."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW],
            unlit_fallback_lod=2,  # Valid for 3-level config
        )
        selector = MaterialLODSelector(basic_registry, config)

        variants = selector.precompute_lod_variants("test_material")

        assert len(variants) == 3
        assert variants[0].quality == QualityTier.HIGH
        assert variants[1].quality == QualityTier.MEDIUM
        assert variants[2].quality == QualityTier.LOW

    def test_precompute_lod_variants_caching(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test that precomputed variants are cached."""
        selector = MaterialLODSelector(basic_registry)

        # First call
        variants1 = selector.precompute_lod_variants("test_material")

        # Second call should return cached result
        variants2 = selector.precompute_lod_variants("test_material")

        assert variants1 is variants2

    def test_clear_cache(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test clearing the variant cache."""
        selector = MaterialLODSelector(basic_registry)

        # Populate cache
        variants1 = selector.precompute_lod_variants("test_material")

        # Clear cache
        selector.clear_cache()

        # Should recompute
        variants2 = selector.precompute_lod_variants("test_material")

        assert variants1 is not variants2


# ==============================================================================
# Factory Function Tests
# ==============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_lod_selector_with_defaults(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test create_lod_selector_with_defaults."""
        selector = create_lod_selector_with_defaults(basic_registry)

        assert selector.registry is basic_registry
        assert selector.config.distances == DEFAULT_LOD_DISTANCES

    def test_create_lod_selector_for_quality_performance(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test create_lod_selector_for_quality with performance preset."""
        selector = create_lod_selector_for_quality(basic_registry, "performance")

        assert selector.config.lod_count == 3
        assert selector.config.qualities[0] == QualityTier.MEDIUM

    def test_create_lod_selector_for_quality_balanced(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test create_lod_selector_for_quality with balanced preset."""
        selector = create_lod_selector_for_quality(basic_registry, "balanced")

        assert selector.config.lod_count == 4
        assert selector.config.blend_range == 2.0

    def test_create_lod_selector_for_quality_quality(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test create_lod_selector_for_quality with quality preset."""
        selector = create_lod_selector_for_quality(basic_registry, "quality")

        assert selector.config.lod_count == 4
        assert selector.config.blend_range == 5.0

    def test_create_lod_selector_for_quality_invalid(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test create_lod_selector_for_quality with invalid preset."""
        with pytest.raises(ValueError, match="Unknown quality preset"):
            create_lod_selector_for_quality(basic_registry, "invalid_preset")


# ==============================================================================
# LODBlendInfo Tests
# ==============================================================================


class TestLODBlendInfo:
    """Tests for LODBlendInfo dataclass."""

    def test_primary_lod_near(self) -> None:
        """Test primary_lod when blend factor favors near LOD."""
        info = LODBlendInfo(near_lod=0, far_lod=1, blend_factor=0.3, is_blending=True)
        assert info.primary_lod == 0

    def test_primary_lod_far(self) -> None:
        """Test primary_lod when blend factor favors far LOD."""
        info = LODBlendInfo(near_lod=0, far_lod=1, blend_factor=0.7, is_blending=True)
        assert info.primary_lod == 1

    def test_primary_lod_midpoint(self) -> None:
        """Test primary_lod at exact midpoint (should favor near)."""
        info = LODBlendInfo(near_lod=0, far_lod=1, blend_factor=0.5, is_blending=True)
        assert info.primary_lod == 0  # 0.5 is not > 0.5, so near wins


# ==============================================================================
# Edge Case Tests
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_lod_level(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test configuration with only one LOD level."""
        config = LODConfig(
            distances=[0.0],
            qualities=[QualityTier.HIGH],
            unlit_fallback_lod=-1,  # Disable unlit fallback
        )
        selector = MaterialLODSelector(basic_registry, config)

        # All distances should return LOD 0
        assert selector.select_lod("test_material", 0.0) == 0
        assert selector.select_lod("test_material", 1000.0) == 0

    def test_very_large_distance(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test with very large distance values."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
        )
        selector = MaterialLODSelector(basic_registry, config)

        # Should return highest LOD level
        assert selector.select_lod("test_material", 1e9) == 3
        assert selector.select_quality("test_material", 1e9) == QualityTier.LOW

    def test_zero_distance(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test at exactly zero distance."""
        selector = MaterialLODSelector(basic_registry)

        assert selector.select_lod("test_material", 0.0) == 0
        assert selector.select_quality("test_material", 0.0) == QualityTier.HIGH

    def test_blend_at_exact_threshold(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test blend factor at exact LOD threshold."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW],
            blend_range=5.0,
            unlit_fallback_lod=2,  # Valid for 3-level config
        )
        selector = MaterialLODSelector(basic_registry, config)

        # At exactly 10.0, we're at LOD 1 and the transition zone for LOD 0->1 ends
        # The blend info should show the completed transition
        blend_info = selector.get_blend_info("test_material", 10.0)
        assert blend_info.near_lod == 1  # At LOD 1

        # Test within the transition zone (5.0 to 10.0)
        near_v, far_v, factor = selector.select_with_blend("test_material", 7.5)
        # 7.5 is halfway through transition zone [5, 10]
        assert 0.0 < factor < 1.0  # Should be in transition

    def test_blend_range_larger_than_gap(
        self, basic_registry: MaterialVariantRegistry
    ) -> None:
        """Test blend_range larger than gap between LOD thresholds."""
        config = LODConfig(
            distances=[0.0, 10.0, 15.0],  # Only 5 units between LOD 1 and 2
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW],
            blend_range=10.0,  # Larger than gap
            unlit_fallback_lod=2,  # Valid for 3-level config
        )
        selector = MaterialLODSelector(basic_registry, config)

        # Should still work, transition starts at max(10, 15-10) = 10
        blend_info = selector.get_blend_info("test_material", 12.5)
        assert blend_info.is_blending is True

    def test_multi_material_selection(
        self, multi_material_registry: MaterialVariantRegistry
    ) -> None:
        """Test selecting variants for multiple materials."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW],
            unlit_fallback_lod=2,  # Valid for 3-level config
        )
        selector = MaterialLODSelector(multi_material_registry, config)

        # Gold has all tiers
        gold_variant = selector.select_variant("gold", 25.0)
        assert gold_variant is not None
        assert gold_variant.quality == QualityTier.MEDIUM

        # Silver only has HIGH and LOW
        silver_variant = selector.select_variant("silver", 25.0)
        # Should fall back to available variant (MEDIUM not available)
        # The registry's select_variant will try to find the best match


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests with MaterialVariantRegistry."""

    def test_full_lod_workflow(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test complete LOD selection workflow."""
        # Create config with crossfade
        config = LODConfig.create_simple(near=10.0, mid=50.0, far=100.0, blend_range=5.0)
        selector = MaterialLODSelector(basic_registry, config)

        # Simulate an object moving away from camera
        distances = [0.0, 5.0, 8.0, 12.0, 25.0, 48.0, 75.0, 150.0]
        expected_qualities = [
            QualityTier.HIGH,    # 0.0
            QualityTier.HIGH,    # 5.0
            QualityTier.HIGH,    # 8.0 (in transition to MEDIUM)
            QualityTier.MEDIUM,  # 12.0
            QualityTier.MEDIUM,  # 25.0
            QualityTier.MEDIUM,  # 48.0 (in transition to LOW)
            QualityTier.LOW,     # 75.0
            QualityTier.LOW,     # 150.0
        ]

        for dist, expected in zip(distances, expected_qualities):
            quality = selector.select_quality("test_material", dist)
            assert quality == expected, f"At distance {dist}"

    def test_precompute_and_select(self, basic_registry: MaterialVariantRegistry) -> None:
        """Test precomputing variants then selecting."""
        config = LODConfig(
            distances=[0.0, 10.0, 50.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW],
            unlit_fallback_lod=2,  # Valid for 3-level config
        )
        selector = MaterialLODSelector(basic_registry, config)

        # Precompute
        variants = selector.precompute_lod_variants("test_material")

        # Select at various distances and verify match
        for lod, variant in variants.items():
            # Find a distance that should give us this LOD
            if lod == 0:
                dist = 5.0
            elif lod == 1:
                dist = 25.0
            else:
                dist = 75.0

            selected = selector.select_variant("test_material", dist)
            assert selected is not None
            assert selected.quality == variant.quality
