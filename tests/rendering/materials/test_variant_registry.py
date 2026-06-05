"""Tests for MaterialVariantRegistry quality-driven variant compilation.

Task: T-MAT-5.1 Quality-Driven Variant Compilation
Gap: S3-G9 (HIGH)

Tests verify:
1. Each material has 3 quality variants (LOW, MEDIUM, HIGH)
2. Switching quality tier at runtime selects correct variant
3. variant_key uniquely identifies each variant
4. Missing material returns None
"""

import pytest
from typing import Dict, List

from trinity.materials.variant_registry import (
    CompiledVariant,
    MaterialVariantRegistry,
    select_material_variant,
    create_quality_optimized_registry,
)
from trinity.materials.variants import (
    MaterialDomain,
    BlendMode,
    QualityTier,
    VariantConfig,
)
from trinity.materials.compiler import MaterialCompiler
from trinity.materials.dsl import Material, MaterialMeta, SurfaceContext, SurfaceOutput, surface


# =============================================================================
# Test Materials
# =============================================================================

class TestGoldMaterial(Material, metaclass=MaterialMeta):
    """Simple gold metallic material for testing."""

    @surface
    def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
        out.base_color = (0.83, 0.69, 0.22)
        out.metallic = 0.9
        out.roughness = 0.3


class TestSilverMaterial(Material, metaclass=MaterialMeta):
    """Simple silver metallic material for testing."""

    @surface
    def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
        out.base_color = (0.95, 0.93, 0.88)
        out.metallic = 1.0
        out.roughness = 0.1


class TestPlasticMaterial(Material, metaclass=MaterialMeta):
    """Simple non-metallic material for testing."""

    @surface
    def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
        out.base_color = (1.0, 0.0, 0.0)
        out.metallic = 0.0
        out.roughness = 0.5


# =============================================================================
# CompiledVariant Tests
# =============================================================================

class TestCompiledVariant:
    """Tests for CompiledVariant dataclass."""

    def test_from_config_creates_variant(self):
        """Test creating CompiledVariant from VariantConfig."""
        config = VariantConfig(
            domain=MaterialDomain.SURFACE,
            blend=BlendMode.OPAQUE,
            quality=QualityTier.HIGH,
        )
        wgsl = "// Test shader"

        variant = CompiledVariant.from_config(wgsl, config)

        assert variant.wgsl == wgsl
        assert variant.quality == QualityTier.HIGH
        assert variant.domain == MaterialDomain.SURFACE
        assert variant.blend == BlendMode.OPAQUE
        assert variant.variant_key == "surface_opaque_high"
        assert variant.hash_key == config.get_variant_key()

    def test_variant_key_format(self):
        """Test variant_key follows expected format."""
        config = VariantConfig(
            domain=MaterialDomain.VOLUME,
            blend=BlendMode.TRANSLUCENT,
            quality=QualityTier.MEDIUM,
        )

        variant = CompiledVariant.from_config("// test", config)

        assert variant.variant_key == "volume_translucent_medium"

    def test_variant_is_immutable(self):
        """Test CompiledVariant is frozen (immutable)."""
        config = VariantConfig()
        variant = CompiledVariant.from_config("// test", config)

        with pytest.raises(AttributeError):
            variant.wgsl = "// modified"

    def test_unique_keys_for_different_configs(self):
        """Test different configs produce different keys."""
        configs = [
            VariantConfig(quality=QualityTier.LOW),
            VariantConfig(quality=QualityTier.MEDIUM),
            VariantConfig(quality=QualityTier.HIGH),
        ]

        keys = set()
        hash_keys = set()
        for config in configs:
            variant = CompiledVariant.from_config("// test", config)
            keys.add(variant.variant_key)
            hash_keys.add(variant.hash_key)

        assert len(keys) == 3, "Each quality tier should have unique variant_key"
        assert len(hash_keys) == 3, "Each quality tier should have unique hash_key"


# =============================================================================
# MaterialVariantRegistry Core Tests
# =============================================================================

class TestMaterialVariantRegistryCore:
    """Core functionality tests for MaterialVariantRegistry."""

    @pytest.fixture
    def registry(self) -> MaterialVariantRegistry:
        """Create empty registry."""
        return MaterialVariantRegistry()

    @pytest.fixture
    def compiler(self) -> MaterialCompiler:
        """Create material compiler."""
        return MaterialCompiler()

    def test_empty_registry(self, registry: MaterialVariantRegistry):
        """Test empty registry returns expected values."""
        assert registry.get_variant_count("nonexistent") == 0
        assert registry.has_material("nonexistent") is False
        assert registry.list_materials() == []
        assert registry.select_variant("nonexistent", QualityTier.HIGH) is None

    def test_register_material_creates_variants(
        self,
        registry: MaterialVariantRegistry,
        compiler: MaterialCompiler,
    ):
        """Test registering a material creates expected variant count."""
        count = registry.register_material("gold", compiler, TestGoldMaterial)

        # 5 domains x 5 blends x 3 qualities = 75 variants
        assert count == 75
        assert registry.get_variant_count("gold") == 75
        assert registry.has_material("gold") is True

    def test_register_quality_only_variants(
        self,
        registry: MaterialVariantRegistry,
        compiler: MaterialCompiler,
    ):
        """Test registering only quality variants for single domain/blend."""
        count = registry.register_material(
            "gold",
            compiler,
            TestGoldMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
            qualities=list(QualityTier),
        )

        # 1 domain x 1 blend x 3 qualities = 3 variants
        assert count == 3
        assert registry.get_variant_count("gold") == 3

    def test_material_has_3_quality_variants(
        self,
        registry: MaterialVariantRegistry,
        compiler: MaterialCompiler,
    ):
        """Test each material has exactly 3 quality variants for SURFACE OPAQUE."""
        registry.register_material(
            "gold",
            compiler,
            TestGoldMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )

        # Get variants for each quality tier
        low_variants = registry.get_quality_variants("gold", QualityTier.LOW)
        med_variants = registry.get_quality_variants("gold", QualityTier.MEDIUM)
        high_variants = registry.get_quality_variants("gold", QualityTier.HIGH)

        assert len(low_variants) == 1
        assert len(med_variants) == 1
        assert len(high_variants) == 1

        # Verify quality is correct
        assert low_variants[0].quality == QualityTier.LOW
        assert med_variants[0].quality == QualityTier.MEDIUM
        assert high_variants[0].quality == QualityTier.HIGH


# =============================================================================
# Variant Selection Tests
# =============================================================================

class TestVariantSelection:
    """Tests for variant selection at runtime."""

    @pytest.fixture
    def registry_with_gold(self) -> MaterialVariantRegistry:
        """Create registry with gold material registered (quality variants only)."""
        registry = MaterialVariantRegistry()
        compiler = MaterialCompiler()
        registry.register_material(
            "gold",
            compiler,
            TestGoldMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )
        return registry

    def test_select_high_quality_variant(self, registry_with_gold: MaterialVariantRegistry):
        """Test selecting HIGH quality variant."""
        variant = registry_with_gold.select_variant("gold", QualityTier.HIGH)

        assert variant is not None
        assert variant.quality == QualityTier.HIGH
        assert "QUALITY_HIGH: bool = true" in variant.wgsl

    def test_select_medium_quality_variant(self, registry_with_gold: MaterialVariantRegistry):
        """Test selecting MEDIUM quality variant."""
        variant = registry_with_gold.select_variant("gold", QualityTier.MEDIUM)

        assert variant is not None
        assert variant.quality == QualityTier.MEDIUM
        assert "QUALITY_MEDIUM: bool = true" in variant.wgsl

    def test_select_low_quality_variant(self, registry_with_gold: MaterialVariantRegistry):
        """Test selecting LOW quality variant."""
        variant = registry_with_gold.select_variant("gold", QualityTier.LOW)

        assert variant is not None
        assert variant.quality == QualityTier.LOW
        assert "QUALITY_LOW: bool = true" in variant.wgsl

    def test_switching_quality_tier_selects_correct_variant(
        self,
        registry_with_gold: MaterialVariantRegistry,
    ):
        """Test that switching quality tier at runtime selects correct variant."""
        # Simulate runtime quality switching
        qualities = [QualityTier.LOW, QualityTier.MEDIUM, QualityTier.HIGH]

        for expected_quality in qualities:
            variant = registry_with_gold.select_variant("gold", expected_quality)

            assert variant is not None
            assert variant.quality == expected_quality

            # Verify the correct const is set to true
            quality_const = f"QUALITY_{expected_quality.name}: bool = true"
            assert quality_const in variant.wgsl

            # Verify other quality consts are false
            for other_quality in qualities:
                if other_quality != expected_quality:
                    other_const = f"QUALITY_{other_quality.name}: bool = false"
                    assert other_const in variant.wgsl

    def test_missing_material_returns_none(self, registry_with_gold: MaterialVariantRegistry):
        """Test selecting from non-existent material returns None."""
        variant = registry_with_gold.select_variant("nonexistent", QualityTier.HIGH)
        assert variant is None

    def test_select_variant_exact(self, registry_with_gold: MaterialVariantRegistry):
        """Test exact variant selection."""
        variant = registry_with_gold.select_variant_exact(
            "gold",
            QualityTier.HIGH,
            MaterialDomain.SURFACE,
            BlendMode.OPAQUE,
        )

        assert variant is not None
        assert variant.quality == QualityTier.HIGH
        assert variant.domain == MaterialDomain.SURFACE
        assert variant.blend == BlendMode.OPAQUE

    def test_select_variant_exact_missing_returns_none(
        self,
        registry_with_gold: MaterialVariantRegistry,
    ):
        """Test exact selection with non-existent combination returns None."""
        # Registry only has SURFACE OPAQUE variants
        variant = registry_with_gold.select_variant_exact(
            "gold",
            QualityTier.HIGH,
            MaterialDomain.VOLUME,  # Not registered
            BlendMode.OPAQUE,
        )

        assert variant is None


# =============================================================================
# Variant Key Uniqueness Tests
# =============================================================================

class TestVariantKeyUniqueness:
    """Tests for variant_key uniqueness guarantees."""

    @pytest.fixture
    def full_registry(self) -> MaterialVariantRegistry:
        """Create registry with full 75 variants."""
        registry = MaterialVariantRegistry()
        compiler = MaterialCompiler()
        registry.register_material("gold", compiler, TestGoldMaterial)
        return registry

    def test_variant_keys_unique_across_all_variants(
        self,
        full_registry: MaterialVariantRegistry,
    ):
        """Test all 75 variant_keys are unique."""
        variants = full_registry.get_all_variants("gold")
        keys = [v.variant_key for v in variants]

        assert len(keys) == 75
        assert len(set(keys)) == 75, "All variant_keys must be unique"

    def test_hash_keys_unique_across_all_variants(
        self,
        full_registry: MaterialVariantRegistry,
    ):
        """Test all 75 hash_keys are unique."""
        variants = full_registry.get_all_variants("gold")
        hash_keys = [v.hash_key for v in variants]

        assert len(hash_keys) == 75
        assert len(set(hash_keys)) == 75, "All hash_keys must be unique"

    def test_variant_key_identifies_config(self, full_registry: MaterialVariantRegistry):
        """Test variant_key encodes domain, blend, and quality."""
        variants = full_registry.get_all_variants("gold")

        for variant in variants:
            # variant_key format: "{domain}_{blend}_{quality}"
            # Note: some domain values contain underscores (e.g., "deferred_decal")
            # So we split from the right to get the quality and blend first
            key = variant.variant_key

            # Extract quality (last part)
            quality_str = key.rsplit("_", 1)[-1]
            remaining = key.rsplit("_", 1)[0]

            # Extract blend (second to last part)
            blend_str = remaining.rsplit("_", 1)[-1]

            # Extract domain (everything before blend)
            domain_str = remaining.rsplit("_", 1)[0]

            # Verify encoding matches attributes
            assert variant.domain.value == domain_str
            assert variant.blend.value == blend_str
            assert variant.quality.value == quality_str


# =============================================================================
# Multiple Material Tests
# =============================================================================

class TestMultipleMaterials:
    """Tests for registry with multiple materials."""

    @pytest.fixture
    def multi_registry(self) -> MaterialVariantRegistry:
        """Create registry with multiple materials."""
        registry = MaterialVariantRegistry()
        compiler = MaterialCompiler()

        registry.register_material(
            "gold",
            compiler,
            TestGoldMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )
        registry.register_material(
            "silver",
            compiler,
            TestSilverMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )
        registry.register_material(
            "plastic",
            compiler,
            TestPlasticMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )

        return registry

    def test_list_materials(self, multi_registry: MaterialVariantRegistry):
        """Test listing all registered materials."""
        materials = multi_registry.list_materials()

        assert len(materials) == 3
        assert "gold" in materials
        assert "silver" in materials
        assert "plastic" in materials

    def test_independent_variant_selection(self, multi_registry: MaterialVariantRegistry):
        """Test selecting variants from different materials independently."""
        gold_high = multi_registry.select_variant("gold", QualityTier.HIGH)
        silver_low = multi_registry.select_variant("silver", QualityTier.LOW)
        plastic_med = multi_registry.select_variant("plastic", QualityTier.MEDIUM)

        assert gold_high is not None
        assert silver_low is not None
        assert plastic_med is not None

        # Verify correct quality tier
        assert gold_high.quality == QualityTier.HIGH
        assert silver_low.quality == QualityTier.LOW
        assert plastic_med.quality == QualityTier.MEDIUM

        # Verify different WGSL content (different material definitions)
        assert gold_high.wgsl != silver_low.wgsl
        assert silver_low.wgsl != plastic_med.wgsl

    def test_unregister_material(self, multi_registry: MaterialVariantRegistry):
        """Test unregistering a material removes it completely."""
        assert multi_registry.has_material("gold") is True

        result = multi_registry.unregister_material("gold")

        assert result is True
        assert multi_registry.has_material("gold") is False
        assert multi_registry.select_variant("gold", QualityTier.HIGH) is None
        assert len(multi_registry.list_materials()) == 2


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_select_material_variant_returns_wgsl(self):
        """Test select_material_variant returns WGSL string."""
        registry = MaterialVariantRegistry()
        compiler = MaterialCompiler()
        registry.register_material(
            "gold",
            compiler,
            TestGoldMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )

        wgsl = select_material_variant(registry, "gold", QualityTier.HIGH)

        assert wgsl is not None
        assert isinstance(wgsl, str)
        assert "QUALITY_HIGH: bool = true" in wgsl

    def test_select_material_variant_missing_returns_none(self):
        """Test select_material_variant returns None for missing material."""
        registry = MaterialVariantRegistry()

        wgsl = select_material_variant(registry, "nonexistent", QualityTier.HIGH)

        assert wgsl is None

    def test_create_quality_optimized_registry(self):
        """Test creating quality-optimized registry."""
        compiler = MaterialCompiler()
        materials = {
            "gold": TestGoldMaterial,
            "silver": TestSilverMaterial,
        }

        registry = create_quality_optimized_registry(compiler, materials)

        # 2 materials x 3 qualities = 6 variants total
        assert registry.get_variant_count("gold") == 3
        assert registry.get_variant_count("silver") == 3

        # All variants should be SURFACE OPAQUE
        for name in ["gold", "silver"]:
            for quality in QualityTier:
                variant = registry.select_variant(name, quality)
                assert variant is not None
                assert variant.domain == MaterialDomain.SURFACE
                assert variant.blend == BlendMode.OPAQUE


# =============================================================================
# Precompiled Variant Tests
# =============================================================================

class TestPrecompiledVariants:
    """Tests for registering pre-compiled variants."""

    def test_register_precompiled_variants(self):
        """Test registering pre-compiled variants."""
        registry = MaterialVariantRegistry()

        configs_and_wgsl = [
            (VariantConfig(quality=QualityTier.LOW), "// LOW shader"),
            (VariantConfig(quality=QualityTier.MEDIUM), "// MEDIUM shader"),
            (VariantConfig(quality=QualityTier.HIGH), "// HIGH shader"),
        ]

        count = registry.register_precompiled("cached_material", configs_and_wgsl)

        assert count == 3
        assert registry.get_variant_count("cached_material") == 3

        # Verify we can select them
        for config, expected_wgsl in configs_and_wgsl:
            variant = registry.select_variant("cached_material", config.quality)
            assert variant is not None
            assert variant.wgsl == expected_wgsl


# =============================================================================
# Iteration Tests
# =============================================================================

class TestIteration:
    """Tests for iterating over variants."""

    def test_iter_variants(self):
        """Test iterating over material variants."""
        registry = MaterialVariantRegistry()
        compiler = MaterialCompiler()
        registry.register_material(
            "gold",
            compiler,
            TestGoldMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )

        variants = list(registry.iter_variants("gold"))

        assert len(variants) == 3
        qualities = {v.quality for v in variants}
        assert qualities == {QualityTier.LOW, QualityTier.MEDIUM, QualityTier.HIGH}

    def test_iter_variants_nonexistent(self):
        """Test iterating over non-existent material yields nothing."""
        registry = MaterialVariantRegistry()

        variants = list(registry.iter_variants("nonexistent"))

        assert variants == []


# =============================================================================
# Material Class Reference Tests
# =============================================================================

class TestMaterialClassReference:
    """Tests for storing and retrieving material class references."""

    def test_get_material_class(self):
        """Test retrieving stored material class."""
        registry = MaterialVariantRegistry()
        compiler = MaterialCompiler()
        registry.register_material(
            "gold",
            compiler,
            TestGoldMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )

        material_class = registry.get_material_class("gold")

        assert material_class is TestGoldMaterial

    def test_get_material_class_nonexistent(self):
        """Test getting class for non-existent material returns None."""
        registry = MaterialVariantRegistry()

        material_class = registry.get_material_class("nonexistent")

        assert material_class is None


# =============================================================================
# Clear Registry Tests
# =============================================================================

class TestClearRegistry:
    """Tests for clearing the registry."""

    def test_clear_removes_all_materials(self):
        """Test clear removes all registered materials."""
        registry = MaterialVariantRegistry()
        compiler = MaterialCompiler()

        registry.register_material(
            "gold",
            compiler,
            TestGoldMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )
        registry.register_material(
            "silver",
            compiler,
            TestSilverMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )

        registry.clear()

        assert registry.list_materials() == []
        assert registry.has_material("gold") is False
        assert registry.has_material("silver") is False


# =============================================================================
# Performance Characteristics Tests
# =============================================================================

class TestPerformanceCharacteristics:
    """Tests verifying performance characteristics."""

    def test_variant_selection_is_dict_lookup(self):
        """Test that variant selection uses dict lookup (O(1))."""
        registry = MaterialVariantRegistry()
        compiler = MaterialCompiler()
        registry.register_material("gold", compiler, TestGoldMaterial)

        # Access internal structure to verify dict-based storage
        assert isinstance(registry._variants, dict)
        assert isinstance(registry._variants["gold"], dict)

        # Verify keys are strings (enabling O(1) dict lookup)
        for key in registry._variants["gold"].keys():
            assert isinstance(key, str)

    def test_wgsl_contains_quality_const_declarations(self):
        """Test compiled WGSL contains correct quality const declarations."""
        registry = MaterialVariantRegistry()
        compiler = MaterialCompiler()
        registry.register_material(
            "gold",
            compiler,
            TestGoldMaterial,
            domains=[MaterialDomain.SURFACE],
            blends=[BlendMode.OPAQUE],
        )

        for quality in QualityTier:
            variant = registry.select_variant("gold", quality)
            assert variant is not None

            # The active quality should be true
            active_const = f"const QUALITY_{quality.name}: bool = true;"
            assert active_const in variant.wgsl

            # Other qualities should be false
            for other in QualityTier:
                if other != quality:
                    inactive_const = f"const QUALITY_{other.name}: bool = false;"
                    assert inactive_const in variant.wgsl
