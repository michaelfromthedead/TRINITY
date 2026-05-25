"""
Tests for PCG placement rules and filters.

Tests cover:
- Slope filter accuracy
- Height filter accuracy
- Layer filter
- Compound filters
- Biome rules
- Transform rules
"""

import pytest

from engine.world.pcg.rules import (
    TerrainData,
    SlopeFilter,
    HeightFilter,
    LayerFilter,
    NoiseFilter,
    ExclusionZone,
    PlacementFilter,
    SlopeFilterImpl,
    HeightFilterImpl,
    LayerFilterImpl,
    NoiseFilterImpl,
    ExclusionZoneFilter,
    CompoundFilter,
    BiomeRule,
    PlacementRuleSet,
    TransformRule,
    Transform,
    PlacementValidator,
    create_slope_filter,
    create_height_filter,
    create_layer_filter,
    create_noise_filter,
    create_exclusion_filter,
)
from engine.world.pcg.noise import NoiseSettings


class TestTerrainData:
    """Tests for TerrainData dataclass."""

    def test_default_values(self):
        """Test default terrain data values."""
        data = TerrainData()
        assert data.height == 0.0
        assert data.slope == 0.0
        assert data.normal == (0.0, 1.0, 0.0)
        assert data.layer_id == 0
        assert data.biome_id == "default"
        assert data.moisture == 0.5
        assert data.temperature == 0.5

    def test_custom_values(self):
        """Test custom terrain data values."""
        data = TerrainData(
            height=100.0,
            slope=30.0,
            normal=(0.5, 0.866, 0.0),
            layer_id=2,
            biome_id="forest",
            moisture=0.8,
            temperature=0.6,
        )
        assert data.height == 100.0
        assert data.slope == 30.0
        assert data.biome_id == "forest"


class TestSlopeFilter:
    """Tests for SlopeFilter dataclass and implementation."""

    def test_default_values(self):
        """Test default slope filter values."""
        f = SlopeFilter()
        assert f.min_slope == 0.0
        assert f.max_slope == 90.0

    def test_custom_values(self):
        """Test custom slope filter values."""
        f = SlopeFilter(min_slope=10.0, max_slope=45.0)
        assert f.min_slope == 10.0
        assert f.max_slope == 45.0

    def test_invalid_min_slope(self):
        """Test validation of min_slope."""
        with pytest.raises(ValueError, match="min_slope must be >= 0"):
            SlopeFilter(min_slope=-5.0)

    def test_invalid_max_slope(self):
        """Test validation of max_slope."""
        with pytest.raises(ValueError, match="max_slope must be <= 90"):
            SlopeFilter(max_slope=95.0)

    def test_invalid_range(self):
        """Test validation of slope range."""
        with pytest.raises(ValueError, match="min_slope.*must be <= max_slope"):
            SlopeFilter(min_slope=60.0, max_slope=30.0)

    def test_impl_evaluate_pass(self):
        """Test slope filter implementation passes valid slopes."""
        impl = SlopeFilterImpl(SlopeFilter(min_slope=0.0, max_slope=45.0))
        terrain = TerrainData(slope=30.0)

        assert impl.evaluate(0, 0, terrain) is True

    def test_impl_evaluate_fail_too_steep(self):
        """Test slope filter implementation fails steep slopes."""
        impl = SlopeFilterImpl(SlopeFilter(min_slope=0.0, max_slope=45.0))
        terrain = TerrainData(slope=60.0)

        assert impl.evaluate(0, 0, terrain) is False

    def test_impl_evaluate_fail_too_flat(self):
        """Test slope filter implementation fails flat terrain."""
        impl = SlopeFilterImpl(SlopeFilter(min_slope=30.0, max_slope=60.0))
        terrain = TerrainData(slope=15.0)

        assert impl.evaluate(0, 0, terrain) is False

    def test_impl_boundary_values(self):
        """Test slope filter at boundary values."""
        impl = SlopeFilterImpl(SlopeFilter(min_slope=30.0, max_slope=60.0))

        # Exactly at min
        assert impl.evaluate(0, 0, TerrainData(slope=30.0)) is True

        # Exactly at max
        assert impl.evaluate(0, 0, TerrainData(slope=60.0)) is True

        # Just below min
        assert impl.evaluate(0, 0, TerrainData(slope=29.9)) is False

        # Just above max
        assert impl.evaluate(0, 0, TerrainData(slope=60.1)) is False


class TestHeightFilter:
    """Tests for HeightFilter dataclass and implementation."""

    def test_default_values(self):
        """Test default height filter values."""
        f = HeightFilter()
        assert f.min_height == -1000.0
        assert f.max_height == 1000.0

    def test_custom_values(self):
        """Test custom height filter values."""
        f = HeightFilter(min_height=50.0, max_height=200.0)
        assert f.min_height == 50.0
        assert f.max_height == 200.0

    def test_invalid_range(self):
        """Test validation of height range."""
        with pytest.raises(ValueError, match="min_height.*must be <= max_height"):
            HeightFilter(min_height=200.0, max_height=100.0)

    def test_impl_evaluate_pass(self):
        """Test height filter implementation passes valid heights."""
        impl = HeightFilterImpl(HeightFilter(min_height=50.0, max_height=200.0))
        terrain = TerrainData(height=100.0)

        assert impl.evaluate(0, 0, terrain) is True

    def test_impl_evaluate_fail_too_low(self):
        """Test height filter implementation fails low terrain."""
        impl = HeightFilterImpl(HeightFilter(min_height=50.0, max_height=200.0))
        terrain = TerrainData(height=25.0)

        assert impl.evaluate(0, 0, terrain) is False

    def test_impl_evaluate_fail_too_high(self):
        """Test height filter implementation fails high terrain."""
        impl = HeightFilterImpl(HeightFilter(min_height=50.0, max_height=200.0))
        terrain = TerrainData(height=250.0)

        assert impl.evaluate(0, 0, terrain) is False


class TestLayerFilter:
    """Tests for LayerFilter dataclass and implementation."""

    def test_empty_layers(self):
        """Test empty layer filter allows all."""
        f = LayerFilter(allowed_layers=[])
        assert f.is_allowed(0) is True
        assert f.is_allowed(5) is True

    def test_specific_layers(self):
        """Test specific layer filter."""
        f = LayerFilter(allowed_layers=[1, 3, 5])
        assert f.is_allowed(1) is True
        assert f.is_allowed(3) is True
        assert f.is_allowed(5) is True
        assert f.is_allowed(2) is False
        assert f.is_allowed(4) is False

    def test_impl_evaluate(self):
        """Test layer filter implementation."""
        impl = LayerFilterImpl(LayerFilter(allowed_layers=[1, 2]))

        assert impl.evaluate(0, 0, TerrainData(layer_id=1)) is True
        assert impl.evaluate(0, 0, TerrainData(layer_id=2)) is True
        assert impl.evaluate(0, 0, TerrainData(layer_id=3)) is False


class TestNoiseFilter:
    """Tests for NoiseFilter dataclass and implementation."""

    def test_default_values(self):
        """Test default noise filter values."""
        f = NoiseFilter()
        assert f.threshold == 0.5
        assert f.invert is False

    def test_evaluate_determinism(self):
        """Test noise filter evaluation is deterministic."""
        f = NoiseFilter(noise_settings=NoiseSettings(seed=42), threshold=0.5)

        result1 = f.evaluate(10.0, 20.0)
        result2 = f.evaluate(10.0, 20.0)

        assert result1 == result2

    def test_evaluate_threshold(self):
        """Test noise filter threshold behavior."""
        f_low = NoiseFilter(
            noise_settings=NoiseSettings(seed=42),
            threshold=0.0,  # Should pass most
        )
        f_high = NoiseFilter(
            noise_settings=NoiseSettings(seed=42),
            threshold=0.99,  # Should fail most
        )

        # Count passes for many samples
        low_passes = sum(1 for x in range(10) for y in range(10) if f_low.evaluate(x, y))
        high_passes = sum(1 for x in range(10) for y in range(10) if f_high.evaluate(x, y))

        assert low_passes > high_passes

    def test_evaluate_invert(self):
        """Test noise filter inversion."""
        f = NoiseFilter(noise_settings=NoiseSettings(seed=42), threshold=0.5)
        f_inv = NoiseFilter(
            noise_settings=NoiseSettings(seed=42),
            threshold=0.5,
            invert=True,
        )

        # Should be opposite results
        for x in range(5):
            for y in range(5):
                assert f.evaluate(x, y) != f_inv.evaluate(x, y)


class TestExclusionZone:
    """Tests for ExclusionZone dataclass."""

    def test_default_values(self):
        """Test default exclusion zone values."""
        z = ExclusionZone()
        assert z.center == (0.0, 0.0)
        assert z.radius == 10.0

    def test_invalid_radius(self):
        """Test validation of radius."""
        with pytest.raises(ValueError, match="radius must be > 0"):
            ExclusionZone(radius=0)

    def test_contains_inside(self):
        """Test point inside exclusion zone."""
        z = ExclusionZone(center=(50.0, 50.0), radius=10.0)

        assert z.contains(50.0, 50.0) is True  # Center
        assert z.contains(55.0, 50.0) is True  # Inside
        assert z.contains(50.0, 55.0) is True  # Inside

    def test_contains_outside(self):
        """Test point outside exclusion zone."""
        z = ExclusionZone(center=(50.0, 50.0), radius=10.0)

        assert z.contains(100.0, 100.0) is False
        assert z.contains(61.0, 50.0) is False  # Just outside

    def test_contains_boundary(self):
        """Test point on boundary."""
        z = ExclusionZone(center=(50.0, 50.0), radius=10.0)

        assert z.contains(60.0, 50.0) is True  # On boundary


class TestExclusionZoneFilter:
    """Tests for ExclusionZoneFilter class."""

    def test_empty_zones(self):
        """Test filter with no zones."""
        flt = ExclusionZoneFilter([])
        assert flt.evaluate(50, 50, TerrainData()) is True

    def test_single_zone(self):
        """Test filter with single zone."""
        zone = ExclusionZone(center=(50.0, 50.0), radius=10.0)
        flt = ExclusionZoneFilter([zone])

        assert flt.evaluate(50, 50, TerrainData()) is False  # In zone
        assert flt.evaluate(100, 100, TerrainData()) is True  # Outside

    def test_multiple_zones(self):
        """Test filter with multiple zones."""
        zones = [
            ExclusionZone(center=(25.0, 25.0), radius=10.0),
            ExclusionZone(center=(75.0, 75.0), radius=10.0),
        ]
        flt = ExclusionZoneFilter(zones)

        assert flt.evaluate(25, 25, TerrainData()) is False
        assert flt.evaluate(75, 75, TerrainData()) is False
        assert flt.evaluate(50, 50, TerrainData()) is True

    def test_add_zone(self):
        """Test adding zones dynamically."""
        flt = ExclusionZoneFilter([])
        assert flt.evaluate(50, 50, TerrainData()) is True

        flt.add_zone(ExclusionZone(center=(50.0, 50.0), radius=10.0))
        assert flt.evaluate(50, 50, TerrainData()) is False


class TestCompoundFilter:
    """Tests for CompoundFilter class."""

    def test_all_mode_all_pass(self):
        """Test 'all' mode when all filters pass."""
        filters = [
            SlopeFilterImpl(SlopeFilter(min_slope=0, max_slope=90)),
            HeightFilterImpl(HeightFilter(min_height=-1000, max_height=1000)),
        ]
        compound = CompoundFilter(filters, mode="all")

        terrain = TerrainData(slope=30, height=100)
        assert compound.evaluate(0, 0, terrain) is True

    def test_all_mode_one_fails(self):
        """Test 'all' mode when one filter fails."""
        filters = [
            SlopeFilterImpl(SlopeFilter(min_slope=0, max_slope=30)),
            HeightFilterImpl(HeightFilter(min_height=-1000, max_height=1000)),
        ]
        compound = CompoundFilter(filters, mode="all")

        terrain = TerrainData(slope=45, height=100)  # Slope too steep
        assert compound.evaluate(0, 0, terrain) is False

    def test_any_mode_one_passes(self):
        """Test 'any' mode when one filter passes."""
        filters = [
            SlopeFilterImpl(SlopeFilter(min_slope=0, max_slope=30)),
            HeightFilterImpl(HeightFilter(min_height=0, max_height=50)),
        ]
        compound = CompoundFilter(filters, mode="any")

        terrain = TerrainData(slope=15, height=100)  # Slope passes
        assert compound.evaluate(0, 0, terrain) is True

    def test_any_mode_all_fail(self):
        """Test 'any' mode when all filters fail."""
        filters = [
            SlopeFilterImpl(SlopeFilter(min_slope=60, max_slope=90)),
            HeightFilterImpl(HeightFilter(min_height=500, max_height=1000)),
        ]
        compound = CompoundFilter(filters, mode="any")

        terrain = TerrainData(slope=30, height=100)
        assert compound.evaluate(0, 0, terrain) is False

    def test_invalid_mode(self):
        """Test validation of mode."""
        with pytest.raises(ValueError, match="mode must be"):
            CompoundFilter([], mode="invalid")

    def test_empty_filters(self):
        """Test compound filter with no filters."""
        compound = CompoundFilter([], mode="all")
        assert compound.evaluate(0, 0, TerrainData()) is True

    def test_add_filter(self):
        """Test adding filters dynamically."""
        compound = CompoundFilter([], mode="all")

        compound.add_filter(SlopeFilterImpl(SlopeFilter(min_slope=0, max_slope=30)))
        terrain = TerrainData(slope=45)
        assert compound.evaluate(0, 0, terrain) is False

    def test_operator_and(self):
        """Test __and__ operator."""
        f1 = SlopeFilterImpl(SlopeFilter(min_slope=0, max_slope=45))
        f2 = HeightFilterImpl(HeightFilter(min_height=0, max_height=100))

        combined = f1 & f2
        assert isinstance(combined, CompoundFilter)
        assert combined.mode == "all"

    def test_operator_or(self):
        """Test __or__ operator."""
        f1 = SlopeFilterImpl(SlopeFilter(min_slope=0, max_slope=45))
        f2 = HeightFilterImpl(HeightFilter(min_height=0, max_height=100))

        combined = f1 | f2
        assert isinstance(combined, CompoundFilter)
        assert combined.mode == "any"


class TestBiomeRule:
    """Tests for BiomeRule dataclass."""

    def test_creation(self):
        """Test basic creation."""
        rule = BiomeRule(
            biome_id="forest",
            foliage_types=["oak", "pine", "birch"],
            density_multipliers={"oak": 1.0, "pine": 0.8},
        )
        assert rule.biome_id == "forest"
        assert len(rule.foliage_types) == 3

    def test_get_density_multiplier(self):
        """Test density multiplier lookup."""
        rule = BiomeRule(
            biome_id="forest",
            density_multipliers={"oak": 1.0, "pine": 0.8},
        )
        assert rule.get_density_multiplier("oak") == 1.0
        assert rule.get_density_multiplier("pine") == 0.8
        assert rule.get_density_multiplier("unknown") == 1.0  # Default

    def test_is_foliage_allowed_no_restriction(self):
        """Test foliage allowed when no types specified."""
        rule = BiomeRule(biome_id="any", foliage_types=[])
        assert rule.is_foliage_allowed("anything") is True

    def test_is_foliage_allowed_restricted(self):
        """Test foliage allowed with restrictions."""
        rule = BiomeRule(biome_id="forest", foliage_types=["oak", "pine"])
        assert rule.is_foliage_allowed("oak") is True
        assert rule.is_foliage_allowed("pine") is True
        assert rule.is_foliage_allowed("palm") is False

    def test_evaluate_filters(self):
        """Test filter evaluation."""
        rule = BiomeRule(
            biome_id="forest",
            filters=[
                SlopeFilterImpl(SlopeFilter(min_slope=0, max_slope=30)),
            ],
        )

        assert rule.evaluate_filters(0, 0, TerrainData(slope=15)) is True
        assert rule.evaluate_filters(0, 0, TerrainData(slope=45)) is False


class TestPlacementRuleSet:
    """Tests for PlacementRuleSet class."""

    def test_creation(self):
        """Test basic creation."""
        ruleset = PlacementRuleSet()
        assert len(ruleset.rules) == 0

    def test_add_rule(self):
        """Test adding rules."""
        ruleset = PlacementRuleSet()
        rule = BiomeRule(biome_id="forest", foliage_types=["oak"])
        ruleset.add_rule(rule)

        assert "forest" in ruleset.rules
        assert ruleset.get_rule("forest") is rule

    def test_add_global_filter(self):
        """Test adding global filters."""
        ruleset = PlacementRuleSet()
        flt = SlopeFilterImpl(SlopeFilter(min_slope=0, max_slope=45))
        ruleset.add_global_filter(flt)

        assert flt in ruleset.global_filters

    def test_default_rule(self):
        """Test default rule for unknown biomes."""
        ruleset = PlacementRuleSet()
        default = BiomeRule(biome_id="default", foliage_types=["grass"])
        ruleset.set_default_rule(default)

        assert ruleset.get_rule("unknown") is default

    def test_evaluate_global_filter_fail(self):
        """Test evaluation fails when global filter fails."""
        ruleset = PlacementRuleSet()
        ruleset.add_global_filter(SlopeFilterImpl(SlopeFilter(min_slope=0, max_slope=30)))
        ruleset.add_rule(BiomeRule(biome_id="forest", foliage_types=["oak"]))

        terrain = TerrainData(biome_id="forest", slope=45)  # Too steep
        result = ruleset.evaluate(0, 0, terrain)
        assert result == []

    def test_evaluate_no_matching_biome(self):
        """Test evaluation with no matching biome."""
        ruleset = PlacementRuleSet()
        ruleset.add_rule(BiomeRule(biome_id="forest", foliage_types=["oak"]))

        terrain = TerrainData(biome_id="desert")
        result = ruleset.evaluate(0, 0, terrain)
        assert result == []

    def test_evaluate_success(self):
        """Test successful evaluation."""
        ruleset = PlacementRuleSet()
        ruleset.add_rule(BiomeRule(
            biome_id="forest",
            foliage_types=["oak", "pine"],
        ))

        terrain = TerrainData(biome_id="forest")
        result = ruleset.evaluate(0, 0, terrain)
        assert "oak" in result
        assert "pine" in result

    def test_evaluate_specific_type(self):
        """Test evaluation for specific foliage type."""
        ruleset = PlacementRuleSet()
        ruleset.add_rule(BiomeRule(
            biome_id="forest",
            foliage_types=["oak", "pine"],
        ))

        terrain = TerrainData(biome_id="forest")
        result = ruleset.evaluate(0, 0, terrain, foliage_type="oak")
        assert result == ["oak"]

        result = ruleset.evaluate(0, 0, terrain, foliage_type="palm")
        assert result == []


class TestTransformRule:
    """Tests for TransformRule dataclass."""

    def test_default_values(self):
        """Test default transform rule values."""
        rule = TransformRule()
        assert rule.scale_range == (0.8, 1.2)
        assert rule.rotation_range == (0.0, 360.0)
        assert rule.offset_range == (0.0, 0.0)

    def test_invalid_scale_range(self):
        """Test validation of scale range."""
        with pytest.raises(ValueError, match="scale_range min must be <= max"):
            TransformRule(scale_range=(2.0, 1.0))

        with pytest.raises(ValueError, match="scale_range min must be > 0"):
            TransformRule(scale_range=(0.0, 1.0))

    def test_invalid_rotation_range(self):
        """Test validation of rotation range."""
        with pytest.raises(ValueError, match="rotation_range min must be <= max"):
            TransformRule(rotation_range=(360.0, 0.0))

    def test_apply_determinism(self):
        """Test that apply is deterministic."""
        rule = TransformRule()
        base = Transform.identity()

        result1 = rule.apply(base, seed=42)
        result2 = rule.apply(base, seed=42)

        assert result1.scale == result2.scale
        assert result1.rotation == result2.rotation
        assert result1.position == result2.position

    def test_apply_different_seeds(self):
        """Test that different seeds produce different results."""
        rule = TransformRule()
        base = Transform.identity()

        result1 = rule.apply(base, seed=42)
        result2 = rule.apply(base, seed=43)

        assert result1.scale != result2.scale or result1.rotation != result2.rotation


class TestTransform:
    """Tests for Transform dataclass."""

    def test_default_values(self):
        """Test default transform values."""
        t = Transform()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_identity(self):
        """Test identity factory method."""
        t = Transform.identity()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_from_position(self):
        """Test from_position factory method."""
        t = Transform.from_position(10.0, 20.0, 30.0)
        assert t.position == (10.0, 20.0, 30.0)
        assert t.scale == (1.0, 1.0, 1.0)


class TestPlacementValidator:
    """Tests for PlacementValidator class."""

    def test_creation(self):
        """Test basic creation."""
        validator = PlacementValidator()
        assert validator.rule_set is not None

    def test_is_valid_no_rules(self):
        """Test validation with no rules."""
        validator = PlacementValidator()
        # No rules = nothing valid
        assert validator.is_valid(0, 0) is False

    def test_is_valid_with_rules(self):
        """Test validation with rules."""
        ruleset = PlacementRuleSet()
        ruleset.add_rule(BiomeRule(biome_id="default", foliage_types=["grass"]))

        validator = PlacementValidator(ruleset)
        assert validator.is_valid(0, 0, foliage_type="grass") is True
        assert validator.is_valid(0, 0, foliage_type="tree") is False

    def test_terrain_sampler(self):
        """Test with terrain sampler."""
        ruleset = PlacementRuleSet()
        ruleset.add_rule(BiomeRule(biome_id="forest", foliage_types=["oak"]))

        def sampler(x, z):
            return TerrainData(biome_id="forest")

        validator = PlacementValidator(ruleset, sampler)
        assert validator.is_valid(0, 0, foliage_type="oak") is True

    def test_get_valid_types(self):
        """Test getting valid types."""
        ruleset = PlacementRuleSet()
        ruleset.add_rule(BiomeRule(
            biome_id="default",
            foliage_types=["grass", "flower"],
        ))

        validator = PlacementValidator(ruleset)
        types = validator.get_valid_types(0, 0)
        assert "grass" in types
        assert "flower" in types


class TestFactoryFunctions:
    """Tests for filter factory functions."""

    def test_create_slope_filter(self):
        """Test slope filter factory."""
        flt = create_slope_filter(min_slope=10, max_slope=40)
        assert isinstance(flt, SlopeFilterImpl)
        assert flt.config.min_slope == 10
        assert flt.config.max_slope == 40

    def test_create_height_filter(self):
        """Test height filter factory."""
        flt = create_height_filter(min_height=50, max_height=200)
        assert isinstance(flt, HeightFilterImpl)
        assert flt.config.min_height == 50
        assert flt.config.max_height == 200

    def test_create_layer_filter(self):
        """Test layer filter factory."""
        flt = create_layer_filter([1, 2, 3])
        assert isinstance(flt, LayerFilterImpl)
        assert flt.config.allowed_layers == [1, 2, 3]

    def test_create_noise_filter(self):
        """Test noise filter factory."""
        flt = create_noise_filter(seed=42, threshold=0.6, invert=True, frequency=2.0)
        assert isinstance(flt, NoiseFilterImpl)
        assert flt.config.threshold == 0.6
        assert flt.config.invert is True

    def test_create_exclusion_filter(self):
        """Test exclusion filter factory."""
        zones = [ExclusionZone(center=(50, 50), radius=10)]
        flt = create_exclusion_filter(zones)
        assert isinstance(flt, ExclusionZoneFilter)
        assert len(flt.zones) == 1
