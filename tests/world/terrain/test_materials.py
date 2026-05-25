"""Tests for terrain materials and weight maps."""

import math
from typing import Tuple

import pytest

from engine.world.terrain.materials import (
    AutoLayerRule,
    BlendTechnique,
    MaterialPalette,
    TerrainLayer,
    TerrainLayerType,
    TerrainMaterial,
    WeightMap,
)


class MockHeightfield:
    """Mock heightfield for testing."""

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        sample_spacing: float = 1.0,
    ):
        self._width = width
        self._height = height
        self._sample_spacing = sample_spacing
        self._heights = [[0.0 for _ in range(width)] for _ in range(height)]
        self._slopes = [[0.0 for _ in range(width)] for _ in range(height)]

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def sample_spacing(self) -> float:
        return self._sample_spacing

    def get_height_at(self, x: int, z: int) -> float:
        return self._heights[z][x]

    def set_height_at(self, x: int, z: int, height: float) -> None:
        self._heights[z][x] = height

    def get_slope_at(self, x: int, z: int) -> float:
        return self._slopes[z][x]

    def set_slope_at(self, x: int, z: int, slope: float) -> None:
        self._slopes[z][x] = slope


# ============================================================================
# TerrainLayer tests
# ============================================================================


class TestTerrainLayer:
    """Tests for TerrainLayer dataclass."""

    def test_default_values(self):
        """Test default layer values."""
        layer = TerrainLayer()
        assert layer.name == ""
        assert layer.material_id == ""
        assert layer.layer_type == TerrainLayerType.BLEND
        assert layer.tiling_scale == 1.0
        assert layer.normal_scale == 1.0
        assert layer.height_offset == 0.0

    def test_custom_values(self):
        """Test custom layer values."""
        layer = TerrainLayer(
            name="Grass",
            material_id="mat_grass",
            layer_type=TerrainLayerType.AUTO,
            tiling_scale=2.0,
            normal_scale=0.5,
            height_offset=0.1,
        )
        assert layer.name == "Grass"
        assert layer.material_id == "mat_grass"
        assert layer.layer_type == TerrainLayerType.AUTO
        assert layer.tiling_scale == 2.0
        assert layer.normal_scale == 0.5
        assert layer.height_offset == 0.1

    def test_invalid_tiling_scale(self):
        """Test that tiling_scale must be positive."""
        with pytest.raises(ValueError, match="tiling_scale must be > 0"):
            TerrainLayer(tiling_scale=0)

        with pytest.raises(ValueError, match="tiling_scale must be > 0"):
            TerrainLayer(tiling_scale=-1.0)

    def test_invalid_normal_scale(self):
        """Test that normal_scale must be non-negative."""
        with pytest.raises(ValueError, match="normal_scale must be >= 0"):
            TerrainLayer(normal_scale=-0.1)


# ============================================================================
# WeightMap tests
# ============================================================================


class TestWeightMap:
    """Tests for WeightMap class."""

    def test_initialization(self):
        """Test weight map initialization."""
        wm = WeightMap(width=32, height=32, num_layers=4)
        assert wm.width == 32
        assert wm.height == 32
        assert wm.num_layers == 4

    def test_default_layer_has_full_weight(self):
        """Test that default layer starts with full weight."""
        wm = WeightMap(width=32, height=32, num_layers=4, default_layer=0)

        assert wm.get_weight_at(0, 0, 0) == 1.0
        assert wm.get_weight_at(0, 0, 1) == 0.0
        assert wm.get_weight_at(0, 0, 2) == 0.0
        assert wm.get_weight_at(0, 0, 3) == 0.0

    def test_custom_default_layer(self):
        """Test custom default layer."""
        wm = WeightMap(width=32, height=32, num_layers=4, default_layer=2)

        assert wm.get_weight_at(0, 0, 0) == 0.0
        assert wm.get_weight_at(0, 0, 2) == 1.0

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError, match="width and height must be > 0"):
            WeightMap(width=0, height=32, num_layers=4)

        with pytest.raises(ValueError, match="width and height must be > 0"):
            WeightMap(width=32, height=0, num_layers=4)

    def test_invalid_num_layers(self):
        """Test that invalid num_layers raises error."""
        with pytest.raises(ValueError, match="num_layers must be > 0"):
            WeightMap(width=32, height=32, num_layers=0)

    def test_invalid_default_layer(self):
        """Test that invalid default_layer raises error."""
        with pytest.raises(ValueError, match="default_layer must be in range"):
            WeightMap(width=32, height=32, num_layers=4, default_layer=4)

    def test_set_and_get_weight(self):
        """Test setting and getting weights."""
        wm = WeightMap(width=32, height=32, num_layers=4)

        wm.set_weight_at(5, 10, 1, 0.75)
        assert wm.get_weight_at(5, 10, 1) == 0.75

    def test_weight_clamping(self):
        """Test that weights are clamped to [0, 1]."""
        wm = WeightMap(width=32, height=32, num_layers=4)

        wm.set_weight_at(0, 0, 1, 2.0)
        assert wm.get_weight_at(0, 0, 1) == 1.0

        wm.set_weight_at(0, 0, 1, -0.5)
        assert wm.get_weight_at(0, 0, 1) == 0.0

    def test_invalid_coords(self):
        """Test that invalid coordinates raise errors."""
        wm = WeightMap(width=32, height=32, num_layers=4)

        with pytest.raises(ValueError, match="x must be in range"):
            wm.get_weight_at(32, 0, 0)

        with pytest.raises(ValueError, match="z must be in range"):
            wm.get_weight_at(0, 32, 0)

    def test_invalid_layer_index(self):
        """Test that invalid layer index raises error."""
        wm = WeightMap(width=32, height=32, num_layers=4)

        with pytest.raises(ValueError, match="layer_index must be in range"):
            wm.get_weight_at(0, 0, 4)

    def test_get_all_weights_at(self):
        """Test getting all weights at a position."""
        wm = WeightMap(width=32, height=32, num_layers=4, default_layer=0)

        weights = wm.get_all_weights_at(0, 0)
        assert len(weights) == 4
        assert weights[0] == 1.0
        assert weights[1] == 0.0
        assert weights[2] == 0.0
        assert weights[3] == 0.0

    def test_normalize_at(self):
        """Test normalizing weights at a position."""
        wm = WeightMap(width=32, height=32, num_layers=4)

        # Set non-normalized weights
        wm.set_weight_at(0, 0, 0, 0.5)
        wm.set_weight_at(0, 0, 1, 0.5)
        wm.set_weight_at(0, 0, 2, 0.5)
        wm.set_weight_at(0, 0, 3, 0.5)

        wm.normalize_at(0, 0)

        weights = wm.get_all_weights_at(0, 0)
        total = sum(weights)
        assert abs(total - 1.0) < 0.001

    def test_normalize_all_zero_weights(self):
        """Test normalizing when all weights are zero."""
        wm = WeightMap(width=32, height=32, num_layers=4)

        # Set all weights to zero
        for i in range(4):
            wm.set_weight_at(0, 0, i, 0.0)

        wm.normalize_at(0, 0)

        # First layer should be set to 1.0
        assert wm.get_weight_at(0, 0, 0) == 1.0

    def test_normalize_all(self):
        """Test normalizing all positions."""
        wm = WeightMap(width=8, height=8, num_layers=4)

        # Set non-normalized weights everywhere
        for z in range(8):
            for x in range(8):
                wm.set_weight_at(x, z, 0, 0.3)
                wm.set_weight_at(x, z, 1, 0.3)

        wm.normalize_all()

        # Check all positions are normalized
        for z in range(8):
            for x in range(8):
                total = sum(wm.get_all_weights_at(x, z))
                assert abs(total - 1.0) < 0.001

    def test_paint(self):
        """Test painting weights."""
        wm = WeightMap(width=32, height=32, num_layers=4, default_layer=0)

        # Paint layer 1 at center
        wm.paint(
            center_x=16,
            center_z=16,
            radius=5,
            layer_index=1,
            strength=1.0,
            falloff=0.0,
        )

        # Center should have layer 1 weight
        assert wm.get_weight_at(16, 16, 1) > 0.0
        # Layer 0 should be reduced
        assert wm.get_weight_at(16, 16, 0) < 1.0

    def test_paint_invalid_radius(self):
        """Test that invalid paint radius raises error."""
        wm = WeightMap(width=32, height=32, num_layers=4)

        with pytest.raises(ValueError, match="radius must be > 0"):
            wm.paint(16, 16, 0, 1, 0.5, 0.5)

    def test_paint_invalid_strength(self):
        """Test that invalid paint strength raises error."""
        wm = WeightMap(width=32, height=32, num_layers=4)

        with pytest.raises(ValueError, match="strength must be in range"):
            wm.paint(16, 16, 5, 1, 1.5, 0.5)

    def test_paint_maintains_normalization(self):
        """Test that painting maintains weight normalization."""
        wm = WeightMap(width=32, height=32, num_layers=4, default_layer=0)

        wm.paint(16, 16, 5, 1, 0.5, 0.5)

        # All weights should still approximately sum to 1
        weights = wm.get_all_weights_at(16, 16)
        total = sum(weights)
        assert abs(total - 1.0) < 0.01

    def test_get_dominant_layer_at(self):
        """Test getting dominant layer."""
        wm = WeightMap(width=32, height=32, num_layers=4)

        wm.set_weight_at(0, 0, 0, 0.1)
        wm.set_weight_at(0, 0, 1, 0.2)
        wm.set_weight_at(0, 0, 2, 0.5)
        wm.set_weight_at(0, 0, 3, 0.2)

        assert wm.get_dominant_layer_at(0, 0) == 2

    def test_clear_layer(self):
        """Test clearing a layer."""
        wm = WeightMap(width=8, height=8, num_layers=4, default_layer=0)

        wm.clear_layer(0)

        for z in range(8):
            for x in range(8):
                assert wm.get_weight_at(x, z, 0) == 0.0

    def test_resize(self):
        """Test resizing weight map."""
        wm = WeightMap(width=16, height=16, num_layers=4, default_layer=0)

        # Paint something before resize
        wm.paint(8, 8, 4, 1, 1.0, 0.0)

        wm.resize(32, 32)

        assert wm.width == 32
        assert wm.height == 32
        # Should preserve weights (interpolated)
        assert wm.get_weight_at(16, 16, 1) > 0.0


# ============================================================================
# AutoLayerRule tests
# ============================================================================


class TestAutoLayerRule:
    """Tests for AutoLayerRule dataclass."""

    def test_default_values(self):
        """Test default rule values."""
        rule = AutoLayerRule(layer_index=0)
        assert rule.layer_index == 0
        assert rule.slope_range == (0.0, 90.0)
        assert rule.height_range is None
        assert rule.noise_scale == 0.0
        assert rule.noise_threshold == 0.5

    def test_invalid_layer_index(self):
        """Test that negative layer_index raises error."""
        with pytest.raises(ValueError, match="layer_index must be >= 0"):
            AutoLayerRule(layer_index=-1)

    def test_invalid_slope_range(self):
        """Test that invalid slope_range raises error."""
        with pytest.raises(ValueError, match="slope_range\\[0\\] must be <= slope_range\\[1\\]"):
            AutoLayerRule(layer_index=0, slope_range=(90.0, 0.0))

    def test_invalid_height_range(self):
        """Test that invalid height_range raises error."""
        with pytest.raises(ValueError, match="height_range\\[0\\] must be <= height_range\\[1\\]"):
            AutoLayerRule(layer_index=0, height_range=(100.0, 0.0))

    def test_invalid_noise_scale(self):
        """Test that negative noise_scale raises error."""
        with pytest.raises(ValueError, match="noise_scale must be >= 0"):
            AutoLayerRule(layer_index=0, noise_scale=-0.1)

    def test_invalid_noise_threshold(self):
        """Test that invalid noise_threshold raises error."""
        with pytest.raises(ValueError, match="noise_threshold must be in range"):
            AutoLayerRule(layer_index=0, noise_threshold=1.5)

    def test_evaluate_within_slope_range(self):
        """Test evaluation within slope range."""
        rule = AutoLayerRule(layer_index=0, slope_range=(0.0, 30.0))

        # Center of range should give high weight
        weight = rule.evaluate(slope=15.0, height=0.0)
        assert weight > 0.5

    def test_evaluate_outside_slope_range(self):
        """Test evaluation outside slope range."""
        rule = AutoLayerRule(layer_index=0, slope_range=(0.0, 30.0))

        # Outside range should give zero weight
        weight = rule.evaluate(slope=45.0, height=0.0)
        assert weight == 0.0

    def test_evaluate_height_range(self):
        """Test evaluation with height range."""
        rule = AutoLayerRule(
            layer_index=0,
            slope_range=(0.0, 90.0),
            height_range=(100.0, 200.0),
        )

        # Within height range - use slope at center of range for max weight
        weight_in = rule.evaluate(slope=45.0, height=150.0)
        assert weight_in > 0.0

        # Outside height range
        weight_out = rule.evaluate(slope=45.0, height=50.0)
        assert weight_out == 0.0

    def test_evaluate_with_noise(self):
        """Test evaluation with noise."""
        rule = AutoLayerRule(
            layer_index=0,
            slope_range=(0.0, 90.0),
            noise_scale=0.1,
            noise_threshold=0.5,
        )

        # Low noise value (below threshold)
        weight_low = rule.evaluate(slope=45.0, height=0.0, noise_value=0.3)
        assert weight_low == 0.0

        # High noise value (above threshold) - use center of slope range
        weight_high = rule.evaluate(slope=45.0, height=0.0, noise_value=0.8)
        assert weight_high > 0.0


# ============================================================================
# TerrainMaterial tests
# ============================================================================


class TestTerrainMaterial:
    """Tests for TerrainMaterial class."""

    def test_initialization(self):
        """Test terrain material initialization."""
        tm = TerrainMaterial(width=64, height=64)
        assert tm.layer_count == 0
        assert tm.weight_map is None
        assert tm.blend_technique == BlendTechnique.HEIGHT_BLEND

    def test_add_layer(self):
        """Test adding layers."""
        tm = TerrainMaterial(width=64, height=64)

        layer = TerrainLayer(name="Grass", material_id="mat_grass")
        index = tm.add_layer(layer)

        assert index == 0
        assert tm.layer_count == 1
        assert tm.weight_map is not None
        assert tm.weight_map.num_layers == 1

    def test_add_multiple_layers(self):
        """Test adding multiple layers."""
        tm = TerrainMaterial(width=64, height=64)

        tm.add_layer(TerrainLayer(name="Grass"))
        tm.add_layer(TerrainLayer(name="Rock"))
        tm.add_layer(TerrainLayer(name="Dirt"))

        assert tm.layer_count == 3
        assert tm.weight_map.num_layers == 3

    def test_get_layer(self):
        """Test getting layer by index."""
        tm = TerrainMaterial(width=64, height=64)

        grass = TerrainLayer(name="Grass")
        rock = TerrainLayer(name="Rock")
        tm.add_layer(grass)
        tm.add_layer(rock)

        assert tm.get_layer(0).name == "Grass"
        assert tm.get_layer(1).name == "Rock"

    def test_get_layer_invalid_index(self):
        """Test getting layer with invalid index."""
        tm = TerrainMaterial(width=64, height=64)
        tm.add_layer(TerrainLayer(name="Grass"))

        with pytest.raises(ValueError, match="layer_index must be in range"):
            tm.get_layer(1)

    def test_remove_layer(self):
        """Test removing a layer."""
        tm = TerrainMaterial(width=64, height=64)

        tm.add_layer(TerrainLayer(name="Grass"))
        tm.add_layer(TerrainLayer(name="Rock"))
        tm.add_layer(TerrainLayer(name="Dirt"))

        tm.remove_layer(1)

        assert tm.layer_count == 2
        assert tm.get_layer(0).name == "Grass"
        assert tm.get_layer(1).name == "Dirt"

    def test_remove_last_layer_raises(self):
        """Test that removing the last layer raises error."""
        tm = TerrainMaterial(width=64, height=64)
        tm.add_layer(TerrainLayer(name="Grass"))

        with pytest.raises(ValueError, match="Cannot remove the last layer"):
            tm.remove_layer(0)

    def test_layers_property_returns_copy(self):
        """Test that layers property returns a copy."""
        tm = TerrainMaterial(width=64, height=64)
        tm.add_layer(TerrainLayer(name="Grass"))

        layers = tm.layers
        layers.append(TerrainLayer(name="Rock"))

        assert tm.layer_count == 1

    def test_blend_technique_property(self):
        """Test blend technique property."""
        tm = TerrainMaterial(width=64, height=64)

        assert tm.blend_technique == BlendTechnique.HEIGHT_BLEND

        tm.blend_technique = BlendTechnique.LINEAR
        assert tm.blend_technique == BlendTechnique.LINEAR

    def test_add_auto_rule(self):
        """Test adding auto layer rules."""
        tm = TerrainMaterial(width=64, height=64)
        tm.add_layer(TerrainLayer(name="Grass"))
        tm.add_layer(TerrainLayer(name="Rock"))

        rule = AutoLayerRule(layer_index=1, slope_range=(30.0, 90.0))
        tm.add_auto_rule(rule)

        # Rule should be stored (no explicit accessor, just test no error)

    def test_add_auto_rule_invalid_layer(self):
        """Test adding auto rule with invalid layer index."""
        tm = TerrainMaterial(width=64, height=64)
        tm.add_layer(TerrainLayer(name="Grass"))

        rule = AutoLayerRule(layer_index=1)

        with pytest.raises(ValueError, match="rule.layer_index must be < number of layers"):
            tm.add_auto_rule(rule)

    def test_apply_auto_rules(self):
        """Test applying auto rules."""
        tm = TerrainMaterial(width=16, height=16)
        tm.add_layer(TerrainLayer(name="Grass"))
        tm.add_layer(TerrainLayer(name="Rock"))

        rule = AutoLayerRule(layer_index=1, slope_range=(30.0, 90.0))
        tm.add_auto_rule(rule)

        heightfield = MockHeightfield(width=16, height=16)
        # Set some steep slopes
        for z in range(16):
            for x in range(16):
                if x > 8:
                    heightfield.set_slope_at(x, z, 45.0)

        tm.apply_auto_rules(heightfield)

        # Areas with steep slopes should have rock
        rock_weight = tm.weight_map.get_weight_at(10, 8, 1)
        grass_weight = tm.weight_map.get_weight_at(4, 8, 0)

        assert rock_weight > 0.0

    def test_get_blend_weights_at(self):
        """Test getting blend weights at a position."""
        tm = TerrainMaterial(width=64, height=64)
        tm.add_layer(TerrainLayer(name="Grass"))
        tm.add_layer(TerrainLayer(name="Rock"))

        weights = tm.get_blend_weights_at(0, 0)

        assert len(weights) == 2
        assert sum(weights) > 0.99  # Should be normalized

    def test_get_blend_weights_no_layers(self):
        """Test getting blend weights with no layers."""
        tm = TerrainMaterial(width=64, height=64)

        weights = tm.get_blend_weights_at(0, 0)
        assert weights == []

    def test_sample_blend_weights(self):
        """Test sampling blend weights with interpolation."""
        tm = TerrainMaterial(width=16, height=16)
        tm.add_layer(TerrainLayer(name="Grass"))
        tm.add_layer(TerrainLayer(name="Rock"))

        # Paint rock in one corner
        tm.weight_map.paint(0, 0, 4, 1, 1.0, 0.0)

        # Sample at corner should have rock
        weights_corner = tm.sample_blend_weights(1.5, 1.5, 1.0)
        assert weights_corner[1] > 0.5

        # Sample far away should have mostly grass
        weights_far = tm.sample_blend_weights(12.0, 12.0, 1.0)
        assert weights_far[0] > weights_far[1]

    def test_apply_height_blend(self):
        """Test height-based blending."""
        tm = TerrainMaterial(width=64, height=64)
        tm.add_layer(TerrainLayer(name="Grass", height_offset=0.0))
        tm.add_layer(TerrainLayer(name="Rock", height_offset=0.2))

        base_weights = [0.5, 0.5]
        height_values = [0.3, 0.5]

        blended = tm.apply_height_blend(base_weights, height_values)

        assert len(blended) == 2
        # Rock should have more weight due to height blend
        assert abs(sum(blended) - 1.0) < 0.001

    def test_clear_auto_rules(self):
        """Test clearing auto rules."""
        tm = TerrainMaterial(width=64, height=64)
        tm.add_layer(TerrainLayer(name="Grass"))
        tm.add_layer(TerrainLayer(name="Rock"))

        tm.add_auto_rule(AutoLayerRule(layer_index=0))
        tm.add_auto_rule(AutoLayerRule(layer_index=1))

        tm.clear_auto_rules()

        # Rules should be cleared (verify by applying empty rules)
        heightfield = MockHeightfield(width=16, height=16)
        tm.apply_auto_rules(heightfield)


# ============================================================================
# MaterialPalette tests
# ============================================================================


class TestMaterialPalette:
    """Tests for MaterialPalette class."""

    def test_create_natural_terrain(self):
        """Test creating natural terrain preset."""
        setup = MaterialPalette.create_natural_terrain()

        assert len(setup) == 4
        layer_names = [layer.name for layer, rule in setup]
        assert "Grass" in layer_names
        assert "Rock" in layer_names
        assert "Dirt" in layer_names
        assert "Snow" in layer_names

    def test_create_desert_terrain(self):
        """Test creating desert terrain preset."""
        setup = MaterialPalette.create_desert_terrain()

        assert len(setup) == 3
        layer_names = [layer.name for layer, rule in setup]
        assert "Sand" in layer_names
        assert "Sandstone" in layer_names
        assert "Gravel" in layer_names

    def test_preset_layers_have_valid_rules(self):
        """Test that preset layers have properly configured rules."""
        setup = MaterialPalette.create_natural_terrain()

        for layer, rule in setup:
            assert rule.layer_index >= 0
            assert rule.slope_range[0] <= rule.slope_range[1]


# ============================================================================
# Integration tests
# ============================================================================


class TestMaterialIntegration:
    """Integration tests for material system."""

    def test_full_workflow(self):
        """Test complete material workflow."""
        # Create material system
        tm = TerrainMaterial(width=32, height=32, blend_technique=BlendTechnique.HEIGHT_BLEND)

        # Add layers from preset
        setup = MaterialPalette.create_natural_terrain()
        for layer, rule in setup:
            tm.add_layer(layer)
            tm.add_auto_rule(rule)

        # Create heightfield with varying slopes
        heightfield = MockHeightfield(width=32, height=32)
        for z in range(32):
            for x in range(32):
                # Create gradient of slopes
                slope = (x / 32.0) * 90.0
                heightfield.set_slope_at(x, z, slope)
                # Create varying heights
                height = (z / 32.0) * 600.0
                heightfield.set_height_at(x, z, height)

        # Apply auto rules
        tm.apply_auto_rules(heightfield)

        # Verify weights are valid
        for z in range(32):
            for x in range(32):
                weights = tm.get_blend_weights_at(x, z)
                total = sum(weights)
                assert abs(total - 1.0) < 0.01, f"Weights not normalized at ({x}, {z})"

    def test_paint_over_auto_rules(self):
        """Test painting over automatically applied layers."""
        tm = TerrainMaterial(width=32, height=32)
        tm.add_layer(TerrainLayer(name="Grass"))
        tm.add_layer(TerrainLayer(name="Rock"))

        # Apply auto rule for rock on slopes
        tm.add_auto_rule(AutoLayerRule(layer_index=1, slope_range=(30.0, 90.0)))

        heightfield = MockHeightfield(width=32, height=32)
        for z in range(32):
            for x in range(32):
                heightfield.set_slope_at(x, z, 45.0)

        tm.apply_auto_rules(heightfield)

        # Now paint grass over rock
        tm.weight_map.paint(16, 16, 5, 0, 1.0, 0.0)

        # Center should have grass
        weights = tm.get_blend_weights_at(16, 16)
        assert weights[0] > 0.5


# ============================================================================
# Enhanced weight normalization tests
# ============================================================================


class TestEnhancedWeightNormalization:
    """Enhanced tests for weight normalization edge cases."""

    def test_normalize_exact_sum_to_one(self):
        """Verify weights sum to exactly 1.0 after normalization."""
        wm = WeightMap(width=16, height=16, num_layers=4)

        # Set various weight combinations
        wm.set_weight_at(0, 0, 0, 0.3)
        wm.set_weight_at(0, 0, 1, 0.3)
        wm.set_weight_at(0, 0, 2, 0.2)
        wm.set_weight_at(0, 0, 3, 0.2)

        wm.normalize_at(0, 0)
        weights = wm.get_all_weights_at(0, 0)
        total = sum(weights)

        # Should sum to exactly 1.0 (within floating point precision)
        assert abs(total - 1.0) < 1e-10, f"Expected sum of 1.0, got {total}"

    def test_normalize_zero_weights_uses_default_layer(self):
        """Verify zero weights use the configured default layer, not always layer 0."""
        # Use layer 2 as default
        wm = WeightMap(width=16, height=16, num_layers=4, default_layer=2)

        # Set all weights to zero
        for i in range(4):
            wm.set_weight_at(5, 5, i, 0.0)

        wm.normalize_at(5, 5)
        weights = wm.get_all_weights_at(5, 5)

        # Default layer (2) should be 1.0, others should be 0.0
        assert weights[0] == 0.0
        assert weights[1] == 0.0
        assert weights[2] == 1.0, f"Expected default layer (2) to be 1.0, got {weights[2]}"
        assert weights[3] == 0.0

    def test_normalize_very_small_weights(self):
        """Verify normalization works with very small weights."""
        wm = WeightMap(width=16, height=16, num_layers=4)

        # Set very small weights
        wm.set_weight_at(0, 0, 0, 1e-10)
        wm.set_weight_at(0, 0, 1, 1e-10)
        wm.set_weight_at(0, 0, 2, 1e-10)
        wm.set_weight_at(0, 0, 3, 1e-10)

        wm.normalize_at(0, 0)
        weights = wm.get_all_weights_at(0, 0)
        total = sum(weights)

        assert abs(total - 1.0) < 1e-6, f"Expected sum of 1.0, got {total}"

    def test_paint_preserves_exact_sum(self):
        """Verify painting maintains exact weight sum of 1.0."""
        wm = WeightMap(width=32, height=32, num_layers=4, default_layer=0)

        # Paint multiple layers
        wm.paint(16, 16, 5, 1, 0.5, 0.5)
        wm.paint(16, 16, 3, 2, 0.3, 0.5)

        weights = wm.get_all_weights_at(16, 16)
        total = sum(weights)

        # Sum should be exactly 1.0
        assert abs(total - 1.0) < 1e-10, f"Expected sum of 1.0, got {total}"

    def test_paint_weight_reduction_proportional(self):
        """Verify that painting reduces other layers proportionally."""
        wm = WeightMap(width=32, height=32, num_layers=3, default_layer=0)

        # First set up equal weights manually
        wm.set_weight_at(16, 16, 0, 0.5)
        wm.set_weight_at(16, 16, 1, 0.3)
        wm.set_weight_at(16, 16, 2, 0.2)

        # Paint layer 0 to full
        wm.paint(16, 16, 1, 0, 1.0, 0.0)

        weights = wm.get_all_weights_at(16, 16)

        # Layer 0 should be 1.0, others should be 0.0
        assert weights[0] == 1.0, f"Expected layer 0 to be 1.0, got {weights[0]}"
        # Other layers should be reduced to 0
        assert weights[1] < 0.01
        assert weights[2] < 0.01
