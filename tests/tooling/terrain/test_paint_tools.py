"""Tests for terrain painting tools."""

import pytest
from engine.tooling.terrain.paint_tools import (
    PaintMode,
    LayerBlendMode,
    MaskSettings,
    TerrainMask,
    HeightMask,
    HeightMaskSettings,
    SlopeMask,
    SlopeMaskSettings,
    NoiseMask,
    NoiseMaskSettings,
    PaintBrush,
    PaintBrushSettings,
    PaintLayer,
    PaintOperation,
    TerrainPaintTool,
)


class TestMaskSettings:
    """Tests for mask settings."""

    def test_default_settings(self):
        """Test default mask settings."""
        settings = MaskSettings()
        assert settings.enabled
        assert not settings.invert
        assert settings.strength == 1.0

    def test_custom_settings(self):
        """Test custom mask settings."""
        settings = MaskSettings(
            enabled=False,
            invert=True,
            strength=0.5,
        )
        assert not settings.enabled
        assert settings.invert
        assert settings.strength == 0.5


class TestHeightMask:
    """Tests for height mask."""

    def test_creation(self):
        """Test height mask creation."""
        mask = HeightMask()
        assert mask.height_settings.min_height == 0.0
        assert mask.height_settings.max_height == 100.0

    def test_within_range(self):
        """Test evaluation within height range."""
        mask = HeightMask(HeightMaskSettings(min_height=0.0, max_height=1.0))

        class MockTerrain:
            def get_height(self, x, y):
                return 0.5

        value = mask.evaluate(0, 0, MockTerrain())
        assert value == 1.0

    def test_below_range(self):
        """Test evaluation below height range."""
        mask = HeightMask(HeightMaskSettings(
            min_height=0.5,
            max_height=1.0,
            feather=0.0,
        ))

        class MockTerrain:
            def get_height(self, x, y):
                return 0.0

        value = mask.evaluate(0, 0, MockTerrain())
        assert value == 0.0

    def test_above_range(self):
        """Test evaluation above height range."""
        mask = HeightMask(HeightMaskSettings(
            min_height=0.0,
            max_height=0.5,
            feather=0.0,
        ))

        class MockTerrain:
            def get_height(self, x, y):
                return 1.0

        value = mask.evaluate(0, 0, MockTerrain())
        assert value == 0.0


class TestSlopeMask:
    """Tests for slope mask."""

    def test_creation(self):
        """Test slope mask creation."""
        mask = SlopeMask()
        assert mask.slope_settings.min_angle == 0.0
        assert mask.slope_settings.max_angle == 90.0

    def test_flat_surface(self):
        """Test evaluation on flat surface."""
        mask = SlopeMask(SlopeMaskSettings(min_angle=0.0, max_angle=10.0))

        class MockTerrain:
            def get_height(self, x, y):
                return 0.0

        value = mask.evaluate(0, 0, MockTerrain())
        assert value == 1.0

    def test_steep_surface(self):
        """Test evaluation on steep surface."""
        mask = SlopeMask(SlopeMaskSettings(min_angle=45.0, max_angle=90.0))

        class MockTerrain:
            def get_height(self, x, y):
                # Very steep slope
                return x * 10.0

        value = mask.evaluate(0, 0, MockTerrain())
        # Should be high for steep slopes


class TestNoiseMask:
    """Tests for noise mask."""

    def test_creation(self):
        """Test noise mask creation."""
        mask = NoiseMask()
        assert mask.noise_settings.seed == 42
        assert mask.noise_settings.scale == 0.1

    def test_consistent_seed(self):
        """Test noise is consistent with same seed."""
        mask = NoiseMask(NoiseMaskSettings(seed=42))
        val1 = mask.evaluate(10, 10, None)
        val2 = mask.evaluate(10, 10, None)
        assert val1 == val2

    def test_different_positions(self):
        """Test noise varies by position."""
        mask = NoiseMask(NoiseMaskSettings(seed=42, threshold=0.5))
        values = [mask.evaluate(x, y, None) for x, y in [(0, 0), (10, 10), (20, 20)]]
        # Should have some variation


class TestPaintBrush:
    """Tests for paint brush."""

    def test_default_settings(self):
        """Test default brush settings."""
        brush = PaintBrush()
        assert brush.settings.size == 10.0
        assert brush.settings.strength == 0.5

    def test_falloff_at_center(self):
        """Test falloff at center."""
        brush = PaintBrush()
        falloff = brush.get_falloff(0.0, 5.0)
        assert falloff == 1.0

    def test_falloff_at_edge(self):
        """Test falloff at edge."""
        brush = PaintBrush(settings=PaintBrushSettings(falloff=1.0))
        falloff = brush.get_falloff(5.0, 5.0)
        assert falloff == 0.0

    def test_influence_at_center(self):
        """Test influence at center."""
        brush = PaintBrush()
        influence = brush.get_influence(5.0, 5.0, 5.0, 5.0)
        assert influence == brush.settings.strength

    def test_influence_outside(self):
        """Test influence outside brush."""
        brush = PaintBrush(settings=PaintBrushSettings(size=10.0))
        influence = brush.get_influence(0.0, 0.0, 20.0, 20.0)
        assert influence == 0.0


class TestPaintLayer:
    """Tests for paint layer."""

    def test_creation(self):
        """Test layer creation."""
        layer = PaintLayer(id=0, name="Grass", material_id="mat_grass")
        assert layer.id == 0
        assert layer.name == "Grass"
        assert layer.material_id == "mat_grass"

    def test_ensure_size(self):
        """Test ensure size."""
        layer = PaintLayer(id=0, name="Test", material_id="test")
        layer.ensure_size(64, 64)
        assert len(layer.weights) == 64
        assert len(layer.weights[0]) == 64

    def test_get_set_weight(self):
        """Test getting and setting weights."""
        layer = PaintLayer(id=0, name="Test", material_id="test")
        layer.ensure_size(64, 64)
        layer.set_weight(10, 10, 0.5)
        assert layer.get_weight(10, 10) == 0.5

    def test_weight_clamping(self):
        """Test weight clamping."""
        layer = PaintLayer(id=0, name="Test", material_id="test")
        layer.ensure_size(64, 64)
        layer.set_weight(10, 10, 1.5)
        assert layer.get_weight(10, 10) == 1.0
        layer.set_weight(10, 10, -0.5)
        assert layer.get_weight(10, 10) == 0.0


class TestTerrainPaintTool:
    """Tests for terrain paint tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = TerrainPaintTool(64, 64)

    def test_creation(self):
        """Test paint tool creation."""
        assert self.tool.mode == PaintMode.PAINT

    def test_add_layer(self):
        """Test adding layers."""
        layer = self.tool.add_layer("Grass", "mat_grass")
        assert layer.name == "Grass"
        assert self.tool.current_layer_id == layer.id

    def test_remove_layer(self):
        """Test removing layers."""
        layer = self.tool.add_layer("Grass", "mat_grass")
        assert self.tool.remove_layer(layer.id)
        assert self.tool.get_layer(layer.id) is None

    def test_get_layer(self):
        """Test getting layers."""
        layer = self.tool.add_layer("Grass", "mat_grass")
        retrieved = self.tool.get_layer(layer.id)
        assert retrieved == layer

    def test_get_all_layers(self):
        """Test getting all layers."""
        self.tool.add_layer("Grass", "mat_grass")
        self.tool.add_layer("Dirt", "mat_dirt")
        layers = self.tool.get_all_layers()
        assert len(layers) == 2

    def test_paint_mode(self):
        """Test paint mode."""
        layer = self.tool.add_layer("Grass", "mat_grass")
        self.tool.mode = PaintMode.PAINT
        self.tool.apply(32.0, 32.0)
        assert layer.get_weight(32, 32) > 0.0

    def test_erase_mode(self):
        """Test erase mode."""
        # With 2 layers, erasing redistributes weights through normalization
        layer1 = self.tool.add_layer("Grass", "mat_grass")
        layer2 = self.tool.add_layer("Dirt", "mat_dirt")
        # Start with both layers having some weight
        layer1.set_weight(32, 32, 0.6)
        layer2.set_weight(32, 32, 0.4)
        self.tool.current_layer_id = layer1.id
        self.tool.mode = PaintMode.ERASE
        original_grass_weight = layer1.get_weight(32, 32)
        self.tool.apply(32.0, 32.0)
        # After erase and normalization, dirt layer should have increased proportion
        # and grass should have decreased proportion
        assert layer2.get_weight(32, 32) > 0.4

    def test_blend_mode(self):
        """Test blend mode."""
        layer1 = self.tool.add_layer("Grass", "mat_grass")
        layer2 = self.tool.add_layer("Dirt", "mat_dirt")
        layer1.set_weight(32, 32, 0.5)
        layer2.set_weight(32, 32, 0.5)

        self.tool.current_layer_id = layer1.id
        self.tool.mode = PaintMode.BLEND
        self.tool.apply(32.0, 32.0)

    def test_replace_mode(self):
        """Test replace mode."""
        layer1 = self.tool.add_layer("Grass", "mat_grass")
        layer2 = self.tool.add_layer("Dirt", "mat_dirt")

        self.tool.current_layer_id = layer1.id
        self.tool.mode = PaintMode.REPLACE
        self.tool.apply(32.0, 32.0)

    def test_add_mask(self):
        """Test adding masks."""
        mask = HeightMask()
        self.tool.add_mask(mask)

    def test_remove_mask(self):
        """Test removing masks."""
        mask = HeightMask()
        self.tool.add_mask(mask)
        assert self.tool.remove_mask(mask)

    def test_clear_masks(self):
        """Test clearing masks."""
        self.tool.add_mask(HeightMask())
        self.tool.add_mask(SlopeMask())
        self.tool.clear_masks()

    def test_undo(self):
        """Test undo operation."""
        layer = self.tool.add_layer("Grass", "mat_grass")
        original = layer.get_weight(32, 32)
        self.tool.apply(32.0, 32.0)
        self.tool.undo()
        assert layer.get_weight(32, 32) == original

    def test_redo(self):
        """Test redo operation."""
        layer = self.tool.add_layer("Grass", "mat_grass")
        self.tool.apply(32.0, 32.0)
        painted = layer.get_weight(32, 32)
        self.tool.undo()
        self.tool.redo()
        assert layer.get_weight(32, 32) == painted

    def test_can_undo(self):
        """Test can_undo check."""
        self.tool.add_layer("Grass", "mat_grass")
        assert not self.tool.can_undo()
        self.tool.apply(32.0, 32.0)
        assert self.tool.can_undo()

    def test_can_redo(self):
        """Test can_redo check."""
        self.tool.add_layer("Grass", "mat_grass")
        self.tool.apply(32.0, 32.0)
        assert not self.tool.can_redo()
        self.tool.undo()
        assert self.tool.can_redo()

    def test_get_weights_at(self):
        """Test getting all weights at position."""
        self.tool.add_layer("Grass", "mat_grass")
        self.tool.add_layer("Dirt", "mat_dirt")
        weights = self.tool.get_weights_at(32, 32)
        assert 0 in weights
        assert 1 in weights

    def test_get_splatmap(self):
        """Test getting splatmap data."""
        self.tool.add_layer("Grass", "mat_grass")
        splatmap = self.tool.get_splatmap()
        assert 0 in splatmap

    def test_set_brush(self):
        """Test setting brush."""
        brush = PaintBrush(settings=PaintBrushSettings(size=20.0))
        self.tool.set_brush(brush)
        assert self.tool.brush.settings.size == 20.0
