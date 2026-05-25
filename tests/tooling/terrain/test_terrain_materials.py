"""Tests for terrain material system."""

import pytest
from engine.tooling.terrain.terrain_materials import (
    BlendMode,
    TextureCoordinates,
    MaterialProperties,
    TerrainMaterialLayer,
    MaterialBlendSettings,
    MaterialSample,
    TerrainMaterialStack,
    TerrainMaterialManager,
)


class TestTextureCoordinates:
    """Tests for texture coordinates."""

    def test_default_values(self):
        """Test default UV settings."""
        uv = TextureCoordinates()
        assert uv.scale_u == 1.0
        assert uv.scale_v == 1.0
        assert uv.offset_u == 0.0
        assert uv.rotation == 0.0

    def test_custom_values(self):
        """Test custom UV settings."""
        uv = TextureCoordinates(scale_u=2.0, scale_v=2.0, rotation=0.5)
        assert uv.scale_u == 2.0
        assert uv.rotation == 0.5


class TestMaterialProperties:
    """Tests for material properties."""

    def test_default_values(self):
        """Test default material properties."""
        props = MaterialProperties()
        assert props.roughness == 0.5
        assert props.metallic == 0.0
        assert props.normal_strength == 1.0

    def test_custom_values(self):
        """Test custom material properties."""
        props = MaterialProperties(roughness=0.8, metallic=0.5)
        assert props.roughness == 0.8
        assert props.metallic == 0.5


class TestTerrainMaterialLayer:
    """Tests for material layer."""

    def test_creation(self):
        """Test layer creation."""
        layer = TerrainMaterialLayer(id=0, name="Grass")
        assert layer.id == 0
        assert layer.name == "Grass"
        assert layer.blend_mode == BlendMode.LINEAR

    def test_get_tint(self):
        """Test getting tint color."""
        layer = TerrainMaterialLayer(id=0, name="Test")
        tint = layer.get_tint()
        assert tint == (1.0, 1.0, 1.0)

    def test_set_tint(self):
        """Test setting tint color."""
        layer = TerrainMaterialLayer(id=0, name="Test")
        layer.set_tint(0.5, 0.6, 0.7)
        assert layer.get_tint() == (0.5, 0.6, 0.7)

    def test_tint_clamping(self):
        """Test tint value clamping."""
        layer = TerrainMaterialLayer(id=0, name="Test")
        layer.set_tint(1.5, -0.5, 0.5)
        tint = layer.get_tint()
        assert tint[0] == 1.0
        assert tint[1] == 0.0
        assert tint[2] == 0.5


class TestTerrainMaterialStack:
    """Tests for material stack."""

    def setup_method(self):
        """Set up test stack."""
        self.stack = TerrainMaterialStack(64, 64)

    def test_creation(self):
        """Test stack creation."""
        assert self.stack is not None

    def test_add_layer(self):
        """Test adding layers."""
        layer = self.stack.add_layer("Grass", albedo_texture="grass.png")
        assert layer.name == "Grass"
        assert layer.albedo_texture == "grass.png"

    def test_remove_layer(self):
        """Test removing layers."""
        layer = self.stack.add_layer("Grass")
        assert self.stack.remove_layer(layer.id)
        assert self.stack.get_layer(layer.id) is None

    def test_locked_layer(self):
        """Test locked layer cannot be removed."""
        layer = self.stack.add_layer("Base")
        layer.locked = True
        assert not self.stack.remove_layer(layer.id)

    def test_get_layer_by_name(self):
        """Test getting layer by name."""
        self.stack.add_layer("Grass")
        layer = self.stack.get_layer_by_name("Grass")
        assert layer is not None
        assert layer.name == "Grass"

    def test_get_all_layers(self):
        """Test getting all layers."""
        self.stack.add_layer("Grass")
        self.stack.add_layer("Rock")
        layers = self.stack.get_all_layers()
        assert len(layers) == 2

    def test_move_layer(self):
        """Test moving layer in stack."""
        self.stack.add_layer("Layer1")
        layer2 = self.stack.add_layer("Layer2")

        order_before = self.stack.get_layer_order()
        self.stack.move_layer(layer2.id, 0)
        order_after = self.stack.get_layer_order()

        assert order_before != order_after
        assert order_after[0] == layer2.id

    def test_set_get_weight(self):
        """Test setting and getting weights."""
        layer = self.stack.add_layer("Grass")
        self.stack.set_weight(layer.id, 10, 10, 0.5)
        assert self.stack.get_weight(layer.id, 10, 10) == 0.5

    def test_weight_clamping(self):
        """Test weight value clamping."""
        layer = self.stack.add_layer("Grass")
        self.stack.set_weight(layer.id, 10, 10, 1.5)
        assert self.stack.get_weight(layer.id, 10, 10) == 1.0

    def test_get_weights_at(self):
        """Test getting all weights at position."""
        layer1 = self.stack.add_layer("Grass")
        layer2 = self.stack.add_layer("Rock")

        self.stack.set_weight(layer1.id, 10, 10, 0.6)
        self.stack.set_weight(layer2.id, 10, 10, 0.4)

        weights = self.stack.get_weights_at(10, 10)
        assert layer1.id in weights
        assert layer2.id in weights

    def test_normalize_weights(self):
        """Test weight normalization."""
        layer1 = self.stack.add_layer("Grass")
        layer2 = self.stack.add_layer("Rock")

        self.stack.set_weight(layer1.id, 10, 10, 0.8)
        self.stack.set_weight(layer2.id, 10, 10, 0.8)

        self.stack.normalize_weights(10, 10)

        weights = self.stack.get_weights_at(10, 10)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001


class TestTerrainMaterialManager:
    """Tests for material manager."""

    def setup_method(self):
        """Set up test manager."""
        self.heights = [[0.5 for _ in range(64)] for _ in range(64)]
        self.manager = TerrainMaterialManager(64, 64, self.heights)

    def test_creation(self):
        """Test manager creation."""
        assert self.manager.stack is not None

    def test_create_default_layers(self):
        """Test creating default layers."""
        self.manager.create_default_layers()
        layers = self.manager.stack.get_all_layers()
        assert len(layers) == 4

    def test_sample_material(self):
        """Test material sampling."""
        self.manager.stack.add_layer("Grass")
        self.manager.stack.set_weight(0, 32, 32, 1.0)

        sample = self.manager.sample_material(32, 32)
        assert sample is not None
        assert 0 in sample.layer_weights

    def test_calculate_blend_weights_linear(self):
        """Test linear blend weight calculation."""
        layer = self.manager.stack.add_layer("Grass")
        self.manager.stack.set_weight(layer.id, 32, 32, 0.5)

        weights = self.manager.calculate_blend_weights(32, 32, {layer.id: 0.5})
        assert layer.id in weights

    def test_calculate_blend_weights_height_based(self):
        """Test height-based blend weight calculation."""
        layer = self.manager.stack.add_layer(
            "Snow",
            blend_mode=BlendMode.HEIGHT_BASED,
            height_offset=0.5,
        )
        self.manager.stack.set_weight(layer.id, 32, 32, 1.0)

        weights = self.manager.calculate_blend_weights(32, 32, {layer.id: 1.0})
        assert layer.id in weights

    def test_calculate_blend_weights_slope_based(self):
        """Test slope-based blend weight calculation."""
        layer = self.manager.stack.add_layer(
            "Rock",
            blend_mode=BlendMode.SLOPE_BASED,
        )
        self.manager.stack.set_weight(layer.id, 32, 32, 1.0)

        weights = self.manager.calculate_blend_weights(32, 32, {layer.id: 1.0})
        assert layer.id in weights

    def test_get_shader_data(self):
        """Test shader data generation."""
        self.manager.stack.add_layer("Grass", albedo_texture="grass.png")
        data = self.manager.get_shader_data()

        assert "layer_count" in data
        assert "layers" in data
        assert data["layer_count"] == 1

    def test_get_splatmap_textures(self):
        """Test splatmap texture generation."""
        self.manager.stack.add_layer("Layer1")
        self.manager.stack.add_layer("Layer2")

        textures = self.manager.get_splatmap_textures()
        assert len(textures) >= 1

    def test_auto_paint_by_slope(self):
        """Test auto-painting by slope."""
        flat = self.manager.stack.add_layer("Flat")
        steep = self.manager.stack.add_layer("Steep")

        self.manager.auto_paint_by_slope(flat.id, steep.id, threshold=0.5)

        # All weights should be set

    def test_auto_paint_by_height(self):
        """Test auto-painting by height."""
        layer = self.manager.stack.add_layer("Snow")

        self.manager.auto_paint_by_height(layer.id, 0.4, 0.6, feather=0.05)

        # Some weights should be set based on height
