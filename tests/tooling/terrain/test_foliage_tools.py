"""Tests for foliage placement tools."""

import pytest
import math
from engine.tooling.terrain.foliage_tools import (
    FoliageType,
    FoliageLODLevel,
    FoliageTransform,
    FoliageInstance,
    FoliageLODSettings,
    FoliageLayerSettings,
    FoliageLayer,
    DensityBrushSettings,
    FoliageDensityBrush,
    FoliagePlacementTool,
)


class TestFoliageTransform:
    """Tests for foliage transform."""

    def test_default_values(self):
        """Test default transform values."""
        transform = FoliageTransform()
        assert transform.position_x == 0.0
        assert transform.position_y == 0.0
        assert transform.position_z == 0.0
        assert transform.rotation_y == 0.0
        assert transform.scale == 1.0

    def test_custom_values(self):
        """Test custom transform values."""
        transform = FoliageTransform(
            position_x=10.0,
            position_y=5.0,
            position_z=20.0,
            rotation_y=math.pi,
            scale=2.0,
        )
        assert transform.position_x == 10.0
        assert transform.scale == 2.0


class TestFoliageInstance:
    """Tests for foliage instance."""

    def test_creation(self):
        """Test instance creation."""
        instance = FoliageInstance(id=0, layer_id=0)
        assert instance.id == 0
        assert instance.layer_id == 0
        assert instance.health == 1.0

    def test_distance_to(self):
        """Test distance calculation."""
        instance = FoliageInstance(id=0, layer_id=0)
        instance.transform.position_x = 10.0
        instance.transform.position_y = 0.0
        instance.transform.position_z = 0.0

        dist = instance.distance_to(0.0, 0.0, 0.0)
        assert dist == 10.0

    def test_distance_3d(self):
        """Test 3D distance calculation."""
        instance = FoliageInstance(id=0, layer_id=0)
        instance.transform.position_x = 3.0
        instance.transform.position_y = 4.0
        instance.transform.position_z = 0.0

        dist = instance.distance_to(0.0, 0.0, 0.0)
        assert dist == 5.0


class TestFoliageLODSettings:
    """Tests for LOD settings."""

    def test_default_distances(self):
        """Test default LOD distances."""
        settings = FoliageLODSettings()
        assert settings.lod0_distance == 50.0
        assert settings.lod1_distance == 100.0
        assert settings.cull_distance == 800.0

    def test_get_lod_for_distance(self):
        """Test LOD level selection."""
        settings = FoliageLODSettings()
        # Default distances: lod0=50, lod1=100, lod2=200, lod3=400, cull=800
        # Implementation uses >= so distances at or above threshold get that LOD

        assert settings.get_lod_for_distance(25.0) == FoliageLODLevel.LOD0   # < 50
        assert settings.get_lod_for_distance(75.0) == FoliageLODLevel.LOD0   # >= 50 but < 100
        assert settings.get_lod_for_distance(100.0) == FoliageLODLevel.LOD1  # >= 100
        assert settings.get_lod_for_distance(150.0) == FoliageLODLevel.LOD1  # >= 100 but < 200
        assert settings.get_lod_for_distance(200.0) == FoliageLODLevel.LOD2  # >= 200
        assert settings.get_lod_for_distance(400.0) == FoliageLODLevel.LOD3  # >= 400
        assert settings.get_lod_for_distance(1000.0) == FoliageLODLevel.CULLED  # >= 800


class TestFoliageLayer:
    """Tests for foliage layer."""

    def test_creation(self):
        """Test layer creation."""
        layer = FoliageLayer(id=0, name="Grass")
        assert layer.id == 0
        assert layer.name == "Grass"
        assert layer.instance_count == 0

    def test_add_instance(self):
        """Test adding instances."""
        layer = FoliageLayer(id=0, name="Grass")
        transform = FoliageTransform(position_x=10.0, position_z=20.0)

        instance = layer.add_instance(transform)
        assert instance.id == 0
        assert instance.layer_id == 0
        assert layer.instance_count == 1

    def test_remove_instance(self):
        """Test removing instances."""
        layer = FoliageLayer(id=0, name="Grass")
        instance = layer.add_instance(FoliageTransform())

        assert layer.remove_instance(instance.id)
        assert layer.instance_count == 0

    def test_get_instance(self):
        """Test getting instance by ID."""
        layer = FoliageLayer(id=0, name="Grass")
        instance = layer.add_instance(FoliageTransform())

        retrieved = layer.get_instance(instance.id)
        assert retrieved == instance

    def test_get_instances_in_radius(self):
        """Test getting instances in radius."""
        layer = FoliageLayer(id=0, name="Grass")

        # Add instances at known positions
        layer.add_instance(FoliageTransform(position_x=0.0, position_z=0.0))
        layer.add_instance(FoliageTransform(position_x=5.0, position_z=0.0))
        layer.add_instance(FoliageTransform(position_x=20.0, position_z=0.0))

        near = layer.get_instances_in_radius(0.0, 0.0, 0.0, 10.0)
        assert len(near) == 2

    def test_clear_instances(self):
        """Test clearing all instances."""
        layer = FoliageLayer(id=0, name="Grass")
        layer.add_instance(FoliageTransform())
        layer.add_instance(FoliageTransform())

        layer.clear_instances()
        assert layer.instance_count == 0


class TestFoliageDensityBrush:
    """Tests for density brush."""

    def test_default_settings(self):
        """Test default brush settings."""
        brush = FoliageDensityBrush()
        assert brush.settings.size == 10.0
        assert brush.settings.strength == 0.5

    def test_falloff(self):
        """Test brush falloff."""
        brush = FoliageDensityBrush()
        assert brush.get_falloff(0.0, 5.0) == 1.0
        assert brush.get_falloff(5.0, 5.0) == 0.0

    def test_influence(self):
        """Test brush influence."""
        brush = FoliageDensityBrush()
        influence = brush.get_influence(5.0, 5.0, 5.0, 5.0)
        assert influence == brush.settings.strength


class TestFoliagePlacementTool:
    """Tests for foliage placement tool."""

    def setup_method(self):
        """Set up test terrain."""
        self.heights = [[0.0 for _ in range(64)] for _ in range(64)]
        self.tool = FoliagePlacementTool(64, 64, self.heights)

    def test_creation(self):
        """Test tool creation."""
        assert self.tool is not None

    def test_add_layer(self):
        """Test adding layers."""
        layer = self.tool.add_layer("Grass", "mesh_grass")
        assert layer.name == "Grass"
        assert self.tool.current_layer_id == layer.id

    def test_remove_layer(self):
        """Test removing layers."""
        layer = self.tool.add_layer("Grass", "mesh_grass")
        assert self.tool.remove_layer(layer.id)
        assert self.tool.get_layer(layer.id) is None

    def test_get_all_layers(self):
        """Test getting all layers."""
        self.tool.add_layer("Grass", "mesh_grass")
        self.tool.add_layer("Trees", "mesh_tree")
        layers = self.tool.get_all_layers()
        assert len(layers) == 2

    def test_paint_density_add(self):
        """Test painting density (add)."""
        layer = self.tool.add_layer("Grass", "mesh_grass")
        layer.settings.density = 10.0

        count = self.tool.paint_density(32.0, 32.0, add=True)
        assert count > 0
        assert layer.instance_count > 0

    def test_paint_density_remove(self):
        """Test painting density (remove)."""
        layer = self.tool.add_layer("Grass", "mesh_grass")

        # Add some instances first
        for i in range(10):
            self.tool.place_instance(layer.id, 32.0 + i * 0.5, 32.0)

        initial_count = layer.instance_count
        self.tool.paint_density(32.0, 32.0, add=False)

        # Some instances should be removed
        assert layer.instance_count < initial_count

    def test_place_instance(self):
        """Test placing single instance."""
        layer = self.tool.add_layer("Tree", "mesh_tree")
        instance = self.tool.place_instance(layer.id, 10.0, 20.0)

        assert instance is not None
        assert instance.transform.position_x == 10.0
        assert instance.transform.position_z == 20.0

    def test_remove_instances_in_radius(self):
        """Test removing instances in radius."""
        layer = self.tool.add_layer("Grass", "mesh_grass")

        # Add instances
        self.tool.place_instance(layer.id, 10.0, 10.0)
        self.tool.place_instance(layer.id, 11.0, 10.0)
        self.tool.place_instance(layer.id, 100.0, 100.0)

        removed = self.tool.remove_instances_in_radius(10.0, 0.0, 10.0, 5.0)
        assert removed == 2

    def test_fill_area(self):
        """Test filling area with foliage."""
        layer = self.tool.add_layer("Grass", "mesh_grass")
        layer.settings.density = 1.0

        count = self.tool.fill_area(layer.id, 0.0, 0.0, 10.0, 10.0)
        assert count > 0

    def test_update_lod(self):
        """Test LOD updates."""
        layer = self.tool.add_layer("Grass", "mesh_grass")

        # Add instances at various distances
        self.tool.place_instance(layer.id, 10.0, 10.0)
        self.tool.place_instance(layer.id, 200.0, 200.0)

        lod_counts = self.tool.update_lod(0.0, 0.0, 0.0)

        assert FoliageLODLevel.LOD0 in lod_counts
        assert sum(lod_counts.values()) == 2

    def test_get_visible_instances(self):
        """Test getting visible instances."""
        layer = self.tool.add_layer("Grass", "mesh_grass")

        self.tool.place_instance(layer.id, 10.0, 10.0)
        self.tool.place_instance(layer.id, 1000.0, 1000.0)

        visible = self.tool.get_visible_instances(0.0, 0.0, 0.0, max_distance=100.0)
        assert len(visible) == 1

    def test_get_total_instance_count(self):
        """Test total instance count."""
        layer1 = self.tool.add_layer("Grass", "mesh_grass")
        layer2 = self.tool.add_layer("Trees", "mesh_tree")

        self.tool.place_instance(layer1.id, 10.0, 10.0)
        self.tool.place_instance(layer1.id, 20.0, 20.0)
        self.tool.place_instance(layer2.id, 30.0, 30.0)

        assert self.tool.get_total_instance_count() == 3

    def test_clear_all(self):
        """Test clearing all foliage."""
        layer1 = self.tool.add_layer("Grass", "mesh_grass")
        layer2 = self.tool.add_layer("Trees", "mesh_tree")

        self.tool.place_instance(layer1.id, 10.0, 10.0)
        self.tool.place_instance(layer2.id, 20.0, 20.0)

        self.tool.clear_all()
        assert self.tool.get_total_instance_count() == 0

    def test_seed_consistency(self):
        """Test seed produces consistent results."""
        layer = self.tool.add_layer("Grass", "mesh_grass")
        layer.settings.density = 5.0

        self.tool.set_seed(42)
        self.tool.fill_area(layer.id, 0.0, 0.0, 10.0, 10.0)
        count1 = layer.instance_count

        layer.clear_instances()
        self.tool.set_seed(42)
        self.tool.fill_area(layer.id, 0.0, 0.0, 10.0, 10.0)
        count2 = layer.instance_count

        assert count1 == count2
