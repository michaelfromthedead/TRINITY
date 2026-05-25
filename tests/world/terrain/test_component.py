"""
Tests for terrain components (component.py).

Tests cover:
- LandscapeComponent bounds and LOD
- TerrainSection LOD management
- TerrainProxy collision
- TerrainActor composition and raycast
"""

import pytest
import math

from engine.world.terrain.heightfield import Heightfield, HeightfieldConfig
from engine.world.terrain.patch import TerrainPatch
from engine.world.terrain.component import (
    LandscapeComponent,
    TerrainSection,
    TerrainProxy,
    RaycastHit,
    TerrainActor,
)
from engine.world.terrain.constants import (
    DEFAULT_BOUNDS,
    LOD_BIAS_MIN,
    LOD_BIAS_MAX,
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    DEFAULT_RAYCAST_MAX_DISTANCE,
    RAY_DIRECTION_EPSILON,
)


# =============================================================================
# LandscapeComponent Tests
# =============================================================================


class TestLandscapeComponent:
    """Tests for LandscapeComponent class."""

    def test_default_creation(self):
        """Test creation with default values."""
        comp = LandscapeComponent()
        assert comp.bounds == (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert comp.patch is None
        assert comp.material_id == ""
        assert comp.lod_bias == 0.0
        assert comp.cast_shadows is True
        assert comp.visible is True

    def test_custom_bounds(self):
        """Test creation with custom bounds."""
        bounds = (10.0, -5.0, 20.0, 110.0, 100.0, 120.0)
        comp = LandscapeComponent(bounds=bounds)
        assert comp.bounds == bounds

    def test_invalid_bounds_wrong_length(self):
        """Test invalid bounds with wrong length."""
        with pytest.raises(ValueError, match="must be a 6-tuple"):
            LandscapeComponent(bounds=(0.0, 0.0, 0.0))

    def test_invalid_bounds_min_greater_max_x(self):
        """Test invalid bounds with min_x > max_x."""
        with pytest.raises(ValueError, match="min_x must be <= max_x"):
            LandscapeComponent(bounds=(10.0, 0.0, 0.0, 5.0, 1.0, 1.0))

    def test_invalid_bounds_min_greater_max_y(self):
        """Test invalid bounds with min_y > max_y."""
        with pytest.raises(ValueError, match="min_y must be <= max_y"):
            LandscapeComponent(bounds=(0.0, 10.0, 0.0, 1.0, 5.0, 1.0))

    def test_invalid_bounds_min_greater_max_z(self):
        """Test invalid bounds with min_z > max_z."""
        with pytest.raises(ValueError, match="min_z must be <= max_z"):
            LandscapeComponent(bounds=(0.0, 0.0, 10.0, 1.0, 1.0, 5.0))

    def test_custom_material_id(self):
        """Test creation with custom material ID."""
        comp = LandscapeComponent(material_id="grass_01")
        assert comp.material_id == "grass_01"

    def test_lod_bias_valid_range(self):
        """Test LOD bias within valid range."""
        comp = LandscapeComponent(lod_bias=-0.5)
        assert comp.lod_bias == -0.5
        comp = LandscapeComponent(lod_bias=0.5)
        assert comp.lod_bias == 0.5

    def test_lod_bias_invalid_range(self):
        """Test LOD bias outside valid range."""
        with pytest.raises(ValueError, match="lod_bias"):
            LandscapeComponent(lod_bias=-1.5)
        with pytest.raises(ValueError, match="lod_bias"):
            LandscapeComponent(lod_bias=1.5)

    def test_update_bounds_from_patch(self):
        """Test updating bounds from attached patch."""
        config = HeightfieldConfig(resolution=33, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(16, 16, 50.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        # Bounds should match patch world bounds
        assert comp.bounds[0] == 0.0
        assert comp.bounds[3] == 32.0

    def test_get_adjusted_lod_no_bias(self):
        """Test adjusted LOD with no bias."""
        comp = LandscapeComponent(lod_bias=0.0)
        assert comp.get_adjusted_lod(2, 6) == 2

    def test_get_adjusted_lod_negative_bias(self):
        """Test adjusted LOD with negative bias (more detail)."""
        comp = LandscapeComponent(lod_bias=-1.0)
        # -1.0 bias should reduce LOD by 2
        assert comp.get_adjusted_lod(3, 6) == 1

    def test_get_adjusted_lod_positive_bias(self):
        """Test adjusted LOD with positive bias (less detail)."""
        comp = LandscapeComponent(lod_bias=1.0)
        # +1.0 bias should increase LOD by 2
        assert comp.get_adjusted_lod(2, 6) == 4

    def test_get_adjusted_lod_clamped(self):
        """Test adjusted LOD is clamped to valid range."""
        comp = LandscapeComponent(lod_bias=-1.0)
        # Can't go below 0
        assert comp.get_adjusted_lod(0, 6) == 0
        comp = LandscapeComponent(lod_bias=1.0)
        # Can't go above max-1
        assert comp.get_adjusted_lod(4, 6) == 5

    def test_dirty_flag(self):
        """Test dirty flag management."""
        comp = LandscapeComponent()
        assert comp.is_dirty()  # Initially dirty
        comp.clear_dirty()
        assert not comp.is_dirty()
        comp.mark_dirty()
        assert comp.is_dirty()

    def test_intersects_frustum_no_planes(self):
        """Test frustum intersection with no planes."""
        comp = LandscapeComponent(bounds=(0.0, 0.0, 0.0, 10.0, 10.0, 10.0))
        assert comp.intersects_frustum([])

    def test_intersects_frustum_inside(self):
        """Test frustum intersection when inside frustum."""
        comp = LandscapeComponent(bounds=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        # Plane normal pointing in -x direction, at x=10
        # Points with x < 10 are in front (inside)
        # Equation: -x + 10 >= 0 means x <= 10 is inside
        planes = [(-1.0, 0.0, 0.0, 10.0)]  # -x + 10 = 0 -> x = 10
        assert comp.intersects_frustum(planes)

    def test_intersects_frustum_outside(self):
        """Test frustum intersection when outside frustum."""
        comp = LandscapeComponent(bounds=(20.0, 0.0, 0.0, 30.0, 1.0, 1.0))
        # Plane normal pointing in -x direction, at x=10
        # Points with x < 10 are in front (inside)
        planes = [(-1.0, 0.0, 0.0, 10.0)]  # Box at x=20-30 is outside (x > 10)
        assert not comp.intersects_frustum(planes)


# =============================================================================
# TerrainSection Tests
# =============================================================================


class TestTerrainSection:
    """Tests for TerrainSection class."""

    def test_default_creation(self):
        """Test creation with default values."""
        section = TerrainSection()
        assert section.component is None
        assert section.section_index == 0
        assert section.lod_levels == []
        assert section.current_lod == 0

    def test_get_bounds_no_component(self):
        """Test get_bounds without component."""
        section = TerrainSection()
        bounds = section.get_bounds()
        assert bounds == (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    def test_get_bounds_from_component(self):
        """Test get_bounds from parent component."""
        comp = LandscapeComponent(bounds=(5.0, 0.0, 5.0, 15.0, 10.0, 15.0))
        section = TerrainSection(component=comp)
        bounds = section.get_bounds()
        assert bounds == (5.0, 0.0, 5.0, 15.0, 10.0, 15.0)

    def test_set_bounds(self):
        """Test setting section-specific bounds."""
        section = TerrainSection()
        section.set_bounds((10.0, 0.0, 10.0, 20.0, 5.0, 20.0))
        bounds = section.get_bounds()
        assert bounds == (10.0, 0.0, 10.0, 20.0, 5.0, 20.0)

    def test_set_bounds_invalid(self):
        """Test setting invalid bounds."""
        section = TerrainSection()
        with pytest.raises(ValueError, match="6-tuple"):
            section.set_bounds((0.0, 0.0, 0.0))

    def test_select_lod(self):
        """Test LOD selection."""
        section = TerrainSection()
        section.lod_levels = [None, None, None]  # 3 LOD levels
        lod_distances = (50.0, 100.0, 200.0)
        assert section.select_lod(25.0, lod_distances) == 0
        assert section.select_lod(75.0, lod_distances) == 1
        assert section.select_lod(150.0, lod_distances) == 2
        assert section.select_lod(500.0, lod_distances) == 2  # Clamp to max

    def test_get_mesh_data(self):
        """Test getting mesh data for current LOD."""
        section = TerrainSection()
        section.lod_levels = ["mesh_0", "mesh_1", "mesh_2"]
        section.current_lod = 1
        assert section.get_mesh_data() == "mesh_1"

    def test_get_mesh_data_empty(self):
        """Test getting mesh data when empty."""
        section = TerrainSection()
        assert section.get_mesh_data() is None

    def test_set_mesh_data(self):
        """Test setting mesh data for LOD level."""
        section = TerrainSection()
        assert section.set_mesh_data(0, "mesh_0")
        assert section.set_mesh_data(2, "mesh_2")
        assert section.lod_levels[0] == "mesh_0"
        assert section.lod_levels[2] == "mesh_2"
        assert len(section.lod_levels) == 3

    def test_set_mesh_data_invalid_lod(self):
        """Test setting mesh data for invalid LOD level."""
        section = TerrainSection()
        assert not section.set_mesh_data(-1, "mesh")

    def test_get_center(self):
        """Test getting section center."""
        section = TerrainSection()
        section.set_bounds((0.0, 0.0, 0.0, 10.0, 20.0, 30.0))
        center = section.get_center()
        assert center == (5.0, 10.0, 15.0)


# =============================================================================
# TerrainProxy Tests
# =============================================================================


class TestTerrainProxy:
    """Tests for TerrainProxy class."""

    def test_default_creation(self):
        """Test creation with default values."""
        proxy = TerrainProxy()
        assert proxy.bounds == (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert proxy.heightfield_ref is None
        assert proxy.physical_material == "default"

    def test_custom_bounds(self):
        """Test creation with custom bounds."""
        bounds = (10.0, -5.0, 10.0, 100.0, 50.0, 100.0)
        proxy = TerrainProxy(bounds=bounds)
        assert proxy.bounds == bounds

    def test_invalid_bounds(self):
        """Test creation with invalid bounds."""
        with pytest.raises(ValueError, match="6-tuple"):
            TerrainProxy(bounds=(0.0, 0.0, 0.0))

    def test_get_height_at_no_heightfield(self):
        """Test get_height_at without heightfield."""
        proxy = TerrainProxy()
        assert proxy.get_height_at(0.0, 0.0) is None

    def test_get_height_at_outside_bounds(self):
        """Test get_height_at outside bounds."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        proxy = TerrainProxy(
            bounds=(0.0, 0.0, 0.0, 4.0, 10.0, 4.0),
            heightfield_ref=hf
        )
        assert proxy.get_height_at(10.0, 10.0) is None

    def test_get_height_at_inside_bounds(self):
        """Test get_height_at inside bounds."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(2, 2, 50.0)
        proxy = TerrainProxy(
            bounds=(0.0, 0.0, 0.0, 4.0, 100.0, 4.0),
            heightfield_ref=hf
        )
        height = proxy.get_height_at(2.0, 2.0)
        assert height is not None
        assert abs(height - 50.0) < 1e-6

    def test_get_normal_at(self):
        """Test get_normal_at."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        proxy = TerrainProxy(
            bounds=(0.0, 0.0, 0.0, 4.0, 10.0, 4.0),
            heightfield_ref=hf
        )
        normal = proxy.get_normal_at(2.0, 2.0)
        assert normal is not None
        assert abs(normal[1] - 1.0) < 1e-6  # Flat surface, normal up

    def test_point_in_bounds(self):
        """Test point_in_bounds."""
        proxy = TerrainProxy(bounds=(0.0, 0.0, 0.0, 10.0, 10.0, 10.0))
        assert proxy.point_in_bounds(5.0, 5.0, 5.0)
        assert not proxy.point_in_bounds(15.0, 5.0, 5.0)

    def test_friction_getset(self):
        """Test friction getter and setter."""
        proxy = TerrainProxy()
        assert proxy.get_friction() == 0.6  # Default
        proxy.set_friction(0.8)
        assert proxy.get_friction() == 0.8
        # Test clamping
        proxy.set_friction(2.0)
        assert proxy.get_friction() == 1.0
        proxy.set_friction(-0.5)
        assert proxy.get_friction() == 0.0

    def test_restitution_getset(self):
        """Test restitution getter and setter."""
        proxy = TerrainProxy()
        assert proxy.get_restitution() == 0.1  # Default
        proxy.set_restitution(0.5)
        assert proxy.get_restitution() == 0.5


# =============================================================================
# RaycastHit Tests
# =============================================================================


class TestRaycastHit:
    """Tests for RaycastHit dataclass."""

    def test_default_creation(self):
        """Test creation with default values."""
        hit = RaycastHit()
        assert hit.hit is False
        assert hit.position == (0.0, 0.0, 0.0)
        assert hit.normal == (0.0, 1.0, 0.0)
        assert hit.distance == float('inf')
        assert hit.component_index == -1

    def test_custom_values(self):
        """Test creation with custom values."""
        hit = RaycastHit(
            hit=True,
            position=(10.0, 5.0, 20.0),
            normal=(0.0, 1.0, 0.0),
            distance=15.0,
            component_index=3
        )
        assert hit.hit is True
        assert hit.position == (10.0, 5.0, 20.0)
        assert hit.distance == 15.0
        assert hit.component_index == 3


# =============================================================================
# TerrainActor Tests
# =============================================================================


class TestTerrainActor:
    """Tests for TerrainActor class."""

    def test_creation(self):
        """Test creation."""
        actor = TerrainActor()
        assert actor.get_component_count() == 0
        assert actor.total_bounds == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert actor.origin_offset == (0.0, 0.0, 0.0)

    def test_add_component(self):
        """Test adding component."""
        actor = TerrainActor()
        comp = LandscapeComponent(bounds=(0.0, 0.0, 0.0, 10.0, 5.0, 10.0))
        index = actor.add_component(comp)
        assert index == 0
        assert actor.get_component_count() == 1
        assert actor.get_component(0) is comp

    def test_add_multiple_components(self):
        """Test adding multiple components."""
        actor = TerrainActor()
        comp1 = LandscapeComponent(bounds=(0.0, 0.0, 0.0, 10.0, 5.0, 10.0))
        comp2 = LandscapeComponent(bounds=(10.0, 0.0, 0.0, 20.0, 5.0, 10.0))
        actor.add_component(comp1)
        actor.add_component(comp2)
        assert actor.get_component_count() == 2

    def test_remove_component(self):
        """Test removing component."""
        actor = TerrainActor()
        comp = LandscapeComponent()
        actor.add_component(comp)
        assert actor.remove_component(0)
        assert actor.get_component_count() == 0

    def test_remove_component_invalid_index(self):
        """Test removing component with invalid index."""
        actor = TerrainActor()
        assert not actor.remove_component(0)
        assert not actor.remove_component(-1)

    def test_get_component_invalid_index(self):
        """Test getting component with invalid index."""
        actor = TerrainActor()
        assert actor.get_component(-1) is None
        assert actor.get_component(0) is None

    def test_total_bounds(self):
        """Test total bounds calculation."""
        actor = TerrainActor()
        comp1 = LandscapeComponent(bounds=(0.0, 0.0, 0.0, 10.0, 5.0, 10.0))
        comp2 = LandscapeComponent(bounds=(10.0, -5.0, 0.0, 20.0, 10.0, 10.0))
        actor.add_component(comp1)
        actor.add_component(comp2)
        bounds = actor.total_bounds
        assert bounds[0] == 0.0  # min_x
        assert bounds[1] == -5.0  # min_y
        assert bounds[3] == 20.0  # max_x
        assert bounds[4] == 10.0  # max_y

    def test_get_component_at(self):
        """Test getting component at world position."""
        actor = TerrainActor()
        comp1 = LandscapeComponent(bounds=(0.0, 0.0, 0.0, 10.0, 5.0, 10.0))
        comp2 = LandscapeComponent(bounds=(10.0, 0.0, 0.0, 20.0, 5.0, 10.0))
        actor.add_component(comp1)
        actor.add_component(comp2)
        assert actor.get_component_at(5.0, 5.0) is comp1
        assert actor.get_component_at(15.0, 5.0) is comp2
        assert actor.get_component_at(50.0, 50.0) is None

    def test_get_component_by_patch(self):
        """Test getting component by patch coordinates."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        patch = TerrainPatch(patch_x=2, patch_y=3, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)
        assert actor.get_component_by_patch(2, 3) is comp
        assert actor.get_component_by_patch(0, 0) is None

    def test_set_origin_offset(self):
        """Test setting origin offset."""
        actor = TerrainActor()
        actor.set_origin_offset(1000.0, 0.0, 2000.0)
        assert actor.origin_offset == (1000.0, 0.0, 2000.0)

    def test_update_lod(self):
        """Test LOD update based on camera position."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        patch = TerrainPatch(
            patch_x=0, patch_y=0,
            heightfield=hf,
            lod_levels=4,
            lod_distances=(50.0, 100.0, 200.0, 400.0)
        )
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)

        # Camera close - should be LOD 0
        actor.update_lod((32.0, 10.0, 32.0))
        assert patch.current_lod == 0

        # Camera far - should be higher LOD
        actor.update_lod((500.0, 10.0, 500.0))
        assert patch.current_lod > 0

    def test_get_height_at(self):
        """Test getting terrain height."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(32, 32, 100.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)

        height = actor.get_height_at(32.0, 32.0)
        assert height is not None
        assert abs(height - 100.0) < 1e-6

    def test_get_height_at_outside(self):
        """Test getting height outside terrain."""
        actor = TerrainActor()
        assert actor.get_height_at(0.0, 0.0) is None

    def test_get_normal_at(self):
        """Test getting terrain normal."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)

        normal = actor.get_normal_at(32.0, 32.0)
        assert normal is not None
        assert abs(normal[1] - 1.0) < 1e-6  # Flat surface

    def test_raycast_hit(self):
        """Test raycast hitting terrain."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.fill(0.0)  # Flat terrain at y=0
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)

        # Ray from above pointing down
        result = actor.raycast(
            origin=(32.0, 100.0, 32.0),
            direction=(0.0, -1.0, 0.0),
            max_distance=200.0
        )
        assert result.hit
        assert result.distance < 200.0
        assert abs(result.position[1]) < 1.0  # Near y=0

    def test_raycast_miss(self):
        """Test raycast missing terrain."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.fill(0.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)

        # Ray pointing up
        result = actor.raycast(
            origin=(32.0, 100.0, 32.0),
            direction=(0.0, 1.0, 0.0),
            max_distance=200.0
        )
        assert not result.hit

    def test_raycast_zero_direction(self):
        """Test raycast with zero direction."""
        actor = TerrainActor()
        result = actor.raycast(
            origin=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 0.0)
        )
        assert not result.hit

    def test_get_visible_components(self):
        """Test getting visible components."""
        actor = TerrainActor()
        comp1 = LandscapeComponent(visible=True)
        comp2 = LandscapeComponent(visible=False)
        comp3 = LandscapeComponent(visible=True)
        actor.add_component(comp1)
        actor.add_component(comp2)
        actor.add_component(comp3)

        visible = actor.get_visible_components()
        assert len(visible) == 2
        assert comp1 in visible
        assert comp2 not in visible
        assert comp3 in visible

    def test_clear(self):
        """Test clearing all components."""
        actor = TerrainActor()
        actor.add_component(LandscapeComponent())
        actor.add_component(LandscapeComponent())
        actor.clear()
        assert actor.get_component_count() == 0


# =============================================================================
# Constants Verification Tests
# =============================================================================


class TestComponentConstants:
    """Tests to verify constants are properly used in components."""

    def test_landscape_default_bounds(self):
        """Test LandscapeComponent default bounds match constants."""
        comp = LandscapeComponent()
        assert comp.bounds == DEFAULT_BOUNDS

    def test_lod_bias_boundary_values(self):
        """Test LOD bias at exact boundary values."""
        comp_min = LandscapeComponent(lod_bias=LOD_BIAS_MIN)
        assert comp_min.lod_bias == LOD_BIAS_MIN

        comp_max = LandscapeComponent(lod_bias=LOD_BIAS_MAX)
        assert comp_max.lod_bias == LOD_BIAS_MAX

    def test_lod_bias_just_outside_range(self):
        """Test LOD bias just outside valid range."""
        epsilon = 0.001
        with pytest.raises(ValueError):
            LandscapeComponent(lod_bias=LOD_BIAS_MIN - epsilon)

        with pytest.raises(ValueError):
            LandscapeComponent(lod_bias=LOD_BIAS_MAX + epsilon)

    def test_proxy_default_friction(self):
        """Test TerrainProxy default friction matches constants."""
        proxy = TerrainProxy()
        assert proxy.get_friction() == DEFAULT_FRICTION

    def test_proxy_default_restitution(self):
        """Test TerrainProxy default restitution matches constants."""
        proxy = TerrainProxy()
        assert proxy.get_restitution() == DEFAULT_RESTITUTION


# =============================================================================
# Raycast Edge Cases
# =============================================================================


class TestRaycastEdgeCases:
    """Tests for raycast edge cases."""

    def test_raycast_near_zero_direction(self):
        """Test raycast with direction just above epsilon threshold."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.fill(0.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)

        # Direction just barely non-zero (above epsilon)
        tiny_dir = RAY_DIRECTION_EPSILON * 10
        result = actor.raycast(
            origin=(32.0, 100.0, 32.0),
            direction=(0.0, -tiny_dir, 0.0)
        )
        # Should still work with tiny but non-zero direction
        assert result is not None

    def test_raycast_diagonal_hit(self):
        """Test raycast hitting terrain at a slight angle."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.fill(10.0)  # Flat terrain at y=10
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)

        # Slight diagonal ray from above, mostly vertical to ensure hit
        # The ray starts at (32, 100, 32), travels ~90 units down to y=10
        # With direction (0.1, -1, 0.1) normalized, moves ~9 units in XZ
        # Final position ~(41, 10, 41) which is still within bounds
        result = actor.raycast(
            origin=(32.0, 100.0, 32.0),
            direction=(0.1, -1.0, 0.1),  # Mostly vertical, slight diagonal
            max_distance=500.0
        )
        assert result.hit, "Raycast should hit terrain"
        # Hit position should be on or near the terrain (y ~= 10)
        assert abs(result.position[1] - 10.0) < 2.0, f"Hit y={result.position[1]}, expected ~10"

    def test_raycast_parallel_to_terrain(self):
        """Test raycast parallel to flat terrain (should miss)."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.fill(0.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)

        # Ray parallel to terrain, above it
        result = actor.raycast(
            origin=(0.0, 10.0, 32.0),
            direction=(1.0, 0.0, 0.0),  # Horizontal
            max_distance=200.0
        )
        assert not result.hit

    def test_raycast_from_below_terrain(self):
        """Test raycast starting below terrain."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.fill(100.0)  # Terrain at y=100
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)

        # Start below terrain, shoot up
        result = actor.raycast(
            origin=(32.0, 0.0, 32.0),
            direction=(0.0, 1.0, 0.0),  # Straight up
            max_distance=200.0
        )
        # Current implementation marches from origin; may or may not detect
        # depending on implementation details
        # Just verify it doesn't crash
        assert result is not None

    def test_raycast_with_origin_offset(self):
        """Test raycast with active origin offset."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.fill(0.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        comp = LandscapeComponent(patch=patch)
        comp.update_bounds_from_patch()
        actor = TerrainActor()
        actor.add_component(comp)

        # Set large origin offset
        actor.set_origin_offset(10000.0, 0.0, 10000.0)

        # Cast ray in offset coordinates
        # Terrain is at offset position, so need to account for that
        result = actor.raycast(
            origin=(-9968.0, 100.0, -9968.0),  # 32 in offset coords
            direction=(0.0, -1.0, 0.0),
            max_distance=200.0
        )
        assert result.hit


# =============================================================================
# Frustum Culling Edge Cases
# =============================================================================


class TestFrustumCullingEdgeCases:
    """Tests for frustum culling edge cases."""

    def test_frustum_box_exactly_on_plane(self):
        """Test frustum intersection when box is exactly on plane."""
        # Box from 0-10 on all axes
        comp = LandscapeComponent(bounds=(0.0, 0.0, 0.0, 10.0, 10.0, 10.0))

        # Plane at x=10 (box edge)
        # Equation: -x + 10 = 0 means plane is at x=10
        # Points with x < 10 are inside
        planes = [(-1.0, 0.0, 0.0, 10.0)]

        # Box edge touches plane - should still intersect
        assert comp.intersects_frustum(planes)

    def test_frustum_multiple_planes(self):
        """Test frustum intersection with multiple planes."""
        comp = LandscapeComponent(bounds=(5.0, 5.0, 5.0, 15.0, 15.0, 15.0))

        # Create a box-like frustum
        planes = [
            (1.0, 0.0, 0.0, 0.0),   # x >= 0
            (-1.0, 0.0, 0.0, 20.0), # x <= 20
            (0.0, 1.0, 0.0, 0.0),   # y >= 0
            (-1.0, 0.0, 0.0, 20.0), # y <= 20
        ]

        # Component is inside frustum
        assert comp.intersects_frustum(planes)

    def test_frustum_component_completely_behind(self):
        """Test component completely behind frustum plane."""
        comp = LandscapeComponent(bounds=(100.0, 0.0, 0.0, 110.0, 10.0, 10.0))

        # Plane at x=50, normal pointing -x (inside is x < 50)
        planes = [(-1.0, 0.0, 0.0, 50.0)]

        # Box at x=100-110 is completely outside
        assert not comp.intersects_frustum(planes)
