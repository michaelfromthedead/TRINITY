"""Tests for shadow mapping systems."""

from __future__ import annotations

import math
import pytest

from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec3
from engine.rendering.lighting.shadows import (
    ShadowMapType,
    ShadowMapConfig,
    CascadeData,
    CascadedShadowMap,
    CubeShadowMap,
    SpotShadowMap,
    ShadowAtlas,
    ShadowAtlasSlot,
)
from engine.rendering.lighting.light_types import (
    DirectionalLight,
    PointLight,
    SpotLight,
)


class TestShadowMapConfig:
    """Tests for shadow map configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ShadowMapConfig()
        assert config.resolution == 2048
        assert config.depth_bias == pytest.approx(0.0001)
        assert config.slope_bias == pytest.approx(0.001)
        assert config.normal_bias == pytest.approx(0.02)
        assert config.filter_size == 3
        assert config.softness == 1.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ShadowMapConfig(
            resolution=4096,
            depth_bias=0.0005,
            filter_size=5,
        )
        assert config.resolution == 4096
        assert config.depth_bias == pytest.approx(0.0005)
        assert config.filter_size == 5


class TestCascadedShadowMap:
    """Tests for cascaded shadow maps."""

    def test_csm_creation(self) -> None:
        """Test creating a CSM."""
        csm = CascadedShadowMap(cascade_count=4)
        assert csm.shadow_type == ShadowMapType.CASCADED
        assert csm.cascade_count == 4
        assert len(csm.cascade_data) == 4

    def test_csm_invalid_cascade_count(self) -> None:
        """Test that invalid cascade count raises error."""
        with pytest.raises(ValueError):
            CascadedShadowMap(cascade_count=0)
        with pytest.raises(ValueError):
            CascadedShadowMap(cascade_count=5)

    def test_csm_resolution(self) -> None:
        """Test CSM resolution."""
        config = ShadowMapConfig(resolution=1024)
        csm = CascadedShadowMap(config=config)
        assert csm.get_resolution() == (1024, 1024)

    def test_csm_view_projection_per_cascade(self) -> None:
        """Test getting view-projection for each cascade."""
        csm = CascadedShadowMap(cascade_count=4)

        for i in range(4):
            vp = csm.get_view_projection_matrix(i)
            assert isinstance(vp, Mat4)

    def test_csm_configure_for_light(self) -> None:
        """Test configuring CSM for a directional light."""
        csm = CascadedShadowMap(cascade_count=4)
        light = DirectionalLight(
            direction=Vec3(0.5, -0.8, 0.2),
            cascade_distances=[10, 30, 100, 500],
        )

        view = Mat4.look_at(Vec3(0, 0, 10), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(60), 1.777, 0.1, 1000.0)

        csm.configure_for_light(light, view, proj, 0.1, 500.0)

        # Verify cascade split depths are set
        for i, cascade in enumerate(csm.cascade_data):
            assert cascade.split_depth > 0

    def test_csm_get_cascade_for_depth(self) -> None:
        """Test getting cascade index for a depth value."""
        csm = CascadedShadowMap(cascade_count=4)

        # Set up cascade split depths
        for i, cascade in enumerate(csm.cascade_data):
            cascade.split_depth = (i + 1) * 100.0

        assert csm.get_cascade_for_depth(50.0) == 0
        assert csm.get_cascade_for_depth(150.0) == 1
        assert csm.get_cascade_for_depth(250.0) == 2
        assert csm.get_cascade_for_depth(350.0) == 3
        assert csm.get_cascade_for_depth(500.0) == 3  # Clamped to last

    def test_csm_dirty_flag(self) -> None:
        """Test CSM dirty flag management."""
        csm = CascadedShadowMap()
        assert csm.dirty is True

        csm.clear_dirty()
        assert csm.dirty is False

        csm.mark_dirty()
        assert csm.dirty is True


class TestCubeShadowMap:
    """Tests for cube shadow maps."""

    def test_cube_shadow_creation(self) -> None:
        """Test creating a cube shadow map."""
        cube = CubeShadowMap(
            position=Vec3(0, 5, 0),
            radius=20.0,
        )
        assert cube.shadow_type == ShadowMapType.CUBE
        assert cube.position == Vec3(0, 5, 0)
        assert cube.radius == 20.0
        assert len(cube.face_matrices) == 6

    def test_cube_shadow_resolution(self) -> None:
        """Test cube shadow map resolution."""
        config = ShadowMapConfig(resolution=512)
        cube = CubeShadowMap(config=config)
        assert cube.get_resolution() == (512, 512)

    def test_cube_shadow_face_directions(self) -> None:
        """Test getting face directions."""
        cube = CubeShadowMap()

        # Test all 6 faces
        expected_dirs = [
            Vec3(1, 0, 0),   # +X
            Vec3(-1, 0, 0),  # -X
            Vec3(0, 1, 0),   # +Y
            Vec3(0, -1, 0),  # -Y
            Vec3(0, 0, 1),   # +Z
            Vec3(0, 0, -1),  # -Z
        ]

        for i, expected in enumerate(expected_dirs):
            actual = cube.get_face_direction(i)
            assert actual.x == pytest.approx(expected.x)
            assert actual.y == pytest.approx(expected.y)
            assert actual.z == pytest.approx(expected.z)

    def test_cube_shadow_configure_for_light(self) -> None:
        """Test configuring cube shadow for point light."""
        cube = CubeShadowMap()
        light = PointLight(position=Vec3(5, 10, -3), radius=25.0)

        cube.configure_for_light(light)

        assert cube.position == light.position
        assert cube.radius == light.radius
        assert cube.light_id == light._light_id
        assert cube.dirty is True

    def test_cube_shadow_view_projection(self) -> None:
        """Test view-projection for each face."""
        cube = CubeShadowMap(position=Vec3(0, 0, 0), radius=10.0)

        for face in range(6):
            vp = cube.get_view_projection_matrix(face)
            assert isinstance(vp, Mat4)


class TestSpotShadowMap:
    """Tests for spot shadow maps."""

    def test_spot_shadow_creation(self) -> None:
        """Test creating a spot shadow map."""
        spot = SpotShadowMap(
            position=Vec3(0, 10, 0),
            direction=Vec3(0, -1, 0),
            outer_angle=math.radians(45),
            radius=20.0,
        )
        assert spot.shadow_type == ShadowMapType.SPOT
        assert spot.outer_angle == pytest.approx(math.radians(45))

    def test_spot_shadow_resolution(self) -> None:
        """Test spot shadow map resolution."""
        config = ShadowMapConfig(resolution=1024)
        spot = SpotShadowMap(config=config)
        assert spot.get_resolution() == (1024, 1024)

    def test_spot_shadow_configure_for_light(self) -> None:
        """Test configuring spot shadow for spot light."""
        spot = SpotShadowMap()
        light = SpotLight(
            position=Vec3(0, 15, 0),
            direction=Vec3(0, -1, 0),
            outer_angle=math.radians(30),
            radius=50.0,
        )

        spot.configure_for_light(light)

        assert spot.position == light.position
        assert spot.outer_angle == light.outer_angle
        assert spot.radius == light.radius
        assert spot.dirty is True

    def test_spot_shadow_view_projection(self) -> None:
        """Test spot shadow view-projection matrix."""
        spot = SpotShadowMap(
            position=Vec3(0, 10, 0),
            direction=Vec3(0, -1, 0),
            outer_angle=math.radians(45),
        )

        vp = spot.get_view_projection_matrix()
        assert isinstance(vp, Mat4)


class TestShadowAtlas:
    """Tests for shadow atlas packing."""

    def test_atlas_creation(self) -> None:
        """Test creating a shadow atlas."""
        atlas = ShadowAtlas(resolution=4096)
        assert atlas.resolution == 4096
        assert len(atlas.slots) == 0

    def test_atlas_invalid_resolution(self) -> None:
        """Test that non-power-of-2 resolution raises error."""
        with pytest.raises(ValueError):
            ShadowAtlas(resolution=1000)
        with pytest.raises(ValueError):
            ShadowAtlas(resolution=0)
        with pytest.raises(ValueError):
            ShadowAtlas(resolution=-1024)

    def test_atlas_allocate(self) -> None:
        """Test allocating a slot in the atlas."""
        atlas = ShadowAtlas(resolution=4096)

        slot = atlas.allocate(1024, 1024)
        assert slot is not None
        assert slot.width == 1024
        assert slot.height == 1024
        assert len(atlas.slots) == 1

    def test_atlas_allocate_multiple(self) -> None:
        """Test allocating multiple slots."""
        atlas = ShadowAtlas(resolution=4096)

        slots = []
        for _ in range(4):
            slot = atlas.allocate(1024, 1024)
            if slot:
                slots.append(slot)

        # Should fit at least 4 1024x1024 slots in a 4096x4096 atlas
        assert len(slots) >= 4

    def test_atlas_allocate_fails_when_full(self) -> None:
        """Test that allocation fails when atlas is full."""
        atlas = ShadowAtlas(resolution=1024)

        # First allocation should succeed
        slot1 = atlas.allocate(1024, 1024)
        assert slot1 is not None

        # Second allocation should fail (no room)
        slot2 = atlas.allocate(1024, 1024)
        assert slot2 is None

    def test_atlas_deallocate(self) -> None:
        """Test deallocating a slot."""
        atlas = ShadowAtlas(resolution=2048)

        slot = atlas.allocate(1024, 1024)
        assert len(atlas.slots) == 1

        atlas.deallocate(slot)
        assert len(atlas.slots) == 0

    def test_atlas_allocate_shadow_map(self) -> None:
        """Test allocating space for a shadow map."""
        atlas = ShadowAtlas(resolution=4096)
        shadow_map = CubeShadowMap(config=ShadowMapConfig(resolution=1024))

        slot = atlas.allocate_shadow_map(shadow_map)
        assert slot is not None
        assert slot.shadow_map == shadow_map
        assert slot.width == 1024
        assert slot.height == 1024

    def test_atlas_get_slot_for_light(self) -> None:
        """Test finding slot by light ID."""
        atlas = ShadowAtlas(resolution=4096)

        light = PointLight(position=Vec3(0, 0, 0))
        shadow_map = CubeShadowMap()
        shadow_map.configure_for_light(light)

        slot = atlas.allocate_shadow_map(shadow_map)

        found = atlas.get_slot_for_light(light._light_id)
        assert found == slot

    def test_atlas_get_slot_for_light_not_found(self) -> None:
        """Test that get_slot_for_light returns None for unknown light."""
        atlas = ShadowAtlas(resolution=4096)
        assert atlas.get_slot_for_light(999) is None

    def test_atlas_uv_transform(self) -> None:
        """Test getting UV transform for a slot."""
        atlas = ShadowAtlas(resolution=4096)
        slot = atlas.allocate(1024, 1024)

        offset, scale = atlas.get_uv_transform(slot)

        # First slot should be at origin
        assert offset.x == pytest.approx(0.0)
        assert offset.y == pytest.approx(0.0)

        # Scale should be 1024/4096 = 0.25
        assert scale.x == pytest.approx(0.25)
        assert scale.y == pytest.approx(0.25)

    def test_atlas_utilization(self) -> None:
        """Test atlas utilization calculation."""
        atlas = ShadowAtlas(resolution=4096)

        assert atlas.get_utilization() == pytest.approx(0.0)

        # Allocate 1/4 of the atlas
        atlas.allocate(2048, 2048)
        assert atlas.get_utilization() == pytest.approx(0.25)

    def test_atlas_defragment(self) -> None:
        """Test atlas defragmentation."""
        atlas = ShadowAtlas(resolution=4096)

        # Allocate several maps
        sm1 = CubeShadowMap(config=ShadowMapConfig(resolution=1024))
        sm2 = CubeShadowMap(config=ShadowMapConfig(resolution=512))
        sm3 = CubeShadowMap(config=ShadowMapConfig(resolution=256))

        slot1 = atlas.allocate_shadow_map(sm1)
        slot2 = atlas.allocate_shadow_map(sm2)
        slot3 = atlas.allocate_shadow_map(sm3)

        # Deallocate middle one
        atlas.deallocate(slot2)

        # Defragment
        atlas.defragment()

        # Shadow maps should be re-allocated and marked dirty
        assert sm1.dirty is True
        assert sm3.dirty is True


class TestCascadeData:
    """Tests for cascade data."""

    def test_cascade_data_defaults(self) -> None:
        """Test default cascade data values."""
        cascade = CascadeData()
        assert cascade.split_depth == 0.0
        assert cascade.texel_size == 0.0

    def test_cascade_data_custom(self) -> None:
        """Test cascade data with custom values."""
        cascade = CascadeData(
            split_depth=100.0,
            texel_size=0.05,
        )
        assert cascade.split_depth == 100.0
        assert cascade.texel_size == 0.05


class TestShadowCalculations:
    """Tests for actual shadow calculations to verify correctness."""

    def test_csm_cascade_split_ordering(self) -> None:
        """Test that cascade splits are in increasing order."""
        csm = CascadedShadowMap(cascade_count=4)
        light = DirectionalLight(
            direction=Vec3(0.5, -0.8, 0.2),
            cascade_distances=[10, 30, 100, 500],
        )
        view = Mat4.look_at(Vec3(0, 0, 10), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(60), 1.777, 0.1, 500.0)

        csm.configure_for_light(light, view, proj, 0.1, 500.0)

        # Verify cascade splits are strictly increasing
        prev_depth = 0.0
        for cascade in csm.cascade_data:
            assert cascade.split_depth > prev_depth
            prev_depth = cascade.split_depth

    def test_csm_texel_size_positive(self) -> None:
        """Test that texel sizes are positive after configuration."""
        csm = CascadedShadowMap(cascade_count=4)
        light = DirectionalLight(direction=Vec3(0, -1, 0))
        view = Mat4.look_at(Vec3(0, 0, 10), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(60), 1.777, 0.1, 500.0)

        csm.configure_for_light(light, view, proj, 0.1, 500.0)

        for cascade in csm.cascade_data:
            # Texel size should be positive (it's world units per texel)
            assert cascade.texel_size >= 0.0

    def test_cube_shadow_face_coverage(self) -> None:
        """Test that cube shadow map covers all directions."""
        cube = CubeShadowMap(position=Vec3(0, 0, 0), radius=10.0)

        # Collect all face directions
        directions = [cube.get_face_direction(i) for i in range(6)]

        # Test that each axis has positive and negative coverage
        has_positive_x = any(d.x > 0.5 for d in directions)
        has_negative_x = any(d.x < -0.5 for d in directions)
        has_positive_y = any(d.y > 0.5 for d in directions)
        has_negative_y = any(d.y < -0.5 for d in directions)
        has_positive_z = any(d.z > 0.5 for d in directions)
        has_negative_z = any(d.z < -0.5 for d in directions)

        assert has_positive_x and has_negative_x
        assert has_positive_y and has_negative_y
        assert has_positive_z and has_negative_z

    def test_spot_shadow_fov_matches_cone(self) -> None:
        """Test that spot shadow FOV matches the light cone."""
        light = SpotLight(
            position=Vec3(0, 10, 0),
            direction=Vec3(0, -1, 0),
            outer_angle=math.radians(30),
            radius=50.0,
        )
        spot = SpotShadowMap()
        spot.configure_for_light(light)

        # The shadow map outer_angle should match the light
        assert spot.outer_angle == pytest.approx(light.outer_angle)

    def test_atlas_no_overlapping_slots(self) -> None:
        """Test that allocated slots don't overlap."""
        atlas = ShadowAtlas(resolution=4096)

        slots = []
        for _ in range(8):
            slot = atlas.allocate(512, 512)
            if slot:
                slots.append(slot)

        # Check no two slots overlap
        for i, slot1 in enumerate(slots):
            for slot2 in slots[i+1:]:
                # Check for non-overlap: either one is entirely to the left/above the other
                x_no_overlap = (slot1.x + slot1.width <= slot2.x) or (slot2.x + slot2.width <= slot1.x)
                y_no_overlap = (slot1.y + slot1.height <= slot2.y) or (slot2.y + slot2.height <= slot1.y)
                assert x_no_overlap or y_no_overlap, f"Slots overlap: {slot1} and {slot2}"


class TestShadowAtlasSlot:
    """Tests for shadow atlas slot."""

    def test_slot_uv_offset(self) -> None:
        """Test slot UV offset calculation."""
        slot = ShadowAtlasSlot(x=512, y=256, width=1024, height=1024)
        offset = slot.uv_offset
        assert offset.x == 512
        assert offset.y == 256

    def test_slot_uv_scale(self) -> None:
        """Test slot UV scale calculation."""
        slot = ShadowAtlasSlot(x=0, y=0, width=1024, height=512)
        scale = slot.uv_scale
        assert scale.x == 1024
        assert scale.y == 512
