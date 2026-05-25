"""
Tests for the decal system.

Tests:
    - DecalConfig creation
    - DecalVolume projection
    - Decal lifecycle (spawn, update, fade, death)
    - DeferredDecal G-Buffer modification
    - DecalAtlas texture packing
    - DecalSorting by priority and depth
    - DecalSystem management
"""

import pytest

from engine.rendering.particles.particle_system import Vec3, Vec4
from engine.rendering.particles.decal_system import (
    DecalChannel,
    DecalBlendMode,
    DecalProjection,
    DecalSortMode,
    DecalConfig,
    DecalVolume,
    AtlasRegion,
    Decal,
    DeferredDecal,
    DecalAtlas,
    DecalSorting,
    DecalSystem,
)


class TestDecalConfig:
    """Test DecalConfig creation."""

    def test_default_config(self):
        """Test default configuration."""
        config = DecalConfig()
        assert config.lifetime is None  # Infinite
        assert config.fade_time == 1.0
        assert config.channel == 0
        assert config.priority == 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = DecalConfig(
            lifetime=5.0,
            fade_time=2.0,
            channel=1,
            priority=10,
        )
        assert config.lifetime == 5.0
        assert config.fade_time == 2.0
        assert config.channel == 1
        assert config.priority == 10

    def test_from_decorator_params(self):
        """Test creation from decorator parameters."""
        config = DecalConfig.from_decorator_params(
            lifetime=3.0,
            fade_time=0.5,
            channel=2,
        )
        assert config.lifetime == 3.0
        assert config.fade_time == 0.5
        assert config.channel == 2


class TestDecalVolume:
    """Test DecalVolume projection."""

    def test_default_volume(self):
        """Test default volume."""
        volume = DecalVolume()
        assert volume.position.x == 0
        assert volume.size.x == 1

    def test_contains_point(self):
        """Test point containment check."""
        volume = DecalVolume(
            position=Vec3(0, 0, 0),
            size=Vec3(2, 2, 2),
        )

        # Point inside
        assert volume.contains_point(Vec3(0.5, 0.5, 0.5))

        # Point outside
        assert not volume.contains_point(Vec3(2, 0, 0))

    def test_get_corners(self):
        """Test getting volume corners."""
        volume = DecalVolume(
            position=Vec3(0, 0, 0),
            size=Vec3(2, 2, 2),
        )

        corners = volume.get_corners()
        assert len(corners) == 8

        # Check one corner
        has_corner = any(
            c.x == -1 and c.y == -1 and c.z == -1
            for c in corners
        )
        assert has_corner

    def test_get_uv_for_point(self):
        """Test UV coordinate calculation for decal projection."""
        volume = DecalVolume(
            position=Vec3(0, 0, 0),
            size=Vec3(2, 2, 2),
        )

        # Center should be (0.5, 0.5)
        u, v = volume.get_uv_for_point(Vec3(0, 0, 0))
        assert abs(u - 0.5) < 0.01
        assert abs(v - 0.5) < 0.01

        # Corner should be (0, 0)
        u, v = volume.get_uv_for_point(Vec3(-1, 0, -1))
        assert abs(u - 0.0) < 0.01
        assert abs(v - 0.0) < 0.01

        # Opposite corner should be (1, 1)
        u, v = volume.get_uv_for_point(Vec3(1, 0, 1))
        assert abs(u - 1.0) < 0.01
        assert abs(v - 1.0) < 0.01

        # Test UV clamping for points outside volume
        u, v = volume.get_uv_for_point(Vec3(5, 0, 5))
        assert 0.0 <= u <= 1.0
        assert 0.0 <= v <= 1.0

    def test_volume_projection_accuracy(self):
        """Test that decal volume correctly projects onto surfaces."""
        volume = DecalVolume(
            position=Vec3(5, 5, 5),
            size=Vec3(4, 4, 4),  # 4x4x4 volume centered at (5,5,5)
        )

        # Test corners of the volume
        assert volume.contains_point(Vec3(5, 5, 5))  # Center
        assert volume.contains_point(Vec3(3, 3, 3))  # Min corner
        assert volume.contains_point(Vec3(7, 7, 7))  # Max corner

        # Points just outside should not be contained
        assert not volume.contains_point(Vec3(2.9, 5, 5))
        assert not volume.contains_point(Vec3(7.1, 5, 5))


class TestDecal:
    """Test Decal lifecycle."""

    def test_creation(self):
        """Test decal creation."""
        decal = Decal()
        assert decal.is_alive
        assert decal.age == 0.0
        assert decal.alpha == 1.0

    def test_position(self):
        """Test setting position."""
        decal = Decal()
        decal.position = Vec3(1, 2, 3)

        assert decal.position.x == 1
        assert decal.position.y == 2
        assert decal.position.z == 3

    def test_update_aging(self):
        """Test aging during update."""
        decal = Decal()
        decal.update(0.5)

        assert decal.age == 0.5

    def test_update_fade_with_lifetime(self):
        """Test fading after lifetime expires."""
        config = DecalConfig(lifetime=1.0, fade_time=1.0)
        decal = Decal(config=config)

        # Before lifetime
        decal.update(0.5)
        assert decal.alpha == 1.0
        assert not decal.is_fading

        # After lifetime, starts fading
        decal.update(0.6)
        assert decal.is_fading
        assert decal.alpha < 1.0

    def test_update_death(self):
        """Test decal death after full fade."""
        config = DecalConfig(lifetime=1.0, fade_time=0.5)
        decal = Decal(config=config)

        # Fast forward past lifetime + fade
        decal.update(2.0)

        assert not decal.is_alive
        assert decal.alpha <= 0.0

    def test_kill_immediate(self):
        """Test immediate kill."""
        decal = Decal()
        decal.kill(immediate=True)

        assert not decal.is_alive

    def test_set_channels(self):
        """Test setting G-Buffer channels."""
        decal = Decal()
        decal.set_channels(DecalChannel.ALBEDO, DecalChannel.NORMAL)

        assert DecalChannel.ALBEDO in decal.channels
        assert DecalChannel.NORMAL in decal.channels


class TestDeferredDecal:
    """Test DeferredDecal G-Buffer modification."""

    def test_creation(self):
        """Test deferred decal creation."""
        decal = DeferredDecal(
            albedo_texture="albedo.png",
            normal_texture="normal.png",
        )

        assert decal.albedo_texture == "albedo.png"
        assert decal.normal_texture == "normal.png"

    def test_auto_channels(self):
        """Test automatic channel detection from textures."""
        decal = DeferredDecal(
            albedo_texture="albedo.png",
            normal_texture="normal.png",
        )

        assert DecalChannel.ALBEDO in decal.channels
        assert DecalChannel.NORMAL in decal.channels
        assert DecalChannel.ROUGHNESS not in decal.channels

    def test_channel_weights(self):
        """Test channel blend weights."""
        decal = DeferredDecal()
        decal.set_channel_weight(DecalChannel.ALBEDO, 0.5)

        weight = decal.get_channel_weight(DecalChannel.ALBEDO)
        assert weight == 0.5


class TestDecalAtlas:
    """Test DecalAtlas texture packing."""

    def test_creation(self):
        """Test atlas creation."""
        atlas = DecalAtlas(width=1024, height=1024)
        assert atlas.width == 1024
        assert atlas.height == 1024

    def test_add_texture(self):
        """Test adding texture to atlas."""
        atlas = DecalAtlas(width=1024, height=1024)

        region = atlas.add_texture("tex1", 100, 100)

        assert region is not None
        assert region.texture_id == "tex1"
        assert region.width == 100
        assert region.height == 100

    def test_add_duplicate_texture(self):
        """Test adding same texture returns existing region."""
        atlas = DecalAtlas()

        region1 = atlas.add_texture("tex1", 100, 100)
        region2 = atlas.add_texture("tex1", 100, 100)

        assert region1 is region2

    def test_atlas_full(self):
        """Test atlas full behavior."""
        atlas = DecalAtlas(width=100, height=100)

        # Fill atlas
        atlas.add_texture("tex1", 80, 80)

        # Should not fit
        region = atlas.add_texture("tex2", 80, 80)
        assert region is None

    def test_get_uv_rect(self):
        """Test getting UV rectangle."""
        atlas = DecalAtlas(width=1000, height=1000, padding=0)

        atlas.add_texture("tex1", 100, 100)
        uv = atlas.get_uv_rect("tex1")

        assert uv is not None
        u_min, v_min, u_max, v_max = uv
        assert u_min == 0.0
        assert v_min == 0.0
        assert abs(u_max - 0.1) < 0.01
        assert abs(v_max - 0.1) < 0.01

    def test_occupancy(self):
        """Test occupancy calculation."""
        atlas = DecalAtlas(width=100, height=100, padding=0)

        assert atlas.get_occupancy() == 0.0

        atlas.add_texture("tex1", 50, 50)
        occupancy = atlas.get_occupancy()

        assert occupancy > 0.0


class TestDecalSorting:
    """Test DecalSorting utilities."""

    def test_sort_by_priority(self):
        """Test sorting by priority."""
        decals = [
            Decal(DecalConfig(priority=3)),
            Decal(DecalConfig(priority=1)),
            Decal(DecalConfig(priority=2)),
        ]

        sorted_decals = DecalSorting.sort_by_priority(decals)

        assert sorted_decals[0].priority == 1
        assert sorted_decals[1].priority == 2
        assert sorted_decals[2].priority == 3

    def test_sort_by_depth(self):
        """Test sorting by depth (back to front)."""
        decals = []

        d1 = Decal()
        d1.position = Vec3(0, 0, 5)  # Closest
        decals.append(d1)

        d2 = Decal()
        d2.position = Vec3(0, 0, 15)  # Farthest
        decals.append(d2)

        d3 = Decal()
        d3.position = Vec3(0, 0, 10)  # Middle
        decals.append(d3)

        camera = Vec3(0, 0, 0)
        sorted_decals = DecalSorting.sort_by_depth(decals, camera)

        # Should be back to front (farthest first)
        assert sorted_decals[0].position.z == 15
        assert sorted_decals[1].position.z == 10
        assert sorted_decals[2].position.z == 5


class TestDecalSystem:
    """Test DecalSystem management."""

    def test_creation(self):
        """Test system creation."""
        system = DecalSystem(max_decals=100)
        assert system.decal_count == 0

    def test_spawn(self):
        """Test spawning decal."""
        system = DecalSystem()
        decal = system.spawn(
            position=Vec3(0, 0, 0),
            size=Vec3(1, 1, 1),
            texture_id="bullet_hole",
        )

        assert decal is not None
        assert system.decal_count == 1

    def test_spawn_deferred(self):
        """Test spawning deferred decal."""
        system = DecalSystem()
        decal = system.spawn_deferred(
            position=Vec3(0, 0, 0),
            albedo_texture="albedo.png",
            normal_texture="normal.png",
        )

        assert decal is not None
        assert isinstance(decal, DeferredDecal)

    def test_spawn_at_capacity(self):
        """Test spawning when at capacity."""
        system = DecalSystem(max_decals=2)

        system.spawn(Vec3(0, 0, 0))
        system.spawn(Vec3(1, 0, 0))
        decal = system.spawn(Vec3(2, 0, 0))

        # Should fail when at capacity
        assert decal is None

    def test_update_removes_dead(self):
        """Test update removes dead decals."""
        config = DecalConfig(lifetime=0.5, fade_time=0.1)
        system = DecalSystem(default_config=config)

        system.spawn(Vec3(0, 0, 0))
        assert system.decal_count == 1

        # Fast forward past lifetime + fade
        system.update(1.0)

        assert system.decal_count == 0

    def test_get_decal(self):
        """Test getting decal by ID."""
        system = DecalSystem()
        decal = system.spawn(Vec3(0, 0, 0))

        retrieved = system.get_decal(decal.id)
        assert retrieved is decal

    def test_remove_decal(self):
        """Test removing decal."""
        system = DecalSystem()
        decal = system.spawn(Vec3(0, 0, 0))
        system.remove_decal(decal.id)

        assert system.decal_count == 0

    def test_iter_visible(self):
        """Test iterating visible decals."""
        system = DecalSystem()

        d1 = system.spawn(Vec3(0, 0, 0))
        d2 = system.spawn(Vec3(1, 0, 0))
        d2.visible = False

        visible = list(system.iter_visible())

        assert len(visible) == 1
        assert visible[0] is d1

    def test_iter_by_channel(self):
        """Test iterating by G-Buffer channel."""
        system = DecalSystem()

        d1 = system.spawn(Vec3(0, 0, 0))
        d1.set_channels(DecalChannel.ALBEDO)

        d2 = system.spawn(Vec3(1, 0, 0))
        d2.set_channels(DecalChannel.NORMAL)

        albedo_decals = list(system.iter_by_channel(DecalChannel.ALBEDO))
        normal_decals = list(system.iter_by_channel(DecalChannel.NORMAL))

        assert len(albedo_decals) == 1
        assert len(normal_decals) == 1

    def test_stats(self):
        """Test getting statistics."""
        system = DecalSystem()
        system.spawn(Vec3(0, 0, 0))

        stats = system.get_stats()

        assert stats["total_decals"] == 1
        assert "visible_decals" in stats
        assert "max_decals" in stats


class TestDecalSortModes:
    """Test different sort modes in system."""

    def test_priority_sort_mode(self):
        """Test priority-based sorting."""
        system = DecalSystem()
        system.set_sort_mode(DecalSortMode.PRIORITY)

        d1 = system.spawn(Vec3(0, 0, 0), config=DecalConfig(priority=2))
        d2 = system.spawn(Vec3(0, 0, 0), config=DecalConfig(priority=1))

        system.update(0.016)

        visible = list(system.iter_visible())
        assert visible[0].priority <= visible[1].priority

    def test_depth_sort_mode(self):
        """Test depth-based sorting."""
        system = DecalSystem()
        system.set_sort_mode(DecalSortMode.DEPTH)
        system.set_camera(Vec3(0, 0, 0))

        d1 = system.spawn(Vec3(0, 0, 5))
        d2 = system.spawn(Vec3(0, 0, 10))

        system.update(0.016)

        visible = list(system.iter_visible())
        # Back to front: farther should be first
        assert visible[0].depth >= visible[1].depth
