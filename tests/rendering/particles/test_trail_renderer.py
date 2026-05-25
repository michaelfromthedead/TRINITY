"""
Tests for the trail/ribbon rendering system.

Tests:
    - TrailConfig creation
    - TrailBuffer ring buffer operations
    - TrailPoint data structure
    - TrailRenderer mesh generation
    - Texture mode (stretch/tile)
"""

import pytest

from engine.rendering.particles.particle_system import Vec3, Vec4
from engine.rendering.particles.trail_renderer import (
    TextureMode,
    TrailAlignment,
    TrailCapStyle,
    TrailConfig,
    TrailPoint,
    TrailBuffer,
    TrailVertex,
    TrailMesh,
    TrailRenderer,
    TrailManager,
)


class TestTrailConfig:
    """Test TrailConfig creation."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrailConfig()
        assert config.width == 0.1
        assert config.fade_time == 1.0
        assert config.texture_mode == TextureMode.STRETCH
        assert config.alignment == TrailAlignment.VIEW

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrailConfig(
            width=0.5,
            fade_time=2.0,
            texture_mode=TextureMode.TILE,
            max_points=50,
        )
        assert config.width == 0.5
        assert config.fade_time == 2.0
        assert config.texture_mode == TextureMode.TILE
        assert config.max_points == 50

    def test_from_decorator_params(self):
        """Test creation from decorator parameters."""
        config = TrailConfig.from_decorator_params(
            width=0.2,
            fade_time=3.0,
            texture_mode="tile",
        )
        assert config.width == 0.2
        assert config.fade_time == 3.0
        assert config.texture_mode == TextureMode.TILE


class TestTrailPoint:
    """Test TrailPoint data structure."""

    def test_default_point(self):
        """Test default point values."""
        point = TrailPoint()
        assert point.position.x == 0
        assert point.width == 0.1
        assert point.age == 0.0
        assert point.is_alive  # Alpha > 0

    def test_copy(self):
        """Test point copy."""
        original = TrailPoint(
            position=Vec3(1, 2, 3),
            width=0.5,
            age=1.0,
        )
        copy = original.copy()

        assert copy.position.x == 1
        assert copy.width == 0.5
        assert copy.age == 1.0

        # Modify copy should not affect original
        copy.position.x = 100
        assert original.position.x == 1


class TestTrailBuffer:
    """Test TrailBuffer ring buffer operations."""

    def test_creation(self):
        """Test buffer creation."""
        buffer = TrailBuffer(max_points=100)
        assert buffer.max_points == 100
        assert buffer.count == 0
        assert buffer.is_empty

    def test_add_point(self):
        """Test adding points."""
        buffer = TrailBuffer(max_points=10, min_distance=0.0)

        point = TrailPoint(position=Vec3(0, 0, 0))
        result = buffer.add_point(point)

        assert result is True
        assert buffer.count == 1

    def test_add_point_min_distance(self):
        """Test minimum distance filtering."""
        buffer = TrailBuffer(max_points=10, min_distance=1.0)

        p1 = TrailPoint(position=Vec3(0, 0, 0))
        p2 = TrailPoint(position=Vec3(0.5, 0, 0))  # Too close
        p3 = TrailPoint(position=Vec3(2, 0, 0))  # Far enough

        buffer.add_point(p1)
        result2 = buffer.add_point(p2)
        result3 = buffer.add_point(p3)

        assert result2 is False  # Rejected
        assert result3 is True
        assert buffer.count == 2

    def test_ring_buffer_overflow(self):
        """Test ring buffer overwrites oldest."""
        buffer = TrailBuffer(max_points=3, min_distance=0.0)

        for i in range(5):
            buffer.add_point(TrailPoint(position=Vec3(i, 0, 0)))

        assert buffer.count == 3

        # Oldest should be position 2, newest 4
        oldest = buffer.get_oldest()
        newest = buffer.get_newest()

        assert oldest.position.x == 2
        assert newest.position.x == 4

    def test_iter_points(self):
        """Test iterating over points."""
        buffer = TrailBuffer(max_points=10, min_distance=0.0)

        for i in range(5):
            buffer.add_point(TrailPoint(position=Vec3(i, 0, 0)))

        points = list(buffer.iter_points())
        assert len(points) == 5
        assert points[0].position.x == 0  # Oldest
        assert points[4].position.x == 4  # Newest

    def test_update_fade(self):
        """Test point fading during update."""
        buffer = TrailBuffer(max_points=10, min_distance=0.0)

        point = TrailPoint(position=Vec3(0, 0, 0))
        buffer.add_point(point)

        # Update with partial fade
        buffer.update(0.5, fade_time=1.0)

        stored_point = buffer.get_newest()
        assert stored_point.age == 0.5
        # Alpha should be reduced
        assert stored_point.color.w < 1.0

    def test_clear(self):
        """Test clearing buffer."""
        buffer = TrailBuffer(max_points=10, min_distance=0.0)

        for i in range(5):
            buffer.add_point(TrailPoint(position=Vec3(i, 0, 0)))

        buffer.clear()

        assert buffer.count == 0
        assert buffer.is_empty


class TestTrailRenderer:
    """Test TrailRenderer mesh generation."""

    def test_creation(self):
        """Test renderer creation."""
        config = TrailConfig(width=0.5)
        renderer = TrailRenderer(config)

        assert renderer.config.width == 0.5
        assert renderer.is_emitting

    def test_update_adds_points(self):
        """Test update adds trail points."""
        renderer = TrailRenderer()

        renderer.update(0.016, position=Vec3(0, 0, 0))
        renderer.update(0.016, position=Vec3(1, 0, 0))
        renderer.update(0.016, position=Vec3(2, 0, 0))

        assert renderer.buffer.count == 3

    def test_mesh_generation(self):
        """Test mesh is generated from points with correct ribbon geometry."""
        config = TrailConfig(min_distance=0.0, width=1.0)
        renderer = TrailRenderer(config)
        renderer.set_camera(Vec3(0, 0, 10))  # Set camera for view-aligned

        # Add several points
        for i in range(5):
            renderer.update(0.016, position=Vec3(i, 0, 0))

        # Should have vertices and indices
        assert renderer.mesh.vertex_count > 0
        assert renderer.mesh.index_count > 0

        # 2 vertices per point (left and right edge of ribbon)
        assert renderer.mesh.vertex_count == 10  # 5 points * 2 vertices

        # Verify ribbon geometry: each segment needs 2 triangles (6 indices)
        # 4 segments (between 5 points) = 4 * 6 = 24 indices
        # But actually implementation uses extend with 6 indices per segment
        expected_indices = (5 - 1) * 6  # segments * indices_per_segment
        assert renderer.mesh.index_count == expected_indices

        # Verify vertices have correct UV coordinates
        vertices = renderer.mesh.vertices
        # Left vertices should have v=0, right should have v=1
        left_uvs = [v.uv[1] for v in vertices[::2]]  # Every other starting at 0
        right_uvs = [v.uv[1] for v in vertices[1::2]]  # Every other starting at 1
        assert all(uv == 0.0 for uv in left_uvs)
        assert all(uv == 1.0 for uv in right_uvs)

    def test_stop_emitting(self):
        """Test stopping emission."""
        renderer = TrailRenderer()

        renderer.update(0.016, position=Vec3(0, 0, 0))
        renderer.stop_emitting()
        renderer.update(0.016, position=Vec3(1, 0, 0))

        # Should not add new points when stopped
        assert renderer.buffer.count == 1

    def test_clear(self):
        """Test clearing trail."""
        renderer = TrailRenderer()

        renderer.update(0.016, position=Vec3(0, 0, 0))
        renderer.update(0.016, position=Vec3(1, 0, 0))
        renderer.clear()

        assert renderer.buffer.count == 0
        assert renderer.mesh.vertex_count == 0

    def test_stats(self):
        """Test getting statistics."""
        config = TrailConfig(min_distance=0.0)
        renderer = TrailRenderer(config)

        for i in range(3):
            renderer.update(0.016, position=Vec3(i, 0, 0))

        stats = renderer.get_stats()

        assert stats["point_count"] == 3
        assert stats["is_emitting"] is True
        assert "vertex_count" in stats


class TestTrailManager:
    """Test TrailManager for multiple trails."""

    def test_create_trail(self):
        """Test creating trails."""
        manager = TrailManager()
        trail = manager.create_trail("test")

        assert trail is not None
        assert manager.get_trail("test") is trail

    def test_remove_trail(self):
        """Test removing trails."""
        manager = TrailManager()
        manager.create_trail("test")
        manager.remove_trail("test")

        assert manager.get_trail("test") is None

    def test_set_camera_all(self):
        """Test setting camera for all trails."""
        manager = TrailManager()
        manager.create_trail("t1")
        manager.create_trail("t2")

        # Should not raise
        manager.set_camera_all(Vec3(0, 10, 10))

    def test_get_total_vertex_count(self):
        """Test total vertex count."""
        config = TrailConfig(min_distance=0.0)
        manager = TrailManager(default_config=config)

        t1 = manager.create_trail("t1")
        t2 = manager.create_trail("t2")

        t1.update(0.016, position=Vec3(0, 0, 0))
        t1.update(0.016, position=Vec3(1, 0, 0))
        t2.update(0.016, position=Vec3(0, 0, 0))
        t2.update(0.016, position=Vec3(0, 1, 0))

        total = manager.get_total_vertex_count()
        assert total == 8  # 2 trails * 2 points * 2 vertices


class TestTrailTextureMode:
    """Test trail texture mapping modes."""

    def test_stretch_mode(self):
        """Test stretch texture mode UV calculation."""
        config = TrailConfig(
            texture_mode=TextureMode.STRETCH,
            min_distance=0.0,
        )
        renderer = TrailRenderer(config)

        # Create 3 points
        for i in range(3):
            renderer.update(0.016, position=Vec3(i, 0, 0))

        # UVs should span 0-1 along trail
        vertices = renderer.mesh.vertices
        u_values = [v.uv[0] for v in vertices]

        assert min(u_values) == 0.0
        assert max(u_values) == 1.0  # UV stretches full length


class TestTrailAlignment:
    """Test trail alignment modes."""

    def test_view_alignment(self):
        """Test view-aligned (billboard) trails."""
        config = TrailConfig(
            alignment=TrailAlignment.VIEW,
            min_distance=0.0,
        )
        renderer = TrailRenderer(config)
        renderer.set_camera(Vec3(0, 0, 10))

        renderer.update(0.016, position=Vec3(0, 0, 0))
        renderer.update(0.016, position=Vec3(1, 0, 0))

        # Should generate mesh without errors
        assert renderer.mesh.vertex_count > 0

    def test_velocity_alignment(self):
        """Test velocity-aligned trails."""
        config = TrailConfig(
            alignment=TrailAlignment.VELOCITY,
            min_distance=0.0,
        )
        renderer = TrailRenderer(config)

        renderer.update(0.016, position=Vec3(0, 0, 0), velocity=Vec3(1, 0, 0))
        renderer.update(0.016, position=Vec3(1, 0, 0), velocity=Vec3(1, 0, 0))

        # Should generate mesh without errors
        assert renderer.mesh.vertex_count > 0


class TestTrailCaps:
    """Test trail cap styles."""

    def test_no_cap(self):
        """Test trail without caps."""
        config = TrailConfig(
            cap_style=TrailCapStyle.NONE,
            min_distance=0.0,
        )
        renderer = TrailRenderer(config)

        for i in range(3):
            renderer.update(0.016, position=Vec3(i, 0, 0))

        # Base vertices: 6 (3 points * 2)
        assert renderer.mesh.vertex_count == 6

    def test_round_cap(self):
        """Test round cap style."""
        config = TrailConfig(
            cap_style=TrailCapStyle.ROUND,
            min_distance=0.0,
        )
        renderer = TrailRenderer(config)

        for i in range(3):
            renderer.update(0.016, position=Vec3(i, 0, 0))

        # Should have additional vertices for caps
        assert renderer.mesh.vertex_count > 6
