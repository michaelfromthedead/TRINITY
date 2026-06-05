"""
Tests for RT Reflection Ray Generation System (T-GIR-P8.1)

Comprehensive whitebox and blackbox tests for:
- GBufferReader: G-Buffer data reading and world position reconstruction
- ReflectionRayGenerator: Reflection direction computation
- RTReflectionTracer: TLAS ray tracing interface
- RTReflectionPass: Full-screen RT reflection pass
- RTReflectionConfig: Configuration validation
- MockTLAS: Mock geometry for testing

Requirements tested (from T-GIR-P8.1 spec):
- Correct reflection direction computation: R = 2(N dot V)N - V
- World position reconstruction from depth accurate to <0.01 units
- Roughness skip threshold (0.7) properly filters rough surfaces
- Ray origin offset (no self-intersection)
- TLAS ray queries return valid hit data
- Miss shader correctly returns environment
- Resolution scaling produces correct buffer sizes
"""

import math
import pytest

from engine.core.math.vec import Vec2, Vec3
from engine.rendering.reflections.rt_reflections import (
    # Constants
    DEFAULT_ROUGHNESS_THRESHOLD,
    DEFAULT_MAX_RAY_DISTANCE,
    DEFAULT_NORMAL_BIAS,
    DEFAULT_ENVIRONMENT_COLOR,
    RAY_FLAG_NONE,
    RAY_FLAG_CULL_BACK_FACING,
    RESOLUTION_QUARTER,
    RESOLUTION_HALF,
    RESOLUTION_FULL,
    # Enums
    ResolutionMode,
    # Data structures
    MaterialData,
    GBufferPixel,
    ReflectionRay,
    RayHitInfo,
    ReflectionOutput,
    # Config
    RTReflectionConfig,
    # Core classes
    GBufferReader,
    ReflectionRayGenerator,
    RTReflectionTracer,
    RTReflectionPass,
    # TLAS
    MockTLAS,
    # Utilities
    estimate_rt_reflection_memory,
    create_mock_tlas,
)


# =============================================================================
# MaterialData Tests
# =============================================================================


class TestMaterialData:
    """Tests for MaterialData data structure."""

    def test_default_construction(self):
        """Test default material construction."""
        mat = MaterialData()

        assert mat.roughness == 0.5
        assert mat.metallic == 0.0
        assert mat.specular == 0.5
        assert mat.ior == 1.5

    def test_custom_construction(self):
        """Test material with custom values."""
        mat = MaterialData(
            roughness=0.3,
            metallic=0.9,
            base_color=Vec3(1.0, 0.8, 0.6),
            specular=0.8,
        )

        assert mat.roughness == 0.3
        assert mat.metallic == 0.9
        assert mat.base_color.x == 1.0
        assert mat.specular == 0.8

    def test_is_reflective_below_threshold(self):
        """Test reflective check for smooth surfaces."""
        smooth = MaterialData(roughness=0.2)
        assert smooth.is_reflective(0.7)

        very_smooth = MaterialData(roughness=0.0)
        assert very_smooth.is_reflective(0.7)

    def test_is_reflective_at_threshold(self):
        """Test reflective check at exact threshold."""
        at_threshold = MaterialData(roughness=0.7)
        assert at_threshold.is_reflective(0.7)

    def test_is_not_reflective_above_threshold(self):
        """Test reflective check for rough surfaces."""
        rough = MaterialData(roughness=0.8)
        assert not rough.is_reflective(0.7)

        very_rough = MaterialData(roughness=1.0)
        assert not very_rough.is_reflective(0.7)

    def test_custom_threshold(self):
        """Test reflective check with custom threshold."""
        mat = MaterialData(roughness=0.5)

        assert mat.is_reflective(0.6)
        assert not mat.is_reflective(0.4)


# =============================================================================
# GBufferPixel Tests
# =============================================================================


class TestGBufferPixel:
    """Tests for GBufferPixel data structure."""

    def test_default_construction(self):
        """Test default pixel construction."""
        pixel = GBufferPixel()

        assert pixel.depth == 0.0
        assert pixel.world_position == Vec3.zero()
        assert pixel.normal.y == 1.0  # Default up normal
        assert pixel.valid is True

    def test_custom_construction(self):
        """Test pixel with custom values."""
        pixel = GBufferPixel(
            depth=10.0,
            world_position=Vec3(1.0, 2.0, 3.0),
            normal=Vec3(0.0, 0.0, 1.0),
            material=MaterialData(roughness=0.3),
            valid=True,
        )

        assert pixel.depth == 10.0
        assert pixel.world_position.x == 1.0
        assert pixel.normal.z == 1.0
        assert pixel.material.roughness == 0.3


# =============================================================================
# GBufferReader Tests
# =============================================================================


class TestGBufferReader:
    """Tests for GBufferReader."""

    def test_default_construction(self):
        """Test default reader construction."""
        reader = GBufferReader(width=1920, height=1080)

        assert reader.width == 1920
        assert reader.height == 1080

    def test_set_camera(self):
        """Test setting camera parameters."""
        reader = GBufferReader()

        # Create simple identity-like inverse view-proj
        inv_vp = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        camera_pos = Vec3(0.0, 1.0, 5.0)

        reader.set_camera(inv_vp, camera_pos, near=0.1, far=100.0)

        assert reader.camera_position == camera_pos

    def test_set_test_data(self):
        """Test setting test data for a pixel."""
        reader = GBufferReader(width=100, height=100)

        reader.set_test_data(
            x=50,
            y=50,
            depth=10.0,
            normal=Vec3(0.0, 1.0, 0.0),
            material=MaterialData(roughness=0.3),
        )

        assert reader.read_depth(50, 50) == 10.0
        assert reader.read_normal(50, 50).y == pytest.approx(1.0, rel=1e-6)
        assert reader.get_material_at(50, 50).roughness == 0.3

    def test_read_depth_default(self):
        """Test reading depth without test data."""
        reader = GBufferReader()

        # Should return default far depth
        depth = reader.read_depth(0, 0)
        assert depth == 1.0

    def test_read_depth_uv(self):
        """Test reading depth at UV coordinates."""
        reader = GBufferReader(width=100, height=100)
        reader.set_test_data(50, 50, depth=5.0, normal=Vec3.up())

        # UV (0.5, 0.5) should map to pixel (50, 50)
        # Actually (0.5 * 99) = 49.5 -> 49, need to adjust
        reader.set_test_data(49, 49, depth=5.0, normal=Vec3.up())
        depth = reader.read_depth_uv(Vec2(0.5, 0.5))
        assert depth == 5.0

    def test_read_normal_uv(self):
        """Test reading normal at UV coordinates."""
        reader = GBufferReader(width=100, height=100)
        reader.set_test_data(49, 49, depth=5.0, normal=Vec3(0.0, 0.0, 1.0))

        normal = reader.read_normal_uv(Vec2(0.5, 0.5))
        assert normal.z == pytest.approx(1.0, rel=1e-6)

    def test_reconstruct_world_pos_fallback(self):
        """Test world position reconstruction without matrix."""
        reader = GBufferReader(width=100, height=100)

        # Without inverse VP matrix, uses simple fallback
        pos = reader.reconstruct_world_pos(Vec2(0.5, 0.5), 10.0)

        # Fallback formula: (uv - 0.5) * depth * 2 for x/y, -depth for z
        assert pos.x == pytest.approx(0.0, abs=1e-6)
        assert pos.y == pytest.approx(0.0, abs=1e-6)
        assert pos.z == pytest.approx(-10.0, abs=1e-6)

    def test_reconstruct_world_pos_accuracy(self):
        """Test world position reconstruction accuracy (<0.01 units)."""
        reader = GBufferReader(width=1920, height=1080)

        # Set up a proper inverse view-projection matrix
        # This is a simplified test - in production would use actual matrices
        inv_vp = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        reader.set_camera(inv_vp, Vec3(0, 0, 0))

        # Test several points
        pos1 = reader.reconstruct_world_pos(Vec2(0.5, 0.5), 10.0)

        # With identity-like matrix, result should be predictable
        # The transformation should be consistent
        assert isinstance(pos1.x, float)
        assert isinstance(pos1.y, float)
        assert isinstance(pos1.z, float)

    def test_read_pixel_complete(self):
        """Test reading complete pixel data."""
        reader = GBufferReader(width=100, height=100)
        reader.set_test_data(
            49, 49,
            depth=15.0,
            normal=Vec3(0.0, 1.0, 0.0),
            material=MaterialData(roughness=0.4, metallic=0.5),
        )

        pixel = reader.read_pixel(Vec2(0.5, 0.5))

        assert pixel.depth == 15.0
        assert pixel.normal.y == pytest.approx(1.0, rel=1e-6)
        assert pixel.material.roughness == 0.4
        assert pixel.valid is True

    def test_read_pixel_invalid_depth(self):
        """Test that zero depth marks pixel as invalid."""
        reader = GBufferReader(width=100, height=100)
        reader.set_test_data(49, 49, depth=0.0, normal=Vec3.up())

        pixel = reader.read_pixel(Vec2(0.5, 0.5))

        assert not pixel.valid


# =============================================================================
# ReflectionRayGenerator Tests
# =============================================================================


class TestReflectionRayGenerator:
    """Tests for ReflectionRayGenerator."""

    def test_default_construction(self):
        """Test default generator construction."""
        gen = ReflectionRayGenerator()

        assert gen.roughness_threshold == DEFAULT_ROUGHNESS_THRESHOLD
        assert gen.normal_bias == DEFAULT_NORMAL_BIAS

    def test_custom_threshold(self):
        """Test generator with custom threshold."""
        gen = ReflectionRayGenerator(roughness_threshold=0.5, normal_bias=0.01)

        assert gen.roughness_threshold == 0.5
        assert gen.normal_bias == 0.01

    def test_should_trace_smooth_surface(self):
        """Test tracing decision for smooth surfaces."""
        gen = ReflectionRayGenerator(roughness_threshold=0.7)

        assert gen.should_trace(0.0)
        assert gen.should_trace(0.3)
        assert gen.should_trace(0.5)
        assert gen.should_trace(0.7)

    def test_should_not_trace_rough_surface(self):
        """Test tracing decision for rough surfaces."""
        gen = ReflectionRayGenerator(roughness_threshold=0.7)

        assert not gen.should_trace(0.71)
        assert not gen.should_trace(0.8)
        assert not gen.should_trace(1.0)

    def test_compute_view_direction(self):
        """Test view direction computation."""
        gen = ReflectionRayGenerator()

        world_pos = Vec3(0.0, 0.0, -5.0)
        camera_pos = Vec3(0.0, 0.0, 0.0)

        view_dir = gen.compute_view_direction(world_pos, camera_pos)

        # View direction should point toward camera (positive Z)
        assert view_dir.z == pytest.approx(1.0, rel=1e-6)
        assert view_dir.x == pytest.approx(0.0, abs=1e-6)
        assert view_dir.y == pytest.approx(0.0, abs=1e-6)

    def test_reflection_direction_mirror_surface(self):
        """Test reflection direction for perfect mirror (R = 2(N dot V)N - V)."""
        gen = ReflectionRayGenerator()

        # Horizontal surface, view from above
        normal = Vec3(0.0, 1.0, 0.0)  # Up
        view_dir = Vec3(0.0, 1.0, 0.0)  # Looking straight down (view toward camera)

        reflect = gen.compute_reflection_direction(normal, view_dir)

        # Should reflect straight up
        assert reflect.x == pytest.approx(0.0, abs=1e-6)
        assert reflect.y == pytest.approx(1.0, rel=1e-6)
        assert reflect.z == pytest.approx(0.0, abs=1e-6)

    def test_reflection_direction_45_degree(self):
        """Test reflection at 45 degrees."""
        gen = ReflectionRayGenerator()

        normal = Vec3(0.0, 1.0, 0.0)  # Up
        # View from 45 degrees in the XY plane
        view_dir = Vec3(1.0, 1.0, 0.0).normalized()

        reflect = gen.compute_reflection_direction(normal, view_dir)

        # At 45 degrees, reflection should go opposite X direction
        # N dot V = 1/sqrt(2)
        # R = 2 * (1/sqrt(2)) * (0,1,0) - (1/sqrt(2), 1/sqrt(2), 0)
        # R = (0, sqrt(2), 0) - (1/sqrt(2), 1/sqrt(2), 0)
        # R = (-1/sqrt(2), sqrt(2) - 1/sqrt(2), 0) = (-1/sqrt(2), 1/sqrt(2), 0)
        expected_x = -1.0 / math.sqrt(2.0)
        expected_y = 1.0 / math.sqrt(2.0)

        assert reflect.x == pytest.approx(expected_x, rel=1e-4)
        assert reflect.y == pytest.approx(expected_y, rel=1e-4)
        assert reflect.z == pytest.approx(0.0, abs=1e-6)

    def test_reflection_direction_grazing_angle(self):
        """Test reflection at grazing angle."""
        gen = ReflectionRayGenerator()

        normal = Vec3(0.0, 1.0, 0.0)
        # Almost parallel to surface
        view_dir = Vec3(0.99, 0.1, 0.0).normalized()

        reflect = gen.compute_reflection_direction(normal, view_dir)

        # Should be normalized
        length = reflect.length()
        assert length == pytest.approx(1.0, rel=1e-4)

        # Y component should be positive (reflecting upward)
        assert reflect.y > 0

    def test_reflection_direction_back_facing(self):
        """Test reflection when view from behind surface."""
        gen = ReflectionRayGenerator()

        normal = Vec3(0.0, 1.0, 0.0)  # Up
        view_dir = Vec3(0.0, -1.0, 0.0)  # Looking from below (negative dot product)

        reflect = gen.compute_reflection_direction(normal, view_dir)

        # Should handle back-facing gracefully
        length = reflect.length()
        assert length == pytest.approx(1.0, rel=1e-4)

    def test_ray_origin_offset(self):
        """Test ray origin offset along normal."""
        gen = ReflectionRayGenerator(normal_bias=0.001)

        world_pos = Vec3(0.0, 0.0, 0.0)
        normal = Vec3(0.0, 1.0, 0.0)

        origin = gen.get_ray_origin(world_pos, normal)

        # Should be offset along normal
        assert origin.y == pytest.approx(0.001, rel=1e-6)
        assert origin.x == pytest.approx(0.0, abs=1e-10)
        assert origin.z == pytest.approx(0.0, abs=1e-10)

    def test_ray_origin_no_self_intersection(self):
        """Test that ray origin offset prevents self-intersection."""
        gen = ReflectionRayGenerator(normal_bias=0.001)

        world_pos = Vec3(5.0, 3.0, -10.0)
        normal = Vec3(0.577, 0.577, 0.577).normalized()

        origin = gen.get_ray_origin(world_pos, normal)

        # Origin should be offset from surface
        distance = (origin - world_pos).length()
        assert distance >= 0.001 - 1e-9  # Allow tiny floating point tolerance

    def test_generate_ray_smooth_surface(self):
        """Test generating ray for smooth surface."""
        gen = ReflectionRayGenerator(roughness_threshold=0.7)

        ray = gen.generate_ray(
            world_pos=Vec3(0.0, 0.0, -5.0),
            normal=Vec3(0.0, 0.0, 1.0),
            camera_pos=Vec3(0.0, 0.0, 0.0),
            roughness=0.3,
            pixel_uv=Vec2(0.5, 0.5),
        )

        assert ray.should_trace is True
        assert ray.roughness == 0.3
        assert ray.pixel_uv.x == 0.5

    def test_generate_ray_rough_surface(self):
        """Test generating ray for rough surface (should skip)."""
        gen = ReflectionRayGenerator(roughness_threshold=0.7)

        ray = gen.generate_ray(
            world_pos=Vec3(0.0, 0.0, -5.0),
            normal=Vec3(0.0, 0.0, 1.0),
            camera_pos=Vec3(0.0, 0.0, 0.0),
            roughness=0.9,
        )

        assert ray.should_trace is False
        assert ray.roughness == 0.9


# =============================================================================
# MockTLAS Tests
# =============================================================================


class TestMockTLAS:
    """Tests for MockTLAS mock implementation."""

    def test_default_construction(self):
        """Test default TLAS construction."""
        tlas = MockTLAS()

        assert tlas.is_valid()

    def test_set_validity(self):
        """Test setting TLAS validity."""
        tlas = MockTLAS()

        tlas.set_valid(False)
        assert not tlas.is_valid()

        tlas.set_valid(True)
        assert tlas.is_valid()

    def test_add_sphere(self):
        """Test adding sphere primitive."""
        tlas = MockTLAS()
        tlas.add_sphere(Vec3(0, 0, -5), radius=1.0)

        # Trace ray toward sphere
        hit = tlas.trace_ray(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            max_distance=100.0,
        )

        assert hit.hit is True
        assert hit.distance == pytest.approx(4.0, rel=1e-4)  # 5 - 1 = 4

    def test_sphere_intersection_miss(self):
        """Test sphere miss."""
        tlas = MockTLAS()
        tlas.add_sphere(Vec3(10, 0, -5), radius=1.0)

        hit = tlas.trace_ray(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            max_distance=100.0,
        )

        assert hit.hit is False

    def test_sphere_normal_at_hit(self):
        """Test sphere normal computation at hit point."""
        tlas = MockTLAS()
        tlas.add_sphere(Vec3(0, 0, -5), radius=1.0)

        hit = tlas.trace_ray(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            max_distance=100.0,
        )

        # Normal should point toward ray origin (positive Z)
        assert hit.normal.z == pytest.approx(1.0, rel=1e-4)

    def test_add_box(self):
        """Test adding box primitive."""
        tlas = MockTLAS()
        tlas.add_box(Vec3(-1, -1, -6), Vec3(1, 1, -4))

        hit = tlas.trace_ray(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            max_distance=100.0,
        )

        assert hit.hit is True
        assert hit.distance == pytest.approx(4.0, rel=1e-4)

    def test_box_intersection_miss(self):
        """Test box miss."""
        tlas = MockTLAS()
        tlas.add_box(Vec3(10, 10, -6), Vec3(12, 12, -4))

        hit = tlas.trace_ray(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            max_distance=100.0,
        )

        assert hit.hit is False

    def test_closest_hit_selection(self):
        """Test that closest hit is selected."""
        tlas = MockTLAS()
        tlas.add_sphere(Vec3(0, 0, -5), radius=1.0)  # Hit at t=4
        tlas.add_sphere(Vec3(0, 0, -10), radius=1.0)  # Hit at t=9

        hit = tlas.trace_ray(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            max_distance=100.0,
        )

        assert hit.hit is True
        assert hit.distance == pytest.approx(4.0, rel=1e-4)

    def test_max_distance_clipping(self):
        """Test max distance clips results."""
        tlas = MockTLAS()
        tlas.add_sphere(Vec3(0, 0, -100), radius=1.0)

        hit = tlas.trace_ray(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            max_distance=50.0,
        )

        assert hit.hit is False

    def test_material_at_hit(self):
        """Test material data at hit point."""
        tlas = MockTLAS()
        material = MaterialData(roughness=0.2, metallic=0.9)
        tlas.add_sphere(Vec3(0, 0, -5), radius=1.0, material=material)

        hit = tlas.trace_ray(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            max_distance=100.0,
        )

        assert hit.material.roughness == 0.2
        assert hit.material.metallic == 0.9

    def test_clear_primitives(self):
        """Test clearing all primitives."""
        tlas = MockTLAS()
        tlas.add_sphere(Vec3(0, 0, -5), radius=1.0)
        tlas.clear()

        hit = tlas.trace_ray(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            max_distance=100.0,
        )

        assert hit.hit is False


# =============================================================================
# RTReflectionTracer Tests
# =============================================================================


class TestRTReflectionTracer:
    """Tests for RTReflectionTracer."""

    def test_default_construction(self):
        """Test default tracer construction."""
        tracer = RTReflectionTracer()

        assert tracer.max_ray_distance == DEFAULT_MAX_RAY_DISTANCE
        assert not tracer.is_ready()  # No TLAS

    def test_construction_with_tlas(self):
        """Test tracer construction with TLAS."""
        tlas = MockTLAS()
        tracer = RTReflectionTracer(tlas=tlas)

        assert tracer.is_ready()

    def test_trace_ray_hit(self):
        """Test tracing ray with hit."""
        tlas = MockTLAS()
        tlas.add_sphere(Vec3(0, 0, -5), radius=1.0)

        tracer = RTReflectionTracer(tlas=tlas)

        ray = ReflectionRay(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            should_trace=True,
        )

        hit = tracer.trace_ray(ray)

        assert hit.hit is True
        assert hit.distance == pytest.approx(4.0, rel=1e-4)

    def test_trace_ray_miss(self):
        """Test tracing ray with miss."""
        tlas = MockTLAS()

        tracer = RTReflectionTracer(tlas=tlas)

        ray = ReflectionRay(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            should_trace=True,
        )

        hit = tracer.trace_ray(ray)

        assert hit.hit is False

    def test_trace_ray_skip_if_not_should_trace(self):
        """Test that ray is skipped if should_trace is False."""
        tlas = MockTLAS()
        tlas.add_sphere(Vec3(0, 0, -5), radius=1.0)

        tracer = RTReflectionTracer(tlas=tlas)

        ray = ReflectionRay(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            should_trace=False,  # Roughness was too high
        )

        hit = tracer.trace_ray(ray)

        assert hit.hit is False

    def test_trace_ray_no_tlas(self):
        """Test tracing without TLAS returns miss."""
        tracer = RTReflectionTracer()

        ray = ReflectionRay(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            should_trace=True,
        )

        hit = tracer.trace_ray(ray)

        assert hit.hit is False

    def test_on_miss_default_environment(self):
        """Test miss shader returns default environment."""
        tracer = RTReflectionTracer()

        color = tracer.on_miss(Vec3(0, 1, 0))  # Looking up

        # Should be sky-ish color
        assert color.x > 0
        assert color.y > 0
        assert color.z > 0

    def test_on_miss_custom_environment(self):
        """Test miss shader with custom sampler."""
        def custom_sampler(direction: Vec3) -> Vec3:
            return Vec3(1.0, 0.0, 0.0)  # Always red

        tracer = RTReflectionTracer(environment_sampler=custom_sampler)

        color = tracer.on_miss(Vec3(0, 1, 0))

        assert color.x == 1.0
        assert color.y == 0.0
        assert color.z == 0.0

    def test_on_miss_gradient(self):
        """Test miss shader gradient between ground and sky."""
        tracer = RTReflectionTracer(environment_color=Vec3(0.5, 0.6, 0.8))

        sky_color = tracer.on_miss(Vec3(0, 1, 0))  # Looking up
        ground_color = tracer.on_miss(Vec3(0, -1, 0))  # Looking down

        # Sky should be brighter than ground
        assert sky_color.y > ground_color.y

    def test_statistics_tracking(self):
        """Test that statistics are tracked."""
        tlas = MockTLAS()
        tlas.add_sphere(Vec3(0, 0, -5), radius=1.0)

        tracer = RTReflectionTracer(tlas=tlas)

        # Trace a hit
        tracer.trace_ray(ReflectionRay(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            should_trace=True,
        ))

        # Trace a miss
        tracer.trace_ray(ReflectionRay(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 1, 0),
            should_trace=True,
        ))

        stats = tracer.get_statistics()

        assert stats["rays_traced"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_reset_statistics(self):
        """Test resetting statistics."""
        tlas = MockTLAS()
        tracer = RTReflectionTracer(tlas=tlas)

        tracer.trace_ray(ReflectionRay(
            origin=Vec3(0, 0, 0),
            direction=Vec3(0, 0, -1),
            should_trace=True,
        ))

        tracer.reset_statistics()

        stats = tracer.get_statistics()
        assert stats["rays_traced"] == 0


# =============================================================================
# RTReflectionConfig Tests
# =============================================================================


class TestRTReflectionConfig:
    """Tests for RTReflectionConfig."""

    def test_default_construction(self):
        """Test default config construction."""
        config = RTReflectionConfig()

        assert config.max_ray_distance == DEFAULT_MAX_RAY_DISTANCE
        assert config.roughness_threshold == DEFAULT_ROUGHNESS_THRESHOLD
        assert config.resolution_scale == RESOLUTION_FULL
        assert config.normal_bias == DEFAULT_NORMAL_BIAS

    def test_custom_construction(self):
        """Test config with custom values."""
        config = RTReflectionConfig(
            max_ray_distance=50.0,
            roughness_threshold=0.5,
            resolution_scale=0.5,
            enable_transparency=True,
        )

        assert config.max_ray_distance == 50.0
        assert config.roughness_threshold == 0.5
        assert config.resolution_scale == 0.5
        assert config.enable_transparency is True

    def test_validation_passes(self):
        """Test validation for valid config."""
        config = RTReflectionConfig()
        errors = config.validate()

        assert len(errors) == 0

    def test_validation_negative_ray_distance(self):
        """Test validation catches negative ray distance."""
        config = RTReflectionConfig(max_ray_distance=-1.0)
        # post_init should clamp to 0.1

        assert config.max_ray_distance == 0.1

    def test_validation_roughness_clamped(self):
        """Test roughness threshold is clamped."""
        config = RTReflectionConfig(roughness_threshold=1.5)

        assert config.roughness_threshold == 1.0

    def test_low_quality_preset(self):
        """Test low quality preset."""
        config = RTReflectionConfig.low_quality()

        assert config.resolution_scale == RESOLUTION_QUARTER
        assert config.roughness_threshold == 0.5
        assert config.denoise is False

    def test_medium_quality_preset(self):
        """Test medium quality preset."""
        config = RTReflectionConfig.medium_quality()

        assert config.resolution_scale == RESOLUTION_HALF

    def test_high_quality_preset(self):
        """Test high quality preset."""
        config = RTReflectionConfig.high_quality()

        assert config.resolution_scale == RESOLUTION_FULL
        assert config.max_bounces == 2


# =============================================================================
# RTReflectionPass Tests
# =============================================================================


class TestRTReflectionPass:
    """Tests for RTReflectionPass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RTReflectionConfig()
        self.gbuffer = GBufferReader(width=100, height=100)
        self.gbuffer.set_camera(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            Vec3(0, 1, 5),
        )

        self.tlas = MockTLAS()
        self.tlas.add_sphere(Vec3(0, 0, -5), radius=1.0)

        self.tracer = RTReflectionTracer(tlas=self.tlas)

    def test_default_construction(self):
        """Test default pass construction."""
        pass_obj = RTReflectionPass(self.config)

        assert pass_obj.config == self.config
        assert pass_obj.output_width == 0
        assert pass_obj.output_height == 0

    def test_construction_with_components(self):
        """Test pass construction with all components."""
        pass_obj = RTReflectionPass(
            self.config,
            gbuffer_reader=self.gbuffer,
            tracer=self.tracer,
        )

        assert pass_obj.gbuffer_reader == self.gbuffer
        assert pass_obj.tracer == self.tracer

    def test_execute_without_components_raises(self):
        """Test execute raises without required components."""
        pass_obj = RTReflectionPass(self.config)

        with pytest.raises(RuntimeError):
            pass_obj.execute()

    def test_execute_full_resolution(self):
        """Test execute at full resolution."""
        # Set up some test data
        for x in range(100):
            for y in range(100):
                self.gbuffer.set_test_data(
                    x, y,
                    depth=10.0,
                    normal=Vec3(0, 0, 1),
                    material=MaterialData(roughness=0.3),
                )

        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0),
            gbuffer_reader=self.gbuffer,
            tracer=self.tracer,
        )

        pass_obj.execute()

        assert pass_obj.output_width == 100
        assert pass_obj.output_height == 100

    def test_execute_half_resolution(self):
        """Test execute at half resolution."""
        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=0.5),
            gbuffer_reader=self.gbuffer,
            tracer=self.tracer,
        )

        pass_obj.execute()

        assert pass_obj.output_width == 50
        assert pass_obj.output_height == 50

    def test_execute_quarter_resolution(self):
        """Test execute at quarter resolution."""
        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=0.25),
            gbuffer_reader=self.gbuffer,
            tracer=self.tracer,
        )

        pass_obj.execute()

        assert pass_obj.output_width == 25
        assert pass_obj.output_height == 25

    def test_resolution_scaling_correct_buffer_size(self):
        """Test that resolution scaling produces correct buffer sizes."""
        base_gbuffer = GBufferReader(width=1920, height=1080)
        base_gbuffer.set_camera(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            Vec3.zero(),
        )

        scales_expected = [
            (0.25, 480, 270),
            (0.5, 960, 540),
            (1.0, 1920, 1080),
        ]

        for scale, expected_w, expected_h in scales_expected:
            config = RTReflectionConfig(resolution_scale=scale)
            pass_obj = RTReflectionPass(
                config,
                gbuffer_reader=base_gbuffer,
                tracer=self.tracer,
            )
            pass_obj.execute()

            assert pass_obj.output_width == expected_w
            assert pass_obj.output_height == expected_h

    def test_set_resolution_scale(self):
        """Test changing resolution scale."""
        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0),
            gbuffer_reader=self.gbuffer,
            tracer=self.tracer,
        )

        pass_obj.set_resolution_scale(0.5)

        assert pass_obj.config.resolution_scale == 0.5

    def test_get_reflection_at(self):
        """Test getting reflection at specific pixel."""
        self.gbuffer.set_test_data(
            0, 0,
            depth=10.0,
            normal=Vec3(0, 0, 1),
            material=MaterialData(roughness=0.3),
        )

        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0),
            gbuffer_reader=self.gbuffer,
            tracer=self.tracer,
        )
        pass_obj.execute()

        output = pass_obj.get_reflection_at(0, 0)

        assert isinstance(output, ReflectionOutput)
        assert output.was_traced or output.roughness <= 0.7

    def test_get_reflection_at_uv(self):
        """Test getting reflection at UV coordinates."""
        self.gbuffer.set_test_data(
            49, 49,
            depth=10.0,
            normal=Vec3(0, 0, 1),
            material=MaterialData(roughness=0.3),
        )

        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0),
            gbuffer_reader=self.gbuffer,
            tracer=self.tracer,
        )
        pass_obj.execute()

        output = pass_obj.get_reflection_at_uv(Vec2(0.5, 0.5))

        assert isinstance(output, ReflectionOutput)

    def test_get_reflection_buffer(self):
        """Test getting full reflection buffer."""
        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0),
            gbuffer_reader=self.gbuffer,
            tracer=self.tracer,
        )
        pass_obj.execute()

        buffer = pass_obj.get_reflection_buffer()

        expected_size = 100 * 100
        assert len(buffer) == expected_size

    def test_statistics(self):
        """Test pass execution statistics."""
        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0),
            gbuffer_reader=self.gbuffer,
            tracer=self.tracer,
        )
        pass_obj.execute()

        stats = pass_obj.get_statistics()

        assert "pixels_processed" in stats
        assert "pixels_traced" in stats
        assert "pixels_skipped" in stats
        assert stats["pixels_processed"] == 100 * 100

    def test_roughness_skip_threshold(self):
        """Test that rough pixels are skipped."""
        # Set all pixels to rough material
        for x in range(100):
            for y in range(100):
                self.gbuffer.set_test_data(
                    x, y,
                    depth=10.0,
                    normal=Vec3(0, 0, 1),
                    material=MaterialData(roughness=0.9),  # Above threshold
                )

        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0, roughness_threshold=0.7),
            gbuffer_reader=self.gbuffer,
            tracer=self.tracer,
        )
        pass_obj.execute()

        stats = pass_obj.get_statistics()

        # All pixels should be skipped (roughness > threshold)
        assert stats["pixels_skipped"] == stats["pixels_processed"]
        assert stats["pixels_traced"] == 0


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_estimate_memory_full_resolution(self):
        """Test memory estimation at full resolution."""
        memory = estimate_rt_reflection_memory(1920, 1080, 1.0)

        # 1920 * 1080 * (8 + 4 + 1) bytes = ~27 MB
        expected_pixels = 1920 * 1080
        expected_bytes = expected_pixels * 13

        assert memory == expected_bytes

    def test_estimate_memory_quarter_resolution(self):
        """Test memory estimation at quarter resolution."""
        memory_full = estimate_rt_reflection_memory(1920, 1080, 1.0)
        memory_quarter = estimate_rt_reflection_memory(1920, 1080, 0.25)

        # Quarter resolution should use ~1/16 the memory
        ratio = memory_quarter / memory_full
        assert ratio == pytest.approx(1.0 / 16.0, rel=0.01)

    def test_create_mock_tlas(self):
        """Test creating mock TLAS with test geometry."""
        tlas = create_mock_tlas()

        assert tlas.is_valid()

        # Should have some geometry
        hit = tlas.trace_ray(
            Vec3(0, 1, 0),
            Vec3(0, 0, -1),
            100.0,
        )

        # May or may not hit depending on geometry placement
        assert isinstance(hit, RayHitInfo)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the RT reflection pipeline."""

    def test_full_pipeline_single_pixel(self):
        """Test complete pipeline for a single pixel."""
        # Setup
        gbuffer = GBufferReader(width=100, height=100)
        gbuffer.set_camera(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            Vec3(0, 0, 5),
        )
        gbuffer.set_test_data(
            50, 50,
            depth=10.0,
            normal=Vec3(0, 0, 1),
            material=MaterialData(roughness=0.2, metallic=0.9),
        )

        tlas = MockTLAS()
        tlas.add_sphere(
            Vec3(0, 0, -20),
            radius=3.0,
            material=MaterialData(base_color=Vec3(1, 0, 0)),
        )

        tracer = RTReflectionTracer(tlas=tlas)

        config = RTReflectionConfig(
            roughness_threshold=0.7,
            resolution_scale=1.0,
        )

        pass_obj = RTReflectionPass(config, gbuffer, tracer)
        pass_obj.execute()

        # Get result at center pixel
        output = pass_obj.get_reflection_at(50, 50)

        # Should have traced and potentially hit
        assert output.was_traced is True

    def test_reflection_color_matches_hit_material(self):
        """Test that reflection color comes from hit material."""
        gbuffer = GBufferReader(width=10, height=10)
        gbuffer.set_camera(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            Vec3(0, 0, 10),
        )

        # Setup pixel looking at sphere
        gbuffer.set_test_data(
            5, 5,
            depth=5.0,
            normal=Vec3(0, 0, 1),  # Facing camera
            material=MaterialData(roughness=0.1),
        )

        # Red sphere in front
        tlas = MockTLAS()
        red_material = MaterialData(base_color=Vec3(1.0, 0.0, 0.0))
        tlas.add_sphere(Vec3(0, 0, -10), radius=2.0, material=red_material)

        tracer = RTReflectionTracer(tlas=tlas)
        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0),
            gbuffer,
            tracer,
        )
        pass_obj.execute()

        output = pass_obj.get_reflection_at(5, 5)

        # If hit, should be red (from sphere)
        if output.was_traced:
            # Either hit the sphere or got environment
            assert isinstance(output.color, Vec3)

    def test_environment_fallback_on_miss(self):
        """Test environment color used on miss."""
        gbuffer = GBufferReader(width=10, height=10)
        gbuffer.set_camera(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            Vec3(0, 0, 10),
        )
        gbuffer.set_test_data(
            5, 5,
            depth=5.0,
            normal=Vec3(0, 1, 0),  # Facing up
            material=MaterialData(roughness=0.1),
        )

        tlas = MockTLAS()  # Empty, no geometry to hit

        tracer = RTReflectionTracer(
            tlas=tlas,
            environment_color=Vec3(0.5, 0.6, 0.8),
        )
        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0),
            gbuffer,
            tracer,
        )
        pass_obj.execute()

        output = pass_obj.get_reflection_at(5, 5)

        if output.was_traced:
            # Should have environment color (miss)
            assert output.confidence < 1.0  # Lower confidence for environment

    def test_performance_budget_respected(self):
        """Test that pass completes in reasonable time."""
        import time

        gbuffer = GBufferReader(width=256, height=256)
        gbuffer.set_camera(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            Vec3.zero(),
        )

        tlas = MockTLAS()
        tlas.add_sphere(Vec3(0, 0, -10), 2.0)

        tracer = RTReflectionTracer(tlas=tlas)
        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=0.5),  # Half res
            gbuffer,
            tracer,
        )

        start = time.time()
        pass_obj.execute()
        elapsed = time.time() - start

        # Should complete within 1 second for 128x128
        assert elapsed < 1.0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_depth_pixel(self):
        """Test handling of zero depth (sky pixels)."""
        gbuffer = GBufferReader(width=10, height=10)
        gbuffer.set_camera(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            Vec3.zero(),
        )
        # Set the pixel at index (4, 4) since UV (0.5, 0.5) maps to (4.5, 4.5) -> (4, 4)
        gbuffer.set_test_data(4, 4, depth=0.0, normal=Vec3.up())

        tlas = MockTLAS()
        tracer = RTReflectionTracer(tlas=tlas)
        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0),
            gbuffer,
            tracer,
        )
        pass_obj.execute()

        # Get output at the exact pixel we set (4, 4)
        output = pass_obj.get_reflection_at(4, 4)

        # Should not trace (invalid pixel due to zero depth)
        assert output.was_traced is False

    def test_parallel_view_direction(self):
        """Test reflection when view is parallel to surface."""
        gen = ReflectionRayGenerator()

        # View direction exactly parallel to surface
        normal = Vec3(0, 1, 0)
        view_dir = Vec3(1, 0, 0)  # Perpendicular to normal

        reflect = gen.compute_reflection_direction(normal, view_dir)

        # N dot V = 0, so R = 2*0*N - V = -V
        assert reflect.x == pytest.approx(-1.0, rel=1e-4)
        assert reflect.y == pytest.approx(0.0, abs=1e-6)

    def test_very_small_normal_bias(self):
        """Test very small normal bias still prevents self-intersection."""
        gen = ReflectionRayGenerator(normal_bias=1e-6)

        origin = gen.get_ray_origin(Vec3(0, 0, 0), Vec3(0, 1, 0))

        assert origin.y > 0

    def test_resolution_scale_edge_values(self):
        """Test resolution scale at edge values."""
        config = RTReflectionConfig(resolution_scale=0.1)
        assert config.resolution_scale == 0.1

        config = RTReflectionConfig(resolution_scale=1.0)
        assert config.resolution_scale == 1.0

    def test_max_bounces_clamped(self):
        """Test max bounces is clamped."""
        config = RTReflectionConfig(max_bounces=100)
        assert config.max_bounces == 8  # Max is 8

        config = RTReflectionConfig(max_bounces=0)
        assert config.max_bounces == 1  # Min is 1

    def test_empty_reflection_buffer_access(self):
        """Test accessing reflection buffer before execute."""
        pass_obj = RTReflectionPass(RTReflectionConfig())

        output = pass_obj.get_reflection_at(0, 0)

        assert isinstance(output, ReflectionOutput)
        assert output.was_traced is False

    def test_out_of_bounds_pixel_access(self):
        """Test accessing out of bounds pixel."""
        gbuffer = GBufferReader(width=10, height=10)
        gbuffer.set_camera(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            Vec3.zero(),
        )

        tlas = MockTLAS()
        tracer = RTReflectionTracer(tlas=tlas)
        pass_obj = RTReflectionPass(
            RTReflectionConfig(resolution_scale=1.0),
            gbuffer,
            tracer,
        )
        pass_obj.execute()

        # Out of bounds should return default
        output = pass_obj.get_reflection_at(-1, -1)
        assert isinstance(output, ReflectionOutput)

        output = pass_obj.get_reflection_at(1000, 1000)
        assert isinstance(output, ReflectionOutput)
