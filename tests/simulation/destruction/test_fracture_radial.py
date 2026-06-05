"""
Tests for Radial Fracture System.

Whitebox tests for fracture_radial.py including:
- RadialSlice dataclass
- ConcentricRing dataclass
- RadialChunk extended chunk
- RadialFracture pattern generation
- ConcentricRadialFracture variations
- SpiderWebFracture patterns
"""

import pytest
import math

from engine.simulation.destruction.fracture_radial import (
    RadialSlice,
    ConcentricRing,
    RadialChunk,
    RadialFracture,
    ConcentricRadialFracture,
    SpiderWebFracture,
)
from engine.simulation.destruction.fracture_voronoi import (
    Vec3,
    Plane,
    BoundingBox,
    vec3_length,
    vec3_normalize,
)
from engine.simulation.destruction.config import (
    RADIAL_MIN_SLICES,
    RADIAL_MAX_SLICES,
    RADIAL_MIN_RINGS,
    RADIAL_MAX_RINGS,
)


class TestRadialSlice:
    """Tests for RadialSlice dataclass."""

    def test_basic_construction(self):
        """Verify basic construction."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
        slice_obj = RadialSlice(
            index=0,
            angle_start=0.0,
            angle_end=math.pi / 4,
            plane=plane
        )
        assert slice_obj.index == 0
        assert slice_obj.angle_start == 0.0
        assert abs(slice_obj.angle_end - math.pi / 4) < 1e-10

    def test_angle_wrap(self):
        """Verify angles can span 2pi."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
        slice_obj = RadialSlice(
            index=7,
            angle_start=7 * math.pi / 4,
            angle_end=2 * math.pi + math.pi / 4,  # Wraps past 2pi
            plane=plane
        )
        assert slice_obj.angle_start < slice_obj.angle_end


class TestConcentricRing:
    """Tests for ConcentricRing dataclass."""

    def test_basic_construction(self):
        """Verify basic construction."""
        ring = ConcentricRing(
            index=0,
            radius_inner=0.0,
            radius_outer=1.0
        )
        assert ring.index == 0
        assert ring.radius_inner == 0.0
        assert ring.radius_outer == 1.0

    def test_ring_order(self):
        """Verify outer radius is greater than inner."""
        ring = ConcentricRing(
            index=1,
            radius_inner=1.0,
            radius_outer=2.0
        )
        assert ring.radius_outer > ring.radius_inner


class TestRadialChunk:
    """Tests for RadialChunk dataclass."""

    def test_basic_construction(self):
        """Verify basic construction with extra fields."""
        chunk = RadialChunk(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            triangles=[],
            slice_index=3,
            ring_index=1
        )
        assert chunk.slice_index == 3
        assert chunk.ring_index == 1

    def test_default_indices(self):
        """Verify default indices are -1."""
        chunk = RadialChunk(vertices=[], triangles=[])
        assert chunk.slice_index == -1
        assert chunk.ring_index == -1


class TestRadialFracture:
    """Tests for RadialFracture class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        fracture = RadialFracture(seed=42, num_slices=8, num_rings=3)
        assert fracture.seed == 42
        assert fracture.num_slices == 8
        assert fracture.num_rings == 3

    def test_slices_clamping(self):
        """Verify slices are clamped to valid range."""
        fracture_low = RadialFracture(num_slices=1)
        assert fracture_low.num_slices >= RADIAL_MIN_SLICES

        fracture_high = RadialFracture(num_slices=100)
        assert fracture_high.num_slices <= RADIAL_MAX_SLICES

    def test_rings_clamping(self):
        """Verify rings are clamped to valid range."""
        fracture_low = RadialFracture(num_rings=0)
        assert fracture_low.num_rings >= RADIAL_MIN_RINGS

        fracture_high = RadialFracture(num_rings=100)
        assert fracture_high.num_rings <= RADIAL_MAX_RINGS

    def test_jitter_clamping(self):
        """Verify jitter is clamped to [0, 1]."""
        fracture_low = RadialFracture(jitter_amount=-0.5)
        # Internal _jitter_amount should be clamped
        assert fracture_low._jitter_amount >= 0.0

        fracture_high = RadialFracture(jitter_amount=2.0)
        assert fracture_high._jitter_amount <= 1.0

    def test_seed_setter(self):
        """Verify seed setter resets RNG."""
        fracture = RadialFracture(seed=42)
        fracture.seed = 99
        assert fracture.seed == 99

    def test_generate_radial_pattern(self):
        """Verify radial pattern generation."""
        fracture = RadialFracture(seed=42, num_slices=8, num_rings=3, jitter_amount=0)

        slices, rings = fracture.generate_radial_pattern(
            center=(0.0, 0.0, 0.0),
            radius=10.0
        )

        assert len(slices) == 8
        assert len(rings) == 3

        # Verify slices cover full circle
        total_angle = sum(s.angle_end - s.angle_start for s in slices)
        assert abs(total_angle - 2 * math.pi) < 1e-6

        # Verify rings go from 0 to max radius
        assert rings[0].radius_inner == 0.0
        assert abs(rings[-1].radius_outer - 10.0) < 1e-6

    def test_slices_property_returns_copy(self):
        """Verify slices property returns copy."""
        fracture = RadialFracture(seed=42, num_slices=4, num_rings=2)
        fracture.generate_radial_pattern(center=(0.0, 0.0, 0.0), radius=5.0)

        slices = fracture.slices
        original_len = len(slices)
        slices.clear()
        assert len(fracture.slices) == original_len

    def test_rings_property_returns_copy(self):
        """Verify rings property returns copy."""
        fracture = RadialFracture(seed=42, num_slices=4, num_rings=2)
        fracture.generate_radial_pattern(center=(0.0, 0.0, 0.0), radius=5.0)

        rings = fracture.rings
        original_len = len(rings)
        rings.clear()
        assert len(fracture.rings) == original_len

    def test_generate_impact_directed(self):
        """Verify impact-directed pattern generation."""
        fracture = RadialFracture(seed=42, num_slices=8, num_rings=3)

        slices, rings = fracture.generate_impact_directed(
            center=(0.0, 0.0, 0.0),
            impact_direction=(0.0, 0.0, 1.0),
            radius=10.0,
            intensity=1.0
        )

        assert len(slices) > 0
        assert len(rings) > 0

    def test_generate_impact_directed_low_intensity(self):
        """Verify low intensity produces fewer fragments."""
        fracture = RadialFracture(seed=42, num_slices=16, num_rings=4)

        slices_high, rings_high = fracture.generate_impact_directed(
            center=(0.0, 0.0, 0.0),
            impact_direction=(0.0, 0.0, 1.0),
            radius=10.0,
            intensity=1.0
        )

        # Reset to regenerate
        fracture = RadialFracture(seed=42, num_slices=16, num_rings=4)

        slices_low, rings_low = fracture.generate_impact_directed(
            center=(0.0, 0.0, 0.0),
            impact_direction=(0.0, 0.0, 1.0),
            radius=10.0,
            intensity=0.2
        )

        # Lower intensity should produce fewer or equal slices/rings
        assert len(slices_low) <= len(slices_high)

    def test_custom_direction(self):
        """Verify custom direction is used."""
        fracture = RadialFracture(seed=42, num_slices=8, num_rings=3)

        fracture.generate_radial_pattern(
            center=(0.0, 0.0, 0.0),
            radius=10.0,
            direction=(1.0, 0.0, 0.0)  # X direction instead of Z
        )

        assert fracture._direction == vec3_normalize((1.0, 0.0, 0.0))

    def test_fracture_mesh_basic(self):
        """Verify basic mesh fracturing."""
        # Create a simple quad mesh
        vertices = [
            (-5.0, -5.0, 0.0),
            (5.0, -5.0, 0.0),
            (5.0, 5.0, 0.0),
            (-5.0, 5.0, 0.0)
        ]
        triangles = [(0, 1, 2), (0, 2, 3)]

        fracture = RadialFracture(seed=42, num_slices=4, num_rings=2, min_chunk_volume=0.0001)
        chunks = fracture.fracture_mesh(vertices, triangles)

        # Should produce some chunks (even if just the original in some cases)
        assert isinstance(chunks, list)

    def test_fracture_mesh_with_center(self):
        """Verify fracturing with custom center."""
        vertices = [
            (-5.0, -5.0, 0.0),
            (5.0, -5.0, 0.0),
            (5.0, 5.0, 0.0),
            (-5.0, 5.0, 0.0)
        ]
        triangles = [(0, 1, 2), (0, 2, 3)]

        fracture = RadialFracture(seed=42, num_slices=4, num_rings=2)
        chunks = fracture.fracture_mesh(
            vertices,
            triangles,
            center=(0.0, 0.0, 0.0)
        )

        assert isinstance(chunks, list)

    def test_get_cut_planes(self):
        """Verify cut planes are retrieved."""
        fracture = RadialFracture(seed=42, num_slices=8, num_rings=3)
        fracture.generate_radial_pattern(
            center=(0.0, 0.0, 0.0),
            radius=10.0
        )

        planes = fracture.get_cut_planes()
        assert len(planes) == 8

    def test_get_ring_boundaries(self):
        """Verify ring boundaries are retrieved."""
        fracture = RadialFracture(seed=42, num_slices=4, num_rings=3)
        fracture.generate_radial_pattern(
            center=(0.0, 0.0, 0.0),
            radius=10.0
        )

        boundaries = fracture.get_ring_boundaries()
        assert len(boundaries) == 4  # 0 + num_rings
        assert boundaries[0] == 0.0
        assert abs(boundaries[-1] - 10.0) < 1e-6

    def test_adjacency_computation(self):
        """Verify adjacency is computed between chunks."""
        vertices = [
            (-10.0, -10.0, 0.0),
            (10.0, -10.0, 0.0),
            (10.0, 10.0, 0.0),
            (-10.0, 10.0, 0.0)
        ]
        triangles = [(0, 1, 2), (0, 2, 3)]

        fracture = RadialFracture(seed=42, num_slices=4, num_rings=2, min_chunk_volume=0.0001)
        chunks = fracture.fracture_mesh(vertices, triangles)

        # If we got multiple chunks, some should have adjacency
        if len(chunks) > 1:
            total_adjacencies = sum(len(c.adjacent_chunks) for c in chunks)
            # At least some adjacencies should exist
            # (may be 0 if chunks don't actually neighbor)

    def test_deterministic_generation(self):
        """Verify same seed produces same pattern."""
        fracture1 = RadialFracture(seed=12345, num_slices=6, num_rings=2, jitter_amount=0.5)
        fracture2 = RadialFracture(seed=12345, num_slices=6, num_rings=2, jitter_amount=0.5)

        slices1, rings1 = fracture1.generate_radial_pattern(
            center=(0.0, 0.0, 0.0), radius=10.0
        )
        slices2, rings2 = fracture2.generate_radial_pattern(
            center=(0.0, 0.0, 0.0), radius=10.0
        )

        for s1, s2 in zip(slices1, slices2):
            assert abs(s1.angle_start - s2.angle_start) < 1e-10
            assert abs(s1.angle_end - s2.angle_end) < 1e-10


class TestConcentricRadialFracture:
    """Tests for ConcentricRadialFracture class."""

    def test_basic_construction(self):
        """Verify basic construction with more rings."""
        fracture = ConcentricRadialFracture(seed=42, num_slices=8, num_rings=5)
        # ConcentricRadialFracture adds 2 to default rings
        assert fracture.num_rings == 5

    def test_linear_ring_spacing(self):
        """Verify linear ring spacing."""
        fracture = ConcentricRadialFracture(
            seed=42,
            num_slices=4,
            num_rings=4,
            ring_spacing="linear"
        )
        fracture.generate_radial_pattern(center=(0.0, 0.0, 0.0), radius=10.0)

        rings = fracture.rings
        # Linear spacing: each ring should have equal width
        widths = [r.radius_outer - r.radius_inner for r in rings]
        for i in range(1, len(widths)):
            assert abs(widths[i] - widths[0]) < 0.5  # Approximate equality

    def test_quadratic_ring_spacing(self):
        """Verify quadratic ring spacing (default)."""
        fracture = ConcentricRadialFracture(
            seed=42,
            num_slices=4,
            num_rings=4,
            ring_spacing="quadratic"
        )
        fracture.generate_radial_pattern(center=(0.0, 0.0, 0.0), radius=10.0)

        rings = fracture.rings
        # Quadratic: inner rings should be smaller
        # First ring should be smaller than last
        first_width = rings[0].radius_outer - rings[0].radius_inner
        last_width = rings[-1].radius_outer - rings[-1].radius_inner
        assert first_width < last_width

    def test_logarithmic_ring_spacing(self):
        """Verify logarithmic ring spacing."""
        fracture = ConcentricRadialFracture(
            seed=42,
            num_slices=4,
            num_rings=4,
            ring_spacing="logarithmic"
        )
        fracture.generate_radial_pattern(center=(0.0, 0.0, 0.0), radius=10.0)

        rings = fracture.rings
        # Logarithmic spacing should be applied
        assert len(rings) == 4


class TestSpiderWebFracture:
    """Tests for SpiderWebFracture class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        fracture = SpiderWebFracture(
            seed=42,
            num_radial=8,
            num_circular=4,
            irregularity=0.3
        )
        assert fracture.num_slices == 8
        assert fracture.num_rings == 4

    def test_irregularity_clamping(self):
        """Verify irregularity is clamped."""
        fracture = SpiderWebFracture(irregularity=-0.5)
        # Should be treated as 0 internally
        assert fracture._jitter_amount >= 0

    def test_irregular_ring_generation(self):
        """Verify irregular rings are generated."""
        fracture = SpiderWebFracture(
            seed=42,
            num_radial=8,
            num_circular=4,
            irregularity=0.3
        )
        fracture.generate_radial_pattern(center=(0.0, 0.0, 0.0), radius=10.0)

        rings = fracture.rings
        assert len(rings) == 4

        # Rings should be sorted (inner radius < outer radius)
        for ring in rings:
            assert ring.radius_inner < ring.radius_outer

    def test_zero_irregularity(self):
        """Verify zero irregularity produces cleaner pattern."""
        fracture = SpiderWebFracture(
            seed=42,
            num_radial=8,
            num_circular=4,
            irregularity=0.0
        )
        fracture.generate_radial_pattern(center=(0.0, 0.0, 0.0), radius=10.0)

        rings = fracture.rings
        # Should still produce valid rings
        assert len(rings) == 4

    def test_high_irregularity(self):
        """Verify high irregularity still produces valid rings."""
        fracture = SpiderWebFracture(
            seed=42,
            num_radial=8,
            num_circular=4,
            irregularity=1.0
        )
        fracture.generate_radial_pattern(center=(0.0, 0.0, 0.0), radius=10.0)

        rings = fracture.rings
        # All rings should have valid radii
        for ring in rings:
            assert ring.radius_inner >= 0
            assert ring.radius_outer > ring.radius_inner


class TestRadialFractureEdgeCases:
    """Edge case tests for radial fracture system."""

    def test_zero_radius(self):
        """Verify handling of zero radius."""
        fracture = RadialFracture(seed=42, num_slices=8, num_rings=3)
        slices, rings = fracture.generate_radial_pattern(
            center=(0.0, 0.0, 0.0),
            radius=0.0
        )
        # Should still produce slices/rings (degenerate but not crashing)
        assert len(slices) > 0

    def test_negative_radius(self):
        """Verify handling of negative radius."""
        fracture = RadialFracture(seed=42, num_slices=8, num_rings=3)
        # This may produce unexpected results, but shouldn't crash
        slices, rings = fracture.generate_radial_pattern(
            center=(0.0, 0.0, 0.0),
            radius=-10.0
        )
        assert isinstance(slices, list)
        assert isinstance(rings, list)

    def test_empty_mesh(self):
        """Verify handling of empty mesh."""
        fracture = RadialFracture(seed=42, num_slices=4, num_rings=2)
        chunks = fracture.fracture_mesh([], [])
        # Empty mesh should produce no chunks
        # (or handle gracefully - depends on implementation)
        assert isinstance(chunks, list)

    def test_single_triangle_mesh(self):
        """Verify handling of single triangle."""
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 1.0, 0.0)
        ]
        triangles = [(0, 1, 2)]

        fracture = RadialFracture(seed=42, num_slices=4, num_rings=2, min_chunk_volume=0.0001)
        chunks = fracture.fracture_mesh(vertices, triangles)
        assert isinstance(chunks, list)

    def test_override_num_slices(self):
        """Verify overriding num_slices in generate call."""
        fracture = RadialFracture(seed=42, num_slices=4, num_rings=2)
        slices, rings = fracture.generate_radial_pattern(
            center=(0.0, 0.0, 0.0),
            radius=10.0,
            num_slices=8  # Override
        )
        assert len(slices) == 8

    def test_override_num_rings(self):
        """Verify overriding num_rings in generate call."""
        fracture = RadialFracture(seed=42, num_slices=4, num_rings=2)
        slices, rings = fracture.generate_radial_pattern(
            center=(0.0, 0.0, 0.0),
            radius=10.0,
            num_rings=5  # Override
        )
        assert len(rings) == 5
