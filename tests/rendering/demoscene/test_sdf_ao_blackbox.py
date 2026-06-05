"""
Blackbox tests for SDF Ambient Occlusion (T-DEMO-3.5).

Tests the ambient occlusion calculation as a black box, verifying
observable behavior without knowledge of internal implementation.
Focuses on perceptual correctness and edge-case behavior.

BLACKBOX acceptance criteria:
  AC1: Crevices and corners are darkened
  AC2: Flat surfaces show minimal AO
  AC3: Perceptually correct for known scenes
  AC4: AO value always in [0, 1] range
  AC5: Consistent behavior across similar inputs
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.sdf_ao import (
    AOConfig,
    calculate_ao,
    calculate_ao_multi_direction,
    generate_ao_wgsl,
    make_scene_ao_evaluator,
)


# =============================================================================
# Test Scene SDFs
# =============================================================================

def sdf_sphere(p: tuple[float, float, float], radius: float = 1.0,
               center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> float:
    """Sphere SDF."""
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    dz = p[2] - center[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz) - radius


def sdf_ground_plane(p: tuple[float, float, float]) -> float:
    """Ground plane at y=0."""
    return p[1]


def sdf_box(p: tuple[float, float, float],
            half: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """Axis-aligned box SDF."""
    qx = abs(p[0]) - half[0]
    qy = abs(p[1]) - half[1]
    qz = abs(p[2]) - half[2]
    outer = math.sqrt(max(qx, 0)**2 + max(qy, 0)**2 + max(qz, 0)**2)
    inner = min(max(qx, max(qy, qz)), 0.0)
    return outer + inner


def sdf_corner_room(p: tuple[float, float, float]) -> float:
    """Room corner: floor, wall-x, wall-z meeting at origin."""
    d_floor = p[1]       # y-plane
    d_wall_x = p[0]      # x-plane
    d_wall_z = p[2]      # z-plane
    return min(d_floor, d_wall_x, d_wall_z)


def sdf_sphere_on_plane(p: tuple[float, float, float]) -> float:
    """Sphere resting on ground plane."""
    d_plane = p[1]
    d_sphere = sdf_sphere(p, 0.5, (0.0, 0.5, 0.0))
    return min(d_plane, d_sphere)


def sdf_two_walls(p: tuple[float, float, float], angle: float = math.pi/4) -> float:
    """Two walls meeting at an angle."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    d1 = p[0] * cos_a + p[2] * sin_a
    d2 = p[0] * cos_a - p[2] * sin_a
    return min(d1, d2)


# =============================================================================
# AC1: Crevices and Corners are Darkened
# =============================================================================

class TestCrevicesAndCorners:
    """Verify crevices and corners have lower AO (darker)."""

    def test_room_corner_darkened(self):
        """Room corner (3 planes meeting) should have low AO."""
        p = (0.1, 0.1, 0.1)  # Near corner
        n = (1.0, 1.0, 1.0)
        n_len = math.sqrt(3)
        n = (n[0]/n_len, n[1]/n_len, n[2]/n_len)

        ao = calculate_ao(p, n, sdf_corner_room)

        assert ao < 0.7, f"Corner should be darkened, got ao={ao}"

    def test_deep_corner_very_dark(self):
        """Very deep in corner should be even darker."""
        p_near = (0.05, 0.05, 0.05)
        p_far = (0.3, 0.3, 0.3)
        n = (1.0, 1.0, 1.0)
        n_len = math.sqrt(3)
        n = (n[0]/n_len, n[1]/n_len, n[2]/n_len)

        ao_near = calculate_ao(p_near, n, sdf_corner_room)
        ao_far = calculate_ao(p_far, n, sdf_corner_room)

        assert ao_near < ao_far, f"Deeper corner should be darker: ao_near={ao_near}, ao_far={ao_far}"

    def test_wall_floor_edge(self):
        """Edge where wall meets floor should be darkened."""
        p = (0.05, 0.1, 1.0)  # Near wall-floor edge
        n = (1.0, 1.0, 0.0)
        n_len = math.sqrt(2)
        n = (n[0]/n_len, n[1]/n_len, n[2]/n_len)

        ao = calculate_ao(p, n, sdf_corner_room)

        assert ao < 0.85, f"Wall-floor edge should show some occlusion, got ao={ao}"

    def test_sphere_ground_contact(self):
        """Contact point between sphere and ground darkened."""
        # Point at base of sphere where it touches ground
        p = (0.0, 0.01, 0.0)  # Just above ground, near sphere
        n = (0.0, 1.0, 0.0)

        ao = calculate_ao(p, n, sdf_sphere_on_plane)

        # Contact shadows should darken this area
        assert ao < 0.9, f"Sphere-ground contact should show occlusion, got ao={ao}"

    def test_v_shaped_crevice(self):
        """V-shaped crevice bottom should be darkened."""
        p = (0.0, 0.1, 0.0)
        n = (0.0, 1.0, 0.0)

        def v_crevice(pos):
            return sdf_two_walls(pos, math.pi/6)  # Narrow angle

        ao = calculate_ao(p, n, v_crevice)

        assert ao < 0.8, f"V-crevice should be darkened, got ao={ao}"


# =============================================================================
# AC2: Flat Surfaces Show Minimal AO
# =============================================================================

class TestFlatSurfaces:
    """Verify flat surfaces have high AO (little occlusion)."""

    def test_flat_ground_plane(self):
        """Center of flat ground plane should have high AO."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)

        ao = calculate_ao(p, n, sdf_ground_plane)

        assert ao > 0.95, f"Flat plane should have high AO, got ao={ao}"

    def test_sphere_exterior(self):
        """Exterior of sphere facing outward should have high AO."""
        p = (1.0, 0.0, 0.0)
        n = (1.0, 0.0, 0.0)

        ao = calculate_ao(p, n, lambda pos: sdf_sphere(pos))

        assert ao > 0.95, f"Sphere exterior should have high AO, got ao={ao}"

    def test_large_flat_wall(self):
        """Large flat wall should have high AO."""
        def infinite_wall(pos):
            return pos[0]  # x-plane

        p = (0.0, 5.0, 5.0)  # On wall, away from edges
        n = (1.0, 0.0, 0.0)

        ao = calculate_ao(p, n, infinite_wall)

        assert ao > 0.95, f"Large flat wall should have high AO, got ao={ao}"

    def test_box_face_center(self):
        """Center of box face should have high AO."""
        p = (1.0, 0.0, 0.0)  # On +X face
        n = (1.0, 0.0, 0.0)

        ao = calculate_ao(p, n, lambda pos: sdf_box(pos))

        assert ao > 0.9, f"Box face center should have high AO, got ao={ao}"


# =============================================================================
# AC3: Perceptually Correct for Known Scenes
# =============================================================================

class TestPerceptualCorrectness:
    """Verify AO is perceptually correct for well-understood scenes."""

    def test_ao_gradient_from_corner(self):
        """AO should gradually increase from corner to open space."""
        n = (1.0, 1.0, 1.0)
        n_len = math.sqrt(3)
        n = (n[0]/n_len, n[1]/n_len, n[2]/n_len)

        ao_values = []
        for dist in [0.05, 0.1, 0.2, 0.4, 0.8]:
            p = (dist, dist, dist)
            ao_values.append(calculate_ao(p, n, sdf_corner_room))

        # AO should generally increase as we move away from corner
        for i in range(len(ao_values) - 1):
            assert ao_values[i] <= ao_values[i+1] + 0.1, \
                f"AO should increase from corner: {ao_values}"

    def test_ao_symmetry(self):
        """AO should be symmetric for symmetric scenes."""
        n = (0.0, 1.0, 0.0)

        ao_1 = calculate_ao((1.0, 0.0, 0.0), n, sdf_ground_plane)
        ao_2 = calculate_ao((-1.0, 0.0, 0.0), n, sdf_ground_plane)
        ao_3 = calculate_ao((0.0, 0.0, 1.0), n, sdf_ground_plane)

        # All points on plane should have same AO
        assert ao_1 == pytest.approx(ao_2, abs=0.01)
        assert ao_1 == pytest.approx(ao_3, abs=0.01)

    def test_sphere_ao_uniform(self):
        """AO should be uniform around sphere surface."""
        ao_values = []
        for angle in [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2]:
            x = math.cos(angle)
            z = math.sin(angle)
            p = (x, 0.0, z)
            n = (x, 0.0, z)
            ao_values.append(calculate_ao(p, n, lambda pos: sdf_sphere(pos)))

        # All values should be similar
        ao_min = min(ao_values)
        ao_max = max(ao_values)
        assert ao_max - ao_min < 0.1, f"Sphere AO should be uniform: {ao_values}"

    def test_box_edge_darker_than_face(self):
        """Box edges should be darker than face centers."""
        # Face center
        p_face = (1.0, 0.0, 0.0)
        n_face = (1.0, 0.0, 0.0)

        # Edge (where two faces meet)
        p_edge = (1.0, 1.0, 0.0)
        n_edge = (1.0, 1.0, 0.0)
        n_len = math.sqrt(2)
        n_edge = (n_edge[0]/n_len, n_edge[1]/n_len, n_edge[2]/n_len)

        ao_face = calculate_ao(p_face, n_face, lambda pos: sdf_box(pos))
        ao_edge = calculate_ao(p_edge, n_edge, lambda pos: sdf_box(pos))

        assert ao_edge < ao_face, f"Edge should be darker: ao_edge={ao_edge}, ao_face={ao_face}"

    def test_box_corner_darkest(self):
        """Box corners should be darkest."""
        # Face center
        ao_face = calculate_ao(
            (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            lambda pos: sdf_box(pos)
        )

        # Corner
        p_corner = (1.0, 1.0, 1.0)
        n_corner = (1.0, 1.0, 1.0)
        n_len = math.sqrt(3)
        n_corner = (n_corner[0]/n_len, n_corner[1]/n_len, n_corner[2]/n_len)
        ao_corner = calculate_ao(p_corner, n_corner, lambda pos: sdf_box(pos))

        assert ao_corner < ao_face, f"Corner should be darker than face"


# =============================================================================
# AC4: AO Value Always in [0, 1] Range
# =============================================================================

class TestAORange:
    """Verify AO always returns values in [0, 1]."""

    def test_ao_bounds_normal_cases(self):
        """Normal cases produce AO in [0, 1]."""
        test_cases = [
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), sdf_ground_plane),
            ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), lambda p: sdf_sphere(p)),
            ((0.1, 0.1, 0.1), (0.577, 0.577, 0.577), sdf_corner_room),
        ]

        for p, n, sdf in test_cases:
            ao = calculate_ao(p, n, sdf)
            assert 0.0 <= ao <= 1.0, f"AO out of range: {ao} for p={p}"

    def test_ao_bounds_extreme_sdf_values(self):
        """Extreme SDF values still produce valid AO."""
        p = (0.0, 0.0, 0.0)
        n = (1.0, 0.0, 0.0)

        # Very large SDF
        ao_large = calculate_ao(p, n, lambda pos: 1000.0)
        assert 0.0 <= ao_large <= 1.0

        # Very small positive SDF
        ao_small = calculate_ao(p, n, lambda pos: 1e-10)
        assert 0.0 <= ao_small <= 1.0

        # Negative SDF
        ao_neg = calculate_ao(p, n, lambda pos: -1.0)
        assert 0.0 <= ao_neg <= 1.0

    def test_ao_bounds_various_configs(self):
        """Various configs produce AO in [0, 1]."""
        configs = [
            AOConfig(samples=1),
            AOConfig(samples=16),
            AOConfig(step_scale=0.01),
            AOConfig(step_scale=0.5),
            AOConfig(falloff=0.1),
            AOConfig(falloff=0.99),
            AOConfig(intensity=0.5),
            AOConfig(intensity=2.0),
        ]

        p = (0.1, 0.1, 0.0)
        n = (0.707, 0.707, 0.0)

        for config in configs:
            ao = calculate_ao(p, n, sdf_corner_room, config)
            assert 0.0 <= ao <= 1.0, f"AO out of range for config {config}"

    def test_ao_fuzz_random_inputs(self):
        """Random inputs produce valid AO."""
        import random
        random.seed(42)

        for _ in range(50):
            p = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5))
            n = (random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))

            ao = calculate_ao(p, n, sdf_ground_plane)
            assert 0.0 <= ao <= 1.0, f"AO out of range for random input"


# =============================================================================
# AC5: Consistent Behavior
# =============================================================================

class TestConsistentBehavior:
    """Verify AO is consistent and deterministic."""

    def test_ao_deterministic(self):
        """Same inputs produce same output."""
        p = (0.2, 0.2, 0.0)
        n = (0.707, 0.707, 0.0)

        ao_1 = calculate_ao(p, n, sdf_corner_room)
        ao_2 = calculate_ao(p, n, sdf_corner_room)
        ao_3 = calculate_ao(p, n, sdf_corner_room)

        assert ao_1 == ao_2 == ao_3

    def test_ao_continuous(self):
        """Small position changes produce small AO changes."""
        base_p = (0.5, 0.5, 0.0)
        n = (0.707, 0.707, 0.0)

        base_ao = calculate_ao(base_p, n, sdf_corner_room)

        # Small perturbation
        perturbed_p = (0.501, 0.5, 0.0)
        perturbed_ao = calculate_ao(perturbed_p, n, sdf_corner_room)

        # Should be similar
        assert abs(base_ao - perturbed_ao) < 0.1, \
            f"Small position change caused large AO change: {base_ao} -> {perturbed_ao}"

    def test_ao_normal_continuous(self):
        """Small normal changes produce small AO changes."""
        p = (0.3, 0.3, 0.0)
        n1 = (0.707, 0.707, 0.0)
        n2 = (0.71, 0.704, 0.0)

        ao_1 = calculate_ao(p, n1, sdf_corner_room)
        ao_2 = calculate_ao(p, n2, sdf_corner_room)

        assert abs(ao_1 - ao_2) < 0.1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_scene_evaluator_integration(self):
        """Scene evaluator works correctly."""
        evaluator = make_scene_ao_evaluator(sdf_corner_room)

        # Test multiple points
        ao_corner = evaluator((0.1, 0.1, 0.1), (0.577, 0.577, 0.577))
        ao_floor = evaluator((1.0, 0.0, 1.0), (0.0, 1.0, 0.0))

        assert ao_corner < ao_floor, "Corner should be darker than floor"

    def test_multi_direction_integration(self):
        """Multi-direction AO works correctly."""
        p_corner = (0.1, 0.1, 0.1)
        n_corner = (0.577, 0.577, 0.577)

        ao_single = calculate_ao(p_corner, n_corner, sdf_corner_room)
        ao_multi = calculate_ao_multi_direction(p_corner, n_corner, sdf_corner_room)

        # Both should indicate occlusion
        assert ao_single < 0.8
        assert ao_multi < 0.9

    def test_wgsl_generation_integration(self):
        """WGSL generation produces valid shader code."""
        config = AOConfig(samples=5, step_scale=0.1, falloff=0.5)
        wgsl = generate_ao_wgsl(config)

        # Basic syntax checks
        assert wgsl.count("{") == wgsl.count("}")
        assert "fn calculate_ao" in wgsl
        assert "return" in wgsl
        assert "for" in wgsl


# =============================================================================
# Performance Characteristic Tests
# =============================================================================

class TestPerformanceCharacteristics:
    """Test performance-related characteristics."""

    def test_fewer_samples_faster(self):
        """Fewer samples should complete faster (implicit)."""
        # This test verifies the code runs with different sample counts
        p = (0.2, 0.2, 0.0)
        n = (0.707, 0.707, 0.0)

        for samples in [1, 3, 5, 8, 12]:
            config = AOConfig(samples=samples)
            ao = calculate_ao(p, n, sdf_corner_room, config)
            assert 0.0 <= ao <= 1.0

    def test_complex_sdf_works(self):
        """Complex SDF composition works."""
        def complex_scene(p):
            d1 = sdf_ground_plane(p)
            d2 = sdf_sphere(p, 0.5, (0.0, 0.5, 0.0))
            d3 = sdf_sphere(p, 0.3, (0.8, 0.3, 0.0))
            d4 = sdf_box(p, (0.2, 0.2, 0.2))
            return min(d1, d2, d3, d4)

        p = (0.5, 0.5, 0.5)
        n = (0.577, 0.577, 0.577)

        ao = calculate_ao(p, n, complex_scene)
        assert 0.0 <= ao <= 1.0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case behavior tests."""

    def test_origin_point(self):
        """Point at origin works."""
        ao = calculate_ao((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), sdf_ground_plane)
        assert 0.0 <= ao <= 1.0

    def test_large_coordinates(self):
        """Large coordinate values work."""
        ao = calculate_ao((1000.0, 1000.0, 1000.0), (0.0, 1.0, 0.0), sdf_ground_plane)
        assert 0.0 <= ao <= 1.0

    def test_negative_coordinates(self):
        """Negative coordinates work."""
        ao = calculate_ao((-5.0, 0.5, -3.0), (0.0, 1.0, 0.0), sdf_ground_plane)
        assert 0.0 <= ao <= 1.0

    def test_unnormalized_normal_handled(self):
        """Non-unit normal is handled correctly."""
        ao_unit = calculate_ao((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), lambda p: sdf_sphere(p))
        ao_scaled = calculate_ao((1.0, 0.0, 0.0), (5.0, 0.0, 0.0), lambda p: sdf_sphere(p))

        assert ao_unit == pytest.approx(ao_scaled, abs=1e-10)
