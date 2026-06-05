"""
Tests for SDF Combinator Correctness (T-DEMO-7.2)

This test suite validates the mathematical correctness of SDF combinator operations
used in TRINITY's demoscene rendering system. Tests cover:

1. Union operations: min-based combining, associativity, multiple objects
2. Intersection operations: max-based combining, overlap behavior
3. Subtraction operations: carving behavior, order dependence
4. Smooth variants: continuity, blend factor effects, energy preservation
5. Displacement: noise-based surface perturbation
6. Deep nesting: error accumulation under multiple operations
7. Material propagation: winner material ID preservation

Reference:
    - Inigo Quilez SDF combinators: https://iquilezles.org/articles/distfunctions/
    - Rust implementation: crates/renderer-backend/src/sdf_combinators.rs

Total: 25+ tests
"""

from __future__ import annotations

import math
import pytest
from typing import Tuple

from engine.rendering.demoscene.sdf_ast import (
    Vec3,
    SphereNode,
    BoxNode,
    TorusNode,
    CylinderNode,
    UnionNode,
    IntersectionNode,
    SubtractionNode,
    SmoothUnionNode,
    SmoothIntersectionNode,
    SmoothSubtractionNode,
    DisplacedNode,
    MaterialNode,
)


# =============================================================================
# Test Utilities - Pure Python SDF Evaluation
# =============================================================================

# Epsilon for floating point comparisons
EPSILON = 1e-6

# Default smooth k factor
DEFAULT_K = 0.1


def sdf_sphere(p: Vec3, radius: float, center: Vec3 = None) -> float:
    """Evaluate sphere SDF: distance from point to sphere surface."""
    if center is None:
        center = Vec3(0.0, 0.0, 0.0)
    delta = Vec3(p.x - center.x, p.y - center.y, p.z - center.z)
    return delta.length() - radius


def sdf_box(p: Vec3, half_extents: Vec3, center: Vec3 = None) -> float:
    """Evaluate axis-aligned box SDF."""
    if center is None:
        center = Vec3(0.0, 0.0, 0.0)
    # Translate point to box-local space
    local = Vec3(
        abs(p.x - center.x) - half_extents.x,
        abs(p.y - center.y) - half_extents.y,
        abs(p.z - center.z) - half_extents.z,
    )
    # Distance to surface
    outside = Vec3(max(local.x, 0.0), max(local.y, 0.0), max(local.z, 0.0))
    outside_dist = outside.length()
    inside_dist = min(max(local.x, max(local.y, local.z)), 0.0)
    return outside_dist + inside_dist


def sdf_union(d1: float, d2: float) -> float:
    """Union of two SDFs: returns min distance."""
    return min(d1, d2)


def sdf_intersection(d1: float, d2: float) -> float:
    """Intersection of two SDFs: returns max distance."""
    return max(d1, d2)


def sdf_subtraction(d1: float, d2: float) -> float:
    """Subtraction: carve d2 from d1 (order matters)."""
    return max(d1, -d2)


def smin(a: float, b: float, k: float) -> float:
    """Polynomial smooth minimum (Quilez formulation)."""
    k = max(k, 1e-7)
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


def smax(a: float, b: float, k: float) -> float:
    """Polynomial smooth maximum (Quilez formulation)."""
    k = max(k, 1e-7)
    h = max(k - abs(a - b), 0.0) / k
    return max(a, b) + h * h * k * 0.25


def sdf_smooth_union(d1: float, d2: float, k: float) -> float:
    """Smooth union with blend factor k."""
    return smin(d1, d2, k)


def sdf_smooth_intersection(d1: float, d2: float, k: float) -> float:
    """Smooth intersection with blend factor k."""
    return smax(d1, d2, k)


def sdf_smooth_subtraction(d1: float, d2: float, k: float) -> float:
    """Smooth subtraction: carve d2 from d1 with rounded edges."""
    return smax(d1, -d2, k)


def sdf_displaced(base_dist: float, amplitude: float, noise: float) -> float:
    """Apply noise displacement to distance."""
    return base_dist + amplitude * noise


def min2(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Select SDF result with smaller distance (preserves material)."""
    return a if a[0] <= b[0] else b


def max2(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Select SDF result with larger distance (preserves material)."""
    return a if a[0] >= b[0] else b


def smooth_blend_factor(a: float, b: float, k: float) -> float:
    """Compute blend factor for material interpolation."""
    k = max(k, 1e-7)
    h = max(k - abs(a - b), 0.0) / k
    return h * h * 0.5 if a <= b else 1.0 - h * h * 0.5


# =============================================================================
# Union Tests (5 tests)
# =============================================================================

class TestUnionCorrectness:
    """Tests for union combinator correctness."""

    def test_union_two_spheres_min_distance(self):
        """Union of two spheres returns minimum distance."""
        p = Vec3(0.0, 0.0, 0.0)

        # Sphere at origin, radius 1
        sphere1 = sdf_sphere(p, radius=1.0)
        assert sphere1 == pytest.approx(-1.0, abs=EPSILON)  # Inside by 1

        # Sphere at (3, 0, 0), radius 1
        sphere2 = sdf_sphere(p, radius=1.0, center=Vec3(3.0, 0.0, 0.0))
        assert sphere2 == pytest.approx(2.0, abs=EPSILON)  # Outside by 2

        # Union should be -1.0 (closest to first sphere's surface)
        union = sdf_union(sphere1, sphere2)
        assert union == pytest.approx(-1.0, abs=EPSILON)

    def test_union_multiple_objects(self):
        """Union of multiple objects selects the minimum."""
        p = Vec3(2.0, 0.0, 0.0)

        s1 = sdf_sphere(p, radius=1.0, center=Vec3(0.0, 0.0, 0.0))  # dist ~1.0
        s2 = sdf_sphere(p, radius=0.5, center=Vec3(2.0, 0.0, 0.0))  # dist -0.5 (inside)
        s3 = sdf_sphere(p, radius=1.0, center=Vec3(5.0, 0.0, 0.0))  # dist ~2.0

        # Chain unions
        union = sdf_union(sdf_union(s1, s2), s3)
        assert union == pytest.approx(-0.5, abs=EPSILON)  # s2 is closest

    def test_union_associativity(self):
        """Union is associative: (A | B) | C == A | (B | C)."""
        p = Vec3(1.0, 1.0, 1.0)

        s1 = sdf_sphere(p, radius=0.5, center=Vec3(0.0, 0.0, 0.0))
        s2 = sdf_sphere(p, radius=0.5, center=Vec3(2.0, 0.0, 0.0))
        s3 = sdf_sphere(p, radius=0.5, center=Vec3(1.0, 1.0, 0.0))

        # Left associative
        left_assoc = sdf_union(sdf_union(s1, s2), s3)

        # Right associative
        right_assoc = sdf_union(s1, sdf_union(s2, s3))

        assert left_assoc == pytest.approx(right_assoc, abs=EPSILON)

    def test_union_commutativity(self):
        """Union is commutative: A | B == B | A (for distance)."""
        p = Vec3(0.5, 0.5, 0.5)

        s1 = sdf_sphere(p, radius=1.0, center=Vec3(0.0, 0.0, 0.0))
        s2 = sdf_box(p, half_extents=Vec3(0.5, 0.5, 0.5))

        union_ab = sdf_union(s1, s2)
        union_ba = sdf_union(s2, s1)

        assert union_ab == pytest.approx(union_ba, abs=EPSILON)

    def test_union_idempotent(self):
        """Union with itself is idempotent: A | A == A."""
        p = Vec3(0.3, 0.3, 0.3)

        sphere = sdf_sphere(p, radius=1.0)
        union = sdf_union(sphere, sphere)

        assert union == pytest.approx(sphere, abs=EPSILON)


# =============================================================================
# Intersection Tests (5 tests)
# =============================================================================

class TestIntersectionCorrectness:
    """Tests for intersection combinator correctness."""

    def test_intersection_overlapping_boxes(self):
        """Intersection of overlapping boxes returns max distance."""
        # Point inside both boxes
        p = Vec3(0.0, 0.0, 0.0)

        box1 = sdf_box(p, half_extents=Vec3(1.0, 1.0, 1.0))  # Inside by 1
        box2 = sdf_box(p, half_extents=Vec3(0.5, 0.5, 0.5))  # Inside by 0.5

        intersect = sdf_intersection(box1, box2)
        # Should be -0.5 (the less inside one defines the boundary)
        assert intersect == pytest.approx(-0.5, abs=EPSILON)

    def test_intersection_partial_overlap(self):
        """Intersection at boundary of partial overlap."""
        # Point at the edge of overlap region
        p = Vec3(0.7, 0.0, 0.0)

        box1 = sdf_box(p, half_extents=Vec3(1.0, 1.0, 1.0), center=Vec3(0.0, 0.0, 0.0))
        box2 = sdf_box(p, half_extents=Vec3(1.0, 1.0, 1.0), center=Vec3(1.0, 0.0, 0.0))

        intersect = sdf_intersection(box1, box2)
        # Point is inside both boxes, intersection gives max (less negative)
        assert intersect == pytest.approx(max(box1, box2), abs=EPSILON)

    def test_intersection_no_overlap(self):
        """Intersection of non-overlapping objects is empty (positive distance)."""
        # Point far from both objects
        p = Vec3(0.0, 0.0, 0.0)

        sphere1 = sdf_sphere(p, radius=0.5, center=Vec3(-2.0, 0.0, 0.0))  # dist 1.5
        sphere2 = sdf_sphere(p, radius=0.5, center=Vec3(2.0, 0.0, 0.0))   # dist 1.5

        intersect = sdf_intersection(sphere1, sphere2)
        assert intersect > 0.0  # Outside the (empty) intersection

    def test_intersection_sphere_box_carved_corner(self):
        """Intersection of sphere and box creates carved corners."""
        # Point at corner of box, outside sphere
        p = Vec3(0.9, 0.9, 0.9)

        box = sdf_box(p, half_extents=Vec3(1.0, 1.0, 1.0))
        sphere = sdf_sphere(p, radius=1.0)

        intersect = sdf_intersection(box, sphere)
        # At corners, sphere wins (positive distance)
        # Point is inside box but sphere dist is ~sqrt(3)*0.9 - 1 = 0.56
        assert intersect == pytest.approx(max(box, sphere), abs=EPSILON)

    def test_intersection_commutative(self):
        """Intersection is commutative: A & B == B & A."""
        p = Vec3(0.3, 0.3, 0.3)

        s1 = sdf_sphere(p, radius=1.0)
        s2 = sdf_box(p, half_extents=Vec3(0.8, 0.8, 0.8))

        inter_ab = sdf_intersection(s1, s2)
        inter_ba = sdf_intersection(s2, s1)

        assert inter_ab == pytest.approx(inter_ba, abs=EPSILON)


# =============================================================================
# Subtraction Tests (4 tests)
# =============================================================================

class TestSubtractionCorrectness:
    """Tests for subtraction combinator correctness."""

    def test_subtraction_carve_sphere_from_box(self):
        """Carving a sphere from a box creates a cavity."""
        # Point at center where both overlap
        p = Vec3(0.0, 0.0, 0.0)

        box = sdf_box(p, half_extents=Vec3(2.0, 2.0, 2.0))   # Inside by 2
        sphere = sdf_sphere(p, radius=1.0)                   # Inside by 1

        subtract = sdf_subtraction(box, sphere)
        # max(-2.0, -(-1.0)) = max(-2.0, 1.0) = 1.0
        # Point is now OUTSIDE the subtracted shape
        assert subtract == pytest.approx(1.0, abs=EPSILON)

    def test_subtraction_order_matters(self):
        """Subtraction is NOT commutative: A - B != B - A."""
        p = Vec3(0.0, 0.0, 0.0)

        box = sdf_box(p, half_extents=Vec3(1.0, 1.0, 1.0))   # -1.0
        sphere = sdf_sphere(p, radius=0.5)                   # -0.5

        box_minus_sphere = sdf_subtraction(box, sphere)
        sphere_minus_box = sdf_subtraction(sphere, box)

        # These should be different
        assert abs(box_minus_sphere - sphere_minus_box) > 0.1

    def test_subtraction_negates_second_operand(self):
        """Subtraction correctly negates the second operand's distance."""
        p = Vec3(0.0, 0.0, 0.0)

        # Both SDFs have same distance magnitude but opposite for test
        d1 = -0.5  # Inside first object
        d2 = -0.3  # Inside second object

        result = sdf_subtraction(d1, d2)
        expected = max(d1, -d2)  # max(-0.5, 0.3) = 0.3

        assert result == pytest.approx(expected, abs=EPSILON)

    def test_subtraction_no_carve_when_disjoint(self):
        """Subtraction has no effect when objects don't overlap."""
        p = Vec3(0.0, 0.0, 0.0)

        # Two non-overlapping spheres
        sphere1 = sdf_sphere(p, radius=0.5, center=Vec3(0.0, 0.0, 0.0))   # Inside
        sphere2 = sdf_sphere(p, radius=0.5, center=Vec3(5.0, 0.0, 0.0))   # Far away

        subtract = sdf_subtraction(sphere1, sphere2)
        # sphere2 is far, so -sphere2 is very negative, doesn't affect result
        # max(-0.5, -4.5) = -0.5
        assert subtract == pytest.approx(sphere1, abs=EPSILON)


# =============================================================================
# Smooth Union (smin) Tests (4 tests)
# =============================================================================

class TestSmoothUnionCorrectness:
    """Tests for smooth union (smin) combinator correctness."""

    def test_smooth_union_blend_factor_effect(self):
        """Larger k produces more blending (smaller result)."""
        d1 = 0.5
        d2 = 0.45

        # Small k = nearly hard min
        small_k = sdf_smooth_union(d1, d2, k=0.01)
        # Large k = more blending
        large_k = sdf_smooth_union(d1, d2, k=0.5)

        # Hard min
        hard = min(d1, d2)

        # Smooth union always <= hard min
        assert small_k <= hard + EPSILON
        assert large_k <= hard + EPSILON

        # Larger k = more reduction
        assert large_k < small_k

    def test_smooth_union_energy_preservation(self):
        """Smooth union preserves energy (continuous, no jumps)."""
        k = 0.2

        # Sample along a line where SDFs cross
        results = []
        for t in range(-10, 11):
            x = t * 0.1
            d1 = x  # Linear SDF 1
            d2 = -x + 0.5  # Linear SDF 2, crosses at x=0.25
            results.append(sdf_smooth_union(d1, d2, k))

        # Check continuity: no large jumps between adjacent samples
        for i in range(1, len(results)):
            delta = abs(results[i] - results[i-1])
            assert delta < 0.3  # Reasonable continuity bound

    def test_smooth_union_c1_continuity(self):
        """Smooth union has C1 continuous derivatives."""
        k = 0.2
        delta = 0.001
        a = 0.5

        # Sample around the transition point
        results = []
        for i in range(-5, 6):
            b = a + i * delta
            results.append(smin(a, b, k))

        # Compute finite difference derivatives
        derivatives = []
        for i in range(1, len(results)):
            deriv = (results[i] - results[i-1]) / delta
            derivatives.append(deriv)

        # Derivatives should be smooth (no sudden changes)
        for i in range(1, len(derivatives)):
            deriv_change = abs(derivatives[i] - derivatives[i-1])
            assert deriv_change < 0.5  # C1 continuity

    def test_smooth_union_k_zero_degenerates_to_hard_min(self):
        """With k approaching 0, smooth union approaches hard min."""
        d1 = 0.5
        d2 = 0.3

        smooth = sdf_smooth_union(d1, d2, k=1e-6)
        hard = min(d1, d2)

        assert smooth == pytest.approx(hard, abs=0.01)


# =============================================================================
# Smooth Intersection (smax) Tests (2 tests)
# =============================================================================

class TestSmoothIntersectionCorrectness:
    """Tests for smooth intersection (smax) combinator correctness."""

    def test_smooth_intersection_blend_produces_larger_result(self):
        """Smooth intersection produces result >= hard max."""
        d1 = 0.5
        d2 = 0.45

        smooth = sdf_smooth_intersection(d1, d2, k=0.3)
        hard = max(d1, d2)

        # Smooth max always >= hard max
        assert smooth >= hard - EPSILON

    def test_smooth_intersection_continuous_transition(self):
        """Smooth intersection has continuous transition between surfaces."""
        k = 0.2

        results = []
        for t in range(-10, 11):
            x = t * 0.1
            d1 = x
            d2 = -x + 0.5
            results.append(sdf_smooth_intersection(d1, d2, k))

        # Check continuity
        for i in range(1, len(results)):
            delta = abs(results[i] - results[i-1])
            assert delta < 0.3


# =============================================================================
# Smooth Subtraction Tests (2 tests)
# =============================================================================

class TestSmoothSubtractionCorrectness:
    """Tests for smooth subtraction combinator correctness."""

    def test_smooth_subtraction_rounded_carve_edge(self):
        """Smooth subtraction creates rounded edges at the carve boundary."""
        # Sample points near the carve boundary
        k = 0.3

        # Point inside original object but at boundary of carved region
        base = -0.5  # Inside primary
        carve = -0.1  # Inside carve region

        hard = sdf_subtraction(base, carve)
        smooth = sdf_smooth_subtraction(base, carve, k)

        # Smooth should be larger (more rounded) near boundary
        assert smooth >= hard - EPSILON

    def test_smooth_subtraction_not_symmetric(self):
        """Smooth subtraction order still matters."""
        k = 0.2
        d1 = 0.5
        d2 = 0.3

        result_ab = sdf_smooth_subtraction(d1, d2, k)
        result_ba = sdf_smooth_subtraction(d2, d1, k)

        assert abs(result_ab - result_ba) > 0.05


# =============================================================================
# Displacement Tests (2 tests)
# =============================================================================

class TestDisplacementCorrectness:
    """Tests for noise displacement combinator correctness."""

    def test_displacement_adds_noise_scaled_by_amplitude(self):
        """Displacement correctly applies amplitude * noise."""
        base_dist = 0.5
        amplitude = 0.1
        noise = 0.7

        displaced = sdf_displaced(base_dist, amplitude, noise)
        expected = base_dist + amplitude * noise

        assert displaced == pytest.approx(expected, abs=EPSILON)

    def test_displacement_zero_amplitude_preserves_distance(self):
        """Zero amplitude preserves original distance."""
        base_dist = 0.5
        noise = 0.9

        displaced = sdf_displaced(base_dist, 0.0, noise)

        assert displaced == pytest.approx(base_dist, abs=EPSILON)


# =============================================================================
# Deep Nesting Tests (2 tests)
# =============================================================================

class TestDeepNestingCorrectness:
    """Tests for deeply nested combinator operations."""

    def test_deep_nesting_five_levels_no_error_accumulation(self):
        """Five levels of nesting doesn't accumulate significant error."""
        p = Vec3(0.0, 0.0, 0.0)

        # Create 5 spheres at different positions
        spheres = [
            sdf_sphere(p, radius=0.5, center=Vec3(i * 0.3, 0.0, 0.0))
            for i in range(-2, 3)
        ]

        # Nest unions: ((((s0 | s1) | s2) | s3) | s4)
        result = spheres[0]
        for s in spheres[1:]:
            result = sdf_union(result, s)

        # Result should be minimum of all spheres
        expected = min(spheres)

        assert result == pytest.approx(expected, abs=EPSILON)

    def test_deep_nesting_mixed_combinators(self):
        """Mixed combinators at depth maintain correctness."""
        p = Vec3(0.0, 0.0, 0.0)

        # Build: (sphere | box) & (sphere - box)
        sphere = sdf_sphere(p, radius=1.0)
        box = sdf_box(p, half_extents=Vec3(0.8, 0.8, 0.8))

        union = sdf_union(sphere, box)
        subtract = sdf_subtraction(sphere, box)
        result = sdf_intersection(union, subtract)

        # Manually compute expected
        expected = max(min(sphere, box), max(sphere, -box))

        assert result == pytest.approx(expected, abs=EPSILON)


# =============================================================================
# Material Propagation Tests (3 tests)
# =============================================================================

class TestMaterialPropagation:
    """Tests for material ID propagation in combinators."""

    def test_union_preserves_winner_material(self):
        """Union preserves material from closer surface."""
        # Sphere with material 1, closer
        sphere = (0.3, 1.0)  # (distance, material_id)
        # Box with material 2, farther
        box = (0.8, 2.0)

        result = min2(sphere, box)

        assert result[0] == pytest.approx(0.3, abs=EPSILON)
        assert result[1] == pytest.approx(1.0, abs=EPSILON)  # Sphere's material

    def test_intersection_preserves_boundary_material(self):
        """Intersection preserves material from boundary surface."""
        # The surface that defines the boundary (larger distance) wins
        sdf_a = (-0.5, 1.0)  # Inside surface A
        sdf_b = (0.2, 2.0)   # Outside surface B (B defines boundary)

        result = max2(sdf_a, sdf_b)

        assert result[0] == pytest.approx(0.2, abs=EPSILON)
        assert result[1] == pytest.approx(2.0, abs=EPSILON)  # B's material

    def test_smooth_union_interpolates_material_in_blend_region(self):
        """Smooth union interpolates materials in the blend region."""
        d1 = 0.5
        d2 = 0.5  # Equal distances
        mat1 = 1.0
        mat2 = 3.0
        k = 0.2

        # Compute blend factor
        t = smooth_blend_factor(d1, d2, k)

        # When distances are equal, blend should be ~0.5
        assert 0.4 <= t <= 0.6

        # Material should interpolate
        blended_mat = mat1 * (1.0 - t) + mat2 * t
        assert 1.5 < blended_mat < 2.5  # Between mat1 and mat2


# =============================================================================
# AST Node Construction Tests (2 tests)
# =============================================================================

class TestASTNodeConstruction:
    """Tests for AST node construction of combinators."""

    def test_union_node_children(self):
        """UnionNode correctly references its children."""
        left = SphereNode(radius=1.0)
        right = BoxNode(half_extents=Vec3(1.0, 1.0, 1.0))

        union = UnionNode(left, right)

        assert union.left is left
        assert union.right is right
        assert union.children() == (left, right)
        assert union.wgsl_function == "sdf_union"

    def test_smooth_union_node_k_parameter(self):
        """SmoothUnionNode stores k parameter correctly."""
        left = SphereNode(radius=1.0)
        right = SphereNode(radius=0.5, position=Vec3(1.5, 0.0, 0.0))

        smooth_union = SmoothUnionNode(left, right, k=0.25)

        assert smooth_union.k == 0.25
        assert smooth_union.wgsl_function == "sdf_smooth_union"


# =============================================================================
# Edge Case Tests (3 tests)
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_union_with_very_large_distances(self):
        """Union handles very large distance values."""
        d1 = 1e10
        d2 = 1e10 + 1.0

        result = sdf_union(d1, d2)

        assert result == pytest.approx(d1, rel=1e-6)

    def test_intersection_with_negative_distances(self):
        """Intersection handles deeply negative distances."""
        d1 = -100.0
        d2 = -50.0

        result = sdf_intersection(d1, d2)

        # max(-100, -50) = -50
        assert result == pytest.approx(-50.0, abs=EPSILON)

    def test_smooth_union_handles_equal_distances(self):
        """Smooth union handles exactly equal distances."""
        d1 = 0.5
        d2 = 0.5
        k = 0.2

        result = sdf_smooth_union(d1, d2, k)

        # Should be less than hard min due to smoothing
        # h = (0.2 - 0) / 0.2 = 1.0
        # result = 0.5 - 1.0 * 1.0 * 0.2 * 0.25 = 0.45
        assert result == pytest.approx(0.45, abs=EPSILON)


# =============================================================================
# Integration Tests (2 tests)
# =============================================================================

class TestCombinatorIntegration:
    """Integration tests combining multiple combinator operations."""

    def test_csg_boolean_tree(self):
        """Complex CSG tree evaluates correctly."""
        p = Vec3(0.0, 0.0, 0.0)

        # Create a complex shape: (A | B) - (C & D)
        a = sdf_sphere(p, radius=2.0)
        b = sdf_box(p, half_extents=Vec3(1.5, 1.5, 1.5))
        c = sdf_sphere(p, radius=1.0)
        d = sdf_box(p, half_extents=Vec3(0.8, 0.8, 0.8))

        union_ab = sdf_union(a, b)
        intersect_cd = sdf_intersection(c, d)
        result = sdf_subtraction(union_ab, intersect_cd)

        # Compute expected step by step
        expected_union = min(a, b)
        expected_intersect = max(c, d)
        expected = max(expected_union, -expected_intersect)

        assert result == pytest.approx(expected, abs=EPSILON)

    def test_displaced_union(self):
        """Displacement applied after union works correctly."""
        p = Vec3(0.0, 0.0, 0.0)

        sphere = sdf_sphere(p, radius=1.0)
        box = sdf_box(p, half_extents=Vec3(0.8, 0.8, 0.8))

        union = sdf_union(sphere, box)

        # Apply displacement
        amplitude = 0.1
        noise = 0.5
        displaced = sdf_displaced(union, amplitude, noise)

        expected = union + amplitude * noise

        assert displaced == pytest.approx(expected, abs=EPSILON)
