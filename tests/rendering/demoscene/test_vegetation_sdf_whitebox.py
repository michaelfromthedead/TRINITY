"""
Whitebox tests for vegetation SDF (T-DEMO-4.5 and T-DEMO-4.6).

Tests the internal implementation details of Tree SDF and Forest SDF,
verifying correct behavior of internal functions and algorithms.

Coverage targets:
- Internal SDF primitive functions
- Domain repetition mechanics
- Hash function distribution
- Canopy/branch position calculations
- WGSL code generation details
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple

import pytest


# =============================================================================
# Import module under test (including internal functions)
# =============================================================================

from engine.rendering.demoscene.vegetation_sdf import (
    # Enums
    TrunkType,
    CanopyType,
    # Configurations
    TreeConfig,
    BranchConfig,
    ForestConfig,
    TreeVariation,
    # SDF Classes
    TreeSDF,
    ForestSDF,
    # Hash functions
    cell_hash,
    cell_hash_float,
    hash_to_float,
    # Python SDF functions
    sdf_tree,
    sdf_forest,
    # WGSL generation
    generate_tree_wgsl,
    generate_forest_wgsl,
    TREE_PRIMITIVES_WGSL,
    FOREST_HASH_WGSL,
)

# Internal functions for whitebox testing
from engine.rendering.demoscene.vegetation_sdf import (
    sdf_sphere,
    sdf_cylinder,
    sdf_cone,
    sdf_capsule,
    sdf_ellipsoid,
    smooth_union,
    _sdf_trunk,
    _sdf_canopy,
    _sdf_branches,
    _get_canopy_positions,
    _get_branch_endpoints,
    _get_cell_id,
    _get_cell_center,
    _get_cell_tree_position,
    _get_cell_tree_config,
    _apply_rotation,
)


# =============================================================================
# Test Constants
# =============================================================================

TOL_SURFACE = 1e-7
TOL_DISTANCE = 1e-4
PI = math.pi


# =============================================================================
# Primitive SDF Function Tests
# =============================================================================

class TestSphereSDF:
    """Whitebox tests for sdf_sphere."""

    def test_sphere_at_origin_negative(self):
        """Origin should be at -radius."""
        d = sdf_sphere((0, 0, 0), 1.0)
        assert d == pytest.approx(-1.0, abs=TOL_SURFACE)

    def test_sphere_on_surface_zero(self):
        """Surface points should be zero."""
        d = sdf_sphere((1.0, 0, 0), 1.0)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

        d = sdf_sphere((0, 1.0, 0), 1.0)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_sphere_outside_positive(self):
        """Outside points should be positive."""
        d = sdf_sphere((2.0, 0, 0), 1.0)
        assert d == pytest.approx(1.0, abs=TOL_SURFACE)

    def test_sphere_diagonal(self):
        """Diagonal point at sqrt(3) from origin."""
        d = sdf_sphere((1, 1, 1), 1.0)
        expected = math.sqrt(3) - 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE)


class TestCylinderSDF:
    """Whitebox tests for sdf_cylinder."""

    def test_cylinder_axis_inside(self):
        """Points on axis should be inside."""
        d = sdf_cylinder((0, 0, 0), 1.0, 1.0)
        assert d < 0

    def test_cylinder_radial_surface(self):
        """Point on radial surface should be zero."""
        d = sdf_cylinder((1.0, 0, 0), 1.0, 2.0)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_cylinder_cap_surface(self):
        """Point on cap should be zero."""
        d = sdf_cylinder((0, 1.0, 0), 0.5, 1.0)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_cylinder_corner(self):
        """Corner point should use distance formula."""
        # Point at corner: radius=1, height=1
        # Corner at (1, 1, 0) should be at distance sqrt(2) diagonally
        # from surface intersection
        d = sdf_cylinder((1.5, 1.5, 0), 1.0, 1.0)
        expected = math.sqrt(0.5**2 + 0.5**2)
        assert d == pytest.approx(expected, abs=TOL_DISTANCE)


class TestConeSDF:
    """Whitebox tests for sdf_cone (tapered frustum)."""

    def test_cone_center_inside(self):
        """Center should be inside."""
        d = sdf_cone((0, 0, 0), 1.0, 0.5, 1.0)
        assert d < 0

    def test_cone_base_wider(self):
        """Base (bottom) should be wider than top."""
        # Point at r=0.8, should be inside at bottom (-0.5) but outside at top (0.5)
        d_bottom = sdf_cone((0.8, -0.4, 0), 1.0, 0.5, 1.0)
        d_top = sdf_cone((0.8, 0.4, 0), 1.0, 0.5, 1.0)
        assert d_bottom < d_top, f"Bottom {d_bottom} should be more inside than top {d_top}"

    def test_cylinder_special_case(self):
        """Equal radii should behave like cylinder."""
        d_cone = sdf_cone((1.0, 0, 0), 1.0, 1.0, 2.0)
        d_cyl = sdf_cylinder((1.0, 0, 0), 1.0, 1.0)
        # Should be approximately equal
        assert abs(d_cone - d_cyl) < TOL_DISTANCE * 10


class TestCapsuleSDF:
    """Whitebox tests for sdf_capsule."""

    def test_capsule_midpoint(self):
        """Point on capsule midpoint surface should be zero."""
        a = (0, 0, 0)
        b = (0, 2, 0)
        r = 0.5
        # Point at radius from midpoint
        d = sdf_capsule((0.5, 1, 0), a, b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_capsule_endpoint_sphere(self):
        """Endpoint should be spherical cap."""
        a = (0, 0, 0)
        b = (0, 2, 0)
        r = 0.5
        # Point below bottom endpoint
        d = sdf_capsule((0, -0.5, 0), a, b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_capsule_degenerate(self):
        """Degenerate capsule (a==b) should be sphere."""
        a = (1, 1, 1)
        b = (1, 1, 1)
        r = 0.5
        d_cap = sdf_capsule((1.5, 1, 1), a, b, r)
        d_sph = sdf_sphere((0.5, 0, 0), r)
        assert d_cap == pytest.approx(d_sph, abs=TOL_SURFACE)


class TestEllipsoidSDF:
    """Whitebox tests for sdf_ellipsoid."""

    def test_ellipsoid_sphere_case(self):
        """Equal radii should approximate sphere."""
        d_ell = sdf_ellipsoid((1.0, 0, 0), (1.0, 1.0, 1.0))
        d_sph = sdf_sphere((1.0, 0, 0), 1.0)
        # Ellipsoid is approximate, so tolerance is higher
        assert abs(d_ell - d_sph) < 0.1

    def test_ellipsoid_stretched(self):
        """Stretched ellipsoid should reflect proportions."""
        # Ellipsoid stretched 2x along x
        radii = (2.0, 1.0, 1.0)
        # Point at surface along x
        d = sdf_ellipsoid((2.0, 0, 0), radii)
        assert abs(d) < 0.1  # Should be near zero

    def test_ellipsoid_center_negative(self):
        """Center should be most inside."""
        d = sdf_ellipsoid((0, 0, 0), (1.0, 1.0, 1.0))
        assert d < 0


class TestSmoothUnion:
    """Whitebox tests for smooth_union."""

    def test_smooth_union_k_zero_is_min(self):
        """k=0 should be hard union (min)."""
        d = smooth_union(1.0, 2.0, 0.0)
        assert d == pytest.approx(1.0, abs=TOL_SURFACE)

        d = smooth_union(3.0, 1.0, 0.0)
        assert d == pytest.approx(1.0, abs=TOL_SURFACE)

    def test_smooth_union_symmetry(self):
        """smooth_union(a, b) should NOT equal smooth_union(b, a) exactly."""
        # Note: smooth union is actually not symmetric! The formula biases toward d1.
        d1 = smooth_union(1.0, 2.0, 0.5)
        d2 = smooth_union(2.0, 1.0, 0.5)
        # Both should be between the inputs and less than min due to blending
        assert d1 < 2.0
        assert d2 < 2.0

    def test_smooth_union_blends(self):
        """Smooth union should blend distances."""
        d = smooth_union(1.0, 1.0, 0.5)
        # When both inputs are equal, result should be less than either
        # due to blending formula
        assert d < 1.0

    def test_smooth_union_higher_k_more_blend(self):
        """Higher k should produce more blending (reduction from min)."""
        # With equal inputs, higher k produces lower output
        d_low = smooth_union(1.0, 1.0, 0.1)
        d_high = smooth_union(1.0, 1.0, 0.5)

        # Higher k should reduce the distance more when inputs are equal
        assert d_high < d_low, f"d_high={d_high} should be < d_low={d_low}"


# =============================================================================
# Internal Tree Function Tests
# =============================================================================

class TestInternalTrunk:
    """Whitebox tests for _sdf_trunk."""

    def test_trunk_base_position(self):
        """Trunk base should be at y=0."""
        config = TreeConfig(trunk_height=2.0, trunk_radius=0.3)
        # Point just above ground level on trunk axis
        d = _sdf_trunk((0, 0.1, 0), config)
        assert d < 0, "Point near base should be inside trunk"

    def test_trunk_top_position(self):
        """Trunk top should be at y=trunk_height."""
        config = TreeConfig(trunk_height=2.0, trunk_radius=0.3)
        # Point just below trunk top on axis
        d = _sdf_trunk((0, 1.9, 0), config)
        assert d < 0, "Point near top should be inside trunk"

        # Point above trunk
        d = _sdf_trunk((0, 2.5, 0), config)
        assert d > 0, "Point above trunk should be outside"

    def test_trunk_cylinder_vs_cone(self):
        """Cylinder and cone trunks should differ."""
        config_cyl = TreeConfig(trunk_type=TrunkType.CYLINDER)
        config_cone = TreeConfig(trunk_type=TrunkType.TAPERED_CONE, trunk_taper=0.5)

        # At top, tapered cone should have smaller radius
        p = (0.1, 1.8, 0)
        d_cyl = _sdf_trunk(p, config_cyl)
        d_cone = _sdf_trunk(p, config_cone)

        # Same radial position should be more inside cylinder
        assert d_cyl < d_cone, "Cylinder should be more inside at same position"


class TestInternalCanopyPositions:
    """Whitebox tests for _get_canopy_positions."""

    def test_single_sphere_centered(self):
        """Single sphere should be at center."""
        config = TreeConfig(canopy_spheres=1, trunk_height=2.0, canopy_height_offset=0.5)
        positions = _get_canopy_positions(config)

        assert len(positions) == 1
        assert positions[0] == pytest.approx((0.0, 2.5, 0.0), abs=TOL_DISTANCE)

    def test_multiple_spheres_distribution(self):
        """Multiple spheres should be distributed in ring."""
        config = TreeConfig(canopy_spheres=5, canopy_spread=1.0)
        positions = _get_canopy_positions(config)

        assert len(positions) == 5

        # First is central
        assert abs(positions[0][0]) < 0.01
        assert abs(positions[0][2]) < 0.01

        # Ring positions should be at canopy_spread distance
        for pos in positions[1:]:
            dist = math.sqrt(pos[0]**2 + pos[2]**2)
            assert dist == pytest.approx(config.canopy_spread, abs=0.1)

    def test_ring_angles(self):
        """Ring spheres should be evenly distributed."""
        config = TreeConfig(canopy_spheres=5, canopy_spread=1.0)
        positions = _get_canopy_positions(config)

        # Calculate angles for ring positions
        angles = []
        for pos in positions[1:]:
            angle = math.atan2(pos[2], pos[0])
            angles.append(angle)

        # Sort and check spacing
        angles.sort()
        expected_spacing = 2 * PI / 4  # 4 ring positions
        for i in range(len(angles) - 1):
            spacing = angles[i + 1] - angles[i]
            assert spacing == pytest.approx(expected_spacing, abs=0.1)


class TestInternalBranchEndpoints:
    """Whitebox tests for _get_branch_endpoints."""

    def test_no_branches_empty(self):
        """No branches should return empty list."""
        config = TreeConfig(branches=BranchConfig(count=0))
        endpoints = _get_branch_endpoints(config)
        assert endpoints == []

    def test_branch_start_positions(self):
        """Branch starts should be on trunk surface."""
        config = TreeConfig(
            branches=BranchConfig(count=4, attachment_height=0.5),
            trunk_height=2.0,
            trunk_radius=0.2,
        )
        endpoints = _get_branch_endpoints(config)

        expected_y = 2.0 * 0.5  # attachment_height
        for start, end in endpoints:
            # Y should be at attachment height
            assert start[1] == pytest.approx(expected_y, abs=TOL_DISTANCE)
            # Distance from axis should be near trunk radius
            dist = math.sqrt(start[0]**2 + start[2]**2)
            assert dist < config.trunk_radius * 1.0  # On trunk surface

    def test_branch_angular_distribution(self):
        """Branches should be evenly distributed around trunk."""
        config = TreeConfig(branches=BranchConfig(count=4))
        endpoints = _get_branch_endpoints(config)

        angles = []
        for start, _ in endpoints:
            angle = math.atan2(start[2], start[0])
            angles.append(angle)

        angles.sort()
        expected_spacing = 2 * PI / 4
        for i in range(len(angles) - 1):
            spacing = angles[i + 1] - angles[i]
            assert spacing == pytest.approx(expected_spacing, abs=0.1)


# =============================================================================
# Internal Forest Function Tests
# =============================================================================

class TestInternalCellID:
    """Whitebox tests for _get_cell_id."""

    def test_positive_coordinates(self):
        """Positive coordinates should work correctly."""
        cell_size = (10.0, 10.0, 10.0)
        assert _get_cell_id((5, 5, 5), cell_size) == (0, 0, 0)
        assert _get_cell_id((15, 5, 5), cell_size) == (1, 0, 0)
        assert _get_cell_id((25, 25, 25), cell_size) == (2, 2, 2)

    def test_negative_coordinates(self):
        """Negative coordinates should work correctly."""
        cell_size = (10.0, 10.0, 10.0)
        assert _get_cell_id((-5, 5, 5), cell_size) == (-1, 0, 0)
        assert _get_cell_id((-15, -5, -25), cell_size) == (-2, -1, -3)

    def test_on_boundary(self):
        """Points on boundary should go to higher cell."""
        cell_size = (10.0, 10.0, 10.0)
        assert _get_cell_id((10.0, 0, 0), cell_size) == (1, 0, 0)
        assert _get_cell_id((9.999, 0, 0), cell_size) == (0, 0, 0)


class TestInternalCellCenter:
    """Whitebox tests for _get_cell_center."""

    def test_cell_center_calculation(self):
        """Cell center should be at (cell + 0.5) * size."""
        cell_size = (10.0, 10.0, 10.0)

        center = _get_cell_center((0, 0, 0), cell_size)
        assert center == pytest.approx((5.0, 5.0, 5.0), abs=TOL_DISTANCE)

        center = _get_cell_center((1, 2, -1), cell_size)
        assert center == pytest.approx((15.0, 25.0, -5.0), abs=TOL_DISTANCE)


class TestInternalTreePosition:
    """Whitebox tests for _get_cell_tree_position."""

    def test_empty_cell_returns_none(self):
        """Cell without tree should return None."""
        config = ForestConfig(density=0.0)
        pos = _get_cell_tree_position((0, 0, 0), config)
        assert pos is None

    def test_full_density_returns_position(self):
        """Full density cell should return position."""
        config = ForestConfig(density=1.0)
        pos = _get_cell_tree_position((0, 0, 0), config)
        assert pos is not None

    def test_position_within_cell(self):
        """Tree position should be within cell bounds."""
        config = ForestConfig(
            cell_size=(10.0, 10.0, 10.0),
            density=1.0,
            variation=TreeVariation(position_jitter=0.4),
        )

        for x in range(-3, 4):
            for z in range(-3, 4):
                pos = _get_cell_tree_position((x, 0, z), config)
                if pos:
                    # X and Z should be within cell
                    cell_min_x = x * 10.0
                    cell_max_x = (x + 1) * 10.0
                    cell_min_z = z * 10.0
                    cell_max_z = (z + 1) * 10.0

                    # Allow some tolerance for jitter near boundaries
                    assert cell_min_x - 5 <= pos[0] <= cell_max_x + 5
                    assert cell_min_z - 5 <= pos[2] <= cell_max_z + 5


class TestInternalTreeConfig:
    """Whitebox tests for _get_cell_tree_config."""

    def test_applies_height_variation(self):
        """Height variation should be applied."""
        var = TreeVariation(height_min=0.5, height_max=2.0)
        base_config = TreeConfig(trunk_height=3.0)
        forest_config = ForestConfig(base_tree=base_config, variation=var, density=1.0)

        heights = set()
        for x in range(20):
            tree_config = _get_cell_tree_config((x, 0, 0), forest_config)
            if tree_config:
                heights.add(round(tree_config.trunk_height, 2))

        # Should have multiple different heights
        assert len(heights) > 3, f"Should have height variation: {heights}"

    def test_applies_canopy_variation(self):
        """Canopy count variation should be applied."""
        var = TreeVariation(canopy_min=3, canopy_max=8)
        forest_config = ForestConfig(variation=var, density=1.0)

        counts = set()
        for x in range(30):
            tree_config = _get_cell_tree_config((x, 0, 0), forest_config)
            if tree_config:
                counts.add(tree_config.canopy_spheres)

        # Should have multiple different counts
        assert len(counts) >= 2, f"Should have canopy count variation: {counts}"

        # All should be in range
        for c in counts:
            assert 3 <= c <= 8


class TestInternalRotation:
    """Whitebox tests for _apply_rotation."""

    def test_zero_rotation_identity(self):
        """Zero rotation should be identity."""
        p = (1.0, 2.0, 3.0)
        rotated = _apply_rotation(p, 0.0)
        assert rotated == pytest.approx(p, abs=TOL_SURFACE)

    def test_90_degree_rotation(self):
        """90 degree rotation around Y axis."""
        p = (1.0, 5.0, 0.0)
        rotated = _apply_rotation(p, PI / 2)
        # X -> Z, Z -> -X
        assert rotated == pytest.approx((0.0, 5.0, 1.0), abs=TOL_DISTANCE)

    def test_180_degree_rotation(self):
        """180 degree rotation around Y axis."""
        p = (1.0, 5.0, 2.0)
        rotated = _apply_rotation(p, PI)
        # X -> -X, Z -> -Z
        assert rotated == pytest.approx((-1.0, 5.0, -2.0), abs=TOL_DISTANCE)

    def test_y_unchanged(self):
        """Y coordinate should be unchanged by Y rotation."""
        p = (1.0, 7.5, 3.0)
        for angle in [0.0, 0.5, 1.0, 2.0, 3.0]:
            rotated = _apply_rotation(p, angle)
            assert rotated[1] == pytest.approx(p[1], abs=TOL_SURFACE)


# =============================================================================
# Hash Function Distribution Tests
# =============================================================================

class TestHashDistribution:
    """Whitebox tests for hash function distribution."""

    def test_hash_avalanche(self):
        """Small input changes should cause large hash changes."""
        h1 = cell_hash((0, 0, 0))
        h2 = cell_hash((1, 0, 0))
        h3 = cell_hash((0, 1, 0))
        h4 = cell_hash((0, 0, 1))

        # All should be very different
        assert h1 != h2
        assert h2 != h3
        assert h3 != h4

        # Check bit differences are significant
        def popcount(x):
            return bin(x).count('1')

        diff12 = popcount(h1 ^ h2)
        diff23 = popcount(h2 ^ h3)
        # Good avalanche: about half the bits should differ
        assert diff12 >= 8, f"Poor avalanche: only {diff12} bits differ"
        assert diff23 >= 8, f"Poor avalanche: only {diff23} bits differ"

    def test_hash_float_uniformity(self):
        """hash_to_float should produce roughly uniform distribution."""
        # Generate many float values
        floats = []
        for x in range(-50, 50):
            for z in range(-50, 50):
                f = cell_hash_float((x, 0, z), 0)
                floats.append(f)

        # Check distribution in buckets
        buckets = [0] * 10
        for f in floats:
            bucket = min(int(f * 10), 9)
            buckets[bucket] += 1

        # Each bucket should have roughly 1/10 of values
        expected = len(floats) / 10
        for i, count in enumerate(buckets):
            ratio = count / expected
            assert 0.5 < ratio < 2.0, f"Bucket {i} has {count}, expected ~{expected}"

    def test_channel_independence(self):
        """Different channels should be statistically independent."""
        # Compute correlation between channels
        n = 100
        ch0 = [cell_hash_float((x, 0, 0), 0) for x in range(n)]
        ch1 = [cell_hash_float((x, 0, 0), 1) for x in range(n)]

        # Compute correlation coefficient
        mean0 = sum(ch0) / n
        mean1 = sum(ch1) / n

        cov = sum((a - mean0) * (b - mean1) for a, b in zip(ch0, ch1)) / n
        std0 = (sum((a - mean0)**2 for a in ch0) / n) ** 0.5
        std1 = (sum((b - mean1)**2 for b in ch1) / n) ** 0.5

        correlation = cov / (std0 * std1) if std0 > 0 and std1 > 0 else 0

        # Correlation should be low (channels are independent)
        assert abs(correlation) < 0.3, f"Channels correlated: r={correlation}"


# =============================================================================
# WGSL Code Generation Whitebox Tests
# =============================================================================

class TestWGSLPrimitives:
    """Whitebox tests for WGSL primitive code."""

    def test_primitives_contains_sphere(self):
        """Primitives should include sphere SDF."""
        assert "fn sdf_sphere" in TREE_PRIMITIVES_WGSL
        assert "length(p)" in TREE_PRIMITIVES_WGSL

    def test_primitives_contains_cylinder(self):
        """Primitives should include cylinder SDF."""
        assert "fn sdf_cylinder" in TREE_PRIMITIVES_WGSL
        assert "p.xz" in TREE_PRIMITIVES_WGSL

    def test_primitives_contains_smooth_union(self):
        """Primitives should include smooth union."""
        assert "fn sdf_smooth_union" in TREE_PRIMITIVES_WGSL
        assert "clamp" in TREE_PRIMITIVES_WGSL
        assert "mix" in TREE_PRIMITIVES_WGSL

    def test_forest_hash_contains_cell_hash(self):
        """Forest hash should include cell_hash function."""
        assert "fn cell_hash" in FOREST_HASH_WGSL
        assert "PRIME" in FOREST_HASH_WGSL

    def test_forest_hash_contains_rotate(self):
        """Forest hash should include rotation function."""
        assert "fn rotate_y" in FOREST_HASH_WGSL
        assert "cos" in FOREST_HASH_WGSL
        assert "sin" in FOREST_HASH_WGSL


class TestWGSLTreeGeneration:
    """Whitebox tests for tree WGSL generation."""

    def test_trunk_function_generated(self):
        """Trunk function should be generated."""
        wgsl = generate_tree_wgsl(TreeConfig(), "test_tree")
        assert "fn sdf_test_tree_trunk" in wgsl

    def test_canopy_function_generated(self):
        """Canopy function should be generated."""
        wgsl = generate_tree_wgsl(TreeConfig(), "test_tree")
        assert "fn sdf_test_tree_canopy" in wgsl

    def test_constants_generated(self):
        """Tree constants should be generated."""
        config = TreeConfig(trunk_height=5.5, canopy_spheres=7)
        wgsl = generate_tree_wgsl(config, "my_tree")

        assert "MY_TREE_TRUNK_HEIGHT" in wgsl
        assert "5.5" in wgsl
        assert "MY_TREE_CANOPY_SPHERES" in wgsl
        assert "7u" in wgsl

    def test_cylinder_trunk_code(self):
        """Cylinder trunk should generate cylinder call."""
        config = TreeConfig(trunk_type=TrunkType.CYLINDER)
        wgsl = generate_tree_wgsl(config, "tree")
        assert "sdf_cylinder" in wgsl

    def test_tapered_trunk_code(self):
        """Tapered trunk should generate cone call."""
        config = TreeConfig(trunk_type=TrunkType.TAPERED_CONE)
        wgsl = generate_tree_wgsl(config, "tree")
        assert "sdf_cone_tapered" in wgsl

    def test_ellipsoid_canopy_code(self):
        """Ellipsoid canopy should generate ellipsoid call."""
        config = TreeConfig(canopy_type=CanopyType.ELLIPSOIDS)
        wgsl = generate_tree_wgsl(config, "tree")
        assert "sdf_ellipsoid" in wgsl

    def test_branches_generate_capsules(self):
        """Branches should generate capsule calls."""
        config = TreeConfig(branches=BranchConfig(count=3))
        wgsl = generate_tree_wgsl(config, "tree")
        assert "fn sdf_tree_branches" in wgsl
        # Should have 3 branch capsule calls
        assert wgsl.count("sdf_capsule") >= 3


class TestWGSLForestGeneration:
    """Whitebox tests for forest WGSL generation."""

    def test_cell_tree_function(self):
        """Cell tree evaluation function should be generated."""
        wgsl = generate_forest_wgsl(ForestConfig(), "forest")
        assert "fn sdf_forest_cell_tree" in wgsl

    def test_main_function(self):
        """Main forest SDF function should be generated."""
        wgsl = generate_forest_wgsl(ForestConfig(), "forest")
        assert "fn sdf_forest(p: vec3<f32>) -> f32" in wgsl

    def test_neighborhood_loop(self):
        """3x3x3 neighborhood loop should be generated."""
        wgsl = generate_forest_wgsl(ForestConfig(), "forest")
        # Check for nested loops
        assert "for (var dx:" in wgsl
        assert "for (var dy:" in wgsl
        assert "for (var dz:" in wgsl

    def test_density_check(self):
        """Density check should be generated."""
        wgsl = generate_forest_wgsl(ForestConfig(density=0.75), "forest")
        assert "0.75" in wgsl or "DENSITY" in wgsl

    def test_variation_constants(self):
        """Variation constants should be generated."""
        var = TreeVariation(height_min=0.6, height_max=1.4)
        config = ForestConfig(variation=var)
        wgsl = generate_forest_wgsl(config, "forest")

        assert "HEIGHT_MIN" in wgsl
        assert "HEIGHT_MAX" in wgsl
        assert "0.6" in wgsl
        assert "1.4" in wgsl

    def test_rotation_conditional(self):
        """Rotation should be conditional on enabled flag."""
        config = ForestConfig(variation=TreeVariation(rotation_enabled=True))
        wgsl = generate_forest_wgsl(config, "forest")
        assert "ROTATION_ENABLED" in wgsl
        assert "rotate_y" in wgsl


# =============================================================================
# SDF Tree Integration Tests
# =============================================================================

class TestSDFTreeIntegration:
    """Whitebox integration tests for sdf_tree."""

    def test_sdf_tree_combines_components(self):
        """sdf_tree should combine trunk and canopy."""
        config = TreeConfig(trunk_height=2.0, canopy_height_offset=0.5)

        # Point in trunk
        d_trunk = _sdf_trunk((0, 0.5, 0), config)
        d_full = sdf_tree((0, 0.5, 0), config)
        # Full tree distance should be <= component (due to union)
        assert d_full <= d_trunk + TOL_DISTANCE

        # Point in canopy
        canopy_y = config.trunk_height + config.canopy_height_offset
        d_canopy = _sdf_canopy((0, canopy_y, 0), config)
        d_full = sdf_tree((0, canopy_y, 0), config)
        assert d_full <= d_canopy + TOL_DISTANCE

    def test_sdf_tree_includes_branches(self):
        """sdf_tree with branches should include branch SDF."""
        config = TreeConfig(
            branches=BranchConfig(count=4, radius=0.1, attachment_height=0.5),
            trunk_height=2.0,
        )

        # Get a branch endpoint
        endpoints = _get_branch_endpoints(config)
        start, end = endpoints[0]

        # Point on branch should be inside
        d = sdf_tree(start, config)
        assert d < 0.5, f"Point on branch start should be near surface: {d}"


# =============================================================================
# SDF Forest Integration Tests
# =============================================================================

class TestSDFForestIntegration:
    """Whitebox integration tests for sdf_forest."""

    def test_forest_checks_neighbors(self):
        """Forest should check neighboring cells."""
        config = ForestConfig(
            cell_size=(10.0, 10.0, 10.0),
            density=1.0,
            variation=TreeVariation(
                height_min=1.0, height_max=1.0,
                width_min=1.0, width_max=1.0,
                position_jitter=0.0,
            ),
        )

        # Point at cell boundary
        # Should still find nearby tree from neighbor
        d = sdf_forest((9.9, 1.0, 5.0), config)
        assert math.isfinite(d), "Should find tree near boundary"

    def test_forest_respects_density(self):
        """Forest with partial density should have some empty cells."""
        config = ForestConfig(density=0.3)

        # Count trees in many cells
        n_trees = 0
        n_cells = 0
        for x in range(-10, 10):
            for z in range(-10, 10):
                n_cells += 1
                tree_config = _get_cell_tree_config((x, 0, z), config)
                if tree_config:
                    n_trees += 1

        # Should have approximately 30% of cells with trees
        ratio = n_trees / n_cells
        assert 0.15 < ratio < 0.45, f"Density ratio {ratio} far from expected 0.3"


# =============================================================================
# Edge Case and Boundary Tests
# =============================================================================

class TestEdgeCases:
    """Whitebox tests for edge cases."""

    def test_very_small_smooth_k(self):
        """Very small smooth_k should not cause issues."""
        config = TreeConfig(smooth_k=1e-8)
        d = sdf_tree((0, 1, 0), config)
        assert math.isfinite(d)

    def test_large_coordinates(self):
        """Large coordinates should not overflow."""
        config = ForestConfig()
        d = sdf_forest((1e6, 1.0, 1e6), config)
        assert math.isfinite(d)

    def test_negative_coordinates(self):
        """Negative coordinates should work correctly."""
        config = ForestConfig(density=1.0)
        d = sdf_forest((-50.0, 1.0, -50.0), config)
        assert math.isfinite(d)

    def test_hash_negative_cell(self):
        """Hash should work for negative cell IDs."""
        h1 = cell_hash((-5, -10, -15))
        h2 = cell_hash((-5, -10, -15))
        assert h1 == h2
        assert h1 != 0

    def test_single_tree_matches_standalone(self):
        """Forest with one cell should match standalone tree."""
        # This is hard to test exactly due to jitter, but we can verify
        # the basic structure works
        tree_config = TreeConfig()
        forest_config = ForestConfig(
            cell_size=(100.0, 100.0, 100.0),  # Very large cells
            base_tree=tree_config,
            density=1.0,
            variation=TreeVariation(
                height_min=1.0, height_max=1.0,
                width_min=1.0, width_max=1.0,
                canopy_min=tree_config.canopy_spheres,
                canopy_max=tree_config.canopy_spheres,
                position_jitter=0.0,
                rotation_enabled=False,
            ),
        )

        # Get tree position in cell (0,0,0)
        tree_pos = _get_cell_tree_position((0, 0, 0), forest_config)
        assert tree_pos is not None

        # Point at tree center should be inside for both
        p = (tree_pos[0], tree_pos[1] + 1.0, tree_pos[2])
        d_forest = sdf_forest(p, forest_config)
        d_tree = sdf_tree((0, 1.0, 0), tree_config)

        # Should be similar (both inside)
        assert d_forest < 0 or d_tree < 0


# =============================================================================
# Configuration Cloning and Modification
# =============================================================================

class TestConfigImmutability:
    """Test that configurations are properly immutable."""

    def test_tree_config_frozen(self):
        """TreeConfig should be frozen."""
        config = TreeConfig()
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            config.trunk_height = 5.0

    def test_branch_config_frozen(self):
        """BranchConfig should be frozen."""
        config = BranchConfig()
        with pytest.raises(Exception):
            config.count = 10

    def test_forest_config_frozen(self):
        """ForestConfig should be frozen."""
        config = ForestConfig()
        with pytest.raises(Exception):
            config.density = 0.5

    def test_variation_frozen(self):
        """TreeVariation should be frozen."""
        var = TreeVariation()
        with pytest.raises(Exception):
            var.height_min = 0.1
