"""
Whitebox tests for T-DEMO-3.4: Normal Estimation (Central Differences)

Tests the internal implementation of estimate_normal and the
NormalEstimator class.

Test coverage:
- 6-point central differences formula
- Tetrahedron stencil alternative
- Normal unit-length verification
- Outward-pointing normal verification
- Edge cases and error handling
"""

import math
import pytest

from engine.rendering.demoscene.sdf_ast import Vec3
from engine.rendering.demoscene.ray_march import (
    estimate_normal,
    NormalEstimationConfig,
    NormalEstimator,
    sdf_sphere,
    sdf_box,
    sdf_plane,
    sdf_torus,
    sdf_cylinder,
    DEFAULT_EPSILON,
)


# =============================================================================
# Helper Functions
# =============================================================================

def vec3_approx_equal(v1: Vec3, v2: Vec3, rel: float = 1e-5) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < rel and
        abs(v1.y - v2.y) < rel and
        abs(v1.z - v2.z) < rel
    )


def is_unit_length(v: Vec3, tol: float = 1e-5) -> bool:
    """Check if vector is unit length."""
    return abs(v.length() - 1.0) < tol


def dot(v1: Vec3, v2: Vec3) -> float:
    """Compute dot product of two Vec3."""
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


# =============================================================================
# T-DEMO-3.4.1: Sphere Normal Tests
# =============================================================================

class TestSphereNormals:
    """Tests for normal estimation on spheres."""

    def test_sphere_normal_positive_x(self):
        """Normal at +X should point in +X direction."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        normal = estimate_normal(Vec3(1.0, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert normal.x == pytest.approx(1.0, abs=1e-4)
        assert abs(normal.y) < 1e-4
        assert abs(normal.z) < 1e-4

    def test_sphere_normal_negative_x(self):
        """Normal at -X should point in -X direction."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        normal = estimate_normal(Vec3(-1.0, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert normal.x == pytest.approx(-1.0, abs=1e-4)
        assert abs(normal.y) < 1e-4
        assert abs(normal.z) < 1e-4

    def test_sphere_normal_positive_y(self):
        """Normal at +Y should point in +Y direction."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        normal = estimate_normal(Vec3(0.0, 1.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert abs(normal.x) < 1e-4
        assert normal.y == pytest.approx(1.0, abs=1e-4)
        assert abs(normal.z) < 1e-4

    def test_sphere_normal_positive_z(self):
        """Normal at +Z should point in +Z direction."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        normal = estimate_normal(Vec3(0.0, 0.0, 1.0), sdf)

        assert is_unit_length(normal)
        assert abs(normal.x) < 1e-4
        assert abs(normal.y) < 1e-4
        assert normal.z == pytest.approx(1.0, abs=1e-4)

    def test_sphere_normal_diagonal(self):
        """Normal at diagonal point should point diagonally."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        # Point on sphere at (1,1,1) direction
        inv_sqrt3 = 1.0 / math.sqrt(3)
        p = Vec3(inv_sqrt3, inv_sqrt3, inv_sqrt3)
        normal = estimate_normal(p, sdf)

        assert is_unit_length(normal)
        # All components should be equal (diagonal)
        assert normal.x == pytest.approx(inv_sqrt3, abs=1e-4)
        assert normal.y == pytest.approx(inv_sqrt3, abs=1e-4)
        assert normal.z == pytest.approx(inv_sqrt3, abs=1e-4)

    def test_sphere_different_radii(self):
        """Normals should be consistent for different sphere radii."""
        for radius in [0.5, 1.0, 2.0, 10.0]:
            sdf = lambda p, r=radius: sdf_sphere(p, r)
            # Point on surface
            normal = estimate_normal(Vec3(radius, 0.0, 0.0), sdf)

            assert is_unit_length(normal)
            assert normal.x == pytest.approx(1.0, abs=1e-4)


# =============================================================================
# T-DEMO-3.4.2: Box Normal Tests
# =============================================================================

class TestBoxNormals:
    """Tests for normal estimation on boxes."""

    def test_box_normal_positive_x_face(self):
        """Normal on +X face should point in +X direction."""
        half = Vec3(1.0, 1.0, 1.0)
        sdf = lambda p: sdf_box(p, half)
        # Point on +X face, centered
        normal = estimate_normal(Vec3(1.0, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert normal.x == pytest.approx(1.0, abs=1e-3)

    def test_box_normal_negative_y_face(self):
        """Normal on -Y face should point in -Y direction."""
        half = Vec3(1.0, 1.0, 1.0)
        sdf = lambda p: sdf_box(p, half)
        normal = estimate_normal(Vec3(0.0, -1.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert normal.y == pytest.approx(-1.0, abs=1e-3)

    def test_box_normal_positive_z_face(self):
        """Normal on +Z face should point in +Z direction."""
        half = Vec3(1.0, 1.0, 1.0)
        sdf = lambda p: sdf_box(p, half)
        normal = estimate_normal(Vec3(0.0, 0.0, 1.0), sdf)

        assert is_unit_length(normal)
        assert normal.z == pytest.approx(1.0, abs=1e-3)

    def test_box_asymmetric_extents(self):
        """Box with asymmetric extents should have correct normals."""
        half = Vec3(2.0, 0.5, 1.0)
        sdf = lambda p: sdf_box(p, half)

        # +X face
        normal_x = estimate_normal(Vec3(2.0, 0.0, 0.0), sdf)
        assert normal_x.x == pytest.approx(1.0, abs=1e-3)

        # +Y face
        normal_y = estimate_normal(Vec3(0.0, 0.5, 0.0), sdf)
        assert normal_y.y == pytest.approx(1.0, abs=1e-3)


# =============================================================================
# T-DEMO-3.4.3: Plane Normal Tests
# =============================================================================

class TestPlaneNormals:
    """Tests for normal estimation on planes."""

    def test_plane_normal_y_up(self):
        """Horizontal plane should have Y-up normal."""
        normal_vec = Vec3(0.0, 1.0, 0.0)
        sdf = lambda p: sdf_plane(p, normal_vec)

        # Sample anywhere on plane
        normal = estimate_normal(Vec3(5.0, 0.0, 3.0), sdf)

        assert is_unit_length(normal)
        assert abs(normal.x) < 1e-4
        assert normal.y == pytest.approx(1.0, abs=1e-4)
        assert abs(normal.z) < 1e-4

    def test_plane_normal_diagonal(self):
        """Diagonal plane should have diagonal normal."""
        inv_sqrt3 = 1.0 / math.sqrt(3)
        normal_vec = Vec3(inv_sqrt3, inv_sqrt3, inv_sqrt3)
        sdf = lambda p: sdf_plane(p, normal_vec)

        normal = estimate_normal(Vec3(0.0, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert normal.x == pytest.approx(inv_sqrt3, abs=1e-4)
        assert normal.y == pytest.approx(inv_sqrt3, abs=1e-4)
        assert normal.z == pytest.approx(inv_sqrt3, abs=1e-4)

    def test_plane_normal_consistent_everywhere(self):
        """Plane normal should be same everywhere."""
        normal_vec = Vec3(0.0, 1.0, 0.0)
        sdf = lambda p: sdf_plane(p, normal_vec)

        positions = [
            Vec3(0.0, 0.0, 0.0),
            Vec3(100.0, 0.0, 0.0),
            Vec3(-50.0, 0.0, 50.0),
            Vec3(0.0, 0.0, -1000.0),
        ]

        normals = [estimate_normal(p, sdf) for p in positions]

        for n in normals:
            assert is_unit_length(n)
            assert vec3_approx_equal(n, Vec3(0.0, 1.0, 0.0))


# =============================================================================
# T-DEMO-3.4.4: Torus Normal Tests
# =============================================================================

class TestTorusNormals:
    """Tests for normal estimation on tori."""

    def test_torus_outer_edge(self):
        """Normal at outer edge (+X) should point outward."""
        major, minor = 2.0, 0.5
        sdf = lambda p: sdf_torus(p, major, minor)

        # Outer edge: major + minor along X
        normal = estimate_normal(Vec3(major + minor, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert normal.x == pytest.approx(1.0, abs=1e-3)

    def test_torus_inner_edge(self):
        """Normal at inner edge should point inward (toward center)."""
        major, minor = 2.0, 0.5
        sdf = lambda p: sdf_torus(p, major, minor)

        # Inner edge: major - minor along X
        normal = estimate_normal(Vec3(major - minor, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        # Should point in -X direction (toward hole)
        assert normal.x == pytest.approx(-1.0, abs=1e-3)

    def test_torus_top(self):
        """Normal at top should point upward."""
        major, minor = 2.0, 0.5
        sdf = lambda p: sdf_torus(p, major, minor)

        # Top of torus at (major, minor, 0)
        normal = estimate_normal(Vec3(major, minor, 0.0), sdf)

        assert is_unit_length(normal)
        # Should point primarily in +Y
        assert normal.y == pytest.approx(1.0, abs=1e-3)


# =============================================================================
# T-DEMO-3.4.5: Cylinder Normal Tests
# =============================================================================

class TestCylinderNormals:
    """Tests for normal estimation on cylinders."""

    def test_cylinder_side_positive_x(self):
        """Normal on side at +X should point in +X direction."""
        radius, height = 1.0, 2.0
        sdf = lambda p: sdf_cylinder(p, radius, height)

        # On side, at Y=0
        normal = estimate_normal(Vec3(radius, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert normal.x == pytest.approx(1.0, abs=1e-3)

    def test_cylinder_side_diagonal(self):
        """Normal on side at XZ diagonal should point diagonally."""
        radius, height = 1.0, 2.0
        sdf = lambda p: sdf_cylinder(p, radius, height)

        # On side at 45 degrees in XZ
        inv_sqrt2 = 1.0 / math.sqrt(2)
        normal = estimate_normal(Vec3(inv_sqrt2, 0.0, inv_sqrt2), sdf)

        assert is_unit_length(normal)
        assert normal.x == pytest.approx(inv_sqrt2, abs=1e-3)
        assert abs(normal.y) < 1e-3
        assert normal.z == pytest.approx(inv_sqrt2, abs=1e-3)


# =============================================================================
# T-DEMO-3.4.6: Unit Length Verification
# =============================================================================

class TestNormalUnitLength:
    """Verify all normals are unit length."""

    @pytest.mark.parametrize("sdf_func,pos", [
        (lambda p: sdf_sphere(p, 1.0), Vec3(1.0, 0.0, 0.0)),
        (lambda p: sdf_sphere(p, 2.0), Vec3(0.0, 2.0, 0.0)),
        (lambda p: sdf_box(p, Vec3(1, 1, 1)), Vec3(1.0, 0.0, 0.0)),
        (lambda p: sdf_plane(p, Vec3(0, 1, 0)), Vec3(0.0, 0.0, 0.0)),
        (lambda p: sdf_torus(p, 2.0, 0.5), Vec3(2.5, 0.0, 0.0)),
        (lambda p: sdf_cylinder(p, 1.0, 2.0), Vec3(1.0, 0.0, 0.0)),
    ])
    def test_unit_length(self, sdf_func, pos):
        """All normals should be unit length."""
        normal = estimate_normal(pos, sdf_func)
        assert abs(normal.length() - 1.0) < 1e-5


# =============================================================================
# T-DEMO-3.4.7: Outward Pointing Verification
# =============================================================================

class TestNormalOutwardPointing:
    """Verify normals point outward from surface."""

    def test_sphere_normal_points_away_from_center(self):
        """Sphere normal should point away from center."""
        sdf = lambda p: sdf_sphere(p, 1.0)

        test_points = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(-1.0, 0.0, 0.0),
            Vec3(0.577, 0.577, 0.577),
        ]

        for p in test_points:
            normal = estimate_normal(p, sdf)
            # Normal should point in same direction as position (from center)
            p_normalized = p.normalized()
            dot_product = dot(normal, p_normalized)
            assert dot_product > 0.99  # Should be nearly 1.0

    def test_box_normal_points_outward(self):
        """Box normal should point outward from surface."""
        half = Vec3(1.0, 1.0, 1.0)
        sdf = lambda p: sdf_box(p, half)

        # Point on +X face
        normal = estimate_normal(Vec3(1.0, 0.5, 0.3), sdf)
        # Stepping along normal should increase distance
        p_outer = Vec3(1.0 + normal.x * 0.1, 0.5 + normal.y * 0.1, 0.3 + normal.z * 0.1)
        p_inner = Vec3(1.0 - normal.x * 0.1, 0.5 - normal.y * 0.1, 0.3 - normal.z * 0.1)

        assert sdf(p_outer) > sdf(p_inner)


# =============================================================================
# T-DEMO-3.4.8: Epsilon Parameter Tests
# =============================================================================

class TestNormalEpsilon:
    """Tests for epsilon parameter behavior."""

    def test_smaller_epsilon_more_accurate(self):
        """Smaller epsilon should give more accurate results."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        p = Vec3(1.0, 0.0, 0.0)

        # Expected: exactly (1, 0, 0)
        normal_small = estimate_normal(p, sdf, epsilon=0.0001)
        normal_medium = estimate_normal(p, sdf, epsilon=0.01)
        normal_large = estimate_normal(p, sdf, epsilon=0.1)

        # Smaller epsilon should be closer to exact
        assert abs(normal_small.x - 1.0) <= abs(normal_large.x - 1.0)

    def test_zero_epsilon_raises(self):
        """Zero epsilon should raise ValueError."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        with pytest.raises(ValueError, match="positive"):
            estimate_normal(Vec3(1.0, 0.0, 0.0), sdf, epsilon=0.0)

    def test_negative_epsilon_raises(self):
        """Negative epsilon should raise ValueError."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        with pytest.raises(ValueError, match="positive"):
            estimate_normal(Vec3(1.0, 0.0, 0.0), sdf, epsilon=-0.001)


# =============================================================================
# T-DEMO-3.4.9: NormalEstimator Class Tests
# =============================================================================

class TestNormalEstimator:
    """Tests for NormalEstimator class."""

    def test_default_config(self):
        """Default estimator should work correctly."""
        estimator = NormalEstimator()
        sdf = lambda p: sdf_sphere(p, 1.0)

        normal = estimator.estimate(Vec3(1.0, 0.0, 0.0), sdf)
        assert is_unit_length(normal)
        assert normal.x == pytest.approx(1.0, abs=1e-4)

    def test_custom_epsilon(self):
        """Custom epsilon should be used."""
        config = NormalEstimationConfig(epsilon=0.01)
        estimator = NormalEstimator(config)

        assert estimator.epsilon == 0.01

    def test_tetrahedron_stencil(self):
        """Tetrahedron stencil should produce valid normals."""
        config = NormalEstimationConfig(use_tetrahedron=True)
        estimator = NormalEstimator(config)
        sdf = lambda p: sdf_sphere(p, 1.0)

        normal = estimator.estimate(Vec3(1.0, 0.0, 0.0), sdf)
        assert is_unit_length(normal)
        # Tetrahedron is slightly less accurate but should be close
        assert normal.x > 0.99

    def test_tetrahedron_vs_central_differences(self):
        """Tetrahedron and central differences should give similar results."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        p = Vec3(0.577, 0.577, 0.577)  # Diagonal

        config_cd = NormalEstimationConfig(use_tetrahedron=False)
        config_tetra = NormalEstimationConfig(use_tetrahedron=True)

        est_cd = NormalEstimator(config_cd)
        est_tetra = NormalEstimator(config_tetra)

        n_cd = est_cd.estimate(p, sdf)
        n_tetra = est_tetra.estimate(p, sdf)

        # Should be within reasonable tolerance
        assert abs(n_cd.x - n_tetra.x) < 0.01
        assert abs(n_cd.y - n_tetra.y) < 0.01
        assert abs(n_cd.z - n_tetra.z) < 0.01


# =============================================================================
# T-DEMO-3.4.10: Config Validation Tests
# =============================================================================

class TestNormalEstimationConfig:
    """Tests for NormalEstimationConfig validation."""

    def test_valid_config(self):
        """Valid config should not raise."""
        config = NormalEstimationConfig(epsilon=0.001)
        assert config.epsilon == 0.001

    def test_zero_epsilon_raises(self):
        """Zero epsilon in config should raise."""
        with pytest.raises(ValueError, match="positive"):
            NormalEstimationConfig(epsilon=0.0)

    def test_negative_epsilon_raises(self):
        """Negative epsilon in config should raise."""
        with pytest.raises(ValueError, match="positive"):
            NormalEstimationConfig(epsilon=-0.001)


# =============================================================================
# T-DEMO-3.4.11: Edge Cases
# =============================================================================

class TestNormalEdgeCases:
    """Edge case tests for normal estimation."""

    def test_very_small_epsilon(self):
        """Very small epsilon should still work."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        normal = estimate_normal(Vec3(1.0, 0.0, 0.0), sdf, epsilon=1e-8)

        assert is_unit_length(normal)

    def test_large_epsilon_still_valid(self):
        """Large epsilon should produce valid (if imprecise) normal."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        normal = estimate_normal(Vec3(1.0, 0.0, 0.0), sdf, epsilon=0.5)

        # Should still be unit length even if imprecise
        assert is_unit_length(normal, tol=1e-4)

    def test_point_far_from_surface(self):
        """Normal estimation far from surface should still work."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        # Point at 10 units from origin (well outside sphere)
        normal = estimate_normal(Vec3(10.0, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        # Should still point outward
        assert normal.x > 0.99

    def test_point_inside_surface(self):
        """Normal estimation inside surface should work."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        # Point at 0.5 (inside sphere of radius 1)
        normal = estimate_normal(Vec3(0.5, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        # Should still point outward (in +X direction)
        assert normal.x > 0.99


# =============================================================================
# T-DEMO-3.4.12: Central Differences Formula Verification
# =============================================================================

class TestCentralDifferencesFormula:
    """Verify the central differences formula is correct."""

    def test_formula_x_component(self):
        """X component should be sdf(p+ex) - sdf(p-ex)."""
        epsilon = 0.001

        def custom_sdf(p: Vec3) -> float:
            # Linear SDF for testing: d = x
            return p.x

        p = Vec3(0.0, 0.0, 0.0)
        normal = estimate_normal(p, custom_sdf, epsilon)

        # For d = x, gradient is (1, 0, 0)
        assert normal.x == pytest.approx(1.0, abs=1e-4)
        assert abs(normal.y) < 1e-4
        assert abs(normal.z) < 1e-4

    def test_formula_y_component(self):
        """Y component should be sdf(p+ey) - sdf(p-ey)."""
        epsilon = 0.001

        def custom_sdf(p: Vec3) -> float:
            # Linear SDF: d = y
            return p.y

        p = Vec3(0.0, 0.0, 0.0)
        normal = estimate_normal(p, custom_sdf, epsilon)

        # For d = y, gradient is (0, 1, 0)
        assert abs(normal.x) < 1e-4
        assert normal.y == pytest.approx(1.0, abs=1e-4)
        assert abs(normal.z) < 1e-4

    def test_formula_z_component(self):
        """Z component should be sdf(p+ez) - sdf(p-ez)."""
        epsilon = 0.001

        def custom_sdf(p: Vec3) -> float:
            # Linear SDF: d = z
            return p.z

        p = Vec3(0.0, 0.0, 0.0)
        normal = estimate_normal(p, custom_sdf, epsilon)

        # For d = z, gradient is (0, 0, 1)
        assert abs(normal.x) < 1e-4
        assert abs(normal.y) < 1e-4
        assert normal.z == pytest.approx(1.0, abs=1e-4)

    def test_formula_diagonal_gradient(self):
        """Diagonal gradient should produce diagonal normal."""
        epsilon = 0.001

        def custom_sdf(p: Vec3) -> float:
            # Linear SDF: d = x + y + z
            return p.x + p.y + p.z

        p = Vec3(0.0, 0.0, 0.0)
        normal = estimate_normal(p, custom_sdf, epsilon)

        # For d = x+y+z, gradient is (1,1,1), normalized to (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
        inv_sqrt3 = 1.0 / math.sqrt(3)
        assert normal.x == pytest.approx(inv_sqrt3, abs=1e-4)
        assert normal.y == pytest.approx(inv_sqrt3, abs=1e-4)
        assert normal.z == pytest.approx(inv_sqrt3, abs=1e-4)
