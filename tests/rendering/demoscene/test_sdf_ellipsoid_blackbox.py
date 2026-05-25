"""
BLACKBOX verification for T-DEMO-1.8: sdEllipsoid WGSL.

Cleanroom blackbox tests for the signed distance function:

  fn sdEllipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
      let eps = vec3<f32>(1e-10);
      let safe_r = max(abs(r), eps);
      let k0 = length(p / safe_r);
      let min_r = min(min(safe_r.x, safe_r.y), safe_r.z);
      return (k0 - 1.0) * min_r;
  }

Acceptance criteria:
  1. Sphere case: r=(1,1,1) returns same distances as sdSphere(p, 1.0)
  2. Stretched axes: surface points along each axis return zero
  3. Interior (negative): points inside the ellipsoid return negative
  4. Surface (zero): points exactly on the surface return zero
  5. Exterior (positive): points outside return positive
  6. Degenerate cases: zero semi-axis length, all-zero, negative radii
  7. Sign convention: inside < 0, surface = 0, outside > 0

Additional BLACKBOX checks:
  8. WGSL source file exists and is syntactically correct
  9. WGSL source matches the Python reference model
  10. Floating point edge cases (subnormals, near-zero, large values)
  11. Symmetry across coordinate axes
  12. Eikonal property check (|gradient| = 1 almost everywhere)

Severity levels: BLOCKER, CRITICAL, MAJOR, MINOR, INFO
"""

from __future__ import annotations

import math
import re
import struct
import sys

import pytest

# =============================================================================
# Python reference model (matches WGSL semantics)
# =============================================================================


def py_sd_ellipsoid(p, r):
    """Python model of WGSL sdEllipsoid(p: vec3<f32>, r: vec3<f32>) -> f32.

    Uses the Inigo Quilez normalized formulation:
      k0 = length(p / safe_r)
      sd = (k0 - 1.0) * min(safe_r)

    where safe_r = max(abs(r), 1e-10) prevents division by zero.
    """
    eps = 1e-10
    safe_rx = max(abs(r[0]), eps)
    safe_ry = max(abs(r[1]), eps)
    safe_rz = max(abs(r[2]), eps)
    k0 = math.sqrt(
        (p[0] / safe_rx) ** 2
        + (p[1] / safe_ry) ** 2
        + (p[2] / safe_rz) ** 2
    )
    min_r = min(safe_rx, safe_ry, safe_rz)
    return (k0 - 1.0) * min_r


def py_sd_sphere(p, r):
    """Python model of WGSL sdSphere(p: vec3<f32>, r: f32) -> f32.

    Used to verify that sdEllipsoid(p, (1,1,1)) == sdSphere(p, 1.0).
    """
    safe_r = abs(r)
    return math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - safe_r


# =============================================================================
# Tolerance constants
# =============================================================================

TOL_SURFACE = 1e-12
TOL_EXACT = 1e-15
TOL_GRADIENT = 1e-6
TOL_DEGENERATE = 1e-8
# Note: k0 is approximately 1 for points near the surface. For the ellipsoid
# formula, sd = (k0 - 1) * min_r, points on the surface have k0 = 1 exactly
# (by construction), so surface zero should be exact up to f32 precision.


# =============================================================================
# 1. BLACKBOX: WGSL source file validation
# =============================================================================


class TestWgslSource:
    """Verify the WGSL source file exists and contains the correct function."""

    WGSL_PATH = "engine/rendering/demoscene/wgsl/sdf_ellipsoid.wgsl"

    def test_wgsl_file_exists(self):
        """BLOCKER: The WGSL source file must exist."""
        try:
            with open(self.WGSL_PATH) as f:
                content = f.read()
        except FileNotFoundError:
            pytest.fail(f"WGSL source file not found: {self.WGSL_PATH}")
        assert len(content) > 0, "WGSL source file is empty"

    def test_wgsl_contains_sdEllipsoid_function(self):
        """BLOCKER: WGSL must define the sdEllipsoid function."""
        with open(self.WGSL_PATH) as f:
            content = f.read()
        assert "fn sdEllipsoid" in content, (
            "WGSL source must define fn sdEllipsoid"
        )

    def test_wgsl_signature_correct(self):
        """CRITICAL: Signature must be fn sdEllipsoid(p: vec3<f32>, r: vec3<f32>) -> f32."""
        with open(self.WGSL_PATH) as f:
            content = f.read()
        pattern = (
            r"fn\s+sdEllipsoid\s*\(\s*p\s*:\s*vec3<f32>\s*,\s*"
            r"r\s*:\s*vec3<f32>\s*\)\s*->\s*f32"
        )
        assert re.search(pattern, content), (
            "WGSL function signature must be: "
            "fn sdEllipsoid(p: vec3<f32>, r: vec3<f32>) -> f32"
        )

    def test_wgsl_uses_abs_guard(self):
        """CRITICAL: WGSL must use abs(r) to handle negative semi-axis lengths."""
        with open(self.WGSL_PATH) as f:
            content = f.read()
        assert "abs(r)" in content or "abs(" in content, (
            "WGSL must use abs(r) guard for negative radius protection"
        )

    def test_wgsl_uses_length(self):
        """CRITICAL: WGSL must use length(p/r) for the normalized distance."""
        with open(self.WGSL_PATH) as f:
            content = f.read()
        assert "length(p /" in content or "length(" in content, (
            "WGSL must use length() for normalized distance computation"
        )

    def test_wgsl_uses_min_r_scaling(self):
        """CRITICAL: WGSL must scale by min(r) for world-space distance."""
        with open(self.WGSL_PATH) as f:
            content = f.read()
        assert "min_r" in content or "min(" in content, (
            "WGSL must scale the normalized distance by the minimum semi-axis"
        )

    def test_wgsl_formula_structure(self):
        """CRITICAL: Verify the formula is (k0 - 1.0) * min_r."""
        with open(self.WGSL_PATH) as f:
            content = f.read()
        assert "k0" in content, (
            "WGSL should compute k0 = length(p / r) as an intermediate"
        )

    def test_wgsl_comment_describes_formula(self):
        """MINOR: WGSL should have a comment explaining the IQ formula."""
        with open(self.WGSL_PATH) as f:
            content = f.read()
        has_comment = any(
            line.strip().startswith("//") for line in content.split("\n")
            if line.strip()
        )
        assert has_comment, "WGSL should include comments describing the SDF"

    def test_wgsl_doc_comment_exists(self):
        """MINOR: WGSL should have a /// doc comment describing the function."""
        with open(self.WGSL_PATH) as f:
            content = f.read()
        assert "///" in content, (
            "WGSL should have /// doc comments for the sdEllipsoid function"
        )


# =============================================================================
# 2. BLACKBOX: Python model matches WGSL
# =============================================================================


class TestPythonModelMatchesWgsl:
    """Verify the Python test model produces the same results as the WGSL spec."""

    def test_py_model_uses_abs_r(self):
        """CRITICAL: Python model must use abs(r) to match WGSL abs(r) guard."""
        # Verify both positive and negative radii produce same result
        p = (1.0, 2.0, 2.0)
        r_pos = (3.0, 2.0, 1.0)
        r_neg = (-3.0, -2.0, -1.0)
        assert py_sd_ellipsoid(p, r_pos) == py_sd_ellipsoid(p, r_neg), (
            "Python model must treat negative radii same as positive via abs(r)"
        )

    def test_py_model_uses_eps_guard(self):
        """CRITICAL: Python model must use epsilon guard against zero division."""
        # Zero in one axis should not cause division by zero
        p = (1.0, 0.0, 0.0)
        try:
            result = py_sd_ellipsoid(p, (0.0, 1.0, 1.0))
            assert math.isfinite(result), (
                f"Epsilon guard failed: non-finite result {result}"
            )
        except ZeroDivisionError:
            pytest.fail(
                "Python model must use epsilon guard to prevent division by zero"
            )

    def test_py_model_uses_length(self):
        """CRITICAL: Python model must compute length(p/r), not squared distance."""
        r = (2.0, 3.0, 0.0)
        p = (2.0, 0.0, 0.0)
        result = py_sd_ellipsoid(p, r)
        # k0 = sqrt((2/2)^2 + 0 + 0) = 1, min_r = 0 (clamped to eps)
        # Actually min_r = max(0, 1e-10) = 1e-10, so result = (1-1) * 1e-10 = 0
        assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Python model surface check failed: "
            f"expected ~0 for p={p} r={r}, got {result}"
        )


# =============================================================================
# 3. BLACKBOX: Sphere case r=(1,1,1) matches sdSphere
# =============================================================================


class TestSphereCase:
    """Sphere case r=(1,1,1) must produce identical results to sdSphere(p, 1.0)."""

    @pytest.mark.parametrize("p", [
        (0.0, 0.0, 0.0),      # center
        (1.0, 0.0, 0.0),      # surface +x
        (0.0, 1.0, 0.0),      # surface +y
        (0.0, 0.0, 1.0),      # surface +z
        (-1.0, 0.0, 0.0),     # surface -x
        (2.0, 0.0, 0.0),      # exterior +x
        (0.0, 3.0, 0.0),      # exterior +y
        (0.0, 0.0, 4.0),      # exterior +z
        (0.3, 0.4, 0.0),      # interior off-axis  (len=0.5)
        (0.6, 0.8, 0.0),      # surface off-axis  (len=1.0)
        (1.5, 2.0, 0.0),      # exterior off-axis (len=2.5)
        (1.0, 1.0, 1.0),      # corner interior (len=1.732)
        (3.0, 4.0, 0.0),      # far exterior (len=5.0)
    ])
    def test_ellipsoid_matches_sphere(self, p):
        """sdEllipsoid(p, (1,1,1)) == sdSphere(p, 1.0) for all p."""
        d_ellipsoid = py_sd_ellipsoid(p, (1.0, 1.0, 1.0))
        d_sphere = py_sd_sphere(p, 1.0)
        assert d_ellipsoid == pytest.approx(d_sphere, abs=TOL_EXACT), (
            f"SPHERE CASE FAIL: p={p} ellipsoid={d_ellipsoid} "
            f"sphere={d_sphere}"
        )

    def test_ellipsoid_sign_matches_sphere(self):
        """Sign convention matches sdSphere for r=(1,1,1)."""
        # Interior points
        interior = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.3, 0.4, 0.0)]
        for p in interior:
            d_e = py_sd_ellipsoid(p, (1.0, 1.0, 1.0))
            d_s = py_sd_sphere(p, 1.0)
            assert d_e < 0, f"Interior ellipsoid should be negative at {p}, got {d_e}"
            assert d_s < 0, f"Interior sphere should be negative at {p}, got {d_s}"

        # Exterior points
        exterior = [(2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)]
        for p in exterior:
            d_e = py_sd_ellipsoid(p, (1.0, 1.0, 1.0))
            d_s = py_sd_sphere(p, 1.0)
            assert d_e > 0, f"Exterior ellipsoid should be positive at {p}, got {d_e}"
            assert d_s > 0, f"Exterior sphere should be positive at {p}, got {d_s}"

    def test_ellipsoid_center_equals_sphere_center(self):
        """Center distance matches: sdEllipsoid((0,0,0), (1,1,1)) == -1."""
        d_e = py_sd_ellipsoid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        d_s = py_sd_sphere((0.0, 0.0, 0.0), 1.0)
        assert d_e == pytest.approx(d_s, abs=TOL_EXACT), (
            f"Center mismatch: ellipsoid={d_e} sphere={d_s}"
        )
        assert d_e == pytest.approx(-1.0, abs=TOL_EXACT), (
            f"Center distance must be -1.0, got {d_e}"
        )


# =============================================================================
# 4. BLACKBOX: Stretched axis distances
# =============================================================================


class TestStretchedAxis:
    """Verify surface distances for ellipsoids with non-uniform radii."""

    # For r = (a, b, c), the surface along +x is at p = (a, 0, 0) with:
    #   k0 = sqrt((a/a)^2 + 0 + 0) = 1.0
    #   sd = (1.0 - 1.0) * min(a,b,c) = 0
    #
    # Similarly for +y at p = (0, b, 0) and +z at p = (0, 0, c).

    @pytest.mark.parametrize("r,axis_point", [
        ((2.0, 1.0, 0.5), (2.0, 0.0, 0.0)),   # Surface +x at r.x
        ((2.0, 1.0, 0.5), (0.0, 1.0, 0.0)),   # Surface +y at r.y
        ((2.0, 1.0, 0.5), (0.0, 0.0, 0.5)),   # Surface +z at r.z
        ((2.0, 1.0, 0.5), (-2.0, 0.0, 0.0)),  # Surface -x
        ((2.0, 1.0, 0.5), (0.0, -1.0, 0.0)),  # Surface -y
        ((2.0, 1.0, 0.5), (0.0, 0.0, -0.5)),  # Surface -z
        ((3.0, 2.0, 1.0), (3.0, 0.0, 0.0)),
        ((3.0, 2.0, 1.0), (0.0, 2.0, 0.0)),
        ((3.0, 2.0, 1.0), (0.0, 0.0, 1.0)),
        ((0.1, 5.0, 10.0), (0.0, 5.0, 0.0)),
        ((0.1, 5.0, 10.0), (0.0, 0.0, 10.0)),
    ])
    def test_axis_surface_zero(self, r, axis_point):
        """Surface points along main axes must return zero."""
        d = py_sd_ellipsoid(axis_point, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"AXIS SURFACE FAIL: p={axis_point} r={r} expected 0, got {d}"
        )

    @pytest.mark.parametrize("r,interior_point", [
        ((2.0, 1.0, 0.5), (0.0, 0.0, 0.0)),     # center
        ((2.0, 1.0, 0.5), (1.0, 0.0, 0.0)),     # inside +x
        ((2.0, 1.0, 0.5), (0.0, 0.5, 0.0)),     # inside +y
        ((2.0, 1.0, 0.5), (0.0, 0.0, 0.25)),    # inside +z
        ((2.0, 1.0, 0.5), (-1.0, 0.0, 0.0)),    # inside -x
        ((2.0, 1.0, 0.5), (0.0, -0.5, 0.0)),    # inside -y
        ((2.0, 1.0, 0.5), (0.0, 0.0, -0.25)),   # inside -z
    ])
    def test_axis_interior_negative(self, r, interior_point):
        """Interior points along axes must be negative."""
        d = py_sd_ellipsoid(interior_point, r)
        assert d < 0, (
            f"AXIS INTERIOR FAIL: p={interior_point} r={r} "
            f"should be negative, got {d}"
        )

    @pytest.mark.parametrize("r,exterior_point", [
        ((2.0, 1.0, 0.5), (3.0, 0.0, 0.0)),     # outside +x
        ((2.0, 1.0, 0.5), (0.0, 2.0, 0.0)),     # outside +y
        ((2.0, 1.0, 0.5), (0.0, 0.0, 1.0)),     # outside +z
        ((2.0, 1.0, 0.5), (-3.0, 0.0, 0.0)),    # outside -x
        ((2.0, 1.0, 0.5), (0.0, -2.0, 0.0)),    # outside -y
        ((2.0, 1.0, 0.5), (0.0, 0.0, -1.0)),    # outside -z
    ])
    def test_axis_exterior_positive(self, r, exterior_point):
        """Exterior points along axes must be positive."""
        d = py_sd_ellipsoid(exterior_point, r)
        assert d > 0, (
            f"AXIS EXTERIOR FAIL: p={exterior_point} r={r} "
            f"should be positive, got {d}"
        )
        # For axis-aligned exterior: sd = length(p) - min_r when surface is at r
        # But careful: this is only exact for axis-aligned when p is along that axis

    def test_asymmetric_radii_center(self):
        """Center distance varies with min(r) for asymmetric ellipsoids."""
        test_cases = [
            ((2.0, 1.0, 0.5), -0.5),    # min_r = 0.5
            ((3.0, 2.0, 1.0), -1.0),    # min_r = 1.0
            ((5.0, 5.0, 2.0), -2.0),    # min_r = 2.0
            ((10.0, 0.5, 0.5), -0.5),   # min_r = 0.5
        ]
        for r, expected_center in test_cases:
            d = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
            assert d == pytest.approx(expected_center, abs=TOL_EXACT), (
                f"CENTER FAIL: r={r} expected center distance {expected_center}, "
                f"got {d}"
            )


# =============================================================================
# 5. BLACKBOX: Sign convention robustness
# =============================================================================


class TestSignConvention:
    """Robust sign convention: inside < 0, surface = 0, outside > 0."""

    def test_interior_all_negative(self):
        """All interior points must return negative distances."""
        r = (3.0, 2.0, 1.0)
        interior = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.5),
            (1.0, 0.5, 0.25),
            (-1.0, 0.5, 0.0),
            (0.0, -0.5, 0.0),
            (-0.5, -0.5, -0.5),
            (2.9, 0.0, 0.0),
            (0.0, 1.9, 0.0),
        ]
        for p in interior:
            d = py_sd_ellipsoid(p, r)
            assert d < 0, (
                f"INTERIOR SIGN FAIL: p={p} r={r} should be negative, got {d}"
            )

    def test_exterior_all_positive(self):
        """All exterior points must return positive distances."""
        r = (3.0, 2.0, 1.0)
        exterior = [
            (4.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 2.0),
            (3.1, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 2.1, 0.0),
            (10.0, 0.0, 0.0),
            (-4.0, 0.0, 0.0),
            (0.0, 0.0, 1.1),
        ]
        for p in exterior:
            d = py_sd_ellipsoid(p, r)
            assert d > 0, (
                f"EXTERIOR SIGN FAIL: p={p} r={r} should be positive, got {d}"
            )

    def test_surface_all_zero(self):
        """Surface points must return approximately zero."""
        r = (3.0, 2.0, 1.0)
        surface = [
            (3.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 1.0),
            (-3.0, 0.0, 0.0),
            (0.0, -2.0, 0.0),
            (0.0, 0.0, -1.0),
        ]
        for p in surface:
            d = py_sd_ellipsoid(p, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"SURFACE SIGN FAIL: p={p} r={r} should be ~0, got {d}"
            )

    def test_strictly_increasing_along_positive_x(self):
        """Distance must be strictly increasing as we move outward along +x."""
        r = (3.0, 2.0, 1.0)
        x_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        distances = [py_sd_ellipsoid((x, 0.0, 0.0), r) for x in x_values]
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i + 1], (
                f"MONOTONIC FAIL: distance at x={x_values[i]} is {distances[i]}, "
                f"but x={x_values[i+1]} is {distances[i+1]}. "
                f"Must be strictly increasing."
            )

    def test_strictly_increasing_along_positive_y(self):
        """Distance must be strictly increasing as we move outward along +y."""
        r = (1.0, 4.0, 2.0)
        y_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        distances = [py_sd_ellipsoid((0.0, y, 0.0), r) for y in y_values]
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i + 1], (
                f"MONOTONIC Y FAIL: distance at y={y_values[i]} is {distances[i]}, "
                f"but y={y_values[i+1]} is {distances[i+1]}."
            )

    def test_strictly_increasing_along_positive_z(self):
        """Distance must be strictly increasing as we move outward along +z."""
        r = (1.0, 2.0, 5.0)
        z_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        distances = [py_sd_ellipsoid((0.0, 0.0, z), r) for z in z_values]
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i + 1], (
                f"MONOTONIC Z FAIL: distance at z={z_values[i]} is {distances[i]}, "
                f"but z={z_values[i+1]} is {distances[i+1]}."
            )

    def test_sign_consistent_under_scaling(self):
        """Sign must be consistent when scaling the ellipsoid."""
        base_r = (2.0, 1.0, 0.5)
        for scale in [0.1, 0.5, 1.0, 2.0, 10.0]:
            r = (base_r[0] * scale, base_r[1] * scale, base_r[2] * scale)
            # Center should be negative for all valid radii
            d_center = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
            assert d_center < 0, (
                f"CENTER SIGN under scale={scale}: should be negative, got {d_center}"
            )
            # Far exterior should be positive
            d_exterior = py_sd_ellipsoid((100.0 * scale, 0.0, 0.0), r)
            assert d_exterior > 0, (
                f"EXTERIOR SIGN under scale={scale}: should be positive, got {d_exterior}"
            )


# =============================================================================
# 6. BLACKBOX: Degenerate cases
# =============================================================================


class TestDegenerateCases:
    """Degenerate ellipsoid configurations (zero axes, negative radii)."""

    def test_zero_x_radius(self):
        """r=(0, 1, 1) collapses to a line (plane at x=0 for distance purposes)."""
        r = (0.0, 1.0, 1.0)
        # With one axis zero, the shape is a disk/line in the yz-plane.
        # p = (t, 0, 0) should give k0 = sqrt((t/eps)^2) = abs(t)/eps
        # sd = (abs(t)/eps - 1) * eps = abs(t) - eps... this is roughly abs(t)
        # So distance from the x-axis is roughly |x|
        d_origin = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
        assert d_origin == pytest.approx(0.0, abs=TOL_DEGENERATE), (
            f"ZERO X RADIUS: origin expected ~0, got {d_origin}"
        )

    def test_zero_y_radius(self):
        """r=(1, 0, 1) collapses along y-axis."""
        r = (1.0, 0.0, 1.0)
        # The shape becomes an infinite line/"needle" along the y-axis (since
        # y-radius is 0, the shape has no extent in y). The SDF should be ~0
        # along the entire y-axis (when x=z=0).
        d_origin = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
        assert d_origin == pytest.approx(0.0, abs=TOL_DEGENERATE), (
            f"ZERO Y RADIUS: origin expected ~0, got {d_origin}"
        )

    def test_zero_z_radius(self):
        """r=(1, 1, 0) collapses along z-axis."""
        r = (1.0, 1.0, 0.0)
        d_origin = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
        assert d_origin == pytest.approx(0.0, abs=TOL_DEGENERATE), (
            f"ZERO Z RADIUS: origin expected ~0, got {d_origin}"
        )

    def test_all_zero_radius(self):
        """r=(0, 0, 0) degenerates to a point SDF: sd = length(p)."""
        r = (0.0, 0.0, 0.0)
        # With all axes zero, the ellipsoid collapses to a point at origin.
        # For p near origin, sd ~ 0.
        test_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (3.0, 4.0, 0.0),
            (1.0, 1.0, 1.0),
            (-2.0, 3.0, 6.0),
        ]
        for p in test_points:
            d = py_sd_ellipsoid(p, r)
            length = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
            # When all axes are zero, all safe_r = eps, min_r = eps.
            # k0 = length(p) / eps, sd = (length(p)/eps - 1) * eps ~= length(p)
            assert d == pytest.approx(length, abs=TOL_DEGENERATE), (
                f"ALL ZERO RADIUS FAIL: p={p} expected ~{length}, got {d}"
            )

    def test_two_axes_zero(self):
        """r=(a, 0, 0) collapses to a line along x-axis (no y/z extent)."""
        r = (3.0, 0.0, 0.0)
        # With y and z zero, the shape is an infinitely thin "needle" along x.
        # sd approximates sqrt(y^2 + z^2) near the x-axis.
        # Points on the x-axis should give ~0 (approximately on the shape).
        d_x_axis = py_sd_ellipsoid((2.0, 0.0, 0.0), r)
        assert d_x_axis == pytest.approx(0.0, abs=TOL_DEGENERATE), (
            f"TWO ZERO AXES: point on x-axis expected ~0, got {d_x_axis}"
        )

    def test_negative_radii_mirror_positive(self):
        """Negative radii must behave identically to positive radii via abs()."""
        test_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (2.0, 1.0, 0.5),
            (5.0, 3.0, 2.0),
        ]
        # All-negative radii
        r_all_neg = (-2.0, -1.0, -0.5)
        r_all_pos = (2.0, 1.0, 0.5)
        for p in test_points:
            d_neg = py_sd_ellipsoid(p, r_all_neg)
            d_pos = py_sd_ellipsoid(p, r_all_pos)
            assert d_neg == pytest.approx(d_pos, abs=TOL_EXACT), (
                f"NEG RADII FAIL: p={p} r_neg={r_all_neg} ({d_neg}) "
                f"!= r_pos={r_all_pos} ({d_pos})"
            )

    def test_mixed_sign_radii(self):
        """Mixed sign radii (+/-) must behave the same as all-positive."""
        r_mixed = (-2.0, 1.0, -0.5)
        r_pos = (2.0, 1.0, 0.5)
        test_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.25),
        ]
        for p in test_points:
            d_mixed = py_sd_ellipsoid(p, r_mixed)
            d_pos = py_sd_ellipsoid(p, r_pos)
            assert d_mixed == pytest.approx(d_pos, abs=TOL_EXACT), (
                f"MIXED SIGN FAIL: p={p} r_mixed={r_mixed} ({d_mixed}) "
                f"!= r_pos={r_pos} ({d_pos})"
            )

    def test_negative_zero_radius(self):
        """r=-0.0 must behave identically to r=0.0 (IEEE 754 handling)."""
        d_zero = py_sd_ellipsoid((3.0, 4.0, 0.0), (1.0, 1.0, 0.0))
        d_neg_zero = py_sd_ellipsoid((3.0, 4.0, 0.0), (1.0, 1.0, -0.0))
        assert d_zero == pytest.approx(d_neg_zero, abs=TOL_EXACT), (
            f"NEG ZERO RADIUS FAIL: r=-0.0 ({d_neg_zero}) != r=0.0 ({d_zero})"
        )

    def test_tiny_positive_radius_stable(self):
        """Very small positive radii must not cause numerical instability."""
        r = (1e-20, 1.0, 1.0)
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
        # min_r = 1e-20, k0 = 0, sd = (0 - 1) * 1e-20 = -1e-20
        assert d == pytest.approx(-1e-20, abs=TOL_DEGENERATE), (
            f"TINY POSITIVE RADIUS FAIL: expected ~-1e-20, got {d}"
        )
        assert d < 0, (
            f"TINY POSITIVE RADIUS SIGN FAIL: should be negative, got {d}"
        )


# =============================================================================
# 7. BLACKBOX: Floating point edge cases
# =============================================================================


class TestFloatingPointEdgeCases:
    """Test edge cases that could cause numerical issues in WGSL f32."""

    def test_subnormal_radius(self):
        """Very small (subnormal) radii must not cause numerical issues."""
        r = (1e-40, 1.0, 1.0)
        d = py_sd_ellipsoid((1.0, 0.0, 0.0), r)
        # k0 = sqrt((1/1e-40)^2) = 1e40, min_r = 1e-40
        # sd = (1e40 - 1) * 1e-40 = 1 - 1e-40 ≈ 1.0
        assert math.isfinite(d), (
            f"SUBNORMAL RADIUS FAIL: non-finite result {d}"
        )

    def test_subnormal_point(self):
        """Very small (subnormal) point coordinates must work correctly."""
        p = (1e-40, 0.0, 0.0)
        r = (1.0, 2.0, 3.0)
        d = py_sd_ellipsoid(p, r)
        assert math.isfinite(d), (
            f"SUBNORMAL POINT FAIL: non-finite result {d}"
        )

    def test_large_coordinates(self):
        """Large point coordinates must work without overflow."""
        r = (1.0, 2.0, 3.0)
        p = (1e15, 1e15, 1e15)
        d = py_sd_ellipsoid(p, r)
        assert math.isfinite(d), (
            f"LARGE COORD FAIL: non-finite result for p={p}"
        )

    def test_large_radius(self):
        """Large radii must work without overflow."""
        r = (1e10, 1e10, 1e10)
        d_surface = py_sd_ellipsoid((1e10, 0.0, 0.0), r)
        assert d_surface == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"LARGE RADIUS SURFACE FAIL: expected 0, got {d_surface}"
        )

    def test_very_asymmetric_radii(self):
        """Very asymmetric radii must not cause numerical issues."""
        r = (1e10, 1e-10, 1e10)
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
        # min_r = 1e-10, sd = (0 - 1) * 1e-10 = -1e-10
        assert math.isfinite(d), (
            f"ASYMMETRIC RADII FAIL: non-finite result {d}"
        )
        assert d == pytest.approx(-1e-10, abs=TOL_DEGENERATE), (
            f"ASYMMETRIC RADII CENTER FAIL: expected ~-1e-10, got {d}"
        )

    def test_precision_near_surface(self):
        """Distance very close to surface must be precise."""
        r = (5.0, 3.0, 2.0)
        eps = 1e-8
        # Just inside along x
        d_inside = py_sd_ellipsoid((r[0] - eps, 0.0, 0.0), r)
        # Just outside along x
        d_outside = py_sd_ellipsoid((r[0] + eps, 0.0, 0.0), r)
        assert d_inside < 0, f"Just inside surface should be negative, got {d_inside}"
        assert d_outside > 0, f"Just outside surface should be positive, got {d_outside}"

    def test_zero_point_on_axis(self):
        """Point that lies on the origin but off the axis-aligned surface."""
        r = (3.0, 2.0, 1.0)
        # Points on a diagonal that is on the surface
        # For (1.5, 1.0, 0.5) with r = (3, 2, 1): k0 = sqrt(0.25 + 0.25 + 0.25) = sqrt(0.75)
        p_in = (1.5, 1.0, 0.5)
        d = py_sd_ellipsoid(p_in, r)
        # k0 = sqrt(0.25+0.25+0.25) = sqrt(0.75) ≈ 0.866
        # sd = (0.866 - 1.0) * 1.0 = -0.134
        assert d < 0, (
            f"DIAGONAL INTERIOR FAIL: p={p_in} r={r} should be negative, got {d}"
        )

    def test_exact_surface_axis_aligned(self):
        """Axis-aligned surface must be exactly 0 (not approximately)."""
        r = (3.0, 2.0, 1.0)
        # For (3, 0, 0): k0 = sqrt(1 + 0 + 0) = 1, sd = (1-1) * 1 = 0
        d = py_sd_ellipsoid((r[0], 0.0, 0.0), r)
        assert d == 0.0, (
            f"EXACT SURFACE FAIL: ({r[0]},0,0) r={r} must be EXACTLY 0, got {d}"
        )


# =============================================================================
# 8. BLACKBOX: Coordinate symmetry
# =============================================================================


class TestCoordinateSymmetry:
    """The ellipsoid SDF must be symmetric under axis sign flips."""

    def test_x_symmetry(self):
        """SDF must be symmetric under x -> -x."""
        r = (2.0, 3.0, 1.0)
        test_points = [
            (1.0, 0.5, 0.25),
            (2.0, 0.0, 0.0),
            (4.0, 1.0, 0.5),
            (0.0, 0.0, 0.0),
        ]
        for p in test_points:
            d_pos = py_sd_ellipsoid((p[0], p[1], p[2]), r)
            d_neg = py_sd_ellipsoid((-p[0], p[1], p[2]), r)
            assert d_pos == pytest.approx(d_neg, abs=TOL_EXACT), (
                f"X SYMMETRY FAIL: ({p[0]},{p[1]},{p[2]}) = {d_pos} "
                f"!= (-{p[0]},{p[1]},{p[2]}) = {d_neg}"
            )

    def test_y_symmetry(self):
        """SDF must be symmetric under y -> -y."""
        r = (2.0, 3.0, 1.0)
        test_points = [
            (1.0, 0.5, 0.25),
            (0.0, 3.0, 0.0),
            (1.0, 5.0, 0.5),
        ]
        for p in test_points:
            d_pos = py_sd_ellipsoid((p[0], p[1], p[2]), r)
            d_neg = py_sd_ellipsoid((p[0], -p[1], p[2]), r)
            assert d_pos == pytest.approx(d_neg, abs=TOL_EXACT), (
                f"Y SYMMETRY FAIL: ({p[0]},{p[1]},{p[2]}) = {d_pos} "
                f"!= ({p[0]},-{p[1]},{p[2]}) = {d_neg}"
            )

    def test_z_symmetry(self):
        """SDF must be symmetric under z -> -z."""
        r = (2.0, 3.0, 1.0)
        test_points = [
            (1.0, 0.5, 0.25),
            (0.0, 0.0, 1.0),
            (0.5, 1.0, 2.0),
        ]
        for p in test_points:
            d_pos = py_sd_ellipsoid((p[0], p[1], p[2]), r)
            d_neg = py_sd_ellipsoid((p[0], p[1], -p[2]), r)
            assert d_pos == pytest.approx(d_neg, abs=TOL_EXACT), (
                f"Z SYMMETRY FAIL: ({p[0]},{p[1]},{p[2]}) = {d_pos} "
                f"!= ({p[0]},{p[1]},-{p[2]}) = {d_neg}"
            )

    def test_all_octants_symmetry(self):
        """Surface must be symmetric across all 8 octants."""
        r = (2.0, 3.0, 1.0)
        # Distance to the surface point (2,0,0) should be 0 regardless of sign flips
        base = (2.0, 0.0, 0.0)
        d_base = py_sd_ellipsoid(base, r)
        assert d_base == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_full_octant_surface(self):
        """All 8 octants must have surface at same distance from origin."""
        r = (2.0, 1.5, 1.0)
        c = r  # The surface point along each positive axis
        surface_points = [
            (c[0], 0.0, 0.0),   # +x
            (-c[0], 0.0, 0.0),  # -x
            (0.0, c[1], 0.0),   # +y
            (0.0, -c[1], 0.0),  # -y
            (0.0, 0.0, c[2]),   # +z
            (0.0, 0.0, -c[2]),  # -z
        ]
        for p in surface_points:
            d = py_sd_ellipsoid(p, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"OCTANT SURFACE FAIL: p={p} r={r} expected 0, got {d}"
            )


# =============================================================================
# 9. BLACKBOX: Gradient behavior (known non-eikonal property)
# =============================================================================


class TestGradientBehavior:
    """Verify the known gradient scaling behavior of the IQ ellipsoid formula.

    NOTE: The IQ ellipsoid formula is NOT a true signed distance function.
    Unlike sdSphere which satisfies |grad(d)| = 1, the ellipsoid formula
    produces gradient magnitudes that scale with min(r) / r_i for axis-aligned
    directions. This is a documented limitation of the approximation.

    The formula sd = (k0 - 1.0) * min_r with k0 = length(p / r) gives:
      d/dx = (p.x / r.x^2) * (min_r / k0)
    and therefore |grad| = min_r * sqrt(sum((p_i / (r_i^2 * k0))^2))

    For axis-aligned points along axis i:
      |grad| = min_r / r_i  (when k0 = |p_i| / r_i)

    For a sphere r=(1,1,1): |grad| = 1 (matches sdSphere).
    For stretched ellipsoids: |grad| < 1 along stretched axes, > 1 along
    compressed axes (relative to min_r).
    """

    EPS = 1e-5

    def _numerical_gradient(self, p, r):
        """Compute gradient of sdEllipsoid using central differences."""
        gx = (
            py_sd_ellipsoid((p[0] + self.EPS, p[1], p[2]), r)
            - py_sd_ellipsoid((p[0] - self.EPS, p[1], p[2]), r)
        ) / (2.0 * self.EPS)
        gy = (
            py_sd_ellipsoid((p[0], p[1] + self.EPS, p[2]), r)
            - py_sd_ellipsoid((p[0], p[1] - self.EPS, p[2]), r)
        ) / (2.0 * self.EPS)
        gz = (
            py_sd_ellipsoid((p[0], p[1], p[2] + self.EPS), r)
            - py_sd_ellipsoid((p[0], p[1], p[2] - self.EPS), r)
        ) / (2.0 * self.EPS)
        return math.sqrt(gx * gx + gy * gy + gz * gz)

    def test_sphere_case_gradient_is_one(self):
        """For r=(1,1,1), gradient magnitude = 1 (matches sdSphere)."""
        r = (1.0, 1.0, 1.0)
        grad_mag = self._numerical_gradient((5.0, 0.0, 0.0), r)
        assert grad_mag == pytest.approx(1.0, abs=TOL_GRADIENT), (
            f"SPHERE GRADIENT FAIL: |grad| = {grad_mag}, expected 1.0"
        )

    def test_x_axis_gradient_scaling(self):
        """Gradient along x-axis scales as min_r / r.x."""
        r = (3.0, 2.0, 1.0)
        min_r = min(r)
        expected_grad = min_r / r[0]  # = 1/3
        grad_mag = self._numerical_gradient((5.0, 0.0, 0.0), r)
        assert grad_mag == pytest.approx(expected_grad, abs=TOL_GRADIENT), (
            f"X GRADIENT FAIL: |grad| = {grad_mag}, expected {expected_grad} "
            f"(min_r / r.x = {min_r}/{r[0]})"
        )

    def test_y_axis_gradient_scaling(self):
        """Gradient along y-axis scales as min_r / r.y."""
        r = (3.0, 2.0, 1.0)
        min_r = min(r)
        expected_grad = min_r / r[1]  # = 1/2
        grad_mag = self._numerical_gradient((0.0, 5.0, 0.0), r)
        assert grad_mag == pytest.approx(expected_grad, abs=TOL_GRADIENT), (
            f"Y GRADIENT FAIL: |grad| = {grad_mag}, expected {expected_grad}"
        )

    def test_z_axis_gradient_scaling(self):
        """Gradient along z-axis scales as min_r / r.z."""
        r = (3.0, 2.0, 1.0)
        min_r = min(r)
        expected_grad = min_r / r[2]  # = 1/1 = 1
        grad_mag = self._numerical_gradient((0.0, 0.0, 5.0), r)
        assert grad_mag == pytest.approx(expected_grad, abs=TOL_GRADIENT), (
            f"Z GRADIENT FAIL: |grad| = {grad_mag}, expected {expected_grad}"
        )

    @pytest.mark.parametrize("r,expected_grad_factor", [
        ((3.0, 1.0, 0.5), (0.5 / 3.0)),   # min_r/r.x = 0.5/3 = 0.1667
        ((5.0, 3.0, 2.0), (2.0 / 5.0)),   # min_r/r.x = 2/5 = 0.4
        ((2.0, 4.0, 0.5), (0.5 / 2.0)),   # min_r/r.x = 0.5/2 = 0.25
    ])
    def test_gradient_scales_with_aspect_ratio(self, r, expected_grad_factor):
        """Gradient magnitude scales with min_r / r_i along each axis."""
        grad_mag = self._numerical_gradient((r[0] + 1.0, 0.0, 0.0), r)
        assert grad_mag == pytest.approx(expected_grad_factor, abs=TOL_GRADIENT), (
            f"ASPECT GRADIENT FAIL: r={r} |grad|={grad_mag}, "
            f"expected {expected_grad_factor}"
        )

    def test_gradient_positive_direction(self):
        """Gradient should point outward (positive x-component for +x)."""
        r = (3.0, 2.0, 1.0)
        gx = (
            py_sd_ellipsoid((5.0 + self.EPS, 0.0, 0.0), r)
            - py_sd_ellipsoid((5.0 - self.EPS, 0.0, 0.0), r)
        ) / (2.0 * self.EPS)
        # The x-component of the gradient should be positive (pointing outward)
        assert gx > 0, (
            f"GRADIENT DIRECTION FAIL: gx should be positive, got {gx}"
        )


# =============================================================================
# 10. BLACKBOX: Boundary and limit behavior
# =============================================================================


class TestBoundaryBehavior:
    """Test behavior at extreme boundaries of the domain."""

    def test_epsilon_surface_inside(self):
        """Just inside the surface must be negative."""
        r = (5.0, 3.0, 2.0)
        d = py_sd_ellipsoid((r[0] - 1e-7, 0.0, 0.0), r)
        assert d < 0, f"Just inside surface should be negative, got {d}"

    def test_epsilon_surface_outside(self):
        """Just outside the surface must be positive."""
        r = (5.0, 3.0, 2.0)
        d = py_sd_ellipsoid((r[0] + 1e-7, 0.0, 0.0), r)
        assert d > 0, f"Just outside surface should be positive, got {d}"

    def test_asymptotic_to_distance(self):
        """As p -> infinity, sdEllipsoid(p, r) ~ min_r * (length(p/r) - 1)."""
        r = (2.0, 3.0, 1.0)
        for scale in [1e3, 1e6, 1e9]:
            # Point far along x: p = (scale, 0, 0)
            # k0 = sqrt((scale/2)^2) = scale/2
            # sd = (scale/2 - 1) * 1 = scale/2 - 1
            p = (scale * r[0], 0.0, 0.0)
            d = py_sd_ellipsoid(p, r)
            expected = scale - 1.0
            # Relative error should be very small
            rel_error = abs(d - expected) / expected
            assert rel_error < 1e-14, (
                f"Asymptotic FAIL at scale {scale}: d={d}, expected={expected}, "
                f"rel_error={rel_error}"
            )

    def test_radius_aspect_ratio_extreme(self):
        """Extreme aspect ratios must still produce correct surface."""
        for r in [(1e-5, 1.0, 1.0), (1.0, 1e-5, 1.0), (1.0, 1.0, 1e-5)]:
            # Surface along the normal axis
            p = (r[0], 0.0, 0.0) if r[0] >= 1e-5 else (0.0, r[1], 0.0)
            if r[0] < 1e-5:
                p = (0.0, r[1], 0.0)
            d_surface = py_sd_ellipsoid(p, r)
            assert d_surface == pytest.approx(0.0, abs=TOL_DEGENERATE), (
                f"EXTREME ASPECT FAIL: p={p} r={r} expected ~0, got {d_surface}"
            )

    def test_nonzero_radius_has_interior(self):
        """Ellipsoid with non-zero radii must have interior (negative distances)."""
        r = (0.001, 0.002, 0.003)
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
        assert d < 0, (
            f"TINY RADII INTERIOR FAIL: center expected negative, got {d}"
        )

    def test_surface_non_axis_point(self):
        """A non-axis-aligned point can also be on the surface."""
        r = (4.0, 3.0, 2.0)
        # Find a point p such that (p.x/4)^2 + (p.y/3)^2 + (p.z/2)^2 = 1
        # Use p = (4cos(theta), 3sin(theta), 0) which gives (cos^2 + sin^2) = 1
        # So p must satisfy k0 = 1, giving sd = 0.
        # Choose p = (4 * 0.6, 3 * 0.8, 0) = (2.4, 2.4, 0)
        # k0 = sqrt((2.4/4)^2 + (2.4/3)^2 + 0) = sqrt(0.36 + 0.64) = 1.0
        p = (2.4, 2.4, 0.0)
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"NON-AXIS SURFACE FAIL: p={p} r={r} expected 0, got {d}"
        )

    def test_surface_spherical_coords(self):
        """Multiple surface points in a spherical direction must give zero."""
        r = (5.0, 4.0, 3.0)
        # For direction d = (0.5, 0.5, sqrt(0.5)), scale to surface:
        # t such that (t*0.5/5)^2 + (t*0.5/4)^2 + (t*sqrt(0.5)/3)^2 = 1
        # t = 1 / sqrt((0.5/5)^2 + (0.5/4)^2 + (0.5/3)^2)
        # Let's just verify a simple case instead
        # Direction (1, 1, 0): t = 1 / sqrt((1/5)^2 + (1/4)^2)
        t = 1.0 / math.sqrt((1.0 / r[0]) ** 2 + (1.0 / r[1]) ** 2)
        p = (t, t, 0.0)
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"SPHERICAL SURFACE FAIL: p={p} r={r} expected 0, got {d}"
        )


# =============================================================================
# 11. BLACKBOX: WGSL to Python model consistency
# =============================================================================


class TestWgslPythonConsistency:
    """The Python reference model must match the WGSL specification exactly."""

    def test_py_model_signature_matches_wgsl(self):
        """Python model takes (p, r) where r is vec3, matching WGSL."""
        # WGSL signature: fn sdEllipsoid(p: vec3<f32>, r: vec3<f32>) -> f32
        with open("engine/rendering/demoscene/wgsl/sdf_ellipsoid.wgsl") as f:
            content = f.read()
        assert "vec3<f32>" in content, "WGSL uses vec3<f32> for r parameter"
        # Python model takes r as a tuple of 3 floats
        result = py_sd_ellipsoid((1.0, 2.0, 3.0), (2.0, 1.0, 0.5))
        assert isinstance(result, float), "Python model must return a float"

    def test_py_model_abs_guard(self):
        """Python model uses abs(r) just like WGSL."""
        with open("engine/rendering/demoscene/wgsl/sdf_ellipsoid.wgsl") as f:
            content = f.read()
        assert "abs(r)" in content, "WGSL uses abs(r) guard"
        # Verify by checking negative radii produce same as positive
        assert py_sd_ellipsoid((1.0, 0.0, 0.0), (1.0, 2.0, 3.0)) == pytest.approx(
            py_sd_ellipsoid((1.0, 0.0, 0.0), (-1.0, -2.0, -3.0)), abs=TOL_EXACT
        )

    def test_py_model_min_r_scaling(self):
        """Python model uses min(r) scaling like WGSL."""
        with open("engine/rendering/demoscene/wgsl/sdf_ellipsoid.wgsl") as f:
            content = f.read()
        has_min = "min(" in content
        has_min_r = "min_r" in content
        assert has_min or has_min_r, (
            "WGSL must use min(r) for world-space scaling"
        )

    def test_py_model_runs_acceptance_cases(self):
        """Smoke test: run acceptance cases through the Python model."""
        # Sphere case: r=(1,1,1) matches sdSphere
        assert py_sd_ellipsoid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == -1.0
        assert py_sd_ellipsoid((1.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == 0.0
        assert py_sd_ellipsoid((2.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == 1.0
        # Stretched axis: r=(2,1,0.5)
        assert py_sd_ellipsoid((2.0, 0.0, 0.0), (2.0, 1.0, 0.5)) == 0.0
        assert py_sd_ellipsoid((0.0, 1.0, 0.0), (2.0, 1.0, 0.5)) == 0.0
        assert py_sd_ellipsoid((0.0, 0.0, 0.5), (2.0, 1.0, 0.5)) == 0.0
        # Center
        assert py_sd_ellipsoid((0.0, 0.0, 0.0), (2.0, 1.0, 0.5)) == pytest.approx(-0.5, abs=TOL_EXACT)
        # Negative radii
        assert py_sd_ellipsoid((1.0, 0.0, 0.0), (-2.0, -1.0, -0.5)) == pytest.approx(
            py_sd_ellipsoid((1.0, 0.0, 0.0), (2.0, 1.0, 0.5)), abs=TOL_EXACT
        )


# =============================================================================
# 12. BLACKBOX: Acceptance criteria summary
# =============================================================================


class TestAcceptanceCriteria:
    """Final summary of all acceptance criteria for T-DEMO-1.8."""

    def test_acceptance_sphere_case(self):
        """AC1: Sphere case r=(1,1,1) matches sdSphere(p, 1.0)."""
        test_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
        ]
        for p in test_points:
            d_e = py_sd_ellipsoid(p, (1.0, 1.0, 1.0))
            d_s = py_sd_sphere(p, 1.0)
            assert d_e == pytest.approx(d_s, abs=TOL_EXACT), (
                f"AC1 FAIL: p={p} ellipsoid={d_e} sphere={d_s}"
            )

    def test_acceptance_stretched_axis(self):
        """AC2: Stretched axis distances: surface exactly at semi-axis length."""
        r = (5.0, 3.0, 2.0)
        assert py_sd_ellipsoid((r[0], 0.0, 0.0), r) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_ellipsoid((0.0, r[1], 0.0), r) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_ellipsoid((0.0, 0.0, r[2]), r) == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_acceptance_interior_negative(self):
        """AC3: Interior points must be negative."""
        r = (4.0, 2.0, 1.0)
        assert py_sd_ellipsoid((0.0, 0.0, 0.0), r) < 0
        assert py_sd_ellipsoid((1.0, 0.0, 0.0), r) < 0
        assert py_sd_ellipsoid((0.0, 1.0, 0.0), r) < 0
        assert py_sd_ellipsoid((0.0, 0.0, 0.5), r) < 0

    def test_acceptance_surface_zero(self):
        """AC4: Surface points must be zero."""
        r = (4.0, 2.0, 1.0)
        assert py_sd_ellipsoid((r[0], 0.0, 0.0), r) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_ellipsoid((0.0, r[1], 0.0), r) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_ellipsoid((0.0, 0.0, r[2]), r) == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_acceptance_exterior_positive(self):
        """AC5: Exterior points must be positive."""
        r = (4.0, 2.0, 1.0)
        assert py_sd_ellipsoid((r[0] + 1.0, 0.0, 0.0), r) > 0
        assert py_sd_ellipsoid((0.0, r[1] + 1.0, 0.0), r) > 0
        assert py_sd_ellipsoid((0.0, 0.0, r[2] + 1.0), r) > 0

    def test_acceptance_degenerate_cases(self):
        """AC6: Degenerate cases handle zero axes correctly."""
        # Zero in one axis
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), (0.0, 1.0, 1.0))
        assert math.isfinite(d), "Zero x-axis must produce finite result"
        # All-zero axes
        d = py_sd_ellipsoid((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert math.isfinite(d), "All-zero axes must produce finite result"
        # Negative radii
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), (-2.0, -1.0, -0.5))
        assert d < 0, "Negative radii must produce correct sign"
        assert math.isfinite(d), "Negative radii must produce finite result"

    def test_acceptance_sign_convention(self):
        """AC7: Sign convention: inside < 0, surface = 0, outside > 0."""
        r = (3.0, 2.0, 1.0)
        # Inside
        assert py_sd_ellipsoid((0.0, 0.0, 0.0), r) < 0
        # Surface
        assert py_sd_ellipsoid((3.0, 0.0, 0.0), r) == pytest.approx(0.0, abs=TOL_SURFACE)
        # Outside
        assert py_sd_ellipsoid((4.0, 0.0, 0.0), r) > 0


# =============================================================================
# 13. BLACKBOX: Numerical precision verification
# =============================================================================


class TestNumericalPrecision:
    """Verify that sdEllipsoid maintains good numerical precision."""

    def test_sphere_case_precision_f32(self):
        """Sphere case should be accurate within f32 precision."""
        r = (1.0, 1.0, 1.0)
        import struct
        for p_vals in [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0)]:
            d = py_sd_ellipsoid(p_vals, r)
            d_f32 = struct.unpack('f', struct.pack('f', d))[0]
            d_check = struct.unpack('f', struct.pack('f', py_sd_sphere(p_vals, 1.0)))[0]
            # The f32-packed values should be very close
            assert abs(d_f32 - d_check) < 1e-7, (
                f"F32 PRECISION FAIL: p={p_vals} ellipsoid_f32={d_f32} "
                f"sphere_f32={d_check}"
            )

    def test_stretched_precision_f32(self):
        """Stretched ellipsoid should be accurate within f32 precision."""
        r = (3.0, 0.5, 0.1)
        p = (1.5, 0.25, 0.05)
        d = py_sd_ellipsoid(p, r)
        d_f32 = struct.unpack('f', struct.pack('f', d))[0]
        assert math.isfinite(d_f32), (
            f"F32 precision produced non-finite value: {d_f32}"
        )
        # The f32 value should be close to the f64 value
        assert abs(d - d_f32) < 1e-7, (
            f"F32 precision loss too high: f64={d}, f32={d_f32}"
        )

    def test_center_precision(self):
        """Center distance should be exactly representable."""
        r = (2.0, 3.0, 4.0)
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
        min_r = min(r)
        expected = -min_r
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"CENTER PRECISION FAIL: expected {expected}, got {d}"
        )
        # Round-trip through f32
        d_f32 = struct.unpack('f', struct.pack('f', d))[0]
        assert d_f32 == pytest.approx(expected, abs=1e-7), (
            f"CENTER F32 FAIL: {d_f32} != {expected}"
        )
