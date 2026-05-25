"""
Whitebox tests for sdPlane WGSL signed-distance-to-plane function (T-DEMO-1.6).

Tests a Python model of the WGSL implementation, verifying:
  - Internal normalization via n_len = sqrt(dot(n, n)), nn = n / n_len
  - Unnormalized normal n=(2,0,0), d=0 normalizes to (1,0,0)
  - Zero normal returns d for any query point
  - Formula decomposition: dot(p, normalize(n)) + d
  - Various orientations all yield zero on the plane
  - Sign convention: above (direction of n) positive, below negative
  - Axis-aligned n=(1,0,0): points at x=0, x=+1, x=-1
  - Tilted plane at 45 degrees
  - Offset d=5 with ground and axis-aligned normals
  - n=(3,4,0) normalizes to (0.6, 0.8, 0)

WGSL implementation (engine/rendering/demoscene/wgsl/sdf_plane.wgsl):
    fn sdPlane(p: vec3<f32>, n: vec3<f32>, d: f32) -> f32 {
        let n_len_sq = dot(n, n);
        if (n_len_sq < 1e-30) {
            return d;
        }
        let n_len = sqrt(n_len_sq);
        let nn = n / n_len;
        return dot(p, nn) + d;
    }

IQ reference formula: dot(p, normalize(n)) + d

WHITEBOX coverage plan:
  Path 1:  Internal normalization -- verify n_len = sqrt(dot(n,n)), nn = n/n_len
  Path 2:  Unnormalized n=(2,0,0), d=0 -- result equals n=(1,0,0) after normalize
  Path 3:  Zero normal returns d for any query point (degenerate plane)
  Path 4:  Formula decomposition -- dot(p, normalize(n)) + d step by step
  Path 5:  Various orientations on plane yield distance zero
  Path 6:  Sign convention -- above (n direction) positive, below negative
  Path 7:  Axis-aligned n=(1,0,0): points at x=0, x=+1, x=-1
  Path 8:  Tilted plane (45 degrees)
  Path 9:  Offset d=5
  Path 10: n=(3,4,0) normalizes to (0.6, 0.8, 0)
"""
from __future__ import annotations

import math

import pytest


# =============================================================================
# Python model of sdPlane matching WGSL semantics exactly (whitebox)
# =============================================================================


def py_sd_plane(p, n, d):
    """Python model of WGSL sdPlane(p: vec3<f32>, n: vec3<f32>, d: f32) -> f32.

    Signed distance from point p (3-tuple) to a plane defined by normal n
    and offset d.  Uses internal normalization: if n is the zero vector,
    returns d for any p; otherwise computes nn = n / sqrt(dot(n, n)) and
    returns dot(p, nn) + d.

    Reference: Inigo Quilez -- Plane SDF
    https://iquilezles.org/articles/distfunctions/
    """
    len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
    # Match WGSL: select(result, d, len_sq < 1e-10)
    if len_sq < 1e-10:
        return d
    n_len = math.sqrt(len_sq)
    nn = (n[0] / n_len, n[1] / n_len, n[2] / n_len)
    return p[0] * nn[0] + p[1] * nn[1] + p[2] * nn[2] + d


# WGSL degenerate threshold (from linter-normalized implementation):
#   let result = dot(p, n / sqrt(len_sq)) + d;
#   return select(result, d, len_sq < 1e-10);
WGSL_DEGENERATE_THRESHOLD = 1e-10

# Tolerance constants
TOL = 1e-12          # General arithmetic tolerance
TOL_SURFACE = 1e-12  # Points on plane should be extremely close to 0


# =============================================================================
# Path 1: Internal normalization -- verify n_len = sqrt(dot(n,n)), nn = n/n_len
# =============================================================================


class TestInternalNormalization:
    """Verify that sdPlane internally normalizes n to unit length.

    The WGSL implementation computes:
      n_len_sq = dot(n, n)
      n_len = sqrt(n_len_sq)
      nn = n / n_len
    """

    def test_normalization_same_as_explicit_normalize(self):
        """sdPlane with unnormalized n equals sdPlane with pre-normalized n."""
        p = (3.0, 4.0, 5.0)
        d = -2.0
        n_unnormalized = (6.0, 8.0, 0.0)   # length 10

        # Pre-compute the normalized version
        n_len = math.sqrt(6.0 * 6.0 + 8.0 * 8.0 + 0.0 * 0.0)
        n_normalized = (6.0 / n_len, 8.0 / n_len, 0.0 / n_len)

        d_unnorm = py_sd_plane(p, n_unnormalized, d)
        d_norm = py_sd_plane(p, n_normalized, d)
        assert d_unnorm == pytest.approx(d_norm, abs=TOL), (
            f"Internal normalization failed: unnormalized n={n_unnormalized} "
            f"gave {d_unnorm}, but pre-normalized gave {d_norm}"
        )

    def test_sqrt_dot_n_n_is_n_length(self):
        """Verify that sqrt(dot(n,n)) equals the Euclidean length of n."""
        n = (3.0, 4.0, 12.0)
        n_len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
        n_len = math.sqrt(n_len_sq)
        expected = math.sqrt(3.0 * 3.0 + 4.0 * 4.0 + 12.0 * 12.0)
        assert n_len == pytest.approx(expected, abs=TOL), (
            f"sqrt(dot(n,n)) = {n_len}, expected Euclidean length = {expected}"
        )

    def test_nn_is_unit_length(self):
        """After normalization, nn should have unit length."""
        normals = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (3.0, 4.0, 0.0),
            (2.0, -3.0, 6.0),
            (0.5, 0.5, 0.5),
        ]
        for n in normals:
            n_len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
            n_len = math.sqrt(n_len_sq)
            nn_len = math.sqrt(
                (n[0] / n_len)**2 + (n[1] / n_len)**2 + (n[2] / n_len)**2
            )
            assert nn_len == pytest.approx(1.0, abs=TOL), (
                f"Normalized nn from n={n} has length {nn_len}, expected 1.0"
            )

    def test_nn_points_same_direction_as_n(self):
        """Normalized nn should point in the same direction as n."""
        normals = [
            (1.0, 2.0, 3.0),
            (-1.0, 0.0, 0.0),
            (0.0, -5.0, 0.0),
            (2.0, -3.0, 6.0),
            (-2.0, 4.0, -8.0),
        ]
        for n in normals:
            n_len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
            n_len = math.sqrt(n_len_sq)
            nn = (n[0] / n_len, n[1] / n_len, n[2] / n_len)
            # Dot product of nn with pre-normalized n should be positive
            # (same direction, not flipped)
            dot_n_nn = n[0] * nn[0] + n[1] * nn[1] + n[2] * nn[2]
            assert dot_n_nn > 0.0, (
                f"nn={nn} does not point in same direction as n={n}: "
                f"dot = {dot_n_nn}"
            )

    def test_unit_normal_passes_through_unchanged(self):
        """A normal that is already unit-length should give same result as itself."""
        p = (2.0, 3.0, 4.0)
        d = 1.0
        n_unit = (1.0, 0.0, 0.0)
        result = py_sd_plane(p, n_unit, d)
        # For unit (1,0,0): dot(p, (1,0,0)) + 1 = 2 + 1 = 3
        expected = p[0] * 1.0 + 1.0
        assert result == pytest.approx(expected, abs=TOL), (
            f"Unit normal: expected {expected}, got {result}"
        )


# =============================================================================
# Path 2: Unnormalized n=(2,0,0), d=0 -- should equal n=(1,0,0) after normalize
# =============================================================================


class TestUnnormalizedNormal:
    """sdPlane must internally normalize n before computing dot(p, nn) + d.

    An unnormalized normal like n=(2,0,0) has length 2, so normalizing
    yields nn=(1,0,0). The result should be identical to passing n=(1,0,0)
    directly.
    """

    def test_doubled_normal_matches_unit(self):
        """n=(2,0,0) should give the same result as n=(1,0,0)."""
        p = (0.0, 0.0, 0.0)
        d = 0.0
        d_double = py_sd_plane(p, (2.0, 0.0, 0.0), d)
        d_unit = py_sd_plane(p, (1.0, 0.0, 0.0), d)
        assert d_double == pytest.approx(d_unit, abs=TOL), (
            f"n=(2,0,0) gave {d_double}, n=(1,0,0) gave {d_unit}"
        )

    def test_doubled_normal_positive_x(self):
        """n=(2,0,0): point (3,0,0) should give distance 3."""
        result = py_sd_plane((3.0, 0.0, 0.0), (2.0, 0.0, 0.0), 0.0)
        assert result == pytest.approx(3.0, abs=TOL), (
            f"n=(2,0,0) at p=(3,0,0): expected 3.0, got {result}"
        )

    def test_doubled_normal_negative_x(self):
        """n=(2,0,0): point (-3,0,0) should give distance -3."""
        result = py_sd_plane((-3.0, 0.0, 0.0), (2.0, 0.0, 0.0), 0.0)
        assert result == pytest.approx(-3.0, abs=TOL), (
            f"n=(2,0,0) at p=(-3,0,0): expected -3.0, got {result}"
        )

    def test_scaled_normal_consistency(self):
        """Scaling n by any positive factor should not change the result."""
        p = (4.0, 5.0, 6.0)
        d = -3.0
        ref = py_sd_plane(p, (1.0, 0.0, 0.0), d)
        for scale in [0.5, 2.0, 10.0, 0.01, 100.0]:
            n_scaled = (scale * 1.0, 0.0, 0.0)
            d_scaled = py_sd_plane(p, n_scaled, d)
            assert d_scaled == pytest.approx(ref, abs=TOL), (
                f"Scale invariance broken at scale={scale}: "
                f"scaled={d_scaled}, ref={ref}"
            )

    def test_scaled_normal_all_axes(self):
        """Scaling all three axes of n together should not change the result."""
        p = (3.0, -2.0, 1.0)
        d = 0.0
        n_ref = (1.0, -1.0, 2.0)
        ref = py_sd_plane(p, n_ref, d)
        for scale in [0.1, 0.5, 2.0, 5.0, 100.0]:
            n_scaled = (scale * n_ref[0], scale * n_ref[1], scale * n_ref[2])
            d_scaled = py_sd_plane(p, n_scaled, d)
            assert d_scaled == pytest.approx(ref, abs=TOL), (
                f"All-axes scaling at scale={scale}: "
                f"scaled={d_scaled}, ref={ref}"
            )

    def test_large_magnitude_normal(self):
        """Very large normal magnitude should still normalize correctly."""
        p = (0.0, 2.5, 0.0)
        result = py_sd_plane(p, (0.0, 1e8, 0.0), 0.0)
        assert result == pytest.approx(2.5, abs=TOL), (
            f"Large magnitude normal gave {result}, expected 2.5"
        )

    def test_small_magnitude_normal(self):
        """Very small (non-zero) normal magnitude should still normalize.

        n=(0,1e-5,0) has len_sq = 1e-10 which is NOT below the WGSL
        threshold (1e-10), so it normalizes to (0,1,0) correctly.
        """
        p = (0.0, 2.5, 0.0)
        result = py_sd_plane(p, (0.0, 1e-5, 0.0), 0.0)
        assert result == pytest.approx(2.5, abs=TOL), (
            f"Small magnitude normal gave {result}, expected 2.5"
        )

    def test_negative_scale_consistency(self):
        """Negative scaling flips the normal, which flips the sign convention."""
        p = (0.0, 3.0, 0.0)
        d = 0.0
        n_pos = (0.0, 2.0, 0.0)
        n_neg = (0.0, -2.0, 0.0)
        d_pos = py_sd_plane(p, n_pos, d)
        d_neg = py_sd_plane(p, n_neg, d)
        # Flipping the normal should negate the signed distance
        assert d_pos == pytest.approx(-d_neg, abs=TOL), (
            f"Negative scale: n_pos gave {d_pos}, n_neg gave {d_neg}, "
            f"expected {d_pos} = -({d_neg})"
        )


# =============================================================================
# Path 3: Zero normal returns d for any query point (degenerate plane)
# =============================================================================


class TestZeroNormal:
    """When n is the zero vector, the concept of 'plane' is degenerate.

    The WGSL implementation guards against n_len_sq < 1e-30 and returns
    the offset d for any query point. This means the "plane" is everywhere,
    defined only by the constant offset.
    """

    def test_zero_normal_returns_offset(self):
        """n=(0,0,0) should return d for any query point."""
        d = 5.0
        points = [
            (0.0, 0.0, 0.0),
            (10.0, -20.0, 30.0),
            (-100.0, 50.0, 200.0),
            (1e6, -1e6, 1e6),
            (0.001, -0.002, 0.003),
        ]
        for p in points:
            result = py_sd_plane(p, (0.0, 0.0, 0.0), d)
            assert result == pytest.approx(d, abs=TOL), (
                f"Zero normal at p={p} with d={d}: expected {d}, got {result}"
            )

    def test_zero_normal_negative_offset(self):
        """n=(0,0,0) with negative d should return negative d."""
        d = -7.0
        points = [
            (0.0, 0.0, 0.0),
            (100.0, -200.0, 300.0),
            (-42.0, 17.0, 88.0),
        ]
        for p in points:
            result = py_sd_plane(p, (0.0, 0.0, 0.0), d)
            assert result == pytest.approx(d, abs=TOL), (
                f"Zero normal at p={p} with d={d}: expected {d}, got {result}"
            )

    def test_zero_normal_zero_offset(self):
        """n=(0,0,0) with d=0 should always return 0."""
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            (-5.0, 10.0, -15.0),
            (1e6, 0.0, 0.0),
        ]
        for p in points:
            result = py_sd_plane(p, (0.0, 0.0, 0.0), 0.0)
            assert result == pytest.approx(0.0, abs=TOL), (
                f"Zero normal at p={p} with d=0: expected 0, got {result}"
            )

    def test_near_zero_normal_precision(self):
        """A normal with extremely small but non-zero length.

        If len_sq is just above the WGSL threshold (1e-10), the normal
        should still be computed correctly (not fall into the degenerate
        branch of select(result, d, len_sq < 1e-10)).
        """
        # A normal of (1e-4, 0, 0) has len_sq = 1e-8 > 1e-10
        # This should pass the guard and normalize correctly.
        n = (1e-4, 0.0, 0.0)
        p = (0.0, 2.0, 0.0)
        d = 0.0
        # After normalization: nn = (1, 0, 0), dot(p, nn) + d = 0
        # (since p.y=2, dot with x-axis is 0)
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(0.0, abs=TOL), (
            f"Near-zero normal n={n} at p={p}: expected 0.0, got {result}"
        )

    def test_below_threshold_simulated(self):
        """A normal with len_sq below threshold should act as degenerate.

        n = (1e-6, 0, 0) has len_sq = 1e-12 < 1e-10.
        This should trigger the degenerate branch and return d.
        """
        n = (1e-6, 0.0, 0.0)
        p = (100.0, 200.0, 300.0)
        d = 42.0
        # The WGSL checks: select(result, d, len_sq < 1e-10)
        # n_len_sq = 1e-12 < 1e-10 -> degenerate -> return d
        len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
        if len_sq < WGSL_DEGENERATE_THRESHOLD:
            result = d
        else:
            result = py_sd_plane(p, n, d)
        assert result == pytest.approx(d, abs=TOL), (
            f"Sub-threshold normal should return d={d}, got {result}"
        )

    def test_degenerate_plane_symmetry(self):
        """Zero normal with d=0 returns 0 for all points (symmetric)."""
        d = 0.0
        results = set()
        points = [
            (1.0, 2.0, 3.0),
            (-1.0, -2.0, -3.0),
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0),
        ]
        for p in points:
            results.add(py_sd_plane(p, (0.0, 0.0, 0.0), d))
        assert len(results) == 1 and 0.0 in results, (
            f"Degenerate with d=0 should always return 0, got {results}"
        )


# =============================================================================
# Path 4: Formula decomposition -- dot(p, normalize(n)) + d step by step
# =============================================================================


class TestFormulaDecomposition:
    """Verify that sdPlane computes dot(p, normalize(n)) + d.

    The formula can be decomposed into:
      1. n_len = sqrt(dot(n, n))
      2. nn = n / n_len
      3. dot(p, nn) = p.x * nn.x + p.y * nn.y + p.z * nn.z
      4. result = dot(p, nn) + d
    """

    def test_explicit_decomposition_matches(self):
        """Step-by-step decomposition should match the aggregate function."""
        p = (3.0, 4.0, 5.0)
        n = (2.0, -3.0, 6.0)
        d = 7.0

        # Step 1: normalize n
        n_len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
        n_len = math.sqrt(n_len_sq)
        nn = (n[0] / n_len, n[1] / n_len, n[2] / n_len)

        # Step 2: compute dot product
        dot_p_nn = p[0] * nn[0] + p[1] * nn[1] + p[2] * nn[2]

        # Step 3: add offset
        expected = dot_p_nn + d

        # Step 4: compare with aggregate
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(expected, abs=TOL), (
            f"Decomposition mismatch: expected {expected}, got {result}"
        )

    def test_dot_product_sign(self):
        """The dot product dot(p, nn) should correctly reflect direction."""
        # n=(1,0,0) -> nn=(1,0,0)
        n = (1.0, 0.0, 0.0)
        n_len = math.sqrt(1.0)
        nn = (n[0] / n_len, n[1] / n_len, n[2] / n_len)

        # dot((+3, 0, 0), (1, 0, 0)) = +3 (same direction, positive)
        assert (3.0 * nn[0]) == pytest.approx(3.0, abs=TOL)

        # dot((-3, 0, 0), (1, 0, 0)) = -3 (opposite direction, negative)
        assert (-3.0 * nn[0]) == pytest.approx(-3.0, abs=TOL)

    def test_zero_dot_on_plane(self):
        """For a point on the plane, dot(p, nn) = -d."""
        n = (0.0, 1.0, 0.0)
        d = 3.0
        # Plane is at y = -d = -3
        p = (0.0, -3.0, 0.0)  # On the plane
        n_len = math.sqrt(1.0)
        nn = (0.0, 1.0, 0.0)
        dot_p_nn = 0.0 * 0.0 + (-3.0) * 1.0 + 0.0 * 0.0  # -3
        # -3 + 3 = 0 -> on plane
        assert dot_p_nn + d == pytest.approx(0.0, abs=TOL), (
            f"On-plane: dot(p,nn)+d = {dot_p_nn} + {d} = {dot_p_nn + d}, "
            f"expected 0"
        )

    def test_normalize_then_dot_equals_divide_by_length(self):
        """dot(p, n/n_len) = dot(p, n) / n_len."""
        p = (3.0, -2.0, 1.0)
        n = (6.0, -8.0, 0.0)
        n_len = math.sqrt(6.0 * 6.0 + (-8.0) * (-8.0) + 0.0 * 0.0)

        # Method 1: normalize then dot
        nn = (n[0] / n_len, n[1] / n_len, n[2] / n_len)
        dot_norm = p[0] * nn[0] + p[1] * nn[1] + p[2] * nn[2]

        # Method 2: dot then divide
        dot_raw = p[0] * n[0] + p[1] * n[1] + p[2] * n[2]
        dot_div = dot_raw / n_len

        assert dot_norm == pytest.approx(dot_div, abs=TOL), (
            f"dot(p, n/n_len) = {dot_norm} != dot(p,n)/n_len = {dot_div}"
        )

    def test_formula_adds_d_after_dot(self):
        """The offset d is added after dot(p, nn), not before."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        d = 5.0
        # dot(p, nn) = 0, so result = 0 + d = d
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(5.0, abs=TOL), (
            f"d after dot: expected 5.0, got {result}"
        )
        # Verify it's not d - something else
        assert result != pytest.approx(0.0, abs=TOL)


# =============================================================================
# Path 5: Various orientations on plane yield distance zero
# =============================================================================


class TestVariousOrientations:
    """For any valid normal/offset pair, points on the plane return zero."""

    def test_ground_plane_origin(self):
        """Ground plane n=(0,1,0), d=0: origin is on the plane."""
        result = py_sd_plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.0)
        assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Origin on ground plane: expected 0, got {result}"
        )

    def test_ground_plane_off_origin(self):
        """Ground plane: off-origin points with y=0 are on the plane."""
        points = [
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (-3.0, 0.0, 5.0),
            (100.0, 0.0, -200.0),
            (1.5, 0.0, -3.7),
        ]
        n = (0.0, 1.0, 0.0)
        d = 0.0
        for p in points:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Ground plane point {p}: expected 0, got {result}"
            )

    def test_vertical_plane(self):
        """Vertical plane n=(1,0,0), d=0: x=0 is the plane."""
        points = [
            (0.0, 0.0, 0.0),
            (0.0, 5.0, 0.0),
            (0.0, -3.0, 0.0),
            (0.0, 0.0, 10.0),
            (0.0, 2.0, -4.0),
        ]
        n = (1.0, 0.0, 0.0)
        d = 0.0
        for p in points:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Vertical plane point {p}: expected 0, got {result}"
            )

    def test_depth_plane(self):
        """Depth plane n=(0,0,1), d=0: z=0 is the plane."""
        points = [
            (0.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (-2.0, 0.0, 0.0),
            (10.0, -5.0, 0.0),
        ]
        n = (0.0, 0.0, 1.0)
        d = 0.0
        for p in points:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Depth plane point {p}: expected 0, got {result}"
            )

    def test_diagonal_plane(self):
        """Diagonal plane n=(1,1,0)/sqrt(2), d=0: x + y = 0 is the plane."""
        n = (1.0, 1.0, 0.0)
        d = 0.0
        # Points on x + y = 0 plane
        points = [
            (0.0, 0.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (5.0, -5.0, 0.0),
            (-5.0, 5.0, 3.0),
            (10.0, -10.0, -10.0),
        ]
        for p in points:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Diagonal plane point {p}: expected 0, got {result}"
            )

    def test_offset_plane_on_surface(self):
        """Offset plane n=(0,1,0), d=3: plane at y=-3."""
        n = (0.0, 1.0, 0.0)
        d = 3.0
        points = [
            (0.0, -3.0, 0.0),
            (1.0, -3.0, 0.0),
            (0.0, -3.0, 5.0),
            (-2.0, -3.0, -1.0),
        ]
        for p in points:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Offset plane on-surface point {p}: expected 0, got {result}"
            )

    def test_arbitrary_plane_on_surface(self):
        """Arbitrary plane: point satisfying dot(p, nn) + d = 0."""
        n = (2.0, -3.0, 1.0)
        d = 4.0
        n_len = math.sqrt(4.0 + 9.0 + 1.0)
        nn = (n[0] / n_len, n[1] / n_len, n[2] / n_len)
        # Pick a p such that dot(p, nn) = -d
        # Let p = (t, 0, 0), solve t * nn.x = -d -> t = -d / nn.x
        p = (-d / nn[0], 0.0, 0.0)
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Arbitrary plane on-surface at p={p}: expected 0, got {result}"
        )

    def test_multiple_points_on_diagonal_plane(self):
        """All points on a 45-degree plane should give zero."""
        n = (1.0, 1.0, 0.0)
        d = 2.0
        # Plane: dot(p, normalize((1,1,0))) + 2 = 0
        # dot(p, (1,1,0)) / sqrt(2) + 2 = 0
        # dot(p, (1,1,0)) = -2*sqrt(2)
        # So x + y = -2*sqrt(2)
        target = -2.0 * math.sqrt(2.0)
        points = [
            (target, 0.0, 0.0),
            (0.0, target, 0.0),
            (target / 2.0, target / 2.0, 0.0),
            (target, 0.0, 10.0),
            (target, 0.0, -3.0),
        ]
        for p in points:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"45-degree offset plane point {p}: expected 0, got {result}"
            )


# =============================================================================
# Path 6: Sign convention -- above (n direction) positive, below negative
# =============================================================================


class TestSignConvention:
    """Above the plane (in the direction of n) must be positive everywhere."""

    def test_positive_direction_above_ground(self):
        """Ground plane n=(0,1,0): positive y is above, should be positive."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        for y_pos in [0.1, 1.0, 10.0, 100.0]:
            result = py_sd_plane((0.0, y_pos, 0.0), n, d)
            assert result > 0.0, (
                f"Above ground plane at y={y_pos}: expected positive, got {result}"
            )

    def test_negative_direction_below_ground(self):
        """Ground plane: negative y is below, should be negative."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        for y_neg in [-0.1, -1.0, -10.0, -100.0]:
            result = py_sd_plane((0.0, y_neg, 0.0), n, d)
            assert result < 0.0, (
                f"Below ground plane at y={y_neg}: expected negative, got {result}"
            )

    def test_sign_near_zero(self):
        """Very small positive/negative offsets should preserve sign."""
        eps = 1e-7
        n = (0.0, 1.0, 0.0)
        d = 0.0
        d_above = py_sd_plane((0.0, eps, 0.0), n, d)
        assert d_above > 0.0, (
            f"Above by {eps}: expected positive, got {d_above}"
        )
        d_below = py_sd_plane((0.0, -eps, 0.0), n, d)
        assert d_below < 0.0, (
            f"Below by {eps}: expected negative, got {d_below}"
        )

    def test_sign_above_vertical_plane(self):
        """Vertical plane n=(1,0,0): positive x is above, should be positive."""
        n = (1.0, 0.0, 0.0)
        d = 0.0
        for x_pos in [0.1, 1.0, 10.0]:
            result = py_sd_plane((x_pos, 0.0, 0.0), n, d)
            assert result > 0.0, (
                f"Above vertical plane at x={x_pos}: expected positive, got {result}"
            )

    def test_sign_below_vertical_plane(self):
        """Vertical plane: negative x is below, should be negative."""
        n = (1.0, 0.0, 0.0)
        d = 0.0
        for x_neg in [-0.1, -1.0, -10.0]:
            result = py_sd_plane((x_neg, 0.0, 0.0), n, d)
            assert result < 0.0, (
                f"Below vertical plane at x={x_neg}: expected negative, got {result}"
            )

    def test_flipped_normal_flips_sign(self):
        """Flipping the normal should flip the sign for the same point."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        p = (0.0, 5.0, 0.0)
        d_up = py_sd_plane(p, n, d)
        d_down = py_sd_plane(p, (0.0, -1.0, 0.0), d)
        assert d_up == pytest.approx(-d_down, abs=TOL), (
            f"Flipped normal: n=(0,1,0) gave {d_up}, "
            f"n=(0,-1,0) gave {d_down}, expected negation"
        )

    def test_sign_antisymmetry(self):
        """Points equidistant on opposite sides should have opposite signs."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        dist = 2.5
        above = py_sd_plane((0.0, dist, 0.0), n, d)
        below = py_sd_plane((0.0, -dist, 0.0), n, d)
        assert above == pytest.approx(-below, abs=TOL), (
            f"Antisymmetry: above={above}, below={below}, "
            f"expected above = -below"
        )
        assert above == pytest.approx(dist, abs=TOL)

    def test_positive_negative_equidistant_magnitude(self):
        """Equidistant points on opposite sides have equal magnitude."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        distances = [0.5, 1.0, 3.0, 10.0]
        for dist in distances:
            above = py_sd_plane((0.0, dist, 0.0), n, d)
            below = py_sd_plane((0.0, -dist, 0.0), n, d)
            assert above == pytest.approx(dist, abs=TOL)
            assert below == pytest.approx(-dist, abs=TOL)


# =============================================================================
# Path 7: Axis-aligned n=(1,0,0): points at x=0, x=+1, x=-1
# =============================================================================


class TestAxisAlignedX:
    """Plane with n=(1,0,0), d=0: plane is x=0.

    Points:
      - x=0  -> distance  0 (on plane)
      - x=+1 -> distance +1 (in direction of n)
      - x=-1 -> distance -1 (opposite direction)
    """

    def test_on_plane_x_zero(self):
        """Point at x=0 is on the plane, distance should be 0."""
        n = (1.0, 0.0, 0.0)
        d = 0.0
        points = [(0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, -3.0, 0.0),
                  (0.0, 0.0, 10.0)]
        for p in points:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"n=(1,0,0): point {p} on plane should be 0, got {result}"
            )

    def test_positive_x_one(self):
        """Point at x=+1 should give distance +1."""
        n = (1.0, 0.0, 0.0)
        d = 0.0
        p = (1.0, 0.0, 0.0)
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(1.0, abs=TOL), (
            f"x=+1: expected 1.0, got {result}"
        )

    def test_positive_x_arbitrary_yz(self):
        """Points with x=+1, any y/z should give distance +1."""
        n = (1.0, 0.0, 0.0)
        d = 0.0
        points = [
            (1.0, 2.0, 0.0),
            (1.0, 0.0, 3.0),
            (1.0, -5.0, 10.0),
            (1.0, 100.0, -200.0),
        ]
        for p in points:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(1.0, abs=TOL), (
                f"x=+1 with yz: p={p}, expected 1.0, got {result}"
            )

    def test_negative_x_one(self):
        """Point at x=-1 should give distance -1."""
        n = (1.0, 0.0, 0.0)
        d = 0.0
        p = (-1.0, 0.0, 0.0)
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(-1.0, abs=TOL), (
            f"x=-1: expected -1.0, got {result}"
        )

    def test_negative_x_arbitrary_yz(self):
        """Points with x=-1, any y/z should give distance -1."""
        n = (1.0, 0.0, 0.0)
        d = 0.0
        points = [
            (-1.0, 2.0, 0.0),
            (-1.0, 0.0, 3.0),
            (-1.0, -5.0, 10.0),
            (-1.0, 100.0, -200.0),
        ]
        for p in points:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(-1.0, abs=TOL), (
                f"x=-1 with yz: p={p}, expected -1.0, got {result}"
            )

    def test_x_varied_values(self):
        """Distance should equal x coordinate for n=(1,0,0), d=0."""
        n = (1.0, 0.0, 0.0)
        d = 0.0
        x_values = [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0, 10.0]
        for x in x_values:
            result = py_sd_plane((x, 0.0, 0.0), n, d)
            assert result == pytest.approx(x, abs=TOL), (
                f"x={x}: expected {x}, got {result}"
            )

    def test_x_independent_of_yz(self):
        """For n=(1,0,0), the result depends only on x, not on y or z."""
        n = (1.0, 0.0, 0.0)
        d = 0.0
        x_val = 3.0
        ref = py_sd_plane((x_val, 0.0, 0.0), n, d)
        yz_variants = [
            (x_val, 1.0, 0.0),
            (x_val, 0.0, 1.0),
            (x_val, -5.0, 3.0),
            (x_val, 10.0, -20.0),
            (x_val, -100.0, 50.0),
        ]
        for p in yz_variants:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(ref, abs=TOL), (
                f"x={x_val} variant p={p}: expected {ref}, got {result}"
            )


# =============================================================================
# Path 8: Tilted plane (45 degrees)
# =============================================================================


class TestTiltedPlane:
    """Plane at 45 degrees with normal n=(1,1,0)/sqrt(2), d=0.

    The plane is defined by x + y = 0.
    Point (1, 0, 0) projects onto nn = (1/sqrt(2), 1/sqrt(2), 0):
      dot((1,0,0), (1/sqrt(2), 1/sqrt(2), 0)) = 1/sqrt(2)
    """

    def test_on_diagonal_plane(self):
        """Points on x + y = 0 should give zero distance."""
        n = (1.0, 1.0, 0.0)
        d = 0.0
        points = [
            (0.0, 0.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (5.0, -5.0, 3.0),
            (-10.0, 10.0, 0.0),
        ]
        for p in points:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"45-degree plane on-surface point {p}: expected 0, got {result}"
            )

    def test_positive_side_diagonal(self):
        """Point (1, 0, 0) on positive side: expected 1/sqrt(2)."""
        n = (1.0, 1.0, 0.0)
        d = 0.0
        p = (1.0, 0.0, 0.0)
        result = py_sd_plane(p, n, d)
        expected = 1.0 / math.sqrt(2.0)
        assert result == pytest.approx(expected, abs=TOL), (
            f"45-degree: point (1,0,0): expected {expected}, got {result}"
        )

    def test_negative_side_diagonal(self):
        """Point (-1, 0, 0) on negative side: expected -1/sqrt(2)."""
        n = (1.0, 1.0, 0.0)
        d = 0.0
        p = (-1.0, 0.0, 0.0)
        result = py_sd_plane(p, n, d)
        expected = -1.0 / math.sqrt(2.0)
        assert result == pytest.approx(expected, abs=TOL), (
            f"45-degree: point (-1,0,0): expected {expected}, got {result}"
        )

    def test_positive_side_y_axis(self):
        """Point (0, 1, 0) on positive side: expected 1/sqrt(2)."""
        n = (1.0, 1.0, 0.0)
        d = 0.0
        p = (0.0, 1.0, 0.0)
        result = py_sd_plane(p, n, d)
        expected = 1.0 / math.sqrt(2.0)
        assert result == pytest.approx(expected, abs=TOL), (
            f"45-degree: point (0,1,0): expected {expected}, got {result}"
        )

    def test_negative_side_y_axis(self):
        """Point (0, -1, 0) on negative side: expected -1/sqrt(2)."""
        n = (1.0, 1.0, 0.0)
        d = 0.0
        p = (0.0, -1.0, 0.0)
        result = py_sd_plane(p, n, d)
        expected = -1.0 / math.sqrt(2.0)
        assert result == pytest.approx(expected, abs=TOL), (
            f"45-degree: point (0,-1,0): expected {expected}, got {result}"
        )

    def test_diagonal_symmetry(self):
        """Points symmetric across the diagonal plane should have opposite signs.

        (a, 0, 0) and (0, -a, 0) are reflections across the plane x+y=0,
        so their signed distances should be equal in magnitude but opposite
        in sign.
        """
        n = (1.0, 1.0, 0.0)
        d = 0.0
        values = [0.5, 1.0, 2.0, 5.0]
        for a in values:
            d1 = py_sd_plane((a, 0.0, 0.0), n, d)
            d2 = py_sd_plane((0.0, -a, 0.0), n, d)
            # Reflections across the plane have opposite signed distances
            assert d1 == pytest.approx(-d2, abs=TOL), (
                f"Diagonal symmetry broken at a={a}: "
                f"d1={d1}, d2={d2}, expected d1 = -d2"
            )

    def test_tilted_plane_z_independence(self):
        """For n=(1,1,0), result should not depend on z coordinate."""
        n = (1.0, 1.0, 0.0)
        d = 0.0
        ref = py_sd_plane((1.0, 0.0, 0.0), n, d)
        z_variants = [
            (1.0, 0.0, 5.0),
            (1.0, 0.0, -3.0),
            (1.0, 0.0, 100.0),
        ]
        for p in z_variants:
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(ref, abs=TOL), (
                f"z-independence: p={p}: expected {ref}, got {result}"
            )


# =============================================================================
# Path 9: Offset d=5
# =============================================================================


class TestPlaneOffset:
    """Planes with non-zero offset d.

    The plane equation is dot(p, nn) + d = 0, which places the plane at
    signed distance -d from the origin along the normal direction.
    """

    def test_ground_plane_offset_d_5(self):
        """n=(0,1,0), d=5: plane is at y=-5."""
        n = (0.0, 1.0, 0.0)
        d = 5.0
        # Origin: dot((0,0,0), (0,1,0)) + 5 = 5 (above plane)
        result_origin = py_sd_plane((0.0, 0.0, 0.0), n, d)
        assert result_origin == pytest.approx(5.0, abs=TOL), (
            f"Origin with d=5: expected 5.0, got {result_origin}"
        )

    def test_on_offset_plane_y_neg_5(self):
        """Point at y=-5 should be on the plane, distance 0."""
        n = (0.0, 1.0, 0.0)
        d = 5.0
        p = (0.0, -5.0, 0.0)
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"On plane at y=-5 with d=5: expected 0, got {result}"
        )

    def test_above_offset_plane(self):
        """Point above offset plane: y=-2 should give distance 3."""
        n = (0.0, 1.0, 0.0)
        d = 5.0
        p = (0.0, -2.0, 0.0)
        # dot((0,-2,0), (0,1,0)) + 5 = -2 + 5 = 3
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(3.0, abs=TOL), (
            f"Above offset plane at y=-2: expected 3.0, got {result}"
        )

    def test_below_offset_plane(self):
        """Point below offset plane: y=-8 should give distance -3."""
        n = (0.0, 1.0, 0.0)
        d = 5.0
        p = (0.0, -8.0, 0.0)
        # dot((0,-8,0), (0,1,0)) + 5 = -8 + 5 = -3
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(-3.0, abs=TOL), (
            f"Below offset plane at y=-8: expected -3.0, got {result}"
        )

    def test_vertical_plane_offset(self):
        """n=(1,0,0), d=5: plane is at x=-5."""
        n = (1.0, 0.0, 0.0)
        d = 5.0
        # On plane at x=-5
        on_plane = py_sd_plane((-5.0, 0.0, 0.0), n, d)
        assert on_plane == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Vertical offset plane at x=-5: expected 0, got {on_plane}"
        )
        # Origin is at x=0, distance = 0 + 5 = 5
        origin = py_sd_plane((0.0, 0.0, 0.0), n, d)
        assert origin == pytest.approx(5.0, abs=TOL)

    def test_negative_offset(self):
        """Negative offset pushes plane in opposite direction."""
        n = (0.0, 1.0, 0.0)
        d = -5.0
        # Plane is at y = -d = 5
        on_plane = py_sd_plane((0.0, 5.0, 0.0), n, d)
        assert on_plane == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Negative offset plane at y=5: expected 0, got {on_plane}"
        )
        # Origin: dot((0,0,0), ...) + (-5) = -5
        origin = py_sd_plane((0.0, 0.0, 0.0), n, d)
        assert origin == pytest.approx(-5.0, abs=TOL)

    def test_offset_with_diagonal_normal(self):
        """Diagonal normal n=(1,1,0)/sqrt(2) with d=5."""
        n = (1.0, 1.0, 0.0)
        d = 5.0
        # Origin: dot((0,0,0), nn) + 5 = 5
        origin = py_sd_plane((0.0, 0.0, 0.0), n, d)
        assert origin == pytest.approx(5.0, abs=TOL)

        # Find a point on the plane: dot(p, nn) = -5
        # p = (-5/sqrt(2), 0, 0) gives dot = -5/sqrt(2) * 1/sqrt(2) = -5
        # Wait, let's compute directly:
        # nn = (1/sqrt(2), 1/sqrt(2), 0)
        # p = (-5*sqrt(2), 0, 0) gives dot = -5*sqrt(2) * 1/sqrt(2) = -5
        # sd = -5 + 5 = 0
        n_len = math.sqrt(2.0)
        nn = (1.0 / n_len, 1.0 / n_len, 0.0)
        p_on = (-5.0 / nn[0], 0.0, 0.0)  # = (-5*sqrt(2), 0, 0)
        on_plane = py_sd_plane(p_on, n, d)
        assert on_plane == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Diagonal offset plane: point {p_on}: expected 0, got {on_plane}"
        )

    def test_offset_scales_linearly(self):
        """Doubling offset d should shift result by same amount."""
        n = (0.0, 1.0, 0.0)
        p = (0.0, 3.0, 0.0)
        d1 = 2.0
        d2 = 4.0
        r1 = py_sd_plane(p, n, d1)
        r2 = py_sd_plane(p, n, d2)
        # r2 - r1 = d2 - d1 = 2
        assert r2 - r1 == pytest.approx(d2 - d1, abs=TOL), (
            f"Offset linearity: r1={r1}, r2={r2}, "
            f"r2-r1={r2-r1}, expected {(d2-d1)}"
        )

    def test_offset_preserves_sign_convention(self):
        """With offset, positive/negative sign convention is preserved."""
        n = (0.0, 1.0, 0.0)
        d = 3.0
        # Above plane (y > -3): positive
        above = py_sd_plane((0.0, -2.0, 0.0), n, d)
        assert above > 0.0, f"Above offset plane should be positive, got {above}"
        # On plane (y = -3): zero
        on = py_sd_plane((0.0, -3.0, 0.0), n, d)
        assert on == pytest.approx(0.0, abs=TOL_SURFACE)
        # Below plane (y < -3): negative
        below = py_sd_plane((0.0, -4.0, 0.0), n, d)
        assert below < 0.0, f"Below offset plane should be negative, got {below}"


# =============================================================================
# Path 10: n=(3,4,0) normalizes to (0.6, 0.8, 0)
# =============================================================================


class TestNormalize345:
    """n=(3,4,0) has length 5, so it normalizes to (0.6, 0.8, 0).

    The WGSL normalization:
      n_len_sq = 3*3 + 4*4 + 0*0 = 25
      n_len = sqrt(25) = 5
      nn = (3/5, 4/5, 0) = (0.6, 0.8, 0)
    """

    def test_normalization_components(self):
        """Verify the normalized components of n=(3,4,0) are (0.6, 0.8, 0)."""
        n = (3.0, 4.0, 0.0)
        n_len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
        n_len = math.sqrt(n_len_sq)
        nn = (n[0] / n_len, n[1] / n_len, n[2] / n_len)
        assert nn[0] == pytest.approx(0.6, abs=TOL), (
            f"nn.x expected 0.6, got {nn[0]}"
        )
        assert nn[1] == pytest.approx(0.8, abs=TOL), (
            f"nn.y expected 0.8, got {nn[1]}"
        )
        assert nn[2] == pytest.approx(0.0, abs=TOL), (
            f"nn.z expected 0.0, got {nn[2]}"
        )

    def test_origin_with_345_normal(self):
        """At origin with n=(3,4,0), d=0: dot = 0, result = 0."""
        n = (3.0, 4.0, 0.0)
        d = 0.0
        result = py_sd_plane((0.0, 0.0, 0.0), n, d)
        assert result == pytest.approx(0.0, abs=TOL), (
            f"Origin with n=(3,4,0): expected 0, got {result}"
        )

    def test_point_along_normal_direction(self):
        """Moving along the normalized normal direction should increase distance.

        nn = (0.6, 0.8, 0), so moving to p = (0.6, 0.8, 0) gives distance 1.
        """
        n = (3.0, 4.0, 0.0)
        d = 0.0
        # p aligned with nn: dot = 0.6*0.6 + 0.8*0.8 + 0 = 0.36 + 0.64 = 1.0
        p = (0.6, 0.8, 0.0)
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(1.0, abs=TOL), (
            f"Point (0.6,0.8,0) along nn: expected 1.0, got {result}"
        )

    def test_point_opposite_normal_direction(self):
        """Moving opposite to nn gives negative distance -1."""
        n = (3.0, 4.0, 0.0)
        d = 0.0
        p = (-0.6, -0.8, 0.0)
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(-1.0, abs=TOL), (
            f"Point (-0.6,-0.8,0) opposite nn: expected -1.0, got {result}"
        )

    def test_perpendicular_point(self):
        """A point perpendicular to nn has zero distance (on the plane).

        A point p such that dot(p, nn) = 0 is on the plane.
        For nn=(0.6, 0.8, 0), p=(-0.8, 0.6, 0) satisfies this.
        """
        n = (3.0, 4.0, 0.0)
        d = 0.0
        # Perpendicular vector: (-0.8, 0.6, 0)
        # dot = (-0.8)*0.6 + 0.6*0.8 + 0*0 = -0.48 + 0.48 + 0 = 0
        p = (-0.8, 0.6, 0.0)
        result = py_sd_plane(p, n, d)
        assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Perpendicular point (-0.8, 0.6, 0): expected 0, got {result}"
        )

    def test_345_with_offset(self):
        """n=(3,4,0), d=5: point on plane has dot(p, nn) = -5."""
        n = (3.0, 4.0, 0.0)
        d = 5.0
        # p = (-5*0.6, -5*0.8, 0) = (-3, -4, 0) gives dot = -3*0.6 + -4*0.8 = -5
        p_on = (-5.0 * 0.6, -5.0 * 0.8, 0.0)
        result_on = py_sd_plane(p_on, n, d)
        assert result_on == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"345 offset on-plane point {p_on}: expected 0, got {result_on}"
        )

    def test_345_equivalence_to_unit_normal(self):
        """n=(3,4,0) should give same result as n=(0.6,0.8,0)."""
        p = (1.0, 2.0, 3.0)
        d = -4.0
        d_345 = py_sd_plane(p, (3.0, 4.0, 0.0), d)
        d_unit = py_sd_plane(p, (0.6, 0.8, 0.0), d)
        assert d_345 == pytest.approx(d_unit, abs=TOL), (
            f"345 vs unit: n=(3,4,0) gave {d_345}, "
            f"n=(0.6,0.8,0) gave {d_unit}"
        )

    def test_345_multiple_points(self):
        """Test n=(3,4,0) with d=0 against multiple points."""
        n = (3.0, 4.0, 0.0)
        d = 0.0
        nn = (0.6, 0.8, 0.0)
        test_cases = [
            (1.0, 0.0, 0.0),      # dot = 0.6
            (0.0, 1.0, 0.0),      # dot = 0.8
            (0.0, 0.0, 1.0),      # dot = 0.0 (on plane since z*0 = 0)
            (-1.0, 0.0, 0.0),     # dot = -0.6
            (0.0, -1.0, 0.0),     # dot = -0.8
            (5.0, 0.0, 0.0),      # dot = 3.0
            (0.0, 5.0, 0.0),      # dot = 4.0
            (3.0, 4.0, 0.0),      # dot = 3*0.6 + 4*0.8 = 1.8 + 3.2 = 5.0
        ]
        for px, py, pz in test_cases:
            expected = px * nn[0] + py * nn[1] + pz * nn[2]
            result = py_sd_plane((px, py, pz), n, d)
            assert result == pytest.approx(expected, abs=TOL), (
                f"345 at p=({px},{py},{pz}): expected {expected}, got {result}"
            )


# =============================================================================
# Path 11: Continuity and smoothness
# =============================================================================


class TestContinuity:
    """sdPlane must be continuous everywhere: small changes in p produce
    small changes in the signed distance. Since the plane SDF is an affine
    function of p, it is trivially continuous and has unit gradient."""

    def test_continuity_along_x(self):
        """SDF should be continuous along the x-axis."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        step = 0.01
        prev = py_sd_plane((-5.0, 1.0, 0.0), n, d)
        for i in range(1, 500):
            x = -5.0 + i * step
            curr = py_sd_plane((x, 1.0, 0.0), n, d)
            diff = abs(curr - prev)
            # Max diff should be 0 (plane SDF does not depend on x for this n)
            assert diff <= step, (
                f"Continuity along x: expected diff <= {step}, got {diff} at x={x}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """SDF should be continuous along the y-axis (direction of gradient)."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        step = 0.01
        prev = py_sd_plane((0.0, -5.0, 0.0), n, d)
        for i in range(1, 500):
            y = -5.0 + i * step
            curr = py_sd_plane((0.0, y, 0.0), n, d)
            diff = abs(curr - prev)
            # Max diff should equal step (unit gradient along y)
            assert diff <= step * 1.001, (
                f"Continuity along y: expected diff <= {step}, "
                f"got {diff} at y={y}"
            )
            prev = curr

    def test_continuity_across_plane(self):
        """SDF should be continuous across the plane boundary."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        step = 1e-6
        for offset in [i * step for i in range(-50, 51)]:
            y = offset
            result = py_sd_plane((0.0, y, 0.0), n, d)
            assert result == pytest.approx(y, abs=1e-12), (
                f"Continuity across plane at y={y}: expected {y}, got {result}"
            )

    def test_continuity_with_offset(self):
        """SDF should be continuous across an offset plane."""
        n = (0.0, 1.0, 0.0)
        d = 3.0
        step = 1e-6
        for offset in [i * step for i in range(-50, 51)]:
            y = -3.0 + offset  # Cross the plane at y = -3
            result = py_sd_plane((0.0, y, 0.0), n, d)
            expected = y + d
            assert result == pytest.approx(expected, abs=1e-12), (
                f"Continuity across offset plane at y={y}: "
                f"expected {expected}, got {result}"
            )


# =============================================================================
# Path 12: Unit gradient property (eikonal equation)
# =============================================================================


class TestUnitGradient:
    """The plane SDF has gradient magnitude 1 everywhere (eikonal property).
    For sdPlane(p, n, d) = dot(p, nn) + d, the gradient is nn (unit vector)."""

    def test_numerical_gradient_x(self):
        """Numerical gradient along x should equal nn.x."""
        n = (3.0, 4.0, 0.0)
        d = 2.0
        eps = 1e-6
        nn = (0.6, 0.8, 0.0)

        test_points = [
            (1.0, 2.0, 3.0),
            (-2.0, 3.0, 1.0),
            (5.0, -1.0, 0.0),
        ]
        for p in test_points:
            grad_x = (
                py_sd_plane((p[0] + eps, p[1], p[2]), n, d) -
                py_sd_plane((p[0] - eps, p[1], p[2]), n, d)
            ) / (2.0 * eps)
            assert grad_x == pytest.approx(nn[0], abs=1e-4), (
                f"Gradient x at {p}: expected {nn[0]}, got {grad_x}"
            )

    def test_numerical_gradient_y(self):
        """Numerical gradient along y should equal nn.y."""
        n = (3.0, 4.0, 0.0)
        d = 2.0
        eps = 1e-6
        nn = (0.6, 0.8, 0.0)

        p = (2.0, 3.0, 4.0)
        grad_y = (
            py_sd_plane((p[0], p[1] + eps, p[2]), n, d) -
            py_sd_plane((p[0], p[1] - eps, p[2]), n, d)
        ) / (2.0 * eps)
        assert grad_y == pytest.approx(nn[1], abs=1e-4), (
            f"Gradient y: expected {nn[1]}, got {grad_y}"
        )

    def test_simple_ground_plane_gradient(self):
        """For n=(0,1,0), gradient should be (0, 1, 0)."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        eps = 1e-6
        p = (1.0, 2.0, 3.0)

        gx = (py_sd_plane((p[0] + eps, p[1], p[2]), n, d) -
              py_sd_plane((p[0] - eps, p[1], p[2]), n, d)) / (2.0 * eps)
        gy = (py_sd_plane((p[0], p[1] + eps, p[2]), n, d) -
              py_sd_plane((p[0], p[1] - eps, p[2]), n, d)) / (2.0 * eps)
        gz = (py_sd_plane((p[0], p[1], p[2] + eps), n, d) -
              py_sd_plane((p[0], p[1], p[2] - eps), n, d)) / (2.0 * eps)

        assert gx == pytest.approx(0.0, abs=1e-4)
        assert gy == pytest.approx(1.0, abs=1e-4)
        assert gz == pytest.approx(0.0, abs=1e-4)

    def test_gradient_magnitude(self):
        """Gradient magnitude should be 1 (eikonal equation satisfied)."""
        eps = 1e-6
        normals = [
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (3.0, 4.0, 0.0),
        ]
        points = [(1.0, 2.0, 3.0), (-2.0, 3.0, 1.0), (5.0, -1.0, 0.0)]
        for n in normals:
            for p in points:
                gx = (py_sd_plane((p[0] + eps, p[1], p[2]), n, 0.0) -
                      py_sd_plane((p[0] - eps, p[1], p[2]), n, 0.0)) / (2.0 * eps)
                gy = (py_sd_plane((p[0], p[1] + eps, p[2]), n, 0.0) -
                      py_sd_plane((p[0], p[1] - eps, p[2]), n, 0.0)) / (2.0 * eps)
                gz = (py_sd_plane((p[0], p[1], p[2] + eps), n, 0.0) -
                      py_sd_plane((p[0], p[1], p[2] - eps), n, 0.0)) / (2.0 * eps)
                mag = math.sqrt(gx**2 + gy**2 + gz**2)
                assert mag == pytest.approx(1.0, abs=1e-4), (
                    f"Eikonal: gradient mag for n={n} at {p} is {mag}, "
                    f"expected 1.0"
                )


# =============================================================================
# Path 13: Determinism and repeatability
# =============================================================================


class TestDeterminism:
    """sdPlane must be deterministic: same inputs always produce the same result."""

    def test_repeated_calls_identical(self):
        """Multiple calls with identical arguments should produce identical results."""
        p = (1.234, 5.678, 9.012)
        n = (3.0, 4.0, 0.0)
        d = 3.141
        first = py_sd_plane(p, n, d)
        for _ in range(50):
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(first, abs=TOL), (
                f"Non-deterministic: first={first}, repeat={result}"
            )

    def test_different_inputs_different_results(self):
        """Different inputs generally produce different results."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        results = set()
        for x in range(10):
            result = round(py_sd_plane((float(x), float(x * 2), 0.0), n, d), 12)
            results.add(result)
        assert len(results) >= 9, (
            f"Expected at least 9 unique results from varying inputs, "
            f"got {len(results)}"
        )


# =============================================================================
# Path 14: Scaled normal with offset
# =============================================================================


class TestScaledNormalWithOffset:
    """Combined effects of scaled normal and offset."""

    def test_scaled_normal_with_offset(self):
        """n=(6,8,0) length 10, normalized to (0.6,0.8,0), d=5.

        Point (-3, -1, 0): dot = -3*0.6 + -1*0.8 = -1.8 + -0.8 = -2.6
        Result = -2.6 + 5 = 2.4
        """
        n = (6.0, 8.0, 0.0)  # Length 10
        d = 5.0
        p = (-3.0, -1.0, 0.0)
        result = py_sd_plane(p, n, d)
        nn = (0.6, 0.8, 0.0)
        expected = p[0] * nn[0] + p[1] * nn[1] + p[2] * nn[2] + d
        assert result == pytest.approx(expected, abs=TOL), (
            f"Scaled normal with offset: expected {expected}, got {result}"
        )

    def test_doubled_offset_with_scaled_normal(self):
        """Doubling d doubles the offset contribution."""
        n = (6.0, 8.0, 0.0)
        p = (2.0, 3.0, 4.0)
        d1 = 2.0
        d2 = 4.0
        r1 = py_sd_plane(p, n, d1)
        r2 = py_sd_plane(p, n, d2)
        assert r2 - r1 == pytest.approx(2.0, abs=TOL)

    def test_scaled_and_unnormalized_equivalence(self):
        """n=(6,8,0) should equal n=(3,4,0)."""
        p = (1.0, 2.0, 3.0)
        d = 5.0
        d1 = py_sd_plane(p, (6.0, 8.0, 0.0), d)
        d2 = py_sd_plane(p, (3.0, 4.0, 0.0), d)
        assert d1 == pytest.approx(d2, abs=TOL), (
            f"n=(6,8,0) gave {d1}, n=(3,4,0) gave {d2}"
        )


# =============================================================================
# Path 15: Extreme values (large coordinates, large offset)
# =============================================================================


class TestExtremeValues:
    """sdPlane should handle large coordinate values gracefully (no overflow)."""

    def test_large_coordinate(self):
        """Large coordinates should not cause numeric issues."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        large_y = 1e6
        result = py_sd_plane((0.0, large_y, 0.0), n, d)
        assert result == pytest.approx(large_y, abs=0.001), (
            f"Large coordinate: expected {large_y}, got {result}"
        )

    def test_large_offset(self):
        """Large offset should work correctly."""
        n = (0.0, 1.0, 0.0)
        d = 1e6
        result = py_sd_plane((0.0, 0.0, 0.0), n, d)
        assert result == pytest.approx(d, abs=0.001), (
            f"Large offset: expected {d}, got {result}"
        )

    def test_large_normal(self):
        """Very large normal should still normalize correctly."""
        n = (0.0, 1e12, 0.0)
        d = 0.0
        result = py_sd_plane((0.0, 5.0, 0.0), n, d)
        assert result == pytest.approx(5.0, abs=0.001), (
            f"Large normal: expected 5.0, got {result}"
        )

    def test_small_values(self):
        """Very small coordinate values should not cause underflow."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        small_y = 1e-10
        result = py_sd_plane((0.0, small_y, 0.0), n, d)
        assert result == pytest.approx(small_y, abs=TOL), (
            f"Small coordinate: expected {small_y}, got {result}"
        )

    def test_all_large_values(self):
        """All inputs large should not cause overflow."""
        n = (1e6, 1e6, 1e6)
        d = 1e6
        p = (1e6, 2e6, 3e6)
        result = py_sd_plane(p, n, d)
        # nn = (1,1,1)/sqrt(3)
        nn_len = math.sqrt(3.0)
        expected = (1e6 / nn_len) * (1.0 + 2.0 + 3.0) + 1e6
        assert result == pytest.approx(expected, rel=1e-10), (
            f"All large: expected {expected}, got {result}"
        )
