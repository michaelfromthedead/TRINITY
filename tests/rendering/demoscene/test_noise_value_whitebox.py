"""
Whitebox tests for value noise WGSL functions (T-DEMO-1.29).

Tests Python model implementations of each WGSL function, verifying:
  - Correctness against mathematical definition
  - Range validation (output in [-1, 1])
  - Determinism (same input = same output)
  - Continuity (smooth interpolation between grid points)
  - Grid alignment (at integer positions, noise = hash value)
  - Fade curve (zero first derivative at integer grid points)
  - 1D/2D/3D sampling consistency

WHITEBOX coverage plan:
  Path A: value_noise_1d on integer grid -> maps to hash11
  Path B: value_noise_1d half-integer -> interpolated
  Path C: value_noise_1d range in [-1, 1]
  Path D: value_noise_1d deterministic
  Path E: value_noise_1d continuous (no discontinuities)
  Path F: value_noise_2d on integer grid -> maps to hash21
  Path G: value_noise_2d range in [-1, 1]
  Path H: value_noise_2d deterministic
  Path I: value_noise_2d bilinear interpolation
  Path J: value_noise_2d continuous
  Path K: value_noise_3d on integer grid -> maps to hash31
  Path L: value_noise_3d range in [-1, 1]
  Path M: value_noise_3d deterministic
  Path N: value_noise_3d trilinear interpolation
  Path O: value_noise_3d continuous
  Path P: Cross-dimension consistency at z=0
  Path Q: Fade curve zero derivative at integer boundaries
"""

import math

import pytest

# =============================================================================
# Python model implementations matching WGSL semantics
# =============================================================================

# WGSL fract: x - floor(x)
def wgsl_fract(x: float) -> float:
    return x - math.floor(x)


def py_hash11(p: float) -> float:
    """Model of WGSL hash11: 1D float -> [0, 1) float."""
    q = p
    q = wgsl_fract(q * 0.1031)
    q = q * (q + 33.33)
    q = q * (q + q)
    return wgsl_fract(q)


def py_hash21(p) -> float:
    """Model of WGSL hash21: 2D -> [0, 1) float.

    WGSL: q = fract(q * vec2<f32>(0.1031, 0.1030));
          q = q + dot(q, q + 33.33);
          return fract(q.x * q.y);
    """
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)

    d = qx * (qx + 33.33) + qy * (qy + 33.33)
    qx = qx + d
    qy = qy + d

    return wgsl_fract(qx * qy)


def py_hash31(p) -> float:
    """Model of WGSL hash31: 3D -> [0, 1) float.

    WGSL: q = fract(q * vec3<f32>(0.1031, 0.1030, 0.0973));
          q = q + dot(q, q + 33.33);
          return fract(q.x * q.y * q.z);
    """
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)
    qz = wgsl_fract(p[2] * 0.0973)

    d = qx * (qx + 33.33) + qy * (qy + 33.33) + qz * (qz + 33.33)
    qx = qx + d
    qy = qy + d
    qz = qz + d

    return wgsl_fract(qx * qy * qz)


def py_smoothstep(t: float) -> float:
    """Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def py_value_noise_1d(p: float) -> float:
    """Model of WGSL value_noise_1d."""
    i = math.floor(p)
    f = p - i

    u = py_smoothstep(f)

    a = py_hash11(i)
    b = py_hash11(i + 1.0)

    va = a * 2.0 - 1.0
    vb = b * 2.0 - 1.0

    return va + u * (vb - va)


def py_value_noise_2d(p) -> float:
    """Model of WGSL value_noise_2d."""
    ix = math.floor(p[0])
    iy = math.floor(p[1])
    fx = p[0] - ix
    fy = p[1] - iy

    ux = py_smoothstep(fx)
    uy = py_smoothstep(fy)

    a = py_hash21((ix, iy))
    b = py_hash21((ix + 1.0, iy))
    c = py_hash21((ix, iy + 1.0))
    d = py_hash21((ix + 1.0, iy + 1.0))

    va = a * 2.0 - 1.0
    vb = b * 2.0 - 1.0
    vc = c * 2.0 - 1.0
    vd = d * 2.0 - 1.0

    vx0 = va + ux * (vb - va)
    vx1 = vc + ux * (vd - vc)
    return vx0 + uy * (vx1 - vx0)


def py_value_noise_3d(p) -> float:
    """Model of WGSL value_noise_3d."""
    ix = math.floor(p[0])
    iy = math.floor(p[1])
    iz = math.floor(p[2])
    fx = p[0] - ix
    fy = p[1] - iy
    fz = p[2] - iz

    ux = py_smoothstep(fx)
    uy = py_smoothstep(fy)
    uz = py_smoothstep(fz)

    a = py_hash31((ix, iy, iz))
    b = py_hash31((ix + 1.0, iy, iz))
    c = py_hash31((ix, iy + 1.0, iz))
    d = py_hash31((ix + 1.0, iy + 1.0, iz))
    e = py_hash31((ix, iy, iz + 1.0))
    f = py_hash31((ix + 1.0, iy, iz + 1.0))
    g = py_hash31((ix, iy + 1.0, iz + 1.0))
    h = py_hash31((ix + 1.0, iy + 1.0, iz + 1.0))

    va = a * 2.0 - 1.0
    vb = b * 2.0 - 1.0
    vc = c * 2.0 - 1.0
    vd = d * 2.0 - 1.0
    ve = e * 2.0 - 1.0
    vf = f * 2.0 - 1.0
    vg = g * 2.0 - 1.0
    vh = h * 2.0 - 1.0

    vx00 = va + ux * (vb - va)
    vx10 = vc + ux * (vd - vc)
    vx01 = ve + ux * (vf - ve)
    vx11 = vg + ux * (vh - vg)

    vy0 = vx00 + uy * (vx10 - vx00)
    vy1 = vx01 + uy * (vx11 - vx01)

    return vy0 + uz * (vy1 - vy0)


TOL_REL = 1e-9
TOL_ABS = 1e-9

# =============================================================================
# Test: T-DEMO-1.29 Smoothstep Fade Curve
# =============================================================================


class TestSmoothstepCurve:
    """Tests for the smoothstep fade curve (shared by 1D/2D/3D)."""

    def test_smoothstep_at_zero(self):
        """At t=0, smoothstep should be 0."""
        assert py_smoothstep(0.0) == pytest.approx(0.0, abs=TOL_ABS)

    def test_smoothstep_at_one(self):
        """At t=1, smoothstep should be 1."""
        assert py_smoothstep(1.0) == pytest.approx(1.0, abs=TOL_ABS)

    def test_smoothstep_at_half(self):
        """At t=0.5, smoothstep should be exactly 0.5."""
        assert py_smoothstep(0.5) == pytest.approx(0.5, abs=TOL_ABS)

    def test_smoothstep_monotonic(self):
        """Smoothstep should be monotonic in [0, 1]."""
        t_values = [i * 0.01 for i in range(101)]
        values = [py_smoothstep(t) for t in t_values]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-12, (
                f"smoothstep not monotonic at t={t_values[i]}"
            )

    def test_smoothstep_zero_derivative_at_boundaries(self):
        """The smoothstep should have near-zero slope at t=0 and t=1.

        The derivative is 30t^4 - 60t^3 + 30t^2 = 30t^2(t-1)^2, which is
        exactly zero at both t=0 and t=1.
        """
        eps = 1e-6
        d0 = (py_smoothstep(eps) - py_smoothstep(0.0)) / eps
        assert abs(d0) < 1e-8, (
            f"Derivative at 0 should be near zero, got {d0}"
        )
        d1 = (py_smoothstep(1.0) - py_smoothstep(1.0 - eps)) / eps
        assert abs(d1) < 1e-8, (
            f"Derivative at 1 should be near zero, got {d1}"
        )


# =============================================================================
# Test: T-DEMO-1.29 Value Noise 1D
# =============================================================================


class TestValueNoise1D:
    """Tests for value_noise_1d."""

    def test_range(self):
        """Output should always be in [-1, 1]."""
        for i in range(-100, 101):
            p = i * 0.137
            v = py_value_noise_1d(p)
            assert -1.0 <= v <= 1.0 + 1e-12, (
                f"value_noise_1d({p}) = {v} outside [-1, 1]"
            )

    def test_range_dense(self):
        """Dense sampling across multiple grid cells should stay in range."""
        for i in range(500):
            p = -25.0 + i * 0.1
            v = py_value_noise_1d(p)
            assert -1.0 <= v <= 1.0 + 1e-12, (
                f"value_noise_1d({p}) = {v} outside [-1, 1]"
            )

    def test_deterministic(self):
        """Same input always produces same output."""
        v1 = py_value_noise_1d(3.14159)
        for _ in range(10):
            v2 = py_value_noise_1d(3.14159)
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_different_inputs_differ(self):
        """Different inputs should produce different outputs (most of the time)."""
        values = [py_value_noise_1d(i * 0.17) for i in range(20)]
        unique = len(set(round(v, 10) for v in values))
        assert unique > 15, (
            f"Expected at least 15 unique values out of 20, got {unique}"
        )

    def test_integer_grid_matches_hash(self):
        """At integer positions, value noise should equal hash11 remapped."""
        for i in range(-10, 11):
            pi = float(i)
            expected_hash = py_hash11(pi)
            expected = expected_hash * 2.0 - 1.0
            result = py_value_noise_1d(pi)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"At integer {pi}, expected {expected}, got {result}"
            )

    def test_half_integer_between_hash_values(self):
        """At half-integer positions, value should be between adjacent hash values."""
        for i in range(-5, 5):
            pi = float(i)
            half = pi + 0.5
            v_left = py_hash11(pi) * 2.0 - 1.0
            v_right = py_hash11(pi + 1.0) * 2.0 - 1.0
            v_mid = py_value_noise_1d(half)
            lo = min(v_left, v_right)
            hi = max(v_left, v_right)
            assert lo <= v_mid <= hi + 1e-12, (
                f"At half-integer {half}, value {v_mid} not between "
                f"{lo} and {hi}"
            )

    def test_continuity(self):
        """Noise should be continuous with no large jumps over small steps."""
        step = 0.001
        prev = py_value_noise_1d(0.0)
        for i in range(1, 500):
            curr = py_value_noise_1d(i * step)
            diff = abs(curr - prev)
            assert diff < 0.01, (
                f"Discontinuity at {i * step}: diff={diff}"
            )
            prev = curr

    def test_zero_derivative_at_integer_boundaries(self):
        """The first derivative should approach zero at integer boundaries."""
        eps = 1e-6
        for i in range(-5, 6):
            pi = float(i)
            interior = pi + eps
            d = (py_value_noise_1d(interior) - py_value_noise_1d(pi)) / eps
            assert abs(d) < 0.01, (
                f"Derivative at integer {pi} should be near zero, got {d}"
            )


# =============================================================================
# Test: T-DEMO-1.29 Value Noise 2D
# =============================================================================


class TestValueNoise2D:
    """Tests for value_noise_2d."""

    def test_range(self):
        """Output should always be in [-1, 1]."""
        for ix in range(-10, 11):
            for iy in range(-10, 11):
                p = (ix * 0.3, iy * 0.3)
                v = py_value_noise_2d(p)
                assert -1.0 <= v <= 1.0 + 1e-12, (
                    f"value_noise_2d({p}) = {v} outside [-1, 1]"
                )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (2.71828, 3.14159)
        v1 = py_value_noise_2d(p)
        for _ in range(10):
            v2 = py_value_noise_2d(p)
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_integer_grid_matches_hash(self):
        """At integer grid positions, value noise should equal hash21 remapped."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                pi = (float(ix), float(iy))
                expected_hash = py_hash21(pi)
                expected = expected_hash * 2.0 - 1.0
                result = py_value_noise_2d(pi)
                assert result == pytest.approx(expected, abs=TOL_ABS), (
                    f"At integer grid ({ix},{iy}), expected {expected}, "
                    f"got {result}"
                )

    def test_bilinear_interpolation_center(self):
        """At the center of a cell, value should be a weighted average."""
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                center = (ix + 0.5, iy + 0.5)
                a_val = py_hash21((float(ix), float(iy))) * 2.0 - 1.0
                b_val = py_hash21((float(ix + 1), float(iy))) * 2.0 - 1.0
                c_val = py_hash21((float(ix), float(iy + 1))) * 2.0 - 1.0
                d_val = py_hash21((float(ix + 1), float(iy + 1))) * 2.0 - 1.0
                center_avg = (a_val + b_val + c_val + d_val) * 0.25
                result = py_value_noise_2d(center)
                assert abs(result - center_avg) < 0.01, (
                    f"At center ({center}), expected ~{center_avg}, got {result}"
                )

    def test_continuity_along_x(self):
        """Noise should be continuous along x for fixed y."""
        step = 0.001
        y_fixed = 1.5
        prev = py_value_noise_2d((0.0, y_fixed))
        for i in range(1, 300):
            curr = py_value_noise_2d((i * step, y_fixed))
            diff = abs(curr - prev)
            assert diff < 0.02, (
                f"Discontinuity at x={i * step}, y={y_fixed}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """Noise should be continuous along y for fixed x."""
        step = 0.001
        x_fixed = 2.5
        prev = py_value_noise_2d((x_fixed, 0.0))
        for i in range(1, 300):
            curr = py_value_noise_2d((x_fixed, i * step))
            diff = abs(curr - prev)
            assert diff < 0.02, (
                f"Discontinuity at x={x_fixed}, y={i * step}: diff={diff}"
            )
            prev = curr

    def test_grid_alignment_across_cell_boundary(self):
        """Moving from one cell to adjacent should be continuous at boundary."""
        for ix in range(-3, 4):
            left = py_value_noise_2d((ix - 1e-6, 0.5))
            right = py_value_noise_2d((ix + 1e-6, 0.5))
            diff = abs(left - right)
            assert diff < 1e-6, (
                f"Gap at cell boundary x={ix}: diff={diff}"
            )


# =============================================================================
# Test: T-DEMO-1.29 Value Noise 3D
# =============================================================================


class TestValueNoise3D:
    """Tests for value_noise_3d."""

    def test_range(self):
        """Output should always be in [-1, 1]."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5, iy * 0.5, iz * 0.5)
                    v = py_value_noise_3d(p)
                    assert -1.0 <= v <= 1.0 + 1e-12, (
                        f"value_noise_3d({p}) = {v} outside [-1, 1]"
                    )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (1.618, 2.718, 3.142)
        v1 = py_value_noise_3d(p)
        for _ in range(10):
            v2 = py_value_noise_3d(p)
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_integer_grid_matches_hash(self):
        """At integer grid positions, value noise should equal hash31 remapped."""
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                for iz in range(-3, 4):
                    pi = (float(ix), float(iy), float(iz))
                    expected_hash = py_hash31(pi)
                    expected = expected_hash * 2.0 - 1.0
                    result = py_value_noise_3d(pi)
                    assert result == pytest.approx(expected, abs=TOL_ABS), (
                        f"At integer grid ({ix},{iy},{iz}), expected "
                        f"{expected}, got {result}"
                    )

    def test_continuity(self):
        """Noise should be continuous along each axis."""
        step = 0.01
        prev = py_value_noise_3d((0.0, 1.0, 2.0))
        for i in range(1, 200):
            curr = py_value_noise_3d((i * step, 1.0, 2.0))
            assert abs(curr - prev) < 0.05
            prev = curr
        prev = py_value_noise_3d((1.0, 0.0, 2.0))
        for i in range(1, 200):
            curr = py_value_noise_3d((1.0, i * step, 2.0))
            assert abs(curr - prev) < 0.05
            prev = curr
        prev = py_value_noise_3d((1.0, 2.0, 0.0))
        for i in range(1, 200):
            curr = py_value_noise_3d((1.0, 2.0, i * step))
            assert abs(curr - prev) < 0.05
            prev = curr


# =============================================================================
# Test: T-DEMO-1.29 Cross-Dimension Consistency
# =============================================================================


class TestCrossDimension:
    """Tests that value noise dimensions are consistent."""

    def test_2d_at_z_equals_0(self):
        """3D noise with z=0 should NOT equal 2D noise (different hash)."""
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                if ix == 0 and iy == 0:
                    continue
                p2 = (ix * 0.3, iy * 0.3)
                p3 = (ix * 0.3, iy * 0.3, 0.0)
                v2 = py_value_noise_2d(p2)
                v3 = py_value_noise_3d(p3)
                assert -1.0 <= v2 <= 1.0
                assert -1.0 <= v3 <= 1.0
                assert not math.isclose(v2, v3, rel_tol=1e-6), (
                    f"2D and 3D noise should differ at ({p2}, z=0): "
                    f"2D={v2}, 3D={v3}"
                )

    def test_all_dimensions_same_seed_behavior(self):
        """All value noise variants should be deterministic."""
        v1a = py_value_noise_1d(1.5)
        v1b = py_value_noise_1d(1.5)
        assert v1a == pytest.approx(v1b, abs=TOL_ABS)
        v2a = py_value_noise_2d((1.5, 2.5))
        v2b = py_value_noise_2d((1.5, 2.5))
        assert v2a == pytest.approx(v2b, abs=TOL_ABS)
        v3a = py_value_noise_3d((1.5, 2.5, 3.5))
        v3b = py_value_noise_3d((1.5, 2.5, 3.5))
        assert v3a == pytest.approx(v3b, abs=TOL_ABS)


# =============================================================================
# Test: T-DEMO-1.29 Distribution Properties
# =============================================================================


class TestDistribution:
    """Tests that value noise output has reasonable distribution properties."""

    NUM_SAMPLES = 10000

    def test_mean_near_zero(self):
        """Mean of value noise over many points should be near 0."""
        samples_1d = [py_value_noise_1d(i * 0.07) for i in range(self.NUM_SAMPLES)]
        mean_1d = sum(samples_1d) / len(samples_1d)
        assert abs(mean_1d) < 0.1, (
            f"1D value noise mean {mean_1d} not near 0"
        )

        samples_2d = [
            py_value_noise_2d((i * 0.07, i * 0.11))
            for i in range(self.NUM_SAMPLES)
        ]
        mean_2d = sum(samples_2d) / len(samples_2d)
        assert abs(mean_2d) < 0.1, (
            f"2D value noise mean {mean_2d} not near 0"
        )

        samples_3d = [
            py_value_noise_3d((i * 0.07, i * 0.11, i * 0.13))
            for i in range(self.NUM_SAMPLES)
        ]
        mean_3d = sum(samples_3d) / len(samples_3d)
        assert abs(mean_3d) < 0.1, (
            f"3D value noise mean {mean_3d} not near 0"
        )


# =============================================================================
# Test: T-DEMO-1.29 Value Noise Determinism
# =============================================================================


class TestValueNoiseDeterminism:
    """Confirm determinism holds across repeated calls."""

    def test_1d_repeated_calls(self):
        """Repeated calls produce identical results."""
        base = py_value_noise_1d(42.0)
        for _ in range(20):
            assert py_value_noise_1d(42.0) == pytest.approx(base, abs=TOL_ABS)

    def test_2d_repeated_calls(self):
        """Repeated calls produce identical results."""
        base = py_value_noise_2d((42.0, 17.0))
        for _ in range(20):
            assert py_value_noise_2d((42.0, 17.0)) == pytest.approx(base, abs=TOL_ABS)

    def test_3d_repeated_calls(self):
        """Repeated calls produce identical results."""
        base = py_value_noise_3d((42.0, 17.0, 88.0))
        for _ in range(20):
            assert py_value_noise_3d((42.0, 17.0, 88.0)) == pytest.approx(base, abs=TOL_ABS)

    def test_new_seed_1d(self):
        """Different seed should produce different results."""
        results = [py_value_noise_1d(i * 1.0) for i in range(10)]
        unique = len(set(round(v, 10) for v in results))
        assert unique >= 8, (
            f"Expected sufficient unique values, got {unique}"
        )
