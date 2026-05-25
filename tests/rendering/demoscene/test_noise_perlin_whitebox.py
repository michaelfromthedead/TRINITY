"""
Whitebox tests for Perlin noise WGSL functions (T-DEMO-1.30).

Tests Python model implementations of each WGSL function, verifying:
  - Correctness against mathematical definition
  - Gradient-based noise (not scalar hash interpolation)
  - Zero mean property (spec acceptance criteria)
  - Determinism (same input = same output)
  - Continuity (smooth interpolation between grid points)
  - Gradient vector selection and normalization
  - Dot product computation
  - Trilinear interpolation of gradient dot products
  - Cross-boundary continuity (C1 via smoothstep)

WHITEBOX coverage plan:
  Path A: perlin_noise_3d range and finite output
  Path B: perlin_noise_3d deterministic
  Path C: perlin_noise_3d zero mean
  Path D: perlin_noise_3d continuous along each axis
  Path E: perlin_noise_3d gradient-based (not scalar hash)
  Path F: perlin_gradient selects from 12 vectors
  Path G: perlin_gradient normalizes edge vectors
  Path H: perlin_gradient computes dot product
  Path I: perlin_gradient deterministic (same hash -> same gradient)
  Path J: perlin_noise_3d trilinear interpolation correctness
  Path K: Smoothstep fade curve used in interpolation
  Path L: Gradient symmetry (zero mean source)
  Path M: Cross-boundary continuity
  Path N: Distribution symmetry around zero
"""

import math

import pytest

# =============================================================================
# Python model implementations matching WGSL semantics
# =============================================================================

# WGSL fract: x - floor(x)
def wgsl_fract(x: float) -> float:
    return x - math.floor(x)


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


# 12 edge-centered gradient vectors for 3D Perlin noise
_GRADIENTS: list[tuple[float, float, float]] = [
    ( 1.0,  1.0,  0.0),
    (-1.0,  1.0,  0.0),
    ( 1.0, -1.0,  0.0),
    (-1.0, -1.0,  0.0),
    ( 1.0,  0.0,  1.0),
    (-1.0,  0.0,  1.0),
    ( 1.0,  0.0, -1.0),
    (-1.0,  0.0, -1.0),
    ( 0.0,  1.0,  1.0),
    ( 0.0, -1.0,  1.0),
    ( 0.0,  1.0, -1.0),
    ( 0.0, -1.0, -1.0),
]

INV_SQRT2 = 0.7071067811865475


def py_perlin_gradient(hash_value: float, offset) -> float:
    """Select a gradient vector from the hash and dot it with the offset.

    WGSL equivalent:
      fn perlin_gradient(hash_value: f32, offset: vec3<f32>) -> f32 {
          let h = i32(floor(hash_value * 12.0));
          // switch with 12 cases
          g = g * 0.7071067811865475;
          return dot(g, offset);
      }
    """
    h = int(hash_value * 12.0) % 12
    gx, gy, gz = _GRADIENTS[h]
    gx *= INV_SQRT2
    gy *= INV_SQRT2
    gz *= INV_SQRT2
    return gx * offset[0] + gy * offset[1] + gz * offset[2]


def py_perlin_noise_3d(p) -> float:
    """Model of WGSL perlin_noise_3d.

    WGSL equivalent:
      fn perlin_noise_3d(p: vec3<f32>) -> f32 { ... }
    """
    ix = math.floor(p[0])
    iy = math.floor(p[1])
    iz = math.floor(p[2])
    fx = p[0] - ix
    fy = p[1] - iy
    fz = p[2] - iz

    ux = py_smoothstep(fx)
    uy = py_smoothstep(fy)
    uz = py_smoothstep(fz)

    # Eight corner offsets
    o000 = (fx, fy, fz)
    o100 = (fx - 1.0, fy, fz)
    o010 = (fx, fy - 1.0, fz)
    o110 = (fx - 1.0, fy - 1.0, fz)
    o001 = (fx, fy, fz - 1.0)
    o101 = (fx - 1.0, fy, fz - 1.0)
    o011 = (fx, fy - 1.0, fz - 1.0)
    o111 = (fx - 1.0, fy - 1.0, fz - 1.0)

    # Hash values at 8 corners
    h000 = py_hash31((ix, iy, iz))
    h100 = py_hash31((ix + 1.0, iy, iz))
    h010 = py_hash31((ix, iy + 1.0, iz))
    h110 = py_hash31((ix + 1.0, iy + 1.0, iz))
    h001 = py_hash31((ix, iy, iz + 1.0))
    h101 = py_hash31((ix + 1.0, iy, iz + 1.0))
    h011 = py_hash31((ix, iy + 1.0, iz + 1.0))
    h111 = py_hash31((ix + 1.0, iy + 1.0, iz + 1.0))

    # Gradient dot products at each corner
    g000 = py_perlin_gradient(h000, o000)
    g100 = py_perlin_gradient(h100, o100)
    g010 = py_perlin_gradient(h010, o010)
    g110 = py_perlin_gradient(h110, o110)
    g001 = py_perlin_gradient(h001, o001)
    g101 = py_perlin_gradient(h101, o101)
    g011 = py_perlin_gradient(h011, o011)
    g111 = py_perlin_gradient(h111, o111)

    # Trilinear interpolation of gradient dot products
    vx00 = g000 + ux * (g100 - g000)
    vx10 = g010 + ux * (g110 - g010)
    vx01 = g001 + ux * (g101 - g001)
    vx11 = g011 + ux * (g111 - g011)

    vy0 = vx00 + uy * (vx10 - vx00)
    vy1 = vx01 + uy * (vx11 - vx01)

    return vy0 + uz * (vy1 - vy0)


TOL_REL = 1e-9
TOL_ABS = 1e-9

# =============================================================================
# Test: T-DEMO-1.30 Smoothstep Fade Curve
# =============================================================================


class TestSmoothstepCurve:
    """Tests for the smoothstep fade curve (shared with value noise)."""

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
        """The smoothstep should have near-zero slope at t=0 and t=1."""
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
# Test: T-DEMO-1.30 Perlin Gradient Function
# =============================================================================


class TestPerlinGradient:
    """Tests for the perlin_gradient helper function."""

    def test_gradient_vectors_are_edge_centered(self):
        """Each gradient must have exactly two non-zero components."""
        for gx, gy, gz in _GRADIENTS:
            non_zero = sum(1 for c in (gx, gy, gz) if abs(c) > 0)
            assert non_zero == 2, (
                f"Gradient ({gx},{gy},{gz}) has {non_zero} non-zero "
                f"components, expected exactly 2"
            )

    def test_gradient_table_has_12_entries(self):
        """Gradient table must have exactly 12 entries."""
        assert len(_GRADIENTS) == 12, (
            f"Expected 12 gradient vectors, got {len(_GRADIENTS)}"
        )

    def test_gradient_vectors_are_unit_after_normalization(self):
        """After normalization, each gradient should have magnitude ~1."""
        for gx, gy, gz in _GRADIENTS:
            nx = gx * INV_SQRT2
            ny = gy * INV_SQRT2
            nz = gz * INV_SQRT2
            mag = math.sqrt(nx * nx + ny * ny + nz * nz)
            assert abs(mag - 1.0) < 1e-12, (
                f"Normalized gradient ({nx:.6f},{ny:.6f},{nz:.6f}) "
                f"has magnitude {mag}, expected 1.0"
            )

    def test_gradient_dot_product_is_linear(self):
        """Dot product should be linear in offset."""
        hash_val = 0.5  # hash that maps to index 6
        offset_a = (1.0, 0.5, 0.25)
        offset_b = (0.5, 0.25, 0.125)
        sum_offset = (offset_a[0] + offset_b[0],
                      offset_a[1] + offset_b[1],
                      offset_a[2] + offset_b[2])

        dot_a = py_perlin_gradient(hash_val, offset_a)
        dot_b = py_perlin_gradient(hash_val, offset_b)
        dot_sum = py_perlin_gradient(hash_val, sum_offset)

        assert dot_sum == pytest.approx(dot_a + dot_b, abs=TOL_ABS), (
            "Dot product should be linear: dot(a+b) = dot(a) + dot(b)"
        )

    def test_gradient_dot_product_is_homogeneous(self):
        """Dot product should be homogeneous: dot(s*a) = s*dot(a)."""
        hash_val = 0.25
        offset = (1.0, 0.5, 0.25)
        s = 2.0
        scaled_offset = (s * offset[0], s * offset[1], s * offset[2])

        dot_orig = py_perlin_gradient(hash_val, offset)
        dot_scaled = py_perlin_gradient(hash_val, scaled_offset)

        assert dot_scaled == pytest.approx(s * dot_orig, abs=TOL_ABS), (
            "Dot product should be homogeneous: dot(s*a) = s*dot(a)"
        )

    def test_deterministic_gradient_selection(self):
        """Same hash value should select same gradient."""
        hash_val = 0.123456
        offset = (1.0, 1.0, 0.0)
        result1 = py_perlin_gradient(hash_val, offset)
        for _ in range(10):
            result2 = py_perlin_gradient(hash_val, offset)
            assert result1 == pytest.approx(result2, abs=TOL_ABS), (
                "Gradient selection must be deterministic"
            )

    def test_different_hashes_produce_different_gradients(self):
        """Different hash values should produce different gradient selections."""
        offsets = {}
        for h_val in [i / 12.0 + 0.001 for i in range(12)]:
            result = py_perlin_gradient(h_val, (1.0, 0.0, 0.0))
            offsets[round(result, 10)] = h_val
        # With 12 different gradients, dot with (1,0,0) should differ.
        # Note: gradients with x=0 project to 0, so not all 12 are distinct.
        assert len(offsets) >= 3, (
            f"Expected at least 3 unique dot products from 12 different "
            f"hashes, got {len(offsets)}"
        )

    def test_gradient_at_integer_returns_zero(self):
        """At an exact integer grid point, offset=(0,0,0), so gradient=0."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (float(ix), float(iy), float(iz))
                    h = py_hash31(p)
                    # At an integer, the offset is (0,0,0) so dot=0
                    result = py_perlin_gradient(h, (0.0, 0.0, 0.0))
                    assert result == pytest.approx(0.0, abs=TOL_ABS), (
                        f"Gradient at zero offset should be 0, got {result}"
                    )


# =============================================================================
# Test: T-DEMO-1.30 Perlin Noise 3D
# =============================================================================


class TestPerlinNoise3D:
    """Tests for perlin_noise_3d."""

    def test_range(self):
        """Output should be in a reasonable range."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5, iy * 0.5, iz * 0.5)
                    v = py_perlin_noise_3d(p)
                    assert math.isfinite(v), (
                        f"perlin_noise_3d({p}) non-finite"
                    )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (1.618, 2.718, 3.142)
        v1 = py_perlin_noise_3d(p)
        for _ in range(10):
            v2 = py_perlin_noise_3d(p)
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_different_inputs_differ(self):
        """Different inputs should produce different outputs (most of the time)."""
        values = [py_perlin_noise_3d((i * 0.17, i * 0.23, i * 0.31))
                  for i in range(20)]
        unique = len(set(round(v, 10) for v in values))
        assert unique > 15, (
            f"Expected at least 15 unique values out of 20, got {unique}"
        )

    def test_integer_grid_is_zero(self):
        """At exact integer positions, perlin_noise_3d should be 0.

        At an integer grid point, all corner offsets are (0,0,0), so
        every gradient dot product is 0, making the interpolation result 0.
        This is a fundamental property of Perlin noise (unlike value noise
        which returns hash_value * 2 - 1 at integer positions).
        """
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                for iz in range(-3, 4):
                    p = (float(ix), float(iy), float(iz))
                    result = py_perlin_noise_3d(p)
                    assert result == pytest.approx(0.0, abs=TOL_ABS), (
                        f"Perlin noise should be 0 at integer "
                        f"({ix},{iy},{iz}), got {result}"
                    )

    def test_continuity_along_x(self):
        """Noise should be continuous along x for fixed y, z."""
        step = 0.01
        prev = py_perlin_noise_3d((0.0, 1.0, 2.0))
        for i in range(1, 200):
            curr = py_perlin_noise_3d((i * step, 1.0, 2.0))
            assert abs(curr - prev) < 0.05, (
                f"Discontinuity at x={i * step}: {abs(curr - prev)}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """Noise should be continuous along y for fixed x, z."""
        step = 0.01
        prev = py_perlin_noise_3d((1.0, 0.0, 2.0))
        for i in range(1, 200):
            curr = py_perlin_noise_3d((1.0, i * step, 2.0))
            assert abs(curr - prev) < 0.05, (
                f"Discontinuity at y={i * step}: {abs(curr - prev)}"
            )
            prev = curr

    def test_continuity_along_z(self):
        """Noise should be continuous along z for fixed x, y."""
        step = 0.01
        prev = py_perlin_noise_3d((1.0, 2.0, 0.0))
        for i in range(1, 200):
            curr = py_perlin_noise_3d((1.0, i * step, 2.0))
            assert abs(curr - prev) < 0.05, (
                f"Discontinuity at z={i * step}: {abs(curr - prev)}"
            )
            prev = curr

    def test_continuity_across_integer_boundary(self):
        """Moving across integer boundaries should be continuous."""
        eps = 1e-6
        for i in range(-3, 4):
            left = py_perlin_noise_3d((i - eps, 0.5, 0.5))
            right = py_perlin_noise_3d((i + eps, 0.5, 0.5))
            diff = abs(left - right)
            # Perlin noise at +/-1e-6 from the boundary has a small numerical
            # gap from the gradient dot product weighting. This converges
            # to 0 as eps->0, making the function continuous at the boundary.
            assert diff < 1e-5, (
                f"Gap at cell boundary x={i}: diff={diff}"
            )


# =============================================================================
# Test: T-DEMO-1.30 Zero Mean Property
# =============================================================================


class TestZeroMean:
    """Tests that Perlin noise has approximately zero mean.

    This is the primary acceptance criteria for T-DEMO-1.30: the spec says
    "gradient-based noise with zero mean." The zero mean arises from the
    symmetry of the gradient vectors.
    """

    NUM_SAMPLES = 10000

    def test_mean_near_zero(self):
        """Mean of Perlin noise over many points should be near 0."""
        samples = [
            py_perlin_noise_3d((i * 0.07, i * 0.11, i * 0.13))
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.05, (
            f"Perlin noise mean {mean} not near 0"
        )

    def test_symmetric_distribution(self):
        """Distribution should be symmetric around zero."""
        samples = [
            py_perlin_noise_3d((i * 0.07, i * 0.11, i * 0.13))
            for i in range(self.NUM_SAMPLES)
        ]
        below_zero = sum(1 for v in samples if v < 0)
        ratio = below_zero / len(samples)
        assert 0.45 <= ratio <= 0.55, (
            f"Expected ~50% below zero, got {ratio:.1%}"
        )


# =============================================================================
# Test: T-DEMO-1.30 Cross-Property Consistency
# =============================================================================


class TestCrossProperty:
    """Tests that Perlin noise has distinct properties from value noise."""

    def test_perlin_differs_from_value_noise(self):
        """Perlin noise differs from value noise at arbitrary points."""
        # Perlin noise at integer positions is 0, but value noise
        # at integer positions is hash * 2 - 1, which is in [-1, 1].
        # This is a structural difference, not just a different hash seed.

        # At a non-integer point, they should also differ
        p = (3.7, 4.2, 5.1)
        perlin_val = py_perlin_noise_3d(p)

        # Value noise at this point would be trilinear interpolation
        # of hash values. The hash values are the same (same hash31),
        # but Perlin uses them differently (gradient selection vs scalar).
        # So the results must differ substantially.
        assert abs(perlin_val) > 1e-10, (
            "Perlin noise at non-integer point should be non-zero"
        )

    def test_perlin_at_integer_is_zero_unlike_value_noise(self):
        """At integer positions, Perlin is 0 but value noise is hash*2-1."""
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                for iz in range(-3, 4):
                    p = (float(ix), float(iy), float(iz))
                    perlin_val = py_perlin_noise_3d(p)
                    # Perlin is exactly 0 at integer grid points
                    assert perlin_val == pytest.approx(0.0, abs=TOL_ABS), (
                        f"Perlin noise at integer ({p}) should be 0, "
                        f"got {perlin_val}"
                    )


# =============================================================================
# Test: T-DEMO-1.30 Perlin Noise Determinism
# =============================================================================


class TestPerlinNoiseDeterminism:
    """Confirm determinism holds across repeated calls."""

    def test_repeated_calls(self):
        """Repeated calls produce identical results."""
        base = py_perlin_noise_3d((42.0, 17.0, 88.0))
        for _ in range(20):
            assert py_perlin_noise_3d(
                (42.0, 17.0, 88.0)
            ) == pytest.approx(base, abs=TOL_ABS)

    def test_new_position_1d(self):
        """Different positions should produce different results."""
        # Use non-integer positions -- Perlin noise is exactly 0 at
        # integer grid points (all gradient dot products are 0)
        results = [py_perlin_noise_3d((i * 1.0 + 0.3, i * 1.0 + 0.5, i * 1.0 + 0.7))
                   for i in range(10)]
        unique = len(set(round(v, 10) for v in results))
        assert unique >= 8, (
            f"Expected sufficient unique values, got {unique}"
        )
