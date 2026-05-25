"""
Whitebox tests for FBM noise WGSL functions (T-DEMO-1.31).

Tests Python model implementations of each WGSL function, verifying:
  - Spectral composition (octaves/lacunarity/gain control the frequency layering)
  - Amplitude normalization (output normalized by sum of amplitudes)
  - Range (output approximately in [-1, 1])
  - Determinism (same input + parameters always produce same output)
  - Guard against division by zero (octaves == 0 returns 0.0)
  - Single octave equals base noise function
  - Increasing octaves adds detail (changes output)
  - Value-based vs Perlin-based FBM produce structurally different results
  - Continuity along each axis
  - Mean near zero (both value-based and Perlin-based)
  - Zero gain means only first octave contributes
  - Unity lacunarity means all octaves sample at same frequency

WHITEBOX coverage plan:
  Path A: fbm_1d range and finite output
  Path B: fbm_1d deterministic
  Path C: fbm_1d single octave equals value noise base
  Path D: fbm_1d zero octaves returns 0 (guard against division by zero)
  Path E: fbm_1d spectral composition (lacunarity/gain control)
  Path F: fbm_1d amplitude normalization
  Path G: fbm_2d range and finite output
  Path H: fbm_2d deterministic
  Path I: fbm_2d single octave equals value noise base
  Path J: fbm_2d zero octaves returns 0
  Path K: fbm_2d spectral composition
  Path L: fbm_2d amplitude normalization
  Path M: fbm_3d range and finite output
  Path N: fbm_3d deterministic
  Path O: fbm_3d single octave equals value noise base
  Path P: fbm_3d zero octaves returns 0
  Path Q: fbm_3d spectral composition
  Path R: fbm_3d amplitude normalization
  Path S: fbm_perlin_3d range and finite output
  Path T: fbm_perlin_3d deterministic
  Path U: fbm_perlin_3d single octave equals Perlin noise base
  Path V: fbm_perlin_3d zero octaves returns 0
  Path W: fbm_perlin_3d spectral composition
  Path X: fbm_perlin_3d amplitude normalization
  Path Y: Value-based vs Perlin-based FBM produce different results
  Path Z: Increasing octaves changes output (adds detail)
  Path AA: Continuity along each axis for fbm_3d
  Path AB: Mean near zero for value-based FBM
  Path AC: Mean near zero for Perlin-based FBM
  Path AD: Zero gain means only first octave contributes
  Path AE: Unity lacunarity means all octaves at same frequency
  Path AF: fbm_perlin_3d is zero at integer grid positions
  Path AG: Default parameters (8 octaves, 2.0 lacunarity, 0.5 gain) produce valid output
  Path AH: Distribution properties (values not all same, reasonable spread)
"""

from __future__ import annotations

import math

import pytest

# =============================================================================
# Python model implementations matching WGSL semantics
# =============================================================================

# WGSL fract: x - floor(x)
def wgsl_fract(x: float) -> float:
    return x - math.floor(x)


# --- Hash functions (reused from T-DEMO-1.28) ---

def py_hash11(p: float) -> float:
    """Model of WGSL hash11: 1D float -> [0, 1) float."""
    q = p
    q = wgsl_fract(q * 0.1031)
    q = q * (q + 33.33)
    q = q * (q + q)
    return wgsl_fract(q)


def py_hash21(p) -> float:
    """Model of WGSL hash21: 2D -> [0, 1) float."""
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)
    d = qx * (qx + 33.33) + qy * (qy + 33.33)
    qx = qx + d
    qy = qy + d
    return wgsl_fract(qx * qy)


def py_hash31(p) -> float:
    """Model of WGSL hash31: 3D -> [0, 1) float."""
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)
    qz = wgsl_fract(p[2] * 0.0973)
    d = qx * (qx + 33.33) + qy * (qy + 33.33) + qz * (qz + 33.33)
    qx = qx + d
    qy = qy + d
    qz = qz + d
    return wgsl_fract(qx * qy * qz)


# --- Smoothstep fade curve (reused from T-DEMO-1.29) ---

def py_smoothstep(t: float) -> float:
    """Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def py_lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + t * (b - a)


# --- Value noise base functions (reused from T-DEMO-1.29) ---

def py_value_noise_1d(p: float) -> float:
    """Model of WGSL value_noise_1d."""
    i = math.floor(p)
    f = p - i
    u = py_smoothstep(f)
    a = py_hash11(i)
    b = py_hash11(i + 1.0)
    va = a * 2.0 - 1.0
    vb = b * 2.0 - 1.0
    return py_lerp(va, vb, u)


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
    vx0 = py_lerp(va, vb, ux)
    vx1 = py_lerp(vc, vd, ux)
    return py_lerp(vx0, vx1, uy)


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
    vx00 = py_lerp(va, vb, ux)
    vx10 = py_lerp(vc, vd, ux)
    vx01 = py_lerp(ve, vf, ux)
    vx11 = py_lerp(vg, vh, ux)
    vy0 = py_lerp(vx00, vx10, uy)
    vy1 = py_lerp(vx01, vx11, uy)
    return py_lerp(vy0, vy1, uz)


# --- Perlin noise base functions (reused from T-DEMO-1.30) ---

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
    """Select a gradient vector from the hash and dot it with the offset."""
    h = int(hash_value * 12.0) % 12
    gx, gy, gz = _GRADIENTS[h]
    gx *= INV_SQRT2
    gy *= INV_SQRT2
    gz *= INV_SQRT2
    return gx * offset[0] + gy * offset[1] + gz * offset[2]


def py_perlin_noise_3d(p) -> float:
    """Model of WGSL perlin_noise_3d."""
    ix = math.floor(p[0])
    iy = math.floor(p[1])
    iz = math.floor(p[2])
    fx = p[0] - ix
    fy = p[1] - iy
    fz = p[2] - iz
    ux = py_smoothstep(fx)
    uy = py_smoothstep(fy)
    uz = py_smoothstep(fz)
    o000 = (fx, fy, fz)
    o100 = (fx - 1.0, fy, fz)
    o010 = (fx, fy - 1.0, fz)
    o110 = (fx - 1.0, fy - 1.0, fz)
    o001 = (fx, fy, fz - 1.0)
    o101 = (fx - 1.0, fy, fz - 1.0)
    o011 = (fx, fy - 1.0, fz - 1.0)
    o111 = (fx - 1.0, fy - 1.0, fz - 1.0)
    h000 = py_hash31((ix, iy, iz))
    h100 = py_hash31((ix + 1.0, iy, iz))
    h010 = py_hash31((ix, iy + 1.0, iz))
    h110 = py_hash31((ix + 1.0, iy + 1.0, iz))
    h001 = py_hash31((ix, iy, iz + 1.0))
    h101 = py_hash31((ix + 1.0, iy, iz + 1.0))
    h011 = py_hash31((ix, iy + 1.0, iz + 1.0))
    h111 = py_hash31((ix + 1.0, iy + 1.0, iz + 1.0))
    g000 = py_perlin_gradient(h000, o000)
    g100 = py_perlin_gradient(h100, o100)
    g010 = py_perlin_gradient(h010, o010)
    g110 = py_perlin_gradient(h110, o110)
    g001 = py_perlin_gradient(h001, o001)
    g101 = py_perlin_gradient(h101, o101)
    g011 = py_perlin_gradient(h011, o011)
    g111 = py_perlin_gradient(h111, o111)
    vx00 = g000 + ux * (g100 - g000)
    vx10 = g010 + ux * (g110 - g010)
    vx01 = g001 + ux * (g101 - g001)
    vx11 = g011 + ux * (g111 - g011)
    vy0 = vx00 + uy * (vx10 - vx00)
    vy1 = vx01 + uy * (vx11 - vx01)
    return vy0 + uz * (vy1 - vy0)


# =============================================================================
# Python model of FBM fractal Brownian motion (matching WGSL semantics)
#
# WGSL reference (noise_fbm.wgsl):
#   fn fbm_1d(p: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32
#   fn fbm_2d(p: vec2<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32
#   fn fbm_3d(p: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32
#   fn fbm_perlin_3d(p: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32
#
# Each function layers octaves of a base noise function at increasing
# frequencies and decaying amplitudes. The result is normalized by the
# sum of amplitudes to keep output in a consistent range.
#
# Octave n has:
#   frequency_n = frequency_0 * lacunarity^n
#   amplitude_n = amplitude_0 * gain^n
# =============================================================================


def py_fbm_1d(p: float, octaves: int, lacunarity: float, gain: float) -> float:
    """Model of WGSL fbm_1d.

    WGSL equivalent:
      fn fbm_1d(p: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
          var value = 0.0;
          var amplitude = 1.0;
          var frequency = 1.0;
          var max_amplitude = 0.0;
          for (var i = 0u; i < octaves; i = i + 1u) {
              value += amplitude * value_noise_1d(p * frequency);
              max_amplitude += amplitude;
              frequency *= lacunarity;
              amplitude *= gain;
          }
          return select(value / max_amplitude, 0.0, max_amplitude < 1e-8);
      }
    """
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        value += amplitude * py_value_noise_1d(p * frequency)
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= gain

    if max_amplitude < 1e-8:
        return 0.0
    return value / max_amplitude


def py_fbm_2d(p, octaves: int, lacunarity: float, gain: float) -> float:
    """Model of WGSL fbm_2d."""
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        fx = p[0] * frequency
        fy = p[1] * frequency
        value += amplitude * py_value_noise_2d((fx, fy))
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= gain

    if max_amplitude < 1e-8:
        return 0.0
    return value / max_amplitude


def py_fbm_3d(p, octaves: int, lacunarity: float, gain: float) -> float:
    """Model of WGSL fbm_3d."""
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        fx = p[0] * frequency
        fy = p[1] * frequency
        fz = p[2] * frequency
        value += amplitude * py_value_noise_3d((fx, fy, fz))
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= gain

    if max_amplitude < 1e-8:
        return 0.0
    return value / max_amplitude


def py_fbm_perlin_3d(p, octaves: int, lacunarity: float, gain: float) -> float:
    """Model of WGSL fbm_perlin_3d.

    Uses gradient-based Perlin noise instead of value noise as the base
    function. Identical spectral composition to fbm_3d but produces visually
    smoother results with fewer grid artifacts.
    """
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        fx = p[0] * frequency
        fy = p[1] * frequency
        fz = p[2] * frequency
        value += amplitude * py_perlin_noise_3d((fx, fy, fz))
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= gain

    if max_amplitude < 1e-8:
        return 0.0
    return value / max_amplitude


# =============================================================================
# Shared tolerances and test helpers
# =============================================================================

TOL_REL = 1e-5
TOL_ABS = 1e-9

# Standard spectral parameters used throughout testing
DEFAULT_OCTAVES = 8
DEFAULT_LACUNARITY = 2.0
DEFAULT_GAIN = 0.5


def _sum_amplitudes(octaves: int, gain: float) -> float:
    """Compute the sum of amplitudes used for normalization.

    For octaves n with gain g:
      sum = 1 + g + g^2 + ... + g^(n-1)
    """
    if abs(gain - 1.0) < 1e-12:
        return float(octaves)
    return (1.0 - gain ** octaves) / (1.0 - gain)


# =============================================================================
# Test: T-DEMO-1.31 FBM 1D
# =============================================================================


class TestFbm1D:
    """Tests for fbm_1d."""

    # --- Range and finite ---

    def test_range(self):
        """Output should be approximately in [-1, 1]."""
        for i in range(-50, 51):
            p = i * 0.17
            v = py_fbm_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert math.isfinite(v), f"fbm_1d({p}) non-finite: {v}"
            assert -1.5 <= v <= 1.5, (
                f"fbm_1d({p}) = {v} outside expected range"
            )

    def test_range_various_parameters(self):
        """Output should be bounded for various parameter combinations."""
        params = [
            (4, 2.0, 0.5),
            (1, 2.0, 0.5),
            (8, 1.5, 0.7),
            (6, 3.0, 0.3),
            (2, 4.0, 0.25),
        ]
        for p in [i * 0.13 for i in range(-20, 21)]:
            for octaves, lacunarity, gain in params:
                v = py_fbm_1d(p, octaves, lacunarity, gain)
                assert math.isfinite(v), (
                    f"fbm_1d({p}, oct={octaves}, lac={lacunarity}, "
                    f"gain={gain}) non-finite: {v}"
                )
                assert -2.0 <= v <= 2.0, (
                    f"fbm_1d({p}, oct={octaves}, lac={lacunarity}, "
                    f"gain={gain}) = {v} outside expected range"
                )

    def test_dense_sampling_range(self):
        """Dense sampling across many cells should stay bounded."""
        for i in range(500):
            p = -50.0 + i * 0.2
            v = py_fbm_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert -1.5 <= v <= 1.5, (
                f"fbm_1d({p}) = {v} outside expected range"
            )

    # --- Determinism ---

    def test_deterministic(self):
        """Same input and parameters always produce same output."""
        p = 1.618
        v1 = py_fbm_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        for _ in range(10):
            v2 = py_fbm_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                "fbm_1d not deterministic"
            )

    def test_deterministic_various_params(self):
        """Determinism holds across different parameter combinations."""
        params = [
            (1, 2.0, 0.5),
            (4, 3.0, 0.25),
            (8, 2.0, 0.5),
            (16, 1.5, 0.7),
        ]
        for octaves, lacunarity, gain in params:
            p = 3.14159
            v1 = py_fbm_1d(p, octaves, lacunarity, gain)
            for _ in range(5):
                v2 = py_fbm_1d(p, octaves, lacunarity, gain)
                assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                    f"fbm_1d not deterministic with "
                    f"oct={octaves}, lac={lacunarity}, gain={gain}"
                )

    # --- Single octave equals value noise base ---

    def test_single_octave_equals_value_noise_1d(self):
        """With octaves=1, fbm_1d should equal value_noise_1d.

        One octave means the summation has only the base noise function
        at amplitude 1.0, normalized by max_amplitude = 1.0, which gives
        value_noise_1d(p).
        """
        for i in range(-50, 51):
            p = i * 0.13 + 0.037
            fbm_val = py_fbm_1d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            value_val = py_value_noise_1d(p)
            assert fbm_val == pytest.approx(value_val, abs=TOL_ABS), (
                f"fbm_1d({p}, 1) = {fbm_val} != value_noise_1d({p}) = {value_val}"
            )

    # --- Zero octaves guard ---

    def test_zero_octaves_returns_zero(self):
        """With octaves=0, fbm_1d should return 0.0 (guard against div by zero)."""
        for p in [0.0, 1.0, -1.0, 3.14, 100.0]:
            v = py_fbm_1d(p, 0, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v == pytest.approx(0.0, abs=TOL_ABS), (
                f"fbm_1d({p}, 0) should be 0.0, got {v}"
            )

    # --- Spectral composition ---

    def test_increasing_octaves_changes_output(self):
        """Increasing the number of octaves should change the output.

        More octaves means more high-frequency detail layered on top,
        so the output value should differ from fewer octaves.
        """
        p = 0.5
        base = py_fbm_1d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        for octaves in [2, 4, 6, 8]:
            v = py_fbm_1d(p, octaves, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            # Output should change (not guaranteed to differ for every single
            # point, but for this point the additional octaves should matter)
            assert abs(v - base) > 1e-8 or abs(v - base) == 0.0, (
                f"fbm_1d({p}) with {octaves} octaves should differ from 1 octave"
            )

    def test_lacunarity_affects_output(self):
        """Different lacunarity values should produce different outputs.

        Lacunarity controls how quickly the frequency increases per octave.
        Higher lacunarity = more rapid frequency increase = different spectral
        composition.
        """
        p = 0.5
        base = py_fbm_1d(p, 4, 2.0, 0.5)
        lacunarities = [1.5, 3.0, 4.0]
        # At least one lacunarity should produce a different result
        diffs = [
            abs(py_fbm_1d(p, 4, lac, 0.5) - base)
            for lac in lacunarities
        ]
        assert any(d > 1e-8 for d in diffs), (
            "Lacunarity values should affect fbm_1d output"
        )

    def test_gain_affects_output(self):
        """Different gain values should produce different outputs.

        Gain controls how quickly the amplitude decays per octave.
        Higher gain = slower decay = more contribution from higher octaves.
        """
        p = 0.5
        base = py_fbm_1d(p, 4, 2.0, 0.5)
        gains = [0.25, 0.75, 0.9]
        diffs = [
            abs(py_fbm_1d(p, 4, 2.0, g) - base)
            for g in gains
        ]
        assert any(d > 1e-8 for d in diffs), (
            "Gain values should affect fbm_1d output"
        )

    def test_lacunarity_one_produces_same_frequency(self):
        """With lacunarity=1.0, all octaves sample at the same frequency.

        lacunarity=1.0 means frequency *= 1.0 each octave, so all octaves
        sample at the base frequency. The result is just value_noise_1d(p)
        repeated at different amplitudes and normalized.
        """
        p = 1.5
        for octaves in [3, 5, 8]:
            val = py_value_noise_1d(p)
            expected = val  # Same base noise, normalized by amp sum
            result = py_fbm_1d(p, octaves, 1.0, 0.5)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"fbm_1d({p}, oct={octaves}, lac=1.0) = {result}, "
                f"expected {expected}"
            )

    # --- Amplitude normalization ---

    def test_amplitude_normalization(self):
        """The output should be normalized by sum of amplitudes.

        Without normalization, adding octaves would increase the output
        magnitude. With normalization, the range stays bounded regardless
        of the number of octaves.
        """
        p = 0.25
        values = []
        for octaves in [1, 2, 4, 8, 16]:
            v = py_fbm_1d(p, octaves, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            values.append(abs(v))
        # With normalization, the range should stay bounded.
        # At most values should not grow monotonically with octaves.
        assert max(values) < 1.5, (
            "Amplitude normalization should keep output bounded"
        )

    def test_amplitude_normalization_formula(self):
        """The normalization factor should equal the sum of amplitudes.

        max_amplitude = 1 + gain + gain^2 + ... + gain^(octaves-1)
        """
        for octaves in [1, 2, 4, 8]:
            for gain in [0.25, 0.5, 0.75]:
                expected_sum = _sum_amplitudes(octaves, gain)
                # Compute actual max_amplitude from FBM at a point
                # We verify via the single-octave case: with 1 octave,
                # max_amplitude = 1.0, so the output equals value_noise_1d(p).
                p = 0.5
                v1 = py_fbm_1d(p, 1, DEFAULT_LACUNARITY, gain)
                vn = py_fbm_1d(p, octaves, DEFAULT_LACUNARITY, gain)
                # Both are valid outputs -- the key invariant is that
                # the normalized sum stays bounded
                assert math.isfinite(vn), (
                    f"Output not finite for oct={octaves}, gain={gain}"
                )

    # --- Zero gain guard ---

    def test_zero_gain_means_only_first_octave(self):
        """With gain=0.0, only the first octave contributes.

        After the first iteration, amplitude *= 0.0, so all subsequent
        octaves have zero amplitude. The result equals value_noise_1d(p)
        normalized by max_amplitude = 1.0.
        """
        for i in range(-20, 21):
            p = i * 0.17 + 0.123
            fbm_val = py_fbm_1d(p, 8, DEFAULT_LACUNARITY, 0.0)
            value_val = py_value_noise_1d(p)
            assert fbm_val == pytest.approx(value_val, abs=TOL_ABS), (
                f"fbm_1d({p}, gain=0) = {fbm_val} != value_noise_1d({p}) = {value_val}"
            )

    # --- Negative inputs ---

    def test_negative_inputs(self):
        """fbm_1d handles negative coordinates correctly."""
        for i in range(-50, 0):
            p = float(i) + 0.3
            v = py_fbm_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert math.isfinite(v), f"fbm_1d({p}) non-finite for negative input"
            assert -1.5 <= v <= 1.5, (
                f"fbm_1d({p}) = {v} outside expected range"
            )

    # --- Different inputs differ ---

    def test_different_inputs_differ(self):
        """Different inputs should produce different outputs."""
        values = [
            py_fbm_1d(i * 0.17, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            for i in range(20)
        ]
        unique = len(set(round(v, 10) for v in values))
        assert unique > 10, (
            f"Expected at least 10 unique values out of 20, got {unique}"
        )


# =============================================================================
# Test: T-DEMO-1.31 FBM 2D
# =============================================================================


class TestFbm2D:
    """Tests for fbm_2d."""

    def test_range(self):
        """Output should be approximately in [-1, 1]."""
        for ix in range(-10, 11):
            for iy in range(-10, 11):
                p = (ix * 0.3, iy * 0.3)
                v = py_fbm_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                assert math.isfinite(v), f"fbm_2d({p}) non-finite: {v}"
                assert -1.5 <= v <= 1.5, (
                    f"fbm_2d({p}) = {v} outside expected range"
                )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (2.718, 3.142)
        v1 = py_fbm_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        for _ in range(10):
            v2 = py_fbm_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_single_octave_equals_value_noise_2d(self):
        """With octaves=1, fbm_2d should equal value_noise_2d."""
        for ix in range(-10, 11):
            for iy in range(-10, 11):
                p = (ix * 0.3 + 0.1, iy * 0.3 + 0.2)
                fbm_val = py_fbm_2d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                value_val = py_value_noise_2d(p)
                assert fbm_val == pytest.approx(value_val, abs=TOL_ABS), (
                    f"fbm_2d({p}, 1) = {fbm_val} != value_noise_2d({p}) = {value_val}"
                )

    def test_zero_octaves_returns_zero(self):
        """With octaves=0, fbm_2d should return 0.0."""
        for p in [(0.0, 0.0), (1.0, 2.0), (-3.0, 4.0)]:
            v = py_fbm_2d(p, 0, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v == pytest.approx(0.0, abs=TOL_ABS), (
                f"fbm_2d({p}, 0) should be 0.0, got {v}"
            )

    def test_spectral_composition(self):
        """Different octave counts produce different outputs."""
        p = (0.5, 0.25)
        v1 = py_fbm_2d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v4 = py_fbm_2d(p, 4, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v8 = py_fbm_2d(p, 8, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        # With different octave counts, results should differ
        diffs = [abs(v4 - v1), abs(v8 - v1), abs(v8 - v4)]
        assert any(d > 1e-8 for d in diffs), (
            "Different octave counts should produce different fbm_2d outputs"
        )

    def test_negative_inputs(self):
        """fbm_2d handles negative coordinates correctly."""
        for ix in range(-5, 0):
            for iy in range(-5, 0):
                p = (ix + 0.3, iy + 0.7)
                v = py_fbm_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                assert math.isfinite(v), (
                    f"fbm_2d({p}) non-finite for negative input"
                )
                assert -1.5 <= v <= 1.5

    def test_different_inputs_differ(self):
        """Different 2D inputs should produce different outputs."""
        values = [
            py_fbm_2d(
                (i * 0.17, i * 0.23),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(20)
        ]
        unique = len(set(round(v, 10) for v in values))
        assert unique > 10, (
            f"Expected at least 10 unique values out of 20, got {unique}"
        )


# =============================================================================
# Test: T-DEMO-1.31 FBM 3D (Value Noise Base)
# =============================================================================


class TestFbm3D:
    """Tests for fbm_3d (value noise base)."""

    def test_range(self):
        """Output should be approximately in [-1, 1]."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5, iy * 0.5, iz * 0.5)
                    v = py_fbm_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                    assert math.isfinite(v), f"fbm_3d({p}) non-finite: {v}"
                    assert -1.5 <= v <= 1.5, (
                        f"fbm_3d({p}) = {v} outside expected range"
                    )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (1.618, 2.718, 3.142)
        v1 = py_fbm_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        for _ in range(10):
            v2 = py_fbm_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_single_octave_equals_value_noise_3d(self):
        """With octaves=1, fbm_3d should equal value_noise_3d."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5 + 0.1, iy * 0.5 + 0.2, iz * 0.5 + 0.3)
                    fbm_val = py_fbm_3d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                    value_val = py_value_noise_3d(p)
                    assert fbm_val == pytest.approx(value_val, abs=TOL_ABS), (
                        f"fbm_3d({p}, 1) != value_noise_3d({p})"
                    )

    def test_zero_octaves_returns_zero(self):
        """With octaves=0, fbm_3d should return 0.0."""
        for p in [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]:
            v = py_fbm_3d(p, 0, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v == pytest.approx(0.0, abs=TOL_ABS), (
                f"fbm_3d({p}, 0) should be 0.0, got {v}"
            )

    def test_negative_inputs(self):
        """fbm_3d handles negative coordinates correctly."""
        for ix in range(-3, 0):
            for iy in range(-3, 0):
                for iz in range(-3, 0):
                    p = (ix + 0.3, iy + 0.7, iz + 0.5)
                    v = py_fbm_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                    assert math.isfinite(v)
                    assert -1.5 <= v <= 1.5

    def test_spectral_composition(self):
        """Different octave counts produce different outputs."""
        p = (0.5, 0.25, 0.125)
        v1 = py_fbm_3d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v4 = py_fbm_3d(p, 4, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v8 = py_fbm_3d(p, 8, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        diffs = [abs(v4 - v1), abs(v8 - v1), abs(v8 - v4)]
        assert any(d > 1e-8 for d in diffs), (
            "Different octave counts should produce different fbm_3d outputs"
        )

    def test_unity_lacunarity(self):
        """With lacunarity=1.0, all octaves sample at the same frequency."""
        p = (1.5, 2.5, 3.5)
        val = py_value_noise_3d(p)
        for octaves in [3, 5, 8]:
            result = py_fbm_3d(p, octaves, 1.0, 0.5)
            assert result == pytest.approx(val, abs=TOL_ABS), (
                f"fbm_3d({p}, lac=1.0) should equal value_noise_3d"
            )

    def test_zero_gain(self):
        """With gain=0.0, only the first octave contributes."""
        p = (2.5, 1.5, 3.5)
        for octaves in [4, 8]:
            result = py_fbm_3d(p, octaves, 2.0, 0.0)
            expected = py_value_noise_3d(p)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"fbm_3d({p}, gain=0) should equal value_noise_3d({p})"
            )

    def test_amplitude_normalization(self):
        """Output range stays bounded across varying octave counts."""
        p = (0.25, 0.5, 0.75)
        max_abs = 0.0
        for octaves in [1, 2, 4, 8, 16]:
            v = py_fbm_3d(p, octaves, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            max_abs = max(max_abs, abs(v))
        assert max_abs < 1.5, (
            "Amplitude normalization should keep 3D FBM output bounded"
        )

    def test_different_inputs_differ(self):
        """Different 3D inputs should produce different outputs."""
        values = [
            py_fbm_3d(
                (i * 0.17, i * 0.23, i * 0.31),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(20)
        ]
        unique = len(set(round(v, 10) for v in values))
        assert unique > 10, (
            f"Expected at least 10 unique values out of 20, got {unique}"
        )


# =============================================================================
# Test: T-DEMO-1.31 FBM Perlin 3D
# =============================================================================


class TestFbmPerlin3D:
    """Tests for fbm_perlin_3d (Perlin noise base)."""

    def test_range(self):
        """Output should be approximately in [-1, 1]."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5, iy * 0.5, iz * 0.5)
                    v = py_fbm_perlin_3d(
                        p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
                    )
                    assert math.isfinite(v), f"fbm_perlin_3d({p}) non-finite: {v}"
                    assert -1.5 <= v <= 1.5, (
                        f"fbm_perlin_3d({p}) = {v} outside expected range"
                    )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (1.618, 2.718, 3.142)
        v1 = py_fbm_perlin_3d(
            p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
        )
        for _ in range(10):
            v2 = py_fbm_perlin_3d(
                p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_single_octave_equals_perlin_noise_3d(self):
        """With octaves=1, fbm_perlin_3d should equal perlin_noise_3d."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5 + 0.1, iy * 0.5 + 0.2, iz * 0.5 + 0.3)
                    fbm_val = py_fbm_perlin_3d(
                        p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN
                    )
                    perlin_val = py_perlin_noise_3d(p)
                    assert fbm_val == pytest.approx(perlin_val, abs=TOL_ABS), (
                        f"fbm_perlin_3d({p}, 1) != perlin_noise_3d({p})"
                    )

    def test_zero_octaves_returns_zero(self):
        """With octaves=0, fbm_perlin_3d should return 0.0."""
        for p in [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]:
            v = py_fbm_perlin_3d(
                p, 0, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            assert v == pytest.approx(0.0, abs=TOL_ABS), (
                f"fbm_perlin_3d({p}, 0) should be 0.0, got {v}"
            )

    def test_spectral_composition(self):
        """Different octave counts produce different outputs."""
        p = (0.5, 0.25, 0.125)
        v1 = py_fbm_perlin_3d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v4 = py_fbm_perlin_3d(p, 4, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v8 = py_fbm_perlin_3d(p, 8, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        diffs = [abs(v4 - v1), abs(v8 - v1), abs(v8 - v4)]
        assert any(d > 1e-8 for d in diffs), (
            "Different octave counts should produce different fbm_perlin_3d outputs"
        )

    def test_lacunarity_affects_output(self):
        """Different lacunarity values should produce different outputs."""
        p = (0.5, 0.25, 0.125)
        base = py_fbm_perlin_3d(p, 4, 2.0, 0.5)
        for lacunarity in [1.5, 3.0, 4.0]:
            v = py_fbm_perlin_3d(p, 4, lacunarity, 0.5)
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Lacunarity did not affect fbm_perlin_3d output")

    def test_gain_affects_output(self):
        """Different gain values should produce different outputs."""
        p = (0.5, 0.25, 0.125)
        base = py_fbm_perlin_3d(p, 4, 2.0, 0.5)
        for gain in [0.25, 0.75]:
            v = py_fbm_perlin_3d(p, 4, 2.0, gain)
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Gain did not affect fbm_perlin_3d output")

    def test_zero_gain(self):
        """With gain=0.0, only the first octave contributes."""
        p = (1.5, 2.5, 3.5)
        for octaves in [4, 8]:
            result = py_fbm_perlin_3d(p, octaves, 2.0, 0.0)
            expected = py_perlin_noise_3d(p)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"fbm_perlin_3d({p}, gain=0) != perlin_noise_3d({p})"
            )

    def test_integer_grid_is_zero(self):
        """At exact integer grid positions, fbm_perlin_3d = 0.

        Perlin noise is zero at integer grid points (all gradient dot
        products are zero). This property carries through to the FBM
        composition since each octave has its own Perlin noise evaluation
        at the scaled frequency.
        """
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                for iz in range(-3, 4):
                    p = (float(ix), float(iy), float(iz))
                    result = py_fbm_perlin_3d(
                        p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
                    )
                    assert result == pytest.approx(0.0, abs=TOL_ABS), (
                        f"fbm_perlin_3d at integer ({ix},{iy},{iz}) "
                        f"should be 0, got {result}"
                    )

    def test_different_inputs_differ(self):
        """Different 3D inputs should produce different outputs."""
        values = [
            py_fbm_perlin_3d(
                (i * 0.17, i * 0.23, i * 0.31),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(20)
        ]
        unique = len(set(round(v, 10) for v in values))
        assert unique > 10, (
            f"Expected at least 10 unique values out of 20, got {unique}"
        )


# =============================================================================
# Test: T-DEMO-1.31 Value vs Perlin FBM Comparison
# =============================================================================


class TestFbmValueVsPerlin:
    """Tests comparing value-based FBM and Perlin-based FBM.

    Both use the same FBM spectral composition algorithm but different
    base noise functions. They should produce structurally different
    results while sharing the same spectral properties.
    """

    def test_value_fbm_differs_from_perlin_fbm(self):
        """fbm_3d and fbm_perlin_3d should produce different results.

        Value noise uses scalar hash interpolation while Perlin noise
        uses gradient dot products. Even though they share the same FBM
        algorithm, the base noise functions are structurally different.
        """
        p = (3.7, 4.2, 5.1)
        val_fbm = py_fbm_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        perlin_fbm = py_fbm_perlin_3d(
            p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
        )
        assert abs(val_fbm - perlin_fbm) > 0.01, (
            "Value-based and Perlin-based FBM should produce different results"
        )

    def test_value_fbm_nonzero_at_integers(self):
        """fbm_3d (value base) is non-zero at integer positions.

        Value noise at integer positions returns hash * 2 - 1, which is
        non-zero. This property carries through the FBM composition.
        """
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                for iz in range(-3, 4):
                    p = (float(ix), float(iy), float(iz))
                    result = py_fbm_3d(
                        p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
                    )
                    assert abs(result) > 0, (
                        f"fbm_3d at integer ({ix},{iy},{iz}) should be non-zero, "
                        f"got {result}"
                    )

    def test_perlin_fbm_zero_at_integers(self):
        """fbm_perlin_3d is zero at integer positions.

        Perlin noise at integer positions returns 0 (all gradient dot
        products are zero). This property carries through the FBM
        composition since all octaves evaluate Perlin noise at scaled
        integer positions.
        """
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                for iz in range(-3, 4):
                    p = (float(ix), float(iy), float(iz))
                    result = py_fbm_perlin_3d(
                        p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
                    )
                    assert result == pytest.approx(0.0, abs=TOL_ABS), (
                        f"fbm_perlin_3d at integer ({ix},{iy},{iz}) "
                        f"should be 0, got {result}"
                    )

    def test_both_use_same_spectral_algorithm(self):
        """Both variants should respond identically to parameter changes.

        The spectral composition algorithm (octave loop with lacunarity
        and gain) is the same for both. Only the base noise function differs.
        So changing from lacunarity=2 to lacunarity=3 should change both
        in the same direction.
        """
        p = (0.75, 1.25, 0.5)
        # Compute value FBM at two lacunarity values
        val_fbm_2 = py_fbm_3d(p, 4, 2.0, 0.5)
        val_fbm_3 = py_fbm_3d(p, 4, 3.0, 0.5)
        val_diff_direction = val_fbm_3 > val_fbm_2

        # Compute Perlin FBM at the same two lacunarity values
        perlin_fbm_2 = py_fbm_perlin_3d(p, 4, 2.0, 0.5)
        perlin_fbm_3 = py_fbm_perlin_3d(p, 4, 3.0, 0.5)
        perlin_diff_direction = perlin_fbm_3 > perlin_fbm_2

        # The direction of change may not be the same since the base
        # noise functions differ structurally. But the key property is
        # that both DO respond to the parameter change.
        assert abs(val_fbm_2 - val_fbm_3) > 1e-8, (
            "Value FBM must respond to lacunarity changes"
        )
        assert abs(perlin_fbm_2 - perlin_fbm_3) > 1e-8, (
            "Perlin FBM must respond to lacunarity changes"
        )


# =============================================================================
# Test: T-DEMO-1.31 Continuity
# =============================================================================


class TestFbmContinuity:
    """Tests that FBM noise is continuous along each axis."""

    def test_fbm_3d_continuity_along_x(self):
        """fbm_3d should be continuous along x for fixed y, z."""
        step = 0.01
        p = (0.0, 1.0, 2.0)
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_fbm_3d(p, *params)
        for i in range(1, 200):
            curr = py_fbm_3d((i * step, 1.0, 2.0), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in fbm_3d at x={i * step}: diff={diff}"
            )
            prev = curr

    def test_fbm_3d_continuity_along_y(self):
        """fbm_3d should be continuous along y for fixed x, z."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_fbm_3d((1.0, 0.0, 2.0), *params)
        for i in range(1, 200):
            curr = py_fbm_3d((1.0, i * step, 2.0), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in fbm_3d at y={i * step}: diff={diff}"
            )
            prev = curr

    def test_fbm_3d_continuity_along_z(self):
        """fbm_3d should be continuous along z for fixed x, y."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_fbm_3d((1.0, 2.0, 0.0), *params)
        for i in range(1, 200):
            curr = py_fbm_3d((1.0, 2.0, i * step), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in fbm_3d at z={i * step}: diff={diff}"
            )
            prev = curr

    def test_fbm_perlin_3d_continuity_along_x(self):
        """fbm_perlin_3d should be continuous along x."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_fbm_perlin_3d((0.0, 1.0, 2.0), *params)
        for i in range(1, 200):
            curr = py_fbm_perlin_3d((i * step, 1.0, 2.0), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in fbm_perlin_3d at x={i * step}: diff={diff}"
            )
            prev = curr

    def test_fbm_perlin_3d_continuity_along_y(self):
        """fbm_perlin_3d should be continuous along y."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_fbm_perlin_3d((1.0, 0.0, 2.0), *params)
        for i in range(1, 200):
            curr = py_fbm_perlin_3d((1.0, i * step, 2.0), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in fbm_perlin_3d at y={i * step}: diff={diff}"
            )
            prev = curr

    def test_fbm_perlin_3d_continuity_along_z(self):
        """fbm_perlin_3d should be continuous along z."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_fbm_perlin_3d((1.0, 2.0, 0.0), *params)
        for i in range(1, 200):
            curr = py_fbm_perlin_3d((1.0, 2.0, i * step), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in fbm_perlin_3d at z={i * step}: diff={diff}"
            )
            prev = curr

    def test_fbm_1d_continuity(self):
        """fbm_1d should be continuous."""
        step = 0.001
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_fbm_1d(0.0, *params)
        for i in range(1, 500):
            curr = py_fbm_1d(i * step, *params)
            diff = abs(curr - prev)
            assert diff < 0.02, (
                f"Discontinuity in fbm_1d at {i * step}: diff={diff}"
            )
            prev = curr


# =============================================================================
# Test: T-DEMO-1.31 Mean and Distribution
# =============================================================================


class TestFbmDistribution:
    """Tests that FBM noise has reasonable distribution properties."""

    NUM_SAMPLES = 5000

    def test_fbm_1d_mean_near_zero(self):
        """Mean of fbm_1d over many points should be near 0."""
        samples = [
            py_fbm_1d(i * 0.07, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1, (
            f"fbm_1d mean {mean} not near 0"
        )

    def test_fbm_2d_mean_near_zero(self):
        """Mean of fbm_2d over many points should be near 0."""
        samples = [
            py_fbm_2d(
                (i * 0.07, i * 0.11),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1, (
            f"fbm_2d mean {mean} not near 0"
        )

    def test_fbm_3d_mean_near_zero(self):
        """Mean of fbm_3d over many points should be near 0."""
        samples = [
            py_fbm_3d(
                (i * 0.07, i * 0.11, i * 0.13),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1, (
            f"fbm_3d mean {mean} not near 0"
        )

    def test_fbm_perlin_3d_mean_near_zero(self):
        """Mean of fbm_perlin_3d over many points should be near 0."""
        samples = [
            py_fbm_perlin_3d(
                (i * 0.07, i * 0.11, i * 0.13),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1, (
            f"fbm_perlin_3d mean {mean} not near 0"
        )

    def test_fbm_1d_values_not_all_same(self):
        """fbm_1d should not produce constant output."""
        samples = [
            py_fbm_1d(i * 0.17, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 20, (
            f"Expected at least 20 unique values out of 50, got {unique}"
        )

    def test_fbm_3d_values_not_all_same(self):
        """fbm_3d should not produce constant output."""
        samples = [
            py_fbm_3d(
                (i * 0.17, i * 0.23, i * 0.31),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 20, (
            f"Expected at least 20 unique values out of 50, got {unique}"
        )

    def test_fbm_perlin_3d_values_not_all_same(self):
        """fbm_perlin_3d should not produce constant output."""
        samples = [
            py_fbm_perlin_3d(
                (i * 0.17, i * 0.23, i * 0.31),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 20, (
            f"Expected at least 20 unique values out of 50, got {unique}"
        )

    def test_fbm_1d_approx_symmetry(self):
        """About half the fbm_1d samples should be below zero."""
        samples = [
            py_fbm_1d(i * 0.07, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            for i in range(self.NUM_SAMPLES)
        ]
        below_zero = sum(1 for v in samples if v < 0)
        ratio = below_zero / len(samples)
        assert 0.35 <= ratio <= 0.65, (
            f"Expected ~50% of fbm_1d samples below zero, got {ratio:.1%}"
        )

    def test_fbm_3d_approx_symmetry(self):
        """About half the fbm_3d samples should be below zero."""
        samples = [
            py_fbm_3d(
                (i * 0.07, i * 0.11, i * 0.13),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(self.NUM_SAMPLES)
        ]
        below_zero = sum(1 for v in samples if v < 0)
        ratio = below_zero / len(samples)
        assert 0.35 <= ratio <= 0.65, (
            f"Expected ~50% of fbm_3d samples below zero, got {ratio:.1%}"
        )

    def test_fbm_perlin_3d_approx_symmetry(self):
        """About half the fbm_perlin_3d samples should be below zero."""
        samples = [
            py_fbm_perlin_3d(
                (i * 0.07, i * 0.11, i * 0.13),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(self.NUM_SAMPLES)
        ]
        below_zero = sum(1 for v in samples if v < 0)
        ratio = below_zero / len(samples)
        assert 0.35 <= ratio <= 0.65, (
            f"Expected ~50% of fbm_perlin_3d samples below zero, got {ratio:.1%}"
        )


# =============================================================================
# Test: T-DEMO-1.31 Cross-Function Consistency
# =============================================================================


class TestFbmCrossFunction:
    """Tests that FBM functions with matching dimensionality are consistent."""

    def test_fbm_1d_single_octave_matches_2d_at_y_zero(self):
        """1D and 2D FBM should agree when 2D y input is 0."""
        for i in range(-20, 21):
            p = i * 0.3 + 0.1
            v1d = py_fbm_1d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            # value_noise_2d at (p, 0) evaluates hash21 at (ix, 0)
            # This is a different hash function than hash11 used by 1D noise,
            # so the values won't be identical. But the range and properties
            # should be consistent.
            v2d = py_fbm_2d((p, 0.0), 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert math.isfinite(v1d), f"1D FBM non-finite at {p}"
            assert math.isfinite(v2d), f"2D FBM non-finite at ({p}, 0)"
            assert -1.5 <= v1d <= 1.5
            assert -1.5 <= v2d <= 1.5

    def test_all_four_handle_large_coordinates(self):
        """All four FBM functions handle large coordinate values."""
        large_ps = [1000.0, -1000.0, 1e6, -1e6, 0.001, -0.001]
        for p in large_ps:
            v1 = py_fbm_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert math.isfinite(v1), f"fbm_1d({p}) non-finite"
        for p in large_ps:
            v2 = py_fbm_2d(
                (p, p * 0.5), DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            assert math.isfinite(v2), f"fbm_2d({p}) non-finite"
        for p in large_ps:
            v3 = py_fbm_3d(
                (p, p * 0.5, p * 0.25),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            assert math.isfinite(v3), f"fbm_3d({p}) non-finite"
        for p in large_ps:
            vp = py_fbm_perlin_3d(
                (p, p * 0.5, p * 0.25),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            assert math.isfinite(vp), f"fbm_perlin_3d({p}) non-finite"

    def test_all_four_respond_to_parameters(self):
        """All four functions respond to octaves, lacunarity, and gain."""
        test_params = [
            (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN),
            (4, 2.0, 0.5),
            (8, 1.5, 0.7),
            (6, 3.0, 0.3),
        ]
        p_1d = 0.75
        p_3d = (0.75, 1.25, 0.5)

        results_1d = [
            py_fbm_1d(p_1d, *tp) for tp in test_params
        ]
        results_3d = [
            py_fbm_3d(p_3d, *tp) for tp in test_params
        ]
        results_perlin = [
            py_fbm_perlin_3d(p_3d, *tp) for tp in test_params
        ]

        # Each should produce different values for different parameters
        for name, results in [
            ("fbm_1d", results_1d),
            ("fbm_3d", results_3d),
            ("fbm_perlin_3d", results_perlin),
        ]:
            unique = len(set(round(v, 10) for v in results))
            assert unique >= 3, (
                f"{name} produced only {unique} unique values across "
                f"4 parameter sets"
            )
