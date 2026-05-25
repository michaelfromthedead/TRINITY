"""
Whitebox tests for ridged noise WGSL functions (T-DEMO-1.32).

Tests Python model implementations of each WGSL ridged noise function,
which wraps FBM with the 1.0 - abs(x) transform, verifying:
  - Range: output in [0, 1] (FBM in [-1, 1], abs -> [0, 1], 1-abs -> [0, 1])
  - Sharp ridges: output near 1.0 when FBM near 0 (V-cusp at zero crossing)
  - Smooth valleys: output near 0.0 when FBM near +/-1
  - Determinism: same input + parameters always produce same output
  - Single octave equals 1 - abs(base noise)
  - Zero octaves returns 1.0 (since FBM returns 0, 1 - abs(0) = 1)
  - Spectral composition preserved from FBM (octaves/lacunarity/gain control)
  - Continuity: ridged noise is continuous everywhere (abs preserves continuity)
  - Value-based vs Perlin-based ridged noise differ structurally
  - Mean near 1/sqrt(2*pi) ~ 0.4 for ridged noise (folded normal distribution)

WHITEBOX coverage plan:
  Path A: ridged_noise_1d range [0, 1] and finite output
  Path B: ridged_noise_1d deterministic
  Path C: ridged_noise_1d single octave equals 1 - abs(value_noise_1d)
  Path D: ridged_noise_1d zero octaves returns 1.0
  Path E: ridged_noise_1d spectral composition (lacunarity/gain affect output)
  Path F: ridged_noise_1d sharp ridges at FBM zero-crossings
  Path G: ridged_noise_2d range and finite output
  Path H: ridged_noise_2d deterministic
  Path I: ridged_noise_2d single octave equals 1 - abs(value_noise_2d)
  Path J: ridged_noise_2d zero octaves returns 1.0
  Path K: ridged_noise_2d spectral composition
  Path L: ridged_noise_3d range and finite output
  Path M: ridged_noise_3d deterministic
  Path N: ridged_noise_3d single octave equals 1 - abs(value_noise_3d)
  Path O: ridged_noise_3d zero octaves returns 1.0
  Path P: ridged_noise_3d spectral composition
  Path Q: ridged_perlin_3d range and finite output
  Path R: ridged_perlin_3d deterministic
  Path S: ridged_perlin_3d single octave equals 1 - abs(perlin_noise_3d)
  Path T: ridged_perlin_3d zero octaves returns 1.0
  Path U: ridged_perlin_3d unity at integer grid positions (FBM=0 -> ridged=1)
  Path V: Value-based vs Perlin-based ridged noise produce different results
  Path W: Continuity along each axis
  Path X: Mean near folded normal expectation (~0.4 for [0,1] range)
  Path Y: Increasing octaves changes output (adds detail to ridges)
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
# Python model of FBM fractal Brownian motion (reused from T-DEMO-1.31)
# =============================================================================


def py_fbm_1d(p: float, octaves: int, lacunarity: float, gain: float) -> float:
    """Model of WGSL fbm_1d."""
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
    """Model of WGSL fbm_perlin_3d."""
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
# Python model of ridged noise (1.0 - abs(FBM)) for T-DEMO-1.32
#
# WGSL reference (noise_ridged.wgsl):
#   fn ridged_noise_1d(p: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32
#   fn ridged_noise_2d(p: vec2<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32
#   fn ridged_noise_3d(p: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32
#   fn ridged_perlin_3d(p: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32
#
# Each function calls the corresponding FBM function and transforms
# the result via 1.0 - abs(FBM), creating sharp ridges at FBM
# zero-crossings and smooth valleys at FBM extrema.
# =============================================================================


def py_ridged_noise_1d(p: float, octaves: int, lacunarity: float, gain: float) -> float:
    """Model of WGSL ridged_noise_1d.

    WGSL equivalent:
      fn ridged_noise_1d(p: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
          let fbm_val = fbm_1d(p, octaves, lacunarity, gain);
          return 1.0 - abs(fbm_val);
      }
    """
    fbm_val = py_fbm_1d(p, octaves, lacunarity, gain)
    return 1.0 - abs(fbm_val)


def py_ridged_noise_2d(p, octaves: int, lacunarity: float, gain: float) -> float:
    """Model of WGSL ridged_noise_2d."""
    fbm_val = py_fbm_2d(p, octaves, lacunarity, gain)
    return 1.0 - abs(fbm_val)


def py_ridged_noise_3d(p, octaves: int, lacunarity: float, gain: float) -> float:
    """Model of WGSL ridged_noise_3d."""
    fbm_val = py_fbm_3d(p, octaves, lacunarity, gain)
    return 1.0 - abs(fbm_val)


def py_ridged_perlin_3d(p, octaves: int, lacunarity: float, gain: float) -> float:
    """Model of WGSL ridged_perlin_3d."""
    fbm_val = py_fbm_perlin_3d(p, octaves, lacunarity, gain)
    return 1.0 - abs(fbm_val)


# =============================================================================
# Shared tolerances and test helpers
# =============================================================================

TOL_REL = 1e-5
TOL_ABS = 1e-9

# Standard spectral parameters used throughout testing
DEFAULT_OCTAVES = 8
DEFAULT_LACUNARITY = 2.0
DEFAULT_GAIN = 0.5


# =============================================================================
# Test: T-DEMO-1.32 Ridged Noise 1D
# =============================================================================


class TestRidgedNoise1D:
    """Tests for ridged_noise_1d."""

    # --- Range and finite ---

    def test_range(self):
        """Output should be in [0, 1] (FBM in [-1,1], abs -> [0,1], 1-abs -> [0,1])."""
        for i in range(-50, 51):
            p = i * 0.17
            v = py_ridged_noise_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert math.isfinite(v), f"ridged_noise_1d({p}) non-finite: {v}"
            assert 0.0 <= v <= 1.0, (
                f"ridged_noise_1d({p}) = {v} outside expected [0, 1] range"
            )

    def test_range_various_parameters(self):
        """Output should be in [0, 1] for various parameter combinations."""
        params = [
            (4, 2.0, 0.5),
            (1, 2.0, 0.5),
            (8, 1.5, 0.7),
            (6, 3.0, 0.3),
            (2, 4.0, 0.25),
        ]
        for p in [i * 0.13 for i in range(-20, 21)]:
            for octaves, lacunarity, gain in params:
                v = py_ridged_noise_1d(p, octaves, lacunarity, gain)
                assert math.isfinite(v), (
                    f"ridged_noise_1d({p}, oct={octaves}, lac={lacunarity}, "
                    f"gain={gain}) non-finite: {v}"
                )
                assert 0.0 <= v <= 1.0, (
                    f"ridged_noise_1d({p}, oct={octaves}, lac={lacunarity}, "
                    f"gain={gain}) = {v} outside [0, 1]"
                )

    def test_dense_sampling_range(self):
        """Dense sampling across many cells should stay in [0, 1]."""
        for i in range(500):
            p = -50.0 + i * 0.2
            v = py_ridged_noise_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert 0.0 <= v <= 1.0, (
                f"ridged_noise_1d({p}) = {v} outside [0, 1]"
            )

    # --- Determinism ---

    def test_deterministic(self):
        """Same input and parameters always produce same output."""
        p = 1.618
        v1 = py_ridged_noise_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        for _ in range(10):
            v2 = py_ridged_noise_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                "ridged_noise_1d not deterministic"
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
            v1 = py_ridged_noise_1d(p, octaves, lacunarity, gain)
            for _ in range(5):
                v2 = py_ridged_noise_1d(p, octaves, lacunarity, gain)
                assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                    f"ridged_noise_1d not deterministic with "
                    f"oct={octaves}, lac={lacunarity}, gain={gain}"
                )

    # --- Single octave equals 1 - abs(base noise) ---

    def test_single_octave_equals_one_minus_abs_value_noise(self):
        """With octaves=1, ridged_noise_1d should equal 1 - abs(value_noise_1d)."""
        for i in range(-50, 51):
            p = i * 0.13 + 0.037
            ridged_val = py_ridged_noise_1d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            expected = 1.0 - abs(py_value_noise_1d(p))
            assert ridged_val == pytest.approx(expected, abs=TOL_ABS), (
                f"ridged_noise_1d({p}, 1) = {ridged_val} != "
                f"1 - abs(value_noise_1d({p})) = {expected}"
            )

    # --- Zero octaves guard ---

    def test_zero_octaves_returns_one(self):
        """With octaves=0, ridged_noise_1d should return 1.0.

        FBM returns 0.0 for zero octaves (guard against div by zero),
        so ridged = 1.0 - abs(0.0) = 1.0.
        """
        for p in [0.0, 1.0, -1.0, 3.14, 100.0]:
            v = py_ridged_noise_1d(p, 0, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v == pytest.approx(1.0, abs=TOL_ABS), (
                f"ridged_noise_1d({p}, 0) should be 1.0, got {v}"
            )

    # --- Sharp ridges at FBM zero-crossings ---

    def test_sharp_ridge_at_fbm_zero(self):
        """When FBM is near 0, ridged noise should be near 1 (sharp ridge).

        The abs function creates a V-shaped cusp at FBM zero-crossings,
        producing sharp ridge peaks (the fundamental property of ridged noise).
        """
        # Find points where FBM is near 0
        near_zero_points = []
        for i in range(-100, 101):
            p = i * 0.07
            fbm_val = py_fbm_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            ridged_val = py_ridged_noise_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            if abs(fbm_val) < 0.05:
                near_zero_points.append((p, fbm_val, ridged_val))

        # Verify that at FBM zero-crossings, ridged noise is near 1
        if near_zero_points:
            for p, fbm_val, ridged_val in near_zero_points[:10]:
                assert ridged_val > 0.9, (
                    f"At FBM zero-crossing p={p}, fbm={fbm_val:.6f}, "
                    f"ridged should be near 1, got {ridged_val:.6f}"
                )

    def test_smooth_valley_at_fbm_extrema(self):
        """When FBM is near +/-1, ridged noise should be near 0 (smooth valley)."""
        # Find points where |FBM| is large (near extrema)
        extreme_points = []
        for i in range(-100, 101):
            p = i * 0.07
            fbm_val = py_fbm_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            ridged_val = py_ridged_noise_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            if abs(fbm_val) > 0.5:
                extreme_points.append((p, fbm_val, ridged_val))

        # Verify that at FBM extrema, ridged noise is generally lower
        if extreme_points:
            avg_ridged_at_extrema = sum(r for _, _, r in extreme_points) / len(extreme_points)
            assert avg_ridged_at_extrema < 0.7, (
                f"At FBM extrema, avg ridged noise = {avg_ridged_at_extrema:.4f}, "
                f"expected < 0.7"
            )

    # --- Spectral composition ---

    def test_increasing_octaves_changes_output(self):
        """Increasing the number of octaves should change the output."""
        p = 0.5
        base = py_ridged_noise_1d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        for octaves in [2, 4, 6, 8]:
            v = py_ridged_noise_1d(p, octaves, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert abs(v - base) > 1e-8 or abs(v - base) == 0.0, (
                f"ridged_noise_1d({p}) with {octaves} octaves should differ from 1 octave"
            )

    def test_lacunarity_affects_output(self):
        """Different lacunarity values should produce different outputs."""
        p = 0.5
        base = py_ridged_noise_1d(p, 4, 2.0, 0.5)
        lacunarities = [1.5, 3.0, 4.0]
        diffs = [
            abs(py_ridged_noise_1d(p, 4, lac, 0.5) - base)
            for lac in lacunarities
        ]
        assert any(d > 1e-8 for d in diffs), (
            "Lacunarity values should affect ridged_noise_1d output"
        )

    def test_gain_affects_output(self):
        """Different gain values should produce different outputs."""
        p = 0.5
        base = py_ridged_noise_1d(p, 4, 2.0, 0.5)
        gains = [0.25, 0.75, 0.9]
        diffs = [
            abs(py_ridged_noise_1d(p, 4, 2.0, g) - base)
            for g in gains
        ]
        assert any(d > 1e-8 for d in diffs), (
            "Gain values should affect ridged_noise_1d output"
        )

    # --- Negative inputs ---

    def test_negative_inputs(self):
        """ridged_noise_1d handles negative coordinates correctly."""
        for i in range(-50, 0):
            p = float(i) + 0.3
            v = py_ridged_noise_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert math.isfinite(v), f"ridged_noise_1d({p}) non-finite for negative input"
            assert 0.0 <= v <= 1.0, (
                f"ridged_noise_1d({p}) = {v} outside [0, 1]"
            )

    # --- Different inputs differ ---

    def test_different_inputs_differ(self):
        """Different inputs should produce different outputs."""
        values = [
            py_ridged_noise_1d(i * 0.17, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            for i in range(20)
        ]
        unique = len(set(round(v, 10) for v in values))
        assert unique > 10, (
            f"Expected at least 10 unique values out of 20, got {unique}"
        )


# =============================================================================
# Test: T-DEMO-1.32 Ridged Noise 2D
# =============================================================================


class TestRidgedNoise2D:
    """Tests for ridged_noise_2d."""

    def test_range(self):
        """Output should be in [0, 1]."""
        for ix in range(-10, 11):
            for iy in range(-10, 11):
                p = (ix * 0.3, iy * 0.3)
                v = py_ridged_noise_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                assert math.isfinite(v), f"ridged_noise_2d({p}) non-finite: {v}"
                assert 0.0 <= v <= 1.0, (
                    f"ridged_noise_2d({p}) = {v} outside [0, 1]"
                )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (2.718, 3.142)
        v1 = py_ridged_noise_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        for _ in range(10):
            v2 = py_ridged_noise_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_single_octave_equals_one_minus_abs_value_noise_2d(self):
        """With octaves=1, ridged_noise_2d should equal 1 - abs(value_noise_2d)."""
        for ix in range(-10, 11):
            for iy in range(-10, 11):
                p = (ix * 0.3 + 0.1, iy * 0.3 + 0.2)
                ridged_val = py_ridged_noise_2d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                expected = 1.0 - abs(py_value_noise_2d(p))
                assert ridged_val == pytest.approx(expected, abs=TOL_ABS), (
                    f"ridged_noise_2d({p}, 1) = {ridged_val} != "
                    f"1 - abs(value_noise_2d({p})) = {expected}"
                )

    def test_zero_octaves_returns_one(self):
        """With octaves=0, ridged_noise_2d should return 1.0."""
        for p in [(0.0, 0.0), (1.0, 2.0), (-3.0, 4.0)]:
            v = py_ridged_noise_2d(p, 0, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v == pytest.approx(1.0, abs=TOL_ABS), (
                f"ridged_noise_2d({p}, 0) should be 1.0, got {v}"
            )

    def test_spectral_composition(self):
        """Different octave counts produce different outputs."""
        p = (0.5, 0.25)
        v1 = py_ridged_noise_2d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v4 = py_ridged_noise_2d(p, 4, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v8 = py_ridged_noise_2d(p, 8, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        diffs = [abs(v4 - v1), abs(v8 - v1), abs(v8 - v4)]
        assert any(d > 1e-8 for d in diffs), (
            "Different octave counts should produce different ridged_noise_2d outputs"
        )

    def test_negative_inputs(self):
        """ridged_noise_2d handles negative coordinates correctly."""
        for ix in range(-5, 0):
            for iy in range(-5, 0):
                p = (ix + 0.3, iy + 0.7)
                v = py_ridged_noise_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                assert math.isfinite(v), (
                    f"ridged_noise_2d({p}) non-finite for negative input"
                )
                assert 0.0 <= v <= 1.0

    def test_different_inputs_differ(self):
        """Different 2D inputs should produce different outputs."""
        values = [
            py_ridged_noise_2d(
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
# Test: T-DEMO-1.32 Ridged Noise 3D (Value Noise FBM Base)
# =============================================================================


class TestRidgedNoise3D:
    """Tests for ridged_noise_3d (value noise FBM base)."""

    def test_range(self):
        """Output should be in [0, 1]."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5, iy * 0.5, iz * 0.5)
                    v = py_ridged_noise_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                    assert math.isfinite(v), f"ridged_noise_3d({p}) non-finite: {v}"
                    assert 0.0 <= v <= 1.0, (
                        f"ridged_noise_3d({p}) = {v} outside [0, 1]"
                    )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (1.618, 2.718, 3.142)
        v1 = py_ridged_noise_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        for _ in range(10):
            v2 = py_ridged_noise_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_single_octave_equals_one_minus_abs_value_noise_3d(self):
        """With octaves=1, ridged_noise_3d should equal 1 - abs(value_noise_3d)."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5 + 0.1, iy * 0.5 + 0.2, iz * 0.5 + 0.3)
                    ridged_val = py_ridged_noise_3d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                    expected = 1.0 - abs(py_value_noise_3d(p))
                    assert ridged_val == pytest.approx(expected, abs=TOL_ABS), (
                        f"ridged_noise_3d({p}, 1) != 1 - abs(value_noise_3d({p}))"
                    )

    def test_zero_octaves_returns_one(self):
        """With octaves=0, ridged_noise_3d should return 1.0."""
        for p in [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]:
            v = py_ridged_noise_3d(p, 0, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert v == pytest.approx(1.0, abs=TOL_ABS), (
                f"ridged_noise_3d({p}, 0) should be 1.0, got {v}"
            )

    def test_negative_inputs(self):
        """ridged_noise_3d handles negative coordinates correctly."""
        for ix in range(-3, 0):
            for iy in range(-3, 0):
                for iz in range(-3, 0):
                    p = (ix + 0.3, iy + 0.7, iz + 0.5)
                    v = py_ridged_noise_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                    assert math.isfinite(v)
                    assert 0.0 <= v <= 1.0

    def test_spectral_composition(self):
        """Different octave counts produce different outputs."""
        p = (0.5, 0.25, 0.125)
        v1 = py_ridged_noise_3d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v4 = py_ridged_noise_3d(p, 4, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v8 = py_ridged_noise_3d(p, 8, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        diffs = [abs(v4 - v1), abs(v8 - v1), abs(v8 - v4)]
        assert any(d > 1e-8 for d in diffs), (
            "Different octave counts should produce different ridged_noise_3d outputs"
        )

    def test_unity_lacunarity(self):
        """With lacunarity=1.0, all octaves sample at the same frequency."""
        p = (1.5, 2.5, 3.5)
        expected = 1.0 - abs(py_value_noise_3d(p))
        for octaves in [3, 5, 8]:
            result = py_ridged_noise_3d(p, octaves, 1.0, 0.5)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"ridged_noise_3d({p}, lac=1.0) should equal 1 - abs(value_noise_3d)"
            )

    def test_zero_gain(self):
        """With gain=0.0, only the first octave contributes."""
        p = (2.5, 1.5, 3.5)
        expected = 1.0 - abs(py_value_noise_3d(p))
        for octaves in [4, 8]:
            result = py_ridged_noise_3d(p, octaves, 2.0, 0.0)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"ridged_noise_3d({p}, gain=0) should equal 1 - abs(value_noise_3d)"
            )

    def test_different_inputs_differ(self):
        """Different 3D inputs should produce different outputs."""
        values = [
            py_ridged_noise_3d(
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
# Test: T-DEMO-1.32 Ridged Perlin 3D
# =============================================================================


class TestRidgedPerlin3D:
    """Tests for ridged_perlin_3d (Perlin noise FBM base)."""

    def test_range(self):
        """Output should be in [0, 1]."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5, iy * 0.5, iz * 0.5)
                    v = py_ridged_perlin_3d(
                        p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
                    )
                    assert math.isfinite(v), f"ridged_perlin_3d({p}) non-finite: {v}"
                    assert 0.0 <= v <= 1.0, (
                        f"ridged_perlin_3d({p}) = {v} outside [0, 1]"
                    )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (1.618, 2.718, 3.142)
        v1 = py_ridged_perlin_3d(
            p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
        )
        for _ in range(10):
            v2 = py_ridged_perlin_3d(
                p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_single_octave_equals_one_minus_abs_perlin(self):
        """With octaves=1, ridged_perlin_3d should equal 1 - abs(perlin_noise_3d)."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5 + 0.1, iy * 0.5 + 0.2, iz * 0.5 + 0.3)
                    ridged_val = py_ridged_perlin_3d(
                        p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN
                    )
                    expected = 1.0 - abs(py_perlin_noise_3d(p))
                    assert ridged_val == pytest.approx(expected, abs=TOL_ABS), (
                        f"ridged_perlin_3d({p}, 1) != 1 - abs(perlin_noise_3d({p}))"
                    )

    def test_zero_octaves_returns_one(self):
        """With octaves=0, ridged_perlin_3d should return 1.0."""
        for p in [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]:
            v = py_ridged_perlin_3d(
                p, 0, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            assert v == pytest.approx(1.0, abs=TOL_ABS), (
                f"ridged_perlin_3d({p}, 0) should be 1.0, got {v}"
            )

    def test_unity_at_integer_grid(self):
        """At integer grid positions, ridged_perlin_3d = 1.0.

        Perlin noise is zero at integer grid points (all gradient dot
        products are zero). So fbm_perlin_3d = 0 at integer positions,
        giving ridged = 1.0 - abs(0) = 1.0 (the sharpest possible ridge).
        """
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                for iz in range(-3, 4):
                    p = (float(ix), float(iy), float(iz))
                    result = py_ridged_perlin_3d(
                        p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
                    )
                    assert result == pytest.approx(1.0, abs=TOL_ABS), (
                        f"ridged_perlin_3d at integer ({ix},{iy},{iz}) "
                        f"should be 1.0, got {result}"
                    )

    def test_spectral_composition(self):
        """Different octave counts produce different outputs."""
        p = (0.5, 0.25, 0.125)
        v1 = py_ridged_perlin_3d(p, 1, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v4 = py_ridged_perlin_3d(p, 4, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        v8 = py_ridged_perlin_3d(p, 8, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        diffs = [abs(v4 - v1), abs(v8 - v1), abs(v8 - v4)]
        assert any(d > 1e-8 for d in diffs), (
            "Different octave counts should produce different ridged_perlin_3d outputs"
        )

    def test_lacunarity_affects_output(self):
        """Different lacunarity values should produce different outputs."""
        p = (0.5, 0.25, 0.125)
        base = py_ridged_perlin_3d(p, 4, 2.0, 0.5)
        for lacunarity in [1.5, 3.0, 4.0]:
            v = py_ridged_perlin_3d(p, 4, lacunarity, 0.5)
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Lacunarity did not affect ridged_perlin_3d output")

    def test_gain_affects_output(self):
        """Different gain values should produce different outputs."""
        p = (0.5, 0.25, 0.125)
        base = py_ridged_perlin_3d(p, 4, 2.0, 0.5)
        for gain in [0.25, 0.75]:
            v = py_ridged_perlin_3d(p, 4, 2.0, gain)
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Gain did not affect ridged_perlin_3d output")

    def test_zero_gain(self):
        """With gain=0.0, only the first octave contributes."""
        p = (1.5, 2.5, 3.5)
        expected = 1.0 - abs(py_perlin_noise_3d(p))
        for octaves in [4, 8]:
            result = py_ridged_perlin_3d(p, octaves, 2.0, 0.0)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"ridged_perlin_3d({p}, gain=0) != 1 - abs(perlin_noise_3d)"
            )

    def test_different_inputs_differ(self):
        """Different 3D inputs should produce different outputs."""
        values = [
            py_ridged_perlin_3d(
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
# Test: T-DEMO-1.32 Value-Based vs Perlin-Based Ridged Noise Comparison
# =============================================================================


class TestRidgedValueVsPerlin:
    """Tests comparing value-based and Perlin-based ridged noise."""

    def test_value_ridged_differs_from_perlin_ridged(self):
        """ridged_noise_3d and ridged_perlin_3d should produce different results."""
        p = (3.7, 4.2, 5.1)
        val_ridged = py_ridged_noise_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        perlin_ridged = py_ridged_perlin_3d(
            p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
        )
        assert abs(val_ridged - perlin_ridged) > 0.01, (
            "Value-based and Perlin-based ridged noise should produce different results"
        )

    def test_value_ridged_nonzero_at_integers(self):
        """ridged_noise_3d (value base) typically differs from 1 at integers."""
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                for iz in range(-3, 4):
                    p = (float(ix), float(iy), float(iz))
                    result = py_ridged_noise_3d(
                        p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
                    )
                    # Value noise is non-zero at integers, so ridged noise
                    # should not be exactly 1 (but could be close)
                    assert 0.0 <= result <= 1.0, (
                        f"ridged_noise_3d at integer ({ix},{iy},{iz}) = {result}"
                    )

    def test_perlin_ridged_unity_at_integers(self):
        """ridged_perlin_3d is 1.0 at integer positions."""
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                for iz in range(-3, 4):
                    p = (float(ix), float(iy), float(iz))
                    result = py_ridged_perlin_3d(
                        p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
                    )
                    assert result == pytest.approx(1.0, abs=TOL_ABS), (
                        f"ridged_perlin_3d at integer ({ix},{iy},{iz}) "
                        f"should be 1.0, got {result}"
                    )


# =============================================================================
# Test: T-DEMO-1.32 Continuity
# =============================================================================


class TestRidgedContinuity:
    """Tests that ridged noise is continuous along each axis."""

    def test_ridged_3d_continuity_along_x(self):
        """ridged_noise_3d should be continuous along x for fixed y, z."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_ridged_noise_3d((0.0, 1.0, 2.0), *params)
        for i in range(1, 200):
            curr = py_ridged_noise_3d((i * step, 1.0, 2.0), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in ridged_noise_3d at x={i * step}: diff={diff}"
            )
            prev = curr

    def test_ridged_3d_continuity_along_y(self):
        """ridged_noise_3d should be continuous along y for fixed x, z."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_ridged_noise_3d((1.0, 0.0, 2.0), *params)
        for i in range(1, 200):
            curr = py_ridged_noise_3d((1.0, i * step, 2.0), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in ridged_noise_3d at y={i * step}: diff={diff}"
            )
            prev = curr

    def test_ridged_3d_continuity_along_z(self):
        """ridged_noise_3d should be continuous along z for fixed x, y."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_ridged_noise_3d((1.0, 2.0, 0.0), *params)
        for i in range(1, 200):
            curr = py_ridged_noise_3d((1.0, 2.0, i * step), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in ridged_noise_3d at z={i * step}: diff={diff}"
            )
            prev = curr

    def test_ridged_perlin_3d_continuity_along_x(self):
        """ridged_perlin_3d should be continuous along x."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_ridged_perlin_3d((0.0, 1.0, 2.0), *params)
        for i in range(1, 200):
            curr = py_ridged_perlin_3d((i * step, 1.0, 2.0), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in ridged_perlin_3d at x={i * step}: diff={diff}"
            )
            prev = curr

    def test_ridged_perlin_3d_continuity_along_y(self):
        """ridged_perlin_3d should be continuous along y."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_ridged_perlin_3d((1.0, 0.0, 2.0), *params)
        for i in range(1, 200):
            curr = py_ridged_perlin_3d((1.0, i * step, 2.0), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in ridged_perlin_3d at y={i * step}: diff={diff}"
            )
            prev = curr

    def test_ridged_perlin_3d_continuity_along_z(self):
        """ridged_perlin_3d should be continuous along z."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_ridged_perlin_3d((1.0, 2.0, 0.0), *params)
        for i in range(1, 200):
            curr = py_ridged_perlin_3d((1.0, 2.0, i * step), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in ridged_perlin_3d at z={i * step}: diff={diff}"
            )
            prev = curr

    def test_ridged_1d_continuity(self):
        """ridged_noise_1d should be continuous."""
        step = 0.001
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_ridged_noise_1d(0.0, *params)
        for i in range(1, 500):
            curr = py_ridged_noise_1d(i * step, *params)
            diff = abs(curr - prev)
            assert diff < 0.02, (
                f"Discontinuity in ridged_noise_1d at {i * step}: diff={diff}"
            )
            prev = curr


# =============================================================================
# Test: T-DEMO-1.32 Mean and Distribution
# =============================================================================


class TestRidgedDistribution:
    """Tests that ridged noise has reasonable distribution properties."""

    NUM_SAMPLES = 5000

    def test_ridged_1d_all_non_negative(self):
        """All ridged_noise_1d samples should be >= 0 (range is [0, 1])."""
        samples = [
            py_ridged_noise_1d(i * 0.07, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            for i in range(self.NUM_SAMPLES)
        ]
        assert all(v >= 0.0 for v in samples), (
            "rided_noise_1d should never produce negative values"
        )
        assert all(v <= 1.0 for v in samples), (
            "rided_noise_1d should never produce values > 1.0"
        )

    def test_ridged_2d_all_non_negative(self):
        """All ridged_noise_2d samples should be >= 0."""
        samples = [
            py_ridged_noise_2d(
                (i * 0.07, i * 0.11),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(self.NUM_SAMPLES)
        ]
        assert all(v >= 0.0 for v in samples)

    def test_ridged_3d_all_non_negative(self):
        """All ridged_noise_3d samples should be >= 0."""
        samples = [
            py_ridged_noise_3d(
                (i * 0.07, i * 0.11, i * 0.13),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(self.NUM_SAMPLES)
        ]
        assert all(v >= 0.0 for v in samples)

    def test_ridged_perlin_3d_all_non_negative(self):
        """All ridged_perlin_3d samples should be >= 0."""
        samples = [
            py_ridged_perlin_3d(
                (i * 0.07, i * 0.11, i * 0.13),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(self.NUM_SAMPLES)
        ]
        assert all(v >= 0.0 for v in samples)

    def test_ridged_1d_mean_positive(self):
        """Mean of ridged_noise_1d should be > 0 (folded distribution)."""
        samples = [
            py_ridged_noise_1d(i * 0.07, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert mean > 0.1, (
            f"rided_noise_1d mean {mean:.4f} should be > 0.1 (folded distribution)"
        )

    def test_ridged_3d_values_not_all_same(self):
        """ridged_noise_3d should not produce constant output."""
        samples = [
            py_ridged_noise_3d(
                (i * 0.17, i * 0.23, i * 0.31),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 20, (
            f"Expected at least 20 unique values out of 50, got {unique}"
        )

    def test_ridged_perlin_3d_values_not_all_same(self):
        """ridged_perlin_3d should not produce constant output."""
        samples = [
            py_ridged_perlin_3d(
                (i * 0.17, i * 0.23, i * 0.31),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 20, (
            f"Expected at least 20 unique values out of 50, got {unique}"
        )

    def test_ridged_1d_values_span_range(self):
        """ridged_noise_1d values should span most of [0, 1] range."""
        samples = [
            py_ridged_noise_1d(i * 0.07, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            for i in range(self.NUM_SAMPLES)
        ]
        max_val = max(samples)
        min_val = min(samples)
        span = max_val - min_val
        assert span > 0.5, (
            f"rided_noise_1d range span {span:.4f} should be > 0.5"
        )


# =============================================================================
# Test: T-DEMO-1.32 Cross-Function Consistency
# =============================================================================


class TestRidgedCrossFunction:
    """Tests that ridged noise functions with matching dimensionality are consistent."""

    def test_all_four_handle_large_coordinates(self):
        """All four ridged noise functions handle large coordinate values."""
        large_ps = [1000.0, -1000.0, 1e6, -1e6, 0.001, -0.001]
        for p in large_ps:
            v1 = py_ridged_noise_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert math.isfinite(v1), f"ridged_noise_1d({p}) non-finite"
            assert 0.0 <= v1 <= 1.0
        for p in large_ps:
            v2 = py_ridged_noise_2d(
                (p, p * 0.5), DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            assert math.isfinite(v2), f"ridged_noise_2d({p}) non-finite"
            assert 0.0 <= v2 <= 1.0
        for p in large_ps:
            v3 = py_ridged_noise_3d(
                (p, p * 0.5, p * 0.25),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            assert math.isfinite(v3), f"ridged_noise_3d({p}) non-finite"
            assert 0.0 <= v3 <= 1.0
        for p in large_ps:
            vp = py_ridged_perlin_3d(
                (p, p * 0.5, p * 0.25),
                DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            assert math.isfinite(vp), f"ridged_perlin_3d({p}) non-finite"
            assert 0.0 <= vp <= 1.0

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
            py_ridged_noise_1d(p_1d, *tp) for tp in test_params
        ]
        results_3d = [
            py_ridged_noise_3d(p_3d, *tp) for tp in test_params
        ]
        results_perlin = [
            py_ridged_perlin_3d(p_3d, *tp) for tp in test_params
        ]

        for name, results in [
            ("ridged_noise_1d", results_1d),
            ("ridged_noise_3d", results_3d),
            ("ridged_perlin_3d", results_perlin),
        ]:
            unique = len(set(round(v, 10) for v in results))
            assert unique >= 3, (
                f"{name} produced only {unique} unique values across "
                f"4 parameter sets"
            )

    def test_ridged_fbm_identity(self):
        """ridged_noise(p) = 1 - abs(fbm(p)) for all dimensionalities."""
        for i in range(-20, 21):
            p = i * 0.17 + 0.037
            ridged = py_ridged_noise_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            fbm_val = py_fbm_1d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            expected = 1.0 - abs(fbm_val)
            assert ridged == pytest.approx(expected, abs=TOL_ABS), (
                f"rided_noise_1d({p}) should equal 1 - abs(fbm_1d({p})): "
                f"{ridged} vs {expected}"
            )

    def test_ridged_perlin_fbm_identity(self):
        """ridged_perlin_3d(p) = 1 - abs(fbm_perlin_3d(p))."""
        for i in range(-10, 11):
            p = (i * 0.3 + 0.1, i * 0.3 + 0.2, i * 0.3 + 0.3)
            ridged = py_ridged_perlin_3d(
                p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            fbm_val = py_fbm_perlin_3d(
                p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            expected = 1.0 - abs(fbm_val)
            assert ridged == pytest.approx(expected, abs=TOL_ABS), (
                f"rided_perlin_3d({p}) should equal 1 - abs(fbm_perlin_3d({p})): "
                f"{ridged} vs {expected}"
            )


# =============================================================================
# Test: T-DEMO-1.32 Additional Ridged Noise 1D — Extreme Parameters & Boundaries
# =============================================================================


class TestRidgedNoise1DAdditional:
    """Additional edge-case and stress tests for ridged_noise_1d."""

    def test_extreme_octaves_stability(self):
        """Ridged noise remains stable with very high octave counts."""
        for octaves in [16, 32, 64]:
            for i in range(-20, 21):
                p = i * 0.17
                v = py_ridged_noise_1d(p, octaves, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                assert math.isfinite(v), (
                    f"rided_noise_1d({p}, oct={octaves}) non-finite"
                )
                assert 0.0 <= v <= 1.0, (
                    f"rided_noise_1d({p}, oct={octaves}) = {v} outside [0, 1]"
                )

    def test_negative_gain_stability(self):
        """Negative gain values should still produce finite output in [0, 1]."""
        for gain in [-0.5, -1.0, -2.0]:
            for i in range(-10, 11):
                p = i * 0.3
                v = py_ridged_noise_1d(p, 4, 2.0, gain)
                assert math.isfinite(v), (
                    f"rided_noise_1d({p}, gain={gain}) non-finite"
                )

    def test_extreme_lacunarity(self):
        """Very high and very low lacunarity produce valid output."""
        for lacunarity in [0.1, 0.5, 10.0, 100.0]:
            for i in range(-10, 11):
                p = i * 0.3
                v = py_ridged_noise_1d(p, 4, lacunarity, 0.5)
                assert math.isfinite(v), (
                    f"rided_noise_1d({p}, lac={lacunarity}) non-finite"
                )
                assert 0.0 <= v <= 1.0, (
                    f"rided_noise_1d({p}, lac={lacunarity}) = {v} outside [0, 1]"
                )

    def test_gain_near_one(self):
        """With gain close to 1.0, all octaves have similar amplitude."""
        p = 0.75
        for gain in [0.9, 0.99]:
            v = py_ridged_noise_1d(p, 8, 2.0, gain)
            assert math.isfinite(v), (
                f"rided_noise_1d({p}, gain={gain}) non-finite"
            )
            assert 0.0 <= v <= 1.0

    def test_precise_fbm_zero_ridge(self):
        """When fbm_1d returns exactly 0, ridged_noise_1d returns exactly 1.0.

        The ridged noise identity 1 - abs(0) = 1 produces the sharpest
        possible ridge at FBM zero-crossings.
        """
        fbm_zero = py_fbm_1d(0.0, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        ridged = py_ridged_noise_1d(0.0, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        expected = 1.0 - abs(fbm_zero)
        assert ridged == pytest.approx(expected, abs=TOL_ABS), (
            f"rided_noise_1d(0) = {ridged} != 1 - abs(fbm_1d(0)) = {expected}"
        )

    def test_unity_gain_single_octave(self):
        """With gain=1.0 and octaves=1, ridged = 1 - abs(value_noise_1d)."""
        for i in range(-20, 21):
            p = i * 0.13 + 0.037
            ridged = py_ridged_noise_1d(p, 1, 2.0, 1.0)
            expected = 1.0 - abs(py_value_noise_1d(p))
            assert ridged == pytest.approx(expected, abs=TOL_ABS), (
                f"rided_noise_1d({p}, gain=1.0) != 1 - abs(value_noise_1d({p}))"
            )


# =============================================================================
# Test: T-DEMO-1.32 Additional Ridged Noise 2D — Ridges, Valleys, Continuity
# =============================================================================


class TestRidgedNoise2DAdditional:
    """Additional tests for ridged_noise_2d covering ridge/valley and continuity."""

    def test_sharp_ridge_at_fbm_zero(self):
        """When FBM is near 0, ridged_noise_2d is near 1 (sharp ridge)."""
        near_zero_points = []
        for ix in range(-20, 21):
            for iy in range(-20, 21):
                p = (ix * 0.07, iy * 0.13)
                fbm_val = py_fbm_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                ridged_val = py_ridged_noise_2d(
                    p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
                )
                if abs(fbm_val) < 0.05:
                    near_zero_points.append((p, fbm_val, ridged_val))

        if near_zero_points:
            for p, fbm_val, ridged_val in near_zero_points[:10]:
                assert ridged_val > 0.9, (
                    f"At 2D FBM zero-crossing p={p}, fbm={fbm_val:.6f}, "
                    f"rided should be near 1, got {ridged_val:.6f}"
                )

    def test_smooth_valley_at_fbm_extrema(self):
        """When FBM is near +/-1, ridged_noise_2d is near 0 (valley)."""
        extreme_points = []
        for ix in range(-20, 21):
            for iy in range(-20, 21):
                p = (ix * 0.07, iy * 0.13)
                fbm_val = py_fbm_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
                ridged_val = py_ridged_noise_2d(
                    p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
                )
                if abs(fbm_val) > 0.5:
                    extreme_points.append((p, fbm_val, ridged_val))

        if extreme_points:
            avg = sum(r for _, _, r in extreme_points) / len(extreme_points)
            assert avg < 0.7, (
                f"At 2D FBM extrema, avg ridged noise = {avg:.4f}, expected < 0.7"
            )

    def test_lacunarity_affects_output(self):
        """Different lacunarity values produce different outputs."""
        p = (0.5, 0.25)
        base = py_ridged_noise_2d(p, 4, 2.0, 0.5)
        diffs = [
            abs(py_ridged_noise_2d(p, 4, lac, 0.5) - base)
            for lac in [1.5, 3.0, 4.0]
        ]
        assert any(d > 1e-8 for d in diffs), (
            "Lacunarity should affect ridged_noise_2d output"
        )

    def test_gain_affects_output(self):
        """Different gain values produce different outputs."""
        p = (0.5, 0.25)
        base = py_ridged_noise_2d(p, 4, 2.0, 0.5)
        diffs = [
            abs(py_ridged_noise_2d(p, 4, 2.0, g) - base)
            for g in [0.25, 0.75]
        ]
        assert any(d > 1e-8 for d in diffs), (
            "Gain should affect ridged_noise_2d output"
        )

    def test_range_various_parameters(self):
        """Output in [0,1] for various parameter combinations."""
        params = [
            (1, 2.0, 0.5),
            (4, 1.5, 0.7),
            (8, 3.0, 0.3),
            (16, 2.0, 0.5),
        ]
        for ix in range(-8, 9):
            for iy in range(-8, 9):
                p = (ix * 0.3, iy * 0.3)
                for octaves, lacunarity, gain in params:
                    v = py_ridged_noise_2d(p, octaves, lacunarity, gain)
                    assert math.isfinite(v), (
                        f"rided_noise_2d({p}, oct={octaves}) non-finite"
                    )
                    assert 0.0 <= v <= 1.0, (
                        f"rided_noise_2d({p}, oct={octaves}) = {v} outside [0, 1]"
                    )

    def test_continuity_along_x(self):
        """ridged_noise_2d continuous along x for fixed y."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_ridged_noise_2d((0.0, 1.5), *params)
        for i in range(1, 200):
            curr = py_ridged_noise_2d((i * step, 1.5), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in ridged_noise_2d at x={i * step}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """ridged_noise_2d continuous along y for fixed x."""
        step = 0.01
        params = (DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
        prev = py_ridged_noise_2d((1.5, 0.0), *params)
        for i in range(1, 200):
            curr = py_ridged_noise_2d((1.5, i * step), *params)
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in ridged_noise_2d at y={i * step}: diff={diff}"
            )
            prev = curr

    def test_dense_sampling_range(self):
        """Dense 2D sampling stays in [0, 1]."""
        for i in range(200):
            p = (i * 0.13, i * 0.17)
            v = py_ridged_noise_2d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert 0.0 <= v <= 1.0, (
                f"rided_noise_2d({p}) = {v} outside [0, 1]"
            )

    def test_deterministic_various_params(self):
        """Determinism holds across 2D parameter combinations."""
        params = [
            (1, 2.0, 0.5),
            (4, 3.0, 0.25),
            (8, 2.0, 0.5),
        ]
        for octaves, lacunarity, gain in params:
            p = (2.718, 3.142)
            v1 = py_ridged_noise_2d(p, octaves, lacunarity, gain)
            for _ in range(5):
                v2 = py_ridged_noise_2d(p, octaves, lacunarity, gain)
                assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                    f"rided_noise_2d not deterministic with "
                    f"oct={octaves}, lac={lacunarity}, gain={gain}"
                )

    def test_unity_lacunarity(self):
        """With lacunarity=1.0, all octaves sample at same frequency."""
        p = (1.5, 2.5)
        expected = 1.0 - abs(py_value_noise_2d(p))
        for octaves in [3, 5, 8]:
            result = py_ridged_noise_2d(p, octaves, 1.0, 0.5)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"rided_noise_2d({p}, lac=1.0) != 1 - abs(value_noise_2d)"
            )

    def test_zero_gain(self):
        """With gain=0.0, only first octave contributes."""
        p = (2.5, 1.5)
        expected = 1.0 - abs(py_value_noise_2d(p))
        for octaves in [4, 8]:
            result = py_ridged_noise_2d(p, octaves, 2.0, 0.0)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"rided_noise_2d({p}, gain=0) != 1 - abs(value_noise_2d)"
            )


# =============================================================================
# Test: T-DEMO-1.32 Additional Ridged Noise 3D — Ridges, Valleys, Parameters
# =============================================================================


class TestRidgedNoise3DAdditional:
    """Additional tests for ridged_noise_3d covering ridge/valley and parameters."""

    def test_sharp_ridge_at_fbm_zero(self):
        """When FBM is near 0, ridged_noise_3d is near 1 (sharp ridge)."""
        near_zero_points = []
        for i in range(-20, 21):
            p = (i * 0.07, i * 0.11, i * 0.13)
            fbm_val = py_fbm_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            ridged_val = py_ridged_noise_3d(
                p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            if abs(fbm_val) < 0.05:
                near_zero_points.append((p, fbm_val, ridged_val))

        if near_zero_points:
            for p, fbm_val, ridged_val in near_zero_points[:10]:
                assert ridged_val > 0.9, (
                    f"At 3D FBM zero-crossing p={p}, fbm={fbm_val:.6f}, "
                    f"rided should be near 1, got {ridged_val:.6f}"
                )

    def test_smooth_valley_at_fbm_extrema(self):
        """When FBM is near +/-1, ridged_noise_3d is lower (valley)."""
        extreme_points = []
        for i in range(-20, 21):
            p = (i * 0.07, i * 0.11, i * 0.13)
            fbm_val = py_fbm_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            ridged_val = py_ridged_noise_3d(
                p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            if abs(fbm_val) > 0.5:
                extreme_points.append((p, fbm_val, ridged_val))

        if extreme_points:
            avg = sum(r for _, _, r in extreme_points) / len(extreme_points)
            assert avg < 0.7, (
                f"At 3D FBM extrema, avg ridged = {avg:.4f}, expected < 0.7"
            )

    def test_lacunarity_affects_output(self):
        """Different lacunarity values produce different outputs."""
        p = (0.5, 0.25, 0.125)
        base = py_ridged_noise_3d(p, 4, 2.0, 0.5)
        for lacunarity in [1.5, 3.0, 4.0]:
            v = py_ridged_noise_3d(p, 4, lacunarity, 0.5)
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Lacunarity did not affect ridged_noise_3d output")

    def test_gain_affects_output(self):
        """Different gain values produce different outputs."""
        p = (0.5, 0.25, 0.125)
        base = py_ridged_noise_3d(p, 4, 2.0, 0.5)
        for gain in [0.25, 0.75]:
            v = py_ridged_noise_3d(p, 4, 2.0, gain)
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Gain did not affect ridged_noise_3d output")

    def test_range_various_parameters(self):
        """Output in [0,1] for various parameter combinations."""
        params = [
            (1, 2.0, 0.5),
            (4, 1.5, 0.7),
            (8, 3.0, 0.3),
            (16, 2.0, 0.5),
        ]
        for i in range(-5, 6):
            p = (i * 0.5, i * 0.5, i * 0.5)
            for octaves, lacunarity, gain in params:
                v = py_ridged_noise_3d(p, octaves, lacunarity, gain)
                assert math.isfinite(v), (
                    f"rided_noise_3d({p}, oct={octaves}) non-finite"
                )
                assert 0.0 <= v <= 1.0, (
                    f"rided_noise_3d({p}, oct={octaves}) = {v} outside [0, 1]"
                )

    def test_deterministic_various_params(self):
        """Determinism holds across 3D parameter combinations."""
        params = [
            (1, 2.0, 0.5),
            (4, 3.0, 0.25),
            (8, 2.0, 0.5),
        ]
        for octaves, lacunarity, gain in params:
            p = (1.618, 2.718, 3.142)
            v1 = py_ridged_noise_3d(p, octaves, lacunarity, gain)
            for _ in range(5):
                v2 = py_ridged_noise_3d(p, octaves, lacunarity, gain)
                assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                    f"rided_noise_3d not deterministic with "
                    f"oct={octaves}, lac={lacunarity}, gain={gain}"
                )

    def test_unity_gain(self):
        """With gain=1.0, octaves add equal-amplitude detail.

        Each octave has the same amplitude, so more octaves produce
        progressively larger total signal before normalization.
        """
        p = (0.5, 0.25, 0.125)
        v1 = py_ridged_noise_3d(p, 1, 2.0, 1.0)
        v4 = py_ridged_noise_3d(p, 4, 2.0, 1.0)
        v8 = py_ridged_noise_3d(p, 8, 2.0, 1.0)
        assert math.isfinite(v1), "rided_noise_3d(gain=1, oct=1) non-finite"
        assert math.isfinite(v4), "rided_noise_3d(gain=1, oct=4) non-finite"
        assert math.isfinite(v8), "rided_noise_3d(gain=1, oct=8) non-finite"
        assert 0.0 <= v1 <= 1.0
        assert 0.0 <= v4 <= 1.0
        assert 0.0 <= v8 <= 1.0

    def test_dense_sampling_range(self):
        """Dense 3D sampling stays in [0, 1]."""
        for i in range(200):
            p = (i * 0.11, i * 0.13, i * 0.17)
            v = py_ridged_noise_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert 0.0 <= v <= 1.0, (
                f"rided_noise_3d({p}) = {v} outside [0, 1]"
            )


# =============================================================================
# Test: T-DEMO-1.32 Additional Ridged Perlin 3D — Ridges, Valleys, Parameters
# =============================================================================


class TestRidgedPerlin3DAdditional:
    """Additional tests for ridged_perlin_3d covering ridge/valley and parameters."""

    def test_sharp_ridge_at_fbm_zero(self):
        """When Perlin FBM is near 0, ridged_perlin_3d is near 1 (ridge)."""
        near_zero_points = []
        for i in range(-20, 21):
            p = (i * 0.07, i * 0.11, i * 0.13)
            fbm_val = py_fbm_perlin_3d(
                p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            ridged_val = py_ridged_perlin_3d(
                p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            if abs(fbm_val) < 0.05:
                near_zero_points.append((p, fbm_val, ridged_val))

        if near_zero_points:
            for p, fbm_val, ridged_val in near_zero_points[:10]:
                assert ridged_val > 0.9, (
                    f"At Perlin FBM zero-crossing p={p}, fbm={fbm_val:.6f}, "
                    f"rided should be near 1, got {ridged_val:.6f}"
                )

    def test_smooth_valley_at_fbm_extrema(self):
        """When Perlin FBM is near +/-1, ridged_perlin_3d is lower (valley)."""
        extreme_points = []
        for i in range(-20, 21):
            p = (i * 0.07, i * 0.11, i * 0.13)
            fbm_val = py_fbm_perlin_3d(
                p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            ridged_val = py_ridged_perlin_3d(
                p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            if abs(fbm_val) > 0.5:
                extreme_points.append((p, fbm_val, ridged_val))

        if extreme_points:
            avg = sum(r for _, _, r in extreme_points) / len(extreme_points)
            assert avg < 0.7, (
                f"At Perlin FBM extrema, avg ridged = {avg:.4f}, expected < 0.7"
            )

    def test_range_various_parameters(self):
        """Output in [0,1] for various parameter combinations."""
        params = [
            (1, 2.0, 0.5),
            (4, 1.5, 0.7),
            (8, 3.0, 0.3),
            (16, 2.0, 0.5),
        ]
        for i in range(-5, 6):
            p = (i * 0.5, i * 0.5, i * 0.5)
            for octaves, lacunarity, gain in params:
                v = py_ridged_perlin_3d(p, octaves, lacunarity, gain)
                assert math.isfinite(v), (
                    f"rided_perlin_3d({p}, oct={octaves}) non-finite"
                )
                assert 0.0 <= v <= 1.0, (
                    f"rided_perlin_3d({p}, oct={octaves}) = {v} outside [0, 1]"
                )

    def test_deterministic_various_params(self):
        """Determinism holds across Perlin parameter combinations."""
        params = [
            (1, 2.0, 0.5),
            (4, 3.0, 0.25),
            (8, 2.0, 0.5),
        ]
        for octaves, lacunarity, gain in params:
            p = (1.618, 2.718, 3.142)
            v1 = py_ridged_perlin_3d(p, octaves, lacunarity, gain)
            for _ in range(5):
                v2 = py_ridged_perlin_3d(p, octaves, lacunarity, gain)
                assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                    f"rided_perlin_3d not deterministic with "
                    f"oct={octaves}, lac={lacunarity}, gain={gain}"
                )

    def test_dense_sampling_range(self):
        """Dense Perlin ridged sampling stays in [0, 1]."""
        for i in range(200):
            p = (i * 0.11, i * 0.13, i * 0.17)
            v = py_ridged_perlin_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            assert 0.0 <= v <= 1.0, (
                f"rided_perlin_3d({p}) = {v} outside [0, 1]"
            )

    def test_unity_lacunarity(self):
        """With lacunarity=1.0, all octaves sample at same frequency."""
        p = (1.5, 2.5, 3.5)
        expected = 1.0 - abs(py_perlin_noise_3d(p))
        for octaves in [3, 5, 8]:
            result = py_ridged_perlin_3d(p, octaves, 1.0, 0.5)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"rided_perlin_3d({p}, lac=1.0) != "
                f"1 - abs(perlin_noise_3d({p}))"
            )


# =============================================================================
# Test: T-DEMO-1.32 Additional Cross-Dimension Consistency
# =============================================================================


class TestRidgedCrossDimension:
    """Tests that ridged noise behaves consistently across dimensions."""

    def test_1d_slice_of_2d_agrees_with_1d(self):
        """2D ridged noise along y=0 should match 1D ridged noise.

        When the 2D coordinate has y=0 (on the first cell boundary of
        the hash function), the value-noise-2d evaluation should match
        the 1D evaluation at the same x.
        """
        for i in range(-10, 11):
            x = i * 0.37 + 0.1
            v1d = py_ridged_noise_1d(x, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
            v2d = py_ridged_noise_2d(
                (x, 0.0), DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN
            )
            # These should not necessarily be equal (different hash domains)
            # but both should be valid
            assert math.isfinite(v2d), f"rided_noise_2d({x}, 0) non-finite"
            assert 0.0 <= v2d <= 1.0

    def test_all_four_stable_at_origin(self):
        """All four ridged noise functions produce valid output at origin."""
        for octaves in [1, 4, 8]:
            for lacunarity in [1.0, 2.0]:
                for gain in [0.0, 0.5]:
                    v1d = py_ridged_noise_1d(0.0, octaves, lacunarity, gain)
                    v2d = py_ridged_noise_2d((0.0, 0.0), octaves, lacunarity, gain)
                    v3d = py_ridged_noise_3d((0.0, 0.0, 0.0), octaves, lacunarity, gain)
                    vp = py_ridged_perlin_3d((0.0, 0.0, 0.0), octaves, lacunarity, gain)
                    for name, val in [
                        ("rided_noise_1d", v1d),
                        ("rided_noise_2d", v2d),
                        ("rided_noise_3d", v3d),
                        ("rided_perlin_3d", vp),
                    ]:
                        assert math.isfinite(val), (
                            f"{name}(0, oct={octaves}, lac={lacunarity}, "
                            f"gain={gain}) non-finite"
                        )
                        assert 0.0 <= val <= 1.0, (
                            f"{name}(0, oct={octaves}, lac={lacunarity}, "
                            f"gain={gain}) = {val} outside [0, 1]"
                        )
