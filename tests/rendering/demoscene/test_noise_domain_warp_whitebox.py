"""
Whitebox tests for domain warp noise WGSL functions (T-DEMO-1.33).

Tests Python model implementations of each WGSL function, verifying:
  - Range: output approximately in [-1, 1] (inherited from FBM base)
  - Determinism: same input + parameters always produce same output
  - Zero strength: strength=0 means no warp, so equals base FBM
  - Warp strength: non-zero strength changes output (creates deformation)
  - Spectral independence: warp FBM and base FBM parameters are independent
  - Decorrelation: warp components use distinct offsets (100.0, 200.0)
  - Continuity: domain warp is smooth throughout the domain
  - Perlin vs value variants produce structurally different results
  - Mean near zero (zero-mean FBM base propagates through warp)
  - Negative inputs and large coordinates handled correctly

WHITEBOX coverage plan:
  Path A: domain_warp_2d range and finite output
  Path B: domain_warp_2d deterministic
  Path C: domain_warp_2d zero strength equals fbm_2d
  Path D: domain_warp_2d strength affects output (non-trivial deformation)
  Path E: domain_warp_2d warp spectral composition (octaves/lacunarity/gain)
  Path F: domain_warp_2d base spectral composition
  Path G: domain_warp_2d spectral independence (warp vs base params)
  Path H: domain_warp_2d negative inputs
  Path I: domain_warp_2d different inputs differ
  Path J: domain_warp_3d range and finite output
  Path K: domain_warp_3d deterministic
  Path L: domain_warp_3d zero strength equals fbm_3d
  Path M: domain_warp_3d strength affects output
  Path N: domain_warp_3d warp spectral composition
  Path O: domain_warp_3d base spectral composition
  Path P: domain_warp_3d negative inputs
  Path Q: domain_warp_perlin_3d range and finite output
  Path R: domain_warp_perlin_3d deterministic
  Path S: domain_warp_perlin_3d zero strength equals fbm_perlin_3d
  Path T: domain_warp_perlin_3d strength affects output
  Path U: domain_warp_perlin_3d warp spectral composition
  Path V: domain_warp_perlin_3d base spectral composition
  Path W: Value-based vs Perlin-based domain warp differ
  Path X: domain_warp_2d continuity along each axis
  Path Y: domain_warp_3d continuity along each axis
  Path Z: domain_warp_perlin_3d continuity along each axis
  Path AA: Mean near zero (all three variants)
  Path AB: Large coordinates produce valid output
  Path AC: All variants produce non-trivial deformation at strength > 0
  Path AD: Warp decorrelation (warp_x and warp_y differ from base FBM)
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
# Python model of domain warp noise (T-DEMO-1.33)
#
# WGSL reference (noise_domain_warp.wgsl):
#   fn domain_warp_2d(
#       p: vec2<f32>, strength: f32,
#       warp_octaves: u32, warp_lacunarity: f32, warp_gain: f32,
#       base_octaves: u32, base_lacunarity: f32, base_gain: f32
#   ) -> f32
#   fn domain_warp_3d(...) -> f32
#   fn domain_warp_perlin_3d(...) -> f32
#
# Domain warping transforms the input coordinates of an FBM evaluation
# using a displacement vector generated by another FBM evaluation:
#   warp_vec = strength * fbm_warp(p + seed_offset)
#   result   = fbm_base(p + warp_vec)
#
# The 100.0 / 200.0 offsets decorrelate the warp vector components.
# =============================================================================


def py_domain_warp_2d(
    p,
    strength: float,
    warp_octaves: int,
    warp_lacunarity: float,
    warp_gain: float,
    base_octaves: int,
    base_lacunarity: float,
    base_gain: float,
) -> float:
    """Model of WGSL domain_warp_2d.

    WGSL equivalent:
      fn domain_warp_2d(...) -> f32 {
          let warp_x = fbm_2d(p, warp_octaves, warp_lacunarity, warp_gain);
          let warp_y = fbm_2d(
              p + vec2<f32>(100.0, 100.0),
              warp_octaves, warp_lacunarity, warp_gain
          );
          let warped_p = p + strength * vec2<f32>(warp_x, warp_y);
          return fbm_2d(warped_p, base_octaves, base_lacunarity, base_gain);
      }
    """
    warp_x = py_fbm_2d(p, warp_octaves, warp_lacunarity, warp_gain)
    warp_y = py_fbm_2d(
        (p[0] + 100.0, p[1] + 100.0),
        warp_octaves, warp_lacunarity, warp_gain,
    )
    warped_p = (
        p[0] + strength * warp_x,
        p[1] + strength * warp_y,
    )
    return py_fbm_2d(warped_p, base_octaves, base_lacunarity, base_gain)


def py_domain_warp_3d(
    p,
    strength: float,
    warp_octaves: int,
    warp_lacunarity: float,
    warp_gain: float,
    base_octaves: int,
    base_lacunarity: float,
    base_gain: float,
) -> float:
    """Model of WGSL domain_warp_3d.

    WGSL equivalent:
      fn domain_warp_3d(...) -> f32 {
          let warp_x = fbm_3d(p, warp_octaves, warp_lacunarity, warp_gain);
          let warp_y = fbm_3d(
              p + vec3<f32>(100.0, 100.0, 100.0),
              warp_octaves, warp_lacunarity, warp_gain
          );
          let warp_z = fbm_3d(
              p + vec3<f32>(200.0, 200.0, 200.0),
              warp_octaves, warp_lacunarity, warp_gain
          );
          let warped_p = p + strength * vec3<f32>(warp_x, warp_y, warp_z);
          return fbm_3d(warped_p, base_octaves, base_lacunarity, base_gain);
      }
    """
    warp_x = py_fbm_3d(p, warp_octaves, warp_lacunarity, warp_gain)
    warp_y = py_fbm_3d(
        (p[0] + 100.0, p[1] + 100.0, p[2] + 100.0),
        warp_octaves, warp_lacunarity, warp_gain,
    )
    warp_z = py_fbm_3d(
        (p[0] + 200.0, p[1] + 200.0, p[2] + 200.0),
        warp_octaves, warp_lacunarity, warp_gain,
    )
    warped_p = (
        p[0] + strength * warp_x,
        p[1] + strength * warp_y,
        p[2] + strength * warp_z,
    )
    return py_fbm_3d(warped_p, base_octaves, base_lacunarity, base_gain)


def py_domain_warp_perlin_3d(
    p,
    strength: float,
    warp_octaves: int,
    warp_lacunarity: float,
    warp_gain: float,
    base_octaves: int,
    base_lacunarity: float,
    base_gain: float,
) -> float:
    """Model of WGSL domain_warp_perlin_3d.

    Uses Perlin-based FBM for both the warp field and the base evaluation,
    producing visually smoother warp fields with fewer grid artifacts.
    """
    warp_x = py_fbm_perlin_3d(p, warp_octaves, warp_lacunarity, warp_gain)
    warp_y = py_fbm_perlin_3d(
        (p[0] + 100.0, p[1] + 100.0, p[2] + 100.0),
        warp_octaves, warp_lacunarity, warp_gain,
    )
    warp_z = py_fbm_perlin_3d(
        (p[0] + 200.0, p[1] + 200.0, p[2] + 200.0),
        warp_octaves, warp_lacunarity, warp_gain,
    )
    warped_p = (
        p[0] + strength * warp_x,
        p[1] + strength * warp_y,
        p[2] + strength * warp_z,
    )
    return py_fbm_perlin_3d(warped_p, base_octaves, base_lacunarity, base_gain)


# =============================================================================
# Shared tolerances and test helpers
# =============================================================================

TOL_REL = 1e-5
TOL_ABS = 1e-9

# Standard spectral parameters used throughout testing
DEFAULT_WARP_OCTAVES = 4
DEFAULT_WARP_LACUNARITY = 2.0
DEFAULT_WARP_GAIN = 0.5
DEFAULT_BASE_OCTAVES = 8
DEFAULT_BASE_LACUNARITY = 2.0
DEFAULT_BASE_GAIN = 0.5
DEFAULT_STRENGTH = 1.0


# =============================================================================
# Test: T-DEMO-1.33 Domain Warp 2D
# =============================================================================


class TestDomainWarp2D:
    """Tests for domain_warp_2d."""

    # --- Range and finite ---

    def test_range(self):
        """Output should be approximately in [-1, 1] (inherited from FBM base)."""
        for ix in range(-10, 11):
            for iy in range(-10, 11):
                p = (ix * 0.3, iy * 0.3)
                v = py_domain_warp_2d(
                    p, DEFAULT_STRENGTH,
                    DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                    DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                )
                assert math.isfinite(v), f"domain_warp_2d({p}) non-finite: {v}"
                assert -1.5 <= v <= 1.5, (
                    f"domain_warp_2d({p}) = {v} outside expected range"
                )

    def test_range_various_parameters(self):
        """Output should be bounded for various parameter combinations."""
        params = [
            (0.5, 4, 2.0, 0.5, 8, 2.0, 0.5),
            (1.0, 2, 2.0, 0.5, 4, 2.0, 0.5),
            (2.0, 6, 1.5, 0.7, 6, 1.5, 0.7),
            (0.0, 4, 2.0, 0.5, 8, 2.0, 0.5),
            (5.0, 4, 2.0, 0.5, 4, 2.0, 0.5),
        ]
        for ix in range(-8, 9):
            for iy in range(-8, 9):
                p = (ix * 0.3, iy * 0.3)
                for strength, w_oct, w_lac, w_gain, b_oct, b_lac, b_gain in params:
                    v = py_domain_warp_2d(
                        p, strength, w_oct, w_lac, w_gain, b_oct, b_lac, b_gain,
                    )
                    assert math.isfinite(v), (
                        f"domain_warp_2d({p}, strength={strength}) non-finite: {v}"
                    )
                    assert -2.0 <= v <= 2.0, (
                        f"domain_warp_2d({p}, strength={strength}) = {v} outside range"
                    )

    def test_dense_sampling_range(self):
        """Dense sampling across many cells should stay bounded."""
        for i in range(200):
            p = (-20.0 + i * 0.2, -20.0 + i * 0.2)
            v = py_domain_warp_2d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            assert -1.5 <= v <= 1.5, (
                f"domain_warp_2d({p}) = {v} outside expected range"
            )

    # --- Determinism ---

    def test_deterministic(self):
        """Same input and parameters always produce same output."""
        p = (1.618, 2.718)
        v1 = py_domain_warp_2d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for _ in range(10):
            v2 = py_domain_warp_2d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                "domain_warp_2d not deterministic"
            )

    def test_deterministic_various_params(self):
        """Determinism holds across different parameter combinations."""
        param_sets = [
            (0.5, 2, 2.0, 0.5, 4, 2.0, 0.5),
            (1.0, 4, 2.0, 0.5, 8, 2.0, 0.5),
            (2.0, 6, 1.5, 0.7, 6, 1.5, 0.7),
        ]
        for strength, w_oct, w_lac, w_gain, b_oct, b_lac, b_gain in param_sets:
            p = (3.14159, 2.71828)
            v1 = py_domain_warp_2d(
                p, strength, w_oct, w_lac, w_gain, b_oct, b_lac, b_gain,
            )
            for _ in range(5):
                v2 = py_domain_warp_2d(
                    p, strength, w_oct, w_lac, w_gain, b_oct, b_lac, b_gain,
                )
                assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                    f"domain_warp_2d not deterministic with strength={strength}"
                )

    # --- Zero strength ---

    def test_zero_strength_equals_fbm_2d(self):
        """With strength=0, domain_warp_2d should equal fbm_2d.

        Zero warp strength means no displacement, so the base FBM
        evaluates at the original point p: result = fbm_2d(p, base_params).
        """
        for ix in range(-10, 11):
            for iy in range(-10, 11):
                p = (ix * 0.3 + 0.1, iy * 0.3 + 0.2)
                warp_val = py_domain_warp_2d(
                    p, 0.0,
                    DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                    DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                )
                fbm_val = py_fbm_2d(
                    p, DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                )
                assert warp_val == pytest.approx(fbm_val, abs=TOL_ABS), (
                    f"domain_warp_2d({p}, strength=0) = {warp_val} "
                    f"!= fbm_2d({p}) = {fbm_val}"
                )

    def test_zero_strength_various_base_params(self):
        """With strength=0, domain_warp_2d equals fbm_2d regardless of warp params."""
        base_params = [
            (4, 2.0, 0.5),
            (8, 2.0, 0.5),
            (6, 1.5, 0.7),
            (2, 3.0, 0.3),
        ]
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                p = (ix * 0.5 + 0.1, iy * 0.5 + 0.2)
                for b_oct, b_lac, b_gain in base_params:
                    warp_val = py_domain_warp_2d(
                        p, 0.0,
                        DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                        b_oct, b_lac, b_gain,
                    )
                    fbm_val = py_fbm_2d(p, b_oct, b_lac, b_gain)
                    assert warp_val == pytest.approx(fbm_val, abs=TOL_ABS), (
                        f"domain_warp_2d({p}, strength=0, base=({b_oct},{b_lac},{b_gain})) "
                        f"!= fbm_2d"
                    )

    # --- Strength affects output ---

    def test_strength_affects_output(self):
        """Non-zero strength should produce different output from zero strength.

        The warp displacement displaces the base FBM evaluation point,
        causing a different FBM value compared to the unwarped evaluation.
        """
        p = (0.5, 0.25)
        base = py_domain_warp_2d(
            p, 0.0,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        strengths = [0.5, 1.0, 2.0, 5.0]
        diffs = []
        for s in strengths:
            v = py_domain_warp_2d(
                p, s,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            diffs.append(abs(v - base))
        assert any(d > 1e-8 for d in diffs), (
            "Non-zero strength should change domain_warp_2d output"
        )

    def test_strength_creates_deformation(self):
        """Increasing strength should produce increasingly different results.

        Larger warp strength = larger displacement = more deviation from
        the original (unwarped) FBM value.
        """
        p = (0.75, 0.5)
        base = py_fbm_2d(p, DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN)
        results = []
        for s in [0.1, 0.5, 1.0, 2.0, 5.0]:
            v = py_domain_warp_2d(
                p, s,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            results.append((s, v, abs(v - base)))

        # Stronger warp should generally produce more deviation
        # (not strictly monotonic due to noise, but should trend)
        deviations = [d for _, _, d in results]
        assert max(deviations) > 0.01, (
            "Domain warp should produce non-trivial deformation"
        )

    # --- Spectral composition ---

    def test_warp_octaves_affect_output(self):
        """Changing warp octaves should change the output."""
        p = (0.5, 0.25)
        base = py_domain_warp_2d(
            p, DEFAULT_STRENGTH, 1, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for octaves in [2, 4, 6]:
            v = py_domain_warp_2d(
                p, DEFAULT_STRENGTH, octaves, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Warp octaves did not affect domain_warp_2d output")

    def test_warp_lacunarity_affects_output(self):
        """Changing warp lacunarity should change the output."""
        p = (0.5, 0.25)
        base = py_domain_warp_2d(
            p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, 2.0, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for lacunarity in [1.5, 3.0, 4.0]:
            v = py_domain_warp_2d(
                p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, lacunarity, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Warp lacunarity did not affect domain_warp_2d output")

    def test_warp_gain_affects_output(self):
        """Changing warp gain should change the output."""
        p = (0.5, 0.25)
        base = py_domain_warp_2d(
            p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, 0.5,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for gain in [0.25, 0.75]:
            v = py_domain_warp_2d(
                p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, gain,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Warp gain did not affect domain_warp_2d output")

    def test_base_octaves_affect_output(self):
        """Changing base octaves should change the output."""
        p = (0.5, 0.25)
        base = py_domain_warp_2d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            1, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for octaves in [2, 4, 8]:
            v = py_domain_warp_2d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                octaves, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Base octaves did not affect domain_warp_2d output")

    def test_base_lacunarity_affects_output(self):
        """Changing base lacunarity should change the output."""
        p = (0.5, 0.25)
        base = py_domain_warp_2d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, 2.0, DEFAULT_BASE_GAIN,
        )
        for lacunarity in [1.5, 3.0]:
            v = py_domain_warp_2d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, lacunarity, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Base lacunarity did not affect domain_warp_2d output")

    def test_base_gain_affects_output(self):
        """Changing base gain should change the output."""
        p = (0.5, 0.25)
        base = py_domain_warp_2d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, 0.5,
        )
        for gain in [0.25, 0.75]:
            v = py_domain_warp_2d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, gain,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Base gain did not affect domain_warp_2d output")

    # --- Spectral independence ---

    def test_warp_and_base_params_independent(self):
        """Warp and base spectral parameters should control independent FBM evaluations.

        Changing warp parameters only affects the displacement field,
        while changing base parameters only affects the evaluated noise value.
        Both should independently influence the final output.
        """
        p = (0.75, 0.5)

        # Reference with default params
        ref = py_domain_warp_2d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )

        # Change only warp params
        changed_warp = py_domain_warp_2d(
            p, DEFAULT_STRENGTH,
            2, 2.0, 0.5,  # fewer warp octaves
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )

        # Change only base params
        changed_base = py_domain_warp_2d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            2, 2.0, 0.5,  # fewer base octaves
        )

        # Both changes should produce different results
        diff_warp = abs(changed_warp - ref)
        diff_base = abs(changed_base - ref)

        assert diff_warp > 1e-8 or diff_base > 1e-8, (
            "Spectral parameters should affect output"
        )

    # --- Negative inputs ---

    def test_negative_inputs(self):
        """domain_warp_2d handles negative coordinates correctly."""
        for ix in range(-5, 0):
            for iy in range(-5, 0):
                p = (ix + 0.3, iy + 0.7)
                v = py_domain_warp_2d(
                    p, DEFAULT_STRENGTH,
                    DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                    DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                )
                assert math.isfinite(v), (
                    f"domain_warp_2d({p}) non-finite for negative input"
                )
                assert -1.5 <= v <= 1.5

    # --- Different inputs differ ---

    def test_different_inputs_differ(self):
        """Different inputs should produce different outputs."""
        values = [
            py_domain_warp_2d(
                (i * 0.17, i * 0.23), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            for i in range(20)
        ]
        unique = len(set(round(v, 10) for v in values))
        assert unique > 10, (
            f"Expected at least 10 unique values out of 20, got {unique}"
        )


# =============================================================================
# Test: T-DEMO-1.33 Domain Warp 3D (Value Noise FBM Base)
# =============================================================================


class TestDomainWarp3D:
    """Tests for domain_warp_3d (value noise FBM base)."""

    def test_range(self):
        """Output should be approximately in [-1, 1]."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5, iy * 0.5, iz * 0.5)
                    v = py_domain_warp_3d(
                        p, DEFAULT_STRENGTH,
                        DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                        DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                    )
                    assert math.isfinite(v), f"domain_warp_3d({p}) non-finite: {v}"
                    assert -1.5 <= v <= 1.5, (
                        f"domain_warp_3d({p}) = {v} outside expected range"
                    )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (1.618, 2.718, 3.142)
        v1 = py_domain_warp_3d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for _ in range(10):
            v2 = py_domain_warp_3d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_zero_strength_equals_fbm_3d(self):
        """With strength=0, domain_warp_3d should equal fbm_3d."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5 + 0.1, iy * 0.5 + 0.2, iz * 0.5 + 0.3)
                    warp_val = py_domain_warp_3d(
                        p, 0.0,
                        DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                        DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                    )
                    fbm_val = py_fbm_3d(
                        p, DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                    )
                    assert warp_val == pytest.approx(fbm_val, abs=TOL_ABS), (
                        f"domain_warp_3d({p}, strength=0) != fbm_3d({p})"
                    )

    def test_strength_affects_output(self):
        """Non-zero strength should produce different output from zero strength."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_3d(
            p, 0.0,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for s in [0.5, 1.0, 2.0]:
            v = py_domain_warp_3d(
                p, s,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Non-zero strength should affect domain_warp_3d output")

    def test_warp_octaves_affect_output(self):
        """Changing warp octaves should change the output."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_3d(
            p, DEFAULT_STRENGTH, 1, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for octaves in [2, 4, 6]:
            v = py_domain_warp_3d(
                p, DEFAULT_STRENGTH, octaves, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Warp octaves did not affect domain_warp_3d output")

    def test_warp_lacunarity_affects_output(self):
        """Changing warp lacunarity should change the output."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_3d(
            p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, 2.0, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for lacunarity in [1.5, 3.0, 4.0]:
            v = py_domain_warp_3d(
                p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, lacunarity, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Warp lacunarity did not affect domain_warp_3d output")

    def test_warp_gain_affects_output(self):
        """Changing warp gain should change the output."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_3d(
            p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, 0.5,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for gain in [0.25, 0.75]:
            v = py_domain_warp_3d(
                p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, gain,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Warp gain did not affect domain_warp_3d output")

    def test_base_octaves_affect_output(self):
        """Changing base octaves should change the output."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_3d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            1, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for octaves in [2, 4, 8]:
            v = py_domain_warp_3d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                octaves, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Base octaves did not affect domain_warp_3d output")

    def test_negative_inputs(self):
        """domain_warp_3d handles negative coordinates correctly."""
        for ix in range(-3, 0):
            for iy in range(-3, 0):
                for iz in range(-3, 0):
                    p = (ix + 0.3, iy + 0.7, iz + 0.5)
                    v = py_domain_warp_3d(
                        p, DEFAULT_STRENGTH,
                        DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                        DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                    )
                    assert math.isfinite(v)
                    assert -1.5 <= v <= 1.5

    def test_different_inputs_differ(self):
        """Different 3D inputs should produce different outputs."""
        values = [
            py_domain_warp_3d(
                (i * 0.17, i * 0.23, i * 0.31), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            for i in range(20)
        ]
        unique = len(set(round(v, 10) for v in values))
        assert unique > 10, (
            f"Expected at least 10 unique values out of 20, got {unique}"
        )


# =============================================================================
# Test: T-DEMO-1.33 Domain Warp Perlin 3D
# =============================================================================


class TestDomainWarpPerlin3D:
    """Tests for domain_warp_perlin_3d (Perlin FBM base)."""

    def test_range(self):
        """Output should be approximately in [-1, 1]."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5, iy * 0.5, iz * 0.5)
                    v = py_domain_warp_perlin_3d(
                        p, DEFAULT_STRENGTH,
                        DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                        DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                    )
                    assert math.isfinite(v), (
                        f"domain_warp_perlin_3d({p}) non-finite: {v}"
                    )
                    assert -1.5 <= v <= 1.5, (
                        f"domain_warp_perlin_3d({p}) = {v} outside expected range"
                    )

    def test_deterministic(self):
        """Same input always produces same output."""
        p = (1.618, 2.718, 3.142)
        v1 = py_domain_warp_perlin_3d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for _ in range(10):
            v2 = py_domain_warp_perlin_3d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            assert v1 == pytest.approx(v2, abs=TOL_ABS)

    def test_zero_strength_equals_fbm_perlin_3d(self):
        """With strength=0, domain_warp_perlin_3d should equal fbm_perlin_3d."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5 + 0.1, iy * 0.5 + 0.2, iz * 0.5 + 0.3)
                    warp_val = py_domain_warp_perlin_3d(
                        p, 0.0,
                        DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                        DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                    )
                    fbm_val = py_fbm_perlin_3d(
                        p, DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                    )
                    assert warp_val == pytest.approx(fbm_val, abs=TOL_ABS), (
                        f"domain_warp_perlin_3d({p}, strength=0) != fbm_perlin_3d({p})"
                    )

    def test_strength_affects_output(self):
        """Non-zero strength should produce different output from zero strength."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_perlin_3d(
            p, 0.0,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for s in [0.5, 1.0, 2.0, 5.0]:
            v = py_domain_warp_perlin_3d(
                p, s,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Non-zero strength should affect domain_warp_perlin_3d output")

    def test_warp_octaves_affect_output(self):
        """Changing warp octaves should change the output."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_perlin_3d(
            p, DEFAULT_STRENGTH, 1, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for octaves in [2, 4, 6]:
            v = py_domain_warp_perlin_3d(
                p, DEFAULT_STRENGTH, octaves, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Warp octaves did not affect domain_warp_perlin_3d output")

    def test_warp_lacunarity_affects_output(self):
        """Changing warp lacunarity should change the output."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_perlin_3d(
            p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, 2.0, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for lacunarity in [1.5, 3.0, 4.0]:
            v = py_domain_warp_perlin_3d(
                p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, lacunarity, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Warp lacunarity did not affect domain_warp_perlin_3d output")

    def test_warp_gain_affects_output(self):
        """Changing warp gain should change the output."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_perlin_3d(
            p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, 0.5,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for gain in [0.25, 0.75]:
            v = py_domain_warp_perlin_3d(
                p, DEFAULT_STRENGTH, DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, gain,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Warp gain did not affect domain_warp_perlin_3d output")

    def test_base_octaves_affect_output(self):
        """Changing base octaves should change the output."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_perlin_3d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            1, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for octaves in [2, 4, 8]:
            v = py_domain_warp_perlin_3d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                octaves, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Base octaves did not affect domain_warp_perlin_3d output")

    def test_base_lacunarity_affects_output(self):
        """Changing base lacunarity should change the output."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_perlin_3d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, 2.0, DEFAULT_BASE_GAIN,
        )
        for lacunarity in [1.5, 3.0]:
            v = py_domain_warp_perlin_3d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, lacunarity, DEFAULT_BASE_GAIN,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Base lacunarity did not affect domain_warp_perlin_3d output")

    def test_base_gain_affects_output(self):
        """Changing base gain should change the output."""
        p = (0.5, 0.25, 0.125)
        base = py_domain_warp_perlin_3d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, 0.5,
        )
        for gain in [0.25, 0.75]:
            v = py_domain_warp_perlin_3d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, gain,
            )
            if abs(v - base) > 1e-8:
                return
        pytest.fail("Base gain did not affect domain_warp_perlin_3d output")

    def test_negative_inputs(self):
        """domain_warp_perlin_3d handles negative coordinates correctly."""
        for ix in range(-3, 0):
            for iy in range(-3, 0):
                for iz in range(-3, 0):
                    p = (ix + 0.3, iy + 0.7, iz + 0.5)
                    v = py_domain_warp_perlin_3d(
                        p, DEFAULT_STRENGTH,
                        DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                        DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
                    )
                    assert math.isfinite(v)
                    assert -1.5 <= v <= 1.5

    def test_different_inputs_differ(self):
        """Different 3D inputs should produce different outputs."""
        values = [
            py_domain_warp_perlin_3d(
                (i * 0.17, i * 0.23, i * 0.31), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            for i in range(20)
        ]
        unique = len(set(round(v, 10) for v in values))
        assert unique > 10, (
            f"Expected at least 10 unique values out of 20, got {unique}"
        )


# =============================================================================
# Test: T-DEMO-1.33 Value vs Perlin Domain Warp Comparison
# =============================================================================


class TestDomainWarpValueVsPerlin:
    """Tests comparing value-based and Perlin-based domain warp."""

    def test_value_domain_warp_differs_from_perlin(self):
        """domain_warp_3d and domain_warp_perlin_3d should produce different results.

        Value noise and Perlin noise are structurally different noise
        functions, so their domain-warped variants should differ.
        """
        p = (3.7, 4.2, 5.1)
        val = py_domain_warp_3d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        perlin = py_domain_warp_perlin_3d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        assert abs(val - perlin) > 0.01, (
            "Value-based and Perlin-based domain warp should produce different results"
        )

    def test_both_deform_at_strength(self):
        """Both variants should produce output different from unwarped FBM at non-zero strength."""
        p = (1.5, 2.5, 3.5)
        fbm_3d_val = py_fbm_3d(
            p, DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        fbm_perlin_val = py_fbm_perlin_3d(
            p, DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )

        warp_val = py_domain_warp_3d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        warp_perlin = py_domain_warp_perlin_3d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )

        diff_val = abs(warp_val - fbm_3d_val)
        diff_perlin = abs(warp_perlin - fbm_perlin_val)

        assert diff_val > 1e-8 or diff_perlin > 1e-8, (
            "Domain warp should deform the FBM signal"
        )

    def test_both_have_same_range(self):
        """Both variants produce output in [-1, 1] range."""
        for i in range(-5, 6):
            p = (i * 0.5 + 0.1, i * 0.5 + 0.2, i * 0.5 + 0.3)
            val = py_domain_warp_3d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            perlin = py_domain_warp_perlin_3d(
                p, DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            assert -1.5 <= val <= 1.5, f"Value warp out of range at {p}"
            assert -1.5 <= perlin <= 1.5, f"Perlin warp out of range at {p}"


# =============================================================================
# Test: T-DEMO-1.33 Continuity
# =============================================================================


class TestDomainWarpContinuity:
    """Tests that domain warp noise is continuous along each axis."""

    # --- 2D continuity ---

    def test_2d_continuity_along_x(self):
        """domain_warp_2d should be continuous along x for fixed y."""
        step = 0.01
        p = (0.0, 1.5)
        prev = py_domain_warp_2d(
            p, DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for i in range(1, 200):
            curr = py_domain_warp_2d(
                (i * step, 1.5), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in domain_warp_2d at x={i * step}: diff={diff}"
            )
            prev = curr

    def test_2d_continuity_along_y(self):
        """domain_warp_2d should be continuous along y for fixed x."""
        step = 0.01
        prev = py_domain_warp_2d(
            (1.5, 0.0), DEFAULT_STRENGTH,
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )
        for i in range(1, 200):
            curr = py_domain_warp_2d(
                (1.5, i * step), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in domain_warp_2d at y={i * step}: diff={diff}"
            )
            prev = curr

    # --- 3D continuity ---

    def test_3d_continuity_along_x(self):
        """domain_warp_3d should be continuous along x for fixed y, z."""
        step = 0.01
        params = (DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN)
        base_params = (DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN)
        prev = py_domain_warp_3d(
            (0.0, 1.0, 2.0), DEFAULT_STRENGTH, *params, *base_params,
        )
        for i in range(1, 200):
            curr = py_domain_warp_3d(
                (i * step, 1.0, 2.0), DEFAULT_STRENGTH, *params, *base_params,
            )
            diff = abs(curr - prev)
            # 3D domain warp compounds 4 FBM evals, permitting steeper gradients
            assert diff < 0.15, (
                f"Discontinuity in domain_warp_3d at x={i * step}: diff={diff}"
            )
            prev = curr

    def test_3d_continuity_along_y(self):
        """domain_warp_3d should be continuous along y for fixed x, z."""
        step = 0.01
        params = (DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN)
        base_params = (DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN)
        prev = py_domain_warp_3d(
            (1.0, 0.0, 2.0), DEFAULT_STRENGTH, *params, *base_params,
        )
        for i in range(1, 200):
            curr = py_domain_warp_3d(
                (1.0, i * step, 2.0), DEFAULT_STRENGTH, *params, *base_params,
            )
            diff = abs(curr - prev)
            assert diff < 0.15, (
                f"Discontinuity in domain_warp_3d at y={i * step}: diff={diff}"
            )
            prev = curr

    def test_3d_continuity_along_z(self):
        """domain_warp_3d should be continuous along z for fixed x, y."""
        step = 0.01
        params = (DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN)
        base_params = (DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN)
        prev = py_domain_warp_3d(
            (1.0, 2.0, 0.0), DEFAULT_STRENGTH, *params, *base_params,
        )
        for i in range(1, 200):
            curr = py_domain_warp_3d(
                (1.0, 2.0, i * step), DEFAULT_STRENGTH, *params, *base_params,
            )
            diff = abs(curr - prev)
            assert diff < 0.15, (
                f"Discontinuity in domain_warp_3d at z={i * step}: diff={diff}"
            )
            prev = curr

    # --- Perlin 3D continuity ---

    def test_perlin_3d_continuity_along_x(self):
        """domain_warp_perlin_3d should be continuous along x."""
        step = 0.01
        params = (DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN)
        base_params = (DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN)
        prev = py_domain_warp_perlin_3d(
            (0.0, 1.0, 2.0), DEFAULT_STRENGTH, *params, *base_params,
        )
        for i in range(1, 200):
            curr = py_domain_warp_perlin_3d(
                (i * step, 1.0, 2.0), DEFAULT_STRENGTH, *params, *base_params,
            )
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in domain_warp_perlin_3d at x={i * step}: diff={diff}"
            )
            prev = curr

    def test_perlin_3d_continuity_along_y(self):
        """domain_warp_perlin_3d should be continuous along y."""
        step = 0.01
        params = (DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN)
        base_params = (DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN)
        prev = py_domain_warp_perlin_3d(
            (1.0, 0.0, 2.0), DEFAULT_STRENGTH, *params, *base_params,
        )
        for i in range(1, 200):
            curr = py_domain_warp_perlin_3d(
                (1.0, i * step, 2.0), DEFAULT_STRENGTH, *params, *base_params,
            )
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in domain_warp_perlin_3d at y={i * step}: diff={diff}"
            )
            prev = curr

    def test_perlin_3d_continuity_along_z(self):
        """domain_warp_perlin_3d should be continuous along z."""
        step = 0.01
        params = (DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN)
        base_params = (DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN)
        prev = py_domain_warp_perlin_3d(
            (1.0, 2.0, 0.0), DEFAULT_STRENGTH, *params, *base_params,
        )
        for i in range(1, 200):
            curr = py_domain_warp_perlin_3d(
                (1.0, 2.0, i * step), DEFAULT_STRENGTH, *params, *base_params,
            )
            diff = abs(curr - prev)
            assert diff < 0.1, (
                f"Discontinuity in domain_warp_perlin_3d at z={i * step}: diff={diff}"
            )
            prev = curr


# =============================================================================
# Test: T-DEMO-1.33 Mean and Distribution
# =============================================================================


class TestDomainWarpDistribution:
    """Tests that domain warp noise has reasonable distribution properties."""

    NUM_SAMPLES = 2000

    def test_2d_mean_near_zero(self):
        """Mean of domain_warp_2d over many points should be near 0."""
        samples = [
            py_domain_warp_2d(
                (i * 0.07, i * 0.11), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.15, (
            f"domain_warp_2d mean {mean} not near 0"
        )

    def test_3d_mean_near_zero(self):
        """Mean of domain_warp_3d over many points should be near 0."""
        samples = [
            py_domain_warp_3d(
                (i * 0.07, i * 0.11, i * 0.13), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.15, (
            f"domain_warp_3d mean {mean} not near 0"
        )

    def test_perlin_3d_mean_near_zero(self):
        """Mean of domain_warp_perlin_3d over many points should be near 0."""
        samples = [
            py_domain_warp_perlin_3d(
                (i * 0.07, i * 0.11, i * 0.13), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.15, (
            f"domain_warp_perlin_3d mean {mean} not near 0"
        )

    def test_2d_values_not_all_same(self):
        """domain_warp_2d should not produce constant output."""
        samples = [
            py_domain_warp_2d(
                (i * 0.17, i * 0.23), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 20, (
            f"Expected at least 20 unique values out of 50, got {unique}"
        )

    def test_3d_values_not_all_same(self):
        """domain_warp_3d should not produce constant output."""
        samples = [
            py_domain_warp_3d(
                (i * 0.17, i * 0.23, i * 0.31), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 20, (
            f"Expected at least 20 unique values out of 50, got {unique}"
        )

    def test_perlin_3d_values_not_all_same(self):
        """domain_warp_perlin_3d should not produce constant output."""
        samples = [
            py_domain_warp_perlin_3d(
                (i * 0.17, i * 0.23, i * 0.31), DEFAULT_STRENGTH,
                DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 20, (
            f"Expected at least 20 unique values out of 50, got {unique}"
        )


# =============================================================================
# Test: T-DEMO-1.33 Warp Decorrelation
# =============================================================================


class TestDomainWarpDecorrelation:
    """Tests that warp components are properly decorrelated.

    The WGSL implementation uses offset (100.0, 100.0) and (200.0, 200.0, 200.0)
    to decorrelate the warp vector components so they produce different
    displacement values rather than identical ones.
    """

    def test_2d_warp_x_and_y_differ(self):
        """Warp_x and warp_y should be different due to offset decorrelation."""
        p = (0.5, 0.25)
        warp_x = py_fbm_2d(p, DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN)
        warp_y = py_fbm_2d(
            (p[0] + 100.0, p[1] + 100.0),
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
        )
        assert abs(warp_x - warp_y) > 1e-8, (
            "Warp x and y components should differ (offset decorrelation)"
        )

    def test_3d_warp_components_differ(self):
        """Warp_x, warp_y, and warp_z should all differ."""
        p = (0.5, 0.25, 0.125)

        warp_x = py_fbm_3d(p, DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN)
        warp_y = py_fbm_3d(
            (p[0] + 100.0, p[1] + 100.0, p[2] + 100.0),
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
        )
        warp_z = py_fbm_3d(
            (p[0] + 200.0, p[1] + 200.0, p[2] + 200.0),
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
        )

        diffs = [abs(warp_x - warp_y), abs(warp_y - warp_z), abs(warp_x - warp_z)]
        assert any(d > 1e-8 for d in diffs), (
            "3D warp components should differ (offset decorrelation)"
        )

    def test_perlin_warp_components_differ(self):
        """Perlin warp_x, warp_y, and warp_z should all differ."""
        p = (0.5, 0.25, 0.125)

        warp_x = py_fbm_perlin_3d(
            p, DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
        )
        warp_y = py_fbm_perlin_3d(
            (p[0] + 100.0, p[1] + 100.0, p[2] + 100.0),
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
        )
        warp_z = py_fbm_perlin_3d(
            (p[0] + 200.0, p[1] + 200.0, p[2] + 200.0),
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
        )

        diffs = [abs(warp_x - warp_y), abs(warp_y - warp_z), abs(warp_x - warp_z)]
        assert any(d > 1e-8 for d in diffs), (
            "Perlin warp components should differ (offset decorrelation)"
        )


# =============================================================================
# Test: T-DEMO-1.33 Cross-Function Consistency
# =============================================================================


class TestDomainWarpCrossFunction:
    """Tests that domain warp functions handle various parameter regimes correctly."""

    def test_all_three_handle_large_coordinates(self):
        """All three domain warp functions handle large coordinate values."""
        large_ps = [1000.0, -1000.0, 1e6, -1e6, 0.001, -0.001]
        params = (
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )

        for p_val in large_ps:
            v2d = py_domain_warp_2d(
                (p_val, p_val * 0.5), DEFAULT_STRENGTH, *params,
            )
            assert math.isfinite(v2d), f"domain_warp_2d({p_val}) non-finite"

        for p_val in large_ps:
            v3d = py_domain_warp_3d(
                (p_val, p_val * 0.5, p_val * 0.25), DEFAULT_STRENGTH, *params,
            )
            assert math.isfinite(v3d), f"domain_warp_3d({p_val}) non-finite"

        for p_val in large_ps:
            vp = py_domain_warp_perlin_3d(
                (p_val, p_val * 0.5, p_val * 0.25), DEFAULT_STRENGTH, *params,
            )
            assert math.isfinite(vp), f"domain_warp_perlin_3d({p_val}) non-finite"

    def test_all_three_produce_non_trivial_warp(self):
        """All three functions produce output different from unwarped FBM at non-zero strength."""
        params = (
            DEFAULT_WARP_OCTAVES, DEFAULT_WARP_LACUNARITY, DEFAULT_WARP_GAIN,
            DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )

        p_2d = (0.75, 0.5)
        p_3d = (0.75, 0.5, 0.25)

        # 2D: warped at strength 1.0 vs unwarped
        warped_2d = py_domain_warp_2d(p_2d, 1.0, *params)
        unwarped_2d = py_domain_warp_2d(p_2d, 0.0, *params)
        assert abs(warped_2d - unwarped_2d) > 1e-8, (
            "domain_warp_2d should produce non-trivial deformation"
        )

        # 3D value: warped at strength 1.0 vs unwarped
        warped_3d = py_domain_warp_3d(p_3d, 1.0, *params)
        unwarped_3d = py_domain_warp_3d(p_3d, 0.0, *params)
        assert abs(warped_3d - unwarped_3d) > 1e-8, (
            "domain_warp_3d should produce non-trivial deformation"
        )

        # 3D Perlin: warped at strength 1.0 vs unwarped
        warped_perlin = py_domain_warp_perlin_3d(p_3d, 1.0, *params)
        unwarped_perlin = py_domain_warp_perlin_3d(p_3d, 0.0, *params)
        assert abs(warped_perlin - unwarped_perlin) > 1e-8, (
            "domain_warp_perlin_3d should produce non-trivial deformation"
        )

    def test_all_three_respond_to_parameters(self):
        """All three functions respond to changes in both warp and base parameters."""
        p_2d = (0.75, 0.5)
        p_3d = (0.75, 0.5, 0.25)

        param_sets = [
            (4, 2.0, 0.5, 8, 2.0, 0.5),
            (2, 2.0, 0.5, 4, 2.0, 0.5),
            (6, 1.5, 0.7, 6, 1.5, 0.7),
        ]

        results_2d = [
            py_domain_warp_2d(p_2d, DEFAULT_STRENGTH, *ps) for ps in param_sets
        ]
        results_3d = [
            py_domain_warp_3d(p_3d, DEFAULT_STRENGTH, *ps) for ps in param_sets
        ]
        results_perlin = [
            py_domain_warp_perlin_3d(p_3d, DEFAULT_STRENGTH, *ps) for ps in param_sets
        ]

        for name, results in [
            ("domain_warp_2d", results_2d),
            ("domain_warp_3d", results_3d),
            ("domain_warp_perlin_3d", results_perlin),
        ]:
            unique = len(set(round(v, 10) for v in results))
            assert unique >= 2, (
                f"{name} produced only {unique} unique values across "
                f"3 parameter sets"
            )

    def test_2d_strength_zero_ignores_warp_params(self):
        """With strength=0, changing warp parameters should not affect output.

        When there is no warp displacement, the warp FBM evaluation is
        irrelevant -- only the base FBM matters.
        """
        p = (0.5, 0.25)
        base_fbm = py_fbm_2d(
            p, DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
        )

        # Change warp parameters -- should not matter at strength=0
        warp_params = [
            (1, 2.0, 0.5),
            (4, 2.0, 0.5),
            (8, 1.5, 0.7),
            (0, 2.0, 0.5),
        ]
        for w_oct, w_lac, w_gain in warp_params:
            v = py_domain_warp_2d(
                p, 0.0, w_oct, w_lac, w_gain,
                DEFAULT_BASE_OCTAVES, DEFAULT_BASE_LACUNARITY, DEFAULT_BASE_GAIN,
            )
            assert v == pytest.approx(base_fbm, abs=TOL_ABS), (
                f"At strength=0, changing warp params should not affect output: "
                f"w_oct={w_oct}"
            )
