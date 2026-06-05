"""Material DSL builtins: noise functions, math utilities, and color conversion.

This module provides WGSL helper functions that can be called from material
surface() bodies. When used in a DSL surface shader, these functions emit
the corresponding WGSL code and ensure the helper functions are included
in the compiled shader.

Categories:
    - Noise: value, perlin, simplex, worley, fbm
    - Math: lerp, smoothstep, normalize, reflect, refract, clamp, saturate, mix
    - Color: rgb_to_hsv, hsv_to_rgb, linear_to_srgb, srgb_to_linear, tonemap

Example usage in a material::

    class NoisyMaterial(Material, metaclass=MaterialMeta):
        @surface
        def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
            noise_val = perlin_noise(ctx.uv * 10.0)
            out.base_color = mix(Vec3(0.2, 0.1, 0.0), Vec3(0.8, 0.6, 0.4), noise_val)
            out.roughness = saturate(noise_val * 0.5 + 0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

__all__ = [
    # Registry
    "BUILTIN_REGISTRY",
    "get_builtin_wgsl",
    "get_required_builtins",
    # Noise functions
    "value_noise",
    "perlin_noise",
    "simplex_noise",
    "worley_noise",
    "fbm",
    # Math utilities (these are WGSL built-ins, provided for documentation)
    "lerp",
    "smoothstep",
    "saturate",
    # Color conversion
    "rgb_to_hsv",
    "hsv_to_rgb",
    "linear_to_srgb",
    "srgb_to_linear",
    "tonemap_reinhard",
    "tonemap_aces",
]


# =============================================================================
# WGSL HELPER FUNCTION DEFINITIONS
# =============================================================================

# Hash functions used by noise algorithms
WGSL_HASH = """\
// Hash functions for procedural noise
fn hash11(p: f32) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 * (p3 + 33.33);
    return fract((p3 + p3) * p3);
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract(vec2<f32>((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
}

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract(vec3<f32>((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y, (p3.y + p3.z) * p3.x));
}
"""

# Value noise
WGSL_VALUE_NOISE = """\
// Value noise (hash-based, 2D)
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Quintic interpolation for smoother gradients
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Value noise 3D
fn value_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    return mix(
        mix(
            mix(hash31(i + vec3<f32>(0.0, 0.0, 0.0)), hash31(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash31(i + vec3<f32>(0.0, 1.0, 0.0)), hash31(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash31(i + vec3<f32>(0.0, 0.0, 1.0)), hash31(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash31(i + vec3<f32>(0.0, 1.0, 1.0)), hash31(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}
"""

# Perlin gradient noise
WGSL_PERLIN_NOISE = """\
// Gradient hash for Perlin noise
fn gradient_hash(p: vec2<f32>) -> vec2<f32> {
    let h = hash22(p) * 6.283185307;
    return vec2<f32>(cos(h.x), sin(h.x));
}

fn gradient_hash_3d(p: vec3<f32>) -> vec3<f32> {
    let h = hash33(p);
    let theta = h.x * 6.283185307;
    let phi = acos(2.0 * h.y - 1.0);
    return vec3<f32>(
        sin(phi) * cos(theta),
        sin(phi) * sin(theta),
        cos(phi)
    );
}

// Perlin gradient noise (2D)
fn perlin_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Quintic interpolation
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let g00 = gradient_hash(i + vec2<f32>(0.0, 0.0));
    let g10 = gradient_hash(i + vec2<f32>(1.0, 0.0));
    let g01 = gradient_hash(i + vec2<f32>(0.0, 1.0));
    let g11 = gradient_hash(i + vec2<f32>(1.0, 1.0));

    let d00 = dot(g00, f - vec2<f32>(0.0, 0.0));
    let d10 = dot(g10, f - vec2<f32>(1.0, 0.0));
    let d01 = dot(g01, f - vec2<f32>(0.0, 1.0));
    let d11 = dot(g11, f - vec2<f32>(1.0, 1.0));

    return mix(mix(d00, d10, u.x), mix(d01, d11, u.x), u.y) * 0.5 + 0.5;
}

// Perlin gradient noise (3D)
fn perlin_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    var result = 0.0;
    for (var z = 0; z <= 1; z = z + 1) {
        for (var y = 0; y <= 1; y = y + 1) {
            for (var x = 0; x <= 1; x = x + 1) {
                let corner = vec3<f32>(f32(x), f32(y), f32(z));
                let g = gradient_hash_3d(i + corner);
                let d = dot(g, f - corner);
                let w = select(1.0 - u, u, corner > vec3<f32>(0.5, 0.5, 0.5));
                result = result + d * w.x * w.y * w.z;
            }
        }
    }

    return result * 0.5 + 0.5;
}
"""

# Simplex noise
WGSL_SIMPLEX_NOISE = """\
// Simplex noise constants
const SIMPLEX_SKEW_2D: f32 = 0.366025403784; // (sqrt(3) - 1) / 2
const SIMPLEX_UNSKEW_2D: f32 = 0.211324865405; // (3 - sqrt(3)) / 6

// Simplex noise (2D)
fn simplex_noise(p: vec2<f32>) -> f32 {
    // Skew input space to simplex grid
    let s = (p.x + p.y) * SIMPLEX_SKEW_2D;
    let i = floor(p + s);
    let t = (i.x + i.y) * SIMPLEX_UNSKEW_2D;
    let x0 = p - (i - t);

    // Determine which simplex we're in
    var i1: vec2<f32>;
    if (x0.x > x0.y) {
        i1 = vec2<f32>(1.0, 0.0);
    } else {
        i1 = vec2<f32>(0.0, 1.0);
    }

    let x1 = x0 - i1 + SIMPLEX_UNSKEW_2D;
    let x2 = x0 - 1.0 + 2.0 * SIMPLEX_UNSKEW_2D;

    // Gradient contributions
    var n0: f32 = 0.0;
    var n1: f32 = 0.0;
    var n2: f32 = 0.0;

    var t0 = 0.5 - dot(x0, x0);
    if (t0 > 0.0) {
        t0 = t0 * t0;
        let g0 = gradient_hash(i);
        n0 = t0 * t0 * dot(g0, x0);
    }

    var t1 = 0.5 - dot(x1, x1);
    if (t1 > 0.0) {
        t1 = t1 * t1;
        let g1 = gradient_hash(i + i1);
        n1 = t1 * t1 * dot(g1, x1);
    }

    var t2 = 0.5 - dot(x2, x2);
    if (t2 > 0.0) {
        t2 = t2 * t2;
        let g2 = gradient_hash(i + vec2<f32>(1.0, 1.0));
        n2 = t2 * t2 * dot(g2, x2);
    }

    // Scale to [0, 1]
    return (n0 + n1 + n2) * 70.0 * 0.5 + 0.5;
}
"""

# Worley (cellular/Voronoi) noise
WGSL_WORLEY_NOISE = """\
// Worley/cellular noise (2D) - returns (distance to closest, distance to second closest)
fn worley_noise(p: vec2<f32>) -> vec2<f32> {
    let n = floor(p);
    let f = fract(p);

    var d1 = 8.0;
    var d2 = 8.0;

    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let g = vec2<f32>(f32(i), f32(j));
            let o = hash22(n + g);
            let r = g + o - f;
            let d = dot(r, r);

            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }

    return vec2<f32>(sqrt(d1), sqrt(d2));
}

// Worley noise (3D)
fn worley_noise_3d(p: vec3<f32>) -> vec2<f32> {
    let n = floor(p);
    let f = fract(p);

    var d1 = 8.0;
    var d2 = 8.0;

    for (var k = -1; k <= 1; k = k + 1) {
        for (var j = -1; j <= 1; j = j + 1) {
            for (var i = -1; i <= 1; i = i + 1) {
                let g = vec3<f32>(f32(i), f32(j), f32(k));
                let o = hash33(n + g);
                let r = g + o - f;
                let d = dot(r, r);

                if (d < d1) {
                    d2 = d1;
                    d1 = d;
                } else if (d < d2) {
                    d2 = d;
                }
            }
        }
    }

    return vec2<f32>(sqrt(d1), sqrt(d2));
}
"""

# Fractional Brownian Motion (FBM)
WGSL_FBM = """\
// Fractional Brownian Motion using Perlin noise
fn fbm(p: vec2<f32>, octaves: i32, lacunarity: f32, gain: f32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < octaves; i = i + 1) {
        value = value + amplitude * (perlin_noise(pos * frequency) * 2.0 - 1.0);
        amplitude = amplitude * gain;
        frequency = frequency * lacunarity;
    }

    return value * 0.5 + 0.5;
}

// FBM with customizable noise function (using value noise)
fn fbm_value(p: vec2<f32>, octaves: i32, lacunarity: f32, gain: f32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < octaves; i = i + 1) {
        value = value + amplitude * (value_noise(pos * frequency) * 2.0 - 1.0);
        amplitude = amplitude * gain;
        frequency = frequency * lacunarity;
    }

    return value * 0.5 + 0.5;
}

// Turbulence (absolute value FBM)
fn turbulence(p: vec2<f32>, octaves: i32, lacunarity: f32, gain: f32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < octaves; i = i + 1) {
        value = value + amplitude * abs(perlin_noise(pos * frequency) * 2.0 - 1.0);
        amplitude = amplitude * gain;
        frequency = frequency * lacunarity;
    }

    return value;
}
"""

# Color space conversion functions
WGSL_COLOR_CONVERSION = """\
// RGB to HSV conversion
fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let c_max = max(max(rgb.r, rgb.g), rgb.b);
    let c_min = min(min(rgb.r, rgb.g), rgb.b);
    let delta = c_max - c_min;

    var h: f32 = 0.0;
    var s: f32 = 0.0;
    let v = c_max;

    if (delta > 0.00001) {
        s = delta / c_max;

        if (c_max == rgb.r) {
            h = (rgb.g - rgb.b) / delta;
            if (rgb.g < rgb.b) {
                h = h + 6.0;
            }
        } else if (c_max == rgb.g) {
            h = (rgb.b - rgb.r) / delta + 2.0;
        } else {
            h = (rgb.r - rgb.g) / delta + 4.0;
        }
        h = h / 6.0;
    }

    return vec3<f32>(h, s, v);
}

// HSV to RGB conversion
fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x * 6.0;
    let s = hsv.y;
    let v = hsv.z;

    let i = floor(h);
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    let idx = i32(i) % 6;

    if (idx == 0) {
        return vec3<f32>(v, t, p);
    } else if (idx == 1) {
        return vec3<f32>(q, v, p);
    } else if (idx == 2) {
        return vec3<f32>(p, v, t);
    } else if (idx == 3) {
        return vec3<f32>(p, q, v);
    } else if (idx == 4) {
        return vec3<f32>(t, p, v);
    } else {
        return vec3<f32>(v, p, q);
    }
}

// sRGB to linear color space (gamma decode)
fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = step(srgb, vec3<f32>(0.04045));
    let low = srgb / 12.92;
    let high = pow((srgb + 0.055) / 1.055, vec3<f32>(2.4));
    return mix(high, low, cutoff);
}

// Linear to sRGB color space (gamma encode)
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = step(linear, vec3<f32>(0.0031308));
    let low = linear * 12.92;
    let high = 1.055 * pow(linear, vec3<f32>(1.0 / 2.4)) - 0.055;
    return mix(high, low, cutoff);
}
"""

# Tone mapping functions
WGSL_TONEMAP = """\
// Reinhard tonemapping
fn tonemap_reinhard(hdr: vec3<f32>) -> vec3<f32> {
    return hdr / (hdr + vec3<f32>(1.0));
}

// Extended Reinhard with white point
fn tonemap_reinhard_white(hdr: vec3<f32>, white_point: f32) -> vec3<f32> {
    let white_sq = white_point * white_point;
    let numerator = hdr * (1.0 + hdr / white_sq);
    return numerator / (1.0 + hdr);
}

// ACES filmic tonemapping (approximation)
fn tonemap_aces(hdr: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e));
}

// Uncharted 2 filmic tonemapping
fn tonemap_uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn tonemap_uncharted2(hdr: vec3<f32>) -> vec3<f32> {
    let exposure_bias = 2.0;
    let curr = tonemap_uncharted2_partial(hdr * exposure_bias);
    let W = vec3<f32>(11.2);
    let white_scale = vec3<f32>(1.0) / tonemap_uncharted2_partial(W);
    return curr * white_scale;
}

// AgX tonemapping (modern, perceptually accurate)
fn tonemap_agx(hdr: vec3<f32>) -> vec3<f32> {
    // AgX base compression
    let compressed = max(hdr, vec3<f32>(1e-10));
    let log_color = clamp(log2(compressed), vec3<f32>(-10.0), vec3<f32>(6.5));
    let normalized = (log_color + 10.0) / 16.5;

    // AgX contrast curve
    let x = normalized;
    let x2 = x * x;
    let x4 = x2 * x2;
    let result = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;

    return saturate(result);
}
"""

# Math utilities (supplements WGSL built-ins)
WGSL_MATH_UTILS = """\
// Saturate (clamp to [0, 1]) - WGSL built-in, but provided for compatibility
// Note: WGSL already has saturate() as a built-in

// Remap value from one range to another
fn remap(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    let t = (value - from_min) / (from_max - from_min);
    return mix(to_min, to_max, t);
}

// Inverse lerp (get t from value)
fn inverse_lerp(a: f32, b: f32, value: f32) -> f32 {
    return (value - a) / (b - a);
}

// Smooth minimum (soft blend between two values)
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// Smooth maximum
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    return -smooth_min(-a, -b, k);
}

// Quintic smoothstep (smoother than smoothstep)
fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Gain function (contrast adjustment in [0, 1])
fn gain(x: f32, k: f32) -> f32 {
    let a = 0.5 * pow(2.0 * select(1.0 - x, x, x < 0.5), k);
    return select(1.0 - a, a, x < 0.5);
}

// Bias function (shifts midpoint)
fn bias(x: f32, b: f32) -> f32 {
    return pow(x, log(b) / log(0.5));
}
"""


# =============================================================================
# BUILTIN REGISTRY
# =============================================================================

@dataclass
class BuiltinFunction:
    """Descriptor for a DSL builtin function."""
    name: str  # Python function name
    wgsl_name: str  # WGSL function name
    wgsl_source: str  # WGSL source code
    dependencies: tuple[str, ...]  # Other builtins this depends on
    return_type: str  # WGSL return type
    description: str  # Documentation


# Registry of all builtin functions
BUILTIN_REGISTRY: Dict[str, BuiltinFunction] = {}


def _register(name: str, wgsl_name: str, wgsl_source: str,
              dependencies: tuple[str, ...] = (),
              return_type: str = "f32",
              description: str = "") -> None:
    """Register a builtin function."""
    BUILTIN_REGISTRY[name] = BuiltinFunction(
        name=name,
        wgsl_name=wgsl_name,
        wgsl_source=wgsl_source,
        dependencies=dependencies,
        return_type=return_type,
        description=description,
    )


# Register all builtins
_register("hash", "hash21", WGSL_HASH, (), "f32", "Hash function")
_register("value_noise", "value_noise", WGSL_VALUE_NOISE, ("hash",), "f32",
          "Value noise (2D)")
_register("perlin_noise", "perlin_noise", WGSL_PERLIN_NOISE, ("hash",), "f32",
          "Perlin gradient noise (2D)")
_register("simplex_noise", "simplex_noise", WGSL_SIMPLEX_NOISE, ("hash",), "f32",
          "Simplex noise (2D)")
_register("worley_noise", "worley_noise", WGSL_WORLEY_NOISE, ("hash",), "vec2<f32>",
          "Worley/cellular noise (2D)")
_register("fbm", "fbm", WGSL_FBM, ("perlin_noise",), "f32",
          "Fractional Brownian Motion")
_register("turbulence", "turbulence", WGSL_FBM, ("perlin_noise",), "f32",
          "Turbulence (absolute value FBM)")
_register("rgb_to_hsv", "rgb_to_hsv", WGSL_COLOR_CONVERSION, (), "vec3<f32>",
          "RGB to HSV conversion")
_register("hsv_to_rgb", "hsv_to_rgb", WGSL_COLOR_CONVERSION, (), "vec3<f32>",
          "HSV to RGB conversion")
_register("srgb_to_linear", "srgb_to_linear", WGSL_COLOR_CONVERSION, (), "vec3<f32>",
          "sRGB to linear color space")
_register("linear_to_srgb", "linear_to_srgb", WGSL_COLOR_CONVERSION, (), "vec3<f32>",
          "Linear to sRGB color space")
_register("tonemap_reinhard", "tonemap_reinhard", WGSL_TONEMAP, (), "vec3<f32>",
          "Reinhard tonemapping")
_register("tonemap_aces", "tonemap_aces", WGSL_TONEMAP, (), "vec3<f32>",
          "ACES filmic tonemapping")
_register("tonemap_uncharted2", "tonemap_uncharted2", WGSL_TONEMAP, (), "vec3<f32>",
          "Uncharted 2 tonemapping")
_register("tonemap_agx", "tonemap_agx", WGSL_TONEMAP, (), "vec3<f32>",
          "AgX tonemapping")
_register("remap", "remap", WGSL_MATH_UTILS, (), "f32",
          "Remap value from one range to another")
_register("inverse_lerp", "inverse_lerp", WGSL_MATH_UTILS, (), "f32",
          "Inverse lerp")
_register("smooth_min", "smooth_min", WGSL_MATH_UTILS, (), "f32",
          "Smooth minimum")
_register("smooth_max", "smooth_max", WGSL_MATH_UTILS, (), "f32",
          "Smooth maximum")
_register("smootherstep", "smootherstep", WGSL_MATH_UTILS, (), "f32",
          "Quintic smoothstep")


def get_required_builtins(function_names: Set[str]) -> str:
    """Get all WGSL source code for the required builtin functions.

    Resolves dependencies recursively and returns concatenated WGSL source.

    Args:
        function_names: Set of builtin function names used in the shader.

    Returns:
        WGSL source code for all required builtins.
    """
    required: Set[str] = set()
    to_process = list(function_names)

    # Resolve all dependencies
    while to_process:
        name = to_process.pop()
        if name in required:
            continue
        if name not in BUILTIN_REGISTRY:
            continue

        required.add(name)
        builtin = BUILTIN_REGISTRY[name]
        for dep in builtin.dependencies:
            if dep not in required:
                to_process.append(dep)

    # Collect unique WGSL sources (deduped since some share source blocks)
    sources: Set[str] = set()
    for name in required:
        builtin = BUILTIN_REGISTRY[name]
        sources.add(builtin.wgsl_source)

    return "\n\n".join(sorted(sources))


def get_builtin_wgsl(name: str) -> str:
    """Get the WGSL source code for a single builtin function.

    Args:
        name: Builtin function name.

    Returns:
        WGSL source code, or empty string if not found.
    """
    if name in BUILTIN_REGISTRY:
        return BUILTIN_REGISTRY[name].wgsl_source
    return ""


# =============================================================================
# PYTHON STUBS (for type checking and documentation)
# =============================================================================

def value_noise(p: tuple[float, float]) -> float:
    """Value noise at 2D position.

    Args:
        p: 2D position (x, y)

    Returns:
        Noise value in [0, 1]
    """
    return 0.5


def perlin_noise(p: tuple[float, float]) -> float:
    """Perlin gradient noise at 2D position.

    Args:
        p: 2D position (x, y)

    Returns:
        Noise value in [0, 1]
    """
    return 0.5


def simplex_noise(p: tuple[float, float]) -> float:
    """Simplex noise at 2D position.

    Args:
        p: 2D position (x, y)

    Returns:
        Noise value in [0, 1]
    """
    return 0.5


def worley_noise(p: tuple[float, float]) -> tuple[float, float]:
    """Worley/cellular noise at 2D position.

    Args:
        p: 2D position (x, y)

    Returns:
        (distance to nearest cell, distance to second nearest)
    """
    return (0.5, 0.7)


def fbm(p: tuple[float, float], octaves: int = 4,
        lacunarity: float = 2.0, gain: float = 0.5) -> float:
    """Fractional Brownian Motion noise.

    Args:
        p: 2D position (x, y)
        octaves: Number of noise octaves
        lacunarity: Frequency multiplier per octave
        gain: Amplitude multiplier per octave

    Returns:
        FBM noise value in [0, 1]
    """
    return 0.5


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation. Maps to WGSL mix()."""
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite smoothstep interpolation."""
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def saturate(x: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, x))


def rgb_to_hsv(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert RGB to HSV color space.

    Args:
        rgb: RGB color (r, g, b) in [0, 1]

    Returns:
        HSV color (h, s, v) in [0, 1]
    """
    return (0.0, 1.0, 1.0)


def hsv_to_rgb(hsv: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert HSV to RGB color space.

    Args:
        hsv: HSV color (h, s, v) in [0, 1]

    Returns:
        RGB color (r, g, b) in [0, 1]
    """
    return (1.0, 0.0, 0.0)


def linear_to_srgb(linear: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert linear color to sRGB (gamma encode)."""
    return linear


def srgb_to_linear(srgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert sRGB to linear color (gamma decode)."""
    return srgb


def tonemap_reinhard(hdr: tuple[float, float, float]) -> tuple[float, float, float]:
    """Apply Reinhard tonemapping."""
    return hdr


def tonemap_aces(hdr: tuple[float, float, float]) -> tuple[float, float, float]:
    """Apply ACES filmic tonemapping."""
    return hdr
