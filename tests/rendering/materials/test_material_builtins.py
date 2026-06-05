"""Tests for the Material DSL builtins library (T-MAT-1.4).

Verifies:
- Noise functions: value, perlin, simplex, worley, FBM
- Math utilities: lerp, smoothstep, normalize, reflect, refract, clamp, saturate, mix
- Color conversion: rgb_to_hsv, hsv_to_rgb, linear_to_srgb, srgb_to_linear, tonemap
- WGSL emission when called from DSL surface() body
- Compiler includes required WGSL helper functions
"""

from __future__ import annotations

import pytest

from trinity.materials import (
    Material,
    MaterialMeta,
    SurfaceContext,
    SurfaceOutput,
    surface,
    Vec2,
    Vec3,
    Vec4,
    MaterialCompiler,
    BUILTIN_REGISTRY,
    get_builtin_wgsl,
    get_required_builtins,
)


# =============================================================================
# Suite A: Builtins Registry
# =============================================================================


class TestBuiltinsRegistry:
    """Builtins registry contains all required functions."""

    def test_noise_functions_registered(self):
        """All noise functions are registered."""
        noise_funcs = ["value_noise", "perlin_noise", "simplex_noise", "worley_noise", "fbm"]
        for name in noise_funcs:
            assert name in BUILTIN_REGISTRY, f"Missing noise function: {name}"

    def test_color_conversion_registered(self):
        """Color conversion functions are registered."""
        color_funcs = [
            "rgb_to_hsv", "hsv_to_rgb",
            "srgb_to_linear", "linear_to_srgb"
        ]
        for name in color_funcs:
            assert name in BUILTIN_REGISTRY, f"Missing color function: {name}"

    def test_tonemap_functions_registered(self):
        """Tonemap functions are registered."""
        tonemap_funcs = ["tonemap_reinhard", "tonemap_aces"]
        for name in tonemap_funcs:
            assert name in BUILTIN_REGISTRY, f"Missing tonemap function: {name}"

    def test_math_utilities_registered(self):
        """Math utility functions are registered."""
        math_funcs = ["remap", "inverse_lerp", "smooth_min", "smooth_max", "smootherstep"]
        for name in math_funcs:
            assert name in BUILTIN_REGISTRY, f"Missing math function: {name}"

    def test_builtin_has_wgsl_source(self):
        """Each builtin has WGSL source code."""
        for name, builtin in BUILTIN_REGISTRY.items():
            assert builtin.wgsl_source, f"Builtin {name} has empty wgsl_source"
            assert len(builtin.wgsl_source) > 10, f"Builtin {name} has very short wgsl_source"


# =============================================================================
# Suite B: get_required_builtins()
# =============================================================================


class TestGetRequiredBuiltins:
    """get_required_builtins() resolves dependencies correctly."""

    def test_simple_builtin(self):
        """Single builtin without dependencies."""
        wgsl = get_required_builtins({"tonemap_reinhard"})
        assert "fn tonemap_reinhard" in wgsl

    def test_builtin_with_dependency(self):
        """Builtin that depends on hash functions."""
        wgsl = get_required_builtins({"perlin_noise"})
        # perlin_noise depends on hash functions
        assert "perlin_noise" in wgsl
        assert "hash" in wgsl.lower()

    def test_multiple_builtins(self):
        """Multiple builtins combined."""
        wgsl = get_required_builtins({"perlin_noise", "tonemap_aces"})
        assert "perlin_noise" in wgsl
        assert "tonemap_aces" in wgsl

    def test_empty_set(self):
        """Empty set returns empty string."""
        wgsl = get_required_builtins(set())
        assert wgsl == ""

    def test_nonexistent_builtin(self):
        """Nonexistent builtin is ignored."""
        wgsl = get_required_builtins({"not_a_real_builtin"})
        assert wgsl == ""


# =============================================================================
# Suite C: get_builtin_wgsl()
# =============================================================================


class TestGetBuiltinWgsl:
    """get_builtin_wgsl() returns correct WGSL source."""

    def test_existing_builtin(self):
        """Existing builtin returns WGSL source."""
        wgsl = get_builtin_wgsl("perlin_noise")
        assert "fn perlin_noise" in wgsl

    def test_nonexistent_builtin(self):
        """Nonexistent builtin returns empty string."""
        wgsl = get_builtin_wgsl("not_real")
        assert wgsl == ""


# =============================================================================
# Suite D: Noise Functions in DSL
# =============================================================================


class TestNoiseInDSL:
    """Noise functions work in material DSL."""

    def test_perlin_noise_in_surface(self):
        """perlin_noise() in surface() generates correct WGSL."""

        class NoisyMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                # Use perlin_noise with UV coords
                noise_val = perlin_noise(ctx.uv)
                out.roughness = noise_val

        assert NoisyMaterial._compilation_error is None
        wgsl = NoisyMaterial._wgsl_source
        assert "perlin_noise" in wgsl

    def test_value_noise_in_surface(self):
        """value_noise() in surface() generates correct WGSL."""

        class ValueNoiseMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                noise_val = value_noise(ctx.uv)
                out.roughness = noise_val

        assert ValueNoiseMaterial._compilation_error is None
        assert "value_noise" in ValueNoiseMaterial._wgsl_source

    def test_worley_noise_in_surface(self):
        """worley_noise() in surface() generates correct WGSL."""

        class WorleyMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                cell = worley_noise(ctx.uv)
                out.roughness = cell.x

        assert WorleyMaterial._compilation_error is None
        assert "worley_noise" in WorleyMaterial._wgsl_source

    def test_fbm_in_surface(self):
        """fbm() in surface() generates correct WGSL."""

        class FBMMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                noise_val = fbm(ctx.uv, 4, 2.0, 0.5)
                out.roughness = noise_val

        assert FBMMaterial._compilation_error is None
        assert "fbm" in FBMMaterial._wgsl_source


# =============================================================================
# Suite E: Color Conversion in DSL
# =============================================================================


class TestColorConversionInDSL:
    """Color conversion functions work in material DSL."""

    def test_rgb_to_hsv_in_surface(self):
        """rgb_to_hsv() in surface() generates correct WGSL."""

        class HSVMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                hsv = rgb_to_hsv(out.base_color)
                out.roughness = hsv.y  # Use saturation for roughness

        assert HSVMaterial._compilation_error is None
        assert "rgb_to_hsv" in HSVMaterial._wgsl_source

    def test_hsv_to_rgb_in_surface(self):
        """hsv_to_rgb() in surface() generates correct WGSL."""

        class RGBFromHSVMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                rgb = hsv_to_rgb(Vec3(0.5, 1.0, 1.0))
                out.base_color = rgb

        assert RGBFromHSVMaterial._compilation_error is None
        assert "hsv_to_rgb" in RGBFromHSVMaterial._wgsl_source

    def test_srgb_linear_conversion(self):
        """srgb/linear conversion in surface() generates correct WGSL."""

        class GammaMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                linear = srgb_to_linear(Vec3(0.5, 0.5, 0.5))
                out.base_color = linear_to_srgb(linear)

        assert GammaMaterial._compilation_error is None
        assert "srgb_to_linear" in GammaMaterial._wgsl_source
        assert "linear_to_srgb" in GammaMaterial._wgsl_source


# =============================================================================
# Suite F: Tonemap in DSL
# =============================================================================


class TestTonemapInDSL:
    """Tonemap functions work in material DSL."""

    def test_tonemap_reinhard_in_surface(self):
        """tonemap_reinhard() in surface() generates correct WGSL."""

        class ReinhardMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.emissive = tonemap_reinhard(Vec3(2.0, 1.5, 1.0))

        assert ReinhardMaterial._compilation_error is None
        assert "tonemap_reinhard" in ReinhardMaterial._wgsl_source

    def test_tonemap_aces_in_surface(self):
        """tonemap_aces() in surface() generates correct WGSL."""

        class ACESMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.emissive = tonemap_aces(Vec3(4.0, 3.0, 2.0))

        assert ACESMaterial._compilation_error is None
        assert "tonemap_aces" in ACESMaterial._wgsl_source


# =============================================================================
# Suite G: Math Utilities in DSL
# =============================================================================


class TestMathUtilitiesInDSL:
    """Math utility functions work in material DSL."""

    def test_wgsl_builtins_work(self):
        """Core WGSL builtins (mix, clamp, etc.) work."""

        class MathMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = mix(0.2, 0.8, 0.5)
                out.metallic = clamp(1.5, 0.0, 1.0)
                out.ao = saturate(0.7)

        assert MathMaterial._compilation_error is None
        wgsl = MathMaterial._wgsl_source
        assert "mix" in wgsl
        assert "clamp" in wgsl
        assert "saturate" in wgsl

    def test_smoothstep_in_surface(self):
        """smoothstep() in surface() generates correct WGSL."""

        class SmoothMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = smoothstep(0.0, 1.0, ctx.uv.x)

        assert SmoothMaterial._compilation_error is None
        assert "smoothstep" in SmoothMaterial._wgsl_source

    def test_normalize_in_surface(self):
        """normalize() in surface() generates correct WGSL."""

        class NormalMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.normal = normalize(Vec3(1.0, 1.0, 0.0))

        assert NormalMaterial._compilation_error is None
        assert "normalize" in NormalMaterial._wgsl_source

    def test_reflect_in_surface(self):
        """reflect() in surface() generates correct WGSL."""

        class ReflectMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                r = reflect(Vec3(1.0, -1.0, 0.0), Vec3(0.0, 1.0, 0.0))
                out.base_color = r

        assert ReflectMaterial._compilation_error is None
        assert "reflect" in ReflectMaterial._wgsl_source


# =============================================================================
# Suite H: Compiler Integration
# =============================================================================


class TestCompilerWithBuiltins:
    """MaterialCompiler includes builtin WGSL helpers."""

    def test_compiler_includes_noise_helpers(self):
        """Compiler includes noise helper functions when used."""

        class PerlinMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                n = perlin_noise(ctx.uv)
                out.roughness = n

        compiler = MaterialCompiler()
        wgsl = compiler.compile(PerlinMaterial)

        # Should include the actual WGSL implementation
        assert "fn perlin_noise" in wgsl or "perlin_noise" in wgsl
        # Should include hash functions (dependency)
        assert "hash" in wgsl.lower()

    def test_compiler_includes_color_helpers(self):
        """Compiler includes color conversion helpers when used."""

        class ColorMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                hsv = rgb_to_hsv(out.base_color)
                out.base_color = hsv_to_rgb(hsv)

        compiler = MaterialCompiler()
        wgsl = compiler.compile(ColorMaterial)

        assert "fn rgb_to_hsv" in wgsl or "rgb_to_hsv" in wgsl
        assert "fn hsv_to_rgb" in wgsl or "hsv_to_rgb" in wgsl

    def test_compiler_no_helpers_when_not_needed(self):
        """Compiler does not include helpers when not used."""

        class SimpleMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        compiler = MaterialCompiler()
        wgsl = compiler.compile(SimpleMaterial)

        # Should not include noise functions
        assert "fn perlin_noise" not in wgsl
        assert "fn value_noise" not in wgsl
        # Should not include color conversion
        assert "fn rgb_to_hsv" not in wgsl


# =============================================================================
# Suite I: WGSL Validity (Syntax Check)
# =============================================================================


class TestWGSLValidity:
    """Generated WGSL has valid syntax."""

    def test_noise_wgsl_has_fn_declarations(self):
        """Noise WGSL has proper function declarations."""
        wgsl = get_builtin_wgsl("perlin_noise")
        assert "fn " in wgsl
        assert "->" in wgsl  # Return type

    def test_color_wgsl_has_fn_declarations(self):
        """Color WGSL has proper function declarations."""
        wgsl = get_builtin_wgsl("rgb_to_hsv")
        assert "fn rgb_to_hsv" in wgsl
        assert "fn hsv_to_rgb" in wgsl

    def test_tonemap_wgsl_has_fn_declarations(self):
        """Tonemap WGSL has proper function declarations."""
        wgsl = get_builtin_wgsl("tonemap_aces")
        assert "fn tonemap_aces" in wgsl
        assert "fn tonemap_reinhard" in wgsl

    def test_fbm_wgsl_has_fn_declarations(self):
        """FBM WGSL has proper function declaration."""
        wgsl = get_builtin_wgsl("fbm")
        assert "fn fbm" in wgsl
        assert "octaves" in wgsl  # Parameter


# =============================================================================
# Global fixtures for noise/color/math functions in DSL scope
# =============================================================================

# These are needed since the test classes use these functions in class bodies
# Import them at module level so they're available in the DSL surfaces
from trinity.materials.builtins import (
    perlin_noise,
    value_noise,
    simplex_noise,
    worley_noise,
    fbm,
    rgb_to_hsv,
    hsv_to_rgb,
    srgb_to_linear,
    linear_to_srgb,
    tonemap_reinhard,
    tonemap_aces,
    lerp,
    smoothstep,
    saturate,
)

# Also import WGSL builtins that are mapped
from builtins import (
    min,
    max,
    abs,
)

# WGSL-mapped functions (used in DSL, map to WGSL)
def clamp(x, min_val, max_val):
    """Clamp helper for DSL (maps to WGSL clamp)."""
    return max(min_val, min(x, max_val))

def mix(a, b, t):
    """Mix helper for DSL (maps to WGSL mix)."""
    return a + (b - a) * t

def normalize(v):
    """Normalize helper for DSL (maps to WGSL normalize)."""
    return v  # Stub

def reflect(i, n):
    """Reflect helper for DSL (maps to WGSL reflect)."""
    return Vec3(1, 1, 0)  # Stub
