"""Blend mode variant logic for GPU blend state configuration.

This module provides blend-specific behavior for all 5 blend modes:
- OPAQUE: No transparency, full depth write
- MASKED: Alpha test with discard
- TRANSLUCENT: Alpha blend, transmissive lighting
- ADDITIVE: Additive blending
- MODULATE: Multiplicative blending

Each blend mode has:
1. BlendState: GPU pipeline blend state configuration
2. BlendShaderCode: Blend-specific WGSL shader snippets

Task: T-MAT-2.3 Blend Mode Variants
Gap: S3-G3 (CRITICAL)
Dependency: T-MAT-2.1 (DONE - variant const system)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Dict, Optional

from trinity.materials.variants import BlendMode


class BlendFactor(Enum):
    """GPU blend factors for source/destination blending."""
    ZERO = "zero"
    ONE = "one"
    SRC_COLOR = "src_color"
    ONE_MINUS_SRC_COLOR = "one_minus_src_color"
    DST_COLOR = "dst_color"
    ONE_MINUS_DST_COLOR = "one_minus_dst_color"
    SRC_ALPHA = "src_alpha"
    ONE_MINUS_SRC_ALPHA = "one_minus_src_alpha"
    DST_ALPHA = "dst_alpha"
    ONE_MINUS_DST_ALPHA = "one_minus_dst_alpha"
    SRC_ALPHA_SATURATE = "src_alpha_saturate"
    CONSTANT = "constant"
    ONE_MINUS_CONSTANT = "one_minus_constant"


class BlendOperation(Enum):
    """GPU blend operations."""
    ADD = "add"
    SUBTRACT = "subtract"
    REVERSE_SUBTRACT = "reverse_subtract"
    MIN = "min"
    MAX = "max"


class ColorWriteMask(Enum):
    """Color channel write mask options."""
    NONE = "none"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    ALPHA = "alpha"
    RGB = "rgb"
    ALL = "all"


@dataclass(slots=True, frozen=True)
class BlendState:
    """GPU blend state configuration for pipeline creation.

    Configures the fixed-function blend stage of the GPU pipeline.
    Each blend mode maps to a specific BlendState configuration.

    Attributes:
        src_factor: Source blend factor for RGB channels.
        dst_factor: Destination blend factor for RGB channels.
        operation: Blend operation for RGB channels.
        alpha_src: Source blend factor for alpha channel.
        alpha_dst: Destination blend factor for alpha channel.
        alpha_op: Blend operation for alpha channel.
        write_mask: Color channel write mask.
        depth_write: Whether to write to depth buffer.
        depth_test: Whether to perform depth testing.
    """
    src_factor: BlendFactor = BlendFactor.ONE
    dst_factor: BlendFactor = BlendFactor.ZERO
    operation: BlendOperation = BlendOperation.ADD
    alpha_src: BlendFactor = BlendFactor.ONE
    alpha_dst: BlendFactor = BlendFactor.ZERO
    alpha_op: BlendOperation = BlendOperation.ADD
    write_mask: ColorWriteMask = ColorWriteMask.ALL
    depth_write: bool = True
    depth_test: bool = True

    # Pre-computed blend states for each mode (cached class variables)
    _BLEND_STATES: ClassVar[Dict[BlendMode, "BlendState"]] = {}

    @classmethod
    def for_blend_mode(cls, mode: BlendMode) -> "BlendState":
        """Get the BlendState configuration for a given blend mode.

        Args:
            mode: The BlendMode to get configuration for.

        Returns:
            BlendState configured appropriately for the blend mode.

        Examples:
            >>> state = BlendState.for_blend_mode(BlendMode.OPAQUE)
            >>> state.depth_write
            True
            >>> state = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)
            >>> state.depth_write
            False
        """
        # Return cached state if available
        if mode in cls._BLEND_STATES:
            return cls._BLEND_STATES[mode]

        # Create and cache the state
        if mode == BlendMode.OPAQUE:
            # Opaque: No blending, full depth write
            # output = source
            state = cls(
                src_factor=BlendFactor.ONE,
                dst_factor=BlendFactor.ZERO,
                operation=BlendOperation.ADD,
                alpha_src=BlendFactor.ONE,
                alpha_dst=BlendFactor.ZERO,
                alpha_op=BlendOperation.ADD,
                write_mask=ColorWriteMask.ALL,
                depth_write=True,
                depth_test=True,
            )
        elif mode == BlendMode.MASKED:
            # Masked: Same as opaque for pipeline state, alpha test happens in shader
            # output = source (after discard for alpha < threshold)
            state = cls(
                src_factor=BlendFactor.ONE,
                dst_factor=BlendFactor.ZERO,
                operation=BlendOperation.ADD,
                alpha_src=BlendFactor.ONE,
                alpha_dst=BlendFactor.ZERO,
                alpha_op=BlendOperation.ADD,
                write_mask=ColorWriteMask.ALL,
                depth_write=True,
                depth_test=True,
            )
        elif mode == BlendMode.TRANSLUCENT:
            # Translucent: Standard alpha blending, no depth write
            # output = source.rgb * source.a + dest.rgb * (1 - source.a)
            state = cls(
                src_factor=BlendFactor.SRC_ALPHA,
                dst_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
                operation=BlendOperation.ADD,
                alpha_src=BlendFactor.ONE,
                alpha_dst=BlendFactor.ONE_MINUS_SRC_ALPHA,
                alpha_op=BlendOperation.ADD,
                write_mask=ColorWriteMask.ALL,
                depth_write=False,
                depth_test=True,
            )
        elif mode == BlendMode.ADDITIVE:
            # Additive: Add source to destination, no depth write
            # output = source.rgb + dest.rgb
            state = cls(
                src_factor=BlendFactor.ONE,
                dst_factor=BlendFactor.ONE,
                operation=BlendOperation.ADD,
                alpha_src=BlendFactor.ONE,
                alpha_dst=BlendFactor.ONE,
                alpha_op=BlendOperation.ADD,
                write_mask=ColorWriteMask.ALL,
                depth_write=False,
                depth_test=True,
            )
        elif mode == BlendMode.MODULATE:
            # Modulate: Multiply source with destination, no depth write
            # output = source.rgb * dest.rgb
            state = cls(
                src_factor=BlendFactor.DST_COLOR,
                dst_factor=BlendFactor.ZERO,
                operation=BlendOperation.ADD,
                alpha_src=BlendFactor.DST_ALPHA,
                alpha_dst=BlendFactor.ZERO,
                alpha_op=BlendOperation.ADD,
                write_mask=ColorWriteMask.ALL,
                depth_write=False,
                depth_test=True,
            )
        else:
            # Fallback to opaque
            state = cls()

        cls._BLEND_STATES[mode] = state
        return state

    def to_wgpu_descriptor(self) -> Dict:
        """Convert to wgpu blend state descriptor format.

        Returns:
            Dictionary compatible with wgpu pipeline creation.
        """
        return {
            "color": {
                "srcFactor": self.src_factor.value,
                "dstFactor": self.dst_factor.value,
                "operation": self.operation.value,
            },
            "alpha": {
                "srcFactor": self.alpha_src.value,
                "dstFactor": self.alpha_dst.value,
                "operation": self.alpha_op.value,
            },
        }

    def to_depth_stencil_descriptor(self) -> Dict:
        """Convert depth/stencil state to wgpu descriptor format.

        Returns:
            Dictionary compatible with wgpu depth stencil state.
        """
        return {
            "depthWriteEnabled": self.depth_write,
            "depthCompare": "less" if self.depth_test else "always",
        }

    @property
    def requires_sorting(self) -> bool:
        """Check if this blend state requires back-to-front sorting.

        Translucent and modulate blending require proper depth sorting
        for correct visual results.

        Returns:
            True if objects using this blend state should be depth-sorted.
        """
        return not self.depth_write and self.dst_factor != BlendFactor.ZERO

    @property
    def is_opaque(self) -> bool:
        """Check if this is an opaque blend state (no blending)."""
        return (
            self.src_factor == BlendFactor.ONE
            and self.dst_factor == BlendFactor.ZERO
            and self.depth_write
        )


class BlendShaderCode:
    """Blend-mode specific WGSL shader code snippets.

    Provides shader code that implements blend-mode specific behavior
    that cannot be achieved through fixed-function blending alone
    (e.g., alpha testing via discard).
    """

    # Alpha test for masked blend mode
    MASKED_DISCARD: ClassVar[str] = """\
// Alpha test for masked blend mode
// Discards fragment if alpha is below cutoff threshold
const ALPHA_CUTOFF: f32 = 0.5;

fn apply_alpha_test(alpha: f32) {
    if (alpha < ALPHA_CUTOFF) {
        discard;
    }
}

fn apply_alpha_test_threshold(alpha: f32, threshold: f32) {
    if (alpha < threshold) {
        discard;
    }
}
"""

    # Premultiplied alpha for translucent blend mode
    TRANSLUCENT_PREMULTIPLY: ClassVar[str] = """\
// Premultiply alpha for translucent blend
// This ensures correct blending with pre-multiplied alpha textures
fn premultiply_alpha(color: vec3<f32>, alpha: f32) -> vec4<f32> {
    return vec4<f32>(color * alpha, alpha);
}

// Unpremultiply alpha (for compositing)
fn unpremultiply_alpha(premultiplied: vec4<f32>) -> vec4<f32> {
    if (premultiplied.a > 0.001) {
        return vec4<f32>(premultiplied.rgb / premultiplied.a, premultiplied.a);
    }
    return vec4<f32>(0.0);
}
"""

    # Additive emission helpers
    ADDITIVE_EMISSION: ClassVar[str] = """\
// Additive blend emission helpers
// For glow, fire, and other emissive effects
fn apply_additive_emission(base_color: vec3<f32>, emission_strength: f32) -> vec4<f32> {
    return vec4<f32>(base_color * emission_strength, 1.0);
}

fn apply_additive_emission_colored(base_color: vec3<f32>, emission_color: vec3<f32>, strength: f32) -> vec4<f32> {
    return vec4<f32>(base_color * emission_color * strength, 1.0);
}
"""

    # Modulate (multiply) helpers
    MODULATE_HELPERS: ClassVar[str] = """\
// Modulate (multiply) blend helpers
// For shadow decals, dirt overlays, etc.
fn apply_modulate_factor(factor: f32) -> vec4<f32> {
    // Factor of 1.0 = no darkening, 0.0 = full black
    return vec4<f32>(vec3<f32>(factor), 1.0);
}

fn apply_modulate_color(tint: vec3<f32>) -> vec4<f32> {
    // Tints the underlying surface by multiplying
    return vec4<f32>(tint, 1.0);
}
"""

    # Depth fade for soft particles (translucent/additive)
    DEPTH_FADE: ClassVar[str] = """\
// Soft particle depth fade
// Fades particles near geometry to avoid hard intersections
fn calculate_depth_fade(
    fragment_depth: f32,
    scene_depth: f32,
    fade_distance: f32
) -> f32 {
    let depth_diff = scene_depth - fragment_depth;
    return saturate(depth_diff / fade_distance);
}
"""

    @classmethod
    def get_for_blend(cls, mode: BlendMode) -> str:
        """Get the shader code snippet for a given blend mode.

        Args:
            mode: The BlendMode to get shader code for.

        Returns:
            WGSL shader code string for the blend mode.
            Returns empty string for modes that don't need special shader code.
        """
        if mode == BlendMode.MASKED:
            return cls.MASKED_DISCARD
        elif mode == BlendMode.TRANSLUCENT:
            return cls.TRANSLUCENT_PREMULTIPLY
        elif mode == BlendMode.ADDITIVE:
            return cls.ADDITIVE_EMISSION
        elif mode == BlendMode.MODULATE:
            return cls.MODULATE_HELPERS
        return ""

    @classmethod
    def get_depth_fade_code(cls) -> str:
        """Get the depth fade shader code for soft particles.

        Returns:
            WGSL code for depth-based fading.
        """
        return cls.DEPTH_FADE

    @classmethod
    def get_all_blend_helpers(cls) -> str:
        """Get all blend helper functions combined.

        Returns:
            All blend-related shader helper functions.
        """
        return "\n".join([
            "// =============================================================================",
            "// Blend Mode Helper Functions",
            "// =============================================================================",
            "",
            cls.MASKED_DISCARD,
            cls.TRANSLUCENT_PREMULTIPLY,
            cls.ADDITIVE_EMISSION,
            cls.MODULATE_HELPERS,
            cls.DEPTH_FADE,
        ])


def get_blend_state_for_variant(blend_mode_str: str) -> BlendState:
    """Get BlendState from a string blend mode value.

    Args:
        blend_mode_str: Blend mode as string (e.g., "opaque", "translucent").

    Returns:
        BlendState for the specified mode.

    Raises:
        ValueError: If blend_mode_str is not a valid BlendMode value.
    """
    mode_map = {mode.value: mode for mode in BlendMode}
    if blend_mode_str not in mode_map:
        raise ValueError(f"Invalid blend mode: {blend_mode_str}")
    return BlendState.for_blend_mode(mode_map[blend_mode_str])


def validate_blend_combination(
    blend_mode: BlendMode,
    depth_prepass: bool = False,
    two_sided: bool = False,
) -> tuple[bool, Optional[str]]:
    """Validate a blend mode configuration for common issues.

    Args:
        blend_mode: The blend mode being used.
        depth_prepass: Whether a depth prepass is enabled.
        two_sided: Whether two-sided rendering is enabled.

    Returns:
        Tuple of (is_valid, warning_message).
        If valid, warning_message is None.
    """
    state = BlendState.for_blend_mode(blend_mode)

    # Warn about two-sided translucent (check first, more specific issue)
    if two_sided and blend_mode == BlendMode.TRANSLUCENT:
        return True, (
            "Two-sided translucent materials may have sorting artifacts. "
            "Consider using separate front/back passes."
        )

    # Warn about sorting requirements
    if state.requires_sorting and not depth_prepass:
        return True, (
            f"Blend mode {blend_mode.value} requires back-to-front sorting "
            "for correct results. Consider enabling depth prepass."
        )

    return True, None


def compile_blend_mode_wgsl(blend_mode: BlendMode) -> str:
    """Generate WGSL code for a specific blend mode.

    This function creates a minimal WGSL shader module that demonstrates
    the blend mode behavior including:
    - Variant const declarations (BLEND_OPAQUE, BLEND_MASKED, etc.)
    - Blend-derived feature consts (ALPHA_TEST_ENABLED, ALPHA_BLEND_ENABLED)
    - The apply_blend_mode function with discard for MASKED mode
    - Blend-specific helper functions

    Args:
        blend_mode: The BlendMode to generate WGSL for.

    Returns:
        Complete WGSL shader code string.

    Example:
        >>> wgsl = compile_blend_mode_wgsl(BlendMode.MASKED)
        >>> assert 'discard' in wgsl
        >>> assert 'ALPHA_TEST_ENABLED: bool = true' in wgsl
    """
    from trinity.materials.variants import VariantConfig

    config = VariantConfig(blend=blend_mode)

    # Generate const declarations
    const_decls = config.generate_const_declarations()

    # Get blend-specific shader code (helper functions)
    blend_helpers = BlendShaderCode.get_for_blend(blend_mode)

    # Generate the apply_blend_mode function
    blend_handling = config.generate_blend_handling_code()

    # Build a minimal complete WGSL module for testing
    wgsl = f"""\
// SPDX-License-Identifier: MIT
// Generated by TRINITY compile_blend_mode_wgsl
// Blend Mode: {blend_mode.name}

{const_decls}

// ============================================================================
// Blend Mode Helper Functions
// ============================================================================
{blend_helpers}

// ============================================================================
// Blend Mode Handling
// ============================================================================
{blend_handling}

// ============================================================================
// Test Fragment Shader
// ============================================================================

struct FragmentInput {{
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}}

struct FragmentOutput {{
    @location(0) color: vec4<f32>,
}}

@fragment
fn fs_main(input: FragmentInput) -> FragmentOutput {{
    // Simple test: create a color with variable alpha
    var color = vec4<f32>(1.0, 0.5, 0.25, input.uv.x);

    // Apply blend mode handling (this is where discard happens for MASKED)
    let blended = apply_blend_mode(color, 0.5);

    return FragmentOutput(blended);
}}
"""
    return wgsl


__all__ = [
    "BlendFactor",
    "BlendOperation",
    "ColorWriteMask",
    "BlendState",
    "BlendShaderCode",
    "get_blend_state_for_variant",
    "validate_blend_combination",
    "compile_blend_mode_wgsl",
]
