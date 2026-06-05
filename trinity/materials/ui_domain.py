"""UI Material Domain - Unlit vertex-color rendering for UI elements.

This module provides specialized UI material configurations for screen-space
rendering with optimized unlit shading. UI materials bypass the PBR pipeline
entirely, using simple vertex color multiplication with optional texturing.

Features:
- Screen-space coordinate system (clip-space input positions)
- Vertex color multiplication with textures
- sRGB conversion for correct color space
- Premultiplied alpha for compositing
- Optional clip rectangle support for scissoring
- No lighting evaluation (unlit rendering)

Task: T-MAT-5.8 UI Material Domain
Gap: Variant coverage
Dependency: T-MAT-2.2 (domain variants)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class UIBlendMode(Enum):
    """UI-specific blend modes."""
    NORMAL = "normal"           # Standard alpha blending
    PREMULTIPLIED = "premultiplied"  # Premultiplied alpha
    ADDITIVE = "additive"       # Additive blending for glow effects
    MULTIPLY = "multiply"       # Multiplicative blending


@dataclass(slots=True, frozen=True)
class UIMaterialConfig:
    """Configuration for UI materials.

    UI materials are unlit, screen-space rendered materials optimized for
    2D user interface rendering. They bypass the full PBR pipeline.

    Attributes:
        premultiply_alpha: Apply premultiplied alpha for correct compositing.
            When True, RGB values are multiplied by alpha before output.
        use_srgb: Convert output to sRGB color space. When True, linear
            colors are converted to sRGB for correct display.
        vertex_color_enabled: Multiply texture color by vertex color.
            This allows per-vertex tinting of UI elements.
        texture_enabled: Sample from bound texture. When False, only
            vertex color is used.
        clip_rect_enabled: Enable clip rectangle testing. When True,
            fragments outside the clip rect are discarded.
        blend_mode: UI-specific blend mode for compositing.
        opacity: Global opacity multiplier (0.0 - 1.0).
    """
    premultiply_alpha: bool = True
    use_srgb: bool = True
    vertex_color_enabled: bool = True
    texture_enabled: bool = True
    clip_rect_enabled: bool = False
    blend_mode: UIBlendMode = UIBlendMode.PREMULTIPLIED
    opacity: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be in [0, 1], got {self.opacity}")


# WGSL template for UI materials
# Note: This is a standalone shader, not integrated with PBR pipeline
UI_DOMAIN_WGSL = '''\
// =============================================================================
// UI Material Domain - Unlit Screen-Space Rendering
// =============================================================================
// Task: T-MAT-5.8 UI Material Domain
// No PBR, no lighting evaluation, optimized for 2D UI
// =============================================================================

// UI input vertex structure
struct UIInput {
    @location(0) position: vec4<f32>,  // Clip-space position (pre-transformed)
    @location(1) uv: vec2<f32>,         // Texture coordinates
    @location(2) color: vec4<f32>,      // Vertex color (RGBA)
}

// UI output structure for interpolation
struct UIOutput {
    @builtin(position) position: vec4<f32>,  // Clip-space position
    @location(0) uv: vec2<f32>,               // Interpolated UVs
    @location(1) color: vec4<f32>,            // Interpolated vertex color
}

// UI uniforms for clip rectangle
struct UIUniforms {
    clip_rect: vec4<f32>,    // (x_min, y_min, x_max, y_max) in screen space
    screen_size: vec2<f32>,  // Viewport dimensions
    time: f32,               // Animation time
    opacity: f32,            // Global opacity multiplier
}

@group(0) @binding(0) var<uniform> ui_uniforms: UIUniforms;
@group(0) @binding(1) var ui_texture: texture_2d<f32>;
@group(0) @binding(2) var ui_sampler: sampler;

// Vertex shader for UI
// Input positions are expected to be in clip space (-1 to 1) for UI
@vertex
fn vs_ui(input: UIInput) -> UIOutput {
    var output: UIOutput;

    // Pass through clip-space position (UI is pre-transformed)
    output.position = input.position;

    // Pass through UVs and vertex color
    output.uv = input.uv;
    output.color = input.color;

    return output;
}

// Linear to sRGB conversion
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3<f32>(0.0031308);
    let low = linear * 12.92;
    let high = pow(linear, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055;
    return select(high, low, cutoff);
}

// Check if fragment is inside clip rectangle
fn is_inside_clip_rect(screen_pos: vec2<f32>, clip_rect: vec4<f32>) -> bool {
    return screen_pos.x >= clip_rect.x
        && screen_pos.y >= clip_rect.y
        && screen_pos.x <= clip_rect.z
        && screen_pos.y <= clip_rect.w;
}

// Fragment shader for UI - unlit
@fragment
fn fs_ui(input: UIOutput) -> @location(0) vec4<f32> {
    // Check clip rectangle if enabled
    if CLIP_RECT_ENABLED {
        let screen_pos = input.position.xy;
        if !is_inside_clip_rect(screen_pos, ui_uniforms.clip_rect) {
            discard;
        }
    }

    // Start with vertex color or white
    var color: vec4<f32>;
    if VERTEX_COLOR_ENABLED {
        color = input.color;
    } else {
        color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    // Sample texture if enabled
    if TEXTURE_ENABLED {
        let tex_color = textureSample(ui_texture, ui_sampler, input.uv);
        color = color * tex_color;
    }

    // Apply global opacity
    color.a = color.a * ui_uniforms.opacity;

    // Convert to sRGB if enabled
    if USE_SRGB {
        color = vec4<f32>(linear_to_srgb(color.rgb), color.a);
    }

    // Apply premultiplied alpha if enabled
    if PREMULTIPLY_ALPHA {
        color = vec4<f32>(color.rgb * color.a, color.a);
    }

    return color;
}

// =============================================================================
// Blend mode-specific fragment shaders
// =============================================================================

// Additive blend - for glow/bloom effects
@fragment
fn fs_ui_additive(input: UIOutput) -> @location(0) vec4<f32> {
    var color = input.color;

    if TEXTURE_ENABLED {
        let tex_color = textureSample(ui_texture, ui_sampler, input.uv);
        color = color * tex_color;
    }

    color.a = color.a * ui_uniforms.opacity;

    if USE_SRGB {
        color = vec4<f32>(linear_to_srgb(color.rgb), color.a);
    }

    // Additive: output RGB weighted by alpha, alpha can be 0 or preserved
    return vec4<f32>(color.rgb * color.a, 0.0);
}

// Multiply blend - for tinting/overlays
@fragment
fn fs_ui_multiply(input: UIOutput) -> @location(0) vec4<f32> {
    var color = input.color;

    if TEXTURE_ENABLED {
        let tex_color = textureSample(ui_texture, ui_sampler, input.uv);
        color = color * tex_color;
    }

    color.a = color.a * ui_uniforms.opacity;

    // Multiply blend outputs 1.0 - (1.0 - src) * alpha = src if alpha=1, 1 if alpha=0
    let inv_alpha = 1.0 - color.a;
    let blend = color.rgb * color.a + vec3<f32>(inv_alpha);

    return vec4<f32>(blend, 1.0);
}
'''

# Minimal UI shader without uniforms (for simple use cases)
UI_DOMAIN_MINIMAL_WGSL = '''\
// =============================================================================
// UI Material Domain - Minimal Unlit Shader
// =============================================================================
// Simplified UI shader without clip rect or complex features

struct UIInputSimple {
    @location(0) position: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct UIOutputSimple {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@group(0) @binding(0) var ui_texture: texture_2d<f32>;
@group(0) @binding(1) var ui_sampler: sampler;

@vertex
fn vs_ui_simple(input: UIInputSimple) -> UIOutputSimple {
    var output: UIOutputSimple;
    output.position = input.position;
    output.uv = input.uv;
    output.color = input.color;
    return output;
}

fn linear_to_srgb_simple(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3<f32>(0.0031308);
    let low = linear * 12.92;
    let high = pow(linear, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055;
    return select(high, low, cutoff);
}

@fragment
fn fs_ui_simple(input: UIOutputSimple) -> @location(0) vec4<f32> {
    let tex_color = textureSample(ui_texture, ui_sampler, input.uv);
    var color = input.color * tex_color;

    // sRGB conversion
    color = vec4<f32>(linear_to_srgb_simple(color.rgb), color.a);

    // Premultiplied alpha
    color = vec4<f32>(color.rgb * color.a, color.a);

    return color;
}
'''


def generate_ui_material_consts(config: UIMaterialConfig) -> str:
    """Generate WGSL const declarations for UI material configuration.

    Args:
        config: UIMaterialConfig with feature flags.

    Returns:
        WGSL const declarations string.
    """
    lines = [
        "// UI Material Configuration",
        f"const TEXTURE_ENABLED: bool = {str(config.texture_enabled).lower()};",
        f"const USE_SRGB: bool = {str(config.use_srgb).lower()};",
        f"const PREMULTIPLY_ALPHA: bool = {str(config.premultiply_alpha).lower()};",
        f"const VERTEX_COLOR_ENABLED: bool = {str(config.vertex_color_enabled).lower()};",
        f"const CLIP_RECT_ENABLED: bool = {str(config.clip_rect_enabled).lower()};",
        "",
    ]
    return "\n".join(lines)


def generate_ui_material(config: UIMaterialConfig, minimal: bool = False) -> str:
    """Generate complete UI material WGSL shader with configuration.

    Args:
        config: UIMaterialConfig specifying feature flags.
        minimal: If True, use minimal shader without uniforms/clip rect.

    Returns:
        Complete WGSL shader code for UI rendering.
    """
    consts = generate_ui_material_consts(config)

    if minimal:
        # Minimal shader doesn't use const guards
        return UI_DOMAIN_MINIMAL_WGSL

    return consts + UI_DOMAIN_WGSL


def get_ui_entry_point(config: UIMaterialConfig) -> str:
    """Get the appropriate fragment shader entry point for blend mode.

    Args:
        config: UIMaterialConfig with blend mode.

    Returns:
        Fragment shader entry point name.
    """
    mapping = {
        UIBlendMode.NORMAL: "fs_ui",
        UIBlendMode.PREMULTIPLIED: "fs_ui",
        UIBlendMode.ADDITIVE: "fs_ui_additive",
        UIBlendMode.MULTIPLY: "fs_ui_multiply",
    }
    return mapping.get(config.blend_mode, "fs_ui")


def validate_ui_material_wgsl(wgsl: str) -> list[str]:
    """Validate that UI material WGSL has no lighting code.

    UI materials should be completely unlit - no references to PBR
    lighting evaluation, shadow sampling, or BRDF functions.

    This function checks for specific PBR patterns, not general words.
    For example, comments mentioning "light" descriptively are fine,
    but "evaluate_direct_light" or "LIGHTING_ENABLED" are not.

    Args:
        wgsl: WGSL shader source code.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    # Forbidden patterns for UI materials - specific PBR identifiers
    # These are actual function/variable names, not general words
    forbidden_patterns = [
        # Lighting evaluation
        ("evaluate_direct_light", "evaluate_direct_light"),
        ("evaluate_ibl", "evaluate_ibl"),
        ("LIGHTING_ENABLED", "LIGHTING_ENABLED"),
        # Shadow sampling
        ("sample_shadow", "sample_shadow"),
        ("shadow_map", "shadow_map"),
        ("SHADOWS_ENABLED", "SHADOWS_ENABLED"),
        # BRDF functions
        ("BRDF", "BRDF"),
        ("evaluate_brdf", "evaluate_brdf"),
        ("brdf_diffuse", "brdf_diffuse"),
        ("brdf_specular", "brdf_specular"),
        # PBR structs
        ("PBRParams", "PBRParams"),
        ("PBRInput", "PBRInput"),
        # PBR-specific material properties (as variable names)
        ("params.metallic", "metallic property"),
        ("params.roughness", "roughness property"),
        ("params.emissive", "emissive property"),
    ]

    for pattern, description in forbidden_patterns:
        if pattern in wgsl:
            errors.append(f"UI material contains forbidden pattern: '{description}'")

    return errors


class UIMaterialBuilder:
    """Builder for constructing UI materials with fluent interface.

    Example::

        material = (
            UIMaterialBuilder()
            .with_texture()
            .with_srgb()
            .with_premultiplied_alpha()
            .with_clip_rect()
            .build()
        )
        wgsl = generate_ui_material(material)
    """

    def __init__(self) -> None:
        """Initialize builder with default config."""
        self._premultiply_alpha = True
        self._use_srgb = True
        self._vertex_color = True
        self._texture = True
        self._clip_rect = False
        self._blend_mode = UIBlendMode.PREMULTIPLIED
        self._opacity = 1.0

    def with_texture(self, enabled: bool = True) -> "UIMaterialBuilder":
        """Enable or disable texture sampling."""
        self._texture = enabled
        return self

    def with_vertex_color(self, enabled: bool = True) -> "UIMaterialBuilder":
        """Enable or disable vertex color multiplication."""
        self._vertex_color = enabled
        return self

    def with_srgb(self, enabled: bool = True) -> "UIMaterialBuilder":
        """Enable or disable sRGB conversion."""
        self._use_srgb = enabled
        return self

    def with_premultiplied_alpha(self, enabled: bool = True) -> "UIMaterialBuilder":
        """Enable or disable premultiplied alpha output."""
        self._premultiply_alpha = enabled
        return self

    def with_clip_rect(self, enabled: bool = True) -> "UIMaterialBuilder":
        """Enable or disable clip rectangle testing."""
        self._clip_rect = enabled
        return self

    def with_blend_mode(self, mode: UIBlendMode) -> "UIMaterialBuilder":
        """Set blend mode."""
        self._blend_mode = mode
        return self

    def with_opacity(self, opacity: float) -> "UIMaterialBuilder":
        """Set global opacity (0.0 - 1.0)."""
        self._opacity = opacity
        return self

    def build(self) -> UIMaterialConfig:
        """Build the UIMaterialConfig."""
        return UIMaterialConfig(
            premultiply_alpha=self._premultiply_alpha,
            use_srgb=self._use_srgb,
            vertex_color_enabled=self._vertex_color,
            texture_enabled=self._texture,
            clip_rect_enabled=self._clip_rect,
            blend_mode=self._blend_mode,
            opacity=self._opacity,
        )


# Default configurations for common UI scenarios
UI_MATERIAL_PRESETS = {
    "default": UIMaterialConfig(),
    "text": UIMaterialConfig(
        premultiply_alpha=True,
        use_srgb=True,
        vertex_color_enabled=True,
        texture_enabled=True,  # Font atlas
        clip_rect_enabled=False,
    ),
    "icon": UIMaterialConfig(
        premultiply_alpha=True,
        use_srgb=True,
        vertex_color_enabled=True,  # Tinting
        texture_enabled=True,
        clip_rect_enabled=False,
    ),
    "solid": UIMaterialConfig(
        premultiply_alpha=True,
        use_srgb=True,
        vertex_color_enabled=True,
        texture_enabled=False,  # No texture, just vertex colors
        clip_rect_enabled=False,
    ),
    "clipped": UIMaterialConfig(
        premultiply_alpha=True,
        use_srgb=True,
        vertex_color_enabled=True,
        texture_enabled=True,
        clip_rect_enabled=True,  # For scroll views
    ),
    "glow": UIMaterialConfig(
        premultiply_alpha=False,
        use_srgb=True,
        vertex_color_enabled=True,
        texture_enabled=True,
        clip_rect_enabled=False,
        blend_mode=UIBlendMode.ADDITIVE,
    ),
}


def get_ui_material_preset(name: str) -> UIMaterialConfig:
    """Get a predefined UI material configuration.

    Args:
        name: Preset name ("default", "text", "icon", "solid", "clipped", "glow").

    Returns:
        UIMaterialConfig for the preset.

    Raises:
        KeyError: If preset name is not found.
    """
    if name not in UI_MATERIAL_PRESETS:
        available = ", ".join(UI_MATERIAL_PRESETS.keys())
        raise KeyError(f"Unknown UI material preset '{name}'. Available: {available}")
    return UI_MATERIAL_PRESETS[name]


__all__ = [
    "UIBlendMode",
    "UIMaterialConfig",
    "UIMaterialBuilder",
    "UI_DOMAIN_WGSL",
    "UI_DOMAIN_MINIMAL_WGSL",
    "UI_MATERIAL_PRESETS",
    "generate_ui_material",
    "generate_ui_material_consts",
    "get_ui_entry_point",
    "get_ui_material_preset",
    "validate_ui_material_wgsl",
]
