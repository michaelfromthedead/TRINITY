"""Deferred Decal Domain - G-buffer modification via box projection.

This module provides specialized decal material configurations for deferred
rendering. Decals project textures onto existing geometry, modifying the
G-buffer without direct lighting evaluation.

Features:
- Box projection: world_pos -> decal_uv via inverse projection matrix
- G-buffer modification: albedo, normal, roughness, metallic writes
- Blend modes: Alpha, Additive, Multiply per channel
- Normal fade: attenuate at glancing angles to avoid stretching
- Angle fade: fade based on projection angle

Task: T-MAT-5.3 Decal Domain Implementation
Gap: S5-G3 (ABSENT -> PRESENT)
Dependency: T-MAT-3.4 (pipeline integration)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class DecalBlendMode(Enum):
    """Blend modes for decal channels."""
    ALPHA = "alpha"           # Standard alpha blend: src * alpha + dst * (1 - alpha)
    ADDITIVE = "additive"     # Additive: src + dst
    MULTIPLY = "multiply"     # Multiplicative: src * dst


@dataclass(slots=True, frozen=True)
class DecalChannelBlendConfig:
    """Blend configuration for individual G-buffer channels.

    Each channel can have its own blend mode, allowing for effects like:
    - Alpha blend albedo but additive normal (for detail normals)
    - Multiply roughness but alpha blend metallic

    Attributes:
        albedo: Blend mode for albedo/diffuse channel.
        normal: Blend mode for normal channel.
        roughness: Blend mode for roughness channel.
        metallic: Blend mode for metallic channel.
    """
    albedo: DecalBlendMode = DecalBlendMode.ALPHA
    normal: DecalBlendMode = DecalBlendMode.ALPHA
    roughness: DecalBlendMode = DecalBlendMode.ALPHA
    metallic: DecalBlendMode = DecalBlendMode.ALPHA

    def to_wgsl_vec4u(self) -> str:
        """Convert to WGSL vec4<u32> for shader use.

        Returns:
            WGSL vec4<u32> literal with blend mode indices.
        """
        mode_map = {
            DecalBlendMode.ALPHA: 0,
            DecalBlendMode.ADDITIVE: 1,
            DecalBlendMode.MULTIPLY: 2,
        }
        return f"vec4<u32>({mode_map[self.albedo]}u, {mode_map[self.normal]}u, {mode_map[self.roughness]}u, {mode_map[self.metallic]}u)"


@dataclass(slots=True, frozen=True)
class DecalNormalFadeConfig:
    """Configuration for normal-based fade to prevent stretching.

    When a decal projects onto a surface at a glancing angle, the texture
    can appear stretched. This config controls fade based on the angle
    between the decal projection direction and the surface normal.

    Attributes:
        start_angle_deg: Angle in degrees where fade begins (0 = head-on).
        end_angle_deg: Angle in degrees where decal is fully faded.
        enabled: Whether normal fade is active.
    """
    start_angle_deg: float = 60.0
    end_angle_deg: float = 85.0
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.start_angle_deg <= 90.0:
            raise ValueError(f"start_angle_deg must be in [0, 90], got {self.start_angle_deg}")
        if not 0.0 <= self.end_angle_deg <= 90.0:
            raise ValueError(f"end_angle_deg must be in [0, 90], got {self.end_angle_deg}")
        if self.start_angle_deg >= self.end_angle_deg:
            raise ValueError("start_angle_deg must be less than end_angle_deg")

    @property
    def start_cos(self) -> float:
        """Get cosine of start angle for shader use."""
        return math.cos(math.radians(self.start_angle_deg))

    @property
    def end_cos(self) -> float:
        """Get cosine of end angle for shader use."""
        return math.cos(math.radians(self.end_angle_deg))

    def to_wgsl_vec4f(self) -> str:
        """Convert to WGSL vec4<f32> for shader use.

        Returns:
            WGSL vec4<f32> literal with (start_cos, end_cos, 0, 0).
        """
        if not self.enabled:
            return "vec4<f32>(0.0, 0.0, 0.0, 0.0)"
        return f"vec4<f32>({self.start_cos:.6f}, {self.end_cos:.6f}, 0.0, 0.0)"


@dataclass(slots=True, frozen=True)
class DecalAngleFadeConfig:
    """Configuration for angle-based fade.

    Controls how the decal fades based on the projection angle,
    using a power function for smooth falloff.

    Attributes:
        enabled: Whether angle fade is active.
        strength: Overall fade strength (0.0 - 1.0).
        exponent: Power exponent for fade curve (higher = sharper falloff).
    """
    enabled: bool = True
    strength: float = 1.0
    exponent: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        if self.exponent <= 0.0:
            raise ValueError(f"exponent must be > 0, got {self.exponent}")

    def to_wgsl_vec4f(self) -> str:
        """Convert to WGSL vec4<f32> for shader use.

        Returns:
            WGSL vec4<f32> literal with (enabled, strength, exponent, 0).
        """
        enabled_f = 1.0 if self.enabled else 0.0
        return f"vec4<f32>({enabled_f}, {self.strength}, {self.exponent}, 0.0)"


@dataclass(slots=True)
class DecalParams:
    """Complete decal projection and blending parameters.

    This dataclass encapsulates all parameters needed for decal rendering,
    including the projection matrix, bounds, fade settings, and blend modes.

    Attributes:
        projection_matrix: 4x4 inverse projection matrix (world -> decal local).
        bounds: Half-extents of decal box (width/2, height/2, depth/2).
        opacity: Global opacity multiplier (0.0 - 1.0).
        normal_intensity: Intensity of normal contribution (0.0 - 1.0).
        blend_config: Per-channel blend mode configuration.
        normal_fade: Normal-based fade configuration.
        angle_fade: Angle-based fade configuration.
    """
    projection_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    bounds: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    opacity: float = 1.0
    normal_intensity: float = 1.0
    blend_config: DecalChannelBlendConfig = field(default_factory=DecalChannelBlendConfig)
    normal_fade: DecalNormalFadeConfig = field(default_factory=DecalNormalFadeConfig)
    angle_fade: DecalAngleFadeConfig = field(default_factory=DecalAngleFadeConfig)

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be in [0, 1], got {self.opacity}")
        if not 0.0 <= self.normal_intensity <= 1.0:
            raise ValueError(f"normal_intensity must be in [0, 1], got {self.normal_intensity}")
        if self.projection_matrix.shape != (4, 4):
            raise ValueError(f"projection_matrix must be 4x4, got {self.projection_matrix.shape}")
        if len(self.bounds) != 3:
            raise ValueError(f"bounds must have 3 components, got {len(self.bounds)}")
        for b in self.bounds:
            if b <= 0:
                raise ValueError(f"bounds must be positive, got {self.bounds}")

    @classmethod
    def from_transform(
        cls,
        position: Tuple[float, float, float],
        rotation_euler_deg: Tuple[float, float, float],
        scale: Tuple[float, float, float],
        **kwargs
    ) -> "DecalParams":
        """Create decal params from position, rotation, and scale.

        Args:
            position: World-space position (x, y, z).
            rotation_euler_deg: Euler angles in degrees (pitch, yaw, roll).
            scale: Scale factors (x, y, z).
            **kwargs: Additional DecalParams arguments.

        Returns:
            DecalParams with computed projection matrix.
        """
        # Build transformation matrix
        tx, ty, tz = position
        rx, ry, rz = [math.radians(a) for a in rotation_euler_deg]
        sx, sy, sz = scale

        # Rotation matrices
        cx, sx_r = math.cos(rx), math.sin(rx)
        cy, sy_r = math.cos(ry), math.sin(ry)
        cz, sz_r = math.cos(rz), math.sin(rz)

        # Combined rotation (ZYX order)
        rot = np.array([
            [cy * cz, cz * sx_r * sy_r - cx * sz_r, sx_r * sz_r + cx * cz * sy_r, 0],
            [cy * sz_r, cx * cz + sx_r * sy_r * sz_r, cx * sy_r * sz_r - cz * sx_r, 0],
            [-sy_r, cy * sx_r, cx * cy, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Translation
        trans = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Scale
        scale_mat = np.diag([sx, sy, sz, 1.0]).astype(np.float32)

        # Combined: T * R * S
        world_to_decal = trans @ rot @ scale_mat

        # Invert for projection
        try:
            projection_matrix = np.linalg.inv(world_to_decal)
        except np.linalg.LinAlgError:
            projection_matrix = np.eye(4, dtype=np.float32)

        return cls(
            projection_matrix=projection_matrix,
            bounds=(sx / 2, sy / 2, sz / 2),
            **kwargs
        )

    def generate_wgsl_struct(self) -> str:
        """Generate WGSL struct initialization for shader use.

        Returns:
            WGSL code initializing a DecalParams struct.
        """
        # Format projection matrix as WGSL mat4x4
        mat_rows = []
        for i in range(4):
            row = ", ".join(f"{self.projection_matrix[i, j]:.6f}" for j in range(4))
            mat_rows.append(f"        vec4<f32>({row})")

        mat_str = ",\n".join(mat_rows)

        return f"""\
DecalParams(
    mat4x4<f32>(
{mat_str}
    ),
    vec3<f32>({self.bounds[0]}, {self.bounds[1]}, {self.bounds[2]}),
    {self.normal_fade.to_wgsl_vec4f()},
    {self.angle_fade.to_wgsl_vec4f()},
    {self.blend_config.to_wgsl_vec4u()},
    {self.opacity},
    {self.normal_intensity},
    vec2<f32>(0.0, 0.0)
)"""


# WGSL template for decal domain materials
# Note: This is the domain-specific shading code, not the full shader
DECAL_DOMAIN_WGSL = '''\
// =============================================================================
// Deferred Decal Domain - G-buffer Modification
// =============================================================================
// Task: T-MAT-5.3 Decal Domain Implementation
// Box projection, per-channel blending, normal fade
// =============================================================================

// Decal projection result
struct DecalProjectionResult {
    uv: vec2<f32>,
    depth: f32,
    valid: bool,
    fade: f32,
}

// Project world position to decal UV space
fn decal_project(world_pos: vec3<f32>, inv_proj: mat4x4<f32>, bounds: vec3<f32>) -> DecalProjectionResult {
    var result: DecalProjectionResult;

    let local = inv_proj * vec4<f32>(world_pos, 1.0);
    let local_pos = local.xyz / local.w;

    result.valid = abs(local_pos.x) <= bounds.x
                && abs(local_pos.y) <= bounds.y
                && abs(local_pos.z) <= bounds.z;

    result.uv = (local_pos.xy / bounds.xy) * 0.5 + 0.5;
    result.depth = local_pos.z / bounds.z;
    result.fade = 1.0;

    return result;
}

// Compute normal-based fade
fn decal_normal_fade(surface_n: vec3<f32>, decal_n: vec3<f32>, start_cos: f32, end_cos: f32) -> f32 {
    let cos_angle = dot(surface_n, decal_n);
    if (cos_angle >= start_cos) { return 1.0; }
    if (cos_angle <= end_cos) { return 0.0; }
    let t = (cos_angle - end_cos) / (start_cos - end_cos);
    return t * t * (3.0 - 2.0 * t);
}

// Blend modes
fn decal_blend_alpha(dst: vec3<f32>, src: vec3<f32>, alpha: f32) -> vec3<f32> {
    return mix(dst, src, alpha);
}

fn decal_blend_additive(dst: vec3<f32>, src: vec3<f32>, alpha: f32) -> vec3<f32> {
    return dst + src * alpha;
}

fn decal_blend_multiply(dst: vec3<f32>, src: vec3<f32>, alpha: f32) -> vec3<f32> {
    return mix(dst, dst * src, alpha);
}

// Reoriented normal mapping blend
fn decal_blend_normal(base: vec3<f32>, detail: vec3<f32>, alpha: f32) -> vec3<f32> {
    let t = base + vec3<f32>(0.0, 0.0, 1.0);
    let u = detail * vec3<f32>(-1.0, -1.0, 1.0);
    let blended = t * dot(t, u) - u * t.z;
    return normalize(mix(base, blended, alpha));
}
'''


def generate_decal_domain_consts(params: DecalParams) -> str:
    """Generate WGSL const declarations for decal domain.

    Args:
        params: DecalParams with configuration.

    Returns:
        WGSL const declarations string.
    """
    lines = [
        "// Decal Domain Configuration",
        f"const DECAL_OPACITY: f32 = {params.opacity};",
        f"const DECAL_NORMAL_INTENSITY: f32 = {params.normal_intensity};",
        f"const DECAL_NORMAL_FADE_ENABLED: bool = {str(params.normal_fade.enabled).lower()};",
        f"const DECAL_NORMAL_FADE_START_COS: f32 = {params.normal_fade.start_cos:.6f};",
        f"const DECAL_NORMAL_FADE_END_COS: f32 = {params.normal_fade.end_cos:.6f};",
        f"const DECAL_ANGLE_FADE_ENABLED: bool = {str(params.angle_fade.enabled).lower()};",
        f"const DECAL_ANGLE_FADE_STRENGTH: f32 = {params.angle_fade.strength};",
        f"const DECAL_ANGLE_FADE_EXPONENT: f32 = {params.angle_fade.exponent};",
        "",
    ]
    return "\n".join(lines)


def generate_decal_material(params: DecalParams) -> str:
    """Generate complete decal domain WGSL shader with configuration.

    Args:
        params: DecalParams specifying projection and blending.

    Returns:
        Complete WGSL shader code for decal rendering.
    """
    consts = generate_decal_domain_consts(params)
    return consts + DECAL_DOMAIN_WGSL


class DecalMaterialBuilder:
    """Builder for constructing decal materials with fluent interface.

    Example::

        params = (
            DecalMaterialBuilder()
            .with_position(0, 1, 0)
            .with_rotation(0, 45, 0)
            .with_scale(2, 2, 0.5)
            .with_opacity(0.8)
            .with_blend(DecalBlendMode.ALPHA)
            .with_normal_fade(60, 85)
            .build()
        )
        wgsl = generate_decal_material(params)
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._position = (0.0, 0.0, 0.0)
        self._rotation = (0.0, 0.0, 0.0)
        self._scale = (1.0, 1.0, 1.0)
        self._opacity = 1.0
        self._normal_intensity = 1.0
        self._albedo_blend = DecalBlendMode.ALPHA
        self._normal_blend = DecalBlendMode.ALPHA
        self._roughness_blend = DecalBlendMode.ALPHA
        self._metallic_blend = DecalBlendMode.ALPHA
        self._normal_fade_start = 60.0
        self._normal_fade_end = 85.0
        self._normal_fade_enabled = True
        self._angle_fade_enabled = True
        self._angle_fade_strength = 1.0
        self._angle_fade_exponent = 1.0

    def with_position(self, x: float, y: float, z: float) -> "DecalMaterialBuilder":
        """Set decal world position."""
        self._position = (x, y, z)
        return self

    def with_rotation(self, pitch: float, yaw: float, roll: float) -> "DecalMaterialBuilder":
        """Set decal rotation in degrees (pitch, yaw, roll)."""
        self._rotation = (pitch, yaw, roll)
        return self

    def with_scale(self, x: float, y: float, z: float) -> "DecalMaterialBuilder":
        """Set decal scale (width, height, depth)."""
        self._scale = (x, y, z)
        return self

    def with_opacity(self, opacity: float) -> "DecalMaterialBuilder":
        """Set global opacity (0.0 - 1.0)."""
        self._opacity = opacity
        return self

    def with_normal_intensity(self, intensity: float) -> "DecalMaterialBuilder":
        """Set normal contribution intensity (0.0 - 1.0)."""
        self._normal_intensity = intensity
        return self

    def with_blend(self, mode: DecalBlendMode) -> "DecalMaterialBuilder":
        """Set blend mode for all channels."""
        self._albedo_blend = mode
        self._normal_blend = mode
        self._roughness_blend = mode
        self._metallic_blend = mode
        return self

    def with_albedo_blend(self, mode: DecalBlendMode) -> "DecalMaterialBuilder":
        """Set blend mode for albedo channel."""
        self._albedo_blend = mode
        return self

    def with_normal_blend(self, mode: DecalBlendMode) -> "DecalMaterialBuilder":
        """Set blend mode for normal channel."""
        self._normal_blend = mode
        return self

    def with_roughness_blend(self, mode: DecalBlendMode) -> "DecalMaterialBuilder":
        """Set blend mode for roughness channel."""
        self._roughness_blend = mode
        return self

    def with_metallic_blend(self, mode: DecalBlendMode) -> "DecalMaterialBuilder":
        """Set blend mode for metallic channel."""
        self._metallic_blend = mode
        return self

    def with_normal_fade(
        self,
        start_deg: float = 60.0,
        end_deg: float = 85.0,
        enabled: bool = True
    ) -> "DecalMaterialBuilder":
        """Set normal-based fade parameters."""
        self._normal_fade_start = start_deg
        self._normal_fade_end = end_deg
        self._normal_fade_enabled = enabled
        return self

    def with_angle_fade(
        self,
        strength: float = 1.0,
        exponent: float = 1.0,
        enabled: bool = True
    ) -> "DecalMaterialBuilder":
        """Set angle-based fade parameters."""
        self._angle_fade_strength = strength
        self._angle_fade_exponent = exponent
        self._angle_fade_enabled = enabled
        return self

    def build(self) -> DecalParams:
        """Build the DecalParams."""
        blend_config = DecalChannelBlendConfig(
            albedo=self._albedo_blend,
            normal=self._normal_blend,
            roughness=self._roughness_blend,
            metallic=self._metallic_blend,
        )

        normal_fade = DecalNormalFadeConfig(
            start_angle_deg=self._normal_fade_start,
            end_angle_deg=self._normal_fade_end,
            enabled=self._normal_fade_enabled,
        )

        angle_fade = DecalAngleFadeConfig(
            enabled=self._angle_fade_enabled,
            strength=self._angle_fade_strength,
            exponent=self._angle_fade_exponent,
        )

        return DecalParams.from_transform(
            position=self._position,
            rotation_euler_deg=self._rotation,
            scale=self._scale,
            opacity=self._opacity,
            normal_intensity=self._normal_intensity,
            blend_config=blend_config,
            normal_fade=normal_fade,
            angle_fade=angle_fade,
        )


def validate_decal_projection(
    world_pos: Tuple[float, float, float],
    params: DecalParams
) -> Tuple[bool, Tuple[float, float], float]:
    """Validate decal projection for a world position.

    Useful for testing and debugging decal placement.

    Args:
        world_pos: World-space position (x, y, z).
        params: DecalParams with projection matrix and bounds.

    Returns:
        Tuple of (is_valid, (u, v), depth).
    """
    pos = np.array([*world_pos, 1.0], dtype=np.float32)
    local = params.projection_matrix @ pos
    local_pos = local[:3] / local[3]

    bx, by, bz = params.bounds
    is_valid = (
        abs(local_pos[0]) <= bx
        and abs(local_pos[1]) <= by
        and abs(local_pos[2]) <= bz
    )

    u = (local_pos[0] / bx) * 0.5 + 0.5
    v = (local_pos[1] / by) * 0.5 + 0.5
    depth = local_pos[2] / bz

    return is_valid, (u, v), depth


# Preset decal configurations for common use cases
DECAL_PRESETS: Dict[str, DecalParams] = {}


def _init_presets() -> None:
    """Initialize preset configurations."""
    global DECAL_PRESETS

    # Standard decal - balanced settings
    DECAL_PRESETS["standard"] = DecalParams(
        opacity=1.0,
        normal_intensity=1.0,
        blend_config=DecalChannelBlendConfig(),
        normal_fade=DecalNormalFadeConfig(),
        angle_fade=DecalAngleFadeConfig(),
    )

    # Blood splatter - additive albedo, no normal contribution
    DECAL_PRESETS["blood"] = DecalParams(
        opacity=0.9,
        normal_intensity=0.0,
        blend_config=DecalChannelBlendConfig(
            albedo=DecalBlendMode.MULTIPLY,
            normal=DecalBlendMode.ALPHA,
            roughness=DecalBlendMode.ALPHA,
            metallic=DecalBlendMode.ALPHA,
        ),
        normal_fade=DecalNormalFadeConfig(start_angle_deg=45.0, end_angle_deg=80.0),
        angle_fade=DecalAngleFadeConfig(strength=0.8, exponent=1.5),
    )

    # Bullet hole - strong normal, small size
    DECAL_PRESETS["bullet_hole"] = DecalParams(
        opacity=1.0,
        normal_intensity=1.0,
        blend_config=DecalChannelBlendConfig(
            albedo=DecalBlendMode.ALPHA,
            normal=DecalBlendMode.ALPHA,
            roughness=DecalBlendMode.ALPHA,
            metallic=DecalBlendMode.ALPHA,
        ),
        normal_fade=DecalNormalFadeConfig(start_angle_deg=70.0, end_angle_deg=88.0),
        angle_fade=DecalAngleFadeConfig(strength=1.0, exponent=2.0),
    )

    # Graffiti - strong colors, weak normal
    DECAL_PRESETS["graffiti"] = DecalParams(
        opacity=0.95,
        normal_intensity=0.2,
        blend_config=DecalChannelBlendConfig(
            albedo=DecalBlendMode.ALPHA,
            normal=DecalBlendMode.ALPHA,
            roughness=DecalBlendMode.MULTIPLY,
            metallic=DecalBlendMode.ALPHA,
        ),
        normal_fade=DecalNormalFadeConfig(start_angle_deg=50.0, end_angle_deg=85.0),
        angle_fade=DecalAngleFadeConfig(strength=0.9, exponent=1.0),
    )

    # Glow effect - additive blending
    DECAL_PRESETS["glow"] = DecalParams(
        opacity=0.7,
        normal_intensity=0.0,
        blend_config=DecalChannelBlendConfig(
            albedo=DecalBlendMode.ADDITIVE,
            normal=DecalBlendMode.ALPHA,
            roughness=DecalBlendMode.ALPHA,
            metallic=DecalBlendMode.ALPHA,
        ),
        normal_fade=DecalNormalFadeConfig(enabled=False),
        angle_fade=DecalAngleFadeConfig(enabled=False),
    )


_init_presets()


def get_decal_preset(name: str) -> DecalParams:
    """Get a predefined decal configuration.

    Args:
        name: Preset name ("standard", "blood", "bullet_hole", "graffiti", "glow").

    Returns:
        DecalParams for the preset.

    Raises:
        KeyError: If preset name is not found.
    """
    if name not in DECAL_PRESETS:
        available = ", ".join(DECAL_PRESETS.keys())
        raise KeyError(f"Unknown decal preset '{name}'. Available: {available}")
    return DECAL_PRESETS[name]


def load_decal_wgsl() -> str:
    """Load the decal WGSL shader from file.

    Returns:
        Contents of decal.wgsl file.
    """
    wgsl_path = Path(__file__).parent / "wgsl" / "decal.wgsl"
    return wgsl_path.read_text()


__all__ = [
    "DecalBlendMode",
    "DecalChannelBlendConfig",
    "DecalNormalFadeConfig",
    "DecalAngleFadeConfig",
    "DecalParams",
    "DecalMaterialBuilder",
    "DECAL_DOMAIN_WGSL",
    "DECAL_PRESETS",
    "generate_decal_material",
    "generate_decal_domain_consts",
    "get_decal_preset",
    "validate_decal_projection",
    "load_decal_wgsl",
]
