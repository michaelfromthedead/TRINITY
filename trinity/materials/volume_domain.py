"""Volume Material Domain - Ray marching for fog, clouds, and smoke.

This module provides specialized volume material configurations for participating
media rendering with single-scattering integration. Volume materials use ray
marching through bounded regions with physically-based light transport.

Features:
- Ray-AABB intersection for volume bounds
- Adaptive step size ray marching (smaller steps in dense regions)
- Single-scattering from directional and point lights
- Henyey-Greenstein phase function for anisotropic scattering
- Beer's law transmittance (absorption + scattering)
- Procedural density functions (exponential fog, distance fog)
- 3D texture sampling for volumetric clouds/smoke

Task: T-MAT-5.4 Volume Domain Implementation
Gap: S5-G4
Dependency: T-MAT-3.4 (pipeline integration)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

# WGSL shader path
VOLUME_WGSL_PATH = Path(__file__).parent / "wgsl" / "volume.wgsl"


class VolumeDensityMode(Enum):
    """Volume density source mode."""
    HOMOGENEOUS = "homogeneous"         # Constant density throughout
    EXPONENTIAL_HEIGHT = "exponential"  # Exponential falloff with height
    DISTANCE = "distance"               # Distance-based fog
    TEXTURE_3D = "texture_3d"           # 3D texture sampling
    PROCEDURAL = "procedural"           # Custom procedural function


class VolumePhaseFunction(Enum):
    """Phase function for scattering direction distribution."""
    ISOTROPIC = "isotropic"             # Uniform in all directions
    HENYEY_GREENSTEIN = "henyey_greenstein"  # Single-lobe HG
    TWO_LOBE_HG = "two_lobe_hg"         # Forward + backward lobes
    RAYLEIGH = "rayleigh"               # Atmospheric scattering


@dataclass(slots=True, frozen=True)
class VolumeParams:
    """Physical parameters for volume rendering.

    Describes the optical properties of a participating medium using
    absorption and scattering coefficients.

    Attributes:
        density_scale: Overall density multiplier (1.0 = base density).
        absorption: RGB absorption coefficients (light lost per unit distance).
            Higher values = darker, more opaque medium.
        scattering: RGB scattering coefficients (light redirected per unit distance).
            Higher values = brighter medium from in-scattered light.
        phase_g: Henyey-Greenstein asymmetry parameter [-1, 1].
            0 = isotropic, >0 = forward scattering (fog/clouds), <0 = back scattering.
        emission: RGB self-emission (for glowing volumes like fire).
        max_march_distance: Maximum ray march distance in world units.
        max_march_steps: Maximum number of ray marching steps.
    """
    density_scale: float = 1.0
    absorption: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    scattering: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    phase_g: float = 0.0
    emission: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_march_distance: float = 100.0
    max_march_steps: int = 128

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if self.density_scale < 0:
            raise ValueError(f"density_scale must be >= 0, got {self.density_scale}")
        if not -1.0 <= self.phase_g <= 1.0:
            raise ValueError(f"phase_g must be in [-1, 1], got {self.phase_g}")
        if self.max_march_distance <= 0:
            raise ValueError(f"max_march_distance must be > 0, got {self.max_march_distance}")
        if self.max_march_steps < 1:
            raise ValueError(f"max_march_steps must be >= 1, got {self.max_march_steps}")

    @property
    def extinction(self) -> Tuple[float, float, float]:
        """Compute extinction coefficient (absorption + scattering)."""
        return (
            self.absorption[0] + self.scattering[0],
            self.absorption[1] + self.scattering[1],
            self.absorption[2] + self.scattering[2],
        )

    @property
    def single_scattering_albedo(self) -> Tuple[float, float, float]:
        """Compute single-scattering albedo (scattering / extinction).

        Returns the fraction of extinction that is due to scattering vs absorption.
        Values close to 1.0 indicate highly scattering media (clouds, milk).
        Values close to 0.0 indicate highly absorbing media (smoke, soot).
        """
        ext = self.extinction
        if sum(ext) < 1e-6:
            return (1.0, 1.0, 1.0)
        return (
            self.scattering[0] / max(ext[0], 1e-6),
            self.scattering[1] / max(ext[1], 1e-6),
            self.scattering[2] / max(ext[2], 1e-6),
        )


@dataclass(slots=True, frozen=True)
class VolumeMaterialConfig:
    """Configuration for volume materials.

    Volume materials render participating media like fog, clouds, and smoke
    using ray marching with physically-based light transport.

    Attributes:
        params: Physical volume parameters (absorption, scattering, etc.).
        density_mode: How density is sampled (homogeneous, texture, procedural).
        phase_function: Phase function type for scattering.
        enable_shadows: Whether volume receives shadows from scene geometry.
        enable_self_shadowing: Whether volume casts shadows on itself.
        adaptive_stepping: Use smaller steps in dense regions.
        early_termination: Stop marching when transmittance is negligible.
        transmittance_threshold: Threshold for early termination (default 0.001).
    """
    params: VolumeParams = VolumeParams()
    density_mode: VolumeDensityMode = VolumeDensityMode.HOMOGENEOUS
    phase_function: VolumePhaseFunction = VolumePhaseFunction.HENYEY_GREENSTEIN
    enable_shadows: bool = False
    enable_self_shadowing: bool = False
    adaptive_stepping: bool = True
    early_termination: bool = True
    transmittance_threshold: float = 0.001

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 < self.transmittance_threshold < 1.0:
            raise ValueError(
                f"transmittance_threshold must be in (0, 1), got {self.transmittance_threshold}"
            )


# WGSL template for volume domain integration
# This extends the base volume.wgsl with domain-specific binding declarations
VOLUME_DOMAIN_WGSL = '''\
// =============================================================================
// Volume Material Domain - Integration Layer
// =============================================================================
// Task: T-MAT-5.4 Volume Domain Implementation
// Provides binding declarations and integration with material system
// =============================================================================

// Volume-specific uniform buffer
struct VolumeUniforms {
    /// AABB bounds for the volume (min xyz in first 3, max xyz in second 3)
    aabb_min: vec3<f32>,
    _padding1: f32,
    aabb_max: vec3<f32>,
    _padding2: f32,
    /// Volume parameters
    density_scale: f32,
    phase_g: f32,
    max_distance: f32,
    transmittance_threshold: f32,
    /// Absorption coefficient RGB + unused
    absorption: vec4<f32>,
    /// Scattering coefficient RGB + unused
    scattering: vec4<f32>,
    /// Emission RGB + base density
    emission: vec4<f32>,
}

// Bind group 0: Volume data
@group(0) @binding(0) var<uniform> volume_uniforms: VolumeUniforms;
@group(0) @binding(1) var density_texture: texture_3d<f32>;
@group(0) @binding(2) var density_sampler: sampler;

// Bind group 1: Camera and scene
@group(1) @binding(0) var<uniform> camera: CameraUniforms;

// Bind group 2: Lighting (reuses existing light structures)
@group(2) @binding(0) var<storage, read> lights: array<Light>;
@group(2) @binding(1) var<uniform> light_count: u32;

// Camera uniform structure (expected by material system)
struct CameraUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position: vec3<f32>,
    near: f32,
    far: f32,
    _padding: vec3<f32>,
}

// Light structure (matches existing lighting module)
struct Light {
    position: vec4<f32>,   // xyz = position, w = type (0=dir, 1=point, 2=spot)
    direction: vec4<f32>,  // xyz = direction, w = range
    color: vec4<f32>,      // rgb = color, a = intensity
    params: vec4<f32>,     // x = inner cone, y = outer cone, zw = unused
}

// Vertex input for volume proxy geometry (box or fullscreen quad)
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

// Vertex output / fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) view_ray: vec3<f32>,
}

// Vertex shader for volume proxy geometry
@vertex
fn vs_volume(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Transform vertex to world space (assuming volume is in world space)
    let world_pos = vec4<f32>(input.position, 1.0);
    output.world_position = world_pos.xyz;

    // Transform to clip space
    output.clip_position = camera.view_proj * world_pos;

    // View ray from camera to world position
    output.view_ray = normalize(world_pos.xyz - camera.position);

    return output;
}

// Include core volume functions
// (In actual shader compilation, volume.wgsl is prepended)

// Build VolumeParams from uniforms
fn build_volume_params() -> VolumeParams {
    var params: VolumeParams;
    params.density_scale = volume_uniforms.density_scale;
    params.absorption = volume_uniforms.absorption.rgb;
    params.scattering = volume_uniforms.scattering.rgb;
    params.phase_g = volume_uniforms.phase_g;
    params.emission = volume_uniforms.emission.rgb;
    params.max_distance = volume_uniforms.max_distance;
    return params;
}

// Get primary light direction and color
fn get_primary_light() -> vec2<vec3<f32>> {
    if light_count > 0u {
        let light = lights[0];
        if light.position.w < 0.5 {
            // Directional light
            return vec2<vec3<f32>>(-light.direction.xyz, light.color.rgb * light.color.a);
        } else {
            // Point light - use direction toward volume center
            let center = (volume_uniforms.aabb_min + volume_uniforms.aabb_max) * 0.5;
            return vec2<vec3<f32>>(normalize(light.position.xyz - center), light.color.rgb * light.color.a);
        }
    }
    // Default: downward white light
    return vec2<vec3<f32>>(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0));
}

// Fragment shader for volume rendering
@fragment
fn fs_volume(input: VertexOutput) -> @location(0) vec4<f32> {
    // Build parameters from uniforms
    let params = build_volume_params();

    // Get primary light
    let light_data = get_primary_light();
    let light_dir = light_data[0];
    let light_color = light_data[1];

    // Base density from emission.w (repurposed field)
    let base_density = volume_uniforms.emission.w;

    // Evaluate volume
    let result = evaluate_volume(
        camera.position,
        normalize(input.view_ray),
        volume_uniforms.aabb_min,
        volume_uniforms.aabb_max,
        params,
        light_dir,
        light_color,
        base_density
    );

    return result;
}

// Fragment shader for fullscreen fog (post-process style)
@fragment
fn fs_fullscreen_fog(input: VertexOutput) -> @location(0) vec4<f32> {
    let params = build_volume_params();

    let light_data = get_primary_light();
    let light_dir = light_data[0];
    let light_color = light_data[1];

    let base_density = volume_uniforms.emission.w;

    // Use homogeneous fog for fullscreen pass
    return evaluate_homogeneous_fog(
        camera.position,
        normalize(input.view_ray),
        params.max_distance,
        params,
        light_dir,
        light_color,
        base_density
    );
}
'''


def generate_volume_material_consts(config: VolumeMaterialConfig) -> str:
    """Generate WGSL const declarations for volume material configuration.

    Args:
        config: VolumeMaterialConfig with feature flags.

    Returns:
        WGSL const declarations string.
    """
    lines = [
        "// Volume Material Configuration",
        f"const DENSITY_MODE_HOMOGENEOUS: bool = {str(config.density_mode == VolumeDensityMode.HOMOGENEOUS).lower()};",
        f"const DENSITY_MODE_EXPONENTIAL: bool = {str(config.density_mode == VolumeDensityMode.EXPONENTIAL_HEIGHT).lower()};",
        f"const DENSITY_MODE_DISTANCE: bool = {str(config.density_mode == VolumeDensityMode.DISTANCE).lower()};",
        f"const DENSITY_MODE_TEXTURE: bool = {str(config.density_mode == VolumeDensityMode.TEXTURE_3D).lower()};",
        f"const DENSITY_MODE_PROCEDURAL: bool = {str(config.density_mode == VolumeDensityMode.PROCEDURAL).lower()};",
        "",
        f"const PHASE_ISOTROPIC: bool = {str(config.phase_function == VolumePhaseFunction.ISOTROPIC).lower()};",
        f"const PHASE_HG: bool = {str(config.phase_function == VolumePhaseFunction.HENYEY_GREENSTEIN).lower()};",
        f"const PHASE_TWO_LOBE_HG: bool = {str(config.phase_function == VolumePhaseFunction.TWO_LOBE_HG).lower()};",
        f"const PHASE_RAYLEIGH: bool = {str(config.phase_function == VolumePhaseFunction.RAYLEIGH).lower()};",
        "",
        f"const ENABLE_SHADOWS: bool = {str(config.enable_shadows).lower()};",
        f"const ENABLE_SELF_SHADOWING: bool = {str(config.enable_self_shadowing).lower()};",
        f"const ADAPTIVE_STEPPING: bool = {str(config.adaptive_stepping).lower()};",
        f"const EARLY_TERMINATION: bool = {str(config.early_termination).lower()};",
        f"const TRANSMITTANCE_THRESHOLD: f32 = {config.transmittance_threshold};",
        "",
        f"const MAX_MARCH_STEPS: u32 = {config.params.max_march_steps}u;",
        "",
    ]
    return "\n".join(lines)


def generate_volume_material(
    config: VolumeMaterialConfig,
    include_core: bool = True
) -> str:
    """Generate complete volume material WGSL shader with configuration.

    Args:
        config: VolumeMaterialConfig specifying volume behavior.
        include_core: If True, prepend core volume.wgsl functions.

    Returns:
        Complete WGSL shader code for volume rendering.
    """
    consts = generate_volume_material_consts(config)

    parts = [consts]

    if include_core:
        # Read and include core volume functions
        if VOLUME_WGSL_PATH.exists():
            core_wgsl = VOLUME_WGSL_PATH.read_text()
            parts.append(core_wgsl)

    parts.append(VOLUME_DOMAIN_WGSL)

    return "\n".join(parts)


def get_volume_entry_point(config: VolumeMaterialConfig) -> str:
    """Get the appropriate fragment shader entry point for volume config.

    Args:
        config: VolumeMaterialConfig to determine entry point.

    Returns:
        Fragment shader entry point name.
    """
    if config.density_mode == VolumeDensityMode.HOMOGENEOUS:
        return "fs_fullscreen_fog"
    return "fs_volume"


def validate_volume_material_wgsl(wgsl: str) -> list[str]:
    """Validate that volume material WGSL has correct structure.

    Volume materials should have:
    - VolumeParams or VolumeUniforms struct
    - Ray marching functions (ray_aabb_intersect, march_volume)
    - Phase function (henyey_greenstein or similar)
    - Transmittance calculation

    Args:
        wgsl: WGSL shader source code.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    # Required patterns for volume rendering
    required_patterns = [
        ("VolumeParams", "Volume parameters struct"),
        ("ray_aabb_intersect", "Ray-AABB intersection function"),
        ("henyey_greenstein", "Phase function"),
        ("transmittance", "Transmittance calculation (Beer's law)"),
    ]

    for pattern, description in required_patterns:
        if pattern.lower() not in wgsl.lower():
            errors.append(f"Volume material missing required: {description} ({pattern})")

    # Forbidden patterns (these indicate wrong domain)
    forbidden_patterns = [
        ("evaluate_surface_domain", "Surface domain evaluation"),
        ("evaluate_ui_domain", "UI domain evaluation"),
        ("BRDF_Specular", "Specular BRDF (not for volumes)"),
    ]

    for pattern, description in forbidden_patterns:
        if pattern in wgsl:
            errors.append(f"Volume material contains forbidden pattern: {description}")

    return errors


class VolumeMaterialBuilder:
    """Builder for constructing volume materials with fluent interface.

    Example::

        material = (
            VolumeMaterialBuilder()
            .with_scattering(0.2, 0.2, 0.2)
            .with_absorption(0.01, 0.01, 0.01)
            .with_phase_g(0.7)
            .with_density_mode(VolumeDensityMode.EXPONENTIAL_HEIGHT)
            .build()
        )
        wgsl = generate_volume_material(material)
    """

    def __init__(self) -> None:
        """Initialize builder with default config."""
        self._density_scale = 1.0
        self._absorption = (0.01, 0.01, 0.01)
        self._scattering = (0.1, 0.1, 0.1)
        self._phase_g = 0.0
        self._emission = (0.0, 0.0, 0.0)
        self._max_march_distance = 100.0
        self._max_march_steps = 128
        self._density_mode = VolumeDensityMode.HOMOGENEOUS
        self._phase_function = VolumePhaseFunction.HENYEY_GREENSTEIN
        self._enable_shadows = False
        self._enable_self_shadowing = False
        self._adaptive_stepping = True
        self._early_termination = True
        self._transmittance_threshold = 0.001

    def with_density_scale(self, scale: float) -> "VolumeMaterialBuilder":
        """Set overall density multiplier."""
        self._density_scale = scale
        return self

    def with_absorption(self, r: float, g: float, b: float) -> "VolumeMaterialBuilder":
        """Set absorption coefficient RGB."""
        self._absorption = (r, g, b)
        return self

    def with_scattering(self, r: float, g: float, b: float) -> "VolumeMaterialBuilder":
        """Set scattering coefficient RGB."""
        self._scattering = (r, g, b)
        return self

    def with_phase_g(self, g: float) -> "VolumeMaterialBuilder":
        """Set Henyey-Greenstein asymmetry parameter."""
        self._phase_g = g
        return self

    def with_emission(self, r: float, g: float, b: float) -> "VolumeMaterialBuilder":
        """Set emissive color RGB."""
        self._emission = (r, g, b)
        return self

    def with_max_distance(self, distance: float) -> "VolumeMaterialBuilder":
        """Set maximum ray march distance."""
        self._max_march_distance = distance
        return self

    def with_max_steps(self, steps: int) -> "VolumeMaterialBuilder":
        """Set maximum ray marching steps."""
        self._max_march_steps = steps
        return self

    def with_density_mode(self, mode: VolumeDensityMode) -> "VolumeMaterialBuilder":
        """Set density sampling mode."""
        self._density_mode = mode
        return self

    def with_phase_function(self, func: VolumePhaseFunction) -> "VolumeMaterialBuilder":
        """Set phase function type."""
        self._phase_function = func
        return self

    def with_shadows(self, enabled: bool = True) -> "VolumeMaterialBuilder":
        """Enable or disable shadow reception."""
        self._enable_shadows = enabled
        return self

    def with_self_shadowing(self, enabled: bool = True) -> "VolumeMaterialBuilder":
        """Enable or disable self-shadowing."""
        self._enable_self_shadowing = enabled
        return self

    def with_adaptive_stepping(self, enabled: bool = True) -> "VolumeMaterialBuilder":
        """Enable or disable adaptive step size."""
        self._adaptive_stepping = enabled
        return self

    def with_early_termination(
        self, enabled: bool = True, threshold: float = 0.001
    ) -> "VolumeMaterialBuilder":
        """Enable or disable early termination."""
        self._early_termination = enabled
        self._transmittance_threshold = threshold
        return self

    def build(self) -> VolumeMaterialConfig:
        """Build the VolumeMaterialConfig."""
        params = VolumeParams(
            density_scale=self._density_scale,
            absorption=self._absorption,
            scattering=self._scattering,
            phase_g=self._phase_g,
            emission=self._emission,
            max_march_distance=self._max_march_distance,
            max_march_steps=self._max_march_steps,
        )
        return VolumeMaterialConfig(
            params=params,
            density_mode=self._density_mode,
            phase_function=self._phase_function,
            enable_shadows=self._enable_shadows,
            enable_self_shadowing=self._enable_self_shadowing,
            adaptive_stepping=self._adaptive_stepping,
            early_termination=self._early_termination,
            transmittance_threshold=self._transmittance_threshold,
        )


# Default configurations for common volume scenarios
VOLUME_MATERIAL_PRESETS = {
    "fog_light": VolumeMaterialConfig(
        params=VolumeParams(
            density_scale=0.5,
            absorption=(0.005, 0.005, 0.005),
            scattering=(0.05, 0.05, 0.05),
            phase_g=0.3,
        ),
        density_mode=VolumeDensityMode.HOMOGENEOUS,
    ),
    "fog_dense": VolumeMaterialConfig(
        params=VolumeParams(
            density_scale=2.0,
            absorption=(0.02, 0.02, 0.02),
            scattering=(0.2, 0.2, 0.2),
            phase_g=0.5,
        ),
        density_mode=VolumeDensityMode.HOMOGENEOUS,
    ),
    "fog_height": VolumeMaterialConfig(
        params=VolumeParams(
            density_scale=1.0,
            absorption=(0.01, 0.01, 0.01),
            scattering=(0.1, 0.1, 0.1),
            phase_g=0.4,
        ),
        density_mode=VolumeDensityMode.EXPONENTIAL_HEIGHT,
    ),
    "cloud": VolumeMaterialConfig(
        params=VolumeParams(
            density_scale=1.0,
            absorption=(0.001, 0.001, 0.001),
            scattering=(0.3, 0.3, 0.3),
            phase_g=0.8,  # Strong forward scattering
        ),
        density_mode=VolumeDensityMode.TEXTURE_3D,
        phase_function=VolumePhaseFunction.TWO_LOBE_HG,
    ),
    "smoke": VolumeMaterialConfig(
        params=VolumeParams(
            density_scale=1.5,
            absorption=(0.1, 0.08, 0.05),  # Warm absorption
            scattering=(0.15, 0.12, 0.1),
            phase_g=-0.2,  # Slight back scattering
        ),
        density_mode=VolumeDensityMode.TEXTURE_3D,
    ),
    "fire_glow": VolumeMaterialConfig(
        params=VolumeParams(
            density_scale=0.8,
            absorption=(0.1, 0.2, 0.5),
            scattering=(0.05, 0.03, 0.01),
            phase_g=0.0,
            emission=(1.0, 0.4, 0.1),  # Orange-red glow
        ),
        density_mode=VolumeDensityMode.PROCEDURAL,
    ),
    "atmospheric": VolumeMaterialConfig(
        params=VolumeParams(
            density_scale=0.1,
            absorption=(0.001, 0.0005, 0.0002),  # Blue-ish absorption
            scattering=(0.02, 0.03, 0.05),  # Blue-ish scattering (Rayleigh)
            phase_g=0.0,
        ),
        density_mode=VolumeDensityMode.EXPONENTIAL_HEIGHT,
        phase_function=VolumePhaseFunction.RAYLEIGH,
    ),
}


def get_volume_material_preset(name: str) -> VolumeMaterialConfig:
    """Get a predefined volume material configuration.

    Args:
        name: Preset name (fog_light, fog_dense, fog_height, cloud, smoke,
              fire_glow, atmospheric).

    Returns:
        VolumeMaterialConfig for the preset.

    Raises:
        KeyError: If preset name is not found.
    """
    if name not in VOLUME_MATERIAL_PRESETS:
        available = ", ".join(VOLUME_MATERIAL_PRESETS.keys())
        raise KeyError(f"Unknown volume material preset '{name}'. Available: {available}")
    return VOLUME_MATERIAL_PRESETS[name]


__all__ = [
    "VolumeDensityMode",
    "VolumePhaseFunction",
    "VolumeParams",
    "VolumeMaterialConfig",
    "VolumeMaterialBuilder",
    "VOLUME_DOMAIN_WGSL",
    "VOLUME_WGSL_PATH",
    "VOLUME_MATERIAL_PRESETS",
    "generate_volume_material",
    "generate_volume_material_consts",
    "get_volume_entry_point",
    "get_volume_material_preset",
    "validate_volume_material_wgsl",
]
