"""PBR Type Definitions for TRINITY Material System.

T-MAT-3.1: Python mirrors of WGSL PBR structs for type checking and validation.

This module provides Python dataclasses that mirror the WGSL PBR structs,
enabling:
- Type-safe material property definitions
- Default value consistency between Python and WGSL
- Validation before GPU upload
- Code generation for material serialization

Example::

    from trinity.materials.pbr_types import PBRParams

    # Create default PBR parameters
    params = PBRParams()

    # Create custom metal material
    gold = PBRParams(
        base_color=(1.0, 0.766, 0.336),  # Gold albedo
        metallic=1.0,
        roughness=0.3,
    )

    # Validate parameters
    is_valid, errors = gold.validate()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# Type aliases for vector types
Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]


@dataclass
class PBRInput:
    """Python mirror of WGSL PBRInput struct.

    Data from vertex shader and scene uniforms passed to surface().

    Attributes:
        world_position: World-space position of the fragment.
        world_normal: World-space normal (may not be normalized).
        world_tangent: World-space tangent with handedness in w.
        world_view: Normalized view direction (fragment to camera).
        uv: Primary UV coordinates.
        vertex_color: Vertex color (linear, premultiplied alpha).
        time: Time in seconds since scene start.
        light_count: Number of active lights affecting this fragment.
    """

    world_position: Vec3 = (0.0, 0.0, 0.0)
    world_normal: Vec3 = (0.0, 1.0, 0.0)
    world_tangent: Vec4 = (1.0, 0.0, 0.0, 1.0)
    world_view: Vec3 = (0.0, 0.0, 1.0)
    uv: Vec2 = (0.0, 0.0)
    vertex_color: Vec4 = (1.0, 1.0, 1.0, 1.0)
    time: float = 0.0
    light_count: int = 0


@dataclass
class PBRParams:
    """Python mirror of WGSL PBRParams struct.

    Material properties output by surface() that define surface appearance.

    Attributes:
        base_color: Base diffuse/albedo color (linear RGB).
        normal: Tangent-space normal perturbation.
        roughness: Surface roughness (0=mirror, 1=fully diffuse).
        metallic: Metallic factor (0=dielectric, 1=metal).
        specular: Specular reflectance for dielectrics at normal incidence.
        occlusion: Ambient occlusion (0=occluded, 1=no occlusion).
        emissive: Emissive color (linear RGB, HDR values allowed).
        alpha: Alpha/opacity (0=transparent, 1=opaque).
        subsurface: Subsurface scattering intensity.
        anisotropy: Anisotropic roughness factor (-1 to 1).
        clearcoat: Clearcoat layer intensity.
        clearcoat_roughness: Clearcoat layer roughness.
    """

    base_color: Vec3 = (1.0, 1.0, 1.0)
    normal: Vec3 = (0.0, 0.0, 1.0)
    roughness: float = 0.5
    metallic: float = 0.0
    specular: float = 0.5
    occlusion: float = 1.0
    emissive: Vec3 = (0.0, 0.0, 0.0)
    alpha: float = 1.0
    subsurface: float = 0.0
    anisotropy: float = 0.0
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.0

    def validate(self) -> Tuple[bool, list[str]]:
        """Validate all parameters are within valid ranges.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: list[str] = []

        # Validate base_color components (0-1 for non-HDR)
        for i, c in enumerate(self.base_color):
            if c < 0.0:
                errors.append(f"base_color[{i}] must be >= 0.0, got {c}")

        # Validate normal is approximately unit length
        nx, ny, nz = self.normal
        length_sq = nx * nx + ny * ny + nz * nz
        if abs(length_sq - 1.0) > 0.01:
            errors.append(f"normal should be unit length, got length^2={length_sq}")

        # Validate 0-1 range parameters
        for name, value in [
            ("roughness", self.roughness),
            ("metallic", self.metallic),
            ("specular", self.specular),
            ("occlusion", self.occlusion),
            ("alpha", self.alpha),
            ("subsurface", self.subsurface),
            ("clearcoat", self.clearcoat),
            ("clearcoat_roughness", self.clearcoat_roughness),
        ]:
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be in [0, 1], got {value}")

        # Validate anisotropy is in -1 to 1 range
        if not -1.0 <= self.anisotropy <= 1.0:
            errors.append(f"anisotropy must be in [-1, 1], got {self.anisotropy}")

        # Validate emissive components are non-negative
        for i, c in enumerate(self.emissive):
            if c < 0.0:
                errors.append(f"emissive[{i}] must be >= 0.0, got {c}")

        return (len(errors) == 0, errors)

    def clamp(self) -> "PBRParams":
        """Return a copy with all values clamped to valid ranges.

        Returns:
            New PBRParams with clamped values.
        """

        def clamp01(v: float) -> float:
            return max(0.0, min(1.0, v))

        def clamp_vec3_positive(v: Vec3) -> Vec3:
            return (max(0.0, v[0]), max(0.0, v[1]), max(0.0, v[2]))

        return PBRParams(
            base_color=clamp_vec3_positive(self.base_color),
            normal=self.normal,  # Don't clamp, but could normalize
            roughness=clamp01(self.roughness),
            metallic=clamp01(self.metallic),
            specular=clamp01(self.specular),
            occlusion=clamp01(self.occlusion),
            emissive=clamp_vec3_positive(self.emissive),
            alpha=clamp01(self.alpha),
            subsurface=clamp01(self.subsurface),
            anisotropy=max(-1.0, min(1.0, self.anisotropy)),
            clearcoat=clamp01(self.clearcoat),
            clearcoat_roughness=clamp01(self.clearcoat_roughness),
        )


@dataclass
class PBROutput:
    """Python mirror of WGSL PBROutput struct.

    Final fragment shader output produced by BRDF evaluation.

    Attributes:
        color: Final fragment color (linear RGBA, pre-multiplied alpha).
    """

    color: Vec4 = (0.0, 0.0, 0.0, 1.0)


# WGSL source for the PBR structs (embedded for compile-time access)
def get_pbr_structs_wgsl() -> str:
    """Load the PBR structs WGSL source.

    Returns:
        WGSL source code for PBRInput, PBRParams, PBROutput structs.
    """
    wgsl_path = Path(__file__).parent / "wgsl" / "pbr_structs.wgsl"
    return wgsl_path.read_text(encoding="utf-8")


# Embedded WGSL for when file access is not available (e.g., frozen builds)
PBR_STRUCTS_WGSL = '''// PBR Struct Definitions for TRINITY Material System
// T-MAT-3.1: Foundational BRDF structs

struct PBRInput {
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    world_tangent: vec4<f32>,
    world_view: vec3<f32>,
    uv: vec2<f32>,
    vertex_color: vec4<f32>,
    time: f32,
    light_count: u32,
}

struct PBRParams {
    base_color: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metallic: f32,
    specular: f32,
    occlusion: f32,
    emissive: vec3<f32>,
    alpha: f32,
    subsurface: f32,
    anisotropy: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
}

struct PBROutput {
    color: vec4<f32>,
}

fn pbr_params_default() -> PBRParams {
    var params: PBRParams;
    params.base_color = vec3<f32>(1.0, 1.0, 1.0);
    params.normal = vec3<f32>(0.0, 0.0, 1.0);
    params.roughness = 0.5;
    params.metallic = 0.0;
    params.specular = 0.5;
    params.occlusion = 1.0;
    params.emissive = vec3<f32>(0.0, 0.0, 0.0);
    params.alpha = 1.0;
    params.subsurface = 0.0;
    params.anisotropy = 0.0;
    params.clearcoat = 0.0;
    params.clearcoat_roughness = 0.0;
    return params;
}
'''

# Field metadata for code generation and serialization
PBR_PARAMS_FIELDS = {
    "base_color": {"type": "vec3<f32>", "default": (1.0, 1.0, 1.0), "range": ">=0"},
    "normal": {"type": "vec3<f32>", "default": (0.0, 0.0, 1.0), "range": "unit"},
    "roughness": {"type": "f32", "default": 0.5, "range": "[0,1]"},
    "metallic": {"type": "f32", "default": 0.0, "range": "[0,1]"},
    "specular": {"type": "f32", "default": 0.5, "range": "[0,1]"},
    "occlusion": {"type": "f32", "default": 1.0, "range": "[0,1]"},
    "emissive": {"type": "vec3<f32>", "default": (0.0, 0.0, 0.0), "range": ">=0"},
    "alpha": {"type": "f32", "default": 1.0, "range": "[0,1]"},
    "subsurface": {"type": "f32", "default": 0.0, "range": "[0,1]"},
    "anisotropy": {"type": "f32", "default": 0.0, "range": "[-1,1]"},
    "clearcoat": {"type": "f32", "default": 0.0, "range": "[0,1]"},
    "clearcoat_roughness": {"type": "f32", "default": 0.0, "range": "[0,1]"},
}


__all__ = [
    "PBRInput",
    "PBRParams",
    "PBROutput",
    "get_pbr_structs_wgsl",
    "PBR_STRUCTS_WGSL",
    "PBR_PARAMS_FIELDS",
]
