"""Advanced shading models beyond standard PBR.

This module implements advanced shading models:
- Subsurface Scattering (SSS) - Burley diffusion profiles
- Clear Coat - Secondary specular layer
- Anisotropy - Directional roughness
- Sheen - Fabric, velvet
- Iridescence - Thin film interference
- Transmission - Glass, thin surfaces
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from engine.core.math.vec import Vec2, Vec3, Vec4

__all__ = [
    "SubsurfaceProfile",
    "SubsurfaceScattering",
    "ClearCoat",
    "Anisotropy",
    "Sheen",
    "Iridescence",
    "Transmission",
    "AdvancedShadingModel",
    "ShadingModelType",
]


class ShadingModelType(Enum):
    """Types of advanced shading models."""
    SUBSURFACE = auto()
    CLEAR_COAT = auto()
    ANISOTROPY = auto()
    SHEEN = auto()
    IRIDESCENCE = auto()
    TRANSMISSION = auto()


@dataclass(slots=True)
class SubsurfaceProfile:
    """Subsurface scattering diffusion profile.

    Defines how light scatters within a translucent material.
    Uses Burley's normalized diffusion profile by default.

    Attributes:
        name: Profile name for identification
        scatter_radius: Mean free path in world units
        scatter_color: Subsurface color (absorption tint)
        falloff_color: Color at the edge of scattering
        transmittance_color: Color of transmitted light
        boundary_color_bleed: Boundary color bleeding factor
        curvature_scale: How much surface curvature affects scattering
    """
    name: str = "Default"
    scatter_radius: float = 1.0
    scatter_color: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.2, 0.1))
    falloff_color: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.37, 0.3))
    transmittance_color: Vec3 = field(default_factory=lambda: Vec3(0.88, 0.23, 0.17))
    boundary_color_bleed: float = 0.5
    curvature_scale: float = 0.75

    def get_diffusion_profile(self, num_samples: int = 16) -> List[float]:
        """Generate normalized diffusion profile samples.

        Uses Burley's approximation: R(r) = A * e^(-r/d) + B * e^(-r/(3d))

        Args:
            num_samples: Number of profile samples

        Returns:
            List of diffusion weights
        """
        d = self.scatter_radius
        samples = []
        for i in range(num_samples):
            r = (i + 0.5) * (d * 3.0) / num_samples
            # Burley normalized diffusion
            weight = (
                math.exp(-r / d) / (2.0 * math.pi * d * d)
                + math.exp(-r / (3.0 * d)) / (6.0 * math.pi * d * d)
            )
            samples.append(weight)

        # Normalize weights
        total = sum(samples)
        if total > 0:
            samples = [s / total for s in samples]

        return samples

    def to_shader_data(self) -> Dict[str, Any]:
        """Convert to shader-ready format."""
        return {
            "scatterRadius": self.scatter_radius,
            "scatterColor": (
                self.scatter_color.x,
                self.scatter_color.y,
                self.scatter_color.z,
            ),
            "falloffColor": (
                self.falloff_color.x,
                self.falloff_color.y,
                self.falloff_color.z,
            ),
            "transmittanceColor": (
                self.transmittance_color.x,
                self.transmittance_color.y,
                self.transmittance_color.z,
            ),
            "boundaryColorBleed": self.boundary_color_bleed,
            "curvatureScale": self.curvature_scale,
        }


class SubsurfaceScattering:
    """Subsurface scattering shading model.

    Implements screen-space subsurface scattering using separable
    convolution with Burley's normalized diffusion profile.

    Attributes:
        profile: Diffusion profile
        subsurface_color: Subsurface tint color
        opacity: Subsurface effect opacity
        enable_transmission: Enable back-face transmission
        transmission_tint: Color for transmitted light
    """

    __slots__ = (
        "_profile",
        "_subsurface_color",
        "_opacity",
        "_enable_transmission",
        "_transmission_tint",
        "_dirty",
    )

    # Common presets
    SKIN_PROFILE = SubsurfaceProfile(
        name="Skin",
        scatter_radius=1.0,
        scatter_color=Vec3(0.48, 0.25, 0.17),
        falloff_color=Vec3(1.0, 0.37, 0.3),
    )

    WAX_PROFILE = SubsurfaceProfile(
        name="Wax",
        scatter_radius=0.5,
        scatter_color=Vec3(0.9, 0.85, 0.7),
        falloff_color=Vec3(0.95, 0.9, 0.8),
    )

    JADE_PROFILE = SubsurfaceProfile(
        name="Jade",
        scatter_radius=0.3,
        scatter_color=Vec3(0.5, 0.9, 0.5),
        falloff_color=Vec3(0.3, 0.7, 0.3),
    )

    MILK_PROFILE = SubsurfaceProfile(
        name="Milk",
        scatter_radius=0.8,
        scatter_color=Vec3(0.95, 0.95, 0.9),
        falloff_color=Vec3(0.98, 0.98, 0.95),
    )

    def __init__(
        self,
        profile: Optional[SubsurfaceProfile] = None,
        subsurface_color: Optional[Vec3] = None,
        opacity: float = 1.0,
        enable_transmission: bool = False,
        transmission_tint: Optional[Vec3] = None,
    ) -> None:
        self._profile = profile or SubsurfaceProfile()
        self._subsurface_color = subsurface_color or Vec3(1.0, 1.0, 1.0)
        self._opacity = max(0.0, min(1.0, opacity))
        self._enable_transmission = enable_transmission
        self._transmission_tint = transmission_tint or Vec3(1.0, 0.0, 0.0)
        self._dirty = True

    @property
    def profile(self) -> SubsurfaceProfile:
        return self._profile

    @profile.setter
    def profile(self, value: SubsurfaceProfile) -> None:
        self._profile = value
        self._dirty = True

    @property
    def subsurface_color(self) -> Vec3:
        return self._subsurface_color

    @subsurface_color.setter
    def subsurface_color(self, value: Vec3) -> None:
        self._subsurface_color = value
        self._dirty = True

    @property
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        self._opacity = max(0.0, min(1.0, value))
        self._dirty = True

    @property
    def enable_transmission(self) -> bool:
        return self._enable_transmission

    @enable_transmission.setter
    def enable_transmission(self, value: bool) -> None:
        self._enable_transmission = value
        self._dirty = True

    def to_shader_data(self) -> Dict[str, Any]:
        """Convert to shader-ready format."""
        return {
            "profile": self._profile.to_shader_data(),
            "subsurfaceColor": (
                self._subsurface_color.x,
                self._subsurface_color.y,
                self._subsurface_color.z,
            ),
            "opacity": self._opacity,
            "enableTransmission": self._enable_transmission,
            "transmissionTint": (
                self._transmission_tint.x,
                self._transmission_tint.y,
                self._transmission_tint.z,
            ),
        }

    def get_shader_defines(self) -> List[str]:
        """Get shader defines for this model."""
        defines = ["HAS_SUBSURFACE_SCATTERING"]
        if self._enable_transmission:
            defines.append("HAS_SSS_TRANSMISSION")
        return defines


@dataclass(slots=True)
class ClearCoat:
    """Clear coat shading model.

    Adds a secondary specular layer on top of the base material,
    simulating automotive paint, lacquered wood, etc.

    Attributes:
        intensity: Clear coat layer intensity [0, 1]
        roughness: Clear coat layer roughness [0, 1]
        ior: Index of refraction for clear coat
        normal_map: Optional separate normal for clear coat
        tint: Optional clear coat tint color
    """
    intensity: float = 1.0
    roughness: float = 0.0
    ior: float = 1.5
    normal_map: Optional[str] = None
    tint: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))

    def __post_init__(self) -> None:
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.roughness = max(0.0, min(1.0, self.roughness))
        self.ior = max(1.0, min(3.0, self.ior))

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate clear coat parameters."""
        errors = []
        if not 0.0 <= self.intensity <= 1.0:
            errors.append("intensity must be in [0, 1]")
        if not 0.0 <= self.roughness <= 1.0:
            errors.append("roughness must be in [0, 1]")
        if not 1.0 <= self.ior <= 3.0:
            errors.append("ior must be in [1, 3]")
        return len(errors) == 0, errors

    def to_shader_data(self) -> Dict[str, Any]:
        """Convert to shader-ready format."""
        # Calculate F0 from IOR
        f0 = ((self.ior - 1.0) / (self.ior + 1.0)) ** 2
        return {
            "clearCoatIntensity": self.intensity,
            "clearCoatRoughness": self.roughness,
            "clearCoatF0": f0,
            "clearCoatTint": (self.tint.x, self.tint.y, self.tint.z),
            "hasClearCoatNormal": self.normal_map is not None,
        }

    def get_shader_defines(self) -> List[str]:
        """Get shader defines for this model."""
        defines = ["HAS_CLEAR_COAT"]
        if self.normal_map:
            defines.append("HAS_CLEAR_COAT_NORMAL")
        return defines


@dataclass(slots=True)
class Anisotropy:
    """Anisotropic reflection model.

    Creates directional roughness for materials like brushed metal,
    hair, or satin fabrics.

    Attributes:
        strength: Anisotropy strength [-1, 1]. Negative stretches
                 perpendicular to tangent, positive along tangent.
        rotation: Rotation of anisotropy direction in radians
        tangent_map: Optional tangent direction texture
    """
    strength: float = 0.0
    rotation: float = 0.0
    tangent_map: Optional[str] = None

    def __post_init__(self) -> None:
        self.strength = max(-1.0, min(1.0, self.strength))
        self.rotation = self.rotation % (2.0 * math.pi)

    def get_anisotropic_roughness(
        self,
        base_roughness: float,
    ) -> Tuple[float, float]:
        """Calculate anisotropic roughness values.

        Args:
            base_roughness: Base roughness value

        Returns:
            Tuple of (roughness_t, roughness_b) for tangent and bitangent
        """
        # GGX anisotropy parameterization
        aspect = math.sqrt(1.0 - 0.9 * abs(self.strength))
        if self.strength >= 0:
            roughness_t = base_roughness / aspect
            roughness_b = base_roughness * aspect
        else:
            roughness_t = base_roughness * aspect
            roughness_b = base_roughness / aspect
        return roughness_t, roughness_b

    def to_shader_data(self) -> Dict[str, Any]:
        """Convert to shader-ready format."""
        return {
            "anisotropyStrength": self.strength,
            "anisotropyRotation": self.rotation,
            "anisotropyDirection": (
                math.cos(self.rotation),
                math.sin(self.rotation),
            ),
            "hasAnisotropyTangentMap": self.tangent_map is not None,
        }

    def get_shader_defines(self) -> List[str]:
        """Get shader defines for this model."""
        defines = ["HAS_ANISOTROPY"]
        if self.tangent_map:
            defines.append("HAS_ANISOTROPY_TANGENT_MAP")
        return defines


@dataclass(slots=True)
class Sheen:
    """Sheen shading model for fabrics.

    Simulates the soft, fuzzy appearance of velvet, cloth,
    and other fabric materials.

    Attributes:
        color: Sheen tint color
        roughness: Sheen roughness [0, 1]
        intensity: Sheen intensity multiplier
    """
    color: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    roughness: float = 0.5
    intensity: float = 1.0

    def __post_init__(self) -> None:
        self.roughness = max(0.0, min(1.0, self.roughness))
        self.intensity = max(0.0, min(2.0, self.intensity))

    def to_shader_data(self) -> Dict[str, Any]:
        """Convert to shader-ready format."""
        return {
            "sheenColor": (self.color.x, self.color.y, self.color.z),
            "sheenRoughness": self.roughness,
            "sheenIntensity": self.intensity,
        }

    def get_shader_defines(self) -> List[str]:
        """Get shader defines for this model."""
        return ["HAS_SHEEN"]


@dataclass(slots=True)
class Iridescence:
    """Thin film iridescence model.

    Simulates thin film interference effects seen on soap bubbles,
    oil slicks, beetle shells, etc.

    Attributes:
        intensity: Iridescence effect strength [0, 1]
        ior: Index of refraction for thin film
        thickness_min: Minimum film thickness in nanometers
        thickness_max: Maximum film thickness in nanometers
        thickness_map: Optional thickness texture
    """
    intensity: float = 1.0
    ior: float = 1.3
    thickness_min: float = 100.0
    thickness_max: float = 400.0
    thickness_map: Optional[str] = None

    def __post_init__(self) -> None:
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.ior = max(1.0, min(3.0, self.ior))
        self.thickness_min = max(0.0, self.thickness_min)
        self.thickness_max = max(self.thickness_min, self.thickness_max)

    def get_interference_color(
        self,
        thickness: float,
        cos_theta: float,
    ) -> Vec3:
        """Calculate interference color for given thickness and angle.

        Args:
            thickness: Film thickness in nanometers
            cos_theta: Cosine of view angle

        Returns:
            RGB interference color
        """
        # Simplified thin film interference
        # Actual implementation would use spectral rendering
        optical_path = 2.0 * self.ior * thickness * cos_theta

        # Convert to RGB using wavelength approximation
        wavelengths = (650.0, 550.0, 450.0)  # R, G, B
        colors = []
        for wl in wavelengths:
            phase = (optical_path / wl) * 2.0 * math.pi
            intensity = 0.5 + 0.5 * math.cos(phase)
            colors.append(intensity)

        return Vec3(colors[0], colors[1], colors[2])

    def to_shader_data(self) -> Dict[str, Any]:
        """Convert to shader-ready format."""
        return {
            "iridescenceIntensity": self.intensity,
            "iridescenceIOR": self.ior,
            "iridescenceThicknessMin": self.thickness_min,
            "iridescenceThicknessMax": self.thickness_max,
            "hasIridescenceThicknessMap": self.thickness_map is not None,
        }

    def get_shader_defines(self) -> List[str]:
        """Get shader defines for this model."""
        defines = ["HAS_IRIDESCENCE"]
        if self.thickness_map:
            defines.append("HAS_IRIDESCENCE_THICKNESS_MAP")
        return defines


@dataclass(slots=True)
class Transmission:
    """Transmission model for transparent materials.

    Simulates light passing through glass, thin surfaces, etc.

    Attributes:
        factor: Transmission factor [0, 1]
        ior: Index of refraction
        color: Transmission tint
        thickness: Material thickness for absorption
        attenuation_color: Color absorbed over distance
        attenuation_distance: Distance for full attenuation
        roughness: Transmitted ray roughness
    """
    factor: float = 1.0
    ior: float = 1.5
    color: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    thickness: float = 0.0
    attenuation_color: Vec3 = field(
        default_factory=lambda: Vec3(1.0, 1.0, 1.0)
    )
    attenuation_distance: float = float("inf")
    roughness: float = 0.0

    def __post_init__(self) -> None:
        self.factor = max(0.0, min(1.0, self.factor))
        self.ior = max(1.0, min(3.0, self.ior))
        self.thickness = max(0.0, self.thickness)
        self.roughness = max(0.0, min(1.0, self.roughness))

    def get_fresnel_mix(self, cos_theta: float) -> float:
        """Calculate Fresnel-based transmission/reflection mix.

        Args:
            cos_theta: Cosine of incidence angle

        Returns:
            Reflection amount (transmission = 1 - reflection)
        """
        # Schlick approximation
        f0 = ((self.ior - 1.0) / (self.ior + 1.0)) ** 2
        return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5

    def get_attenuation(self, distance: float) -> Vec3:
        """Calculate Beer-Lambert attenuation over distance.

        Args:
            distance: Travel distance through material

        Returns:
            Attenuation color multiplier
        """
        if self.attenuation_distance == float("inf"):
            return Vec3(1.0, 1.0, 1.0)

        t = distance / self.attenuation_distance
        return Vec3(
            self.attenuation_color.x ** t,
            self.attenuation_color.y ** t,
            self.attenuation_color.z ** t,
        )

    def to_shader_data(self) -> Dict[str, Any]:
        """Convert to shader-ready format."""
        return {
            "transmissionFactor": self.factor,
            "transmissionIOR": self.ior,
            "transmissionColor": (
                self.color.x,
                self.color.y,
                self.color.z,
            ),
            "transmissionThickness": self.thickness,
            "transmissionAttenuationColor": (
                self.attenuation_color.x,
                self.attenuation_color.y,
                self.attenuation_color.z,
            ),
            "transmissionAttenuationDistance": self.attenuation_distance,
            "transmissionRoughness": self.roughness,
        }

    def get_shader_defines(self) -> List[str]:
        """Get shader defines for this model."""
        defines = ["HAS_TRANSMISSION"]
        if self.thickness > 0:
            defines.append("HAS_TRANSMISSION_VOLUME")
        return defines


@dataclass(slots=True)
class AdvancedShadingModel:
    """Container for multiple advanced shading features.

    Allows combining multiple advanced shading models on a single
    material (e.g., clear coat + anisotropy).

    Attributes:
        subsurface: Subsurface scattering model
        clear_coat: Clear coat model
        anisotropy: Anisotropic reflection model
        sheen: Sheen model for fabrics
        iridescence: Thin film iridescence
        transmission: Transmission model
    """
    subsurface: Optional[SubsurfaceScattering] = None
    clear_coat: Optional[ClearCoat] = None
    anisotropy: Optional[Anisotropy] = None
    sheen: Optional[Sheen] = None
    iridescence: Optional[Iridescence] = None
    transmission: Optional[Transmission] = None

    def get_active_models(self) -> List[ShadingModelType]:
        """Get list of active shading model types."""
        active = []
        if self.subsurface is not None:
            active.append(ShadingModelType.SUBSURFACE)
        if self.clear_coat is not None:
            active.append(ShadingModelType.CLEAR_COAT)
        if self.anisotropy is not None:
            active.append(ShadingModelType.ANISOTROPY)
        if self.sheen is not None:
            active.append(ShadingModelType.SHEEN)
        if self.iridescence is not None:
            active.append(ShadingModelType.IRIDESCENCE)
        if self.transmission is not None:
            active.append(ShadingModelType.TRANSMISSION)
        return active

    def get_all_shader_defines(self) -> List[str]:
        """Get combined shader defines for all active models."""
        defines = []
        if self.subsurface:
            defines.extend(self.subsurface.get_shader_defines())
        if self.clear_coat:
            defines.extend(self.clear_coat.get_shader_defines())
        if self.anisotropy:
            defines.extend(self.anisotropy.get_shader_defines())
        if self.sheen:
            defines.extend(self.sheen.get_shader_defines())
        if self.iridescence:
            defines.extend(self.iridescence.get_shader_defines())
        if self.transmission:
            defines.extend(self.transmission.get_shader_defines())
        return defines

    def to_shader_data(self) -> Dict[str, Any]:
        """Convert all active models to shader-ready format."""
        data: Dict[str, Any] = {}
        if self.subsurface:
            data["subsurface"] = self.subsurface.to_shader_data()
        if self.clear_coat:
            data["clearCoat"] = self.clear_coat.to_shader_data()
        if self.anisotropy:
            data["anisotropy"] = self.anisotropy.to_shader_data()
        if self.sheen:
            data["sheen"] = self.sheen.to_shader_data()
        if self.iridescence:
            data["iridescence"] = self.iridescence.to_shader_data()
        if self.transmission:
            data["transmission"] = self.transmission.to_shader_data()
        return data
