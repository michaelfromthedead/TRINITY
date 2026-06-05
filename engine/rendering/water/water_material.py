"""Physically-based water material stub (T-ENV-1.12).

This module provides PBR water material with realistic optical
properties including absorption, scattering, and Fresnel effects.

Features:
- Wavelength-dependent light absorption
- Subsurface scattering approximation
- Fresnel reflection/refraction
- Depth-based fog and color

Expanded by T-ENV-1.8 with full material system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from trinity.decorators import component

if TYPE_CHECKING:
    from engine.platform.rhi.resources import Texture
    from engine.rendering.materials import MaterialInstance


class WaterType(IntEnum):
    """Predefined water type presets.

    Each type has characteristic optical properties.
    """

    OCEAN = 0        # Deep blue ocean water
    LAKE = 1         # Clear lake water
    RIVER = 2        # Slightly murky river
    POOL = 3         # Crystal clear pool
    SWAMP = 4        # Murky swamp water
    CUSTOM = 5       # User-defined properties


@dataclass
class WaterOpticalProperties:
    """Optical properties for water material.

    Attributes:
        absorption_color: RGB absorption coefficients (per meter).
        scattering_color: RGB scattering coefficients.
        extinction_distance: Distance for 90% light extinction.
        ior: Index of refraction (1.33 for water).
        clarity: Water clarity (0=opaque, 1=crystal clear).
    """

    absorption_color: Tuple[float, float, float] = (0.45, 0.089, 0.04)
    scattering_color: Tuple[float, float, float] = (0.02, 0.02, 0.02)
    extinction_distance: float = 15.0
    ior: float = 1.333
    clarity: float = 0.8

    def __post_init__(self) -> None:
        """Validate optical parameters."""
        for i, c in enumerate(self.absorption_color):
            if c < 0.0:
                raise ValueError(f"absorption_color[{i}] must be non-negative")
        for i, c in enumerate(self.scattering_color):
            if c < 0.0:
                raise ValueError(f"scattering_color[{i}] must be non-negative")
        if self.extinction_distance <= 0.0:
            raise ValueError(
                f"extinction_distance must be positive, got {self.extinction_distance}"
            )
        if self.ior < 1.0:
            raise ValueError(f"ior must be >= 1.0, got {self.ior}")
        if not 0.0 <= self.clarity <= 1.0:
            raise ValueError(f"clarity must be in [0, 1], got {self.clarity}")


@dataclass
class WaterSurfaceProperties:
    """Surface properties for water material.

    Attributes:
        roughness: Surface micro-roughness (affects reflections).
        foam_color: RGB color of surface foam.
        foam_intensity: Foam brightness multiplier.
        specular_intensity: Specular highlight intensity.
        normal_strength: Normal map influence.
    """

    roughness: float = 0.05
    foam_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    foam_intensity: float = 0.8
    specular_intensity: float = 1.0
    normal_strength: float = 1.0

    def __post_init__(self) -> None:
        """Validate surface parameters."""
        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError(f"roughness must be in [0, 1], got {self.roughness}")
        if self.foam_intensity < 0.0:
            raise ValueError(
                f"foam_intensity must be non-negative, got {self.foam_intensity}"
            )
        if self.specular_intensity < 0.0:
            raise ValueError(
                f"specular_intensity must be non-negative, got {self.specular_intensity}"
            )


@dataclass
class WaterMaterialConfig:
    """Configuration for water material.

    Attributes:
        water_type: Predefined water type preset.
        optical: Optical properties (absorption, scattering).
        surface: Surface properties (roughness, foam).
        depth_fog_enabled: Enable depth-based fog.
        caustics_enabled: Enable underwater caustics.
    """

    water_type: WaterType = WaterType.OCEAN
    optical: WaterOpticalProperties = field(default_factory=WaterOpticalProperties)
    surface: WaterSurfaceProperties = field(default_factory=WaterSurfaceProperties)
    depth_fog_enabled: bool = True
    caustics_enabled: bool = True


# Predefined optical properties for water types
WATER_TYPE_PRESETS: Dict[WaterType, WaterOpticalProperties] = {
    WaterType.OCEAN: WaterOpticalProperties(
        absorption_color=(0.45, 0.089, 0.04),
        scattering_color=(0.02, 0.02, 0.02),
        extinction_distance=15.0,
        clarity=0.7,
    ),
    WaterType.LAKE: WaterOpticalProperties(
        absorption_color=(0.2, 0.1, 0.05),
        scattering_color=(0.01, 0.01, 0.01),
        extinction_distance=25.0,
        clarity=0.85,
    ),
    WaterType.RIVER: WaterOpticalProperties(
        absorption_color=(0.3, 0.15, 0.08),
        scattering_color=(0.05, 0.04, 0.03),
        extinction_distance=10.0,
        clarity=0.6,
    ),
    WaterType.POOL: WaterOpticalProperties(
        absorption_color=(0.1, 0.05, 0.02),
        scattering_color=(0.005, 0.005, 0.005),
        extinction_distance=50.0,
        clarity=0.95,
    ),
    WaterType.SWAMP: WaterOpticalProperties(
        absorption_color=(0.5, 0.4, 0.2),
        scattering_color=(0.15, 0.12, 0.08),
        extinction_distance=3.0,
        clarity=0.3,
    ),
}


@component
class WaterMaterial:
    """Physically-based water material.

    Implements realistic water rendering with absorption, scattering,
    Fresnel effects, and subsurface approximation.

    This is a stub class that will be expanded by T-ENV-1.8.

    Example:
        material = WaterMaterial(water_type=WaterType.OCEAN)
        material.set_normal_map(normal_texture)
        color = material.compute_color(view_dir, depth=5.0)

    Attributes:
        config: Material configuration.
        water_type: Current water type preset.
    """

    # Class-level attributes for Trinity component system
    _component_name: str = "WaterMaterial"

    def __init__(
        self,
        config: Optional[WaterMaterialConfig] = None,
        water_type: WaterType = WaterType.OCEAN,
    ) -> None:
        """Initialize water material.

        Args:
            config: Material configuration. Uses defaults if None.
            water_type: Water type preset (overridden by config if provided).
        """
        if config is not None:
            self._config = config
        else:
            # Create config with preset optical properties
            optical = WATER_TYPE_PRESETS.get(
                water_type, WaterOpticalProperties()
            )
            self._config = WaterMaterialConfig(
                water_type=water_type,
                optical=optical,
            )

        self._normal_map: Optional["Texture"] = None
        self._foam_map: Optional["Texture"] = None
        self._caustics_map: Optional["Texture"] = None

    @property
    def config(self) -> WaterMaterialConfig:
        """Get material configuration."""
        return self._config

    @property
    def water_type(self) -> WaterType:
        """Get water type preset."""
        return self._config.water_type

    @property
    def optical(self) -> WaterOpticalProperties:
        """Get optical properties."""
        return self._config.optical

    @property
    def surface(self) -> WaterSurfaceProperties:
        """Get surface properties."""
        return self._config.surface

    def set_water_type(self, water_type: WaterType) -> None:
        """Apply a water type preset.

        Args:
            water_type: Preset to apply.
        """
        if water_type in WATER_TYPE_PRESETS:
            self._config.water_type = water_type
            self._config.optical = WATER_TYPE_PRESETS[water_type]

    def set_normal_map(self, texture: "Texture") -> None:
        """Set water surface normal map.

        Args:
            texture: Normal map texture.
        """
        self._normal_map = texture

    def set_foam_map(self, texture: "Texture") -> None:
        """Set foam mask texture.

        Args:
            texture: Foam mask texture.
        """
        self._foam_map = texture

    def set_caustics_map(self, texture: "Texture") -> None:
        """Set underwater caustics texture.

        Args:
            texture: Caustics pattern texture.
        """
        self._caustics_map = texture

    def compute_fresnel(
        self, cos_theta: float, f0: float = 0.02
    ) -> float:
        """Compute Fresnel reflection coefficient.

        Uses Schlick's approximation for efficiency.

        Args:
            cos_theta: Cosine of angle between view and normal.
            f0: Reflection at normal incidence (0.02 for water).

        Returns:
            Fresnel reflection coefficient (0-1).
        """
        if cos_theta < 0.0:
            cos_theta = 0.0
        elif cos_theta > 1.0:
            cos_theta = 1.0

        # Schlick's approximation
        return f0 + (1.0 - f0) * ((1.0 - cos_theta) ** 5)

    def compute_absorption(
        self, depth: float
    ) -> Tuple[float, float, float]:
        """Compute light absorption at a given depth.

        Args:
            depth: Water depth in meters.

        Returns:
            RGB absorption factors (0-1, where 1=no absorption).
        """
        if depth <= 0.0:
            return (1.0, 1.0, 1.0)

        optical = self._config.optical
        extinction = optical.extinction_distance

        # Beer-Lambert law approximation
        factor = -depth / extinction
        r = min(1.0, max(0.0, 1.0 - optical.absorption_color[0] * (-factor)))
        g = min(1.0, max(0.0, 1.0 - optical.absorption_color[1] * (-factor)))
        b = min(1.0, max(0.0, 1.0 - optical.absorption_color[2] * (-factor)))

        return (r, g, b)

    def compute_fog_color(self, depth: float) -> Tuple[float, float, float]:
        """Compute underwater fog color at depth.

        Args:
            depth: Water depth in meters.

        Returns:
            RGB fog color.
        """
        # Blend from clear to absorption color with depth
        absorption = self.compute_absorption(depth)
        optical = self._config.optical

        # Fog color is inverse of absorption (what remains after filtering)
        r = 0.1 + 0.3 * (1.0 - optical.absorption_color[0]) * absorption[0]
        g = 0.2 + 0.5 * (1.0 - optical.absorption_color[1]) * absorption[1]
        b = 0.3 + 0.6 * (1.0 - optical.absorption_color[2]) * absorption[2]

        return (r, g, b)

    def get_shader_params(self) -> Dict[str, any]:
        """Get material parameters for shader binding.

        Returns:
            Dictionary with shader-bindable parameters.
        """
        optical = self._config.optical
        surface = self._config.surface

        return {
            "absorption_color": list(optical.absorption_color),
            "scattering_color": list(optical.scattering_color),
            "extinction_distance": optical.extinction_distance,
            "ior": optical.ior,
            "clarity": optical.clarity,
            "roughness": surface.roughness,
            "foam_color": list(surface.foam_color),
            "foam_intensity": surface.foam_intensity,
            "specular_intensity": surface.specular_intensity,
            "normal_strength": surface.normal_strength,
        }
