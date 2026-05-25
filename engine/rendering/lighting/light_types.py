"""Light type definitions for the rendering engine.

Implements all light types specified in Section 6.4 of RENDERING_CONTEXT.md:
- DirectionalLight (sun/moon with CSM shadows)
- PointLight (position, radius, cube shadows)
- SpotLight (position, direction, inner/outer angle, spot shadows)
- RectAreaLight (LTC-based rectangular area light)
- DiskAreaLight (LTC-based disk area light)
- IESLight (IES profile data)
- SkyLight (cubemap-based ambient)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

from engine.core.math.vec import Vec2, Vec3, Vec4

if TYPE_CHECKING:
    from engine.rendering.lighting.shadows import ShadowMap


class LightType(Enum):
    """Enumeration of supported light types."""
    DIRECTIONAL = auto()
    POINT = auto()
    SPOT = auto()
    RECT_AREA = auto()
    DISK_AREA = auto()
    IES = auto()
    SKY = auto()


class ShadowMode(Enum):
    """Shadow casting modes from @shadow_caster decorator."""
    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"


class GIImportance(Enum):
    """GI contribution importance from @gi_contributor decorator."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ShadowCasterConfig:
    """Configuration from @shadow_caster decorator.

    Attributes:
        mode: Shadow mode (static/dynamic/none)
        resolution_scale: Scale factor for shadow map resolution (>0)
        cascade_bias: Bias for cascade shadow map selection
    """
    mode: ShadowMode = ShadowMode.DYNAMIC
    resolution_scale: float = 1.0
    cascade_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.resolution_scale <= 0:
            raise ValueError("resolution_scale must be > 0")


@dataclass
class GIContributorConfig:
    """Configuration from @gi_contributor decorator.

    Attributes:
        importance: How important this light is for GI calculations
        emissive: Whether this light contributes emissive GI
    """
    importance: GIImportance = GIImportance.MEDIUM
    emissive: bool = False


# Type variable for decorator functions
T = TypeVar('T')


def shadow_caster(
    mode: str = "dynamic",
    resolution_scale: float = 1.0,
    cascade_bias: float = 0.0,
) -> Callable[[type[T]], type[T]]:
    """Decorator to configure shadow casting for a light.

    Args:
        mode: Shadow mode ("static", "dynamic", or "none")
        resolution_scale: Scale factor for shadow map resolution
        cascade_bias: Bias for CSM cascade selection

    Returns:
        Decorated class with shadow configuration
    """
    def decorator(cls: type[T]) -> type[T]:
        shadow_mode = ShadowMode(mode)
        config = ShadowCasterConfig(
            mode=shadow_mode,
            resolution_scale=resolution_scale,
            cascade_bias=cascade_bias,
        )
        cls._shadow_caster = True  # type: ignore[attr-defined]
        cls._shadow_mode = shadow_mode  # type: ignore[attr-defined]
        cls._shadow_resolution_scale = resolution_scale  # type: ignore[attr-defined]
        cls._shadow_cascade_bias = cascade_bias  # type: ignore[attr-defined]
        cls._shadow_config = config  # type: ignore[attr-defined]
        return cls
    return decorator


def gi_contributor(
    importance: str = "medium",
    emissive: bool = False,
) -> Callable[[type[T]], type[T]]:
    """Decorator to configure GI contribution for a light.

    Args:
        importance: GI importance level ("low", "medium", "high", "critical")
        emissive: Whether this light contributes emissive GI

    Returns:
        Decorated class with GI configuration
    """
    def decorator(cls: type[T]) -> type[T]:
        gi_importance = GIImportance(importance)
        config = GIContributorConfig(
            importance=gi_importance,
            emissive=emissive,
        )
        cls._gi_contributor = True  # type: ignore[attr-defined]
        cls._gi_importance = gi_importance  # type: ignore[attr-defined]
        cls._gi_emissive = emissive  # type: ignore[attr-defined]
        cls._gi_config = config  # type: ignore[attr-defined]
        return cls
    return decorator


@dataclass
class Light(ABC):
    """Base class for all light types.

    Attributes:
        color: Light color as RGB in [0,1] range
        intensity: Light intensity multiplier (>=0)
        enabled: Whether the light is active
        shadow_config: Optional shadow casting configuration
        gi_config: Optional GI contribution configuration
    """
    color: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    intensity: float = 1.0
    enabled: bool = True
    shadow_config: Optional[ShadowCasterConfig] = None
    gi_config: Optional[GIContributorConfig] = None

    # Unique ID for light indexing
    _light_id: int = field(default=0, init=False)
    _id_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        Light._id_counter += 1
        self._light_id = Light._id_counter
        # Bridge class-level decorator config to instance fields.
        # The @shadow_caster and @gi_contributor decorators set _shadow_config
        # and _gi_config on the class, but casts_shadows reads self.shadow_config.
        # Without this bridge, every decorated light returns shadow_config=None.
        if self.shadow_config is None and hasattr(type(self), '_shadow_config'):
            self.shadow_config = type(self)._shadow_config
        if self.gi_config is None and hasattr(type(self), '_gi_config'):
            self.gi_config = type(self)._gi_config
        self._validate()

    def _validate(self) -> None:
        """Validate light parameters."""
        if self.intensity < 0:
            raise ValueError("Light intensity must be >= 0")
        # Clamp color components to [0, 1]
        self.color = Vec3(
            max(0.0, min(1.0, self.color.x)),
            max(0.0, min(1.0, self.color.y)),
            max(0.0, min(1.0, self.color.z)),
        )

    @property
    @abstractmethod
    def light_type(self) -> LightType:
        """Return the type of this light."""
        ...

    @property
    def casts_shadows(self) -> bool:
        """Check if this light casts shadows."""
        return (
            self.shadow_config is not None
            and self.shadow_config.mode != ShadowMode.NONE
        )

    @property
    def contributes_gi(self) -> bool:
        """Check if this light contributes to global illumination."""
        return self.gi_config is not None

    @abstractmethod
    def get_luminous_power(self) -> float:
        """Calculate the total luminous power (lumens) of the light."""
        ...

    def get_color_intensity(self) -> Vec3:
        """Get color multiplied by intensity."""
        return self.color * self.intensity


@dataclass
@shadow_caster(mode="dynamic", resolution_scale=1.0, cascade_bias=0.001)
@gi_contributor(importance="high", emissive=False)
class DirectionalLight(Light):
    """Directional light (sun/moon) with cascaded shadow maps.

    Attributes:
        direction: Normalized direction the light is shining
        cascade_count: Number of CSM cascades (1-4)
        cascade_distances: Distance thresholds for each cascade
        angular_diameter: Angular diameter in radians (for soft shadows)
    """
    direction: Vec3 = field(default_factory=lambda: Vec3(0.0, -1.0, 0.0))
    cascade_count: int = 4
    cascade_distances: list[float] = field(default_factory=lambda: [10.0, 30.0, 100.0, 500.0])
    angular_diameter: float = 0.00935  # Sun's angular diameter in radians

    def __post_init__(self) -> None:
        super().__post_init__()
        # Safely normalize direction, defaulting to down if zero vector
        if self.direction.length_squared() < 1e-10:
            self.direction = Vec3(0.0, -1.0, 0.0)
        else:
            self.direction = self.direction.normalized()
        if not 1 <= self.cascade_count <= 4:
            raise ValueError("cascade_count must be between 1 and 4")
        if len(self.cascade_distances) != self.cascade_count:
            # Generate default cascade distances
            self.cascade_distances = self._generate_cascade_distances()

    def _generate_cascade_distances(self) -> list[float]:
        """Generate logarithmic cascade split distances."""
        near = 0.1
        far = 500.0
        lambda_param = 0.75  # Blend between linear and logarithmic
        distances = []
        for i in range(self.cascade_count):
            t = (i + 1) / self.cascade_count
            log_split = near * (far / near) ** t
            linear_split = near + (far - near) * t
            distances.append(lambda_param * log_split + (1 - lambda_param) * linear_split)
        return distances

    @property
    def light_type(self) -> LightType:
        return LightType.DIRECTIONAL

    def get_luminous_power(self) -> float:
        """Directional lights have intensity in lux (lm/m^2)."""
        return self.intensity * 1000.0  # Approximate


@dataclass
@shadow_caster(mode="dynamic", resolution_scale=1.0, cascade_bias=0.0)
@gi_contributor(importance="medium", emissive=False)
class PointLight(Light):
    """Point light with cube shadow mapping.

    Attributes:
        position: World position of the light
        radius: Light influence radius (attenuation reaches 0 at radius)
        falloff_exponent: Attenuation falloff exponent (default 2 for inverse-square)
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    radius: float = 10.0
    falloff_exponent: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.radius <= 0:
            raise ValueError("radius must be > 0")
        if self.falloff_exponent < 0:
            raise ValueError("falloff_exponent must be >= 0")

    @property
    def light_type(self) -> LightType:
        return LightType.POINT

    def get_luminous_power(self) -> float:
        """Point lights have intensity in lumens."""
        return self.intensity * 4.0 * math.pi

    def get_attenuation(self, distance: float) -> float:
        """Calculate light attenuation at a given distance.

        Uses smooth attenuation that reaches 0 at radius:
        attenuation = saturate(1 - (d/r)^4)^2 / (d^2 + 1)
        """
        if distance >= self.radius:
            return 0.0
        d_over_r = distance / self.radius
        numerator = max(0.0, 1.0 - d_over_r ** 4) ** 2
        denominator = distance ** self.falloff_exponent + 1.0
        return numerator / denominator


@dataclass
@shadow_caster(mode="dynamic", resolution_scale=1.0, cascade_bias=0.0)
@gi_contributor(importance="medium", emissive=False)
class SpotLight(Light):
    """Spot light with spot shadow mapping.

    Attributes:
        position: World position of the light
        direction: Direction the light is pointing
        inner_angle: Inner cone angle in radians (full intensity)
        outer_angle: Outer cone angle in radians (zero intensity at edge)
        radius: Light influence radius
        falloff_exponent: Distance attenuation exponent
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=lambda: Vec3(0.0, -1.0, 0.0))
    inner_angle: float = math.radians(25.0)
    outer_angle: float = math.radians(45.0)
    radius: float = 20.0
    falloff_exponent: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        # Safely normalize direction, defaulting to down if zero vector
        if self.direction.length_squared() < 1e-10:
            self.direction = Vec3(0.0, -1.0, 0.0)
        else:
            self.direction = self.direction.normalized()
        if self.inner_angle < 0 or self.inner_angle > math.pi:
            raise ValueError("inner_angle must be in [0, pi]")
        if self.outer_angle < self.inner_angle:
            raise ValueError("outer_angle must be >= inner_angle")
        if self.radius <= 0:
            raise ValueError("radius must be > 0")

    @property
    def light_type(self) -> LightType:
        return LightType.SPOT

    def get_luminous_power(self) -> float:
        """Spot lights have intensity in candelas."""
        # Solid angle of the cone
        solid_angle = 2.0 * math.pi * (1.0 - math.cos(self.outer_angle))
        return self.intensity * solid_angle

    def get_angular_attenuation(self, to_light_dir: Vec3) -> float:
        """Calculate angular attenuation for a direction towards the light.

        Uses smooth falloff between inner and outer cone angles.
        """
        cos_angle = -to_light_dir.dot(self.direction)
        cos_inner = math.cos(self.inner_angle)
        cos_outer = math.cos(self.outer_angle)

        if cos_angle >= cos_inner:
            return 1.0
        if cos_angle <= cos_outer:
            return 0.0

        # Guard against division by zero when inner equals outer
        cos_range = cos_inner - cos_outer
        if abs(cos_range) < 1e-6:
            return 1.0 if cos_angle >= cos_inner else 0.0

        # Smooth interpolation
        t = (cos_angle - cos_outer) / cos_range
        return t * t * (3.0 - 2.0 * t)  # Smoothstep


@dataclass
@gi_contributor(importance="medium", emissive=True)
class RectAreaLight(Light):
    """Rectangular area light using Linearly Transformed Cosines (LTC).

    Attributes:
        position: Center position of the rectangle
        direction: Normal direction of the light surface
        up: Up vector defining rectangle orientation
        width: Width of the rectangle
        height: Height of the rectangle
        two_sided: Whether light emits from both sides
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=lambda: Vec3(0.0, -1.0, 0.0))
    up: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 1.0))
    width: float = 1.0
    height: float = 1.0
    two_sided: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        # Safely normalize direction, defaulting to down if zero vector
        if self.direction.length_squared() < 1e-10:
            self.direction = Vec3(0.0, -1.0, 0.0)
        else:
            self.direction = self.direction.normalized()
        # Ensure up is perpendicular to direction
        cross_result = self.direction.cross(self.up)
        if cross_result.length_squared() < 1e-10:
            # Direction and up are parallel, choose a different up vector
            self.up = Vec3(1.0, 0.0, 0.0) if abs(self.direction.x) < 0.9 else Vec3(0.0, 0.0, 1.0)
            cross_result = self.direction.cross(self.up)
        right = cross_result.normalized()
        self.up = right.cross(self.direction).normalized()
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0")

    @property
    def light_type(self) -> LightType:
        return LightType.RECT_AREA

    @property
    def area(self) -> float:
        """Calculate the area of the light source."""
        return self.width * self.height

    def get_luminous_power(self) -> float:
        """Area lights have intensity in nits (cd/m^2)."""
        return self.intensity * self.area * math.pi

    def get_corners(self) -> list[Vec3]:
        """Get the four corners of the rectangle."""
        right = self.direction.cross(self.up).normalized()
        hw = self.width * 0.5
        hh = self.height * 0.5
        return [
            self.position + right * (-hw) + self.up * (-hh),
            self.position + right * (hw) + self.up * (-hh),
            self.position + right * (hw) + self.up * (hh),
            self.position + right * (-hw) + self.up * (hh),
        ]


@dataclass
@gi_contributor(importance="medium", emissive=True)
class DiskAreaLight(Light):
    """Disk area light using Linearly Transformed Cosines (LTC).

    Attributes:
        position: Center position of the disk
        direction: Normal direction of the light surface
        disk_radius: Radius of the disk
        two_sided: Whether light emits from both sides
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=lambda: Vec3(0.0, -1.0, 0.0))
    disk_radius: float = 0.5
    two_sided: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        # Safely normalize direction, defaulting to down if zero vector
        if self.direction.length_squared() < 1e-10:
            self.direction = Vec3(0.0, -1.0, 0.0)
        else:
            self.direction = self.direction.normalized()
        if self.disk_radius <= 0:
            raise ValueError("disk_radius must be > 0")

    @property
    def light_type(self) -> LightType:
        return LightType.DISK_AREA

    @property
    def area(self) -> float:
        """Calculate the area of the disk."""
        return math.pi * self.disk_radius ** 2

    def get_luminous_power(self) -> float:
        """Disk lights have intensity in nits (cd/m^2)."""
        return self.intensity * self.area * math.pi


@dataclass
class IESProfile:
    """IES (Illuminating Engineering Society) light profile data.

    Stores photometric data for realistic light distribution patterns.
    """
    name: str = ""
    vertical_angles: list[float] = field(default_factory=list)
    horizontal_angles: list[float] = field(default_factory=list)
    candela_values: list[list[float]] = field(default_factory=list)
    lumens: float = 0.0

    def sample(self, vertical_angle: float, horizontal_angle: float) -> float:
        """Sample the candela value at given angles.

        Uses bilinear interpolation for smooth results.
        """
        if not self.vertical_angles or not self.horizontal_angles:
            return 1.0

        # Clamp angles to range
        v_angle = max(0.0, min(math.pi, vertical_angle))
        h_angle = horizontal_angle % (2.0 * math.pi)

        # Find indices for interpolation
        v_idx = self._find_index(v_angle, self.vertical_angles)
        h_idx = self._find_index(h_angle, self.horizontal_angles)

        # Bilinear interpolation
        v0, v1 = v_idx
        h0, h1 = h_idx

        if v1 >= len(self.candela_values) or h1 >= len(self.candela_values[0]):
            return self.candela_values[v0][h0] if self.candela_values else 1.0

        tv = self._interp_factor(v_angle, self.vertical_angles[v0],
                                  self.vertical_angles[min(v1, len(self.vertical_angles) - 1)])
        th = self._interp_factor(h_angle, self.horizontal_angles[h0],
                                  self.horizontal_angles[min(h1, len(self.horizontal_angles) - 1)])

        c00 = self.candela_values[v0][h0]
        c10 = self.candela_values[min(v1, len(self.candela_values) - 1)][h0]
        c01 = self.candela_values[v0][min(h1, len(self.candela_values[0]) - 1)]
        c11 = self.candela_values[min(v1, len(self.candela_values) - 1)][min(h1, len(self.candela_values[0]) - 1)]

        return (c00 * (1 - tv) * (1 - th) + c10 * tv * (1 - th) +
                c01 * (1 - tv) * th + c11 * tv * th)

    def _find_index(self, value: float, angles: list[float]) -> tuple[int, int]:
        """Find bracketing indices for interpolation."""
        for i, angle in enumerate(angles):
            if value < angle:
                return (max(0, i - 1), i)
        return (len(angles) - 1, len(angles) - 1)

    def _interp_factor(self, value: float, low: float, high: float) -> float:
        """Calculate interpolation factor."""
        if abs(high - low) < 1e-6:
            return 0.0
        return (value - low) / (high - low)


@dataclass
@shadow_caster(mode="dynamic", resolution_scale=1.0, cascade_bias=0.0)
@gi_contributor(importance="medium", emissive=False)
class IESLight(Light):
    """Light with IES profile for realistic distribution.

    Attributes:
        position: World position of the light
        direction: Primary direction (down in IES coordinates)
        profile: IES profile data
        radius: Light influence radius
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=lambda: Vec3(0.0, -1.0, 0.0))
    profile: IESProfile = field(default_factory=IESProfile)
    radius: float = 10.0

    def __post_init__(self) -> None:
        super().__post_init__()
        # Safely normalize direction, defaulting to down if zero vector
        if self.direction.length_squared() < 1e-10:
            self.direction = Vec3(0.0, -1.0, 0.0)
        else:
            self.direction = self.direction.normalized()
        if self.radius <= 0:
            raise ValueError("radius must be > 0")

    @property
    def light_type(self) -> LightType:
        return LightType.IES

    def get_luminous_power(self) -> float:
        """Return the total lumens from the IES profile."""
        if self.profile.lumens > 0:
            return self.profile.lumens * self.intensity
        return self.intensity * 4.0 * math.pi

    def get_intensity_at_direction(self, light_to_point: Vec3) -> float:
        """Get light intensity for a given direction from the light.

        Transforms the direction into IES coordinate space and samples
        the profile.
        """
        light_to_point = light_to_point.normalized()

        # Calculate vertical angle (0 = down, pi = up)
        vertical_angle = math.acos(max(-1.0, min(1.0, -light_to_point.dot(self.direction))))

        # Calculate horizontal angle
        # Project onto plane perpendicular to direction
        right = Vec3(1, 0, 0) if abs(self.direction.y) > 0.9 else Vec3(0, 1, 0)
        right = self.direction.cross(right).normalized()
        forward = right.cross(self.direction).normalized()

        proj_x = light_to_point.dot(right)
        proj_y = light_to_point.dot(forward)
        horizontal_angle = math.atan2(proj_y, proj_x)
        if horizontal_angle < 0:
            horizontal_angle += 2.0 * math.pi

        return self.profile.sample(vertical_angle, horizontal_angle) * self.intensity


@dataclass
@gi_contributor(importance="high", emissive=False)
class SkyLight(Light):
    """Sky light for ambient lighting from environment cubemap.

    Attributes:
        cubemap_path: Path to the environment cubemap texture
        rotation: Rotation angle around Y axis in radians
        lower_hemisphere_color: Color for lower hemisphere (optional override)
        use_cubemap_for_gi: Whether to use cubemap for GI or just ambient
    """
    cubemap_path: Optional[str] = None
    rotation: float = 0.0
    lower_hemisphere_color: Optional[Vec3] = None
    use_cubemap_for_gi: bool = True

    # Cubemap mip levels for roughness-based sampling
    mip_count: int = 8

    @property
    def light_type(self) -> LightType:
        return LightType.SKY

    def get_luminous_power(self) -> float:
        """Sky lights contribute ambient illumination."""
        return self.intensity * 10000.0  # Approximate sky luminance

    def sample_direction(self, direction: Vec3, roughness: float = 0.0) -> Vec3:
        """Sample the sky in a direction with optional roughness blur.

        Higher roughness samples from lower mip levels (more blurred).
        This is a placeholder - actual implementation would sample the cubemap.
        """
        # In a real implementation, this would:
        # 1. Apply rotation to the direction
        # 2. Sample the cubemap at the appropriate mip level based on roughness
        # 3. Return the sampled color

        # For now, return base color * intensity
        if direction.y < 0 and self.lower_hemisphere_color is not None:
            return self.lower_hemisphere_color * self.intensity
        return self.color * self.intensity


# Type alias for any light
AnyLight = DirectionalLight | PointLight | SpotLight | RectAreaLight | DiskAreaLight | IESLight | SkyLight
