"""PBR (Physically Based Rendering) metallic-roughness model.

This module implements the standard PBR metallic-roughness workflow with:
- PBRParameters: Core PBR parameter dataclass
- PBRMaterial: Component using tracked descriptors for dirty flags
- TextureMaps: Standard PBR texture map bindings
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from engine.core.math.vec import Vec2, Vec3, Vec4

__all__ = [
    "PBRParameters",
    "PBRMaterial",
    "PBRTextureSet",
    "TextureChannel",
    "PBRWorkflow",
    "validate_pbr_parameter",
    "clamp_pbr_parameter",
]


class TextureChannel(Enum):
    """Texture channels for PBR textures."""
    R = 0
    G = 1
    B = 2
    A = 3
    RGB = 4
    RGBA = 5


class PBRWorkflow(Enum):
    """PBR workflow type."""
    METALLIC_ROUGHNESS = "metallic_roughness"
    SPECULAR_GLOSSINESS = "specular_glossiness"


@dataclass(slots=True)
class PBRParameters:
    """Core PBR metallic-roughness parameters.

    This dataclass holds all the standard PBR parameters following
    the metallic-roughness workflow.

    Attributes:
        base_color: Albedo color with alpha. Range [0,1] per channel.
                   Default (1,1,1,1) - white opaque.
        metallic: How metallic the surface is. Range [0,1].
                 0 = dielectric (plastic, wood), 1 = metal.
                 Default 0.0.
        roughness: Surface roughness. Range [0,1].
                  0 = perfectly smooth (mirror), 1 = fully rough.
                  Default 0.5.
        normal_scale: Normal map intensity multiplier. Range [0,2].
                     Default 1.0.
        ao: Ambient occlusion factor. Range [0,1].
           1 = fully lit, 0 = fully occluded.
           Default 1.0.
        emissive: Emission color and intensity. Range [0, infinity).
                 Default (0,0,0) - no emission.
    """
    base_color: Vec4 = field(default_factory=lambda: Vec4(1.0, 1.0, 1.0, 1.0))
    metallic: float = 0.0
    roughness: float = 0.5
    normal_scale: float = 1.0
    ao: float = 1.0
    emissive: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate all parameters are within valid ranges.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []

        # Validate base_color channels [0,1]
        for channel, value in [
            ("base_color.x", self.base_color.x),
            ("base_color.y", self.base_color.y),
            ("base_color.z", self.base_color.z),
            ("base_color.w", self.base_color.w),
        ]:
            if not 0.0 <= value <= 1.0:
                errors.append(f"{channel} must be in range [0,1], got {value}")

        # Validate metallic [0,1]
        if not 0.0 <= self.metallic <= 1.0:
            errors.append(
                f"metallic must be in range [0,1], got {self.metallic}"
            )

        # Validate roughness [0,1]
        if not 0.0 <= self.roughness <= 1.0:
            errors.append(
                f"roughness must be in range [0,1], got {self.roughness}"
            )

        # Validate normal_scale [0,2]
        if not 0.0 <= self.normal_scale <= 2.0:
            errors.append(
                f"normal_scale must be in range [0,2], got {self.normal_scale}"
            )

        # Validate ao [0,1]
        if not 0.0 <= self.ao <= 1.0:
            errors.append(f"ao must be in range [0,1], got {self.ao}")

        # Validate emissive channels [0, infinity)
        for channel, value in [
            ("emissive.x", self.emissive.x),
            ("emissive.y", self.emissive.y),
            ("emissive.z", self.emissive.z),
        ]:
            if value < 0.0:
                errors.append(f"{channel} must be >= 0, got {value}")

        return len(errors) == 0, errors

    def clamp(self) -> PBRParameters:
        """Return a new PBRParameters with all values clamped to valid ranges."""
        return PBRParameters(
            base_color=Vec4(
                max(0.0, min(1.0, self.base_color.x)),
                max(0.0, min(1.0, self.base_color.y)),
                max(0.0, min(1.0, self.base_color.z)),
                max(0.0, min(1.0, self.base_color.w)),
            ),
            metallic=max(0.0, min(1.0, self.metallic)),
            roughness=max(0.0, min(1.0, self.roughness)),
            normal_scale=max(0.0, min(2.0, self.normal_scale)),
            ao=max(0.0, min(1.0, self.ao)),
            emissive=Vec3(
                max(0.0, self.emissive.x),
                max(0.0, self.emissive.y),
                max(0.0, self.emissive.z),
            ),
        )

    def to_shader_data(self) -> Dict[str, Any]:
        """Convert to shader-ready data format."""
        return {
            "baseColor": (
                self.base_color.x,
                self.base_color.y,
                self.base_color.z,
                self.base_color.w,
            ),
            "metallic": self.metallic,
            "roughness": self.roughness,
            "normalScale": self.normal_scale,
            "ao": self.ao,
            "emissive": (
                self.emissive.x,
                self.emissive.y,
                self.emissive.z,
            ),
        }

    def lerp(self, other: PBRParameters, t: float) -> PBRParameters:
        """Linearly interpolate between this and another PBRParameters."""
        t = max(0.0, min(1.0, t))
        return PBRParameters(
            base_color=self.base_color.lerp(other.base_color, t),
            metallic=self.metallic + (other.metallic - self.metallic) * t,
            roughness=self.roughness + (other.roughness - self.roughness) * t,
            normal_scale=(
                self.normal_scale
                + (other.normal_scale - self.normal_scale) * t
            ),
            ao=self.ao + (other.ao - self.ao) * t,
            emissive=self.emissive.lerp(other.emissive, t),
        )


@dataclass(slots=True)
class PBRTextureSet:
    """Collection of PBR texture maps.

    Attributes:
        base_color_map: Albedo/diffuse texture path
        normal_map: Normal map texture path
        metallic_map: Metallic texture path (often packed with roughness)
        roughness_map: Roughness texture path
        ao_map: Ambient occlusion texture path
        emissive_map: Emission texture path
        metallic_channel: Which channel holds metallic data
        roughness_channel: Which channel holds roughness data
        ao_channel: Which channel holds AO data
    """
    base_color_map: Optional[str] = None
    normal_map: Optional[str] = None
    metallic_map: Optional[str] = None
    roughness_map: Optional[str] = None
    ao_map: Optional[str] = None
    emissive_map: Optional[str] = None

    # Channel configuration for packed textures
    metallic_channel: TextureChannel = TextureChannel.B
    roughness_channel: TextureChannel = TextureChannel.G
    ao_channel: TextureChannel = TextureChannel.R

    def has_any_texture(self) -> bool:
        """Check if any texture is assigned."""
        return any([
            self.base_color_map,
            self.normal_map,
            self.metallic_map,
            self.roughness_map,
            self.ao_map,
            self.emissive_map,
        ])

    def get_texture_paths(self) -> List[str]:
        """Get list of all assigned texture paths."""
        paths = []
        for tex in [
            self.base_color_map,
            self.normal_map,
            self.metallic_map,
            self.roughness_map,
            self.ao_map,
            self.emissive_map,
        ]:
            if tex:
                paths.append(tex)
        return paths


class PBRDirtyFlags:
    """Dirty flag tracking for PBR material properties.

    Each flag indicates whether the corresponding data needs
    to be re-uploaded to the GPU.
    """

    __slots__ = (
        "_base_color",
        "_metallic",
        "_roughness",
        "_normal_scale",
        "_ao",
        "_emissive",
        "_textures",
        "_all_dirty",
    )

    def __init__(self) -> None:
        self._base_color = True
        self._metallic = True
        self._roughness = True
        self._normal_scale = True
        self._ao = True
        self._emissive = True
        self._textures = True
        self._all_dirty = True

    @property
    def base_color(self) -> bool:
        return self._base_color

    @base_color.setter
    def base_color(self, value: bool) -> None:
        self._base_color = value
        if value:
            self._all_dirty = True

    @property
    def metallic(self) -> bool:
        return self._metallic

    @metallic.setter
    def metallic(self, value: bool) -> None:
        self._metallic = value
        if value:
            self._all_dirty = True

    @property
    def roughness(self) -> bool:
        return self._roughness

    @roughness.setter
    def roughness(self, value: bool) -> None:
        self._roughness = value
        if value:
            self._all_dirty = True

    @property
    def normal_scale(self) -> bool:
        return self._normal_scale

    @normal_scale.setter
    def normal_scale(self, value: bool) -> None:
        self._normal_scale = value
        if value:
            self._all_dirty = True

    @property
    def ao(self) -> bool:
        return self._ao

    @ao.setter
    def ao(self, value: bool) -> None:
        self._ao = value
        if value:
            self._all_dirty = True

    @property
    def emissive(self) -> bool:
        return self._emissive

    @emissive.setter
    def emissive(self, value: bool) -> None:
        self._emissive = value
        if value:
            self._all_dirty = True

    @property
    def textures(self) -> bool:
        return self._textures

    @textures.setter
    def textures(self, value: bool) -> None:
        self._textures = value
        if value:
            self._all_dirty = True

    def any_dirty(self) -> bool:
        """Check if any parameter is dirty."""
        return (
            self._base_color
            or self._metallic
            or self._roughness
            or self._normal_scale
            or self._ao
            or self._emissive
            or self._textures
        )

    def mark_all(self) -> None:
        """Mark all parameters as dirty."""
        self._base_color = True
        self._metallic = True
        self._roughness = True
        self._normal_scale = True
        self._ao = True
        self._emissive = True
        self._textures = True
        self._all_dirty = True

    def clear_all(self) -> None:
        """Clear all dirty flags."""
        self._base_color = False
        self._metallic = False
        self._roughness = False
        self._normal_scale = False
        self._ao = False
        self._emissive = False
        self._textures = False
        self._all_dirty = False


class PBRMaterial:
    """PBR material component with tracked descriptors.

    This class implements a PBR material following the metallic-roughness
    workflow with dirty flag tracking for efficient GPU updates.

    The dirty flag pattern works as follows:
    1. Property setter marks the flag dirty
    2. Render system checks dirty flags each frame
    3. Only dirty parameters are re-uploaded to GPU
    4. After upload, flags are cleared

    Attributes:
        material_id: Unique material identifier
        name: Material name
        params: PBR parameters
        textures: Texture set
        dirty: Dirty flags
        workflow: PBR workflow type
    """

    __slots__ = (
        "_material_id",
        "_name",
        "_base_color",
        "_metallic",
        "_roughness",
        "_normal_scale",
        "_ao",
        "_emissive",
        "_textures",
        "_dirty",
        "_workflow",
        "_gpu_buffer_handle",
        "_on_change_callbacks",
    )

    _next_id: int = 0

    def __init__(
        self,
        name: str = "PBRMaterial",
        params: Optional[PBRParameters] = None,
        textures: Optional[PBRTextureSet] = None,
        workflow: PBRWorkflow = PBRWorkflow.METALLIC_ROUGHNESS,
    ) -> None:
        PBRMaterial._next_id += 1
        self._material_id = PBRMaterial._next_id
        self._name = name

        params = params or PBRParameters()
        self._base_color = params.base_color
        self._metallic = params.metallic
        self._roughness = params.roughness
        self._normal_scale = params.normal_scale
        self._ao = params.ao
        self._emissive = params.emissive

        self._textures = textures or PBRTextureSet()
        self._dirty = PBRDirtyFlags()
        self._workflow = workflow
        self._gpu_buffer_handle: Optional[int] = None
        self._on_change_callbacks: List[Callable[[str, Any], None]] = []

    @property
    def material_id(self) -> int:
        return self._material_id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def base_color(self) -> Vec4:
        return self._base_color

    @base_color.setter
    def base_color(self, value: Vec4) -> None:
        # Validate and clamp
        value = Vec4(
            max(0.0, min(1.0, value.x)),
            max(0.0, min(1.0, value.y)),
            max(0.0, min(1.0, value.z)),
            max(0.0, min(1.0, value.w)),
        )
        if self._base_color != value:
            self._base_color = value
            self._dirty.base_color = True
            self._notify_change("base_color", value)

    @property
    def metallic(self) -> float:
        return self._metallic

    @metallic.setter
    def metallic(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._metallic != value:
            self._metallic = value
            self._dirty.metallic = True
            self._notify_change("metallic", value)

    @property
    def roughness(self) -> float:
        return self._roughness

    @roughness.setter
    def roughness(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._roughness != value:
            self._roughness = value
            self._dirty.roughness = True
            self._notify_change("roughness", value)

    @property
    def normal_scale(self) -> float:
        return self._normal_scale

    @normal_scale.setter
    def normal_scale(self, value: float) -> None:
        value = max(0.0, min(2.0, value))
        if self._normal_scale != value:
            self._normal_scale = value
            self._dirty.normal_scale = True
            self._notify_change("normal_scale", value)

    @property
    def ao(self) -> float:
        return self._ao

    @ao.setter
    def ao(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._ao != value:
            self._ao = value
            self._dirty.ao = True
            self._notify_change("ao", value)

    @property
    def emissive(self) -> Vec3:
        return self._emissive

    @emissive.setter
    def emissive(self, value: Vec3) -> None:
        value = Vec3(
            max(0.0, value.x),
            max(0.0, value.y),
            max(0.0, value.z),
        )
        if self._emissive != value:
            self._emissive = value
            self._dirty.emissive = True
            self._notify_change("emissive", value)

    @property
    def textures(self) -> PBRTextureSet:
        return self._textures

    @textures.setter
    def textures(self, value: PBRTextureSet) -> None:
        self._textures = value
        self._dirty.textures = True
        self._notify_change("textures", value)

    @property
    def dirty(self) -> PBRDirtyFlags:
        return self._dirty

    @property
    def workflow(self) -> PBRWorkflow:
        return self._workflow

    def get_parameters(self) -> PBRParameters:
        """Get current parameters as a PBRParameters dataclass."""
        return PBRParameters(
            base_color=self._base_color,
            metallic=self._metallic,
            roughness=self._roughness,
            normal_scale=self._normal_scale,
            ao=self._ao,
            emissive=self._emissive,
        )

    def set_parameters(self, params: PBRParameters) -> None:
        """Set all parameters from a PBRParameters dataclass."""
        self.base_color = params.base_color
        self.metallic = params.metallic
        self.roughness = params.roughness
        self.normal_scale = params.normal_scale
        self.ao = params.ao
        self.emissive = params.emissive

    def to_shader_data(self) -> Dict[str, Any]:
        """Convert to shader-ready data format."""
        return self.get_parameters().to_shader_data()

    def bind_gpu_buffer(self, handle: int) -> None:
        """Bind a GPU buffer handle for this material."""
        self._gpu_buffer_handle = handle

    def get_gpu_buffer_handle(self) -> Optional[int]:
        """Get the GPU buffer handle if bound."""
        return self._gpu_buffer_handle

    def on_change(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for property changes."""
        self._on_change_callbacks.append(callback)

    def _notify_change(self, property_name: str, value: Any) -> None:
        """Notify all registered callbacks of a change."""
        for callback in self._on_change_callbacks:
            callback(property_name, value)

    def clone(self, new_name: Optional[str] = None) -> PBRMaterial:
        """Create a copy of this material."""
        return PBRMaterial(
            name=new_name or f"{self._name}_copy",
            params=self.get_parameters(),
            textures=PBRTextureSet(
                base_color_map=self._textures.base_color_map,
                normal_map=self._textures.normal_map,
                metallic_map=self._textures.metallic_map,
                roughness_map=self._textures.roughness_map,
                ao_map=self._textures.ao_map,
                emissive_map=self._textures.emissive_map,
                metallic_channel=self._textures.metallic_channel,
                roughness_channel=self._textures.roughness_channel,
                ao_channel=self._textures.ao_channel,
            ),
            workflow=self._workflow,
        )

    def __repr__(self) -> str:
        return (
            f"<PBRMaterial id={self._material_id} name={self._name!r} "
            f"metallic={self._metallic:.2f} roughness={self._roughness:.2f}>"
        )


def validate_pbr_parameter(name: str, value: Any) -> Tuple[bool, str]:
    """Validate a single PBR parameter.

    Args:
        name: Parameter name
        value: Parameter value

    Returns:
        Tuple of (is_valid, error_message)
    """
    ranges = {
        "base_color": (0.0, 1.0, Vec4),
        "metallic": (0.0, 1.0, float),
        "roughness": (0.0, 1.0, float),
        "normal_scale": (0.0, 2.0, float),
        "ao": (0.0, 1.0, float),
        "emissive": (0.0, float("inf"), Vec3),
    }

    if name not in ranges:
        return False, f"Unknown parameter: {name}"

    min_val, max_val, expected_type = ranges[name]

    if expected_type == Vec4:
        if not isinstance(value, Vec4):
            return False, f"{name} must be Vec4"
        for channel in [value.x, value.y, value.z, value.w]:
            if not min_val <= channel <= max_val:
                return False, f"{name} channels must be in [{min_val}, {max_val}]"
    elif expected_type == Vec3:
        if not isinstance(value, Vec3):
            return False, f"{name} must be Vec3"
        for channel in [value.x, value.y, value.z]:
            if channel < min_val:
                return False, f"{name} channels must be >= {min_val}"
    elif expected_type == float:
        if not isinstance(value, (int, float)):
            return False, f"{name} must be a number"
        if not min_val <= value <= max_val:
            return False, f"{name} must be in [{min_val}, {max_val}]"

    return True, ""


def clamp_pbr_parameter(name: str, value: Any) -> Any:
    """Clamp a PBR parameter to its valid range.

    Args:
        name: Parameter name
        value: Parameter value

    Returns:
        Clamped value
    """
    ranges = {
        "base_color": (0.0, 1.0),
        "metallic": (0.0, 1.0),
        "roughness": (0.0, 1.0),
        "normal_scale": (0.0, 2.0),
        "ao": (0.0, 1.0),
        "emissive": (0.0, float("inf")),
    }

    if name not in ranges:
        return value

    min_val, max_val = ranges[name]

    if isinstance(value, Vec4):
        return Vec4(
            max(min_val, min(max_val, value.x)),
            max(min_val, min(max_val, value.y)),
            max(min_val, min(max_val, value.z)),
            max(min_val, min(max_val, value.w)),
        )
    elif isinstance(value, Vec3):
        return Vec3(
            max(min_val, min(max_val, value.x)),
            max(min_val, min(max_val, value.y)),
            max(min_val, min(max_val, value.z)),
        )
    elif isinstance(value, (int, float)):
        return max(min_val, min(max_val, value))

    return value
