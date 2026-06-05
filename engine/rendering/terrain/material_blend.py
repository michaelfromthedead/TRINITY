"""Terrain material blending stub (T-ENV-1.12).

This module provides multi-material terrain blending for realistic
terrain surfaces with smooth transitions between material types.

Features:
- Height-based material selection
- Slope-based material blending
- Splatmap-driven material masks
- Triplanar projection for steep surfaces

Expanded by T-ENV-1.10 with full material system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from trinity.decorators import component

if TYPE_CHECKING:
    from engine.platform.rhi.resources import Texture
    from engine.rendering.materials import MaterialInstance


class BlendMode(IntEnum):
    """Material blending modes.

    Controls how materials are combined at boundaries.
    """

    LINEAR = 0       # Simple linear interpolation
    HEIGHT = 1       # Height-based blending (more natural)
    SLOPE = 2        # Slope-based transition
    COMBINED = 3     # Height + slope combination


@dataclass
class TerrainMaterial:
    """Definition of a terrain material layer.

    Attributes:
        name: Material identifier.
        albedo: Albedo/diffuse texture.
        normal: Normal map texture.
        mask: Optional material mask texture.
        uv_scale: Texture coordinate scale factor.
        height_range: Valid height range (min, max).
        slope_range: Valid slope range in degrees (min, max).
        blend_sharpness: Edge sharpness (0=soft, 1=sharp).
    """

    name: str
    albedo: Optional["Texture"] = None
    normal: Optional["Texture"] = None
    mask: Optional["Texture"] = None
    uv_scale: float = 1.0
    height_range: Tuple[float, float] = (-1000.0, 1000.0)
    slope_range: Tuple[float, float] = (0.0, 90.0)
    blend_sharpness: float = 0.5

    def __post_init__(self) -> None:
        """Validate material parameters."""
        if self.uv_scale <= 0.0:
            raise ValueError(
                f"uv_scale must be positive, got {self.uv_scale}"
            )
        if self.height_range[0] >= self.height_range[1]:
            raise ValueError(
                f"height_range min must be < max, got {self.height_range}"
            )
        if not 0.0 <= self.slope_range[0] <= 90.0:
            raise ValueError(
                f"slope_range min must be in [0, 90], got {self.slope_range[0]}"
            )
        if not 0.0 <= self.slope_range[1] <= 90.0:
            raise ValueError(
                f"slope_range max must be in [0, 90], got {self.slope_range[1]}"
            )
        if not 0.0 <= self.blend_sharpness <= 1.0:
            raise ValueError(
                f"blend_sharpness must be in [0, 1], got {self.blend_sharpness}"
            )


@dataclass
class TerrainMaterialBlendConfig:
    """Configuration for terrain material blending.

    Attributes:
        max_layers: Maximum simultaneous material layers.
        blend_mode: How materials are blended.
        triplanar_enabled: Use triplanar projection for steep slopes.
        triplanar_sharpness: Triplanar blend sharpness.
        detail_enabled: Enable detail textures.
        detail_scale: Detail texture UV scale.
    """

    max_layers: int = 4
    blend_mode: BlendMode = BlendMode.COMBINED
    triplanar_enabled: bool = True
    triplanar_sharpness: float = 8.0
    detail_enabled: bool = True
    detail_scale: float = 10.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 1 <= self.max_layers <= 8:
            raise ValueError(
                f"max_layers must be in [1, 8], got {self.max_layers}"
            )
        if self.triplanar_sharpness <= 0.0:
            raise ValueError(
                f"triplanar_sharpness must be positive, got {self.triplanar_sharpness}"
            )
        if self.detail_scale <= 0.0:
            raise ValueError(
                f"detail_scale must be positive, got {self.detail_scale}"
            )


@component
class TerrainMaterialBlend:
    """Multi-material terrain blending system.

    Manages terrain material layers and computes blending weights
    based on height, slope, and splatmap data.

    This is a stub class that will be expanded by T-ENV-1.10.

    Example:
        blend = TerrainMaterialBlend()
        blend.add_material(grass_material)
        blend.add_material(rock_material)
        blend.set_splatmap(splatmap_texture)
        weights = blend.compute_weights(height=100.0, slope=45.0)

    Attributes:
        config: Blending configuration.
        materials: Registered material layers.
    """

    # Class-level attributes for Trinity component system
    _component_name: str = "TerrainMaterialBlend"

    def __init__(
        self,
        config: Optional[TerrainMaterialBlendConfig] = None,
    ) -> None:
        """Initialize material blend system.

        Args:
            config: Blend configuration. Uses defaults if None.
        """
        self._config = config or TerrainMaterialBlendConfig()
        self._materials: Dict[str, TerrainMaterial] = {}
        self._splatmap: Optional["Texture"] = None
        self._material_order: List[str] = []

    @property
    def config(self) -> TerrainMaterialBlendConfig:
        """Get blend configuration."""
        return self._config

    @property
    def materials(self) -> Dict[str, TerrainMaterial]:
        """Get registered materials."""
        return dict(self._materials)

    @property
    def material_count(self) -> int:
        """Get number of registered materials."""
        return len(self._materials)

    @property
    def splatmap(self) -> Optional["Texture"]:
        """Get splatmap texture."""
        return self._splatmap

    def add_material(self, material: TerrainMaterial) -> bool:
        """Add a material layer.

        Args:
            material: Material to add.

        Returns:
            True if material was added, False if at max capacity.

        Raises:
            ValueError: If material name already exists.
        """
        if material.name in self._materials:
            raise ValueError(f"Material '{material.name}' already exists")

        if len(self._materials) >= self._config.max_layers:
            return False

        self._materials[material.name] = material
        self._material_order.append(material.name)
        return True

    def remove_material(self, name: str) -> bool:
        """Remove a material layer.

        Args:
            name: Name of material to remove.

        Returns:
            True if material was removed, False if not found.
        """
        if name not in self._materials:
            return False

        del self._materials[name]
        self._material_order.remove(name)
        return True

    def get_material(self, name: str) -> Optional[TerrainMaterial]:
        """Get material by name.

        Args:
            name: Material name.

        Returns:
            Material if found, None otherwise.
        """
        return self._materials.get(name)

    def set_splatmap(self, texture: "Texture") -> None:
        """Set the splatmap texture.

        The splatmap encodes material weights per-pixel in RGBA channels,
        with each channel corresponding to a material layer.

        Args:
            texture: Splatmap texture (should have 4 channels).
        """
        self._splatmap = texture

    def compute_weights(
        self,
        height: float,
        slope: float,
        splatmap_sample: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, float]:
        """Compute material blend weights for a surface point.

        Args:
            height: World-space height of the point.
            slope: Surface slope in degrees (0=flat, 90=vertical).
            splatmap_sample: Optional RGBA splatmap values (0-1 each).

        Returns:
            Dictionary mapping material names to blend weights (sum = 1.0).
        """
        if not self._materials:
            return {}

        weights: Dict[str, float] = {}
        total = 0.0

        for i, name in enumerate(self._material_order):
            material = self._materials[name]
            weight = 1.0

            # Height-based weight
            if self._config.blend_mode in (BlendMode.HEIGHT, BlendMode.COMBINED):
                h_min, h_max = material.height_range
                if height < h_min or height > h_max:
                    weight *= 0.0
                else:
                    # Smooth falloff at edges
                    range_size = h_max - h_min
                    if range_size > 0:
                        edge = range_size * 0.1
                        if height < h_min + edge:
                            weight *= (height - h_min) / edge
                        elif height > h_max - edge:
                            weight *= (h_max - height) / edge

            # Slope-based weight
            if self._config.blend_mode in (BlendMode.SLOPE, BlendMode.COMBINED):
                s_min, s_max = material.slope_range
                if slope < s_min or slope > s_max:
                    weight *= 0.0
                else:
                    # Smooth falloff at edges
                    range_size = s_max - s_min
                    if range_size > 0:
                        edge = range_size * 0.1
                        if slope < s_min + edge:
                            weight *= (slope - s_min) / edge
                        elif slope > s_max - edge:
                            weight *= (s_max - slope) / edge

            # Splatmap weight (if available)
            if splatmap_sample is not None and i < 4:
                weight *= splatmap_sample[i]

            weights[name] = max(0.0, weight)
            total += weights[name]

        # Normalize weights to sum to 1.0
        if total > 0.0:
            for name in weights:
                weights[name] /= total
        elif weights:
            # Fallback: equal distribution if all weights are zero
            equal = 1.0 / len(weights)
            for name in weights:
                weights[name] = equal

        return weights

    def get_shader_data(self) -> Dict[str, List]:
        """Get material data formatted for shader binding.

        Returns:
            Dictionary with shader-bindable arrays.
        """
        return {
            "material_count": [len(self._materials)],
            "uv_scales": [m.uv_scale for m in self._materials.values()],
            "blend_sharpness": [m.blend_sharpness for m in self._materials.values()],
        }

    def clear(self) -> None:
        """Remove all materials."""
        self._materials.clear()
        self._material_order.clear()
        self._splatmap = None
