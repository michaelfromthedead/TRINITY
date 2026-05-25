"""
Terrain material system for the AI Game Engine.

Provides material layer management with blending modes
for multi-textured terrain surfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, Callable
import math


class BlendMode(Enum):
    """Material blending modes."""
    LINEAR = auto()  # Simple linear interpolation
    HEIGHT_BASED = auto()  # Blend based on height differences
    SLOPE_BASED = auto()  # Blend based on terrain slope
    NOISE_BASED = auto()  # Blend with noise pattern
    SHARP = auto()  # Sharp transition
    OVERLAY = auto()  # Overlay blending


@dataclass(slots=True)
class TextureCoordinates:
    """UV coordinate settings for a material layer."""
    scale_u: float = 1.0
    scale_v: float = 1.0
    offset_u: float = 0.0
    offset_v: float = 0.0
    rotation: float = 0.0  # Radians


@dataclass(slots=True)
class MaterialProperties:
    """Physical properties of a terrain material."""
    roughness: float = 0.5
    metallic: float = 0.0
    normal_strength: float = 1.0
    displacement_strength: float = 0.0
    ao_strength: float = 1.0
    emissive_strength: float = 0.0


@dataclass(slots=True)
class TerrainMaterialLayer:
    """
    A single terrain material layer.

    Represents one material that can be painted on the terrain,
    with its own textures and blending properties.
    """
    id: int
    name: str
    # Texture IDs
    albedo_texture: str = ""
    normal_texture: str = ""
    roughness_texture: str = ""
    height_texture: str = ""  # For parallax/displacement
    ao_texture: str = ""
    # Coordinates
    uv_settings: TextureCoordinates = field(default_factory=TextureCoordinates)
    # Properties
    properties: MaterialProperties = field(default_factory=MaterialProperties)
    # Tint
    tint_r: float = 1.0
    tint_g: float = 1.0
    tint_b: float = 1.0
    # Blending
    blend_mode: BlendMode = BlendMode.LINEAR
    blend_sharpness: float = 0.5
    height_offset: float = 0.0  # For height-based blending
    # Visibility
    visible: bool = True
    locked: bool = False

    def get_tint(self) -> tuple[float, float, float]:
        """Get tint color as tuple."""
        return (self.tint_r, self.tint_g, self.tint_b)

    def set_tint(self, r: float, g: float, b: float) -> None:
        """Set tint color."""
        self.tint_r = max(0.0, min(1.0, r))
        self.tint_g = max(0.0, min(1.0, g))
        self.tint_b = max(0.0, min(1.0, b))


@dataclass(slots=True)
class MaterialBlendSettings:
    """Settings for material blending at a specific location."""
    height_blend_range: float = 0.1
    slope_blend_range: float = 0.2
    noise_scale: float = 0.05
    noise_strength: float = 0.3
    transition_width: float = 0.1


@dataclass(slots=True)
class MaterialSample:
    """Sampled material properties at a terrain location."""
    layer_weights: dict[int, float] = field(default_factory=dict)
    blended_roughness: float = 0.5
    blended_metallic: float = 0.0
    blended_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)


class TerrainMaterialStack:
    """
    Stack of terrain material layers.

    Manages the ordering and blending of multiple material layers.
    """
    __slots__ = (
        "_layers",
        "_layer_order",
        "_blend_settings",
        "_splatmap",
        "_width",
        "_height",
    )

    def __init__(self, width: int, height: int):
        """
        Initialize material stack.

        Args:
            width: Terrain width
            height: Terrain height
        """
        self._width = width
        self._height = height
        self._layers: dict[int, TerrainMaterialLayer] = {}
        self._layer_order: list[int] = []
        self._blend_settings = MaterialBlendSettings()
        self._splatmap: dict[int, list[list[float]]] = {}

    @property
    def blend_settings(self) -> MaterialBlendSettings:
        """Get blend settings."""
        return self._blend_settings

    def add_layer(self, name: str, **kwargs: Any) -> TerrainMaterialLayer:
        """
        Add a new material layer.

        Args:
            name: Layer name
            **kwargs: Additional layer properties

        Returns:
            The created layer
        """
        layer_id = len(self._layers)
        layer = TerrainMaterialLayer(id=layer_id, name=name, **kwargs)
        self._layers[layer_id] = layer
        self._layer_order.append(layer_id)

        # Initialize splatmap for this layer
        self._splatmap[layer_id] = [
            [0.0 for _ in range(self._width)]
            for _ in range(self._height)
        ]

        return layer

    def remove_layer(self, layer_id: int) -> bool:
        """Remove a material layer."""
        if layer_id in self._layers:
            if self._layers[layer_id].locked:
                return False
            del self._layers[layer_id]
            self._layer_order.remove(layer_id)
            if layer_id in self._splatmap:
                del self._splatmap[layer_id]
            return True
        return False

    def get_layer(self, layer_id: int) -> Optional[TerrainMaterialLayer]:
        """Get a layer by ID."""
        return self._layers.get(layer_id)

    def get_layer_by_name(self, name: str) -> Optional[TerrainMaterialLayer]:
        """Get a layer by name."""
        for layer in self._layers.values():
            if layer.name == name:
                return layer
        return None

    def get_all_layers(self) -> list[TerrainMaterialLayer]:
        """Get all layers in order."""
        return [self._layers[lid] for lid in self._layer_order if lid in self._layers]

    def move_layer(self, layer_id: int, new_index: int) -> bool:
        """Move a layer to a new position in the stack."""
        if layer_id not in self._layer_order:
            return False

        self._layer_order.remove(layer_id)
        new_index = max(0, min(len(self._layer_order), new_index))
        self._layer_order.insert(new_index, layer_id)
        return True

    def get_layer_order(self) -> list[int]:
        """Get layer IDs in render order."""
        return self._layer_order.copy()

    def set_weight(self, layer_id: int, x: int, y: int, weight: float) -> None:
        """Set splatmap weight for a layer at a position."""
        if layer_id in self._splatmap:
            if 0 <= x < self._width and 0 <= y < self._height:
                self._splatmap[layer_id][y][x] = max(0.0, min(1.0, weight))

    def get_weight(self, layer_id: int, x: int, y: int) -> float:
        """Get splatmap weight for a layer at a position."""
        if layer_id in self._splatmap:
            if 0 <= x < self._width and 0 <= y < self._height:
                return self._splatmap[layer_id][y][x]
        return 0.0

    def get_weights_at(self, x: int, y: int) -> dict[int, float]:
        """Get all layer weights at a position."""
        return {
            lid: self.get_weight(lid, x, y)
            for lid in self._layer_order
        }

    def normalize_weights(self, x: int, y: int) -> None:
        """Normalize weights at a position to sum to 1.0."""
        total = sum(self.get_weight(lid, x, y) for lid in self._layer_order)
        if total > 0:
            for lid in self._layer_order:
                current = self.get_weight(lid, x, y)
                self.set_weight(lid, x, y, current / total)

    def normalize_all_weights(self) -> None:
        """Normalize weights across the entire terrain."""
        for y in range(self._height):
            for x in range(self._width):
                self.normalize_weights(x, y)


class TerrainMaterialManager:
    """
    Manages terrain materials and their application.

    Provides high-level interface for material management,
    sampling, and shader data generation.
    """
    __slots__ = (
        "_stack",
        "_terrain_heights",
        "_width",
        "_height",
        "_noise_seed",
    )

    def __init__(
        self,
        width: int,
        height: int,
        terrain_heights: Optional[list[list[float]]] = None
    ):
        """
        Initialize material manager.

        Args:
            width: Terrain width
            height: Terrain height
            terrain_heights: Optional height data for height-based blending
        """
        self._width = width
        self._height = height
        self._terrain_heights = terrain_heights
        self._stack = TerrainMaterialStack(width, height)
        self._noise_seed = 42

    @property
    def stack(self) -> TerrainMaterialStack:
        """Get material stack."""
        return self._stack

    def set_terrain_heights(self, heights: list[list[float]]) -> None:
        """Set terrain height data."""
        self._terrain_heights = heights

    def create_default_layers(self) -> None:
        """Create a default set of terrain material layers."""
        # Base layer (usually grass or rock)
        base = self._stack.add_layer(
            "Base",
            albedo_texture="terrain/grass_albedo",
            normal_texture="terrain/grass_normal",
        )

        # Rock layer
        rock = self._stack.add_layer(
            "Rock",
            albedo_texture="terrain/rock_albedo",
            normal_texture="terrain/rock_normal",
            blend_mode=BlendMode.SLOPE_BASED,
        )
        rock.properties.roughness = 0.8

        # Dirt layer
        dirt = self._stack.add_layer(
            "Dirt",
            albedo_texture="terrain/dirt_albedo",
            normal_texture="terrain/dirt_normal",
        )
        dirt.set_tint(0.8, 0.7, 0.5)

        # Snow layer
        snow = self._stack.add_layer(
            "Snow",
            albedo_texture="terrain/snow_albedo",
            normal_texture="terrain/snow_normal",
            blend_mode=BlendMode.HEIGHT_BASED,
            height_offset=0.8,
        )
        snow.properties.roughness = 0.3

    def _get_terrain_height(self, x: int, y: int) -> float:
        """Get terrain height at a position."""
        if self._terrain_heights is None:
            return 0.0
        if 0 <= x < self._width and 0 <= y < self._height:
            return self._terrain_heights[y][x]
        return 0.0

    def _get_terrain_slope(self, x: int, y: int) -> float:
        """Get terrain slope at a position (0-1 range)."""
        if self._terrain_heights is None:
            return 0.0

        h = self._get_terrain_height(x, y)
        h_right = self._get_terrain_height(x + 1, y)
        h_up = self._get_terrain_height(x, y + 1)

        dx = h_right - h
        dy = h_up - h
        slope = math.sqrt(dx * dx + dy * dy)

        # Convert to 0-1 range (45 degrees = 1.0)
        return min(1.0, slope)

    def _get_noise(self, x: int, y: int, scale: float) -> float:
        """Simple noise function for blending."""
        import random
        random.seed(int(x * scale * 1000 + y * scale * 1000000 + self._noise_seed))
        return random.random()

    def calculate_blend_weights(
        self,
        x: int,
        y: int,
        base_weights: dict[int, float]
    ) -> dict[int, float]:
        """
        Calculate final blend weights considering blend modes.

        Args:
            x, y: Terrain position
            base_weights: Painted splatmap weights

        Returns:
            Modified weights based on blend modes
        """
        settings = self._stack.blend_settings
        final_weights: dict[int, float] = {}

        height = self._get_terrain_height(x, y)
        slope = self._get_terrain_slope(x, y)

        for layer_id, base_weight in base_weights.items():
            layer = self._stack.get_layer(layer_id)
            if layer is None or not layer.visible:
                continue

            weight = base_weight

            if layer.blend_mode == BlendMode.HEIGHT_BASED:
                # Modify weight based on height
                height_diff = height - layer.height_offset
                height_factor = 1.0 - abs(height_diff) / settings.height_blend_range
                height_factor = max(0.0, min(1.0, height_factor))
                weight *= height_factor

            elif layer.blend_mode == BlendMode.SLOPE_BASED:
                # More weight on steeper slopes
                slope_factor = slope / settings.slope_blend_range
                slope_factor = max(0.0, min(1.0, slope_factor))
                weight = weight * (1.0 - slope_factor) + slope_factor

            elif layer.blend_mode == BlendMode.NOISE_BASED:
                # Add noise variation
                noise = self._get_noise(x, y, settings.noise_scale)
                noise_factor = (noise - 0.5) * settings.noise_strength
                weight = max(0.0, min(1.0, weight + noise_factor))

            elif layer.blend_mode == BlendMode.SHARP:
                # Sharp transition
                if weight > 0.5:
                    weight = 1.0
                else:
                    weight = 0.0

            final_weights[layer_id] = weight

        # Normalize
        total = sum(final_weights.values())
        if total > 0:
            for lid in final_weights:
                final_weights[lid] /= total

        return final_weights

    def sample_material(self, x: int, y: int) -> MaterialSample:
        """
        Sample blended material properties at a position.

        Args:
            x, y: Terrain position

        Returns:
            Blended material sample
        """
        base_weights = self._stack.get_weights_at(x, y)
        final_weights = self.calculate_blend_weights(x, y, base_weights)

        sample = MaterialSample(layer_weights=final_weights)

        # Blend properties
        total_roughness = 0.0
        total_metallic = 0.0
        total_tint = [0.0, 0.0, 0.0]

        for layer_id, weight in final_weights.items():
            if weight <= 0:
                continue

            layer = self._stack.get_layer(layer_id)
            if layer is None:
                continue

            total_roughness += layer.properties.roughness * weight
            total_metallic += layer.properties.metallic * weight
            tint = layer.get_tint()
            total_tint[0] += tint[0] * weight
            total_tint[1] += tint[1] * weight
            total_tint[2] += tint[2] * weight

        sample.blended_roughness = total_roughness
        sample.blended_metallic = total_metallic
        sample.blended_tint = tuple(total_tint)

        return sample

    def get_shader_data(self) -> dict[str, Any]:
        """
        Get data for terrain shader.

        Returns:
            Dictionary containing shader-ready data
        """
        layers_data = []
        for layer in self._stack.get_all_layers():
            layers_data.append({
                "id": layer.id,
                "name": layer.name,
                "albedo_texture": layer.albedo_texture,
                "normal_texture": layer.normal_texture,
                "roughness_texture": layer.roughness_texture,
                "height_texture": layer.height_texture,
                "uv_scale": (layer.uv_settings.scale_u, layer.uv_settings.scale_v),
                "uv_offset": (layer.uv_settings.offset_u, layer.uv_settings.offset_v),
                "uv_rotation": layer.uv_settings.rotation,
                "tint": layer.get_tint(),
                "roughness": layer.properties.roughness,
                "metallic": layer.properties.metallic,
                "normal_strength": layer.properties.normal_strength,
                "blend_mode": layer.blend_mode.value,
                "blend_sharpness": layer.blend_sharpness,
            })

        return {
            "layer_count": len(layers_data),
            "layers": layers_data,
            "blend_settings": {
                "height_blend_range": self._stack.blend_settings.height_blend_range,
                "slope_blend_range": self._stack.blend_settings.slope_blend_range,
                "noise_scale": self._stack.blend_settings.noise_scale,
                "noise_strength": self._stack.blend_settings.noise_strength,
                "transition_width": self._stack.blend_settings.transition_width,
            },
        }

    def get_splatmap_textures(self) -> list[list[list[tuple[float, float, float, float]]]]:
        """
        Get splatmap data as RGBA texture data.

        Packs 4 layer weights per texture.

        Returns:
            List of RGBA textures
        """
        layers = self._stack.get_all_layers()
        num_textures = (len(layers) + 3) // 4  # 4 layers per texture

        textures: list[list[list[tuple[float, float, float, float]]]] = []

        for tex_idx in range(num_textures):
            texture: list[list[tuple[float, float, float, float]]] = []

            for y in range(self._height):
                row: list[tuple[float, float, float, float]] = []
                for x in range(self._width):
                    r = self._stack.get_weight(layers[tex_idx * 4].id, x, y) if tex_idx * 4 < len(layers) else 0.0
                    g = self._stack.get_weight(layers[tex_idx * 4 + 1].id, x, y) if tex_idx * 4 + 1 < len(layers) else 0.0
                    b = self._stack.get_weight(layers[tex_idx * 4 + 2].id, x, y) if tex_idx * 4 + 2 < len(layers) else 0.0
                    a = self._stack.get_weight(layers[tex_idx * 4 + 3].id, x, y) if tex_idx * 4 + 3 < len(layers) else 0.0
                    row.append((r, g, b, a))
                texture.append(row)

            textures.append(texture)

        return textures

    def auto_paint_by_slope(
        self,
        flat_layer_id: int,
        steep_layer_id: int,
        threshold: float = 0.5
    ) -> None:
        """
        Automatically paint materials based on slope.

        Args:
            flat_layer_id: Layer for flat areas
            steep_layer_id: Layer for steep areas
            threshold: Slope threshold (0-1)
        """
        for y in range(self._height):
            for x in range(self._width):
                slope = self._get_terrain_slope(x, y)

                if slope < threshold:
                    flat_weight = 1.0 - (slope / threshold)
                    steep_weight = slope / threshold
                else:
                    flat_weight = 0.0
                    steep_weight = 1.0

                self._stack.set_weight(flat_layer_id, x, y, flat_weight)
                self._stack.set_weight(steep_layer_id, x, y, steep_weight)

    def auto_paint_by_height(
        self,
        layer_id: int,
        min_height: float,
        max_height: float,
        feather: float = 0.1
    ) -> None:
        """
        Automatically paint a material based on height.

        Args:
            layer_id: Target layer
            min_height: Minimum height
            max_height: Maximum height
            feather: Transition feather amount
        """
        for y in range(self._height):
            for x in range(self._width):
                height = self._get_terrain_height(x, y)

                if height < min_height - feather:
                    weight = 0.0
                elif height < min_height:
                    weight = (height - (min_height - feather)) / feather
                elif height > max_height + feather:
                    weight = 0.0
                elif height > max_height:
                    weight = 1.0 - ((height - max_height) / feather)
                else:
                    weight = 1.0

                self._stack.set_weight(layer_id, x, y, weight)
