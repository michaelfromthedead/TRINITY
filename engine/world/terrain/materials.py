"""
Terrain material layers and blending system.

Provides material layer management, weight maps for blending, and automatic
layer rules based on slope and height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol, Tuple
from typing import runtime_checkable

from engine.world.terrain.constants import (
    DEFAULT_RESOLUTION,
    MATERIAL_WEIGHT_EPSILON,
)

if TYPE_CHECKING:
    from typing import TypeAlias


class TerrainLayerType(Enum):
    """Type of terrain layer."""

    BASE = auto()  # Default coverage (always present)
    BLEND = auto()  # Painted materials (manual)
    AUTO = auto()  # Slope/height based (procedural)


class BlendTechnique(Enum):
    """Blending technique for terrain materials."""

    LINEAR = auto()  # Simple linear blend
    HEIGHT_BLEND = auto()  # Height-based blending for natural transitions
    TRIPLANAR = auto()  # Triplanar projection for cliffs
    STOCHASTIC = auto()  # Stochastic sampling to reduce tiling


@dataclass
class TerrainLayer:
    """Configuration for a single terrain material layer.

    Attributes:
        name: Human-readable layer name.
        material_id: Reference to the material asset.
        layer_type: How this layer is applied (base, blend, auto).
        tiling_scale: UV tiling scale for the material.
        normal_scale: Scale factor for normal map intensity.
        height_offset: Offset for height-based blending.
    """

    name: str = ""
    material_id: str = ""
    layer_type: TerrainLayerType = TerrainLayerType.BLEND
    tiling_scale: float = 1.0
    normal_scale: float = 1.0
    height_offset: float = 0.0

    def __post_init__(self) -> None:
        """Validate layer settings."""
        if self.tiling_scale <= 0:
            raise ValueError("tiling_scale must be > 0")
        if self.normal_scale < 0:
            raise ValueError("normal_scale must be >= 0")


@runtime_checkable
class Heightfield(Protocol):
    """Protocol for heightfield data structures."""

    @property
    def width(self) -> int:
        """Width of the heightfield in samples."""
        ...

    @property
    def height(self) -> int:
        """Height of the heightfield in samples."""
        ...

    @property
    def sample_spacing(self) -> float:
        """World units between samples."""
        ...

    def get_height_at(self, x: int, z: int) -> float:
        """Get height at sample coordinates."""
        ...

    def get_slope_at(self, x: int, z: int) -> float:
        """Get slope in degrees at sample coordinates."""
        ...


class WeightMap:
    """Stores blend weights for terrain material layers.

    Each cell in the weight map contains weights for all layers.
    Weights at each position should sum to 1.0.
    """

    def __init__(
        self,
        width: int,
        height: int,
        num_layers: int,
        default_layer: int = 0,
    ) -> None:
        """Initialize weight map.

        Args:
            width: Width in samples.
            height: Height in samples.
            num_layers: Number of material layers.
            default_layer: Layer index to initialize with full weight.
        """
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if not 0 <= default_layer < num_layers:
            raise ValueError("default_layer must be in range [0, num_layers)")

        self._width = width
        self._height = height
        self._num_layers = num_layers
        self._default_layer = default_layer

        # Initialize weights: [layer][z][x]
        self._weights: List[List[List[float]]] = [
            [[0.0 for _ in range(width)] for _ in range(height)]
            for _ in range(num_layers)
        ]

        # Set default layer to full weight
        for z in range(height):
            for x in range(width):
                self._weights[default_layer][z][x] = 1.0

    @property
    def width(self) -> int:
        """Get width in samples."""
        return self._width

    @property
    def height(self) -> int:
        """Get height in samples."""
        return self._height

    @property
    def num_layers(self) -> int:
        """Get number of layers."""
        return self._num_layers

    def _validate_coords(self, x: int, z: int) -> None:
        """Validate sample coordinates."""
        if not 0 <= x < self._width:
            raise ValueError(f"x must be in range [0, {self._width})")
        if not 0 <= z < self._height:
            raise ValueError(f"z must be in range [0, {self._height})")

    def _validate_layer(self, layer_index: int) -> None:
        """Validate layer index."""
        if not 0 <= layer_index < self._num_layers:
            raise ValueError(f"layer_index must be in range [0, {self._num_layers})")

    def get_weight_at(self, x: int, z: int, layer_index: int) -> float:
        """Get weight for a specific layer at a position.

        Args:
            x: X coordinate.
            z: Z coordinate.
            layer_index: Index of the layer.

        Returns:
            Weight value between 0 and 1.
        """
        self._validate_coords(x, z)
        self._validate_layer(layer_index)
        return self._weights[layer_index][z][x]

    def set_weight_at(self, x: int, z: int, layer_index: int, weight: float) -> None:
        """Set weight for a specific layer at a position.

        Args:
            x: X coordinate.
            z: Z coordinate.
            layer_index: Index of the layer.
            weight: Weight value (will be clamped to [0, 1]).
        """
        self._validate_coords(x, z)
        self._validate_layer(layer_index)
        self._weights[layer_index][z][x] = max(0.0, min(1.0, weight))

    def get_all_weights_at(self, x: int, z: int) -> List[float]:
        """Get all layer weights at a position.

        Args:
            x: X coordinate.
            z: Z coordinate.

        Returns:
            List of weights for all layers.
        """
        self._validate_coords(x, z)
        return [self._weights[i][z][x] for i in range(self._num_layers)]

    def normalize_at(self, x: int, z: int) -> None:
        """Normalize weights at a position to sum to 1.0.

        Args:
            x: X coordinate.
            z: Z coordinate.
        """
        self._validate_coords(x, z)

        total = sum(self._weights[i][z][x] for i in range(self._num_layers))

        if total > 0:
            for i in range(self._num_layers):
                self._weights[i][z][x] /= total
        else:
            # If all weights are 0, set default layer to 1
            self._weights[self._default_layer][z][x] = 1.0

    def normalize_all(self) -> None:
        """Normalize weights at all positions."""
        for z in range(self._height):
            for x in range(self._width):
                self.normalize_at(x, z)

    def paint(
        self,
        center_x: int,
        center_z: int,
        radius: int,
        layer_index: int,
        strength: float,
        falloff: float = 0.5,
    ) -> None:
        """Paint weight values within a circular area.

        Args:
            center_x: Center X coordinate.
            center_z: Center Z coordinate.
            radius: Radius in samples.
            layer_index: Layer to paint.
            strength: Paint strength (0-1).
            falloff: Edge falloff (0=hard, 1=soft).
        """
        self._validate_layer(layer_index)

        if radius <= 0:
            raise ValueError("radius must be > 0")
        if not 0 <= strength <= 1:
            raise ValueError("strength must be in range [0, 1]")
        if not 0 <= falloff <= 1:
            raise ValueError("falloff must be in range [0, 1]")

        min_x = max(0, center_x - radius)
        max_x = min(self._width - 1, center_x + radius)
        min_z = max(0, center_z - radius)
        max_z = min(self._height - 1, center_z + radius)

        for z in range(min_z, max_z + 1):
            for x in range(min_x, max_x + 1):
                dx = x - center_x
                dz = z - center_z
                distance = math.sqrt(dx * dx + dz * dz)

                if distance > radius:
                    continue

                # Calculate falloff
                normalized_dist = distance / radius
                inner_radius_ratio = 1.0 - falloff

                if normalized_dist <= inner_radius_ratio:
                    effect = 1.0
                elif falloff > 0:
                    t = (normalized_dist - inner_radius_ratio) / falloff
                    effect = 0.5 * (1.0 + math.cos(math.pi * t))
                else:
                    effect = 1.0

                # Apply paint
                current_weight = self._weights[layer_index][z][x]
                new_weight = current_weight + (1.0 - current_weight) * strength * effect
                self._weights[layer_index][z][x] = min(1.0, new_weight)

                # Reduce other layers proportionally
                total_other = sum(
                    self._weights[i][z][x]
                    for i in range(self._num_layers)
                    if i != layer_index
                )
                if total_other > 0:
                    reduction_factor = (1.0 - self._weights[layer_index][z][x]) / total_other
                    for i in range(self._num_layers):
                        if i != layer_index:
                            self._weights[i][z][x] *= reduction_factor

    def get_dominant_layer_at(self, x: int, z: int) -> int:
        """Get the layer with the highest weight at a position.

        Args:
            x: X coordinate.
            z: Z coordinate.

        Returns:
            Index of the dominant layer.
        """
        self._validate_coords(x, z)

        max_weight = -1.0
        dominant = 0

        for i in range(self._num_layers):
            if self._weights[i][z][x] > max_weight:
                max_weight = self._weights[i][z][x]
                dominant = i

        return dominant

    def clear_layer(self, layer_index: int) -> None:
        """Set all weights for a layer to 0.

        Args:
            layer_index: Index of the layer to clear.
        """
        self._validate_layer(layer_index)

        for z in range(self._height):
            for x in range(self._width):
                self._weights[layer_index][z][x] = 0.0

    def resize(self, new_width: int, new_height: int) -> None:
        """Resize the weight map using bilinear interpolation.

        Args:
            new_width: New width in samples.
            new_height: New height in samples.
        """
        if new_width <= 0 or new_height <= 0:
            raise ValueError("new_width and new_height must be > 0")

        # Create new weight arrays
        new_weights: List[List[List[float]]] = [
            [[0.0 for _ in range(new_width)] for _ in range(new_height)]
            for _ in range(self._num_layers)
        ]

        # Bilinear interpolation
        x_ratio = (self._width - 1) / (new_width - 1) if new_width > 1 else 0
        z_ratio = (self._height - 1) / (new_height - 1) if new_height > 1 else 0

        for nz in range(new_height):
            for nx in range(new_width):
                # Find source coordinates
                src_x = nx * x_ratio
                src_z = nz * z_ratio

                x0 = int(math.floor(src_x))
                z0 = int(math.floor(src_z))
                x1 = min(x0 + 1, self._width - 1)
                z1 = min(z0 + 1, self._height - 1)

                x_frac = src_x - x0
                z_frac = src_z - z0

                for layer in range(self._num_layers):
                    # Bilinear interpolation
                    v00 = self._weights[layer][z0][x0]
                    v10 = self._weights[layer][z0][x1]
                    v01 = self._weights[layer][z1][x0]
                    v11 = self._weights[layer][z1][x1]

                    v0 = v00 + (v10 - v00) * x_frac
                    v1 = v01 + (v11 - v01) * x_frac
                    new_weights[layer][nz][nx] = v0 + (v1 - v0) * z_frac

        self._width = new_width
        self._height = new_height
        self._weights = new_weights

        # Normalize to ensure weights sum to 1
        self.normalize_all()


@dataclass
class AutoLayerRule:
    """Rule for automatic layer application based on terrain properties.

    Attributes:
        layer_index: Index of the layer to apply.
        slope_range: (min, max) slope in degrees for this layer.
        height_range: (min, max) height for this layer, or None for any height.
        noise_scale: Scale of noise for variation.
        noise_threshold: Threshold for noise (higher = sparser application).
    """

    layer_index: int
    slope_range: Tuple[float, float] = (0.0, 90.0)
    height_range: Optional[Tuple[float, float]] = None
    noise_scale: float = 0.0
    noise_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate rule settings."""
        if self.layer_index < 0:
            raise ValueError("layer_index must be >= 0")
        if self.slope_range[0] > self.slope_range[1]:
            raise ValueError("slope_range[0] must be <= slope_range[1]")
        if self.height_range is not None and self.height_range[0] > self.height_range[1]:
            raise ValueError("height_range[0] must be <= height_range[1]")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be >= 0")
        if not 0 <= self.noise_threshold <= 1:
            raise ValueError("noise_threshold must be in range [0, 1]")

    def evaluate(
        self,
        slope: float,
        height: float,
        noise_value: float = 0.0,
    ) -> float:
        """Evaluate rule weight at given terrain properties.

        Args:
            slope: Slope in degrees.
            height: Terrain height.
            noise_value: Noise value for variation (0-1).

        Returns:
            Weight contribution (0-1).
        """
        # Check slope
        slope_min, slope_max = self.slope_range
        if slope < slope_min or slope > slope_max:
            return 0.0

        # Check height
        if self.height_range is not None:
            height_min, height_max = self.height_range
            if height < height_min or height > height_max:
                return 0.0

        # Calculate weight based on position within ranges
        slope_center = (slope_min + slope_max) / 2
        slope_half_range = (slope_max - slope_min) / 2
        if slope_half_range > 0:
            slope_weight = 1.0 - abs(slope - slope_center) / slope_half_range
        else:
            slope_weight = 1.0

        # Apply noise
        if self.noise_scale > 0:
            if noise_value < self.noise_threshold:
                return 0.0
            noise_factor = (noise_value - self.noise_threshold) / (1.0 - self.noise_threshold)
            slope_weight *= noise_factor

        return max(0.0, slope_weight)


class TerrainMaterial:
    """Manages terrain material layers and blending.

    Combines multiple terrain layers with a weight map for blending.
    """

    def __init__(
        self,
        width: int,
        height: int,
        blend_technique: BlendTechnique = BlendTechnique.HEIGHT_BLEND,
    ) -> None:
        """Initialize terrain material system.

        Args:
            width: Width of the terrain in samples.
            height: Height of the terrain in samples.
            blend_technique: Technique for blending layers.
        """
        self._width = width
        self._height = height
        self._blend_technique = blend_technique
        self._layers: List[TerrainLayer] = []
        self._weight_map: Optional[WeightMap] = None
        self._auto_rules: List[AutoLayerRule] = []

    @property
    def layers(self) -> List[TerrainLayer]:
        """Get list of layers (read-only copy)."""
        return list(self._layers)

    @property
    def layer_count(self) -> int:
        """Get number of layers."""
        return len(self._layers)

    @property
    def weight_map(self) -> Optional[WeightMap]:
        """Get the weight map."""
        return self._weight_map

    @property
    def blend_technique(self) -> BlendTechnique:
        """Get the blend technique."""
        return self._blend_technique

    @blend_technique.setter
    def blend_technique(self, value: BlendTechnique) -> None:
        """Set the blend technique."""
        self._blend_technique = value

    def add_layer(self, layer: TerrainLayer) -> int:
        """Add a new layer.

        Args:
            layer: The layer to add.

        Returns:
            Index of the added layer.
        """
        self._layers.append(layer)
        layer_index = len(self._layers) - 1

        # Recreate weight map with new layer count
        if self._weight_map is not None:
            self._recreate_weight_map()
        else:
            self._weight_map = WeightMap(
                self._width, self._height, len(self._layers), default_layer=0
            )

        return layer_index

    def remove_layer(self, layer_index: int) -> None:
        """Remove a layer.

        Args:
            layer_index: Index of the layer to remove.
        """
        if not 0 <= layer_index < len(self._layers):
            raise ValueError(f"layer_index must be in range [0, {len(self._layers)})")

        if len(self._layers) <= 1:
            raise ValueError("Cannot remove the last layer")

        self._layers.pop(layer_index)
        self._recreate_weight_map()

        # Update auto rules that reference higher indices
        updated_rules = []
        for rule in self._auto_rules:
            if rule.layer_index == layer_index:
                continue  # Remove rules for deleted layer
            if rule.layer_index > layer_index:
                # Adjust index
                updated_rules.append(
                    AutoLayerRule(
                        layer_index=rule.layer_index - 1,
                        slope_range=rule.slope_range,
                        height_range=rule.height_range,
                        noise_scale=rule.noise_scale,
                        noise_threshold=rule.noise_threshold,
                    )
                )
            else:
                updated_rules.append(rule)
        self._auto_rules = updated_rules

    def get_layer(self, layer_index: int) -> TerrainLayer:
        """Get a layer by index.

        Args:
            layer_index: Index of the layer.

        Returns:
            The terrain layer.
        """
        if not 0 <= layer_index < len(self._layers):
            raise ValueError(f"layer_index must be in range [0, {len(self._layers)})")
        return self._layers[layer_index]

    def _recreate_weight_map(self) -> None:
        """Recreate weight map when layers change."""
        if len(self._layers) == 0:
            self._weight_map = None
            return

        old_weights = self._weight_map
        self._weight_map = WeightMap(
            self._width, self._height, len(self._layers), default_layer=0
        )

        if old_weights is not None:
            # Copy existing weights where possible
            copy_count = min(old_weights.num_layers, len(self._layers))
            for layer in range(copy_count):
                for z in range(min(old_weights.height, self._height)):
                    for x in range(min(old_weights.width, self._width)):
                        self._weight_map.set_weight_at(
                            x, z, layer, old_weights.get_weight_at(x, z, layer)
                        )

            self._weight_map.normalize_all()

    def add_auto_rule(self, rule: AutoLayerRule) -> None:
        """Add an automatic layer application rule.

        Args:
            rule: The rule to add.
        """
        if rule.layer_index >= len(self._layers):
            raise ValueError(
                f"rule.layer_index must be < number of layers ({len(self._layers)})"
            )
        self._auto_rules.append(rule)

    def clear_auto_rules(self) -> None:
        """Remove all automatic layer rules."""
        self._auto_rules.clear()

    def apply_auto_rules(
        self,
        heightfield: Heightfield,
        noise_func: Optional[Callable[[float, float], float]] = None,
    ) -> None:
        """Apply automatic layer rules based on heightfield properties.

        Args:
            heightfield: The heightfield to analyze.
            noise_func: Optional noise function (x, z) -> 0-1 for variation.
        """
        if not self._auto_rules or self._weight_map is None:
            return

        for z in range(min(heightfield.height, self._height)):
            for x in range(min(heightfield.width, self._width)):
                height = heightfield.get_height_at(x, z)
                slope = heightfield.get_slope_at(x, z)

                # Get noise value if function provided
                noise = noise_func(x, z) if noise_func else 0.5

                # Evaluate all rules
                weights = [0.0] * len(self._layers)

                for rule in self._auto_rules:
                    weight = rule.evaluate(slope, height, noise)
                    weights[rule.layer_index] = max(weights[rule.layer_index], weight)

                # Normalize and apply
                total = sum(weights)
                if total > 0:
                    for layer_index, weight in enumerate(weights):
                        normalized = weight / total
                        self._weight_map.set_weight_at(x, z, layer_index, normalized)
                    self._weight_map.normalize_at(x, z)

    def get_blend_weights_at(self, x: int, z: int) -> List[float]:
        """Get blend weights at a position.

        Args:
            x: X coordinate.
            z: Z coordinate.

        Returns:
            List of weights for all layers.
        """
        if self._weight_map is None:
            return [1.0] if len(self._layers) > 0 else []
        return self._weight_map.get_all_weights_at(x, z)

    def sample_blend_weights(
        self,
        world_x: float,
        world_z: float,
        sample_spacing: float,
    ) -> List[float]:
        """Sample blend weights with bilinear interpolation.

        Args:
            world_x: World X coordinate.
            world_z: World Z coordinate.
            sample_spacing: World units between samples.

        Returns:
            Interpolated weights for all layers.
        """
        if self._weight_map is None:
            return [1.0] if len(self._layers) > 0 else []

        # Convert to sample coordinates
        sx = world_x / sample_spacing
        sz = world_z / sample_spacing

        x0 = int(math.floor(sx))
        z0 = int(math.floor(sz))
        x1 = min(x0 + 1, self._width - 1)
        z1 = min(z0 + 1, self._height - 1)

        x0 = max(0, x0)
        z0 = max(0, z0)

        x_frac = sx - x0
        z_frac = sz - z0

        # Bilinear interpolation for each layer
        weights = []
        for layer in range(len(self._layers)):
            v00 = self._weight_map.get_weight_at(x0, z0, layer)
            v10 = self._weight_map.get_weight_at(x1, z0, layer)
            v01 = self._weight_map.get_weight_at(x0, z1, layer)
            v11 = self._weight_map.get_weight_at(x1, z1, layer)

            v0 = v00 + (v10 - v00) * x_frac
            v1 = v01 + (v11 - v01) * x_frac
            weights.append(v0 + (v1 - v0) * z_frac)

        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        return weights

    def apply_height_blend(
        self,
        weights: List[float],
        height_values: List[float],
        blend_sharpness: float = 8.0,
    ) -> List[float]:
        """Apply height-based blending to weights.

        Uses displacement/height textures for more natural transitions.

        Args:
            weights: Base blend weights.
            height_values: Height/displacement values for each layer (0-1).
            blend_sharpness: Sharpness of the blend transition.

        Returns:
            Modified weights with height blending applied.
        """
        if len(weights) != len(height_values) or len(weights) != len(self._layers):
            raise ValueError("weights and height_values must match layer count")

        # Add height offset from layer configuration
        adjusted_heights = [
            height_values[i] + self._layers[i].height_offset
            for i in range(len(self._layers))
        ]

        # Height blend algorithm
        # Find maximum height at each position, blend based on distance from max
        max_height = max(
            h * w for h, w in zip(adjusted_heights, weights) if w > 0.001
        ) if any(w > 0.001 for w in weights) else 0.0

        new_weights = []
        for i, (weight, height) in enumerate(zip(weights, adjusted_heights)):
            if weight < 0.001:
                new_weights.append(0.0)
            else:
                # Calculate blend factor based on height difference
                height_diff = max_height - (height * weight)
                blend_factor = max(0.0, 1.0 - height_diff * blend_sharpness)
                new_weights.append(weight * blend_factor)

        # Normalize
        total = sum(new_weights)
        if total > 0:
            new_weights = [w / total for w in new_weights]
        elif len(weights) > 0:
            # Fallback to original weights
            return list(weights)

        return new_weights


class MaterialPalette:
    """Predefined material configurations for common terrain types."""

    @staticmethod
    def create_natural_terrain() -> List[Tuple[TerrainLayer, AutoLayerRule]]:
        """Create a natural terrain material setup.

        Returns:
            List of (layer, rule) tuples for grass, rock, dirt, and snow.
        """
        return [
            (
                TerrainLayer(
                    name="Grass",
                    material_id="mat_grass",
                    layer_type=TerrainLayerType.AUTO,
                    tiling_scale=1.0,
                ),
                AutoLayerRule(
                    layer_index=0,
                    slope_range=(0.0, 30.0),
                    height_range=(0.0, 500.0),
                ),
            ),
            (
                TerrainLayer(
                    name="Rock",
                    material_id="mat_rock",
                    layer_type=TerrainLayerType.AUTO,
                    tiling_scale=0.5,
                    height_offset=0.2,
                ),
                AutoLayerRule(
                    layer_index=1,
                    slope_range=(30.0, 90.0),
                ),
            ),
            (
                TerrainLayer(
                    name="Dirt",
                    material_id="mat_dirt",
                    layer_type=TerrainLayerType.AUTO,
                    tiling_scale=1.0,
                ),
                AutoLayerRule(
                    layer_index=2,
                    slope_range=(0.0, 45.0),
                    height_range=(0.0, 200.0),
                    noise_scale=0.1,
                    noise_threshold=0.6,
                ),
            ),
            (
                TerrainLayer(
                    name="Snow",
                    material_id="mat_snow",
                    layer_type=TerrainLayerType.AUTO,
                    tiling_scale=1.0,
                ),
                AutoLayerRule(
                    layer_index=3,
                    slope_range=(0.0, 45.0),
                    height_range=(400.0, 1000.0),
                ),
            ),
        ]

    @staticmethod
    def create_desert_terrain() -> List[Tuple[TerrainLayer, AutoLayerRule]]:
        """Create a desert terrain material setup.

        Returns:
            List of (layer, rule) tuples for sand, sandstone, and gravel.
        """
        return [
            (
                TerrainLayer(
                    name="Sand",
                    material_id="mat_sand",
                    layer_type=TerrainLayerType.AUTO,
                    tiling_scale=1.0,
                ),
                AutoLayerRule(
                    layer_index=0,
                    slope_range=(0.0, 25.0),
                ),
            ),
            (
                TerrainLayer(
                    name="Sandstone",
                    material_id="mat_sandstone",
                    layer_type=TerrainLayerType.AUTO,
                    tiling_scale=0.75,
                    height_offset=0.15,
                ),
                AutoLayerRule(
                    layer_index=1,
                    slope_range=(20.0, 90.0),
                ),
            ),
            (
                TerrainLayer(
                    name="Gravel",
                    material_id="mat_gravel",
                    layer_type=TerrainLayerType.AUTO,
                    tiling_scale=1.0,
                ),
                AutoLayerRule(
                    layer_index=2,
                    slope_range=(0.0, 40.0),
                    noise_scale=0.15,
                    noise_threshold=0.7,
                ),
            ),
        ]
