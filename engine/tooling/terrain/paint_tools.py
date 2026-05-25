"""
Terrain painting tools for the AI Game Engine.

Provides material painting with layers, masks, and blending modes
for terrain texture application.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any
import math
import random


class PaintMode(Enum):
    """Terrain painting modes."""
    PAINT = auto()
    ERASE = auto()
    BLEND = auto()
    REPLACE = auto()


class LayerBlendMode(Enum):
    """Layer blending modes for terrain materials."""
    HEIGHT = auto()
    SLOPE = auto()
    NOISE = auto()
    COMBINED = auto()
    MANUAL = auto()


@dataclass(slots=True)
class MaskSettings:
    """Base settings for terrain masks."""
    enabled: bool = True
    invert: bool = False
    strength: float = 1.0
    feather: float = 0.1


class TerrainMask(ABC):
    """
    Abstract base class for terrain painting masks.

    Masks control where paint operations are applied based on
    terrain properties like height, slope, or noise patterns.
    """
    __slots__ = ("_settings",)

    def __init__(self, settings: Optional[MaskSettings] = None):
        """
        Initialize mask.

        Args:
            settings: Mask settings
        """
        self._settings = settings or MaskSettings()

    @property
    def settings(self) -> MaskSettings:
        """Get mask settings."""
        return self._settings

    @abstractmethod
    def evaluate(self, x: int, y: int, terrain_data: Any) -> float:
        """
        Evaluate mask at a terrain position.

        Args:
            x, y: Terrain sample coordinates
            terrain_data: Reference to terrain height/slope data

        Returns:
            Mask value between 0.0 and 1.0
        """
        pass

    def apply(self, x: int, y: int, terrain_data: Any) -> float:
        """
        Apply mask with settings.

        Args:
            x, y: Terrain sample coordinates
            terrain_data: Reference to terrain data

        Returns:
            Final mask value after applying settings
        """
        if not self._settings.enabled:
            return 1.0

        value = self.evaluate(x, y, terrain_data)

        if self._settings.invert:
            value = 1.0 - value

        value *= self._settings.strength
        return max(0.0, min(1.0, value))


@dataclass(slots=True)
class HeightMaskSettings(MaskSettings):
    """Settings for height-based mask."""
    min_height: float = 0.0
    max_height: float = 100.0


class HeightMask(TerrainMask):
    """
    Height-based terrain mask.

    Allows painting only within a specified height range.
    """
    __slots__ = ("_height_settings",)

    def __init__(self, settings: Optional[HeightMaskSettings] = None):
        """
        Initialize height mask.

        Args:
            settings: Height mask settings
        """
        self._height_settings = settings or HeightMaskSettings()
        super().__init__(self._height_settings)

    @property
    def height_settings(self) -> HeightMaskSettings:
        """Get height-specific settings."""
        return self._height_settings

    def evaluate(self, x: int, y: int, terrain_data: Any) -> float:
        """Evaluate height mask at position."""
        height = terrain_data.get_height(x, y) if hasattr(terrain_data, 'get_height') else 0.0

        min_h = self._height_settings.min_height
        max_h = self._height_settings.max_height
        feather = self._height_settings.feather

        if height < min_h - feather or height > max_h + feather:
            return 0.0

        if height < min_h:
            return (height - (min_h - feather)) / feather
        elif height > max_h:
            return 1.0 - ((height - max_h) / feather)

        return 1.0


@dataclass(slots=True)
class SlopeMaskSettings(MaskSettings):
    """Settings for slope-based mask."""
    min_angle: float = 0.0  # Degrees
    max_angle: float = 90.0  # Degrees


class SlopeMask(TerrainMask):
    """
    Slope-based terrain mask.

    Allows painting only on terrain within a specified slope range.
    """
    __slots__ = ("_slope_settings",)

    def __init__(self, settings: Optional[SlopeMaskSettings] = None):
        """
        Initialize slope mask.

        Args:
            settings: Slope mask settings
        """
        self._slope_settings = settings or SlopeMaskSettings()
        super().__init__(self._slope_settings)

    @property
    def slope_settings(self) -> SlopeMaskSettings:
        """Get slope-specific settings."""
        return self._slope_settings

    def evaluate(self, x: int, y: int, terrain_data: Any) -> float:
        """Evaluate slope mask at position."""
        # Calculate slope from height differences
        if not hasattr(terrain_data, 'get_height'):
            return 1.0

        h = terrain_data.get_height(x, y)
        h_right = terrain_data.get_height(x + 1, y)
        h_up = terrain_data.get_height(x, y + 1)

        dx = h_right - h
        dy = h_up - h
        slope = math.sqrt(dx * dx + dy * dy)
        angle = math.degrees(math.atan(slope))

        min_angle = self._slope_settings.min_angle
        max_angle = self._slope_settings.max_angle
        feather = self._slope_settings.feather * 90.0  # Convert to degrees

        if angle < min_angle - feather or angle > max_angle + feather:
            return 0.0

        if angle < min_angle:
            return (angle - (min_angle - feather)) / feather
        elif angle > max_angle:
            return 1.0 - ((angle - max_angle) / feather)

        return 1.0


@dataclass(slots=True)
class NoiseMaskSettings(MaskSettings):
    """Settings for noise-based mask."""
    seed: int = 42
    scale: float = 0.1
    threshold: float = 0.5
    octaves: int = 4


class NoiseMask(TerrainMask):
    """
    Noise-based terrain mask.

    Creates organic, procedural paint patterns using noise functions.
    """
    __slots__ = ("_noise_settings",)

    def __init__(self, settings: Optional[NoiseMaskSettings] = None):
        """
        Initialize noise mask.

        Args:
            settings: Noise mask settings
        """
        self._noise_settings = settings or NoiseMaskSettings()
        super().__init__(self._noise_settings)

    @property
    def noise_settings(self) -> NoiseMaskSettings:
        """Get noise-specific settings."""
        return self._noise_settings

    def _simple_noise(self, x: float, y: float, seed: int) -> float:
        """Simple pseudo-random noise function."""
        random.seed(int(x * 1000 + y * 1000000 + seed))
        return random.random()

    def _smoothed_noise(self, x: float, y: float, seed: int) -> float:
        """Smoothed noise with interpolation."""
        ix = int(x)
        iy = int(y)
        fx = x - ix
        fy = y - iy

        v00 = self._simple_noise(ix, iy, seed)
        v10 = self._simple_noise(ix + 1, iy, seed)
        v01 = self._simple_noise(ix, iy + 1, seed)
        v11 = self._simple_noise(ix + 1, iy + 1, seed)

        # Bilinear interpolation
        i1 = v00 * (1 - fx) + v10 * fx
        i2 = v01 * (1 - fx) + v11 * fx
        return i1 * (1 - fy) + i2 * fy

    def _fbm(self, x: float, y: float) -> float:
        """Fractal Brownian Motion noise."""
        value = 0.0
        amplitude = 1.0
        total_amplitude = 0.0

        for i in range(self._noise_settings.octaves):
            scaled_x = x * (2 ** i) * self._noise_settings.scale
            scaled_y = y * (2 ** i) * self._noise_settings.scale
            value += self._smoothed_noise(scaled_x, scaled_y, self._noise_settings.seed + i) * amplitude
            total_amplitude += amplitude
            amplitude *= 0.5

        return value / total_amplitude if total_amplitude > 0 else 0.0

    def evaluate(self, x: int, y: int, terrain_data: Any) -> float:
        """Evaluate noise mask at position."""
        noise_val = self._fbm(float(x), float(y))

        threshold = self._noise_settings.threshold
        feather = self._noise_settings.feather

        if noise_val < threshold - feather:
            return 0.0
        elif noise_val > threshold + feather:
            return 1.0
        else:
            return (noise_val - (threshold - feather)) / (2 * feather)


@dataclass(slots=True)
class PaintBrushSettings:
    """Settings for a paint brush."""
    size: float = 10.0
    strength: float = 0.5
    falloff: float = 0.5
    spacing: float = 0.25  # As fraction of brush size


@dataclass(slots=True)
class PaintBrush:
    """
    Terrain paint brush.

    Used for painting material layers onto terrain.
    """
    settings: PaintBrushSettings = field(default_factory=PaintBrushSettings)

    def get_falloff(self, distance: float, max_distance: float) -> float:
        """Calculate falloff at a distance."""
        if max_distance <= 0:
            return 0.0

        normalized = min(1.0, distance / max_distance)
        falloff_start = 1.0 - self.settings.falloff

        if normalized < falloff_start:
            return 1.0

        t = (normalized - falloff_start) / self.settings.falloff if self.settings.falloff > 0 else 1.0
        return 1.0 - (t * t * (3.0 - 2.0 * t))  # Smooth falloff

    def get_influence(self, x: float, y: float, center_x: float, center_y: float) -> float:
        """Get paint influence at a position."""
        dx = x - center_x
        dy = y - center_y
        distance = math.sqrt(dx * dx + dy * dy)
        radius = self.settings.size / 2.0

        if distance > radius:
            return 0.0

        return self.get_falloff(distance, radius) * self.settings.strength


@dataclass(slots=True)
class PaintLayer:
    """
    A layer in the terrain splatmap.

    Each layer represents a material that can be painted on the terrain.
    """
    id: int
    name: str
    material_id: str
    weights: list[list[float]] = field(default_factory=list)

    def ensure_size(self, width: int, height: int) -> None:
        """Ensure weight map is the correct size."""
        if len(self.weights) != height or (height > 0 and len(self.weights[0]) != width):
            self.weights = [[0.0 for _ in range(width)] for _ in range(height)]

    def get_weight(self, x: int, y: int) -> float:
        """Get weight at position."""
        if 0 <= y < len(self.weights) and 0 <= x < len(self.weights[0]):
            return self.weights[y][x]
        return 0.0

    def set_weight(self, x: int, y: int, value: float) -> None:
        """Set weight at position."""
        if 0 <= y < len(self.weights) and 0 <= x < len(self.weights[0]):
            self.weights[y][x] = max(0.0, min(1.0, value))


@dataclass(slots=True)
class PaintOperation:
    """Represents a paint operation for undo/redo."""
    mode: PaintMode
    layer_id: int
    center_x: float
    center_y: float
    previous_weights: dict[int, dict[tuple[int, int], float]] = field(default_factory=dict)
    new_weights: dict[int, dict[tuple[int, int], float]] = field(default_factory=dict)


class TerrainPaintTool:
    """
    Main terrain painting tool.

    Provides layer-based terrain painting with masks and blending.
    """
    __slots__ = (
        "_terrain_width",
        "_terrain_height",
        "_layers",
        "_brush",
        "_current_mode",
        "_current_layer_id",
        "_masks",
        "_operation_history",
        "_redo_stack",
    )

    def __init__(self, width: int, height: int):
        """
        Initialize paint tool.

        Args:
            width: Terrain width in samples
            height: Terrain height in samples
        """
        self._terrain_width = width
        self._terrain_height = height
        self._layers: dict[int, PaintLayer] = {}
        self._brush = PaintBrush()
        self._current_mode = PaintMode.PAINT
        self._current_layer_id: Optional[int] = None
        self._masks: list[TerrainMask] = []
        self._operation_history: list[PaintOperation] = []
        self._redo_stack: list[PaintOperation] = []

    @property
    def brush(self) -> PaintBrush:
        """Get current brush."""
        return self._brush

    @property
    def mode(self) -> PaintMode:
        """Get current paint mode."""
        return self._current_mode

    @mode.setter
    def mode(self, value: PaintMode) -> None:
        """Set paint mode."""
        self._current_mode = value

    @property
    def current_layer_id(self) -> Optional[int]:
        """Get current target layer ID."""
        return self._current_layer_id

    @current_layer_id.setter
    def current_layer_id(self, value: Optional[int]) -> None:
        """Set current target layer ID."""
        self._current_layer_id = value

    def set_brush(self, brush: PaintBrush) -> None:
        """Set the current brush."""
        self._brush = brush

    def add_layer(self, name: str, material_id: str) -> PaintLayer:
        """
        Add a new paint layer.

        Args:
            name: Layer name
            material_id: Associated material ID

        Returns:
            The created layer
        """
        layer_id = len(self._layers)
        layer = PaintLayer(id=layer_id, name=name, material_id=material_id)
        layer.ensure_size(self._terrain_width, self._terrain_height)
        self._layers[layer_id] = layer

        if self._current_layer_id is None:
            self._current_layer_id = layer_id

        return layer

    def remove_layer(self, layer_id: int) -> bool:
        """
        Remove a paint layer.

        Args:
            layer_id: Layer to remove

        Returns:
            True if removed successfully
        """
        if layer_id in self._layers:
            del self._layers[layer_id]
            if self._current_layer_id == layer_id:
                self._current_layer_id = next(iter(self._layers.keys()), None)
            return True
        return False

    def get_layer(self, layer_id: int) -> Optional[PaintLayer]:
        """Get a layer by ID."""
        return self._layers.get(layer_id)

    def get_all_layers(self) -> list[PaintLayer]:
        """Get all paint layers."""
        return list(self._layers.values())

    def add_mask(self, mask: TerrainMask) -> None:
        """Add a painting mask."""
        self._masks.append(mask)

    def remove_mask(self, mask: TerrainMask) -> bool:
        """Remove a painting mask."""
        if mask in self._masks:
            self._masks.remove(mask)
            return True
        return False

    def clear_masks(self) -> None:
        """Clear all masks."""
        self._masks.clear()

    def _evaluate_masks(self, x: int, y: int, terrain_data: Any) -> float:
        """Evaluate all masks at a position."""
        if not self._masks:
            return 1.0

        total = 1.0
        for mask in self._masks:
            total *= mask.apply(x, y, terrain_data)
        return total

    def apply(
        self,
        center_x: float,
        center_y: float,
        terrain_data: Any = None
    ) -> Optional[PaintOperation]:
        """
        Apply paint operation at a position.

        Args:
            center_x, center_y: World position
            terrain_data: Optional terrain data for mask evaluation

        Returns:
            The paint operation that was applied
        """
        if self._current_layer_id is None or self._current_layer_id not in self._layers:
            return None

        current_layer = self._layers[self._current_layer_id]
        operation = PaintOperation(
            mode=self._current_mode,
            layer_id=self._current_layer_id,
            center_x=center_x,
            center_y=center_y,
        )

        radius = int(self._brush.settings.size / 2) + 1
        cx = int(center_x)
        cy = int(center_y)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = cx + dx
                y = cy + dy

                if not (0 <= x < self._terrain_width and 0 <= y < self._terrain_height):
                    continue

                influence = self._brush.get_influence(x, y, center_x, center_y)
                if influence <= 0:
                    continue

                # Apply masks
                mask_value = self._evaluate_masks(x, y, terrain_data)
                influence *= mask_value

                if influence <= 0:
                    continue

                # Store previous weights for all layers
                for layer_id, layer in self._layers.items():
                    if layer_id not in operation.previous_weights:
                        operation.previous_weights[layer_id] = {}
                        operation.new_weights[layer_id] = {}

                    prev_weight = layer.get_weight(x, y)
                    operation.previous_weights[layer_id][(x, y)] = prev_weight

                # Apply paint based on mode
                self._apply_paint_mode(x, y, influence, operation)

        # Normalize weights if needed
        self._normalize_weights(operation)

        self._operation_history.append(operation)
        self._redo_stack.clear()

        return operation

    def _apply_paint_mode(
        self, x: int, y: int, influence: float, operation: PaintOperation
    ) -> None:
        """Apply paint mode at a position."""
        current_layer = self._layers[self._current_layer_id]
        current_weight = current_layer.get_weight(x, y)

        if self._current_mode == PaintMode.PAINT:
            new_weight = min(1.0, current_weight + influence)
            current_layer.set_weight(x, y, new_weight)
            operation.new_weights[self._current_layer_id][(x, y)] = new_weight

        elif self._current_mode == PaintMode.ERASE:
            new_weight = max(0.0, current_weight - influence)
            current_layer.set_weight(x, y, new_weight)
            operation.new_weights[self._current_layer_id][(x, y)] = new_weight

        elif self._current_mode == PaintMode.BLEND:
            # Blend increases current layer while proportionally decreasing others
            target_weight = min(1.0, current_weight + influence)
            current_layer.set_weight(x, y, target_weight)
            operation.new_weights[self._current_layer_id][(x, y)] = target_weight

            # Calculate redistribution for other layers
            added = target_weight - current_weight
            other_total = sum(
                layer.get_weight(x, y)
                for lid, layer in self._layers.items()
                if lid != self._current_layer_id
            )

            if other_total > 0 and added > 0:
                for lid, layer in self._layers.items():
                    if lid != self._current_layer_id:
                        old_weight = layer.get_weight(x, y)
                        reduction = (old_weight / other_total) * added
                        new_weight = max(0.0, old_weight - reduction)
                        layer.set_weight(x, y, new_weight)
                        operation.new_weights[lid][(x, y)] = new_weight

        elif self._current_mode == PaintMode.REPLACE:
            # Set current layer to influence, zero out others
            current_layer.set_weight(x, y, influence)
            operation.new_weights[self._current_layer_id][(x, y)] = influence

            remaining = 1.0 - influence
            other_count = len(self._layers) - 1
            other_weight = remaining / other_count if other_count > 0 else 0.0

            for lid, layer in self._layers.items():
                if lid != self._current_layer_id:
                    layer.set_weight(x, y, other_weight)
                    operation.new_weights[lid][(x, y)] = other_weight

    def _normalize_weights(self, operation: PaintOperation) -> None:
        """Normalize layer weights to sum to 1.0 at each position."""
        positions = set()
        for weights in operation.new_weights.values():
            positions.update(weights.keys())

        for x, y in positions:
            total = sum(layer.get_weight(x, y) for layer in self._layers.values())

            if total > 0 and abs(total - 1.0) > 0.001:
                for layer in self._layers.values():
                    normalized = layer.get_weight(x, y) / total
                    layer.set_weight(x, y, normalized)
                    if layer.id in operation.new_weights:
                        operation.new_weights[layer.id][(x, y)] = normalized

    def undo(self) -> Optional[PaintOperation]:
        """Undo the last paint operation."""
        if not self._operation_history:
            return None

        operation = self._operation_history.pop()

        # Restore previous weights
        for layer_id, weights in operation.previous_weights.items():
            layer = self._layers.get(layer_id)
            if layer:
                for (x, y), weight in weights.items():
                    layer.set_weight(x, y, weight)

        self._redo_stack.append(operation)
        return operation

    def redo(self) -> Optional[PaintOperation]:
        """Redo the last undone paint operation."""
        if not self._redo_stack:
            return None

        operation = self._redo_stack.pop()

        # Apply new weights
        for layer_id, weights in operation.new_weights.items():
            layer = self._layers.get(layer_id)
            if layer:
                for (x, y), weight in weights.items():
                    layer.set_weight(x, y, weight)

        self._operation_history.append(operation)
        return operation

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._operation_history) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def get_weights_at(self, x: int, y: int) -> dict[int, float]:
        """Get all layer weights at a position."""
        return {lid: layer.get_weight(x, y) for lid, layer in self._layers.items()}

    def get_splatmap(self) -> dict[int, list[list[float]]]:
        """Get complete splatmap data for all layers."""
        return {lid: layer.weights for lid, layer in self._layers.items()}
