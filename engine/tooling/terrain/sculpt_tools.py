"""
Terrain sculpting tools for the AI Game Engine.

Provides sculpt modes: Raise, Lower, Smooth, Flatten, Noise, Level, Stamp
with configurable brush shapes, sizes, and falloff curves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any
import math
import random


class SculptMode(Enum):
    """Terrain sculpting modes."""
    RAISE = auto()
    LOWER = auto()
    SMOOTH = auto()
    FLATTEN = auto()
    NOISE = auto()
    LEVEL = auto()
    STAMP = auto()


class BrushShape(Enum):
    """Brush shape types."""
    CIRCLE = auto()
    SQUARE = auto()
    CUSTOM = auto()


class FalloffCurve(Enum):
    """Brush falloff curve types."""
    LINEAR = auto()
    SMOOTH = auto()
    SPHERE = auto()
    TIP = auto()
    FLAT = auto()
    CUSTOM = auto()


@dataclass(slots=True)
class BrushSettings:
    """Settings for a terrain brush."""
    size: float = 10.0
    strength: float = 0.5
    falloff: float = 0.5
    shape: BrushShape = BrushShape.CIRCLE
    falloff_curve: FalloffCurve = FalloffCurve.SMOOTH
    custom_shape_data: Optional[list[list[float]]] = None
    custom_falloff_func: Optional[Callable[[float], float]] = None


@dataclass(slots=True)
class TerrainBrush:
    """
    Terrain brush for sculpting operations.

    Supports multiple shapes and falloff curves for precise terrain editing.
    """
    settings: BrushSettings = field(default_factory=BrushSettings)

    def get_falloff(self, distance: float, max_distance: float) -> float:
        """
        Calculate falloff value at a given distance.

        Args:
            distance: Distance from brush center
            max_distance: Maximum brush radius

        Returns:
            Falloff multiplier between 0.0 and 1.0
        """
        if max_distance <= 0:
            return 0.0

        normalized = min(1.0, distance / max_distance)

        # Apply falloff setting
        falloff_start = 1.0 - self.settings.falloff
        if normalized < falloff_start:
            return 1.0

        falloff_normalized = (normalized - falloff_start) / self.settings.falloff if self.settings.falloff > 0 else 1.0

        curve = self.settings.falloff_curve
        if curve == FalloffCurve.LINEAR:
            return 1.0 - falloff_normalized
        elif curve == FalloffCurve.SMOOTH:
            # Hermite interpolation
            t = falloff_normalized
            return 1.0 - (t * t * (3.0 - 2.0 * t))
        elif curve == FalloffCurve.SPHERE:
            return math.sqrt(max(0, 1.0 - falloff_normalized * falloff_normalized))
        elif curve == FalloffCurve.TIP:
            return (1.0 - falloff_normalized) ** 2
        elif curve == FalloffCurve.FLAT:
            return 1.0 if normalized < 0.9 else 0.0
        elif curve == FalloffCurve.CUSTOM and self.settings.custom_falloff_func:
            return self.settings.custom_falloff_func(falloff_normalized)

        return 1.0 - falloff_normalized

    def is_in_brush(self, x: float, y: float, center_x: float, center_y: float) -> bool:
        """
        Check if a point is within the brush area.

        Args:
            x, y: Point coordinates
            center_x, center_y: Brush center coordinates

        Returns:
            True if point is within brush area
        """
        dx = x - center_x
        dy = y - center_y
        radius = self.settings.size / 2.0

        if self.settings.shape == BrushShape.CIRCLE:
            return (dx * dx + dy * dy) <= (radius * radius)
        elif self.settings.shape == BrushShape.SQUARE:
            return abs(dx) <= radius and abs(dy) <= radius
        elif self.settings.shape == BrushShape.CUSTOM and self.settings.custom_shape_data:
            # Sample custom shape texture
            shape_data = self.settings.custom_shape_data
            if not shape_data:
                return False
            size = len(shape_data)
            sx = int((dx / radius + 1.0) * 0.5 * (size - 1))
            sy = int((dy / radius + 1.0) * 0.5 * (size - 1))
            if 0 <= sx < size and 0 <= sy < size:
                return shape_data[sy][sx] > 0.5
            return False

        return False

    def get_influence(self, x: float, y: float, center_x: float, center_y: float) -> float:
        """
        Get brush influence at a point.

        Args:
            x, y: Point coordinates
            center_x, center_y: Brush center coordinates

        Returns:
            Influence value between 0.0 and strength
        """
        if not self.is_in_brush(x, y, center_x, center_y):
            return 0.0

        dx = x - center_x
        dy = y - center_y
        radius = self.settings.size / 2.0

        if self.settings.shape == BrushShape.CIRCLE:
            distance = math.sqrt(dx * dx + dy * dy)
        elif self.settings.shape == BrushShape.SQUARE:
            distance = max(abs(dx), abs(dy))
        else:
            distance = math.sqrt(dx * dx + dy * dy)

        falloff = self.get_falloff(distance, radius)
        return falloff * self.settings.strength


@dataclass(slots=True)
class SculptOperation:
    """
    Represents a single sculpt operation for undo/redo support.
    """
    mode: SculptMode
    center_x: float
    center_y: float
    brush: TerrainBrush
    affected_heights: dict[tuple[int, int], float] = field(default_factory=dict)
    previous_heights: dict[tuple[int, int], float] = field(default_factory=dict)
    timestamp: float = 0.0


class TerrainData:
    """
    Terrain heightmap data container.

    Manages a 2D grid of height values with chunked update support.
    """
    __slots__ = ("_width", "_height", "_heights", "_dirty_chunks", "_chunk_size")

    def __init__(self, width: int, height: int, chunk_size: int = 32):
        """
        Initialize terrain data.

        Args:
            width: Terrain width in samples
            height: Terrain height in samples
            chunk_size: Size of chunks for dirty tracking
        """
        self._width = width
        self._height = height
        self._heights: list[list[float]] = [[0.0 for _ in range(width)] for _ in range(height)]
        self._chunk_size = chunk_size
        self._dirty_chunks: set[tuple[int, int]] = set()

    @property
    def width(self) -> int:
        """Get terrain width."""
        return self._width

    @property
    def height(self) -> int:
        """Get terrain height."""
        return self._height

    def get_height(self, x: int, y: int) -> float:
        """
        Get height at a sample position.

        Args:
            x, y: Sample coordinates

        Returns:
            Height value
        """
        if 0 <= x < self._width and 0 <= y < self._height:
            return self._heights[y][x]
        return 0.0

    def set_height(self, x: int, y: int, value: float) -> None:
        """
        Set height at a sample position.

        Args:
            x, y: Sample coordinates
            value: New height value
        """
        if 0 <= x < self._width and 0 <= y < self._height:
            self._heights[y][x] = value
            chunk_x = x // self._chunk_size
            chunk_y = y // self._chunk_size
            self._dirty_chunks.add((chunk_x, chunk_y))

    def get_dirty_chunks(self) -> set[tuple[int, int]]:
        """Get set of dirty chunk coordinates."""
        return self._dirty_chunks.copy()

    def clear_dirty_chunks(self) -> None:
        """Clear dirty chunk tracking."""
        self._dirty_chunks.clear()

    def get_average_height(self, x: int, y: int, radius: int) -> float:
        """
        Get average height in a radius around a point.

        Args:
            x, y: Center coordinates
            radius: Sampling radius

        Returns:
            Average height value
        """
        total = 0.0
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self._width and 0 <= ny < self._height:
                    total += self._heights[ny][nx]
                    count += 1
        return total / count if count > 0 else 0.0


class TerrainSculptTool:
    """
    Main terrain sculpting tool.

    Provides all sculpt operations with real-time chunked updates
    and undo/redo support.
    """
    __slots__ = (
        "_terrain",
        "_brush",
        "_current_mode",
        "_operation_history",
        "_redo_stack",
        "_stamp_data",
        "_level_height",
        "_noise_seed",
        "_noise_scale",
    )

    def __init__(self, terrain: TerrainData):
        """
        Initialize sculpt tool.

        Args:
            terrain: Terrain data to edit
        """
        self._terrain = terrain
        self._brush = TerrainBrush()
        self._current_mode = SculptMode.RAISE
        self._operation_history: list[SculptOperation] = []
        self._redo_stack: list[SculptOperation] = []
        self._stamp_data: Optional[list[list[float]]] = None
        self._level_height: float = 0.0
        self._noise_seed: int = 42
        self._noise_scale: float = 1.0

    @property
    def terrain(self) -> TerrainData:
        """Get terrain data."""
        return self._terrain

    @property
    def brush(self) -> TerrainBrush:
        """Get current brush."""
        return self._brush

    @property
    def mode(self) -> SculptMode:
        """Get current sculpt mode."""
        return self._current_mode

    @mode.setter
    def mode(self, value: SculptMode) -> None:
        """Set sculpt mode."""
        self._current_mode = value

    def set_brush(self, brush: TerrainBrush) -> None:
        """
        Set the current brush.

        Args:
            brush: New brush to use
        """
        self._brush = brush

    def set_stamp_data(self, data: list[list[float]]) -> None:
        """
        Set stamp heightmap data.

        Args:
            data: 2D array of height values for stamping
        """
        self._stamp_data = data

    def set_level_height(self, height: float) -> None:
        """
        Set target height for level mode.

        Args:
            height: Target height value
        """
        self._level_height = height

    def set_noise_params(self, seed: int, scale: float) -> None:
        """
        Set noise generation parameters.

        Args:
            seed: Random seed
            scale: Noise scale factor
        """
        self._noise_seed = seed
        self._noise_scale = scale

    def apply(self, center_x: float, center_y: float, delta_time: float = 1.0) -> SculptOperation:
        """
        Apply sculpt operation at a position.

        Args:
            center_x, center_y: World position to apply brush
            delta_time: Time delta for time-based operations

        Returns:
            The operation that was applied
        """
        operation = SculptOperation(
            mode=self._current_mode,
            center_x=center_x,
            center_y=center_y,
            brush=TerrainBrush(settings=BrushSettings(
                size=self._brush.settings.size,
                strength=self._brush.settings.strength,
                falloff=self._brush.settings.falloff,
                shape=self._brush.settings.shape,
                falloff_curve=self._brush.settings.falloff_curve,
            ))
        )

        radius = int(self._brush.settings.size / 2) + 1
        cx = int(center_x)
        cy = int(center_y)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = cx + dx
                y = cy + dy

                if not (0 <= x < self._terrain.width and 0 <= y < self._terrain.height):
                    continue

                influence = self._brush.get_influence(x, y, center_x, center_y)
                if influence <= 0:
                    continue

                # Store previous height
                prev_height = self._terrain.get_height(x, y)
                operation.previous_heights[(x, y)] = prev_height

                # Calculate new height based on mode
                new_height = self._apply_mode(x, y, prev_height, influence, delta_time)

                self._terrain.set_height(x, y, new_height)
                operation.affected_heights[(x, y)] = new_height

        self._operation_history.append(operation)
        self._redo_stack.clear()

        return operation

    def _apply_mode(
        self, x: int, y: int, current_height: float, influence: float, delta_time: float
    ) -> float:
        """Apply sculpt mode at a point."""
        mode = self._current_mode

        if mode == SculptMode.RAISE:
            return current_height + influence * delta_time

        elif mode == SculptMode.LOWER:
            return current_height - influence * delta_time

        elif mode == SculptMode.SMOOTH:
            avg = self._terrain.get_average_height(x, y, 2)
            return current_height + (avg - current_height) * influence

        elif mode == SculptMode.FLATTEN:
            target = self._terrain.get_height(
                int(self._brush.settings.size / 2),
                int(self._brush.settings.size / 2)
            )
            return current_height + (target - current_height) * influence

        elif mode == SculptMode.NOISE:
            random.seed(self._noise_seed + x * 1000 + y)
            noise_val = (random.random() * 2.0 - 1.0) * self._noise_scale
            return current_height + noise_val * influence * delta_time

        elif mode == SculptMode.LEVEL:
            return current_height + (self._level_height - current_height) * influence

        elif mode == SculptMode.STAMP:
            if self._stamp_data:
                stamp_h = len(self._stamp_data)
                stamp_w = len(self._stamp_data[0]) if stamp_h > 0 else 0
                # Map world position to stamp position
                sx = int((x - (int(self._brush.settings.size / 2))) % stamp_w)
                sy = int((y - (int(self._brush.settings.size / 2))) % stamp_h)
                if 0 <= sx < stamp_w and 0 <= sy < stamp_h:
                    stamp_val = self._stamp_data[sy][sx]
                    return current_height + stamp_val * influence * delta_time
            return current_height

        return current_height

    def undo(self) -> Optional[SculptOperation]:
        """
        Undo the last operation.

        Returns:
            The undone operation, or None if nothing to undo
        """
        if not self._operation_history:
            return None

        operation = self._operation_history.pop()

        # Restore previous heights
        for (x, y), height in operation.previous_heights.items():
            self._terrain.set_height(x, y, height)

        self._redo_stack.append(operation)
        return operation

    def redo(self) -> Optional[SculptOperation]:
        """
        Redo the last undone operation.

        Returns:
            The redone operation, or None if nothing to redo
        """
        if not self._redo_stack:
            return None

        operation = self._redo_stack.pop()

        # Apply stored heights
        for (x, y), height in operation.affected_heights.items():
            self._terrain.set_height(x, y, height)

        self._operation_history.append(operation)
        return operation

    def get_dirty_chunks(self) -> set[tuple[int, int]]:
        """
        Get chunks that need updating.

        Returns:
            Set of dirty chunk coordinates
        """
        return self._terrain.get_dirty_chunks()

    def clear_dirty_chunks(self) -> None:
        """Clear dirty chunk tracking."""
        self._terrain.clear_dirty_chunks()

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._operation_history) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def clear_history(self) -> None:
        """Clear operation history."""
        self._operation_history.clear()
        self._redo_stack.clear()
