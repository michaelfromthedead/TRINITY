"""
Erosion simulation tools for terrain in the AI Game Engine.

Provides hydraulic and thermal erosion algorithms for
realistic terrain weathering effects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any
import math
import random


class ErosionType(Enum):
    """Types of erosion simulation."""
    HYDRAULIC = auto()
    THERMAL = auto()
    COMBINED = auto()


@dataclass(slots=True)
class ErosionParams:
    """Base erosion parameters."""
    iterations: int = 50000
    seed: int = 42
    brush_radius: int = 3


@dataclass(slots=True)
class HydraulicErosionParams(ErosionParams):
    """Parameters for hydraulic erosion simulation."""
    inertia: float = 0.05
    sediment_capacity_factor: float = 4.0
    min_sediment_capacity: float = 0.01
    erosion_speed: float = 0.3
    deposition_speed: float = 0.3
    evaporation_speed: float = 0.01
    gravity: float = 4.0
    max_droplet_lifetime: int = 30
    initial_water_volume: float = 1.0
    initial_speed: float = 1.0


@dataclass(slots=True)
class ThermalErosionParams(ErosionParams):
    """Parameters for thermal erosion simulation."""
    talus_angle: float = 0.5  # Maximum stable slope (radians)
    erosion_rate: float = 0.5
    cell_size: float = 1.0


@dataclass(slots=True)
class ErosionBrush:
    """Pre-computed erosion brush weights."""
    radius: int
    weights: list[tuple[int, int, float]] = field(default_factory=list)
    weight_sum: float = 0.0


@dataclass(slots=True)
class WaterDroplet:
    """Simulates a water droplet for hydraulic erosion."""
    x: float
    y: float
    dir_x: float = 0.0
    dir_y: float = 0.0
    speed: float = 1.0
    water: float = 1.0
    sediment: float = 0.0


class ErosionSimulator:
    """
    Terrain erosion simulation engine.

    Implements hydraulic and thermal erosion algorithms for
    creating realistic terrain weathering effects.
    """
    __slots__ = (
        "_width",
        "_height",
        "_heights",
        "_erosion_brush",
        "_random",
    )

    def __init__(self, width: int, height: int, heights: list[list[float]]):
        """
        Initialize erosion simulator.

        Args:
            width: Terrain width
            height: Terrain height
            heights: Initial height data
        """
        self._width = width
        self._height = height
        self._heights = [row[:] for row in heights]  # Deep copy
        self._erosion_brush: Optional[ErosionBrush] = None
        self._random = random.Random()

    @property
    def heights(self) -> list[list[float]]:
        """Get current height data."""
        return self._heights

    @property
    def width(self) -> int:
        """Get terrain width."""
        return self._width

    @property
    def height(self) -> int:
        """Get terrain height."""
        return self._height

    def _create_erosion_brush(self, radius: int) -> ErosionBrush:
        """Create pre-computed erosion brush."""
        brush = ErosionBrush(radius=radius)

        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                dist_sq = x * x + y * y
                if dist_sq <= radius * radius:
                    weight = 1.0 - math.sqrt(dist_sq) / radius
                    brush.weights.append((x, y, weight))
                    brush.weight_sum += weight

        # Normalize weights
        if brush.weight_sum > 0:
            brush.weights = [
                (x, y, w / brush.weight_sum)
                for x, y, w in brush.weights
            ]

        return brush

    def _get_height(self, x: int, y: int) -> float:
        """Get height with bounds checking."""
        x = max(0, min(self._width - 1, x))
        y = max(0, min(self._height - 1, y))
        return self._heights[y][x]

    def _set_height(self, x: int, y: int, value: float) -> None:
        """Set height with bounds checking."""
        if 0 <= x < self._width and 0 <= y < self._height:
            self._heights[y][x] = max(0.0, value)

    def _interpolate_height(self, x: float, y: float) -> float:
        """Bilinear interpolation of height."""
        ix = int(x)
        iy = int(y)
        fx = x - ix
        fy = y - iy

        h00 = self._get_height(ix, iy)
        h10 = self._get_height(ix + 1, iy)
        h01 = self._get_height(ix, iy + 1)
        h11 = self._get_height(ix + 1, iy + 1)

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx

        return h0 * (1 - fy) + h1 * fy

    def _calculate_gradient(self, x: float, y: float) -> tuple[float, float, float]:
        """
        Calculate terrain gradient at a position.

        Returns:
            (gradient_x, gradient_y, height)
        """
        ix = int(x)
        iy = int(y)
        fx = x - ix
        fy = y - iy

        h00 = self._get_height(ix, iy)
        h10 = self._get_height(ix + 1, iy)
        h01 = self._get_height(ix, iy + 1)
        h11 = self._get_height(ix + 1, iy + 1)

        # Calculate gradient
        grad_x = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
        grad_y = (h01 - h00) * (1 - fx) + (h11 - h10) * fx

        # Interpolated height
        height = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + \
                 h01 * (1 - fx) * fy + h11 * fx * fy

        return grad_x, grad_y, height

    def simulate_hydraulic(
        self,
        params: Optional[HydraulicErosionParams] = None,
        progress_callback: Optional[callable] = None
    ) -> None:
        """
        Simulate hydraulic erosion.

        Water droplets flow down the terrain, eroding and depositing
        sediment based on their speed and carrying capacity.

        Args:
            params: Erosion parameters
            progress_callback: Optional callback for progress updates
        """
        if params is None:
            params = HydraulicErosionParams()

        self._random.seed(params.seed)
        self._erosion_brush = self._create_erosion_brush(params.brush_radius)

        for i in range(params.iterations):
            # Create droplet at random position
            droplet = WaterDroplet(
                x=self._random.random() * (self._width - 1),
                y=self._random.random() * (self._height - 1),
                speed=params.initial_speed,
                water=params.initial_water_volume,
            )

            self._simulate_droplet(droplet, params)

            if progress_callback and i % 1000 == 0:
                progress_callback(i / params.iterations)

        if progress_callback:
            progress_callback(1.0)

    def _simulate_droplet(
        self,
        droplet: WaterDroplet,
        params: HydraulicErosionParams
    ) -> None:
        """Simulate a single water droplet."""
        for _ in range(params.max_droplet_lifetime):
            ix = int(droplet.x)
            iy = int(droplet.y)

            # Calculate gradient
            grad_x, grad_y, old_height = self._calculate_gradient(droplet.x, droplet.y)

            # Update direction with inertia
            droplet.dir_x = droplet.dir_x * params.inertia - grad_x * (1 - params.inertia)
            droplet.dir_y = droplet.dir_y * params.inertia - grad_y * (1 - params.inertia)

            # Normalize direction
            length = math.sqrt(droplet.dir_x ** 2 + droplet.dir_y ** 2)
            if length > 0:
                droplet.dir_x /= length
                droplet.dir_y /= length
            else:
                # Random direction if flat
                angle = self._random.random() * 2 * math.pi
                droplet.dir_x = math.cos(angle)
                droplet.dir_y = math.sin(angle)

            # Move droplet
            new_x = droplet.x + droplet.dir_x
            new_y = droplet.y + droplet.dir_y

            # Check bounds
            if new_x < 0 or new_x >= self._width - 1 or \
               new_y < 0 or new_y >= self._height - 1:
                break

            new_height = self._interpolate_height(new_x, new_y)
            height_diff = new_height - old_height

            # Calculate sediment capacity
            capacity = max(
                -height_diff * droplet.speed * droplet.water * params.sediment_capacity_factor,
                params.min_sediment_capacity
            )

            # Deposit or erode
            if droplet.sediment > capacity or height_diff > 0:
                # Deposit sediment
                amount = (droplet.sediment - capacity) * params.deposition_speed \
                         if height_diff <= 0 else \
                         min(height_diff, droplet.sediment)

                droplet.sediment -= amount
                self._deposit_sediment(ix, iy, amount)
            else:
                # Erode terrain
                amount = min(
                    (capacity - droplet.sediment) * params.erosion_speed,
                    -height_diff
                )

                droplet.sediment += amount
                self._erode_terrain(ix, iy, amount)

            # Update droplet state
            droplet.speed = math.sqrt(max(0, droplet.speed ** 2 + height_diff * params.gravity))
            droplet.water *= (1 - params.evaporation_speed)

            droplet.x = new_x
            droplet.y = new_y

            if droplet.water < 0.001:
                break

    def _erode_terrain(self, cx: int, cy: int, amount: float) -> None:
        """Erode terrain using erosion brush."""
        if self._erosion_brush is None:
            return

        for dx, dy, weight in self._erosion_brush.weights:
            x = cx + dx
            y = cy + dy
            if 0 <= x < self._width and 0 <= y < self._height:
                erode = amount * weight
                self._heights[y][x] = max(0, self._heights[y][x] - erode)

    def _deposit_sediment(self, cx: int, cy: int, amount: float) -> None:
        """Deposit sediment using brush."""
        if self._erosion_brush is None:
            return

        for dx, dy, weight in self._erosion_brush.weights:
            x = cx + dx
            y = cy + dy
            if 0 <= x < self._width and 0 <= y < self._height:
                deposit = amount * weight
                self._heights[y][x] += deposit

    def simulate_thermal(
        self,
        params: Optional[ThermalErosionParams] = None,
        progress_callback: Optional[callable] = None
    ) -> None:
        """
        Simulate thermal erosion.

        Material falls from steep slopes to neighboring lower cells
        until the terrain reaches a stable angle of repose.

        Args:
            params: Erosion parameters
            progress_callback: Optional callback for progress updates
        """
        if params is None:
            params = ThermalErosionParams()

        self._random.seed(params.seed)

        # Pre-calculate talus threshold
        max_height_diff = math.tan(params.talus_angle) * params.cell_size

        for iteration in range(params.iterations):
            # Process each cell
            for y in range(1, self._height - 1):
                for x in range(1, self._width - 1):
                    self._thermal_erode_cell(x, y, max_height_diff, params.erosion_rate)

            if progress_callback and iteration % 100 == 0:
                progress_callback(iteration / params.iterations)

        if progress_callback:
            progress_callback(1.0)

    def _thermal_erode_cell(
        self,
        x: int,
        y: int,
        max_diff: float,
        erosion_rate: float
    ) -> None:
        """Apply thermal erosion to a single cell."""
        center_height = self._heights[y][x]

        # Check all 8 neighbors
        neighbors = [
            (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
            (x - 1, y),                  (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1),
        ]

        max_slope = 0.0
        max_slope_neighbor = None

        for nx, ny in neighbors:
            if 0 <= nx < self._width and 0 <= ny < self._height:
                neighbor_height = self._heights[ny][nx]
                height_diff = center_height - neighbor_height

                if height_diff > max_slope:
                    max_slope = height_diff
                    max_slope_neighbor = (nx, ny)

        # Transfer material if slope exceeds threshold
        if max_slope > max_diff and max_slope_neighbor is not None:
            transfer = (max_slope - max_diff) * 0.5 * erosion_rate

            self._heights[y][x] -= transfer
            nx, ny = max_slope_neighbor
            self._heights[ny][nx] += transfer

    def simulate_combined(
        self,
        hydraulic_params: Optional[HydraulicErosionParams] = None,
        thermal_params: Optional[ThermalErosionParams] = None,
        hydraulic_weight: float = 0.7,
        progress_callback: Optional[callable] = None
    ) -> None:
        """
        Simulate combined hydraulic and thermal erosion.

        Args:
            hydraulic_params: Hydraulic erosion parameters
            thermal_params: Thermal erosion parameters
            hydraulic_weight: Weight for hydraulic erosion (0-1)
            progress_callback: Optional progress callback
        """
        # Run hydraulic erosion
        if hydraulic_params is None:
            hydraulic_params = HydraulicErosionParams()
        hydraulic_params.iterations = int(hydraulic_params.iterations * hydraulic_weight)

        def hydraulic_progress(p: float) -> None:
            if progress_callback:
                progress_callback(p * hydraulic_weight)

        self.simulate_hydraulic(hydraulic_params, hydraulic_progress)

        # Run thermal erosion
        if thermal_params is None:
            thermal_params = ThermalErosionParams()
        thermal_params.iterations = int(thermal_params.iterations * (1 - hydraulic_weight))

        def thermal_progress(p: float) -> None:
            if progress_callback:
                progress_callback(hydraulic_weight + p * (1 - hydraulic_weight))

        self.simulate_thermal(thermal_params, thermal_progress)

    def get_erosion_map(self, original_heights: list[list[float]]) -> list[list[float]]:
        """
        Calculate erosion difference map.

        Args:
            original_heights: Original height data before erosion

        Returns:
            2D array of erosion amounts (positive = erosion, negative = deposition)
        """
        return [
            [original_heights[y][x] - self._heights[y][x]
             for x in range(self._width)]
            for y in range(self._height)
        ]

    def reset(self, heights: list[list[float]]) -> None:
        """
        Reset heights to new data.

        Args:
            heights: New height data
        """
        self._heights = [row[:] for row in heights]
