"""
Terrain sculpting tools for modifying heightfield data.

Provides brush-based sculpting operations including raise, lower, smooth,
flatten, erosion, noise, and ramp tools with full undo/redo support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Protocol, Tuple
from typing import runtime_checkable

from engine.world.terrain.constants import (
    DEFAULT_BRUSH_RADIUS,
    DEFAULT_BRUSH_STRENGTH,
    DEFAULT_BRUSH_FALLOFF,
)

if TYPE_CHECKING:
    from typing import TypeAlias


class SculptTool(Enum):
    """Available terrain sculpting tools."""

    RAISE = auto()
    LOWER = auto()
    SMOOTH = auto()
    FLATTEN = auto()
    EROSION = auto()
    NOISE = auto()
    RAMP = auto()


class BrushShape(Enum):
    """Shape of the sculpting brush."""

    CIRCLE = auto()
    SQUARE = auto()


@dataclass
class BrushSettings:
    """Configuration for a terrain sculpting brush.

    Attributes:
        size: Radius of the brush in world units.
        strength: Intensity of the brush effect (0-1).
        falloff: Edge softness factor (0-1), where 0 is hard edge.
        shape: Shape of the brush (circle or square).
    """

    size: float = 10.0
    strength: float = 0.5
    falloff: float = 0.5
    shape: BrushShape = BrushShape.CIRCLE

    def __post_init__(self) -> None:
        """Validate brush settings."""
        if self.size <= 0:
            raise ValueError("size must be > 0")
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be in range [0, 1]")
        if not 0 <= self.falloff <= 1:
            raise ValueError("falloff must be in range [0, 1]")


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

    def set_height_at(self, x: int, z: int, height: float) -> None:
        """Set height at sample coordinates."""
        ...

    def world_to_sample(self, world_x: float, world_z: float) -> Tuple[int, int]:
        """Convert world coordinates to sample coordinates."""
        ...

    def sample_to_world(self, sample_x: int, sample_z: int) -> Tuple[float, float]:
        """Convert sample coordinates to world coordinates."""
        ...


class TerrainBrush:
    """Brush for terrain sculpting operations.

    Calculates falloff values and determines which samples are affected
    by a brush stroke.
    """

    def __init__(self, settings: BrushSettings) -> None:
        """Initialize the brush with settings.

        Args:
            settings: Configuration for the brush.
        """
        self._settings = settings

    @property
    def settings(self) -> BrushSettings:
        """Get brush settings."""
        return self._settings

    def get_falloff_at(self, distance_from_center: float) -> float:
        """Calculate falloff value at a given distance from brush center.

        Args:
            distance_from_center: Distance from the center of the brush.

        Returns:
            Falloff value between 0 and 1, where 1 is full strength.
        """
        if distance_from_center < 0:
            raise ValueError("distance_from_center must be >= 0")

        radius = self._settings.size
        if distance_from_center >= radius:
            return 0.0

        # Normalize distance to [0, 1]
        normalized_dist = distance_from_center / radius

        # Apply falloff curve
        # falloff=0 means hard edge (step function)
        # falloff=1 means maximum softness (linear gradient)
        falloff = self._settings.falloff

        if falloff == 0:
            # Hard edge
            return 1.0
        else:
            # Smooth falloff using cosine interpolation
            # The falloff parameter controls where the transition begins
            inner_radius_ratio = 1.0 - falloff
            if normalized_dist <= inner_radius_ratio:
                return 1.0
            else:
                # Smooth transition in outer region
                t = (normalized_dist - inner_radius_ratio) / falloff
                return 0.5 * (1.0 + math.cos(math.pi * t))

    def get_affected_samples(
        self, heightfield: Heightfield, center_x: float, center_z: float
    ) -> List[Tuple[int, int]]:
        """Get list of sample coordinates affected by the brush.

        Args:
            heightfield: The heightfield to operate on.
            center_x: World X coordinate of brush center.
            center_z: World Z coordinate of brush center.

        Returns:
            List of (x, z) sample coordinates within brush radius.
        """
        samples = []
        radius = self._settings.size
        spacing = heightfield.sample_spacing

        # Calculate sample bounds
        center_sx, center_sz = heightfield.world_to_sample(center_x, center_z)
        sample_radius = int(math.ceil(radius / spacing))

        min_x = max(0, center_sx - sample_radius)
        max_x = min(heightfield.width - 1, center_sx + sample_radius)
        min_z = max(0, center_sz - sample_radius)
        max_z = min(heightfield.height - 1, center_sz + sample_radius)

        for sz in range(min_z, max_z + 1):
            for sx in range(min_x, max_x + 1):
                world_x, world_z = heightfield.sample_to_world(sx, sz)
                dx = world_x - center_x
                dz = world_z - center_z

                if self._settings.shape == BrushShape.CIRCLE:
                    distance = math.sqrt(dx * dx + dz * dz)
                    if distance <= radius:
                        samples.append((sx, sz))
                else:  # SQUARE
                    if abs(dx) <= radius and abs(dz) <= radius:
                        samples.append((sx, sz))

        return samples


@dataclass
class HeightDelta:
    """Records height changes for undo/redo operations.

    Attributes:
        changes: Dictionary mapping (x, z) sample coords to (old_height, new_height).
    """

    changes: dict = field(default_factory=dict)

    def add_change(
        self, x: int, z: int, old_height: float, new_height: float
    ) -> None:
        """Record a height change.

        Args:
            x: Sample X coordinate.
            z: Sample Z coordinate.
            old_height: Height before change.
            new_height: Height after change.
        """
        key = (x, z)
        if key in self.changes:
            # Keep original old height, update new height
            self.changes[key] = (self.changes[key][0], new_height)
        else:
            self.changes[key] = (old_height, new_height)

    def is_empty(self) -> bool:
        """Check if delta contains any changes."""
        return len(self.changes) == 0


class BaseSculptTool:
    """Base class for sculpting tools."""

    def __init__(self, brush: TerrainBrush) -> None:
        """Initialize the tool with a brush.

        Args:
            brush: The brush to use for sculpting.
        """
        self._brush = brush

    @property
    def brush(self) -> TerrainBrush:
        """Get the brush."""
        return self._brush

    def apply(
        self,
        heightfield: Heightfield,
        center_x: float,
        center_z: float,
        delta: HeightDelta,
    ) -> None:
        """Apply the tool to the heightfield.

        Args:
            heightfield: The heightfield to modify.
            center_x: World X coordinate of brush center.
            center_z: World Z coordinate of brush center.
            delta: Delta object to record changes for undo.
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def _get_distance_to_center(
        self, heightfield: Heightfield, sx: int, sz: int, center_x: float, center_z: float
    ) -> float:
        """Calculate world distance from sample to brush center.

        Args:
            heightfield: The heightfield.
            sx: Sample X coordinate.
            sz: Sample Z coordinate.
            center_x: World X coordinate of brush center.
            center_z: World Z coordinate of brush center.

        Returns:
            Distance in world units.
        """
        world_x, world_z = heightfield.sample_to_world(sx, sz)
        dx = world_x - center_x
        dz = world_z - center_z

        if self._brush.settings.shape == BrushShape.CIRCLE:
            return math.sqrt(dx * dx + dz * dz)
        else:  # SQUARE - use Chebyshev distance
            return max(abs(dx), abs(dz))


class RaiseTool(BaseSculptTool):
    """Tool that increases terrain height."""

    def apply(
        self,
        heightfield: Heightfield,
        center_x: float,
        center_z: float,
        delta: HeightDelta,
    ) -> None:
        """Raise terrain height within brush area.

        Args:
            heightfield: The heightfield to modify.
            center_x: World X coordinate of brush center.
            center_z: World Z coordinate of brush center.
            delta: Delta object to record changes.
        """
        samples = self._brush.get_affected_samples(heightfield, center_x, center_z)
        strength = self._brush.settings.strength

        for sx, sz in samples:
            distance = self._get_distance_to_center(heightfield, sx, sz, center_x, center_z)
            falloff = self._brush.get_falloff_at(distance)

            old_height = heightfield.get_height_at(sx, sz)
            new_height = old_height + strength * falloff
            heightfield.set_height_at(sx, sz, new_height)
            delta.add_change(sx, sz, old_height, new_height)


class LowerTool(BaseSculptTool):
    """Tool that decreases terrain height."""

    def apply(
        self,
        heightfield: Heightfield,
        center_x: float,
        center_z: float,
        delta: HeightDelta,
    ) -> None:
        """Lower terrain height within brush area.

        Args:
            heightfield: The heightfield to modify.
            center_x: World X coordinate of brush center.
            center_z: World Z coordinate of brush center.
            delta: Delta object to record changes.
        """
        samples = self._brush.get_affected_samples(heightfield, center_x, center_z)
        strength = self._brush.settings.strength

        for sx, sz in samples:
            distance = self._get_distance_to_center(heightfield, sx, sz, center_x, center_z)
            falloff = self._brush.get_falloff_at(distance)

            old_height = heightfield.get_height_at(sx, sz)
            new_height = old_height - strength * falloff
            heightfield.set_height_at(sx, sz, new_height)
            delta.add_change(sx, sz, old_height, new_height)


class SmoothTool(BaseSculptTool):
    """Tool that smooths terrain by averaging with neighbors (Gaussian blur)."""

    def __init__(self, brush: TerrainBrush, kernel_size: int = 3) -> None:
        """Initialize smooth tool.

        Args:
            brush: The brush to use.
            kernel_size: Size of the smoothing kernel (must be odd).
        """
        super().__init__(brush)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        self._kernel_size = kernel_size

    def apply(
        self,
        heightfield: Heightfield,
        center_x: float,
        center_z: float,
        delta: HeightDelta,
    ) -> None:
        """Smooth terrain within brush area.

        Args:
            heightfield: The heightfield to modify.
            center_x: World X coordinate of brush center.
            center_z: World Z coordinate of brush center.
            delta: Delta object to record changes.
        """
        samples = self._brush.get_affected_samples(heightfield, center_x, center_z)
        strength = self._brush.settings.strength
        half_kernel = self._kernel_size // 2

        # Calculate new heights first (don't modify during calculation)
        new_heights = {}

        for sx, sz in samples:
            distance = self._get_distance_to_center(heightfield, sx, sz, center_x, center_z)
            falloff = self._brush.get_falloff_at(distance)

            # Gaussian-weighted average of neighbors
            total_weight = 0.0
            weighted_sum = 0.0

            for dz in range(-half_kernel, half_kernel + 1):
                for dx in range(-half_kernel, half_kernel + 1):
                    nx, nz = sx + dx, sz + dz
                    if 0 <= nx < heightfield.width and 0 <= nz < heightfield.height:
                        # Gaussian weight based on distance from center of kernel
                        dist_sq = dx * dx + dz * dz
                        weight = math.exp(-dist_sq / (2.0 * (half_kernel * 0.5) ** 2))
                        weighted_sum += heightfield.get_height_at(nx, nz) * weight
                        total_weight += weight

            if total_weight > 0:
                smoothed_height = weighted_sum / total_weight
                old_height = heightfield.get_height_at(sx, sz)
                # Blend between original and smoothed based on strength and falloff
                new_height = old_height + (smoothed_height - old_height) * strength * falloff
                new_heights[(sx, sz)] = (old_height, new_height)

        # Apply changes
        for (sx, sz), (old_height, new_height) in new_heights.items():
            heightfield.set_height_at(sx, sz, new_height)
            delta.add_change(sx, sz, old_height, new_height)


class FlattenTool(BaseSculptTool):
    """Tool that moves terrain toward a target height."""

    def __init__(self, brush: TerrainBrush, target_height: Optional[float] = None) -> None:
        """Initialize flatten tool.

        Args:
            brush: The brush to use.
            target_height: Target height to flatten to. If None, uses height at
                brush center when applied.
        """
        super().__init__(brush)
        self._target_height = target_height

    @property
    def target_height(self) -> Optional[float]:
        """Get target height."""
        return self._target_height

    @target_height.setter
    def target_height(self, value: Optional[float]) -> None:
        """Set target height."""
        self._target_height = value

    def apply(
        self,
        heightfield: Heightfield,
        center_x: float,
        center_z: float,
        delta: HeightDelta,
    ) -> None:
        """Flatten terrain toward target height.

        Args:
            heightfield: The heightfield to modify.
            center_x: World X coordinate of brush center.
            center_z: World Z coordinate of brush center.
            delta: Delta object to record changes.
        """
        samples = self._brush.get_affected_samples(heightfield, center_x, center_z)
        strength = self._brush.settings.strength

        # Determine target height
        if self._target_height is not None:
            target = self._target_height
        else:
            # Use height at brush center
            cx, cz = heightfield.world_to_sample(center_x, center_z)
            cx = max(0, min(heightfield.width - 1, cx))
            cz = max(0, min(heightfield.height - 1, cz))
            target = heightfield.get_height_at(cx, cz)

        for sx, sz in samples:
            distance = self._get_distance_to_center(heightfield, sx, sz, center_x, center_z)
            falloff = self._brush.get_falloff_at(distance)

            old_height = heightfield.get_height_at(sx, sz)
            # Move toward target height
            new_height = old_height + (target - old_height) * strength * falloff
            heightfield.set_height_at(sx, sz, new_height)
            delta.add_change(sx, sz, old_height, new_height)


class ErosionTool(BaseSculptTool):
    """Tool that simulates hydraulic erosion (water flow)."""

    def __init__(
        self,
        brush: TerrainBrush,
        iterations: int = 10,
        sediment_capacity: float = 0.1,
        deposition_rate: float = 0.3,
        erosion_rate: float = 0.3,
    ) -> None:
        """Initialize erosion tool.

        Args:
            brush: The brush to use.
            iterations: Number of erosion simulation iterations.
            sediment_capacity: Maximum sediment a water droplet can carry.
            deposition_rate: Rate at which sediment is deposited.
            erosion_rate: Rate at which terrain is eroded.
        """
        super().__init__(brush)
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if sediment_capacity <= 0:
            raise ValueError("sediment_capacity must be > 0")
        if not 0 < deposition_rate <= 1:
            raise ValueError("deposition_rate must be in (0, 1]")
        if not 0 < erosion_rate <= 1:
            raise ValueError("erosion_rate must be in (0, 1]")

        self._iterations = iterations
        self._sediment_capacity = sediment_capacity
        self._deposition_rate = deposition_rate
        self._erosion_rate = erosion_rate

    def apply(
        self,
        heightfield: Heightfield,
        center_x: float,
        center_z: float,
        delta: HeightDelta,
    ) -> None:
        """Apply erosion simulation within brush area.

        Args:
            heightfield: The heightfield to modify.
            center_x: World X coordinate of brush center.
            center_z: World Z coordinate of brush center.
            delta: Delta object to record changes.
        """
        samples = self._brush.get_affected_samples(heightfield, center_x, center_z)
        strength = self._brush.settings.strength

        # Store original heights
        original_heights = {
            (sx, sz): heightfield.get_height_at(sx, sz) for sx, sz in samples
        }

        # Create working copy of heights in affected area
        working_heights = dict(original_heights)

        # Simulate water droplets
        for _ in range(self._iterations):
            for sx, sz in samples:
                distance = self._get_distance_to_center(
                    heightfield, sx, sz, center_x, center_z
                )
                falloff = self._brush.get_falloff_at(distance)
                if falloff == 0:
                    continue

                # Find lowest neighbor
                current_height = working_heights.get(
                    (sx, sz), heightfield.get_height_at(sx, sz)
                )
                lowest_neighbor = None
                lowest_height = current_height

                for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, nz = sx + dx, sz + dz
                    if 0 <= nx < heightfield.width and 0 <= nz < heightfield.height:
                        neighbor_height = working_heights.get(
                            (nx, nz), heightfield.get_height_at(nx, nz)
                        )
                        if neighbor_height < lowest_height:
                            lowest_height = neighbor_height
                            lowest_neighbor = (nx, nz)

                # Erode if there's a lower neighbor
                if lowest_neighbor is not None:
                    height_diff = current_height - lowest_height
                    erosion_amount = (
                        min(height_diff, self._sediment_capacity)
                        * self._erosion_rate
                        * strength
                        * falloff
                    )

                    working_heights[(sx, sz)] = current_height - erosion_amount
                    # Deposit some sediment at lower neighbor
                    deposit_amount = erosion_amount * self._deposition_rate
                    working_heights[lowest_neighbor] = (
                        working_heights.get(
                            lowest_neighbor, heightfield.get_height_at(*lowest_neighbor)
                        )
                        + deposit_amount
                    )

        # Apply changes
        for (sx, sz), new_height in working_heights.items():
            old_height = original_heights.get((sx, sz), heightfield.get_height_at(sx, sz))
            if abs(new_height - old_height) > 1e-6:
                heightfield.set_height_at(sx, sz, new_height)
                delta.add_change(sx, sz, old_height, new_height)


class NoiseTool(BaseSculptTool):
    """Tool that adds Perlin-like noise to terrain."""

    def __init__(
        self,
        brush: TerrainBrush,
        noise_scale: float = 0.1,
        octaves: int = 4,
        persistence: float = 0.5,
        seed: int = 42,
    ) -> None:
        """Initialize noise tool.

        Args:
            brush: The brush to use.
            noise_scale: Scale of the noise pattern.
            octaves: Number of noise octaves for fractal noise.
            persistence: Amplitude decay per octave.
            seed: Random seed for reproducible noise.
        """
        super().__init__(brush)
        if noise_scale <= 0:
            raise ValueError("noise_scale must be > 0")
        if octaves < 1:
            raise ValueError("octaves must be >= 1")
        if not 0 < persistence <= 1:
            raise ValueError("persistence must be in (0, 1]")

        self._noise_scale = noise_scale
        self._octaves = octaves
        self._persistence = persistence
        self._seed = seed
        self._permutation = self._generate_permutation(seed)

    def _generate_permutation(self, seed: int) -> List[int]:
        """Generate permutation table for noise."""
        import random

        rng = random.Random(seed)
        p = list(range(256))
        rng.shuffle(p)
        return p + p  # Duplicate for overflow

    def _fade(self, t: float) -> float:
        """Fade function for smooth interpolation."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + t * (b - a)

    def _grad(self, hash_val: int, x: float, y: float) -> float:
        """Calculate gradient."""
        h = hash_val & 3
        if h == 0:
            return x + y
        elif h == 1:
            return -x + y
        elif h == 2:
            return x - y
        else:
            return -x - y

    def _perlin(self, x: float, y: float) -> float:
        """Calculate Perlin noise value at (x, y)."""
        p = self._permutation

        # Find unit square containing point
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255

        # Relative position in square
        xf = x - math.floor(x)
        yf = y - math.floor(y)

        # Fade curves
        u = self._fade(xf)
        v = self._fade(yf)

        # Hash corners
        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]

        # Blend
        x1 = self._lerp(self._grad(aa, xf, yf), self._grad(ba, xf - 1, yf), u)
        x2 = self._lerp(self._grad(ab, xf, yf - 1), self._grad(bb, xf - 1, yf - 1), u)

        return self._lerp(x1, x2, v)

    def _fractal_noise(self, x: float, y: float) -> float:
        """Calculate fractal Brownian motion noise."""
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(self._octaves):
            total += self._perlin(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= self._persistence
            frequency *= 2

        return total / max_value

    def apply(
        self,
        heightfield: Heightfield,
        center_x: float,
        center_z: float,
        delta: HeightDelta,
    ) -> None:
        """Add noise to terrain within brush area.

        Args:
            heightfield: The heightfield to modify.
            center_x: World X coordinate of brush center.
            center_z: World Z coordinate of brush center.
            delta: Delta object to record changes.
        """
        samples = self._brush.get_affected_samples(heightfield, center_x, center_z)
        strength = self._brush.settings.strength

        for sx, sz in samples:
            distance = self._get_distance_to_center(heightfield, sx, sz, center_x, center_z)
            falloff = self._brush.get_falloff_at(distance)

            world_x, world_z = heightfield.sample_to_world(sx, sz)
            noise_val = self._fractal_noise(
                world_x * self._noise_scale, world_z * self._noise_scale
            )

            old_height = heightfield.get_height_at(sx, sz)
            new_height = old_height + noise_val * strength * falloff
            heightfield.set_height_at(sx, sz, new_height)
            delta.add_change(sx, sz, old_height, new_height)


class RampTool(BaseSculptTool):
    """Tool that creates a gradient/ramp between two points."""

    def __init__(
        self,
        brush: TerrainBrush,
        start_point: Optional[Tuple[float, float]] = None,
        end_point: Optional[Tuple[float, float]] = None,
        start_height: Optional[float] = None,
        end_height: Optional[float] = None,
    ) -> None:
        """Initialize ramp tool.

        Args:
            brush: The brush to use.
            start_point: World (x, z) coordinates for ramp start.
            end_point: World (x, z) coordinates for ramp end.
            start_height: Height at ramp start (uses terrain height if None).
            end_height: Height at ramp end (uses terrain height if None).
        """
        super().__init__(brush)
        self._start_point = start_point
        self._end_point = end_point
        self._start_height = start_height
        self._end_height = end_height

    def set_ramp_points(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        start_height: Optional[float] = None,
        end_height: Optional[float] = None,
    ) -> None:
        """Set the ramp start and end points.

        Args:
            start_point: World (x, z) coordinates for ramp start.
            end_point: World (x, z) coordinates for ramp end.
            start_height: Height at ramp start (uses terrain height if None).
            end_height: Height at ramp end (uses terrain height if None).
        """
        self._start_point = start_point
        self._end_point = end_point
        self._start_height = start_height
        self._end_height = end_height

    def apply(
        self,
        heightfield: Heightfield,
        center_x: float,
        center_z: float,
        delta: HeightDelta,
    ) -> None:
        """Create ramp in terrain (center is ignored, uses ramp points).

        Args:
            heightfield: The heightfield to modify.
            center_x: Ignored, ramp uses configured start/end points.
            center_z: Ignored, ramp uses configured start/end points.
            delta: Delta object to record changes.
        """
        if self._start_point is None or self._end_point is None:
            raise ValueError("Ramp start and end points must be set")

        start_x, start_z = self._start_point
        end_x, end_z = self._end_point

        # Get heights
        if self._start_height is not None:
            start_h = self._start_height
        else:
            sx, sz = heightfield.world_to_sample(start_x, start_z)
            sx = max(0, min(heightfield.width - 1, sx))
            sz = max(0, min(heightfield.height - 1, sz))
            start_h = heightfield.get_height_at(sx, sz)

        if self._end_height is not None:
            end_h = self._end_height
        else:
            ex, ez = heightfield.world_to_sample(end_x, end_z)
            ex = max(0, min(heightfield.width - 1, ex))
            ez = max(0, min(heightfield.height - 1, ez))
            end_h = heightfield.get_height_at(ex, ez)

        # Calculate ramp direction and length
        dx = end_x - start_x
        dz = end_z - start_z
        ramp_length = math.sqrt(dx * dx + dz * dz)

        if ramp_length < 1e-6:
            return  # Points are too close

        # Normalize direction
        dir_x = dx / ramp_length
        dir_z = dz / ramp_length

        # Get samples along the ramp corridor
        samples = self._brush.get_affected_samples(
            heightfield, (start_x + end_x) / 2, (start_z + end_z) / 2
        )
        # Also include samples at each point along the ramp
        all_samples = set(samples)

        strength = self._brush.settings.strength
        width = self._brush.settings.size

        for sx, sz in list(all_samples):
            world_x, world_z = heightfield.sample_to_world(sx, sz)

            # Project point onto ramp line
            to_point_x = world_x - start_x
            to_point_z = world_z - start_z
            projection = to_point_x * dir_x + to_point_z * dir_z

            # Clamp to ramp length
            t = max(0, min(1, projection / ramp_length))

            # Calculate perpendicular distance from ramp line
            closest_x = start_x + dir_x * projection
            closest_z = start_z + dir_z * projection
            perp_dist = math.sqrt(
                (world_x - closest_x) ** 2 + (world_z - closest_z) ** 2
            )

            if perp_dist > width:
                continue

            # Calculate target height along ramp
            target_height = start_h + (end_h - start_h) * t

            # Calculate falloff based on perpendicular distance
            falloff = self._brush.get_falloff_at(perp_dist)

            old_height = heightfield.get_height_at(sx, sz)
            new_height = old_height + (target_height - old_height) * strength * falloff
            heightfield.set_height_at(sx, sz, new_height)
            delta.add_change(sx, sz, old_height, new_height)


class SculptingSession:
    """Manages a terrain sculpting session with undo/redo support.

    Maintains history of changes and provides undo/redo functionality.
    """

    def __init__(
        self,
        heightfield: Heightfield,
        max_undo_levels: int = 50,
    ) -> None:
        """Initialize sculpting session.

        Args:
            heightfield: The heightfield to sculpt.
            max_undo_levels: Maximum number of undo levels to keep.
        """
        self._heightfield = heightfield
        self._max_undo_levels = max_undo_levels
        self._undo_stack: List[HeightDelta] = []
        self._redo_stack: List[HeightDelta] = []

    @property
    def heightfield(self) -> Heightfield:
        """Get the heightfield."""
        return self._heightfield

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    @property
    def undo_count(self) -> int:
        """Get number of available undo levels."""
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        """Get number of available redo levels."""
        return len(self._redo_stack)

    def apply_tool(
        self,
        tool: BaseSculptTool,
        center_x: float,
        center_z: float,
    ) -> None:
        """Apply a sculpting tool at a position.

        Args:
            tool: The sculpting tool to apply.
            center_x: World X coordinate of brush center.
            center_z: World Z coordinate of brush center.
        """
        delta = HeightDelta()
        tool.apply(self._heightfield, center_x, center_z, delta)

        if not delta.is_empty():
            # Clear redo stack when new changes are made
            self._redo_stack.clear()

            # Add to undo stack
            self._undo_stack.append(delta)

            # Limit undo stack size
            while len(self._undo_stack) > self._max_undo_levels:
                self._undo_stack.pop(0)

    def undo(self) -> bool:
        """Undo the last sculpting operation.

        Returns:
            True if undo was performed, False if nothing to undo.
        """
        if not self._undo_stack:
            return False

        delta = self._undo_stack.pop()

        # Reverse the changes
        reverse_delta = HeightDelta()
        for (x, z), (old_height, new_height) in delta.changes.items():
            current = self._heightfield.get_height_at(x, z)
            self._heightfield.set_height_at(x, z, old_height)
            reverse_delta.add_change(x, z, current, old_height)

        # Add to redo stack (store original delta for redo)
        self._redo_stack.append(delta)

        return True

    def redo(self) -> bool:
        """Redo the last undone operation.

        Returns:
            True if redo was performed, False if nothing to redo.
        """
        if not self._redo_stack:
            return False

        delta = self._redo_stack.pop()

        # Reapply the changes
        for (x, z), (old_height, new_height) in delta.changes.items():
            self._heightfield.set_height_at(x, z, new_height)

        # Add back to undo stack
        self._undo_stack.append(delta)

        return True

    def clear_history(self) -> None:
        """Clear all undo/redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()


def create_tool(
    tool_type: SculptTool,
    brush: "TerrainBrush",
    **kwargs: Any,
) -> "BaseSculptTool":
    """Factory function to create sculpting tools.

    Args:
        tool_type: Type of tool to create.
        brush: Brush to use with the tool.
        **kwargs: Additional arguments for specific tools.

    Returns:
        The created sculpting tool.

    Raises:
        ValueError: If tool_type is not recognized.
    """
    tool_classes = {
        SculptTool.RAISE: RaiseTool,
        SculptTool.LOWER: LowerTool,
        SculptTool.SMOOTH: SmoothTool,
        SculptTool.FLATTEN: FlattenTool,
        SculptTool.EROSION: ErosionTool,
        SculptTool.NOISE: NoiseTool,
        SculptTool.RAMP: RampTool,
    }

    if tool_type not in tool_classes:
        raise ValueError(f"Unknown tool type: {tool_type}")

    tool_class = tool_classes[tool_type]
    return tool_class(brush, **kwargs)
