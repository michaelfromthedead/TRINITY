"""
Foliage placement system for the game engine World Layer.

Provides procedural and manual foliage placement with:
- Slope and height filtering
- Terrain layer masking
- Noise-based distribution
- Deterministic seeded generation
- Manual placement editing

Uses @procedural_placement decorator for scatter rules.
"""

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Type
from typing import runtime_checkable

from engine.world.foliage.constants import (
    DEFAULT_DENSITY,
    DEFAULT_MIN_SPACING,
    DEFAULT_SCALE_RANGE,
)


@dataclass
class PlacementRule:
    """
    Rule for filtering placement locations.

    Defines constraints that positions must satisfy to be valid
    placement locations for foliage instances.

    Attributes:
        slope_range: Min/max slope in degrees (0-90)
        height_range: Optional min/max height constraint
        terrain_layers: Allowed terrain layer indices (empty = all)
        noise_threshold: Minimum noise value for placement (0-1)
        noise_scale: Scale factor for noise sampling
        exclude_water: Whether to exclude water areas
        exclude_roads: Whether to exclude road areas
    """

    slope_range: Tuple[float, float] = (0.0, 90.0)
    height_range: Optional[Tuple[float, float]] = None
    terrain_layers: List[int] = field(default_factory=list)
    noise_threshold: float = 0.0
    noise_scale: float = 10.0
    exclude_water: bool = True
    exclude_roads: bool = True

    def __post_init__(self) -> None:
        """Validate placement rule parameters."""
        if self.slope_range[0] < 0 or self.slope_range[1] > 90:
            raise ValueError("slope_range must be between 0 and 90 degrees")
        if self.slope_range[0] > self.slope_range[1]:
            raise ValueError("slope_range min must be <= max")
        if self.height_range is not None:
            if self.height_range[0] > self.height_range[1]:
                raise ValueError("height_range min must be <= max")
        if not 0.0 <= self.noise_threshold <= 1.0:
            raise ValueError("noise_threshold must be between 0 and 1")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be > 0")


@dataclass
class PlacementResult:
    """
    Single placement instance result.

    Represents a computed placement location with position,
    rotation, and scale values.

    Attributes:
        position: World position (x, y, z)
        rotation: Euler rotation angles in degrees (pitch, yaw, roll)
        scale: Scale factors (x, y, z)
        foliage_type_id: Associated foliage type identifier
    """

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    foliage_type_id: str = ""

    def get_transform_matrix(self) -> List[List[float]]:
        """
        Get 4x4 transformation matrix for this placement.

        Returns:
            4x4 transformation matrix as nested lists
        """
        # Convert rotation to radians
        pitch = math.radians(self.rotation[0])
        yaw = math.radians(self.rotation[1])
        roll = math.radians(self.rotation[2])

        # Compute rotation matrices
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cos_r, sin_r = math.cos(roll), math.sin(roll)

        # Combined rotation matrix (Y * X * Z order)
        r00 = cos_y * cos_r + sin_y * sin_p * sin_r
        r01 = -cos_y * sin_r + sin_y * sin_p * cos_r
        r02 = sin_y * cos_p
        r10 = cos_p * sin_r
        r11 = cos_p * cos_r
        r12 = -sin_p
        r20 = -sin_y * cos_r + cos_y * sin_p * sin_r
        r21 = sin_y * sin_r + cos_y * sin_p * cos_r
        r22 = cos_y * cos_p

        # Apply scale
        sx, sy, sz = self.scale
        px, py, pz = self.position

        return [
            [r00 * sx, r01 * sy, r02 * sz, px],
            [r10 * sx, r11 * sy, r12 * sz, py],
            [r20 * sx, r21 * sy, r22 * sz, pz],
            [0.0, 0.0, 0.0, 1.0],
        ]


@runtime_checkable
class TerrainInterface(Protocol):
    """Protocol for terrain data access."""

    def get_height_at(self, x: float, z: float) -> float:
        """Get terrain height at position."""
        ...

    def get_normal_at(self, x: float, z: float) -> Tuple[float, float, float]:
        """Get terrain normal at position."""
        ...

    def get_layer_at(self, x: float, z: float) -> int:
        """Get terrain layer index at position."""
        ...

    def is_water_at(self, x: float, z: float) -> bool:
        """Check if position is in water."""
        ...

    def is_road_at(self, x: float, z: float) -> bool:
        """Check if position is on a road."""
        ...


@dataclass
class Bounds:
    """Axis-aligned bounding box for placement regions."""

    min_x: float = 0.0
    min_z: float = 0.0
    max_x: float = 100.0
    max_z: float = 100.0

    def __post_init__(self) -> None:
        """Validate bounds."""
        if self.min_x > self.max_x:
            raise ValueError("min_x must be <= max_x")
        if self.min_z > self.max_z:
            raise ValueError("min_z must be <= max_z")

    @property
    def width(self) -> float:
        """Get width (X extent)."""
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        """Get depth (Z extent)."""
        return self.max_z - self.min_z

    @property
    def area(self) -> float:
        """Get area of bounds."""
        return self.width * self.depth

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point."""
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_z + self.max_z) / 2,
        )

    def contains(self, x: float, z: float) -> bool:
        """Check if point is within bounds."""
        return self.min_x <= x <= self.max_x and self.min_z <= z <= self.max_z

    def intersects(self, other: "Bounds") -> bool:
        """Check if bounds intersect."""
        return (
            self.min_x <= other.max_x
            and self.max_x >= other.min_x
            and self.min_z <= other.max_z
            and self.max_z >= other.min_z
        )


class NoiseGenerator:
    """
    Deterministic noise generator for placement distribution.

    Uses a combination of hash functions for reproducible pseudo-random
    noise values at any position.
    """

    __slots__ = ("_seed",)

    def __init__(self, seed: int = 0) -> None:
        """
        Initialize noise generator.

        Args:
            seed: Random seed for deterministic generation
        """
        self._seed = seed

    def sample(self, x: float, z: float, scale: float = 1.0) -> float:
        """
        Sample noise value at position.

        Args:
            x: X coordinate
            z: Z coordinate
            scale: Scale factor for noise frequency

        Returns:
            Noise value between 0 and 1
        """
        # Scale coordinates
        sx = x / scale
        sz = z / scale

        # Get integer and fractional parts
        ix = int(math.floor(sx))
        iz = int(math.floor(sz))
        fx = sx - ix
        fz = sz - iz

        # Sample corners
        n00 = self._hash_2d(ix, iz)
        n10 = self._hash_2d(ix + 1, iz)
        n01 = self._hash_2d(ix, iz + 1)
        n11 = self._hash_2d(ix + 1, iz + 1)

        # Smooth interpolation
        u = self._smoothstep(fx)
        v = self._smoothstep(fz)

        # Bilinear interpolation
        nx0 = self._lerp(n00, n10, u)
        nx1 = self._lerp(n01, n11, u)
        return self._lerp(nx0, nx1, v)

    def _hash_2d(self, x: int, z: int) -> float:
        """Hash 2D integer coordinates to float [0, 1]."""
        data = f"{self._seed}:{x}:{z}".encode()
        h = hashlib.md5(data).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    @staticmethod
    def _smoothstep(t: float) -> float:
        """Smooth Hermite interpolation."""
        return t * t * (3 - 2 * t)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + t * (b - a)


class ProceduralPlacer:
    """
    Procedural foliage placement generator.

    Generates placement locations based on terrain data, placement rules,
    and noise distribution. All generation is deterministic with a given seed.
    """

    __slots__ = ("_noise", "_seed")

    def __init__(self, seed: int = 0) -> None:
        """
        Initialize procedural placer.

        Args:
            seed: Random seed for deterministic generation
        """
        self._seed = seed
        self._noise = NoiseGenerator(seed)

    @property
    def seed(self) -> int:
        """Get the random seed."""
        return self._seed

    def evaluate_position(
        self,
        terrain: TerrainInterface,
        x: float,
        z: float,
        rule: PlacementRule,
    ) -> bool:
        """
        Evaluate if a position is valid for placement.

        Args:
            terrain: Terrain data interface
            x: X coordinate
            z: Z coordinate
            rule: Placement rule to evaluate against

        Returns:
            True if position passes all rule checks
        """
        # Check slope
        slope = self.get_slope_at(terrain, x, z)
        if slope < rule.slope_range[0] or slope > rule.slope_range[1]:
            return False

        # Check height
        if rule.height_range is not None:
            height = terrain.get_height_at(x, z)
            if height < rule.height_range[0] or height > rule.height_range[1]:
                return False

        # Check terrain layer
        if rule.terrain_layers:
            layer = terrain.get_layer_at(x, z)
            if layer not in rule.terrain_layers:
                return False

        # Check water exclusion
        if rule.exclude_water and terrain.is_water_at(x, z):
            return False

        # Check road exclusion
        if rule.exclude_roads and terrain.is_road_at(x, z):
            return False

        # Check noise threshold
        noise = self._noise.sample(x, z, rule.noise_scale)
        if noise < rule.noise_threshold:
            return False

        return True

    def get_slope_at(self, terrain: TerrainInterface, x: float, z: float) -> float:
        """
        Get terrain slope at position in degrees.

        Args:
            terrain: Terrain data interface
            x: X coordinate
            z: Z coordinate

        Returns:
            Slope angle in degrees (0 = flat, 90 = vertical)
        """
        normal = terrain.get_normal_at(x, z)
        # Dot product with up vector (0, 1, 0)
        dot = normal[1]
        # Clamp to valid range
        dot = max(-1.0, min(1.0, dot))
        # Convert to angle
        angle = math.degrees(math.acos(dot))
        return angle

    def sample_noise(self, x: float, z: float, scale: float) -> float:
        """
        Sample noise value at position.

        Args:
            x: X coordinate
            z: Z coordinate
            scale: Noise scale factor

        Returns:
            Noise value between 0 and 1
        """
        return self._noise.sample(x, z, scale)

    def generate_in_bounds(
        self,
        terrain: TerrainInterface,
        bounds: Bounds,
        foliage_type_id: str,
        density: float,
        min_spacing: float,
        scale_range: Tuple[float, float],
        rotation_random: bool,
        rule: PlacementRule,
    ) -> List[PlacementResult]:
        """
        Generate placements within bounds.

        Args:
            terrain: Terrain data interface
            bounds: Region to generate within
            foliage_type_id: Foliage type identifier
            density: Target instances per square unit
            min_spacing: Minimum distance between instances
            scale_range: Min/max scale values
            rotation_random: Whether to randomize rotation
            rule: Placement rule for filtering

        Returns:
            List of valid placement results
        """
        results: List[PlacementResult] = []

        # Calculate grid spacing from density
        if density <= 0:
            return results

        spacing = max(min_spacing, 1.0 / math.sqrt(density))

        # Generate candidate positions
        x = bounds.min_x
        while x <= bounds.max_x:
            z = bounds.min_z
            while z <= bounds.max_z:
                # Add jitter for natural look
                jitter_x = self._hash_position(x, z, 0) * spacing * 0.5
                jitter_z = self._hash_position(x, z, 1) * spacing * 0.5
                px = x + jitter_x - spacing * 0.25
                pz = z + jitter_z - spacing * 0.25

                # Clamp to bounds
                px = max(bounds.min_x, min(bounds.max_x, px))
                pz = max(bounds.min_z, min(bounds.max_z, pz))

                # Check validity
                if self.evaluate_position(terrain, px, pz, rule):
                    # Get height
                    py = terrain.get_height_at(px, pz)

                    # Generate variation
                    scale_factor = self._hash_position(px, pz, 2)
                    scale_val = scale_range[0] + scale_factor * (
                        scale_range[1] - scale_range[0]
                    )

                    rotation_y = 0.0
                    if rotation_random:
                        rotation_y = self._hash_position(px, pz, 3) * 360.0

                    results.append(
                        PlacementResult(
                            position=(px, py, pz),
                            rotation=(0.0, rotation_y, 0.0),
                            scale=(scale_val, scale_val, scale_val),
                            foliage_type_id=foliage_type_id,
                        )
                    )

                z += spacing
            x += spacing

        return results

    def _hash_position(self, x: float, z: float, channel: int) -> float:
        """Hash position for deterministic randomness."""
        data = f"{self._seed}:{x:.6f}:{z:.6f}:{channel}".encode()
        h = hashlib.md5(data).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF


@dataclass
class FoliagePlacement:
    """
    Foliage placement configuration for a foliage type.

    Combines a foliage type with placement rules for generation.
    """

    foliage_type_id: str = ""
    rules: PlacementRule = field(default_factory=PlacementRule)
    seed: int = 0
    density: float = 1.0
    min_spacing: float = 1.0
    scale_range: Tuple[float, float] = (0.8, 1.2)
    rotation_random: bool = True

    _placer: Optional[ProceduralPlacer] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize placer."""
        self._placer = ProceduralPlacer(self.seed)

    def generate_placements(
        self, terrain: TerrainInterface, bounds: Bounds
    ) -> List[PlacementResult]:
        """
        Generate placements for this foliage type.

        Args:
            terrain: Terrain data interface
            bounds: Region to generate within

        Returns:
            List of placement results
        """
        if self._placer is None:
            self._placer = ProceduralPlacer(self.seed)

        return self._placer.generate_in_bounds(
            terrain=terrain,
            bounds=bounds,
            foliage_type_id=self.foliage_type_id,
            density=self.density,
            min_spacing=self.min_spacing,
            scale_range=self.scale_range,
            rotation_random=self.rotation_random,
            rule=self.rules,
        )


class ManualPlacement:
    """
    Manual foliage placement manager.

    Allows direct manipulation of foliage instances without
    procedural generation.
    """

    __slots__ = ("_instances", "_next_id")

    def __init__(self) -> None:
        """Initialize manual placement manager."""
        self._instances: Dict[int, PlacementResult] = {}
        self._next_id = 0

    def add_instance(self, placement: PlacementResult) -> int:
        """
        Add a manual placement instance.

        Args:
            placement: Placement to add

        Returns:
            Instance ID for later reference
        """
        instance_id = self._next_id
        self._next_id += 1
        self._instances[instance_id] = placement
        return instance_id

    def remove_instance(self, instance_id: int) -> bool:
        """
        Remove a placement instance.

        Args:
            instance_id: ID of instance to remove

        Returns:
            True if instance was removed, False if not found
        """
        if instance_id in self._instances:
            del self._instances[instance_id]
            return True
        return False

    def move_instance(
        self, instance_id: int, new_position: Tuple[float, float, float]
    ) -> bool:
        """
        Move an instance to a new position.

        Args:
            instance_id: ID of instance to move
            new_position: New position (x, y, z)

        Returns:
            True if instance was moved, False if not found
        """
        if instance_id not in self._instances:
            return False

        old = self._instances[instance_id]
        self._instances[instance_id] = PlacementResult(
            position=new_position,
            rotation=old.rotation,
            scale=old.scale,
            foliage_type_id=old.foliage_type_id,
        )
        return True

    def update_instance(
        self,
        instance_id: int,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float]] = None,
        scale: Optional[Tuple[float, float, float]] = None,
    ) -> bool:
        """
        Update instance properties.

        Args:
            instance_id: ID of instance to update
            position: New position (optional)
            rotation: New rotation (optional)
            scale: New scale (optional)

        Returns:
            True if instance was updated, False if not found
        """
        if instance_id not in self._instances:
            return False

        old = self._instances[instance_id]
        self._instances[instance_id] = PlacementResult(
            position=position if position is not None else old.position,
            rotation=rotation if rotation is not None else old.rotation,
            scale=scale if scale is not None else old.scale,
            foliage_type_id=old.foliage_type_id,
        )
        return True

    def get_instance(self, instance_id: int) -> Optional[PlacementResult]:
        """
        Get an instance by ID.

        Args:
            instance_id: ID of instance

        Returns:
            PlacementResult if found, None otherwise
        """
        return self._instances.get(instance_id)

    def get_all_instances(self) -> List[PlacementResult]:
        """
        Get all placement instances.

        Returns:
            List of all placements
        """
        return list(self._instances.values())

    def get_instances_in_bounds(self, bounds: Bounds) -> List[PlacementResult]:
        """
        Get instances within bounds.

        Args:
            bounds: Bounding region to query

        Returns:
            List of placements within bounds
        """
        results = []
        for placement in self._instances.values():
            x, _, z = placement.position
            if bounds.contains(x, z):
                results.append(placement)
        return results

    def get_instances_by_type(self, foliage_type_id: str) -> List[PlacementResult]:
        """
        Get instances of a specific foliage type.

        Args:
            foliage_type_id: Type to filter by

        Returns:
            List of placements with matching type
        """
        return [
            p for p in self._instances.values() if p.foliage_type_id == foliage_type_id
        ]

    def count(self) -> int:
        """Get total instance count."""
        return len(self._instances)

    def clear(self) -> None:
        """Remove all instances."""
        self._instances.clear()
        self._next_id = 0


def procedural_placement(
    foliage_type_id: str,
    seed: int = 0,
    density: float = 1.0,
    min_spacing: float = 1.0,
    slope_range: Tuple[float, float] = (0.0, 90.0),
    height_range: Optional[Tuple[float, float]] = None,
    terrain_layers: Optional[List[int]] = None,
    noise_threshold: float = 0.0,
    noise_scale: float = 10.0,
) -> Callable[[Type], Type]:
    """
    Decorator for defining procedural placement rules.

    Creates a FoliagePlacement configuration from decorated class.

    Args:
        foliage_type_id: Associated foliage type
        seed: Random seed for generation
        density: Instances per square unit
        min_spacing: Minimum distance between instances
        slope_range: Allowed slope range in degrees
        height_range: Optional allowed height range
        terrain_layers: Allowed terrain layer indices
        noise_threshold: Minimum noise value for placement
        noise_scale: Noise frequency scale

    Returns:
        Decorator function
    """

    def decorator(cls: Type) -> Type:
        rule = PlacementRule(
            slope_range=slope_range,
            height_range=height_range,
            terrain_layers=terrain_layers if terrain_layers is not None else [],
            noise_threshold=noise_threshold,
            noise_scale=noise_scale,
        )

        placement = FoliagePlacement(
            foliage_type_id=foliage_type_id,
            rules=rule,
            seed=seed,
            density=density,
            min_spacing=min_spacing,
            scale_range=getattr(cls, "scale_range", (0.8, 1.2)),
            rotation_random=getattr(cls, "rotation_random", True),
        )

        cls._foliage_placement = placement
        return cls

    return decorator
