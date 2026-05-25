"""
PCG Placement Rules and Filters.

Provides filtering and rule systems for procedural content placement:
- SlopeFilter: Filter by terrain slope
- HeightFilter: Filter by terrain height
- LayerFilter: Filter by terrain layer
- NoiseFilter: Filter by noise threshold
- ExclusionZone: Exclude regions
- BiomeRule: Biome-specific placement rules
- TransformRule: Random transform variations

All filters compose correctly for complex placement logic.
Uses Trinity Pattern with @constraint for generation rules.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.world.pcg.noise import NoiseGenerator, NoiseSettings, create_noise_generator, NoiseType


@dataclass
class TerrainData:
    """
    Terrain data at a specific point.

    Used by filters to make placement decisions.
    """

    height: float = 0.0
    slope: float = 0.0  # Degrees
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    layer_id: int = 0
    layer_weights: Dict[int, float] = field(default_factory=dict)
    biome_id: str = "default"
    moisture: float = 0.5
    temperature: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlopeFilter:
    """
    Filter based on terrain slope.

    Allows placement only within a slope range.
    """

    min_slope: float = 0.0   # Degrees
    max_slope: float = 90.0  # Degrees

    def __post_init__(self) -> None:
        """Validate filter parameters."""
        if self.min_slope < 0:
            raise ValueError(f"min_slope must be >= 0, got {self.min_slope}")
        if self.max_slope > 90:
            raise ValueError(f"max_slope must be <= 90, got {self.max_slope}")
        if self.min_slope > self.max_slope:
            raise ValueError(
                f"min_slope ({self.min_slope}) must be <= max_slope ({self.max_slope})"
            )


@dataclass
class HeightFilter:
    """
    Filter based on terrain height.

    Allows placement only within a height range.
    """

    min_height: float = -1000.0
    max_height: float = 1000.0

    def __post_init__(self) -> None:
        """Validate filter parameters."""
        if self.min_height > self.max_height:
            raise ValueError(
                f"min_height ({self.min_height}) must be <= max_height ({self.max_height})"
            )


@dataclass
class LayerFilter:
    """
    Filter based on terrain layer.

    Allows placement only on specified layers.
    """

    allowed_layers: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and convert to set for fast lookup."""
        self._allowed_set: Set[int] = set(self.allowed_layers)

    def is_allowed(self, layer_id: int) -> bool:
        """Check if a layer is allowed."""
        if not self._allowed_set:
            return True  # No restriction
        return layer_id in self._allowed_set


@dataclass
class NoiseFilter:
    """
    Filter based on noise value at position.

    Uses noise to create organic placement boundaries.
    """

    noise_settings: NoiseSettings = field(default_factory=NoiseSettings)
    threshold: float = 0.5
    invert: bool = False
    _generator: Optional[NoiseGenerator] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Create noise generator."""
        self._generator = create_noise_generator(
            self.noise_settings.noise_type,
            self.noise_settings.seed,
            self.noise_settings,
        )

    def evaluate(self, x: float, z: float) -> bool:
        """
        Evaluate filter at position.

        Args:
            x: X coordinate
            z: Z coordinate

        Returns:
            True if position passes filter
        """
        if self._generator is None:
            return True

        # Sample noise (normalize from [-1,1] to [0,1])
        value = (self._generator.sample(x, z) + 1.0) / 2.0

        # Apply threshold
        passes = value >= self.threshold

        # Apply inversion
        return not passes if self.invert else passes


@dataclass
class ExclusionZone:
    """
    Circular exclusion zone.

    Prevents placement within a radius of a center point.
    """

    center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 10.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.radius <= 0:
            raise ValueError(f"radius must be > 0, got {self.radius}")

    def contains(self, x: float, z: float) -> bool:
        """
        Check if point is within exclusion zone.

        Args:
            x: X coordinate
            z: Z coordinate

        Returns:
            True if point is excluded
        """
        dx = x - self.center[0]
        dz = z - self.center[1]
        return (dx * dx + dz * dz) <= (self.radius * self.radius)


class PlacementFilter(ABC):
    """
    Abstract base class for placement filters.

    Filters determine whether placement is valid at a location.
    """

    @abstractmethod
    def evaluate(self, x: float, z: float, terrain_data: TerrainData) -> bool:
        """
        Evaluate filter at position.

        Args:
            x: X coordinate
            z: Z coordinate
            terrain_data: Terrain information at position

        Returns:
            True if position passes filter
        """
        pass

    def __and__(self, other: "PlacementFilter") -> "CompoundFilter":
        """Combine filters with AND logic."""
        return CompoundFilter([self, other], mode="all")

    def __or__(self, other: "PlacementFilter") -> "CompoundFilter":
        """Combine filters with OR logic."""
        return CompoundFilter([self, other], mode="any")


class SlopeFilterImpl(PlacementFilter):
    """Implementation of slope-based placement filter."""

    def __init__(self, config: SlopeFilter) -> None:
        """
        Initialize slope filter.

        Args:
            config: Slope filter configuration
        """
        self._config = config

    @property
    def config(self) -> SlopeFilter:
        """Get filter configuration."""
        return self._config

    def evaluate(self, x: float, z: float, terrain_data: TerrainData) -> bool:
        """
        Evaluate slope filter.

        Args:
            x: X coordinate (unused)
            z: Z coordinate (unused)
            terrain_data: Terrain data with slope

        Returns:
            True if slope is within range
        """
        return self._config.min_slope <= terrain_data.slope <= self._config.max_slope


class HeightFilterImpl(PlacementFilter):
    """Implementation of height-based placement filter."""

    def __init__(self, config: HeightFilter) -> None:
        """
        Initialize height filter.

        Args:
            config: Height filter configuration
        """
        self._config = config

    @property
    def config(self) -> HeightFilter:
        """Get filter configuration."""
        return self._config

    def evaluate(self, x: float, z: float, terrain_data: TerrainData) -> bool:
        """
        Evaluate height filter.

        Args:
            x: X coordinate (unused)
            z: Z coordinate (unused)
            terrain_data: Terrain data with height

        Returns:
            True if height is within range
        """
        return (
            self._config.min_height <= terrain_data.height <= self._config.max_height
        )


class LayerFilterImpl(PlacementFilter):
    """Implementation of layer-based placement filter."""

    def __init__(self, config: LayerFilter) -> None:
        """
        Initialize layer filter.

        Args:
            config: Layer filter configuration
        """
        self._config = config

    @property
    def config(self) -> LayerFilter:
        """Get filter configuration."""
        return self._config

    def evaluate(self, x: float, z: float, terrain_data: TerrainData) -> bool:
        """
        Evaluate layer filter.

        Args:
            x: X coordinate (unused)
            z: Z coordinate (unused)
            terrain_data: Terrain data with layer info

        Returns:
            True if layer is allowed
        """
        return self._config.is_allowed(terrain_data.layer_id)


class NoiseFilterImpl(PlacementFilter):
    """Implementation of noise-based placement filter."""

    def __init__(self, config: NoiseFilter) -> None:
        """
        Initialize noise filter.

        Args:
            config: Noise filter configuration
        """
        self._config = config

    @property
    def config(self) -> NoiseFilter:
        """Get filter configuration."""
        return self._config

    def evaluate(self, x: float, z: float, terrain_data: TerrainData) -> bool:
        """
        Evaluate noise filter.

        Args:
            x: X coordinate
            z: Z coordinate
            terrain_data: Terrain data (unused)

        Returns:
            True if noise passes threshold
        """
        return self._config.evaluate(x, z)


class ExclusionZoneFilter(PlacementFilter):
    """Placement filter for exclusion zones."""

    def __init__(self, zones: List[ExclusionZone]) -> None:
        """
        Initialize exclusion zone filter.

        Args:
            zones: List of exclusion zones
        """
        self._zones = list(zones)

    @property
    def zones(self) -> List[ExclusionZone]:
        """Get exclusion zones."""
        return self._zones

    def add_zone(self, zone: ExclusionZone) -> None:
        """Add an exclusion zone."""
        self._zones.append(zone)

    def evaluate(self, x: float, z: float, terrain_data: TerrainData) -> bool:
        """
        Evaluate exclusion zones.

        Args:
            x: X coordinate
            z: Z coordinate
            terrain_data: Terrain data (unused)

        Returns:
            True if point is NOT in any exclusion zone
        """
        for zone in self._zones:
            if zone.contains(x, z):
                return False
        return True


class CompoundFilter(PlacementFilter):
    """
    Compound filter combining multiple filters.

    Supports "all" (AND) and "any" (OR) combination modes.
    """

    def __init__(
        self,
        filters: List[PlacementFilter],
        mode: str = "all",
    ) -> None:
        """
        Initialize compound filter.

        Args:
            filters: List of filters to combine
            mode: "all" for AND, "any" for OR
        """
        if mode not in ("all", "any"):
            raise ValueError(f"mode must be 'all' or 'any', got '{mode}'")

        self._filters = list(filters)
        self._mode = mode

    @property
    def filters(self) -> List[PlacementFilter]:
        """Get component filters."""
        return self._filters

    @property
    def mode(self) -> str:
        """Get combination mode."""
        return self._mode

    def add_filter(self, filter_: PlacementFilter) -> None:
        """Add a filter to the compound."""
        self._filters.append(filter_)

    def evaluate(self, x: float, z: float, terrain_data: TerrainData) -> bool:
        """
        Evaluate compound filter.

        Args:
            x: X coordinate
            z: Z coordinate
            terrain_data: Terrain data

        Returns:
            Result based on mode and component filters
        """
        if not self._filters:
            return True

        if self._mode == "all":
            return all(f.evaluate(x, z, terrain_data) for f in self._filters)
        else:  # mode == "any"
            return any(f.evaluate(x, z, terrain_data) for f in self._filters)


@dataclass
class BiomeRule:
    """
    Biome-specific placement rules.

    Defines what can be placed in a biome and how.
    """

    biome_id: str
    foliage_types: List[str] = field(default_factory=list)
    density_multipliers: Dict[str, float] = field(default_factory=dict)
    filters: List[PlacementFilter] = field(default_factory=list)

    def get_density_multiplier(self, foliage_type: str) -> float:
        """
        Get density multiplier for a foliage type.

        Args:
            foliage_type: Type of foliage

        Returns:
            Density multiplier (1.0 if not specified)
        """
        return self.density_multipliers.get(foliage_type, 1.0)

    def is_foliage_allowed(self, foliage_type: str) -> bool:
        """
        Check if a foliage type is allowed in this biome.

        Args:
            foliage_type: Type to check

        Returns:
            True if allowed
        """
        if not self.foliage_types:
            return True  # No restriction
        return foliage_type in self.foliage_types

    def evaluate_filters(
        self, x: float, z: float, terrain_data: TerrainData
    ) -> bool:
        """
        Evaluate all biome filters.

        Args:
            x: X coordinate
            z: Z coordinate
            terrain_data: Terrain data

        Returns:
            True if all filters pass
        """
        return all(f.evaluate(x, z, terrain_data) for f in self.filters)


class PlacementRuleSet:
    """
    Collection of placement rules organized by biome.

    Provides unified evaluation of placement validity.
    """

    def __init__(self) -> None:
        """Initialize empty rule set."""
        self._rules: Dict[str, BiomeRule] = {}
        self._global_filters: List[PlacementFilter] = []
        self._default_rule: Optional[BiomeRule] = None

    @property
    def rules(self) -> Dict[str, BiomeRule]:
        """Get all biome rules."""
        return self._rules

    @property
    def global_filters(self) -> List[PlacementFilter]:
        """Get global filters."""
        return self._global_filters

    def add_rule(self, rule: BiomeRule) -> None:
        """
        Add a biome rule.

        Args:
            rule: Biome rule to add
        """
        self._rules[rule.biome_id] = rule

    def add_global_filter(self, filter_: PlacementFilter) -> None:
        """
        Add a global filter that applies to all biomes.

        Args:
            filter_: Filter to add
        """
        self._global_filters.append(filter_)

    def set_default_rule(self, rule: BiomeRule) -> None:
        """
        Set default rule for unknown biomes.

        Args:
            rule: Default rule
        """
        self._default_rule = rule

    def get_rule(self, biome_id: str) -> Optional[BiomeRule]:
        """
        Get rule for a biome.

        Args:
            biome_id: Biome identifier

        Returns:
            BiomeRule or None
        """
        return self._rules.get(biome_id, self._default_rule)

    def evaluate(
        self,
        x: float,
        z: float,
        terrain_data: TerrainData,
        foliage_type: Optional[str] = None,
    ) -> List[str]:
        """
        Evaluate rules and return valid foliage types.

        Args:
            x: X coordinate
            z: Z coordinate
            terrain_data: Terrain data
            foliage_type: Optional specific type to check

        Returns:
            List of valid foliage type names
        """
        # Check global filters first
        for filter_ in self._global_filters:
            if not filter_.evaluate(x, z, terrain_data):
                return []

        # Get biome rule
        rule = self.get_rule(terrain_data.biome_id)
        if rule is None:
            return []

        # Check biome filters
        if not rule.evaluate_filters(x, z, terrain_data):
            return []

        # Get valid foliage types
        if foliage_type is not None:
            if rule.is_foliage_allowed(foliage_type):
                return [foliage_type]
            return []

        return [ft for ft in rule.foliage_types if rule.is_foliage_allowed(ft)]


@dataclass
class TransformRule:
    """
    Rules for random transform variations.

    Used to add natural variation to placed objects.
    """

    scale_range: Tuple[float, float] = (0.8, 1.2)
    rotation_range: Tuple[float, float] = (0.0, 360.0)  # Degrees
    offset_range: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate ranges."""
        if self.scale_range[0] > self.scale_range[1]:
            raise ValueError("scale_range min must be <= max")
        if self.scale_range[0] <= 0:
            raise ValueError("scale_range min must be > 0")
        if self.rotation_range[0] > self.rotation_range[1]:
            raise ValueError("rotation_range min must be <= max")

    def apply(
        self,
        base_transform: "Transform",
        seed: int,
    ) -> "Transform":
        """
        Apply random variation to a base transform.

        Args:
            base_transform: Base transform to modify
            seed: Seed for deterministic randomization

        Returns:
            Modified transform
        """
        # Simple LCG for deterministic random
        def random_float(state: int, min_val: float, max_val: float) -> Tuple[int, float]:
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            t = state / 0x7FFFFFFF
            return state, min_val + t * (max_val - min_val)

        state = seed

        # Generate random values
        state, scale_mult = random_float(state, self.scale_range[0], self.scale_range[1])
        state, rotation_offset = random_float(
            state, self.rotation_range[0], self.rotation_range[1]
        )
        state, offset_x = random_float(state, -self.offset_range[0], self.offset_range[0])
        state, offset_z = random_float(state, -self.offset_range[1], self.offset_range[1])

        # Apply to base transform
        return Transform(
            position=(
                base_transform.position[0] + offset_x,
                base_transform.position[1],
                base_transform.position[2] + offset_z,
            ),
            rotation=(
                base_transform.rotation[0],
                base_transform.rotation[1] + rotation_offset,
                base_transform.rotation[2],
            ),
            scale=(
                base_transform.scale[0] * scale_mult,
                base_transform.scale[1] * scale_mult,
                base_transform.scale[2] * scale_mult,
            ),
        )


@dataclass
class Transform:
    """3D transform with position, rotation, and scale."""

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Euler angles in degrees
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    @staticmethod
    def identity() -> "Transform":
        """Create an identity transform."""
        return Transform()

    @staticmethod
    def from_position(x: float, y: float, z: float) -> "Transform":
        """Create a transform with only position."""
        return Transform(position=(x, y, z))


class PlacementValidator:
    """
    Validates placement positions using rules and filters.

    Provides a convenient interface for checking placement validity.
    """

    def __init__(
        self,
        rule_set: Optional[PlacementRuleSet] = None,
        terrain_sampler: Optional[Callable[[float, float], TerrainData]] = None,
    ) -> None:
        """
        Initialize validator.

        Args:
            rule_set: Rules to apply
            terrain_sampler: Function to sample terrain at position
        """
        self._rule_set = rule_set or PlacementRuleSet()
        self._terrain_sampler = terrain_sampler

    @property
    def rule_set(self) -> PlacementRuleSet:
        """Get the rule set."""
        return self._rule_set

    def set_terrain_sampler(
        self, sampler: Callable[[float, float], TerrainData]
    ) -> None:
        """
        Set the terrain sampling function.

        Args:
            sampler: Function taking (x, z) and returning TerrainData
        """
        self._terrain_sampler = sampler

    def is_valid(
        self,
        x: float,
        z: float,
        foliage_type: Optional[str] = None,
        terrain_data: Optional[TerrainData] = None,
    ) -> bool:
        """
        Check if placement is valid at position.

        Args:
            x: X coordinate
            z: Z coordinate
            foliage_type: Optional specific type to check
            terrain_data: Pre-sampled terrain data (samples if None)

        Returns:
            True if placement is valid
        """
        # Get terrain data
        if terrain_data is None:
            if self._terrain_sampler is not None:
                terrain_data = self._terrain_sampler(x, z)
            else:
                terrain_data = TerrainData()

        # Evaluate rules
        valid_types = self._rule_set.evaluate(x, z, terrain_data, foliage_type)
        return len(valid_types) > 0

    def get_valid_types(
        self,
        x: float,
        z: float,
        terrain_data: Optional[TerrainData] = None,
    ) -> List[str]:
        """
        Get all valid foliage types at position.

        Args:
            x: X coordinate
            z: Z coordinate
            terrain_data: Pre-sampled terrain data

        Returns:
            List of valid foliage type names
        """
        if terrain_data is None:
            if self._terrain_sampler is not None:
                terrain_data = self._terrain_sampler(x, z)
            else:
                terrain_data = TerrainData()

        return self._rule_set.evaluate(x, z, terrain_data)


# Factory functions for creating filters
def create_slope_filter(min_slope: float = 0.0, max_slope: float = 45.0) -> SlopeFilterImpl:
    """Create a slope filter."""
    return SlopeFilterImpl(SlopeFilter(min_slope=min_slope, max_slope=max_slope))


def create_height_filter(
    min_height: float = -1000.0, max_height: float = 1000.0
) -> HeightFilterImpl:
    """Create a height filter."""
    return HeightFilterImpl(HeightFilter(min_height=min_height, max_height=max_height))


def create_layer_filter(allowed_layers: List[int]) -> LayerFilterImpl:
    """Create a layer filter."""
    return LayerFilterImpl(LayerFilter(allowed_layers=allowed_layers))


def create_noise_filter(
    seed: int = 0,
    threshold: float = 0.5,
    invert: bool = False,
    frequency: float = 1.0,
) -> NoiseFilterImpl:
    """Create a noise filter."""
    settings = NoiseSettings(seed=seed, frequency=frequency)
    return NoiseFilterImpl(NoiseFilter(
        noise_settings=settings,
        threshold=threshold,
        invert=invert,
    ))


def create_exclusion_filter(zones: List[ExclusionZone]) -> ExclusionZoneFilter:
    """Create an exclusion zone filter."""
    return ExclusionZoneFilter(zones)


__all__ = [
    "TerrainData",
    "SlopeFilter",
    "HeightFilter",
    "LayerFilter",
    "NoiseFilter",
    "ExclusionZone",
    "PlacementFilter",
    "SlopeFilterImpl",
    "HeightFilterImpl",
    "LayerFilterImpl",
    "NoiseFilterImpl",
    "ExclusionZoneFilter",
    "CompoundFilter",
    "BiomeRule",
    "PlacementRuleSet",
    "TransformRule",
    "Transform",
    "PlacementValidator",
    "create_slope_filter",
    "create_height_filter",
    "create_layer_filter",
    "create_noise_filter",
    "create_exclusion_filter",
]
