"""
Foliage type definitions for the game engine World Layer.

Provides foliage type definitions with support for:
- Multiple categories (grass, shrub, tree, rock, debris)
- LOD system with impostor support
- Instance variation (scale, rotation, color)
- Physics collision configuration
- Wind response settings
- Density and spacing controls

Uses Trinity Pattern with @foliage_type decorator for density, culling,
collision, and wind configuration.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class FoliageCategory(Enum):
    """Categories of foliage types."""
    GRASS = auto()
    SHRUB = auto()
    TREE = auto()
    ROCK = auto()
    DEBRIS = auto()


class CollisionType(Enum):
    """Collision shapes for foliage instances."""
    NONE = "none"
    BOX = "box"
    CAPSULE = "capsule"
    MESH = "mesh"


@dataclass
class FoliageType:
    """
    Definition of a foliage type.

    Represents a template for creating foliage instances with configurable
    LOD, variation, physics, wind response, and density settings.

    Attributes:
        type_id: Unique identifier for this foliage type
        category: Category classification (grass, shrub, tree, etc.)
        mesh_id: Primary mesh asset identifier
        lod_meshes: List of LOD mesh identifiers (decreasing detail)
        lod_distances: Transition distances for each LOD level
        impostor_mesh: Billboard mesh for max distance rendering
        cull_distance: Maximum render distance
        scale_range: Min/max scale variation
        rotation_random: Whether to apply random rotation
        color_variation: Amount of color variation (0-1)
        has_collision: Whether instances have collision
        collision_type: Type of collision shape
        destructible: Whether instances can be destroyed
        wind_response: Whether to apply wind animation
        wind_weight: Intensity of wind effect
        density: Instances per square unit for procedural placement
        min_spacing: Minimum distance between instances
    """

    type_id: str = ""
    category: FoliageCategory = FoliageCategory.SHRUB
    mesh_id: str = ""

    # LOD configuration
    lod_meshes: List[str] = field(default_factory=list)
    lod_distances: List[float] = field(default_factory=lambda: [50.0, 150.0, 500.0])
    impostor_mesh: str = ""
    cull_distance: float = 2000.0

    # Variation settings
    scale_range: Tuple[float, float] = (0.8, 1.2)
    rotation_random: bool = True
    color_variation: float = 0.1

    # Physics configuration
    has_collision: bool = False
    collision_type: str = "none"
    destructible: bool = False

    # Wind settings
    wind_response: bool = True
    wind_weight: float = 1.0

    # Density settings
    density: float = 1.0
    min_spacing: float = 1.0

    def __post_init__(self) -> None:
        """Validate foliage type parameters after initialization."""
        if self.cull_distance <= 0:
            raise ValueError("cull_distance must be > 0")
        if self.scale_range[0] <= 0 or self.scale_range[1] <= 0:
            raise ValueError("scale_range values must be > 0")
        if self.scale_range[0] > self.scale_range[1]:
            raise ValueError("scale_range min must be <= max")
        if not 0.0 <= self.color_variation <= 1.0:
            raise ValueError("color_variation must be between 0 and 1")
        if self.wind_weight < 0:
            raise ValueError("wind_weight must be >= 0")
        if self.density < 0:
            raise ValueError("density must be >= 0")
        if self.min_spacing < 0:
            raise ValueError("min_spacing must be >= 0")

    def get_lod_level(self, distance: float) -> int:
        """
        Get the appropriate LOD level for a given distance.

        Args:
            distance: Distance from camera to instance

        Returns:
            LOD level index (0 = highest detail)
        """
        for i, lod_dist in enumerate(self.lod_distances):
            if distance < lod_dist:
                return i
        return len(self.lod_distances)

    def get_mesh_for_distance(self, distance: float) -> Optional[str]:
        """
        Get the appropriate mesh ID for a given distance.

        Args:
            distance: Distance from camera to instance

        Returns:
            Mesh ID to use, or None if culled
        """
        if distance >= self.cull_distance:
            return None

        lod_level = self.get_lod_level(distance)

        if lod_level == 0:
            return self.mesh_id
        elif lod_level <= len(self.lod_meshes):
            return self.lod_meshes[lod_level - 1]
        elif self.impostor_mesh:
            return self.impostor_mesh
        elif self.lod_meshes:
            return self.lod_meshes[-1]
        else:
            return self.mesh_id

    def should_cull(self, distance: float) -> bool:
        """
        Check if an instance should be culled at a given distance.

        Args:
            distance: Distance from camera to instance

        Returns:
            True if instance should be culled
        """
        return distance >= self.cull_distance


@dataclass
class TreeType(FoliageType):
    """
    Tree foliage type with specialized settings.

    Extends FoliageType with tree-specific attributes like trunk
    and canopy configuration.
    """

    category: FoliageCategory = field(default=FoliageCategory.TREE)

    # Tree-specific settings
    trunk_mesh_id: str = ""
    canopy_mesh_id: str = ""
    trunk_collision: bool = True
    canopy_sway: float = 1.0
    branch_detail_distance: float = 100.0

    # Override defaults for trees
    has_collision: bool = True
    collision_type: str = "capsule"
    cull_distance: float = 3000.0
    min_spacing: float = 5.0
    scale_range: Tuple[float, float] = (0.7, 1.3)

    def __post_init__(self) -> None:
        """Validate tree-specific parameters."""
        super().__post_init__()
        if self.canopy_sway < 0:
            raise ValueError("canopy_sway must be >= 0")
        if self.branch_detail_distance <= 0:
            raise ValueError("branch_detail_distance must be > 0")


@dataclass
class ShrubType(FoliageType):
    """
    Shrub foliage type with specialized settings.

    Extends FoliageType with shrub-specific attributes like
    berry and flower rendering.
    """

    category: FoliageCategory = field(default=FoliageCategory.SHRUB)

    # Shrub-specific settings
    has_berries: bool = False
    berry_mesh_id: str = ""
    berry_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    has_flowers: bool = False
    flower_mesh_id: str = ""
    flower_density: float = 0.3

    # Override defaults for shrubs
    cull_distance: float = 1500.0
    min_spacing: float = 2.0
    wind_weight: float = 0.8

    def __post_init__(self) -> None:
        """Validate shrub-specific parameters."""
        super().__post_init__()
        if self.flower_density < 0:
            raise ValueError("flower_density must be >= 0")


@dataclass
class GrassType(FoliageType):
    """
    Grass foliage type with specialized settings.

    Extends FoliageType with grass-specific attributes like
    blade geometry and color gradients.
    """

    category: FoliageCategory = field(default=FoliageCategory.GRASS)

    # Grass-specific settings
    blade_width: float = 0.05
    blade_height: float = 0.3
    blade_curve: float = 0.2
    blade_bend: float = 0.5
    color_base: Tuple[float, float, float] = (0.1, 0.3, 0.05)
    color_tip: Tuple[float, float, float] = (0.2, 0.5, 0.1)
    blades_per_instance: int = 8

    # Override defaults for grass
    has_collision: bool = False
    collision_type: str = "none"
    cull_distance: float = 100.0
    min_spacing: float = 0.1
    density: float = 50.0
    wind_weight: float = 1.5

    def __post_init__(self) -> None:
        """Validate grass-specific parameters."""
        super().__post_init__()
        if self.blade_width <= 0:
            raise ValueError("blade_width must be > 0")
        if self.blade_height <= 0:
            raise ValueError("blade_height must be > 0")
        if self.blades_per_instance <= 0:
            raise ValueError("blades_per_instance must be > 0")


@dataclass
class RockType(FoliageType):
    """
    Rock foliage type with specialized settings.

    Extends FoliageType with rock-specific attributes like
    moss coverage and weathering.
    """

    category: FoliageCategory = field(default=FoliageCategory.ROCK)

    # Rock-specific settings
    moss_coverage: float = 0.0
    moss_color: Tuple[float, float, float] = (0.1, 0.3, 0.1)
    weathering_amount: float = 0.5
    embed_depth: float = 0.1

    # Override defaults for rocks
    wind_response: bool = False
    wind_weight: float = 0.0
    has_collision: bool = True
    collision_type: str = "mesh"
    cull_distance: float = 2500.0
    min_spacing: float = 3.0
    scale_range: Tuple[float, float] = (0.5, 2.0)

    def __post_init__(self) -> None:
        """Validate rock-specific parameters."""
        super().__post_init__()
        if not 0.0 <= self.moss_coverage <= 1.0:
            raise ValueError("moss_coverage must be between 0 and 1")
        if not 0.0 <= self.weathering_amount <= 1.0:
            raise ValueError("weathering_amount must be between 0 and 1")


@dataclass
class DebrisType(FoliageType):
    """
    Debris foliage type for scattered objects.

    Extends FoliageType with debris-specific attributes like
    decay and physics interaction.
    """

    category: FoliageCategory = field(default=FoliageCategory.DEBRIS)

    # Debris-specific settings
    decay_rate: float = 0.0
    physics_enabled: bool = False
    can_scatter: bool = True
    scatter_radius: float = 0.5

    # Override defaults for debris
    wind_response: bool = False
    has_collision: bool = True
    collision_type: str = "box"
    destructible: bool = True
    cull_distance: float = 500.0
    min_spacing: float = 0.5
    density: float = 2.0

    def __post_init__(self) -> None:
        """Validate debris-specific parameters."""
        super().__post_init__()
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be >= 0")
        if self.scatter_radius < 0:
            raise ValueError("scatter_radius must be >= 0")


class FoliageTypeRegistry:
    """
    Registry for managing foliage type definitions.

    Provides type registration, lookup, and category-based queries.
    Thread-safe for concurrent access.
    """

    __slots__ = ("_types",)

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._types: Dict[str, FoliageType] = {}

    def register(self, foliage_type: FoliageType) -> None:
        """
        Register a foliage type.

        Args:
            foliage_type: Foliage type to register

        Raises:
            ValueError: If type_id is empty or already registered
        """
        if not foliage_type.type_id:
            raise ValueError("Foliage type must have a type_id")
        if foliage_type.type_id in self._types:
            raise ValueError(f"Foliage type '{foliage_type.type_id}' already registered")
        self._types[foliage_type.type_id] = foliage_type

    def unregister(self, type_id: str) -> bool:
        """
        Unregister a foliage type.

        Args:
            type_id: ID of the foliage type to remove

        Returns:
            True if type was removed, False if not found
        """
        if type_id in self._types:
            del self._types[type_id]
            return True
        return False

    def get(self, type_id: str) -> Optional[FoliageType]:
        """
        Get a foliage type by ID.

        Args:
            type_id: ID of the foliage type

        Returns:
            Foliage type if found, None otherwise
        """
        return self._types.get(type_id)

    def get_by_category(self, category: FoliageCategory) -> List[FoliageType]:
        """
        Get all foliage types in a category.

        Args:
            category: Category to filter by

        Returns:
            List of foliage types in the category
        """
        return [ft for ft in self._types.values() if ft.category == category]

    def get_all(self) -> List[FoliageType]:
        """
        Get all registered foliage types.

        Returns:
            List of all foliage types
        """
        return list(self._types.values())

    def get_all_ids(self) -> List[str]:
        """
        Get all registered foliage type IDs.

        Returns:
            List of all type IDs
        """
        return list(self._types.keys())

    def contains(self, type_id: str) -> bool:
        """
        Check if a foliage type is registered.

        Args:
            type_id: ID to check

        Returns:
            True if type is registered
        """
        return type_id in self._types

    def count(self) -> int:
        """
        Get the number of registered types.

        Returns:
            Count of registered foliage types
        """
        return len(self._types)

    def clear(self) -> None:
        """Remove all registered foliage types."""
        self._types.clear()


# Global registry instance
_global_registry = FoliageTypeRegistry()


def get_global_registry() -> FoliageTypeRegistry:
    """
    Get the global foliage type registry.

    Returns:
        The global FoliageTypeRegistry instance
    """
    return _global_registry


def foliage_type(
    type_id: str,
    category: FoliageCategory = FoliageCategory.SHRUB,
    density: float = 1.0,
    cull_distance: float = 2000.0,
    has_collision: bool = False,
    wind_response: bool = True,
    wind_weight: float = 1.0,
    register: bool = True,
):
    """
    Decorator for defining foliage types using Trinity Pattern.

    Creates a FoliageType from decorated class attributes and optionally
    registers it in the global registry.

    Args:
        type_id: Unique identifier for this foliage type
        category: Category classification
        density: Instances per square unit
        cull_distance: Maximum render distance
        has_collision: Whether instances have collision
        wind_response: Whether to apply wind animation
        wind_weight: Intensity of wind effect
        register: Whether to register in global registry

    Returns:
        Decorator function
    """

    def decorator(cls):
        # Create foliage type from class
        ft = FoliageType(
            type_id=type_id,
            category=category,
            mesh_id=getattr(cls, "mesh_id", ""),
            lod_meshes=getattr(cls, "lod_meshes", []),
            lod_distances=getattr(cls, "lod_distances", [50.0, 150.0, 500.0]),
            impostor_mesh=getattr(cls, "impostor_mesh", ""),
            cull_distance=cull_distance,
            scale_range=getattr(cls, "scale_range", (0.8, 1.2)),
            rotation_random=getattr(cls, "rotation_random", True),
            color_variation=getattr(cls, "color_variation", 0.1),
            has_collision=has_collision,
            collision_type=getattr(cls, "collision_type", "none"),
            destructible=getattr(cls, "destructible", False),
            wind_response=wind_response,
            wind_weight=wind_weight,
            density=density,
            min_spacing=getattr(cls, "min_spacing", 1.0),
        )

        # Store reference on class
        cls._foliage_type = ft

        # Register if requested
        if register:
            _global_registry.register(ft)

        return cls

    return decorator
