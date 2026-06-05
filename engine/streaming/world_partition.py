"""
World partition streaming decorators and core types.

Provides decorators for defining streamable world chunks with configurable
size, overlap, streaming priority, and unload policies.

Decorators:
    @chunk: Cell definition with size and overlap params
    @streamable: Streaming priority and keep_loaded flags
    @loading_priority: Visibility and velocity weights for load ordering
    @unloadable: Min age and save state policy for unloading

Example:
    @chunk(size=256.0, overlap=16.0)
    @streamable(priority=StreamPriority.HIGH, keep_loaded=False)
    @loading_priority(visibility_weight=1.5, player_velocity_weight=0.8)
    @unloadable(min_age=30.0, save_state=True)
    @dataclass
    class TerrainChunk:
        terrain_data: TerrainData
        height_map: HeightMap
        foliage: List[FoliageInstance]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

T = TypeVar("T")


class StreamPriority(Enum):
    """Streaming priority levels for chunks."""

    CRITICAL = 100  # Must load immediately (player spawn area)
    HIGH = 75       # Important gameplay areas
    NORMAL = 50     # Standard streaming priority
    LOW = 25        # Background/distant areas
    BACKGROUND = 10 # Load when nothing else pending

    @classmethod
    def from_string(cls, value: str) -> "StreamPriority":
        """Convert string to StreamPriority."""
        mapping = {
            "critical": cls.CRITICAL,
            "high": cls.HIGH,
            "normal": cls.NORMAL,
            "low": cls.LOW,
            "background": cls.BACKGROUND,
        }
        result = mapping.get(value.lower())
        if result is None:
            raise ValueError(
                f"Invalid priority '{value}'. "
                f"Valid values: {list(mapping.keys())}"
            )
        return result


@dataclass
class ChunkConfig:
    """Configuration for a world chunk."""

    size: float = 256.0
    overlap: float = 16.0
    lod_levels: int = 4
    lod_distances: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """Validate chunk configuration."""
        if self.size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.size}")
        if self.overlap < 0:
            raise ValueError(f"Chunk overlap must be non-negative, got {self.overlap}")
        if self.overlap >= self.size / 2:
            raise ValueError(
                f"Chunk overlap ({self.overlap}) must be less than half "
                f"the chunk size ({self.size / 2})"
            )
        if self.lod_levels < 1:
            raise ValueError(f"LOD levels must be at least 1, got {self.lod_levels}")
        if self.lod_distances is not None:
            if len(self.lod_distances) != self.lod_levels:
                raise ValueError(
                    f"LOD distances count ({len(self.lod_distances)}) must match "
                    f"LOD levels ({self.lod_levels})"
                )
            for i, dist in enumerate(self.lod_distances):
                if dist <= 0:
                    raise ValueError(f"LOD distance at index {i} must be positive")
            for i in range(len(self.lod_distances) - 1):
                if self.lod_distances[i] >= self.lod_distances[i + 1]:
                    raise ValueError("LOD distances must be strictly increasing")

    @property
    def effective_size(self) -> float:
        """Get the effective size including overlap."""
        return self.size + self.overlap * 2

    def get_lod_distance(self, lod: int) -> float:
        """Get the distance threshold for a LOD level."""
        if lod < 0 or lod >= self.lod_levels:
            raise IndexError(f"LOD level {lod} out of range [0, {self.lod_levels})")
        if self.lod_distances is not None:
            return self.lod_distances[lod]
        # Default exponential distribution
        base_distance = self.size * 2
        return base_distance * (2 ** lod)


@dataclass
class StreamableConfig:
    """Configuration for streamable behavior."""

    priority: StreamPriority = StreamPriority.NORMAL
    keep_loaded: bool = False
    preload_distance: float = 0.0
    unload_delay: float = 0.0

    def __post_init__(self) -> None:
        """Validate streamable configuration."""
        if isinstance(self.priority, str):
            self.priority = StreamPriority.from_string(self.priority)
        if self.preload_distance < 0:
            raise ValueError(
                f"Preload distance must be non-negative, got {self.preload_distance}"
            )
        if self.unload_delay < 0:
            raise ValueError(
                f"Unload delay must be non-negative, got {self.unload_delay}"
            )


@dataclass
class LoadingPriorityConfig:
    """Configuration for loading priority weights."""

    visibility_weight: float = 1.0
    player_velocity_weight: float = 1.0
    lod_bonus: float = 0.5
    distance_falloff: float = 1.0

    def __post_init__(self) -> None:
        """Validate loading priority configuration."""
        if self.visibility_weight < 0:
            raise ValueError(
                f"Visibility weight must be non-negative, got {self.visibility_weight}"
            )
        if self.player_velocity_weight < 0:
            raise ValueError(
                f"Player velocity weight must be non-negative, "
                f"got {self.player_velocity_weight}"
            )
        if self.lod_bonus < 0:
            raise ValueError(f"LOD bonus must be non-negative, got {self.lod_bonus}")
        if self.distance_falloff <= 0:
            raise ValueError(
                f"Distance falloff must be positive, got {self.distance_falloff}"
            )


@dataclass
class UnloadableConfig:
    """Configuration for unload behavior."""

    min_age: float = 60.0
    save_state: bool = True
    priority_threshold: float = 0.0
    force_unload_distance: float = 0.0

    def __post_init__(self) -> None:
        """Validate unload configuration."""
        if self.min_age <= 0:
            raise ValueError(f"Min age must be positive, got {self.min_age}")
        if self.priority_threshold < 0:
            raise ValueError(
                f"Priority threshold must be non-negative, "
                f"got {self.priority_threshold}"
            )
        if self.force_unload_distance < 0:
            raise ValueError(
                f"Force unload distance must be non-negative, "
                f"got {self.force_unload_distance}"
            )


@dataclass
class WorldChunk:
    """
    Base dataclass for world chunks.

    Combines configuration from all streaming decorators and provides
    the core chunk interface.
    """

    # Cell identification
    cell_x: int = 0
    cell_y: int = 0

    # Configurations (set by decorators)
    chunk_config: ChunkConfig = field(default_factory=ChunkConfig)
    streamable_config: StreamableConfig = field(default_factory=StreamableConfig)
    loading_priority_config: LoadingPriorityConfig = field(
        default_factory=LoadingPriorityConfig
    )
    unloadable_config: UnloadableConfig = field(default_factory=UnloadableConfig)

    # Runtime state
    load_timestamp: float = 0.0
    last_access_timestamp: float = 0.0
    current_lod: int = 0

    # Class-level registry for tracking decorated chunk types
    _registry: ClassVar[Dict[str, Type["WorldChunk"]]] = {}

    @property
    def cell_key(self) -> Tuple[int, int]:
        """Get the cell coordinate as a tuple key."""
        return (self.cell_x, self.cell_y)

    @property
    def age(self) -> float:
        """Get the age since loading (requires current time comparison)."""
        return 0.0  # Must be computed with current time

    def get_age(self, current_time: float) -> float:
        """Get the age since loading."""
        if self.load_timestamp <= 0:
            return 0.0
        return current_time - self.load_timestamp

    def can_unload(self, current_time: float) -> bool:
        """Check if this chunk can be unloaded based on age policy."""
        if self.streamable_config.keep_loaded:
            return False
        age = self.get_age(current_time)
        return age >= self.unloadable_config.min_age

    def get_world_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get the world-space bounds of this chunk."""
        size = self.chunk_config.size
        min_x = self.cell_x * size
        min_y = self.cell_y * size
        max_x = min_x + size
        max_y = min_y + size
        return ((min_x, min_y), (max_x, max_y))

    def get_center(self) -> Tuple[float, float]:
        """Get the world-space center of this chunk."""
        size = self.chunk_config.size
        center_x = (self.cell_x + 0.5) * size
        center_y = (self.cell_y + 0.5) * size
        return (center_x, center_y)

    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance from chunk center to a point."""
        cx, cy = self.get_center()
        dx = x - cx
        dy = y - cy
        return (dx * dx + dy * dy) ** 0.5

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the chunk bounds."""
        (min_x, min_y), (max_x, max_y) = self.get_world_bounds()
        return min_x <= x <= max_x and min_y <= y <= max_y

    @classmethod
    def register_chunk_type(cls, name: str, chunk_type: Type["WorldChunk"]) -> None:
        """Register a chunk type in the global registry."""
        if name in cls._registry:
            raise ValueError(f"Chunk type '{name}' already registered")
        cls._registry[name] = chunk_type

    @classmethod
    def get_chunk_type(cls, name: str) -> Optional[Type["WorldChunk"]]:
        """Get a registered chunk type by name."""
        return cls._registry.get(name)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the chunk type registry."""
        cls._registry.clear()


# =============================================================================
# DECORATOR IMPLEMENTATIONS
# =============================================================================


def chunk(
    size: float = 256.0,
    overlap: float = 16.0,
    lod_levels: int = 4,
    lod_distances: Optional[List[float]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for defining world chunk properties.

    Configures the cell size, overlap region, and LOD levels for a chunk type.

    Args:
        size: Size of the chunk in world units (default: 256.0)
        overlap: Overlap region size for seamless streaming (default: 16.0)
        lod_levels: Number of LOD levels supported (default: 4)
        lod_distances: Custom distance thresholds for each LOD level

    Example:
        @chunk(size=512.0, overlap=32.0, lod_levels=3)
        @dataclass
        class TerrainChunk:
            height_data: np.ndarray

    Raises:
        ValueError: If size is not positive
        ValueError: If overlap is negative or too large
        ValueError: If lod_levels < 1
        ValueError: If lod_distances count doesn't match lod_levels
    """
    # Validate configuration at decoration time
    config = ChunkConfig(
        size=size,
        overlap=overlap,
        lod_levels=lod_levels,
        lod_distances=lod_distances,
    )

    def decorator(cls: Type[T]) -> Type[T]:
        # Store configuration on the class
        cls._chunk_config = config
        cls._chunk_decorator_applied = True

        # Add accessor methods if not present
        if not hasattr(cls, "get_chunk_config"):
            def get_chunk_config(self) -> ChunkConfig:
                return getattr(self.__class__, "_chunk_config", ChunkConfig())
            cls.get_chunk_config = get_chunk_config

        if not hasattr(cls, "cell_size"):
            @property
            def cell_size(self) -> float:
                return self.get_chunk_config().size
            cls.cell_size = cell_size

        if not hasattr(cls, "cell_overlap"):
            @property
            def cell_overlap(self) -> float:
                return self.get_chunk_config().overlap
            cls.cell_overlap = cell_overlap

        return cls

    return decorator


def streamable(
    priority: Union[StreamPriority, str] = StreamPriority.NORMAL,
    keep_loaded: bool = False,
    preload_distance: float = 0.0,
    unload_delay: float = 0.0,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for configuring streaming behavior.

    Sets the streaming priority and loading persistence flags for a chunk type.

    Args:
        priority: Streaming priority level (default: NORMAL)
        keep_loaded: If True, chunk is never unloaded once loaded (default: False)
        preload_distance: Extra distance at which to start preloading (default: 0.0)
        unload_delay: Delay in seconds before unloading (default: 0.0)

    Example:
        @streamable(priority=StreamPriority.HIGH, keep_loaded=True)
        @dataclass
        class SpawnAreaChunk:
            spawn_points: List[SpawnPoint]

    Raises:
        ValueError: If priority string is invalid
        ValueError: If preload_distance is negative
        ValueError: If unload_delay is negative
    """
    # Convert string priority if needed
    if isinstance(priority, str):
        priority = StreamPriority.from_string(priority)

    config = StreamableConfig(
        priority=priority,
        keep_loaded=keep_loaded,
        preload_distance=preload_distance,
        unload_delay=unload_delay,
    )

    def decorator(cls: Type[T]) -> Type[T]:
        cls._streamable_config = config
        cls._streamable_decorator_applied = True

        if not hasattr(cls, "get_streamable_config"):
            def get_streamable_config(self) -> StreamableConfig:
                return getattr(self.__class__, "_streamable_config", StreamableConfig())
            cls.get_streamable_config = get_streamable_config

        if not hasattr(cls, "stream_priority"):
            @property
            def stream_priority(self) -> StreamPriority:
                return self.get_streamable_config().priority
            cls.stream_priority = stream_priority

        if not hasattr(cls, "should_keep_loaded"):
            @property
            def should_keep_loaded(self) -> bool:
                return self.get_streamable_config().keep_loaded
            cls.should_keep_loaded = should_keep_loaded

        return cls

    return decorator


def loading_priority(
    visibility_weight: float = 1.0,
    player_velocity_weight: float = 1.0,
    lod_bonus: float = 0.5,
    distance_falloff: float = 1.0,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for configuring loading priority weights.

    Sets weights used in priority computation for load ordering decisions.

    Args:
        visibility_weight: Weight for visibility-based priority (default: 1.0)
        player_velocity_weight: Weight for velocity prediction (default: 1.0)
        lod_bonus: Bonus priority for higher LOD levels (default: 0.5)
        distance_falloff: Rate of priority falloff with distance (default: 1.0)

    Example:
        @loading_priority(visibility_weight=2.0, player_velocity_weight=1.5)
        @dataclass
        class ImportantAreaChunk:
            data: bytes

    Raises:
        ValueError: If any weight is negative
        ValueError: If distance_falloff is not positive
    """
    config = LoadingPriorityConfig(
        visibility_weight=visibility_weight,
        player_velocity_weight=player_velocity_weight,
        lod_bonus=lod_bonus,
        distance_falloff=distance_falloff,
    )

    def decorator(cls: Type[T]) -> Type[T]:
        cls._loading_priority_config = config
        cls._loading_priority_decorator_applied = True

        if not hasattr(cls, "get_loading_priority_config"):
            def get_loading_priority_config(self) -> LoadingPriorityConfig:
                return getattr(
                    self.__class__, "_loading_priority_config", LoadingPriorityConfig()
                )
            cls.get_loading_priority_config = get_loading_priority_config

        if not hasattr(cls, "visibility_weight"):
            @property
            def visibility_weight(self) -> float:
                return self.get_loading_priority_config().visibility_weight
            cls.visibility_weight = visibility_weight

        if not hasattr(cls, "velocity_weight"):
            @property
            def velocity_weight(self) -> float:
                return self.get_loading_priority_config().player_velocity_weight
            cls.velocity_weight = velocity_weight

        return cls

    return decorator


def unloadable(
    min_age: float = 60.0,
    save_state: bool = True,
    priority_threshold: float = 0.0,
    force_unload_distance: float = 0.0,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for configuring unload behavior.

    Sets the minimum age and state saving policy for chunk unloading.

    Args:
        min_age: Minimum seconds before chunk can be unloaded (default: 60.0)
        save_state: If True, save chunk state before unloading (default: True)
        priority_threshold: Priority below which chunk can be unloaded (default: 0.0)
        force_unload_distance: Distance at which to force unload (default: 0.0)

    Example:
        @unloadable(min_age=30.0, save_state=True)
        @dataclass
        class GameplayChunk:
            entities: List[Entity]

    Raises:
        ValueError: If min_age is not positive
        ValueError: If priority_threshold is negative
        ValueError: If force_unload_distance is negative
    """
    config = UnloadableConfig(
        min_age=min_age,
        save_state=save_state,
        priority_threshold=priority_threshold,
        force_unload_distance=force_unload_distance,
    )

    def decorator(cls: Type[T]) -> Type[T]:
        cls._unloadable_config = config
        cls._unloadable_decorator_applied = True

        if not hasattr(cls, "get_unloadable_config"):
            def get_unloadable_config(self) -> UnloadableConfig:
                return getattr(
                    self.__class__, "_unloadable_config", UnloadableConfig()
                )
            cls.get_unloadable_config = get_unloadable_config

        if not hasattr(cls, "min_unload_age"):
            @property
            def min_unload_age(self) -> float:
                return self.get_unloadable_config().min_age
            cls.min_unload_age = min_unload_age

        if not hasattr(cls, "should_save_state"):
            @property
            def should_save_state(self) -> bool:
                return self.get_unloadable_config().save_state
            cls.should_save_state = should_save_state

        return cls

    return decorator


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_chunk_decorators(cls: Type) -> Dict[str, bool]:
    """Get which chunk decorators have been applied to a class."""
    return {
        "chunk": getattr(cls, "_chunk_decorator_applied", False),
        "streamable": getattr(cls, "_streamable_decorator_applied", False),
        "loading_priority": getattr(cls, "_loading_priority_decorator_applied", False),
        "unloadable": getattr(cls, "_unloadable_decorator_applied", False),
    }


def validate_chunk_class(cls: Type) -> List[str]:
    """
    Validate that a class has all required chunk decorators.

    Returns a list of missing decorator names.
    """
    decorators = get_chunk_decorators(cls)
    missing = [name for name, applied in decorators.items() if not applied]
    return missing


def is_fully_decorated_chunk(cls: Type) -> bool:
    """Check if a class has all chunk decorators applied."""
    return len(validate_chunk_class(cls)) == 0


__all__ = [
    # Decorators
    "chunk",
    "streamable",
    "loading_priority",
    "unloadable",
    # Config classes
    "ChunkConfig",
    "StreamableConfig",
    "LoadingPriorityConfig",
    "UnloadableConfig",
    # Core types
    "WorldChunk",
    "StreamPriority",
    # Helpers
    "get_chunk_decorators",
    "validate_chunk_class",
    "is_fully_decorated_chunk",
]
