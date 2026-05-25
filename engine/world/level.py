"""
Level architecture for the World Layer.

Provides level types, streaming levels, sub-levels, level instances, and world composition
for managing complex open-world and multi-level game environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from engine.core.math.vec import Vec3
from engine.core.math.geometry import AABB
from engine.world.constants import (
    HALF_MULTIPLIER,
    TILE_VERTICAL_MIN,
    TILE_VERTICAL_MAX,
    DEFAULT_STREAMING_LOAD_DISTANCE,
    DEFAULT_STREAMING_UNLOAD_DISTANCE,
    DEFAULT_STREAMING_HYSTERESIS,
    DEFAULT_ORIGIN_SHIFT_THRESHOLD,
    DEFAULT_TILE_SIZE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_ROTATION_QUATERNION,
    DEFAULT_SCALE,
)


class LevelType(Enum):
    """Types of levels in the world hierarchy."""
    PERSISTENT = auto()  # Always loaded base level
    STREAMING = auto()   # Dynamically loaded based on distance
    SUB_LEVEL = auto()   # Nested content within another level
    INSTANCE = auto()    # Runtime copy of a level template


class LevelLayer(Enum):
    """Layers for organizing level content."""
    BASE_GEOMETRY = auto()   # Static geometry and collision
    LIGHTING = auto()        # Lights, probes, GI data
    GAMEPLAY = auto()        # Interactive objects, triggers
    AUDIO = auto()           # Sound emitters, ambient audio
    NAVIGATION = auto()      # NavMesh and pathfinding data
    VFX = auto()             # Particle systems, decals


class LevelState(Enum):
    """Loading state of a level."""
    UNLOADED = auto()
    LOADING = auto()
    LOADED = auto()
    VISIBLE = auto()
    UNLOADING = auto()


@dataclass
class LevelBounds:
    """Axis-aligned bounding box for a level."""
    min_point: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    max_point: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))

    @property
    def center(self) -> Vec3:
        """Get the center point of the bounds."""
        return (self.min_point + self.max_point) * HALF_MULTIPLIER

    @property
    def extents(self) -> Vec3:
        """Get the half-extents of the bounds."""
        return (self.max_point - self.min_point) * HALF_MULTIPLIER

    @property
    def size(self) -> Vec3:
        """Get the full size of the bounds."""
        return self.max_point - self.min_point

    def contains_point(self, point: Vec3) -> bool:
        """Check if a point is within the bounds."""
        return (
            self.min_point.x <= point.x <= self.max_point.x
            and self.min_point.y <= point.y <= self.max_point.y
            and self.min_point.z <= point.z <= self.max_point.z
        )

    def intersects(self, other: LevelBounds) -> bool:
        """Check if this bounds intersects another."""
        return (
            self.min_point.x <= other.max_point.x and self.max_point.x >= other.min_point.x
            and self.min_point.y <= other.max_point.y and self.max_point.y >= other.min_point.y
            and self.min_point.z <= other.max_point.z and self.max_point.z >= other.min_point.z
        )

    def expand(self, point: Vec3) -> None:
        """Expand bounds to include a point."""
        self.min_point = Vec3(
            min(self.min_point.x, point.x),
            min(self.min_point.y, point.y),
            min(self.min_point.z, point.z),
        )
        self.max_point = Vec3(
            max(self.max_point.x, point.x),
            max(self.max_point.y, point.y),
            max(self.max_point.z, point.z),
        )

    def merge(self, other: LevelBounds) -> LevelBounds:
        """Create a new bounds that contains both bounds."""
        return LevelBounds(
            min_point=Vec3(
                min(self.min_point.x, other.min_point.x),
                min(self.min_point.y, other.min_point.y),
                min(self.min_point.z, other.min_point.z),
            ),
            max_point=Vec3(
                max(self.max_point.x, other.max_point.x),
                max(self.max_point.y, other.max_point.y),
                max(self.max_point.z, other.max_point.z),
            ),
        )

    def to_aabb(self) -> AABB:
        """Convert to core AABB type."""
        return AABB(self.min_point, self.max_point)


@dataclass
class Actor:
    """Represents an actor/entity within a level."""
    id: str = ""
    name: str = ""
    transform_position: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    transform_rotation: Tuple[float, float, float, float] = DEFAULT_ROTATION_QUATERNION
    transform_scale: Vec3 = field(default_factory=lambda: Vec3(DEFAULT_SCALE, DEFAULT_SCALE, DEFAULT_SCALE))
    layer: LevelLayer = LevelLayer.GAMEPLAY
    tags: Set[str] = field(default_factory=set)
    persistent: bool = False
    serializable: bool = True

    def get_position(self) -> Vec3:
        """Get the actor's world position."""
        return self.transform_position

    def set_position(self, pos: Vec3) -> None:
        """Set the actor's world position."""
        self.transform_position = pos


@dataclass
class Level:
    """
    Base level class representing a collection of actors and content.

    Levels are the fundamental unit of world organization, containing
    actors grouped by layers for efficient streaming and rendering.
    """
    name: str = ""
    level_type: LevelType = LevelType.PERSISTENT
    bounds: LevelBounds = field(default_factory=LevelBounds)
    actors: List[Actor] = field(default_factory=list)
    layers_enabled: Dict[LevelLayer, bool] = field(default_factory=lambda: {
        layer: True for layer in LevelLayer
    })
    state: LevelState = LevelState.UNLOADED
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Callbacks
    _on_load_callbacks: List[Callable[["Level"], None]] = field(
        default_factory=list, repr=False
    )
    _on_unload_callbacks: List[Callable[["Level"], None]] = field(
        default_factory=list, repr=False
    )

    def add_actor(self, actor: Actor) -> None:
        """Add an actor to the level."""
        self.actors.append(actor)
        self.bounds.expand(actor.transform_position)

    def remove_actor(self, actor: Actor) -> bool:
        """Remove an actor from the level. Returns True if found and removed."""
        if actor in self.actors:
            self.actors.remove(actor)
            return True
        return False

    def get_actor_by_id(self, actor_id: str) -> Optional[Actor]:
        """Find an actor by its ID."""
        for actor in self.actors:
            if actor.id == actor_id:
                return actor
        return None

    def get_actors_by_layer(self, layer: LevelLayer) -> List[Actor]:
        """Get all actors in a specific layer."""
        return [a for a in self.actors if a.layer == layer]

    def get_actors_by_tag(self, tag: str) -> List[Actor]:
        """Get all actors with a specific tag."""
        return [a for a in self.actors if tag in a.tags]

    def get_actors_in_bounds(self, bounds: LevelBounds) -> List[Actor]:
        """Get all actors within specified bounds."""
        return [
            a for a in self.actors
            if bounds.contains_point(a.transform_position)
        ]

    def is_layer_enabled(self, layer: LevelLayer) -> bool:
        """Check if a layer is enabled."""
        return self.layers_enabled.get(layer, True)

    def set_layer_enabled(self, layer: LevelLayer, enabled: bool) -> None:
        """Enable or disable a layer."""
        self.layers_enabled[layer] = enabled

    def load(self) -> bool:
        """
        Load the level content.

        Returns True if loading succeeded.
        """
        if self.state != LevelState.UNLOADED:
            return False

        self.state = LevelState.LOADING
        # Actual loading logic would be implemented here
        self.state = LevelState.LOADED

        for callback in self._on_load_callbacks:
            callback(self)

        return True

    def unload(self) -> bool:
        """
        Unload the level content.

        Returns True if unloading succeeded.
        """
        if self.state not in (LevelState.LOADED, LevelState.VISIBLE):
            return False

        self.state = LevelState.UNLOADING

        for callback in self._on_unload_callbacks:
            callback(self)

        # Actual unloading logic would be implemented here
        self.state = LevelState.UNLOADED

        return True

    def on_load(self, callback: Callable[["Level"], None]) -> None:
        """Register a callback for when the level is loaded."""
        self._on_load_callbacks.append(callback)

    def on_unload(self, callback: Callable[["Level"], None]) -> None:
        """Register a callback for when the level is unloaded."""
        self._on_unload_callbacks.append(callback)

    def serialize(self) -> Dict[str, Any]:
        """Serialize the level to a dictionary."""
        return {
            "name": self.name,
            "level_type": self.level_type.name,
            "bounds": {
                "min": [self.bounds.min_point.x, self.bounds.min_point.y, self.bounds.min_point.z],
                "max": [self.bounds.max_point.x, self.bounds.max_point.y, self.bounds.max_point.z],
            },
            "actors": [
                {
                    "id": a.id,
                    "name": a.name,
                    "position": [a.transform_position.x, a.transform_position.y, a.transform_position.z],
                    "rotation": list(a.transform_rotation),
                    "scale": [a.transform_scale.x, a.transform_scale.y, a.transform_scale.z],
                    "layer": a.layer.name,
                    "tags": list(a.tags),
                    "persistent": a.persistent,
                }
                for a in self.actors if a.serializable
            ],
            "layers_enabled": {layer.name: enabled for layer, enabled in self.layers_enabled.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "Level":
        """Deserialize a level from a dictionary."""
        bounds = LevelBounds(
            min_point=Vec3(*data["bounds"]["min"]),
            max_point=Vec3(*data["bounds"]["max"]),
        )

        actors = []
        for actor_data in data.get("actors", []):
            actor = Actor(
                id=actor_data["id"],
                name=actor_data["name"],
                transform_position=Vec3(*actor_data["position"]),
                transform_rotation=tuple(actor_data["rotation"]),
                transform_scale=Vec3(*actor_data["scale"]),
                layer=LevelLayer[actor_data["layer"]],
                tags=set(actor_data.get("tags", [])),
                persistent=actor_data.get("persistent", False),
            )
            actors.append(actor)

        layers_enabled = {
            LevelLayer[name]: enabled
            for name, enabled in data.get("layers_enabled", {}).items()
        }

        return cls(
            name=data["name"],
            level_type=LevelType[data["level_type"]],
            bounds=bounds,
            actors=actors,
            layers_enabled=layers_enabled,
            metadata=data.get("metadata", {}),
        )

    def recalculate_bounds(self) -> None:
        """Recalculate bounds based on all actors."""
        if not self.actors:
            self.bounds = LevelBounds()
            return

        first_pos = self.actors[0].transform_position
        self.bounds = LevelBounds(
            min_point=Vec3(first_pos.x, first_pos.y, first_pos.z),
            max_point=Vec3(first_pos.x, first_pos.y, first_pos.z),
        )

        for actor in self.actors[1:]:
            self.bounds.expand(actor.transform_position)


@dataclass
class StreamingLevel(Level):
    """
    A level that is dynamically loaded based on distance from streaming sources.

    Supports load distance, unload distance with hysteresis, and priority-based loading.
    """
    load_distance: float = DEFAULT_STREAMING_LOAD_DISTANCE
    unload_distance: float = DEFAULT_STREAMING_UNLOAD_DISTANCE
    priority: int = 0  # Higher priority loads first
    hysteresis: float = DEFAULT_STREAMING_HYSTERESIS
    load_progress: float = 0.0  # 0.0 to 1.0

    # Reference point for distance calculations
    reference_point: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))

    def __post_init__(self) -> None:
        """Ensure level type is set correctly."""
        self.level_type = LevelType.STREAMING
        # Ensure unload distance accounts for hysteresis
        if self.unload_distance <= self.load_distance:
            self.unload_distance = self.load_distance + self.hysteresis

    def should_load(self, source_position: Vec3) -> bool:
        """
        Determine if the level should be loaded based on distance to source.

        Args:
            source_position: Position of the streaming source (e.g., player).

        Returns:
            True if the level should be loaded.
        """
        if self.state != LevelState.UNLOADED:
            return False

        distance = self._distance_to_source(source_position)
        return distance <= self.load_distance

    def should_unload(self, source_position: Vec3) -> bool:
        """
        Determine if the level should be unloaded based on distance to source.

        Uses hysteresis to prevent thrashing.

        Args:
            source_position: Position of the streaming source.

        Returns:
            True if the level should be unloaded.
        """
        if self.state not in (LevelState.LOADED, LevelState.VISIBLE):
            return False

        distance = self._distance_to_source(source_position)
        return distance > self.unload_distance

    def _distance_to_source(self, source_position: Vec3) -> float:
        """Calculate distance from the level's reference point to a source."""
        # Use bounds center if no explicit reference point set
        ref = self.reference_point if self.reference_point != Vec3(0, 0, 0) else self.bounds.center
        return ref.distance(source_position)

    def update_load_progress(self, progress: float) -> None:
        """
        Update the loading progress.

        Args:
            progress: Loading progress from 0.0 to 1.0.
        """
        self.load_progress = max(0.0, min(1.0, progress))

        if self.load_progress >= 1.0 and self.state == LevelState.LOADING:
            self.state = LevelState.LOADED


@dataclass
class SubLevel(Level):
    """
    A level nested within another parent level.

    Sub-levels share the parent's coordinate space with an optional
    relative transform offset.
    """
    parent: Optional[Level] = None
    relative_offset: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    relative_rotation: Tuple[float, float, float, float] = DEFAULT_ROTATION_QUATERNION
    inherit_layers: bool = True

    def __post_init__(self) -> None:
        """Ensure level type is set correctly."""
        self.level_type = LevelType.SUB_LEVEL

    def set_parent(self, parent: Level) -> None:
        """Set the parent level."""
        self.parent = parent

    def get_world_position(self, local_position: Vec3) -> Vec3:
        """
        Convert a local position to world position.

        Args:
            local_position: Position in the sub-level's local space.

        Returns:
            Position in world space.
        """
        # Simple offset transform (full rotation would need quaternion math)
        return local_position + self.relative_offset

    def get_local_position(self, world_position: Vec3) -> Vec3:
        """
        Convert a world position to local position.

        Args:
            world_position: Position in world space.

        Returns:
            Position in the sub-level's local space.
        """
        return world_position - self.relative_offset

    def load(self) -> bool:
        """Load the sub-level content."""
        if self.inherit_layers and self.parent:
            # Inherit layer enabled states from parent (override defaults with parent values)
            for layer, enabled in self.parent.layers_enabled.items():
                self.layers_enabled[layer] = enabled

        return super().load()


@dataclass
class LevelInstance(Level):
    """
    A runtime instance of a level template.

    Multiple instances can be created from the same source level,
    each with their own runtime state but sharing the template data.
    """
    source_level: Optional[Level] = None
    instance_id: str = ""
    instance_transform: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    runtime_state: Dict[str, Any] = field(default_factory=dict)

    # Instance-specific modifications
    modified_actors: Set[str] = field(default_factory=set)
    spawned_actors: List[Actor] = field(default_factory=list)
    destroyed_actors: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Ensure level type is set correctly."""
        self.level_type = LevelType.INSTANCE
        if not self.instance_id:
            import uuid
            self.instance_id = str(uuid.uuid4())

    @classmethod
    def from_source(cls, source: Level, instance_id: str = "") -> "LevelInstance":
        """
        Create an instance from a source level.

        Args:
            source: The template level to instance.
            instance_id: Optional ID for the instance.

        Returns:
            A new LevelInstance.
        """
        instance = cls(
            name=f"{source.name}_instance",
            source_level=source,
            instance_id=instance_id,
            bounds=LevelBounds(
                min_point=Vec3(
                    source.bounds.min_point.x,
                    source.bounds.min_point.y,
                    source.bounds.min_point.z,
                ),
                max_point=Vec3(
                    source.bounds.max_point.x,
                    source.bounds.max_point.y,
                    source.bounds.max_point.z,
                ),
            ),
            layers_enabled=dict(source.layers_enabled),
            metadata=dict(source.metadata),
        )
        return instance

    def get_actors(self) -> List[Actor]:
        """
        Get all actors in the instance, combining source and runtime actors.

        Returns:
            List of all actors including source and spawned, excluding destroyed.
        """
        result = []

        # Add source actors (if loaded)
        if self.source_level:
            for actor in self.source_level.actors:
                if actor.id not in self.destroyed_actors:
                    result.append(actor)

        # Add instance-specific spawned actors
        result.extend(self.spawned_actors)

        return result

    def spawn_actor(self, actor: Actor) -> None:
        """Spawn a new actor in this instance."""
        self.spawned_actors.append(actor)
        self.bounds.expand(actor.transform_position)

    def destroy_actor(self, actor_id: str) -> bool:
        """Mark an actor as destroyed in this instance."""
        self.destroyed_actors.add(actor_id)
        return True

    def modify_actor(self, actor_id: str) -> None:
        """Mark an actor as modified in this instance."""
        self.modified_actors.add(actor_id)

    def reset_instance(self) -> None:
        """Reset the instance to match the source level state."""
        self.spawned_actors.clear()
        self.destroyed_actors.clear()
        self.modified_actors.clear()
        self.runtime_state.clear()

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a runtime state value."""
        return self.runtime_state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set a runtime state value."""
        self.runtime_state[key] = value


@dataclass
class WorldComposition:
    """
    Manages multiple levels in a world, handling composition,
    origin rebasing, and tile-based streaming.

    Provides the top-level container for organizing levels in an open world.
    """
    name: str = ""
    levels: List[Level] = field(default_factory=list)
    streaming_levels: List[StreamingLevel] = field(default_factory=list)
    sub_levels: List[SubLevel] = field(default_factory=list)

    # World origin for origin rebasing
    world_origin: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    origin_shift_threshold: float = DEFAULT_ORIGIN_SHIFT_THRESHOLD

    # Tile settings for large worlds
    tile_size: float = DEFAULT_TILE_SIZE
    tile_overlap: float = DEFAULT_TILE_OVERLAP

    # Registered levels by name for fast lookup
    _level_registry: Dict[str, Level] = field(default_factory=dict, repr=False)

    def add_level(self, level: Level) -> None:
        """Add a level to the world composition."""
        self.levels.append(level)
        self._level_registry[level.name] = level

        if isinstance(level, StreamingLevel):
            self.streaming_levels.append(level)
        elif isinstance(level, SubLevel):
            self.sub_levels.append(level)

    def remove_level(self, level: Level) -> bool:
        """Remove a level from the world composition."""
        if level not in self.levels:
            return False

        self.levels.remove(level)
        self._level_registry.pop(level.name, None)

        if isinstance(level, StreamingLevel) and level in self.streaming_levels:
            self.streaming_levels.remove(level)
        elif isinstance(level, SubLevel) and level in self.sub_levels:
            self.sub_levels.remove(level)

        return True

    def get_level(self, name: str) -> Optional[Level]:
        """Get a level by name."""
        return self._level_registry.get(name)

    def get_levels_at_position(self, position: Vec3) -> List[Level]:
        """Get all levels that contain a world position."""
        return [
            level for level in self.levels
            if level.bounds.contains_point(position)
        ]

    def get_loaded_levels(self) -> List[Level]:
        """Get all currently loaded levels."""
        return [
            level for level in self.levels
            if level.state in (LevelState.LOADED, LevelState.VISIBLE)
        ]

    def get_visible_levels(self) -> List[Level]:
        """Get all currently visible levels."""
        return [
            level for level in self.levels
            if level.state == LevelState.VISIBLE
        ]

    def update_streaming(self, source_position: Vec3) -> Tuple[List[StreamingLevel], List[StreamingLevel]]:
        """
        Update streaming levels based on source position.

        Args:
            source_position: Position of the streaming source.

        Returns:
            Tuple of (levels_to_load, levels_to_unload).
        """
        to_load = []
        to_unload = []

        for level in self.streaming_levels:
            if level.should_load(source_position):
                to_load.append(level)
            elif level.should_unload(source_position):
                to_unload.append(level)

        # Sort by priority (higher priority first for loading)
        to_load.sort(key=lambda l: l.priority, reverse=True)

        return to_load, to_unload

    def check_origin_rebase_needed(self, source_position: Vec3) -> bool:
        """
        Check if origin rebasing is needed.

        Args:
            source_position: Current position of the primary actor.

        Returns:
            True if rebasing should occur.
        """
        distance = source_position.distance(self.world_origin)
        return distance > self.origin_shift_threshold

    def perform_origin_rebase(self, new_origin: Vec3) -> Vec3:
        """
        Shift the world origin to prevent floating point precision issues.

        Args:
            new_origin: The new world origin point.

        Returns:
            The delta offset applied.
        """
        delta = new_origin - self.world_origin
        old_origin = self.world_origin

        # Update world origin
        self.world_origin = new_origin

        # Shift all loaded actors
        for level in self.get_loaded_levels():
            for actor in level.actors:
                actor.transform_position = actor.transform_position - delta

        # Shift streaming level reference points
        for level in self.streaming_levels:
            level.reference_point = level.reference_point - delta

        return delta

    def get_tile_at_position(self, position: Vec3) -> Tuple[int, int]:
        """
        Get the tile coordinates for a world position.

        Args:
            position: World position.

        Returns:
            Tuple of (tile_x, tile_y).
        """
        tile_x = int((position.x - self.world_origin.x) // self.tile_size)
        tile_y = int((position.z - self.world_origin.z) // self.tile_size)
        return tile_x, tile_y

    def get_tile_bounds(self, tile_x: int, tile_y: int) -> LevelBounds:
        """
        Get the bounds for a specific tile.

        Args:
            tile_x: Tile X coordinate.
            tile_y: Tile Y coordinate.

        Returns:
            The bounds of the tile.
        """
        min_x = self.world_origin.x + tile_x * self.tile_size
        min_z = self.world_origin.z + tile_y * self.tile_size

        return LevelBounds(
            min_point=Vec3(min_x, TILE_VERTICAL_MIN, min_z),
            max_point=Vec3(min_x + self.tile_size, TILE_VERTICAL_MAX, min_z + self.tile_size),
        )

    def get_surrounding_tiles(self, center_tile: Tuple[int, int], radius: int = 1) -> List[Tuple[int, int]]:
        """
        Get all tiles within a radius of a center tile.

        Args:
            center_tile: The center tile coordinates.
            radius: Number of tiles in each direction.

        Returns:
            List of tile coordinates.
        """
        tiles = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                tiles.append((center_tile[0] + dx, center_tile[1] + dy))
        return tiles

    def get_world_bounds(self) -> LevelBounds:
        """Get the combined bounds of all levels."""
        if not self.levels:
            return LevelBounds()

        result = LevelBounds(
            min_point=Vec3(
                self.levels[0].bounds.min_point.x,
                self.levels[0].bounds.min_point.y,
                self.levels[0].bounds.min_point.z,
            ),
            max_point=Vec3(
                self.levels[0].bounds.max_point.x,
                self.levels[0].bounds.max_point.y,
                self.levels[0].bounds.max_point.z,
            ),
        )

        for level in self.levels[1:]:
            result = result.merge(level.bounds)

        return result

    def __iter__(self) -> Iterator[Level]:
        """Iterate over all levels."""
        return iter(self.levels)

    def __len__(self) -> int:
        """Get the number of levels."""
        return len(self.levels)
