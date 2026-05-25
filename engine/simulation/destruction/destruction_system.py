"""
Destruction System.

Main coordinator for all destruction-related functionality including
damage processing, fracture triggering, debris spawning, and support
structure management.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Callable, Any, Protocol
from enum import IntEnum, auto
from collections import deque

from .config import (
    DestructionSystemConfig,
    DEFAULT_CONFIG,
    FracturePattern,
    MIN_CHUNK_VOLUME,
    MAX_CHUNKS_PER_OBJECT,
    DAMAGE_PROPAGATION_FACTOR,
    DEBRIS_LIFETIME,
    MAX_ACTIVE_DEBRIS,
    FRACTURE_BATCH_SIZE,
    FRACTURE_VELOCITY_MULTIPLIER,
    FRACTURE_SPREAD_MULTIPLIER,
)
from .damage_types import (
    DamageType,
    Damage,
    DamageResistance,
    DamageAccumulator,
    DamageResult,
    apply_damage_modifiers,
    get_damage_type_properties,
)
from .fracture_voronoi import (
    Vec3,
    Triangle,
    BoundingBox,
    Chunk,
    VoronoiFracture,
    SiteDistribution,
    vec3_add,
    vec3_sub,
    vec3_mul,
    vec3_length,
    vec3_normalize,
)
from .fracture_radial import RadialFracture
from .fracture_slice import SliceFracture
from .support_graph import (
    SupportGraph,
    SupportNode,
    SupportEdge,
    build_support_graph_from_chunks,
)
from .debris import (
    Debris,
    DebrisManager,
    DebrisSpawnParams,
    spawn_debris_from_fracture,
)


class DestructibleState(IntEnum):
    """State of a destructible object."""
    INTACT = 0
    DAMAGED = auto()
    FRACTURED = auto()
    DESTROYED = auto()


class PhysicsBodyProtocol(Protocol):
    """Protocol for physics body interaction."""

    @property
    def id(self) -> int: ...

    @property
    def position(self) -> Vec3: ...

    @property
    def velocity(self) -> Vec3: ...

    @property
    def mass(self) -> float: ...


@dataclass(slots=True)
class Destructible:
    """
    Represents a destructible object registered with the system.

    Attributes:
        id: Unique identifier.
        body_id: Physics body ID.
        vertices: Mesh vertex positions.
        triangles: Mesh triangle indices.
        health: Maximum health.
        damage_accumulator: Tracks accumulated damage.
        resistance: Damage resistance configuration.
        fracture_pattern: Pattern to use for fracturing.
        fracture_depth: Maximum recursive fracture depth.
        debris_lifetime: Lifetime for spawned debris.
        state: Current destruction state.
        chunks: Fractured chunks (after fracturing).
        support_graph: Support structure (after fracturing).
        current_generation: Current fracture generation.
        on_damage: Callback when damage is applied.
        on_fracture: Callback when object fractures.
        on_destroy: Callback when object is destroyed.
    """
    id: int
    body_id: Optional[int] = None
    vertices: List[Vec3] = field(default_factory=list)
    triangles: List[Triangle] = field(default_factory=list)
    health: float = 100.0
    damage_accumulator: DamageAccumulator = field(default_factory=DamageAccumulator)
    resistance: DamageResistance = field(default_factory=DamageResistance)
    fracture_pattern: FracturePattern = FracturePattern.VORONOI
    fracture_depth: int = 2
    debris_lifetime: float = DEBRIS_LIFETIME
    state: DestructibleState = DestructibleState.INTACT
    chunks: List[Chunk] = field(default_factory=list)
    support_graph: Optional[SupportGraph] = None
    current_generation: int = 0
    on_damage: Optional[Callable[[Damage, float], None]] = None
    on_fracture: Optional[Callable[[List[Chunk]], None]] = None
    on_destroy: Optional[Callable[[], None]] = None

    def __post_init__(self) -> None:
        # Initialize damage accumulator with health as threshold
        if self.damage_accumulator.threshold != self.health:
            self.damage_accumulator = DamageAccumulator(
                threshold=self.health,
                decay_rate=self.damage_accumulator._decay_rate
            )


@dataclass(slots=True)
class FractureRequest:
    """
    Request to fracture a destructible object.

    Attributes:
        destructible_id: ID of the destructible.
        impact_point: Point of impact.
        impact_direction: Direction of impact.
        intensity: Fracture intensity (affects chunk count).
        pattern_override: Optional pattern override.
    """
    destructible_id: int
    impact_point: Vec3
    impact_direction: Vec3
    intensity: float = 1.0
    pattern_override: Optional[FracturePattern] = None


@dataclass(slots=True)
class DamageEvent:
    """
    Event generated when damage is applied.

    Attributes:
        destructible_id: ID of the damaged object.
        damage: The damage that was applied.
        final_amount: Final damage after modifiers.
        remaining_health: Health remaining after damage.
        caused_fracture: Whether damage triggered fracture.
    """
    destructible_id: int
    damage: Damage
    final_amount: float
    remaining_health: float
    caused_fracture: bool = False


@dataclass(slots=True)
class FractureEvent:
    """
    Event generated when an object fractures.

    Attributes:
        destructible_id: ID of the fractured object.
        chunks: Generated chunks.
        impact_point: Point of fracture.
        debris_ids: IDs of spawned debris.
    """
    destructible_id: int
    chunks: List[Chunk]
    impact_point: Vec3
    debris_ids: List[int] = field(default_factory=list)


class DestructionSystem:
    """
    Main destruction system coordinator.

    Manages destructible objects, processes damage, triggers fractures,
    and coordinates debris spawning.
    """

    __slots__ = (
        '_config', '_destructibles', '_next_id', '_voronoi_fracture',
        '_radial_fracture', '_slice_fracture', '_debris_manager',
        '_pending_fractures', '_pending_damage', '_damage_events',
        '_fracture_events', '_current_time', '_physics_callback',
        '_fractures_this_frame', '_custom_fracture_callback'
    )

    def __init__(self, config: Optional[DestructionSystemConfig] = None) -> None:
        """
        Initialize destruction system.

        Args:
            config: System configuration.
        """
        self._config = config or DEFAULT_CONFIG

        self._destructibles: Dict[int, Destructible] = {}
        self._next_id = 0

        # Fracture generators
        self._voronoi_fracture = VoronoiFracture(
            seed=self._config.fracture.seed,
            num_sites=self._config.fracture.num_sites,
            min_chunk_volume=self._config.fracture.min_chunk_volume,
            max_chunks=self._config.fracture.max_chunks
        )
        self._radial_fracture = RadialFracture(
            seed=self._config.fracture.seed,
            num_slices=self._config.fracture.num_slices,
            num_rings=self._config.fracture.num_rings,
            min_chunk_volume=self._config.fracture.min_chunk_volume,
            max_chunks=self._config.fracture.max_chunks
        )
        self._slice_fracture = SliceFracture(
            seed=self._config.fracture.seed,
            min_chunk_volume=self._config.fracture.min_chunk_volume,
            max_chunks=self._config.fracture.max_chunks
        )

        # Debris manager
        self._debris_manager = DebrisManager(
            max_active=self._config.debris.max_active,
            pool_size=self._config.debris.pool_size,
            merge_distance=self._config.debris.merge_distance,
            sleep_velocity=self._config.debris.sleep_velocity,
            sleep_time=self._config.debris.sleep_time
        )

        # Queues
        self._pending_fractures: deque[FractureRequest] = deque()
        self._pending_damage: deque[Tuple[int, Damage]] = deque()
        self._damage_events: List[DamageEvent] = []
        self._fracture_events: List[FractureEvent] = []

        self._current_time = 0.0
        self._physics_callback: Optional[Callable[[Debris], int]] = None
        self._fractures_this_frame = 0
        self._custom_fracture_callback: Optional[Callable[
            [List[Vec3], List[Triangle], Vec3, Vec3, float], List[Chunk]
        ]] = None

    @property
    def config(self) -> DestructionSystemConfig:
        """System configuration."""
        return self._config

    @property
    def destructibles(self) -> Dict[int, Destructible]:
        """All registered destructibles."""
        return self._destructibles

    @property
    def debris_manager(self) -> DebrisManager:
        """Debris manager instance."""
        return self._debris_manager

    @property
    def damage_events(self) -> List[DamageEvent]:
        """Damage events from last update."""
        return self._damage_events

    @property
    def fracture_events(self) -> List[FractureEvent]:
        """Fracture events from last update."""
        return self._fracture_events

    def set_physics_callback(
        self,
        callback: Callable[[Debris], int]
    ) -> None:
        """
        Set callback for creating physics bodies for debris.

        Args:
            callback: Function that takes Debris and returns body ID.
        """
        self._physics_callback = callback

    def set_custom_fracture_callback(
        self,
        callback: Callable[[List[Vec3], List[Triangle], Vec3, Vec3, float], List[Chunk]]
    ) -> None:
        """
        Set callback for custom fracture pattern generation.

        This callback is used when FracturePattern.CUSTOM is selected.

        Args:
            callback: Function that takes (vertices, triangles, impact_point,
                      impact_direction, intensity) and returns List[Chunk].
        """
        self._custom_fracture_callback = callback

    def register_destructible(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        health: float = 100.0,
        resistance: Optional[DamageResistance] = None,
        fracture_pattern: FracturePattern = FracturePattern.VORONOI,
        fracture_depth: int = 2,
        debris_lifetime: float = DEBRIS_LIFETIME,
        body_id: Optional[int] = None
    ) -> int:
        """
        Register a destructible object with the system.

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            health: Maximum health.
            resistance: Damage resistance configuration.
            fracture_pattern: Pattern to use for fracturing.
            fracture_depth: Maximum recursive fracture depth.
            debris_lifetime: Lifetime for spawned debris.
            body_id: Optional physics body ID.

        Returns:
            ID of the registered destructible.
        """
        destructible_id = self._next_id
        self._next_id += 1

        destructible = Destructible(
            id=destructible_id,
            body_id=body_id,
            vertices=list(vertices),
            triangles=list(triangles),
            health=health,
            damage_accumulator=DamageAccumulator(threshold=health),
            resistance=resistance or DamageResistance(),
            fracture_pattern=fracture_pattern,
            fracture_depth=fracture_depth,
            debris_lifetime=debris_lifetime
        )

        self._destructibles[destructible_id] = destructible
        return destructible_id

    def unregister_destructible(self, destructible_id: int) -> bool:
        """
        Unregister a destructible object.

        Args:
            destructible_id: ID of the destructible.

        Returns:
            True if object was found and removed.
        """
        if destructible_id in self._destructibles:
            destructible = self._destructibles[destructible_id]

            # Clean up debris from this destructible
            self._debris_manager.destroy_by_parent(destructible_id)

            del self._destructibles[destructible_id]
            return True

        return False

    def apply_damage(
        self,
        destructible_id: int,
        damage: Damage,
        immediate: bool = False
    ) -> Optional[DamageResult]:
        """
        Apply damage to a destructible object.

        Args:
            destructible_id: ID of the target.
            damage: Damage to apply.
            immediate: Process immediately vs queue for next update.

        Returns:
            DamageResult if processed immediately, None if queued.
        """
        if destructible_id not in self._destructibles:
            return None

        if not immediate:
            self._pending_damage.append((destructible_id, damage))
            return None

        return self._process_damage(destructible_id, damage)

    def _process_damage(
        self,
        destructible_id: int,
        damage: Damage
    ) -> DamageResult:
        """Process damage immediately."""
        destructible = self._destructibles[destructible_id]

        # Skip if already destroyed
        if destructible.state == DestructibleState.DESTROYED:
            return DamageResult(
                original_amount=damage.amount,
                final_amount=0.0,
                damage_type=damage.damage_type,
                was_resisted=True
            )

        # Apply modifiers
        final_amount = apply_damage_modifiers(damage, destructible.resistance)

        # Check minimum threshold
        if final_amount < self._config.damage.min_threshold:
            return DamageResult(
                original_amount=damage.amount,
                final_amount=0.0,
                damage_type=damage.damage_type,
                was_resisted=True
            )

        # Accumulate damage
        destructible.damage_accumulator.accumulate(final_amount, damage.damage_type)

        # Update state
        if destructible.state == DestructibleState.INTACT:
            destructible.state = DestructibleState.DAMAGED

        # Check for fracture
        caused_fracture = False
        properties = get_damage_type_properties(damage.damage_type)

        if properties.causes_fracture:
            total_damage = destructible.damage_accumulator.total_damage
            if total_damage >= properties.fracture_threshold:
                # Queue fracture
                self._pending_fractures.append(FractureRequest(
                    destructible_id=destructible_id,
                    impact_point=damage.position,
                    impact_direction=damage.direction,
                    intensity=min(1.0, final_amount / properties.fracture_threshold)
                ))
                caused_fracture = True

        # Check for destruction
        was_lethal = destructible.damage_accumulator.is_destroyed

        if was_lethal:
            destructible.state = DestructibleState.DESTROYED
            if destructible.on_destroy:
                destructible.on_destroy()

        # Create event
        event = DamageEvent(
            destructible_id=destructible_id,
            damage=damage,
            final_amount=final_amount,
            remaining_health=destructible.damage_accumulator.remaining_health,
            caused_fracture=caused_fracture
        )
        self._damage_events.append(event)

        # Callback
        if destructible.on_damage:
            destructible.on_damage(damage, final_amount)

        return DamageResult(
            original_amount=damage.amount,
            final_amount=final_amount,
            damage_type=damage.damage_type,
            was_resisted=final_amount < damage.amount,
            was_lethal=was_lethal,
            caused_fracture=caused_fracture
        )

    def trigger_fracture(
        self,
        destructible_id: int,
        impact_point: Vec3,
        impact_direction: Vec3,
        intensity: float = 1.0,
        pattern_override: Optional[FracturePattern] = None,
        immediate: bool = False
    ) -> Optional[List[Chunk]]:
        """
        Trigger fracturing of a destructible object.

        Args:
            destructible_id: ID of the target.
            impact_point: Point of impact.
            impact_direction: Direction of impact.
            intensity: Fracture intensity.
            pattern_override: Optional pattern override.
            immediate: Process immediately vs queue.

        Returns:
            List of chunks if processed immediately, None if queued.
        """
        if destructible_id not in self._destructibles:
            return None

        request = FractureRequest(
            destructible_id=destructible_id,
            impact_point=impact_point,
            impact_direction=impact_direction,
            intensity=intensity,
            pattern_override=pattern_override
        )

        if not immediate:
            self._pending_fractures.append(request)
            return None

        return self._process_fracture(request)

    def _process_fracture(self, request: FractureRequest) -> List[Chunk]:
        """Process a fracture request."""
        destructible = self._destructibles.get(request.destructible_id)
        if destructible is None:
            return []

        # Check fracture depth
        if destructible.current_generation >= destructible.fracture_depth:
            return []

        # Select fracture pattern
        pattern = request.pattern_override or destructible.fracture_pattern

        # Generate chunks
        if pattern == FracturePattern.VORONOI:
            chunks = self._voronoi_fracture.fracture(
                destructible.vertices,
                destructible.triangles,
                impact_point=request.impact_point,
                distribution=SiteDistribution.IMPACT_CENTERED
            )
        elif pattern == FracturePattern.RADIAL:
            chunks = self._radial_fracture.fracture_mesh(
                destructible.vertices,
                destructible.triangles,
                center=request.impact_point,
                direction=request.impact_direction
            )
        elif pattern == FracturePattern.SLICE:
            bounds = BoundingBox.from_points(destructible.vertices)
            planes = self._slice_fracture.random_slice_planes(
                bounds,
                num_planes=max(2, int(4 * request.intensity)),
                bias_direction=request.impact_direction,
                bias_strength=0.5
            )
            chunks = self._slice_fracture.multi_slice(
                destructible.vertices,
                destructible.triangles,
                planes
            )
        else:
            # Custom pattern - use callback if registered, otherwise fallback to Voronoi
            if self._custom_fracture_callback is not None:
                chunks = self._custom_fracture_callback(
                    destructible.vertices,
                    destructible.triangles,
                    request.impact_point,
                    request.impact_direction,
                    request.intensity
                )
            else:
                # Fallback to Voronoi if no custom callback registered
                chunks = self._voronoi_fracture.fracture(
                    destructible.vertices,
                    destructible.triangles,
                    impact_point=request.impact_point
                )

        if not chunks:
            return []

        # Update destructible state
        destructible.state = DestructibleState.FRACTURED
        destructible.chunks = chunks
        destructible.current_generation += 1

        # Build support graph
        destructible.support_graph = build_support_graph_from_chunks(chunks)

        # Spawn debris
        debris_ids = []
        if self._debris_manager:
            debris_list = spawn_debris_from_fracture(
                self._debris_manager,
                chunks,
                center_velocity=vec3_mul(
                    vec3_normalize(request.impact_direction),
                    request.intensity * FRACTURE_VELOCITY_MULTIPLIER
                ),
                spread_factor=request.intensity * FRACTURE_SPREAD_MULTIPLIER,
                parent_id=request.destructible_id,
                generation=destructible.current_generation,
                lifetime=destructible.debris_lifetime
            )
            debris_ids = [d.id for d in debris_list]

            # Create physics bodies for debris if callback set
            if self._physics_callback:
                for debris in debris_list:
                    debris.body_id = self._physics_callback(debris)

        # Create event
        event = FractureEvent(
            destructible_id=request.destructible_id,
            chunks=chunks,
            impact_point=request.impact_point,
            debris_ids=debris_ids
        )
        self._fracture_events.append(event)

        # Callback
        if destructible.on_fracture:
            destructible.on_fracture(chunks)

        return chunks

    def update(
        self,
        dt: float,
        camera_position: Optional[Vec3] = None
    ) -> None:
        """
        Update the destruction system.

        Args:
            dt: Delta time since last update.
            camera_position: Camera position for LOD calculations.
        """
        self._current_time += dt
        self._damage_events.clear()
        self._fracture_events.clear()
        self._fractures_this_frame = 0

        # Process pending damage
        while self._pending_damage:
            destructible_id, damage = self._pending_damage.popleft()
            self._process_damage(destructible_id, damage)

        # Process pending fractures (limited per frame)
        while self._pending_fractures and self._fractures_this_frame < FRACTURE_BATCH_SIZE:
            request = self._pending_fractures.popleft()
            self._process_fracture(request)
            self._fractures_this_frame += 1

        # Update debris
        self._debris_manager.update(dt, camera_position)

        # Update damage decay
        for destructible in self._destructibles.values():
            destructible.damage_accumulator.update(self._current_time)

    def apply_area_damage(
        self,
        center: Vec3,
        radius: float,
        damage: Damage,
        falloff: str = "linear"
    ) -> List[DamageResult]:
        """
        Apply area damage to all destructibles in range.

        Args:
            center: Center of the area.
            radius: Damage radius.
            damage: Base damage to apply.
            falloff: Falloff type ("linear", "quadratic", "none").

        Returns:
            List of damage results.
        """
        results = []

        # Create area damage with falloff
        area_damage = Damage(
            amount=damage.amount,
            damage_type=damage.damage_type,
            position=center,
            direction=damage.direction,
            source_id=damage.source_id,
            impulse=damage.impulse,
            radius=radius,
            falloff=falloff,
            timestamp=self._current_time
        )

        for destructible_id, destructible in self._destructibles.items():
            # Get center of destructible
            bounds = BoundingBox.from_points(destructible.vertices)
            target_center = bounds.center

            # Calculate distance
            distance = vec3_length(vec3_sub(target_center, center))

            if distance <= radius:
                # Apply damage with falloff
                falloff_damage = area_damage.with_falloff(distance)

                # Direction from center to target
                if distance > 0:
                    direction = vec3_normalize(vec3_sub(target_center, center))
                    falloff_damage.direction = direction

                result = self.apply_damage(destructible_id, falloff_damage, immediate=True)
                if result:
                    results.append(result)

        return results

    def get_destructible(self, destructible_id: int) -> Optional[Destructible]:
        """Get a destructible by ID."""
        return self._destructibles.get(destructible_id)

    def get_destructibles_in_radius(
        self,
        center: Vec3,
        radius: float
    ) -> List[int]:
        """
        Get all destructible IDs within a radius.

        Args:
            center: Search center.
            radius: Search radius.

        Returns:
            List of destructible IDs.
        """
        radius_sq = radius ** 2
        result = []

        for destructible_id, destructible in self._destructibles.items():
            bounds = BoundingBox.from_points(destructible.vertices)
            target_center = bounds.center

            dist_sq = (
                (target_center[0] - center[0]) ** 2 +
                (target_center[1] - center[1]) ** 2 +
                (target_center[2] - center[2]) ** 2
            )

            if dist_sq <= radius_sq:
                result.append(destructible_id)

        return result

    def propagate_support_damage(
        self,
        destructible_id: int,
        start_chunk_index: int,
        damage_amount: float,
        direction: Optional[Vec3] = None
    ) -> List[int]:
        """
        Propagate damage through a destructible's support graph.

        Args:
            destructible_id: ID of the destructible.
            start_chunk_index: Starting chunk for propagation.
            damage_amount: Amount of damage to propagate.
            direction: Direction of damage propagation.

        Returns:
            List of chunk indices that became unsupported.
        """
        destructible = self._destructibles.get(destructible_id)
        if destructible is None or destructible.support_graph is None:
            return []

        # Propagate damage through support graph
        broken = destructible.support_graph.propagate_damage(
            start_chunk_index,
            damage_amount,
            direction
        )

        if broken:
            # Recompute support and find falling chunks
            destructible.support_graph.compute_stress_paths()
            unsupported = destructible.support_graph.detect_unsupported()
            return unsupported

        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        state_counts = {state: 0 for state in DestructibleState}
        total_chunks = 0

        for destructible in self._destructibles.values():
            state_counts[destructible.state] += 1
            total_chunks += len(destructible.chunks)

        debris_stats = self._debris_manager.get_stats()

        return {
            'destructible_count': len(self._destructibles),
            'state_counts': {state.name: count for state, count in state_counts.items()},
            'total_chunks': total_chunks,
            'pending_fractures': len(self._pending_fractures),
            'pending_damage': len(self._pending_damage),
            'debris': debris_stats
        }

    def clear(self) -> None:
        """Clear all destructibles and debris."""
        self._destructibles.clear()
        self._pending_fractures.clear()
        self._pending_damage.clear()
        self._damage_events.clear()
        self._fracture_events.clear()
        self._debris_manager.destroy_all()
        self._next_id = 0
