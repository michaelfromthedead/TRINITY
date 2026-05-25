"""
Destructible Component.

Provides destruction physics component for breakable objects,
fracture patterns, debris spawning, and damage propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..character.character_controller import Vector3


class DestructionType(str, Enum):
    """Type of destruction behavior."""
    NONE = "none"
    FRACTURE = "fracture"     # Pre-fractured chunks
    VORONOI = "voronoi"       # Runtime Voronoi fracture
    SLICE = "slice"           # Cut along plane
    CRUMBLE = "crumble"       # Gradual collapse
    SHATTER = "shatter"       # Glass-like shatter


class DamageType(str, Enum):
    """Type of damage affecting destruction."""
    IMPACT = "impact"
    EXPLOSION = "explosion"
    FIRE = "fire"
    BULLET = "bullet"
    MELEE = "melee"
    GENERIC = "generic"


@dataclass
class FractureChunk:
    """
    A chunk of fractured geometry.

    Attributes:
        chunk_id: Unique identifier
        vertices: Mesh vertices
        indices: Mesh indices
        mass: Chunk mass
        center_of_mass: Local center of mass
        connected_to: IDs of connected chunks
        is_detached: Whether chunk is detached
    """
    chunk_id: int = 0
    vertices: list[Vector3] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    mass: float = 1.0
    center_of_mass: Vector3 = field(default_factory=Vector3.zero)
    connected_to: list[int] = field(default_factory=list)
    is_detached: bool = False
    body_id: Optional[int] = None


@dataclass
class DestructionConfig:
    """
    Configuration for destruction behavior.

    Attributes:
        destruction_type: Type of destruction
        health: Total health before destruction
        min_damage_threshold: Minimum damage to register
        debris_lifetime: How long debris persists
        debris_count_limit: Maximum debris pieces
        propagate_damage: Whether damage spreads
        propagation_factor: How much damage spreads
    """
    destruction_type: DestructionType = DestructionType.FRACTURE
    health: float = 100.0
    min_damage_threshold: float = 5.0
    debris_lifetime: float = 10.0
    debris_count_limit: int = 50
    propagate_damage: bool = True
    propagation_factor: float = 0.5
    fracture_seed: int = 12345


@dataclass
class DamageInfo:
    """
    Information about damage applied.

    Attributes:
        amount: Damage amount
        damage_type: Type of damage
        position: World position of damage
        direction: Direction of force
        impulse: Impulse magnitude
    """
    amount: float = 0.0
    damage_type: DamageType = DamageType.GENERIC
    position: Vector3 = field(default_factory=Vector3.zero)
    direction: Vector3 = field(default_factory=Vector3.zero)
    impulse: float = 0.0


class DestructibleComponent:
    """
    Component for destructible objects.

    Provides:
    - Damage tracking and health
    - Fracture into chunks
    - Debris physics
    - Damage propagation
    - Staged destruction
    """

    def __init__(
        self,
        entity_id: int,
        config: Optional[DestructionConfig] = None,
    ):
        self._entity_id = entity_id
        self._config = config or DestructionConfig()

        # Health state
        self._current_health = self._config.health
        self._max_health = self._config.health
        self._is_destroyed = False

        # Fracture data
        self._chunks: list[FractureChunk] = []
        self._detached_chunks: list[int] = []

        # Staged destruction
        self._destruction_stage = 0
        self._stage_thresholds: list[float] = [0.75, 0.5, 0.25, 0.0]  # Health ratios

        # Damage tracking
        self._total_damage_taken = 0.0
        self._damage_sources: list[DamageInfo] = []

        # State
        self._enabled = True
        self._initialized = False

        # Callbacks
        self._on_damage: Optional[Callable[[DamageInfo], None]] = None
        self._on_stage_change: Optional[Callable[[int], None]] = None
        self._on_destroyed: Optional[Callable[[], None]] = None
        self._on_chunk_detached: Optional[Callable[[FractureChunk], None]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def entity_id(self) -> int:
        """Entity this component belongs to."""
        return self._entity_id

    @property
    def config(self) -> DestructionConfig:
        """Destruction configuration."""
        return self._config

    @property
    def current_health(self) -> float:
        """Current health."""
        return self._current_health

    @property
    def max_health(self) -> float:
        """Maximum health."""
        return self._max_health

    @property
    def health_ratio(self) -> float:
        """Health as ratio (0-1)."""
        return self._current_health / self._max_health if self._max_health > 0 else 0

    @property
    def is_destroyed(self) -> bool:
        """Whether object is fully destroyed."""
        return self._is_destroyed

    @property
    def destruction_stage(self) -> int:
        """Current destruction stage."""
        return self._destruction_stage

    @property
    def chunk_count(self) -> int:
        """Number of fracture chunks."""
        return len(self._chunks)

    @property
    def detached_chunk_count(self) -> int:
        """Number of detached chunks."""
        return len(self._detached_chunks)

    @property
    def enabled(self) -> bool:
        """Whether destruction is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_damage_callback(
        self, callback: Optional[Callable[[DamageInfo], None]]
    ) -> None:
        """Set callback for damage events."""
        self._on_damage = callback

    def set_stage_change_callback(
        self, callback: Optional[Callable[[int], None]]
    ) -> None:
        """Set callback for stage changes."""
        self._on_stage_change = callback

    def set_destroyed_callback(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        """Set callback for destruction."""
        self._on_destroyed = callback

    def set_chunk_detached_callback(
        self, callback: Optional[Callable[[FractureChunk], None]]
    ) -> None:
        """Set callback for chunk detachment."""
        self._on_chunk_detached = callback

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def set_fracture_chunks(self, chunks: list[FractureChunk]) -> None:
        """Set pre-computed fracture chunks."""
        self._chunks = chunks
        self._initialized = True

    def set_stage_thresholds(self, thresholds: list[float]) -> None:
        """Set health thresholds for staged destruction."""
        self._stage_thresholds = sorted(thresholds, reverse=True)

    def generate_voronoi_chunks(
        self,
        mesh_vertices: list[Vector3],
        mesh_indices: list[int],
        num_chunks: int = 10,
    ) -> None:
        """
        Generate Voronoi fracture chunks.

        Args:
            mesh_vertices: Original mesh vertices
            mesh_indices: Original mesh indices
            num_chunks: Number of chunks to generate
        """
        import random
        random.seed(self._config.fracture_seed)

        # Generate seed points
        # Simplified - real implementation would use proper Voronoi
        self._chunks = []

        for i in range(num_chunks):
            chunk = FractureChunk(
                chunk_id=i,
                vertices=[],  # Would be computed
                indices=[],
                mass=1.0 / num_chunks,
                center_of_mass=Vector3(
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                ),
                connected_to=[],
            )

            # Connect to neighbors (simplified)
            if i > 0:
                chunk.connected_to.append(i - 1)
                self._chunks[i - 1].connected_to.append(i)

            self._chunks.append(chunk)

        self._initialized = True

    # -------------------------------------------------------------------------
    # Damage
    # -------------------------------------------------------------------------

    def apply_damage(self, damage_info: DamageInfo) -> bool:
        """
        Apply damage to the object.

        Args:
            damage_info: Damage information

        Returns:
            True if damage was applied
        """
        if not self._enabled or self._is_destroyed:
            return False

        # Check threshold
        if damage_info.amount < self._config.min_damage_threshold:
            return False

        # Apply damage
        actual_damage = damage_info.amount
        self._current_health = max(0, self._current_health - actual_damage)
        self._total_damage_taken += actual_damage
        self._damage_sources.append(damage_info)

        # Fire callback
        if self._on_damage:
            self._on_damage(damage_info)

        # Check for stage change
        self._check_stage_change()

        # Check for destruction
        if self._current_health <= 0:
            self._destroy(damage_info)

        # Propagate damage to nearby chunks
        if self._config.propagate_damage and self._chunks:
            self._propagate_damage(damage_info)

        return True

    def apply_explosion_damage(
        self,
        center: Vector3,
        radius: float,
        max_damage: float,
    ) -> None:
        """
        Apply explosion damage.

        Args:
            center: Explosion center
            radius: Explosion radius
            max_damage: Maximum damage at center
        """
        # Would calculate damage based on distance to center
        damage_info = DamageInfo(
            amount=max_damage,
            damage_type=DamageType.EXPLOSION,
            position=center,
            direction=Vector3.zero(),  # Radial
            impulse=max_damage * 10.0,
        )
        self.apply_damage(damage_info)

    def _check_stage_change(self) -> None:
        """Check and update destruction stage."""
        for i, threshold in enumerate(self._stage_thresholds):
            if self.health_ratio <= threshold and self._destruction_stage < i + 1:
                self._destruction_stage = i + 1

                if self._on_stage_change:
                    self._on_stage_change(self._destruction_stage)

                # Detach some chunks based on stage
                self._detach_chunks_for_stage(self._destruction_stage)
                break

    def _detach_chunks_for_stage(self, stage: int) -> None:
        """Detach chunks based on destruction stage."""
        if not self._chunks:
            return

        # Detach a portion of chunks
        chunks_to_detach = len(self._chunks) * stage // len(self._stage_thresholds)

        for chunk in self._chunks:
            if len(self._detached_chunks) >= chunks_to_detach:
                break

            if not chunk.is_detached and len(chunk.connected_to) == 1:
                self._detach_chunk(chunk)

    def _propagate_damage(self, damage_info: DamageInfo) -> None:
        """Propagate damage to connected chunks."""
        if not self._chunks:
            return

        # Find closest chunk to damage position
        closest_chunk: Optional[FractureChunk] = None
        closest_dist = float("inf")

        for chunk in self._chunks:
            if chunk.is_detached:
                continue
            dist = (chunk.center_of_mass - damage_info.position).magnitude()
            if dist < closest_dist:
                closest_dist = dist
                closest_chunk = chunk

        if closest_chunk is None:
            return

        # Propagate to connected chunks
        propagated_damage = damage_info.amount * self._config.propagation_factor

        for connected_id in closest_chunk.connected_to:
            if connected_id < len(self._chunks):
                connected_chunk = self._chunks[connected_id]
                if not connected_chunk.is_detached:
                    # Check if should detach
                    if propagated_damage > self._config.min_damage_threshold * 2:
                        self._detach_chunk(connected_chunk)

    # -------------------------------------------------------------------------
    # Destruction
    # -------------------------------------------------------------------------

    def _destroy(self, final_damage: DamageInfo) -> None:
        """Trigger full destruction."""
        self._is_destroyed = True

        # Detach all remaining chunks
        for chunk in self._chunks:
            if not chunk.is_detached:
                self._detach_chunk(chunk, impulse=final_damage.impulse)

        if self._on_destroyed:
            self._on_destroyed()

    def _detach_chunk(
        self,
        chunk: FractureChunk,
        impulse: float = 0.0,
    ) -> None:
        """Detach a single chunk."""
        if chunk.is_detached:
            return

        chunk.is_detached = True
        self._detached_chunks.append(chunk.chunk_id)

        # Remove from connections
        for connected_id in chunk.connected_to:
            if connected_id < len(self._chunks):
                other = self._chunks[connected_id]
                if chunk.chunk_id in other.connected_to:
                    other.connected_to.remove(chunk.chunk_id)

        chunk.connected_to.clear()

        # Callback
        if self._on_chunk_detached:
            self._on_chunk_detached(chunk)

    def force_destroy(self) -> None:
        """Force immediate destruction."""
        if not self._is_destroyed:
            damage = DamageInfo(
                amount=self._current_health + 1,
                damage_type=DamageType.GENERIC,
            )
            self.apply_damage(damage)

    # -------------------------------------------------------------------------
    # Repair
    # -------------------------------------------------------------------------

    def repair(self, amount: float) -> float:
        """
        Repair damage.

        Args:
            amount: Amount to repair

        Returns:
            Actual amount repaired
        """
        if self._is_destroyed:
            return 0.0

        old_health = self._current_health
        self._current_health = min(self._max_health, self._current_health + amount)
        return self._current_health - old_health

    def reset(self) -> None:
        """Reset to undamaged state."""
        self._current_health = self._max_health
        self._is_destroyed = False
        self._destruction_stage = 0
        self._total_damage_taken = 0.0
        self._damage_sources.clear()
        self._detached_chunks.clear()

        for chunk in self._chunks:
            chunk.is_detached = False
            # Would need to restore connections

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_chunk(self, chunk_id: int) -> Optional[FractureChunk]:
        """Get a chunk by ID."""
        if 0 <= chunk_id < len(self._chunks):
            return self._chunks[chunk_id]
        return None

    def get_attached_chunks(self) -> list[FractureChunk]:
        """Get all attached chunks."""
        return [c for c in self._chunks if not c.is_detached]

    def get_detached_chunks(self) -> list[FractureChunk]:
        """Get all detached chunks."""
        return [c for c in self._chunks if c.is_detached]

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def cleanup(self) -> None:
        """Cleanup component."""
        self._chunks.clear()
        self._detached_chunks.clear()
        self._damage_sources.clear()

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return {
            "entity_id": self._entity_id,
            "current_health": self._current_health,
            "max_health": self._max_health,
            "is_destroyed": self._is_destroyed,
            "destruction_stage": self._destruction_stage,
            "total_damage": self._total_damage_taken,
            "chunk_count": len(self._chunks),
            "detached_count": len(self._detached_chunks),
            "enabled": self._enabled,
        }


__all__ = [
    "DestructionType",
    "DamageType",
    "FractureChunk",
    "DestructionConfig",
    "DamageInfo",
    "DestructibleComponent",
]
