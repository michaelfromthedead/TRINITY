"""
Combat System - Death Module

Provides the DeathSystem for managing entity death, cleanup, and respawning:
- Death detection and state management
- Death event emission (EntityDied)
- Component cleanup triggers
- Respawn queue and timing
- Death animations and effects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Set
from enum import Enum, auto
import time

from .constants import (
    DeathState,
    DeathConfig,
    DEFAULT_DEATH_CONFIG,
    DEFAULT_RESPAWN_TIME,
    MIN_RESPAWN_TIME,
    MAX_RESPAWN_TIME,
    DYING_DURATION,
    RESPAWN_HEALTH_PERCENTAGE,
    RESPAWN_INVULNERABILITY_DURATION,
    CombatEventType,
)


# =============================================================================
# PROTOCOLS
# =============================================================================


class DeathSubject(Protocol):
    """Protocol for entities that can die."""

    @property
    def entity_id(self) -> int:
        """Get entity's unique ID."""
        ...

    @property
    def is_dead(self) -> bool:
        """Check if entity is dead."""
        ...

    def kill(self, source_id: Optional[int] = None) -> bool:
        """Kill the entity."""
        ...

    def revive(
        self,
        health_percentage: float = 1.0,
        source_id: Optional[int] = None,
        add_invulnerability: bool = True,
        invulnerability_duration: float = RESPAWN_INVULNERABILITY_DURATION,
    ) -> bool:
        """Revive the entity."""
        ...


class CleanupHandler(Protocol):
    """Protocol for components that need cleanup on death."""

    def on_entity_death(self, entity_id: int, death_info: "DeathInfo") -> None:
        """Handle entity death cleanup."""
        ...


class RespawnProvider(Protocol):
    """Protocol for providing respawn locations."""

    def get_respawn_position(self, entity_id: int, team_id: Optional[int] = None) -> Any:
        """Get respawn position for an entity."""
        ...


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class DeathInfo:
    """Information about an entity's death."""

    entity_id: int
    killer_id: Optional[int] = None
    death_state: DeathState = DeathState.DYING
    timestamp: float = field(default_factory=time.time)
    death_position: Any = None  # Vec3 or similar
    death_cause: str = "unknown"
    weapon_id: Optional[int] = None
    ability_id: Optional[int] = None
    was_headshot: bool = False
    was_critical: bool = False
    overkill_damage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_since_death(self) -> float:
        """Time elapsed since death."""
        return time.time() - self.timestamp

    @property
    def is_fully_dead(self) -> bool:
        """Check if entity has completed dying phase."""
        return self.death_state == DeathState.DEAD


@dataclass
class RespawnRequest:
    """Request to respawn an entity."""

    entity_id: int
    respawn_time: float  # When to respawn (absolute time)
    health_percentage: float = RESPAWN_HEALTH_PERCENTAGE
    position: Any = None  # Override spawn position
    team_id: Optional[int] = None
    add_invulnerability: bool = True
    invulnerability_duration: float = RESPAWN_INVULNERABILITY_DURATION
    source_id: Optional[int] = None  # Who requested the respawn
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_until_respawn(self) -> float:
        """Time remaining until respawn."""
        return max(0.0, self.respawn_time - time.time())

    @property
    def is_ready(self) -> bool:
        """Check if respawn time has been reached."""
        return time.time() >= self.respawn_time


@dataclass
class DeathEvent:
    """Event emitted when an entity dies."""

    death_info: DeathInfo
    event_type: CombatEventType = CombatEventType.DEATH


@dataclass
class RespawnEvent:
    """Event emitted when an entity respawns."""

    entity_id: int
    respawn_position: Any
    health_percentage: float
    team_id: Optional[int] = None
    event_type: CombatEventType = CombatEventType.RESPAWN


# =============================================================================
# DEATH SYSTEM
# =============================================================================


class DeathSystem:
    """
    System for managing entity death and respawning.

    Features:
    - Death state tracking (DYING -> DEAD -> RESPAWNING)
    - Configurable dying duration for animations
    - Component cleanup triggers
    - Respawn queue with configurable timing
    - Death and respawn event emission
    - Support for instant respawn or queued respawn
    """

    def __init__(
        self,
        config: Optional[DeathConfig] = None,
        respawn_provider: Optional[RespawnProvider] = None,
    ) -> None:
        """
        Initialize the death system.

        Args:
            config: Death configuration
            respawn_provider: Optional provider for respawn positions
        """
        self._config = config or DEFAULT_DEATH_CONFIG
        self._respawn_provider = respawn_provider

        # Active death states
        self._death_states: Dict[int, DeathInfo] = {}

        # Respawn queue
        self._respawn_queue: List[RespawnRequest] = []

        # Cleanup handlers
        self._cleanup_handlers: List[CleanupHandler] = []

        # Event handlers
        self._on_death: List[Callable[[DeathEvent], None]] = []
        self._on_respawn: List[Callable[[RespawnEvent], None]] = []
        self._on_state_changed: List[Callable[[int, DeathState, DeathState], None]] = []

        # Entities pending cleanup
        self._pending_cleanup: Set[int] = set()

    @property
    def config(self) -> DeathConfig:
        """Get death configuration."""
        return self._config

    # =========================================================================
    # DEATH HANDLING
    # =========================================================================

    def process_death(
        self,
        entity_id: int,
        killer_id: Optional[int] = None,
        death_cause: str = "unknown",
        death_position: Any = None,
        weapon_id: Optional[int] = None,
        ability_id: Optional[int] = None,
        was_headshot: bool = False,
        was_critical: bool = False,
        overkill_damage: float = 0.0,
        **metadata: Any,
    ) -> DeathInfo:
        """
        Process an entity death.

        Args:
            entity_id: ID of dying entity
            killer_id: ID of killing entity (if any)
            death_cause: Cause of death description
            death_position: Position of death
            weapon_id: Weapon used for kill
            ability_id: Ability used for kill
            was_headshot: Whether death was from headshot
            was_critical: Whether death was from critical hit
            overkill_damage: Damage beyond lethal
            **metadata: Additional death metadata

        Returns:
            DeathInfo for the death
        """
        # Create death info
        death_info = DeathInfo(
            entity_id=entity_id,
            killer_id=killer_id,
            death_state=DeathState.DYING,
            death_position=death_position,
            death_cause=death_cause,
            weapon_id=weapon_id,
            ability_id=ability_id,
            was_headshot=was_headshot,
            was_critical=was_critical,
            overkill_damage=overkill_damage,
            metadata=metadata,
        )

        # Store death state
        self._death_states[entity_id] = death_info

        # Emit death event
        self._emit_death(DeathEvent(death_info=death_info))

        # Emit state change
        self._emit_state_changed(entity_id, DeathState.ALIVE, DeathState.DYING)

        return death_info

    def get_death_info(self, entity_id: int) -> Optional[DeathInfo]:
        """Get death info for an entity."""
        return self._death_states.get(entity_id)

    def get_death_state(self, entity_id: int) -> DeathState:
        """Get current death state for an entity."""
        info = self._death_states.get(entity_id)
        return info.death_state if info else DeathState.ALIVE

    def is_dying(self, entity_id: int) -> bool:
        """Check if entity is in dying state."""
        return self.get_death_state(entity_id) == DeathState.DYING

    def is_dead(self, entity_id: int) -> bool:
        """Check if entity is fully dead."""
        return self.get_death_state(entity_id) == DeathState.DEAD

    def is_respawning(self, entity_id: int) -> bool:
        """Check if entity is respawning."""
        return self.get_death_state(entity_id) == DeathState.RESPAWNING

    # =========================================================================
    # STATE TRANSITIONS
    # =========================================================================

    def transition_to_dead(self, entity_id: int) -> bool:
        """
        Transition entity from DYING to DEAD state.

        Args:
            entity_id: Entity to transition

        Returns:
            True if transition occurred
        """
        info = self._death_states.get(entity_id)
        if not info or info.death_state != DeathState.DYING:
            return False

        old_state = info.death_state
        info.death_state = DeathState.DEAD

        # Mark for cleanup
        self._pending_cleanup.add(entity_id)

        # Emit state change
        self._emit_state_changed(entity_id, old_state, DeathState.DEAD)

        return True

    def transition_to_respawning(self, entity_id: int) -> bool:
        """
        Transition entity from DEAD to RESPAWNING state.

        Args:
            entity_id: Entity to transition

        Returns:
            True if transition occurred
        """
        info = self._death_states.get(entity_id)
        if not info or info.death_state != DeathState.DEAD:
            return False

        old_state = info.death_state
        info.death_state = DeathState.RESPAWNING

        # Emit state change
        self._emit_state_changed(entity_id, old_state, DeathState.RESPAWNING)

        return True

    def complete_respawn(self, entity_id: int) -> bool:
        """
        Complete respawn and clear death state.

        Args:
            entity_id: Entity that respawned

        Returns:
            True if respawn was completed
        """
        info = self._death_states.get(entity_id)
        if not info:
            return False

        old_state = info.death_state

        # Clear death state
        del self._death_states[entity_id]
        self._pending_cleanup.discard(entity_id)

        # Emit state change
        self._emit_state_changed(entity_id, old_state, DeathState.ALIVE)

        return True

    # =========================================================================
    # RESPAWN QUEUE
    # =========================================================================

    def queue_respawn(
        self,
        entity_id: int,
        delay: Optional[float] = None,
        health_percentage: float = RESPAWN_HEALTH_PERCENTAGE,
        position: Any = None,
        team_id: Optional[int] = None,
        add_invulnerability: bool = True,
        invulnerability_duration: Optional[float] = None,
        source_id: Optional[int] = None,
        **metadata: Any,
    ) -> RespawnRequest:
        """
        Queue an entity for respawn.

        Args:
            entity_id: Entity to respawn
            delay: Respawn delay in seconds (None = use default)
            health_percentage: Health to restore on respawn
            position: Override respawn position
            team_id: Team for spawn point selection
            add_invulnerability: Whether to add spawn protection
            invulnerability_duration: Duration of spawn protection
            source_id: Who requested the respawn
            **metadata: Additional metadata

        Returns:
            RespawnRequest for the queued respawn
        """
        # Use configured defaults
        if delay is None:
            delay = self._config.default_respawn_time
        delay = max(self._config.min_respawn_time,
                   min(delay, self._config.max_respawn_time))

        if invulnerability_duration is None:
            invulnerability_duration = self._config.respawn_invulnerability_duration

        # Calculate respawn time
        respawn_time = time.time() + delay

        # Create request
        request = RespawnRequest(
            entity_id=entity_id,
            respawn_time=respawn_time,
            health_percentage=health_percentage,
            position=position,
            team_id=team_id,
            add_invulnerability=add_invulnerability,
            invulnerability_duration=invulnerability_duration,
            source_id=source_id,
            metadata=metadata,
        )

        # Remove any existing request for this entity
        self._respawn_queue = [r for r in self._respawn_queue if r.entity_id != entity_id]

        # Add to queue
        self._respawn_queue.append(request)

        # Transition to respawning state
        self.transition_to_respawning(entity_id)

        return request

    def cancel_respawn(self, entity_id: int) -> bool:
        """
        Cancel a queued respawn.

        Args:
            entity_id: Entity whose respawn to cancel

        Returns:
            True if respawn was cancelled
        """
        original_len = len(self._respawn_queue)
        self._respawn_queue = [r for r in self._respawn_queue if r.entity_id != entity_id]

        if len(self._respawn_queue) < original_len:
            # Revert to DEAD state
            info = self._death_states.get(entity_id)
            if info and info.death_state == DeathState.RESPAWNING:
                info.death_state = DeathState.DEAD
                self._emit_state_changed(entity_id, DeathState.RESPAWNING, DeathState.DEAD)
            return True

        return False

    def get_respawn_request(self, entity_id: int) -> Optional[RespawnRequest]:
        """Get respawn request for an entity."""
        for request in self._respawn_queue:
            if request.entity_id == entity_id:
                return request
        return None

    def get_respawn_time_remaining(self, entity_id: int) -> float:
        """Get time remaining until respawn."""
        request = self.get_respawn_request(entity_id)
        return request.time_until_respawn if request else 0.0

    def instant_respawn(
        self,
        entity: DeathSubject,
        health_percentage: float = RESPAWN_HEALTH_PERCENTAGE,
        position: Any = None,
        team_id: Optional[int] = None,
        add_invulnerability: bool = True,
        invulnerability_duration: Optional[float] = None,
    ) -> bool:
        """
        Immediately respawn an entity.

        Args:
            entity: Entity to respawn
            health_percentage: Health to restore
            position: Respawn position
            team_id: Team for spawn point selection
            add_invulnerability: Whether to add spawn protection
            invulnerability_duration: Duration of spawn protection

        Returns:
            True if respawn succeeded
        """
        if invulnerability_duration is None:
            invulnerability_duration = self._config.respawn_invulnerability_duration

        # Get respawn position
        if position is None and self._respawn_provider:
            position = self._respawn_provider.get_respawn_position(
                entity.entity_id, team_id
            )

        # Revive entity
        success = entity.revive(
            health_percentage=health_percentage,
            add_invulnerability=add_invulnerability,
            invulnerability_duration=invulnerability_duration,
        )

        if success:
            # Clear death state
            self.complete_respawn(entity.entity_id)

            # Emit respawn event
            self._emit_respawn(RespawnEvent(
                entity_id=entity.entity_id,
                respawn_position=position,
                health_percentage=health_percentage,
                team_id=team_id,
            ))

        return success

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(self, delta_time: float, entities: Optional[Dict[int, DeathSubject]] = None) -> List[int]:
        """
        Update death system state.

        Args:
            delta_time: Time since last update
            entities: Map of entity IDs to DeathSubject instances (for respawning)

        Returns:
            List of entity IDs that were respawned
        """
        respawned: List[int] = []

        # Update dying entities -> dead
        for entity_id, info in list(self._death_states.items()):
            if info.death_state == DeathState.DYING:
                if info.time_since_death >= self._config.dying_duration:
                    self.transition_to_dead(entity_id)

        # Process cleanup for newly dead entities
        self._process_cleanup()

        # Process respawn queue
        if entities:
            for request in list(self._respawn_queue):
                if request.is_ready:
                    entity = entities.get(request.entity_id)
                    if entity:
                        # Get respawn position
                        position = request.position
                        if position is None and self._respawn_provider:
                            position = self._respawn_provider.get_respawn_position(
                                request.entity_id, request.team_id
                            )

                        # Respawn entity
                        success = entity.revive(
                            health_percentage=request.health_percentage,
                            add_invulnerability=request.add_invulnerability,
                            invulnerability_duration=request.invulnerability_duration,
                        )

                        if success:
                            respawned.append(request.entity_id)
                            self.complete_respawn(request.entity_id)

                            # Emit respawn event
                            self._emit_respawn(RespawnEvent(
                                entity_id=request.entity_id,
                                respawn_position=position,
                                health_percentage=request.health_percentage,
                                team_id=request.team_id,
                            ))

                    # Remove from queue
                    self._respawn_queue.remove(request)

        return respawned

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def register_cleanup_handler(self, handler: CleanupHandler) -> None:
        """Register a handler for entity cleanup on death."""
        self._cleanup_handlers.append(handler)

    def unregister_cleanup_handler(self, handler: CleanupHandler) -> bool:
        """Unregister a cleanup handler."""
        try:
            self._cleanup_handlers.remove(handler)
            return True
        except ValueError:
            return False

    def _process_cleanup(self) -> None:
        """Process cleanup for pending entities."""
        for entity_id in list(self._pending_cleanup):
            info = self._death_states.get(entity_id)
            if info:
                for handler in self._cleanup_handlers:
                    try:
                        handler.on_entity_death(entity_id, info)
                    except Exception:
                        pass
            self._pending_cleanup.discard(entity_id)

    def force_cleanup(self, entity_id: int) -> None:
        """Force immediate cleanup for an entity."""
        info = self._death_states.get(entity_id)
        if info:
            for handler in self._cleanup_handlers:
                try:
                    handler.on_entity_death(entity_id, info)
                except Exception:
                    pass

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def on_death(self, handler: Callable[[DeathEvent], None]) -> None:
        """Register a handler for death events."""
        self._on_death.append(handler)

    def on_respawn(self, handler: Callable[[RespawnEvent], None]) -> None:
        """Register a handler for respawn events."""
        self._on_respawn.append(handler)

    def on_state_changed(
        self,
        handler: Callable[[int, DeathState, DeathState], None],
    ) -> None:
        """Register a handler for state changes."""
        self._on_state_changed.append(handler)

    def _emit_death(self, event: DeathEvent) -> None:
        """Emit death event."""
        for handler in self._on_death:
            try:
                handler(event)
            except Exception:
                pass

    def _emit_respawn(self, event: RespawnEvent) -> None:
        """Emit respawn event."""
        for handler in self._on_respawn:
            try:
                handler(event)
            except Exception:
                pass

    def _emit_state_changed(
        self,
        entity_id: int,
        old_state: DeathState,
        new_state: DeathState,
    ) -> None:
        """Emit state changed event."""
        for handler in self._on_state_changed:
            try:
                handler(entity_id, old_state, new_state)
            except Exception:
                pass

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_all_dead(self) -> List[int]:
        """Get all dead entity IDs."""
        return [
            eid for eid, info in self._death_states.items()
            if info.death_state in (DeathState.DEAD, DeathState.DYING)
        ]

    def get_all_respawning(self) -> List[int]:
        """Get all respawning entity IDs."""
        return [
            eid for eid, info in self._death_states.items()
            if info.death_state == DeathState.RESPAWNING
        ]

    def get_pending_respawns(self) -> List[RespawnRequest]:
        """Get all pending respawn requests."""
        return list(self._respawn_queue)

    def get_recent_deaths(
        self,
        time_window: float = 60.0,
        killer_id: Optional[int] = None,
    ) -> List[DeathInfo]:
        """
        Get recent deaths within a time window.

        Args:
            time_window: Time window in seconds
            killer_id: Optional filter by killer

        Returns:
            List of DeathInfo for recent deaths
        """
        cutoff = time.time() - time_window
        deaths = []

        for info in self._death_states.values():
            if info.timestamp >= cutoff:
                if killer_id is None or info.killer_id == killer_id:
                    deaths.append(info)

        return sorted(deaths, key=lambda d: d.timestamp, reverse=True)

    # =========================================================================
    # UTILITY
    # =========================================================================

    def clear(self) -> None:
        """Clear all death states and respawn queue."""
        self._death_states.clear()
        self._respawn_queue.clear()
        self._pending_cleanup.clear()

    def remove_entity(self, entity_id: int) -> None:
        """Remove all tracking for an entity."""
        self._death_states.pop(entity_id, None)
        self._respawn_queue = [r for r in self._respawn_queue if r.entity_id != entity_id]
        self._pending_cleanup.discard(entity_id)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Protocols
    "DeathSubject",
    "CleanupHandler",
    "RespawnProvider",
    # Data classes
    "DeathInfo",
    "RespawnRequest",
    "DeathEvent",
    "RespawnEvent",
    # System
    "DeathSystem",
]
