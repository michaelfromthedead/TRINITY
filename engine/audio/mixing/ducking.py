"""
Ducking system for automatic volume reduction.

This module implements:
- Dialogue ducking: Reduce music/sfx when VO is playing
- Event ducking: Reduce other sounds for big moments
- Focus ducking: Player attention management

Ducking uses envelope followers with attack/release for smooth transitions.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
from uuid import uuid4

from .config import (
    DIALOGUE_DUCK_AMOUNT_DB,
    DUCK_ATTACK_MS,
    DUCK_HOLD_MS,
    DUCK_RELEASE_MS,
    DUCK_THRESHOLD_DB,
    EVENT_DUCK_AMOUNT_DB,
    EVENT_DUCK_ATTACK_MS,
    EVENT_DUCK_HOLD_MS,
    FOCUS_DUCK_AMOUNT_DB,
    FOCUS_DUCK_ATTACK_MS,
    FOCUS_DUCK_HOLD_MS,
    FOCUS_DUCK_RELEASE_MS,
    LOCK_TIMEOUT,
    clamp,
    db_to_linear,
    linear_to_db,
)
from .mix_bus import MixBus


class DuckType(Enum):
    """Types of ducking behavior."""
    DIALOGUE = "dialogue"  # VO ducks other sounds
    EVENT = "event"        # Big moments duck everything
    FOCUS = "focus"        # Player attention management
    CUSTOM = "custom"      # User-defined ducking


class DuckState(Enum):
    """Current state of a duck envelope."""
    IDLE = "idle"          # Not ducking
    ATTACKING = "attacking"  # Ramping down
    HOLDING = "holding"    # At maximum duck
    RELEASING = "releasing"  # Ramping up

    # Aliases for compatibility
    ATTACK = "attacking"
    HOLD = "holding"
    RELEASE = "releasing"


@dataclass
class DuckEnvelope:
    """
    Envelope follower for smooth ducking transitions.

    Tracks the current duck amount with attack, hold, and release phases.
    Uses accumulated delta time for deterministic timing in tests.
    """
    attack_ms: float = DUCK_ATTACK_MS
    hold_ms: float = DUCK_HOLD_MS
    release_ms: float = DUCK_RELEASE_MS
    state: DuckState = DuckState.IDLE
    current_amount: float = 0.0  # 0.0 = no duck, 1.0 = full duck
    target_amount: float = 0.0
    state_elapsed_ms: float = 0.0  # Time elapsed in current state
    hold_remaining_ms: float = 0.0  # Remaining hold time

    @property
    def value(self) -> float:
        """
        Get current envelope value (1.0 = no duck, 0.0 = full duck).

        This is the inverse of current_amount for volume multiplication.
        """
        return 1.0 - self.current_amount

    @property
    def hold_end_time(self) -> float:
        """
        Get hold end time (for compatibility).

        Returns a positive value when in HOLDING state with remaining time.
        """
        if self.state == DuckState.HOLDING and self.hold_remaining_ms > 0:
            return time.time() + (self.hold_remaining_ms / 1000.0)
        return 0.0

    def trigger(self, amount: float = 1.0, target: Optional[float] = None) -> None:
        """
        Trigger the duck envelope.

        Args:
            amount: Duck amount (0.0 to 1.0). Ignored if target is provided.
            target: Target value (0.0 to 1.0). If provided, used instead of amount.
        """
        # Support both 'amount' (original) and 'target' (test expected) parameters
        if target is not None:
            # target is the target value (0.5 means duck to 50% volume)
            self.target_amount = clamp(1.0 - target, 0.0, 1.0)
        else:
            self.target_amount = clamp(amount, 0.0, 1.0)

        if self.state in (DuckState.IDLE, DuckState.RELEASING):
            self.state = DuckState.ATTACKING
            self.state_elapsed_ms = 0.0

    def release(self) -> None:
        """Release the duck envelope."""
        if self.state in (DuckState.ATTACKING, DuckState.HOLDING):
            self.state = DuckState.HOLDING
            self.hold_remaining_ms = self.hold_ms

    def update(self, delta_time: float) -> float:
        """
        Update the envelope and return current duck amount.

        Args:
            delta_time: Time since last update in seconds.

        Returns:
            Current duck amount (0.0 to 1.0).
        """
        delta_ms = delta_time * 1000.0

        if self.state == DuckState.ATTACKING:
            # Check if we're already at or past the target (allows manual setting)
            if self.current_amount >= self.target_amount:
                self.current_amount = self.target_amount
                self.state = DuckState.HOLDING
                self.hold_remaining_ms = self.hold_ms
            elif self.attack_ms > 0:
                self.state_elapsed_ms += delta_ms
                progress = self.state_elapsed_ms / self.attack_ms
                self.current_amount = min(self.target_amount, self.target_amount * progress)
                # Transition to holding when attack complete
                if self.current_amount >= self.target_amount:
                    self.current_amount = self.target_amount
                    self.state = DuckState.HOLDING
                    self.hold_remaining_ms = self.hold_ms
            else:
                self.current_amount = self.target_amount
                self.state = DuckState.HOLDING
                self.hold_remaining_ms = self.hold_ms

        elif self.state == DuckState.HOLDING:
            self.hold_remaining_ms -= delta_ms
            if self.hold_remaining_ms <= 0:
                self.state = DuckState.RELEASING
                self.state_elapsed_ms = 0.0

        elif self.state == DuckState.RELEASING:
            if self.release_ms > 0:
                self.state_elapsed_ms += delta_ms
                progress = self.state_elapsed_ms / self.release_ms
                self.current_amount = max(0.0, self.target_amount * (1.0 - progress))
            else:
                self.current_amount = 0.0

            if self.current_amount <= 0.0:
                self.current_amount = 0.0
                self.state = DuckState.IDLE

        return self.current_amount

    def reset(self) -> None:
        """Reset the envelope to idle state."""
        self.state = DuckState.IDLE
        self.current_amount = 0.0
        self.target_amount = 0.0
        self.state_elapsed_ms = 0.0
        self.hold_remaining_ms = 0.0

    def is_active(self) -> bool:
        """Check if the envelope is actively ducking."""
        return self.state != DuckState.IDLE or self.current_amount > 0.0


@dataclass
class DuckConfig:
    """Configuration for a ducking relationship."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    duck_type: DuckType = DuckType.DIALOGUE
    source_bus: Optional[MixBus] = None  # Bus that triggers ducking
    target_buses: list[MixBus] = field(default_factory=list)  # Buses to duck
    amount_db: float = DIALOGUE_DUCK_AMOUNT_DB
    threshold_db: float = DUCK_THRESHOLD_DB
    attack_ms: float = DUCK_ATTACK_MS
    hold_ms: float = DUCK_HOLD_MS
    release_ms: float = DUCK_RELEASE_MS
    enabled: bool = True
    priority: int = 100  # Higher = takes precedence

    @property
    def amount_linear(self) -> float:
        """Get duck amount as linear multiplier."""
        return db_to_linear(self.amount_db)

    def copy(self) -> DuckConfig:
        """Create a copy of this config."""
        return DuckConfig(
            id=self.id,
            name=self.name,
            duck_type=self.duck_type,
            source_bus=self.source_bus,
            target_buses=list(self.target_buses),
            amount_db=self.amount_db,
            threshold_db=self.threshold_db,
            attack_ms=self.attack_ms,
            hold_ms=self.hold_ms,
            release_ms=self.release_ms,
            enabled=self.enabled,
            priority=self.priority,
        )


class DuckingInstance:
    """
    Active ducking instance with envelope.

    Tracks one ducking relationship between a source and targets.
    """

    def __init__(
        self,
        config: Optional[DuckConfig] = None,
        *,
        id: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
    ) -> None:
        """
        Initialize ducking instance.

        Args:
            config: Ducking configuration (optional if id/source/target provided).
            id: Instance ID (alternative constructor).
            source: Source bus name (alternative constructor).
            target: Target bus name (alternative constructor).
        """
        # Support alternative constructor: DuckingInstance(id=..., source=..., target=..., config=...)
        if config is None:
            config = DuckConfig()

        self._config = config.copy()
        self._id = id or self._config.id
        self._source = source
        self._target = target
        self._envelope = DuckEnvelope(
            attack_ms=config.attack_ms,
            hold_ms=config.hold_ms,
            release_ms=config.release_ms,
        )
        self._triggered = False
        self._source_level_db = -80.0

    @property
    def id(self) -> str:
        """Get the instance ID."""
        return self._id

    @property
    def source(self) -> Optional[str]:
        """Get the source bus name."""
        return self._source

    @property
    def target(self) -> Optional[str]:
        """Get the target bus name."""
        return self._target

    @property
    def config(self) -> DuckConfig:
        """Get the ducking configuration."""
        return self._config

    @property
    def envelope(self) -> DuckEnvelope:
        """Get the ducking envelope."""
        return self._envelope

    @property
    def state(self) -> DuckState:
        """Get the current envelope state."""
        return self._envelope.state

    @property
    def is_active(self) -> bool:
        """Check if ducking is currently active."""
        return self._envelope.is_active()

    @property
    def current_duck_db(self) -> float:
        """Get current duck amount in dB."""
        return self._config.amount_db * self._envelope.current_amount

    @property
    def current_duck_linear(self) -> float:
        """Get current duck as linear multiplier."""
        if self._envelope.current_amount <= 0:
            return 1.0
        return db_to_linear(self.current_duck_db)

    def set_source_level(self, level_db: float) -> None:
        """
        Set the current level of the source bus.

        Args:
            level_db: Source level in dB.
        """
        self._source_level_db = level_db

        if not self._config.enabled:
            return

        # Trigger based on threshold
        if level_db > self._config.threshold_db:
            if not self._triggered:
                self._triggered = True
                self._envelope.trigger(1.0)
        else:
            if self._triggered:
                self._triggered = False
                self._envelope.release()

    def trigger(self, amount: float = 1.0) -> None:
        """Manually trigger ducking."""
        self._triggered = True
        self._envelope.trigger(amount)

    def release(self) -> None:
        """Manually release ducking."""
        self._triggered = False
        self._envelope.release()

    def update(self, delta_time: float) -> float:
        """
        Update the ducking envelope.

        Args:
            delta_time: Time since last update.

        Returns:
            Current duck multiplier (linear).
        """
        self._envelope.update(delta_time)
        return self.current_duck_linear

    def apply_to_targets(self) -> dict[str, float]:
        """
        Get the duck multipliers for target buses.

        Returns:
            Dictionary mapping bus ID to duck multiplier (linear).

        Note:
            This returns multipliers rather than modifying buses directly,
            as the actual mixing is handled by the Mixer/DuckingManager
            which tracks natural vs ducked volumes separately.
        """
        duck_linear = self.current_duck_linear
        return {bus.id: duck_linear for bus in self._config.target_buses}

    def reset(self) -> None:
        """Reset ducking state."""
        self._triggered = False
        self._envelope.reset()

    def __repr__(self) -> str:
        return (
            f"DuckingInstance(id={self._id!r}, "
            f"source={self._source!r}, target={self._target!r}, "
            f"state={self._envelope.state.value})"
        )


class DuckingManager:
    """
    Manages all ducking relationships in the mix.

    Features:
    - Multiple ducking configurations
    - Automatic level analysis
    - Priority-based duck stacking
    - Per-bus duck amount tracking

    Thread Safety:
        All operations are protected by a lock.
    """

    def __init__(self, mixer: Optional[object] = None) -> None:
        """
        Initialize the ducking manager.

        Args:
            mixer: Optional reference to the Mixer for integration.
        """
        self._lock = threading.RLock()
        self._mixer = mixer
        self._instances: dict[str, DuckingInstance] = {}
        self._bus_duck_amounts: dict[str, float] = {}  # bus_id -> total duck linear
        self._on_duck_change: list[Callable[[MixBus, float], None]] = []

    @property
    def active_duck_count(self) -> int:
        """Get the number of currently active ducks."""
        with self._lock:
            return sum(1 for inst in self._instances.values() if inst.is_active)

    # =========================================================================
    # Simple API (string-based bus names)
    # =========================================================================

    def apply_duck(
        self,
        source: str,
        target: str,
        amount_db: float,
        attack_ms: float = DUCK_ATTACK_MS,
        release_ms: float = DUCK_RELEASE_MS,
        hold_ms: float = DUCK_HOLD_MS,
        priority: int = 100,
    ) -> str:
        """
        Apply a duck from source bus to target bus.

        Args:
            source: Source bus name (e.g., "vo").
            target: Target bus name (e.g., "sfx").
            amount_db: Duck amount in dB (negative).
            attack_ms: Attack time in milliseconds.
            release_ms: Release time in milliseconds.
            hold_ms: Hold time in milliseconds.
            priority: Duck priority (higher wins).

        Returns:
            The duck instance ID.
        """
        duck_id = str(uuid4())
        config = DuckConfig(
            id=duck_id,
            name=f"{source}_to_{target}",
            duck_type=DuckType.CUSTOM,
            amount_db=amount_db,
            attack_ms=attack_ms,
            hold_ms=hold_ms,
            release_ms=release_ms,
            priority=priority,
        )
        with self._lock:
            instance = DuckingInstance(
                config,
                id=duck_id,
                source=source,
                target=target,
            )
            # Trigger the duck immediately
            instance.trigger(1.0)
            self._instances[duck_id] = instance
        return duck_id

    def apply_dialogue_duck(self, target: str) -> str:
        """
        Apply a standard dialogue duck to target.

        Args:
            target: Target bus name.

        Returns:
            The duck instance ID.
        """
        return self.apply_duck(
            source="vo",
            target=target,
            amount_db=DIALOGUE_DUCK_AMOUNT_DB,
            attack_ms=DUCK_ATTACK_MS,
            release_ms=DUCK_RELEASE_MS,
            priority=200,
        )

    def apply_event_duck(self, target: str) -> str:
        """
        Apply a standard event duck to target.

        Args:
            target: Target bus name.

        Returns:
            The duck instance ID.
        """
        return self.apply_duck(
            source="event",
            target=target,
            amount_db=EVENT_DUCK_AMOUNT_DB,
            attack_ms=EVENT_DUCK_ATTACK_MS,
            hold_ms=EVENT_DUCK_HOLD_MS,
            release_ms=DUCK_RELEASE_MS,
            priority=250,
        )

    def apply_focus_duck(self, target: str) -> str:
        """
        Apply a standard focus duck to target.

        Args:
            target: Target bus name.

        Returns:
            The duck instance ID.
        """
        return self.apply_duck(
            source="focus",
            target=target,
            amount_db=FOCUS_DUCK_AMOUNT_DB,
            attack_ms=FOCUS_DUCK_ATTACK_MS,
            hold_ms=FOCUS_DUCK_HOLD_MS,
            release_ms=FOCUS_DUCK_RELEASE_MS,
            priority=150,
        )

    def release_duck(self, duck_id: str) -> None:
        """
        Release a duck by ID.

        Args:
            duck_id: ID of the duck to release.
        """
        with self._lock:
            instance = self._instances.get(duck_id)
            if instance is not None:
                instance.release()

    def release_all(self) -> None:
        """Release all active ducks."""
        with self._lock:
            for instance in self._instances.values():
                instance.release()

    def get_duck_state(self, duck_id: str) -> DuckState:
        """
        Get the current state of a duck.

        Args:
            duck_id: ID of the duck.

        Returns:
            Current DuckState.
        """
        with self._lock:
            instance = self._instances.get(duck_id)
            if instance is not None:
                return instance.envelope.state
            return DuckState.IDLE

    def get_active_ducks(self) -> list[DuckingInstance]:
        """
        Get all currently active ducks.

        Returns:
            List of active DuckingInstance objects.
        """
        with self._lock:
            return [inst for inst in self._instances.values() if inst.is_active]

    # =========================================================================
    # Configuration
    # =========================================================================

    def create_duck(self, config: DuckConfig) -> DuckingInstance:
        """
        Create a new ducking relationship.

        Args:
            config: Ducking configuration.

        Returns:
            The created ducking instance.
        """
        with self._lock:
            instance = DuckingInstance(config)
            self._instances[config.id] = instance
            return instance

    def remove_duck(self, duck_id: str) -> bool:
        """
        Remove a ducking relationship.

        Args:
            duck_id: ID of duck to remove.

        Returns:
            True if removed.
        """
        with self._lock:
            return self._instances.pop(duck_id, None) is not None

    def get_duck(self, duck_id: str) -> Optional[DuckingInstance]:
        """
        Get a ducking instance by ID.

        Args:
            duck_id: ID of duck.

        Returns:
            DuckingInstance if found.
        """
        with self._lock:
            return self._instances.get(duck_id)

    def get_ducks_by_type(self, duck_type: DuckType) -> list[DuckingInstance]:
        """Get all ducks of a specific type."""
        with self._lock:
            return [
                d for d in self._instances.values()
                if d.config.duck_type == duck_type
            ]

    def get_ducks_for_target(self, bus: MixBus) -> list[DuckingInstance]:
        """Get all ducks that affect a target bus."""
        with self._lock:
            return [
                d for d in self._instances.values()
                if bus in d.config.target_buses
            ]

    # =========================================================================
    # Preset Configurations
    # =========================================================================

    def create_dialogue_duck(
        self,
        source: MixBus,
        targets: list[MixBus],
        amount_db: float = DIALOGUE_DUCK_AMOUNT_DB,
    ) -> DuckingInstance:
        """
        Create standard dialogue ducking.

        Args:
            source: VO bus that triggers ducking.
            targets: Buses to duck (typically music, sfx).
            amount_db: Duck amount in dB.

        Returns:
            The created ducking instance.
        """
        config = DuckConfig(
            name="dialogue_duck",
            duck_type=DuckType.DIALOGUE,
            source_bus=source,
            target_buses=targets,
            amount_db=amount_db,
            threshold_db=DUCK_THRESHOLD_DB,
            attack_ms=DUCK_ATTACK_MS,
            hold_ms=DUCK_HOLD_MS,
            release_ms=DUCK_RELEASE_MS,
            priority=200,  # High priority
        )
        return self.create_duck(config)

    def create_event_duck(
        self,
        targets: list[MixBus],
        amount_db: float = EVENT_DUCK_AMOUNT_DB,
        attack_ms: float = EVENT_DUCK_ATTACK_MS,
    ) -> DuckingInstance:
        """
        Create event ducking (for explosions, etc).

        Args:
            targets: Buses to duck.
            amount_db: Duck amount in dB.
            attack_ms: Fast attack for sudden events.

        Returns:
            The created ducking instance.
        """
        config = DuckConfig(
            name="event_duck",
            duck_type=DuckType.EVENT,
            source_bus=None,  # Triggered manually
            target_buses=targets,
            amount_db=amount_db,
            attack_ms=attack_ms,
            hold_ms=EVENT_DUCK_HOLD_MS,
            release_ms=DUCK_RELEASE_MS,
            priority=250,  # Highest priority
        )
        return self.create_duck(config)

    def create_focus_duck(
        self,
        targets: list[MixBus],
        amount_db: float = FOCUS_DUCK_AMOUNT_DB,
    ) -> DuckingInstance:
        """
        Create focus ducking for player attention.

        Args:
            targets: Buses to duck.
            amount_db: Duck amount in dB.

        Returns:
            The created ducking instance.
        """
        config = DuckConfig(
            name="focus_duck",
            duck_type=DuckType.FOCUS,
            source_bus=None,  # Triggered manually
            target_buses=targets,
            amount_db=amount_db,
            attack_ms=FOCUS_DUCK_ATTACK_MS,
            hold_ms=FOCUS_DUCK_HOLD_MS,
            release_ms=FOCUS_DUCK_RELEASE_MS,
            priority=150,
        )
        return self.create_duck(config)

    # =========================================================================
    # Update
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """
        Update all ducking envelopes.

        Args:
            delta_time: Time since last update in seconds.
        """
        with self._lock:
            # Reset bus duck amounts
            self._bus_duck_amounts.clear()

            # Update all instances and accumulate duck amounts
            for instance in self._instances.values():
                if not instance.config.enabled:
                    continue

                instance.update(delta_time)

                if instance.is_active:
                    duck_linear = instance.current_duck_linear
                    for bus in instance.config.target_buses:
                        current = self._bus_duck_amounts.get(bus.id, 1.0)
                        # Multiply ducks together (most aggressive wins)
                        self._bus_duck_amounts[bus.id] = min(current, duck_linear)

            # Copy for callback notification
            changes = dict(self._bus_duck_amounts)
            callbacks = list(self._on_duck_change)
            instances = list(self._instances.values())

        # Notify callbacks outside lock
        for bus_id, amount in changes.items():
            for instance in instances:
                for bus in instance.config.target_buses:
                    if bus.id == bus_id:
                        for callback in callbacks:
                            try:
                                callback(bus, amount)
                            except Exception:
                                pass
                        break

    def get_duck_amount(self, bus: MixBus) -> float:
        """
        Get the current duck amount for a bus.

        Args:
            bus: Bus to check.

        Returns:
            Duck multiplier (1.0 = no duck, 0.5 = -6dB, etc).
        """
        with self._lock:
            return self._bus_duck_amounts.get(bus.id, 1.0)

    def get_duck_amount_db(self, bus: MixBus) -> float:
        """
        Get the current duck amount in dB.

        Args:
            bus: Bus to check.

        Returns:
            Duck amount in dB (0 = no duck, -6 = 6dB reduction).
        """
        linear = self.get_duck_amount(bus)
        if linear >= 1.0:
            return 0.0
        return linear_to_db(linear)

    # =========================================================================
    # Manual Triggers
    # =========================================================================

    def trigger_event_duck(self, duration_ms: float = 500.0) -> None:
        """
        Trigger all event ducks.

        Args:
            duration_ms: How long to hold the duck.
        """
        with self._lock:
            for instance in self._instances.values():
                if instance.config.duck_type == DuckType.EVENT:
                    instance.trigger()
                    # Schedule release by setting hold remaining time
                    instance._envelope.hold_remaining_ms = duration_ms

    def trigger_focus_duck(self) -> None:
        """Trigger all focus ducks."""
        with self._lock:
            for instance in self._instances.values():
                if instance.config.duck_type == DuckType.FOCUS:
                    instance.trigger()

    def release_focus_duck(self) -> None:
        """Release all focus ducks."""
        with self._lock:
            for instance in self._instances.values():
                if instance.config.duck_type == DuckType.FOCUS:
                    instance.release()

    # =========================================================================
    # Level Analysis
    # =========================================================================

    def analyze_source_levels(self, bus_levels: dict[str, float]) -> None:
        """
        Analyze source bus levels to trigger automatic ducking.

        Args:
            bus_levels: Dictionary of bus_id -> level_db.
        """
        with self._lock:
            for instance in self._instances.values():
                if instance.config.source_bus is None:
                    continue

                source_id = instance.config.source_bus.id
                if source_id in bus_levels:
                    instance.set_source_level(bus_levels[source_id])

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_duck_change(self, callback: Callable[[MixBus, float], None]) -> None:
        """
        Register a callback for duck amount changes.

        Args:
            callback: Function(bus, duck_linear) called on changes.
        """
        with self._lock:
            self._on_duck_change.append(callback)

    def remove_callback(self, callback: Callable[[MixBus, float], None]) -> bool:
        """Remove a duck change callback."""
        with self._lock:
            if callback in self._on_duck_change:
                self._on_duck_change.remove(callback)
                return True
        return False

    # =========================================================================
    # State Management
    # =========================================================================

    def reset_all(self) -> None:
        """Reset all ducking instances."""
        with self._lock:
            for instance in self._instances.values():
                instance.reset()
            self._bus_duck_amounts.clear()

    def clear(self) -> None:
        """Remove all ducking configurations."""
        with self._lock:
            self._instances.clear()
            self._bus_duck_amounts.clear()

    def get_state(self) -> dict:
        """Get current ducking state for debugging."""
        with self._lock:
            return {
                "instances": {
                    id: {
                        "name": inst.config.name,
                        "type": inst.config.duck_type.value,
                        "enabled": inst.config.enabled,
                        "is_active": inst.is_active,
                        "current_duck_db": inst.current_duck_db,
                        "envelope_state": inst.envelope.state.value,
                    }
                    for id, inst in self._instances.items()
                },
                "bus_amounts": {
                    bus_id: linear_to_db(amount) if amount < 1.0 else 0.0
                    for bus_id, amount in self._bus_duck_amounts.items()
                },
            }

    def __repr__(self) -> str:
        with self._lock:
            active = sum(1 for i in self._instances.values() if i.is_active)
            return f"DuckingManager(instances={len(self._instances)}, active={active})"
