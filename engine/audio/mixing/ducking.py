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


@dataclass
class DuckEnvelope:
    """
    Envelope follower for smooth ducking transitions.

    Tracks the current duck amount with attack, hold, and release phases.
    """
    attack_ms: float = DUCK_ATTACK_MS
    hold_ms: float = DUCK_HOLD_MS
    release_ms: float = DUCK_RELEASE_MS
    state: DuckState = DuckState.IDLE
    current_amount: float = 0.0  # 0.0 = no duck, 1.0 = full duck
    target_amount: float = 0.0
    state_start_time: float = 0.0
    hold_end_time: float = 0.0

    def trigger(self, amount: float = 1.0) -> None:
        """
        Trigger the duck envelope.

        Args:
            amount: Duck amount (0.0 to 1.0).
        """
        self.target_amount = clamp(amount, 0.0, 1.0)
        if self.state in (DuckState.IDLE, DuckState.RELEASING):
            self.state = DuckState.ATTACKING
            self.state_start_time = time.time()

    def release(self) -> None:
        """Release the duck envelope."""
        if self.state in (DuckState.ATTACKING, DuckState.HOLDING):
            self.state = DuckState.HOLDING
            self.hold_end_time = time.time() + (self.hold_ms / 1000.0)

    def update(self, delta_time: float) -> float:
        """
        Update the envelope and return current duck amount.

        Args:
            delta_time: Time since last update in seconds.

        Returns:
            Current duck amount (0.0 to 1.0).
        """
        current_time = time.time()

        if self.state == DuckState.ATTACKING:
            attack_time = self.attack_ms / 1000.0
            if attack_time > 0:
                elapsed = current_time - self.state_start_time
                self.current_amount = min(
                    self.target_amount,
                    self.current_amount + (delta_time / attack_time) * self.target_amount
                )
            else:
                self.current_amount = self.target_amount

            if self.current_amount >= self.target_amount:
                self.current_amount = self.target_amount
                self.state = DuckState.HOLDING
                self.hold_end_time = current_time + (self.hold_ms / 1000.0)

        elif self.state == DuckState.HOLDING:
            if current_time >= self.hold_end_time:
                self.state = DuckState.RELEASING
                self.state_start_time = current_time

        elif self.state == DuckState.RELEASING:
            release_time = self.release_ms / 1000.0
            if release_time > 0:
                elapsed = current_time - self.state_start_time
                self.current_amount = max(
                    0.0,
                    self.current_amount - (delta_time / release_time) * self.target_amount
                )
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

    def __init__(self, config: DuckConfig) -> None:
        """
        Initialize ducking instance.

        Args:
            config: Ducking configuration.
        """
        self._config = config.copy()
        self._envelope = DuckEnvelope(
            attack_ms=config.attack_ms,
            hold_ms=config.hold_ms,
            release_ms=config.release_ms,
        )
        self._triggered = False
        self._source_level_db = -80.0

    @property
    def config(self) -> DuckConfig:
        """Get the ducking configuration."""
        return self._config

    @property
    def envelope(self) -> DuckEnvelope:
        """Get the ducking envelope."""
        return self._envelope

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

    def __init__(self) -> None:
        """Initialize the ducking manager."""
        self._lock = threading.RLock()
        self._instances: dict[str, DuckingInstance] = {}
        self._bus_duck_amounts: dict[str, float] = {}  # bus_id -> total duck linear
        self._on_duck_change: list[Callable[[MixBus, float], None]] = []

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
                    # Schedule release
                    instance._envelope.hold_end_time = (
                        time.time() + (duration_ms / 1000.0)
                    )

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
