"""
Mix Snapshots for storing and recalling mix states.

This module provides:
- MixSnapshot: Complete mix state storage
- SnapshotManager: Handles transitions and layering
- Parameter interpolation between snapshots
- Priority-based conflict resolution
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

from .config import (
    DEFAULT_HIGH_PASS,
    DEFAULT_INTERPOLATION_CURVE,
    DEFAULT_LOW_PASS,
    DEFAULT_PITCH,
    DEFAULT_SNAPSHOT_PITCH,
    DEFAULT_SNAPSHOT_PRIORITY,
    DEFAULT_SNAPSHOT_VOLUME,
    LOCK_TIMEOUT,
    MAX_ACTIVE_SNAPSHOTS,
    SNAPSHOT_BLEND_TIME,
    InterpolationCurve,
    apply_curve,
    lerp,
)
from .mix_bus import BusStateData, FilterParams, MixBus


class SnapshotState(Enum):
    """State of a snapshot transition."""
    INACTIVE = "inactive"
    BLENDING_IN = "blending_in"
    ACTIVE = "active"
    BLENDING_OUT = "blending_out"


@dataclass
class BusSnapshot:
    """Snapshot of a single bus's state."""
    bus_name: str
    volume_linear: float = DEFAULT_SNAPSHOT_VOLUME
    pitch: float = DEFAULT_SNAPSHOT_PITCH
    muted: bool = False
    low_pass_freq: float = DEFAULT_LOW_PASS
    low_pass_enabled: bool = False
    high_pass_freq: float = DEFAULT_HIGH_PASS
    high_pass_enabled: bool = False

    def copy(self) -> BusSnapshot:
        """Create a copy of this snapshot."""
        return BusSnapshot(
            bus_name=self.bus_name,
            volume_linear=self.volume_linear,
            pitch=self.pitch,
            muted=self.muted,
            low_pass_freq=self.low_pass_freq,
            low_pass_enabled=self.low_pass_enabled,
            high_pass_freq=self.high_pass_freq,
            high_pass_enabled=self.high_pass_enabled,
        )

    @classmethod
    def from_bus(cls, bus: MixBus) -> BusSnapshot:
        """Create a snapshot from a bus's current state."""
        state = bus.get_snapshot()
        filters = state.filters
        return cls(
            bus_name=bus.name,
            volume_linear=state.volume_linear,
            pitch=state.pitch,
            muted=state.muted,
            low_pass_freq=filters.low_pass_freq,
            low_pass_enabled=filters.low_pass_enabled,
            high_pass_freq=filters.high_pass_freq,
            high_pass_enabled=filters.high_pass_enabled,
        )

    def to_bus_state(self) -> BusStateData:
        """Convert to a BusStateData."""
        filters = FilterParams(
            low_pass_freq=self.low_pass_freq,
            low_pass_enabled=self.low_pass_enabled,
            high_pass_freq=self.high_pass_freq,
            high_pass_enabled=self.high_pass_enabled,
        )
        return BusStateData(
            volume_linear=self.volume_linear,
            pitch=self.pitch,
            muted=self.muted,
            filters=filters,
        )

    @classmethod
    def interpolate(
        cls,
        a: BusSnapshot,
        b: BusSnapshot,
        t: float,
    ) -> BusSnapshot:
        """
        Interpolate between two bus snapshots.

        Args:
            a: Start snapshot.
            b: End snapshot.
            t: Interpolation factor (0.0 to 1.0).

        Returns:
            Interpolated snapshot.
        """
        return cls(
            bus_name=a.bus_name,
            volume_linear=lerp(a.volume_linear, b.volume_linear, t),
            pitch=lerp(a.pitch, b.pitch, t),
            muted=b.muted if t >= 0.5 else a.muted,
            low_pass_freq=lerp(a.low_pass_freq, b.low_pass_freq, t),
            low_pass_enabled=b.low_pass_enabled if t >= 0.5 else a.low_pass_enabled,
            high_pass_freq=lerp(a.high_pass_freq, b.high_pass_freq, t),
            high_pass_enabled=b.high_pass_enabled if t >= 0.5 else a.high_pass_enabled,
        )


@dataclass
class MixSnapshot:
    """
    Complete snapshot of a mix state.

    A snapshot captures the state of all buses at a point in time,
    allowing mix states to be saved and recalled with smooth transitions.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    priority: int = DEFAULT_SNAPSHOT_PRIORITY
    bus_states: dict[str, BusSnapshot] = field(default_factory=dict)
    blend_time: float = SNAPSHOT_BLEND_TIME
    curve: InterpolationCurve = DEFAULT_INTERPOLATION_CURVE
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> MixSnapshot:
        """Create a deep copy of this snapshot."""
        return MixSnapshot(
            id=self.id,
            name=self.name,
            priority=self.priority,
            bus_states={k: v.copy() for k, v in self.bus_states.items()},
            blend_time=self.blend_time,
            curve=self.curve,
            metadata=dict(self.metadata),
        )

    @classmethod
    def capture(
        cls,
        name: str,
        buses: dict[str, MixBus],
        priority: int = DEFAULT_SNAPSHOT_PRIORITY,
        blend_time: float = SNAPSHOT_BLEND_TIME,
        curve: InterpolationCurve = DEFAULT_INTERPOLATION_CURVE,
    ) -> MixSnapshot:
        """
        Capture the current state of buses into a snapshot.

        Args:
            name: Name for this snapshot.
            buses: Dictionary of bus name -> MixBus.
            priority: Snapshot priority for conflict resolution.
            blend_time: Default transition time.
            curve: Default interpolation curve.

        Returns:
            New snapshot with captured state.
        """
        bus_states = {
            name: BusSnapshot.from_bus(bus)
            for name, bus in buses.items()
        }
        return cls(
            name=name,
            priority=priority,
            bus_states=bus_states,
            blend_time=blend_time,
            curve=curve,
        )

    def get_bus_state(self, bus_name: str) -> Optional[BusSnapshot]:
        """Get the snapshot state for a specific bus."""
        return self.bus_states.get(bus_name)

    def set_bus_state(self, bus_name: str, state: BusSnapshot) -> None:
        """Set the snapshot state for a specific bus."""
        self.bus_states[bus_name] = state

    def apply_to_bus(self, bus: MixBus) -> None:
        """
        Apply this snapshot's state to a bus immediately.

        Args:
            bus: Bus to apply state to.
        """
        if bus.name in self.bus_states:
            state = self.bus_states[bus.name]
            bus.set_snapshot(state.to_bus_state())

    def apply_to_all(self, buses: dict[str, MixBus]) -> None:
        """
        Apply this snapshot to all matching buses.

        Args:
            buses: Dictionary of bus name -> MixBus.
        """
        for name, bus in buses.items():
            if name in self.bus_states:
                self.apply_to_bus(bus)


@dataclass
class ActiveSnapshot:
    """An actively blending or applied snapshot."""
    snapshot: MixSnapshot
    state: SnapshotState = SnapshotState.INACTIVE
    blend_start: float = 0.0
    blend_progress: float = 0.0
    weight: float = 1.0  # For layered snapshots


class SnapshotManager:
    """
    Manages mix snapshots and transitions.

    Features:
    - Smooth transitions between snapshots
    - Multiple active snapshots with layering
    - Priority-based conflict resolution
    - Snapshot capture and recall

    Thread Safety:
        All operations are protected by a lock.
    """

    def __init__(self, buses: Optional[dict[str, MixBus]] = None) -> None:
        """
        Initialize the snapshot manager.

        Args:
            buses: Optional dictionary of managed buses.
        """
        self._lock = threading.RLock()
        self._buses: dict[str, MixBus] = buses or {}
        self._snapshots: dict[str, MixSnapshot] = {}
        self._active_snapshots: list[ActiveSnapshot] = []
        self._base_snapshot: Optional[MixSnapshot] = None
        self._on_transition_complete: list[Callable[[MixSnapshot], None]] = []

    # =========================================================================
    # Bus Management
    # =========================================================================

    def set_buses(self, buses: dict[str, MixBus]) -> None:
        """Set the managed buses."""
        with self._lock:
            self._buses = buses

    def get_bus(self, name: str) -> Optional[MixBus]:
        """Get a bus by name."""
        with self._lock:
            return self._buses.get(name)

    # =========================================================================
    # Snapshot Storage
    # =========================================================================

    def save_snapshot(self, snapshot: MixSnapshot) -> None:
        """
        Save a snapshot to storage.

        Args:
            snapshot: Snapshot to save.
        """
        with self._lock:
            self._snapshots[snapshot.name] = snapshot.copy()

    def load_snapshot(self, name: str) -> Optional[MixSnapshot]:
        """
        Load a snapshot from storage.

        Args:
            name: Name of snapshot to load.

        Returns:
            Snapshot if found, None otherwise.
        """
        with self._lock:
            snapshot = self._snapshots.get(name)
            return snapshot.copy() if snapshot else None

    def delete_snapshot(self, name: str) -> bool:
        """
        Delete a snapshot from storage.

        Args:
            name: Name of snapshot to delete.

        Returns:
            True if deleted.
        """
        with self._lock:
            return self._snapshots.pop(name, None) is not None

    def list_snapshots(self) -> list[str]:
        """Get list of stored snapshot names."""
        with self._lock:
            return list(self._snapshots.keys())

    def capture_snapshot(
        self,
        name: str,
        priority: int = DEFAULT_SNAPSHOT_PRIORITY,
        blend_time: float = SNAPSHOT_BLEND_TIME,
        curve: InterpolationCurve = DEFAULT_INTERPOLATION_CURVE,
    ) -> MixSnapshot:
        """
        Capture current mix state as a snapshot.

        Args:
            name: Name for the snapshot.
            priority: Snapshot priority.
            blend_time: Default transition time.
            curve: Default interpolation curve.

        Returns:
            The captured snapshot.
        """
        with self._lock:
            snapshot = MixSnapshot.capture(
                name=name,
                buses=self._buses,
                priority=priority,
                blend_time=blend_time,
                curve=curve,
            )
            self._snapshots[name] = snapshot.copy()
            return snapshot

    # =========================================================================
    # Transitions
    # =========================================================================

    def transition_to(
        self,
        snapshot: MixSnapshot,
        blend_time: Optional[float] = None,
        curve: Optional[InterpolationCurve] = None,
    ) -> None:
        """
        Start transition to a snapshot.

        Args:
            snapshot: Target snapshot.
            blend_time: Override blend time (uses snapshot default if None).
            curve: Override interpolation curve.
        """
        with self._lock:
            # Mark current active snapshots for blending out
            for active in self._active_snapshots:
                if active.state == SnapshotState.ACTIVE:
                    active.state = SnapshotState.BLENDING_OUT
                    active.blend_start = time.time()

            # Remove any already blending out that finished
            self._active_snapshots = [
                a for a in self._active_snapshots
                if a.state != SnapshotState.INACTIVE
            ]

            # Check max snapshots
            if len(self._active_snapshots) >= MAX_ACTIVE_SNAPSHOTS:
                # Force oldest to inactive
                oldest = min(self._active_snapshots, key=lambda a: a.blend_start)
                self._active_snapshots.remove(oldest)

            # Add new snapshot as blending in
            active = ActiveSnapshot(
                snapshot=snapshot.copy(),
                state=SnapshotState.BLENDING_IN,
                blend_start=time.time(),
            )

            # Override blend parameters if specified
            if blend_time is not None:
                active.snapshot.blend_time = blend_time
            if curve is not None:
                active.snapshot.curve = curve

            self._active_snapshots.append(active)

    def transition_to_named(
        self,
        name: str,
        blend_time: Optional[float] = None,
        curve: Optional[InterpolationCurve] = None,
    ) -> bool:
        """
        Start transition to a named snapshot.

        Args:
            name: Name of stored snapshot.
            blend_time: Override blend time.
            curve: Override interpolation curve.

        Returns:
            True if snapshot found and transition started.
        """
        snapshot = self.load_snapshot(name)
        if snapshot:
            self.transition_to(snapshot, blend_time, curve)
            return True
        return False

    def apply_immediate(self, snapshot: MixSnapshot) -> None:
        """
        Apply a snapshot immediately without blending.

        Args:
            snapshot: Snapshot to apply.
        """
        with self._lock:
            # Clear all active snapshots
            self._active_snapshots.clear()

            # Apply directly to buses
            snapshot.apply_to_all(self._buses)

            # Add as active
            active = ActiveSnapshot(
                snapshot=snapshot.copy(),
                state=SnapshotState.ACTIVE,
                blend_progress=1.0,
            )
            self._active_snapshots.append(active)

    def update(self, delta_time: float) -> None:
        """
        Update snapshot transitions.

        Call this every frame to progress transitions.

        Args:
            delta_time: Time since last update in seconds.
        """
        with self._lock:
            current_time = time.time()
            completed = []

            for active in self._active_snapshots:
                if active.state in (SnapshotState.BLENDING_IN, SnapshotState.BLENDING_OUT):
                    elapsed = current_time - active.blend_start
                    blend_time = active.snapshot.blend_time

                    if blend_time > 0:
                        raw_progress = min(1.0, elapsed / blend_time)
                        active.blend_progress = apply_curve(
                            raw_progress, active.snapshot.curve
                        )
                    else:
                        active.blend_progress = 1.0

                    if active.blend_progress >= 1.0:
                        if active.state == SnapshotState.BLENDING_IN:
                            active.state = SnapshotState.ACTIVE
                            completed.append(active.snapshot)
                        else:
                            active.state = SnapshotState.INACTIVE

            # Remove inactive snapshots
            self._active_snapshots = [
                a for a in self._active_snapshots
                if a.state != SnapshotState.INACTIVE
            ]

            # Apply blended state to buses
            self._apply_blended_state()

            # Notify callbacks outside lock
            callbacks = list(self._on_transition_complete)

        for snapshot in completed:
            for callback in callbacks:
                try:
                    callback(snapshot)
                except Exception:
                    pass

    def _apply_blended_state(self) -> None:
        """Apply the blended state of all active snapshots to buses."""
        if not self._active_snapshots:
            return

        # Calculate weighted blend for each bus
        for bus_name, bus in self._buses.items():
            blended_state: Optional[BusSnapshot] = None
            total_weight = 0.0

            for active in self._active_snapshots:
                if bus_name not in active.snapshot.bus_states:
                    continue

                state = active.snapshot.bus_states[bus_name]

                # Calculate weight based on state
                if active.state == SnapshotState.BLENDING_IN:
                    weight = active.blend_progress
                elif active.state == SnapshotState.BLENDING_OUT:
                    weight = 1.0 - active.blend_progress
                else:  # ACTIVE
                    weight = 1.0

                weight *= active.weight

                if weight <= 0:
                    continue

                if blended_state is None:
                    # Initialize with first weighted state
                    blended_state = state.copy()
                    # Don't multiply volume by weight here - accumulate properly below
                else:
                    # Blend with existing state using weighted interpolation
                    # t represents how much of the new state to blend in
                    t = weight / (total_weight + weight)
                    blended_state = BusSnapshot.interpolate(blended_state, state, t)

                total_weight += weight

            # Apply blended state to bus
            if blended_state is not None and total_weight > 0:
                bus.set_snapshot(blended_state.to_bus_state())

    # =========================================================================
    # Active Snapshot Queries
    # =========================================================================

    def get_active_snapshots(self) -> list[tuple[str, SnapshotState, float]]:
        """
        Get information about active snapshots.

        Returns:
            List of (name, state, progress) tuples.
        """
        with self._lock:
            return [
                (a.snapshot.name, a.state, a.blend_progress)
                for a in self._active_snapshots
            ]

    def is_transitioning(self) -> bool:
        """Check if any transitions are in progress."""
        with self._lock:
            return any(
                a.state in (SnapshotState.BLENDING_IN, SnapshotState.BLENDING_OUT)
                for a in self._active_snapshots
            )

    def get_current_snapshot_name(self) -> Optional[str]:
        """Get the name of the most recent active snapshot."""
        with self._lock:
            for active in reversed(self._active_snapshots):
                if active.state == SnapshotState.ACTIVE:
                    return active.snapshot.name
            return None

    # =========================================================================
    # Base Snapshot
    # =========================================================================

    def set_base_snapshot(self, snapshot: MixSnapshot) -> None:
        """
        Set the base/default snapshot.

        Args:
            snapshot: Snapshot to use as base.
        """
        with self._lock:
            self._base_snapshot = snapshot.copy()

    def reset_to_base(self, blend_time: Optional[float] = None) -> bool:
        """
        Transition back to the base snapshot.

        Args:
            blend_time: Override blend time.

        Returns:
            True if base snapshot exists and transition started.
        """
        with self._lock:
            if self._base_snapshot is not None:
                self.transition_to(self._base_snapshot, blend_time)
                return True
        return False

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_transition_complete(
        self, callback: Callable[[MixSnapshot], None]
    ) -> None:
        """Register a callback for transition completion."""
        with self._lock:
            self._on_transition_complete.append(callback)

    def remove_transition_callback(
        self, callback: Callable[[MixSnapshot], None]
    ) -> bool:
        """Remove a transition completion callback."""
        with self._lock:
            if callback in self._on_transition_complete:
                self._on_transition_complete.remove(callback)
                return True
        return False

    # =========================================================================
    # Predefined Snapshots
    # =========================================================================

    def create_preset_snapshots(self) -> None:
        """Create common preset snapshots."""
        presets = {
            "default": {},  # All defaults
            "combat": {
                "sfx": {"volume_linear": 1.2},
                "music": {"volume_linear": 0.8, "low_pass_freq": 8000, "low_pass_enabled": True},
                "ambient": {"volume_linear": 0.4},
            },
            "stealth": {
                "sfx": {"volume_linear": 0.6},
                "music": {"volume_linear": 0.4},
                "ambient": {"volume_linear": 1.2},
                "vo": {"volume_linear": 1.0},
            },
            "menu": {
                "sfx": {"volume_linear": 0.3},
                "music": {"volume_linear": 1.0},
                "ambient": {"volume_linear": 0.0},
                "ui": {"volume_linear": 1.0},
            },
            "cutscene": {
                "sfx": {"volume_linear": 0.7},
                "music": {"volume_linear": 0.6},
                "vo": {"volume_linear": 1.2},
                "ambient": {"volume_linear": 0.3},
            },
        }

        for name, bus_overrides in presets.items():
            # Capture current state first
            snapshot = self.capture_snapshot(name)

            # Apply overrides
            for bus_name, overrides in bus_overrides.items():
                if bus_name in snapshot.bus_states:
                    state = snapshot.bus_states[bus_name]
                    for key, value in overrides.items():
                        setattr(state, key, value)

            self._snapshots[name] = snapshot
