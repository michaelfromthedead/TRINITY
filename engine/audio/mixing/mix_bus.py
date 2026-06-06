"""
Mix Bus implementation for audio routing hierarchy.

This module provides the MixBus class which represents a node in the
audio mixing hierarchy. Buses can have:
- Parent-child relationships (hierarchical routing)
- Volume, pitch, and filter controls
- Mute/solo states
- Aux sends for parallel processing

Bus Types:
- Master: Final output bus (no parent)
- Category: Top-level grouping (SFX, Music, VO)
- Sub-bus: Specialized grouping (Footsteps, Weapons)
- Aux/Send: Effect returns (Reverb, Delay)
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from ..dsp.dsp_graph import DSPChain
from ..dsp.filters import HighPassFilter, LowPassFilter
from .config import (
    DEFAULT_BUS_VOLUME,
    DEFAULT_HIGH_PASS,
    DEFAULT_LOW_PASS,
    DEFAULT_PITCH,
    FILTER_Q,
    LOCK_TIMEOUT,
    MAX_FILTER_FREQ,
    MAX_PITCH,
    MAX_VOLUME_DB,
    MIN_FILTER_FREQ,
    MIN_PITCH,
    MIN_VOLUME_DB,
    MIXER_BUFFER_SIZE,
    MIXER_NUM_CHANNELS,
    clamp,
    db_to_linear,
    linear_to_db,
)


class BusType(Enum):
    """Types of mix buses in the hierarchy."""
    MASTER = "master"
    CATEGORY = "category"
    SUB = "sub"
    AUX = "aux"


class BusStatus(Enum):
    """Runtime status of a bus."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    SUSPENDED = "suspended"
    STOPPING = "stopping"


class FilterState(Enum):
    """Filter enabled state."""
    DISABLED = "disabled"
    LOW_PASS = "low_pass"
    HIGH_PASS = "high_pass"
    BOTH = "both"


@dataclass
class FilterParams:
    """State of bus filters (low-pass and high-pass)."""
    low_pass_freq: float = DEFAULT_LOW_PASS
    high_pass_freq: float = DEFAULT_HIGH_PASS
    low_pass_q: float = FILTER_Q
    high_pass_q: float = FILTER_Q
    low_pass_enabled: bool = False
    high_pass_enabled: bool = False

    def reset(self) -> None:
        """Reset filters to default state."""
        self.low_pass_freq = DEFAULT_LOW_PASS
        self.high_pass_freq = DEFAULT_HIGH_PASS
        self.low_pass_q = FILTER_Q
        self.high_pass_q = FILTER_Q
        self.low_pass_enabled = False
        self.high_pass_enabled = False

    def copy(self) -> FilterParams:
        """Create a copy of this filter params."""
        return FilterParams(
            low_pass_freq=self.low_pass_freq,
            high_pass_freq=self.high_pass_freq,
            low_pass_q=self.low_pass_q,
            high_pass_q=self.high_pass_q,
            low_pass_enabled=self.low_pass_enabled,
            high_pass_enabled=self.high_pass_enabled,
        )


@dataclass
class BusStateData:
    """Complete state of a mix bus for snapshots."""
    volume_linear: float = DEFAULT_BUS_VOLUME
    pitch: float = DEFAULT_PITCH
    muted: bool = False
    soloed: bool = False
    filters: FilterParams = field(default_factory=FilterParams)

    def copy(self) -> BusStateData:
        """Create a copy of this bus state."""
        return BusStateData(
            volume_linear=self.volume_linear,
            pitch=self.pitch,
            muted=self.muted,
            soloed=self.soloed,
            filters=self.filters.copy(),
        )


class MixBus:
    """
    A mix bus in the audio routing hierarchy.

    Mix buses form a tree structure with the master bus at the root.
    Audio flows from child buses up through parents to the master output.

    Thread Safety:
        All state modifications are protected by a lock for thread-safe
        access from audio and game threads.
    """

    def __init__(
        self,
        name: str,
        bus_type: BusType = BusType.SUB,
        parent: Optional[MixBus] = None,
        volume: float = DEFAULT_BUS_VOLUME,
        pitch: float = DEFAULT_PITCH,
    ) -> None:
        """
        Initialize a mix bus.

        Args:
            name: Unique identifier for this bus
            bus_type: Type of bus (master, category, sub, aux)
            parent: Parent bus in hierarchy (None for master)
            volume: Initial volume (linear, 0.0 to 1.0+)
            pitch: Initial pitch multiplier (0.1 to 4.0)
        """
        self._id = str(uuid.uuid4())
        self._name = name
        self._bus_type = bus_type
        self._lock = threading.RLock()

        # State
        self._snapshot = BusStateData(
            volume_linear=clamp(volume, 0.0, db_to_linear(MAX_VOLUME_DB)),
            pitch=clamp(pitch, MIN_PITCH, MAX_PITCH),
        )
        self._status = BusStatus.ACTIVE

        # DSP chain for bus effects
        self._effect_chain = DSPChain()

        # Bus filters (lazy initialized for filter processing)
        self._low_pass_filter: Optional[LowPassFilter] = None
        self._high_pass_filter: Optional[HighPassFilter] = None

        # Hierarchy
        self._parent: Optional[MixBus] = None
        self._children: list[MixBus] = []

        # Accumulation buffer for audio thread
        self._acc_buffer: Optional[np.ndarray] = None
        self._acc_lock = threading.Lock()

        # Callbacks for state changes (must be before set_parent which calls _notify_change)
        self._on_change_callbacks: list[Callable[[MixBus], None]] = []

        # Set parent after all attributes initialized
        if parent is not None:
            self.set_parent(parent)

    @property
    def id(self) -> str:
        """Unique identifier for this bus."""
        return self._id

    @property
    def name(self) -> str:
        """Name of this bus."""
        return self._name

    @property
    def bus_type(self) -> BusType:
        """Type of this bus."""
        return self._bus_type

    @property
    def parent(self) -> Optional[MixBus]:
        """Parent bus in hierarchy."""
        with self._lock:
            return self._parent

    @parent.setter
    def parent(self, value: Optional[MixBus]) -> None:
        """Set parent bus."""
        self.set_parent(value)

    @property
    def children(self) -> list[MixBus]:
        """Child buses (read-only copy)."""
        with self._lock:
            return list(self._children)

    # =========================================================================
    # Volume
    # =========================================================================

    @property
    def volume(self) -> float:
        """Volume in linear scale (0.0 to ~4.0)."""
        with self._lock:
            return self._snapshot.volume_linear

    @volume.setter
    def volume(self, value: float) -> None:
        """Set volume in linear scale."""
        with self._lock:
            self._snapshot.volume_linear = clamp(value, 0.0, db_to_linear(MAX_VOLUME_DB))
            self._notify_change()

    @property
    def volume_db(self) -> float:
        """Volume in decibels (-80 to +12 dB)."""
        with self._lock:
            return linear_to_db(self._snapshot.volume_linear)

    @volume_db.setter
    def volume_db(self, value: float) -> None:
        """Set volume in decibels."""
        value = clamp(value, MIN_VOLUME_DB, MAX_VOLUME_DB)
        with self._lock:
            self._snapshot.volume_linear = db_to_linear(value)
            self._notify_change()

    @property
    def effective_volume(self) -> float:
        """
        Effective volume considering hierarchy and mute state.

        Returns:
            Combined volume from this bus and all parents (linear).
        """
        return self.get_effective_volume()

    def get_effective_volume(self) -> float:
        """
        Get the effective volume considering hierarchy and mute state.

        Returns:
            Combined volume from this bus and all parents (linear).
        """
        with self._lock:
            if self._snapshot.muted:
                return 0.0

            volume = self._snapshot.volume_linear
            parent = self._parent

        # Walk up the hierarchy (outside lock to avoid deadlock)
        while parent is not None:
            with parent._lock:
                if parent._snapshot.muted:
                    return 0.0
                volume *= parent._snapshot.volume_linear
                parent = parent._parent

        return volume

    def set_volume_db(self, db: float) -> None:
        """
        Set volume in decibels.

        Args:
            db: Volume level in dB (MIN_VOLUME_DB to MAX_VOLUME_DB).
        """
        self.volume_db = db

    def set_volume_linear(self, linear: float) -> None:
        """
        Set volume in linear scale.

        Args:
            linear: Volume level (0.0 to db_to_linear(MAX_VOLUME_DB)).
        """
        self.volume = linear

    def fade_to_volume(self, target: float, duration: float = 1.0) -> None:
        """
        Fade volume to target over duration.

        Args:
            target: Target volume (linear scale).
            duration: Fade duration in seconds.

        Note:
            Currently sets volume immediately. Smooth fading requires
            real-time update loop integration.
        """
        # For now, immediately set the volume
        # A proper implementation would schedule the fade
        self.volume = target

    # =========================================================================
    # Pitch
    # =========================================================================

    @property
    def pitch(self) -> float:
        """Pitch multiplier (0.1 to 4.0)."""
        with self._lock:
            return self._snapshot.pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        """Set pitch multiplier."""
        with self._lock:
            self._snapshot.pitch = clamp(value, MIN_PITCH, MAX_PITCH)
            self._notify_change()

    def get_effective_pitch(self) -> float:
        """
        Get the effective pitch considering hierarchy.

        Returns:
            Combined pitch from this bus and all parents.
        """
        with self._lock:
            pitch = self._snapshot.pitch
            parent = self._parent

        while parent is not None:
            with parent._lock:
                pitch *= parent._snapshot.pitch
                parent = parent._parent

        return clamp(pitch, MIN_PITCH, MAX_PITCH)

    # =========================================================================
    # Mute/Solo
    # =========================================================================

    @property
    def muted(self) -> bool:
        """Whether this bus is muted."""
        with self._lock:
            return self._snapshot.muted

    @muted.setter
    def muted(self, value: bool) -> None:
        """Set mute state."""
        with self._lock:
            self._snapshot.muted = value
            self._notify_change()

    @property
    def soloed(self) -> bool:
        """Whether this bus is soloed."""
        with self._lock:
            return self._snapshot.soloed

    @soloed.setter
    def soloed(self, value: bool) -> None:
        """Set solo state."""
        with self._lock:
            self._snapshot.soloed = value
            self._notify_change()

    def toggle_mute(self) -> bool:
        """Toggle mute state. Returns new state."""
        with self._lock:
            self._snapshot.muted = not self._snapshot.muted
            self._notify_change()
            return self._snapshot.muted

    def toggle_solo(self) -> bool:
        """Toggle solo state. Returns new state."""
        with self._lock:
            self._snapshot.soloed = not self._snapshot.soloed
            self._notify_change()
            return self._snapshot.soloed

    # =========================================================================
    # Filters
    # =========================================================================

    @property
    def filters(self) -> FilterParams:
        """Get a copy of the filter state."""
        with self._lock:
            return self._snapshot.filters.copy()

    def set_low_pass(
        self,
        frequency: float,
        q: float = FILTER_Q,
        enabled: bool = True,
    ) -> None:
        """
        Set low-pass filter parameters.

        Args:
            frequency: Cutoff frequency in Hz (20 to 20000)
            q: Filter Q factor (resonance)
            enabled: Whether filter is active
        """
        with self._lock:
            self._snapshot.filters.low_pass_freq = clamp(
                frequency, MIN_FILTER_FREQ, MAX_FILTER_FREQ
            )
            self._snapshot.filters.low_pass_q = max(0.1, q)
            self._snapshot.filters.low_pass_enabled = enabled
            self._notify_change()

    def set_high_pass(
        self,
        frequency: float,
        q: float = FILTER_Q,
        enabled: bool = True,
    ) -> None:
        """
        Set high-pass filter parameters.

        Args:
            frequency: Cutoff frequency in Hz (20 to 20000)
            q: Filter Q factor (resonance)
            enabled: Whether filter is active
        """
        with self._lock:
            self._snapshot.filters.high_pass_freq = clamp(
                frequency, MIN_FILTER_FREQ, MAX_FILTER_FREQ
            )
            self._snapshot.filters.high_pass_q = max(0.1, q)
            self._snapshot.filters.high_pass_enabled = enabled
            self._notify_change()

    def reset_filters(self) -> None:
        """Reset all filters to default state."""
        with self._lock:
            self._snapshot.filters.reset()
            self._notify_change()

    @property
    def low_pass_freq(self) -> float:
        """Low-pass filter cutoff frequency in Hz."""
        with self._lock:
            return self._snapshot.filters.low_pass_freq

    @low_pass_freq.setter
    def low_pass_freq(self, value: float) -> None:
        """Set low-pass filter cutoff frequency."""
        with self._lock:
            self._snapshot.filters.low_pass_freq = clamp(
                value, MIN_FILTER_FREQ, MAX_FILTER_FREQ
            )
            self._notify_change()

    @property
    def high_pass_freq(self) -> float:
        """High-pass filter cutoff frequency in Hz."""
        with self._lock:
            return self._snapshot.filters.high_pass_freq

    @high_pass_freq.setter
    def high_pass_freq(self, value: float) -> None:
        """Set high-pass filter cutoff frequency."""
        with self._lock:
            self._snapshot.filters.high_pass_freq = clamp(
                value, MIN_FILTER_FREQ, MAX_FILTER_FREQ
            )
            self._notify_change()

    @property
    def filter_state(self) -> FilterState:
        """Get the current filter enabled state."""
        with self._lock:
            lp = self._snapshot.filters.low_pass_enabled
            hp = self._snapshot.filters.high_pass_enabled
            if lp and hp:
                return FilterState.BOTH
            elif lp:
                return FilterState.LOW_PASS
            elif hp:
                return FilterState.HIGH_PASS
            else:
                return FilterState.DISABLED

    def enable_low_pass(self, frequency: float, q: float = FILTER_Q) -> None:
        """
        Enable the low-pass filter.

        Args:
            frequency: Cutoff frequency in Hz.
            q: Filter Q factor.
        """
        self.set_low_pass(frequency, q, enabled=True)

    def enable_high_pass(self, frequency: float, q: float = FILTER_Q) -> None:
        """
        Enable the high-pass filter.

        Args:
            frequency: Cutoff frequency in Hz.
            q: Filter Q factor.
        """
        self.set_high_pass(frequency, q, enabled=True)

    def disable_filters(self) -> None:
        """Disable both filters."""
        with self._lock:
            self._snapshot.filters.low_pass_enabled = False
            self._snapshot.filters.high_pass_enabled = False
            self._notify_change()

    # =========================================================================
    # DSP Chain
    # =========================================================================

    @property
    def effect_chain(self) -> DSPChain:
        """Get the DSP chain for bus effects."""
        return self._effect_chain

    def has_effects(self) -> bool:
        """Check if the bus has any effects in its DSP chain."""
        return len(self._effect_chain.nodes) > 0

    # =========================================================================
    # Audio Processing
    # =========================================================================

    def _ensure_acc_buffer(self, num_samples: int) -> None:
        """
        Ensure the accumulation buffer is allocated and sized correctly.

        Args:
            num_samples: Required number of samples per channel.
        """
        with self._acc_lock:
            if self._acc_buffer is None or self._acc_buffer.shape[1] < num_samples:
                self._acc_buffer = np.zeros(
                    (MIXER_NUM_CHANNELS, num_samples), dtype=np.float32
                )

    def clear_acc_buffer(self, num_samples: int) -> None:
        """
        Clear (zero) the accumulation buffer.

        Must be called before accumulating new audio data each tick.

        Args:
            num_samples: Number of samples to clear.
        """
        self._ensure_acc_buffer(num_samples)
        with self._acc_lock:
            self._acc_buffer[:, :num_samples] = 0.0

    def accumulate(self, samples: np.ndarray, num_samples: int) -> None:
        """
        Accumulate audio samples into the bus buffer.

        Args:
            samples: Audio samples to add (channels, samples) float32.
                Mono (1, N) is broadcast to stereo.
            num_samples: Number of samples to accumulate.
        """
        self._ensure_acc_buffer(num_samples)
        with self._acc_lock:
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            if samples.shape[0] == 1 and MIXER_NUM_CHANNELS > 1:
                samples_bc = np.broadcast_to(samples[0:1, :num_samples], (MIXER_NUM_CHANNELS, num_samples))
                self._acc_buffer[:, :num_samples] += samples_bc
            else:
                channels = min(samples.shape[0], MIXER_NUM_CHANNELS)
                self._acc_buffer[:channels, :num_samples] += samples[:channels, :num_samples]

    def write_output(self, samples: np.ndarray, num_samples: int) -> None:
        """
        Write audio samples into the accumulation buffer for output.

        This is called by child buses or sources to feed audio into this bus.
        Delegates to accumulate() for the actual buffer write.

        Args:
            samples: Audio samples to write (channels, samples) float32.
            num_samples: Number of samples to write.
        """
        self.accumulate(samples, num_samples)

    def read_acc_buffer(self, num_samples: int) -> np.ndarray:
        """
        Read a copy of the accumulation buffer.

        Returns a copy to avoid race conditions.

        Args:
            num_samples: Number of samples to read.

        Returns:
            Copy of buffer data (MIXER_NUM_CHANNELS, num_samples) float32.
        """
        with self._acc_lock:
            if self._acc_buffer is None:
                return np.zeros((MIXER_NUM_CHANNELS, num_samples), dtype=np.float32)
            available = min(num_samples, self._acc_buffer.shape[1])
            return self._acc_buffer[:, :available].copy()

    def process_audio(self, num_samples: int) -> np.ndarray:
        """
        Process the accumulated audio through volume, mute, and effects.

        Returns a processed copy without modifying the accumulation buffer.
        If muted, returns silence.

        Args:
            num_samples: Number of samples to process.

        Returns:
            Processed audio (MIXER_NUM_CHANNELS, num_samples) float32, clipped to [-1, 1].
        """
        with self._lock:
            if self._snapshot.muted:
                return np.zeros((MIXER_NUM_CHANNELS, num_samples), dtype=np.float32)
            volume = self._snapshot.volume_linear
            lp_enabled = self._snapshot.filters.low_pass_enabled
            lp_freq = self._snapshot.filters.low_pass_freq
            lp_q = self._snapshot.filters.low_pass_q
            hp_enabled = self._snapshot.filters.high_pass_enabled
            hp_freq = self._snapshot.filters.high_pass_freq
            hp_q = self._snapshot.filters.high_pass_q

        raw = self.read_acc_buffer(num_samples)

        # Apply volume
        output = raw * volume

        # Apply bus low-pass filter
        if lp_enabled:
            if self._low_pass_filter is None:
                self._low_pass_filter = LowPassFilter(
                    cutoff=lp_freq,
                    q=lp_q,
                    num_channels=MIXER_NUM_CHANNELS,
                )
            else:
                self._low_pass_filter.cutoff = lp_freq
                self._low_pass_filter.q = lp_q
            filtered = np.zeros_like(output)
            self._low_pass_filter.process_block(output, filtered)
            output = filtered

        # Apply bus high-pass filter
        if hp_enabled:
            if self._high_pass_filter is None:
                self._high_pass_filter = HighPassFilter(
                    cutoff=hp_freq,
                    q=hp_q,
                    num_channels=MIXER_NUM_CHANNELS,
                )
            else:
                self._high_pass_filter.cutoff = hp_freq
                self._high_pass_filter.q = hp_q
            filtered = np.zeros_like(output)
            self._high_pass_filter.process_block(output, filtered)
            output = filtered

        # Apply DSP chain effects if any
        if self.has_effects():
            effect_output = np.zeros_like(output)
            try:
                self._effect_chain.process_block(output, effect_output)
                output = effect_output
            except Exception:
                pass  # Silently skip DSP errors, use raw output

        # Hard clip to [-1.0, 1.0]
        output = np.clip(output, -1.0, 1.0)

        return output

    # =========================================================================
    # Hierarchy
    # =========================================================================

    def set_parent(self, parent: Optional[MixBus]) -> None:
        """
        Set the parent bus.

        Args:
            parent: New parent bus, or None to remove parent.

        Raises:
            ValueError: If setting parent would create a cycle.
        """
        if parent is self:
            raise ValueError("A bus cannot be its own parent")

        if parent is not None and self._would_create_cycle(parent):
            raise ValueError("Setting this parent would create a cycle")

        with self._lock:
            old_parent = self._parent
            self._parent = parent

        # Update old parent's children list
        if old_parent is not None:
            with old_parent._lock:
                if self in old_parent._children:
                    old_parent._children.remove(self)

        # Update new parent's children list
        if parent is not None:
            with parent._lock:
                if self not in parent._children:
                    parent._children.append(self)

        self._notify_change()

    def _would_create_cycle(self, proposed_parent: MixBus) -> bool:
        """Check if setting proposed_parent would create a cycle."""
        current = proposed_parent
        while current is not None:
            if current is self:
                return True
            current = current.parent
        return False

    def add_child(self, child: MixBus) -> None:
        """
        Add a child bus.

        Args:
            child: Bus to add as child.
        """
        child.set_parent(self)

    def remove_child(self, child: MixBus) -> bool:
        """
        Remove a child bus.

        Args:
            child: Bus to remove.

        Returns:
            True if child was removed, False if not found.
        """
        with self._lock:
            if child in self._children:
                self._children.remove(child)
                with child._lock:
                    child._parent = None
                return True
        return False

    def get_ancestors(self) -> list[MixBus]:
        """Get all ancestor buses from parent to master."""
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def get_descendants(self) -> list[MixBus]:
        """Get all descendant buses recursively."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

    # =========================================================================
    # State Management
    # =========================================================================

    @property
    def state(self) -> BusStatus:
        """Get the current bus status (active/suspended/stopping)."""
        with self._lock:
            return self._status

    @state.setter
    def state(self, value: BusStatus) -> None:
        """Set the bus status."""
        with self._lock:
            self._status = value
            self._notify_change()

    def get_snapshot(self) -> BusStateData:
        """Get a copy of the current bus state snapshot."""
        with self._lock:
            return self._snapshot.copy()

    def set_snapshot(self, snapshot: BusStateData) -> None:
        """
        Set the bus state from a snapshot.

        Args:
            snapshot: New state to apply.
        """
        with self._lock:
            self._snapshot = snapshot.copy()
            self._notify_change()

    def reset(self) -> None:
        """Reset bus to default state."""
        with self._lock:
            self._snapshot = BusStateData()
            self._status = BusStatus.ACTIVE
            self._notify_change()

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_change(self, callback: Callable[[MixBus], None]) -> None:
        """
        Register a callback for state changes.

        Args:
            callback: Function to call when bus state changes.
        """
        with self._lock:
            self._on_change_callbacks.append(callback)

    def remove_callback(self, callback: Callable[[MixBus], None]) -> bool:
        """
        Remove a change callback.

        Args:
            callback: Callback to remove.

        Returns:
            True if callback was removed.
        """
        with self._lock:
            if callback in self._on_change_callbacks:
                self._on_change_callbacks.remove(callback)
                return True
        return False

    def _notify_change(self) -> None:
        """Notify all registered callbacks of a state change."""
        # Copy callbacks to avoid holding lock during callback execution
        with self._lock:
            callbacks = list(self._on_change_callbacks)

        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                pass  # Silently ignore callback errors

    # =========================================================================
    # String Representation
    # =========================================================================

    # =========================================================================
    # Aux Sends (convenience methods that wrap internal send list)
    # =========================================================================

    def add_send(
        self,
        target: MixBus,
        level: float = 1.0,
        pre_fader: bool = False,
    ) -> None:
        """
        Add an aux send to a target bus.

        Args:
            target: Target bus for the send.
            level: Send level (linear, 0.0 to 1.0+).
            pre_fader: If True, send before fader.
        """
        from .bus_routing import AuxSend, RoutingMode

        with self._lock:
            if not hasattr(self, '_aux_sends'):
                self._aux_sends: list[AuxSend] = []

            send = AuxSend(
                source_bus=self,
                target_bus=target,
                send_level_db=linear_to_db(level) if level > 0 else -80.0,
                mode=RoutingMode.PRE_FADER if pre_fader else RoutingMode.POST_FADER,
                enabled=True,
            )
            self._aux_sends.append(send)

    def get_sends(self) -> list:
        """
        Get all aux sends from this bus.

        Returns:
            List of AuxSend objects.
        """
        with self._lock:
            if not hasattr(self, '_aux_sends'):
                self._aux_sends = []
            return list(self._aux_sends)

    def set_send_level(self, target: MixBus, level: float) -> bool:
        """
        Set the send level to a target bus.

        Args:
            target: Target bus.
            level: New send level (linear).

        Returns:
            True if send was found and updated.
        """
        with self._lock:
            if not hasattr(self, '_aux_sends'):
                return False

            for send in self._aux_sends:
                if send.target_bus is target or getattr(send, 'target', None) is target:
                    send.send_level_db = linear_to_db(level) if level > 0 else -80.0
                    send.level = level  # Also set linear level for compatibility
                    return True
            return False

    def remove_send(self, target: MixBus) -> bool:
        """
        Remove an aux send to a target bus.

        Args:
            target: Target bus to remove send for.

        Returns:
            True if send was found and removed.
        """
        with self._lock:
            if not hasattr(self, '_aux_sends'):
                return False

            for send in self._aux_sends:
                if send.target_bus is target or getattr(send, 'target', None) is target:
                    self._aux_sends.remove(send)
                    return True
            return False

    def __repr__(self) -> str:
        return (
            f"MixBus(name={self._name!r}, type={self._bus_type.value}, "
            f"volume={self.volume:.2f}, muted={self.muted})"
        )

    def __str__(self) -> str:
        return f"{self._bus_type.value}:{self._name}"


def create_default_hierarchy() -> dict[str, MixBus]:
    """
    Create the default bus hierarchy.

    Returns:
        Dictionary mapping bus names to MixBus instances.

    Hierarchy:
        master
        ├── sfx
        │   ├── footsteps
        │   ├── weapons
        │   └── impacts
        ├── music
        │   ├── combat
        │   └── exploration
        ├── vo
        │   ├── dialogue
        │   └── barks
        ├── ambient
        └── ui
    """
    buses: dict[str, MixBus] = {}

    # Master bus
    buses["master"] = MixBus("master", BusType.MASTER)

    # Category buses
    for category in ["sfx", "music", "vo", "ambient", "ui"]:
        buses[category] = MixBus(
            category,
            BusType.CATEGORY,
            parent=buses["master"],
        )

    # Sub-buses for SFX
    for sub in ["footsteps", "weapons", "impacts"]:
        buses[sub] = MixBus(sub, BusType.SUB, parent=buses["sfx"])

    # Sub-buses for Music
    for sub in ["combat", "exploration"]:
        buses[sub] = MixBus(sub, BusType.SUB, parent=buses["music"])

    # Sub-buses for VO
    for sub in ["dialogue", "barks"]:
        buses[sub] = MixBus(sub, BusType.SUB, parent=buses["vo"])

    return buses


# Alias: BusState now refers to BusStateData for snapshot compatibility
# (The enum is now BusStatus)
BusState = BusStateData
