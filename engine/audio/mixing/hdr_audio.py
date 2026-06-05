"""
HDR (High Dynamic Range) Audio system.

HDR audio manages a wide dynamic range by using a sliding
"audibility window" that adapts to the loudest sounds.
This allows for:
- Wide dynamic range (quiet whispers to loud explosions)
- Intelligent volume adaptation
- Priority-based mixing (important sounds stay audible)
- Natural sounding volume transitions

The system analyzes loudness of all active sounds and
shifts the mix window to keep the most important sounds
in the audible range.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
from uuid import uuid4

from .config import (
    HDR_ACTIVE_THRESHOLD_DB,
    HDR_ADAPTATION_SPEED,
    HDR_CEILING_DB,
    HDR_DEFAULT_CENTER_DB,
    HDR_FLOOR_DB,
    HDR_PRIORITY_CRITICAL,
    HDR_PRIORITY_HIGH,
    HDR_PRIORITY_LOW,
    HDR_PRIORITY_NORMAL,
    HDR_WINDOW_DB,
    HDR_WINDOW_MAX_DB,
    HDR_WINDOW_MIN_DB,
    LOCK_TIMEOUT,
    LOUDNESS_UPDATE_RATE,
    MIN_VOLUME_DB,
    clamp,
    db_to_linear,
    lerp,
    linear_to_db,
)
from .mix_bus import MixBus


class HDRPriority(Enum):
    """Priority levels for HDR audio sources."""
    CRITICAL = HDR_PRIORITY_CRITICAL  # Always audible (alerts, dialogue)
    HIGH = HDR_PRIORITY_HIGH          # Usually audible (important SFX)
    NORMAL = HDR_PRIORITY_NORMAL      # Standard sounds
    LOW = HDR_PRIORITY_LOW            # Ambient, background


@dataclass
class AudioSource:
    """
    An audio source in the HDR system.

    Tracks the loudness and priority of a sound source
    for mix window adaptation.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    bus: Optional[MixBus] = None
    priority: int = HDR_PRIORITY_NORMAL
    loudness_db: float = MIN_VOLUME_DB  # Current perceived loudness
    target_loudness_db: float = MIN_VOLUME_DB  # Natural loudness before HDR
    is_active: bool = False
    is_protected: bool = False  # If True, never ducked by HDR

    def copy(self) -> AudioSource:
        """Create a copy of this source."""
        return AudioSource(
            id=self.id,
            name=self.name,
            bus=self.bus,
            priority=self.priority,
            loudness_db=self.loudness_db,
            target_loudness_db=self.target_loudness_db,
            is_active=self.is_active,
            is_protected=self.is_protected,
        )


@dataclass
class MixWindow:
    """
    The audible mix window in HDR audio.

    The window defines the dB range that maps to audible output.
    Sounds below the floor are inaudible, sounds above ceiling are clipped.
    """
    floor_db: float = HDR_FLOOR_DB
    ceiling_db: float = HDR_CEILING_DB
    window_db: float = HDR_WINDOW_DB
    center_db: float = HDR_DEFAULT_CENTER_DB  # Center of the current window

    @property
    def window_floor(self) -> float:
        """Get the current window floor (minimum audible level)."""
        return self.center_db - self.window_db / 2.0

    @property
    def window_ceiling(self) -> float:
        """Get the current window ceiling (maximum before limiting)."""
        return self.center_db + self.window_db / 2.0

    def map_level(self, input_db: float) -> float:
        """
        Map an input level through the HDR window.

        Args:
            input_db: Input level in dB.

        Returns:
            Output level in dB (mapped to output range).
        """
        if input_db <= self.window_floor:
            # Below window - silent
            return MIN_VOLUME_DB

        if input_db >= self.window_ceiling:
            # Above window - limit to ceiling
            return self.ceiling_db

        # Within window - linear mapping
        window_position = (input_db - self.window_floor) / self.window_db
        output_range = self.ceiling_db - self.floor_db
        return self.floor_db + window_position * output_range

    def contains(self, level_db: float) -> bool:
        """Check if a level is within the audible window."""
        return self.window_floor <= level_db <= self.window_ceiling

    def copy(self) -> MixWindow:
        """Create a copy of this window."""
        return MixWindow(
            floor_db=self.floor_db,
            ceiling_db=self.ceiling_db,
            window_db=self.window_db,
            center_db=self.center_db,
        )


class HDRAudioManager:
    """
    Manages HDR audio mixing.

    Features:
    - Tracks all active audio sources
    - Adapts mix window to loudest sources
    - Priority-based source management
    - Smooth window transitions

    Thread Safety:
        All operations are protected by a lock.
    """

    def __init__(self) -> None:
        """Initialize the HDR audio manager."""
        self._lock = threading.RLock()
        self._sources: dict[str, AudioSource] = {}
        self._window = MixWindow()
        self._target_center_db = HDR_DEFAULT_CENTER_DB
        self._adaptation_speed = HDR_ADAPTATION_SPEED
        self._enabled = True
        self._on_window_change: list[Callable[[MixWindow], None]] = []
        self._last_update_time = time.time()

    @property
    def window(self) -> MixWindow:
        """Get a copy of the current mix window."""
        with self._lock:
            return self._window.copy()

    @property
    def enabled(self) -> bool:
        """Check if HDR audio is enabled."""
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable HDR audio."""
        with self._lock:
            self._enabled = value

    def enable(self) -> None:
        """Enable HDR audio processing."""
        self.enabled = True

    def disable(self) -> None:
        """Disable HDR audio processing."""
        self.enabled = False

    # =========================================================================
    # Source Management
    # =========================================================================

    def register_source(
        self,
        name: str,
        bus: Optional[MixBus] = None,
        priority: int = HDR_PRIORITY_NORMAL,
        protected: bool = False,
    ) -> AudioSource:
        """
        Register an audio source for HDR management.

        Args:
            name: Source name.
            bus: Associated mix bus.
            priority: Source priority (higher = more important).
            protected: If True, source is never ducked.

        Returns:
            The registered source.
        """
        with self._lock:
            source = AudioSource(
                name=name,
                bus=bus,
                priority=priority,
                is_protected=protected,
            )
            self._sources[source.id] = source
            return source

    def unregister_source(self, source_id: str) -> bool:
        """
        Unregister an audio source.

        Args:
            source_id: ID of source to remove.

        Returns:
            True if removed.
        """
        with self._lock:
            return self._sources.pop(source_id, None) is not None

    def get_source(self, source_id: str) -> Optional[AudioSource]:
        """Get a source by ID."""
        with self._lock:
            source = self._sources.get(source_id)
            return source.copy() if source else None

    def set_source_loudness(
        self,
        source_id: str,
        loudness_db: float,
        is_active: bool = True,
    ) -> None:
        """
        Update a source's current loudness.

        Args:
            source_id: Source ID.
            loudness_db: Current loudness in dB.
            is_active: Whether the source is currently playing.
        """
        with self._lock:
            if source_id in self._sources:
                self._sources[source_id].target_loudness_db = loudness_db
                self._sources[source_id].is_active = is_active

    def set_source_priority(self, source_id: str, priority: int) -> None:
        """Update a source's priority."""
        with self._lock:
            if source_id in self._sources:
                self._sources[source_id].priority = priority

    # =========================================================================
    # Window Management
    # =========================================================================

    def set_window_size(self, window_db: float) -> None:
        """
        Set the size of the audibility window.

        Args:
            window_db: Window size in dB (typically 12-36 dB).
        """
        with self._lock:
            self._window.window_db = clamp(window_db, HDR_WINDOW_MIN_DB, HDR_WINDOW_MAX_DB)

    def set_adaptation_speed(self, speed: float) -> None:
        """
        Set how quickly the window adapts.

        Args:
            speed: Adaptation speed in seconds (lower = faster).
        """
        with self._lock:
            self._adaptation_speed = max(0.01, speed)

    def force_window_center(self, center_db: float) -> None:
        """
        Force the window to a specific center position.

        Args:
            center_db: New window center in dB.
        """
        with self._lock:
            self._window.center_db = center_db
            self._target_center_db = center_db

    # =========================================================================
    # Update
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """
        Update the HDR system.

        Analyzes all sources and adapts the mix window.

        Args:
            delta_time: Time since last update in seconds.
        """
        with self._lock:
            if not self._enabled:
                return

            # Find the loudest active sources by priority
            loudest_by_priority: dict[int, float] = {}

            for source in self._sources.values():
                if not source.is_active:
                    continue

                priority = source.priority
                loudness = source.target_loudness_db

                if priority not in loudest_by_priority:
                    loudest_by_priority[priority] = loudness
                else:
                    loudest_by_priority[priority] = max(
                        loudest_by_priority[priority], loudness
                    )

            if not loudest_by_priority:
                return

            # Calculate target window center based on priority-weighted loudness
            total_weight = 0.0
            weighted_loudness = 0.0

            for priority, loudness in loudest_by_priority.items():
                # Higher priority = more influence on window position
                weight = priority / 100.0
                total_weight += weight
                weighted_loudness += loudness * weight

            if total_weight > 0:
                target_loudness = weighted_loudness / total_weight

                # Position window so loudest sound is near top
                self._target_center_db = target_loudness - self._window.window_db / 4.0

                # Clamp to valid range
                min_center = self._window.floor_db + self._window.window_db / 2.0
                max_center = self._window.ceiling_db - self._window.window_db / 2.0
                self._target_center_db = clamp(
                    self._target_center_db, min_center, max_center
                )

            # Smoothly adapt window center
            if self._adaptation_speed > 0:
                rate = delta_time / self._adaptation_speed
                self._window.center_db = lerp(
                    self._window.center_db,
                    self._target_center_db,
                    min(1.0, rate),
                )
            else:
                self._window.center_db = self._target_center_db

            # Update source output levels
            for source in self._sources.values():
                if source.is_protected:
                    source.loudness_db = source.target_loudness_db
                else:
                    source.loudness_db = self._window.map_level(
                        source.target_loudness_db
                    )

            # Notify callbacks
            callbacks = list(self._on_window_change)
            window = self._window.copy()

        for callback in callbacks:
            try:
                callback(window)
            except Exception:
                pass

    def get_output_level(self, source_id: str) -> float:
        """
        Get the HDR-processed output level for a source.

        Args:
            source_id: Source ID.

        Returns:
            Output level in dB after HDR processing.
        """
        with self._lock:
            source = self._sources.get(source_id)
            if source is None:
                return MIN_VOLUME_DB
            return source.loudness_db

    def get_gain_adjustment(self, source_id: str) -> float:
        """
        Get the gain adjustment applied by HDR.

        Args:
            source_id: Source ID.

        Returns:
            Gain adjustment in dB (positive or negative).
        """
        with self._lock:
            source = self._sources.get(source_id)
            if source is None:
                return 0.0
            return source.loudness_db - source.target_loudness_db

    # =========================================================================
    # Batch Level Updates
    # =========================================================================

    def update_source_levels(self, levels: dict[str, float]) -> None:
        """
        Update multiple source levels at once.

        Args:
            levels: Dictionary of source_id -> loudness_db.
        """
        with self._lock:
            for source_id, loudness in levels.items():
                if source_id in self._sources:
                    self._sources[source_id].target_loudness_db = loudness
                    self._sources[source_id].is_active = loudness > HDR_ACTIVE_THRESHOLD_DB

    def analyze_bus_levels(self, bus_levels: dict[str, float]) -> None:
        """
        Update source levels based on bus levels.

        Args:
            bus_levels: Dictionary of bus_id -> level_db.
        """
        with self._lock:
            for source in self._sources.values():
                if source.bus and source.bus.id in bus_levels:
                    source.target_loudness_db = bus_levels[source.bus.id]
                    source.is_active = source.target_loudness_db > HDR_ACTIVE_THRESHOLD_DB

    # =========================================================================
    # Queries
    # =========================================================================

    def get_active_sources(self) -> list[AudioSource]:
        """Get all active sources sorted by priority."""
        with self._lock:
            active = [s.copy() for s in self._sources.values() if s.is_active]
            return sorted(active, key=lambda s: -s.priority)

    def get_loudest_source(self) -> Optional[AudioSource]:
        """Get the loudest currently active source."""
        with self._lock:
            active = [s for s in self._sources.values() if s.is_active]
            if not active:
                return None
            loudest = max(active, key=lambda s: s.target_loudness_db)
            return loudest.copy()

    def is_in_window(self, level_db: float) -> bool:
        """Check if a level falls within the current audibility window."""
        with self._lock:
            return self._window.contains(level_db)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_window_change(self, callback: Callable[[MixWindow], None]) -> None:
        """Register a callback for window changes."""
        with self._lock:
            self._on_window_change.append(callback)

    def remove_callback(self, callback: Callable[[MixWindow], None]) -> bool:
        """Remove a window change callback."""
        with self._lock:
            if callback in self._on_window_change:
                self._on_window_change.remove(callback)
                return True
        return False

    # =========================================================================
    # State Management
    # =========================================================================

    def reset(self) -> None:
        """Reset the HDR system to defaults."""
        with self._lock:
            self._window = MixWindow()
            self._target_center_db = HDR_DEFAULT_CENTER_DB
            for source in self._sources.values():
                source.is_active = False
                source.loudness_db = MIN_VOLUME_DB

    def clear(self) -> None:
        """Remove all sources and reset."""
        with self._lock:
            self._sources.clear()
            self.reset()

    def get_state(self) -> dict:
        """Get current state for debugging."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "window": {
                    "center_db": self._window.center_db,
                    "window_db": self._window.window_db,
                    "floor": self._window.window_floor,
                    "ceiling": self._window.window_ceiling,
                },
                "target_center_db": self._target_center_db,
                "adaptation_speed": self._adaptation_speed,
                "sources": {
                    id: {
                        "name": s.name,
                        "priority": s.priority,
                        "target_db": s.target_loudness_db,
                        "output_db": s.loudness_db,
                        "active": s.is_active,
                        "protected": s.is_protected,
                    }
                    for id, s in self._sources.items()
                },
            }

    def __repr__(self) -> str:
        with self._lock:
            active = sum(1 for s in self._sources.values() if s.is_active)
            return (
                f"HDRAudioManager(sources={len(self._sources)}, active={active}, "
                f"window_center={self._window.center_db:.1f}dB)"
            )
