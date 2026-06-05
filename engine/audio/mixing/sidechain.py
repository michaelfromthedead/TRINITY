"""
Sidechain Compression for dynamic audio mixing.

Sidechain compression uses a key input signal to control
compression on a different target signal. Common uses:
- Kick drum pumping bass
- VO ducking music (as an alternative to simple ducking)
- Sound design effects

The compressor envelope follows the key signal's level.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
from uuid import uuid4

from .config import (
    LOCK_TIMEOUT,
    MIN_VOLUME_DB,
    SIDECHAIN_ATTACK_MS,
    SIDECHAIN_KNEE_DB,
    SIDECHAIN_MAKEUP_GAIN_DB,
    SIDECHAIN_RATIO,
    SIDECHAIN_RELEASE_MS,
    SIDECHAIN_THRESHOLD_DB,
    clamp,
    db_to_linear,
    linear_to_db,
)
from .mix_bus import MixBus


class CompressorState(Enum):
    """State of the compressor envelope."""
    IDLE = "idle"
    ATTACKING = "attacking"
    COMPRESSING = "compressing"
    RELEASING = "releasing"


@dataclass
class SidechainConfig:
    """Configuration for a sidechain compressor."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    key_bus: Optional[MixBus] = None      # Signal that triggers compression
    target_bus: Optional[MixBus] = None   # Signal that gets compressed
    threshold_db: float = SIDECHAIN_THRESHOLD_DB
    ratio: float = SIDECHAIN_RATIO        # 4:1 means 4dB in = 1dB out above threshold
    attack_ms: float = SIDECHAIN_ATTACK_MS
    release_ms: float = SIDECHAIN_RELEASE_MS
    knee_db: float = SIDECHAIN_KNEE_DB    # Soft knee width
    makeup_gain_db: float = SIDECHAIN_MAKEUP_GAIN_DB
    enabled: bool = True
    mix: float = 1.0  # Wet/dry mix (0.0 = no compression, 1.0 = full compression)

    def copy(self) -> SidechainConfig:
        """Create a copy of this config."""
        return SidechainConfig(
            id=self.id,
            name=self.name,
            key_bus=self.key_bus,
            target_bus=self.target_bus,
            threshold_db=self.threshold_db,
            ratio=self.ratio,
            attack_ms=self.attack_ms,
            release_ms=self.release_ms,
            knee_db=self.knee_db,
            makeup_gain_db=self.makeup_gain_db,
            enabled=self.enabled,
            mix=self.mix,
        )


class SidechainCompressor:
    """
    Sidechain compressor with envelope follower.

    Uses the key signal level to determine gain reduction
    applied to the target signal.
    """

    def __init__(self, config: SidechainConfig) -> None:
        """
        Initialize the sidechain compressor.

        Args:
            config: Compressor configuration.
        """
        self._config = config.copy()
        self._state = CompressorState.IDLE
        self._current_gain_reduction_db = 0.0
        self._target_gain_reduction_db = 0.0
        self._key_level_db = MIN_VOLUME_DB
        self._envelope_level = 0.0  # Smoothed key level
        self._last_update = time.time()

    @property
    def config(self) -> SidechainConfig:
        """Get the compressor configuration."""
        return self._config

    @property
    def state(self) -> CompressorState:
        """Get the current compressor state."""
        return self._state

    @property
    def gain_reduction_db(self) -> float:
        """Get current gain reduction in dB (negative value)."""
        return self._current_gain_reduction_db

    @property
    def gain_reduction_linear(self) -> float:
        """Get current gain reduction as linear multiplier."""
        return db_to_linear(self._current_gain_reduction_db)

    @property
    def output_gain_linear(self) -> float:
        """Get total output gain including makeup gain."""
        total_db = self._current_gain_reduction_db + self._config.makeup_gain_db
        return db_to_linear(total_db)

    @property
    def is_compressing(self) -> bool:
        """Check if compression is currently active."""
        return self._current_gain_reduction_db < -0.1

    def set_key_level(self, level_db: float) -> None:
        """
        Set the current key signal level.

        Args:
            level_db: Key signal level in dB.
        """
        self._key_level_db = level_db

    def _calculate_gain_reduction(self, input_db: float) -> float:
        """
        Calculate gain reduction for a given input level.

        Uses soft knee if configured.

        Args:
            input_db: Input level in dB.

        Returns:
            Gain reduction in dB (negative or zero).
        """
        threshold = self._config.threshold_db
        ratio = self._config.ratio
        knee = self._config.knee_db

        if ratio <= 1.0:
            return 0.0

        # Calculate gain reduction with soft knee
        if knee > 0.0:
            # Soft knee region
            knee_start = threshold - knee / 2.0
            knee_end = threshold + knee / 2.0

            if input_db <= knee_start:
                return 0.0
            elif input_db >= knee_end:
                # Above knee - full compression
                overshoot = input_db - threshold
                reduction = overshoot * (1.0 - 1.0 / ratio)
                return -reduction
            else:
                # In knee - gradual compression
                knee_factor = (input_db - knee_start) / knee
                effective_ratio = 1.0 + (ratio - 1.0) * knee_factor
                overshoot = input_db - knee_start
                reduction = overshoot * (1.0 - 1.0 / effective_ratio) * knee_factor
                return -reduction
        else:
            # Hard knee
            if input_db <= threshold:
                return 0.0
            overshoot = input_db - threshold
            reduction = overshoot * (1.0 - 1.0 / ratio)
            return -reduction

    def update(self, delta_time: float) -> float:
        """
        Update the compressor envelope.

        Args:
            delta_time: Time since last update in seconds.

        Returns:
            Current output gain multiplier (including compression and makeup).
        """
        if not self._config.enabled:
            self._current_gain_reduction_db = 0.0
            return 1.0

        # Calculate target gain reduction based on key level
        self._target_gain_reduction_db = self._calculate_gain_reduction(
            self._key_level_db
        )

        # Determine envelope direction
        target_abs = abs(self._target_gain_reduction_db)
        current_abs = abs(self._current_gain_reduction_db)

        if target_abs > current_abs:
            # Need more compression - use attack
            self._state = CompressorState.ATTACKING
            attack_time = self._config.attack_ms / 1000.0
            if attack_time > 0:
                rate = delta_time / attack_time
                self._current_gain_reduction_db += (
                    (self._target_gain_reduction_db - self._current_gain_reduction_db)
                    * min(1.0, rate * 4)  # Faster convergence
                )
            else:
                self._current_gain_reduction_db = self._target_gain_reduction_db

        elif target_abs < current_abs:
            # Less compression needed - use release
            self._state = CompressorState.RELEASING
            release_time = self._config.release_ms / 1000.0
            if release_time > 0:
                rate = delta_time / release_time
                self._current_gain_reduction_db += (
                    (self._target_gain_reduction_db - self._current_gain_reduction_db)
                    * min(1.0, rate * 4)
                )
            else:
                self._current_gain_reduction_db = self._target_gain_reduction_db

        else:
            if abs(self._current_gain_reduction_db) > 0.1:
                self._state = CompressorState.COMPRESSING
            else:
                self._state = CompressorState.IDLE

        # Apply mix
        final_reduction = self._current_gain_reduction_db * self._config.mix
        total_gain = final_reduction + self._config.makeup_gain_db

        return db_to_linear(total_gain)

    def reset(self) -> None:
        """Reset the compressor to initial state."""
        self._state = CompressorState.IDLE
        self._current_gain_reduction_db = 0.0
        self._target_gain_reduction_db = 0.0
        self._key_level_db = MIN_VOLUME_DB

    def get_stats(self) -> dict:
        """Get compressor statistics."""
        return {
            "state": self._state.value,
            "key_level_db": self._key_level_db,
            "gain_reduction_db": self._current_gain_reduction_db,
            "target_reduction_db": self._target_gain_reduction_db,
            "is_compressing": self.is_compressing,
        }


class SidechainManager:
    """
    Manages multiple sidechain compressors.

    Features:
    - Multiple compressor instances
    - Automatic key level analysis
    - Per-bus compression tracking

    Thread Safety:
        All operations are protected by a lock.
    """

    def __init__(self) -> None:
        """Initialize the sidechain manager."""
        self._lock = threading.RLock()
        self._compressors: dict[str, SidechainCompressor] = {}
        self._bus_gains: dict[str, float] = {}  # bus_id -> total gain linear
        self._on_compression_change: list[Callable[[MixBus, float], None]] = []

    # =========================================================================
    # Compressor Management
    # =========================================================================

    def create(
        self,
        key_source: str,
        target: str,
        ratio: float = SIDECHAIN_RATIO,
        threshold_db: float = SIDECHAIN_THRESHOLD_DB,
        attack_ms: float = SIDECHAIN_ATTACK_MS,
        release_ms: float = SIDECHAIN_RELEASE_MS,
    ) -> str:
        """
        Create a sidechain compressor with string-based bus names.

        Args:
            key_source: Name of the key input bus.
            target: Name of the target bus.
            ratio: Compression ratio.
            threshold_db: Compression threshold in dB.
            attack_ms: Attack time in ms.
            release_ms: Release time in ms.

        Returns:
            ID of the created compressor.
        """
        config = SidechainConfig(
            name=f"{key_source}_to_{target}",
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )
        self.create_compressor(config)
        return config.id

    def remove(self, compressor_id: str) -> bool:
        """
        Remove a compressor by ID.

        Args:
            compressor_id: ID of compressor to remove.

        Returns:
            True if removed.
        """
        return self.remove_compressor(compressor_id)

    def create_compressor(self, config: SidechainConfig) -> SidechainCompressor:
        """
        Create a new sidechain compressor.

        Args:
            config: Compressor configuration.

        Returns:
            The created compressor.
        """
        with self._lock:
            compressor = SidechainCompressor(config)
            self._compressors[config.id] = compressor
            return compressor

    def remove_compressor(self, compressor_id: str) -> bool:
        """
        Remove a compressor.

        Args:
            compressor_id: ID of compressor to remove.

        Returns:
            True if removed.
        """
        with self._lock:
            return self._compressors.pop(compressor_id, None) is not None

    def get_compressor(self, compressor_id: str) -> Optional[SidechainCompressor]:
        """
        Get a compressor by ID.

        Args:
            compressor_id: ID of compressor.

        Returns:
            Compressor if found.
        """
        with self._lock:
            return self._compressors.get(compressor_id)

    def get_compressors_for_target(
        self, bus: MixBus
    ) -> list[SidechainCompressor]:
        """Get all compressors affecting a target bus."""
        with self._lock:
            return [
                c for c in self._compressors.values()
                if c.config.target_bus is bus
            ]

    def get_compressors_with_key(
        self, bus: MixBus
    ) -> list[SidechainCompressor]:
        """Get all compressors using a bus as key input."""
        with self._lock:
            return [
                c for c in self._compressors.values()
                if c.config.key_bus is bus
            ]

    # =========================================================================
    # Update
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """
        Update all compressors.

        Args:
            delta_time: Time since last update in seconds.
        """
        with self._lock:
            self._bus_gains.clear()

            for compressor in self._compressors.values():
                if not compressor.config.enabled:
                    continue

                gain = compressor.update(delta_time)

                # Track gain for target bus
                target = compressor.config.target_bus
                if target is not None:
                    current = self._bus_gains.get(target.id, 1.0)
                    # Multiply gains (stacking compressors)
                    self._bus_gains[target.id] = current * gain

            # Copy for callback notification
            changes = dict(self._bus_gains)
            callbacks = list(self._on_compression_change)
            compressors = list(self._compressors.values())

        # Notify callbacks outside lock
        for bus_id, gain in changes.items():
            for comp in compressors:
                if comp.config.target_bus and comp.config.target_bus.id == bus_id:
                    for callback in callbacks:
                        try:
                            callback(comp.config.target_bus, gain)
                        except Exception:
                            pass
                    break

    def analyze_key_levels(self, bus_levels: dict[str, float]) -> None:
        """
        Update key input levels for all compressors.

        Args:
            bus_levels: Dictionary of bus_id -> level_db.
        """
        with self._lock:
            for compressor in self._compressors.values():
                key_bus = compressor.config.key_bus
                if key_bus is not None and key_bus.id in bus_levels:
                    compressor.set_key_level(bus_levels[key_bus.id])

    def get_gain(self, bus: MixBus) -> float:
        """
        Get the current compression gain for a bus.

        Args:
            bus: Bus to check.

        Returns:
            Gain multiplier (1.0 = no compression).
        """
        with self._lock:
            return self._bus_gains.get(bus.id, 1.0)

    def get_gain_db(self, bus: MixBus) -> float:
        """
        Get the current compression in dB.

        Args:
            bus: Bus to check.

        Returns:
            Gain in dB.
        """
        gain = self.get_gain(bus)
        return linear_to_db(gain) if gain > 0 else MIN_VOLUME_DB

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_compression_change(
        self, callback: Callable[[MixBus, float], None]
    ) -> None:
        """
        Register a callback for compression changes.

        Args:
            callback: Function(bus, gain_linear) called on changes.
        """
        with self._lock:
            self._on_compression_change.append(callback)

    def remove_callback(
        self, callback: Callable[[MixBus, float], None]
    ) -> bool:
        """Remove a compression change callback."""
        with self._lock:
            if callback in self._on_compression_change:
                self._on_compression_change.remove(callback)
                return True
        return False

    # =========================================================================
    # State Management
    # =========================================================================

    def reset_all(self) -> None:
        """Reset all compressors."""
        with self._lock:
            for compressor in self._compressors.values():
                compressor.reset()
            self._bus_gains.clear()

    def clear(self) -> None:
        """Remove all compressors."""
        with self._lock:
            self._compressors.clear()
            self._bus_gains.clear()

    def get_state(self) -> dict:
        """Get current state for debugging."""
        with self._lock:
            return {
                "compressors": {
                    id: comp.get_stats()
                    for id, comp in self._compressors.items()
                },
                "bus_gains": {
                    bus_id: linear_to_db(gain)
                    for bus_id, gain in self._bus_gains.items()
                },
            }

    def __repr__(self) -> str:
        with self._lock:
            active = sum(1 for c in self._compressors.values() if c.is_compressing)
            return (
                f"SidechainManager(compressors={len(self._compressors)}, "
                f"active={active})"
            )
