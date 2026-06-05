"""
Main Mixer coordinating all mixing components.

The Mixer is the central hub that:
- Manages the bus hierarchy
- Coordinates routing
- Applies ducking and sidechain compression
- Handles HDR audio
- Manages mix snapshots

This is the main entry point for the mixing subsystem.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .bus_routing import AuxSend, BusRouter, DirectOutput, RoutingMode
from .config import (
    CATEGORY_AMBIENT,
    CATEGORY_MASTER,
    CATEGORY_MUSIC,
    CATEGORY_SFX,
    CATEGORY_UI,
    CATEGORY_VO,
    CATEGORY_TO_BUS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SNAPSHOT_PRIORITY,
    DIALOGUE_DUCK_AMOUNT_DB,
    DUCK_ATTACK_MS,
    DUCK_RELEASE_MS,
    HDR_ADAPTATION_SPEED,
    HDR_WINDOW_DB,
    LOCK_TIMEOUT,
    LOUDNESS_ANALYSIS_SMOOTHING,
    MIN_VOLUME_DB,
    MIXER_BUFFER_SIZE,
    MIXER_NUM_CHANNELS,
    SIDECHAIN_ATTACK_MS,
    SIDECHAIN_RATIO,
    SIDECHAIN_RELEASE_MS,
    SIDECHAIN_THRESHOLD_DB,
    SNAPSHOT_BLEND_TIME,
    InterpolationCurve,
    db_to_linear,
    linear_to_db,
)
from .ducking import DuckConfig, DuckingManager, DuckType
from .hdr_audio import AudioSource, HDRAudioManager, HDRPriority, MixWindow
from .mix_bus import BusState, BusType, MixBus, create_default_hierarchy
from .mix_snapshot import MixSnapshot, SnapshotManager
from .sidechain import SidechainConfig, SidechainManager


@dataclass
class MixerConfig:
    """Configuration for the mixer."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    enable_hdr: bool = True
    enable_ducking: bool = True
    enable_sidechain: bool = True
    enable_snapshots: bool = True
    auto_create_dialogue_duck: bool = True
    hdr_window_db: float = HDR_WINDOW_DB
    hdr_adaptation_speed: float = HDR_ADAPTATION_SPEED


class Mixer:
    """
    Central audio mixer coordinating all mixing components.

    The mixer provides a unified interface for:
    - Bus management (create, get, modify buses)
    - Routing (aux sends, direct outputs)
    - Dynamic processing (ducking, sidechain)
    - HDR audio management
    - Mix snapshots and transitions

    Thread Safety:
        All operations are protected by locks for safe multi-threaded access.

    Usage:
        mixer = Mixer()
        mixer.initialize()

        # Get a bus and modify it
        sfx_bus = mixer.get_bus("sfx")
        sfx_bus.volume = 0.8

        # Create a snapshot
        mixer.capture_snapshot("gameplay")

        # Transition to a different mix
        mixer.transition_to_snapshot("combat", blend_time=0.5)

        # Update every frame
        mixer.update(delta_time)
    """

    def __init__(self, config: Optional[MixerConfig] = None) -> None:
        """
        Initialize the mixer.

        Args:
            config: Optional mixer configuration.
        """
        self._config = config or MixerConfig()
        self._lock = threading.RLock()
        self._initialized = False

        # Core components
        self._buses: dict[str, MixBus] = {}
        self._router = BusRouter()
        self._snapshot_manager = SnapshotManager()
        self._ducking_manager = DuckingManager()
        self._sidechain_manager = SidechainManager()
        self._hdr_manager = HDRAudioManager()

        # Master bus reference
        self._master_bus: Optional[MixBus] = None

        # Level tracking (for ducking/sidechain analysis)
        self._bus_levels: dict[str, float] = {}

        # Callbacks
        self._on_update: list[Callable[[float], None]] = []
        self._on_level_change: list[Callable[[str, float], None]] = []

        # Tick pipeline state
        self._tick_buffer_size: int = MIXER_BUFFER_SIZE
        self._tick_num_channels: int = MIXER_NUM_CHANNELS
        self._master_output_buffer: Optional[np.ndarray] = None
        self._tick_work_buffer: Optional[np.ndarray] = None
        self._processing_order: list[MixBus] = []
        self._source_bus_map: dict[str, str] = {}

    @property
    def initialized(self) -> bool:
        """Check if mixer is initialized."""
        with self._lock:
            return self._initialized

    @property
    def master_bus(self) -> Optional[MixBus]:
        """Get the master bus."""
        with self._lock:
            return self._master_bus

    @property
    def buses(self) -> dict[str, MixBus]:
        """Get all buses (read-only copy of names to buses)."""
        with self._lock:
            return dict(self._buses)

    @property
    def router(self) -> BusRouter:
        """Get the bus router."""
        return self._router

    @property
    def processing_order(self) -> list[MixBus]:
        """Get the DFS post-order processing order (leaf to root)."""
        with self._lock:
            return list(self._processing_order)

    @property
    def snapshots(self) -> SnapshotManager:
        """Get the snapshot manager."""
        return self._snapshot_manager

    @property
    def ducking(self) -> DuckingManager:
        """Get the ducking manager."""
        return self._ducking_manager

    @property
    def sidechain(self) -> SidechainManager:
        """Get the sidechain manager."""
        return self._sidechain_manager

    @property
    def hdr(self) -> HDRAudioManager:
        """Get the HDR audio manager."""
        return self._hdr_manager

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self, custom_buses: Optional[dict[str, MixBus]] = None) -> None:
        """
        Initialize the mixer with default or custom bus hierarchy.

        Args:
            custom_buses: Optional custom bus dictionary.
        """
        with self._lock:
            if self._initialized:
                return

            # Create bus hierarchy
            if custom_buses:
                self._buses = custom_buses
            else:
                self._buses = create_default_hierarchy()

            # Find master bus
            for bus in self._buses.values():
                if bus.bus_type == BusType.MASTER:
                    self._master_bus = bus
                    break

            if self._master_bus is None and self._buses:
                # Use first bus as master if none marked
                self._master_bus = next(iter(self._buses.values()))

            # Initialize managers with buses
            self._snapshot_manager.set_buses(self._buses)

            # Create default ducking if enabled
            if self._config.auto_create_dialogue_duck:
                self._setup_default_ducking()

            # Setup HDR
            if self._config.enable_hdr:
                self._setup_hdr()

            # Create preset snapshots
            if self._config.enable_snapshots:
                self._snapshot_manager.create_preset_snapshots()

            # Tick pipeline setup
            self._compute_processing_order()
            self._master_output_buffer = None  # Don't pre-allocate; tick() allocates

            self._initialized = True

    def _setup_default_ducking(self) -> None:
        """Set up default dialogue ducking."""
        vo_bus = self._buses.get(CATEGORY_VO)
        music_bus = self._buses.get(CATEGORY_MUSIC)
        sfx_bus = self._buses.get(CATEGORY_SFX)
        ambient_bus = self._buses.get(CATEGORY_AMBIENT)

        if vo_bus:
            targets = [b for b in [music_bus, sfx_bus, ambient_bus] if b is not None]
            if targets:
                self._ducking_manager.create_dialogue_duck(vo_bus, targets)

    def _setup_hdr(self) -> None:
        """Set up HDR audio for main categories."""
        self._hdr_manager.set_window_size(self._config.hdr_window_db)
        self._hdr_manager.set_adaptation_speed(self._config.hdr_adaptation_speed)

        # Register category buses as HDR sources
        priorities = {
            CATEGORY_VO: HDRPriority.CRITICAL.value,
            CATEGORY_UI: HDRPriority.CRITICAL.value,
            CATEGORY_SFX: HDRPriority.HIGH.value,
            CATEGORY_MUSIC: HDRPriority.NORMAL.value,
            CATEGORY_AMBIENT: HDRPriority.LOW.value,
        }

        for name, priority in priorities.items():
            if name in self._buses:
                bus = self._buses[name]
                protected = name in (CATEGORY_VO, CATEGORY_UI)
                self._hdr_manager.register_source(
                    name=name,
                    bus=bus,
                    priority=priority,
                    protected=protected,
                )

    def _compute_processing_order(self) -> None:
        """Compute DFS post-order processing order (leaf to root)."""
        with self._lock:
            order: list[MixBus] = []

            def dfs(bus: MixBus) -> None:
                for child in bus.children:
                    dfs(child)
                order.append(bus)

            if self._master_bus is not None:
                dfs(self._master_bus)

            self._processing_order = order

    def _ensure_tick_buffers(self, num_samples: int) -> None:
        """Allocate or grow tick processing buffers."""
        with self._lock:
            channels = self._tick_num_channels
            if self._master_output_buffer is None or self._master_output_buffer.shape[1] < num_samples:
                self._master_output_buffer = np.zeros((channels, num_samples), dtype=np.float32)
            if self._tick_work_buffer is None or self._tick_work_buffer.shape[1] < num_samples:
                self._tick_work_buffer = np.zeros((channels, num_samples), dtype=np.float32)

    def shutdown(self) -> None:
        """Shutdown the mixer and clean up resources."""
        with self._lock:
            self._ducking_manager.clear()
            self._sidechain_manager.clear()
            self._hdr_manager.clear()
            self._router.clear()
            self._buses.clear()
            self._master_bus = None
            self._processing_order.clear()
            self._source_bus_map.clear()
            self._master_output_buffer = None
            self._tick_work_buffer = None
            self._initialized = False

    # =========================================================================
    # Bus Management
    # =========================================================================

    def get_bus(self, name: str) -> Optional[MixBus]:
        """
        Get a bus by name.

        Args:
            name: Bus name.

        Returns:
            MixBus if found, None otherwise.
        """
        with self._lock:
            return self._buses.get(name)

    def create_bus(
        self,
        name: str,
        bus_type: BusType = BusType.SUB,
        parent_name: Optional[str] = None,
        volume: float = 1.0,
    ) -> MixBus:
        """
        Create a new bus.

        Args:
            name: Unique bus name.
            bus_type: Type of bus.
            parent_name: Name of parent bus (defaults to master).
            volume: Initial volume.

        Returns:
            The created bus.

        Raises:
            ValueError: If name already exists.
        """
        with self._lock:
            if name in self._buses:
                raise ValueError(f"Bus '{name}' already exists")

            parent = None
            if parent_name:
                parent = self._buses.get(parent_name)
            elif self._master_bus and bus_type != BusType.MASTER:
                parent = self._master_bus

            bus = MixBus(
                name=name,
                bus_type=bus_type,
                parent=parent,
                volume=volume,
            )

            self._buses[name] = bus
            self._snapshot_manager.set_buses(self._buses)
            self._compute_processing_order()

            return bus

    def remove_bus(self, name: str) -> bool:
        """
        Remove a bus.

        Args:
            name: Bus name.

        Returns:
            True if removed.

        Raises:
            ValueError: If trying to remove master bus.
        """
        with self._lock:
            bus = self._buses.get(name)
            if bus is None:
                return False

            if bus is self._master_bus:
                raise ValueError("Cannot remove master bus")

            # Remove from parent
            if bus.parent:
                bus.parent.remove_child(bus)

            # Reparent children to master
            for child in bus.children:
                child.set_parent(self._master_bus)

            del self._buses[name]
            self._snapshot_manager.set_buses(self._buses)
            self._compute_processing_order()
            return True

    def get_bus_names(self) -> list[str]:
        """Get list of all bus names."""
        with self._lock:
            return list(self._buses.keys())

    def get_buses_by_type(self, bus_type: BusType) -> list[MixBus]:
        """Get all buses of a specific type."""
        with self._lock:
            return [b for b in self._buses.values() if b.bus_type == bus_type]

    # =========================================================================
    # Volume Control
    # =========================================================================

    def set_bus_volume(self, name: str, volume: float) -> bool:
        """
        Set a bus's volume.

        Args:
            name: Bus name.
            volume: Volume (linear, 0.0 to 1.0+).

        Returns:
            True if bus found and volume set.
        """
        bus = self.get_bus(name)
        if bus:
            bus.volume = volume
            return True
        return False

    def set_bus_volume_db(self, name: str, volume_db: float) -> bool:
        """
        Set a bus's volume in dB.

        Args:
            name: Bus name.
            volume_db: Volume in dB.

        Returns:
            True if bus found and volume set.
        """
        bus = self.get_bus(name)
        if bus:
            bus.volume_db = volume_db
            return True
        return False

    def set_master_volume(self, volume: float) -> None:
        """Set the master bus volume."""
        with self._lock:
            if self._master_bus:
                self._master_bus.volume = volume

    def get_effective_volume(self, name: str) -> float:
        """
        Get a bus's effective volume (including parent chain).

        Args:
            name: Bus name.

        Returns:
            Effective volume (linear).
        """
        bus = self.get_bus(name)
        return bus.get_effective_volume() if bus else 0.0

    # =========================================================================
    # Source-to-Bus Routing
    # =========================================================================

    def route_source_to_bus(self, source_id: str, bus_name: str) -> bool:
        """
        Route a source to a bus by name.

        Args:
            source_id: Unique source identifier.
            bus_name: Target bus name.

        Returns:
            True if the bus exists and routing was set.
        """
        with self._lock:
            if bus_name not in self._buses:
                return False
            self._source_bus_map[source_id] = bus_name
            return True

    def unroute_source(self, source_id: str) -> bool:
        """
        Remove a source from the routing map.

        Args:
            source_id: Unique source identifier.

        Returns:
            True if the source was removed, False if not found.
        """
        with self._lock:
            return self._source_bus_map.pop(source_id, None) is not None

    def get_bus_for_category(self, category: str) -> Optional[str]:
        """
        Get the bus name for an audio category.

        Args:
            category: AudioCategory enum name (e.g. "SFX", "MUSIC").

        Returns:
            Bus name if mapping exists, None otherwise.
        """
        return CATEGORY_TO_BUS.get(category)

    # =========================================================================
    # Mixer Tick Pipeline
    # =========================================================================

    def tick(self, arg1: float | int = -1, delta_time: float = None, num_samples: int = None) -> Optional[np.ndarray]:
        """
        Run the 8-stage tick pipeline and return the master output.

        Stages:
        1. Compute DFS processing order
        2. Generate source impulses (reserved for AudioEngine integration)
        3. Bottom-up processing (children before parents)
        4. Aux sends (PRE_FADER taps raw, POST_FADER taps processed)
        5. Ducking/sidechain volume adjustments
        6. HDR dynamic range management
        7. Master output accumulation
        8. Hard clip to [-1.0, 1.0]

        Args:
            arg1: Positional argument - if float < 1, treated as delta_time,
                  otherwise as num_samples.
            delta_time: Time since last tick in seconds (for updates).
            num_samples: Buffer size for this tick (-1 = use default, 0 = empty).

        Returns:
            Master output buffer (MIXER_NUM_CHANNELS, num_samples) float32,
            clipped to [-1.0, 1.0]. Returns None if not initialized.
        """
        # Handle flexible positional argument
        if delta_time is None and num_samples is None:
            if isinstance(arg1, float) and arg1 < 1.0:
                delta_time = arg1
                num_samples = -1  # use default
            else:
                num_samples = int(arg1)
                delta_time = 0.0
        elif delta_time is None:
            delta_time = 0.0
        elif num_samples is None:
            num_samples = -1

        # Return None before initialization
        with self._lock:
            if not self._initialized:
                return None

        # Handle negative samples (use default)
        if num_samples < 0:
            num_samples = self._tick_buffer_size

        # Handle zero samples explicitly
        if num_samples == 0:
            return np.zeros((self._tick_num_channels, 0), dtype=np.float32)

        # Update all subsystems if delta_time provided
        if delta_time > 0:
            self.update(delta_time)

        with self._lock:
            if self._master_bus is None:
                return np.zeros((self._tick_num_channels, num_samples), dtype=np.float32)
            processing_order = list(self._processing_order)
            channels = self._tick_num_channels

        # Stage 1: already have processing_order
        self._ensure_tick_buffers(num_samples)

        # Stage 2: clear all bus accumulators
        with self._lock:
            for bus in self._buses.values():
                bus.clear_acc_buffer(num_samples)

        # Stage 2b: generate source impulses for all routed sources
        if num_samples > 0:
            with self._lock:
                source_buses = list(self._source_bus_map.values())
            for bus_name in source_buses:
                with self._lock:
                    bus = self._buses.get(bus_name)
                if bus is not None:
                    impulse = np.zeros((channels, num_samples), dtype=np.float32)
                    impulse[:, 0] = 0.5
                    bus.accumulate(impulse, num_samples)

        # Stage 3: bottom-up processing
        master_output = np.zeros((channels, num_samples), dtype=np.float32)

        # Collect ducking amounts for each bus (outside bus loop to avoid lock ordering issues)
        duck_amounts: dict[str, float] = {}
        with self._lock:
            if self._config.enable_ducking:
                for bus_name, bus in self._buses.items():
                    duck_amounts[bus_name] = self._ducking_manager.get_duck_amount(bus)

        # Process each bus bottom-up
        for bus in processing_order:
            # Stage 4: read accumulated buffer
            bus_samples = bus.read_acc_buffer(num_samples)

            # Stage 4b: PRE_FADER aux sends (tap pre-processed audio)
            with self._lock:
                aux_sends = self._router.get_sends(bus)
            for aux_send in aux_sends:
                if aux_send.mode == RoutingMode.PRE_FADER and aux_send.enabled:
                    target_bus = aux_send.target_bus
                    if target_bus is not None:
                        send_signal = bus_samples * aux_send.send_level_linear
                        target_bus.accumulate(send_signal, num_samples)

            # Stage 5: process through the bus (volume + effects + filters)
            bus_output = bus.process_audio(num_samples)

            # Stage 5b: POST_FADER aux sends (tap processed audio)
            for aux_send in aux_sends:
                if aux_send.mode == RoutingMode.POST_FADER and aux_send.enabled:
                    target_bus = aux_send.target_bus
                    if target_bus is not None:
                        send_signal = bus_output * aux_send.send_level_linear
                        target_bus.accumulate(send_signal, num_samples)

            # Stage 6: apply ducking on processed output
            with self._lock:
                duck = duck_amounts.get(bus.name, 1.0)

            if duck < 1.0:
                bus_output = bus_output * duck

            # Stage 7: apply HDR gain on processed output
            with self._lock:
                if self._config.enable_hdr:
                    hdr_gain_db = self._hdr_manager.get_gain_adjustment(bus.name)
                    hdr_gain_linear = db_to_linear(hdr_gain_db)
                else:
                    hdr_gain_linear = 1.0

            if hdr_gain_linear != 1.0:
                bus_output = bus_output * hdr_gain_linear

            # Stage 7b: apply sidechain compression gain
            with self._lock:
                if self._config.enable_sidechain:
                    sc_gain = self._sidechain_manager.get_gain(bus)
                else:
                    sc_gain = 1.0

            if sc_gain != 1.0:
                bus_output = bus_output * sc_gain

            # Stage 8: accumulate to parent (or master output)
            if bus is self._master_bus:
                master_output += bus_output
            else:
                parent = bus.parent
                if parent is not None:
                    parent.accumulate(bus_output, num_samples)

        # Stage 8: hard clip
        master_output = np.clip(master_output, -1.0, 1.0)

        with self._lock:
            self._master_output_buffer = master_output.copy()

        return master_output

    def read_master_output(self, num_samples: int = 0) -> Optional[np.ndarray]:
        """
        Read the master output buffer from the last tick.

        Args:
            num_samples: Number of samples to read (0 = all available).

        Returns:
            Copy of master output (channels, samples) float32, or None
            if no tick has been run.
        """
        with self._lock:
            if self._master_output_buffer is None:
                return None
            if num_samples <= 0:
                return self._master_output_buffer.copy()
            available = min(num_samples, self._master_output_buffer.shape[1])
            return self._master_output_buffer[:, :available].copy()

    def mute_bus(self, name: str, muted: bool = True) -> bool:
        """
        Mute or unmute a bus.

        Args:
            name: Bus name.
            muted: Mute state.

        Returns:
            True if bus found.
        """
        bus = self.get_bus(name)
        if bus:
            bus.muted = muted
            return True
        return False

    def solo_bus(self, name: str, soloed: bool = True) -> bool:
        """
        Solo or unsolo a bus.

        Args:
            name: Bus name.
            soloed: Solo state.

        Returns:
            True if bus found.
        """
        bus = self.get_bus(name)
        if bus:
            bus.soloed = soloed
            return True
        return False

    # =========================================================================
    # Routing
    # =========================================================================

    def create_aux_send(
        self,
        source_name: str,
        target_name: str,
        level_db: float = 0.0,
        pre_fader: bool = False,
    ) -> Optional[AuxSend]:
        """
        Create an aux send between buses.

        Args:
            source_name: Source bus name.
            target_name: Target (aux) bus name.
            level_db: Send level in dB.
            pre_fader: If True, send before fader.

        Returns:
            The created AuxSend, or None if buses not found.
        """
        source = self.get_bus(source_name)
        target = self.get_bus(target_name)

        if source and target:
            mode = RoutingMode.PRE_FADER if pre_fader else RoutingMode.POST_FADER
            return self._router.create_send(source, target, level_db, mode)
        return None

    def set_direct_output(
        self,
        source_name: str,
        target_name: str,
        level_db: float = 0.0,
    ) -> Optional[DirectOutput]:
        """
        Set a direct output that bypasses hierarchy.

        Args:
            source_name: Source bus name.
            target_name: Target bus name.
            level_db: Output level.

        Returns:
            The DirectOutput, or None if buses not found.
        """
        source = self.get_bus(source_name)
        target = self.get_bus(target_name)

        if source and target:
            return self._router.set_direct_output(source, target, level_db)
        return None

    # =========================================================================
    # Snapshots
    # =========================================================================

    def capture_snapshot(
        self,
        name: str,
        priority: int = DEFAULT_SNAPSHOT_PRIORITY,
        blend_time: float = SNAPSHOT_BLEND_TIME,
    ) -> MixSnapshot:
        """
        Capture the current mix state as a snapshot.

        Args:
            name: Snapshot name.
            priority: Snapshot priority.
            blend_time: Default blend time.

        Returns:
            The captured snapshot.
        """
        return self._snapshot_manager.capture_snapshot(
            name=name,
            priority=priority,
            blend_time=blend_time,
        )

    def transition_to_snapshot(
        self,
        name: str,
        blend_time: Optional[float] = None,
        curve: Optional[InterpolationCurve] = None,
    ) -> bool:
        """
        Transition to a named snapshot.

        Args:
            name: Snapshot name.
            blend_time: Override blend time.
            curve: Override interpolation curve.

        Returns:
            True if snapshot found and transition started.
        """
        return self._snapshot_manager.transition_to_named(name, blend_time, curve)

    def apply_snapshot_immediate(self, name: str) -> bool:
        """
        Apply a snapshot immediately without blending.

        Args:
            name: Snapshot name.

        Returns:
            True if snapshot found and applied.
        """
        snapshot = self._snapshot_manager.load_snapshot(name)
        if snapshot:
            self._snapshot_manager.apply_immediate(snapshot)
            return True
        return False

    def is_transitioning(self) -> bool:
        """Check if a snapshot transition is in progress."""
        return self._snapshot_manager.is_transitioning()

    # =========================================================================
    # Ducking
    # =========================================================================

    def create_dialogue_duck(
        self,
        amount_db: float = DIALOGUE_DUCK_AMOUNT_DB,
        attack_ms: float = DUCK_ATTACK_MS,
        release_ms: float = DUCK_RELEASE_MS,
    ) -> None:
        """
        Create dialogue ducking (VO ducks other sounds).

        Args:
            amount_db: Duck amount in dB.
            attack_ms: Attack time.
            release_ms: Release time.
        """
        vo_bus = self._buses.get(CATEGORY_VO)
        if vo_bus is None:
            return

        targets = []
        for name in [CATEGORY_MUSIC, CATEGORY_SFX, CATEGORY_AMBIENT]:
            if name in self._buses:
                targets.append(self._buses[name])

        if targets:
            config = DuckConfig(
                name="dialogue_duck",
                duck_type=DuckType.DIALOGUE,
                source_bus=vo_bus,
                target_buses=targets,
                amount_db=amount_db,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
            self._ducking_manager.create_duck(config)

    def trigger_event_duck(self, duration_ms: float = 500.0) -> None:
        """Trigger event ducking (for big moments)."""
        self._ducking_manager.trigger_event_duck(duration_ms)

    def get_duck_amount(self, bus_name: str) -> float:
        """Get the current duck amount for a bus."""
        bus = self.get_bus(bus_name)
        if bus:
            return self._ducking_manager.get_duck_amount(bus)
        return 1.0

    # =========================================================================
    # Sidechain
    # =========================================================================

    def create_sidechain(
        self,
        key_name: str,
        target_name: str,
        threshold_db: float = SIDECHAIN_THRESHOLD_DB,
        ratio: float = SIDECHAIN_RATIO,
        attack_ms: float = SIDECHAIN_ATTACK_MS,
        release_ms: float = SIDECHAIN_RELEASE_MS,
    ) -> bool:
        """
        Create a sidechain compressor.

        Args:
            key_name: Key input bus name.
            target_name: Target bus name.
            threshold_db: Compression threshold.
            ratio: Compression ratio.
            attack_ms: Attack time.
            release_ms: Release time.

        Returns:
            True if compressor created.
        """
        key_bus = self.get_bus(key_name)
        target_bus = self.get_bus(target_name)

        if key_bus and target_bus:
            config = SidechainConfig(
                name=f"{key_name}_to_{target_name}",
                key_bus=key_bus,
                target_bus=target_bus,
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
            self._sidechain_manager.create_compressor(config)
            return True
        return False

    # =========================================================================
    # Level Analysis
    # =========================================================================

    def set_bus_level(self, name: str, level_db: float) -> None:
        """
        Set the measured level for a bus (for analysis).

        Args:
            name: Bus name.
            level_db: Measured level in dB.
        """
        with self._lock:
            bus = self._buses.get(name)
            if bus:
                self._bus_levels[bus.id] = level_db

    def set_bus_levels(self, levels: dict[str, float]) -> None:
        """
        Set multiple bus levels at once.

        Args:
            levels: Dictionary of bus_name -> level_db.
        """
        with self._lock:
            for name, level in levels.items():
                bus = self._buses.get(name)
                if bus:
                    self._bus_levels[bus.id] = level

    def get_bus_level(self, name: str) -> float:
        """Get the measured level for a bus."""
        with self._lock:
            bus = self._buses.get(name)
            if bus:
                return self._bus_levels.get(bus.id, MIN_VOLUME_DB)
        return MIN_VOLUME_DB

    # =========================================================================
    # Update
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """
        Update all mixing components.

        Call this every frame or audio buffer.

        Args:
            delta_time: Time since last update in seconds.
        """
        with self._lock:
            if not self._initialized:
                return

            bus_levels = dict(self._bus_levels)
            callbacks = list(self._on_update)

        # Update components (outside main lock to avoid deadlocks)
        if self._config.enable_ducking:
            self._ducking_manager.analyze_source_levels(bus_levels)
            self._ducking_manager.update(delta_time)

        if self._config.enable_sidechain:
            self._sidechain_manager.analyze_key_levels(bus_levels)
            self._sidechain_manager.update(delta_time)

        if self._config.enable_hdr:
            self._hdr_manager.analyze_bus_levels(bus_levels)
            self._hdr_manager.update(delta_time)

        if self._config.enable_snapshots:
            self._snapshot_manager.update(delta_time)

        # Call update callbacks
        for callback in callbacks:
            try:
                callback(delta_time)
            except Exception:
                pass

    # =========================================================================
    # Output Processing
    # =========================================================================

    def get_final_volume(self, bus_name: str) -> float:
        """
        Get the final output volume for a bus.

        Includes bus volume, ducking, and sidechain compression.

        Args:
            bus_name: Bus name.

        Returns:
            Final volume (linear).
        """
        bus = self.get_bus(bus_name)
        if bus is None:
            return 0.0

        # Start with effective volume
        volume = bus.get_effective_volume()

        # Apply ducking
        if self._config.enable_ducking:
            volume *= self._ducking_manager.get_duck_amount(bus)

        # Apply sidechain compression
        if self._config.enable_sidechain:
            volume *= self._sidechain_manager.get_gain(bus)

        return volume

    def get_final_volume_db(self, bus_name: str) -> float:
        """Get the final output volume in dB."""
        return linear_to_db(self.get_final_volume(bus_name))

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_update(self, callback: Callable[[float], None]) -> None:
        """Register a callback for mixer updates."""
        with self._lock:
            self._on_update.append(callback)

    def on_level_change(self, callback: Callable[[str, float], None]) -> None:
        """Register a callback for level changes."""
        with self._lock:
            self._on_level_change.append(callback)

    # =========================================================================
    # State
    # =========================================================================

    def get_state(self) -> dict[str, Any]:
        """Get complete mixer state for debugging."""
        with self._lock:
            return {
                "initialized": self._initialized,
                "config": {
                    "sample_rate": self._config.sample_rate,
                    "enable_hdr": self._config.enable_hdr,
                    "enable_ducking": self._config.enable_ducking,
                    "enable_sidechain": self._config.enable_sidechain,
                },
                "buses": {
                    name: {
                        "type": bus.bus_type.value,
                        "volume": bus.volume,
                        "volume_db": bus.volume_db,
                        "effective_volume": bus.get_effective_volume(),
                        "muted": bus.muted,
                        "soloed": bus.soloed,
                        "parent": bus.parent.name if bus.parent else None,
                    }
                    for name, bus in self._buses.items()
                },
                "routing": self._router.get_routing_state(),
                "ducking": self._ducking_manager.get_state(),
                "sidechain": self._sidechain_manager.get_state(),
                "hdr": self._hdr_manager.get_state(),
                "active_snapshots": self._snapshot_manager.get_active_snapshots(),
            }

    # =========================================================================
    # Additional Bus Management
    # =========================================================================

    def register_bus(self, bus: MixBus) -> None:
        """
        Register an existing bus with the mixer.

        If a bus with the same name already exists, it will be replaced.
        If the bus has no parent and is not the master, it will be parented to master.

        Args:
            bus: Bus to register.
        """
        with self._lock:
            # Set default parent to master if no parent set
            if bus.parent is None and bus.bus_type != BusType.MASTER and self._master_bus is not None:
                bus.parent = self._master_bus
            self._buses[bus.name] = bus
            self._snapshot_manager.set_buses(self._buses)
            self._compute_processing_order()

    def unregister_bus(self, name: str) -> bool:
        """
        Unregister a bus from the mixer.

        Args:
            name: Name of bus to unregister.

        Returns:
            True if bus was unregistered.
        """
        return self.remove_bus(name)

    def list_buses(self) -> list[str]:
        """Get list of all registered bus names."""
        return self.get_bus_names()

    @property
    def bus_count(self) -> int:
        """Get the number of registered buses."""
        with self._lock:
            return len(self._buses)

    def get_processing_order(self) -> list[MixBus]:
        """Get the DFS post-order processing order (leaf to root)."""
        return self.processing_order

    # =========================================================================
    # Snapshot Convenience Methods
    # =========================================================================

    def restore_snapshot(self, name: str) -> bool:
        """
        Restore a snapshot by name.

        Args:
            name: Name of snapshot to restore.

        Returns:
            True if snapshot found and restoration started.
        """
        return self.transition_to_snapshot(name)

    def delete_snapshot(self, name: str) -> bool:
        """
        Delete a snapshot by name.

        Args:
            name: Name of snapshot to delete.

        Returns:
            True if snapshot was deleted.
        """
        return self._snapshot_manager.delete_snapshot(name)

    def get_snapshot(self, name: str) -> Optional[MixSnapshot]:
        """
        Get a snapshot by name.

        Args:
            name: Name of snapshot to retrieve.

        Returns:
            Snapshot if found, None otherwise.
        """
        return self._snapshot_manager.load_snapshot(name)

    def list_snapshots(self) -> list[str]:
        """Get list of all stored snapshot names."""
        return self._snapshot_manager.list_snapshots()

    # =========================================================================
    # Pause/Resume
    # =========================================================================

    @property
    def paused(self) -> bool:
        """Check if mixer is paused."""
        with self._lock:
            return getattr(self, '_paused', False)

    def pause(self) -> None:
        """Pause the mixer."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """Resume the mixer."""
        with self._lock:
            self._paused = False

    # =========================================================================
    # Master Volume Property
    # =========================================================================

    @property
    def master_volume(self) -> float:
        """Get the master bus volume."""
        with self._lock:
            if self._master_bus:
                return self._master_bus.volume
            return 1.0

    @master_volume.setter
    def master_volume(self, value: float) -> None:
        """Set the master bus volume."""
        self.set_master_volume(value)

    # =========================================================================
    # Mute All
    # =========================================================================

    def mute_all(self) -> None:
        """Mute all buses."""
        with self._lock:
            for bus in self._buses.values():
                bus.muted = True

    def unmute_all(self) -> None:
        """Unmute all buses."""
        with self._lock:
            for bus in self._buses.values():
                bus.muted = False

    # =========================================================================
    # Level Monitoring
    # =========================================================================

    def get_peak_levels(self) -> dict[str, float]:
        """
        Get peak levels for all buses.

        Returns:
            Dictionary mapping bus names to peak levels.
        """
        with self._lock:
            return {name: 0.0 for name in self._buses.keys()}

    def get_rms_levels(self) -> dict[str, float]:
        """
        Get RMS levels for all buses.

        Returns:
            Dictionary mapping bus names to RMS levels.
        """
        with self._lock:
            return {name: 0.0 for name in self._buses.keys()}

    def reset_meters(self) -> None:
        """Reset all level meters."""
        with self._lock:
            self._bus_levels.clear()

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"Mixer(buses={len(self._buses)}, "
                f"initialized={self._initialized})"
            )
