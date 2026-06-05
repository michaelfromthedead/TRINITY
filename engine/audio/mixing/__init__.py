"""
Audio Mixing Subsystem

This module provides comprehensive audio mixing capabilities including:
- Mix Bus hierarchy (master, category, sub, aux buses)
- Bus routing (parent output, aux sends, direct output)
- Mix snapshots with smooth transitions
- Ducking (dialogue, event, focus)
- Sidechain compression
- HDR audio (dynamic range management)

Main Entry Point:
    Use the `Mixer` class for typical usage:

    ```python
    from engine.audio.mixing import Mixer

    mixer = Mixer()
    mixer.initialize()

    # Access buses
    sfx = mixer.get_bus("sfx")
    sfx.volume = 0.8

    # Snapshot transitions
    mixer.transition_to_snapshot("combat", blend_time=0.5)

    # Update every frame
    mixer.update(delta_time)
    ```

Components:
    - Mixer: Main coordinator
    - MixBus: Audio bus with volume, filters, mute/solo
    - BusRouter: Aux sends and direct outputs
    - SnapshotManager: Mix state storage and transitions
    - DuckingManager: Automatic volume reduction
    - SidechainManager: Sidechain compression
    - HDRAudioManager: Dynamic range management
"""

# Configuration and utilities
from .config import (
    # Bus defaults
    MASTER_VOLUME,
    DEFAULT_BUS_VOLUME,
    MIN_VOLUME_DB,
    MAX_VOLUME_DB,
    DEFAULT_PITCH,
    MIN_PITCH,
    MAX_PITCH,
    # Filters
    DEFAULT_LOW_PASS,
    DEFAULT_HIGH_PASS,
    FILTER_Q,
    MIN_FILTER_FREQ,
    MAX_FILTER_FREQ,
    # Ducking
    DIALOGUE_DUCK_AMOUNT_DB,
    EVENT_DUCK_AMOUNT_DB,
    FOCUS_DUCK_AMOUNT_DB,
    DUCK_ATTACK_MS,
    DUCK_RELEASE_MS,
    DUCK_THRESHOLD_DB,
    DUCK_HOLD_MS,
    # Sidechain
    SIDECHAIN_RATIO,
    SIDECHAIN_THRESHOLD_DB,
    SIDECHAIN_ATTACK_MS,
    SIDECHAIN_RELEASE_MS,
    SIDECHAIN_KNEE_DB,
    # Snapshots
    SNAPSHOT_BLEND_TIME,
    MAX_ACTIVE_SNAPSHOTS,
    DEFAULT_SNAPSHOT_PRIORITY,
    InterpolationCurve,
    # HDR
    HDR_WINDOW_DB,
    HDR_ADAPTATION_SPEED,
    HDR_CEILING_DB,
    HDR_FLOOR_DB,
    HDR_PRIORITY_CRITICAL,
    HDR_PRIORITY_HIGH,
    HDR_PRIORITY_NORMAL,
    HDR_PRIORITY_LOW,
    # Categories
    CATEGORY_MASTER,
    CATEGORY_SFX,
    CATEGORY_MUSIC,
    CATEGORY_VO,
    CATEGORY_AMBIENT,
    CATEGORY_UI,
    DEFAULT_CATEGORIES,
    SUBCATEGORIES,
    # Routing
    MAX_AUX_SENDS,
    DEFAULT_SEND_LEVEL,
    # Tick Pipeline
    CATEGORY_TO_BUS,
    MIXER_BUFFER_SIZE,
    MIXER_NUM_CHANNELS,
    LOUDNESS_ANALYSIS_SMOOTHING,
    # Utility functions
    db_to_linear,
    linear_to_db,
    clamp,
    lerp,
    apply_curve,
)

# Mix Bus
from .mix_bus import (
    MixBus,
    BusType,
    BusState,
    BusStatus,
    FilterState as _FilterStateDataclass,
    FilterStateEnum,
    create_default_hierarchy,
)
# Re-export FilterStateEnum as FilterState for test compatibility
FilterState = FilterStateEnum

# Bus Routing
from .bus_routing import (
    BusRouter,
    AuxSend,
    DirectOutput,
    RoutingMode,
)

# Snapshots
from .mix_snapshot import (
    MixSnapshot,
    BusSnapshot,
    SnapshotManager,
    SnapshotState,
    ActiveSnapshot,
)

# Ducking
from .ducking import (
    DuckingManager,
    DuckingInstance,
    DuckConfig,
    DuckEnvelope,
    DuckType,
    DuckState,
)

# Sidechain
from .sidechain import (
    SidechainManager,
    SidechainCompressor,
    SidechainConfig,
    CompressorState,
)

# HDR Audio
from .hdr_audio import (
    HDRAudioManager,
    AudioSource,
    MixWindow,
    HDRPriority,
)

# Main Mixer
from .mixer import (
    Mixer,
    MixerConfig,
)

__all__ = [
    # Main classes
    "Mixer",
    "MixerConfig",
    "MixBus",
    "BusRouter",
    "SnapshotManager",
    "DuckingManager",
    "SidechainManager",
    "HDRAudioManager",
    # Bus types and states
    "BusType",
    "BusState",
    "BusStatus",
    "FilterState",
    "RoutingMode",
    # Routing
    "AuxSend",
    "DirectOutput",
    # Snapshots
    "MixSnapshot",
    "BusSnapshot",
    "SnapshotState",
    "ActiveSnapshot",
    # Ducking
    "DuckingInstance",
    "DuckConfig",
    "DuckEnvelope",
    "DuckType",
    "DuckState",
    # Sidechain
    "SidechainCompressor",
    "SidechainConfig",
    "CompressorState",
    # HDR
    "AudioSource",
    "MixWindow",
    "HDRPriority",
    # Config/Constants
    "InterpolationCurve",
    "CATEGORY_MASTER",
    "CATEGORY_SFX",
    "CATEGORY_MUSIC",
    "CATEGORY_VO",
    "CATEGORY_AMBIENT",
    "CATEGORY_UI",
    "DEFAULT_CATEGORIES",
    "MIN_VOLUME_DB",
    "MAX_VOLUME_DB",
    # Utilities
    "db_to_linear",
    "linear_to_db",
    "clamp",
    "lerp",
    "apply_curve",
    "create_default_hierarchy",
    # Tick pipeline
    "CATEGORY_TO_BUS",
    "MIXER_BUFFER_SIZE",
    "MIXER_NUM_CHANNELS",
    "LOUDNESS_ANALYSIS_SMOOTHING",
]
