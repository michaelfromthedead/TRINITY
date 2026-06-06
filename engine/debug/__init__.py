"""
Debug and diagnostics layer for the game engine.

Provides comprehensive debugging, profiling, and testing infrastructure:
- Logging: Structured, categorized logging with multiple output targets
- Console: In-game console with CVars, commands, and scripting
- Visual: Debug drawing, overlays, and render views
- Profiling: CPU, GPU, memory, and network profiling
- Crash: Assertions, crash handling, and reporting
- Replay: Recording and playback for debugging
- Testing: Unit, integration, and automation testing framework
- TimeTravel: Snapshot-based time-travel debugging (T-CC-4.1)
- Breakpoints: Conditional breakpoints and value watches (T-CC-4.2)
- TimeTravelUI: Time-travel debugging UI components (T-CC-4.3)
"""

from . import testing
from .time_travel import (
    TickSnapshot,
    SnapshotMetadata,
    SnapshotState,
    SnapshotRingBuffer,
    RingBufferConfig,
    TimeTravel,
    TimeTravelConfig,
    TimeTravelState,
    TimeTravelEvent,
    ReplayController,
    ReplayConfig,
    ReplayState,
    ReplayResult,
    StateProvider,
    WorldStateProvider,
    SnapshotComparison,
    TickRange,
)
from .breakpoints import (
    ConditionalBreakpoint,
    BreakpointState,
    BreakpointHit,
    BreakpointConfig,
    ValueWatch,
    WatchState,
    WatchRecord,
    ValueChange,
    ExpressionEvaluator,
    ExpressionContext,
    EvaluationResult,
    ExpressionError,
    BinarySearchResult,
    ChangeSearcher,
    BreakpointManager,
    ManagerConfig,
    ManagerEvent,
    BreakpointSerializer,
)
from .time_travel_ui import (
    # Configuration
    TimeTravelUIConfig,
    # Main UI components
    TimeTravelUI,
    TimelinePanel,
    ScrubBar,
    StepControls,
    StateDiffView,
    # Timeline widgets
    SnapshotMarkerWidget,
    TickMarkerWidget,
    PlayheadWidget,
    # Diff widgets
    DiffEntry,
    DiffEntryWidget,
    DiffTreeNode,
    DiffType,
    # Events
    UIActionType,
    UIAction,
    # Playback
    PlaybackState,
    PlaybackController,
)

__all__ = [
    "testing",
    # time_travel.py (T-CC-4.1)
    "TickSnapshot",
    "SnapshotMetadata",
    "SnapshotState",
    "SnapshotRingBuffer",
    "RingBufferConfig",
    "TimeTravel",
    "TimeTravelConfig",
    "TimeTravelState",
    "TimeTravelEvent",
    "ReplayController",
    "ReplayConfig",
    "ReplayState",
    "ReplayResult",
    "StateProvider",
    "WorldStateProvider",
    "SnapshotComparison",
    "TickRange",
    # breakpoints.py (T-CC-4.2)
    "ConditionalBreakpoint",
    "BreakpointState",
    "BreakpointHit",
    "BreakpointConfig",
    "ValueWatch",
    "WatchState",
    "WatchRecord",
    "ValueChange",
    "ExpressionEvaluator",
    "ExpressionContext",
    "EvaluationResult",
    "ExpressionError",
    "BinarySearchResult",
    "ChangeSearcher",
    "BreakpointManager",
    "ManagerConfig",
    "ManagerEvent",
    "BreakpointSerializer",
    # time_travel_ui.py (T-CC-4.3)
    "TimeTravelUIConfig",
    "TimeTravelUI",
    "TimelinePanel",
    "ScrubBar",
    "StepControls",
    "StateDiffView",
    "SnapshotMarkerWidget",
    "TickMarkerWidget",
    "PlayheadWidget",
    "DiffEntry",
    "DiffEntryWidget",
    "DiffTreeNode",
    "DiffType",
    "UIActionType",
    "UIAction",
    "PlaybackState",
    "PlaybackController",
]
