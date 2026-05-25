"""
Hot-Reload System - Live code reloading for the AI Game Engine tooling layer.

This module provides:
- @reloadable decorator for marking classes safe for hot-reload
- File system watching for code changes
- State preservation across reloads using Foundation's Serializer
- Schema hash detection for breaking change prevention
- Pre/post reload callbacks and hooks
- Module dependency tracking for cascade reloads

Usage:
    from engine.tooling.hotreload import (
        reloadable,
        HotReloader,
        ModuleWatcher,
        StatePreserver,
        SchemaHasher,
        ReloadCallbacks,
        DependencyTracker,
    )
"""

from engine.tooling.hotreload.hot_reload import (
    reloadable,
    ReloadError,
    SchemaBreakingChangeError,
    ReloadableClass,
    HotReloader,
)

from engine.tooling.hotreload.module_watcher import (
    ModuleWatcher,
    ModuleChangeEvent,
    ModuleChangeType,
)

from engine.tooling.hotreload.state_preservation import (
    StatePreserver,
    PreservationStrategy,
    StateSnapshot,
)

from engine.tooling.hotreload.schema_hash import (
    SchemaHasher,
    SchemaComparison,
    SchemaChange,
    SchemaChangeType,
)

from engine.tooling.hotreload.reload_callbacks import (
    ReloadCallbacks,
    ReloadPhase,
    ReloadContext,
    CallbackPriority,
)

from engine.tooling.hotreload.dependency_tracker import (
    DependencyTracker,
    DependencyGraph,
    ModuleNode,
)

__all__ = [
    # Hot reload core
    "reloadable",
    "ReloadError",
    "SchemaBreakingChangeError",
    "ReloadableClass",
    "HotReloader",
    # Module watcher
    "ModuleWatcher",
    "ModuleChangeEvent",
    "ModuleChangeType",
    # State preservation
    "StatePreserver",
    "PreservationStrategy",
    "StateSnapshot",
    # Schema hashing
    "SchemaHasher",
    "SchemaComparison",
    "SchemaChange",
    "SchemaChangeType",
    # Callbacks
    "ReloadCallbacks",
    "ReloadPhase",
    "ReloadContext",
    "CallbackPriority",
    # Dependency tracking
    "DependencyTracker",
    "DependencyGraph",
    "ModuleNode",
]
