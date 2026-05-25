"""
Core Foundation - Runtime infrastructure for the AI Game Engine.

This package provides the six systems that form the runtime layer:

Layer 0 (Essential):
    - Mirror: Uniform reflection for any object
    - Serializer: Save/load any object

Layer 1 (Structural):
    - Registry: Type registration and lookup

Layer 2 (Reactive):
    - Tracker: Change detection and undo/redo

Layer 3 (Interactive):
    - Inspector: Object visualization and editing
    - Shell: Live code execution
    - Capabilities: Capability-based security for AI agents and modding

Usage:
    from foundation import mirror, registry, tracker, inspector, shell
    from foundation import to_dict, from_dict, deep_copy
    from foundation import FieldInfo, MethodInfo, Change, Transaction
"""

# Constants (configuration, magic numbers)
from foundation.constants import (
    # Undo/Redo
    MAX_UNDO_STACK_SIZE,
    MAX_REDO_STACK_SIZE,
    # Pool sizes
    DEFAULT_POOL_SIZE,
    MAX_POOL_SIZE,
    # Transaction limits
    MAX_TRANSACTION_CHANGES,
    # Tracker limits
    MAX_CALLBACKS_PER_OBJECT,
    MAX_DIRTY_OBJECTS,
    # UI/Inspector
    INDENT_SPACES,
    HISTORY_FIELD_WIDTH,
    HISTORY_TICK_WIDTH,
    # Hashing
    HASH_LENGTH,
    SHORT_HASH_LENGTH,
    FULL_HASH_LENGTH,
    SCHEMA_HASH_LENGTH,
    # Caching
    DEFAULT_QUERY_CACHE_SIZE,
    DEFAULT_CONTENT_CACHE_SIZE,
    # File backend
    FILE_BACKEND_PREFIX_LENGTH,
    # EventLog
    MAX_CAUSAL_DEPTH,
)

# Layer 0: Essential - Path Utilities
from foundation.paths import (
    PathError,
    parse_path,
    get_path,
    set_path,
)

# Layer 0: Essential - Mirror
from foundation.mirror import (
    mirror,
    schema_hash,
    FieldInfo,
    MethodInfo,
    ObjectMirror,
    ClassMirror,
    STANDARD_METADATA_KEYS,
)

# Layer 0: Essential - Serializer
from foundation.serializer import (
    register_type,
    to_dict,
    from_dict,
    to_bytes,
    from_bytes,
    to_file,
    from_file,
    deep_copy,
    diff,
    patch,
    Delta,
    SerializationContext,
    DeserializationContext,
    SchemaMismatchError,
    set_migration_registry,
    get_migration_registry,
)

# Layer 0: Essential - Migrations
from foundation.migrations import (
    MigrationRegistry,
    register_migration,
    migrate,
    has_migration_path,
    get_migration_path,
    clear_migrations,
)

# Layer 0: Essential - EventLog
from foundation.eventlog import (
    Change as EventChange,  # Renamed to avoid conflict with tracker.Change
    Event,
    EventLog,
    traced,
    set_current_tick,
    get_current_tick,
    get_event_log,
    get_current_event,
    add_change_to_current_event,
    clear_event_log,
)

# Layer 0: Essential - Provenance
from foundation.provenance import (
    ReadRecord,
    DerivationNode,
    ComputedProvenance,
    track_provenance,
    record_input,
    record_read,
    get_current_reads_collector,
    provenance,
    clear_provenance,
    derivation_tree,
)

# Layer 1: Structural - Registry
from foundation.registry import (
    Registry,
    registry,
)

# Layer 2: Reactive - Tracker
from foundation.tracker import (
    tracker,
    Change,
    Transaction,
    Tracker,
    ChangeCallback,
)

# Layer 2: Reactive - Query
from foundation.query import (
    Query,
    QueryCache,
    QuerySubscriber,
    TrackedQueryCache,
    Filter,
    WhereFilter,
    NearFilter,
    HasComponentFilter,
    AndFilter,
    OrFilter,
    NotFilter,
)

# Layer 2: Reactive - Query Cache Mirror
from foundation.query_cache_mirror import (
    QueryInfo,
    QueryCacheMirror,
    mirror_query_cache,
)

# Layer 3: Interactive - Inspector
from foundation.inspector import (
    View,
    UIContext,
    TextUIContext,
    FieldsView,
    RawView,
    JSONView,
    CollectionView,
    InspectorPanel,
    Inspector,
    inspector,
    HistoryEntry,
)

# Layer 3: Interactive - Inspector Views (History, Causality, Provenance)
from foundation.inspector_views import (
    HistoryView,
    CausalityView,
    ProvenanceView,
    RootCauseSummary,
    register_inspector_views,
)

# Layer 3: Interactive - Shell
from foundation.shell import (
    ExecutionResult,
    Shell,
    shell,
    inspect,
)

# Layer 3: Interactive - Capabilities (Security)
from foundation.capabilities import (
    Capability,
    CapabilitySet,
    CapabilityError,
    SecureContext,
    require_capability,
    with_capabilities,
    check_capability,
    assert_capability,
    get_current_capabilities,
    CAPS_NONE,
    CAPS_READONLY,
    CAPS_READWRITE,
    CAPS_FULL,
)

# Layer 3: Interactive - SecureShell (Shell with capability enforcement)
from foundation.secure_shell import (
    SecureShell,
    create_readonly_shell,
    create_sandbox_shell,
    create_full_shell,
)

# Layer 4: Integration - Bridge (Trinity <-> Foundation)
from foundation.bridge import (
    get_trinity_registry,
    create_world_from_trinity,
    create_ai_interface,
    create_shell,
    TrinityWorldAdapter,
)

# Layer 0: Essential - ContentStore (Content-addressable storage)
from foundation.content_store import (
    ContentHash,
    ContentStore,
    MemoryBackend,
    FileBackend,
    ContentDiffer,
    Difference,
)

# Layer 0: Essential - DeltaSync (Minimal change patches)
from foundation.delta_sync import (
    DeltaPatch,
    DeltaSync,
)

__version__ = "0.1.0"

__all__ = [
    # Constants
    "MAX_UNDO_STACK_SIZE",
    "MAX_REDO_STACK_SIZE",
    "DEFAULT_POOL_SIZE",
    "MAX_POOL_SIZE",
    "MAX_TRANSACTION_CHANGES",
    "MAX_CALLBACKS_PER_OBJECT",
    "MAX_DIRTY_OBJECTS",
    "INDENT_SPACES",
    "HISTORY_FIELD_WIDTH",
    "HISTORY_TICK_WIDTH",
    "HASH_LENGTH",
    "SHORT_HASH_LENGTH",
    "FULL_HASH_LENGTH",
    "SCHEMA_HASH_LENGTH",
    "DEFAULT_QUERY_CACHE_SIZE",
    "DEFAULT_CONTENT_CACHE_SIZE",
    "FILE_BACKEND_PREFIX_LENGTH",
    "MAX_CAUSAL_DEPTH",
    # Path utilities (Layer 0)
    "PathError",
    "parse_path",
    "get_path",
    "set_path",
    # Mirror system (Layer 0)
    "mirror",
    "schema_hash",
    "FieldInfo",
    "MethodInfo",
    "ObjectMirror",
    "ClassMirror",
    "STANDARD_METADATA_KEYS",
    # Serializer system (Layer 0)
    "register_type",
    "to_dict",
    "from_dict",
    "to_bytes",
    "from_bytes",
    "to_file",
    "from_file",
    "deep_copy",
    "diff",
    "patch",
    "Delta",
    "SerializationContext",
    "DeserializationContext",
    "SchemaMismatchError",
    "set_migration_registry",
    "get_migration_registry",
    # Migrations system (Layer 0)
    "MigrationRegistry",
    "register_migration",
    "migrate",
    "has_migration_path",
    "get_migration_path",
    "clear_migrations",
    # EventLog system (Layer 0)
    "EventChange",
    "Event",
    "EventLog",
    "traced",
    "set_current_tick",
    "get_current_tick",
    "get_event_log",
    "get_current_event",
    "add_change_to_current_event",
    "clear_event_log",
    # Provenance system (Layer 0)
    "ReadRecord",
    "DerivationNode",
    "ComputedProvenance",
    "track_provenance",
    "record_input",
    "record_read",
    "get_current_reads_collector",
    "provenance",
    "clear_provenance",
    "derivation_tree",
    # Registry system (Layer 1)
    "Registry",
    "registry",
    # Tracker system (Layer 2)
    "tracker",
    "Change",
    "Transaction",
    "Tracker",
    "ChangeCallback",
    # Query system (Layer 2)
    "Query",
    "QueryCache",
    "QuerySubscriber",
    "TrackedQueryCache",
    "Filter",
    "WhereFilter",
    "NearFilter",
    "HasComponentFilter",
    "AndFilter",
    "OrFilter",
    "NotFilter",
    # Query Cache Mirror (Layer 2)
    "QueryInfo",
    "QueryCacheMirror",
    "mirror_query_cache",
    # Inspector system (Layer 3)
    "View",
    "UIContext",
    "TextUIContext",
    "FieldsView",
    "RawView",
    "JSONView",
    "CollectionView",
    "InspectorPanel",
    "Inspector",
    "inspector",
    "HistoryEntry",
    # Inspector Views (Layer 3)
    "HistoryView",
    "CausalityView",
    "ProvenanceView",
    "RootCauseSummary",
    "register_inspector_views",
    # Shell system (Layer 3)
    "ExecutionResult",
    "Shell",
    "shell",
    "inspect",
    # Capabilities system (Layer 3)
    "Capability",
    "CapabilitySet",
    "CapabilityError",
    "SecureContext",
    "require_capability",
    "with_capabilities",
    "check_capability",
    "assert_capability",
    "get_current_capabilities",
    "CAPS_NONE",
    "CAPS_READONLY",
    "CAPS_READWRITE",
    "CAPS_FULL",
    # SecureShell system (Layer 3)
    "SecureShell",
    "create_readonly_shell",
    "create_sandbox_shell",
    "create_full_shell",
    # Bridge system (Layer 4)
    "get_trinity_registry",
    "create_world_from_trinity",
    "create_ai_interface",
    "create_shell",
    "TrinityWorldAdapter",
    # ContentStore system (Layer 0)
    "ContentHash",
    "ContentStore",
    "MemoryBackend",
    "FileBackend",
    "ContentDiffer",
    "Difference",
    # DeltaSync system (Layer 0)
    "DeltaPatch",
    "DeltaSync",
]
