"""
Trinity Pattern Descriptors.

Descriptors control how attributes are read and written. They run on every
attribute access and are the workhorse of the Trinity Pattern.

Descriptor families:
- Storage: Base storage mechanism
- Validation: Type checking, range validation, choices, patterns
- Tracking: Change detection, dirty flags, versioning, diffs
- Networking: Replication, authority, interpolation, prediction
- Caching: Computed values, TTL
- Observable: Change notification, two-way binding
- Persistence: Serialization, transient fields, migration, encryption
- Async: Lazy loading, async loading
- Debug: Profiling, logging, watched fields
"""

from trinity.descriptors.base import (
    BaseDescriptor,
    TrinityDescriptor,
    Computation,
    set_current_computation,
    get_current_computation,
)
from trinity.descriptors.caching import CachedDescriptor, ComputedDescriptor
from trinity.descriptors.composer import DescriptorComposer, DescriptorCompositionError
from trinity.descriptors.storage import StorageDescriptor
from trinity.descriptors.validation import (
    RangeDescriptor,
    ValidatedDescriptor,
    TypeDescriptor,
    ChoiceDescriptor,
    PatternDescriptor,
)
from trinity.descriptors.tracking import (
    TrackedDescriptor,
    VersionedDescriptor,
    DiffDescriptor,
    is_dirty,
    get_dirty_fields,
    clear_dirty,
    clear_dirty_field,
)
from trinity.descriptors.networking import (
    NetworkedDescriptor,
    InterpolatedDescriptor,
    PredictedDescriptor,
    ThrottledNetworkDescriptor,
    get_network_queue,
    clear_network_queue,
    pop_network_updates,
)
from trinity.descriptors.persistence import (
    SerializableDescriptor,
    TransientDescriptor,
    MigratedDescriptor,
    EncryptedDescriptor,
)
from trinity.descriptors.async_descriptors import (
    LazyDescriptor,
    AsyncLoadDescriptor,
    AsyncLoadState,
)
from trinity.descriptors.debug import (
    ProfiledDescriptor,
    LoggedDescriptor,
    WatchedDescriptor,
)
from trinity.descriptors.observable import (
    ObservableDescriptor,
    BoundDescriptor,
    Observer,
    add_observer,
    remove_observer,
    clear_observers,
)
# Phase 7 descriptors
from trinity.descriptors.immutable import ImmutableDescriptor
from trinity.descriptors.indexing import IndexedDescriptor, find_by_index
from trinity.descriptors.atomic import AtomicDescriptor, compare_and_swap
from trinity.descriptors.sparse import SparseDescriptor, sparse_count
from trinity.descriptors.rate_limiting import RateLimitedDescriptor, RateLimitExceeded
from trinity.descriptors.conditional import ConditionalDescriptor, WriteConditionError
from trinity.descriptors.transform import TransformDescriptor
from trinity.descriptors.expiring import ExpiringDescriptor
from trinity.descriptors.audit import AuditDescriptor, get_audit_log, clear_audit_log
from trinity.descriptors.pooled_field import PooledDescriptor, acquire
from trinity.descriptors.priority import PriorityDescriptor
from trinity.descriptors.mirror import MirrorDescriptor
from trinity.descriptors.event_sourced import EventSourcedDescriptor, get_events, replay_events
from trinity.descriptors.batched import BatchedDescriptor, flush_batch
from trinity.descriptors.broadcast import BroadcastDescriptor, subscribe, unsubscribe
from trinity.descriptors.compressed import CompressedDescriptor
from trinity.descriptors.schema import SchemaDescriptor
from trinity.descriptors.proxy import ProxyDescriptor

__all__ = [
    # Base
    "BaseDescriptor",
    "TrinityDescriptor",
    # Read-tracking (for incremental computation)
    "Computation",
    "set_current_computation",
    "get_current_computation",
    # Storage
    "StorageDescriptor",
    # Validation
    "ValidatedDescriptor",
    "RangeDescriptor",
    "TypeDescriptor",
    "ChoiceDescriptor",
    "PatternDescriptor",
    # Tracking
    "TrackedDescriptor",
    "VersionedDescriptor",
    "DiffDescriptor",
    "is_dirty",
    "get_dirty_fields",
    "clear_dirty",
    "clear_dirty_field",
    # Networking
    "NetworkedDescriptor",
    "InterpolatedDescriptor",
    "PredictedDescriptor",
    "ThrottledNetworkDescriptor",
    "get_network_queue",
    "clear_network_queue",
    "pop_network_updates",
    # Caching
    "CachedDescriptor",
    "ComputedDescriptor",
    # Observable
    "ObservableDescriptor",
    "BoundDescriptor",
    "Observer",
    "add_observer",
    "remove_observer",
    "clear_observers",
    # Composition
    "DescriptorComposer",
    "DescriptorCompositionError",
    # Persistence
    "SerializableDescriptor",
    "TransientDescriptor",
    "MigratedDescriptor",
    "EncryptedDescriptor",
    # Async
    "LazyDescriptor",
    "AsyncLoadDescriptor",
    "AsyncLoadState",
    # Debug
    "ProfiledDescriptor",
    "LoggedDescriptor",
    "WatchedDescriptor",
    # Phase 7 — Immutable, Indexed, Atomic, Sparse
    "ImmutableDescriptor",
    "IndexedDescriptor",
    "find_by_index",
    "AtomicDescriptor",
    "compare_and_swap",
    "SparseDescriptor",
    "sparse_count",
    # Phase 7 — Rate limiting, Conditional, Transform
    "RateLimitedDescriptor",
    "RateLimitExceeded",
    "ConditionalDescriptor",
    "WriteConditionError",
    "TransformDescriptor",
    # Phase 7 — Expiring, Audit, Pooled
    "ExpiringDescriptor",
    "AuditDescriptor",
    "get_audit_log",
    "clear_audit_log",
    "PooledDescriptor",
    "acquire",
    # Phase 7 — Priority, Mirror, EventSourced, Batched, Broadcast
    "PriorityDescriptor",
    "MirrorDescriptor",
    "EventSourcedDescriptor",
    "get_events",
    "replay_events",
    "BatchedDescriptor",
    "flush_batch",
    "BroadcastDescriptor",
    "subscribe",
    "unsubscribe",
    # Phase 7 — Compressed, Schema, Proxy
    "CompressedDescriptor",
    "SchemaDescriptor",
    "ProxyDescriptor",
]
