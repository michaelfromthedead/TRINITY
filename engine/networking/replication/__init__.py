"""Replication system for network state synchronization.

This module provides property-level replication between server and clients,
including relevancy filtering, bandwidth management, and actor channels.

Components:
    - NetGUID: Unique network identifiers for replicated objects
    - PropertyReplication: Per-property change detection and serialization
    - Relevancy: Interest management for filtering updates by distance/owner
    - Bandwidth: Budget allocation and priority scheduling
    - ReplicationManager: Central coordinator for entity replication
    - ActorChannel: Per-entity replication streams
"""

from .net_guid import (
    NetGUID,
    NetGUIDManager,
    GUIDAuthority,
    INVALID_GUID,
    NULL_GUID,
)

from .property_replication import (
    ReplicatedProperty,
    PropertyReplicationGroup,
    ReplicationCondition,
    ChangeNotifyMode,
    create_replicated_property,
    register_serializer,
)

from .relevancy import (
    RelevancyType,
    RelevancyResult,
    InterestArea,
    AlwaysRelevant,
    OwnerRelevant,
    RadiusRelevancy,
    GridRelevancy,
    CustomRelevancy,
    CompositeRelevancy,
    RelevancyManager,
    DEFAULT_RELEVANCY_RADIUS,
    DEFAULT_GRID_CELL_SIZE,
)

from .bandwidth import (
    EntityPriority,
    BandwidthBudget,
    PriorityQueue,
    PrioritizedEntity,
    BandwidthManager,
    allocate_bandwidth,
    allocate_bandwidth_fair,
    DEFAULT_MAX_BPS,
    DEFAULT_BURST_BPS,
)

from .replication_manager import (
    ReplicationManager,
    ReplicatedEntity,
    ReplicationRole,
    EntityState,
)

from .actor_channel import (
    ActorChannel,
    ActorChannelManager,
    ChannelState,
    ChannelCloseReason,
    ChannelMessage,
)

__all__ = [
    # Net GUID
    'NetGUID',
    'NetGUIDManager',
    'GUIDAuthority',
    'INVALID_GUID',
    'NULL_GUID',

    # Property Replication
    'ReplicatedProperty',
    'PropertyReplicationGroup',
    'ReplicationCondition',
    'ChangeNotifyMode',
    'create_replicated_property',
    'register_serializer',

    # Relevancy
    'RelevancyType',
    'RelevancyResult',
    'InterestArea',
    'AlwaysRelevant',
    'OwnerRelevant',
    'RadiusRelevancy',
    'GridRelevancy',
    'CustomRelevancy',
    'CompositeRelevancy',
    'RelevancyManager',
    'DEFAULT_RELEVANCY_RADIUS',
    'DEFAULT_GRID_CELL_SIZE',

    # Bandwidth
    'EntityPriority',
    'BandwidthBudget',
    'PriorityQueue',
    'PrioritizedEntity',
    'BandwidthManager',
    'allocate_bandwidth',
    'allocate_bandwidth_fair',
    'DEFAULT_MAX_BPS',
    'DEFAULT_BURST_BPS',

    # Replication Manager
    'ReplicationManager',
    'ReplicatedEntity',
    'ReplicationRole',
    'EntityState',

    # Actor Channel
    'ActorChannel',
    'ActorChannelManager',
    'ChannelState',
    'ChannelCloseReason',
    'ChannelMessage',
]
