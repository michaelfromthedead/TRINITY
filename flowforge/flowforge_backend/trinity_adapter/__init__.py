"""Trinity Adapter module for FlowForge Backend.

This module provides integration with the Trinity game engine:
- Introspection of Trinity game objects and components
- Real-time synchronization with game state
- Event handling and callbacks
- Registry API for type introspection
- Mirror API for instance introspection
- EventLog API for event tracking

Phase 3.3.1 Implementation:
- trinity_introspection: Core introspection module for Trinity ECS runtime
"""

# Import from the parent package's trinity_introspection module
from ..trinity_introspection import (
    # Status
    TrinityStatus,
    check_trinity_status,
    initialize_trinity,
    # Registry
    RegisteredType,
    list_registered_types,
    get_type_info,
    # Mirror
    ActiveInstance,
    query_active_instances,
    get_instance_mirror,
    # EventLog
    RecentEvent,
    get_recent_events,
    get_event_statistics,
    # Decorators
    list_decorators,
    # Connection Management
    connect_trinity,
    disconnect_trinity,
    poll_trinity,
    inspector_get,
)

__all__ = [
    # Status
    "TrinityStatus",
    "check_trinity_status",
    "initialize_trinity",
    # Registry
    "RegisteredType",
    "list_registered_types",
    "get_type_info",
    # Mirror
    "ActiveInstance",
    "query_active_instances",
    "get_instance_mirror",
    # EventLog
    "RecentEvent",
    "get_recent_events",
    "get_event_statistics",
    # Decorators
    "list_decorators",
    # Connection Management
    "connect_trinity",
    "disconnect_trinity",
    "poll_trinity",
    "inspector_get",
]
