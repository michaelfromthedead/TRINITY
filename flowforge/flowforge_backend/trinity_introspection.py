"""Trinity ECS Runtime Introspection Module.

Provides live connection to Trinity ECS runtime for introspection:
- Check Trinity availability/importability
- Initialize Trinity in debug/introspection mode
- List registered types via Registry API
- Query active instances via Mirror API
- Get recent events via EventLog API

All functions handle cases where Trinity is not installed gracefully.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class TrinityStatus:
    """Status of Trinity ECS runtime availability."""
    available: bool
    version: Optional[str] = None
    foundation_available: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RegisteredType:
    """Information about a registered Trinity type."""
    name: str
    qualified_name: str
    category: str  # "component", "system", "resource", "event", "asset", "protocol", "state"
    type_id: int
    field_types: dict[str, str] = field(default_factory=dict)
    field_defaults: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ActiveInstance:
    """Information about an active instance in the runtime."""
    type_name: str
    instance_id: int
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RecentEvent:
    """Information about a recent event from the EventLog."""
    tick: int
    operation: str
    entity: Optional[int] = None
    changes: list[dict[str, Any]] = field(default_factory=list)
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# Status Check
# =============================================================================

def check_trinity_status() -> TrinityStatus:
    """Check if Trinity ECS runtime is available and importable.

    Returns:
        TrinityStatus with availability information.
    """
    try:
        import trinity
        trinity_version = getattr(trinity, "__version__", "unknown")

        # Check if foundation is also available
        foundation_available = False
        try:
            import foundation
            foundation_available = True
        except ImportError:
            pass

        return TrinityStatus(
            available=True,
            version=trinity_version,
            foundation_available=foundation_available,
        )
    except ImportError as e:
        return TrinityStatus(
            available=False,
            error=f"Trinity not installed: {e}",
        )
    except Exception as e:
        return TrinityStatus(
            available=False,
            error=f"Error checking Trinity: {e}",
        )


def initialize_trinity(debug_mode: bool = True) -> dict[str, Any]:
    """Initialize Trinity in debug/introspection mode.

    Args:
        debug_mode: Whether to enable debug/introspection features.

    Returns:
        Dict with initialization status and available features.
    """
    status = check_trinity_status()
    if not status.available:
        return {
            "success": False,
            "error": status.error,
            "features": [],
        }

    features = []

    try:
        import trinity
        features.append("trinity.core")

        # Check for registry capabilities
        from trinity.metaclasses import (
            ComponentMeta,
            SystemMeta,
            ResourceMeta,
            EventMeta,
            AssetMeta,
            ProtocolMeta,
            StateMeta,
        )
        features.append("trinity.registry")

        # Check for decorator registry
        from trinity.decorators.registry import registry as decorator_registry
        features.append("trinity.decorators")

        if status.foundation_available:
            import foundation
            features.append("foundation.mirror")
            features.append("foundation.registry")
            features.append("foundation.eventlog")

        return {
            "success": True,
            "debug_mode": debug_mode,
            "features": features,
            "trinity_version": status.version,
            "foundation_available": status.foundation_available,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to initialize Trinity: {e}",
            "features": features,
        }


# =============================================================================
# Registry API Wrappers
# =============================================================================

def list_registered_types(
    category: Optional[str] = None,
    search: Optional[str] = None,
) -> dict[str, Any]:
    """List all registered Trinity types.

    Args:
        category: Optional filter by category ("component", "system", etc.)
        search: Optional search term to filter by name.

    Returns:
        Dict with list of registered types and metadata.
    """
    status = check_trinity_status()
    if not status.available:
        return {
            "success": False,
            "error": status.error,
            "types": [],
            "categories": [],
        }

    try:
        from trinity.metaclasses import (
            ComponentMeta,
            SystemMeta,
            ResourceMeta,
            EventMeta,
            AssetMeta,
            ProtocolMeta,
            StateMeta,
            EngineMeta,
        )

        registered_types: list[RegisteredType] = []

        # Collect components
        if category is None or category == "component":
            for type_id, cls in ComponentMeta._registry.items():
                reg_type = RegisteredType(
                    name=cls.__name__,
                    qualified_name=getattr(cls, "_component_name", f"{cls.__module__}.{cls.__name__}"),
                    category="component",
                    type_id=type_id,
                    field_types={k: str(v) for k, v in getattr(cls, "_field_types", {}).items()},
                    field_defaults=_safe_serialize_defaults(getattr(cls, "_field_defaults", {})),
                    metadata={
                        "track_changes": getattr(cls, "_track_changes", False),
                        "has_network_config": getattr(cls, "_network_config", None) is not None,
                    },
                )
                registered_types.append(reg_type)

        # Collect systems
        if category is None or category == "system":
            for type_id, cls in SystemMeta._registry.items():
                reg_type = RegisteredType(
                    name=cls.__name__,
                    qualified_name=getattr(cls, "_system_name", f"{cls.__module__}.{cls.__name__}"),
                    category="system",
                    type_id=type_id,
                    metadata={
                        "reads": [c.__name__ for c in getattr(cls, "_reads", ())],
                        "writes": [c.__name__ for c in getattr(cls, "_writes", ())],
                        "exclusive": getattr(cls, "_exclusive", False),
                        "priority": getattr(cls, "_priority", 0),
                        "can_parallelize": getattr(cls, "_can_parallelize", False),
                    },
                )
                registered_types.append(reg_type)

        # Collect resources
        if category is None or category == "resource":
            for type_id, cls in ResourceMeta._registry.items():
                reg_type = RegisteredType(
                    name=cls.__name__,
                    qualified_name=getattr(cls, "_resource_name", f"{cls.__module__}.{cls.__name__}"),
                    category="resource",
                    type_id=type_id,
                    metadata={
                        "priority": getattr(cls, "_resource_priority", 0),
                    },
                )
                registered_types.append(reg_type)

        # Collect events
        if category is None or category == "event":
            for type_id, cls in EventMeta._registry.items():
                reg_type = RegisteredType(
                    name=cls.__name__,
                    qualified_name=getattr(cls, "_event_name", f"{cls.__module__}.{cls.__name__}"),
                    category="event",
                    type_id=type_id,
                    field_types={k: str(v) for k, v in getattr(cls, "_event_fields", {}).items()},
                    metadata={
                        "priority": getattr(cls, "_event_priority", 0),
                        "channels": list(getattr(cls, "_event_channels", ())),
                        "pooled": getattr(cls, "_event_pooled", False),
                    },
                )
                registered_types.append(reg_type)

        # Collect assets
        if category is None or category == "asset":
            for type_id, cls in AssetMeta._registry.items():
                reg_type = RegisteredType(
                    name=cls.__name__,
                    qualified_name=getattr(cls, "_asset_name", f"{cls.__module__}.{cls.__name__}"),
                    category="asset",
                    type_id=type_id,
                    metadata={
                        "extensions": list(getattr(cls, "_asset_extensions", ())),
                        "hot_reload": getattr(cls, "_asset_hot_reload", False),
                    },
                )
                registered_types.append(reg_type)

        # Collect protocols
        if category is None or category == "protocol":
            for type_id, cls in ProtocolMeta._registry.items():
                reg_type = RegisteredType(
                    name=cls.__name__,
                    qualified_name=getattr(cls, "_protocol_qualified_name", f"{cls.__module__}.{cls.__name__}"),
                    category="protocol",
                    type_id=type_id,
                    metadata={
                        "version": getattr(cls, "_protocol_version", 0),
                        "min_version": getattr(cls, "_protocol_min_version", 1),
                    },
                )
                registered_types.append(reg_type)

        # Collect states
        if category is None or category == "state":
            for type_id, cls in StateMeta._registry.items():
                reg_type = RegisteredType(
                    name=cls.__name__,
                    qualified_name=getattr(cls, "_state_qualified_name", f"{cls.__module__}.{cls.__name__}"),
                    category="state",
                    type_id=type_id,
                    metadata={
                        "transitions": list(getattr(cls, "_state_transitions", set())),
                    },
                )
                registered_types.append(reg_type)

        # Apply search filter
        if search:
            search_lower = search.lower()
            registered_types = [
                t for t in registered_types
                if search_lower in t.name.lower() or search_lower in t.qualified_name.lower()
            ]

        # Get unique categories
        categories = sorted(set(t.category for t in registered_types))

        return {
            "success": True,
            "types": [t.to_dict() for t in registered_types],
            "categories": categories,
            "total_count": len(registered_types),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to list registered types: {e}",
            "types": [],
            "categories": [],
        }


def get_type_info(qualified_name: str) -> dict[str, Any]:
    """Get detailed information about a specific registered type.

    Args:
        qualified_name: The fully qualified type name.

    Returns:
        Dict with detailed type information.
    """
    status = check_trinity_status()
    if not status.available:
        return {
            "success": False,
            "error": status.error,
        }

    try:
        from trinity.metaclasses import EngineMeta

        all_types = EngineMeta.get_all_types()
        cls = all_types.get(qualified_name)

        if cls is None:
            return {
                "success": False,
                "error": f"Type not found: {qualified_name}",
            }

        # Determine category from metaclass
        category = "unknown"
        metaclass_name = type(cls).__name__
        category_map = {
            "ComponentMeta": "component",
            "SystemMeta": "system",
            "ResourceMeta": "resource",
            "EventMeta": "event",
            "AssetMeta": "asset",
            "ProtocolMeta": "protocol",
            "StateMeta": "state",
        }
        category = category_map.get(metaclass_name, "unknown")

        # Get source location if available
        source_info = {}
        try:
            import inspect
            source_file = inspect.getfile(cls)
            source_lines = inspect.getsourcelines(cls)
            source_info = {
                "file": source_file,
                "line": source_lines[1] if source_lines else None,
            }
        except Exception:
            pass

        return {
            "success": True,
            "name": cls.__name__,
            "qualified_name": qualified_name,
            "category": category,
            "module": cls.__module__,
            "doc": cls.__doc__,
            "source": source_info,
            "metaclass": metaclass_name,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get type info: {e}",
        }


# =============================================================================
# Mirror API Wrappers (Instance Introspection)
# =============================================================================

def query_active_instances(
    type_name: Optional[str] = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Query active instances in the runtime via Foundation's registry.

    Args:
        type_name: Optional filter by type name.
        limit: Maximum number of instances to return.

    Returns:
        Dict with list of active instances.
    """
    status = check_trinity_status()
    if not status.available:
        return {
            "success": False,
            "error": status.error,
            "instances": [],
        }

    if not status.foundation_available:
        return {
            "success": False,
            "error": "Foundation not available - instance tracking requires Foundation",
            "instances": [],
        }

    try:
        from foundation import registry, mirror

        instances: list[ActiveInstance] = []
        instance_id_counter = 0

        # Get all registered types and check which ones have instance tracking
        for cls in registry.all_types():
            # Get the registered name for this type
            cls_name = registry.get_name(cls)
            if not cls_name:
                continue

            # Skip if filtering by type and this doesn't match
            if type_name and type_name.lower() not in cls_name.lower():
                continue

            # Check if this type has instance tracking enabled
            instance_count = registry.instance_count(cls)
            if instance_count == 0:
                continue

            # Get instances for this type
            for instance in registry.instances(cls):
                if len(instances) >= limit:
                    break

                instance_id_counter += 1

                # Use mirror to get field values
                try:
                    obj_mirror = mirror(instance)
                    fields = {}
                    for field_info in obj_mirror.fields():
                        try:
                            fields[field_info.name] = _safe_serialize_value(field_info.value)
                        except Exception:
                            fields[field_info.name] = "<unserializable>"
                except Exception:
                    fields = {"_error": "Failed to mirror instance"}

                instances.append(ActiveInstance(
                    type_name=cls_name,
                    instance_id=id(instance),  # Use Python object id
                    fields=fields,
                ))

            if len(instances) >= limit:
                break

        return {
            "success": True,
            "instances": [i.to_dict() for i in instances],
            "total_count": len(instances),
            "limited": len(instances) >= limit,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to query instances: {e}",
            "instances": [],
        }


def get_instance_mirror(type_name: str, instance_id: int) -> dict[str, Any]:
    """Get detailed mirror information for a specific instance.

    Args:
        type_name: The type name of the instance.
        instance_id: The instance ID (Python object id).

    Returns:
        Dict with detailed instance information via mirror.
    """
    status = check_trinity_status()
    if not status.available or not status.foundation_available:
        return {
            "success": False,
            "error": "Trinity and Foundation required for instance mirroring",
        }

    try:
        from foundation import registry, mirror

        # Get the type class
        cls = registry.get(type_name)
        if cls is None:
            return {
                "success": False,
                "error": f"Type not found: {type_name}",
            }

        # Find instance by id
        instance = None
        for inst in registry.instances(cls):
            if id(inst) == instance_id:
                instance = inst
                break

        if instance is None:
            return {
                "success": False,
                "error": f"Instance not found: {type_name}#{instance_id}",
            }

        obj_mirror = mirror(instance)

        fields = []
        for field_info in obj_mirror.fields():
            fields.append({
                "name": field_info.name,
                "type": str(field_info.type_hint),
                "value": _safe_serialize_value(field_info.value),
                "readonly": field_info.readonly,
            })

        methods = []
        for method_info in obj_mirror.methods():
            methods.append({
                "name": method_info.name,
                "signature": str(method_info.signature),
                "doc": method_info.doc,
            })

        return {
            "success": True,
            "type_name": type_name,
            "instance_id": instance_id,
            "fields": fields,
            "methods": methods,
            "schema_hash": obj_mirror.schema_hash(),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to mirror instance: {e}",
        }


# =============================================================================
# EventLog API Wrappers
# =============================================================================

def get_recent_events(
    limit: int = 50,
    entity: Optional[int] = None,
    operation: Optional[str] = None,
    tick: Optional[int] = None,
) -> dict[str, Any]:
    """Get recent events from the EventLog.

    Args:
        limit: Maximum number of events to return.
        entity: Optional filter by entity ID.
        operation: Optional filter by operation name.
        tick: Optional filter by specific tick.

    Returns:
        Dict with list of recent events.
    """
    status = check_trinity_status()
    if not status.available:
        return {
            "success": False,
            "error": status.error,
            "events": [],
        }

    if not status.foundation_available:
        return {
            "success": False,
            "error": "Foundation not available - EventLog requires Foundation",
            "events": [],
        }

    try:
        from foundation import get_event_log, get_current_tick

        event_log = get_event_log()
        current_tick = get_current_tick()

        events: list[RecentEvent] = []

        # Get events based on filters using the correct API
        if entity is not None:
            raw_events = event_log.events_for_entity(entity)
        elif operation is not None:
            raw_events = event_log.events_for_operation(operation)
        elif tick is not None:
            raw_events = event_log.events_at(tick)
        else:
            # Get all events (most recent first)
            raw_events = list(reversed(event_log._events))

        for event in raw_events[:limit]:
            changes = []
            for change in event.changes:
                changes.append({
                    "entity": change.entity,
                    "field": change.field,
                    "old_value": _safe_serialize_value(change.old_value),
                    "new_value": _safe_serialize_value(change.new_value),
                })

            events.append(RecentEvent(
                tick=event.tick,
                operation=event.operation,
                entity=event.entity,
                changes=changes,
                depth=event.depth,
            ))

        return {
            "success": True,
            "events": [e.to_dict() for e in events],
            "total_count": len(events),
            "current_tick": current_tick,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get recent events: {e}",
            "events": [],
        }


def get_event_statistics() -> dict[str, Any]:
    """Get statistics about the EventLog.

    Returns:
        Dict with event statistics.
    """
    status = check_trinity_status()
    if not status.available or not status.foundation_available:
        return {
            "success": False,
            "error": "Trinity and Foundation required for event statistics",
        }

    try:
        from foundation import get_event_log, get_current_tick

        event_log = get_event_log()

        return {
            "success": True,
            "total_events": len(event_log._events),
            "entities_with_events": len(event_log._by_entity),
            "unique_operations": len(event_log._by_operation),
            "ticks_with_events": len(event_log._by_tick),
            "current_tick": get_current_tick(),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get event statistics: {e}",
        }


# =============================================================================
# Decorator Registry Introspection
# =============================================================================

def list_decorators(tier: Optional[int] = None) -> dict[str, Any]:
    """List all registered Trinity decorators.

    Args:
        tier: Optional filter by tier number.

    Returns:
        Dict with list of decorators.
    """
    status = check_trinity_status()
    if not status.available:
        return {
            "success": False,
            "error": status.error,
            "decorators": [],
        }

    try:
        from trinity.decorators.registry import registry as decorator_registry, Tier

        decorators = []

        if tier is not None:
            # Get decorators for specific tier
            try:
                tier_enum = Tier(tier)
                specs = decorator_registry.by_tier(tier_enum)
                for spec in specs:
                    decorators.append({
                        "name": spec.name,
                        "tier": spec.tier.value,
                        "tier_name": spec.tier.name,
                        "foundation": spec.foundation,
                        "unique": spec.unique,
                        "requires": list(spec.requires),
                        "excludes": list(spec.excludes),
                        "doc": spec.doc,
                        "target_types": list(spec.target_types),
                    })
            except ValueError:
                pass
        else:
            # Get all decorators
            for name, spec in decorator_registry.all().items():
                decorators.append({
                    "name": spec.name,
                    "tier": spec.tier.value,
                    "tier_name": spec.tier.name,
                    "foundation": spec.foundation,
                    "unique": spec.unique,
                    "requires": list(spec.requires),
                    "excludes": list(spec.excludes),
                    "doc": spec.doc,
                    "target_types": list(spec.target_types),
                })

        # Get tier info
        tiers = [
            {"value": t.value, "name": t.name}
            for t in Tier
        ]

        return {
            "success": True,
            "decorators": decorators,
            "total_count": len(decorators),
            "tiers": tiers,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to list decorators: {e}",
            "decorators": [],
        }


# =============================================================================
# Connection Management (Live Introspection)
# =============================================================================

# Global connection state for Trinity runtime
_connection_state: dict[str, Any] = {
    "connected": False,
    "session_id": None,
    "debug_mode": False,
    "pending_events": [],
}


def connect_trinity(debug_mode: bool = True) -> dict[str, Any]:
    """Connect to Trinity runtime for live introspection.

    Initializes a connection to the Trinity ECS runtime, enabling
    live introspection of types, instances, and events.

    Args:
        debug_mode: Whether to enable debug/introspection features.

    Returns:
        Dict with connection status and session information.
    """
    global _connection_state

    status = check_trinity_status()
    if not status.available:
        return {
            "success": False,
            "error": status.error,
            "connected": False,
        }

    # Initialize Trinity if not already done
    init_result = initialize_trinity(debug_mode=debug_mode)
    if not init_result.get("success"):
        return {
            "success": False,
            "error": init_result.get("error", "Failed to initialize Trinity"),
            "connected": False,
        }

    # Generate session ID
    import time
    import hashlib
    session_id = hashlib.md5(f"trinity-{time.time()}".encode()).hexdigest()[:12]

    _connection_state = {
        "connected": True,
        "session_id": session_id,
        "debug_mode": debug_mode,
        "pending_events": [],
        "connected_at": time.time(),
        "features": init_result.get("features", []),
    }

    return {
        "success": True,
        "connected": True,
        "session_id": session_id,
        "debug_mode": debug_mode,
        "features": init_result.get("features", []),
        "trinity_version": status.version,
        "foundation_available": status.foundation_available,
    }


def disconnect_trinity() -> dict[str, Any]:
    """Disconnect from Trinity runtime.

    Closes the connection to the Trinity ECS runtime and cleans up
    any associated resources.

    Returns:
        Dict with disconnection status.
    """
    global _connection_state

    if not _connection_state.get("connected"):
        return {
            "success": True,
            "message": "Not connected",
            "connected": False,
        }

    session_id = _connection_state.get("session_id")

    # Reset connection state
    _connection_state = {
        "connected": False,
        "session_id": None,
        "debug_mode": False,
        "pending_events": [],
    }

    return {
        "success": True,
        "message": "Disconnected from Trinity runtime",
        "connected": False,
        "previous_session_id": session_id,
    }


def poll_trinity(
    include_events: bool = True,
    include_instances: bool = False,
    event_limit: int = 50,
) -> dict[str, Any]:
    """Poll Trinity runtime for updates.

    Returns current state and any pending events since last poll.
    This is the primary method for live introspection updates.

    Args:
        include_events: Whether to include recent events.
        include_instances: Whether to include instance snapshots.
        event_limit: Maximum number of events to return.

    Returns:
        Dict with current state and any updates.
    """
    global _connection_state

    if not _connection_state.get("connected"):
        return {
            "success": False,
            "error": "Not connected to Trinity runtime",
            "connected": False,
        }

    result: dict[str, Any] = {
        "success": True,
        "connected": True,
        "session_id": _connection_state.get("session_id"),
    }

    # Get current status
    status = check_trinity_status()
    result["status"] = status.to_dict()

    # Include events if requested
    if include_events:
        events_result = get_recent_events(limit=event_limit)
        result["events"] = events_result.get("events", [])
        result["current_tick"] = events_result.get("current_tick")

    # Include instance snapshots if requested
    if include_instances:
        instances_result = query_active_instances(limit=100)
        result["instances"] = instances_result.get("instances", [])
        result["instance_count"] = instances_result.get("total_count", 0)

    return result


def inspector_get(
    target_type: str,
    target_id: Optional[int] = None,
    qualified_name: Optional[str] = None,
) -> dict[str, Any]:
    """Get detailed inspector information for a target.

    Unified inspector API that can get information about types,
    instances, or decorators.

    Args:
        target_type: Type of target ("type", "instance", "decorator")
        target_id: Instance ID (required for "instance" target_type)
        qualified_name: Qualified name for type or decorator lookup

    Returns:
        Dict with detailed inspector information.
    """
    status = check_trinity_status()
    if not status.available:
        return {
            "success": False,
            "error": status.error,
        }

    if target_type == "type":
        if not qualified_name:
            return {
                "success": False,
                "error": "qualified_name required for type inspection",
            }
        # Use enhanced type info with hierarchy and decorators
        return get_detailed_type_info(qualified_name)

    elif target_type == "instance":
        if not qualified_name or target_id is None:
            return {
                "success": False,
                "error": "qualified_name and target_id required for instance inspection",
            }
        return get_instance_mirror(qualified_name, target_id)

    elif target_type == "decorator":
        # Get decorator info from list
        decorators_result = list_decorators()
        if not decorators_result.get("success"):
            return decorators_result

        decorators = decorators_result.get("decorators", [])

        if qualified_name:
            # Find specific decorator
            for dec in decorators:
                if dec.get("name") == qualified_name:
                    return {
                        "success": True,
                        "target_type": "decorator",
                        "decorator": dec,
                    }
            return {
                "success": False,
                "error": f"Decorator not found: {qualified_name}",
            }

        # Return all decorators if no specific one requested
        return {
            "success": True,
            "target_type": "decorator",
            "decorators": decorators,
            "count": len(decorators),
        }

    else:
        return {
            "success": False,
            "error": f"Unknown target_type: {target_type}. Must be 'type', 'instance', or 'decorator'",
        }


def get_detailed_type_info(qualified_name: str) -> dict[str, Any]:
    """Get detailed information about a specific registered type.

    Enhanced version that includes hierarchy, decorators, and metaclass info.

    Args:
        qualified_name: The fully qualified type name or simple name.

    Returns:
        Dict with detailed type information including hierarchy and decorators.
    """
    status = check_trinity_status()
    if not status.available:
        return {
            "success": False,
            "error": status.error,
        }

    try:
        from trinity.metaclasses import EngineMeta

        # Try to find the type by qualified name or simple name
        all_types = EngineMeta.get_all_types()
        cls = all_types.get(qualified_name)

        # If not found by qualified name, try to find by simple name
        if cls is None:
            simple_name = qualified_name.split('.')[-1] if '.' in qualified_name else qualified_name
            for type_qname, type_cls in all_types.items():
                if type_cls.__name__ == simple_name:
                    cls = type_cls
                    qualified_name = type_qname
                    break

        if cls is None:
            return {
                "success": False,
                "error": f"Type not found: {qualified_name}",
            }

        # Determine category from metaclass
        category = "unknown"
        metaclass_name = type(cls).__name__
        category_map = {
            "ComponentMeta": "component",
            "SystemMeta": "system",
            "ResourceMeta": "resource",
            "EventMeta": "event",
            "AssetMeta": "asset",
            "ProtocolMeta": "protocol",
            "StateMeta": "state",
        }
        category = category_map.get(metaclass_name, "unknown")

        # Get source location if available
        source_info = {"file": "", "line": None}
        try:
            import inspect
            source_file = inspect.getfile(cls)
            try:
                source_lines = inspect.getsourcelines(cls)
                source_info = {
                    "file": source_file,
                    "line": source_lines[1] if source_lines else None,
                }
            except OSError:
                source_info = {"file": source_file, "line": None}
        except Exception:
            pass

        # Build hierarchy from bases
        hierarchy = []
        trinity_bases = ["Component", "System", "Resource", "Event", "Entity", "World", "Asset", "Protocol", "State"]
        for base in cls.__bases__:
            if base.__name__ != "object":
                hierarchy.append({
                    "name": base.__name__,
                    "module": base.__module__,
                    "isTrinityBase": base.__name__ in trinity_bases,
                })

        # Get decorators from decorator registry if available
        decorators = []
        try:
            from trinity.decorators.registry import registry as decorator_registry

            # Check which decorators apply to this type
            if category != "unknown":
                # Add the main type decorator
                spec = decorator_registry.get(category)
                if spec:
                    decorators.append({
                        "name": spec.name,
                        "tier": spec.tier.value,
                        "tierName": spec.tier.name,
                        "foundation": spec.foundation,
                        "doc": spec.doc,
                    })

            # Add any additional decorators from class attributes
            if hasattr(cls, "_decorators"):
                for dec_name in cls._decorators:
                    spec = decorator_registry.get(dec_name)
                    if spec and dec_name != category:
                        decorators.append({
                            "name": spec.name,
                            "tier": spec.tier.value,
                            "tierName": spec.tier.name,
                            "foundation": spec.foundation,
                            "doc": spec.doc,
                        })
        except ImportError:
            # Decorator registry not available, use category as decorator
            if category != "unknown":
                decorators.append({
                    "name": category,
                    "tier": 1,
                    "tierName": "FOUNDATION",
                    "foundation": True,
                    "doc": None,
                })

        # Get field types for components
        field_types = {}
        field_defaults = {}
        if hasattr(cls, "_field_types"):
            field_types = {k: str(v) for k, v in cls._field_types.items()}
        if hasattr(cls, "_field_defaults"):
            field_defaults = _safe_serialize_defaults(cls._field_defaults)

        # Get metadata
        metadata = {}
        if category == "component":
            metadata["track_changes"] = getattr(cls, "_track_changes", False)
            metadata["has_network_config"] = getattr(cls, "_network_config", None) is not None
        elif category == "system":
            metadata["reads"] = [c.__name__ for c in getattr(cls, "_reads", ())]
            metadata["writes"] = [c.__name__ for c in getattr(cls, "_writes", ())]
            metadata["exclusive"] = getattr(cls, "_exclusive", False)
            metadata["priority"] = getattr(cls, "_priority", 0)
        elif category == "event":
            metadata["priority"] = getattr(cls, "_event_priority", 0)
            metadata["channels"] = list(getattr(cls, "_event_channels", ()))
            metadata["pooled"] = getattr(cls, "_event_pooled", False)

        return {
            "success": True,
            "name": cls.__name__,
            "qualified_name": qualified_name,
            "category": category,
            "module": cls.__module__,
            "doc": cls.__doc__,
            "source": source_info,
            "metaclass": metaclass_name,
            "hierarchy": hierarchy,
            "decorators": decorators,
            "field_types": field_types,
            "field_defaults": field_defaults,
            "metadata": metadata,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get type info: {e}",
        }


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_serialize_value(value: Any) -> Any:
    """Safely serialize a value for JSON output."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_serialize_value(v) for k, v in value.items()}
    if isinstance(value, set):
        return list(value)
    if isinstance(value, bytes):
        return f"<bytes: {len(value)} bytes>"
    # For complex objects, return a string representation
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _safe_serialize_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    """Safely serialize field defaults for JSON output."""
    return {k: _safe_serialize_value(v) for k, v in defaults.items()}


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Status
    "TrinityStatus",
    "check_trinity_status",
    "initialize_trinity",
    # Registry
    "RegisteredType",
    "list_registered_types",
    "get_type_info",
    "get_detailed_type_info",
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
