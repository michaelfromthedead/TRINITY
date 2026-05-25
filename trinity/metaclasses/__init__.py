"""
Trinity Pattern Metaclasses.

The foundational layer of the Trinity Pattern. Metaclasses control class creation
and registration, running once at class definition time.

Hierarchy:
    type (Python built-in)
    └── EngineMeta (base for all engine types)
        ├── ComponentMeta (ECS components)
        ├── SystemMeta (ECS systems)
        ├── ResourceMeta (global singletons)
        ├── EventMeta (event types)
        ├── AssetMeta (asset handles)
        ├── ProtocolMeta (network protocols)
        └── StateMeta (state machine states)
"""

from trinity.metaclasses.asset_meta import AssetMeta
from trinity.metaclasses.component_meta import ComponentMeta
from trinity.metaclasses.engine_meta import EngineMeta
from trinity.metaclasses.event_meta import EventMeta
from trinity.metaclasses.protocol_meta import ProtocolMeta
from trinity.metaclasses.resource_meta import ResourceMeta
from trinity.metaclasses.state_meta import StateMeta
from trinity.metaclasses.system_meta import SystemMeta

__all__ = [
    "EngineMeta",
    "ComponentMeta",
    "SystemMeta",
    "ResourceMeta",
    "EventMeta",
    "AssetMeta",
    "ProtocolMeta",
    "StateMeta",
]
