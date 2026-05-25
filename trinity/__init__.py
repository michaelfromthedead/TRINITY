"""
Trinity Pattern - A three-layer Python metaprogramming architecture for game engines.

This package provides:
- Metaclasses: Class creation, registration, validation
- Descriptors: Attribute access interception
- Decorators: User-facing API, orchestration
- Constants: Centralized configuration values

Usage:
    from trinity import component, system, resource, event, asset
    from trinity.metaclasses import ComponentMeta, SystemMeta
    from trinity.descriptors import TrackedDescriptor, NetworkedDescriptor
    from trinity.constants import DEFAULT_POOL_SIZE, CACHE_LINE_BYTES
"""

from trinity.metaclasses import (
    EngineMeta,
    ComponentMeta,
    SystemMeta,
    ResourceMeta,
    EventMeta,
    AssetMeta,
    ProtocolMeta,
    StateMeta,
)

from trinity.base import (
    Component,
    System,
    Resource,
    Event,
    Asset,
    Protocol,
    State,
)

# Import constants module for easy access
from trinity import constants

__version__ = "0.1.0"

__all__ = [
    # Metaclasses
    "EngineMeta",
    "ComponentMeta",
    "SystemMeta",
    "ResourceMeta",
    "EventMeta",
    "AssetMeta",
    "ProtocolMeta",
    "StateMeta",
    # Base classes
    "Component",
    "System",
    "Resource",
    "Event",
    "Asset",
    "Protocol",
    "State",
    # Constants module
    "constants",
]
