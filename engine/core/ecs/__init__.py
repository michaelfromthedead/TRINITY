"""ECS subsystem: World, Entity, Query, CommandBuffer, Hierarchy, EventBus."""

from .entity import Entity, EntityAllocator
from .component import ComponentId, component_id, ComponentMask, TagComponent
from .archetype import Archetype, ArchetypeGraph
from .query import With, Without, Optional, Changed, QueryDescriptor, Query, QueryResult
from .command_buffer import (
    Command, SpawnCommand, DespawnCommand,
    InsertComponentCommand, RemoveComponentCommand, CommandBuffer,
)
from .deterministic_buffer import DeterministicCommandBuffer
from .world import World
from .hierarchy import Parent, Children, set_parent, remove_parent, get_children, get_parent, destroy_hierarchy
from .event_bus import EventBus

__all__ = [
    "Entity", "EntityAllocator",
    "ComponentId", "component_id", "ComponentMask", "TagComponent",
    "Archetype", "ArchetypeGraph",
    "With", "Without", "Optional", "Changed", "QueryDescriptor", "Query", "QueryResult",
    "Command", "SpawnCommand", "DespawnCommand",
    "InsertComponentCommand", "RemoveComponentCommand", "CommandBuffer",
    "DeterministicCommandBuffer",
    "World",
    "Parent", "Children", "set_parent", "remove_parent",
    "get_children", "get_parent", "destroy_hierarchy",
    "EventBus",
]
