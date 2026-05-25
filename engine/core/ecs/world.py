"""Central World container for the ECS."""
from __future__ import annotations

from typing import Any, Callable, Iterator, Type

from .archetype import ArchetypeGraph
from .command_buffer import CommandBuffer
from .component import ComponentId, ComponentMask, component_id
from .entity import Entity, EntityAllocator
from .query import Query, QueryDescriptor, QueryResult

__all__ = ["World"]


class World:
    """Central ECS container: entities, components, archetypes, queries."""
    __slots__ = (
        "_allocator", "_graph", "_entity_archetype", "_command_buffer",
    )

    def __init__(self) -> None:
        self._allocator = EntityAllocator()
        self._graph = ArchetypeGraph()
        # entity -> current ComponentMask
        self._entity_archetype: dict[Entity, ComponentMask] = {}
        self._command_buffer = CommandBuffer()

    # -- Entity lifecycle --

    def spawn(self, *components: Any) -> Entity:
        """Create an entity with the given component instances."""
        entity = self._allocator.allocate()
        comp_map: dict[ComponentId, Any] = {}
        for comp in components:
            cid = component_id(type(comp))
            comp_map[cid] = comp
        mask = frozenset(comp_map.keys())
        arch = self._graph.get_or_create(mask)
        arch.add_entity(entity, comp_map)
        self._entity_archetype[entity] = mask
        return entity

    def spawn_bundle(self, bundle: Any) -> Entity:
        """Spawn from a bundle object — reads its __dict__ or __dataclass_fields__."""
        components: list[Any] = []
        if hasattr(bundle, "__dataclass_fields__"):
            for fname in bundle.__dataclass_fields__:
                components.append(getattr(bundle, fname))
        elif hasattr(bundle, "__dict__"):
            components.extend(bundle.__dict__.values())
        else:
            raise TypeError(f"Cannot extract components from {type(bundle)}")
        return self.spawn(*components)

    def destroy(self, entity: Entity) -> None:
        if not self.is_alive(entity):
            return
        mask = self._entity_archetype.pop(entity, None)
        if mask is not None:
            arch = self._graph.get_or_create(mask)
            arch.remove_entity(entity)
        self._allocator.deallocate(entity)

    def is_alive(self, entity: Entity) -> bool:
        return self._allocator.is_alive(entity)

    # -- Component operations --

    def add_component(self, entity: Entity, component: Any) -> None:
        if not self.is_alive(entity):
            return
        cid = component_id(type(component))
        old_mask = self._entity_archetype.get(entity, frozenset())
        if cid in old_mask:
            # overwrite
            arch = self._graph.get_or_create(old_mask)
            arch.set_component(entity, cid, component)
            return
        new_mask = self._graph.get_add_target(old_mask, cid)
        # migrate
        old_arch = self._graph.get_or_create(old_mask)
        data = old_arch.remove_entity(entity) or {}
        data[cid] = component
        new_arch = self._graph.get_or_create(new_mask)
        new_arch.add_entity(entity, data)
        self._entity_archetype[entity] = new_mask

    def remove_component(self, entity: Entity, component_type: Type) -> None:
        if not self.is_alive(entity):
            return
        cid = component_id(component_type)
        old_mask = self._entity_archetype.get(entity, frozenset())
        if cid not in old_mask:
            return
        new_mask = self._graph.get_remove_target(old_mask, cid)
        old_arch = self._graph.get_or_create(old_mask)
        data = old_arch.remove_entity(entity) or {}
        data.pop(cid, None)
        new_arch = self._graph.get_or_create(new_mask)
        new_arch.add_entity(entity, data)
        self._entity_archetype[entity] = new_mask

    def get_component(self, entity: Entity, component_type: Type) -> Any | None:
        if not self.is_alive(entity):
            return None
        cid = component_id(component_type)
        mask = self._entity_archetype.get(entity)
        if mask is None or cid not in mask:
            return None
        arch = self._graph.get_or_create(mask)
        return arch.get_component(entity, cid)

    def has_component(self, entity: Entity, component_type: Type) -> bool:
        cid = component_id(component_type)
        mask = self._entity_archetype.get(entity, frozenset())
        return cid in mask

    # -- Queries --

    def query(
        self,
        *component_types: Type,
        with_: tuple[Type, ...] = (),
        without: tuple[Type, ...] = (),
    ) -> QueryResult:
        required = tuple(component_id(t) for t in component_types)
        with_ids = frozenset(component_id(t) for t in with_)
        without_ids = frozenset(component_id(t) for t in without)
        desc = QueryDescriptor(required=required, with_=with_ids, without=without_ids)
        q = Query(desc, self._graph)
        return q.iter()

    def for_each(
        self,
        *component_types: Type,
        callback: Callable[..., Any],
    ) -> None:
        required = tuple(component_id(t) for t in component_types)
        desc = QueryDescriptor(required=required)
        q = Query(desc, self._graph)
        for row in q.iter():
            # row is (entity, *components)
            callback(*row[1:])

    # -- Command buffer --

    @property
    def command_buffer(self) -> CommandBuffer:
        return self._command_buffer

    def flush_commands(self) -> None:
        self._command_buffer.flush(self)
