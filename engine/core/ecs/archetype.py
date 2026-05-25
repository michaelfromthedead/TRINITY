"""Archetype storage with SoA layout."""
from __future__ import annotations

import logging
from typing import Any

from .component import ComponentId, ComponentMask, component_id
from .entity import Entity

logger = logging.getLogger(__name__)

__all__ = ["Archetype", "ArchetypeGraph"]


class Archetype:
    """Stores entities sharing the same set of component types in SoA layout."""
    __slots__ = ("mask", "entities", "entity_to_row", "columns")

    def __init__(self, mask: ComponentMask) -> None:
        self.mask: ComponentMask = mask
        self.entities: list[Entity] = []
        self.entity_to_row: dict[Entity, int] = {}
        # SoA columns: ComponentId -> list of component instances
        self.columns: dict[ComponentId, list[Any]] = {cid: [] for cid in mask}

    def __len__(self) -> int:
        return len(self.entities)

    def add_entity(self, entity: Entity, components: dict[ComponentId, Any]) -> None:
        row = len(self.entities)
        self.entities.append(entity)
        self.entity_to_row[entity] = row
        for cid in self.mask:
            self.columns[cid].append(components.get(cid))

    def remove_entity(self, entity: Entity) -> dict[ComponentId, Any] | None:
        """Swap-remove entity for O(1). Returns removed component data."""
        row = self.entity_to_row.pop(entity, None)
        if row is None:
            return None

        last = len(self.entities) - 1
        removed: dict[ComponentId, Any] = {}

        if row != last:
            # swap with last
            last_entity = self.entities[last]
            self.entities[row] = last_entity
            self.entity_to_row[last_entity] = row
            for cid, col in self.columns.items():
                removed[cid] = col[row]
                col[row] = col[last]
                col.pop()
        else:
            for cid, col in self.columns.items():
                removed[cid] = col.pop()

        self.entities.pop()
        return removed

    def get_component(self, entity: Entity, cid: ComponentId) -> Any:
        row = self.entity_to_row.get(entity)
        if row is None or cid not in self.columns:
            logger.warning("get_component: entity %s not in archetype (mask=%s)", entity, self.mask)
            return None
        return self.columns[cid][row]

    def set_component(self, entity: Entity, cid: ComponentId, value: Any) -> None:
        row = self.entity_to_row.get(entity)
        if row is None or cid not in self.columns:
            logger.warning("set_component: entity %s not in archetype (mask=%s)", entity, self.mask)
            return
        self.columns[cid][row] = value

    def has_entity(self, entity: Entity) -> bool:
        return entity in self.entity_to_row


class ArchetypeGraph:
    """Manages archetypes and transition edges for component add/remove."""
    __slots__ = ("_archetypes", "_add_edges", "_remove_edges")

    def __init__(self) -> None:
        self._archetypes: dict[ComponentMask, Archetype] = {}
        # Edges: (source_mask, component_id) -> target_mask
        self._add_edges: dict[tuple[ComponentMask, ComponentId], ComponentMask] = {}
        self._remove_edges: dict[tuple[ComponentMask, ComponentId], ComponentMask] = {}

    def get_or_create(self, mask: ComponentMask) -> Archetype:
        arch = self._archetypes.get(mask)
        if arch is None:
            arch = Archetype(mask)
            self._archetypes[mask] = arch
        return arch

    def get_add_target(self, source: ComponentMask, cid: ComponentId) -> ComponentMask:
        key = (source, cid)
        target = self._add_edges.get(key)
        if target is None:
            target = source | frozenset({cid})
            self._add_edges[key] = target
        return target

    def get_remove_target(self, source: ComponentMask, cid: ComponentId) -> ComponentMask:
        key = (source, cid)
        target = self._remove_edges.get(key)
        if target is None:
            target = source - frozenset({cid})
            self._remove_edges[key] = target
        return target

    def archetypes(self) -> list[Archetype]:
        return list(self._archetypes.values())
