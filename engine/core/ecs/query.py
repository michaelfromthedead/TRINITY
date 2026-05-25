"""Query system for matching archetypes and iterating entities."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Type

logger = logging.getLogger(__name__)

from .archetype import Archetype, ArchetypeGraph
from .component import ComponentId, ComponentMask, component_id
from .entity import Entity

__all__ = [
    "With", "Without", "Optional", "Changed",
    "QueryDescriptor", "Query", "QueryResult",
]


class With:
    """Filter marker: entity must have this component type."""
    __slots__ = ("type",)
    def __init__(self, component_type: Type) -> None:
        self.type = component_type


class Without:
    """Filter marker: entity must not have this component type."""
    __slots__ = ("type",)
    def __init__(self, component_type: Type) -> None:
        self.type = component_type


class Optional:
    """Filter marker: component returned if present, else None."""
    __slots__ = ("type",)
    def __init__(self, component_type: Type) -> None:
        self.type = component_type


class Changed:
    """Filter marker: only entities whose component was changed."""
    __slots__ = ("type",)
    def __init__(self, component_type: Type) -> None:
        self.type = component_type


@dataclass(frozen=True)
class QueryDescriptor:
    """Describes a query's required and filtered component types."""
    required: tuple[ComponentId, ...]
    with_: frozenset[ComponentId] = field(default_factory=frozenset)
    without: frozenset[ComponentId] = field(default_factory=frozenset)
    optional: tuple[ComponentId, ...] = ()
    changed: frozenset[ComponentId] = field(default_factory=frozenset)


class QueryResult:
    """Iterator yielding (entity, *components) tuples from matching archetypes."""
    __slots__ = ("_archetypes", "_required", "_optional")

    def __init__(
        self,
        archetypes: list[Archetype],
        required: tuple[ComponentId, ...],
        optional: tuple[ComponentId, ...] = (),
    ) -> None:
        self._archetypes = archetypes
        self._required = required
        self._optional = optional

    def __iter__(self) -> Iterator[tuple]:
        for arch in self._archetypes:
            entities = arch.entities
            req_cols = [arch.columns[cid] for cid in self._required]
            opt_cols = [
                arch.columns.get(cid) for cid in self._optional
            ]
            for i in range(len(entities)):
                row: list[Any] = [entities[i]]
                for col in req_cols:
                    row.append(col[i])
                for col in opt_cols:
                    row.append(col[i] if col is not None else None)
                yield tuple(row)


class Query:
    """Matches archetypes against a descriptor and iterates results."""

    def __init__(self, descriptor: QueryDescriptor, graph: ArchetypeGraph) -> None:
        self._descriptor = descriptor
        self._graph = graph

    def _matching_archetypes(self) -> list[Archetype]:
        d = self._descriptor
        if d.changed:
            logger.warning("Changed filter not yet implemented, ignored")
        required_set = frozenset(d.required) | d.with_
        result: list[Archetype] = []
        for arch in self._graph.archetypes():
            if not required_set.issubset(arch.mask):
                continue
            if d.without and d.without & arch.mask:
                continue
            result.append(arch)
        return result

    def iter(self) -> QueryResult:
        return QueryResult(
            self._matching_archetypes(),
            self._descriptor.required,
            self._descriptor.optional,
        )
