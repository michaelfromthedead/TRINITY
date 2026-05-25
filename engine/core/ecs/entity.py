"""Entity identification with generational indices."""
from __future__ import annotations

from engine.core.constants import ENTITY_INDEX_BITS, ENTITY_GENERATION_BITS, MAX_ENTITIES

__all__ = ["Entity", "EntityAllocator", "ENTITY_INDEX_BITS", "ENTITY_GENERATION_BITS", "MAX_ENTITIES"]
GENERATION_MASK = (1 << ENTITY_GENERATION_BITS) - 1
INDEX_MASK = (1 << ENTITY_INDEX_BITS) - 1
_NULL_INDEX = INDEX_MASK  # sentinel


class Entity:
    """Lightweight entity handle: index (uint24) + generation (uint16) packed into one int."""
    __slots__ = ("_packed",)

    def __init__(self, index: int, generation: int) -> None:
        self._packed = ((generation & GENERATION_MASK) << ENTITY_INDEX_BITS) | (index & INDEX_MASK)

    @classmethod
    def from_packed(cls, packed: int) -> Entity:
        e = object.__new__(cls)
        e._packed = packed
        return e

    @classmethod
    def null(cls) -> Entity:
        return cls(_NULL_INDEX, 0)

    @property
    def index(self) -> int:
        return self._packed & INDEX_MASK

    @property
    def generation(self) -> int:
        return (self._packed >> ENTITY_INDEX_BITS) & GENERATION_MASK

    def is_valid(self) -> bool:
        return self.index != _NULL_INDEX

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self._packed == other._packed

    def __hash__(self) -> int:
        return self._packed

    def __repr__(self) -> str:
        if not self.is_valid():
            return "Entity(null)"
        return f"Entity(index={self.index}, gen={self.generation})"


class EntityAllocator:
    """Creates and recycles entity IDs with generation bumping."""
    __slots__ = ("_generations", "_free_list", "_next_index")

    def __init__(self) -> None:
        self._generations: list[int] = []
        self._free_list: list[int] = []
        self._next_index: int = 0

    def allocate(self) -> Entity:
        if self._free_list:
            index = self._free_list.pop()
            gen = self._generations[index]
            return Entity(index, gen)
        if self._next_index >= MAX_ENTITIES:
            raise RuntimeError("Maximum entity count reached")
        index = self._next_index
        self._next_index += 1
        self._generations.append(0)
        return Entity(index, 0)

    def deallocate(self, entity: Entity) -> None:
        index = entity.index
        if index >= len(self._generations):
            return
        # bump generation (wraps at 65535)
        self._generations[index] = (self._generations[index] + 1) & GENERATION_MASK
        self._free_list.append(index)

    def is_alive(self, entity: Entity) -> bool:
        index = entity.index
        if not entity.is_valid() or index >= len(self._generations):
            return False
        return self._generations[index] == entity.generation
