"""Deferred command buffer for batched world mutations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Type

from .component import component_id
from .entity import Entity

if TYPE_CHECKING:
    from .world import World

__all__ = [
    "Command", "SpawnCommand", "DespawnCommand",
    "InsertComponentCommand", "RemoveComponentCommand", "CommandBuffer",
]


class Command(ABC):
    """Base class for deferred world commands."""
    __slots__ = ()

    @abstractmethod
    def apply(self, world: World) -> None: ...


class SpawnCommand(Command):
    """Spawn an entity with the given components."""
    __slots__ = ("_components", "_result")

    def __init__(self, *components: Any) -> None:
        self._components = components
        self._result: Entity | None = None

    def apply(self, world: World) -> None:
        self._result = world.spawn(*self._components)

    @property
    def entity(self) -> Entity | None:
        return self._result


class DespawnCommand(Command):
    __slots__ = ("_entity",)

    def __init__(self, entity: Entity) -> None:
        self._entity = entity

    def apply(self, world: World) -> None:
        if self._entity is None:
            return
        world.destroy(self._entity)


class InsertComponentCommand(Command):
    __slots__ = ("_entity", "_component")

    def __init__(self, entity: Entity, component: Any) -> None:
        self._entity = entity
        self._component = component

    def apply(self, world: World) -> None:
        world.add_component(self._entity, self._component)


class RemoveComponentCommand(Command):
    __slots__ = ("_entity", "_component_type")

    def __init__(self, entity: Entity, component_type: Type) -> None:
        self._entity = entity
        self._component_type = component_type

    def apply(self, world: World) -> None:
        world.remove_component(self._entity, self._component_type)


class CommandBuffer:
    """Records commands and flushes them to a World."""
    __slots__ = ("_commands",)

    def __init__(self) -> None:
        self._commands: list[Command] = []

    def spawn(self, *components: Any) -> SpawnCommand:
        cmd = SpawnCommand(*components)
        self._commands.append(cmd)
        return cmd

    def despawn(self, entity: Entity) -> None:
        self._commands.append(DespawnCommand(entity))

    def insert(self, entity: Entity, component: Any) -> None:
        self._commands.append(InsertComponentCommand(entity, component))

    def remove(self, entity: Entity, component_type: Type) -> None:
        self._commands.append(RemoveComponentCommand(entity, component_type))

    def flush(self, world: World) -> None:
        commands = self._commands
        self._commands = []
        for cmd in commands:
            cmd.apply(world)

    def __len__(self) -> int:
        return len(self._commands)
