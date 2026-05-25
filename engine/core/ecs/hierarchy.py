"""Parent-child entity relationships."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .entity import Entity

if TYPE_CHECKING:
    from .world import World

__all__ = [
    "Parent", "Children",
    "set_parent", "remove_parent", "get_children", "get_parent",
    "destroy_hierarchy",
]


class Parent:
    """Component storing the parent entity reference."""
    __slots__ = ("entity",)

    def __init__(self, entity: Entity) -> None:
        self.entity = entity


class Children:
    """Component storing a list of child entities."""
    __slots__ = ("entities",)

    def __init__(self, entities: list[Entity] | None = None) -> None:
        self.entities: list[Entity] = entities if entities is not None else []


def set_parent(world: World, child: Entity, parent: Entity) -> None:
    """Set *parent* as the parent of *child*."""
    # Remove from old parent if any
    old_parent_comp = world.get_component(child, Parent)
    if old_parent_comp is not None:
        remove_parent(world, child)

    # Set parent component on child
    world.add_component(child, Parent(parent))

    # Add child to parent's children list
    children_comp = world.get_component(parent, Children)
    if children_comp is None:
        world.add_component(parent, Children([child]))
    else:
        children_comp.entities.append(child)


def remove_parent(world: World, child: Entity) -> None:
    """Remove the parent relationship from *child*."""
    parent_comp = world.get_component(child, Parent)
    if parent_comp is None:
        return
    parent_entity = parent_comp.entity

    # Remove child from parent's children list
    children_comp = world.get_component(parent_entity, Children)
    if children_comp is not None:
        try:
            children_comp.entities.remove(child)
        except ValueError:
            pass

    world.remove_component(child, Parent)


def get_parent(world: World, entity: Entity) -> Entity | None:
    comp = world.get_component(entity, Parent)
    return comp.entity if comp is not None else None


def get_children(world: World, entity: Entity) -> list[Entity]:
    comp = world.get_component(entity, Children)
    return list(comp.entities) if comp is not None else []


def destroy_hierarchy(world: World, root: Entity) -> None:
    """Destroy *root* and all descendants using iterative traversal."""
    stack = [root]
    to_destroy: list[Entity] = []
    while stack:
        entity = stack.pop()
        to_destroy.append(entity)
        children_comp = world.get_component(entity, Children)
        if children_comp is not None:
            stack.extend(children_comp.entities)
    # Destroy in reverse order (leaves first)
    for entity in reversed(to_destroy):
        world.destroy(entity)
