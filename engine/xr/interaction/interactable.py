"""Base XR interactable component for VR/AR interactions.

This module provides the foundation for all XR-interactable objects,
supporting hover, select, and grab interaction states.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, TypeVar, Generic
from functools import wraps

logger = logging.getLogger(__name__)

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform, RigidTransform


class InteractionState(Enum):
    """Possible states of an interactable object."""
    IDLE = auto()
    HOVERED = auto()
    SELECTED = auto()
    GRABBED = auto()


class InteractionType(Enum):
    """Types of interaction supported."""
    HOVER = auto()
    SELECT = auto()
    GRAB = auto()
    ACTIVATE = auto()


class InteractorType(Enum):
    """Types of interactors that can interact with objects."""
    RAY = auto()
    DIRECT = auto()
    GAZE = auto()
    POKE = auto()


@dataclass(slots=True)
class InteractionEvent:
    """Event data for interaction callbacks."""
    interactor_type: InteractorType
    interactor_id: int
    interaction_type: InteractionType
    position: Vec3
    rotation: Quat
    timestamp: float
    data: dict = field(default_factory=dict)


@dataclass(slots=True)
class InteractionHit:
    """Result of an interaction raycast or collision."""
    interactable: 'XRInteractable'
    hit_point: Vec3
    hit_normal: Vec3
    distance: float
    collider_id: Optional[int] = None


# Type variable for decorator typing
T = TypeVar('T', bound=type)


def xr_interactable(
    interaction_layers: list[str] | None = None,
    priority: int = 0,
    enabled: bool = True
) -> Callable[[T], T]:
    """Decorator to mark a class as XR interactable.

    Args:
        interaction_layers: List of interaction layer names this object belongs to
        priority: Priority when multiple interactables compete (higher = first)
        enabled: Whether interaction is enabled by default

    Returns:
        Decorated class with XR interactable metadata

    Example:
        @xr_interactable(interaction_layers=["default", "ui"], priority=1)
        class Button(XRInteractable):
            pass
    """
    def decorator(cls: T) -> T:
        # Store metadata on class
        cls._xr_interactable = True
        cls._interaction_layers = interaction_layers or ["default"]
        cls._interaction_priority = priority
        cls._interaction_enabled_default = enabled

        # Track applied decorators
        if not hasattr(cls, '_applied_decorators'):
            cls._applied_decorators = set()
        cls._applied_decorators.add('xr_interactable')

        # Store tags for registry (class-level)
        if not hasattr(cls, '_class_tags'):
            cls._class_tags = {}
        cls._class_tags['xr_interactable'] = True
        cls._class_tags['interaction_layers'] = cls._interaction_layers
        cls._class_tags['interaction_priority'] = priority

        return cls
    return decorator


class XRInteractable(ABC):
    """Base class for all XR interactable objects.

    Provides the foundation for objects that can be interacted with
    in XR environments through hover, select, and grab operations.

    Attributes:
        entity_id: The entity this component is attached to
        is_hovered: Whether the object is currently being hovered
        is_selected: Whether the object is currently selected
        is_grabbed: Whether the object is currently grabbed
        interaction_layers: Layers for filtering interactions
        priority: Priority for interaction selection
    """
    __slots__ = (
        '_entity_id', '_state', '_hover_interactors', '_select_interactors',
        '_grab_interactor', '_interaction_layers', '_priority', '_enabled',
        '_callbacks', '_hover_start_time', '_select_start_time', '_tags'
    )

    def __init__(
        self,
        entity_id: int = 0,
        interaction_layers: list[str] | None = None,
        priority: int = 0,
        enabled: bool = True
    ):
        """Initialize the interactable component.

        Args:
            entity_id: The entity this component is attached to
            interaction_layers: List of layer names for filtering
            priority: Selection priority (higher = selected first)
            enabled: Whether interactions are enabled
        """
        self._entity_id = entity_id
        self._state = InteractionState.IDLE
        self._hover_interactors: set[int] = set()
        self._select_interactors: set[int] = set()
        self._grab_interactor: Optional[int] = None
        self._interaction_layers = interaction_layers or ["default"]
        self._priority = priority
        self._enabled = enabled
        self._callbacks: dict[InteractionType, list[Callable[[InteractionEvent], None]]] = {
            InteractionType.HOVER: [],
            InteractionType.SELECT: [],
            InteractionType.GRAB: [],
            InteractionType.ACTIVATE: [],
        }
        self._hover_start_time: float = 0.0
        self._tags: dict = {}
        self._select_start_time: float = 0.0

    @property
    def entity_id(self) -> int:
        """Get the entity ID."""
        return self._entity_id

    @property
    def is_hovered(self) -> bool:
        """Check if object is being hovered by any interactor."""
        return len(self._hover_interactors) > 0

    @property
    def is_selected(self) -> bool:
        """Check if object is selected by any interactor."""
        return len(self._select_interactors) > 0

    @property
    def is_grabbed(self) -> bool:
        """Check if object is currently grabbed."""
        return self._grab_interactor is not None

    @property
    def state(self) -> InteractionState:
        """Get the current interaction state."""
        return self._state

    @property
    def interaction_layers(self) -> list[str]:
        """Get the interaction layers."""
        return self._interaction_layers.copy()

    @property
    def priority(self) -> int:
        """Get the interaction priority."""
        return self._priority

    @property
    def enabled(self) -> bool:
        """Check if interactions are enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable interactions."""
        self._enabled = value
        if not value:
            self._clear_all_interactors()

    def _update_state(self) -> None:
        """Update the interaction state based on current interactors."""
        if self._grab_interactor is not None:
            self._state = InteractionState.GRABBED
        elif self._select_interactors:
            self._state = InteractionState.SELECTED
        elif self._hover_interactors:
            self._state = InteractionState.HOVERED
        else:
            self._state = InteractionState.IDLE

    def _clear_all_interactors(self) -> None:
        """Clear all interactor references."""
        self._hover_interactors.clear()
        self._select_interactors.clear()
        self._grab_interactor = None
        self._state = InteractionState.IDLE

    def is_in_layer(self, layer: str) -> bool:
        """Check if this interactable is in a specific layer.

        Args:
            layer: The layer name to check

        Returns:
            True if the interactable is in the specified layer
        """
        return layer in self._interaction_layers

    def add_callback(
        self,
        interaction_type: InteractionType,
        callback: Callable[[InteractionEvent], None]
    ) -> None:
        """Register a callback for an interaction type.

        Args:
            interaction_type: The type of interaction to listen for
            callback: Function to call when interaction occurs
        """
        self._callbacks[interaction_type].append(callback)

    def remove_callback(
        self,
        interaction_type: InteractionType,
        callback: Callable[[InteractionEvent], None]
    ) -> None:
        """Remove a callback for an interaction type.

        Args:
            interaction_type: The type of interaction
            callback: The callback to remove
        """
        try:
            self._callbacks[interaction_type].remove(callback)
        except ValueError:
            pass

    def _emit_event(self, event: InteractionEvent) -> None:
        """Emit an interaction event to registered callbacks.

        Args:
            event: The interaction event to emit
        """
        for callback in self._callbacks[event.interaction_type]:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Interaction callback error for {event.interaction_type}: {e}")

    # Hover methods
    def on_hover_enter(self, interactor_id: int, event: InteractionEvent) -> None:
        """Called when an interactor starts hovering.

        Args:
            interactor_id: The ID of the hovering interactor
            event: The interaction event data
        """
        if not self._enabled:
            return

        was_hovered = self.is_hovered
        self._hover_interactors.add(interactor_id)

        if not was_hovered:
            self._hover_start_time = event.timestamp
            self._on_first_hover_enter(event)

        self._update_state()
        self._emit_event(event)

    def on_hover_exit(self, interactor_id: int, event: InteractionEvent) -> None:
        """Called when an interactor stops hovering.

        Args:
            interactor_id: The ID of the interactor that stopped hovering
            event: The interaction event data
        """
        self._hover_interactors.discard(interactor_id)

        if not self.is_hovered:
            self._on_last_hover_exit(event)

        self._update_state()
        self._emit_event(event)

    # Select methods
    def on_select_enter(self, interactor_id: int, event: InteractionEvent) -> None:
        """Called when an interactor selects this object.

        Args:
            interactor_id: The ID of the selecting interactor
            event: The interaction event data
        """
        if not self._enabled:
            return

        was_selected = self.is_selected
        self._select_interactors.add(interactor_id)

        if not was_selected:
            self._select_start_time = event.timestamp
            self._on_first_select_enter(event)

        self._update_state()
        self._emit_event(event)

    def on_select_exit(self, interactor_id: int, event: InteractionEvent) -> None:
        """Called when an interactor deselects this object.

        Args:
            interactor_id: The ID of the interactor that deselected
            event: The interaction event data
        """
        self._select_interactors.discard(interactor_id)

        if not self.is_selected:
            self._on_last_select_exit(event)

        self._update_state()
        self._emit_event(event)

    # Grab methods
    def on_grab_enter(self, interactor_id: int, event: InteractionEvent) -> bool:
        """Called when an interactor attempts to grab this object.

        Args:
            interactor_id: The ID of the grabbing interactor
            event: The interaction event data

        Returns:
            True if grab was successful, False otherwise
        """
        if not self._enabled or self._grab_interactor is not None:
            return False

        self._grab_interactor = interactor_id
        self._update_state()
        self._on_grab_started(event)
        self._emit_event(event)
        return True

    def on_grab_exit(self, interactor_id: int, event: InteractionEvent) -> None:
        """Called when an interactor releases this object.

        Args:
            interactor_id: The ID of the interactor that released
            event: The interaction event data
        """
        if self._grab_interactor != interactor_id:
            return

        self._grab_interactor = None
        self._on_grab_ended(event)
        self._update_state()
        self._emit_event(event)

    def on_activate(self, event: InteractionEvent) -> None:
        """Called when the object is activated (e.g., button press while selected).

        Args:
            event: The interaction event data
        """
        if not self._enabled:
            return

        self._on_activated(event)
        self._emit_event(event)

    # Override points for subclasses
    def _on_first_hover_enter(self, event: InteractionEvent) -> None:
        """Called when first interactor starts hovering. Override in subclass."""
        pass

    def _on_last_hover_exit(self, event: InteractionEvent) -> None:
        """Called when last interactor stops hovering. Override in subclass."""
        pass

    def _on_first_select_enter(self, event: InteractionEvent) -> None:
        """Called when first interactor selects. Override in subclass."""
        pass

    def _on_last_select_exit(self, event: InteractionEvent) -> None:
        """Called when last interactor deselects. Override in subclass."""
        pass

    def _on_grab_started(self, event: InteractionEvent) -> None:
        """Called when grab starts. Override in subclass."""
        pass

    def _on_grab_ended(self, event: InteractionEvent) -> None:
        """Called when grab ends. Override in subclass."""
        pass

    def _on_activated(self, event: InteractionEvent) -> None:
        """Called on activation. Override in subclass."""
        pass

    def get_hover_duration(self, current_time: float) -> float:
        """Get how long the object has been hovered.

        Args:
            current_time: The current timestamp

        Returns:
            Duration in seconds, or 0 if not hovered
        """
        if not self.is_hovered:
            return 0.0
        return current_time - self._hover_start_time

    def get_select_duration(self, current_time: float) -> float:
        """Get how long the object has been selected.

        Args:
            current_time: The current timestamp

        Returns:
            Duration in seconds, or 0 if not selected
        """
        if not self.is_selected:
            return 0.0
        return current_time - self._select_start_time

    def get_grabbing_interactor(self) -> Optional[int]:
        """Get the ID of the interactor currently grabbing this object.

        Returns:
            Interactor ID if grabbed, None otherwise
        """
        return self._grab_interactor

    def get_hovering_interactors(self) -> set[int]:
        """Get IDs of all interactors currently hovering.

        Returns:
            Set of interactor IDs
        """
        return self._hover_interactors.copy()

    def get_selecting_interactors(self) -> set[int]:
        """Get IDs of all interactors currently selecting.

        Returns:
            Set of interactor IDs
        """
        return self._select_interactors.copy()


class InteractableManager:
    """Manages all interactable objects in the scene.

    Provides efficient lookup by layer and handles interaction
    prioritization when multiple objects compete.
    """
    __slots__ = ('_interactables', '_by_layer', '_next_id')

    def __init__(self):
        """Initialize the interactable manager."""
        self._interactables: dict[int, XRInteractable] = {}
        self._by_layer: dict[str, set[int]] = {}
        self._next_id = 0

    def register(self, interactable: XRInteractable) -> int:
        """Register an interactable and return its assigned ID.

        Args:
            interactable: The interactable to register

        Returns:
            The assigned interactable ID
        """
        interactable_id = self._next_id
        self._next_id += 1

        self._interactables[interactable_id] = interactable

        for layer in interactable.interaction_layers:
            if layer not in self._by_layer:
                self._by_layer[layer] = set()
            self._by_layer[layer].add(interactable_id)

        return interactable_id

    def unregister(self, interactable_id: int) -> None:
        """Unregister an interactable.

        Args:
            interactable_id: The ID of the interactable to remove
        """
        interactable = self._interactables.pop(interactable_id, None)
        if interactable:
            for layer in interactable.interaction_layers:
                if layer in self._by_layer:
                    self._by_layer[layer].discard(interactable_id)

    def get(self, interactable_id: int) -> Optional[XRInteractable]:
        """Get an interactable by ID.

        Args:
            interactable_id: The interactable ID

        Returns:
            The interactable if found, None otherwise
        """
        return self._interactables.get(interactable_id)

    def get_by_layer(self, layer: str) -> list[XRInteractable]:
        """Get all interactables in a layer.

        Args:
            layer: The layer name

        Returns:
            List of interactables in the layer
        """
        ids = self._by_layer.get(layer, set())
        return [self._interactables[i] for i in ids if i in self._interactables]

    def get_sorted_by_priority(
        self,
        hits: list[InteractionHit],
        layer_mask: list[str] | None = None
    ) -> list[InteractionHit]:
        """Sort interaction hits by priority, filtering by layer.

        Args:
            hits: List of interaction hits
            layer_mask: Optional list of layers to filter by

        Returns:
            Hits sorted by priority (highest first), filtered by layer
        """
        filtered = []
        for hit in hits:
            if not hit.interactable.enabled:
                continue
            if layer_mask:
                if not any(hit.interactable.is_in_layer(l) for l in layer_mask):
                    continue
            filtered.append(hit)

        return sorted(filtered, key=lambda h: h.interactable.priority, reverse=True)
