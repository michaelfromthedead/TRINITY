"""
Controller & Possession System
==============================
Controller classes for possessing pawns:
- Controller: Base controller class
- PlayerController: Human player input
- AIController: AI-driven control

Uses the Trinity Pattern with:
- ControllerMeta metaclass for registration
- Possession descriptors for state tracking
- @controller decorator for controller definition
"""
from __future__ import annotations

import threading
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from trinity.decorators.ops import Op, Step, make_decorator, run_steps
from trinity.descriptors.base import BaseDescriptor
from trinity.metaclasses.engine_meta import EngineMeta

import time

from .constants import (
    DEFAULT_ACCEPTANCE_RADIUS,
    DEFAULT_AI_MOVE_SPEED,
    ENTITY_ID_INVALID,
    ENTITY_ID_START,
    POSSESSION_HISTORY_MAX_LENGTH,
    POSSESSION_TRANSITION_TIMEOUT_MS,
    UNPOSSESS_CLEANUP_DELAY_MS,
    ControllerType,
    LifecycleState,
)

if TYPE_CHECKING:
    from .actor import Pawn

T = TypeVar("T")


# =============================================================================
# CONTROLLER METACLASS
# =============================================================================


class ControllerMeta(EngineMeta):
    """
    Metaclass for Controller types.

    Responsibilities:
    - Assign unique controller type IDs
    - Register controller types
    - Validate controller definitions
    """

    _registry: ClassVar[Dict[int, Type["Controller"]]] = {}
    _name_to_id: ClassVar[Dict[str, int]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()

    _BASE_CLASS_NAMES: ClassVar[frozenset[str]] = frozenset({
        "Controller",
        "PlayerController",
        "AIController",
    })

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "ControllerMeta":
        """Create a new controller type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base classes
        if name in mcs._BASE_CLASS_NAMES:
            cls._controller_type_id = 0
            cls._controller_type_name = name
            return cls

        with mcs._lock:
            # Assign unique type ID
            cls._controller_type_id = mcs._next_id
            mcs._next_id += 1
            cls._controller_type_name = f"{cls.__module__}.{name}"

            # Record metaclass steps
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "controller_type_id", "value": cls._controller_type_id})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "controller_type_name", "value": cls._controller_type_name})
            )

            # Register
            mcs._registry[cls._controller_type_id] = cls
            mcs._name_to_id[cls._controller_type_name] = cls._controller_type_id
            cls._metaclass_steps.append(
                Step(Op.REGISTER, {"registry": "controller_registry", "id": cls._controller_type_id})
            )

        return cls

    @classmethod
    def get_by_id(mcs, controller_type_id: int) -> Optional[Type["Controller"]]:
        """Get controller class by type ID."""
        return mcs._registry.get(controller_type_id)

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[Type["Controller"]]:
        """Get controller class by qualified name."""
        controller_type_id = mcs._name_to_id.get(name)
        return mcs._registry.get(controller_type_id) if controller_type_id else None

    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the controller registry (for testing)."""
        with mcs._lock:
            mcs._registry.clear()
            mcs._name_to_id.clear()
            mcs._next_id = 1
        super().clear_registry()


# =============================================================================
# POSSESSION STATE
# =============================================================================


@dataclass
class PossessionState:
    """Current possession state for a controller."""

    pawn: Optional[weakref.ref["Pawn"]] = None
    is_possessing: bool = False
    possession_time: float = 0.0
    pending_pawn: Optional[weakref.ref["Pawn"]] = None


# =============================================================================
# POSSESSION DESCRIPTOR
# =============================================================================


class PossessionDescriptor(BaseDescriptor["Pawn"]):
    """
    Descriptor for tracking possessed pawn state.

    Features:
    - Validates possession transitions
    - Triggers possession callbacks
    - Tracks possession history
    """

    descriptor_id: str = "possession"
    accepts_inner: tuple[str, ...] = ("*",)
    accepts_outer: tuple[str, ...] = ("tracked", "observable")
    excludes: tuple[str, ...] = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor] = None,
        validate_transitions: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(field_type, inner, **config)
        self._validate_transitions = validate_transitions

    def pre_set(self, obj: Any, value: Optional["Pawn"]) -> Optional["Pawn"]:
        """Validate possession transition."""
        if value is None:
            return value

        # Check if pawn can be possessed
        if hasattr(value, "is_possessed") and value.is_possessed:
            if hasattr(value, "controller"):
                current_controller = value.controller
                if current_controller is not obj:
                    raise ValueError(
                        f"Pawn is already possessed by {current_controller}"
                    )
        return value

    def post_set(
        self,
        obj: Any,
        value: Optional["Pawn"],
        old_value: Optional["Pawn"],
    ) -> None:
        """Handle possession change callbacks."""
        # Notify old pawn of unpossession
        if old_value is not None:
            if hasattr(obj, "_on_unpossessed"):
                obj._on_unpossessed(old_value)

        # Notify new pawn of possession
        if value is not None:
            if hasattr(obj, "_on_possessed"):
                obj._on_possessed(value)

    @property
    def descriptor_steps(self) -> list[Step]:
        """Return steps this descriptor performs."""
        return [
            Step(Op.TRACK, {"field": self._name}),
            Step(Op.VALIDATE, {"constraint": "possession_valid"}),
            Step(Op.HOOK, {"event": "on_possession_changed"}),
        ]


# =============================================================================
# BASE CONTROLLER
# =============================================================================


class Controller(metaclass=ControllerMeta):
    """
    Base class for pawn controllers.

    Features:
    - Pawn possession/unpossession
    - Input processing
    - State management
    """

    _controller_type: ClassVar[ControllerType] = ControllerType.PLAYER
    _next_controller_id: ClassVar[int] = ENTITY_ID_START
    _id_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the controller."""
        with self._id_lock:
            self._controller_id = Controller._next_controller_id
            Controller._next_controller_id += 1

        self._possessed_pawn: Optional[weakref.ref["Pawn"]] = None
        self._is_active = True
        self._possession_history: deque[Tuple[float, Optional["Pawn"]]] = deque(
            maxlen=POSSESSION_HISTORY_MAX_LENGTH
        )

    # =========================================================================
    # IDENTIFICATION
    # =========================================================================

    @property
    def controller_id(self) -> int:
        """Get the unique controller ID."""
        return self._controller_id

    @property
    def controller_type(self) -> ControllerType:
        """Get the controller type."""
        return self._controller_type

    @property
    def is_active(self) -> bool:
        """Check if controller is active."""
        return self._is_active

    # =========================================================================
    # POSSESSION
    # =========================================================================

    @property
    def pawn(self) -> Optional["Pawn"]:
        """Get the currently possessed pawn."""
        return self._possessed_pawn() if self._possessed_pawn else None

    @property
    def is_possessing(self) -> bool:
        """Check if currently possessing a pawn."""
        return self._possessed_pawn is not None and self._possessed_pawn() is not None

    def possess(self, pawn: "Pawn") -> bool:
        """
        Attempt to possess a pawn.

        Args:
            pawn: The pawn to possess

        Returns:
            True if possession was successful
        """
        # Validate pawn
        if pawn is None:
            return False

        # Unpossess current pawn if any
        if self.is_possessing:
            self.unpossess()

        # Attempt possession on pawn side
        if hasattr(pawn, "possess"):
            if not pawn.possess(self):
                return False
        else:
            return False

        # Record possession with timestamp
        self._possessed_pawn = weakref.ref(pawn)
        self._possession_history.append((time.monotonic(), pawn))

        # Trigger callbacks
        self._on_possess(pawn)

        return True

    def unpossess(self) -> Optional["Pawn"]:
        """
        Release possession of current pawn.

        Returns:
            The previously possessed pawn, if any
        """
        pawn = self.pawn
        if pawn is None:
            return None

        # Release on pawn side
        if hasattr(pawn, "unpossess"):
            pawn.unpossess()

        # Clear possession with timestamp
        self._possessed_pawn = None
        self._possession_history.append((time.monotonic(), None))

        # Trigger callbacks
        self._on_unpossess(pawn)

        return pawn

    def _on_possess(self, pawn: "Pawn") -> None:
        """Called when possession succeeds."""
        pass

    def _on_unpossess(self, pawn: "Pawn") -> None:
        """Called when unpossession occurs."""
        pass

    def _on_possessed(self, pawn: "Pawn") -> None:
        """Called when a pawn is possessed (descriptor callback)."""
        pass

    def _on_unpossessed(self, pawn: "Pawn") -> None:
        """Called when a pawn is unpossessed (descriptor callback)."""
        pass

    # =========================================================================
    # INPUT
    # =========================================================================

    def setup_input(self) -> None:
        """
        Set up input bindings.

        Override this method to bind input actions to pawn methods.
        """
        pass

    def process_input(self, delta_time: float) -> None:
        """
        Process input for the current frame.

        Args:
            delta_time: Time since last frame
        """
        pass

    # =========================================================================
    # TICK
    # =========================================================================

    def tick(self, delta_time: float) -> None:
        """
        Update controller each frame.

        Args:
            delta_time: Time since last frame
        """
        if self._is_active:
            self.process_input(delta_time)

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def activate(self) -> None:
        """Activate this controller."""
        self._is_active = True

    def deactivate(self) -> None:
        """Deactivate this controller."""
        self._is_active = False

    def destroy(self) -> None:
        """Clean up and destroy this controller."""
        self.unpossess()
        self._is_active = False

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        pawn_info = f"pawn={self.pawn}" if self.pawn else "no pawn"
        return (
            f"<{self.__class__.__name__} "
            f"id={self._controller_id} "
            f"{pawn_info}>"
        )

    @classmethod
    def reset_controller_ids(cls) -> None:
        """Reset controller ID generation (for testing)."""
        with cls._id_lock:
            cls._next_controller_id = ENTITY_ID_START


# =============================================================================
# PLAYER CONTROLLER
# =============================================================================


class PlayerController(Controller):
    """
    Controller for human player input.

    Features:
    - Input action mapping
    - Camera control
    - Player-specific state
    """

    _controller_type: ClassVar[ControllerType] = ControllerType.PLAYER

    def __init__(
        self,
        player_index: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._player_index = player_index
        self._input_bindings: Dict[str, Callable] = {}
        self._camera_target: Optional[weakref.ref] = None
        self._show_mouse_cursor = True
        self._enable_click_events = True

    # =========================================================================
    # PLAYER PROPERTIES
    # =========================================================================

    @property
    def player_index(self) -> int:
        """Get the player index (for local multiplayer)."""
        return self._player_index

    @property
    def is_local_player(self) -> bool:
        """Check if this is the local player controller."""
        return True  # Override in networked implementation

    @property
    def show_mouse_cursor(self) -> bool:
        """Get mouse cursor visibility."""
        return self._show_mouse_cursor

    @show_mouse_cursor.setter
    def show_mouse_cursor(self, value: bool) -> None:
        """Set mouse cursor visibility."""
        self._show_mouse_cursor = value

    # =========================================================================
    # INPUT BINDING
    # =========================================================================

    def bind_action(
        self,
        action_name: str,
        callback: Callable,
    ) -> None:
        """
        Bind an input action to a callback.

        Args:
            action_name: Name of the input action
            callback: Function to call when action triggers
        """
        self._input_bindings[action_name] = callback

    def unbind_action(self, action_name: str) -> bool:
        """
        Unbind an input action.

        Args:
            action_name: Name of the input action

        Returns:
            True if action was unbound
        """
        return self._input_bindings.pop(action_name, None) is not None

    def get_bound_actions(self) -> List[str]:
        """Get list of bound action names."""
        return list(self._input_bindings.keys())

    def trigger_action(self, action_name: str, *args: Any, **kwargs: Any) -> bool:
        """
        Manually trigger an input action.

        Args:
            action_name: Name of the action to trigger
            *args, **kwargs: Arguments to pass to callback

        Returns:
            True if action was triggered
        """
        callback = self._input_bindings.get(action_name)
        if callback is not None:
            callback(*args, **kwargs)
            return True
        return False

    # =========================================================================
    # CAMERA
    # =========================================================================

    def set_camera_target(self, target: Any) -> None:
        """Set the camera target for this player."""
        self._camera_target = weakref.ref(target) if target else None

    def get_camera_target(self) -> Optional[Any]:
        """Get the current camera target."""
        return self._camera_target() if self._camera_target else None

    # =========================================================================
    # POSSESSION OVERRIDES
    # =========================================================================

    def _on_possess(self, pawn: "Pawn") -> None:
        """Set up player input when possessing a pawn."""
        super()._on_possess(pawn)

        # Set up input on the pawn
        if hasattr(pawn, "setup_player_input"):
            pawn.setup_player_input()

        # Set camera target to pawn
        self.set_camera_target(pawn)

    def _on_unpossess(self, pawn: "Pawn") -> None:
        """Clean up when unpossessing."""
        super()._on_unpossess(pawn)

        # Clear camera target if it was the pawn
        if self.get_camera_target() is pawn:
            self.set_camera_target(None)


# =============================================================================
# AI CONTROLLER
# =============================================================================


class AIController(Controller):
    """
    Controller for AI-driven pawns.

    Features:
    - Behavior tree execution
    - AI perception
    - Navigation requests
    """

    _controller_type: ClassVar[ControllerType] = ControllerType.AI

    def __init__(
        self,
        behavior_tree: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._behavior_tree = behavior_tree
        self._blackboard: Dict[str, Any] = {}
        self._focus_target: Optional[weakref.ref] = None
        self._ai_enabled = True

    # =========================================================================
    # AI PROPERTIES
    # =========================================================================

    @property
    def behavior_tree(self) -> Optional[Any]:
        """Get the behavior tree."""
        return self._behavior_tree

    @behavior_tree.setter
    def behavior_tree(self, value: Any) -> None:
        """Set the behavior tree."""
        self._behavior_tree = value

    @property
    def blackboard(self) -> Dict[str, Any]:
        """Get the AI blackboard."""
        return self._blackboard

    @property
    def ai_enabled(self) -> bool:
        """Check if AI is enabled."""
        return self._ai_enabled

    @ai_enabled.setter
    def ai_enabled(self, value: bool) -> None:
        """Enable or disable AI."""
        self._ai_enabled = value

    # =========================================================================
    # BLACKBOARD
    # =========================================================================

    def set_blackboard_value(self, key: str, value: Any) -> None:
        """Set a value in the blackboard."""
        self._blackboard[key] = value

    def get_blackboard_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the blackboard."""
        return self._blackboard.get(key, default)

    def clear_blackboard(self) -> None:
        """Clear all blackboard values."""
        self._blackboard.clear()

    # =========================================================================
    # FOCUS
    # =========================================================================

    def set_focus(self, target: Any) -> None:
        """Set the AI focus target."""
        self._focus_target = weakref.ref(target) if target else None

    def get_focus(self) -> Optional[Any]:
        """Get the current focus target."""
        return self._focus_target() if self._focus_target else None

    def clear_focus(self) -> None:
        """Clear the focus target."""
        self._focus_target = None

    # =========================================================================
    # MOVEMENT
    # =========================================================================

    def move_to_location(
        self,
        location: Tuple[float, float, float],
        acceptance_radius: float = DEFAULT_ACCEPTANCE_RADIUS,
    ) -> bool:
        """
        Request movement to a location.

        Args:
            location: Target location
            acceptance_radius: How close to get before stopping

        Returns:
            True if movement was started
        """
        pawn = self.pawn
        if pawn is None:
            return False

        # Store target in blackboard
        self._blackboard["move_target"] = location
        self._blackboard["acceptance_radius"] = acceptance_radius
        return True

    def move_to_actor(
        self,
        actor: Any,
        acceptance_radius: float = DEFAULT_ACCEPTANCE_RADIUS,
    ) -> bool:
        """
        Request movement to an actor.

        Args:
            actor: Target actor
            acceptance_radius: How close to get before stopping

        Returns:
            True if movement was started
        """
        if actor is None or not hasattr(actor, "position"):
            return False

        return self.move_to_location(actor.position, acceptance_radius)

    def stop_movement(self) -> None:
        """Stop any current movement."""
        self._blackboard.pop("move_target", None)
        self._blackboard.pop("acceptance_radius", None)

    # =========================================================================
    # TICK
    # =========================================================================

    def tick(self, delta_time: float) -> None:
        """Update AI each frame."""
        if not self._is_active or not self._ai_enabled:
            return

        # Execute behavior tree if present
        if self._behavior_tree is not None:
            if hasattr(self._behavior_tree, "tick"):
                self._behavior_tree.tick(delta_time, self)

        # Process movement
        self._process_movement(delta_time)

    def _process_movement(self, delta_time: float) -> None:
        """Process AI movement requests."""
        target = self._blackboard.get("move_target")
        if target is None:
            return

        pawn = self.pawn
        if pawn is None:
            return

        # Simple movement towards target (proper pathfinding would be separate)
        current_pos = pawn.position
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        dz = target[2] - current_pos[2]

        distance = (dx * dx + dy * dy + dz * dz) ** 0.5
        acceptance = self._blackboard.get("acceptance_radius", DEFAULT_ACCEPTANCE_RADIUS)

        if distance <= acceptance:
            # Reached target
            self.stop_movement()
            return

        # Normalize and apply movement
        if distance > 0:
            speed = DEFAULT_AI_MOVE_SPEED
            if hasattr(pawn, "max_walk_speed"):
                speed = pawn.max_walk_speed

            move_dist = min(speed * delta_time, distance)
            factor = move_dist / distance

            new_pos = (
                current_pos[0] + dx * factor,
                current_pos[1] + dy * factor,
                current_pos[2] + dz * factor,
            )
            pawn.position = new_pos


# =============================================================================
# POSSESSION MANAGER
# =============================================================================


class PossessionManager:
    """
    Manages controller-pawn possession relationships.

    Features:
    - Track all active possessions
    - Handle possession switching
    - Clean up on destruction
    """

    _instance: ClassVar[Optional["PossessionManager"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "PossessionManager":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._controllers: weakref.WeakValueDictionary[int, Controller] = weakref.WeakValueDictionary()
        self._possessions: Dict[int, int] = {}  # controller_id -> pawn_id
        self._initialized = True

    def register_controller(self, controller: Controller) -> None:
        """Register a controller with the manager."""
        self._controllers[controller.controller_id] = controller

    def unregister_controller(self, controller: Controller) -> None:
        """Unregister a controller from the manager."""
        self._controllers.pop(controller.controller_id, None)
        self._possessions.pop(controller.controller_id, None)

    def get_controller_for_pawn(self, pawn: "Pawn") -> Optional[Controller]:
        """Get the controller possessing a pawn."""
        pawn_id = getattr(pawn, "_entity_id", id(pawn))
        for controller_id, possessed_pawn_id in self._possessions.items():
            if possessed_pawn_id == pawn_id:
                return self._controllers.get(controller_id)
        return None

    def get_pawn_for_controller(self, controller: Controller) -> Optional["Pawn"]:
        """Get the pawn possessed by a controller."""
        return controller.pawn

    def switch_possession(
        self,
        controller: Controller,
        new_pawn: "Pawn",
    ) -> bool:
        """
        Switch a controller's possession to a new pawn.

        Args:
            controller: The controller
            new_pawn: The new pawn to possess

        Returns:
            True if switch was successful
        """
        return controller.possess(new_pawn)

    def get_all_controllers(self) -> List[Controller]:
        """Get all registered controllers."""
        return list(self._controllers.values())

    def get_player_controllers(self) -> List[PlayerController]:
        """Get all player controllers."""
        return [c for c in self._controllers.values() if isinstance(c, PlayerController)]

    def get_ai_controllers(self) -> List[AIController]:
        """Get all AI controllers."""
        return [c for c in self._controllers.values() if isinstance(c, AIController)]

    def clear(self) -> None:
        """Clear all tracked possessions (for testing)."""
        self._controllers.clear()
        self._possessions.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
            cls._instance = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Metaclass
    "ControllerMeta",
    # Data structures
    "PossessionState",
    # Descriptor
    "PossessionDescriptor",
    # Controllers
    "Controller",
    "PlayerController",
    "AIController",
    # Manager
    "PossessionManager",
]
