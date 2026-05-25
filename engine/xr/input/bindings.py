"""
XR Input action binding system.

Provides decorators and infrastructure for binding XR controller inputs
to game actions. Supports action-based input mapping similar to OpenXR
action system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry, DecoratorSpec


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================


class XRActionType(Enum):
    """Types of XR input actions."""
    BOOLEAN = auto()   # On/off (button press)
    FLOAT = auto()     # Analog value (trigger)
    VECTOR2 = auto()   # 2D axis (thumbstick)
    POSE = auto()      # 6-DOF pose
    HAPTIC = auto()    # Haptic output


class XRInputSource(Enum):
    """XR input source identifiers."""
    # Buttons
    LEFT_TRIGGER = "xr_left_trigger"
    RIGHT_TRIGGER = "xr_right_trigger"
    LEFT_GRIP = "xr_left_grip"
    RIGHT_GRIP = "xr_right_grip"
    LEFT_PRIMARY = "xr_left_primary"
    RIGHT_PRIMARY = "xr_right_primary"
    LEFT_SECONDARY = "xr_left_secondary"
    RIGHT_SECONDARY = "xr_right_secondary"
    LEFT_THUMBSTICK_CLICK = "xr_left_thumbstick_click"
    RIGHT_THUMBSTICK_CLICK = "xr_right_thumbstick_click"
    LEFT_MENU = "xr_left_menu"
    RIGHT_MENU = "xr_right_menu"

    # Axes
    LEFT_THUMBSTICK = "xr_left_thumbstick"
    RIGHT_THUMBSTICK = "xr_right_thumbstick"
    LEFT_THUMBSTICK_X = "xr_left_thumbstick_x"
    LEFT_THUMBSTICK_Y = "xr_left_thumbstick_y"
    RIGHT_THUMBSTICK_X = "xr_right_thumbstick_x"
    RIGHT_THUMBSTICK_Y = "xr_right_thumbstick_y"
    LEFT_THUMBSTICK_UP = "xr_left_thumbstick_up"
    LEFT_THUMBSTICK_DOWN = "xr_left_thumbstick_down"
    LEFT_THUMBSTICK_LEFT = "xr_left_thumbstick_left"
    LEFT_THUMBSTICK_RIGHT = "xr_left_thumbstick_right"
    RIGHT_THUMBSTICK_UP = "xr_right_thumbstick_up"
    RIGHT_THUMBSTICK_DOWN = "xr_right_thumbstick_down"
    RIGHT_THUMBSTICK_LEFT = "xr_right_thumbstick_left"
    RIGHT_THUMBSTICK_RIGHT = "xr_right_thumbstick_right"

    # Poses
    LEFT_GRIP_POSE = "xr_left_grip_pose"
    RIGHT_GRIP_POSE = "xr_right_grip_pose"
    LEFT_AIM_POSE = "xr_left_aim_pose"
    RIGHT_AIM_POSE = "xr_right_aim_pose"
    HEAD_POSE = "xr_head_pose"

    # Touch
    LEFT_TRIGGER_TOUCH = "xr_left_trigger_touch"
    RIGHT_TRIGGER_TOUCH = "xr_right_trigger_touch"
    LEFT_THUMBSTICK_TOUCH = "xr_left_thumbstick_touch"
    RIGHT_THUMBSTICK_TOUCH = "xr_right_thumbstick_touch"
    LEFT_THUMBREST_TOUCH = "xr_left_thumbrest_touch"
    RIGHT_THUMBREST_TOUCH = "xr_right_thumbrest_touch"

    # Haptics
    LEFT_HAPTIC = "xr_left_haptic"
    RIGHT_HAPTIC = "xr_right_haptic"


@dataclass
class XRActionBinding:
    """Binding between an action and input source(s)."""
    action_name: str
    action_type: XRActionType
    sources: List[str]
    threshold: float = 0.5  # For converting analog to boolean
    invert: bool = False  # Invert axis direction
    scale: float = 1.0  # Scale factor for analog values


@dataclass
class XRInputProfile:
    """Input profile with bindings for a specific controller type."""
    name: str
    vendor: str = ""
    controller_type: str = ""
    bindings: Dict[str, XRActionBinding] = field(default_factory=dict)

    def add_binding(self, binding: XRActionBinding) -> None:
        """Add an action binding to this profile."""
        self.bindings[binding.action_name] = binding

    def get_binding(self, action_name: str) -> Optional[XRActionBinding]:
        """Get binding for an action."""
        return self.bindings.get(action_name)


# =============================================================================
# ACTION REGISTRY
# =============================================================================


class XRActionRegistry:
    """
    Registry for XR input actions.

    Manages action definitions and their bindings to input sources.
    """

    def __init__(self) -> None:
        self._actions: Dict[str, XRActionBinding] = {}
        self._profiles: Dict[str, XRInputProfile] = {}
        self._active_profile: Optional[str] = None
        self._action_handlers: Dict[str, List[Callable]] = {}

    def register_action(
        self,
        name: str,
        action_type: XRActionType,
        default_bindings: List[str],
        threshold: float = 0.5,
    ) -> None:
        """
        Register an XR input action.

        Args:
            name: Action name (e.g., "grab", "teleport")
            action_type: Type of action
            default_bindings: Default input source bindings
            threshold: Threshold for analog-to-boolean conversion
        """
        self._actions[name] = XRActionBinding(
            action_name=name,
            action_type=action_type,
            sources=default_bindings,
            threshold=threshold,
        )

    def get_action(self, name: str) -> Optional[XRActionBinding]:
        """Get action by name."""
        return self._actions.get(name)

    def register_profile(self, profile: XRInputProfile) -> None:
        """Register an input profile."""
        self._profiles[profile.name] = profile

    def set_active_profile(self, profile_name: str) -> bool:
        """Set the active input profile."""
        if profile_name in self._profiles:
            self._active_profile = profile_name
            return True
        return False

    def get_binding_for_action(self, action_name: str) -> Optional[XRActionBinding]:
        """Get the binding for an action from active profile or defaults."""
        # Check active profile first
        if self._active_profile and self._active_profile in self._profiles:
            profile = self._profiles[self._active_profile]
            if action_name in profile.bindings:
                return profile.bindings[action_name]

        # Fall back to default binding
        return self._actions.get(action_name)

    def add_handler(self, action_name: str, handler: Callable) -> None:
        """Add a handler for an action."""
        if action_name not in self._action_handlers:
            self._action_handlers[action_name] = []
        self._action_handlers[action_name].append(handler)

    def remove_handler(self, action_name: str, handler: Callable) -> None:
        """Remove a handler for an action."""
        if action_name in self._action_handlers:
            try:
                self._action_handlers[action_name].remove(handler)
            except ValueError:
                pass

    def get_handlers(self, action_name: str) -> List[Callable]:
        """Get all handlers for an action."""
        return self._action_handlers.get(action_name, [])

    def list_actions(self) -> List[str]:
        """List all registered action names."""
        return list(self._actions.keys())

    def list_profiles(self) -> List[str]:
        """List all registered profile names."""
        return list(self._profiles.keys())


# Global action registry
_xr_action_registry = XRActionRegistry()


def get_xr_action_registry() -> XRActionRegistry:
    """Get the global XR action registry."""
    return _xr_action_registry


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_xr_action(
    name: str = "",
    action_type: Union[XRActionType, str] = XRActionType.BOOLEAN,
    bindings: Any = None,
    **_: Any,
) -> None:
    """Validate @xr_action parameters."""
    if not name:
        raise ValueError("@xr_action: 'name' parameter is required and must be non-empty")
    if bindings is None or len(bindings) == 0:
        raise ValueError("@xr_action: 'bindings' parameter is required and must be non-empty")


def _validate_xr_axis(
    name: str = "",
    positive: Any = None,
    negative: Any = None,
    **_: Any,
) -> None:
    """Validate @xr_axis parameters."""
    if not name:
        raise ValueError("@xr_axis: 'name' parameter is required and must be non-empty")
    if positive is None or len(positive) == 0:
        raise ValueError("@xr_axis: 'positive' parameter is required and must be non-empty")
    if negative is None or len(negative) == 0:
        raise ValueError("@xr_axis: 'negative' parameter is required and must be non-empty")


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _xr_action_steps(params: Dict[str, Any]) -> List[Step]:
    """Build steps for @xr_action decorator."""
    name = params.get("name", "")
    action_type = params.get("action_type", XRActionType.BOOLEAN)
    if isinstance(action_type, str):
        action_type = XRActionType[action_type.upper()]
    bindings = list(params.get("bindings", []))
    threshold = params.get("threshold", 0.5)

    return [
        Step(Op.TAG, {"key": "xr_action", "value": True}),
        Step(Op.TAG, {"key": "xr_action_name", "value": name}),
        Step(Op.TAG, {"key": "xr_action_type", "value": action_type.name}),
        Step(Op.TAG, {"key": "xr_action_bindings", "value": bindings}),
        Step(Op.TAG, {"key": "xr_action_threshold", "value": threshold}),
        Step(Op.REGISTER, {"registry": "xr_input"}),
    ]


def _xr_axis_steps(params: Dict[str, Any]) -> List[Step]:
    """Build steps for @xr_axis decorator."""
    name = params.get("name", "")
    positive = list(params.get("positive", []))
    negative = list(params.get("negative", []))
    deadzone = params.get("deadzone", 0.15)

    return [
        Step(Op.TAG, {"key": "xr_axis", "value": True}),
        Step(Op.TAG, {"key": "xr_axis_name", "value": name}),
        Step(Op.TAG, {"key": "xr_axis_positive", "value": positive}),
        Step(Op.TAG, {"key": "xr_axis_negative", "value": negative}),
        Step(Op.TAG, {"key": "xr_axis_deadzone", "value": deadzone}),
        Step(Op.REGISTER, {"registry": "xr_input"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_xr_action(target: Any, params: Dict[str, Any]) -> Any:
    """Post-processing for @xr_action decorator."""
    name = params.get("name", "")
    action_type = params.get("action_type", XRActionType.BOOLEAN)
    if isinstance(action_type, str):
        action_type = XRActionType[action_type.upper()]
    bindings = list(params.get("bindings", []))
    threshold = params.get("threshold", 0.5)

    # Set attributes on target
    target._xr_action = True
    target._xr_action_name = name
    target._xr_action_type = action_type
    target._xr_action_bindings = bindings
    target._xr_action_threshold = threshold

    # Register action in global registry
    _xr_action_registry.register_action(name, action_type, bindings, threshold)
    _xr_action_registry.add_handler(name, target)

    return None


def _after_xr_axis(target: Any, params: Dict[str, Any]) -> Any:
    """Post-processing for @xr_axis decorator."""
    name = params.get("name", "")
    positive = list(params.get("positive", []))
    negative = list(params.get("negative", []))
    deadzone = params.get("deadzone", 0.15)

    # Set attributes on target
    target._xr_axis = True
    target._xr_axis_name = name
    target._xr_axis_positive = positive
    target._xr_axis_negative = negative
    target._xr_axis_deadzone = deadzone

    # Register as FLOAT action
    _xr_action_registry.register_action(
        name,
        XRActionType.FLOAT,
        positive + negative,
        threshold=deadzone,
    )
    _xr_action_registry.add_handler(name, target)

    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


xr_action = make_decorator(
    name="xr_action",
    steps=_xr_action_steps,
    doc="Register an XR input action with controller bindings.",
    validate=_validate_xr_action,
    after_steps=_after_xr_action,
)


xr_axis = make_decorator(
    name="xr_axis",
    steps=_xr_axis_steps,
    doc="Register an XR input axis with positive and negative bindings.",
    validate=_validate_xr_axis,
    after_steps=_after_xr_axis,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================


_REGISTRY_ENTRIES: List[Tuple[str, Any, Tuple[str, ...]]] = [
    ("xr_action", xr_action, ("function",)),
    ("xr_axis", xr_axis, ("function",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.INPUT,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.INPUT].append(_spec)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def bind_action(
    action_name: str,
    handler: Callable,
) -> None:
    """
    Bind a handler function to an XR action.

    Args:
        action_name: Name of the action to bind to
        handler: Handler function to call when action is triggered
    """
    _xr_action_registry.add_handler(action_name, handler)


def unbind_action(
    action_name: str,
    handler: Callable,
) -> None:
    """
    Unbind a handler function from an XR action.

    Args:
        action_name: Name of the action
        handler: Handler function to remove
    """
    _xr_action_registry.remove_handler(action_name, handler)


def create_profile(
    name: str,
    vendor: str = "",
    controller_type: str = "",
) -> XRInputProfile:
    """
    Create a new input profile.

    Args:
        name: Profile name
        vendor: Controller vendor
        controller_type: Controller type identifier

    Returns:
        New input profile
    """
    profile = XRInputProfile(name=name, vendor=vendor, controller_type=controller_type)
    _xr_action_registry.register_profile(profile)
    return profile


def get_action_value(action_name: str, controller_state: Dict[str, Any]) -> Any:
    """
    Get the current value of an action from controller state.

    Args:
        action_name: Action to query
        controller_state: Current controller input state

    Returns:
        Action value (bool, float, or tuple depending on type)
    """
    binding = _xr_action_registry.get_binding_for_action(action_name)
    if binding is None:
        return None

    # Aggregate values from all bound sources
    if binding.action_type == XRActionType.BOOLEAN:
        for source in binding.sources:
            value = controller_state.get(source, 0.0)
            if isinstance(value, bool):
                if value:
                    return True
            elif isinstance(value, (int, float)):
                if value >= binding.threshold:
                    return True
        return False

    elif binding.action_type == XRActionType.FLOAT:
        max_value = 0.0
        for source in binding.sources:
            value = controller_state.get(source, 0.0)
            if isinstance(value, (int, float)):
                if binding.invert:
                    value = -value
                value *= binding.scale
                if abs(value) > abs(max_value):
                    max_value = value
        return max_value

    elif binding.action_type == XRActionType.VECTOR2:
        x = 0.0
        y = 0.0
        for source in binding.sources:
            value = controller_state.get(source)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                x = value[0] * binding.scale
                y = value[1] * binding.scale
                if binding.invert:
                    x, y = -x, -y
                break
        return (x, y)

    elif binding.action_type == XRActionType.POSE:
        for source in binding.sources:
            value = controller_state.get(source)
            if value is not None:
                return value
        return None

    return None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Decorators
    "xr_action",
    "xr_axis",
    # Enums
    "XRActionType",
    "XRInputSource",
    # Data classes
    "XRActionBinding",
    "XRInputProfile",
    # Registry
    "XRActionRegistry",
    "get_xr_action_registry",
    # Functions
    "bind_action",
    "unbind_action",
    "create_profile",
    "get_action_value",
]
