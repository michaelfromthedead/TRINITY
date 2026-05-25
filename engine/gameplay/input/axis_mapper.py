"""Axis mapping for the gameplay input system.

This module provides axis mapping with positive/negative bindings,
supporting both digital (keyboard) and analog (gamepad) inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from time import time
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from .constants import (
    DEFAULT_AXIS_SENSITIVITY,
    DEFAULT_DEAD_ZONE,
    MAX_AXIS_SENSITIVITY,
    MIN_AXIS_SENSITIVITY,
)
from .processing import apply_dead_zone, apply_power_curve, InputSmoother, SmoothingType


# =============================================================================
# Axis Binding Types
# =============================================================================

class AxisBindingType(Enum):
    """Type of axis binding."""
    DIGITAL = auto()      # Uses positive/negative key bindings
    ANALOG = auto()       # Uses analog stick or trigger
    COMPOSITE = auto()    # Combines multiple inputs


@dataclass
class AxisBinding:
    """Binding for an axis input."""
    binding_type: AxisBindingType = AxisBindingType.DIGITAL
    positive_keys: List[str] = field(default_factory=list)
    negative_keys: List[str] = field(default_factory=list)
    analog_key: str = ""
    scale: float = 1.0
    dead_zone: float = DEFAULT_DEAD_ZONE
    invert: bool = False


# =============================================================================
# Axis Definition
# =============================================================================

AxisCallback = Callable[['AxisEvent'], None]


@dataclass
class AxisEvent:
    """Event data for an axis value change."""
    axis_name: str
    value: float
    raw_value: float
    delta: float
    timestamp: float


@dataclass
class AxisDefinition:
    """Defines a gameplay axis with bindings."""
    name: str
    bindings: List[AxisBinding] = field(default_factory=list)
    sensitivity: float = DEFAULT_AXIS_SENSITIVITY
    dead_zone: float = DEFAULT_DEAD_ZONE
    smoothing: float = 0.0
    snap_to_zero: bool = True  # Snap to zero when crossing
    clamp: bool = True  # Clamp to -1.0/1.0
    description: str = ""


# =============================================================================
# Axis State
# =============================================================================

@dataclass
class AxisState:
    """Current state of an axis."""
    value: float = 0.0
    raw_value: float = 0.0
    target_value: float = 0.0
    previous_value: float = 0.0


# =============================================================================
# Axis Mapper
# =============================================================================

class AxisMapper:
    """Maps raw input to gameplay axes."""
    __slots__ = (
        '_axes', '_states', '_callbacks', '_input_states',
        '_smoothers', '_enabled'
    )

    def __init__(self):
        """Initialize the axis mapper."""
        self._axes: Dict[str, AxisDefinition] = {}
        self._states: Dict[str, AxisState] = {}
        self._callbacks: Dict[str, List[AxisCallback]] = {}
        self._input_states: Dict[str, Tuple[bool, float]] = {}
        self._smoothers: Dict[str, InputSmoother] = {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if mapper is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the mapper."""
        self._enabled = value

    def register_axis(self, axis: AxisDefinition) -> bool:
        """Register an axis.

        Args:
            axis: Axis definition

        Returns:
            True if registered
        """
        if axis.name in self._axes:
            return False

        self._axes[axis.name] = axis
        self._states[axis.name] = AxisState()
        self._callbacks[axis.name] = []

        # Create smoother if needed
        if axis.smoothing > 0.0:
            self._smoothers[axis.name] = InputSmoother(
                SmoothingType.EXPONENTIAL,
                alpha=1.0 - axis.smoothing
            )

        return True

    def unregister_axis(self, axis_name: str) -> bool:
        """Unregister an axis.

        Args:
            axis_name: Name of axis to remove

        Returns:
            True if removed
        """
        if axis_name not in self._axes:
            return False

        del self._axes[axis_name]
        del self._states[axis_name]
        del self._callbacks[axis_name]
        self._smoothers.pop(axis_name, None)
        return True

    def get_axis(self, axis_name: str) -> Optional[AxisDefinition]:
        """Get an axis definition.

        Args:
            axis_name: Axis name

        Returns:
            Axis definition if found
        """
        return self._axes.get(axis_name)

    def add_binding(self, axis_name: str, binding: AxisBinding) -> bool:
        """Add a binding to an axis.

        Args:
            axis_name: Axis name
            binding: Binding to add

        Returns:
            True if added
        """
        axis = self._axes.get(axis_name)
        if axis is None:
            return False

        axis.bindings.append(binding)
        return True

    def remove_binding(self, axis_name: str, index: int) -> bool:
        """Remove a binding from an axis.

        Args:
            axis_name: Axis name
            index: Binding index to remove

        Returns:
            True if removed
        """
        axis = self._axes.get(axis_name)
        if axis is None or index < 0 or index >= len(axis.bindings):
            return False

        axis.bindings.pop(index)
        return True

    def bind_callback(
        self,
        axis_name: str,
        callback: AxisCallback
    ) -> bool:
        """Bind a callback to an axis.

        Args:
            axis_name: Axis name
            callback: Function to call on axis change

        Returns:
            True if bound
        """
        if axis_name not in self._axes:
            return False

        self._callbacks[axis_name].append(callback)
        return True

    def unbind_callback(
        self,
        axis_name: str,
        callback: AxisCallback
    ) -> bool:
        """Unbind a callback from an axis.

        Args:
            axis_name: Axis name
            callback: Function to remove

        Returns:
            True if removed
        """
        if axis_name not in self._callbacks:
            return False

        try:
            self._callbacks[axis_name].remove(callback)
            return True
        except ValueError:
            return False

    def set_input_state(
        self,
        input_key: str,
        is_active: bool,
        value: float = 1.0
    ) -> None:
        """Set the state of an input.

        Args:
            input_key: Input identifier
            is_active: Whether input is active
            value: Input value (0.0 to 1.0 or -1.0 to 1.0)
        """
        self._input_states[input_key] = (is_active, value)

    def clear_input_state(self, input_key: str) -> None:
        """Clear the state of an input.

        Args:
            input_key: Input to clear
        """
        self._input_states.pop(input_key, None)

    def get_axis_value(self, axis_name: str) -> float:
        """Get the current value of an axis.

        Args:
            axis_name: Axis to query

        Returns:
            Axis value (-1.0 to 1.0)
        """
        state = self._states.get(axis_name)
        return state.value if state else 0.0

    def get_raw_axis_value(self, axis_name: str) -> float:
        """Get the raw (unsmoothed) value of an axis.

        Args:
            axis_name: Axis to query

        Returns:
            Raw axis value
        """
        state = self._states.get(axis_name)
        return state.raw_value if state else 0.0

    def update(self, delta_time: float) -> List[AxisEvent]:
        """Update the axis mapper.

        Args:
            delta_time: Time since last update

        Returns:
            List of axis events for changed axes
        """
        if not self._enabled:
            return []

        events: List[AxisEvent] = []
        current_time = time()

        for axis_name, axis in self._axes.items():
            state = self._states[axis_name]
            state.previous_value = state.value

            # Calculate raw value from bindings
            raw_value = self._calculate_axis_value(axis)

            # Apply dead zone
            processed_value = apply_dead_zone(raw_value, axis.dead_zone)

            # Apply sensitivity
            processed_value *= axis.sensitivity

            # Apply smoothing
            if axis_name in self._smoothers:
                processed_value = self._smoothers[axis_name].update(processed_value)

            # Snap to zero when crossing
            if axis.snap_to_zero:
                if (state.previous_value > 0 and processed_value < 0) or \
                   (state.previous_value < 0 and processed_value > 0):
                    processed_value = 0.0

            # Clamp
            if axis.clamp:
                processed_value = max(-1.0, min(1.0, processed_value))

            # Update state
            state.raw_value = raw_value
            state.value = processed_value
            state.target_value = raw_value

            # Create event if value changed
            delta = state.value - state.previous_value
            if abs(delta) > 0.0001:
                event = AxisEvent(
                    axis_name=axis_name,
                    value=state.value,
                    raw_value=state.raw_value,
                    delta=delta,
                    timestamp=current_time
                )
                events.append(event)

                # Notify callbacks
                for callback in self._callbacks.get(axis_name, []):
                    try:
                        callback(event)
                    except Exception as e:
                        logger.warning(
                            "Exception in axis callback for '%s': %s",
                            axis_name, e
                        )

        return events

    def _calculate_axis_value(self, axis: AxisDefinition) -> float:
        """Calculate the raw axis value from bindings.

        Args:
            axis: Axis definition

        Returns:
            Combined axis value
        """
        total_value = 0.0

        for binding in axis.bindings:
            binding_value = 0.0

            if binding.binding_type == AxisBindingType.DIGITAL:
                # Calculate from positive/negative keys
                positive = 0.0
                negative = 0.0

                for key in binding.positive_keys:
                    is_active, value = self._input_states.get(key, (False, 0.0))
                    if is_active:
                        positive = max(positive, value)

                for key in binding.negative_keys:
                    is_active, value = self._input_states.get(key, (False, 0.0))
                    if is_active:
                        negative = max(negative, value)

                binding_value = positive - negative

            elif binding.binding_type == AxisBindingType.ANALOG:
                # Get analog value directly
                is_active, value = self._input_states.get(
                    binding.analog_key, (False, 0.0)
                )
                if is_active:
                    binding_value = value

                # Apply binding dead zone
                binding_value = apply_dead_zone(
                    binding_value, binding.dead_zone
                )

            elif binding.binding_type == AxisBindingType.COMPOSITE:
                # Combine digital and analog
                for key in binding.positive_keys:
                    is_active, value = self._input_states.get(key, (False, 0.0))
                    if is_active:
                        binding_value = max(binding_value, value)

                for key in binding.negative_keys:
                    is_active, value = self._input_states.get(key, (False, 0.0))
                    if is_active:
                        binding_value = min(binding_value, -value)

                if binding.analog_key:
                    is_active, value = self._input_states.get(
                        binding.analog_key, (False, 0.0)
                    )
                    if is_active and abs(value) > abs(binding_value):
                        binding_value = value

            # Apply scale and invert
            binding_value *= binding.scale
            if binding.invert:
                binding_value = -binding_value

            # Accumulate (take maximum magnitude)
            if abs(binding_value) > abs(total_value):
                total_value = binding_value

        return total_value

    def reset(self) -> None:
        """Reset all axis states."""
        for state in self._states.values():
            state.value = 0.0
            state.raw_value = 0.0
            state.target_value = 0.0
            state.previous_value = 0.0

        for smoother in self._smoothers.values():
            smoother.reset()

        self._input_states.clear()


# =============================================================================
# 2D Axis (Vector) Mapper
# =============================================================================

@dataclass
class Vector2Binding:
    """Binding for a 2D axis input."""
    x_axis: str = ""  # Name of X axis
    y_axis: str = ""  # Name of Y axis
    # Or direct bindings
    up_keys: List[str] = field(default_factory=list)
    down_keys: List[str] = field(default_factory=list)
    left_keys: List[str] = field(default_factory=list)
    right_keys: List[str] = field(default_factory=list)
    analog_x: str = ""
    analog_y: str = ""


@dataclass
class Vector2Definition:
    """Defines a 2D vector axis."""
    name: str
    bindings: List[Vector2Binding] = field(default_factory=list)
    normalize: bool = True  # Normalize diagonal movement
    dead_zone: float = DEFAULT_DEAD_ZONE
    sensitivity: float = DEFAULT_AXIS_SENSITIVITY


@dataclass
class Vector2Event:
    """Event for 2D axis change."""
    axis_name: str
    x: float
    y: float
    magnitude: float
    angle: float  # Radians
    timestamp: float


class Vector2Mapper:
    """Maps input to 2D vector axes."""
    __slots__ = (
        '_vectors', '_values', '_callbacks', '_axis_mapper', '_enabled'
    )

    def __init__(self, axis_mapper: Optional[AxisMapper] = None):
        """Initialize the vector mapper.

        Args:
            axis_mapper: Optional axis mapper to use for axis references
        """
        self._vectors: Dict[str, Vector2Definition] = {}
        self._values: Dict[str, Tuple[float, float]] = {}
        self._callbacks: Dict[str, List[Callable[[Vector2Event], None]]] = {}
        self._axis_mapper = axis_mapper
        self._enabled = True

    def register_vector(self, vector: Vector2Definition) -> bool:
        """Register a 2D vector axis.

        Args:
            vector: Vector definition

        Returns:
            True if registered
        """
        if vector.name in self._vectors:
            return False

        self._vectors[vector.name] = vector
        self._values[vector.name] = (0.0, 0.0)
        self._callbacks[vector.name] = []
        return True

    def unregister_vector(self, vector_name: str) -> bool:
        """Unregister a 2D vector axis.

        Args:
            vector_name: Name of vector to remove

        Returns:
            True if removed
        """
        if vector_name not in self._vectors:
            return False

        del self._vectors[vector_name]
        del self._values[vector_name]
        del self._callbacks[vector_name]
        return True

    def get_vector(self, vector_name: str) -> Tuple[float, float]:
        """Get the current value of a 2D axis.

        Args:
            vector_name: Vector name

        Returns:
            (x, y) values
        """
        return self._values.get(vector_name, (0.0, 0.0))

    def bind_callback(
        self,
        vector_name: str,
        callback: Callable[[Vector2Event], None]
    ) -> bool:
        """Bind a callback to a vector.

        Args:
            vector_name: Vector name
            callback: Function to call on change

        Returns:
            True if bound
        """
        if vector_name not in self._vectors:
            return False

        self._callbacks[vector_name].append(callback)
        return True

    def update(
        self,
        delta_time: float,
        input_states: Dict[str, Tuple[bool, float]]
    ) -> List[Vector2Event]:
        """Update the vector mapper.

        Args:
            delta_time: Time since last update
            input_states: Current input states

        Returns:
            List of vector events
        """
        import math

        if not self._enabled:
            return []

        events: List[Vector2Event] = []
        current_time = time()

        for name, vector in self._vectors.items():
            x = 0.0
            y = 0.0

            for binding in vector.bindings:
                bx = 0.0
                by = 0.0

                # Use referenced axes if available
                if binding.x_axis and self._axis_mapper:
                    bx = self._axis_mapper.get_axis_value(binding.x_axis)
                if binding.y_axis and self._axis_mapper:
                    by = self._axis_mapper.get_axis_value(binding.y_axis)

                # Or calculate from direct bindings
                if not binding.x_axis:
                    right = 0.0
                    left = 0.0
                    for key in binding.right_keys:
                        is_active, value = input_states.get(key, (False, 0.0))
                        if is_active:
                            right = max(right, value)
                    for key in binding.left_keys:
                        is_active, value = input_states.get(key, (False, 0.0))
                        if is_active:
                            left = max(left, value)
                    bx = right - left

                if not binding.y_axis:
                    up = 0.0
                    down = 0.0
                    for key in binding.up_keys:
                        is_active, value = input_states.get(key, (False, 0.0))
                        if is_active:
                            up = max(up, value)
                    for key in binding.down_keys:
                        is_active, value = input_states.get(key, (False, 0.0))
                        if is_active:
                            down = max(down, value)
                    by = up - down

                # Analog overrides
                if binding.analog_x:
                    is_active, value = input_states.get(binding.analog_x, (False, 0.0))
                    if is_active and abs(value) > abs(bx):
                        bx = value
                if binding.analog_y:
                    is_active, value = input_states.get(binding.analog_y, (False, 0.0))
                    if is_active and abs(value) > abs(by):
                        by = value

                # Take maximum magnitude
                if abs(bx) > abs(x):
                    x = bx
                if abs(by) > abs(y):
                    y = by

            # Apply dead zone (radial)
            magnitude = math.sqrt(x * x + y * y)
            if magnitude < vector.dead_zone:
                x = 0.0
                y = 0.0
                magnitude = 0.0
            elif magnitude > 0:
                # Rescale
                scale = (magnitude - vector.dead_zone) / (1.0 - vector.dead_zone)
                scale = min(1.0, scale)
                x = x / magnitude * scale
                y = y / magnitude * scale
                magnitude = scale

            # Normalize if needed
            if vector.normalize and magnitude > 1.0:
                x /= magnitude
                y /= magnitude
                magnitude = 1.0

            # Apply sensitivity
            x *= vector.sensitivity
            y *= vector.sensitivity

            # Clamp
            x = max(-1.0, min(1.0, x))
            y = max(-1.0, min(1.0, y))

            # Update value
            old_x, old_y = self._values[name]
            self._values[name] = (x, y)

            # Create event if changed
            if abs(x - old_x) > 0.0001 or abs(y - old_y) > 0.0001:
                magnitude = math.sqrt(x * x + y * y)
                angle = math.atan2(y, x) if magnitude > 0 else 0.0

                event = Vector2Event(
                    axis_name=name,
                    x=x,
                    y=y,
                    magnitude=magnitude,
                    angle=angle,
                    timestamp=current_time
                )
                events.append(event)

                for callback in self._callbacks.get(name, []):
                    try:
                        callback(event)
                    except Exception as e:
                        logger.warning(
                            "Exception in vector callback for '%s': %s",
                            name, e
                        )

        return events

    def reset(self) -> None:
        """Reset all vector values."""
        for name in self._values:
            self._values[name] = (0.0, 0.0)


# =============================================================================
# Decorators
# =============================================================================

def input_axis(
    name: str,
    positive: List[str],
    negative: List[str],
    sensitivity: float = DEFAULT_AXIS_SENSITIVITY
):
    """Decorator to mark a function as an input axis handler.

    Args:
        name: Axis name
        positive: Positive direction bindings
        negative: Negative direction bindings
        sensitivity: Axis sensitivity

    Returns:
        Decorator function
    """
    if not name:
        raise ValueError("'name' parameter is required")
    if not positive:
        raise ValueError("'positive' parameter is required")
    if not negative:
        raise ValueError("'negative' parameter is required")

    def decorator(func):
        # Store metadata on the function
        func._input_axis = True
        func._axis_name = name
        func._axis_positive = list(positive)
        func._axis_negative = list(negative)
        func._axis_sensitivity = sensitivity

        # For compatibility with trinity decorators
        if not hasattr(func, '_applied_decorators'):
            func._applied_decorators = set()
        func._applied_decorators.add('input_axis')

        if not hasattr(func, '_applied_steps'):
            func._applied_steps = []
        func._applied_steps.append(('input_axis', {'name': name}))

        if not hasattr(func, '_tags'):
            func._tags = {}
        func._tags['input_axis'] = True
        func._tags['axis_name'] = name
        func._tags['axis_positive'] = list(positive)
        func._tags['axis_negative'] = list(negative)

        if not hasattr(func, '_registries'):
            func._registries = set()
        func._registries.add('input')

        return func

    return decorator
