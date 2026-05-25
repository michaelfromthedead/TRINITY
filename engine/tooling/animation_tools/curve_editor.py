"""Animation curve editor with tangent editing and easing functions.

Provides curve types including Linear, Bezier, Hermite, and Stepped,
with comprehensive tangent control and easing function support.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# ENUMS
# =============================================================================


class CurveType(Enum):
    """Types of animation curves."""

    LINEAR = auto()
    BEZIER = auto()
    HERMITE = auto()
    STEPPED = auto()
    CONSTANT = auto()


class TangentMode(Enum):
    """Tangent modes for curve keys."""

    AUTO = auto()
    FREE = auto()
    LINEAR = auto()
    FLAT = auto()
    WEIGHTED = auto()
    BREAK = auto()  # Separate in/out tangents


class EasingType(Enum):
    """Easing function types."""

    LINEAR = auto()
    EASE_IN = auto()
    EASE_OUT = auto()
    EASE_IN_OUT = auto()
    SINE_IN = auto()
    SINE_OUT = auto()
    SINE_IN_OUT = auto()
    QUAD_IN = auto()
    QUAD_OUT = auto()
    QUAD_IN_OUT = auto()
    CUBIC_IN = auto()
    CUBIC_OUT = auto()
    CUBIC_IN_OUT = auto()
    QUART_IN = auto()
    QUART_OUT = auto()
    QUART_IN_OUT = auto()
    QUINT_IN = auto()
    QUINT_OUT = auto()
    QUINT_IN_OUT = auto()
    EXPO_IN = auto()
    EXPO_OUT = auto()
    EXPO_IN_OUT = auto()
    CIRC_IN = auto()
    CIRC_OUT = auto()
    CIRC_IN_OUT = auto()
    ELASTIC_IN = auto()
    ELASTIC_OUT = auto()
    ELASTIC_IN_OUT = auto()
    BACK_IN = auto()
    BACK_OUT = auto()
    BACK_IN_OUT = auto()
    BOUNCE_IN = auto()
    BOUNCE_OUT = auto()
    BOUNCE_IN_OUT = auto()


# =============================================================================
# TANGENT HANDLE
# =============================================================================


@dataclass
class TangentHandle:
    """Handle for tangent control.

    Attributes:
        x: Horizontal offset (time direction)
        y: Vertical offset (value direction)
        weight: Weight for weighted tangents
    """

    x: float = 0.0
    y: float = 0.0
    weight: float = 1.0

    @property
    def slope(self) -> float:
        """Get tangent slope."""
        if abs(self.x) < 1e-6:
            return 0.0
        return self.y / self.x

    @property
    def length(self) -> float:
        """Get handle length."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> TangentHandle:
        """Return normalized handle."""
        length = self.length
        if length < 1e-6:
            return TangentHandle(1.0, 0.0, self.weight)
        return TangentHandle(self.x / length, self.y / length, self.weight)

    def scale(self, factor: float) -> TangentHandle:
        """Return scaled handle."""
        return TangentHandle(self.x * factor, self.y * factor, self.weight)

    def copy(self) -> TangentHandle:
        """Create a copy."""
        return TangentHandle(self.x, self.y, self.weight)


# =============================================================================
# CURVE KEY
# =============================================================================


@dataclass
class CurveKey:
    """A key (control point) on an animation curve.

    Attributes:
        time: Time position
        value: Value at this key
        tangent_mode: How tangents are computed
        tangent_in: Incoming tangent handle
        tangent_out: Outgoing tangent handle
        interpolation: Interpolation to next key
    """

    time: float
    value: float
    tangent_mode: TangentMode = TangentMode.AUTO
    tangent_in: TangentHandle = field(default_factory=TangentHandle)
    tangent_out: TangentHandle = field(default_factory=TangentHandle)
    interpolation: CurveType = CurveType.LINEAR

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"Key time must be >= 0, got {self.time}")

    def set_flat_tangents(self) -> None:
        """Set tangents to flat (horizontal)."""
        self.tangent_in = TangentHandle(-0.33, 0.0)
        self.tangent_out = TangentHandle(0.33, 0.0)
        self.tangent_mode = TangentMode.FLAT

    def set_linear_tangents(self, prev_key: Optional[CurveKey], next_key: Optional[CurveKey]) -> None:
        """Set tangents to point at adjacent keys."""
        if prev_key:
            dt = self.time - prev_key.time
            dv = self.value - prev_key.value
            self.tangent_in = TangentHandle(-dt * 0.33, -dv * 0.33)

        if next_key:
            dt = next_key.time - self.time
            dv = next_key.value - self.value
            self.tangent_out = TangentHandle(dt * 0.33, dv * 0.33)

        self.tangent_mode = TangentMode.LINEAR

    def copy(self) -> CurveKey:
        """Create a copy of this key."""
        return CurveKey(
            time=self.time,
            value=self.value,
            tangent_mode=self.tangent_mode,
            tangent_in=self.tangent_in.copy(),
            tangent_out=self.tangent_out.copy(),
            interpolation=self.interpolation,
        )


# =============================================================================
# EASING FUNCTIONS
# =============================================================================


class EasingFunction:
    """Collection of easing functions."""

    @staticmethod
    def linear(t: float) -> float:
        """Linear interpolation."""
        return t

    @staticmethod
    def ease_in(t: float) -> float:
        """Quadratic ease-in."""
        return t * t

    @staticmethod
    def ease_out(t: float) -> float:
        """Quadratic ease-out."""
        return t * (2 - t)

    @staticmethod
    def ease_in_out(t: float) -> float:
        """Quadratic ease-in-out."""
        return t * t * (3 - 2 * t)

    @staticmethod
    def sine_in(t: float) -> float:
        """Sine ease-in."""
        return 1 - math.cos((t * math.pi) / 2)

    @staticmethod
    def sine_out(t: float) -> float:
        """Sine ease-out."""
        return math.sin((t * math.pi) / 2)

    @staticmethod
    def sine_in_out(t: float) -> float:
        """Sine ease-in-out."""
        return -(math.cos(math.pi * t) - 1) / 2

    @staticmethod
    def quad_in(t: float) -> float:
        """Quadratic ease-in."""
        return t * t

    @staticmethod
    def quad_out(t: float) -> float:
        """Quadratic ease-out."""
        return 1 - (1 - t) * (1 - t)

    @staticmethod
    def quad_in_out(t: float) -> float:
        """Quadratic ease-in-out."""
        if t < 0.5:
            return 2 * t * t
        return 1 - pow(-2 * t + 2, 2) / 2

    @staticmethod
    def cubic_in(t: float) -> float:
        """Cubic ease-in."""
        return t * t * t

    @staticmethod
    def cubic_out(t: float) -> float:
        """Cubic ease-out."""
        return 1 - pow(1 - t, 3)

    @staticmethod
    def cubic_in_out(t: float) -> float:
        """Cubic ease-in-out."""
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2

    @staticmethod
    def quart_in(t: float) -> float:
        """Quartic ease-in."""
        return t * t * t * t

    @staticmethod
    def quart_out(t: float) -> float:
        """Quartic ease-out."""
        return 1 - pow(1 - t, 4)

    @staticmethod
    def quart_in_out(t: float) -> float:
        """Quartic ease-in-out."""
        if t < 0.5:
            return 8 * t * t * t * t
        return 1 - pow(-2 * t + 2, 4) / 2

    @staticmethod
    def quint_in(t: float) -> float:
        """Quintic ease-in."""
        return t * t * t * t * t

    @staticmethod
    def quint_out(t: float) -> float:
        """Quintic ease-out."""
        return 1 - pow(1 - t, 5)

    @staticmethod
    def quint_in_out(t: float) -> float:
        """Quintic ease-in-out."""
        if t < 0.5:
            return 16 * t * t * t * t * t
        return 1 - pow(-2 * t + 2, 5) / 2

    @staticmethod
    def expo_in(t: float) -> float:
        """Exponential ease-in."""
        return 0 if t == 0 else pow(2, 10 * t - 10)

    @staticmethod
    def expo_out(t: float) -> float:
        """Exponential ease-out."""
        return 1 if t == 1 else 1 - pow(2, -10 * t)

    @staticmethod
    def expo_in_out(t: float) -> float:
        """Exponential ease-in-out."""
        if t == 0:
            return 0
        if t == 1:
            return 1
        if t < 0.5:
            return pow(2, 20 * t - 10) / 2
        return (2 - pow(2, -20 * t + 10)) / 2

    @staticmethod
    def circ_in(t: float) -> float:
        """Circular ease-in."""
        return 1 - math.sqrt(1 - pow(t, 2))

    @staticmethod
    def circ_out(t: float) -> float:
        """Circular ease-out."""
        return math.sqrt(1 - pow(t - 1, 2))

    @staticmethod
    def circ_in_out(t: float) -> float:
        """Circular ease-in-out."""
        if t < 0.5:
            return (1 - math.sqrt(1 - pow(2 * t, 2))) / 2
        return (math.sqrt(1 - pow(-2 * t + 2, 2)) + 1) / 2

    @staticmethod
    def elastic_in(t: float) -> float:
        """Elastic ease-in."""
        c4 = (2 * math.pi) / 3
        if t == 0:
            return 0
        if t == 1:
            return 1
        return -pow(2, 10 * t - 10) * math.sin((t * 10 - 10.75) * c4)

    @staticmethod
    def elastic_out(t: float) -> float:
        """Elastic ease-out."""
        c4 = (2 * math.pi) / 3
        if t == 0:
            return 0
        if t == 1:
            return 1
        return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1

    @staticmethod
    def elastic_in_out(t: float) -> float:
        """Elastic ease-in-out."""
        c5 = (2 * math.pi) / 4.5
        if t == 0:
            return 0
        if t == 1:
            return 1
        if t < 0.5:
            return -(pow(2, 20 * t - 10) * math.sin((20 * t - 11.125) * c5)) / 2
        return (pow(2, -20 * t + 10) * math.sin((20 * t - 11.125) * c5)) / 2 + 1

    @staticmethod
    def back_in(t: float) -> float:
        """Back ease-in (slight overshoot)."""
        c1 = 1.70158
        c3 = c1 + 1
        return c3 * t * t * t - c1 * t * t

    @staticmethod
    def back_out(t: float) -> float:
        """Back ease-out."""
        c1 = 1.70158
        c3 = c1 + 1
        return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)

    @staticmethod
    def back_in_out(t: float) -> float:
        """Back ease-in-out."""
        c1 = 1.70158
        c2 = c1 * 1.525
        if t < 0.5:
            return (pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
        return (pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2

    @staticmethod
    def bounce_out(t: float) -> float:
        """Bounce ease-out."""
        n1 = 7.5625
        d1 = 2.75

        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375

    @staticmethod
    def bounce_in(t: float) -> float:
        """Bounce ease-in."""
        return 1 - EasingFunction.bounce_out(1 - t)

    @staticmethod
    def bounce_in_out(t: float) -> float:
        """Bounce ease-in-out."""
        if t < 0.5:
            return (1 - EasingFunction.bounce_out(1 - 2 * t)) / 2
        return (1 + EasingFunction.bounce_out(2 * t - 1)) / 2

    @staticmethod
    def get_function(easing_type: EasingType) -> Callable[[float], float]:
        """Get easing function by type."""
        mapping = {
            EasingType.LINEAR: EasingFunction.linear,
            EasingType.EASE_IN: EasingFunction.ease_in,
            EasingType.EASE_OUT: EasingFunction.ease_out,
            EasingType.EASE_IN_OUT: EasingFunction.ease_in_out,
            EasingType.SINE_IN: EasingFunction.sine_in,
            EasingType.SINE_OUT: EasingFunction.sine_out,
            EasingType.SINE_IN_OUT: EasingFunction.sine_in_out,
            EasingType.QUAD_IN: EasingFunction.quad_in,
            EasingType.QUAD_OUT: EasingFunction.quad_out,
            EasingType.QUAD_IN_OUT: EasingFunction.quad_in_out,
            EasingType.CUBIC_IN: EasingFunction.cubic_in,
            EasingType.CUBIC_OUT: EasingFunction.cubic_out,
            EasingType.CUBIC_IN_OUT: EasingFunction.cubic_in_out,
            EasingType.QUART_IN: EasingFunction.quart_in,
            EasingType.QUART_OUT: EasingFunction.quart_out,
            EasingType.QUART_IN_OUT: EasingFunction.quart_in_out,
            EasingType.QUINT_IN: EasingFunction.quint_in,
            EasingType.QUINT_OUT: EasingFunction.quint_out,
            EasingType.QUINT_IN_OUT: EasingFunction.quint_in_out,
            EasingType.EXPO_IN: EasingFunction.expo_in,
            EasingType.EXPO_OUT: EasingFunction.expo_out,
            EasingType.EXPO_IN_OUT: EasingFunction.expo_in_out,
            EasingType.CIRC_IN: EasingFunction.circ_in,
            EasingType.CIRC_OUT: EasingFunction.circ_out,
            EasingType.CIRC_IN_OUT: EasingFunction.circ_in_out,
            EasingType.ELASTIC_IN: EasingFunction.elastic_in,
            EasingType.ELASTIC_OUT: EasingFunction.elastic_out,
            EasingType.ELASTIC_IN_OUT: EasingFunction.elastic_in_out,
            EasingType.BACK_IN: EasingFunction.back_in,
            EasingType.BACK_OUT: EasingFunction.back_out,
            EasingType.BACK_IN_OUT: EasingFunction.back_in_out,
            EasingType.BOUNCE_IN: EasingFunction.bounce_in,
            EasingType.BOUNCE_OUT: EasingFunction.bounce_out,
            EasingType.BOUNCE_IN_OUT: EasingFunction.bounce_in_out,
        }
        return mapping.get(easing_type, EasingFunction.linear)


# =============================================================================
# ANIMATION CURVES
# =============================================================================


class AnimationCurve(ABC):
    """Base class for animation curves.

    An animation curve maps time to value using various interpolation
    methods.
    """

    def __init__(self, name: str = "Curve") -> None:
        self._name = name
        self._keys: List[CurveKey] = []
        self._curve_type = CurveType.LINEAR
        self._pre_infinity: str = "constant"  # constant, linear, cycle, oscillate
        self._post_infinity: str = "constant"

    @property
    def name(self) -> str:
        """Get curve name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set curve name."""
        self._name = value

    @property
    def curve_type(self) -> CurveType:
        """Get curve type."""
        return self._curve_type

    @property
    def keys(self) -> List[CurveKey]:
        """Get all keys."""
        return list(self._keys)

    @property
    def key_count(self) -> int:
        """Get number of keys."""
        return len(self._keys)

    def add_key(self, key: CurveKey) -> int:
        """Add a key to the curve."""
        # Insert in sorted order
        index = 0
        for i, k in enumerate(self._keys):
            if k.time > key.time:
                break
            index = i + 1

        self._keys.insert(index, key)
        self._update_auto_tangents()
        return index

    def add_key_at(self, time: float, value: float) -> CurveKey:
        """Add a key at the specified time and value."""
        key = CurveKey(time=time, value=value)
        self.add_key(key)
        return key

    def remove_key(self, index: int) -> bool:
        """Remove key at index."""
        if 0 <= index < len(self._keys):
            self._keys.pop(index)
            self._update_auto_tangents()
            return True
        return False

    def get_key(self, index: int) -> Optional[CurveKey]:
        """Get key at index."""
        if 0 <= index < len(self._keys):
            return self._keys[index]
        return None

    def move_key(self, index: int, new_time: float, new_value: float) -> bool:
        """Move a key to new time and value."""
        if not (0 <= index < len(self._keys)):
            return False

        key = self._keys.pop(index)
        key.time = new_time
        key.value = new_value
        self.add_key(key)
        return True

    def get_surrounding_keys(self, time: float) -> Tuple[Optional[CurveKey], Optional[CurveKey]]:
        """Get keys before and after time."""
        before: Optional[CurveKey] = None
        after: Optional[CurveKey] = None

        for key in self._keys:
            if key.time <= time:
                before = key
            elif after is None:
                after = key
                break

        return before, after

    @abstractmethod
    def evaluate(self, time: float) -> float:
        """Evaluate the curve at time."""
        pass

    def _update_auto_tangents(self) -> None:
        """Update auto tangents for all keys."""
        for i, key in enumerate(self._keys):
            if key.tangent_mode == TangentMode.AUTO:
                prev_key = self._keys[i - 1] if i > 0 else None
                next_key = self._keys[i + 1] if i < len(self._keys) - 1 else None
                self._compute_auto_tangent(key, prev_key, next_key)

    def _compute_auto_tangent(
        self,
        key: CurveKey,
        prev_key: Optional[CurveKey],
        next_key: Optional[CurveKey],
    ) -> None:
        """Compute automatic tangent for a key."""
        if prev_key is None and next_key is None:
            key.tangent_in = TangentHandle(-0.33, 0.0)
            key.tangent_out = TangentHandle(0.33, 0.0)
            return

        if prev_key is None:
            # First key - match slope to next
            dt = next_key.time - key.time
            dv = next_key.value - key.value
            key.tangent_out = TangentHandle(dt * 0.33, dv * 0.33)
            key.tangent_in = TangentHandle(-dt * 0.33, -dv * 0.33)
            return

        if next_key is None:
            # Last key - match slope from prev
            dt = key.time - prev_key.time
            dv = key.value - prev_key.value
            key.tangent_in = TangentHandle(-dt * 0.33, -dv * 0.33)
            key.tangent_out = TangentHandle(dt * 0.33, dv * 0.33)
            return

        # Middle key - average slopes
        dt1 = key.time - prev_key.time
        dv1 = key.value - prev_key.value
        dt2 = next_key.time - key.time
        dv2 = next_key.value - key.value

        # Use Catmull-Rom style tangent
        slope = (dv1 / dt1 + dv2 / dt2) / 2.0 if dt1 > 0 and dt2 > 0 else 0.0

        key.tangent_in = TangentHandle(-dt1 * 0.33, -dt1 * 0.33 * slope)
        key.tangent_out = TangentHandle(dt2 * 0.33, dt2 * 0.33 * slope)

    @property
    def duration(self) -> float:
        """Get curve duration."""
        if not self._keys:
            return 0.0
        return self._keys[-1].time

    def get_value_range(self) -> Tuple[float, float]:
        """Get min and max values."""
        if not self._keys:
            return (0.0, 0.0)

        min_val = min(k.value for k in self._keys)
        max_val = max(k.value for k in self._keys)
        return (min_val, max_val)


class LinearCurve(AnimationCurve):
    """Linear interpolation curve."""

    def __init__(self, name: str = "Linear Curve") -> None:
        super().__init__(name)
        self._curve_type = CurveType.LINEAR

    def evaluate(self, time: float) -> float:
        """Evaluate with linear interpolation."""
        if not self._keys:
            return 0.0

        if time <= self._keys[0].time:
            return self._keys[0].value

        if time >= self._keys[-1].time:
            return self._keys[-1].value

        before, after = self.get_surrounding_keys(time)

        if before is None:
            return self._keys[0].value
        if after is None:
            return before.value

        t = (time - before.time) / (after.time - before.time)
        return before.value + (after.value - before.value) * t


class SteppedCurve(AnimationCurve):
    """Stepped (constant) interpolation curve."""

    def __init__(self, name: str = "Stepped Curve") -> None:
        super().__init__(name)
        self._curve_type = CurveType.STEPPED

    def evaluate(self, time: float) -> float:
        """Evaluate with stepped interpolation."""
        if not self._keys:
            return 0.0

        # Return value of key at or before time
        result = self._keys[0].value
        for key in self._keys:
            if key.time <= time:
                result = key.value
            else:
                break

        return result


class BezierCurve(AnimationCurve):
    """Bezier interpolation curve with tangent handles."""

    def __init__(self, name: str = "Bezier Curve") -> None:
        super().__init__(name)
        self._curve_type = CurveType.BEZIER

    def evaluate(self, time: float) -> float:
        """Evaluate with cubic Bezier interpolation."""
        if not self._keys:
            return 0.0

        if time <= self._keys[0].time:
            return self._keys[0].value

        if time >= self._keys[-1].time:
            return self._keys[-1].value

        before, after = self.get_surrounding_keys(time)

        if before is None:
            return self._keys[0].value
        if after is None:
            return before.value

        # Cubic Bezier with tangent handles
        dt = after.time - before.time
        if dt < 1e-6:
            return before.value

        t = (time - before.time) / dt

        # Control points
        p0 = before.value
        p1 = before.value + before.tangent_out.y
        p2 = after.value + after.tangent_in.y
        p3 = after.value

        # De Casteljau's algorithm
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        return mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3


class HermiteCurve(AnimationCurve):
    """Hermite interpolation curve."""

    def __init__(self, name: str = "Hermite Curve") -> None:
        super().__init__(name)
        self._curve_type = CurveType.HERMITE

    def evaluate(self, time: float) -> float:
        """Evaluate with Hermite interpolation."""
        if not self._keys:
            return 0.0

        if time <= self._keys[0].time:
            return self._keys[0].value

        if time >= self._keys[-1].time:
            return self._keys[-1].value

        before, after = self.get_surrounding_keys(time)

        if before is None:
            return self._keys[0].value
        if after is None:
            return before.value

        dt = after.time - before.time
        if dt < 1e-6:
            return before.value

        t = (time - before.time) / dt

        # Hermite basis functions
        h00 = 2 * t * t * t - 3 * t * t + 1
        h10 = t * t * t - 2 * t * t + t
        h01 = -2 * t * t * t + 3 * t * t
        h11 = t * t * t - t * t

        # Tangent slopes
        m0 = before.tangent_out.slope * dt
        m1 = after.tangent_in.slope * dt

        return h00 * before.value + h10 * m0 + h01 * after.value + h11 * m1


# =============================================================================
# CURVE SELECTION
# =============================================================================


@dataclass
class CurveSelection:
    """Represents a selection in the curve editor.

    Attributes:
        curve_index: Index of selected curve
        key_indices: Selected key indices
        tangent_handles: Selected tangent handles (in, out)
    """

    curve_index: int = -1
    key_indices: List[int] = field(default_factory=list)
    tangent_handles: List[Tuple[int, str]] = field(default_factory=list)  # (key_idx, "in"/"out")

    @property
    def has_selection(self) -> bool:
        """Check if anything is selected."""
        return self.curve_index >= 0 or len(self.key_indices) > 0

    def clear(self) -> None:
        """Clear the selection."""
        self.curve_index = -1
        self.key_indices.clear()
        self.tangent_handles.clear()

    def add_key(self, index: int) -> None:
        """Add a key to selection."""
        if index not in self.key_indices:
            self.key_indices.append(index)

    def remove_key(self, index: int) -> None:
        """Remove a key from selection."""
        if index in self.key_indices:
            self.key_indices.remove(index)


# =============================================================================
# CURVE EDITOR
# =============================================================================


class CurveEditor:
    """Animation curve editor.

    Provides functionality for editing animation curves including key
    manipulation, tangent editing, and curve operations.
    """

    def __init__(self) -> None:
        self._curves: List[AnimationCurve] = []
        self._selection = CurveSelection()
        self._view_range_x: Tuple[float, float] = (0.0, 10.0)
        self._view_range_y: Tuple[float, float] = (-1.0, 1.0)
        self._grid_size_x: float = 1.0
        self._grid_size_y: float = 0.1
        self._snap_to_grid: bool = False
        self._show_tangents: bool = True

    @property
    def curves(self) -> List[AnimationCurve]:
        """Get all curves."""
        return list(self._curves)

    @property
    def curve_count(self) -> int:
        """Get number of curves."""
        return len(self._curves)

    @property
    def selection(self) -> CurveSelection:
        """Get current selection."""
        return self._selection

    def add_curve(self, curve: AnimationCurve) -> int:
        """Add a curve to the editor."""
        self._curves.append(curve)
        return len(self._curves) - 1

    def remove_curve(self, index: int) -> bool:
        """Remove curve at index."""
        if 0 <= index < len(self._curves):
            self._curves.pop(index)
            if self._selection.curve_index == index:
                self._selection.clear()
            return True
        return False

    def get_curve(self, index: int) -> Optional[AnimationCurve]:
        """Get curve at index."""
        if 0 <= index < len(self._curves):
            return self._curves[index]
        return None

    def select_curve(self, index: int) -> None:
        """Select a curve."""
        self._selection.clear()
        if 0 <= index < len(self._curves):
            self._selection.curve_index = index

    def select_key(self, curve_index: int, key_index: int, add_to_selection: bool = False) -> None:
        """Select a key."""
        if not add_to_selection:
            self._selection.key_indices.clear()
            self._selection.tangent_handles.clear()

        self._selection.curve_index = curve_index
        self._selection.add_key(key_index)

    def add_key_at_time(self, curve_index: int, time: float) -> Optional[CurveKey]:
        """Add a key at the current evaluated value."""
        curve = self.get_curve(curve_index)
        if curve is None:
            return None

        value = curve.evaluate(time)
        key = curve.add_key_at(time, value)
        return key

    def delete_selected_keys(self) -> int:
        """Delete selected keys."""
        if self._selection.curve_index < 0:
            return 0

        curve = self.get_curve(self._selection.curve_index)
        if curve is None:
            return 0

        # Delete in reverse order to maintain indices
        deleted = 0
        for index in sorted(self._selection.key_indices, reverse=True):
            if curve.remove_key(index):
                deleted += 1

        self._selection.key_indices.clear()
        return deleted

    def move_selected_keys(self, delta_time: float, delta_value: float) -> None:
        """Move selected keys."""
        if self._selection.curve_index < 0:
            return

        curve = self.get_curve(self._selection.curve_index)
        if curve is None:
            return

        for index in self._selection.key_indices:
            key = curve.get_key(index)
            if key:
                new_time = key.time + delta_time
                new_value = key.value + delta_value

                if self._snap_to_grid:
                    new_time = round(new_time / self._grid_size_x) * self._grid_size_x
                    new_value = round(new_value / self._grid_size_y) * self._grid_size_y

                curve.move_key(index, new_time, new_value)

    def set_tangent_mode(self, mode: TangentMode) -> None:
        """Set tangent mode for selected keys."""
        if self._selection.curve_index < 0:
            return

        curve = self.get_curve(self._selection.curve_index)
        if curve is None:
            return

        for index in self._selection.key_indices:
            key = curve.get_key(index)
            if key:
                key.tangent_mode = mode

                if mode == TangentMode.FLAT:
                    key.set_flat_tangents()
                elif mode == TangentMode.LINEAR:
                    prev_key = curve.get_key(index - 1)
                    next_key = curve.get_key(index + 1)
                    key.set_linear_tangents(prev_key, next_key)

    def flatten_tangents(self) -> None:
        """Flatten tangents on selected keys."""
        self.set_tangent_mode(TangentMode.FLAT)

    def break_tangents(self) -> None:
        """Break tangents to allow separate in/out editing."""
        self.set_tangent_mode(TangentMode.BREAK)

    def unify_tangents(self) -> None:
        """Unify tangents to mirror each other."""
        self.set_tangent_mode(TangentMode.AUTO)

    def set_view_range(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        """Set the view range."""
        self._view_range_x = (x_min, x_max)
        self._view_range_y = (y_min, y_max)

    def frame_all(self) -> None:
        """Frame view to show all curves."""
        if not self._curves:
            self._view_range_x = (0.0, 10.0)
            self._view_range_y = (-1.0, 1.0)
            return

        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        for curve in self._curves:
            for key in curve.keys:
                x_min = min(x_min, key.time)
                x_max = max(x_max, key.time)
                y_min = min(y_min, key.value)
                y_max = max(y_max, key.value)

        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min

        if x_range < 0.1:
            x_range = 1.0
        if y_range < 0.1:
            y_range = 1.0

        self._view_range_x = (x_min - x_range * 0.1, x_max + x_range * 0.1)
        self._view_range_y = (y_min - y_range * 0.1, y_max + y_range * 0.1)

    def frame_selection(self) -> None:
        """Frame view to show selected keys."""
        if self._selection.curve_index < 0 or not self._selection.key_indices:
            return

        curve = self.get_curve(self._selection.curve_index)
        if curve is None:
            return

        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        for index in self._selection.key_indices:
            key = curve.get_key(index)
            if key:
                x_min = min(x_min, key.time)
                x_max = max(x_max, key.time)
                y_min = min(y_min, key.value)
                y_max = max(y_max, key.value)

        # Add padding
        x_range = max(x_max - x_min, 1.0)
        y_range = max(y_max - y_min, 1.0)

        self._view_range_x = (x_min - x_range * 0.2, x_max + x_range * 0.2)
        self._view_range_y = (y_min - y_range * 0.2, y_max + y_range * 0.2)

    def normalize_curve(self, curve_index: int) -> None:
        """Normalize curve values to 0-1 range."""
        curve = self.get_curve(curve_index)
        if curve is None or curve.key_count == 0:
            return

        min_val, max_val = curve.get_value_range()
        value_range = max_val - min_val

        if value_range < 1e-6:
            return

        for key in curve.keys:
            key.value = (key.value - min_val) / value_range

    def bake_curve(self, curve_index: int, interval: float = 0.1) -> Optional[LinearCurve]:
        """Bake curve to linear keys at regular intervals."""
        curve = self.get_curve(curve_index)
        if curve is None:
            return None

        baked = LinearCurve(f"{curve.name}_baked")
        duration = curve.duration

        t = 0.0
        while t <= duration:
            value = curve.evaluate(t)
            baked.add_key_at(t, value)
            t += interval

        return baked

    def apply_easing(self, curve_index: int, easing_type: EasingType) -> None:
        """Apply easing function to curve."""
        curve = self.get_curve(curve_index)
        if curve is None or curve.key_count < 2:
            return

        first_key = curve.get_key(0)
        last_key = curve.get_key(curve.key_count - 1)

        if first_key is None or last_key is None:
            return

        start_value = first_key.value
        end_value = last_key.value
        value_range = end_value - start_value

        duration = last_key.time - first_key.time
        easing_func = EasingFunction.get_function(easing_type)

        for key in curve.keys:
            t = (key.time - first_key.time) / duration if duration > 0 else 0
            eased_t = easing_func(t)
            key.value = start_value + value_range * eased_t


__all__ = [
    "CurveType",
    "TangentMode",
    "TangentHandle",
    "CurveKey",
    "EasingType",
    "EasingFunction",
    "AnimationCurve",
    "LinearCurve",
    "SteppedCurve",
    "BezierCurve",
    "HermiteCurve",
    "CurveSelection",
    "CurveEditor",
]
