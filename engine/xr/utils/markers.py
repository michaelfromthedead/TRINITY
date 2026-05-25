"""Type markers for XR descriptor annotations.

These marker classes are used with Python's Annotated type hint to provide
metadata about descriptor fields. They enable features like change detection,
value validation, and serialization control.

Example usage:
    from typing import Annotated
    from engine.xr.utils.markers import Tracked, Range

    class MyComponent:
        speed: Annotated[float, Tracked(), Range(0.0, 10.0)] = 5.0
"""

from __future__ import annotations


class Tracked:
    """Marker for tracked descriptor - enables change detection.

    Fields marked with Tracked will notify observers when their value changes,
    enabling reactive updates in the XR system.
    """

    pass


class Range:
    """Marker for range-constrained values.

    Specifies minimum and maximum bounds for a value. Can be used for
    validation and UI generation (e.g., sliders).

    Args:
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
    """

    def __init__(self, min_val: float, max_val: float):
        self.min = min_val
        self.max = max_val

    def clamp(self, value: float) -> float:
        """Clamp a value to this range.

        Args:
            value: Value to clamp

        Returns:
            Value clamped to [min, max]
        """
        return max(self.min, min(self.max, value))

    def contains(self, value: float) -> bool:
        """Check if a value is within this range.

        Args:
            value: Value to check

        Returns:
            True if value is within [min, max]
        """
        return self.min <= value <= self.max


class Observable:
    """Marker for observable values that trigger UI updates.

    Fields marked with Observable will emit events when modified,
    allowing UI components to reactively update.
    """

    pass


class Transient:
    """Marker for non-serialized runtime state.

    Fields marked with Transient are not saved when the component state
    is serialized. Use for temporary runtime data like cached computations.
    """

    pass


class Immutable:
    """Marker for read-only configuration.

    Fields marked with Immutable should not be modified after initialization.
    This is a hint for tooling and documentation - not enforced at runtime.
    """

    pass
