"""Shared utilities for the XR module.

This package contains common utility classes and functions used across
the XR subsystem to eliminate code duplication.

Modules:
    math_utils: Rotation and quaternion conversion utilities
    markers: Type marker classes for descriptor annotations
    shading: Variable Rate Shading (VRS) utilities
"""

from .math_utils import (
    rotation_from_direction,
    rotation_from_axes,
    multiply_quaternions,
    quaternion_to_tuple,
    tuple_to_quaternion,
)
from .markers import (
    Tracked,
    Range,
    Observable,
    Transient,
    Immutable,
)
from .shading import (
    ShadingRateUtils,
    shading_rate_to_int,
    get_rate_multiplier,
)

__all__ = [
    # Math utilities
    "rotation_from_direction",
    "rotation_from_axes",
    "multiply_quaternions",
    "quaternion_to_tuple",
    "tuple_to_quaternion",
    # Type markers
    "Tracked",
    "Range",
    "Observable",
    "Transient",
    "Immutable",
    # Shading utilities
    "ShadingRateUtils",
    "shading_rate_to_int",
    "get_rate_multiplier",
]
