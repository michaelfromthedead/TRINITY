"""Animation system configuration constants.

This module centralizes magic numbers and configuration values for the
animation crowds and systems modules, making them easy to tune and maintain.

Configuration values can be modified at runtime without restart. All values
are validated on assignment to ensure they remain within acceptable bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar, get_type_hints
import copy


T = TypeVar("T")


class ConfigValidationError(ValueError):
    """Raised when a configuration value fails validation."""

    pass


class ValidatedDescriptor(Generic[T]):
    """Descriptor that validates values on assignment.

    Provides runtime validation for configuration values with custom
    validators and type checking.
    """

    def __init__(
        self,
        default: T,
        validator: Callable[[T], bool] | None = None,
        error_message: str | None = None,
    ) -> None:
        self.default = default
        self.validator = validator
        self.error_message = error_message
        self.name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: object | None, objtype: type | None = None) -> T | "ValidatedDescriptor[T]":
        if obj is None:
            # Return the descriptor itself when accessed on the class
            return self  # type: ignore[return-value]
        return getattr(obj, f"_val_{self.name}", self.default)

    def __set__(self, obj: object, value: T) -> None:
        # Type validation
        expected_type = type(self.default)
        if not isinstance(value, expected_type):
            # Allow int for float fields
            if expected_type is float and isinstance(value, int):
                value = float(value)
            else:
                raise ConfigValidationError(
                    f"'{self.name}' must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        # Custom validation
        if self.validator is not None and not self.validator(value):
            msg = self.error_message or f"Invalid value for '{self.name}': {value}"
            raise ConfigValidationError(msg)

        setattr(obj, f"_val_{self.name}", value)


def positive(value: float | int) -> bool:
    """Validate that value is positive (> 0)."""
    return value > 0


def non_negative(value: float | int) -> bool:
    """Validate that value is non-negative (>= 0)."""
    return value >= 0


def at_least(minimum: float | int) -> Callable[[float | int], bool]:
    """Create validator that checks value >= minimum."""
    return lambda v: v >= minimum


def in_range(
    min_val: float | int, max_val: float | int
) -> Callable[[float | int], bool]:
    """Create validator that checks min_val <= value <= max_val."""
    return lambda v: min_val <= v <= max_val


def power_of_two(value: int) -> bool:
    """Validate that value is a power of 2."""
    return value > 0 and (value & (value - 1)) == 0


class MutableConfig:
    """Base class for mutable configuration objects.

    Supports runtime modification with validation, reset to defaults,
    and change notification callbacks.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_change_callbacks", [])

    def _ensure_init(self) -> None:
        """Ensure instance is properly initialized."""
        if not hasattr(self, "_change_callbacks"):
            object.__setattr__(self, "_change_callbacks", [])

    def _get_descriptors(self) -> dict[str, ValidatedDescriptor]:
        """Get all ValidatedDescriptor attributes from class hierarchy."""
        descriptors = {}
        for cls in type(self).__mro__:
            for name, attr in vars(cls).items():
                if isinstance(attr, ValidatedDescriptor) and name not in descriptors:
                    descriptors[name] = attr
        return descriptors

    def reset(self) -> None:
        """Reset all values to their defaults."""
        self._ensure_init()
        for name, descriptor in self._get_descriptors().items():
            # Directly set the underlying storage to avoid callbacks during reset
            object.__setattr__(self, f"_val_{name}", descriptor.default)

    def on_change(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Register a callback for configuration changes.

        Callback receives (field_name, old_value, new_value).
        """
        self._ensure_init()
        self._change_callbacks.append(callback)

    def remove_change_callback(
        self, callback: Callable[[str, Any, Any], None]
    ) -> None:
        """Remove a previously registered change callback."""
        self._ensure_init()
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, ValidatedDescriptor):
            self._ensure_init()
            old_value = getattr(self, name)
            descriptor.__set__(self, value)
            new_value = getattr(self, name)
            # Notify callbacks
            for cb in self._change_callbacks:
                cb(name, old_value, new_value)
        else:
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as a dictionary."""
        result = {}
        for name, descriptor in self._get_descriptors().items():
            result[name] = getattr(self, name)
        return result

    def from_dict(self, data: dict[str, Any]) -> None:
        """Import configuration from a dictionary.

        Raises ConfigValidationError if any value is invalid.
        """
        for key, value in data.items():
            if hasattr(type(self), key):
                setattr(self, key, value)


@dataclass(frozen=True)
class AnimationTextureConfig:
    """Configuration for animation texture baking."""
    # Maximum texture dimensions
    MAX_TEXTURE_WIDTH: int = 4096
    MAX_TEXTURE_HEIGHT: int = 4096

    # Default texture dimensions (power of 2 for GPU efficiency)
    DEFAULT_TEXTURE_WIDTH: int = 1024
    DEFAULT_TEXTURE_HEIGHT: int = 2048

    # Maximum bones per skeleton for texture baking
    MAX_BONES_PER_TEXTURE: int = 256  # 256 bones * 2 pixels = 512 width

    # Maximum animation frames
    MAX_FRAMES_PER_ANIMATION: int = 4096

    # Pack/unpack value ranges for RGBA8 encoding
    PACK_MIN_VALUE: float = -100.0
    PACK_MAX_VALUE: float = 100.0


@dataclass(frozen=True)
class CrowdRendererConfig:
    """Configuration for crowd rendering."""
    # Instance buffer settings
    MAX_INSTANCES_PER_BATCH: int = 1000
    DEFAULT_BUFFER_CAPACITY: int = 64
    BUFFER_GROWTH_FACTOR: int = 2

    # GPU buffer alignment (bytes, typically 16 for SSE/SIMD)
    BUFFER_ALIGNMENT: int = 16

    # Floats per instance data
    TRANSFORM_FLOATS: int = 16  # 4x4 matrix
    ANIMATION_FLOATS: int = 4   # anim_index, time, speed, lod
    COLOR_FLOATS: int = 4       # RGBA tint


class CrowdLODConfig(MutableConfig):
    """Configuration for crowd LOD system.

    All values can be modified at runtime. Invalid values raise
    ConfigValidationError.
    """

    # Default LOD distance thresholds (meters) - immutable tuple
    DEFAULT_LOD_DISTANCES: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0)

    # Maximum LOD levels
    MAX_LOD_LEVELS = ValidatedDescriptor(
        8,
        in_range(1, 16),
        "MAX_LOD_LEVELS must be between 1 and 16",
    )

    # Default culling distance (meters)
    DEFAULT_CULL_DISTANCE = ValidatedDescriptor(
        300.0,
        positive,
        "DEFAULT_CULL_DISTANCE must be positive",
    )

    # LOD transition settings
    DEFAULT_TRANSITION_DURATION = ValidatedDescriptor(
        0.2,
        non_negative,
        "DEFAULT_TRANSITION_DURATION must be non-negative",
    )

    DEFAULT_HYSTERESIS = ValidatedDescriptor(
        1.0,
        non_negative,
        "DEFAULT_HYSTERESIS must be non-negative",
    )

    # Minimum update rate for distant LODs
    MIN_UPDATE_RATE = ValidatedDescriptor(
        0.25,
        positive,
        "MIN_UPDATE_RATE must be positive",
    )

    # Minimum bone count at lowest LOD
    MIN_BONES_AT_LOWEST_LOD = ValidatedDescriptor(
        4,
        at_least(1),
        "MIN_BONES_AT_LOWEST_LOD must be at least 1",
    )


class CrowdBehaviorConfig(MutableConfig):
    """Configuration for crowd behavior simulation.

    All values can be modified at runtime without restart. Invalid values
    are rejected with ConfigValidationError.

    Example:
        >>> config = CrowdBehaviorConfig()
        >>> config.DEFAULT_AGENT_SPEED = 2.0  # OK
        >>> config.DEFAULT_AGENT_SPEED = -1   # Raises ConfigValidationError
    """

    # Default agent settings
    DEFAULT_AGENT_SPEED = ValidatedDescriptor(
        1.4,
        positive,
        "DEFAULT_AGENT_SPEED must be positive (got negative or zero value)",
    )

    DEFAULT_AGENT_TURN_SPEED = ValidatedDescriptor(
        3.14,
        positive,
        "DEFAULT_AGENT_TURN_SPEED must be positive",
    )

    DEFAULT_AGENT_RADIUS = ValidatedDescriptor(
        0.4,
        positive,
        "DEFAULT_AGENT_RADIUS must be positive",
    )

    # Avoidance settings
    DEFAULT_AVOIDANCE_RADIUS = ValidatedDescriptor(
        2.0,
        positive,
        "DEFAULT_AVOIDANCE_RADIUS must be positive",
    )

    # Alias for backward compatibility
    @property
    def avoidance_radius(self) -> float:
        """Alias for DEFAULT_AVOIDANCE_RADIUS."""
        return self.DEFAULT_AVOIDANCE_RADIUS

    @avoidance_radius.setter
    def avoidance_radius(self, value: float) -> None:
        """Set avoidance radius with validation."""
        self.DEFAULT_AVOIDANCE_RADIUS = value

    DEFAULT_AVOIDANCE_STRENGTH = ValidatedDescriptor(
        1.5,
        positive,
        "DEFAULT_AVOIDANCE_STRENGTH must be positive",
    )

    AVOIDANCE_PRIORITY_MULTIPLIER = ValidatedDescriptor(
        1.5,
        at_least(1.0),
        "AVOIDANCE_PRIORITY_MULTIPLIER must be >= 1.0",
    )

    # Movement settings
    ARRIVAL_THRESHOLD = ValidatedDescriptor(
        0.5,
        positive,
        "ARRIVAL_THRESHOLD must be positive",
    )

    VELOCITY_SMOOTHING = ValidatedDescriptor(
        4.0,
        positive,
        "VELOCITY_SMOOTHING must be positive",
    )

    FLEE_ACCELERATION = ValidatedDescriptor(
        8.0,
        positive,
        "FLEE_ACCELERATION must be positive",
    )

    # Idle behavior
    IDLE_VARIATION_MIN = ValidatedDescriptor(
        3.0,
        non_negative,
        "IDLE_VARIATION_MIN must be non-negative",
    )

    IDLE_VARIATION_MAX = ValidatedDescriptor(
        8.0,
        positive,
        "IDLE_VARIATION_MAX must be positive",
    )

    # Fleeing behavior
    FLEE_SPEED_MULTIPLIER = ValidatedDescriptor(
        1.5,
        positive,
        "FLEE_SPEED_MULTIPLIER must be positive",
    )

    FLEE_SAFE_DISTANCE = ValidatedDescriptor(
        20.0,
        positive,
        "FLEE_SAFE_DISTANCE must be positive",
    )

    # Minimum distance to avoid division by zero
    MIN_DISTANCE_EPSILON = ValidatedDescriptor(
        0.01,
        positive,
        "MIN_DISTANCE_EPSILON must be positive (prevents division by zero)",
    )


class AnimationSystemConfig(MutableConfig):
    """Configuration for animation ECS systems.

    Priorities can be adjusted at runtime to reorder system execution.
    """

    # System execution priorities (lower = earlier)
    PRIORITY_ANIMATION_GRAPH = ValidatedDescriptor(
        100,
        non_negative,
        "PRIORITY_ANIMATION_GRAPH must be non-negative",
    )

    PRIORITY_MOTION_MATCHING = ValidatedDescriptor(
        150,
        non_negative,
        "PRIORITY_MOTION_MATCHING must be non-negative",
    )

    PRIORITY_IK = ValidatedDescriptor(
        200,
        non_negative,
        "PRIORITY_IK must be non-negative",
    )

    PRIORITY_PROCEDURAL = ValidatedDescriptor(
        300,
        non_negative,
        "PRIORITY_PROCEDURAL must be non-negative",
    )

    PRIORITY_FACIAL = ValidatedDescriptor(
        300,  # Parallel to procedural (different bones)
        non_negative,
        "PRIORITY_FACIAL must be non-negative",
    )

    PRIORITY_SKINNING = ValidatedDescriptor(
        400,
        non_negative,
        "PRIORITY_SKINNING must be non-negative",
    )

    PRIORITY_CROWD = ValidatedDescriptor(
        500,
        non_negative,
        "PRIORITY_CROWD must be non-negative",
    )

    # Default transition durations
    DEFAULT_GRAPH_TRANSITION = ValidatedDescriptor(
        0.2,
        non_negative,
        "DEFAULT_GRAPH_TRANSITION must be non-negative",
    )

    DEFAULT_MOTION_MATCH_TRANSITION = ValidatedDescriptor(
        0.2,
        non_negative,
        "DEFAULT_MOTION_MATCH_TRANSITION must be non-negative",
    )

    # Motion matching settings
    MOTION_MATCH_SEARCH_INTERVAL = ValidatedDescriptor(
        10,
        at_least(1),
        "MOTION_MATCH_SEARCH_INTERVAL must be at least 1",
    )

    MOTION_MATCH_CONTINUATION_COST = ValidatedDescriptor(
        0.5,
        non_negative,
        "MOTION_MATCH_CONTINUATION_COST must be non-negative",
    )


class IKConfig(MutableConfig):
    """Configuration for IK system."""

    # Solver defaults
    DEFAULT_MAX_ITERATIONS = ValidatedDescriptor(
        10,
        at_least(1),
        "DEFAULT_MAX_ITERATIONS must be at least 1",
    )

    DEFAULT_POSITION_TOLERANCE = ValidatedDescriptor(
        0.001,
        positive,
        "DEFAULT_POSITION_TOLERANCE must be positive",
    )

    DEFAULT_ROTATION_TOLERANCE = ValidatedDescriptor(
        0.01,
        positive,
        "DEFAULT_ROTATION_TOLERANCE must be positive",
    )

    # Solver constraints
    MAX_CHAIN_LENGTH = ValidatedDescriptor(
        10,
        at_least(1),
        "MAX_CHAIN_LENGTH must be at least 1",
    )

    # Distance thresholds to avoid numerical issues
    MIN_BONE_LENGTH = ValidatedDescriptor(
        0.001,
        positive,
        "MIN_BONE_LENGTH must be positive",
    )

    MIN_TARGET_DISTANCE = ValidatedDescriptor(
        0.001,
        positive,
        "MIN_TARGET_DISTANCE must be positive",
    )


class ProceduralConfig(MutableConfig):
    """Configuration for procedural animation."""

    # Spring defaults
    DEFAULT_SPRING_STIFFNESS = ValidatedDescriptor(
        10.0,
        positive,
        "DEFAULT_SPRING_STIFFNESS must be positive",
    )

    DEFAULT_SPRING_DAMPING = ValidatedDescriptor(
        0.5,
        non_negative,
        "DEFAULT_SPRING_DAMPING must be non-negative",
    )

    DEFAULT_SPRING_MASS = ValidatedDescriptor(
        1.0,
        positive,
        "DEFAULT_SPRING_MASS must be positive",
    )

    DEFAULT_MAX_STRETCH = ValidatedDescriptor(
        0.5,
        positive,
        "DEFAULT_MAX_STRETCH must be positive",
    )

    # Look-at defaults
    DEFAULT_LOOK_SPEED = ValidatedDescriptor(
        5.0,
        positive,
        "DEFAULT_LOOK_SPEED must be positive",
    )

    DEFAULT_HORIZONTAL_LIMIT = ValidatedDescriptor(
        1.5708,  # pi/2 radians (90 degrees)
        positive,
        "DEFAULT_HORIZONTAL_LIMIT must be positive",
    )

    DEFAULT_VERTICAL_LIMIT = ValidatedDescriptor(
        1.0472,  # pi/3 radians (60 degrees)
        positive,
        "DEFAULT_VERTICAL_LIMIT must be positive",
    )

    # Sway defaults
    DEFAULT_SWAY_FREQUENCY = ValidatedDescriptor(
        1.0,
        positive,
        "DEFAULT_SWAY_FREQUENCY must be positive",
    )

    DEFAULT_NOISE_AMOUNT = ValidatedDescriptor(
        0.2,
        non_negative,
        "DEFAULT_NOISE_AMOUNT must be non-negative",
    )

    # Breathing defaults
    DEFAULT_BREATH_RATE = ValidatedDescriptor(
        0.25,  # breaths/sec (15/min)
        positive,
        "DEFAULT_BREATH_RATE must be positive",
    )

    DEFAULT_BREATH_DEPTH = ValidatedDescriptor(
        0.02,
        positive,
        "DEFAULT_BREATH_DEPTH must be positive",
    )


class SkinningConfig(MutableConfig):
    """Configuration for skinning system."""

    # Vertex skinning
    DEFAULT_MAX_INFLUENCES = ValidatedDescriptor(
        4,
        at_least(1),
        "DEFAULT_MAX_INFLUENCES must be at least 1",
    )

    # Dual quaternion threshold
    DQ_BLEND_THRESHOLD = ValidatedDescriptor(
        0.5,
        in_range(0.0, 1.0),
        "DQ_BLEND_THRESHOLD must be between 0.0 and 1.0",
    )

    # Numerical stability
    MIN_QUATERNION_LENGTH = ValidatedDescriptor(
        0.0001,
        positive,
        "MIN_QUATERNION_LENGTH must be positive",
    )

    MIN_WEIGHT_THRESHOLD = ValidatedDescriptor(
        0.0001,
        positive,
        "MIN_WEIGHT_THRESHOLD must be positive",
    )


class FacialConfig(MutableConfig):
    """Configuration for facial animation."""

    # Lip sync
    DEFAULT_PHONEME_TRANSITION = ValidatedDescriptor(
        0.08,
        positive,
        "DEFAULT_PHONEME_TRANSITION must be positive",
    )

    SILENCE_VOLUME_THRESHOLD = ValidatedDescriptor(
        0.01,
        non_negative,
        "SILENCE_VOLUME_THRESHOLD must be non-negative",
    )

    # Eye tracking
    DEFAULT_BLINK_INTERVAL_MIN = ValidatedDescriptor(
        2.0,
        positive,
        "DEFAULT_BLINK_INTERVAL_MIN must be positive",
    )

    DEFAULT_BLINK_INTERVAL_MAX = ValidatedDescriptor(
        6.0,
        positive,
        "DEFAULT_BLINK_INTERVAL_MAX must be positive",
    )

    DEFAULT_BLINK_DURATION = ValidatedDescriptor(
        0.15,
        positive,
        "DEFAULT_BLINK_DURATION must be positive",
    )

    DEFAULT_SACCADE_INTENSITY = ValidatedDescriptor(
        0.01,
        non_negative,
        "DEFAULT_SACCADE_INTENSITY must be non-negative",
    )


class CrowdSystemConfig(MutableConfig):
    """Configuration for crowd system."""

    # Default update rate
    DEFAULT_UPDATE_RATE = ValidatedDescriptor(
        30.0,
        positive,
        "DEFAULT_UPDATE_RATE must be positive",
    )

    # Instance limits
    DEFAULT_MAX_VISIBLE = ValidatedDescriptor(
        10000,
        at_least(1),
        "DEFAULT_MAX_VISIBLE must be at least 1",
    )

    DEFAULT_MAX_AGENTS = ValidatedDescriptor(
        100000,
        at_least(1),
        "DEFAULT_MAX_AGENTS must be at least 1",
    )

    # Default LOD distances for crowd system (immutable tuple)
    DEFAULT_LOD_DISTANCES: tuple[float, ...] = (20.0, 50.0, 100.0, 200.0)

    DEFAULT_CULL_DISTANCE = ValidatedDescriptor(
        300.0,
        positive,
        "DEFAULT_CULL_DISTANCE must be positive",
    )


# Global configuration instances
# These can be modified at runtime without application restart.
# Invalid values will raise ConfigValidationError.
ANIMATION_TEXTURE_CONFIG = AnimationTextureConfig()
CROWD_RENDERER_CONFIG = CrowdRendererConfig()
CROWD_LOD_CONFIG = CrowdLODConfig()
CROWD_BEHAVIOR_CONFIG = CrowdBehaviorConfig()
ANIMATION_SYSTEM_CONFIG = AnimationSystemConfig()
IK_CONFIG = IKConfig()
PROCEDURAL_CONFIG = ProceduralConfig()
SKINNING_CONFIG = SkinningConfig()
FACIAL_CONFIG = FacialConfig()
CROWD_SYSTEM_CONFIG = CrowdSystemConfig()


# Convenience function to reset all configs to defaults
def reset_all_configs() -> None:
    """Reset all mutable configurations to their default values."""
    for config in [
        CROWD_LOD_CONFIG,
        CROWD_BEHAVIOR_CONFIG,
        ANIMATION_SYSTEM_CONFIG,
        IK_CONFIG,
        PROCEDURAL_CONFIG,
        SKINNING_CONFIG,
        FACIAL_CONFIG,
        CROWD_SYSTEM_CONFIG,
    ]:
        config.reset()


__all__ = [
    # Exception
    "ConfigValidationError",
    # Validation helpers
    "ValidatedDescriptor",
    "positive",
    "non_negative",
    "at_least",
    "in_range",
    "power_of_two",
    # Base class
    "MutableConfig",
    # Config classes
    "AnimationTextureConfig",
    "CrowdRendererConfig",
    "CrowdLODConfig",
    "CrowdBehaviorConfig",
    "AnimationSystemConfig",
    "IKConfig",
    "ProceduralConfig",
    "SkinningConfig",
    "FacialConfig",
    "CrowdSystemConfig",
    # Global instances
    "ANIMATION_TEXTURE_CONFIG",
    "CROWD_RENDERER_CONFIG",
    "CROWD_LOD_CONFIG",
    "CROWD_BEHAVIOR_CONFIG",
    "ANIMATION_SYSTEM_CONFIG",
    "IK_CONFIG",
    "PROCEDURAL_CONFIG",
    "SKINNING_CONFIG",
    "FACIAL_CONFIG",
    "CROWD_SYSTEM_CONFIG",
    # Utility
    "reset_all_configs",
]
