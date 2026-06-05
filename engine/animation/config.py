"""Animation system configuration constants.

This module centralizes magic numbers and configuration values for the
animation crowds and systems modules, making them easy to tune and maintain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar, Generic, Callable, List, Dict


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


T = TypeVar('T')


class MutableConfig(Generic[T]):
    """Runtime-mutable configuration wrapper with change notification."""

    def __init__(self, initial: T):
        self._value = initial
        self._listeners: List[Callable[[T], None]] = []

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, new_value: T) -> None:
        self._value = new_value
        for listener in self._listeners:
            listener(new_value)

    def subscribe(self, callback: Callable[[T], None]) -> None:
        """Register a callback for value changes."""
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[T], None]) -> None:
        """Unregister a callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)


class ValidatedDescriptor:
    """Descriptor that validates values on assignment."""

    def __init__(self, validator: Callable[[Any], bool], error_msg: str = "Validation failed"):
        self.validator = validator
        self.error_msg = error_msg
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return getattr(obj, f"_validated_{self.name}", None)

    def __set__(self, obj: Any, value: Any) -> None:
        if not self.validator(value):
            raise ConfigValidationError(f"{self.name}: {self.error_msg}")
        setattr(obj, f"_validated_{self.name}", value)


def positive(value: float) -> bool:
    """Validate that value is positive (> 0)."""
    return value > 0


def non_negative(value: float) -> bool:
    """Validate that value is non-negative (>= 0)."""
    return value >= 0


def at_least(minimum: float) -> Callable[[float], bool]:
    """Create validator that checks value >= minimum."""
    def validator(value: float) -> bool:
        return value >= minimum
    return validator


def in_range(min_val: float, max_val: float) -> Callable[[float], bool]:
    """Create validator that checks min_val <= value <= max_val."""
    def validator(value: float) -> bool:
        return min_val <= value <= max_val
    return validator


class MutableConfigBase:
    """Base class for mutable configuration with validation, serialization, and callbacks."""

    # Subclasses should define _field_defaults, _field_validators, _field_types
    _field_defaults: Dict[str, Any] = {}
    _field_validators: Dict[str, Callable[[Any], bool]] = {}
    _field_types: Dict[str, type] = {}

    def __init__(self) -> None:
        self._change_callbacks: List[Callable[[str, Any, Any], None]] = []
        self._values: Dict[str, Any] = {}
        # Initialize with defaults
        for name, default in self._field_defaults.items():
            self._values[name] = default

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self._field_defaults:
            return self._values.get(name, self._field_defaults[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return

        if name not in self._field_defaults:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Type validation
        expected_type = self._field_types.get(name, type(self._field_defaults[name]))
        if expected_type == float and isinstance(value, int):
            value = float(value)
        elif expected_type == int and isinstance(value, float) and value == int(value):
            value = int(value)
        elif not isinstance(value, expected_type):
            raise ConfigValidationError(f"{name}: expected {expected_type.__name__}, got {type(value).__name__}")

        # Custom validation
        validator = self._field_validators.get(name)
        if validator and not validator(value):
            raise ConfigValidationError(f"{name}: validation failed")

        old_value = self._values.get(name, self._field_defaults[name])
        self._values[name] = value

        # Notify callbacks
        for callback in self._change_callbacks:
            callback(name, old_value, value)

    def reset(self) -> None:
        """Reset all fields to default values."""
        self._values = dict(self._field_defaults)

    def on_change(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Register a callback for value changes."""
        self._change_callbacks.append(callback)

    def remove_change_callback(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Remove a change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return dict(self._values)

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Import configuration from dictionary."""
        for name, value in data.items():
            if name in self._field_defaults:
                setattr(self, name, value)


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


class CrowdLODConfig(MutableConfigBase):
    """Configuration for crowd LOD system."""

    _field_defaults = {
        'DEFAULT_LOD_DISTANCES': (10.0, 25.0, 50.0, 100.0),
        'MAX_LOD_LEVELS': 8,
        'DEFAULT_CULL_DISTANCE': 300.0,
        'DEFAULT_TRANSITION_DURATION': 0.2,
        'DEFAULT_HYSTERESIS': 1.0,
        'MIN_UPDATE_RATE': 0.25,
        'MIN_BONES_AT_LOWEST_LOD': 4,
    }

    _field_validators = {
        'DEFAULT_CULL_DISTANCE': positive,
        'DEFAULT_TRANSITION_DURATION': non_negative,
        'DEFAULT_HYSTERESIS': non_negative,
        'MIN_UPDATE_RATE': positive,
        'MAX_LOD_LEVELS': lambda x: x >= 1,
        'MIN_BONES_AT_LOWEST_LOD': lambda x: x >= 1,
    }

    _field_types = {
        'DEFAULT_LOD_DISTANCES': tuple,
        'MAX_LOD_LEVELS': int,
        'DEFAULT_CULL_DISTANCE': float,
        'DEFAULT_TRANSITION_DURATION': float,
        'DEFAULT_HYSTERESIS': float,
        'MIN_UPDATE_RATE': float,
        'MIN_BONES_AT_LOWEST_LOD': int,
    }


class CrowdBehaviorConfig(MutableConfigBase):
    """Configuration for crowd behavior simulation."""

    _field_defaults = {
        'DEFAULT_AGENT_SPEED': 1.4,
        'DEFAULT_AGENT_TURN_SPEED': 3.14,
        'DEFAULT_AGENT_RADIUS': 0.4,
        'DEFAULT_AVOIDANCE_RADIUS': 2.0,
        'DEFAULT_AVOIDANCE_STRENGTH': 1.5,
        'AVOIDANCE_PRIORITY_MULTIPLIER': 1.5,
        'ARRIVAL_THRESHOLD': 0.5,
        'VELOCITY_SMOOTHING': 4.0,
        'FLEE_ACCELERATION': 8.0,
        'IDLE_VARIATION_MIN': 3.0,
        'IDLE_VARIATION_MAX': 8.0,
        'FLEE_SPEED_MULTIPLIER': 1.5,
        'FLEE_SAFE_DISTANCE': 20.0,
        'MIN_DISTANCE_EPSILON': 0.01,
        'avoidance_radius': 2.0,
    }

    _field_validators = {
        'DEFAULT_AGENT_SPEED': positive,
        'DEFAULT_AGENT_TURN_SPEED': positive,
        'DEFAULT_AGENT_RADIUS': non_negative,
        'DEFAULT_AVOIDANCE_RADIUS': positive,
        'DEFAULT_AVOIDANCE_STRENGTH': positive,
        'AVOIDANCE_PRIORITY_MULTIPLIER': at_least(1.0),
        'ARRIVAL_THRESHOLD': positive,
        'VELOCITY_SMOOTHING': positive,
        'FLEE_ACCELERATION': positive,
        'IDLE_VARIATION_MIN': positive,
        'IDLE_VARIATION_MAX': positive,
        'FLEE_SPEED_MULTIPLIER': at_least(1.0),
        'FLEE_SAFE_DISTANCE': positive,
        'MIN_DISTANCE_EPSILON': positive,
        'avoidance_radius': positive,
    }

    _field_types = {
        'DEFAULT_AGENT_SPEED': float,
        'DEFAULT_AGENT_TURN_SPEED': float,
        'DEFAULT_AGENT_RADIUS': float,
        'DEFAULT_AVOIDANCE_RADIUS': float,
        'DEFAULT_AVOIDANCE_STRENGTH': float,
        'AVOIDANCE_PRIORITY_MULTIPLIER': float,
        'ARRIVAL_THRESHOLD': float,
        'VELOCITY_SMOOTHING': float,
        'FLEE_ACCELERATION': float,
        'IDLE_VARIATION_MIN': float,
        'IDLE_VARIATION_MAX': float,
        'FLEE_SPEED_MULTIPLIER': float,
        'FLEE_SAFE_DISTANCE': float,
        'MIN_DISTANCE_EPSILON': float,
        'avoidance_radius': float,
    }


class AnimationSystemConfig(MutableConfigBase):
    """Configuration for animation ECS systems."""

    _field_defaults = {
        'PRIORITY_ANIMATION_GRAPH': 100,
        'PRIORITY_MOTION_MATCHING': 150,
        'PRIORITY_IK': 200,
        'PRIORITY_PROCEDURAL': 300,
        'PRIORITY_FACIAL': 300,
        'PRIORITY_SKINNING': 400,
        'PRIORITY_CROWD': 500,
        'DEFAULT_GRAPH_TRANSITION': 0.2,
        'DEFAULT_MOTION_MATCH_TRANSITION': 0.2,
        'MOTION_MATCH_SEARCH_INTERVAL': 10,
        'MOTION_MATCH_CONTINUATION_COST': 0.5,
    }

    _field_validators = {
        'PRIORITY_ANIMATION_GRAPH': non_negative,
        'PRIORITY_MOTION_MATCHING': non_negative,
        'PRIORITY_IK': non_negative,
        'PRIORITY_PROCEDURAL': non_negative,
        'PRIORITY_FACIAL': non_negative,
        'PRIORITY_SKINNING': non_negative,
        'PRIORITY_CROWD': non_negative,
        'DEFAULT_GRAPH_TRANSITION': non_negative,
        'DEFAULT_MOTION_MATCH_TRANSITION': non_negative,
        'MOTION_MATCH_SEARCH_INTERVAL': lambda x: x >= 1,
        'MOTION_MATCH_CONTINUATION_COST': non_negative,
    }

    _field_types = {
        'PRIORITY_ANIMATION_GRAPH': int,
        'PRIORITY_MOTION_MATCHING': int,
        'PRIORITY_IK': int,
        'PRIORITY_PROCEDURAL': int,
        'PRIORITY_FACIAL': int,
        'PRIORITY_SKINNING': int,
        'PRIORITY_CROWD': int,
        'DEFAULT_GRAPH_TRANSITION': float,
        'DEFAULT_MOTION_MATCH_TRANSITION': float,
        'MOTION_MATCH_SEARCH_INTERVAL': int,
        'MOTION_MATCH_CONTINUATION_COST': float,
    }


class IKConfig(MutableConfigBase):
    """Configuration for IK system."""

    _field_defaults = {
        'DEFAULT_MAX_ITERATIONS': 10,
        'DEFAULT_POSITION_TOLERANCE': 0.001,
        'DEFAULT_ROTATION_TOLERANCE': 0.01,
        'MAX_CHAIN_LENGTH': 10,
        'MIN_BONE_LENGTH': 0.001,
        'MIN_TARGET_DISTANCE': 0.001,
    }

    _field_validators = {
        'DEFAULT_MAX_ITERATIONS': lambda x: x >= 1,
        'DEFAULT_POSITION_TOLERANCE': positive,
        'DEFAULT_ROTATION_TOLERANCE': positive,
        'MAX_CHAIN_LENGTH': lambda x: x >= 1,
        'MIN_BONE_LENGTH': positive,
        'MIN_TARGET_DISTANCE': positive,
    }

    _field_types = {
        'DEFAULT_MAX_ITERATIONS': int,
        'DEFAULT_POSITION_TOLERANCE': float,
        'DEFAULT_ROTATION_TOLERANCE': float,
        'MAX_CHAIN_LENGTH': int,
        'MIN_BONE_LENGTH': float,
        'MIN_TARGET_DISTANCE': float,
    }


class ProceduralConfig(MutableConfigBase):
    """Configuration for procedural animation."""

    _field_defaults = {
        'DEFAULT_SPRING_STIFFNESS': 10.0,
        'DEFAULT_SPRING_DAMPING': 0.5,
        'DEFAULT_SPRING_MASS': 1.0,
        'DEFAULT_MAX_STRETCH': 0.5,
        'DEFAULT_LOOK_SPEED': 5.0,
        'DEFAULT_HORIZONTAL_LIMIT': 1.5708,
        'DEFAULT_VERTICAL_LIMIT': 1.0472,
        'DEFAULT_SWAY_FREQUENCY': 1.0,
        'DEFAULT_NOISE_AMOUNT': 0.2,
        'DEFAULT_BREATH_RATE': 0.25,
        'DEFAULT_BREATH_DEPTH': 0.02,
    }

    _field_validators = {
        'DEFAULT_SPRING_STIFFNESS': positive,
        'DEFAULT_SPRING_DAMPING': non_negative,
        'DEFAULT_SPRING_MASS': positive,
        'DEFAULT_MAX_STRETCH': non_negative,
        'DEFAULT_LOOK_SPEED': positive,
        'DEFAULT_HORIZONTAL_LIMIT': positive,
        'DEFAULT_VERTICAL_LIMIT': positive,
        'DEFAULT_SWAY_FREQUENCY': positive,
        'DEFAULT_NOISE_AMOUNT': non_negative,
        'DEFAULT_BREATH_RATE': positive,
        'DEFAULT_BREATH_DEPTH': non_negative,
    }

    _field_types = {
        'DEFAULT_SPRING_STIFFNESS': float,
        'DEFAULT_SPRING_DAMPING': float,
        'DEFAULT_SPRING_MASS': float,
        'DEFAULT_MAX_STRETCH': float,
        'DEFAULT_LOOK_SPEED': float,
        'DEFAULT_HORIZONTAL_LIMIT': float,
        'DEFAULT_VERTICAL_LIMIT': float,
        'DEFAULT_SWAY_FREQUENCY': float,
        'DEFAULT_NOISE_AMOUNT': float,
        'DEFAULT_BREATH_RATE': float,
        'DEFAULT_BREATH_DEPTH': float,
    }


class SkinningConfig(MutableConfigBase):
    """Configuration for skinning system."""

    _field_defaults = {
        'DEFAULT_MAX_INFLUENCES': 4,
        'DQ_BLEND_THRESHOLD': 0.5,
        'MIN_QUATERNION_LENGTH': 0.0001,
        'MIN_WEIGHT_THRESHOLD': 0.0001,
    }

    _field_validators = {
        'DEFAULT_MAX_INFLUENCES': lambda x: x >= 1,
        'DQ_BLEND_THRESHOLD': in_range(0.0, 1.0),
        'MIN_QUATERNION_LENGTH': positive,
        'MIN_WEIGHT_THRESHOLD': positive,
    }

    _field_types = {
        'DEFAULT_MAX_INFLUENCES': int,
        'DQ_BLEND_THRESHOLD': float,
        'MIN_QUATERNION_LENGTH': float,
        'MIN_WEIGHT_THRESHOLD': float,
    }


class FacialConfig(MutableConfigBase):
    """Configuration for facial animation."""

    _field_defaults = {
        'DEFAULT_PHONEME_TRANSITION': 0.08,
        'SILENCE_VOLUME_THRESHOLD': 0.01,
        'DEFAULT_BLINK_INTERVAL_MIN': 2.0,
        'DEFAULT_BLINK_INTERVAL_MAX': 6.0,
        'DEFAULT_BLINK_DURATION': 0.15,
        'DEFAULT_SACCADE_INTENSITY': 0.01,
    }

    _field_validators = {
        'DEFAULT_PHONEME_TRANSITION': non_negative,
        'SILENCE_VOLUME_THRESHOLD': non_negative,
        'DEFAULT_BLINK_INTERVAL_MIN': positive,
        'DEFAULT_BLINK_INTERVAL_MAX': positive,
        'DEFAULT_BLINK_DURATION': positive,
        'DEFAULT_SACCADE_INTENSITY': non_negative,
    }

    _field_types = {
        'DEFAULT_PHONEME_TRANSITION': float,
        'SILENCE_VOLUME_THRESHOLD': float,
        'DEFAULT_BLINK_INTERVAL_MIN': float,
        'DEFAULT_BLINK_INTERVAL_MAX': float,
        'DEFAULT_BLINK_DURATION': float,
        'DEFAULT_SACCADE_INTENSITY': float,
    }


class CrowdSystemConfig(MutableConfigBase):
    """Configuration for crowd system."""

    _field_defaults = {
        'DEFAULT_UPDATE_RATE': 30.0,
        'DEFAULT_MAX_VISIBLE': 10000,
        'DEFAULT_MAX_AGENTS': 100000,
        'DEFAULT_LOD_DISTANCES': (20.0, 50.0, 100.0, 200.0),
        'DEFAULT_CULL_DISTANCE': 300.0,
    }

    _field_validators = {
        'DEFAULT_UPDATE_RATE': positive,
        'DEFAULT_MAX_VISIBLE': lambda x: x >= 1,
        'DEFAULT_MAX_AGENTS': lambda x: x >= 1,
        'DEFAULT_CULL_DISTANCE': positive,
    }

    _field_types = {
        'DEFAULT_UPDATE_RATE': float,
        'DEFAULT_MAX_VISIBLE': int,
        'DEFAULT_MAX_AGENTS': int,
        'DEFAULT_LOD_DISTANCES': tuple,
        'DEFAULT_CULL_DISTANCE': float,
    }


# Global configuration instances
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


def reset_all_configs() -> None:
    """Reset all mutable configs to their default values."""
    CROWD_LOD_CONFIG.reset()
    CROWD_BEHAVIOR_CONFIG.reset()
    ANIMATION_SYSTEM_CONFIG.reset()
    IK_CONFIG.reset()
    PROCEDURAL_CONFIG.reset()
    SKINNING_CONFIG.reset()
    FACIAL_CONFIG.reset()
    CROWD_SYSTEM_CONFIG.reset()
