"""
Perception Registry - Decorators and utilities for registering AI perception configs.

This module provides decorators for registering perception configurations with the
Foundation Registry for runtime discovery. Supports multiple sense types (sight,
hearing, damage, squad) with configurable range, FOV, and decay time.

Usage:
    from engine.gameplay.ai.perception_registry import perception, sense

    @perception(sense="sight", range=50.0, fov=90.0)
    class SniperPerception:
        pass

    @sense(type="hearing", range=30.0)
    class HearingConfig:
        pass

    # Query all perception configs:
    >>> from foundation import registry
    >>> registry.query(tag="perception")

    # Query by sense type:
    >>> registry.query(tag="perception", sense="sight")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Type, TypeVar, Union

from foundation import registry, Registry
from engine.gameplay.constants import (
    PerceptionSense,
    PERCEPTION_DEFAULT_SIGHT_RANGE,
    PERCEPTION_DEFAULT_HEARING_RANGE,
    PERCEPTION_DEFAULT_FOV,
)

# Type variable for decorator return types
T = TypeVar("T", bound=type)

# Tag constants for perception types
TAG_PERCEPTION = "perception"
TAG_SENSE = "sense"

# Valid sense types
VALID_SENSE_TYPES = frozenset({
    "sight",
    "hearing",
    "damage",
    "squad",
})

# Sense type to enum mapping
SENSE_TYPE_MAP = {
    "sight": PerceptionSense.SIGHT,
    "hearing": PerceptionSense.HEARING,
    "damage": PerceptionSense.DAMAGE,
    "squad": PerceptionSense.SQUAD,
}

# Default ranges for each sense type
DEFAULT_RANGES = {
    "sight": PERCEPTION_DEFAULT_SIGHT_RANGE,
    "hearing": PERCEPTION_DEFAULT_HEARING_RANGE,
    "damage": 0.0,  # Damage has no range limit
    "squad": 100.0,  # Squad communication range
}

# Default decay times for each sense type (seconds)
DEFAULT_DECAY_TIMES = {
    "sight": 3.0,
    "hearing": 2.0,
    "damage": 5.0,
    "squad": 10.0,
}


def perception(
    sense: str,
    range: Optional[float] = None,
    fov: Optional[float] = None,
    *,
    decay_time: Optional[float] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    track_instances: bool = False,
) -> Callable[[T], T]:
    """
    Decorator to register a class as a perception configuration with the Foundation Registry.

    This decorator:
    1. Registers the perception class with the Foundation Registry
    2. Tags it as "perception" and with the sense type tag
    3. Stores metadata (sense, range, fov, decay_time)

    Args:
        sense: The sense type ("sight", "hearing", "damage", "squad")
        range: Detection range for this sense. Defaults vary by sense type.
        fov: Field of view in degrees (primarily for sight). Defaults to 90.
        decay_time: How long before stimuli from this sense are forgotten.
        name: Optional custom registry name. Defaults to module.classname.
        description: Human-readable description of this perception config.
        track_instances: If True, track all instances via WeakSet.

    Returns:
        Decorated class registered with Foundation Registry.

    Example:
        @perception(sense="sight", range=100.0, fov=120.0)
        class EagleEyePerception:
            pass

        # Query all perception configs:
        >>> from foundation import registry
        >>> registry.query(tag="perception")

        # Query sight perceptions only:
        >>> registry.query(tag="perception", sense="sight")
    """
    if sense not in VALID_SENSE_TYPES:
        valid_senses = ", ".join(sorted(VALID_SENSE_TYPES))
        raise ValueError(f"Invalid sense type '{sense}'. Valid types: {valid_senses}")

    # Use defaults if not provided
    actual_range = range if range is not None else DEFAULT_RANGES.get(sense, 50.0)
    actual_fov = fov if fov is not None else PERCEPTION_DEFAULT_FOV
    actual_decay = decay_time if decay_time is not None else DEFAULT_DECAY_TIMES.get(sense, 3.0)

    def decorator(cls: T) -> T:
        # Mark class attributes for perception identification
        cls._perception = True
        cls._perception_sense = sense
        cls._perception_sense_enum = SENSE_TYPE_MAP[sense]
        cls._perception_range = actual_range
        cls._perception_fov = actual_fov
        cls._perception_decay_time = actual_decay
        cls._perception_description = description or ""

        # Register with Foundation Registry
        registry_name = name or f"perception.{cls.__module__}.{cls.__name__}"
        try:
            registry.register(cls, name=registry_name, track_instances=track_instances)
        except ValueError:
            # Already registered - fine in reload scenarios
            pass

        # Add tags for query-based discovery
        registry.add_tag(cls, TAG_PERCEPTION)
        registry.add_tag(cls, f"sense_{sense}")

        # Store metadata
        registry.set_metadata(cls, "sense", sense)
        registry.set_metadata(cls, "sense_enum", SENSE_TYPE_MAP[sense])
        registry.set_metadata(cls, "range", actual_range)
        registry.set_metadata(cls, "fov", actual_fov)
        registry.set_metadata(cls, "decay_time", actual_decay)
        if description:
            registry.set_metadata(cls, "description", description)

        return cls

    return decorator


def sense(
    type: str,
    range: Optional[float] = None,
    fov: Optional[float] = None,
    *,
    decay_time: Optional[float] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    track_instances: bool = False,
) -> Callable[[T], T]:
    """
    Decorator to register a class as an individual sense configuration.

    This is a more specialized version of @perception for configuring
    individual senses that can be attached to AI agents.

    Args:
        type: The sense type ("sight", "hearing", "damage", "squad")
        range: Detection range for this sense.
        fov: Field of view in degrees.
        decay_time: How long before stimuli from this sense are forgotten.
        name: Optional custom registry name.
        description: Human-readable description.
        track_instances: If True, track all instances via WeakSet.

    Returns:
        Decorated class registered with Foundation Registry.

    Example:
        @sense(type="hearing", range=50.0)
        class EnhancedHearing:
            pass

        # Query all sense configs:
        >>> registry.query(tag="sense")

        # Query hearing senses:
        >>> registry.query(tag="sense", sense_type="hearing")
    """
    if type not in VALID_SENSE_TYPES:
        valid_senses = ", ".join(sorted(VALID_SENSE_TYPES))
        raise ValueError(f"Invalid sense type '{type}'. Valid types: {valid_senses}")

    # Use defaults if not provided
    actual_range = range if range is not None else DEFAULT_RANGES.get(type, 50.0)
    actual_fov = fov if fov is not None else PERCEPTION_DEFAULT_FOV
    actual_decay = decay_time if decay_time is not None else DEFAULT_DECAY_TIMES.get(type, 3.0)

    def decorator(cls: T) -> T:
        # Mark class attributes for sense identification
        cls._sense = True
        cls._sense_type = type
        cls._sense_type_enum = SENSE_TYPE_MAP[type]
        cls._sense_range = actual_range
        cls._sense_fov = actual_fov
        cls._sense_decay_time = actual_decay
        cls._sense_description = description or ""

        # Register with Foundation Registry
        registry_name = name or f"sense.{cls.__module__}.{cls.__name__}"
        try:
            registry.register(cls, name=registry_name, track_instances=track_instances)
        except ValueError:
            # Already registered - fine in reload scenarios
            pass

        # Add tags for query-based discovery
        registry.add_tag(cls, TAG_SENSE)
        registry.add_tag(cls, f"sense_{type}")

        # Store metadata
        registry.set_metadata(cls, "sense_type", type)
        registry.set_metadata(cls, "sense_type_enum", SENSE_TYPE_MAP[type])
        registry.set_metadata(cls, "range", actual_range)
        registry.set_metadata(cls, "fov", actual_fov)
        registry.set_metadata(cls, "decay_time", actual_decay)
        if description:
            registry.set_metadata(cls, "description", description)

        return cls

    return decorator


# =============================================================================
# Query Helpers
# =============================================================================


def get_all_perception_configs() -> list[type]:
    """Get all registered perception configurations."""
    return registry.query(tag=TAG_PERCEPTION)


def get_perception_configs_by_sense(sense_type: str) -> list[type]:
    """Get all perception configurations for a specific sense type."""
    if sense_type not in VALID_SENSE_TYPES:
        return []
    return registry.query(tag=TAG_PERCEPTION, sense=sense_type)


def get_all_sense_configs() -> list[type]:
    """Get all registered sense configurations."""
    return registry.query(tag=TAG_SENSE)


def get_sense_configs_by_type(sense_type: str) -> list[type]:
    """Get all sense configurations of a specific type."""
    if sense_type not in VALID_SENSE_TYPES:
        return []
    return registry.query(tag=TAG_SENSE, sense_type=sense_type)


def get_perception_by_name(name: str) -> Optional[type]:
    """Get a perception configuration by its registered name."""
    cls = registry.get(name)
    if cls is None:
        # Try with perception prefix
        cls = registry.get(f"perception.{name}")
    return cls


def get_sense_by_name(name: str) -> Optional[type]:
    """Get a sense configuration by its registered name."""
    cls = registry.get(name)
    if cls is None:
        # Try with sense prefix
        cls = registry.get(f"sense.{name}")
    return cls


# =============================================================================
# Factory Functions
# =============================================================================


@dataclass
class PerceptionConfig:
    """
    Unified perception configuration data class.

    Stores all perception parameters for an AI agent, including
    multiple sense configurations.
    """

    name: str
    sense: str
    sense_enum: PerceptionSense
    range: float
    fov: float
    decay_time: float
    description: str = ""
    source_class: Optional[type] = None
    additional_senses: List["PerceptionConfig"] = field(default_factory=list)

    @classmethod
    def from_registry(cls, name: str) -> "PerceptionConfig":
        """
        Create a PerceptionConfig from a registered perception class.

        Args:
            name: The registered name of the perception class.

        Returns:
            A new PerceptionConfig populated from registry metadata.

        Raises:
            ValueError: If the perception class is not found in the registry.
        """
        # Try to find the class
        perception_cls = registry.get(name)
        if perception_cls is None:
            # Try with perception prefix
            perception_cls = registry.get(f"perception.{name}")
        if perception_cls is None:
            raise ValueError(f"Perception config '{name}' not found in registry")

        # Check if it's a perception config
        if not registry.has_tag(perception_cls, TAG_PERCEPTION):
            raise ValueError(f"'{name}' is not a registered perception config")

        # Get metadata
        sense_str = registry.get_metadata(perception_cls, "sense")
        sense_enum = registry.get_metadata(perception_cls, "sense_enum")
        range_val = registry.get_metadata(perception_cls, "range")
        fov_val = registry.get_metadata(perception_cls, "fov")
        decay_val = registry.get_metadata(perception_cls, "decay_time")
        desc = registry.get_metadata(perception_cls, "description") or ""

        return cls(
            name=name,
            sense=sense_str,
            sense_enum=sense_enum,
            range=range_val,
            fov=fov_val,
            decay_time=decay_val,
            description=desc,
            source_class=perception_cls,
        )

    @classmethod
    def from_class(cls, perception_cls: type) -> "PerceptionConfig":
        """
        Create a PerceptionConfig from a perception class with @perception decorator.

        Args:
            perception_cls: The decorated perception class.

        Returns:
            A new PerceptionConfig populated from class attributes.

        Raises:
            ValueError: If the class is not a valid perception class.
        """
        if not getattr(perception_cls, "_perception", False):
            raise ValueError(f"Class '{perception_cls.__name__}' is not a perception config")

        return cls(
            name=perception_cls.__name__,
            sense=perception_cls._perception_sense,
            sense_enum=perception_cls._perception_sense_enum,
            range=perception_cls._perception_range,
            fov=perception_cls._perception_fov,
            decay_time=perception_cls._perception_decay_time,
            description=perception_cls._perception_description,
            source_class=perception_cls,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "sense": self.sense,
            "sense_enum": self.sense_enum.value,
            "range": self.range,
            "fov": self.fov,
            "decay_time": self.decay_time,
            "description": self.description,
        }


def create_perception_config_from_registry(
    name: str,
    **overrides: Any,
) -> PerceptionConfig:
    """
    Create a PerceptionConfig instance from the registry with optional overrides.

    Args:
        name: The registered name of the perception config.
        **overrides: Override values for range, fov, decay_time, etc.

    Returns:
        A new PerceptionConfig instance.

    Raises:
        ValueError: If the perception config is not found.
    """
    config = PerceptionConfig.from_registry(name)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


@dataclass
class SenseConfig:
    """
    Individual sense configuration data class.

    Represents a single sense capability that can be attached
    to an AI agent's perception system.
    """

    name: str
    sense_type: str
    sense_type_enum: PerceptionSense
    range: float
    fov: float
    decay_time: float
    description: str = ""
    source_class: Optional[type] = None

    @classmethod
    def from_registry(cls, name: str) -> "SenseConfig":
        """
        Create a SenseConfig from a registered sense class.

        Args:
            name: The registered name of the sense class.

        Returns:
            A new SenseConfig populated from registry metadata.

        Raises:
            ValueError: If the sense class is not found in the registry.
        """
        # Try to find the class
        sense_cls = registry.get(name)
        if sense_cls is None:
            # Try with sense prefix
            sense_cls = registry.get(f"sense.{name}")
        if sense_cls is None:
            raise ValueError(f"Sense config '{name}' not found in registry")

        # Check if it's a sense config
        if not registry.has_tag(sense_cls, TAG_SENSE):
            raise ValueError(f"'{name}' is not a registered sense config")

        # Get metadata
        sense_type = registry.get_metadata(sense_cls, "sense_type")
        sense_type_enum = registry.get_metadata(sense_cls, "sense_type_enum")
        range_val = registry.get_metadata(sense_cls, "range")
        fov_val = registry.get_metadata(sense_cls, "fov")
        decay_val = registry.get_metadata(sense_cls, "decay_time")
        desc = registry.get_metadata(sense_cls, "description") or ""

        return cls(
            name=name,
            sense_type=sense_type,
            sense_type_enum=sense_type_enum,
            range=range_val,
            fov=fov_val,
            decay_time=decay_val,
            description=desc,
            source_class=sense_cls,
        )

    @classmethod
    def from_class(cls, sense_cls: type) -> "SenseConfig":
        """
        Create a SenseConfig from a sense class with @sense decorator.

        Args:
            sense_cls: The decorated sense class.

        Returns:
            A new SenseConfig populated from class attributes.

        Raises:
            ValueError: If the class is not a valid sense class.
        """
        if not getattr(sense_cls, "_sense", False):
            raise ValueError(f"Class '{sense_cls.__name__}' is not a sense config")

        return cls(
            name=sense_cls.__name__,
            sense_type=sense_cls._sense_type,
            sense_type_enum=sense_cls._sense_type_enum,
            range=sense_cls._sense_range,
            fov=sense_cls._sense_fov,
            decay_time=sense_cls._sense_decay_time,
            description=sense_cls._sense_description,
            source_class=sense_cls,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "sense_type": self.sense_type,
            "sense_type_enum": self.sense_type_enum.value,
            "range": self.range,
            "fov": self.fov,
            "decay_time": self.decay_time,
            "description": self.description,
        }


def create_sense_config_from_registry(
    name: str,
    **overrides: Any,
) -> SenseConfig:
    """
    Create a SenseConfig instance from the registry with optional overrides.

    Args:
        name: The registered name of the sense config.
        **overrides: Override values for range, fov, decay_time, etc.

    Returns:
        A new SenseConfig instance.

    Raises:
        ValueError: If the sense config is not found.
    """
    config = SenseConfig.from_registry(name)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Decorators
    "perception",
    "sense",
    # Query helpers
    "get_all_perception_configs",
    "get_perception_configs_by_sense",
    "get_all_sense_configs",
    "get_sense_configs_by_type",
    "get_perception_by_name",
    "get_sense_by_name",
    # Factory functions / Data classes
    "PerceptionConfig",
    "SenseConfig",
    "create_perception_config_from_registry",
    "create_sense_config_from_registry",
    # Constants
    "TAG_PERCEPTION",
    "TAG_SENSE",
    "VALID_SENSE_TYPES",
    "SENSE_TYPE_MAP",
    "DEFAULT_RANGES",
    "DEFAULT_DECAY_TIMES",
]
