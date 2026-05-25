"""Console Variables (CVars) - Typed configuration variables with change notifications.

This module provides a robust CVar system for runtime configuration with:
- Type-safe generic CVars (int, float, bool, str)
- CVar flags (READONLY, CHEAT, CONFIG, SCALABILITY)
- Change callback registration
- Singleton registry for global CVar management

Example:
    >>> r_vsync = CVar("r.VSync", default=1, flags=CVarFlags.CONFIG)
    >>> r_vsync.on_change(lambda old, new: print(f"VSync: {old} -> {new}"))
    >>> r_vsync.value = 0  # Triggers callback
    VSync: 1 -> 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)


class CVarFlags(Flag):
    """Flags that control CVar behavior and access.

    Attributes:
        NONE: No special flags.
        READONLY: CVar cannot be modified at runtime.
        CHEAT: CVar is only accessible when cheats are enabled.
        CONFIG: CVar is saved to/loaded from configuration files.
        SCALABILITY: CVar is affected by scalability settings.
    """
    NONE = 0
    READONLY = auto()
    CHEAT = auto()
    CONFIG = auto()
    SCALABILITY = auto()


# Supported CVar value types
CVarValue = TypeVar("CVarValue", int, float, bool, str)

# Type for change callbacks
ChangeCallback = Callable[[Any, Any], None]


class CVarTypeError(TypeError):
    """Raised when a CVar value doesn't match the expected type."""
    pass


class CVarReadOnlyError(Exception):
    """Raised when attempting to modify a readonly CVar."""
    pass


class CVarCheatError(Exception):
    """Raised when attempting to access a cheat CVar without cheats enabled."""
    pass


class CVarBoundsError(ValueError):
    """Raised when a CVar value is outside the specified bounds."""
    pass


@dataclass
class CVar(Generic[CVarValue]):
    """A typed console variable with flags and change notifications.

    CVars are configurable values that can be modified at runtime through
    the console. They support type validation, flags for access control,
    bounds checking, and callbacks for change notifications.

    Attributes:
        name: Unique identifier for the CVar (e.g., "r.VSync").
        default: The default value used when resetting.
        flags: Bitflags controlling access and behavior.
        description: Human-readable description of the CVar.
        min_value: Optional minimum value (for numeric types).
        max_value: Optional maximum value (for numeric types).

    Example:
        >>> shadow_quality = CVar[int](
        ...     name="r.ShadowQuality",
        ...     default=3,
        ...     flags=CVarFlags.CONFIG | CVarFlags.SCALABILITY,
        ...     description="Shadow quality level (0-4)",
        ...     min_value=0,
        ...     max_value=4
        ... )
        >>> shadow_quality.value = 2
        >>> shadow_quality.value
        2
    """
    name: str
    default: CVarValue
    flags: CVarFlags = CVarFlags.NONE
    description: str = ""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    # Internal state
    _value: Optional[CVarValue] = field(default=None, repr=False)
    _callbacks: List[ChangeCallback] = field(default_factory=list, repr=False)
    _value_type: Optional[Type] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the CVar and register with the global registry."""
        # Infer type from default value
        self._value_type = type(self.default)

        # Validate that the default is a supported type
        if self._value_type not in (int, float, bool, str):
            raise CVarTypeError(
                f"CVar '{self.name}' has unsupported type {self._value_type.__name__}. "
                f"Supported types: int, float, bool, str"
            )

        # Validate bounds are only set for numeric types
        if (self.min_value is not None or self.max_value is not None):
            if self._value_type not in (int, float):
                raise CVarTypeError(
                    f"CVar '{self.name}' has bounds but is not a numeric type"
                )

        # Validate default is within bounds
        if self.min_value is not None and self.default < self.min_value:
            raise CVarBoundsError(
                f"CVar '{self.name}' default value {self.default} is below minimum {self.min_value}"
            )
        if self.max_value is not None and self.default > self.max_value:
            raise CVarBoundsError(
                f"CVar '{self.name}' default value {self.default} is above maximum {self.max_value}"
            )

        # Initialize value to default
        self._value = self.default

        # Register with global registry
        CVarRegistry.instance().register(self)

    @property
    def value(self) -> CVarValue:
        """Get the current CVar value.

        Returns:
            The current value of the CVar.

        Raises:
            CVarCheatError: If the CVar has CHEAT flag and cheats are disabled.
        """
        if CVarFlags.CHEAT in self.flags and not CVarRegistry.instance().cheats_enabled:
            raise CVarCheatError(
                f"CVar '{self.name}' is a cheat variable and cheats are disabled"
            )
        return self._value  # type: ignore

    @value.setter
    def value(self, new_value: CVarValue) -> None:
        """Set the CVar value with type validation and change notification.

        Args:
            new_value: The new value to set.

        Raises:
            CVarReadOnlyError: If the CVar has READONLY flag.
            CVarCheatError: If the CVar has CHEAT flag and cheats are disabled.
            CVarTypeError: If the new value doesn't match the expected type.
        """
        # Check readonly
        if CVarFlags.READONLY in self.flags:
            raise CVarReadOnlyError(
                f"CVar '{self.name}' is readonly and cannot be modified"
            )

        # Check cheat access
        if CVarFlags.CHEAT in self.flags and not CVarRegistry.instance().cheats_enabled:
            raise CVarCheatError(
                f"CVar '{self.name}' is a cheat variable and cheats are disabled"
            )

        # Type coercion and validation
        validated_value = self._validate_and_coerce(new_value)

        # Bounds checking for numeric types
        validated_value = self._check_bounds(validated_value)

        # Store old value for callbacks
        old_value = self._value

        # Only update and notify if value actually changed
        if old_value != validated_value:
            self._value = validated_value
            self._notify_change(old_value, validated_value)

    def _validate_and_coerce(self, value: Any) -> CVarValue:
        """Validate and coerce a value to the CVar's type.

        Args:
            value: The value to validate and coerce.

        Returns:
            The validated and coerced value.

        Raises:
            CVarTypeError: If the value cannot be coerced to the expected type.
        """
        # Handle string parsing for console input
        if isinstance(value, str) and self._value_type != str:
            try:
                if self._value_type == bool:
                    # Handle boolean string values
                    lower = value.lower()
                    if lower in ("true", "1", "yes", "on"):
                        return True  # type: ignore
                    elif lower in ("false", "0", "no", "off"):
                        return False  # type: ignore
                    else:
                        raise ValueError(f"Cannot convert '{value}' to bool")
                elif self._value_type == int:
                    return int(value)  # type: ignore
                elif self._value_type == float:
                    return float(value)  # type: ignore
            except ValueError as e:
                raise CVarTypeError(
                    f"Cannot convert '{value}' to {self._value_type.__name__} "
                    f"for CVar '{self.name}': {e}"
                )

        # Direct type check
        if not isinstance(value, self._value_type):
            # Allow int -> float coercion
            if self._value_type == float and isinstance(value, int):
                return float(value)  # type: ignore

            raise CVarTypeError(
                f"CVar '{self.name}' expects {self._value_type.__name__}, "
                f"got {type(value).__name__}"
            )

        return value  # type: ignore

    def _check_bounds(self, value: CVarValue) -> CVarValue:
        """Check if a numeric value is within the configured bounds.

        Args:
            value: The value to check.

        Returns:
            The value if within bounds.

        Raises:
            CVarBoundsError: If the value is outside the configured bounds.
        """
        if self._value_type not in (int, float):
            return value

        if self.min_value is not None and value < self.min_value:
            raise CVarBoundsError(
                f"CVar '{self.name}' value {value} is below minimum {self.min_value}"
            )
        if self.max_value is not None and value > self.max_value:
            raise CVarBoundsError(
                f"CVar '{self.name}' value {value} is above maximum {self.max_value}"
            )

        return value

    def _notify_change(self, old_value: CVarValue, new_value: CVarValue) -> None:
        """Notify all registered callbacks of a value change.

        Args:
            old_value: The previous value.
            new_value: The new value.
        """
        for callback in self._callbacks:
            try:
                callback(old_value, new_value)
            except Exception as e:
                # Log but don't propagate callback errors
                import logging
                logging.getLogger("engine.debug.console.cvar").warning(
                    f"CVar '{self.name}' change callback raised exception: {e}"
                )

    def on_change(self, callback: ChangeCallback) -> None:
        """Register a callback to be called when the value changes.

        The callback receives (old_value, new_value) as arguments.

        Args:
            callback: A function taking (old_value, new_value).

        Example:
            >>> cvar = CVar("test", default=0)
            >>> cvar.on_change(lambda old, new: print(f"{old} -> {new}"))
            >>> cvar.value = 5
            0 -> 5
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def off_change(self, callback: ChangeCallback) -> None:
        """Unregister a previously registered change callback.

        Args:
            callback: The callback to remove.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def reset(self) -> None:
        """Reset the CVar to its default value.

        This will trigger change callbacks if the value differs from default.

        Raises:
            CVarReadOnlyError: If the CVar has READONLY flag.
        """
        self.value = self.default

    @property
    def is_default(self) -> bool:
        """Check if the current value equals the default.

        Returns:
            True if value equals default, False otherwise.
        """
        return self._value == self.default

    def __str__(self) -> str:
        """Return a string representation for console display."""
        flags_str = ""
        if self.flags != CVarFlags.NONE:
            flags_str = f" [{self.flags.name}]"
        return f"{self.name} = {self._value}{flags_str}"

    def get_info(self) -> Dict[str, Any]:
        """Get detailed information about this CVar.

        Returns:
            A dictionary containing CVar metadata.
        """
        return {
            "name": self.name,
            "value": self._value,
            "default": self.default,
            "type": self._value_type.__name__ if self._value_type else "unknown",
            "flags": self.flags.name if self.flags != CVarFlags.NONE else "NONE",
            "description": self.description,
            "is_default": self.is_default,
        }


class CVarRegistry:
    """Singleton registry for all console variables.

    The registry provides centralized management of CVars, including:
    - Registration and lookup by name
    - Category-based organization (via naming convention like "r.*")
    - Cheats enabled/disabled state
    - Bulk operations (list, search, reset all)

    Example:
        >>> registry = CVarRegistry.instance()
        >>> registry.get("r.VSync")
        CVar(name='r.VSync', default=1, ...)
        >>> registry.find("r.*")
        [CVar(...), CVar(...), ...]
    """

    _instance: Optional[CVarRegistry] = None

    def __init__(self) -> None:
        """Initialize the registry."""
        self._cvars: Dict[str, CVar] = {}
        self._cheats_enabled: bool = False

    @classmethod
    def instance(cls) -> CVarRegistry:
        """Get the singleton registry instance.

        Returns:
            The global CVarRegistry instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for testing)."""
        cls._instance = None

    @property
    def cheats_enabled(self) -> bool:
        """Check if cheats are currently enabled.

        Returns:
            True if cheats are enabled, False otherwise.
        """
        return self._cheats_enabled

    @cheats_enabled.setter
    def cheats_enabled(self, value: bool) -> None:
        """Enable or disable cheats.

        Args:
            value: True to enable cheats, False to disable.
        """
        self._cheats_enabled = value

    def register(self, cvar: CVar) -> None:
        """Register a CVar with the registry.

        Args:
            cvar: The CVar to register.

        Raises:
            ValueError: If a CVar with the same name already exists.
        """
        if cvar.name in self._cvars:
            raise ValueError(f"CVar '{cvar.name}' is already registered")
        self._cvars[cvar.name] = cvar

    def unregister(self, name: str) -> bool:
        """Unregister a CVar by name.

        Args:
            name: The name of the CVar to unregister.

        Returns:
            True if the CVar was found and removed, False otherwise.
        """
        if name in self._cvars:
            del self._cvars[name]
            return True
        return False

    def get(self, name: str) -> Optional[CVar]:
        """Get a CVar by name.

        Args:
            name: The name of the CVar to retrieve.

        Returns:
            The CVar if found, None otherwise.
        """
        return self._cvars.get(name)

    def find(self, pattern: str) -> List[CVar]:
        """Find CVars matching a pattern.

        Supports simple glob patterns with '*' wildcard.

        Args:
            pattern: A pattern to match (e.g., "r.*" for rendering CVars).

        Returns:
            List of CVars matching the pattern.
        """
        import fnmatch
        return [
            cvar for name, cvar in self._cvars.items()
            if fnmatch.fnmatch(name, pattern)
        ]

    def all(self) -> List[CVar]:
        """Get all registered CVars.

        Returns:
            List of all CVars, sorted by name.
        """
        return sorted(self._cvars.values(), key=lambda c: c.name)

    def categories(self) -> List[str]:
        """Get all CVar categories.

        Categories are determined by the prefix before the first '.'.

        Returns:
            Sorted list of unique category names.
        """
        cats = set()
        for name in self._cvars:
            if "." in name:
                cats.add(name.split(".")[0])
        return sorted(cats)

    def by_category(self, category: str) -> List[CVar]:
        """Get all CVars in a category.

        Args:
            category: The category prefix (e.g., "r" for "r.*").

        Returns:
            List of CVars in the category.
        """
        prefix = f"{category}."
        return [
            cvar for name, cvar in self._cvars.items()
            if name.startswith(prefix)
        ]

    def with_flags(self, flags: CVarFlags) -> List[CVar]:
        """Get all CVars with specific flags.

        Args:
            flags: The flags to match (any matching flag will include the CVar).

        Returns:
            List of CVars that have any of the specified flags.
        """
        return [
            cvar for cvar in self._cvars.values()
            if cvar.flags & flags
        ]

    def reset_all(self, include_readonly: bool = False) -> int:
        """Reset all CVars to their default values.

        Args:
            include_readonly: If True, attempt to reset readonly CVars too.

        Returns:
            The number of CVars successfully reset.
        """
        count = 0
        for cvar in self._cvars.values():
            if CVarFlags.READONLY in cvar.flags and not include_readonly:
                continue
            try:
                cvar.reset()
                count += 1
            except (CVarReadOnlyError, CVarCheatError):
                pass
        return count

    def export_config(self) -> Dict[str, Any]:
        """Export all CONFIG-flagged CVars to a dictionary.

        Returns:
            Dictionary mapping CVar names to their current values.
        """
        return {
            cvar.name: cvar._value
            for cvar in self._cvars.values()
            if CVarFlags.CONFIG in cvar.flags
        }

    def import_config(self, config: Dict[str, Any]) -> int:
        """Import CVar values from a configuration dictionary.

        Args:
            config: Dictionary mapping CVar names to values.

        Returns:
            The number of CVars successfully updated.
        """
        count = 0
        for name, value in config.items():
            cvar = self.get(name)
            if cvar is not None:
                try:
                    cvar.value = value
                    count += 1
                except (CVarReadOnlyError, CVarCheatError, CVarTypeError):
                    pass
        return count

    def __len__(self) -> int:
        """Return the number of registered CVars."""
        return len(self._cvars)

    def __contains__(self, name: str) -> bool:
        """Check if a CVar is registered."""
        return name in self._cvars

    def __iter__(self):
        """Iterate over CVar names."""
        return iter(self._cvars)
