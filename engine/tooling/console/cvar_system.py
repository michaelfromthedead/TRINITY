"""Configuration Variable (CVar) system for runtime game configuration.

Provides typed configuration variables with validation, callbacks, and persistence.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, Union


class CVarType(Enum):
    """Types of configuration variables."""
    INT = auto()
    FLOAT = auto()
    BOOL = auto()
    STRING = auto()
    ENUM = auto()


class CVarFlags(Flag):
    """Flags controlling CVar behavior."""
    NONE = 0
    READONLY = auto()  # Cannot be modified at runtime
    CHEAT = auto()  # Requires cheat mode to modify
    ARCHIVE = auto()  # Persisted to config file
    REPLICATED = auto()  # Replicated in multiplayer
    HIDDEN = auto()  # Not shown in console autocomplete
    DEVELOPER = auto()  # Only visible in developer mode
    SERVER_ONLY = auto()  # Only modifiable on server
    USER_INFO = auto()  # Part of user info sent to server


T = TypeVar("T")


@dataclass
class CVarChangeEvent(Generic[T]):
    """Event emitted when a CVar value changes."""
    cvar_name: str
    old_value: T
    new_value: T
    source: str = "code"  # "code", "console", "config", "network"


class CVar(ABC, Generic[T]):
    """Base class for configuration variables.

    CVars provide typed, validated configuration values with callbacks
    for change notification.
    """
    __slots__ = (
        '_name', '_value', '_default', '_description', '_flags',
        '_callbacks', '_lock', '_category'
    )

    def __init__(
        self,
        name: str,
        default: T,
        description: str = "",
        flags: CVarFlags = CVarFlags.NONE,
        category: str = "general"
    ):
        """Initialize a CVar.

        Args:
            name: Unique identifier for the CVar
            default: Default value
            description: Human-readable description
            flags: Behavior flags
            category: Category for organization
        """
        if not name or not name.strip():
            raise ValueError("CVar name cannot be empty")

        self._name = name.strip()
        self._default = default
        self._value = default
        self._description = description
        self._flags = flags
        self._category = category
        self._callbacks: list[Callable[[CVarChangeEvent[T]], None]] = []
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        """Get the CVar name."""
        return self._name

    @property
    def value(self) -> T:
        """Get the current value."""
        with self._lock:
            return self._value

    @value.setter
    def value(self, new_value: T) -> None:
        """Set the value with validation."""
        self.set(new_value)

    @property
    def default(self) -> T:
        """Get the default value."""
        return self._default

    @property
    def description(self) -> str:
        """Get the description."""
        return self._description

    @property
    def flags(self) -> CVarFlags:
        """Get the flags."""
        return self._flags

    @property
    def category(self) -> str:
        """Get the category."""
        return self._category

    @property
    @abstractmethod
    def cvar_type(self) -> CVarType:
        """Get the CVar type."""
        pass

    def set(self, new_value: T, source: str = "code") -> bool:
        """Set the value with validation and callbacks.

        Args:
            new_value: The new value to set
            source: Source of the change

        Returns:
            True if value was changed, False otherwise

        Raises:
            ValueError: If value fails validation
            PermissionError: If CVar is readonly
        """
        if CVarFlags.READONLY in self._flags:
            raise PermissionError(f"CVar '{self._name}' is readonly")

        validated = self._validate(new_value)

        with self._lock:
            if validated == self._value:
                return False

            old_value = self._value
            self._value = validated

            event = CVarChangeEvent(
                cvar_name=self._name,
                old_value=old_value,
                new_value=validated,
                source=source
            )

            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception:
                    pass  # Don't let callback errors break the set

            return True

    def reset(self) -> None:
        """Reset to default value."""
        if CVarFlags.READONLY not in self._flags:
            self.set(self._default, source="reset")

    @abstractmethod
    def _validate(self, value: Any) -> T:
        """Validate and potentially convert a value.

        Args:
            value: The value to validate

        Returns:
            The validated/converted value

        Raises:
            ValueError: If validation fails
        """
        pass

    @abstractmethod
    def parse(self, string_value: str) -> T:
        """Parse a string representation into the typed value.

        Args:
            string_value: String to parse

        Returns:
            The parsed value

        Raises:
            ValueError: If parsing fails
        """
        pass

    def add_callback(self, callback: Callable[[CVarChangeEvent[T]], None]) -> None:
        """Add a change callback.

        Args:
            callback: Function to call when value changes
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[CVarChangeEvent[T]], None]) -> None:
        """Remove a change callback.

        Args:
            callback: Function to remove
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    def __str__(self) -> str:
        """String representation showing name and value."""
        return f"{self._name} = {self._value}"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"{self.__class__.__name__}(name={self._name!r}, "
            f"value={self._value!r}, default={self._default!r})"
        )


class IntCVar(CVar[int]):
    """Integer configuration variable with optional range."""
    __slots__ = ('_min', '_max')

    def __init__(
        self,
        name: str,
        default: int,
        description: str = "",
        flags: CVarFlags = CVarFlags.NONE,
        category: str = "general",
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ):
        """Initialize an integer CVar.

        Args:
            name: Unique identifier
            default: Default value
            description: Human-readable description
            flags: Behavior flags
            category: Category for organization
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
        """
        self._min = min_value
        self._max = max_value

        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("min_value cannot be greater than max_value")

        super().__init__(name, default, description, flags, category)

    @property
    def cvar_type(self) -> CVarType:
        return CVarType.INT

    @property
    def min_value(self) -> Optional[int]:
        """Get the minimum allowed value."""
        return self._min

    @property
    def max_value(self) -> Optional[int]:
        """Get the maximum allowed value."""
        return self._max

    def _validate(self, value: Any) -> int:
        """Validate and convert to integer."""
        if isinstance(value, bool):
            raise ValueError(f"Boolean not allowed for integer CVar '{self._name}'")

        try:
            int_value = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot convert '{value}' to integer: {e}")

        if self._min is not None and int_value < self._min:
            raise ValueError(
                f"Value {int_value} below minimum {self._min} for '{self._name}'"
            )
        if self._max is not None and int_value > self._max:
            raise ValueError(
                f"Value {int_value} above maximum {self._max} for '{self._name}'"
            )

        return int_value

    def parse(self, string_value: str) -> int:
        """Parse string to integer."""
        return self._validate(string_value)


class FloatCVar(CVar[float]):
    """Floating-point configuration variable with optional range."""
    __slots__ = ('_min', '_max', '_precision')

    def __init__(
        self,
        name: str,
        default: float,
        description: str = "",
        flags: CVarFlags = CVarFlags.NONE,
        category: str = "general",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        precision: Optional[int] = None
    ):
        """Initialize a float CVar.

        Args:
            name: Unique identifier
            default: Default value
            description: Human-readable description
            flags: Behavior flags
            category: Category for organization
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            precision: Decimal places to round to
        """
        self._min = min_value
        self._max = max_value
        self._precision = precision

        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("min_value cannot be greater than max_value")

        super().__init__(name, default, description, flags, category)

    @property
    def cvar_type(self) -> CVarType:
        return CVarType.FLOAT

    @property
    def min_value(self) -> Optional[float]:
        """Get the minimum allowed value."""
        return self._min

    @property
    def max_value(self) -> Optional[float]:
        """Get the maximum allowed value."""
        return self._max

    @property
    def precision(self) -> Optional[int]:
        """Get the precision (decimal places)."""
        return self._precision

    def _validate(self, value: Any) -> float:
        """Validate and convert to float."""
        try:
            float_value = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot convert '{value}' to float: {e}")

        if self._precision is not None:
            float_value = round(float_value, self._precision)

        if self._min is not None and float_value < self._min:
            raise ValueError(
                f"Value {float_value} below minimum {self._min} for '{self._name}'"
            )
        if self._max is not None and float_value > self._max:
            raise ValueError(
                f"Value {float_value} above maximum {self._max} for '{self._name}'"
            )

        return float_value

    def parse(self, string_value: str) -> float:
        """Parse string to float."""
        return self._validate(string_value)


class BoolCVar(CVar[bool]):
    """Boolean configuration variable."""
    __slots__ = ()

    TRUE_VALUES = frozenset({"true", "1", "yes", "on", "enabled"})
    FALSE_VALUES = frozenset({"false", "0", "no", "off", "disabled"})

    @property
    def cvar_type(self) -> CVarType:
        return CVarType.BOOL

    def _validate(self, value: Any) -> bool:
        """Validate and convert to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lower = value.lower().strip()
            if lower in self.TRUE_VALUES:
                return True
            if lower in self.FALSE_VALUES:
                return False
            raise ValueError(f"Cannot convert '{value}' to boolean")
        raise ValueError(f"Cannot convert '{value}' to boolean")

    def parse(self, string_value: str) -> bool:
        """Parse string to boolean."""
        return self._validate(string_value)

    def toggle(self) -> bool:
        """Toggle the boolean value.

        Returns:
            The new value after toggling
        """
        new_value = not self.value
        self.set(new_value)
        return new_value


class StringCVar(CVar[str]):
    """String configuration variable with optional validation."""
    __slots__ = ('_max_length', '_allowed_values', '_pattern')

    def __init__(
        self,
        name: str,
        default: str,
        description: str = "",
        flags: CVarFlags = CVarFlags.NONE,
        category: str = "general",
        max_length: Optional[int] = None,
        allowed_values: Optional[list[str]] = None,
        pattern: Optional[str] = None
    ):
        """Initialize a string CVar.

        Args:
            name: Unique identifier
            default: Default value
            description: Human-readable description
            flags: Behavior flags
            category: Category for organization
            max_length: Maximum string length
            allowed_values: List of allowed values
            pattern: Regex pattern for validation
        """
        self._max_length = max_length
        self._allowed_values = frozenset(allowed_values) if allowed_values else None
        self._pattern = pattern

        super().__init__(name, default, description, flags, category)

    @property
    def cvar_type(self) -> CVarType:
        return CVarType.STRING

    @property
    def max_length(self) -> Optional[int]:
        """Get the maximum length."""
        return self._max_length

    @property
    def allowed_values(self) -> Optional[frozenset[str]]:
        """Get the allowed values set."""
        return self._allowed_values

    def _validate(self, value: Any) -> str:
        """Validate and convert to string."""
        str_value = str(value)

        if self._max_length is not None and len(str_value) > self._max_length:
            raise ValueError(
                f"String length {len(str_value)} exceeds maximum "
                f"{self._max_length} for '{self._name}'"
            )

        if self._allowed_values is not None and str_value not in self._allowed_values:
            raise ValueError(
                f"Value '{str_value}' not in allowed values for '{self._name}'"
            )

        if self._pattern is not None:
            import re
            if not re.match(self._pattern, str_value):
                raise ValueError(
                    f"Value '{str_value}' does not match pattern for '{self._name}'"
                )

        return str_value

    def parse(self, string_value: str) -> str:
        """Parse string (no conversion needed)."""
        return self._validate(string_value)


class EnumCVar(CVar[str]):
    """Enumeration configuration variable."""
    __slots__ = ('_enum_values', '_enum_type')

    def __init__(
        self,
        name: str,
        default: Union[str, Enum],
        enum_type: type[Enum],
        description: str = "",
        flags: CVarFlags = CVarFlags.NONE,
        category: str = "general"
    ):
        """Initialize an enum CVar.

        Args:
            name: Unique identifier
            default: Default value (enum member or name string)
            enum_type: The Enum class
            description: Human-readable description
            flags: Behavior flags
            category: Category for organization
        """
        self._enum_type = enum_type
        self._enum_values = frozenset(e.name for e in enum_type)

        # Convert default to string name if it's an enum member
        if isinstance(default, Enum):
            default = default.name

        super().__init__(name, default, description, flags, category)

    @property
    def cvar_type(self) -> CVarType:
        return CVarType.ENUM

    @property
    def enum_type(self) -> type[Enum]:
        """Get the enum type."""
        return self._enum_type

    @property
    def enum_values(self) -> frozenset[str]:
        """Get the valid enum value names."""
        return self._enum_values

    def get_enum_value(self) -> Enum:
        """Get the current value as an enum member.

        Returns:
            The enum member corresponding to current value
        """
        return self._enum_type[self.value]

    def _validate(self, value: Any) -> str:
        """Validate and convert to enum name string."""
        if isinstance(value, self._enum_type):
            return value.name

        str_value = str(value).upper()

        if str_value not in self._enum_values:
            raise ValueError(
                f"Value '{value}' not a valid member of {self._enum_type.__name__}. "
                f"Valid values: {', '.join(sorted(self._enum_values))}"
            )

        return str_value

    def parse(self, string_value: str) -> str:
        """Parse string to enum name."""
        return self._validate(string_value)


class CVarRegistry:
    """Central registry for all configuration variables.

    Provides registration, lookup, persistence, and category management.
    """
    __slots__ = ('_cvars', '_categories', '_lock', '_config_path')

    _instance: Optional[CVarRegistry] = None
    _instance_lock = threading.Lock()

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the CVar registry.

        Args:
            config_path: Path for config file persistence
        """
        self._cvars: dict[str, CVar] = {}
        self._categories: dict[str, set[str]] = {}
        self._lock = threading.RLock()
        self._config_path = config_path

    @classmethod
    def get_instance(cls) -> CVarRegistry:
        """Get the singleton instance.

        Returns:
            The global CVarRegistry instance
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._instance_lock:
            cls._instance = None

    def register(self, cvar: CVar) -> None:
        """Register a CVar.

        Args:
            cvar: The CVar to register

        Raises:
            ValueError: If a CVar with the same name exists
        """
        with self._lock:
            if cvar.name in self._cvars:
                raise ValueError(f"CVar '{cvar.name}' already registered")

            self._cvars[cvar.name] = cvar

            if cvar.category not in self._categories:
                self._categories[cvar.category] = set()
            self._categories[cvar.category].add(cvar.name)

    def unregister(self, name: str) -> Optional[CVar]:
        """Unregister a CVar.

        Args:
            name: Name of the CVar to remove

        Returns:
            The removed CVar, or None if not found
        """
        with self._lock:
            cvar = self._cvars.pop(name, None)
            if cvar:
                self._categories[cvar.category].discard(name)
            return cvar

    def get(self, name: str) -> Optional[CVar]:
        """Get a CVar by name.

        Args:
            name: The CVar name

        Returns:
            The CVar if found, None otherwise
        """
        with self._lock:
            return self._cvars.get(name)

    def get_value(self, name: str, default: Any = None) -> Any:
        """Get a CVar's value by name.

        Args:
            name: The CVar name
            default: Default if CVar not found

        Returns:
            The CVar's value, or default if not found
        """
        cvar = self.get(name)
        return cvar.value if cvar else default

    def set_value(self, name: str, value: Any, source: str = "code") -> bool:
        """Set a CVar's value by name.

        Args:
            name: The CVar name
            value: The new value
            source: Source of the change

        Returns:
            True if value was changed

        Raises:
            KeyError: If CVar not found
        """
        cvar = self.get(name)
        if cvar is None:
            raise KeyError(f"CVar '{name}' not found")
        return cvar.set(value, source)

    def all_cvars(self) -> list[CVar]:
        """Get all registered CVars.

        Returns:
            List of all CVars
        """
        with self._lock:
            return list(self._cvars.values())

    def by_category(self, category: str) -> list[CVar]:
        """Get all CVars in a category.

        Args:
            category: The category name

        Returns:
            List of CVars in the category
        """
        with self._lock:
            names = self._categories.get(category, set())
            return [self._cvars[name] for name in names if name in self._cvars]

    def categories(self) -> list[str]:
        """Get all category names.

        Returns:
            List of category names
        """
        with self._lock:
            return list(self._categories.keys())

    def find(self, pattern: str) -> list[CVar]:
        """Find CVars matching a pattern.

        Args:
            pattern: Glob-style pattern (supports * wildcard)

        Returns:
            List of matching CVars
        """
        import fnmatch

        with self._lock:
            return [
                cvar for name, cvar in self._cvars.items()
                if fnmatch.fnmatch(name.lower(), pattern.lower())
            ]

    def save(self, path: Optional[Path] = None) -> None:
        """Save archivable CVars to config file.

        Args:
            path: Path to save to (uses default if not specified)
        """
        path = path or self._config_path
        if path is None:
            return

        with self._lock:
            data = {}
            for name, cvar in self._cvars.items():
                if CVarFlags.ARCHIVE in cvar.flags:
                    data[name] = cvar.value

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    def load(self, path: Optional[Path] = None) -> int:
        """Load CVars from config file.

        Args:
            path: Path to load from (uses default if not specified)

        Returns:
            Number of CVars loaded
        """
        path = path or self._config_path
        if path is None or not path.exists():
            return 0

        with open(path, 'r') as f:
            data = json.load(f)

        count = 0
        with self._lock:
            for name, value in data.items():
                cvar = self._cvars.get(name)
                if cvar and CVarFlags.ARCHIVE in cvar.flags:
                    try:
                        cvar.set(value, source="config")
                        count += 1
                    except (ValueError, PermissionError):
                        pass  # Skip invalid values

        return count

    def reset_all(self) -> None:
        """Reset all CVars to default values."""
        with self._lock:
            for cvar in self._cvars.values():
                try:
                    cvar.reset()
                except PermissionError:
                    pass  # Skip readonly CVars

    def clear(self) -> None:
        """Clear all registered CVars (for testing)."""
        with self._lock:
            self._cvars.clear()
            self._categories.clear()
