"""T-CC-1.8: DataBoundDescriptor for runtime data binding.

Provides descriptors that bind class attributes to external data sources,
enabling automatic synchronization between objects and configuration/data.
"""
from __future__ import annotations

import json
import threading
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)


T = TypeVar('T')


class BindingMode(Enum):
    """Data binding direction modes."""
    ONE_WAY = auto()  # Source -> Instance only
    TWO_WAY = auto()  # Source <-> Instance
    ONE_TIME = auto()  # Initial load only


class BindingState(Enum):
    """Current state of a binding."""
    UNBOUND = auto()
    BOUND = auto()
    ERROR = auto()
    STALE = auto()


@dataclass
class BindingContext:
    """Context for a data binding."""
    source_path: str
    key_path: str
    mode: BindingMode = BindingMode.ONE_WAY
    default: Any = None
    transform: Optional[Callable[[Any], Any]] = None
    inverse_transform: Optional[Callable[[Any], Any]] = None
    validate: Optional[Callable[[Any], bool]] = None


@dataclass
class BindingError:
    """Represents a binding error."""
    source: str
    key: str
    message: str
    timestamp: float = field(default_factory=lambda: __import__('time').time())

    def __str__(self) -> str:
        return f"[{self.source}:{self.key}] {self.message}"


class DataSource:
    """Abstract base for data sources."""

    def get(self, key_path: str) -> Tuple[Any, bool]:
        """Get value at key path. Returns (value, found)."""
        raise NotImplementedError

    def set(self, key_path: str, value: Any) -> bool:
        """Set value at key path. Returns success."""
        raise NotImplementedError

    def watch(self, key_path: str, callback: Callable[[Any], None]) -> None:
        """Watch for changes to a key path."""
        raise NotImplementedError

    def unwatch(self, key_path: str, callback: Callable[[Any], None]) -> None:
        """Stop watching a key path."""
        raise NotImplementedError


class DictDataSource(DataSource):
    """Data source backed by a dictionary."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}
        self._watchers: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock = threading.RLock()

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def get(self, key_path: str) -> Tuple[Any, bool]:
        """Get value at key path (dot-separated)."""
        with self._lock:
            parts = key_path.split('.') if key_path else []
            current = self._data

            for part in parts:
                if isinstance(current, dict):
                    if part not in current:
                        return None, False
                    current = current[part]
                elif isinstance(current, list):
                    try:
                        idx = int(part)
                        if 0 <= idx < len(current):
                            current = current[idx]
                        else:
                            return None, False
                    except ValueError:
                        return None, False
                else:
                    return None, False

            return current, True

    def set(self, key_path: str, value: Any) -> bool:
        """Set value at key path."""
        with self._lock:
            parts = key_path.split('.') if key_path else []
            if not parts:
                return False

            current = self._data
            for part in parts[:-1]:
                if isinstance(current, dict):
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                else:
                    return False

            if isinstance(current, dict):
                current[parts[-1]] = value
                self._notify(key_path, value)
                return True

            return False

    def _notify(self, key_path: str, value: Any) -> None:
        """Notify watchers of a change."""
        with self._lock:
            callbacks = self._watchers.get(key_path, []).copy()

        for callback in callbacks:
            try:
                callback(value)
            except Exception:
                pass

    def watch(self, key_path: str, callback: Callable[[Any], None]) -> None:
        """Watch for changes to a key path."""
        with self._lock:
            if key_path not in self._watchers:
                self._watchers[key_path] = []
            if callback not in self._watchers[key_path]:
                self._watchers[key_path].append(callback)

    def unwatch(self, key_path: str, callback: Callable[[Any], None]) -> None:
        """Stop watching a key path."""
        with self._lock:
            if key_path in self._watchers and callback in self._watchers[key_path]:
                self._watchers[key_path].remove(callback)

    def update(self, data: Dict[str, Any]) -> None:
        """Update data and notify all relevant watchers."""
        with self._lock:
            old_data = self._data.copy()
            self._data.update(data)

        for key_path in self._watchers:
            new_val, found = self.get(key_path)
            if found:
                self._notify(key_path, new_val)


class JsonFileSource(DataSource):
    """Data source backed by a JSON file."""

    def __init__(self, path: Union[str, Path]):
        self._path = Path(path)
        self._data: Dict[str, Any] = {}
        self._watchers: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock = threading.RLock()
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> bool:
        """Load data from file."""
        try:
            with self._lock:
                with open(self._path, 'r') as f:
                    self._data = json.load(f)
                self._loaded = True
            return True
        except (OSError, json.JSONDecodeError):
            return False

    def save(self) -> bool:
        """Save data to file."""
        try:
            with self._lock:
                with open(self._path, 'w') as f:
                    json.dump(self._data, f, indent=2)
            return True
        except OSError:
            return False

    def get(self, key_path: str) -> Tuple[Any, bool]:
        """Get value at key path."""
        if not self._loaded:
            self.load()

        with self._lock:
            parts = key_path.split('.') if key_path else []
            current = self._data

            for part in parts:
                if isinstance(current, dict):
                    if part not in current:
                        return None, False
                    current = current[part]
                else:
                    return None, False

            return current, True

    def set(self, key_path: str, value: Any) -> bool:
        """Set value at key path and save."""
        with self._lock:
            parts = key_path.split('.') if key_path else []
            if not parts:
                return False

            current = self._data
            for part in parts[:-1]:
                if isinstance(current, dict):
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                else:
                    return False

            if isinstance(current, dict):
                current[parts[-1]] = value
                self.save()
                self._notify(key_path, value)
                return True

            return False

    def _notify(self, key_path: str, value: Any) -> None:
        """Notify watchers."""
        with self._lock:
            callbacks = self._watchers.get(key_path, []).copy()

        for callback in callbacks:
            try:
                callback(value)
            except Exception:
                pass

    def watch(self, key_path: str, callback: Callable[[Any], None]) -> None:
        with self._lock:
            if key_path not in self._watchers:
                self._watchers[key_path] = []
            if callback not in self._watchers[key_path]:
                self._watchers[key_path].append(callback)

    def unwatch(self, key_path: str, callback: Callable[[Any], None]) -> None:
        with self._lock:
            if key_path in self._watchers and callback in self._watchers[key_path]:
                self._watchers[key_path].remove(callback)

    def reload(self) -> bool:
        """Reload from file and notify watchers."""
        success = self.load()
        if success:
            with self._lock:
                for key_path in self._watchers:
                    value, found = self.get(key_path)
                    if found:
                        self._notify(key_path, value)
        return success


class DataSourceRegistry:
    """Registry for named data sources."""

    _instance: Optional["DataSourceRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DataSourceRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sources: Dict[str, DataSource] = {}
                cls._instance._source_lock = threading.RLock()
            return cls._instance

    @classmethod
    def get_instance(cls) -> "DataSourceRegistry":
        """Get the singleton instance."""
        return cls()

    def register(self, name: str, source: DataSource) -> None:
        """Register a data source."""
        with self._source_lock:
            self._sources[name] = source

    def unregister(self, name: str) -> bool:
        """Unregister a data source."""
        with self._source_lock:
            if name in self._sources:
                del self._sources[name]
                return True
            return False

    def get(self, name: str) -> Optional[DataSource]:
        """Get a data source by name."""
        with self._source_lock:
            return self._sources.get(name)

    def list_sources(self) -> List[str]:
        """List all registered source names."""
        with self._source_lock:
            return list(self._sources.keys())

    def clear(self) -> None:
        """Clear all registered sources."""
        with self._source_lock:
            self._sources.clear()


class DataBoundDescriptor(Generic[T]):
    """Descriptor that binds an attribute to a data source."""

    def __init__(
        self,
        source_name: str,
        key_path: str,
        *,
        mode: BindingMode = BindingMode.ONE_WAY,
        default: T = None,
        transform: Optional[Callable[[Any], T]] = None,
        inverse_transform: Optional[Callable[[T], Any]] = None,
        validate: Optional[Callable[[T], bool]] = None,
    ):
        self.source_name = source_name
        self.key_path = key_path
        self.mode = mode
        self.default = default
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.validate = validate
        self._name: Optional[str] = None
        self._instances: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, instance: Any, owner: type) -> T:
        if instance is None:
            return self

        with self._lock:
            if instance not in self._instances:
                self._bind_instance(instance)

            return self._instances[instance].get('value', self.default)

    def __set__(self, instance: Any, value: T) -> None:
        if self.mode == BindingMode.ONE_TIME:
            return

        if self.validate and not self.validate(value):
            raise ValueError(f"Validation failed for {self._name}")

        with self._lock:
            if instance not in self._instances:
                self._bind_instance(instance)

            self._instances[instance]['value'] = value

            if self.mode == BindingMode.TWO_WAY:
                self._write_to_source(value)

    def _bind_instance(self, instance: Any) -> None:
        """Initialize binding for an instance."""
        registry = DataSourceRegistry.get_instance()
        source = registry.get(self.source_name)

        self._instances[instance] = {
            'value': self.default,
            'state': BindingState.UNBOUND,
            'source': source,
        }

        if source:
            value, found = source.get(self.key_path)
            if found:
                if self.transform:
                    value = self.transform(value)
                self._instances[instance]['value'] = value
                self._instances[instance]['state'] = BindingState.BOUND
            else:
                self._instances[instance]['state'] = BindingState.ERROR

            if self.mode in (BindingMode.ONE_WAY, BindingMode.TWO_WAY):
                def on_change(new_value: Any) -> None:
                    if self.transform:
                        new_value = self.transform(new_value)
                    with self._lock:
                        if instance in self._instances:
                            self._instances[instance]['value'] = new_value

                source.watch(self.key_path, on_change)
                self._instances[instance]['callback'] = on_change

    def _write_to_source(self, value: T) -> None:
        """Write value back to source."""
        registry = DataSourceRegistry.get_instance()
        source = registry.get(self.source_name)

        if source:
            write_value = value
            if self.inverse_transform:
                write_value = self.inverse_transform(value)
            source.set(self.key_path, write_value)

    def get_state(self, instance: Any) -> BindingState:
        """Get binding state for an instance."""
        with self._lock:
            if instance in self._instances:
                return self._instances[instance].get('state', BindingState.UNBOUND)
        return BindingState.UNBOUND

    def refresh(self, instance: Any) -> bool:
        """Refresh value from source."""
        registry = DataSourceRegistry.get_instance()
        source = registry.get(self.source_name)

        if not source:
            return False

        value, found = source.get(self.key_path)
        if found:
            if self.transform:
                value = self.transform(value)
            with self._lock:
                if instance in self._instances:
                    self._instances[instance]['value'] = value
                    self._instances[instance]['state'] = BindingState.BOUND
            return True

        return False


def bound(
    source: str,
    key: str,
    *,
    mode: BindingMode = BindingMode.ONE_WAY,
    default: Any = None,
    transform: Optional[Callable[[Any], Any]] = None,
    inverse_transform: Optional[Callable[[Any], Any]] = None,
    validate: Optional[Callable[[Any], bool]] = None,
) -> DataBoundDescriptor:
    """Create a data-bound descriptor."""
    return DataBoundDescriptor(
        source,
        key,
        mode=mode,
        default=default,
        transform=transform,
        inverse_transform=inverse_transform,
        validate=validate,
    )


class BindingManager:
    """Manages data bindings for a class or instance."""

    def __init__(self, target: Any):
        self._target = target
        self._descriptors: Dict[str, DataBoundDescriptor] = {}
        self._discover_bindings()

    def _discover_bindings(self) -> None:
        """Find all DataBoundDescriptor attributes."""
        cls = type(self._target) if not isinstance(self._target, type) else self._target
        for name in dir(cls):
            try:
                attr = getattr(cls, name)
                if isinstance(attr, DataBoundDescriptor):
                    self._descriptors[name] = attr
            except AttributeError:
                pass

    @property
    def bindings(self) -> Dict[str, DataBoundDescriptor]:
        """Get all bindings."""
        return self._descriptors.copy()

    def refresh_all(self) -> Dict[str, bool]:
        """Refresh all bindings. Returns success status per binding."""
        results = {}
        for name, desc in self._descriptors.items():
            results[name] = desc.refresh(self._target)
        return results

    def get_states(self) -> Dict[str, BindingState]:
        """Get state of all bindings."""
        return {
            name: desc.get_state(self._target)
            for name, desc in self._descriptors.items()
        }


def with_bindings(cls: Type[T]) -> Type[T]:
    """Class decorator that adds binding management."""

    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._binding_manager = BindingManager(self)

    cls.__init__ = new_init

    def get_binding_manager(self) -> BindingManager:
        if not hasattr(self, '_binding_manager'):
            self._binding_manager = BindingManager(self)
        return self._binding_manager

    cls.bindings = property(get_binding_manager)

    return cls


class ComputedBinding(Generic[T]):
    """A binding computed from multiple source values."""

    def __init__(
        self,
        compute: Callable[..., T],
        *dependencies: DataBoundDescriptor,
        default: T = None,
    ):
        self.compute = compute
        self.dependencies = dependencies
        self.default = default
        self._name: Optional[str] = None
        self._cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, instance: Any, owner: type) -> T:
        if instance is None:
            return self

        try:
            values = []
            for dep in self.dependencies:
                val = dep.__get__(instance, owner)
                values.append(val)
            return self.compute(*values)
        except Exception:
            return self.default

    def __set__(self, instance: Any, value: T) -> None:
        pass  # Computed bindings are read-only


def computed(
    *dependencies: DataBoundDescriptor,
    default: Any = None,
) -> Callable[[Callable[..., T]], ComputedBinding[T]]:
    """Decorator to create a computed binding."""
    def decorator(func: Callable[..., T]) -> ComputedBinding[T]:
        return ComputedBinding(func, *dependencies, default=default)
    return decorator
