"""
Core data binding system for UI.

Provides reactive data binding between models and UI widgets following MVVM pattern.
Supports one-way, two-way, one-time, and one-way-to-source binding modes.
"""
from __future__ import annotations

import asyncio
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from .converter import IConverter, IAsyncConverter, MultiValueConverter
from .validation import (
    IValidator,
    IAsyncValidator,
    ValidationResult,
    ValidationTrigger,
    ValidationContext,
)


T = TypeVar("T")
TSource = TypeVar("TSource")
TTarget = TypeVar("TTarget")

# Default configuration constants
DEFAULT_BINDING_DELAY = 0.0

# Callback type for property changes
PropertyChangeCallback = Callable[[str, Any, Any], None]


class BindingMode(Enum):
    """Binding mode determines data flow direction."""

    ONE_WAY = auto()  # Source -> Target (default)
    TWO_WAY = auto()  # Source <-> Target
    ONE_TIME = auto()  # Source -> Target (once, at binding creation)
    ONE_WAY_TO_SOURCE = auto()  # Target -> Source


class UpdateSourceTrigger(Enum):
    """When to update the source in two-way binding."""

    ON_PROPERTY_CHANGED = auto()  # Update immediately when target changes
    ON_LOST_FOCUS = auto()  # Update when target loses focus
    EXPLICIT = auto()  # Update only when explicitly requested


class BindingStatus(Enum):
    """Current status of a binding."""

    UNATTACHED = auto()
    ACTIVE = auto()
    INACTIVE = auto()
    DETACHED = auto()
    ERROR = auto()


@dataclass
class BindingError:
    """Information about a binding error."""

    message: str
    exception: Optional[Exception] = None
    source_path: Optional[str] = None
    target_path: Optional[str] = None
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class PropertyPath:
    """
    Represents a path to a property, supporting nested access.

    Examples:
        PropertyPath("name")           -> obj.name
        PropertyPath("address.city")   -> obj.address.city
        PropertyPath("[0]")            -> obj[0]
        PropertyPath("items[0].name")  -> obj.items[0].name
    """

    def __init__(self, path: str):
        self._path = path
        self._segments = self._parse_path(path)

    def _parse_path(self, path: str) -> List[Tuple[str, Optional[int]]]:
        """Parse path into segments (property_name, index)."""
        segments: List[Tuple[str, Optional[int]]] = []
        current = ""

        i = 0
        while i < len(path):
            char = path[i]

            if char == ".":
                if current:
                    segments.append((current, None))
                    current = ""
            elif char == "[":
                if current:
                    segments.append((current, None))
                    current = ""
                # Parse index
                end = path.find("]", i)
                if end == -1:
                    raise ValueError(f"Unclosed bracket in path: {path}")
                index_str = path[i + 1:end]
                try:
                    index = int(index_str)
                    segments.append(("", index))
                except ValueError:
                    # String key for dict
                    segments.append((index_str.strip("'\""), None))
                i = end
            else:
                current += char

            i += 1

        if current:
            segments.append((current, None))

        return segments

    @property
    def path(self) -> str:
        """Return the original path string."""
        return self._path

    @property
    def segments(self) -> List[Tuple[str, Optional[int]]]:
        """Return the parsed segments."""
        return self._segments

    @property
    def is_simple(self) -> bool:
        """Check if this is a simple (single property) path."""
        return len(self._segments) == 1 and self._segments[0][1] is None

    @property
    def root(self) -> str:
        """Get the root property name."""
        if self._segments:
            name, _ = self._segments[0]
            return name
        return ""

    def get_value(self, obj: Any) -> Any:
        """Get the value at this path from an object."""
        current = obj

        for name, index in self._segments:
            if current is None:
                return None

            if index is not None:
                # Indexer access
                try:
                    current = current[index]
                except (IndexError, KeyError, TypeError):
                    return None
            elif name:
                # Property access
                if isinstance(current, dict):
                    current = current.get(name)
                else:
                    current = getattr(current, name, None)

        return current

    def set_value(self, obj: Any, value: Any) -> bool:
        """Set the value at this path on an object. Returns True if successful."""
        if not self._segments:
            return False

        # Navigate to parent
        current = obj
        for name, index in self._segments[:-1]:
            if current is None:
                return False

            if index is not None:
                try:
                    current = current[index]
                except (IndexError, KeyError, TypeError):
                    return False
            elif name:
                if isinstance(current, dict):
                    current = current.get(name)
                else:
                    current = getattr(current, name, None)

        if current is None:
            return False

        # Set on parent
        final_name, final_index = self._segments[-1]
        try:
            if final_index is not None:
                current[final_index] = value
            elif final_name:
                if isinstance(current, dict):
                    current[final_name] = value
                else:
                    setattr(current, final_name, value)
            return True
        except (AttributeError, TypeError, IndexError, KeyError):
            return False

    def __repr__(self) -> str:
        return f"PropertyPath({self._path!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PropertyPath):
            return self._path == other._path
        if isinstance(other, str):
            return self._path == other
        return False

    def __hash__(self) -> int:
        return hash(self._path)


class BindingContext:
    """
    Provides context for data binding, including default converters,
    validators, and fallback values.
    """

    def __init__(
        self,
        source: Any = None,
        fallback_value: Any = None,
        target_null_value: Any = None,
        string_format: Optional[str] = None,
    ):
        self._source = source
        self._fallback_value = fallback_value
        self._target_null_value = target_null_value
        self._string_format = string_format
        self._resources: Dict[str, Any] = {}
        self._parent: Optional[BindingContext] = None
        self._converters: Dict[str, IConverter] = {}
        self._validators: Dict[str, IValidator] = {}

    @property
    def source(self) -> Any:
        """Get the binding source."""
        return self._source

    @source.setter
    def source(self, value: Any) -> None:
        """Set the binding source."""
        self._source = value

    @property
    def fallback_value(self) -> Any:
        """Value to use when binding fails."""
        return self._fallback_value

    @property
    def target_null_value(self) -> Any:
        """Value to use when source is null."""
        return self._target_null_value

    @property
    def string_format(self) -> Optional[str]:
        """Format string to apply to values."""
        return self._string_format

    @property
    def parent(self) -> Optional["BindingContext"]:
        """Get parent context."""
        return self._parent

    @parent.setter
    def parent(self, value: Optional["BindingContext"]) -> None:
        """Set parent context."""
        self._parent = value

    def register_converter(self, name: str, converter: IConverter) -> None:
        """Register a named converter."""
        self._converters[name] = converter

    def get_converter(self, name: str) -> Optional[IConverter]:
        """Get a converter by name, checking parent if not found."""
        if name in self._converters:
            return self._converters[name]
        if self._parent:
            return self._parent.get_converter(name)
        return None

    def register_validator(self, name: str, validator: IValidator) -> None:
        """Register a named validator."""
        self._validators[name] = validator

    def get_validator(self, name: str) -> Optional[IValidator]:
        """Get a validator by name, checking parent if not found."""
        if name in self._validators:
            return self._validators[name]
        if self._parent:
            return self._parent.get_validator(name)
        return None

    def set_resource(self, key: str, value: Any) -> None:
        """Set a resource value."""
        self._resources[key] = value

    def get_resource(self, key: str) -> Any:
        """Get a resource, checking parent if not found."""
        if key in self._resources:
            return self._resources[key]
        if self._parent:
            return self._parent.get_resource(key)
        return None

    def create_child(self, source: Any = None) -> "BindingContext":
        """Create a child context."""
        child = BindingContext(
            source=source if source is not None else self._source,
            fallback_value=self._fallback_value,
            target_null_value=self._target_null_value,
            string_format=self._string_format,
        )
        child._parent = self
        return child


class BindingExpression:
    """
    Represents a computed binding expression.

    Allows binding to computed values from multiple source properties.

    Example:
        expr = BindingExpression(
            lambda ctx: f"{ctx['firstName']} {ctx['lastName']}",
            dependencies=["firstName", "lastName"]
        )
    """

    def __init__(
        self,
        expression: Callable[[Dict[str, Any]], Any],
        dependencies: Optional[List[str]] = None,
        fallback_value: Any = None,
    ):
        self._expression = expression
        self._dependencies = dependencies or []
        self._fallback_value = fallback_value

    @property
    def dependencies(self) -> List[str]:
        """Property paths this expression depends on."""
        return self._dependencies

    @property
    def fallback_value(self) -> Any:
        """Value to use if expression fails."""
        return self._fallback_value

    def evaluate(self, context: Dict[str, Any]) -> Any:
        """Evaluate the expression with the given context."""
        try:
            return self._expression(context)
        except Exception:
            return self._fallback_value

    def __repr__(self) -> str:
        return f"BindingExpression(deps={self._dependencies})"


class BindingBase(ABC):
    """Base class for all binding types."""

    def __init__(
        self,
        mode: BindingMode = BindingMode.ONE_WAY,
        update_source_trigger: UpdateSourceTrigger = UpdateSourceTrigger.ON_PROPERTY_CHANGED,
        converter: Optional[IConverter] = None,
        converter_parameter: Any = None,
        fallback_value: Any = None,
        target_null_value: Any = None,
        string_format: Optional[str] = None,
        delay: float = DEFAULT_BINDING_DELAY,
    ):
        self._mode = mode
        self._update_source_trigger = update_source_trigger
        self._converter = converter
        self._converter_parameter = converter_parameter
        self._fallback_value = fallback_value
        self._target_null_value = target_null_value
        self._string_format = string_format
        self._delay = delay
        self._status = BindingStatus.UNATTACHED
        self._errors: List[BindingError] = []
        self._lock = threading.RLock()

    @property
    def mode(self) -> BindingMode:
        """Get the binding mode."""
        return self._mode

    @property
    def status(self) -> BindingStatus:
        """Get the current binding status."""
        return self._status

    @property
    def errors(self) -> List[BindingError]:
        """Get binding errors."""
        return list(self._errors)

    @property
    def has_error(self) -> bool:
        """Check if binding has errors."""
        return len(self._errors) > 0

    def clear_errors(self) -> None:
        """Clear all binding errors."""
        with self._lock:
            self._errors.clear()
            if self._status == BindingStatus.ERROR:
                self._status = BindingStatus.ACTIVE

    @abstractmethod
    def attach(self) -> None:
        """Attach the binding (start observing)."""
        pass

    @abstractmethod
    def detach(self) -> None:
        """Detach the binding (stop observing)."""
        pass

    @abstractmethod
    def update_target(self) -> None:
        """Force update target from source."""
        pass

    @abstractmethod
    def update_source(self) -> None:
        """Force update source from target."""
        pass


class Binding(BindingBase, Generic[TSource, TTarget]):
    """
    Core binding class connecting a source property to a target property.

    Supports:
    - One-way, two-way, one-time, one-way-to-source modes
    - Value converters
    - Validation
    - Fallback values
    - Update delay
    - String formatting
    """

    def __init__(
        self,
        source: Any,
        source_path: Union[str, PropertyPath],
        target: Any = None,
        target_path: Union[str, PropertyPath, None] = None,
        mode: BindingMode = BindingMode.ONE_WAY,
        update_source_trigger: UpdateSourceTrigger = UpdateSourceTrigger.ON_PROPERTY_CHANGED,
        converter: Optional[IConverter[TSource, TTarget]] = None,
        converter_parameter: Any = None,
        validators: Optional[List[IValidator]] = None,
        fallback_value: Any = None,
        target_null_value: Any = None,
        string_format: Optional[str] = None,
        delay: float = DEFAULT_BINDING_DELAY,
        validation_context: Optional[ValidationContext] = None,
    ):
        super().__init__(
            mode=mode,
            update_source_trigger=update_source_trigger,
            converter=converter,
            converter_parameter=converter_parameter,
            fallback_value=fallback_value,
            target_null_value=target_null_value,
            string_format=string_format,
            delay=delay,
        )

        # Source
        self._source_ref = weakref.ref(source) if source is not None else None
        self._source_path = (
            source_path if isinstance(source_path, PropertyPath)
            else PropertyPath(source_path)
        )

        # Target
        self._target_ref = weakref.ref(target) if target is not None else None
        self._target_path = (
            target_path if isinstance(target_path, PropertyPath)
            else PropertyPath(target_path) if target_path else None
        )

        # Validation
        self._validators = validators or []
        self._validation_context = validation_context
        self._validation_result: Optional[ValidationResult] = None

        # Internal state
        self._source_subscription: Optional[Callable] = None
        self._target_subscription: Optional[Callable] = None
        self._pending_update_timer: Optional[asyncio.TimerHandle] = None
        self._is_updating = False

    @property
    def source(self) -> Optional[Any]:
        """Get the source object."""
        return self._source_ref() if self._source_ref else None

    @property
    def target(self) -> Optional[Any]:
        """Get the target object."""
        return self._target_ref() if self._target_ref else None

    @property
    def source_path(self) -> PropertyPath:
        """Get the source property path."""
        return self._source_path

    @property
    def target_path(self) -> Optional[PropertyPath]:
        """Get the target property path."""
        return self._target_path

    @property
    def validation_result(self) -> Optional[ValidationResult]:
        """Get the last validation result."""
        return self._validation_result

    def attach(self) -> None:
        """Attach the binding and start observing changes."""
        with self._lock:
            if self._status in (BindingStatus.ACTIVE, BindingStatus.DETACHED):
                return

            source = self.source
            target = self.target

            if source is None:
                self._status = BindingStatus.ERROR
                self._errors.append(BindingError("Source is None"))
                return

            # Subscribe to source changes
            if self._mode in (BindingMode.ONE_WAY, BindingMode.TWO_WAY, BindingMode.ONE_TIME):
                self._subscribe_source()

            # Subscribe to target changes for two-way
            if self._mode == BindingMode.TWO_WAY and target is not None:
                self._subscribe_target()

            # Subscribe for one-way-to-source
            if self._mode == BindingMode.ONE_WAY_TO_SOURCE and target is not None:
                self._subscribe_target()

            self._status = BindingStatus.ACTIVE

            # Initial update
            if self._mode != BindingMode.ONE_WAY_TO_SOURCE:
                self.update_target()

            # For ONE_TIME mode, unsubscribe after initial update
            if self._mode == BindingMode.ONE_TIME:
                self._unsubscribe_source()
                self._status = BindingStatus.INACTIVE

    def detach(self) -> None:
        """Detach the binding and stop observing changes."""
        with self._lock:
            # Cancel any pending delayed update
            if self._pending_update_timer is not None:
                self._pending_update_timer.cancel()
                self._pending_update_timer = None
            self._unsubscribe_source()
            self._unsubscribe_target()
            self._status = BindingStatus.DETACHED

    def update_target(self) -> None:
        """Update target with current source value."""
        if self._is_updating:
            return

        with self._lock:
            self._is_updating = True
            try:
                source = self.source
                target = self.target

                if source is None:
                    self._apply_fallback_to_target()
                    return

                # Get source value
                value = self._source_path.get_value(source)

                # Handle null
                if value is None:
                    if self._target_null_value is not None:
                        value = self._target_null_value
                    elif self._fallback_value is not None:
                        value = self._fallback_value

                # Apply converter
                if self._converter is not None and value is not None:
                    try:
                        value = self._converter.convert(value, self._converter_parameter)
                    except Exception as e:
                        self._errors.append(BindingError(
                            f"Converter error: {e}",
                            exception=e,
                            source_path=self._source_path.path,
                        ))
                        value = self._fallback_value

                # Apply string format
                if self._string_format and value is not None:
                    try:
                        value = self._string_format.format(value)
                    except Exception:
                        pass

                # Set target value
                if target is not None and self._target_path is not None:
                    self._target_path.set_value(target, value)

            finally:
                self._is_updating = False

    def update_source(self) -> None:
        """Update source with current target value."""
        if self._is_updating:
            return

        if self._mode not in (BindingMode.TWO_WAY, BindingMode.ONE_WAY_TO_SOURCE):
            return

        with self._lock:
            self._is_updating = True
            try:
                source = self.source
                target = self.target

                if source is None or target is None or self._target_path is None:
                    return

                # Get target value
                value = self._target_path.get_value(target)

                # Validate
                if self._validators:
                    for validator in self._validators:
                        result = validator.validate(value)
                        self._validation_result = result
                        if not result.is_valid:
                            return  # Don't update source if validation fails

                if self._validation_context:
                    result = self._validation_context.validate_field(
                        self._source_path.path, value
                    )
                    self._validation_result = result
                    if not result.is_valid:
                        return

                # Apply converter (reverse)
                if self._converter is not None:
                    try:
                        value = self._converter.convert_back(
                            value, self._converter_parameter
                        )
                    except Exception as e:
                        self._errors.append(BindingError(
                            f"Converter back error: {e}",
                            exception=e,
                            target_path=str(self._target_path),
                        ))
                        return

                # Set source value
                self._source_path.set_value(source, value)

            finally:
                self._is_updating = False

    def validate(self) -> ValidationResult:
        """Validate the current target value."""
        target = self.target
        if target is None or self._target_path is None:
            return ValidationResult.valid()

        value = self._target_path.get_value(target)

        for validator in self._validators:
            result = validator.validate(value)
            if not result.is_valid:
                self._validation_result = result
                return result

        if self._validation_context:
            result = self._validation_context.validate_field(
                self._source_path.path, value
            )
            self._validation_result = result
            return result

        self._validation_result = ValidationResult.valid()
        return self._validation_result

    def _subscribe_source(self) -> None:
        """Subscribe to source property changes."""
        source = self.source
        if source is None:
            return

        # Check for observable pattern
        root_prop = self._source_path.root

        # Try to find add_observer/add_listener method
        add_observer = getattr(source, "add_observer", None)
        if add_observer and callable(add_observer):
            def callback(obj, field, old_val, new_val):
                if field == root_prop:
                    self.update_target()
            self._source_subscription = callback
            add_observer(callback)
            return

        # Try property change notification
        add_listener = getattr(source, "add_property_change_listener", None)
        if add_listener and callable(add_listener):
            def callback(prop_name, old_val, new_val):
                if prop_name == root_prop:
                    self.update_target()
            self._source_subscription = callback
            add_listener(callback)

    def _unsubscribe_source(self) -> None:
        """Unsubscribe from source property changes."""
        if self._source_subscription is None:
            return

        source = self.source
        if source is None:
            return

        remove_observer = getattr(source, "remove_observer", None)
        if remove_observer and callable(remove_observer):
            try:
                remove_observer(self._source_subscription)
            except Exception:
                pass

        remove_listener = getattr(source, "remove_property_change_listener", None)
        if remove_listener and callable(remove_listener):
            try:
                remove_listener(self._source_subscription)
            except Exception:
                pass

        self._source_subscription = None

    def _subscribe_target(self) -> None:
        """Subscribe to target property changes."""
        target = self.target
        if target is None or self._target_path is None:
            return

        root_prop = self._target_path.root

        add_observer = getattr(target, "add_observer", None)
        if add_observer and callable(add_observer):
            def callback(obj, field, old_val, new_val):
                if field == root_prop:
                    if self._delay > 0:
                        self._schedule_source_update()
                    else:
                        self.update_source()
            self._target_subscription = callback
            add_observer(callback)
            return

        add_listener = getattr(target, "add_property_change_listener", None)
        if add_listener and callable(add_listener):
            def callback(prop_name, old_val, new_val):
                if prop_name == root_prop:
                    if self._delay > 0:
                        self._schedule_source_update()
                    else:
                        self.update_source()
            self._target_subscription = callback
            add_listener(callback)

    def _unsubscribe_target(self) -> None:
        """Unsubscribe from target property changes."""
        if self._target_subscription is None:
            return

        target = self.target
        if target is None:
            return

        remove_observer = getattr(target, "remove_observer", None)
        if remove_observer and callable(remove_observer):
            try:
                remove_observer(self._target_subscription)
            except Exception:
                pass

        remove_listener = getattr(target, "remove_property_change_listener", None)
        if remove_listener and callable(remove_listener):
            try:
                remove_listener(self._target_subscription)
            except Exception:
                pass

        self._target_subscription = None

    def _apply_fallback_to_target(self) -> None:
        """Apply fallback value to target."""
        target = self.target
        if target is not None and self._target_path is not None:
            value = self._fallback_value
            if value is not None:
                self._target_path.set_value(target, value)

    def _schedule_source_update(self) -> None:
        """Schedule a delayed source update."""
        with self._lock:
            # Cancel pending update
            if self._pending_update_timer is not None:
                self._pending_update_timer.cancel()
                self._pending_update_timer = None

            # Don't schedule if detached
            if self._status == BindingStatus.DETACHED:
                return

            # Schedule new update
            try:
                loop = asyncio.get_event_loop()
                self._pending_update_timer = loop.call_later(
                    self._delay, self.update_source
                )
            except RuntimeError:
                # No event loop - update immediately
                self.update_source()

    def __repr__(self) -> str:
        return (
            f"Binding({self._source_path} -> {self._target_path}, "
            f"mode={self._mode.name}, status={self._status.name})"
        )


class MultiBinding(BindingBase):
    """
    Binding with multiple source properties to a single target.

    Uses a MultiValueConverter to combine source values.

    Example:
        multi = MultiBinding(
            sources=[
                (person, "firstName"),
                (person, "lastName"),
            ],
            target=label,
            target_path="text",
            converter=StringConcatConverter(" "),
        )
    """

    def __init__(
        self,
        sources: List[Tuple[Any, Union[str, PropertyPath]]],
        target: Any,
        target_path: Union[str, PropertyPath],
        converter: Optional[MultiValueConverter] = None,
        converter_parameter: Any = None,
        mode: BindingMode = BindingMode.ONE_WAY,
        fallback_value: Any = None,
        string_format: Optional[str] = None,
    ):
        super().__init__(
            mode=mode,
            converter_parameter=converter_parameter,
            fallback_value=fallback_value,
            string_format=string_format,
        )

        self._sources: List[Tuple[weakref.ref, PropertyPath]] = []
        for src, path in sources:
            path_obj = path if isinstance(path, PropertyPath) else PropertyPath(path)
            self._sources.append((weakref.ref(src), path_obj))

        self._target_ref = weakref.ref(target) if target else None
        self._target_path = (
            target_path if isinstance(target_path, PropertyPath)
            else PropertyPath(target_path)
        )

        self._multi_converter = converter
        self._source_subscriptions: List[Optional[Callable]] = []
        self._is_updating = False

    @property
    def target(self) -> Optional[Any]:
        """Get the target object."""
        return self._target_ref() if self._target_ref else None

    def get_source_values(self) -> List[Any]:
        """Get all current source values."""
        values = []
        for ref, path in self._sources:
            source = ref()
            if source is not None:
                values.append(path.get_value(source))
            else:
                values.append(None)
        return values

    def attach(self) -> None:
        """Attach the multi-binding."""
        with self._lock:
            if self._status == BindingStatus.ACTIVE:
                return

            # Subscribe to all sources
            for i, (ref, path) in enumerate(self._sources):
                source = ref()
                if source is not None:
                    self._subscribe_source(source, path, i)

            self._status = BindingStatus.ACTIVE
            self.update_target()

    def detach(self) -> None:
        """Detach the multi-binding."""
        with self._lock:
            for i, (ref, path) in enumerate(self._sources):
                source = ref()
                if source is not None:
                    self._unsubscribe_source(source, i)
            self._source_subscriptions.clear()
            self._status = BindingStatus.DETACHED

    def update_target(self) -> None:
        """Update target with combined source values."""
        if self._is_updating:
            return

        with self._lock:
            self._is_updating = True
            try:
                target = self.target
                if target is None:
                    return

                # Get all source values
                values = self.get_source_values()

                # Apply multi-converter
                if self._multi_converter:
                    try:
                        result = self._multi_converter.convert(
                            values, self._converter_parameter
                        )
                    except Exception as e:
                        self._errors.append(BindingError(
                            f"MultiConverter error: {e}",
                            exception=e,
                        ))
                        result = self._fallback_value
                else:
                    # Default: use first non-None value
                    result = next((v for v in values if v is not None), None)

                if result is None:
                    result = self._fallback_value

                # Apply string format
                if self._string_format and result is not None:
                    try:
                        result = self._string_format.format(result)
                    except Exception:
                        pass

                # Set target
                self._target_path.set_value(target, result)

            finally:
                self._is_updating = False

    def update_source(self) -> None:
        """Update sources from target (for two-way binding)."""
        if self._mode != BindingMode.TWO_WAY:
            return

        if self._is_updating:
            return

        with self._lock:
            self._is_updating = True
            try:
                target = self.target
                if target is None or self._multi_converter is None:
                    return

                # Get target value
                value = self._target_path.get_value(target)

                # Convert back to multiple values
                try:
                    values = self._multi_converter.convert_back(
                        value, self._converter_parameter
                    )
                except NotImplementedError:
                    return
                except Exception as e:
                    self._errors.append(BindingError(
                        f"MultiConverter back error: {e}",
                        exception=e,
                    ))
                    return

                # Set source values
                for i, (ref, path) in enumerate(self._sources):
                    if i < len(values):
                        source = ref()
                        if source is not None:
                            path.set_value(source, values[i])

            finally:
                self._is_updating = False

    def _subscribe_source(
        self, source: Any, path: PropertyPath, index: int
    ) -> None:
        """Subscribe to a single source."""
        # Extend subscription list if needed
        while len(self._source_subscriptions) <= index:
            self._source_subscriptions.append(None)

        root_prop = path.root

        add_observer = getattr(source, "add_observer", None)
        if add_observer and callable(add_observer):
            def callback(obj, field, old_val, new_val):
                if field == root_prop:
                    self.update_target()
            self._source_subscriptions[index] = callback
            add_observer(callback)

    def _unsubscribe_source(self, source: Any, index: int) -> None:
        """Unsubscribe from a single source."""
        if index >= len(self._source_subscriptions):
            return

        callback = self._source_subscriptions[index]
        if callback is None:
            return

        remove_observer = getattr(source, "remove_observer", None)
        if remove_observer and callable(remove_observer):
            try:
                remove_observer(callback)
            except Exception:
                pass

        self._source_subscriptions[index] = None

    def __repr__(self) -> str:
        source_count = len(self._sources)
        return f"MultiBinding({source_count} sources -> {self._target_path})"


class BindingGroup:
    """
    Manages a group of related bindings.

    Useful for forms or views that need to attach/detach all bindings together.
    """

    def __init__(self, context: Optional[BindingContext] = None):
        self._bindings: List[BindingBase] = []
        self._context = context
        self._lock = threading.RLock()

    @property
    def context(self) -> Optional[BindingContext]:
        """Get the binding context."""
        return self._context

    @property
    def bindings(self) -> List[BindingBase]:
        """Get all bindings in the group."""
        return list(self._bindings)

    def add(self, binding: BindingBase) -> None:
        """Add a binding to the group."""
        with self._lock:
            if binding not in self._bindings:
                self._bindings.append(binding)

    def remove(self, binding: BindingBase) -> None:
        """Remove a binding from the group."""
        with self._lock:
            if binding in self._bindings:
                binding.detach()
                self._bindings.remove(binding)

    def attach_all(self) -> None:
        """Attach all bindings in the group."""
        for binding in self._bindings:
            binding.attach()

    def detach_all(self) -> None:
        """Detach all bindings in the group."""
        for binding in self._bindings:
            binding.detach()

    def update_targets(self) -> None:
        """Update all binding targets."""
        for binding in self._bindings:
            binding.update_target()

    def update_sources(self) -> None:
        """Update all binding sources."""
        for binding in self._bindings:
            binding.update_source()

    def validate_all(self) -> List[ValidationResult]:
        """Validate all bindings and return results."""
        results = []
        for binding in self._bindings:
            if isinstance(binding, Binding):
                results.append(binding.validate())
        return results

    @property
    def is_valid(self) -> bool:
        """Check if all bindings are valid."""
        return all(r.is_valid for r in self.validate_all())

    @property
    def errors(self) -> List[BindingError]:
        """Get all binding errors."""
        all_errors = []
        for binding in self._bindings:
            all_errors.extend(binding.errors)
        return all_errors

    def clear(self) -> None:
        """Remove and detach all bindings."""
        with self._lock:
            for binding in self._bindings:
                binding.detach()
            self._bindings.clear()

    def __len__(self) -> int:
        return len(self._bindings)

    def __iter__(self):
        return iter(self._bindings)


# Convenience factory functions
def bind(
    source: Any,
    source_path: str,
    target: Any = None,
    target_path: Optional[str] = None,
    mode: BindingMode = BindingMode.ONE_WAY,
    converter: Optional[IConverter] = None,
) -> Binding:
    """Create and return a binding (not attached)."""
    return Binding(
        source=source,
        source_path=source_path,
        target=target,
        target_path=target_path,
        mode=mode,
        converter=converter,
    )


def bind_two_way(
    source: Any,
    source_path: str,
    target: Any,
    target_path: str,
    converter: Optional[IConverter] = None,
    validators: Optional[List[IValidator]] = None,
) -> Binding:
    """Create a two-way binding."""
    return Binding(
        source=source,
        source_path=source_path,
        target=target,
        target_path=target_path,
        mode=BindingMode.TWO_WAY,
        converter=converter,
        validators=validators,
    )


def bind_one_time(
    source: Any,
    source_path: str,
    target: Any,
    target_path: str,
) -> Binding:
    """Create a one-time binding (only sets initial value)."""
    return Binding(
        source=source,
        source_path=source_path,
        target=target,
        target_path=target_path,
        mode=BindingMode.ONE_TIME,
    )


def multi_bind(
    sources: List[Tuple[Any, str]],
    target: Any,
    target_path: str,
    converter: Optional[MultiValueConverter] = None,
) -> MultiBinding:
    """Create a multi-source binding."""
    return MultiBinding(
        sources=sources,
        target=target,
        target_path=target_path,
        converter=converter,
    )


__all__ = [
    # Enums
    "BindingMode",
    "UpdateSourceTrigger",
    "BindingStatus",
    # Core types
    "BindingError",
    "PropertyPath",
    "BindingContext",
    "BindingExpression",
    # Base class
    "BindingBase",
    # Main binding classes
    "Binding",
    "MultiBinding",
    "BindingGroup",
    # Factory functions
    "bind",
    "bind_two_way",
    "bind_one_time",
    "multi_bind",
]
