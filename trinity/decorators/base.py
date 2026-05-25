"""
Base decorator utilities for the Trinity Pattern.

Provides common functionality shared by all decorators including:
- Attribute attachment helpers
- Decorator tracking on targets
- Validation utilities
- Introspection helpers
"""

from __future__ import annotations

import sys
import warnings
from typing import Any, Callable, TypeVar

from trinity.decorators.registry import (
    DecoratorRegistry,
    DecoratorValidationError,
    Tier,
    registry,
)

T = TypeVar("T")

_UNKNOWN_TIER = "UNKNOWN"
_UNKNOWN_TIER_VALUE = -1


# =============================================================================
# DECORATOR TRACKING
# =============================================================================


def track_decorator(target: Any, decorator_name: str) -> None:
    """
    Track that a decorator has been applied to a target.

    This maintains a list of applied decorators on the target for
    introspection and validation purposes.

    Args:
        target: The decorated function or class.
        decorator_name: Name of the decorator being applied.
    """
    if not hasattr(target, "_applied_decorators"):
        target._applied_decorators = []

    target._applied_decorators.append(decorator_name)


def get_applied_decorators(target: Any) -> list[str]:
    """
    Get list of decorators applied to a target.

    Args:
        target: The decorated function or class.

    Returns:
        List of decorator names in application order.
    """
    return getattr(target, "_applied_decorators", []).copy()


def has_decorator(target: Any, decorator_name: str) -> bool:
    """
    Check if a target has a specific decorator applied.

    Args:
        target: The decorated function or class.
        decorator_name: Decorator name to check.

    Returns:
        True if the decorator is applied.
    """
    return decorator_name in getattr(target, "_applied_decorators", [])


# =============================================================================
# ATTRIBUTE ATTACHMENT
# =============================================================================


def attach_attributes(target: Any, **attributes: Any) -> None:
    """
    Attach multiple attributes to a target.

    This is the standard way for decorators to store their configuration
    on the decorated target.

    Args:
        target: The decorated function or class.
        **attributes: Attribute name-value pairs to attach.

    Example:
        attach_attributes(cls,
            _native=True,
            _native_backend='cython',
            _native_nogil=True
        )
    """
    for name, value in attributes.items():
        setattr(target, name, value)


def get_attribute(target: Any, name: str, default: T = None) -> T:
    """
    Get a decorator-attached attribute from a target.

    Args:
        target: The decorated function or class.
        name: Attribute name.
        default: Default value if attribute not found.

    Returns:
        The attribute value or default.
    """
    return getattr(target, name, default)


def merge_attributes(
    target: Any, name: str, value: Any, merge_type: str = "replace"
) -> None:
    """
    Merge a value into an existing attribute.

    Args:
        target: The decorated function or class.
        name: Attribute name.
        value: Value to merge.
        merge_type: How to merge - 'replace', 'union' (sets), 'extend' (lists), 'update' (dicts).
    """
    existing = getattr(target, name, None)

    if existing is None or merge_type == "replace":
        setattr(target, name, value)
    elif merge_type == "union":
        if not isinstance(existing, set):
            raise TypeError(
                f"merge_attributes: merge_type='union' requires existing attribute '{name}' "
                f"to be a set, got {type(existing).__name__}"
            )
        setattr(target, name, existing | value)
    elif merge_type == "extend":
        if not isinstance(existing, list):
            raise TypeError(
                f"merge_attributes: merge_type='extend' requires existing attribute '{name}' "
                f"to be a list, got {type(existing).__name__}"
            )
        if isinstance(value, str):
            value = [value]
        setattr(target, name, existing + list(value))
    elif merge_type == "update":
        if not isinstance(existing, dict):
            raise TypeError(
                f"merge_attributes: merge_type='update' requires existing attribute '{name}' "
                f"to be a dict, got {type(existing).__name__}"
            )
        existing.update(value)
    else:
        setattr(target, name, value)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_target_type(
    target: Any,
    decorator_name: str,
    allowed_types: tuple[str, ...] = ("any",),
) -> None:
    """
    Validate that a target is of an allowed type.

    Args:
        target: The decorated function or class.
        decorator_name: Name of the decorator (for error messages).
        allowed_types: Tuple of allowed types ('class', 'function', 'method', 'any').

    Raises:
        DecoratorValidationError: If target type is not allowed.
    """
    if "any" in allowed_types:
        return

    is_class = isinstance(target, type)
    is_function = callable(target) and not is_class

    if is_class and "class" not in allowed_types:
        raise DecoratorValidationError(
            f"@{decorator_name} cannot be applied to classes, "
            f"allowed types: {allowed_types}"
        )

    if is_function:
        if "function" not in allowed_types and "method" not in allowed_types:
            raise DecoratorValidationError(
                f"@{decorator_name} cannot be applied to functions, "
                f"allowed types: {allowed_types}"
            )


def validate_parameters(
    decorator_name: str,
    **params: tuple[Any, type, Optional[Callable[[Any], bool]]],
) -> None:
    """
    Validate decorator parameters.

    Args:
        decorator_name: Name of the decorator (for error messages).
        **params: Parameter specifications as (value, expected_type, validator).

    Raises:
        DecoratorValidationError: If validation fails.

    Example:
        validate_parameters(
            'native',
            backend=(backend, str, lambda x: x in VALID_BACKENDS),
            nogil=(nogil, bool, None),
        )
    """
    for param_name, (value, expected_type, validator) in params.items():
        if not isinstance(value, expected_type):
            raise DecoratorValidationError(
                f"@{decorator_name}: parameter '{param_name}' must be "
                f"{expected_type.__name__}, got {type(value).__name__}"
            )

        if validator is not None and not validator(value):
            raise DecoratorValidationError(
                f"@{decorator_name}: invalid value for parameter '{param_name}': {value}"
            )


def check_required_decorators(
    target: Any,
    decorator_name: str,
    required: tuple[str, ...],
) -> None:
    """
    Check that required decorators are present.

    Args:
        target: The decorated function or class.
        decorator_name: Name of the decorator being applied.
        required: Tuple of required decorator names.

    Raises:
        DecoratorValidationError: If required decorators are missing.
    """
    applied = get_applied_decorators(target)

    for req in required:
        if req not in applied:
            raise DecoratorValidationError(
                f"@{decorator_name} requires @{req} to be applied first"
            )


def check_excluded_decorators(
    target: Any,
    decorator_name: str,
    excluded: tuple[str, ...],
) -> None:
    """
    Check that excluded decorators are not present.

    Args:
        target: The decorated function or class.
        decorator_name: Name of the decorator being applied.
        excluded: Tuple of excluded decorator names.

    Raises:
        DecoratorValidationError: If excluded decorators are present.
    """
    applied = get_applied_decorators(target)

    for exc in excluded:
        if exc in applied:
            raise DecoratorValidationError(
                f"@{decorator_name} cannot be combined with @{exc}"
            )


# =============================================================================
# DECORATOR FACTORIES
# =============================================================================


def make_marker_decorator(
    name: str,
    tier: Tier,
    attribute_name: str,
    attribute_value: Any = True,
    target_types: tuple[str, ...] = ("any",),
    requires: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
    doc: str = "",
) -> Callable[[T], T]:
    """
    .. deprecated::
        Superseded by `make_decorator()` from `trinity.decorators.ops`. Use `make_decorator()`
        for all new decorators — it produces Op-aware decorators with `_steps` metadata.

    Create a simple marker decorator that sets an attribute.

    This is a factory for decorators that just mark their targets
    with a boolean or simple value.

    Args:
        name: Decorator name for registration.
        tier: Decorator tier.
        attribute_name: Name of the attribute to set.
        attribute_value: Value to set (default True).
        target_types: Allowed target types.
        requires: Required decorators.
        excludes: Excluded decorators.
        doc: Documentation string.

    Returns:
        The marker decorator.

    Example:
        unsafe = make_marker_decorator(
            name='unsafe',
            tier=Tier.COMPILATION,
            attribute_name='_unsafe',
            doc='Marks code as containing unsafe operations.'
        )
    """

    warnings.warn(
        "make_marker_decorator() is deprecated. Use make_decorator() from trinity.decorators.ops instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    @registry.register(
        name=name,
        tier=tier,
        requires=requires,
        excludes=excludes,
        unique=True,
        doc=doc,
        target_types=target_types,
    )
    def decorator(target: T) -> T:
        validate_target_type(target, name, target_types)
        check_required_decorators(target, name, requires)
        check_excluded_decorators(target, name, excludes)

        attach_attributes(target, **{attribute_name: attribute_value})
        track_decorator(target, name)

        return target

    decorator.__doc__ = doc
    decorator.__name__ = name

    return decorator


def make_configurable_decorator(
    name: str,
    tier: Tier,
    config_class: type,
    attribute_name: str,
    target_types: tuple[str, ...] = ("any",),
    requires: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
    doc: str = "",
) -> Callable[..., Callable[[T], T]]:
    """
    .. deprecated::
        Superseded by `make_decorator()` from `trinity.decorators.ops`. Use `make_decorator()`
        for all new decorators — it produces Op-aware decorators with `_steps` metadata.

    Create a configurable decorator that stores configuration.

    This is a factory for decorators that accept parameters and
    store a configuration object on the target.

    Args:
        name: Decorator name for registration.
        tier: Decorator tier.
        config_class: Dataclass to use for configuration.
        attribute_name: Name of the attribute to store config.
        target_types: Allowed target types.
        requires: Required decorators.
        excludes: Excluded decorators.
        doc: Documentation string.

    Returns:
        The configurable decorator factory.
    """

    warnings.warn(
        "make_configurable_decorator() is deprecated. Use make_decorator() from trinity.decorators.ops instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    @registry.register(
        name=name,
        tier=tier,
        requires=requires,
        excludes=excludes,
        unique=True,
        doc=doc,
        target_types=target_types,
    )
    def decorator_factory(**kwargs: Any) -> Callable[[T], T]:
        config = config_class(**kwargs)

        def decorator(target: T) -> T:
            validate_target_type(target, name, target_types)
            check_required_decorators(target, name, requires)
            check_excluded_decorators(target, name, excludes)

            attach_attributes(target, **{attribute_name: config})
            # Set a boolean flag indicating this decorator is active.
            # E.g., attribute_name="_native_config" → sets "_native" = True
            setattr(
                target,
                f"{attribute_name.removesuffix('_config')}"
                if attribute_name.endswith("_config")
                else f"_{name}",
                True,
            )
            track_decorator(target, name)

            return target

        return decorator

    decorator_factory.__doc__ = doc
    decorator_factory.__name__ = name

    return decorator_factory


# =============================================================================
# INTROSPECTION
# =============================================================================


def inspect_decorated(target: Any) -> dict[str, Any]:
    """
    Get detailed information about all decorators on a target.

    Args:
        target: The decorated function or class.

    Returns:
        Dictionary with decorator information.
    """
    applied = get_applied_decorators(target)

    decorators_info = []
    for dec_name in applied:
        spec = registry.get(dec_name)
        if spec:
            decorators_info.append(
                {
                    "name": dec_name,
                    "tier": spec.tier.name,
                    "tier_value": spec.tier.value,
                }
            )
        else:
            decorators_info.append(
                {
                    "name": dec_name,
                    "tier": _UNKNOWN_TIER,
                    "tier_value": _UNKNOWN_TIER_VALUE,
                }
            )

    # Collect all underscore attributes (decorator-set)
    attributes = {}
    for attr_name in dir(target):
        if attr_name.startswith("_") and not attr_name.startswith("__"):
            try:
                value = getattr(target, attr_name)
                # Skip methods and complex objects
                if not callable(value) and not attr_name.startswith("_applied"):
                    attributes[attr_name] = repr(value)
            except Exception:
                pass

    return {
        "target": target.__name__ if hasattr(target, "__name__") else str(target),
        "type": "class" if isinstance(target, type) else "function",
        "decorators": decorators_info,
        "attributes": attributes,
        "decorator_count": len(applied),
    }


def get_decorator_chain(target: Any) -> list[str]:
    """
    Get ordered list of decorators applied to target in application order (first applied → first in list).

    Args:
        target: The decorated function or class.

    Returns:
        List of decorator names in application order.
    """
    return get_applied_decorators(target)


# =============================================================================
# PLATFORM DETECTION
# =============================================================================


def get_current_platform() -> str:
    """
    Get the current platform identifier.

    Returns:
        Platform string: 'windows', 'linux', 'macos', 'ios', 'android', etc.
    """
    platform = sys.platform

    if platform == "win32":
        return "windows"
    elif platform == "darwin":
        return "macos"
    elif platform.startswith("linux"):
        return "linux"
    elif platform == "emscripten":
        return "web"
    else:
        return platform


def get_current_arch() -> str:
    """
    Get the current CPU architecture.

    Returns:
        Architecture string: 'x86_64', 'arm64', 'wasm32', etc.
    """
    import platform

    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        return "x86_64"
    elif machine in ("arm64", "aarch64"):
        return "arm64"
    elif machine in ("i386", "i686", "x86"):
        return "x86"
    elif "arm" in machine:
        return "arm"
    else:
        return machine


# =============================================================================
# STUB GENERATOR
# =============================================================================


class PlatformUnavailableError(RuntimeError):
    """Raised when attempting to use platform-unavailable code."""

    pass


def create_unavailable_stub(
    target_name: str,
    decorator_name: str,
    reason: str,
) -> Callable[..., None]:
    """
    Create a stub function that raises PlatformUnavailableError.

    Used by @platform and @target decorators when code is not available
    on the current platform.

    Args:
        target_name: Name of the unavailable function/class.
        decorator_name: Decorator that caused unavailability.
        reason: Human-readable reason for unavailability.

    Returns:
        A function that raises PlatformUnavailableError when called.
    """

    def stub(*args: Any, **kwargs: Any) -> None:
        raise PlatformUnavailableError(
            f"{target_name} is not available: {reason}\n"
            f"Marked unavailable by @{decorator_name}"
        )

    stub.__name__ = target_name
    stub.__doc__ = f"UNAVAILABLE: {reason}"
    stub._unavailable = True
    stub._unavailable_reason = reason

    return stub


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Tracking
    "track_decorator",
    "get_applied_decorators",
    "has_decorator",
    # Attributes
    "attach_attributes",
    "get_attribute",
    "merge_attributes",
    # Validation
    "validate_target_type",
    "validate_parameters",
    "check_required_decorators",
    "check_excluded_decorators",
    "DecoratorValidationError",
    # Factories
    "make_marker_decorator",
    "make_configurable_decorator",
    # Introspection
    "inspect_decorated",
    "get_decorator_chain",
    # Platform
    "get_current_platform",
    "get_current_arch",
    "PlatformUnavailableError",
    "create_unavailable_stub",
    # Registry access
    "registry",
    "Tier",
]
