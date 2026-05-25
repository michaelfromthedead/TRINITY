"""
Capability Security - Restrict what code can access.
Part of Core Foundation Layer 3.

Provides capability-based security for AI agents and modding.
Uses immutable capability sets and context variables for thread-safe
capability enforcement.
"""
from __future__ import annotations
from enum import Flag, auto
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar
from contextvars import ContextVar, Token
from functools import wraps


class Capability(Flag):
    """
    Capabilities that can be granted to code.

    Capabilities are bit flags that can be combined using bitwise operators.
    Each capability grants permission to perform specific operations.
    """
    NONE = 0

    # Basic data access
    READ = auto()           # Read entity fields
    WRITE = auto()          # Modify entity fields
    CREATE = auto()         # Create new entities
    DELETE = auto()         # Delete entities

    # Execution capabilities
    EXECUTE = auto()        # Execute shell commands
    SPAWN = auto()          # Spawn new agents/processes

    # External access
    NETWORK = auto()        # Network access
    FILESYSTEM = auto()     # File system access

    # Common combinations
    READONLY = READ
    READWRITE = READ | WRITE
    ENTITY_FULL = READ | WRITE | CREATE | DELETE
    FULL = READ | WRITE | CREATE | DELETE | EXECUTE | SPAWN | NETWORK | FILESYSTEM


@dataclass(frozen=True)
class CapabilitySet:
    """
    Immutable set of capabilities.

    Once created, a CapabilitySet cannot be modified. Operations like
    grant() and revoke() return new CapabilitySet instances.

    Examples:
        >>> caps = CapabilitySet(Capability.READ | Capability.WRITE)
        >>> caps.has(Capability.READ)
        True
        >>> caps.has(Capability.DELETE)
        False
        >>> new_caps = caps.grant(Capability.DELETE)
        >>> new_caps.has(Capability.DELETE)
        True
        >>> caps.has(Capability.DELETE)  # Original unchanged
        False
    """
    capabilities: Capability

    def has(self, cap: Capability) -> bool:
        """
        Check if this set has the specified capability.

        For compound capabilities (multiple flags), ALL must be present.

        Args:
            cap: The capability or combination to check for

        Returns:
            True if all requested capabilities are present
        """
        return (self.capabilities & cap) == cap

    def has_any(self, cap: Capability) -> bool:
        """
        Check if this set has any of the specified capabilities.

        Args:
            cap: The capability or combination to check for

        Returns:
            True if any of the requested capabilities are present
        """
        return bool(self.capabilities & cap)

    def grant(self, cap: Capability) -> CapabilitySet:
        """
        Create a new CapabilitySet with the additional capability.

        Args:
            cap: The capability to add

        Returns:
            New CapabilitySet with the capability added
        """
        return CapabilitySet(self.capabilities | cap)

    def revoke(self, cap: Capability) -> CapabilitySet:
        """
        Create a new CapabilitySet without the specified capability.

        Args:
            cap: The capability to remove

        Returns:
            New CapabilitySet with the capability removed
        """
        return CapabilitySet(self.capabilities & ~cap)

    def __contains__(self, cap: Capability) -> bool:
        """Support 'in' operator for capability checking."""
        return self.has(cap)

    def __or__(self, other: CapabilitySet) -> CapabilitySet:
        """Combine two capability sets."""
        return CapabilitySet(self.capabilities | other.capabilities)

    def __and__(self, other: CapabilitySet) -> CapabilitySet:
        """Intersect two capability sets."""
        return CapabilitySet(self.capabilities & other.capabilities)

    def __repr__(self) -> str:
        """Return a readable representation of the capability set."""
        if self.capabilities == Capability.NONE:
            return "CapabilitySet(NONE)"
        flags = [f.name for f in Capability if f in self.capabilities and f.name]
        return f"CapabilitySet({' | '.join(flags)})"


# Context variable for current capability set (thread-safe)
_current_capabilities: ContextVar[Optional[CapabilitySet]] = ContextVar(
    'current_capabilities',
    default=None
)


class CapabilityError(Exception):
    """
    Raised when code lacks a required capability.

    Attributes:
        required: The capability that was required
        available: The capabilities that were available (if any)
    """
    def __init__(
        self,
        message: str,
        required: Optional[Capability] = None,
        available: Optional[CapabilitySet] = None
    ):
        super().__init__(message)
        self.required = required
        self.available = available


def get_current_capabilities() -> Optional[CapabilitySet]:
    """
    Get the capabilities of the current context.

    Returns:
        The current CapabilitySet, or None if no capability context is active
    """
    return _current_capabilities.get()


def check_capability(cap: Capability) -> bool:
    """
    Check if the current context has the specified capability.

    Args:
        cap: The capability to check for

    Returns:
        True if the capability is present, False if not present or no context
    """
    caps = _current_capabilities.get()
    if caps is None:
        return False
    return caps.has(cap)


def assert_capability(cap: Capability) -> None:
    """
    Assert that the current context has the specified capability.

    Raises:
        CapabilityError: If the capability is not present
    """
    caps = _current_capabilities.get()
    if caps is None:
        raise CapabilityError(
            f"No capability context active; {cap.name} required",
            required=cap,
            available=None
        )
    if not caps.has(cap):
        raise CapabilityError(
            f"Missing capability: {cap.name}",
            required=cap,
            available=caps
        )


F = TypeVar('F', bound=Callable[..., Any])


def require_capability(cap: Capability) -> Callable[[F], F]:
    """
    Decorator that requires a capability to call the function.

    If the current context does not have the required capability,
    raises CapabilityError.

    Args:
        cap: The capability required to call the function

    Returns:
        A decorator that enforces the capability requirement

    Example:
        >>> @require_capability(Capability.WRITE)
        ... def modify_entity(entity, value):
        ...     entity.value = value
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            assert_capability(cap)
            return fn(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


class SecureContext:
    """
    Context manager that sets capabilities for a block of code.

    Capabilities are scoped to the context and automatically restored
    when the context exits. Contexts can be nested, and inner contexts
    can only restrict (not expand) capabilities.

    Examples:
        >>> caps = CapabilitySet(Capability.READ | Capability.WRITE)
        >>> with SecureContext(caps):
        ...     check_capability(Capability.READ)  # True
        ...     check_capability(Capability.DELETE)  # False
        >>> check_capability(Capability.READ)  # False (outside context)

    Nested contexts with restriction:
        >>> outer = CapabilitySet(Capability.READ | Capability.WRITE)
        >>> inner = CapabilitySet(Capability.READ)
        >>> with SecureContext(outer):
        ...     check_capability(Capability.WRITE)  # True
        ...     with SecureContext(inner):
        ...         check_capability(Capability.WRITE)  # False
        ...     check_capability(Capability.WRITE)  # True (restored)
    """
    __slots__ = ('_capabilities', '_token', '_restricted')

    def __init__(self, capabilities: CapabilitySet, restrict: bool = True):
        """
        Create a secure context with the specified capabilities.

        Args:
            capabilities: The capabilities to grant within this context
            restrict: If True, inner contexts cannot exceed outer capabilities
        """
        self._capabilities = capabilities
        self._token: Optional[Token[Optional[CapabilitySet]]] = None
        self._restricted = restrict

    def __enter__(self) -> SecureContext:
        """Enter the secure context, setting capabilities."""
        effective = self._capabilities

        # If restricting, intersect with parent capabilities
        if self._restricted:
            parent = _current_capabilities.get()
            if parent is not None:
                effective = CapabilitySet(
                    self._capabilities.capabilities & parent.capabilities
                )

        self._token = _current_capabilities.set(effective)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any
    ) -> None:
        """Exit the secure context, restoring previous capabilities."""
        if self._token is not None:
            _current_capabilities.reset(self._token)
            self._token = None

    @property
    def capabilities(self) -> CapabilitySet:
        """Get the capabilities this context was created with."""
        return self._capabilities


def with_capabilities(capabilities: CapabilitySet) -> Callable[[F], F]:
    """
    Decorator that runs a function within a capability context.

    Args:
        capabilities: The capabilities to use when calling the function

    Returns:
        A decorator that wraps the function in a SecureContext

    Example:
        >>> @with_capabilities(CapabilitySet(Capability.READ))
        ... def read_data():
        ...     # This function runs with READ capability
        ...     pass
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with SecureContext(capabilities):
                return fn(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


# Pre-defined capability sets for common use cases
CAPS_NONE = CapabilitySet(Capability.NONE)
CAPS_READONLY = CapabilitySet(Capability.READONLY)
CAPS_READWRITE = CapabilitySet(Capability.READWRITE)
CAPS_FULL = CapabilitySet(Capability.FULL)


__all__ = [
    # Core types
    "Capability",
    "CapabilitySet",
    "CapabilityError",
    # Context management
    "SecureContext",
    # Functions
    "require_capability",
    "with_capabilities",
    "check_capability",
    "assert_capability",
    "get_current_capabilities",
    # Pre-defined sets
    "CAPS_NONE",
    "CAPS_READONLY",
    "CAPS_READWRITE",
    "CAPS_FULL",
]
