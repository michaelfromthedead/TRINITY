"""
Base descriptor classes for the Trinity Pattern.

Provides the TrinityDescriptor protocol and BaseDescriptor implementation
that all concrete descriptors inherit from.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from typing import Callable

    from trinity.decorators.ops import Step

T = TypeVar("T")


# =============================================================================
# READ-TRACKING FOR INCREMENTAL COMPUTATION
# =============================================================================

class Computation(Protocol):
    """
    Protocol for computations that track field reads.

    Used for future incremental/reactive computation support.
    Implementations can record which fields were read during execution
    to enable automatic invalidation and recomputation.
    """

    def record_read(self, obj: Any, field_name: str) -> None:
        """
        Record that a field was read during this computation.

        Args:
            obj: The object whose field was read.
            field_name: The name of the field that was read.
        """
        ...


# Context variable for tracking reads within a computation
_current_computation: ContextVar[Optional[Computation]] = ContextVar(
    'current_computation', default=None
)


def set_current_computation(computation: Optional[Computation]) -> None:
    """
    Set the current computation context for read tracking.

    Args:
        computation: The computation to track reads for, or None to disable.
    """
    _current_computation.set(computation)


def get_current_computation() -> Optional[Computation]:
    """Get the current computation context, if any."""
    return _current_computation.get()


@runtime_checkable
class TrinityDescriptor(Protocol[T]):
    """
    Protocol for all descriptors in the Trinity Pattern.

    Descriptors implementing this protocol can be:
    - Composed (wrapped) with guaranteed interface
    - Introspected at runtime
    - Validated for compatibility
    """

    # =========================================================================
    # IDENTITY
    # =========================================================================

    @property
    def name(self) -> str:
        """Field name this descriptor is bound to."""
        ...

    @property
    def field_type(self) -> type:
        """The annotated type of the field."""
        ...

    @property
    def descriptor_id(self) -> str:
        """Unique identifier for this descriptor type (e.g., 'tracked', 'networked')."""
        ...

    # =========================================================================
    # COMPOSITION
    # =========================================================================

    @property
    def inner(self) -> Optional["TrinityDescriptor[T]"]:
        """The wrapped descriptor, or None if this is the innermost."""
        ...

    @property
    def accepts_inner(self) -> tuple[str, ...]:
        """Descriptor IDs this can wrap. Empty tuple or ('*',) = any."""
        ...

    @property
    def accepts_outer(self) -> tuple[str, ...]:
        """Descriptor IDs that can wrap this. Empty tuple or ('*',) = any."""
        ...

    @property
    def excludes(self) -> tuple[str, ...]:
        """Descriptor IDs that cannot coexist in the same chain."""
        ...

    # =========================================================================
    # CORE DESCRIPTOR PROTOCOL
    # =========================================================================

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T: ...

    def __set__(self, obj: Any, value: T) -> None: ...

    def __delete__(self, obj: Any) -> None: ...

    def __set_name__(self, owner: type, name: str) -> None: ...

    # =========================================================================
    # LIFECYCLE HOOKS
    # =========================================================================

    def pre_get(self, obj: Any) -> None:
        """Called before retrieving value. Can modify obj state."""
        ...

    def post_get(self, obj: Any, value: T) -> T:
        """Called after retrieving value. Can transform the value."""
        ...

    def pre_set(self, obj: Any, value: T) -> T:
        """Called before storing value. Can transform/validate the value."""
        ...

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        """Called after storing value. Can trigger side effects."""
        ...

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    @property
    def descriptor_steps(self) -> list["Step"]:
        """The Ops this descriptor performs, expressed as Steps."""
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Return descriptor-specific metadata for introspection."""
        ...

    def get_chain(self) -> list["TrinityDescriptor[T]"]:
        """Return the full descriptor chain, outermost first."""
        ...


class BaseDescriptor(Generic[T]):
    """
    Base implementation of TrinityDescriptor protocol.

    Subclasses should override:
    - descriptor_id (class attribute, required)
    - pre_set / post_set (for write interception)
    - pre_get / post_get (for read interception)
    - accepts_inner / accepts_outer / excludes (for composition rules)

    This class provides:
    - Proper delegation to inner descriptor
    - Storage fallback when no inner descriptor
    - Chain introspection
    - Lifecycle hook framework
    """

    __slots__ = ("_name", "_field_type", "_inner", "_owner", "_config")

    # =========================================================================
    # CLASS ATTRIBUTES (override in subclasses)
    # =========================================================================

    descriptor_id: str = "base"
    accepts_inner: tuple[str, ...] = ("*",)  # Accept any
    accepts_outer: tuple[str, ...] = ("*",)  # Accept any
    excludes: tuple[str, ...] = ()  # No exclusions

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        field_type: type = object,
        inner: Optional["BaseDescriptor[T]"] = None,
        **config: Any,
    ) -> None:
        """
        Initialize the descriptor.

        Args:
            field_type: The type annotation for this field.
            inner: Optional inner descriptor to wrap.
            **config: Additional configuration passed to subclasses.
        """
        self._name: str = ""
        self._field_type = field_type
        self._inner = inner
        self._owner: Optional[type] = None
        self._config = config

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when descriptor is assigned to a class attribute."""
        self._name = name
        self._owner = owner
        if self._inner is not None:
            self._inner.__set_name__(owner, name)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def name(self) -> str:
        return self._name

    @property
    def field_type(self) -> type:
        return self._field_type

    @property
    def inner(self) -> Optional["BaseDescriptor[T]"]:
        return self._inner

    # =========================================================================
    # CORE DESCRIPTOR METHODS
    # =========================================================================

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T:
        """Get the field value."""
        if obj is None:
            return self  # type: ignore - Class-level access returns descriptor

        # Record read for incremental computation (future-proofing)
        comp = _current_computation.get()
        if comp is not None:
            comp.record_read(obj, self._name)

        self.pre_get(obj)

        # Get value early so we can record it for provenance
        if self._inner is not None:
            raw_value = self._inner.__get__(obj, objtype)
        else:
            raw_value = self._get_stored(obj)

        # Record read for provenance tracking (integrates with foundation.provenance)
        try:
            from foundation.provenance import get_current_reads_collector, record_read
            reads_collector = get_current_reads_collector()
            if reads_collector is not None:
                record_read(obj, self._name, raw_value)
        except ImportError:
            pass  # Provenance not available

        return self.post_get(obj, raw_value)

    def __set__(self, obj: Any, value: T) -> None:
        """Set the field value."""
        # Transform value
        value = self.pre_set(obj, value)

        # Get old value for post_set
        old_value = self._get_stored_safe(obj)

        # Store via inner descriptor or directly
        if self._inner is not None:
            self._inner.__set__(obj, value)
        else:
            self._set_stored(obj, value)

        self.post_set(obj, value, old_value)

    def __delete__(self, obj: Any) -> None:
        """Delete the field value."""
        if self._inner is not None:
            self._inner.__delete__(obj)
        else:
            self._delete_stored(obj)

    # =========================================================================
    # STORAGE (innermost descriptor uses these)
    # =========================================================================

    def _get_stored(self, obj: Any) -> T:
        """Retrieve value from object storage."""
        return obj.__dict__.get(self._name)  # type: ignore

    def _get_stored_safe(self, obj: Any) -> Optional[T]:
        """Retrieve value, returning None if not present."""
        return obj.__dict__.get(self._name)

    def _set_stored(self, obj: Any, value: T) -> None:
        """Store value in object storage."""
        obj.__dict__[self._name] = value

    def _delete_stored(self, obj: Any) -> None:
        """Remove value from object storage."""
        obj.__dict__.pop(self._name, None)

    # =========================================================================
    # LIFECYCLE HOOKS (override in subclasses)
    # =========================================================================

    def pre_get(self, obj: Any) -> None:
        """Override to add pre-read behavior."""
        pass

    def post_get(self, obj: Any, value: T) -> T:
        """Override to transform read value."""
        return value

    def pre_set(self, obj: Any, value: T) -> T:
        """Override to validate/transform write value."""
        return value

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        """Override to add post-write side effects."""
        pass

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    @property
    def descriptor_steps(self) -> list[Step]:
        return []  # Subclasses override

    def get_metadata(self) -> dict[str, Any]:
        """Return descriptor configuration and state."""
        return {
            "descriptor_id": self.descriptor_id,
            "name": self._name,
            "field_type": self._field_type.__name__ if self._field_type else "unknown",
            "config": self._config.copy(),
            "has_inner": self._inner is not None,
        }

    def get_chain(self) -> list["BaseDescriptor[T]"]:
        """Return full descriptor chain, outermost first."""
        chain: list[BaseDescriptor[T]] = [self]
        current = self._inner
        while current is not None:
            chain.append(current)
            current = current._inner
        return chain

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        inner_repr = f" -> {self._inner.descriptor_id}" if self._inner else ""
        return f"<{self.descriptor_id}:{self._name}{inner_repr}>"
