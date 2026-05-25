"""
Tier 10 (DEBUG_SAFETY) and Tier 11 (CHANGE_DETECTION) decorators.

Tier 10 decorators:
    @reads          - Declare read access to component types
    @writes         - Declare write access to component types
    @trace_stack    - Enhanced error stack traces with decorator chain

Tier 11 decorators:
    @track_changes  - Track changes to component fields (requires @component)
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator, run_steps
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# TIER 10: DEBUG_SAFETY
# =============================================================================


def reads(*components: type) -> Callable[[F], F]:
    """
    Declare read access to component types.

    Applied to system functions to document which components they read.

    Args:
        *components: Component types this system reads from.

    Example:
        @reads(Transform, Velocity)
        def movement_system(query: Query[Transform, Velocity]):
            ...
    """
    def decorator(fn: F) -> F:
        steps = [
            Step(Op.TAG, {"key": "reads", "value": True}),
            Step(Op.TAG, {"key": "reads_components", "value": components}),
            Step(Op.REGISTER, {"registry": "debug_safety"}),
        ]
        run_steps(fn, steps)

        # Track decorator application
        if not hasattr(fn, '_applied_decorators'):
            fn._applied_decorators = []
        fn._applied_decorators.append('reads')

        # Set convenience attributes
        fn._reads = True
        fn._reads_components = components

        return fn

    return decorator


# Mark as decorator for introspection
reads._is_decorator = True  # type: ignore
reads._decorator_name = "reads"  # type: ignore
reads.__name__ = "reads"


def writes(*components: type) -> Callable[[F], F]:
    """
    Declare write access to component types.

    Applied to system functions to document which components they modify.

    Args:
        *components: Component types this system writes to.

    Example:
        @writes(Transform, Velocity)
        def physics_system(query: Query[Transform, Velocity]):
            ...
    """
    def decorator(fn: F) -> F:
        steps = [
            Step(Op.TAG, {"key": "writes", "value": True}),
            Step(Op.TAG, {"key": "writes_components", "value": components}),
            Step(Op.REGISTER, {"registry": "debug_safety"}),
        ]
        run_steps(fn, steps)

        # Track decorator application
        if not hasattr(fn, '_applied_decorators'):
            fn._applied_decorators = []
        fn._applied_decorators.append('writes')

        # Set convenience attributes
        fn._writes = True
        fn._writes_components = components

        return fn

    return decorator


# Mark as decorator for introspection
writes._is_decorator = True  # type: ignore
writes._decorator_name = "writes"  # type: ignore
writes.__name__ = "writes"


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _trace_stack_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @trace_stack decorator."""
    depth = params.get("depth", 3)
    show_decorator_chain = params.get("show_decorator_chain", True)
    return [
        Step(Op.TAG, {"key": "trace_stack", "value": True}),
        Step(Op.TAG, {"key": "trace_stack_depth", "value": depth}),
        Step(Op.TAG, {"key": "trace_stack_show_chain", "value": show_decorator_chain}),
        Step(Op.HOOK, {"event": "on_error"}),
        Step(Op.REGISTER, {"registry": "debug_safety"}),
    ]


def _track_changes_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @track_changes decorator."""
    fields = params.get("fields")
    return [
        Step(Op.TAG, {"key": "track_changes", "value": True}),
        Step(Op.TAG, {"key": "track_changes_fields", "value": fields}),
        Step(Op.TRACK, {}),
        Step(Op.REGISTER, {"registry": "change_detection"}),
    ]


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_trace_stack(**kwargs: Any) -> None:
    """Validate @trace_stack parameters."""
    depth = kwargs.get("depth", 3)
    if not isinstance(depth, int) or depth < 1:
        raise TypeError("depth must be a positive integer")

    show_decorator_chain = kwargs.get("show_decorator_chain", True)
    if not isinstance(show_decorator_chain, bool):
        raise TypeError("show_decorator_chain must be a boolean")


def _validate_track_changes(**kwargs: Any) -> None:
    """Validate @track_changes parameters."""
    fields = kwargs.get("fields")
    if fields is not None:
        if not isinstance(fields, list):
            raise TypeError("fields must be a list of strings or None")
        if not all(isinstance(f, str) for f in fields):
            raise TypeError("all fields must be strings")


# =============================================================================
# AFTER STEPS
# =============================================================================


def _after_trace_stack(target: Any, params: dict[str, Any]) -> None:
    """Set convenience attributes after @trace_stack steps."""
    target._trace_stack = True
    target._trace_stack_depth = params.get("depth", 3)
    target._trace_stack_show_chain = params.get("show_decorator_chain", True)


def _after_track_changes(target: Any, params: dict[str, Any]) -> None:
    """Set convenience attributes after @track_changes steps."""
    target._tracked = True
    target._tracked_fields = params.get("fields")


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


trace_stack = make_decorator(
    name="trace_stack",
    steps=_trace_stack_steps,
    doc="""
    Enhanced error stack traces with decorator chain information.

    Args:
        depth: Number of stack frames to show (default: 3)
        show_decorator_chain: Whether to show applied decorators (default: True)

    Example:
        @trace_stack(depth=5, show_decorator_chain=True)
        def risky_system():
            ...
    """,
    validate=_validate_trace_stack,
    after_steps=_after_trace_stack,
)


track_changes = make_decorator(
    name="track_changes",
    steps=_track_changes_steps,
    doc="""
    Track changes to component fields for change detection.

    Requires @component decorator to be applied first.

    Args:
        fields: List of field names to track, or None to track all fields (default: None)

    Example:
        @component
        @track_changes(fields=["x", "y"])
        class Position:
            x: float
            y: float
            z: float  # not tracked
    """,
    validate=_validate_track_changes,
    after_steps=_after_track_changes,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================


_REGISTRY_ENTRIES = [
    ("reads", reads, ("function",), ()),
    ("writes", writes, ("function",), ()),
    ("trace_stack", trace_stack, ("function",), ()),
    ("track_changes", track_changes, ("class",), ("component",)),
]

for _name, _func, _targets, _requires in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        # Determine tier based on decorator name
        _tier = Tier.DEBUG_SAFETY if _name in ("reads", "writes", "trace_stack") else Tier.CHANGE_DETECTION

        _spec = DecoratorSpec(
            name=_name,
            tier=_tier,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
            requires=_requires,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[_tier].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Tier 10: DEBUG_SAFETY
    "reads",
    "writes",
    "trace_stack",
    # Tier 11: CHANGE_DETECTION
    "track_changes",
]
