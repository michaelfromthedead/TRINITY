"""
ECS Core decorators — built from Ops.

Foundation decorators for the Entity-Component-System pattern.
Every decorator here is a named list of Steps, created by make_decorator.

Decorators:
    @component  - Marks class as ECS component
    @tag        - Zero-sized component for filtering
    @resource   - Singleton resource
    @event      - Event type for event bus
    @system     - Marks function as ECS system
    @query      - Declarative query definition
    @bundle     - Components that spawn together
    @relation   - Entity-to-entity relationships
    @derived    - Computed components
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from trinity.constants import DEFAULT_RESOURCE_PRIORITY
from trinity.decorators.base import check_excluded_decorators as validate_not_combined
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# HELPERS (shared logic, not decorators)
# =============================================================================


def _extract_queries(fn: Callable[..., Any]) -> list[Any]:
    """Extract Query types from function signature."""
    queries = []
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    for param_name, param_type in hints.items():
        if param_name == "return":
            continue
        origin = get_origin(param_type)
        if (
            origin is not None
            and hasattr(origin, "__name__")
            and origin.__name__ == "Query"
        ):
            queries.append(param_type)
        elif hasattr(param_type, "__name__") and param_type.__name__ == "Query":
            queries.append(param_type)
    return queries


def _extract_resources(fn: Callable[..., Any]) -> list[Any]:
    """Extract Res types from function signature."""
    resources = []
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    for param_name, param_type in hints.items():
        if param_name == "return":
            continue
        origin = get_origin(param_type)
        if (
            origin is not None
            and hasattr(origin, "__name__")
            and origin.__name__ == "Res"
        ):
            args = get_args(param_type)
            if args:
                resources.append(args[0])
        elif hasattr(param_type, "__name__") and param_type.__name__ == "Res":
            resources.append(param_type)
    return resources


def _map_phase_string(phase: str) -> Any:
    """Map string phase name to SystemPhase enum."""
    from trinity.types import SystemPhase

    phase_map = {
        "pre_physics": SystemPhase.PRE_PHYSICS,
        "physics": SystemPhase.PHYSICS,
        "post_physics": SystemPhase.POST_PHYSICS,
        "pre_update": SystemPhase.PRE_UPDATE,
        "update": SystemPhase.UPDATE,
        "post_update": SystemPhase.POST_UPDATE,
        "pre_render": SystemPhase.PRE_RENDER,
        "render": SystemPhase.RENDER,
    }
    return phase_map.get(phase.lower(), SystemPhase.UPDATE)


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _component_steps(params: dict[str, Any]) -> list[Step]:
    name = params.get("name")
    return [
        Step(Op.TAG, {"key": "component", "value": True}),
        Step(Op.TAG, {"key": "component_name", "value": name}),  # resolved in after
        Step(Op.REGISTER, {"registry": "ecs_core"}),
    ]


def _tag_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "component", "value": True}),
        Step(Op.TAG, {"key": "tag", "value": True}),
        Step(Op.REGISTER, {"registry": "ecs_core"}),
    ]


def _resource_steps(params: dict[str, Any]) -> list[Step]:
    name = params.get("name")
    return [
        Step(Op.TAG, {"key": "resource", "value": True}),
        Step(Op.TAG, {"key": "resource_name", "value": name}),  # resolved in after
        Step(Op.REGISTER, {"registry": "ecs_core"}),
    ]


def _event_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "event", "value": True}),
        Step(Op.REGISTER, {"registry": "ecs_core"}),
    ]


def _system_steps(params: dict[str, Any]) -> list[Step]:
    phase = params.get("phase", "update")
    return [
        Step(Op.TAG, {"key": "system", "value": True}),
        Step(Op.TAG, {"key": "system_phase", "value": phase}),
        Step(Op.REGISTER, {"registry": "ecs_core"}),
    ]


def _query_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "query", "value": True}),
        Step(
            Op.TAG, {"key": "query_components", "value": params.get("components", ())}
        ),
        Step(Op.TAG, {"key": "query_with", "value": params.get("with_", ())}),
        Step(Op.TAG, {"key": "query_without", "value": params.get("without", ())}),
        Step(Op.TAG, {"key": "query_maybe", "value": params.get("maybe", ())}),
        Step(Op.REGISTER, {"registry": "ecs_core"}),
    ]


def _bundle_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "bundle", "value": True}),
        Step(Op.DESCRIBE, {}),
        Step(Op.REGISTER, {"registry": "ecs_core"}),
    ]


def _relation_steps(params: dict[str, Any]) -> list[Step]:
    kind = params.get("kind", "one_to_many")
    exclusive = params.get("exclusive", False)
    return [
        Step(Op.TAG, {"key": "relation", "value": True}),
        Step(Op.TAG, {"key": "relation_kind", "value": kind}),
        Step(Op.TAG, {"key": "relation_exclusive", "value": exclusive}),
        Step(Op.REGISTER, {"registry": "ecs_core"}),
    ]


def _derived_steps(params: dict[str, Any]) -> list[Step]:
    from_components = params.get("from_components", ())
    cache = params.get("cache", True)
    return [
        Step(Op.TAG, {"key": "derived", "value": True}),
        Step(Op.TAG, {"key": "derived_from", "value": from_components}),
        Step(Op.TAG, {"key": "derived_cache", "value": cache}),
        Step(Op.REGISTER, {"registry": "ecs_core"}),
    ]


# =============================================================================
# AFTER-STEPS (domain behavior — metaclass registration, etc.)
# =============================================================================


def _after_component(target: Any, params: dict[str, Any]) -> Any:
    validate_not_combined(target, "component", ("resource", "event"))

    name = params.get("name") or target.__name__
    target._component = True
    target._component_name = name

    if not hasattr(target, "_component_id"):
        from trinity.metaclasses import ComponentMeta

        if target.__name__ != "Component":
            # Initialize _metaclass_steps if not present (class was not created with ComponentMeta)
            if not hasattr(target, "_metaclass_steps"):
                target._metaclass_steps = []
            ComponentMeta._process_fields(target)
            ComponentMeta._install_descriptors(target)
            with ComponentMeta._lock:
                target._component_id = ComponentMeta._next_id
                ComponentMeta._next_id += 1
                ComponentMeta._registry[target._component_id] = target
                ComponentMeta._name_to_id[target._component_name] = target._component_id

    return None


def _after_tag(target: Any, params: dict[str, Any]) -> Any:
    validate_not_combined(target, "tag", ("resource", "event", "serializable"))

    target._component = True
    target._component_name = target.__name__
    target._tag = True
    target._field_types = {}
    target._field_defaults = {}
    target._field_descriptors = {}
    target._field_offsets = {}

    from trinity.metaclasses import ComponentMeta

    with ComponentMeta._lock:
        target._component_id = ComponentMeta._next_id
        ComponentMeta._next_id += 1
        ComponentMeta._registry[target._component_id] = target
        ComponentMeta._name_to_id[target._component_name] = target._component_id

    return None


def _after_resource(target: Any, params: dict[str, Any]) -> Any:
    validate_not_combined(target, "resource", ("component", "tag", "event"))

    name = params.get("name") or target.__name__
    target._resource = True
    target._resource_name = name

    if not hasattr(target, "_resource_id"):
        from trinity.metaclasses import ResourceMeta

        with ResourceMeta._lock:
            target._resource_id = ResourceMeta._next_id
            ResourceMeta._next_id += 1
            target._resource_priority = getattr(
                target, "_resource_priority", DEFAULT_RESOURCE_PRIORITY
            )
            target._resource_dependencies = getattr(
                target, "_resource_dependencies", ()
            )
            ResourceMeta._registry[target._resource_id] = target

    return None


def _after_event(target: Any, params: dict[str, Any]) -> Any:
    validate_not_combined(target, "event", ("component", "tag", "resource"))

    target._event = True
    target._event_name = f"{target.__module__}.{target.__name__}"

    if not hasattr(target, "_event_id"):
        from trinity.metaclasses import EventMeta

        try:
            annotations = get_type_hints(target)
        except Exception:
            annotations = getattr(target, "__annotations__", {})

        target._event_fields = {
            name: typ for name, typ in annotations.items() if not name.startswith("_")
        }

        with EventMeta._lock:
            target._event_id = EventMeta._next_id
            EventMeta._next_id += 1
            target._event_priority = getattr(target, "_event_priority", 0)
            target._event_channels = getattr(target, "_event_channels", ())
            target._event_pooled = getattr(target, "_event_pooled", False)
            target._event_parent_ids = ()
            EventMeta._registry[target._event_id] = target
            EventMeta._name_to_id[target._event_name] = target._event_id

    return None


def _after_system(target: Any, params: dict[str, Any]) -> Any:
    phase = params.get("phase", "update")
    target._system = True
    target._system_phase = phase
    target._system_queries = _extract_queries(target)
    target._system_resources = _extract_resources(target)

    if not hasattr(target, "_reads"):
        target._reads = ()
    if not hasattr(target, "_writes"):
        target._writes = ()
    if not hasattr(target, "_exclusive"):
        target._exclusive = False
    if not hasattr(target, "_priority"):
        target._priority = 0

    if isinstance(target, type):
        from trinity.metaclasses import SystemMeta
        from trinity.types import SystemPhase

        phase_enum = _map_phase_string(phase)
        target._system_phase = phase_enum

        with SystemMeta._lock:
            if not hasattr(target, "_system_id"):
                target._system_id = SystemMeta._next_id
                SystemMeta._next_id += 1
                target._system_name = f"{target.__module__}.{target.__name__}"
                target._dependencies = set()
                target._can_parallelize = True
                SystemMeta._registry[target._system_id] = target
                SystemMeta._phases[phase_enum].append(target._system_id)

    return None


def _after_query(target: Any, params: dict[str, Any]) -> Any:
    target._query = True
    target._query_components = params.get("components", ())
    target._query_with = params.get("with_", ())
    target._query_without = params.get("without", ())
    target._query_maybe = params.get("maybe", ())
    return None


def _after_bundle(target: Any, params: dict[str, Any]) -> Any:
    target._bundle = True

    try:
        annotations = get_type_hints(target)
    except Exception:
        annotations = getattr(target, "__annotations__", {})

    target._bundle_components = {
        name: typ for name, typ in annotations.items() if not name.startswith("_")
    }
    return None


def _after_relation(target: Any, params: dict[str, Any]) -> Any:
    kind = params.get("kind", "one_to_many")
    exclusive = params.get("exclusive", False)

    # Ensure @component is applied
    if not hasattr(target, "_component"):
        target = component()(target)

    target._relation = True
    target._relation_kind = kind
    target._relation_exclusive = exclusive
    return None


def _after_derived(target: Any, params: dict[str, Any]) -> Any:
    from_components = params.get("from_components", ())
    cache = params.get("cache", True)

    # Ensure @component is applied
    if not hasattr(target, "_component"):
        target = component()(target)

    target._derived = True
    target._derived_from = from_components
    target._derived_cache = cache

    if not hasattr(target, "compute"):
        import warnings

        warnings.warn(
            f"{target.__name__}: @derived components should define a static 'compute' method.",
            UserWarning,
            stacklevel=5,
        )

    return None


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_query(
    components: tuple = (),
    with_: tuple = (),
    without: tuple = (),
    maybe: tuple = (),
    **_: Any,
) -> None:
    pass  # No validation needed beyond type checks


def _validate_relation(
    kind: str = "one_to_many",
    exclusive: bool = False,
    **_: Any,
) -> None:
    valid_kinds = {"one_to_one", "one_to_many", "many_to_many"}
    if kind not in valid_kinds:
        raise ValueError(
            f"@relation: invalid kind '{kind}'. Valid: {sorted(valid_kinds)}"
        )


def _validate_derived(
    from_components: tuple = (),
    cache: bool = True,
    **_: Any,
) -> None:
    if not from_components:
        raise ValueError("@derived: 'from_components' is required")


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

# component: supports @component and @component(name="...")
# make_decorator handles both no-arg and parameterized usage
component = make_decorator(
    name="component",
    steps=_component_steps,
    doc="Mark a class as an ECS component.",
    after_steps=_after_component,
)

tag = make_decorator(
    name="tag",
    steps=_tag_steps,
    doc="Zero-sized component for filtering (no data).",
    after_steps=_after_tag,
)

resource = make_decorator(
    name="resource",
    steps=_resource_steps,
    doc="Singleton resource (one instance per World).",
    after_steps=_after_resource,
)

event = make_decorator(
    name="event",
    steps=_event_steps,
    doc="Event type for the event bus.",
    after_steps=_after_event,
)

system = make_decorator(
    name="system",
    steps=_system_steps,
    doc="Marks function as ECS system.",
    after_steps=_after_system,
)

query = make_decorator(
    name="query",
    steps=_query_steps,
    doc="Declarative query definition.",
    validate=_validate_query,
    after_steps=_after_query,
)

bundle = make_decorator(
    name="bundle",
    steps=_bundle_steps,
    doc="Components that spawn together.",
    after_steps=_after_bundle,
)

relation = make_decorator(
    name="relation",
    steps=_relation_steps,
    doc="Entity-to-entity relationships.",
    validate=_validate_relation,
    after_steps=_after_relation,
)

derived = make_decorator(
    name="derived",
    steps=_derived_steps,
    doc="Computed component that auto-updates.",
    validate=_validate_derived,
    after_steps=_after_derived,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, dict[str, Any]]] = [
    (
        "component",
        component,
        {"unique": True, "foundation": True, "excludes": ("resource", "event")},
    ),
    (
        "tag",
        tag,
        {
            "unique": True,
            "foundation": True,
            "excludes": ("resource", "event", "serializable"),
        },
    ),
    (
        "resource",
        resource,
        {"unique": True, "foundation": True, "excludes": ("component", "tag", "event")},
    ),
    (
        "event",
        event,
        {
            "unique": True,
            "foundation": True,
            "excludes": ("component", "tag", "resource"),
        },
    ),
    ("system", system, {"unique": True, "foundation": True}),
    ("query", query, {}),
    ("bundle", bundle, {"unique": True}),
    ("relation", relation, {"requires": ("component",)}),
    ("derived", derived, {"requires": ("component",)}),
]

for _name, _func, _extra in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.ECS_CORE,
            func=_func,
            unique=_extra.get("unique", False),
            foundation=_extra.get("foundation", False),
            requires=_extra.get("requires", ()),
            excludes=_extra.get("excludes", ()),
            doc=getattr(_func, "__doc__", ""),
            target_types=("class",) if _name != "system" else ("function", "class"),
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.ECS_CORE].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "component",
    "tag",
    "resource",
    "event",
    "system",
    "query",
    "bundle",
    "relation",
    "derived",
]
