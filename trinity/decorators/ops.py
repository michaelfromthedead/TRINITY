"""
The 7 Ops: the only things a decorator can do to a class.

Every decorator in the Trinity system is a named list of Steps.
Each Step is one Op with arguments. The Op functions do the real work.
Decorators are just configuration.

Ops:
    TAG         Attach queryable metadata
    HOOK        Wire a lifecycle callback
    REGISTER    Add target to a named registry
    DESCRIBE    Extract schema from annotations
    TRACK       Enable change monitoring / dirty flags
    VALIDATE    Enforce a constraint
    INTERCEPT   Wrap get/set/delete on fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union

T = TypeVar("T")


# =============================================================================
# OP ENUM
# =============================================================================


class Op(Enum):
    """The 7 operations. Everything a decorator can do."""

    TAG = "tag"
    HOOK = "hook"
    REGISTER = "register"
    DESCRIBE = "describe"
    TRACK = "track"
    VALIDATE = "validate"
    INTERCEPT = "intercept"


# =============================================================================
# STEP
# =============================================================================


@dataclass(frozen=True)
class Step:
    """
    One operation with its arguments. One step in a decorator's recipe.

    Examples:
        Step(Op.TAG, {"key": "pool", "value": {"size": 1024}})
        Step(Op.HOOK, {"event": "on_create", "callback": my_fn})
        Step(Op.REGISTER, {"registry": "PoolManager"})
        Step(Op.VALIDATE, {"constraint": "budget_limit"})
    """

    op: Op
    args: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        if not self.args:
            return self.op.value.upper()
        args_str = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.op.value.upper()}({args_str})"


# =============================================================================
# HOOK EVENTS (common lifecycle events)
# =============================================================================


class HookEvent(Enum):
    """Standard lifecycle events."""

    ON_CREATE = "on_create"
    ON_DESTROY = "on_destroy"
    ON_CHANGE = "on_change"
    ON_ACCESS = "on_access"
    ON_SERIALIZE = "on_serialize"
    ON_DESERIALIZE = "on_deserialize"
    ON_ATTACH = "on_attach"
    ON_DETACH = "on_detach"
    ON_ENTER = "on_enter"
    ON_EXIT = "on_exit"


# =============================================================================
# OP FUNCTIONS — the real implementation
# =============================================================================


def run_tag(target: Any, key: str, value: Any = True) -> Any:
    """TAG: attach queryable metadata."""
    if not hasattr(target, "_tags"):
        target._tags = {}
    target._tags[key] = value
    return target


def run_hook(
    target: Any, event: Union[HookEvent, str], callback: Optional[Callable] = None
) -> Any:
    """HOOK: wire a lifecycle callback."""
    if not hasattr(target, "_hooks"):
        target._hooks = {}
    event_key = event.value if isinstance(event, HookEvent) else event
    if event_key not in target._hooks:
        target._hooks[event_key] = []
    if callback is not None:
        target._hooks[event_key].append(callback)
    return target


def run_register(target: Any, registry_name: str) -> Any:
    """REGISTER: add target to a named registry."""
    if not hasattr(target, "_registries"):
        target._registries = []
    if registry_name not in target._registries:
        target._registries.append(registry_name)
    return target


def run_describe(target: Any) -> Any:
    """DESCRIBE: extract schema from type annotations."""
    annotations = {}
    if isinstance(target, type):
        try:
            from typing import get_type_hints

            annotations = {
                k: v
                for k, v in get_type_hints(target, include_extras=True).items()
                if not k.startswith("_")
            }
        except Exception:
            annotations = {
                k: v
                for k, v in getattr(target, "__annotations__", {}).items()
                if not k.startswith("_")
            }
    target._schema = annotations
    target._described = True
    return target


def run_track(target: Any, field_name: Optional[str] = None) -> Any:
    """TRACK: enable change monitoring."""
    if not hasattr(target, "_tracked_fields"):
        target._tracked_fields = set()
    if field_name:
        target._tracked_fields.add(field_name)
    else:
        target._tracked = True
    return target


def run_validate(target: Any, constraint: str, field_name: Optional[str] = None) -> Any:
    """VALIDATE: enforce a constraint."""
    if not hasattr(target, "_constraints"):
        target._constraints = []
    entry: dict[str, str] = {"constraint": constraint}
    if field_name:
        entry["field"] = field_name
    target._constraints.append(entry)
    return target


def run_intercept(
    target: Any,
    get: Optional[str] = None,
    set: Optional[str] = None,
    delete: Optional[str] = None,
) -> Any:
    """INTERCEPT: wrap field access."""
    if not hasattr(target, "_intercepts"):
        target._intercepts = []
    intercept: dict[str, str] = {}
    if get is not None:
        intercept["get"] = get
    if set is not None:
        intercept["set"] = set
    if delete is not None:
        intercept["delete"] = delete
    target._intercepts.append(intercept)
    return target


# Dispatch table
_OP_RUNNERS = {
    Op.TAG: lambda target, args: run_tag(
        target, args.get("key", ""), args.get("value", True)
    ),
    Op.HOOK: lambda target, args: run_hook(
        target, args.get("event", ""), args.get("callback")
    ),
    Op.REGISTER: lambda target, args: run_register(target, args.get("registry", "")),
    Op.DESCRIBE: lambda target, args: run_describe(target),
    Op.TRACK: lambda target, args: run_track(target, args.get("field")),
    Op.VALIDATE: lambda target, args: run_validate(
        target, args.get("constraint", ""), args.get("field")
    ),
    Op.INTERCEPT: lambda target, args: run_intercept(
        target, get=args.get("get"), set=args.get("set"), delete=args.get("delete")
    ),
}


def run_step(target: Any, step: Step) -> Any:
    """Execute one Step on a target."""
    runner = _OP_RUNNERS.get(step.op)
    if runner:
        return runner(target, step.args)
    return target


def run_steps(target: Any, steps: list[Step]) -> Any:
    """Execute a list of Steps on a target. Records what was applied."""
    if not hasattr(target, "_applied_steps"):
        target._applied_steps = []
    for step in steps:
        target = run_step(target, step)
        target._applied_steps.append(step)
    return target


# =============================================================================
# COMPOSITION RULES
# =============================================================================


@dataclass
class Rule:
    """A validation rule for step combinations."""

    name: str
    when: Callable[[list[Step]], bool]
    requires: Optional[Callable[[list[Step]], bool]] = None
    conflicts: Optional[Callable[[list[Step]], bool]] = None


def _has_op(steps: list[Step], op: Op, **kwargs: Any) -> bool:
    """Check if steps contain an op, optionally with specific args."""
    for s in steps:
        if s.op == op:
            if not kwargs:
                return True
            if all(s.args.get(k) == v for k, v in kwargs.items()):
                return True
    return False


RULES: list[Rule] = [
    Rule(
        name="HOOK(on_change) requires TRACK",
        when=lambda steps: (
            _has_op(steps, Op.HOOK, event=HookEvent.ON_CHANGE)
            or _has_op(steps, Op.HOOK, event="on_change")
        ),
        requires=lambda steps: _has_op(steps, Op.TRACK),
    ),
    Rule(
        name="INTERCEPT(set=deny) conflicts with TRACK",
        when=lambda steps: _has_op(steps, Op.INTERCEPT, set="deny"),
        conflicts=lambda steps: _has_op(steps, Op.TRACK),
    ),
    Rule(
        name="REGISTER should be applied last",
        when=lambda steps: _has_op(steps, Op.REGISTER),
        requires=lambda steps: (
            not steps or not any(
                steps[j].op != Op.REGISTER
                for j in range(
                    next((i for i, s in enumerate(steps) if s.op == Op.REGISTER), len(steps)),
                    len(steps),
                )
            )
        ),
    ),
    Rule(
        name="TAG(network) requires TAG(serialization)",
        when=lambda steps: _has_op(steps, Op.TAG, key="networked"),
        requires=lambda steps: _has_op(steps, Op.TAG, key="serialization_format") or _has_op(steps, Op.TAG, key="serializable"),
    ),
    Rule(
        name="INTERCEPT(set=deny) conflicts with VALIDATE",
        when=lambda steps: _has_op(steps, Op.INTERCEPT, set="deny"),
        conflicts=lambda steps: _has_op(steps, Op.VALIDATE),
    ),
]


def validate_steps(
    steps: list[Step], rules: Optional[list[Rule]] = None, check_ordering: bool = False
) -> dict[str, Any]:
    """Check if a combination of steps is valid."""
    if rules is None:
        rules = RULES
    errors: list[str] = []
    for rule in rules:
        if not rule.when(steps):
            continue
        if rule.requires is not None and not rule.requires(steps):
            errors.append(rule.name)
        if rule.conflicts is not None and rule.conflicts(steps):
            errors.append(rule.name)
    if check_ordering:
        ordering_result = validate_ordering(steps)
        if not ordering_result["valid"]:
            errors.extend(ordering_result["errors"])
    result: dict[str, Any] = {"valid": len(errors) == 0}
    if errors:
        result["errors"] = errors
    return result



_OP_ORDER = {Op.TAG: 0, Op.VALIDATE: 1, Op.TRACK: 2, Op.INTERCEPT: 3, Op.HOOK: 4, Op.DESCRIBE: 5, Op.REGISTER: 6}


def validate_ordering(steps: list[Step]) -> dict[str, Any]:
    """Check if Steps follow canonical ordering: TAG -> VALIDATE -> TRACK -> INTERCEPT -> HOOK -> DESCRIBE -> REGISTER."""
    errors = []
    last_order = -1
    last_step_idx = -1
    for i, step in enumerate(steps):
        order = _OP_ORDER.get(step.op, -1)
        if order < last_order:
            errors.append(
                f"{step.op.value.upper()} (position {i}) should come before "
                f"{steps[last_step_idx].op.value.upper()} (position {last_step_idx})"
            )
        if order > last_order:
            last_order = order
            last_step_idx = i
    result: dict[str, Any] = {"valid": len(errors) == 0}
    if errors:
        result["errors"] = errors
    return result


# =============================================================================
# DECORATOR DEFINITION REGISTRY
# =============================================================================

# Global: decorator name -> list of Steps
_definitions: dict[str, list[Step]] = {}


def get_definitions() -> dict[str, list[Step]]:
    """Return a copy of the decorator definitions registry."""
    return dict(_definitions)


def _collect_descriptor_steps(cls: type) -> list[Step]:
    """Iterate class _field_descriptors and collect Steps from each descriptor."""
    steps: list[Step] = []
    for field_name, desc in getattr(cls, "_field_descriptors", {}).items():
        for step in getattr(desc, "descriptor_steps", []):
            steps.append(step)
    return steps


def decompose(
    target: Any,
    include_metaclass: bool = True,
    include_descriptors: bool = True,
) -> list[Step]:
    """Return the step list of a decorator or all steps from a class.

    If *target* is a class, steps are collected from all three Trinity layers:
    decorator (_applied_steps), metaclass (_metaclass_steps), and descriptor
    (descriptor_steps on each field descriptor).

    If *target* is a decorator (has ``_steps`` or is registered), return its
    step list as before.
    """
    if isinstance(target, type):
        combined: list[Step] = list(getattr(target, "_applied_steps", []))
        if include_metaclass:
            combined.extend(getattr(target, "_metaclass_steps", []))
        if include_descriptors:
            combined.extend(_collect_descriptor_steps(target))
        return combined

    # Decorator path (backward compatible)
    steps = getattr(target, "_steps", None)
    if steps is not None:
        return list(steps)
    name = getattr(target, "__name__", str(target))
    return list(_definitions.get(name, []))


def decompose_layered(cls: type) -> dict[str, list[Step]]:
    """Return Steps grouped by layer.

    For non-class targets, only the ``"decorators"`` key has data.
    """
    if not isinstance(cls, type):
        return {
            "decorators": decompose(cls),
            "metaclass": [],
            "descriptors": [],
        }
    return {
        "decorators": list(getattr(cls, "_applied_steps", [])),
        "metaclass": list(getattr(cls, "_metaclass_steps", [])),
        "descriptors": _collect_descriptor_steps(cls),
    }


def expand(target: Any) -> str:
    """Human-readable expansion of a decorator or class to its steps.

    For classes, shows steps grouped by layer. For decorators, shows a flat
    single-line expansion.
    """
    if isinstance(target, type):
        layered = decompose_layered(target)
        lines: list[str] = []
        for label, key in [
            ("Decorators", "decorators"),
            ("Metaclass", "metaclass"),
            ("Descriptors", "descriptors"),
        ]:
            steps = layered[key]
            if steps:
                lines.append(f"[{label}]  " + " + ".join(repr(s) for s in steps))
        if not lines:
            name = getattr(target, "__name__", str(target))
            return f"<{name}: no steps defined>"
        return "\n".join(lines)

    # Decorator path (backward compatible)
    steps = decompose(target)
    if not steps:
        name = getattr(target, "__name__", str(target))
        return f"<{name}: no steps defined>"
    return " + ".join(repr(s) for s in steps)


# =============================================================================
# make_decorator — THE factory
# =============================================================================


def make_decorator(
    name: str,
    steps: Union[list[Step], Callable[..., list[Step]]],
    doc: str = "",
    validate: Optional[Callable[..., None]] = None,
    after_steps: Optional[Callable[[Any, dict[str, Any]], Any]] = None,
):
    """
    Create a decorator from a list of Steps.

    This is the only way to create decorators in the new architecture.
    The returned function, when called with params, returns a decorator
    that runs the steps against the target.

    Args:
        name: Decorator name (e.g., "pooled", "native").
        steps: Either a static list of Steps, or a callable that takes
               a params dict and returns Steps.
        doc: Docstring for the decorator.
        validate: Optional function(params) that raises on invalid params.
                  Called before steps are built.
        after_steps: Optional function(target, params) for domain-specific
                     behavior that can't be expressed as steps (e.g., adding
                     methods, wrapping __init__). Called after all steps run.

    Returns:
        A decorator factory (callable that accepts params and returns a decorator).
    """

    if callable(steps) and not isinstance(steps, list):
        # Parameterized: steps is a function that builds steps from params
        step_builder = steps
        is_static = False
    else:
        # Static: steps is a fixed list
        static_steps = steps
        step_builder = lambda params: static_steps
        is_static = True

    def decorator_factory(**params: Any) -> Callable[[T], T]:
        # Validate params if validator provided
        if validate is not None:
            validate(**params)

        def decorator(target: T) -> T:
            # Build steps (may depend on params)
            built_steps = step_builder(params)

            # Run all steps -- this is where the real work happens
            run_steps(target, built_steps)

            # Track which decorator was applied
            if not hasattr(target, "_applied_decorators"):
                target._applied_decorators = []
            if name not in target._applied_decorators:
                target._applied_decorators.append(name)

            # Domain-specific post-processing
            if after_steps is not None:
                result = after_steps(target, params)
                if result is not None:
                    target = result

            return target

        decorator.__name__ = name
        decorator.__qualname__ = name
        decorator._decorator_name = name
        return decorator

    # For no-arg decorators: @unsafe directly on a class (not @unsafe())
    def direct_or_factory(*args: Any, **kwargs: Any) -> Any:
        if (
            len(args) == 1
            and len(kwargs) == 0
            and (isinstance(args[0], type) or callable(args[0]))
        ):
            # Called as @decorator (no parens) -- apply directly with default params
            return decorator_factory()(args[0])
        else:
            # Called as @decorator(...) -- return the inner decorator
            return decorator_factory(*args, **kwargs)

    # Store metadata for introspection
    if is_static:
        direct_or_factory._steps = static_steps
        _definitions[name] = static_steps
    else:
        # For parameterized decorators, store steps built with empty/default params
        try:
            default_steps = step_builder({})
            direct_or_factory._steps = default_steps
            _definitions[name] = default_steps
        except (TypeError, KeyError):
            direct_or_factory._steps = []
            _definitions[name] = []

    direct_or_factory.__name__ = name
    direct_or_factory.__qualname__ = name
    direct_or_factory.__doc__ = doc
    direct_or_factory._is_decorator = True
    direct_or_factory._decorator_name = name

    return direct_or_factory


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core types
    "Op",
    "Step",
    "HookEvent",
    # Op functions
    "run_tag",
    "run_hook",
    "run_register",
    "run_describe",
    "run_track",
    "run_validate",
    "run_intercept",
    "run_step",
    "run_steps",
    # Factory
    "make_decorator",
    # Registry access
    "get_definitions",
    # Introspection
    "decompose",
    "decompose_layered",
    "expand",
    # Validation
    "Rule",
    "RULES",
    "validate_steps",
    "validate_ordering",
]
