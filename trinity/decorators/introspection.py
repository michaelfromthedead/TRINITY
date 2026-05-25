"""
Introspection API for the Trinity Pattern.

Provides functions documented in LANG_DEC.md for querying
decorator/descriptor/metaclass metadata on classes.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

from trinity.decorators.ops import (
    Op,
    Rule,
    RULES,
    Step,
    decompose,
    get_definitions,
    make_decorator,
    validate_steps,
)


def primitives(cls: type, field: Optional[str] = None) -> list[Step]:
    """Return primitive Steps for a class or field."""
    if field is None:
        return decompose(cls)
    descriptors = getattr(cls, "_field_descriptors", {})
    if field not in descriptors:
        return []
    return getattr(descriptors[field], "descriptor_steps", [])


def composites(cls: type, field: Optional[str] = None) -> list[str]:
    """Return composite decorator/descriptor names for a class or field."""
    if field is None:
        return getattr(cls, "_applied_decorators", []).copy()
    descriptors = getattr(cls, "_field_descriptors", {})
    if field not in descriptors:
        return []
    desc = descriptors[field]
    if hasattr(desc, "get_chain"):
        chain_items = desc.get_chain()
        return [getattr(d, "descriptor_id", type(d).__name__) for d in chain_items]
    return [getattr(desc, "descriptor_id", type(desc).__name__)]


def chain(cls: type, field: str) -> str:
    """Return human-readable descriptor chain explanation for a field."""
    descriptors = getattr(cls, "_field_descriptors", {})
    if field not in descriptors:
        return f"<{field}: no descriptor>"
    desc = descriptors[field]
    # Try DescriptorComposer.explain_chain if available
    try:
        from trinity.descriptors.composer import DescriptorComposer
        return DescriptorComposer.explain_chain(desc)
    except (ImportError, AttributeError, TypeError):
        return f"<{field}: {getattr(desc, 'descriptor_id', type(desc).__name__)}>"


def find_decorators(primitive: Op, **filters: Any) -> list[str]:
    """Find decorators that use a given Op, optionally filtered by args."""
    results = []
    for name, steps in get_definitions().items():
        for step in steps:
            if step.op == primitive:
                if not filters or all(step.args.get(k) == v for k, v in filters.items()):
                    results.append(name)
                    break
    return results


def compose(*steps: Step) -> Callable:
    """Create an anonymous decorator from Steps."""
    # 8 hex chars = 32 bits, sufficient for runtime uniqueness
    name = f"_composed_{uuid.uuid4().hex[:8]}"
    return make_decorator(name=name, steps=list(steps), doc="Dynamically composed decorator")


def validate_combination(steps: list[Step]) -> dict[str, Any]:
    """Validate that a combination of Steps is valid."""
    return validate_steps(steps)


def all_rules() -> list[Rule]:
    """Return all composition rules."""
    return list(RULES)


__all__ = [
    "primitives",
    "composites",
    "chain",
    "find_decorators",
    "compose",
    "validate_combination",
    "all_rules",
]
