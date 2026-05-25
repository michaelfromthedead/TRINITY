"""
Trinity Doctor - validate all registered classes against composition rules.
"""
from __future__ import annotations
from typing import Any

from trinity.decorators.ops import decompose, validate_steps


def doctor() -> dict[str, Any]:
    """
    Validate all registered Trinity classes against composition rules.

    Returns:
        Dict with:
        - "total": int — number of classes checked
        - "passed": int — classes with no errors
        - "failed": int — classes with errors
        - "errors": dict[str, list[str]] — class name → list of error messages
    """
    # Import here to avoid circular imports
    from trinity.metaclasses.engine_meta import EngineMeta

    errors: dict[str, list[str]] = {}
    total = 0

    for cls in EngineMeta._all_engine_types.values():
        total += 1
        name = getattr(cls, "__qualname__", cls.__name__)
        steps = decompose(cls)
        result = validate_steps(steps)
        error_list = result.get("errors", [])
        if error_list:
            errors[name] = error_list

    return {
        "total": total,
        "passed": total - len(errors),
        "failed": len(errors),
        "errors": errors,
    }
