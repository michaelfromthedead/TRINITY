"""
Trinity Op Coverage - analyze Op usage across all registered classes.
"""
from __future__ import annotations
from typing import Any

from trinity.decorators.ops import Op, decompose


def op_coverage() -> dict[str, Any]:
    """
    Analyze Op usage across all registered Trinity classes.

    Returns:
        Dict with:
        - "op_counts": dict[str, int] — Op name -> number of classes using it
        - "zero_step_classes": list[str] — class names with no Steps at all
        - "total_classes": int
        - "total_steps": int
        - "coverage": dict[str, list[str]] — Op name -> list of class names using it
    """
    from trinity.metaclasses.engine_meta import EngineMeta

    op_counts: dict[str, int] = {op.value: 0 for op in Op}
    coverage: dict[str, list[str]] = {op.value: [] for op in Op}
    zero_step_classes: list[str] = []
    total_steps = 0

    for cls in EngineMeta._all_engine_types.values():
        name = getattr(cls, "__qualname__", cls.__name__)
        steps = decompose(cls)

        if not steps:
            zero_step_classes.append(name)
            continue

        total_steps += len(steps)
        seen_ops: set[str] = set()
        for step in steps:
            op_name = step.op.value
            if op_name not in seen_ops:
                op_counts[op_name] += 1
                coverage[op_name].append(name)
                seen_ops.add(op_name)

    return {
        "op_counts": op_counts,
        "zero_step_classes": zero_step_classes,
        "total_classes": len(EngineMeta._all_engine_types),
        "total_steps": total_steps,
        "coverage": coverage,
    }
