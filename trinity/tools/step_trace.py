"""
Trinity Step Trace - show all Steps on a class grouped by layer.
"""
from __future__ import annotations
from typing import Any

from trinity.decorators.ops import Step


def trace(cls: type) -> str:
    """
    Return a formatted string showing all Steps on a class, grouped by layer.

    Layers:
    - Decorator: from cls._applied_steps
    - Descriptor: from cls._field_descriptors[field].descriptor_steps
    - Metaclass: from cls._metaclass_steps

    Args:
        cls: A Trinity class to trace.

    Returns:
        Formatted multi-line string.
    """
    lines = [f"=== Step Trace: {cls.__name__} ===", ""]

    # Layer 1: Decorator steps
    applied = getattr(cls, "_applied_steps", [])
    lines.append(f"[Decorator] ({len(applied)} steps)")
    if applied:
        for step in applied:
            lines.append(f"  {step.op.value}({_fmt_args(step.args)})")
    else:
        lines.append("  (none)")
    lines.append("")

    # Layer 2: Descriptor steps (per field)
    field_descriptors = getattr(cls, "_field_descriptors", {})
    desc_count = 0
    desc_lines: list[str] = []
    for field_name, desc in field_descriptors.items():
        chain = desc.get_chain() if hasattr(desc, "get_chain") else [desc]
        for d in chain:
            steps = getattr(d, "descriptor_steps", [])
            if steps:
                for step in steps:
                    desc_count += 1
                    desc_lines.append(
                        f"  {field_name}.{d.descriptor_id}: {step.op.value}({_fmt_args(step.args)})"
                    )
    lines.append(f"[Descriptor] ({desc_count} steps)")
    if desc_lines:
        lines.extend(desc_lines)
    else:
        lines.append("  (none)")
    lines.append("")

    # Layer 3: Metaclass steps
    meta_steps = getattr(cls, "_metaclass_steps", [])
    lines.append(f"[Metaclass] ({len(meta_steps)} steps)")
    if meta_steps:
        for step in meta_steps:
            lines.append(f"  {step.op.value}({_fmt_args(step.args)})")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


def _fmt_args(args: dict[str, Any]) -> str:
    """Format step args as key=value pairs."""
    if not args:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in args.items())
