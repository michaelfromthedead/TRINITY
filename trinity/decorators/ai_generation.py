"""AI Generation decorators (Tier 9).

Provides decorators for AI-assisted code generation, patterns, and complexity annotations.
All decorators use the ops-based system from trinity.decorators.ops.
"""

from __future__ import annotations

from typing import Any, Literal

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

__all__ = [
    "example",
    "constraints",
    "stub",
    "pattern",
    "complexity",
    "generates",
    "pure",
    "VALID_PATTERN_CATEGORIES",
]

# Valid pattern categories
VALID_PATTERN_CATEGORIES = frozenset({"creational", "behavioral", "structural"})


# ============================================================================
# Step Builder Functions
# ============================================================================


def _example_steps(params):
    """Build steps for example decorator."""
    return [
        Step(Op.TAG, {"key": "example", "value": True}),
        Step(Op.TAG, {"key": "example_config", "value": {
            "inputs": params.get("inputs", {}),
            "output": params.get("output"),
            "description": params.get("description", ""),
        }}),
        Step(Op.REGISTER, {"registry": "ai_generation"}),
    ]


def _constraints_steps(params):
    """Build steps for constraints decorator."""
    return [
        Step(Op.TAG, {"key": "constraints", "value": True}),
        Step(Op.TAG, {"key": "constraint_rules", "value": params.get("rules", [])}),
        Step(Op.REGISTER, {"registry": "ai_generation"}),
    ]


def _stub_steps(params):
    """Build steps for stub decorator."""
    return [
        Step(Op.TAG, {"key": "stub", "value": True}),
        Step(Op.TAG, {"key": "stub_config", "value": {
            "signature_only": params.get("signature_only", True),
            "hints": params.get("implementation_hints", []),
        }}),
        Step(Op.REGISTER, {"registry": "ai_generation"}),
    ]


def _pattern_steps(params):
    """Build steps for pattern decorator."""
    return [
        Step(Op.TAG, {"key": "pattern", "value": True}),
        Step(Op.TAG, {"key": "pattern_config", "value": {
            "name": params.get("name", ""),
            "category": params.get("category", ""),
        }}),
        Step(Op.REGISTER, {"registry": "ai_generation"}),
    ]


def _complexity_steps(params):
    """Build steps for complexity decorator."""
    return [
        Step(Op.TAG, {"key": "complexity", "value": True}),
        Step(Op.TAG, {"key": "complexity_config", "value": {
            "time": params.get("time", ""),
            "space": params.get("space", ""),
        }}),
        Step(Op.REGISTER, {"registry": "ai_generation"}),
    ]


def _generates_steps(params):
    """Build steps for generates decorator."""
    return [
        Step(Op.TAG, {"key": "generates", "value": True}),
        Step(Op.TAG, {"key": "generates_config", "value": {
            "output_type": params.get("output_type"),
            "count": params.get("count", 1),
        }}),
        Step(Op.REGISTER, {"registry": "ai_generation"}),
    ]


def _pure_steps(params):
    """Build steps for pure decorator."""
    return [
        Step(Op.TAG, {"key": "pure", "value": True}),
        Step(Op.REGISTER, {"registry": "ai_generation"}),
    ]


# ============================================================================
# After Functions
# ============================================================================


def _after_example(target: Any, params: dict[str, Any]) -> None:
    """Accumulate examples on target."""
    if not hasattr(target, "_examples"):
        target._examples = []
    target._examples.append({
        "inputs": params.get("inputs", {}),
        "output": params.get("output"),
        "description": params.get("description", ""),
    })
    return None


def _after_constraints(target: Any, params: dict[str, Any]) -> None:
    """Accumulate constraints on target."""
    rules = params.get("rules", [])
    if not hasattr(target, "_constraints"):
        target._constraints = []
    target._constraints.extend(rules)
    return None


def _after_stub(target: Any, params: dict[str, Any]) -> None:
    """Mark target as stub and store hints."""
    target._stub = True
    target._stub_signature_only = params.get("signature_only", True)
    target._stub_hints = list(params.get("implementation_hints", []))
    return None


def _after_pattern(target: Any, params: dict[str, Any]) -> None:
    """Mark target as pattern and store details."""
    target._pattern = True
    target._pattern_name = params.get("name", "")
    target._pattern_category = params.get("category", "")
    return None


def _after_complexity(target: Any, params: dict[str, Any]) -> None:
    """Store complexity annotations."""
    target._complexity = True
    target._complexity_time = params.get("time", "")
    target._complexity_space = params.get("space", "")
    return None


def _after_generates(target: Any, params: dict[str, Any]) -> None:
    """Store generation metadata."""
    target._generates = True
    target._generates_output_type = params.get("output_type")
    target._generates_count = params.get("count", 1)
    return None


def _after_pure(target: Any, params: dict[str, Any]) -> None:
    """Mark target as pure function."""
    target._pure = True
    return None


# ============================================================================
# Validation Functions
# ============================================================================


def _validate_example(**kwargs):
    """Validate example parameters."""
    inputs = kwargs.get("inputs")
    if inputs is None or not isinstance(inputs, dict):
        raise ValueError("inputs must be a dict")
    if "output" not in kwargs:
        raise ValueError("output is required")


def _validate_constraints(**kwargs):
    """Validate constraints parameters."""
    rules = kwargs.get("rules", [])
    if not rules or not isinstance(rules, list):
        raise ValueError("rules must be a non-empty list")


def _validate_stub(**kwargs):
    """Validate stub parameters."""
    signature_only = kwargs.get("signature_only", True)
    if not isinstance(signature_only, bool):
        raise ValueError("signature_only must be a bool")

    hints = kwargs.get("implementation_hints")
    if hints is not None and not isinstance(hints, list):
        raise ValueError("implementation_hints must be a list")


def _validate_pattern(**kwargs):
    """Validate pattern parameters."""
    name = kwargs.get("name", "")
    category = kwargs.get("category", "")

    if not name:
        raise ValueError("name must be non-empty")

    if category not in VALID_PATTERN_CATEGORIES:
        raise ValueError(
            f"category must be one of {VALID_PATTERN_CATEGORIES}, got {category!r}"
        )


def _validate_complexity(**kwargs):
    """Validate complexity parameters."""
    time = kwargs.get("time", "")
    space = kwargs.get("space", "")

    if not time or not isinstance(time, str):
        raise ValueError("time must be a non-empty string")
    if not space or not isinstance(space, str):
        raise ValueError("space must be a non-empty string")


def _validate_generates(**kwargs):
    """Validate generates parameters."""
    count = kwargs.get("count", 1)

    if isinstance(count, int):
        if count <= 0:
            raise ValueError("count must be positive")
    elif count != "many":
        raise ValueError("count must be a positive int or 'many'")

    if "output_type" not in kwargs:
        raise ValueError("output_type is required")


# ============================================================================
# Decorator Definitions
# ============================================================================


example = make_decorator(
    name="example",
    steps=_example_steps,
    doc="""Provide example inputs/outputs for AI generation.

Multiple @example decorators can be stacked to provide multiple examples.

Args:
    inputs: Dictionary of input parameter names to values
    output: Expected output value
    description: Optional description of the example

Example:
    @example(inputs={"x": 1, "y": 2}, output=3, description="Simple addition")
    @example(inputs={"x": 0, "y": 0}, output=0, description="Zero case")
    def add(x, y):
        return x + y
""",
    validate=_validate_example,
    after_steps=_after_example,
)


constraints = make_decorator(
    name="constraints",
    steps=_constraints_steps,
    doc="""Specify constraints for AI-generated implementations.

Args:
    rules: List of constraint rules as strings

Example:
    @constraints(rules=["must handle None", "no side effects"])
    def process(data):
        pass
""",
    validate=_validate_constraints,
    after_steps=_after_constraints,
)


stub = make_decorator(
    name="stub",
    steps=_stub_steps,
    doc="""Mark function as stub for AI implementation.

Args:
    signature_only: If True, only signature is defined (default: True)
    implementation_hints: Optional hints for implementation

Example:
    @stub(signature_only=False, implementation_hints=["use binary search"])
    def find_item(items, target):
        pass
""",
    validate=_validate_stub,
    after_steps=_after_stub,
)


pattern = make_decorator(
    name="pattern",
    steps=_pattern_steps,
    doc="""Mark class/function as implementing a design pattern.

Args:
    name: Pattern name (e.g., "Singleton", "Observer")
    category: Pattern category - must be one of:
              "creational", "behavioral", "structural"

Example:
    @pattern(name="Singleton", category="creational")
    class Config:
        pass
""",
    validate=_validate_pattern,
    after_steps=_after_pattern,
)


complexity = make_decorator(
    name="complexity",
    steps=_complexity_steps,
    doc="""Annotate algorithmic complexity.

Args:
    time: Time complexity (e.g., "O(n)", "O(log n)")
    space: Space complexity (e.g., "O(1)", "O(n)")

Example:
    @complexity(time="O(n log n)", space="O(n)")
    def merge_sort(items):
        pass
""",
    validate=_validate_complexity,
    after_steps=_after_complexity,
)


generates = make_decorator(
    name="generates",
    steps=_generates_steps,
    doc="""Specify what the function generates.

Args:
    output_type: Type of output generated
    count: Number of outputs - positive int or "many" (default: 1)

Example:
    @generates(output_type=str, count="many")
    def generate_names():
        pass
""",
    validate=_validate_generates,
    after_steps=_after_generates,
)


pure = make_decorator(
    name="pure",
    steps=_pure_steps,
    doc="""Mark function as pure (no side effects, deterministic).

Example:
    @pure()
    def add(x, y):
        return x + y
""",
    after_steps=_after_pure,
)


# ============================================================================
# Registry Registration
# ============================================================================


_REGISTRY_ENTRIES = [
    ("example", example, ("function", "class")),
    ("constraints", constraints, ("function", "class")),
    ("stub", stub, ("function", "class")),
    ("pattern", pattern, ("function", "class")),
    ("complexity", complexity, ("function", "class")),
    ("generates", generates, ("function", "class")),
    ("pure", pure, ("function", "class")),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.AI_GENERATION,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.AI_GENERATION].append(_spec)
