"""
Modding decorators — built from Ops.

These decorators support mod metadata, dependencies, conflicts,
feature declarations, and load ordering for a modding system.

Decorators:
    @mod          - Mod metadata (name, version, author)
    @requires     - Dependency declaration
    @conflicts    - Incompatibility declaration
    @provides     - Feature declaration
    @replaces     - Successor declaration
    @mod_extends  - Mod data modification
    @patch        - Compatibility patch
    @load_order   - Load order declaration
    @moddable     - Expose for modding
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")

# =============================================================================
# VALID VALUES
# =============================================================================

VALID_EXTEND_MODES = frozenset({"merge", "replace"})

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_mod(
    name: str = "", version: Any = (0, 0, 0), author: str = "", **_: Any
) -> None:
    if not name:
        raise ValueError("@mod: 'name' parameter is required and must be non-empty")
    if not author:
        raise ValueError("@mod: 'author' parameter is required and must be non-empty")
    if not isinstance(version, tuple) or len(version) != 3:
        raise ValueError(
            "@mod: 'version' must be a tuple of exactly 3 integers, "
            f"got {version!r}"
        )
    for i, v in enumerate(version):
        if not isinstance(v, int) or v < 0:
            raise ValueError(
                f"@mod: version element {i} must be a non-negative integer, got {v!r}"
            )


def _validate_requires(mod: str = "", **_: Any) -> None:
    if not mod:
        raise ValueError("@requires: 'mod' parameter is required and must be non-empty")


def _validate_conflicts(mod: str = "", reason: str = "", **_: Any) -> None:
    if not mod:
        raise ValueError("@conflicts: 'mod' parameter is required and must be non-empty")
    if not reason:
        raise ValueError(
            "@conflicts: 'reason' parameter is required and must be non-empty"
        )


def _validate_provides(feature: str = "", **_: Any) -> None:
    if not feature:
        raise ValueError(
            "@provides: 'feature' parameter is required and must be non-empty"
        )


def _validate_replaces(mod: str = "", **_: Any) -> None:
    if not mod:
        raise ValueError(
            "@replaces: 'mod' parameter is required and must be non-empty"
        )


def _validate_mod_extends(target_name: str = "", mode: str = "merge", **_: Any) -> None:
    if not target_name:
        raise ValueError(
            "@mod_extends: 'target_name' parameter is required and must be non-empty"
        )
    if mode not in VALID_EXTEND_MODES:
        raise ValueError(
            f"@mod_extends: invalid mode '{mode}'. "
            f"Valid modes: {sorted(VALID_EXTEND_MODES)}"
        )


def _validate_patch(base_mod: str = "", target_mod: str = "", **_: Any) -> None:
    if not base_mod:
        raise ValueError(
            "@patch: 'base_mod' parameter is required and must be non-empty"
        )
    if not target_mod:
        raise ValueError(
            "@patch: 'target_mod' parameter is required and must be non-empty"
        )


def _validate_moddable(namespace: str = "", version: int = 1, **_: Any) -> None:
    if not namespace:
        raise ValueError(
            "@moddable: 'namespace' parameter is required and must be non-empty"
        )
    if not isinstance(version, int) or version <= 0:
        raise ValueError(
            f"@moddable: 'version' must be a positive integer, got {version!r}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _mod_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "mod", "value": True}),
        Step(Op.TAG, {"key": "mod_name", "value": params.get("name", "")}),
        Step(Op.TAG, {"key": "mod_version", "value": params.get("version", (0, 0, 0))}),
        Step(Op.TAG, {"key": "mod_author", "value": params.get("author", "")}),
        Step(
            Op.TAG,
            {"key": "mod_description", "value": params.get("description", "")},
        ),
        Step(Op.REGISTER, {"registry": "modding"}),
    ]


def _requires_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "requires", "value": True}),
        Step(Op.REGISTER, {"registry": "modding"}),
    ]


def _conflicts_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "conflicts", "value": True}),
        Step(Op.REGISTER, {"registry": "modding"}),
    ]


def _provides_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "provides", "value": True}),
        Step(Op.REGISTER, {"registry": "modding"}),
    ]


def _replaces_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "replaces", "value": True}),
        Step(Op.TAG, {"key": "replaces_mod", "value": params.get("mod", "")}),
        Step(Op.TAG, {"key": "replaces_reason", "value": params.get("reason", "")}),
        Step(Op.REGISTER, {"registry": "modding"}),
    ]


def _mod_extends_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "mod_extends", "value": True}),
        Step(
            Op.TAG,
            {"key": "mod_extends_target", "value": params.get("target_name", "")},
        ),
        Step(Op.TAG, {"key": "mod_extends_mode", "value": params.get("mode", "merge")}),
        Step(Op.REGISTER, {"registry": "modding"}),
    ]


def _patch_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "patch", "value": True}),
        Step(Op.TAG, {"key": "patch_base", "value": params.get("base_mod", "")}),
        Step(Op.TAG, {"key": "patch_target", "value": params.get("target_mod", "")}),
        Step(Op.REGISTER, {"registry": "modding"}),
    ]


def _load_order_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "load_order", "value": True}),
        Step(
            Op.TAG,
            {"key": "load_order_after", "value": list(params.get("after_mods") or [])},
        ),
        Step(
            Op.TAG,
            {
                "key": "load_order_before",
                "value": list(params.get("before_mods") or []),
            },
        ),
        Step(Op.REGISTER, {"registry": "modding"}),
    ]


def _moddable_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "moddable", "value": True}),
        Step(
            Op.TAG,
            {"key": "moddable_namespace", "value": params.get("namespace", "")},
        ),
        Step(Op.TAG, {"key": "moddable_version", "value": params.get("version", 1)}),
        Step(Op.REGISTER, {"registry": "modding"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_mod(target: Any, params: dict[str, Any]) -> Any:
    target._mod = True
    target._mod_name = params.get("name", "")
    target._mod_version = params.get("version", (0, 0, 0))
    target._mod_author = params.get("author", "")
    target._mod_description = params.get("description", "")
    return None


def _after_requires(target: Any, params: dict[str, Any]) -> Any:
    if not hasattr(target, "_requires"):
        target._requires = []
    target._requires.append(
        {
            "mod": params.get("mod", ""),
            "version": params.get("version", "*"),
            "optional": params.get("optional", False),
        }
    )
    return None


def _after_conflicts(target: Any, params: dict[str, Any]) -> Any:
    if not hasattr(target, "_conflicts"):
        target._conflicts = []
    target._conflicts.append(
        {
            "mod": params.get("mod", ""),
            "reason": params.get("reason", ""),
        }
    )
    return None


def _after_provides(target: Any, params: dict[str, Any]) -> Any:
    if not hasattr(target, "_provides"):
        target._provides = []
    target._provides.append(params.get("feature", ""))
    return None


def _after_replaces(target: Any, params: dict[str, Any]) -> Any:
    target._replaces = True
    target._replaces_mod = params.get("mod", "")
    target._replaces_reason = params.get("reason", "")
    return None


def _after_mod_extends(target: Any, params: dict[str, Any]) -> Any:
    target._mod_extends = True
    target._mod_extends_target = params.get("target_name", "")
    target._mod_extends_mode = params.get("mode", "merge")
    return None


def _after_patch(target: Any, params: dict[str, Any]) -> Any:
    target._patch = True
    target._patch_base = params.get("base_mod", "")
    target._patch_target = params.get("target_mod", "")
    return None


def _after_load_order(target: Any, params: dict[str, Any]) -> Any:
    target._load_order = True
    target._load_order_after = list(params.get("after_mods") or [])
    target._load_order_before = list(params.get("before_mods") or [])
    return None


def _after_moddable(target: Any, params: dict[str, Any]) -> Any:
    target._moddable = True
    target._moddable_namespace = params.get("namespace", "")
    target._moddable_version = params.get("version", 1)
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

mod = make_decorator(
    name="mod",
    steps=_mod_steps,
    doc="Declare mod metadata: name, version, author, description.",
    validate=_validate_mod,
    after_steps=_after_mod,
)

requires = make_decorator(
    name="requires",
    steps=_requires_steps,
    doc="Declare a mod dependency.",
    validate=_validate_requires,
    after_steps=_after_requires,
)

conflicts = make_decorator(
    name="conflicts",
    steps=_conflicts_steps,
    doc="Declare an incompatibility with another mod.",
    validate=_validate_conflicts,
    after_steps=_after_conflicts,
)

provides = make_decorator(
    name="provides",
    steps=_provides_steps,
    doc="Declare a feature provided by this mod.",
    validate=_validate_provides,
    after_steps=_after_provides,
)

replaces = make_decorator(
    name="replaces",
    steps=_replaces_steps,
    doc="Declare that this mod replaces another.",
    validate=_validate_replaces,
    after_steps=_after_replaces,
)

mod_extends = make_decorator(
    name="mod_extends",
    steps=_mod_extends_steps,
    doc="Declare that this mod extends another's data.",
    validate=_validate_mod_extends,
    after_steps=_after_mod_extends,
)

patch = make_decorator(
    name="patch",
    steps=_patch_steps,
    doc="Declare a compatibility patch between two mods.",
    validate=_validate_patch,
    after_steps=_after_patch,
)

load_order = make_decorator(
    name="load_order",
    steps=_load_order_steps,
    doc="Declare load ordering constraints.",
    after_steps=_after_load_order,
)

moddable = make_decorator(
    name="moddable",
    steps=_moddable_steps,
    doc="Expose a class for modding in a namespace.",
    validate=_validate_moddable,
    after_steps=_after_moddable,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("mod", mod, ("class",)),
    ("requires", requires, ("class",)),
    ("conflicts", conflicts, ("class",)),
    ("provides", provides, ("class",)),
    ("replaces", replaces, ("class",)),
    ("mod_extends", mod_extends, ("class",)),
    ("patch", patch, ("class",)),
    ("load_order", load_order, ("class",)),
    ("moddable", moddable, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.MODDING,
            func=_func,
            unique=_name not in ("requires", "conflicts", "provides"),
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.MODDING].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "mod",
    "requires",
    "conflicts",
    "provides",
    "replaces",
    "mod_extends",
    "patch",
    "load_order",
    "moddable",
    "VALID_EXTEND_MODES",
]
