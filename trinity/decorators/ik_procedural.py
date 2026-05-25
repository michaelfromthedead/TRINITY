"""
Trinity Pattern - Tier 44: IK_PROCEDURAL Decorators

IK chain, procedural bone animation, motion matching, and ragdoll decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_IK_SOLVERS = frozenset({"fabrik", "ccd", "jacobian", "fullbody"})
VALID_BONE_TYPES = frozenset({"jiggle", "spring", "lookat", "aim", "twist"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _ik_chain_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @ik_chain decorator."""
    solver = params.get("solver", "fabrik")
    iterations = params.get("iterations", 10)

    return [
        Step(Op.TAG, {"key": "ik_chain", "value": True}),
        Step(Op.TAG, {"key": "ik_solver", "value": solver}),
        Step(Op.TAG, {"key": "ik_iterations", "value": iterations}),
        Step(Op.REGISTER, {"registry": "ik_procedural"}),
    ]


def _ik_goal_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @ik_goal decorator."""
    priority = params.get("priority", 0)
    blend_speed = params.get("blend_speed", 10.0)

    return [
        Step(Op.TAG, {"key": "ik_goal", "value": True}),
        Step(Op.TAG, {"key": "ik_goal_priority", "value": priority}),
        Step(Op.TAG, {"key": "ik_goal_blend_speed", "value": blend_speed}),
        Step(Op.REGISTER, {"registry": "ik_procedural"}),
    ]


def _procedural_bone_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @procedural_bone decorator."""
    bone_type = params.get("type")

    return [
        Step(Op.TAG, {"key": "procedural_bone", "value": True}),
        Step(Op.TAG, {"key": "procedural_bone_type", "value": bone_type}),
        Step(Op.REGISTER, {"registry": "ik_procedural"}),
    ]


def _motion_matching_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @motion_matching decorator."""
    database = params.get("database", "")
    trajectory_weight = params.get("trajectory_weight", 1.0)
    pose_weight = params.get("pose_weight", 1.0)

    return [
        Step(Op.TAG, {"key": "motion_matching", "value": True}),
        Step(Op.TAG, {"key": "motion_database", "value": database}),
        Step(Op.TAG, {"key": "motion_trajectory_weight", "value": trajectory_weight}),
        Step(Op.TAG, {"key": "motion_pose_weight", "value": pose_weight}),
        Step(Op.REGISTER, {"registry": "ik_procedural"}),
    ]


def _ragdoll_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @ragdoll decorator."""
    blend_time = params.get("blend_time", 0.2)
    joint_limits = params.get("joint_limits", True)

    return [
        Step(Op.TAG, {"key": "ragdoll", "value": True}),
        Step(Op.TAG, {"key": "ragdoll_blend_time", "value": blend_time}),
        Step(Op.TAG, {"key": "ragdoll_joint_limits", "value": joint_limits}),
        Step(Op.REGISTER, {"registry": "ik_procedural"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_ik_chain_params(**kwargs: Any) -> None:
    """Validate @ik_chain parameters."""
    solver = kwargs.get("solver", "fabrik")
    if solver not in VALID_IK_SOLVERS:
        raise ValueError(
            f"Invalid solver '{solver}'. Must be one of {sorted(VALID_IK_SOLVERS)}"
        )

    iterations = kwargs.get("iterations", 10)
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {iterations}")


def _validate_ik_goal_params(**kwargs: Any) -> None:
    """Validate @ik_goal parameters."""
    blend_speed = kwargs.get("blend_speed", 10.0)
    if blend_speed <= 0:
        raise ValueError(f"blend_speed must be > 0, got {blend_speed}")


def _validate_procedural_bone_params(**kwargs: Any) -> None:
    """Validate @procedural_bone parameters."""
    bone_type = kwargs.get("type")
    if bone_type not in VALID_BONE_TYPES:
        raise ValueError(
            f"Invalid type '{bone_type}'. Must be one of {sorted(VALID_BONE_TYPES)}"
        )


def _validate_motion_matching_params(**kwargs: Any) -> None:
    """Validate @motion_matching parameters."""
    database = kwargs.get("database", "")
    if not database:
        raise ValueError("database must be a non-empty string")

    trajectory_weight = kwargs.get("trajectory_weight", 1.0)
    if trajectory_weight <= 0:
        raise ValueError(f"trajectory_weight must be > 0, got {trajectory_weight}")

    pose_weight = kwargs.get("pose_weight", 1.0)
    if pose_weight <= 0:
        raise ValueError(f"pose_weight must be > 0, got {pose_weight}")


def _validate_ragdoll_params(**kwargs: Any) -> None:
    """Validate @ragdoll parameters."""
    blend_time = kwargs.get("blend_time", 0.2)
    if blend_time < 0:
        raise ValueError(f"blend_time must be >= 0, got {blend_time}")


# ============================================================================
# After-apply functions
# ============================================================================


def _ik_chain_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @ik_chain is applied."""
    solver = params.get("solver", "fabrik")
    iterations = params.get("iterations", 10)

    obj._ik_chain = True
    obj._ik_solver = solver
    obj._ik_iterations = iterations


def _ik_goal_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @ik_goal is applied."""
    priority = params.get("priority", 0)
    blend_speed = params.get("blend_speed", 10.0)

    obj._ik_goal = True
    obj._ik_goal_priority = priority
    obj._ik_goal_blend_speed = blend_speed


def _procedural_bone_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @procedural_bone is applied."""
    bone_type = params.get("type")

    obj._procedural_bone = True
    obj._procedural_bone_type = bone_type


def _motion_matching_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @motion_matching is applied."""
    database = params.get("database", "")
    trajectory_weight = params.get("trajectory_weight", 1.0)
    pose_weight = params.get("pose_weight", 1.0)

    obj._motion_matching = True
    obj._motion_database = database
    obj._motion_trajectory_weight = trajectory_weight
    obj._motion_pose_weight = pose_weight


def _ragdoll_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @ragdoll is applied."""
    blend_time = params.get("blend_time", 0.2)
    joint_limits = params.get("joint_limits", True)

    obj._ragdoll = True
    obj._ragdoll_blend_time = blend_time
    obj._ragdoll_joint_limits = joint_limits


# ============================================================================
# Decorator creation
# ============================================================================

ik_chain = make_decorator(
    name="ik_chain",
    steps=_ik_chain_steps,
    doc="IK chain solver configuration.",
    validate=_validate_ik_chain_params,
    after_steps=_ik_chain_after_apply,
)

ik_goal = make_decorator(
    name="ik_goal",
    steps=_ik_goal_steps,
    doc="IK target goal configuration.",
    validate=_validate_ik_goal_params,
    after_steps=_ik_goal_after_apply,
)

procedural_bone = make_decorator(
    name="procedural_bone",
    steps=_procedural_bone_steps,
    doc="Procedural bone animation configuration.",
    validate=_validate_procedural_bone_params,
    after_steps=_procedural_bone_after_apply,
)

motion_matching = make_decorator(
    name="motion_matching",
    steps=_motion_matching_steps,
    doc="Motion matching animation system configuration.",
    validate=_validate_motion_matching_params,
    after_steps=_motion_matching_after_apply,
)

ragdoll = make_decorator(
    name="ragdoll",
    steps=_ragdoll_steps,
    doc="Ragdoll physics configuration.",
    validate=_validate_ragdoll_params,
    after_steps=_ragdoll_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("ik_chain", ik_chain, ("class",)),
    ("ik_goal", ik_goal, ("class",)),
    ("procedural_bone", procedural_bone, ("class",)),
    ("motion_matching", motion_matching, ("class",)),
    ("ragdoll", ragdoll, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.IK_PROCEDURAL,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.IK_PROCEDURAL].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "ik_chain",
    "ik_goal",
    "procedural_bone",
    "motion_matching",
    "ragdoll",
    "VALID_IK_SOLVERS",
    "VALID_BONE_TYPES",
]
