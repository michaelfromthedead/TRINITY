"""
Trinity Pattern - Tier 46: PHYSICS_SIM Decorators

Physics simulation decorators for rigid bodies, soft bodies, cloth, fluids,
and vehicles. Supports sub-stepping, solver configuration, sleep thresholds,
continuous collision detection, buoyancy, and wind effects.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")

# =============================================================================
# VALID VALUES
# =============================================================================

VALID_DOMAINS = frozenset({"rigid_body", "soft_body", "cloth", "fluid", "vehicle"})
VALID_SOLVER_TYPES = frozenset({"pgs", "tgs", "xpbd"})
VALID_CCD_MODES = frozenset({"none", "speculative", "sweep"})

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_simulation_domain(domain: str = "", **_: Any) -> None:
    """Validate @simulation_domain parameters."""
    if domain not in VALID_DOMAINS:
        raise ValueError(
            f"@simulation_domain: invalid domain '{domain}'. "
            f"Valid domains: {sorted(VALID_DOMAINS)}"
        )


def _validate_substep(
    min_hz: int = 60, max_hz: int = 240, max_substeps: int = 4, **_: Any
) -> None:
    """Validate @substep parameters."""
    if min_hz <= 0:
        raise ValueError(f"@substep: min_hz must be > 0, got {min_hz}")
    if max_hz < min_hz:
        raise ValueError(
            f"@substep: max_hz must be >= min_hz, got max_hz={max_hz}, min_hz={min_hz}"
        )
    if max_substeps <= 0:
        raise ValueError(f"@substep: max_substeps must be > 0, got {max_substeps}")


def _validate_solver_hint(
    type: str = "pgs", iterations: int = 4, warm_starting: bool = True, **_: Any
) -> None:
    """Validate @solver_hint parameters."""
    if type not in VALID_SOLVER_TYPES:
        raise ValueError(
            f"@solver_hint: invalid type '{type}'. "
            f"Valid types: {sorted(VALID_SOLVER_TYPES)}"
        )
    if iterations <= 0:
        raise ValueError(f"@solver_hint: iterations must be > 0, got {iterations}")


def _validate_sleep_threshold(
    linear: float = 0.1, angular: float = 0.05, time: float = 0.5, **_: Any
) -> None:
    """Validate @sleep_threshold parameters."""
    if linear < 0:
        raise ValueError(f"@sleep_threshold: linear must be >= 0, got {linear}")
    if angular < 0:
        raise ValueError(f"@sleep_threshold: angular must be >= 0, got {angular}")
    if time < 0:
        raise ValueError(f"@sleep_threshold: time must be >= 0, got {time}")


def _validate_continuous_collision(mode: str = "none", **_: Any) -> None:
    """Validate @continuous_collision parameters."""
    if mode not in VALID_CCD_MODES:
        raise ValueError(
            f"@continuous_collision: invalid mode '{mode}'. "
            f"Valid modes: {sorted(VALID_CCD_MODES)}"
        )


def _validate_buoyancy(
    density: float = 1.0, drag: float = 0.5, angular_drag: float = 0.1, **_: Any
) -> None:
    """Validate @buoyancy parameters."""
    if density <= 0:
        raise ValueError(f"@buoyancy: density must be > 0, got {density}")
    if drag < 0:
        raise ValueError(f"@buoyancy: drag must be >= 0, got {drag}")
    if angular_drag < 0:
        raise ValueError(f"@buoyancy: angular_drag must be >= 0, got {angular_drag}")


def _validate_wind_affected(
    drag_coefficient: float = 1.0, area: float | str = "auto", **_: Any
) -> None:
    """Validate @wind_affected parameters."""
    if drag_coefficient <= 0:
        raise ValueError(
            f"@wind_affected: drag_coefficient must be > 0, got {drag_coefficient}"
        )
    if isinstance(area, float) and area <= 0:
        raise ValueError(f"@wind_affected: area must be > 0, got {area}")
    if isinstance(area, str) and area != "auto":
        raise ValueError(f"@wind_affected: area must be 'auto' or a positive float, got '{area}'")


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _simulation_domain_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @simulation_domain decorator."""
    domain = params.get("domain", "")

    return [
        Step(Op.TAG, {"key": "simulation_domain", "value": True}),
        Step(Op.TAG, {"key": "physics_domain", "value": domain}),
        Step(Op.REGISTER, {"registry": "physics_sim"}),
    ]


def _substep_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @substep decorator."""
    min_hz = params.get("min_hz", 60)
    max_hz = params.get("max_hz", 240)
    max_substeps = params.get("max_substeps", 4)

    return [
        Step(Op.TAG, {"key": "substep", "value": True}),
        Step(Op.TAG, {"key": "substep_min_hz", "value": min_hz}),
        Step(Op.TAG, {"key": "substep_max_hz", "value": max_hz}),
        Step(Op.TAG, {"key": "substep_max_substeps", "value": max_substeps}),
        Step(Op.REGISTER, {"registry": "physics_sim"}),
    ]


def _solver_hint_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @solver_hint decorator."""
    solver_type = params.get("type", "pgs")
    iterations = params.get("iterations", 4)
    warm_starting = params.get("warm_starting", True)

    return [
        Step(Op.TAG, {"key": "solver_hint", "value": True}),
        Step(Op.TAG, {"key": "solver_type", "value": solver_type}),
        Step(Op.TAG, {"key": "solver_iterations", "value": iterations}),
        Step(Op.TAG, {"key": "solver_warm_starting", "value": warm_starting}),
        Step(Op.REGISTER, {"registry": "physics_sim"}),
    ]


def _sleep_threshold_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @sleep_threshold decorator."""
    linear = params.get("linear", 0.1)
    angular = params.get("angular", 0.05)
    time = params.get("time", 0.5)

    return [
        Step(Op.TAG, {"key": "sleep_threshold", "value": True}),
        Step(Op.TAG, {"key": "sleep_linear", "value": linear}),
        Step(Op.TAG, {"key": "sleep_angular", "value": angular}),
        Step(Op.TAG, {"key": "sleep_time", "value": time}),
        Step(Op.REGISTER, {"registry": "physics_sim"}),
    ]


def _continuous_collision_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @continuous_collision decorator."""
    mode = params.get("mode", "none")

    return [
        Step(Op.TAG, {"key": "continuous_collision", "value": True}),
        Step(Op.TAG, {"key": "ccd_mode", "value": mode}),
        Step(Op.REGISTER, {"registry": "physics_sim"}),
    ]


def _buoyancy_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @buoyancy decorator."""
    density = params.get("density", 1.0)
    drag = params.get("drag", 0.5)
    angular_drag = params.get("angular_drag", 0.1)

    return [
        Step(Op.TAG, {"key": "buoyancy", "value": True}),
        Step(Op.TAG, {"key": "buoyancy_density", "value": density}),
        Step(Op.TAG, {"key": "buoyancy_drag", "value": drag}),
        Step(Op.TAG, {"key": "buoyancy_angular_drag", "value": angular_drag}),
        Step(Op.REGISTER, {"registry": "physics_sim"}),
    ]


def _wind_affected_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @wind_affected decorator."""
    drag_coefficient = params.get("drag_coefficient", 1.0)
    area = params.get("area", "auto")

    return [
        Step(Op.TAG, {"key": "wind_affected", "value": True}),
        Step(Op.TAG, {"key": "wind_drag_coefficient", "value": drag_coefficient}),
        Step(Op.TAG, {"key": "wind_area", "value": area}),
        Step(Op.REGISTER, {"registry": "physics_sim"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_simulation_domain(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @simulation_domain is applied."""
    target._simulation_domain = True
    target._physics_domain = params.get("domain", "")
    return None


def _after_substep(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @substep is applied."""
    target._substep = True
    target._substep_min_hz = params.get("min_hz", 60)
    target._substep_max_hz = params.get("max_hz", 240)
    target._substep_max_substeps = params.get("max_substeps", 4)
    return None


def _after_solver_hint(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @solver_hint is applied."""
    target._solver_hint = True
    target._solver_type = params.get("type", "pgs")
    target._solver_iterations = params.get("iterations", 4)
    target._solver_warm_starting = params.get("warm_starting", True)
    return None


def _after_sleep_threshold(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @sleep_threshold is applied."""
    target._sleep_threshold = True
    target._sleep_linear = params.get("linear", 0.1)
    target._sleep_angular = params.get("angular", 0.05)
    target._sleep_time = params.get("time", 0.5)
    return None


def _after_continuous_collision(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @continuous_collision is applied."""
    target._continuous_collision = True
    target._ccd_mode = params.get("mode", "none")
    return None


def _after_buoyancy(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @buoyancy is applied."""
    target._buoyancy = True
    target._buoyancy_density = params.get("density", 1.0)
    target._buoyancy_drag = params.get("drag", 0.5)
    target._buoyancy_angular_drag = params.get("angular_drag", 0.1)
    return None


def _after_wind_affected(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @wind_affected is applied."""
    target._wind_affected = True
    target._wind_drag_coefficient = params.get("drag_coefficient", 1.0)
    target._wind_area = params.get("area", "auto")
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

simulation_domain = make_decorator(
    name="simulation_domain",
    steps=_simulation_domain_steps,
    doc="Declare physics simulation domain (rigid_body, soft_body, cloth, fluid, vehicle).",
    validate=_validate_simulation_domain,
    after_steps=_after_simulation_domain,
)

substep = make_decorator(
    name="substep",
    steps=_substep_steps,
    doc="Configure physics sub-stepping for improved stability.",
    validate=_validate_substep,
    after_steps=_after_substep,
)

solver_hint = make_decorator(
    name="solver_hint",
    steps=_solver_hint_steps,
    doc="Configure physics solver (pgs, tgs, xpbd) with iterations and warm starting.",
    validate=_validate_solver_hint,
    after_steps=_after_solver_hint,
)

sleep_threshold = make_decorator(
    name="sleep_threshold",
    steps=_sleep_threshold_steps,
    doc="Configure sleep thresholds for physics bodies to reduce computation.",
    validate=_validate_sleep_threshold,
    after_steps=_after_sleep_threshold,
)

continuous_collision = make_decorator(
    name="continuous_collision",
    steps=_continuous_collision_steps,
    doc="Configure continuous collision detection mode (none, speculative, sweep).",
    validate=_validate_continuous_collision,
    after_steps=_after_continuous_collision,
)

buoyancy = make_decorator(
    name="buoyancy",
    steps=_buoyancy_steps,
    doc="Configure water buoyancy physics with density and drag.",
    validate=_validate_buoyancy,
    after_steps=_after_buoyancy,
)

wind_affected = make_decorator(
    name="wind_affected",
    steps=_wind_affected_steps,
    doc="Configure wind response with drag coefficient and area.",
    validate=_validate_wind_affected,
    after_steps=_after_wind_affected,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("simulation_domain", simulation_domain, ("class",)),
    ("substep", substep, ("class",)),
    ("solver_hint", solver_hint, ("class",)),
    ("sleep_threshold", sleep_threshold, ("class",)),
    ("continuous_collision", continuous_collision, ("class",)),
    ("buoyancy", buoyancy, ("class",)),
    ("wind_affected", wind_affected, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.PHYSICS_SIM,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.PHYSICS_SIM].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "simulation_domain",
    "substep",
    "solver_hint",
    "sleep_threshold",
    "continuous_collision",
    "buoyancy",
    "wind_affected",
    "VALID_DOMAINS",
    "VALID_SOLVER_TYPES",
    "VALID_CCD_MODES",
]
