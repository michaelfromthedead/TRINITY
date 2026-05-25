"""
Trinity Pattern - Tier 43: DESTRUCTION Decorators

Destruction system decorators for destructible objects, damage types,
fracture patterns, physics materials, and joints.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_FRACTURE_PATTERN = frozenset({"voronoi", "radial", "slice", "custom"})
VALID_JOINT_TYPE = frozenset({"fixed", "hinge", "slider", "ball", "spring"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Configuration dataclasses
# ============================================================================


@dataclass(frozen=True)
class DestructibleConfig:
    """Destructible object configuration."""

    health: float
    fracture_depth: int
    debris_lifetime: float


@dataclass(frozen=True)
class DamageTypeConfig:
    """Damage type configuration."""

    id: str
    base_multiplier: float


@dataclass(frozen=True)
class DamageResistanceConfig:
    """Damage resistance configuration."""

    resistances: dict[str, float]


@dataclass(frozen=True)
class FractureConfig:
    """Fracture pattern configuration."""

    pattern: str
    min_size: float
    interior_material: str | None


@dataclass(frozen=True)
class PhysicsMaterialConfig:
    """Physics material configuration."""

    friction: float
    restitution: float
    density: float


@dataclass(frozen=True)
class JointConfig:
    """Physics joint configuration."""

    type: str
    break_force: float | None
    break_torque: float | None


# ============================================================================
# Step builders
# ============================================================================


def _destructible_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @destructible decorator."""
    health = params.get("health", 100.0)
    fracture_depth = params.get("fracture_depth", 2)
    debris_lifetime = params.get("debris_lifetime", 10.0)

    return [
        Step(Op.TAG, {"key": "destructible", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "destructible_config",
                "value": DestructibleConfig(
                    health=health,
                    fracture_depth=fracture_depth,
                    debris_lifetime=debris_lifetime,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "destruction"}),
    ]


def _damage_type_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @damage_type decorator."""
    id = params["id"]  # REQUIRED parameter
    base_multiplier = params.get("base_multiplier", 1.0)

    return [
        Step(Op.TAG, {"key": "damage_type", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "damage_type_config",
                "value": DamageTypeConfig(id=id, base_multiplier=base_multiplier),
            },
        ),
        Step(Op.REGISTER, {"registry": "destruction"}),
    ]


def _damage_resistance_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @damage_resistance decorator."""
    resistances = params.get("resistances", {})

    return [
        Step(Op.TAG, {"key": "damage_resistance", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "damage_resistance_config",
                "value": DamageResistanceConfig(resistances=dict(resistances)),
            },
        ),
        Step(Op.REGISTER, {"registry": "destruction"}),
    ]


def _fracture_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @fracture decorator."""
    pattern = params.get("pattern", "voronoi")
    min_size = params.get("min_size", 0.1)
    interior_material = params.get("interior_material")

    return [
        Step(Op.TAG, {"key": "fracture", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "fracture_config",
                "value": FractureConfig(
                    pattern=pattern,
                    min_size=min_size,
                    interior_material=interior_material,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "destruction"}),
    ]


def _physics_material_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @physics_material decorator."""
    friction = params.get("friction", 0.5)
    restitution = params.get("restitution", 0.3)
    density = params.get("density", 1.0)

    return [
        Step(Op.TAG, {"key": "physics_material", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "physics_material_config",
                "value": PhysicsMaterialConfig(
                    friction=friction,
                    restitution=restitution,
                    density=density,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "destruction"}),
    ]


def _joint_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @joint decorator."""
    type = params.get("type", "fixed")
    break_force = params.get("break_force")
    break_torque = params.get("break_torque")

    return [
        Step(Op.TAG, {"key": "joint", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "joint_config",
                "value": JointConfig(
                    type=type,
                    break_force=break_force,
                    break_torque=break_torque,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "destruction"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_destructible_params(**kwargs: Any) -> None:
    """Validate @destructible parameters."""
    health = kwargs.get("health", 100.0)
    if health <= 0:
        raise ValueError(f"health must be > 0, got {health}")

    fracture_depth = kwargs.get("fracture_depth", 2)
    if fracture_depth < 0:
        raise ValueError(f"fracture_depth must be >= 0, got {fracture_depth}")

    debris_lifetime = kwargs.get("debris_lifetime", 10.0)
    if debris_lifetime < 0:
        raise ValueError(f"debris_lifetime must be >= 0, got {debris_lifetime}")


def _validate_damage_type_params(**kwargs: Any) -> None:
    """Validate @damage_type parameters."""
    id = kwargs.get("id")
    if not id:
        raise ValueError("id must be a non-empty string")

    base_multiplier = kwargs.get("base_multiplier", 1.0)
    if base_multiplier <= 0:
        raise ValueError(f"base_multiplier must be > 0, got {base_multiplier}")


def _validate_damage_resistance_params(**kwargs: Any) -> None:
    """Validate @damage_resistance parameters."""
    resistances = kwargs.get("resistances", {})
    if not resistances:
        raise ValueError("resistances must be a non-empty dict")
    if not isinstance(resistances, dict):
        raise TypeError("resistances must be a dict")


def _validate_fracture_params(**kwargs: Any) -> None:
    """Validate @fracture parameters."""
    pattern = kwargs.get("pattern", "voronoi")
    if pattern not in VALID_FRACTURE_PATTERN:
        raise ValueError(
            f"Invalid pattern '{pattern}'. Must be one of {VALID_FRACTURE_PATTERN}"
        )

    min_size = kwargs.get("min_size", 0.1)
    if min_size <= 0:
        raise ValueError(f"min_size must be > 0, got {min_size}")


def _validate_physics_material_params(**kwargs: Any) -> None:
    """Validate @physics_material parameters."""
    friction = kwargs.get("friction", 0.5)
    if friction < 0:
        raise ValueError(f"friction must be >= 0, got {friction}")

    restitution = kwargs.get("restitution", 0.3)
    if restitution < 0:
        raise ValueError(f"restitution must be >= 0, got {restitution}")

    density = kwargs.get("density", 1.0)
    if density < 0:
        raise ValueError(f"density must be >= 0, got {density}")


def _validate_joint_params(**kwargs: Any) -> None:
    """Validate @joint parameters."""
    type = kwargs.get("type", "fixed")
    if type not in VALID_JOINT_TYPE:
        raise ValueError(
            f"Invalid type '{type}'. Must be one of {VALID_JOINT_TYPE}"
        )


# ============================================================================
# After-apply functions
# ============================================================================


def _destructible_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @destructible is applied."""
    health = params.get("health", 100.0)
    fracture_depth = params.get("fracture_depth", 2)
    debris_lifetime = params.get("debris_lifetime", 10.0)

    obj._destructible = True
    obj._destructible_health = health
    obj._destructible_fracture_depth = fracture_depth
    obj._destructible_debris_lifetime = debris_lifetime
    obj._destructible_config = DestructibleConfig(
        health=health,
        fracture_depth=fracture_depth,
        debris_lifetime=debris_lifetime,
    )


def _damage_type_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @damage_type is applied."""
    id = params["id"]  # REQUIRED parameter
    base_multiplier = params.get("base_multiplier", 1.0)

    obj._damage_type = True
    obj._damage_type_id = id
    obj._damage_type_multiplier = base_multiplier
    obj._damage_type_config = DamageTypeConfig(id=id, base_multiplier=base_multiplier)


def _damage_resistance_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @damage_resistance is applied."""
    resistances = params.get("resistances", {})

    obj._damage_resistance = True
    obj._damage_resistance_values = dict(resistances)
    obj._damage_resistance_config = DamageResistanceConfig(resistances=dict(resistances))


def _fracture_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @fracture is applied."""
    pattern = params.get("pattern", "voronoi")
    min_size = params.get("min_size", 0.1)
    interior_material = params.get("interior_material")

    obj._fracture = True
    obj._fracture_pattern = pattern
    obj._fracture_min_size = min_size
    obj._fracture_interior_material = interior_material
    obj._fracture_config = FractureConfig(
        pattern=pattern,
        min_size=min_size,
        interior_material=interior_material,
    )


def _physics_material_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @physics_material is applied."""
    friction = params.get("friction", 0.5)
    restitution = params.get("restitution", 0.3)
    density = params.get("density", 1.0)

    obj._physics_material = True
    obj._physics_friction = friction
    obj._physics_restitution = restitution
    obj._physics_density = density
    obj._physics_material_config = PhysicsMaterialConfig(
        friction=friction,
        restitution=restitution,
        density=density,
    )


def _joint_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @joint is applied."""
    type = params.get("type", "fixed")
    break_force = params.get("break_force")
    break_torque = params.get("break_torque")

    obj._joint = True
    obj._joint_type = type
    obj._joint_break_force = break_force
    obj._joint_break_torque = break_torque
    obj._joint_config = JointConfig(
        type=type,
        break_force=break_force,
        break_torque=break_torque,
    )


# ============================================================================
# Decorator creation
# ============================================================================

destructible = make_decorator(
    name="destructible",
    steps=_destructible_steps,
    validate=_validate_destructible_params,
    after_steps=_destructible_after_apply,
)

damage_type = make_decorator(
    name="damage_type",
    steps=_damage_type_steps,
    validate=_validate_damage_type_params,
    after_steps=_damage_type_after_apply,
)

damage_resistance = make_decorator(
    name="damage_resistance",
    steps=_damage_resistance_steps,
    validate=_validate_damage_resistance_params,
    after_steps=_damage_resistance_after_apply,
)

fracture = make_decorator(
    name="fracture",
    steps=_fracture_steps,
    validate=_validate_fracture_params,
    after_steps=_fracture_after_apply,
)

physics_material = make_decorator(
    name="physics_material",
    steps=_physics_material_steps,
    validate=_validate_physics_material_params,
    after_steps=_physics_material_after_apply,
)

joint = make_decorator(
    name="joint",
    steps=_joint_steps,
    validate=_validate_joint_params,
    after_steps=_joint_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("destructible", destructible, ("class",)),
    ("damage_type", damage_type, ("class",)),
    ("damage_resistance", damage_resistance, ("class",)),
    ("fracture", fracture, ("class",)),
    ("physics_material", physics_material, ("class",)),
    ("joint", joint, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.DESTRUCTION,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.DESTRUCTION].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "destructible",
    "damage_type",
    "damage_resistance",
    "fracture",
    "physics_material",
    "joint",
    "DestructibleConfig",
    "DamageTypeConfig",
    "DamageResistanceConfig",
    "FractureConfig",
    "PhysicsMaterialConfig",
    "JointConfig",
    "VALID_FRACTURE_PATTERN",
    "VALID_JOINT_TYPE",
]
