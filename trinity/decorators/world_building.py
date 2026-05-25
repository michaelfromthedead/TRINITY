"""
Trinity Pattern - Tier 48: WORLD_BUILDING Decorators

World building, foliage, water, navmesh, and trigger volume decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_WATER_TYPES = frozenset({"ocean", "lake", "river"})
VALID_NAVMESH_MODIFIERS = frozenset({"include", "exclude", "replace"})
VALID_TRIGGER_EVENTS = frozenset({"on_enter", "on_exit", "on_overlap"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _foliage_type_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @foliage_type decorator."""
    density = params.get("density", 1.0)
    cull_distance = params.get("cull_distance", 10000.0)
    collision = params.get("collision", False)
    wind_response = params.get("wind_response", 1.0)

    return [
        Step(Op.TAG, {"key": "foliage_type", "value": True}),
        Step(Op.TAG, {"key": "foliage_density", "value": density}),
        Step(Op.TAG, {"key": "foliage_cull_distance", "value": cull_distance}),
        Step(Op.TAG, {"key": "foliage_collision", "value": collision}),
        Step(Op.TAG, {"key": "foliage_wind_response", "value": wind_response}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _procedural_placement_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @procedural_placement decorator."""
    density = params.get("density", 1.0)
    noise = params.get("noise", "perlin")
    slope_range = params.get("slope_range", (0, 90))
    height_range = params.get("height_range")

    return [
        Step(Op.TAG, {"key": "procedural_placement", "value": True}),
        Step(Op.TAG, {"key": "placement_density", "value": density}),
        Step(Op.TAG, {"key": "placement_noise", "value": noise}),
        Step(Op.TAG, {"key": "placement_slope_range", "value": slope_range}),
        Step(Op.TAG, {"key": "placement_height_range", "value": height_range}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _level_instance_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @level_instance decorator."""
    always_loaded = params.get("always_loaded", False)
    load_on_proximity = params.get("load_on_proximity", True)
    proximity_radius = params.get("proximity_radius", 10000.0)

    return [
        Step(Op.TAG, {"key": "level_instance", "value": True}),
        Step(Op.TAG, {"key": "level_always_loaded", "value": always_loaded}),
        Step(Op.TAG, {"key": "level_load_on_proximity", "value": load_on_proximity}),
        Step(Op.TAG, {"key": "level_proximity_radius", "value": proximity_radius}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _water_body_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @water_body decorator."""
    water_type = params.get("type", "lake")
    wave_source = params.get("wave_source", False)

    return [
        Step(Op.TAG, {"key": "water_body", "value": True}),
        Step(Op.TAG, {"key": "water_type", "value": water_type}),
        Step(Op.TAG, {"key": "water_wave_source", "value": wave_source}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _navmesh_modifier_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @navmesh_modifier decorator."""
    area_class = params.get("area_class", "default")
    modifier = params.get("modifier", "replace")

    return [
        Step(Op.TAG, {"key": "navmesh_modifier", "value": True}),
        Step(Op.TAG, {"key": "navmesh_area_class", "value": area_class}),
        Step(Op.TAG, {"key": "navmesh_modifier_type", "value": modifier}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _trigger_volume_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @trigger_volume decorator."""
    events = params.get("events", [])
    filter_tags = params.get("filter_tags", set())

    return [
        Step(Op.TAG, {"key": "trigger_volume", "value": True}),
        Step(Op.TAG, {"key": "trigger_events", "value": list(events)}),
        Step(Op.TAG, {"key": "trigger_filter_tags", "value": set(filter_tags)}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_foliage_type_params(**kwargs: Any) -> None:
    """Validate @foliage_type parameters."""
    density = kwargs.get("density", 1.0)
    if density <= 0:
        raise ValueError(f"density must be > 0, got {density}")

    cull_distance = kwargs.get("cull_distance", 10000.0)
    if cull_distance <= 0:
        raise ValueError(f"cull_distance must be > 0, got {cull_distance}")

    wind_response = kwargs.get("wind_response", 1.0)
    if wind_response < 0:
        raise ValueError(f"wind_response must be >= 0, got {wind_response}")


def _validate_procedural_placement_params(**kwargs: Any) -> None:
    """Validate @procedural_placement parameters."""
    density = kwargs.get("density")
    if density is None:
        raise ValueError("density is required")
    if density <= 0:
        raise ValueError(f"density must be > 0, got {density}")

    slope_range = kwargs.get("slope_range", (0, 90))
    if not isinstance(slope_range, (tuple, list)) or len(slope_range) != 2:
        raise TypeError("slope_range must be a tuple or list of two floats")
    if slope_range[0] > slope_range[1]:
        raise ValueError(
            f"slope_range[0] must be <= slope_range[1], got {slope_range}"
        )


def _validate_level_instance_params(**kwargs: Any) -> None:
    """Validate @level_instance parameters."""
    proximity_radius = kwargs.get("proximity_radius", 10000.0)
    if proximity_radius <= 0:
        raise ValueError(f"proximity_radius must be > 0, got {proximity_radius}")


def _validate_water_body_params(**kwargs: Any) -> None:
    """Validate @water_body parameters."""
    water_type = kwargs.get("type")
    if water_type not in VALID_WATER_TYPES:
        raise ValueError(
            f"Invalid type '{water_type}'. Must be one of {sorted(VALID_WATER_TYPES)}"
        )


def _validate_navmesh_modifier_params(**kwargs: Any) -> None:
    """Validate @navmesh_modifier parameters."""
    modifier = kwargs.get("modifier", "replace")
    if modifier not in VALID_NAVMESH_MODIFIERS:
        raise ValueError(
            f"Invalid modifier '{modifier}'. Must be one of {sorted(VALID_NAVMESH_MODIFIERS)}"
        )


def _validate_trigger_volume_params(**kwargs: Any) -> None:
    """Validate @trigger_volume parameters."""
    events = kwargs.get("events", [])
    if not events:
        raise ValueError("events must be a non-empty list")

    for event in events:
        if event not in VALID_TRIGGER_EVENTS:
            raise ValueError(
                f"Invalid event '{event}'. Must be one of {sorted(VALID_TRIGGER_EVENTS)}"
            )


# ============================================================================
# After-apply functions
# ============================================================================


def _foliage_type_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @foliage_type is applied."""
    obj._foliage_type = True
    obj._foliage_density = params.get("density", 1.0)
    obj._foliage_cull_distance = params.get("cull_distance", 10000.0)
    obj._foliage_collision = params.get("collision", False)
    obj._foliage_wind_response = params.get("wind_response", 1.0)


def _procedural_placement_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @procedural_placement is applied."""
    obj._procedural_placement = True
    obj._placement_density = params.get("density", 1.0)
    obj._placement_noise = params.get("noise", "perlin")
    obj._placement_slope_range = params.get("slope_range", (0, 90))
    obj._placement_height_range = params.get("height_range")


def _level_instance_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @level_instance is applied."""
    obj._level_instance = True
    obj._level_always_loaded = params.get("always_loaded", False)
    obj._level_load_on_proximity = params.get("load_on_proximity", True)
    obj._level_proximity_radius = params.get("proximity_radius", 10000.0)


def _water_body_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @water_body is applied."""
    obj._water_body = True
    obj._water_type = params.get("type", "lake")
    obj._water_wave_source = params.get("wave_source", False)


def _navmesh_modifier_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @navmesh_modifier is applied."""
    obj._navmesh_modifier = True
    obj._navmesh_area_class = params.get("area_class", "default")
    obj._navmesh_modifier_type = params.get("modifier", "replace")


def _trigger_volume_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @trigger_volume is applied."""
    obj._trigger_volume = True
    obj._trigger_events = list(params.get("events", []))
    obj._trigger_filter_tags = set(params.get("filter_tags", set()))


# ============================================================================
# Decorator creation
# ============================================================================

foliage_type = make_decorator(
    name="foliage_type",
    steps=_foliage_type_steps,
    doc="Instanced foliage configuration with density, culling, and wind response.",
    validate=_validate_foliage_type_params,
    after_steps=_foliage_type_after_apply,
)

procedural_placement = make_decorator(
    name="procedural_placement",
    steps=_procedural_placement_steps,
    doc="Procedural scattering with noise-based distribution and terrain constraints.",
    validate=_validate_procedural_placement_params,
    after_steps=_procedural_placement_after_apply,
)

level_instance = make_decorator(
    name="level_instance",
    steps=_level_instance_steps,
    doc="Sub-level streaming with proximity-based loading.",
    validate=_validate_level_instance_params,
    after_steps=_level_instance_after_apply,
)

water_body = make_decorator(
    name="water_body",
    steps=_water_body_steps,
    doc="Water volume configuration for oceans, lakes, and rivers.",
    validate=_validate_water_body_params,
    after_steps=_water_body_after_apply,
)

navmesh_modifier = make_decorator(
    name="navmesh_modifier",
    steps=_navmesh_modifier_steps,
    doc="Navigation mesh modifier for pathfinding.",
    validate=_validate_navmesh_modifier_params,
    after_steps=_navmesh_modifier_after_apply,
)

trigger_volume = make_decorator(
    name="trigger_volume",
    steps=_trigger_volume_steps,
    doc="Trigger zone with event handling and tag filtering.",
    validate=_validate_trigger_volume_params,
    after_steps=_trigger_volume_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("foliage_type", foliage_type, ("class",)),
    ("procedural_placement", procedural_placement, ("class",)),
    ("level_instance", level_instance, ("class",)),
    ("water_body", water_body, ("class",)),
    ("navmesh_modifier", navmesh_modifier, ("class",)),
    ("trigger_volume", trigger_volume, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.WORLD_BUILDING,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.WORLD_BUILDING].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "foliage_type",
    "procedural_placement",
    "level_instance",
    "water_body",
    "navmesh_modifier",
    "trigger_volume",
    "VALID_WATER_TYPES",
    "VALID_NAVMESH_MODIFIERS",
    "VALID_TRIGGER_EVENTS",
]
