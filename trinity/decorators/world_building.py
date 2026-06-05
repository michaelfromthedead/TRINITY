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

# New terrain/environment constants
VALID_BLEND_MODES = frozenset({"alpha", "height", "slope", "normal", "overlay"})
VALID_CLIMATE_ZONES = frozenset({"tropical", "temperate", "arid", "polar", "alpine", "mediterranean"})
VALID_NOISE_TYPES = frozenset({"perlin", "simplex", "worley", "value", "fractal"})
VALID_VOLUME_SHAPES = frozenset({"box", "sphere", "capsule", "convex"})

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
# NEW TERRAIN/ENVIRONMENT DECORATORS
# ============================================================================


def _set_registered_tiers(obj: Any) -> None:
    """Set _registered_tiers attribute for world_building decorators."""
    if not hasattr(obj, "_registered_tiers"):
        obj._registered_tiers = set()
    obj._registered_tiers.add("world_building")


def _terrain_patch_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @terrain_patch decorator."""
    size = params.get("size", 128)
    overlap = params.get("overlap", 0.1)
    height_data = params.get("height_data")

    return [
        Step(Op.TAG, {"key": "terrain_patch", "value": True}),
        Step(Op.TAG, {"key": "terrain_patch_size", "value": size}),
        Step(Op.TAG, {"key": "terrain_patch_overlap", "value": overlap}),
        Step(Op.TAG, {"key": "terrain_patch_height_data", "value": height_data}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _validate_terrain_patch_params(**kwargs: Any) -> None:
    """Validate @terrain_patch parameters."""
    size = kwargs.get("size", 128)
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"size must be a positive integer, got {size}")

    overlap = kwargs.get("overlap", 0.1)
    if not (0 <= overlap <= 1):
        raise ValueError(f"overlap must be between 0 and 1, got {overlap}")


def _terrain_patch_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @terrain_patch is applied."""
    _set_registered_tiers(obj)
    obj._terrain_patch = True
    obj._terrain_patch_params = {
        "size": params.get("size", 128),
        "overlap": params.get("overlap", 0.1),
        "height_data": params.get("height_data"),
    }


terrain_patch = make_decorator(
    name="terrain_patch",
    steps=_terrain_patch_steps,
    doc="Terrain patch configuration with size, overlap, and height data.",
    validate=_validate_terrain_patch_params,
    after_steps=_terrain_patch_after_apply,
)


def _heightfield_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @heightfield decorator."""
    resolution = params.get("resolution", 1024)
    height_scale = params.get("height_scale", 1.0)
    height_bias = params.get("height_bias", 0.0)

    return [
        Step(Op.TAG, {"key": "heightfield", "value": True}),
        Step(Op.TAG, {"key": "heightfield_resolution", "value": resolution}),
        Step(Op.TAG, {"key": "heightfield_height_scale", "value": height_scale}),
        Step(Op.TAG, {"key": "heightfield_height_bias", "value": height_bias}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _validate_heightfield_params(**kwargs: Any) -> None:
    """Validate @heightfield parameters."""
    resolution = kwargs.get("resolution", 1024)
    if resolution <= 0 or (resolution & (resolution - 1)) != 0:
        raise ValueError(f"resolution must be a power of 2, got {resolution}")

    height_scale = kwargs.get("height_scale", 1.0)
    if height_scale <= 0:
        raise ValueError(f"height_scale must be > 0, got {height_scale}")


def _heightfield_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @heightfield is applied."""
    _set_registered_tiers(obj)
    obj._heightfield = True
    obj._heightfield_params = {
        "resolution": params.get("resolution", 1024),
        "height_scale": params.get("height_scale", 1.0),
        "height_bias": params.get("height_bias", 0.0),
    }


heightfield = make_decorator(
    name="heightfield",
    steps=_heightfield_steps,
    doc="Heightfield configuration with resolution and scale.",
    validate=_validate_heightfield_params,
    after_steps=_heightfield_after_apply,
)


def _terrain_layer_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @terrain_layer decorator."""
    index = params.get("index", 0)
    name = params.get("name", "")
    blend_mode = params.get("blend_mode", "alpha")

    return [
        Step(Op.TAG, {"key": "terrain_layer", "value": True}),
        Step(Op.TAG, {"key": "terrain_layer_index", "value": index}),
        Step(Op.TAG, {"key": "terrain_layer_name", "value": name}),
        Step(Op.TAG, {"key": "terrain_layer_blend_mode", "value": blend_mode}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _validate_terrain_layer_params(**kwargs: Any) -> None:
    """Validate @terrain_layer parameters."""
    index = kwargs.get("index", 0)
    if not isinstance(index, int) or index < 0:
        raise ValueError(f"index must be a non-negative integer, got {index}")

    blend_mode = kwargs.get("blend_mode", "alpha")
    if blend_mode not in VALID_BLEND_MODES:
        raise ValueError(f"Invalid blend_mode '{blend_mode}'. Must be one of {sorted(VALID_BLEND_MODES)}")


def _terrain_layer_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @terrain_layer is applied."""
    _set_registered_tiers(obj)
    obj._terrain_layer = True
    obj._terrain_layer_params = {
        "index": params.get("index", 0),
        "name": params.get("name", ""),
        "blend_mode": params.get("blend_mode", "alpha"),
    }


terrain_layer = make_decorator(
    name="terrain_layer",
    steps=_terrain_layer_steps,
    doc="Terrain layer configuration with index, name, and blend mode.",
    validate=_validate_terrain_layer_params,
    after_steps=_terrain_layer_after_apply,
)


def _grass_type_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @grass_type decorator."""
    density = params.get("density", 1.0)
    blade_height = params.get("blade_height", 0.5)
    color_variation = params.get("color_variation", 0.1)

    return [
        Step(Op.TAG, {"key": "grass_type", "value": True}),
        Step(Op.TAG, {"key": "grass_type_density", "value": density}),
        Step(Op.TAG, {"key": "grass_type_blade_height", "value": blade_height}),
        Step(Op.TAG, {"key": "grass_type_color_variation", "value": color_variation}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _validate_grass_type_params(**kwargs: Any) -> None:
    """Validate @grass_type parameters."""
    density = kwargs.get("density", 1.0)
    if density <= 0:
        raise ValueError(f"density must be > 0, got {density}")

    blade_height = kwargs.get("blade_height", 0.5)
    if blade_height <= 0:
        raise ValueError(f"blade_height must be > 0, got {blade_height}")

    color_variation = kwargs.get("color_variation", 0.1)
    if not (0 <= color_variation <= 1):
        raise ValueError(f"color_variation must be between 0 and 1, got {color_variation}")


def _grass_type_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @grass_type is applied."""
    _set_registered_tiers(obj)
    obj._grass_type = True
    obj._grass_type_params = {
        "density": params.get("density", 1.0),
        "blade_height": params.get("blade_height", 0.5),
        "color_variation": params.get("color_variation", 0.1),
    }


grass_type = make_decorator(
    name="grass_type",
    steps=_grass_type_steps,
    doc="Grass type configuration with density, blade height, and color variation.",
    validate=_validate_grass_type_params,
    after_steps=_grass_type_after_apply,
)


def _scatter_rule_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @scatter_rule decorator."""
    noise_type = params.get("noise_type", "perlin")
    density = params.get("density", 1.0)
    slope_range = params.get("slope_range", (0.0, 90.0))
    height_range = params.get("height_range")

    return [
        Step(Op.TAG, {"key": "scatter_rule", "value": True}),
        Step(Op.TAG, {"key": "scatter_rule_noise_type", "value": noise_type}),
        Step(Op.TAG, {"key": "scatter_rule_density", "value": density}),
        Step(Op.TAG, {"key": "scatter_rule_slope_range", "value": slope_range}),
        Step(Op.TAG, {"key": "scatter_rule_height_range", "value": height_range}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _validate_scatter_rule_params(**kwargs: Any) -> None:
    """Validate @scatter_rule parameters."""
    noise_type = kwargs.get("noise_type", "perlin")
    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(f"Invalid noise_type '{noise_type}'. Must be one of {sorted(VALID_NOISE_TYPES)}")

    slope_range = kwargs.get("slope_range", (0.0, 90.0))
    if slope_range[0] > slope_range[1]:
        raise ValueError(f"slope_range[0] must be <= slope_range[1], got {slope_range}")

    height_range = kwargs.get("height_range")
    if height_range is not None and height_range[0] > height_range[1]:
        raise ValueError(f"height_range[0] must be <= height_range[1], got {height_range}")


def _scatter_rule_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @scatter_rule is applied."""
    _set_registered_tiers(obj)
    obj._scatter_rule = True
    obj._scatter_rule_params = {
        "noise_type": params.get("noise_type", "perlin"),
        "density": params.get("density", 1.0),
        "slope_range": params.get("slope_range", (0.0, 90.0)),
        "height_range": params.get("height_range"),
    }


scatter_rule = make_decorator(
    name="scatter_rule",
    steps=_scatter_rule_steps,
    doc="Scatter rule configuration with noise type, density, and terrain constraints.",
    validate=_validate_scatter_rule_params,
    after_steps=_scatter_rule_after_apply,
)


def _biome_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @biome decorator."""
    climate_zone = params.get("climate_zone", "temperate")
    vegetation_set = params.get("vegetation_set", [])
    terrain_materials = params.get("terrain_materials", [])

    return [
        Step(Op.TAG, {"key": "biome", "value": True}),
        Step(Op.TAG, {"key": "biome_climate_zone", "value": climate_zone}),
        Step(Op.TAG, {"key": "biome_vegetation_set", "value": vegetation_set}),
        Step(Op.TAG, {"key": "biome_terrain_materials", "value": terrain_materials}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _validate_biome_params(**kwargs: Any) -> None:
    """Validate @biome parameters."""
    climate_zone = kwargs.get("climate_zone", "temperate")
    if climate_zone not in VALID_CLIMATE_ZONES:
        raise ValueError(f"Invalid climate_zone '{climate_zone}'. Must be one of {sorted(VALID_CLIMATE_ZONES)}")


def _biome_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @biome is applied."""
    _set_registered_tiers(obj)
    obj._biome = True
    obj._biome_params = {
        "climate_zone": params.get("climate_zone", "temperate"),
        "vegetation_set": params.get("vegetation_set", []),
        "terrain_materials": params.get("terrain_materials", []),
    }


biome = make_decorator(
    name="biome",
    steps=_biome_steps,
    doc="Biome configuration with climate zone, vegetation, and terrain materials.",
    validate=_validate_biome_params,
    after_steps=_biome_after_apply,
)


def _weather_zone_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @weather_zone decorator."""
    coverage_range = params.get("coverage_range", (0.0, 1.0))
    wind_range = params.get("wind_range", (0.0, 10.0))
    fog_range = params.get("fog_range", (0.0, 1.0))

    return [
        Step(Op.TAG, {"key": "weather_zone", "value": True}),
        Step(Op.TAG, {"key": "weather_zone_coverage_range", "value": coverage_range}),
        Step(Op.TAG, {"key": "weather_zone_wind_range", "value": wind_range}),
        Step(Op.TAG, {"key": "weather_zone_fog_range", "value": fog_range}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _validate_weather_zone_params(**kwargs: Any) -> None:
    """Validate @weather_zone parameters."""
    coverage_range = kwargs.get("coverage_range", (0.0, 1.0))
    if coverage_range[0] > coverage_range[1]:
        raise ValueError(f"coverage_range[0] must be <= coverage_range[1], got {coverage_range}")

    wind_range = kwargs.get("wind_range", (0.0, 10.0))
    if wind_range[0] < 0 or wind_range[1] < 0:
        raise ValueError(f"wind_range values must be >= 0, got {wind_range}")

    fog_range = kwargs.get("fog_range", (0.0, 1.0))
    if fog_range[0] < 0 or fog_range[1] > 1:
        raise ValueError(f"fog_range values must be between 0 and 1, got {fog_range}")


def _weather_zone_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @weather_zone is applied."""
    _set_registered_tiers(obj)
    obj._weather_zone = True
    obj._weather_zone_params = {
        "coverage_range": params.get("coverage_range", (0.0, 1.0)),
        "wind_range": params.get("wind_range", (0.0, 10.0)),
        "fog_range": params.get("fog_range", (0.0, 1.0)),
    }


weather_zone = make_decorator(
    name="weather_zone",
    steps=_weather_zone_steps,
    doc="Weather zone configuration with coverage, wind, and fog ranges.",
    validate=_validate_weather_zone_params,
    after_steps=_weather_zone_after_apply,
)


def _hlod_layer_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @hlod_layer decorator."""
    level = params.get("level", 0)
    cell_size = params.get("cell_size", 1000.0)
    merge_threshold = params.get("merge_threshold", 0.5)

    return [
        Step(Op.TAG, {"key": "hlod_layer", "value": True}),
        Step(Op.TAG, {"key": "hlod_layer_level", "value": level}),
        Step(Op.TAG, {"key": "hlod_layer_cell_size", "value": cell_size}),
        Step(Op.TAG, {"key": "hlod_layer_merge_threshold", "value": merge_threshold}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _validate_hlod_layer_params(**kwargs: Any) -> None:
    """Validate @hlod_layer parameters."""
    level = kwargs.get("level", 0)
    if not isinstance(level, int) or level < 0:
        raise ValueError(f"level must be a non-negative integer, got {level}")

    cell_size = kwargs.get("cell_size", 1000.0)
    if cell_size <= 0:
        raise ValueError(f"cell_size must be > 0, got {cell_size}")

    merge_threshold = kwargs.get("merge_threshold", 0.5)
    if not (0 <= merge_threshold <= 1):
        raise ValueError(f"merge_threshold must be between 0 and 1, got {merge_threshold}")


def _hlod_layer_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @hlod_layer is applied."""
    _set_registered_tiers(obj)
    obj._hlod_layer = True
    obj._hlod_layer_params = {
        "level": params.get("level", 0),
        "cell_size": params.get("cell_size", 1000.0),
        "merge_threshold": params.get("merge_threshold", 0.5),
    }


hlod_layer = make_decorator(
    name="hlod_layer",
    steps=_hlod_layer_steps,
    doc="HLOD layer configuration with level, cell size, and merge threshold.",
    validate=_validate_hlod_layer_params,
    after_steps=_hlod_layer_after_apply,
)


def _environment_volume_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @environment_volume decorator."""
    shape = params.get("shape", "box")
    blend_radius = params.get("blend_radius", 100.0)
    priority = params.get("priority", 0)

    return [
        Step(Op.TAG, {"key": "environment_volume", "value": True}),
        Step(Op.TAG, {"key": "environment_volume_shape", "value": shape}),
        Step(Op.TAG, {"key": "environment_volume_blend_radius", "value": blend_radius}),
        Step(Op.TAG, {"key": "environment_volume_priority", "value": priority}),
        Step(Op.REGISTER, {"registry": "world_building"}),
    ]


def _validate_environment_volume_params(**kwargs: Any) -> None:
    """Validate @environment_volume parameters."""
    shape = kwargs.get("shape", "box")
    if shape not in VALID_VOLUME_SHAPES:
        raise ValueError(f"Invalid shape '{shape}'. Must be one of {sorted(VALID_VOLUME_SHAPES)}")

    blend_radius = kwargs.get("blend_radius", 100.0)
    if blend_radius < 0:
        raise ValueError(f"blend_radius must be >= 0, got {blend_radius}")

    priority = kwargs.get("priority", 0)
    if not isinstance(priority, int):
        raise ValueError(f"priority must be an integer, got {type(priority).__name__}")


def _environment_volume_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @environment_volume is applied."""
    _set_registered_tiers(obj)
    obj._environment_volume = True
    obj._environment_volume_params = {
        "shape": params.get("shape", "box"),
        "blend_radius": params.get("blend_radius", 100.0),
        "priority": params.get("priority", 0),
    }


environment_volume = make_decorator(
    name="environment_volume",
    steps=_environment_volume_steps,
    doc="Environment volume configuration with shape, blend radius, and priority.",
    validate=_validate_environment_volume_params,
    after_steps=_environment_volume_after_apply,
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
    # New terrain/environment decorators
    ("terrain_patch", terrain_patch, ("class",)),
    ("heightfield", heightfield, ("class",)),
    ("terrain_layer", terrain_layer, ("class",)),
    ("grass_type", grass_type, ("class",)),
    ("scatter_rule", scatter_rule, ("class",)),
    ("biome", biome, ("class",)),
    ("weather_zone", weather_zone, ("class",)),
    ("hlod_layer", hlod_layer, ("class",)),
    ("environment_volume", environment_volume, ("class",)),
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
    # Original decorators
    "foliage_type",
    "procedural_placement",
    "level_instance",
    "water_body",
    "navmesh_modifier",
    "trigger_volume",
    # New terrain/environment decorators
    "terrain_patch",
    "heightfield",
    "terrain_layer",
    "grass_type",
    "scatter_rule",
    "biome",
    "weather_zone",
    "hlod_layer",
    "environment_volume",
    # Constants
    "VALID_WATER_TYPES",
    "VALID_NAVMESH_MODIFIERS",
    "VALID_TRIGGER_EVENTS",
    "VALID_BLEND_MODES",
    "VALID_CLIMATE_ZONES",
    "VALID_NOISE_TYPES",
    "VALID_VOLUME_SHAPES",
]
