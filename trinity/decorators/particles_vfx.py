"""
Trinity Pattern - Tier 45: PARTICLES_VFX Decorators

Particle emitters, VFX events, GPU particles, trails, and decals decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_PARTICLE_SIMULATIONS = frozenset({"cpu", "gpu", "auto"})
VALID_PARTICLE_STAGES = frozenset({"spawn", "update", "render"})
VALID_VFX_TRIGGERS = frozenset({"spawn", "death", "collision", "custom"})
VALID_TEXTURE_MODES = frozenset({"stretch", "tile"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _particle_emitter_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @particle_emitter decorator."""
    max_particles = params.get("max_particles", 1000)
    simulation = params.get("simulation", "auto")
    budget_category = params.get("budget_category")

    return [
        Step(Op.TAG, {"key": "particle_emitter", "value": True}),
        Step(Op.TAG, {"key": "particle_max_particles", "value": max_particles}),
        Step(Op.TAG, {"key": "particle_simulation", "value": simulation}),
        Step(Op.TAG, {"key": "particle_budget_category", "value": budget_category}),
        Step(Op.REGISTER, {"registry": "particles_vfx"}),
    ]


def _particle_module_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @particle_module decorator."""
    stage = params.get("stage")
    lod_range = params.get("lod_range", (0, 3))

    return [
        Step(Op.TAG, {"key": "particle_module", "value": True}),
        Step(Op.TAG, {"key": "particle_module_stage", "value": stage}),
        Step(Op.TAG, {"key": "particle_module_lod_range", "value": lod_range}),
        Step(Op.REGISTER, {"registry": "particles_vfx"}),
    ]


def _vfx_event_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @vfx_event decorator."""
    trigger = params.get("trigger")

    return [
        Step(Op.TAG, {"key": "vfx_event", "value": True}),
        Step(Op.TAG, {"key": "vfx_event_trigger", "value": trigger}),
        Step(Op.REGISTER, {"registry": "particles_vfx"}),
    ]


def _gpu_particle_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @gpu_particle decorator."""
    attributes = params.get("attributes", [])
    compute_shader = params.get("compute_shader")

    return [
        Step(Op.TAG, {"key": "gpu_particle", "value": True}),
        Step(Op.TAG, {"key": "gpu_particle_attributes", "value": list(attributes)}),
        Step(Op.TAG, {"key": "gpu_particle_compute_shader", "value": compute_shader}),
        Step(Op.REGISTER, {"registry": "particles_vfx"}),
    ]


def _trail_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @trail decorator."""
    width = params.get("width", 0.1)
    fade_time = params.get("fade_time", 1.0)
    texture_mode = params.get("texture_mode", "stretch")

    return [
        Step(Op.TAG, {"key": "trail", "value": True}),
        Step(Op.TAG, {"key": "trail_width", "value": width}),
        Step(Op.TAG, {"key": "trail_fade_time", "value": fade_time}),
        Step(Op.TAG, {"key": "trail_texture_mode", "value": texture_mode}),
        Step(Op.REGISTER, {"registry": "particles_vfx"}),
    ]


def _decal_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @decal decorator."""
    lifetime = params.get("lifetime")
    fade_time = params.get("fade_time", 1.0)
    channel = params.get("channel", 0)

    return [
        Step(Op.TAG, {"key": "decal", "value": True}),
        Step(Op.TAG, {"key": "decal_lifetime", "value": lifetime}),
        Step(Op.TAG, {"key": "decal_fade_time", "value": fade_time}),
        Step(Op.TAG, {"key": "decal_channel", "value": channel}),
        Step(Op.REGISTER, {"registry": "particles_vfx"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_particle_emitter_params(**kwargs: Any) -> None:
    """Validate @particle_emitter parameters."""
    max_particles = kwargs.get("max_particles", 1000)
    if not isinstance(max_particles, int) or max_particles <= 0:
        raise ValueError(f"max_particles must be > 0, got {max_particles}")

    simulation = kwargs.get("simulation", "auto")
    if simulation not in VALID_PARTICLE_SIMULATIONS:
        raise ValueError(
            f"Invalid simulation '{simulation}'. Must be one of {sorted(VALID_PARTICLE_SIMULATIONS)}"
        )


def _validate_particle_module_params(**kwargs: Any) -> None:
    """Validate @particle_module parameters."""
    stage = kwargs.get("stage")
    if stage not in VALID_PARTICLE_STAGES:
        raise ValueError(
            f"Invalid stage '{stage}'. Must be one of {sorted(VALID_PARTICLE_STAGES)}"
        )

    lod_range = kwargs.get("lod_range", (0, 3))
    if not isinstance(lod_range, tuple) or len(lod_range) != 2:
        raise ValueError(f"lod_range must be a tuple of 2 integers, got {lod_range}")
    if lod_range[0] > lod_range[1]:
        raise ValueError(
            f"lod_range[0] must be <= lod_range[1], got {lod_range}"
        )


def _validate_vfx_event_params(**kwargs: Any) -> None:
    """Validate @vfx_event parameters."""
    trigger = kwargs.get("trigger")
    if trigger not in VALID_VFX_TRIGGERS:
        raise ValueError(
            f"Invalid trigger '{trigger}'. Must be one of {sorted(VALID_VFX_TRIGGERS)}"
        )


def _validate_gpu_particle_params(**kwargs: Any) -> None:
    """Validate @gpu_particle parameters."""
    attributes = kwargs.get("attributes", [])
    if not isinstance(attributes, list) or len(attributes) == 0:
        raise ValueError("attributes must be a non-empty list")


def _validate_trail_params(**kwargs: Any) -> None:
    """Validate @trail parameters."""
    width = kwargs.get("width", 0.1)
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")

    fade_time = kwargs.get("fade_time", 1.0)
    if fade_time <= 0:
        raise ValueError(f"fade_time must be > 0, got {fade_time}")

    texture_mode = kwargs.get("texture_mode", "stretch")
    if texture_mode not in VALID_TEXTURE_MODES:
        raise ValueError(
            f"Invalid texture_mode '{texture_mode}'. Must be one of {sorted(VALID_TEXTURE_MODES)}"
        )


def _validate_decal_params(**kwargs: Any) -> None:
    """Validate @decal parameters."""
    lifetime = kwargs.get("lifetime")
    if lifetime is not None and lifetime <= 0:
        raise ValueError(f"lifetime must be > 0 or None, got {lifetime}")

    fade_time = kwargs.get("fade_time", 1.0)
    if fade_time < 0:
        raise ValueError(f"fade_time must be >= 0, got {fade_time}")

    channel = kwargs.get("channel", 0)
    if channel < 0:
        raise ValueError(f"channel must be >= 0, got {channel}")


# ============================================================================
# After-apply functions
# ============================================================================


def _particle_emitter_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @particle_emitter is applied."""
    max_particles = params.get("max_particles", 1000)
    simulation = params.get("simulation", "auto")
    budget_category = params.get("budget_category")

    obj._particle_emitter = True
    obj._particle_max_particles = max_particles
    obj._particle_simulation = simulation
    obj._particle_budget_category = budget_category


def _particle_module_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @particle_module is applied."""
    stage = params.get("stage")
    lod_range = params.get("lod_range", (0, 3))

    obj._particle_module = True
    obj._particle_module_stage = stage
    obj._particle_module_lod_range = lod_range


def _vfx_event_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @vfx_event is applied."""
    trigger = params.get("trigger")

    obj._vfx_event = True
    obj._vfx_event_trigger = trigger


def _gpu_particle_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @gpu_particle is applied."""
    attributes = params.get("attributes", [])
    compute_shader = params.get("compute_shader")

    obj._gpu_particle = True
    obj._gpu_particle_attributes = list(attributes)
    obj._gpu_particle_compute_shader = compute_shader


def _trail_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @trail is applied."""
    width = params.get("width", 0.1)
    fade_time = params.get("fade_time", 1.0)
    texture_mode = params.get("texture_mode", "stretch")

    obj._trail = True
    obj._trail_width = width
    obj._trail_fade_time = fade_time
    obj._trail_texture_mode = texture_mode


def _decal_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @decal is applied."""
    lifetime = params.get("lifetime")
    fade_time = params.get("fade_time", 1.0)
    channel = params.get("channel", 0)

    obj._decal = True
    obj._decal_lifetime = lifetime
    obj._decal_fade_time = fade_time
    obj._decal_channel = channel


# ============================================================================
# Decorator creation
# ============================================================================

particle_emitter = make_decorator(
    name="particle_emitter",
    steps=_particle_emitter_steps,
    doc="Particle emitter configuration.",
    validate=_validate_particle_emitter_params,
    after_steps=_particle_emitter_after_apply,
)

particle_module = make_decorator(
    name="particle_module",
    steps=_particle_module_steps,
    doc="Particle behavior module configuration.",
    validate=_validate_particle_module_params,
    after_steps=_particle_module_after_apply,
)

vfx_event = make_decorator(
    name="vfx_event",
    steps=_vfx_event_steps,
    doc="VFX event trigger configuration.",
    validate=_validate_vfx_event_params,
    after_steps=_vfx_event_after_apply,
)

gpu_particle = make_decorator(
    name="gpu_particle",
    steps=_gpu_particle_steps,
    doc="GPU particle system configuration.",
    validate=_validate_gpu_particle_params,
    after_steps=_gpu_particle_after_apply,
)

trail = make_decorator(
    name="trail",
    steps=_trail_steps,
    doc="Trail renderer configuration.",
    validate=_validate_trail_params,
    after_steps=_trail_after_apply,
)

decal = make_decorator(
    name="decal",
    steps=_decal_steps,
    doc="Projected decal configuration.",
    validate=_validate_decal_params,
    after_steps=_decal_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("particle_emitter", particle_emitter, ("class",)),
    ("particle_module", particle_module, ("class",)),
    ("vfx_event", vfx_event, ("class",)),
    ("gpu_particle", gpu_particle, ("class",)),
    ("trail", trail, ("class",)),
    ("decal", decal, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.PARTICLES_VFX,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.PARTICLES_VFX].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "particle_emitter",
    "particle_module",
    "vfx_event",
    "gpu_particle",
    "trail",
    "decal",
    "VALID_PARTICLE_SIMULATIONS",
    "VALID_PARTICLE_STAGES",
    "VALID_VFX_TRIGGERS",
    "VALID_TEXTURE_MODES",
]
