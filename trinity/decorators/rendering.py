"""
Trinity Pattern - Tier 42: RENDERING Decorators

Rendering pipeline decorators for GI contribution, shadow casting,
reflection probes, material properties, and render layers.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_GI_IMPORTANCE = frozenset({"low", "medium", "high", "critical"})
VALID_SHADOW_MODE = frozenset({"static", "dynamic", "none"})
VALID_CAPTURE_MODE = frozenset({"baked", "realtime", "mixed"})
VALID_MATERIAL_DOMAIN = frozenset({"surface", "deferred_decal", "volume", "post_process", "ui"})
VALID_BLEND_MODE = frozenset({"opaque", "masked", "translucent", "additive", "modulate"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Configuration dataclasses
# ============================================================================


@dataclass(frozen=True)
class GIContributorConfig:
    """GI contribution configuration."""

    importance: str
    emissive: bool


@dataclass(frozen=True)
class ShadowCasterConfig:
    """Shadow casting configuration."""

    mode: str
    resolution_scale: float
    cascade_bias: float


@dataclass(frozen=True)
class ReflectionProbeConfig:
    """Reflection probe configuration."""

    capture_mode: str
    resolution: int
    update_rate: float


@dataclass(frozen=True)
class MaterialDomainConfig:
    """Material domain configuration."""

    domain: str


@dataclass(frozen=True)
class MaterialBlendConfig:
    """Material blend mode configuration."""

    mode: str


@dataclass(frozen=True)
class RenderLayerConfig:
    """Render layer configuration."""

    layer: str
    order: int


# ============================================================================
# Step builders
# ============================================================================


def _gi_contributor_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @gi_contributor decorator."""
    importance = params.get("importance", "medium")
    emissive = params.get("emissive", False)

    return [
        Step(Op.TAG, {"key": "gi_contributor", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "gi_config",
                "value": GIContributorConfig(importance=importance, emissive=emissive),
            },
        ),
        Step(Op.REGISTER, {"registry": "rendering"}),
    ]


def _shadow_caster_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @shadow_caster decorator."""
    mode = params.get("mode", "dynamic")
    resolution_scale = params.get("resolution_scale", 1.0)
    cascade_bias = params.get("cascade_bias", 0.0)

    return [
        Step(Op.TAG, {"key": "shadow_caster", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "shadow_config",
                "value": ShadowCasterConfig(
                    mode=mode,
                    resolution_scale=resolution_scale,
                    cascade_bias=cascade_bias,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "rendering"}),
    ]


def _reflection_probe_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @reflection_probe decorator."""
    capture_mode = params.get("capture_mode", "baked")
    resolution = params.get("resolution", 256)
    update_rate = params.get("update_rate", 0.0)

    return [
        Step(Op.TAG, {"key": "reflection_probe", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "reflection_config",
                "value": ReflectionProbeConfig(
                    capture_mode=capture_mode,
                    resolution=resolution,
                    update_rate=update_rate,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "rendering"}),
    ]


def _material_domain_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @material_domain decorator."""
    domain = params["domain"]  # REQUIRED parameter

    return [
        Step(Op.TAG, {"key": "material_domain", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "material_domain_config",
                "value": MaterialDomainConfig(domain=domain),
            },
        ),
        Step(Op.REGISTER, {"registry": "rendering"}),
    ]


def _material_blend_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @material_blend decorator."""
    mode = params["mode"]  # REQUIRED parameter

    return [
        Step(Op.TAG, {"key": "material_blend", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "material_blend_config",
                "value": MaterialBlendConfig(mode=mode),
            },
        ),
        Step(Op.REGISTER, {"registry": "rendering"}),
    ]


def _render_layer_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @render_layer decorator."""
    layer = params["layer"]  # REQUIRED parameter
    order = params.get("order", 0)

    return [
        Step(Op.TAG, {"key": "render_layer", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "render_layer_config",
                "value": RenderLayerConfig(layer=layer, order=order),
            },
        ),
        Step(Op.REGISTER, {"registry": "rendering"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_gi_contributor_params(**kwargs: Any) -> None:
    """Validate @gi_contributor parameters."""
    importance = kwargs.get("importance", "medium")
    if importance not in VALID_GI_IMPORTANCE:
        raise ValueError(
            f"Invalid importance '{importance}'. Must be one of {VALID_GI_IMPORTANCE}"
        )


def _validate_shadow_caster_params(**kwargs: Any) -> None:
    """Validate @shadow_caster parameters."""
    mode = kwargs.get("mode", "dynamic")
    if mode not in VALID_SHADOW_MODE:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of {VALID_SHADOW_MODE}"
        )

    resolution_scale = kwargs.get("resolution_scale", 1.0)
    if resolution_scale <= 0:
        raise ValueError(f"resolution_scale must be > 0, got {resolution_scale}")


def _validate_reflection_probe_params(**kwargs: Any) -> None:
    """Validate @reflection_probe parameters."""
    capture_mode = kwargs.get("capture_mode", "baked")
    if capture_mode not in VALID_CAPTURE_MODE:
        raise ValueError(
            f"Invalid capture_mode '{capture_mode}'. Must be one of {VALID_CAPTURE_MODE}"
        )

    resolution = kwargs.get("resolution", 256)
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")


def _validate_material_domain_params(**kwargs: Any) -> None:
    """Validate @material_domain parameters."""
    domain = kwargs.get("domain")
    if domain not in VALID_MATERIAL_DOMAIN:
        raise ValueError(
            f"Invalid domain '{domain}'. Must be one of {VALID_MATERIAL_DOMAIN}"
        )


def _validate_material_blend_params(**kwargs: Any) -> None:
    """Validate @material_blend parameters."""
    mode = kwargs.get("mode")
    if mode not in VALID_BLEND_MODE:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of {VALID_BLEND_MODE}"
        )


def _validate_render_layer_params(**kwargs: Any) -> None:
    """Validate @render_layer parameters."""
    layer = kwargs.get("layer")
    if not layer:
        raise ValueError("layer must be a non-empty string")


# ============================================================================
# After-apply functions
# ============================================================================


def _gi_contributor_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @gi_contributor is applied."""
    importance = params.get("importance", "medium")
    emissive = params.get("emissive", False)

    obj._gi_contributor = True
    obj._gi_importance = importance
    obj._gi_emissive = emissive
    obj._gi_config = GIContributorConfig(importance=importance, emissive=emissive)


def _shadow_caster_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @shadow_caster is applied."""
    mode = params.get("mode", "dynamic")
    resolution_scale = params.get("resolution_scale", 1.0)
    cascade_bias = params.get("cascade_bias", 0.0)

    obj._shadow_caster = True
    obj._shadow_mode = mode
    obj._shadow_resolution_scale = resolution_scale
    obj._shadow_cascade_bias = cascade_bias
    obj._shadow_config = ShadowCasterConfig(
        mode=mode,
        resolution_scale=resolution_scale,
        cascade_bias=cascade_bias,
    )


def _reflection_probe_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @reflection_probe is applied."""
    capture_mode = params.get("capture_mode", "baked")
    resolution = params.get("resolution", 256)
    update_rate = params.get("update_rate", 0.0)

    obj._reflection_probe = True
    obj._reflection_capture_mode = capture_mode
    obj._reflection_resolution = resolution
    obj._reflection_update_rate = update_rate
    obj._reflection_config = ReflectionProbeConfig(
        capture_mode=capture_mode,
        resolution=resolution,
        update_rate=update_rate,
    )


def _material_domain_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @material_domain is applied."""
    domain = params["domain"]  # REQUIRED parameter

    obj._material_domain = True
    obj._material_domain_type = domain
    obj._material_domain_config = MaterialDomainConfig(domain=domain)


def _material_blend_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @material_blend is applied."""
    mode = params["mode"]  # REQUIRED parameter

    obj._material_blend = True
    obj._material_blend_mode = mode
    obj._material_blend_config = MaterialBlendConfig(mode=mode)


def _render_layer_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @render_layer is applied."""
    layer = params["layer"]  # REQUIRED parameter
    order = params.get("order", 0)

    obj._render_layer = True
    obj._render_layer_name = layer
    obj._render_layer_order = order
    obj._render_layer_config = RenderLayerConfig(layer=layer, order=order)


# ============================================================================
# Decorator creation
# ============================================================================

gi_contributor = make_decorator(
    name="gi_contributor",
    steps=_gi_contributor_steps,
    validate=_validate_gi_contributor_params,
    after_steps=_gi_contributor_after_apply,
)

shadow_caster = make_decorator(
    name="shadow_caster",
    steps=_shadow_caster_steps,
    validate=_validate_shadow_caster_params,
    after_steps=_shadow_caster_after_apply,
)

reflection_probe = make_decorator(
    name="reflection_probe",
    steps=_reflection_probe_steps,
    validate=_validate_reflection_probe_params,
    after_steps=_reflection_probe_after_apply,
)

material_domain = make_decorator(
    name="material_domain",
    steps=_material_domain_steps,
    validate=_validate_material_domain_params,
    after_steps=_material_domain_after_apply,
)

material_blend = make_decorator(
    name="material_blend",
    steps=_material_blend_steps,
    validate=_validate_material_blend_params,
    after_steps=_material_blend_after_apply,
)

render_layer = make_decorator(
    name="render_layer",
    steps=_render_layer_steps,
    validate=_validate_render_layer_params,
    after_steps=_render_layer_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("gi_contributor", gi_contributor, ("class",)),
    ("shadow_caster", shadow_caster, ("class",)),
    ("reflection_probe", reflection_probe, ("class",)),
    ("material_domain", material_domain, ("class",)),
    ("material_blend", material_blend, ("class",)),
    ("render_layer", render_layer, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.RENDERING,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.RENDERING].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "gi_contributor",
    "shadow_caster",
    "reflection_probe",
    "material_domain",
    "material_blend",
    "render_layer",
    "GIContributorConfig",
    "ShadowCasterConfig",
    "ReflectionProbeConfig",
    "MaterialDomainConfig",
    "MaterialBlendConfig",
    "RenderLayerConfig",
    "VALID_GI_IMPORTANCE",
    "VALID_SHADOW_MODE",
    "VALID_CAPTURE_MODE",
    "VALID_MATERIAL_DOMAIN",
    "VALID_BLEND_MODE",
]
