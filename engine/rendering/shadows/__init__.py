"""
RT Shadows Module

Provides ray-traced shadow dispatch and fallback shadow techniques.

Classes:
    RTShadowQuality: Quality presets controlling rays per pixel.
    RTShadowParams: Shadow ray configuration parameters.
    RTShadowDispatcher: Main RT shadow dispatch class.
    ShadowFallbackDispatcher: Fallback shadow techniques (CSM, contact shadows).

Functions:
    create_shadow_dispatcher: Factory returning RT or fallback dispatcher.
    estimate_shadow_cost: Performance estimation for shadow rendering.
"""

from .rt_shadows import (
    RTShadowQuality,
    RTShadowParams,
    RTShadowDispatcher,
    ShadowFallbackDispatcher,
    create_shadow_dispatcher,
    estimate_shadow_cost,
)

__all__ = [
    "RTShadowQuality",
    "RTShadowParams",
    "RTShadowDispatcher",
    "ShadowFallbackDispatcher",
    "create_shadow_dispatcher",
    "estimate_shadow_cost",
]
