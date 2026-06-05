"""Quality tier configuration for Ray Tracing subsystem (S10)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class RayTracingCapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Ray Tracing subsystem.

    Configures RT features and fallbacks per tier. Note: Most RT features
    require ULTRA tier; lower tiers use rasterized fallbacks.
    """

    @property
    def subsystem_name(self) -> str:
        return "raytracing"

    def _init_tier_configs(self) -> None:
        # LOW tier: No RT, fully rasterized
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "rasterized_shadows",
                    "rasterized_ao",
                }),
                disabled_features=frozenset({
                    "rt_shadows",
                    "rt_ao",
                    "rt_reflections",
                    "rt_gi",
                    "inline_ray_query",
                }),
                parameters={
                    "rt_available": False,
                },
            ),
        )

        # MEDIUM tier: No RT, rasterized with better quality
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "rasterized_shadows",
                    "rasterized_ao",
                    "ssao",
                    "ssr",
                }),
                parameters={
                    "rt_available": False,
                },
            ),
        )

        # HIGH tier: Inline ray queries (if supported)
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "rasterized_shadows",
                    "rasterized_ao",
                    "ssao",
                    "ssr",
                    "inline_ray_query",
                    "rt_shadows",  # Via inline queries
                }),
                parameters={
                    "rt_available": True,
                    "rt_shadow_rays": 1,
                    "use_denoiser": True,
                },
            ),
        )

        # ULTRA tier: Full RT pipeline
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "inline_ray_query",
                    "rt_shadows",
                    "rt_ao",
                    "rt_reflections",
                    "rt_gi",
                    "rt_soft_shadows",
                    "rt_glossy",
                    "denoiser_temporal",
                    "denoiser_spatial",
                }),
                parameters={
                    "rt_available": True,
                    "rt_shadow_rays": 2,
                    "rt_ao_rays": 4,
                    "rt_reflection_rays": 2,
                    "rt_gi_rays": 4,
                    "rt_bounces": 2,
                    "use_denoiser": True,
                },
            ),
        )

        # Budgets (RT is expensive)
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=0.0, memory_mb=0),
        )
        self._set_budget(
            QualityTier.MEDIUM,
            TierBudget(tier=QualityTier.MEDIUM, gpu_time_ms=0.0, memory_mb=0),
        )
        self._set_budget(
            QualityTier.HIGH,
            TierBudget(tier=QualityTier.HIGH, gpu_time_ms=2.0, memory_mb=128),
        )
        self._set_budget(
            QualityTier.ULTRA,
            TierBudget(tier=QualityTier.ULTRA, gpu_time_ms=6.0, memory_mb=512),
        )

        # Fallback chains
        self._set_fallback(
            "shadows",
            FallbackChain(
                primary="rt_soft_shadows",
                fallbacks=("rt_hard_shadows", "pcss", "vsm"),
            ),
        )
        self._set_fallback(
            "ambient_occlusion",
            FallbackChain(
                primary="rt_ao",
                fallbacks=("gtao", "hbao", "ssao"),
            ),
        )
        self._set_fallback(
            "global_illumination",
            FallbackChain(
                primary="rt_gi",
                fallbacks=("ddgi", "irradiance_volumes", "light_probes"),
            ),
        )
