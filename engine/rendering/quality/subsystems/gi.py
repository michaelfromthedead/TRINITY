"""Quality tier configuration for Global Illumination subsystem (S6)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class GICapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Global Illumination subsystem.

    Configures indirect lighting, DDGI, reflections, and AO per tier.
    """

    @property
    def subsystem_name(self) -> str:
        return "gi"

    def _init_tier_configs(self) -> None:
        # LOW tier: Baked lightmaps only
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "ambient_light",
                    "baked_lightmaps",
                }),
                disabled_features=frozenset({
                    "ssao",
                    "light_probes",
                    "irradiance_volumes",
                    "ddgi",
                    "rtgi",
                    "ssr",
                }),
                parameters={
                    "ambient_intensity": 0.3,
                    "lightmap_scale": 0.5,
                },
            ),
        )

        # MEDIUM tier: Light probes + SSAO
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "ambient_light",
                    "baked_lightmaps",
                    "light_probes",
                    "ssao",
                }),
                parameters={
                    "ambient_intensity": 0.2,
                    "ssao_samples": 8,
                    "ssao_radius": 0.5,
                    "probe_density": "low",
                },
            ),
        )

        # HIGH tier: DDGI + SSR
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "ambient_light",
                    "baked_lightmaps",
                    "light_probes",
                    "irradiance_volumes",
                    "ssao",
                    "gtao",  # Ground Truth AO
                    "ssr",
                    "ddgi",  # Dynamic Diffuse GI
                }),
                parameters={
                    "ambient_intensity": 0.1,
                    "ssao_samples": 16,
                    "ssao_radius": 1.0,
                    "ssr_steps": 32,
                    "ssr_thickness": 0.1,
                    "ddgi_probe_count": (8, 4, 8),
                    "ddgi_rays_per_probe": 128,
                },
            ),
        )

        # ULTRA tier: Full RTGI
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "ambient_light",
                    "baked_lightmaps",
                    "light_probes",
                    "irradiance_volumes",
                    "gtao",
                    "ssr",
                    "ddgi",
                    "rtgi",  # Ray-traced GI
                    "rt_reflections",
                }),
                parameters={
                    "ambient_intensity": 0.05,
                    "gtao_samples": 32,
                    "ssr_steps": 64,
                    "ssr_binary_search_steps": 8,
                    "ddgi_probe_count": (16, 8, 16),
                    "ddgi_rays_per_probe": 256,
                    "rtgi_samples": 2,
                    "rt_reflection_samples": 2,
                },
            ),
        )

        # Resolutions
        self._set_resolution(
            QualityTier.LOW,
            TierResolution(tier=QualityTier.LOW, gi_resolution=64),
        )
        self._set_resolution(
            QualityTier.MEDIUM,
            TierResolution(tier=QualityTier.MEDIUM, gi_resolution=128),
        )
        self._set_resolution(
            QualityTier.HIGH,
            TierResolution(tier=QualityTier.HIGH, gi_resolution=256, reflection_resolution=512),
        )
        self._set_resolution(
            QualityTier.ULTRA,
            TierResolution(tier=QualityTier.ULTRA, gi_resolution=512, reflection_resolution=1024),
        )

        # Budgets
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=0.2, memory_mb=16),
        )
        self._set_budget(
            QualityTier.MEDIUM,
            TierBudget(tier=QualityTier.MEDIUM, gpu_time_ms=1.0, memory_mb=64),
        )
        self._set_budget(
            QualityTier.HIGH,
            TierBudget(tier=QualityTier.HIGH, gpu_time_ms=3.0, memory_mb=256),
        )
        self._set_budget(
            QualityTier.ULTRA,
            TierBudget(tier=QualityTier.ULTRA, gpu_time_ms=6.0, memory_mb=512),
        )

        # Fallback chains
        self._set_fallback(
            "indirect_diffuse",
            FallbackChain(
                primary="rtgi",
                fallbacks=("ddgi", "irradiance_volumes", "light_probes"),
            ),
        )
        self._set_fallback(
            "indirect_specular",
            FallbackChain(
                primary="rt_reflections",
                fallbacks=("ssr_hiz", "ssr_linear", "env_map_only"),
            ),
        )
        self._set_fallback(
            "ambient_occlusion",
            FallbackChain(
                primary="gtao",
                fallbacks=("hbao", "ssao", "none"),
            ),
        )
