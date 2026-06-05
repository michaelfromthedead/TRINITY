"""Quality tier configuration for Atmosphere subsystem (S11)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class AtmosphereCapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Atmosphere subsystem.

    Configures sky rendering, clouds, fog, and aerial perspective per tier.
    """

    @property
    def subsystem_name(self) -> str:
        return "atmosphere"

    def _init_tier_configs(self) -> None:
        # LOW tier: Simple gradient sky
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "gradient_sky",
                    "sun_disk",
                    "height_fog",
                }),
                disabled_features=frozenset({
                    "bruneton_scattering",
                    "volumetric_clouds",
                    "aerial_perspective",
                    "god_rays",
                    "stars",
                }),
                parameters={
                    "sky_samples": 4,
                    "fog_density": 0.01,
                },
            ),
        )

        # MEDIUM tier: Precomputed scattering
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "bruneton_scattering",
                    "sun_disk",
                    "moon_disk",
                    "height_fog",
                    "exponential_fog",
                    "stars",
                }),
                parameters={
                    "scattering_lut_size": 128,
                    "sky_samples": 16,
                    "fog_density": 0.01,
                    "star_count": 2000,
                },
            ),
        )

        # HIGH tier: + Volumetric clouds
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "bruneton_scattering",
                    "sun_disk",
                    "moon_disk",
                    "volumetric_clouds",
                    "aerial_perspective",
                    "height_fog",
                    "exponential_fog",
                    "god_rays",
                    "stars",
                }),
                parameters={
                    "scattering_lut_size": 256,
                    "sky_samples": 32,
                    "cloud_samples": 64,
                    "cloud_light_samples": 6,
                    "aerial_samples": 16,
                    "god_ray_samples": 32,
                    "star_count": 5000,
                },
            ),
        )

        # ULTRA tier: Full volumetric atmosphere
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "bruneton_scattering",
                    "sun_disk",
                    "moon_disk",
                    "volumetric_clouds",
                    "aerial_perspective",
                    "height_fog",
                    "exponential_fog",
                    "god_rays",
                    "stars",
                    "cloud_shadows",
                    "temporal_reprojection",
                    "multiple_scattering",
                }),
                parameters={
                    "scattering_lut_size": 512,
                    "sky_samples": 64,
                    "cloud_samples": 128,
                    "cloud_light_samples": 12,
                    "cloud_shadow_samples": 8,
                    "aerial_samples": 32,
                    "god_ray_samples": 64,
                    "star_count": 10000,
                    "temporal_blend": 0.95,
                },
            ),
        )

        # Resolutions
        self._set_resolution(
            QualityTier.LOW,
            TierResolution(tier=QualityTier.LOW, render_scale=0.5),
        )
        self._set_resolution(
            QualityTier.MEDIUM,
            TierResolution(tier=QualityTier.MEDIUM, render_scale=0.75),
        )
        self._set_resolution(
            QualityTier.HIGH,
            TierResolution(tier=QualityTier.HIGH, render_scale=1.0),
        )
        self._set_resolution(
            QualityTier.ULTRA,
            TierResolution(tier=QualityTier.ULTRA, render_scale=1.0),
        )

        # Budgets
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=0.5, memory_mb=16),
        )
        self._set_budget(
            QualityTier.MEDIUM,
            TierBudget(tier=QualityTier.MEDIUM, gpu_time_ms=1.0, memory_mb=32),
        )
        self._set_budget(
            QualityTier.HIGH,
            TierBudget(tier=QualityTier.HIGH, gpu_time_ms=3.0, memory_mb=128),
        )
        self._set_budget(
            QualityTier.ULTRA,
            TierBudget(tier=QualityTier.ULTRA, gpu_time_ms=5.0, memory_mb=256),
        )

        # Fallback chains
        self._set_fallback(
            "sky",
            FallbackChain(
                primary="bruneton_scattering",
                fallbacks=("precomputed_lut", "gradient_sky", "solid_color"),
            ),
        )
        self._set_fallback(
            "clouds",
            FallbackChain(
                primary="volumetric_raymarched",
                fallbacks=("billboard_clouds", "skybox_clouds", "none"),
            ),
        )
        self._set_fallback(
            "fog",
            FallbackChain(
                primary="froxel_volumetric",
                fallbacks=("height_fog", "linear_fog", "none"),
            ),
        )
