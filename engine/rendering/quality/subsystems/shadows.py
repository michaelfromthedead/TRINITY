"""Quality tier configuration for Shadows subsystem (S5)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class ShadowsCapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Shadows subsystem.

    Configures shadow resolution, filtering, and cascade settings per tier.
    """

    @property
    def subsystem_name(self) -> str:
        return "shadows"

    def _init_tier_configs(self) -> None:
        # LOW tier: Basic hard shadows
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "directional_shadow",
                    "pcf_filtering",
                }),
                disabled_features=frozenset({
                    "cascaded_shadows",
                    "vsm",
                    "pcss",
                    "ray_traced_shadows",
                    "contact_shadows",
                }),
                parameters={
                    "cascade_count": 1,
                    "pcf_samples": 4,
                    "shadow_distance": 50.0,
                    "shadow_fade": 0.9,
                },
            ),
        )

        # MEDIUM tier: Cascaded PCF shadows
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "directional_shadow",
                    "spot_shadow",
                    "pcf_filtering",
                    "cascaded_shadows",
                }),
                parameters={
                    "cascade_count": 2,
                    "pcf_samples": 9,
                    "shadow_distance": 100.0,
                    "shadow_fade": 0.85,
                },
            ),
        )

        # HIGH tier: VSM with contact shadows
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "directional_shadow",
                    "spot_shadow",
                    "point_shadow",
                    "cascaded_shadows",
                    "vsm",  # Variance Shadow Maps
                    "contact_shadows",
                }),
                parameters={
                    "cascade_count": 4,
                    "vsm_blur_samples": 5,
                    "shadow_distance": 200.0,
                    "shadow_fade": 0.8,
                    "contact_shadow_steps": 16,
                },
            ),
        )

        # ULTRA tier: Ray-traced shadows
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "directional_shadow",
                    "spot_shadow",
                    "point_shadow",
                    "area_shadow",
                    "cascaded_shadows",
                    "ray_traced_shadows",
                    "contact_shadows",
                    "pcss",  # Percentage Closer Soft Shadows
                }),
                parameters={
                    "cascade_count": 4,
                    "rt_shadow_rays": 2,
                    "shadow_distance": 500.0,
                    "shadow_fade": 0.75,
                    "contact_shadow_steps": 32,
                    "pcss_blocker_samples": 16,
                    "pcss_pcf_samples": 32,
                },
            ),
        )

        # Resolutions
        self._set_resolution(
            QualityTier.LOW,
            TierResolution(tier=QualityTier.LOW, shadow_resolution=512),
        )
        self._set_resolution(
            QualityTier.MEDIUM,
            TierResolution(tier=QualityTier.MEDIUM, shadow_resolution=1024),
        )
        self._set_resolution(
            QualityTier.HIGH,
            TierResolution(tier=QualityTier.HIGH, shadow_resolution=2048),
        )
        self._set_resolution(
            QualityTier.ULTRA,
            TierResolution(tier=QualityTier.ULTRA, shadow_resolution=4096),
        )

        # Budgets
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=0.5, memory_mb=8),
        )
        self._set_budget(
            QualityTier.MEDIUM,
            TierBudget(tier=QualityTier.MEDIUM, gpu_time_ms=1.0, memory_mb=32),
        )
        self._set_budget(
            QualityTier.HIGH,
            TierBudget(tier=QualityTier.HIGH, gpu_time_ms=2.0, memory_mb=64),
        )
        self._set_budget(
            QualityTier.ULTRA,
            TierBudget(tier=QualityTier.ULTRA, gpu_time_ms=3.0, memory_mb=128),
        )

        # Fallback chains
        self._set_fallback(
            "shadow_filtering",
            FallbackChain(
                primary="ray_traced",
                fallbacks=("pcss", "vsm", "pcf"),
            ),
        )
        self._set_fallback(
            "contact_shadows",
            FallbackChain(
                primary="screen_space_raymarched",
                fallbacks=("screen_space_stepped", "none"),
            ),
        )
