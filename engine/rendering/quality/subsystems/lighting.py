"""Quality tier configuration for Lighting subsystem (S4)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class LightingCapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Lighting subsystem.

    Configures light count, clustering, and advanced lighting features per tier.
    """

    @property
    def subsystem_name(self) -> str:
        return "lighting"

    def _init_tier_configs(self) -> None:
        # LOW tier: Forward lighting, limited lights
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "directional_light",
                    "point_lights",
                    "forward_shading",
                }),
                disabled_features=frozenset({
                    "clustered_lighting",
                    "deferred_shading",
                    "area_lights",
                    "ies_profiles",
                    "light_cookies",
                }),
                parameters={
                    "max_lights": 8,
                    "max_point_lights": 4,
                    "max_spot_lights": 4,
                    "light_culling": "per_object",
                },
            ),
        )

        # MEDIUM tier: Clustered forward
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "directional_light",
                    "point_lights",
                    "spot_lights",
                    "clustered_lighting",
                    "forward_shading",
                    "light_cookies",
                }),
                parameters={
                    "max_lights": 64,
                    "max_point_lights": 32,
                    "max_spot_lights": 32,
                    "cluster_size": (16, 8, 24),
                    "light_culling": "clustered",
                },
            ),
        )

        # HIGH tier: Full deferred with area lights
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "directional_light",
                    "point_lights",
                    "spot_lights",
                    "area_lights",
                    "clustered_lighting",
                    "deferred_shading",
                    "light_cookies",
                    "ies_profiles",
                    "ltc_area_lights",  # Linearly Transformed Cosines
                }),
                parameters={
                    "max_lights": 256,
                    "max_point_lights": 128,
                    "max_spot_lights": 64,
                    "max_area_lights": 32,
                    "cluster_size": (16, 8, 32),
                    "light_culling": "tiled_clustered",
                },
            ),
        )

        # ULTRA tier: Unlimited with volumetric
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "directional_light",
                    "point_lights",
                    "spot_lights",
                    "area_lights",
                    "clustered_lighting",
                    "deferred_shading",
                    "light_cookies",
                    "ies_profiles",
                    "ltc_area_lights",
                    "volumetric_lighting",
                    "light_probes",
                    "emissive_lighting",
                }),
                parameters={
                    "max_lights": -1,  # Unlimited
                    "max_point_lights": -1,
                    "max_spot_lights": -1,
                    "max_area_lights": -1,
                    "cluster_size": (16, 8, 48),
                    "light_culling": "gpu_driven",
                    "volumetric_samples": 64,
                },
            ),
        )

        # Budgets
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=1.0, memory_mb=32),
        )
        self._set_budget(
            QualityTier.MEDIUM,
            TierBudget(tier=QualityTier.MEDIUM, gpu_time_ms=2.0, memory_mb=64),
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
            "area_lights",
            FallbackChain(
                primary="ltc_polygonal",
                fallbacks=("ltc_sphere", "point_approximation", "none"),
            ),
        )
        self._set_fallback(
            "volumetric",
            FallbackChain(
                primary="raymarched_volumetric",
                fallbacks=("froxel_volumetric", "screen_space_fog", "none"),
            ),
        )
