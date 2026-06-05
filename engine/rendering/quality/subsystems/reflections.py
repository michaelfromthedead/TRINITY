"""Quality tier configuration for Reflections subsystem (S7)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class ReflectionsCapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Reflections subsystem.

    Configures screen-space, probe-based, and ray-traced reflections per tier.
    """

    @property
    def subsystem_name(self) -> str:
        return "reflections"

    def _init_tier_configs(self) -> None:
        # LOW tier: Environment map only
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "environment_map",
                    "box_projection",
                }),
                disabled_features=frozenset({
                    "ssr",
                    "planar_reflections",
                    "reflection_probes",
                    "rt_reflections",
                }),
                parameters={
                    "env_map_resolution": 128,
                    "roughness_mips": 4,
                },
            ),
        )

        # MEDIUM tier: + Reflection probes
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "environment_map",
                    "box_projection",
                    "reflection_probes",
                    "probe_blending",
                }),
                parameters={
                    "env_map_resolution": 256,
                    "roughness_mips": 6,
                    "max_probes": 16,
                    "probe_update_rate": 1,  # Per frame
                },
            ),
        )

        # HIGH tier: + SSR
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "environment_map",
                    "box_projection",
                    "reflection_probes",
                    "probe_blending",
                    "ssr",
                    "ssr_hiz",  # Hierarchical Z tracing
                    "planar_reflections",
                }),
                parameters={
                    "env_map_resolution": 512,
                    "roughness_mips": 8,
                    "max_probes": 32,
                    "probe_update_rate": 2,
                    "ssr_steps": 32,
                    "ssr_binary_steps": 8,
                    "ssr_thickness": 0.1,
                    "planar_resolution": 512,
                },
            ),
        )

        # ULTRA tier: + RT reflections
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "environment_map",
                    "box_projection",
                    "reflection_probes",
                    "probe_blending",
                    "ssr",
                    "ssr_hiz",
                    "planar_reflections",
                    "rt_reflections",
                    "rt_glossy",
                    "denoise_reflections",
                }),
                parameters={
                    "env_map_resolution": 1024,
                    "roughness_mips": 10,
                    "max_probes": 64,
                    "probe_update_rate": 4,
                    "ssr_steps": 64,
                    "ssr_binary_steps": 16,
                    "planar_resolution": 1024,
                    "rt_samples": 2,
                    "rt_bounces": 1,
                },
            ),
        )

        # Resolutions
        self._set_resolution(
            QualityTier.LOW,
            TierResolution(tier=QualityTier.LOW, reflection_resolution=128),
        )
        self._set_resolution(
            QualityTier.MEDIUM,
            TierResolution(tier=QualityTier.MEDIUM, reflection_resolution=256),
        )
        self._set_resolution(
            QualityTier.HIGH,
            TierResolution(tier=QualityTier.HIGH, reflection_resolution=512),
        )
        self._set_resolution(
            QualityTier.ULTRA,
            TierResolution(tier=QualityTier.ULTRA, reflection_resolution=1024),
        )

        # Budgets
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=0.3, memory_mb=16),
        )
        self._set_budget(
            QualityTier.MEDIUM,
            TierBudget(tier=QualityTier.MEDIUM, gpu_time_ms=1.0, memory_mb=64),
        )
        self._set_budget(
            QualityTier.HIGH,
            TierBudget(tier=QualityTier.HIGH, gpu_time_ms=2.5, memory_mb=128),
        )
        self._set_budget(
            QualityTier.ULTRA,
            TierBudget(tier=QualityTier.ULTRA, gpu_time_ms=4.0, memory_mb=256),
        )

        # Fallback chains
        self._set_fallback(
            "reflections",
            FallbackChain(
                primary="rt_reflections",
                fallbacks=("ssr_hiz", "ssr_linear", "reflection_probes"),
            ),
        )
        self._set_fallback(
            "glossy",
            FallbackChain(
                primary="rt_glossy",
                fallbacks=("ssr_roughness", "probe_roughness", "env_mip"),
            ),
        )
