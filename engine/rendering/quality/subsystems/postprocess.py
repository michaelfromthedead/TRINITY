"""Quality tier configuration for Post-Processing subsystem (S8)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class PostProcessCapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Post-Processing subsystem.

    Configures tonemapping, bloom, DOF, motion blur, and AA per tier.
    """

    @property
    def subsystem_name(self) -> str:
        return "postprocess"

    def _init_tier_configs(self) -> None:
        # LOW tier: Basic tonemapping + bloom only
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "tonemapping",
                    "bloom",
                    "fxaa",
                }),
                disabled_features=frozenset({
                    "dof",
                    "motion_blur",
                    "taa",
                    "chromatic_aberration",
                    "film_grain",
                    "vignette",
                }),
                parameters={
                    "tonemap_operator": "reinhard",
                    "bloom_iterations": 3,
                    "bloom_threshold": 1.0,
                },
            ),
        )

        # MEDIUM tier: + DOF + TAA
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "tonemapping",
                    "bloom",
                    "dof",
                    "taa",
                    "vignette",
                }),
                parameters={
                    "tonemap_operator": "aces",
                    "bloom_iterations": 5,
                    "bloom_threshold": 0.8,
                    "dof_samples": 8,
                    "taa_jitter_samples": 8,
                },
            ),
        )

        # HIGH tier: Full post-process stack
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "tonemapping",
                    "bloom",
                    "dof",
                    "motion_blur",
                    "taa",
                    "chromatic_aberration",
                    "vignette",
                    "auto_exposure",
                    "color_grading",
                }),
                parameters={
                    "tonemap_operator": "aces_fitted",
                    "bloom_iterations": 6,
                    "bloom_threshold": 0.6,
                    "dof_samples": 16,
                    "motion_blur_samples": 8,
                    "taa_jitter_samples": 16,
                },
            ),
        )

        # ULTRA tier: + upscaling + advanced effects
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "tonemapping",
                    "bloom",
                    "dof",
                    "motion_blur",
                    "taa",
                    "chromatic_aberration",
                    "film_grain",
                    "vignette",
                    "auto_exposure",
                    "color_grading",
                    "lens_flare",
                    "temporal_upscaling",  # FSR/DLSS style
                    "bokeh_dof",
                }),
                parameters={
                    "tonemap_operator": "aces_fitted",
                    "bloom_iterations": 8,
                    "bloom_threshold": 0.5,
                    "dof_samples": 32,
                    "bokeh_shape": "circular",
                    "motion_blur_samples": 16,
                    "taa_jitter_samples": 32,
                    "upscaling_method": "fsr2",
                },
            ),
        )

        # Resolutions (for intermediate buffers)
        self._set_resolution(
            QualityTier.LOW,
            TierResolution(tier=QualityTier.LOW, render_scale=0.75),
        )
        self._set_resolution(
            QualityTier.MEDIUM,
            TierResolution(tier=QualityTier.MEDIUM, render_scale=1.0),
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
            "anti_aliasing",
            FallbackChain(
                primary="taa",
                fallbacks=("smaa", "fxaa", "none"),
            ),
        )
        self._set_fallback(
            "depth_of_field",
            FallbackChain(
                primary="bokeh_dof",
                fallbacks=("hex_dof", "gaussian_dof", "none"),
            ),
        )
        self._set_fallback(
            "upscaling",
            FallbackChain(
                primary="fsr2",
                fallbacks=("fsr1", "bilinear", "none"),
            ),
        )
