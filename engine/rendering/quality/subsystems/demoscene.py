"""Quality tier configuration for Demoscene subsystem (S13)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class DemosceneCapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Demoscene/SDF rendering subsystem.

    Configures ray marching quality, SDF complexity, and effects per tier.
    """

    @property
    def subsystem_name(self) -> str:
        return "demoscene"

    def _init_tier_configs(self) -> None:
        # LOW tier: Basic ray marching
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "sphere_tracing",
                    "basic_sdf",
                    "hard_shadows",
                    "simple_ao",
                }),
                disabled_features=frozenset({
                    "soft_shadows",
                    "advanced_ao",
                    "subsurface",
                    "reflections",
                    "dof",
                    "motion_blur",
                }),
                parameters={
                    "max_steps": 64,
                    "max_distance": 100,
                    "epsilon": 0.01,
                    "ao_steps": 3,
                },
            ),
        )

        # MEDIUM tier: + Soft shadows, better AO
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "sphere_tracing",
                    "basic_sdf",
                    "domain_operations",
                    "soft_shadows",
                    "quilez_ao",
                    "blinn_phong",
                }),
                parameters={
                    "max_steps": 128,
                    "max_distance": 200,
                    "epsilon": 0.005,
                    "ao_steps": 5,
                    "shadow_softness": 8.0,
                },
            ),
        )

        # HIGH tier: + Reflections, GGX, DOF
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "sphere_tracing",
                    "basic_sdf",
                    "domain_operations",
                    "noise_functions",
                    "soft_shadows",
                    "quilez_ao",
                    "ggx_brdf",
                    "reflections",
                    "dof",
                    "tonemapping",
                    "temporal_aa",
                }),
                parameters={
                    "max_steps": 256,
                    "max_distance": 500,
                    "epsilon": 0.001,
                    "ao_steps": 8,
                    "shadow_softness": 16.0,
                    "reflection_bounces": 1,
                    "dof_samples": 16,
                },
            ),
        )

        # ULTRA tier: Full demoscene features
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "sphere_tracing",
                    "enhanced_sphere_tracing",
                    "basic_sdf",
                    "domain_operations",
                    "noise_functions",
                    "fractals",
                    "soft_shadows",
                    "quilez_ao",
                    "ggx_brdf",
                    "subsurface",
                    "reflections",
                    "refractions",
                    "dof",
                    "motion_blur",
                    "chromatic_aberration",
                    "bloom",
                    "tonemapping",
                    "temporal_aa",
                    "volumetric_effects",
                }),
                parameters={
                    "max_steps": 512,
                    "max_distance": 1000,
                    "epsilon": 0.0001,
                    "ao_steps": 16,
                    "shadow_softness": 32.0,
                    "reflection_bounces": 3,
                    "refraction_bounces": 2,
                    "dof_samples": 64,
                    "fractal_iterations": 12,
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

        # Budgets (demoscene is GPU intensive)
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=4.0, memory_mb=32),
        )
        self._set_budget(
            QualityTier.MEDIUM,
            TierBudget(tier=QualityTier.MEDIUM, gpu_time_ms=8.0, memory_mb=64),
        )
        self._set_budget(
            QualityTier.HIGH,
            TierBudget(tier=QualityTier.HIGH, gpu_time_ms=12.0, memory_mb=128),
        )
        self._set_budget(
            QualityTier.ULTRA,
            TierBudget(tier=QualityTier.ULTRA, gpu_time_ms=16.0, memory_mb=256),
        )

        # Fallback chains
        self._set_fallback(
            "ray_marching",
            FallbackChain(
                primary="enhanced_sphere_tracing",
                fallbacks=("sphere_tracing", "relaxed_sphere_tracing", "fixed_step"),
            ),
        )
        self._set_fallback(
            "lighting",
            FallbackChain(
                primary="ggx_brdf",
                fallbacks=("blinn_phong", "lambert", "unlit"),
            ),
        )
