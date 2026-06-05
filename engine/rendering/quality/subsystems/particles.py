"""Quality tier configuration for Particles subsystem (S9)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class ParticlesCapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Particles subsystem.

    Configures particle count, simulation, and rendering quality per tier.
    """

    @property
    def subsystem_name(self) -> str:
        return "particles"

    def _init_tier_configs(self) -> None:
        # LOW tier: Simple billboards
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "billboard_particles",
                    "cpu_simulation",
                    "basic_lighting",
                }),
                disabled_features=frozenset({
                    "gpu_simulation",
                    "mesh_particles",
                    "trails",
                    "collision",
                    "soft_particles",
                    "distortion",
                }),
                parameters={
                    "max_particles": 1000,
                    "max_emitters": 16,
                    "update_rate": 30,  # Hz
                },
            ),
        )

        # MEDIUM tier: GPU simulation + soft particles
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "billboard_particles",
                    "gpu_simulation",
                    "soft_particles",
                    "basic_lighting",
                    "sorting",
                }),
                parameters={
                    "max_particles": 10000,
                    "max_emitters": 64,
                    "update_rate": 60,
                    "sort_frequency": 2,  # Every N frames
                },
            ),
        )

        # HIGH tier: + Trails, mesh particles, collision
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "billboard_particles",
                    "mesh_particles",
                    "gpu_simulation",
                    "soft_particles",
                    "trails",
                    "collision",
                    "pbr_lighting",
                    "sorting",
                    "vector_fields",
                }),
                parameters={
                    "max_particles": 100000,
                    "max_emitters": 256,
                    "max_trail_points": 64,
                    "collision_iterations": 2,
                    "sort_frequency": 1,
                },
            ),
        )

        # ULTRA tier: Full features + distortion
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "billboard_particles",
                    "mesh_particles",
                    "gpu_simulation",
                    "soft_particles",
                    "trails",
                    "collision",
                    "pbr_lighting",
                    "sorting",
                    "vector_fields",
                    "distortion",
                    "shadows",
                    "volumetric_particles",
                }),
                parameters={
                    "max_particles": 1000000,
                    "max_emitters": 1024,
                    "max_trail_points": 128,
                    "collision_iterations": 4,
                    "sort_frequency": 1,
                    "shadow_casting": True,
                },
            ),
        )

        # Budgets
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=0.5, memory_mb=16),
        )
        self._set_budget(
            QualityTier.MEDIUM,
            TierBudget(tier=QualityTier.MEDIUM, gpu_time_ms=1.5, memory_mb=64),
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
            "simulation",
            FallbackChain(
                primary="gpu_compute",
                fallbacks=("gpu_vertex", "cpu_threaded", "cpu_single"),
            ),
        )
        self._set_fallback(
            "rendering",
            FallbackChain(
                primary="mesh_instanced",
                fallbacks=("billboard_instanced", "billboard_batched", "billboard_simple"),
            ),
        )
