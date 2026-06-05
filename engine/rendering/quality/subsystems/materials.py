"""Quality tier configuration for Materials subsystem (S3)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class MaterialsCapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Materials subsystem.

    Configures PBR features, texture quality, and shader variants per tier.
    """

    @property
    def subsystem_name(self) -> str:
        return "materials"

    def _init_tier_configs(self) -> None:
        # LOW tier: Basic PBR only
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "base_color",
                    "normal_mapping",
                    "roughness_metallic",
                }),
                disabled_features=frozenset({
                    "subsurface_scattering",
                    "clear_coat",
                    "anisotropy",
                    "sheen",
                    "transmission",
                    "iridescence",
                    "parallax_mapping",
                }),
                parameters={
                    "max_variants": 1,
                    "texture_lod_bias": 2.0,
                    "trilinear_filtering": False,
                    "anisotropic_filtering": 1,
                },
            ),
        )

        # MEDIUM tier: Standard PBR with some advanced features
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "base_color",
                    "normal_mapping",
                    "roughness_metallic",
                    "ambient_occlusion",
                    "emissive",
                    "detail_maps",
                }),
                parameters={
                    "max_variants": 3,
                    "texture_lod_bias": 1.0,
                    "trilinear_filtering": True,
                    "anisotropic_filtering": 4,
                },
            ),
        )

        # HIGH tier: Full PBR with advanced shading
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "base_color",
                    "normal_mapping",
                    "roughness_metallic",
                    "ambient_occlusion",
                    "emissive",
                    "detail_maps",
                    "subsurface_scattering",
                    "clear_coat",
                    "anisotropy",
                    "parallax_mapping",
                }),
                parameters={
                    "max_variants": 10,
                    "texture_lod_bias": 0.0,
                    "trilinear_filtering": True,
                    "anisotropic_filtering": 8,
                    "sss_samples": 8,
                },
            ),
        )

        # ULTRA tier: All features enabled
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "base_color",
                    "normal_mapping",
                    "roughness_metallic",
                    "ambient_occlusion",
                    "emissive",
                    "detail_maps",
                    "subsurface_scattering",
                    "clear_coat",
                    "anisotropy",
                    "sheen",
                    "transmission",
                    "iridescence",
                    "parallax_mapping",
                    "tessellation",
                    "displacement_mapping",
                }),
                parameters={
                    "max_variants": -1,  # Unlimited
                    "texture_lod_bias": -0.5,  # Sharper textures
                    "trilinear_filtering": True,
                    "anisotropic_filtering": 16,
                    "sss_samples": 16,
                    "parallax_steps": 32,
                },
            ),
        )

        # Budgets
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=0.5, memory_mb=64),
        )
        self._set_budget(
            QualityTier.MEDIUM,
            TierBudget(tier=QualityTier.MEDIUM, gpu_time_ms=1.0, memory_mb=128),
        )
        self._set_budget(
            QualityTier.HIGH,
            TierBudget(tier=QualityTier.HIGH, gpu_time_ms=2.0, memory_mb=256),
        )
        self._set_budget(
            QualityTier.ULTRA,
            TierBudget(tier=QualityTier.ULTRA, gpu_time_ms=4.0, memory_mb=512),
        )

        # Resolutions
        self._set_resolution(
            QualityTier.LOW,
            TierResolution(tier=QualityTier.LOW, texture_max_size=512),
        )
        self._set_resolution(
            QualityTier.MEDIUM,
            TierResolution(tier=QualityTier.MEDIUM, texture_max_size=1024),
        )
        self._set_resolution(
            QualityTier.HIGH,
            TierResolution(tier=QualityTier.HIGH, texture_max_size=2048),
        )
        self._set_resolution(
            QualityTier.ULTRA,
            TierResolution(tier=QualityTier.ULTRA, texture_max_size=4096),
        )

        # Fallback chains
        self._set_fallback(
            "subsurface",
            FallbackChain(
                primary="separable_sss",
                fallbacks=("simple_sss", "diffuse_wrap", "none"),
            ),
        )
        self._set_fallback(
            "parallax",
            FallbackChain(
                primary="pom",  # Parallax occlusion mapping
                fallbacks=("steep_parallax", "simple_parallax", "none"),
            ),
        )
