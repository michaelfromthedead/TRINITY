"""Quality tier configuration for Water/Terrain subsystem (S12)."""

from trinity.types import QualityTier

from ..capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)


class TerrainCapabilities(BaseQualityCapabilities):
    """
    Quality capabilities for the Water/Terrain subsystem.

    Configures terrain LOD, water simulation, and vegetation per tier.
    """

    @property
    def subsystem_name(self) -> str:
        return "terrain"

    def _init_tier_configs(self) -> None:
        # LOW tier: Basic heightmap terrain
        self._set_features(
            QualityTier.LOW,
            TierFeatureSet(
                tier=QualityTier.LOW,
                enabled_features=frozenset({
                    "heightmap_terrain",
                    "simple_water",
                    "distance_fog",
                }),
                disabled_features=frozenset({
                    "clipmap_terrain",
                    "virtual_texturing",
                    "tessellation",
                    "gerstner_waves",
                    "underwater",
                    "foliage",
                }),
                parameters={
                    "terrain_lod_levels": 2,
                    "terrain_tile_size": 64,
                    "water_resolution": 128,
                    "view_distance": 500,
                },
            ),
        )

        # MEDIUM tier: Clipmap terrain + Gerstner water
        self._set_features(
            QualityTier.MEDIUM,
            TierFeatureSet(
                tier=QualityTier.MEDIUM,
                enabled_features=frozenset({
                    "heightmap_terrain",
                    "clipmap_terrain",
                    "gerstner_waves",
                    "water_reflections",
                    "basic_foliage",
                }),
                parameters={
                    "terrain_lod_levels": 4,
                    "terrain_tile_size": 128,
                    "water_resolution": 256,
                    "gerstner_waves": 4,
                    "view_distance": 1000,
                    "foliage_density": 0.3,
                },
            ),
        )

        # HIGH tier: + Tessellation, FFT ocean, dense foliage
        self._set_features(
            QualityTier.HIGH,
            TierFeatureSet(
                tier=QualityTier.HIGH,
                enabled_features=frozenset({
                    "heightmap_terrain",
                    "clipmap_terrain",
                    "tessellation",
                    "gerstner_waves",
                    "fft_ocean",
                    "water_reflections",
                    "water_refraction",
                    "foam",
                    "foliage",
                    "foliage_wind",
                    "virtual_texturing",
                }),
                parameters={
                    "terrain_lod_levels": 6,
                    "terrain_tile_size": 256,
                    "tessellation_factor": 16,
                    "water_resolution": 512,
                    "fft_resolution": 256,
                    "gerstner_waves": 8,
                    "view_distance": 2000,
                    "foliage_density": 0.6,
                    "foliage_draw_distance": 200,
                },
            ),
        )

        # ULTRA tier: Full features
        self._set_features(
            QualityTier.ULTRA,
            TierFeatureSet(
                tier=QualityTier.ULTRA,
                enabled_features=frozenset({
                    "heightmap_terrain",
                    "clipmap_terrain",
                    "tessellation",
                    "gerstner_waves",
                    "fft_ocean",
                    "water_reflections",
                    "water_refraction",
                    "foam",
                    "underwater",
                    "caustics",
                    "foliage",
                    "foliage_wind",
                    "foliage_interaction",
                    "virtual_texturing",
                    "procedural_detail",
                }),
                parameters={
                    "terrain_lod_levels": 8,
                    "terrain_tile_size": 512,
                    "tessellation_factor": 64,
                    "water_resolution": 1024,
                    "fft_resolution": 512,
                    "fft_cascades": 4,
                    "gerstner_waves": 16,
                    "view_distance": 5000,
                    "foliage_density": 1.0,
                    "foliage_draw_distance": 500,
                },
            ),
        )

        # Budgets
        self._set_budget(
            QualityTier.LOW,
            TierBudget(tier=QualityTier.LOW, gpu_time_ms=1.0, memory_mb=64),
        )
        self._set_budget(
            QualityTier.MEDIUM,
            TierBudget(tier=QualityTier.MEDIUM, gpu_time_ms=2.5, memory_mb=128),
        )
        self._set_budget(
            QualityTier.HIGH,
            TierBudget(tier=QualityTier.HIGH, gpu_time_ms=4.0, memory_mb=256),
        )
        self._set_budget(
            QualityTier.ULTRA,
            TierBudget(tier=QualityTier.ULTRA, gpu_time_ms=6.0, memory_mb=512),
        )

        # Fallback chains
        self._set_fallback(
            "terrain",
            FallbackChain(
                primary="clipmap_tessellated",
                fallbacks=("clipmap_lod", "chunked_lod", "simple_grid"),
            ),
        )
        self._set_fallback(
            "water",
            FallbackChain(
                primary="fft_ocean",
                fallbacks=("gerstner_sum", "sine_waves", "flat_plane"),
            ),
        )
        self._set_fallback(
            "foliage",
            FallbackChain(
                primary="gpu_instanced",
                fallbacks=("billboard_impostors", "texture_cards", "none"),
            ),
        )
