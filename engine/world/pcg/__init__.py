"""
PCG (Procedural Content Generation) System.

This module provides comprehensive procedural generation utilities for the
World Layer, including:

- Noise Generation: Perlin, Simplex, Worley, Value, and Fractal noise
- Scatter Placement: Random, Poisson Disk, Grid, Clustered, and Organic patterns
- Placement Rules: Slope, Height, Layer, Noise, and Exclusion zone filters
- Seed Management: Deterministic seed generation for reproducible worlds

All generators are fully deterministic given the same seed, ensuring
consistent world generation across sessions and platforms.

Trinity Pattern Integration:
- @seeded: For deterministic generation based on world/chunk/entity seeds
- @procedural: For cached procedural generation with validation
- @constraint: For generation rules and filters

Example usage:
    from engine.world.pcg import (
        PerlinNoise,
        PoissonDiskScatter,
        PlacementRuleSet,
        ChunkSeed,
        RandomStream,
    )

    # Create deterministic noise
    noise = PerlinNoise(seed=12345)
    value = noise.sample(10.5, 20.3)

    # Generate scattered points
    scatter = PoissonDiskScatter(settings=ScatterSettings(min_spacing=5.0))
    points = scatter.generate(Bounds(0, 0, 100, 100))

    # Use chunk-based seeds
    chunk_seed = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
    random = RandomStream(chunk_seed.get_seed())
"""

from engine.world.pcg.noise import (
    NoiseType,
    NoiseSettings,
    NoiseGenerator,
    PerlinNoise,
    SimplexNoise,
    WorleyNoise,
    ValueNoise,
    WhiteNoise,
    FractalNoise,
    NoiseMap,
    create_noise_generator,
)

from engine.world.pcg.scatter import (
    ScatterPattern,
    ScatterSettings,
    ScatterPoint,
    Bounds,
    DeterministicRandom as ScatterRandom,
    ScatterGenerator,
    RandomScatter,
    PoissonDiskScatter,
    GridScatter,
    JitteredGridScatter,
    ClusteredScatter,
    OrganicScatter,
    ScatterSystem,
)

from engine.world.pcg.rules import (
    TerrainData,
    SlopeFilter,
    HeightFilter,
    LayerFilter,
    NoiseFilter,
    ExclusionZone,
    PlacementFilter,
    SlopeFilterImpl,
    HeightFilterImpl,
    LayerFilterImpl,
    NoiseFilterImpl,
    ExclusionZoneFilter,
    CompoundFilter,
    BiomeRule,
    PlacementRuleSet,
    TransformRule,
    Transform,
    PlacementValidator,
    create_slope_filter,
    create_height_filter,
    create_layer_filter,
    create_noise_filter,
    create_exclusion_filter,
)

from engine.world.pcg.seeds import (
    SeedConfig,
    SeedGenerator,
    ChunkSeed,
    LayerSeed,
    InstanceSeed,
    RandomStream,
    DeterministicRandom,
    combine_seeds,
    position_to_seed,
    string_to_seed,
)

# Constants module for magic numbers and defaults
from engine.world.pcg import constants

__all__ = [
    # Noise
    "NoiseType",
    "NoiseSettings",
    "NoiseGenerator",
    "PerlinNoise",
    "SimplexNoise",
    "WorleyNoise",
    "ValueNoise",
    "WhiteNoise",
    "FractalNoise",
    "NoiseMap",
    "create_noise_generator",
    # Scatter
    "ScatterPattern",
    "ScatterSettings",
    "ScatterPoint",
    "Bounds",
    "ScatterRandom",
    "ScatterGenerator",
    "RandomScatter",
    "PoissonDiskScatter",
    "GridScatter",
    "JitteredGridScatter",
    "ClusteredScatter",
    "OrganicScatter",
    "ScatterSystem",
    # Rules
    "TerrainData",
    "SlopeFilter",
    "HeightFilter",
    "LayerFilter",
    "NoiseFilter",
    "ExclusionZone",
    "PlacementFilter",
    "SlopeFilterImpl",
    "HeightFilterImpl",
    "LayerFilterImpl",
    "NoiseFilterImpl",
    "ExclusionZoneFilter",
    "CompoundFilter",
    "BiomeRule",
    "PlacementRuleSet",
    "TransformRule",
    "Transform",
    "PlacementValidator",
    "create_slope_filter",
    "create_height_filter",
    "create_layer_filter",
    "create_noise_filter",
    "create_exclusion_filter",
    # Seeds
    "SeedConfig",
    "SeedGenerator",
    "ChunkSeed",
    "LayerSeed",
    "InstanceSeed",
    "RandomStream",
    "DeterministicRandom",
    "combine_seeds",
    "position_to_seed",
    "string_to_seed",
    # Constants module
    "constants",
]
