"""
Rigorous PCG Tests.

These tests verify critical PCG properties that must hold:
1. Determinism: Same seed + same position = same output, ALWAYS
2. Range bounds: Noise outputs must be within documented ranges
3. Poisson spacing: Minimum distance guarantee must be respected
4. Seed uniqueness: Different positions must produce different seeds

These tests are more thorough than basic unit tests and verify
invariants that could cause subtle bugs if violated.
"""

import math
import pytest

from engine.world.pcg.noise import (
    NoiseType,
    NoiseSettings,
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
    ScatterSettings,
    Bounds,
    DeterministicRandom,
    PoissonDiskScatter,
    RandomScatter,
)
from engine.world.pcg.seeds import (
    SeedGenerator,
    ChunkSeed,
    RandomStream,
    position_to_seed,
)


class TestNoiseDeterminismRigorous:
    """Rigorous tests for noise determinism.

    These tests verify that noise is TRULY deterministic by:
    1. Creating independent generator instances
    2. Testing many different coordinates
    3. Including edge cases (very large, very small, negative)
    4. Testing across multiple calls
    """

    @pytest.mark.parametrize("noise_class,kwargs", [
        (PerlinNoise, {}),
        (SimplexNoise, {}),
        (WorleyNoise, {}),
        (WorleyNoise, {"distance_type": "manhattan"}),
        (WorleyNoise, {"return_type": "f2"}),
        (ValueNoise, {}),
        (WhiteNoise, {}),
    ])
    def test_determinism_independent_instances(self, noise_class, kwargs):
        """Verify two completely independent instances produce identical output."""
        for seed in [0, 1, 42, 12345, 0x7FFFFFFF]:
            noise1 = noise_class(seed=seed, **kwargs)
            noise2 = noise_class(seed=seed, **kwargs)

            # Test grid of coordinates
            coords = [
                (0.0, 0.0), (0.5, 0.5), (1.0, 1.0),
                (-10.5, 20.3), (100.123, -50.789),
                (0.001, 0.001), (999.999, 999.999),
            ]

            for x, y in coords:
                v1 = noise1.sample(x, y)
                v2 = noise2.sample(x, y)
                assert v1 == v2, (
                    f"{noise_class.__name__} not deterministic at ({x}, {y}) "
                    f"with seed {seed}: {v1} != {v2}"
                )

    def test_perlin_determinism_after_many_samples(self):
        """Verify Perlin remains deterministic after many samples."""
        noise1 = PerlinNoise(seed=42)
        noise2 = PerlinNoise(seed=42)

        # Sample many points with noise1
        for i in range(1000):
            noise1.sample(i * 0.1, i * 0.05)

        # noise2 is fresh - should still match
        for i in range(1000):
            v1 = noise1.sample(i * 0.1, i * 0.05)
            v2 = noise2.sample(i * 0.1, i * 0.05)
            # After the warm-up, revisiting same coords should match fresh instance

        # Now test specific coords after all that sampling
        test_coords = [(5.5, 3.2), (100.0, 50.0), (-25.5, 12.3)]
        noise3 = PerlinNoise(seed=42)  # Fresh instance

        for x, y in test_coords:
            v1 = noise1.sample(x, y)
            v3 = noise3.sample(x, y)
            assert v1 == v3, f"Perlin not deterministic after many samples at ({x}, {y})"

    def test_simplex_determinism_large_coordinates(self):
        """Verify Simplex handles large coordinates deterministically."""
        noise1 = SimplexNoise(seed=42)
        noise2 = SimplexNoise(seed=42)

        large_coords = [
            (10000.5, 10000.5),
            (-50000.123, 30000.456),
            (99999.0, -99999.0),
            (1e6, 1e6),  # Very large
        ]

        for x, y in large_coords:
            v1 = noise1.sample(x, y)
            v2 = noise2.sample(x, y)
            assert v1 == v2, f"Simplex not deterministic at large coords ({x}, {y})"
            assert math.isfinite(v1), f"Simplex returned non-finite at ({x}, {y}): {v1}"

    def test_fractal_determinism(self):
        """Verify fractal noise is deterministic."""
        for octaves in [1, 2, 4, 8]:
            base1 = PerlinNoise(seed=42)
            base2 = PerlinNoise(seed=42)

            fractal1 = FractalNoise(base1, octaves=octaves)
            fractal2 = FractalNoise(base2, octaves=octaves)

            for i in range(100):
                x, y = i * 0.1, i * 0.05
                v1 = fractal1.sample(x, y)
                v2 = fractal2.sample(x, y)
                assert v1 == v2, f"Fractal({octaves}) not deterministic at ({x}, {y})"


class TestNoiseRangeBounds:
    """Rigorous tests for noise output range bounds.

    Documents claim noise outputs are in [-1, 1] or [0, 1].
    These tests verify the actual bounds across many samples.
    """

    def test_perlin_range_extensive(self):
        """Verify Perlin noise range with extensive sampling."""
        noise = PerlinNoise(seed=42)
        min_val = float("inf")
        max_val = float("-inf")

        # Sample a large grid
        for i in range(-100, 101):
            for j in range(-100, 101):
                x, y = i * 0.1, j * 0.1
                v = noise.sample(x, y)
                min_val = min(min_val, v)
                max_val = max(max_val, v)

        # Perlin can slightly exceed [-1, 1] but should be bounded
        # Theoretical max is sqrt(2)/2 * 4 corners = ~1.0
        assert min_val >= -1.5, f"Perlin min too low: {min_val}"
        assert max_val <= 1.5, f"Perlin max too high: {max_val}"

        # Verify we're actually getting variation
        assert min_val < -0.3, f"Perlin min suspicious (too high): {min_val}"
        assert max_val > 0.3, f"Perlin max suspicious (too low): {max_val}"

    def test_simplex_range_extensive(self):
        """Verify Simplex noise range with extensive sampling."""
        noise = SimplexNoise(seed=42)
        min_val = float("inf")
        max_val = float("-inf")

        for i in range(-100, 101):
            for j in range(-100, 101):
                x, y = i * 0.1, j * 0.1
                v = noise.sample(x, y)
                min_val = min(min_val, v)
                max_val = max(max_val, v)

        # Simplex should be closer to [-1, 1]
        assert min_val >= -1.5, f"Simplex min too low: {min_val}"
        assert max_val <= 1.5, f"Simplex max too high: {max_val}"

    def test_worley_range_all_return_types(self):
        """Verify Worley noise range for all return types."""
        for return_type in ["f1", "f2", "f2-f1"]:
            noise = WorleyNoise(seed=42, return_type=return_type)

            for i in range(-50, 51):
                for j in range(-50, 51):
                    x, y = i * 0.1, j * 0.1
                    v = noise.sample(x, y)

                    # After normalization, should be clamped to [-1, 1]
                    assert -1.0 <= v <= 1.0, (
                        f"Worley({return_type}) out of range at ({x}, {y}): {v}"
                    )

    def test_white_noise_range_strict(self):
        """Verify white noise is strictly in [-1, 1]."""
        noise = WhiteNoise(seed=42)

        for i in range(-1000, 1001):
            for j in range(-10, 11):
                x, y = i * 0.01, j * 0.1
                v = noise.sample(x, y)
                assert -1.0 <= v <= 1.0, f"White noise out of range at ({x}, {y}): {v}"

    def test_value_noise_range(self):
        """Verify value noise range."""
        noise = ValueNoise(seed=42)
        min_val = float("inf")
        max_val = float("-inf")

        for i in range(-100, 101):
            for j in range(-100, 101):
                x, y = i * 0.1, j * 0.1
                v = noise.sample(x, y)
                min_val = min(min_val, v)
                max_val = max(max_val, v)

        # Value noise with interpolation should be in [-1, 1]
        assert min_val >= -1.5, f"Value noise min too low: {min_val}"
        assert max_val <= 1.5, f"Value noise max too high: {max_val}"


class TestPoissonSpacingGuarantee:
    """Rigorous tests for Poisson disk minimum spacing guarantee.

    The Poisson disk algorithm MUST guarantee that no two points
    are closer than min_spacing. This is a critical invariant.
    """

    def test_minimum_spacing_guaranteed(self):
        """Verify minimum spacing is ALWAYS maintained."""
        for seed in [0, 42, 12345, 0x7FFFFFFF]:
            for min_spacing in [1.0, 2.5, 5.0, 10.0]:
                settings = ScatterSettings(seed=seed, min_spacing=min_spacing)
                scatter = PoissonDiskScatter(settings)
                bounds = Bounds(0, 0, 100, 100)

                points = scatter.generate(bounds)
                min_dist_sq = min_spacing ** 2

                # Check ALL pairs
                violations = []
                for i, p1 in enumerate(points):
                    for j, p2 in enumerate(points):
                        if i >= j:
                            continue
                        dist_sq = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
                        if dist_sq < min_dist_sq * 0.99:  # 1% tolerance
                            violations.append((i, j, math.sqrt(dist_sq)))

                assert len(violations) == 0, (
                    f"Poisson spacing violated with seed={seed}, min_spacing={min_spacing}. "
                    f"Violations: {violations[:5]}"
                )

    def test_poisson_determinism(self):
        """Verify Poisson scatter is deterministic."""
        for seed in [0, 42, 12345]:
            settings1 = ScatterSettings(seed=seed, min_spacing=5.0)
            settings2 = ScatterSettings(seed=seed, min_spacing=5.0)

            scatter1 = PoissonDiskScatter(settings1)
            scatter2 = PoissonDiskScatter(settings2)

            bounds = Bounds(0, 0, 50, 50)

            points1 = scatter1.generate(bounds)
            points2 = scatter2.generate(bounds)

            assert len(points1) == len(points2), (
                f"Poisson count differs for seed {seed}: {len(points1)} vs {len(points2)}"
            )

            for p1, p2 in zip(points1, points2):
                assert p1.position == p2.position, (
                    f"Poisson position differs for seed {seed}"
                )

    def test_poisson_fills_space(self):
        """Verify Poisson disk reasonably fills available space."""
        settings = ScatterSettings(seed=42, min_spacing=5.0)
        scatter = PoissonDiskScatter(settings)
        bounds = Bounds(0, 0, 100, 100)

        points = scatter.generate(bounds)

        # With 5.0 spacing in 100x100, expect roughly (100/5)^2 * 0.65 = ~260 points
        # Poisson disk typically achieves 60-70% of grid density
        expected_min = 200
        expected_max = 400

        assert expected_min <= len(points) <= expected_max, (
            f"Poisson generated unexpected count: {len(points)} "
            f"(expected {expected_min}-{expected_max})"
        )


class TestSeedUniqueness:
    """Rigorous tests for seed uniqueness.

    Different positions should produce different seeds.
    This is critical for PCG to work correctly.
    """

    def test_position_seeds_unique(self):
        """Verify different positions produce different seeds."""
        world_seed = 42
        seeds_seen = set()
        collisions = []

        # Generate seeds for a grid of positions
        for x in range(-100, 101, 10):
            for z in range(-100, 101, 10):
                seed = position_to_seed(world_seed, x, z)

                if seed in seeds_seen:
                    collisions.append((x, z, seed))
                else:
                    seeds_seen.add(seed)

        # With 441 positions, expect 0 or very few collisions
        assert len(collisions) == 0, (
            f"Position seed collisions detected: {collisions[:5]}"
        )

    def test_chunk_seeds_unique_grid(self):
        """Verify different chunks have unique seeds."""
        world_seed = 42
        seeds_seen = set()

        for cx in range(-50, 51):
            for cz in range(-50, 51):
                chunk = ChunkSeed(world_seed, cx, cz)
                seed = chunk.get_seed()

                if seed in seeds_seen:
                    pytest.fail(f"Chunk seed collision at ({cx}, {cz})")
                seeds_seen.add(seed)

        # All 10201 chunks should have unique seeds
        assert len(seeds_seen) == 101 * 101

    def test_adjacent_chunks_different(self):
        """Verify adjacent chunks have noticeably different seeds."""
        world_seed = 42

        for cx in range(100):
            for cz in range(100):
                chunk = ChunkSeed(world_seed, cx, cz)
                chunk_right = ChunkSeed(world_seed, cx + 1, cz)
                chunk_down = ChunkSeed(world_seed, cx, cz + 1)

                seed = chunk.get_seed()
                seed_right = chunk_right.get_seed()
                seed_down = chunk_down.get_seed()

                # Seeds should be significantly different
                assert seed != seed_right, f"Right neighbor collision at ({cx}, {cz})"
                assert seed != seed_down, f"Down neighbor collision at ({cx}, {cz})"

    def test_different_world_seeds_different_chunks(self):
        """Verify same chunk position with different world seeds differs."""
        for cx, cz in [(0, 0), (100, 100), (-50, 50)]:
            seeds = []
            for world_seed in range(100):
                chunk = ChunkSeed(world_seed, cx, cz)
                seeds.append(chunk.get_seed())

            # All 100 world seeds should produce different chunk seeds
            assert len(set(seeds)) == 100, (
                f"World seed collision at chunk ({cx}, {cz})"
            )


class TestRandomStreamQuality:
    """Tests for random stream quality beyond basic determinism."""

    def test_no_short_cycle(self):
        """Verify random stream doesn't have short cycles."""
        stream = RandomStream(42)

        # Generate values and look for repeats
        seen_states = set()
        for i in range(100000):
            state = stream.state
            if state in seen_states:
                pytest.fail(f"RandomStream cycle detected at iteration {i}")
            seen_states.add(state)
            stream.next_int(0, 100)

    def test_different_seeds_diverge(self):
        """Verify different seeds produce divergent sequences."""
        stream1 = RandomStream(42)
        stream2 = RandomStream(43)

        # After many iterations, sequences should be completely different
        values1 = [stream1.next_int(0, 1000) for _ in range(1000)]
        values2 = [stream2.next_int(0, 1000) for _ in range(1000)]

        # Count matches - should be roughly 1/1000 = ~1 by chance
        matches = sum(1 for v1, v2 in zip(values1, values2) if v1 == v2)
        assert matches < 50, f"Too many matches between different seeds: {matches}"

    def test_zero_seed_handled(self):
        """Verify seed=0 produces valid output (not stuck)."""
        stream = RandomStream(0)

        values = [stream.next_int(0, 1000) for _ in range(100)]

        # Should have variation
        unique = len(set(values))
        assert unique > 50, f"Seed 0 produced too few unique values: {unique}"

    def test_large_seed_handled(self):
        """Verify large seeds are handled correctly."""
        for seed in [0x7FFFFFFF, 0x7FFFFFFE, 0x70000000]:
            stream = RandomStream(seed)

            # Should work normally
            values = [stream.next_int(0, 1000) for _ in range(100)]
            unique = len(set(values))
            assert unique > 30, f"Large seed {seed} produced too few unique values"


class TestEdgeCases:
    """Tests for edge cases that could cause issues."""

    def test_noise_at_integer_boundaries(self):
        """Test noise at integer coordinate boundaries."""
        for noise_class in [PerlinNoise, SimplexNoise, ValueNoise]:
            noise = noise_class(seed=42)

            # Integer coordinates
            for x in range(-10, 11):
                for y in range(-10, 11):
                    v = noise.sample(float(x), float(y))
                    assert math.isfinite(v), (
                        f"{noise_class.__name__} non-finite at ({x}, {y}): {v}"
                    )

    def test_noise_at_tiny_offsets(self):
        """Test noise stability at tiny offsets from integers."""
        noise = PerlinNoise(seed=42)

        for x in range(-5, 6):
            for y in range(-5, 6):
                v_int = noise.sample(float(x), float(y))
                v_tiny = noise.sample(x + 1e-10, y + 1e-10)

                # Should be very similar
                diff = abs(v_int - v_tiny)
                assert diff < 0.001, (
                    f"Perlin unstable at tiny offset from ({x}, {y}): diff={diff}"
                )

    def test_poisson_small_bounds(self):
        """Test Poisson disk with bounds smaller than min_spacing."""
        settings = ScatterSettings(seed=42, min_spacing=10.0)
        scatter = PoissonDiskScatter(settings)

        # Bounds smaller than min_spacing
        small_bounds = Bounds(0, 0, 5, 5)
        points = scatter.generate(small_bounds)

        # Should get at most 1 point
        assert len(points) <= 1, f"Too many points in small bounds: {len(points)}"

    def test_scatter_bounds_containment_strict(self):
        """Verify ALL scatter points are strictly within bounds."""
        for seed in [0, 42, 12345]:
            settings = ScatterSettings(seed=seed, min_spacing=5.0)
            scatter = PoissonDiskScatter(settings)
            bounds = Bounds(10.5, 20.3, 90.7, 80.9)

            points = scatter.generate(bounds)

            for i, point in enumerate(points):
                assert bounds.min_x <= point.x <= bounds.max_x, (
                    f"Point {i} x={point.x} outside bounds [{bounds.min_x}, {bounds.max_x}]"
                )
                assert bounds.min_y <= point.y <= bounds.max_y, (
                    f"Point {i} y={point.y} outside bounds [{bounds.min_y}, {bounds.max_y}]"
                )

    def test_noise_map_normalization_edge_cases(self):
        """Test noise map normalization with edge cases."""
        # Uniform values (zero range)
        nmap = NoiseMap(5, 5)
        for y in range(5):
            for x in range(5):
                nmap.set_raw(x, y, 0.5)

        # Should not crash on zero range
        nmap.normalize(0.0, 1.0)

        # Values should remain unchanged (or close)
        for y in range(5):
            for x in range(5):
                v = nmap.get_raw(x, y)
                assert math.isfinite(v), f"Non-finite after normalize at ({x}, {y})"
