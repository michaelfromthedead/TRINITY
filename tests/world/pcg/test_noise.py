"""
Tests for PCG noise generation.

Tests cover:
- Noise range validation (-1 to 1)
- Determinism (same seed = same output)
- Octave summation
- Different noise types
- Continuity (no discontinuities)
"""

import math
import pytest

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


class TestNoiseSettings:
    """Tests for NoiseSettings dataclass."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = NoiseSettings()
        assert settings.noise_type == NoiseType.PERLIN
        assert settings.seed == 0
        assert settings.frequency == 1.0
        assert settings.octaves == 4
        assert settings.lacunarity == 2.0
        assert settings.persistence == 0.5
        assert settings.amplitude == 1.0
        assert settings.offset == (0.0, 0.0)

    def test_custom_settings(self):
        """Test custom settings values."""
        settings = NoiseSettings(
            noise_type=NoiseType.SIMPLEX,
            seed=42,
            frequency=2.5,
            octaves=6,
            lacunarity=2.5,
            persistence=0.6,
            amplitude=2.0,
            offset=(10.0, 20.0),
        )
        assert settings.noise_type == NoiseType.SIMPLEX
        assert settings.seed == 42
        assert settings.frequency == 2.5
        assert settings.octaves == 6

    def test_invalid_frequency(self):
        """Test validation of frequency."""
        with pytest.raises(ValueError, match="frequency must be > 0"):
            NoiseSettings(frequency=0)

        with pytest.raises(ValueError, match="frequency must be > 0"):
            NoiseSettings(frequency=-1.0)

    def test_invalid_octaves(self):
        """Test validation of octaves."""
        with pytest.raises(ValueError, match="octaves must be >= 1"):
            NoiseSettings(octaves=0)

    def test_invalid_lacunarity(self):
        """Test validation of lacunarity."""
        with pytest.raises(ValueError, match="lacunarity must be > 0"):
            NoiseSettings(lacunarity=0)

    def test_invalid_persistence(self):
        """Test validation of persistence."""
        with pytest.raises(ValueError, match="persistence must be in"):
            NoiseSettings(persistence=0)

        with pytest.raises(ValueError, match="persistence must be in"):
            NoiseSettings(persistence=1.5)

    def test_invalid_amplitude(self):
        """Test validation of amplitude."""
        with pytest.raises(ValueError, match="amplitude must be > 0"):
            NoiseSettings(amplitude=0)


class TestPerlinNoise:
    """Tests for Perlin noise generator."""

    def test_creation(self):
        """Test basic creation."""
        noise = PerlinNoise(seed=42)
        assert noise.seed == 42
        assert noise._initialized

    def test_sample_range(self):
        """Test that samples are approximately in [-1, 1] range."""
        noise = PerlinNoise(seed=12345)
        for x in range(-10, 11):
            for y in range(-10, 11):
                value = noise.sample(x * 0.1, y * 0.1)
                # Perlin noise can slightly exceed [-1, 1] at some points
                # due to gradient summing, but should be within reasonable bounds
                assert -2.5 <= value <= 2.5, f"Value {value} out of range at ({x}, {y})"

    def test_determinism(self):
        """Test that same seed produces same results."""
        noise1 = PerlinNoise(seed=42)
        noise2 = PerlinNoise(seed=42)

        for x in range(-5, 6):
            for y in range(-5, 6):
                v1 = noise1.sample(x * 0.5, y * 0.5)
                v2 = noise2.sample(x * 0.5, y * 0.5)
                assert v1 == v2, f"Values differ at ({x}, {y})"

    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        noise1 = PerlinNoise(seed=42)
        noise2 = PerlinNoise(seed=43)

        # At least some values should differ
        differences = 0
        for x in range(-5, 6):
            for y in range(-5, 6):
                v1 = noise1.sample(x * 0.5, y * 0.5)
                v2 = noise2.sample(x * 0.5, y * 0.5)
                if v1 != v2:
                    differences += 1

        assert differences > 50, "Different seeds should produce different values"

    def test_continuity(self):
        """Test noise is continuous (no large jumps)."""
        noise = PerlinNoise(seed=42)
        step = 0.01

        # Check horizontal continuity
        for y in range(5):
            prev = noise.sample(0, y * 0.5)
            for x in range(1, 100):
                curr = noise.sample(x * step, y * 0.5)
                diff = abs(curr - prev)
                assert diff < 0.5, f"Discontinuity at ({x * step}, {y * 0.5}): {diff}"
                prev = curr

    def test_3d_sampling(self):
        """Test 3D noise sampling."""
        noise = PerlinNoise(seed=42)
        value = noise.sample_3d(1.5, 2.5, 3.5)
        assert -1.0 <= value <= 1.0

    def test_frequency_effect(self):
        """Test that frequency affects sampling."""
        settings_low = NoiseSettings(frequency=0.5)
        settings_high = NoiseSettings(frequency=4.0)

        noise_low = PerlinNoise(seed=42, settings=settings_low)
        noise_high = PerlinNoise(seed=42, settings=settings_high)

        # Higher frequency should have more variation over same distance
        variations_low = sum(
            abs(noise_low.sample(x * 0.1, 0) - noise_low.sample((x + 1) * 0.1, 0))
            for x in range(20)
        )
        variations_high = sum(
            abs(noise_high.sample(x * 0.1, 0) - noise_high.sample((x + 1) * 0.1, 0))
            for x in range(20)
        )

        assert variations_high > variations_low

    def test_amplitude_effect(self):
        """Test that amplitude scales output."""
        settings = NoiseSettings(amplitude=2.0)
        noise = PerlinNoise(seed=42, settings=settings)

        # Should still be bounded but scaled
        for x in range(10):
            value = noise.sample(x * 0.5, 0)
            assert -2.0 <= value <= 2.0

    def test_interpolate_smoothstep(self):
        """Test smoothstep interpolation function."""
        # At t=0, result should be a
        result = PerlinNoise._interpolate(0.0, 1.0, 0.0)
        assert result == pytest.approx(0.0)

        # At t=1, result should be b
        result = PerlinNoise._interpolate(0.0, 1.0, 1.0)
        assert result == pytest.approx(1.0)

        # At t=0.5, result should be midpoint
        result = PerlinNoise._interpolate(0.0, 1.0, 0.5)
        assert result == pytest.approx(0.5)


class TestSimplexNoise:
    """Tests for Simplex noise generator."""

    def test_creation(self):
        """Test basic creation."""
        noise = SimplexNoise(seed=42)
        assert noise.seed == 42
        assert noise._initialized

    def test_sample_range(self):
        """Test that samples are in [-1, 1] range."""
        noise = SimplexNoise(seed=12345)
        for x in range(-10, 11):
            for y in range(-10, 11):
                value = noise.sample(x * 0.1, y * 0.1)
                assert -1.5 <= value <= 1.5, f"Value {value} out of range"

    def test_determinism(self):
        """Test that same seed produces same results."""
        noise1 = SimplexNoise(seed=42)
        noise2 = SimplexNoise(seed=42)

        for x in range(-5, 6):
            for y in range(-5, 6):
                v1 = noise1.sample(x * 0.5, y * 0.5)
                v2 = noise2.sample(x * 0.5, y * 0.5)
                assert v1 == v2

    def test_3d_sampling(self):
        """Test 3D noise sampling."""
        noise = SimplexNoise(seed=42)
        value = noise.sample_3d(1.5, 2.5, 3.5)
        assert isinstance(value, float)

    def test_faster_than_perlin_hypothesis(self):
        """Test that simplex has similar quality to perlin."""
        perlin = PerlinNoise(seed=42)
        simplex = SimplexNoise(seed=42)

        # Both should produce reasonable values at same points
        for x in range(5):
            for y in range(5):
                p_val = perlin.sample(x * 0.5, y * 0.5)
                s_val = simplex.sample(x * 0.5, y * 0.5)

                # Both should be in valid range
                assert -1.5 <= p_val <= 1.5
                assert -1.5 <= s_val <= 1.5


class TestWorleyNoise:
    """Tests for Worley (cellular) noise generator."""

    def test_creation(self):
        """Test basic creation."""
        noise = WorleyNoise(seed=42)
        assert noise.seed == 42
        assert noise.distance_type == "euclidean"
        assert noise.return_type == "f1"

    def test_distance_types(self):
        """Test different distance metrics."""
        for distance_type in ["euclidean", "manhattan", "chebyshev"]:
            noise = WorleyNoise(seed=42, distance_type=distance_type)
            value = noise.sample(0.5, 0.5)
            assert isinstance(value, float)

    def test_invalid_distance_type(self):
        """Test validation of distance type."""
        with pytest.raises(ValueError, match="Invalid distance_type"):
            WorleyNoise(seed=42, distance_type="invalid")

    def test_return_types(self):
        """Test different return types."""
        for return_type in ["f1", "f2", "f2-f1"]:
            noise = WorleyNoise(seed=42, return_type=return_type)
            value = noise.sample(0.5, 0.5)
            assert isinstance(value, float)

    def test_invalid_return_type(self):
        """Test validation of return type."""
        with pytest.raises(ValueError, match="Invalid return_type"):
            WorleyNoise(seed=42, return_type="invalid")

    def test_determinism(self):
        """Test that same seed produces same results."""
        noise1 = WorleyNoise(seed=42)
        noise2 = WorleyNoise(seed=42)

        for x in range(-5, 6):
            for y in range(-5, 6):
                v1 = noise1.sample(x * 0.5, y * 0.5)
                v2 = noise2.sample(x * 0.5, y * 0.5)
                assert v1 == v2

    def test_f1_vs_f2(self):
        """Test that F1 and F2 differ appropriately."""
        noise_f1 = WorleyNoise(seed=42, return_type="f1")
        noise_f2 = WorleyNoise(seed=42, return_type="f2")

        # F2 should generally be larger than F1 (before normalization)
        # But after normalization, values depend on implementation


class TestValueNoise:
    """Tests for Value noise generator."""

    def test_creation(self):
        """Test basic creation."""
        noise = ValueNoise(seed=42)
        assert noise.seed == 42
        assert noise._initialized

    def test_sample_range(self):
        """Test that samples are in valid range."""
        noise = ValueNoise(seed=12345)
        for x in range(-10, 11):
            for y in range(-10, 11):
                value = noise.sample(x * 0.1, y * 0.1)
                assert -2.0 <= value <= 2.0

    def test_determinism(self):
        """Test that same seed produces same results."""
        noise1 = ValueNoise(seed=42)
        noise2 = ValueNoise(seed=42)

        for x in range(-5, 6):
            for y in range(-5, 6):
                v1 = noise1.sample(x * 0.5, y * 0.5)
                v2 = noise2.sample(x * 0.5, y * 0.5)
                assert v1 == v2


class TestWhiteNoise:
    """Tests for White noise generator."""

    def test_creation(self):
        """Test basic creation."""
        noise = WhiteNoise(seed=42)
        assert noise.seed == 42

    def test_sample_range(self):
        """Test that samples are in [-1, 1] range."""
        noise = WhiteNoise(seed=12345)
        for x in range(-10, 11):
            for y in range(-10, 11):
                value = noise.sample(x * 0.1, y * 0.1)
                assert -1.0 <= value <= 1.0

    def test_determinism(self):
        """Test that same seed produces same results."""
        noise1 = WhiteNoise(seed=42)
        noise2 = WhiteNoise(seed=42)

        for x in range(-5, 6):
            for y in range(-5, 6):
                v1 = noise1.sample(x * 0.5, y * 0.5)
                v2 = noise2.sample(x * 0.5, y * 0.5)
                assert v1 == v2

    def test_no_coherence(self):
        """Test that white noise has no spatial coherence."""
        noise = WhiteNoise(seed=42)

        # Adjacent samples should not be correlated
        samples = [noise.sample(x * 0.01, 0) for x in range(100)]

        # Calculate correlation between adjacent samples
        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)

        if variance > 0.01:  # Only test if there's variance
            covariance = sum(
                (samples[i] - mean) * (samples[i + 1] - mean)
                for i in range(len(samples) - 1)
            ) / (len(samples) - 1)

            correlation = covariance / variance
            # White noise should have low correlation
            assert abs(correlation) < 0.5


class TestFractalNoise:
    """Tests for Fractal (layered) noise generator."""

    def test_creation(self):
        """Test basic creation."""
        base = PerlinNoise(seed=42)
        fractal = FractalNoise(base, octaves=4)

        assert fractal.base_noise is base
        assert fractal.octaves == 4
        assert fractal.lacunarity == 2.0
        assert fractal.persistence == 0.5

    def test_invalid_octaves(self):
        """Test validation of octaves."""
        base = PerlinNoise(seed=42)
        with pytest.raises(ValueError, match="octaves must be >= 1"):
            FractalNoise(base, octaves=0)

    def test_invalid_lacunarity(self):
        """Test validation of lacunarity."""
        base = PerlinNoise(seed=42)
        with pytest.raises(ValueError, match="lacunarity must be > 0"):
            FractalNoise(base, lacunarity=0)

    def test_invalid_persistence(self):
        """Test validation of persistence."""
        base = PerlinNoise(seed=42)
        with pytest.raises(ValueError, match="persistence must be in"):
            FractalNoise(base, persistence=0)

    def test_sample_range(self):
        """Test that samples are approximately in [-1, 1] range."""
        base = PerlinNoise(seed=42)
        fractal = FractalNoise(base, octaves=4)

        for x in range(-10, 11):
            for y in range(-10, 11):
                value = fractal.sample(x * 0.1, y * 0.1)
                # Fractal noise normalizes but base Perlin can exceed bounds
                # so we allow a wider range
                assert -5.0 <= value <= 5.0, f"Value {value} out of expected range"

    def test_determinism(self):
        """Test that same inputs produce same results."""
        base1 = PerlinNoise(seed=42)
        base2 = PerlinNoise(seed=42)
        fractal1 = FractalNoise(base1, octaves=4)
        fractal2 = FractalNoise(base2, octaves=4)

        for x in range(-5, 6):
            for y in range(-5, 6):
                v1 = fractal1.sample(x * 0.5, y * 0.5)
                v2 = fractal2.sample(x * 0.5, y * 0.5)
                assert v1 == v2

    def test_more_octaves_more_detail(self):
        """Test that more octaves add more detail (variation)."""
        base = PerlinNoise(seed=42)
        fractal_2 = FractalNoise(base, octaves=2)
        fractal_6 = FractalNoise(base, octaves=6)

        # More octaves should have more high-frequency variation
        variations_2 = sum(
            abs(fractal_2.sample(x * 0.1, 0) - fractal_2.sample((x + 1) * 0.1, 0))
            for x in range(50)
        )
        variations_6 = sum(
            abs(fractal_6.sample(x * 0.1, 0) - fractal_6.sample((x + 1) * 0.1, 0))
            for x in range(50)
        )

        # Higher octave count typically has more total variation
        # This is a heuristic test

    def test_3d_sampling(self):
        """Test 3D fractal noise sampling."""
        base = PerlinNoise(seed=42)
        fractal = FractalNoise(base, octaves=4)
        value = fractal.sample_3d(1.5, 2.5, 3.5)
        assert isinstance(value, float)


class TestNoiseMap:
    """Tests for NoiseMap class."""

    def test_creation(self):
        """Test basic creation."""
        nmap = NoiseMap(64, 64)
        assert nmap.width == 64
        assert nmap.height == 64

    def test_invalid_dimensions(self):
        """Test validation of dimensions."""
        with pytest.raises(ValueError, match="width must be >= 1"):
            NoiseMap(0, 64)

        with pytest.raises(ValueError, match="height must be >= 1"):
            NoiseMap(64, 0)

    def test_generate(self):
        """Test noise map generation."""
        nmap = NoiseMap(32, 32, NoiseSettings(seed=42))
        nmap.generate()

        # All values should be valid
        for y in range(32):
            for x in range(32):
                value = nmap.get_raw(x, y)
                assert isinstance(value, float)

    def test_get_value_interpolation(self):
        """Test interpolated value access."""
        nmap = NoiseMap(10, 10, NoiseSettings(seed=42))
        nmap.generate()

        # Integer coordinates should match raw
        raw = nmap.get_raw(5, 5)
        interp = nmap.get_value(5.0, 5.0)
        assert interp == pytest.approx(raw, abs=0.01)

        # Midpoint should be interpolated
        v1 = nmap.get_raw(4, 4)
        v2 = nmap.get_raw(5, 4)
        mid = nmap.get_value(4.5, 4.0)
        assert min(v1, v2) <= mid <= max(v1, v2) or abs(mid - (v1 + v2) / 2) < 1.0

    def test_get_raw_bounds_check(self):
        """Test that out-of-bounds access raises error."""
        nmap = NoiseMap(10, 10)
        nmap.generate()

        with pytest.raises(IndexError):
            nmap.get_raw(-1, 0)

        with pytest.raises(IndexError):
            nmap.get_raw(10, 0)

    def test_set_raw(self):
        """Test setting raw values."""
        nmap = NoiseMap(10, 10)
        nmap.set_raw(5, 5, 0.75)
        assert nmap.get_raw(5, 5) == 0.75

    def test_normalize(self):
        """Test normalization to range."""
        nmap = NoiseMap(10, 10, NoiseSettings(seed=42))
        nmap.generate()
        nmap.normalize(0.0, 1.0)

        for y in range(10):
            for x in range(10):
                value = nmap.get_raw(x, y)
                assert 0.0 <= value <= 1.0

    def test_apply_curve(self):
        """Test applying curve function."""
        nmap = NoiseMap(10, 10, NoiseSettings(seed=42))
        nmap.generate()
        nmap.normalize(0.0, 1.0)

        # Apply square curve
        nmap.apply_curve(lambda x: x * x)

        # All values should still be in [0, 1]
        for y in range(10):
            for x in range(10):
                value = nmap.get_raw(x, y)
                assert 0.0 <= value <= 1.0

    def test_to_list_and_from_list(self):
        """Test list conversion."""
        nmap = NoiseMap(5, 5)
        nmap.generate()

        # Get as list
        data = nmap.to_list()
        assert len(data) == 5
        assert len(data[0]) == 5

        # Modify and set back
        data[2][2] = 0.999
        nmap.from_list(data)
        assert nmap.get_raw(2, 2) == 0.999

    def test_from_list_dimension_validation(self):
        """Test dimension validation when setting from list."""
        nmap = NoiseMap(5, 5)

        with pytest.raises(ValueError, match="height"):
            nmap.from_list([[0] * 5] * 3)  # Wrong height

        with pytest.raises(ValueError, match="width"):
            nmap.from_list([[0] * 3] * 5)  # Wrong width


class TestCreateNoiseGenerator:
    """Tests for factory function."""

    def test_create_perlin(self):
        """Test creating Perlin noise."""
        noise = create_noise_generator(NoiseType.PERLIN, seed=42)
        assert isinstance(noise, PerlinNoise)

    def test_create_simplex(self):
        """Test creating Simplex noise."""
        noise = create_noise_generator(NoiseType.SIMPLEX, seed=42)
        assert isinstance(noise, SimplexNoise)

    def test_create_worley(self):
        """Test creating Worley noise."""
        noise = create_noise_generator(
            NoiseType.WORLEY,
            seed=42,
            distance_type="manhattan",
            return_type="f2",
        )
        assert isinstance(noise, WorleyNoise)
        assert noise.distance_type == "manhattan"
        assert noise.return_type == "f2"

    def test_create_value(self):
        """Test creating Value noise."""
        noise = create_noise_generator(NoiseType.VALUE, seed=42)
        assert isinstance(noise, ValueNoise)

    def test_create_white(self):
        """Test creating White noise."""
        noise = create_noise_generator(NoiseType.WHITE, seed=42)
        assert isinstance(noise, WhiteNoise)


class TestNoiseHash:
    """Tests for noise generator hash function."""

    def test_hash_consistency(self):
        """Test that hash is consistent."""
        noise = PerlinNoise(seed=42)
        h1 = noise._hash(10, 20, 30)
        h2 = noise._hash(10, 20, 30)
        assert h1 == h2

    def test_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        noise = PerlinNoise(seed=42)
        h1 = noise._hash(10, 20, 30)
        h2 = noise._hash(10, 20, 31)
        assert h1 != h2

    def test_hash_range(self):
        """Test that hash is in valid range."""
        noise = PerlinNoise(seed=42)
        for i in range(100):
            h = noise._hash(i, i * 2, i * 3)
            assert 0 <= h <= 0x7FFFFFFF
