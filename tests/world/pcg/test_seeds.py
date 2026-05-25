"""
Tests for PCG seed management.

Tests cover:
- Seed determinism
- Position hashing
- Seed combining
- RandomStream distribution
"""

import math
import pytest

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


class TestSeedConfig:
    """Tests for SeedConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = SeedConfig()
        assert config.world_seed == 0
        assert config.chunk_seed_offset != 0
        assert config.layer_seed_offset != 0

    def test_custom_values(self):
        """Test custom config values."""
        config = SeedConfig(
            world_seed=12345,
            chunk_seed_offset=11111,
            layer_seed_offset=22222,
        )
        assert config.world_seed == 12345
        assert config.chunk_seed_offset == 11111
        assert config.layer_seed_offset == 22222

    def test_large_seed_clipping(self):
        """Test that large seeds are clipped to valid range."""
        config = SeedConfig(world_seed=0xFFFFFFFF)
        assert 0 <= config.world_seed <= 0x7FFFFFFF


class TestSeedGenerator:
    """Tests for SeedGenerator class."""

    def test_creation(self):
        """Test basic creation."""
        gen = SeedGenerator(42)
        assert gen.base_seed == 42

    def test_hash_position_determinism(self):
        """Test that position hashing is deterministic."""
        gen = SeedGenerator(42)

        h1 = gen.hash_position(100, 200)
        h2 = gen.hash_position(100, 200)

        assert h1 == h2

    def test_hash_position_different_positions(self):
        """Test that different positions produce different hashes."""
        gen = SeedGenerator(42)

        h1 = gen.hash_position(100, 200)
        h2 = gen.hash_position(100, 201)
        h3 = gen.hash_position(101, 200)

        assert h1 != h2
        assert h1 != h3
        assert h2 != h3

    def test_hash_position_negative_coords(self):
        """Test hashing with negative coordinates."""
        gen = SeedGenerator(42)

        h1 = gen.hash_position(-100, -200)
        h2 = gen.hash_position(-100, -200)

        assert h1 == h2
        assert 0 <= h1 <= 0x7FFFFFFF

    def test_hash_position_3d(self):
        """Test 3D position hashing."""
        gen = SeedGenerator(42)

        h1 = gen.hash_position_3d(10, 20, 30)
        h2 = gen.hash_position_3d(10, 20, 30)
        h3 = gen.hash_position_3d(10, 20, 31)

        assert h1 == h2
        assert h1 != h3

    def test_hash_string_determinism(self):
        """Test that string hashing is deterministic."""
        gen = SeedGenerator(42)

        h1 = gen.hash_string("test_layer")
        h2 = gen.hash_string("test_layer")

        assert h1 == h2

    def test_hash_string_different_strings(self):
        """Test that different strings produce different hashes."""
        gen = SeedGenerator(42)

        h1 = gen.hash_string("layer_a")
        h2 = gen.hash_string("layer_b")

        assert h1 != h2

    def test_hash_string_empty(self):
        """Test hashing empty string."""
        gen = SeedGenerator(42)
        h = gen.hash_string("")
        assert 0 <= h <= 0x7FFFFFFF

    def test_combine_seeds_determinism(self):
        """Test that seed combining is deterministic."""
        gen = SeedGenerator(42)

        h1 = gen.combine_seeds(1, 2, 3)
        h2 = gen.combine_seeds(1, 2, 3)

        assert h1 == h2

    def test_combine_seeds_order_matters(self):
        """Test that order of seeds matters."""
        gen = SeedGenerator(42)

        h1 = gen.combine_seeds(1, 2, 3)
        h2 = gen.combine_seeds(3, 2, 1)

        assert h1 != h2

    def test_combine_seeds_different_count(self):
        """Test combining different numbers of seeds."""
        gen = SeedGenerator(42)

        h1 = gen.combine_seeds(1, 2)
        h2 = gen.combine_seeds(1, 2, 3)

        assert h1 != h2


class TestChunkSeed:
    """Tests for ChunkSeed class."""

    def test_creation(self):
        """Test basic creation."""
        chunk = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        assert chunk.world_seed == 42
        assert chunk.chunk_x == 10
        assert chunk.chunk_z == 20

    def test_get_seed_determinism(self):
        """Test that get_seed is deterministic."""
        chunk1 = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        chunk2 = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)

        assert chunk1.get_seed() == chunk2.get_seed()

    def test_get_seed_caching(self):
        """Test that seed is cached."""
        chunk = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)

        seed1 = chunk.get_seed()
        seed2 = chunk.get_seed()

        assert seed1 == seed2
        assert chunk._cached_seed is not None

    def test_different_chunks_different_seeds(self):
        """Test that different chunks have different seeds."""
        chunk1 = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        chunk2 = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=21)
        chunk3 = ChunkSeed(world_seed=42, chunk_x=11, chunk_z=20)

        seeds = {chunk1.get_seed(), chunk2.get_seed(), chunk3.get_seed()}
        assert len(seeds) == 3  # All different

    def test_different_worlds_different_seeds(self):
        """Test that different world seeds produce different chunk seeds."""
        chunk1 = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        chunk2 = ChunkSeed(world_seed=43, chunk_x=10, chunk_z=20)

        assert chunk1.get_seed() != chunk2.get_seed()

    def test_get_sub_seed(self):
        """Test sub-seed generation."""
        chunk = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)

        sub1 = chunk.get_sub_seed(1)
        sub2 = chunk.get_sub_seed(2)

        assert sub1 != sub2
        assert 0 <= sub1 <= 0x7FFFFFFF

    def test_equality(self):
        """Test equality comparison."""
        chunk1 = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        chunk2 = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        chunk3 = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=21)

        assert chunk1 == chunk2
        assert chunk1 != chunk3

    def test_hash(self):
        """Test hash function."""
        chunk1 = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        chunk2 = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)

        assert hash(chunk1) == hash(chunk2)

        # Can use in sets
        chunks = {chunk1, chunk2}
        assert len(chunks) == 1


class TestLayerSeed:
    """Tests for LayerSeed class."""

    def test_creation(self):
        """Test basic creation."""
        chunk = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        layer = LayerSeed(chunk, "foliage")

        assert layer.chunk_seed is chunk
        assert layer.layer_name == "foliage"

    def test_get_seed_determinism(self):
        """Test that get_seed is deterministic."""
        chunk = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        layer1 = LayerSeed(chunk, "foliage")
        layer2 = LayerSeed(
            ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20),
            "foliage",
        )

        assert layer1.get_seed() == layer2.get_seed()

    def test_different_layers_different_seeds(self):
        """Test that different layer names produce different seeds."""
        chunk = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        layer1 = LayerSeed(chunk, "foliage")
        layer2 = LayerSeed(chunk, "structures")

        assert layer1.get_seed() != layer2.get_seed()


class TestInstanceSeed:
    """Tests for InstanceSeed class."""

    def test_creation(self):
        """Test basic creation."""
        instance = InstanceSeed(parent_seed=12345, instance_index=0)
        assert instance.parent_seed == 12345
        assert instance.instance_index == 0

    def test_get_seed_determinism(self):
        """Test that get_seed is deterministic."""
        inst1 = InstanceSeed(parent_seed=12345, instance_index=5)
        inst2 = InstanceSeed(parent_seed=12345, instance_index=5)

        assert inst1.get_seed() == inst2.get_seed()

    def test_different_instances_different_seeds(self):
        """Test that different instance indices produce different seeds."""
        inst1 = InstanceSeed(parent_seed=12345, instance_index=0)
        inst2 = InstanceSeed(parent_seed=12345, instance_index=1)

        assert inst1.get_seed() != inst2.get_seed()


class TestRandomStream:
    """Tests for RandomStream class."""

    def test_creation(self):
        """Test basic creation."""
        stream = RandomStream(42)
        assert stream.seed == 42

    def test_determinism(self):
        """Test that same seed produces same sequence."""
        stream1 = RandomStream(42)
        stream2 = RandomStream(42)

        for _ in range(100):
            assert stream1.next_int(0, 1000) == stream2.next_int(0, 1000)

    def test_reset(self):
        """Test resetting the stream."""
        stream = RandomStream(42)

        values1 = [stream.next_int(0, 1000) for _ in range(10)]
        stream.reset()
        values2 = [stream.next_int(0, 1000) for _ in range(10)]

        assert values1 == values2

    def test_reset_new_seed(self):
        """Test resetting with a new seed."""
        stream = RandomStream(42)
        values1 = [stream.next_int(0, 1000) for _ in range(10)]

        stream.reset(43)
        values2 = [stream.next_int(0, 1000) for _ in range(10)]

        assert values1 != values2

    def test_next_int_range(self):
        """Test integer range."""
        stream = RandomStream(42)

        for _ in range(1000):
            value = stream.next_int(10, 20)
            assert 10 <= value <= 20

    def test_next_int_single_value(self):
        """Test integer range with single value."""
        stream = RandomStream(42)
        value = stream.next_int(5, 5)
        assert value == 5

    def test_next_int_swapped_range(self):
        """Test integer range with swapped min/max."""
        stream = RandomStream(42)
        value = stream.next_int(20, 10)
        assert 10 <= value <= 20

    def test_next_float_range(self):
        """Test float range."""
        stream = RandomStream(42)

        for _ in range(1000):
            value = stream.next_float(0.5, 1.5)
            assert 0.5 <= value <= 1.5

    def test_next_float_default_range(self):
        """Test float default range [0, 1]."""
        stream = RandomStream(42)

        for _ in range(100):
            value = stream.next_float()
            assert 0.0 <= value <= 1.0

    def test_next_bool_probability(self):
        """Test boolean with probability."""
        stream = RandomStream(42)

        # With p=1.0, should always be True
        for _ in range(100):
            assert stream.next_bool(1.0) is True

        stream.reset()

        # With p=0.0, should always be False
        for _ in range(100):
            assert stream.next_bool(0.0) is False

    def test_next_bool_distribution(self):
        """Test boolean distribution."""
        stream = RandomStream(42)
        count = sum(1 for _ in range(10000) if stream.next_bool(0.5))

        # Should be roughly 50%
        assert 4500 < count < 5500

    def test_next_point_in_circle(self):
        """Test point generation in circle."""
        stream = RandomStream(42)
        radius = 5.0

        for _ in range(1000):
            x, y = stream.next_point_in_circle(radius)
            dist = math.sqrt(x * x + y * y)
            assert dist <= radius

    def test_next_point_on_sphere(self):
        """Test point generation on sphere surface."""
        stream = RandomStream(42)

        for _ in range(100):
            x, y, z = stream.next_point_on_sphere()
            dist = math.sqrt(x * x + y * y + z * z)
            assert dist == pytest.approx(1.0, abs=0.01)

    def test_next_direction_2d(self):
        """Test 2D unit direction generation."""
        stream = RandomStream(42)

        for _ in range(100):
            x, y = stream.next_direction_2d()
            length = math.sqrt(x * x + y * y)
            assert length == pytest.approx(1.0, abs=0.001)

    def test_next_gaussian_distribution(self):
        """Test Gaussian distribution."""
        stream = RandomStream(42)
        values = [stream.next_gaussian(0.0, 1.0) for _ in range(10000)]

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)

        # Should be close to mean=0, std=1
        assert abs(mean) < 0.1
        assert abs(std_dev - 1.0) < 0.1

    def test_shuffle_determinism(self):
        """Test that shuffle is deterministic."""
        stream1 = RandomStream(42)
        stream2 = RandomStream(42)

        items = list(range(10))
        shuffled1 = stream1.shuffle(items)
        shuffled2 = stream2.shuffle(items)

        assert shuffled1 == shuffled2

    def test_shuffle_completeness(self):
        """Test that shuffle contains all items."""
        stream = RandomStream(42)
        items = list(range(10))
        shuffled = stream.shuffle(items)

        assert sorted(shuffled) == items

    def test_shuffle_original_unchanged(self):
        """Test that original list is unchanged."""
        stream = RandomStream(42)
        items = list(range(10))
        original = items.copy()

        stream.shuffle(items)
        assert items == original

    def test_choice(self):
        """Test random choice."""
        stream = RandomStream(42)
        items = ["a", "b", "c", "d", "e"]

        for _ in range(100):
            choice = stream.choice(items)
            assert choice in items

    def test_choice_empty_list(self):
        """Test choice from empty list."""
        stream = RandomStream(42)
        with pytest.raises(IndexError):
            stream.choice([])

    def test_choices(self):
        """Test multiple choices with replacement."""
        stream = RandomStream(42)
        items = ["a", "b", "c"]

        choices = stream.choices(items, 5)
        assert len(choices) == 5
        for c in choices:
            assert c in items

    def test_sample(self):
        """Test sampling without replacement."""
        stream = RandomStream(42)
        items = list(range(10))

        sample = stream.sample(items, 5)
        assert len(sample) == 5
        assert len(set(sample)) == 5  # All unique

    def test_sample_more_than_available(self):
        """Test sampling more items than available."""
        stream = RandomStream(42)
        items = [1, 2, 3]

        sample = stream.sample(items, 10)
        assert len(sample) == 3

    def test_weighted_choice(self):
        """Test weighted random choice."""
        stream = RandomStream(42)
        items = ["a", "b", "c"]
        weights = [10.0, 1.0, 1.0]  # "a" should be most common

        # Count occurrences
        counts = {"a": 0, "b": 0, "c": 0}
        for _ in range(1000):
            choice = stream.weighted_choice(items, weights)
            counts[choice] += 1

        # "a" should be roughly 10x more common
        assert counts["a"] > counts["b"] * 5
        assert counts["a"] > counts["c"] * 5

    def test_weighted_choice_mismatched_lengths(self):
        """Test weighted choice with mismatched lengths."""
        stream = RandomStream(42)
        with pytest.raises(ValueError, match="same length"):
            stream.weighted_choice(["a", "b"], [1.0])

    def test_weighted_choice_empty(self):
        """Test weighted choice with empty list."""
        stream = RandomStream(42)
        with pytest.raises(IndexError):
            stream.weighted_choice([], [])


class TestDeterministicRandom:
    """Tests for DeterministicRandom factory class."""

    def test_from_seed(self):
        """Test creating stream from seed."""
        stream = DeterministicRandom.from_seed(42)
        assert isinstance(stream, RandomStream)
        assert stream.seed == 42

    def test_from_position(self):
        """Test creating stream from position."""
        stream1 = DeterministicRandom.from_position(42, 100, 200)
        stream2 = DeterministicRandom.from_position(42, 100, 200)

        # Same position = same sequence
        assert stream1.next_int(0, 1000) == stream2.next_int(0, 1000)

    def test_from_position_3d(self):
        """Test creating stream from 3D position."""
        stream1 = DeterministicRandom.from_position_3d(42, 10, 20, 30)
        stream2 = DeterministicRandom.from_position_3d(42, 10, 20, 30)

        assert stream1.next_int(0, 1000) == stream2.next_int(0, 1000)

    def test_from_chunk(self):
        """Test creating stream from chunk seed."""
        chunk = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        stream = DeterministicRandom.from_chunk(chunk)

        assert isinstance(stream, RandomStream)

    def test_from_layer(self):
        """Test creating stream from layer seed."""
        chunk = ChunkSeed(world_seed=42, chunk_x=10, chunk_z=20)
        layer = LayerSeed(chunk, "foliage")
        stream = DeterministicRandom.from_layer(layer)

        assert isinstance(stream, RandomStream)

    def test_from_string(self):
        """Test creating stream from string."""
        stream1 = DeterministicRandom.from_string(42, "test")
        stream2 = DeterministicRandom.from_string(42, "test")

        assert stream1.next_int(0, 1000) == stream2.next_int(0, 1000)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_combine_seeds(self):
        """Test combine_seeds function."""
        result1 = combine_seeds(1, 2, 3)
        result2 = combine_seeds(1, 2, 3)

        assert result1 == result2

    def test_combine_seeds_with_base(self):
        """Test combine_seeds with base parameter."""
        result1 = combine_seeds(1, 2, 3, base=42)
        result2 = combine_seeds(1, 2, 3, base=43)

        assert result1 != result2

    def test_position_to_seed(self):
        """Test position_to_seed function."""
        seed1 = position_to_seed(42, 100, 200)
        seed2 = position_to_seed(42, 100, 200)

        assert seed1 == seed2
        assert 0 <= seed1 <= 0x7FFFFFFF

    def test_string_to_seed(self):
        """Test string_to_seed function."""
        seed1 = string_to_seed(42, "foliage")
        seed2 = string_to_seed(42, "foliage")

        assert seed1 == seed2
        assert 0 <= seed1 <= 0x7FFFFFFF


class TestDistributionQuality:
    """Tests for distribution quality of random numbers."""

    def test_int_uniformity(self):
        """Test uniform distribution of integers."""
        stream = RandomStream(42)
        buckets = [0] * 10

        for _ in range(10000):
            value = stream.next_int(0, 9)
            buckets[value] += 1

        # Each bucket should have roughly 1000 values
        for count in buckets:
            assert 800 < count < 1200

    def test_float_uniformity(self):
        """Test uniform distribution of floats."""
        stream = RandomStream(42)
        buckets = [0] * 10

        for _ in range(10000):
            value = stream.next_float(0.0, 1.0)
            bucket = min(9, int(value * 10))
            buckets[bucket] += 1

        # Each bucket should have roughly 1000 values
        for count in buckets:
            assert 800 < count < 1200

    def test_circle_uniformity(self):
        """Test uniform distribution in circle."""
        stream = RandomStream(42)
        radius = 1.0

        # Check that points cover the circle evenly
        quadrant_counts = [0, 0, 0, 0]  # NE, NW, SW, SE

        for _ in range(4000):
            x, y = stream.next_point_in_circle(radius)
            if x >= 0 and y >= 0:
                quadrant_counts[0] += 1
            elif x < 0 and y >= 0:
                quadrant_counts[1] += 1
            elif x < 0 and y < 0:
                quadrant_counts[2] += 1
            else:
                quadrant_counts[3] += 1

        # Each quadrant should have roughly 1000 points
        for count in quadrant_counts:
            assert 800 < count < 1200
