"""
PCG Deterministic Seed Management.

Provides seed generation and management for procedural content:
- SeedGenerator: Core seed hashing and combining
- ChunkSeed: Deterministic seeds from world position
- LayerSeed: Layer-specific seeds
- InstanceSeed: Instance-specific seeds
- RandomStream: Deterministic random number sequence

All operations are fully deterministic given the same inputs.
Uses Trinity Pattern with @seeded for deterministic generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TypeVar

from engine.world.pcg.constants import (
    LCG_MULTIPLIER,
    LCG_INCREMENT,
    LCG_MODULUS,
    MINSTD_MULTIPLIER,
    MINSTD_MODULUS,
)

T = TypeVar("T")


@dataclass
class SeedConfig:
    """Configuration for seed generation."""

    world_seed: int = 0
    chunk_seed_offset: int = 0x12345678
    layer_seed_offset: int = 0x87654321

    def __post_init__(self) -> None:
        """Ensure seeds are within valid range."""
        self.world_seed = self.world_seed & 0x7FFFFFFF
        self.chunk_seed_offset = self.chunk_seed_offset & 0x7FFFFFFF
        self.layer_seed_offset = self.layer_seed_offset & 0x7FFFFFFF


class SeedGenerator:
    """
    Core seed generation and hashing utilities.

    Provides deterministic seed computation from various inputs.
    """

    def __init__(self, base_seed: int = 0) -> None:
        """
        Initialize seed generator.

        Args:
            base_seed: Base seed for all operations
        """
        self._base_seed = base_seed & 0x7FFFFFFF

    @property
    def base_seed(self) -> int:
        """Get the base seed."""
        return self._base_seed

    def hash_position(self, x: int, z: int) -> int:
        """
        Hash 2D position to a seed.

        Uses a simple but effective spatial hash.

        Args:
            x: X coordinate (integer)
            z: Z coordinate (integer)

        Returns:
            Deterministic hash value
        """
        # Ensure positive values
        x = x & 0x7FFFFFFF
        z = z & 0x7FFFFFFF

        # Combine with base seed using mix function
        h = self._base_seed
        h = self._mix(h, x)
        h = self._mix(h, z)
        return h & 0x7FFFFFFF

    def hash_position_3d(self, x: int, y: int, z: int) -> int:
        """
        Hash 3D position to a seed.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Deterministic hash value
        """
        x = x & 0x7FFFFFFF
        y = y & 0x7FFFFFFF
        z = z & 0x7FFFFFFF

        h = self._base_seed
        h = self._mix(h, x)
        h = self._mix(h, y)
        h = self._mix(h, z)
        return h & 0x7FFFFFFF

    def hash_string(self, s: str) -> int:
        """
        Hash a string to a seed.

        Args:
            s: String to hash

        Returns:
            Deterministic hash value
        """
        h = self._base_seed
        for char in s:
            h = self._mix(h, ord(char))
        return h & 0x7FFFFFFF

    def combine_seeds(self, *seeds: int) -> int:
        """
        Combine multiple seeds into one.

        Args:
            *seeds: Seeds to combine

        Returns:
            Combined seed value
        """
        h = self._base_seed
        for seed in seeds:
            h = self._mix(h, seed & 0x7FFFFFFF)
        return h & 0x7FFFFFFF

    @staticmethod
    def _mix(h: int, value: int) -> int:
        """
        Mix a value into a hash.

        Uses a variant of the FNV-1a mixing function.

        Args:
            h: Current hash
            value: Value to mix in

        Returns:
            Updated hash
        """
        h ^= value
        h = (h * 0x01000193) & 0xFFFFFFFF
        h ^= (h >> 16)
        return h & 0x7FFFFFFF


class ChunkSeed:
    """
    Generates deterministic seeds for world chunks.

    Each chunk at a given position will always have the same seed
    given the same world seed.
    """

    def __init__(
        self,
        world_seed: int,
        chunk_x: int,
        chunk_z: int,
        config: Optional[SeedConfig] = None,
    ) -> None:
        """
        Initialize chunk seed.

        Args:
            world_seed: World-level seed
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate
            config: Optional seed configuration
        """
        self._world_seed = world_seed & 0x7FFFFFFF
        self._chunk_x = chunk_x
        self._chunk_z = chunk_z
        self._config = config or SeedConfig(world_seed=world_seed)
        self._cached_seed: Optional[int] = None

    @property
    def world_seed(self) -> int:
        """Get the world seed."""
        return self._world_seed

    @property
    def chunk_x(self) -> int:
        """Get chunk X coordinate."""
        return self._chunk_x

    @property
    def chunk_z(self) -> int:
        """Get chunk Z coordinate."""
        return self._chunk_z

    def get_seed(self) -> int:
        """
        Get the deterministic seed for this chunk.

        Returns:
            Chunk seed value
        """
        if self._cached_seed is not None:
            return self._cached_seed

        generator = SeedGenerator(self._world_seed)
        base = generator.combine_seeds(
            self._config.chunk_seed_offset,
            self._chunk_x,
            self._chunk_z,
        )

        self._cached_seed = base
        return base

    def get_sub_seed(self, sub_id: int) -> int:
        """
        Get a sub-seed for features within this chunk.

        Args:
            sub_id: Unique identifier for the sub-feature

        Returns:
            Sub-seed value
        """
        generator = SeedGenerator(self.get_seed())
        return generator.combine_seeds(sub_id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on coordinates and world seed."""
        if not isinstance(other, ChunkSeed):
            return NotImplemented
        return (
            self._world_seed == other._world_seed
            and self._chunk_x == other._chunk_x
            and self._chunk_z == other._chunk_z
        )

    def __hash__(self) -> int:
        """Hash based on coordinates and world seed."""
        return hash((self._world_seed, self._chunk_x, self._chunk_z))


class LayerSeed:
    """
    Generates seeds for specific layers within a chunk.

    Different layers (terrain, foliage, structures) need different
    but deterministic random sequences.
    """

    def __init__(
        self,
        chunk_seed: ChunkSeed,
        layer_name: str,
        config: Optional[SeedConfig] = None,
    ) -> None:
        """
        Initialize layer seed.

        Args:
            chunk_seed: Parent chunk seed
            layer_name: Name of the layer
            config: Optional seed configuration
        """
        self._chunk_seed = chunk_seed
        self._layer_name = layer_name
        self._config = config or SeedConfig(world_seed=chunk_seed.world_seed)
        self._cached_seed: Optional[int] = None

    @property
    def chunk_seed(self) -> ChunkSeed:
        """Get the parent chunk seed."""
        return self._chunk_seed

    @property
    def layer_name(self) -> str:
        """Get the layer name."""
        return self._layer_name

    def get_seed(self) -> int:
        """
        Get the deterministic seed for this layer.

        Returns:
            Layer seed value
        """
        if self._cached_seed is not None:
            return self._cached_seed

        generator = SeedGenerator(self._chunk_seed.get_seed())
        layer_hash = generator.hash_string(self._layer_name)
        base = generator.combine_seeds(
            self._config.layer_seed_offset,
            layer_hash,
        )

        self._cached_seed = base
        return base


class InstanceSeed:
    """
    Generates seeds for specific instances within a layer.

    Each placed object or feature gets a unique but deterministic seed.
    """

    def __init__(
        self,
        parent_seed: int,
        instance_index: int,
    ) -> None:
        """
        Initialize instance seed.

        Args:
            parent_seed: Parent seed (chunk or layer)
            instance_index: Index of this instance
        """
        self._parent_seed = parent_seed & 0x7FFFFFFF
        self._instance_index = instance_index
        self._cached_seed: Optional[int] = None

    @property
    def parent_seed(self) -> int:
        """Get the parent seed."""
        return self._parent_seed

    @property
    def instance_index(self) -> int:
        """Get the instance index."""
        return self._instance_index

    def get_seed(self) -> int:
        """
        Get the deterministic seed for this instance.

        Returns:
            Instance seed value
        """
        if self._cached_seed is not None:
            return self._cached_seed

        generator = SeedGenerator(self._parent_seed)
        self._cached_seed = generator.combine_seeds(self._instance_index)
        return self._cached_seed


class RandomStream:
    """
    Deterministic random number stream.

    Provides a sequence of random values that is reproducible
    given the same seed.
    """

    # LCG parameters (same as MINSTD)
    _MULTIPLIER = 48271
    _MODULUS = 2147483647  # 2^31 - 1

    def __init__(self, seed: int = 0) -> None:
        """
        Initialize random stream.

        Args:
            seed: Starting seed
        """
        self._seed = seed & 0x7FFFFFFF
        self._state = self._seed
        # MINSTD LCG requires state in range [1, MODULUS-1]
        # State 0 gets stuck at 0, state == MODULUS gives 0 on first advance
        if self._state == 0 or self._state == self._MODULUS:
            self._state = 1  # Avoid degenerate states

    @property
    def seed(self) -> int:
        """Get the original seed."""
        return self._seed

    @property
    def state(self) -> int:
        """Get current state."""
        return self._state

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the stream to initial or new seed.

        Args:
            seed: New seed (uses original if None)
        """
        if seed is not None:
            self._seed = seed & 0x7FFFFFFF
        self._state = self._seed
        # MINSTD LCG requires state in range [1, MODULUS-1]
        if self._state == 0 or self._state == self._MODULUS:
            self._state = 1

    def _advance(self) -> int:
        """Advance state and return new value."""
        self._state = (self._state * self._MULTIPLIER) % self._MODULUS
        return self._state

    def next_int(self, min_val: int, max_val: int) -> int:
        """
        Generate random integer in [min_val, max_val].

        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            Random integer in range
        """
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        if min_val == max_val:
            return min_val

        range_val = max_val - min_val + 1
        return min_val + (self._advance() % range_val)

    def next_float(self, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Generate random float in [min_val, max_val].

        Args:
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Random float in range
        """
        normalized = self._advance() / self._MODULUS
        return min_val + normalized * (max_val - min_val)

    def next_bool(self, probability: float = 0.5) -> bool:
        """
        Generate random boolean.

        Args:
            probability: Probability of True [0, 1]

        Returns:
            Random boolean
        """
        return self.next_float() < probability

    def next_point_in_circle(self, radius: float) -> Tuple[float, float]:
        """
        Generate random point in a circle centered at origin.

        Args:
            radius: Circle radius

        Returns:
            (x, y) point within circle
        """
        # Rejection sampling for uniform distribution
        max_attempts = 100
        for _ in range(max_attempts):
            x = self.next_float(-radius, radius)
            y = self.next_float(-radius, radius)
            if x * x + y * y <= radius * radius:
                return (x, y)

        # Fallback: use angle-based sampling
        angle = self.next_float(0, 2 * math.pi)
        r = radius * math.sqrt(self.next_float())
        return (r * math.cos(angle), r * math.sin(angle))

    def next_point_on_sphere(self) -> Tuple[float, float, float]:
        """
        Generate random point on unit sphere surface.

        Uses rejection sampling for uniform distribution.

        Returns:
            (x, y, z) point on unit sphere
        """
        # Marsaglia's method
        while True:
            u = self.next_float(-1, 1)
            v = self.next_float(-1, 1)
            s = u * u + v * v
            if s < 1:
                factor = 2 * math.sqrt(1 - s)
                return (u * factor, v * factor, 1 - 2 * s)

    def next_direction_2d(self) -> Tuple[float, float]:
        """
        Generate random 2D unit direction.

        Returns:
            (x, y) unit vector
        """
        angle = self.next_float(0, 2 * math.pi)
        return (math.cos(angle), math.sin(angle))

    def next_gaussian(self, mean: float = 0.0, std_dev: float = 1.0) -> float:
        """
        Generate random value from Gaussian distribution.

        Uses Box-Muller transform.

        Args:
            mean: Distribution mean
            std_dev: Distribution standard deviation

        Returns:
            Random value from Gaussian distribution
        """
        # Box-Muller transform
        u1 = self.next_float(1e-10, 1.0)  # Avoid log(0)
        u2 = self.next_float(0, 2 * math.pi)
        z = math.sqrt(-2 * math.log(u1)) * math.cos(u2)
        return mean + z * std_dev

    def shuffle(self, items: List[T]) -> List[T]:
        """
        Return a shuffled copy of the list.

        Uses Fisher-Yates shuffle.

        Args:
            items: List to shuffle

        Returns:
            New shuffled list
        """
        result = list(items)
        for i in range(len(result) - 1, 0, -1):
            j = self.next_int(0, i)
            result[i], result[j] = result[j], result[i]
        return result

    def choice(self, items: List[T]) -> T:
        """
        Select a random item from a list.

        Args:
            items: List to choose from

        Returns:
            Random item

        Raises:
            IndexError: If list is empty
        """
        if not items:
            raise IndexError("Cannot choose from empty list")
        return items[self.next_int(0, len(items) - 1)]

    def choices(self, items: List[T], count: int) -> List[T]:
        """
        Select multiple random items (with replacement).

        Args:
            items: List to choose from
            count: Number of items to select

        Returns:
            List of selected items
        """
        if not items:
            return []
        return [self.choice(items) for _ in range(count)]

    def sample(self, items: List[T], count: int) -> List[T]:
        """
        Select multiple random items (without replacement).

        Args:
            items: List to choose from
            count: Number of items to select

        Returns:
            List of selected items (up to min(count, len(items)))
        """
        if not items:
            return []
        count = min(count, len(items))
        shuffled = self.shuffle(items)
        return shuffled[:count]

    def weighted_choice(
        self, items: List[T], weights: List[float]
    ) -> T:
        """
        Select a random item using weights.

        Args:
            items: List to choose from
            weights: Weight for each item

        Returns:
            Weighted random item

        Raises:
            ValueError: If items and weights have different lengths
        """
        if len(items) != len(weights):
            raise ValueError("Items and weights must have same length")
        if not items:
            raise IndexError("Cannot choose from empty list")

        total = sum(weights)
        if total <= 0:
            raise ValueError("Total weight must be positive")

        r = self.next_float(0, total)
        cumulative = 0.0
        for item, weight in zip(items, weights):
            cumulative += weight
            if r <= cumulative:
                return item

        return items[-1]  # Fallback


class DeterministicRandom:
    """
    Factory for creating deterministic random streams.

    Provides convenient methods for creating streams from various inputs.
    """

    @staticmethod
    def from_seed(seed: int) -> RandomStream:
        """
        Create a random stream from a seed.

        Args:
            seed: Starting seed

        Returns:
            New random stream
        """
        return RandomStream(seed)

    @staticmethod
    def from_position(world_seed: int, x: int, z: int) -> RandomStream:
        """
        Create a random stream from world position.

        Args:
            world_seed: World seed
            x: X coordinate
            z: Z coordinate

        Returns:
            New random stream with position-based seed
        """
        generator = SeedGenerator(world_seed)
        seed = generator.hash_position(x, z)
        return RandomStream(seed)

    @staticmethod
    def from_position_3d(world_seed: int, x: int, y: int, z: int) -> RandomStream:
        """
        Create a random stream from 3D position.

        Args:
            world_seed: World seed
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            New random stream with position-based seed
        """
        generator = SeedGenerator(world_seed)
        seed = generator.hash_position_3d(x, y, z)
        return RandomStream(seed)

    @staticmethod
    def from_chunk(chunk_seed: ChunkSeed) -> RandomStream:
        """
        Create a random stream from a chunk seed.

        Args:
            chunk_seed: Chunk seed object

        Returns:
            New random stream for the chunk
        """
        return RandomStream(chunk_seed.get_seed())

    @staticmethod
    def from_layer(layer_seed: LayerSeed) -> RandomStream:
        """
        Create a random stream from a layer seed.

        Args:
            layer_seed: Layer seed object

        Returns:
            New random stream for the layer
        """
        return RandomStream(layer_seed.get_seed())

    @staticmethod
    def from_string(base_seed: int, s: str) -> RandomStream:
        """
        Create a random stream from a string.

        Args:
            base_seed: Base seed
            s: String to hash

        Returns:
            New random stream with string-based seed
        """
        generator = SeedGenerator(base_seed)
        seed = generator.hash_string(s)
        return RandomStream(seed)


# Utility functions
def combine_seeds(*seeds: int, base: int = 0) -> int:
    """
    Combine multiple seeds into one.

    Args:
        *seeds: Seeds to combine
        base: Base seed value

    Returns:
        Combined seed
    """
    generator = SeedGenerator(base)
    return generator.combine_seeds(*seeds)


def position_to_seed(world_seed: int, x: int, z: int) -> int:
    """
    Convert a position to a seed.

    Args:
        world_seed: World seed
        x: X coordinate
        z: Z coordinate

    Returns:
        Position-based seed
    """
    generator = SeedGenerator(world_seed)
    return generator.hash_position(x, z)


def string_to_seed(base_seed: int, s: str) -> int:
    """
    Convert a string to a seed.

    Args:
        base_seed: Base seed
        s: String to hash

    Returns:
        String-based seed
    """
    generator = SeedGenerator(base_seed)
    return generator.hash_string(s)


__all__ = [
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
]
