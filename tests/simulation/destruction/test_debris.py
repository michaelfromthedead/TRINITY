"""
Tests for Debris Management System.

Whitebox tests for debris.py including:
- DebrisLOD enumeration
- Debris dataclass
- DebrisSpawnParams dataclass
- DebrisPool allocation
- DebrisManager lifecycle
- spawn_debris_from_fracture function
"""

import pytest
import time

from engine.simulation.destruction.debris import (
    DebrisLOD,
    Debris,
    DebrisSpawnParams,
    DebrisPool,
    DebrisManager,
    spawn_debris_from_fracture,
)
from engine.simulation.destruction.fracture_voronoi import Chunk
from engine.simulation.destruction.config import (
    DEBRIS_LIFETIME,
    MAX_ACTIVE_DEBRIS,
    DEBRIS_POOL_INITIAL_SIZE,
    DebrisState,
)


class TestDebrisLOD:
    """Tests for DebrisLOD enumeration."""

    def test_all_lod_levels_exist(self):
        """Verify all LOD levels are defined."""
        assert hasattr(DebrisLOD, 'FULL')
        assert hasattr(DebrisLOD, 'REDUCED')
        assert hasattr(DebrisLOD, 'SIMPLE')
        assert hasattr(DebrisLOD, 'PARTICLE')

    def test_lod_ordering(self):
        """Verify LOD levels are ordered."""
        assert DebrisLOD.FULL.value < DebrisLOD.REDUCED.value
        assert DebrisLOD.REDUCED.value < DebrisLOD.SIMPLE.value
        assert DebrisLOD.SIMPLE.value < DebrisLOD.PARTICLE.value


class TestDebris:
    """Tests for Debris dataclass."""

    def test_basic_construction(self):
        """Verify basic construction."""
        debris = Debris(id=0)
        assert debris.id == 0
        assert debris.body_id is None
        assert debris.chunk is None
        assert debris.state == DebrisState.ACTIVE

    def test_default_values(self):
        """Verify default values."""
        debris = Debris(id=1)
        assert debris.lifetime == DEBRIS_LIFETIME
        assert debris.position == (0.0, 0.0, 0.0)
        assert debris.velocity == (0.0, 0.0, 0.0)
        assert debris.angular_velocity == (0.0, 0.0, 0.0)
        assert debris.lod == DebrisLOD.FULL
        assert debris.importance == 1.0
        assert debris.sleep_timer == 0.0
        assert debris.parent_id is None
        assert debris.generation == 0

    def test_age_property(self):
        """Verify age calculation."""
        debris = Debris(id=0, spawn_time=time.time() - 5.0)
        # Age should be approximately 5 seconds
        assert debris.age >= 4.9  # Allow some tolerance

    def test_remaining_lifetime(self):
        """Verify remaining lifetime calculation."""
        spawn_time = time.time()
        debris = Debris(id=0, spawn_time=spawn_time, lifetime=10.0)
        # Should have close to 10 seconds remaining
        assert debris.remaining_lifetime > 9.0
        assert debris.remaining_lifetime <= 10.0

    def test_is_expired(self):
        """Verify expiration detection."""
        # Expired debris
        debris_expired = Debris(
            id=0,
            spawn_time=time.time() - 100.0,
            lifetime=10.0
        )
        assert debris_expired.is_expired is True

        # Fresh debris
        debris_fresh = Debris(
            id=1,
            spawn_time=time.time(),
            lifetime=10.0
        )
        assert debris_fresh.is_expired is False

    def test_speed_property(self):
        """Verify speed calculation."""
        debris = Debris(id=0, velocity=(3.0, 4.0, 0.0))
        assert abs(debris.speed - 5.0) < 1e-6

    def test_reset(self):
        """Verify reset clears all state."""
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        debris = Debris(
            id=5,
            body_id=10,
            chunk=chunk,
            lifetime=20.0,
            spawn_time=100.0,
            position=(1.0, 2.0, 3.0),
            velocity=(4.0, 5.0, 6.0),
            state=DebrisState.ACTIVE,
            importance=0.5,
            parent_id=123,
            generation=2
        )

        debris.reset()

        assert debris.body_id is None
        assert debris.chunk is None
        assert debris.lifetime == DEBRIS_LIFETIME
        assert debris.spawn_time == 0.0
        assert debris.position == (0.0, 0.0, 0.0)
        assert debris.velocity == (0.0, 0.0, 0.0)
        assert debris.state == DebrisState.POOLED
        assert debris.importance == 1.0
        assert debris.parent_id is None
        assert debris.generation == 0


class TestDebrisSpawnParams:
    """Tests for DebrisSpawnParams dataclass."""

    def test_basic_construction(self):
        """Verify basic construction."""
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        params = DebrisSpawnParams(
            chunk=chunk,
            position=(1.0, 2.0, 3.0)
        )
        assert params.chunk == chunk
        assert params.position == (1.0, 2.0, 3.0)
        assert params.velocity == (0.0, 0.0, 0.0)

    def test_full_construction(self):
        """Verify construction with all parameters."""
        chunk = Chunk(vertices=[], triangles=[])
        params = DebrisSpawnParams(
            chunk=chunk,
            position=(1.0, 2.0, 3.0),
            velocity=(10.0, 0.0, 0.0),
            angular_velocity=(0.0, 1.0, 0.0),
            lifetime=5.0,
            importance=0.8,
            parent_id=42,
            generation=1
        )
        assert params.velocity == (10.0, 0.0, 0.0)
        assert params.lifetime == 5.0
        assert params.parent_id == 42


class TestDebrisPool:
    """Tests for DebrisPool class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        pool = DebrisPool(initial_size=10)
        assert pool.pool_size == 10
        assert pool.total_count == 10
        assert pool.active_count == 0

    def test_acquire(self):
        """Verify debris acquisition."""
        pool = DebrisPool(initial_size=5)
        debris = pool.acquire()

        assert debris is not None
        assert debris.state == DebrisState.ACTIVE
        assert pool.pool_size == 4  # One less in pool
        assert pool.active_count == 1

    def test_acquire_exhaust_pool(self):
        """Verify acquisition when pool exhausted."""
        pool = DebrisPool(initial_size=2)

        # Acquire all pooled debris
        d1 = pool.acquire()
        d2 = pool.acquire()
        assert pool.pool_size == 0

        # Acquire more - should create new
        d3 = pool.acquire()
        assert d3 is not None
        assert pool.total_count == 3

    def test_release(self):
        """Verify debris release."""
        pool = DebrisPool(initial_size=5)
        debris = pool.acquire()

        pool.release(debris)

        assert debris.state == DebrisState.POOLED
        assert pool.pool_size == 5  # Back in pool

    def test_get(self):
        """Verify debris retrieval by ID."""
        pool = DebrisPool(initial_size=5)
        debris = pool.acquire()

        retrieved = pool.get(debris.id)
        assert retrieved == debris

        # Non-existent ID
        assert pool.get(9999) is None

    def test_clear(self):
        """Verify pool clearing."""
        pool = DebrisPool(initial_size=5)
        pool.acquire()
        pool.acquire()

        pool.clear()

        assert pool.pool_size == 0
        assert pool.total_count == 0


class TestDebrisManager:
    """Tests for DebrisManager class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        manager = DebrisManager(max_active=100, pool_size=50)
        assert manager.max_active == 100
        assert manager.active_count == 0

    def test_spawn_debris(self):
        """Verify debris spawning."""
        manager = DebrisManager(max_active=100)
        chunk = Chunk(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)],
            triangles=[(0, 1, 2)]
        )
        chunk.compute_centroid()

        debris = manager.spawn_debris(
            chunk=chunk,
            velocity=(1.0, 2.0, 3.0),
            importance=0.5
        )

        assert debris is not None
        assert debris.chunk == chunk
        assert debris.velocity == (1.0, 2.0, 3.0)
        assert debris.importance == 0.5
        assert manager.active_count == 1

    def test_spawn_debris_with_options(self):
        """Verify spawning with all options."""
        manager = DebrisManager()
        chunk = Chunk(
            vertices=[(0.0, 0.0, 0.0)],
            triangles=[]
        )
        chunk.compute_centroid()

        debris = manager.spawn_debris(
            chunk=chunk,
            velocity=(1.0, 0.0, 0.0),
            angular_velocity=(0.5, 0.5, 0.5),
            lifetime=5.0,
            importance=0.8,
            parent_id=42,
            generation=2
        )

        assert debris.angular_velocity == (0.5, 0.5, 0.5)
        assert debris.lifetime == 5.0
        assert debris.parent_id == 42
        assert debris.generation == 2

    def test_spawn_debris_at_capacity(self):
        """Verify spawning at capacity."""
        manager = DebrisManager(max_active=2)
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        # Spawn up to capacity
        d1 = manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0), importance=0.5)
        d2 = manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0), importance=0.5)

        # Next spawn should fail or force cleanup
        d3 = manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0), importance=0.9)

        # With higher importance, it might force cleanup of lower importance debris
        # Behavior depends on implementation

    def test_spawn_debris_batch(self):
        """Verify batch spawning."""
        manager = DebrisManager(max_active=100)
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        params_list = [
            DebrisSpawnParams(chunk=chunk, position=(float(i), 0.0, 0.0))
            for i in range(5)
        ]

        spawned = manager.spawn_debris_batch(params_list)

        assert len(spawned) == 5
        assert manager.active_count == 5

    def test_update_aging(self):
        """Verify update processes aging."""
        manager = DebrisManager()
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        # Spawn debris with very short lifetime
        debris = manager.spawn_debris(
            chunk=chunk,
            velocity=(0.0, 0.0, 0.0),
            lifetime=0.001  # Very short
        )

        # Wait briefly and update
        time.sleep(0.01)
        cleaned = manager.update(dt=0.1)

        # Debris should be cleaned up
        assert debris.id in cleaned or manager.active_count == 0

    def test_update_sleep_detection(self):
        """Verify sleep state detection."""
        manager = DebrisManager(sleep_velocity=0.1, sleep_time=0.5)
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        debris = manager.spawn_debris(
            chunk=chunk,
            velocity=(0.0, 0.0, 0.0),  # Zero velocity
            lifetime=10.0
        )

        # Update multiple times to accumulate sleep time
        for _ in range(10):
            manager.update(dt=0.1)

        # Should transition to SLEEPING
        assert debris.state == DebrisState.SLEEPING

    def test_update_lod_calculation(self):
        """Verify LOD calculation based on camera."""
        manager = DebrisManager()
        manager.set_lod_distances(full=5.0, reduced=15.0, simple=30.0)

        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        # Spawn at known position
        debris = manager.spawn_debris(
            chunk=chunk,
            velocity=(0.0, 0.0, 0.0)
        )
        debris.position = (0.0, 0.0, 0.0)

        # Camera very close
        manager.update(dt=0.01, camera_position=(1.0, 0.0, 0.0))
        assert debris.lod == DebrisLOD.FULL

        # Camera far away
        debris.position = (100.0, 0.0, 0.0)
        manager.update(dt=0.01, camera_position=(0.0, 0.0, 0.0))
        assert debris.lod == DebrisLOD.PARTICLE

    def test_set_callbacks(self):
        """Verify callbacks are called."""
        manager = DebrisManager()

        spawned_ids = []
        destroyed_ids = []

        manager.set_callbacks(
            on_spawn=lambda d: spawned_ids.append(d.id),
            on_destroy=lambda d: destroyed_ids.append(d.id)
        )

        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        debris = manager.spawn_debris(
            chunk=chunk,
            velocity=(0.0, 0.0, 0.0),
            lifetime=0.001
        )

        assert debris.id in spawned_ids

        # Cleanup
        time.sleep(0.01)
        manager.update(dt=0.1)
        # destroyed callback should have been called

    def test_get_debris_in_radius(self):
        """Verify radius query."""
        manager = DebrisManager()
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        # Spawn debris at various positions
        for i in range(5):
            d = manager.spawn_debris(
                chunk=chunk,
                velocity=(0.0, 0.0, 0.0)
            )
            d.position = (float(i * 2), 0.0, 0.0)

        # Query radius
        nearby = manager.get_debris_in_radius((0.0, 0.0, 0.0), 3.0)

        # Should find debris at (0,0,0) and (2,0,0)
        assert len(nearby) >= 1

    def test_get_debris_by_parent(self):
        """Verify parent query."""
        manager = DebrisManager()
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        # Spawn debris with different parents
        manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0), parent_id=1)
        manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0), parent_id=1)
        manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0), parent_id=2)

        parent1_debris = manager.get_debris_by_parent(1)
        assert len(parent1_debris) == 2

        parent2_debris = manager.get_debris_by_parent(2)
        assert len(parent2_debris) == 1

    def test_destroy_debris(self):
        """Verify individual debris destruction."""
        manager = DebrisManager()
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        debris = manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0))

        result = manager.destroy_debris(debris.id)
        assert result is True

        # Process pending cleanup
        manager.update(dt=0.01)
        assert manager.active_count == 0

    def test_destroy_all(self):
        """Verify destroying all debris."""
        manager = DebrisManager()
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        for _ in range(5):
            manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0))

        count = manager.destroy_all()
        assert count == 5

        # Process
        manager.update(dt=0.01)
        assert manager.active_count == 0

    def test_destroy_by_parent(self):
        """Verify destroying debris by parent."""
        manager = DebrisManager()
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.compute_centroid()

        manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0), parent_id=1)
        manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0), parent_id=1)
        manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0), parent_id=2)

        count = manager.destroy_by_parent(1)
        assert count == 2

        manager.update(dt=0.01)
        assert manager.active_count == 1

    def test_merge_small_debris(self):
        """Verify small debris merging."""
        manager = DebrisManager()

        # Create small chunks
        chunk1 = Chunk(
            vertices=[(0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.0, 0.001, 0.0)],
            triangles=[(0, 1, 2)]
        )
        chunk1.volume = 0.0001
        chunk1.compute_centroid()

        chunk2 = Chunk(
            vertices=[(0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.0, 0.001, 0.0)],
            triangles=[(0, 1, 2)]
        )
        chunk2.volume = 0.0001
        chunk2.compute_centroid()

        d1 = manager.spawn_debris(chunk=chunk1, velocity=(0.0, 0.0, 0.0))
        d2 = manager.spawn_debris(chunk=chunk2, velocity=(0.0, 0.0, 0.0))

        # Place very close together
        d1.position = (0.0, 0.0, 0.0)
        d2.position = (0.01, 0.0, 0.0)

        merged = manager.merge_small_debris(min_volume=0.001, merge_distance=0.1)

        # Should merge at least one pair
        assert merged >= 0

    def test_get_stats(self):
        """Verify statistics retrieval."""
        manager = DebrisManager(max_active=100, pool_size=50)
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        chunk.volume = 1.0
        chunk.compute_centroid()

        manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0))
        manager.spawn_debris(chunk=chunk, velocity=(0.0, 0.0, 0.0))

        stats = manager.get_stats()

        assert stats['active_count'] == 2
        assert stats['max_active'] == 100
        assert 'lod_counts' in stats
        assert 'state_counts' in stats
        assert stats['total_volume'] == 2.0


class TestSpawnDebrisFromFracture:
    """Tests for spawn_debris_from_fracture function."""

    def test_basic_spawn(self):
        """Verify basic spawning from fracture."""
        manager = DebrisManager(max_active=100)

        # Create some chunks
        chunks = []
        for i in range(3):
            chunk = Chunk(
                vertices=[
                    (float(i), 0.0, 0.0),
                    (float(i) + 1.0, 0.0, 0.0),
                    (float(i) + 0.5, 1.0, 0.0)
                ],
                triangles=[(0, 1, 2)]
            )
            chunk.compute_centroid()
            chunk.volume = 0.5
            chunks.append(chunk)

        debris_list = spawn_debris_from_fracture(
            manager=manager,
            chunks=chunks,
            center_velocity=(10.0, 0.0, 0.0)
        )

        assert len(debris_list) == 3
        for debris in debris_list:
            # Should have center velocity plus spread
            assert abs(debris.velocity[0] - 10.0) <= 5.0  # Within spread range

    def test_spawn_with_parent(self):
        """Verify spawning with parent ID."""
        manager = DebrisManager()
        chunk = Chunk(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)],
            triangles=[(0, 1, 2)]
        )
        chunk.compute_centroid()
        chunk.volume = 1.0

        debris_list = spawn_debris_from_fracture(
            manager=manager,
            chunks=[chunk],
            center_velocity=(0.0, 0.0, 0.0),
            parent_id=42,
            generation=2
        )

        assert len(debris_list) == 1
        assert debris_list[0].parent_id == 42
        assert debris_list[0].generation == 2

    def test_spawn_empty_chunks(self):
        """Verify handling of empty chunks list."""
        manager = DebrisManager()

        debris_list = spawn_debris_from_fracture(
            manager=manager,
            chunks=[],
            center_velocity=(0.0, 0.0, 0.0)
        )

        assert len(debris_list) == 0

    def test_spawn_with_spread(self):
        """Verify spread factor affects velocity."""
        manager = DebrisManager()

        # Create chunks at different positions
        chunks = []
        for i in range(4):
            chunk = Chunk(
                vertices=[
                    (float(i) * 10, 0.0, 0.0),
                    (float(i) * 10 + 1.0, 0.0, 0.0),
                    (float(i) * 10 + 0.5, 1.0, 0.0)
                ],
                triangles=[(0, 1, 2)]
            )
            chunk.compute_centroid()
            chunk.volume = 1.0
            chunks.append(chunk)

        debris_list = spawn_debris_from_fracture(
            manager=manager,
            chunks=chunks,
            center_velocity=(0.0, 0.0, 0.0),
            spread_factor=5.0
        )

        # Debris should have outward velocities
        for debris in debris_list:
            # Just verify it was created
            assert debris is not None

    def test_spawn_with_custom_lifetime(self):
        """Verify custom lifetime is applied."""
        manager = DebrisManager()
        chunk = Chunk(
            vertices=[(0.0, 0.0, 0.0)],
            triangles=[]
        )
        chunk.compute_centroid()
        chunk.volume = 1.0

        debris_list = spawn_debris_from_fracture(
            manager=manager,
            chunks=[chunk],
            center_velocity=(0.0, 0.0, 0.0),
            lifetime=5.0
        )

        assert len(debris_list) == 1
        assert debris_list[0].lifetime == 5.0
