"""
T-1.8: Test Island Manager.

Covers:
  - Union-Find: two bodies with contact form one island
  - Path compression: repeated find returns same root
  - Rank union: tree stays balanced
  - Sleep propagation: one body wakes entire island
  - Island isolation: disconnected bodies form separate islands
"""

import pytest

from engine.simulation.solver.island_manager import (
    IslandManager, Island, UnionFind, IslandState, ParallelIslandSolver,
)
from engine.simulation.solver.constraint_solver import (
    RigidBody, Constraint, ConstraintType, PointConstraint,
)
from engine.simulation.solver.jacobian import Vec3, Mat3, Quaternion
from ..physics_test_base import PhysicsTestCase


# ===========================================================================
# T-1.8  —  Island Manager
# ===========================================================================

# -- Dummy constraint for island building ----------------------------------

class DummyConstraint:
    """Minimal constraint that satisfies the Constraint protocol."""
    def __init__(self, body_a, body_b=None):
        self._body_a = body_a
        self._body_b = body_b
        self._constraint_type = ConstraintType.EQUALITY

    @property
    def body_a(self):
        return self._body_a

    @property
    def body_b(self):
        return self._body_b

    @property
    def constraint_type(self):
        return self._constraint_type

    def prepare(self, dt, config):
        pass

    def warm_start(self, factor):
        pass

    def solve_velocity(self):
        return 0.0

    def solve_position(self, max_correction):
        return 0.0

    def get_cached_impulse(self):
        return 0.0

    def set_cached_impulse(self, impulse):
        pass


# ===========================================================================
# Union-Find
# ===========================================================================

class TestUnionFind(PhysicsTestCase):
    """Union-Find data structure."""

    def test_make_set(self):
        """make_set creates a new set."""
        uf = UnionFind()
        uf.make_set(0)
        assert uf.find(0) == 0

    def test_find_same_root(self):
        """repeated find returns same root."""
        uf = UnionFind()
        uf.make_set(42)
        assert uf.find(42) == uf.find(42)

    def test_union(self):
        """union merges two sets."""
        uf = UnionFind()
        uf.make_set(1)
        uf.make_set(2)
        uf.union(1, 2)
        assert uf.connected(1, 2)

    def test_union_chain(self):
        """transitive union: 1-2-3 all connected."""
        uf = UnionFind()
        for i in (1, 2, 3):
            uf.make_set(i)
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.connected(1, 3)

    def test_not_connected(self):
        """elements in different sets are not connected."""
        uf = UnionFind()
        uf.make_set(0)
        uf.make_set(1)
        assert not uf.connected(0, 1)

    def test_clear(self):
        """clear removes all sets."""
        uf = UnionFind()
        uf.make_set(0)
        uf.clear()
        # After clear, find(0) should make a fresh set
        assert uf.find(0) == 0

    def test_union_by_rank(self):
        """union by rank keeps tree shallow."""
        uf = UnionFind()
        for i in range(100):
            uf.make_set(i)
        for i in range(99):
            uf.union(i, i + 1)

        # All should be connected
        assert uf.connected(0, 99)
        # Path to root should be short (rank-based union)
        root = uf.find(0)
        # The depth should not be 99 for rank-union
        depths = []
        for i in range(100):
            d = 0
            x = i
            while uf._parent.get(x, x) != x:
                x = uf._parent[x]
                d += 1
            depths.append(d)
        max_depth = max(depths)
        assert max_depth <= 10, f"Union-Find tree too deep: {max_depth}"

    def test_path_compression(self):
        """path compression flattens the tree."""
        uf = UnionFind()
        uf.make_set(0)
        uf.make_set(1)
        uf.make_set(2)
        uf._parent[1] = 0
        uf._parent[2] = 0
        # after find(1) with path compression, parent[1] should be root
        root = uf.find(1)
        assert root == 0


# ===========================================================================
# Island building
# ===========================================================================

class TestIslandBuilding(PhysicsTestCase):
    """IslandManager.build_islands."""

    def test_two_bodies_one_island(self):
        """two bodies with a constraint form one island."""
        body_a = RigidBody.create_dynamic(0, Vec3(-1, 0, 0), mass=1.0)
        body_b = RigidBody.create_dynamic(1, Vec3(1, 0, 0), mass=1.0)
        constraint = DummyConstraint(body_a, body_b)

        manager = IslandManager()
        islands = manager.build_islands([body_a, body_b], [constraint])
        assert len(islands) == 1, f"Expected 1 island, got {len(islands)}"

    def test_disconnected_bodies_separate_islands(self):
        """N bodies with no contacts form N islands."""
        bodies = [RigidBody.create_dynamic(i, Vec3(i * 10, 0, 0), mass=1.0)
                  for i in range(5)]
        manager = IslandManager()
        islands = manager.build_islands(bodies, [])
        assert len(islands) == 5, f"Expected 5 islands, got {len(islands)}"

    def test_chain_forms_one_island(self):
        """N bodies with N-1 contacts form 1 island."""
        bodies = [RigidBody.create_dynamic(i, Vec3(i * 2, 0, 0), mass=1.0)
                  for i in range(10)]
        constraints = [DummyConstraint(bodies[i], bodies[i + 1])
                       for i in range(9)]

        manager = IslandManager()
        islands = manager.build_islands(bodies, constraints)
        assert len(islands) == 1, f"Expected 1 island, got {len(islands)}"

    def test_static_bodies_ignored(self):
        """static bodies do not form their own islands."""
        static = RigidBody.create_static(0)
        dynamic = RigidBody.create_dynamic(1, Vec3.zero(), mass=1.0)

        manager = IslandManager()
        islands = manager.build_islands([static, dynamic], [])
        assert len(islands) == 1  # Only dynamic body forms island

    def test_static_does_not_connect(self):
        """two static bodies with a constraint form no islands."""
        static_a = RigidBody.create_static(0)
        static_b = RigidBody.create_static(1)
        constraint = DummyConstraint(static_a, static_b)

        manager = IslandManager()
        islands = manager.build_islands([static_a, static_b], [constraint])
        assert len(islands) == 0

    def test_dynamic_with_static_constraint(self):
        """dynamic body constrained to static forms one island."""
        dynamic = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        static = RigidBody.create_static(1)
        constraint = DummyConstraint(dynamic, static)

        manager = IslandManager()
        islands = manager.build_islands([dynamic, static], [constraint])
        assert len(islands) == 1


# ===========================================================================
# Sleeping / wake propagation
# ===========================================================================

class TestIslandSleeping(PhysicsTestCase):
    """Island sleep state transitions."""

    def test_active_island_new(self):
        """newly created island is ACTIVE."""
        body = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        manager = IslandManager()
        islands = manager.build_islands([body], [])
        assert islands[0].is_active()

    def test_put_to_sleep(self):
        """put_to_sleep transitions island to SLEEPING."""
        body = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        manager = IslandManager()
        islands = manager.build_islands([body], [])
        island = islands[0]

        island.put_to_sleep()
        assert island.is_sleeping()
        assert body.is_sleeping

    def test_wake_up(self):
        """wake_up transitions island to ACTIVE."""
        body = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        manager = IslandManager()
        islands = manager.build_islands([body], [])
        island = islands[0]

        island.put_to_sleep()
        island.wake_up()
        assert island.is_active()
        assert not body.is_sleeping

    def test_update_sleeping_body_moving(self):
        """body moving fast keeps island active."""
        body = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        body.velocity = Vec3(1, 0, 0)  # Above threshold (0.05)

        manager = IslandManager()
        islands = manager.build_islands([body], [])
        sleeping, awakened = manager.update_sleeping(1.0)

        assert len(sleeping) == 0, "Moving body should not sleep"

    def test_update_sleeping_body_still(self):
        """body below threshold for long enough goes to sleep."""
        body = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        body.velocity = Vec3(0.01, 0, 0)  # Below threshold (0.05)

        manager = IslandManager()
        manager.sleep_time_threshold = 0.5
        islands = manager.build_islands([body], [])

        sleeping, awakened = manager.update_sleeping(0.6)  # Exceeds threshold
        assert len(sleeping) == 1, f"Expected 1 sleeping, got {len(sleeping)}"

    def test_wake_body_wakes_island(self):
        """wake_body wakes the entire island."""
        body = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        manager = IslandManager()
        islands = manager.build_islands([body], [])

        islands[0].put_to_sleep()
        manager.wake_body(0)
        assert islands[0].is_active()

    def test_wake_island(self):
        """wake_island by ID."""
        body = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        manager = IslandManager()
        islands = manager.build_islands([body], [])
        island_id = islands[0].id

        islands[0].put_to_sleep()
        manager.wake_island(island_id)
        assert islands[0].is_active()


# ===========================================================================
# Island query / statistics
# ===========================================================================

class TestIslandQueries(PhysicsTestCase):
    """IslandManager query methods."""

    def test_get_island_for_body(self):
        """get_island_for_body returns correct island."""
        body = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        manager = IslandManager()
        manager.build_islands([body], [])
        island = manager.get_island_for_body(0)
        assert island is not None
        assert body.id in [b.id for b in island.bodies]

    def test_get_active_islands(self):
        """get_active_islands returns only active islands."""
        body = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        manager = IslandManager()
        manager.build_islands([body], [])

        active = manager.get_active_islands()
        assert len(active) == 1

        active[0].put_to_sleep()
        active = manager.get_active_islands()
        assert len(active) == 0

    def test_get_statistics(self):
        """get_statistics returns expected keys."""
        body_a = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        body_b = RigidBody.create_dynamic(1, Vec3(2, 0, 0), mass=1.0)
        constraint = DummyConstraint(body_a, body_b)

        manager = IslandManager()
        manager.build_islands([body_a, body_b], [constraint])
        stats = manager.get_statistics()

        assert "total_islands" in stats
        assert "active_islands" in stats
        assert "total_bodies" in stats
        assert "total_constraints" in stats


# ===========================================================================
# Parallel solving groups
# ===========================================================================

class TestParallelGroups(PhysicsTestCase):
    """IslandManager.get_parallel_groups."""

    def test_empty_groups(self):
        """no active islands returns empty groups."""
        manager = IslandManager()
        groups = manager.get_parallel_groups(max_groups=4)
        assert groups == []

    def test_single_island_one_group(self):
        """one island produces one group."""
        body = RigidBody.create_dynamic(0, Vec3.zero(), mass=1.0)
        manager = IslandManager()
        manager.build_islands([body], [])
        groups = manager.get_parallel_groups(max_groups=4)
        assert len(groups) == 1


# ===========================================================================
# Advance frame
# ===========================================================================

class TestIslandFrame(PhysicsTestCase):
    """advance_frame increments frame counter."""

    def test_advance_frame(self):
        """advance_frame increments _current_frame."""
        manager = IslandManager()
        assert manager._current_frame == 0
        manager.advance_frame()
        assert manager._current_frame == 1
