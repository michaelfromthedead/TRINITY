"""
Island Detection and Management for Constraint Solver.

Islands are groups of bodies connected by constraints. Bodies within
an island must be solved together, but separate islands can be solved
independently and potentially in parallel.

Key features:
- Union-Find algorithm for connected components
- Sleeping island detection
- Parallel island solving support
- Dynamic island merging and splitting
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Callable
from enum import Enum, auto
import heapq

from .constraint_solver import Constraint, RigidBody


class IslandState(Enum):
    """State of an island."""
    ACTIVE = auto()      # Island has active bodies
    SLEEPING = auto()    # All bodies are sleeping
    STATIC = auto()      # All bodies are static


@dataclass
class Island:
    """
    Represents an island of connected bodies and constraints.

    Attributes:
        id: Unique island identifier.
        bodies: List of bodies in the island.
        constraints: List of constraints in the island.
        state: Current state of the island.
        sleep_timer: Time since all bodies became slow.
        last_update_frame: Frame when island was last updated.
    """
    id: int
    bodies: List[RigidBody] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    state: IslandState = IslandState.ACTIVE
    sleep_timer: float = 0.0
    last_update_frame: int = 0

    def is_sleeping(self) -> bool:
        """Check if island is sleeping."""
        return self.state == IslandState.SLEEPING

    def is_static(self) -> bool:
        """Check if island contains only static bodies."""
        return self.state == IslandState.STATIC

    def is_active(self) -> bool:
        """Check if island is active."""
        return self.state == IslandState.ACTIVE

    def get_body_count(self) -> int:
        """Get number of bodies in island."""
        return len(self.bodies)

    def get_constraint_count(self) -> int:
        """Get number of constraints in island."""
        return len(self.constraints)

    def get_dynamic_body_count(self) -> int:
        """Get number of dynamic (non-static) bodies."""
        return sum(1 for body in self.bodies if not body.is_static)

    def wake_up(self) -> None:
        """Wake up the island."""
        self.state = IslandState.ACTIVE
        self.sleep_timer = 0.0
        for body in self.bodies:
            body.is_sleeping = False

    def put_to_sleep(self) -> None:
        """Put the island to sleep."""
        self.state = IslandState.SLEEPING
        for body in self.bodies:
            body.is_sleeping = True
            body.velocity = type(body.velocity).zero() if hasattr(type(body.velocity), 'zero') else type(body.velocity)()
            body.angular_velocity = type(body.angular_velocity).zero() if hasattr(type(body.angular_velocity), 'zero') else type(body.angular_velocity)()


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure.

    Used for efficiently finding connected components.
    """

    def __init__(self):
        """Initialize empty Union-Find structure."""
        self._parent: Dict[int, int] = {}
        self._rank: Dict[int, int] = {}

    def make_set(self, x: int) -> None:
        """Create a new set containing only x."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: int) -> int:
        """
        Find the root of the set containing x.

        Uses path compression for efficiency.

        Args:
            x: Element to find.

        Returns:
            Root of the set.
        """
        if x not in self._parent:
            self.make_set(x)
            return x

        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # Path compression
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        """
        Unite the sets containing x and y.

        Uses union by rank for efficiency.

        Args:
            x: First element.
            y: Second element.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self._rank[root_x] < self._rank[root_y]:
            self._parent[root_x] = root_y
        elif self._rank[root_x] > self._rank[root_y]:
            self._parent[root_y] = root_x
        else:
            self._parent[root_y] = root_x
            self._rank[root_x] += 1

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def clear(self) -> None:
        """Clear all sets."""
        self._parent.clear()
        self._rank.clear()


class IslandManager:
    """
    Manages islands for physics simulation.

    Handles island detection, sleeping, and provides support
    for parallel island solving.

    Attributes:
        islands: Dictionary of islands by ID.
        sleep_velocity_threshold: Velocity below which bodies may sleep.
        sleep_angular_threshold: Angular velocity below which bodies may sleep.
        sleep_time_threshold: Time body must be slow before sleeping.
    """

    def __init__(
        self,
        sleep_velocity_threshold: float = 0.05,
        sleep_angular_threshold: float = 0.05,
        sleep_time_threshold: float = 0.5
    ):
        """
        Initialize island manager.

        Args:
            sleep_velocity_threshold: Linear velocity threshold for sleeping.
            sleep_angular_threshold: Angular velocity threshold for sleeping.
            sleep_time_threshold: Time before bodies can sleep.
        """
        self.islands: Dict[int, Island] = {}
        self.sleep_velocity_threshold = sleep_velocity_threshold
        self.sleep_angular_threshold = sleep_angular_threshold
        self.sleep_time_threshold = sleep_time_threshold

        self._union_find = UnionFind()
        self._body_to_island: Dict[int, int] = {}
        self._next_island_id: int = 0
        self._current_frame: int = 0

    def build_islands(
        self,
        bodies: List[RigidBody],
        constraints: List[Constraint]
    ) -> List[Island]:
        """
        Build islands from bodies and constraints.

        Uses Union-Find to identify connected components.

        Args:
            bodies: List of all bodies.
            constraints: List of all constraints.

        Returns:
            List of islands.
        """
        self._union_find.clear()
        self._body_to_island.clear()
        self.islands.clear()

        # Create sets for all non-static bodies
        for body in bodies:
            if not body.is_static:
                self._union_find.make_set(body.id)

        # Union bodies connected by constraints
        for constraint in constraints:
            body_a = constraint.body_a
            body_b = constraint.body_b

            # Skip if both bodies are static
            if body_a.is_static and (body_b is None or body_b.is_static):
                continue

            if body_b is not None:
                if not body_a.is_static and not body_b.is_static:
                    self._union_find.union(body_a.id, body_b.id)

        # Group bodies by root
        root_to_bodies: Dict[int, List[RigidBody]] = {}
        root_to_constraints: Dict[int, List[Constraint]] = {}

        for body in bodies:
            if body.is_static:
                continue

            root = self._union_find.find(body.id)
            if root not in root_to_bodies:
                root_to_bodies[root] = []
            root_to_bodies[root].append(body)

        # Assign constraints to islands
        for constraint in constraints:
            body_a = constraint.body_a
            body_b = constraint.body_b

            # Find the root for this constraint
            root = None
            if not body_a.is_static:
                root = self._union_find.find(body_a.id)
            elif body_b is not None and not body_b.is_static:
                root = self._union_find.find(body_b.id)

            if root is not None:
                if root not in root_to_constraints:
                    root_to_constraints[root] = []
                root_to_constraints[root].append(constraint)

        # Create islands
        result = []
        for root, island_bodies in root_to_bodies.items():
            island = Island(
                id=self._next_island_id,
                bodies=island_bodies,
                constraints=root_to_constraints.get(root, []),
                last_update_frame=self._current_frame
            )
            self._next_island_id += 1

            # Assign island ID to bodies
            for body in island_bodies:
                body.island_id = island.id
                self._body_to_island[body.id] = island.id

            self.islands[island.id] = island
            result.append(island)

        return result

    def update_sleeping(self, dt: float) -> Tuple[List[Island], List[Island]]:
        """
        Update island sleeping states.

        Args:
            dt: Time step.

        Returns:
            Tuple of (newly sleeping islands, newly awakened islands).
        """
        newly_sleeping = []
        newly_awakened = []

        for island in self.islands.values():
            if island.state == IslandState.STATIC:
                continue

            # Check if any body is moving fast
            is_active = False
            for body in island.bodies:
                if body.is_static:
                    continue

                vel_sq = body.velocity.length_squared()
                ang_vel_sq = body.angular_velocity.length_squared()

                if (vel_sq > self.sleep_velocity_threshold * self.sleep_velocity_threshold or
                    ang_vel_sq > self.sleep_angular_threshold * self.sleep_angular_threshold):
                    is_active = True
                    break

            if is_active:
                if island.state == IslandState.SLEEPING:
                    island.wake_up()
                    newly_awakened.append(island)
                else:
                    island.sleep_timer = 0.0
            else:
                island.sleep_timer += dt
                if (island.state == IslandState.ACTIVE and
                    island.sleep_timer >= self.sleep_time_threshold):
                    island.put_to_sleep()
                    newly_sleeping.append(island)

        return newly_sleeping, newly_awakened

    def get_active_islands(self) -> List[Island]:
        """Get list of active (non-sleeping) islands."""
        return [island for island in self.islands.values() if island.is_active()]

    def get_sleeping_islands(self) -> List[Island]:
        """Get list of sleeping islands."""
        return [island for island in self.islands.values() if island.is_sleeping()]

    def wake_island(self, island_id: int) -> None:
        """Wake up a specific island."""
        if island_id in self.islands:
            self.islands[island_id].wake_up()

    def wake_body(self, body_id: int) -> None:
        """Wake up the island containing a specific body."""
        if body_id in self._body_to_island:
            island_id = self._body_to_island[body_id]
            self.wake_island(island_id)

    def get_island_for_body(self, body_id: int) -> Optional[Island]:
        """Get the island containing a specific body."""
        if body_id in self._body_to_island:
            island_id = self._body_to_island[body_id]
            return self.islands.get(island_id)
        return None

    def merge_islands(self, island_a: Island, island_b: Island) -> Island:
        """
        Merge two islands into one.

        Args:
            island_a: First island.
            island_b: Second island.

        Returns:
            Merged island.
        """
        # Create new merged island
        merged = Island(
            id=self._next_island_id,
            bodies=island_a.bodies + island_b.bodies,
            constraints=island_a.constraints + island_b.constraints,
            state=IslandState.ACTIVE,  # Merged island is active
            last_update_frame=self._current_frame
        )
        self._next_island_id += 1

        # Update body references
        for body in merged.bodies:
            body.island_id = merged.id
            self._body_to_island[body.id] = merged.id

        # Remove old islands
        if island_a.id in self.islands:
            del self.islands[island_a.id]
        if island_b.id in self.islands:
            del self.islands[island_b.id]

        # Add merged island
        self.islands[merged.id] = merged

        return merged

    def split_island(self, island: Island) -> List[Island]:
        """
        Split an island into connected components.

        Used when a constraint is removed and may disconnect bodies.

        Args:
            island: Island to split.

        Returns:
            List of resulting islands.
        """
        # Rebuild connectivity with current constraints
        new_islands = self.build_islands(island.bodies, island.constraints)

        # Remove original island if it was replaced
        if island.id in self.islands:
            del self.islands[island.id]

        return new_islands

    def get_parallel_groups(self, max_groups: int = 4) -> List[List[Island]]:
        """
        Group islands for parallel solving.

        Groups islands that don't share bodies for parallel execution.

        Args:
            max_groups: Maximum number of parallel groups.

        Returns:
            List of island groups that can be solved in parallel.
        """
        active_islands = self.get_active_islands()

        if not active_islands:
            return []

        # Sort islands by constraint count (larger first)
        sorted_islands = sorted(
            active_islands,
            key=lambda i: i.get_constraint_count(),
            reverse=True
        )

        # Greedy assignment to groups
        groups: List[List[Island]] = [[] for _ in range(max_groups)]
        group_loads = [0] * max_groups

        for island in sorted_islands:
            # Assign to group with lowest load
            min_load_idx = group_loads.index(min(group_loads))
            groups[min_load_idx].append(island)
            group_loads[min_load_idx] += island.get_constraint_count()

        # Filter out empty groups
        return [g for g in groups if g]

    def advance_frame(self) -> None:
        """Advance to next frame."""
        self._current_frame += 1

    def get_statistics(self) -> Dict[str, int]:
        """
        Get island statistics.

        Returns:
            Dictionary with statistics.
        """
        active = sum(1 for i in self.islands.values() if i.is_active())
        sleeping = sum(1 for i in self.islands.values() if i.is_sleeping())
        static = sum(1 for i in self.islands.values() if i.is_static())

        total_bodies = sum(i.get_body_count() for i in self.islands.values())
        total_constraints = sum(i.get_constraint_count() for i in self.islands.values())

        return {
            "total_islands": len(self.islands),
            "active_islands": active,
            "sleeping_islands": sleeping,
            "static_islands": static,
            "total_bodies": total_bodies,
            "total_constraints": total_constraints,
            "current_frame": self._current_frame,
        }

    def clear(self) -> None:
        """Clear all islands."""
        self.islands.clear()
        self._body_to_island.clear()
        self._union_find.clear()


class ParallelIslandSolver:
    """
    Parallel island solver using island independence.

    Provides infrastructure for solving independent islands
    in parallel using multiple threads or workers.
    """

    def __init__(
        self,
        island_manager: IslandManager,
        num_workers: int = 4
    ):
        """
        Initialize parallel solver.

        Args:
            island_manager: Island manager to use.
            num_workers: Number of parallel workers.
        """
        self.island_manager = island_manager
        self.num_workers = num_workers
        self._solve_callback: Optional[Callable[[Island, float], None]] = None

    def set_solve_callback(
        self,
        callback: Callable[[Island, float], None]
    ) -> None:
        """
        Set callback function for solving an island.

        Args:
            callback: Function that takes (island, dt) and solves it.
        """
        self._solve_callback = callback

    def solve_parallel(self, dt: float) -> None:
        """
        Solve all active islands in parallel.

        Args:
            dt: Time step.
        """
        if self._solve_callback is None:
            return

        # Get parallel groups
        groups = self.island_manager.get_parallel_groups(self.num_workers)

        # In a real implementation, this would use ThreadPoolExecutor
        # or similar. Here we solve sequentially as a placeholder.
        for group in groups:
            for island in group:
                self._solve_callback(island, dt)

    def solve_sequential(self, dt: float) -> None:
        """
        Solve all active islands sequentially.

        Args:
            dt: Time step.
        """
        if self._solve_callback is None:
            return

        for island in self.island_manager.get_active_islands():
            self._solve_callback(island, dt)


@dataclass
class IslandEvent:
    """Event representing a change in island state."""
    event_type: str  # "created", "merged", "split", "sleeping", "awakened"
    island_ids: List[int]
    timestamp: int = 0


class IslandEventListener:
    """Interface for receiving island events."""

    def on_island_created(self, island: Island) -> None:
        """Called when a new island is created."""
        pass

    def on_island_merged(self, source_ids: List[int], result: Island) -> None:
        """Called when islands are merged."""
        pass

    def on_island_split(self, source_id: int, results: List[Island]) -> None:
        """Called when an island is split."""
        pass

    def on_island_sleeping(self, island: Island) -> None:
        """Called when an island goes to sleep."""
        pass

    def on_island_awakened(self, island: Island) -> None:
        """Called when an island wakes up."""
        pass
