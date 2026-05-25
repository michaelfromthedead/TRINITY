"""
Sleep Manager Module

Handles sleep state management for physics bodies to optimize simulation.
Bodies that have been at rest for a period of time are put to sleep,
reducing computational cost.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, TYPE_CHECKING
from enum import Enum, auto
import math

from .config import (
    SLEEP_THRESHOLD_LINEAR,
    SLEEP_THRESHOLD_ANGULAR,
    SLEEP_TIME_THRESHOLD,
    MIN_ISLAND_SIZE,
    MAX_ISLAND_SIZE,
)

if TYPE_CHECKING:
    from .rigid_body import RigidBody


# Type aliases
Vector3 = Tuple[float, float, float]


def _vector_length_sq(v: Vector3) -> float:
    """Get squared length of a vector."""
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def _vector_length(v: Vector3) -> float:
    """Get length of a vector."""
    return math.sqrt(_vector_length_sq(v))


class IslandState(Enum):
    """State of a physics island."""
    ACTIVE = auto()      # At least one body is awake
    SLEEPING = auto()    # All bodies are asleep
    WAKING = auto()      # Transitioning from sleep to active


@dataclass
class Island:
    """
    A group of connected bodies that can sleep together.

    Bodies connected through contacts or joints form islands.
    An island can only sleep when all its bodies are at rest.
    """
    id: int
    bodies: Set[str] = field(default_factory=set)  # Body IDs
    state: IslandState = IslandState.ACTIVE
    sleep_timer: float = 0.0
    contacts: List[Tuple[str, str]] = field(default_factory=list)  # Body ID pairs
    joints: List[str] = field(default_factory=list)  # Joint IDs

    @property
    def body_count(self) -> int:
        """Get number of bodies in island."""
        return len(self.bodies)

    @property
    def is_sleeping(self) -> bool:
        """Check if island is sleeping."""
        return self.state == IslandState.SLEEPING

    @property
    def is_active(self) -> bool:
        """Check if island is active."""
        return self.state == IslandState.ACTIVE


class SleepManager:
    """
    Manages sleep states for physics bodies.

    Tracks body velocities over time and puts bodies to sleep when they
    have been at rest for a sufficient period. Uses island-based sleeping
    to ensure connected bodies sleep/wake together.
    """

    def __init__(
        self,
        linear_threshold: float = SLEEP_THRESHOLD_LINEAR,
        angular_threshold: float = SLEEP_THRESHOLD_ANGULAR,
        time_threshold: float = SLEEP_TIME_THRESHOLD,
    ):
        """
        Initialize sleep manager.

        Args:
            linear_threshold: Linear velocity below which body may sleep (m/s)
            angular_threshold: Angular velocity below which body may sleep (rad/s)
            time_threshold: Time below thresholds before sleeping (s)
        """
        self._linear_threshold = linear_threshold
        self._angular_threshold = angular_threshold
        self._time_threshold = time_threshold

        # Body sleep timers
        self._sleep_timers: Dict[str, float] = {}

        # Island management
        self._islands: Dict[int, Island] = {}
        self._body_to_island: Dict[str, int] = {}
        self._next_island_id = 0

        # Bodies registered with the manager
        self._bodies: Dict[str, 'RigidBody'] = {}

        # Statistics
        self._sleeping_count = 0
        self._awake_count = 0

    @property
    def linear_threshold(self) -> float:
        """Get linear velocity sleep threshold."""
        return self._linear_threshold

    @linear_threshold.setter
    def linear_threshold(self, value: float) -> None:
        """Set linear velocity sleep threshold."""
        self._linear_threshold = max(0.0, value)

    @property
    def angular_threshold(self) -> float:
        """Get angular velocity sleep threshold."""
        return self._angular_threshold

    @angular_threshold.setter
    def angular_threshold(self, value: float) -> None:
        """Set angular velocity sleep threshold."""
        self._angular_threshold = max(0.0, value)

    @property
    def time_threshold(self) -> float:
        """Get time threshold for sleeping."""
        return self._time_threshold

    @time_threshold.setter
    def time_threshold(self, value: float) -> None:
        """Set time threshold for sleeping."""
        self._time_threshold = max(0.0, value)

    @property
    def sleeping_count(self) -> int:
        """Get count of sleeping bodies."""
        return self._sleeping_count

    @property
    def awake_count(self) -> int:
        """Get count of awake bodies."""
        return self._awake_count

    @property
    def island_count(self) -> int:
        """Get number of islands."""
        return len(self._islands)

    # =========================================================================
    # Body Management
    # =========================================================================

    def register_body(self, body: 'RigidBody') -> None:
        """
        Register a body with the sleep manager.

        Args:
            body: RigidBody to register
        """
        body_id = body.id
        self._bodies[body_id] = body
        self._sleep_timers[body_id] = 0.0

        # Create single-body island
        island_id = self._create_island()
        self._islands[island_id].bodies.add(body_id)
        self._body_to_island[body_id] = island_id

        if body.is_sleeping:
            self._sleeping_count += 1
        else:
            self._awake_count += 1

    def unregister_body(self, body: 'RigidBody') -> None:
        """
        Unregister a body from the sleep manager.

        Args:
            body: RigidBody to unregister
        """
        body_id = body.id

        if body_id not in self._bodies:
            return

        # Update counts
        if body.is_sleeping:
            self._sleeping_count -= 1
        else:
            self._awake_count -= 1

        # Remove from island
        if body_id in self._body_to_island:
            island_id = self._body_to_island[body_id]
            if island_id in self._islands:
                self._islands[island_id].bodies.discard(body_id)
                # Remove empty islands
                if not self._islands[island_id].bodies:
                    del self._islands[island_id]
            del self._body_to_island[body_id]

        # Remove from tracking
        del self._bodies[body_id]
        if body_id in self._sleep_timers:
            del self._sleep_timers[body_id]

    def _create_island(self) -> int:
        """Create a new island and return its ID."""
        island_id = self._next_island_id
        self._next_island_id += 1
        self._islands[island_id] = Island(id=island_id)
        return island_id

    # =========================================================================
    # Sleep State Queries
    # =========================================================================

    def is_sleeping(self, body: 'RigidBody') -> bool:
        """
        Check if a body is sleeping.

        Args:
            body: Body to check

        Returns:
            True if body is sleeping
        """
        return body.is_sleeping

    def can_sleep(self, body: 'RigidBody') -> bool:
        """
        Check if a body is allowed to sleep.

        Args:
            body: Body to check

        Returns:
            True if body can sleep
        """
        from .rigid_body import BodyType

        # Static bodies don't sleep (they're effectively always "asleep")
        if body.body_type == BodyType.STATIC:
            return False

        # Kinematic bodies don't sleep
        if body.body_type == BodyType.KINEMATIC:
            return False

        # Check flags
        return body.flags.can_sleep

    def is_below_threshold(self, body: 'RigidBody') -> bool:
        """
        Check if body velocity is below sleep threshold.

        Args:
            body: Body to check

        Returns:
            True if velocity is below threshold
        """
        linear_speed_sq = _vector_length_sq(body.linear_velocity)
        angular_speed_sq = _vector_length_sq(body.angular_velocity)

        return (
            linear_speed_sq < self._linear_threshold * self._linear_threshold and
            angular_speed_sq < self._angular_threshold * self._angular_threshold
        )

    # =========================================================================
    # Sleep State Changes
    # =========================================================================

    def wake_up(self, body: 'RigidBody') -> None:
        """
        Wake up a body and all bodies in its island.

        Args:
            body: Body to wake
        """
        body_id = body.id

        if body_id not in self._body_to_island:
            # Body not registered, wake it directly
            if body.is_sleeping:
                body.wake_up()
            return

        # Wake entire island
        island_id = self._body_to_island[body_id]
        self._wake_island(island_id)

    def put_to_sleep(self, body: 'RigidBody') -> None:
        """
        Put a body to sleep if allowed.

        Args:
            body: Body to put to sleep
        """
        if not self.can_sleep(body):
            return

        body_id = body.id

        if body_id not in self._body_to_island:
            # Body not registered, sleep it directly
            if not body.is_sleeping:
                body.put_to_sleep()
            return

        # Check if entire island can sleep
        island_id = self._body_to_island[body_id]
        island = self._islands.get(island_id)

        if island and self._can_island_sleep(island):
            self._sleep_island(island_id)

    def _wake_island(self, island_id: int) -> None:
        """Wake all bodies in an island."""
        island = self._islands.get(island_id)
        if not island:
            return

        # Always wake all sleeping bodies in the island, even if island state is ACTIVE
        # This handles the case where body was put to sleep individually
        island.state = IslandState.ACTIVE
        island.sleep_timer = 0.0

        for body_id in island.bodies:
            body = self._bodies.get(body_id)
            if body and body.is_sleeping:
                body.wake_up()
                self._sleeping_count -= 1
                self._awake_count += 1
                self._sleep_timers[body_id] = 0.0

    def _sleep_island(self, island_id: int) -> None:
        """Put all bodies in an island to sleep."""
        island = self._islands.get(island_id)
        if not island:
            return

        if island.state == IslandState.SLEEPING:
            return

        island.state = IslandState.SLEEPING

        for body_id in island.bodies:
            body = self._bodies.get(body_id)
            if body and not body.is_sleeping and self.can_sleep(body):
                body.put_to_sleep()
                self._sleeping_count += 1
                self._awake_count -= 1

    def _can_island_sleep(self, island: Island) -> bool:
        """Check if all bodies in an island can sleep."""
        for body_id in island.bodies:
            body = self._bodies.get(body_id)
            if body:
                if not self.can_sleep(body):
                    return False
                if not self.is_below_threshold(body):
                    return False

        return True

    # =========================================================================
    # Island Management
    # =========================================================================

    def merge_islands(self, body_a: 'RigidBody', body_b: 'RigidBody') -> None:
        """
        Merge the islands containing two bodies.

        Called when bodies become connected (contact or joint).

        Args:
            body_a: First body
            body_b: Second body
        """
        id_a = body_a.id
        id_b = body_b.id

        if id_a not in self._body_to_island or id_b not in self._body_to_island:
            return

        island_a = self._body_to_island[id_a]
        island_b = self._body_to_island[id_b]

        if island_a == island_b:
            return  # Already in same island

        # Merge smaller into larger
        a_size = len(self._islands[island_a].bodies)
        b_size = len(self._islands[island_b].bodies)

        if a_size < b_size:
            island_a, island_b = island_b, island_a

        # Merge island_b into island_a
        for body_id in self._islands[island_b].bodies:
            self._islands[island_a].bodies.add(body_id)
            self._body_to_island[body_id] = island_a

        # If either island was active, merged island is active
        if self._islands[island_b].state == IslandState.ACTIVE:
            self._wake_island(island_a)

        # Remove merged island
        del self._islands[island_b]

    def split_islands_if_needed(self) -> None:
        """
        Check if islands should be split due to disconnected bodies.

        This is called after contacts are updated.
        """
        # For simplicity, rebuild islands from scratch
        # A more efficient approach would track connectivity incrementally
        pass  # Implemented in rebuild_islands

    def rebuild_islands(self, contacts: List[Tuple[str, str]]) -> None:
        """
        Rebuild islands from contact graph.

        Args:
            contacts: List of (body_id_a, body_id_b) contact pairs
        """
        # Build connectivity graph using union-find
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Initialize each body as its own component
        for body_id in self._bodies:
            parent[body_id] = body_id

        # Union connected bodies
        for id_a, id_b in contacts:
            if id_a in self._bodies and id_b in self._bodies:
                union(id_a, id_b)

        # Group bodies by their root
        components: Dict[str, Set[str]] = {}
        for body_id in self._bodies:
            root = find(body_id)
            if root not in components:
                components[root] = set()
            components[root].add(body_id)

        # Create new islands
        old_islands = self._islands
        self._islands = {}
        self._body_to_island = {}
        self._next_island_id = 0

        for root, body_ids in components.items():
            island_id = self._create_island()
            island = self._islands[island_id]
            island.bodies = body_ids

            # Determine initial state
            all_sleeping = True
            for body_id in body_ids:
                body = self._bodies.get(body_id)
                if body and not body.is_sleeping:
                    all_sleeping = False
                    break

            island.state = IslandState.SLEEPING if all_sleeping else IslandState.ACTIVE

            # Update body-to-island mapping
            for body_id in body_ids:
                self._body_to_island[body_id] = island_id

    # =========================================================================
    # Update
    # =========================================================================

    def update(self, dt: float) -> None:
        """
        Update sleep states for all bodies.

        Should be called each physics step.

        Args:
            dt: Time step
        """
        # Update sleep timers for each body
        bodies_to_wake: List[int] = []
        bodies_to_check_sleep: List[int] = []

        for island_id, island in self._islands.items():
            if island.state == IslandState.SLEEPING:
                # Check if any body in sleeping island should wake
                for body_id in island.bodies:
                    body = self._bodies.get(body_id)
                    if body and not self.is_below_threshold(body):
                        bodies_to_wake.append(island_id)
                        break
            else:
                # Check if island can sleep
                can_sleep = True
                for body_id in island.bodies:
                    body = self._bodies.get(body_id)
                    if body:
                        if not self.can_sleep(body):
                            can_sleep = False
                            break
                        if not self.is_below_threshold(body):
                            can_sleep = False
                            self._sleep_timers[body_id] = 0.0
                            break
                        else:
                            self._sleep_timers[body_id] = self._sleep_timers.get(body_id, 0.0) + dt

                if can_sleep:
                    # Check if all bodies have been below threshold long enough
                    all_ready = True
                    for body_id in island.bodies:
                        if self._sleep_timers.get(body_id, 0.0) < self._time_threshold:
                            all_ready = False
                            break

                    if all_ready:
                        bodies_to_check_sleep.append(island_id)

        # Process wake-ups
        for island_id in bodies_to_wake:
            self._wake_island(island_id)

        # Process sleep attempts
        for island_id in bodies_to_check_sleep:
            island = self._islands.get(island_id)
            if island and self._can_island_sleep(island):
                self._sleep_island(island_id)

    # =========================================================================
    # Wake Through Joints
    # =========================================================================

    def wake_connected_bodies(self, body: 'RigidBody', through_joints: bool = True) -> None:
        """
        Wake a body and optionally all bodies connected through joints.

        Args:
            body: Body to start from
            through_joints: If True, wake bodies connected through joints
        """
        self.wake_up(body)

        if not through_joints:
            return

        # Wake all bodies in the same island (includes joint-connected bodies)
        body_id = body.id
        if body_id in self._body_to_island:
            island_id = self._body_to_island[body_id]
            self._wake_island(island_id)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, any]:
        """Get sleep manager statistics."""
        return {
            'total_bodies': len(self._bodies),
            'sleeping_bodies': self._sleeping_count,
            'awake_bodies': self._awake_count,
            'island_count': len(self._islands),
            'active_islands': sum(1 for i in self._islands.values() if i.is_active),
            'sleeping_islands': sum(1 for i in self._islands.values() if i.is_sleeping),
            'linear_threshold': self._linear_threshold,
            'angular_threshold': self._angular_threshold,
            'time_threshold': self._time_threshold,
        }

    def get_island_info(self, body: 'RigidBody') -> Optional[Dict[str, any]]:
        """
        Get information about the island containing a body.

        Args:
            body: Body to query

        Returns:
            Island information or None if body not registered
        """
        body_id = body.id
        if body_id not in self._body_to_island:
            return None

        island_id = self._body_to_island[body_id]
        island = self._islands.get(island_id)

        if not island:
            return None

        return {
            'island_id': island_id,
            'body_count': island.body_count,
            'state': island.state.name,
            'sleep_timer': island.sleep_timer,
            'body_ids': list(island.bodies),
        }

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:
        """Reset all sleep states."""
        for body in self._bodies.values():
            if body.is_sleeping:
                body.wake_up()

        self._sleep_timers = {body_id: 0.0 for body_id in self._bodies}
        self._sleeping_count = 0
        self._awake_count = len(self._bodies)

        for island in self._islands.values():
            island.state = IslandState.ACTIVE
            island.sleep_timer = 0.0

    def clear(self) -> None:
        """Clear all bodies and islands."""
        self._bodies.clear()
        self._sleep_timers.clear()
        self._islands.clear()
        self._body_to_island.clear()
        self._next_island_id = 0
        self._sleeping_count = 0
        self._awake_count = 0
