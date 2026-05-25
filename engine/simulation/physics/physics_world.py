"""
Physics World Module

The main physics simulation container that manages bodies, steps the
simulation, and handles collision detection and response.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Any, TYPE_CHECKING
from enum import Enum, auto
import time
import math

from .config import (
    PhysicsConfig,
    DEFAULT_GRAVITY,
    DEFAULT_TIMESTEP,
    MIN_TIMESTEP,
    MAX_SUBSTEPS,
    MAX_BODIES,
    SOLVER_ITERATIONS,
    POSITION_ITERATIONS,
)
from .rigid_body import RigidBody, BodyType, BodyState
from .collision_shapes import AABB, CollisionShape
from .physics_material import PhysicsMaterial, combine_materials
from .sleeping import SleepManager, Island
from .queries import (
    CollisionFilter, RaycastHit, OverlapResult, SweepResult,
    raycast_single, raycast_all,
    overlap_sphere, overlap_box, overlap_capsule,
    sweep_sphere, sweep_box, sweep_capsule,
)


# Type aliases
Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


def _vector_add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vector_sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vector_scale(v: Vector3, s: float) -> Vector3:
    return (v[0] * s, v[1] * s, v[2] * s)


def _vector_length(v: Vector3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vector_dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vector_normalize(v: Vector3) -> Vector3:
    from .config import FLOAT_COMPARISON_EPSILON
    length = _vector_length(v)
    if length < FLOAT_COMPARISON_EPSILON:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _vector_cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@dataclass
class Contact:
    """
    Contact point between two bodies.
    """
    body_a: RigidBody
    body_b: RigidBody
    point: Vector3  # World-space contact point
    normal: Vector3  # Contact normal (from A to B)
    penetration: float  # Penetration depth
    friction: float = 0.5
    restitution: float = 0.0

    # Solver data
    normal_impulse: float = 0.0
    tangent_impulse_1: float = 0.0
    tangent_impulse_2: float = 0.0


@dataclass
class ContactManifold:
    """
    Collection of contact points between two bodies.
    """
    body_a: RigidBody
    body_b: RigidBody
    contacts: List[Contact] = field(default_factory=list)
    is_new: bool = True

    @property
    def contact_count(self) -> int:
        return len(self.contacts)


class SimulationState(Enum):
    """State of the physics simulation."""
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()


# Callback types
CollisionCallback = Callable[[RigidBody, RigidBody, Contact], None]
TriggerCallback = Callable[[RigidBody, RigidBody], None]


class PhysicsWorld:
    """
    The physics simulation world.

    Manages all physics bodies, performs collision detection,
    and advances the simulation.
    """

    def __init__(self, config: Optional[PhysicsConfig] = None):
        """
        Initialize the physics world.

        Args:
            config: Physics configuration (uses defaults if None)
        """
        self._config = config or PhysicsConfig()
        self._config.validate()

        # Bodies
        self._bodies: Dict[str, RigidBody] = {}
        self._body_list: List[RigidBody] = []  # For iteration order

        # Gravity
        self._gravity: Vector3 = self._config.gravity

        # Time
        self._timestep = self._config.timestep
        self._accumulated_time = 0.0
        self._simulation_time = 0.0
        self._step_count = 0

        # State
        self._state = SimulationState.STOPPED

        # Sleep manager
        self._sleep_manager = SleepManager(
            linear_threshold=self._config.sleep_threshold_linear,
            angular_threshold=self._config.sleep_threshold_angular,
            time_threshold=self._config.sleep_time_threshold,
        )

        # Collision detection
        self._contact_pairs: Dict[Tuple[str, str], ContactManifold] = {}
        self._broad_phase_pairs: Set[Tuple[str, str]] = set()

        # Callbacks
        self._collision_enter_callbacks: List[CollisionCallback] = []
        self._collision_stay_callbacks: List[CollisionCallback] = []
        self._collision_exit_callbacks: List[CollisionCallback] = []
        self._trigger_enter_callbacks: List[TriggerCallback] = []
        self._trigger_stay_callbacks: List[TriggerCallback] = []
        self._trigger_exit_callbacks: List[TriggerCallback] = []

        # Islands (for parallel solving)
        self._islands: List[List[RigidBody]] = []

        # Statistics
        self._stats = {
            'bodies': 0,
            'active_bodies': 0,
            'sleeping_bodies': 0,
            'contacts': 0,
            'islands': 0,
            'step_time_ms': 0.0,
            'broadphase_time_ms': 0.0,
            'narrowphase_time_ms': 0.0,
            'solver_time_ms': 0.0,
        }

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> PhysicsConfig:
        """Get physics configuration."""
        return self._config

    @property
    def gravity(self) -> Vector3:
        """Get world gravity."""
        return self._gravity

    @gravity.setter
    def gravity(self, value: Vector3) -> None:
        """Set world gravity."""
        self._gravity = value

    @property
    def timestep(self) -> float:
        """Get fixed timestep."""
        return self._timestep

    @timestep.setter
    def timestep(self, value: float) -> None:
        """Set fixed timestep."""
        self._timestep = max(MIN_TIMESTEP, value)

    @property
    def simulation_time(self) -> float:
        """Get total simulation time."""
        return self._simulation_time

    @property
    def step_count(self) -> int:
        """Get number of simulation steps."""
        return self._step_count

    @property
    def state(self) -> SimulationState:
        """Get simulation state."""
        return self._state

    @property
    def body_count(self) -> int:
        """Get number of bodies."""
        return len(self._bodies)

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        self._stats['bodies'] = len(self._bodies)
        self._stats['active_bodies'] = self._sleep_manager.awake_count
        self._stats['sleeping_bodies'] = self._sleep_manager.sleeping_count
        self._stats['contacts'] = sum(m.contact_count for m in self._contact_pairs.values())
        self._stats['islands'] = self._sleep_manager.island_count
        return self._stats.copy()

    # =========================================================================
    # Body Management
    # =========================================================================

    def add_body(self, body: RigidBody) -> bool:
        """
        Add a body to the world.

        Args:
            body: Body to add

        Returns:
            True if added successfully
        """
        if body.id in self._bodies:
            return False

        if len(self._bodies) >= self._config.max_bodies:
            return False

        self._bodies[body.id] = body
        self._body_list.append(body)
        body._world = self

        # Register with sleep manager
        self._sleep_manager.register_body(body)

        return True

    def remove_body(self, body: RigidBody) -> bool:
        """
        Remove a body from the world.

        Args:
            body: Body to remove

        Returns:
            True if removed successfully
        """
        if body.id not in self._bodies:
            return False

        # Remove from sleep manager
        self._sleep_manager.unregister_body(body)

        # Remove contacts involving this body
        pairs_to_remove = [
            pair for pair in self._contact_pairs
            if body.id in pair
        ]
        for pair in pairs_to_remove:
            del self._contact_pairs[pair]

        # Remove body
        del self._bodies[body.id]
        self._body_list.remove(body)
        body._world = None

        return True

    def get_body(self, body_id: str) -> Optional[RigidBody]:
        """
        Get a body by ID.

        Args:
            body_id: Body identifier

        Returns:
            Body or None if not found
        """
        return self._bodies.get(body_id)

    def get_bodies(self) -> List[RigidBody]:
        """Get all bodies in the world."""
        return self._body_list.copy()

    def get_bodies_in_aabb(self, aabb: AABB) -> List[RigidBody]:
        """
        Get all bodies overlapping an AABB.

        Args:
            aabb: Axis-aligned bounding box

        Returns:
            List of overlapping bodies
        """
        results = []
        for body in self._body_list:
            if aabb.intersects(body.get_aabb()):
                results.append(body)
        return results

    def clear(self) -> None:
        """Remove all bodies from the world."""
        for body in list(self._body_list):
            self.remove_body(body)

        self._contact_pairs.clear()
        self._broad_phase_pairs.clear()
        self._islands.clear()
        self._simulation_time = 0.0
        self._step_count = 0
        self._accumulated_time = 0.0

    # =========================================================================
    # Simulation Control
    # =========================================================================

    def start(self) -> None:
        """Start the simulation."""
        self._state = SimulationState.RUNNING

    def pause(self) -> None:
        """Pause the simulation."""
        self._state = SimulationState.PAUSED

    def resume(self) -> None:
        """Resume the simulation."""
        if self._state == SimulationState.PAUSED:
            self._state = SimulationState.RUNNING

    def stop(self) -> None:
        """Stop the simulation."""
        self._state = SimulationState.STOPPED

    # =========================================================================
    # Simulation Step
    # =========================================================================

    def step(self, dt: float) -> int:
        """
        Advance the simulation by the given time.

        Uses fixed timestep with substeps for stability.

        Args:
            dt: Time to advance (seconds)

        Returns:
            Number of substeps performed
        """
        if self._state != SimulationState.RUNNING:
            return 0

        start_time = time.perf_counter()

        # Accumulate time
        self._accumulated_time += dt

        # Perform substeps
        substeps = 0
        while self._accumulated_time >= self._timestep and substeps < self._config.max_substeps:
            self._single_step(self._timestep)
            self._accumulated_time -= self._timestep
            substeps += 1

        # Update statistics
        self._stats['step_time_ms'] = (time.perf_counter() - start_time) * 1000.0

        return substeps

    def _single_step(self, dt: float) -> None:
        """
        Perform a single physics step.

        Args:
            dt: Fixed timestep
        """
        # Save previous states for interpolation
        for body in self._body_list:
            body.save_state()

        # 1. Broad phase collision detection
        bp_start = time.perf_counter()
        self._broad_phase()
        self._stats['broadphase_time_ms'] = (time.perf_counter() - bp_start) * 1000.0

        # 2. Narrow phase collision detection
        np_start = time.perf_counter()
        self._narrow_phase()
        self._stats['narrowphase_time_ms'] = (time.perf_counter() - np_start) * 1000.0

        # 3. Build islands
        self._build_islands()

        # 4. Integrate velocities (apply forces)
        for body in self._body_list:
            if not body.is_sleeping:
                body.integrate_velocities(dt, self._gravity)

        # 5. Solve constraints (velocity solver)
        solver_start = time.perf_counter()
        self._solve_velocities(dt)
        self._stats['solver_time_ms'] = (time.perf_counter() - solver_start) * 1000.0

        # 6. Integrate positions
        for body in self._body_list:
            if not body.is_sleeping:
                body.integrate_positions(dt)

        # 7. Position correction
        self._solve_positions()

        # 8. Update sleep states
        if self._config.enable_sleeping:
            self._sleep_manager.update(dt)

        # 9. Fire callbacks
        self._fire_callbacks()

        # Update simulation time
        self._simulation_time += dt
        self._step_count += 1

    # =========================================================================
    # Collision Detection
    # =========================================================================

    def _broad_phase(self) -> None:
        """
        Broad phase collision detection.

        Finds potentially colliding pairs using AABB tests.
        """
        self._broad_phase_pairs.clear()

        # Simple O(n^2) for now - would use spatial partitioning in production
        bodies = self._body_list
        n = len(bodies)

        for i in range(n):
            body_a = bodies[i]
            if body_a.is_sleeping:
                continue

            aabb_a = body_a.get_aabb()

            for j in range(i + 1, n):
                body_b = bodies[j]

                # Skip if both static
                if body_a.is_static and body_b.is_static:
                    continue

                # Skip if both sleeping
                if body_a.is_sleeping and body_b.is_sleeping:
                    continue

                # Check layer filtering
                if (body_a.collision_layer & body_b.collision_mask) == 0:
                    continue
                if (body_b.collision_layer & body_a.collision_mask) == 0:
                    continue

                # AABB test
                aabb_b = body_b.get_aabb()
                if aabb_a.intersects(aabb_b):
                    pair = (min(body_a.id, body_b.id), max(body_a.id, body_b.id))
                    self._broad_phase_pairs.add(pair)

    def _narrow_phase(self) -> None:
        """
        Narrow phase collision detection.

        Generates contact manifolds for broad phase pairs.
        """
        # Track which pairs are still active
        active_pairs: Set[Tuple[str, str]] = set()

        for pair in self._broad_phase_pairs:
            id_a, id_b = pair
            body_a = self._bodies.get(id_a)
            body_b = self._bodies.get(id_b)

            if body_a is None or body_b is None:
                continue

            # Generate contacts
            contacts = self._generate_contacts(body_a, body_b)

            if contacts:
                active_pairs.add(pair)

                if pair in self._contact_pairs:
                    # Update existing manifold
                    manifold = self._contact_pairs[pair]
                    manifold.contacts = contacts
                    manifold.is_new = False
                else:
                    # New manifold
                    self._contact_pairs[pair] = ContactManifold(
                        body_a=body_a,
                        body_b=body_b,
                        contacts=contacts,
                        is_new=True,
                    )

                # Merge islands
                self._sleep_manager.merge_islands(body_a, body_b)

        # Remove stale manifolds
        stale_pairs = set(self._contact_pairs.keys()) - active_pairs
        for pair in stale_pairs:
            del self._contact_pairs[pair]

    def _generate_contacts(self, body_a: RigidBody, body_b: RigidBody) -> List[Contact]:
        """
        Generate contact points between two bodies.

        Args:
            body_a: First body
            body_b: Second body

        Returns:
            List of contact points
        """
        # Simplified contact generation using penetration
        aabb_a = body_a.get_aabb()
        aabb_b = body_b.get_aabb()

        # Check for overlap
        if not aabb_a.intersects(aabb_b):
            return []

        # Compute penetration and normal using AABB
        center_a = aabb_a.center
        center_b = aabb_b.center

        # Half sizes
        half_a = aabb_a.half_extents
        half_b = aabb_b.half_extents

        # Overlap on each axis
        diff = _vector_sub(center_b, center_a)
        overlap_x = half_a[0] + half_b[0] - abs(diff[0])
        overlap_y = half_a[1] + half_b[1] - abs(diff[1])
        overlap_z = half_a[2] + half_b[2] - abs(diff[2])

        if overlap_x <= 0 or overlap_y <= 0 or overlap_z <= 0:
            return []

        # Find minimum penetration axis
        if overlap_x < overlap_y and overlap_x < overlap_z:
            penetration = overlap_x
            normal = (1.0 if diff[0] > 0 else -1.0, 0.0, 0.0)
        elif overlap_y < overlap_z:
            penetration = overlap_y
            normal = (0.0, 1.0 if diff[1] > 0 else -1.0, 0.0)
        else:
            penetration = overlap_z
            normal = (0.0, 0.0, 1.0 if diff[2] > 0 else -1.0)

        # Contact point (midpoint of overlap)
        contact_point = (
            (center_a[0] + center_b[0]) * 0.5,
            (center_a[1] + center_b[1]) * 0.5,
            (center_a[2] + center_b[2]) * 0.5,
        )

        # Combine materials
        static_friction, dynamic_friction, restitution = combine_materials(
            body_a.material, body_b.material
        )

        return [Contact(
            body_a=body_a,
            body_b=body_b,
            point=contact_point,
            normal=normal,
            penetration=penetration,
            friction=dynamic_friction,
            restitution=restitution,
        )]

    # =========================================================================
    # Constraint Solving
    # =========================================================================

    def _build_islands(self) -> None:
        """Build simulation islands from contact graph."""
        contact_pairs = [
            (manifold.body_a.id, manifold.body_b.id)
            for manifold in self._contact_pairs.values()
        ]
        self._sleep_manager.rebuild_islands(contact_pairs)

    def _solve_velocities(self, dt: float) -> None:
        """
        Solve velocity constraints.

        Args:
            dt: Timestep
        """
        for _ in range(self._config.solver_iterations):
            for manifold in self._contact_pairs.values():
                for contact in manifold.contacts:
                    self._solve_contact_velocity(contact, dt)

    def _solve_contact_velocity(self, contact: Contact, dt: float) -> None:
        """
        Solve velocity constraint for a single contact.

        Args:
            contact: Contact to solve
            dt: Timestep
        """
        body_a = contact.body_a
        body_b = contact.body_b

        # Skip if both static/kinematic
        if body_a.inverse_mass == 0 and body_b.inverse_mass == 0:
            return

        # Relative velocity at contact point
        r_a = _vector_sub(contact.point, body_a.world_center_of_mass)
        r_b = _vector_sub(contact.point, body_b.world_center_of_mass)

        vel_a = body_a.get_velocity_at_point(contact.point)
        vel_b = body_b.get_velocity_at_point(contact.point)

        relative_vel = _vector_sub(vel_b, vel_a)

        # Normal velocity
        normal_vel = _vector_dot(relative_vel, contact.normal)

        # Don't resolve if separating
        if normal_vel > 0:
            return

        # Compute effective mass
        from .config import FLOAT_COMPARISON_EPSILON
        inv_mass_sum = body_a.inverse_mass + body_b.inverse_mass

        # Add rotational contribution
        r_a_cross_n = _vector_cross(r_a, contact.normal)
        r_b_cross_n = _vector_cross(r_b, contact.normal)

        # Simplified effective mass (ignoring inertia for now)
        effective_mass = inv_mass_sum
        if effective_mass < FLOAT_COMPARISON_EPSILON:
            return

        # Compute impulse magnitude (safe division)
        j = -(1.0 + contact.restitution) * normal_vel / effective_mass

        # Accumulate impulse
        old_impulse = contact.normal_impulse
        contact.normal_impulse = max(old_impulse + j, 0.0)
        j = contact.normal_impulse - old_impulse

        # Apply impulse
        impulse = _vector_scale(contact.normal, j)

        if body_a.is_dynamic:
            body_a.apply_impulse(_vector_scale(impulse, -1.0), contact.point)
        if body_b.is_dynamic:
            body_b.apply_impulse(impulse, contact.point)

        # Friction (simplified)
        tangent = _vector_sub(relative_vel, _vector_scale(contact.normal, normal_vel))
        tangent_len = _vector_length(tangent)

        if tangent_len > FLOAT_COMPARISON_EPSILON:
            tangent = _vector_scale(tangent, 1.0 / tangent_len)

            # Friction impulse magnitude (safe division - inv_mass_sum already validated above)
            jt = -_vector_dot(relative_vel, tangent) / effective_mass

            # Coulomb friction
            max_friction = contact.friction * contact.normal_impulse
            jt = max(-max_friction, min(max_friction, jt))

            friction_impulse = _vector_scale(tangent, jt)

            if body_a.is_dynamic:
                body_a.apply_impulse(_vector_scale(friction_impulse, -1.0), contact.point)
            if body_b.is_dynamic:
                body_b.apply_impulse(friction_impulse, contact.point)

    def _solve_positions(self) -> None:
        """Solve position constraints to prevent penetration."""
        for _ in range(self._config.position_iterations):
            for manifold in self._contact_pairs.values():
                for contact in manifold.contacts:
                    self._solve_contact_position(contact)

    def _solve_contact_position(self, contact: Contact) -> None:
        """
        Solve position constraint for a single contact.

        Args:
            contact: Contact to solve
        """
        body_a = contact.body_a
        body_b = contact.body_b

        if body_a.inverse_mass == 0 and body_b.inverse_mass == 0:
            return

        # Position correction
        slop = self._config.contact_slop
        baumgarte = self._config.contact_baumgarte

        penetration = contact.penetration - slop
        if penetration <= 0:
            return

        from .config import FLOAT_COMPARISON_EPSILON
        inv_mass_sum = body_a.inverse_mass + body_b.inverse_mass
        if inv_mass_sum < FLOAT_COMPARISON_EPSILON:
            return

        correction = (penetration * baumgarte) / inv_mass_sum

        if body_a.is_dynamic:
            body_a.position = _vector_add(
                body_a.position,
                _vector_scale(contact.normal, -correction * body_a.inverse_mass)
            )

        if body_b.is_dynamic:
            body_b.position = _vector_add(
                body_b.position,
                _vector_scale(contact.normal, correction * body_b.inverse_mass)
            )

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _fire_callbacks(self) -> None:
        """Fire collision and trigger callbacks."""
        for manifold in self._contact_pairs.values():
            if manifold.body_a.shape.is_trigger or manifold.body_b.shape.is_trigger:
                # Trigger callbacks
                if manifold.is_new:
                    for callback in self._trigger_enter_callbacks:
                        callback(manifold.body_a, manifold.body_b)
                else:
                    for callback in self._trigger_stay_callbacks:
                        callback(manifold.body_a, manifold.body_b)
            else:
                # Collision callbacks
                if manifold.is_new and manifold.contacts:
                    for callback in self._collision_enter_callbacks:
                        callback(manifold.body_a, manifold.body_b, manifold.contacts[0])
                elif manifold.contacts:
                    for callback in self._collision_stay_callbacks:
                        callback(manifold.body_a, manifold.body_b, manifold.contacts[0])

    def on_collision_enter(self, callback: CollisionCallback) -> None:
        """Register collision enter callback."""
        self._collision_enter_callbacks.append(callback)

    def on_collision_stay(self, callback: CollisionCallback) -> None:
        """Register collision stay callback."""
        self._collision_stay_callbacks.append(callback)

    def on_collision_exit(self, callback: CollisionCallback) -> None:
        """Register collision exit callback."""
        self._collision_exit_callbacks.append(callback)

    def on_trigger_enter(self, callback: TriggerCallback) -> None:
        """Register trigger enter callback."""
        self._trigger_enter_callbacks.append(callback)

    def on_trigger_stay(self, callback: TriggerCallback) -> None:
        """Register trigger stay callback."""
        self._trigger_stay_callbacks.append(callback)

    def on_trigger_exit(self, callback: TriggerCallback) -> None:
        """Register trigger exit callback."""
        self._trigger_exit_callbacks.append(callback)

    # =========================================================================
    # Queries
    # =========================================================================

    def raycast(
        self,
        origin: Vector3,
        direction: Vector3,
        max_distance: float = 1000.0,
        filter: Optional[CollisionFilter] = None,
    ) -> Optional[RaycastHit]:
        """
        Cast a ray and return the closest hit.

        Args:
            origin: Ray origin
            direction: Ray direction
            max_distance: Maximum distance
            filter: Collision filter

        Returns:
            RaycastHit or None
        """
        return raycast_single(self._body_list, origin, direction, max_distance, filter)

    def raycast_all(
        self,
        origin: Vector3,
        direction: Vector3,
        max_distance: float = 1000.0,
        filter: Optional[CollisionFilter] = None,
    ) -> List[RaycastHit]:
        """
        Cast a ray and return all hits.

        Args:
            origin: Ray origin
            direction: Ray direction
            max_distance: Maximum distance
            filter: Collision filter

        Returns:
            List of RaycastHit sorted by distance
        """
        return raycast_all(self._body_list, origin, direction, max_distance, filter)

    def overlap_test(
        self,
        shape: CollisionShape,
        position: Vector3,
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        filter: Optional[CollisionFilter] = None,
    ) -> List[OverlapResult]:
        """
        Test for overlapping bodies.

        Args:
            shape: Test shape
            position: Shape position
            rotation: Shape rotation
            filter: Collision filter

        Returns:
            List of overlapping bodies
        """
        from .collision_shapes import ShapeType, SphereShape, BoxShape, CapsuleShape

        if shape.shape_type == ShapeType.SPHERE:
            sphere = shape
            center = _vector_add(position, sphere.local_offset)
            return overlap_sphere(self._body_list, center, sphere.radius, filter)

        elif shape.shape_type == ShapeType.BOX:
            box = shape
            center = _vector_add(position, box.local_offset)
            return overlap_box(self._body_list, center, box.half_extents, rotation, filter)

        elif shape.shape_type == ShapeType.CAPSULE:
            capsule = shape
            center = _vector_add(position, capsule.local_offset)
            half_height = (0.0, capsule.half_height, 0.0)
            start = _vector_sub(center, half_height)
            end = _vector_add(center, half_height)
            return overlap_capsule(self._body_list, start, end, capsule.radius, filter)

        # Default: use sphere approximation
        aabb = shape.compute_aabb(position, rotation)
        max_extent = max(aabb.half_extents)
        return overlap_sphere(self._body_list, position, max_extent, filter)

    def sweep_test(
        self,
        shape: CollisionShape,
        start: Vector3,
        direction: Vector3,
        distance: float,
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        filter: Optional[CollisionFilter] = None,
    ) -> SweepResult:
        """
        Sweep a shape along a direction.

        Args:
            shape: Shape to sweep
            start: Start position
            direction: Sweep direction
            distance: Sweep distance
            rotation: Shape rotation
            filter: Collision filter

        Returns:
            SweepResult
        """
        from .collision_shapes import ShapeType, SphereShape, BoxShape, CapsuleShape

        if shape.shape_type == ShapeType.SPHERE:
            sphere = shape
            return sweep_sphere(
                self._body_list, start, direction,
                sphere.radius, distance, filter
            )

        elif shape.shape_type == ShapeType.BOX:
            box = shape
            return sweep_box(
                self._body_list, start, direction,
                box.half_extents, distance, rotation, filter
            )

        elif shape.shape_type == ShapeType.CAPSULE:
            capsule = shape
            half_height = (0.0, capsule.half_height, 0.0)
            start_a = _vector_sub(start, half_height)
            start_b = _vector_add(start, half_height)
            return sweep_capsule(
                self._body_list, start_a, start_b, direction,
                capsule.radius, distance, filter
            )

        # Default: use sphere approximation
        aabb = shape.compute_aabb(start, rotation)
        max_extent = max(aabb.half_extents)
        return sweep_sphere(self._body_list, start, direction, max_extent, distance, filter)

    # =========================================================================
    # Utility
    # =========================================================================

    def wake_all(self) -> None:
        """Wake all sleeping bodies."""
        for body in self._body_list:
            if body.is_sleeping:
                body.wake_up()

    def get_interpolated_state(self, body: RigidBody, alpha: float) -> BodyState:
        """
        Get interpolated body state.

        Args:
            body: Body to interpolate
            alpha: Interpolation factor (0=previous, 1=current)

        Returns:
            Interpolated body state
        """
        return body.interpolate_state(alpha)

    def get_contact_manifolds(self) -> List[ContactManifold]:
        """Get all current contact manifolds."""
        return list(self._contact_pairs.values())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize world state to dictionary."""
        return {
            'gravity': self._gravity,
            'timestep': self._timestep,
            'simulation_time': self._simulation_time,
            'step_count': self._step_count,
            'bodies': [body.to_dict() for body in self._body_list],
            'config': {
                'max_substeps': self._config.max_substeps,
                'solver_iterations': self._config.solver_iterations,
                'position_iterations': self._config.position_iterations,
                'enable_sleeping': self._config.enable_sleeping,
            },
        }
