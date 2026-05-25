"""
Rigid Body Module

Defines the RigidBody class which represents a physics-simulated object
with mass, velocity, and collision properties.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict, Callable, TYPE_CHECKING
from enum import Enum, auto
import math
import uuid

from .body_flags import BodyFlags, BodyFlagBits
from .collision_shapes import CollisionShape, AABB, MassProperties, SphereShape
from .physics_material import PhysicsMaterial
from .config import (
    DEFAULT_LINEAR_DAMPING,
    DEFAULT_ANGULAR_DAMPING,
    MAX_LINEAR_VELOCITY,
    MAX_ANGULAR_VELOCITY,
    MIN_MASS,
    MAX_MASS,
    MIN_INERTIA,
)

if TYPE_CHECKING:
    from .physics_world import PhysicsWorld


# Type aliases
Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (x, y, z, w)
Matrix3x3 = Tuple[Tuple[float, float, float], ...]


class BodyType(Enum):
    """
    Type of physics body determining how it interacts with simulation.
    """
    STATIC = auto()     # Never moves, infinite mass
    KINEMATIC = auto()  # Moves via scripts, not affected by forces
    DYNAMIC = auto()    # Fully simulated, affected by forces and collisions


@dataclass
class BodyState:
    """
    Snapshot of rigid body state for interpolation/rollback.
    """
    position: Vector3 = (0.0, 0.0, 0.0)
    rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    linear_velocity: Vector3 = (0.0, 0.0, 0.0)
    angular_velocity: Vector3 = (0.0, 0.0, 0.0)
    timestamp: float = 0.0


def _vector_add(a: Vector3, b: Vector3) -> Vector3:
    """Add two vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vector_sub(a: Vector3, b: Vector3) -> Vector3:
    """Subtract two vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vector_scale(v: Vector3, s: float) -> Vector3:
    """Scale a vector."""
    return (v[0] * s, v[1] * s, v[2] * s)


def _vector_dot(a: Vector3, b: Vector3) -> float:
    """Dot product of two vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vector_cross(a: Vector3, b: Vector3) -> Vector3:
    """Cross product of two vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vector_length(v: Vector3) -> float:
    """Get length of a vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vector_length_sq(v: Vector3) -> float:
    """Get squared length of a vector."""
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def _vector_normalize(v: Vector3) -> Vector3:
    """Normalize a vector."""
    from .config import FLOAT_COMPARISON_EPSILON
    length = _vector_length(v)
    if length < FLOAT_COMPARISON_EPSILON:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _vector_clamp(v: Vector3, max_length: float) -> Vector3:
    """Clamp vector magnitude."""
    length = _vector_length(v)
    if length > max_length and length > 0:
        scale = max_length / length
        return _vector_scale(v, scale)
    return v


def _quaternion_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Multiply two quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quaternion_normalize(q: Quaternion) -> Quaternion:
    """Normalize a quaternion."""
    from .config import FLOAT_COMPARISON_EPSILON
    length = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    if length < FLOAT_COMPARISON_EPSILON:
        return (0.0, 0.0, 0.0, 1.0)
    return (q[0] / length, q[1] / length, q[2] / length, q[3] / length)


def _quaternion_conjugate(q: Quaternion) -> Quaternion:
    """Get quaternion conjugate."""
    return (-q[0], -q[1], -q[2], q[3])


def _rotate_vector(v: Vector3, q: Quaternion) -> Vector3:
    """Rotate a vector by a quaternion."""
    qx, qy, qz, qw = q
    vx, vy, vz = v

    # q * v * q^-1
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _inverse_rotate_vector(v: Vector3, q: Quaternion) -> Vector3:
    """Rotate a vector by inverse quaternion."""
    return _rotate_vector(v, _quaternion_conjugate(q))


def _matrix_vector_multiply(m: Matrix3x3, v: Vector3) -> Vector3:
    """Multiply a 3x3 matrix by a vector."""
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


class RigidBody:
    """
    A rigid body in the physics simulation.

    Represents a physical object with mass, collision shape, and motion properties.
    Can be static (immovable), kinematic (script-controlled), or dynamic (simulated).
    """

    def __init__(
        self,
        body_type: BodyType = BodyType.DYNAMIC,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        mass: float = 1.0,
        shape: Optional[CollisionShape] = None,
        material: Optional[PhysicsMaterial] = None,
        flags: Optional[BodyFlags] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a rigid body.

        Args:
            body_type: Type of body (STATIC, KINEMATIC, DYNAMIC)
            position: Initial world position
            rotation: Initial world rotation as quaternion (x, y, z, w)
            mass: Body mass in kg (ignored for static/kinematic)
            shape: Collision shape (default: unit sphere)
            material: Physics material for friction/restitution
            flags: Body behavior flags
            name: Optional name for debugging
        """
        # Identification
        self._id: str = str(uuid.uuid4())
        self._name: str = name or f"Body_{self._id[:8]}"

        # Type
        self._body_type = body_type

        # Transform
        self._position = position
        self._rotation = _quaternion_normalize(rotation)

        # Velocity
        self._linear_velocity: Vector3 = (0.0, 0.0, 0.0)
        self._angular_velocity: Vector3 = (0.0, 0.0, 0.0)

        # Accumulated forces (cleared each step)
        self._force_accumulator: Vector3 = (0.0, 0.0, 0.0)
        self._torque_accumulator: Vector3 = (0.0, 0.0, 0.0)

        # Mass properties
        self._mass = 0.0 if body_type == BodyType.STATIC else max(MIN_MASS, min(MAX_MASS, mass))
        self._inverse_mass = 0.0 if self._mass == 0 else 1.0 / self._mass

        # Shape and material
        self._shape = shape or SphereShape(radius=0.5)
        self._material = material or PhysicsMaterial()

        # Compute inertia from shape
        self._update_mass_properties()

        # Flags
        self._flags = flags or self._default_flags_for_type(body_type)

        # Damping
        self._linear_damping = DEFAULT_LINEAR_DAMPING
        self._angular_damping = DEFAULT_ANGULAR_DAMPING

        # Sleeping state
        self._is_sleeping = False
        self._sleep_timer = 0.0

        # Collision state (layer 1 is the default, 0 means "no layer" which would never collide)
        self._collision_layer = 1
        self._collision_mask = 0xFFFFFFFF

        # World reference (set when added to world)
        self._world: Optional['PhysicsWorld'] = None

        # Island index for sleeping optimization
        self._island_index = -1

        # User data
        self._user_data: Dict[str, Any] = {}

        # Contact list (updated each frame by collision detection)
        self._contacts: List[Any] = []

        # Joint connections
        self._joints: List[Any] = []

        # Cached AABB
        self._cached_aabb: Optional[AABB] = None
        self._aabb_dirty = True

        # Previous state for interpolation
        self._previous_state: Optional[BodyState] = None

    def _default_flags_for_type(self, body_type: BodyType) -> BodyFlags:
        """Get default flags for a body type."""
        if body_type == BodyType.STATIC:
            return BodyFlags.static_body()
        elif body_type == BodyType.KINEMATIC:
            return BodyFlags.kinematic_body()
        else:
            return BodyFlags.dynamic_body()

    def _update_mass_properties(self) -> None:
        """Update inertia tensor from shape and mass."""
        if self._body_type == BodyType.STATIC or self._mass == 0:
            self._inertia_tensor = (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )
            self._inverse_inertia_tensor = self._inertia_tensor
            self._inverse_mass = 0.0
            self._center_of_mass = (0.0, 0.0, 0.0)
            return

        # Get mass properties from shape using material density
        props = self._shape.compute_mass_properties(self._material.density)

        # Scale inertia to match actual mass (avoid division by near-zero)
        if props.mass > MIN_MASS:
            scale = self._mass / props.mass
        else:
            scale = 1.0

        self._inertia_tensor = tuple(
            tuple(v * scale for v in row)
            for row in props.inertia_tensor
        )

        # Compute inverse inertia tensor
        inv = []
        for i in range(3):
            row = []
            for j in range(3):
                if i == j and self._inertia_tensor[i][j] > MIN_INERTIA:
                    row.append(1.0 / self._inertia_tensor[i][j])
                else:
                    row.append(0.0)
            inv.append(tuple(row))
        self._inverse_inertia_tensor = tuple(inv)

        self._center_of_mass = props.center_of_mass

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def id(self) -> str:
        """Get unique body ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get body name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set body name."""
        self._name = value

    @property
    def body_type(self) -> BodyType:
        """Get body type."""
        return self._body_type

    @body_type.setter
    def body_type(self, value: BodyType) -> None:
        """Set body type."""
        old_type = self._body_type
        self._body_type = value

        if value != old_type:
            if value == BodyType.STATIC:
                self._mass = 0.0
                self._inverse_mass = 0.0
                self._linear_velocity = (0.0, 0.0, 0.0)
                self._angular_velocity = (0.0, 0.0, 0.0)
            elif old_type == BodyType.STATIC and value == BodyType.DYNAMIC:
                self._mass = 1.0
                self._inverse_mass = 1.0

            self._update_mass_properties()

    @property
    def is_static(self) -> bool:
        """Check if body is static."""
        return self._body_type == BodyType.STATIC

    @property
    def is_kinematic(self) -> bool:
        """Check if body is kinematic."""
        return self._body_type == BodyType.KINEMATIC

    @property
    def is_dynamic(self) -> bool:
        """Check if body is dynamic."""
        return self._body_type == BodyType.DYNAMIC

    @property
    def position(self) -> Vector3:
        """Get world position."""
        return self._position

    @position.setter
    def position(self, value: Vector3) -> None:
        """Set world position."""
        self._position = value
        self._aabb_dirty = True
        if self._is_sleeping:
            self.wake_up()

    @property
    def rotation(self) -> Quaternion:
        """Get world rotation as quaternion (x, y, z, w)."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Quaternion) -> None:
        """Set world rotation as quaternion."""
        self._rotation = _quaternion_normalize(value)
        self._aabb_dirty = True
        if self._is_sleeping:
            self.wake_up()

    @property
    def linear_velocity(self) -> Vector3:
        """Get linear velocity in m/s."""
        return self._linear_velocity

    @linear_velocity.setter
    def linear_velocity(self, value: Vector3) -> None:
        """Set linear velocity."""
        if self._body_type == BodyType.STATIC:
            return
        self._linear_velocity = _vector_clamp(value, MAX_LINEAR_VELOCITY)
        if self._is_sleeping:
            self.wake_up()

    @property
    def angular_velocity(self) -> Vector3:
        """Get angular velocity in rad/s."""
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value: Vector3) -> None:
        """Set angular velocity."""
        if self._body_type == BodyType.STATIC:
            return
        self._angular_velocity = _vector_clamp(value, MAX_ANGULAR_VELOCITY)
        if self._is_sleeping:
            self.wake_up()

    @property
    def mass(self) -> float:
        """Get mass in kg."""
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        """Set mass."""
        if self._body_type == BodyType.STATIC:
            return
        self._mass = max(MIN_MASS, min(MAX_MASS, value))
        self._inverse_mass = 1.0 / self._mass if self._mass > 0 else 0.0
        self._update_mass_properties()

    @property
    def inverse_mass(self) -> float:
        """Get inverse mass."""
        return self._inverse_mass

    @property
    def inertia_tensor(self) -> Matrix3x3:
        """Get inertia tensor."""
        return self._inertia_tensor

    @property
    def inverse_inertia_tensor(self) -> Matrix3x3:
        """Get inverse inertia tensor."""
        return self._inverse_inertia_tensor

    @property
    def center_of_mass(self) -> Vector3:
        """Get center of mass in local space."""
        return self._center_of_mass

    @property
    def world_center_of_mass(self) -> Vector3:
        """Get center of mass in world space."""
        local_com = _rotate_vector(self._center_of_mass, self._rotation)
        return _vector_add(self._position, local_com)

    @property
    def shape(self) -> CollisionShape:
        """Get collision shape."""
        return self._shape

    @shape.setter
    def shape(self, value: CollisionShape) -> None:
        """Set collision shape."""
        self._shape = value
        self._update_mass_properties()
        self._aabb_dirty = True

    @property
    def material(self) -> PhysicsMaterial:
        """Get physics material."""
        return self._material

    @material.setter
    def material(self, value: PhysicsMaterial) -> None:
        """Set physics material."""
        self._material = value
        self._update_mass_properties()

    @property
    def flags(self) -> BodyFlags:
        """Get body flags."""
        return self._flags

    @flags.setter
    def flags(self, value: BodyFlags) -> None:
        """Set body flags."""
        self._flags = value

    @property
    def linear_damping(self) -> float:
        """Get linear damping."""
        return self._linear_damping

    @linear_damping.setter
    def linear_damping(self, value: float) -> None:
        """Set linear damping."""
        self._linear_damping = max(0.0, min(1.0, value))

    @property
    def angular_damping(self) -> float:
        """Get angular damping."""
        return self._angular_damping

    @angular_damping.setter
    def angular_damping(self, value: float) -> None:
        """Set angular damping."""
        self._angular_damping = max(0.0, min(1.0, value))

    @property
    def is_sleeping(self) -> bool:
        """Check if body is sleeping."""
        return self._is_sleeping

    @property
    def collision_layer(self) -> int:
        """Get collision layer."""
        return self._collision_layer

    @collision_layer.setter
    def collision_layer(self, value: int) -> None:
        """Set collision layer."""
        self._collision_layer = value

    @property
    def collision_mask(self) -> int:
        """Get collision mask."""
        return self._collision_mask

    @collision_mask.setter
    def collision_mask(self, value: int) -> None:
        """Set collision mask."""
        self._collision_mask = value

    @property
    def world(self) -> Optional['PhysicsWorld']:
        """Get owning physics world."""
        return self._world

    @property
    def user_data(self) -> Dict[str, Any]:
        """Get user data dictionary."""
        return self._user_data

    @property
    def contacts(self) -> List[Any]:
        """Get current contact list."""
        return self._contacts.copy()

    @property
    def joints(self) -> List[Any]:
        """Get connected joints."""
        return self._joints.copy()

    @property
    def island_index(self) -> int:
        """Get island index for sleeping optimization."""
        return self._island_index

    # =========================================================================
    # Force and Impulse Application
    # =========================================================================

    def apply_force(self, force: Vector3, world_point: Optional[Vector3] = None) -> None:
        """
        Apply a force to the body.

        Forces are accumulated and applied during integration.
        Specify world_point to apply force at a specific location (generates torque).

        Args:
            force: Force vector in world space (Newtons)
            world_point: Point of application in world space (default: center of mass)
        """
        if self._body_type != BodyType.DYNAMIC or self._is_sleeping:
            if self._is_sleeping:
                self.wake_up()
            elif self._body_type != BodyType.DYNAMIC:
                return

        # Apply position lock
        mask = self._flags.get_position_lock_mask()
        force = (force[0] * mask[0], force[1] * mask[1], force[2] * mask[2])

        self._force_accumulator = _vector_add(self._force_accumulator, force)

        # If applied at a point, generate torque
        if world_point is not None:
            r = _vector_sub(world_point, self.world_center_of_mass)
            torque = _vector_cross(r, force)
            self.apply_torque(torque)

    def apply_force_local(self, force: Vector3, local_point: Optional[Vector3] = None) -> None:
        """
        Apply a force in local space.

        Args:
            force: Force vector in local space
            local_point: Point of application in local space
        """
        world_force = _rotate_vector(force, self._rotation)
        world_point = None
        if local_point is not None:
            world_point = _vector_add(
                self._position,
                _rotate_vector(local_point, self._rotation)
            )
        self.apply_force(world_force, world_point)

    def apply_impulse(self, impulse: Vector3, world_point: Optional[Vector3] = None) -> None:
        """
        Apply an instantaneous impulse to the body.

        Directly modifies velocity without going through force accumulation.

        Args:
            impulse: Impulse vector in world space (kg*m/s)
            world_point: Point of application in world space
        """
        if self._body_type != BodyType.DYNAMIC:
            return

        if self._is_sleeping:
            self.wake_up()

        # Apply position lock
        mask = self._flags.get_position_lock_mask()
        impulse = (impulse[0] * mask[0], impulse[1] * mask[1], impulse[2] * mask[2])

        # Linear impulse: dv = J / m
        dv = _vector_scale(impulse, self._inverse_mass)
        self._linear_velocity = _vector_add(self._linear_velocity, dv)

        # If applied at a point, generate angular impulse
        if world_point is not None:
            r = _vector_sub(world_point, self.world_center_of_mass)
            angular_impulse = _vector_cross(r, impulse)
            self.apply_angular_impulse(angular_impulse)

        # Clamp velocity
        self._linear_velocity = _vector_clamp(self._linear_velocity, MAX_LINEAR_VELOCITY)

    def apply_impulse_local(self, impulse: Vector3, local_point: Optional[Vector3] = None) -> None:
        """
        Apply an impulse in local space.

        Args:
            impulse: Impulse vector in local space
            local_point: Point of application in local space
        """
        world_impulse = _rotate_vector(impulse, self._rotation)
        world_point = None
        if local_point is not None:
            world_point = _vector_add(
                self._position,
                _rotate_vector(local_point, self._rotation)
            )
        self.apply_impulse(world_impulse, world_point)

    def apply_torque(self, torque: Vector3) -> None:
        """
        Apply a torque to the body.

        Torques are accumulated and applied during integration.

        Args:
            torque: Torque vector in world space (N*m)
        """
        if self._body_type != BodyType.DYNAMIC:
            return

        if self._is_sleeping:
            self.wake_up()

        # Apply rotation lock
        mask = self._flags.get_rotation_lock_mask()
        torque = (torque[0] * mask[0], torque[1] * mask[1], torque[2] * mask[2])

        self._torque_accumulator = _vector_add(self._torque_accumulator, torque)

    def apply_angular_impulse(self, impulse: Vector3) -> None:
        """
        Apply an instantaneous angular impulse.

        Args:
            impulse: Angular impulse in world space (kg*m^2/s)
        """
        if self._body_type != BodyType.DYNAMIC:
            return

        if self._is_sleeping:
            self.wake_up()

        # Apply rotation lock
        mask = self._flags.get_rotation_lock_mask()
        impulse = (impulse[0] * mask[0], impulse[1] * mask[1], impulse[2] * mask[2])

        # Transform impulse to local space
        local_impulse = _inverse_rotate_vector(impulse, self._rotation)

        # dw = I^-1 * L
        dw_local = _matrix_vector_multiply(self._inverse_inertia_tensor, local_impulse)

        # Transform back to world space
        dw = _rotate_vector(dw_local, self._rotation)

        self._angular_velocity = _vector_add(self._angular_velocity, dw)

        # Clamp angular velocity
        self._angular_velocity = _vector_clamp(self._angular_velocity, MAX_ANGULAR_VELOCITY)

    def clear_forces(self) -> None:
        """Clear accumulated forces and torques."""
        self._force_accumulator = (0.0, 0.0, 0.0)
        self._torque_accumulator = (0.0, 0.0, 0.0)

    # =========================================================================
    # Velocity at Point
    # =========================================================================

    def get_velocity_at_point(self, world_point: Vector3) -> Vector3:
        """
        Get velocity at a specific world point.

        Accounts for both linear and angular velocity.

        Args:
            world_point: Point in world space

        Returns:
            Velocity at that point
        """
        r = _vector_sub(world_point, self.world_center_of_mass)
        angular_contribution = _vector_cross(self._angular_velocity, r)
        return _vector_add(self._linear_velocity, angular_contribution)

    def get_velocity_at_local_point(self, local_point: Vector3) -> Vector3:
        """
        Get velocity at a local point.

        Args:
            local_point: Point in local space

        Returns:
            Velocity at that point in world space
        """
        world_point = _vector_add(
            self._position,
            _rotate_vector(local_point, self._rotation)
        )
        return self.get_velocity_at_point(world_point)

    # =========================================================================
    # Transform Operations
    # =========================================================================

    def transform_point_to_world(self, local_point: Vector3) -> Vector3:
        """Transform a point from local to world space."""
        rotated = _rotate_vector(local_point, self._rotation)
        return _vector_add(self._position, rotated)

    def transform_point_to_local(self, world_point: Vector3) -> Vector3:
        """Transform a point from world to local space."""
        relative = _vector_sub(world_point, self._position)
        return _inverse_rotate_vector(relative, self._rotation)

    def transform_direction_to_world(self, local_direction: Vector3) -> Vector3:
        """Transform a direction from local to world space."""
        return _rotate_vector(local_direction, self._rotation)

    def transform_direction_to_local(self, world_direction: Vector3) -> Vector3:
        """Transform a direction from world to local space."""
        return _inverse_rotate_vector(world_direction, self._rotation)

    # =========================================================================
    # AABB
    # =========================================================================

    def get_aabb(self) -> AABB:
        """Get world-space axis-aligned bounding box."""
        if self._aabb_dirty or self._cached_aabb is None:
            self._cached_aabb = self._shape.compute_aabb(self._position, self._rotation)
            self._aabb_dirty = False
        return self._cached_aabb

    # =========================================================================
    # Sleep Management
    # =========================================================================

    def wake_up(self) -> None:
        """Wake up the body if sleeping."""
        if self._is_sleeping:
            self._is_sleeping = False
            self._sleep_timer = 0.0
            self._flags.is_sleeping = False

    def put_to_sleep(self) -> None:
        """Put the body to sleep."""
        if self._flags.can_sleep and self._body_type == BodyType.DYNAMIC:
            self._is_sleeping = True
            self._flags.is_sleeping = True
            self._linear_velocity = (0.0, 0.0, 0.0)
            self._angular_velocity = (0.0, 0.0, 0.0)
            self.clear_forces()

    def update_sleep_timer(self, dt: float, threshold_linear: float, threshold_angular: float) -> bool:
        """
        Update sleep timer based on velocity.

        Args:
            dt: Time step
            threshold_linear: Linear velocity threshold
            threshold_angular: Angular velocity threshold

        Returns:
            True if body should sleep
        """
        if not self._flags.can_sleep:
            return False

        linear_speed = _vector_length(self._linear_velocity)
        angular_speed = _vector_length(self._angular_velocity)

        if linear_speed < threshold_linear and angular_speed < threshold_angular:
            self._sleep_timer += dt
        else:
            self._sleep_timer = 0.0

        return self._sleep_timer > 0

    @property
    def sleep_timer(self) -> float:
        """Get current sleep timer value."""
        return self._sleep_timer

    # =========================================================================
    # State Management
    # =========================================================================

    def get_state(self) -> BodyState:
        """Get current body state."""
        return BodyState(
            position=self._position,
            rotation=self._rotation,
            linear_velocity=self._linear_velocity,
            angular_velocity=self._angular_velocity,
        )

    def set_state(self, state: BodyState) -> None:
        """Set body state."""
        self._position = state.position
        self._rotation = state.rotation
        self._linear_velocity = state.linear_velocity
        self._angular_velocity = state.angular_velocity
        self._aabb_dirty = True

    def save_state(self) -> None:
        """Save current state for interpolation."""
        self._previous_state = self.get_state()

    def interpolate_state(self, alpha: float) -> BodyState:
        """
        Get interpolated state between previous and current.

        Args:
            alpha: Interpolation factor (0 = previous, 1 = current)

        Returns:
            Interpolated state
        """
        if self._previous_state is None:
            return self.get_state()

        prev = self._previous_state
        curr = self.get_state()

        # Linear interpolation for position
        pos = (
            prev.position[0] + alpha * (curr.position[0] - prev.position[0]),
            prev.position[1] + alpha * (curr.position[1] - prev.position[1]),
            prev.position[2] + alpha * (curr.position[2] - prev.position[2]),
        )

        # SLERP for rotation (simplified: nlerp)
        rot = (
            prev.rotation[0] + alpha * (curr.rotation[0] - prev.rotation[0]),
            prev.rotation[1] + alpha * (curr.rotation[1] - prev.rotation[1]),
            prev.rotation[2] + alpha * (curr.rotation[2] - prev.rotation[2]),
            prev.rotation[3] + alpha * (curr.rotation[3] - prev.rotation[3]),
        )
        rot = _quaternion_normalize(rot)

        return BodyState(
            position=pos,
            rotation=rot,
            linear_velocity=curr.linear_velocity,
            angular_velocity=curr.angular_velocity,
        )

    # =========================================================================
    # Integration (called by PhysicsWorld)
    # =========================================================================

    def integrate_velocities(self, dt: float, gravity: Vector3) -> None:
        """
        Integrate velocities from accumulated forces.

        Args:
            dt: Time step
            gravity: World gravity
        """
        if self._body_type != BodyType.DYNAMIC or self._is_sleeping:
            return

        # Apply gravity
        if self._flags.use_gravity:
            gravity_force = _vector_scale(gravity, self._mass)
            self._force_accumulator = _vector_add(self._force_accumulator, gravity_force)

        # Linear acceleration: a = F / m
        acceleration = _vector_scale(self._force_accumulator, self._inverse_mass)

        # Update linear velocity
        self._linear_velocity = _vector_add(
            self._linear_velocity,
            _vector_scale(acceleration, dt)
        )

        # Angular acceleration
        from .config import FLOAT_COMPARISON_EPSILON
        if _vector_length_sq(self._torque_accumulator) > FLOAT_COMPARISON_EPSILON * FLOAT_COMPARISON_EPSILON:
            # Transform torque to local space
            local_torque = _inverse_rotate_vector(self._torque_accumulator, self._rotation)

            # Compute angular acceleration: alpha = I^-1 * tau
            alpha_local = _matrix_vector_multiply(self._inverse_inertia_tensor, local_torque)

            # Gyroscopic torque (optional)
            if self._flags.enable_gyroscopic:
                omega_local = _inverse_rotate_vector(self._angular_velocity, self._rotation)
                inertia_omega = _matrix_vector_multiply(self._inertia_tensor, omega_local)
                gyro = _vector_cross(omega_local, inertia_omega)
                gyro_accel = _matrix_vector_multiply(self._inverse_inertia_tensor, gyro)
                alpha_local = _vector_sub(alpha_local, gyro_accel)

            # Transform back to world space
            alpha = _rotate_vector(alpha_local, self._rotation)

            # Update angular velocity
            self._angular_velocity = _vector_add(
                self._angular_velocity,
                _vector_scale(alpha, dt)
            )

        # Apply damping
        linear_damp = max(0.0, 1.0 - self._linear_damping * dt)
        angular_damp = max(0.0, 1.0 - self._angular_damping * dt)

        self._linear_velocity = _vector_scale(self._linear_velocity, linear_damp)
        self._angular_velocity = _vector_scale(self._angular_velocity, angular_damp)

        # Clamp velocities
        self._linear_velocity = _vector_clamp(self._linear_velocity, MAX_LINEAR_VELOCITY)
        self._angular_velocity = _vector_clamp(self._angular_velocity, MAX_ANGULAR_VELOCITY)

        # Clear accumulators
        self.clear_forces()

    def integrate_positions(self, dt: float) -> None:
        """
        Integrate positions from velocities.

        Args:
            dt: Time step
        """
        if self._body_type != BodyType.DYNAMIC or self._is_sleeping:
            return

        # Apply position lock
        mask = self._flags.get_position_lock_mask()
        effective_linear = (
            self._linear_velocity[0] * mask[0],
            self._linear_velocity[1] * mask[1],
            self._linear_velocity[2] * mask[2],
        )

        # Update position
        self._position = _vector_add(
            self._position,
            _vector_scale(effective_linear, dt)
        )

        # Apply rotation lock
        rot_mask = self._flags.get_rotation_lock_mask()
        effective_angular = (
            self._angular_velocity[0] * rot_mask[0],
            self._angular_velocity[1] * rot_mask[1],
            self._angular_velocity[2] * rot_mask[2],
        )

        # Update rotation
        from .config import FLOAT_COMPARISON_EPSILON
        angular_speed = _vector_length(effective_angular)
        if angular_speed > FLOAT_COMPARISON_EPSILON:
            angle = angular_speed * dt
            axis = _vector_normalize(effective_angular)
            half_angle = angle * 0.5
            s = math.sin(half_angle)
            c = math.cos(half_angle)

            dq: Quaternion = (axis[0] * s, axis[1] * s, axis[2] * s, c)
            self._rotation = _quaternion_normalize(
                _quaternion_multiply(dq, self._rotation)
            )

        self._aabb_dirty = True

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize body to dictionary."""
        return {
            'id': self._id,
            'name': self._name,
            'body_type': self._body_type.name,
            'position': self._position,
            'rotation': self._rotation,
            'linear_velocity': self._linear_velocity,
            'angular_velocity': self._angular_velocity,
            'mass': self._mass,
            'linear_damping': self._linear_damping,
            'angular_damping': self._angular_damping,
            'collision_layer': self._collision_layer,
            'collision_mask': self._collision_mask,
            'shape': self._shape.to_dict(),
            'material': self._material.to_dict(),
            'user_data': self._user_data,
        }

    def __repr__(self) -> str:
        return (
            f"RigidBody('{self._name}', type={self._body_type.name}, "
            f"pos={self._position}, mass={self._mass:.2f})"
        )
