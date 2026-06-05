"""
Pytest fixtures for constraint tests.
"""
import pytest
import math
import sys
import os

# Add engine path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from simulation.solver.constraint_solver import RigidBody, ConstraintType
from simulation.solver.config import SolverConfig


@pytest.fixture(autouse=True)
def clear_all_registries():
    """No-op: physics tests do not use Trinity metaclass registries."""
    yield


@pytest.fixture
def solver_config():
    """Default solver configuration."""
    return SolverConfig()


@pytest.fixture
def dynamic_body_a():
    """Create a dynamic body A at origin."""
    return RigidBody(
        id=1,
        position=Vec3(0, 0, 0),
        orientation=Quaternion.identity(),
        mass=1.0,
        local_inertia=Mat3.identity(),
    )


@pytest.fixture
def dynamic_body_b():
    """Create a dynamic body B offset from origin."""
    return RigidBody(
        id=2,
        position=Vec3(1, 0, 0),
        orientation=Quaternion.identity(),
        mass=1.0,
        local_inertia=Mat3.identity(),
    )


@pytest.fixture
def static_body():
    """Create a static body."""
    return RigidBody.create_static(
        id=0,
        position=Vec3(0, 0, 0),
        orientation=Quaternion.identity()
    )


@pytest.fixture
def heavy_body():
    """Create a heavy dynamic body."""
    return RigidBody(
        id=3,
        position=Vec3(2, 0, 0),
        orientation=Quaternion.identity(),
        mass=10.0,
        local_inertia=Mat3.diagonal(10.0, 10.0, 10.0),
    )


@pytest.fixture
def light_body():
    """Create a light dynamic body."""
    return RigidBody(
        id=4,
        position=Vec3(0, 1, 0),
        orientation=Quaternion.identity(),
        mass=0.1,
        local_inertia=Mat3.diagonal(0.1, 0.1, 0.1),
    )


@pytest.fixture
def rotated_body():
    """Create a body rotated 45 degrees around Y axis."""
    return RigidBody(
        id=5,
        position=Vec3(0, 0, 0),
        orientation=Quaternion.from_axis_angle(Vec3.unit_y(), math.pi / 4),
        mass=1.0,
        local_inertia=Mat3.identity(),
    )


@pytest.fixture
def body_with_velocity():
    """Create a body with initial velocity."""
    body = RigidBody(
        id=6,
        position=Vec3(0, 0, 0),
        orientation=Quaternion.identity(),
        mass=1.0,
        local_inertia=Mat3.identity(),
    )
    body.velocity = Vec3(1, 0, 0)
    body.angular_velocity = Vec3(0, 1, 0)
    return body


@pytest.fixture
def body_pair():
    """Create a pair of connected bodies."""
    body_a = RigidBody(
        id=10,
        position=Vec3(0, 0, 0),
        orientation=Quaternion.identity(),
        mass=1.0,
        local_inertia=Mat3.identity(),
    )
    body_b = RigidBody(
        id=11,
        position=Vec3(1, 0, 0),
        orientation=Quaternion.identity(),
        mass=1.0,
        local_inertia=Mat3.identity(),
    )
    return body_a, body_b


def make_body_at(position: Vec3, mass: float = 1.0) -> RigidBody:
    """Helper to create a body at a specific position."""
    import random
    return RigidBody(
        id=random.randint(100, 99999),
        position=position,
        orientation=Quaternion.identity(),
        mass=mass,
        local_inertia=Mat3.diagonal(mass, mass, mass) if mass > 0 else Mat3.zero(),
        is_static=(mass == 0),
    )
