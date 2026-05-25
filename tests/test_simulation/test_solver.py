"""
Comprehensive Tests for Constraint Solver Module.

Tests cover:
- Constraint solver convergence
- Joint types (Fixed, Hinge, Slider, Ball, Spring, Distance, D6)
- Joint limits and motors
- Island building
- Warm starting
- XPBD compliance
- TGS solver
"""

import pytest
import math
from typing import List, Tuple

# Import solver components
from engine.simulation.solver.config import (
    SolverConfig,
    SolverType,
    WarmStartMode,
    DEFAULT_VELOCITY_ITERATIONS,
    DEFAULT_POSITION_ITERATIONS,
    BAUMGARTE_FACTOR,
    SLOP,
    WARM_START_FACTOR,
    MAX_CORRECTION_VELOCITY,
    RELAXATION_FACTOR,
)
from engine.simulation.solver.jacobian import (
    Vec3,
    Mat3,
    Quaternion,
    Jacobian,
    compute_jacobian,
    compute_effective_mass,
    apply_impulse,
    compute_relative_velocity,
    compute_friction_basis,
    clamp_impulse,
    solve_2x2,
    solve_3x3,
)
from engine.simulation.solver.constraint_solver import (
    ConstraintSolver,
    RigidBody,
    ConstraintType,
    BaseConstraint,
    PointConstraint,
    AxisConstraint,
    AngularConstraint,
)
from engine.simulation.solver.tgs_solver import (
    TGSSolver,
    TGSContactSolver,
    TGSContactData,
)
from engine.simulation.solver.xpbd_solver import (
    XPBDSolver,
    XPBDParticle,
    XPBDDistanceConstraint,
    XPBDBendingConstraint,
    XPBDVolumeConstraint,
    XPBDCollisionConstraint,
    XPBDRigidBodyConstraint,
)
from engine.simulation.solver.island_manager import (
    Island,
    IslandManager,
    IslandState,
    UnionFind,
    ParallelIslandSolver,
)

# Import constraint components
from engine.simulation.constraints.joint_base import (
    Joint,
    JointState,
    JointBreakEvent,
)
from engine.simulation.constraints.joint_fixed import FixedJoint
from engine.simulation.constraints.joint_hinge import HingeJoint
from engine.simulation.constraints.joint_slider import SliderJoint
from engine.simulation.constraints.joint_ball import BallJoint
from engine.simulation.constraints.joint_spring import SpringJoint
from engine.simulation.constraints.joint_distance import DistanceJoint
from engine.simulation.constraints.joint_d6 import (
    D6Joint,
    D6MotionType,
    D6Axis,
)
from engine.simulation.constraints.joint_motors import (
    Motor,
    MotorMode,
    compute_motor_impulse,
    MotorController,
)
from engine.simulation.constraints.joint_limits import (
    LinearLimit,
    AngularLimit,
    SwingLimit,
    TwistLimit,
    LimitState,
    compute_limit_impulse,
)
from engine.simulation.constraints.contact_constraint import (
    ContactConstraint,
    ContactPoint,
    ContactManifold,
    compute_contact_jacobian,
    combine_friction,
    combine_restitution,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def solver_config():
    """Create default solver configuration."""
    return SolverConfig.default()


@pytest.fixture
def high_quality_config():
    """Create high quality solver configuration."""
    return SolverConfig.high_quality()


@pytest.fixture
def xpbd_config():
    """Create XPBD solver configuration."""
    return SolverConfig.xpbd_default()


@pytest.fixture
def tgs_config():
    """Create TGS solver configuration."""
    return SolverConfig.tgs_default()


@pytest.fixture
def dynamic_body():
    """Create a dynamic rigid body."""
    return RigidBody.create_dynamic(
        id=1,
        position=Vec3(0, 1, 0),
        mass=1.0,
        inertia=Mat3.from_diagonal(1, 1, 1)
    )


@pytest.fixture
def static_body():
    """Create a static rigid body."""
    return RigidBody.create_static(id=0, position=Vec3.zero())


@pytest.fixture
def two_bodies():
    """Create two connected dynamic bodies."""
    body_a = RigidBody.create_dynamic(
        id=1,
        position=Vec3(0, 1, 0),
        mass=1.0,
        inertia=Mat3.from_diagonal(1, 1, 1)
    )
    body_b = RigidBody.create_dynamic(
        id=2,
        position=Vec3(1, 1, 0),
        mass=1.0,
        inertia=Mat3.from_diagonal(1, 1, 1)
    )
    return body_a, body_b


# ============================================================================
# Vec3 Tests
# ============================================================================

class TestVec3:
    """Tests for Vec3 class."""

    def test_creation(self):
        v = Vec3(1, 2, 3)
        assert v.x == 1
        assert v.y == 2
        assert v.z == 3

    def test_zero(self):
        v = Vec3.zero()
        assert v.x == 0 and v.y == 0 and v.z == 0

    def test_unit_vectors(self):
        assert Vec3.unit_x().x == 1
        assert Vec3.unit_y().y == 1
        assert Vec3.unit_z().z == 1

    def test_addition(self):
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        c = a + b
        assert c.x == 5 and c.y == 7 and c.z == 9

    def test_subtraction(self):
        a = Vec3(5, 5, 5)
        b = Vec3(1, 2, 3)
        c = a - b
        assert c.x == 4 and c.y == 3 and c.z == 2

    def test_scalar_multiplication(self):
        v = Vec3(1, 2, 3)
        r = v * 2
        assert r.x == 2 and r.y == 4 and r.z == 6

    def test_dot_product(self):
        a = Vec3(1, 0, 0)
        b = Vec3(0, 1, 0)
        assert a.dot(b) == 0

        c = Vec3(1, 2, 3)
        d = Vec3(4, 5, 6)
        assert c.dot(d) == 32

    def test_cross_product(self):
        x = Vec3.unit_x()
        y = Vec3.unit_y()
        z = x.cross(y)
        assert abs(z.x) < 1e-10
        assert abs(z.y) < 1e-10
        assert abs(z.z - 1) < 1e-10

    def test_length(self):
        v = Vec3(3, 4, 0)
        assert abs(v.length() - 5) < 1e-10

    def test_normalized(self):
        v = Vec3(3, 4, 0)
        n = v.normalized()
        assert abs(n.length() - 1) < 1e-10

    def test_is_zero(self):
        assert Vec3.zero().is_zero()
        assert not Vec3(1, 0, 0).is_zero()


# ============================================================================
# Mat3 Tests
# ============================================================================

class TestMat3:
    """Tests for Mat3 class."""

    def test_identity(self):
        m = Mat3.identity()
        v = Vec3(1, 2, 3)
        r = m * v
        assert r.x == 1 and r.y == 2 and r.z == 3

    def test_diagonal(self):
        m = Mat3.from_diagonal(2, 3, 4)
        v = Vec3(1, 1, 1)
        r = m * v
        assert r.x == 2 and r.y == 3 and r.z == 4

    def test_transpose(self):
        m = Mat3(1, 2, 3, 4, 5, 6, 7, 8, 9)
        t = m.transpose()
        assert t.m01 == 4 and t.m10 == 2

    def test_determinant(self):
        m = Mat3.identity()
        assert abs(m.determinant() - 1) < 1e-10

    def test_inverse(self):
        m = Mat3.from_diagonal(2, 2, 2)
        inv = m.inverse()
        result = m.mat_mul(inv)
        # Should be close to identity
        assert abs(result.m00 - 1) < 1e-10
        assert abs(result.m11 - 1) < 1e-10
        assert abs(result.m22 - 1) < 1e-10

    def test_skew_symmetric(self):
        v = Vec3(1, 2, 3)
        skew = Mat3.skew_symmetric(v)
        # v cross u = skew * u
        u = Vec3(4, 5, 6)
        cross_result = v.cross(u)
        skew_result = skew * u
        assert abs(cross_result.x - skew_result.x) < 1e-10
        assert abs(cross_result.y - skew_result.y) < 1e-10
        assert abs(cross_result.z - skew_result.z) < 1e-10


# ============================================================================
# Quaternion Tests
# ============================================================================

class TestQuaternion:
    """Tests for Quaternion class."""

    def test_identity(self):
        q = Quaternion.identity()
        assert q.w == 1 and q.x == 0 and q.y == 0 and q.z == 0

    def test_from_axis_angle(self):
        q = Quaternion.from_axis_angle(Vec3.unit_z(), math.pi / 2)
        # Should rotate X axis to Y axis
        v = q.rotate_vector(Vec3.unit_x())
        assert abs(v.x) < 1e-10
        assert abs(v.y - 1) < 1e-10
        assert abs(v.z) < 1e-10

    def test_multiplication(self):
        q1 = Quaternion.from_axis_angle(Vec3.unit_z(), math.pi / 2)
        q2 = Quaternion.from_axis_angle(Vec3.unit_z(), math.pi / 2)
        q3 = q1 * q2
        # Should be 180 degree rotation
        v = q3.rotate_vector(Vec3.unit_x())
        assert abs(v.x + 1) < 1e-10

    def test_conjugate(self):
        q = Quaternion.from_axis_angle(Vec3.unit_z(), math.pi / 2)
        q_conj = q.conjugate()
        result = q * q_conj
        assert abs(result.w - 1) < 1e-10

    def test_to_matrix(self):
        q = Quaternion.from_axis_angle(Vec3.unit_z(), math.pi / 2)
        m = q.to_matrix()
        v = m * Vec3.unit_x()
        assert abs(v.x) < 1e-10
        assert abs(v.y - 1) < 1e-10


# ============================================================================
# Jacobian Tests
# ============================================================================

class TestJacobian:
    """Tests for Jacobian computation."""

    def test_jacobian_creation(self):
        j = Jacobian(
            linear_a=Vec3.unit_x(),
            angular_a=Vec3.unit_y(),
            linear_b=Vec3.unit_x(),
            angular_b=Vec3.unit_z()
        )
        assert j.linear_a.x == 1

    def test_compute_velocity(self):
        j = Jacobian(
            linear_a=Vec3.unit_x(),
            angular_a=Vec3.zero(),
            linear_b=Vec3(-1, 0, 0),
            angular_b=Vec3.zero()
        )
        vel = j.compute_velocity(
            Vec3(1, 0, 0), Vec3.zero(),
            Vec3(1, 0, 0), Vec3.zero()
        )
        assert abs(vel) < 1e-10  # Same velocity, zero relative

    def test_compute_jacobian_point(self):
        j = compute_jacobian(
            "point",
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            r_a=Vec3.zero(),
            r_b=Vec3.zero()
        )
        assert j.linear_b.x == 1

    def test_compute_jacobian_contact(self):
        j = compute_jacobian(
            "contact",
            Vec3.zero(),
            Vec3.zero(),
            axis=Vec3.unit_y(),
            r_a=Vec3.zero(),
            r_b=Vec3.zero()
        )
        assert j.linear_b.y == 1

    def test_compute_effective_mass(self):
        j = Jacobian(
            linear_a=Vec3.unit_x(),
            angular_a=Vec3.zero(),
            linear_b=Vec3(-1, 0, 0),
            angular_b=Vec3.zero()
        )
        mass = compute_effective_mass(
            j,
            inv_mass_a=1.0,
            inv_inertia_a=Mat3.identity(),
            inv_mass_b=1.0,
            inv_inertia_b=Mat3.identity()
        )
        assert mass > 0

    def test_apply_impulse(self):
        j = Jacobian(
            linear_a=Vec3.unit_x(),
            angular_a=Vec3.zero(),
            linear_b=Vec3(-1, 0, 0),
            angular_b=Vec3.zero()
        )
        dv_a, dw_a, dv_b, dw_b = apply_impulse(
            j,
            impulse=1.0,
            inv_mass_a=1.0,
            inv_inertia_a=Mat3.identity(),
            inv_mass_b=1.0,
            inv_inertia_b=Mat3.identity()
        )
        assert dv_a.x == 1.0
        assert dv_b.x == -1.0

    def test_friction_basis(self):
        normal = Vec3.unit_y()
        t1, t2 = compute_friction_basis(normal)
        # Tangents should be perpendicular to normal
        assert abs(t1.dot(normal)) < 1e-10
        assert abs(t2.dot(normal)) < 1e-10
        # Tangents should be perpendicular to each other
        assert abs(t1.dot(t2)) < 1e-10

    def test_clamp_impulse(self):
        impulse, acc = clamp_impulse(5.0, 0.0, 0.0, 3.0)
        assert impulse == 3.0
        assert acc == 3.0

        impulse, acc = clamp_impulse(-5.0, 0.0, -2.0, 10.0)
        assert impulse == -2.0
        assert acc == -2.0


# ============================================================================
# Solver Config Tests
# ============================================================================

class TestSolverConfig:
    """Tests for SolverConfig."""

    def test_default_values(self):
        config = SolverConfig.default()
        assert config.velocity_iterations == DEFAULT_VELOCITY_ITERATIONS
        assert config.position_iterations == DEFAULT_POSITION_ITERATIONS

    def test_validate(self):
        config = SolverConfig()
        config.validate()  # Should not raise

        config.velocity_iterations = 0
        with pytest.raises(ValueError):
            config.validate()

    def test_presets(self):
        high_quality = SolverConfig.high_quality()
        assert high_quality.velocity_iterations > DEFAULT_VELOCITY_ITERATIONS

        performance = SolverConfig.performance()
        assert performance.velocity_iterations < DEFAULT_VELOCITY_ITERATIONS

        xpbd = SolverConfig.xpbd_default()
        assert xpbd.solver_type == SolverType.XPBD

    def test_with_iterations(self):
        config = SolverConfig.default()
        new_config = config.with_iterations(velocity=16, position=8)
        assert new_config.velocity_iterations == 16
        assert new_config.position_iterations == 8


# ============================================================================
# RigidBody Tests
# ============================================================================

class TestRigidBody:
    """Tests for RigidBody."""

    def test_create_static(self):
        body = RigidBody.create_static(id=0)
        assert body.is_static
        assert body.inv_mass == 0

    def test_create_dynamic(self):
        body = RigidBody.create_dynamic(id=1, mass=2.0)
        assert not body.is_static
        assert body.inv_mass == 0.5

    def test_local_to_world(self, dynamic_body):
        local_point = Vec3(1, 0, 0)
        world_point = dynamic_body.local_to_world(local_point)
        assert world_point.x == 1
        assert world_point.y == 1  # body at y=1

    def test_world_to_local(self, dynamic_body):
        world_point = Vec3(1, 1, 0)
        local_point = dynamic_body.world_to_local(world_point)
        assert local_point.x == 1
        assert local_point.y == 0

    def test_apply_impulse(self, dynamic_body):
        dynamic_body.apply_impulse(Vec3(1, 0, 0), Vec3.zero())
        assert dynamic_body.velocity.x == 1.0

    def test_get_velocity_at_point(self, dynamic_body):
        dynamic_body.velocity = Vec3(1, 0, 0)
        dynamic_body.angular_velocity = Vec3(0, 0, 1)
        vel = dynamic_body.get_velocity_at_point(Vec3(0, 2, 0))
        # Angular contribution: omega cross r = (0,0,1) x (0,1,0) = (-1,0,0)
        # Total: (1,0,0) + (-1,0,0) = (0,0,0)
        assert abs(vel.x) < 1e-10


# ============================================================================
# Constraint Solver Tests
# ============================================================================

class TestConstraintSolver:
    """Tests for ConstraintSolver."""

    def test_create_solver(self, solver_config):
        solver = ConstraintSolver(solver_config)
        assert solver.config == solver_config

    def test_add_remove_body(self, solver_config, dynamic_body):
        solver = ConstraintSolver(solver_config)
        solver.add_body(dynamic_body)
        assert dynamic_body.id in solver.bodies
        solver.remove_body(dynamic_body.id)
        assert dynamic_body.id not in solver.bodies

    def test_solve_empty(self, solver_config):
        solver = ConstraintSolver(solver_config)
        solver.solve(1.0 / 60)  # Should not raise

    def test_point_constraint_convergence(self, two_bodies, solver_config):
        body_a, body_b = two_bodies
        solver = ConstraintSolver(solver_config)
        solver.add_body(body_a)
        solver.add_body(body_b)

        # Create point constraint
        constraint = PointConstraint(
            _body_a=body_a,
            _body_b=body_b,
            local_anchor_a=Vec3(0.5, 0, 0),
            local_anchor_b=Vec3(-0.5, 0, 0)
        )
        solver.add_constraint(constraint)

        # Initial anchor positions should be apart
        anchor_a = body_a.local_to_world(constraint.local_anchor_a)
        anchor_b = body_b.local_to_world(constraint.local_anchor_b)
        initial_error = (anchor_b - anchor_a).length()

        # Solve with multiple iterations to allow convergence
        dt = 1.0 / 60
        for _ in range(20):  # Run enough iterations for convergence
            solver.solve(dt)

        # Verify constraint satisfaction: anchors should be closer
        final_anchor_a = body_a.local_to_world(constraint.local_anchor_a)
        final_anchor_b = body_b.local_to_world(constraint.local_anchor_b)
        final_error = (final_anchor_b - final_anchor_a).length()

        # Error should decrease significantly after iterations
        assert final_error < initial_error, f"Constraint did not converge: initial={initial_error}, final={final_error}"

    def test_velocity_only_solve(self, two_bodies, solver_config):
        body_a, body_b = two_bodies
        solver = ConstraintSolver(solver_config)
        solver.add_body(body_a)
        solver.add_body(body_b)

        constraint = PointConstraint(
            _body_a=body_a,
            _body_b=body_b,
            local_anchor_a=Vec3.zero(),
            local_anchor_b=Vec3.zero()
        )
        solver.add_constraint(constraint)
        solver.solve_velocity_constraints(10)

    def test_position_only_solve(self, two_bodies, solver_config):
        body_a, body_b = two_bodies
        solver = ConstraintSolver(solver_config)
        solver.add_body(body_a)
        solver.add_body(body_b)

        constraint = PointConstraint(
            _body_a=body_a,
            _body_b=body_b,
            local_anchor_a=Vec3.zero(),
            local_anchor_b=Vec3.zero()
        )
        solver.add_constraint(constraint)
        constraint.prepare(1.0/60, solver_config)
        error = solver.solve_position_constraints(10)
        assert error >= 0


# ============================================================================
# TGS Solver Tests
# ============================================================================

class TestTGSSolver:
    """Tests for TGS Solver."""

    def test_create_tgs_solver(self, tgs_config):
        solver = TGSSolver(tgs_config)
        assert solver.config.solver_type == SolverType.TEMPORAL_GAUSS_SEIDEL

    def test_tgs_substeps(self, two_bodies, tgs_config):
        body_a, body_b = two_bodies
        solver = TGSSolver(tgs_config)
        solver.add_body(body_a)
        solver.add_body(body_b)

        constraint = PointConstraint(
            _body_a=body_a,
            _body_b=body_b,
            local_anchor_a=Vec3.zero(),
            local_anchor_b=Vec3.zero()
        )
        solver.add_constraint(constraint)

        dt = 1.0 / 60
        solver.solve(dt)

        assert solver.substep_dt == dt / tgs_config.substeps

    def test_tgs_regularization(self, tgs_config):
        solver = TGSSolver(tgs_config)
        solver.set_regularization(0.5)
        assert solver._regularization_factor == 0.5

    def test_tgs_contact_solver(self, dynamic_body, static_body, tgs_config):
        contact_solver = TGSContactSolver(tgs_config)

        contact = TGSContactData(
            body_a=dynamic_body,
            body_b=static_body,
            point=Vec3(0, 0.5, 0),
            normal=Vec3.unit_y(),
            penetration=0.1,
            friction=0.4,
            restitution=0.0
        )
        contact_solver.add_contact(contact)
        contact_solver.solve_contacts(1.0 / 60)


# ============================================================================
# XPBD Solver Tests
# ============================================================================

class TestXPBDSolver:
    """Tests for XPBD Solver."""

    def test_create_xpbd_solver(self, xpbd_config):
        solver = XPBDSolver(xpbd_config)
        assert solver.config.solver_type == SolverType.XPBD

    def test_xpbd_particle(self):
        particle = XPBDParticle(
            id=0,
            position=Vec3(0, 1, 0),
            inv_mass=1.0
        )
        assert particle.position.y == 1

    def test_xpbd_particle_predict(self):
        particle = XPBDParticle(
            id=0,
            position=Vec3(0, 0, 0),
            velocity=Vec3(1, 0, 0),
            inv_mass=1.0
        )
        particle.predict_position(1.0)
        assert particle.position.x == 1

    def test_xpbd_particle_predict_gravity(self):
        particle = XPBDParticle(
            id=0,
            position=Vec3(0, 10, 0),
            velocity=Vec3.zero(),
            inv_mass=1.0
        )
        gravity = Vec3(0, -10, 0)
        particle.predict_position(0.1, gravity)
        assert particle.velocity.y < 0

    def test_xpbd_distance_constraint(self):
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(2, 0, 0), inv_mass=1.0)

        constraint = XPBDDistanceConstraint(
            particle_a=p1,
            particle_b=p2,
            rest_length=1.0,
            compliance=0.0
        )

        # Constraint value should be distance - rest_length
        c = constraint.compute_constraint()
        assert abs(c - 1.0) < 1e-10

    def test_xpbd_distance_solve(self):
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(2, 0, 0), inv_mass=1.0)

        constraint = XPBDDistanceConstraint(
            particle_a=p1,
            particle_b=p2,
            rest_length=1.0,
            compliance=0.0
        )

        # Solve should bring particles closer
        initial_dist = (p2.position - p1.position).length()
        constraint.solve(1.0 / 60)
        final_dist = (p2.position - p1.position).length()

        assert final_dist < initial_dist

    def test_xpbd_compliance(self):
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(2, 0, 0), inv_mass=1.0)

        # Higher compliance = softer
        soft_constraint = XPBDDistanceConstraint(
            particle_a=p1,
            particle_b=p2,
            rest_length=1.0,
            compliance=0.1
        )

        p1_soft = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2_soft = XPBDParticle(id=1, position=Vec3(2, 0, 0), inv_mass=1.0)
        soft_constraint2 = XPBDDistanceConstraint(
            particle_a=p1_soft,
            particle_b=p2_soft,
            rest_length=1.0,
            compliance=0.1
        )

        soft_constraint2.solve(1.0 / 60)
        # Soft constraint should converge slower
        # (harder to test precisely without full simulation)

    def test_xpbd_bending_constraint(self):
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(1, 0, 0), inv_mass=1.0)
        p3 = XPBDParticle(id=2, position=Vec3(2, 1, 0), inv_mass=1.0)  # Bent

        constraint = XPBDBendingConstraint(
            particle_a=p1,
            particle_b=p2,
            particle_c=p3,
            rest_angle=math.pi,  # Straight
            compliance=0.001
        )

        c = constraint.compute_constraint()
        assert c != 0  # Should have error since bent

    def test_xpbd_collision_constraint(self):
        particle = XPBDParticle(
            id=0,
            position=Vec3(0, -0.1, 0),
            inv_mass=1.0
        )

        collision = XPBDCollisionConstraint(
            particle=particle,
            contact_point=Vec3(0, 0, 0),
            contact_normal=Vec3.unit_y(),
            penetration=0.1,
            friction=0.4
        )

        collision.solve(1.0 / 60)
        # Particle should be pushed up
        assert particle.position.y >= -0.1

    def test_xpbd_solver_integration(self, xpbd_config):
        solver = XPBDSolver(xpbd_config)

        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=0.0, is_static=True)
        p2 = XPBDParticle(id=1, position=Vec3(0, -2, 0), inv_mass=1.0)

        solver.add_particle(p1)
        solver.add_particle(p2)

        constraint = XPBDDistanceConstraint(
            particle_a=p1,
            particle_b=p2,
            rest_length=1.0,
            compliance=0.0
        )
        solver.add_constraint(constraint)

        solver.solve(1.0 / 60)

    def test_xpbd_kinetic_energy(self, xpbd_config):
        solver = XPBDSolver(xpbd_config)

        p = XPBDParticle(
            id=0,
            position=Vec3(0, 0, 0),
            velocity=Vec3(1, 0, 0),
            inv_mass=1.0
        )
        solver.add_particle(p)

        energy = solver.get_kinetic_energy()
        assert abs(energy - 0.5) < 1e-10


# ============================================================================
# Island Manager Tests
# ============================================================================

class TestIslandManager:
    """Tests for Island Manager."""

    def test_create_island_manager(self):
        manager = IslandManager()
        assert len(manager.islands) == 0

    def test_union_find(self):
        uf = UnionFind()
        uf.make_set(1)
        uf.make_set(2)
        uf.make_set(3)

        assert not uf.connected(1, 2)
        uf.union(1, 2)
        assert uf.connected(1, 2)
        assert not uf.connected(1, 3)

    def test_build_islands(self, two_bodies, solver_config):
        body_a, body_b = two_bodies
        manager = IslandManager()

        constraint = PointConstraint(
            _body_a=body_a,
            _body_b=body_b,
            local_anchor_a=Vec3.zero(),
            local_anchor_b=Vec3.zero()
        )

        islands = manager.build_islands([body_a, body_b], [constraint])

        # Should have one island with both bodies
        assert len(islands) == 1
        assert len(islands[0].bodies) == 2

    def test_separate_islands(self, solver_config):
        body_a = RigidBody.create_dynamic(id=1, position=Vec3(0, 0, 0), mass=1.0)
        body_b = RigidBody.create_dynamic(id=2, position=Vec3(10, 0, 0), mass=1.0)

        manager = IslandManager()
        islands = manager.build_islands([body_a, body_b], [])

        # Should have two separate islands
        assert len(islands) == 2

    def test_island_sleeping(self):
        manager = IslandManager(
            sleep_velocity_threshold=0.05,
            sleep_time_threshold=0.5
        )

        body = RigidBody.create_dynamic(id=1, position=Vec3.zero(), mass=1.0)
        body.velocity = Vec3.zero()
        body.angular_velocity = Vec3.zero()

        manager.build_islands([body], [])

        # Simulate time passing with body at rest
        for _ in range(10):
            sleeping, awakened = manager.update_sleeping(0.1)

        assert len(manager.get_sleeping_islands()) > 0 or len(manager.islands) == 0

    def test_wake_island(self):
        manager = IslandManager()
        body = RigidBody.create_dynamic(id=1, position=Vec3.zero(), mass=1.0)
        manager.build_islands([body], [])

        # Get island
        if manager.islands:
            island_id = list(manager.islands.keys())[0]
            manager.islands[island_id].put_to_sleep()
            assert manager.islands[island_id].is_sleeping()

            manager.wake_island(island_id)
            assert manager.islands[island_id].is_active()

    def test_parallel_groups(self):
        manager = IslandManager()

        bodies = [
            RigidBody.create_dynamic(id=i, position=Vec3(i * 10, 0, 0), mass=1.0)
            for i in range(8)
        ]
        manager.build_islands(bodies, [])

        groups = manager.get_parallel_groups(max_groups=4)
        # Should distribute islands across groups
        assert len(groups) <= 4

    def test_island_statistics(self):
        manager = IslandManager()
        body = RigidBody.create_dynamic(id=1, position=Vec3.zero(), mass=1.0)
        manager.build_islands([body], [])

        stats = manager.get_statistics()
        assert "total_islands" in stats
        assert "active_islands" in stats


# ============================================================================
# Fixed Joint Tests
# ============================================================================

class TestFixedJoint:
    """Tests for Fixed Joint."""

    def test_create_fixed_joint(self, two_bodies):
        body_a, body_b = two_bodies
        joint = FixedJoint(body_a, body_b)
        assert joint.body_a == body_a
        assert joint.body_b == body_b
        assert joint.get_constraint_count() == 6

    def test_fixed_joint_weld(self, two_bodies):
        body_a, body_b = two_bodies
        joint = FixedJoint.create_weld(body_a, body_b)
        assert joint.get_constraint_count() == 6

    def test_fixed_joint_soft(self, two_bodies):
        body_a, body_b = two_bodies
        joint = FixedJoint.create_soft_weld(
            body_a, body_b,
            stiffness=1000.0,
            damping=10.0
        )
        assert joint.stiffness == 1000.0
        assert joint.damping == 10.0

    def test_fixed_joint_prepare(self, two_bodies, solver_config):
        body_a, body_b = two_bodies
        joint = FixedJoint(body_a, body_b)
        joint.prepare(1.0 / 60, solver_config)

    def test_fixed_joint_break(self, two_bodies):
        body_a, body_b = two_bodies
        joint = FixedJoint(body_a, body_b, break_force=100.0)

        broken = False
        def on_break(event):
            nonlocal broken
            broken = True

        joint.set_break_callback(on_break)
        joint._break(150.0, 0.0)
        assert joint.is_broken
        assert broken


# ============================================================================
# Hinge Joint Tests
# ============================================================================

class TestHingeJoint:
    """Tests for Hinge Joint."""

    def test_create_hinge_joint(self, two_bodies):
        body_a, body_b = two_bodies
        joint = HingeJoint(body_a, body_b)
        assert joint.body_a == body_a

    def test_hinge_joint_axis(self, two_bodies):
        body_a, body_b = two_bodies
        axis = Vec3(0, 1, 0)
        joint = HingeJoint(
            body_a, body_b,
            local_axis_a=axis,
            local_axis_b=axis
        )
        assert joint.local_axis_a.y == 1

    def test_hinge_joint_limits(self, two_bodies):
        body_a, body_b = two_bodies
        joint = HingeJoint(body_a, body_b)
        joint.set_limits(-math.pi / 4, math.pi / 4)
        assert joint.limits_enabled
        assert joint.min_angle == -math.pi / 4
        assert joint.max_angle == math.pi / 4

    def test_hinge_joint_motor_velocity(self, two_bodies):
        body_a, body_b = two_bodies
        joint = HingeJoint(body_a, body_b)
        joint.set_motor_speed(1.0, 10.0)
        assert joint.motor_enabled
        assert joint.motor.mode == MotorMode.VELOCITY
        assert joint.motor.target == 1.0

    def test_hinge_joint_motor_position(self, two_bodies):
        body_a, body_b = two_bodies
        joint = HingeJoint(body_a, body_b)
        joint.set_motor_position(math.pi / 2, 10.0)
        assert joint.motor.mode == MotorMode.POSITION

    def test_hinge_joint_angle(self, two_bodies):
        body_a, body_b = two_bodies
        joint = HingeJoint(body_a, body_b)
        angle = joint.get_current_angle()
        assert isinstance(angle, float)

    def test_hinge_joint_speed(self, two_bodies):
        body_a, body_b = two_bodies
        joint = HingeJoint(body_a, body_b)
        speed = joint.get_joint_speed()
        assert speed == 0  # No angular velocity

    def test_hinge_joint_create_at_point(self, two_bodies):
        body_a, body_b = two_bodies
        joint = HingeJoint.create_at_point(
            body_a, body_b,
            Vec3(0.5, 1, 0),
            Vec3.unit_y(),
            min_angle=-1.0,
            max_angle=1.0
        )
        assert joint.limits_enabled


# ============================================================================
# Slider Joint Tests
# ============================================================================

class TestSliderJoint:
    """Tests for Slider Joint."""

    def test_create_slider_joint(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SliderJoint(body_a, body_b)
        assert joint.body_a == body_a

    def test_slider_joint_axis(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SliderJoint(body_a, body_b, local_axis_a=Vec3.unit_x())
        assert joint.local_axis_a.x == 1

    def test_slider_joint_limits(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SliderJoint(body_a, body_b)
        joint.set_limits(-1.0, 1.0)
        assert joint.limits_enabled
        assert joint.min_distance == -1.0
        assert joint.max_distance == 1.0

    def test_slider_joint_motor(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SliderJoint(body_a, body_b)
        joint.set_motor_speed(2.0, 50.0)
        assert joint.motor_enabled
        assert joint.motor.target == 2.0

    def test_slider_joint_translation(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SliderJoint(body_a, body_b)
        trans = joint.get_current_translation()
        assert isinstance(trans, float)


# ============================================================================
# Ball Joint Tests
# ============================================================================

class TestBallJoint:
    """Tests for Ball Joint."""

    def test_create_ball_joint(self, two_bodies):
        body_a, body_b = two_bodies
        joint = BallJoint(body_a, body_b)
        assert joint.get_constraint_count() == 3

    def test_ball_joint_cone_limit(self, two_bodies):
        body_a, body_b = two_bodies
        joint = BallJoint(body_a, body_b)
        joint.set_cone_limit(math.pi / 4)
        assert joint.cone_limit_enabled
        assert joint.cone_limit_angle == math.pi / 4

    def test_ball_joint_twist_limit(self, two_bodies):
        body_a, body_b = two_bodies
        joint = BallJoint(body_a, body_b)
        joint.set_twist_limits(-math.pi / 2, math.pi / 2)
        assert joint.twist_limit_enabled

    def test_ball_joint_swing_angle(self, two_bodies):
        body_a, body_b = two_bodies
        joint = BallJoint(body_a, body_b)
        angle = joint.get_swing_angle()
        # Swing angle should be non-negative (it's a magnitude)
        assert angle >= 0
        # For aligned bodies, swing angle should be near zero
        assert angle < 0.1, f"Expected near-zero swing angle for aligned bodies, got {angle}"

    def test_ball_joint_twist_angle(self, two_bodies):
        body_a, body_b = two_bodies
        joint = BallJoint(body_a, body_b)
        angle = joint.get_twist_angle()
        assert isinstance(angle, float)

    def test_ball_joint_create_at_point(self, two_bodies):
        body_a, body_b = two_bodies
        joint = BallJoint.create_at_point(
            body_a, body_b,
            Vec3(0.5, 1, 0),
            cone_limit=0.5
        )
        assert joint.cone_limit_enabled


# ============================================================================
# Spring Joint Tests
# ============================================================================

class TestSpringJoint:
    """Tests for Spring Joint."""

    def test_create_spring_joint(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SpringJoint(body_a, body_b)
        assert joint.rest_length > 0

    def test_spring_joint_parameters(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SpringJoint(
            body_a, body_b,
            stiffness=200.0,
            damping=5.0
        )
        assert joint.stiffness == 200.0
        assert joint.damping == 5.0

    def test_spring_joint_rest_length(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SpringJoint(body_a, body_b, rest_length=2.0)
        assert joint.rest_length == 2.0

    def test_spring_joint_frequency_damping(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SpringJoint(body_a, body_b)
        joint.set_frequency_damping(frequency=5.0, damping_ratio=0.5)
        assert joint.frequency == 5.0
        assert joint.damping_ratio == 0.5

    def test_spring_joint_length_limits(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SpringJoint(body_a, body_b)
        joint.set_length_limits(min_length=0.5, max_length=2.0)
        assert joint.min_length == 0.5
        assert joint.max_length == 2.0

    def test_spring_joint_bungee(self, two_bodies):
        body_a, body_b = two_bodies
        joint = SpringJoint.create_bungee(
            body_a, body_b,
            Vec3(0, 1, 0),
            Vec3(1, 1, 0)
        )
        assert joint.min_length is not None


# ============================================================================
# Distance Joint Tests
# ============================================================================

class TestDistanceJoint:
    """Tests for Distance Joint."""

    def test_create_distance_joint(self, two_bodies):
        body_a, body_b = two_bodies
        joint = DistanceJoint(body_a, body_b)
        assert joint.rest_length > 0

    def test_distance_joint_explicit_length(self, two_bodies):
        body_a, body_b = two_bodies
        joint = DistanceJoint(body_a, body_b, rest_length=5.0)
        assert joint.rest_length == 5.0

    def test_distance_joint_rope(self, two_bodies):
        body_a, body_b = two_bodies
        joint = DistanceJoint(body_a, body_b)
        joint.set_as_rope(max_length=3.0)
        assert joint.max_distance == 3.0
        assert joint._constraint_mode == "max"

    def test_distance_joint_rod(self, two_bodies):
        body_a, body_b = two_bodies
        joint = DistanceJoint(body_a, body_b)
        joint.set_as_rod(length=2.0)
        assert joint.rest_length == 2.0
        assert joint._constraint_mode == "equality"

    def test_distance_joint_range(self, two_bodies):
        body_a, body_b = two_bodies
        joint = DistanceJoint(body_a, body_b)
        joint.set_range(0.5, 2.0)
        assert joint.min_distance == 0.5
        assert joint.max_distance == 2.0


# ============================================================================
# D6 Joint Tests
# ============================================================================

class TestD6Joint:
    """Tests for D6 Joint."""

    def test_create_d6_joint(self, two_bodies):
        body_a, body_b = two_bodies
        joint = D6Joint(body_a, body_b)
        # Default: all locked
        assert joint.get_motion(D6Axis.LINEAR_X) == D6MotionType.LOCKED

    def test_d6_joint_motion_types(self, two_bodies):
        body_a, body_b = two_bodies
        joint = D6Joint(body_a, body_b)

        joint.set_motion(D6Axis.LINEAR_X, D6MotionType.FREE)
        assert joint.get_motion(D6Axis.LINEAR_X) == D6MotionType.FREE

        joint.set_motion(D6Axis.ANGULAR_X, D6MotionType.LIMITED)
        assert joint.get_motion(D6Axis.ANGULAR_X) == D6MotionType.LIMITED

    def test_d6_joint_linear_limit(self, two_bodies):
        body_a, body_b = two_bodies
        joint = D6Joint(body_a, body_b)
        joint.set_linear_limit(D6Axis.LINEAR_X, -1.0, 1.0)
        assert joint.get_motion(D6Axis.LINEAR_X) == D6MotionType.LIMITED

    def test_d6_joint_angular_limit(self, two_bodies):
        body_a, body_b = two_bodies
        joint = D6Joint(body_a, body_b)
        joint.set_angular_limit(D6Axis.ANGULAR_X, -0.5, 0.5)
        assert joint.get_motion(D6Axis.ANGULAR_X) == D6MotionType.LIMITED

    def test_d6_joint_swing_cone(self, two_bodies):
        body_a, body_b = two_bodies
        joint = D6Joint(body_a, body_b)
        joint.set_swing_cone_limit(math.pi / 4)
        assert joint._use_swing_cone

    def test_d6_joint_twist_limit(self, two_bodies):
        body_a, body_b = two_bodies
        joint = D6Joint(body_a, body_b)
        joint.set_twist_limit(-1.0, 1.0)
        assert joint.get_motion(D6Axis.ANGULAR_X) == D6MotionType.LIMITED

    def test_d6_joint_motor(self, two_bodies):
        body_a, body_b = two_bodies
        joint = D6Joint(body_a, body_b)
        joint.set_motor(D6Axis.LINEAR_X, MotorMode.VELOCITY, 1.0, 100.0)
        assert joint._axis_config[D6Axis.LINEAR_X].motor_enabled

    def test_d6_joint_lock_free_all(self, two_bodies):
        body_a, body_b = two_bodies
        joint = D6Joint(body_a, body_b)

        joint.free_all()
        for axis in D6Axis:
            assert joint.get_motion(axis) == D6MotionType.FREE

        joint.lock_all()
        for axis in D6Axis:
            assert joint.get_motion(axis) == D6MotionType.LOCKED

    def test_d6_joint_presets(self, two_bodies):
        body_a, body_b = two_bodies

        fixed = D6Joint.create_fixed(body_a, body_b)
        for axis in D6Axis:
            assert fixed.get_motion(axis) == D6MotionType.LOCKED


# ============================================================================
# Motor Tests
# ============================================================================

class TestMotor:
    """Tests for Motor helpers."""

    def test_motor_velocity_mode(self):
        motor = Motor(MotorMode.VELOCITY, target=5.0, max_force=100.0)
        assert motor.mode == MotorMode.VELOCITY
        assert motor.target == 5.0

    def test_motor_position_mode(self):
        motor = Motor(MotorMode.POSITION, target=1.0, max_force=50.0)
        assert motor.mode == MotorMode.POSITION

    def test_motor_set_methods(self):
        motor = Motor()
        motor.set_velocity_target(10.0, 200.0)
        assert motor.mode == MotorMode.VELOCITY
        assert motor.target == 10.0
        assert motor.max_force == 200.0

        motor.set_position_target(0.5, 150.0)
        assert motor.mode == MotorMode.POSITION

    def test_motor_gains(self):
        motor = Motor()
        motor.set_gains(position_gain=100.0, velocity_gain=10.0)
        assert motor.position_gain == 100.0
        assert motor.velocity_gain == 10.0

    def test_compute_motor_impulse(self):
        motor = Motor(MotorMode.VELOCITY, target=1.0, max_force=100.0)
        impulse = compute_motor_impulse(
            motor,
            current_value=0.0,
            current_velocity=0.0,
            effective_mass=1.0,
            dt=1.0 / 60
        )
        assert impulse != 0

    def test_motor_controller(self):
        controller = MotorController(max_force=100.0, max_velocity=10.0)
        controller.set_position_target(1.0)
        force = controller.compute_force(0.0, 0.0, 1.0 / 60)
        assert abs(force) <= 100.0


# ============================================================================
# Limit Tests
# ============================================================================

class TestLimits:
    """Tests for Limit helpers."""

    def test_linear_limit(self):
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert limit.range == 2.0
        assert limit.center == 0.0

    def test_linear_limit_state(self):
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert limit.check_state(0.0) == LimitState.INACTIVE
        assert limit.check_state(-2.0) == LimitState.AT_LOWER
        assert limit.check_state(2.0) == LimitState.AT_UPPER

    def test_linear_limit_boundary_activation(self):
        """Test that limits activate near boundaries (within contact_distance)."""
        limit = LinearLimit(lower=-1.0, upper=1.0, contact_distance=0.01)
        # Exactly at boundary should be active
        assert limit.check_state(-1.0) == LimitState.AT_LOWER
        assert limit.check_state(1.0) == LimitState.AT_UPPER
        # Just inside contact_distance should be active
        assert limit.check_state(-0.995) == LimitState.AT_LOWER
        assert limit.check_state(0.995) == LimitState.AT_UPPER
        # Just outside contact_distance should be inactive
        assert limit.check_state(-0.98) == LimitState.INACTIVE
        assert limit.check_state(0.98) == LimitState.INACTIVE

    def test_linear_limit_error(self):
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert limit.compute_error(0.0) == 0.0
        assert limit.compute_error(-2.0) == 1.0
        assert limit.compute_error(2.0) == 1.0

    def test_linear_limit_clamp(self):
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert limit.clamp(0.0) == 0.0
        assert limit.clamp(-5.0) == -1.0
        assert limit.clamp(5.0) == 1.0

    def test_angular_limit(self):
        limit = AngularLimit(lower=-math.pi / 2, upper=math.pi / 2)
        assert limit.check_state(0.0) == LimitState.INACTIVE
        assert limit.check_state(-math.pi) == LimitState.AT_LOWER

    def test_swing_limit(self):
        limit = SwingLimit(y_angle=0.5, z_angle=0.5)
        assert limit.is_circular

        assert limit.check_within_cone(0.0, 0.0)
        assert not limit.check_within_cone(1.0, 0.0)

    def test_twist_limit(self):
        limit = TwistLimit(lower=-1.0, upper=1.0)
        assert limit.check_state(0.0) == LimitState.INACTIVE
        assert limit.check_state(-2.0) == LimitState.AT_LOWER

    def test_compute_limit_impulse(self):
        limit = LinearLimit(lower=-1.0, upper=1.0)
        impulse, state = compute_limit_impulse(
            limit,
            current_value=-2.0,
            current_velocity=-1.0,
            effective_mass=1.0,
            dt=1.0 / 60
        )
        assert state == LimitState.AT_LOWER
        assert impulse >= 0


# ============================================================================
# Contact Constraint Tests
# ============================================================================

class TestContactConstraint:
    """Tests for Contact Constraint."""

    def test_contact_point(self):
        point = ContactPoint(
            position=Vec3(0, 0, 0),
            normal=Vec3.unit_y(),
            penetration=0.1
        )
        assert point.penetration == 0.1

    def test_contact_manifold(self, two_bodies):
        body_a, body_b = two_bodies
        manifold = ContactManifold(body_a, body_b)
        assert manifold.point_count == 0

        point = ContactPoint(Vec3(0, 0, 0), Vec3.unit_y(), 0.1)
        manifold.add_point(point)
        assert manifold.point_count == 1

    def test_contact_manifold_max_points(self, two_bodies):
        body_a, body_b = two_bodies
        manifold = ContactManifold(body_a, body_b)

        for i in range(6):
            point = ContactPoint(
                position=Vec3(i, 0, 0),
                normal=Vec3.unit_y(),
                penetration=0.1
            )
            manifold.add_point(point)

        # Should be capped at 4
        assert manifold.point_count == 4

    def test_contact_constraint_creation(self, dynamic_body, static_body):
        constraint = ContactConstraint(dynamic_body, static_body)
        constraint.set_contact_point(
            Vec3(0, 0.5, 0),
            Vec3.unit_y(),
            0.1
        )
        assert constraint.manifold.point_count == 1

    def test_contact_constraint_friction(self, dynamic_body, static_body):
        constraint = ContactConstraint(dynamic_body, static_body)
        constraint.friction_coefficient = 0.6
        assert constraint.friction_coefficient == 0.6

    def test_contact_constraint_restitution(self, dynamic_body, static_body):
        constraint = ContactConstraint(dynamic_body, static_body)
        constraint.restitution_coefficient = 0.8
        assert constraint.restitution_coefficient == 0.8

    def test_compute_contact_jacobian(self, dynamic_body, static_body):
        j = compute_contact_jacobian(
            Vec3(0, 0.5, 0),
            Vec3.unit_y(),
            dynamic_body,
            static_body
        )
        assert j.linear_b.y == 1

    def test_combine_friction(self):
        combined = combine_friction(0.4, 0.6)
        assert abs(combined - math.sqrt(0.24)) < 1e-10

    def test_combine_restitution(self):
        combined = combine_restitution(0.3, 0.8)
        assert combined == 0.8

    def test_contact_constraint_prepare(self, dynamic_body, static_body, solver_config):
        constraint = ContactConstraint(dynamic_body, static_body)
        constraint.set_contact_point(
            Vec3(0, 0.5, 0),
            Vec3.unit_y(),
            0.1
        )
        constraint.prepare(1.0 / 60, solver_config)


# ============================================================================
# Joint Break Tests
# ============================================================================

class TestJointBreaking:
    """Tests for joint breaking mechanics."""

    def test_break_force_threshold(self, two_bodies):
        body_a, body_b = two_bodies
        joint = FixedJoint(body_a, body_b, break_force=100.0)
        assert joint.break_force == 100.0
        assert not joint.is_broken

    def test_break_torque_threshold(self, two_bodies):
        body_a, body_b = two_bodies
        joint = FixedJoint(body_a, body_b, break_torque=50.0)
        assert joint.break_torque == 50.0

    def test_break_event(self, two_bodies):
        body_a, body_b = two_bodies
        joint = FixedJoint(body_a, body_b, break_force=100.0)

        event_data = None
        def callback(event):
            nonlocal event_data
            event_data = event

        joint.set_break_callback(callback)
        joint._break(150.0, 0.0)

        assert joint.is_broken
        assert joint.state == JointState.BROKEN
        assert event_data is not None
        assert event_data.applied_force == 150.0

    def test_break_disables_joint(self, two_bodies, solver_config):
        body_a, body_b = two_bodies
        joint = FixedJoint(body_a, body_b, break_force=100.0)
        joint._break(150.0, 0.0)

        # Solve should do nothing on broken joint
        impulse = joint.solve_velocity()
        assert impulse == 0

    def test_joint_reset(self, two_bodies):
        body_a, body_b = two_bodies
        joint = FixedJoint(body_a, body_b, break_force=100.0)
        joint._break(150.0, 0.0)
        assert joint.is_broken

        joint.reset()
        assert not joint.is_broken
        assert joint.state == JointState.ACTIVE


# ============================================================================
# Warm Starting Tests
# ============================================================================

class TestWarmStarting:
    """Tests for warm starting."""

    def test_warm_start_factor_config(self, solver_config):
        assert solver_config.warm_start_factor == WARM_START_FACTOR

    def test_store_impulses(self, two_bodies, solver_config):
        body_a, body_b = two_bodies
        joint = FixedJoint(body_a, body_b)
        joint.prepare(1.0 / 60, solver_config)

        # Simulate some impulses
        joint._accumulated_impulse = [1.0, 2.0, 3.0, 0.5, 0.5, 0.5]
        joint.store_impulses_for_warm_start()

        assert joint._warm_start_impulse == joint._accumulated_impulse

    def test_warm_start_application(self, two_bodies, solver_config):
        body_a, body_b = two_bodies
        joint = FixedJoint(body_a, body_b)
        joint.prepare(1.0 / 60, solver_config)

        # Store initial velocities
        initial_vel_a = Vec3(body_a.velocity.x, body_a.velocity.y, body_a.velocity.z)
        initial_vel_b = Vec3(body_b.velocity.x, body_b.velocity.y, body_b.velocity.z)

        # Set up warm start with non-zero impulses
        joint._warm_start_impulse = [0.1] * 6

        joint.warm_start(0.8)

        # Verify velocities changed after warm starting
        vel_change_a = (body_a.velocity - initial_vel_a).length()
        vel_change_b = (body_b.velocity - initial_vel_b).length()
        total_change = vel_change_a + vel_change_b

        # At least one body should have its velocity modified
        assert total_change > 1e-10, "Warm start should modify body velocities"


# ============================================================================
# Solver Integration Tests
# ============================================================================

class TestSolverIntegration:
    """Integration tests for full solve cycles."""

    def test_pendulum_simulation(self, static_body, solver_config):
        """Test a simple pendulum with hinge joint."""
        bob = RigidBody.create_dynamic(
            id=1,
            position=Vec3(1, 0, 0),
            mass=1.0,
            inertia=Mat3.from_diagonal(0.1, 0.1, 0.1)
        )

        joint = HingeJoint(
            static_body, bob,
            local_anchor_a=Vec3.zero(),
            local_anchor_b=Vec3(-1, 0, 0),
            local_axis_a=Vec3.unit_z(),
            local_axis_b=Vec3.unit_z()
        )

        solver = ConstraintSolver(solver_config)
        solver.add_body(static_body)
        solver.add_body(bob)
        solver.add_constraint(joint)

        initial_pos = Vec3(bob.position.x, bob.position.y, bob.position.z)

        # Simulate a few steps
        dt = 1.0 / 60
        for _ in range(10):
            # Apply gravity
            bob.velocity = bob.velocity + Vec3(0, -9.81 * dt, 0)
            solver.solve(dt)

        # Verify constraint is approximately satisfied
        anchor_world = joint.get_world_anchor_a()
        bob_anchor = joint.get_world_anchor_b()
        constraint_error = (bob_anchor - anchor_world).length()

        # Constraint error should be small (allowing for some slack)
        assert constraint_error < 0.5, f"Pendulum constraint not satisfied: error={constraint_error}"

        # Bob should have moved due to gravity
        pos_change = (bob.position - initial_pos).length()
        assert pos_change > 0.01, "Pendulum bob should move under gravity"

    def test_chain_of_bodies(self, solver_config):
        """Test a chain of bodies connected by distance joints."""
        bodies = []
        constraints = []

        # Create chain
        for i in range(5):
            if i == 0:
                body = RigidBody.create_static(id=i, position=Vec3(0, 5, 0))
            else:
                body = RigidBody.create_dynamic(
                    id=i,
                    position=Vec3(0, 5 - i, 0),
                    mass=1.0
                )
            bodies.append(body)

            if i > 0:
                joint = DistanceJoint(
                    bodies[i-1], body,
                    local_anchor_a=Vec3.zero(),
                    local_anchor_b=Vec3.zero(),
                    rest_length=1.0
                )
                constraints.append(joint)

        solver = ConstraintSolver(solver_config)
        for body in bodies:
            solver.add_body(body)
        for constraint in constraints:
            solver.add_constraint(constraint)

        # Simulate
        dt = 1.0 / 60
        for _ in range(10):
            for body in bodies[1:]:
                body.velocity = body.velocity + Vec3(0, -9.81 * dt, 0)
            solver.solve(dt)

    def test_box_stack(self, static_body, solver_config):
        """Test stacked boxes with contact constraints."""
        boxes = [static_body]  # Ground

        for i in range(3):
            box = RigidBody.create_dynamic(
                id=i + 1,
                position=Vec3(0, 0.5 + i, 0),
                mass=1.0
            )
            boxes.append(box)

        solver = ConstraintSolver(solver_config)
        for box in boxes:
            solver.add_body(box)

        # Add contact constraints between adjacent boxes
        for i in range(len(boxes) - 1):
            contact = ContactConstraint(boxes[i], boxes[i+1])
            contact.set_contact_point(
                Vec3(0, 0.5 * (i + 1), 0),
                Vec3.unit_y(),
                0.0
            )
            solver.add_constraint(contact)

        solver.solve(1.0 / 60)

    def test_xpbd_cloth_patch(self, xpbd_config):
        """Test a small cloth patch with XPBD."""
        solver = XPBDSolver(xpbd_config)

        # Create 3x3 grid of particles
        particles = []
        for i in range(3):
            row = []
            for j in range(3):
                p = XPBDParticle(
                    id=i * 3 + j,
                    position=Vec3(j, 3, i),
                    inv_mass=1.0 if i > 0 else 0.0,  # Fix top row
                    is_static=(i == 0)
                )
                solver.add_particle(p)
                row.append(p)
            particles.append(row)

        # Add distance constraints
        for i in range(3):
            for j in range(3):
                # Horizontal
                if j < 2:
                    c = XPBDDistanceConstraint(
                        particles[i][j],
                        particles[i][j+1],
                        rest_length=1.0
                    )
                    solver.add_constraint(c)
                # Vertical
                if i < 2:
                    c = XPBDDistanceConstraint(
                        particles[i][j],
                        particles[i+1][j],
                        rest_length=1.0
                    )
                    solver.add_constraint(c)

        # Simulate
        solver.solve(1.0 / 60)


# ============================================================================
# Performance / Stress Tests
# ============================================================================

class TestPerformance:
    """Performance and stress tests."""

    def test_many_constraints(self, solver_config):
        """Test solver with many constraints."""
        bodies = [
            RigidBody.create_dynamic(id=i, position=Vec3(i, 0, 0), mass=1.0)
            for i in range(20)
        ]

        constraints = []
        for i in range(len(bodies) - 1):
            c = DistanceJoint(bodies[i], bodies[i+1])
            constraints.append(c)

        solver = ConstraintSolver(solver_config)
        for body in bodies:
            solver.add_body(body)
        for c in constraints:
            solver.add_constraint(c)

        # Should complete without timeout
        solver.solve(1.0 / 60)

    def test_many_islands(self):
        """Test island manager with many separate islands."""
        manager = IslandManager()

        bodies = [
            RigidBody.create_dynamic(id=i, position=Vec3(i * 10, 0, 0), mass=1.0)
            for i in range(50)
        ]

        manager.build_islands(bodies, [])
        assert len(manager.islands) == 50

        groups = manager.get_parallel_groups(4)
        assert len(groups) == 4


# ============================================================================
# Solver Convergence Tests
# ============================================================================

class TestSolverConvergence:
    """Tests that verify solver properly converges."""

    def test_distance_constraint_converges(self, solver_config):
        """Verify distance constraint error decreases with iterations."""
        body_a = RigidBody.create_static(id=0, position=Vec3.zero())
        body_b = RigidBody.create_dynamic(
            id=1,
            position=Vec3(2.0, 0, 0),  # Start 2m apart
            mass=1.0
        )

        joint = DistanceJoint(
            body_a, body_b,
            local_anchor_a=Vec3.zero(),
            local_anchor_b=Vec3.zero(),
            rest_length=1.0  # Target 1m distance
        )

        solver = ConstraintSolver(solver_config)
        solver.add_body(body_a)
        solver.add_body(body_b)
        solver.add_constraint(joint)

        # Measure initial error
        initial_distance = body_b.position.length()
        initial_error = abs(initial_distance - 1.0)

        # Run solver for multiple frames
        dt = 1.0 / 60
        for _ in range(30):
            solver.solve(dt)

        # Measure final error
        final_distance = body_b.position.length()
        final_error = abs(final_distance - 1.0)

        # Error should have decreased significantly
        assert final_error < initial_error * 0.5, (
            f"Solver did not converge: initial_error={initial_error}, final_error={final_error}"
        )

    def test_fixed_joint_converges(self, solver_config):
        """Verify fixed joint maintains relative position."""
        body_a = RigidBody.create_static(id=0, position=Vec3.zero())
        body_b = RigidBody.create_dynamic(
            id=1,
            position=Vec3(1.0, 0.5, 0),  # Start offset
            mass=1.0
        )

        # Joint should lock bodies together at anchors
        joint = FixedJoint(
            body_a, body_b,
            local_anchor_a=Vec3.zero(),
            local_anchor_b=Vec3.zero()
        )

        solver = ConstraintSolver(solver_config)
        solver.add_body(body_a)
        solver.add_body(body_b)
        solver.add_constraint(joint)

        # Measure initial offset
        initial_offset = body_b.position.length()

        # Run solver
        dt = 1.0 / 60
        for _ in range(50):
            solver.solve(dt)

        # Body B should have moved towards anchor at origin
        final_offset = body_b.position.length()

        # Final offset should be much smaller than initial
        assert final_offset < initial_offset * 0.3, (
            f"Fixed joint did not converge: initial={initial_offset}, final={final_offset}"
        )

    def test_hinge_joint_limits_enforced(self, solver_config):
        """Verify hinge joint respects angle limits."""
        body_a = RigidBody.create_static(id=0, position=Vec3.zero())
        body_b = RigidBody.create_dynamic(
            id=1,
            position=Vec3(1.0, 0, 0),
            mass=1.0,
            inertia=Mat3.from_diagonal(0.1, 0.1, 0.1)
        )

        joint = HingeJoint(
            body_a, body_b,
            local_anchor_a=Vec3.zero(),
            local_anchor_b=Vec3(-1, 0, 0),
            local_axis_a=Vec3.unit_z(),
            local_axis_b=Vec3.unit_z()
        )
        joint.set_limits(-math.pi / 4, math.pi / 4)  # +/- 45 degrees

        solver = ConstraintSolver(solver_config)
        solver.add_body(body_a)
        solver.add_body(body_b)
        solver.add_constraint(joint)

        # Apply torque that would push past limit
        dt = 1.0 / 60
        for _ in range(60):
            body_b.angular_velocity = body_b.angular_velocity + Vec3(0, 0, 5.0 * dt)
            solver.solve(dt)

        # Angle should be within limits (with some tolerance)
        angle = joint.get_current_angle()
        tolerance = 0.1  # Allow small overshoot
        assert angle <= math.pi / 4 + tolerance, (
            f"Hinge joint exceeded upper limit: angle={angle}, limit={math.pi / 4}"
        )


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
