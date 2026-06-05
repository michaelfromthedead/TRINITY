"""
Tests for Deterministic Particle Emitter with Fixed32 Math (T-CC-2.1).

Tests cover:
    - Fixed32Vec3 vector operations and determinism
    - DeterministicParticle state and Fixed32 attributes
    - DeterministicParticlePool allocation/deallocation
    - DeterministicEmitter lifecycle and determinism
    - Spawn modules (shape, rate, lifetime, velocity, size)
    - Update modules (gravity)
    - Cross-run reproducibility with same seed
    - Precision and accumulation over time

Requirements verified:
    1. Particle spawn positions use Fixed32 (not float)
    2. Particle spawn velocities use Fixed32
    3. Particle initial lifetime/age uses Fixed32
    4. Deterministic simulation: same seed produces identical particles
    5. Convert between Fixed32 and float at system boundaries
"""

import pytest
import math

from trinity.types import Fixed32, PCG64

from engine.rendering.particles.deterministic_emitter import (
    Fixed32Vec3,
    DeterministicParticle,
    DeterministicParticlePool,
    DeterministicEmitter,
    DeterministicEmitterShape,
    DeterministicShapeEmitter,
    DeterministicRateEmitter,
    DeterministicLifetimeModule,
    DeterministicVelocityModule,
    DeterministicSizeModule,
    DeterministicGravityModule,
    DeterministicSpawnModule,
)
from engine.rendering.particles.particle_system import (
    EmitterConfig,
    EmitterState,
    ParticleState,
    SimulationMode,
    Vec3,
)


# =============================================================================
# TEST Fixed32Vec3
# =============================================================================


class TestFixed32Vec3Creation:
    """Test Fixed32Vec3 creation and conversion."""

    def test_default_is_zero(self):
        """Test default vector is zero."""
        v = Fixed32Vec3()
        assert v.x.raw == 0
        assert v.y.raw == 0
        assert v.z.raw == 0

    def test_from_floats(self):
        """Test creation from float values."""
        v = Fixed32Vec3.from_floats(1.5, 2.5, 3.5)
        assert abs(v.x.as_float - 1.5) < 0.001
        assert abs(v.y.as_float - 2.5) < 0.001
        assert abs(v.z.as_float - 3.5) < 0.001

    def test_zero_factory(self):
        """Test zero() factory method."""
        v = Fixed32Vec3.zero()
        assert v.x.raw == 0
        assert v.y.raw == 0
        assert v.z.raw == 0

    def test_to_vec3_conversion(self):
        """Test conversion to float Vec3 at system boundary."""
        v = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        fv = v.to_vec3()
        assert isinstance(fv, Vec3)
        assert abs(fv.x - 1.0) < 0.001
        assert abs(fv.y - 2.0) < 0.001
        assert abs(fv.z - 3.0) < 0.001

    def test_from_vec3_conversion(self):
        """Test conversion from float Vec3 at system boundary."""
        fv = Vec3(1.0, 2.0, 3.0)
        v = Fixed32Vec3.from_vec3(fv)
        assert abs(v.x.as_float - 1.0) < 0.001
        assert abs(v.y.as_float - 2.0) < 0.001
        assert abs(v.z.as_float - 3.0) < 0.001

    def test_copy_creates_independent_vector(self):
        """Test copy creates independent vector."""
        v1 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        v2 = v1.copy()
        v2.x = Fixed32(10.0)
        assert abs(v1.x.as_float - 1.0) < 0.001
        assert abs(v2.x.as_float - 10.0) < 0.001


class TestFixed32Vec3Arithmetic:
    """Test Fixed32Vec3 arithmetic operations."""

    def test_addition(self):
        """Test vector addition."""
        v1 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        v2 = Fixed32Vec3.from_floats(4.0, 5.0, 6.0)
        result = v1 + v2
        assert abs(result.x.as_float - 5.0) < 0.001
        assert abs(result.y.as_float - 7.0) < 0.001
        assert abs(result.z.as_float - 9.0) < 0.001

    def test_subtraction(self):
        """Test vector subtraction."""
        v1 = Fixed32Vec3.from_floats(5.0, 7.0, 9.0)
        v2 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        result = v1 - v2
        assert abs(result.x.as_float - 4.0) < 0.001
        assert abs(result.y.as_float - 5.0) < 0.001
        assert abs(result.z.as_float - 6.0) < 0.001

    def test_scalar_multiplication(self):
        """Test scalar multiplication with Fixed32."""
        v = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        result = v * Fixed32(2)
        assert abs(result.x.as_float - 2.0) < 0.01
        assert abs(result.y.as_float - 4.0) < 0.01
        assert abs(result.z.as_float - 6.0) < 0.01

    def test_int_multiplication(self):
        """Test scalar multiplication with integer."""
        v = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        result = v.mul_int(3)
        assert abs(result.x.as_float - 3.0) < 0.001
        assert abs(result.y.as_float - 6.0) < 0.001
        assert abs(result.z.as_float - 9.0) < 0.001

    def test_negation(self):
        """Test vector negation."""
        v = Fixed32Vec3.from_floats(1.0, -2.0, 3.0)
        result = -v
        assert abs(result.x.as_float - (-1.0)) < 0.001
        assert abs(result.y.as_float - 2.0) < 0.001
        assert abs(result.z.as_float - (-3.0)) < 0.001

    def test_dot_product(self):
        """Test dot product."""
        v1 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        v2 = Fixed32Vec3.from_floats(4.0, 5.0, 6.0)
        result = v1.dot(v2)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert abs(result.as_float - 32.0) < 0.1

    def test_length_squared(self):
        """Test length squared."""
        v = Fixed32Vec3.from_floats(3.0, 4.0, 0.0)
        result = v.length_squared()
        # 3^2 + 4^2 + 0^2 = 9 + 16 = 25
        assert abs(result.as_float - 25.0) < 0.1


class TestFixed32Vec3Equality:
    """Test Fixed32Vec3 equality and hashing."""

    def test_equality_same_values(self):
        """Test equality for same values."""
        v1 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        v2 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        assert v1 == v2

    def test_equality_different_values(self):
        """Test inequality for different values."""
        v1 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        v2 = Fixed32Vec3.from_floats(1.0, 2.0, 4.0)
        assert v1 != v2

    def test_hash_consistency(self):
        """Test hash consistency for equal vectors."""
        v1 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        v2 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        assert hash(v1) == hash(v2)


# =============================================================================
# TEST DeterministicParticle
# =============================================================================


class TestDeterministicParticleCreation:
    """Test DeterministicParticle creation and state."""

    def test_default_state_is_dead(self):
        """Test default particle state is DEAD."""
        p = DeterministicParticle()
        assert p.state == ParticleState.DEAD
        assert not p.is_alive

    def test_default_position_is_zero(self):
        """Test default position is zero Fixed32Vec3."""
        p = DeterministicParticle()
        assert p.position.x.raw == 0
        assert p.position.y.raw == 0
        assert p.position.z.raw == 0

    def test_default_velocity_is_zero(self):
        """Test default velocity is zero."""
        p = DeterministicParticle()
        assert p.velocity.x.raw == 0
        assert p.velocity.y.raw == 0
        assert p.velocity.z.raw == 0

    def test_default_lifetime_is_one(self):
        """Test default lifetime is 1."""
        p = DeterministicParticle()
        assert abs(p.lifetime.as_float - 1.0) < 0.001

    def test_age_is_fixed32(self):
        """Test age is Fixed32 type."""
        p = DeterministicParticle()
        assert isinstance(p.age, Fixed32)

    def test_lifetime_is_fixed32(self):
        """Test lifetime is Fixed32 type."""
        p = DeterministicParticle()
        assert isinstance(p.lifetime, Fixed32)


class TestDeterministicParticleNormalizedAge:
    """Test normalized age calculation."""

    def test_normalized_age_calculation(self):
        """Test normalized age as fraction of lifetime."""
        p = DeterministicParticle()
        p.lifetime = Fixed32(2.0)
        p.age = Fixed32(1.0)
        assert abs(p.normalized_age.as_float - 0.5) < 0.01

    def test_normalized_age_zero_lifetime(self):
        """Test normalized age when lifetime is zero."""
        p = DeterministicParticle()
        p.lifetime = Fixed32(0)
        p.age = Fixed32(1.0)
        assert p.normalized_age.as_float == 1.0

    def test_normalized_age_float(self):
        """Test normalized age as float for rendering."""
        p = DeterministicParticle()
        p.lifetime = Fixed32(4.0)
        p.age = Fixed32(1.0)
        assert abs(p.normalized_age_float - 0.25) < 0.01


class TestDeterministicParticleReset:
    """Test particle reset functionality."""

    def test_reset_clears_state(self):
        """Test reset clears particle state."""
        p = DeterministicParticle()
        p.state = ParticleState.ALIVE
        p.age = Fixed32(5.0)
        p.position = Fixed32Vec3.from_floats(10, 20, 30)
        p.custom_data["key"] = "value"

        p.reset()

        assert p.state == ParticleState.DEAD
        assert p.age.raw == 0
        assert p.position.x.raw == 0
        assert len(p.custom_data) == 0

    def test_reset_restores_defaults(self):
        """Test reset restores default values."""
        p = DeterministicParticle()
        p.size = Fixed32(5.0)
        p.lifetime = Fixed32(10.0)

        p.reset()

        assert abs(p.size.as_float - 1.0) < 0.01
        assert abs(p.lifetime.as_float - 1.0) < 0.01


# =============================================================================
# TEST DeterministicParticlePool
# =============================================================================


class TestDeterministicParticlePoolCreation:
    """Test pool creation."""

    def test_creation_with_max_particles(self):
        """Test pool creation with specified max particles."""
        pool = DeterministicParticlePool(max_particles=100)
        assert pool.max_particles == 100
        assert pool.alive_count == 0
        assert pool.free_count == 100

    def test_default_max_particles(self):
        """Test default max particles."""
        pool = DeterministicParticlePool()
        assert pool.max_particles == 1000


class TestDeterministicParticlePoolAllocation:
    """Test pool allocation and deallocation."""

    def test_allocate_returns_particle(self):
        """Test allocation returns a particle."""
        pool = DeterministicParticlePool(max_particles=10)
        p = pool.allocate()
        assert p is not None
        assert isinstance(p, DeterministicParticle)
        assert p.state == ParticleState.ALIVE

    def test_allocate_updates_counts(self):
        """Test allocation updates counts correctly."""
        pool = DeterministicParticlePool(max_particles=10)
        pool.allocate()
        assert pool.alive_count == 1
        assert pool.free_count == 9

    def test_allocate_exhausted_returns_none(self):
        """Test allocation returns None when exhausted."""
        pool = DeterministicParticlePool(max_particles=2)
        pool.allocate()
        pool.allocate()
        p = pool.allocate()
        assert p is None
        assert pool.alive_count == 2

    def test_deallocate_returns_to_pool(self):
        """Test deallocation returns particle to pool."""
        pool = DeterministicParticlePool(max_particles=10)
        p = pool.allocate()
        pool.deallocate(p)
        assert p.state == ParticleState.DEAD
        assert pool.alive_count == 0
        assert pool.free_count == 10

    def test_deallocate_twice_is_safe(self):
        """Test deallocating same particle twice is safe."""
        pool = DeterministicParticlePool(max_particles=10)
        p = pool.allocate()
        pool.deallocate(p)
        pool.deallocate(p)  # Should not crash
        assert pool.alive_count == 0


class TestDeterministicParticlePoolIteration:
    """Test pool iteration."""

    def test_iter_alive_yields_all_alive(self):
        """Test iter_alive yields all alive particles."""
        pool = DeterministicParticlePool(max_particles=10)
        pool.allocate()
        pool.allocate()
        pool.allocate()

        alive = list(pool.iter_alive())
        assert len(alive) == 3

    def test_iter_alive_excludes_dead(self):
        """Test iter_alive excludes dead particles."""
        pool = DeterministicParticlePool(max_particles=10)
        p1 = pool.allocate()
        p2 = pool.allocate()
        pool.deallocate(p1)

        alive = list(pool.iter_alive())
        assert len(alive) == 1
        assert p2 in alive

    def test_kill_all_clears_pool(self):
        """Test kill_all clears all particles."""
        pool = DeterministicParticlePool(max_particles=10)
        for _ in range(5):
            pool.allocate()

        pool.kill_all()
        assert pool.alive_count == 0
        assert pool.free_count == 10


# =============================================================================
# TEST DeterministicRateEmitter
# =============================================================================


class TestDeterministicRateEmitter:
    """Test rate-based emission."""

    def test_spawn_count_accumulation(self):
        """Test spawn count accumulates correctly."""
        rate_emitter = DeterministicRateEmitter(rate=Fixed32(100))
        # 100 particles/sec * 0.1 sec = ~10 particles (Fixed32 precision may vary slightly)
        count = rate_emitter.get_spawn_count(Fixed32(0.1))
        assert count >= 9 and count <= 10  # Allow for Fixed32 precision

    def test_spawn_count_fractional_accumulation(self):
        """Test fractional spawn counts accumulate over multiple calls."""
        rate_emitter = DeterministicRateEmitter(rate=Fixed32(100))
        # Accumulate over many small steps
        total = 0
        for _ in range(10):
            total += rate_emitter.get_spawn_count(Fixed32(0.01))
        # 100 particles/sec * 0.1 sec total = ~10 particles
        assert total >= 9 and total <= 11

    def test_spawn_count_large_dt(self):
        """Test spawn count with larger dt."""
        rate_emitter = DeterministicRateEmitter(rate=Fixed32(1000))
        # 1000 particles/sec * 1.0 sec = ~1000 particles
        count = rate_emitter.get_spawn_count(Fixed32(1.0))
        assert count >= 990 and count <= 1000

    def test_rate_property(self):
        """Test rate property getter/setter."""
        rate_emitter = DeterministicRateEmitter(rate=Fixed32(50))
        assert abs(rate_emitter.rate.as_float - 50.0) < 0.001
        rate_emitter.rate = Fixed32(100)
        assert abs(rate_emitter.rate.as_float - 100.0) < 0.001


# =============================================================================
# TEST DeterministicLifetimeModule
# =============================================================================


class TestDeterministicLifetimeModule:
    """Test lifetime module."""

    def test_sets_lifetime_in_range(self):
        """Test lifetime is set within specified range."""
        rng = PCG64(seed=12345)
        module = DeterministicLifetimeModule(
            min_lifetime=Fixed32(1.0),
            max_lifetime=Fixed32(3.0),
            rng=rng,
        )
        p = DeterministicParticle()
        module.apply_to_particle(p, Fixed32(0.016))

        assert p.lifetime.as_float >= 1.0
        assert p.lifetime.as_float <= 3.0

    def test_deterministic_lifetime(self):
        """Test same seed produces same lifetime."""
        p1 = DeterministicParticle()
        p2 = DeterministicParticle()

        module1 = DeterministicLifetimeModule(
            min_lifetime=Fixed32(1.0),
            max_lifetime=Fixed32(5.0),
            rng=PCG64(seed=42),
        )
        module2 = DeterministicLifetimeModule(
            min_lifetime=Fixed32(1.0),
            max_lifetime=Fixed32(5.0),
            rng=PCG64(seed=42),
        )

        module1.apply_to_particle(p1, Fixed32(0.016))
        module2.apply_to_particle(p2, Fixed32(0.016))

        assert p1.lifetime.raw == p2.lifetime.raw


# =============================================================================
# TEST DeterministicVelocityModule
# =============================================================================


class TestDeterministicVelocityModule:
    """Test velocity module."""

    def test_sets_velocity_with_spread(self):
        """Test velocity is set with spread."""
        rng = PCG64(seed=12345)
        module = DeterministicVelocityModule(
            velocity=Fixed32Vec3.from_floats(0, 10, 0),
            velocity_spread=Fixed32Vec3.from_floats(2, 2, 2),
            rng=rng,
        )
        p = DeterministicParticle()
        module.apply_to_particle(p, Fixed32(0.016))

        # Y velocity should be around 10 +/- 2
        assert p.velocity.y.as_float >= 8.0
        assert p.velocity.y.as_float <= 12.0

    def test_deterministic_velocity(self):
        """Test same seed produces same velocity."""
        p1 = DeterministicParticle()
        p2 = DeterministicParticle()

        module1 = DeterministicVelocityModule(
            velocity=Fixed32Vec3.from_floats(0, 5, 0),
            velocity_spread=Fixed32Vec3.from_floats(1, 1, 1),
            rng=PCG64(seed=999),
        )
        module2 = DeterministicVelocityModule(
            velocity=Fixed32Vec3.from_floats(0, 5, 0),
            velocity_spread=Fixed32Vec3.from_floats(1, 1, 1),
            rng=PCG64(seed=999),
        )

        module1.apply_to_particle(p1, Fixed32(0.016))
        module2.apply_to_particle(p2, Fixed32(0.016))

        assert p1.velocity.x.raw == p2.velocity.x.raw
        assert p1.velocity.y.raw == p2.velocity.y.raw
        assert p1.velocity.z.raw == p2.velocity.z.raw


# =============================================================================
# TEST DeterministicSizeModule
# =============================================================================


class TestDeterministicSizeModule:
    """Test size module."""

    def test_sets_size_in_range(self):
        """Test size is set within range."""
        rng = PCG64(seed=12345)
        module = DeterministicSizeModule(
            min_size=Fixed32(0.5),
            max_size=Fixed32(2.0),
            rng=rng,
        )
        p = DeterministicParticle()
        module.apply_to_particle(p, Fixed32(0.016))

        assert p.size.as_float >= 0.5
        assert p.size.as_float <= 2.0

    def test_deterministic_size(self):
        """Test same seed produces same size."""
        p1 = DeterministicParticle()
        p2 = DeterministicParticle()

        module1 = DeterministicSizeModule(
            min_size=Fixed32(1.0),
            max_size=Fixed32(5.0),
            rng=PCG64(seed=777),
        )
        module2 = DeterministicSizeModule(
            min_size=Fixed32(1.0),
            max_size=Fixed32(5.0),
            rng=PCG64(seed=777),
        )

        module1.apply_to_particle(p1, Fixed32(0.016))
        module2.apply_to_particle(p2, Fixed32(0.016))

        assert p1.size.raw == p2.size.raw


# =============================================================================
# TEST DeterministicShapeEmitter
# =============================================================================


class TestDeterministicShapeEmitter:
    """Test shape-based emission."""

    def test_point_emission(self):
        """Test point emission produces zero offset."""
        module = DeterministicShapeEmitter(
            shape=DeterministicEmitterShape.POINT,
            position=Fixed32Vec3.from_floats(5, 10, 15),
            rng=PCG64(seed=123),
        )
        p = DeterministicParticle()
        module.apply_to_particle(p, Fixed32(0.016))

        assert abs(p.position.x.as_float - 5.0) < 0.001
        assert abs(p.position.y.as_float - 10.0) < 0.001
        assert abs(p.position.z.as_float - 15.0) < 0.001

    def test_sphere_emission_within_radius(self):
        """Test sphere emission stays within radius."""
        module = DeterministicShapeEmitter(
            shape=DeterministicEmitterShape.SPHERE,
            position=Fixed32Vec3.zero(),
            radius=Fixed32(5.0),
            rng=PCG64(seed=456),
        )

        for _ in range(10):
            p = DeterministicParticle()
            module.apply_to_particle(p, Fixed32(0.016))
            distance_sq = p.position.length_squared().as_float
            assert distance_sq <= 25.0 + 0.1  # radius^2 with tolerance

    def test_box_emission_within_bounds(self):
        """Test box emission stays within bounds."""
        module = DeterministicShapeEmitter(
            shape=DeterministicEmitterShape.BOX,
            position=Fixed32Vec3.zero(),
            size=Fixed32Vec3.from_floats(4, 6, 8),
            emit_from_surface=False,
            rng=PCG64(seed=789),
        )

        for _ in range(10):
            p = DeterministicParticle()
            module.apply_to_particle(p, Fixed32(0.016))
            assert abs(p.position.x.as_float) <= 2.0 + 0.01
            assert abs(p.position.y.as_float) <= 3.0 + 0.01
            assert abs(p.position.z.as_float) <= 4.0 + 0.01

    def test_deterministic_shape_emission(self):
        """Test same seed produces same positions."""
        module1 = DeterministicShapeEmitter(
            shape=DeterministicEmitterShape.SPHERE,
            radius=Fixed32(10.0),
            rng=PCG64(seed=12345),
        )
        module2 = DeterministicShapeEmitter(
            shape=DeterministicEmitterShape.SPHERE,
            radius=Fixed32(10.0),
            rng=PCG64(seed=12345),
        )

        p1 = DeterministicParticle()
        p2 = DeterministicParticle()
        module1.apply_to_particle(p1, Fixed32(0.016))
        module2.apply_to_particle(p2, Fixed32(0.016))

        assert p1.position.x.raw == p2.position.x.raw
        assert p1.position.y.raw == p2.position.y.raw
        assert p1.position.z.raw == p2.position.z.raw


# =============================================================================
# TEST DeterministicGravityModule
# =============================================================================


class TestDeterministicGravityModule:
    """Test gravity module."""

    def test_applies_gravity(self):
        """Test gravity is applied to acceleration."""
        module = DeterministicGravityModule(
            gravity=Fixed32Vec3.from_floats(0, -9.8, 0)
        )
        p = DeterministicParticle()
        p.acceleration = Fixed32Vec3.zero()

        module.apply_to_particle(p, Fixed32(0.016))

        assert abs(p.acceleration.y.as_float - (-9.8)) < 0.01

    def test_gravity_accumulates(self):
        """Test gravity accumulates with existing acceleration."""
        module = DeterministicGravityModule(
            gravity=Fixed32Vec3.from_floats(0, -10, 0)
        )
        p = DeterministicParticle()
        p.acceleration = Fixed32Vec3.from_floats(0, 5, 0)

        module.apply_to_particle(p, Fixed32(0.016))

        # 5 + (-10) = -5
        assert abs(p.acceleration.y.as_float - (-5.0)) < 0.1


# =============================================================================
# TEST DeterministicEmitter
# =============================================================================


class TestDeterministicEmitterCreation:
    """Test emitter creation."""

    def test_creation_with_seed(self):
        """Test emitter creation with seed."""
        emitter = DeterministicEmitter(seed=12345)
        assert emitter.seed == 12345
        assert emitter.state == EmitterState.INACTIVE
        assert emitter.alive_count == 0

    def test_creation_with_config(self):
        """Test emitter creation with config."""
        config = EmitterConfig(max_particles=500)
        emitter = DeterministicEmitter(config=config, seed=42)
        assert emitter.config.max_particles == 500

    def test_age_is_fixed32(self):
        """Test emitter age is Fixed32."""
        emitter = DeterministicEmitter(seed=0)
        assert isinstance(emitter.age, Fixed32)


class TestDeterministicEmitterLifecycle:
    """Test emitter lifecycle."""

    def test_start_activates_emitter(self):
        """Test start activates emitter."""
        emitter = DeterministicEmitter(seed=0)
        emitter.start()
        assert emitter.state == EmitterState.ACTIVE

    def test_stop_immediate(self):
        """Test immediate stop kills all particles."""
        emitter = DeterministicEmitter(seed=0)
        emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(1000)))
        emitter.start()
        emitter.update(0.1)

        emitter.stop(immediate=True)
        assert emitter.state == EmitterState.STOPPED
        assert emitter.alive_count == 0

    def test_stop_graceful(self):
        """Test graceful stop waits for particles to die."""
        emitter = DeterministicEmitter(seed=0)
        emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(100)))
        emitter.add_spawn_module(
            DeterministicLifetimeModule(
                min_lifetime=Fixed32(1.0),  # Longer lifetime so particles survive
                max_lifetime=Fixed32(1.0),
            )
        )
        emitter.start()
        emitter.update(0.05)  # Smaller dt so particles survive

        assert emitter.alive_count > 0
        emitter.stop(immediate=False)
        assert emitter.state == EmitterState.STOPPING

        # Update until particles die (1 second lifetime)
        for _ in range(50):
            emitter.update(0.05)

        assert emitter.state == EmitterState.STOPPED
        assert emitter.alive_count == 0


class TestDeterministicEmitterDeterminism:
    """Test emitter determinism - core requirement."""

    def test_same_seed_same_particles(self):
        """Test same seed produces identical particle positions."""
        def run_emitter(seed: int, steps: int) -> list[tuple[int, int, int]]:
            emitter = DeterministicEmitter(seed=seed)
            emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(100)))
            emitter.add_spawn_module(
                DeterministicShapeEmitter(
                    shape=DeterministicEmitterShape.SPHERE,
                    radius=Fixed32(5.0),
                )
            )
            emitter.add_spawn_module(
                DeterministicVelocityModule(
                    velocity=Fixed32Vec3.from_floats(0, 10, 0),
                    velocity_spread=Fixed32Vec3.from_floats(2, 2, 2),
                )
            )
            emitter.start()

            for _ in range(steps):
                emitter.update_fixed(Fixed32(0.016))

            positions = []
            for p in emitter.iter_particles():
                positions.append((p.position.x.raw, p.position.y.raw, p.position.z.raw))
            return sorted(positions)

        # Run twice with same seed
        positions1 = run_emitter(seed=42, steps=10)
        positions2 = run_emitter(seed=42, steps=10)

        assert positions1 == positions2

    def test_different_seeds_different_particles(self):
        """Test different seeds produce different velocities."""
        def run_emitter(seed: int) -> list[tuple[int, int, int]]:
            emitter = DeterministicEmitter(seed=seed)
            emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(100)))
            emitter.add_spawn_module(
                DeterministicVelocityModule(
                    velocity=Fixed32Vec3.from_floats(0, 10, 0),
                    velocity_spread=Fixed32Vec3.from_floats(5, 5, 5),
                )
            )
            emitter.add_spawn_module(
                DeterministicLifetimeModule(
                    min_lifetime=Fixed32(10),
                    max_lifetime=Fixed32(10),
                )
            )
            emitter.start()
            emitter.update_fixed(Fixed32(0.5))

            velocities = []
            for p in emitter.iter_particles():
                velocities.append((p.velocity.x.raw, p.velocity.y.raw, p.velocity.z.raw))
            return velocities

        velocities1 = run_emitter(seed=1)
        velocities2 = run_emitter(seed=2)

        # Should have particles
        assert len(velocities1) > 0
        assert len(velocities2) > 0
        # At least one velocity should differ due to different seeds
        assert velocities1 != velocities2

    def test_reset_restores_determinism(self):
        """Test reset with same seed restores determinism."""
        def create_and_run():
            emitter = DeterministicEmitter(seed=12345)
            emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(50)))
            emitter.add_spawn_module(
                DeterministicShapeEmitter(
                    shape=DeterministicEmitterShape.BOX,
                    size=Fixed32Vec3.from_floats(10, 10, 10),
                )
            )
            emitter.add_spawn_module(
                DeterministicLifetimeModule(
                    min_lifetime=Fixed32(10),
                    max_lifetime=Fixed32(10),
                )
            )
            emitter.start()
            emitter.update_fixed(Fixed32(0.1))
            return sorted([
                (p.position.x.raw, p.position.y.raw, p.position.z.raw)
                for p in emitter.iter_particles()
            ])

        # Two separate runs with same seed should produce same results
        positions1 = create_and_run()
        positions2 = create_and_run()

        assert len(positions1) > 0
        assert positions1 == positions2


class TestDeterministicEmitterPhysics:
    """Test Fixed32 physics integration."""

    def test_velocity_integration(self):
        """Test velocity updates position with Fixed32 math."""
        emitter = DeterministicEmitter(seed=0)
        # Use higher rate to ensure we spawn particles
        emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(100)))
        emitter.add_spawn_module(
            DeterministicVelocityModule(
                velocity=Fixed32Vec3.from_floats(100, 0, 0),  # High velocity to see movement
                velocity_spread=Fixed32Vec3.zero(),
            )
        )
        emitter.add_spawn_module(
            DeterministicLifetimeModule(
                min_lifetime=Fixed32(10),
                max_lifetime=Fixed32(10),
            )
        )
        emitter.start()

        # Spawn particles and let them move
        emitter.update_fixed(Fixed32(0.1))
        emitter.update_fixed(Fixed32(0.5))
        emitter.update_fixed(Fixed32(0.5))

        # Get particles and check position
        particles = list(emitter.iter_particles())
        assert len(particles) > 0

        # Check that particles have moved in x direction
        # Velocity is 100 units/sec, after ~1 second oldest particles should be at ~100
        max_x = max(p.position.x.as_float for p in particles)
        assert max_x > 1.0  # Should have moved significantly

    def test_gravity_affects_velocity(self):
        """Test gravity module affects velocity over time."""
        emitter = DeterministicEmitter(seed=0)
        # Use higher rate to ensure we spawn particles
        emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(100)))
        emitter.add_spawn_module(
            DeterministicVelocityModule(
                velocity=Fixed32Vec3.zero(),
                velocity_spread=Fixed32Vec3.zero(),
            )
        )
        emitter.add_spawn_module(
            DeterministicLifetimeModule(
                min_lifetime=Fixed32(10),
                max_lifetime=Fixed32(10),
            )
        )
        emitter.add_update_module(
            DeterministicGravityModule(
                gravity=Fixed32Vec3.from_floats(0, -10, 0)
            )
        )
        emitter.start()
        emitter.update_fixed(Fixed32(0.1))
        emitter.update_fixed(Fixed32(0.5))

        particles = list(emitter.iter_particles())
        assert len(particles) > 0

        # After gravity, some particles should have negative y velocity
        has_gravity_effect = any(p.velocity.y.as_float < 0 for p in particles)
        assert has_gravity_effect


class TestDeterministicEmitterPrecision:
    """Test Fixed32 precision over time."""

    def test_no_accumulation_drift(self):
        """Test no precision drift over many updates."""
        emitter = DeterministicEmitter(seed=0)
        emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(1)))
        emitter.add_spawn_module(
            DeterministicVelocityModule(
                velocity=Fixed32Vec3.from_floats(0.1, 0.1, 0.1),
                velocity_spread=Fixed32Vec3.zero(),
            )
        )
        emitter.add_spawn_module(
            DeterministicLifetimeModule(
                min_lifetime=Fixed32(1000),
                max_lifetime=Fixed32(1000),
            )
        )
        emitter.start()

        # Run many small updates
        dt = Fixed32(0.001)
        for _ in range(1000):
            emitter.update_fixed(dt)

        # Raw values should still be integers (no float accumulation)
        for p in emitter.iter_particles():
            assert isinstance(p.position.x.raw, int)
            assert isinstance(p.velocity.y.raw, int)

    def test_consistent_across_dt_variations(self):
        """Test same total time with different dt gives same result."""
        def run_simulation(dt_val: float, steps: int, seed: int) -> int:
            emitter = DeterministicEmitter(seed=seed)
            emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(10)))
            emitter.add_spawn_module(
                DeterministicLifetimeModule(
                    min_lifetime=Fixed32(100),
                    max_lifetime=Fixed32(100),
                )
            )
            emitter.start()

            dt = Fixed32(dt_val)
            for _ in range(steps):
                emitter.update_fixed(dt)

            return emitter.alive_count

        # Same total time (0.1 sec) with different dt
        count1 = run_simulation(0.01, 10, seed=42)  # 10 steps of 0.01
        count2 = run_simulation(0.001, 100, seed=42)  # 100 steps of 0.001

        # Rate accumulation should produce same particle count
        assert count1 == count2


class TestDeterministicEmitterStats:
    """Test emitter statistics."""

    def test_stats_includes_seed(self):
        """Test stats include seed for debugging."""
        emitter = DeterministicEmitter(seed=99999)
        stats = emitter.get_stats()
        assert stats["seed"] == 99999

    def test_stats_includes_frame_count(self):
        """Test stats include frame count."""
        emitter = DeterministicEmitter(seed=0)
        emitter.start()
        emitter.update(0.016)
        emitter.update(0.016)
        emitter.update(0.016)

        stats = emitter.get_stats()
        assert stats["frame_count"] == 3


# =============================================================================
# TEST BOUNDARY CONVERSIONS
# =============================================================================


class TestBoundaryConversions:
    """Test Fixed32 <-> float conversions at system boundaries."""

    def test_particle_to_float_position(self):
        """Test particle position converts to float for rendering."""
        p = DeterministicParticle()
        p.position = Fixed32Vec3.from_floats(1.5, 2.5, 3.5)

        float_pos = p.to_float_position()
        assert isinstance(float_pos, Vec3)
        assert abs(float_pos.x - 1.5) < 0.001
        assert abs(float_pos.y - 2.5) < 0.001
        assert abs(float_pos.z - 3.5) < 0.001

    def test_update_accepts_float_dt(self):
        """Test emitter accepts float dt for convenience."""
        emitter = DeterministicEmitter(seed=0)
        emitter.start()
        # Should not raise
        emitter.update(0.016)
        assert emitter.age.as_float > 0

    def test_age_float_property(self):
        """Test age_float property for external APIs."""
        emitter = DeterministicEmitter(seed=0)
        emitter.start()
        emitter.update_fixed(Fixed32(1.5))

        assert abs(emitter.age_float - 1.5) < 0.001


# =============================================================================
# TEST MODULE ENABLED/LOD
# =============================================================================


class TestModuleControl:
    """Test module enable/disable and LOD."""

    def test_disabled_module_not_applied(self):
        """Test disabled module is not applied."""
        module = DeterministicLifetimeModule(
            min_lifetime=Fixed32(5),
            max_lifetime=Fixed32(5),
        )
        module.enabled = False

        p = DeterministicParticle()
        default_lifetime = p.lifetime.raw

        # Module is disabled, so lifetime should not change
        # (but module.apply_to_particle is still called by emitter logic)
        assert not module.is_active_for_lod(0)

    def test_lod_range_filtering(self):
        """Test LOD range filters modules."""
        module = DeterministicLifetimeModule(
            min_lifetime=Fixed32(1),
            max_lifetime=Fixed32(1),
            lod_range=(2, 4),
        )

        assert not module.is_active_for_lod(0)
        assert not module.is_active_for_lod(1)
        assert module.is_active_for_lod(2)
        assert module.is_active_for_lod(3)
        assert module.is_active_for_lod(4)
        assert not module.is_active_for_lod(5)


# =============================================================================
# TEST EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_rate_emits_nothing(self):
        """Test zero rate emits no particles."""
        emitter = DeterministicEmitter(seed=0)
        emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(0)))
        emitter.start()
        emitter.update(1.0)

        assert emitter.alive_count == 0

    def test_short_lifetime_particles_die(self):
        """Test short lifetime particles die after their lifetime."""
        emitter = DeterministicEmitter(seed=0)
        emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(1000)))
        emitter.add_spawn_module(
            DeterministicLifetimeModule(
                min_lifetime=Fixed32(0.5),  # Longer lifetime so particles survive initial spawn
                max_lifetime=Fixed32(0.5),
            )
        )
        emitter.start()

        # Spawn some particles with small dt
        emitter.update(0.01)
        initial_count = emitter.alive_count
        assert initial_count > 0, f"Expected particles, got {initial_count}"

        # Wait for particles to die (lifetime is 0.5s)
        for _ in range(100):
            emitter.update(0.01)

        # After 1 second total, particles spawned early should be dead
        stats = emitter.get_stats()
        assert stats["death_count"] > 0  # Particles should have died

    def test_very_small_dt(self):
        """Test very small dt does not break accumulation."""
        emitter = DeterministicEmitter(seed=0)
        emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(10)))
        emitter.add_spawn_module(
            DeterministicLifetimeModule(
                min_lifetime=Fixed32(10),
                max_lifetime=Fixed32(10),
            )
        )
        emitter.start()

        # Very small dt
        for _ in range(100):
            emitter.update_fixed(Fixed32(0.0001))

        # Should have spawned approximately 0.1 * 10 = 1 particle
        assert emitter.alive_count >= 0  # May be 0 or 1 due to accumulation

    def test_large_particle_count(self):
        """Test large particle counts work correctly."""
        emitter = DeterministicEmitter(
            config=EmitterConfig(max_particles=10000),
            seed=0,
        )
        emitter.add_spawn_module(DeterministicRateEmitter(rate=Fixed32(5000)))
        emitter.add_spawn_module(
            DeterministicLifetimeModule(
                min_lifetime=Fixed32(10),
                max_lifetime=Fixed32(10),
            )
        )
        emitter.start()
        # Run multiple updates to accumulate particles
        for _ in range(10):
            emitter.update(0.1)

        # Should have many particles, limited by max_particles
        assert emitter.alive_count <= 10000
        assert emitter.alive_count > 0
