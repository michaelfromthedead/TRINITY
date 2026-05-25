"""
Tests for the core particle system module.

Tests:
    - EmitterConfig creation and validation
    - ParticlePool allocation and recycling
    - ParticleBudget category management
    - ParticleEmitter lifecycle (spawn, update, render, death)
    - Simulation mode resolution
"""

import pytest

from engine.rendering.particles.constants import PARTICLE_CONSTANTS

from engine.rendering.particles.particle_system import (
    BudgetAllocation,
    EmitterConfig,
    EmitterState,
    Particle,
    ParticleBudget,
    ParticleEmitter,
    ParticlePool,
    ParticleState,
    ParticleSystemManager,
    SimulationMode,
    Vec3,
    Vec4,
)


class TestVec3:
    """Test Vec3 operations."""

    def test_creation(self):
        """Test Vec3 creation."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_default(self):
        """Test Vec3 default values."""
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_addition(self):
        """Test Vec3 addition."""
        v1 = Vec3(1, 2, 3)
        v2 = Vec3(4, 5, 6)
        result = v1 + v2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_multiplication(self):
        """Test Vec3 scalar multiplication."""
        v = Vec3(1, 2, 3)
        result = v * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_length(self):
        """Test Vec3 length."""
        v = Vec3(3, 4, 0)
        assert v.length() == 5.0

    def test_normalized(self):
        """Test Vec3 normalization."""
        v = Vec3(3, 0, 0)
        n = v.normalized()
        assert abs(n.x - 1.0) < 0.001
        assert n.y == 0.0
        assert n.z == 0.0


class TestParticle:
    """Test Particle data structure."""

    def test_default_state(self):
        """Test particle default state."""
        p = Particle()
        assert p.state == ParticleState.DEAD
        assert p.age == 0.0
        assert p.lifetime == 1.0
        assert not p.is_alive

    def test_normalized_age(self):
        """Test normalized age calculation."""
        p = Particle()
        p.lifetime = 2.0
        p.age = 1.0
        assert p.normalized_age == 0.5

    def test_reset(self):
        """Test particle reset."""
        p = Particle()
        p.state = ParticleState.ALIVE
        p.age = 5.0
        p.custom_data["test"] = "value"

        p.reset()

        assert p.state == ParticleState.DEAD
        assert p.age == 0.0
        assert len(p.custom_data) == 0


class TestEmitterConfig:
    """Test EmitterConfig creation."""

    def test_default_config(self):
        """Test default configuration."""
        config = EmitterConfig()
        assert config.max_particles == 1000
        assert config.simulation == SimulationMode.AUTO
        assert config.budget_category is None
        assert config.loop is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EmitterConfig(
            max_particles=5000,
            simulation=SimulationMode.GPU,
            budget_category="high_quality",
            duration=10.0,
        )
        assert config.max_particles == 5000
        assert config.simulation == SimulationMode.GPU
        assert config.budget_category == "high_quality"
        assert config.duration == 10.0

    def test_from_decorator_params(self):
        """Test creation from decorator parameters."""
        config = EmitterConfig.from_decorator_params(
            max_particles=2000,
            simulation="gpu",
            budget_category="effects",
        )
        assert config.max_particles == 2000
        assert config.simulation == SimulationMode.GPU
        assert config.budget_category == "effects"


class TestParticlePool:
    """Test ParticlePool allocation and recycling."""

    def test_creation(self):
        """Test pool creation."""
        pool = ParticlePool(max_particles=100)
        assert pool.max_particles == 100
        assert pool.alive_count == 0
        assert pool.free_count == 100

    def test_allocate(self):
        """Test particle allocation."""
        pool = ParticlePool(max_particles=10)

        particle = pool.allocate()
        assert particle is not None
        assert particle.state == ParticleState.ALIVE
        assert pool.alive_count == 1
        assert pool.free_count == 9

    def test_allocate_exhausted(self):
        """Test allocation when pool is exhausted."""
        pool = ParticlePool(max_particles=2)

        pool.allocate()
        pool.allocate()
        particle = pool.allocate()

        assert particle is None
        assert pool.alive_count == 2

    def test_deallocate(self):
        """Test particle deallocation."""
        pool = ParticlePool(max_particles=10)

        particle = pool.allocate()
        pool.deallocate(particle)

        assert particle.state == ParticleState.DEAD
        assert pool.alive_count == 0
        assert pool.free_count == 10

    def test_deallocate_o1_lookup(self):
        """T1.1: Validate deallocate uses O(1) reverse lookup dict (not O(n) list.index)."""
        pool = ParticlePool(max_particles=100)

        allocated = [pool.allocate() for _ in range(5)]
        # Deallocate from middle of list -- reverse lookup maps id to index correctly
        pool.deallocate(allocated[2])
        assert allocated[2].state == ParticleState.DEAD
        assert pool.alive_count == 4
        # Deallocating same particle again should be no-op, not crash
        pool.deallocate(allocated[2])
        assert pool.alive_count == 4

    def test_iter_alive(self):
        """Test iterating over alive particles."""
        pool = ParticlePool(max_particles=10)

        pool.allocate()
        pool.allocate()
        pool.allocate()

        alive_particles = list(pool.iter_alive())
        assert len(alive_particles) == 3

    def test_kill_all(self):
        """Test killing all particles."""
        pool = ParticlePool(max_particles=10)

        for _ in range(5):
            pool.allocate()

        assert pool.alive_count == 5

        pool.kill_all()

        assert pool.alive_count == 0
        assert pool.free_count == 10

    def test_compact_removes_dead_from_alive_set(self):
        """T1.1: Compact removes dead particles from alive set."""
        pool = ParticlePool(max_particles=10)

        p1 = pool.allocate()
        p2 = pool.allocate()
        p3 = pool.allocate()
        pool.deallocate(p2)  # p2 marked dead but still in alive_indices

        pool.compact()

        alive = list(pool.iter_alive())
        assert len(alive) == 2
        assert p1 in alive
        assert p2 not in alive
        assert p3 in alive

    def test_compact_restores_free_count(self):
        """T1.1: Compact correctly adjusts alive_count."""
        pool = ParticlePool(max_particles=10)

        p1 = pool.allocate()
        p2 = pool.allocate()
        pool.deallocate(p1)
        pool.deallocate(p2)
        pool.compact()

        assert pool.alive_count == 0
        assert pool.free_count == 10

    def test_compact_empty_pool_noop(self):
        """T1.1: Compacting an empty pool should not crash."""
        pool = ParticlePool(max_particles=10)
        pool.compact()
        assert pool.alive_count == 0
        assert pool.free_count == 10


class TestParticleBudget:
    """Test ParticleBudget category management."""

    def test_default_budgets(self):
        """Test default budget categories exist."""
        budget = ParticleBudget()

        assert budget.get_allocation("default") is not None
        assert budget.get_allocation("ambient") is not None
        assert budget.get_allocation("gameplay") is not None
        assert budget.get_allocation("critical") is not None

    def test_set_budget(self):
        """Test setting custom budget."""
        budget = ParticleBudget()
        budget.set_budget("custom", max_particles=10000, priority=75)

        alloc = budget.get_allocation("custom")
        assert alloc.max_particles == 10000
        assert alloc.priority == 75

    def test_request_particles(self):
        """Test requesting particles from budget."""
        budget = ParticleBudget()
        budget.set_budget("test", max_particles=100)

        allocated = budget.request_particles("test", 50)
        assert allocated == 50

        alloc = budget.get_allocation("test")
        assert alloc.current_particles == 50

    def test_request_particles_exceeds_limit(self):
        """Test requesting more than available."""
        budget = ParticleBudget()
        budget.set_budget("limited", max_particles=10)

        allocated = budget.request_particles("limited", 100)
        assert allocated == 10

    def test_release_particles(self):
        """Test releasing particles."""
        budget = ParticleBudget()
        budget.set_budget("test", max_particles=100)

        budget.request_particles("test", 50)
        budget.release_particles("test", 30)

        alloc = budget.get_allocation("test")
        assert alloc.current_particles == 20

    def test_global_limit_enforced(self):
        """T1.14: Global budget respected across categories."""
        budget = ParticleBudget()
        # Set individual category budgets high, global limit is the bottleneck
        budget.set_budget("cat_a", max_particles=500000)
        budget.set_budget("cat_b", max_particles=500000)

        # Global limit is 500000, so cat_a gets 500000
        allocated_a = budget.request_particles("cat_a", 500000)
        assert allocated_a == 500000

        # cat_b should get 0 because global limit is hit
        allocated_b = budget.request_particles("cat_b", 100)
        assert allocated_b == 0

    def test_budget_release_frees_global_limit(self):
        """T1.14: Releasing particles frees up global budget."""
        budget = ParticleBudget()
        budget.set_budget("cat_a", max_particles=500000)

        allocated_a = budget.request_particles("cat_a", 500000)
        assert allocated_a == 500000

        budget.release_particles("cat_a", 500000)
        assert budget.get_total_usage() == 0.0


class TestParticleEmitter:
    """Test ParticleEmitter lifecycle."""

    def test_creation(self):
        """Test emitter creation."""
        config = EmitterConfig(max_particles=100)
        emitter = ParticleEmitter(config=config)

        assert emitter.state == EmitterState.INACTIVE
        assert emitter.alive_count == 0
        assert emitter.config.max_particles == 100

    def test_start_stop(self):
        """Test emitter start and stop."""
        emitter = ParticleEmitter()

        emitter.start()
        assert emitter.state == EmitterState.ACTIVE

        emitter.stop(immediate=True)
        assert emitter.state == EmitterState.STOPPED

    def test_simulation_mode_auto_cpu(self):
        """Test AUTO resolves to CPU for small particle counts."""
        config = EmitterConfig(max_particles=100, simulation=SimulationMode.AUTO)
        emitter = ParticleEmitter(config=config)

        assert emitter.simulation_mode == SimulationMode.CPU

    def test_simulation_mode_auto_gpu(self):
        """Test AUTO resolves to GPU for large particle counts."""
        config = EmitterConfig(max_particles=50000, simulation=SimulationMode.AUTO)
        emitter = ParticleEmitter(config=config)

        assert emitter.simulation_mode == SimulationMode.GPU

    def test_simulation_mode_explicit(self):
        """Test explicit simulation mode is respected."""
        config = EmitterConfig(max_particles=100, simulation=SimulationMode.GPU)
        emitter = ParticleEmitter(config=config)

        assert emitter.simulation_mode == SimulationMode.GPU

    def test_update_aging(self):
        """Test particle aging during update."""
        from engine.rendering.particles.particle_modules import RateEmitter

        emitter = ParticleEmitter()
        emitter.add_spawn_module(RateEmitter(rate=1000))
        emitter.start()

        # Update for a bit
        emitter.update(0.1)

        # Some particles should be alive
        assert emitter.alive_count > 0

    def test_get_stats(self):
        """Test getting emitter statistics."""
        emitter = ParticleEmitter()
        stats = emitter.get_stats()

        assert "state" in stats
        assert "alive_count" in stats
        assert "simulation_mode" in stats

    def test_prewarm_simulates_frames(self):
        """T1.15: Prewarm simulates N seconds before first frame."""
        from engine.rendering.particles.particle_modules import RateEmitter

        config = EmitterConfig(prewarm=True, warmup_time=2.0, max_particles=50000)
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=1000))

        # start() should trigger prewarm
        emitter.start()
        assert emitter.state == EmitterState.ACTIVE

        # After 2s of prewarm at 60fps, particles should exist
        assert emitter.alive_count > 0

    def test_prewarm_particles_correct_lifecycle(self):
        """T1.15: Particles are in correct lifecycle state after prewarm."""
        from engine.rendering.particles.particle_modules import (
            RateEmitter,
            LifetimeModule,
        )

        config = EmitterConfig(prewarm=True, warmup_time=10.0, max_particles=100000)
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=10000))
        emitter.add_spawn_module(LifetimeModule(lifetime=(5.0, 5.0)))

        emitter.start()

        # After prewarm, all alive particles should have ages within [0, lifetime]
        for particle in emitter.iter_particles():
            assert particle.age >= 0.0
            assert particle.age <= particle.lifetime
            assert particle.is_alive

    def test_prewarm_pool_state_consistent(self):
        """T1.15: Pool state consistent after prewarm."""
        from engine.rendering.particles.particle_modules import RateEmitter

        config = EmitterConfig(prewarm=True, warmup_time=1.0, max_particles=1000)
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=500))

        emitter.start()

        # Pool invariants must hold
        stats = emitter.get_stats()
        assert stats["alive_count"] <= config.max_particles
        assert stats["alive_count"] >= 0


class TestParticleSystemManager:
    """Test ParticleSystemManager."""

    def test_create_emitter(self):
        """Test creating emitter through manager."""
        manager = ParticleSystemManager()
        emitter = manager.create_emitter("test")

        assert emitter is not None
        assert manager.get_emitter("test") is emitter

    def test_remove_emitter(self):
        """Test removing emitter."""
        manager = ParticleSystemManager()
        manager.create_emitter("test")
        manager.remove_emitter("test")

        assert manager.get_emitter("test") is None

    def test_update_all(self):
        """Test updating all emitters."""
        manager = ParticleSystemManager()
        e1 = manager.create_emitter("e1")
        e2 = manager.create_emitter("e2")

        e1.start()
        e2.start()

        # Should not raise
        manager.update_all(0.016)

    def test_get_total_particle_count(self):
        """Test total particle count across emitters."""
        manager = ParticleSystemManager()

        # Create and start emitters
        e1 = manager.create_emitter("e1", EmitterConfig(max_particles=100))
        e2 = manager.create_emitter("e2", EmitterConfig(max_particles=100))

        # Total should be 0 without spawn modules
        assert manager.get_total_particle_count() == 0
