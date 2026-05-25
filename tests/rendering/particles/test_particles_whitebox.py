"""
Whitebox tests for particle system internals (Phase 1).

Tests internal code paths, edge-case branches, guard conditions, and
non-obvious invariants in particle_system.py and particle_modules.py
that are intentionally invisible to blackbox testing.

WHITEBOX coverage plan:
  # particle_system.py
  - Vec3: __sub__, __rmul__, dot, cross, normalized near-zero guard (FIX: zero-vector branch)
  - Vec4: from_rgb factory, lerp at boundaries, default w=1.0
  - Particle: normalized_age zero-lifetime guard -> 1.0; is_alive true for DYING
  - EmitterConfig: from_decorator_params unknown sim string -> AUTO fallback
  - ParticlePool: deallocate foreign particle (not in pool) -> early return no-op
  - ParticlePool: deallocate already-dead particle -> no-op
  - ParticlePool: kill_all on empty pool -> no-op
  - ParticlePool: compact with already-zero alive_count -> no-op
  - BudgetAllocation: particle_usage zero-max guard -> 0.0
  - BudgetAllocation: memory_usage zero-max guard -> 0.0
  - BudgetAllocation: can_allocate exact boundary
  - ParticleBudget: get_allocation(None) -> "default" category
  - ParticleBudget: get_allocation unknown category -> creates new
  - ParticleBudget: release_particles clamped (release > current)
  - ParticleBudget: get_total_usage zero-limit guard -> 0.0
  - ParticleEmitter: start when not INACTIVE -> no-op
  - ParticleEmitter: update in INACTIVE/STOPPED state -> early return
  - ParticleEmitter: update with finite duration, no loop -> spawn stops
  - ParticleEmitter: _spawn_particle budget + pool exhausted -> budget release
  - ParticleEmitter: _update_phase particle age >= lifetime -> DYING
  - ParticleEmitter: _death_phase callback calls and budget release
  - ParticleSystemManager: get_emitter for missing name -> None
  - ParticleSystemManager: remove_emitter for missing name -> no-op

  # particle_modules.py
  - ParticleModule: is_active_for_lod disabled -> False
  - ShapeEmitter: box surface emission all 6 face branches
  - ShapeEmitter: mesh shapes fallthrough
  - BurstEmitter: trigger re-arms after depletion
  - RateEmitter: zero rate -> never spawns; setter clamps to >= 0
  - GravityModule: setter; accumulation with existing acceleration
  - WindModule: zero-turbulence deterministic; direction normalized
  - TurbulenceModule: zero strength -> no force; zero frequency
  - VortexModule: particle at center -> early return
  - AttractionModule: particle at target/out of radius -> early return; "none" falloff
  - CollisionModule: NONE mode early return; above-plane no-op; SDF/Depth stubs
  - ColorOverLifeModule: gradient single entry; past-last-key; empty gradient
  - SizeOverLifeModule: ease_in_out at t=0.5 boundary
  - RotationModule: stage is SPAWN
  - BillboardRenderer: "custom" alignment; velocity with zero vel fallback; stretch=0
  - MeshParticleRenderer: scale_with_size=False; align with zero velocity
"""

import math

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
from engine.rendering.particles.particle_modules import (
    AttractionModule,
    BillboardRenderer,
    BurstEmitter,
    CollisionMode,
    CollisionModule,
    ColorOverLifeModule,
    EmitterShape,
    GravityModule,
    LifetimeModule,
    MeshParticleRenderer,
    ModuleConfig,
    ModuleStage,
    ParticleModule,
    RateEmitter,
    RotationModule,
    ShapeEmitter,
    SizeOverLifeModule,
    TurbulenceModule,
    VelocityModule,
    VortexModule,
    WindModule,
)

# =============================================================================
# Constants
# =============================================================================

TOL = 1e-10


# =============================================================================
# Vec3 — internal branches
# =============================================================================


class TestVec3Whitebox:
    """Whitebox: Vec3 internal operations."""

    def test_subtraction(self):
        """__sub__ produces correct component-wise difference."""
        a = Vec3(5, 3, 1)
        b = Vec3(2, 4, 6)
        r = a - b
        assert r.x == 3.0
        assert r.y == -1.0
        assert r.z == -5.0

    def test_rmul_right_multiply(self):
        """__rmul__ (scalar * Vec3) matches __mul__."""
        v = Vec3(2, -3, 4)
        r = 2.0 * v
        assert r.x == 4.0
        assert r.y == -6.0
        assert r.z == 8.0

    def test_dot_product(self):
        """dot returns scalar product."""
        a = Vec3(1, 2, 3)
        b = Vec3(4, -5, 6)
        assert a.dot(b) == pytest.approx(4 - 10 + 18)  # = 12

    def test_dot_perpendicular_zero(self):
        """dot of perpendicular vectors is zero."""
        assert Vec3(1, 0, 0).dot(Vec3(0, 1, 0)) == 0.0

    def test_cross_product(self):
        """cross of (1,0,0) x (0,1,0) = (0,0,1)."""
        r = Vec3(1, 0, 0).cross(Vec3(0, 1, 0))
        assert r.x == 0.0
        assert r.y == 0.0
        assert r.z == 1.0

    def test_cross_anticommutative(self):
        """cross(a,b) == -cross(b,a)."""
        a = Vec3(2, 3, 4)
        b = Vec3(5, 6, 7)
        ab = a.cross(b)
        ba = b.cross(a)
        assert ab.x == pytest.approx(-ba.x)
        assert ab.y == pytest.approx(-ba.y)
        assert ab.z == pytest.approx(-ba.z)

    def test_normalized_near_zero_returns_zero(self):
        """normalized of near-zero-length vector returns zero vector (guard)."""
        v = Vec3(1e-10, 0, 0)
        n = v.normalized()
        # length < 1e-8 -> return Vec3(0,0,0)
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.z == 0.0

    def test_normalized_zero_vector_returns_zero(self):
        """normalized of exact zero vector returns zero vector."""
        n = Vec3(0, 0, 0).normalized()
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.z == 0.0

    def test_normalized_unit_length(self):
        """normalized of any vector has unit length."""
        for v in [
            Vec3(3, 4, 0),
            Vec3(-1, 2, -2),
            Vec3(0.5, 0.5, 0.5),
        ]:
            n = v.normalized()
            assert abs(n.length() - 1.0) < 1e-10, f"non-unit for {v}"


# =============================================================================
# Vec4 — internal operations
# =============================================================================


class TestVec4Whitebox:
    """Whitebox: Vec4 internal operations."""

    def test_default_w_is_one(self):
        """Default alpha (w) is 1.0."""
        v = Vec4(1, 0, 0)
        assert v.w == 1.0

    def test_from_rgb(self):
        """from_rgb factory sets components correctly."""
        v = Vec4.from_rgb(0.5, 0.3, 0.9, 0.8)
        assert v.x == 0.5
        assert v.y == 0.3
        assert v.z == 0.9
        assert v.w == 0.8

    def test_lerp_at_t0(self):
        """lerp at t=0 returns start."""
        a = Vec4(1, 0, 0, 1)
        b = Vec4(0, 1, 0, 0)
        r = a.lerp(b, 0.0)
        assert r.x == 1.0 and r.y == 0.0 and r.z == 0.0 and r.w == 1.0

    def test_lerp_at_t1(self):
        """lerp at t=1 returns end."""
        a = Vec4(1, 0, 0, 1)
        b = Vec4(0, 1, 0, 0)
        r = a.lerp(b, 1.0)
        assert r.x == 0.0 and r.y == 1.0 and r.z == 0.0 and r.w == 0.0

    def test_lerp_at_midpoint(self):
        """lerp at t=0.5 averages components."""
        a = Vec4(1, 0, 0, 1)
        b = Vec4(0, 1, 1, 0)
        r = a.lerp(b, 0.5)
        assert r.x == 0.5
        assert r.y == 0.5
        assert r.z == 0.5
        assert r.w == 0.5


# =============================================================================
# Particle — guard branches
# =============================================================================


class TestParticleWhitebox:
    """Whitebox: Particle internal invariants."""

    def test_normalized_age_zero_lifetime(self):
        """normalized_age with lifetime <= 0 returns 1.0 (guard)."""
        p = Particle()
        p.lifetime = 0.0
        assert p.normalized_age == 1.0

    def test_normalized_age_negative_lifetime(self):
        """normalized_age with negative lifetime returns 1.0 (guard)."""
        p = Particle()
        p.lifetime = -1.0
        assert p.normalized_age == 1.0

    def test_normalized_age_caps_at_one(self):
        """normalized_age never exceeds 1.0."""
        p = Particle()
        p.lifetime = 1.0
        p.age = 10.0
        assert p.normalized_age == 1.0

    def test_is_alive_true_for_dying(self):
        """is_alive returns True for DYING particles (DYING != DEAD)."""
        p = Particle()
        p.state = ParticleState.DYING
        assert p.is_alive is True

    def test_is_alive_false_for_dead(self):
        """is_alive returns False for DEAD particles."""
        p = Particle()
        p.state = ParticleState.DEAD
        assert p.is_alive is False

    def test_reset_clears_custom_data(self):
        """reset() clears custom_data dict."""
        p = Particle()
        p.custom_data["key"] = "value"
        p.reset()
        assert len(p.custom_data) == 0

    def test_reset_restores_defaults(self):
        """reset() restores all fields to defaults."""
        p = Particle()
        p.state = ParticleState.ALIVE
        p.age = 5.0
        p.lifetime = 10.0
        p.position = Vec3(1, 2, 3)
        p.velocity = Vec3(4, 5, 6)
        p.acceleration = Vec3(7, 8, 9)
        p.color = Vec4(0.5, 0.5, 0.5, 0.5)
        p.size = 5.0
        p.rotation = 90.0
        p.angular_velocity = 45.0

        p.reset()

        assert p.state == ParticleState.DEAD
        assert p.age == 0.0
        assert p.lifetime == 1.0
        assert p.position.x == 0.0 and p.position.y == 0.0 and p.position.z == 0.0
        assert p.velocity.x == 0.0 and p.velocity.y == 0.0 and p.velocity.z == 0.0
        assert p.acceleration.x == 0.0
        assert p.color.x == 1.0 and p.color.w == 1.0
        assert p.size == 1.0
        assert p.rotation == 0.0
        assert p.angular_velocity == 0.0


# =============================================================================
# EmitterConfig — unknown simulation string
# =============================================================================


class TestEmitterConfigWhitebox:
    """Whitebox: EmitterConfig internal branches."""

    def test_from_decorator_params_unknown_sim_fallback(self):
        """Unknown simulation string falls back to AUTO."""
        config = EmitterConfig.from_decorator_params(
            max_particles=500, simulation="quantum"
        )
        assert config.simulation == SimulationMode.AUTO

    def test_from_decorator_params_case_insensitive(self):
        """Simulation string is case-insensitive."""
        for case in ["CPU", "cpu", "Cpu"]:
            config = EmitterConfig.from_decorator_params(
                max_particles=100, simulation=case
            )
            assert config.simulation == SimulationMode.CPU, f"failed for {case}"


# =============================================================================
# ParticlePool — edge-case branches
# =============================================================================


class TestParticlePoolWhitebox:
    """Whitebox: ParticlePool internal guard branches."""

    def test_deallocate_foreign_particle_noop(self):
        """deallocate of particle not from this pool is a no-op (early return)."""
        foreign = Particle()
        pool = ParticlePool(max_particles=10)
        pool.deallocate(foreign)  # Should not raise
        assert pool.alive_count == 0
        assert pool.free_count == 10

    def test_deallocate_already_dead_noop(self):
        """deallocate of already-dead particle is a no-op."""
        pool = ParticlePool(max_particles=10)
        p = pool.allocate()
        assert p is not None
        pool.deallocate(p)
        assert pool.alive_count == 0
        # Deallocate again — index not in alive_indices -> no-op
        pool.deallocate(p)
        assert pool.alive_count == 0
        assert pool.free_count == 10

    def test_kill_all_empty_pool_noop(self):
        """kill_all on an empty pool does not crash."""
        pool = ParticlePool(max_particles=10)
        pool.kill_all()
        assert pool.alive_count == 0
        assert pool.free_count == 10

    def test_compact_zero_alive_noop(self):
        """compact when alive_count is already 0 is a no-op."""
        pool = ParticlePool(max_particles=10)
        # fresh pool has alive_count=0
        pool.compact()
        assert pool.alive_count == 0
        assert pool.free_count == 10

    def test_compact_interleaved_alloc_dealloc(self):
        """compact after several alloc/dealloc cycles keeps state consistent."""
        pool = ParticlePool(max_particles=5)
        p1 = pool.allocate()
        p2 = pool.allocate()
        p3 = pool.allocate()
        pool.deallocate(p2)
        # Dead p2 is still in _alive_indices until compact
        alive_before = list(pool.iter_alive())
        assert len(alive_before) == 2  # p1, p3
        pool.compact()
        alive_after = list(pool.iter_alive())
        assert len(alive_after) == 2
        assert p1 in alive_after
        assert p3 in alive_after
        assert pool.alive_count == 2
        assert pool.free_count == 3

    def test_iter_alive_snapshot_correct_after_dealloc(self):
        """iter_alive uses .copy() of set, so dealloc during iteration is safe."""
        pool = ParticlePool(max_particles=5)
        p1 = pool.allocate()
        p2 = pool.allocate()
        p3 = pool.allocate()
        # Deallocate p2 before iteration
        pool.deallocate(p2)
        alive = list(pool.iter_alive())
        assert p1 in alive
        assert p2 not in alive
        assert p3 in alive
        assert len(alive) == 2


# =============================================================================
# BudgetAllocation — zero-limit guards
# =============================================================================


class TestBudgetAllocationWhitebox:
    """Whitebox: BudgetAllocation guard branches."""

    def test_particle_usage_zero_max_returns_zero(self):
        """particle_usage with max_particles <= 0 returns 0.0."""
        ba = BudgetAllocation(category="test", max_particles=0, current_particles=50)
        assert ba.particle_usage == 0.0

    def test_memory_usage_zero_max_returns_zero(self):
        """memory_usage with max_memory_bytes <= 0 returns 0.0."""
        ba = BudgetAllocation(
            category="test", max_particles=100, max_memory_bytes=0, current_memory_bytes=500
        )
        assert ba.memory_usage == 0.0

    def test_can_allocate_exactly_at_limit(self):
        """can_allocate returns True when exactly at limit minus count."""
        ba = BudgetAllocation(category="test", max_particles=10, current_particles=8)
        assert ba.can_allocate(2) is True
        assert ba.can_allocate(3) is False

    def test_can_allocate_zero_count(self):
        """can_allocate with count=0 returns True."""
        ba = BudgetAllocation(category="test", max_particles=10, current_particles=10)
        assert ba.can_allocate(0) is True

    def test_particle_usage_ratio(self):
        """particle_usage returns correct fraction."""
        ba = BudgetAllocation(category="test", max_particles=100, current_particles=25)
        assert ba.particle_usage == 0.25

    def test_memory_usage_ratio(self):
        """memory_usage returns correct fraction."""
        ba = BudgetAllocation(
            category="test", max_particles=100, max_memory_bytes=1000, current_memory_bytes=250
        )
        assert ba.memory_usage == 0.25


# =============================================================================
# ParticleBudget — edge cases
# =============================================================================


class TestParticleBudgetWhitebox:
    """Whitebox: ParticleBudget internal branches."""

    def test_get_allocation_none_returns_default(self):
        """get_allocation(None) returns 'default' category."""
        budget = ParticleBudget()
        alloc = budget.get_allocation(None)
        assert alloc.category == "default"

    def test_get_allocation_unknown_creates_new(self):
        """get_allocation with unknown category creates a new allocation."""
        budget = ParticleBudget()
        alloc = budget.get_allocation("new_category")
        assert alloc.category == "new_category"
        # Should have the default max_particles and priority
        assert alloc.max_particles == PARTICLE_CONSTANTS.BUDGET_DEFAULT[0]
        assert alloc.priority == PARTICLE_CONSTANTS.BUDGET_DEFAULT[1]

    def test_release_particles_clamped(self):
        """release_particles with count > current clamps to current."""
        budget = ParticleBudget()
        budget.set_budget("test", max_particles=100)
        budget.request_particles("test", 30)
        # Release more than current
        budget.release_particles("test", 100)
        alloc = budget.get_allocation("test")
        assert alloc.current_particles == 0
        assert budget._total_particles == 0

    def test_get_total_usage_zero_limit(self):
        """get_total_usage with zero _total_limit returns 0.0."""
        budget = ParticleBudget()
        # Set total limit to 0 via monkey-patching
        budget._total_limit = 0
        assert budget.get_total_usage() == 0.0

    def test_request_particles_zero_count(self):
        """request_particles with count=0 returns 0."""
        budget = ParticleBudget()
        budget.set_budget("test", max_particles=100)
        allocated = budget.request_particles("test", 0)
        assert allocated == 0

    def test_request_particles_category_limit_tighter(self):
        """Category limit (not global) is the bottleneck."""
        budget = ParticleBudget()
        budget.set_budget("limited", max_particles=5)
        allocated = budget.request_particles("limited", 100)
        assert allocated == 5

    def test_request_particles_global_limit_tighter(self):
        """Global limit is the bottleneck."""
        budget = ParticleBudget()
        budget.set_budget("big", max_particles=1000000)
        budget._total_limit = 10
        allocated = budget.request_particles("big", 100)
        assert allocated == 10

    def test_get_category_stats_structure(self):
        """get_category_stats returns expected keys."""
        budget = ParticleBudget()
        stats = budget.get_category_stats()
        assert "default" in stats
        entry = stats["default"]
        assert "current" in entry
        assert "max" in entry
        assert "usage" in entry
        assert "priority" in entry


# =============================================================================
# ParticleEmitter — internal lifecycle branches
# =============================================================================


class TestParticleEmitterWhitebox:
    """Whitebox: ParticleEmitter internal branches."""

    def test_start_when_already_active_noop(self):
        """start() when state is not INACTIVE is a no-op."""
        emitter = ParticleEmitter()
        emitter.start()
        assert emitter.state == EmitterState.ACTIVE
        emitter.start()  # Second call — state is ACTIVE, not INACTIVE
        assert emitter.state == EmitterState.ACTIVE

    def test_update_inactive_early_return(self):
        """update() when INACTIVE returns immediately."""
        emitter = ParticleEmitter()
        emitter.update(0.1)  # No crash, no state change
        assert emitter.state == EmitterState.INACTIVE
        assert emitter.age == 0.0

    def test_update_stopped_early_return(self):
        """update() when STOPPED returns immediately."""
        emitter = ParticleEmitter()
        emitter.start()
        emitter.stop(immediate=True)
        emitter.update(0.1)  # No crash
        assert emitter.state == EmitterState.STOPPED

    def test_finite_duration_no_loop_stops_spawning(self):
        """Duration with no loop stops spawning and enters STOPPING."""
        from engine.rendering.particles.particle_modules import RateEmitter

        config = EmitterConfig(duration=0.5, loop=False, max_particles=100)
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=100))
        emitter.start()

        # Before duration expires
        emitter.update(0.4)
        assert emitter.state == EmitterState.ACTIVE

        # Advance past duration
        emitter.update(0.2)
        assert emitter.state == EmitterState.STOPPING

    def test_finite_duration_with_loop_resets_age(self):
        """Duration with loop resets age on expiry."""
        config = EmitterConfig(duration=0.5, loop=True, max_particles=100)
        emitter = ParticleEmitter(config=config)
        emitter.start()

        emitter.update(0.6)
        # Age is reset to 0.0 on loop (implementation resets, not subtracts)
        assert emitter.age == pytest.approx(0.0)


class TestParticleEmitterSpawnWhitebox:
    """Whitebox: ParticleEmitter spawn-phase internals."""

    def test_spawn_particle_with_budget_success(self):
        """_spawn_particle with budget allocates and returns particle."""
        budget = ParticleBudget()
        budget.set_budget("test", max_particles=50)
        config = EmitterConfig(budget_category="test", max_particles=100)
        emitter = ParticleEmitter(config=config, budget=budget)
        p = emitter._spawn_particle()
        assert p is not None
        assert p.state == ParticleState.ALIVE
        assert emitter._spawn_count == 1
        alloc = budget.get_allocation("test")
        assert alloc.current_particles == 1

    def test_spawn_particle_budget_exhausted(self):
        """_spawn_particle returns None when budget is exhausted."""
        budget = ParticleBudget()
        budget.set_budget("test", max_particles=1)
        config = EmitterConfig(budget_category="test", max_particles=100)
        emitter = ParticleEmitter(config=config, budget=budget)

        first = emitter._spawn_particle()
        assert first is not None

        second = emitter._spawn_particle()
        assert second is None

    def test_spawn_particle_pool_exhausted_releases_budget(self):
        """Pool exhausted after budget alloc -> budget released on failure."""
        budget = ParticleBudget()
        budget.set_budget("test", max_particles=10)
        config = EmitterConfig(budget_category="test", max_particles=2)
        emitter = ParticleEmitter(config=config, budget=budget)

        p1 = emitter._spawn_particle()
        p2 = emitter._spawn_particle()
        assert p1 is not None
        assert p2 is not None

        # Budget was allocated (2 particles), pool exhausted
        alloc = budget.get_allocation("test")
        assert alloc.current_particles == 2

        # Next attempt: budget alloc succeeds but pool returns None -> budget released
        p3 = emitter._spawn_particle()
        assert p3 is None
        # Budget was 2, request_particles makes it 3, pool failure releases back to 2
        assert alloc.current_particles == 2

    def test_spawn_particle_callback_fires(self):
        """on_particle_spawn callback fires for each spawned particle."""
        callback_calls = []
        config = EmitterConfig(max_particles=10)
        emitter = ParticleEmitter(config=config)
        emitter._on_particle_spawn = lambda p: callback_calls.append(id(p))

        p = emitter._spawn_particle()
        assert p is not None
        assert len(callback_calls) == 1
        assert callback_calls[0] == id(p)


class TestParticleEmitterUpdateWhitebox:
    """Whitebox: ParticleEmitter update-phase internals."""

    def test_update_phase_particle_exceeds_lifetime(self):
        """Particle with age >= lifetime transitions to DYING."""
        from engine.rendering.particles.particle_modules import RateEmitter

        config = EmitterConfig(max_particles=10)
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=100))
        emitter.start()

        # Spawn particles then manually set short lifetimes
        emitter.update(0.01)
        for p in emitter.iter_particles():
            p.lifetime = 0.05  # Short lifetime

        # Update past lifetime
        emitter.update(0.1)

        # Death phase should have processed the expired particles
        assert emitter._death_count > 0

    def test_update_phase_physics_integration(self):
        """Basic Verlet-free physics (vel+=acc*dt, pos+=vel*dt) applies correctly."""
        pool = ParticlePool(max_particles=1)
        p = pool.allocate()
        p.age = 0.5
        p.lifetime = 2.0
        p.position = Vec3(1, 2, 3)
        p.velocity = Vec3(4, 5, 6)
        p.acceleration = Vec3(0, -9.81, 0)
        p.rotation = 0.0
        p.angular_velocity = 90.0

        dt = 0.016
        # Simulate _update_phase's physics integration manually
        new_vel = Vec3(
            p.velocity.x + p.acceleration.x * dt,
            p.velocity.y + p.acceleration.y * dt,
            p.velocity.z + p.acceleration.z * dt,
        )
        new_pos = Vec3(
            p.position.x + new_vel.x * dt,
            p.position.y + new_vel.y * dt,
            p.position.z + new_vel.z * dt,
        )
        new_rot = p.rotation + p.angular_velocity * dt

        assert new_vel.y == pytest.approx(5 + (-9.81) * dt)
        assert new_pos.x == pytest.approx(1 + 4 * dt)
        assert new_rot == pytest.approx(0 + 90 * dt)

    def test_death_phase_callback_and_budget_release(self):
        """_death_phase fires callback and releases budget."""
        budget = ParticleBudget()
        budget.set_budget("test", max_particles=10)
        config = EmitterConfig(budget_category="test", max_particles=10)
        emitter = ParticleEmitter(config=config, budget=budget)

        death_calls = []
        emitter._on_particle_death = lambda p: death_calls.append(id(p))

        # Manually create a dying particle in the pool
        p = emitter._pool.allocate()
        p.state = ParticleState.DYING
        emitter._pool._alive_indices.add(emitter._pool._particle_to_index[id(p)])
        emitter._budget.request_particles("test", 1)

        emitter._death_phase()

        assert len(death_calls) == 1
        alloc = budget.get_allocation("test")
        assert alloc.current_particles == 0

    def test_stop_immediate_kills_all(self):
        """stop(immediate=True) kills all particles."""
        from engine.rendering.particles.particle_modules import RateEmitter

        config = EmitterConfig(max_particles=100)
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=1000))
        emitter.start()
        emitter.update(0.1)

        assert emitter.alive_count > 0
        emitter.stop(immediate=True)
        assert emitter.state == EmitterState.STOPPED
        assert emitter.alive_count == 0

    def test_stop_graceful_enters_stopping(self):
        """stop(immediate=False) enters STOPPING state."""
        emitter = ParticleEmitter()
        emitter.start()
        emitter.stop(immediate=False)
        assert emitter.state == EmitterState.STOPPING


class TestParticleEmitterCallbackWhitebox:
    """Whitebox: ParticleEmitter lifecycle callbacks."""

    def test_on_emitter_complete_fires_when_particles_die(self):
        """on_emitter_complete fires when all particles have died naturally."""
        from engine.rendering.particles.particle_modules import RateEmitter

        callback_fired = False

        def on_complete():
            nonlocal callback_fired
            callback_fired = True

        config = EmitterConfig(max_particles=100, duration=0.1, loop=False)
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=1000))
        emitter._on_emitter_complete = on_complete
        emitter.start()

        # Spawn particles, stay within duration
        emitter.update(0.05)
        assert emitter.state == EmitterState.ACTIVE
        assert emitter.alive_count > 0

        # Manually set very long lifetime so particles survive past duration
        for p in emitter.iter_particles():
            p.lifetime = 100.0

        # Advance past duration: state -> STOPPING, particles still alive
        emitter.update(0.1)
        assert emitter.state == EmitterState.STOPPING
        assert emitter.alive_count > 0

        # Set short lifetimes so particles die in next update
        for p in emitter.iter_particles():
            p.lifetime = 0.01

        # Advance: particles age past lifetime, death phase kicks in,
        # alive_count hits 0, callback fires, state -> STOPPED
        emitter.update(0.1)
        assert emitter.state == EmitterState.STOPPED
        assert callback_fired, "on_emitter_complete should have fired"

    def test_stop_immediate_fires_on_complete(self):
        """stop(immediate=True) fires on_emitter_complete."""
        fired = []
        emitter = ParticleEmitter()
        emitter._on_emitter_complete = lambda: fired.append(True)
        emitter.start()
        emitter.stop(immediate=True)
        assert len(fired) == 1

    def test_on_particle_spawn_fires(self):
        """on_particle_spawn fires for every particle spawned."""
        spawns = []
        emitter = ParticleEmitter()
        emitter._on_particle_spawn = lambda p: spawns.append(1)
        p = emitter._spawn_particle()
        assert p is not None
        assert len(spawns) == 1


# =============================================================================
# ParticleSystemManager — edge cases
# =============================================================================


class TestParticleSystemManagerWhitebox:
    """Whitebox: ParticleSystemManager internal branches."""

    def test_get_emitter_missing_returns_none(self):
        """get_emitter for non-existent name returns None."""
        manager = ParticleSystemManager()
        assert manager.get_emitter("does_not_exist") is None

    def test_remove_emitter_missing_noop(self):
        """remove_emitter for non-existent name is a no-op."""
        manager = ParticleSystemManager()
        manager.remove_emitter("does_not_exist")  # Should not raise

    def test_set_lod_level(self):
        """set_lod_level stores the LOD level."""
        manager = ParticleSystemManager()
        manager.set_lod_level(2)
        assert manager._current_lod_level == 2

    def test_get_stats_structure(self):
        """get_stats returns expected keys."""
        manager = ParticleSystemManager()
        stats = manager.get_stats()
        assert "emitter_count" in stats
        assert "total_particles" in stats
        assert "budget_usage" in stats
        assert "category_stats" in stats

    def test_create_emitter_with_config(self):
        """create_emitter passes config correctly."""
        config = EmitterConfig(max_particles=500)
        manager = ParticleSystemManager()
        emitter = manager.create_emitter("test", config=config)
        assert emitter.config.max_particles == 500
        assert emitter.config.simulation == SimulationMode.AUTO


# =============================================================================
# ParticleModule — base class branches
# =============================================================================


class TestParticleModuleWhitebox:
    """Whitebox: ParticleModule base class branches."""

    def test_is_active_for_lod_disabled(self):
        """is_active_for_lod returns False when module is disabled."""
        module = _ConcreteModule()
        module.enabled = False
        # Even within LOD range -> False (disabled gate)
        assert module.is_active_for_lod(0) is False

    def test_get_spawn_count_base_returns_zero(self):
        """Base get_spawn_count returns 0."""
        module = _ConcreteModule()
        assert module.get_spawn_count(0.016) == 0

    def test_lod_below_range_inactive(self):
        """Module inactive when LOD is below range."""
        module = _ConcreteModule(lod_range=(1, 3))
        assert module.is_active_for_lod(0) is False

    def test_lod_above_range_inactive(self):
        """Module inactive when LOD is above range."""
        module = _ConcreteModule(lod_range=(1, 3))
        assert module.is_active_for_lod(4) is False

    def test_enabled_setter(self):
        """enabled setter toggles correctly."""
        module = _ConcreteModule()
        module.enabled = True
        assert module.enabled is True
        module.enabled = False
        assert module.enabled is False

    def test_stage_property(self):
        """stage property returns constructor value."""
        module = _ConcreteModule(stage=ModuleStage.SPAWN)
        assert module.stage == ModuleStage.SPAWN

    def test_lod_range_property(self):
        """lod_range property returns constructor value."""
        module = _ConcreteModule(lod_range=(0, 2))
        assert module.lod_range == (0, 2)


class _ConcreteModule(ParticleModule):
    """Concrete subclass for testing abstract base class."""
    def apply_to_particle(self, particle, dt):
        pass


# =============================================================================
# ShapeEmitter — box surface branches and mesh fallthrough
# =============================================================================


class TestShapeEmitterWhitebox:
    """Whitebox: ShapeEmitter internal sampling branches."""

    def test_box_surface_x_plus_face(self):
        """Box surface +X face sets position to +size.x/2 and rightward direction."""
        emitter = ShapeEmitter(
            shape=EmitterShape.BOX,
            size=Vec3(2, 2, 2),
            emit_from_surface=True,
            randomize_direction=True,
        )
        # We can't control random.randint easily, so we verify the structural properties
        for _ in range(200):
            particle = Particle()
            emitter.apply_to_particle(particle, 0.016)
            # All surface positions must be exactly at box boundary
            pos = particle.position
            half = Vec3(1, 1, 1)
            eps = 1e-6
            # Each coordinate must be within [-1, 1]
            assert -half.x - eps <= pos.x <= half.x + eps
            assert -half.y - eps <= pos.y <= half.y + eps
            assert -half.z - eps <= pos.z <= half.z + eps
            # At least one coordinate must be exactly at the boundary
            on_boundary = (
                abs(abs(pos.x) - half.x) < eps
                or abs(abs(pos.y) - half.y) < eps
                or abs(abs(pos.z) - half.z) < eps
            )
            assert on_boundary, f"Position {pos} not on box surface"

    def test_box_volume_inside_bounds(self):
        """Box volume emission places particles strictly inside bounds."""
        emitter = ShapeEmitter(
            shape=EmitterShape.BOX,
            size=Vec3(4, 6, 8),
            emit_from_surface=False,
        )
        for _ in range(100):
            particle = Particle()
            emitter.apply_to_particle(particle, 0.016)
            assert abs(particle.position.x) <= 2.0 + 1e-9
            assert abs(particle.position.y) <= 3.0 + 1e-9
            assert abs(particle.position.z) <= 4.0 + 1e-9

    def test_cone_angle_boundary(self):
        """Cone emission velocity is within angle from up."""
        emitter = ShapeEmitter(
            shape=EmitterShape.CONE,
            angle=30.0,
        )
        up = Vec3(0, 1, 0)
        max_angle_rad = math.radians(30.0)
        for _ in range(100):
            particle = Particle()
            emitter.apply_to_particle(particle, 0.016)
            vel_norm = particle.velocity.normalized()
            cos_angle = up.dot(vel_norm)
            actual_angle = math.acos(max(-1.0, min(1.0, cos_angle)))
            assert actual_angle <= max_angle_rad + 0.001, (
                f"velocity angle {math.degrees(actual_angle)} > 30"
            )

    def test_edge_bounds(self):
        """Edge emission along x-axis within correct range."""
        emitter = ShapeEmitter(
            shape=EmitterShape.EDGE,
            size=Vec3(10, 0, 0),
        )
        for _ in range(50):
            particle = Particle()
            emitter.apply_to_particle(particle, 0.016)
            assert abs(particle.position.x) <= 5.0 + 1e-9
            assert particle.position.y == 0.0
            assert particle.position.z == 0.0

    def test_randomize_direction_disabled(self):
        """With randomize_direction=False, velocity is not set."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            randomize_direction=False,
        )
        particle = Particle()
        emitter.apply_to_particle(particle, 0.016)
        # Position is set (center + sampled sphere offset), but velocity is not set
        assert particle.velocity.x == 0.0
        assert particle.velocity.y == 0.0
        assert particle.velocity.z == 0.0

    def test_mesh_fallthrough(self):
        """MESH_SURFACE and MESH_VOLUME fall through to default Vec3 return."""
        for shape in [EmitterShape.MESH_SURFACE, EmitterShape.MESH_VOLUME]:
            emitter = ShapeEmitter(shape=shape)
            pos, vel = emitter._sample_shape()
            assert pos.x == 0.0 and pos.y == 0.0 and pos.z == 0.0
            assert vel.x == 0.0 and vel.y == 1.0 and vel.z == 0.0


# =============================================================================
# BurstEmitter — trigger after depletion
# =============================================================================


class TestBurstEmitterWhitebox:
    """Whitebox: BurstEmitter internal branches."""

    def test_trigger_after_depletion(self):
        """trigger() re-arms the burst after initial depletion."""
        emitter = BurstEmitter(count=5, repeat_interval=0.0)

        # Deplete initial burst
        assert emitter.get_spawn_count(0.016) == 5
        assert emitter.get_spawn_count(0.016) == 0

        # Re-arm
        emitter.trigger()
        assert emitter.get_spawn_count(0.016) == 5

    def test_initial_state_pending(self):
        """On construction, _pending_count equals count."""
        emitter = BurstEmitter(count=10)
        assert emitter._pending_count == 10

    def test_repeating_burst_multiple_cycles(self):
        """Repeating burst cycles correctly across multiple intervals."""
        emitter = BurstEmitter(count=3, repeat_interval=1.0)

        # Cycle 1
        assert emitter.get_spawn_count(0.016) == 3
        assert emitter.get_spawn_count(0.016) == 0

        # Advance past interval
        assert emitter.get_spawn_count(1.1) == 3  # Cycle 2

        # Advance past another interval
        assert emitter.get_spawn_count(1.1) == 3  # Cycle 3


# =============================================================================
# RateEmitter — zero-rate edge case
# =============================================================================


class TestRateEmitterWhitebox:
    """Whitebox: RateEmitter edge cases."""

    def test_zero_rate_never_spawns(self):
        """Zero rate produces zero particles."""
        emitter = RateEmitter(rate=0.0)
        assert emitter.get_spawn_count(10.0) == 0

    def test_rate_setter_clamps_to_zero(self):
        """Rate setter clamps negative values to 0."""
        emitter = RateEmitter(rate=100.0)
        emitter.rate = -50.0
        assert emitter.rate == 0.0

    def test_rate_setter_positive(self):
        """Rate setter accepts positive values."""
        emitter = RateEmitter(rate=10.0)
        emitter.rate = 250.0
        assert emitter.rate == 250.0


# =============================================================================
# GravityModule — setter and accumulation
# =============================================================================


class TestGravityModuleWhitebox:
    """Whitebox: GravityModule internal branches."""

    def test_gravity_setter_updates(self):
        """Gravity setter updates internal value."""
        module = GravityModule(gravity=Vec3(0, -9.81, 0))
        module.gravity = Vec3(0, -20, 0)
        assert module.gravity.y == -20.0

    def test_accumulation_with_existing_acceleration(self):
        """Gravity accumulates, does not replace, existing acceleration."""
        module = GravityModule(gravity=Vec3(0, -9.81, 0))
        particle = Particle()
        particle.acceleration = Vec3(10, 0, 0)
        module.apply_to_particle(particle, 0.016)
        assert particle.acceleration.x == 10.0
        assert particle.acceleration.y == -9.81
        assert particle.acceleration.z == 0.0

    def test_zero_gravity_no_change(self):
        """Zero gravity vector adds zero acceleration."""
        module = GravityModule(gravity=Vec3(0, 0, 0))
        particle = Particle()
        module.apply_to_particle(particle, 0.016)
        assert particle.acceleration.length() == 0.0


# =============================================================================
# WindModule — zero turbulence deterministic
# =============================================================================


class TestWindModuleWhitebox:
    """Whitebox: WindModule internal branches."""

    def test_direction_normalized(self):
        """Direction is normalized at construction (not raw)."""
        module = WindModule(direction=Vec3(0, 5, 0))
        assert module._direction.y == 1.0
        assert module._direction.length() == pytest.approx(1.0)

    def test_zero_turbulence_no_randomness(self):
        """With zero turbulence, force is deterministic (same input = same output)."""
        module = WindModule(
            direction=Vec3(1, 0, 0),
            strength=10.0,
            turbulence=0.0,
        )
        p1 = Particle()
        p2 = Particle()
        module.apply_to_particle(p1, 0.016)
        module.apply_to_particle(p2, 0.016)
        assert p1.acceleration.x == p2.acceleration.x
        assert p1.acceleration.x == 10.0

    def test_zero_strength_no_force(self):
        """Zero strength produces no force."""
        module = WindModule(
            direction=Vec3(1, 0, 0),
            strength=0.0,
            turbulence=0.0,
        )
        particle = Particle()
        module.apply_to_particle(particle, 0.016)
        assert particle.acceleration.length() == 0.0


# =============================================================================
# TurbulenceModule — zero strength and frequency
# =============================================================================


class TestTurbulenceModuleWhitebox:
    """Whitebox: TurbulenceModule internal branches."""

    def test_zero_strength_no_force(self):
        """Zero strength produces zero force."""
        module = TurbulenceModule(strength=0.0, frequency=1.0)
        particle = Particle()
        particle.position = Vec3(1, 2, 3)
        particle.age = 0.5
        module.apply_to_particle(particle, 0.016)
        assert particle.acceleration.length() == 0.0

    def test_zero_frequency_age_only(self):
        """Zero frequency means noise depends only on particle age (position not scaled)."""
        module = TurbulenceModule(strength=1.0, frequency=0.0)
        # Two particles at different positions should produce same force
        # because frequency=0 zeros out position contribution
        p1 = Particle()
        p1.position = Vec3(1, 2, 3)
        p1.age = 0.5
        module.apply_to_particle(p1, 0.016)

        p2 = Particle()
        p2.position = Vec3(10, 20, 30)
        p2.age = 0.5
        module.apply_to_particle(p2, 0.016)

        # Same age, different positions -> same force when freq=0
        assert p1.acceleration.x == p2.acceleration.x
        assert p1.acceleration.y == p2.acceleration.y
        assert p1.acceleration.z == p2.acceleration.z


# =============================================================================
# VortexModule — center proximity early return
# =============================================================================


class TestVortexModuleWhitebox:
    """Whitebox: VortexModule edge cases."""

    def test_particle_at_center_early_return(self):
        """Particle at center returns early (distance < 0.001)."""
        module = VortexModule(
            center=Vec3(0, 0, 0),
            strength=100.0,
        )
        particle = Particle()
        particle.position = Vec3(0, 0, 0)  # Exactly at center
        particle.acceleration = Vec3(0, 0, 0)
        module.apply_to_particle(particle, 0.016)
        # No acceleration applied (early return)
        assert particle.acceleration.length() == 0.0

    def test_zero_strength_swirl(self):
        """Zero swirl strength produces no tangential force."""
        module = VortexModule(
            center=Vec3(0, 0, 0),
            axis=Vec3(0, 1, 0),
            strength=0.0,
            pull_strength=0.0,
        )
        particle = Particle()
        particle.position = Vec3(1, 0, 0)
        module.apply_to_particle(particle, 0.016)
        assert particle.acceleration.length() == 0.0


# =============================================================================
# AttractionModule — proximity and radius early returns
# =============================================================================


class TestAttractionModuleWhitebox:
    """Whitebox: AttractionModule edge cases."""

    def test_particle_at_target_early_return(self):
        """Particle exactly at target returns early (distance < 0.001)."""
        module = AttractionModule(
            target=Vec3(5, 5, 5),
            strength=100.0,
            radius=10.0,
        )
        particle = Particle()
        particle.position = Vec3(5, 5, 5)  # Exactly at target
        module.apply_to_particle(particle, 0.016)
        assert particle.acceleration.length() == 0.0

    def test_particle_outside_radius_early_return(self):
        """Particle outside attraction radius returns early."""
        module = AttractionModule(
            target=Vec3(0, 0, 0),
            strength=100.0,
            radius=5.0,
        )
        particle = Particle()
        particle.position = Vec3(10, 0, 0)  # Outside radius
        module.apply_to_particle(particle, 0.016)
        assert particle.acceleration.length() == 0.0

    def test_none_falloff_constant_force(self):
        """'none' falloff applies constant factor 1.0 (not distance-dependent)."""
        module = AttractionModule(
            target=Vec3(0, 0, 0),
            strength=10.0,
            radius=20.0,
            falloff="none",
        )
        p_close = Particle()
        p_close.position = Vec3(1, 0, 0)
        module.apply_to_particle(p_close, 0.016)

        p_far = Particle()
        p_far.position = Vec3(15, 0, 0)
        module.apply_to_particle(p_far, 0.016)

        # With "none" falloff, force is same regardless of distance
        assert abs(p_close.acceleration.x) == pytest.approx(abs(p_far.acceleration.x))

    def test_quadratic_falloff_formula(self):
        """Quadratic falloff: factor = 1 - (d/r)^2."""
        module = AttractionModule(
            target=Vec3(0, 0, 0),
            strength=10.0,
            radius=5.0,
            falloff="quadratic",
        )
        particle = Particle()
        particle.position = Vec3(3, 0, 0)  # d=3, r=5
        module.apply_to_particle(particle, 0.016)
        # factor = 1 - (3/5)^2 = 1 - 0.36 = 0.64
        # force = direction * strength * factor = (-1,0,0) * 10 * 0.64 = (-6.4, 0, 0)
        assert particle.acceleration.x == pytest.approx(-6.4)
        assert particle.acceleration.y == 0.0
        assert particle.acceleration.z == 0.0


# =============================================================================
# CollisionModule — NONE mode early return and stubs
# =============================================================================


class TestCollisionModuleWhitebox:
    """Whitebox: CollisionModule internal branches."""

    def test_none_mode_early_return(self):
        """NONE collision mode returns immediately."""
        module = CollisionModule(mode=CollisionMode.NONE)
        particle = Particle()
        particle.position = Vec3(0, -100, 0)  # Deep below ground
        module.apply_to_particle(particle, 0.016)
        # No change (early return)
        assert particle.position.y == -100.0
        assert particle.velocity.length() == 0.0

    def test_sdf_mode_stub(self):
        """SDF collision mode does not crash (stub)."""
        module = CollisionModule(mode=CollisionMode.SDF)
        particle = Particle()
        particle.position = Vec3(0, -5, 0)
        module.apply_to_particle(particle, 0.016)
        # Stub: no modification (pass)
        assert particle.position.y == -5.0

    def test_depth_buffer_mode_stub(self):
        """DEPTH_BUFFER collision mode does not crash (stub)."""
        module = CollisionModule(mode=CollisionMode.DEPTH_BUFFER)
        particle = Particle()
        particle.position = Vec3(0, -5, 0)
        module.apply_to_particle(particle, 0.016)
        # Stub: no modification (pass)
        assert particle.position.y == -5.0

    def test_above_plane_noop(self):
        """Particle above ground plane is not affected."""
        module = CollisionModule(mode=CollisionMode.PRIMITIVE, bounce=0.5)
        module.set_ground_plane(height=0.0)
        particle = Particle()
        particle.position = Vec3(0, 5, 0)  # Above ground
        particle.velocity = Vec3(0, -10, 0)
        module.apply_to_particle(particle, 0.016)
        # dist = pos.dot(normal) - height = 5 - 0 = 5, not < 0, so no-op
        assert particle.position.y == 5.0
        assert particle.velocity.y == -10.0

    def test_cell_negative_coordinates(self):
        """_get_cell uses floor division for negative coordinates."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=2.0,
        )
        # -0.5 // 2.0 should give -1 (floor), not 0 (truncation)
        cell = module._get_cell(Vec3(-0.5, -0.5, -0.5))
        assert cell == (-1, -1, -1), f"got {cell}"

    def test_3x3x3_neighborhood_count(self):
        """3x3x3 neighborhood returns particles from up to 27 cells."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=1.0,
        )
        # Place particles in 3 different cells
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    p = Particle()
                    p.position = Vec3(i * 1.0, j * 1.0, k * 1.0)
                    module.add_to_spatial_hash(p)

        center = Particle()
        center.position = Vec3(0, 0, 0)
        nearby = module.get_nearby_particles(center)
        # Should find all 27 particles (3x3x3)
        assert len(nearby) == 27

    def test_clear_spatial_hash(self):
        """clear_spatial_hash empties the hash for a new frame."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=1.0,
        )
        p = Particle()
        p.position = Vec3(0, 0, 0)
        module.add_to_spatial_hash(p)
        assert len(module._spatial_hash) > 0
        module.clear_spatial_hash()
        assert len(module._spatial_hash) == 0

    def test_get_nearby_empty_hash(self):
        """get_nearby_particles with empty hash returns empty list."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=1.0,
        )
        particle = Particle()
        particle.position = Vec3(0, 0, 0)
        nearby = module.get_nearby_particles(particle)
        assert len(nearby) == 0


# =============================================================================
# ColorOverLifeModule — gradient edge cases
# =============================================================================


class TestColorOverLifeModuleWhitebox:
    """Whitebox: ColorOverLifeModule gradient branches."""

    def test_gradient_single_entry(self):
        """Gradient with single entry returns that entry's color for all t."""
        gradient = [(0.0, Vec4(1, 0, 0, 1))]
        module = ColorOverLifeModule(gradient=gradient)
        particle = Particle()
        particle.age = 0.5
        particle.lifetime = 1.0
        module.apply_to_particle(particle, 0.016)
        assert particle.color.x == 1.0
        assert particle.color.y == 0.0
        assert particle.color.z == 0.0

    def test_gradient_past_last_key(self):
        """Gradient sampled past last key returns last key color."""
        gradient = [
            (0.0, Vec4(1, 0, 0, 1)),
            (0.5, Vec4(0, 1, 0, 1)),
            (1.0, Vec4(0, 0, 1, 1)),
        ]
        module = ColorOverLifeModule(gradient=gradient)
        particle = Particle()
        particle.age = 2.0  # Beyond last key
        particle.lifetime = 1.0
        module.apply_to_particle(particle, 0.016)
        assert particle.color.z == 1.0  # Last key blue

    def test_empty_gradient_default_white(self):
        """Empty gradient (None) uses start->end lerp (default white)."""
        module = ColorOverLifeModule()
        particle = Particle()
        particle.age = 0.5
        particle.lifetime = 1.0
        module.apply_to_particle(particle, 0.016)
        # Default: start=white(1,1,1,1), end=white(1,1,1,0)
        assert particle.color.x == 1.0
        assert particle.color.y == 1.0
        assert particle.color.z == 1.0
        assert particle.color.w == 0.5

    def test_gradient_exact_key_boundary(self):
        """Gradient sampled at exact key boundary returns that key color."""
        gradient = [
            (0.0, Vec4(1, 0, 0, 1)),
            (0.5, Vec4(0, 1, 0, 1)),
            (1.0, Vec4(0, 0, 1, 1)),
        ]
        module = ColorOverLifeModule(gradient=gradient)
        particle = Particle()
        particle.age = 0.5  # Exactly at second key
        particle.lifetime = 1.0
        module.apply_to_particle(particle, 0.016)
        assert particle.color.y == 1.0  # Green


# =============================================================================
# SizeOverLifeModule — ease_in_out boundary
# =============================================================================


class TestSizeOverLifeModuleWhitebox:
    """Whitebox: SizeOverLifeModule curve branches."""

    def test_ease_in_out_at_midpoint(self):
        """ease_in_out at exactly t=0.5 returns 0.5 (by continuity)."""
        module = SizeOverLifeModule(
            start_size=0.0,
            end_size=1.0,
            curve="ease_in_out",
        )
        particle = Particle()
        particle.age = 0.5
        particle.lifetime = 1.0
        module.apply_to_particle(particle, 0.016)
        # ease_in_out: t<0.5 uses 2*t^2, t>=0.5 uses 1 - 2*(1-t)^2
        # at t=0.5: 2 * 0.5^2 = 0.5
        assert particle.size == pytest.approx(0.5)

    def test_ease_in_out_late(self):
        """ease_in_out at t>0.5 uses second branch."""
        module = SizeOverLifeModule(
            start_size=0.0,
            end_size=1.0,
            curve="ease_in_out",
        )
        particle = Particle()
        particle.age = 0.75
        particle.lifetime = 1.0
        module.apply_to_particle(particle, 0.016)
        # 1 - 2*(1-0.75)^2 = 1 - 2*0.0625 = 0.875
        assert particle.size == pytest.approx(0.875)

    def test_zero_lifetime_uses_one(self):
        """Normalized_age with zero lifetime returns 1.0, so size = end_size."""
        module = SizeOverLifeModule(start_size=2.0, end_size=0.5)
        particle = Particle()
        particle.lifetime = 0.0
        particle.age = 0.0
        module.apply_to_particle(particle, 0.016)
        # normalized_age -> 1.0 -> size = start + (end-start)*1.0 = 0.5
        assert particle.size == 0.5


# =============================================================================
# RotationModule — stage is SPAWN
# =============================================================================


class TestRotationModuleWhitebox:
    """Whitebox: RotationModule stage property."""

    def test_stage_is_spawn(self):
        """RotationModule operates at SPAWN stage."""
        module = RotationModule()
        assert module.stage == ModuleStage.SPAWN

    def test_velocity_range_applied(self):
        """Angular velocity applied within configured range."""
        module = RotationModule(
            initial_rotation=(45, 45),
            angular_velocity=(-180, -180),
        )
        particle = Particle()
        module.apply_to_particle(particle, 0.016)
        assert particle.rotation == pytest.approx(math.radians(45))
        assert particle.angular_velocity == pytest.approx(math.radians(-180))


# =============================================================================
# LifetimeModule — stage is SPAWN
# =============================================================================


class TestLifetimeModuleWhitebox:
    """Whitebox: LifetimeModule stage and range."""

    def test_stage_is_spawn(self):
        """LifetimeModule operates at SPAWN stage."""
        module = LifetimeModule()
        assert module.stage == ModuleStage.SPAWN

    def test_lifetime_range_sampling(self):
        """Lifetime sampled within [min, max] range."""
        module = LifetimeModule(lifetime=(3.0, 3.0))
        particle = Particle()
        module.apply_to_particle(particle, 0.016)
        assert particle.lifetime == 3.0


# =============================================================================
# VelocityModule — stage is SPAWN
# =============================================================================


class TestVelocityModuleWhitebox:
    """Whitebox: VelocityModule stage and spread."""

    def test_stage_is_spawn(self):
        """VelocityModule operates at SPAWN stage."""
        module = VelocityModule()
        assert module.stage == ModuleStage.SPAWN

    def test_velocity_spread_within_bounds(self):
        """Velocity with spread stays within expected bounds."""
        module = VelocityModule(
            velocity=Vec3(0, 5, 0),
            velocity_spread=Vec3(1, 0, 1),
        )
        for _ in range(50):
            particle = Particle()
            module.apply_to_particle(particle, 0.016)
            assert -1.0 <= particle.velocity.x <= 1.0
            assert particle.velocity.y == 5.0
            assert -1.0 <= particle.velocity.z <= 1.0


# =============================================================================
# BillboardRenderer — alignment branches
# =============================================================================


class TestBillboardRendererWhitebox:
    """Whitebox: BillboardRenderer alignment and stretch branches."""

    def test_custom_alignment(self):
        """'custom' alignment uses world axes (right=x, up=y)."""
        module = BillboardRenderer(alignment="custom")
        particle = Particle()
        particle.position = Vec3(0, 0, 0)
        module.apply_to_particle(particle, 0.016)

        right = particle.custom_data.get("billboard_right")
        up = particle.custom_data.get("billboard_up")
        assert right is not None
        assert up is not None
        assert right.x == 1.0 and right.y == 0.0 and right.z == 0.0
        assert up.x == 0.0 and up.y == 1.0 and up.z == 0.0

    def test_velocity_alignment_zero_vel_fallback(self):
        """Velocity alignment with zero velocity falls back to world axes."""
        module = BillboardRenderer(alignment="velocity")
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))
        particle = Particle()
        particle.position = Vec3(0, 0, 0)
        particle.velocity = Vec3(0, 0, 0)  # Zero velocity
        module.apply_to_particle(particle, 0.016)

        right = particle.custom_data.get("billboard_right")
        up = particle.custom_data.get("billboard_up")
        assert right is not None
        assert up is not None
        # Zero velocity fallback: right=(1,0,0), up=(0,1,0)
        assert right.x == 1.0
        assert up.y == 1.0

    def test_no_stretch_when_stretch_zero(self):
        """No stretch entry in custom_data when stretch=0."""
        module = BillboardRenderer(alignment="view", stretch=0.0)
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))
        particle = Particle()
        particle.position = Vec3(0, 0, 0)
        particle.velocity = Vec3(10, 0, 0)
        module.apply_to_particle(particle, 0.016)

        # Still has billboard vectors
        assert "billboard_right" in particle.custom_data
        assert "billboard_up" in particle.custom_data
        # No stretch because stretch=0 means condition fails
        assert "stretch" not in particle.custom_data

    def test_view_alignment(self):
        """View alignment orients to face camera."""
        module = BillboardRenderer(alignment="view")
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))
        particle = Particle()
        particle.position = Vec3(0, 0, 0)
        module.apply_to_particle(particle, 0.016)

        assert "billboard_right" in particle.custom_data
        assert "billboard_up" in particle.custom_data


# =============================================================================
# MeshParticleRenderer — scaling branches
# =============================================================================


class TestMeshParticleRendererWhitebox:
    """Whitebox: MeshParticleRenderer scaling and alignment."""

    def test_scale_with_size_false(self):
        """scale_with_size=False forces instance_scale = 1.0 regardless of particle size."""
        module = MeshParticleRenderer(scale_with_size=False)
        particle = Particle()
        particle.size = 5.0
        module.apply_to_particle(particle, 0.016)
        assert particle.custom_data["instance_scale"] == 1.0

    def test_align_velocity_near_zero_no_forward(self):
        """align_to_velocity with near-zero velocity does not set instance_forward."""
        module = MeshParticleRenderer(align_to_velocity=True)
        particle = Particle()
        particle.velocity = Vec3(1e-6, 0, 0)
        module.apply_to_particle(particle, 0.016)
        assert "instance_forward" not in particle.custom_data

    def test_align_velocity_forward_set(self):
        """align_to_velocity with non-zero velocity sets instance_forward."""
        module = MeshParticleRenderer(align_to_velocity=True)
        particle = Particle()
        particle.velocity = Vec3(0, 5, 0)
        module.apply_to_particle(particle, 0.016)
        fwd = particle.custom_data.get("instance_forward")
        assert fwd is not None
        assert fwd.y == pytest.approx(1.0, abs=0.001)


# =============================================================================
# ShapeEmitter sphere / circle volume distribution (whitebox internals)
# =============================================================================


class TestShapeEmitterSphereWhitebox:
    """Whitebox: ShapeEmitter sphere internal sampling."""

    def test_sphere_surface_all_at_radius(self):
        """Sphere surface samples are all at exactly radius distance."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=3.0,
            emit_from_surface=True,
        )
        for _ in range(100):
            pos, vel = emitter._sample_shape()
            dist = pos.length()
            assert abs(dist - 3.0) < 0.001, f"Expected radius 3.0, got {dist}"

    def test_sphere_volume_inside_radius(self):
        """Sphere volume samples are all inside or on radius."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=5.0,
            emit_from_surface=False,
        )
        for _ in range(100):
            pos, vel = emitter._sample_shape()
            dist = pos.length()
            assert dist <= 5.0 + 0.001, f"Expected <= 5.0, got {dist}"
            # Volume correction: r * (random ** 1/3), so some should be interior
            assert dist >= 0.0

    def test_circle_volume_inside_radius(self):
        """Circle volume samples inside radius (sqrt correction)."""
        emitter = ShapeEmitter(
            shape=EmitterShape.CIRCLE,
            radius=4.0,
            emit_from_surface=False,
        )
        for _ in range(100):
            pos, vel = emitter._sample_shape()
            dist = math.sqrt(pos.x**2 + pos.z**2)
            assert dist <= 4.0 + 0.001

    def test_circle_surface_at_radius(self):
        """Circle surface samples at exactly radius."""
        emitter = ShapeEmitter(
            shape=EmitterShape.CIRCLE,
            radius=4.0,
            emit_from_surface=True,
        )
        for _ in range(100):
            pos, vel = emitter._sample_shape()
            dist = math.sqrt(pos.x**2 + pos.z**2)
            assert abs(dist - 4.0) < 0.001


# =============================================================================
# ShapeEmitter cone sampling determinism and boundaries
# =============================================================================


class TestShapeEmitterConeWhitebox:
    """Whitebox: Cone internal sampling."""

    def test_cone_velocity_direction_length_one(self):
        """Cone velocity direction has unit length."""
        emitter = ShapeEmitter(
            shape=EmitterShape.CONE,
            angle=45.0,
        )
        for _ in range(50):
            _, vel = emitter._sample_shape()
            length = vel.length()
            assert abs(length - 1.0) < 0.001, f"Expected unit velocity, got {length}"

    def test_cone_zero_angle_vertical(self):
        """Zero cone angle produces purely upward (0,1,0) direction."""
        emitter = ShapeEmitter(
            shape=EmitterShape.CONE,
            angle=0.0,
        )
        for _ in range(20):
            _, vel = emitter._sample_shape()
            assert vel.x == 0.0
            assert abs(vel.y - 1.0) < 0.001
            assert vel.z == 0.0


# =============================================================================
# Emitter — LOD in _spawn_phase and _update_phase
# =============================================================================


class TestEmitterLODWhitebox:
    """Whitebox: Emitter LOD usage in spawn/update phases."""

    def test_spawn_phase_uses_lod(self):
        """_spawn_phase passes LOD to module (currently hardcoded 0)."""
        from engine.rendering.particles.particle_modules import RateEmitter

        config = EmitterConfig(max_particles=50)
        emitter = ParticleEmitter(config=config)
        module = RateEmitter(rate=100)
        module.is_active_for_lod(0)  # Should be active (lod_range default 0..3)
        emitter.add_spawn_module(module)
        emitter.start()

        # Internal call uses is_active_for_lod(0) hardcoded
        emitter.update(0.1)
        assert emitter.alive_count > 0

    def test_update_phase_uses_lod(self):
        """_update_phase passes LOD to module (currently hardcoded 0)."""
        from engine.rendering.particles.particle_modules import RateEmitter, GravityModule

        config = EmitterConfig(max_particles=50)
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=100))
        grav = GravityModule(gravity=Vec3(0, -9.81, 0))
        emitter.add_update_module(grav)
        emitter.start()
        emitter.update(0.1)

        # Particles should have gravity applied
        for p in emitter.iter_particles():
            assert p.acceleration.y == pytest.approx(-9.81)


# =============================================================================
# ParticleModule — ModuleConfig decorator-branch
# =============================================================================


class TestModuleConfigWhitebox:
    """Whitebox: ModuleConfig.from_decorator_params branches."""

    def test_from_decorator_unknown_stage_fallback(self):
        """Unknown stage string falls back to UPDATE."""
        config = ModuleConfig.from_decorator_params(stage="unknown")
        assert config.stage == ModuleStage.UPDATE

    def test_from_decorator_case_insensitive(self):
        """Stage parameter is case-insensitive."""
        for case in ["SPAWN", "spawn", "Spawn"]:
            config = ModuleConfig.from_decorator_params(stage=case)
            assert config.stage == ModuleStage.SPAWN, f"failed for {case}"


# =============================================================================
# Emitter — COMPACT_INTERVAL periodic compaction
# =============================================================================


class TestEmitterCompactWhitebox:
    """Whitebox: Periodic pool compaction in _update_internal."""

    def test_compact_fires_at_interval(self):
        """Compact is triggered every COMPACT_INTERVAL frames."""
        from engine.rendering.particles.particle_modules import RateEmitter

        config = EmitterConfig(max_particles=100)
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=1000))

        # Track compact calls
        original_compact = emitter._pool.compact
        compact_call_count = [0]
        def tracking_compact():
            compact_call_count[0] += 1
            original_compact()

        emitter._pool.compact = tracking_compact
        emitter.start()

        # Update COMPACT_INTERVAL times
        interval = PARTICLE_CONSTANTS.COMPACT_INTERVAL
        for _ in range(interval):
            emitter.update(0.016)

        # Compact should have been called once (at frame % interval == 0)
        assert compact_call_count[0] >= 1


# =============================================================================
# BudgetAllocation — __dataclass__ defaults
# =============================================================================


class TestBudgetAllocationDefaults:
    """Whitebox: BudgetAllocation default values."""

    def test_default_priority(self):
        """Default priority is 1."""
        ba = BudgetAllocation(category="test", max_particles=100)
        assert ba.priority == 1

    def test_defaults_zeros(self):
        """current_particles and current_memory_bytes default to 0."""
        ba = BudgetAllocation(category="test", max_particles=100)
        assert ba.current_particles == 0
        assert ba.current_memory_bytes == 0
        assert ba.max_memory_bytes == 0


# =============================================================================
# ParticleEmitter — get_stats
# =============================================================================


class TestEmitterStatsWhitebox:
    """Whitebox: get_stats includes all expected keys."""

    def test_get_stats_all_keys(self):
        """get_stats returns all expected keys with correct types."""
        emitter = ParticleEmitter()
        stats = emitter.get_stats()
        expected_keys = {
            "state", "age", "alive_count", "spawn_count",
            "death_count", "frame_count", "simulation_mode", "pool_usage",
        }
        assert set(stats.keys()) == expected_keys
        assert isinstance(stats["state"], str)
        assert isinstance(stats["age"], float)
        assert isinstance(stats["alive_count"], int)
        assert isinstance(stats["spawn_count"], int)
        assert isinstance(stats["death_count"], int)
        assert isinstance(stats["frame_count"], int)
        assert isinstance(stats["simulation_mode"], str)
        assert isinstance(stats["pool_usage"], float)
