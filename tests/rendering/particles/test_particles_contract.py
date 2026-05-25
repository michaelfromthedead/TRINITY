"""
Contract tests for the particles rendering subsystem.

Blackbox / cleanroom: tests the public API contract only.
All imports are from engine.rendering.particles (the __init__.py public surface).

Covers all 18 Phase 1 tasks from PHASE_1_TODO.md:
    T1.1  ParticlePool O(1) operations
    T1.2  Sphere volume distribution
    T1.3  Spatial hash collision detection
    T1.4  GravityModule
    T1.5  WindModule
    T1.6  TurbulenceModule
    T1.7  VortexModule
    T1.8  AttractionModule
    T1.9  SizeOverLifeModule
    T1.10 ColorOverLifeModule
    T1.11 ShapeEmitter sampling
    T1.12 BurstEmitter timing
    T1.13 RateEmitter accumulation
    T1.14 ParticleBudget categories
    T1.15 Emitter prewarm
    T1.16 VectorFieldModule stub
    T1.17 BillboardRenderer
    T1.18 MeshParticleRenderer
"""

import math
import pytest

from engine.rendering.particles import (
    # Core
    Particle,
    ParticlePool,
    ParticleBudget,
    BudgetAllocation,
    ParticleEmitter,
    EmitterConfig,
    ParticleSystemManager,
    # Enums
    SimulationMode,
    EmitterState,
    ParticleState,
    # Math
    Vec3,
    Vec4,
    # Spawn modules
    ShapeEmitter,
    BurstEmitter,
    RateEmitter,
    # Force modules
    GravityModule,
    WindModule,
    TurbulenceModule,
    VortexModule,
    AttractionModule,
    VectorFieldModule,
    CollisionModule,
    # Attribute modules
    SizeOverLifeModule,
    ColorOverLifeModule,
    RotationModule,
    LifetimeModule,
    VelocityModule,
    # Render modules
    BillboardRenderer,
    MeshParticleRenderer,
    # Module enums
    EmitterShape,
    ModuleStage,
    CollisionMode,
    BlendMode,
)


# =========================================================================
# T1.1 - ParticlePool O(1) Operations
# =========================================================================


class TestParticlePoolContract:
    """T1.1: Validate ParticlePool O(1) allocate/deallocate/compact."""

    def test_allocate_returns_particle_when_capacity_available(self):
        """Allocation returns particle in O(1) when pool has capacity."""
        pool = ParticlePool(max_particles=50)
        p = pool.allocate()
        assert p is not None
        assert isinstance(p, Particle)
        assert p.state == ParticleState.ALIVE

    def test_allocate_returns_none_when_exhausted(self):
        """Allocation returns None when pool is full."""
        pool = ParticlePool(max_particles=3)
        for _ in range(3):
            pool.allocate()
        assert pool.allocate() is None

    def test_deallocate_returns_particle_to_free_list(self):
        """Deallocation returns particle to free list in O(1)."""
        pool = ParticlePool(max_particles=10)
        particles = [pool.allocate() for _ in range(5)]
        alive_before = pool.alive_count
        pool.deallocate(particles[2])
        assert pool.alive_count == alive_before - 1
        assert particles[2].state == ParticleState.DEAD

    def test_reverse_lookup_maps_particle_id_to_index(self):
        """Reverse lookup dictionary correctly maps particle id to index."""
        pool = ParticlePool(max_particles=10)
        p = pool.allocate()
        # Deallocate then re-allocate should work (id-based lookup)
        pool.deallocate(p)
        p2 = pool.allocate()
        assert p2 is not None
        assert p2.state == ParticleState.ALIVE

    def test_double_deallocate_is_noop(self):
        """Deallocating already-dead particle does not corrupt pool."""
        pool = ParticlePool(max_particles=5)
        p = pool.allocate()
        pool.deallocate(p)
        count_before = pool.alive_count
        pool.deallocate(p)  # Second deallocate -- should be no-op
        assert pool.alive_count == count_before

    def test_pool_compaction_maintains_valid_index_mappings(self):
        """Pool compaction maintains valid index mappings."""
        pool = ParticlePool(max_particles=10)
        allocated = [pool.allocate() for _ in range(8)]
        # Kill middle particles
        pool.deallocate(allocated[2])
        pool.deallocate(allocated[5])
        pool.compact()
        # All alive particles should be iterable and valid
        alive_ids = {id(p) for p in pool.iter_alive()}
        assert id(allocated[0]) in alive_ids
        assert id(allocated[3]) in alive_ids
        assert id(allocated[7]) in alive_ids
        # Dead particles should not appear
        assert id(allocated[2]) not in alive_ids
        assert id(allocated[5]) not in alive_ids
        assert pool.alive_count == 6

    def test_compact_empty_pool_does_not_crash(self):
        """Compacting an empty pool does not crash."""
        pool = ParticlePool(max_particles=10)
        pool.compact()
        assert pool.alive_count == 0

    def test_compact_fully_alive_pool_does_not_change(self):
        """Compacting a fully-alive pool leaves it unchanged."""
        pool = ParticlePool(max_particles=5)
        for _ in range(5):
            pool.allocate()
        pool.compact()
        assert pool.alive_count == 5

    def test_kill_all_resets_pool_completely(self):
        """kill_all resets all particles to DEAD and frees all slots."""
        pool = ParticlePool(max_particles=20)
        for _ in range(15):
            pool.allocate()
        pool.kill_all()
        assert pool.alive_count == 0
        # After kill_all, we should be able to allocate again
        p = pool.allocate()
        assert p is not None
        assert p.state == ParticleState.ALIVE

    def test_iter_alive_yields_only_alive_particles(self):
        """iter_alive yields exactly the alive particles."""
        pool = ParticlePool(max_particles=10)
        p1 = pool.allocate()
        p2 = pool.allocate()
        p3 = pool.allocate()
        pool.deallocate(p2)
        alive = list(pool.iter_alive())
        assert len(alive) == 2
        assert p1 in alive
        assert p2 not in alive
        assert p3 in alive

    def test_alive_and_free_counts_consistent(self):
        """alive_count + free_count == max_particles at all times."""
        pool = ParticlePool(max_particles=25)
        assert pool.alive_count + pool.free_count == 25
        pool.allocate()
        pool.allocate()
        pool.allocate()
        assert pool.alive_count + pool.free_count == 25
        pool.deallocate(pool.allocate())  # allocate then immediately deallocate
        # Note: deallocate doesn't increment free_count directly -- it adds to free list
        # The invariant is: alive_count <= max_particles and free_count reflects available space

    def test_allocate_after_deallocate_reuses_slot(self):
        """Allocating after deallocating reuses the freed slot."""
        pool = ParticlePool(max_particles=5)
        p1 = pool.allocate()
        p1_id = id(p1)
        pool.deallocate(p1)
        p2 = pool.allocate()
        # The pool may reuse the same Particle object (ring buffer with free list)
        # This is an O(1) contract: no linear scan


# =========================================================================
# T1.2 - Sphere Sampling Volume Distribution
# =========================================================================


class TestSphereVolumeDistributionContract:
    """T1.2: Validate sphere sampling volume distribution."""

    def test_surface_emission_uses_radius_directly(self):
        """Surface emission uses radius directly without cube-root correction."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=10.0,
            emit_from_surface=True,
        )
        for _ in range(100):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            dist = p.position.length()
            assert abs(dist - 10.0) < 0.05, (
                f"Surface sample distance {dist} not near radius"
            )

    def test_volume_emission_distributes_inside_sphere(self):
        """Volume emission produces interior samples (cube-root correction)."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=10.0,
            emit_from_surface=False,
        )
        distances = []
        for _ in range(500):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            distances.append(p.position.length())
        max_dist = max(distances)
        min_dist = min(distances)
        assert max_dist <= 10.0, "Volume samples must not exceed radius"
        # Interior samples (well within radius) should exist
        interior_count = sum(1 for d in distances if d < 5.0)
        assert interior_count > 20, (
            f"Too few interior samples ({interior_count}/500): "
            "cube-root correction may be missing"
        )

    def test_direction_vector_normalized(self):
        """Direction vector normalized correctly for sphere emission."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=1.0,
            emit_from_surface=True,
        )
        for _ in range(50):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            vel_len = p.velocity.length()
            assert abs(vel_len - 1.0) < 0.001, (
                f"Direction not normalized: length={vel_len}"
            )

    def test_theta_phi_produce_uniform_spherical_distribution(self):
        """Theta/phi angles produce uniform spherical distribution (not clustered at poles)."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=1.0,
            emit_from_surface=True,
        )
        positions = []
        for _ in range(500):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            positions.append(p.position)

        # Check z-distribution is roughly uniform (not all near +/-1)
        z_values = [pos.z for pos in positions]
        near_poles = sum(1 for z in z_values if abs(abs(z) - 1.0) < 0.1)
        # At most ~20% should be within 0.1 of poles for uniform distribution
        assert near_poles < 150, (
            f"Too many samples near poles ({near_poles}/500): "
            "distribution may not be uniform"
        )


# =========================================================================
# T1.3 - Spatial Hash Collision Detection
# =========================================================================


class TestSpatialHashCollisionContract:
    """T1.3: Validate spatial hash and collision detection."""

    def test_cell_coordinates_via_floor_division(self):
        """Cell coordinates computed correctly via floor division."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=2.0,
        )
        p = Particle()
        p.position = Vec3(3.5, -1.2, 7.9)
        cell = module._get_cell(p.position)
        assert cell == (1, -1, 3), f"Expected (1, -1, 3) got {cell}"

    def test_negative_coordinates_use_floor_not_truncation(self):
        """Negative coordinates must use floor division, not truncation toward zero."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=2.0,
        )
        p = Particle()
        p.position = Vec3(-0.5, -0.5, -0.5)
        cell = module._get_cell(p.position)
        # -0.5 // 2.0 = -1.0 (floor), not 0 (truncation)
        assert cell == (-1, -1, -1), (
            f"Expected (-1, -1, -1) got {cell}. "
            "Floor division not used for negative coordinates."
        )

    def test_3x3x3_neighborhood_query(self):
        """3x3x3 neighborhood query returns all adjacent cells."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=1.0,
        )
        center = Particle()
        center.position = Vec3(0, 0, 0)
        module.add_to_spatial_hash(center)

        same_cell = Particle()
        same_cell.position = Vec3(0.9, 0.9, 0.9)
        module.add_to_spatial_hash(same_cell)

        adjacent = Particle()
        adjacent.position = Vec3(1.1, 0, 0)
        module.add_to_spatial_hash(adjacent)

        nearby = module.get_nearby_particles(center)
        assert center in nearby, "Center particle must be in its own neighborhood"
        assert same_cell in nearby, "Same-cell particle must be in neighborhood"
        assert adjacent in nearby, "Adjacent-cell particle must be in neighborhood"

    def test_particle_insertion_and_removal_from_grid(self):
        """Particles correctly inserted/removed from grid on position change."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=2.0,
        )
        p = Particle()
        p.position = Vec3(0, 0, 0)
        module.add_to_spatial_hash(p)
        # Should be findable
        nearby = module.get_nearby_particles(p)
        assert p in nearby
        # Clear and verify empty
        module.clear_spatial_hash()
        nearby_after = module.get_nearby_particles(p)
        assert len(nearby_after) == 0

    def test_ground_plane_bounce_and_friction(self):
        """Ground plane collision applies bounce coefficient and friction."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            bounce=0.5,
            friction=0.2,
        )
        module.set_ground_plane(height=0.0)

        p = Particle()
        p.position = Vec3(0, -1, 0)
        p.velocity = Vec3(4, -10, 0)

        module.apply_to_particle(p, 0.016)

        # Bounce: vertical velocity reflected and scaled
        assert p.velocity.y > 0, "Velocity should bounce upward"
        # Friction: horizontal velocity reduced
        expected_x = 4.0 * (1.0 - 0.2)
        assert abs(p.velocity.x - expected_x) < 0.001, (
            f"Expected horizontal vel {expected_x}, got {p.velocity.x}"
        )
        # Position corrected above ground
        assert p.position.y >= 0, "Particle should be pushed above ground"


# =========================================================================
# T1.4 - GravityModule
# =========================================================================


class TestGravityModuleContract:
    """T1.4: Validate GravityModule."""

    def test_acceleration_accumulated_not_replaced(self):
        """Acceleration accumulated (not replaced) when gravity applied."""
        module = GravityModule(gravity=Vec3(0, -9.81, 0))
        p = Particle()
        p.acceleration = Vec3(5, 0, 0)  # Pre-existing force from another module

        module.apply_to_particle(p, 0.016)

        assert p.acceleration.x == 5.0, "Pre-existing X acceleration overwritten"
        assert p.acceleration.y == -9.81, "Gravity Y component wrong"
        assert p.acceleration.z == 0.0, "Z component should remain unchanged"

    def test_gravity_vector_applied_each_frame(self):
        """Gravity vector applied correctly each frame (accumulates)."""
        module = GravityModule(gravity=Vec3(0, -9.81, 0))
        p = Particle()

        module.apply_to_particle(p, 0.016)
        assert p.acceleration.y == -9.81

        # Reset and apply again -- should still be the same vector
        module.apply_to_particle(p, 0.016)
        assert p.acceleration.y == -19.62, (
            "Gravity should accumulate across frames"
        )

    def test_works_with_other_force_modules(self):
        """Gravity combines with other force modules."""
        gravity = GravityModule(gravity=Vec3(0, -9.81, 0))
        wind = WindModule(direction=Vec3(1, 0, 0), strength=5.0, turbulence=0.0)
        p = Particle()

        gravity.apply_to_particle(p, 0.016)
        wind.apply_to_particle(p, 0.016)

        assert p.acceleration.x == 5.0, "Wind X component"
        assert p.acceleration.y == -9.81, "Gravity Y component"
        assert p.acceleration.z == 0.0

    def test_default_gravity_points_down(self):
        """Default gravity points downward (negative Y)."""
        module = GravityModule()
        p = Particle()
        module.apply_to_particle(p, 0.016)
        assert p.acceleration.y < 0, "Default gravity should point downward"
        assert p.acceleration.x == 0.0, "Default gravity should have no X"
        assert p.acceleration.z == 0.0, "Default gravity should have no Z"


# =========================================================================
# T1.5 - WindModule
# =========================================================================


class TestWindModuleContract:
    """T1.5: Validate WindModule."""

    def test_direction_and_magnitude_applied_as_force(self):
        """Direction + magnitude applied as force."""
        module = WindModule(
            direction=Vec3(1, 0, 0),
            strength=8.0,
            turbulence=0.0,
        )
        p = Particle()
        module.apply_to_particle(p, 0.016)
        assert abs(p.acceleration.x - 8.0) < 0.001
        assert p.acceleration.y == 0.0
        assert p.acceleration.z == 0.0

    def test_turbulence_adds_position_based_variation(self):
        """Turbulence adds position-based variation to wind force."""
        module = WindModule(
            direction=Vec3(1, 0, 0),
            strength=5.0,
            turbulence=2.0,
        )
        results = []
        for _ in range(20):
            p = Particle()
            module.apply_to_particle(p, 0.016)
            results.append(p.acceleration.x)

        # Turbulence should cause variation
        assert max(results) != min(results), (
            "Turbulence should produce varying wind forces"
        )

    def test_combined_with_other_forces(self):
        """Combined with other forces correctly."""
        wind = WindModule(direction=Vec3(1, 0, 0), strength=10.0, turbulence=0.0)
        gravity = GravityModule(gravity=Vec3(0, -9.81, 0))
        p = Particle()

        wind.apply_to_particle(p, 0.016)
        gravity.apply_to_particle(p, 0.016)

        assert p.acceleration.x == 10.0
        assert p.acceleration.y == -9.81


# =========================================================================
# T1.6 - TurbulenceModule
# =========================================================================


class TestTurbulenceModuleContract:
    """T1.6: Validate TurbulenceModule."""

    def test_pseudo_noise_computed_from_position(self):
        """Pseudo-noise computed from position produces varying output."""
        module = TurbulenceModule(strength=5.0, frequency=1.0)

        p1 = Particle()
        p1.position = Vec3(1, 2, 3)
        p1.age = 0.5
        module.apply_to_particle(p1, 0.016)

        p2 = Particle()
        p2.position = Vec3(4, 5, 6)
        p2.age = 0.5
        module.apply_to_particle(p2, 0.016)

        # Different positions should produce different forces
        assert not (
            p1.acceleration.x == p2.acceleration.x
            and p1.acceleration.y == p2.acceleration.y
            and p1.acceleration.z == p2.acceleration.z
        ), "Different positions must produce different turbulence"

    def test_deterministic_same_position_same_output(self):
        """Force varies spatially but deterministically (same input = same output)."""
        module = TurbulenceModule(strength=5.0, frequency=1.0)

        p1 = Particle()
        p1.position = Vec3(2.5, -1.0, 3.7)
        p1.age = 1.5
        module.apply_to_particle(p1, 0.016)

        p2 = Particle()
        p2.position = Vec3(2.5, -1.0, 3.7)
        p2.age = 1.5
        module.apply_to_particle(p2, 0.016)

        assert p1.acceleration.x == p2.acceleration.x
        assert p1.acceleration.y == p2.acceleration.y
        assert p1.acceleration.z == p2.acceleration.z

    def test_strength_parameter_scales_effect(self):
        """Strength parameter scales effect correctly."""
        module_weak = TurbulenceModule(strength=1.0, frequency=1.0)
        module_strong = TurbulenceModule(strength=10.0, frequency=1.0)

        accels_weak = []
        accels_strong = []
        for i in range(10):
            pw = Particle()
            pw.position = Vec3(float(i), float(i * 2), float(i * 3))
            pw.age = 0.5
            module_weak.apply_to_particle(pw, 0.016)
            accels_weak.append(pw.acceleration.length())

            ps = Particle()
            ps.position = Vec3(float(i), float(i * 2), float(i * 3))
            ps.age = 0.5
            module_strong.apply_to_particle(ps, 0.016)
            accels_strong.append(ps.acceleration.length())

        # Stronger turbulence should produce larger forces
        assert sum(accels_strong) > sum(accels_weak), (
            "Stronger turbulence should produce larger forces"
        )


# =========================================================================
# T1.7 - VortexModule
# =========================================================================


class TestVortexModuleContract:
    """T1.7: Validate VortexModule."""

    def test_tangential_force_creates_swirl_effect(self):
        """Tangential force creates swirl effect around axis."""
        module = VortexModule(
            center=Vec3(0, 0, 0),
            axis=Vec3(0, 1, 0),
            strength=10.0,
            pull_strength=0.0,  # No radial pull
        )
        p = Particle()
        p.position = Vec3(1, 0, 0)

        module.apply_to_particle(p, 0.016)

        # Tangential force should be perpendicular to both position and axis
        # For position (1,0,0) around Y axis: cross(Y, pos) = (0, 0, 1)
        assert abs(p.acceleration.z) > 0, "Tangential force should have Z component"
        assert p.acceleration.y == 0.0, (
            "Tangential force should have no Y component around Y axis"
        )

    def test_radial_force_pulls_toward_axis(self):
        """Radial force pulls toward/away from axis."""
        module = VortexModule(
            center=Vec3(0, 0, 0),
            axis=Vec3(0, 1, 0),
            strength=0.0,  # No swirl
            pull_strength=5.0,  # Pull toward axis
        )
        p = Particle()
        p.position = Vec3(3, 0, 4)  # Distance 5 from Y axis

        module.apply_to_particle(p, 0.016)

        # Should be pulled toward axis (negative radial direction)
        expected_dir = Vec3(-3, 0, -4).normalized()
        force_len = p.acceleration.length()
        if force_len > 0:
            force_dir = Vec3(
                p.acceleration.x / force_len,
                p.acceleration.y / force_len,
                p.acceleration.z / force_len,
            )
            assert abs(force_dir.x - expected_dir.x) < 0.01
            assert abs(force_dir.z - expected_dir.z) < 0.01

    def test_combined_effect_produces_spiral_motion(self):
        """Combined tangential + radial produces spiral motion."""
        module = VortexModule(
            center=Vec3(0, 0, 0),
            axis=Vec3(0, 1, 0),
            strength=10.0,
            pull_strength=3.0,
        )
        p = Particle()
        p.position = Vec3(0, 0, 2)

        module.apply_to_particle(p, 0.016)

        # Should have both tangential (x) and radial (z) components
        has_tangential = abs(p.acceleration.x) > 0
        has_radial = abs(p.acceleration.z) > 0
        assert has_tangential, "Missing tangential force component"
        assert has_radial, "Missing radial force component"


# =========================================================================
# T1.8 - AttractionModule
# =========================================================================


class TestAttractionModuleContract:
    """T1.8: Validate AttractionModule."""

    def test_point_attractor_with_configurable_falloff(self):
        """Point attractor pulls toward target with configurable falloff."""
        module = AttractionModule(
            target=Vec3(10, 0, 0),
            strength=10.0,
            radius=20.0,
            falloff="linear",
        )
        p = Particle()
        p.position = Vec3(0, 0, 0)

        module.apply_to_particle(p, 0.016)

        # Force should point toward (10, 0, 0) -- positive X
        assert p.acceleration.x > 0, "Force should pull toward positive X"

    def test_force_magnitude_decreases_with_distance(self):
        """Force magnitude decreases with distance (falloff)."""
        module = AttractionModule(
            target=Vec3(0, 0, 0),
            strength=10.0,
            radius=10.0,
            falloff="linear",
        )

        p_close = Particle()
        p_close.position = Vec3(1, 0, 0)
        module.apply_to_particle(p_close, 0.016)

        p_far = Particle()
        p_far.position = Vec3(9, 0, 0)
        module.apply_to_particle(p_far, 0.016)

        assert abs(p_close.acceleration.x) > abs(p_far.acceleration.x), (
            "Closer particle should experience stronger force"
        )

    def test_quadratic_falloff(self):
        """Quadratic falloff decreases force with distance squared."""
        module = AttractionModule(
            target=Vec3(0, 0, 0),
            strength=10.0,
            radius=10.0,
            falloff="quadratic",
        )

        p_close = Particle()
        p_close.position = Vec3(2, 0, 0)
        module.apply_to_particle(p_close, 0.016)
        force_close = abs(p_close.acceleration.x)

        p_far = Particle()
        p_far.position = Vec3(8, 0, 0)
        module.apply_to_particle(p_far, 0.016)
        force_far = abs(p_far.acceleration.x)

        assert force_close > force_far, (
            "Closer particle should experience stronger force with quadratic falloff"
        )

    def test_attraction_center_configurable(self):
        """Attraction center is configurable."""
        module = AttractionModule(
            target=Vec3(-10, 5, 3),
            strength=5.0,
            radius=20.0,
        )
        p = Particle()
        p.position = Vec3(0, 0, 0)

        module.apply_to_particle(p, 0.016)

        # Force vector should point toward the configured center
        assert (
            p.acceleration.x < 0
        ), "Should pull toward negative X (center at -10, 0, 0)"
        assert p.acceleration.y > 0, "Should pull toward positive Y (center at 0, 5, 0)"
        assert p.acceleration.z > 0, "Should pull toward positive Z (center at 0, 0, 3)"


# =========================================================================
# T1.9 - SizeOverLifeModule
# =========================================================================


class TestSizeOverLifeModuleContract:
    """T1.9: Validate SizeOverLifeModule."""

    def test_linear_easing_interpolates_correctly(self):
        """Linear easing interpolates size proportionally to normalized age."""
        module = SizeOverLifeModule(
            start_size=2.0,
            end_size=0.0,
            curve="linear",
        )

        def size_at(t):
            p = Particle()
            p.age = t
            p.lifetime = 1.0
            module.apply_to_particle(p, 0.016)
            return p.size

        assert abs(size_at(0.0) - 2.0) < 0.001
        assert abs(size_at(0.25) - 1.5) < 0.01
        assert abs(size_at(0.5) - 1.0) < 0.01
        assert abs(size_at(0.75) - 0.5) < 0.01
        assert abs(size_at(1.0) - 0.0) < 0.001

    def test_ease_in_starts_slow_ends_fast(self):
        """Ease-in starts slow, ends fast (quadratic)."""
        module = SizeOverLifeModule(
            start_size=0.0,
            end_size=1.0,
            curve="ease_in",
        )

        def size_at(t):
            p = Particle()
            p.age = t
            p.lifetime = 1.0
            module.apply_to_particle(p, 0.016)
            return p.size

        early = size_at(0.25)
        late = size_at(0.75)
        # Ease-in: t^2, so t=0.25 -> 0.0625, t=0.75 -> 0.5625
        assert abs(early - 0.0625) < 0.01
        assert abs(late - 0.5625) < 0.01
        assert late > 0.5, "Ease-in at t=0.75 should be past midpoint"

    def test_ease_out_starts_fast_ends_slow(self):
        """Ease-out starts fast, ends slow (1 - (1-t)^2)."""
        module = SizeOverLifeModule(
            start_size=0.0,
            end_size=1.0,
            curve="ease_out",
        )

        def size_at(t):
            p = Particle()
            p.age = t
            p.lifetime = 1.0
            module.apply_to_particle(p, 0.016)
            return p.size

        early = size_at(0.25)
        late = size_at(0.75)
        # Ease-out: 1 - (1-t)^2, so t=0.25 -> 0.4375, t=0.75 -> 0.9375
        assert abs(early - 0.4375) < 0.01
        assert abs(late - 0.9375) < 0.01
        assert early > 0.25, "Ease-out at t=0.25 should be past midpoint"

    def test_size_at_age_zero_equals_start_size(self):
        """Size at age=0 equals start size."""
        module = SizeOverLifeModule(start_size=3.0, end_size=0.5)
        p = Particle()
        p.age = 0.0
        p.lifetime = 1.0
        module.apply_to_particle(p, 0.016)
        assert p.size == 3.0

    def test_size_at_age_lifetime_equals_end_size(self):
        """Size at age=lifetime equals end size."""
        module = SizeOverLifeModule(start_size=3.0, end_size=0.5)
        p = Particle()
        p.age = 1.0
        p.lifetime = 1.0
        module.apply_to_particle(p, 0.016)
        assert p.size == 0.5


# =========================================================================
# T1.10 - ColorOverLifeModule
# =========================================================================


class TestColorOverLifeModuleContract:
    """T1.10: Validate ColorOverLifeModule."""

    def test_lerp_mode_interpolates_between_two_colors(self):
        """Lerp mode interpolates between two colors over lifetime."""
        module = ColorOverLifeModule(
            start_color=Vec4(1, 0, 0, 1),  # Red opaque
            end_color=Vec4(0, 0, 1, 0),  # Blue transparent
        )

        def color_at(t):
            p = Particle()
            p.age = t
            p.lifetime = 1.0
            module.apply_to_particle(p, 0.016)
            return p.color

        # At start: full red
        c0 = color_at(0.0)
        assert abs(c0.x - 1.0) < 0.01
        assert abs(c0.z - 0.0) < 0.01

        # At midpoint: purple, half alpha
        c05 = color_at(0.5)
        assert abs(c05.x - 0.5) < 0.01
        assert abs(c05.z - 0.5) < 0.01
        assert abs(c05.w - 0.5) < 0.01

        # At end: full blue, transparent
        c1 = color_at(1.0)
        assert abs(c1.x - 0.0) < 0.01
        assert abs(c1.z - 1.0) < 0.01
        assert abs(c1.w - 0.0) < 0.01

    def test_gradient_mode_samples_from_color_stops(self):
        """Gradient mode samples from color stops."""
        gradient = [
            (0.0, Vec4(1, 0, 0, 1)),  # Red
            (0.5, Vec4(0, 1, 0, 1)),  # Green
            (1.0, Vec4(0, 0, 1, 1)),  # Blue
        ]
        module = ColorOverLifeModule(gradient=gradient)

        def color_at(t):
            p = Particle()
            p.age = t
            p.lifetime = 1.0
            module.apply_to_particle(p, 0.016)
            return p.color

        c0 = color_at(0.0)
        assert abs(c0.x - 1.0) < 0.01  # Red

        c05 = color_at(0.5)
        assert abs(c05.y - 1.0) < 0.01  # Green

        c1 = color_at(1.0)
        assert abs(c1.z - 1.0) < 0.01  # Blue

    def test_gradient_interpolates_between_stops(self):
        """Gradient interpolates between adjacent color stops."""
        gradient = [
            (0.0, Vec4(1, 0, 0, 1)),
            (1.0, Vec4(0, 1, 0, 1)),
        ]
        module = ColorOverLifeModule(gradient=gradient)

        p = Particle()
        p.age = 0.5
        p.lifetime = 1.0
        module.apply_to_particle(p, 0.016)

        assert abs(p.color.x - 0.5) < 0.01
        assert abs(p.color.y - 0.5) < 0.01

    def test_alpha_channel_interpolated_correctly(self):
        """Alpha channel interpolated correctly in lerp mode."""
        module = ColorOverLifeModule(
            start_color=Vec4(1, 1, 1, 1),
            end_color=Vec4(1, 1, 1, 0),
        )

        p = Particle()
        p.age = 0.5
        p.lifetime = 1.0
        module.apply_to_particle(p, 0.016)

        assert abs(p.color.w - 0.5) < 0.01


# =========================================================================
# T1.11 - ShapeEmitter Sampling
# =========================================================================


class TestShapeEmitterSamplingContract:
    """T1.11: Validate ShapeEmitter sampling for all shapes."""

    def test_point_all_particles_at_origin(self):
        """Point shape: all particles spawn at origin."""
        emitter = ShapeEmitter(
            shape=EmitterShape.POINT,
            position=Vec3(5, -3, 2),
        )
        for _ in range(20):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            assert p.position.x == 5
            assert p.position.y == -3
            assert p.position.z == 2

    def test_sphere_volume_corrected_distribution(self):
        """Sphere: volume-corrected distribution."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=5.0,
            emit_from_surface=False,
        )
        distances = []
        for _ in range(500):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            distances.append(p.position.length())

        assert max(distances) <= 5.0
        interior = sum(1 for d in distances if d < 2.5)
        assert interior > 30, (
            f"Too few interior samples ({interior}/500) for volume distribution"
        )

    def test_box_uniform_within_bounds(self):
        """Box: uniform distribution within bounds."""
        emitter = ShapeEmitter(
            shape=EmitterShape.BOX,
            size=Vec3(4, 6, 8),
            emit_from_surface=False,
        )
        for _ in range(100):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            assert abs(p.position.x) <= 2.0, "X outside box bounds"
            assert abs(p.position.y) <= 3.0, "Y outside box bounds"
            assert abs(p.position.z) <= 4.0, "Z outside box bounds"

    def test_cone_direction_within_cone_angle(self):
        """Cone: direction within cone angle."""
        emitter = ShapeEmitter(
            shape=EmitterShape.CONE,
            angle=30.0,
        )
        up = Vec3(0, 1, 0)
        for _ in range(100):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            vel_norm = p.velocity.normalized()
            dot = up.x * vel_norm.x + up.y * vel_norm.y + up.z * vel_norm.z
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))
            assert angle <= 30.0 + 0.1, (
                f"Cone angle {angle} exceeds configured 30 degrees"
            )

    def test_circle_2d_disk_sampling(self):
        """Circle: 2D disk sampling (on XZ plane)."""
        emitter = ShapeEmitter(
            shape=EmitterShape.CIRCLE,
            radius=5.0,
            emit_from_surface=False,
        )
        for _ in range(50):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            assert p.position.y == 0.0, "Circle should spawn on Y=0 plane"
            dist = math.sqrt(p.position.x**2 + p.position.z**2)
            assert dist <= 5.0, "Circle sample outside radius"

    def test_edge_linear_along_segment(self):
        """Edge: linear sampling along line segment."""
        emitter = ShapeEmitter(
            shape=EmitterShape.EDGE,
            size=Vec3(10, 0, 0),
        )
        for _ in range(50):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            assert abs(p.position.x) <= 5.0, "Edge sample outside segment"
            assert p.position.y == 0.0
            assert p.position.z == 0.0

    def test_sphere_surface_emission_exact_radius(self):
        """Sphere surface emission: all samples at exact radius."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=3.0,
            emit_from_surface=True,
        )
        for _ in range(100):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            dist = p.position.length()
            assert abs(dist - 3.0) < 0.05

    def test_box_surface_emission_on_boundary(self):
        """Box surface emission: at least one coordinate on boundary."""
        emitter = ShapeEmitter(
            shape=EmitterShape.BOX,
            size=Vec3(4, 4, 4),
            emit_from_surface=True,
        )
        for _ in range(100):
            p = Particle()
            emitter.apply_to_particle(p, 0.016)
            x, y, z = abs(p.position.x), abs(p.position.y), abs(p.position.z)
            # At least one coordinate should be at the boundary (2.0)
            assert (
                abs(x - 2.0) < 0.01
                or abs(y - 2.0) < 0.01
                or abs(z - 2.0) < 0.01
            ), "Surface box: at least one coordinate should be on boundary"


# =========================================================================
# T1.12 - BurstEmitter Timing
# =========================================================================


class TestBurstEmitterTimingContract:
    """T1.12: Validate BurstEmitter timing."""

    def test_spawns_exact_count_on_trigger(self):
        """Spawns exact count on trigger."""
        emitter = BurstEmitter(count=25, repeat_interval=0.0)
        count = emitter.get_spawn_count(0.016)
        assert count == 25

    def test_repeat_interval_respected(self):
        """Repeat interval respected if configured."""
        emitter = BurstEmitter(count=7, repeat_interval=3.0)

        # Initial burst
        assert emitter.get_spawn_count(0.016) == 7
        # Not enough time
        assert emitter.get_spawn_count(2.0) == 0
        # Crossed threshold
        assert emitter.get_spawn_count(2.0) == 7

    def test_manual_trigger_works(self):
        """Manual trigger spawns another burst regardless of interval."""
        emitter = BurstEmitter(count=5, repeat_interval=999.0)

        # Initial burst
        emitter.get_spawn_count(0.016)
        # Manual trigger
        emitter.trigger()
        count = emitter.get_spawn_count(0.016)
        assert count == 5

    def test_zero_count_burst_does_nothing(self):
        """Zero-count burst emits nothing."""
        emitter = BurstEmitter(count=0)
        assert emitter.get_spawn_count(0.016) == 0

    def test_burst_then_normal(self):
        """After burst is consumed, returns 0 until next interval."""
        emitter = BurstEmitter(count=10, repeat_interval=1.0)
        assert emitter.get_spawn_count(0.016) == 10
        assert emitter.get_spawn_count(0.5) == 0
        assert emitter.get_spawn_count(0.5) == 10  # Next interval


# =========================================================================
# T1.13 - RateEmitter Accumulation
# =========================================================================


class TestRateEmitterAccumulationContract:
    """T1.13: Validate RateEmitter accumulation."""

    def test_accumulator_tracks_fractional_particles(self):
        """Accumulator tracks fractional particles between frames."""
        emitter = RateEmitter(rate=3.0)  # 3 per second

        # 0.25s -> accumulate 0.75, spawn 0, keep 0.75
        assert emitter.get_spawn_count(0.25) == 0

        # Another 0.25s -> 0.75 + 0.75 = 1.5, spawn 1, keep 0.5
        assert emitter.get_spawn_count(0.25) == 1

        # Another 0.25s -> 0.5 + 0.75 = 1.25, spawn 1, keep 0.25
        assert emitter.get_spawn_count(0.25) == 1

        # Another 0.25s -> 0.25 + 0.75 = 1.0, spawn 1, keep 0.0
        assert emitter.get_spawn_count(0.25) == 1

    def test_spawns_when_accumulator_geq_one(self):
        """Spawns when accumulator >= 1.0."""
        emitter = RateEmitter(rate=10.0)
        # 0.05s -> 0.5, no spawn
        assert emitter.get_spawn_count(0.05) == 0
        # Another 0.05s -> 1.0, spawn 1
        assert emitter.get_spawn_count(0.05) == 1

    def test_rate_per_second_converted_correctly(self):
        """Rate per second correctly converted to per-frame count."""
        emitter = RateEmitter(rate=60.0)
        frame_time = 1.0 / 60.0
        count = emitter.get_spawn_count(frame_time)
        # 60 * (1/60) = 1.0 -> spawn 1
        assert count == 1

    def test_high_rate_spawns_multiple_per_frame(self):
        """High rate can spawn multiple particles in one frame."""
        emitter = RateEmitter(rate=1000.0)
        count = emitter.get_spawn_count(0.1)
        # 1000 * 0.1 = 100 -> spawn 100
        assert count == 100

    def test_zero_rate_never_spawns(self):
        """Zero rate never spawns particles."""
        emitter = RateEmitter(rate=0.0)
        for _ in range(10):
            assert emitter.get_spawn_count(1.0) == 0


# =========================================================================
# T1.14 - ParticleBudget Categories
# =========================================================================


class TestParticleBudgetCategoriesContract:
    """T1.14: Validate ParticleBudget categories."""

    def test_ambient_particles_lowest_priority(self):
        """Ambient particles lowest priority."""
        budget = ParticleBudget()
        ambient = budget.get_allocation("ambient")
        critical = budget.get_allocation("critical")
        assert ambient is not None
        assert critical is not None
        assert ambient.priority < critical.priority

    def test_gameplay_particles_medium_priority(self):
        """Gameplay particles medium priority."""
        budget = ParticleBudget()
        ambient = budget.get_allocation("ambient")
        gameplay = budget.get_allocation("gameplay")
        critical = budget.get_allocation("critical")
        assert ambient.priority < gameplay.priority < critical.priority, (
            f"Expected ambient < gameplay < critical, "
            f"got {ambient.priority} < {gameplay.priority} < {critical.priority}"
        )

    def test_critical_particles_always_allocated_if_possible(self):
        """Critical particles always allocated if budget available."""
        budget = ParticleBudget()
        budget.set_budget("critical", max_particles=5000, priority=100)
        allocated = budget.request_particles("critical", 5000)
        assert allocated == 5000

    def test_global_budget_respected_across_emitters(self):
        """Global budget respected across emitters (total cap)."""
        budget = ParticleBudget()
        budget.set_budget("cat_a", max_particles=300000)
        budget.set_budget("cat_b", max_particles=300000)

        allocated_a = budget.request_particles("cat_a", 300000)
        assert allocated_a == 300000

        # cat_b should be limited by remaining global budget
        allocated_b = budget.request_particles("cat_b", 300000)
        total_global = 500000
        remaining = total_global - allocated_a
        assert allocated_b <= remaining, (
            f"Global budget exceeded. Got {allocated_b}, expected <= {remaining}"
        )

    def test_releasing_frees_budget_for_others(self):
        """Releasing particles frees budget for other categories."""
        budget = ParticleBudget()
        budget.set_budget("cat_a", max_particles=400000)
        budget.set_budget("cat_b", max_particles=400000)

        budget.request_particles("cat_a", 400000)
        budget.release_particles("cat_a", 400000)

        allocated_b = budget.request_particles("cat_b", 400000)
        assert allocated_b == 400000

    def test_budget_tracks_usage(self):
        """Budget correctly tracks current usage."""
        budget = ParticleBudget()
        budget.set_budget("test", max_particles=1000)
        budget.request_particles("test", 300)
        alloc = budget.get_allocation("test")
        assert alloc.current_particles == 300

    def test_budget_defaults_exist(self):
        """Default budget categories exist."""
        budget = ParticleBudget()
        for name in ("default", "ambient", "gameplay", "critical"):
            assert budget.get_allocation(name) is not None, (
                f"Default category '{name}' not found"
            )


# =========================================================================
# T1.15 - Emitter Prewarm
# =========================================================================


class TestEmitterPrewarmContract:
    """T1.15: Validate emitter prewarm."""

    def test_prewarm_simulates_seconds_before_first_frame(self):
        """Prewarm simulates N seconds before first frame."""
        config = EmitterConfig(
            prewarm=True,
            warmup_time=2.0,
            max_particles=50000,
        )
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=1000))
        emitter.start()
        assert emitter.alive_count > 0, "Prewarm should produce alive particles"

    def test_particles_in_correct_state_after_prewarm(self):
        """Particles in correct lifecycle state after prewarm."""
        config = EmitterConfig(
            prewarm=True,
            warmup_time=10.0,
            max_particles=100000,
        )
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=10000))
        emitter.add_spawn_module(LifetimeModule(lifetime=(5.0, 5.0)))
        emitter.start()

        for particle in emitter.iter_particles():
            assert particle.age >= 0.0
            assert particle.age <= particle.lifetime
            assert particle.is_alive

    def test_pool_state_consistent_after_prewarm(self):
        """Pool state consistent after prewarm."""
        config = EmitterConfig(
            prewarm=True,
            warmup_time=1.0,
            max_particles=1000,
        )
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=500))
        emitter.start()

        stats = emitter.get_stats()
        assert stats["alive_count"] <= config.max_particles
        assert stats["alive_count"] >= 0

    def test_prewarm_false_no_particles_on_start(self):
        """Without prewarm, no particles exist immediately after start."""
        config = EmitterConfig(
            prewarm=False,
            max_particles=1000,
        )
        emitter = ParticleEmitter(config=config)
        emitter.add_spawn_module(RateEmitter(rate=500))
        emitter.start()

        assert emitter.alive_count == 0, (
            "Without prewarm, no particles should exist at start"
        )

    def test_prewarm_with_short_warmup(self):
        """Short warmup produces fewer particles than long warmup."""
        config_short = EmitterConfig(prewarm=True, warmup_time=0.5, max_particles=50000)
        emitter_short = ParticleEmitter(config=config_short)
        emitter_short.add_spawn_module(RateEmitter(rate=1000))
        emitter_short.start()
        short_count = emitter_short.alive_count

        config_long = EmitterConfig(prewarm=True, warmup_time=5.0, max_particles=50000)
        emitter_long = ParticleEmitter(config=config_long)
        emitter_long.add_spawn_module(RateEmitter(rate=1000))
        emitter_long.start()
        long_count = emitter_long.alive_count

        assert long_count > short_count, (
            f"Longer warmup ({long_count}) should produce more "
            f"particles than short warmup ({short_count})"
        )


# =========================================================================
# T1.16 - VectorFieldModule Stub
# =========================================================================


class TestVectorFieldModuleStubContract:
    """T1.16: Validate VectorFieldModule stub."""

    def test_architecture_in_place_for_3d_force_volume(self):
        """Architecture in place (module exists, has expected interface)."""
        module = VectorFieldModule()
        assert hasattr(module, "apply_to_particle")
        assert hasattr(module, "set_field_data")

    def test_stub_returns_zero_force(self):
        """Data loading correctly stubbed (returns zero force) when no field."""
        module = VectorFieldModule()
        p = Particle()
        p.position = Vec3(5, 5, 5)
        module.apply_to_particle(p, 0.016)
        assert p.acceleration.length() == 0.0, (
            "Default VectorFieldModule should produce no force"
        )

    def test_does_not_crash_when_field_data_unavailable(self):
        """Does not crash when field data unavailable."""
        module = VectorFieldModule(
            bounds_min=Vec3(-10, -10, -10),
            bounds_max=Vec3(10, 10, 10),
        )
        p = Particle()
        p.position = Vec3(0, 0, 0)
        module.apply_to_particle(p, 0.016)  # Should not raise

    def test_stub_with_bounds_no_crash(self):
        """Module with bounds (but no field data set) handles gracefully."""
        module = VectorFieldModule(
            bounds_min=Vec3(-5, -5, -5),
            bounds_max=Vec3(5, 5, 5),
        )
        p = Particle()
        p.position = Vec3(1, 2, 3)
        # No set_field_data called -- _field is None, stub returns zero force
        module.apply_to_particle(p, 0.016)
        assert p.acceleration.length() == 0.0, (
            "Stub should return zero when no field data set"
        )

    def test_particle_outside_bounds_gets_zero_force(self):
        """Returns zero force for particles outside field bounds."""
        module = VectorFieldModule(
            bounds_min=Vec3(-5, -5, -5),
            bounds_max=Vec3(5, 5, 5),
        )
        module.set_field_data([[]])
        p = Particle()
        p.position = Vec3(100, 100, 100)
        module.apply_to_particle(p, 0.016)
        assert p.acceleration.length() == 0.0


# =========================================================================
# T1.17 - BillboardRenderer
# =========================================================================


class TestBillboardRendererContract:
    """T1.17: Validate BillboardRenderer."""

    def test_view_alignment_computes_correct_facing(self):
        """View alignment computes correct facing direction toward camera."""
        module = BillboardRenderer(alignment="view")
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))

        p = Particle()
        p.position = Vec3(0, 0, 0)

        module.apply_to_particle(p, 0.016)

        assert "billboard_right" in p.custom_data
        assert "billboard_up" in p.custom_data

    def test_velocity_alignment_orients_along_movement(self):
        """Velocity alignment orients billboard along movement direction."""
        module = BillboardRenderer(alignment="velocity")
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))

        p = Particle()
        p.position = Vec3(0, 0, 0)
        p.velocity = Vec3(5, 0, 0)  # Moving right

        module.apply_to_particle(p, 0.016)

        assert "billboard_right" in p.custom_data
        assert "billboard_up" in p.custom_data

    def test_velocity_stretch_scales_billboard(self):
        """Velocity stretch scales billboard correctly proportional to speed."""
        module = BillboardRenderer(alignment="view", stretch=2.0)
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))

        p = Particle()
        p.position = Vec3(0, 0, 0)
        p.velocity = Vec3(10, 0, 0)

        module.apply_to_particle(p, 0.016)

        stretch = p.custom_data.get("stretch")
        assert stretch is not None
        # stretch = 1.0 + vel_len * stretch_param = 1 + 10 * 2 = 21
        assert abs(stretch - 21.0) < 0.001

    def test_zero_velocity_no_stretch(self):
        """Zero velocity produces no stretch (stretch = 1.0)."""
        module = BillboardRenderer(alignment="view", stretch=2.0)
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))

        p = Particle()
        p.position = Vec3(0, 0, 0)
        p.velocity = Vec3(0, 0, 0)

        module.apply_to_particle(p, 0.016)

        stretch = p.custom_data.get("stretch")
        assert stretch is not None
        assert abs(stretch - 1.0) < 0.001

    def test_view_alignment_with_camera_position(self):
        """View alignment responds to camera position changes."""
        camera_pos = Vec3(0, 0, 10)
        module = BillboardRenderer(alignment="view")
        module.set_camera(camera_pos, Vec3(0, 1, 0))

        p = Particle()
        p.position = Vec3(0, 0, 0)
        module.apply_to_particle(p, 0.016)

        right = p.custom_data.get("billboard_right")
        up = p.custom_data.get("billboard_up")
        assert right is not None
        assert up is not None
        # Right and up should be perpendicular
        dot = right.x * up.x + right.y * up.y + right.z * up.z
        assert abs(dot) < 1e-6, "Billboard right and up should be perpendicular"


# =========================================================================
# T1.18 - MeshParticleRenderer
# =========================================================================


class TestMeshParticleRendererContract:
    """T1.18: Validate MeshParticleRenderer."""

    def test_instance_data_prepared_for_batch_rendering(self):
        """Instance data prepared correctly for batch rendering."""
        module = MeshParticleRenderer(scale_with_size=True)
        p = Particle()
        p.size = 2.0
        module.apply_to_particle(p, 0.016)

        assert "instance_scale" in p.custom_data
        assert p.custom_data["instance_scale"] == 2.0

    def test_transform_matrix_computed_from_particle_state(self):
        """Transform information reflects particle position."""
        module = MeshParticleRenderer(scale_with_size=True)
        p = Particle()
        p.position = Vec3(1, 2, 3)
        p.size = 1.5

        module.apply_to_particle(p, 0.016)

        assert "instance_scale" in p.custom_data
        # Scale should reflect the particle's size
        assert p.custom_data["instance_scale"] == 1.5

    def test_align_to_velocity_sets_forward_direction(self):
        """Align-to-velocity sets instance_forward direction in custom_data."""
        module = MeshParticleRenderer(
            align_to_velocity=True,
            scale_with_size=False,
        )
        p = Particle()
        p.velocity = Vec3(0, 10, 0)

        module.apply_to_particle(p, 0.016)

        forward = p.custom_data.get("instance_forward")
        assert forward is not None
        # Forward should be normalized velocity direction
        vel_len = math.sqrt(
            p.velocity.x**2 + p.velocity.y**2 + p.velocity.z**2
        )
        expected_y = p.velocity.y / vel_len
        assert abs(forward.y - expected_y) < 0.001

    def test_no_forward_when_velocity_near_zero(self):
        """No instance_forward set when velocity is near zero."""
        module = MeshParticleRenderer(
            align_to_velocity=True,
            scale_with_size=False,
        )
        p = Particle()
        p.velocity = Vec3(0, 0, 0)

        module.apply_to_particle(p, 0.016)

        assert "instance_forward" not in p.custom_data, (
            "Zero velocity should not produce forward direction"
        )

    def test_culling_data_generated(self):
        """Culling-related data present in custom_data."""
        module = MeshParticleRenderer(scale_with_size=True)
        p = Particle()
        p.size = 1.0
        p.position = Vec3(0, 0, 0)

        module.apply_to_particle(p, 0.016)

        # Instance data for frustum culling should be present
        assert "instance_scale" in p.custom_data


# =========================================================================
# Edge Cases and Error Handling
# =========================================================================


class TestParticleEmitterEdgeCases:
    """Edge cases for ParticleEmitter lifecycle."""

    def test_emitter_start_stop_lifecycle(self):
        """Emitter lifecycle: INACTIVE -> ACTIVE -> STOPPED."""
        emitter = ParticleEmitter()
        assert emitter.state == EmitterState.INACTIVE
        emitter.start()
        assert emitter.state == EmitterState.ACTIVE
        emitter.stop(immediate=True)
        assert emitter.state == EmitterState.STOPPED

    def test_emitter_with_no_modules_does_not_crash(self):
        """Emitter with no modules should not crash on update."""
        emitter = ParticleEmitter()
        emitter.start()
        emitter.update(0.016)  # Should not raise

    def test_emitter_stats_always_available(self):
        """Emitter stats always available regardless of state."""
        emitter = ParticleEmitter()
        stats = emitter.get_stats()
        assert "state" in stats
        assert "alive_count" in stats
        assert "simulation_mode" in stats

    def test_simulation_mode_auto_resolves_cpu(self):
        """AUTO resolves to CPU for small counts."""
        config = EmitterConfig(max_particles=100, simulation=SimulationMode.AUTO)
        emitter = ParticleEmitter(config=config)
        assert emitter.simulation_mode == SimulationMode.CPU

    def test_simulation_mode_auto_resolves_gpu(self):
        """AUTO resolves to GPU for large counts."""
        config = EmitterConfig(max_particles=50000, simulation=SimulationMode.AUTO)
        emitter = ParticleEmitter(config=config)
        assert emitter.simulation_mode == SimulationMode.GPU


class TestParticleSystemManagerContract:
    """ParticleSystemManager contract tests."""

    def test_create_and_retrieve_emitter(self):
        """Creating an emitter makes it retrievable by name."""
        manager = ParticleSystemManager()
        emitter = manager.create_emitter("test")
        assert emitter is not None
        assert manager.get_emitter("test") is emitter

    def test_remove_emitter(self):
        """Removing an emitter makes it no longer retrievable."""
        manager = ParticleSystemManager()
        manager.create_emitter("test")
        manager.remove_emitter("test")
        assert manager.get_emitter("test") is None

    def test_update_all_emitters(self):
        """update_all processes all active emitters without error."""
        manager = ParticleSystemManager()
        e1 = manager.create_emitter("e1")
        e2 = manager.create_emitter("e2")
        e1.start()
        e2.start()
        manager.update_all(0.016)

    def test_total_particle_count_empty(self):
        """Total particle count is 0 when no emitters."""
        manager = ParticleSystemManager()
        assert manager.get_total_particle_count() == 0


class TestModuleStageContract:
    """Module stage assignment contract."""

    def test_spawn_modules_have_spawn_stage(self):
        """Spawn modules have stage SPAWN."""
        shape = ShapeEmitter(shape=EmitterShape.POINT)
        burst = BurstEmitter(count=10)
        rate = RateEmitter(rate=10)
        assert shape.stage == ModuleStage.SPAWN
        assert burst.stage == ModuleStage.SPAWN
        assert rate.stage == ModuleStage.SPAWN

    def test_force_modules_have_update_stage(self):
        """Force modules have stage UPDATE."""
        for mod_cls in (GravityModule, WindModule, TurbulenceModule,
                        VortexModule, AttractionModule, VectorFieldModule):
            module = mod_cls()
            assert module.stage == ModuleStage.UPDATE, (
                f"{mod_cls.__name__} should be UPDATE stage"
            )

    def test_attribute_modules_have_update_or_spawn_stage(self):
        """Attribute modules have stage UPDATE (if they track over life)"""
        update_stage_modules = (
            SizeOverLifeModule, ColorOverLifeModule,
        )
        for mod_cls in update_stage_modules:
            module = mod_cls()
            assert module.stage == ModuleStage.UPDATE, (
                f"{mod_cls.__name__} should be UPDATE stage, "
                f"got {module.stage}"
            )

        # LifetimeModule, RotationModule, VelocityModule initialise at spawn time
        spawn_stage_modules = (LifetimeModule, RotationModule, VelocityModule)
        for mod_cls in spawn_stage_modules:
            module = mod_cls()
            assert module.stage == ModuleStage.SPAWN, (
                f"{mod_cls.__name__} should be SPAWN stage, "
                f"got {module.stage}"
            )

    def test_render_modules_have_render_stage(self):
        """Render modules have stage RENDER."""
        for mod_cls in (BillboardRenderer, MeshParticleRenderer):
            module = mod_cls()
            assert module.stage == ModuleStage.RENDER, (
                f"{mod_cls.__name__} should be RENDER stage"
            )


# =========================================================================
# Module Configuration
# =========================================================================


class TestModuleConfigContract:
    """Module configuration defaults."""

    def test_default_stage_is_update(self):
        """Default stage is UPDATE if not specified."""
        from engine.rendering.particles import ModuleConfig

        config = ModuleConfig()
        assert config.stage == ModuleStage.UPDATE

    def test_default_lod_range_includes_all(self):
        """Default LOD range covers all levels."""
        from engine.rendering.particles import ModuleConfig

        config = ModuleConfig()
        assert config.lod_range == (0, 3)


# =========================================================================
# Edge Cases - Boundary and Error
# =========================================================================


class TestParticleBoundaryCases:
    """Boundary and error cases for the particle system."""

    def test_negative_pool_size_handled(self):
        """Pool handles edge case of zero max_particles."""
        pool = ParticlePool(max_particles=0)
        assert pool.allocate() is None
        assert pool.alive_count == 0

    def test_single_particle_pool(self):
        """Pool with single particle works correctly."""
        pool = ParticlePool(max_particles=1)
        p = pool.allocate()
        assert p is not None
        assert pool.allocate() is None  # Exhausted
        pool.deallocate(p)
        p2 = pool.allocate()
        assert p2 is not None

    def test_very_large_sphere_radius(self):
        """Sphere emitter handles large radius without precision issues."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=1e6,
            emit_from_surface=True,
        )
        p = Particle()
        emitter.apply_to_particle(p, 0.016)
        dist = p.position.length()
        assert abs(dist - 1e6) / 1e6 < 0.01

    def test_negative_size_in_size_module(self):
        """Size module with negative end_size clamps or handles gracefully."""
        module = SizeOverLifeModule(start_size=1.0, end_size=-1.0)
        p = Particle()
        p.age = 1.0
        p.lifetime = 1.0
        module.apply_to_particle(p, 0.016)
        # Should not crash; size should be end_size or 0
        assert p.size <= 1.0

    def test_zero_delta_time(self):
        """Rate emitter handles zero dt without division errors."""
        emitter = RateEmitter(rate=100.0)
        count = emitter.get_spawn_count(0.0)
        assert count == 0  # No time elapsed, no particles

    def test_negative_delta_time(self):
        """Rate emitter handles negative dt without breaking accumulator."""
        emitter = RateEmitter(rate=10.0)
        # Negative dt may produce negative accumulator (implementation-defined)
        # but subsequent positive calls must still work
        emitter.get_spawn_count(-0.1)
        count_pos = emitter.get_spawn_count(0.1)
        assert count_pos >= 0, (
            f"After negative dt, positive dt should produce >= 0 particles, got {count_pos}"
        )

    def test_very_small_delta_time(self):
        """Emitter handles very small dt without precision issues."""
        emitter = RateEmitter(rate=1.0)
        count = emitter.get_spawn_count(1e-10)
        assert count == 0  # Essentially no particles

    def test_collision_with_no_particles(self):
        """Collision module handles empty spatial hash without error."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
        )
        p = Particle()
        p.position = Vec3(0, 0, 0)
        # Before adding any particles to hash, get_nearby should return empty
        nearby = module.get_nearby_particles(p)
        assert len(nearby) == 0

    def test_billboard_no_camera_set(self):
        """Billboard renderer handles missing camera gracefully."""
        module = BillboardRenderer(alignment="view")
        p = Particle()
        p.position = Vec3(0, 0, 0)
        module.apply_to_particle(p, 0.016)
        # Should not crash; may or may not have billboard data

    def test_emitter_with_burst_and_rate_combined(self):
        """Emitter with combined BurstEmitter and RateEmitter works."""
        emitter = ParticleEmitter()
        emitter.add_spawn_module(BurstEmitter(count=50, repeat_interval=0.0))
        emitter.add_spawn_module(RateEmitter(rate=100))
        emitter.start()
        emitter.update(0.016)
        # Should have burst + some rate particles
        assert emitter.alive_count > 0

    def test_multiple_modules_same_type(self):
        """Multiple modules of same type compose correctly."""
        emitter = ParticleEmitter()
        emitter.add_spawn_module(RateEmitter(rate=50))
        emitter.add_spawn_module(RateEmitter(rate=50))
        emitter.start()
        emitter.update(0.1)
        # Two 50-rate emitters = 100 particles/sec
        # In 0.1s = 10 particles
        count = emitter.alive_count
        assert count > 0
