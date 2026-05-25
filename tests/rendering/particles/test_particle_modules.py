"""
Tests for particle behavior modules.

Tests:
    - Spawn modules (ShapeEmitter, BurstEmitter, RateEmitter)
    - Force modules (Gravity, Wind, Turbulence, Vortex, Attraction)
    - Attribute modules (SizeOverLife, ColorOverLife, Rotation)
    - Render modules (Billboard, MeshParticle)
"""

import math
import pytest

from engine.rendering.particles.particle_system import (
    Particle,
    ParticleState,
    Vec3,
    Vec4,
)
from engine.rendering.particles.particle_modules import (
    # Enums
    ModuleStage,
    EmitterShape,
    CollisionMode,
    BlendMode,
    # Config
    ModuleConfig,
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
)


class TestModuleConfig:
    """Test ModuleConfig creation."""

    def test_default_config(self):
        """Test default configuration."""
        config = ModuleConfig()
        assert config.stage == ModuleStage.UPDATE
        assert config.lod_range == (0, 3)

    def test_from_decorator_params(self):
        """Test creation from decorator parameters."""
        config = ModuleConfig.from_decorator_params(
            stage="spawn",
            lod_range=(0, 2),
        )
        assert config.stage == ModuleStage.SPAWN
        assert config.lod_range == (0, 2)


class TestShapeEmitter:
    """Test ShapeEmitter spawn module."""

    def test_point_emitter(self):
        """Test point shape emission."""
        emitter = ShapeEmitter(
            shape=EmitterShape.POINT,
            position=Vec3(1, 2, 3),
        )
        particle = Particle()

        emitter.apply_to_particle(particle, 0.016)

        assert particle.position.x == 1
        assert particle.position.y == 2
        assert particle.position.z == 3

    def test_sphere_emitter(self):
        """Test sphere shape emission."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            position=Vec3(0, 0, 0),
            radius=1.0,
            emit_from_surface=True,
        )
        particle = Particle()

        emitter.apply_to_particle(particle, 0.016)

        # Position should be on sphere surface
        distance = particle.position.length()
        assert abs(distance - 1.0) < 0.01

    def test_box_emitter(self):
        """Test box shape emission."""
        emitter = ShapeEmitter(
            shape=EmitterShape.BOX,
            position=Vec3(0, 0, 0),
            size=Vec3(2, 2, 2),
            emit_from_surface=False,
        )
        particle = Particle()

        emitter.apply_to_particle(particle, 0.016)

        # Position should be within box bounds
        assert abs(particle.position.x) <= 1.0
        assert abs(particle.position.y) <= 1.0
        assert abs(particle.position.z) <= 1.0

    def test_cone_emitter(self):
        """Test cone shape emission."""
        emitter = ShapeEmitter(
            shape=EmitterShape.CONE,
            angle=45.0,
        )
        particle = Particle()

        emitter.apply_to_particle(particle, 0.016)

        # Velocity should be within cone angle from up
        up = Vec3(0, 1, 0)
        vel_normalized = particle.velocity.normalized()
        dot = up.dot(vel_normalized)
        angle = math.acos(max(-1, min(1, dot)))
        assert angle <= math.radians(45)


class TestBurstEmitter:
    """Test BurstEmitter spawn module."""

    def test_single_burst(self):
        """Test single burst emission."""
        emitter = BurstEmitter(count=10, repeat_interval=0.0)

        # First call should return burst count
        count = emitter.get_spawn_count(0.016)
        assert count == 10

        # Second call should return 0
        count = emitter.get_spawn_count(0.016)
        assert count == 0

    def test_repeating_burst(self):
        """Test repeating burst emission."""
        emitter = BurstEmitter(count=5, repeat_interval=1.0)

        # First burst
        count = emitter.get_spawn_count(0.5)
        assert count == 5

        # Not enough time passed
        count = emitter.get_spawn_count(0.4)
        assert count == 0

        # Enough time passed
        count = emitter.get_spawn_count(0.7)
        assert count == 5

    def test_manual_trigger(self):
        """Test manual burst trigger."""
        emitter = BurstEmitter(count=10)

        # Exhaust initial burst
        emitter.get_spawn_count(0.016)

        # Manual trigger
        emitter.trigger()
        count = emitter.get_spawn_count(0.016)
        assert count == 10


class TestRateEmitter:
    """Test RateEmitter spawn module."""

    def test_rate_emission(self):
        """Test rate-based emission."""
        emitter = RateEmitter(rate=100.0)  # 100 per second

        # 0.1 seconds should spawn ~10 particles
        count = emitter.get_spawn_count(0.1)
        assert count == 10

    def test_accumulation(self):
        """Test fractional accumulation."""
        emitter = RateEmitter(rate=10.0)  # 10 per second

        # 0.05 seconds should accumulate 0.5, spawn 0
        count = emitter.get_spawn_count(0.05)
        assert count == 0

        # Another 0.05 seconds should complete to 1
        count = emitter.get_spawn_count(0.05)
        assert count == 1


class TestGravityModule:
    """Test GravityModule force module."""

    def test_default_gravity(self):
        """Test default gravity application."""
        module = GravityModule()
        particle = Particle()

        module.apply_to_particle(particle, 0.016)

        assert particle.acceleration.y < 0  # Negative Y (downward)

    def test_custom_gravity(self):
        """Test custom gravity vector."""
        module = GravityModule(gravity=Vec3(0, -20, 0))
        particle = Particle()

        module.apply_to_particle(particle, 0.016)

        assert particle.acceleration.y == -20


class TestWindModule:
    """Test WindModule force module."""

    def test_basic_wind(self):
        """Test basic wind force."""
        module = WindModule(
            direction=Vec3(1, 0, 0),
            strength=5.0,
            turbulence=0.0,
        )
        particle = Particle()

        module.apply_to_particle(particle, 0.016)

        assert particle.acceleration.x > 0

    def test_wind_with_turbulence(self):
        """Test wind with turbulence."""
        module = WindModule(
            direction=Vec3(1, 0, 0),
            strength=5.0,
            turbulence=1.0,
        )
        particle = Particle()

        # Multiple applications should show variance
        results = []
        for _ in range(10):
            p = Particle()
            module.apply_to_particle(p, 0.016)
            results.append(p.acceleration)

        # Should have some variance due to turbulence
        # (test is probabilistic but should pass almost always)
        x_values = [r.x for r in results]
        assert max(x_values) != min(x_values)


class TestVortexModule:
    """Test VortexModule force module."""

    def test_vortex_force(self):
        """Test vortex swirl force."""
        module = VortexModule(
            center=Vec3(0, 0, 0),
            axis=Vec3(0, 1, 0),
            strength=10.0,
        )
        particle = Particle()
        particle.position = Vec3(1, 0, 0)  # On X axis

        module.apply_to_particle(particle, 0.016)

        # Force should be tangential (in Z direction for this position)
        assert abs(particle.acceleration.z) > 0


class TestAttractionModule:
    """Test AttractionModule force module."""

    def test_attraction_force(self):
        """Test attraction toward point."""
        module = AttractionModule(
            target=Vec3(0, 0, 0),
            strength=10.0,
            radius=10.0,
        )
        particle = Particle()
        particle.position = Vec3(5, 0, 0)

        module.apply_to_particle(particle, 0.016)

        # Force should point toward origin (negative X)
        assert particle.acceleration.x < 0

    def test_attraction_falloff(self):
        """Test attraction falloff with distance."""
        module = AttractionModule(
            target=Vec3(0, 0, 0),
            strength=10.0,
            radius=10.0,
            falloff="linear",
        )

        # Close particle
        p_close = Particle()
        p_close.position = Vec3(2, 0, 0)
        module.apply_to_particle(p_close, 0.016)

        # Far particle
        p_far = Particle()
        p_far.position = Vec3(8, 0, 0)
        module.apply_to_particle(p_far, 0.016)

        # Close should have stronger force
        assert abs(p_close.acceleration.x) > abs(p_far.acceleration.x)


class TestCollisionModule:
    """Test CollisionModule."""

    def test_ground_plane_collision(self):
        """Test collision with ground plane."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            bounce=0.5,
        )
        module.set_ground_plane(height=0.0)

        particle = Particle()
        particle.position = Vec3(0, -0.5, 0)  # Below ground
        particle.velocity = Vec3(0, -10, 0)  # Moving down

        module.apply_to_particle(particle, 0.016)

        # Particle should be pushed above ground
        assert particle.position.y >= 0

        # Velocity should be reflected upward
        assert particle.velocity.y > 0

    def test_kill_on_collision(self):
        """Test killing particle on collision."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            kill_on_collision=True,
        )
        module.set_ground_plane(height=0.0)

        particle = Particle()
        particle.position = Vec3(0, -0.5, 0)
        particle.state = ParticleState.ALIVE

        module.apply_to_particle(particle, 0.016)

        assert particle.state == ParticleState.DYING

    def test_spatial_hash(self):
        """Test spatial hash for efficient collision detection."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=1.0,
        )

        # Create particles
        p1 = Particle()
        p1.position = Vec3(0.5, 0.5, 0.5)

        p2 = Particle()
        p2.position = Vec3(0.6, 0.6, 0.6)  # Same cell as p1

        p3 = Particle()
        p3.position = Vec3(10.0, 10.0, 10.0)  # Different cell

        # Add to spatial hash
        module.add_to_spatial_hash(p1)
        module.add_to_spatial_hash(p2)
        module.add_to_spatial_hash(p3)

        # Query nearby particles for p1
        nearby = module.get_nearby_particles(p1)

        # Should find p1 and p2 (same cell or adjacent)
        assert p1 in nearby
        assert p2 in nearby
        assert p3 not in nearby

    def test_spatial_hash_clear(self):
        """Test clearing spatial hash between frames."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
        )

        p1 = Particle()
        p1.position = Vec3(0, 0, 0)
        module.add_to_spatial_hash(p1)

        # Clear hash
        module.clear_spatial_hash()

        # Should be empty now
        nearby = module.get_nearby_particles(p1)
        assert len(nearby) == 0


class TestSizeOverLifeModule:
    """Test SizeOverLifeModule attribute module."""

    def test_size_interpolation(self):
        """Test size interpolation over lifetime."""
        module = SizeOverLifeModule(
            start_size=2.0,
            end_size=0.0,
        )

        # At birth
        p1 = Particle()
        p1.age = 0.0
        p1.lifetime = 1.0
        module.apply_to_particle(p1, 0.016)
        assert p1.size == 2.0

        # At halfway
        p2 = Particle()
        p2.age = 0.5
        p2.lifetime = 1.0
        module.apply_to_particle(p2, 0.016)
        assert abs(p2.size - 1.0) < 0.01

        # At end
        p3 = Particle()
        p3.age = 1.0
        p3.lifetime = 1.0
        module.apply_to_particle(p3, 0.016)
        assert p3.size == 0.0


class TestColorOverLifeModule:
    """Test ColorOverLifeModule attribute module."""

    def test_color_interpolation(self):
        """Test color interpolation over lifetime."""
        module = ColorOverLifeModule(
            start_color=Vec4(1, 0, 0, 1),  # Red
            end_color=Vec4(0, 0, 1, 0),  # Blue (transparent)
        )

        # At halfway
        particle = Particle()
        particle.age = 0.5
        particle.lifetime = 1.0
        module.apply_to_particle(particle, 0.016)

        # Should be purple-ish
        assert abs(particle.color.x - 0.5) < 0.01  # Red decreased
        assert abs(particle.color.z - 0.5) < 0.01  # Blue increased
        assert abs(particle.color.w - 0.5) < 0.01  # Alpha decreased


class TestRotationModule:
    """Test RotationModule attribute module."""

    def test_rotation_initialization(self):
        """Test rotation initialization at spawn."""
        module = RotationModule(
            initial_rotation=(0, 0),  # Deterministic for test
            angular_velocity=(90, 90),  # Deterministic
        )
        particle = Particle()

        module.apply_to_particle(particle, 0.016)

        # Rotation should be set
        assert particle.rotation == 0.0
        assert particle.angular_velocity == math.radians(90)


class TestLifetimeModule:
    """Test LifetimeModule attribute module."""

    def test_lifetime_initialization(self):
        """Test lifetime initialization at spawn."""
        module = LifetimeModule(lifetime=(2.0, 2.0))  # Deterministic
        particle = Particle()

        module.apply_to_particle(particle, 0.016)

        assert particle.lifetime == 2.0


class TestVelocityModule:
    """Test VelocityModule attribute module."""

    def test_velocity_initialization(self):
        """Test velocity initialization at spawn."""
        module = VelocityModule(
            velocity=Vec3(0, 10, 0),
            velocity_spread=Vec3(0, 0, 0),  # No spread for deterministic test
        )
        particle = Particle()

        module.apply_to_particle(particle, 0.016)

        assert particle.velocity.y == 10


class TestBillboardRenderer:
    """Test BillboardRenderer render module."""

    def test_billboard_orientation(self):
        """Test billboard orientation calculation."""
        module = BillboardRenderer(alignment="view")
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))

        particle = Particle()
        particle.position = Vec3(0, 0, 0)

        module.apply_to_particle(particle, 0.016)

        # Should have billboard vectors in custom_data
        assert "billboard_right" in particle.custom_data
        assert "billboard_up" in particle.custom_data


class TestMeshParticleRenderer:
    """Test MeshParticleRenderer render module."""

    def test_mesh_instance_data(self):
        """Test mesh instance data preparation."""
        module = MeshParticleRenderer(
            scale_with_size=True,
        )

        particle = Particle()
        particle.size = 2.0

        module.apply_to_particle(particle, 0.016)

        assert particle.custom_data["instance_scale"] == 2.0


class TestLODFiltering:
    """Test LOD-based module filtering."""

    def test_lod_active_in_range(self):
        """Test module is active within LOD range."""
        module = GravityModule(lod_range=(0, 2))

        assert module.is_active_for_lod(0)
        assert module.is_active_for_lod(1)
        assert module.is_active_for_lod(2)

    def test_lod_inactive_out_of_range(self):
        """Test module is inactive outside LOD range."""
        module = GravityModule(lod_range=(0, 2))

        assert not module.is_active_for_lod(3)
        assert not module.is_active_for_lod(10)

    def test_disabled_module(self):
        """Test disabled module is inactive."""
        module = GravityModule(lod_range=(0, 10))
        module.enabled = False

        assert not module.is_active_for_lod(0)


# =============================================================================
# T1.2  - Sphere Volume Distribution
# =============================================================================


class TestSphereVolumeDistribution:
    """T1.2: Validate sphere sampling volume distribution."""

    def test_surface_emission_uses_radius_directly(self):
        """Surface emission uses radius directly (no cube-root)."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=5.0,
            emit_from_surface=True,
        )
        for _ in range(50):
            particle = Particle()
            emitter.apply_to_particle(particle, 0.016)
            dist = particle.position.length()
            # All samples should be within 1% of radius on surface
            assert abs(dist - 5.0) < 0.05, f"Surface distance {dist} not near radius"

    def test_volume_emission_uses_cube_root_correction(self):
        """Cube-root correction (random ** (1/3)) applied for volume sampling."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=10.0,
            emit_from_surface=False,
        )
        distances = []
        for _ in range(200):
            particle = Particle()
            emitter.apply_to_particle(particle, 0.016)
            distances.append(particle.position.length())

        # Volume samples should be spread across the full radius, not clustered at surface
        assert max(distances) <= 10.0  # Never exceed radius
        # Some should be well inside the sphere
        has_interior = any(d < 5.0 for d in distances)
        assert has_interior, "No interior samples found -- cube-root correction may be missing"

    def test_direction_vector_normalized(self):
        """Direction vector should be normalized for sphere emission."""
        emitter = ShapeEmitter(
            shape=EmitterShape.SPHERE,
            radius=1.0,
            emit_from_surface=True,
        )
        particle = Particle()
        emitter.apply_to_particle(particle, 0.016)
        vel_len = particle.velocity.length()
        assert abs(vel_len - 1.0) < 0.001, f"Direction not normalized: length={vel_len}"


# =============================================================================
# T1.3  - Spatial Hash & Collision
# =============================================================================


class TestSpatialHashCollision:
    """T1.3: Validate spatial hash and collision detection."""

    def test_cell_coordinates_via_floor_division(self):
        """Cell coordinates computed correctly via floor division."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=2.0,
        )

        # Test with various positions
        p = Particle()
        p.position = Vec3(3.5, -1.2, 7.9)
        cell = module._get_cell(p.position)
        assert cell == (1, -1, 3), f"Expected (1, -1, 3) got {cell}"

        # Negative coordinates must use floor, not truncation
        p.position = Vec3(-0.5, -0.5, -0.5)
        cell = module._get_cell(p.position)
        assert cell == (-1, -1, -1), f"Expected (-1, -1, -1) got {cell}"

    def test_3x3x3_neighborhood(self):
        """3x3x3 neighborhood query returns all adjacent cells."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            use_spatial_hash=True,
            cell_size=1.0,
        )

        # Place particles in a cluster
        center = Particle()
        center.position = Vec3(0, 0, 0)
        module.add_to_spatial_hash(center)

        # Place a particle at (0.9, 0.9, 0.9) -- same cell
        same_cell = Particle()
        same_cell.position = Vec3(0.9, 0.9, 0.9)
        module.add_to_spatial_hash(same_cell)

        # Place a particle at (1.1, 0, 0) -- adjacent cell
        adjacent = Particle()
        adjacent.position = Vec3(1.1, 0, 0)
        module.add_to_spatial_hash(adjacent)

        nearby = module.get_nearby_particles(center)
        assert center in nearby
        assert same_cell in nearby
        assert adjacent in nearby

    def test_ground_plane_bounce_coefficient(self):
        """Ground plane collision applies bounce coefficient."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            bounce=0.75,
            friction=0.0,
        )
        module.set_ground_plane(height=0.0)

        particle = Particle()
        particle.position = Vec3(0, -1, 0)
        particle.velocity = Vec3(0, -10, 0)

        module.apply_to_particle(particle, 0.016)

        # Velocity should reflect upward with bounce coefficient
        expected_y = 10.0 * 0.75  # -vn * bounce = -(-10) * 0.75 = 7.5
        assert abs(particle.velocity.y - expected_y) < 0.001

    def test_ground_plane_friction(self):
        """Ground plane collision applies friction to tangential component."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            bounce=0.5,
            friction=0.3,
        )
        module.set_ground_plane(height=0.0)

        particle = Particle()
        particle.position = Vec3(0, -1, 0)
        particle.velocity = Vec3(5, -10, 0)  # Horizontal + vertical

        module.apply_to_particle(particle, 0.016)

        # Horizontal velocity should be reduced by friction
        expected_x = 5.0 * (1.0 - 0.3)  # vt * (1 - friction) = 5 * 0.7 = 3.5
        assert abs(particle.velocity.x - expected_x) < 0.001
        # Vertical should bounce
        assert particle.velocity.y > 0

    def test_ground_plane_kill_on_collision(self):
        """T1.3: kill_on_collision marks particle DYING."""
        module = CollisionModule(
            mode=CollisionMode.PRIMITIVE,
            kill_on_collision=True,
        )
        module.set_ground_plane(height=0.0)

        particle = Particle()
        particle.position = Vec3(0, -0.5, 0)
        particle.state = ParticleState.ALIVE

        module.apply_to_particle(particle, 0.016)

        assert particle.state == ParticleState.DYING


# =============================================================================
# T1.4  - Gravity Module
# =============================================================================


class TestGravityModuleAccumulation:
    """T1.4: Validate gravity acceleration accumulation."""

    def test_acceleration_accumulated_not_replaced(self):
        """Acceleration accumulated (not replaced) when gravity applied twice."""
        module = GravityModule(gravity=Vec3(0, -9.81, 0))
        particle = Particle()
        # Simulate initial acceleration from another module
        particle.acceleration = Vec3(10, 0, 0)

        # Apply gravity
        module.apply_to_particle(particle, 0.016)

        # Acceleration should be sum of existing + gravity
        assert particle.acceleration.x == 10.0
        assert particle.acceleration.y == -9.81
        assert particle.acceleration.z == 0.0

    def test_custom_gravity(self):
        """T1.4: Custom gravity vector."""
        module = GravityModule(gravity=Vec3(0, -20, 0))
        particle = Particle()

        module.apply_to_particle(particle, 0.016)

        assert particle.acceleration.y == -20


# =============================================================================
# T1.5  - Wind Module
# =============================================================================


class TestWindModuleDetails:
    """T1.5: Validate wind module forces."""

    def test_magnitude_applied_as_force(self):
        """Direction + magnitude applied as force."""
        module = WindModule(
            direction=Vec3(1, 0, 0),
            strength=15.0,
            turbulence=0.0,
        )
        particle = Particle()
        module.apply_to_particle(particle, 0.016)

        # Acceleration should be exactly strength in direction
        assert abs(particle.acceleration.x - 15.0) < 0.001

    def test_combined_with_other_forces(self):
        """Combined with other forces correctly."""
        wind = WindModule(
            direction=Vec3(1, 0, 0),
            strength=10.0,
            turbulence=0.0,
        )
        gravity = GravityModule(gravity=Vec3(0, -9.81, 0))
        particle = Particle()

        wind.apply_to_particle(particle, 0.016)
        gravity.apply_to_particle(particle, 0.016)

        assert particle.acceleration.x == 10.0
        assert particle.acceleration.y == -9.81


# =============================================================================
# T1.6  - Turbulence Module
# =============================================================================


class TestTurbulenceModule:
    """T1.6: Validate turbulence noise force."""

    def test_pseudo_noise_from_position(self):
        """Pseudo-noise computed from position and age."""
        module = TurbulenceModule(strength=5.0, frequency=1.0)
        p1 = Particle()
        p1.position = Vec3(1, 2, 3)
        p1.age = 0.5

        p2 = Particle()
        p2.position = Vec3(4, 5, 6)
        p2.age = 0.5

        module.apply_to_particle(p1, 0.016)
        module.apply_to_particle(p2, 0.016)

        # Different positions should produce different forces
        forces_match = (
            p1.acceleration.x == p2.acceleration.x
            and p1.acceleration.y == p2.acceleration.y
            and p1.acceleration.z == p2.acceleration.z
        )
        assert not forces_match, "Different positions should produce different turbulence"

    def test_force_varies_spatially_but_deterministically(self):
        """Force varies spatially but deterministically (same input = same output)."""
        module = TurbulenceModule(strength=5.0, frequency=1.0)

        # Apply to same state twice
        p1 = Particle()
        p1.position = Vec3(2.5, -1.0, 3.7)
        p1.age = 1.5
        module.apply_to_particle(p1, 0.016)

        p2 = Particle()
        p2.position = Vec3(2.5, -1.0, 3.7)
        p2.age = 1.5
        module.apply_to_particle(p2, 0.016)

        # Same position + age produces same force deterministically (sin/cos are deterministic)
        assert p1.acceleration.x == p2.acceleration.x
        assert p1.acceleration.y == p2.acceleration.y
        assert p1.acceleration.z == p2.acceleration.z

    def test_strength_scales_effect(self):
        """Strength parameter scales effect correctly."""
        module_weak = TurbulenceModule(strength=1.0, frequency=1.0)
        module_strong = TurbulenceModule(strength=10.0, frequency=1.0)

        p_weak = Particle()
        p_weak.position = Vec3(3, 4, 5)
        p_weak.age = 0.7
        module_weak.apply_to_particle(p_weak, 0.016)

        p_strong = Particle()
        p_strong.position = Vec3(3, 4, 5)
        p_strong.age = 0.7
        module_strong.apply_to_particle(p_strong, 0.016)

        # Stronger module should produce proportionally larger force
        assert abs(p_strong.acceleration.x) >= abs(p_weak.acceleration.x)
        assert abs(p_strong.acceleration.y) >= abs(p_weak.acceleration.y)
        assert abs(p_strong.acceleration.z) >= abs(p_weak.acceleration.z)


# =============================================================================
# T1.7  - Vortex Module
# =============================================================================


class TestVortexModuleExtended:
    """T1.7: Validate vortex module forces."""

    def test_radial_pull_toward_axis(self):
        """Radial force pulls toward/away from axis."""
        module = VortexModule(
            center=Vec3(0, 0, 0),
            axis=Vec3(0, 1, 0),
            strength=0.0,  # No swirl
            pull_strength=5.0,  # Pull toward axis
        )
        particle = Particle()
        particle.position = Vec3(3, 0, 4)  # Distance 5 from axis

        module.apply_to_particle(particle, 0.016)

        # Should be pulled toward axis (negative radial direction)
        expected_dir = Vec3(-3, 0, -4).normalized()
        force_dir = particle.acceleration.normalized()
        assert abs(force_dir.x - expected_dir.x) < 0.01
        assert abs(force_dir.z - expected_dir.z) < 0.01

    def test_combined_swirl_and_pull_spiral(self):
        """Combined effect produces spiral motion (both tangential + radial)."""
        module = VortexModule(
            center=Vec3(0, 0, 0),
            axis=Vec3(0, 1, 0),
            strength=10.0,
            pull_strength=3.0,
        )
        particle = Particle()
        particle.position = Vec3(0, 0, 2)

        module.apply_to_particle(particle, 0.016)

        # Should have both tangential (x) and radial (z) components
        assert abs(particle.acceleration.x) > 0  # Tangential
        assert particle.acceleration.z < 0  # Radial toward axis (negative z)


# =============================================================================
# T1.8  - Attraction Module
# =============================================================================


class TestAttractionModuleExtended:
    """T1.8: Validate attraction module falloff and configuration."""

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

        p_far = Particle()
        p_far.position = Vec3(8, 0, 0)
        module.apply_to_particle(p_far, 0.016)

        # Close should have stronger force
        assert abs(p_close.acceleration.x) > abs(p_far.acceleration.x)

    def test_configurable_center(self):
        """Attraction center configurable."""
        module = AttractionModule(
            target=Vec3(10, 0, 0),
            strength=5.0,
            radius=20.0,
        )
        particle = Particle()
        particle.position = Vec3(0, 0, 0)

        module.apply_to_particle(particle, 0.016)

        # Force should point toward (10, 0, 0) -- positive X
        assert particle.acceleration.x > 0


# =============================================================================
# T1.9  - Size Over Life
# =============================================================================


class TestSizeOverLifeExtended:
    """T1.9: Validate size easing curves."""

    def test_ease_in_starts_slow_ends_fast(self):
        """Ease-in starts slow, ends fast."""
        module = SizeOverLifeModule(
            start_size=0.0,
            end_size=1.0,
            curve="ease_in",
        )

        p_early = Particle()
        p_early.age = 0.25
        p_early.lifetime = 1.0
        module.apply_to_particle(p_early, 0.016)

        p_late = Particle()
        p_late.age = 0.75
        p_late.lifetime = 1.0
        module.apply_to_particle(p_late, 0.016)

        # Ease-in: t^2, so at t=0.25 size=0.0625, at t=0.75 size=0.5625
        assert abs(p_early.size - 0.0625) < 0.01
        assert abs(p_late.size - 0.5625) < 0.01
        # Late should have proportionally MORE than linear
        assert p_late.size > 0.5

    def test_ease_out_starts_fast_ends_slow(self):
        """Ease-out starts fast, ends slow."""
        module = SizeOverLifeModule(
            start_size=0.0,
            end_size=1.0,
            curve="ease_out",
        )

        p_early = Particle()
        p_early.age = 0.25
        p_early.lifetime = 1.0
        module.apply_to_particle(p_early, 0.016)

        p_late = Particle()
        p_late.age = 0.75
        p_late.lifetime = 1.0
        module.apply_to_particle(p_late, 0.016)

        # Ease-out: 1 - (1-t)^2, so at t=0.25 size=0.4375, at t=0.75 size=0.9375
        assert abs(p_early.size - 0.4375) < 0.01
        assert abs(p_late.size - 0.9375) < 0.01
        # Early should have proportionally MORE than linear
        assert p_early.size > 0.25

    def test_size_at_age_zero_equals_start(self):
        """Size at age=0 equals start size."""
        module = SizeOverLifeModule(start_size=3.0, end_size=0.5)
        particle = Particle()
        particle.age = 0.0
        particle.lifetime = 1.0
        module.apply_to_particle(particle, 0.016)
        assert particle.size == 3.0

    def test_size_at_lifetime_equals_end(self):
        """Size at age=lifetime equals end size."""
        module = SizeOverLifeModule(start_size=3.0, end_size=0.5)
        particle = Particle()
        particle.age = 1.0
        particle.lifetime = 1.0
        module.apply_to_particle(particle, 0.016)
        assert particle.size == 0.5


# =============================================================================
# T1.10 - Color Over Life
# =============================================================================


class TestColorOverLifeExtended:
    """T1.10: Validate color interpolation modes."""

    def test_gradient_mode_samples_from_stops(self):
        """Gradient mode samples from color stops."""
        gradient = [
            (0.0, Vec4(1, 0, 0, 1)),  # Red at 0
            (0.5, Vec4(0, 1, 0, 1)),  # Green at 0.5
            (1.0, Vec4(0, 0, 1, 1)),  # Blue at 1.0
        ]
        module = ColorOverLifeModule(gradient=gradient)

        # At t=0 (red)
        p0 = Particle()
        p0.age = 0.0
        p0.lifetime = 1.0
        module.apply_to_particle(p0, 0.016)
        assert abs(p0.color.x - 1.0) < 0.01
        assert abs(p0.color.y - 0.0) < 0.01

        # At t=0.5 (green)
        p05 = Particle()
        p05.age = 0.5
        p05.lifetime = 1.0
        module.apply_to_particle(p05, 0.016)
        assert abs(p05.color.y - 1.0) < 0.01

        # At t=0.25 (red->green midpoint = (0.5, 0.5, 0))
        p025 = Particle()
        p025.age = 0.25
        p025.lifetime = 1.0
        module.apply_to_particle(p025, 0.016)
        assert abs(p025.color.x - 0.5) < 0.01
        assert abs(p025.color.y - 0.5) < 0.01

    def test_alpha_interpolated_separately(self):
        """Alpha channel interpolated correctly."""
        module = ColorOverLifeModule(
            start_color=Vec4(1, 1, 1, 1),
            end_color=Vec4(1, 1, 1, 0),
        )
        particle = Particle()
        particle.age = 0.5
        particle.lifetime = 1.0
        module.apply_to_particle(particle, 0.016)
        assert abs(particle.color.w - 0.5) < 0.01


# =============================================================================
# T1.11 - ShapeEmitter Edge and Circle
# =============================================================================


class TestShapeEmitterEdgeAndCircle:
    """T1.11: Validate edge and circle sampling."""

    def test_circle_sampling(self):
        """Circle shape creates 2D disk positions."""
        emitter = ShapeEmitter(
            shape=EmitterShape.CIRCLE,
            radius=5.0,
            emit_from_surface=False,
        )
        for _ in range(50):
            particle = Particle()
            emitter.apply_to_particle(particle, 0.016)
            # Y should be 0 for circle
            assert particle.position.y == 0.0
            # Within radius
            dist = math.sqrt(
                particle.position.x**2 + particle.position.z**2
            )
            assert dist <= 5.0 + 0.001

    def test_circle_surface_sampling(self):
        """Circle surface emission samples exactly at radius."""
        emitter = ShapeEmitter(
            shape=EmitterShape.CIRCLE,
            radius=5.0,
            emit_from_surface=True,
        )
        for _ in range(50):
            particle = Particle()
            emitter.apply_to_particle(particle, 0.016)
            dist = math.sqrt(
                particle.position.x**2 + particle.position.z**2
            )
            assert abs(dist - 5.0) < 0.01

    def test_edge_sampling(self):
        """Edge shape samples along line segment."""
        emitter = ShapeEmitter(
            shape=EmitterShape.EDGE,
            size=Vec3(10, 0, 0),
        )
        for _ in range(50):
            particle = Particle()
            emitter.apply_to_particle(particle, 0.016)
            # Position should be along X axis within [-5, 5]
            assert abs(particle.position.x) <= 5.0 + 0.001
            assert particle.position.y == 0.0
            assert particle.position.z == 0.0


# =============================================================================
# T1.12 - BurstEmitter Prewarm
# =============================================================================


class TestBurstEmitterExtended:
    """T1.12: Validate burst emitter timing with prewarm."""

    def test_repeat_interval_respected(self):
        """Repeat interval respected if configured."""
        emitter = BurstEmitter(count=10, repeat_interval=2.0)

        # First call: initial burst
        assert emitter.get_spawn_count(0.016) == 10

        # Advance 1s (not enough)
        assert emitter.get_spawn_count(1.0) == 0

        # Advance another 1.5s (crossed 2s threshold)
        assert emitter.get_spawn_count(1.5) == 10


# =============================================================================
# T1.13 - RateEmitter Accumulation
# =============================================================================


class TestRateEmitterExtended:
    """T1.13: Validate rate emitter fractional accumulation."""

    def test_accumulator_tracks_fractional_particles(self):
        """Accumulator tracks fractional particles correctly."""
        emitter = RateEmitter(rate=3.0)  # 3 per second

        # 0.3s should accumulate 0.9 -> spawn 0, keep 0.9
        assert emitter.get_spawn_count(0.3) == 0

        # Another 0.3s -> 0.9 + 0.9 = 1.8 -> spawn 1, keep 0.8
        assert emitter.get_spawn_count(0.3) == 1

        # Another 0.3s -> 0.8 + 0.9 = 1.7 -> spawn 1, keep 0.7
        assert emitter.get_spawn_count(0.3) == 1

    def test_rate_per_second_converted_to_per_frame(self):
        """Rate per second correctly converted to per-frame counts."""
        emitter = RateEmitter(rate=1000.0)  # 1000 particles/second

        # 1/60 second = ~16.67 particles
        count = emitter.get_spawn_count(1.0 / 60.0)
        assert count == 16  # int(1000/60) = 16


# =============================================================================
# T1.16 - VectorFieldModule Stub
# =============================================================================


class TestVectorFieldModule:
    """T1.16: Validate vector field module stub."""

    def test_stub_returns_zero_force_when_no_data(self):
        """Data loading correctly stubbed (returns zero force) when field not set."""
        module = VectorFieldModule()
        particle = Particle()
        particle.position = Vec3(5, 5, 5)

        module.apply_to_particle(particle, 0.016)

        # No crash, no force
        assert particle.acceleration.length() == 0.0

    def test_does_not_crash_when_field_unavailable(self):
        """Does not crash when field data unavailable."""
        module = VectorFieldModule(
            bounds_min=Vec3(-10, -10, -10),
            bounds_max=Vec3(10, 10, 10),
        )
        particle = Particle()
        particle.position = Vec3(0, 0, 0)

        # Should not raise
        module.apply_to_particle(particle, 0.016)

    def test_returns_zero_for_particle_outside_bounds(self):
        """Returns zero force for particles outside field bounds."""
        module = VectorFieldModule(
            bounds_min=Vec3(-5, -5, -5),
            bounds_max=Vec3(5, 5, 5),
        )
        module.set_field_data([[[]]])

        outside = Particle()
        outside.position = Vec3(100, 100, 100)
        module.apply_to_particle(outside, 0.016)
        assert outside.acceleration.length() == 0.0


# =============================================================================
# T1.17 - Billboard Renderer
# =============================================================================


class TestBillboardRendererExtended:
    """T1.17: Validate billboard renderer alignment and stretch."""

    def test_velocity_alignment_orients_movement(self):
        """Velocity alignment orients along movement direction."""
        module = BillboardRenderer(alignment="velocity")
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))

        particle = Particle()
        particle.position = Vec3(0, 0, 0)
        particle.velocity = Vec3(5, 0, 0)  # Moving right

        module.apply_to_particle(particle, 0.016)

        # Should have billboard vectors related to velocity direction
        assert "billboard_right" in particle.custom_data
        assert "billboard_up" in particle.custom_data

    def test_velocity_stretch_scales_billboard(self):
        """Velocity stretch scales billboard correctly."""
        module = BillboardRenderer(alignment="view", stretch=2.0)
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))

        particle = Particle()
        particle.position = Vec3(0, 0, 0)
        particle.velocity = Vec3(10, 0, 0)  # Fast movement

        module.apply_to_particle(particle, 0.016)

        # Stretch value should reflect velocity
        stretch = particle.custom_data.get("stretch")
        assert stretch is not None
        # stretch = 1.0 + vel_len * stretch_param = 1 + 10 * 2 = 21
        assert abs(stretch - 21.0) < 0.001

    def test_stretch_zero_when_stationary(self):
        """Zero velocity produces no stretch."""
        module = BillboardRenderer(alignment="view", stretch=2.0)
        module.set_camera(Vec3(0, 0, 10), Vec3(0, 1, 0))

        particle = Particle()
        particle.position = Vec3(0, 0, 0)
        particle.velocity = Vec3(0, 0, 0)

        module.apply_to_particle(particle, 0.016)

        stretch = particle.custom_data.get("stretch")
        assert stretch is not None
        assert abs(stretch - 1.0) < 0.001


# =============================================================================
# T1.18 - MeshParticleRenderer
# =============================================================================


class TestMeshParticleRendererExtended:
    """T1.18: Validate mesh particle renderer data preparation."""

    def test_instance_data_prepared(self):
        """Instance data prepared correctly for batch rendering."""
        module = MeshParticleRenderer(scale_with_size=True)
        particle = Particle()
        particle.size = 2.0
        particle.velocity = Vec3(1, 2, 3)

        module.apply_to_particle(particle, 0.016)

        # scale should be set
        assert particle.custom_data.get("instance_scale") == 2.0

    def test_align_to_velocity_sets_forward(self):
        """Align-to-velocity sets forward direction in custom_data."""
        module = MeshParticleRenderer(
            align_to_velocity=True,
            scale_with_size=False,
        )
        particle = Particle()
        particle.velocity = Vec3(0, 10, 0)  # Moving up

        module.apply_to_particle(particle, 0.016)

        forward = particle.custom_data.get("instance_forward")
        assert forward is not None
        assert abs(forward.y - 1.0) < 0.001

    def test_no_forward_when_velocity_near_zero(self):
        """No forward set when velocity near zero with align_to_velocity."""
        module = MeshParticleRenderer(
            align_to_velocity=True,
            scale_with_size=False,
        )
        particle = Particle()
        particle.velocity = Vec3(0, 0, 0)

        module.apply_to_particle(particle, 0.016)

        assert "instance_forward" not in particle.custom_data
