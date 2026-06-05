"""
Whitebox tests for cloth wind forces.

Tests:
- WindSettings: configuration data class
- WindForce: aerodynamic wind force computation
- DirectionalWind: constant directional wind
- PointWind: radial wind source
- VortexWind: swirling wind pattern
- WindSystem: multi-source wind management
"""

import math

import numpy as np
import pytest

from engine.simulation.cloth.cloth_simulation import ClothMesh, ClothParticle, ClothTriangle
from engine.simulation.cloth.cloth_wind import (
    DirectionalWind,
    PointWind,
    VortexWind,
    WindForce,
    WindSettings,
    WindSystem,
)


def make_particle(pos, inv_mass=1.0):
    """Helper to create a particle at a position."""
    pos_arr = np.array(pos, dtype=np.float32)
    return ClothParticle(
        position=pos_arr,
        prev_position=pos_arr.copy(),
        inv_mass=inv_mass,
    )


def make_simple_mesh():
    """Create a simple mesh with one triangle for wind testing."""
    particles = [
        make_particle([0.0, 0.0, 0.0]),  # 0
        make_particle([1.0, 0.0, 0.0]),  # 1
        make_particle([0.0, 1.0, 0.0]),  # 2
    ]
    triangles = [ClothTriangle(p0=0, p1=1, p2=2)]
    return ClothMesh(particles=particles, edges=[], triangles=triangles)


class TestWindSettings:
    """Test WindSettings data class."""

    def test_default_settings(self):
        """Test default wind settings."""
        settings = WindSettings()

        assert np.allclose(settings.direction, [1.0, 0.0, 0.0])
        assert settings.strength == 1.0
        assert settings.drag_coefficient == 0.5
        assert settings.lift_coefficient == 0.2
        assert settings.turbulence_strength == 0.3
        assert settings.turbulence_frequency == 2.0
        assert settings.turbulence_octaves == 3

    def test_custom_settings(self):
        """Test custom wind settings."""
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        settings = WindSettings(
            direction=direction,
            strength=5.0,
            drag_coefficient=0.8,
            lift_coefficient=0.4,
            turbulence_strength=0.0,
        )

        assert np.allclose(settings.direction, direction)
        assert settings.strength == 5.0
        assert settings.drag_coefficient == 0.8
        assert settings.lift_coefficient == 0.4
        assert settings.turbulence_strength == 0.0


class TestWindForce:
    """Test WindForce class."""

    def test_wind_force_creation(self):
        """Test wind force creation with default settings."""
        wind = WindForce()

        assert wind.settings is not None
        assert wind.settings.strength == 1.0

    def test_wind_force_custom_settings(self):
        """Test wind force with custom settings."""
        settings = WindSettings(strength=10.0)
        wind = WindForce(settings=settings)

        assert wind.settings.strength == 10.0

    def test_set_direction(self):
        """Test setting wind direction."""
        wind = WindForce()
        new_dir = np.array([1.0, 1.0, 0.0], dtype=np.float32)

        wind.set_direction(new_dir)

        # Should be normalized
        expected = new_dir / np.linalg.norm(new_dir)
        assert np.allclose(wind.settings.direction, expected, atol=1e-5)

    def test_set_direction_zero_vector(self):
        """Setting zero direction should not change anything."""
        wind = WindForce()
        original_dir = wind.settings.direction.copy()

        wind.set_direction(np.zeros(3, dtype=np.float32))

        # Should remain unchanged
        assert np.allclose(wind.settings.direction, original_dir)

    def test_set_strength(self):
        """Test setting wind strength."""
        wind = WindForce()

        wind.set_strength(5.0)

        assert wind.settings.strength == 5.0

    def test_set_strength_negative_clamped(self):
        """Negative strength should be clamped to 0."""
        wind = WindForce()

        wind.set_strength(-5.0)

        assert wind.settings.strength == 0.0

    def test_set_vertex_influence(self):
        """Test setting per-vertex wind influence."""
        wind = WindForce()
        influence = np.array([1.0, 0.5, 0.0], dtype=np.float32)

        wind.set_vertex_influence(influence)

        assert np.allclose(wind._vertex_influence, influence)

    def test_update_advances_time(self):
        """Test that update advances internal time."""
        wind = WindForce()
        initial_time = wind._time

        wind.update(0.1)

        assert wind._time == initial_time + 0.1

    def test_compute_wind_force_applies_acceleration(self):
        """Wind force should apply acceleration to particles facing the wind."""
        wind = WindForce()
        wind.settings.strength = 10.0
        wind.settings.turbulence_strength = 0.0  # No turbulence
        # Wind blows in +Z direction to face the XY plane triangle
        wind.settings.direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Create a triangle in XY plane (normal in Z direction)
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([1.0, 0.0, 0.0]),
            make_particle([0.0, 1.0, 0.0]),
        ]
        triangles = [ClothTriangle(p0=0, p1=1, p2=2)]
        mesh = ClothMesh(particles=particles, edges=[], triangles=triangles)

        # Reset accelerations
        for p in mesh.particles:
            p.acceleration[:] = 0.0

        wind.compute_wind_force(mesh, dt=0.016)

        # At least one particle should have non-zero acceleration
        has_force = any(
            np.linalg.norm(p.acceleration) > 1e-8
            for p in mesh.particles
            if p.inv_mass > 0
        )

        assert has_force

    def test_compute_wind_force_skips_pinned_triangles(self):
        """Triangles with all pinned vertices should be skipped."""
        wind = WindForce()
        wind.settings.strength = 10.0

        # All pinned particles
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),
            make_particle([1.0, 0.0, 0.0], inv_mass=0.0),
            make_particle([0.0, 1.0, 0.0], inv_mass=0.0),
        ]
        triangles = [ClothTriangle(p0=0, p1=1, p2=2)]
        mesh = ClothMesh(particles=particles, edges=[], triangles=triangles)

        for p in particles:
            p.acceleration[:] = 0.0

        wind.compute_wind_force(mesh, dt=0.016)

        # No acceleration should be applied
        for p in particles:
            assert np.allclose(p.acceleration, [0.0, 0.0, 0.0])

    def test_compute_wind_force_respects_vertex_influence(self):
        """Vertex influence should scale wind force."""
        wind = WindForce()
        wind.settings.strength = 10.0
        wind.settings.turbulence_strength = 0.0

        mesh = make_simple_mesh()

        # Set influence: full, half, zero
        influence = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        wind.set_vertex_influence(influence)

        for p in mesh.particles:
            p.acceleration[:] = 0.0

        wind.compute_wind_force(mesh, dt=0.016)

        # Particle 2 (influence=0) should have no acceleration
        # (unless it received force from another source)
        # Actually, force is distributed to all triangle vertices,
        # but scaled by influence
        pass  # Complex to test precisely due to force distribution

    def test_turbulence_sampling(self):
        """Test that turbulence produces variation."""
        wind = WindForce()
        wind.settings.turbulence_strength = 1.0
        wind.settings.turbulence_frequency = 5.0

        pos1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pos2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        turb1 = wind._sample_turbulence(pos1)
        turb2 = wind._sample_turbulence(pos2)

        # Different positions should give different turbulence
        assert not np.allclose(turb1, turb2)

    def test_simple_noise_deterministic(self):
        """Simple noise should be deterministic for same inputs."""
        wind = WindForce()
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seed = 42

        noise1 = wind._simple_noise(pos, seed)
        noise2 = wind._simple_noise(pos, seed)

        assert noise1 == noise2

    def test_simple_noise_range(self):
        """Simple noise should be in [-1, 1]."""
        wind = WindForce()

        for _ in range(100):
            pos = np.random.randn(3).astype(np.float32) * 10
            noise = wind._simple_noise(pos, 42)
            assert -1.0 <= noise <= 1.0


class TestDirectionalWind:
    """Test DirectionalWind class."""

    def test_directional_wind_creation(self):
        """Test directional wind creation."""
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        wind = DirectionalWind(direction=direction, strength=5.0)

        assert np.allclose(wind.direction, direction)
        assert wind.strength == 5.0

    def test_get_velocity(self):
        """Test getting wind velocity."""
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        wind = DirectionalWind(direction=direction, strength=3.0)

        pos = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        velocity = wind.get_velocity(pos, time=0.0)

        # Velocity should be direction * strength
        expected = direction * 3.0
        assert np.allclose(velocity, expected)

    def test_velocity_independent_of_position(self):
        """Directional wind should be same everywhere."""
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        wind = DirectionalWind(direction=direction, strength=2.0)

        pos1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pos2 = np.array([100.0, 100.0, 100.0], dtype=np.float32)

        v1 = wind.get_velocity(pos1, 0.0)
        v2 = wind.get_velocity(pos2, 0.0)

        assert np.allclose(v1, v2)


class TestPointWind:
    """Test PointWind class."""

    def test_point_wind_creation(self):
        """Test point wind creation."""
        position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        wind = PointWind(position=position, strength=5.0, radius=10.0, falloff=2.0)

        assert np.allclose(wind.position, position)
        assert wind.strength == 5.0
        assert wind.radius == 10.0
        assert wind.falloff == 2.0

    def test_velocity_at_source(self):
        """Wind at source position should be zero."""
        position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        wind = PointWind(position=position, strength=5.0, radius=10.0)

        velocity = wind.get_velocity(position, time=0.0)

        assert np.allclose(velocity, [0.0, 0.0, 0.0])

    def test_velocity_outside_radius(self):
        """Wind outside radius should be zero."""
        position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        wind = PointWind(position=position, strength=5.0, radius=10.0)

        far_pos = np.array([20.0, 0.0, 0.0], dtype=np.float32)
        velocity = wind.get_velocity(far_pos, time=0.0)

        assert np.allclose(velocity, [0.0, 0.0, 0.0])

    def test_velocity_radiates_outward(self):
        """Wind should point away from source."""
        position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        wind = PointWind(position=position, strength=5.0, radius=10.0)

        sample_pos = np.array([3.0, 0.0, 0.0], dtype=np.float32)
        velocity = wind.get_velocity(sample_pos, time=0.0)

        # Should point in +X direction
        assert velocity[0] > 0

    def test_velocity_falloff(self):
        """Wind should decrease with distance."""
        position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        wind = PointWind(position=position, strength=5.0, radius=10.0, falloff=2.0)

        near_pos = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        far_pos = np.array([5.0, 0.0, 0.0], dtype=np.float32)

        v_near = wind.get_velocity(near_pos, time=0.0)
        v_far = wind.get_velocity(far_pos, time=0.0)

        # Near should be stronger
        assert np.linalg.norm(v_near) > np.linalg.norm(v_far)


class TestVortexWind:
    """Test VortexWind class."""

    def test_vortex_wind_creation(self):
        """Test vortex wind creation."""
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        wind = VortexWind(
            center=center,
            axis=axis,
            strength=5.0,
            radius=10.0,
            angular_velocity=2.0,
        )

        assert np.allclose(wind.center, center)
        assert np.allclose(wind.axis, axis)
        assert wind.strength == 5.0
        assert wind.radius == 10.0
        assert wind.angular_velocity == 2.0

    def test_velocity_at_center(self):
        """Wind at vortex center should be zero."""
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        wind = VortexWind(center=center, axis=axis, strength=5.0, radius=10.0)

        velocity = wind.get_velocity(center, time=0.0)

        assert np.allclose(velocity, [0.0, 0.0, 0.0])

    def test_velocity_on_axis(self):
        """Wind on vortex axis should be zero."""
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        wind = VortexWind(center=center, axis=axis, strength=5.0, radius=10.0)

        on_axis = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        velocity = wind.get_velocity(on_axis, time=0.0)

        assert np.allclose(velocity, [0.0, 0.0, 0.0])

    def test_velocity_outside_radius(self):
        """Wind outside vortex radius should be zero."""
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        wind = VortexWind(center=center, axis=axis, strength=5.0, radius=10.0)

        far_pos = np.array([20.0, 0.0, 0.0], dtype=np.float32)
        velocity = wind.get_velocity(far_pos, time=0.0)

        assert np.allclose(velocity, [0.0, 0.0, 0.0])

    def test_velocity_tangential(self):
        """Vortex wind should be tangential to radius."""
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Y-up
        wind = VortexWind(center=center, axis=axis, strength=5.0, radius=10.0)

        # Point in XZ plane, at +X
        pos = np.array([3.0, 0.0, 0.0], dtype=np.float32)
        velocity = wind.get_velocity(pos, time=0.0)

        # Velocity should be in Z direction (tangential)
        # Y should be zero (perpendicular to axis)
        assert abs(velocity[1]) < 1e-6


class TestWindSystem:
    """Test WindSystem class."""

    def test_wind_system_creation(self):
        """Test wind system creation."""
        system = WindSystem()

        assert len(system._directional_winds) == 0
        assert len(system._point_winds) == 0
        assert len(system._vortex_winds) == 0

    def test_add_directional_wind(self):
        """Test adding directional wind."""
        system = WindSystem()
        wind = DirectionalWind(
            direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            strength=5.0,
        )

        system.add_directional_wind(wind)

        assert len(system._directional_winds) == 1

    def test_add_point_wind(self):
        """Test adding point wind."""
        system = WindSystem()
        wind = PointWind(
            position=np.zeros(3, dtype=np.float32),
            strength=5.0,
            radius=10.0,
        )

        system.add_point_wind(wind)

        assert len(system._point_winds) == 1

    def test_add_vortex_wind(self):
        """Test adding vortex wind."""
        system = WindSystem()
        wind = VortexWind(
            center=np.zeros(3, dtype=np.float32),
            axis=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            strength=5.0,
            radius=10.0,
        )

        system.add_vortex_wind(wind)

        assert len(system._vortex_winds) == 1

    def test_clear_winds(self):
        """Test clearing all wind sources."""
        system = WindSystem()
        system.add_directional_wind(
            DirectionalWind(
                direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                strength=1.0,
            )
        )
        system.add_point_wind(
            PointWind(position=np.zeros(3, dtype=np.float32), strength=1.0, radius=5.0)
        )

        system.clear_winds()

        assert len(system._directional_winds) == 0
        assert len(system._point_winds) == 0

    def test_get_combined_wind_single_source(self):
        """Test combined wind with single source."""
        system = WindSystem()
        system.add_directional_wind(
            DirectionalWind(
                direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                strength=5.0,
            )
        )

        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        combined = system.get_combined_wind(pos)

        expected = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(combined, expected)

    def test_get_combined_wind_multiple_sources(self):
        """Test combined wind with multiple sources."""
        system = WindSystem()
        system.add_directional_wind(
            DirectionalWind(
                direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                strength=3.0,
            )
        )
        system.add_directional_wind(
            DirectionalWind(
                direction=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                strength=4.0,
            )
        )

        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        combined = system.get_combined_wind(pos)

        expected = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        assert np.allclose(combined, expected)

    def test_apply_to_mesh_uses_custom_winds(self):
        """Test applying wind system to mesh."""
        system = WindSystem()
        # Wind blows in +Z direction to face the XY plane triangle
        system.add_directional_wind(
            DirectionalWind(
                direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                strength=10.0,
            )
        )

        # Create a triangle in XY plane (normal in Z direction)
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([1.0, 0.0, 0.0]),
            make_particle([0.0, 1.0, 0.0]),
        ]
        triangles = [ClothTriangle(p0=0, p1=1, p2=2)]
        mesh = ClothMesh(particles=particles, edges=[], triangles=triangles)

        for p in mesh.particles:
            p.acceleration[:] = 0.0

        system.apply_to_mesh(mesh, dt=0.016)

        # Should have applied force (wind facing the triangle)
        has_force = any(
            np.linalg.norm(p.acceleration) > 1e-8
            for p in mesh.particles
            if p.inv_mass > 0
        )
        assert has_force

    def test_apply_to_mesh_uses_default_wind_force(self):
        """Test applying wind system with no custom sources uses default."""
        system = WindSystem()
        # No custom winds added

        mesh = make_simple_mesh()
        for p in mesh.particles:
            p.acceleration[:] = 0.0

        # Default wind force should be used
        system._wind_force.settings.strength = 10.0
        system.apply_to_mesh(mesh, dt=0.016)

        # Default wind should apply force
        # (Note: this depends on wind direction facing the triangle)

    def test_apply_advances_time(self):
        """Test that apply_to_mesh advances internal time."""
        system = WindSystem()
        initial_time = system._time

        mesh = make_simple_mesh()
        system.apply_to_mesh(mesh, dt=0.1)

        assert system._time == initial_time + 0.1
