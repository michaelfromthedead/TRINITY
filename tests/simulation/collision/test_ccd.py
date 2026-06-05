"""
Whitebox tests for engine.simulation.collision.ccd module.

Tests Continuous Collision Detection algorithms:
- Linear sweep tests
- Time of impact calculations
- Conservative advancement
- Speculative contacts
- CCDManager
"""

import pytest
import math
from engine.simulation.collision.ccd import (
    CCDMode,
    CCDResult,
    MotionState,
    linear_sweep_sphere,
    linear_sweep_capsule,
    linear_sweep_box,
    linear_sweep_test,
    time_of_impact,
    time_of_impact_sphere_sphere,
    conservative_advancement,
    speculative_contacts,
    CCDManager,
)
from engine.simulation.collision.narrowphase import Sphere, Capsule, Box, ConvexHull
from engine.simulation.collision.broadphase import Vec3, AABB
from engine.simulation.collision.config import CCD_THRESHOLD_VELOCITY


class TestCCDMode:
    """Tests for CCDMode enum."""

    def test_all_modes_exist(self):
        """All CCD modes should exist."""
        assert hasattr(CCDMode, "NONE")
        assert hasattr(CCDMode, "SWEPT")
        assert hasattr(CCDMode, "SPECULATIVE")


class TestCCDResult:
    """Tests for CCDResult dataclass."""

    def test_default_no_hit(self):
        """Default CCDResult should be no hit."""
        result = CCDResult()
        assert not result.hit
        assert result.toi == 1.0

    def test_bool_conversion(self):
        """CCDResult bool should return hit state."""
        assert not CCDResult()
        assert CCDResult(hit=True)


class TestMotionState:
    """Tests for MotionState dataclass."""

    def test_default_construction(self):
        """Default MotionState should be at rest."""
        motion = MotionState()
        assert motion.position.x == 0
        assert motion.velocity.x == 0
        assert motion.speed() == 0

    def test_position_at(self):
        """position_at should interpolate correctly."""
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(10, 0, 0),
        )
        pos = motion.position_at(0.5)
        assert pos.x == 5.0

    def test_speed(self):
        """speed should compute velocity magnitude."""
        motion = MotionState(velocity=Vec3(3, 4, 0))
        assert motion.speed() == 5.0


class TestLinearSweepSphere:
    """Tests for linear_sweep_sphere function."""

    def test_fast_sphere_hits_static_sphere(self):
        """Fast-moving sphere should detect collision with static sphere."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(20, 0, 0),  # Fast enough for CCD
        )
        target = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        result = linear_sweep_sphere(sphere, motion, target, dt=1.0)
        assert result.hit
        assert 0 < result.toi < 1.0

    def test_slow_sphere_uses_discrete(self):
        """Slow-moving sphere should use discrete test."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(1, 0, 0),  # Too slow for CCD
        )
        target = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        result = linear_sweep_sphere(sphere, motion, target, dt=1.0)
        assert not result.hit  # No collision at start position

    def test_sphere_already_colliding(self):
        """Sphere already colliding should return toi=0."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(20, 0, 0),
        )
        target = Sphere(center=Vec3(1.5, 0, 0), radius=1.0)  # Overlapping
        result = linear_sweep_sphere(sphere, motion, target, dt=1.0)
        assert result.hit
        assert result.toi == 0.0

    def test_sphere_misses_target(self):
        """Sphere moving away from target should not hit."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(-20, 0, 0),  # Moving away
        )
        target = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        result = linear_sweep_sphere(sphere, motion, target, dt=1.0)
        assert not result.hit

    def test_sphere_passes_through_thin_wall(self):
        """Sphere should detect collision even when passing through thin object."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=0.5)
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(100, 0, 0),  # Very fast
        )
        # Thin target
        target = Box(center=Vec3(50, 0, 0), half_extents=Vec3(0.1, 2, 2))
        result = linear_sweep_sphere(sphere, motion, target, dt=1.0)
        assert result.hit


class TestLinearSweepCapsule:
    """Tests for linear_sweep_capsule function."""

    def test_fast_capsule_hits_target(self):
        """Fast-moving capsule should detect collision."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.5)
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(20, 0, 0),
        )
        target = Sphere(center=Vec3(10, 1, 0), radius=1.0)
        result = linear_sweep_capsule(capsule, motion, target, dt=1.0)
        assert result.hit
        assert 0 < result.toi < 1.0

    def test_slow_capsule_uses_discrete(self):
        """Slow-moving capsule should use discrete test."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.5)
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(1, 0, 0),
        )
        target = Sphere(center=Vec3(10, 1, 0), radius=1.0)
        result = linear_sweep_capsule(capsule, motion, target, dt=1.0)
        assert not result.hit


class TestLinearSweepBox:
    """Tests for linear_sweep_box function."""

    def test_fast_box_hits_target(self):
        """Fast-moving box should detect collision."""
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(0.5, 0.5, 0.5))
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(20, 0, 0),
        )
        target = Box(center=Vec3(10, 0, 0), half_extents=Vec3(1, 1, 1))
        result = linear_sweep_box(box, motion, target, dt=1.0)
        assert result.hit
        assert 0 < result.toi < 1.0

    def test_slow_box_uses_discrete(self):
        """Slow-moving box should use discrete test."""
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(0.5, 0.5, 0.5))
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(1, 0, 0),
        )
        target = Box(center=Vec3(10, 0, 0), half_extents=Vec3(1, 1, 1))
        result = linear_sweep_box(box, motion, target, dt=1.0)
        assert not result.hit


class TestLinearSweepTest:
    """Tests for generic linear_sweep_test function."""

    def test_sphere_dispatch(self):
        """linear_sweep_test should dispatch to sphere sweep."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion = MotionState(velocity=Vec3(20, 0, 0))
        target = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        result = linear_sweep_test(sphere, motion, target, dt=1.0)
        assert result.hit

    def test_capsule_dispatch(self):
        """linear_sweep_test should dispatch to capsule sweep."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.5)
        motion = MotionState(velocity=Vec3(20, 0, 0))
        target = Sphere(center=Vec3(10, 1, 0), radius=1.0)
        result = linear_sweep_test(capsule, motion, target, dt=1.0)
        assert result.hit

    def test_box_dispatch(self):
        """linear_sweep_test should dispatch to box sweep."""
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(0.5, 0.5, 0.5))
        motion = MotionState(velocity=Vec3(20, 0, 0))
        target = Box(center=Vec3(10, 0, 0), half_extents=Vec3(1, 1, 1))
        result = linear_sweep_test(box, motion, target, dt=1.0)
        assert result.hit

    def test_convex_hull_dispatch(self):
        """linear_sweep_test should handle convex hulls."""
        hull = ConvexHull(vertices=[
            Vec3(-0.5, -0.5, -0.5), Vec3(0.5, -0.5, -0.5),
            Vec3(0.5, 0.5, -0.5), Vec3(-0.5, 0.5, -0.5),
            Vec3(-0.5, -0.5, 0.5), Vec3(0.5, -0.5, 0.5),
            Vec3(0.5, 0.5, 0.5), Vec3(-0.5, 0.5, 0.5),
        ])
        motion = MotionState(velocity=Vec3(20, 0, 0))
        target = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        result = linear_sweep_test(hull, motion, target, dt=1.0)
        assert result.hit


class TestTimeOfImpact:
    """Tests for time_of_impact function."""

    def test_two_moving_spheres_collide(self):
        """Two moving spheres should detect collision."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(20, 0, 0))
        sphere_b = Sphere(center=Vec3(20, 0, 0), radius=1.0)
        motion_b = MotionState(velocity=Vec3(-20, 0, 0))  # Moving toward each other
        result = time_of_impact(sphere_a, motion_a, sphere_b, motion_b, dt=1.0)
        assert result.hit
        assert 0 < result.toi < 1.0

    def test_two_moving_spheres_same_direction(self):
        """Two spheres moving same direction at same speed should not collide."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(20, 0, 0))
        sphere_b = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        motion_b = MotionState(velocity=Vec3(20, 0, 0))  # Same velocity
        result = time_of_impact(sphere_a, motion_a, sphere_b, motion_b, dt=1.0)
        # No relative motion, check if already colliding
        assert not result.hit or result.toi == 0.0

    def test_slow_relative_motion_uses_discrete(self):
        """Slow relative motion should use discrete test."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(20, 0, 0))
        sphere_b = Sphere(center=Vec3(100, 0, 0), radius=1.0)
        motion_b = MotionState(velocity=Vec3(19, 0, 0))  # Only 1 unit relative
        result = time_of_impact(sphere_a, motion_a, sphere_b, motion_b, dt=1.0)
        assert not result.hit


class TestTimeOfImpactSphereSphere:
    """Tests for analytical sphere-sphere TOI."""

    def test_spheres_collide_analytically(self):
        """Analytical sphere-sphere should find exact TOI."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        sphere_b = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        velocity_a = Vec3(20, 0, 0)
        velocity_b = Vec3(-20, 0, 0)
        result = time_of_impact_sphere_sphere(
            sphere_a, velocity_a, sphere_b, velocity_b, dt=1.0
        )
        assert result.hit
        # TOI should be around 0.2 (4 units to close / 40 units relative speed)
        assert 0.1 < result.toi < 0.3

    def test_spheres_already_overlapping(self):
        """Overlapping spheres should return toi=0."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        sphere_b = Sphere(center=Vec3(1.5, 0, 0), radius=1.0)
        velocity_a = Vec3(10, 0, 0)
        velocity_b = Vec3(0, 0, 0)
        result = time_of_impact_sphere_sphere(
            sphere_a, velocity_a, sphere_b, velocity_b, dt=1.0
        )
        assert result.hit
        assert result.toi == 0.0

    def test_spheres_moving_apart(self):
        """Spheres moving apart should not collide."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        sphere_b = Sphere(center=Vec3(5, 0, 0), radius=1.0)
        velocity_a = Vec3(-10, 0, 0)  # Moving away
        velocity_b = Vec3(10, 0, 0)   # Moving away
        result = time_of_impact_sphere_sphere(
            sphere_a, velocity_a, sphere_b, velocity_b, dt=1.0
        )
        assert not result.hit

    def test_spheres_no_collision(self):
        """Spheres with perpendicular velocities may not collide."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        sphere_b = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        velocity_a = Vec3(0, 10, 0)  # Moving up
        velocity_b = Vec3(0, 0, 0)
        result = time_of_impact_sphere_sphere(
            sphere_a, velocity_a, sphere_b, velocity_b, dt=1.0
        )
        assert not result.hit

    def test_spheres_no_relative_motion(self):
        """Spheres with no relative motion but not overlapping."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        sphere_b = Sphere(center=Vec3(5, 0, 0), radius=1.0)
        velocity_a = Vec3(10, 0, 0)
        velocity_b = Vec3(10, 0, 0)  # Same velocity
        result = time_of_impact_sphere_sphere(
            sphere_a, velocity_a, sphere_b, velocity_b, dt=1.0
        )
        assert not result.hit


class TestConservativeAdvancement:
    """Tests for conservative_advancement function."""

    def test_spheres_collide(self):
        """Conservative advancement should detect sphere collision."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(20, 0, 0))
        sphere_b = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        motion_b = MotionState(velocity=Vec3(0, 0, 0))
        result = conservative_advancement(
            sphere_a, motion_a, sphere_b, motion_b, dt=1.0
        )
        assert result.hit
        assert 0 < result.toi < 1.0

    def test_no_relative_motion(self):
        """No relative motion should return no hit."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(0, 0, 0))
        sphere_b = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        motion_b = MotionState(velocity=Vec3(0, 0, 0))
        result = conservative_advancement(
            sphere_a, motion_a, sphere_b, motion_b, dt=1.0
        )
        assert not result.hit


class TestSpeculativeContacts:
    """Tests for speculative_contacts function."""

    def test_generates_speculative_contact(self):
        """Should generate speculative contact for approaching shapes."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        aabb_a = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        velocity_a = Vec3(5, 0, 0)
        sphere_b = Sphere(center=Vec3(3, 0, 0), radius=1.0)
        aabb_b = AABB(Vec3(2, -1, -1), Vec3(4, 1, 1))
        velocity_b = Vec3(0, 0, 0)
        results = speculative_contacts(
            sphere_a, aabb_a, velocity_a,
            sphere_b, aabb_b, velocity_b,
            dt=1.0,
        )
        # Should have a speculative contact
        assert len(results) >= 1

    def test_no_contact_for_separating_shapes(self):
        """Should not generate contact for shapes moving apart."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        aabb_a = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        velocity_a = Vec3(-10, 0, 0)  # Moving away
        sphere_b = Sphere(center=Vec3(5, 0, 0), radius=1.0)
        aabb_b = AABB(Vec3(4, -1, -1), Vec3(6, 1, 1))
        velocity_b = Vec3(10, 0, 0)  # Also moving away
        results = speculative_contacts(
            sphere_a, aabb_a, velocity_a,
            sphere_b, aabb_b, velocity_b,
            dt=1.0,
        )
        assert len(results) == 0

    def test_already_colliding(self):
        """Already colliding shapes should return immediate contact."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        aabb_a = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        velocity_a = Vec3(0, 0, 0)
        sphere_b = Sphere(center=Vec3(1.5, 0, 0), radius=1.0)  # Overlapping
        aabb_b = AABB(Vec3(0.5, -1, -1), Vec3(2.5, 1, 1))
        velocity_b = Vec3(0, 0, 0)
        results = speculative_contacts(
            sphere_a, aabb_a, velocity_a,
            sphere_b, aabb_b, velocity_b,
            dt=1.0,
        )
        assert len(results) == 1
        assert results[0].toi == 0.0


class TestCCDManager:
    """Tests for CCDManager class."""

    def test_default_construction(self):
        """Default CCDManager should use SWEPT mode."""
        manager = CCDManager()
        assert manager.mode == CCDMode.SWEPT

    def test_mode_setter(self):
        """Mode setter should work."""
        manager = CCDManager()
        manager.mode = CCDMode.SPECULATIVE
        assert manager.mode == CCDMode.SPECULATIVE

    def test_needs_ccd_fast_velocity(self):
        """Fast velocity should need CCD."""
        manager = CCDManager(velocity_threshold=10.0)
        assert manager.needs_ccd(Vec3(15, 0, 0))

    def test_needs_ccd_slow_velocity(self):
        """Slow velocity should not need CCD."""
        manager = CCDManager(velocity_threshold=10.0)
        assert not manager.needs_ccd(Vec3(5, 0, 0))

    def test_test_pair_none_mode(self):
        """NONE mode should use discrete collision only."""
        manager = CCDManager(mode=CCDMode.NONE)
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(20, 0, 0))
        sphere_b = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        motion_b = MotionState()
        result = manager.test_pair(sphere_a, motion_a, sphere_b, motion_b)
        # No collision at start position
        assert not result.hit

    def test_test_pair_swept_mode(self):
        """SWEPT mode should detect collision along path."""
        manager = CCDManager(mode=CCDMode.SWEPT)
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(20, 0, 0))
        sphere_b = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        motion_b = MotionState()
        result = manager.test_pair(sphere_a, motion_a, sphere_b, motion_b)
        assert result.hit

    def test_test_pair_speculative_mode(self):
        """SPECULATIVE mode should use conservative advancement."""
        manager = CCDManager(mode=CCDMode.SPECULATIVE)
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(20, 0, 0))
        sphere_b = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        motion_b = MotionState()
        result = manager.test_pair(sphere_a, motion_a, sphere_b, motion_b)
        assert result.hit

    def test_test_pair_slow_relative_motion(self):
        """Slow relative motion should use discrete test."""
        manager = CCDManager(velocity_threshold=10.0)
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(5, 0, 0))  # Slow
        sphere_b = Sphere(center=Vec3(100, 0, 0), radius=1.0)
        motion_b = MotionState()
        result = manager.test_pair(sphere_a, motion_a, sphere_b, motion_b)
        assert not result.hit

    def test_test_pair_sphere_sphere_specialized(self):
        """Sphere-sphere should use specialized analytical test."""
        manager = CCDManager(mode=CCDMode.SWEPT)
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(20, 0, 0))
        sphere_b = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        motion_b = MotionState(velocity=Vec3(-20, 0, 0))
        result = manager.test_pair(sphere_a, motion_a, sphere_b, motion_b)
        assert result.hit
        # Analytical test should give accurate TOI
        assert 0.1 < result.toi < 0.3

    def test_find_first_impact(self):
        """find_first_impact should find closest hit."""
        manager = CCDManager()
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion = MotionState(velocity=Vec3(100, 0, 0))
        targets = [
            (Sphere(center=Vec3(20, 0, 0), radius=1.0), 1),
            (Sphere(center=Vec3(10, 0, 0), radius=1.0), 2),  # Closest
            (Sphere(center=Vec3(30, 0, 0), radius=1.0), 3),
        ]
        result, target_id = manager.find_first_impact(sphere, motion, targets)
        assert result.hit
        assert target_id == 2  # Should hit closest first

    def test_find_first_impact_no_hits(self):
        """find_first_impact with no hits should return -1."""
        manager = CCDManager()
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion = MotionState(velocity=Vec3(0, 100, 0))  # Moving perpendicular
        targets = [
            (Sphere(center=Vec3(10, 0, 0), radius=1.0), 1),
            (Sphere(center=Vec3(20, 0, 0), radius=1.0), 2),
        ]
        result, target_id = manager.find_first_impact(sphere, motion, targets)
        assert not result.hit
        assert target_id == -1


class TestCCDEdgeCases:
    """Edge case tests for CCD."""

    def test_tunneling_prevention(self):
        """CCD should prevent tunneling through thin objects."""
        # Very fast-moving small sphere
        sphere = Sphere(center=Vec3(0, 0, 0), radius=0.1)
        motion = MotionState(velocity=Vec3(1000, 0, 0))  # 1000 units/second
        # Very thin wall
        wall = Box(center=Vec3(100, 0, 0), half_extents=Vec3(0.01, 10, 10))
        result = linear_sweep_sphere(sphere, motion, wall, dt=1.0)
        assert result.hit

    def test_grazing_collision(self):
        """Sphere grazing another should detect collision."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion = MotionState(velocity=Vec3(20, 0, 0))
        # Target just barely in path
        target = Sphere(center=Vec3(10, 1.9, 0), radius=1.0)
        result = linear_sweep_sphere(sphere, motion, target, dt=1.0)
        assert result.hit

    def test_coincident_start_positions(self):
        """Coincident starting positions should return toi=0."""
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(20, 0, 0))
        sphere_b = Sphere(center=Vec3(0, 0, 0), radius=1.0)  # Same position
        motion_b = MotionState()
        result = time_of_impact(sphere_a, motion_a, sphere_b, motion_b, dt=1.0)
        assert result.hit
        assert result.toi == 0.0

    def test_very_small_time_step(self):
        """Very small time step should still work."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion = MotionState(velocity=Vec3(1000, 0, 0))  # Very fast
        target = Sphere(center=Vec3(0.5, 0, 0), radius=0.1)
        # dt = 0.001 means only 1 unit of travel
        result = linear_sweep_sphere(sphere, motion, target, dt=0.001)
        assert result.hit

    def test_zero_velocity(self):
        """Zero velocity should use discrete test."""
        manager = CCDManager()
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        motion_a = MotionState(velocity=Vec3(0, 0, 0))
        sphere_b = Sphere(center=Vec3(1.5, 0, 0), radius=1.0)  # Overlapping
        motion_b = MotionState()
        result = manager.test_pair(sphere_a, motion_a, sphere_b, motion_b)
        assert result.hit
        assert result.toi == 0.0
