"""
Tests for Tier 46: PHYSICS_SIM decorators.
"""

import pytest

from trinity.decorators.physics_sim import (
    VALID_CCD_MODES,
    VALID_DOMAINS,
    VALID_SOLVER_TYPES,
    buoyancy,
    continuous_collision,
    simulation_domain,
    sleep_threshold,
    solver_hint,
    substep,
    wind_affected,
)
from trinity.decorators.registry import Tier, registry


# =============================================================================
# simulation_domain tests
# =============================================================================


def test_simulation_domain_basic():
    """Test @simulation_domain with valid domain."""

    @simulation_domain(domain="rigid_body")
    class RigidBody:
        pass

    assert hasattr(RigidBody, "_simulation_domain")
    assert RigidBody._simulation_domain is True
    assert RigidBody._physics_domain == "rigid_body"
    assert "simulation_domain" in RigidBody._tags
    assert "physics_sim" in RigidBody._registries


def test_simulation_domain_all_domains():
    """Test @simulation_domain with all valid domains."""
    for domain in VALID_DOMAINS:

        @simulation_domain(domain=domain)
        class PhysicsBody:
            pass

        assert PhysicsBody._physics_domain == domain


def test_simulation_domain_invalid():
    """Test @simulation_domain with invalid domain."""
    with pytest.raises(ValueError, match="invalid domain"):

        @simulation_domain(domain="invalid_domain")
        class BadBody:
            pass


def test_simulation_domain_registry():
    """Test that @simulation_domain is registered correctly."""
    spec = registry._decorators.get("simulation_domain")
    assert spec is not None
    assert spec.tier == Tier.PHYSICS_SIM
    assert spec.name == "simulation_domain"


# =============================================================================
# substep tests
# =============================================================================


def test_substep_defaults():
    """Test @substep with default parameters."""

    @substep()
    class PhysicsObject:
        pass

    assert hasattr(PhysicsObject, "_substep")
    assert PhysicsObject._substep is True
    assert PhysicsObject._substep_min_hz == 60
    assert PhysicsObject._substep_max_hz == 240
    assert PhysicsObject._substep_max_substeps == 4


def test_substep_custom():
    """Test @substep with custom parameters."""

    @substep(min_hz=120, max_hz=480, max_substeps=8)
    class HighPrecisionPhysics:
        pass

    assert HighPrecisionPhysics._substep_min_hz == 120
    assert HighPrecisionPhysics._substep_max_hz == 480
    assert HighPrecisionPhysics._substep_max_substeps == 8


def test_substep_invalid_min_hz():
    """Test @substep with invalid min_hz."""
    with pytest.raises(ValueError, match="min_hz must be > 0"):

        @substep(min_hz=0)
        class BadPhysics:
            pass


def test_substep_invalid_max_hz():
    """Test @substep with max_hz < min_hz."""
    with pytest.raises(ValueError, match="max_hz must be >= min_hz"):

        @substep(min_hz=240, max_hz=60)
        class BadPhysics:
            pass


def test_substep_invalid_max_substeps():
    """Test @substep with invalid max_substeps."""
    with pytest.raises(ValueError, match="max_substeps must be > 0"):

        @substep(max_substeps=0)
        class BadPhysics:
            pass


# =============================================================================
# solver_hint tests
# =============================================================================


def test_solver_hint_defaults():
    """Test @solver_hint with default parameters."""

    @solver_hint()
    class PhysicsBody:
        pass

    assert hasattr(PhysicsBody, "_solver_hint")
    assert PhysicsBody._solver_hint is True
    assert PhysicsBody._solver_type == "pgs"
    assert PhysicsBody._solver_iterations == 4
    assert PhysicsBody._solver_warm_starting is True


def test_solver_hint_all_types():
    """Test @solver_hint with all valid solver types."""
    for solver_type in VALID_SOLVER_TYPES:

        @solver_hint(type=solver_type, iterations=8, warm_starting=False)
        class PhysicsBody:
            pass

        assert PhysicsBody._solver_type == solver_type
        assert PhysicsBody._solver_iterations == 8
        assert PhysicsBody._solver_warm_starting is False


def test_solver_hint_invalid_type():
    """Test @solver_hint with invalid solver type."""
    with pytest.raises(ValueError, match="invalid type"):

        @solver_hint(type="invalid_solver")
        class BadBody:
            pass


def test_solver_hint_invalid_iterations():
    """Test @solver_hint with invalid iterations."""
    with pytest.raises(ValueError, match="iterations must be > 0"):

        @solver_hint(iterations=0)
        class BadBody:
            pass


# =============================================================================
# sleep_threshold tests
# =============================================================================


def test_sleep_threshold_defaults():
    """Test @sleep_threshold with default parameters."""

    @sleep_threshold()
    class SleepableBody:
        pass

    assert hasattr(SleepableBody, "_sleep_threshold")
    assert SleepableBody._sleep_threshold is True
    assert SleepableBody._sleep_linear == 0.1
    assert SleepableBody._sleep_angular == 0.05
    assert SleepableBody._sleep_time == 0.5


def test_sleep_threshold_custom():
    """Test @sleep_threshold with custom parameters."""

    @sleep_threshold(linear=0.05, angular=0.02, time=1.0)
    class CustomSleep:
        pass

    assert CustomSleep._sleep_linear == 0.05
    assert CustomSleep._sleep_angular == 0.02
    assert CustomSleep._sleep_time == 1.0


def test_sleep_threshold_zero_values():
    """Test @sleep_threshold with zero values (edge case)."""

    @sleep_threshold(linear=0.0, angular=0.0, time=0.0)
    class ZeroSleep:
        pass

    assert ZeroSleep._sleep_linear == 0.0
    assert ZeroSleep._sleep_angular == 0.0
    assert ZeroSleep._sleep_time == 0.0


def test_sleep_threshold_invalid_linear():
    """Test @sleep_threshold with negative linear."""
    with pytest.raises(ValueError, match="linear must be >= 0"):

        @sleep_threshold(linear=-0.1)
        class BadBody:
            pass


def test_sleep_threshold_invalid_angular():
    """Test @sleep_threshold with negative angular."""
    with pytest.raises(ValueError, match="angular must be >= 0"):

        @sleep_threshold(angular=-0.05)
        class BadBody:
            pass


def test_sleep_threshold_invalid_time():
    """Test @sleep_threshold with negative time."""
    with pytest.raises(ValueError, match="time must be >= 0"):

        @sleep_threshold(time=-0.5)
        class BadBody:
            pass


# =============================================================================
# continuous_collision tests
# =============================================================================


def test_continuous_collision_defaults():
    """Test @continuous_collision with default mode."""

    @continuous_collision()
    class CCDBody:
        pass

    assert hasattr(CCDBody, "_continuous_collision")
    assert CCDBody._continuous_collision is True
    assert CCDBody._ccd_mode == "none"


def test_continuous_collision_all_modes():
    """Test @continuous_collision with all valid modes."""
    for mode in VALID_CCD_MODES:

        @continuous_collision(mode=mode)
        class CCDBody:
            pass

        assert CCDBody._ccd_mode == mode


def test_continuous_collision_invalid_mode():
    """Test @continuous_collision with invalid mode."""
    with pytest.raises(ValueError, match="invalid mode"):

        @continuous_collision(mode="invalid_mode")
        class BadBody:
            pass


# =============================================================================
# buoyancy tests
# =============================================================================


def test_buoyancy_defaults():
    """Test @buoyancy with default parameters."""

    @buoyancy()
    class FloatingBody:
        pass

    assert hasattr(FloatingBody, "_buoyancy")
    assert FloatingBody._buoyancy is True
    assert FloatingBody._buoyancy_density == 1.0
    assert FloatingBody._buoyancy_drag == 0.5
    assert FloatingBody._buoyancy_angular_drag == 0.1


def test_buoyancy_custom():
    """Test @buoyancy with custom parameters."""

    @buoyancy(density=0.8, drag=1.0, angular_drag=0.2)
    class CustomFloat:
        pass

    assert CustomFloat._buoyancy_density == 0.8
    assert CustomFloat._buoyancy_drag == 1.0
    assert CustomFloat._buoyancy_angular_drag == 0.2


def test_buoyancy_invalid_density():
    """Test @buoyancy with invalid density."""
    with pytest.raises(ValueError, match="density must be > 0"):

        @buoyancy(density=0.0)
        class BadBody:
            pass

    with pytest.raises(ValueError, match="density must be > 0"):

        @buoyancy(density=-1.0)
        class BadBody2:
            pass


def test_buoyancy_invalid_drag():
    """Test @buoyancy with invalid drag."""
    with pytest.raises(ValueError, match="drag must be >= 0"):

        @buoyancy(drag=-0.5)
        class BadBody:
            pass


def test_buoyancy_invalid_angular_drag():
    """Test @buoyancy with invalid angular_drag."""
    with pytest.raises(ValueError, match="angular_drag must be >= 0"):

        @buoyancy(angular_drag=-0.1)
        class BadBody:
            pass


# =============================================================================
# wind_affected tests
# =============================================================================


def test_wind_affected_defaults():
    """Test @wind_affected with default parameters."""

    @wind_affected()
    class WindBody:
        pass

    assert hasattr(WindBody, "_wind_affected")
    assert WindBody._wind_affected is True
    assert WindBody._wind_drag_coefficient == 1.0
    assert WindBody._wind_area == "auto"


def test_wind_affected_custom():
    """Test @wind_affected with custom parameters."""

    @wind_affected(drag_coefficient=2.5, area=10.0)
    class CustomWind:
        pass

    assert CustomWind._wind_drag_coefficient == 2.5
    assert CustomWind._wind_area == 10.0


def test_wind_affected_auto_area():
    """Test @wind_affected with auto area."""

    @wind_affected(area="auto")
    class AutoAreaWind:
        pass

    assert AutoAreaWind._wind_area == "auto"


def test_wind_affected_invalid_drag():
    """Test @wind_affected with invalid drag_coefficient."""
    with pytest.raises(ValueError, match="drag_coefficient must be > 0"):

        @wind_affected(drag_coefficient=0.0)
        class BadBody:
            pass


def test_wind_affected_invalid_area_float():
    """Test @wind_affected with invalid area (negative float)."""
    with pytest.raises(ValueError, match="area must be > 0"):

        @wind_affected(area=-5.0)
        class BadBody:
            pass


def test_wind_affected_invalid_area_string():
    """Test @wind_affected with invalid area string."""
    with pytest.raises(ValueError, match="area must be 'auto' or a positive float"):

        @wind_affected(area="invalid")
        class BadBody:
            pass


# =============================================================================
# Composition tests
# =============================================================================


def test_multiple_decorators():
    """Test composing multiple PHYSICS_SIM decorators."""

    @simulation_domain(domain="rigid_body")
    @substep(min_hz=120, max_hz=240, max_substeps=8)
    @solver_hint(type="tgs", iterations=6)
    @sleep_threshold(linear=0.05, angular=0.02, time=1.0)
    @continuous_collision(mode="sweep")
    class ComplexPhysicsBody:
        pass

    # Check all decorators applied
    assert ComplexPhysicsBody._simulation_domain is True
    assert ComplexPhysicsBody._physics_domain == "rigid_body"
    assert ComplexPhysicsBody._substep is True
    assert ComplexPhysicsBody._substep_min_hz == 120
    assert ComplexPhysicsBody._solver_hint is True
    assert ComplexPhysicsBody._solver_type == "tgs"
    assert ComplexPhysicsBody._sleep_threshold is True
    assert ComplexPhysicsBody._continuous_collision is True
    assert ComplexPhysicsBody._ccd_mode == "sweep"


def test_water_physics():
    """Test water physics composition."""

    @simulation_domain(domain="fluid")
    @buoyancy(density=1.025, drag=0.8, angular_drag=0.15)
    class WaterBody:
        pass

    assert WaterBody._physics_domain == "fluid"
    assert WaterBody._buoyancy is True
    assert WaterBody._buoyancy_density == 1.025


def test_cloth_physics():
    """Test cloth physics composition."""

    @simulation_domain(domain="cloth")
    @wind_affected(drag_coefficient=1.5, area=5.0)
    class ClothObject:
        pass

    assert ClothObject._physics_domain == "cloth"
    assert ClothObject._wind_affected is True
    assert ClothObject._wind_drag_coefficient == 1.5


# =============================================================================
# Tags and registry tests
# =============================================================================


def test_physics_sim_tags():
    """Test that decorators add correct tags."""

    @simulation_domain(domain="vehicle")
    @substep()
    class Vehicle:
        pass

    tags = Vehicle._tags
    assert "simulation_domain" in tags
    assert "physics_domain" in tags
    assert tags["physics_domain"] == "vehicle"
    assert "substep" in tags


def test_physics_sim_registries():
    """Test that all decorators register to physics_sim."""

    @simulation_domain(domain="soft_body")
    @substep()
    @solver_hint()
    @sleep_threshold()
    @continuous_collision()
    @buoyancy()
    @wind_affected()
    class PhysicsEntity:
        pass

    assert "physics_sim" in PhysicsEntity._registries


def test_all_decorators_registered():
    """Test that all PHYSICS_SIM decorators are in registry."""
    expected = [
        "simulation_domain",
        "substep",
        "solver_hint",
        "sleep_threshold",
        "continuous_collision",
        "buoyancy",
        "wind_affected",
    ]

    for name in expected:
        spec = registry._decorators.get(name)
        assert spec is not None, f"Decorator {name} not registered"
        assert spec.tier == Tier.PHYSICS_SIM
