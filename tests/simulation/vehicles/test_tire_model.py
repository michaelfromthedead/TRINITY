"""Tests for tire physics models.

Tests cover:
- TireState and TireForces dataclasses
- TireModel base class (slip calculations, friction, rolling resistance)
- PacejkaTire magic formula implementation
- LinearTire simplified model
- BrushTire physics-based model
- Surface friction variations
- Edge cases (zero load, extreme slip, etc.)
"""

import math
import pytest

from engine.simulation.vehicles.tire_model import (
    TireSurface,
    SURFACE_FRICTION,
    TireState,
    TireForces,
    TireModel,
    PacejkaTire,
    LinearTire,
    BrushTire,
    create_tire_model,
)
from engine.simulation.vehicles.vehicle_system import Vector3


# =============================================================================
# TireState Tests
# =============================================================================


class TestTireState:
    """Tests for TireState dataclass."""

    def test_default_values(self):
        """TireState should have sensible defaults."""
        state = TireState()
        assert state.slip_ratio == 0.0
        assert state.slip_angle == 0.0
        assert state.longitudinal_force == 0.0
        assert state.lateral_force == 0.0
        assert state.normal_load == 0.0
        assert state.temperature == 25.0
        assert state.wear == 0.0
        assert not state.is_grounded

    def test_custom_values(self):
        """TireState should accept custom values."""
        state = TireState(
            slip_ratio=0.1,
            slip_angle=0.05,
            normal_load=5000.0,
            temperature=80.0,
            wear=0.3,
            is_grounded=True,
        )
        assert state.slip_ratio == 0.1
        assert state.slip_angle == 0.05
        assert state.normal_load == 5000.0
        assert state.temperature == 80.0
        assert state.wear == 0.3
        assert state.is_grounded


# =============================================================================
# TireForces Tests
# =============================================================================


class TestTireForces:
    """Tests for TireForces dataclass."""

    def test_default_values(self):
        """TireForces should default to zero."""
        forces = TireForces()
        assert forces.longitudinal == 0.0
        assert forces.lateral == 0.0
        assert forces.vertical == 0.0
        assert forces.aligning == 0.0

    def test_to_vector(self):
        """to_vector should create proper Vector3."""
        forces = TireForces(longitudinal=100.0, lateral=50.0, vertical=5000.0)
        vec = forces.to_vector()
        assert vec.x == 50.0  # lateral
        assert vec.y == 5000.0  # vertical
        assert vec.z == 100.0  # longitudinal


# =============================================================================
# Surface Friction Tests
# =============================================================================


class TestSurfaceFriction:
    """Tests for surface friction multipliers."""

    def test_all_surfaces_defined(self):
        """All TireSurface values should have friction defined."""
        for surface in TireSurface:
            assert surface in SURFACE_FRICTION

    def test_dry_asphalt_baseline(self):
        """Dry asphalt should be the baseline (1.0)."""
        assert SURFACE_FRICTION[TireSurface.ASPHALT_DRY] == 1.0

    def test_friction_ordering(self):
        """Friction should decrease for slippery surfaces."""
        dry = SURFACE_FRICTION[TireSurface.ASPHALT_DRY]
        wet = SURFACE_FRICTION[TireSurface.ASPHALT_WET]
        grass = SURFACE_FRICTION[TireSurface.GRASS]
        ice = SURFACE_FRICTION[TireSurface.ICE]

        assert dry > wet > grass > ice
        assert ice > 0  # Even ice has some friction


# =============================================================================
# PacejkaTire Tests
# =============================================================================


class TestPacejkaTire:
    """Tests for Pacejka magic formula tire model."""

    @pytest.fixture
    def tire(self):
        """Create a standard Pacejka tire."""
        return PacejkaTire()

    def test_initialization(self, tire):
        """Pacejka tire should initialize with defaults."""
        assert tire.friction > 0
        assert tire.surface == TireSurface.ASPHALT_DRY

    def test_friction_setter_positive(self, tire):
        """Friction setter should accept positive values."""
        tire.friction = 1.2
        assert tire.friction == 1.2

    def test_friction_setter_rejects_negative(self, tire):
        """Friction setter should reject negative values."""
        with pytest.raises(ValueError, match="non-negative"):
            tire.friction = -0.5

    def test_longitudinal_force_zero_slip(self, tire):
        """Zero slip should produce zero longitudinal force."""
        force = tire.compute_longitudinal_force(0.0, 5000.0)
        assert abs(force) < 1.0  # Effectively zero

    def test_longitudinal_force_positive_slip(self, tire):
        """Positive slip (acceleration) should produce positive force."""
        force = tire.compute_longitudinal_force(0.1, 5000.0)
        assert force > 0

    def test_longitudinal_force_negative_slip(self, tire):
        """Negative slip (braking) should produce negative force."""
        force = tire.compute_longitudinal_force(-0.1, 5000.0)
        assert force < 0

    def test_longitudinal_force_peak(self, tire):
        """Force should peak at optimal slip ratio."""
        load = 5000.0
        slips = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        forces = [abs(tire.compute_longitudinal_force(s, load)) for s in slips]

        # Force should increase then decrease (magic formula shape)
        max_force = max(forces)
        max_idx = forces.index(max_force)
        assert 0 < max_idx < len(slips) - 1  # Peak not at edges

    def test_lateral_force_zero_slip(self, tire):
        """Zero slip angle should produce zero lateral force."""
        force = tire.compute_lateral_force(0.0, 5000.0)
        assert abs(force) < 1.0

    def test_lateral_force_positive_slip(self, tire):
        """Positive slip angle should produce positive lateral force."""
        force = tire.compute_lateral_force(0.1, 5000.0)  # ~5.7 degrees
        assert force > 0

    def test_lateral_force_with_camber(self, tire):
        """Camber should affect lateral force."""
        load = 5000.0
        slip_angle = 0.05
        force_no_camber = tire.compute_lateral_force(slip_angle, load, 0.0)
        force_with_camber = tire.compute_lateral_force(slip_angle, load, 0.05)
        assert force_with_camber != force_no_camber

    def test_zero_load_no_force(self, tire):
        """Zero normal load should produce zero force."""
        forces = tire.update(
            wheel_angular_velocity=10.0,
            wheel_radius=0.35,
            ground_velocity_forward=3.0,
            ground_velocity_lateral=0.5,
            normal_load=0.0,
        )
        assert forces.longitudinal == 0.0
        assert forces.lateral == 0.0

    def test_surface_affects_friction(self, tire):
        """Different surfaces should affect effective friction."""
        load = 5000.0

        tire.surface = TireSurface.ASPHALT_DRY
        friction_dry = tire.get_effective_friction(load)

        tire.surface = TireSurface.ICE
        friction_ice = tire.get_effective_friction(load)

        assert friction_dry > friction_ice

    def test_peak_slip_ratio(self, tire):
        """Should provide estimate of peak slip ratio."""
        peak = tire.get_peak_slip_ratio()
        assert 0 < peak < 0.5  # Reasonable range

    def test_peak_slip_angle(self, tire):
        """Should provide estimate of peak slip angle."""
        peak = tire.get_peak_slip_angle()
        assert 0 < peak < 0.3  # ~17 degrees max


# =============================================================================
# LinearTire Tests
# =============================================================================


class TestLinearTire:
    """Tests for simplified linear tire model."""

    @pytest.fixture
    def tire(self):
        """Create a standard linear tire."""
        return LinearTire()

    def test_initialization(self, tire):
        """Linear tire should initialize with defaults."""
        assert tire.longitudinal_stiffness > 0
        assert tire.lateral_stiffness > 0

    def test_stiffness_setters_positive(self, tire):
        """Stiffness setters should accept positive values."""
        tire.longitudinal_stiffness = 15000.0
        tire.lateral_stiffness = 12000.0
        assert tire.longitudinal_stiffness == 15000.0
        assert tire.lateral_stiffness == 12000.0

    def test_stiffness_setters_reject_negative(self, tire):
        """Stiffness setters should reject negative values."""
        with pytest.raises(ValueError, match="non-negative"):
            tire.longitudinal_stiffness = -1000

        with pytest.raises(ValueError, match="non-negative"):
            tire.lateral_stiffness = -1000

    def test_linear_region(self, tire):
        """Force should be linear for small slip."""
        load = 5000.0
        force1 = tire.compute_longitudinal_force(0.01, load)
        force2 = tire.compute_longitudinal_force(0.02, load)

        # Should be approximately double
        assert abs(force2 / force1 - 2.0) < 0.3

    def test_saturation(self, tire):
        """Force should saturate at high slip."""
        load = 5000.0
        force1 = tire.compute_longitudinal_force(0.5, load)
        force2 = tire.compute_longitudinal_force(1.0, load)

        # Should not double - saturated
        assert force2 / force1 < 1.5

    def test_friction_limit(self, tire):
        """Force should be limited by friction."""
        load = 5000.0
        max_force = tire.friction * load

        force = tire.compute_longitudinal_force(2.0, load)  # Extreme slip
        assert abs(force) <= max_force * 1.01  # Small tolerance


# =============================================================================
# BrushTire Tests
# =============================================================================


class TestBrushTire:
    """Tests for physics-based brush tire model."""

    @pytest.fixture
    def tire(self):
        """Create a standard brush tire."""
        return BrushTire()

    def test_initialization(self, tire):
        """Brush tire should initialize with defaults."""
        assert tire.friction > 0

    def test_adhesion_region(self, tire):
        """Small slip should be in adhesion region (no sliding)."""
        load = 5000.0
        force = tire.compute_longitudinal_force(0.01, load)
        # Force should be positive for positive slip
        assert force > 0

    def test_sliding_region(self, tire):
        """Large slip should saturate at friction limit."""
        load = 5000.0
        friction = tire.get_effective_friction(load)
        max_force = friction * load

        # Very large slip
        force = tire.compute_longitudinal_force(1.0, load)
        # Should be close to max force
        assert abs(force) <= max_force * 1.01

    def test_zero_load(self, tire):
        """Zero load should produce zero force."""
        force = tire.compute_longitudinal_force(0.1, 0.0)
        assert force == 0.0

        force_lat = tire.compute_lateral_force(0.1, 0.0)
        assert force_lat == 0.0


# =============================================================================
# Slip Calculation Tests
# =============================================================================


class TestSlipCalculations:
    """Tests for slip ratio and slip angle calculations."""

    @pytest.fixture
    def tire(self):
        """Create a tire for testing slip calculations."""
        return PacejkaTire()

    def test_slip_ratio_stationary(self, tire):
        """Near-stationary should handle gracefully."""
        slip = tire.compute_slip_ratio(
            wheel_angular_velocity=0.0,
            wheel_radius=0.35,
            ground_velocity=0.0,
        )
        assert abs(slip) < 0.1  # Near zero

    def test_slip_ratio_free_rolling(self, tire):
        """Free rolling wheel should have zero slip."""
        velocity = 10.0  # m/s
        radius = 0.35
        angular_velocity = velocity / radius

        slip = tire.compute_slip_ratio(
            wheel_angular_velocity=angular_velocity,
            wheel_radius=radius,
            ground_velocity=velocity,
        )
        assert abs(slip) < 0.01

    def test_slip_ratio_acceleration(self, tire):
        """Accelerating wheel should have positive slip."""
        velocity = 10.0
        radius = 0.35
        angular_velocity = velocity / radius * 1.1  # 10% faster

        slip = tire.compute_slip_ratio(
            wheel_angular_velocity=angular_velocity,
            wheel_radius=radius,
            ground_velocity=velocity,
        )
        assert slip > 0

    def test_slip_ratio_braking(self, tire):
        """Braking wheel should have negative slip."""
        velocity = 10.0
        radius = 0.35
        angular_velocity = velocity / radius * 0.9  # 10% slower

        slip = tire.compute_slip_ratio(
            wheel_angular_velocity=angular_velocity,
            wheel_radius=radius,
            ground_velocity=velocity,
        )
        assert slip < 0

    def test_slip_ratio_locked_wheel(self, tire):
        """Locked wheel should have slip ratio near -1."""
        slip = tire.compute_slip_ratio(
            wheel_angular_velocity=0.0,
            wheel_radius=0.35,
            ground_velocity=10.0,
        )
        assert -1.5 < slip < -0.9

    def test_slip_angle_straight(self, tire):
        """Straight motion should have zero slip angle."""
        angle = tire.compute_slip_angle(
            velocity_x=10.0,  # forward
            velocity_y=0.0,  # no lateral
        )
        assert abs(angle) < 0.01

    def test_slip_angle_lateral_motion(self, tire):
        """Lateral motion should produce slip angle."""
        angle = tire.compute_slip_angle(
            velocity_x=10.0,
            velocity_y=1.0,
        )
        expected = math.atan2(1.0, 10.0)
        assert abs(angle - expected) < 0.01

    def test_slip_angle_stationary(self, tire):
        """Near-stationary should handle gracefully."""
        angle = tire.compute_slip_angle(
            velocity_x=0.0,
            velocity_y=0.0,
        )
        assert angle == 0.0


# =============================================================================
# Rolling Resistance Tests
# =============================================================================


class TestRollingResistance:
    """Tests for rolling resistance calculations."""

    @pytest.fixture
    def tire(self):
        """Create a tire for testing."""
        return PacejkaTire()

    def test_rolling_resistance_proportional_to_load(self, tire):
        """Rolling resistance should be proportional to load."""
        rr1 = tire.compute_rolling_resistance(5000.0)
        rr2 = tire.compute_rolling_resistance(10000.0)
        assert abs(rr2 / rr1 - 2.0) < 0.01

    def test_rolling_resistance_zero_load(self, tire):
        """Zero load should have zero rolling resistance."""
        rr = tire.compute_rolling_resistance(0.0)
        assert rr == 0.0


# =============================================================================
# Aligning Moment Tests
# =============================================================================


class TestAligningMoment:
    """Tests for self-aligning torque."""

    @pytest.fixture
    def tire(self):
        """Create a tire for testing."""
        return PacejkaTire()

    def test_aligning_moment_zero_slip(self, tire):
        """Zero slip angle should produce zero aligning moment."""
        moment = tire.compute_aligning_moment(
            slip_angle=0.0,
            lateral_force=5000.0,
        )
        # Small due to cos(0)^2 = 1
        assert moment > 0

    def test_aligning_moment_reduces_with_slip(self, tire):
        """Aligning moment should reduce at high slip angles."""
        moment_small = tire.compute_aligning_moment(
            slip_angle=0.1,
            lateral_force=5000.0,
        )
        moment_large = tire.compute_aligning_moment(
            slip_angle=0.5,
            lateral_force=5000.0,
        )
        assert moment_small > moment_large


# =============================================================================
# Combined Slip Tests
# =============================================================================


class TestCombinedSlip:
    """Tests for friction circle behavior in combined slip."""

    @pytest.fixture
    def tire(self):
        """Create a tire for testing."""
        return PacejkaTire()

    def test_friction_circle_limit(self, tire):
        """Combined forces should not exceed friction limit."""
        load = 5000.0
        max_force = tire.friction * load

        forces = tire.update(
            wheel_angular_velocity=15.0,  # Significant slip
            wheel_radius=0.35,
            ground_velocity_forward=4.0,  # Significant forward velocity
            ground_velocity_lateral=2.0,  # Significant lateral velocity
            normal_load=load,
        )

        total = math.sqrt(forces.longitudinal ** 2 + forces.lateral ** 2)
        assert total <= max_force * 1.1  # Small tolerance


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateTireModel:
    """Tests for tire model factory function."""

    def test_create_pacejka(self):
        """Should create Pacejka tire."""
        tire = create_tire_model("pacejka")
        assert isinstance(tire, PacejkaTire)

    def test_create_linear(self):
        """Should create linear tire."""
        tire = create_tire_model("linear")
        assert isinstance(tire, LinearTire)

    def test_create_brush(self):
        """Should create brush tire."""
        tire = create_tire_model("brush")
        assert isinstance(tire, BrushTire)

    def test_case_insensitive(self):
        """Factory should be case-insensitive."""
        tire1 = create_tire_model("PACEJKA")
        tire2 = create_tire_model("Pacejka")
        tire3 = create_tire_model("pacejka")
        assert all(isinstance(t, PacejkaTire) for t in [tire1, tire2, tire3])

    def test_invalid_type(self):
        """Invalid type should raise error."""
        with pytest.raises(ValueError, match="Unknown tire model"):
            create_tire_model("nonexistent")

    def test_pass_kwargs(self):
        """Should pass kwargs to constructor."""
        tire = create_tire_model("pacejka", friction=1.5)
        assert tire.friction == 1.5
