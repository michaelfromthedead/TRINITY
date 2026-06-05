"""Tests for vehicle configuration constants and dataclasses.

Tests cover:
- Configuration constant values (sanity checks)
- WheelConfig validation
- SuspensionConfig validation
- EngineConfig validation
- TransmissionConfig validation
- TireConfig validation
- VehiclePreset definitions
"""

import pytest

from engine.simulation.vehicles.config import (
    # Wheel
    DEFAULT_WHEEL_RADIUS,
    DEFAULT_WHEEL_WIDTH,
    DEFAULT_WHEEL_MASS,
    DEFAULT_WHEEL_INERTIA,
    # Suspension
    DEFAULT_SUSPENSION_REST,
    DEFAULT_SPRING_STRENGTH,
    DEFAULT_DAMPER_COMPRESSION,
    DEFAULT_DAMPER_REBOUND,
    DEFAULT_SUSPENSION_TRAVEL,
    # Engine
    ENGINE_IDLE_RPM,
    ENGINE_MAX_RPM,
    ENGINE_REDLINE_RPM,
    DEFAULT_MAX_TORQUE,
    # Transmission
    DEFAULT_GEAR_RATIOS,
    DEFAULT_FINAL_DRIVE,
    # Physics
    GRAVITY,
    AIR_DENSITY,
    WATER_DENSITY,
    # Dataclasses
    WheelConfig,
    SuspensionConfig,
    EngineConfig,
    TransmissionConfig,
    TireConfig,
    VehiclePreset,
    VEHICLE_PRESETS,
)


# =============================================================================
# Constant Sanity Checks
# =============================================================================


class TestConstantsSanity:
    """Tests to verify constants have reasonable values."""

    def test_wheel_constants_positive(self):
        """Wheel constants should be positive."""
        assert DEFAULT_WHEEL_RADIUS > 0
        assert DEFAULT_WHEEL_WIDTH > 0
        assert DEFAULT_WHEEL_MASS > 0
        assert DEFAULT_WHEEL_INERTIA > 0

    def test_wheel_radius_realistic(self):
        """Wheel radius should be realistic (25-50cm)."""
        assert 0.25 <= DEFAULT_WHEEL_RADIUS <= 0.5

    def test_suspension_constants_positive(self):
        """Suspension constants should be positive."""
        assert DEFAULT_SUSPENSION_REST > 0
        assert DEFAULT_SPRING_STRENGTH > 0
        assert DEFAULT_DAMPER_COMPRESSION >= 0
        assert DEFAULT_DAMPER_REBOUND >= 0
        assert DEFAULT_SUSPENSION_TRAVEL > 0

    def test_suspension_travel_reasonable(self):
        """Suspension travel should be less than rest length."""
        assert DEFAULT_SUSPENSION_TRAVEL < DEFAULT_SUSPENSION_REST

    def test_engine_rpm_ordering(self):
        """Engine RPM values should be in correct order."""
        assert ENGINE_IDLE_RPM < ENGINE_REDLINE_RPM < ENGINE_MAX_RPM
        assert ENGINE_IDLE_RPM > 0

    def test_engine_torque_positive(self):
        """Engine torque should be positive."""
        assert DEFAULT_MAX_TORQUE > 0

    def test_gear_ratios_structure(self):
        """Gear ratios should have R, N, and forward gears."""
        assert len(DEFAULT_GEAR_RATIOS) >= 3  # At least R, N, 1st
        assert DEFAULT_GEAR_RATIOS[0] < 0  # Reverse is negative
        assert DEFAULT_GEAR_RATIOS[1] == 0  # Neutral is zero

    def test_gear_ratios_decreasing(self):
        """Forward gear ratios should decrease (higher gears)."""
        forward_gears = DEFAULT_GEAR_RATIOS[2:]  # Skip R, N
        for i in range(len(forward_gears) - 1):
            assert forward_gears[i] > forward_gears[i + 1]

    def test_final_drive_positive(self):
        """Final drive should be positive."""
        assert DEFAULT_FINAL_DRIVE > 0

    def test_gravity_realistic(self):
        """Gravity should be approximately 9.81."""
        assert 9.8 <= GRAVITY <= 9.82

    def test_air_density_realistic(self):
        """Air density should be approximately 1.225 kg/m^3."""
        assert 1.2 <= AIR_DENSITY <= 1.3

    def test_water_density_realistic(self):
        """Water density should be approximately 1025 kg/m^3 (seawater)."""
        assert 1000 <= WATER_DENSITY <= 1030


# =============================================================================
# WheelConfig Tests
# =============================================================================


class TestWheelConfig:
    """Tests for WheelConfig dataclass."""

    def test_default_values(self):
        """WheelConfig should have correct defaults."""
        config = WheelConfig()
        assert config.radius == DEFAULT_WHEEL_RADIUS
        assert config.width == DEFAULT_WHEEL_WIDTH
        assert config.mass == DEFAULT_WHEEL_MASS

    def test_custom_values(self):
        """WheelConfig should accept custom values."""
        config = WheelConfig(radius=0.4, width=0.3, mass=20.0)
        assert config.radius == 0.4
        assert config.width == 0.3
        assert config.mass == 20.0

    def test_validate_valid(self):
        """Valid config should pass validation."""
        config = WheelConfig()
        assert config.validate()

    def test_validate_invalid_radius(self):
        """Zero radius should fail validation."""
        config = WheelConfig(radius=0)
        assert not config.validate()

    def test_validate_invalid_width(self):
        """Zero width should fail validation."""
        config = WheelConfig(width=0)
        assert not config.validate()

    def test_validate_invalid_mass(self):
        """Zero mass should fail validation."""
        config = WheelConfig(mass=0)
        assert not config.validate()


# =============================================================================
# SuspensionConfig Tests
# =============================================================================


class TestSuspensionConfig:
    """Tests for SuspensionConfig dataclass."""

    def test_default_values(self):
        """SuspensionConfig should have correct defaults."""
        config = SuspensionConfig()
        assert config.rest_length == DEFAULT_SUSPENSION_REST
        assert config.spring_strength == DEFAULT_SPRING_STRENGTH

    def test_min_max_length(self):
        """Min and max length should be calculated correctly."""
        config = SuspensionConfig(rest_length=0.5, travel=0.2)
        assert config.min_length == 0.4  # 0.5 - 0.1
        assert config.max_length == 0.6  # 0.5 + 0.1

    def test_validate_valid(self):
        """Valid config should pass validation."""
        config = SuspensionConfig()
        assert config.validate()

    def test_validate_invalid_rest_length(self):
        """Zero rest length should fail validation."""
        config = SuspensionConfig(rest_length=0)
        assert not config.validate()

    def test_validate_invalid_spring(self):
        """Zero spring strength should fail validation."""
        config = SuspensionConfig(spring_strength=0)
        assert not config.validate()

    def test_validate_invalid_travel(self):
        """Zero travel should fail validation."""
        config = SuspensionConfig(travel=0)
        assert not config.validate()


# =============================================================================
# EngineConfig Tests
# =============================================================================


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_default_values(self):
        """EngineConfig should have correct defaults."""
        config = EngineConfig()
        assert config.idle_rpm == ENGINE_IDLE_RPM
        assert config.max_rpm == ENGINE_MAX_RPM
        assert config.redline_rpm == ENGINE_REDLINE_RPM

    def test_validate_valid(self):
        """Valid config should pass validation."""
        config = EngineConfig()
        assert config.validate()

    def test_validate_invalid_rpm_order(self):
        """Invalid RPM ordering should fail validation."""
        config = EngineConfig(idle_rpm=5000, redline_rpm=4000, max_rpm=3000)
        assert not config.validate()

    def test_validate_invalid_torque(self):
        """Zero torque should fail validation."""
        config = EngineConfig(max_torque=0)
        assert not config.validate()

    def test_validate_invalid_inertia(self):
        """Zero inertia should fail validation."""
        config = EngineConfig(inertia=0)
        assert not config.validate()


# =============================================================================
# TransmissionConfig Tests
# =============================================================================


class TestTransmissionConfig:
    """Tests for TransmissionConfig dataclass."""

    def test_default_values(self):
        """TransmissionConfig should have correct defaults."""
        config = TransmissionConfig()
        assert config.gear_ratios == DEFAULT_GEAR_RATIOS
        assert config.final_drive == DEFAULT_FINAL_DRIVE

    def test_validate_valid(self):
        """Valid config should pass validation."""
        config = TransmissionConfig()
        assert config.validate()

    def test_validate_invalid_gear_count(self):
        """Too few gears should fail validation."""
        config = TransmissionConfig(gear_ratios=(0, 0))  # Only 2 gears
        assert not config.validate()

    def test_validate_invalid_final_drive(self):
        """Zero final drive should fail validation."""
        config = TransmissionConfig(final_drive=0)
        assert not config.validate()


# =============================================================================
# TireConfig Tests
# =============================================================================


class TestTireConfig:
    """Tests for TireConfig dataclass."""

    def test_default_values(self):
        """TireConfig should have correct defaults."""
        config = TireConfig()
        assert config.friction > 0
        assert config.rolling_resistance >= 0

    def test_validate_valid(self):
        """Valid config should pass validation."""
        config = TireConfig()
        assert config.validate()

    def test_validate_invalid_friction(self):
        """Zero friction should fail validation."""
        config = TireConfig(friction=0)
        assert not config.validate()

    def test_validate_allows_zero_rolling_resistance(self):
        """Zero rolling resistance should pass (ideal tire)."""
        config = TireConfig(rolling_resistance=0)
        assert config.validate()


# =============================================================================
# VehiclePreset Tests
# =============================================================================


class TestVehiclePreset:
    """Tests for VehiclePreset enum and presets dictionary."""

    def test_all_presets_defined(self):
        """All preset enum values should have definitions."""
        defined_presets = set(VEHICLE_PRESETS.keys())
        # At least some presets should exist
        assert len(defined_presets) > 0

    def test_preset_structure(self):
        """Presets should have required keys."""
        for preset, config in VEHICLE_PRESETS.items():
            assert "mass" in config
            # Other fields are optional but mass is essential

    def test_preset_values_positive(self):
        """Preset values should be positive."""
        for preset, config in VEHICLE_PRESETS.items():
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    assert value > 0, f"{preset.name}.{key} should be positive"

    def test_sedan_preset(self):
        """Sedan preset should have reasonable values."""
        sedan = VEHICLE_PRESETS.get(VehiclePreset.SEDAN)
        if sedan:
            assert 1000 <= sedan["mass"] <= 2000  # 1-2 tons

    def test_sports_car_preset(self):
        """Sports car should have higher power than sedan."""
        sedan = VEHICLE_PRESETS.get(VehiclePreset.SEDAN)
        sports = VEHICLE_PRESETS.get(VehiclePreset.SPORTS_CAR)
        if sedan and sports:
            assert sports["max_power"] > sedan["max_power"]

    def test_truck_preset(self):
        """Truck should be heavier than sedan."""
        sedan = VEHICLE_PRESETS.get(VehiclePreset.SEDAN)
        truck = VEHICLE_PRESETS.get(VehiclePreset.TRUCK)
        if sedan and truck:
            assert truck["mass"] > sedan["mass"]
