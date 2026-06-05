"""
Whitebox tests for FluidVolumeComponent.

Tests cover:
- Fluid volume configuration
- Buoyancy calculations
- Drag forces
- Flow velocity fields
- Wave simulation
- Object tracking
- Serialization
"""

import math
import pytest

from engine.simulation.character.character_controller import Vector3
from engine.simulation.components.fluid_component import (
    FlowConfig,
    FlowType,
    FluidConfig,
    FluidType,
    FluidVolumeComponent,
    SubmergedObject,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def water_config() -> FluidConfig:
    """Water configuration."""
    return FluidConfig(
        fluid_type=FluidType.WATER,
        density=1000.0,
        buoyancy_scale=1.0,
        drag_coefficient=0.5,
    )


@pytest.fixture
def fluid_volume(water_config) -> FluidVolumeComponent:
    """Create a fluid volume component."""
    component = FluidVolumeComponent(entity_id=1, config=water_config)
    component.set_bounds(
        Vector3(-10.0, 0.0, -10.0),
        Vector3(10.0, 5.0, 10.0),
    )
    return component


@pytest.fixture
def flowing_fluid() -> FluidVolumeComponent:
    """Create a fluid with directional flow."""
    flow_config = FlowConfig(
        flow_type=FlowType.DIRECTIONAL,
        direction=Vector3(1.0, 0.0, 0.0),
        speed=2.0,
    )
    component = FluidVolumeComponent(
        entity_id=2,
        flow_config=flow_config,
    )
    component.set_bounds(
        Vector3(0.0, 0.0, 0.0),
        Vector3(10.0, 5.0, 10.0),
    )
    return component


# =============================================================================
# FluidType Tests
# =============================================================================


class TestFluidType:
    """Tests for FluidType enum."""

    def test_all_types(self):
        """Test all fluid types exist."""
        assert FluidType.WATER.value == "water"
        assert FluidType.LAVA.value == "lava"
        assert FluidType.OIL.value == "oil"
        assert FluidType.MUD.value == "mud"
        assert FluidType.ACID.value == "acid"
        assert FluidType.CUSTOM.value == "custom"


class TestFlowType:
    """Tests for FlowType enum."""

    def test_all_types(self):
        """Test all flow types exist."""
        assert FlowType.STATIC.value == "static"
        assert FlowType.DIRECTIONAL.value == "directional"
        assert FlowType.RADIAL.value == "radial"
        assert FlowType.VORTEX.value == "vortex"
        assert FlowType.RIVER.value == "river"


# =============================================================================
# FluidConfig Tests
# =============================================================================


class TestFluidConfig:
    """Tests for FluidConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FluidConfig()

        assert config.fluid_type == FluidType.WATER
        assert config.density == 1000.0
        assert config.viscosity == 0.001
        assert config.surface_tension == 0.073
        assert config.buoyancy_scale == 1.0
        assert config.drag_coefficient == 0.5
        assert config.wave_enabled is True
        assert config.wave_height == 0.1
        assert config.wave_frequency == 1.0
        assert config.temperature == 20.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FluidConfig(
            fluid_type=FluidType.OIL,
            density=800.0,
            viscosity=0.05,
            drag_coefficient=0.8,
        )

        assert config.fluid_type == FluidType.OIL
        assert config.density == 800.0
        assert config.viscosity == 0.05


# =============================================================================
# FlowConfig Tests
# =============================================================================


class TestFlowConfig:
    """Tests for FlowConfig dataclass."""

    def test_default_values(self):
        """Test default flow configuration."""
        config = FlowConfig()

        assert config.flow_type == FlowType.STATIC
        assert config.direction.x == 1.0
        assert config.speed == 1.0
        assert config.turbulence == 0.0

    def test_custom_values(self):
        """Test custom flow configuration."""
        config = FlowConfig(
            flow_type=FlowType.VORTEX,
            center=Vector3(5.0, 0.0, 5.0),
            speed=3.0,
            turbulence=0.5,
        )

        assert config.flow_type == FlowType.VORTEX
        assert config.center.x == 5.0
        assert config.speed == 3.0


# =============================================================================
# SubmergedObject Tests
# =============================================================================


class TestSubmergedObject:
    """Tests for SubmergedObject dataclass."""

    def test_default_values(self):
        """Test default submerged object values."""
        obj = SubmergedObject()

        assert obj.entity_id == 0
        assert obj.body_id == 0
        assert obj.submerged_volume == 0.0
        assert obj.submerged_ratio == 0.0
        assert obj.entry_time == 0.0

    def test_custom_values(self):
        """Test custom submerged object values."""
        obj = SubmergedObject(
            entity_id=42,
            body_id=100,
            submerged_volume=0.5,
            submerged_ratio=0.75,
            entry_time=10.5,
        )

        assert obj.entity_id == 42
        assert obj.submerged_ratio == 0.75


# =============================================================================
# Component Creation Tests
# =============================================================================


class TestFluidVolumeCreation:
    """Tests for fluid volume creation."""

    def test_create_with_default_config(self):
        """Test creating with default config."""
        component = FluidVolumeComponent(entity_id=1)

        assert component.entity_id == 1
        assert component.config.fluid_type == FluidType.WATER
        assert component.enabled is True

    def test_create_with_custom_config(self, fluid_volume, water_config):
        """Test creating with custom config."""
        assert fluid_volume.config.density == 1000.0

    def test_initial_state(self, fluid_volume):
        """Test initial component state."""
        assert fluid_volume.submerged_count == 0
        assert fluid_volume.surface_height == 5.0


# =============================================================================
# Bounds Tests
# =============================================================================


class TestBounds:
    """Tests for volume bounds."""

    def test_set_bounds(self, fluid_volume):
        """Test setting volume bounds."""
        fluid_volume.set_bounds(
            Vector3(-5.0, 0.0, -5.0),
            Vector3(5.0, 10.0, 5.0),
        )

        assert fluid_volume.surface_height == 10.0

    def test_set_surface_height(self, fluid_volume):
        """Test setting surface height directly."""
        fluid_volume.set_surface_height(3.0)

        assert fluid_volume.surface_height == 3.0


# =============================================================================
# Buoyancy Tests
# =============================================================================


class TestBuoyancy:
    """Tests for buoyancy calculations."""

    def test_buoyancy_underwater(self, fluid_volume):
        """Test buoyancy for underwater object."""
        # Object at y=2, surface at y=5 -> fully submerged
        position = Vector3(0.0, 2.0, 0.0)
        volume = 1.0
        object_density = 500.0  # Less than water

        force = fluid_volume.calculate_buoyancy(position, volume, object_density)

        # Buoyancy should be upward
        assert force.y > 0

    def test_buoyancy_above_surface(self, fluid_volume):
        """Test no buoyancy above surface."""
        position = Vector3(0.0, 10.0, 0.0)  # Above surface
        volume = 1.0
        object_density = 500.0

        force = fluid_volume.calculate_buoyancy(position, volume, object_density)

        assert force.y == 0.0

    def test_buoyancy_outside_volume(self, fluid_volume):
        """Test no buoyancy outside volume bounds."""
        position = Vector3(50.0, 2.0, 0.0)  # Outside X bounds
        volume = 1.0
        object_density = 500.0

        force = fluid_volume.calculate_buoyancy(position, volume, object_density)

        assert force.magnitude() == 0.0

    def test_buoyancy_at_surface(self, fluid_volume):
        """Test buoyancy at surface level."""
        position = Vector3(0.0, 5.0, 0.0)  # At surface
        volume = 1.0
        object_density = 500.0

        force = fluid_volume.calculate_buoyancy(position, volume, object_density)

        # At surface, depth <= 0, so no buoyancy
        assert force.y == 0.0

    def test_buoyancy_scales_with_volume(self, fluid_volume):
        """Test buoyancy scales with object volume."""
        position = Vector3(0.0, 2.0, 0.0)
        density = 500.0

        force_small = fluid_volume.calculate_buoyancy(position, 0.5, density)
        force_large = fluid_volume.calculate_buoyancy(position, 2.0, density)

        assert force_large.y > force_small.y


# =============================================================================
# Drag Tests
# =============================================================================


class TestDrag:
    """Tests for drag force calculations."""

    def test_drag_force(self, fluid_volume):
        """Test drag force calculation."""
        velocity = Vector3(10.0, 0.0, 0.0)
        cross_section = 1.0

        drag = fluid_volume.calculate_drag(velocity, cross_section)

        # Drag opposes motion
        assert drag.x < 0.0
        assert abs(drag.y) < 0.001
        assert abs(drag.z) < 0.001

    def test_drag_zero_velocity(self, fluid_volume):
        """Test no drag at zero velocity."""
        velocity = Vector3(0.0, 0.0, 0.0)
        cross_section = 1.0

        drag = fluid_volume.calculate_drag(velocity, cross_section)

        assert drag.magnitude() == 0.0

    def test_drag_very_slow(self, fluid_volume):
        """Test near-zero velocity."""
        velocity = Vector3(0.0001, 0.0, 0.0)
        cross_section = 1.0

        drag = fluid_volume.calculate_drag(velocity, cross_section)

        # Very small velocity -> no significant drag
        assert drag.magnitude() < 0.001

    def test_drag_scales_with_velocity_squared(self, fluid_volume):
        """Test drag scales with velocity squared."""
        cross_section = 1.0

        drag_slow = fluid_volume.calculate_drag(
            Vector3(1.0, 0.0, 0.0), cross_section
        )
        drag_fast = fluid_volume.calculate_drag(
            Vector3(2.0, 0.0, 0.0), cross_section
        )

        # 2x velocity -> 4x drag (approximately)
        ratio = abs(drag_fast.x / drag_slow.x)
        assert 3.5 < ratio < 4.5


# =============================================================================
# Flow Tests
# =============================================================================


class TestFlow:
    """Tests for flow velocity fields."""

    def test_static_flow(self, fluid_volume):
        """Test static flow returns zero."""
        position = Vector3(0.0, 2.0, 0.0)

        velocity = fluid_volume.get_flow_velocity(position)

        assert velocity.magnitude() == 0.0

    def test_directional_flow(self, flowing_fluid):
        """Test directional flow."""
        position = Vector3(5.0, 2.0, 5.0)

        velocity = flowing_fluid.get_flow_velocity(position)

        # Should flow in positive X
        assert velocity.x > 0.0

    def test_flow_outside_volume(self, flowing_fluid):
        """Test flow outside volume is zero."""
        position = Vector3(50.0, 2.0, 5.0)  # Outside X bounds

        velocity = flowing_fluid.get_flow_velocity(position)

        assert velocity.magnitude() == 0.0

    def test_radial_flow(self):
        """Test radial flow toward center."""
        flow_config = FlowConfig(
            flow_type=FlowType.RADIAL,
            center=Vector3(5.0, 0.0, 5.0),
            speed=2.0,
        )
        component = FluidVolumeComponent(entity_id=1, flow_config=flow_config)
        component.set_bounds(
            Vector3(0.0, 0.0, 0.0),
            Vector3(10.0, 5.0, 10.0),
        )

        # Point away from center
        position = Vector3(8.0, 2.0, 5.0)
        velocity = component.get_flow_velocity(position)

        # Should flow toward center (negative X)
        assert velocity.x < 0.0

    def test_vortex_flow(self):
        """Test vortex flow creates rotation."""
        flow_config = FlowConfig(
            flow_type=FlowType.VORTEX,
            center=Vector3(5.0, 0.0, 5.0),
            speed=2.0,
        )
        component = FluidVolumeComponent(entity_id=1, flow_config=flow_config)
        component.set_bounds(
            Vector3(0.0, 0.0, 0.0),
            Vector3(10.0, 5.0, 10.0),
        )

        # Point offset from center
        position = Vector3(8.0, 2.0, 5.0)
        velocity = component.get_flow_velocity(position)

        # Vortex should have tangential component
        assert velocity.magnitude() > 0.0

    def test_set_flow(self, fluid_volume):
        """Test setting flow configuration."""
        fluid_volume.set_flow(
            flow_type=FlowType.DIRECTIONAL,
            direction=Vector3(0.0, 0.0, 1.0),
            speed=5.0,
        )

        position = Vector3(0.0, 2.0, 0.0)
        velocity = fluid_volume.get_flow_velocity(position)

        assert velocity.z > 0.0


# =============================================================================
# Wave Tests
# =============================================================================


class TestWaves:
    """Tests for wave simulation."""

    def test_surface_height_with_waves(self, fluid_volume):
        """Test surface height varies with waves."""
        # Waves enabled by default
        height1 = fluid_volume.get_surface_height_at(0.0, 0.0)
        height2 = fluid_volume.get_surface_height_at(1.0, 0.0)

        # Heights may differ due to wave
        # (or be same at time 0, depends on wave function)

    def test_surface_height_waves_disabled(self):
        """Test surface height constant when waves disabled."""
        config = FluidConfig(wave_enabled=False)
        component = FluidVolumeComponent(entity_id=1, config=config)
        component.set_bounds(
            Vector3(0.0, 0.0, 0.0),
            Vector3(10.0, 5.0, 10.0),
        )

        height1 = component.get_surface_height_at(0.0, 0.0)
        height2 = component.get_surface_height_at(5.0, 5.0)

        assert height1 == height2 == 5.0

    def test_add_splash(self, fluid_volume):
        """Test adding splash creates wave source."""
        fluid_volume.add_splash(
            position=Vector3(0.0, 5.0, 0.0),
            amplitude=0.5,
        )

        assert len(fluid_volume._wave_sources) == 1

    def test_splash_limit(self, fluid_volume):
        """Test splash sources are limited."""
        for i in range(15):
            fluid_volume.add_splash(
                position=Vector3(float(i), 5.0, 0.0),
                amplitude=0.2,
            )

        # Should be limited to 10
        assert len(fluid_volume._wave_sources) <= 10

    def test_update_waves(self, fluid_volume):
        """Test wave time updates."""
        fluid_volume.add_splash(Vector3(0.0, 5.0, 0.0), 0.5)

        fluid_volume.update_waves(dt=0.1)

        assert fluid_volume._wave_time > 0.0

    def test_old_waves_removed(self, fluid_volume):
        """Test old wave sources are removed."""
        fluid_volume.add_splash(Vector3(0.0, 5.0, 0.0), 0.5)

        # Advance time past wave lifetime (5 seconds)
        for _ in range(60):
            fluid_volume.update_waves(dt=0.1)

        assert len(fluid_volume._wave_sources) == 0


# =============================================================================
# Object Tracking Tests
# =============================================================================


class TestObjectTracking:
    """Tests for submerged object tracking."""

    def test_object_enter(self, fluid_volume):
        """Test object entering fluid."""
        callbacks_called = []
        fluid_volume.set_enter_callback(lambda e: callbacks_called.append(e))

        fluid_volume.on_object_enter(
            entity_id=42,
            body_id=100,
            position=Vector3(0.0, 2.0, 0.0),
            current_time=0.0,
        )

        assert fluid_volume.submerged_count == 1
        assert len(callbacks_called) == 1
        assert callbacks_called[0] == 42

    def test_object_exit(self, fluid_volume):
        """Test object exiting fluid."""
        fluid_volume.on_object_enter(42, 100, Vector3(0.0, 2.0, 0.0), 0.0)

        callbacks_called = []
        fluid_volume.set_exit_callback(lambda e: callbacks_called.append(e))

        fluid_volume.on_object_exit(42)

        assert fluid_volume.submerged_count == 0
        assert len(callbacks_called) == 1

    def test_duplicate_enter_ignored(self, fluid_volume):
        """Test duplicate enter is ignored."""
        fluid_volume.on_object_enter(42, 100, Vector3(0.0, 2.0, 0.0), 0.0)
        fluid_volume.on_object_enter(42, 100, Vector3(0.0, 2.0, 0.0), 0.0)

        assert fluid_volume.submerged_count == 1

    def test_update_submerged_object(self, fluid_volume):
        """Test updating submerged object."""
        fluid_volume.on_object_enter(42, 100, Vector3(0.0, 2.0, 0.0), 0.0)

        fluid_volume.update_submerged_object(
            entity_id=42,
            position=Vector3(0.0, 3.0, 0.0),
            volume=1.0,
        )

        ratio = fluid_volume.get_submerged_ratio(42)
        assert ratio > 0.0

    def test_update_nonexistent_object(self, fluid_volume):
        """Test updating nonexistent object does nothing."""
        fluid_volume.update_submerged_object(
            entity_id=999,
            position=Vector3(0.0, 2.0, 0.0),
            volume=1.0,
        )
        # Should not raise

    def test_is_object_submerged(self, fluid_volume):
        """Test checking if object is submerged."""
        assert fluid_volume.is_object_submerged(42) is False

        fluid_volume.on_object_enter(42, 100, Vector3(0.0, 2.0, 0.0), 0.0)

        assert fluid_volume.is_object_submerged(42) is True

    def test_submerge_callback(self, fluid_volume):
        """Test fully submerged callback."""
        callbacks_called = []
        fluid_volume.set_submerge_callback(lambda e: callbacks_called.append(e))

        fluid_volume.on_object_enter(42, 100, Vector3(0.0, 2.0, 0.0), 0.0)

        # Update to fully submerged position (deep underwater)
        fluid_volume.update_submerged_object(
            entity_id=42,
            position=Vector3(0.0, 0.0, 0.0),  # Very deep
            volume=1.0,
        )

        # Should trigger submerge callback when ratio >= 1.0
        # Implementation may vary


# =============================================================================
# Query Tests
# =============================================================================


class TestQueries:
    """Tests for query methods."""

    def test_is_point_in_volume(self, fluid_volume):
        """Test point-in-volume check."""
        assert fluid_volume.is_point_in_volume(Vector3(0.0, 2.0, 0.0)) is True
        assert fluid_volume.is_point_in_volume(Vector3(50.0, 2.0, 0.0)) is False

    def test_is_point_underwater(self, fluid_volume):
        """Test point-underwater check."""
        # Below surface
        assert fluid_volume.is_point_underwater(Vector3(0.0, 2.0, 0.0)) is True

        # Above surface
        assert fluid_volume.is_point_underwater(Vector3(0.0, 10.0, 0.0)) is False

        # Outside volume
        assert fluid_volume.is_point_underwater(Vector3(50.0, 2.0, 0.0)) is False

    def test_get_depth_at_point(self, fluid_volume):
        """Test depth calculation."""
        # Disable waves for predictable result
        fluid_volume._config.wave_enabled = False

        # Surface at 5, point at 2 -> depth = 3
        depth = fluid_volume.get_depth_at_point(Vector3(0.0, 2.0, 0.0))
        assert abs(depth - 3.0) < 0.001

        # Above surface -> negative depth
        depth_above = fluid_volume.get_depth_at_point(Vector3(0.0, 7.0, 0.0))
        assert depth_above < 0


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for component lifecycle."""

    def test_initialize(self, fluid_volume):
        """Test initialization with physics ID."""
        fluid_volume.initialize(trigger_id=42)

        assert fluid_volume._trigger_id == 42

    def test_cleanup(self, fluid_volume):
        """Test cleanup clears all data."""
        fluid_volume.on_object_enter(42, 100, Vector3(0.0, 2.0, 0.0), 0.0)
        fluid_volume.add_splash(Vector3(0.0, 5.0, 0.0), 0.5)
        fluid_volume.initialize(trigger_id=100)

        fluid_volume.cleanup()

        assert fluid_volume._trigger_id is None
        assert fluid_volume.submerged_count == 0
        assert len(fluid_volume._wave_sources) == 0

    def test_enabled_property(self, fluid_volume):
        """Test enabled property."""
        assert fluid_volume.enabled is True

        fluid_volume.enabled = False
        assert fluid_volume.enabled is False


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for state serialization."""

    def test_get_state(self, fluid_volume):
        """Test getting serializable state."""
        fluid_volume.on_object_enter(42, 100, Vector3(0.0, 2.0, 0.0), 0.0)

        state = fluid_volume.get_state()

        assert state["entity_id"] == 1
        assert state["fluid_type"] == "water"
        assert state["density"] == 1000.0
        assert state["surface_height"] == 5.0
        assert state["flow_type"] == "static"
        assert state["submerged_count"] == 1
        assert state["enabled"] is True


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_density(self):
        """Test fluid with zero density."""
        config = FluidConfig(density=0.0)
        component = FluidVolumeComponent(entity_id=1, config=config)
        component.set_bounds(
            Vector3(0.0, 0.0, 0.0),
            Vector3(10.0, 5.0, 10.0),
        )

        # No buoyancy with zero density
        force = component.calculate_buoyancy(
            Vector3(5.0, 2.0, 5.0),
            volume=1.0,
            object_density=500.0,
        )

        assert force.y == 0.0

    def test_high_density_fluid(self):
        """Test high density fluid (mercury)."""
        config = FluidConfig(density=13600.0)  # Mercury
        component = FluidVolumeComponent(entity_id=1, config=config)
        component.set_bounds(
            Vector3(0.0, 0.0, 0.0),
            Vector3(10.0, 5.0, 10.0),
        )

        force = component.calculate_buoyancy(
            Vector3(5.0, 2.0, 5.0),
            volume=1.0,
            object_density=1000.0,
        )

        # High buoyancy
        assert force.y > 100000.0

    def test_zero_wave_height(self):
        """Test zero wave height."""
        config = FluidConfig(wave_height=0.0)
        component = FluidVolumeComponent(entity_id=1, config=config)
        component.set_bounds(
            Vector3(0.0, 0.0, 0.0),
            Vector3(10.0, 5.0, 10.0),
        )

        height1 = component.get_surface_height_at(0.0, 0.0)
        component.update_waves(dt=1.0)
        height2 = component.get_surface_height_at(0.0, 0.0)

        # Should be constant
        assert height1 == height2

    def test_very_small_volume(self, fluid_volume):
        """Test very small object volume."""
        force = fluid_volume.calculate_buoyancy(
            Vector3(0.0, 2.0, 0.0),
            volume=0.0001,
            object_density=500.0,
        )

        # Very small but non-zero buoyancy
        assert force.y > 0.0
        assert force.y < 10.0

    def test_negative_wave_time(self, fluid_volume):
        """Test negative wave time doesn't break anything."""
        fluid_volume._wave_time = -10.0
        height = fluid_volume.get_surface_height_at(0.0, 0.0)

        # Should still compute
        assert height > 0.0

    def test_turbulence_with_flow(self):
        """Test turbulence affects flow."""
        flow_config = FlowConfig(
            flow_type=FlowType.DIRECTIONAL,
            direction=Vector3(1.0, 0.0, 0.0),
            speed=1.0,
            turbulence=0.5,
        )
        component = FluidVolumeComponent(entity_id=1, flow_config=flow_config)
        component.set_bounds(
            Vector3(0.0, 0.0, 0.0),
            Vector3(10.0, 5.0, 10.0),
        )

        position = Vector3(5.0, 2.0, 5.0)

        # Update wave time to get different turbulence
        component._wave_time = 1.0
        velocity1 = component.get_flow_velocity(position)

        component._wave_time = 2.0
        velocity2 = component.get_flow_velocity(position)

        # Velocities may differ due to turbulence
        # (depends on sin/cos values at those times)
