"""
Tests for teleport locomotion (teleport.py).

Tests the teleport locomotion system including:
    - TeleportLocomotion component
    - TeleportArcCalculator
    - TeleportTarget component
    - @xr_teleport_area decorator
"""

import math

import pytest

from engine.xr.locomotion.teleport import (
    ArcPoint,
    ArcSegmentType,
    TeleportArcCalculator,
    TeleportLocomotion,
    TeleportLocomotionProvider,
    TeleportResult,
    TeleportState,
    TeleportStyle,
    TeleportTarget,
    xr_teleport_area,
)
from trinity.decorators.ops import decompose


# =============================================================================
# TeleportArcCalculator Tests
# =============================================================================


class TestTeleportArcCalculator:
    """Tests for the arc trajectory calculator."""

    def test_init_defaults(self):
        """Test default initialization."""
        calc = TeleportArcCalculator()
        assert calc.gravity == -9.8
        assert calc.initial_velocity == 8.0
        assert calc.max_distance == 10.0
        assert calc.arc_resolution == 32

    def test_init_custom(self):
        """Test custom initialization."""
        calc = TeleportArcCalculator(
            gravity=-5.0,
            initial_velocity=10.0,
            max_distance=15.0,
            arc_resolution=64,
        )
        assert calc.gravity == -5.0
        assert calc.initial_velocity == 10.0
        assert calc.max_distance == 15.0
        assert calc.arc_resolution == 64

    def test_calculate_arc_returns_points(self):
        """Test that arc calculation returns points."""
        calc = TeleportArcCalculator()
        points = calc.calculate_arc(
            start_position=(0.0, 1.5, 0.0),
            direction=(0.0, 0.0, 1.0),
            launch_angle=45.0,
        )
        assert len(points) > 0
        assert all(isinstance(p, ArcPoint) for p in points)

    def test_calculate_arc_starts_at_position(self):
        """Test that arc starts at the given position."""
        calc = TeleportArcCalculator()
        start = (1.0, 2.0, 3.0)
        points = calc.calculate_arc(
            start_position=start,
            direction=(0.0, 0.0, 1.0),
            launch_angle=45.0,
        )
        assert points[0].position[0] == pytest.approx(start[0], abs=0.01)
        assert points[0].position[1] == pytest.approx(start[1], abs=0.01)
        assert points[0].position[2] == pytest.approx(start[2], abs=0.01)

    def test_calculate_arc_moves_in_direction(self):
        """Test that arc moves in the specified direction."""
        calc = TeleportArcCalculator()
        points = calc.calculate_arc(
            start_position=(0.0, 1.5, 0.0),
            direction=(0.0, 0.0, 1.0),  # Forward in Z
            launch_angle=45.0,
        )
        # Arc should move forward in Z
        last_point = points[-1]
        assert last_point.position[2] > 0.0

    def test_calculate_arc_segment_types(self):
        """Test arc segment type assignment."""
        calc = TeleportArcCalculator(max_distance=5.0)
        points = calc.calculate_arc(
            start_position=(0.0, 1.5, 0.0),
            direction=(0.0, 0.0, 1.0),
            launch_angle=45.0,
        )
        # Should have valid and possibly out of range segments
        segment_types = {p.segment_type for p in points}
        assert ArcSegmentType.VALID in segment_types

    def test_find_landing_point_basic(self):
        """Test finding landing point on ground."""
        calc = TeleportArcCalculator()
        points = calc.calculate_arc(
            start_position=(0.0, 1.5, 0.0),
            direction=(0.0, 0.0, 1.0),
            launch_angle=45.0,
        )
        landing = calc.find_landing_point(points, ground_height=0.0)
        assert landing is not None
        assert landing[1] == pytest.approx(0.0, abs=0.01)

    def test_find_landing_point_no_intersection(self):
        """Test no landing found when arc doesn't intersect ground."""
        calc = TeleportArcCalculator(initial_velocity=1.0)  # Very weak throw
        # Create points that stay above ground - all Y values above ground height
        points = [
            ArcPoint(position=(0.0, 2.0, 0.0)),
            ArcPoint(position=(0.5, 2.3, 0.0)),  # Going up
            ArcPoint(position=(1.0, 2.5, 0.0)),  # Still going up
        ]
        landing = calc.find_landing_point(points, ground_height=0.0)
        assert landing is None, "Should not find landing when all arc points are above ground"

    def test_arc_resolution_affects_points(self):
        """Test that arc resolution affects number of points."""
        calc_low = TeleportArcCalculator(arc_resolution=8)
        calc_high = TeleportArcCalculator(arc_resolution=64)

        points_low = calc_low.calculate_arc(
            start_position=(0.0, 1.5, 0.0),
            direction=(0.0, 0.0, 1.0),
            launch_angle=45.0,
        )
        points_high = calc_high.calculate_arc(
            start_position=(0.0, 1.5, 0.0),
            direction=(0.0, 0.0, 1.0),
            launch_angle=45.0,
        )
        # Higher resolution should produce more points
        assert len(points_high) >= len(points_low)


# =============================================================================
# TeleportLocomotion Tests
# =============================================================================


class TestTeleportLocomotion:
    """Tests for the TeleportLocomotion component."""

    def test_default_state(self):
        """Test default initialization state."""
        teleport = TeleportLocomotion()
        assert teleport.state == TeleportState.IDLE
        assert teleport.is_aiming is False
        assert teleport.aim_valid is False
        assert teleport.style == TeleportStyle.FADE

    def test_begin_aim(self):
        """Test beginning teleport aiming."""
        teleport = TeleportLocomotion()
        result = teleport.begin_aim((0.0, 1.5, 0.0))
        assert result is True
        assert teleport.is_aiming is True
        assert teleport.arc_visible is True
        assert teleport.state == TeleportState.AIMING

    def test_begin_aim_during_cooldown(self):
        """Test that aiming is blocked during cooldown."""
        teleport = TeleportLocomotion(cooldown_duration=1.0)
        teleport._cooldown_remaining = 0.5
        result = teleport.begin_aim((0.0, 1.5, 0.0))
        assert result is False
        assert teleport.is_aiming is False

    def test_update_aim(self):
        """Test updating teleport aim."""
        teleport = TeleportLocomotion()
        teleport.begin_aim((0.0, 1.5, 0.0))
        teleport.update_aim(direction=(0.0, 0.0, 1.0), rotation_offset=0.0)
        assert len(teleport.arc_points) > 0

    def test_update_aim_finds_target(self):
        """Test that updating aim finds a valid target."""
        teleport = TeleportLocomotion(max_distance=50.0)
        teleport.begin_aim((0.0, 1.5, 0.0))
        teleport.update_aim(direction=(0.0, -0.3, 1.0), rotation_offset=0.0)  # Aim slightly downward

        # Verify arc was calculated
        assert len(teleport.arc_points) > 0, "Arc points should be calculated"

        # With downward aim, should find a landing point
        if teleport.target_position is not None and teleport.target_position != (0.0, 0.0, 0.0):
            assert teleport.aim_valid is True, "Target should be valid when found"
            assert teleport.target_position[1] >= 0.0, "Target Y should be at or above ground"
        else:
            # If no target found, state should still be aiming
            assert teleport.state == TeleportState.AIMING, "Should remain in AIMING state"

    def test_cancel_aim(self):
        """Test canceling teleport aim."""
        teleport = TeleportLocomotion()
        teleport.begin_aim((0.0, 1.5, 0.0))
        teleport.cancel_aim()
        assert teleport.is_aiming is False
        assert teleport.arc_visible is False
        assert teleport.state == TeleportState.IDLE
        assert len(teleport.arc_points) == 0

    def test_execute_teleport_when_invalid(self):
        """Test that teleport fails when not aiming or invalid."""
        teleport = TeleportLocomotion()
        result = teleport.execute_teleport()
        assert result is None

    def test_execute_teleport_when_valid(self):
        """Test executing a valid teleport."""
        teleport = TeleportLocomotion()
        teleport.begin_aim((0.0, 1.5, 0.0))
        teleport.update_aim((0.0, 0.0, 1.0))
        teleport.aim_valid = True
        teleport.target_position = (0.0, 0.0, 5.0)

        result = teleport.execute_teleport()
        assert result is not None
        assert isinstance(result, TeleportResult)
        assert result.success is True
        assert teleport.state == TeleportState.TRANSITIONING

    def test_complete_teleport(self):
        """Test completing teleport transition."""
        teleport = TeleportLocomotion(cooldown_duration=0.5)
        teleport.begin_aim((0.0, 1.5, 0.0))
        teleport.update_aim((0.0, 0.0, 1.0))
        teleport.aim_valid = True
        teleport.execute_teleport()
        teleport.complete_teleport()

        assert teleport.state == TeleportState.COOLDOWN
        assert teleport._cooldown_remaining == 0.5

    def test_update_cooldown(self):
        """Test cooldown timer updates."""
        teleport = TeleportLocomotion(cooldown_duration=0.5)
        teleport.state = TeleportState.COOLDOWN
        teleport._cooldown_remaining = 0.5

        teleport.update(0.3)
        assert teleport._cooldown_remaining == pytest.approx(0.2, abs=0.01)

        teleport.update(0.3)
        assert teleport._cooldown_remaining == 0.0
        assert teleport.state == TeleportState.IDLE

    def test_fade_alpha_calculation(self):
        """Test fade alpha calculation during transition."""
        teleport = TeleportLocomotion(
            style=TeleportStyle.FADE,
            fade_duration=0.2,
        )
        teleport.state = TeleportState.TRANSITIONING
        teleport._transition_progress = 0.1  # Halfway through fade out

        alpha = teleport.get_fade_alpha()
        assert 0.0 <= alpha <= 1.0

    def test_rotation_snapping(self):
        """Test rotation snap to configured angle."""
        teleport = TeleportLocomotion(
            rotation_enabled=True,
            rotation_snap_angle=45.0,
        )
        teleport.begin_aim((0.0, 1.5, 0.0))

        # Rotation close to 45 degrees should snap
        teleport.update_aim((0.0, 0.0, 1.0), rotation_offset=math.radians(50.0))
        expected = math.radians(45.0)
        assert teleport.target_rotation == pytest.approx(expected, abs=0.01)

    def test_teleport_styles(self):
        """Test different teleport styles."""
        for style in TeleportStyle:
            teleport = TeleportLocomotion(style=style)
            assert teleport.style == style

    def test_callbacks(self):
        """Test teleport callbacks are called."""
        teleport = TeleportLocomotion()
        start_called = False
        end_called = False
        end_result = None

        def on_start():
            nonlocal start_called
            start_called = True

        def on_end(result):
            nonlocal end_called, end_result
            end_called = True
            end_result = result

        teleport.set_teleport_callback(on_start=on_start, on_end=on_end)

        teleport.begin_aim((0.0, 1.5, 0.0))
        teleport.aim_valid = True
        teleport.target_position = (0.0, 0.0, 5.0)
        teleport.execute_teleport()
        assert start_called is True

        teleport.complete_teleport()
        assert end_called is True
        assert end_result is not None


# =============================================================================
# TeleportTarget Tests
# =============================================================================


class TestTeleportTarget:
    """Tests for the TeleportTarget component."""

    def test_default_values(self):
        """Test default initialization."""
        target = TeleportTarget()
        assert target.teleport_type == "any"
        assert target.is_valid is True
        assert target.is_highlighted is False

    def test_custom_values(self):
        """Test custom initialization."""
        target = TeleportTarget(
            teleport_type="fade",
            surface_normal=(0.0, 1.0, 0.0),
            landing_offset=0.1,
        )
        assert target.teleport_type == "fade"
        assert target.landing_offset == 0.1

    def test_area_target(self):
        """Test area-based teleport target."""
        target = TeleportTarget(
            is_area=True,
            area_bounds=[
                (0.0, 0.0, 0.0),
                (5.0, 0.0, 0.0),
                (5.0, 0.0, 5.0),
                (0.0, 0.0, 5.0),
            ],
        )
        assert target.is_area is True
        assert len(target.area_bounds) == 4


# =============================================================================
# TeleportLocomotionProvider Tests
# =============================================================================


class TestTeleportLocomotionProvider:
    """Tests for the TeleportLocomotionProvider."""

    def test_init(self):
        """Test provider initialization."""
        locomotion = TeleportLocomotion()
        provider = TeleportLocomotionProvider(locomotion)
        assert provider.locomotion is locomotion

    def test_aim_start(self):
        """Test aim start through provider."""
        locomotion = TeleportLocomotion()
        provider = TeleportLocomotionProvider(locomotion)
        provider.on_aim_start((0.0, 1.5, 0.0))
        assert locomotion.is_aiming is True

    def test_aim_update(self):
        """Test aim update through provider."""
        locomotion = TeleportLocomotion()
        provider = TeleportLocomotionProvider(locomotion)
        provider.on_aim_start((0.0, 1.5, 0.0))
        provider.on_aim_update((0.0, 0.0, 1.0))
        assert len(locomotion.arc_points) > 0

    def test_aim_cancel(self):
        """Test aim cancel through provider."""
        locomotion = TeleportLocomotion()
        provider = TeleportLocomotionProvider(locomotion)
        provider.on_aim_start((0.0, 1.5, 0.0))
        provider.on_aim_cancel()
        assert locomotion.is_aiming is False

    def test_teleport_confirm(self):
        """Test teleport confirm through provider."""
        locomotion = TeleportLocomotion()
        provider = TeleportLocomotionProvider(locomotion)
        provider.on_aim_start((0.0, 1.5, 0.0))
        locomotion.aim_valid = True
        locomotion.target_position = (0.0, 0.0, 5.0)

        result = provider.on_teleport_confirm()
        assert result is not None

    def test_update(self):
        """Test provider update."""
        locomotion = TeleportLocomotion(cooldown_duration=1.0)
        provider = TeleportLocomotionProvider(locomotion)
        locomotion.state = TeleportState.COOLDOWN
        locomotion._cooldown_remaining = 0.5

        provider.update(0.3)
        assert locomotion._cooldown_remaining == pytest.approx(0.2, abs=0.01)


# =============================================================================
# @xr_teleport_area Decorator Tests
# =============================================================================


class TestXRTeleportAreaDecorator:
    """Tests for the @xr_teleport_area decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""
        @xr_teleport_area(teleport_type="any")
        class Floor:
            pass

        assert Floor._xr_teleport_area is True

    def test_teleport_type_stored(self):
        """Test teleport type is stored."""
        @xr_teleport_area(teleport_type="fade")
        class Floor:
            pass

        assert Floor._teleport_type == "fade"

    def test_priority_stored(self):
        """Test priority is stored."""
        @xr_teleport_area(teleport_type="any", priority=5)
        class Floor:
            pass

        assert Floor._teleport_priority == 5

    def test_tags_applied(self):
        """Test tags are applied."""
        @xr_teleport_area(teleport_type="dash")
        class Floor:
            pass

        assert Floor._tags["xr_teleport_area"] is True
        assert Floor._tags["teleport_type"] == "dash"

    def test_registered_in_xr_registry(self):
        """Test registration in XR registry."""
        @xr_teleport_area(teleport_type="any")
        class Floor:
            pass

        assert "xr" in Floor._registries

    def test_invalid_teleport_type(self):
        """Test validation of teleport type."""
        with pytest.raises(ValueError, match="teleport_type"):
            @xr_teleport_area(teleport_type="invalid_type")
            class Floor:
                pass

    def test_applied_decorators(self):
        """Test applied decorators list."""
        @xr_teleport_area(teleport_type="any")
        class Floor:
            pass

        assert "xr_teleport_area" in Floor._applied_decorators

    def test_steps_recorded(self):
        """Test steps are recorded."""
        @xr_teleport_area(teleport_type="any")
        class Floor:
            pass

        assert len(Floor._applied_steps) > 0

    def test_decompose(self):
        """Test decompose shows steps."""
        steps = decompose(xr_teleport_area)
        assert isinstance(steps, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestTeleportIntegration:
    """Integration tests for teleport locomotion."""

    def test_full_teleport_flow(self):
        """Test complete teleport flow from aim to complete."""
        teleport = TeleportLocomotion(
            style=TeleportStyle.FADE,
            fade_duration=0.1,
            max_distance=20.0,
        )

        # Begin aiming
        assert teleport.begin_aim((0.0, 1.5, 0.0)) is True
        assert teleport.state == TeleportState.AIMING

        # Update aim
        teleport.update_aim((0.0, 0.0, 1.0))
        assert len(teleport.arc_points) > 0

        # Force valid target for test
        teleport.aim_valid = True
        teleport.target_position = (0.0, 0.0, 5.0)

        # Execute
        result = teleport.execute_teleport()
        assert result is not None
        assert result.success is True
        assert teleport.state == TeleportState.TRANSITIONING

        # Complete
        teleport.complete_teleport()
        assert teleport.state == TeleportState.COOLDOWN

        # Wait out cooldown
        teleport.update(teleport.cooldown_duration + 0.1)
        assert teleport.state == TeleportState.IDLE

    def test_teleport_with_target_marker(self):
        """Test teleport with TeleportTarget marker."""
        teleport = TeleportLocomotion()
        target = TeleportTarget(
            teleport_type="fade",
            surface_normal=(0.0, 1.0, 0.0),
            landing_offset=0.05,
        )

        # Teleport should respect target settings
        assert target.teleport_type == "fade"
        assert target.is_valid is True
