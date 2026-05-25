"""Tests for XR guardian/boundary system.

Tests cover:
- GuardianMode enumeration
- BoundaryType enumeration
- ProximityLevel enumeration
- PlayAreaBounds calculations
- GuardianSystem proximity detection
- Platform-specific guardian implementations
"""

import pytest
import math
from unittest.mock import MagicMock, patch

from engine.xr.platform.guardian import (
    GuardianMode,
    BoundaryType,
    ProximityLevel,
    BoundaryVertex,
    PlayAreaBounds,
    GuardianConfig,
    ProximityInfo,
    GuardianSystem,
    OpenXRGuardian,
    SteamVRGuardian,
    QuestGuardian,
    create_guardian_system,
)


class TestGuardianEnums:
    """Tests for guardian enumerations."""

    def test_guardian_modes(self) -> None:
        """Test guardian mode enumeration."""
        assert GuardianMode.DISABLED
        assert GuardianMode.STATIONARY
        assert GuardianMode.ROOM_SCALE
        assert GuardianMode.CUSTOM
        assert GuardianMode.PASS_THROUGH

    def test_boundary_types(self) -> None:
        """Test boundary type enumeration."""
        assert BoundaryType.RECTANGLE
        assert BoundaryType.POLYGON
        assert BoundaryType.CYLINDER
        assert BoundaryType.CUSTOM_MESH

    def test_proximity_levels(self) -> None:
        """Test proximity level enumeration."""
        assert ProximityLevel.SAFE
        assert ProximityLevel.APPROACHING
        assert ProximityLevel.NEAR
        assert ProximityLevel.AT_BOUNDARY
        assert ProximityLevel.OUTSIDE


class TestPlayAreaBounds:
    """Tests for PlayAreaBounds calculations."""

    def test_default_bounds(self) -> None:
        """Test default bounds creation."""
        bounds = PlayAreaBounds()

        assert bounds.boundary_type == BoundaryType.POLYGON
        assert bounds.width == 2.0
        assert bounds.depth == 2.0
        assert bounds.height == 2.5
        assert bounds.center_x == 0.0
        assert bounds.center_z == 0.0

    def test_rectangular_area(self) -> None:
        """Test rectangular area calculation."""
        bounds = PlayAreaBounds(
            boundary_type=BoundaryType.RECTANGLE,
            width=3.0,
            depth=4.0,
        )

        assert bounds.get_area() == 12.0

    def test_cylinder_area(self) -> None:
        """Test cylindrical area calculation."""
        bounds = PlayAreaBounds(
            boundary_type=BoundaryType.CYLINDER,
            width=2.0,  # Diameter
        )

        expected_area = math.pi * 1.0 ** 2  # pi * radius^2
        assert abs(bounds.get_area() - expected_area) < 0.001

    def test_polygon_area(self) -> None:
        """Test polygon area calculation using shoelace formula."""
        # Create a 2x2 square polygon
        bounds = PlayAreaBounds(
            boundary_type=BoundaryType.POLYGON,
            vertices=[
                BoundaryVertex(-1, 0, -1),
                BoundaryVertex(1, 0, -1),
                BoundaryVertex(1, 0, 1),
                BoundaryVertex(-1, 0, 1),
            ]
        )

        assert abs(bounds.get_area() - 4.0) < 0.001

    def test_contains_point_rectangle(self) -> None:
        """Test point containment for rectangular bounds."""
        bounds = PlayAreaBounds(
            boundary_type=BoundaryType.RECTANGLE,
            width=2.0,
            depth=2.0,
            center_x=0.0,
            center_z=0.0,
        )

        # Point at center
        assert bounds.contains_point(0.0, 0.0) is True

        # Point inside
        assert bounds.contains_point(0.5, 0.5) is True

        # Point outside
        assert bounds.contains_point(2.0, 0.0) is False
        assert bounds.contains_point(0.0, 2.0) is False

    def test_contains_point_cylinder(self) -> None:
        """Test point containment for cylindrical bounds."""
        bounds = PlayAreaBounds(
            boundary_type=BoundaryType.CYLINDER,
            width=2.0,  # Diameter = 2, radius = 1
            center_x=0.0,
            center_z=0.0,
        )

        # Point at center
        assert bounds.contains_point(0.0, 0.0) is True

        # Point inside (at radius 0.5)
        assert bounds.contains_point(0.5, 0.0) is True

        # Point at edge (radius = 1)
        assert bounds.contains_point(1.0, 0.0) is True

        # Point outside (radius > 1)
        assert bounds.contains_point(1.5, 0.0) is False

    def test_contains_point_polygon(self) -> None:
        """Test point containment for polygon bounds."""
        bounds = PlayAreaBounds(
            boundary_type=BoundaryType.POLYGON,
            vertices=[
                BoundaryVertex(-1, 0, -1),
                BoundaryVertex(1, 0, -1),
                BoundaryVertex(1, 0, 1),
                BoundaryVertex(-1, 0, 1),
            ]
        )

        # Point inside
        assert bounds.contains_point(0.0, 0.0) is True
        assert bounds.contains_point(0.5, 0.5) is True

        # Point outside
        assert bounds.contains_point(2.0, 0.0) is False


class TestGuardianConfig:
    """Tests for GuardianConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GuardianConfig()

        assert config.mode == GuardianMode.ROOM_SCALE
        assert config.approaching_distance == 1.0
        assert config.near_distance == 0.5
        assert config.boundary_distance == 0.3  # From XR_CONFIG.platform.GUARDIAN_BOUNDARY_DISTANCE_M
        assert config.wall_height == 2.5
        assert config.passthrough_on_proximity is True
        assert config.audio_warning_enabled is True
        assert config.haptic_warning_enabled is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = GuardianConfig(
            mode=GuardianMode.STATIONARY,
            approaching_distance=0.5,
            near_distance=0.25,
            passthrough_on_proximity=False,
        )

        assert config.mode == GuardianMode.STATIONARY
        assert config.approaching_distance == 0.5
        assert config.near_distance == 0.25
        assert config.passthrough_on_proximity is False


class TestOpenXRGuardian:
    """Tests for OpenXR guardian implementation."""

    def test_initialization(self) -> None:
        """Test guardian initialization."""
        guardian = OpenXRGuardian()
        result = guardian.initialize()

        assert result is True
        assert guardian.bounds is not None

    def test_request_bounds(self) -> None:
        """Test bounds request."""
        guardian = OpenXRGuardian()
        bounds = guardian.request_bounds()

        assert bounds is not None
        assert isinstance(bounds, PlayAreaBounds)
        assert bounds.boundary_type == BoundaryType.RECTANGLE

    def test_recenter(self) -> None:
        """Test recentering."""
        guardian = OpenXRGuardian()
        guardian.initialize()
        result = guardian.recenter()

        assert result is True

    def test_set_custom_bounds_not_supported(self) -> None:
        """Test that custom bounds are not supported in OpenXR."""
        guardian = OpenXRGuardian()
        guardian.initialize()

        custom_bounds = PlayAreaBounds(width=5.0, depth=5.0)
        result = guardian.set_custom_bounds(custom_bounds)

        assert result is False


class TestGuardianProximity:
    """Tests for guardian proximity detection."""

    def test_proximity_safe(self) -> None:
        """Test proximity detection when far from boundary."""
        guardian = OpenXRGuardian()
        guardian.initialize()

        # Position at center, well within bounds
        proximity = guardian.update((0.0, 1.5, 0.0), 0.016)

        assert proximity.level == ProximityLevel.SAFE
        assert guardian.visible is False

    def test_proximity_approaching(self) -> None:
        """Test proximity detection when approaching boundary."""
        config = GuardianConfig(
            approaching_distance=1.0,
            near_distance=0.5,
            boundary_distance=0.1,
        )
        guardian = OpenXRGuardian(config)
        guardian.initialize()

        # Set up rectangular bounds: 2.5m x 2.5m centered at origin
        # Edge is at x = 1.25m, approaching threshold at x = 0.25m (1.25 - 1.0)
        guardian._bounds = PlayAreaBounds(
            boundary_type=BoundaryType.RECTANGLE,
            width=2.5,
            depth=2.5,
            center_x=0.0,
            center_z=0.0,
        )

        # Position at center - should be SAFE
        proximity_center = guardian.update((0.0, 1.5, 0.0), 0.016)
        assert proximity_center.level == ProximityLevel.SAFE, "Center position should be SAFE"

        # Position just inside approaching threshold (edge at 1.25, approaching at 0.25)
        # At x=0.5, distance to edge is 0.75m, which is within approaching_distance of 1.0m
        proximity_approaching = guardian.update((0.5, 1.5, 0.0), 0.016)
        assert proximity_approaching.level == ProximityLevel.APPROACHING, "Near edge should be APPROACHING"

    def test_proximity_callback(self) -> None:
        """Test proximity change callback fires when proximity level changes."""
        config = GuardianConfig(
            approaching_distance=0.5,
            near_distance=0.25,
            boundary_distance=0.1,
        )
        guardian = OpenXRGuardian(config)
        guardian.initialize()

        received_proximity = []

        def on_proximity(info: ProximityInfo):
            received_proximity.append(info)

        guardian.on_proximity_changed(on_proximity)

        # Set up small bounds: 2.0m x 2.0m centered at origin
        # Edge is at x = 1.0m, approaching threshold at x = 0.5m
        guardian._bounds = PlayAreaBounds(
            boundary_type=BoundaryType.RECTANGLE,
            width=2.0,
            depth=2.0,
            center_x=0.0,
            center_z=0.0,
        )

        # Start at center (safe) - establish baseline
        guardian.update((0.0, 1.5, 0.0), 0.016)
        initial_callback_count = len(received_proximity)

        # Move to position that guarantees APPROACHING level
        # At x=0.7, distance to edge (at x=1.0) is 0.3m, within approaching_distance of 0.5m
        guardian.update((0.7, 1.5, 0.0), 0.016)

        # Callback should have been triggered by proximity change
        assert len(received_proximity) > initial_callback_count, \
            "Proximity callback should fire when level changes from SAFE to APPROACHING"

        # Verify the last callback received shows APPROACHING level
        assert received_proximity[-1].level == ProximityLevel.APPROACHING, \
            "Last proximity callback should indicate APPROACHING level"

    def test_boundary_crossed_callback(self) -> None:
        """Test boundary crossing callback fires when exiting play area."""
        config = GuardianConfig(
            approaching_distance=0.3,
            near_distance=0.15,
            boundary_distance=0.05,
        )
        guardian = OpenXRGuardian(config)
        guardian.initialize()

        crossed_events = []

        def on_boundary_crossed(exiting: bool):
            crossed_events.append(exiting)

        guardian.on_boundary_crossed(on_boundary_crossed)

        # Set up small bounds: 1.0m x 1.0m centered at origin
        # Boundary edge is at x = 0.5m
        guardian._bounds = PlayAreaBounds(
            boundary_type=BoundaryType.RECTANGLE,
            width=1.0,
            depth=1.0,
            center_x=0.0,
            center_z=0.0,
        )

        # Start inside at center
        guardian.update((0.0, 1.5, 0.0), 0.016)
        assert len(crossed_events) == 0, "No boundary cross at center"

        # Move clearly outside boundary (edge at 0.5m, we go to 0.8m which is 0.3m outside)
        guardian.update((0.8, 1.5, 0.0), 0.016)

        # Boundary crossing callback should have fired
        assert len(crossed_events) > 0, "Boundary crossed callback should fire when moving outside"
        assert crossed_events[-1] is True, "Callback should indicate exiting (True)"


class TestGuardianPassthrough:
    """Tests for passthrough blending."""

    def test_passthrough_disabled(self) -> None:
        """Test passthrough when disabled."""
        config = GuardianConfig(passthrough_on_proximity=False)
        guardian = OpenXRGuardian(config)

        blend = guardian.get_passthrough_blend()
        assert blend == 0.0

    def test_passthrough_blend_calculation(self) -> None:
        """Test passthrough blend calculation."""
        config = GuardianConfig(
            passthrough_on_proximity=True,
            passthrough_trigger_distance=0.2,
            passthrough_blend_distance=0.3,
        )
        guardian = OpenXRGuardian(config)
        guardian.initialize()

        # At safe distance (no blend)
        guardian._current_proximity = ProximityInfo(
            level=ProximityLevel.SAFE,
            distance=1.0,
        )
        blend = guardian.get_passthrough_blend()
        assert blend == 0.0

        # At trigger distance (full blend)
        guardian._current_proximity = ProximityInfo(
            level=ProximityLevel.AT_BOUNDARY,
            distance=0.15,
        )
        blend = guardian.get_passthrough_blend()
        assert blend == 1.0


class TestGuardianWarning:
    """Tests for visual warning intensity."""

    def test_warning_intensity_safe(self) -> None:
        """Test warning intensity when safe."""
        guardian = OpenXRGuardian()
        guardian.initialize()

        guardian._current_proximity = ProximityInfo(level=ProximityLevel.SAFE)
        intensity = guardian.get_warning_intensity()
        assert intensity == 0.0

    def test_warning_intensity_outside(self) -> None:
        """Test warning intensity when outside bounds."""
        guardian = OpenXRGuardian()
        guardian.initialize()

        guardian._current_proximity = ProximityInfo(level=ProximityLevel.OUTSIDE)
        intensity = guardian.get_warning_intensity()
        assert intensity == 1.0

    def test_warning_intensity_at_boundary(self) -> None:
        """Test warning intensity at boundary."""
        guardian = OpenXRGuardian()
        guardian.initialize()

        guardian._current_proximity = ProximityInfo(
            level=ProximityLevel.AT_BOUNDARY
        )
        intensity = guardian.get_warning_intensity()
        assert intensity == 0.9


class TestQuestGuardian:
    """Tests for Quest guardian implementation."""

    def test_initialization(self) -> None:
        """Test Quest guardian initialization."""
        guardian = QuestGuardian()
        result = guardian.initialize()

        assert result is True
        assert guardian.bounds is not None

    def test_polygon_bounds(self) -> None:
        """Test Quest guardian returns polygon bounds."""
        guardian = QuestGuardian()
        bounds = guardian.request_bounds()

        assert bounds.boundary_type == BoundaryType.POLYGON
        assert len(bounds.vertices) == 4


class TestCreateGuardianSystem:
    """Tests for guardian system factory."""

    def test_create_openxr_guardian(self) -> None:
        """Test creating OpenXR guardian."""
        guardian = create_guardian_system("openxr")
        assert isinstance(guardian, OpenXRGuardian)

    def test_create_steamvr_guardian(self) -> None:
        """Test creating SteamVR guardian."""
        guardian = create_guardian_system("steamvr")
        assert isinstance(guardian, SteamVRGuardian)

    def test_create_quest_guardian(self) -> None:
        """Test creating Quest guardian."""
        guardian = create_guardian_system("quest")
        assert isinstance(guardian, QuestGuardian)

    def test_default_is_openxr(self) -> None:
        """Test that default guardian is OpenXR."""
        guardian = create_guardian_system("unknown")
        assert isinstance(guardian, OpenXRGuardian)
