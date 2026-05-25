"""
Tests for the measurements module.

Tests accuracy and unit conversion.
"""

import math
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.leveleditor.measurements import (
    MeasurementUnit,
    MeasurementType,
    MeasurementTool,
    MeasurementResult,
    MeasurementPoint,
    DistanceMeasurement,
    AngleMeasurement,
    AreaMeasurement,
    AngleUnit,
)
from engine.tooling.leveleditor.placement import Vector3
from foundation.tracker import tracker


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset tracker state before each test."""
    tracker._dirty.clear()
    tracker._cb_global.clear()
    tracker._cb_type.clear()
    tracker._cb_obj.clear()
    tracker._undo.clear()
    tracker._redo.clear()
    tracker._txn = None
    yield


class TestMeasurementUnit:
    """Tests for MeasurementUnit enum and conversions."""

    def test_unit_symbols(self):
        """Units should have correct symbols."""
        assert MeasurementUnit.METERS.symbol == "m"
        assert MeasurementUnit.CENTIMETERS.symbol == "cm"
        assert MeasurementUnit.FEET.symbol == "ft"
        assert MeasurementUnit.INCHES.symbol == "in"

    def test_meters_to_centimeters(self):
        """Should convert meters to centimeters."""
        result = MeasurementUnit.convert(1.0, MeasurementUnit.METERS, MeasurementUnit.CENTIMETERS)
        assert abs(result - 100.0) < 0.0001

    def test_meters_to_feet(self):
        """Should convert meters to feet."""
        result = MeasurementUnit.convert(1.0, MeasurementUnit.METERS, MeasurementUnit.FEET)
        assert abs(result - 3.28084) < 0.001

    def test_meters_to_inches(self):
        """Should convert meters to inches."""
        result = MeasurementUnit.convert(1.0, MeasurementUnit.METERS, MeasurementUnit.INCHES)
        assert abs(result - 39.3701) < 0.001

    def test_feet_to_meters(self):
        """Should convert feet to meters."""
        result = MeasurementUnit.convert(1.0, MeasurementUnit.FEET, MeasurementUnit.METERS)
        assert abs(result - 0.3048) < 0.0001

    def test_kilometers_to_miles(self):
        """Should convert kilometers to miles."""
        result = MeasurementUnit.convert(1.0, MeasurementUnit.KILOMETERS, MeasurementUnit.MILES)
        assert abs(result - 0.621371) < 0.001

    def test_round_trip_conversion(self):
        """Round trip conversion should preserve value."""
        original = 42.5
        meters = MeasurementUnit.convert(original, MeasurementUnit.FEET, MeasurementUnit.METERS)
        back = MeasurementUnit.convert(meters, MeasurementUnit.METERS, MeasurementUnit.FEET)
        assert abs(back - original) < 0.0001

    def test_same_unit_conversion(self):
        """Converting to same unit should return same value."""
        result = MeasurementUnit.convert(5.0, MeasurementUnit.METERS, MeasurementUnit.METERS)
        assert result == 5.0

    def test_yards_to_feet(self):
        """Should convert yards to feet."""
        result = MeasurementUnit.convert(1.0, MeasurementUnit.YARDS, MeasurementUnit.FEET)
        assert abs(result - 3.0) < 0.0001

    def test_millimeters_to_meters(self):
        """Should convert millimeters to meters."""
        result = MeasurementUnit.convert(1000.0, MeasurementUnit.MILLIMETERS, MeasurementUnit.METERS)
        assert abs(result - 1.0) < 0.0001


class TestMeasurementPoint:
    """Tests for MeasurementPoint dataclass."""

    def test_creation(self):
        """Point should initialize with position."""
        point = MeasurementPoint(Vector3(10, 20, 30), "Test")
        assert point.position.x == 10
        assert point.label == "Test"

    def test_default_values(self):
        """Point should have default values."""
        point = MeasurementPoint(Vector3(0, 0, 0))
        assert point.label == ""
        assert point.snapped is False
        assert point.surface_normal is None

    def test_with_surface_normal(self):
        """Point should store surface normal."""
        normal = Vector3(0, 1, 0)
        point = MeasurementPoint(Vector3(0, 0, 0), surface_normal=normal)
        assert point.surface_normal is not None
        assert point.surface_normal.y == 1


class TestDistanceMeasurement:
    """Tests for DistanceMeasurement class."""

    def test_creation(self):
        """Should initialize with default unit."""
        measure = DistanceMeasurement()
        assert measure.unit == MeasurementUnit.METERS
        assert len(measure.points) == 0

    def test_unique_id(self):
        """Each measurement should have unique ID."""
        m1 = DistanceMeasurement()
        m2 = DistanceMeasurement()
        assert m1.id != m2.id

    def test_add_point(self):
        """Should add measurement point."""
        measure = DistanceMeasurement()
        point = measure.add_point(Vector3(10, 0, 0), "Start")

        assert point.label == "Start"
        assert len(measure.points) == 1

    def test_add_point_auto_label(self):
        """Should auto-generate label if not provided."""
        measure = DistanceMeasurement()
        point = measure.add_point(Vector3(0, 0, 0))

        assert point.label == "P1"

    def test_remove_point(self):
        """Should remove point by index."""
        measure = DistanceMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(10, 0, 0))

        result = measure.remove_point(0)

        assert result is True
        assert len(measure.points) == 1

    def test_remove_invalid_index(self):
        """Should return False for invalid index."""
        measure = DistanceMeasurement()
        result = measure.remove_point(0)
        assert result is False

    def test_clear_points(self):
        """Should clear all points."""
        measure = DistanceMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(10, 0, 0))

        measure.clear_points()

        assert len(measure.points) == 0

    def test_calculate_distance(self):
        """Should calculate distance between points."""
        measure = DistanceMeasurement()
        dist = measure.calculate_distance(Vector3(0, 0, 0), Vector3(3, 4, 0))
        assert abs(dist - 5.0) < 0.0001

    def test_calculate_distance_3d(self):
        """Should calculate 3D distance."""
        measure = DistanceMeasurement()
        dist = measure.calculate_distance(Vector3(0, 0, 0), Vector3(1, 2, 2))
        assert abs(dist - 3.0) < 0.0001

    def test_measure_simple(self):
        """Should measure distance between two points."""
        measure = DistanceMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(10, 0, 0))

        result = measure.measure()

        assert result is not None
        assert result.measurement_type == MeasurementType.DISTANCE
        assert abs(result.value - 10.0) < 0.0001
        assert result.unit == MeasurementUnit.METERS

    def test_measure_with_unit(self):
        """Should measure in specified unit."""
        measure = DistanceMeasurement(MeasurementUnit.CENTIMETERS)
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(1, 0, 0))

        result = measure.measure()

        assert abs(result.value - 100.0) < 0.0001

    def test_measure_insufficient_points(self):
        """Should return None with less than 2 points."""
        measure = DistanceMeasurement()
        measure.add_point(Vector3(0, 0, 0))

        result = measure.measure()

        assert result is None

    def test_accumulated_distance(self):
        """Should measure cumulative distance along path."""
        measure = DistanceMeasurement()
        measure.accumulated = True
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(10, 0, 0))
        measure.add_point(Vector3(10, 10, 0))

        result = measure.measure()

        assert abs(result.value - 20.0) < 0.0001

    def test_non_accumulated_distance(self):
        """Non-accumulated should measure first to last."""
        measure = DistanceMeasurement()
        measure.accumulated = False
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(10, 0, 0))
        measure.add_point(Vector3(10, 10, 0))

        result = measure.measure()

        # sqrt(10^2 + 10^2) = 14.14...
        expected = math.sqrt(200)
        assert abs(result.value - expected) < 0.0001

    def test_axis_distances(self):
        """Should calculate axis-aligned distances."""
        measure = DistanceMeasurement()
        measure.show_axis_distances = True
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(3, 4, 5))

        result = measure.measure()

        assert "x" in result.auxiliary_values
        assert abs(result.auxiliary_values["x"] - 3.0) < 0.0001
        assert abs(result.auxiliary_values["y"] - 4.0) < 0.0001
        assert abs(result.auxiliary_values["z"] - 5.0) < 0.0001

    def test_segment_distances(self):
        """Should return individual segment distances."""
        measure = DistanceMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(10, 0, 0))
        measure.add_point(Vector3(10, 5, 0))

        segments = measure.get_segment_distances()

        assert len(segments) == 2
        assert abs(segments[0] - 10.0) < 0.0001
        assert abs(segments[1] - 5.0) < 0.0001

    def test_formatted_result(self):
        """Result should include formatted string."""
        measure = DistanceMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(5, 0, 0))

        result = measure.measure()

        assert "5.000" in result.formatted
        assert "m" in result.formatted


class TestAngleMeasurement:
    """Tests for AngleMeasurement class."""

    def test_creation(self):
        """Should initialize with default angle unit."""
        measure = AngleMeasurement()
        assert measure.angle_unit == AngleUnit.DEGREES
        assert len(measure.points) == 0

    def test_unique_id(self):
        """Each measurement should have unique ID."""
        m1 = AngleMeasurement()
        m2 = AngleMeasurement()
        assert m1.id != m2.id

    def test_add_point(self):
        """Should add measurement point."""
        measure = AngleMeasurement()
        point = measure.add_point(Vector3(0, 0, 0), "Vertex")

        assert point.label == "Vertex"
        assert len(measure.points) == 1

    def test_add_point_auto_label(self):
        """Should auto-generate label."""
        measure = AngleMeasurement()
        point = measure.add_point(Vector3(0, 0, 0))

        assert point.label == "A1"

    def test_measure_right_angle(self):
        """Should measure 90 degree angle."""
        measure = AngleMeasurement()
        measure.add_point(Vector3(1, 0, 0))   # P1
        measure.add_point(Vector3(0, 0, 0))   # Vertex
        measure.add_point(Vector3(0, 1, 0))   # P2

        result = measure.measure_three_point()

        assert result is not None
        assert abs(result.value - 90.0) < 0.01

    def test_measure_straight_angle(self):
        """Should measure 180 degree angle."""
        measure = AngleMeasurement()
        measure.add_point(Vector3(-1, 0, 0))  # P1
        measure.add_point(Vector3(0, 0, 0))   # Vertex
        measure.add_point(Vector3(1, 0, 0))   # P2

        result = measure.measure_three_point()

        assert abs(result.value - 180.0) < 0.01

    def test_measure_acute_angle(self):
        """Should measure acute angle (45 degrees)."""
        measure = AngleMeasurement()
        measure.add_point(Vector3(1, 0, 0))   # P1
        measure.add_point(Vector3(0, 0, 0))   # Vertex
        measure.add_point(Vector3(1, 1, 0))   # P2

        result = measure.measure_three_point()

        assert abs(result.value - 45.0) < 0.01

    def test_measure_in_radians(self):
        """Should measure angle in radians."""
        measure = AngleMeasurement(AngleUnit.RADIANS)
        measure.add_point(Vector3(1, 0, 0))
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(0, 1, 0))

        result = measure.measure_three_point()

        assert abs(result.value - math.pi/2) < 0.001

    def test_measure_in_gradians(self):
        """Should measure angle in gradians."""
        measure = AngleMeasurement(AngleUnit.GRADIANS)
        measure.add_point(Vector3(1, 0, 0))
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(0, 1, 0))

        result = measure.measure_three_point()

        # 90 degrees = 100 gradians
        assert abs(result.value - 100.0) < 0.01

    def test_measure_insufficient_points(self):
        """Should return None with less than 3 points."""
        measure = AngleMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(1, 0, 0))

        result = measure.measure_three_point()

        assert result is None

    def test_auxiliary_values(self):
        """Should include auxiliary angle values."""
        measure = AngleMeasurement()
        measure.add_point(Vector3(1, 0, 0))
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(0, 1, 0))

        result = measure.measure_three_point()

        assert "radians" in result.auxiliary_values
        assert "degrees" in result.auxiliary_values
        assert abs(result.auxiliary_values["degrees"] - 90.0) < 0.01

    def test_surface_angle_measurement(self):
        """Should measure angle between surface normals."""
        measure = AngleMeasurement()
        measure.add_point(Vector3(0, 0, 0), surface_normal=Vector3(0, 1, 0))
        measure.add_point(Vector3(1, 0, 0), surface_normal=Vector3(1, 0, 0))

        result = measure.measure_surface_angle()

        assert result is not None
        assert abs(result.value - 90.0) < 0.01

    def test_surface_angle_no_normals(self):
        """Should return None if no surface normals."""
        measure = AngleMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(1, 0, 0))

        result = measure.measure_surface_angle()

        assert result is None

    def test_reference_direction_measurement(self):
        """Should measure angle to reference direction."""
        measure = AngleMeasurement()
        measure.reference_direction = Vector3(1, 0, 0)  # X axis
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(1, 1, 0))  # 45 degrees from X

        result = measure.measure_to_reference()

        assert result is not None
        assert abs(result.value - 45.0) < 0.01

    def test_reference_no_direction(self):
        """Should return None if no reference direction."""
        measure = AngleMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(1, 0, 0))

        result = measure.measure_to_reference()

        assert result is None

    def test_clear_points(self):
        """Should clear all points."""
        measure = AngleMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(1, 0, 0))

        measure.clear_points()

        assert len(measure.points) == 0


class TestAreaMeasurement:
    """Tests for AreaMeasurement class."""

    def test_creation(self):
        """Should initialize with default unit."""
        measure = AreaMeasurement()
        assert measure.unit == MeasurementUnit.METERS
        assert measure.closed is True

    def test_unique_id(self):
        """Each measurement should have unique ID."""
        m1 = AreaMeasurement()
        m2 = AreaMeasurement()
        assert m1.id != m2.id

    def test_add_point(self):
        """Should add boundary point."""
        measure = AreaMeasurement()
        point = measure.add_point(Vector3(0, 0, 0), "Corner")

        assert point.label == "Corner"
        assert len(measure.points) == 1

    def test_add_point_auto_label(self):
        """Should auto-generate vertex label."""
        measure = AreaMeasurement()
        point = measure.add_point(Vector3(0, 0, 0))

        assert point.label == "V1"

    def test_measure_triangle(self):
        """Should measure triangle area."""
        measure = AreaMeasurement()
        # Right triangle with base 2, height 2
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(2, 0, 0))
        measure.add_point(Vector3(0, 0, 2))

        result = measure.measure()

        assert result is not None
        # Area = 0.5 * base * height = 2
        assert abs(result.value - 2.0) < 0.01

    def test_measure_square(self):
        """Should measure square area."""
        measure = AreaMeasurement()
        # 10x10 square
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(10, 0, 0))
        measure.add_point(Vector3(10, 0, 10))
        measure.add_point(Vector3(0, 0, 10))

        result = measure.measure()

        assert result.measurement_type == MeasurementType.AREA
        assert abs(result.value - 100.0) < 0.1

    def test_measure_insufficient_points(self):
        """Should return None with less than 3 points."""
        measure = AreaMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(10, 0, 0))

        result = measure.measure()

        assert result is None

    def test_perimeter_auxiliary(self):
        """Should include perimeter in auxiliary values."""
        measure = AreaMeasurement()
        # 10x10 square
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(10, 0, 0))
        measure.add_point(Vector3(10, 0, 10))
        measure.add_point(Vector3(0, 0, 10))

        result = measure.measure()

        assert "perimeter" in result.auxiliary_values
        assert abs(result.auxiliary_values["perimeter"] - 40.0) < 0.1

    def test_measure_rectangle(self):
        """Should measure rectangle from two points."""
        measure = AreaMeasurement()
        result = measure.measure_rectangle(
            Vector3(0, 0, 0),
            Vector3(5, 0, 10)
        )

        assert result.value == 50.0
        assert result.auxiliary_values["width"] == 5.0
        assert result.auxiliary_values["depth"] == 10.0

    def test_measure_circle(self):
        """Should measure circle area."""
        measure = AreaMeasurement()
        result = measure.measure_circle(Vector3(0, 0, 0), radius=1.0)

        assert abs(result.value - math.pi) < 0.0001
        assert abs(result.auxiliary_values["circumference"] - 2 * math.pi) < 0.0001

    def test_circle_auxiliary_values(self):
        """Circle should include radius, diameter, circumference."""
        measure = AreaMeasurement()
        result = measure.measure_circle(Vector3(0, 0, 0), radius=5.0)

        assert result.auxiliary_values["radius"] == 5.0
        assert result.auxiliary_values["diameter"] == 10.0

    def test_formatted_result(self):
        """Result should include formatted string with squared unit."""
        measure = AreaMeasurement()
        result = measure.measure_rectangle(Vector3(0, 0, 0), Vector3(2, 0, 2))

        assert "m" in result.formatted
        assert "4.000" in result.formatted

    def test_clear_points(self):
        """Should clear all points."""
        measure = AreaMeasurement()
        measure.add_point(Vector3(0, 0, 0))
        measure.add_point(Vector3(1, 0, 0))

        measure.clear_points()

        assert len(measure.points) == 0


class TestMeasurementTool:
    """Tests for MeasurementTool main class."""

    def test_creation(self):
        """Tool should initialize with no active measurement."""
        tool = MeasurementTool()
        assert tool.active_measurement is None
        assert len(tool.history) == 0

    def test_start_distance_measurement(self):
        """Should start distance measurement."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.DISTANCE)

        assert tool.active_measurement == MeasurementType.DISTANCE

    def test_start_angle_measurement(self):
        """Should start angle measurement."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.ANGLE)

        assert tool.active_measurement == MeasurementType.ANGLE

    def test_start_area_measurement(self):
        """Should start area measurement."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.AREA)

        assert tool.active_measurement == MeasurementType.AREA

    def test_add_point_distance(self):
        """Should add point to distance measurement."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.DISTANCE)

        point = tool.add_point(Vector3(0, 0, 0))

        assert point is not None
        assert len(tool.distance.points) == 1

    def test_add_point_no_active(self):
        """Should return None if no active measurement."""
        tool = MeasurementTool()
        point = tool.add_point(Vector3(0, 0, 0))

        assert point is None

    def test_measure_distance(self):
        """Should measure distance and add to history."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.DISTANCE)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(10, 0, 0))

        result = tool.measure()

        assert result is not None
        assert abs(result.value - 10.0) < 0.0001
        assert len(tool.history) == 1

    def test_measure_angle(self):
        """Should measure angle and add to history."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.ANGLE)
        tool.add_point(Vector3(1, 0, 0))
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(0, 1, 0))

        result = tool.measure()

        assert result is not None
        assert abs(result.value - 90.0) < 0.01

    def test_measure_area(self):
        """Should measure area and add to history."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.AREA)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(10, 0, 0))
        tool.add_point(Vector3(10, 0, 10))
        tool.add_point(Vector3(0, 0, 10))

        result = tool.measure()

        assert result is not None
        assert abs(result.value - 100.0) < 0.1

    def test_clear_current(self):
        """Should clear current measurement points."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.DISTANCE)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(10, 0, 0))

        tool.clear_current()

        assert len(tool.distance.points) == 0

    def test_clear_all(self):
        """Should clear all measurements and history."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.DISTANCE)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(10, 0, 0))
        tool.measure()

        tool.clear_all()

        assert tool.active_measurement is None
        assert len(tool.history) == 0

    def test_clear_history(self):
        """Should clear only history."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.DISTANCE)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(10, 0, 0))
        tool.measure()

        tool.clear_history()

        assert len(tool.history) == 0

    def test_get_last_measurement(self):
        """Should return last measurement."""
        tool = MeasurementTool()
        tool.start_measurement(MeasurementType.DISTANCE)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(10, 0, 0))
        tool.measure()

        last = tool.get_last_measurement()

        assert last is not None
        assert abs(last.value - 10.0) < 0.0001

    def test_get_last_measurement_empty(self):
        """Should return None if no measurements."""
        tool = MeasurementTool()
        last = tool.get_last_measurement()

        assert last is None

    def test_set_unit(self):
        """Setting unit should propagate to sub-measurements."""
        tool = MeasurementTool()
        tool.unit = MeasurementUnit.CENTIMETERS

        assert tool.distance.unit == MeasurementUnit.CENTIMETERS
        assert tool.area.unit == MeasurementUnit.CENTIMETERS

    def test_set_angle_unit(self):
        """Setting angle unit should propagate."""
        tool = MeasurementTool()
        tool.angle_unit = AngleUnit.RADIANS

        assert tool.angle.angle_unit == AngleUnit.RADIANS

    def test_convert_value(self):
        """Should convert values between units."""
        tool = MeasurementTool()
        result = tool.convert_value(1.0, MeasurementUnit.METERS, MeasurementUnit.FEET)

        assert abs(result - 3.28084) < 0.001

    def test_on_measure_callback(self):
        """Should trigger on_measure callback."""
        tool = MeasurementTool()
        results = []

        tool.on("on_measure", lambda r: results.append(r))
        tool.start_measurement(MeasurementType.DISTANCE)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(10, 0, 0))
        tool.measure()

        assert len(results) == 1

    def test_on_point_add_callback(self):
        """Should trigger on_point_add callback."""
        tool = MeasurementTool()
        points = []

        tool.on("on_point_add", lambda p: points.append(p))
        tool.start_measurement(MeasurementType.DISTANCE)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(10, 0, 0))

        assert len(points) == 2

    def test_on_clear_callback(self):
        """Should trigger on_clear callback."""
        tool = MeasurementTool()
        clears = []

        tool.on("on_clear", lambda: clears.append(True))
        tool.start_measurement(MeasurementType.DISTANCE)  # This calls clear_current()
        tool.clear_current()  # This also calls clear_current()

        # Callback is triggered twice: once by start_measurement and once by explicit clear
        assert len(clears) == 2

    def test_off_callback(self):
        """Should unregister callback."""
        tool = MeasurementTool()
        results = []

        def callback(r):
            results.append(r)

        tool.on("on_measure", callback)
        tool.off("on_measure", callback)

        tool.start_measurement(MeasurementType.DISTANCE)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(10, 0, 0))
        tool.measure()

        assert len(results) == 0

    def test_multiple_measurements_history(self):
        """Multiple measurements should accumulate in history."""
        tool = MeasurementTool()

        # First measurement
        tool.start_measurement(MeasurementType.DISTANCE)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(10, 0, 0))
        tool.measure()

        # Second measurement
        tool.start_measurement(MeasurementType.DISTANCE)
        tool.add_point(Vector3(0, 0, 0))
        tool.add_point(Vector3(5, 0, 0))
        tool.measure()

        assert len(tool.history) == 2


class TestMeasurementResult:
    """Tests for MeasurementResult dataclass."""

    def test_creation(self):
        """Result should store all values."""
        result = MeasurementResult(
            measurement_type=MeasurementType.DISTANCE,
            value=10.0,
            unit=MeasurementUnit.METERS,
            formatted="10.000 m",
        )

        assert result.measurement_type == MeasurementType.DISTANCE
        assert result.value == 10.0
        assert result.unit == MeasurementUnit.METERS
        assert result.formatted == "10.000 m"

    def test_default_points(self):
        """Result should have empty points list by default."""
        result = MeasurementResult(
            measurement_type=MeasurementType.DISTANCE,
            value=10.0,
            unit=MeasurementUnit.METERS,
            formatted="10.000 m",
        )

        assert result.points == []

    def test_default_auxiliary(self):
        """Result should have empty auxiliary dict by default."""
        result = MeasurementResult(
            measurement_type=MeasurementType.DISTANCE,
            value=10.0,
            unit=MeasurementUnit.METERS,
            formatted="10.000 m",
        )

        assert result.auxiliary_values == {}
