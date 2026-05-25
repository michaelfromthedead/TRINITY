"""Tests for eye tracking module.

Tests the eye tracking system including:
    - EyeTrackingData component
    - FixationDetector
    - BlinkDetector
    - EyeCalibration
    - EyeTracker
"""

import math
import sys
import importlib.util
import pytest

# Load the module directly to avoid package import issues
_spec = importlib.util.spec_from_file_location(
    'eye_tracking',
    'engine/xr/input/eye_tracking.py'
)
_eye_tracking = importlib.util.module_from_spec(_spec)
sys.modules['eye_tracking'] = _eye_tracking
_spec.loader.exec_module(_eye_tracking)

# Import from loaded module
EyeId = _eye_tracking.EyeId
CalibrationState = _eye_tracking.CalibrationState
GazeState = _eye_tracking.GazeState
EyeData = _eye_tracking.EyeData
FixationData = _eye_tracking.FixationData
SaccadeData = _eye_tracking.SaccadeData
BlinkData = _eye_tracking.BlinkData
EyeTrackingData = _eye_tracking.EyeTrackingData
FixationDetector = _eye_tracking.FixationDetector
BlinkDetector = _eye_tracking.BlinkDetector
CalibrationPoint = _eye_tracking.CalibrationPoint
EyeCalibration = _eye_tracking.EyeCalibration
EyeTracker = _eye_tracking.EyeTracker


# =============================================================================
# Enum Tests
# =============================================================================


class TestEyeId:
    """Tests for EyeId enumeration."""

    def test_left_eye(self):
        """Left eye should be 0."""
        assert EyeId.LEFT == 0

    def test_right_eye(self):
        """Right eye should be 1."""
        assert EyeId.RIGHT == 1

    def test_combined(self):
        """Combined should be 2."""
        assert EyeId.COMBINED == 2


class TestCalibrationState:
    """Tests for CalibrationState enumeration."""

    def test_uncalibrated(self):
        """Uncalibrated should be 0."""
        assert CalibrationState.UNCALIBRATED == 0

    def test_initial(self):
        """Initial calibration state exists."""
        assert CalibrationState.INITIAL is not None

    def test_dynamic(self):
        """Dynamic calibration state exists."""
        assert CalibrationState.DYNAMIC is not None

    def test_profile_loaded(self):
        """Profile loaded state exists."""
        assert CalibrationState.PROFILE_LOADED is not None


class TestGazeState:
    """Tests for GazeState enumeration."""

    def test_unknown(self):
        """Unknown should be 0."""
        assert GazeState.UNKNOWN == 0

    def test_fixation(self):
        """Fixation state exists."""
        assert GazeState.FIXATION is not None

    def test_saccade(self):
        """Saccade state exists."""
        assert GazeState.SACCADE is not None

    def test_smooth_pursuit(self):
        """Smooth pursuit state exists."""
        assert GazeState.SMOOTH_PURSUIT is not None

    def test_blink(self):
        """Blink state exists."""
        assert GazeState.BLINK is not None


# =============================================================================
# EyeData Tests
# =============================================================================


class TestEyeData:
    """Tests for EyeData dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        eye = EyeData()
        assert eye.pupil_position == (0.0, 0.0)
        assert eye.pupil_diameter == 3.0
        assert eye.openness == 1.0
        assert eye.gaze_origin == (0.0, 0.0, 0.0)
        assert eye.gaze_direction == (0.0, 0.0, -1.0)
        assert eye.is_valid is False
        assert eye.confidence == 0.0

    def test_custom_values(self):
        """Test custom initialization."""
        eye = EyeData(
            pupil_position=(0.1, 0.2),
            pupil_diameter=4.5,
            openness=0.8,
            gaze_origin=(0.0, 0.1, 0.0),
            gaze_direction=(0.0, 0.0, -1.0),
            is_valid=True,
            confidence=0.95,
        )
        assert eye.pupil_position == (0.1, 0.2)
        assert eye.pupil_diameter == 4.5
        assert eye.openness == 0.8
        assert eye.is_valid is True


# =============================================================================
# FixationData Tests
# =============================================================================


class TestFixationData:
    """Tests for FixationData dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        fixation = FixationData()
        assert fixation.position == (0.0, 0.0, 0.0)
        assert fixation.start_time == 0.0
        assert fixation.duration == 0.0
        assert fixation.is_active is False
        assert fixation.stability == 0.0

    def test_active_fixation(self):
        """Test active fixation."""
        fixation = FixationData(
            position=(1.0, 0.0, -2.0),
            start_time=1.5,
            duration=0.3,
            is_active=True,
            stability=0.95,
        )
        assert fixation.is_active is True
        assert fixation.stability == 0.95


# =============================================================================
# SaccadeData Tests
# =============================================================================


class TestSaccadeData:
    """Tests for SaccadeData dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        saccade = SaccadeData()
        assert saccade.start_position == (0.0, 0.0, 0.0)
        assert saccade.end_position == (0.0, 0.0, 0.0)
        assert saccade.amplitude == 0.0
        assert saccade.peak_velocity == 0.0

    def test_saccade_data(self):
        """Test saccade with movement data."""
        saccade = SaccadeData(
            start_position=(0.0, 0.0, -1.0),
            end_position=(0.2, 0.0, -1.0),
            start_time=1.0,
            duration=0.05,
            amplitude=11.3,
            peak_velocity=450.0,
        )
        assert saccade.amplitude == 11.3
        assert saccade.peak_velocity == 450.0


# =============================================================================
# BlinkData Tests
# =============================================================================


class TestBlinkData:
    """Tests for BlinkData dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        blink = BlinkData()
        assert blink.start_time == 0.0
        assert blink.duration == 0.0
        assert blink.is_complete is False
        assert blink.eye == EyeId.COMBINED

    def test_complete_blink(self):
        """Test complete blink data."""
        blink = BlinkData(
            start_time=2.5,
            duration=0.15,
            is_complete=True,
            eye=EyeId.COMBINED,
        )
        assert blink.is_complete is True
        assert blink.duration == 0.15


# =============================================================================
# EyeTrackingData Tests
# =============================================================================


class TestEyeTrackingData:
    """Tests for EyeTrackingData component."""

    def test_default_initialization(self):
        """Test default component initialization."""
        eye_data = EyeTrackingData()
        assert eye_data.gaze_origin == (0.0, 0.0, 0.0)
        assert eye_data.gaze_direction == (0.0, 0.0, -1.0)
        assert eye_data.is_fixating is False
        assert eye_data.calibration_state == CalibrationState.UNCALIBRATED
        assert eye_data.is_calibrated is False
        assert eye_data.confidence == 0.0

    def test_per_eye_data_initialized(self):
        """Per-eye data should be initialized."""
        eye_data = EyeTrackingData()
        assert eye_data.left_eye is not None
        assert eye_data.right_eye is not None
        assert eye_data.left_pupil_position == (0.0, 0.0)
        assert eye_data.right_pupil_position == (0.0, 0.0)

    def test_openness_values(self):
        """Openness values should default to 1.0 (fully open)."""
        eye_data = EyeTrackingData()
        assert eye_data.left_openness == 1.0
        assert eye_data.right_openness == 1.0

    def test_pupil_diameter_defaults(self):
        """Pupil diameters should have reasonable defaults."""
        eye_data = EyeTrackingData()
        assert eye_data.left_pupil_diameter == 3.0
        assert eye_data.right_pupil_diameter == 3.0

    def test_is_blinking_false_when_open(self):
        """is_blinking should be False when eyes are open."""
        eye_data = EyeTrackingData()
        eye_data.left_openness = 0.9
        eye_data.right_openness = 0.9
        assert eye_data.is_blinking is False

    def test_is_blinking_true_when_closed(self):
        """is_blinking should be True when eyes are closed."""
        eye_data = EyeTrackingData()
        eye_data.left_openness = 0.1
        eye_data.right_openness = 0.1
        assert eye_data.is_blinking is True

    def test_average_pupil_diameter_both_valid(self):
        """Average pupil diameter should average both eyes when valid."""
        eye_data = EyeTrackingData()
        eye_data.left_pupil_diameter = 4.0
        eye_data.right_pupil_diameter = 6.0
        eye_data.left_eye.is_valid = True
        eye_data.right_eye.is_valid = True

        assert eye_data.average_pupil_diameter == 5.0

    def test_average_pupil_diameter_one_valid(self):
        """Average should use single eye when only one is valid."""
        eye_data = EyeTrackingData()
        eye_data.left_pupil_diameter = 4.0
        eye_data.right_pupil_diameter = 6.0
        eye_data.left_eye.is_valid = True
        eye_data.right_eye.is_valid = False

        assert eye_data.average_pupil_diameter == 4.0

    def test_get_gaze_point_at_distance(self):
        """get_gaze_point_at_distance should project gaze correctly."""
        eye_data = EyeTrackingData()
        eye_data.gaze_origin = (0.0, 0.0, 0.0)
        eye_data.gaze_direction = (0.0, 0.0, -1.0)

        point = eye_data.get_gaze_point_at_distance(2.0)
        assert point == (0.0, 0.0, -2.0)

    def test_get_gaze_point_with_angled_gaze(self):
        """get_gaze_point_at_distance should work with angled gaze."""
        eye_data = EyeTrackingData()
        eye_data.gaze_origin = (0.0, 0.0, 0.0)
        # Normalize 45 degree down angle
        sqrt2 = math.sqrt(2) / 2
        eye_data.gaze_direction = (0.0, -sqrt2, -sqrt2)

        point = eye_data.get_gaze_point_at_distance(1.0)
        assert abs(point[0]) < 0.001
        assert abs(point[1] - (-sqrt2)) < 0.001
        assert abs(point[2] - (-sqrt2)) < 0.001

    def test_update_method(self):
        """update should set all provided values."""
        eye_data = EyeTrackingData()

        eye_data.update(
            gaze_origin=(0.0, 0.1, 0.0),
            gaze_direction=(0.0, 0.0, -1.0),
            left_pupil_position=(0.1, 0.1),
            right_pupil_position=(-0.1, 0.1),
            left_pupil_diameter=4.0,
            right_pupil_diameter=4.2,
            left_openness=0.95,
            right_openness=0.97,
            confidence=0.9,
            timestamp=1.0,
        )

        assert eye_data.gaze_origin == (0.0, 0.1, 0.0)
        assert eye_data.left_pupil_position == (0.1, 0.1)
        assert eye_data.right_pupil_diameter == 4.2
        assert eye_data.left_openness == 0.95
        assert eye_data.confidence == 0.9
        assert eye_data.is_tracked is True

    def test_update_clamps_values(self):
        """update should clamp values to valid ranges."""
        eye_data = EyeTrackingData()

        eye_data.update(
            gaze_origin=(0.0, 0.0, 0.0),
            gaze_direction=(0.0, 0.0, -1.0),
            left_openness=1.5,  # Should clamp to 1.0
            right_openness=-0.5,  # Should clamp to 0.0
            confidence=2.0,  # Should clamp to 1.0
        )

        assert eye_data.left_openness == 1.0
        assert eye_data.right_openness == 0.0
        assert eye_data.confidence == 1.0

    def test_convergence_distance_parallel(self):
        """convergence_distance should return inf for parallel gaze."""
        eye_data = EyeTrackingData()
        # Parallel forward gaze
        eye_data.left_eye.gaze_origin = (-0.03, 0.0, 0.0)
        eye_data.left_eye.gaze_direction = (0.0, 0.0, -1.0)
        eye_data.left_eye.is_valid = True
        eye_data.right_eye.gaze_origin = (0.03, 0.0, 0.0)
        eye_data.right_eye.gaze_direction = (0.0, 0.0, -1.0)
        eye_data.right_eye.is_valid = True

        # Parallel rays should have infinite convergence
        convergence = eye_data.convergence_distance
        assert convergence == float('inf')


# =============================================================================
# FixationDetector Tests
# =============================================================================


class TestFixationDetector:
    """Tests for FixationDetector."""

    def test_default_initialization(self):
        """Test default detector initialization."""
        detector = FixationDetector()
        assert detector.velocity_threshold == 30.0
        assert detector.min_fixation_duration == 0.1
        assert detector.min_saccade_duration == 0.02

    def test_custom_thresholds(self):
        """Test custom threshold initialization."""
        detector = FixationDetector(
            velocity_threshold=50.0,
            min_fixation_duration=0.15,
            min_saccade_duration=0.03,
        )
        assert detector.velocity_threshold == 50.0
        assert detector.min_fixation_duration == 0.15

    def test_update_untracked_returns_unknown(self):
        """Untracked eye data should return UNKNOWN state."""
        detector = FixationDetector()
        eye_data = EyeTrackingData()
        eye_data.is_tracked = False

        state = detector.update(eye_data, timestamp=1.0)
        assert state == GazeState.UNKNOWN

    def test_stable_gaze_becomes_fixation(self):
        """Stable gaze should become FIXATION after min duration."""
        detector = FixationDetector(min_fixation_duration=0.1)
        eye_data = EyeTrackingData()
        eye_data.is_tracked = True
        eye_data.gaze_direction = (0.0, 0.0, -1.0)

        # Simulate stable gaze over time
        state = GazeState.UNKNOWN
        for i in range(20):  # 200ms at 100Hz
            state = detector.update(eye_data, timestamp=i * 0.01)

        assert state == GazeState.FIXATION

    def test_get_current_fixation(self):
        """get_current_fixation should return active fixation."""
        detector = FixationDetector(min_fixation_duration=0.05)
        eye_data = EyeTrackingData()
        eye_data.is_tracked = True
        eye_data.gaze_direction = (0.0, 0.0, -1.0)

        # Build up fixation
        for i in range(10):
            detector.update(eye_data, timestamp=i * 0.01)

        fixation = detector.get_current_fixation()
        assert fixation is not None
        assert fixation.is_active is True


# =============================================================================
# BlinkDetector Tests
# =============================================================================


class TestBlinkDetector:
    """Tests for BlinkDetector."""

    def test_default_initialization(self):
        """Test default detector initialization."""
        detector = BlinkDetector()
        assert detector.openness_threshold == 0.2
        assert detector.min_blink_duration == 0.05
        assert detector.max_blink_duration == 0.4

    def test_custom_thresholds(self):
        """Test custom threshold initialization."""
        detector = BlinkDetector(
            openness_threshold=0.3,
            min_blink_duration=0.1,
            max_blink_duration=0.5,
        )
        assert detector.openness_threshold == 0.3
        assert detector.max_blink_duration == 0.5

    def test_no_blink_when_eyes_open(self):
        """No blink should be detected when eyes are open."""
        detector = BlinkDetector()
        eye_data = EyeTrackingData()
        eye_data.left_openness = 0.9
        eye_data.right_openness = 0.9

        result = detector.update(eye_data, timestamp=1.0)
        assert result is None

    def test_blink_detection(self):
        """Blink should be detected when eyes close and reopen."""
        detector = BlinkDetector(min_blink_duration=0.05, max_blink_duration=0.4)
        eye_data = EyeTrackingData()

        # Eyes open
        eye_data.left_openness = 0.9
        eye_data.right_openness = 0.9
        detector.update(eye_data, timestamp=0.0)

        # Eyes close
        eye_data.left_openness = 0.1
        eye_data.right_openness = 0.1
        detector.update(eye_data, timestamp=0.1)

        # Eyes reopen (150ms blink)
        eye_data.left_openness = 0.9
        eye_data.right_openness = 0.9
        result = detector.update(eye_data, timestamp=0.25)

        assert result is not None
        assert result.is_complete is True
        assert 0.05 <= result.duration <= 0.4

    def test_blink_callback(self):
        """Blink callbacks should be called on blink detection."""
        detector = BlinkDetector(min_blink_duration=0.05, max_blink_duration=0.4)
        blinks_detected: list[BlinkData] = []

        def on_blink(blink: BlinkData):
            blinks_detected.append(blink)

        detector.add_blink_callback(on_blink)

        eye_data = EyeTrackingData()

        # Simulate blink
        eye_data.left_openness = 0.9
        eye_data.right_openness = 0.9
        detector.update(eye_data, timestamp=0.0)

        eye_data.left_openness = 0.1
        eye_data.right_openness = 0.1
        detector.update(eye_data, timestamp=0.1)

        eye_data.left_openness = 0.9
        eye_data.right_openness = 0.9
        detector.update(eye_data, timestamp=0.25)

        assert len(blinks_detected) == 1

    def test_remove_blink_callback(self):
        """Blink callbacks should be removable."""
        detector = BlinkDetector()

        def on_blink(blink: BlinkData):
            pass

        detector.add_blink_callback(on_blink)
        assert detector.remove_blink_callback(on_blink) is True
        assert detector.remove_blink_callback(on_blink) is False


# =============================================================================
# CalibrationPoint Tests
# =============================================================================


class TestCalibrationPoint:
    """Tests for CalibrationPoint."""

    def test_default_values(self):
        """Test default initialization."""
        point = CalibrationPoint(target_position=(0.0, 0.0, -1.0))
        assert point.target_position == (0.0, 0.0, -1.0)
        assert point.measured_gaze is None
        assert point.error == 0.0
        assert point.is_valid is False

    def test_calibrated_point(self):
        """Test calibrated point."""
        point = CalibrationPoint(
            target_position=(0.0, 0.0, -1.0),
            measured_gaze=(0.01, 0.0, -1.0),
            error=0.57,  # About 0.57 degrees
            is_valid=True,
        )
        assert point.is_valid is True
        assert point.error > 0


# =============================================================================
# EyeCalibration Tests
# =============================================================================


class TestEyeCalibration:
    """Tests for EyeCalibration system."""

    def test_default_initialization(self):
        """Test default calibration initialization."""
        calibration = EyeCalibration()
        assert calibration.state == CalibrationState.UNCALIBRATED
        assert len(calibration.calibration_points) == 9  # Default 3x3 grid
        assert calibration.average_error == float('inf')

    def test_custom_point_count(self):
        """Test calibration with custom point count."""
        cal_5 = EyeCalibration(num_points=5)
        assert len(cal_5.calibration_points) == 5

        cal_13 = EyeCalibration(num_points=13)
        assert len(cal_13.calibration_points) == 13

    def test_start_calibration(self):
        """start_calibration should return first target."""
        calibration = EyeCalibration()
        first_point = calibration.start_calibration()

        assert calibration.state == CalibrationState.INITIAL
        assert first_point is not None
        assert first_point == calibration.calibration_points[0]

    def test_is_calibrating(self):
        """is_calibrating should reflect state."""
        calibration = EyeCalibration()
        assert calibration.is_calibrating() is False

        calibration.start_calibration()
        assert calibration.is_calibrating() is True

    def test_get_current_target(self):
        """get_current_target should return current calibration target."""
        calibration = EyeCalibration()
        assert calibration.get_current_target() is None

        calibration.start_calibration()
        target = calibration.get_current_target()
        assert target is not None
        assert target == calibration.calibration_points[0]

    def test_add_gaze_sample(self):
        """add_gaze_sample should collect samples and advance."""
        calibration = EyeCalibration(num_points=5, samples_per_point=3)
        calibration.start_calibration()

        # Add samples for first point
        gaze_dir = (0.0, 0.0, -1.0)
        for _ in range(3):
            next_point = calibration.add_gaze_sample(gaze_dir)

        # Should have moved to second point
        assert calibration._current_point_index == 1
        assert calibration.calibration_points[0].is_valid is True

    def test_calibration_completes(self):
        """Calibration should complete after all points."""
        calibration = EyeCalibration(num_points=3, samples_per_point=2)
        calibration.start_calibration()

        gaze_dir = (0.0, 0.0, -1.0)

        # Complete all points
        for point_idx in range(3):
            for sample_idx in range(2):
                result = calibration.add_gaze_sample(gaze_dir)

        # Should be complete
        assert result is None
        assert calibration.average_error < float('inf')

    def test_save_and_load_profile(self):
        """Profile should be savable and loadable."""
        calibration = EyeCalibration(num_points=3, samples_per_point=2)
        calibration.start_calibration()

        # Complete calibration
        gaze_dir = (0.0, 0.0, -1.0)
        for _ in range(6):
            calibration.add_gaze_sample(gaze_dir)

        # Save profile
        profile = calibration.save_profile()
        assert 'average_error' in profile
        assert 'points' in profile

        # Load into new calibration
        new_calibration = EyeCalibration()
        success = new_calibration.load_profile(profile)

        assert success is True
        assert new_calibration.state == CalibrationState.PROFILE_LOADED
        assert new_calibration.average_error == calibration.average_error


# =============================================================================
# EyeTracker Tests
# =============================================================================


class TestEyeTracker:
    """Tests for EyeTracker manager."""

    def test_default_initialization(self):
        """Test default tracker initialization."""
        tracker = EyeTracker()
        assert tracker.eye_data is not None
        assert tracker.calibration is not None
        assert tracker.fixation_detector is not None
        assert tracker.blink_detector is not None

    def test_custom_components(self):
        """Test tracker with custom components."""
        custom_calibration = EyeCalibration(num_points=5)
        custom_fixation = FixationDetector(velocity_threshold=50.0)
        custom_blink = BlinkDetector(openness_threshold=0.3)

        tracker = EyeTracker(
            calibration=custom_calibration,
            fixation_detector=custom_fixation,
            blink_detector=custom_blink,
        )

        assert len(tracker.calibration.calibration_points) == 5
        assert tracker.fixation_detector.velocity_threshold == 50.0
        assert tracker.blink_detector.openness_threshold == 0.3

    def test_update_basic(self):
        """update should set eye tracking data."""
        tracker = EyeTracker()

        tracker.update(
            gaze_origin=(0.0, 0.1, 0.0),
            gaze_direction=(0.0, 0.0, -1.0),
            timestamp=1.0,
            confidence=0.95,
        )

        assert tracker.eye_data.gaze_origin == (0.0, 0.1, 0.0)
        assert tracker.eye_data.confidence == 0.95
        assert tracker.eye_data.is_tracked is True

    def test_update_with_pupil_data(self):
        """update should handle pupil data."""
        tracker = EyeTracker()

        tracker.update(
            gaze_origin=(0.0, 0.0, 0.0),
            gaze_direction=(0.0, 0.0, -1.0),
            timestamp=1.0,
            left_pupil_position=(0.1, 0.05),
            right_pupil_position=(-0.1, 0.05),
            left_pupil_diameter=4.0,
            right_pupil_diameter=4.2,
        )

        assert tracker.eye_data.left_pupil_position == (0.1, 0.05)
        assert tracker.eye_data.right_pupil_diameter == 4.2

    def test_update_callback(self):
        """Update callbacks should be called."""
        tracker = EyeTracker()
        updates_received: list[EyeTrackingData] = []

        def on_update(data: EyeTrackingData):
            updates_received.append(data)

        tracker.add_update_callback(on_update)

        tracker.update(
            gaze_origin=(0.0, 0.0, 0.0),
            gaze_direction=(0.0, 0.0, -1.0),
            timestamp=1.0,
        )

        assert len(updates_received) == 1

    def test_remove_update_callback(self):
        """Update callbacks should be removable."""
        tracker = EyeTracker()

        def on_update(data: EyeTrackingData):
            pass

        tracker.add_update_callback(on_update)
        assert tracker.remove_update_callback(on_update) is True
        assert tracker.remove_update_callback(on_update) is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestEyeTrackingIntegration:
    """Integration tests for eye tracking system."""

    def test_full_tracking_workflow(self):
        """Test complete tracking workflow."""
        tracker = EyeTracker()

        # Simulate tracking data over multiple frames
        for frame in range(60):  # 1 second at 60Hz
            tracker.update(
                gaze_origin=(0.0, 0.1, 0.0),
                gaze_direction=(0.0, 0.0, -1.0),
                timestamp=frame / 60.0,
                left_pupil_diameter=3.5,
                right_pupil_diameter=3.5,
                left_openness=0.95,
                right_openness=0.95,
                confidence=0.9,
            )

        assert tracker.eye_data.is_tracked is True
        assert tracker.eye_data.is_fixating is True

    def test_calibration_workflow(self):
        """Test calibration through tracker."""
        tracker = EyeTracker()

        # Start calibration
        first_target = tracker.calibration.start_calibration()
        assert first_target is not None

        # Simulate looking at targets
        for point_idx in range(len(tracker.calibration.calibration_points)):
            target = tracker.calibration.calibration_points[point_idx]
            gaze_dir = target.target_position

            for sample_idx in range(10):  # Default samples per point
                tracker.calibration.add_gaze_sample(gaze_dir)

        # Calibration should be complete
        assert tracker.calibration.average_error < float('inf')

    def test_fixation_detection_workflow(self):
        """Test fixation detection through full workflow."""
        tracker = EyeTracker()

        # Look at same point for a while (stable gaze)
        for frame in range(30):  # 500ms at 60Hz
            tracker.update(
                gaze_origin=(0.0, 0.0, 0.0),
                gaze_direction=(0.0, 0.0, -1.0),
                timestamp=frame / 60.0,
                confidence=0.95,
            )

        assert tracker.eye_data.gaze_state == GazeState.FIXATION
        assert tracker.eye_data.is_fixating is True

    def test_blink_detection_workflow(self):
        """Test blink detection through full workflow."""
        tracker = EyeTracker()
        blinks: list[BlinkData] = []

        def on_blink(blink: BlinkData):
            blinks.append(blink)

        tracker.blink_detector.add_blink_callback(on_blink)

        # Eyes open
        for frame in range(10):
            tracker.update(
                gaze_origin=(0.0, 0.0, 0.0),
                gaze_direction=(0.0, 0.0, -1.0),
                timestamp=frame / 60.0,
                left_openness=0.95,
                right_openness=0.95,
            )

        # Eyes close (blink)
        for frame in range(10, 20):
            tracker.update(
                gaze_origin=(0.0, 0.0, 0.0),
                gaze_direction=(0.0, 0.0, -1.0),
                timestamp=frame / 60.0,
                left_openness=0.1,
                right_openness=0.1,
            )

        # Eyes reopen
        for frame in range(20, 30):
            tracker.update(
                gaze_origin=(0.0, 0.0, 0.0),
                gaze_direction=(0.0, 0.0, -1.0),
                timestamp=frame / 60.0,
                left_openness=0.95,
                right_openness=0.95,
            )

        assert len(blinks) == 1
        assert blinks[0].is_complete is True
