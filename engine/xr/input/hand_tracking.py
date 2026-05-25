"""Hand tracking module for XR input with 26-joint tracking and gesture recognition.

This module provides comprehensive hand tracking support following the Trinity Pattern
with Batched, Interpolated, and Tracked descriptors for joint data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Annotated, Any, Callable, Optional

from engine.xr.config import XR_CONFIG

logger = logging.getLogger(__name__)

# Type aliases for Trinity descriptors (to be replaced with actual imports)
# from trinity.descriptors import Tracked, Batched, Interpolated, Range, Observable, Immutable, Computed
Tracked = "Tracked"
Batched = "Batched"
Interpolated = "Interpolated"
Range = "Range"
Observable = "Observable"
Immutable = "Immutable"
Computed = "Computed"


class HandJoint(IntEnum):
    """26-joint hand tracking enumeration following OpenXR standard.

    The hand skeleton consists of 26 joints arranged hierarchically:
    - WRIST: Root of the hand
    - Each finger has 4 joints: METACARPAL, PROXIMAL, INTERMEDIATE (except thumb), DISTAL, TIP
    """
    # Wrist
    WRIST = 0

    # Thumb (4 joints - no intermediate)
    THUMB_METACARPAL = 1
    THUMB_PROXIMAL = 2
    THUMB_DISTAL = 3
    THUMB_TIP = 4

    # Index finger (5 joints)
    INDEX_METACARPAL = 5
    INDEX_PROXIMAL = 6
    INDEX_INTERMEDIATE = 7
    INDEX_DISTAL = 8
    INDEX_TIP = 9

    # Middle finger (5 joints)
    MIDDLE_METACARPAL = 10
    MIDDLE_PROXIMAL = 11
    MIDDLE_INTERMEDIATE = 12
    MIDDLE_DISTAL = 13
    MIDDLE_TIP = 14

    # Ring finger (5 joints)
    RING_METACARPAL = 15
    RING_PROXIMAL = 16
    RING_INTERMEDIATE = 17
    RING_DISTAL = 18
    RING_TIP = 19

    # Pinky finger (5 joints)
    PINKY_METACARPAL = 20
    PINKY_PROXIMAL = 21
    PINKY_INTERMEDIATE = 22
    PINKY_DISTAL = 23
    PINKY_TIP = 24

    # Palm center (virtual joint for interaction)
    PALM = 25


# Total number of joints (use config value)
HAND_JOINT_COUNT = XR_CONFIG.runtime.HAND_JOINT_COUNT


class GestureType(IntEnum):
    """Standard gesture types recognized by the gesture recognition system."""
    NONE = 0
    PINCH = auto()
    POINT = auto()
    FIST = auto()
    OPEN_HAND = auto()
    THUMBS_UP = auto()
    THUMBS_DOWN = auto()
    PEACE = auto()
    OK_SIGN = auto()
    CUSTOM = auto()


@dataclass
class JointData:
    """Data for a single hand joint.

    Attributes:
        position: 3D position (x, y, z) in tracking space
        orientation: Quaternion (x, y, z, w) representing joint rotation
        radius: Collision radius for the joint in meters
        linear_velocity: Optional velocity vector for prediction
        angular_velocity: Optional angular velocity for prediction
    """
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    radius: float = 0.01
    linear_velocity: Optional[tuple[float, float, float]] = None
    angular_velocity: Optional[tuple[float, float, float]] = None


@dataclass
class GestureResult:
    """Result of gesture recognition.

    Attributes:
        gesture_type: The detected gesture type
        confidence: Confidence score from 0.0 to 1.0
        is_active: Whether the gesture is currently being held
        start_time: Timestamp when gesture started
        custom_name: Name for custom gestures
    """
    gesture_type: GestureType = GestureType.NONE
    confidence: float = 0.0
    is_active: bool = False
    start_time: float = 0.0
    custom_name: Optional[str] = None


@dataclass
class HandTrackingData:
    """Hand tracking component with 26 joints and gesture recognition.

    This component follows the Trinity Pattern with appropriate descriptors:
    - Batched + Interpolated + Tracked for joint arrays
    - Tracked + Observable for gesture state
    - Range for confidence values

    Attributes:
        hand: Which hand ("left" or "right")
        joint_positions: Batched positions for all 26 joints
        joint_orientations: Batched orientations for all 26 joints
        joint_radii: Collision radii for all joints
        is_tracked: Whether the hand is currently being tracked
        confidence: Overall tracking confidence (0.0 to 1.0)
        current_gesture: Currently detected gesture
        gesture_confidence: Confidence of gesture detection
        pinch_strength: Strength of pinch gesture (0.0 to 1.0)
    """
    # Identity (Immutable after creation)
    hand: str = "left"

    # Joint data (Batched + Interpolated + Tracked for smooth updates)
    joint_positions: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 0.0, 0.0)] * HAND_JOINT_COUNT
    )
    joint_orientations: list[tuple[float, float, float, float]] = field(
        default_factory=lambda: [(0.0, 0.0, 0.0, 1.0)] * HAND_JOINT_COUNT
    )
    joint_radii: list[float] = field(
        default_factory=lambda: [0.01] * HAND_JOINT_COUNT
    )

    # Tracking state (Tracked + Observable)
    is_tracked: bool = False
    confidence: float = 0.0  # Range(0, 1)

    # Gesture recognition (Tracked + Observable)
    current_gesture: GestureType = GestureType.NONE
    gesture_confidence: float = 0.0  # Range(0, 1)

    # Pinch detection (Tracked + Range)
    pinch_strength: float = 0.0  # Range(0, 1)

    # Internal state for velocity tracking
    _previous_positions: list[tuple[float, float, float]] = field(
        default_factory=list, repr=False
    )
    _last_update_time: float = field(default=0.0, repr=False)

    @property
    def pinch_position(self) -> tuple[float, float, float]:
        """Compute the midpoint between thumb tip and index tip.

        This is a computed property that returns the interaction point
        for pinch gestures.

        Returns:
            3D position of the pinch point
        """
        thumb_tip = self.joint_positions[HandJoint.THUMB_TIP]
        index_tip = self.joint_positions[HandJoint.INDEX_TIP]
        return (
            (thumb_tip[0] + index_tip[0]) / 2.0,
            (thumb_tip[1] + index_tip[1]) / 2.0,
            (thumb_tip[2] + index_tip[2]) / 2.0,
        )

    @property
    def is_pinching(self) -> bool:
        """Check if the hand is currently in a pinch gesture.

        Returns:
            True if pinch_strength > 0.8
        """
        return self.pinch_strength > 0.8

    @property
    def palm_position(self) -> tuple[float, float, float]:
        """Get the palm center position.

        Returns:
            3D position of the palm
        """
        return self.joint_positions[HandJoint.PALM]

    @property
    def palm_normal(self) -> tuple[float, float, float]:
        """Compute the palm normal vector.

        The normal points outward from the palm (toward the thumb side for left hand,
        toward the pinky side for right hand when palm faces forward).

        Returns:
            Normalized direction vector
        """
        # Use cross product of vectors from palm to index and palm to pinky
        palm = self.joint_positions[HandJoint.PALM]
        index = self.joint_positions[HandJoint.INDEX_METACARPAL]
        pinky = self.joint_positions[HandJoint.PINKY_METACARPAL]

        # Vector from palm to index
        v1 = (index[0] - palm[0], index[1] - palm[1], index[2] - palm[2])
        # Vector from palm to pinky
        v2 = (pinky[0] - palm[0], pinky[1] - palm[1], pinky[2] - palm[2])

        # Cross product
        normal = (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        )

        # Normalize
        length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if length > 1e-6:
            return (normal[0] / length, normal[1] / length, normal[2] / length)
        return (0.0, 1.0, 0.0)  # Default up vector

    def get_joint(self, joint: HandJoint) -> JointData:
        """Get the data for a specific joint.

        Args:
            joint: The joint to retrieve

        Returns:
            JointData containing position, orientation, and radius
        """
        return JointData(
            position=self.joint_positions[joint],
            orientation=self.joint_orientations[joint],
            radius=self.joint_radii[joint],
        )

    def get_finger_curl(self, finger: str) -> float:
        """Calculate the curl amount for a specific finger.

        Curl is measured as how closed the finger is, from 0.0 (fully extended)
        to 1.0 (fully curled/closed).

        Args:
            finger: One of "thumb", "index", "middle", "ring", "pinky"

        Returns:
            Curl amount from 0.0 to 1.0
        """
        finger_map = {
            "thumb": (HandJoint.THUMB_METACARPAL, HandJoint.THUMB_TIP),
            "index": (HandJoint.INDEX_METACARPAL, HandJoint.INDEX_TIP),
            "middle": (HandJoint.MIDDLE_METACARPAL, HandJoint.MIDDLE_TIP),
            "ring": (HandJoint.RING_METACARPAL, HandJoint.RING_TIP),
            "pinky": (HandJoint.PINKY_METACARPAL, HandJoint.PINKY_TIP),
        }

        if finger not in finger_map:
            return 0.0

        base_joint, tip_joint = finger_map[finger]
        base_pos = self.joint_positions[base_joint]
        tip_pos = self.joint_positions[tip_joint]
        palm_pos = self.joint_positions[HandJoint.PALM]

        # Calculate distance from tip to base
        tip_to_base = math.sqrt(
            (tip_pos[0] - base_pos[0])**2 +
            (tip_pos[1] - base_pos[1])**2 +
            (tip_pos[2] - base_pos[2])**2
        )

        # Calculate distance from tip to palm
        tip_to_palm = math.sqrt(
            (tip_pos[0] - palm_pos[0])**2 +
            (tip_pos[1] - palm_pos[1])**2 +
            (tip_pos[2] - palm_pos[2])**2
        )

        # Estimate curl based on tip proximity to palm
        # Fully extended finger has tip far from palm
        # Fully curled has tip close to palm
        max_distance = 0.15  # Approximate max finger length
        min_distance = 0.03  # Approximate curled distance

        curl = 1.0 - (tip_to_palm - min_distance) / (max_distance - min_distance)
        return max(0.0, min(1.0, curl))

    def update_joints(
        self,
        positions: list[tuple[float, float, float]],
        orientations: Optional[list[tuple[float, float, float, float]]] = None,
        radii: Optional[list[float]] = None,
        timestamp: float = 0.0,
    ) -> None:
        """Update all joint data in a batch operation.

        This method supports the Batched descriptor pattern for efficient updates.

        Args:
            positions: New positions for all 26 joints
            orientations: Optional new orientations for all joints
            radii: Optional new radii for all joints
            timestamp: Update timestamp for velocity calculation
        """
        if len(positions) != HAND_JOINT_COUNT:
            raise ValueError(f"Expected {HAND_JOINT_COUNT} positions, got {len(positions)}")

        # Store previous for velocity calculation
        self._previous_positions = self.joint_positions.copy()
        self._last_update_time = timestamp

        # Update positions
        self.joint_positions = list(positions)

        # Update orientations if provided
        if orientations is not None:
            if len(orientations) != HAND_JOINT_COUNT:
                raise ValueError(f"Expected {HAND_JOINT_COUNT} orientations, got {len(orientations)}")
            self.joint_orientations = list(orientations)

        # Update radii if provided
        if radii is not None:
            if len(radii) != HAND_JOINT_COUNT:
                raise ValueError(f"Expected {HAND_JOINT_COUNT} radii, got {len(radii)}")
            self.joint_radii = list(radii)


class GestureRecognizer:
    """Recognizes hand gestures from hand tracking data.

    This class analyzes hand joint positions to detect standard gestures
    and can be extended with custom gesture definitions.

    Attributes:
        pinch_threshold: Distance threshold for pinch detection
        point_curl_threshold: Maximum curl for pointing finger
        fist_curl_threshold: Minimum curl for fist detection
        custom_gestures: Dictionary of custom gesture recognizers
    """

    __slots__ = (
        'pinch_threshold',
        'point_curl_threshold',
        'fist_curl_threshold',
        'custom_gestures',
        '_gesture_history',
        '_smoothing_frames',
    )

    def __init__(
        self,
        pinch_threshold: float = 0.025,
        point_curl_threshold: float = 0.3,
        fist_curl_threshold: float = 0.7,
        smoothing_frames: int = 3,
    ):
        """Initialize the gesture recognizer.

        Args:
            pinch_threshold: Maximum distance between thumb and index for pinch
            point_curl_threshold: Maximum curl for the pointing finger
            fist_curl_threshold: Minimum curl for all fingers in fist
            smoothing_frames: Number of frames to smooth gesture detection
        """
        self.pinch_threshold = pinch_threshold
        self.point_curl_threshold = point_curl_threshold
        self.fist_curl_threshold = fist_curl_threshold
        self.custom_gestures: dict[str, Callable[[HandTrackingData], tuple[bool, float]]] = {}
        self._gesture_history: list[GestureType] = []
        self._smoothing_frames = smoothing_frames

    def recognize(self, hand_data: HandTrackingData) -> GestureResult:
        """Recognize the current gesture from hand tracking data.

        Args:
            hand_data: The hand tracking data to analyze

        Returns:
            GestureResult with the detected gesture and confidence
        """
        if not hand_data.is_tracked:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        # Check gestures in priority order
        results: list[tuple[GestureType, float]] = []

        # Pinch detection
        pinch_conf = self._detect_pinch(hand_data)
        if pinch_conf > 0.5:
            results.append((GestureType.PINCH, pinch_conf))

        # Point detection
        point_conf = self._detect_point(hand_data)
        if point_conf > 0.5:
            results.append((GestureType.POINT, point_conf))

        # Fist detection
        fist_conf = self._detect_fist(hand_data)
        if fist_conf > 0.5:
            results.append((GestureType.FIST, fist_conf))

        # Open hand detection
        open_conf = self._detect_open_hand(hand_data)
        if open_conf > 0.5:
            results.append((GestureType.OPEN_HAND, open_conf))

        # Thumbs up detection
        thumbs_up_conf = self._detect_thumbs_up(hand_data)
        if thumbs_up_conf > 0.5:
            results.append((GestureType.THUMBS_UP, thumbs_up_conf))

        # Custom gestures
        for name, detector in self.custom_gestures.items():
            detected, confidence = detector(hand_data)
            if detected and confidence > 0.5:
                return GestureResult(
                    gesture_type=GestureType.CUSTOM,
                    confidence=confidence,
                    is_active=True,
                    custom_name=name,
                )

        # Return highest confidence gesture
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            gesture, confidence = results[0]

            # Smooth gesture detection
            self._gesture_history.append(gesture)
            if len(self._gesture_history) > self._smoothing_frames:
                self._gesture_history.pop(0)

            # Require consistent detection
            if self._gesture_history.count(gesture) >= (self._smoothing_frames // 2 + 1):
                return GestureResult(
                    gesture_type=gesture,
                    confidence=confidence,
                    is_active=True,
                )

        return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

    def calculate_pinch_strength(self, hand_data: HandTrackingData) -> float:
        """Calculate the pinch strength (0.0 to 1.0).

        Args:
            hand_data: The hand tracking data

        Returns:
            Pinch strength from 0.0 (no pinch) to 1.0 (full pinch)
        """
        thumb_tip = hand_data.joint_positions[HandJoint.THUMB_TIP]
        index_tip = hand_data.joint_positions[HandJoint.INDEX_TIP]

        distance = math.sqrt(
            (thumb_tip[0] - index_tip[0])**2 +
            (thumb_tip[1] - index_tip[1])**2 +
            (thumb_tip[2] - index_tip[2])**2
        )

        # Map distance to strength (closer = stronger)
        max_dist = 0.1  # Maximum distance for any pinch
        min_dist = 0.01  # Minimum distance for full pinch

        strength = 1.0 - (distance - min_dist) / (max_dist - min_dist)
        return max(0.0, min(1.0, strength))

    def register_custom_gesture(
        self,
        name: str,
        detector: Callable[[HandTrackingData], tuple[bool, float]],
    ) -> None:
        """Register a custom gesture detector.

        Args:
            name: Unique name for the gesture
            detector: Function that takes HandTrackingData and returns (detected, confidence)
        """
        self.custom_gestures[name] = detector

    def unregister_custom_gesture(self, name: str) -> bool:
        """Unregister a custom gesture detector.

        Args:
            name: Name of the gesture to remove

        Returns:
            True if the gesture was removed, False if not found
        """
        if name in self.custom_gestures:
            del self.custom_gestures[name]
            return True
        return False

    def _detect_pinch(self, hand_data: HandTrackingData) -> float:
        """Detect pinch gesture.

        Returns:
            Confidence score from 0.0 to 1.0
        """
        thumb_tip = hand_data.joint_positions[HandJoint.THUMB_TIP]
        index_tip = hand_data.joint_positions[HandJoint.INDEX_TIP]

        distance = math.sqrt(
            (thumb_tip[0] - index_tip[0])**2 +
            (thumb_tip[1] - index_tip[1])**2 +
            (thumb_tip[2] - index_tip[2])**2
        )

        if distance < self.pinch_threshold:
            return 1.0 - (distance / self.pinch_threshold)
        return 0.0

    def _detect_point(self, hand_data: HandTrackingData) -> float:
        """Detect pointing gesture (index extended, others curled).

        Returns:
            Confidence score from 0.0 to 1.0
        """
        index_curl = hand_data.get_finger_curl("index")
        middle_curl = hand_data.get_finger_curl("middle")
        ring_curl = hand_data.get_finger_curl("ring")
        pinky_curl = hand_data.get_finger_curl("pinky")

        # Index should be extended, others curled
        index_extended = index_curl < self.point_curl_threshold
        others_curled = (
            middle_curl > self.fist_curl_threshold and
            ring_curl > self.fist_curl_threshold and
            pinky_curl > self.fist_curl_threshold
        )

        if index_extended and others_curled:
            # Confidence based on how well it matches
            conf = (
                (1.0 - index_curl) * 0.4 +
                middle_curl * 0.2 +
                ring_curl * 0.2 +
                pinky_curl * 0.2
            )
            return min(1.0, conf)
        return 0.0

    def _detect_fist(self, hand_data: HandTrackingData) -> float:
        """Detect fist gesture (all fingers curled).

        Returns:
            Confidence score from 0.0 to 1.0
        """
        curls = [
            hand_data.get_finger_curl("thumb"),
            hand_data.get_finger_curl("index"),
            hand_data.get_finger_curl("middle"),
            hand_data.get_finger_curl("ring"),
            hand_data.get_finger_curl("pinky"),
        ]

        # All fingers should be curled
        if all(c > self.fist_curl_threshold for c in curls):
            return sum(curls) / len(curls)
        return 0.0

    def _detect_open_hand(self, hand_data: HandTrackingData) -> float:
        """Detect open hand gesture (all fingers extended).

        Returns:
            Confidence score from 0.0 to 1.0
        """
        curls = [
            hand_data.get_finger_curl("thumb"),
            hand_data.get_finger_curl("index"),
            hand_data.get_finger_curl("middle"),
            hand_data.get_finger_curl("ring"),
            hand_data.get_finger_curl("pinky"),
        ]

        # All fingers should be extended
        if all(c < self.point_curl_threshold for c in curls):
            return 1.0 - (sum(curls) / len(curls))
        return 0.0

    def _detect_thumbs_up(self, hand_data: HandTrackingData) -> float:
        """Detect thumbs up gesture (thumb extended up, others curled).

        Returns:
            Confidence score from 0.0 to 1.0
        """
        thumb_curl = hand_data.get_finger_curl("thumb")
        other_curls = [
            hand_data.get_finger_curl("index"),
            hand_data.get_finger_curl("middle"),
            hand_data.get_finger_curl("ring"),
            hand_data.get_finger_curl("pinky"),
        ]

        # Thumb should be extended, others curled
        thumb_extended = thumb_curl < self.point_curl_threshold
        others_curled = all(c > self.fist_curl_threshold for c in other_curls)

        if thumb_extended and others_curled:
            # Check if thumb is pointing up (y component of thumb direction)
            thumb_base = hand_data.joint_positions[HandJoint.THUMB_METACARPAL]
            thumb_tip = hand_data.joint_positions[HandJoint.THUMB_TIP]

            thumb_dir_y = thumb_tip[1] - thumb_base[1]
            thumb_length = math.sqrt(
                (thumb_tip[0] - thumb_base[0])**2 +
                (thumb_tip[1] - thumb_base[1])**2 +
                (thumb_tip[2] - thumb_base[2])**2
            )

            if thumb_length > 1e-6:
                up_factor = thumb_dir_y / thumb_length
                if up_factor > 0.5:  # Thumb pointing somewhat up
                    return (1.0 - thumb_curl) * 0.5 + up_factor * 0.5
        return 0.0


@dataclass
class GestureEvent:
    """Event fired when a gesture is detected or released.

    This event type follows the EventMeta pattern for pooled events.

    Attributes:
        hand: Which hand triggered the event ("left" or "right")
        gesture: The gesture type
        confidence: Detection confidence
        timestamp: When the gesture was detected
        is_start: True if gesture started, False if ended
        pinch_position: Position of pinch if applicable
    """
    hand: str
    gesture: GestureType
    confidence: float
    timestamp: float
    is_start: bool = True
    pinch_position: Optional[tuple[float, float, float]] = None
    custom_name: Optional[str] = None


class HandTracker:
    """High-level hand tracking manager.

    This class manages hand tracking state and gesture recognition
    for both hands, providing a unified interface for hand input.

    Attributes:
        left_hand: Tracking data for the left hand
        right_hand: Tracking data for the right hand
        gesture_recognizer: Shared gesture recognition system
    """

    __slots__ = (
        'left_hand',
        'right_hand',
        'gesture_recognizer',
        '_event_callbacks',
        '_last_gestures',
    )

    def __init__(self, gesture_recognizer: Optional[GestureRecognizer] = None):
        """Initialize the hand tracker.

        Args:
            gesture_recognizer: Optional custom gesture recognizer
        """
        self.left_hand = HandTrackingData(hand="left")
        self.right_hand = HandTrackingData(hand="right")
        self.gesture_recognizer = gesture_recognizer or GestureRecognizer()
        self._event_callbacks: list[Callable[[GestureEvent], None]] = []
        self._last_gestures: dict[str, GestureType] = {
            "left": GestureType.NONE,
            "right": GestureType.NONE,
        }

    def update(
        self,
        left_positions: Optional[list[tuple[float, float, float]]] = None,
        left_orientations: Optional[list[tuple[float, float, float, float]]] = None,
        right_positions: Optional[list[tuple[float, float, float]]] = None,
        right_orientations: Optional[list[tuple[float, float, float, float]]] = None,
        timestamp: float = 0.0,
    ) -> None:
        """Update hand tracking data for both hands.

        Args:
            left_positions: New positions for left hand joints
            left_orientations: New orientations for left hand joints
            right_positions: New positions for right hand joints
            right_orientations: New orientations for right hand joints
            timestamp: Update timestamp
        """
        # Update left hand
        if left_positions is not None:
            self.left_hand.is_tracked = True
            self.left_hand.update_joints(left_positions, left_orientations, timestamp=timestamp)
            self._process_gestures(self.left_hand, "left", timestamp)
        else:
            if self.left_hand.is_tracked:
                # Hand lost tracking
                self.left_hand.is_tracked = False
                self._fire_gesture_end("left", self._last_gestures["left"], timestamp)

        # Update right hand
        if right_positions is not None:
            self.right_hand.is_tracked = True
            self.right_hand.update_joints(right_positions, right_orientations, timestamp=timestamp)
            self._process_gestures(self.right_hand, "right", timestamp)
        else:
            if self.right_hand.is_tracked:
                # Hand lost tracking
                self.right_hand.is_tracked = False
                self._fire_gesture_end("right", self._last_gestures["right"], timestamp)

    def add_gesture_callback(self, callback: Callable[[GestureEvent], None]) -> None:
        """Register a callback for gesture events.

        Args:
            callback: Function to call when gestures are detected
        """
        self._event_callbacks.append(callback)

    def remove_gesture_callback(self, callback: Callable[[GestureEvent], None]) -> bool:
        """Remove a gesture callback.

        Args:
            callback: The callback to remove

        Returns:
            True if callback was removed
        """
        try:
            self._event_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _process_gestures(self, hand_data: HandTrackingData, hand_name: str, timestamp: float) -> None:
        """Process gesture recognition for a hand.

        Args:
            hand_data: The hand tracking data
            hand_name: "left" or "right"
            timestamp: Current timestamp
        """
        # Update pinch strength
        hand_data.pinch_strength = self.gesture_recognizer.calculate_pinch_strength(hand_data)

        # Recognize gesture
        result = self.gesture_recognizer.recognize(hand_data)
        hand_data.current_gesture = result.gesture_type
        hand_data.gesture_confidence = result.confidence

        # Fire events for gesture changes
        last_gesture = self._last_gestures[hand_name]

        if result.gesture_type != last_gesture:
            # End previous gesture
            if last_gesture != GestureType.NONE:
                self._fire_gesture_end(hand_name, last_gesture, timestamp)

            # Start new gesture
            if result.gesture_type != GestureType.NONE:
                pinch_pos = hand_data.pinch_position if result.gesture_type == GestureType.PINCH else None
                event = GestureEvent(
                    hand=hand_name,
                    gesture=result.gesture_type,
                    confidence=result.confidence,
                    timestamp=timestamp,
                    is_start=True,
                    pinch_position=pinch_pos,
                    custom_name=result.custom_name,
                )
                self._fire_event(event)

            self._last_gestures[hand_name] = result.gesture_type

    def _fire_gesture_end(self, hand_name: str, gesture: GestureType, timestamp: float) -> None:
        """Fire a gesture end event.

        Args:
            hand_name: Which hand
            gesture: The gesture that ended
            timestamp: When it ended
        """
        if gesture != GestureType.NONE:
            event = GestureEvent(
                hand=hand_name,
                gesture=gesture,
                confidence=0.0,
                timestamp=timestamp,
                is_start=False,
            )
            self._fire_event(event)

    def _fire_event(self, event: GestureEvent) -> None:
        """Fire an event to all registered callbacks.

        Args:
            event: The event to fire
        """
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Gesture callback error: {e}")
