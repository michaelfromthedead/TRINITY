"""XR (Virtual/Augmented Reality) input devices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .input_manager import InputDevice, InputDeviceType, InputEvent


class XRButton(Enum):
    """XR controller button identifiers."""
    TRIGGER = auto()
    GRIP = auto()
    A = auto()
    B = auto()
    X = auto()
    Y = auto()
    THUMBSTICK = auto()
    MENU = auto()


@dataclass(slots=True)
class Pose:
    """Represents a 6DOF pose in 3D space."""
    position: tuple[float, float, float]  # x, y, z
    orientation: tuple[float, float, float, float]  # quaternion (x, y, z, w)


class XRController(InputDevice):
    """XR motion controller device."""
    __slots__ = (
        '_pose', '_velocity', '_angular_velocity',
        '_trigger_value', '_grip_value', '_thumbstick',
        '_current_buttons', '_previous_buttons',
        '_pressed_buttons', '_released_buttons'
    )

    def __init__(self, name: str = "XR Controller", device_id: int = 0):
        """Initialize the XR controller.

        Args:
            name: Device name
            device_id: Unique device identifier
        """
        super().__init__(InputDeviceType.XR_CONTROLLER, name, device_id)
        self._pose = Pose(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0)
        )
        self._velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._trigger_value: float = 0.0
        self._grip_value: float = 0.0
        self._thumbstick: tuple[float, float] = (0.0, 0.0)
        self._current_buttons: set[XRButton] = set()
        self._previous_buttons: set[XRButton] = set()
        self._pressed_buttons: set[XRButton] = set()
        self._released_buttons: set[XRButton] = set()

    @property
    def pose(self) -> Pose:
        """Get the current controller pose.

        Returns:
            6DOF pose (position and orientation)
        """
        return self._pose

    @property
    def velocity(self) -> tuple[float, float, float]:
        """Get the linear velocity.

        Returns:
            (vx, vy, vz) in m/s
        """
        return self._velocity

    @property
    def angular_velocity(self) -> tuple[float, float, float]:
        """Get the angular velocity.

        Returns:
            (wx, wy, wz) in rad/s
        """
        return self._angular_velocity

    @property
    def trigger_value(self) -> float:
        """Get the trigger analog value.

        Returns:
            Value from 0.0 to 1.0
        """
        return self._trigger_value

    @property
    def grip_value(self) -> float:
        """Get the grip analog value.

        Returns:
            Value from 0.0 to 1.0
        """
        return self._grip_value

    @property
    def thumbstick(self) -> tuple[float, float]:
        """Get the thumbstick position.

        Returns:
            (x, y) from -1.0 to 1.0
        """
        return self._thumbstick

    def is_button_down(self, button: XRButton) -> bool:
        """Check if a button is currently held down.

        Args:
            button: The button to check

        Returns:
            True if button is down
        """
        return button in self._current_buttons

    def is_button_pressed(self, button: XRButton) -> bool:
        """Check if a button was just pressed this frame.

        Args:
            button: The button to check

        Returns:
            True if button was pressed this frame
        """
        return button in self._pressed_buttons

    def is_button_released(self, button: XRButton) -> bool:
        """Check if a button was just released this frame.

        Args:
            button: The button to check

        Returns:
            True if button was released this frame
        """
        return button in self._released_buttons

    def update(self, events: list[InputEvent]) -> None:
        """Update controller state with new events.

        Args:
            events: List of XR controller events
        """
        # Clear frame-specific states
        self._pressed_buttons.clear()
        self._released_buttons.clear()

        # Store previous frame state
        self._previous_buttons = self._current_buttons.copy()

        # Process events
        for event in events:
            if event.event_type == 'xr_pose':
                pos = event.data.get('position', self._pose.position)
                orient = event.data.get('orientation', self._pose.orientation)
                self._pose = Pose(
                    position=tuple(float(x) for x in pos),
                    orientation=tuple(float(x) for x in orient)
                )

                vel = event.data.get('velocity', self._velocity)
                self._velocity = tuple(float(x) for x in vel)

                ang_vel = event.data.get('angular_velocity', self._angular_velocity)
                self._angular_velocity = tuple(float(x) for x in ang_vel)

            elif event.event_type == 'xr_trigger':
                value = event.data.get('value', 0.0)
                self._trigger_value = max(0.0, min(1.0, float(value)))

            elif event.event_type == 'xr_grip':
                value = event.data.get('value', 0.0)
                self._grip_value = max(0.0, min(1.0, float(value)))

            elif event.event_type == 'xr_thumbstick':
                x = event.data.get('x', 0.0)
                y = event.data.get('y', 0.0)
                self._thumbstick = (
                    max(-1.0, min(1.0, float(x))),
                    max(-1.0, min(1.0, float(y)))
                )

            elif event.event_type == 'xr_button_down':
                button = event.data.get('button')
                if button and isinstance(button, XRButton):
                    self._current_buttons.add(button)
                    if button not in self._previous_buttons:
                        self._pressed_buttons.add(button)

            elif event.event_type == 'xr_button_up':
                button = event.data.get('button')
                if button and isinstance(button, XRButton):
                    if button in self._current_buttons:
                        self._current_buttons.remove(button)
                        self._released_buttons.add(button)


class HandJoint(Enum):
    """Hand tracking joint identifiers."""
    WRIST = auto()

    # Thumb
    THUMB_METACARPAL = auto()
    THUMB_PROXIMAL = auto()
    THUMB_DISTAL = auto()
    THUMB_TIP = auto()

    # Index finger
    INDEX_METACARPAL = auto()
    INDEX_PROXIMAL = auto()
    INDEX_INTERMEDIATE = auto()
    INDEX_DISTAL = auto()
    INDEX_TIP = auto()

    # Middle finger
    MIDDLE_METACARPAL = auto()
    MIDDLE_PROXIMAL = auto()
    MIDDLE_INTERMEDIATE = auto()
    MIDDLE_DISTAL = auto()
    MIDDLE_TIP = auto()

    # Ring finger
    RING_METACARPAL = auto()
    RING_PROXIMAL = auto()
    RING_INTERMEDIATE = auto()
    RING_DISTAL = auto()
    RING_TIP = auto()

    # Pinky finger
    PINKY_METACARPAL = auto()
    PINKY_PROXIMAL = auto()
    PINKY_INTERMEDIATE = auto()
    PINKY_DISTAL = auto()
    PINKY_TIP = auto()


@dataclass(slots=True)
class JointPose:
    """Pose of a hand joint."""
    joint: HandJoint
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # quaternion
    radius: float  # Joint collision radius


class XRHand(InputDevice):
    """XR hand tracking device."""
    __slots__ = ('_joint_poses', '_is_tracking', '_pinch_strength', '_grip_strength')

    def __init__(self, name: str = "XR Hand", device_id: int = 0):
        """Initialize the XR hand.

        Args:
            name: Device name
            device_id: Unique device identifier
        """
        super().__init__(InputDeviceType.XR_HAND, name, device_id)
        self._joint_poses: dict[HandJoint, JointPose] = {}
        self._is_tracking: bool = False
        self._pinch_strength: float = 0.0
        self._grip_strength: float = 0.0

    @property
    def joint_poses(self) -> dict[HandJoint, JointPose]:
        """Get all joint poses.

        Returns:
            Dictionary mapping joints to their poses
        """
        return self._joint_poses.copy()

    @property
    def is_tracking(self) -> bool:
        """Check if hand is currently being tracked.

        Returns:
            True if hand tracking is active
        """
        return self._is_tracking

    @property
    def pinch_strength(self) -> float:
        """Get the pinch gesture strength.

        Returns:
            Value from 0.0 (not pinching) to 1.0 (full pinch)
        """
        return self._pinch_strength

    @property
    def grip_strength(self) -> float:
        """Get the grip gesture strength.

        Returns:
            Value from 0.0 (open) to 1.0 (full grip)
        """
        return self._grip_strength

    def get_joint_pose(self, joint: HandJoint) -> JointPose | None:
        """Get the pose of a specific joint.

        Args:
            joint: The joint to query

        Returns:
            JointPose if available, None otherwise
        """
        return self._joint_poses.get(joint)

    def update(self, events: list[InputEvent]) -> None:
        """Update hand tracking state with new events.

        Args:
            events: List of XR hand tracking events
        """
        for event in events:
            if event.event_type == 'xr_hand_tracking':
                self._is_tracking = event.data.get('tracking', False)

                joints_data = event.data.get('joints', {})
                for joint_name, joint_data in joints_data.items():
                    try:
                        joint = HandJoint[joint_name.upper()]
                        pose = JointPose(
                            joint=joint,
                            position=tuple(float(x) for x in joint_data['position']),
                            orientation=tuple(float(x) for x in joint_data['orientation']),
                            radius=float(joint_data.get('radius', 0.01))
                        )
                        self._joint_poses[joint] = pose
                    except (KeyError, ValueError):
                        pass  # Invalid joint name

            elif event.event_type == 'xr_hand_gesture':
                pinch = event.data.get('pinch', self._pinch_strength)
                self._pinch_strength = max(0.0, min(1.0, float(pinch)))

                grip = event.data.get('grip', self._grip_strength)
                self._grip_strength = max(0.0, min(1.0, float(grip)))
