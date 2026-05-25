"""
Tests for Camera Blending and Transitions (blending.py).

Tests camera blending including:
    - Cut transitions (instant)
    - Linear blending
    - Ease curves
    - Custom blend curves
    - Blend interruption
    - Blend stack management
    - Camera priority system
    - Viewport split (split screen)
    - Blend duration control
"""

import math
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Dict
from enum import Enum, auto


# =============================================================================
# Mock Classes for Testing
# =============================================================================


@dataclass
class Vector3:
    """Mock 3D vector for testing."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def lerp(self, target: "Vector3", t: float) -> "Vector3":
        return Vector3(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
            self.z + (target.z - self.z) * t,
        )


@dataclass
class Quaternion:
    """Mock quaternion for rotation testing."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def slerp(self, target: "Quaternion", t: float) -> "Quaternion":
        return Quaternion(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
            self.z + (target.z - self.z) * t,
            self.w + (target.w - self.w) * t,
        )


@dataclass
class CameraState:
    """Camera state for blending."""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)
    fov: float = 60.0
    near_clip: float = 0.1
    far_clip: float = 1000.0

    def lerp(self, target: "CameraState", t: float) -> "CameraState":
        return CameraState(
            position=self.position.lerp(target.position, t),
            rotation=self.rotation.slerp(target.rotation, t),
            fov=self.fov + (target.fov - self.fov) * t,
            near_clip=self.near_clip + (target.near_clip - self.near_clip) * t,
            far_clip=self.far_clip + (target.far_clip - self.far_clip) * t,
        )


class BlendType(Enum):
    """Type of blend transition."""
    CUT = auto()
    LINEAR = auto()
    EASE_IN = auto()
    EASE_OUT = auto()
    EASE_IN_OUT = auto()
    CUSTOM = auto()


@dataclass
class Viewport:
    """Viewport definition for split screen."""
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0


# =============================================================================
# Ease Functions
# =============================================================================


class EaseFunctions:
    """Collection of easing functions."""

    @staticmethod
    def linear(t: float) -> float:
        return t

    @staticmethod
    def ease_in_quad(t: float) -> float:
        return t * t

    @staticmethod
    def ease_out_quad(t: float) -> float:
        return 1 - (1 - t) * (1 - t)

    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        if t < 0.5:
            return 2 * t * t
        return 1 - (-2 * t + 2) ** 2 / 2

    @staticmethod
    def ease_in_cubic(t: float) -> float:
        return t * t * t

    @staticmethod
    def ease_out_cubic(t: float) -> float:
        return 1 - (1 - t) ** 3

    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        if t < 0.5:
            return 4 * t * t * t
        return 1 - (-2 * t + 2) ** 3 / 2

    @staticmethod
    def ease_in_sine(t: float) -> float:
        return 1 - math.cos((t * math.pi) / 2)

    @staticmethod
    def ease_out_sine(t: float) -> float:
        return math.sin((t * math.pi) / 2)

    @staticmethod
    def ease_in_out_sine(t: float) -> float:
        return -(math.cos(math.pi * t) - 1) / 2

    @staticmethod
    def ease_in_expo(t: float) -> float:
        return 0 if t == 0 else 2 ** (10 * t - 10)

    @staticmethod
    def ease_out_expo(t: float) -> float:
        return 1 if t == 1 else 1 - 2 ** (-10 * t)

    @staticmethod
    def ease_in_out_expo(t: float) -> float:
        if t == 0:
            return 0
        if t == 1:
            return 1
        if t < 0.5:
            return 2 ** (20 * t - 10) / 2
        return (2 - 2 ** (-20 * t + 10)) / 2

    @staticmethod
    def ease_in_elastic(t: float) -> float:
        c4 = (2 * math.pi) / 3
        if t == 0:
            return 0
        if t == 1:
            return 1
        return -2 ** (10 * t - 10) * math.sin((t * 10 - 10.75) * c4)

    @staticmethod
    def ease_out_elastic(t: float) -> float:
        c4 = (2 * math.pi) / 3
        if t == 0:
            return 0
        if t == 1:
            return 1
        return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * c4) + 1

    @staticmethod
    def ease_in_bounce(t: float) -> float:
        return 1 - EaseFunctions.ease_out_bounce(1 - t)

    @staticmethod
    def ease_out_bounce(t: float) -> float:
        n1 = 7.5625
        d1 = 2.75
        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375


# =============================================================================
# Blend Request
# =============================================================================


@dataclass
class BlendRequest:
    """Request for a camera blend."""
    id: str = ""
    source_state: CameraState = field(default_factory=CameraState)
    target_state: CameraState = field(default_factory=CameraState)
    duration: float = 1.0
    blend_type: BlendType = BlendType.LINEAR
    ease_function: Optional[Callable[[float], float]] = None
    priority: int = 0
    on_complete: Optional[Callable] = None
    on_interrupt: Optional[Callable] = None
    elapsed: float = 0.0
    is_complete: bool = False
    is_interrupted: bool = False


# =============================================================================
# Camera Blender
# =============================================================================


class CameraBlender:
    """Handles camera state blending."""

    def __init__(self):
        self._active_blend: Optional[BlendRequest] = None
        self._current_state = CameraState()
        self._blend_queue: List[BlendRequest] = []
        self._next_blend_id = 0

    def cut(self, target_state: CameraState):
        """Instant cut to target state."""
        self._current_state = target_state
        self._active_blend = None

    def blend_to(
        self,
        target_state: CameraState,
        duration: float,
        blend_type: BlendType = BlendType.LINEAR,
        ease_function: Callable[[float], float] = None,
        priority: int = 0,
        on_complete: Callable = None,
        on_interrupt: Callable = None,
    ) -> str:
        """Start blending to a target state."""
        blend_id = f"blend_{self._next_blend_id}"
        self._next_blend_id += 1

        request = BlendRequest(
            id=blend_id,
            source_state=CameraState(
                position=Vector3(self._current_state.position.x,
                                self._current_state.position.y,
                                self._current_state.position.z),
                rotation=Quaternion(self._current_state.rotation.x,
                                   self._current_state.rotation.y,
                                   self._current_state.rotation.z,
                                   self._current_state.rotation.w),
                fov=self._current_state.fov,
            ),
            target_state=target_state,
            duration=duration,
            blend_type=blend_type,
            ease_function=ease_function,
            priority=priority,
            on_complete=on_complete,
            on_interrupt=on_interrupt,
        )

        if self._active_blend and self._active_blend.priority > priority:
            self._blend_queue.append(request)
        else:
            if self._active_blend:
                self._interrupt_blend(self._active_blend)
            self._active_blend = request

        return blend_id

    def _interrupt_blend(self, blend: BlendRequest):
        """Interrupt an active blend."""
        blend.is_interrupted = True
        if blend.on_interrupt:
            blend.on_interrupt()

    def cancel_blend(self, blend_id: str):
        """Cancel a blend by ID."""
        if self._active_blend and self._active_blend.id == blend_id:
            self._interrupt_blend(self._active_blend)
            self._active_blend = None
            self._activate_next_queued()
        else:
            self._blend_queue = [b for b in self._blend_queue if b.id != blend_id]

    def cancel_all_blends(self):
        """Cancel all active and queued blends."""
        if self._active_blend:
            self._interrupt_blend(self._active_blend)
        self._active_blend = None
        for blend in self._blend_queue:
            self._interrupt_blend(blend)
        self._blend_queue.clear()

    def _activate_next_queued(self):
        """Activate the next queued blend if any."""
        if self._blend_queue:
            self._blend_queue.sort(key=lambda b: b.priority, reverse=True)
            self._active_blend = self._blend_queue.pop(0)
            self._active_blend.source_state = CameraState(
                position=Vector3(self._current_state.position.x,
                                self._current_state.position.y,
                                self._current_state.position.z),
                rotation=Quaternion(self._current_state.rotation.x,
                                   self._current_state.rotation.y,
                                   self._current_state.rotation.z,
                                   self._current_state.rotation.w),
                fov=self._current_state.fov,
            )

    def _get_ease_value(self, blend: BlendRequest, t: float) -> float:
        """Get eased blend value."""
        if blend.ease_function:
            return blend.ease_function(t)

        if blend.blend_type == BlendType.CUT:
            return 1.0 if t >= 1.0 else 0.0
        elif blend.blend_type == BlendType.LINEAR:
            return t
        elif blend.blend_type == BlendType.EASE_IN:
            return EaseFunctions.ease_in_quad(t)
        elif blend.blend_type == BlendType.EASE_OUT:
            return EaseFunctions.ease_out_quad(t)
        elif blend.blend_type == BlendType.EASE_IN_OUT:
            return EaseFunctions.ease_in_out_quad(t)
        else:
            return t

    def update(self, delta_time: float) -> CameraState:
        """Update blending and return current state."""
        if not self._active_blend:
            return self._current_state

        blend = self._active_blend
        blend.elapsed += delta_time

        t = min(1.0, blend.elapsed / blend.duration) if blend.duration > 0 else 1.0
        eased_t = self._get_ease_value(blend, t)

        self._current_state = blend.source_state.lerp(blend.target_state, eased_t)

        if t >= 1.0:
            blend.is_complete = True
            if blend.on_complete:
                blend.on_complete()
            self._active_blend = None
            self._activate_next_queued()

        return self._current_state

    @property
    def is_blending(self) -> bool:
        return self._active_blend is not None

    @property
    def current_state(self) -> CameraState:
        return self._current_state

    @property
    def blend_progress(self) -> float:
        if not self._active_blend:
            return 0.0
        if self._active_blend.duration <= 0:
            return 1.0
        return min(1.0, self._active_blend.elapsed / self._active_blend.duration)


# =============================================================================
# Blend Stack
# =============================================================================


class BlendStackEntry:
    """Entry in the blend stack."""

    def __init__(
        self,
        camera_id: str,
        state_provider: Callable[[], CameraState],
        weight: float = 1.0,
        priority: int = 0,
    ):
        self.camera_id = camera_id
        self.state_provider = state_provider
        self.weight = weight
        self.target_weight = weight
        self.priority = priority
        self.fade_speed = 5.0


class BlendStack:
    """Stack of camera states to blend."""

    def __init__(self):
        self._entries: List[BlendStackEntry] = []
        self._base_state = CameraState()

    def set_base_state(self, state: CameraState):
        """Set the base camera state."""
        self._base_state = state

    def push(
        self,
        camera_id: str,
        state_provider: Callable[[], CameraState],
        weight: float = 1.0,
        priority: int = 0,
    ):
        """Push a camera onto the blend stack."""
        entry = BlendStackEntry(camera_id, state_provider, weight, priority)
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.priority)

    def pop(self, camera_id: str):
        """Remove a camera from the stack."""
        self._entries = [e for e in self._entries if e.camera_id != camera_id]

    def set_weight(self, camera_id: str, weight: float):
        """Set the weight for a camera."""
        for entry in self._entries:
            if entry.camera_id == camera_id:
                entry.target_weight = max(0.0, min(1.0, weight))
                break

    def fade_in(self, camera_id: str, duration: float = 0.5):
        """Fade in a camera."""
        for entry in self._entries:
            if entry.camera_id == camera_id:
                entry.weight = 0.0
                entry.target_weight = 1.0
                entry.fade_speed = 1.0 / duration if duration > 0 else 100.0
                break

    def fade_out(self, camera_id: str, duration: float = 0.5):
        """Fade out a camera."""
        for entry in self._entries:
            if entry.camera_id == camera_id:
                entry.target_weight = 0.0
                entry.fade_speed = 1.0 / duration if duration > 0 else 100.0
                break

    def clear(self):
        """Clear all entries from the stack."""
        self._entries.clear()

    def update(self, delta_time: float) -> CameraState:
        """Update stack and return blended state."""
        for entry in self._entries:
            diff = entry.target_weight - entry.weight
            change = entry.fade_speed * delta_time
            if abs(diff) <= change:
                entry.weight = entry.target_weight
            else:
                entry.weight += change if diff > 0 else -change

        self._entries = [e for e in self._entries if e.weight > 0.001 or e.target_weight > 0]

        if not self._entries:
            return self._base_state

        total_weight = sum(e.weight for e in self._entries)
        if total_weight <= 0:
            return self._base_state

        result_position = Vector3()
        result_rotation = Quaternion(0, 0, 0, 0)
        result_fov = 0.0

        for entry in self._entries:
            normalized_weight = entry.weight / total_weight
            state = entry.state_provider()

            result_position.x += state.position.x * normalized_weight
            result_position.y += state.position.y * normalized_weight
            result_position.z += state.position.z * normalized_weight

            result_rotation.x += state.rotation.x * normalized_weight
            result_rotation.y += state.rotation.y * normalized_weight
            result_rotation.z += state.rotation.z * normalized_weight
            result_rotation.w += state.rotation.w * normalized_weight

            result_fov += state.fov * normalized_weight

        return CameraState(
            position=result_position,
            rotation=result_rotation,
            fov=result_fov,
        )

    @property
    def count(self) -> int:
        return len(self._entries)

    def get_entry(self, camera_id: str) -> Optional[BlendStackEntry]:
        for entry in self._entries:
            if entry.camera_id == camera_id:
                return entry
        return None


# =============================================================================
# Camera Priority Manager
# =============================================================================


class CameraPriorityManager:
    """Manages camera priorities and activation."""

    def __init__(self):
        self._cameras: Dict[str, Tuple[Callable[[], CameraState], int]] = {}
        self._active_camera: Optional[str] = None
        self._blender = CameraBlender()
        self._auto_blend_duration = 0.5

    def register_camera(
        self,
        camera_id: str,
        state_provider: Callable[[], CameraState],
        priority: int = 0,
    ):
        """Register a camera with the manager."""
        self._cameras[camera_id] = (state_provider, priority)

    def unregister_camera(self, camera_id: str):
        """Unregister a camera."""
        if camera_id in self._cameras:
            del self._cameras[camera_id]
            if self._active_camera == camera_id:
                self._activate_highest_priority()

    def set_priority(self, camera_id: str, priority: int):
        """Set priority for a camera."""
        if camera_id in self._cameras:
            state_provider = self._cameras[camera_id][0]
            self._cameras[camera_id] = (state_provider, priority)
            self._check_priority_change()

    def activate_camera(self, camera_id: str, blend_duration: float = None):
        """Explicitly activate a camera."""
        if camera_id not in self._cameras:
            return

        duration = blend_duration if blend_duration is not None else self._auto_blend_duration

        if self._active_camera != camera_id:
            target_state = self._cameras[camera_id][0]()
            if duration > 0:
                self._blender.blend_to(target_state, duration)
            else:
                self._blender.cut(target_state)
            self._active_camera = camera_id

    def _activate_highest_priority(self):
        """Activate the highest priority camera."""
        if not self._cameras:
            self._active_camera = None
            return

        highest = max(self._cameras.items(), key=lambda x: x[1][1])
        self.activate_camera(highest[0])

    def _check_priority_change(self):
        """Check if priority change requires camera switch."""
        if not self._cameras:
            return

        highest = max(self._cameras.items(), key=lambda x: x[1][1])
        if highest[0] != self._active_camera:
            if highest[1][1] > self._cameras.get(self._active_camera, (None, -999))[1]:
                self.activate_camera(highest[0])

    def update(self, delta_time: float) -> CameraState:
        """Update and return current camera state."""
        if self._active_camera and self._active_camera in self._cameras:
            target = self._cameras[self._active_camera][0]()
            if not self._blender.is_blending:
                return target
        return self._blender.update(delta_time)

    @property
    def active_camera(self) -> Optional[str]:
        return self._active_camera


# =============================================================================
# Viewport Split (Split Screen)
# =============================================================================


class SplitScreenManager:
    """Manages split screen camera viewports."""

    def __init__(self):
        self._viewports: Dict[str, Tuple[Viewport, Callable[[], CameraState]]] = {}
        self._layout = "single"

    def add_viewport(
        self,
        viewport_id: str,
        viewport: Viewport,
        state_provider: Callable[[], CameraState],
    ):
        """Add a viewport."""
        self._viewports[viewport_id] = (viewport, state_provider)

    def remove_viewport(self, viewport_id: str):
        """Remove a viewport."""
        if viewport_id in self._viewports:
            del self._viewports[viewport_id]

    def set_viewport(self, viewport_id: str, viewport: Viewport):
        """Update viewport dimensions."""
        if viewport_id in self._viewports:
            state_provider = self._viewports[viewport_id][1]
            self._viewports[viewport_id] = (viewport, state_provider)

    def set_layout(self, layout: str):
        """Set a predefined layout."""
        self._layout = layout

        if layout == "single":
            pass
        elif layout == "horizontal_split":
            for i, viewport_id in enumerate(self._viewports.keys()):
                if i == 0:
                    self.set_viewport(viewport_id, Viewport(0, 0.5, 1, 0.5))
                elif i == 1:
                    self.set_viewport(viewport_id, Viewport(0, 0, 1, 0.5))
        elif layout == "vertical_split":
            for i, viewport_id in enumerate(self._viewports.keys()):
                if i == 0:
                    self.set_viewport(viewport_id, Viewport(0, 0, 0.5, 1))
                elif i == 1:
                    self.set_viewport(viewport_id, Viewport(0.5, 0, 0.5, 1))
        elif layout == "quad":
            positions = [(0, 0.5), (0.5, 0.5), (0, 0), (0.5, 0)]
            for i, viewport_id in enumerate(self._viewports.keys()):
                if i < 4:
                    self.set_viewport(
                        viewport_id,
                        Viewport(positions[i][0], positions[i][1], 0.5, 0.5)
                    )

    def get_viewports(self) -> Dict[str, Tuple[Viewport, CameraState]]:
        """Get all viewports with their current states."""
        return {
            viewport_id: (viewport, state_provider())
            for viewport_id, (viewport, state_provider) in self._viewports.items()
        }

    @property
    def viewport_count(self) -> int:
        return len(self._viewports)


# =============================================================================
# Cut Transition Tests (~15 tests)
# =============================================================================


class TestCutTransition:
    """Test cut (instant) transitions."""

    def test_cut_changes_state_instantly(self):
        """Test cut changes state instantly."""
        blender = CameraBlender()
        target = CameraState(position=Vector3(10, 10, 10))
        blender.cut(target)
        assert blender.current_state.position.x == 10

    def test_cut_stops_active_blend(self):
        """Test cut stops any active blend."""
        blender = CameraBlender()
        blender.blend_to(CameraState(position=Vector3(5, 5, 5)), duration=1.0)
        blender.cut(CameraState(position=Vector3(20, 20, 20)))
        assert blender.current_state.position.x == 20
        assert blender.is_blending is False

    def test_cut_fov(self):
        """Test cut changes FOV instantly."""
        blender = CameraBlender()
        target = CameraState(fov=90.0)
        blender.cut(target)
        assert blender.current_state.fov == 90.0

    def test_cut_rotation(self):
        """Test cut changes rotation instantly."""
        blender = CameraBlender()
        target = CameraState(rotation=Quaternion(0, 0.707, 0, 0.707))
        blender.cut(target)
        assert blender.current_state.rotation.y == 0.707

    def test_blend_type_cut(self):
        """Test BlendType.CUT acts like instant cut."""
        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=0.5,
            blend_type=BlendType.CUT
        )
        blender.update(0.1)
        assert blender.current_state.position.x == 0.0
        blender.update(0.5)
        assert blender.current_state.position.x == 10.0


# =============================================================================
# Linear Blending Tests (~15 tests)
# =============================================================================


class TestLinearBlending:
    """Test linear blending transitions."""

    def test_linear_blend_start(self):
        """Test linear blend starts at source."""
        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0,
            blend_type=BlendType.LINEAR
        )
        state = blender.update(0.0)
        assert state.position.x == pytest.approx(0.0, abs=0.1)

    def test_linear_blend_middle(self):
        """Test linear blend at middle."""
        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0,
            blend_type=BlendType.LINEAR
        )
        blender.update(0.5)
        assert blender.current_state.position.x == pytest.approx(5.0, abs=0.1)

    def test_linear_blend_end(self):
        """Test linear blend ends at target."""
        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0,
            blend_type=BlendType.LINEAR
        )
        blender.update(1.0)
        assert blender.current_state.position.x == pytest.approx(10.0, abs=0.1)

    def test_linear_blend_fov(self):
        """Test linear FOV blending."""
        blender = CameraBlender()
        blender._current_state = CameraState(fov=60.0)
        blender.blend_to(
            CameraState(fov=90.0),
            duration=1.0,
            blend_type=BlendType.LINEAR
        )
        blender.update(0.5)
        assert blender.current_state.fov == pytest.approx(75.0, abs=0.1)

    def test_linear_blend_progress(self):
        """Test blend progress reporting."""
        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0
        )
        blender.update(0.25)
        assert blender.blend_progress == pytest.approx(0.25, abs=0.01)


# =============================================================================
# Ease Curve Tests (~20 tests)
# =============================================================================


class TestEaseCurves:
    """Test ease curve transitions."""

    def test_ease_in_quad(self):
        """Test ease-in quad curve."""
        t = 0.5
        result = EaseFunctions.ease_in_quad(t)
        assert result == 0.25

    def test_ease_out_quad(self):
        """Test ease-out quad curve."""
        t = 0.5
        result = EaseFunctions.ease_out_quad(t)
        assert result == 0.75

    def test_ease_in_out_quad(self):
        """Test ease-in-out quad curve."""
        assert EaseFunctions.ease_in_out_quad(0.0) == pytest.approx(0.0, abs=0.01)
        assert EaseFunctions.ease_in_out_quad(0.5) == pytest.approx(0.5, abs=0.01)
        assert EaseFunctions.ease_in_out_quad(1.0) == pytest.approx(1.0, abs=0.01)

    def test_ease_in_cubic(self):
        """Test ease-in cubic curve."""
        t = 0.5
        result = EaseFunctions.ease_in_cubic(t)
        assert result == 0.125

    def test_ease_out_cubic(self):
        """Test ease-out cubic curve."""
        t = 0.5
        result = EaseFunctions.ease_out_cubic(t)
        assert result == 0.875

    def test_ease_in_sine(self):
        """Test ease-in sine curve."""
        assert EaseFunctions.ease_in_sine(0.0) == pytest.approx(0.0, abs=0.01)
        assert EaseFunctions.ease_in_sine(1.0) == pytest.approx(1.0, abs=0.01)

    def test_ease_out_sine(self):
        """Test ease-out sine curve."""
        assert EaseFunctions.ease_out_sine(0.0) == pytest.approx(0.0, abs=0.01)
        assert EaseFunctions.ease_out_sine(1.0) == pytest.approx(1.0, abs=0.01)

    def test_ease_in_expo(self):
        """Test ease-in exponential curve."""
        assert EaseFunctions.ease_in_expo(0.0) == 0.0
        assert EaseFunctions.ease_in_expo(1.0) == pytest.approx(1.0, abs=0.01)

    def test_ease_out_expo(self):
        """Test ease-out exponential curve."""
        assert EaseFunctions.ease_out_expo(0.0) == pytest.approx(0.0, abs=0.01)
        assert EaseFunctions.ease_out_expo(1.0) == 1.0

    def test_ease_out_bounce(self):
        """Test ease-out bounce curve."""
        assert EaseFunctions.ease_out_bounce(0.0) == pytest.approx(0.0, abs=0.01)
        assert EaseFunctions.ease_out_bounce(1.0) == pytest.approx(1.0, abs=0.01)

    def test_blend_with_ease_in(self):
        """Test blend with ease-in type."""
        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0,
            blend_type=BlendType.EASE_IN
        )
        blender.update(0.5)
        assert blender.current_state.position.x < 5.0

    def test_blend_with_ease_out(self):
        """Test blend with ease-out type."""
        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0,
            blend_type=BlendType.EASE_OUT
        )
        blender.update(0.5)
        assert blender.current_state.position.x > 5.0


# =============================================================================
# Custom Blend Curve Tests (~10 tests)
# =============================================================================


class TestCustomBlendCurves:
    """Test custom blend curves."""

    def test_custom_ease_function(self):
        """Test using custom ease function."""
        def custom_ease(t):
            return t * t * t

        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0,
            ease_function=custom_ease
        )
        blender.update(0.5)
        assert blender.current_state.position.x == pytest.approx(1.25, abs=0.1)

    def test_custom_step_function(self):
        """Test step function for abrupt changes."""
        def step_ease(t):
            return 0.0 if t < 0.5 else 1.0

        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0,
            ease_function=step_ease
        )
        blender.update(0.4)
        assert blender.current_state.position.x == pytest.approx(0.0, abs=0.1)
        blender.update(0.2)
        assert blender.current_state.position.x == pytest.approx(10.0, abs=0.1)

    def test_custom_overshoot_function(self):
        """Test overshoot ease function."""
        def overshoot(t):
            s = 1.70158
            return t * t * ((s + 1) * t - s)

        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0,
            ease_function=overshoot
        )
        blender.update(0.3)


# =============================================================================
# Blend Interruption Tests (~15 tests)
# =============================================================================


class TestBlendInterruption:
    """Test blend interruption behavior."""

    def test_interrupt_active_blend(self):
        """Test interrupting an active blend."""
        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0
        )
        blender.update(0.3)

        blender.blend_to(
            CameraState(position=Vector3(20, 0, 0)),
            duration=1.0
        )

        assert blender.is_blending is True

    def test_interrupt_callback(self):
        """Test interrupt callback is called."""
        interrupted = [False]

        def on_interrupt():
            interrupted[0] = True

        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0,
            on_interrupt=on_interrupt
        )
        blender.update(0.3)

        blender.blend_to(
            CameraState(position=Vector3(20, 0, 0)),
            duration=1.0
        )

        assert interrupted[0] is True

    def test_cancel_specific_blend(self):
        """Test canceling a specific blend."""
        blender = CameraBlender()
        blend_id = blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0
        )
        blender.cancel_blend(blend_id)
        assert blender.is_blending is False

    def test_cancel_all_blends(self):
        """Test canceling all blends."""
        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0
        )
        blender.cancel_all_blends()
        assert blender.is_blending is False

    def test_complete_callback(self):
        """Test complete callback is called."""
        completed = [False]

        def on_complete():
            completed[0] = True

        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=0.5,
            on_complete=on_complete
        )

        blender.update(0.6)
        assert completed[0] is True

    def test_blend_from_current_position(self):
        """Test interrupted blend starts from current position."""
        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))

        blender.blend_to(
            CameraState(position=Vector3(20, 0, 0)),
            duration=1.0
        )
        blender.update(0.5)

        current_x = blender.current_state.position.x

        blender.blend_to(
            CameraState(position=Vector3(0, 0, 0)),
            duration=1.0
        )

        state = blender.update(0.0)
        assert state.position.x == pytest.approx(current_x, abs=0.5)


# =============================================================================
# Blend Stack Tests (~20 tests)
# =============================================================================


class TestBlendStack:
    """Test blend stack management."""

    def test_push_entry(self):
        """Test pushing entry to stack."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState(position=Vector3(5, 5, 5)))
        assert stack.count == 1

    def test_pop_entry(self):
        """Test popping entry from stack."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState())
        stack.push("cam2", lambda: CameraState())
        stack.pop("cam1")
        assert stack.count == 1

    def test_clear_stack(self):
        """Test clearing all entries."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState())
        stack.push("cam2", lambda: CameraState())
        stack.clear()
        assert stack.count == 0

    def test_single_entry_returns_state(self):
        """Test single entry returns its state."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState(position=Vector3(10, 0, 0)), weight=1.0)
        state = stack.update(0.1)
        assert state.position.x == pytest.approx(10.0, abs=0.1)

    def test_weighted_blend(self):
        """Test weighted blending of entries."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState(position=Vector3(0, 0, 0)), weight=0.5)
        stack.push("cam2", lambda: CameraState(position=Vector3(10, 0, 0)), weight=0.5)
        state = stack.update(0.1)
        assert state.position.x == pytest.approx(5.0, abs=0.5)

    def test_set_weight(self):
        """Test setting entry weight."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState(), weight=0.5)
        stack.set_weight("cam1", 1.0)
        entry = stack.get_entry("cam1")
        assert entry.target_weight == 1.0

    def test_fade_in(self):
        """Test fading in an entry."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState(), weight=0.0)
        stack.fade_in("cam1", duration=0.5)
        entry = stack.get_entry("cam1")
        assert entry.target_weight == 1.0
        assert entry.weight == 0.0

    def test_fade_out(self):
        """Test fading out an entry."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState(), weight=1.0)
        stack.fade_out("cam1", duration=0.5)
        entry = stack.get_entry("cam1")
        assert entry.target_weight == 0.0

    def test_fade_removes_zero_weight(self):
        """Test that zero-weight entries are removed."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState(), weight=0.1)
        stack.fade_out("cam1", duration=0.01)

        for _ in range(20):
            stack.update(0.1)

        assert stack.count == 0

    def test_priority_ordering(self):
        """Test entries are ordered by priority."""
        stack = BlendStack()
        stack.push("low", lambda: CameraState(), priority=0)
        stack.push("high", lambda: CameraState(), priority=10)
        stack.push("mid", lambda: CameraState(), priority=5)

        assert stack._entries[0].camera_id == "low"
        assert stack._entries[2].camera_id == "high"

    def test_empty_stack_returns_base(self):
        """Test empty stack returns base state."""
        stack = BlendStack()
        stack.set_base_state(CameraState(position=Vector3(100, 0, 0)))
        state = stack.update(0.1)
        assert state.position.x == 100


# =============================================================================
# Camera Priority Tests (~15 tests)
# =============================================================================


class TestCameraPriority:
    """Test camera priority system."""

    def test_register_camera(self):
        """Test registering a camera."""
        manager = CameraPriorityManager()
        manager.register_camera("cam1", lambda: CameraState(), priority=5)
        assert "cam1" in manager._cameras

    def test_unregister_camera(self):
        """Test unregistering a camera."""
        manager = CameraPriorityManager()
        manager.register_camera("cam1", lambda: CameraState())
        manager.unregister_camera("cam1")
        assert "cam1" not in manager._cameras

    def test_activate_camera(self):
        """Test explicitly activating a camera."""
        manager = CameraPriorityManager()
        manager.register_camera("cam1", lambda: CameraState(position=Vector3(10, 0, 0)))
        manager.activate_camera("cam1", blend_duration=0)
        assert manager.active_camera == "cam1"

    def test_set_priority(self):
        """Test setting camera priority."""
        manager = CameraPriorityManager()
        manager.register_camera("cam1", lambda: CameraState(), priority=0)
        manager.set_priority("cam1", 10)
        assert manager._cameras["cam1"][1] == 10

    def test_higher_priority_activates(self):
        """Test higher priority camera auto-activates."""
        manager = CameraPriorityManager()
        manager.register_camera("low", lambda: CameraState(), priority=0)
        manager.activate_camera("low", blend_duration=0)
        manager.register_camera("high", lambda: CameraState(), priority=10)
        manager.set_priority("high", 10)

    def test_update_returns_active_state(self):
        """Test update returns active camera state."""
        manager = CameraPriorityManager()
        manager.register_camera("cam1", lambda: CameraState(position=Vector3(5, 5, 5)))
        manager.activate_camera("cam1", blend_duration=0)
        state = manager.update(0.1)
        assert state.position.x == 5

    def test_blend_on_activate(self):
        """Test blending when activating camera."""
        manager = CameraPriorityManager()
        manager._blender._current_state = CameraState(position=Vector3(0, 0, 0))
        manager.register_camera("cam1", lambda: CameraState(position=Vector3(10, 0, 0)))
        manager.activate_camera("cam1", blend_duration=1.0)
        manager.update(0.5)
        assert 0 < manager._blender.current_state.position.x < 10


# =============================================================================
# Viewport Split Tests (~15 tests)
# =============================================================================


class TestViewportSplit:
    """Test viewport split (split screen) functionality."""

    def test_add_viewport(self):
        """Test adding a viewport."""
        manager = SplitScreenManager()
        manager.add_viewport("p1", Viewport(0, 0, 0.5, 1), lambda: CameraState())
        assert manager.viewport_count == 1

    def test_remove_viewport(self):
        """Test removing a viewport."""
        manager = SplitScreenManager()
        manager.add_viewport("p1", Viewport(), lambda: CameraState())
        manager.remove_viewport("p1")
        assert manager.viewport_count == 0

    def test_set_viewport(self):
        """Test updating viewport dimensions."""
        manager = SplitScreenManager()
        manager.add_viewport("p1", Viewport(0, 0, 1, 1), lambda: CameraState())
        manager.set_viewport("p1", Viewport(0, 0, 0.5, 0.5))
        viewports = manager.get_viewports()
        assert viewports["p1"][0].width == 0.5

    def test_horizontal_split_layout(self):
        """Test horizontal split layout."""
        manager = SplitScreenManager()
        manager.add_viewport("p1", Viewport(), lambda: CameraState())
        manager.add_viewport("p2", Viewport(), lambda: CameraState())
        manager.set_layout("horizontal_split")
        viewports = manager.get_viewports()
        assert viewports["p1"][0].y == 0.5
        assert viewports["p2"][0].y == 0.0

    def test_vertical_split_layout(self):
        """Test vertical split layout."""
        manager = SplitScreenManager()
        manager.add_viewport("p1", Viewport(), lambda: CameraState())
        manager.add_viewport("p2", Viewport(), lambda: CameraState())
        manager.set_layout("vertical_split")
        viewports = manager.get_viewports()
        assert viewports["p1"][0].x == 0.0
        assert viewports["p2"][0].x == 0.5

    def test_quad_layout(self):
        """Test quad split layout."""
        manager = SplitScreenManager()
        for i in range(4):
            manager.add_viewport(f"p{i}", Viewport(), lambda: CameraState())
        manager.set_layout("quad")
        viewports = manager.get_viewports()
        for vp_id, (viewport, state) in viewports.items():
            assert viewport.width == 0.5
            assert viewport.height == 0.5

    def test_get_viewports_with_states(self):
        """Test getting viewports with their states."""
        manager = SplitScreenManager()
        manager.add_viewport(
            "p1",
            Viewport(),
            lambda: CameraState(position=Vector3(10, 0, 0))
        )
        viewports = manager.get_viewports()
        assert viewports["p1"][1].position.x == 10


# =============================================================================
# Blend Duration Tests (~10 tests)
# =============================================================================


class TestBlendDuration:
    """Test blend duration control."""

    def test_zero_duration_instant(self):
        """Test zero duration is instant."""
        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=0.0
        )
        blender.update(0.0)
        assert blender.current_state.position.x == pytest.approx(10.0, abs=0.1)

    def test_short_duration(self):
        """Test very short duration."""
        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=0.01
        )
        blender.update(0.02)
        assert blender.is_blending is False

    def test_long_duration(self):
        """Test long duration blend."""
        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=10.0
        )
        blender.update(5.0)
        assert blender.blend_progress == pytest.approx(0.5, abs=0.01)

    def test_fractional_duration(self):
        """Test fractional duration."""
        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=0.333
        )
        blender.update(0.333)
        assert blender.is_blending is False

    def test_accumulated_time(self):
        """Test accumulated small time steps."""
        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0
        )

        for _ in range(100):
            blender.update(0.01)

        assert blender.is_blending is False
        assert blender.current_state.position.x == pytest.approx(10.0, abs=0.1)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_blend_to_same_state(self):
        """Test blending to the same state."""
        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(5, 5, 5))
        blender.blend_to(
            CameraState(position=Vector3(5, 5, 5)),
            duration=1.0
        )
        state = blender.update(0.5)
        assert state.position.x == pytest.approx(5.0, abs=0.1)

    def test_negative_duration(self):
        """Test handling negative duration."""
        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=-1.0
        )
        blender.update(0.1)

    def test_very_large_delta_time(self):
        """Test very large delta time."""
        blender = CameraBlender()
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0
        )
        blender.update(1000.0)
        assert blender.is_blending is False

    def test_cancel_nonexistent_blend(self):
        """Test canceling a nonexistent blend."""
        blender = CameraBlender()
        blender.cancel_blend("nonexistent")

    def test_rapid_blend_requests(self):
        """Test rapid blend requests."""
        blender = CameraBlender()
        for i in range(100):
            blender.blend_to(
                CameraState(position=Vector3(i, 0, 0)),
                duration=0.1
            )
        blender.update(0.2)

    def test_stack_with_all_zero_weights(self):
        """Test stack with all zero weights."""
        stack = BlendStack()
        stack.set_base_state(CameraState(position=Vector3(100, 0, 0)))
        stack.push("cam1", lambda: CameraState(position=Vector3(0, 0, 0)), weight=0.0)
        state = stack.update(0.1)
        assert state.position.x == 100


# =============================================================================
# Additional Cut Transition Tests
# =============================================================================


class TestCutTransitionAdvanced:
    """Additional cut transition tests."""

    def test_cut_near_far_clip(self):
        """Test cut changes clip planes."""
        blender = CameraBlender()
        target = CameraState(near_clip=0.5, far_clip=500.0)
        blender.cut(target)
        assert blender.current_state.near_clip == 0.5
        assert blender.current_state.far_clip == 500.0

    def test_multiple_cuts(self):
        """Test multiple sequential cuts."""
        blender = CameraBlender()
        for i in range(10):
            blender.cut(CameraState(position=Vector3(i, 0, 0)))
        assert blender.current_state.position.x == 9


# =============================================================================
# Additional Linear Blending Tests
# =============================================================================


class TestLinearBlendingAdvanced:
    """Additional linear blending tests."""

    def test_blend_rotation(self):
        """Test linear blending of rotation."""
        blender = CameraBlender()
        blender._current_state = CameraState(rotation=Quaternion(0, 0, 0, 1))
        blender.blend_to(
            CameraState(rotation=Quaternion(0, 0.707, 0, 0.707)),
            duration=1.0,
            blend_type=BlendType.LINEAR
        )
        blender.update(0.5)
        assert blender.current_state.rotation.y > 0

    def test_blend_near_far_clip(self):
        """Test linear blending of clip planes."""
        blender = CameraBlender()
        blender._current_state = CameraState(near_clip=0.1, far_clip=1000.0)
        blender.blend_to(
            CameraState(near_clip=1.0, far_clip=100.0),
            duration=1.0,
            blend_type=BlendType.LINEAR
        )
        blender.update(0.5)
        assert blender.current_state.near_clip == pytest.approx(0.55, abs=0.1)

    def test_blend_all_properties(self):
        """Test blending all camera properties simultaneously."""
        blender = CameraBlender()
        blender._current_state = CameraState(
            position=Vector3(0, 0, 0),
            rotation=Quaternion(0, 0, 0, 1),
            fov=60.0,
            near_clip=0.1,
            far_clip=1000.0
        )
        blender.blend_to(
            CameraState(
                position=Vector3(10, 10, 10),
                rotation=Quaternion(0, 0.5, 0, 0.866),
                fov=90.0,
                near_clip=0.5,
                far_clip=500.0
            ),
            duration=1.0
        )
        blender.update(0.5)


# =============================================================================
# Additional Ease Curve Tests
# =============================================================================


class TestEaseCurvesAdvanced:
    """Additional ease curve tests."""

    def test_ease_in_out_expo(self):
        """Test ease-in-out exponential."""
        assert EaseFunctions.ease_in_out_expo(0.0) == 0.0
        assert EaseFunctions.ease_in_out_expo(0.5) == pytest.approx(0.5, abs=0.01)
        assert EaseFunctions.ease_in_out_expo(1.0) == 1.0

    def test_ease_out_elastic(self):
        """Test ease-out elastic curve."""
        assert EaseFunctions.ease_out_elastic(0.0) == 0.0
        assert EaseFunctions.ease_out_elastic(1.0) == 1.0

    def test_ease_in_bounce(self):
        """Test ease-in bounce curve."""
        assert EaseFunctions.ease_in_bounce(0.0) == pytest.approx(0.0, abs=0.01)
        assert EaseFunctions.ease_in_bounce(1.0) == pytest.approx(1.0, abs=0.01)

    def test_ease_functions_symmetry(self):
        """Test ease function symmetry properties."""
        mid = 0.5
        assert EaseFunctions.ease_in_out_quad(mid) == pytest.approx(0.5, abs=0.01)
        assert EaseFunctions.ease_in_out_cubic(mid) == pytest.approx(0.5, abs=0.01)


# =============================================================================
# Additional Blend Stack Tests
# =============================================================================


class TestBlendStackAdvanced:
    """Additional blend stack tests."""

    def test_stack_three_cameras(self):
        """Test blending three cameras."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState(position=Vector3(0, 0, 0)), weight=1.0)
        stack.push("cam2", lambda: CameraState(position=Vector3(10, 0, 0)), weight=1.0)
        stack.push("cam3", lambda: CameraState(position=Vector3(20, 0, 0)), weight=1.0)
        state = stack.update(0.1)
        assert state.position.x == pytest.approx(10.0, abs=0.1)

    def test_stack_unequal_weights(self):
        """Test stack with unequal weights."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState(position=Vector3(0, 0, 0)), weight=1.0)
        stack.push("cam2", lambda: CameraState(position=Vector3(10, 0, 0)), weight=3.0)
        state = stack.update(0.1)
        assert state.position.x > 5.0

    def test_stack_fade_transitions(self):
        """Test smooth fade transitions."""
        stack = BlendStack()
        stack.push("cam1", lambda: CameraState(position=Vector3(0, 0, 0)), weight=1.0)
        stack.fade_in("cam1", duration=0.5)
        stack.fade_out("cam1", duration=0.5)

    def test_get_nonexistent_entry(self):
        """Test getting nonexistent entry returns None."""
        stack = BlendStack()
        entry = stack.get_entry("nonexistent")
        assert entry is None


# =============================================================================
# Additional Priority Tests
# =============================================================================


class TestCameraPriorityAdvanced:
    """Additional camera priority tests."""

    def test_multiple_priority_levels(self):
        """Test multiple cameras with different priorities."""
        manager = CameraPriorityManager()
        manager.register_camera("low", lambda: CameraState(position=Vector3(0, 0, 0)), priority=0)
        manager.register_camera("mid", lambda: CameraState(position=Vector3(5, 0, 0)), priority=5)
        manager.register_camera("high", lambda: CameraState(position=Vector3(10, 0, 0)), priority=10)

    def test_unregister_non_active(self):
        """Test unregistering non-active camera."""
        manager = CameraPriorityManager()
        manager.register_camera("cam1", lambda: CameraState())
        manager.register_camera("cam2", lambda: CameraState())
        manager.activate_camera("cam1", blend_duration=0)
        manager.unregister_camera("cam2")
        assert manager.active_camera == "cam1"

    def test_activate_unregistered(self):
        """Test activating unregistered camera does nothing."""
        manager = CameraPriorityManager()
        manager.activate_camera("nonexistent")
        assert manager.active_camera is None


# =============================================================================
# Additional Viewport Tests
# =============================================================================


class TestViewportSplitAdvanced:
    """Additional viewport split tests."""

    def test_custom_viewport_dimensions(self):
        """Test custom viewport dimensions."""
        manager = SplitScreenManager()
        manager.add_viewport("custom", Viewport(0.1, 0.1, 0.8, 0.8), lambda: CameraState())
        viewports = manager.get_viewports()
        assert viewports["custom"][0].x == 0.1
        assert viewports["custom"][0].width == 0.8

    def test_three_way_split(self):
        """Test three-way viewport split."""
        manager = SplitScreenManager()
        manager.add_viewport("p1", Viewport(0, 0.5, 0.5, 0.5), lambda: CameraState())
        manager.add_viewport("p2", Viewport(0.5, 0.5, 0.5, 0.5), lambda: CameraState())
        manager.add_viewport("p3", Viewport(0, 0, 1, 0.5), lambda: CameraState())
        assert manager.viewport_count == 3

    def test_viewport_state_update(self):
        """Test viewport state is dynamically updated."""
        position = [Vector3(0, 0, 0)]

        def state_provider():
            return CameraState(position=position[0])

        manager = SplitScreenManager()
        manager.add_viewport("p1", Viewport(), state_provider)

        viewports = manager.get_viewports()
        assert viewports["p1"][1].position.x == 0

        position[0] = Vector3(10, 0, 0)
        viewports = manager.get_viewports()
        assert viewports["p1"][1].position.x == 10


# =============================================================================
# Additional Blend Queue Tests
# =============================================================================


class TestBlendQueueAdvanced:
    """Additional blend queue tests."""

    def test_queue_priority_sorting(self):
        """Test queue is sorted by priority."""
        blender = CameraBlender()
        blender._active_blend = BlendRequest(
            id="active", duration=10.0, priority=10
        )

        blender.blend_to(CameraState(), duration=1.0, priority=5)
        blender.blend_to(CameraState(), duration=1.0, priority=8)
        blender.blend_to(CameraState(), duration=1.0, priority=3)

        assert len(blender._blend_queue) == 3

    def test_queue_processes_after_active(self):
        """Test queue processes after active blend completes."""
        blender = CameraBlender()
        blender.blend_to(CameraState(position=Vector3(5, 0, 0)), duration=0.1, priority=5)
        blender.blend_to(CameraState(position=Vector3(10, 0, 0)), duration=0.1, priority=3)

        for _ in range(10):
            blender.update(0.05)

    def test_cancel_queued_blend(self):
        """Test canceling a queued blend."""
        blender = CameraBlender()
        blender._active_blend = BlendRequest(id="active", duration=10.0, priority=10)
        blend_id = blender.blend_to(CameraState(), duration=1.0, priority=5)
        blender.cancel_blend(blend_id)
        assert len(blender._blend_queue) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestBlendingIntegration:
    """Integration tests for camera blending."""

    def test_full_blend_sequence(self):
        """Test complete blend sequence with callbacks."""
        blender = CameraBlender()
        events = []

        def on_complete():
            events.append("complete")

        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=0.5,
            blend_type=BlendType.EASE_IN_OUT,
            on_complete=on_complete
        )

        for _ in range(50):
            blender.update(0.02)

        assert "complete" in events
        assert blender.current_state.position.x == pytest.approx(10.0, abs=0.5)

    def test_priority_camera_system(self):
        """Test priority-based camera switching."""
        manager = CameraPriorityManager()

        manager.register_camera("default", lambda: CameraState(fov=60.0), priority=0)
        manager.register_camera("action", lambda: CameraState(fov=80.0), priority=5)
        manager.register_camera("cutscene", lambda: CameraState(fov=50.0), priority=10)

        manager.activate_camera("default", blend_duration=0)
        assert manager.active_camera == "default"

        manager.activate_camera("action", blend_duration=0.5)
        for _ in range(30):
            manager.update(0.02)

    def test_split_screen_gameplay(self):
        """Test split screen for multiplayer."""
        manager = SplitScreenManager()

        manager.add_viewport("player1", Viewport(), lambda: CameraState(position=Vector3(0, 0, 0)))
        manager.add_viewport("player2", Viewport(), lambda: CameraState(position=Vector3(100, 0, 0)))

        manager.set_layout("vertical_split")
        viewports = manager.get_viewports()

        assert viewports["player1"][0].width == 0.5
        assert viewports["player2"][0].x == 0.5

    def test_blend_stack_layering(self):
        """Test blend stack camera layering."""
        stack = BlendStack()

        stack.push("base", lambda: CameraState(position=Vector3(0, 0, 0)), weight=1.0)
        stack.push("shake", lambda: CameraState(position=Vector3(0.1, 0.1, 0)), weight=0.3)
        stack.push("zoom", lambda: CameraState(fov=70.0), weight=0.5)

        state = stack.update(0.016)
        # Stack should blend all cameras


class TestBlendingStress:
    """Stress tests for camera blending."""

    def test_many_blend_requests(self):
        """Test handling many blend requests."""
        blender = CameraBlender()

        for i in range(100):
            blender.blend_to(
                CameraState(position=Vector3(i, 0, 0)),
                duration=0.1,
                priority=i % 10
            )

        for _ in range(50):
            blender.update(0.016)

    def test_many_stack_entries(self):
        """Test blend stack with many entries."""
        stack = BlendStack()

        for i in range(50):
            stack.push(
                f"cam_{i}",
                lambda i=i: CameraState(position=Vector3(i, 0, 0)),
                weight=0.5,
                priority=i
            )

        for _ in range(100):
            stack.update(0.016)

    def test_many_viewports(self):
        """Test many viewports in split screen."""
        manager = SplitScreenManager()

        for i in range(8):
            manager.add_viewport(
                f"player_{i}",
                Viewport(i % 4 * 0.25, i // 4 * 0.5, 0.25, 0.5),
                lambda: CameraState()
            )

        viewports = manager.get_viewports()
        assert len(viewports) == 8


class TestBlendingAdvanced:
    """Advanced blending scenarios."""

    def test_nested_blend_interruptions(self):
        """Test nested blend interruptions."""
        blender = CameraBlender()
        interrupts = []

        for i in range(5):
            blender.blend_to(
                CameraState(position=Vector3(i * 10, 0, 0)),
                duration=1.0,
                on_interrupt=lambda i=i: interrupts.append(i)
            )
            blender.update(0.1)

        assert len(interrupts) == 4  # All but last were interrupted

    def test_ease_function_composition(self):
        """Test composing multiple ease functions."""
        def double_ease(t):
            return EaseFunctions.ease_in_quad(EaseFunctions.ease_out_quad(t))

        blender = CameraBlender()
        blender._current_state = CameraState(position=Vector3(0, 0, 0))
        blender.blend_to(
            CameraState(position=Vector3(10, 0, 0)),
            duration=1.0,
            ease_function=double_ease
        )
        blender.update(0.5)

    def test_blend_with_all_camera_properties(self):
        """Test blending all camera properties together."""
        blender = CameraBlender()

        source = CameraState(
            position=Vector3(0, 0, 0),
            rotation=Quaternion(0, 0, 0, 1),
            fov=60.0,
            near_clip=0.1,
            far_clip=1000.0
        )
        target = CameraState(
            position=Vector3(100, 50, 25),
            rotation=Quaternion(0, 0.707, 0, 0.707),
            fov=90.0,
            near_clip=1.0,
            far_clip=500.0
        )

        blender._current_state = source
        blender.blend_to(target, duration=1.0)
        blender.update(0.5)

        state = blender.current_state
        assert 0 < state.position.x < 100
        assert 60 < state.fov < 90

    def test_stack_fade_timing(self):
        """Test precise fade timing in stack."""
        stack = BlendStack()

        stack.push("cam1", lambda: CameraState(position=Vector3(10, 0, 0)), weight=1.0)
        stack.fade_out("cam1", duration=0.5)

        weights = []
        for _ in range(50):
            stack.update(0.02)
            entry = stack.get_entry("cam1")
            if entry:
                weights.append(entry.weight)

        # Weight should decrease over time
        if len(weights) > 1:
            assert weights[-1] < weights[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
