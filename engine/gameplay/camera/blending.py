"""Camera Blending and Transitions - Smooth transitions between camera states.

This module provides camera blending functionality including blend curves,
blend stacks, viewport splitting, and priority-based camera selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import math

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4

from engine.gameplay.camera.constants import (
    BLEND_DURATION_CUT,
    BLEND_DURATION_FAST,
    BLEND_DURATION_EASE,
    BLEND_DURATION_SMOOTH,
    BLEND_DURATION_LONG,
    DEFAULT_FOV,
    DEFAULT_CAMERA_PRIORITY,
    CINEMATIC_CAMERA_PRIORITY,
    CUTSCENE_CAMERA_PRIORITY,
    DEBUG_CAMERA_PRIORITY,
    DEFAULT_NEAR_PLANE, DEFAULT_FAR_PLANE,
    CAMERA_EPSILON,
    MIN_DELTA_TIME, MAX_DELTA_TIME,
    ELASTIC_EASING_PERIOD,
    BOUNCE_EASING_COEFFICIENT,
    BOUNCE_EASING_DIVISOR,
)

if TYPE_CHECKING:
    from engine.gameplay.camera.controller import BaseCameraController, CameraState


class BlendType(Enum):
    """Types of blend curves."""
    CUT = auto()            # Instant switch (no blend)
    LINEAR = auto()         # Linear interpolation
    EASE_IN = auto()        # Slow start, fast end
    EASE_OUT = auto()       # Fast start, slow end
    EASE_IN_OUT = auto()    # Slow start and end
    CUBIC = auto()          # Cubic interpolation
    CUBIC_IN = auto()       # Cubic ease in
    CUBIC_OUT = auto()      # Cubic ease out
    CUBIC_IN_OUT = auto()   # Cubic ease in/out
    EXPONENTIAL = auto()    # Exponential interpolation
    ELASTIC = auto()        # Elastic bounce effect
    BOUNCE = auto()         # Bounce at end
    CUSTOM = auto()         # User-defined curve


@dataclass(slots=True)
class BlendCurve:
    """
    Defines a blend curve for camera transitions.

    Attributes:
        blend_type: Type of interpolation curve
        duration: Blend duration in seconds
        custom_func: Optional custom curve function
        overshoot: Overshoot amount for elastic/bounce (0.0 to 1.0)
    """
    blend_type: BlendType = BlendType.EASE_IN_OUT
    duration: float = BLEND_DURATION_EASE
    custom_func: Optional[Callable[[float], float]] = None
    overshoot: float = 0.1

    def evaluate(self, t: float) -> float:
        """
        Evaluate the blend curve at parameter t.

        Args:
            t: Progress (0.0 to 1.0)

        Returns:
            Interpolated value (may exceed 0-1 for elastic/bounce)
        """
        t = max(0.0, min(1.0, t))

        if self.blend_type == BlendType.CUT:
            return 1.0 if t > 0.0 else 0.0

        elif self.blend_type == BlendType.LINEAR:
            return t

        elif self.blend_type == BlendType.EASE_IN:
            return t * t

        elif self.blend_type == BlendType.EASE_OUT:
            return t * (2.0 - t)

        elif self.blend_type == BlendType.EASE_IN_OUT:
            if t < 0.5:
                return 2.0 * t * t
            return -1.0 + (4.0 - 2.0 * t) * t

        elif self.blend_type == BlendType.CUBIC:
            return t * t * t

        elif self.blend_type == BlendType.CUBIC_IN:
            return t * t * t

        elif self.blend_type == BlendType.CUBIC_OUT:
            u = t - 1.0
            return u * u * u + 1.0

        elif self.blend_type == BlendType.CUBIC_IN_OUT:
            if t < 0.5:
                return 4.0 * t * t * t
            u = 2.0 * t - 2.0
            return 0.5 * u * u * u + 1.0

        elif self.blend_type == BlendType.EXPONENTIAL:
            if t == 0.0:
                return 0.0
            return math.pow(2.0, 10.0 * (t - 1.0))

        elif self.blend_type == BlendType.ELASTIC:
            return self._elastic(t)

        elif self.blend_type == BlendType.BOUNCE:
            return self._bounce(t)

        elif self.blend_type == BlendType.CUSTOM and self.custom_func is not None:
            return self.custom_func(t)

        return t

    def _elastic(self, t: float) -> float:
        """Elastic easing with overshoot."""
        if t == 0.0 or t == 1.0:
            return t

        p = ELASTIC_EASING_PERIOD
        s = p / 4.0

        t_adj = t - 1.0
        return -(math.pow(2.0, 10.0 * t_adj) *
                 math.sin((t_adj - s) * (2.0 * math.pi) / p) *
                 (1.0 + self.overshoot))

    def _bounce(self, t: float) -> float:
        """Bounce easing."""
        divisor = BOUNCE_EASING_DIVISOR
        coeff = BOUNCE_EASING_COEFFICIENT
        if t < 1.0 / divisor:
            return coeff * t * t
        elif t < 2.0 / divisor:
            t -= 1.5 / divisor
            return coeff * t * t + 0.75
        elif t < 2.5 / divisor:
            t -= 2.25 / divisor
            return coeff * t * t + 0.9375
        else:
            t -= 2.625 / divisor
            return coeff * t * t + 0.984375


@dataclass(slots=True)
class CameraBlendState:
    """
    Snapshot of camera state for blending.

    Lighter weight than full CameraState, just essential data.
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    rotation: Quat = field(default_factory=Quat.identity)
    fov: float = DEFAULT_FOV
    near_plane: float = DEFAULT_NEAR_PLANE
    far_plane: float = DEFAULT_FAR_PLANE

    @classmethod
    def from_controller(cls, controller: BaseCameraController) -> CameraBlendState:
        """Create blend state from camera controller."""
        return cls(
            position=Vec3(controller.position.x, controller.position.y, controller.position.z),
            rotation=Quat(controller.rotation.x, controller.rotation.y, controller.rotation.z, controller.rotation.w),
            fov=controller.fov,
            near_plane=controller.near_plane,
            far_plane=controller.far_plane,
        )

    def lerp(self, other: CameraBlendState, t: float) -> CameraBlendState:
        """Interpolate between two blend states."""
        return CameraBlendState(
            position=self.position.lerp(other.position, t),
            rotation=self.rotation.slerp(other.rotation, t),
            fov=self.fov + (other.fov - self.fov) * t,
            near_plane=self.near_plane + (other.near_plane - self.near_plane) * t,
            far_plane=self.far_plane + (other.far_plane - self.far_plane) * t,
        )


class CameraBlend:
    """
    Represents an active blend between two camera states.

    Features:
    - Configurable blend curves
    - Progress tracking
    - Completion callbacks
    - Pause/resume support
    """

    __slots__ = (
        "_from_state",
        "_to_state",
        "_from_controller",
        "_to_controller",
        "_curve",
        "_progress",
        "_is_complete",
        "_is_paused",
        "_on_complete",
        "_priority",
        "_name",
    )

    def __init__(
        self,
        from_state: CameraBlendState,
        to_state: CameraBlendState,
        curve: Optional[BlendCurve] = None,
        from_controller: Optional[BaseCameraController] = None,
        to_controller: Optional[BaseCameraController] = None,
        name: str = "",
    ) -> None:
        """
        Initialize camera blend.

        Args:
            from_state: Starting camera state
            to_state: Target camera state
            curve: Blend curve to use
            from_controller: Optional source controller
            to_controller: Optional target controller
            name: Optional identifier
        """
        self._from_state = from_state
        self._to_state = to_state
        self._from_controller = from_controller
        self._to_controller = to_controller
        self._curve = curve if curve is not None else BlendCurve()
        self._progress = 0.0
        self._is_complete = False
        self._is_paused = False
        self._on_complete: List[Callable[[CameraBlend], None]] = []
        self._priority = 0
        self._name = name

    @property
    def from_state(self) -> CameraBlendState:
        """Get starting state."""
        return self._from_state

    @property
    def to_state(self) -> CameraBlendState:
        """Get target state."""
        return self._to_state

    @property
    def from_controller(self) -> Optional[BaseCameraController]:
        """Get source controller."""
        return self._from_controller

    @property
    def to_controller(self) -> Optional[BaseCameraController]:
        """Get target controller."""
        return self._to_controller

    @property
    def curve(self) -> BlendCurve:
        """Get blend curve."""
        return self._curve

    @property
    def progress(self) -> float:
        """Get raw progress (0.0 to 1.0)."""
        return self._progress

    @property
    def blend_weight(self) -> float:
        """Get blend weight with curve applied."""
        return self._curve.evaluate(self._progress)

    @property
    def duration(self) -> float:
        """Get blend duration."""
        return self._curve.duration

    @property
    def is_complete(self) -> bool:
        """Check if blend is complete."""
        return self._is_complete

    @property
    def is_paused(self) -> bool:
        """Check if blend is paused."""
        return self._is_paused

    @property
    def name(self) -> str:
        """Get blend name."""
        return self._name

    def pause(self) -> None:
        """Pause the blend."""
        self._is_paused = True

    def resume(self) -> None:
        """Resume the blend."""
        self._is_paused = False

    def skip(self) -> None:
        """Skip to end of blend."""
        self._progress = 1.0
        self._is_complete = True
        self._notify_complete()

    def reverse(self) -> None:
        """Reverse the blend direction."""
        self._from_state, self._to_state = self._to_state, self._from_state
        self._from_controller, self._to_controller = self._to_controller, self._from_controller
        self._progress = 1.0 - self._progress

    def on_complete(self, callback: Callable[[CameraBlend], None]) -> None:
        """Register completion callback."""
        self._on_complete.append(callback)

    def _notify_complete(self) -> None:
        """Notify listeners of completion."""
        for callback in self._on_complete:
            callback(self)

    def get_current_state(self) -> CameraBlendState:
        """
        Get interpolated camera state at current progress.

        Returns:
            Blended camera state
        """
        # If using live controllers, get fresh states
        from_state = self._from_state
        to_state = self._to_state

        if self._from_controller is not None:
            from_state = CameraBlendState.from_controller(self._from_controller)
        if self._to_controller is not None:
            to_state = CameraBlendState.from_controller(self._to_controller)

        return from_state.lerp(to_state, self.blend_weight)

    def update(self, delta_time: float) -> CameraBlendState:
        """
        Update blend progress.

        Args:
            delta_time: Time since last update

        Returns:
            Current blended state
        """
        if self._is_complete or self._is_paused:
            return self.get_current_state()

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Instant cut
        if self._curve.duration <= CAMERA_EPSILON:
            self._progress = 1.0
            self._is_complete = True
            self._notify_complete()
            return self._to_state

        # Update progress
        self._progress += delta_time / self._curve.duration
        if self._progress >= 1.0:
            self._progress = 1.0
            self._is_complete = True
            self._notify_complete()

        return self.get_current_state()


class BlendStack:
    """
    Stack of active camera blends.

    Features:
    - Multiple concurrent blends
    - Blend priority
    - Automatic cleanup of completed blends
    - Combined result calculation
    """

    __slots__ = (
        "_blends",
        "_max_blends",
        "_auto_cleanup",
    )

    def __init__(self, max_blends: int = 8) -> None:
        """
        Initialize blend stack.

        Args:
            max_blends: Maximum concurrent blends
        """
        self._blends: List[CameraBlend] = []
        self._max_blends = max_blends
        self._auto_cleanup = True

    @property
    def blend_count(self) -> int:
        """Get number of active blends."""
        return len(self._blends)

    @property
    def is_blending(self) -> bool:
        """Check if any blends are active."""
        return len(self._blends) > 0

    @property
    def top_blend(self) -> Optional[CameraBlend]:
        """Get the most recent blend."""
        return self._blends[-1] if self._blends else None

    def push_blend(self, blend: CameraBlend) -> None:
        """
        Add a new blend to the stack.

        Args:
            blend: Blend to add
        """
        # Remove oldest if at limit
        while len(self._blends) >= self._max_blends:
            self._blends.pop(0)

        self._blends.append(blend)

    def pop_blend(self) -> Optional[CameraBlend]:
        """
        Remove and return the most recent blend.

        Returns:
            Removed blend or None
        """
        if self._blends:
            return self._blends.pop()
        return None

    def clear(self) -> None:
        """Clear all blends."""
        self._blends.clear()

    def get_blend_by_name(self, name: str) -> Optional[CameraBlend]:
        """Find blend by name."""
        for blend in self._blends:
            if blend.name == name:
                return blend
        return None

    def remove_blend(self, blend: CameraBlend) -> bool:
        """
        Remove specific blend.

        Returns:
            True if blend was found and removed
        """
        if blend in self._blends:
            self._blends.remove(blend)
            return True
        return False

    def update(self, delta_time: float) -> Optional[CameraBlendState]:
        """
        Update all blends and return combined result.

        Args:
            delta_time: Time since last update

        Returns:
            Combined camera state, or None if no blends
        """
        if not self._blends:
            return None

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Update all blends
        for blend in self._blends:
            blend.update(delta_time)

        # Remove completed blends if auto-cleanup is enabled
        if self._auto_cleanup:
            self._blends = [b for b in self._blends if not b.is_complete]

        if not self._blends:
            return None

        # Return the most recent blend's state
        # (Could implement more complex blend combining here)
        return self._blends[-1].get_current_state()


@dataclass(slots=True)
class ViewportRect:
    """Viewport rectangle for split-screen."""
    x: float = 0.0       # Left edge (0.0 to 1.0)
    y: float = 0.0       # Bottom edge (0.0 to 1.0)
    width: float = 1.0   # Width (0.0 to 1.0)
    height: float = 1.0  # Height (0.0 to 1.0)

    def contains_point(self, px: float, py: float) -> bool:
        """Check if normalized point is within viewport."""
        return (self.x <= px < self.x + self.width and
                self.y <= py < self.y + self.height)

    def get_aspect_ratio(self, screen_aspect: float) -> float:
        """Calculate aspect ratio for this viewport."""
        return (self.width / self.height) * screen_aspect if self.height > 0 else 1.0


class SplitScreenLayout(Enum):
    """Predefined split-screen layouts."""
    SINGLE = auto()           # No split
    HORIZONTAL_2 = auto()     # Top/bottom split
    VERTICAL_2 = auto()       # Left/right split
    QUAD = auto()             # 4-way split
    TRIPLE_TOP = auto()       # 1 top, 2 bottom
    TRIPLE_BOTTOM = auto()    # 2 top, 1 bottom
    PIP = auto()              # Picture-in-picture


class ViewportSplit:
    """
    Manages split-screen with multiple cameras.

    Features:
    - Predefined layouts
    - Custom viewport configuration
    - Camera assignment
    - Letterboxing/pillarboxing
    """

    __slots__ = (
        "_viewports",
        "_cameras",
        "_layout",
        "_padding",
        "_screen_aspect",
    )

    def __init__(self, layout: SplitScreenLayout = SplitScreenLayout.SINGLE) -> None:
        """
        Initialize viewport split.

        Args:
            layout: Initial layout
        """
        self._viewports: List[ViewportRect] = []
        self._cameras: List[Optional[BaseCameraController]] = []
        self._layout = layout
        self._padding = 0.01  # Gap between viewports
        self._screen_aspect = 16.0 / 9.0

        self._apply_layout(layout)

    @property
    def layout(self) -> SplitScreenLayout:
        """Get current layout."""
        return self._layout

    @layout.setter
    def layout(self, value: SplitScreenLayout) -> None:
        """Set layout."""
        self._layout = value
        self._apply_layout(value)

    @property
    def viewport_count(self) -> int:
        """Get number of viewports."""
        return len(self._viewports)

    @property
    def screen_aspect(self) -> float:
        """Get screen aspect ratio."""
        return self._screen_aspect

    @screen_aspect.setter
    def screen_aspect(self, value: float) -> None:
        """Set screen aspect ratio."""
        self._screen_aspect = max(0.1, value)

    def get_viewport(self, index: int) -> Optional[ViewportRect]:
        """Get viewport at index."""
        if 0 <= index < len(self._viewports):
            return self._viewports[index]
        return None

    def get_camera(self, index: int) -> Optional[BaseCameraController]:
        """Get camera for viewport."""
        if 0 <= index < len(self._cameras):
            return self._cameras[index]
        return None

    def set_camera(self, index: int, camera: BaseCameraController) -> None:
        """Assign camera to viewport."""
        # Extend lists if needed
        while len(self._cameras) <= index:
            self._cameras.append(None)
        self._cameras[index] = camera

    def _apply_layout(self, layout: SplitScreenLayout) -> None:
        """Apply predefined layout."""
        self._viewports.clear()
        pad = self._padding

        if layout == SplitScreenLayout.SINGLE:
            self._viewports.append(ViewportRect(0, 0, 1, 1))

        elif layout == SplitScreenLayout.HORIZONTAL_2:
            h = (1.0 - pad) / 2.0
            self._viewports.append(ViewportRect(0, h + pad, 1, h))  # Top
            self._viewports.append(ViewportRect(0, 0, 1, h))         # Bottom

        elif layout == SplitScreenLayout.VERTICAL_2:
            w = (1.0 - pad) / 2.0
            self._viewports.append(ViewportRect(0, 0, w, 1))         # Left
            self._viewports.append(ViewportRect(w + pad, 0, w, 1))   # Right

        elif layout == SplitScreenLayout.QUAD:
            w = (1.0 - pad) / 2.0
            h = (1.0 - pad) / 2.0
            self._viewports.append(ViewportRect(0, h + pad, w, h))       # Top-left
            self._viewports.append(ViewportRect(w + pad, h + pad, w, h)) # Top-right
            self._viewports.append(ViewportRect(0, 0, w, h))             # Bottom-left
            self._viewports.append(ViewportRect(w + pad, 0, w, h))       # Bottom-right

        elif layout == SplitScreenLayout.TRIPLE_TOP:
            w = (1.0 - pad) / 2.0
            h = (1.0 - pad) / 2.0
            self._viewports.append(ViewportRect(0, h + pad, 1, h))       # Top (full width)
            self._viewports.append(ViewportRect(0, 0, w, h))             # Bottom-left
            self._viewports.append(ViewportRect(w + pad, 0, w, h))       # Bottom-right

        elif layout == SplitScreenLayout.TRIPLE_BOTTOM:
            w = (1.0 - pad) / 2.0
            h = (1.0 - pad) / 2.0
            self._viewports.append(ViewportRect(0, h + pad, w, h))       # Top-left
            self._viewports.append(ViewportRect(w + pad, h + pad, w, h)) # Top-right
            self._viewports.append(ViewportRect(0, 0, 1, h))             # Bottom (full width)

        elif layout == SplitScreenLayout.PIP:
            pip_w = 0.25
            pip_h = 0.25
            pip_margin = 0.02
            self._viewports.append(ViewportRect(0, 0, 1, 1))  # Main
            self._viewports.append(ViewportRect(
                1.0 - pip_w - pip_margin,
                1.0 - pip_h - pip_margin,
                pip_w,
                pip_h
            ))  # PIP

        # Initialize camera slots
        self._cameras = [None] * len(self._viewports)

    def set_custom_viewports(self, viewports: List[ViewportRect]) -> None:
        """Set custom viewport configuration."""
        self._layout = SplitScreenLayout.SINGLE  # Mark as custom
        self._viewports = viewports.copy()
        self._cameras = [None] * len(self._viewports)

    def get_viewport_at_point(self, x: float, y: float) -> Optional[int]:
        """
        Get viewport index at screen point.

        Args:
            x: Normalized x coordinate (0.0 to 1.0)
            y: Normalized y coordinate (0.0 to 1.0)

        Returns:
            Viewport index or None
        """
        # Check in reverse order (PIP on top)
        for i in range(len(self._viewports) - 1, -1, -1):
            if self._viewports[i].contains_point(x, y):
                return i
        return None


@dataclass(slots=True)
class PrioritizedCamera:
    """Camera with priority for selection."""
    controller: BaseCameraController
    priority: int = DEFAULT_CAMERA_PRIORITY
    blend_curve: Optional[BlendCurve] = None
    tag: str = ""
    is_active: bool = True


class CameraPriority:
    """
    Priority-based camera selection system.

    Features:
    - Register cameras with priority
    - Automatic selection of highest priority
    - Blend transitions on camera change
    - Tag-based camera lookup
    """

    __slots__ = (
        "_cameras",
        "_active_camera",
        "_pending_camera",
        "_blend_stack",
        "_default_blend",
        "_on_camera_changed",
    )

    def __init__(self) -> None:
        """Initialize camera priority system."""
        self._cameras: List[PrioritizedCamera] = []
        self._active_camera: Optional[PrioritizedCamera] = None
        self._pending_camera: Optional[PrioritizedCamera] = None
        self._blend_stack = BlendStack()
        self._default_blend = BlendCurve(BlendType.EASE_IN_OUT, BLEND_DURATION_EASE)
        self._on_camera_changed: List[Callable[[Optional[BaseCameraController], Optional[BaseCameraController]], None]] = []

    @property
    def active_controller(self) -> Optional[BaseCameraController]:
        """Get the active camera controller."""
        if self._active_camera is not None:
            return self._active_camera.controller
        return None

    @property
    def is_blending(self) -> bool:
        """Check if camera is currently blending."""
        return self._blend_stack.is_blending

    @property
    def camera_count(self) -> int:
        """Get number of registered cameras."""
        return len(self._cameras)

    def register(
        self,
        controller: BaseCameraController,
        priority: int = DEFAULT_CAMERA_PRIORITY,
        blend_curve: Optional[BlendCurve] = None,
        tag: str = "",
    ) -> PrioritizedCamera:
        """
        Register a camera with priority.

        Args:
            controller: Camera controller
            priority: Selection priority (higher = takes precedence)
            blend_curve: Blend curve for transitioning to this camera
            tag: Optional identifier tag

        Returns:
            Created PrioritizedCamera entry
        """
        entry = PrioritizedCamera(
            controller=controller,
            priority=priority,
            blend_curve=blend_curve,
            tag=tag,
        )
        self._cameras.append(entry)

        # Sort by priority (highest first)
        self._cameras.sort(key=lambda c: c.priority, reverse=True)

        # Check if this should become active
        self._update_active_camera()

        return entry

    def unregister(self, controller: BaseCameraController) -> bool:
        """
        Unregister a camera.

        Args:
            controller: Camera to remove

        Returns:
            True if camera was found and removed
        """
        for entry in self._cameras:
            if entry.controller is controller:
                self._cameras.remove(entry)

                # If this was active, find new active
                if self._active_camera is entry:
                    self._active_camera = None
                    self._update_active_camera()

                return True
        return False

    def set_active(self, controller: BaseCameraController, immediate: bool = False) -> bool:
        """
        Force a specific camera to be active.

        Args:
            controller: Camera to activate
            immediate: Skip blend transition

        Returns:
            True if camera was found and activated
        """
        for entry in self._cameras:
            if entry.controller is controller:
                if entry != self._active_camera:
                    self._switch_to_camera(entry, immediate)
                return True
        return False

    def set_active_by_tag(self, tag: str, immediate: bool = False) -> bool:
        """
        Activate camera by tag.

        Args:
            tag: Camera tag to find
            immediate: Skip blend transition

        Returns:
            True if camera was found and activated
        """
        for entry in self._cameras:
            if entry.tag == tag:
                if entry != self._active_camera:
                    self._switch_to_camera(entry, immediate)
                return True
        return False

    def get_by_tag(self, tag: str) -> Optional[BaseCameraController]:
        """Get camera controller by tag."""
        for entry in self._cameras:
            if entry.tag == tag:
                return entry.controller
        return None

    def set_priority(self, controller: BaseCameraController, priority: int) -> None:
        """Update camera priority."""
        for entry in self._cameras:
            if entry.controller is controller:
                entry.priority = priority
                break

        self._cameras.sort(key=lambda c: c.priority, reverse=True)
        self._update_active_camera()

    def enable_camera(self, controller: BaseCameraController) -> None:
        """Enable a camera for selection."""
        for entry in self._cameras:
            if entry.controller is controller:
                entry.is_active = True
                break
        self._update_active_camera()

    def disable_camera(self, controller: BaseCameraController) -> None:
        """Disable a camera from selection."""
        for entry in self._cameras:
            if entry.controller is controller:
                entry.is_active = False
                break
        self._update_active_camera()

    def on_camera_changed(
        self,
        callback: Callable[[Optional[BaseCameraController], Optional[BaseCameraController]], None]
    ) -> None:
        """Register callback for camera changes (old, new)."""
        self._on_camera_changed.append(callback)

    def _update_active_camera(self) -> None:
        """Update active camera based on priorities."""
        # Find highest priority active camera
        best_camera: Optional[PrioritizedCamera] = None
        for entry in self._cameras:
            if entry.is_active:
                best_camera = entry
                break  # Already sorted by priority

        if best_camera != self._active_camera:
            self._switch_to_camera(best_camera, immediate=False)

    def _switch_to_camera(self, new_camera: Optional[PrioritizedCamera], immediate: bool) -> None:
        """Switch to a new camera with optional blend."""
        old_controller = self._active_camera.controller if self._active_camera else None
        new_controller = new_camera.controller if new_camera else None

        if immediate or old_controller is None or new_controller is None:
            self._active_camera = new_camera
            for callback in self._on_camera_changed:
                callback(old_controller, new_controller)
            return

        # Create blend
        blend_curve = new_camera.blend_curve if new_camera.blend_curve else self._default_blend

        blend = CameraBlend(
            from_state=CameraBlendState.from_controller(old_controller),
            to_state=CameraBlendState.from_controller(new_controller),
            curve=blend_curve,
            from_controller=old_controller,
            to_controller=new_controller,
        )

        self._blend_stack.push_blend(blend)
        self._active_camera = new_camera

        for callback in self._on_camera_changed:
            callback(old_controller, new_controller)

    def update(self, delta_time: float) -> Optional[CameraBlendState]:
        """
        Update priority system and return current camera state.

        Args:
            delta_time: Time since last update

        Returns:
            Current camera state (may be blended)
        """
        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Update active controller
        if self._active_camera is not None:
            self._active_camera.controller.update(delta_time)

        # Update blends
        blend_state = self._blend_stack.update(delta_time)

        if blend_state is not None:
            return blend_state

        # Return active camera state
        if self._active_camera is not None:
            return CameraBlendState.from_controller(self._active_camera.controller)

        return None


class CameraDirector:
    """
    High-level camera management combining all systems.

    Features:
    - Priority-based camera selection
    - Blend management
    - Split-screen support
    - Effect integration
    """

    __slots__ = (
        "_priority_system",
        "_viewport_split",
        "_active_blends",
        "_default_curve",
    )

    def __init__(self) -> None:
        """Initialize camera director."""
        self._priority_system = CameraPriority()
        self._viewport_split = ViewportSplit()
        self._active_blends: Dict[int, CameraBlend] = {}  # viewport -> blend
        self._default_curve = BlendCurve(BlendType.EASE_IN_OUT, BLEND_DURATION_EASE)

    @property
    def priority_system(self) -> CameraPriority:
        """Get priority system."""
        return self._priority_system

    @property
    def viewport_split(self) -> ViewportSplit:
        """Get viewport split manager."""
        return self._viewport_split

    def blend_to(
        self,
        target: BaseCameraController,
        duration: float = BLEND_DURATION_EASE,
        blend_type: BlendType = BlendType.EASE_IN_OUT,
        viewport: int = 0,
    ) -> CameraBlend:
        """
        Start a blend to target camera.

        Args:
            target: Target camera controller
            duration: Blend duration
            blend_type: Type of blend curve
            viewport: Viewport index for split-screen

        Returns:
            Created blend
        """
        current = self._viewport_split.get_camera(viewport)
        if current is None:
            # No current camera, just set directly
            self._viewport_split.set_camera(viewport, target)
            return CameraBlend(
                CameraBlendState.from_controller(target),
                CameraBlendState.from_controller(target),
                BlendCurve(BlendType.CUT, 0),
            )

        curve = BlendCurve(blend_type, duration)
        blend = CameraBlend(
            CameraBlendState.from_controller(current),
            CameraBlendState.from_controller(target),
            curve,
            current,
            target,
        )

        self._active_blends[viewport] = blend
        return blend

    def cut_to(self, target: BaseCameraController, viewport: int = 0) -> None:
        """Instant cut to camera (no blend)."""
        self._viewport_split.set_camera(viewport, target)
        if viewport in self._active_blends:
            del self._active_blends[viewport]

    def update(self, delta_time: float) -> List[CameraBlendState]:
        """
        Update all cameras and return states.

        Args:
            delta_time: Time since last update

        Returns:
            List of camera states (one per viewport)
        """
        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        states: List[CameraBlendState] = []

        for i in range(self._viewport_split.viewport_count):
            # Check for active blend
            if i in self._active_blends:
                blend = self._active_blends[i]
                state = blend.update(delta_time)

                if blend.is_complete:
                    # Set target camera as current
                    if blend.to_controller is not None:
                        self._viewport_split.set_camera(i, blend.to_controller)
                    del self._active_blends[i]

                states.append(state)
            else:
                # No blend, use camera directly
                camera = self._viewport_split.get_camera(i)
                if camera is not None:
                    camera.update(delta_time)
                    states.append(CameraBlendState.from_controller(camera))
                else:
                    states.append(CameraBlendState())

        return states


__all__ = [
    "BlendType",
    "BlendCurve",
    "CameraBlendState",
    "CameraBlend",
    "BlendStack",
    "ViewportRect",
    "SplitScreenLayout",
    "ViewportSplit",
    "PrioritizedCamera",
    "CameraPriority",
    "CameraDirector",
]
