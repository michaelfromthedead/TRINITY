"""XR UI Panel implementation.

Provides world-space, head-locked, and hand-attached UI panels for XR.

Panel types:
- World: Fixed in 3D space, like a physical sign
- Head-locked: Follows head but with smooth lag (HUD-like)
- Hand-attached: Attached to controller or hand
- Wrist: Watch-style UI on wrist

Interaction modes:
- Ray: Laser pointer from controller
- Poke: Direct touch with finger/controller
- Gaze: Eye tracking with dwell activation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Annotated, Any, Callable, Optional, TYPE_CHECKING
from functools import wraps

if TYPE_CHECKING:
    from typing import TypeVar
    T = TypeVar('T')


class XRPanelType(Enum):
    """Types of XR UI panels."""
    WORLD = auto()  # Fixed in world space
    HEAD_LOCKED = auto()  # Follows head with lag
    HAND_ATTACHED = auto()  # Attached to controller/hand
    WRIST = auto()  # Watch-style on wrist


class XRInteractionMode(Enum):
    """UI interaction modes for XR."""
    RAY = auto()  # Laser pointer interaction
    POKE = auto()  # Direct touch/poke
    GAZE = auto()  # Eye tracking + dwell
    GRAB = auto()  # Grab and manipulate panel


@dataclass(slots=True)
class XRPanelConfig:
    """Configuration for XR UI panel."""
    width: float = 1.0  # Meters
    height: float = 0.75  # Meters
    pixels_per_meter: float = 1000.0
    curved: bool = False
    curve_radius: float = 2.0
    billboard: bool = False
    face_camera: bool = False
    interaction_modes: tuple[XRInteractionMode, ...] = (XRInteractionMode.RAY,)

    def __post_init__(self):
        """Validate configuration."""
        if self.width <= 0:
            raise ValueError("Panel width must be positive")
        if self.height <= 0:
            raise ValueError("Panel height must be positive")
        if self.pixels_per_meter <= 0:
            raise ValueError("Pixels per meter must be positive")
        if self.curved and self.curve_radius <= 0:
            raise ValueError("Curve radius must be positive when curved")


@dataclass(slots=True)
class XRUIPanel:
    """XR UI Panel component.

    A panel that can display UI content in XR space with various
    attachment and interaction modes.

    Attributes:
        panel_type: Type of panel (world, head-locked, hand-attached, wrist)
        position: 3D position in world/local space (x, y, z)
        orientation: Quaternion orientation (x, y, z, w)
        config: Panel configuration settings
        is_visible: Whether panel is currently visible
        is_interactable: Whether panel accepts interaction
        is_hovered: Whether panel is currently being pointed at
        current_interactor: Entity ID of current interactor (if any)
    """
    panel_type: XRPanelType = XRPanelType.WORLD
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    config: XRPanelConfig = field(default_factory=XRPanelConfig)
    is_visible: bool = True
    is_interactable: bool = True
    is_hovered: bool = False
    current_interactor: Optional[int] = None
    _children: list[Any] = field(default_factory=list)
    _parent: Optional['XRUIPanel'] = None
    _dirty: bool = True

    @property
    def width(self) -> float:
        """Get panel width in meters."""
        return self.config.width

    @property
    def height(self) -> float:
        """Get panel height in meters."""
        return self.config.height

    @property
    def pixel_width(self) -> int:
        """Get panel width in pixels."""
        return int(self.config.width * self.config.pixels_per_meter)

    @property
    def pixel_height(self) -> int:
        """Get panel height in pixels."""
        return int(self.config.height * self.config.pixels_per_meter)

    @property
    def is_curved(self) -> bool:
        """Check if panel has curved display."""
        return self.config.curved

    def add_child(self, child: Any) -> None:
        """Add a child UI element to this panel."""
        self._children.append(child)
        if hasattr(child, '_parent'):
            child._parent = self
        self._dirty = True

    def remove_child(self, child: Any) -> bool:
        """Remove a child UI element from this panel."""
        if child in self._children:
            self._children.remove(child)
            if hasattr(child, '_parent'):
                child._parent = None
            self._dirty = True
            return True
        return False

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set panel position."""
        self.position = (x, y, z)
        self._dirty = True

    def set_orientation(self, x: float, y: float, z: float, w: float) -> None:
        """Set panel orientation as quaternion."""
        self.orientation = (x, y, z, w)
        self._dirty = True

    def show(self) -> None:
        """Make panel visible."""
        self.is_visible = True
        self._dirty = True

    def hide(self) -> None:
        """Hide panel."""
        self.is_visible = False
        self._dirty = True

    def toggle(self) -> None:
        """Toggle panel visibility."""
        self.is_visible = not self.is_visible
        self._dirty = True

    def supports_interaction_mode(self, mode: XRInteractionMode) -> bool:
        """Check if panel supports a specific interaction mode."""
        return mode in self.config.interaction_modes

    def world_to_panel(
        self,
        world_point: tuple[float, float, float]
    ) -> Optional[tuple[float, float]]:
        """Convert world position to panel UV coordinates.

        Args:
            world_point: Point in world space (x, y, z)

        Returns:
            UV coordinates (0-1, 0-1) or None if point is not on panel
        """
        # Simplified implementation - full version would use proper
        # matrix transformations based on panel pose
        px, py, pz = self.position
        wx, wy, wz = world_point

        # Calculate relative position
        dx = wx - px
        dy = wy - py
        dz = wz - pz

        # Simple planar projection (assumes panel faces -Z)
        # Full implementation would apply quaternion rotation
        u = (dx / self.width) + 0.5
        v = (dy / self.height) + 0.5

        # Check if within panel bounds
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            return (u, v)
        return None

    def panel_to_world(
        self,
        uv: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Convert panel UV coordinates to world position.

        Args:
            uv: UV coordinates on panel (0-1, 0-1)

        Returns:
            World space position (x, y, z)
        """
        u, v = uv
        px, py, pz = self.position

        # Convert UV to local offset
        local_x = (u - 0.5) * self.width
        local_y = (v - 0.5) * self.height

        # Simple transformation (full version uses quaternion)
        return (px + local_x, py + local_y, pz)


def xr_ui_panel(
    panel_type: str = "world",
    interaction_mode: str = "ray",
    width: float = 1.0,
    height: float = 0.75,
    curved: bool = False,
    billboard: bool = False,
) -> Callable[[type], type]:
    """Decorator to mark a class as an XR UI panel component.

    Args:
        panel_type: Panel type ("world", "head_locked", "hand_attached", "wrist")
        interaction_mode: Primary interaction mode ("ray", "poke", "gaze", "grab")
        width: Panel width in meters
        height: Panel height in meters
        curved: Whether to use curved display
        billboard: Whether panel always faces camera

    Returns:
        Decorated class with XR UI panel metadata

    Example:
        @xr_ui_panel(panel_type="world", interaction_mode="ray", curved=True)
        class MainMenuPanel:
            pass
    """
    # Map string to enum
    panel_type_map = {
        "world": XRPanelType.WORLD,
        "head_locked": XRPanelType.HEAD_LOCKED,
        "hand_attached": XRPanelType.HAND_ATTACHED,
        "wrist": XRPanelType.WRIST,
    }

    interaction_mode_map = {
        "ray": XRInteractionMode.RAY,
        "poke": XRInteractionMode.POKE,
        "gaze": XRInteractionMode.GAZE,
        "grab": XRInteractionMode.GRAB,
    }

    def decorator(cls: type) -> type:
        # Validate panel type
        if panel_type not in panel_type_map:
            raise ValueError(
                f"Invalid panel_type '{panel_type}'. "
                f"Valid types: {list(panel_type_map.keys())}"
            )

        # Validate interaction mode
        if interaction_mode not in interaction_mode_map:
            raise ValueError(
                f"Invalid interaction_mode '{interaction_mode}'. "
                f"Valid modes: {list(interaction_mode_map.keys())}"
            )

        # Validate dimensions
        if width <= 0:
            raise ValueError("Width must be positive")
        if height <= 0:
            raise ValueError("Height must be positive")

        # Apply metadata
        cls._xr_ui_panel = True
        cls._panel_type = panel_type_map[panel_type]
        cls._interaction_mode = interaction_mode_map[interaction_mode]
        cls._panel_width = width
        cls._panel_height = height
        cls._panel_curved = curved
        cls._panel_billboard = billboard

        # Trinity-style tags
        if not hasattr(cls, '_tags'):
            cls._tags = {}
        cls._tags['xr_ui_panel'] = True
        cls._tags['panel_type'] = panel_type
        cls._tags['interaction_mode'] = interaction_mode

        # Applied decorators tracking
        if not hasattr(cls, '_applied_decorators'):
            cls._applied_decorators = set()
        cls._applied_decorators.add('xr_ui_panel')

        # Registry tracking
        if not hasattr(cls, '_registries'):
            cls._registries = set()
        cls._registries.add('xr')

        return cls

    return decorator


@dataclass(slots=True)
class RaycastHit:
    """Result of a UI raycast."""
    panel: XRUIPanel
    hit_point: tuple[float, float, float]
    uv: tuple[float, float]
    distance: float
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(slots=True)
class PokeInteraction:
    """Data for poke/touch interaction."""
    panel: XRUIPanel
    touch_point: tuple[float, float, float]
    uv: tuple[float, float]
    depth: float  # How far into panel (for press detection)
    finger_id: int = 0


@dataclass(slots=True)
class GazeInteraction:
    """Data for gaze interaction with dwell time."""
    panel: XRUIPanel
    gaze_point: tuple[float, float, float]
    uv: tuple[float, float]
    dwell_time: float  # Seconds gazed at this point
    is_fixating: bool = False


class UIInteractionManager:
    """Manages XR UI interactions across panels.

    Handles raycast, poke, and gaze interactions with UI panels.
    Provides a unified interface for different interaction modes.
    """

    __slots__ = (
        '_panels',
        '_active_rays',
        '_active_pokes',
        '_active_gazes',
        '_interaction_callbacks',
        '_hover_callbacks',
        '_dwell_threshold',
    )

    def __init__(self, dwell_threshold: float = 1.0):
        """Initialize interaction manager.

        Args:
            dwell_threshold: Seconds of gaze required to trigger selection
        """
        self._panels: list[XRUIPanel] = []
        self._active_rays: dict[int, RaycastHit] = {}  # interactor_id -> hit
        self._active_pokes: dict[int, PokeInteraction] = {}
        self._active_gazes: dict[int, GazeInteraction] = {}
        self._interaction_callbacks: list[Callable] = []
        self._hover_callbacks: list[Callable] = []
        self._dwell_threshold = dwell_threshold

    def register_panel(self, panel: XRUIPanel) -> None:
        """Register a panel for interaction tracking."""
        if panel not in self._panels:
            self._panels.append(panel)

    def unregister_panel(self, panel: XRUIPanel) -> None:
        """Unregister a panel from interaction tracking."""
        if panel in self._panels:
            self._panels.remove(panel)

    def raycast(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        interactor_id: int,
        max_distance: float = 100.0,
    ) -> Optional[RaycastHit]:
        """Perform raycast against all registered panels.

        Args:
            origin: Ray origin in world space
            direction: Normalized ray direction
            interactor_id: ID of the interacting entity
            max_distance: Maximum ray distance

        Returns:
            Hit result or None if no panel was hit
        """
        closest_hit: Optional[RaycastHit] = None
        closest_distance = max_distance

        for panel in self._panels:
            if not panel.is_visible or not panel.is_interactable:
                continue
            if not panel.supports_interaction_mode(XRInteractionMode.RAY):
                continue

            # Simplified plane intersection
            # Full version would handle curved panels and rotations
            px, py, pz = panel.position
            ox, oy, oz = origin
            dx, dy, dz = direction

            # Assume panel faces -Z direction
            if abs(dz) < 0.0001:
                continue

            t = (pz - oz) / dz
            if t < 0 or t > closest_distance:
                continue

            hit_x = ox + dx * t
            hit_y = oy + dy * t
            hit_z = oz + dz * t

            # Check if within panel bounds
            uv = panel.world_to_panel((hit_x, hit_y, hit_z))
            if uv is not None:
                hit = RaycastHit(
                    panel=panel,
                    hit_point=(hit_x, hit_y, hit_z),
                    uv=uv,
                    distance=t,
                )
                closest_hit = hit
                closest_distance = t

        # Update active rays
        if closest_hit:
            self._active_rays[interactor_id] = closest_hit
            closest_hit.panel.is_hovered = True
            closest_hit.panel.current_interactor = interactor_id
        elif interactor_id in self._active_rays:
            old_hit = self._active_rays.pop(interactor_id)
            old_hit.panel.is_hovered = False
            old_hit.panel.current_interactor = None

        return closest_hit

    def poke(
        self,
        finger_position: tuple[float, float, float],
        finger_id: int,
        poke_threshold: float = 0.02,
    ) -> Optional[PokeInteraction]:
        """Check for poke/touch interaction with panels.

        Args:
            finger_position: Position of finger tip in world space
            finger_id: ID of the finger/touch point
            poke_threshold: Distance threshold for touch detection

        Returns:
            Poke interaction data or None
        """
        for panel in self._panels:
            if not panel.is_visible or not panel.is_interactable:
                continue
            if not panel.supports_interaction_mode(XRInteractionMode.POKE):
                continue

            # Check distance to panel plane
            px, py, pz = panel.position
            fx, fy, fz = finger_position

            # Distance to panel plane (simplified, assumes -Z facing)
            depth = pz - fz

            if abs(depth) > poke_threshold:
                continue

            # Check if within panel bounds
            uv = panel.world_to_panel(finger_position)
            if uv is not None:
                interaction = PokeInteraction(
                    panel=panel,
                    touch_point=finger_position,
                    uv=uv,
                    depth=max(0, -depth),  # Positive when pressed into panel
                    finger_id=finger_id,
                )
                self._active_pokes[finger_id] = interaction
                panel.is_hovered = True
                return interaction

        # Clear poke if no panel hit
        if finger_id in self._active_pokes:
            old_poke = self._active_pokes.pop(finger_id)
            old_poke.panel.is_hovered = False

        return None

    def gaze(
        self,
        gaze_origin: tuple[float, float, float],
        gaze_direction: tuple[float, float, float],
        delta_time: float,
        user_id: int = 0,
    ) -> Optional[GazeInteraction]:
        """Process gaze interaction with dwell time.

        Args:
            gaze_origin: Eye position in world space
            gaze_direction: Gaze direction (normalized)
            delta_time: Time since last update
            user_id: User identifier

        Returns:
            Gaze interaction with accumulated dwell time
        """
        # First raycast to find target
        hit = None
        for panel in self._panels:
            if not panel.is_visible or not panel.is_interactable:
                continue
            if not panel.supports_interaction_mode(XRInteractionMode.GAZE):
                continue

            px, py, pz = panel.position
            ox, oy, oz = gaze_origin
            dx, dy, dz = gaze_direction

            if abs(dz) < 0.0001:
                continue

            t = (pz - oz) / dz
            if t < 0:
                continue

            hit_x = ox + dx * t
            hit_y = oy + dy * t
            hit_z = oz + dz * t

            uv = panel.world_to_panel((hit_x, hit_y, hit_z))
            if uv is not None:
                hit = (panel, (hit_x, hit_y, hit_z), uv)
                break

        if hit:
            panel, hit_point, uv = hit

            # Check if same target as before
            if user_id in self._active_gazes:
                prev = self._active_gazes[user_id]
                if prev.panel == panel:
                    # Same panel, accumulate dwell time
                    new_dwell = prev.dwell_time + delta_time
                    interaction = GazeInteraction(
                        panel=panel,
                        gaze_point=hit_point,
                        uv=uv,
                        dwell_time=new_dwell,
                        is_fixating=new_dwell >= self._dwell_threshold,
                    )
                else:
                    # Different panel, reset dwell
                    prev.panel.is_hovered = False
                    interaction = GazeInteraction(
                        panel=panel,
                        gaze_point=hit_point,
                        uv=uv,
                        dwell_time=delta_time,
                        is_fixating=False,
                    )
            else:
                # New gaze
                interaction = GazeInteraction(
                    panel=panel,
                    gaze_point=hit_point,
                    uv=uv,
                    dwell_time=delta_time,
                    is_fixating=False,
                )

            self._active_gazes[user_id] = interaction
            panel.is_hovered = True
            return interaction
        else:
            # No target, clear gaze
            if user_id in self._active_gazes:
                old_gaze = self._active_gazes.pop(user_id)
                old_gaze.panel.is_hovered = False
            return None

    def on_interaction(self, callback: Callable) -> None:
        """Register callback for interaction events."""
        self._interaction_callbacks.append(callback)

    def on_hover(self, callback: Callable) -> None:
        """Register callback for hover events."""
        self._hover_callbacks.append(callback)

    def clear(self) -> None:
        """Clear all interaction state."""
        for hit in self._active_rays.values():
            hit.panel.is_hovered = False
            hit.panel.current_interactor = None
        for poke in self._active_pokes.values():
            poke.panel.is_hovered = False
        for gaze in self._active_gazes.values():
            gaze.panel.is_hovered = False

        self._active_rays.clear()
        self._active_pokes.clear()
        self._active_gazes.clear()
