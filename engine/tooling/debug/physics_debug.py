"""
Physics Debug - Collision shapes, contact points, and raycast visualization.

Provides comprehensive physics debugging tools including collision shape
visualization, contact point display, and ray/shape cast debugging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, ClassVar, Optional, Any
import threading
import time
import math


class CollisionShapeType(Enum):
    """Types of collision shapes."""
    SPHERE = auto()
    BOX = auto()
    CAPSULE = auto()
    CYLINDER = auto()
    CONE = auto()
    CONVEX_HULL = auto()
    MESH = auto()
    COMPOUND = auto()
    PLANE = auto()
    HEIGHT_FIELD = auto()


class PhysicsBodyType(Enum):
    """Types of physics bodies."""
    STATIC = auto()
    DYNAMIC = auto()
    KINEMATIC = auto()
    TRIGGER = auto()


class RaycastResult(Enum):
    """Raycast result states."""
    HIT = auto()
    MISS = auto()
    BLOCKED = auto()


@dataclass(slots=True)
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Vector3":
        length = self.length()
        if length == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / length, self.y / length, self.z / length)


@dataclass(slots=True)
class Quaternion:
    """Quaternion for rotations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(0, 0, 0, 1)


@dataclass
class CollisionShape:
    """Represents a collision shape."""
    shape_id: str
    shape_type: CollisionShapeType
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    extents: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    radius: float = 1.0
    height: float = 2.0
    vertices: list[Vector3] = field(default_factory=list)
    body_type: PhysicsBodyType = PhysicsBodyType.DYNAMIC
    is_sleeping: bool = False
    layer: int = 0
    mask: int = -1


@dataclass
class ContactPoint:
    """Represents a contact point between two bodies."""
    contact_id: int
    position: Vector3
    normal: Vector3
    penetration_depth: float
    body_a_id: str
    body_b_id: str
    impulse: float = 0.0
    friction: float = 0.5
    restitution: float = 0.3
    timestamp: float = 0.0


@dataclass
class RaycastHit:
    """Result of a raycast."""
    hit: bool
    position: Vector3 = field(default_factory=Vector3)
    normal: Vector3 = field(default_factory=Vector3)
    distance: float = 0.0
    body_id: Optional[str] = None
    shape_id: Optional[str] = None
    surface_type: str = "default"


@dataclass
class RaycastRequest:
    """A raycast request for visualization."""
    ray_id: str
    origin: Vector3
    direction: Vector3
    max_distance: float = 1000.0
    result: Optional[RaycastHit] = None
    timestamp: float = 0.0
    lifetime: float = 1.0  # How long to display


class CollisionShapeVisualizer:
    """Visualizes collision shapes."""

    __slots__ = (
        '_shapes',
        '_enabled',
        '_show_sleeping',
        '_show_triggers',
        '_show_static',
        '_color_by_type',
        '_shape_colors',
        '_body_type_colors',
    )

    def __init__(self):
        self._shapes: dict[str, CollisionShape] = {}
        self._enabled = True
        self._show_sleeping = True
        self._show_triggers = True
        self._show_static = True
        self._color_by_type = True

        # Colors for shape types
        self._shape_colors = {
            CollisionShapeType.SPHERE: (0.0, 1.0, 0.0, 0.5),
            CollisionShapeType.BOX: (0.0, 0.5, 1.0, 0.5),
            CollisionShapeType.CAPSULE: (1.0, 0.5, 0.0, 0.5),
            CollisionShapeType.CYLINDER: (1.0, 1.0, 0.0, 0.5),
            CollisionShapeType.CONE: (1.0, 0.0, 1.0, 0.5),
            CollisionShapeType.CONVEX_HULL: (0.5, 1.0, 0.5, 0.5),
            CollisionShapeType.MESH: (0.5, 0.5, 1.0, 0.5),
            CollisionShapeType.COMPOUND: (1.0, 0.5, 0.5, 0.5),
            CollisionShapeType.PLANE: (0.3, 0.3, 0.3, 0.3),
            CollisionShapeType.HEIGHT_FIELD: (0.4, 0.6, 0.4, 0.3),
        }

        # Colors for body types
        self._body_type_colors = {
            PhysicsBodyType.STATIC: (0.5, 0.5, 0.5, 0.5),
            PhysicsBodyType.DYNAMIC: (0.0, 1.0, 0.0, 0.5),
            PhysicsBodyType.KINEMATIC: (0.0, 0.5, 1.0, 0.5),
            PhysicsBodyType.TRIGGER: (1.0, 1.0, 0.0, 0.3),
        }

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def add_shape(self, shape: CollisionShape) -> None:
        """Add a collision shape."""
        self._shapes[shape.shape_id] = shape

    def remove_shape(self, shape_id: str) -> Optional[CollisionShape]:
        """Remove a shape."""
        return self._shapes.pop(shape_id, None)

    def get_shape(self, shape_id: str) -> Optional[CollisionShape]:
        """Get a shape by ID."""
        return self._shapes.get(shape_id)

    def update_shape(
        self,
        shape_id: str,
        position: Optional[Vector3] = None,
        rotation: Optional[Quaternion] = None,
        is_sleeping: Optional[bool] = None,
    ) -> bool:
        """Update shape state."""
        shape = self._shapes.get(shape_id)
        if not shape:
            return False

        if position is not None:
            shape.position = position
        if rotation is not None:
            shape.rotation = rotation
        if is_sleeping is not None:
            shape.is_sleeping = is_sleeping

        return True

    def set_show_sleeping(self, show: bool) -> None:
        self._show_sleeping = show

    def set_show_triggers(self, show: bool) -> None:
        self._show_triggers = show

    def set_show_static(self, show: bool) -> None:
        self._show_static = show

    def set_color_by_type(self, by_type: bool) -> None:
        """Set whether to color by shape type or body type."""
        self._color_by_type = by_type

    def get_shape_color(self, shape: CollisionShape) -> tuple[float, float, float, float]:
        """Get color for a shape."""
        if self._color_by_type:
            return self._shape_colors.get(shape.shape_type, (1.0, 1.0, 1.0, 0.5))
        else:
            return self._body_type_colors.get(shape.body_type, (1.0, 1.0, 1.0, 0.5))

    def generate_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands for all shapes."""
        if not self._enabled:
            return []

        commands = []
        for shape in self._shapes.values():
            # Filter based on settings
            if shape.is_sleeping and not self._show_sleeping:
                continue
            if shape.body_type == PhysicsBodyType.TRIGGER and not self._show_triggers:
                continue
            if shape.body_type == PhysicsBodyType.STATIC and not self._show_static:
                continue

            commands.extend(self._generate_shape_draws(shape))

        return commands

    def _generate_shape_draws(self, shape: CollisionShape) -> list[dict[str, Any]]:
        """Generate draw commands for a single shape."""
        commands = []
        color = self.get_shape_color(shape)

        # Dim sleeping bodies
        if shape.is_sleeping:
            color = (color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, color[3] * 0.5)

        if shape.shape_type == CollisionShapeType.SPHERE:
            commands.append({
                "type": "sphere",
                "center": shape.position.to_tuple(),
                "radius": shape.radius,
                "color": color,
                "wireframe": True,
            })
        elif shape.shape_type == CollisionShapeType.BOX:
            commands.append({
                "type": "box",
                "center": shape.position.to_tuple(),
                "extents": shape.extents.to_tuple(),
                "rotation": shape.rotation.to_tuple(),
                "color": color,
                "wireframe": True,
            })
        elif shape.shape_type == CollisionShapeType.CAPSULE:
            commands.append({
                "type": "capsule",
                "position": shape.position.to_tuple(),
                "radius": shape.radius,
                "height": shape.height,
                "rotation": shape.rotation.to_tuple(),
                "color": color,
                "wireframe": True,
            })
        elif shape.shape_type == CollisionShapeType.CYLINDER:
            commands.append({
                "type": "cylinder",
                "position": shape.position.to_tuple(),
                "radius": shape.radius,
                "height": shape.height,
                "rotation": shape.rotation.to_tuple(),
                "color": color,
                "wireframe": True,
            })
        elif shape.shape_type == CollisionShapeType.CONVEX_HULL:
            if shape.vertices:
                commands.append({
                    "type": "convex_hull",
                    "vertices": [v.to_tuple() for v in shape.vertices],
                    "position": shape.position.to_tuple(),
                    "rotation": shape.rotation.to_tuple(),
                    "color": color,
                    "wireframe": True,
                })
        elif shape.shape_type == CollisionShapeType.PLANE:
            commands.append({
                "type": "plane",
                "position": shape.position.to_tuple(),
                "normal": (0, 1, 0),
                "size": 10.0,
                "color": color,
            })

        return commands

    @property
    def shape_count(self) -> int:
        return len(self._shapes)

    def get_shapes_by_type(self, shape_type: CollisionShapeType) -> list[CollisionShape]:
        """Get all shapes of a specific type."""
        return [s for s in self._shapes.values() if s.shape_type == shape_type]

    def get_shapes_by_layer(self, layer: int) -> list[CollisionShape]:
        """Get all shapes on a specific layer."""
        return [s for s in self._shapes.values() if s.layer == layer]

    def clear_all_shapes(self) -> None:
        """Remove all shapes."""
        self._shapes.clear()


class ContactPointDisplay:
    """Visualizes contact points."""

    __slots__ = (
        '_contacts',
        '_enabled',
        '_max_contacts',
        '_show_normals',
        '_show_impulses',
        '_contact_color',
        '_normal_color',
        '_penetration_threshold',
        '_next_contact_id',
    )

    def __init__(self, max_contacts: int = 1000):
        self._contacts: dict[int, ContactPoint] = {}
        self._enabled = True
        self._max_contacts = max_contacts
        self._show_normals = True
        self._show_impulses = True
        self._contact_color = (1.0, 0.0, 0.0, 1.0)
        self._normal_color = (0.0, 1.0, 1.0, 1.0)
        self._penetration_threshold = 0.01  # Highlight deep penetration
        self._next_contact_id = 0

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def add_contact(self, contact: ContactPoint) -> None:
        """Add a contact point."""
        self._contacts[contact.contact_id] = contact

        # Remove oldest if over limit
        if len(self._contacts) > self._max_contacts:
            oldest = min(self._contacts.values(), key=lambda c: c.timestamp)
            del self._contacts[oldest.contact_id]

    def add_contact_from_data(
        self,
        position: Vector3,
        normal: Vector3,
        penetration_depth: float,
        body_a_id: str,
        body_b_id: str,
        impulse: float = 0.0,
    ) -> ContactPoint:
        """Create and add a contact from data."""
        contact = ContactPoint(
            contact_id=self._next_contact_id,
            position=position,
            normal=normal,
            penetration_depth=penetration_depth,
            body_a_id=body_a_id,
            body_b_id=body_b_id,
            impulse=impulse,
            timestamp=time.time(),
        )
        self._next_contact_id += 1
        self.add_contact(contact)
        return contact

    def remove_contact(self, contact_id: int) -> Optional[ContactPoint]:
        """Remove a contact."""
        return self._contacts.pop(contact_id, None)

    def clear_contacts(self) -> None:
        """Clear all contacts."""
        self._contacts.clear()

    def set_show_normals(self, show: bool) -> None:
        self._show_normals = show

    def set_show_impulses(self, show: bool) -> None:
        self._show_impulses = show

    def generate_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands for contacts."""
        if not self._enabled:
            return []

        commands = []
        for contact in self._contacts.values():
            commands.extend(self._generate_contact_draws(contact))

        return commands

    def _generate_contact_draws(self, contact: ContactPoint) -> list[dict[str, Any]]:
        """Generate draw commands for a single contact."""
        commands = []

        # Contact point
        color = self._contact_color
        if contact.penetration_depth > self._penetration_threshold:
            color = (1.0, 0.5, 0.0, 1.0)  # Orange for deep penetration

        commands.append({
            "type": "point",
            "position": contact.position.to_tuple(),
            "size": 5.0,
            "color": color,
        })

        # Normal arrow
        if self._show_normals:
            normal_end = contact.position + contact.normal * 0.5
            commands.append({
                "type": "arrow",
                "start": contact.position.to_tuple(),
                "end": normal_end.to_tuple(),
                "color": self._normal_color,
            })

        # Impulse visualization
        if self._show_impulses and contact.impulse > 0:
            impulse_scale = min(contact.impulse * 0.1, 2.0)
            impulse_end = contact.position + contact.normal * impulse_scale
            commands.append({
                "type": "arrow",
                "start": contact.position.to_tuple(),
                "end": impulse_end.to_tuple(),
                "color": (1.0, 1.0, 0.0, 1.0),
                "thickness": 2.0,
            })

        return commands

    @property
    def contact_count(self) -> int:
        return len(self._contacts)

    def get_contacts_for_body(self, body_id: str) -> list[ContactPoint]:
        """Get all contacts involving a body."""
        return [
            c for c in self._contacts.values()
            if c.body_a_id == body_id or c.body_b_id == body_id
        ]


class RaycastVisualizer:
    """Visualizes raycasts and their results."""

    __slots__ = (
        '_raycasts',
        '_enabled',
        '_show_misses',
        '_hit_color',
        '_miss_color',
        '_normal_color',
        '_current_time',
    )

    def __init__(self):
        self._raycasts: dict[str, RaycastRequest] = {}
        self._enabled = True
        self._show_misses = True
        self._hit_color = (0.0, 1.0, 0.0, 1.0)
        self._miss_color = (1.0, 0.0, 0.0, 0.5)
        self._normal_color = (0.0, 1.0, 1.0, 1.0)
        self._current_time = time.time()

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def add_raycast(
        self,
        ray_id: str,
        origin: Vector3,
        direction: Vector3,
        max_distance: float = 1000.0,
        result: Optional[RaycastHit] = None,
        lifetime: float = 1.0,
    ) -> RaycastRequest:
        """Add a raycast for visualization."""
        request = RaycastRequest(
            ray_id=ray_id,
            origin=origin,
            direction=direction.normalized(),
            max_distance=max_distance,
            result=result,
            timestamp=time.time(),
            lifetime=lifetime,
        )
        self._raycasts[ray_id] = request
        return request

    def update_result(self, ray_id: str, result: RaycastHit) -> bool:
        """Update the result of a raycast."""
        raycast = self._raycasts.get(ray_id)
        if raycast:
            raycast.result = result
            return True
        return False

    def remove_raycast(self, ray_id: str) -> Optional[RaycastRequest]:
        """Remove a raycast."""
        return self._raycasts.pop(ray_id, None)

    def clear_expired(self) -> int:
        """Clear expired raycasts. Returns count removed."""
        self._current_time = time.time()
        expired = [
            ray_id for ray_id, ray in self._raycasts.items()
            if (self._current_time - ray.timestamp) > ray.lifetime
        ]
        for ray_id in expired:
            del self._raycasts[ray_id]
        return len(expired)

    def set_show_misses(self, show: bool) -> None:
        self._show_misses = show

    def generate_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands for raycasts."""
        if not self._enabled:
            return []

        self.clear_expired()

        commands = []
        for raycast in self._raycasts.values():
            commands.extend(self._generate_raycast_draws(raycast))

        return commands

    def _generate_raycast_draws(self, raycast: RaycastRequest) -> list[dict[str, Any]]:
        """Generate draw commands for a single raycast."""
        commands = []

        if raycast.result and raycast.result.hit:
            # Hit ray
            commands.append({
                "type": "line",
                "start": raycast.origin.to_tuple(),
                "end": raycast.result.position.to_tuple(),
                "color": self._hit_color,
            })

            # Hit point
            commands.append({
                "type": "point",
                "position": raycast.result.position.to_tuple(),
                "size": 8.0,
                "color": self._hit_color,
            })

            # Hit normal
            normal_end = raycast.result.position + raycast.result.normal * 0.5
            commands.append({
                "type": "arrow",
                "start": raycast.result.position.to_tuple(),
                "end": normal_end.to_tuple(),
                "color": self._normal_color,
            })
        elif self._show_misses:
            # Miss ray
            end_point = raycast.origin + raycast.direction * raycast.max_distance
            commands.append({
                "type": "line",
                "start": raycast.origin.to_tuple(),
                "end": end_point.to_tuple(),
                "color": self._miss_color,
            })

        return commands

    @property
    def raycast_count(self) -> int:
        return len(self._raycasts)


class PhysicsDebugger:
    """Central physics debugging system."""

    _instance: ClassVar[Optional["PhysicsDebugger"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    __slots__ = (
        '_shape_viz',
        '_contact_display',
        '_raycast_viz',
        '_enabled',
        '_show_collision_layer_matrix',
        '_collision_layers',
    )

    def __init__(self):
        self._shape_viz = CollisionShapeVisualizer()
        self._contact_display = ContactPointDisplay()
        self._raycast_viz = RaycastVisualizer()
        self._enabled = True
        self._show_collision_layer_matrix = False
        self._collision_layers: dict[int, str] = {}

    @classmethod
    def get_instance(cls) -> "PhysicsDebugger":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def shape_visualizer(self) -> CollisionShapeVisualizer:
        return self._shape_viz

    @property
    def contact_display(self) -> ContactPointDisplay:
        return self._contact_display

    @property
    def raycast_visualizer(self) -> RaycastVisualizer:
        return self._raycast_viz

    def set_collision_layer_name(self, layer: int, name: str) -> None:
        """Set a name for a collision layer."""
        self._collision_layers[layer] = name

    def get_collision_layer_name(self, layer: int) -> str:
        """Get name of a collision layer."""
        return self._collision_layers.get(layer, f"Layer {layer}")

    def generate_all_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands from all subsystems."""
        if not self._enabled:
            return []

        commands = []
        commands.extend(self._shape_viz.generate_draw_commands())
        commands.extend(self._contact_display.generate_draw_commands())
        commands.extend(self._raycast_viz.generate_draw_commands())
        return commands

    def clear_all(self) -> None:
        """Clear all debug visualizations."""
        self._shape_viz.clear_all_shapes()
        self._contact_display.clear_contacts()
        self._raycast_viz._raycasts.clear()

    def get_stats(self) -> dict[str, int]:
        """Get debug stats."""
        return {
            "shapes": self._shape_viz.shape_count,
            "contacts": self._contact_display.contact_count,
            "raycasts": self._raycast_viz.raycast_count,
        }
