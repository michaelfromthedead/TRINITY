"""
Gameplay Debug - AI visualization, nav mesh display, and trigger volume visualization.

Provides tools for debugging gameplay systems including AI decision-making,
pathfinding, and interactive game volumes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, ClassVar, Optional, Any
import threading
import time


class AIState(Enum):
    """AI behavior states for visualization."""
    IDLE = auto()
    PATROL = auto()
    CHASE = auto()
    ATTACK = auto()
    FLEE = auto()
    SEARCH = auto()
    INVESTIGATE = auto()
    DEAD = auto()
    CUSTOM = auto()


class TriggerType(Enum):
    """Types of trigger volumes."""
    BOX = auto()
    SPHERE = auto()
    CAPSULE = auto()
    MESH = auto()
    CUSTOM = auto()


@dataclass(slots=True)
class Vector3:
    """3D vector for positions."""
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


@dataclass
class AIAgent:
    """Represents an AI agent for debugging."""
    agent_id: str
    position: Vector3
    state: AIState = AIState.IDLE
    target: Optional[Vector3] = None
    path: list[Vector3] = field(default_factory=list)
    perception_radius: float = 10.0
    attack_range: float = 2.0
    current_waypoint_index: int = 0
    health: float = 100.0
    max_health: float = 100.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NavMeshPolygon:
    """A polygon in the navigation mesh."""
    polygon_id: int
    vertices: list[Vector3]
    neighbors: list[int] = field(default_factory=list)
    area_type: str = "walkable"
    cost: float = 1.0


@dataclass
class NavMeshConnection:
    """Connection between nav mesh polygons."""
    from_polygon: int
    to_polygon: int
    edge_start: Vector3 = field(default_factory=Vector3)
    edge_end: Vector3 = field(default_factory=Vector3)
    bidirectional: bool = True


@dataclass
class TriggerVolume:
    """Represents a trigger volume in the game world."""
    volume_id: str
    trigger_type: TriggerType
    position: Vector3
    extents: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    radius: float = 1.0
    enabled: bool = True
    triggered: bool = False
    trigger_count: int = 0
    tags: list[str] = field(default_factory=list)
    on_enter: Optional[str] = None
    on_exit: Optional[str] = None
    on_stay: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AIVisualization:
    """Visualizes AI agent behavior and decision-making."""

    __slots__ = (
        '_agents',
        '_enabled',
        '_show_paths',
        '_show_perception',
        '_show_targets',
        '_show_states',
        '_show_health',
        '_path_color',
        '_perception_color',
        '_target_color',
        '_state_colors',
    )

    def __init__(self):
        self._agents: dict[str, AIAgent] = {}
        self._enabled = True
        self._show_paths = True
        self._show_perception = True
        self._show_targets = True
        self._show_states = True
        self._show_health = True

        # Default colors (RGBA)
        self._path_color = (0.0, 1.0, 0.0, 0.8)
        self._perception_color = (0.5, 0.5, 1.0, 0.3)
        self._target_color = (1.0, 0.0, 0.0, 1.0)
        self._state_colors = {
            AIState.IDLE: (0.5, 0.5, 0.5, 1.0),
            AIState.PATROL: (0.0, 1.0, 0.0, 1.0),
            AIState.CHASE: (1.0, 0.5, 0.0, 1.0),
            AIState.ATTACK: (1.0, 0.0, 0.0, 1.0),
            AIState.FLEE: (1.0, 1.0, 0.0, 1.0),
            AIState.SEARCH: (0.0, 0.5, 1.0, 1.0),
            AIState.INVESTIGATE: (0.5, 0.0, 1.0, 1.0),
            AIState.DEAD: (0.2, 0.2, 0.2, 1.0),
            AIState.CUSTOM: (1.0, 1.0, 1.0, 1.0),
        }

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def register_agent(self, agent: AIAgent) -> None:
        """Register an AI agent for visualization."""
        self._agents[agent.agent_id] = agent

    def unregister_agent(self, agent_id: str) -> Optional[AIAgent]:
        """Unregister an agent."""
        return self._agents.pop(agent_id, None)

    def get_agent(self, agent_id: str) -> Optional[AIAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def update_agent(
        self,
        agent_id: str,
        position: Optional[Vector3] = None,
        state: Optional[AIState] = None,
        target: Optional[Vector3] = None,
        path: Optional[list[Vector3]] = None,
        health: Optional[float] = None,
    ) -> bool:
        """Update agent state."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        if position is not None:
            agent.position = position
        if state is not None:
            agent.state = state
        if target is not None:
            agent.target = target
        if path is not None:
            agent.path = path
            agent.current_waypoint_index = 0
        if health is not None:
            agent.health = health

        return True

    def set_show_paths(self, show: bool) -> None:
        self._show_paths = show

    def set_show_perception(self, show: bool) -> None:
        self._show_perception = show

    def set_show_targets(self, show: bool) -> None:
        self._show_targets = show

    def set_show_states(self, show: bool) -> None:
        self._show_states = show

    def set_show_health(self, show: bool) -> None:
        self._show_health = show

    def get_state_color(self, state: AIState) -> tuple[float, float, float, float]:
        """Get the color for a state."""
        return self._state_colors.get(state, self._state_colors[AIState.CUSTOM])

    def set_state_color(self, state: AIState, color: tuple[float, float, float, float]) -> None:
        """Set the color for a state."""
        self._state_colors[state] = color

    def generate_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands for all agents."""
        if not self._enabled:
            return []

        commands = []
        for agent in self._agents.values():
            commands.extend(self._generate_agent_draws(agent))
        return commands

    def _generate_agent_draws(self, agent: AIAgent) -> list[dict[str, Any]]:
        """Generate draw commands for a single agent."""
        commands = []

        # Agent position sphere
        state_color = self.get_state_color(agent.state)
        commands.append({
            "type": "sphere",
            "position": agent.position.to_tuple(),
            "radius": 0.5,
            "color": state_color,
        })

        # Path visualization
        if self._show_paths and agent.path:
            for i in range(len(agent.path) - 1):
                commands.append({
                    "type": "line",
                    "start": agent.path[i].to_tuple(),
                    "end": agent.path[i + 1].to_tuple(),
                    "color": self._path_color,
                })

            # Current waypoint
            if agent.current_waypoint_index < len(agent.path):
                commands.append({
                    "type": "sphere",
                    "position": agent.path[agent.current_waypoint_index].to_tuple(),
                    "radius": 0.3,
                    "color": (1.0, 1.0, 0.0, 1.0),
                })

        # Perception radius
        if self._show_perception:
            commands.append({
                "type": "circle",
                "center": agent.position.to_tuple(),
                "radius": agent.perception_radius,
                "normal": (0.0, 1.0, 0.0),
                "color": self._perception_color,
            })

        # Target line
        if self._show_targets and agent.target:
            commands.append({
                "type": "arrow",
                "start": agent.position.to_tuple(),
                "end": agent.target.to_tuple(),
                "color": self._target_color,
            })

        # State text
        if self._show_states:
            text_pos = agent.position + Vector3(0, 2.0, 0)
            commands.append({
                "type": "text",
                "position": text_pos.to_tuple(),
                "text": f"{agent.agent_id}: {agent.state.name}",
                "color": state_color,
            })

        # Health bar
        if self._show_health:
            health_pos = agent.position + Vector3(0, 1.5, 0)
            health_pct = agent.health / agent.max_health if agent.max_health > 0 else 0
            commands.append({
                "type": "health_bar",
                "position": health_pos.to_tuple(),
                "percentage": health_pct,
                "width": 1.0,
            })

        return commands

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    def get_agents_by_state(self, state: AIState) -> list[AIAgent]:
        """Get all agents in a specific state."""
        return [a for a in self._agents.values() if a.state == state]

    def clear_all_agents(self) -> None:
        """Remove all agents."""
        self._agents.clear()


class NavMeshDisplay:
    """Visualizes navigation meshes."""

    __slots__ = (
        '_polygons',
        '_connections',
        '_enabled',
        '_show_polygons',
        '_show_connections',
        '_show_costs',
        '_polygon_color',
        '_connection_color',
        '_walkable_color',
        '_unwalkable_color',
        '_selected_polygon',
    )

    def __init__(self):
        self._polygons: dict[int, NavMeshPolygon] = {}
        self._connections: list[NavMeshConnection] = []
        self._enabled = True
        self._show_polygons = True
        self._show_connections = True
        self._show_costs = False

        # Default colors
        self._polygon_color = (0.0, 0.5, 1.0, 0.3)
        self._connection_color = (1.0, 1.0, 0.0, 0.5)
        self._walkable_color = (0.0, 1.0, 0.0, 0.3)
        self._unwalkable_color = (1.0, 0.0, 0.0, 0.3)
        self._selected_polygon: Optional[int] = None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def add_polygon(self, polygon: NavMeshPolygon) -> None:
        """Add a nav mesh polygon."""
        self._polygons[polygon.polygon_id] = polygon

    def remove_polygon(self, polygon_id: int) -> Optional[NavMeshPolygon]:
        """Remove a polygon."""
        return self._polygons.pop(polygon_id, None)

    def get_polygon(self, polygon_id: int) -> Optional[NavMeshPolygon]:
        """Get a polygon by ID."""
        return self._polygons.get(polygon_id)

    def add_connection(self, connection: NavMeshConnection) -> None:
        """Add a connection between polygons."""
        self._connections.append(connection)

    def clear(self) -> None:
        """Clear all nav mesh data."""
        self._polygons.clear()
        self._connections.clear()

    def set_show_polygons(self, show: bool) -> None:
        self._show_polygons = show

    def set_show_connections(self, show: bool) -> None:
        self._show_connections = show

    def set_show_costs(self, show: bool) -> None:
        self._show_costs = show

    def select_polygon(self, polygon_id: Optional[int]) -> None:
        """Select a polygon for highlighting."""
        self._selected_polygon = polygon_id

    def generate_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands for the nav mesh."""
        if not self._enabled:
            return []

        commands = []

        # Draw polygons
        if self._show_polygons:
            for polygon in self._polygons.values():
                color = self._walkable_color if polygon.area_type == "walkable" else self._unwalkable_color
                if polygon.polygon_id == self._selected_polygon:
                    color = (1.0, 1.0, 0.0, 0.5)  # Highlight selected

                commands.append({
                    "type": "polygon",
                    "vertices": [v.to_tuple() for v in polygon.vertices],
                    "color": color,
                })

                # Draw polygon edges
                for i in range(len(polygon.vertices)):
                    next_i = (i + 1) % len(polygon.vertices)
                    commands.append({
                        "type": "line",
                        "start": polygon.vertices[i].to_tuple(),
                        "end": polygon.vertices[next_i].to_tuple(),
                        "color": self._polygon_color,
                    })

                # Show cost
                if self._show_costs and polygon.vertices:
                    center = Vector3(
                        sum(v.x for v in polygon.vertices) / len(polygon.vertices),
                        sum(v.y for v in polygon.vertices) / len(polygon.vertices) + 0.5,
                        sum(v.z for v in polygon.vertices) / len(polygon.vertices),
                    )
                    commands.append({
                        "type": "text",
                        "position": center.to_tuple(),
                        "text": f"{polygon.cost:.1f}",
                        "color": (1.0, 1.0, 1.0, 1.0),
                    })

        # Draw connections
        if self._show_connections:
            for conn in self._connections:
                commands.append({
                    "type": "line",
                    "start": conn.edge_start.to_tuple(),
                    "end": conn.edge_end.to_tuple(),
                    "color": self._connection_color,
                    "thickness": 2.0,
                })

        return commands

    @property
    def polygon_count(self) -> int:
        return len(self._polygons)

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    def find_polygon_at(self, position: Vector3) -> Optional[int]:
        """Find polygon containing a position (simplified point-in-polygon)."""
        for polygon in self._polygons.values():
            if self._point_in_polygon(position, polygon.vertices):
                return polygon.polygon_id
        return None

    def _point_in_polygon(self, point: Vector3, vertices: list[Vector3]) -> bool:
        """Simple 2D point-in-polygon test (XZ plane)."""
        if len(vertices) < 3:
            return False

        n = len(vertices)
        inside = False

        j = n - 1
        for i in range(n):
            vi = vertices[i]
            vj = vertices[j]

            if ((vi.z > point.z) != (vj.z > point.z)) and \
               (point.x < (vj.x - vi.x) * (point.z - vi.z) / (vj.z - vi.z) + vi.x):
                inside = not inside

            j = i

        return inside


class TriggerVolumeVisualizer:
    """Visualizes trigger volumes in the game world."""

    __slots__ = (
        '_volumes',
        '_enabled',
        '_show_bounds',
        '_show_names',
        '_show_events',
        '_default_color',
        '_triggered_color',
        '_disabled_color',
    )

    def __init__(self):
        self._volumes: dict[str, TriggerVolume] = {}
        self._enabled = True
        self._show_bounds = True
        self._show_names = True
        self._show_events = True

        # Default colors
        self._default_color = (0.0, 1.0, 1.0, 0.3)
        self._triggered_color = (1.0, 1.0, 0.0, 0.5)
        self._disabled_color = (0.5, 0.5, 0.5, 0.2)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def add_volume(self, volume: TriggerVolume) -> None:
        """Add a trigger volume."""
        self._volumes[volume.volume_id] = volume

    def remove_volume(self, volume_id: str) -> Optional[TriggerVolume]:
        """Remove a volume."""
        return self._volumes.pop(volume_id, None)

    def get_volume(self, volume_id: str) -> Optional[TriggerVolume]:
        """Get a volume by ID."""
        return self._volumes.get(volume_id)

    def set_triggered(self, volume_id: str, triggered: bool) -> bool:
        """Set triggered state of a volume."""
        volume = self._volumes.get(volume_id)
        if volume:
            if triggered and not volume.triggered:
                volume.trigger_count += 1
            volume.triggered = triggered
            return True
        return False

    def set_enabled(self, volume_id: str, enabled: bool) -> bool:
        """Enable/disable a volume."""
        volume = self._volumes.get(volume_id)
        if volume:
            volume.enabled = enabled
            return True
        return False

    def set_show_bounds(self, show: bool) -> None:
        self._show_bounds = show

    def set_show_names(self, show: bool) -> None:
        self._show_names = show

    def set_show_events(self, show: bool) -> None:
        self._show_events = show

    def generate_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands for all volumes."""
        if not self._enabled:
            return []

        commands = []
        for volume in self._volumes.values():
            commands.extend(self._generate_volume_draws(volume))
        return commands

    def _generate_volume_draws(self, volume: TriggerVolume) -> list[dict[str, Any]]:
        """Generate draw commands for a single volume."""
        commands = []

        # Determine color
        if not volume.enabled:
            color = self._disabled_color
        elif volume.triggered:
            color = self._triggered_color
        else:
            color = self._default_color

        # Draw bounds
        if self._show_bounds:
            if volume.trigger_type == TriggerType.BOX:
                commands.append({
                    "type": "box",
                    "center": volume.position.to_tuple(),
                    "extents": volume.extents.to_tuple(),
                    "color": color,
                    "wireframe": True,
                })
            elif volume.trigger_type == TriggerType.SPHERE:
                commands.append({
                    "type": "sphere",
                    "center": volume.position.to_tuple(),
                    "radius": volume.radius,
                    "color": color,
                    "wireframe": True,
                })
            elif volume.trigger_type == TriggerType.CAPSULE:
                commands.append({
                    "type": "capsule",
                    "position": volume.position.to_tuple(),
                    "radius": volume.radius,
                    "height": volume.extents.y,
                    "color": color,
                    "wireframe": True,
                })

        # Draw name
        if self._show_names:
            text_pos = volume.position + Vector3(0, volume.extents.y + 0.5, 0)
            commands.append({
                "type": "text",
                "position": text_pos.to_tuple(),
                "text": volume.volume_id,
                "color": (1.0, 1.0, 1.0, 1.0),
            })

        # Draw events
        if self._show_events:
            events = []
            if volume.on_enter:
                events.append(f"Enter: {volume.on_enter}")
            if volume.on_exit:
                events.append(f"Exit: {volume.on_exit}")
            if volume.on_stay:
                events.append(f"Stay: {volume.on_stay}")

            if events:
                text_pos = volume.position + Vector3(0, volume.extents.y + 1.0, 0)
                commands.append({
                    "type": "text",
                    "position": text_pos.to_tuple(),
                    "text": "\n".join(events),
                    "color": (0.8, 0.8, 0.8, 1.0),
                    "scale": 0.8,
                })

        return commands

    @property
    def volume_count(self) -> int:
        return len(self._volumes)

    def get_triggered_volumes(self) -> list[TriggerVolume]:
        """Get all currently triggered volumes."""
        return [v for v in self._volumes.values() if v.triggered]

    def get_volumes_by_tag(self, tag: str) -> list[TriggerVolume]:
        """Get volumes with a specific tag."""
        return [v for v in self._volumes.values() if tag in v.tags]

    def clear_all_volumes(self) -> None:
        """Remove all volumes."""
        self._volumes.clear()


class GameplayDebugger:
    """Central gameplay debugging system."""

    _instance: ClassVar[Optional["GameplayDebugger"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    __slots__ = (
        '_ai_viz',
        '_nav_mesh',
        '_trigger_viz',
        '_enabled',
    )

    def __init__(self):
        self._ai_viz = AIVisualization()
        self._nav_mesh = NavMeshDisplay()
        self._trigger_viz = TriggerVolumeVisualizer()
        self._enabled = True

    @classmethod
    def get_instance(cls) -> "GameplayDebugger":
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
    def ai_visualization(self) -> AIVisualization:
        return self._ai_viz

    @property
    def nav_mesh_display(self) -> NavMeshDisplay:
        return self._nav_mesh

    @property
    def trigger_visualizer(self) -> TriggerVolumeVisualizer:
        return self._trigger_viz

    def generate_all_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands from all subsystems."""
        if not self._enabled:
            return []

        commands = []
        commands.extend(self._ai_viz.generate_draw_commands())
        commands.extend(self._nav_mesh.generate_draw_commands())
        commands.extend(self._trigger_viz.generate_draw_commands())
        return commands

    def clear_all(self) -> None:
        """Clear all debug visualizations."""
        self._ai_viz.clear_all_agents()
        self._nav_mesh.clear()
        self._trigger_viz.clear_all_volumes()
