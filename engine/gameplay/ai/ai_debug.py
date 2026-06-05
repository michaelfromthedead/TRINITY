"""
AI Debug - Decorator and utilities for AI visualization debugging.

This module provides the @ai_debug decorator for registering AI debug configurations
with the Foundation Registry, enabling runtime discovery of debug-enabled AI entities.

Usage:
    from engine.gameplay.ai.ai_debug import ai_debug, AIDebugData, get_debug_data

    @ai_debug(enabled=True, show_bt=True, show_perception=True)
    class EnemyAI:
        pass

    # Query all debug configurations:
    >>> from foundation import registry
    >>> registry.query(tag="ai_debug")

    # Get debug data for an entity:
    >>> data = get_debug_data(entity_id)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from foundation import registry, Registry

# Type variable for decorator return types
T = TypeVar("T", bound=type)

# Tag constant for AI debug
TAG_AI_DEBUG = "ai_debug"

# Default visualization colors (RGBA normalized)
DEFAULT_BT_NODE_COLORS = {
    "running": (1.0, 1.0, 0.0, 0.8),    # Yellow
    "success": (0.0, 1.0, 0.0, 0.8),    # Green
    "failure": (1.0, 0.0, 0.0, 0.8),    # Red
    "idle": (0.5, 0.5, 0.5, 0.5),       # Gray
}

DEFAULT_PERCEPTION_COLORS = {
    "sight": (0.0, 1.0, 1.0, 0.3),      # Cyan
    "hearing": (1.0, 0.5, 0.0, 0.3),    # Orange
    "smell": (0.5, 1.0, 0.0, 0.3),      # Lime
}

DEFAULT_INFLUENCE_COLORS = {
    "positive": (0.0, 1.0, 0.0, 0.4),   # Green
    "negative": (1.0, 0.0, 0.0, 0.4),   # Red
    "neutral": (0.5, 0.5, 0.5, 0.2),    # Gray
}


# =============================================================================
# Debug Data Classes
# =============================================================================


class BTNodeDebugStatus(IntEnum):
    """Debug status for BT nodes."""
    IDLE = 0
    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class BTNodeDebugInfo:
    """Debug information for a single BT node."""

    node_id: str
    node_name: str
    node_type: str
    status: BTNodeDebugStatus = BTNodeDebugStatus.IDLE
    execution_time_ms: float = 0.0
    tick_count: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BTDebugState:
    """Complete behavior tree debug state."""

    tree_id: str
    tree_name: str
    root_node_id: Optional[str] = None
    active_node_id: Optional[str] = None
    nodes: Dict[str, BTNodeDebugInfo] = field(default_factory=dict)
    total_ticks: int = 0
    last_tick_time_ms: float = 0.0
    is_running: bool = False

    def add_node(self, node: BTNodeDebugInfo) -> None:
        """Add a node to the debug state."""
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[BTNodeDebugInfo]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_active_path(self) -> List[str]:
        """Get the path from root to active node."""
        if not self.active_node_id:
            return []

        path = []
        current_id = self.active_node_id

        while current_id:
            path.append(current_id)
            node = self.nodes.get(current_id)
            if node:
                current_id = node.parent_id
            else:
                break

        path.reverse()
        return path


@dataclass
class PerceptionRange:
    """Perception range visualization data."""

    sense_type: str
    range_value: float
    fov_degrees: float = 360.0
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # Forward vector
    color: Tuple[float, float, float, float] = field(
        default_factory=lambda: DEFAULT_PERCEPTION_COLORS["sight"]
    )
    is_active: bool = True


@dataclass
class PerceptionDebugState:
    """Perception system debug state."""

    entity_id: int
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # Quaternion
    ranges: Dict[str, PerceptionRange] = field(default_factory=dict)
    detected_entities: List[int] = field(default_factory=list)
    last_update_time: float = 0.0

    def add_range(self, sense_type: str, range_data: PerceptionRange) -> None:
        """Add a perception range."""
        self.ranges[sense_type] = range_data

    def get_range(self, sense_type: str) -> Optional[PerceptionRange]:
        """Get a perception range by sense type."""
        return self.ranges.get(sense_type)


@dataclass
class InfluenceCell:
    """Single cell in an influence map."""

    x: int
    y: int
    value: float = 0.0
    decay_rate: float = 0.1
    source_entity_id: Optional[int] = None


@dataclass
class InfluenceDebugState:
    """Influence map debug state."""

    map_id: str
    map_name: str
    grid_size: Tuple[int, int] = (32, 32)
    cell_size: float = 1.0
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cells: Dict[Tuple[int, int], InfluenceCell] = field(default_factory=dict)
    min_value: float = -1.0
    max_value: float = 1.0
    last_update_time: float = 0.0

    def set_cell(self, x: int, y: int, value: float, source_id: Optional[int] = None) -> None:
        """Set a cell value."""
        cell = self.cells.get((x, y))
        if cell:
            cell.value = value
            cell.source_entity_id = source_id
        else:
            self.cells[(x, y)] = InfluenceCell(x=x, y=y, value=value, source_entity_id=source_id)

    def get_cell(self, x: int, y: int) -> Optional[InfluenceCell]:
        """Get a cell by coordinates."""
        return self.cells.get((x, y))

    def get_value(self, x: int, y: int) -> float:
        """Get the value at a cell."""
        cell = self.cells.get((x, y))
        return cell.value if cell else 0.0


@dataclass
class AIDebugConfig:
    """Configuration for AI debugging."""

    enabled: bool = True
    show_bt: bool = True
    show_perception: bool = True
    show_influence: bool = False
    bt_node_colors: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BT_NODE_COLORS)
    )
    perception_colors: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: dict(DEFAULT_PERCEPTION_COLORS)
    )
    influence_colors: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: dict(DEFAULT_INFLUENCE_COLORS)
    )
    custom_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIDebugData:
    """Complete AI debug data for an entity."""

    entity_id: int
    config: AIDebugConfig
    bt_state: Optional[BTDebugState] = None
    perception_state: Optional[PerceptionDebugState] = None
    influence_states: Dict[str, InfluenceDebugState] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def update_timestamp(self) -> None:
        """Update the timestamp to current time."""
        self.timestamp = time.time()


# =============================================================================
# Debug Data Storage
# =============================================================================


class AIDebugStorage:
    """Thread-safe storage for AI debug data."""

    _instance: Optional[AIDebugStorage] = None
    _lock = threading.Lock()

    def __new__(cls) -> AIDebugStorage:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_storage()
        return cls._instance

    def _init_storage(self) -> None:
        """Initialize storage."""
        self._data: Dict[int, AIDebugData] = {}
        self._storage_lock = threading.Lock()

    def store(self, entity_id: int, data: AIDebugData) -> None:
        """Store debug data for an entity."""
        with self._storage_lock:
            self._data[entity_id] = data

    def get(self, entity_id: int) -> Optional[AIDebugData]:
        """Get debug data for an entity."""
        with self._storage_lock:
            return self._data.get(entity_id)

    def remove(self, entity_id: int) -> bool:
        """Remove debug data for an entity."""
        with self._storage_lock:
            if entity_id in self._data:
                del self._data[entity_id]
                return True
            return False

    def get_all(self) -> Dict[int, AIDebugData]:
        """Get all debug data."""
        with self._storage_lock:
            return dict(self._data)

    def clear(self) -> None:
        """Clear all debug data."""
        with self._storage_lock:
            self._data.clear()

    def count(self) -> int:
        """Get the number of stored debug entries."""
        with self._storage_lock:
            return len(self._data)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Used for testing."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._data.clear()
                cls._instance = None


# Module-level singleton
_debug_storage = AIDebugStorage()


# =============================================================================
# Decorator
# =============================================================================


def ai_debug(
    enabled: bool = True,
    show_bt: bool = True,
    show_perception: bool = True,
    show_influence: bool = False,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    bt_node_colors: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    perception_colors: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    influence_colors: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    custom_options: Optional[Dict[str, Any]] = None,
    track_instances: bool = False,
) -> Callable[[T], T]:
    """
    Decorator to register a class with AI debug configuration in the Foundation Registry.

    This decorator:
    1. Registers the class with the Foundation Registry
    2. Tags it as "ai_debug"
    3. Stores debug configuration metadata

    Args:
        enabled: Whether debug visualization is enabled by default.
        show_bt: Whether to show behavior tree visualization.
        show_perception: Whether to show perception ranges (wireframe cones).
        show_influence: Whether to show influence maps.
        name: Optional custom registry name. Defaults to module.classname.
        description: Optional description of the AI type.
        bt_node_colors: Custom colors for BT node states.
        perception_colors: Custom colors for perception senses.
        influence_colors: Custom colors for influence values.
        custom_options: Additional custom debug options.
        track_instances: If True, track all instances via WeakSet.

    Returns:
        Decorated class registered with Foundation Registry.

    Example:
        @ai_debug(enabled=True, show_bt=True, show_perception=True)
        class EnemyAI:
            pass

        # Query all AI debug configurations:
        >>> from foundation import registry
        >>> registry.query(tag="ai_debug")

        # Query only enabled debug configs:
        >>> registry.query(tag="ai_debug", debug_enabled=True)
    """
    # Build configuration
    config = AIDebugConfig(
        enabled=enabled,
        show_bt=show_bt,
        show_perception=show_perception,
        show_influence=show_influence,
        bt_node_colors=bt_node_colors or dict(DEFAULT_BT_NODE_COLORS),
        perception_colors=perception_colors or dict(DEFAULT_PERCEPTION_COLORS),
        influence_colors=influence_colors or dict(DEFAULT_INFLUENCE_COLORS),
        custom_options=custom_options or {},
    )

    def decorator(cls: T) -> T:
        # Mark class attributes for AI debug identification
        cls._ai_debug = True
        cls._ai_debug_enabled = enabled
        cls._ai_debug_config = config
        cls._ai_debug_description = description or ""

        # Register with Foundation Registry
        already_registered = registry.is_registered(cls)
        if not already_registered:
            try:
                registry.register(cls, name=name, track_instances=track_instances)
            except ValueError:
                # Name conflict with different class - generate unique name
                unique_name = f"{cls.__module__}.{cls.__name__}_{id(cls)}"
                registry.register(cls, name=unique_name, track_instances=track_instances)

        # Add tag for query-based discovery
        registry.add_tag(cls, TAG_AI_DEBUG)

        # Store metadata
        registry.set_metadata(cls, "debug_enabled", enabled)
        registry.set_metadata(cls, "show_bt", show_bt)
        registry.set_metadata(cls, "show_perception", show_perception)
        registry.set_metadata(cls, "show_influence", show_influence)
        registry.set_metadata(cls, "description", description or "")

        # Store color configs as serializable tuples
        registry.set_metadata(cls, "bt_node_colors", config.bt_node_colors)
        registry.set_metadata(cls, "perception_colors", config.perception_colors)
        registry.set_metadata(cls, "influence_colors", config.influence_colors)

        if custom_options:
            registry.set_metadata(cls, "custom_options", custom_options)

        return cls

    return decorator


# =============================================================================
# Debug Data API
# =============================================================================


def get_debug_data(entity_id: int) -> Optional[AIDebugData]:
    """
    Get the current AI debug state for an entity.

    Args:
        entity_id: The entity ID to get debug data for.

    Returns:
        AIDebugData if available, None otherwise.
    """
    return _debug_storage.get(entity_id)


def set_debug_data(entity_id: int, data: AIDebugData) -> None:
    """
    Set AI debug data for an entity.

    Args:
        entity_id: The entity ID to set debug data for.
        data: The debug data to store.
    """
    _debug_storage.store(entity_id, data)


def remove_debug_data(entity_id: int) -> bool:
    """
    Remove AI debug data for an entity.

    Args:
        entity_id: The entity ID to remove debug data for.

    Returns:
        True if data was removed, False if not found.
    """
    return _debug_storage.remove(entity_id)


def get_all_debug_data() -> Dict[int, AIDebugData]:
    """
    Get all stored AI debug data.

    Returns:
        Dictionary mapping entity IDs to debug data.
    """
    return _debug_storage.get_all()


def clear_all_debug_data() -> None:
    """Clear all stored AI debug data."""
    _debug_storage.clear()


def get_debug_data_count() -> int:
    """
    Get the count of stored debug entries.

    Returns:
        Number of debug entries currently stored.
    """
    return _debug_storage.count()


# =============================================================================
# Debug Data Factory Functions
# =============================================================================


def create_debug_data(
    entity_id: int,
    config: Optional[AIDebugConfig] = None,
) -> AIDebugData:
    """
    Create a new AIDebugData instance.

    Args:
        entity_id: The entity ID.
        config: Optional debug config. Uses defaults if not provided.

    Returns:
        New AIDebugData instance.
    """
    return AIDebugData(
        entity_id=entity_id,
        config=config or AIDebugConfig(),
    )


def create_bt_debug_state(
    tree_id: str,
    tree_name: str,
) -> BTDebugState:
    """
    Create a new BTDebugState instance.

    Args:
        tree_id: The behavior tree ID.
        tree_name: The behavior tree name.

    Returns:
        New BTDebugState instance.
    """
    return BTDebugState(tree_id=tree_id, tree_name=tree_name)


def create_perception_debug_state(
    entity_id: int,
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PerceptionDebugState:
    """
    Create a new PerceptionDebugState instance.

    Args:
        entity_id: The entity ID.
        position: Initial position.

    Returns:
        New PerceptionDebugState instance.
    """
    return PerceptionDebugState(entity_id=entity_id, position=position)


def create_influence_debug_state(
    map_id: str,
    map_name: str,
    grid_size: Tuple[int, int] = (32, 32),
    cell_size: float = 1.0,
) -> InfluenceDebugState:
    """
    Create a new InfluenceDebugState instance.

    Args:
        map_id: The influence map ID.
        map_name: The influence map name.
        grid_size: The grid dimensions.
        cell_size: Size of each cell.

    Returns:
        New InfluenceDebugState instance.
    """
    return InfluenceDebugState(
        map_id=map_id,
        map_name=map_name,
        grid_size=grid_size,
        cell_size=cell_size,
    )


# =============================================================================
# Query Helpers
# =============================================================================


def get_all_ai_debug_configs() -> list[type]:
    """Get all registered AI debug configurations."""
    return registry.query(tag=TAG_AI_DEBUG)


def get_enabled_debug_configs() -> list[type]:
    """Get all enabled AI debug configurations."""
    return registry.query(tag=TAG_AI_DEBUG, debug_enabled=True)


def get_debug_configs_with_bt() -> list[type]:
    """Get all AI debug configs that show behavior trees."""
    return registry.query(tag=TAG_AI_DEBUG, show_bt=True)


def get_debug_configs_with_perception() -> list[type]:
    """Get all AI debug configs that show perception."""
    return registry.query(tag=TAG_AI_DEBUG, show_perception=True)


def get_debug_configs_with_influence() -> list[type]:
    """Get all AI debug configs that show influence maps."""
    return registry.query(tag=TAG_AI_DEBUG, show_influence=True)


# =============================================================================
# Runtime Debug Control
# =============================================================================


def enable_debug(cls: type) -> bool:
    """
    Enable debug visualization for a registered AI type.

    Args:
        cls: The registered AI class.

    Returns:
        True if successfully enabled, False if not registered.
    """
    if not registry.is_registered(cls):
        return False

    registry.set_metadata(cls, "debug_enabled", True)
    if hasattr(cls, "_ai_debug_enabled"):
        cls._ai_debug_enabled = True
    if hasattr(cls, "_ai_debug_config"):
        cls._ai_debug_config.enabled = True

    return True


def disable_debug(cls: type) -> bool:
    """
    Disable debug visualization for a registered AI type.

    Args:
        cls: The registered AI class.

    Returns:
        True if successfully disabled, False if not registered.
    """
    if not registry.is_registered(cls):
        return False

    registry.set_metadata(cls, "debug_enabled", False)
    if hasattr(cls, "_ai_debug_enabled"):
        cls._ai_debug_enabled = False
    if hasattr(cls, "_ai_debug_config"):
        cls._ai_debug_config.enabled = False

    return True


def is_debug_enabled(cls: type) -> bool:
    """
    Check if debug visualization is enabled for a registered AI type.

    Args:
        cls: The registered AI class.

    Returns:
        True if debug is enabled, False otherwise.
    """
    if not registry.is_registered(cls):
        return False

    return registry.get_metadata(cls, "debug_enabled") is True


def toggle_debug(cls: type) -> bool:
    """
    Toggle debug visualization for a registered AI type.

    Args:
        cls: The registered AI class.

    Returns:
        The new debug state (True = enabled, False = disabled).

    Raises:
        ValueError: If class is not registered.
    """
    if not registry.is_registered(cls):
        raise ValueError(f"Class {cls.__name__} is not registered")

    current = is_debug_enabled(cls)
    if current:
        disable_debug(cls)
    else:
        enable_debug(cls)

    return not current


def get_debug_config(cls: type) -> Optional[AIDebugConfig]:
    """
    Get the debug configuration for a registered AI type.

    Args:
        cls: The registered AI class.

    Returns:
        AIDebugConfig if available, None otherwise.
    """
    if hasattr(cls, "_ai_debug_config"):
        return cls._ai_debug_config
    return None


# =============================================================================
# Visualization Integration Points
# =============================================================================


@dataclass
class WireframeCone:
    """Wireframe cone data for perception visualization."""

    origin: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    range_value: float
    fov_degrees: float
    color: Tuple[float, float, float, float]
    segments: int = 16


@dataclass
class BTNodeVisualization:
    """BT node visualization data."""

    node_id: str
    node_name: str
    position: Tuple[float, float]  # 2D screen position for tree layout
    size: Tuple[float, float]
    color: Tuple[float, float, float, float]
    is_active: bool
    children: List[str]
    depth: int = 0


def generate_perception_wireframes(
    debug_data: AIDebugData,
) -> List[WireframeCone]:
    """
    Generate wireframe cone data for perception visualization.

    Args:
        debug_data: The AI debug data.

    Returns:
        List of WireframeCone instances for rendering.
    """
    wireframes = []

    if not debug_data.perception_state:
        return wireframes

    perception = debug_data.perception_state
    config = debug_data.config

    for sense_type, perception_range in perception.ranges.items():
        if not perception_range.is_active:
            continue

        color = config.perception_colors.get(
            sense_type,
            DEFAULT_PERCEPTION_COLORS.get("sight", (0.5, 0.5, 0.5, 0.3)),
        )

        wireframes.append(
            WireframeCone(
                origin=perception.position,
                direction=perception_range.direction,
                range_value=perception_range.range_value,
                fov_degrees=perception_range.fov_degrees,
                color=color,
            )
        )

    return wireframes


def generate_bt_visualization(
    debug_data: AIDebugData,
    start_x: float = 0.0,
    start_y: float = 0.0,
    node_width: float = 100.0,
    node_height: float = 40.0,
    h_spacing: float = 20.0,
    v_spacing: float = 60.0,
) -> List[BTNodeVisualization]:
    """
    Generate BT node visualization data for tree layout.

    Args:
        debug_data: The AI debug data.
        start_x: Starting X position.
        start_y: Starting Y position.
        node_width: Width of each node.
        node_height: Height of each node.
        h_spacing: Horizontal spacing between nodes.
        v_spacing: Vertical spacing between levels.

    Returns:
        List of BTNodeVisualization instances for rendering.
    """
    visualizations = []

    if not debug_data.bt_state:
        return visualizations

    bt_state = debug_data.bt_state
    config = debug_data.config
    active_path = set(bt_state.get_active_path())

    def get_node_color(node: BTNodeDebugInfo) -> Tuple[float, float, float, float]:
        status_key = node.status.name.lower()
        return config.bt_node_colors.get(
            status_key,
            DEFAULT_BT_NODE_COLORS.get("idle", (0.5, 0.5, 0.5, 0.5)),
        )

    def layout_node(
        node_id: str,
        x: float,
        y: float,
        depth: int,
    ) -> float:
        """Layout a node and its children, return total width."""
        node = bt_state.get_node(node_id)
        if not node:
            return 0.0

        children_width = 0.0
        children_start_x = x

        # Layout children first
        for child_id in node.children_ids:
            child_width = layout_node(
                child_id,
                children_start_x,
                y + v_spacing,
                depth + 1,
            )
            children_width += child_width + h_spacing
            children_start_x += child_width + h_spacing

        # Remove trailing spacing
        if children_width > 0:
            children_width -= h_spacing

        # Calculate node position (centered over children)
        total_width = max(node_width, children_width)
        node_x = x + (total_width - node_width) / 2

        visualizations.append(
            BTNodeVisualization(
                node_id=node_id,
                node_name=node.node_name,
                position=(node_x, y),
                size=(node_width, node_height),
                color=get_node_color(node),
                is_active=node_id in active_path,
                children=node.children_ids,
                depth=depth,
            )
        )

        return total_width

    # Start layout from root
    if bt_state.root_node_id:
        layout_node(bt_state.root_node_id, start_x, start_y, 0)

    return visualizations


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Main decorator
    "ai_debug",
    # Tag constant
    "TAG_AI_DEBUG",
    # Data classes
    "BTNodeDebugStatus",
    "BTNodeDebugInfo",
    "BTDebugState",
    "PerceptionRange",
    "PerceptionDebugState",
    "InfluenceCell",
    "InfluenceDebugState",
    "AIDebugConfig",
    "AIDebugData",
    # Storage
    "AIDebugStorage",
    # Data API
    "get_debug_data",
    "set_debug_data",
    "remove_debug_data",
    "get_all_debug_data",
    "clear_all_debug_data",
    "get_debug_data_count",
    # Factory functions
    "create_debug_data",
    "create_bt_debug_state",
    "create_perception_debug_state",
    "create_influence_debug_state",
    # Query helpers
    "get_all_ai_debug_configs",
    "get_enabled_debug_configs",
    "get_debug_configs_with_bt",
    "get_debug_configs_with_perception",
    "get_debug_configs_with_influence",
    # Runtime control
    "enable_debug",
    "disable_debug",
    "is_debug_enabled",
    "toggle_debug",
    "get_debug_config",
    # Visualization integration
    "WireframeCone",
    "BTNodeVisualization",
    "generate_perception_wireframes",
    "generate_bt_visualization",
    # Color defaults
    "DEFAULT_BT_NODE_COLORS",
    "DEFAULT_PERCEPTION_COLORS",
    "DEFAULT_INFLUENCE_COLORS",
]
