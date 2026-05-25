"""Visual animation graph editor for state machines and blend trees.

Provides a node-based editor for creating animation graphs with
state machines, blend trees, and IK nodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from engine.core.math import Vec2, Vec3


# =============================================================================
# ENUMS
# =============================================================================


class SocketType(Enum):
    """Types of node sockets."""

    POSE = auto()      # Animation pose data
    FLOAT = auto()     # Float value
    INT = auto()       # Integer value
    BOOL = auto()      # Boolean value
    VECTOR = auto()    # Vector value
    BONE_REF = auto()  # Bone reference
    CURVE = auto()     # Animation curve


class BlendType(Enum):
    """Types of blend operations."""

    LINEAR = auto()        # Linear interpolation
    ADDITIVE = auto()      # Additive blending
    MESH_SPACE = auto()    # Mesh space additive
    LOCAL_SPACE = auto()   # Local space additive


# =============================================================================
# GRAPH SOCKET
# =============================================================================


@dataclass
class GraphSocket:
    """A socket on a graph node for connections.

    Attributes:
        name: Socket name
        socket_type: Type of data
        is_input: True for input, False for output
        default_value: Default value if not connected
        multi_connect: Whether multiple connections are allowed
    """

    name: str
    socket_type: SocketType
    is_input: bool = True
    default_value: Any = None
    multi_connect: bool = False

    def can_connect_to(self, other: GraphSocket) -> bool:
        """Check if this socket can connect to another."""
        # Must be opposite directions
        if self.is_input == other.is_input:
            return False
        # Types must match (or be convertible)
        if self.socket_type != other.socket_type:
            # Allow float -> int conversion
            if {self.socket_type, other.socket_type} <= {SocketType.FLOAT, SocketType.INT}:
                return True
            return False
        return True


# =============================================================================
# GRAPH CONNECTION
# =============================================================================


@dataclass
class GraphConnection:
    """A connection between two node sockets.

    Attributes:
        source_node: Source node ID
        source_socket: Source socket name
        target_node: Target node ID
        target_socket: Target socket name
    """

    source_node: str
    source_socket: str
    target_node: str
    target_socket: str

    def __post_init__(self) -> None:
        if not self.source_node or not self.target_node:
            raise ValueError("Node IDs cannot be empty")
        if not self.source_socket or not self.target_socket:
            raise ValueError("Socket names cannot be empty")


# =============================================================================
# GRAPH NODE (BASE)
# =============================================================================


class GraphNode(ABC):
    """Base class for graph nodes.

    Nodes are the building blocks of animation graphs. Each node
    processes inputs and produces outputs.
    """

    def __init__(self, node_id: str, position: Vec2 = None) -> None:
        if not node_id:
            raise ValueError("Node ID cannot be empty")

        self._node_id = node_id
        self._position = position or Vec2(0, 0)
        self._inputs: List[GraphSocket] = []
        self._outputs: List[GraphSocket] = []
        self._collapsed = False
        self._comment = ""

    @property
    def node_id(self) -> str:
        """Get node ID."""
        return self._node_id

    @property
    def position(self) -> Vec2:
        """Get node position."""
        return self._position

    @position.setter
    def position(self, value: Vec2) -> None:
        """Set node position."""
        self._position = value

    @property
    def inputs(self) -> List[GraphSocket]:
        """Get input sockets."""
        return list(self._inputs)

    @property
    def outputs(self) -> List[GraphSocket]:
        """Get output sockets."""
        return list(self._outputs)

    @property
    def collapsed(self) -> bool:
        """Check if node is collapsed in UI."""
        return self._collapsed

    @collapsed.setter
    def collapsed(self, value: bool) -> None:
        """Set collapsed state."""
        self._collapsed = value

    @property
    def comment(self) -> str:
        """Get node comment."""
        return self._comment

    @comment.setter
    def comment(self, value: str) -> None:
        """Set node comment."""
        self._comment = value

    @property
    @abstractmethod
    def node_type(self) -> str:
        """Get node type name."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Get display name for UI."""
        pass

    def add_input(self, socket: GraphSocket) -> None:
        """Add an input socket."""
        if not socket.is_input:
            raise ValueError("Socket must be an input")
        self._inputs.append(socket)

    def add_output(self, socket: GraphSocket) -> None:
        """Add an output socket."""
        if socket.is_input:
            raise ValueError("Socket must be an output")
        self._outputs.append(socket)

    def get_input(self, name: str) -> Optional[GraphSocket]:
        """Get input socket by name."""
        for socket in self._inputs:
            if socket.name == name:
                return socket
        return None

    def get_output(self, name: str) -> Optional[GraphSocket]:
        """Get output socket by name."""
        for socket in self._outputs:
            if socket.name == name:
                return socket
        return None


# =============================================================================
# STATE MACHINE NODES
# =============================================================================


class StateNode(GraphNode):
    """A state in the animation state machine.

    States represent animation states with associated animations
    and state-specific settings.
    """

    def __init__(
        self,
        node_id: str,
        state_name: str,
        position: Vec2 = None,
    ) -> None:
        super().__init__(node_id, position)
        self._state_name = state_name
        self._animation_asset: Optional[str] = None
        self._speed_multiplier = 1.0
        self._loop = True
        self._is_entry = False

        # Output pose
        self.add_output(GraphSocket(
            name="Pose",
            socket_type=SocketType.POSE,
            is_input=False,
        ))

    @property
    def node_type(self) -> str:
        return "State"

    @property
    def display_name(self) -> str:
        return self._state_name

    @property
    def state_name(self) -> str:
        """Get state name."""
        return self._state_name

    @state_name.setter
    def state_name(self, value: str) -> None:
        """Set state name."""
        if not value:
            raise ValueError("State name cannot be empty")
        self._state_name = value

    @property
    def animation_asset(self) -> Optional[str]:
        """Get animation asset path."""
        return self._animation_asset

    @animation_asset.setter
    def animation_asset(self, value: Optional[str]) -> None:
        """Set animation asset path."""
        self._animation_asset = value

    @property
    def speed_multiplier(self) -> float:
        """Get speed multiplier."""
        return self._speed_multiplier

    @speed_multiplier.setter
    def speed_multiplier(self, value: float) -> None:
        """Set speed multiplier."""
        self._speed_multiplier = value

    @property
    def loop(self) -> bool:
        """Check if animation loops."""
        return self._loop

    @loop.setter
    def loop(self, value: bool) -> None:
        """Set loop state."""
        self._loop = value

    @property
    def is_entry(self) -> bool:
        """Check if this is the entry state."""
        return self._is_entry

    @is_entry.setter
    def is_entry(self, value: bool) -> None:
        """Set entry state flag."""
        self._is_entry = value


class TransitionNode(GraphNode):
    """A transition between states.

    Transitions define conditions and blend settings for moving
    between animation states.
    """

    def __init__(
        self,
        node_id: str,
        position: Vec2 = None,
    ) -> None:
        super().__init__(node_id, position)
        self._conditions: List[Dict[str, Any]] = []
        self._blend_duration = 0.25
        self._blend_curve = "ease_in_out"
        self._exit_time: Optional[float] = None
        self._priority = 0

    @property
    def node_type(self) -> str:
        return "Transition"

    @property
    def display_name(self) -> str:
        return "Transition"

    @property
    def conditions(self) -> List[Dict[str, Any]]:
        """Get transition conditions."""
        return list(self._conditions)

    @property
    def blend_duration(self) -> float:
        """Get blend duration."""
        return self._blend_duration

    @blend_duration.setter
    def blend_duration(self, value: float) -> None:
        """Set blend duration."""
        self._blend_duration = max(0.0, value)

    @property
    def exit_time(self) -> Optional[float]:
        """Get exit time (normalized 0-1)."""
        return self._exit_time

    @exit_time.setter
    def exit_time(self, value: Optional[float]) -> None:
        """Set exit time."""
        if value is not None:
            value = max(0.0, min(1.0, value))
        self._exit_time = value

    @property
    def priority(self) -> int:
        """Get transition priority."""
        return self._priority

    @priority.setter
    def priority(self, value: int) -> None:
        """Set transition priority."""
        self._priority = value

    def add_condition(
        self,
        parameter: str,
        comparison: str,
        value: Any,
    ) -> None:
        """Add a transition condition."""
        self._conditions.append({
            "parameter": parameter,
            "comparison": comparison,
            "value": value,
        })

    def remove_condition(self, index: int) -> bool:
        """Remove a condition by index."""
        if 0 <= index < len(self._conditions):
            self._conditions.pop(index)
            return True
        return False

    def clear_conditions(self) -> None:
        """Clear all conditions."""
        self._conditions.clear()


class ConduitNode(GraphNode):
    """A conduit for routing transitions.

    Conduits allow grouping multiple transitions and applying
    shared conditions or logic.
    """

    def __init__(
        self,
        node_id: str,
        position: Vec2 = None,
    ) -> None:
        super().__init__(node_id, position)

        self.add_input(GraphSocket(
            name="Entry",
            socket_type=SocketType.POSE,
            is_input=True,
            multi_connect=True,
        ))
        self.add_output(GraphSocket(
            name="Exit",
            socket_type=SocketType.POSE,
            is_input=False,
        ))

    @property
    def node_type(self) -> str:
        return "Conduit"

    @property
    def display_name(self) -> str:
        return "Conduit"


class EntryNode(GraphNode):
    """Entry point for a state machine."""

    def __init__(
        self,
        node_id: str,
        position: Vec2 = None,
    ) -> None:
        super().__init__(node_id, position)

        self.add_output(GraphSocket(
            name="Default",
            socket_type=SocketType.POSE,
            is_input=False,
        ))

    @property
    def node_type(self) -> str:
        return "Entry"

    @property
    def display_name(self) -> str:
        return "Entry"


# =============================================================================
# BLEND TREE NODES
# =============================================================================


class BlendNode(GraphNode):
    """Basic blend node for combining poses."""

    def __init__(
        self,
        node_id: str,
        blend_type: BlendType = BlendType.LINEAR,
        position: Vec2 = None,
    ) -> None:
        super().__init__(node_id, position)
        self._blend_type = blend_type

        self.add_input(GraphSocket(
            name="Pose A",
            socket_type=SocketType.POSE,
            is_input=True,
        ))
        self.add_input(GraphSocket(
            name="Pose B",
            socket_type=SocketType.POSE,
            is_input=True,
        ))
        self.add_input(GraphSocket(
            name="Alpha",
            socket_type=SocketType.FLOAT,
            is_input=True,
            default_value=0.5,
        ))
        self.add_output(GraphSocket(
            name="Pose",
            socket_type=SocketType.POSE,
            is_input=False,
        ))

    @property
    def node_type(self) -> str:
        return "Blend"

    @property
    def display_name(self) -> str:
        return f"Blend ({self._blend_type.name})"

    @property
    def blend_type(self) -> BlendType:
        """Get blend type."""
        return self._blend_type

    @blend_type.setter
    def blend_type(self, value: BlendType) -> None:
        """Set blend type."""
        self._blend_type = value


class AdditiveNode(GraphNode):
    """Node for additive pose blending."""

    def __init__(
        self,
        node_id: str,
        position: Vec2 = None,
    ) -> None:
        super().__init__(node_id, position)
        self._mesh_space_additive = False

        self.add_input(GraphSocket(
            name="Base Pose",
            socket_type=SocketType.POSE,
            is_input=True,
        ))
        self.add_input(GraphSocket(
            name="Additive Pose",
            socket_type=SocketType.POSE,
            is_input=True,
        ))
        self.add_input(GraphSocket(
            name="Alpha",
            socket_type=SocketType.FLOAT,
            is_input=True,
            default_value=1.0,
        ))
        self.add_output(GraphSocket(
            name="Pose",
            socket_type=SocketType.POSE,
            is_input=False,
        ))

    @property
    def node_type(self) -> str:
        return "Additive"

    @property
    def display_name(self) -> str:
        return "Apply Additive"

    @property
    def mesh_space_additive(self) -> bool:
        """Check if using mesh-space additive."""
        return self._mesh_space_additive

    @mesh_space_additive.setter
    def mesh_space_additive(self, value: bool) -> None:
        """Set mesh-space additive mode."""
        self._mesh_space_additive = value


class LayeredBlendNode(GraphNode):
    """Node for layered blend with bone masks."""

    def __init__(
        self,
        node_id: str,
        num_layers: int = 2,
        position: Vec2 = None,
    ) -> None:
        super().__init__(node_id, position)
        self._bone_masks: List[List[str]] = [[] for _ in range(num_layers)]
        self._blend_modes: List[BlendType] = [BlendType.LINEAR for _ in range(num_layers)]

        self.add_input(GraphSocket(
            name="Base Pose",
            socket_type=SocketType.POSE,
            is_input=True,
        ))

        for i in range(num_layers):
            self.add_input(GraphSocket(
                name=f"Layer {i}",
                socket_type=SocketType.POSE,
                is_input=True,
            ))
            self.add_input(GraphSocket(
                name=f"Layer {i} Weight",
                socket_type=SocketType.FLOAT,
                is_input=True,
                default_value=0.0,
            ))

        self.add_output(GraphSocket(
            name="Pose",
            socket_type=SocketType.POSE,
            is_input=False,
        ))

    @property
    def node_type(self) -> str:
        return "LayeredBlend"

    @property
    def display_name(self) -> str:
        return "Layered Blend"

    @property
    def num_layers(self) -> int:
        """Get number of blend layers."""
        return len(self._bone_masks)

    def set_layer_mask(self, layer: int, bones: List[str]) -> None:
        """Set bone mask for a layer."""
        if 0 <= layer < len(self._bone_masks):
            self._bone_masks[layer] = list(bones)

    def get_layer_mask(self, layer: int) -> List[str]:
        """Get bone mask for a layer."""
        if 0 <= layer < len(self._bone_masks):
            return list(self._bone_masks[layer])
        return []

    def set_layer_blend_mode(self, layer: int, mode: BlendType) -> None:
        """Set blend mode for a layer."""
        if 0 <= layer < len(self._blend_modes):
            self._blend_modes[layer] = mode


class BoneMaskNode(GraphNode):
    """Node for filtering pose by bone mask."""

    def __init__(
        self,
        node_id: str,
        position: Vec2 = None,
    ) -> None:
        super().__init__(node_id, position)
        self._included_bones: List[str] = []
        self._excluded_bones: List[str] = []

        self.add_input(GraphSocket(
            name="Pose",
            socket_type=SocketType.POSE,
            is_input=True,
        ))
        self.add_output(GraphSocket(
            name="Masked Pose",
            socket_type=SocketType.POSE,
            is_input=False,
        ))

    @property
    def node_type(self) -> str:
        return "BoneMask"

    @property
    def display_name(self) -> str:
        return "Bone Mask"

    def include_bones(self, bones: List[str]) -> None:
        """Set included bones."""
        self._included_bones = list(bones)
        self._excluded_bones.clear()

    def exclude_bones(self, bones: List[str]) -> None:
        """Set excluded bones."""
        self._excluded_bones = list(bones)
        self._included_bones.clear()


# =============================================================================
# BLEND SPACE
# =============================================================================


@dataclass
class BlendSample:
    """A sample in a blend space.

    Attributes:
        animation_path: Path to animation asset
        position: Position in blend space (1D or 2D)
        rate: Playback rate modifier
    """

    animation_path: str
    position: Tuple[float, ...]
    rate: float = 1.0

    def __post_init__(self) -> None:
        if not self.animation_path:
            raise ValueError("Animation path cannot be empty")


class BlendSpace1D:
    """1D blend space for blending animations along one axis."""

    def __init__(self, name: str, axis_name: str = "Value") -> None:
        self._name = name
        self._axis_name = axis_name
        self._samples: List[BlendSample] = []
        self._min_value = 0.0
        self._max_value = 1.0

    @property
    def name(self) -> str:
        """Get blend space name."""
        return self._name

    @property
    def axis_name(self) -> str:
        """Get axis name."""
        return self._axis_name

    @property
    def samples(self) -> List[BlendSample]:
        """Get all samples."""
        return list(self._samples)

    @property
    def value_range(self) -> Tuple[float, float]:
        """Get value range."""
        return (self._min_value, self._max_value)

    def set_range(self, min_value: float, max_value: float) -> None:
        """Set value range."""
        self._min_value = min_value
        self._max_value = max_value

    def add_sample(self, animation_path: str, value: float) -> BlendSample:
        """Add a sample at value."""
        sample = BlendSample(
            animation_path=animation_path,
            position=(value,),
        )
        self._samples.append(sample)
        self._samples.sort(key=lambda s: s.position[0])
        return sample

    def remove_sample(self, index: int) -> bool:
        """Remove sample at index."""
        if 0 <= index < len(self._samples):
            self._samples.pop(index)
            return True
        return False

    def get_blend_weights(self, value: float) -> List[Tuple[int, float]]:
        """Get blend weights for a value.

        Returns:
            List of (sample_index, weight) tuples
        """
        if not self._samples:
            return []

        if len(self._samples) == 1:
            return [(0, 1.0)]

        # Find surrounding samples
        before_idx = -1
        after_idx = -1

        for i, sample in enumerate(self._samples):
            if sample.position[0] <= value:
                before_idx = i
            else:
                after_idx = i
                break

        if before_idx < 0:
            return [(0, 1.0)]
        if after_idx < 0:
            return [(len(self._samples) - 1, 1.0)]

        # Interpolate
        before = self._samples[before_idx]
        after = self._samples[after_idx]
        range_val = after.position[0] - before.position[0]

        if range_val < 1e-6:
            return [(before_idx, 1.0)]

        t = (value - before.position[0]) / range_val
        return [(before_idx, 1.0 - t), (after_idx, t)]


class BlendSpace2D:
    """2D blend space for blending animations in a 2D parameter space."""

    def __init__(
        self,
        name: str,
        axis_x_name: str = "X",
        axis_y_name: str = "Y",
    ) -> None:
        self._name = name
        self._axis_x_name = axis_x_name
        self._axis_y_name = axis_y_name
        self._samples: List[BlendSample] = []
        self._min_x = -1.0
        self._max_x = 1.0
        self._min_y = -1.0
        self._max_y = 1.0

    @property
    def name(self) -> str:
        """Get blend space name."""
        return self._name

    @property
    def samples(self) -> List[BlendSample]:
        """Get all samples."""
        return list(self._samples)

    def set_range_x(self, min_val: float, max_val: float) -> None:
        """Set X axis range."""
        self._min_x = min_val
        self._max_x = max_val

    def set_range_y(self, min_val: float, max_val: float) -> None:
        """Set Y axis range."""
        self._min_y = min_val
        self._max_y = max_val

    def add_sample(self, animation_path: str, x: float, y: float) -> BlendSample:
        """Add a sample at position."""
        sample = BlendSample(
            animation_path=animation_path,
            position=(x, y),
        )
        self._samples.append(sample)
        return sample

    def remove_sample(self, index: int) -> bool:
        """Remove sample at index."""
        if 0 <= index < len(self._samples):
            self._samples.pop(index)
            return True
        return False

    def get_blend_weights(self, x: float, y: float) -> List[Tuple[int, float]]:
        """Get blend weights for position using triangulation.

        This is a simplified implementation using inverse distance weighting.
        A full implementation would use Delaunay triangulation.
        """
        if not self._samples:
            return []

        if len(self._samples) == 1:
            return [(0, 1.0)]

        # Inverse distance weighting
        weights = []
        total_weight = 0.0

        for i, sample in enumerate(self._samples):
            dx = sample.position[0] - x
            dy = sample.position[1] - y
            dist_sq = dx * dx + dy * dy

            if dist_sq < 1e-6:
                return [(i, 1.0)]

            weight = 1.0 / dist_sq
            weights.append((i, weight))
            total_weight += weight

        # Normalize
        return [(i, w / total_weight) for i, w in weights]


class BlendSpaceNode(GraphNode):
    """Node for blend space sampling."""

    def __init__(
        self,
        node_id: str,
        is_2d: bool = False,
        position: Vec2 = None,
    ) -> None:
        super().__init__(node_id, position)
        self._is_2d = is_2d
        self._blend_space_1d: Optional[BlendSpace1D] = None
        self._blend_space_2d: Optional[BlendSpace2D] = None

        if is_2d:
            self.add_input(GraphSocket(
                name="X",
                socket_type=SocketType.FLOAT,
                is_input=True,
                default_value=0.0,
            ))
            self.add_input(GraphSocket(
                name="Y",
                socket_type=SocketType.FLOAT,
                is_input=True,
                default_value=0.0,
            ))
        else:
            self.add_input(GraphSocket(
                name="Value",
                socket_type=SocketType.FLOAT,
                is_input=True,
                default_value=0.0,
            ))

        self.add_output(GraphSocket(
            name="Pose",
            socket_type=SocketType.POSE,
            is_input=False,
        ))

    @property
    def node_type(self) -> str:
        return "BlendSpace2D" if self._is_2d else "BlendSpace1D"

    @property
    def display_name(self) -> str:
        return "Blend Space 2D" if self._is_2d else "Blend Space 1D"

    @property
    def is_2d(self) -> bool:
        """Check if 2D blend space."""
        return self._is_2d

    def set_blend_space_1d(self, space: BlendSpace1D) -> None:
        """Set 1D blend space."""
        if self._is_2d:
            raise ValueError("Cannot set 1D blend space on 2D node")
        self._blend_space_1d = space

    def set_blend_space_2d(self, space: BlendSpace2D) -> None:
        """Set 2D blend space."""
        if not self._is_2d:
            raise ValueError("Cannot set 2D blend space on 1D node")
        self._blend_space_2d = space


# =============================================================================
# NODE PALETTE
# =============================================================================


class NodePalette:
    """Palette of available node types."""

    def __init__(self) -> None:
        self._categories: Dict[str, List[Type[GraphNode]]] = {
            "States": [StateNode, EntryNode, ConduitNode],
            "Transitions": [TransitionNode],
            "Blending": [BlendNode, AdditiveNode, LayeredBlendNode, BoneMaskNode],
            "Blend Spaces": [BlendSpaceNode],
        }

    @property
    def categories(self) -> List[str]:
        """Get category names."""
        return list(self._categories.keys())

    def get_nodes_in_category(self, category: str) -> List[Type[GraphNode]]:
        """Get node types in a category."""
        return list(self._categories.get(category, []))

    def add_node_type(self, category: str, node_type: Type[GraphNode]) -> None:
        """Add a node type to a category."""
        if category not in self._categories:
            self._categories[category] = []
        if node_type not in self._categories[category]:
            self._categories[category].append(node_type)


# =============================================================================
# ANIM GRAPH PREVIEW
# =============================================================================


class AnimGraphPreview:
    """Preview settings for animation graph."""

    def __init__(self) -> None:
        self.show_grid = True
        self.show_connections = True
        self.show_node_ids = False
        self.show_execution_flow = True
        self.grid_size = 20
        self.snap_to_grid = True
        self.zoom_level = 1.0
        self.pan_offset = Vec2(0, 0)
        self.minimap_enabled = True


# =============================================================================
# ANIM GRAPH EDITOR
# =============================================================================


class AnimGraphEditor:
    """Editor for animation graphs.

    Provides functionality for creating and editing animation graphs
    with state machines, blend trees, and other nodes.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._connections: List[GraphConnection] = []
        self._parameters: Dict[str, Dict[str, Any]] = {}
        self._selection: Set[str] = set()
        self._preview = AnimGraphPreview()
        self._palette = NodePalette()
        self._next_node_id = 1
        self._on_change_callbacks: List[Callable[[], None]] = []

    @property
    def nodes(self) -> List[GraphNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    @property
    def connections(self) -> List[GraphConnection]:
        """Get all connections."""
        return list(self._connections)

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get all parameters."""
        return dict(self._parameters)

    @property
    def selection(self) -> Set[str]:
        """Get selected node IDs."""
        return set(self._selection)

    @property
    def preview(self) -> AnimGraphPreview:
        """Get preview settings."""
        return self._preview

    @property
    def palette(self) -> NodePalette:
        """Get node palette."""
        return self._palette

    def generate_node_id(self, prefix: str = "Node") -> str:
        """Generate a unique node ID."""
        node_id = f"{prefix}_{self._next_node_id}"
        self._next_node_id += 1
        return node_id

    def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph."""
        if node.node_id in self._nodes:
            return False
        self._nodes[node.node_id] = node
        self._notify_change()
        return True

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its connections."""
        if node_id not in self._nodes:
            return False

        # Remove connections
        self._connections = [
            c for c in self._connections
            if c.source_node != node_id and c.target_node != node_id
        ]

        del self._nodes[node_id]
        self._selection.discard(node_id)
        self._notify_change()
        return True

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def connect(
        self,
        source_node: str,
        source_socket: str,
        target_node: str,
        target_socket: str,
    ) -> bool:
        """Create a connection between nodes."""
        src = self._nodes.get(source_node)
        tgt = self._nodes.get(target_node)

        if src is None or tgt is None:
            return False

        src_socket = src.get_output(source_socket)
        tgt_socket = tgt.get_input(target_socket)

        if src_socket is None or tgt_socket is None:
            return False

        if not src_socket.can_connect_to(tgt_socket):
            return False

        # Check for existing connection to non-multi input
        if not tgt_socket.multi_connect:
            for conn in self._connections:
                if conn.target_node == target_node and conn.target_socket == target_socket:
                    return False

        connection = GraphConnection(
            source_node=source_node,
            source_socket=source_socket,
            target_node=target_node,
            target_socket=target_socket,
        )
        self._connections.append(connection)
        self._notify_change()
        return True

    def disconnect(
        self,
        source_node: str,
        source_socket: str,
        target_node: str,
        target_socket: str,
    ) -> bool:
        """Remove a connection."""
        for i, conn in enumerate(self._connections):
            if (
                conn.source_node == source_node and
                conn.source_socket == source_socket and
                conn.target_node == target_node and
                conn.target_socket == target_socket
            ):
                self._connections.pop(i)
                self._notify_change()
                return True
        return False

    def get_connections_from(self, node_id: str) -> List[GraphConnection]:
        """Get connections from a node."""
        return [c for c in self._connections if c.source_node == node_id]

    def get_connections_to(self, node_id: str) -> List[GraphConnection]:
        """Get connections to a node."""
        return [c for c in self._connections if c.target_node == node_id]

    def add_parameter(
        self,
        name: str,
        param_type: str,
        default_value: Any = None,
    ) -> bool:
        """Add a graph parameter."""
        if name in self._parameters:
            return False

        self._parameters[name] = {
            "type": param_type,
            "default": default_value,
            "value": default_value,
        }
        self._notify_change()
        return True

    def remove_parameter(self, name: str) -> bool:
        """Remove a parameter."""
        if name in self._parameters:
            del self._parameters[name]
            self._notify_change()
            return True
        return False

    def set_parameter_value(self, name: str, value: Any) -> bool:
        """Set parameter value."""
        if name in self._parameters:
            self._parameters[name]["value"] = value
            return True
        return False

    def select_node(self, node_id: str, add_to_selection: bool = False) -> None:
        """Select a node."""
        if not add_to_selection:
            self._selection.clear()
        if node_id in self._nodes:
            self._selection.add(node_id)

    def deselect_node(self, node_id: str) -> None:
        """Deselect a node."""
        self._selection.discard(node_id)

    def clear_selection(self) -> None:
        """Clear selection."""
        self._selection.clear()

    def delete_selected(self) -> int:
        """Delete selected nodes."""
        deleted = 0
        for node_id in list(self._selection):
            if self.remove_node(node_id):
                deleted += 1
        return deleted

    def move_selected(self, delta: Vec2) -> None:
        """Move selected nodes."""
        for node_id in self._selection:
            node = self._nodes.get(node_id)
            if node:
                node.position = Vec2(
                    node.position.x + delta.x,
                    node.position.y + delta.y,
                )

    def validate(self) -> List[str]:
        """Validate graph for errors."""
        errors = []

        # Check for entry node
        entry_nodes = [n for n in self._nodes.values() if isinstance(n, EntryNode)]
        if not entry_nodes:
            errors.append("Graph has no entry node")
        elif len(entry_nodes) > 1:
            errors.append("Graph has multiple entry nodes")

        # Check for unconnected nodes
        for node_id, node in self._nodes.items():
            if isinstance(node, EntryNode):
                continue

            # Check if node has any connections
            has_input = any(c.target_node == node_id for c in self._connections)
            has_output = any(c.source_node == node_id for c in self._connections)

            if not has_input and not has_output:
                errors.append(f"Node '{node_id}' is unconnected")

        return errors

    def add_on_change(self, callback: Callable[[], None]) -> None:
        """Register change callback."""
        self._on_change_callbacks.append(callback)

    def remove_on_change(self, callback: Callable[[], None]) -> None:
        """Remove change callback."""
        if callback in self._on_change_callbacks:
            self._on_change_callbacks.remove(callback)

    def _notify_change(self) -> None:
        """Notify change callbacks."""
        for callback in self._on_change_callbacks:
            callback()


__all__ = [
    "SocketType",
    "BlendType",
    "GraphSocket",
    "GraphConnection",
    "GraphNode",
    "StateNode",
    "TransitionNode",
    "ConduitNode",
    "EntryNode",
    "BlendNode",
    "AdditiveNode",
    "LayeredBlendNode",
    "BoneMaskNode",
    "BlendSample",
    "BlendSpace1D",
    "BlendSpace2D",
    "BlendSpaceNode",
    "NodePalette",
    "AnimGraphPreview",
    "AnimGraphEditor",
]
