"""
Trinity-aware DSL graph node types with introspection and dirty tracking.

Extends the base graph_types with Trinity Pattern integration:
- TrackedDescriptor fields for automatic dirty tracking via foundation.tracker
- EngineMeta-derived metaclass for type registration and introspection
- Foundation registry registration for cross-pillar type lookup
- Foundation mirror compatibility for reflection

Usage:
    from flowforge_backend.ast_parser.trinity_nodes import (
        TrinityGraphNode, TrinityGraphEdge, TrinityNodeGraph,
        to_trinity_graph,
    )

    # Create Trinity-aware nodes directly
    node = TrinityGraphNode(id="n1", type="component", name="Player")

    # Convert existing graph to Trinity-aware
    trinity_graph = to_trinity_graph(plain_graph)

    # Dirty tracking works automatically
    node.name = "Enemy"
    assert node.is_dirty("name")

    # Full Trinity introspection
    from foundation import mirror, registry
    info = mirror(node)
    assert registry.is_registered(TrinityGraphNode)
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from trinity.decorators.ops import Op, Step
from trinity.descriptors.tracking import TrackedDescriptor, is_dirty, get_dirty_fields, clear_dirty
from trinity.metaclasses.engine_meta import EngineMeta


# =============================================================================
# METACLASS
# =============================================================================


class _GraphNodeMeta(EngineMeta):
    """
    Metaclass for Trinity-aware DSL graph node types.

    Provides:
    - TrackedDescriptor installation on all annotated fields
    - Foundation registry registration for introspection
    - Step recording for decompose/expand introspection
    """

    # Base class names that should not be auto-registered
    _BASE_CLASS_NAMES = EngineMeta._BASE_CLASS_NAMES | frozenset({
        "_TrinityGraphBase",
        "_NodeDataBase",
    })

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> _GraphNodeMeta:
        """Create a new Trinity graph node type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base classes -- they have no fields to process
        if name in mcs._BASE_CLASS_NAMES:
            return cls

        # Install TrackedDescriptor on each annotated field
        _install_field_descriptors(cls)

        # Register with foundation.registry for introspection
        _register_with_foundation(cls)

        return cls


# =============================================================================
# FIELD PROCESSING HELPER
# =============================================================================


def _install_field_descriptors(cls: type) -> None:
    """Install TrackedDescriptor on each annotated field of *cls*."""
    cls._trinity_fields: dict[str, TrackedDescriptor] = {}
    annotations = getattr(cls, "__annotations__", {})

    for field_name, field_type in annotations.items():
        if field_name.startswith("_"):
            continue

        # Create TrackedDescriptor for this field
        desc = TrackedDescriptor(field_type=field_type)

        # Install on the class so __get__/__set__ are invoked on access
        setattr(cls, field_name, desc)
        desc.__set_name__(cls, field_name)
        cls._trinity_fields[field_name] = desc

        # Record steps for introspection
        type_name = getattr(field_type, "__name__", str(field_type))
        cls._metaclass_steps.append(
            Step(Op.DESCRIBE, {"field": field_name, "type": type_name})
        )
        cls._metaclass_steps.append(
            Step(Op.INTERCEPT, {"field": field_name, "descriptor": "tracked"})
        )


def _register_with_foundation(cls: type) -> None:
    """Register *cls* with foundation.registry for cross-pillar introspection."""
    try:
        from foundation import registry

        if not registry.is_registered(cls):
            registry.register(cls)
            cls._metaclass_steps.append(
                Step(Op.REGISTER, {"registry": "foundation"})
            )
    except ImportError:
        pass


# =============================================================================
# BASE CLASS
# =============================================================================


class _TrinityGraphBase(metaclass=_GraphNodeMeta):
    """
    Base class for all Trinity-aware graph types.

    Provides shared __repr__ and __eq__ implementations that inspect
    annotated fields.

    Note: This class intentionally does NOT use __slots__, because the
    descriptor-based storage mechanism (BaseDescriptor._get_stored /
    _set_stored) relies on ``obj.__dict__`` for field value storage.
    """

    def __repr__(self) -> str:
        fields: list[str] = []
        for name in getattr(type(self), "__annotations__", {}):
            if name.startswith("_"):
                continue
            try:
                value = getattr(self, name)
                fields.append(f"{name}={value!r}")
            except AttributeError:
                pass
        return f"{type(self).__name__}({', '.join(fields)})"

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        for name in getattr(type(self), "__annotations__", {}):
            if name.startswith("_"):
                continue
            try:
                if getattr(self, name) != getattr(other, name):
                    return False
            except AttributeError:
                return False
        return True

    def __hash__(self) -> int:
        return id(self)

    def is_dirty(self, field_name: Optional[str] = None) -> bool:
        """Check if a field (or any field) has been modified.

        Args:
            field_name: Optional field name. If None, checks if any field is dirty.

        Returns:
            True if the specified field (or any field) is dirty.
        """
        if field_name is not None:
            return is_dirty(self, field_name)
        return len(get_dirty_fields(self)) > 0

    def get_dirty_fields(self) -> set[str]:
        """Return the set of dirty field names."""
        return get_dirty_fields(self)

    def clear_dirty(self) -> None:
        """Reset all dirty flags."""
        clear_dirty(self)


# =============================================================================
# POSITION AND LOCATION
# =============================================================================


class TrinityNodePosition(_TrinityGraphBase):
    """Trinity-aware NodePosition with dirty tracking."""

    x: float
    y: float

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x = x
        self.y = y

    def to_dict(self) -> list[float]:
        """Convert to JSON-serializable format (tuple as list)."""
        return [self.x, self.y]

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple format."""
        return (self.x, self.y)

    @classmethod
    def from_dict(
        cls, data: Union[list[float], tuple[float, float], dict[str, float]]
    ) -> TrinityNodePosition:
        """Create from dict, list, or tuple.

        Args:
            data: Position data in one of these formats:
                  - [x, y] list
                  - (x, y) tuple
                  - {"x": x, "y": y} dict

        Returns:
            New TrinityNodePosition instance.
        """
        if isinstance(data, (list, tuple)):
            return cls(x=float(data[0]), y=float(data[1]))
        return cls(x=float(data.get("x", 0.0)), y=float(data.get("y", 0.0)))

    @classmethod
    def from_tuple(cls, pos: tuple[float, float]) -> TrinityNodePosition:
        """Create from a tuple (backward compatibility)."""
        return cls(x=pos[0], y=pos[1])


class TrinitySourceLocation(_TrinityGraphBase):
    """Trinity-aware SourceLocation with dirty tracking."""

    file: str
    line: int
    end_line: Optional[int]
    column: Optional[int]


    def __init__(
        self,
        file: str,
        line: int,
        end_line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> None:
        self.file = file
        self.line = line
        self.end_line = end_line
        self.column = column

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "file": self.file,
            "line": self.line,
        }
        if self.end_line is not None:
            result["end_line"] = self.end_line
        if self.column is not None:
            result["column"] = self.column
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinitySourceLocation:
        """Create from dictionary.

        Args:
            data: Dictionary with file and line keys.

        Returns:
            New TrinitySourceLocation instance.
        """
        return cls(
            file=data.get("file", ""),
            line=int(data.get("line", 0)),
            end_line=data.get("end_line"),
            column=data.get("column"),
        )


# =============================================================================
# NODE DATA TYPES
# =============================================================================


class _NodeDataBase(_TrinityGraphBase):
    """Base for node data types (ComponentData, SystemData, etc.)."""


class TrinityFieldData(_NodeDataBase):
    """Trinity-aware FieldData with dirty tracking."""

    name: str
    type_annotation: str
    default_value: Optional[str]
    line_number: Optional[int]
    is_optional: bool


    def __init__(
        self,
        name: str,
        type_annotation: str,
        default_value: Optional[str] = None,
        line_number: Optional[int] = None,
        is_optional: bool = False,
    ) -> None:
        self.name = name
        self.type_annotation = type_annotation
        self.default_value = default_value
        self.line_number = line_number
        self.is_optional = is_optional

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "name": self.name,
            "type_annotation": self.type_annotation,
        }
        if self.default_value is not None:
            result["default_value"] = self.default_value
        if self.line_number is not None:
            result["line_number"] = self.line_number
        if self.is_optional:
            result["is_optional"] = True
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinityFieldData:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type_annotation=data.get("type_annotation", "Any"),
            default_value=data.get("default_value"),
            line_number=data.get("line_number"),
            is_optional=data.get("is_optional", False),
        )


class TrinityParameterData(_NodeDataBase):
    """Trinity-aware ParameterData with dirty tracking."""

    name: str
    type_annotation: Optional[str]
    default_value: Optional[str]


    def __init__(
        self,
        name: str,
        type_annotation: Optional[str] = None,
        default_value: Optional[str] = None,
    ) -> None:
        self.name = name
        self.type_annotation = type_annotation
        self.default_value = default_value

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {"name": self.name}
        if self.type_annotation is not None:
            result["type_annotation"] = self.type_annotation
        if self.default_value is not None:
            result["default_value"] = self.default_value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinityParameterData:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type_annotation=data.get("type_annotation"),
            default_value=data.get("default_value"),
        )


class TrinityMethodData(_NodeDataBase):
    """Trinity-aware MethodData with dirty tracking."""

    name: str
    parameters: list[TrinityParameterData]
    return_type: Optional[str]
    docstring: Optional[str]
    line_number: Optional[int]
    query_components: list[str]
    decorators: list[str]


    def __init__(
        self,
        name: str,
        parameters: Optional[list[TrinityParameterData]] = None,
        return_type: Optional[str] = None,
        docstring: Optional[str] = None,
        line_number: Optional[int] = None,
        query_components: Optional[list[str]] = None,
        decorators: Optional[list[str]] = None,
    ) -> None:
        self.name = name
        self.parameters = parameters or []
        self.return_type = return_type
        self.docstring = docstring
        self.line_number = line_number
        self.query_components = query_components or []
        self.decorators = decorators or []

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "name": self.name,
            "parameters": [p.to_dict() for p in self.parameters],
        }
        if self.return_type is not None:
            result["return_type"] = self.return_type
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.line_number is not None:
            result["line_number"] = self.line_number
        if self.query_components:
            result["query_components"] = self.query_components
        if self.decorators:
            result["decorators"] = self.decorators
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinityMethodData:
        """Create from dictionary.

        Note: Accepts either 'query_components' or 'query_types' for
        backwards compatibility.
        """
        query_comps = data.get("query_types") or data.get("query_components", [])
        return cls(
            name=data["name"],
            parameters=[TrinityParameterData.from_dict(p) for p in data.get("parameters", [])],
            return_type=data.get("return_type"),
            docstring=data.get("docstring"),
            line_number=data.get("line_number"),
            query_components=query_comps,
            decorators=data.get("decorators", []),
        )


class TrinityComponentData(_NodeDataBase):
    """Trinity-aware ComponentData with dirty tracking."""

    fields: list[TrinityFieldData]
    bases: list[str]
    docstring: Optional[str]
    decorator_args: dict[str, Any]


    def __init__(
        self,
        fields: Optional[list[TrinityFieldData]] = None,
        bases: Optional[list[str]] = None,
        docstring: Optional[str] = None,
        decorator_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.fields = fields or []
        self.bases = bases or []
        self.docstring = docstring
        self.decorator_args = decorator_args or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "fields": [f.to_dict() for f in self.fields],
        }
        if self.bases:
            result["bases"] = self.bases
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.decorator_args:
            result["decorator_args"] = self.decorator_args
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinityComponentData:
        """Create from dictionary."""
        return cls(
            fields=[TrinityFieldData.from_dict(f) for f in data.get("fields", [])],
            bases=data.get("bases", []),
            docstring=data.get("docstring"),
            decorator_args=data.get("decorator_args", {}),
        )


class TrinitySystemData(_NodeDataBase):
    """Trinity-aware SystemData with dirty tracking."""

    methods: list[TrinityMethodData]
    queries: list[str]
    fields: list[TrinityFieldData]
    bases: list[str]
    docstring: Optional[str]
    decorator_args: dict[str, Any]


    def __init__(
        self,
        methods: Optional[list[TrinityMethodData]] = None,
        queries: Optional[list[str]] = None,
        fields: Optional[list[TrinityFieldData]] = None,
        bases: Optional[list[str]] = None,
        docstring: Optional[str] = None,
        decorator_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.methods = methods or []
        self.queries = queries or []
        self.fields = fields or []
        self.bases = bases or []
        self.docstring = docstring
        self.decorator_args = decorator_args or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "methods": [m.to_dict() for m in self.methods],
        }
        if self.queries:
            result["queries"] = self.queries
        if self.fields:
            result["fields"] = [f.to_dict() for f in self.fields]
        if self.bases:
            result["bases"] = self.bases
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.decorator_args:
            result["decorator_args"] = self.decorator_args
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinitySystemData:
        """Create from dictionary."""
        return cls(
            methods=[TrinityMethodData.from_dict(m) for m in data.get("methods", [])],
            queries=data.get("queries", []),
            fields=[TrinityFieldData.from_dict(f) for f in data.get("fields", [])],
            bases=data.get("bases", []),
            docstring=data.get("docstring"),
            decorator_args=data.get("decorator_args", {}),
        )


class TrinityResourceData(_NodeDataBase):
    """Trinity-aware ResourceData with dirty tracking."""

    fields: list[TrinityFieldData]
    bases: list[str]
    docstring: Optional[str]
    is_singleton: bool
    decorator_args: dict[str, Any]


    def __init__(
        self,
        fields: Optional[list[TrinityFieldData]] = None,
        bases: Optional[list[str]] = None,
        docstring: Optional[str] = None,
        is_singleton: bool = True,
        decorator_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.fields = fields or []
        self.bases = bases or []
        self.docstring = docstring
        self.is_singleton = is_singleton
        self.decorator_args = decorator_args or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "fields": [f.to_dict() for f in self.fields],
            "is_singleton": self.is_singleton,
        }
        if self.bases:
            result["bases"] = self.bases
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.decorator_args:
            result["decorator_args"] = self.decorator_args
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinityResourceData:
        """Create from dictionary."""
        return cls(
            fields=[TrinityFieldData.from_dict(f) for f in data.get("fields", [])],
            bases=data.get("bases", []),
            docstring=data.get("docstring"),
            is_singleton=data.get("is_singleton", True),
            decorator_args=data.get("decorator_args", {}),
        )


class TrinityEventData(_NodeDataBase):
    """Trinity-aware EventData with dirty tracking."""

    fields: list[TrinityFieldData]
    bases: list[str]
    docstring: Optional[str]
    decorator_args: dict[str, Any]


    def __init__(
        self,
        fields: Optional[list[TrinityFieldData]] = None,
        bases: Optional[list[str]] = None,
        docstring: Optional[str] = None,
        decorator_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.fields = fields or []
        self.bases = bases or []
        self.docstring = docstring
        self.decorator_args = decorator_args or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "fields": [f.to_dict() for f in self.fields],
        }
        if self.bases:
            result["bases"] = self.bases
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.decorator_args:
            result["decorator_args"] = self.decorator_args
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinityEventData:
        """Create from dictionary.

        Note: Accepts either 'fields' or 'payload_fields' for backwards
        compatibility. The frontend may send 'payload_fields' for events.
        """
        fields_data = data.get("payload_fields") or data.get("fields", [])
        return cls(
            fields=[TrinityFieldData.from_dict(f) for f in fields_data],
            bases=data.get("bases", []),
            docstring=data.get("docstring"),
            decorator_args=data.get("decorator_args", {}),
        )


# Type alias for all Trinity node data types
TrinityNodeData = Union[
    TrinityComponentData, TrinitySystemData, TrinityResourceData, TrinityEventData
]


def _parse_trinity_node_data(node_type: str, raw_data: dict[str, Any]) -> TrinityNodeData:
    """Parse node data based on node type, returning Trinity-aware types.

    Args:
        node_type: The type of node (component, system, resource, event).
        raw_data: Raw dictionary data to parse.

    Returns:
        Appropriate TrinityNodeData subclass instance.
    """
    if node_type == "component":
        return TrinityComponentData.from_dict(raw_data)
    elif node_type == "system":
        return TrinitySystemData.from_dict(raw_data)
    elif node_type == "resource":
        return TrinityResourceData.from_dict(raw_data)
    elif node_type == "event":
        return TrinityEventData.from_dict(raw_data)
    else:
        return TrinityComponentData.from_dict(raw_data)


# =============================================================================
# GRAPH NODES
# =============================================================================


class TrinityGraphNode(_TrinityGraphBase):
    """
    Trinity-aware GraphNode with dirty tracking and introspection.

    Matches the LiteGraph GraphNode interface:
        id, type, name, position, data, source

    All fields have TrackedDescriptor installed, enabling:
    - Automatic dirty flag on modification
    - Foundation tracker integration
    - Foundation mirror reflection
    """

    id: str
    type: Literal["component", "system", "resource", "event"]
    name: str
    position: TrinityNodePosition
    data: Union[dict[str, Any], TrinityNodeData]
    source: Optional[TrinitySourceLocation]


    def __init__(
        self,
        id: str,
        type: Literal["component", "system", "resource", "event"],
        name: str,
        position: Optional[TrinityNodePosition] = None,
        data: Optional[Union[dict[str, Any], TrinityNodeData]] = None,
        source: Optional[TrinitySourceLocation] = None,
    ) -> None:
        self.id = id
        self.type = type
        self.name = name
        self.position = position or TrinityNodePosition()
        self.data = data or {}
        self.source = source

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict matching frontend format."""
        if isinstance(self.data, dict):
            data_dict = self.data
        else:
            data_dict = self.data.to_dict()

        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "position": self.position.to_dict(),
            "data": data_dict,
        }
        if self.source is not None:
            result["source"] = self.source.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinityGraphNode:
        """Create from dictionary.

        Args:
            data: Dictionary with node data.

        Returns:
            New TrinityGraphNode instance.
        """
        node_type = data["type"]
        raw_data = data.get("data", {})
        parsed_data = _parse_trinity_node_data(node_type, raw_data)

        source = None
        if "source" in data and data["source"]:
            source = TrinitySourceLocation.from_dict(data["source"])

        return cls(
            id=data["id"],
            type=node_type,
            name=data["name"],
            position=TrinityNodePosition.from_dict(data.get("position", [0, 0])),
            data=parsed_data,
            source=source,
        )


# =============================================================================
# GRAPH EDGES
# =============================================================================


class TrinityGraphEdge(_TrinityGraphBase):
    """
    Trinity-aware GraphEdge with dirty tracking and introspection.

    Matches the LiteGraph GraphEdge interface:
        id, source, target, source_slot, target_slot, type, label, data
    """

    id: str
    source: str
    target: str
    source_slot: int
    target_slot: int
    type: Literal["reference", "inheritance", "query"]
    label: Optional[str]
    data: dict[str, Any]


    def __init__(
        self,
        id: str,
        source: str,
        target: str,
        source_slot: int = 0,
        target_slot: int = 0,
        type: Literal["reference", "inheritance", "query"] = "reference",
        label: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        self.id = id
        self.source = source
        self.target = target
        self.source_slot = source_slot
        self.target_slot = target_slot
        self.type = type
        self.label = label
        self.data = data or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict matching frontend format."""
        result: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
        }
        if self.source_slot != 0:
            result["source_slot"] = self.source_slot
        if self.target_slot != 0:
            result["target_slot"] = self.target_slot
        if self.label is not None:
            result["label"] = self.label
        if self.data:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinityGraphEdge:
        """Create from dictionary.

        Args:
            data: Dictionary with edge data.

        Returns:
            New TrinityGraphEdge instance.
        """
        return cls(
            id=data["id"],
            source=data["source"],
            target=data["target"],
            source_slot=data.get("source_slot", 0),
            target_slot=data.get("target_slot", 0),
            type=data.get("type", "reference"),
            label=data.get("label"),
        )


# =============================================================================
# NODE GRAPH
# =============================================================================


class TrinityNodeGraph(_TrinityGraphBase):
    """
    Trinity-aware NodeGraph with dirty tracking and introspection.

    The primary data structure exchanged between Python backend
    and TypeScript frontend. All mutation methods update dirty state.
    """

    nodes: list[TrinityGraphNode]
    edges: list[TrinityGraphEdge]
    metadata: dict[str, Any]


    def __init__(
        self,
        nodes: Optional[list[TrinityGraphNode]] = None,
        edges: Optional[list[TrinityGraphEdge]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.nodes = nodes or []
        self.edges = edges or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict matching frontend format."""
        result: dict[str, Any] = {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrinityNodeGraph:
        """Create from dictionary.

        Args:
            data: Dictionary with graph data.

        Returns:
            New TrinityNodeGraph instance.
        """
        return cls(
            nodes=[TrinityGraphNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[TrinityGraphEdge.from_dict(e) for e in data.get("edges", [])],
            metadata=data.get("metadata", {}),
        )

    # --- Query methods ---

    def get_node(self, node_id: str) -> Optional[TrinityGraphNode]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_node_by_name(self, name: str) -> Optional[TrinityGraphNode]:
        """Get a node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_nodes_by_type(
        self, node_type: str
    ) -> list[TrinityGraphNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes if n.type == node_type]

    def get_edges_from(self, node_id: str) -> list[TrinityGraphEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> list[TrinityGraphEdge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.target == node_id]

    # --- Mutation methods ---

    def add_node(self, node: TrinityGraphNode) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_edge(self, edge: TrinityGraphEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges from the graph."""
        initial_count = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self.edges = [
            e for e in self.edges
            if e.source != node_id and e.target != node_id
        ]
        return len(self.nodes) < initial_count

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the graph."""
        initial_count = len(self.edges)
        self.edges = [e for e in self.edges if e.id != edge_id]
        return len(self.edges) < initial_count


# =============================================================================
# CONVERSION
# =============================================================================


def _convert_data_to_trinity(data: Any) -> Any:
    """Recursively convert graph type instances to their Trinity equivalents."""
    # Import here to avoid circular imports
    from .graph_types import (
        ComponentData, SystemData, ResourceData, EventData,
        FieldData, ParameterData, MethodData,
        NodePosition, SourceLocation,
        GraphNode, GraphEdge, NodeGraph,
    )

    if isinstance(data, NodePosition):
        return TrinityNodePosition(x=data.x, y=data.y)
    if isinstance(data, SourceLocation):
        return TrinitySourceLocation(
            file=data.file, line=data.line,
            end_line=data.end_line, column=data.column,
        )
    if isinstance(data, FieldData):
        return TrinityFieldData(
            name=data.name, type_annotation=data.type_annotation,
            default_value=data.default_value,
            line_number=data.line_number, is_optional=data.is_optional,
        )
    if isinstance(data, ParameterData):
        return TrinityParameterData(
            name=data.name, type_annotation=data.type_annotation,
            default_value=data.default_value,
        )
    if isinstance(data, MethodData):
        return TrinityMethodData(
            name=data.name,
            parameters=[_convert_data_to_trinity(p) for p in data.parameters],
            return_type=data.return_type, docstring=data.docstring,
            line_number=data.line_number,
            query_components=list(data.query_components),
            decorators=list(data.decorators),
        )
    if isinstance(data, ComponentData):
        return TrinityComponentData(
            fields=[_convert_data_to_trinity(f) for f in data.fields],
            bases=list(data.bases), docstring=data.docstring,
            decorator_args=dict(data.decorator_args),
        )
    if isinstance(data, SystemData):
        return TrinitySystemData(
            methods=[_convert_data_to_trinity(m) for m in data.methods],
            queries=list(data.queries),
            fields=[_convert_data_to_trinity(f) for f in data.fields],
            bases=list(data.bases), docstring=data.docstring,
            decorator_args=dict(data.decorator_args),
        )
    if isinstance(data, ResourceData):
        return TrinityResourceData(
            fields=[_convert_data_to_trinity(f) for f in data.fields],
            bases=list(data.bases), docstring=data.docstring,
            is_singleton=data.is_singleton,
            decorator_args=dict(data.decorator_args),
        )
    if isinstance(data, EventData):
        return TrinityEventData(
            fields=[_convert_data_to_trinity(f) for f in data.fields],
            bases=list(data.bases), docstring=data.docstring,
            decorator_args=dict(data.decorator_args),
        )
    if isinstance(data, GraphNode):
        return TrinityGraphNode(
            id=data.id, type=data.type, name=data.name,
            position=_convert_data_to_trinity(data.position),
            data=_convert_data_to_trinity(data.data) if not isinstance(data.data, dict) else data.data,
            source=_convert_data_to_trinity(data.source) if data.source else None,
        )
    if isinstance(data, GraphEdge):
        return TrinityGraphEdge(
            id=data.id, source=data.source, target=data.target,
            source_slot=data.source_slot, target_slot=data.target_slot,
            type=data.type, label=data.label, data=getattr(data, 'data', {}),
        )
    return data


def to_trinity_graph(graph: Any) -> TrinityNodeGraph:
    """Convert a plain NodeGraph to Trinity-aware types.

    Recursively converts all nodes, edges, and data types to their
    Trinity-aware equivalents, enabling dirty tracking and introspection.

    Args:
        graph: A NodeGraph (or TrinityNodeGraph) instance.

    Returns:
        A TrinityNodeGraph with all content converted to Trinity types.

    Example:
        >>> from flowforge_backend.ast_parser.graph_builder import build_graph_from_parse_result
        >>> from flowforge_backend.ast_parser.trinity_nodes import to_trinity_graph
        >>> result = parse_source(...)
        >>> plain = build_graph_from_parse_result(result)
        >>> trinity = to_trinity_graph(plain)
        >>> trinity.get_node("n1").is_dirty("name")
        False
    """
    if isinstance(graph, TrinityNodeGraph):
        return graph

    from .graph_types import NodeGraph
    if not isinstance(graph, NodeGraph):
        raise TypeError(
            f"Expected NodeGraph or TrinityNodeGraph, got {type(graph).__name__}"
        )

    trinity_nodes = [_convert_data_to_trinity(n) for n in graph.nodes]
    trinity_edges = [_convert_data_to_trinity(e) for e in graph.edges]

    return TrinityNodeGraph(
        nodes=trinity_nodes,
        edges=trinity_edges,
        metadata=dict(graph.metadata) if hasattr(graph, 'metadata') else {},
    )


# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================


def create_trinity_component_node(
    id: str,
    name: str,
    fields: list[TrinityFieldData],
    source: TrinitySourceLocation,
    position: Optional[TrinityNodePosition] = None,
    bases: Optional[list[str]] = None,
    docstring: Optional[str] = None,
) -> TrinityGraphNode:
    """Create a component node with Trinity-aware types."""
    return TrinityGraphNode(
        id=id,
        type="component",
        name=name,
        position=position or TrinityNodePosition(),
        data=TrinityComponentData(
            fields=fields,
            bases=bases or [],
            docstring=docstring,
        ),
        source=source,
    )


def create_trinity_system_node(
    id: str,
    name: str,
    methods: list[TrinityMethodData],
    source: TrinitySourceLocation,
    position: Optional[TrinityNodePosition] = None,
    queries: Optional[list[str]] = None,
    bases: Optional[list[str]] = None,
    docstring: Optional[str] = None,
) -> TrinityGraphNode:
    """Create a system node with Trinity-aware types."""
    return TrinityGraphNode(
        id=id,
        type="system",
        name=name,
        position=position or TrinityNodePosition(),
        data=TrinitySystemData(
            methods=methods,
            queries=queries or [],
            bases=bases or [],
            docstring=docstring,
        ),
        source=source,
    )


def create_trinity_resource_node(
    id: str,
    name: str,
    fields: list[TrinityFieldData],
    source: TrinitySourceLocation,
    position: Optional[TrinityNodePosition] = None,
    is_singleton: bool = True,
    bases: Optional[list[str]] = None,
    docstring: Optional[str] = None,
) -> TrinityGraphNode:
    """Create a resource node with Trinity-aware types."""
    return TrinityGraphNode(
        id=id,
        type="resource",
        name=name,
        position=position or TrinityNodePosition(),
        data=TrinityResourceData(
            fields=fields,
            is_singleton=is_singleton,
            bases=bases or [],
            docstring=docstring,
        ),
        source=source,
    )


def create_trinity_event_node(
    id: str,
    name: str,
    fields: list[TrinityFieldData],
    source: TrinitySourceLocation,
    position: Optional[TrinityNodePosition] = None,
    bases: Optional[list[str]] = None,
    docstring: Optional[str] = None,
) -> TrinityGraphNode:
    """Create an event node with Trinity-aware types."""
    return TrinityGraphNode(
        id=id,
        type="event",
        name=name,
        position=position or TrinityNodePosition(),
        data=TrinityEventData(
            fields=fields,
            bases=bases or [],
            docstring=docstring,
        ),
        source=source,
    )


def create_trinity_edge(
    id: str,
    source: str,
    target: str,
    edge_type: Literal["reference", "inheritance", "query"] = "reference",
    source_slot: int = 0,
    target_slot: int = 0,
    label: Optional[str] = None,
) -> TrinityGraphEdge:
    """Create a Trinity-aware edge between two nodes."""
    return TrinityGraphEdge(
        id=id,
        source=source,
        target=target,
        source_slot=source_slot,
        target_slot=target_slot,
        type=edge_type,
        label=label,
    )


# =============================================================================
# EXPLICIT REGISTRATION
# =============================================================================


def register_all_trinity_graph_types() -> None:
    """Explicitly register all Trinity graph types with foundation.registry.

    This is called automatically by the metaclass for each type as it is
    created. This function is provided for cases where you need to ensure
    all types are registered after imports (e.g., in test setup).
    """
    types = [
        TrinityNodePosition,
        TrinitySourceLocation,
        TrinityFieldData,
        TrinityParameterData,
        TrinityMethodData,
        TrinityComponentData,
        TrinitySystemData,
        TrinityResourceData,
        TrinityEventData,
        TrinityGraphNode,
        TrinityGraphEdge,
        TrinityNodeGraph,
    ]
    for cls in types:
        _register_with_foundation(cls)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "TrinityNodePosition",
    "TrinitySourceLocation",
    "TrinityFieldData",
    "TrinityParameterData",
    "TrinityMethodData",
    "TrinityComponentData",
    "TrinitySystemData",
    "TrinityResourceData",
    "TrinityEventData",
    "TrinityNodeData",
    "TrinityGraphNode",
    "TrinityGraphEdge",
    "TrinityNodeGraph",
    # Conversion
    "to_trinity_graph",
    # Convenience constructors
    "create_trinity_component_node",
    "create_trinity_system_node",
    "create_trinity_resource_node",
    "create_trinity_event_node",
    "create_trinity_edge",
    # Registration
    "register_all_trinity_graph_types",
    # Metaclass (for `isinstance` checks)
    "_GraphNodeMeta",
]
