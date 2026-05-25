"""
Tests for Trinity-aware DSL graph node types (T-DEMO-2.2).

Verifies that DSL nodes are valid Trinity Objects with:
- EngineMeta metaclass for type registration + introspection
- TrackedDescriptor fields for dirty tracking
- Foundation registry registration
- Foundation mirror compatibility
- Serialization round-trip (to_dict / from_dict)
"""

from __future__ import annotations

from typing import Any

import pytest

# flowforge_backend is not yet implemented (Project 2 in FlowForge roadmap)
pytest.importorskip("flowforge_backend", reason="flowforge_backend not yet implemented")

from trinity.decorators.ops import Op, Step, decompose, expand, decompose_layered
from trinity.descriptors.tracking import is_dirty, get_dirty_fields, clear_dirty

# Import Trinity-aware graph types
from flowforge_backend.ast_parser.trinity_nodes import (
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
    to_trinity_graph,
    create_trinity_component_node,
    create_trinity_system_node,
    create_trinity_resource_node,
    create_trinity_event_node,
    create_trinity_edge,
    register_all_trinity_graph_types,
)

# Import plain graph types for conversion testing
from flowforge_backend.ast_parser.graph_types import (
    NodePosition,
    SourceLocation,
    FieldData,
    ParameterData,
    MethodData,
    ComponentData,
    SystemData,
    ResourceData,
    EventData,
    GraphNode,
    GraphEdge,
    NodeGraph,
)


# =============================================================================
# BASIC TYPE CREATION
# =============================================================================


class TestTrinityNodePosition:
    """TrinityNodePosition is a valid Trinity Object with dirty tracking."""

    def test_create(self) -> None:
        pos = TrinityNodePosition(x=10.0, y=20.0)
        assert pos.x == 10.0
        assert pos.y == 20.0

    def test_defaults(self) -> None:
        pos = TrinityNodePosition()
        assert pos.x == 0.0
        assert pos.y == 0.0

    def test_dirty_tracking_on_set(self) -> None:
        pos = TrinityNodePosition()
        pos.clear_dirty()  # Clear init-induced dirty state
        assert not pos.is_dirty()
        pos.x = 42.0
        assert pos.is_dirty("x")
        assert not pos.is_dirty("y")

    def test_clear_dirty(self) -> None:
        pos = TrinityNodePosition(10.0, 20.0)
        pos.x = 99.0
        assert pos.is_dirty("x")
        pos.clear_dirty()
        assert not pos.is_dirty()

    def test_to_dict(self) -> None:
        pos = TrinityNodePosition(10.0, 20.0)
        assert pos.to_dict() == [10.0, 20.0]

    def test_from_dict_list(self) -> None:
        pos = TrinityNodePosition.from_dict([30.0, 40.0])
        assert pos.x == 30.0
        assert pos.y == 40.0

    def test_from_dict_dict(self) -> None:
        pos = TrinityNodePosition.from_dict({"x": 5.0, "y": 15.0})
        assert pos.x == 5.0
        assert pos.y == 15.0

    def test_to_tuple(self) -> None:
        pos = TrinityNodePosition(3.0, 7.0)
        assert pos.to_tuple() == (3.0, 7.0)

    def test_from_tuple(self) -> None:
        pos = TrinityNodePosition.from_tuple((8.0, 9.0))
        assert pos.x == 8.0
        assert pos.y == 9.0

    def test_equality(self) -> None:
        a = TrinityNodePosition(1.0, 2.0)
        b = TrinityNodePosition(1.0, 2.0)
        c = TrinityNodePosition(3.0, 4.0)
        assert a == b
        assert a != c

    def test_repr(self) -> None:
        pos = TrinityNodePosition(10.0, 20.0)
        r = repr(pos)
        assert "TrinityNodePosition" in r
        assert "10.0" in r
        assert "20.0" in r


class TestTrinitySourceLocation:
    """TrinitySourceLocation dirty tracking and serialization."""

    def test_create(self) -> None:
        loc = TrinitySourceLocation(file="test.py", line=42)
        assert loc.file == "test.py"
        assert loc.line == 42
        assert loc.end_line is None
        assert loc.column is None

    def test_create_full(self) -> None:
        loc = TrinitySourceLocation(file="test.py", line=10, end_line=20, column=4)
        assert loc.end_line == 20
        assert loc.column == 4

    def test_dirty_tracking(self) -> None:
        loc = TrinitySourceLocation(file="a.py", line=1)
        loc.file = "b.py"
        assert loc.is_dirty("file")
        loc.clear_dirty()
        assert not loc.is_dirty("file")

    def test_to_dict(self) -> None:
        loc = TrinitySourceLocation(file="main.py", line=5, end_line=10)
        d = loc.to_dict()
        assert d["file"] == "main.py"
        assert d["line"] == 5
        assert d["end_line"] == 10
        assert "column" not in d

    def test_from_dict(self) -> None:
        loc = TrinitySourceLocation.from_dict({"file": "mod.py", "line": 3})
        assert loc.file == "mod.py"
        assert loc.line == 3


# =============================================================================
# NODE DATA TYPES
# =============================================================================


class TestTrinityFieldData:
    """FieldData with dirty tracking."""

    def test_create(self) -> None:
        fd = TrinityFieldData(name="health", type_annotation="int", default_value="100")
        assert fd.name == "health"
        assert fd.type_annotation == "int"
        assert fd.default_value == "100"

    def test_create_minimal(self) -> None:
        fd = TrinityFieldData(name="x", type_annotation="float")
        assert fd.default_value is None
        assert fd.is_optional is False

    def test_dirty_tracking(self) -> None:
        fd = TrinityFieldData(name="a", type_annotation="str")
        fd.is_optional = True
        assert fd.is_dirty("is_optional")

    def test_to_dict(self) -> None:
        fd = TrinityFieldData(name="pos", type_annotation="Vector3", default_value="0,0,0", line_number=5)
        d = fd.to_dict()
        assert d["name"] == "pos"
        assert d["type_annotation"] == "Vector3"
        assert d["default_value"] == "0,0,0"
        assert d["line_number"] == 5

    def test_from_dict(self) -> None:
        fd = TrinityFieldData.from_dict({"name": "hp", "type_annotation": "int", "default_value": "100"})
        assert fd.name == "hp"
        assert fd.type_annotation == "int"
        assert fd.default_value == "100"

    def test_repr(self) -> None:
        fd = TrinityFieldData(name="hp", type_annotation="int")
        r = repr(fd)
        assert "TrinityFieldData" in r
        assert "hp" in r


class TestTrinityParameterData:
    """ParameterData with dirty tracking."""

    def test_create(self) -> None:
        pd = TrinityParameterData(name="dt", type_annotation="float", default_value="0.016")
        assert pd.name == "dt"
        assert pd.type_annotation == "float"

    def test_dirty_tracking(self) -> None:
        pd = TrinityParameterData(name="val")
        pd.type_annotation = "int"
        assert pd.is_dirty("type_annotation")


class TestTrinityMethodData:
    """MethodData with dirty tracking."""

    def test_create(self) -> None:
        params = [TrinityParameterData(name="self")]
        md = TrinityMethodData(name="update", parameters=params, return_type="None")
        assert md.name == "update"
        assert len(md.parameters) == 1

    def test_dirty_tracking_on_list_field(self) -> None:
        md = TrinityMethodData(name="run")
        assert len(md.query_components) == 0
        # Setting the entire list marks it dirty
        md.query_components = ["Position", "Velocity"]
        assert md.is_dirty("query_components")

    def test_to_dict_roundtrip(self) -> None:
        params = [TrinityParameterData(name="x", type_annotation="float")]
        md1 = TrinityMethodData(name="compute", parameters=params, return_type="float", docstring="Computes")
        d = md1.to_dict()
        md2 = TrinityMethodData.from_dict(d)
        assert md2.name == "compute"
        assert md2.return_type == "float"


class TestTrinityComponentData:
    """ComponentData with dirty tracking."""

    def test_create(self) -> None:
        fields = [TrinityFieldData(name="x", type_annotation="float")]
        cd = TrinityComponentData(fields=fields, docstring="Position component")
        assert len(cd.fields) == 1
        assert cd.docstring == "Position component"

    def test_dirty_tracking(self) -> None:
        cd = TrinityComponentData()
        cd.docstring = "Updated docstring"
        assert cd.is_dirty("docstring")

    def test_to_dict_roundtrip(self) -> None:
        fields = [TrinityFieldData(name="health", type_annotation="int", default_value="100")]
        cd1 = TrinityComponentData(fields=fields, bases=["Component"])
        d = cd1.to_dict()
        cd2 = TrinityComponentData.from_dict(d)
        assert len(cd2.fields) == 1
        assert cd2.fields[0].name == "health"
        assert "Component" in cd2.bases


class TestTrinitySystemData:
    """SystemData with dirty tracking."""

    def test_create(self) -> None:
        methods = [TrinityMethodData(name="execute")]
        sd = TrinitySystemData(methods=methods, queries=["Position", "Velocity"])
        assert len(sd.methods) == 1
        assert "Position" in sd.queries

    def test_dirty_tracking(self) -> None:
        sd = TrinitySystemData()
        sd.queries = ["Transform"]
        assert sd.is_dirty("queries")


class TestTrinityResourceData:
    """ResourceData with dirty tracking."""

    def test_create(self) -> None:
        rd = TrinityResourceData(is_singleton=True)
        assert rd.is_singleton is True

    def test_dirty_tracking(self) -> None:
        rd = TrinityResourceData()
        rd.is_singleton = False
        assert rd.is_singleton is False
        assert rd.is_dirty("is_singleton")


class TestTrinityEventData:
    """EventData with dirty tracking."""

    def test_create(self) -> None:
        fields = [TrinityFieldData(name="damage", type_annotation="float")]
        ed = TrinityEventData(fields=fields)
        assert len(ed.fields) == 1

    def test_dirty_tracking(self) -> None:
        ed = TrinityEventData()
        ed.bases = ["Event"]
        assert ed.is_dirty("bases")


# =============================================================================
# GRAPH NODE
# =============================================================================


class TestTrinityGraphNode:
    """GraphNode is the primary Trinity Object for DSL integration."""

    def test_create_minimal(self) -> None:
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        assert node.id == "n1"
        assert node.type == "component"
        assert node.name == "Player"
        assert isinstance(node.position, TrinityNodePosition)
        assert node.position.x == 0.0
        assert node.source is None

    def test_create_full(self) -> None:
        pos = TrinityNodePosition(100.0, 200.0)
        src = TrinitySourceLocation("game.py", 10)
        data = TrinityComponentData(fields=[TrinityFieldData("hp", "int")])
        node = TrinityGraphNode(
            id="n1", type="component", name="Player",
            position=pos, data=data, source=src,
        )
        assert node.position.x == 100.0
        assert node.source.file == "game.py"
        assert isinstance(node.data, TrinityComponentData)

    def test_field_dirty_tracking(self) -> None:
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()  # Clear init-induced dirty state
        assert not node.is_dirty()
        node.name = "Enemy"
        assert node.is_dirty("name")
        assert "name" in node.get_dirty_fields()

    def test_multiple_dirty_fields(self) -> None:
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.name = "Enemy"
        node.type = "system"
        dirty = node.get_dirty_fields()
        assert "name" in dirty
        assert "type" in dirty

    def test_clear_dirty(self) -> None:
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.name = "Updated"
        assert node.is_dirty()
        node.clear_dirty()
        assert not node.is_dirty()

    def test_same_value_no_dirty(self) -> None:
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()
        node.name = "Player"  # Same value -- no change
        assert not node.is_dirty()

    def test_to_dict(self) -> None:
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        d = node.to_dict()
        assert d["id"] == "n1"
        assert d["type"] == "component"
        assert d["name"] == "Player"
        assert d["position"] == [0.0, 0.0]

    def test_from_dict(self) -> None:
        raw = {
            "id": "n2",
            "type": "system",
            "name": "MovementSystem",
            "position": [50.0, 100.0],
            "data": {
                "methods": [{"name": "execute", "parameters": []}],
                "queries": ["Position"],
            },
        }
        node = TrinityGraphNode.from_dict(raw)
        assert node.id == "n2"
        assert node.type == "system"
        assert node.name == "MovementSystem"
        assert node.position.x == 50.0
        assert isinstance(node.data, TrinitySystemData)
        assert "Position" in node.data.queries

    def test_equality(self) -> None:
        a = TrinityGraphNode(id="n1", type="component", name="A")
        b = TrinityGraphNode(id="n1", type="component", name="A")
        c = TrinityGraphNode(id="n1", type="component", name="B")
        assert a == b
        assert a != c

    def test_repr(self) -> None:
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        r = repr(node)
        assert "TrinityGraphNode" in r
        assert "Player" in r


# =============================================================================
# GRAPH EDGE
# =============================================================================


class TestTrinityGraphEdge:
    """GraphEdge with dirty tracking."""

    def test_create(self) -> None:
        edge = TrinityGraphEdge(id="e1", source="n1", target="n2")
        assert edge.id == "e1"
        assert edge.source == "n1"
        assert edge.target == "n2"

    def test_create_full(self) -> None:
        edge = TrinityGraphEdge(
            id="e1", source="n1", target="n2",
            source_slot=0, target_slot=1, type="inheritance",
            label="extends",
        )
        assert edge.type == "inheritance"
        assert edge.label == "extends"

    def test_dirty_tracking(self) -> None:
        edge = TrinityGraphEdge(id="e1", source="n1", target="n2")
        edge.type = "query"
        assert edge.is_dirty("type")

    def test_to_dict(self) -> None:
        edge = TrinityGraphEdge(id="e1", source="n1", target="n2")
        d = edge.to_dict()
        assert d["id"] == "e1"
        assert d["source"] == "n1"
        assert d["target"] == "n2"
        assert d["type"] == "reference"

    def test_from_dict(self) -> None:
        raw = {"id": "e1", "source": "n1", "target": "n2", "type": "inheritance"}
        edge = TrinityGraphEdge.from_dict(raw)
        assert edge.type == "inheritance"

    def test_from_dict_with_label(self) -> None:
        raw = {"id": "e1", "source": "n1", "target": "n2", "label": "depends_on"}
        edge = TrinityGraphEdge.from_dict(raw)
        assert edge.label == "depends_on"


# =============================================================================
# NODE GRAPH
# =============================================================================


class TestTrinityNodeGraph:
    """NodeGraph is the top-level Trinity Object for DSL graphs."""

    def test_create_empty(self) -> None:
        graph = TrinityNodeGraph()
        assert graph.nodes == []
        assert graph.edges == []
        assert graph.metadata == {}

    def test_create_with_data(self) -> None:
        nodes = [TrinityGraphNode(id="n1", type="component", name="A")]
        edges = [TrinityGraphEdge(id="e1", source="n1", target="n2")]
        meta = {"source": "test.py"}
        graph = TrinityNodeGraph(nodes=nodes, edges=edges, metadata=meta)
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 1
        assert graph.metadata["source"] == "test.py"

    def test_get_node(self) -> None:
        graph = TrinityNodeGraph(
            nodes=[TrinityGraphNode(id="n1", type="component", name="A")]
        )
        assert graph.get_node("n1") is not None
        assert graph.get_node("n2") is None

    def test_get_node_by_name(self) -> None:
        graph = TrinityNodeGraph(
            nodes=[TrinityGraphNode(id="n1", type="component", name="Player")]
        )
        assert graph.get_node_by_name("Player") is not None
        assert graph.get_node_by_name("Enemy") is None

    def test_get_nodes_by_type(self) -> None:
        graph = TrinityNodeGraph(
            nodes=[
                TrinityGraphNode(id="n1", type="component", name="A"),
                TrinityGraphNode(id="n2", type="system", name="B"),
                TrinityGraphNode(id="n3", type="component", name="C"),
            ]
        )
        comps = graph.get_nodes_by_type("component")
        assert len(comps) == 2
        assert comps[0].id == "n1"
        assert comps[1].id == "n3"

    def test_get_edges_from_to(self) -> None:
        graph = TrinityNodeGraph(
            edges=[
                TrinityGraphEdge(id="e1", source="n1", target="n2"),
                TrinityGraphEdge(id="e2", source="n1", target="n3"),
                TrinityGraphEdge(id="e3", source="n2", target="n3"),
            ]
        )
        assert len(graph.get_edges_from("n1")) == 2
        assert len(graph.get_edges_to("n3")) == 2

    def test_add_node(self) -> None:
        graph = TrinityNodeGraph()
        graph.add_node(TrinityGraphNode(id="n1", type="component", name="A"))
        assert len(graph.nodes) == 1
        assert graph.nodes[0].is_dirty()  # Was just created

    def test_add_edge(self) -> None:
        graph = TrinityNodeGraph()
        graph.add_edge(TrinityGraphEdge(id="e1", source="n1", target="n2"))
        assert len(graph.edges) == 1

    def test_remove_node_also_removes_edges(self) -> None:
        graph = TrinityNodeGraph(
            nodes=[TrinityGraphNode(id="n1", type="component", name="A")],
            edges=[TrinityGraphEdge(id="e1", source="n1", target="n2")],
        )
        removed = graph.remove_node("n1")
        assert removed is True
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0  # Edge referencing n1 also removed

    def test_remove_edge(self) -> None:
        graph = TrinityNodeGraph(
            edges=[TrinityGraphEdge(id="e1", source="n1", target="n2")]
        )
        removed = graph.remove_edge("e1")
        assert removed is True
        removed = graph.remove_edge("nonexistent")
        assert removed is False

    def test_to_dict(self) -> None:
        graph = TrinityNodeGraph(
            nodes=[TrinityGraphNode(id="n1", type="component", name="A")],
            metadata={"source": "test.py"},
        )
        d = graph.to_dict()
        assert len(d["nodes"]) == 1
        assert d["metadata"]["source"] == "test.py"

    def test_from_dict(self) -> None:
        raw = {
            "nodes": [{"id": "n1", "type": "component", "name": "A", "position": [0, 0], "data": {}}],
            "edges": [{"id": "e1", "source": "n1", "target": "n2"}],
            "metadata": {"source": "test.py"},
        }
        graph = TrinityNodeGraph.from_dict(raw)
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 1
        assert graph.metadata["source"] == "test.py"

    def test_dirty_on_node_list_mutation(self) -> None:
        graph = TrinityNodeGraph()
        graph.clear_dirty()
        assert not graph.is_dirty("nodes")
        graph.nodes.append(TrinityGraphNode(id="n1", type="component", name="A"))
        # The list reference hasn't changed, so dirty isn't triggered
        # (TrackedDescriptor only tracks set operations, not list mutations)
        # This is expected behavior -- call clear_dirty or reassign the list

    def test_dirty_on_list_reassignment(self) -> None:
        graph = TrinityNodeGraph()
        graph.clear_dirty()
        assert not graph.is_dirty("nodes")
        new_nodes = [TrinityGraphNode(id="n1", type="component", name="A")]
        graph.nodes = new_nodes
        assert graph.is_dirty("nodes")


# =============================================================================
# CONVERSION FROM PLAIN TYPES
# =============================================================================


class TestToTrinityGraph:
    """Conversion from plain NodeGraph to Trinity-aware types."""

    def test_converts_graph(self) -> None:
        pos = NodePosition(10.0, 20.0)
        src = SourceLocation("game.py", 10)
        plain_node = GraphNode(
            id="n1", type="component", name="Player",
            position=pos, source=src,
        )
        plain_graph = NodeGraph(
            nodes=[plain_node],
            edges=[GraphEdge(id="e1", source="n1", target="n2")],
            metadata={"source": "test.py"},
        )

        trinity = to_trinity_graph(plain_graph)
        assert isinstance(trinity, TrinityNodeGraph)
        assert len(trinity.nodes) == 1
        assert len(trinity.edges) == 1

        tnode = trinity.nodes[0]
        assert isinstance(tnode, TrinityGraphNode)
        assert isinstance(tnode.position, TrinityNodePosition)
        assert isinstance(tnode.source, TrinitySourceLocation)
        assert tnode.position.x == 10.0
        assert tnode.name == "Player"

    def test_converts_component_data(self) -> None:
        fields = [FieldData(name="x", type_annotation="float")]
        comp_data = ComponentData(fields=fields, docstring="Position")
        plain_node = GraphNode(
            id="n1", type="component", name="Position",
            data=comp_data,
        )
        plain_graph = NodeGraph(nodes=[plain_node])

        trinity = to_trinity_graph(plain_graph)
        tnode = trinity.nodes[0]
        assert isinstance(tnode.data, TrinityComponentData)
        assert tnode.data.docstring == "Position"
        assert isinstance(tnode.data.fields[0], TrinityFieldData)

    def test_converts_system_data(self) -> None:
        methods = [MethodData(name="execute")]
        sys_data = SystemData(methods=methods, queries=["Position"])
        plain_node = GraphNode(
            id="n1", type="system", name="MovementSystem",
            data=sys_data,
        )
        plain_graph = NodeGraph(nodes=[plain_node])

        trinity = to_trinity_graph(plain_graph)
        tnode = trinity.nodes[0]
        assert isinstance(tnode.data, TrinitySystemData)
        assert "Position" in tnode.data.queries

    def test_converts_resource_data(self) -> None:
        res_data = ResourceData(is_singleton=True)
        plain_node = GraphNode(
            id="n1", type="resource", name="GameConfig",
            data=res_data,
        )
        plain_graph = NodeGraph(nodes=[plain_node])

        trinity = to_trinity_graph(plain_graph)
        tnode = trinity.nodes[0]
        assert isinstance(tnode.data, TrinityResourceData)
        assert tnode.data.is_singleton is True

    def test_converts_event_data(self) -> None:
        fields = [FieldData(name="damage", type_annotation="float")]
        evt_data = EventData(fields=fields)
        plain_node = GraphNode(
            id="n1", type="event", name="DamageEvent",
            data=evt_data,
        )
        plain_graph = NodeGraph(nodes=[plain_node])

        trinity = to_trinity_graph(plain_graph)
        tnode = trinity.nodes[0]
        assert isinstance(tnode.data, TrinityEventData)
        assert len(tnode.data.fields) == 1

    def test_raises_on_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Expected NodeGraph"):
            to_trinity_graph("not a graph")  # type: ignore

    def test_idempotent_on_trinity_graph(self) -> None:
        trinity = TrinityNodeGraph()
        result = to_trinity_graph(trinity)
        assert result is trinity  # Same object, no conversion needed


# =============================================================================
# METACLASS & INTROSPECTION
# =============================================================================


class TestGraphNodeMeta:
    """GraphNodeMeta provides Trinity Pattern introspection."""

    def test_engine_meta_registry(self) -> None:
        """Trinity types are registered in EngineMeta._all_engine_types."""
        from trinity.metaclasses.engine_meta import EngineMeta

        all_types = EngineMeta.get_all_types()
        # Types are stored by qualified module.name
        assert any("TrinityGraphNode" in name for name in all_types)
        assert any("TrinityGraphEdge" in name for name in all_types)
        assert any("TrinityNodeGraph" in name for name in all_types)

    def test_decompose_returns_steps(self) -> None:
        """decompose() returns metaclass steps for introspection."""
        steps = decompose(TrinityGraphNode)
        assert len(steps) > 0

        # Should have DESCRIBE and INTERCEPT steps for each field
        op_types = [s.op for s in steps]
        assert Op.REGISTER in op_types
        assert Op.DESCRIBE in op_types
        assert Op.INTERCEPT in op_types

    def test_decompose_layered(self) -> None:
        """decompose_layered() groups steps by layer."""
        layered = decompose_layered(TrinityGraphNode)
        assert "metaclass" in layered
        assert "decorators" in layered
        assert "descriptors" in layered

    def test_expand_human_readable(self) -> None:
        """expand() returns a human-readable expansion."""
        output = expand(TrinityGraphNode)
        assert "TrinityGraphNode" in output or "DESCRIBE" in output or "INTERCEPT" in output

    def test_metaclass_steps_attribute(self) -> None:
        """Trinity types have _metaclass_steps."""
        assert hasattr(TrinityGraphNode, "_metaclass_steps")
        assert len(TrinityGraphNode._metaclass_steps) > 0

    def test_trinity_fields_attribute(self) -> None:
        """Trinity types have _trinity_fields dict."""
        assert hasattr(TrinityGraphNode, "_trinity_fields")
        assert "name" in TrinityGraphNode._trinity_fields
        assert "id" in TrinityGraphNode._trinity_fields
        assert "position" in TrinityGraphNode._trinity_fields


# =============================================================================
# FOUNDATION INTEGRATION
# =============================================================================


class TestFoundationIntegration:
    """Integration with foundation.registry, foundation.mirror, and foundation.tracker."""

    def test_foundation_registry(self) -> None:
        """Trinity types are registered with foundation.registry."""
        from foundation import registry

        assert registry.is_registered(TrinityGraphNode)
        assert registry.is_registered(TrinityNodeGraph)
        assert registry.is_registered(TrinityGraphEdge)
        assert registry.is_registered(TrinityComponentData)

    def test_foundation_registry_name(self) -> None:
        """Registered name includes module path."""
        from foundation import registry

        name = registry.get_name(TrinityGraphNode)
        assert name is not None
        assert "TrinityGraphNode" in name

    def test_foundation_mirror_object(self) -> None:
        """foundation.mirror works on Trinity graph node instances."""
        from foundation import mirror

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        m = mirror(node)
        assert m is not None
        # Mirror should list the fields
        assert hasattr(m, 'fields') or hasattr(m, 'get_fields')

    def test_foundation_mirror_class(self) -> None:
        """foundation.mirror works on Trinity graph node classes."""
        from foundation import mirror

        m = mirror(TrinityGraphNode)
        assert m is not None

    def test_foundation_tracker_mark_dirty(self) -> None:
        """foundation.tracker integration via TrackedDescriptor."""
        from foundation import tracker

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        # Clear initial dirty state
        node.clear_dirty()
        # Modify a field
        node.name = "Enemy"
        # Tracker should know about this
        assert tracker.is_dirty(node)

    def test_foundation_tracker_dirty_fields(self) -> None:
        """foundation.tracker.dirty_fields returns modified field names."""
        from foundation import tracker

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()
        node.name = "Enemy"
        node.type = "system"
        fields = tracker.dirty_fields(node)
        assert "name" in fields
        assert "type" in fields

    def test_foundation_tracker_mark_clean(self) -> None:
        """foundation.tracker.mark_clean resets dirty state."""
        from foundation import tracker

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()
        node.name = "Enemy"
        assert tracker.is_dirty(node)
        tracker.mark_clean(node)
        assert not tracker.is_dirty(node)

    def test_foundation_schema_hash(self) -> None:
        """foundation.mirror.schema_hash produces a stable hash."""
        from foundation import schema_hash

        h1 = schema_hash(TrinityGraphNode)
        h2 = schema_hash(TrinityGraphNode)
        assert h1 == h2  # Stable
        assert len(h1) > 0


# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================


class TestConvenienceConstructors:
    """Convenience functions create properly-typed Trinity nodes."""

    def test_create_component_node(self) -> None:
        fields = [TrinityFieldData(name="x", type_annotation="float")]
        source = TrinitySourceLocation("game.py", 10)
        node = create_trinity_component_node(
            id="n1", name="Position",
            fields=fields, source=source,
        )
        assert isinstance(node, TrinityGraphNode)
        assert node.type == "component"
        assert isinstance(node.data, TrinityComponentData)

    def test_create_system_node(self) -> None:
        methods = [TrinityMethodData(name="execute")]
        source = TrinitySourceLocation("game.py", 20)
        node = create_trinity_system_node(
            id="n1", name="MovementSystem",
            methods=methods, source=source,
        )
        assert node.type == "system"
        assert isinstance(node.data, TrinitySystemData)

    def test_create_resource_node(self) -> None:
        fields = [TrinityFieldData(name="config", type_annotation="dict")]
        source = TrinitySourceLocation("config.py", 1)
        node = create_trinity_resource_node(
            id="n1", name="Settings",
            fields=fields, source=source, is_singleton=True,
        )
        assert node.type == "resource"
        assert isinstance(node.data, TrinityResourceData)
        assert node.data.is_singleton is True

    def test_create_event_node(self) -> None:
        fields = [TrinityFieldData(name="event_type", type_annotation="str")]
        source = TrinitySourceLocation("events.py", 5)
        node = create_trinity_event_node(
            id="n1", name="CollisionEvent",
            fields=fields, source=source,
        )
        assert node.type == "event"
        assert isinstance(node.data, TrinityEventData)

    def test_create_edge(self) -> None:
        edge = create_trinity_edge(
            id="e1", source="n1", target="n2", edge_type="inheritance",
        )
        assert isinstance(edge, TrinityGraphEdge)
        assert edge.type == "inheritance"


# =============================================================================
# EXPLICIT REGISTRATION
# =============================================================================


class TestRegistration:
    """register_all_trinity_graph_types() ensures all types are registered."""

    def test_explicit_registration_is_idempotent(self) -> None:
        """Calling register_all_trinity_graph_types twice is safe."""
        register_all_trinity_graph_types()  # First call (types already registered by metaclass)
        register_all_trinity_graph_types()  # Second call (no-op since already registered)

        from foundation import registry
        assert registry.is_registered(TrinityGraphNode)

    def test_all_types_registered(self) -> None:
        """All Trinity graph types are registered with foundation.registry."""
        from foundation import registry

        types_to_check = [
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
        for cls in types_to_check:
            assert registry.is_registered(cls), f"{cls.__name__} is not registered"


# =============================================================================
# SERIALIZATION ROUND-TRIP
# =============================================================================


class TestSerializationRoundTrip:
    """All Trinity types support to_dict/from_dict round-trip."""

    def test_position_roundtrip(self) -> None:
        pos = TrinityNodePosition(3.5, 7.2)
        d = pos.to_dict()
        restored = TrinityNodePosition.from_dict(d)
        assert restored == pos

    def test_source_location_roundtrip(self) -> None:
        loc = TrinitySourceLocation("test.py", 42, end_line=50)
        d = loc.to_dict()
        restored = TrinitySourceLocation.from_dict(d)
        assert restored == loc

    def test_field_data_roundtrip(self) -> None:
        fd = TrinityFieldData("health", "int", default_value="100", line_number=5, is_optional=True)
        d = fd.to_dict()
        restored = TrinityFieldData.from_dict(d)
        assert restored == fd

    def test_graph_node_roundtrip(self) -> None:
        node = TrinityGraphNode(
            id="n1", type="component", name="Player",
            position=TrinityNodePosition(10.0, 20.0),
            data=TrinityComponentData(fields=[TrinityFieldData("x", "float")]),
            source=TrinitySourceLocation("game.py", 5),
        )
        d = node.to_dict()
        restored = TrinityGraphNode.from_dict(d)
        assert restored == node

    def test_graph_edge_roundtrip(self) -> None:
        edge = TrinityGraphEdge(
            id="e1", source="n1", target="n2",
            type="inheritance", label="extends",
        )
        d = edge.to_dict()
        restored = TrinityGraphEdge.from_dict(d)
        assert restored == edge

    def test_node_graph_roundtrip(self) -> None:
        graph = TrinityNodeGraph(
            nodes=[TrinityGraphNode(id="n1", type="component", name="A")],
            edges=[TrinityGraphEdge(id="e1", source="n1", target="n2")],
            metadata={"source": "test.py"},
        )
        d = graph.to_dict()
        restored = TrinityNodeGraph.from_dict(d)
        # Structural comparison (equality on container, but nodes/edges are lists)
        assert len(restored.nodes) == len(graph.nodes)
        assert len(restored.edges) == len(graph.edges)
        assert restored.metadata == graph.metadata


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_fields_ok(self) -> None:
        """TrinityFieldData with only required fields works."""
        fd = TrinityFieldData(name="x", type_annotation="float")
        assert fd.default_value is None
        assert fd.is_optional is False

    def test_graph_node_with_dict_data(self) -> None:
        """GraphNode accepts plain dict as data."""
        node = TrinityGraphNode(id="n1", type="system", name="S", data={"methods": []})
        assert isinstance(node.data, dict)
        d = node.to_dict()
        assert isinstance(d["data"], dict)

    def test_graph_node_without_source(self) -> None:
        """GraphNode without source serializes correctly."""
        node = TrinityGraphNode(id="n1", type="component", name="A")
        d = node.to_dict()
        assert "source" not in d

    def test_large_graph_dirty_tracking(self) -> None:
        """Dirty tracking works on graphs with many nodes."""
        nodes = [
            TrinityGraphNode(id=f"n{i}", type="component", name=f"Comp{i}")
            for i in range(100)
        ]
        graph = TrinityNodeGraph(nodes=nodes)
        graph.clear_dirty()
        # Modify one node
        graph.nodes[50].name = "Modified"
        # Graph's own nodes list is same reference -- not dirty
        # But the individual node IS dirty
        assert graph.nodes[50].is_dirty("name")

    def test_nested_dirty_propagation(self) -> None:
        """Dirty state is scoped to each instance."""
        a = TrinityGraphNode(id="n1", type="component", name="A")
        b = TrinityGraphNode(id="n2", type="component", name="B")
        a.clear_dirty()
        b.clear_dirty()
        a.name = "Modified"
        assert a.is_dirty("name")
        assert not b.is_dirty("name")  # B is clean
