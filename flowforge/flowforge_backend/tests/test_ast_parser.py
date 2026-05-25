"""Tests for AST parser modules.

This module provides comprehensive tests for the Phase 2 AST parser components:
- TrinityASTVisitor: Parses Python code and extracts Trinity definitions
- GraphBuilder: Converts parsed definitions to graph nodes
- EdgeBuilder: Detects relationships and creates edges
- LayoutEngine: Positions nodes for visualization
"""

from __future__ import annotations

import pytest

from ..ast_parser.visitor import TrinityASTVisitor, parse_source, parse_file
from ..ast_parser.graph_builder import GraphBuilder, build_graph_from_parse_result
from ..ast_parser.edge_builder import EdgeBuilder
from ..ast_parser.layout import LayoutEngine, apply_layout, get_layout_info
from ..ast_parser.types import (
    ComponentDef,
    SystemDef,
    ResourceDef,
    EventDef,
    FieldDef,
    MethodDef,
    DecoratorArgs,
    TrinityDecoratorType,
)
from ..ast_parser.graph_types import (
    GraphNode,
    GraphEdge,
    NodeGraph,
    NodeType,
    EdgeType,
    NodePosition,
    SourceLocation,
)
from ..ast_parser.constants import (
    NODE_WIDTH,
    NODE_HEIGHT,
    VERTICAL_SPACING,
)


# =============================================================================
# TrinityASTVisitor Tests
# =============================================================================


class TestTrinityASTVisitorComponent:
    """Tests for parsing @component decorated classes."""

    def test_parse_simple_component(self):
        """Test parsing a simple @component decorated class."""
        source = """
from trinity import component

@component
class Position:
    x: float = 0.0
    y: float = 0.0
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.components) == 1
        comp = result.components[0]
        assert comp.name == "Position"
        assert comp.decorator_type == TrinityDecoratorType.COMPONENT
        assert len(comp.fields) == 2

    def test_parse_component_with_docstring(self):
        """Test parsing a component with docstring."""
        source = """
from trinity import component

@component
class Player:
    '''The player entity component.'''
    name: str
    health: int = 100
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.components) == 1
        comp = result.components[0]
        assert comp.name == "Player"
        assert comp.docstring == "The player entity component."

    def test_parse_component_called_decorator(self):
        """Test parsing @component() with empty call."""
        source = """
from trinity import component

@component()
class Velocity:
    dx: float
    dy: float
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.components) == 1
        assert result.components[0].name == "Velocity"

    def test_parse_component_with_decorator_args(self):
        """Test parsing @component with arguments."""
        source = """
from trinity import component

@component(priority=10, tag="physics")
class RigidBody:
    mass: float = 1.0
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.components) == 1
        comp = result.components[0]
        assert comp.decorator_args.keyword.get("priority") == 10
        assert comp.decorator_args.keyword.get("tag") == "physics"

    def test_parse_component_complex_field_types(self):
        """Test parsing component with complex type annotations."""
        source = """
from trinity import component
from typing import List, Optional

@component
class Inventory:
    items: List[str]
    gold: Optional[int] = None
    data: dict[str, int] = {}
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.components) == 1
        comp = result.components[0]
        assert len(comp.fields) == 3

        items_field = next(f for f in comp.fields if f.name == "items")
        assert items_field.type_annotation == "List[str]"

        gold_field = next(f for f in comp.fields if f.name == "gold")
        assert gold_field.type_annotation == "Optional[int]"
        assert gold_field.default_value == "None"

    def test_parse_component_aliased_import(self):
        """Test parsing component with aliased decorator import."""
        source = """
from trinity import component as comp

@comp
class Health:
    value: int = 100
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.components) == 1
        assert result.components[0].name == "Health"


class TestTrinityASTVisitorSystem:
    """Tests for parsing @system decorated classes."""

    def test_parse_simple_system(self):
        """Test parsing a simple @system decorated class."""
        source = """
from trinity import system

@system
class MovementSystem:
    def update(self, entities):
        pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.systems) == 1
        sys = result.systems[0]
        assert sys.name == "MovementSystem"
        assert sys.decorator_type == TrinityDecoratorType.SYSTEM

    def test_parse_system_with_query(self):
        """Test parsing system with Query type annotation."""
        source = """
from trinity import system

@system
class PhysicsSystem:
    def update(self, entities: Query[Position, Velocity]) -> None:
        pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.systems) == 1
        sys = result.systems[0]
        assert "Position" in sys.queries
        assert "Velocity" in sys.queries

    def test_parse_system_multiple_methods(self):
        """Test parsing system with multiple methods."""
        source = """
from trinity import system

@system
class RenderSystem:
    def setup(self) -> None:
        pass

    def update(self, entities: Query[Sprite, Position]) -> None:
        pass

    def teardown(self) -> None:
        pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.systems) == 1
        sys = result.systems[0]
        assert len(sys.methods) == 3
        method_names = [m.name for m in sys.methods]
        assert "setup" in method_names
        assert "update" in method_names
        assert "teardown" in method_names

    def test_parse_system_with_fields(self):
        """Test parsing system with class-level fields."""
        source = """
from trinity import system

@system
class TimerSystem:
    elapsed_time: float = 0.0
    delta_time: float = 1/60

    def update(self, entities) -> None:
        pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.systems) == 1
        sys = result.systems[0]
        assert len(sys.fields) == 2

    def test_parse_system_extract_query_info(self):
        """Test that query_info is extracted from method parameters."""
        source = """
from trinity import system

@system
class CollisionSystem:
    def check_collisions(self, entities: Query[Position, Collider, RigidBody]) -> None:
        pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        sys = result.systems[0]
        update_method = sys.methods[0]
        assert update_method.query_info is not None
        assert "Position" in update_method.query_info.component_types
        assert "Collider" in update_method.query_info.component_types
        assert "RigidBody" in update_method.query_info.component_types


class TestTrinityASTVisitorResource:
    """Tests for parsing @resource decorated classes."""

    def test_parse_simple_resource(self):
        """Test parsing a simple @resource decorated class."""
        source = """
from trinity import resource

@resource
class GameConfig:
    screen_width: int = 800
    screen_height: int = 600
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.resources) == 1
        res = result.resources[0]
        assert res.name == "GameConfig"
        assert res.decorator_type == TrinityDecoratorType.RESOURCE
        assert res.is_singleton is True  # Default

    def test_parse_resource_singleton_false(self):
        """Test parsing resource with singleton=False."""
        source = """
from trinity import resource

@resource(singleton=False)
class PlayerStats:
    score: int = 0
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.resources) == 1
        res = result.resources[0]
        assert res.is_singleton is False

    def test_parse_resource_with_methods(self):
        """Test parsing resource with helper methods."""
        source = """
from trinity import resource

@resource
class AssetManager:
    assets: dict = {}

    def load_asset(self, name: str) -> None:
        pass

    def get_asset(self, name: str) -> Any:
        pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.resources) == 1
        res = result.resources[0]
        assert len(res.methods) == 2


class TestTrinityASTVisitorEvent:
    """Tests for parsing @event decorated classes."""

    def test_parse_simple_event(self):
        """Test parsing a simple @event decorated class."""
        source = """
from trinity import event

@event
class CollisionEvent:
    entity_a: int
    entity_b: int
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        assert len(result.events) == 1
        evt = result.events[0]
        assert evt.name == "CollisionEvent"
        assert evt.decorator_type == TrinityDecoratorType.EVENT

    def test_parse_event_payload_fields(self):
        """Test that event fields become payload_fields."""
        source = """
from trinity import event

@event
class DamageEvent:
    target: int
    amount: float
    source: str = "unknown"
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        evt = result.events[0]
        assert len(evt.payload_fields) == 3

        target_field = next(f for f in evt.payload_fields if f.name == "target")
        assert target_field.type_annotation == "int"

    def test_parse_event_with_optional_payload(self):
        """Test parsing event with optional payload fields."""
        source = """
from trinity import event
from typing import Optional

@event
class InputEvent:
    key: str
    modifier: Optional[str] = None
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)
        result = visitor.get_trinity_result()

        evt = result.events[0]
        modifier_field = next(f for f in evt.payload_fields if f.name == "modifier")
        assert modifier_field.default_value == "None"


class TestTrinityASTVisitorFieldExtraction:
    """Tests for field annotation extraction."""

    def test_extract_field_name(self):
        """Test field name extraction."""
        source = """
from trinity import component

@component
class TestComp:
    my_field: int
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        comp = visitor.components[0]
        assert comp.fields[0].name == "my_field"

    def test_extract_field_type(self):
        """Test field type annotation extraction."""
        source = """
from trinity import component

@component
class TestComp:
    simple: int
    generic: List[str]
    nested: Dict[str, List[int]]
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        comp = visitor.components[0]
        fields_by_name = {f.name: f for f in comp.fields}

        assert fields_by_name["simple"].type_annotation == "int"
        assert fields_by_name["generic"].type_annotation == "List[str]"
        assert fields_by_name["nested"].type_annotation == "Dict[str, List[int]]"

    def test_extract_field_default(self):
        """Test field default value extraction."""
        source = """
from trinity import component

@component
class TestComp:
    no_default: int
    int_default: int = 42
    str_default: str = "hello"
    float_default: float = 3.14
    bool_default: bool = True
    list_default: list = []
    none_default: int = None
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        comp = visitor.components[0]
        fields_by_name = {f.name: f for f in comp.fields}

        assert fields_by_name["no_default"].default_value is None
        assert fields_by_name["int_default"].default_value == "42"
        assert fields_by_name["str_default"].default_value == "'hello'"
        assert fields_by_name["float_default"].default_value == "3.14"
        assert fields_by_name["bool_default"].default_value == "True"
        assert fields_by_name["none_default"].default_value == "None"

    def test_extract_field_line_number(self):
        """Test field line number extraction."""
        source = """from trinity import component

@component
class TestComp:
    field_a: int
    field_b: str
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        comp = visitor.components[0]
        # field_a should be on line 5, field_b on line 6
        assert comp.fields[0].line_number == 5
        assert comp.fields[1].line_number == 6


class TestTrinityASTVisitorMethodExtraction:
    """Tests for method signature extraction."""

    def test_extract_method_name(self):
        """Test method name extraction."""
        source = """
from trinity import system

@system
class TestSys:
    def my_method(self):
        pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        sys = visitor.systems[0]
        assert sys.methods[0].name == "my_method"

    def test_extract_method_parameters(self):
        """Test method parameter extraction."""
        source = """
from trinity import system

@system
class TestSys:
    def process(self, value: int, name: str = "default") -> bool:
        pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        sys = visitor.systems[0]
        method = sys.methods[0]

        # self is skipped
        assert len(method.parameters) == 2

        value_param = method.parameters[0]
        assert value_param.name == "value"
        assert value_param.type_annotation == "int"
        assert value_param.default_value is None

        name_param = method.parameters[1]
        assert name_param.name == "name"
        assert name_param.type_annotation == "str"
        assert name_param.default_value == "'default'"

    def test_extract_method_return_type(self):
        """Test method return type extraction."""
        source = """
from trinity import system

@system
class TestSys:
    def no_return(self):
        pass

    def with_return(self) -> int:
        return 42

    def complex_return(self) -> Optional[List[str]]:
        return None
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        sys = visitor.systems[0]
        methods_by_name = {m.name: m for m in sys.methods}

        assert methods_by_name["no_return"].return_type is None
        assert methods_by_name["with_return"].return_type == "int"
        assert methods_by_name["complex_return"].return_type == "Optional[List[str]]"


class TestTrinityASTVisitorImports:
    """Tests for import tracking."""

    def test_track_simple_import(self):
        """Test tracking simple import statements."""
        source = """
import os
import sys

@component
class TestComp:
    pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        imports = visitor.imports
        module_names = [i.module for i in imports]
        assert "os" in module_names
        assert "sys" in module_names

    def test_track_from_import(self):
        """Test tracking from...import statements."""
        source = """
from typing import List, Optional
from trinity import component

@component
class TestComp:
    items: List[str]
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        imports = visitor.imports
        # Check that trinity import is tracked
        trinity_imports = [i for i in imports if i.module == "trinity"]
        assert len(trinity_imports) == 1
        assert "component" in trinity_imports[0].names

    def test_track_aliased_import(self):
        """Test tracking imports with aliases."""
        source = """
import numpy as np
from trinity import component as comp

@comp
class TestComp:
    pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        imports = visitor.imports
        # numpy import should be tracked
        numpy_imports = [i for i in imports if "numpy" in i.module]
        assert len(numpy_imports) == 1


class TestTrinityASTVisitorDecoratorArgs:
    """Tests for decorator argument handling."""

    def test_positional_args(self):
        """Test extracting positional decorator arguments."""
        source = """
from trinity import component

@component("physics", 10)
class TestComp:
    pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        comp = visitor.components[0]
        assert comp.decorator_args.positional == ["physics", 10]

    def test_keyword_args(self):
        """Test extracting keyword decorator arguments."""
        source = """
from trinity import component

@component(priority=5, enabled=True, name="test")
class TestComp:
    pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        comp = visitor.components[0]
        assert comp.decorator_args.keyword["priority"] == 5
        assert comp.decorator_args.keyword["enabled"] is True
        assert comp.decorator_args.keyword["name"] == "test"

    def test_mixed_args(self):
        """Test extracting mixed positional and keyword arguments."""
        source = """
from trinity import component

@component("category", priority=10)
class TestComp:
    pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        comp = visitor.components[0]
        assert comp.decorator_args.positional == ["category"]
        assert comp.decorator_args.keyword["priority"] == 10

    def test_list_arg(self):
        """Test extracting list argument."""
        source = """
from trinity import component

@component(tags=["physics", "collision"])
class TestComp:
    pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        comp = visitor.components[0]
        assert comp.decorator_args.keyword["tags"] == ["physics", "collision"]

    def test_negative_number_arg(self):
        """Test extracting negative number argument."""
        source = """
from trinity import component

@component(priority=-1)
class TestComp:
    pass
"""
        visitor = TrinityASTVisitor()
        visitor.parse(source)

        comp = visitor.components[0]
        assert comp.decorator_args.keyword["priority"] == -1


# =============================================================================
# GraphBuilder Tests
# =============================================================================


class TestGraphBuilderComponent:
    """Tests for building component nodes."""

    def test_build_component_node(self):
        """Test building a GraphNode from ComponentDef."""
        comp = ComponentDef(
            name="Position",
            decorator_type=TrinityDecoratorType.COMPONENT,
            fields=(
                FieldDef(name="x", type_annotation="float", default_value="0.0"),
                FieldDef(name="y", type_annotation="float", default_value="0.0"),
            ),
            source_file="game.py",
            line_number=10,
        )

        builder = GraphBuilder()
        node = builder.add_component(comp)

        assert node.name == "Position"
        assert node.type == NodeType.COMPONENT
        assert "fields" in node.data
        assert len(node.data["fields"]) == 2

    def test_component_node_has_source_location(self):
        """Test that component node preserves source location."""
        comp = ComponentDef(
            name="Health",
            decorator_type=TrinityDecoratorType.COMPONENT,
            source_file="components.py",
            line_number=25,
        )

        builder = GraphBuilder()
        node = builder.add_component(comp)

        assert node.source is not None
        assert node.source.file == "components.py"
        assert node.source.line == 25

    def test_component_node_with_docstring(self):
        """Test that component node includes docstring."""
        comp = ComponentDef(
            name="Sprite",
            decorator_type=TrinityDecoratorType.COMPONENT,
            docstring="Visual representation of an entity.",
        )

        builder = GraphBuilder()
        node = builder.add_component(comp)

        assert node.data.get("docstring") == "Visual representation of an entity."


class TestGraphBuilderSystem:
    """Tests for building system nodes."""

    def test_build_system_node(self):
        """Test building a GraphNode from SystemDef."""
        sys_def = SystemDef(
            name="MovementSystem",
            decorator_type=TrinityDecoratorType.SYSTEM,
            queries=("Position", "Velocity"),
            source_file="systems.py",
            line_number=15,
        )

        builder = GraphBuilder()
        node = builder.add_system(sys_def)

        assert node.name == "MovementSystem"
        assert node.type == NodeType.SYSTEM
        assert "queries" in node.data
        assert "Position" in node.data["queries"]
        assert "Velocity" in node.data["queries"]

    def test_system_node_with_methods(self):
        """Test that system node includes methods."""
        sys_def = SystemDef(
            name="RenderSystem",
            decorator_type=TrinityDecoratorType.SYSTEM,
            methods=(
                MethodDef(name="update", parameters=()),
                MethodDef(name="draw", parameters=()),
            ),
        )

        builder = GraphBuilder()
        node = builder.add_system(sys_def)

        assert "methods" in node.data
        assert len(node.data["methods"]) == 2


class TestGraphBuilderResource:
    """Tests for building resource nodes."""

    def test_build_resource_node(self):
        """Test building a GraphNode from ResourceDef."""
        res_def = ResourceDef(
            name="GameConfig",
            decorator_type=TrinityDecoratorType.RESOURCE,
            is_singleton=True,
            fields=(
                FieldDef(name="fps", type_annotation="int", default_value="60"),
            ),
        )

        builder = GraphBuilder()
        node = builder.add_resource(res_def)

        assert node.name == "GameConfig"
        assert node.type == NodeType.RESOURCE
        assert node.data.get("is_singleton") is True

    def test_resource_node_non_singleton(self):
        """Test building resource node with singleton=False."""
        res_def = ResourceDef(
            name="PlayerData",
            decorator_type=TrinityDecoratorType.RESOURCE,
            is_singleton=False,
        )

        builder = GraphBuilder()
        node = builder.add_resource(res_def)

        assert node.data.get("is_singleton") is False


class TestGraphBuilderEvent:
    """Tests for building event nodes."""

    def test_build_event_node(self):
        """Test building a GraphNode from EventDef."""
        evt_def = EventDef(
            name="CollisionEvent",
            decorator_type=TrinityDecoratorType.EVENT,
            payload_fields=(
                FieldDef(name="entity_a", type_annotation="int"),
                FieldDef(name="entity_b", type_annotation="int"),
            ),
        )

        builder = GraphBuilder()
        node = builder.add_event(evt_def)

        assert node.name == "CollisionEvent"
        assert node.type == NodeType.EVENT
        assert "payload_fields" in node.data
        assert len(node.data["payload_fields"]) == 2


class TestGraphBuilderNodeID:
    """Tests for node ID generation."""

    def test_unique_node_ids(self):
        """Test that different classes get unique IDs."""
        builder = GraphBuilder()

        comp1 = ComponentDef(name="Position", decorator_type=TrinityDecoratorType.COMPONENT)
        comp2 = ComponentDef(name="Velocity", decorator_type=TrinityDecoratorType.COMPONENT)

        node1 = builder.add_component(comp1)
        node2 = builder.add_component(comp2)

        assert node1.id != node2.id

    def test_deterministic_id_with_source_file(self):
        """Test that same class+file produces same ID."""
        comp = ComponentDef(
            name="Position",
            decorator_type=TrinityDecoratorType.COMPONENT,
            source_file="game.py",
        )

        builder1 = GraphBuilder()
        node1 = builder1.add_component(comp)

        builder2 = GraphBuilder()
        node2 = builder2.add_component(comp)

        assert node1.id == node2.id

    def test_node_id_map(self):
        """Test that node_id_map is populated correctly."""
        builder = GraphBuilder()

        comp = ComponentDef(name="Position", decorator_type=TrinityDecoratorType.COMPONENT)
        node = builder.add_component(comp)

        assert "Position" in builder.node_id_map
        assert builder.node_id_map["Position"] == node.id


class TestGraphBuilderBuild:
    """Tests for the build() method."""

    def test_build_returns_node_graph(self):
        """Test that build() returns a NodeGraph."""
        builder = GraphBuilder()
        comp = ComponentDef(name="Test", decorator_type=TrinityDecoratorType.COMPONENT)
        builder.add_component(comp)

        graph = builder.build()

        assert isinstance(graph, NodeGraph)
        assert len(graph.nodes) == 1

    def test_build_includes_metadata(self):
        """Test that build() includes metadata."""
        builder = GraphBuilder()
        comp = ComponentDef(
            name="Test",
            decorator_type=TrinityDecoratorType.COMPONENT,
            source_file="test.py",
        )
        builder.add_component(comp)

        graph = builder.build()

        assert "node_count" in graph.metadata
        assert graph.metadata["node_count"] == 1
        assert "type_counts" in graph.metadata
        assert graph.metadata["type_counts"]["component"] == 1

    def test_build_from_parse_result(self):
        """Test build_graph_from_parse_result convenience function."""
        source = """
from trinity import component, system

@component
class Position:
    x: float = 0.0

@system
class MovementSystem:
    def update(self):
        pass
"""
        result = parse_source(source)
        graph = build_graph_from_parse_result(result)

        assert len(graph.nodes) == 2
        assert any(n.name == "Position" for n in graph.nodes)
        assert any(n.name == "MovementSystem" for n in graph.nodes)


# =============================================================================
# EdgeBuilder Tests
# =============================================================================


class TestEdgeBuilderTypeReferences:
    """Tests for detecting type references in fields."""

    def test_detect_simple_type_reference(self):
        """Test detecting a simple type reference in a field."""
        # Create nodes
        position_node = GraphNode(
            id="node_1",
            type=NodeType.COMPONENT,
            name="Position",
            data={"fields": []},
        )
        player_node = GraphNode(
            id="node_2",
            type=NodeType.COMPONENT,
            name="Player",
            data={
                "fields": [
                    {"name": "position", "type_annotation": "Position"}
                ]
            },
        )

        nodes = [position_node, player_node]
        node_id_map = {"Position": "node_1", "Player": "node_2"}

        builder = EdgeBuilder(nodes, node_id_map)
        builder.build_reference_edges()
        edges = builder.get_edges()

        assert len(edges) == 1
        edge = edges[0]
        assert edge.source == "node_2"  # Player
        assert edge.target == "node_1"  # Position
        assert edge.type == EdgeType.REFERENCE

    def test_detect_generic_type_reference(self):
        """Test detecting type references in generic types."""
        enemy_node = GraphNode(
            id="node_1",
            type=NodeType.COMPONENT,
            name="Enemy",
            data={"fields": []},
        )
        spawner_node = GraphNode(
            id="node_2",
            type=NodeType.COMPONENT,
            name="Spawner",
            data={
                "fields": [
                    {"name": "targets", "type_annotation": "List[Enemy]"}
                ]
            },
        )

        nodes = [enemy_node, spawner_node]
        node_id_map = {"Enemy": "node_1", "Spawner": "node_2"}

        builder = EdgeBuilder(nodes, node_id_map)
        builder.build_reference_edges()
        edges = builder.get_edges()

        assert len(edges) == 1
        assert edges[0].target == "node_1"

    def test_ignore_builtin_types(self):
        """Test that builtin types don't create edges."""
        comp_node = GraphNode(
            id="node_1",
            type=NodeType.COMPONENT,
            name="Stats",
            data={
                "fields": [
                    {"name": "value", "type_annotation": "int"},
                    {"name": "name", "type_annotation": "str"},
                    {"name": "items", "type_annotation": "List[str]"},
                ]
            },
        )

        nodes = [comp_node]
        node_id_map = {"Stats": "node_1"}

        builder = EdgeBuilder(nodes, node_id_map)
        builder.build_reference_edges()
        edges = builder.get_edges()

        assert len(edges) == 0

    def test_no_self_reference(self):
        """Test that self-references are not created."""
        node = GraphNode(
            id="node_1",
            type=NodeType.COMPONENT,
            name="Node",
            data={
                "fields": [
                    {"name": "parent", "type_annotation": "Node"}
                ]
            },
        )

        nodes = [node]
        node_id_map = {"Node": "node_1"}

        builder = EdgeBuilder(nodes, node_id_map)
        builder.build_reference_edges()
        edges = builder.get_edges()

        assert len(edges) == 0


class TestEdgeBuilderQueryDependencies:
    """Tests for detecting Query dependencies in systems."""

    def test_detect_query_dependency(self):
        """Test detecting Query[...] dependencies."""
        position_node = GraphNode(
            id="node_1",
            type=NodeType.COMPONENT,
            name="Position",
            data={"fields": []},
        )
        system_node = GraphNode(
            id="node_2",
            type=NodeType.SYSTEM,
            name="MovementSystem",
            data={
                "queries": ["Position"],
                "methods": [],
            },
        )

        nodes = [position_node, system_node]
        node_id_map = {"Position": "node_1", "MovementSystem": "node_2"}

        builder = EdgeBuilder(nodes, node_id_map)
        builder.build_query_edges()
        edges = builder.get_edges()

        assert len(edges) == 1
        edge = edges[0]
        assert edge.source == "node_2"  # System
        assert edge.target == "node_1"  # Component
        assert edge.type == EdgeType.QUERY

    def test_detect_multiple_query_components(self):
        """Test detecting multiple components in a Query."""
        position_node = GraphNode(
            id="node_1",
            type=NodeType.COMPONENT,
            name="Position",
            data={"fields": []},
        )
        velocity_node = GraphNode(
            id="node_2",
            type=NodeType.COMPONENT,
            name="Velocity",
            data={"fields": []},
        )
        system_node = GraphNode(
            id="node_3",
            type=NodeType.SYSTEM,
            name="PhysicsSystem",
            data={
                "queries": ["Position", "Velocity"],
                "methods": [],
            },
        )

        nodes = [position_node, velocity_node, system_node]
        node_id_map = {
            "Position": "node_1",
            "Velocity": "node_2",
            "PhysicsSystem": "node_3",
        }

        builder = EdgeBuilder(nodes, node_id_map)
        builder.build_query_edges()
        edges = builder.get_edges()

        assert len(edges) == 2
        targets = {e.target for e in edges}
        assert "node_1" in targets  # Position
        assert "node_2" in targets  # Velocity


class TestEdgeBuilderInheritance:
    """Tests for detecting inheritance relationships."""

    def test_detect_inheritance(self):
        """Test detecting class inheritance."""
        base_node = GraphNode(
            id="node_1",
            type=NodeType.COMPONENT,
            name="BaseComponent",
            data={"fields": [], "bases": []},
        )
        derived_node = GraphNode(
            id="node_2",
            type=NodeType.COMPONENT,
            name="DerivedComponent",
            data={"fields": [], "bases": ["BaseComponent"]},
        )

        nodes = [base_node, derived_node]
        node_id_map = {"BaseComponent": "node_1", "DerivedComponent": "node_2"}

        builder = EdgeBuilder(nodes, node_id_map)
        builder.build_inheritance_edges()
        edges = builder.get_edges()

        assert len(edges) == 1
        edge = edges[0]
        assert edge.source == "node_2"  # Derived
        assert edge.target == "node_1"  # Base
        assert edge.type == EdgeType.INHERITANCE


class TestEdgeBuilderEdgeTypes:
    """Tests for edge type filtering."""

    def test_build_all_edges(self):
        """Test that build_all_edges creates all edge types."""
        comp_node = GraphNode(
            id="node_1",
            type=NodeType.COMPONENT,
            name="Position",
            data={"fields": []},
        )
        system_node = GraphNode(
            id="node_2",
            type=NodeType.SYSTEM,
            name="MovementSystem",
            data={"queries": ["Position"], "methods": []},
        )

        nodes = [comp_node, system_node]
        node_id_map = {"Position": "node_1", "MovementSystem": "node_2"}

        builder = EdgeBuilder(nodes, node_id_map)
        edges = builder.build_all_edges()

        assert len(edges) >= 1

    def test_get_edges_by_type(self):
        """Test filtering edges by type."""
        comp_node = GraphNode(
            id="node_1",
            type=NodeType.COMPONENT,
            name="Position",
            data={"fields": []},
        )
        system_node = GraphNode(
            id="node_2",
            type=NodeType.SYSTEM,
            name="MovementSystem",
            data={"queries": ["Position"], "methods": []},
        )

        nodes = [comp_node, system_node]
        node_id_map = {"Position": "node_1", "MovementSystem": "node_2"}

        builder = EdgeBuilder(nodes, node_id_map)
        builder.build_all_edges()

        query_edges = builder.get_edges_by_type(EdgeType.QUERY)
        assert all(e.type == EdgeType.QUERY for e in query_edges)

    def test_no_duplicate_edges(self):
        """Test that duplicate edges are not created."""
        comp_node = GraphNode(
            id="node_1",
            type=NodeType.COMPONENT,
            name="Position",
            data={"fields": []},
        )
        system_node = GraphNode(
            id="node_2",
            type=NodeType.SYSTEM,
            name="MovementSystem",
            data={
                "queries": ["Position", "Position"],  # Duplicate
                "methods": [],
            },
        )

        nodes = [comp_node, system_node]
        node_id_map = {"Position": "node_1", "MovementSystem": "node_2"}

        builder = EdgeBuilder(nodes, node_id_map)
        builder.build_query_edges()
        edges = builder.get_edges()

        # Should only have one edge despite duplicate query
        assert len(edges) == 1


# =============================================================================
# LayoutEngine Tests
# =============================================================================


class TestLayoutEnginePositioning:
    """Tests for node positioning algorithms."""

    def test_position_nodes_without_overlap(self):
        """Test that nodes are positioned without overlap."""
        nodes = [
            GraphNode(id="1", type=NodeType.COMPONENT, name="A", data={}),
            GraphNode(id="2", type=NodeType.COMPONENT, name="B", data={}),
            GraphNode(id="3", type=NodeType.COMPONENT, name="C", data={}),
        ]
        edges: list[GraphEdge] = []

        engine = LayoutEngine(nodes, edges)
        engine.apply_hierarchical_layout()

        # Check that all nodes have unique positions
        positions = [(n.position.x, n.position.y) for n in nodes]
        assert len(set(positions)) == len(positions)

    def test_group_nodes_by_type(self):
        """Test that nodes are grouped by type in hierarchical layout."""
        nodes = [
            GraphNode(id="1", type=NodeType.COMPONENT, name="Position", data={}),
            GraphNode(id="2", type=NodeType.SYSTEM, name="MovementSystem", data={}),
            GraphNode(id="3", type=NodeType.RESOURCE, name="GameConfig", data={}),
            GraphNode(id="4", type=NodeType.EVENT, name="CollisionEvent", data={}),
        ]
        edges: list[GraphEdge] = []

        engine = LayoutEngine(nodes, edges)
        engine.apply_hierarchical_layout()

        # Get positions by type
        pos_by_type = {}
        for node in nodes:
            pos_by_type[node.type] = (node.position.x, node.position.y)

        # Different types should have different x positions in hierarchical layout
        x_positions = {t: p[0] for t, p in pos_by_type.items()}
        unique_x = len(set(x_positions.values()))
        assert unique_x >= 2  # At least some type separation

    def test_handle_empty_graph(self):
        """Test that empty graph is handled gracefully."""
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        engine = LayoutEngine(nodes, edges)
        # Should not raise
        engine.apply_hierarchical_layout()
        engine.apply_grid_layout()
        engine.apply_compact_layout()

    def test_handle_single_node(self):
        """Test positioning a single node."""
        nodes = [
            GraphNode(id="1", type=NodeType.COMPONENT, name="Only", data={}),
        ]
        edges: list[GraphEdge] = []

        engine = LayoutEngine(nodes, edges, start_x=100, start_y=200)
        engine.apply_hierarchical_layout()

        # Single node should be at or near start position
        assert nodes[0].position.x >= 100
        assert nodes[0].position.y >= 200


class TestLayoutEngineAlgorithms:
    """Tests for different layout algorithms."""

    def test_grid_layout(self):
        """Test grid layout algorithm."""
        nodes = [
            GraphNode(id="1", type=NodeType.COMPONENT, name="A", data={}),
            GraphNode(id="2", type=NodeType.COMPONENT, name="B", data={}),
            GraphNode(id="3", type=NodeType.SYSTEM, name="C", data={}),
        ]
        edges: list[GraphEdge] = []

        engine = LayoutEngine(nodes, edges)
        engine.apply_grid_layout()

        # All nodes should have positions
        for node in nodes:
            assert node.position is not None
            assert node.position.x >= 0
            assert node.position.y >= 0

    def test_compact_layout(self):
        """Test compact layout algorithm."""
        nodes = [
            GraphNode(id=str(i), type=NodeType.COMPONENT, name=f"Node{i}", data={})
            for i in range(10)
        ]
        edges: list[GraphEdge] = []

        engine = LayoutEngine(nodes, edges)
        engine.apply_compact_layout()

        # All nodes should have positions
        for node in nodes:
            assert node.position is not None

    def test_apply_layout_convenience(self):
        """Test apply_layout convenience function."""
        nodes = [
            GraphNode(id="1", type=NodeType.COMPONENT, name="A", data={}),
            GraphNode(id="2", type=NodeType.SYSTEM, name="B", data={}),
        ]
        edges: list[GraphEdge] = []

        apply_layout(nodes, edges, layout_type="hierarchical")

        for node in nodes:
            assert node.position is not None

    def test_apply_layout_invalid_type(self):
        """Test that invalid layout type raises ValueError."""
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        with pytest.raises(ValueError, match="Unknown layout type"):
            apply_layout(nodes, edges, layout_type="invalid")


class TestLayoutEngineBoundingBox:
    """Tests for bounding box calculations."""

    def test_get_bounding_box(self):
        """Test bounding box calculation."""
        nodes = [
            GraphNode(id="1", type=NodeType.COMPONENT, name="A", data={},
                     position=NodePosition(x=100, y=100)),
            GraphNode(id="2", type=NodeType.COMPONENT, name="B", data={},
                     position=NodePosition(x=300, y=200)),
        ]
        edges: list[GraphEdge] = []

        engine = LayoutEngine(nodes, edges)
        min_x, min_y, max_x, max_y = engine.get_bounding_box()

        assert min_x == 100
        assert min_y == 100
        assert max_x == 300 + NODE_WIDTH
        assert max_y == 200 + NODE_HEIGHT

    def test_bounding_box_empty_graph(self):
        """Test bounding box for empty graph."""
        engine = LayoutEngine([], [])
        bbox = engine.get_bounding_box()

        assert bbox == (0.0, 0.0, 0.0, 0.0)

    def test_center_in_viewport(self):
        """Test centering graph in viewport."""
        nodes = [
            GraphNode(id="1", type=NodeType.COMPONENT, name="A", data={},
                     position=NodePosition(x=0, y=0)),
        ]
        edges: list[GraphEdge] = []

        engine = LayoutEngine(nodes, edges)
        engine.center_in_viewport(viewport_width=800, viewport_height=600)

        # Node should be roughly centered
        assert nodes[0].position.x > 0
        assert nodes[0].position.y > 0


class TestLayoutInfo:
    """Tests for layout configuration info."""

    def test_get_layout_info(self):
        """Test getting layout configuration info."""
        info = get_layout_info()

        assert "node_dimensions" in info
        assert info["node_dimensions"]["width"] == NODE_WIDTH
        assert info["node_dimensions"]["height"] == NODE_HEIGHT

        assert "spacing" in info
        assert info["spacing"]["vertical"] == VERTICAL_SPACING

        assert "defaults" in info


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullPipeline:
    """Integration tests for the full parsing-to-graph pipeline."""

    def test_parse_build_layout(self):
        """Test full pipeline from source to laid-out graph."""
        source = """
from trinity import component, system, resource, event

@component
class Position:
    '''Entity position in 2D space.'''
    x: float = 0.0
    y: float = 0.0

@component
class Velocity:
    dx: float = 0.0
    dy: float = 0.0

@system
class MovementSystem:
    def update(self, entities: Query[Position, Velocity]) -> None:
        pass

@resource
class GameTime:
    elapsed: float = 0.0
    delta: float = 1/60

@event
class CollisionEvent:
    entity_a: int
    entity_b: int
"""
        # Parse
        result = parse_source(source, source_file="game.py")

        assert len(result.components) == 2
        assert len(result.systems) == 1
        assert len(result.resources) == 1
        assert len(result.events) == 1

        # Build graph
        graph = build_graph_from_parse_result(result)

        assert len(graph.nodes) == 5

        # Apply layout
        apply_layout(graph.nodes, graph.edges, layout_type="hierarchical")

        # All nodes should have positions
        for node in graph.nodes:
            assert node.position is not None
            assert node.position.x >= 0
            assert node.position.y >= 0

    def test_edges_created_correctly(self):
        """Test that edges are created for system-component relationships."""
        source = """
from trinity import component, system

@component
class Position:
    x: float = 0.0

@component
class Velocity:
    dx: float = 0.0

@system
class PhysicsSystem:
    def update(self, entities: Query[Position, Velocity]) -> None:
        pass
"""
        result = parse_source(source)
        graph = build_graph_from_parse_result(result)

        # Should have edges from PhysicsSystem to Position and Velocity
        query_edges = [e for e in graph.edges if e.type == EdgeType.QUERY]
        assert len(query_edges) == 2

        # All edges should originate from the system
        system_node = next(n for n in graph.nodes if n.name == "PhysicsSystem")
        assert all(e.source == system_node.id for e in query_edges)

    def test_serialization_roundtrip(self):
        """Test that graph can be serialized and deserialized."""
        source = """
from trinity import component

@component
class Position:
    x: float = 0.0
    y: float = 0.0
"""
        result = parse_source(source)
        graph = build_graph_from_parse_result(result)

        # Serialize
        graph_dict = graph.to_dict()

        # Deserialize
        restored = NodeGraph.from_dict(graph_dict)

        assert len(restored.nodes) == len(graph.nodes)
        assert restored.nodes[0].name == graph.nodes[0].name
