"""Tests for graph_to_ast module.

This module tests the conversion of node graphs to Python AST,
including helper functions for parsing type annotations, default values,
decorators, and field assignments.
"""

from __future__ import annotations

import ast
import pytest

from ..codegen.graph_to_ast import (
    _parse_type_annotation,
    _parse_default_value,
    _create_decorator,
    _create_field_assignment,
    _create_method_def,
    ASTBuilder,
    graph_to_ast,
    nodes_to_ast,
)
from ..ast_parser.graph_types import (
    GraphNode,
    NodeGraph,
    ComponentData,
    SystemData,
    ResourceData,
    EventData,
    FieldData,
    MethodData,
    ParameterData,
    NodePosition,
    SourceLocation,
)


class TestParseTypeAnnotation:
    """Tests for _parse_type_annotation function."""

    def test_simple_type_int(self):
        """Test parsing simple int type."""
        result = _parse_type_annotation("int")
        source = ast.unparse(result)
        assert source == "int"

    def test_simple_type_str(self):
        """Test parsing simple str type."""
        result = _parse_type_annotation("str")
        source = ast.unparse(result)
        assert source == "str"

    def test_simple_type_float(self):
        """Test parsing simple float type."""
        result = _parse_type_annotation("float")
        source = ast.unparse(result)
        assert source == "float"

    def test_simple_type_bool(self):
        """Test parsing simple bool type."""
        result = _parse_type_annotation("bool")
        source = ast.unparse(result)
        assert source == "bool"

    def test_generic_list_str(self):
        """Test parsing List[str] type."""
        result = _parse_type_annotation("List[str]")
        source = ast.unparse(result)
        assert source == "List[str]"

    def test_generic_list_lowercase(self):
        """Test parsing list[str] type (Python 3.9+ style)."""
        result = _parse_type_annotation("list[str]")
        source = ast.unparse(result)
        assert source == "list[str]"

    def test_generic_dict(self):
        """Test parsing Dict[str, int] type."""
        result = _parse_type_annotation("Dict[str, int]")
        source = ast.unparse(result)
        assert source == "Dict[str, int]"

    def test_optional_type(self):
        """Test parsing Optional[Player] type."""
        result = _parse_type_annotation("Optional[Player]")
        source = ast.unparse(result)
        assert source == "Optional[Player]"

    def test_union_type(self):
        """Test parsing Union[int, str] type."""
        result = _parse_type_annotation("Union[int, str]")
        source = ast.unparse(result)
        assert source == "Union[int, str]"

    def test_pipe_union_type(self):
        """Test parsing int | str union type (Python 3.10+ style)."""
        result = _parse_type_annotation("int | str")
        source = ast.unparse(result)
        assert source == "int | str"

    def test_custom_type(self):
        """Test parsing custom type like Player."""
        result = _parse_type_annotation("Player")
        source = ast.unparse(result)
        assert source == "Player"

    def test_query_type(self):
        """Test parsing Query[Position, Velocity] type."""
        result = _parse_type_annotation("Query[Position, Velocity]")
        source = ast.unparse(result)
        assert source == "Query[Position, Velocity]"

    def test_none_type(self):
        """Test parsing None type returns Constant(None)."""
        result = _parse_type_annotation("None")
        assert isinstance(result, ast.Constant)
        assert result.value is None

    def test_empty_string(self):
        """Test parsing empty string returns Constant(None)."""
        result = _parse_type_annotation("")
        assert isinstance(result, ast.Constant)
        assert result.value is None

    def test_invalid_type_fallback(self):
        """Test that invalid type syntax falls back to Name node."""
        # This should not raise, but fall back to a Name node
        result = _parse_type_annotation("invalid[")
        assert isinstance(result, ast.Name)
        assert result.id == "invalid["

    def test_nested_generic(self):
        """Test parsing nested generic like List[Dict[str, int]]."""
        result = _parse_type_annotation("List[Dict[str, int]]")
        source = ast.unparse(result)
        assert source == "List[Dict[str, int]]"

    def test_callable_type(self):
        """Test parsing Callable[[int], str] type."""
        result = _parse_type_annotation("Callable[[int], str]")
        source = ast.unparse(result)
        assert source == "Callable[[int], str]"


class TestParseDefaultValue:
    """Tests for _parse_default_value function."""

    def test_none_value(self):
        """Test parsing None default value."""
        result = _parse_default_value("None")
        assert isinstance(result, ast.Constant)
        assert result.value is None

    def test_true_value(self):
        """Test parsing True default value."""
        result = _parse_default_value("True")
        assert isinstance(result, ast.Constant)
        assert result.value is True

    def test_false_value(self):
        """Test parsing False default value."""
        result = _parse_default_value("False")
        assert isinstance(result, ast.Constant)
        assert result.value is False

    def test_integer_value(self):
        """Test parsing integer default value."""
        result = _parse_default_value("42")
        assert isinstance(result, ast.Constant)
        assert result.value == 42

    def test_negative_integer(self):
        """Test parsing negative integer default value."""
        result = _parse_default_value("-10")
        source = ast.unparse(result)
        assert source == "-10"

    def test_float_value(self):
        """Test parsing float default value."""
        result = _parse_default_value("3.14")
        assert isinstance(result, ast.Constant)
        assert result.value == 3.14

    def test_string_value(self):
        """Test parsing string default value."""
        result = _parse_default_value('"hello"')
        assert isinstance(result, ast.Constant)
        assert result.value == "hello"

    def test_single_quoted_string(self):
        """Test parsing single-quoted string default value."""
        result = _parse_default_value("'world'")
        assert isinstance(result, ast.Constant)
        assert result.value == "world"

    def test_empty_list(self):
        """Test parsing empty list default value."""
        result = _parse_default_value("[]")
        assert isinstance(result, ast.List)
        assert len(result.elts) == 0

    def test_empty_dict(self):
        """Test parsing empty dict default value."""
        result = _parse_default_value("{}")
        assert isinstance(result, ast.Dict)
        assert len(result.keys) == 0

    def test_list_with_values(self):
        """Test parsing list with values."""
        result = _parse_default_value("[1, 2, 3]")
        assert isinstance(result, ast.List)
        assert len(result.elts) == 3

    def test_dict_with_values(self):
        """Test parsing dict with values."""
        result = _parse_default_value('{"a": 1, "b": 2}')
        assert isinstance(result, ast.Dict)
        assert len(result.keys) == 2

    def test_empty_string_returns_none(self):
        """Test that empty string returns Constant(None)."""
        result = _parse_default_value("")
        assert isinstance(result, ast.Constant)
        assert result.value is None

    def test_invalid_syntax_becomes_string(self):
        """Test that invalid syntax is treated as string constant."""
        result = _parse_default_value("not valid python [")
        assert isinstance(result, ast.Constant)
        assert result.value == "not valid python ["

    def test_tuple_value(self):
        """Test parsing tuple default value."""
        result = _parse_default_value("(1, 2)")
        assert isinstance(result, ast.Tuple)
        assert len(result.elts) == 2

    def test_zero_value(self):
        """Test parsing zero as default value."""
        result = _parse_default_value("0")
        assert isinstance(result, ast.Constant)
        assert result.value == 0

    def test_float_zero(self):
        """Test parsing float zero as default value."""
        result = _parse_default_value("0.0")
        assert isinstance(result, ast.Constant)
        assert result.value == 0.0


class TestCreateDecorator:
    """Tests for _create_decorator function."""

    def test_simple_decorator(self):
        """Test creating simple decorator without arguments."""
        result = _create_decorator("component")
        assert isinstance(result, ast.Name)
        assert result.id == "component"

    def test_simple_decorator_system(self):
        """Test creating simple @system decorator."""
        result = _create_decorator("system")
        assert isinstance(result, ast.Name)
        assert result.id == "system"

    def test_decorator_with_keyword_args(self):
        """Test creating decorator with keyword arguments."""
        result = _create_decorator("component", {"keyword": {"priority": 1}})
        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == "component"
        assert len(result.keywords) == 1
        assert result.keywords[0].arg == "priority"
        assert result.keywords[0].value.value == 1

    def test_decorator_with_positional_args(self):
        """Test creating decorator with positional arguments."""
        result = _create_decorator("system", {"positional": ["first", "second"]})
        assert isinstance(result, ast.Call)
        assert len(result.args) == 2
        assert result.args[0].value == "first"
        assert result.args[1].value == "second"

    def test_decorator_with_mixed_args(self):
        """Test creating decorator with both positional and keyword args."""
        result = _create_decorator("event", {
            "positional": ["name"],
            "keyword": {"priority": 10}
        })
        assert isinstance(result, ast.Call)
        assert len(result.args) == 1
        assert len(result.keywords) == 1

    def test_decorator_with_empty_args_dict(self):
        """Test that empty args dict creates simple decorator."""
        result = _create_decorator("resource", {})
        assert isinstance(result, ast.Name)
        assert result.id == "resource"

    def test_decorator_with_none_args(self):
        """Test that None args creates simple decorator."""
        result = _create_decorator("event", None)
        assert isinstance(result, ast.Name)
        assert result.id == "event"

    def test_decorator_with_empty_keyword_dict(self):
        """Test decorator with empty keyword dict creates simple decorator."""
        result = _create_decorator("system", {"keyword": {}})
        assert isinstance(result, ast.Name)
        assert result.id == "system"


class TestCreateFieldAssignment:
    """Tests for _create_field_assignment function."""

    def test_simple_field_no_default(self):
        """Test creating field without default value."""
        result = _create_field_assignment("health", "int")
        assert isinstance(result, ast.AnnAssign)
        assert result.target.id == "health"
        assert result.value is None

    def test_field_with_default(self):
        """Test creating field with default value."""
        result = _create_field_assignment("health", "int", "100")
        assert isinstance(result, ast.AnnAssign)
        assert result.target.id == "health"
        assert result.value.value == 100

    def test_field_with_string_default(self):
        """Test creating field with string default value."""
        result = _create_field_assignment("name", "str", '"Player"')
        assert isinstance(result, ast.AnnAssign)
        assert result.value.value == "Player"

    def test_field_with_list_type(self):
        """Test creating field with list type."""
        result = _create_field_assignment("items", "List[str]", "[]")
        source = ast.unparse(result)
        assert "items" in source
        assert "List[str]" in source

    def test_field_with_optional_type(self):
        """Test creating field with Optional type."""
        result = _create_field_assignment("parent", "Optional[Node]", "None")
        source = ast.unparse(result)
        assert "parent" in source
        assert "Optional[Node]" in source

    def test_field_simple_is_1(self):
        """Test that simple attribute is set to 1."""
        result = _create_field_assignment("x", "int")
        assert result.simple == 1


class TestASTBuilder:
    """Tests for ASTBuilder class."""

    def test_build_empty_module(self):
        """Test building empty module."""
        builder = ASTBuilder()
        module = builder.build()
        assert isinstance(module, ast.Module)
        assert len(module.body) == 0

    def test_add_component_node(self):
        """Test adding component node."""
        builder = ASTBuilder()
        node = GraphNode(
            id="comp1",
            type="component",
            name="Position",
            data=ComponentData(
                fields=[
                    FieldData(name="x", type_annotation="float", default_value="0.0"),
                    FieldData(name="y", type_annotation="float", default_value="0.0"),
                ]
            ),
        )
        builder.add_node(node)
        module = builder.build()

        assert len(module.body) == 1
        class_def = module.body[0]
        assert isinstance(class_def, ast.ClassDef)
        assert class_def.name == "Position"
        assert len(class_def.decorator_list) == 1

    def test_add_system_node(self):
        """Test adding system node."""
        builder = ASTBuilder()
        node = GraphNode(
            id="sys1",
            type="system",
            name="MovementSystem",
            data=SystemData(
                methods=[
                    MethodData(
                        name="update",
                        parameters=[],
                        return_type="None",
                        query_components=["Position", "Velocity"],
                    )
                ]
            ),
        )
        builder.add_node(node)
        module = builder.build()

        assert len(module.body) == 1
        class_def = module.body[0]
        assert class_def.name == "MovementSystem"
        # Should have update method
        assert any(
            isinstance(stmt, ast.FunctionDef) and stmt.name == "update"
            for stmt in class_def.body
        )

    def test_add_resource_node(self):
        """Test adding resource node."""
        builder = ASTBuilder()
        node = GraphNode(
            id="res1",
            type="resource",
            name="GameConfig",
            data=ResourceData(
                fields=[
                    FieldData(name="difficulty", type_annotation="int", default_value="1"),
                ]
            ),
        )
        builder.add_node(node)
        module = builder.build()

        class_def = module.body[0]
        assert class_def.name == "GameConfig"
        # Check decorator is @resource
        decorator = class_def.decorator_list[0]
        assert isinstance(decorator, ast.Name)
        assert decorator.id == "resource"

    def test_add_event_node(self):
        """Test adding event node."""
        builder = ASTBuilder()
        node = GraphNode(
            id="evt1",
            type="event",
            name="PlayerDied",
            data=EventData(
                fields=[
                    FieldData(name="player_id", type_annotation="str"),
                ]
            ),
        )
        builder.add_node(node)
        module = builder.build()

        class_def = module.body[0]
        assert class_def.name == "PlayerDied"
        decorator = class_def.decorator_list[0]
        assert isinstance(decorator, ast.Name)
        assert decorator.id == "event"

    def test_unknown_node_type_skipped(self):
        """Test that unknown node types are skipped."""
        builder = ASTBuilder()
        node = GraphNode(
            id="unknown1",
            type="unknown",  # type: ignore
            name="UnknownThing",
            data={},
        )
        builder.add_node(node)
        module = builder.build()

        assert len(module.body) == 0

    def test_component_with_docstring(self):
        """Test component with docstring."""
        builder = ASTBuilder()
        node = GraphNode(
            id="comp1",
            type="component",
            name="Health",
            data=ComponentData(
                fields=[FieldData(name="value", type_annotation="int")],
                docstring="Health component for entities.",
            ),
        )
        builder.add_node(node)
        module = builder.build()

        class_def = module.body[0]
        # First statement should be docstring
        assert isinstance(class_def.body[0], ast.Expr)
        assert isinstance(class_def.body[0].value, ast.Constant)
        assert class_def.body[0].value.value == "Health component for entities."

    def test_component_with_bases(self):
        """Test component with base classes."""
        builder = ASTBuilder()
        node = GraphNode(
            id="comp1",
            type="component",
            name="Player",
            data=ComponentData(
                fields=[],
                bases=["Character", "Movable"],
            ),
        )
        builder.add_node(node)
        module = builder.build()

        class_def = module.body[0]
        assert len(class_def.bases) == 2
        assert class_def.bases[0].id == "Character"
        assert class_def.bases[1].id == "Movable"

    def test_event_with_payload_fields(self):
        """Test event with payload_fields (frontend format)."""
        builder = ASTBuilder()
        node = GraphNode(
            id="evt1",
            type="event",
            name="ScoreChanged",
            data={
                "payload_fields": [
                    {"name": "old_score", "type_annotation": "int"},
                    {"name": "new_score", "type_annotation": "int"},
                ]
            },
        )
        builder.add_node(node)
        module = builder.build()

        class_def = module.body[0]
        # Should have two fields from payload_fields
        field_stmts = [s for s in class_def.body if isinstance(s, ast.AnnAssign)]
        assert len(field_stmts) == 2


class TestGraphToAst:
    """Tests for graph_to_ast function."""

    def test_empty_graph(self):
        """Test converting empty graph."""
        graph = NodeGraph(nodes=[], edges=[])
        module = graph_to_ast(graph)

        assert isinstance(module, ast.Module)
        # May have imports even for empty graph
        assert isinstance(module.body, list)

    def test_single_component(self):
        """Test converting graph with single component."""
        graph = NodeGraph(
            nodes=[
                GraphNode(
                    id="1",
                    type="component",
                    name="Velocity",
                    data=ComponentData(
                        fields=[
                            FieldData(name="dx", type_annotation="float"),
                            FieldData(name="dy", type_annotation="float"),
                        ]
                    ),
                )
            ],
            edges=[],
        )
        module = graph_to_ast(graph)
        source = ast.unparse(module)

        assert "class Velocity" in source
        assert "dx" in source
        assert "dy" in source

    def test_graph_from_dict(self):
        """Test converting graph from dictionary format."""
        graph_dict = {
            "nodes": [
                {
                    "id": "n1",
                    "type": "component",
                    "name": "Health",
                    "position": [0, 0],
                    "data": {
                        "fields": [
                            {"name": "value", "type_annotation": "int", "default_value": "100"}
                        ]
                    },
                }
            ],
            "edges": [],
        }
        module = graph_to_ast(graph_dict)
        source = ast.unparse(module)

        assert "class Health" in source
        assert "value" in source

    def test_roundtrip_parse(self):
        """Test that generated code can be parsed back to AST."""
        graph = NodeGraph(
            nodes=[
                GraphNode(
                    id="1",
                    type="component",
                    name="Position",
                    data=ComponentData(
                        fields=[
                            FieldData(name="x", type_annotation="float", default_value="0.0"),
                            FieldData(name="y", type_annotation="float", default_value="0.0"),
                        ]
                    ),
                ),
                GraphNode(
                    id="2",
                    type="system",
                    name="MovementSystem",
                    data=SystemData(
                        methods=[
                            MethodData(name="update", parameters=[], return_type="None")
                        ]
                    ),
                ),
            ],
            edges=[],
        )
        module = graph_to_ast(graph)
        source = ast.unparse(module)

        # Should be valid Python that can be parsed again
        reparsed = ast.parse(source)
        assert isinstance(reparsed, ast.Module)

    def test_node_ordering(self):
        """Test that nodes are ordered by type."""
        graph = NodeGraph(
            nodes=[
                GraphNode(id="1", type="system", name="Sys1", data=SystemData()),
                GraphNode(id="2", type="event", name="Evt1", data=EventData()),
                GraphNode(id="3", type="component", name="Comp1", data=ComponentData()),
                GraphNode(id="4", type="resource", name="Res1", data=ResourceData()),
            ],
            edges=[],
        )
        module = graph_to_ast(graph)

        # Filter to class definitions only
        classes = [s for s in module.body if isinstance(s, ast.ClassDef)]

        # Order should be: component, resource, event, system
        assert classes[0].name == "Comp1"
        assert classes[1].name == "Res1"
        assert classes[2].name == "Evt1"
        assert classes[3].name == "Sys1"

    def test_system_with_query_components(self):
        """Test system with Query[...] parameter in methods."""
        graph = NodeGraph(
            nodes=[
                GraphNode(
                    id="1",
                    type="system",
                    name="PhysicsSystem",
                    data=SystemData(
                        methods=[
                            MethodData(
                                name="update",
                                parameters=[],
                                query_components=["Position", "Velocity"],
                            )
                        ]
                    ),
                )
            ],
            edges=[],
        )
        module = graph_to_ast(graph)
        source = ast.unparse(module)

        assert "Query[Position, Velocity]" in source


class TestNodesToAst:
    """Tests for nodes_to_ast convenience function."""

    def test_nodes_to_ast(self):
        """Test converting nodes list directly."""
        nodes = [
            GraphNode(
                id="1",
                type="component",
                name="Tag",
                data=ComponentData(fields=[]),
            )
        ]
        module = nodes_to_ast(nodes)
        source = ast.unparse(module)

        assert "class Tag" in source


class TestCreateMethodDef:
    """Tests for _create_method_def function."""

    def test_simple_method(self):
        """Test creating simple method with no parameters."""
        result = _create_method_def("update", [], None, None, None, None)
        assert result.name == "update"
        # Should have self parameter
        assert len(result.args.args) == 1
        assert result.args.args[0].arg == "self"
        # Body should have pass
        assert isinstance(result.body[0], ast.Pass)

    def test_method_with_return_type(self):
        """Test method with return type annotation."""
        result = _create_method_def("get_value", [], "int", None, None, None)
        assert result.returns is not None
        source = ast.unparse(result.returns)
        assert source == "int"

    def test_method_with_docstring(self):
        """Test method with docstring."""
        result = _create_method_def("process", [], None, "Process the data.", None, None)
        assert isinstance(result.body[0], ast.Expr)
        assert result.body[0].value.value == "Process the data."

    def test_method_with_parameters(self):
        """Test method with parameters."""
        params = [
            {"name": "delta", "type_annotation": "float"},
            {"name": "count", "type_annotation": "int", "default_value": "1"},
        ]
        result = _create_method_def("update", params, None, None, None, None)
        # self + delta + count
        assert len(result.args.args) == 3
        assert result.args.args[1].arg == "delta"
        assert result.args.args[2].arg == "count"

    def test_method_with_decorators(self):
        """Test method with decorators."""
        result = _create_method_def("cached", [], None, None, ["staticmethod", "cache"], None)
        assert len(result.decorator_list) == 2
        assert result.decorator_list[0].id == "staticmethod"
        assert result.decorator_list[1].id == "cache"

    def test_method_skips_self_in_params(self):
        """Test that explicit self in parameters is not duplicated."""
        params = [{"name": "self"}, {"name": "x", "type_annotation": "int"}]
        result = _create_method_def("method", params, None, None, None, None)
        # Should have exactly 2 args (self + x), not 3
        assert len(result.args.args) == 2
        assert result.args.args[0].arg == "self"
        assert result.args.args[1].arg == "x"
