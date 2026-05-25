"""Blackbox (cleanroom) tests for graph_to_ast module.

These tests treat graph_to_ast() and nodes_to_ast() as black boxes,
verifying the structural fidelity of the AST output relative to the
input scene graph (NodeGraph).  No implementation internals are
inspected -- only the public API contract.

Methodology: Cleanroom / spec-only.  Tests are derived from the type
schema (graph_types.py) and the module docstring contract, NOT from
reading the implementation.
"""

from __future__ import annotations

import ast
import pytest

from ..codegen.graph_to_ast import graph_to_ast, nodes_to_ast
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
)


# =============================================================================
# AST NAVIGATION HELPERS
# =============================================================================

def _find_class(module: ast.Module, name: str) -> ast.ClassDef | None:
    """Return the first ClassDef with the given name, or None."""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _get_decorator_ids(class_def: ast.ClassDef) -> list[str]:
    """Return decorator expression strings for a class."""
    ids: list[str] = []
    for dec in class_def.decorator_list:
        if isinstance(dec, ast.Name):
            ids.append(dec.id)
        elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            ids.append(dec.func.id)
        else:
            ids.append(ast.unparse(dec))
    return ids


def _get_fields(class_def: ast.ClassDef) -> list[tuple[str, str]]:
    """Return (name, type_string) pairs for annotated assignments in a class."""
    fields: list[tuple[str, str]] = []
    for stmt in class_def.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            type_str = ast.unparse(stmt.annotation) if stmt.annotation else ""
            fields.append((stmt.target.id, type_str))
    return fields


def _get_methods(class_def: ast.ClassDef) -> list[ast.FunctionDef]:
    """Return FunctionDef nodes from a class body (excluding pass/docstring)."""
    return [st for st in class_def.body if isinstance(st, ast.FunctionDef)]


def _get_docstring(class_def: ast.ClassDef) -> str | None:
    """Return the docstring of a class, or None."""
    if (class_def.body
            and isinstance(class_def.body[0], ast.Expr)
            and isinstance(class_def.body[0].value, ast.Constant)
            and isinstance(class_def.body[0].value.value, str)):
        return class_def.body[0].value.value
    return None


def _get_bases(class_def: ast.ClassDef) -> list[str]:
    """Return base class names."""
    return [ast.unparse(b) for b in class_def.bases]


# =============================================================================
# 1. SCENE GRAPH COMPLETENESS
# =============================================================================


class TestSceneGraphCompleteness:
    """Every node in the scene graph produces exactly one class definition."""

    def test_all_four_types_produce_classes(self):
        """Component / system / resource / event each produce a ClassDef."""
        nodes = [
            GraphNode(id="c1", type="component", name="Position",
                      data=ComponentData()),
            GraphNode(id="s1", type="system", name="PhysicsSystem",
                      data=SystemData()),
            GraphNode(id="r1", type="resource", name="Config",
                      data=ResourceData()),
            GraphNode(id="e1", type="event", name="CollisionEvent",
                      data=EventData()),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        assert len([n for n in module.body if isinstance(n, ast.ClassDef)]) == 4

    def test_duplicate_node_names_are_not_deduplicated(self):
        """Two nodes with the same name each produce a ClassDef (no dedup)."""
        nodes = [
            GraphNode(id="a", type="component", name="SameName",
                      data=ComponentData()),
            GraphNode(id="b", type="component", name="SameName",
                      data=ComponentData()),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        classes = [n for n in module.body if isinstance(n, ast.ClassDef)
                   and n.name == "SameName"]
        assert len(classes) == 2

    def test_empty_graph_produces_empty_module(self):
        """Graph with no nodes yields a module with no ClassDefs."""
        graph = NodeGraph(nodes=[], edges=[])
        module = graph_to_ast(graph)
        classes = [n for n in module.body if isinstance(n, ast.ClassDef)]
        assert len(classes) == 0

    def test_large_graph_scale(self):
        """100 nodes all produce classes."""
        nodes = [
            GraphNode(id=str(i), type="component", name=f"C{i}",
                      data=ComponentData())
            for i in range(100)
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        classes = [n for n in module.body if isinstance(n, ast.ClassDef)]
        assert len(classes) == 100


# =============================================================================
# 2. NODE TYPE FIDELITY
# =============================================================================


class TestNodeTypeFidelity:
    """Each node type produces the correct decorator."""

    def test_component_decorator(self):
        """@component decorator is applied."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="Pos",
                             data=ComponentData())],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "Pos")
        assert cd is not None
        assert "component" in _get_decorator_ids(cd)

    def test_system_decorator(self):
        """@system decorator is applied."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="MoveSys",
                             data=SystemData())],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "MoveSys")
        assert cd is not None
        assert "system" in _get_decorator_ids(cd)

    def test_resource_decorator(self):
        """@resource decorator is applied."""
        graph = NodeGraph(
            nodes=[GraphNode(id="r1", type="resource", name="Cfg",
                             data=ResourceData())],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "Cfg")
        assert cd is not None
        assert "resource" in _get_decorator_ids(cd)

    def test_event_decorator(self):
        """@event decorator is applied."""
        graph = NodeGraph(
            nodes=[GraphNode(id="e1", type="event", name="Collision",
                             data=EventData())],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "Collision")
        assert cd is not None
        assert "event" in _get_decorator_ids(cd)

    def test_unknown_type_skipped(self):
        """A node with an unrecognised type produces no ClassDef."""
        graph = NodeGraph(
            nodes=[GraphNode(id="x1", type="unknown", name="Foo",
                             data={})],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "Foo")
        assert cd is None


# =============================================================================
# 3. FIELD FIDELITY
# =============================================================================


class TestFieldFidelity:
    """Fields declared in the scene graph appear in the AST."""

    def test_simple_fields(self):
        """int, str, float fields round-trip."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="Pos",
                             data=ComponentData(fields=[
                                 FieldData(name="x", type_annotation="int"),
                                 FieldData(name="name", type_annotation="str"),
                                 FieldData(name="value", type_annotation="float"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "Pos")
        assert cd is not None
        fields = _get_fields(cd)
        assert len(fields) == 3
        assert ("x", "int") in fields
        assert ("name", "str") in fields
        assert ("value", "float") in fields

    def test_field_without_default(self):
        """A field with no default value has no value in the assignment."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="A",
                             data=ComponentData(fields=[
                                 FieldData(name="x", type_annotation="int"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "A")
        assert cd is not None
        ann = cd.body[0]
        assert isinstance(ann, ast.AnnAssign)
        assert ann.value is None

    def test_field_with_default(self):
        """A field with a default value includes it."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="A",
                             data=ComponentData(fields=[
                                 FieldData(name="x", type_annotation="int",
                                           default_value="42"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "A")
        assert cd is not None
        ann = cd.body[0]
        assert isinstance(ann, ast.AnnAssign)
        assert ann.value is not None
        assert ast.unparse(ann.value) == "42"

    def test_bool_defaults(self):
        """True / False default values are preserved."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="A",
                             data=ComponentData(fields=[
                                 FieldData(name="a", type_annotation="bool",
                                           default_value="True"),
                                 FieldData(name="b", type_annotation="bool",
                                           default_value="False"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "A")
        assert cd is not None
        fields = _get_fields(cd)
        # Check by unparsing the full assignment
        body_src = [ast.unparse(st) for st in cd.body if isinstance(st, ast.AnnAssign)]
        assert "a: bool = True" in body_src
        assert "b: bool = False" in body_src

    def test_none_default(self):
        """None default is preserved."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="A",
                             data=ComponentData(fields=[
                                 FieldData(name="x", type_annotation="Optional[int]",
                                           default_value="None"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "A")
        assert cd is not None
        ann = cd.body[0]
        assert isinstance(ann, ast.AnnAssign)
        assert ast.unparse(ann.value) == "None"

    def test_complex_type_annotations(self):
        """list[str], dict[str, int], Optional[str], etc. are parsed."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="A",
                             data=ComponentData(fields=[
                                 FieldData(name="tags", type_annotation="list[str]"),
                                 FieldData(name="counts",
                                           type_annotation="dict[str, int]"),
                                 FieldData(name="maybe",
                                           type_annotation="Optional[str]"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "A")
        assert cd is not None
        fields = _get_fields(cd)
        assert ("tags", "list[str]") in fields
        assert ("counts", "dict[str, int]") in fields
        assert ("maybe", "Optional[str]") in fields

    def test_union_types(self):
        """Union types like int | str are handled."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="A",
                             data=ComponentData(fields=[
                                 FieldData(name="val",
                                           type_annotation="int | str"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "A")
        assert cd is not None
        fields = _get_fields(cd)
        assert ("val", "int | str") in fields

    def test_list_default_value(self):
        """A list default [] round-trips."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="A",
                             data=ComponentData(fields=[
                                 FieldData(name="items",
                                           type_annotation="list[int]",
                                           default_value="[]"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "A")
        assert cd is not None
        ann = cd.body[0]
        assert isinstance(ann, ast.AnnAssign)
        assert ast.unparse(ann.value) == "[]"

    def test_dict_default_value(self):
        """A dict default {} round-trips."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="A",
                             data=ComponentData(fields=[
                                 FieldData(name="mapping",
                                           type_annotation="dict[str, int]",
                                           default_value="{}"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "A")
        assert cd is not None
        ann = cd.body[0]
        assert isinstance(ann, ast.AnnAssign)
        assert ast.unparse(ann.value) == "{}"

    def test_tuple_default_value(self):
        """A tuple default () round-trips."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="A",
                             data=ComponentData(fields=[
                                 FieldData(name="point",
                                           type_annotation="tuple[int, int]",
                                           default_value="(0, 0)"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "A")
        assert cd is not None
        ann = cd.body[0]
        assert isinstance(ann, ast.AnnAssign)
        assert ast.unparse(ann.value) == "(0, 0)"

    def test_empty_component_has_pass(self):
        """A component with no fields produces a `pass` body."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="Empty",
                             data=ComponentData())],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "Empty")
        assert cd is not None
        assert any(isinstance(st, ast.Pass) for st in cd.body)


# =============================================================================
# 4. SYSTEM METHOD FIDELITY
# =============================================================================


class TestSystemMethodFidelity:
    """System node methods are correctly generated."""

    def test_system_with_method(self):
        """A system with one method produces a FunctionDef."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="MoveSys",
                             data=SystemData(methods=[
                                 MethodData(name="update"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "MoveSys")
        assert cd is not None
        methods = _get_methods(cd)
        assert len(methods) == 1
        assert methods[0].name == "update"

    def test_method_with_parameters(self):
        """Method parameters (with types) are preserved."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="S",
                             data=SystemData(methods=[
                                 MethodData(name="run", parameters=[
                                     ParameterData(name="dt",
                                                    type_annotation="float"),
                                     ParameterData(name="count",
                                                    type_annotation="int"),
                                 ]),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "S")
        assert cd is not None
        methods = _get_methods(cd)
        assert len(methods) == 1
        m = methods[0]
        arg_names = [a.arg for a in m.args.args]
        assert "dt" in arg_names
        assert "count" in arg_names

    def test_method_with_return_type(self):
        """Return type annotation is preserved."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="S",
                             data=SystemData(methods=[
                                 MethodData(name="compute", return_type="int"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "S")
        assert cd is not None
        methods = _get_methods(cd)
        assert len(methods) == 1
        assert methods[0].returns is not None
        assert ast.unparse(methods[0].returns) == "int"

    def test_method_docstring(self):
        """Method docstring is preserved."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="S",
                             data=SystemData(methods=[
                                 MethodData(name="run",
                                            docstring="Do the thing."),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "S")
        assert cd is not None
        methods = _get_methods(cd)
        assert len(methods) == 1
        body = methods[0].body
        assert (isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and body[0].value.value == "Do the thing.")

    def test_method_decorators(self):
        """Method-level decorators (e.g. @staticmethod) are preserved."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="S",
                             data=SystemData(methods=[
                                 MethodData(name="helper",
                                            decorators=["staticmethod"]),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "S")
        assert cd is not None
        methods = _get_methods(cd)
        assert len(methods) == 1
        dec_ids = [ast.unparse(d) for d in methods[0].decorator_list]
        assert "staticmethod" in dec_ids

    def test_query_parameter_injection(self):
        """When query_components is set, a Query[...] parameter is injected."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="S",
                             data=SystemData(methods=[
                                 MethodData(
                                     name="run",
                                     query_components=["Position", "Velocity"],
                                 ),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "S")
        assert cd is not None
        methods = _get_methods(cd)
        assert len(methods) == 1
        m = methods[0]
        # First param is self; second should be the Query[...] injection
        assert len(m.args.args) >= 2
        entities_param = m.args.args[1]
        assert entities_param.arg == "entities"
        assert ast.unparse(entities_param.annotation) == "Query[Position, Velocity]"

    def test_multiple_methods(self):
        """Multiple methods on a system are all generated."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="S",
                             data=SystemData(methods=[
                                 MethodData(name="init"),
                                 MethodData(name="run"),
                                 MethodData(name="cleanup"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "S")
        assert cd is not None
        methods = _get_methods(cd)
        assert len(methods) == 3
        names = [m.name for m in methods]
        assert names == ["init", "run", "cleanup"]

    def test_self_not_in_parameters(self):
        """'self' in the parameter list is deduplicated (only one self)."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="S",
                             data=SystemData(methods=[
                                 MethodData(name="run", parameters=[
                                     ParameterData(name="self"),
                                     ParameterData(name="dt",
                                                    type_annotation="float"),
                                 ]),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "S")
        assert cd is not None
        methods = _get_methods(cd)
        assert len(methods) == 1
        m = methods[0]
        self_count = sum(1 for a in m.args.args if a.arg == "self")
        assert self_count == 1

    def test_system_with_fields_and_methods(self):
        """A system can have both fields and methods."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="S",
                             data=SystemData(
                                 fields=[
                                     FieldData(name="speed",
                                               type_annotation="float",
                                               default_value="1.0"),
                                 ],
                                 methods=[
                                     MethodData(name="run"),
                                 ],
                             ),
                         )],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "S")
        assert cd is not None
        assert len(_get_fields(cd)) == 1
        assert len(_get_methods(cd)) == 1


# =============================================================================
# 5. DOCSTRING FIDELITY
# =============================================================================


class TestDocstringFidelity:
    """Docstrings on nodes are preserved in the AST."""

    def test_component_docstring(self):
        """Component docstring appears as class docstring."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="C",
                             data=ComponentData(
                                 docstring="A component docstring.",
                             ))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "C")
        assert cd is not None
        assert _get_docstring(cd) == "A component docstring."

    def test_resource_docstring(self):
        """Resource docstring appears as class docstring."""
        graph = NodeGraph(
            nodes=[GraphNode(id="r1", type="resource", name="R",
                             data=ResourceData(
                                 docstring="A resource docstring.",
                             ))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "R")
        assert cd is not None
        assert _get_docstring(cd) == "A resource docstring."

    def test_event_docstring(self):
        """Event docstring appears as class docstring."""
        graph = NodeGraph(
            nodes=[GraphNode(id="e1", type="event", name="E",
                             data=EventData(
                                 docstring="An event docstring.",
                             ))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "E")
        assert cd is not None
        assert _get_docstring(cd) == "An event docstring."


# =============================================================================
# 6. BASE CLASS FIDELITY
# =============================================================================


class TestBaseClassFidelity:
    """Base classes declared in the scene graph are reflected in the AST."""

    def test_single_base(self):
        """A single base class is reflected."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="C",
                             data=ComponentData(bases=["BaseClass"]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "C")
        assert cd is not None
        assert "BaseClass" in _get_bases(cd)

    def test_multiple_bases(self):
        """Multiple base classes are all reflected."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="C",
                             data=ComponentData(
                                 bases=["MixinA", "MixinB", "MixinC"],
                             ))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "C")
        assert cd is not None
        bases = _get_bases(cd)
        assert "MixinA" in bases
        assert "MixinB" in bases
        assert "MixinC" in bases

    def test_no_bases(self):
        """Empty bases list means no bases."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="C",
                             data=ComponentData(bases=[]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "C")
        assert cd is not None
        assert len(_get_bases(cd)) == 0

    def test_system_with_bases(self):
        """System nodes also support base classes."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="Sys",
                             data=SystemData(bases=["BaseSystem"]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "Sys")
        assert cd is not None
        assert "BaseSystem" in _get_bases(cd)


# =============================================================================
# 7. DECORATOR ARGUMENT FIDELITY
# =============================================================================


class TestDecoratorArgFidelity:
    """Decorator arguments (keyword/positional) are preserved."""

    def test_keyword_args_produce_call_decorator(self):
        """@component(name="test") produces a Call node."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="C",
                             data=ComponentData(decorator_args={
                                 "keyword": {"name": "test"},
                             }))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "C")
        assert cd is not None
        dec = cd.decorator_list[0]
        assert isinstance(dec, ast.Call)
        assert isinstance(dec.func, ast.Name)
        assert dec.func.id == "component"
        assert len(dec.keywords) == 1
        assert dec.keywords[0].arg == "name"

    def test_positional_args(self):
        """Positional decorator args are preserved."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="C",
                             data=ComponentData(decorator_args={
                                 "positional": ["arg1", "arg2"],
                             }))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "C")
        assert cd is not None
        dec = cd.decorator_list[0]
        assert isinstance(dec, ast.Call)
        assert len(dec.args) == 2

    def test_no_args_produces_simple_name(self):
        """When decorator_args is empty or None, decorator is a plain Name."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="C",
                             data=ComponentData())],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "C")
        assert cd is not None
        dec = cd.decorator_list[0]
        assert isinstance(dec, ast.Name)
        assert dec.id == "component"


# =============================================================================
# 8. EVENT PAYLOAD FIDELITY
# =============================================================================


class TestEventPayloadFidelity:
    """Event payload fields are correctly generated."""

    def test_eventdata_fields(self):
        """EventData fields produce annotated assignments."""
        graph = NodeGraph(
            nodes=[GraphNode(id="e1", type="event", name="Collision",
                             data=EventData(fields=[
                                 FieldData(name="entity_a",
                                           type_annotation="str"),
                                 FieldData(name="entity_b",
                                           type_annotation="str"),
                             ]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "Collision")
        assert cd is not None
        fields = _get_fields(cd)
        assert len(fields) == 2

    def test_raw_dict_payload_fields(self):
        """Raw dict with payload_fields produces correct fields."""
        graph = NodeGraph(
            nodes=[GraphNode(id="e1", type="event", name="E",
                             data={
                                 "payload_fields": [
                                     {"name": "msg",
                                      "type_annotation": "str",
                                      "default_value": "\"hello\""},
                                 ],
                             })],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "E")
        assert cd is not None
        fields = _get_fields(cd)
        assert ("msg", "str") in fields

    def test_event_without_fields_has_pass(self):
        """An event with no fields generates a `pass` body."""
        graph = NodeGraph(
            nodes=[GraphNode(id="e1", type="event", name="E",
                             data=EventData())],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "E")
        assert cd is not None
        assert any(isinstance(st, ast.Pass) for st in cd.body)


# =============================================================================
# 9. NODE ORDERING
# =============================================================================


class TestNodeOrdering:
    """Class definitions appear in a consistent type-then-alpha order."""

    def test_type_order(self):
        """Order: component, resource, event, system."""
        nodes = [
            GraphNode(id="s1", type="system", name="ASystem",
                      data=SystemData()),
            GraphNode(id="c1", type="component", name="AComponent",
                      data=ComponentData()),
            GraphNode(id="e1", type="event", name="AnEvent",
                      data=EventData()),
            GraphNode(id="r1", type="resource", name="AResource",
                      data=ResourceData()),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        classes = [n for n in module.body if isinstance(n, ast.ClassDef)]
        types = []
        for cls in classes:
            ids = _get_decorator_ids(cls)
            types.append(ids[0] if ids else "?")
        assert types == ["component", "resource", "event", "system"]

    def test_alphabetical_within_type(self):
        """Within a type, classes are sorted alphabetically."""
        nodes = [
            GraphNode(id="c2", type="component", name="Zebra",
                      data=ComponentData()),
            GraphNode(id="c1", type="component", name="Alpha",
                      data=ComponentData()),
            GraphNode(id="c3", type="component", name="Beta",
                      data=ComponentData()),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        classes = [n for n in module.body if isinstance(n, ast.ClassDef)]
        names = [c.name for c in classes]
        assert names == ["Alpha", "Beta", "Zebra"]


# =============================================================================
# 10. ROUND-TRIP VALIDITY
# =============================================================================


class TestRoundTrip:
    """Output AST must be valid Python that survives unparse + reparse."""

    def test_full_graph_unparse_reparse(self):
        """A representative graph round-trips through unparse/parse cleanly."""
        nodes = [
            GraphNode(id="c1", type="component", name="Position",
                      data=ComponentData(fields=[
                          FieldData(name="x", type_annotation="float",
                                    default_value="0.0"),
                          FieldData(name="y", type_annotation="float",
                                    default_value="0.0"),
                      ])),
            GraphNode(id="s1", type="system", name="Physics",
                      data=SystemData(methods=[
                          MethodData(name="update",
                                     parameters=[
                                         ParameterData(
                                             name="dt",
                                             type_annotation="float"),
                                     ]),
                      ])),
            GraphNode(id="e1", type="event", name="Collision",
                      data=EventData(fields=[
                          FieldData(name="entity", type_annotation="str"),
                      ])),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        source = ast.unparse(module)
        # Must not raise
        ast.parse(source)

    def test_edge_cases_round_trip(self):
        """Boundary values (bools, None, empty lists) produce valid Python."""
        nodes = [
            GraphNode(id="c1", type="component", name="Edge",
                      data=ComponentData(fields=[
                          FieldData(name="a", type_annotation="bool",
                                    default_value="True"),
                          FieldData(name="b", type_annotation="Optional[int]",
                                    default_value="None"),
                          FieldData(name="c", type_annotation="list[str]",
                                    default_value="[]"),
                          FieldData(name="d", type_annotation="dict[str, int]",
                                    default_value="{}"),
                      ])),
            GraphNode(id="s1", type="system", name="Proc",
                      data=SystemData(methods=[
                          MethodData(name="run",
                                     return_type="None"),
                      ])),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        source = ast.unparse(module)
        ast.parse(source)


# =============================================================================
# 11. DICT INPUT
# =============================================================================


class TestDictInput:
    """graph_to_ast accepts raw dicts (as from JSON/frontend)."""

    def test_dict_component(self):
        """A component described as a plain dict is accepted."""
        graph = {
            "nodes": [
                {
                    "id": "c1",
                    "type": "component",
                    "name": "Position",
                    "data": {
                        "fields": [
                            {"name": "x", "type_annotation": "int"},
                        ],
                    },
                },
            ],
            "edges": [],
        }
        module = graph_to_ast(graph)
        cd = _find_class(module, "Position")
        assert cd is not None
        assert "component" in _get_decorator_ids(cd)

    def test_dict_all_types(self):
        """All four node types from a dict work correctly."""
        graph = {
            "nodes": [
                {"id": "c1", "type": "component", "name": "C",
                 "data": {}},
                {"id": "s1", "type": "system", "name": "S",
                 "data": {"methods": []}},
                {"id": "r1", "type": "resource", "name": "R",
                 "data": {}},
                {"id": "e1", "type": "event", "name": "E",
                 "data": {}},
            ],
            "edges": [],
        }
        module = graph_to_ast(graph)
        classes = [(c.name, _get_decorator_ids(c)[0])
                   for c in module.body if isinstance(c, ast.ClassDef)]
        assert ("C", "component") in classes
        assert ("S", "system") in classes
        assert ("R", "resource") in classes
        assert ("E", "event") in classes

    def test_dict_missing_nodes_key(self):
        """Missing 'nodes' key results in empty graph (no crash)."""
        graph = {"edges": []}
        module = graph_to_ast(graph)
        classes = [n for n in module.body if isinstance(n, ast.ClassDef)]
        assert len(classes) == 0


# =============================================================================
# 12. NODES_TO_AST
# =============================================================================


class TestNodesToAst:
    """nodes_to_ast convenience function."""

    def test_single_node(self):
        """Single node produces one class."""
        nodes = [
            GraphNode(id="c1", type="component", name="Pos",
                      data=ComponentData()),
        ]
        module = nodes_to_ast(nodes)
        classes = [n for n in module.body if isinstance(n, ast.ClassDef)]
        assert len(classes) == 1
        assert classes[0].name == "Pos"

    def test_multiple_nodes(self):
        """Multiple nodes all produce classes."""
        nodes = [
            GraphNode(id="c1", type="component", name="A",
                      data=ComponentData()),
            GraphNode(id="c2", type="component", name="B",
                      data=ComponentData()),
            GraphNode(id="c3", type="component", name="C",
                      data=ComponentData()),
        ]
        module = nodes_to_ast(nodes)
        classes = [n for n in module.body if isinstance(n, ast.ClassDef)]
        assert len(classes) == 3

    def test_empty_list(self):
        """Empty list produces an empty module."""
        module = nodes_to_ast([])
        classes = [n for n in module.body if isinstance(n, ast.ClassDef)]
        assert len(classes) == 0


# =============================================================================
# 13. NEGATIVE / EDGE CASES
# =============================================================================


class TestNegativeCases:
    """Resilience to unusual or malformed inputs."""

    def test_missing_data_reference(self):
        """Node with data=None is not a valid input per the type schema;
        the call should still not crash catastrophically."""
        graph = NodeGraph(
            nodes=[GraphNode(id="x1", type="component", name="Bad",
                             data=None)],
            edges=[],
        )
        with pytest.raises((AttributeError, TypeError)):
            graph_to_ast(graph)

    def test_empty_methods_list(self):
        """System with methods=[] still produces a valid class."""
        graph = NodeGraph(
            nodes=[GraphNode(id="s1", type="system", name="S",
                             data=SystemData(methods=[]))],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "S")
        assert cd is not None
        assert "system" in _get_decorator_ids(cd)
        # With no fields and no methods, should produce pass
        assert any(isinstance(st, ast.Pass) for st in cd.body)

    def test_unicode_class_name(self):
        """Unicode class names pass through."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="DataZoNe",
                             data=ComponentData())],
            edges=[],
        )
        module = graph_to_ast(graph)
        cd = _find_class(module, "DataZoNe")
        assert cd is not None

    def test_field_variations(self):
        """Fields with alternate key names (type instead of type_annotation,
        default instead of default_value) work correctly."""
        graph = {
            "nodes": [
                {
                    "id": "c1",
                    "type": "component",
                    "name": "Varied",
                    "data": {
                        "fields": [
                            {"name": "a", "type": "int",
                             "default": "0"},
                            {"name": "b", "type_annotation": "str",
                             "default_value": "\"hi\""},
                            {"name": "c", "type": "float"},
                        ],
                    },
                },
            ],
            "edges": [],
        }
        module = graph_to_ast(graph)
        cd = _find_class(module, "Varied")
        assert cd is not None
        fields = _get_fields(cd)
        assert len(fields) == 3

    def test_type_annotation_fallback_to_name(self):
        """When a type annotation cannot be parsed, it falls back by creating an
        ast.Name node with the raw string. The call must not raise."""
        graph = NodeGraph(
            nodes=[GraphNode(id="c1", type="component", name="Fuzzy",
                             data=ComponentData(fields=[
                                 FieldData(name="data",
                                           type_annotation="invalid[syntax"),
                             ]))],
            edges=[],
        )
        # Must not raise -- fallback creates a Name node
        module = graph_to_ast(graph)
        cd = _find_class(module, "Fuzzy")
        assert cd is not None
        fields = _get_fields(cd)
        assert len(fields) == 1
        field_name, field_type = fields[0]
        assert field_name == "data"
        # The fallback Name node retains the raw string (may not produce valid
        # Python when unparsed -- that is an inherent limitation of the fallback)
        assert "invalid[syntax" in field_type


# =============================================================================
# 14. FIDELITY METRICS: QUANTITATIVE PROPERTY CHECKS
# =============================================================================


class TestFidelityMetrics:
    """Quantitative properties that must hold for ANY valid scene graph."""

    def test_class_count_equals_node_count(self):
        """Number of ClassDefs equals number of non-unknown nodes."""
        nodes = [
            GraphNode(id=str(i), type="component", name=f"C{i}",
                      data=ComponentData())
            for i in range(10)
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        class_count = len([n for n in module.body if isinstance(n, ast.ClassDef)])
        assert class_count == len(nodes)

    def test_field_count_preserved(self):
        """Total field count across all classes equals total declared fields."""
        nodes = [
            GraphNode(id="c1", type="component", name="A",
                      data=ComponentData(fields=[
                          FieldData(name="x", type_annotation="int"),
                          FieldData(name="y", type_annotation="int"),
                      ])),
            GraphNode(id="c2", type="component", name="B",
                      data=ComponentData(fields=[
                          FieldData(name="z", type_annotation="str"),
                      ])),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        total_fields = sum(
            len(_get_fields(c))
            for c in module.body if isinstance(c, ast.ClassDef)
        )
        assert total_fields == 3

    def test_each_class_has_exactly_one_trinity_decorator(self):
        """Every class has exactly one Trinity decorator (component/system/resource/event)."""
        trinity_decorators = {"component", "system", "resource", "event"}
        nodes = [
            GraphNode(id="c1", type="component", name="C",
                      data=ComponentData()),
            GraphNode(id="s1", type="system", name="S",
                      data=SystemData()),
            GraphNode(id="r1", type="resource", name="R",
                      data=ResourceData()),
            GraphNode(id="e1", type="event", name="E",
                      data=EventData()),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        for cls in module.body:
            if isinstance(cls, ast.ClassDef):
                decorator_ids = set(_get_decorator_ids(cls))
                matches = decorator_ids & trinity_decorators
                assert len(matches) == 1, (
                    f"{cls.name} has {len(matches)} trinity decorators"
                )

    def test_method_count_preserved(self):
        """Total method count across all systems matches declared methods."""
        nodes = [
            GraphNode(id="s1", type="system", name="S1",
                      data=SystemData(methods=[
                          MethodData(name="a"),
                          MethodData(name="b"),
                      ])),
            GraphNode(id="s2", type="system", name="S2",
                      data=SystemData(methods=[
                          MethodData(name="c"),
                      ])),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        total_methods = sum(
            len(_get_methods(c))
            for c in module.body if isinstance(c, ast.ClassDef)
        )
        assert total_methods == 3

    def test_parameter_count_preserved(self):
        """Total parameter count across all methods matches declared parameters."""
        nodes = [
            GraphNode(id="s1", type="system", name="S",
                      data=SystemData(methods=[
                          MethodData(name="run", parameters=[
                              ParameterData(name="a",
                                             type_annotation="int"),
                              ParameterData(name="b",
                                             type_annotation="str"),
                          ]),
                          MethodData(name="stop", parameters=[
                              ParameterData(name="reason",
                                             type_annotation="str"),
                          ]),
                      ])),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        cd = _find_class(module, "S")
        assert cd is not None
        methods = _get_methods(cd)
        total_params = sum(
            # Subtract 1 for self
            len(m.args.args) - 1
            for m in methods
        )
        assert total_params == 3


# =============================================================================
# 15. COMPOSITE SCENE GRAPHS: REAL-WORLD TEMPLATES
# =============================================================================


class TestCompositeSceneGraphs:
    """Multi-node scene graphs representing real ECS patterns."""

    def test_ecs_template(self):
        """A minimal ECS: Position + Velocity components, MovementSystem."""
        nodes = [
            GraphNode(id="c1", type="component", name="Position",
                      data=ComponentData(
                          docstring="2D position component.",
                          fields=[
                              FieldData(name="x", type_annotation="float",
                                        default_value="0.0"),
                              FieldData(name="y", type_annotation="float",
                                        default_value="0.0"),
                          ],
                      )),
            GraphNode(id="c2", type="component", name="Velocity",
                      data=ComponentData(
                          docstring="2D velocity component.",
                          fields=[
                              FieldData(name="dx", type_annotation="float",
                                        default_value="0.0"),
                              FieldData(name="dy", type_annotation="float",
                                        default_value="0.0"),
                          ],
                      )),
            GraphNode(id="s1", type="system", name="MovementSystem",
                      data=SystemData(
                          docstring="Moves entities by velocity each tick.",
                          fields=[
                              FieldData(name="speed_multiplier",
                                        type_annotation="float",
                                        default_value="1.0"),
                          ],
                          methods=[
                              MethodData(
                                  name="update",
                                  parameters=[
                                      ParameterData(
                                          name="dt",
                                          type_annotation="float"),
                                  ],
                                  query_components=["Position", "Velocity"],
                              ),
                          ],
                      )),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        source = ast.unparse(module)
        # Must be valid Python
        ast.parse(source)

        pos = _find_class(module, "Position")
        vel = _find_class(module, "Velocity")
        mov = _find_class(module, "MovementSystem")
        assert pos is not None
        assert vel is not None
        assert mov is not None
        assert _get_docstring(pos) == "2D position component."
        assert len(_get_fields(pos)) == 2
        assert len(_get_fields(vel)) == 2
        assert len(_get_fields(mov)) == 1
        assert len(_get_methods(mov)) == 1

    def test_game_loop_template(self):
        """Config resource + Clock system + Collision event."""
        nodes = [
            GraphNode(id="r1", type="resource", name="GameConfig",
                      data=ResourceData(
                          docstring="Global game configuration.",
                          fields=[
                              FieldData(name="width",
                                        type_annotation="int",
                                        default_value="800"),
                              FieldData(name="height",
                                        type_annotation="int",
                                        default_value="600"),
                              FieldData(name="title",
                                        type_annotation="str",
                                        default_value='"My Game"'),
                          ],
                      )),
            GraphNode(id="s1", type="system", name="ClockSystem",
                      data=SystemData(
                          docstring="Manages the game tick.",
                          methods=[
                              MethodData(
                                  name="tick",
                                  parameters=[
                                      ParameterData(
                                          name="delta",
                                          type_annotation="float"),
                                  ],
                                  return_type="float",
                              ),
                          ],
                      )),
            GraphNode(id="e1", type="event", name="CollisionEvent",
                      data=EventData(
                          docstring="Fired when two entities collide.",
                          fields=[
                              FieldData(name="entity_a",
                                        type_annotation="str"),
                              FieldData(name="entity_b",
                                        type_annotation="str"),
                              FieldData(name="position",
                                        type_annotation="tuple[float, float]"),
                          ],
                      )),
        ]
        graph = NodeGraph(nodes=nodes, edges=[])
        module = graph_to_ast(graph)
        source = ast.unparse(module)
        ast.parse(source)

        cfg = _find_class(module, "GameConfig")
        clk = _find_class(module, "ClockSystem")
        col = _find_class(module, "CollisionEvent")
        assert cfg is not None
        assert clk is not None
        assert col is not None
        assert _get_docstring(cfg) == "Global game configuration."
        assert _get_docstring(clk) == "Manages the game tick."
        assert _get_docstring(col) == "Fired when two entities collide."
        assert len(_get_fields(cfg)) == 3
        assert len(_get_methods(clk)) == 1
        assert len(_get_fields(col)) == 3
        # Verify ordering is correct: resource, event, system
        classes = [n for n in module.body if isinstance(n, ast.ClassDef)]
        dec_order = [_get_decorator_ids(c)[0] for c in classes]
        assert dec_order == ["resource", "event", "system"]
