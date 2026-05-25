"""Whitebox tests for AST Builder internal helpers and edge cases (118 tests)."""
from __future__ import annotations
import ast
import pytest
from engine.rendering.demoscene.ast_nodes import (
    Axis, BendNode, BoxNode, CellIdNode, CombineNode, CompensationNode,
    ExprNode, FloatNode, IntersectionNode, KifsNode, Kind, MirrorNode,
    PositionNode, RepeatNode, SceneGraph, SdfPrimitiveNode, SphereNode,
    StretchNode, SubtractionNode, TorusNode, TwistNode, UnionNode, Vec3Node,
)
from engine.rendering.demoscene.ast_builder import (
    AstBuilder, _COMPOSITION_DISPATCH, _COMPOSITE_DISPATCH,
    _MARKER_DISPATCH, _PRIMITIVE_DISPATCH, _ast_arg_to_node,
    _build_ast_from_call, _disassemble_lambda, _to_axis, _to_float,
    _to_vec3, build_from_composition, walk_composition,
)


# =============================================================================
# Internal: _to_float (6 tests)
# =============================================================================
class Test_to_float:
    def test_passthrough(self):
        fn = FloatNode(3.14)
        assert _to_float(fn) is fn
    def test_from_int(self):
        result = _to_float(42)
        assert isinstance(result, FloatNode)
        assert result.value == 42.0
    def test_from_float(self):
        result = _to_float(2.5)
        assert isinstance(result, FloatNode)
        assert result.value == 2.5
    def test_from_negative(self):
        result = _to_float(-1.5)
        assert result.value == -1.5
    def test_from_zero(self):
        result = _to_float(0)
        assert result.value == 0.0
    def test_from_large(self):
        result = _to_float(1e10)
        assert result.value == 1e10


# =============================================================================
# Internal: _to_vec3 (10 tests)
# =============================================================================
class Test_to_vec3:
    def test_passthrough(self):
        v = Vec3Node(1.0, 2.0, 3.0)
        assert _to_vec3(v) is v
    def test_from_tuple(self):
        result = _to_vec3((1.0, 2.0, 3.0))
        assert isinstance(result, Vec3Node)
        assert result.as_tuple() == (1.0, 2.0, 3.0)
    def test_from_list(self):
        result = _to_vec3([4.0, 5.0, 6.0])
        assert isinstance(result, Vec3Node)
        assert result.as_tuple() == (4.0, 5.0, 6.0)
    def test_from_int_tuple(self):
        result = _to_vec3((1, 2, 3))
        assert result.as_tuple() == (1.0, 2.0, 3.0)
    def test_from_negative(self):
        result = _to_vec3((-1.0, -2.0, -3.0))
        assert result.as_tuple() == (-1.0, -2.0, -3.0)
    def test_from_zero(self):
        result = _to_vec3((0.0, 0.0, 0.0))
        assert result.as_tuple() == (0.0, 0.0, 0.0)
    def test_error_wrong_length(self):
        with pytest.raises(TypeError):
            _to_vec3((1.0, 2.0))
    def test_error_string(self):
        with pytest.raises(TypeError):
            _to_vec3("abc")
    def test_error_list_wrong_length(self):
        with pytest.raises(TypeError):
            _to_vec3([1.0, 2.0])
    def test_error_none(self):
        with pytest.raises(TypeError):
            _to_vec3(None)


# =============================================================================
# Internal: _to_axis (7 tests)
# =============================================================================
class Test_to_axis:
    def test_passthrough(self):
        assert _to_axis(Axis.X) is Axis.X
    def test_from_lower_x(self):
        assert _to_axis("x") == Axis.X
    def test_from_upper_y(self):
        assert _to_axis("Y") == Axis.Y
    def test_from_lower_z(self):
        assert _to_axis("z") == Axis.Z
    def test_error_invalid_string(self):
        with pytest.raises(ValueError):
            _to_axis("w")
    def test_error_int(self):
        with pytest.raises(TypeError):
            _to_axis(0)
    def test_error_none(self):
        with pytest.raises(TypeError):
            _to_axis(None)


# =============================================================================
# Internal: _ast_arg_to_node (12 tests)
# =============================================================================
class Test_ast_arg_to_node:
    def test_float_constant(self):
        node = _ast_arg_to_node(ast.Constant(value=3.14))
        assert isinstance(node, FloatNode)
        assert node.value == 3.14
    def test_int_constant(self):
        node = _ast_arg_to_node(ast.Constant(value=42))
        assert isinstance(node, FloatNode)
        assert node.value == 42.0
    def test_string_constant(self):
        node = _ast_arg_to_node(ast.Constant(value="hello"))
        assert node == "hello"
    def test_negative(self):
        node = _ast_arg_to_node(ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=5)))
        assert isinstance(node, FloatNode)
        assert node.value == -5.0
    def test_double_negative(self):
        inner = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=3.0))
        node = _ast_arg_to_node(ast.UnaryOp(op=ast.USub(), operand=inner))
        assert isinstance(node, FloatNode)
        assert node.value == 3.0
    def test_tuple_all_float(self):
        tup = ast.Tuple(elts=[ast.Constant(value=1.0), ast.Constant(value=2.0), ast.Constant(value=3.0)])
        node = _ast_arg_to_node(tup)
        assert isinstance(node, Vec3Node)
        assert node.as_tuple() == (1.0, 2.0, 3.0)
    def test_tuple_mixed(self):
        tup = ast.Tuple(elts=[ast.Constant(value="a"), ast.Constant(value=1)])
        node = _ast_arg_to_node(tup)
        assert isinstance(node, tuple)
        assert len(node) == 2
    def test_list(self):
        lst = ast.List(elts=[ast.Constant(value=1), ast.Constant(value=2)])
        node = _ast_arg_to_node(lst)
        assert isinstance(node, list)
        assert len(node) == 2
    def test_name_p(self):
        node = _ast_arg_to_node(ast.Name(id="p"))
        assert isinstance(node, PositionNode)
    def test_name_not_p(self):
        node = _ast_arg_to_node(ast.Name(id="x"))
        assert node is None
    def test_bool_constant(self):
        node = _ast_arg_to_node(ast.Constant(value=True))
        assert isinstance(node, FloatNode)
        assert node.value == 1.0
    def test_none_constant(self):
        node = _ast_arg_to_node(ast.Constant(value=None))
        assert node is None


# =============================================================================
# Internal: _disassemble_lambda (2 tests)
# =============================================================================
class Test_disassemble_lambda:
    def test_non_callable(self):
        with pytest.raises(ValueError):
            _disassemble_lambda("not_a_function")
    def test_lambda_no_call_body(self):
        fn = lambda p: p
        result = _disassemble_lambda(fn)
        assert result is None


# =============================================================================
# Internal: _build_ast_from_call (5 tests)
# =============================================================================
class Test_build_ast_from_call:
    def test_unknown_function(self):
        call = ast.Call(func=ast.Name(id="unknown_func"), args=[], keywords=[])
        assert _build_ast_from_call(call) is None
    def test_attribute_call(self):
        call = ast.Call(func=ast.Attribute(value=ast.Name(id="foo"), attr="sdSphere"), args=[ast.Name(id="p"), ast.Constant(value=1.0)], keywords=[])
        result = _build_ast_from_call(call)
        assert isinstance(result, SphereNode)
    def test_keyword_args(self):
        call = ast.Call(func=ast.Name(id="sdSphere"), args=[ast.Name(id="p")], keywords=[ast.keyword(arg="r", value=ast.Constant(value=2.0))])
        result = _build_ast_from_call(call)
        assert isinstance(result, SphereNode)
        assert result.radius.value == 2.0
    def test_nested_keyword(self):
        call = ast.Call(func=ast.Name(id="domain_repeat"), args=[ast.Name(id="p")], keywords=[ast.keyword(arg="cell_size", value=ast.Constant(value=(4.0, 4.0, 4.0)))])
        result = _build_ast_from_call(call)
        assert isinstance(result, RepeatNode)
    def test_no_args_call_fails(self):
        call = ast.Call(func=ast.Name(id="sdSphere"), args=[], keywords=[])
        with pytest.raises(TypeError):
            _build_ast_from_call(call)


# =============================================================================
# AstBuilder.walk internals (4 tests)
# =============================================================================
class TestAstBuilderWalk:
    def test_expr_node_passthrough(self):
        n = FloatNode(1.0)
        assert AstBuilder.walk(n) is n
    def test_position_node_passthrough(self):
        p = PositionNode()
        assert AstBuilder.walk(p) is p
    def test_unknown_type(self):
        result = AstBuilder.walk("hello")
        assert result == {}
    def test_callable_delegation(self):
        result = AstBuilder.walk(lambda p: sdSphere(p, 1.0))
        assert isinstance(result, SceneGraph)


# =============================================================================
# AstBuilder._walk_dict internals (4 tests)
# =============================================================================
class TestAstBuilderWalkDict:
    def test_nested_dict_marker(self):
        d = {"type": "repeat", "cell_size": (3.0, 3.0, 3.0), "input": {"type": "sphere", "radius": 1.0}}
        result = AstBuilder._walk_dict(d)
        assert isinstance(result, RepeatNode)
    def test_non_marker_with_pipeline(self):
        d = {"pipeline": [{"type": "repeat", "cell_size": (2.0, 2.0, 2.0)}], "primitives": [{"type": "sphere", "radius": 1.0}]}
        result = AstBuilder._walk_dict(d)
        assert isinstance(result, SceneGraph)
    def test_marker_with_tuple_value(self):
        d = {"type": "box", "size": (2.0, 2.0, 2.0)}
        result = AstBuilder._walk_dict(d)
        assert isinstance(result, BoxNode)
    def test_unknown_type_no_pipeline(self):
        result = AstBuilder._walk_dict({"type": "unknown", "foo": 1})
        assert result == {"foo": {}}


# =============================================================================
# AstBuilder._walk_dsl_object internals (5 tests)
# =============================================================================
class TestAstBuilderWalkDslObject:
    def test_object_with_node_type(self):
        class Obj:
            _node_type = "sphere"
            radius = 2.0
        result = AstBuilder._walk_dsl_object(Obj())
        assert isinstance(result, SphereNode)
    def test_unknown_node_type(self):
        class Obj:
            _node_type = "unknown"
        result = AstBuilder._walk_dsl_object(Obj())
        assert result == {}
    def test_without_node_type(self):
        class Obj:
            def __init__(self):
                self.x = 1
                self.y = 2
        result = AstBuilder._walk_dsl_object(Obj())
        assert result == {"x": 1, "y": 2}
    def test_slots_object(self):
        class Obj:
            __slots__ = ("x", "y")
            def __init__(self):
                self.x = 1
                self.y = 2
        result = AstBuilder._walk_dsl_object(Obj())
        assert result == {}
    def test_partial_attrs(self):
        class Obj:
            _node_type = "mirror"
            axis = "y"
        result = AstBuilder._walk_dsl_object(Obj())
        assert isinstance(result, MirrorNode)


# =============================================================================
# AstBuilder._build_scene internals (4 tests)
# =============================================================================
class TestAstBuilderBuildScene:
    def test_non_domain_op_filtered(self):
        d = {"primitives": [{"type": "sphere", "radius": 1.0}]}
        sg = AstBuilder._build_scene(d)
        assert len(sg.primitives) == 1
    def test_non_sdf_filtered(self):
        d = {"pipeline": [{"type": "repeat", "cell_size": (2.0, 2.0, 2.0)}]}
        sg = AstBuilder._build_scene(d)
        assert len(sg.pipeline) == 1
    def test_empty_pipeline_prims(self):
        sg = AstBuilder._build_scene({})
        assert len(sg.pipeline) == 0
        assert len(sg.primitives) == 0
    def test_name_preserved(self):
        d = {"pipeline": [], "primitives": [], "name": "mytest"}
        sg = AstBuilder._build_scene(d)
        assert sg.name == "mytest"


# =============================================================================
# walk_composition edge cases (5 tests)
# =============================================================================
class TestWalkCompositionEdgeCases:
    def test_external_sdf_primitive(self):
        sg = walk_composition(lambda p: domain_repeat(p, (2.0, 2.0, 2.0)),
                              primitives=[SphereNode(PositionNode(), FloatNode(1.0))])
        assert len(sg.primitives) == 1
    def test_no_primitives_fallback(self):
        sg = walk_composition(lambda p: domain_repeat(p, (2.0, 2.0, 2.0)),
                              primitives=[{"type": "sphere", "radius": 1.0}])
        assert len(sg.primitives) == 1
    def test_lambda_name(self):
        fn = lambda p: sdSphere(p, 1.0)
        sg = walk_composition(fn)
        assert sg.name.startswith("<lambda>")
    def test_build_from_composition_alias(self):
        assert build_from_composition is walk_composition
    def test_domain_op_in_position(self):
        sg = walk_composition(lambda p: sdSphere(domain_twist(p, 0.5), 1.0))
        assert len(sg.pipeline) == 1
        assert isinstance(sg.pipeline[0], TwistNode)


# =============================================================================
# CSG internals (5 tests)
# =============================================================================
class TestCSGInternals:
    def test_union_frozen_bypass(self):
        u = UnionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                      BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0)))
        assert u.kind == "union"
        assert u.left is not None
        assert u.right is not None
    def test_intersection_frozen_bypass(self):
        i = IntersectionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                             BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0)))
        assert i.kind == "intersection"
    def test_subtraction_frozen_bypass(self):
        s = SubtractionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                            BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0)))
        assert s.kind == "subtraction"
    def test_independent_instances(self):
        s = SphereNode(PositionNode(), FloatNode(1.0))
        b = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        u1 = UnionNode(s, b)
        u2 = UnionNode(s, b)
        assert u1 is not u2
    def test_csg_label_formats(self):
        u = UnionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                      BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)))
        assert u.label() == "Union(...)"
        i = IntersectionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                             BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)))
        assert i.label() == "Intersection(...)"
        s = SubtractionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)))
        assert s.label() == "Subtraction(...)"


# =============================================================================
# DomainOp variants (8 tests)
# =============================================================================
class TestDomainOpVariants:
    def test_cell_id_node(self):
        c = CellIdNode(PositionNode(), Vec3Node(4.0, 4.0, 1.0))
        assert c.cell_size.as_tuple() == (4.0, 4.0, 1.0)
    def test_cell_id_label(self):
        c = CellIdNode(PositionNode(), Vec3Node(4.0, 4.0, 1.0))
        assert "CellId" in c.label()
    def test_mirror_z(self):
        m = MirrorNode(PositionNode(), Axis.Z)
        assert m.axis == Axis.Z
    def test_mirror_x_label(self):
        m = MirrorNode(PositionNode(), Axis.X)
        assert "X" in m.label()
    def test_stretch_x(self):
        s = StretchNode(PositionNode(), FloatNode(2.0), Axis.X)
        assert s.stretch.value == 2.0
        assert s.axis == Axis.X
    def test_stretch_y(self):
        s = StretchNode(PositionNode(), FloatNode(3.0), Axis.Y)
        assert s.axis == Axis.Y
    def test_stretch_z(self):
        s = StretchNode(PositionNode(), FloatNode(4.0), Axis.Z)
        assert s.axis == Axis.Z
    def test_domain_op_children(self):
        r = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        children = list(r.children())
        assert len(children) == 1
        assert isinstance(children[0], PositionNode)


# =============================================================================
# CompensationNode (7 tests)
# =============================================================================
class TestCompensationNode:
    def test_repeat_kind(self):
        c = CompensationNode(Kind.REPEAT, 1.0)
        assert c.kind == Kind.REPEAT
    def test_kifs_kind(self):
        c = CompensationNode(Kind.KIFS, 2.0)
        assert c.kind == Kind.KIFS
    def test_stretch_kind(self):
        c = CompensationNode(Kind.STRETCH, 3.0)
        assert c.kind == Kind.STRETCH
    def test_twist_kind(self):
        c = CompensationNode(Kind.TWIST, 4.0)
        assert c.kind == Kind.TWIST
    def test_label_all_kinds(self):
        for k in Kind:
            c = CompensationNode(k, 1.5)
            assert k.value in c.label()
    def test_zero_param(self):
        c = CompensationNode(Kind.REPEAT, 0.0)
        assert c.param == 0.0
    def test_negative_param(self):
        c = CompensationNode(Kind.STRETCH, -1.0)
        assert c.param == -1.0


# =============================================================================
# SceneGraph labels (5 tests)
# =============================================================================
class TestSceneGraphLabels:
    def test_unnamed_label(self):
        sg = SceneGraph(primitives=(SphereNode(PositionNode(), FloatNode(1.0)),))
        assert "(unnamed)" not in sg.label()
    def test_named_label(self):
        sg = SceneGraph(primitives=(), name="myscene")
        assert "'myscene'" in sg.label()
    def test_deep_label_no_pipeline(self):
        sg = SceneGraph(primitives=(SphereNode(PositionNode(), FloatNode(1.0)),), name="test")
        dl = sg.deep_label()
        assert "Pipeline:" not in dl
        assert "Primitives:" in dl
    def test_deep_label_unnamed(self):
        sg = SceneGraph(primitives=(SphereNode(PositionNode(), FloatNode(1.0)),))
        dl = sg.deep_label()
        assert "(unnamed)" in dl
    def test_deep_label_empty_primitives(self):
        sg = SceneGraph(primitives=())
        dl = sg.deep_label()
        assert "Primitives:" in dl


# =============================================================================
# ExprNode base class (8 tests)
# =============================================================================
class TestExprNodeBase:
    def test_label_fallback(self):
        n = ExprNode()
        assert n.label() == "ExprNode"
    def test_children_default(self):
        n = ExprNode()
        assert list(n.children()) == []
    def test_pretty_no_indent(self):
        n = FloatNode(1.0)
        assert n.pretty() == "Float(1.0)"
    def test_pretty_with_indent(self):
        n = FloatNode(1.0)
        assert n.pretty(indent=2) == "    Float(1.0)"
    def test_walk_depth(self):
        n = FloatNode(1.0)
        nodes = list(n.walk())
        assert len(nodes) == 1
        assert nodes[0] == (n, 0)
    def test_walk_nested_depth(self):
        r = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        nodes = list(r.walk())
        assert len(nodes) == 2
        assert nodes[0][1] == 0
        assert nodes[1][1] == 1
    def test_combine_node_label(self):
        c = CombineNode("test", FloatNode(1.0), FloatNode(2.0))
        assert c.label() == "Combine(test)"
    def test_sdf_primitive_children(self):
        s = SphereNode(PositionNode(), FloatNode(1.0))
        children = list(s.children())
        assert len(children) == 1
        assert isinstance(children[0], PositionNode)


# =============================================================================
# Node enums (4 tests)
# =============================================================================
class TestNodeEnums:
    def test_axis_values(self):
        assert Axis.X.value == "x"
        assert Axis.Y.value == "y"
        assert Axis.Z.value == "z"
    def test_axis_members(self):
        assert list(Axis) == [Axis.X, Axis.Y, Axis.Z]
    def test_kind_values(self):
        assert Kind.REPEAT.value == "repeat"
        assert Kind.KIFS.value == "kifs"
        assert Kind.STRETCH.value == "stretch"
        assert Kind.TWIST.value == "twist"
    def test_kind_members(self):
        assert list(Kind) == [Kind.REPEAT, Kind.KIFS, Kind.STRETCH, Kind.TWIST]


# =============================================================================
# Node boundary conditions (8 tests)
# =============================================================================
class TestNodeBoundaries:
    def test_float_zero(self):
        assert FloatNode(0.0).value == 0.0
    def test_float_negative(self):
        assert FloatNode(-1.0).value == -1.0
    def test_float_large(self):
        assert FloatNode(1e10).value == 1e10
    def test_float_tiny(self):
        assert FloatNode(1e-10).value == 1e-10
    def test_vec3_zero(self):
        v = Vec3Node(0.0, 0.0, 0.0)
        assert v.as_tuple() == (0.0, 0.0, 0.0)
    def test_vec3_negative(self):
        v = Vec3Node(-1.0, -2.0, -3.0)
        assert v.as_tuple() == (-1.0, -2.0, -3.0)
    def test_from_tuple_int_coercion(self):
        v = Vec3Node.from_tuple((1, 2, 3))
        assert v.as_tuple() == (1.0, 2.0, 3.0)
    def test_as_tuple_always_float(self):
        v = Vec3Node(1, 2, 3)
        t = v.as_tuple()
        assert all(isinstance(x, float) for x in t)


# =============================================================================
# AstBuilder API (3 tests)
# =============================================================================
class TestAstBuilderAPI:
    def test_walk_list(self):
        result = AstBuilder.walk([1, 2, 3])
        assert result == [{}, {}, {}]
    def test_walk_nested_list(self):
        result = AstBuilder.walk([[1], [2]])
        assert len(result) == 2
    def test_walk_nested_dict(self):
        result = AstBuilder.walk({"a": {"b": 1}})
        assert result == {"a": {"b": {}}}


# =============================================================================
# Declarative named scenes (2 tests)
# =============================================================================
class TestDeclarativeNamed:
    def test_named_scene_deep_label(self):
        d = {"pipeline": [{"type": "repeat", "cell_size": (2.0, 2.0, 2.0)}],
             "primitives": [{"type": "sphere", "radius": 1.0}],
             "name": "declarative_scene"}
        sg = AstBuilder.walk(d)
        assert isinstance(sg, SceneGraph)
        assert sg.name == "declarative_scene"
    def test_all_primitives(self):
        d = {"pipeline": [], "primitives": [
            {"type": "sphere", "radius": 1.0},
            {"type": "box", "size": (2.0, 2.0, 2.0)},
            {"type": "torus", "major_radius": 2.0, "minor_radius": 0.5},
        ]}
        sg = AstBuilder.walk(d)
        assert len(sg.primitives) == 3


# =============================================================================
# All domain ops composition (1 test)
# =============================================================================
class TestAllDomainOpsComposition:
    def test_pipeline_all_11_ops(self):
        from engine.rendering.demoscene.ast_nodes import (
            RepeatNode, CellIdNode, MirrorNode, KifsNode, TwistNode,
            BendNode, StretchNode,
        )
        sg = walk_composition(lambda p: sdSphere(
            domain_stretch_x(
                domain_stretch_y(
                    domain_stretch_z(
                        domain_bend(
                            domain_twist(
                                domain_kifs(
                                    domain_mirror_z(
                                        domain_mirror_y(
                                            domain_mirror_x(
                                                domain_cell_id(
                                                    domain_repeat(p, (2.0, 2.0, 2.0)),
                                                    (4.0, 4.0, 1.0)
                                                )
                                            )
                                        )
                                    ),
                                    6.0
                                ),
                                0.5
                            ),
                            3.0
                        ),
                        2.0
                    ),
                    3.0
                ),
                4.0
            ),
            1.0
        ))
        assert len(sg.pipeline) == 11
        expected_types = [RepeatNode, CellIdNode, MirrorNode, MirrorNode,
                          MirrorNode, KifsNode, TwistNode, BendNode,
                          StretchNode, StretchNode, StretchNode]
        for actual, expected in zip(sg.pipeline, expected_types):
            assert isinstance(actual, expected), f"Expected {expected.__name__}, got {type(actual).__name__}"


# =============================================================================
# Composition errors (3 tests)
# =============================================================================
class TestCompositionErrors:
    def test_lambda_no_calls(self):
        with pytest.raises(ValueError):
            walk_composition(lambda p: p)
    def test_unknown_function(self):
        with pytest.raises(ValueError):
            walk_composition(lambda p: unknown_function(p, 1.0))
    def test_none_in_args(self):
        with pytest.raises(TypeError):
            walk_composition(lambda p: sdSphere(p, None))
