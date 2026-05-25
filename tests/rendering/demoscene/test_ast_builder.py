"""Baseline acceptance tests for AST builder."""
from __future__ import annotations
import pytest
from engine.rendering.demoscene.ast_nodes import (
    Axis, BendNode, BoxNode, CellIdNode, CombineNode, CompensationNode,
    ExprNode, FloatNode, IntersectionNode, KifsNode, Kind, MirrorNode,
    PositionNode, RepeatNode, SceneGraph, SdfPrimitiveNode, SphereNode,
    StretchNode, SubtractionNode, TorusNode, TwistNode, UnionNode, Vec3Node,
)
from engine.rendering.demoscene.ast_builder import (
    AstBuilder, build_from_composition, walk_composition,
)


class TestFloatNode:
    def test_create(self):
        n = FloatNode(3.14)
        assert n.value == 3.14
    def test_label(self):
        assert FloatNode(2.5).label() == "Float(2.5)"
    def test_is_expr_node(self):
        assert isinstance(FloatNode(1.0), ExprNode)
    def test_frozen(self):
        n = FloatNode(1.0)
        with pytest.raises(AttributeError):
            n.value = 2.0


class TestVec3Node:
    def test_create(self):
        v = Vec3Node(1.0, 2.0, 3.0)
        assert (v.x, v.y, v.z) == (1.0, 2.0, 3.0)
    def test_label(self):
        assert Vec3Node(1, 2, 3).label() == "Vec3(1.0, 2.0, 3.0)"
    def test_as_tuple(self):
        assert Vec3Node(1.0, 2.0, 3.0).as_tuple() == (1.0, 2.0, 3.0)


class TestPositionNode:
    def test_create(self):
        p = PositionNode()
        assert isinstance(p, ExprNode)
    def test_label(self):
        assert PositionNode().label() == "Position(p)"


class TestRepeatNode:
    def test_create(self):
        r = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        assert isinstance(r, RepeatNode)
    def test_label(self):
        r = RepeatNode(PositionNode(), Vec3Node(3.0, 3.0, 3.0))
        assert "Repeat" in r.label()
    def test_children(self):
        r = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        children = list(r.children())
        assert len(children) == 1
        assert isinstance(children[0], PositionNode)
    def test_default_cell_size(self):
        r = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        assert r.cell_size.as_tuple() == (2.0, 2.0, 2.0)


class TestMirrorNode:
    def test_create(self):
        m = MirrorNode(PositionNode(), Axis.X)
        assert m.axis == Axis.X
    def test_label(self):
        m = MirrorNode(PositionNode(), Axis.Y)
        assert "Y" in m.label()


class TestKifsNode:
    def test_create(self):
        k = KifsNode(PositionNode(), FloatNode(6.0))
        assert k.folds.value == 6.0


class TestTwistNode:
    def test_create(self):
        t = TwistNode(PositionNode(), FloatNode(1.0))
        assert t.rate.value == 1.0


class TestBendNode:
    def test_create(self):
        b = BendNode(PositionNode(), FloatNode(5.0))
        assert b.radius.value == 5.0


class TestStretchNode:
    def test_create(self):
        s = StretchNode(PositionNode(), FloatNode(2.0), Axis.X)
        assert s.stretch.value == 2.0
        assert s.axis == Axis.X


class TestSphereNode:
    def test_create(self):
        s = SphereNode(PositionNode(), FloatNode(1.0))
        assert s.radius.value == 1.0
    def test_label(self):
        s = SphereNode(PositionNode(), FloatNode(2.0))
        assert "Sphere" in s.label()


class TestBoxNode:
    def test_create(self):
        b = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        assert b.size.as_tuple() == (1.0, 1.0, 1.0)


class TestTorusNode:
    def test_create(self):
        t = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        assert t.major_radius.value == 2.0
        assert t.minor_radius.value == 0.5


class TestCombineNode:
    def test_create_union(self):
        u = UnionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                      BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)))
        assert u.kind == "union"
    def test_create_intersection(self):
        i = IntersectionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                             BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)))
        assert i.kind == "intersection"
    def test_create_subtraction(self):
        s = SubtractionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)))
        assert s.kind == "subtraction"
    def test_children(self):
        c = UnionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                      BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)))
        children = list(c.children())
        assert len(children) == 2
    def test_labels(self):
        u = UnionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                      BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)))
        assert u.label() == "Union(...)"
        i = IntersectionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                             BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)))
        assert i.label() == "Intersection(...)"
        s = SubtractionNode(SphereNode(PositionNode(), FloatNode(1.0)),
                            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)))
        assert s.label() == "Subtraction(...)"


class TestSceneGraph:
    def test_create_empty(self):
        sg = SceneGraph(primitives=())
        assert len(sg.primitives) == 0
        assert len(sg.pipeline) == 0
    def test_with_pipeline(self):
        pipe = (RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),)
        sg = SceneGraph(primitives=(SphereNode(PositionNode(), FloatNode(1.0)),), pipeline=pipe)
        assert len(sg.pipeline) == 1
    def test_with_name(self):
        sg = SceneGraph(primitives=(), name="test_scene")
        assert sg.name == "test_scene"
    def test_children(self):
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),)
        )
        children = list(sg.children())
        assert len(children) == 2
    def test_label(self):
        sg = SceneGraph(primitives=())
        assert "SceneGraph" in sg.label()


class TestAstBuilderDeclarative:
    def test_dict_marker(self):
        result = AstBuilder.walk({"type": "sphere", "radius": 1.5})
        assert isinstance(result, SphereNode)
        assert result.radius.value == 1.5
    def test_marker_with_position(self):
        result = AstBuilder.walk({"type": "sphere", "position": {"type": "repeat", "cell_size": (4.0, 4.0, 4.0)}, "radius": 1.0})
        assert isinstance(result, SphereNode)
    def test_list_of_markers(self):
        result = AstBuilder.walk([
            {"type": "sphere", "radius": 1.0},
            {"type": "box", "size": (2.0, 2.0, 2.0)},
        ])
        assert len(result) == 2


class TestAstBuilderComposition:
    def test_basic_sphere(self):
        sg = walk_composition(lambda p: sdSphere(p, 1.0))
        assert len(sg.primitives) == 1
        assert isinstance(sg.primitives[0], SphereNode)
    def test_basic_box(self):
        sg = walk_composition(lambda p: sdBox(p, 1.0, 2.0, 3.0))
        assert len(sg.primitives) == 1
        assert isinstance(sg.primitives[0], BoxNode)
    def test_domain_repeat(self):
        sg = walk_composition(lambda p: sdSphere(domain_repeat(p, (4.0, 4.0, 4.0)), 1.0))
        assert len(sg.pipeline) == 1
        assert isinstance(sg.pipeline[0], RepeatNode)
    def test_chained_domain_ops(self):
        sg = walk_composition(lambda p: sdSphere(domain_repeat(domain_mirror_x(p), (4.0, 4.0, 4.0)), 1.0))
        assert len(sg.pipeline) == 2
    def test_kifs_composition(self):
        sg = walk_composition(lambda p: sdSphere(domain_kifs(p, 6.0), 1.0))
        assert len(sg.pipeline) == 1
        assert isinstance(sg.pipeline[0], KifsNode)
    def test_torus_composition(self):
        sg = walk_composition(lambda p: sdTorus(p, 2.0, 0.5))
        assert len(sg.primitives) == 1
        assert isinstance(sg.primitives[0], TorusNode)


class TestDomainPipelinePrimitives:
    def test_pipeline_only(self):
        sg = SceneGraph(primitives=(),
                        pipeline=(RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),))
        assert len(sg.pipeline) == 1
        assert len(sg.primitives) == 0
    def test_primitives_only(self):
        sg = SceneGraph(primitives=(SphereNode(PositionNode(), FloatNode(1.0)),))
        assert len(sg.primitives) == 1
        assert len(sg.pipeline) == 0


class TestEdgeCases:
    def test_empty_dict(self):
        result = AstBuilder.walk({})
        assert result == {}
    def test_empty_list(self):
        result = AstBuilder.walk([])
        assert result == []
    def test_none_input(self):
        result = AstBuilder.walk(None)
        assert result == {}
    def test_unknown_type(self):
        result = AstBuilder.walk(42)
        assert result == {}
    def test_unknown_type_no_pipeline(self):
        class Obj:
            pass
        result = AstBuilder.walk(Obj())
        assert result == {}
    def test_label_on_empty_pipeline(self):
        sg = SceneGraph(primitives=())
        assert "0 pipeline" in sg.label()
    def test_walk_expr_node(self):
        n = FloatNode(1.0)
        result = AstBuilder.walk(n)
        assert result is n


class TestAcceptance:
    def test_sphere_with_repeat(self):
        sg = walk_composition(lambda p: sdSphere(domain_repeat(p, (3.0, 3.0, 3.0)), 1.0))
        assert isinstance(sg, SceneGraph)
        assert len(sg.pipeline) == 1
        assert isinstance(sg.pipeline[0], RepeatNode)
        assert len(sg.primitives) == 1
    def test_box_with_mirror(self):
        sg = walk_composition(lambda p: sdBox(domain_mirror_x(p), 1.0, 1.0, 1.0))
        assert len(sg.pipeline) == 1
        assert isinstance(sg.pipeline[0], MirrorNode)
    def test_torus_with_twist(self):
        sg = walk_composition(lambda p: sdTorus(domain_twist(p, 0.5), 2.0, 0.5))
        assert len(sg.pipeline) == 1
        assert isinstance(sg.pipeline[0], TwistNode)
    def test_sphere_with_bend(self):
        sg = walk_composition(lambda p: sdSphere(domain_bend(p, 3.0), 1.0))
        assert len(sg.pipeline) == 1
        assert isinstance(sg.pipeline[0], BendNode)
    def test_box_with_stretch(self):
        sg = walk_composition(lambda p: sdBox(domain_stretch_x(p, 3.0), 1.0, 1.0, 1.0))
        assert len(sg.pipeline) == 1
        assert isinstance(sg.pipeline[0], StretchNode)
    def test_deeply_chained(self):
        sg = walk_composition(lambda p: sdSphere(
            domain_repeat(domain_mirror_x(domain_twist(p, 0.5)), (4.0, 4.0, 4.0)), 1.0))
        assert len(sg.pipeline) == 3
    def test_build_from_composition_alias(self):
        assert build_from_composition is walk_composition
    def test_walk_odd_object(self):
        class Marker:
            _node_type = "sphere"
            radius = 2.0
        result = AstBuilder.walk(Marker())
        assert isinstance(result, SphereNode)
    def test_sphere_as_acceptance(self):
        sg = walk_composition(lambda p: sdSphere(p, 2.0))
        sphere = sg.primitives[0]
        assert sphere.radius.value == 2.0
    def test_repeat_cell_id_defaults(self):
        result = AstBuilder.walk({"type": "cell_id", "cell_size": (4.0, 4.0, 1.0)})
        assert isinstance(result, CellIdNode)
        assert result.cell_size.as_tuple() == (4.0, 4.0, 1.0)
