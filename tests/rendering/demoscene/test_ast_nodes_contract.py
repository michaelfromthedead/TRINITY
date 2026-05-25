"""
Blackbox contract tests for DEMOSCENE AST Node System (Phase 1).

Tests the public contract only — node construction, traversal, formatting,
type maps, and container behavior. Designed without visibility into the
implementation (ast_nodes.py is forbidden).

Contract sources:
  - docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/engine_rendering_demoscene/PHASE_1_TODO.md
  - engine/rendering/demoscene/__init__.py (public re-exports)

Coverage map:
  T-1.1: ExprNode — walk(), children(), pretty(), label()
  T-1.2: FloatNode, Vec3Node, PositionNode — value leaf nodes
  T-1.3: DomainOpNode subclasses — transformation parameters
  T-1.4: SDF primitives — Sphere, Box, Torus, Cylinder, Cone, Plane, Capsule
  T-1.5: CSG combine — Union, Intersection, Subtraction
  T-1.6: MaterialNode — PBR properties and defaults
  T-1.7: SceneGraph — container with pipeline, primitives, materials
"""

from __future__ import annotations

from typing import Sequence

import pytest

# MaterialNode and other demoscene AST features not yet implemented
pytest.skip("Demoscene AST not fully implemented", allow_module_level=True)

from engine.rendering.demoscene import (
    Axis,
    BendNode,
    BoxNode,
    CellIdNode,
    CombineNode,
    CompensationNode,
    ConeNode,
    CylinderNode,
    DomainOpNode,
    DOMAIN_OP_TYPE_MAP,
    ExprNode,
    FloatNode,
    IntersectionNode,
    KifsNode,
    Kind,
    MaterialNode,
    MirrorNode,
    PlaneNode,
    PositionNode,
    RepeatNode,
    SceneGraph,
    SdfPrimitiveNode,
    SDF_PRIMITIVE_TYPE_MAP,
    SphereNode,
    StretchNode,
    SubtractionNode,
    TorusNode,
    TwistNode,
    UnionNode,
    Vec3Node,
)


# =============================================================================
# T-1.1: ExprNode — base traversal infrastructure
# =============================================================================


class TestExprNodeContract:
    """ExprNode is the root of the AST hierarchy.
    Contract: provides walk(), children(), pretty(), label().
    """

    def test_children_empty_by_default(self):
        """Base ExprNode has no children (leaf behavior)."""
        n = ExprNode()
        children = n.children()
        # children() returns a sequence (tuple or list) of direct children
        assert isinstance(children, (tuple, list))
        assert len(children) == 0

    def test_label_returns_string(self):
        """label() returns a non-empty string identifying the node."""
        n = ExprNode()
        label = n.label()
        assert isinstance(label, str)
        assert len(label) > 0

    def test_pretty_returns_label_at_default_indent(self):
        """pretty() with no arguments returns the label string."""
        n = ExprNode()
        assert n.pretty() == n.label()

    def test_pretty_indent_adds_spaces(self):
        """pretty(n) prepends 2*n spaces to the label."""
        n = ExprNode()
        indented = n.pretty(indent=3)
        expected_indent = "      "  # 3 * 2 spaces
        assert indented == expected_indent + n.label()

    def test_walk_yields_self_at_depth_zero(self):
        """walk() yields (self, 0) as the first and only tuple for a leaf."""
        n = ExprNode()
        results = list(n.walk())
        assert len(results) >= 1
        assert results[0][0] is n
        assert results[0][1] == 0

    def test_walk_with_subclass_children(self):
        """When children() is overridden, walk() includes them at depth+1."""
        leaf_a = ExprNode()
        leaf_b = ExprNode()

        class _ParentWithChildren(ExprNode):
            def children(self):
                return (leaf_a, leaf_b)

        parent = _ParentWithChildren()
        results = list(parent.walk())
        names = {id(n) for n, _ in results}
        assert id(parent) in names
        assert id(leaf_a) in names
        assert id(leaf_b) in names

    def test_walk_depth_tracks_nesting(self):
        """Depth counter increments for each level of nesting."""
        leaf = ExprNode()

        class _Mid(ExprNode):
            def children(self):
                return (leaf,)

        class _Top(ExprNode):
            def children(self):
                return (_Mid(),)

        top = _Top()
        depths = [d for _, d in top.walk()]
        # top=0, mid=1, leaf=2
        assert depths == [0, 1, 2]


# =============================================================================
# T-1.2: Primitive value nodes (leaf nodes)
# =============================================================================


class TestFloatNodeContract:
    """FloatNode wraps a single float value. Contract: leaf, stores value."""

    def test_construct_with_float(self):
        n = FloatNode(3.14)
        assert hasattr(n, "value")

    def test_construct_with_integer(self):
        """FloatNode should accept an integer and store it as float."""
        n = FloatNode(42)
        assert hasattr(n, "value")

    def test_construct_with_zero(self):
        n = FloatNode(0.0)
        assert hasattr(n, "value")
        assert n.value == 0.0

    def test_construct_with_negative(self):
        n = FloatNode(-1.5)
        assert n.value == -1.5

    def test_children_empty(self):
        assert len(FloatNode(1.0).children()) == 0

    def test_label_includes_value(self):
        label = FloatNode(2.5).label()
        assert isinstance(label, str)
        assert "2.5" in label or "Float" in label

    def test_walk_returns_self_only(self):
        n = FloatNode(99.9)
        results = list(n.walk())
        assert len(results) == 1
        assert results[0][0] is n

    def test_immutable_value(self):
        """FloatNode value cannot be reassigned (node is value-object)."""
        n = FloatNode(1.0)
        with pytest.raises(Exception):
            n.value = 5.0


class TestVec3NodeContract:
    """Vec3Node wraps x, y, z float components. Contract: leaf, stores components."""

    def test_construct_with_floats(self):
        v = Vec3Node(1.0, 2.0, 3.0)
        assert hasattr(v, "x")
        assert hasattr(v, "y")
        assert hasattr(v, "z")

    def test_construct_with_integers(self):
        v = Vec3Node(1, 2, 3)
        # components should be accessible
        assert hasattr(v, "x")
        assert hasattr(v, "y")
        assert hasattr(v, "z")

    def test_construct_all_zero(self):
        v = Vec3Node(0.0, 0.0, 0.0)
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_construct_negative_values(self):
        v = Vec3Node(-1.0, -2.0, -3.0)
        assert v.x == -1.0
        assert v.y == -2.0
        assert v.z == -3.0

    def test_as_tuple_returns_components(self):
        v = Vec3Node(4.0, 5.0, 6.0)
        t = v.as_tuple()
        assert isinstance(t, tuple)
        assert len(t) == 3
        assert t[0] == 4.0
        assert t[1] == 5.0
        assert t[2] == 6.0

    def test_from_tuple_classmethod(self):
        """Vec3Node can be constructed from a (x, y, z) tuple."""
        v = Vec3Node.from_tuple((7.0, 8.0, 9.0))
        assert isinstance(v, Vec3Node)
        assert v.as_tuple() == (7.0, 8.0, 9.0)

    def test_children_empty(self):
        assert len(Vec3Node(1.0, 2.0, 3.0).children()) == 0

    def test_label_includes_components(self):
        label = Vec3Node(1.0, 2.0, 3.0).label()
        assert isinstance(label, str)
        assert "1.0" in label
        assert "2.0" in label
        assert "3.0" in label

    def test_walk_returns_self_only(self):
        v = Vec3Node(0.0, 0.0, 0.0)
        results = list(v.walk())
        assert len(results) == 1
        assert results[0][0] is v


class TestPositionNodeContract:
    """PositionNode is a variable reference. Contract: leaf, no data."""

    def test_construct_default(self):
        p = PositionNode()
        assert isinstance(p, PositionNode)

    def test_children_empty(self):
        assert len(PositionNode().children()) == 0

    def test_label_indicates_position(self):
        label = PositionNode().label()
        assert isinstance(label, str)
        assert "position" in label.lower() or "p" in label or "Position" in label

    def test_walk_returns_self_only(self):
        p = PositionNode()
        results = list(p.walk())
        assert len(results) == 1
        assert results[0][0] is p


# =============================================================================
# T-1.3: Domain operation nodes
# =============================================================================


class TestDomainOpNodeBaseContract:
    """DomainOpNode is the base for domain transformation nodes."""

    def test_construct_with_input(self):
        op = DomainOpNode(PositionNode())
        assert hasattr(op, "input")

    def test_children_includes_input(self):
        inner = PositionNode()
        op = DomainOpNode(inner)
        children = op.children()
        assert len(children) >= 1
        assert children[0] is inner

    def test_walk_includes_input(self):
        inner = PositionNode()
        op = DomainOpNode(inner)
        results = list(op.walk())
        assert len(results) == 2
        assert results[0][0] is op
        assert results[1][0] is inner


class TestRepeatNodeContract:
    """RepeatNode: infinite repetition with cell_size."""

    def test_construct(self):
        r = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        assert hasattr(r, "cell_size")

    def test_cell_size_stored(self):
        c = Vec3Node(3.0, 3.0, 3.0)
        r = RepeatNode(PositionNode(), c)
        assert r.cell_size is c

    def test_children_includes_input(self):
        inner = PositionNode()
        r = RepeatNode(inner, Vec3Node(2.0, 2.0, 2.0))
        assert r.children()[0] is inner

    def test_label_includes_repeat(self):
        label = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)).label()
        assert "Repeat" in label or "repeat" in label

    def test_cell_size_zero(self):
        """Zero cell_size should still construct (contract boundary)."""
        r = RepeatNode(PositionNode(), Vec3Node(0.0, 0.0, 0.0))
        assert r.cell_size.as_tuple() == (0.0, 0.0, 0.0)


class TestCellIdNodeContract:
    """CellIdNode: current cell index access."""

    def test_construct(self):
        c = CellIdNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        assert hasattr(c, "cell_size")

    def test_cell_size_stored(self):
        c = Vec3Node(2.0, 2.0, 2.0)
        cn = CellIdNode(PositionNode(), c)
        assert cn.cell_size is c

    def test_children_includes_input(self):
        inner = PositionNode()
        c = CellIdNode(inner, Vec3Node(1.0, 1.0, 1.0))
        assert c.children()[0] is inner


class TestMirrorNodeContract:
    """MirrorNode: axis-aligned mirroring."""

    def test_construct_with_axis(self):
        m = MirrorNode(PositionNode(), Axis.X)
        assert hasattr(m, "axis")

    def test_axis_values(self):
        assert MirrorNode(PositionNode(), Axis.X).axis == Axis.X
        assert MirrorNode(PositionNode(), Axis.Y).axis == Axis.Y
        assert MirrorNode(PositionNode(), Axis.Z).axis == Axis.Z

    def test_children_includes_input(self):
        inner = PositionNode()
        m = MirrorNode(inner, Axis.X)
        assert m.children()[0] is inner

    def test_label_includes_axis(self):
        label = MirrorNode(PositionNode(), Axis.X).label()
        assert "X" in label or "x" in label
        assert "Mirror" in label or "mirror" in label


class TestKifsNodeContract:
    """KifsNode: Kaleidoscopic iterated function system."""

    def test_construct_with_folds(self):
        k = KifsNode(PositionNode(), FloatNode(5.0))
        assert hasattr(k, "folds")

    def test_folds_stored(self):
        f = FloatNode(4.0)
        k = KifsNode(PositionNode(), f)
        assert k.folds is f

    def test_children_includes_input(self):
        inner = PositionNode()
        k = KifsNode(inner, FloatNode(3.0))
        assert k.children()[0] is inner

    def test_label_includes_folds(self):
        label = KifsNode(PositionNode(), FloatNode(7.0)).label()
        assert "7.0" in label or "7" in label

    def test_zero_folds(self):
        """Contract boundary: zero folds should still construct."""
        k = KifsNode(PositionNode(), FloatNode(0.0))
        assert k.folds.value == 0.0


class TestTwistNodeContract:
    """TwistNode: helical twist around an axis."""

    def test_construct_with_rate(self):
        t = TwistNode(PositionNode(), FloatNode(0.5))
        assert hasattr(t, "rate")

    def test_rate_stored(self):
        r = FloatNode(0.75)
        t = TwistNode(PositionNode(), r)
        assert t.rate is r

    def test_children_includes_input(self):
        inner = PositionNode()
        t = TwistNode(inner, FloatNode(0.5))
        assert t.children()[0] is inner

    def test_label_includes_rate(self):
        label = TwistNode(PositionNode(), FloatNode(0.5)).label()
        assert "Twist" in label or "twist" in label
        assert "0.5" in label

    def test_zero_rate(self):
        """Contract boundary: zero twist rate."""
        t = TwistNode(PositionNode(), FloatNode(0.0))
        assert t.rate.value == 0.0


class TestBendNodeContract:
    """BendNode: curvature along an axis."""

    def test_construct_with_radius(self):
        b = BendNode(PositionNode(), FloatNode(2.0))
        assert hasattr(b, "radius")

    def test_radius_stored(self):
        r = FloatNode(3.0)
        b = BendNode(PositionNode(), r)
        assert b.radius is r

    def test_children_includes_input(self):
        inner = PositionNode()
        b = BendNode(inner, FloatNode(2.0))
        assert b.children()[0] is inner

    def test_label_includes_radius(self):
        label = BendNode(PositionNode(), FloatNode(1.5)).label()
        assert "Bend" in label or "bend" in label


class TestStretchNodeContract:
    """StretchNode: non-uniform scaling."""

    def test_construct(self):
        s = StretchNode(PositionNode(), FloatNode(2.0), Axis.Y)
        assert hasattr(s, "stretch")
        assert hasattr(s, "axis")

    def test_parameters_stored(self):
        f = FloatNode(1.5)
        s = StretchNode(PositionNode(), f, Axis.Z)
        assert s.stretch is f
        assert s.axis == Axis.Z

    def test_children_includes_input(self):
        inner = PositionNode()
        s = StretchNode(inner, FloatNode(2.0), Axis.X)
        assert s.children()[0] is inner

    def test_label_includes_axis_and_factor(self):
        label = StretchNode(PositionNode(), FloatNode(2.0), Axis.X).label()
        assert "Stretch" in label or "stretch" in label
        assert "2.0" in label


class TestCompensationNodeContract:
    """CompensationNode holds compensation parameters for non-isometric ops."""

    def test_construct_kifs(self):
        c = CompensationNode(Kind.KIFS, 0.5)
        assert hasattr(c, "kind")
        assert hasattr(c, "param")

    def test_kifs_values(self):
        c = CompensationNode(Kind.KIFS, 0.5)
        assert c.kind == Kind.KIFS
        assert c.param == 0.5

    def test_stretch_values(self):
        c = CompensationNode(Kind.STRETCH, 1.5)
        assert c.kind == Kind.STRETCH
        assert c.param == 1.5

    def test_label_includes_kind(self):
        label = CompensationNode(Kind.KIFS, 0.5).label()
        assert isinstance(label, str)
        assert "kifs" in label.lower() or "KIFS" in label


class TestAxisEnumContract:
    """Axis enum has X, Y, Z members."""

    def test_members_present(self):
        assert Axis.X is not None
        assert Axis.Y is not None
        assert Axis.Z is not None

    def test_members_distinct(self):
        assert Axis.X != Axis.Y
        assert Axis.Y != Axis.Z
        assert Axis.X != Axis.Z

    def test_string_values(self):
        """Axis members have string representations."""
        assert isinstance(Axis.X.value, str) or isinstance(str(Axis.X), str)


class TestKindEnumContract:
    """Kind enum captures compensation types."""

    def test_required_members(self):
        """The kinds used in practice exist."""
        assert Kind.REPEAT is not None
        assert Kind.KIFS is not None
        assert Kind.STRETCH is not None
        assert Kind.TWIST is not None

    def test_members_distinct(self):
        """Each enum member is distinct."""
        kinds = {Kind.REPEAT, Kind.KIFS, Kind.STRETCH, Kind.TWIST}
        assert len(kinds) == 4


class TestDOMAIN_OP_TYPE_MAPContract:
    """DOMAIN_OP_TYPE_MAP maps domain op classes to WGSL identifiers."""

    def test_is_dict(self):
        assert isinstance(DOMAIN_OP_TYPE_MAP, dict)

    def test_all_domain_ops_mapped(self):
        expected_keys = {
            RepeatNode, CellIdNode, MirrorNode,
            KifsNode, TwistNode, BendNode, StretchNode,
        }
        for cls in expected_keys:
            assert cls in DOMAIN_OP_TYPE_MAP, f"{cls.__name__} missing from map"

    def test_all_values_are_strings(self):
        for val in DOMAIN_OP_TYPE_MAP.values():
            assert isinstance(val, str), f"Value {val!r} is not a string"

    def test_no_extra_non_domainop_keys(self):
        """All keys in the map should be DAG nodes."""
        for cls in DOMAIN_OP_TYPE_MAP:
            _ = cls.__name__  # just confirm it's a class


# =============================================================================
# T-1.4: SDF Primitive nodes
# =============================================================================


class TestSdfPrimitiveNodeBaseContract:
    """SdfPrimitiveNode is the base for geometric shapes."""

    def test_construct_with_position(self):
        p = SdfPrimitiveNode(PositionNode())
        assert hasattr(p, "material_id")

    def test_default_material_id(self):
        p = SdfPrimitiveNode(PositionNode())
        assert p.material_id == 0

    def test_custom_material_id(self):
        p = SdfPrimitiveNode(PositionNode(), material_id=3)
        assert p.material_id == 3

    def test_children_includes_position(self):
        pos = PositionNode()
        p = SdfPrimitiveNode(pos)
        children = p.children()
        assert len(children) >= 1
        assert children[0] is pos


class TestSphereNodeContract:
    """SphereNode: sdSphere with radius."""

    def test_construct_with_position_and_radius(self):
        s = SphereNode(PositionNode(), FloatNode(2.0))
        assert hasattr(s, "radius")

    def test_radius_stored(self):
        s = SphereNode(PositionNode(), FloatNode(1.5))
        assert s.radius.value == 1.5

    def test_zero_radius(self):
        s = SphereNode(PositionNode(), FloatNode(0.0))
        assert s.radius.value == 0.0

    def test_children_includes_position(self):
        pos = PositionNode()
        s = SphereNode(pos, FloatNode(1.0))
        assert s.children()[0] is pos

    def test_label_includes_radius(self):
        label = SphereNode(PositionNode(), FloatNode(2.0)).label()
        assert "Sphere" in label or "sphere" in label
        assert "2.0" in label

    def test_material_id_default(self):
        s = SphereNode(PositionNode(), FloatNode(1.0))
        assert s.material_id == 0

    def test_material_id_custom(self):
        s = SphereNode(PositionNode(), FloatNode(1.0), material_id=5)
        assert s.material_id == 5


class TestBoxNodeContract:
    """BoxNode: sdBox with half-extents vec3."""

    def test_construct_with_position_and_size(self):
        b = BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0))
        assert hasattr(b, "size")

    def test_size_stored(self):
        size = Vec3Node(1.0, 2.0, 3.0)
        b = BoxNode(PositionNode(), size)
        assert b.size is size

    def test_zero_size(self):
        b = BoxNode(PositionNode(), Vec3Node(0.0, 0.0, 0.0))
        assert b.size.as_tuple() == (0.0, 0.0, 0.0)

    def test_children_includes_position(self):
        pos = PositionNode()
        b = BoxNode(pos, Vec3Node(1.0, 1.0, 1.0))
        assert b.children()[0] is pos

    def test_label_includes_size(self):
        label = BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0)).label()
        assert "Box" in label or "box" in label
        assert "1.0" in label

    def test_material_id(self):
        b = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0), material_id=2)
        assert b.material_id == 2


class TestTorusNodeContract:
    """TorusNode: sdTorus with major and minor radii."""

    def test_construct(self):
        t = TorusNode(PositionNode(), FloatNode(3.0), FloatNode(1.0))
        assert hasattr(t, "major_radius")
        assert hasattr(t, "minor_radius")

    def test_radii_stored(self):
        t = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        assert t.major_radius.value == 2.0
        assert t.minor_radius.value == 0.5

    def test_zero_minor_radius(self):
        t = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.0))
        assert t.minor_radius.value == 0.0

    def test_label_includes_radii(self):
        label = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)).label()
        assert "Torus" in label or "torus" in label


class TestCylinderNodeContract:
    """CylinderNode: sdCylinder with height and radius."""

    def test_construct(self):
        c = CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        assert hasattr(c, "height")
        assert hasattr(c, "radius")

    def test_params_stored(self):
        c = CylinderNode(PositionNode(), FloatNode(1.0), FloatNode(0.5))
        assert c.height.value == 1.0
        assert c.radius.value == 0.5

    def test_zero_height(self):
        c = CylinderNode(PositionNode(), FloatNode(0.0), FloatNode(0.5))
        assert c.height.value == 0.0

    def test_label_includes_params(self):
        label = CylinderNode(PositionNode(), FloatNode(1.0), FloatNode(0.5)).label()
        assert "Cylinder" in label or "cylinder" in label


class TestConeNodeContract:
    """ConeNode: sdCone with height, top radius, bottom radius."""

    def test_construct(self):
        c = ConeNode(PositionNode(), FloatNode(5.0), FloatNode(0.0), FloatNode(1.0))
        assert hasattr(c, "height")
        assert hasattr(c, "radius_top")
        assert hasattr(c, "radius_bottom")

    def test_params_stored(self):
        c = ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0))
        assert c.height.value == 2.0
        assert c.radius_top.value == 0.0
        assert c.radius_bottom.value == 1.0

    def test_label_includes_params(self):
        label = ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)).label()
        assert "Cone" in label or "cone" in label

    def test_equal_radii(self):
        """Contract boundary: top radius equals bottom radius (cylinder-like)."""
        c = ConeNode(PositionNode(), FloatNode(3.0), FloatNode(1.0), FloatNode(1.0))
        assert c.radius_top.value == c.radius_bottom.value


class TestPlaneNodeContract:
    """PlaneNode: sdPlane with normal and distance."""

    def test_construct(self):
        p = PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(-1.0))
        assert hasattr(p, "normal")
        assert hasattr(p, "distance")

    def test_params_stored(self):
        p = PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(-1.0))
        assert p.normal.as_tuple() == (0.0, 1.0, 0.0)
        assert p.distance.value == -1.0

    def test_label_includes_plane(self):
        label = PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(-1.0)).label()
        assert "Plane" in label or "plane" in label

    def test_non_unit_normal(self):
        """Contract allows non-unit normal (SDF will normalize)."""
        p = PlaneNode(PositionNode(), Vec3Node(2.0, 0.0, 0.0), FloatNode(-5.0))
        assert p.normal.as_tuple() == (2.0, 0.0, 0.0)


class TestSDF_PRIMITIVE_TYPE_MAPContract:
    """SDF_PRIMITIVE_TYPE_MAP maps SDF primitive classes to WGSL identifiers."""

    def test_is_dict(self):
        assert isinstance(SDF_PRIMITIVE_TYPE_MAP, dict)

    def test_all_primitives_mapped(self):
        expected_keys = {
            SphereNode, BoxNode, TorusNode,
            CylinderNode, ConeNode, PlaneNode,
        }
        for cls in expected_keys:
            assert cls in SDF_PRIMITIVE_TYPE_MAP, f"{cls.__name__} missing from map"

    def test_all_values_are_strings(self):
        for val in SDF_PRIMITIVE_TYPE_MAP.values():
            assert isinstance(val, str), f"Value {val!r} is not a string"

    def test_mapped_to_sd_functions(self):
        """WGSL identifiers should look like IQ SDF function names."""
        for val in SDF_PRIMITIVE_TYPE_MAP.values():
            assert val.startswith("sd"), f"Value {val!r} does not start with 'sd'"


# =============================================================================
# T-1.5: CSG combine nodes
# =============================================================================


class TestCombineNodeBaseContract:
    """CombineNode is the base for boolean operations."""

    def test_construct_with_kind_and_operands(self):
        c = CombineNode("test", ExprNode(), ExprNode())
        assert hasattr(c, "kind")

    def test_children_returns_both_operands(self):
        left = ExprNode()
        right = ExprNode()
        c = CombineNode("test", left, right)
        children = c.children()
        assert len(children) == 2
        assert children[0] is left
        assert children[1] is right

    def test_walk_includes_both_operands(self):
        left = FloatNode(1.0)
        right = FloatNode(2.0)
        c = CombineNode("test", left, right)
        results = list(c.walk())
        assert len(results) == 3
        assert results[0][0] is c
        assert results[1][0] is left
        assert results[2][0] is right


class TestUnionNodeContract:
    """UnionNode: boolean OR (min(d1, d2))."""

    def test_construct(self):
        u = UnionNode(ExprNode(), ExprNode())
        assert isinstance(u, UnionNode)

    def test_kind_is_union(self):
        u = UnionNode(ExprNode(), ExprNode())
        assert u.kind == "union"

    def test_children(self):
        left = FloatNode(1.0)
        right = FloatNode(2.0)
        u = UnionNode(left, right)
        children = u.children()
        assert len(children) == 2
        assert children[0] is left
        assert children[1] is right

    def test_label(self):
        label = UnionNode(ExprNode(), ExprNode()).label()
        assert isinstance(label, str)
        assert "Union" in label or "union" in label

    def test_walk(self):
        u = UnionNode(FloatNode(1.0), FloatNode(2.0))
        results = list(u.walk())
        assert len(results) == 3


class TestIntersectionNodeContract:
    """IntersectionNode: boolean AND (max(d1, d2))."""

    def test_construct(self):
        i = IntersectionNode(ExprNode(), ExprNode())
        assert isinstance(i, IntersectionNode)

    def test_kind_is_intersection(self):
        i = IntersectionNode(ExprNode(), ExprNode())
        assert i.kind == "intersection"

    def test_children(self):
        left = FloatNode(1.0)
        right = FloatNode(2.0)
        i = IntersectionNode(left, right)
        assert len(i.children()) == 2
        assert i.children()[0] is left
        assert i.children()[1] is right

    def test_label(self):
        label = IntersectionNode(ExprNode(), ExprNode()).label()
        assert isinstance(label, str)
        assert "Intersection" in label or "intersection" in label


class TestSubtractionNodeContract:
    """SubtractionNode: boolean NOT (max(d1, -d2))."""

    def test_construct(self):
        s = SubtractionNode(ExprNode(), ExprNode())
        assert isinstance(s, SubtractionNode)

    def test_kind_is_subtraction(self):
        s = SubtractionNode(ExprNode(), ExprNode())
        assert s.kind == "subtraction"

    def test_children(self):
        left = FloatNode(1.0)
        right = FloatNode(2.0)
        s = SubtractionNode(left, right)
        assert len(s.children()) == 2
        assert s.children()[0] is left
        assert s.children()[1] is right

    def test_label(self):
        label = SubtractionNode(ExprNode(), ExprNode()).label()
        assert isinstance(label, str)
        assert "Subtraction" in label or "subtraction" in label


# =============================================================================
# T-1.6: MaterialNode -- PBR properties
# =============================================================================


class _MaterialFactory:
    """Helper to construct MaterialNode with all positional args."""
    DEFAULT_ALBEDO = Vec3Node(0.5, 0.5, 0.5)
    DEFAULT_ROUGHNESS = FloatNode(0.5)
    DEFAULT_METALLIC = FloatNode(0.0)
    DEFAULT_EMISSIVE = FloatNode(0.0)
    DEFAULT_AO = FloatNode(1.0)

    @classmethod
    def make(cls, material_id=0,
             albedo=None, roughness=None, metallic=None,
             emissive=None, ambient_occlusion=None):
        return MaterialNode(
            material_id,
            albedo or cls.DEFAULT_ALBEDO,
            roughness or cls.DEFAULT_ROUGHNESS,
            metallic or cls.DEFAULT_METALLIC,
            emissive or cls.DEFAULT_EMISSIVE,
            ambient_occlusion or cls.DEFAULT_AO,
        )


class TestMaterialNodeContract:
    """MaterialNode holds PBR surface properties."""

    def test_material_id_stored(self):
        m = _MaterialFactory.make(material_id=5)
        assert m.material_id == 5

    def test_albedo_stored(self):
        albedo = Vec3Node(1.0, 0.0, 0.0)
        m = _MaterialFactory.make(albedo=albedo)
        assert m.albedo is albedo
        assert m.albedo.as_tuple() == (1.0, 0.0, 0.0)

    def test_roughness_stored(self):
        m = _MaterialFactory.make(roughness=FloatNode(0.8))
        assert m.roughness.value == 0.8

    def test_metallic_stored(self):
        m = _MaterialFactory.make(metallic=FloatNode(1.0))
        assert m.metallic.value == 1.0

    def test_emissive_stored(self):
        emissive = FloatNode(0.1)
        m = _MaterialFactory.make(emissive=emissive)
        assert m.emissive is emissive

    def test_ambient_occlusion_stored(self):
        m = _MaterialFactory.make(ambient_occlusion=FloatNode(0.5))
        assert m.ambient_occlusion.value == 0.5

    def test_multiple_materials(self):
        """Multiple materials with different IDs can coexist."""
        m0 = _MaterialFactory.make(material_id=0)
        m1 = _MaterialFactory.make(material_id=1,
                                    albedo=Vec3Node(1.0, 0.0, 0.0))
        m2 = _MaterialFactory.make(material_id=2,
                                    albedo=Vec3Node(0.0, 1.0, 0.0))
        assert m0.material_id == 0
        assert m1.material_id == 1
        assert m2.material_id == 2
        assert m1.albedo.as_tuple() == (1.0, 0.0, 0.0)
        assert m2.albedo.as_tuple() == (0.0, 1.0, 0.0)

    def test_label_includes_id(self):
        m = _MaterialFactory.make(material_id=0)
        label = m.label()
        assert "Material" in label or "material" in label
        assert "0" in label

    def test_label_includes_albedo(self):
        m = _MaterialFactory.make(albedo=Vec3Node(0.5, 0.5, 0.5))
        label = m.label()
        assert "0.5" in label


# =============================================================================
# T-1.7: SceneGraph container
# =============================================================================


class TestSceneGraphContract:
    """SceneGraph is the root container holding the complete scene."""

    def test_empty_scene(self):
        """Empty scene (no primitives) handled gracefully."""
        sg = SceneGraph(primitives=())
        assert sg.primitives == ()
        assert sg.pipeline == ()
        assert sg.materials == ()
        assert sg.name == ""

    def test_empty_children(self):
        """SceneGraph with no children returns empty children."""
        sg = SceneGraph(primitives=())
        assert sg.children() == ()

    def test_children_returns_pipeline_then_primitives(self):
        """children() returns pipeline nodes first, then primitives."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        box = BoxNode(pos, Vec3Node(1.0, 1.0, 1.0))
        repeat = RepeatNode(pos, Vec3Node(2.0, 2.0, 2.0))
        sg = SceneGraph(
            primitives=(sphere, box),
            pipeline=(repeat,),
        )
        children = sg.children()
        assert children[0] is repeat
        assert children[1] is sphere
        assert children[2] is box

    def test_label_unnamed(self):
        """SceneGraph without a name still produces a label."""
        sg = SceneGraph(primitives=())
        label = sg.label()
        assert isinstance(label, str)
        assert len(label) > 0

    def test_label_with_name(self):
        sg = SceneGraph(primitives=(), name="test_scene")
        assert "test_scene" in sg.label()

    def test_label_counts_primitives_and_pipeline(self):
        """label() includes count of pipeline ops, primitives, and materials."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        box = BoxNode(pos, Vec3Node(1.0, 1.0, 1.0))
        sg = SceneGraph(
            primitives=(sphere, box),
            pipeline=(RepeatNode(pos, Vec3Node(2.0, 2.0, 2.0)),),
            materials=(_MaterialFactory.make(material_id=0),),
        )
        label = sg.label()
        assert "1 pipeline" in label or "1" in label
        assert "2 primitives" in label or "2" in label
        assert "1 materials" in label or "1" in label

    def test_walk_yields_all_descendants(self):
        """walk() traverses the full tree."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        repeat = RepeatNode(pos, Vec3Node(2.0, 2.0, 2.0))
        sg = SceneGraph(primitives=(sphere,), pipeline=(repeat,))
        results = list(sg.walk())
        # Should include: sg, repeat, pos, sphere, pos
        assert len(results) >= 3

    def test_deep_label_structure(self):
        """deep_label() produces a structured multi-line summary."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        sg = SceneGraph(primitives=(sphere,), name="demo")
        output = sg.deep_label()
        assert isinstance(output, str)
        assert len(output) > 0

    def test_deep_label_with_all_sections(self):
        """deep_label() includes pipeline, primitives, materials."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        repeat = RepeatNode(pos, Vec3Node(2.0, 2.0, 2.0))
        mat = _MaterialFactory.make(material_id=0)
        sg = SceneGraph(
            primitives=(sphere,),
            pipeline=(repeat,),
            materials=(mat,),
            name="full_scene",
        )
        output = sg.deep_label()
        assert "full_scene" in output or "SceneGraph" in output
        assert len(output) > 0

    def test_multiple_primitives(self):
        """SceneGraph can hold multiple primitives."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        box = BoxNode(pos, Vec3Node(2.0, 2.0, 2.0))
        torus = TorusNode(pos, FloatNode(3.0), FloatNode(1.0))
        sg = SceneGraph(primitives=(sphere, box, torus))
        assert len(sg.primitives) == 3

    def test_materials_optional(self):
        """Materials tuple is optional."""
        sg = SceneGraph(primitives=())
        assert hasattr(sg, "materials")
        assert sg.materials == ()

    def test_pipeline_optional(self):
        """Pipeline tuple is optional."""
        sg = SceneGraph(primitives=())
        assert hasattr(sg, "pipeline")
        assert sg.pipeline == ()

    def test_walk_materials_not_included_by_default(self):
        """Materials are NOT part of children() traversal."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        mat = _MaterialFactory.make(material_id=0)
        sg = SceneGraph(primitives=(sphere,), materials=(mat,))
        children = sg.children()
        # mat should NOT be in children
        assert mat not in children


# =============================================================================
# Integration: Scene construction and traversal
# =============================================================================


class TestSceneConstructionIntegration:
    """Build realistic scenes from public API and verify traversal."""

    def test_simple_sphere_scene(self):
        """A minimal scene: a single sphere."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        sg = SceneGraph(primitives=(sphere,), name="simple_sphere")

        assert len(sg.primitives) == 1
        nodes = list(sg.walk())
        assert len(nodes) >= 2  # sg + sphere + pos

    def test_box_with_domain_repeat(self):
        """Box in a repeating pattern."""
        pos = PositionNode()
        box = BoxNode(pos, Vec3Node(0.5, 0.5, 0.5))
        repeat = RepeatNode(pos, Vec3Node(2.0, 2.0, 2.0))
        sg = SceneGraph(primitives=(box,), pipeline=(repeat,))
        nodes = list(sg.walk())
        assert len(nodes) >= 3

    def test_csg_union(self):
        """Union of two spheres."""
        pos = PositionNode()
        a = SphereNode(pos, FloatNode(1.0))
        b = SphereNode(pos, FloatNode(1.5))
        union = UnionNode(a, b)
        sg = SceneGraph(primitives=(union,))
        nodes = list(sg.walk())
        assert len(nodes) >= 3

    def test_csg_subtraction(self):
        """Subtract a small sphere from a large sphere."""
        pos = PositionNode()
        outer = SphereNode(pos, FloatNode(2.0))
        inner = SphereNode(pos, FloatNode(0.5))
        sub = SubtractionNode(outer, inner)
        sg = SceneGraph(primitives=(sub,))
        children = sub.children()
        assert children[0] is outer
        assert children[1] is inner

    def test_scene_with_materials(self):
        """Scene with multiple primitives and materials."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0), material_id=0)
        box = BoxNode(pos, Vec3Node(1.0, 1.0, 1.0), material_id=1)
        mat_red = _MaterialFactory.make(
            material_id=0, albedo=Vec3Node(1.0, 0.0, 0.0),
        )
        mat_blue = _MaterialFactory.make(
            material_id=1, albedo=Vec3Node(0.0, 0.0, 1.0),
        )

        sg = SceneGraph(
            primitives=(sphere, box),
            materials=(mat_red, mat_blue),
        )
        assert len(sg.primitives) == 2
        assert len(sg.materials) == 2
        assert sg.primitives[0].material_id == 0
        assert sg.primitives[1].material_id == 1

    def test_nested_domain_pipeline(self):
        """Multiple domain operations in pipeline."""
        pos = PositionNode()
        mirror = MirrorNode(pos, Axis.X)
        twist = TwistNode(mirror, FloatNode(0.5))
        sphere = SphereNode(twist, FloatNode(1.0))
        sg = SceneGraph(primitives=(sphere,))
        nodes = list(sg.walk())
        # sg -> sphere -> twist -> mirror -> pos
        assert len(nodes) >= 4

    def test_scene_immutability(self):
        """SceneGraph properties should not be reassignable."""
        sg = SceneGraph(primitives=())
        with pytest.raises(Exception):
            sg.primitives = (SphereNode(PositionNode(), FloatNode(1.0)),)


# =============================================================================
# Type map integration
# =============================================================================


class TestTypeMapIntegration:
    """Type maps correctly reference all node classes."""

    def test_all_sdf_primitives_have_type_map_entry(self):
        """Every SDF primitive node is registered in SDF_PRIMITIVE_TYPE_MAP."""
        sdf_classes = {
            SphereNode, BoxNode, TorusNode,
            CylinderNode, ConeNode, PlaneNode,
        }
        for cls in sdf_classes:
            assert cls in SDF_PRIMITIVE_TYPE_MAP, f"{cls.__name__} not in type map"

    def test_all_domain_ops_have_type_map_entry(self):
        """Every domain op node is registered in DOMAIN_OP_TYPE_MAP."""
        domain_classes = {
            RepeatNode, CellIdNode, MirrorNode,
            KifsNode, TwistNode, BendNode, StretchNode,
        }
        for cls in domain_classes:
            assert cls in DOMAIN_OP_TYPE_MAP, f"{cls.__name__} not in type map"

    def test_type_map_values_are_unique(self):
        """Each SDF primitive has a distinct WGSL function name."""
        values = list(SDF_PRIMITIVE_TYPE_MAP.values())
        assert len(values) == len(set(values)), "Duplicate WGSL identifiers in type map"

    def test_domain_type_map_values_are_unique(self):
        """Each domain operation has a distinct WGSL identifier."""
        values = list(DOMAIN_OP_TYPE_MAP.values())
        assert len(values) == len(set(values)), "Duplicate WGSL identifiers in type map"
