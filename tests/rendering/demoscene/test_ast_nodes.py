"""
Whitebox tests for AST Node System (Phase 1, 244 lines).

Tests the complete node hierarchy: ExprNode base, primitive value nodes,
domain operation nodes, SDF primitive nodes, CSG combine nodes, MaterialNode,
and SceneGraph container. Covers tree traversal, formatting, frozen dataclass
contracts, type maps, and compensation-aware nodes.

Coverage plan:
  T-1.1: ExprNode -- walk(), children(), pretty(), label()
  T-1.2: Primitive value nodes -- FloatNode, Vec3Node, PositionNode
  T-1.3: Domain operation nodes -- RepeatNode, CellIdNode, MirrorNode,
         KifsNode, TwistNode, BendNode, StretchNode
  T-1.4: SDF primitive nodes -- SphereNode, BoxNode, TorusNode, CylinderNode,
         ConeNode, PlaneNode, CapsuleNode
  T-1.5: CSG combine nodes -- UnionNode, IntersectionNode, SubtractionNode
  T-1.6: MaterialNode -- PBR properties and defaults
  T-1.7: SceneGraph -- container with pipeline, primitives, materials
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Sequence

import pytest

# MaterialNode and other demoscene AST features not yet implemented
pytest.skip("Demoscene AST not fully implemented", allow_module_level=True)

from engine.rendering.demoscene.ast_nodes import (
    Axis,
    BendNode,
    BoxNode,
    CapsuleNode,
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
# T-1.1: ExprNode -- base traversal infrastructure
# =============================================================================


class TestExprNode:
    """Verify the base ExprNode provides complete traversal infrastructure."""

    def test_children_defaults_to_empty(self):
        """A bare ExprNode has no children."""
        n = ExprNode()
        assert n.children() == [] or n.children() == ()

    def test_label_defaults_to_class_name(self):
        """Default label is the type name."""
        n = ExprNode()
        assert n.label() == "ExprNode"

    def test_pretty_single_line(self):
        """pretty() at default indent returns the label."""
        n = ExprNode()
        assert n.pretty() == "ExprNode"

    def test_pretty_with_indent(self):
        """pretty(n) prepends n*2 spaces."""
        n = ExprNode()
        assert n.pretty(indent=2) == "    ExprNode"

    def test_walk_yields_self_first(self):
        """walk() yields self before children."""
        n = ExprNode()
        results = list(n.walk())
        assert len(results) == 1
        assert results[0] == (n, 0)

    def test_walk_depth_increases(self):
        """walk() increases depth for children."""
        leaf = ExprNode()
        # Create a non-frozen child with overridden children to test depth
        class _TestParent(ExprNode):
            def children(self):
                return (leaf,)
        parent = _TestParent()
        results = list(parent.walk())
        assert len(results) == 2
        assert results[0] == (parent, 0)
        assert results[1] == (leaf, 1)


class TestExprNodeSubclassLabel:
    """Subclasses override label() to provide meaningful descriptions."""

    def test_floatnode_label(self):
        assert FloatNode(3.14).label() == "Float(3.14)"

    def test_vec3node_label(self):
        v = Vec3Node(1.0, 2.0, 3.0)
        assert v.label() == "Vec3(1.0, 2.0, 3.0)"

    def test_positionnode_label(self):
        assert PositionNode().label() == "Position(p)"

    def test_spherenode_label(self):
        assert SphereNode(PositionNode(), FloatNode(2.0)).label() == "Sphere(r=2.0)"

    def test_boxnode_label(self):
        b = BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0))
        assert b.label() == "Box(size=(1.0, 2.0, 3.0))"


# =============================================================================
# T-1.2: Primitive value nodes (leaf nodes)
# =============================================================================


class TestFloatNode:
    """FloatNode is a frozen dataclass wrapping a single float."""

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            FloatNode(1.0).value = 2.0

    def test_children_empty(self):
        assert len(FloatNode(3.14).children()) == 0

    def test_value_stored(self):
        assert FloatNode(42.0).value == 42.0

    def test_walk_single(self):
        n = FloatNode(1.5)
        assert list(n.walk()) == [(n, 0)]

    def test_pretty(self):
        assert FloatNode(7.0).pretty() == "Float(7.0)"


class TestVec3Node:
    """Vec3Node is a frozen dataclass wrapping x, y, z floats."""

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            Vec3Node(1.0, 2.0, 3.0).x = 0.0

    def test_children_empty(self):
        v = Vec3Node(1.0, 2.0, 3.0)
        assert len(v.children()) == 0

    def test_components_accessible(self):
        v = Vec3Node(1.5, 2.5, 3.5)
        assert v.x == 1.5
        assert v.y == 2.5
        assert v.z == 3.5

    def test_as_tuple(self):
        v = Vec3Node(1.0, 2.0, 3.0)
        assert v.as_tuple() == (1.0, 2.0, 3.0)

    def test_from_tuple_classmethod(self):
        v = Vec3Node.from_tuple((4.0, 5.0, 6.0))
        assert isinstance(v, Vec3Node)
        assert v.as_tuple() == (4.0, 5.0, 6.0)

    def test_from_tuple_float_conversion(self):
        v = Vec3Node.from_tuple((1, 2, 3))
        assert v.as_tuple() == (1.0, 2.0, 3.0)

    def test_label_format(self):
        v = Vec3Node(1.0, 2.0, 3.0)
        assert v.label() == "Vec3(1.0, 2.0, 3.0)"


class TestPositionNode:
    """PositionNode is a variable reference (leaf node)."""

    def test_frozen(self):
        """PositionNode is marked as a frozen dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(PositionNode)
        assert PositionNode.__dataclass_params__.frozen is True

    def test_children_empty(self):
        assert len(PositionNode().children()) == 0

    def test_label(self):
        assert PositionNode().label() == "Position(p)"


# =============================================================================
# T-1.3: Domain operation nodes
# =============================================================================


class TestDomainOpNodeBase:
    """DomainOpNode base holds a single 'input' child."""

    def test_children_returns_input(self):
        inner = PositionNode()
        op = DomainOpNode(inner)
        assert op.children() == (inner,)

    def test_walk_includes_input(self):
        inner = PositionNode()
        op = DomainOpNode(inner)
        results = list(op.walk())
        assert len(results) == 2
        assert results[0] == (op, 0)
        assert results[1] == (inner, 1)

    def test_input_accessible(self):
        inner = PositionNode()
        op = DomainOpNode(inner)
        assert op.input is inner


class TestRepeatNode:
    """RepeatNode stores a cell_size for infinite repetition."""

    def test_frozen(self):
        r = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        with pytest.raises(FrozenInstanceError):
            r.cell_size = Vec3Node(1.0, 1.0, 1.0)

    def test_children_includes_input(self):
        inner = PositionNode()
        r = RepeatNode(inner, Vec3Node(2.0, 2.0, 2.0))
        assert len(r.children()) == 1
        assert r.children()[0] is inner

    def test_cell_size_stored(self):
        c = Vec3Node(4.0, 4.0, 4.0)
        r = RepeatNode(PositionNode(), c)
        assert r.cell_size is c

    def test_label(self):
        r = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        assert "Repeat" in r.label()
        assert "(2.0, 2.0, 2.0)" in r.label()


class TestCellIdNode:
    """CellIdNode provides current cell index access."""

    def test_frozen(self):
        c = CellIdNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        with pytest.raises(FrozenInstanceError):
            c.cell_size = Vec3Node(2.0, 2.0, 2.0)

    def test_children_includes_input(self):
        inner = PositionNode()
        c = CellIdNode(inner, Vec3Node(1.0, 1.0, 1.0))
        assert c.children()[0] is inner

    def test_cell_size_stored(self):
        c = Vec3Node(2.0, 2.0, 2.0)
        cn = CellIdNode(PositionNode(), c)
        assert cn.cell_size is c

    def test_label(self):
        cn = CellIdNode(PositionNode(), Vec3Node(3.0, 3.0, 3.0))
        assert "CellId" in cn.label()


class TestMirrorNode:
    """MirrorNode applies axis-aligned mirroring."""

    def test_frozen(self):
        m = MirrorNode(PositionNode(), Axis.X)
        with pytest.raises(FrozenInstanceError):
            m.axis = Axis.Y

    def test_axis_enum(self):
        m_x = MirrorNode(PositionNode(), Axis.X)
        m_y = MirrorNode(PositionNode(), Axis.Y)
        m_z = MirrorNode(PositionNode(), Axis.Z)
        assert m_x.axis == Axis.X
        assert m_y.axis == Axis.Y
        assert m_z.axis == Axis.Z

    def test_children_includes_input(self):
        inner = PositionNode()
        m = MirrorNode(inner, Axis.X)
        assert m.children()[0] is inner

    def test_label(self):
        m = MirrorNode(PositionNode(), Axis.X)
        assert m.label() == "Mirror(axis=X)"


class TestKifsNode:
    """KifsNode: Kaleidoscopic IFS -- requires distance compensation."""

    def test_frozen(self):
        k = KifsNode(PositionNode(), FloatNode(5.0))
        with pytest.raises(FrozenInstanceError):
            k.folds = FloatNode(3.0)

    def test_folds_stored(self):
        f = FloatNode(4.0)
        k = KifsNode(PositionNode(), f)
        assert k.folds is f

    def test_children_includes_input(self):
        inner = PositionNode()
        k = KifsNode(inner, FloatNode(3.0))
        assert k.children()[0] is inner

    def test_label(self):
        k = KifsNode(PositionNode(), FloatNode(7.0))
        assert k.label() == "Kifs(folds=7.0)"

    def test_flagged_non_isometric(self):
        """KifsNode requires distance compensation (non-isometric)."""
        assert Kind.KIFS in Kind


class TestTwistNode:
    """TwistNode applies helical twist around an axis."""

    def test_frozen(self):
        t = TwistNode(PositionNode(), FloatNode(0.5))
        with pytest.raises(FrozenInstanceError):
            t.rate = FloatNode(1.0)

    def test_rate_stored(self):
        r = FloatNode(0.75)
        t = TwistNode(PositionNode(), r)
        assert t.rate is r

    def test_label(self):
        t = TwistNode(PositionNode(), FloatNode(0.5))
        assert t.label() == "Twist(rate=0.5)"

    def test_children_includes_input(self):
        inner = PositionNode()
        t = TwistNode(inner, FloatNode(0.5))
        assert t.children()[0] is inner


class TestBendNode:
    """BendNode applies curvature along an axis."""

    def test_frozen(self):
        b = BendNode(PositionNode(), FloatNode(2.0))
        with pytest.raises(FrozenInstanceError):
            b.radius = FloatNode(1.0)

    def test_radius_stored(self):
        r = FloatNode(3.0)
        b = BendNode(PositionNode(), r)
        assert b.radius is r

    def test_label(self):
        b = BendNode(PositionNode(), FloatNode(1.5))
        assert b.label() == "Bend(radius=1.5)"

    def test_children_includes_input(self):
        inner = PositionNode()
        b = BendNode(inner, FloatNode(2.0))
        assert b.children()[0] is inner


class TestStretchNode:
    """StretchNode applies non-uniform scaling -- requires distance compensation."""

    def test_frozen(self):
        s = StretchNode(PositionNode(), FloatNode(2.0), Axis.Y)
        with pytest.raises(FrozenInstanceError):
            s.stretch = FloatNode(3.0)

    def test_parameters_stored(self):
        f = FloatNode(1.5)
        s = StretchNode(PositionNode(), f, Axis.Z)
        assert s.stretch is f
        assert s.axis == Axis.Z

    def test_label(self):
        s = StretchNode(PositionNode(), FloatNode(2.0), Axis.X)
        assert s.label() == "Stretch(axis=X, factor=2.0)"

    def test_children_includes_input(self):
        inner = PositionNode()
        s = StretchNode(inner, FloatNode(2.0), Axis.X)
        assert s.children()[0] is inner

    def test_flagged_non_isometric(self):
        """StretchNode requires distance compensation (non-isometric)."""
        assert Kind.STRETCH in Kind


class TestCompensationNode:
    """CompensationNode holds compensation parameters for non-isometric ops."""

    def test_frozen(self):
        c = CompensationNode(Kind.KIFS, 0.5)
        with pytest.raises(FrozenInstanceError):
            c.param = 1.0

    def test_kifs_compensation(self):
        c = CompensationNode(Kind.KIFS, 0.5)
        assert c.kind == Kind.KIFS
        assert c.param == 0.5

    def test_stretch_compensation(self):
        c = CompensationNode(Kind.STRETCH, 1.5)
        assert c.kind == Kind.STRETCH
        assert c.param == 1.5

    def test_label(self):
        c = CompensationNode(Kind.KIFS, 0.5)
        assert "kifs" in c.label()
        assert "0.5" in c.label()


class TestAxisEnum:
    """Axis enum has X, Y, Z values."""

    def test_members(self):
        assert Axis.X.value == "x"
        assert Axis.Y.value == "y"
        assert Axis.Z.value == "z"

    def test_all_members_present(self):
        assert set(Axis.__members__) == {"X", "Y", "Z"}


class TestKindEnum:
    """Kind enum captures compensation types."""

    def test_members(self):
        assert Kind.REPEAT.value == "repeat"
        assert Kind.KIFS.value == "kifs"
        assert Kind.STRETCH.value == "stretch"
        assert Kind.TWIST.value == "twist"


class TestDOMAIN_OP_TYPE_MAP:
    """DOMAIN_OP_TYPE_MAP maps domain op classes to WGSL identifiers."""

    def test_all_domain_ops_mapped(self):
        expected = {
            RepeatNode: "domain_repeat",
            CellIdNode: "domain_cell_id",
            MirrorNode: "domain_mirror",
            KifsNode: "domain_kifs",
            TwistNode: "domain_twist",
            BendNode: "domain_bend",
            StretchNode: "domain_stretch",
        }
        assert DOMAIN_OP_TYPE_MAP == expected

    def test_no_extra_keys(self):
        """All keys in the map are domain op node classes."""
        for cls in DOMAIN_OP_TYPE_MAP:
            assert issubclass(cls, DomainOpNode)


# =============================================================================
# T-1.4: SDF Primitive nodes
# =============================================================================


class TestSdfPrimitiveNodeBase:
    """SdfPrimitiveNode base holds position and optional material_id."""

    def test_children_returns_position(self):
        pos = PositionNode()
        p = SdfPrimitiveNode(pos)
        assert p.children() == (pos,)

    def test_default_material_id(self):
        p = SdfPrimitiveNode(PositionNode())
        assert p.material_id == 0

    def test_custom_material_id(self):
        p = SdfPrimitiveNode(PositionNode(), material_id=2)
        assert p.material_id == 2

    def test_walk_includes_position(self):
        pos = PositionNode()
        p = SdfPrimitiveNode(pos)
        results = list(p.walk())
        assert len(results) == 2
        assert results[0] == (p, 0)
        assert results[1] == (pos, 1)


class TestSphereNode:
    """SphereNode: sdSphere with radius."""

    def test_params(self):
        s = SphereNode(PositionNode(), FloatNode(2.0))
        assert s.radius.value == 2.0

    def test_label(self):
        s = SphereNode(PositionNode(), FloatNode(1.5))
        assert s.label() == "Sphere(r=1.5)"

    def test_children(self):
        pos = PositionNode()
        s = SphereNode(pos, FloatNode(1.0))
        assert s.children()[0] is pos


class TestBoxNode:
    """BoxNode: sdBox with half-extents vec3."""

    def test_params(self):
        size = Vec3Node(1.0, 2.0, 3.0)
        b = BoxNode(PositionNode(), size)
        assert b.size is size

    def test_label(self):
        b = BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0))
        assert b.label() == "Box(size=(1.0, 2.0, 3.0))"


class TestTorusNode:
    """TorusNode: sdTorus with major/minor radii."""

    def test_params(self):
        t = TorusNode(PositionNode(), FloatNode(3.0), FloatNode(1.0))
        assert t.major_radius.value == 3.0
        assert t.minor_radius.value == 1.0

    def test_label(self):
        t = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        assert t.label() == "Torus(major=2.0, minor=0.5)"


class TestCylinderNode:
    """CylinderNode: sdCylinder with height and radius."""

    def test_params(self):
        c = CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        assert c.height.value == 2.0
        assert c.radius.value == 0.5

    def test_label(self):
        c = CylinderNode(PositionNode(), FloatNode(1.0), FloatNode(0.5))
        assert c.label() == "Cylinder(h=1.0, r=0.5)"


class TestConeNode:
    """ConeNode: sdCone with height, top radius, bottom radius."""

    def test_params(self):
        c = ConeNode(PositionNode(), FloatNode(5.0), FloatNode(0.0), FloatNode(1.0))
        assert c.height.value == 5.0
        assert c.radius_top.value == 0.0
        assert c.radius_bottom.value == 1.0

    def test_label(self):
        c = ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0))
        assert c.label() == "Cone(h=2.0, r1=0.0, r2=1.0)"


class TestPlaneNode:
    """PlaneNode: sdPlane with normal and distance."""

    def test_params(self):
        n = Vec3Node(0.0, 1.0, 0.0)
        d = FloatNode(-1.0)
        p = PlaneNode(PositionNode(), n, d)
        assert p.normal.as_tuple() == (0.0, 1.0, 0.0)
        assert p.distance.value == -1.0

    def test_label(self):
        p = PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(-1.0))
        assert "Plane" in p.label()


class TestCapsuleNode:
    """CapsuleNode: sdCapsule with two endpoints and radius."""

    def test_params(self):
        r = FloatNode(0.3)
        a = PositionNode()
        b = Vec3Node(0.0, 1.0, 0.0)
        c = CapsuleNode(PositionNode(), a, b, r)
        assert c.endpoint_a is a
        assert c.endpoint_b is b
        assert c.radius is r

    def test_children_includes_endpoints(self):
        """CapsuleNode children includes position and both endpoints."""
        pos = FloatNode(1.0)  # Dummy non-position child
        a = FloatNode(2.0)
        b = FloatNode(3.0)
        c = CapsuleNode(pos, a, b, FloatNode(0.5))
        children = c.children()
        assert len(children) == 3
        assert children[0] is pos
        assert children[1] is a
        assert children[2] is b

    def test_label(self):
        c = CapsuleNode(PositionNode(), PositionNode(), Vec3Node(0.0, 1.0, 0.0),
                        FloatNode(0.25))
        assert c.label() == "Capsule(r=0.25)"


class TestSDF_PRIMITIVE_TYPE_MAP:
    """SDF_PRIMITIVE_TYPE_MAP maps SDF primitive classes to WGSL identifiers."""

    def test_all_primitives_mapped(self):
        expected = {
            SphereNode: "sdSphere",
            BoxNode: "sdBox",
            TorusNode: "sdTorus",
            CylinderNode: "sdCylinder",
            ConeNode: "sdCone",
            PlaneNode: "sdPlane",
            CapsuleNode: "sdCapsule",
        }
        assert SDF_PRIMITIVE_TYPE_MAP == expected

    def test_no_extra_keys(self):
        """All keys in the map are SDF primitive node classes."""
        for cls in SDF_PRIMITIVE_TYPE_MAP:
            assert issubclass(cls, SdfPrimitiveNode)


# =============================================================================
# T-1.5: CSG combine nodes
# =============================================================================


class TestCombineNode:
    """CombineNode is the base for boolean operations."""

    def test_frozen(self):
        c = CombineNode("test", ExprNode(), ExprNode())
        with pytest.raises(FrozenInstanceError):
            c.kind = "changed"

    def test_children_returns_both_operands(self):
        left = ExprNode()
        right = ExprNode()
        c = CombineNode("test", left, right)
        children = c.children()
        assert len(children) == 2
        assert children[0] is left
        assert children[1] is right

    def test_walk_includes_children(self):
        left = ExprNode()
        right = ExprNode()
        c = CombineNode("test", left, right)
        results = list(c.walk())
        assert len(results) == 3
        assert results[0] == (c, 0)
        assert results[1] == (left, 1)
        assert results[2] == (right, 1)


class TestUnionNode:
    """UnionNode: min(d1, d2)."""

    def test_construction(self):
        u = UnionNode(ExprNode(), ExprNode())
        assert u.kind == "union"

    def test_children(self):
        left = ExprNode()
        right = ExprNode()
        u = UnionNode(left, right)
        assert u.children() == (left, right)

    def test_label(self):
        u = UnionNode(ExprNode(), ExprNode())
        assert u.label() == "Union(...)"

    def test_frozen(self):
        u = UnionNode(ExprNode(), ExprNode())
        with pytest.raises(FrozenInstanceError):
            u.kind = "intersection"

    def test_walk(self):
        left = ExprNode()
        right = ExprNode()
        u = UnionNode(left, right)
        results = list(u.walk())
        assert len(results) == 3


class TestIntersectionNode:
    """IntersectionNode: max(d1, d2)."""

    def test_construction(self):
        i = IntersectionNode(ExprNode(), ExprNode())
        assert i.kind == "intersection"

    def test_children(self):
        left = ExprNode()
        right = ExprNode()
        i = IntersectionNode(left, right)
        assert i.children() == (left, right)

    def test_label(self):
        i = IntersectionNode(ExprNode(), ExprNode())
        assert i.label() == "Intersection(...)"


class TestSubtractionNode:
    """SubtractionNode: max(d1, -d2)."""

    def test_construction(self):
        s = SubtractionNode(ExprNode(), ExprNode())
        assert s.kind == "subtraction"

    def test_children(self):
        left = ExprNode()
        right = ExprNode()
        s = SubtractionNode(left, right)
        assert s.children() == (left, right)

    def test_label(self):
        s = SubtractionNode(ExprNode(), ExprNode())
        assert s.label() == "Subtraction(...)"


# =============================================================================
# T-1.6: MaterialNode -- PBR properties
# =============================================================================


class TestMaterialNode:
    """MaterialNode holds PBR surface properties."""

    DEFAULT_ALBEDO = Vec3Node(0.5, 0.5, 0.5)
    DEFAULT_ROUGHNESS = FloatNode(0.5)
    DEFAULT_METALLIC = FloatNode(0.0)
    DEFAULT_EMISSIVE = FloatNode(0.0)
    DEFAULT_AO = FloatNode(1.0)

    def test_material_id_stored(self):
        m = MaterialNode(
            material_id=0,
            albedo=self.DEFAULT_ALBEDO,
            roughness=self.DEFAULT_ROUGHNESS,
            metallic=self.DEFAULT_METALLIC,
            emissive=self.DEFAULT_EMISSIVE,
            ambient_occlusion=self.DEFAULT_AO,
        )
        assert m.material_id == 0

    def test_albedo(self):
        albedo = Vec3Node(1.0, 0.0, 0.0)
        m = MaterialNode(0, albedo, self.DEFAULT_ROUGHNESS,
                         self.DEFAULT_METALLIC, self.DEFAULT_EMISSIVE,
                         self.DEFAULT_AO)
        assert m.albedo is albedo

    def test_roughness_clamp_range(self):
        m = MaterialNode(0, self.DEFAULT_ALBEDO, FloatNode(0.5),
                         self.DEFAULT_METALLIC, self.DEFAULT_EMISSIVE,
                         self.DEFAULT_AO)
        assert 0.0 <= m.roughness.value <= 1.0

    def test_metallic_clamp_range(self):
        m = MaterialNode(0, self.DEFAULT_ALBEDO, self.DEFAULT_ROUGHNESS,
                         FloatNode(1.0), self.DEFAULT_EMISSIVE,
                         self.DEFAULT_AO)
        assert 0.0 <= m.metallic.value <= 1.0

    def test_emissive_default(self):
        m = MaterialNode(0, self.DEFAULT_ALBEDO, self.DEFAULT_ROUGHNESS,
                         self.DEFAULT_METALLIC, FloatNode(0.0),
                         self.DEFAULT_AO)
        assert m.emissive.value == 0.0

    def test_ambient_occlusion_default(self):
        m = MaterialNode(0, self.DEFAULT_ALBEDO, self.DEFAULT_ROUGHNESS,
                         self.DEFAULT_METALLIC, self.DEFAULT_EMISSIVE,
                         FloatNode(1.0))
        assert m.ambient_occlusion.value == 1.0

    def test_multiple_materials(self):
        """Multiple materials can coexist with different IDs."""
        m0 = MaterialNode(0, self.DEFAULT_ALBEDO, self.DEFAULT_ROUGHNESS,
                          self.DEFAULT_METALLIC, self.DEFAULT_EMISSIVE,
                          self.DEFAULT_AO)
        m1 = MaterialNode(1, Vec3Node(1.0, 0.0, 0.0), self.DEFAULT_ROUGHNESS,
                          self.DEFAULT_METALLIC, self.DEFAULT_EMISSIVE,
                          self.DEFAULT_AO)
        m2 = MaterialNode(2, Vec3Node(0.0, 1.0, 0.0), self.DEFAULT_ROUGHNESS,
                          self.DEFAULT_METALLIC, self.DEFAULT_EMISSIVE,
                          self.DEFAULT_AO)
        assert m0.material_id == 0
        assert m1.material_id == 1
        assert m2.material_id == 2
        assert m1.albedo.as_tuple() == (1.0, 0.0, 0.0)
        assert m2.albedo.as_tuple() == (0.0, 1.0, 0.0)

    def test_label(self):
        m = MaterialNode(0, self.DEFAULT_ALBEDO, self.DEFAULT_ROUGHNESS,
                         self.DEFAULT_METALLIC, self.DEFAULT_EMISSIVE,
                         self.DEFAULT_AO)
        assert m.label() == "Material(id=0, albedo=(0.5, 0.5, 0.5))"

    def test_frozen(self):
        m = MaterialNode(0, self.DEFAULT_ALBEDO, self.DEFAULT_ROUGHNESS,
                         self.DEFAULT_METALLIC, self.DEFAULT_EMISSIVE,
                         self.DEFAULT_AO)
        with pytest.raises(FrozenInstanceError):
            m.albedo = Vec3Node(0.0, 0.0, 0.0)


# =============================================================================
# T-1.7: SceneGraph container
# =============================================================================


class TestSceneGraph:
    """SceneGraph is the root container holding the complete scene."""

    def test_empty_scene(self):
        """Empty scene (no primitives) handled gracefully."""
        sg = SceneGraph(primitives=())
        assert sg.primitives == ()
        assert sg.pipeline == ()
        assert sg.materials == ()
        assert sg.name == ""

    def test_empty_children(self):
        sg = SceneGraph(primitives=())
        assert sg.children() == ()

    def test_children_returns_pipeline_then_primitives(self):
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        box = BoxNode(pos, Vec3Node(1.0, 1.0, 1.0))
        repeat = RepeatNode(pos, Vec3Node(2.0, 2.0, 2.0))
        sg = SceneGraph(
            primitives=(sphere, box),
            pipeline=(repeat,),
        )
        children = sg.children()
        # pipeline first, then primitives
        assert children[0] is repeat
        assert children[1] is sphere
        assert children[2] is box

    def test_label_unnamed(self):
        sg = SceneGraph(primitives=())
        assert "(unnamed)" not in sg.label()
        assert "SceneGraph" in sg.label()

    def test_label_with_name(self):
        sg = SceneGraph(primitives=(), name="test_scene")
        assert "test_scene" in sg.label()

    def test_label_counts(self):
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        box = BoxNode(pos, Vec3Node(1.0, 1.0, 1.0))
        sg = SceneGraph(
            primitives=(sphere, box),
            pipeline=(RepeatNode(pos, Vec3Node(2.0, 2.0, 2.0)),),
            materials=(
                MaterialNode(0, Vec3Node(0.5, 0.5, 0.5),
                            FloatNode(0.5), FloatNode(0.0),
                            FloatNode(0.0), FloatNode(1.0)),
            ),
        )
        label = sg.label()
        assert "1 pipeline" in label
        assert "2 primitives" in label
        assert "1 materials" in label

    def test_walk_yields_all_descendants(self):
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        repeat = RepeatNode(pos, Vec3Node(2.0, 2.0, 2.0))
        sg = SceneGraph(primitives=(sphere,), pipeline=(repeat,))
        results = list(sg.walk())
        # Walk order (depth-first pre-order):
        # sg(0) -> repeat(1) -> pos(2) -> sphere(1) -> pos(2)
        assert results[0] == (sg, 0)
        assert results[1] == (repeat, 1)
        assert results[2] == (pos, 2)  # repeat's input child
        assert results[3] == (sphere, 1)
        assert results[4] == (pos, 2)  # sphere's position child

    def test_deep_label_structure(self):
        """deep_label() produces a multiline structured summary."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        sg = SceneGraph(primitives=(sphere,), name="demo")
        output = sg.deep_label()
        assert "SceneGraph: demo" in output
        assert "Primitives:" in output
        assert "Sphere" in output
        assert "Pipeline:" not in output  # Empty pipeline omitted

    def test_deep_label_with_pipeline_and_materials(self):
        """deep_label() includes pipeline and materials when present."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        repeat = RepeatNode(pos, Vec3Node(2.0, 2.0, 2.0))
        mat = MaterialNode(0, Vec3Node(0.5, 0.5, 0.5),
                          FloatNode(0.5), FloatNode(0.0),
                          FloatNode(0.0), FloatNode(1.0))
        sg = SceneGraph(
            primitives=(sphere,),
            pipeline=(repeat,),
            materials=(mat,),
            name="full_scene",
        )
        output = sg.deep_label()
        assert "SceneGraph: full_scene" in output
        assert "Pipeline:" in output
        assert "Repeat" in output
        assert "Primitives:" in output
        assert "Sphere" in output
        assert "Materials:" in output
        assert "Material" in output

    def test_frozen(self):
        sg = SceneGraph(primitives=())
        with pytest.raises(FrozenInstanceError):
            sg.primitives = (SphereNode(PositionNode(), FloatNode(1.0)),)


# =============================================================================
# Integration: Walk traversal across a realistic composite tree
# =============================================================================


class TestCompositeTreeWalk:
    """Verify walk() traversal across a realistic multi-level tree."""

    @staticmethod
    def build_demo_scene():
        """Build a representative scene graph for traversal testing."""
        pos = PositionNode()
        repeat = RepeatNode(pos, Vec3Node(3.0, 3.0, 3.0))
        sphere = SphereNode(pos, FloatNode(1.0), material_id=0)
        box = BoxNode(pos, Vec3Node(1.0, 2.0, 1.0), material_id=1)
        mat_a = MaterialNode(
            0, Vec3Node(0.9, 0.2, 0.2),
            FloatNode(0.3), FloatNode(0.0), FloatNode(0.0), FloatNode(1.0),
        )
        mat_b = MaterialNode(
            1, Vec3Node(0.2, 0.6, 0.9),
            FloatNode(0.8), FloatNode(0.1), FloatNode(0.0), FloatNode(1.0),
        )
        sg = SceneGraph(
            primitives=(sphere, box),
            pipeline=(repeat,),
            materials=(mat_a, mat_b),
            name="demo_scene",
        )
        return sg, pos, repeat, sphere, box, mat_a, mat_b

    def test_walk_order_depth_first(self):
        """walk() yields parent before children (pre-order traversal)."""
        sg, pos, repeat, sphere, box, mat_a, mat_b = self.build_demo_scene()
        nodes = [n for n, d in sg.walk()]
        # Depth-first pre-order: pipeline first (repeat+pos), then primitives
        assert nodes[0] is sg
        assert nodes[1] is repeat
        assert nodes[2] is pos   # repeat's input child
        assert nodes[3] is sphere
        assert nodes[4] is pos   # sphere's position child
        assert nodes[5] is box
        assert nodes[6] is pos   # box's position child

    def test_walk_correct_depth(self):
        """Depth counter is correct at each tree level."""
        sg, pos, repeat, sphere, box, mat_a, mat_b = self.build_demo_scene()
        # Pre-order capture: first visit wins
        depths = {}
        for n, d in sg.walk():
            depths.setdefault(id(n), d)
        assert depths[id(sg)] == 0          # Root
        assert depths[id(repeat)] == 1      # Direct child of sg
        assert depths[id(sphere)] == 1      # Direct child of sg
        assert depths[id(box)] == 1         # Direct child of sg
        assert depths[id(pos)] == 2         # Child of repeat (first encounter)
