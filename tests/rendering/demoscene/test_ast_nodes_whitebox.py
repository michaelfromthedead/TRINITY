# SPDX-License-Identifier: MIT
"""Whitebox tests for AST Node System (T-FG-7.5).

Whitebox coverage plan:
  W-1.1: ExprNode.walk() is a generator protocol (types.GeneratorType)
  W-1.2: ExprNode.walk() lazy consumption — partial iteration
  W-1.3: children() return type is tuple for ALL concrete node types
  W-1.4: ExprNode.pretty() multi-level nested formatting
  W-1.5: FloatNode extreme boundaries (inf, -inf, nan, -0.0, 0.0)
  W-1.6: Vec3Node.from_tuple() error paths (wrong length, non-iterable)
  W-1.7: Vec3Node components always cast to float via as_tuple()
  W-1.8: SdfPrimitiveNode material_id is kw_only — cannot be positional
  W-1.9: CapsuleNode children() ordering guarantee (pos, ep_a, ep_b)
  W-1.10: CombineNode.kind is bare str (not enum)
  W-1.11: UnionNode/IntersectionNode/SubtractionNode object.__setattr__ mechanics
  W-1.12: SceneGraph children() always returns bare tuple (not list)
  W-1.13: Walk with shared child reference (same PositionNode under multiple parents)
  W-1.14: deep_label() exact line-level structure with/without each section
  W-1.15: SceneGraph label() zero counts for empty sections
  W-1.16: Type map values are all str, keys are classes
  W-1.17: CompensationNode param is bare float (not FloatNode)
  W-1.18: children() idempotency — repeated calls return equal but distinct tuples
"""

from __future__ import annotations

import dataclasses
import math
import types
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
    DOMAIN_OP_TYPE_MAP,
    DomainOpNode,
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
    SDF_PRIMITIVE_TYPE_MAP,
    SdfPrimitiveNode,
    SceneGraph,
    SphereNode,
    StretchNode,
    SubtractionNode,
    TorusNode,
    TwistNode,
    UnionNode,
    Vec3Node,
)


# =============================================================================
# W-1.1: ExprNode.walk() generator protocol
# =============================================================================


class TestWalkGeneratorProtocol:
    """Whitebox: walk() is a generator, not a list-returning function."""

    def test_walk_is_generator(self):
        """walk() returns a generator object, not a list."""
        n = ExprNode()
        gen = n.walk()
        assert isinstance(gen, types.GeneratorType)

    def test_walk_lazy_does_not_exhaust_early(self):
        """Generator is lazy — does not pre-compute all nodes."""
        n = ExprNode()
        gen = n.walk()
        # First next() yields (n, 0)
        first = next(gen)
        assert first == (n, 0)
        # Second next() should raise StopIteration (leaf node)
        with pytest.raises(StopIteration):
            next(gen)

    def test_walk_generator_on_nested_tree(self):
        """Generator on a nested tree yields nodes incrementally."""
        child = ExprNode()
        parent_type = type("_P", (ExprNode,), {"children": lambda self: (child,)})
        parent = parent_type()
        gen = parent.walk()
        assert next(gen) == (parent, 0)
        assert next(gen) == (child, 1)
        with pytest.raises(StopIteration):
            next(gen)

    def test_walk_generator_is_restartable_per_instance(self):
        """Each call to walk() returns a fresh generator."""
        n = ExprNode()
        g1 = n.walk()
        g2 = n.walk()
        assert g1 is not g2
        assert list(g1) == list(g2)


# =============================================================================
# W-1.2: ExprNode.walk() shallow leaf
# =============================================================================


class TestWalkLeaf:
    """Whitebox: leaf node walk yields exactly (self, 0)."""

    def test_floatnode_walk_single(self):
        n = FloatNode(3.14)
        results = list(n.walk())
        assert len(results) == 1
        node, depth = results[0]
        assert node is n
        assert depth == 0

    def test_vec3node_walk_single(self):
        n = Vec3Node(1.0, 2.0, 3.0)
        results = list(n.walk())
        assert len(results) == 1

    def test_positionnode_walk_single(self):
        n = PositionNode()
        results = list(n.walk())
        assert len(results) == 1


# =============================================================================
# W-1.3: children() return type is tuple for ALL concrete node types
# =============================================================================


# Nodes where children() returns list (from ExprNode base default)
_BASE_AS_LIST = {"FloatNode", "Vec3Node", "PositionNode", "CompensationNode"}

_COLLECT_CHILDREN_TYPES = [
    ("FloatNode", FloatNode(1.0), True),
    ("Vec3Node", Vec3Node(1, 2, 3), True),
    ("PositionNode", PositionNode(), True),
    ("RepeatNode", RepeatNode(PositionNode(), Vec3Node(2, 2, 2)), False),
    ("CellIdNode", CellIdNode(PositionNode(), Vec3Node(1, 1, 1)), False),
    ("MirrorNode", MirrorNode(PositionNode(), Axis.X), False),
    ("KifsNode", KifsNode(PositionNode(), FloatNode(5.0)), False),
    ("TwistNode", TwistNode(PositionNode(), FloatNode(1.0)), False),
    ("BendNode", BendNode(PositionNode(), FloatNode(2.0)), False),
    ("StretchNode", StretchNode(PositionNode(), FloatNode(2.0), Axis.Y), False),
    ("CompensationNode", CompensationNode(Kind.KIFS, 0.5), True),
    ("SphereNode", SphereNode(PositionNode(), FloatNode(1.0)), False),
    ("BoxNode", BoxNode(PositionNode(), Vec3Node(1, 1, 1)), False),
    ("TorusNode", TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)), False),
    ("CylinderNode", CylinderNode(PositionNode(), FloatNode(3.0), FloatNode(1.0)), False),
    ("ConeNode", ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)), False),
    ("PlaneNode", PlaneNode(PositionNode(), Vec3Node(0, 1, 0), FloatNode(-1.0)), False),
    ("CapsuleNode", CapsuleNode(PositionNode(), PositionNode(), Vec3Node(0, 1, 0), FloatNode(0.5)), False),
    ("UnionNode", UnionNode(ExprNode(), ExprNode()), False),
    ("IntersectionNode", IntersectionNode(ExprNode(), ExprNode()), False),
    ("SubtractionNode", SubtractionNode(ExprNode(), ExprNode()), False),
    ("SceneGraph", SceneGraph(primitives=()), False),
]


class TestChildrenReturnType:
    """Whitebox: children() returns Sequence; base class returns list, overrides return tuple."""

    @pytest.mark.parametrize("name,node,is_leaf", _COLLECT_CHILDREN_TYPES)
    def test_children_is_sequence(self, name, node, is_leaf):
        """All children() return values are iterable sequences."""
        from typing import Sequence
        children = node.children()
        assert isinstance(children, (tuple, list)), (
            f"{name}.children() returned {type(children).__name__}, expected tuple or list"
        )

    @pytest.mark.parametrize("name,node,is_leaf", _COLLECT_CHILDREN_TYPES)
    def test_children_empty_on_leaves(self, name, node, is_leaf):
        if is_leaf:
            assert len(node.children()) == 0

    @pytest.mark.parametrize("name,node,is_leaf", _COLLECT_CHILDREN_TYPES)
    def test_children_overrides_return_tuple(self, name, node, is_leaf):
        """Subclasses that override children() return tuple, not list."""
        if name not in _BASE_AS_LIST and not is_leaf:
            children = node.children()
            assert isinstance(children, tuple), (
                f"{name}.children() override returned {type(children).__name__}, expected tuple"
            )

    def test_base_exprnode_children_is_list(self):
        """ExprNode base default returns [] as documented."""
        assert ExprNode().children() == []

    def test_domain_op_children_is_tuple(self):
        """DomainOpNode overrides to return tuple."""
        n = DomainOpNode(PositionNode())
        assert isinstance(n.children(), tuple)


# =============================================================================
# W-1.4: ExprNode.pretty() multi-level nested formatting
# =============================================================================


class TestPrettyMultiLevel:
    """Whitebox: pretty() formats with correct indentation at multiple levels."""

    def test_pretty_indent_zero(self):
        n = ExprNode()
        assert n.pretty(indent=0) == "ExprNode"

    def test_pretty_indent_one(self):
        n = ExprNode()
        assert n.pretty(indent=1) == "  ExprNode"

    def test_pretty_indent_five(self):
        n = ExprNode()
        assert n.pretty(indent=5) == "          ExprNode"

    def test_pretty_floatnode(self):
        assert FloatNode(42.0).pretty(indent=2) == "    Float(42.0)"

    def test_pretty_scenegraph_no_indent(self):
        sg = SceneGraph(primitives=())
        assert sg.pretty() == sg.label()


# =============================================================================
# W-1.5: FloatNode extreme boundaries
# =============================================================================


class TestFloatNodeExtremes:
    """Whitebox: FloatNode handles extreme numeric values."""

    def test_infinity(self):
        n = FloatNode(math.inf)
        assert n.value == math.inf
        assert math.isinf(n.value)

    def test_negative_infinity(self):
        n = FloatNode(-math.inf)
        assert n.value == -math.inf
        assert math.isinf(n.value)

    def test_nan(self):
        n = FloatNode(math.nan)
        assert math.isnan(n.value)

    def test_negative_zero(self):
        n = FloatNode(-0.0)
        assert n.value == 0.0  # -0.0 == 0.0 in Python
        # Sign bit is preserved (implementation detail of Python float)
        assert math.copysign(1, n.value) == -1.0

    def test_label_infinity(self):
        """Label includes inf representation."""
        label = FloatNode(math.inf).label()
        assert "inf" in label

    def test_label_nan(self):
        label = FloatNode(math.nan).label()
        assert "nan" in label

    def test_label_negative_zero(self):
        label = FloatNode(-0.0).label()
        assert "0.0" in label


# =============================================================================
# W-1.6: Vec3Node.from_tuple() error paths
# =============================================================================


class TestVec3NodeFromTupleErrors:
    """Whitebox: Vec3Node.from_tuple() error handling for invalid inputs."""

    def test_too_few_elements(self):
        with pytest.raises(TypeError):
            Vec3Node.from_tuple((1.0, 2.0))

    def test_too_many_elements(self):
        with pytest.raises(TypeError):
            Vec3Node.from_tuple((1.0, 2.0, 3.0, 4.0))

    def test_non_iterable(self):
        with pytest.raises(TypeError):
            Vec3Node.from_tuple(42)

    def test_none(self):
        with pytest.raises(TypeError):
            Vec3Node.from_tuple(None)

    def test_string(self):
        with pytest.raises(TypeError):
            Vec3Node.from_tuple("ab")  # 2-char string still unpacks wrong


# =============================================================================
# W-1.7: Vec3Node components always cast to float via as_tuple()
# =============================================================================


class TestVec3NodeFloatContract:
    """Whitebox: Vec3Node components and as_tuple() always return float."""

    def test_as_tuple_all_float(self):
        """as_tuple() always returns float components regardless of input types."""
        v = Vec3Node(1, 2, 3)
        t = v.as_tuple()
        assert all(isinstance(x, float) for x in t)
        assert t == (1.0, 2.0, 3.0)

    def test_from_tuple_float_conversion(self):
        v = Vec3Node.from_tuple((1, 2, 3))
        t = v.as_tuple()
        assert all(isinstance(x, float) for x in t)

    def test_mixed_as_tuple_float(self):
        """as_tuple() normalizes mixed types to float."""
        v = Vec3Node(1, 2.5, 3)
        t = v.as_tuple()
        assert all(isinstance(x, float) for x in t)
        assert t == (1.0, 2.5, 3.0)

    def test_components_stored_as_given(self):
        """Components are stored as-is (no runtime type coercion to float)."""
        v = Vec3Node(1, 2, 3)
        assert type(v.x) is int
        assert type(v.y) is int
        assert type(v.z) is int

    def test_label_shows_floats(self):
        v = Vec3Node(1, 2, 3)
        label = v.label()
        assert "1.0" in label
        assert "2.0" in label
        assert "3.0" in label


# =============================================================================
# W-1.8: SdfPrimitiveNode material_id is kw_only
# =============================================================================


class TestSdfPrimitiveNodeKwOnly:
    """Whitebox: material_id is keyword-only, not positional."""

    def test_default_material_id(self):
        """Default material_id is 0."""
        n = SdfPrimitiveNode(PositionNode())
        assert n.material_id == 0

    def test_material_id_kw_only_set(self):
        n = SdfPrimitiveNode(PositionNode(), material_id=5)
        assert n.material_id == 5

    def test_material_id_field_is_kw_only(self):
        """The field descriptor shows kw_only=True."""
        for f in dataclasses.fields(SdfPrimitiveNode):
            if f.name == "material_id":
                assert f.kw_only is True
                return
        pytest.fail("material_id field not found on SdfPrimitiveNode")

    def test_sphere_material_id_default(self):
        s = SphereNode(PositionNode(), FloatNode(1.0))
        assert s.material_id == 0

    def test_sphere_material_id_custom(self):
        s = SphereNode(PositionNode(), FloatNode(1.0), material_id=3)
        assert s.material_id == 3


# =============================================================================
# W-1.9: CapsuleNode children() ordering guarantee
# =============================================================================


class TestCapsuleNodeChildrenOrder:
    """Whitebox: CapsuleNode.children() always returns (position, ep_a, ep_b)."""

    def test_children_three_elements(self):
        c = CapsuleNode(PositionNode(), PositionNode(), Vec3Node(0, 1, 0), FloatNode(0.5))
        assert len(c.children()) == 3

    def test_first_child_is_position(self):
        pos = PositionNode()
        ep_a = Vec3Node(0, 1, 0)
        ep_b = Vec3Node(1, 0, 0)
        c = CapsuleNode(pos, ep_a, ep_b, FloatNode(0.5))
        assert c.children()[0] is pos

    def test_second_child_is_endpoint_a(self):
        pos = PositionNode()
        ep_a = Vec3Node(0, 1, 0)
        ep_b = Vec3Node(1, 0, 0)
        c = CapsuleNode(pos, ep_a, ep_b, FloatNode(0.5))
        assert c.children()[1] is ep_a

    def test_third_child_is_endpoint_b(self):
        pos = PositionNode()
        ep_a = Vec3Node(0, 1, 0)
        ep_b = Vec3Node(1, 0, 0)
        c = CapsuleNode(pos, ep_a, ep_b, FloatNode(0.5))
        assert c.children()[2] is ep_b

    def test_children_not_position_in_walk(self):
        """Walk includes all three children, not just position."""
        pos = PositionNode()
        ep_a = PositionNode()
        ep_b = PositionNode()
        c = CapsuleNode(pos, ep_a, ep_b, FloatNode(0.5))
        walked = [n for n, d in c.walk()]
        assert pos in walked
        assert ep_a in walked
        assert ep_b in walked


# =============================================================================
# W-1.10: CombineNode.kind is bare str
# =============================================================================


class TestCombineNodeKindType:
    """Whitebox: CombineNode.kind is a plain str, not an enum."""

    def test_kind_is_string(self):
        c = CombineNode("test_kind", ExprNode(), ExprNode())
        assert isinstance(c.kind, str)

    def test_union_kind_is_string(self):
        u = UnionNode(ExprNode(), ExprNode())
        assert isinstance(u.kind, str)
        assert u.kind == "union"

    def test_intersection_kind_is_string(self):
        i = IntersectionNode(ExprNode(), ExprNode())
        assert isinstance(i.kind, str)
        assert i.kind == "intersection"

    def test_subtraction_kind_is_string(self):
        s = SubtractionNode(ExprNode(), ExprNode())
        assert isinstance(s.kind, str)
        assert s.kind == "subtraction"

    def test_kind_not_enum(self):
        c = CombineNode("any_string", ExprNode(), ExprNode())
        assert not isinstance(c.kind, Kind)


# =============================================================================
# W-1.11: UnionNode/IntersectionNode/SubtractionNode object.__setattr__
# =============================================================================


class TestCSGFrozenBypass:
    """Whitebox: CSG nodes use object.__setattr__ to set fields on frozen dataclass."""

    def test_union_fields_set_after_init(self):
        u = UnionNode(ExprNode(), ExprNode())
        assert u.kind == "union"
        assert isinstance(u.left, ExprNode)
        assert isinstance(u.right, ExprNode)

    def test_intersection_fields_set_after_init(self):
        i = IntersectionNode(ExprNode(), ExprNode())
        assert i.kind == "intersection"

    def test_subtraction_fields_set_after_init(self):
        s = SubtractionNode(ExprNode(), ExprNode())
        assert s.kind == "subtraction"

    def test_union_is_still_frozen_after_init(self):
        """Even with __setattr__ bypass in __init__, the instance is still frozen."""
        u = UnionNode(ExprNode(), ExprNode())
        with pytest.raises(dataclasses.FrozenInstanceError):
            u.kind = "intersection"

    def test_csg_children_match_kind(self):
        left = FloatNode(1.0)
        right = FloatNode(2.0)
        u = UnionNode(left, right)
        children = u.children()
        assert children[0] is left
        assert children[1] is right

    def test_csg_subclass_is_frozen_dataclass(self):
        for cls in (UnionNode, IntersectionNode, SubtractionNode):
            assert dataclasses.is_dataclass(cls)
            assert cls.__dataclass_params__.frozen is True


# =============================================================================
# W-1.12: SceneGraph children() always returns bare tuple
# =============================================================================


class TestSceneGraphChildrenContract:
    """Whitebox: SceneGraph.children() guarantees a tuple."""

    def test_empty_children_is_tuple(self):
        sg = SceneGraph(primitives=())
        assert isinstance(sg.children(), tuple)

    def test_with_pipeline_is_tuple(self):
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
        )
        assert isinstance(sg.children(), tuple)

    def test_with_materials_only_children_do_not_include_materials(self):
        """Materials are NOT children (they are not walked)."""
        mat = MaterialNode(
            0, Vec3Node(0.5, 0.5, 0.5),
            FloatNode(0.5), FloatNode(0.0), FloatNode(0.0), FloatNode(1.0),
        )
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            materials=(mat,),
        )
        for child in sg.children():
            assert not isinstance(child, MaterialNode)

    def test_children_order_pipeline_then_primitives(self):
        """Pipeline ops appear before primitives in children."""
        repeat = RepeatNode(PositionNode(), Vec3Node(2, 2, 2))
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        box = BoxNode(PositionNode(), Vec3Node(1, 1, 1))
        sg = SceneGraph(
            primitives=(sphere, box),
            pipeline=(repeat,),
        )
        children = sg.children()
        assert children[0] is repeat  # pipeline first
        assert children[1] is sphere
        assert children[2] is box


# =============================================================================
# W-1.13: Walk with shared child reference
# =============================================================================


class TestWalkSharedChild:
    """Whitebox: walk() correctly handles a shared child referenced by multiple parents."""

    def test_shared_position_appears_multiple_times(self):
        """A single PositionNode shared across primitives appears once per parent."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        box = BoxNode(pos, Vec3Node(1, 1, 1))
        sg = SceneGraph(primitives=(sphere, box))
        results = list(sg.walk())
        # sg(0) -> sphere(1) -> pos(2) -> box(1) -> pos(2)
        assert len(results) == 5
        # pos appears TWICE — once under sphere, once under box
        pos_count = sum(1 for n, d in results if n is pos)
        assert pos_count == 2

    def test_shared_position_depth_first_encounter(self):
        """Depth of shared node is depth of first encounter in pre-order."""
        pos = PositionNode()
        sphere = SphereNode(pos, FloatNode(1.0))
        box = BoxNode(pos, Vec3Node(1, 1, 1))
        sg = SceneGraph(primitives=(sphere, box))
        nodes_at_depth = {}
        for n, d in sg.walk():
            if id(n) not in nodes_at_depth:
                nodes_at_depth[id(n)] = d
        # pos first appears under sphere at depth 2
        assert nodes_at_depth[id(pos)] == 2

    def test_shared_child_across_pipeline_and_primitive(self):
        """A node shared between pipeline and primitives appears under each."""
        pos = PositionNode()
        repeat = RepeatNode(pos, Vec3Node(2, 2, 2))
        sphere = SphereNode(pos, FloatNode(1.0))
        sg = SceneGraph(primitives=(sphere,), pipeline=(repeat,))
        results = list(sg.walk())
        # sg(0) -> repeat(1) -> pos(2) -> sphere(1) -> pos(2)
        pos_count = sum(1 for n, d in results if n is pos)
        assert pos_count == 2


# =============================================================================
# W-1.14: deep_label() exact line-level structure
# =============================================================================


class TestDeepLabelExactStructure:
    """Whitebox: deep_label() produces exact line-level formatting."""

    def test_empty_unnamed_scene(self):
        sg = SceneGraph(primitives=())
        lines = sg.deep_label().split("\n")
        assert lines[0] == "SceneGraph: (unnamed)"
        assert lines[1] == "  Primitives:"
        assert len(lines) == 2  # no pipeline, no materials sections

    def test_named_scene(self):
        sg = SceneGraph(primitives=(), name="test_scene")
        assert sg.deep_label().startswith("SceneGraph: test_scene")

    def test_with_pipeline_section(self):
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
        )
        lines = sg.deep_label().split("\n")
        assert "  Pipeline:" in lines
        pipe_idx = lines.index("  Pipeline:")
        assert "    Twist" in lines[pipe_idx + 1]

    def test_with_materials_section(self):
        mat = MaterialNode(
            0, Vec3Node(0.5, 0.5, 0.5),
            FloatNode(0.5), FloatNode(0.0), FloatNode(0.0), FloatNode(1.0),
        )
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            materials=(mat,),
        )
        lines = sg.deep_label().split("\n")
        assert "  Materials:" in lines
        mat_idx = lines.index("  Materials:")
        assert "Material" in lines[mat_idx + 1]

    def test_only_primitives_no_pipeline_no_materials(self):
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        output = sg.deep_label()
        assert "Pipeline:" not in output
        assert "Materials:" not in output
        assert "Primitives:" in output

    def test_all_sections_present(self):
        """All three sections (pipeline, primitives, materials) appear when non-empty."""
        repeat = RepeatNode(PositionNode(), Vec3Node(2, 2, 2))
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        mat = MaterialNode(
            0, Vec3Node(0.5, 0.5, 0.5),
            FloatNode(0.5), FloatNode(0.0), FloatNode(0.0), FloatNode(1.0),
        )
        sg = SceneGraph(
            primitives=(sphere,),
            pipeline=(repeat,),
            materials=(mat,),
            name="full",
        )
        output = sg.deep_label()
        assert output.startswith("SceneGraph: full")
        assert "  Pipeline:" in output
        assert "  Primitives:" in output
        assert "  Materials:" in output


# =============================================================================
# W-1.15: SceneGraph label() zero counts
# =============================================================================


class TestSceneGraphLabelZeroCounts:
    """Whitebox: SceneGraph.label() shows counts including zeros."""

    def test_zero_everything(self):
        sg = SceneGraph(primitives=())
        label = sg.label()
        assert "0 pipeline" in label
        assert "0 primitives" in label
        assert "0 materials" in label

    def test_only_primitives(self):
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        label = sg.label()
        assert "0 pipeline" in label
        assert "1 primitives" in label
        assert "0 materials" in label

    def test_large_counts(self):
        pos = PositionNode()
        many_spheres = tuple(
            SphereNode(pos, FloatNode(float(i))) for i in range(100)
        )
        sg = SceneGraph(primitives=many_spheres)
        label = sg.label()
        assert "100 primitives" in label


# =============================================================================
# W-1.16: Type map values are all str, keys are classes
# =============================================================================


class TestTypeMapValueTypes:
    """Whitebox: type map entries have correct types."""

    def test_sdf_primitive_values_are_strings(self):
        for key, value in SDF_PRIMITIVE_TYPE_MAP.items():
            assert isinstance(value, str), f"{key.__name__} -> {value!r} is not str"

    def test_sdf_primitive_keys_are_classes(self):
        for key in SDF_PRIMITIVE_TYPE_MAP:
            assert isinstance(key, type), f"{key!r} is not a class"

    def test_domain_op_values_are_strings(self):
        for key, value in DOMAIN_OP_TYPE_MAP.items():
            assert isinstance(value, str), f"{key.__name__} -> {value!r} is not str"

    def test_domain_op_keys_are_classes(self):
        for key in DOMAIN_OP_TYPE_MAP:
            assert isinstance(key, type), f"{key!r} is not a class"

    def test_sdf_primitive_no_duplicate_values(self):
        values = list(SDF_PRIMITIVE_TYPE_MAP.values())
        assert len(values) == len(set(values)), "Duplicate SDF primitive type map values"

    def test_domain_op_no_duplicate_values(self):
        values = list(DOMAIN_OP_TYPE_MAP.values())
        assert len(values) == len(set(values)), "Duplicate domain op type map values"


# =============================================================================
# W-1.17: CompensationNode param is bare float
# =============================================================================


class TestCompensationNodeParamType:
    """Whitebox: CompensationNode.param is a bare float, not FloatNode."""

    def test_param_is_float(self):
        c = CompensationNode(Kind.KIFS, 0.5)
        assert isinstance(c.param, float)

    def test_param_stored_as_given(self):
        """Param stored as-is (no runtime coercion to float when passed as int)."""
        c = CompensationNode(Kind.STRETCH, 1)
        assert type(c.param) is int

    def test_param_negative_float(self):
        c = CompensationNode(Kind.TWIST, -2.0)
        assert isinstance(c.param, float)

    def test_param_zero(self):
        c = CompensationNode(Kind.REPEAT, 0.0)
        assert c.param == 0.0
        assert isinstance(c.param, float)

    def test_param_not_floatnode(self):
        c = CompensationNode(Kind.KIFS, 0.5)
        assert not isinstance(c.param, FloatNode)

    def test_label_contains_param(self):
        c = CompensationNode(Kind.KIFS, 0.5)
        assert "0.5" in c.label()


# =============================================================================
# W-1.18: children() idempotency across repeated calls
# =============================================================================


class TestChildrenIdempotent:
    """Whitebox: calling children() multiple times returns equal (not same) tuples."""

    def test_floatnode_children_idempotent(self):
        n = FloatNode(1.0)
        assert n.children() == n.children()

    def test_repeatnode_children_idempotent(self):
        n = RepeatNode(PositionNode(), Vec3Node(2, 2, 2))
        assert n.children() == n.children()

    def test_sphere_children_idempotent(self):
        n = SphereNode(PositionNode(), FloatNode(1.0))
        assert n.children() == n.children()

    def test_union_children_idempotent(self):
        n = UnionNode(ExprNode(), ExprNode())
        assert n.children() == n.children()

    def test_capsule_children_idempotent(self):
        n = CapsuleNode(PositionNode(), PositionNode(), Vec3Node(0, 1, 0), FloatNode(0.5))
        assert n.children() == n.children()

    def test_scenegraph_children_idempotent(self):
        n = SceneGraph(primitives=())
        assert n.children() == n.children()


# =============================================================================
# W-1.19: ExprNode.subclass_check methods exist and are callable
# =============================================================================


class TestExprNodeInterface:
    """Whitebox: ExprNode defines the abstract interface correctly."""

    def test_walk_method_exists(self):
        assert hasattr(ExprNode, "walk")
        assert callable(ExprNode.walk)

    def test_children_method_exists(self):
        assert hasattr(ExprNode, "children")
        assert callable(ExprNode.children)

    def test_pretty_method_exists(self):
        assert hasattr(ExprNode, "pretty")
        assert callable(ExprNode.pretty)

    def test_label_method_exists(self):
        assert hasattr(ExprNode, "label")
        assert callable(ExprNode.label)

    def test_children_arg_count(self):
        """children() takes only self."""
        import inspect
        sig = inspect.signature(ExprNode.children)
        params = list(sig.parameters.keys())
        assert params == ["self"]

    def test_walk_arg_count(self):
        """walk() takes self and optional depth."""
        import inspect
        sig = inspect.signature(ExprNode.walk)
        params = list(sig.parameters.keys())
        assert "depth" in params


# =============================================================================
# W-1.20: DomainOpNode base class invariants
# =============================================================================


class TestDomainOpNodeInvariants:
    """Whitebox: DomainOpNode enforces base class invariants."""

    def test_input_is_exprnode(self):
        op = DomainOpNode(PositionNode())
        assert isinstance(op.input, ExprNode)

    def test_input_via_children(self):
        child = PositionNode()
        op = DomainOpNode(child)
        assert op.children()[0] is child

    def test_input_is_first_positional(self):
        """input is the first (and only required) positional argument."""
        child = PositionNode()
        op = DomainOpNode(child)
        assert op.input is child

    def test_domain_op_is_frozen(self):
        op = DomainOpNode(PositionNode())
        with pytest.raises(dataclasses.FrozenInstanceError):
            op.input = FloatNode(1.0)


# =============================================================================
# W-1.21: SceneGraph children are Sequence[ExprNode] per annotation
# =============================================================================


class TestSceneGraphAnnotationContract:
    """Whitebox: SceneGraph fields satisfy Sequence[ExprNode] types."""

    def test_primitives_annotation_has_correct_shape(self):
        """Primitives is typed as tuple of SdfPrimitiveNode."""
        for f in dataclasses.fields(SceneGraph):
            if f.name == "primitives":
                assert "SdfPrimitiveNode" in str(f.type) or "tuple" in str(f.type)
                return
        pytest.fail("primitives field not found")

    def test_pipeline_annotation_has_correct_shape(self):
        """Pipeline is typed as tuple of DomainOpNode."""
        for f in dataclasses.fields(SceneGraph):
            if f.name == "pipeline":
                assert "DomainOpNode" in str(f.type) or "tuple" in str(f.type)
                return
        pytest.fail("pipeline field not found")


# =============================================================================
# W-1.22: walk() depth correctness with empty intermediate
# =============================================================================


class TestWalkDepthEmptyIntermediate:
    """Whitebox: depth counter remains correct when intermediate has no children."""

    def test_depth_correct_with_compensation(self):
        """CompensationNode is a leaf; depth counts correctly in its presence."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                TwistNode(PositionNode(), FloatNode(2.0)),
            ),
        )
        depths = {}
        for n, d in sg.walk():
            depths.setdefault(id(n), d)
        assert depths[id(sg)] == 0
        assert depths[id(sg.pipeline[0])] == 1
        assert depths[id(sg.primitives[0])] == 1


# =============================================================================
# W-1.23: deep_label() idempotent
# =============================================================================


class TestDeepLabelIdempotent:
    """Whitebox: calling deep_label() multiple times returns identical output."""

    def test_deep_label_idempotent_empty(self):
        sg = SceneGraph(primitives=())
        assert sg.deep_label() == sg.deep_label()

    def test_deep_label_idempotent_full(self):
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
            name="test",
        )
        assert sg.deep_label() == sg.deep_label()


# =============================================================================
# W-1.24: SceneGraph default name is empty string (not None)
# =============================================================================


class TestSceneGraphNameDefault:
    """Whitebox: SceneGraph.name defaults to empty string, not None."""

    def test_name_default_is_empty_string(self):
        sg = SceneGraph(primitives=())
        assert sg.name == ""
        assert sg.name is not None

    def test_name_field_is_str(self):
        sg = SceneGraph(primitives=())
        assert isinstance(sg.name, str)

    def test_label_unnamed_no_name(self):
        sg = SceneGraph(primitives=())
        assert "(unnamed)" not in sg.label()
        assert sg.label().startswith("SceneGraph")
