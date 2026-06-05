"""
Tests for SDF AST Builder (T-DEMO-2.1 and T-DEMO-2.2)

This test suite covers:
- Node creation for all primitives (T-DEMO-2.1)
- Combinator nodes
- Domain operation nodes
- Scene composition
- Trinity pattern compliance (T-DEMO-2.2):
  - Mirror introspection
  - Tracker dirty tracking
- Tree building via build_ast()

Total: 55+ tests
"""

import math
import pytest
from typing import Any, Dict

from engine.rendering.demoscene.sdf_ast import (
    # Base
    SDFNode,
    SDFNodeMeta,
    # Primitives
    PrimitiveNode,
    SphereNode,
    BoxNode,
    TorusNode,
    CylinderNode,
    ConeNode,
    PlaneNode,
    CapsuleNode,
    EllipsoidNode,
    BoxFrameNode,
    RoundedBoxNode,
    OctahedronNode,
    PyramidNode,
    # Combinators
    CombinatorNode,
    UnionNode,
    IntersectionNode,
    SubtractionNode,
    SmoothUnionNode,
    SmoothIntersectionNode,
    SmoothSubtractionNode,
    DisplacedNode,
    # Domain ops
    DomainOpNode,
    RepeatNode,
    MirrorNode,
    KIFSNode,
    TwistNode,
    BendNode,
    StretchNode,
    # Scene
    MaterialNode,
    SceneNode,
    CameraNode,
    LightNode,
    RenderSettingsNode,
    # Helpers
    Vec3,
    Axis,
    build_ast,
    # Trinity patterns
    Mirror,
    Tracker,
)


# =============================================================================
# Vec3 Helper Tests (5 tests)
# =============================================================================

class TestVec3:
    """Tests for Vec3 helper class."""

    def test_vec3_creation_defaults(self):
        """Test Vec3 with default values."""
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vec3_creation_values(self):
        """Test Vec3 with explicit values."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vec3_from_tuple(self):
        """Test Vec3.from_tuple()."""
        v = Vec3.from_tuple((4.0, 5.0, 6.0))
        assert v.as_tuple() == (4.0, 5.0, 6.0)

    def test_vec3_from_scalar(self):
        """Test Vec3.from_scalar()."""
        v = Vec3.from_scalar(7.0)
        assert v.x == 7.0
        assert v.y == 7.0
        assert v.z == 7.0

    def test_vec3_length(self):
        """Test Vec3.length()."""
        v = Vec3(3.0, 4.0, 0.0)
        assert abs(v.length() - 5.0) < 1e-6

    def test_vec3_normalized(self):
        """Test Vec3.normalized()."""
        v = Vec3(3.0, 0.0, 4.0)
        n = v.normalized()
        assert abs(n.length() - 1.0) < 1e-6
        assert abs(n.x - 0.6) < 1e-6
        assert abs(n.z - 0.8) < 1e-6

    def test_vec3_arithmetic(self):
        """Test Vec3 arithmetic operations."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(4.0, 5.0, 6.0)

        add = a + b
        assert add.as_tuple() == (5.0, 7.0, 9.0)

        sub = b - a
        assert sub.as_tuple() == (3.0, 3.0, 3.0)

        mul = a * 2.0
        assert mul.as_tuple() == (2.0, 4.0, 6.0)

        neg = -a
        assert neg.as_tuple() == (-1.0, -2.0, -3.0)

    def test_vec3_to_wgsl(self):
        """Test Vec3.to_wgsl()."""
        v = Vec3(1.0, 2.5, 3.0)
        wgsl = v.to_wgsl()
        assert "vec3<f32>" in wgsl
        assert "1.0" in wgsl
        assert "2.5" in wgsl
        assert "3.0" in wgsl


# =============================================================================
# Axis Enumeration Tests (2 tests)
# =============================================================================

class TestAxis:
    """Tests for Axis enumeration."""

    def test_axis_values(self):
        """Test Axis enum values."""
        assert Axis.X.value == "x"
        assert Axis.Y.value == "y"
        assert Axis.Z.value == "z"

    def test_axis_to_index(self):
        """Test Axis.to_index()."""
        assert Axis.X.to_index() == 0
        assert Axis.Y.to_index() == 1
        assert Axis.Z.to_index() == 2


# =============================================================================
# Primitive Node Creation Tests (12 tests)
# =============================================================================

class TestPrimitiveNodes:
    """Tests for primitive node creation."""

    def test_sphere_node_defaults(self):
        """Test SphereNode with default values."""
        node = SphereNode()
        assert node.radius == 1.0
        assert node.position.as_tuple() == (0.0, 0.0, 0.0)
        assert node.wgsl_function == "sdf_sphere"

    def test_sphere_node_custom(self):
        """Test SphereNode with custom values."""
        node = SphereNode(radius=2.5, position=Vec3(1.0, 2.0, 3.0))
        assert node.radius == 2.5
        assert node.position.as_tuple() == (1.0, 2.0, 3.0)

    def test_box_node(self):
        """Test BoxNode creation."""
        node = BoxNode(half_extents=Vec3(1.0, 2.0, 3.0))
        assert node.half_extents.as_tuple() == (1.0, 2.0, 3.0)
        assert node.wgsl_function == "sdf_box"

    def test_torus_node(self):
        """Test TorusNode creation."""
        node = TorusNode(major_radius=2.0, minor_radius=0.5)
        assert node.major_radius == 2.0
        assert node.minor_radius == 0.5
        assert node.wgsl_function == "sdf_torus"

    def test_cylinder_node(self):
        """Test CylinderNode creation."""
        node = CylinderNode(radius=1.0, height=2.0)
        assert node.radius == 1.0
        assert node.height == 2.0
        assert node.wgsl_function == "sdf_cylinder"

    def test_cone_node(self):
        """Test ConeNode creation."""
        node = ConeNode(angle=0.5, height=1.5)
        assert node.angle == 0.5
        assert node.height == 1.5
        assert node.wgsl_function == "sdf_cone"

    def test_plane_node(self):
        """Test PlaneNode creation."""
        node = PlaneNode(normal=Vec3(0.0, 1.0, 0.0), distance=1.0)
        # Normal should be normalized
        assert abs(node.normal.length() - 1.0) < 1e-6
        assert node.distance == 1.0
        assert node.wgsl_function == "sdf_plane"

    def test_capsule_node(self):
        """Test CapsuleNode creation."""
        node = CapsuleNode(
            endpoint_a=Vec3(0.0, 0.0, 0.0),
            endpoint_b=Vec3(0.0, 1.0, 0.0),
            radius=0.25,
        )
        assert node.radius == 0.25
        assert node.wgsl_function == "sdf_capsule"

    def test_ellipsoid_node(self):
        """Test EllipsoidNode creation."""
        node = EllipsoidNode(radii=Vec3(1.0, 2.0, 1.5))
        assert node.radii.as_tuple() == (1.0, 2.0, 1.5)
        assert node.wgsl_function == "sdf_ellipsoid"

    def test_box_frame_node(self):
        """Test BoxFrameNode creation."""
        node = BoxFrameNode(half_extents=Vec3(1.0, 1.0, 1.0), edge_thickness=0.1)
        assert node.edge_thickness == 0.1
        assert node.wgsl_function == "sdf_box_frame"

    def test_rounded_box_node(self):
        """Test RoundedBoxNode creation."""
        node = RoundedBoxNode(corner_radius=0.2)
        assert node.corner_radius == 0.2
        assert node.wgsl_function == "sdf_rounded_box"

    def test_octahedron_node(self):
        """Test OctahedronNode creation."""
        node = OctahedronNode(size=1.5)
        assert node.size == 1.5
        assert node.wgsl_function == "sdf_octahedron"

    def test_pyramid_node(self):
        """Test PyramidNode creation."""
        node = PyramidNode(height=2.0)
        assert node.height == 2.0
        assert node.wgsl_function == "sdf_pyramid"


# =============================================================================
# Combinator Node Tests (7 tests)
# =============================================================================

class TestCombinatorNodes:
    """Tests for combinator node creation."""

    def test_union_node(self):
        """Test UnionNode creation."""
        left = SphereNode()
        right = BoxNode()
        node = UnionNode(left, right)
        assert node.left is left
        assert node.right is right
        assert node.wgsl_function == "sdf_union"
        assert node.children() == (left, right)

    def test_intersection_node(self):
        """Test IntersectionNode creation."""
        left = SphereNode()
        right = BoxNode()
        node = IntersectionNode(left, right)
        assert node.wgsl_function == "sdf_intersection"

    def test_subtraction_node(self):
        """Test SubtractionNode creation."""
        left = SphereNode()
        right = BoxNode()
        node = SubtractionNode(left, right)
        assert node.wgsl_function == "sdf_subtraction"

    def test_smooth_union_node(self):
        """Test SmoothUnionNode creation."""
        left = SphereNode()
        right = BoxNode()
        node = SmoothUnionNode(left, right, k=0.2)
        assert node.k == 0.2
        assert node.wgsl_function == "sdf_smooth_union"

    def test_smooth_intersection_node(self):
        """Test SmoothIntersectionNode creation."""
        left = SphereNode()
        right = BoxNode()
        node = SmoothIntersectionNode(left, right, k=0.3)
        assert node.k == 0.3
        assert node.wgsl_function == "sdf_smooth_intersection"

    def test_smooth_subtraction_node(self):
        """Test SmoothSubtractionNode creation."""
        left = SphereNode()
        right = BoxNode()
        node = SmoothSubtractionNode(left, right, k=0.15)
        assert node.k == 0.15
        assert node.wgsl_function == "sdf_smooth_subtraction"

    def test_displaced_node(self):
        """Test DisplacedNode creation."""
        child = SphereNode()
        node = DisplacedNode(child, amplitude=0.2, frequency=2.0)
        assert node.amplitude == 0.2
        assert node.frequency == 2.0
        assert node.children() == (child,)
        assert node.wgsl_function == "sdf_displaced"


# =============================================================================
# Domain Operation Node Tests (6 tests)
# =============================================================================

class TestDomainOpNodes:
    """Tests for domain operation node creation."""

    def test_repeat_node(self):
        """Test RepeatNode creation."""
        child = SphereNode()
        node = RepeatNode(child, cell_size=Vec3(2.0, 2.0, 2.0))
        assert node.cell_size.as_tuple() == (2.0, 2.0, 2.0)
        assert node.children() == (child,)
        assert node.wgsl_function == "domain_repeat"

    def test_mirror_node(self):
        """Test MirrorNode creation."""
        child = BoxNode()
        node = MirrorNode(child, axis=Axis.Y)
        assert node.axis == Axis.Y
        assert node.wgsl_function == "domain_mirror_y"

    def test_kifs_node(self):
        """Test KIFSNode creation."""
        child = TorusNode()
        node = KIFSNode(child, iterations=8, scale=1.8)
        assert node.iterations == 8
        assert node.scale == 1.8
        assert node.wgsl_function == "domain_fold_kifs"

    def test_twist_node(self):
        """Test TwistNode creation."""
        child = CylinderNode()
        node = TwistNode(child, axis=Axis.Z, rate=0.3)
        assert node.axis == Axis.Z
        assert node.rate == 0.3
        assert node.wgsl_function == "domain_twist_z"

    def test_bend_node(self):
        """Test BendNode creation."""
        child = BoxNode()
        node = BendNode(child, axis=Axis.X, radius=5.0)
        assert node.axis == Axis.X
        assert node.radius == 5.0
        assert node.wgsl_function == "domain_bend_x"

    def test_stretch_node(self):
        """Test StretchNode creation."""
        child = SphereNode()
        node = StretchNode(child, axis=Axis.Y, scale=3.0)
        assert node.axis == Axis.Y
        assert node.scale == 3.0
        assert node.wgsl_function == "domain_stretch_y"


# =============================================================================
# Scene Node Tests (4 tests)
# =============================================================================

class TestSceneNodes:
    """Tests for scene node creation."""

    def test_camera_node(self):
        """Test CameraNode creation."""
        node = CameraNode(
            origin=Vec3(0.0, 0.0, 5.0),
            look_at=Vec3(0.0, 0.0, 0.0),
            fov=90.0,
        )
        assert node.origin.as_tuple() == (0.0, 0.0, 5.0)
        assert node.fov == 90.0

    def test_light_node(self):
        """Test LightNode creation."""
        node = LightNode(
            position=Vec3(10.0, 10.0, 10.0),
            color=Vec3(1.0, 0.9, 0.8),
            intensity=2.0,
        )
        assert node.intensity == 2.0

    def test_material_node(self):
        """Test MaterialNode creation."""
        node = MaterialNode(
            color=Vec3(0.9, 0.1, 0.1),
            metallic=0.8,
            roughness=0.2,
            material_id=5,
        )
        assert node.metallic == 0.8
        assert node.material_id == 5

    def test_scene_node(self):
        """Test SceneNode creation."""
        root = SphereNode()
        camera = CameraNode()
        light = LightNode()
        settings = RenderSettingsNode()

        scene = SceneNode(
            root=root,
            camera=camera,
            lights=[light],
            render_settings=settings,
            name="test_scene",
        )

        assert scene.root is root
        assert scene.camera is camera
        assert scene.name == "test_scene"
        assert len(scene.lights) == 1


# =============================================================================
# Mirror Introspection Tests (T-DEMO-2.2) (7 tests)
# =============================================================================

class TestMirrorIntrospection:
    """Tests for Mirror introspection system."""

    def test_mirror_node_type(self):
        """Test Mirror.node_type property."""
        node = SphereNode(radius=1.5)
        mirror = node.mirror
        assert mirror.node_type == "SphereNode"

    def test_mirror_node_id(self):
        """Test Mirror.node_id property."""
        node1 = SphereNode()
        node2 = BoxNode()
        assert node1.mirror.node_id != node2.mirror.node_id

    def test_mirror_fields(self):
        """Test Mirror.fields property."""
        node = SphereNode(radius=2.0)
        fields = node.mirror.fields
        assert "radius" in fields
        assert fields["radius"] == 2.0

    def test_mirror_children(self):
        """Test Mirror.children property."""
        left = SphereNode()
        right = BoxNode()
        union = UnionNode(left, right)
        children = union.mirror.children
        assert len(children) == 2
        assert left in children
        assert right in children

    def test_mirror_is_dirty(self):
        """Test Mirror.is_dirty property."""
        node = SphereNode()
        # Node is marked dirty on creation
        assert node.mirror.is_dirty
        node.tracker.clear()
        assert not node.mirror.is_dirty

    def test_mirror_metadata(self):
        """Test Mirror.metadata property."""
        node = SphereNode()
        meta = node.mirror.metadata
        assert "node_type" in meta
        assert "node_id" in meta
        assert "is_dirty" in meta
        assert "child_count" in meta

    def test_mirror_walk(self):
        """Test Mirror.walk() tree traversal."""
        left = SphereNode()
        right = BoxNode()
        union = UnionNode(left, right)

        walked = list(union.mirror.walk())
        assert len(walked) == 3
        # First is root
        assert walked[0][0] is union
        assert walked[0][1] == 0  # depth


# =============================================================================
# Tracker Dirty Tracking Tests (T-DEMO-2.2) (8 tests)
# =============================================================================

class TestTrackerDirtyTracking:
    """Tests for Tracker dirty tracking system."""

    def test_tracker_initial_dirty(self):
        """Test that new nodes are marked dirty."""
        node = SphereNode()
        assert node.tracker.is_dirty
        assert len(node.tracker.dirty_fields) > 0

    def test_tracker_clear(self):
        """Test Tracker.clear()."""
        node = SphereNode()
        node.tracker.clear()
        assert not node.tracker.is_dirty
        assert len(node.tracker.dirty_fields) == 0

    def test_tracker_mark_dirty(self):
        """Test Tracker.mark_dirty()."""
        node = SphereNode()
        node.tracker.clear()
        initial_version = node.tracker.version

        node.tracker.mark_dirty("radius")

        assert "radius" in node.tracker.dirty_fields
        assert node.tracker.version > initial_version

    def test_tracker_mark_all_dirty(self):
        """Test Tracker.mark_all_dirty()."""
        node = SphereNode()
        node.tracker.clear()

        node.tracker.mark_all_dirty()

        assert node.tracker.is_dirty

    def test_tracker_version_increment(self):
        """Test that version increments on changes."""
        node = SphereNode()
        node.tracker.clear()
        v1 = node.tracker.version

        node.tracker.mark_dirty("radius")
        v2 = node.tracker.version

        node.tracker.mark_dirty("position")
        v3 = node.tracker.version

        assert v2 > v1
        assert v3 > v2

    def test_tracker_recursive_dirty(self):
        """Test Tracker.is_dirty with children."""
        left = SphereNode()
        right = BoxNode()
        left.tracker.clear()
        right.tracker.clear()

        union = UnionNode(left, right)
        union.tracker.clear()

        # Mark child dirty
        left.tracker.mark_dirty("radius")

        # Parent should report dirty because child is dirty
        assert union.tracker.is_dirty

    def test_tracker_clear_recursive(self):
        """Test Tracker.clear_recursive()."""
        left = SphereNode()
        right = BoxNode()
        union = UnionNode(left, right)

        union.tracker.clear_recursive()

        assert not union.tracker.is_dirty
        assert not left.tracker.is_dirty
        assert not right.tracker.is_dirty

    def test_tracker_get_dirty_tree(self):
        """Test Tracker.get_dirty_tree()."""
        left = SphereNode()
        right = BoxNode()
        union = UnionNode(left, right)

        # Clear all, then mark specific fields dirty
        union.tracker.clear_recursive()
        left.tracker.mark_dirty("radius")

        dirty_tree = union.tracker.get_dirty_tree()

        # Should contain left node
        dirty_nodes = [node for node, _ in dirty_tree]
        assert left in dirty_nodes


# =============================================================================
# build_ast() Function Tests (10 tests)
# =============================================================================

class TestBuildAst:
    """Tests for build_ast() function."""

    def test_build_sphere_from_dict(self):
        """Test building sphere from dictionary."""
        node = build_ast({"type": "sphere", "radius": 2.5})
        assert isinstance(node, SphereNode)
        assert node.radius == 2.5

    def test_build_box_from_dict(self):
        """Test building box from dictionary."""
        node = build_ast({
            "type": "box",
            "half_extents": [1.0, 2.0, 3.0],
        })
        assert isinstance(node, BoxNode)
        assert node.half_extents.as_tuple() == (1.0, 2.0, 3.0)

    def test_build_union_from_dict(self):
        """Test building union from dictionary."""
        node = build_ast({
            "type": "union",
            "left": {"type": "sphere"},
            "right": {"type": "box"},
        })
        assert isinstance(node, UnionNode)
        assert isinstance(node.left, SphereNode)
        assert isinstance(node.right, BoxNode)

    def test_build_smooth_union_from_dict(self):
        """Test building smooth union from dictionary."""
        node = build_ast({
            "type": "smooth_union",
            "left": {"type": "sphere"},
            "right": {"type": "box"},
            "k": 0.3,
        })
        assert isinstance(node, SmoothUnionNode)
        assert node.k == 0.3

    def test_build_repeat_from_dict(self):
        """Test building repeat from dictionary."""
        node = build_ast({
            "type": "repeat",
            "child": {"type": "sphere"},
            "cell_size": [4.0, 4.0, 4.0],
        })
        assert isinstance(node, RepeatNode)
        assert node.cell_size.as_tuple() == (4.0, 4.0, 4.0)

    def test_build_mirror_from_dict(self):
        """Test building mirror from dictionary."""
        node = build_ast({
            "type": "mirror",
            "child": {"type": "box"},
            "axis": "y",
        })
        assert isinstance(node, MirrorNode)
        assert node.axis == Axis.Y

    def test_build_scene_from_dict(self):
        """Test building complete scene from dictionary."""
        node = build_ast({
            "type": "scene",
            "root": {
                "type": "union",
                "left": {"type": "sphere", "radius": 1.0},
                "right": {"type": "box"},
            },
            "camera": {
                "type": "camera",
                "origin": [0, 0, 10],
                "fov": 75.0,
            },
            "name": "test",
        })
        assert isinstance(node, SceneNode)
        assert isinstance(node.root, UnionNode)
        assert node.camera.fov == 75.0
        assert node.name == "test"

    def test_build_nested_domain_ops(self):
        """Test building nested domain operations."""
        node = build_ast({
            "type": "repeat",
            "child": {
                "type": "mirror",
                "child": {"type": "sphere"},
                "axis": "x",
            },
            "cell_size": [2, 2, 2],
        })
        assert isinstance(node, RepeatNode)
        assert isinstance(node.child, MirrorNode)
        assert isinstance(node.child.child, SphereNode)

    def test_build_passthrough_node(self):
        """Test that existing nodes pass through unchanged."""
        original = SphereNode(radius=5.0)
        result = build_ast(original)
        assert result is original

    def test_build_all_primitives(self):
        """Test building all primitive types."""
        primitives = [
            ("sphere", SphereNode),
            ("box", BoxNode),
            ("torus", TorusNode),
            ("cylinder", CylinderNode),
            ("cone", ConeNode),
            ("plane", PlaneNode),
            ("capsule", CapsuleNode),
            ("ellipsoid", EllipsoidNode),
            ("box_frame", BoxFrameNode),
            ("rounded_box", RoundedBoxNode),
            ("octahedron", OctahedronNode),
            ("pyramid", PyramidNode),
        ]
        for type_name, expected_class in primitives:
            node = build_ast({"type": type_name})
            assert isinstance(node, expected_class), f"Failed for {type_name}"


# =============================================================================
# Tree Traversal Tests (4 tests)
# =============================================================================

class TestTreeTraversal:
    """Tests for tree traversal functionality."""

    def test_walk_single_node(self):
        """Test walking a single node."""
        node = SphereNode()
        walked = list(node.walk())
        assert len(walked) == 1
        assert walked[0] == (node, 0)

    def test_walk_binary_tree(self):
        """Test walking a binary tree."""
        left = SphereNode()
        right = BoxNode()
        root = UnionNode(left, right)

        walked = list(root.walk())
        nodes = [n for n, d in walked]
        depths = [d for n, d in walked]

        assert len(walked) == 3
        assert root in nodes
        assert left in nodes
        assert right in nodes
        assert depths == [0, 1, 1]

    def test_walk_deep_tree(self):
        """Test walking a deep tree."""
        node = SphereNode()
        for _ in range(5):
            node = MirrorNode(node, Axis.X)

        walked = list(node.walk())
        assert len(walked) == 6
        depths = [d for n, d in walked]
        assert depths == [0, 1, 2, 3, 4, 5]

    def test_pretty_print(self):
        """Test pretty printing."""
        left = SphereNode(radius=1.0)
        right = BoxNode()
        root = UnionNode(left, right)

        pretty = root.pretty()
        assert "Union" in pretty
        assert "Sphere" in pretty
        assert "Box" in pretty


# =============================================================================
# Node Cloning Tests (3 tests)
# =============================================================================

class TestNodeCloning:
    """Tests for node cloning functionality."""

    def test_clone_primitive(self):
        """Test cloning a primitive node."""
        original = SphereNode(radius=2.5, position=Vec3(1.0, 2.0, 3.0))
        clone = original.clone()

        assert clone is not original
        assert clone.radius == original.radius
        assert clone.position.as_tuple() == original.position.as_tuple()
        assert clone._node_id != original._node_id

    def test_clone_combinator(self):
        """Test cloning a combinator node."""
        original = UnionNode(SphereNode(), BoxNode())
        clone = original.clone()

        assert clone is not original
        assert clone.left is not original.left
        assert clone.right is not original.right

    def test_clone_scene(self):
        """Test cloning a scene node."""
        original = SceneNode(
            root=SphereNode(),
            camera=CameraNode(),
            name="original",
        )
        clone = original.clone()

        assert clone is not original
        assert clone.root is not original.root
        assert clone.camera is not original.camera
        assert clone.name == original.name


# =============================================================================
# Metaclass Tests (3 tests)
# =============================================================================

class TestSDFNodeMeta:
    """Tests for SDFNodeMeta metaclass."""

    def test_unique_type_ids(self):
        """Test that each node type has a unique type ID."""
        types = [
            SphereNode, BoxNode, TorusNode, CylinderNode,
            ConeNode, PlaneNode, CapsuleNode, EllipsoidNode,
        ]
        type_ids = [t._node_type_id for t in types]
        assert len(type_ids) == len(set(type_ids))

    def test_field_registration(self):
        """Test that fields are registered on node types."""
        # SphereNode should have radius field registered
        assert "radius" in SphereNode._field_names or hasattr(SphereNode, "radius")

    def test_get_all_node_types(self):
        """Test getting all registered node types."""
        all_types = SDFNodeMeta.get_all_node_types()
        assert len(all_types) > 0


# =============================================================================
# Edge Cases and Error Handling (4 tests)
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_build_unknown_type_raises(self):
        """Test that unknown type raises ValueError."""
        with pytest.raises(ValueError):
            build_ast({"type": "unknown_shape"})

    def test_build_invalid_input_raises(self):
        """Test that invalid input raises TypeError."""
        with pytest.raises(TypeError):
            build_ast(12345)  # Not a dict, node, or DSL object

    def test_normalized_plane_normal(self):
        """Test that plane normal is normalized."""
        node = PlaneNode(normal=Vec3(3.0, 4.0, 0.0))
        assert abs(node.normal.length() - 1.0) < 1e-6

    def test_vec3_normalized_zero_length(self):
        """Test normalizing zero-length vector."""
        v = Vec3(0.0, 0.0, 0.0)
        n = v.normalized()
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.z == 0.0
