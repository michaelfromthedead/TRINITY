"""
Tests for SDF Error Reporting (T-DEMO-2.14).

Comprehensive test coverage for:
  - Exception hierarchy
  - SDFValidator
  - All primitive validators
  - All combinator validators
  - All domain operation validators
  - Scene, camera, light, material, render settings validators
  - Recursion detection
  - Type errors
  - Impossible SDF detection
  - Error message formatting
  - Integration with validation functions
"""

import math
import pytest

from engine.rendering.demoscene.sdf_ast import (
    Axis,
    BendNode,
    BoxFrameNode,
    BoxNode,
    CameraNode,
    CapsuleNode,
    ConeNode,
    CylinderNode,
    DisplacedNode,
    EllipsoidNode,
    IntersectionNode,
    KIFSNode,
    LightNode,
    MaterialNode,
    MirrorNode,
    OctahedronNode,
    PlaneNode,
    PyramidNode,
    RenderSettingsNode,
    RepeatNode,
    RoundedBoxNode,
    SceneNode,
    SmoothIntersectionNode,
    SmoothSubtractionNode,
    SmoothUnionNode,
    SphereNode,
    StretchNode,
    SubtractionNode,
    TorusNode,
    TwistNode,
    UnionNode,
    Vec3,
)
from engine.rendering.demoscene.sdf_errors import (
    SDFError,
    SDFCompilationError,
    SDFImpossibleSDFError,
    SDFRecursionError,
    SDFTypeError,
    SDFValidationError,
    SDFValidator,
    Severity,
    ValidationIssue,
    ValidatingCompiler,
    get_validation_report,
    is_scene_valid,
    validate_scene,
    validate_scene_strict,
)


# =============================================================================
# EXCEPTION HIERARCHY TESTS
# =============================================================================

class TestExceptionHierarchy:
    """Tests for the SDFError exception hierarchy."""

    def test_sdf_error_is_exception(self):
        """SDFError should inherit from Exception."""
        assert issubclass(SDFError, Exception)

    def test_all_errors_inherit_from_sdf_error(self):
        """All SDF error types should inherit from SDFError."""
        assert issubclass(SDFValidationError, SDFError)
        assert issubclass(SDFCompilationError, SDFError)
        assert issubclass(SDFTypeError, SDFError)
        assert issubclass(SDFRecursionError, SDFError)
        assert issubclass(SDFImpossibleSDFError, SDFError)

    def test_error_message_formatting(self):
        """Error messages should include path and suggestion."""
        error = SDFError(
            message="test error",
            path="/scene/root/sphere",
            suggestion="fix it",
        )
        msg = str(error)
        assert "/scene/root/sphere:" in msg
        assert "test error" in msg
        assert "fix it" in msg

    def test_error_message_without_path(self):
        """Error messages should work without a path."""
        error = SDFError(message="test error")
        assert str(error) == "test error"

    def test_recursion_error_shows_cycle(self):
        """Recursion errors should show the cycle path."""
        error = SDFRecursionError(
            message="cycle detected",
            cycle=[1, 2, 3, 1],
        )
        msg = str(error)
        assert "cycle:" in msg
        assert "1 -> 2 -> 3 -> 1" in msg

    def test_error_repr(self):
        """Error repr should show class name and message."""
        error = SDFValidationError(
            message="invalid value",
            path="/scene",
        )
        r = repr(error)
        assert "SDFValidationError" in r
        assert "invalid value" in r


# =============================================================================
# PRIMITIVE VALIDATION TESTS
# =============================================================================

class TestSphereValidation:
    """Tests for SphereNode validation."""

    def test_valid_sphere(self):
        """A valid sphere should pass validation."""
        sphere = SphereNode(radius=1.0)
        scene = SceneNode(root=sphere)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_negative_radius_sphere(self):
        """Sphere with negative radius should fail."""
        sphere = SphereNode(radius=-1.0)
        scene = SceneNode(root=sphere)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert isinstance(errors[0], SDFImpossibleSDFError)
        assert "radius must be positive" in errors[0].message
        assert "-1.0" in errors[0].message

    def test_zero_radius_sphere(self):
        """Sphere with zero radius should fail."""
        sphere = SphereNode(radius=0.0)
        scene = SceneNode(root=sphere)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "radius must be positive" in errors[0].message


class TestBoxValidation:
    """Tests for BoxNode validation."""

    def test_valid_box(self):
        """A valid box should pass validation."""
        box = BoxNode(half_extents=Vec3(1.0, 1.0, 1.0))
        scene = SceneNode(root=box)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_dimension_box(self):
        """Box with zero dimension should fail."""
        box = BoxNode(half_extents=Vec3(0.0, 1.0, 1.0))
        scene = SceneNode(root=box)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "half_extents.x must be positive" in errors[0].message

    def test_negative_dimension_box(self):
        """Box with negative dimension should fail."""
        box = BoxNode(half_extents=Vec3(1.0, -0.5, 1.0))
        scene = SceneNode(root=box)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "half_extents.y must be positive" in errors[0].message


class TestTorusValidation:
    """Tests for TorusNode validation."""

    def test_valid_torus(self):
        """A valid torus should pass validation."""
        torus = TorusNode(major_radius=1.0, minor_radius=0.25)
        scene = SceneNode(root=torus)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_negative_major_radius(self):
        """Torus with negative major radius should fail."""
        torus = TorusNode(major_radius=-1.0, minor_radius=0.25)
        scene = SceneNode(root=torus)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "major_radius must be positive" in errors[0].message


class TestCylinderValidation:
    """Tests for CylinderNode validation."""

    def test_valid_cylinder(self):
        """A valid cylinder should pass validation."""
        cylinder = CylinderNode(radius=0.5, height=1.0)
        scene = SceneNode(root=cylinder)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_height_cylinder(self):
        """Cylinder with zero height should fail."""
        cylinder = CylinderNode(radius=0.5, height=0.0)
        scene = SceneNode(root=cylinder)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "height must be positive" in errors[0].message


class TestConeValidation:
    """Tests for ConeNode validation."""

    def test_valid_cone(self):
        """A valid cone should pass validation."""
        cone = ConeNode(angle=0.5, height=1.0)
        scene = SceneNode(root=cone)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_invalid_angle_cone(self):
        """Cone with angle >= pi/2 should fail."""
        cone = ConeNode(angle=math.pi, height=1.0)
        scene = SceneNode(root=cone)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "angle must be in range" in errors[0].message


class TestPlaneValidation:
    """Tests for PlaneNode validation."""

    def test_valid_plane(self):
        """A valid plane should pass validation."""
        plane = PlaneNode(normal=Vec3(0.0, 1.0, 0.0), distance=0.0)
        scene = SceneNode(root=plane)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_normal_plane(self):
        """Plane with zero normal should fail."""
        plane = PlaneNode()
        # Force zero normal
        plane.normal = Vec3(0.0, 0.0, 0.0)
        scene = SceneNode(root=plane)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "normal must not be zero" in errors[0].message


class TestCapsuleValidation:
    """Tests for CapsuleNode validation."""

    def test_valid_capsule(self):
        """A valid capsule should pass validation."""
        capsule = CapsuleNode(
            endpoint_a=Vec3(0.0, -0.5, 0.0),
            endpoint_b=Vec3(0.0, 0.5, 0.0),
            radius=0.25,
        )
        scene = SceneNode(root=capsule)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_negative_radius_capsule(self):
        """Capsule with negative radius should fail."""
        capsule = CapsuleNode(radius=-0.1)
        scene = SceneNode(root=capsule)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "radius must be positive" in errors[0].message


class TestEllipsoidValidation:
    """Tests for EllipsoidNode validation."""

    def test_valid_ellipsoid(self):
        """A valid ellipsoid should pass validation."""
        ellipsoid = EllipsoidNode(radii=Vec3(1.0, 1.5, 1.0))
        scene = SceneNode(root=ellipsoid)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_radii_ellipsoid(self):
        """Ellipsoid with zero radii should fail."""
        ellipsoid = EllipsoidNode(radii=Vec3(0.0, 1.0, 1.0))
        scene = SceneNode(root=ellipsoid)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "radii.x must be positive" in errors[0].message


class TestBoxFrameValidation:
    """Tests for BoxFrameNode validation."""

    def test_valid_box_frame(self):
        """A valid box frame should pass validation."""
        frame = BoxFrameNode(
            half_extents=Vec3(1.0, 1.0, 1.0),
            edge_thickness=0.05,
        )
        scene = SceneNode(root=frame)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_thickness_box_frame(self):
        """BoxFrame with zero thickness should fail."""
        frame = BoxFrameNode(edge_thickness=0.0)
        scene = SceneNode(root=frame)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "edge_thickness must be positive" in errors[0].message


class TestRoundedBoxValidation:
    """Tests for RoundedBoxNode validation."""

    def test_valid_rounded_box(self):
        """A valid rounded box should pass validation."""
        box = RoundedBoxNode(
            half_extents=Vec3(1.0, 1.0, 1.0),
            corner_radius=0.1,
        )
        scene = SceneNode(root=box)
        errors = validate_scene(scene)
        assert len(errors) == 0


class TestOctahedronValidation:
    """Tests for OctahedronNode validation."""

    def test_valid_octahedron(self):
        """A valid octahedron should pass validation."""
        octa = OctahedronNode(size=1.0)
        scene = SceneNode(root=octa)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_size_octahedron(self):
        """Octahedron with zero size should fail."""
        octa = OctahedronNode(size=0.0)
        scene = SceneNode(root=octa)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "size must be positive" in errors[0].message


class TestPyramidValidation:
    """Tests for PyramidNode validation."""

    def test_valid_pyramid(self):
        """A valid pyramid should pass validation."""
        pyramid = PyramidNode(height=1.0)
        scene = SceneNode(root=pyramid)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_negative_height_pyramid(self):
        """Pyramid with negative height should fail."""
        pyramid = PyramidNode(height=-1.0)
        scene = SceneNode(root=pyramid)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "height must be positive" in errors[0].message


# =============================================================================
# COMBINATOR VALIDATION TESTS
# =============================================================================

class TestCombinatorValidation:
    """Tests for combinator node validation."""

    def test_valid_union(self):
        """A valid union should pass validation."""
        union = UnionNode(SphereNode(radius=1.0), BoxNode())
        scene = SceneNode(root=union)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_valid_intersection(self):
        """A valid intersection should pass validation."""
        inter = IntersectionNode(SphereNode(radius=1.0), BoxNode())
        scene = SceneNode(root=inter)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_valid_subtraction(self):
        """A valid subtraction should pass validation."""
        sub = SubtractionNode(BoxNode(), SphereNode(radius=0.5))
        scene = SceneNode(root=sub)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_invalid_left_child_propagates(self):
        """Errors in left child should be detected."""
        union = UnionNode(
            SphereNode(radius=-1.0),  # Invalid
            BoxNode(),
        )
        scene = SceneNode(root=union)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "radius must be positive" in errors[0].message

    def test_invalid_right_child_propagates(self):
        """Errors in right child should be detected."""
        union = UnionNode(
            SphereNode(radius=1.0),
            BoxNode(half_extents=Vec3(0.0, 1.0, 1.0)),  # Invalid
        )
        scene = SceneNode(root=union)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "half_extents.x must be positive" in errors[0].message


class TestSmoothCombinatorValidation:
    """Tests for smooth combinator validation."""

    def test_valid_smooth_union(self):
        """A valid smooth union should pass validation."""
        smooth = SmoothUnionNode(SphereNode(), BoxNode(), k=0.1)
        scene = SceneNode(root=smooth)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_negative_k_smooth_union(self):
        """Smooth union with negative k should fail."""
        smooth = SmoothUnionNode(SphereNode(), BoxNode(), k=-0.1)
        scene = SceneNode(root=smooth)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "k must be non-negative" in errors[0].message


# =============================================================================
# DOMAIN OPERATION VALIDATION TESTS
# =============================================================================

class TestRepeatValidation:
    """Tests for RepeatNode validation."""

    def test_valid_repeat(self):
        """A valid repeat should pass validation."""
        repeat = RepeatNode(SphereNode(), cell_size=Vec3(2.0, 2.0, 2.0))
        scene = SceneNode(root=repeat)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_cell_size_repeat(self):
        """Repeat with zero cell size should fail."""
        repeat = RepeatNode(SphereNode(), cell_size=Vec3(0.0, 2.0, 2.0))
        scene = SceneNode(root=repeat)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "cell_size must be non-zero" in errors[0].message
        assert "x" in errors[0].message


class TestBendValidation:
    """Tests for BendNode validation."""

    def test_valid_bend(self):
        """A valid bend should pass validation."""
        bend = BendNode(BoxNode(), radius=10.0)
        scene = SceneNode(root=bend)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_radius_bend(self):
        """Bend with zero radius should fail."""
        bend = BendNode(BoxNode(), radius=0.0)
        scene = SceneNode(root=bend)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "bend radius must be non-zero" in errors[0].message


class TestKIFSValidation:
    """Tests for KIFSNode validation."""

    def test_valid_kifs(self):
        """A valid KIFS should pass validation."""
        kifs = KIFSNode(BoxNode(), iterations=6, scale=2.0)
        scene = SceneNode(root=kifs)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_iterations_kifs(self):
        """KIFS with zero iterations should fail."""
        kifs = KIFSNode(BoxNode(), iterations=0, scale=2.0)
        scene = SceneNode(root=kifs)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "iterations must be at least 1" in errors[0].message

    def test_negative_scale_kifs(self):
        """KIFS with negative scale should fail."""
        kifs = KIFSNode(BoxNode(), iterations=6, scale=-1.0)
        scene = SceneNode(root=kifs)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "scale must be positive" in errors[0].message


class TestStretchValidation:
    """Tests for StretchNode validation."""

    def test_valid_stretch(self):
        """A valid stretch should pass validation."""
        stretch = StretchNode(SphereNode(), scale=2.0)
        scene = SceneNode(root=stretch)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_scale_stretch(self):
        """Stretch with zero scale should fail."""
        stretch = StretchNode(SphereNode(), scale=0.0)
        scene = SceneNode(root=stretch)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "scale must be positive" in errors[0].message


class TestDisplacedValidation:
    """Tests for DisplacedNode validation."""

    def test_valid_displaced(self):
        """A valid displaced should pass validation."""
        displaced = DisplacedNode(SphereNode(), amplitude=0.1, frequency=1.0)
        scene = SceneNode(root=displaced)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_frequency_displaced(self):
        """Displaced with zero frequency should fail."""
        displaced = DisplacedNode(SphereNode(), amplitude=0.1, frequency=0.0)
        scene = SceneNode(root=displaced)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "frequency must be positive" in errors[0].message


# =============================================================================
# CAMERA VALIDATION TESTS
# =============================================================================

class TestCameraValidation:
    """Tests for CameraNode validation."""

    def test_valid_camera(self):
        """A valid camera should pass validation."""
        camera = CameraNode(
            origin=Vec3(0.0, 0.0, 5.0),
            look_at=Vec3(0.0, 0.0, 0.0),
            fov=60.0,
        )
        scene = SceneNode(root=SphereNode(), camera=camera)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_invalid_fov_camera(self):
        """Camera with FOV >= 180 should fail."""
        camera = CameraNode(fov=180.0)
        scene = SceneNode(root=SphereNode(), camera=camera)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "fov must be in range" in errors[0].message

    def test_zero_fov_camera(self):
        """Camera with FOV <= 0 should fail."""
        camera = CameraNode(fov=0.0)
        scene = SceneNode(root=SphereNode(), camera=camera)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "fov must be in range" in errors[0].message

    def test_zero_aspect_ratio_camera(self):
        """Camera with zero aspect ratio should fail."""
        camera = CameraNode(aspect_ratio=0.0)
        scene = SceneNode(root=SphereNode(), camera=camera)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "aspect_ratio must be positive" in errors[0].message

    def test_zero_up_vector_camera(self):
        """Camera with zero up vector should fail."""
        camera = CameraNode()
        camera.up = Vec3(0.0, 0.0, 0.0)
        scene = SceneNode(root=SphereNode(), camera=camera)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "up must not be zero" in errors[0].message


# =============================================================================
# LIGHT VALIDATION TESTS
# =============================================================================

class TestLightValidation:
    """Tests for LightNode validation."""

    def test_valid_light(self):
        """A valid light should pass validation."""
        light = LightNode(
            position=Vec3(5.0, 5.0, 5.0),
            color=Vec3(1.0, 1.0, 1.0),
            intensity=1.0,
        )
        scene = SceneNode(root=SphereNode(), lights=[light])
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_negative_intensity_light(self):
        """Light with negative intensity should fail."""
        light = LightNode(intensity=-1.0)
        scene = SceneNode(root=SphereNode(), lights=[light])
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "intensity must be non-negative" in errors[0].message


# =============================================================================
# MATERIAL VALIDATION TESTS
# =============================================================================

class TestMaterialValidation:
    """Tests for MaterialNode validation."""

    def test_valid_material(self):
        """A valid material should pass validation."""
        material = MaterialNode(
            color=Vec3(0.8, 0.2, 0.1),
            metallic=0.5,
            roughness=0.5,
        )
        scene = SceneNode(root=SphereNode(), materials=[material])
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_out_of_range_metallic(self):
        """Material with metallic > 1 should fail."""
        material = MaterialNode(metallic=1.5)
        scene = SceneNode(root=SphereNode(), materials=[material])
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "metallic must be in range" in errors[0].message

    def test_out_of_range_roughness(self):
        """Material with roughness < 0 should fail."""
        material = MaterialNode(roughness=-0.1)
        scene = SceneNode(root=SphereNode(), materials=[material])
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "roughness must be in range" in errors[0].message

    def test_negative_material_id(self):
        """Material with negative ID should fail."""
        material = MaterialNode(material_id=-1)
        scene = SceneNode(root=SphereNode(), materials=[material])
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "material_id must be non-negative" in errors[0].message


# =============================================================================
# RENDER SETTINGS VALIDATION TESTS
# =============================================================================

class TestRenderSettingsValidation:
    """Tests for RenderSettingsNode validation."""

    def test_valid_render_settings(self):
        """Valid render settings should pass validation."""
        settings = RenderSettingsNode(
            width=1920,
            height=1080,
            max_steps=256,
            epsilon=0.001,
        )
        scene = SceneNode(root=SphereNode(), render_settings=settings)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_zero_width(self):
        """Render settings with zero width should fail."""
        settings = RenderSettingsNode(width=0)
        scene = SceneNode(root=SphereNode(), render_settings=settings)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "width must be positive" in errors[0].message

    def test_zero_max_steps(self):
        """Render settings with zero max steps should fail."""
        settings = RenderSettingsNode(max_steps=0)
        scene = SceneNode(root=SphereNode(), render_settings=settings)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "max_steps must be positive" in errors[0].message

    def test_zero_epsilon(self):
        """Render settings with zero epsilon should fail."""
        settings = RenderSettingsNode(epsilon=0.0)
        scene = SceneNode(root=SphereNode(), render_settings=settings)
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "epsilon must be positive" in errors[0].message


# =============================================================================
# RECURSION DETECTION TESTS
# =============================================================================

class TestRecursionDetection:
    """Tests for infinite recursion detection."""

    def test_deep_but_valid_tree(self):
        """A deep tree without cycles should pass validation."""
        # Create a deep nested structure
        node = SphereNode()
        for _ in range(50):
            node = UnionNode(node, SphereNode(radius=0.1))
        scene = SceneNode(root=node)
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_max_depth_exceeded(self):
        """Exceeding max depth should raise error."""
        # Create a very deep tree
        node = SphereNode()
        for _ in range(100):
            node = UnionNode(node, SphereNode(radius=0.1))
        scene = SceneNode(root=node)

        # Use validator with low max_depth
        validator = SDFValidator(max_depth=50)
        errors = validator.validate(scene)
        # Multiple recursion errors may be raised (left and right branches)
        assert len(errors) >= 1
        assert all(isinstance(e, SDFRecursionError) for e in errors)
        assert "maximum recursion depth" in errors[0].message


# =============================================================================
# ERROR PATH TESTS
# =============================================================================

class TestErrorPaths:
    """Tests for error path formatting in messages."""

    def test_error_path_for_primitive(self):
        """Error path should show primitive location."""
        sphere = SphereNode(radius=-1.0)
        scene = SceneNode(root=sphere, name="test")
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "SphereNode" in errors[0].path

    def test_error_path_for_nested_combinator(self):
        """Error path should show nested combinator location."""
        inner = SphereNode(radius=-1.0)  # Invalid
        union = UnionNode(inner, BoxNode())
        scene = SceneNode(root=union, name="test")
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "UnionNode" in errors[0].path
        assert "SphereNode" in errors[0].path

    def test_error_path_for_light(self):
        """Error path should show light location."""
        light = LightNode(intensity=-1.0)
        scene = SceneNode(root=SphereNode(), lights=[light])
        errors = validate_scene(scene)
        assert len(errors) == 1
        assert "lights[0]" in errors[0].path


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience validation functions."""

    def test_validate_scene_returns_errors(self):
        """validate_scene should return list of errors."""
        scene = SceneNode(root=SphereNode(radius=-1.0))
        errors = validate_scene(scene)
        assert isinstance(errors, list)
        assert len(errors) == 1

    def test_validate_scene_strict_raises(self):
        """validate_scene_strict should raise on error."""
        scene = SceneNode(root=SphereNode(radius=-1.0))
        with pytest.raises(SDFImpossibleSDFError):
            validate_scene_strict(scene)

    def test_validate_scene_strict_passes_valid(self):
        """validate_scene_strict should not raise for valid scene."""
        scene = SceneNode(root=SphereNode(radius=1.0))
        validate_scene_strict(scene)  # Should not raise

    def test_is_scene_valid_returns_bool(self):
        """is_scene_valid should return boolean."""
        valid_scene = SceneNode(root=SphereNode(radius=1.0))
        invalid_scene = SceneNode(root=SphereNode(radius=-1.0))
        assert is_scene_valid(valid_scene) is True
        assert is_scene_valid(invalid_scene) is False

    def test_get_validation_report_for_valid_scene(self):
        """get_validation_report should report no issues for valid scene."""
        scene = SceneNode(root=SphereNode())
        report = get_validation_report(scene)
        assert "no issues found" in report

    def test_get_validation_report_for_invalid_scene(self):
        """get_validation_report should list issues for invalid scene."""
        scene = SceneNode(root=SphereNode(radius=-1.0))
        report = get_validation_report(scene)
        assert "ERRORS" in report
        assert "radius must be positive" in report


# =============================================================================
# VALIDATOR CLASS TESTS
# =============================================================================

class TestSDFValidator:
    """Tests for the SDFValidator class."""

    def test_validate_returns_list(self):
        """validate should return a list."""
        validator = SDFValidator()
        scene = SceneNode(root=SphereNode())
        result = validator.validate(scene)
        assert isinstance(result, list)

    def test_validate_all_includes_warnings(self):
        """validate_all should include warnings."""
        # Create a scene with a warning (torus minor >= major)
        torus = TorusNode(major_radius=1.0, minor_radius=1.5)
        scene = SceneNode(root=torus)
        validator = SDFValidator()
        issues = validator.validate_all(scene)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert len(warnings) >= 1

    def test_is_valid_returns_bool(self):
        """is_valid should return boolean."""
        validator = SDFValidator()
        valid_scene = SceneNode(root=SphereNode())
        assert validator.is_valid(valid_scene) is True


# =============================================================================
# VALIDATING COMPILER TESTS
# =============================================================================

class MockCompiler:
    """Mock compiler for testing ValidatingCompiler."""

    def compile(self, scene, *args, **kwargs):
        return "compiled"


class TestValidatingCompiler:
    """Tests for the ValidatingCompiler wrapper."""

    def test_compiles_valid_scene(self):
        """ValidatingCompiler should compile valid scenes."""
        compiler = ValidatingCompiler(MockCompiler())
        scene = SceneNode(root=SphereNode())
        result = compiler.compile(scene)
        assert result == "compiled"

    def test_rejects_invalid_scene(self):
        """ValidatingCompiler should reject invalid scenes."""
        compiler = ValidatingCompiler(MockCompiler())
        scene = SceneNode(root=SphereNode(radius=-1.0))
        with pytest.raises(SDFImpossibleSDFError):
            compiler.compile(scene)

    def test_can_disable_validation(self):
        """ValidatingCompiler should allow disabling validation."""
        compiler = ValidatingCompiler(MockCompiler(), validate_before_compile=False)
        scene = SceneNode(root=SphereNode(radius=-1.0))
        result = compiler.compile(scene)  # Should not raise
        assert result == "compiled"

    def test_stores_last_errors(self):
        """ValidatingCompiler should store last errors."""
        compiler = ValidatingCompiler(MockCompiler())
        scene = SceneNode(root=SphereNode(radius=-1.0))
        try:
            compiler.compile(scene)
        except SDFError:
            pass
        assert len(compiler.last_errors) == 1


# =============================================================================
# VALIDATION ISSUE TESTS
# =============================================================================

class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_validation_issue_str(self):
        """ValidationIssue should have informative string representation."""
        error = SDFValidationError("test error", path="/scene")
        issue = ValidationIssue(
            severity=Severity.ERROR,
            error=error,
            node_id=1,
            node_type="SphereNode",
        )
        s = str(issue)
        assert "[ERROR]" in s
        assert "test error" in s

    def test_validation_issue_properties(self):
        """ValidationIssue should expose error properties."""
        error = SDFValidationError("test error", path="/scene", suggestion="fix it")
        issue = ValidationIssue(
            severity=Severity.WARNING,
            error=error,
            node_id=1,
            node_type="SphereNode",
        )
        assert issue.message == "test error"
        assert issue.path == "/scene"
        assert issue.suggestion == "fix it"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_multiple_errors_in_scene(self):
        """Multiple errors should all be collected."""
        scene = SceneNode(
            root=UnionNode(
                SphereNode(radius=-1.0),  # Error 1
                BoxNode(half_extents=Vec3(0.0, 1.0, 1.0)),  # Error 2
            ),
            camera=CameraNode(fov=0.0),  # Error 3
        )
        errors = validate_scene(scene)
        assert len(errors) == 3

    def test_valid_complex_scene(self):
        """A complex but valid scene should pass."""
        scene = SceneNode(
            root=KIFSNode(
                SmoothUnionNode(
                    RepeatNode(
                        SphereNode(radius=0.5),
                        cell_size=Vec3(3.0, 3.0, 3.0),
                    ),
                    BoxNode(half_extents=Vec3(0.5, 0.5, 0.5)),
                    k=0.2,
                ),
                iterations=4,
                scale=2.5,
            ),
            camera=CameraNode(
                origin=Vec3(0.0, 5.0, 10.0),
                look_at=Vec3(0.0, 0.0, 0.0),
                fov=45.0,
            ),
            lights=[
                LightNode(
                    position=Vec3(10.0, 10.0, 10.0),
                    intensity=1.5,
                ),
            ],
            materials=[
                MaterialNode(
                    color=Vec3(0.9, 0.1, 0.1),
                    metallic=0.8,
                    roughness=0.2,
                ),
            ],
            render_settings=RenderSettingsNode(
                width=1920,
                height=1080,
                max_steps=512,
                epsilon=0.0005,
            ),
            name="complex_demo",
        )
        errors = validate_scene(scene)
        assert len(errors) == 0

    def test_empty_lights_list(self):
        """Scene with empty lights list should be valid."""
        scene = SceneNode(root=SphereNode(), lights=[])
        errors = validate_scene(scene)
        # Default light is added by SceneNode constructor
        assert len(errors) == 0

    def test_very_small_but_valid_values(self):
        """Very small but positive values should pass."""
        scene = SceneNode(
            root=SphereNode(radius=1e-10),
            render_settings=RenderSettingsNode(epsilon=1e-10),
        )
        errors = validate_scene(scene)
        assert len(errors) == 0
