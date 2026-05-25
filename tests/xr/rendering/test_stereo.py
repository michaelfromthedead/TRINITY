"""Tests for XR stereo rendering module."""

import pytest
import math
from typing import Tuple

from engine.xr.rendering.stereo import (
    StereoMethod,
    ProjectionType,
    IPDMode,
    EyeIndex,
    ViewFrustum,
    EyeView,
    StereoConfig,
    StereoRenderTarget,
    StereoRenderer,
    MultiViewStereoRenderer,
    InstancedStereoRenderer,
    SequentialStereoRenderer,
    create_stereo_renderer,
)
from engine.xr.utils.math_utils import multiply_quaternions


class TestStereoConfig:
    """Tests for StereoConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StereoConfig()

        assert config.method == StereoMethod.MULTI_VIEW
        assert config.projection_type == ProjectionType.ASYMMETRIC
        assert config.ipd_mode == IPDMode.SOFTWARE
        assert config.ipd_meters == pytest.approx(0.063, rel=1e-3)
        assert config.world_scale == 1.0
        assert config.near_plane == pytest.approx(0.1)
        assert config.far_plane == pytest.approx(1000.0)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StereoConfig(
            method=StereoMethod.SEQUENTIAL,
            projection_type=ProjectionType.CANTED,
            ipd_meters=0.065,
            near_plane=0.05,
            far_plane=500.0,
            canting_angle=0.1
        )

        assert config.method == StereoMethod.SEQUENTIAL
        assert config.projection_type == ProjectionType.CANTED
        assert config.ipd_meters == pytest.approx(0.065)
        assert config.canting_angle == pytest.approx(0.1)


class TestViewFrustum:
    """Tests for ViewFrustum dataclass."""

    def test_default_frustum(self):
        """Test default frustum angles."""
        frustum = ViewFrustum()

        # Default is 45 degrees in each direction
        assert frustum.left == pytest.approx(-0.785398, rel=1e-3)
        assert frustum.right == pytest.approx(0.785398, rel=1e-3)
        assert frustum.top == pytest.approx(0.785398, rel=1e-3)
        assert frustum.bottom == pytest.approx(-0.785398, rel=1e-3)

    def test_asymmetric_frustum(self):
        """Test asymmetric frustum configuration."""
        frustum = ViewFrustum(
            left=-0.9,
            right=0.7,
            top=0.8,
            bottom=-0.8
        )

        assert frustum.left == pytest.approx(-0.9)
        assert frustum.right == pytest.approx(0.7)


class TestEyeView:
    """Tests for EyeView dataclass."""

    def test_left_eye_view(self):
        """Test left eye view configuration."""
        view = EyeView(
            eye=EyeIndex.LEFT,
            position_offset=(-0.032, 0.0, 0.0)
        )

        assert view.eye == EyeIndex.LEFT
        assert view.position_offset[0] == pytest.approx(-0.032)

    def test_right_eye_view(self):
        """Test right eye view configuration."""
        view = EyeView(
            eye=EyeIndex.RIGHT,
            position_offset=(0.032, 0.0, 0.0)
        )

        assert view.eye == EyeIndex.RIGHT
        assert view.position_offset[0] == pytest.approx(0.032)


class TestMultiViewStereoRenderer:
    """Tests for MultiViewStereoRenderer."""

    def test_creation_default_config(self):
        """Test renderer creation with default config."""
        renderer = MultiViewStereoRenderer()

        assert renderer.config.method == StereoMethod.MULTI_VIEW

    def test_creation_custom_config(self):
        """Test renderer creation with custom config."""
        config = StereoConfig(ipd_meters=0.065)
        renderer = MultiViewStereoRenderer(config)

        assert renderer.config.ipd_meters == pytest.approx(0.065)

    def test_configure_updates_eye_views(self):
        """Test that configure updates eye view offsets."""
        renderer = MultiViewStereoRenderer()

        new_config = StereoConfig(ipd_meters=0.070)
        renderer.configure(new_config)

        left_view = renderer.get_eye_view(EyeIndex.LEFT)
        right_view = renderer.get_eye_view(EyeIndex.RIGHT)

        # IPD/2 = 0.035m offset per eye
        assert left_view.position_offset[0] == pytest.approx(-0.035)
        assert right_view.position_offset[0] == pytest.approx(0.035)

    def test_get_eye_view_left(self):
        """Test getting left eye view."""
        renderer = MultiViewStereoRenderer()
        view = renderer.get_eye_view(EyeIndex.LEFT)

        assert view.eye == EyeIndex.LEFT
        assert view.position_offset[0] < 0  # Left eye has negative X offset

    def test_get_eye_view_right(self):
        """Test getting right eye view."""
        renderer = MultiViewStereoRenderer()
        view = renderer.get_eye_view(EyeIndex.RIGHT)

        assert view.eye == EyeIndex.RIGHT
        assert view.position_offset[0] > 0  # Right eye has positive X offset

    def test_get_view_matrix_identity_pose(self):
        """Test view matrix with identity head pose."""
        renderer = MultiViewStereoRenderer()

        head_pos = (0.0, 0.0, 0.0)
        head_orient = (0.0, 0.0, 0.0, 1.0)  # Identity quaternion

        view_matrix = renderer.get_view_matrix(EyeIndex.LEFT, head_pos, head_orient)

        assert len(view_matrix) == 16

    def test_get_view_matrix_different_for_eyes(self):
        """Test that view matrices differ between eyes."""
        renderer = MultiViewStereoRenderer()

        head_pos = (0.0, 1.6, 0.0)
        head_orient = (0.0, 0.0, 0.0, 1.0)

        left_matrix = renderer.get_view_matrix(EyeIndex.LEFT, head_pos, head_orient)
        right_matrix = renderer.get_view_matrix(EyeIndex.RIGHT, head_pos, head_orient)

        # Matrices should differ due to IPD offset
        assert left_matrix != right_matrix

    def test_get_projection_matrix(self):
        """Test projection matrix generation."""
        renderer = MultiViewStereoRenderer()

        proj_matrix = renderer.get_projection_matrix(EyeIndex.LEFT)

        assert len(proj_matrix) == 16
        # Check it's a valid projection matrix (negative z scaling)
        assert proj_matrix[10] < 0  # Column 3, row 3 in column-major

    def test_frame_lifecycle(self):
        """Test frame begin/end lifecycle."""
        renderer = MultiViewStereoRenderer()

        # Should not raise
        renderer.begin_frame()
        renderer.begin_eye(EyeIndex.LEFT)
        renderer.end_eye(EyeIndex.LEFT)
        renderer.begin_eye(EyeIndex.RIGHT)
        renderer.end_eye(EyeIndex.RIGHT)
        renderer.end_frame()

    def test_world_scale_affects_ipd(self):
        """Test that world scale affects effective IPD."""
        config = StereoConfig(ipd_meters=0.063, world_scale=2.0)
        renderer = MultiViewStereoRenderer(config)

        left_view = renderer.get_eye_view(EyeIndex.LEFT)

        # With 2x world scale, IPD offset should be doubled
        expected_offset = -0.063  # IPD * world_scale / 2
        assert left_view.position_offset[0] == pytest.approx(expected_offset, rel=1e-3)


class TestInstancedStereoRenderer:
    """Tests for InstancedStereoRenderer."""

    def test_creation(self):
        """Test instanced renderer creation."""
        renderer = InstancedStereoRenderer()

        assert renderer.config.method == StereoMethod.INSTANCED

    def test_configure_forces_method(self):
        """Test that configure enforces instanced method."""
        renderer = InstancedStereoRenderer()

        config = StereoConfig(method=StereoMethod.MULTI_VIEW)
        renderer.configure(config)

        # Should force method to INSTANCED
        assert renderer.config.method == StereoMethod.INSTANCED

    def test_view_matrix_matches_multiview(self):
        """Test that view matrices match MultiView implementation."""
        config = StereoConfig(ipd_meters=0.063)
        instanced = InstancedStereoRenderer(config)
        multiview = MultiViewStereoRenderer(config)

        head_pos = (0.0, 1.6, 0.0)
        head_orient = (0.0, 0.0, 0.0, 1.0)

        instanced_matrix = instanced.get_view_matrix(EyeIndex.LEFT, head_pos, head_orient)
        multiview_matrix = multiview.get_view_matrix(EyeIndex.LEFT, head_pos, head_orient)

        for i in range(16):
            assert instanced_matrix[i] == pytest.approx(multiview_matrix[i], rel=1e-5)


class TestSequentialStereoRenderer:
    """Tests for SequentialStereoRenderer."""

    def test_creation(self):
        """Test sequential renderer creation."""
        renderer = SequentialStereoRenderer()

        assert renderer.config.method == StereoMethod.SEQUENTIAL

    def test_configure_forces_method(self):
        """Test that configure enforces sequential method."""
        renderer = SequentialStereoRenderer()

        config = StereoConfig(method=StereoMethod.MULTI_VIEW)
        renderer.configure(config)

        assert renderer.config.method == StereoMethod.SEQUENTIAL


class TestStereoRendererFactory:
    """Tests for create_stereo_renderer factory function."""

    def test_create_default(self):
        """Test default renderer creation."""
        renderer = create_stereo_renderer()

        assert isinstance(renderer, MultiViewStereoRenderer)

    def test_create_multiview(self):
        """Test MultiView renderer creation."""
        config = StereoConfig(method=StereoMethod.MULTI_VIEW)
        renderer = create_stereo_renderer(config)

        assert isinstance(renderer, MultiViewStereoRenderer)

    def test_create_instanced(self):
        """Test Instanced renderer creation."""
        config = StereoConfig(method=StereoMethod.INSTANCED)
        renderer = create_stereo_renderer(config)

        assert isinstance(renderer, InstancedStereoRenderer)

    def test_create_sequential(self):
        """Test Sequential renderer creation."""
        config = StereoConfig(method=StereoMethod.SEQUENTIAL)
        renderer = create_stereo_renderer(config)

        assert isinstance(renderer, SequentialStereoRenderer)


class TestStereoMathOperations:
    """Tests for stereo rendering math operations."""

    def test_quaternion_rotation_identity(self):
        """Test quaternion rotation with identity."""
        renderer = MultiViewStereoRenderer()

        quat = (0.0, 0.0, 0.0, 1.0)  # Identity
        vec = (1.0, 0.0, 0.0)

        result = renderer._apply_rotation_to_vector(quat, vec)

        assert result[0] == pytest.approx(1.0, rel=1e-5)
        assert result[1] == pytest.approx(0.0, abs=1e-5)
        assert result[2] == pytest.approx(0.0, abs=1e-5)

    def test_quaternion_rotation_90_deg_y(self):
        """Test 90 degree Y rotation."""
        renderer = MultiViewStereoRenderer()

        # 90 degrees around Y axis
        angle = math.pi / 2
        quat = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
        vec = (1.0, 0.0, 0.0)

        result = renderer._apply_rotation_to_vector(quat, vec)

        # X axis rotated 90 deg around Y becomes -Z
        assert result[0] == pytest.approx(0.0, abs=1e-5)
        assert result[1] == pytest.approx(0.0, abs=1e-5)
        assert result[2] == pytest.approx(-1.0, rel=1e-5)

    def test_quaternion_multiplication_identity(self):
        """Test quaternion multiplication with identity."""
        identity = (0.0, 0.0, 0.0, 1.0)
        q = (0.1, 0.2, 0.3, 0.9)

        result = multiply_quaternions(q, identity)

        assert result[0] == pytest.approx(q[0], rel=1e-5)
        assert result[1] == pytest.approx(q[1], rel=1e-5)
        assert result[2] == pytest.approx(q[2], rel=1e-5)
        assert result[3] == pytest.approx(q[3], rel=1e-5)


class TestStereoProjection:
    """Tests for stereo projection calculations."""

    def test_symmetric_projection(self):
        """Test symmetric projection matrix."""
        config = StereoConfig(projection_type=ProjectionType.SYMMETRIC)
        renderer = MultiViewStereoRenderer(config)

        proj = renderer.get_projection_matrix(EyeIndex.LEFT)

        # For symmetric projection, element [2][0] should be 0
        # (no horizontal asymmetry)
        # Column-major: index 8 is column 2, row 0
        assert proj[8] == pytest.approx(0.0, abs=0.1)

    def test_projection_near_far_planes(self):
        """Test that near/far planes affect projection."""
        config1 = StereoConfig(near_plane=0.1, far_plane=100.0)
        config2 = StereoConfig(near_plane=1.0, far_plane=1000.0)

        renderer1 = MultiViewStereoRenderer(config1)
        renderer2 = MultiViewStereoRenderer(config2)

        proj1 = renderer1.get_projection_matrix(EyeIndex.LEFT)
        proj2 = renderer2.get_projection_matrix(EyeIndex.LEFT)

        # Projection matrices should differ
        assert proj1 != proj2
