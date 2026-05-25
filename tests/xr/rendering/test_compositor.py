"""Tests for XR compositor module."""

import pytest

from engine.xr.rendering.compositor import (
    LayerType,
    BlendMode,
    LayerFlags,
    LayerPose,
    Viewport,
    ProjectionLayerConfig,
    QuadLayerConfig,
    CylinderLayerConfig,
    CubemapLayerConfig,
    EquirectLayerConfig,
    Layer,
    CompositorConfig,
    CompositorMetrics,
    CompositorLayer,
    XRCompositor,
    NullCompositorLayer,
    NullXRCompositor,
    ProjectionLayer,
    QuadLayer,
    CylinderLayer,
    create_compositor,
)


class TestLayerPose:
    """Tests for LayerPose dataclass."""

    def test_default_pose(self):
        """Test default pose values."""
        pose = LayerPose()

        assert pose.position == (0.0, 0.0, -1.0)
        assert pose.orientation == (0.0, 0.0, 0.0, 1.0)
        assert pose.scale == (1.0, 1.0, 1.0)

    def test_custom_pose(self):
        """Test custom pose values."""
        pose = LayerPose(
            position=(1.0, 2.0, -3.0),
            scale=(2.0, 2.0, 2.0)
        )

        assert pose.position == (1.0, 2.0, -3.0)
        assert pose.scale == (2.0, 2.0, 2.0)


class TestViewport:
    """Tests for Viewport dataclass."""

    def test_default_viewport(self):
        """Test default viewport values."""
        viewport = Viewport()

        assert viewport.x == 0
        assert viewport.y == 0
        assert viewport.width == 1920
        assert viewport.height == 2160

    def test_custom_viewport(self):
        """Test custom viewport values."""
        viewport = Viewport(x=100, y=200, width=800, height=600)

        assert viewport.x == 100
        assert viewport.y == 200
        assert viewport.width == 800
        assert viewport.height == 600


class TestProjectionLayerConfig:
    """Tests for ProjectionLayerConfig dataclass."""

    def test_default_config(self):
        """Test default projection config."""
        config = ProjectionLayerConfig()

        assert config.near_z == pytest.approx(0.1)
        assert config.far_z == pytest.approx(1000.0)
        assert len(config.viewports) == 2
        assert config.depth_info_enabled is True


class TestQuadLayerConfig:
    """Tests for QuadLayerConfig dataclass."""

    def test_default_config(self):
        """Test default quad config."""
        config = QuadLayerConfig()

        assert config.size == (1.0, 1.0)
        assert config.blend_mode == BlendMode.ALPHA_BLEND
        assert config.eye_visibility == 3  # Both eyes

    def test_custom_config(self):
        """Test custom quad config."""
        config = QuadLayerConfig(
            size=(2.0, 1.5),
            pose=LayerPose(position=(0.0, 1.5, -2.0)),
            blend_mode=BlendMode.ADDITIVE,
            eye_visibility=1  # Left eye only
        )

        assert config.size == (2.0, 1.5)
        assert config.blend_mode == BlendMode.ADDITIVE
        assert config.eye_visibility == 1


class TestCylinderLayerConfig:
    """Tests for CylinderLayerConfig dataclass."""

    def test_default_config(self):
        """Test default cylinder config."""
        config = CylinderLayerConfig()

        assert config.radius == pytest.approx(2.0)
        assert config.central_angle == pytest.approx(1.5708, rel=0.01)  # 90 degrees
        assert config.aspect_ratio == pytest.approx(2.0)

    def test_custom_curvature(self):
        """Test custom cylinder curvature."""
        config = CylinderLayerConfig(
            radius=3.0,
            central_angle=3.14159  # 180 degrees
        )

        assert config.radius == pytest.approx(3.0)
        assert config.central_angle == pytest.approx(3.14159, rel=0.01)


class TestCompositorConfig:
    """Tests for CompositorConfig dataclass."""

    def test_default_config(self):
        """Test default compositor config."""
        config = CompositorConfig()

        assert config.max_layers == 16
        assert config.default_blend_mode == BlendMode.ALPHA_BLEND
        assert config.depth_testing_enabled is True
        assert config.chromatic_aberration_correction is True

    def test_custom_config(self):
        """Test custom compositor config."""
        config = CompositorConfig(
            max_layers=32,
            depth_testing_enabled=False
        )

        assert config.max_layers == 32
        assert config.depth_testing_enabled is False


class TestNullCompositorLayer:
    """Tests for NullCompositorLayer."""

    def test_creation(self):
        """Test layer creation."""
        layer_data = Layer(
            id=1,
            name="test",
            type=LayerType.QUAD,
            config=QuadLayerConfig()
        )
        layer = NullCompositorLayer(layer_data)

        assert layer.layer.id == 1
        assert layer.layer.name == "test"
        assert layer.layer.type == LayerType.QUAD

    def test_set_texture(self):
        """Test texture assignment."""
        layer_data = Layer(id=1, name="test", type=LayerType.QUAD, config=QuadLayerConfig())
        layer = NullCompositorLayer(layer_data)

        layer.set_texture(100, 200)

        assert layer.layer.texture_handle == 100
        assert layer.layer.depth_handle == 200

    def test_set_visible(self):
        """Test visibility toggle."""
        layer_data = Layer(id=1, name="test", type=LayerType.QUAD, config=QuadLayerConfig())
        layer = NullCompositorLayer(layer_data)

        layer.set_visible(False)
        assert layer.layer.visible is False

        layer.set_visible(True)
        assert layer.layer.visible is True

    def test_update_config(self):
        """Test config update."""
        layer_data = Layer(id=1, name="test", type=LayerType.QUAD, config=QuadLayerConfig())
        layer = NullCompositorLayer(layer_data)

        new_config = QuadLayerConfig(size=(3.0, 2.0))
        layer.update_config(new_config)

        assert layer.layer.config.size == (3.0, 2.0)


class TestNullXRCompositor:
    """Tests for NullXRCompositor."""

    def test_creation_default_config(self):
        """Test compositor creation with default config."""
        compositor = NullXRCompositor()

        assert compositor.config.max_layers == 16

    def test_creation_custom_config(self):
        """Test compositor creation with custom config."""
        config = CompositorConfig(max_layers=8)
        compositor = NullXRCompositor(config)

        assert compositor.config.max_layers == 8

    def test_configure(self):
        """Test configuration update."""
        compositor = NullXRCompositor()

        new_config = CompositorConfig(max_layers=24)
        compositor.configure(new_config)

        assert compositor.config.max_layers == 24

    def test_create_projection_layer(self):
        """Test projection layer creation."""
        compositor = NullXRCompositor()

        config = ProjectionLayerConfig()
        layer = compositor.create_projection_layer("main", config)

        assert layer.layer.type == LayerType.PROJECTION
        assert layer.layer.name == "main"

    def test_create_quad_layer(self):
        """Test quad layer creation."""
        compositor = NullXRCompositor()

        config = QuadLayerConfig(size=(2.0, 1.0))
        layer = compositor.create_quad_layer("ui", config)

        assert layer.layer.type == LayerType.QUAD
        assert layer.layer.config.size == (2.0, 1.0)

    def test_create_cylinder_layer(self):
        """Test cylinder layer creation."""
        compositor = NullXRCompositor()

        config = CylinderLayerConfig(radius=3.0)
        layer = compositor.create_cylinder_layer("curved_ui", config)

        assert layer.layer.type == LayerType.CYLINDER
        assert layer.layer.config.radius == pytest.approx(3.0)

    def test_create_cubemap_layer(self):
        """Test cubemap layer creation."""
        compositor = NullXRCompositor()

        config = CubemapLayerConfig()
        layer = compositor.create_cubemap_layer("skybox", config)

        assert layer.layer.type == LayerType.CUBEMAP

    def test_create_equirect_layer(self):
        """Test equirectangular layer creation."""
        compositor = NullXRCompositor()

        config = EquirectLayerConfig()
        layer = compositor.create_equirect_layer("360video", config)

        assert layer.layer.type == LayerType.EQUIRECT

    def test_max_layers_limit(self):
        """Test that max layers limit is enforced."""
        config = CompositorConfig(max_layers=2)
        compositor = NullXRCompositor(config)

        compositor.create_quad_layer("layer1", QuadLayerConfig())
        compositor.create_quad_layer("layer2", QuadLayerConfig())

        with pytest.raises(RuntimeError):
            compositor.create_quad_layer("layer3", QuadLayerConfig())

    def test_destroy_layer(self):
        """Test layer destruction."""
        compositor = NullXRCompositor()

        layer = compositor.create_quad_layer("temp", QuadLayerConfig())
        layer_id = layer.layer.id

        result = compositor.destroy_layer(layer_id)
        assert result is True

        # Should not be able to get destroyed layer
        assert compositor.get_layer(layer_id) is None

    def test_destroy_nonexistent_layer(self):
        """Test destroying non-existent layer."""
        compositor = NullXRCompositor()

        result = compositor.destroy_layer(9999)
        assert result is False

    def test_get_layer(self):
        """Test layer retrieval by ID."""
        compositor = NullXRCompositor()

        layer = compositor.create_quad_layer("test", QuadLayerConfig())
        layer_id = layer.layer.id

        retrieved = compositor.get_layer(layer_id)

        assert retrieved is not None
        assert retrieved.layer.id == layer_id

    def test_get_layers_sorted_by_priority(self):
        """Test that layers are returned sorted by priority."""
        compositor = NullXRCompositor()

        # Create layers with different priorities (determined by type)
        # Cubemap has priority -100, Quad has 100, Projection has 0
        compositor.create_cubemap_layer("back", CubemapLayerConfig())
        compositor.create_projection_layer("main", ProjectionLayerConfig())
        compositor.create_quad_layer("front", QuadLayerConfig())

        layers = compositor.get_layers()

        # Should be sorted: cubemap (-100), projection (0), quad (100)
        assert len(layers) == 3
        assert layers[0].layer.type == LayerType.CUBEMAP
        assert layers[1].layer.type == LayerType.PROJECTION
        assert layers[2].layer.type == LayerType.QUAD

    def test_set_layer_priority(self):
        """Test changing layer priority."""
        compositor = NullXRCompositor()

        layer = compositor.create_quad_layer("test", QuadLayerConfig())
        compositor.set_layer_priority(layer.layer.id, -50)

        assert layer.layer.priority == -50

    def test_frame_lifecycle(self):
        """Test frame begin/end lifecycle."""
        compositor = NullXRCompositor()

        layer = compositor.create_quad_layer("test", QuadLayerConfig())

        compositor.begin_frame()
        compositor.submit_layers()
        compositor.end_frame()

        # Should not raise

    def test_submit_layers_counts_visible(self):
        """Test that submit counts visible layers."""
        compositor = NullXRCompositor()

        layer1 = compositor.create_quad_layer("visible", QuadLayerConfig())
        layer2 = compositor.create_quad_layer("hidden", QuadLayerConfig())
        layer2.set_visible(False)

        compositor.begin_frame()
        compositor.submit_layers()
        compositor.end_frame()

        metrics = compositor.get_metrics()
        assert metrics.active_layers == 1

    def test_get_metrics(self):
        """Test metrics retrieval."""
        compositor = NullXRCompositor()

        compositor.create_quad_layer("test", QuadLayerConfig())

        compositor.begin_frame()
        compositor.submit_layers()
        compositor.end_frame()

        metrics = compositor.get_metrics()

        assert isinstance(metrics, CompositorMetrics)
        assert metrics.layers_composited >= 1


class TestProjectionLayer:
    """Tests for ProjectionLayer specialized class."""

    def test_set_eye_textures(self):
        """Test per-eye texture assignment."""
        layer_data = Layer(
            id=1,
            name="main",
            type=LayerType.PROJECTION,
            config=ProjectionLayerConfig()
        )
        layer = ProjectionLayer(layer_data)

        layer.set_eye_textures(
            left_texture=100,
            right_texture=200,
            left_depth=101,
            right_depth=201
        )

        assert layer.left_texture == 100
        assert layer.right_texture == 200


class TestQuadLayer:
    """Tests for QuadLayer specialized class."""

    def test_set_pose(self):
        """Test pose update."""
        layer_data = Layer(
            id=1,
            name="ui",
            type=LayerType.QUAD,
            config=QuadLayerConfig()
        )
        layer = QuadLayer(layer_data)

        new_pose = LayerPose(position=(0.0, 2.0, -3.0))
        layer.set_pose(new_pose)

        assert layer.layer.config.pose.position == (0.0, 2.0, -3.0)

    def test_set_size(self):
        """Test size update."""
        layer_data = Layer(
            id=1,
            name="ui",
            type=LayerType.QUAD,
            config=QuadLayerConfig()
        )
        layer = QuadLayer(layer_data)

        layer.set_size(4.0, 3.0)

        assert layer.layer.config.size == (4.0, 3.0)


class TestCylinderLayer:
    """Tests for CylinderLayer specialized class."""

    def test_set_pose(self):
        """Test pose update."""
        layer_data = Layer(
            id=1,
            name="curved",
            type=LayerType.CYLINDER,
            config=CylinderLayerConfig()
        )
        layer = CylinderLayer(layer_data)

        new_pose = LayerPose(position=(0.0, 1.5, -2.0))
        layer.set_pose(new_pose)

        assert layer.layer.config.pose.position == (0.0, 1.5, -2.0)

    def test_set_curvature(self):
        """Test curvature update."""
        layer_data = Layer(
            id=1,
            name="curved",
            type=LayerType.CYLINDER,
            config=CylinderLayerConfig()
        )
        layer = CylinderLayer(layer_data)

        layer.set_curvature(radius=4.0, angle=2.0)

        assert layer.layer.config.radius == pytest.approx(4.0)
        assert layer.layer.config.central_angle == pytest.approx(2.0)


class TestCompositorFactory:
    """Tests for create_compositor factory function."""

    def test_create_default(self):
        """Test default compositor creation."""
        compositor = create_compositor()

        assert isinstance(compositor, NullXRCompositor)
        assert compositor.config.max_layers == 16

    def test_create_custom_config(self):
        """Test compositor creation with custom config."""
        config = CompositorConfig(max_layers=8)
        compositor = create_compositor(config)

        assert compositor.config.max_layers == 8


class TestBlendModes:
    """Tests for blend mode handling."""

    def test_blend_mode_tracking(self):
        """Test that blend operations are tracked."""
        compositor = NullXRCompositor()

        # Create layers with different blend modes
        compositor.create_quad_layer("alpha", QuadLayerConfig(blend_mode=BlendMode.ALPHA_BLEND))
        compositor.create_quad_layer("additive", QuadLayerConfig(blend_mode=BlendMode.ADDITIVE))
        compositor.create_quad_layer("opaque", QuadLayerConfig(blend_mode=BlendMode.OPAQUE))

        compositor.begin_frame()
        compositor.submit_layers()
        compositor.end_frame()

        metrics = compositor.get_metrics()

        # Should count non-opaque blend operations
        assert metrics.blend_operations == 2  # alpha + additive


class TestLayerOrdering:
    """Tests for layer ordering and priority."""

    def test_background_to_foreground_order(self):
        """Test layers are ordered from background to foreground."""
        compositor = NullXRCompositor()

        # Create in "wrong" order
        ui = compositor.create_quad_layer("ui", QuadLayerConfig())
        scene = compositor.create_projection_layer("scene", ProjectionLayerConfig())
        skybox = compositor.create_cubemap_layer("skybox", CubemapLayerConfig())

        layers = compositor.get_layers()

        # Skybox (background) should be first
        assert layers[0].layer.name == "skybox"
        # Scene (mid-ground) should be second
        assert layers[1].layer.name == "scene"
        # UI (foreground) should be last
        assert layers[2].layer.name == "ui"
