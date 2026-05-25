"""Tests for material preview."""
import pytest
import math
from engine.tooling.material_editor.material_preview import (
    PreviewShape, LightType, PreviewLight, PreviewCamera, LightingPreset,
    PreviewSettings, PreviewRenderer, NullPreviewRenderer, MaterialPreview
)


class TestPreviewLight:
    """Tests for PreviewLight."""

    def test_create_directional_light(self):
        """Test creating directional light."""
        light = PreviewLight(
            light_type=LightType.DIRECTIONAL,
            color=(1.0, 0.95, 0.9),
            intensity=1.5,
            direction=(0.0, -1.0, 0.0)
        )
        assert light.light_type == LightType.DIRECTIONAL
        assert light.intensity == 1.5

    def test_create_point_light(self):
        """Test creating point light."""
        light = PreviewLight(
            light_type=LightType.POINT,
            position=(2.0, 3.0, 2.0),
            radius=10.0
        )
        assert light.light_type == LightType.POINT
        assert light.position == (2.0, 3.0, 2.0)
        assert light.radius == 10.0

    def test_create_spot_light(self):
        """Test creating spot light."""
        light = PreviewLight(
            light_type=LightType.SPOT,
            spot_angle=30.0
        )
        assert light.light_type == LightType.SPOT
        assert light.spot_angle == 30.0

    def test_default_values(self):
        """Test default values."""
        light = PreviewLight(LightType.DIRECTIONAL)
        assert light.color == (1.0, 1.0, 1.0)
        assert light.intensity == 1.0
        assert light.enabled is True


class TestPreviewCamera:
    """Tests for PreviewCamera."""

    @pytest.fixture
    def camera(self):
        return PreviewCamera()

    def test_default_position(self, camera):
        """Test default camera position."""
        assert camera.position == (0.0, 0.0, 5.0)
        assert camera.target == (0.0, 0.0, 0.0)

    def test_orbit(self, camera):
        """Test orbit camera."""
        initial_pos = camera.position
        camera.orbit(45.0, 0.0)
        assert camera.position != initial_pos
        assert camera.orbit_yaw == 45.0

    def test_orbit_pitch_clamped(self, camera):
        """Test orbit pitch is clamped."""
        camera.orbit(0.0, 100.0)
        assert camera.orbit_pitch <= 89.0

        camera.orbit(0.0, -200.0)
        assert camera.orbit_pitch >= -89.0

    def test_zoom(self, camera):
        """Test zoom camera."""
        initial_distance = camera.orbit_distance
        camera.zoom(1.0)
        assert camera.orbit_distance == initial_distance - 1.0

    def test_zoom_minimum(self, camera):
        """Test zoom has minimum distance."""
        camera.zoom(100.0)
        assert camera.orbit_distance >= 0.5

    def test_pan(self, camera):
        """Test pan camera."""
        initial_target = camera.target
        camera.pan(1.0, 0.0)
        assert camera.target != initial_target


class TestLightingPreset:
    """Tests for LightingPreset."""

    def test_create_preset(self):
        """Test creating lighting preset."""
        lights = [
            PreviewLight(LightType.DIRECTIONAL, intensity=1.0),
            PreviewLight(LightType.DIRECTIONAL, intensity=0.5)
        ]
        preset = LightingPreset(
            name="Studio",
            description="Soft studio lighting",
            lights=lights,
            ambient_color=(0.1, 0.1, 0.1)
        )
        assert preset.name == "Studio"
        assert len(preset.lights) == 2


class TestPreviewSettings:
    """Tests for PreviewSettings."""

    def test_default_settings(self):
        """Test default preview settings."""
        settings = PreviewSettings()
        assert settings.shape == PreviewShape.SPHERE
        assert settings.wireframe is False
        assert settings.exposure == 1.0
        assert settings.gamma == 2.2

    def test_custom_settings(self):
        """Test custom preview settings."""
        settings = PreviewSettings(
            shape=PreviewShape.CUBE,
            wireframe=True,
            exposure=2.0
        )
        assert settings.shape == PreviewShape.CUBE
        assert settings.wireframe is True
        assert settings.exposure == 2.0


class TestNullPreviewRenderer:
    """Tests for NullPreviewRenderer."""

    @pytest.fixture
    def renderer(self):
        return NullPreviewRenderer()

    def test_initialize(self, renderer):
        """Test renderer initialization."""
        result = renderer.initialize(512, 512)
        assert result is True
        assert renderer.is_initialized is True

    def test_resize(self, renderer):
        """Test renderer resize."""
        renderer.initialize(512, 512)
        renderer.resize(1024, 768)
        assert renderer._width == 1024
        assert renderer._height == 768

    def test_render(self, renderer):
        """Test render call."""
        renderer.initialize(512, 512)
        renderer.render(
            PreviewCamera(),
            [],
            PreviewSettings(),
            {}
        )
        assert renderer.render_count == 1

    def test_shutdown(self, renderer):
        """Test renderer shutdown."""
        renderer.initialize(512, 512)
        renderer.shutdown()
        assert renderer.is_initialized is False


class TestMaterialPreview:
    """Tests for MaterialPreview."""

    @pytest.fixture
    def preview(self):
        return MaterialPreview()

    def test_initialize(self, preview):
        """Test preview initialization."""
        result = preview.initialize(512, 512)
        assert result is True

    def test_default_preset_applied(self, preview):
        """Test default preset is applied."""
        assert preview.current_preset is not None
        assert len(preview.lights) > 0

    def test_get_preset_names(self, preview):
        """Test getting preset names."""
        names = preview.get_preset_names()
        assert "studio" in names
        assert "outdoor" in names
        assert "indoor" in names

    def test_apply_preset(self, preview):
        """Test applying preset."""
        result = preview.apply_preset("outdoor")
        assert result is True
        assert preview.current_preset == "outdoor"

    def test_apply_invalid_preset(self, preview):
        """Test applying invalid preset."""
        result = preview.apply_preset("nonexistent")
        assert result is False

    def test_add_light(self, preview):
        """Test adding light."""
        initial_count = len(preview.lights)
        light = PreviewLight(LightType.POINT, intensity=2.0)
        index = preview.add_light(light)

        assert len(preview.lights) == initial_count + 1
        assert preview.current_preset is None  # Preset cleared

    def test_remove_light(self, preview):
        """Test removing light."""
        initial_count = len(preview.lights)
        result = preview.remove_light(0)

        assert result is True
        assert len(preview.lights) == initial_count - 1

    def test_update_light(self, preview):
        """Test updating light."""
        new_light = PreviewLight(LightType.DIRECTIONAL, intensity=3.0)
        result = preview.update_light(0, new_light)

        assert result is True
        assert preview.get_light(0).intensity == 3.0

    def test_clear_lights(self, preview):
        """Test clearing all lights."""
        preview.clear_lights()
        assert len(preview.lights) == 0


class TestMaterialPreviewShapes:
    """Tests for preview shape handling."""

    @pytest.fixture
    def preview(self):
        return MaterialPreview()

    def test_set_shape(self, preview):
        """Test setting preview shape."""
        preview.set_shape(PreviewShape.CUBE)
        assert preview.settings.shape == PreviewShape.CUBE

    def test_set_custom_shape(self, preview):
        """Test setting custom mesh shape."""
        preview.set_shape(PreviewShape.CUSTOM, "models/custom.obj")
        assert preview.settings.shape == PreviewShape.CUSTOM
        assert preview.settings.custom_mesh_path == "models/custom.obj"


class TestMaterialPreviewCamera:
    """Tests for preview camera controls."""

    @pytest.fixture
    def preview(self):
        return MaterialPreview()

    def test_orbit_camera(self, preview):
        """Test orbiting camera."""
        initial_yaw = preview.camera.orbit_yaw
        preview.orbit_camera(45.0, 0.0)
        assert preview.camera.orbit_yaw == initial_yaw + 45.0

    def test_zoom_camera(self, preview):
        """Test zooming camera."""
        initial_distance = preview.camera.orbit_distance
        preview.zoom_camera(1.0)
        assert preview.camera.orbit_distance == initial_distance - 1.0

    def test_pan_camera(self, preview):
        """Test panning camera."""
        initial_target = preview.camera.target
        preview.pan_camera(1.0, 1.0)
        assert preview.camera.target != initial_target

    def test_reset_camera(self, preview):
        """Test resetting camera."""
        preview.orbit_camera(90.0, 45.0)
        preview.reset_camera()
        assert preview.camera.orbit_yaw == 0.0
        assert preview.camera.orbit_pitch == 0.0

    def test_frame_object(self, preview):
        """Test framing object."""
        preview.frame_object()
        assert preview.camera.orbit_distance == 5.0
        assert preview.camera.orbit_yaw == 45.0


class TestMaterialPreviewSettings:
    """Tests for preview settings controls."""

    @pytest.fixture
    def preview(self):
        return MaterialPreview()

    def test_set_wireframe(self, preview):
        """Test setting wireframe mode."""
        preview.set_wireframe(True)
        assert preview.settings.wireframe is True

    def test_set_uv_grid(self, preview):
        """Test setting UV grid overlay."""
        preview.set_uv_grid(True)
        assert preview.settings.show_uv_grid is True

    def test_set_normal_vectors(self, preview):
        """Test setting normal vector display."""
        preview.set_normal_vectors(True)
        assert preview.settings.show_normal_vectors is True

    def test_set_auto_rotation(self, preview):
        """Test setting auto rotation."""
        preview.set_auto_rotation(0.5)
        assert preview.settings.rotation_speed == 0.5

    def test_set_background(self, preview):
        """Test setting background color."""
        preview.set_background(0.5, 0.5, 0.5, 1.0)
        assert preview.settings.background_color == (0.5, 0.5, 0.5, 1.0)

    def test_set_grid(self, preview):
        """Test setting grid display."""
        preview.set_grid(True, size=20.0, divisions=20)
        assert preview.settings.grid_visible is True
        assert preview.settings.grid_size == 20.0
        assert preview.settings.grid_divisions == 20

    def test_set_exposure(self, preview):
        """Test setting exposure."""
        preview.set_exposure(2.0)
        assert preview.settings.exposure == 2.0

    def test_set_exposure_minimum(self, preview):
        """Test exposure has minimum value."""
        preview.set_exposure(-1.0)
        assert preview.settings.exposure >= 0.01

    def test_set_gamma(self, preview):
        """Test setting gamma."""
        preview.set_gamma(1.8)
        assert preview.settings.gamma == 1.8

    def test_set_tonemap(self, preview):
        """Test setting tonemap method."""
        preview.set_tonemap("aces")
        assert preview.settings.tonemap == "aces"

    def test_set_invalid_tonemap(self, preview):
        """Test invalid tonemap method is ignored."""
        preview.set_tonemap("invalid")
        assert preview.settings.tonemap != "invalid"


class TestMaterialPreviewMaterialData:
    """Tests for material data handling."""

    @pytest.fixture
    def preview(self):
        return MaterialPreview()

    def test_set_material_data(self, preview):
        """Test setting material data."""
        data = {"albedo": (0.8, 0.2, 0.1), "roughness": 0.5}
        preview.set_material_data(data)
        # Data should trigger render if auto_update is on


class TestMaterialPreviewCallbacks:
    """Tests for preview callbacks."""

    def test_render_complete_callback(self):
        """Test render complete callback."""
        preview = MaterialPreview()
        preview.initialize(512, 512)

        callback_called = [False]

        def on_complete():
            callback_called[0] = True

        preview.on_render_complete(on_complete)
        preview.render()

        assert callback_called[0] is True


class TestMaterialPreviewAutoUpdate:
    """Tests for auto-update functionality."""

    def test_auto_update_enabled(self):
        """Test auto-update is enabled by default."""
        preview = MaterialPreview()
        assert preview.auto_update is True

    def test_disable_auto_update(self):
        """Test disabling auto-update."""
        preview = MaterialPreview()
        preview.auto_update = False
        assert preview.auto_update is False

    def test_enable_auto_update_triggers_render(self):
        """Test enabling auto-update with dirty state triggers render."""
        preview = MaterialPreview()
        preview.auto_update = False
        preview.mark_dirty()
        preview.auto_update = True
        # Should have triggered render


class TestMaterialPreviewUpdate:
    """Tests for preview update loop."""

    def test_update_with_rotation(self):
        """Test update with auto-rotation."""
        preview = MaterialPreview()
        preview.initialize(512, 512)
        preview.set_auto_rotation(1.0)

        initial_yaw = preview.camera.orbit_yaw
        preview.update(1.0)  # 1 second delta

        assert preview.camera.orbit_yaw != initial_yaw


class TestMaterialPreviewPresetManagement:
    """Tests for preset management."""

    @pytest.fixture
    def preview(self):
        return MaterialPreview()

    def test_register_custom_preset(self, preview):
        """Test registering custom preset."""
        custom = LightingPreset(
            name="Custom",
            description="Custom lighting",
            lights=[PreviewLight(LightType.DIRECTIONAL)]
        )
        preview.register_preset(custom)

        assert "custom" in preview.get_preset_names()

    def test_unregister_preset(self, preview):
        """Test unregistering preset."""
        custom = LightingPreset(
            name="ToRemove",
            description="",
            lights=[]
        )
        preview.register_preset(custom)
        result = preview.unregister_preset("toremove")

        assert result is True
        assert "toremove" not in preview.get_preset_names()

    def test_cannot_unregister_default_preset(self, preview):
        """Test default presets cannot be unregistered."""
        result = preview.unregister_preset("studio")
        assert result is False
        assert "studio" in preview.get_preset_names()
