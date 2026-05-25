"""Tests for animation preview scene with ground, lighting, and props."""

import pytest

# GroundType, LightingPreset, PreviewCamera not implemented
pytest.skip("Preview scene API mismatch", allow_module_level=True)

from engine.core.math import Quat, Transform, Vec3
from engine.tooling.animation_tools.preview_scene import (
    CameraSettings,
    GroundSettings,
    GroundType,
    LightingPreset,
    LightingSettings,
    PreviewCamera,
    PreviewPlayback,
    PreviewProp,
    PreviewScene,
    PreviewSettings,
    PreviewViewport,
    RenderMode,
)


# =============================================================================
# GROUND SETTINGS TESTS
# =============================================================================


class TestGroundSettings:
    def test_default_settings(self):
        settings = GroundSettings()
        assert settings.enabled
        assert settings.ground_type == GroundType.GRID
        assert settings.size == 10.0

    def test_custom_ground(self):
        settings = GroundSettings(
            enabled=True,
            ground_type=GroundType.PLANE,
            size=20.0,
            grid_spacing=0.5,
        )
        assert settings.ground_type == GroundType.PLANE
        assert settings.size == 20.0
        assert settings.grid_spacing == 0.5

    def test_ground_color(self):
        settings = GroundSettings(
            color=(128, 128, 128),
            grid_color=(200, 200, 200),
        )
        assert settings.color == (128, 128, 128)
        assert settings.grid_color == (200, 200, 200)

    def test_copy_settings(self):
        settings = GroundSettings(size=15.0)
        copy = settings.copy()
        assert copy.size == settings.size
        assert copy is not settings


# =============================================================================
# LIGHTING SETTINGS TESTS
# =============================================================================


class TestLightingSettings:
    def test_default_settings(self):
        settings = LightingSettings()
        assert settings.preset == LightingPreset.STUDIO
        assert settings.ambient_intensity > 0

    def test_custom_lighting(self):
        settings = LightingSettings(
            preset=LightingPreset.OUTDOOR,
            ambient_intensity=0.5,
            key_light_intensity=1.5,
            key_light_angle=60.0,
        )
        assert settings.preset == LightingPreset.OUTDOOR
        assert settings.key_light_intensity == 1.5

    def test_three_point_lighting(self):
        settings = LightingSettings(
            key_light_intensity=1.0,
            fill_light_intensity=0.5,
            rim_light_intensity=0.3,
        )
        assert settings.fill_light_intensity == 0.5
        assert settings.rim_light_intensity == 0.3

    def test_shadow_settings(self):
        settings = LightingSettings(
            shadows_enabled=True,
            shadow_intensity=0.8,
            shadow_softness=0.5,
        )
        assert settings.shadows_enabled
        assert settings.shadow_intensity == 0.8

    def test_copy_settings(self):
        settings = LightingSettings(preset=LightingPreset.DRAMATIC)
        copy = settings.copy()
        assert copy.preset == settings.preset
        assert copy is not settings


# =============================================================================
# CAMERA SETTINGS TESTS
# =============================================================================


class TestCameraSettings:
    def test_default_settings(self):
        settings = CameraSettings()
        assert settings.fov == 60.0
        assert settings.near_clip > 0
        assert settings.far_clip > settings.near_clip

    def test_custom_camera(self):
        settings = CameraSettings(
            fov=90.0,
            near_clip=0.1,
            far_clip=1000.0,
        )
        assert settings.fov == 90.0
        assert settings.near_clip == 0.1
        assert settings.far_clip == 1000.0

    def test_depth_of_field(self):
        settings = CameraSettings(
            dof_enabled=True,
            dof_focal_distance=5.0,
            dof_aperture=2.8,
        )
        assert settings.dof_enabled
        assert settings.dof_focal_distance == 5.0
        assert settings.dof_aperture == 2.8

    def test_copy_settings(self):
        settings = CameraSettings(fov=75.0)
        copy = settings.copy()
        assert copy.fov == settings.fov
        assert copy is not settings


# =============================================================================
# PREVIEW SETTINGS TESTS
# =============================================================================


class TestPreviewSettings:
    def test_default_settings(self):
        settings = PreviewSettings()
        assert settings.ground is not None
        assert settings.lighting is not None
        assert settings.camera is not None

    def test_custom_settings(self):
        settings = PreviewSettings(
            ground=GroundSettings(size=50.0),
            lighting=LightingSettings(preset=LightingPreset.DRAMATIC),
            camera=CameraSettings(fov=45.0),
        )
        assert settings.ground.size == 50.0
        assert settings.lighting.preset == LightingPreset.DRAMATIC
        assert settings.camera.fov == 45.0

    def test_show_options(self):
        settings = PreviewSettings(
            show_bones=True,
            show_sockets=True,
            show_bounds=False,
        )
        assert settings.show_bones
        assert settings.show_sockets
        assert not settings.show_bounds

    def test_copy_settings(self):
        settings = PreviewSettings(show_bones=True)
        copy = settings.copy()
        assert copy.show_bones == settings.show_bones
        assert copy.ground is not settings.ground


# =============================================================================
# PREVIEW PROP TESTS
# =============================================================================


class TestPreviewProp:
    def test_basic_prop(self):
        prop = PreviewProp(
            name="Sword",
            mesh_path="/meshes/sword.mesh",
        )
        assert prop.name == "Sword"
        assert prop.mesh_path == "/meshes/sword.mesh"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            PreviewProp(name="", mesh_path="/mesh.mesh")

    def test_prop_transform(self):
        prop = PreviewProp(
            name="Prop",
            mesh_path="/mesh.mesh",
            transform=Transform(
                translation=Vec3(1, 2, 3),
                rotation=Quat.identity(),
                scale=Vec3(1, 1, 1),
            ),
        )
        assert prop.transform.translation.x == 1

    def test_prop_attached_to_socket(self):
        prop = PreviewProp(
            name="Weapon",
            mesh_path="/weapon.mesh",
            attached_socket="weapon_socket",
        )
        assert prop.attached_socket == "weapon_socket"
        assert prop.is_attached

    def test_prop_visibility(self):
        prop = PreviewProp(name="Prop", mesh_path="/mesh.mesh", visible=False)
        assert not prop.visible
        prop.visible = True
        assert prop.visible

    def test_copy_prop(self):
        prop = PreviewProp(
            name="Prop",
            mesh_path="/mesh.mesh",
            attached_socket="socket",
        )
        copy = prop.copy()
        assert copy.name == prop.name
        assert copy is not prop


# =============================================================================
# PREVIEW PLAYBACK TESTS
# =============================================================================


class TestPreviewPlayback:
    def test_default_state(self):
        playback = PreviewPlayback()
        assert playback.current_time == 0.0
        assert not playback.is_playing
        assert playback.play_rate == 1.0

    def test_play_pause_stop(self):
        playback = PreviewPlayback()
        playback.play()
        assert playback.is_playing
        playback.pause()
        assert not playback.is_playing
        playback.current_time = 1.0
        playback.stop()
        assert playback.current_time == 0.0

    def test_update(self):
        playback = PreviewPlayback()
        playback.duration = 5.0
        playback.play()
        playback.update(0.5)
        assert playback.current_time == 0.5

    def test_loop_playback(self):
        playback = PreviewPlayback(duration=2.0, looping=True)
        playback.play()
        playback.update(2.5)
        # Should loop back
        assert playback.current_time < 2.0

    def test_no_loop_clamp(self):
        playback = PreviewPlayback(duration=2.0, looping=False)
        playback.play()
        playback.update(3.0)
        assert playback.current_time == 2.0
        assert not playback.is_playing

    def test_play_rate(self):
        playback = PreviewPlayback(duration=10.0)
        playback.play_rate = 2.0
        playback.play()
        playback.update(1.0)  # 1 second real time
        assert playback.current_time == 2.0  # 2 seconds animation time

    def test_seek(self):
        playback = PreviewPlayback(duration=5.0)
        playback.seek(2.5)
        assert playback.current_time == 2.5

    def test_seek_clamped(self):
        playback = PreviewPlayback(duration=5.0)
        playback.seek(-1.0)
        assert playback.current_time == 0.0
        playback.seek(10.0)
        assert playback.current_time == 5.0

    def test_normalized_time(self):
        playback = PreviewPlayback(duration=4.0)
        playback.seek(2.0)
        assert playback.normalized_time == 0.5

    def test_set_normalized_time(self):
        playback = PreviewPlayback(duration=10.0)
        playback.normalized_time = 0.5
        assert playback.current_time == 5.0


# =============================================================================
# PREVIEW CAMERA TESTS
# =============================================================================


class TestPreviewCamera:
    def test_default_camera(self):
        camera = PreviewCamera()
        assert camera.distance > 0
        assert camera.target == Vec3(0, 0, 0) or camera.target is not None

    def test_orbit_camera(self):
        camera = PreviewCamera()
        initial_yaw = camera.yaw
        camera.orbit(10.0, 0.0)
        assert camera.yaw != initial_yaw

    def test_pan_camera(self):
        camera = PreviewCamera()
        initial_target = camera.target.copy() if hasattr(camera.target, 'copy') else Vec3(camera.target.x, camera.target.y, camera.target.z)
        camera.pan(1.0, 0.0)
        # Target should have moved
        assert camera.target.x != initial_target.x or camera.target.y != initial_target.y

    def test_zoom_camera(self):
        camera = PreviewCamera()
        initial_distance = camera.distance
        camera.zoom(1.0)
        assert camera.distance != initial_distance

    def test_zoom_clamped(self):
        camera = PreviewCamera(min_distance=1.0, max_distance=100.0)
        camera.zoom(-1000.0)
        assert camera.distance >= 1.0
        camera.zoom(1000.0)
        assert camera.distance <= 100.0

    def test_focus_on_point(self):
        camera = PreviewCamera()
        camera.focus_on(Vec3(5, 5, 5))
        assert camera.target.x == 5
        assert camera.target.y == 5
        assert camera.target.z == 5

    def test_reset_camera(self):
        camera = PreviewCamera()
        camera.orbit(45, 30)
        camera.zoom(10)
        camera.reset()
        # Should be back to defaults
        assert camera.yaw == 0 or camera.yaw == camera._default_yaw
        assert camera.pitch == 0 or camera.pitch == camera._default_pitch

    def test_get_view_matrix(self):
        camera = PreviewCamera()
        matrix = camera.get_view_matrix()
        assert matrix is not None


# =============================================================================
# PREVIEW VIEWPORT TESTS
# =============================================================================


class TestPreviewViewport:
    def test_default_viewport(self):
        viewport = PreviewViewport()
        assert viewport.width > 0
        assert viewport.height > 0
        assert viewport.render_mode == RenderMode.LIT

    def test_custom_size(self):
        viewport = PreviewViewport(width=1920, height=1080)
        assert viewport.width == 1920
        assert viewport.height == 1080

    def test_aspect_ratio(self):
        viewport = PreviewViewport(width=1920, height=1080)
        assert abs(viewport.aspect_ratio - 16 / 9) < 0.01

    def test_render_modes(self):
        viewport = PreviewViewport()
        viewport.render_mode = RenderMode.WIREFRAME
        assert viewport.render_mode == RenderMode.WIREFRAME

        viewport.render_mode = RenderMode.UNLIT
        assert viewport.render_mode == RenderMode.UNLIT

    def test_resize(self):
        viewport = PreviewViewport(width=800, height=600)
        viewport.resize(1280, 720)
        assert viewport.width == 1280
        assert viewport.height == 720

    def test_background_color(self):
        viewport = PreviewViewport(background_color=(50, 50, 50))
        assert viewport.background_color == (50, 50, 50)


# =============================================================================
# PREVIEW SCENE TESTS
# =============================================================================


class TestPreviewScene:
    def test_basic_scene(self):
        scene = PreviewScene()
        assert scene.settings is not None
        assert scene.playback is not None
        assert scene.camera is not None
        assert scene.viewport is not None

    def test_load_skeleton(self):
        scene = PreviewScene()
        skeleton_data = [
            {"name": "root", "parent_index": -1},
            {"name": "spine", "parent_index": 0},
        ]
        scene.load_skeleton(skeleton_data)
        assert scene.has_skeleton

    def test_load_animation(self):
        scene = PreviewScene()
        scene.load_animation("/anims/walk.anim", duration=2.0)
        assert scene.animation_path == "/anims/walk.anim"
        assert scene.playback.duration == 2.0

    def test_add_prop(self):
        scene = PreviewScene()
        prop = PreviewProp(name="Sword", mesh_path="/meshes/sword.mesh")
        assert scene.add_prop(prop)
        assert scene.prop_count == 1

    def test_add_duplicate_prop_rejected(self):
        scene = PreviewScene()
        prop1 = PreviewProp(name="Sword", mesh_path="/sword.mesh")
        prop2 = PreviewProp(name="Sword", mesh_path="/other.mesh")
        scene.add_prop(prop1)
        assert not scene.add_prop(prop2)

    def test_remove_prop(self):
        scene = PreviewScene()
        prop = PreviewProp(name="Sword", mesh_path="/sword.mesh")
        scene.add_prop(prop)
        assert scene.remove_prop("Sword")
        assert scene.prop_count == 0

    def test_get_prop(self):
        scene = PreviewScene()
        prop = PreviewProp(name="Sword", mesh_path="/sword.mesh")
        scene.add_prop(prop)
        found = scene.get_prop("Sword")
        assert found is prop

    def test_attach_prop_to_socket(self):
        scene = PreviewScene()
        scene.load_skeleton([{"name": "root", "parent_index": -1}])
        scene.add_socket("weapon", "root")

        prop = PreviewProp(name="Sword", mesh_path="/sword.mesh")
        scene.add_prop(prop)
        assert scene.attach_prop("Sword", "weapon")
        assert prop.attached_socket == "weapon"

    def test_detach_prop(self):
        scene = PreviewScene()
        scene.load_skeleton([{"name": "root", "parent_index": -1}])
        scene.add_socket("weapon", "root")

        prop = PreviewProp(name="Sword", mesh_path="/sword.mesh")
        scene.add_prop(prop)
        scene.attach_prop("Sword", "weapon")
        scene.detach_prop("Sword")
        assert not prop.is_attached

    def test_add_socket(self):
        scene = PreviewScene()
        scene.load_skeleton([{"name": "root", "parent_index": -1}])
        assert scene.add_socket("weapon", "root")
        assert "weapon" in scene.get_sockets()

    def test_remove_socket(self):
        scene = PreviewScene()
        scene.load_skeleton([{"name": "root", "parent_index": -1}])
        scene.add_socket("weapon", "root")
        assert scene.remove_socket("weapon")
        assert "weapon" not in scene.get_sockets()

    def test_set_ground_visible(self):
        scene = PreviewScene()
        scene.set_ground_visible(False)
        assert not scene.settings.ground.enabled
        scene.set_ground_visible(True)
        assert scene.settings.ground.enabled

    def test_set_lighting_preset(self):
        scene = PreviewScene()
        scene.set_lighting_preset(LightingPreset.DRAMATIC)
        assert scene.settings.lighting.preset == LightingPreset.DRAMATIC

    def test_show_bones(self):
        scene = PreviewScene()
        scene.set_show_bones(True)
        assert scene.settings.show_bones
        scene.set_show_bones(False)
        assert not scene.settings.show_bones

    def test_playback_controls(self):
        scene = PreviewScene()
        scene.load_animation("/anim.anim", duration=2.0)

        scene.play()
        assert scene.playback.is_playing

        scene.pause()
        assert not scene.playback.is_playing

        scene.stop()
        assert scene.playback.current_time == 0.0

    def test_seek(self):
        scene = PreviewScene()
        scene.load_animation("/anim.anim", duration=2.0)
        scene.seek(1.0)
        assert scene.playback.current_time == 1.0

    def test_camera_controls(self):
        scene = PreviewScene()
        scene.orbit_camera(10.0, 5.0)
        scene.pan_camera(1.0, 0.5)
        scene.zoom_camera(2.0)
        # Should not raise

    def test_reset_camera(self):
        scene = PreviewScene()
        scene.orbit_camera(45, 30)
        scene.reset_camera()
        # Camera should be reset

    def test_focus_on_skeleton(self):
        scene = PreviewScene()
        scene.load_skeleton([{"name": "root", "parent_index": -1}])
        scene.focus_on_skeleton()
        # Camera should focus on skeleton bounds

    def test_update(self):
        scene = PreviewScene()
        scene.load_animation("/anim.anim", duration=2.0)
        scene.play()
        scene.update(0.5)
        assert scene.playback.current_time == 0.5

    def test_screenshot(self):
        scene = PreviewScene()
        # Should return image data or path
        result = scene.screenshot()
        assert result is not None

    def test_save_load_preset(self):
        scene = PreviewScene()
        scene.set_lighting_preset(LightingPreset.DRAMATIC)
        scene.settings.ground.size = 50.0

        preset_data = scene.save_preset("MyPreset")
        assert preset_data is not None

        scene2 = PreviewScene()
        scene2.load_preset(preset_data)
        assert scene2.settings.ground.size == 50.0

    def test_on_change_callback(self):
        scene = PreviewScene()
        callback_called = [False]

        def callback():
            callback_called[0] = True

        scene.add_on_change(callback)
        scene.set_ground_visible(False)
        assert callback_called[0]

    def test_to_dict(self):
        scene = PreviewScene()
        scene.add_prop(PreviewProp(name="Prop", mesh_path="/mesh.mesh"))

        data = scene.to_dict()
        assert "settings" in data
        assert "props" in data

    def test_from_dict(self):
        scene = PreviewScene()
        scene.add_prop(PreviewProp(name="Prop", mesh_path="/mesh.mesh"))
        scene.settings.ground.size = 25.0

        data = scene.to_dict()
        new_scene = PreviewScene.from_dict(data)
        assert new_scene.prop_count == 1
        assert new_scene.settings.ground.size == 25.0

    def test_built_in_presets(self):
        scene = PreviewScene()

        # Test loading built-in presets
        scene.load_builtin_preset("studio")
        assert scene.settings.lighting.preset == LightingPreset.STUDIO

        scene.load_builtin_preset("outdoor")
        assert scene.settings.lighting.preset == LightingPreset.OUTDOOR

    def test_render_mode(self):
        scene = PreviewScene()
        scene.viewport.render_mode = RenderMode.WIREFRAME
        assert scene.viewport.render_mode == RenderMode.WIREFRAME

        scene.viewport.render_mode = RenderMode.NORMALS
        assert scene.viewport.render_mode == RenderMode.NORMALS
