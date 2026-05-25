"""Tests for debug cameras - free-fly, orbit, and transitions."""

import pytest
import math
from engine.tooling.debug.debug_camera import (
    DebugCamera,
    FreeFlyCamera,
    OrbitCamera,
    DebugCameraController,
    CameraMode,
    CameraState,
    Vector3,
    Quaternion,
)


class TestVector3Camera:
    """Tests for Vector3 in camera context."""

    def test_dot_product(self):
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(1, 0, 0)
        assert v1.dot(v2) == 1.0

    def test_dot_product_perpendicular(self):
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        assert v1.dot(v2) == 0.0

    def test_lerp(self):
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 10, 10)
        mid = v1.lerp(v2, 0.5)
        assert mid.x == 5.0
        assert mid.y == 5.0
        assert mid.z == 5.0

    def test_negate(self):
        v = Vector3(1, 2, 3)
        neg = -v
        assert neg.x == -1
        assert neg.y == -2
        assert neg.z == -3

    def test_copy(self):
        v = Vector3(1, 2, 3)
        c = v.copy()
        assert c.x == 1
        c.x = 10
        assert v.x == 1  # Original unchanged


class TestQuaternionCamera:
    """Tests for Quaternion in camera context."""

    def test_from_euler(self):
        q = Quaternion.from_euler(0, 0, 0)
        assert abs(q.w - 1.0) < 0.001

    def test_to_euler(self):
        q = Quaternion.identity()
        pitch, yaw, roll = q.to_euler()
        assert abs(pitch) < 0.001
        assert abs(yaw) < 0.001
        assert abs(roll) < 0.001

    def test_forward(self):
        q = Quaternion.identity()
        forward = q.forward()
        # Default forward is +Z
        assert abs(forward.z - 1.0) < 0.001

    def test_right(self):
        q = Quaternion.identity()
        right = q.right()
        # Default right is +X
        assert abs(right.x - 1.0) < 0.001

    def test_up(self):
        q = Quaternion.identity()
        up = q.up()
        # Default up is +Y
        assert abs(up.y - 1.0) < 0.001

    def test_slerp(self):
        q1 = Quaternion.identity()
        q2 = Quaternion.from_euler(0, math.pi / 2, 0)
        mid = q1.slerp(q2, 0.5)
        # Should be halfway between
        assert mid.w < 1.0

    def test_copy(self):
        q = Quaternion(0.1, 0.2, 0.3, 0.9)
        c = q.copy()
        assert c.x == q.x
        c.x = 0.5
        assert q.x == 0.1


class TestCameraState:
    """Tests for CameraState class."""

    def test_camera_state_defaults(self):
        state = CameraState()
        assert state.fov == 60.0
        assert state.near_plane == 0.1
        assert state.far_plane == 1000.0

    def test_camera_state_copy(self):
        state = CameraState(
            position=Vector3(1, 2, 3),
            fov=90.0,
        )
        copy = state.copy()
        assert copy.position.x == 1
        copy.position.x = 10
        assert state.position.x == 1

    def test_camera_state_lerp(self):
        state1 = CameraState(
            position=Vector3(0, 0, 0),
            fov=60.0,
        )
        state2 = CameraState(
            position=Vector3(10, 10, 10),
            fov=90.0,
        )
        mid = state1.lerp(state2, 0.5)
        assert mid.position.x == 5.0
        assert mid.fov == 75.0


class TestFreeFlyCamera:
    """Tests for FreeFlyCamera class."""

    def test_free_fly_creation(self):
        camera = FreeFlyCamera()
        assert camera.get_mode() == CameraMode.FREE_FLY
        assert camera.enabled is True

    def test_free_fly_with_position(self):
        pos = Vector3(10, 20, 30)
        camera = FreeFlyCamera(position=pos)
        assert camera.position.x == 10
        assert camera.position.y == 20
        assert camera.position.z == 30

    def test_free_fly_enable_disable(self):
        camera = FreeFlyCamera()
        camera.disable()
        assert not camera.enabled
        camera.enable()
        assert camera.enabled

    def test_free_fly_look_at(self):
        camera = FreeFlyCamera(position=Vector3(0, 0, -10))
        camera.look_at(Vector3(0, 0, 0))
        forward = camera.get_forward()
        # Should be looking toward +Z
        assert forward.z > 0

    def test_free_fly_speed(self):
        camera = FreeFlyCamera(speed=20.0)
        assert camera.speed == 20.0
        camera.speed = 30.0
        assert camera.speed == 30.0

    def test_free_fly_speed_minimum(self):
        camera = FreeFlyCamera()
        camera.speed = -10.0
        assert camera.speed >= 0.1

    def test_free_fly_sensitivity(self):
        camera = FreeFlyCamera(sensitivity=0.01)
        assert camera.sensitivity == 0.01

    def test_free_fly_update_forward(self):
        camera = FreeFlyCamera(position=Vector3(0, 0, 0), speed=10.0)
        camera._pitch = 0
        camera._yaw = 0
        camera._update_rotation()

        input_state = {"forward": True}
        camera.update(1.0, input_state)
        # Should have moved forward
        assert camera.position.z != 0

    def test_free_fly_update_disabled(self):
        camera = FreeFlyCamera(position=Vector3(0, 0, 0))
        camera.disable()
        input_state = {"forward": True}
        camera.update(1.0, input_state)
        # Should not have moved
        assert camera.position.x == 0
        assert camera.position.y == 0
        assert camera.position.z == 0

    def test_free_fly_sprint(self):
        camera = FreeFlyCamera(
            position=Vector3(0, 0, 0),
            speed=10.0,
            sprint_multiplier=3.0,
        )
        camera._pitch = 0
        camera._yaw = 0
        camera._update_rotation()

        # Normal movement
        input_state = {"forward": True}
        camera.update(1.0, input_state)
        normal_dist = camera.position.z

        # Reset
        camera.position = Vector3(0, 0, 0)

        # Sprint movement
        input_state = {"forward": True, "sprint": True}
        camera.update(1.0, input_state)
        sprint_dist = camera.position.z

        # Sprint should be faster
        assert abs(sprint_dist) > abs(normal_dist)

    def test_free_fly_vertical_movement(self):
        camera = FreeFlyCamera(position=Vector3(0, 0, 0), speed=10.0)

        input_state = {"up": True}
        camera.update(1.0, input_state)
        assert camera.position.y > 0

        camera.position = Vector3(0, 0, 0)
        input_state = {"down": True}
        camera.update(1.0, input_state)
        assert camera.position.y < 0


class TestOrbitCamera:
    """Tests for OrbitCamera class."""

    def test_orbit_creation(self):
        camera = OrbitCamera()
        assert camera.get_mode() == CameraMode.ORBIT

    def test_orbit_with_target(self):
        target = Vector3(10, 0, 0)
        camera = OrbitCamera(target=target, distance=5.0)
        assert camera.target.x == 10
        assert camera.distance == 5.0

    def test_orbit_distance_clamping(self):
        camera = OrbitCamera(
            min_distance=2.0,
            max_distance=20.0,
            distance=10.0,
        )
        camera.distance = 50.0
        assert camera.distance == 20.0
        camera.distance = 1.0
        assert camera.distance == 2.0

    def test_orbit_focus_on(self):
        camera = OrbitCamera()
        camera.focus_on(Vector3(5, 5, 5), distance=10.0)
        assert camera.target.x == 5
        assert camera.distance == 10.0

    def test_orbit_reset(self):
        camera = OrbitCamera(distance=50.0)
        camera._pitch = 1.0
        camera._yaw = 1.0
        camera.reset()
        assert camera._pitch == 0.3  # Default pitch
        assert camera._yaw == 0.0
        assert camera.distance == 10.0

    def test_orbit_zoom(self):
        camera = OrbitCamera(distance=10.0, zoom_speed=2.0)
        input_state = {"scroll_delta": 1.0}
        camera.update(0.016, input_state)
        assert camera.distance < 10.0

    def test_orbit_update_disabled(self):
        camera = OrbitCamera(distance=10.0)
        camera.disable()
        original_distance = camera.distance
        input_state = {"scroll_delta": 5.0}
        camera.update(0.016, input_state)
        assert camera.distance == original_distance


class TestDebugCameraController:
    """Tests for DebugCameraController class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        DebugCameraController.reset_instance()
        yield
        DebugCameraController.reset_instance()

    def test_singleton(self):
        ctrl1 = DebugCameraController.get_instance()
        ctrl2 = DebugCameraController.get_instance()
        assert ctrl1 is ctrl2

    def test_register_camera(self):
        ctrl = DebugCameraController.get_instance()
        camera = FreeFlyCamera()
        ctrl.register_camera(camera)
        assert ctrl.get_camera(CameraMode.FREE_FLY) is camera

    def test_unregister_camera(self):
        ctrl = DebugCameraController.get_instance()
        camera = FreeFlyCamera()
        ctrl.register_camera(camera)
        removed = ctrl.unregister_camera(CameraMode.FREE_FLY)
        assert removed is camera
        assert ctrl.get_camera(CameraMode.FREE_FLY) is None

    def test_switch_camera(self):
        ctrl = DebugCameraController.get_instance()
        free_fly = FreeFlyCamera()
        orbit = OrbitCamera()
        ctrl.register_camera(free_fly)
        ctrl.register_camera(orbit)

        result = ctrl.switch_camera(CameraMode.FREE_FLY, instant=True)
        assert result is True
        assert ctrl.active_camera is free_fly
        assert ctrl.active_mode == CameraMode.FREE_FLY

    def test_switch_camera_not_found(self):
        ctrl = DebugCameraController.get_instance()
        result = ctrl.switch_camera(CameraMode.FREE_FLY)
        assert result is False

    def test_camera_transition(self):
        ctrl = DebugCameraController.get_instance()
        free_fly = FreeFlyCamera(position=Vector3(0, 0, 0))
        orbit = OrbitCamera(target=Vector3(10, 10, 10))
        ctrl.register_camera(free_fly)
        ctrl.register_camera(orbit)

        ctrl.switch_camera(CameraMode.FREE_FLY, instant=True)
        ctrl.switch_camera(CameraMode.ORBIT, transition_duration=1.0)

        assert ctrl.is_transitioning is True
        assert ctrl.transition_progress == 0.0

    def test_camera_transition_update(self):
        ctrl = DebugCameraController.get_instance()
        free_fly = FreeFlyCamera()
        orbit = OrbitCamera()
        ctrl.register_camera(free_fly)
        ctrl.register_camera(orbit)

        ctrl.switch_camera(CameraMode.FREE_FLY, instant=True)
        ctrl.switch_camera(CameraMode.ORBIT, transition_duration=1.0)

        # Update halfway
        ctrl.update(0.5, {})
        assert 0.4 < ctrl.transition_progress < 0.6

    def test_camera_transition_complete(self):
        ctrl = DebugCameraController.get_instance()
        free_fly = FreeFlyCamera()
        orbit = OrbitCamera()
        ctrl.register_camera(free_fly)
        ctrl.register_camera(orbit)

        ctrl.switch_camera(CameraMode.FREE_FLY, instant=True)
        ctrl.switch_camera(CameraMode.ORBIT, transition_duration=0.5)

        # Update past transition
        ctrl.update(1.0, {})
        assert ctrl.is_transitioning is False
        assert ctrl.transition_progress == 1.0

    def test_save_restore_game_camera(self):
        ctrl = DebugCameraController.get_instance()
        state = CameraState(position=Vector3(1, 2, 3), fov=75.0)
        ctrl.save_game_camera(state)

        restored = ctrl.restore_game_camera()
        assert restored is not None
        assert restored.position.x == 1
        assert restored.fov == 75.0

    def test_on_camera_change_callback(self):
        ctrl = DebugCameraController.get_instance()
        free_fly = FreeFlyCamera()
        ctrl.register_camera(free_fly)

        callback_called = [False]
        callback_mode = [None]

        def on_change(mode):
            callback_called[0] = True
            callback_mode[0] = mode

        ctrl.on_camera_change(on_change)
        ctrl.switch_camera(CameraMode.FREE_FLY, instant=True)

        assert callback_called[0] is True
        assert callback_mode[0] == CameraMode.FREE_FLY

    def test_create_free_fly_camera(self):
        ctrl = DebugCameraController.get_instance()
        camera = ctrl.create_free_fly_camera(speed=20.0)
        assert isinstance(camera, FreeFlyCamera)
        assert camera.speed == 20.0
        assert ctrl.get_camera(CameraMode.FREE_FLY) is camera

    def test_create_orbit_camera(self):
        ctrl = DebugCameraController.get_instance()
        camera = ctrl.create_orbit_camera(distance=15.0)
        assert isinstance(camera, OrbitCamera)
        assert camera.distance == 15.0
        assert ctrl.get_camera(CameraMode.ORBIT) is camera

    def test_available_modes(self):
        ctrl = DebugCameraController.get_instance()
        ctrl.create_free_fly_camera()
        ctrl.create_orbit_camera()

        modes = ctrl.available_modes
        assert CameraMode.FREE_FLY in modes
        assert CameraMode.ORBIT in modes

    def test_cycle_camera(self):
        ctrl = DebugCameraController.get_instance()
        ctrl.create_free_fly_camera()
        ctrl.create_orbit_camera()

        ctrl.switch_camera(CameraMode.FREE_FLY, instant=True)
        next_mode = ctrl.cycle_camera(transition_duration=0)
        # Should cycle to next available mode
        assert next_mode in [CameraMode.FREE_FLY, CameraMode.ORBIT]

    def test_cycle_camera_no_cameras(self):
        ctrl = DebugCameraController.get_instance()
        result = ctrl.cycle_camera()
        assert result is None

    def test_get_current_state(self):
        ctrl = DebugCameraController.get_instance()
        camera = ctrl.create_free_fly_camera(position=Vector3(5, 5, 5))
        ctrl.switch_camera(CameraMode.FREE_FLY, instant=True)

        state = ctrl.get_current_state()
        assert state.position.x == 5

    def test_update_returns_state(self):
        ctrl = DebugCameraController.get_instance()
        camera = ctrl.create_free_fly_camera()
        ctrl.switch_camera(CameraMode.FREE_FLY, instant=True)

        state = ctrl.update(0.016, {})
        assert isinstance(state, CameraState)
