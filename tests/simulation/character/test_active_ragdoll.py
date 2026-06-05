"""
Whitebox tests for engine/simulation/character/active_ragdoll.py

Tests ActiveRagdoll, PDController, JointController, and balance control.
"""

import pytest
from engine.simulation.character.active_ragdoll import (
    ActiveRagdoll,
    ActiveRagdollState,
    BalanceConfig,
    JointController,
    PDController,
    RecoveryBehavior,
)
from engine.simulation.character.ragdoll import (
    BodyPartType,
    Ragdoll,
    RagdollBodyState,
    RagdollPhysicsInterface,
    RagdollPose,
    RagdollSetup,
    SkeletonInterface,
    create_default_humanoid_setup,
)
from engine.simulation.character.character_controller import (
    Quaternion,
    Transform,
    Vector3,
)


class TestActiveRagdollState:
    """Tests for ActiveRagdollState enum."""

    def test_inactive_value(self):
        """INACTIVE should have expected value."""
        assert ActiveRagdollState.INACTIVE.value == "inactive"

    def test_balanced_value(self):
        """BALANCED should have expected value."""
        assert ActiveRagdollState.BALANCED.value == "balanced"

    def test_recovering_value(self):
        """RECOVERING should have expected value."""
        assert ActiveRagdollState.RECOVERING.value == "recovering"

    def test_stumbling_value(self):
        """STUMBLING should have expected value."""
        assert ActiveRagdollState.STUMBLING.value == "stumbling"

    def test_falling_value(self):
        """FALLING should have expected value."""
        assert ActiveRagdollState.FALLING.value == "falling"


class TestRecoveryBehavior:
    """Tests for RecoveryBehavior enum."""

    def test_none_value(self):
        """NONE should have expected value."""
        assert RecoveryBehavior.NONE.value == "none"

    def test_step_value(self):
        """STEP should have expected value."""
        assert RecoveryBehavior.STEP.value == "step"

    def test_stumble_value(self):
        """STUMBLE should have expected value."""
        assert RecoveryBehavior.STUMBLE.value == "stumble"

    def test_fall_value(self):
        """FALL should have expected value."""
        assert RecoveryBehavior.FALL.value == "fall"

    def test_brace_value(self):
        """BRACE should have expected value."""
        assert RecoveryBehavior.BRACE.value == "brace"


class TestPDController:
    """Tests for PDController class."""

    def test_default_construction(self):
        """Default PDController should have default gains."""
        pd = PDController()
        assert pd.kp > 0
        assert pd.kd > 0
        assert pd.max_torque > 0

    def test_custom_construction(self):
        """PDController should accept custom values."""
        pd = PDController(kp=500.0, kd=50.0, max_torque=800.0)
        assert pd.kp == 500.0
        assert pd.kd == 50.0
        assert pd.max_torque == 800.0

    def test_compute_torque_at_target(self):
        """compute_torque at target rotation should return near zero."""
        pd = PDController()
        pd.target_rotation = Quaternion.identity()
        torque = pd.compute_torque(
            Quaternion.identity(),
            Vector3.zero()
        )
        assert torque.magnitude() == pytest.approx(0.0, abs=0.01)

    def test_compute_torque_with_error(self):
        """compute_torque with rotation error should return non-zero."""
        pd = PDController()
        pd.target_rotation = Quaternion.identity()
        # Rotated away from target
        current = Quaternion(0.0, 0.707, 0.0, 0.707)  # 90 deg around Y
        torque = pd.compute_torque(current, Vector3.zero())
        assert torque.magnitude() > 0

    def test_compute_torque_with_velocity_damping(self):
        """compute_torque should damp high velocity."""
        pd = PDController(kd=100.0)
        pd.target_rotation = Quaternion.identity()
        # High angular velocity
        torque_low_vel = pd.compute_torque(
            Quaternion.identity(),
            Vector3.zero()
        )
        torque_high_vel = pd.compute_torque(
            Quaternion.identity(),
            Vector3(10.0, 0.0, 0.0)  # Spinning fast
        )
        # High velocity should produce negative torque (damping)
        assert torque_high_vel.x < torque_low_vel.x

    def test_compute_torque_clamped(self):
        """compute_torque should clamp to max torque."""
        pd = PDController(kp=10000.0, max_torque=100.0)  # High gain, low limit
        pd.target_rotation = Quaternion.identity()
        # Large error
        current = Quaternion(0.0, 0.707, 0.0, 0.707)
        torque = pd.compute_torque(current, Vector3.zero())
        assert torque.magnitude() <= 100.0 + 0.01

    def test_quaternion_error_identity(self):
        """_quaternion_error for same rotation should return zero."""
        pd = PDController()
        error = pd._quaternion_error(
            Quaternion.identity(),
            Quaternion.identity()
        )
        assert error.magnitude() == pytest.approx(0.0, abs=0.01)


class TestJointController:
    """Tests for JointController dataclass."""

    def test_default_construction(self):
        """Default JointController should have default values."""
        controller = JointController(part_type=BodyPartType.HEAD)
        assert controller.part_type == BodyPartType.HEAD
        assert controller.strength == 1.0
        assert controller.enabled is True

    def test_custom_construction(self):
        """JointController should accept custom values."""
        controller = JointController(
            part_type=BodyPartType.UPPER_ARM_L,
            pd_controller=PDController(kp=200.0),
            strength=0.5,
            enabled=False,
        )
        assert controller.strength == 0.5
        assert controller.enabled is False
        assert controller.pd_controller.kp == 200.0


class TestBalanceConfig:
    """Tests for BalanceConfig dataclass."""

    def test_default_construction(self):
        """Default BalanceConfig should have reasonable defaults."""
        config = BalanceConfig()
        assert config.com_threshold > 0
        assert config.ankle_gain > 0
        assert config.hip_gain > 0
        assert config.step_threshold > 0

    def test_custom_construction(self):
        """BalanceConfig should accept custom values."""
        config = BalanceConfig(
            com_threshold=0.5,
            ankle_gain=200.0,
            hip_gain=100.0,
            step_threshold=0.6,
        )
        assert config.com_threshold == 0.5
        assert config.ankle_gain == 200.0


class MockActiveRagdollPhysics(RagdollPhysicsInterface):
    """Mock physics for active ragdoll testing."""

    def __init__(self):
        self.bodies: dict[int, dict] = {}
        self.next_id = 1

    def create_body(self, position, rotation, shape_type, dimensions,
                    mass, collision_group, collision_mask):
        body_id = self.next_id
        self.next_id += 1
        self.bodies[body_id] = {
            "position": position,
            "rotation": rotation,
            "velocity": Vector3.zero(),
            "angular_velocity": Vector3.zero(),
        }
        return body_id

    def get_body_transform(self, body_id):
        body = self.bodies.get(body_id, {})
        return body.get("position", Vector3.zero()), body.get("rotation", Quaternion.identity())

    def get_body_velocity(self, body_id):
        body = self.bodies.get(body_id, {})
        return body.get("velocity", Vector3.zero()), body.get("angular_velocity", Vector3.zero())

    def set_body_kinematic(self, body_id, kinematic):
        pass


class MockActiveRagdollSkeleton(SkeletonInterface):
    """Mock skeleton for active ragdoll testing."""

    def __init__(self):
        self.bones: dict[str, Transform] = {}

    def get_bone_transform(self, bone_name):
        return self.bones.get(bone_name, Transform())


class MockRagdoll:
    """Mock ragdoll for active ragdoll testing."""

    def __init__(self):
        self._is_active = True
        self._pose = RagdollPose()
        # Set up minimal body states
        self._pose.body_states[BodyPartType.PELVIS] = RagdollBodyState(
            position=Vector3(0.0, 0.9, 0.0),
            rotation=Quaternion.identity(),
        )
        self._pose.body_states[BodyPartType.FOOT_L] = RagdollBodyState(
            position=Vector3(-0.2, 0.0, 0.0),
            rotation=Quaternion.identity(),
        )
        self._pose.body_states[BodyPartType.FOOT_R] = RagdollBodyState(
            position=Vector3(0.2, 0.0, 0.0),
            rotation=Quaternion.identity(),
        )

    @property
    def is_active(self):
        return self._is_active

    def get_pose(self):
        return self._pose

    def get_center_of_mass(self):
        return Vector3(0.0, 0.9, 0.0)


class TestActiveRagdoll:
    """Tests for ActiveRagdoll class."""

    def test_construction(self):
        """ActiveRagdoll should be constructible."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        assert active is not None
        assert active.state == ActiveRagdollState.INACTIVE

    def test_state_property(self):
        """state property should return current state."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        assert active.state == ActiveRagdollState.INACTIVE

    def test_is_balanced_false_initially(self):
        """is_balanced should return False initially."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        assert active.is_balanced is False

    def test_center_of_mass_property(self):
        """center_of_mass should return current COM."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        com = active.center_of_mass
        assert isinstance(com, Vector3)

    def test_balance_error_property(self):
        """balance_error should return current error."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        error = active.balance_error
        assert isinstance(error, Vector3)

    def test_recovery_behavior_property(self):
        """recovery_behavior should return current behavior."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        assert active.recovery_behavior == RecoveryBehavior.NONE

    def test_set_lose_balance_callback(self):
        """set_lose_balance_callback should set callback."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        callback_data = []
        active.set_lose_balance_callback(lambda: callback_data.append(True))
        # Callback would be called on balance loss

    def test_set_recover_balance_callback(self):
        """set_recover_balance_callback should set callback."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        callback_data = []
        active.set_recover_balance_callback(lambda: callback_data.append(True))

    def test_set_fall_callback(self):
        """set_fall_callback should set callback."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        callback_data = []
        active.set_fall_callback(lambda: callback_data.append(True))

    def test_set_balance_config(self):
        """set_balance_config should update config."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        config = BalanceConfig(com_threshold=0.5)
        active.set_balance_config(config)
        assert active._balance_config.com_threshold == 0.5

    def test_set_joint_strength(self):
        """set_joint_strength should update joint strength."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.set_joint_strength(BodyPartType.HEAD, 0.5)
        assert active._controllers[BodyPartType.HEAD].strength == 0.5

    def test_set_joint_strength_clamped(self):
        """set_joint_strength should clamp to 0-1."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.set_joint_strength(BodyPartType.HEAD, 2.0)
        assert active._controllers[BodyPartType.HEAD].strength == 1.0
        active.set_joint_strength(BodyPartType.HEAD, -1.0)
        assert active._controllers[BodyPartType.HEAD].strength == 0.0

    def test_set_global_strength(self):
        """set_global_strength should update all joints."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.set_global_strength(0.5)
        for controller in active._controllers.values():
            assert controller.strength == 0.5

    def test_enable_joint(self):
        """enable_joint should update joint enabled state."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.enable_joint(BodyPartType.HEAD, False)
        assert active._controllers[BodyPartType.HEAD].enabled is False
        active.enable_joint(BodyPartType.HEAD, True)
        assert active._controllers[BodyPartType.HEAD].enabled is True

    def test_set_target_pose(self):
        """set_target_pose should update controller targets."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        target = Quaternion(0.0, 0.707, 0.0, 0.707)
        active.set_target_pose({BodyPartType.HEAD: target})
        assert active._controllers[BodyPartType.HEAD].pd_controller.target_rotation.y == 0.707

    def test_activate(self):
        """activate should change state to BALANCED."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        assert active.state == ActiveRagdollState.BALANCED

    def test_activate_when_ragdoll_inactive(self):
        """activate when ragdoll inactive should do nothing."""
        ragdoll = MockRagdoll()
        ragdoll._is_active = False
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        assert active.state == ActiveRagdollState.INACTIVE

    def test_deactivate(self):
        """deactivate should change state to INACTIVE."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        active.deactivate()
        assert active.state == ActiveRagdollState.INACTIVE

    def test_set_falling(self):
        """set_falling should change state to FALLING."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        fall_callback = []
        active.set_fall_callback(lambda: fall_callback.append(True))
        active.activate()
        active.set_falling()
        assert active.state == ActiveRagdollState.FALLING
        assert active.recovery_behavior == RecoveryBehavior.FALL
        assert len(fall_callback) == 1

    def test_update_inactive(self):
        """update when inactive should do nothing."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.update(0.1)  # Should not error

    def test_update_when_ragdoll_deactivated(self):
        """update should deactivate if ragdoll is deactivated."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        ragdoll._is_active = False
        active.update(0.1)
        assert active.state == ActiveRagdollState.INACTIVE

    def test_update_balanced(self):
        """update when balanced should track balance."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        active.update(0.1)
        # Should remain balanced if COM is within threshold

    def test_compute_torque(self):
        """compute_torque should return torque for body part."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        # Add body state for head
        ragdoll._pose.body_states[BodyPartType.HEAD] = RagdollBodyState(
            body_id=1,
            position=Vector3(0.0, 1.8, 0.0),
            rotation=Quaternion.identity(),
            angular_velocity=Vector3.zero(),
        )
        torque = active.compute_torque(BodyPartType.HEAD)
        assert isinstance(torque, Vector3)

    def test_compute_torque_disabled_joint(self):
        """compute_torque for disabled joint should return zero."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.enable_joint(BodyPartType.HEAD, False)
        torque = active.compute_torque(BodyPartType.HEAD)
        assert torque.magnitude() == 0.0

    def test_compute_torque_unknown_part(self):
        """compute_torque for non-existent part should return zero."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        # Remove HEAD from controllers
        if BodyPartType.HEAD in active._controllers:
            del active._controllers[BodyPartType.HEAD]
        torque = active.compute_torque(BodyPartType.HEAD)
        assert torque.magnitude() == 0.0

    def test_ankle_strategy(self):
        """ankle_strategy should return correction torque."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        # Set some balance error
        active._balance_error = Vector3(0.1, 0.0, 0.0)
        correction = active.ankle_strategy()
        assert isinstance(correction, Vector3)
        # Should push opposite to error
        assert correction.x < 0  # Negative because -error * gain

    def test_get_debug_info(self):
        """get_debug_info should return debug dictionary."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        info = active.get_debug_info()
        assert "state" in info
        assert "recovery_behavior" in info
        assert "center_of_mass" in info
        assert "balance_error" in info
        assert "balance_error_magnitude" in info
        assert "controller_count" in info


class TestActiveRagdollRecovery:
    """Tests for recovery behavior mechanics."""

    def test_determine_recovery_behavior_small_error(self):
        """Small error should not trigger recovery behavior."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        active._balance_error = Vector3(0.1, 0.0, 0.0)  # Small error
        active._determine_recovery_behavior()
        assert active.recovery_behavior == RecoveryBehavior.NONE

    def test_determine_recovery_behavior_medium_error(self):
        """Medium error should trigger step recovery."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        # Set error just above step threshold
        active._balance_config.step_threshold = 0.4
        active._balance_error = Vector3(0.42, 0.0, 0.0)
        active._determine_recovery_behavior()
        assert active.recovery_behavior == RecoveryBehavior.STEP

    def test_determine_recovery_behavior_large_error(self):
        """Large error should trigger stumble recovery."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        active._balance_config.step_threshold = 0.4
        active._balance_error = Vector3(0.55, 0.0, 0.0)  # > step * 1.2
        active._determine_recovery_behavior()
        assert active.recovery_behavior == RecoveryBehavior.STUMBLE

    def test_determine_recovery_behavior_extreme_error(self):
        """Extreme error should trigger fall."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        active._balance_config.step_threshold = 0.4
        active._balance_error = Vector3(0.7, 0.0, 0.0)  # > step * 1.5
        active._determine_recovery_behavior()
        assert active.recovery_behavior == RecoveryBehavior.FALL

    def test_execute_step_recovery(self):
        """execute_step_recovery should adjust joint strengths."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        active._recovery_behavior = RecoveryBehavior.STEP
        active._step_direction = Vector3(1.0, 0.0, 0.0)  # Right step
        active._recovery_time = 0.0
        active._execute_step_recovery()
        # Right foot should have reduced strength
        assert active._controllers[BodyPartType.FOOT_R].strength < 1.0

    def test_execute_stumble_recovery(self):
        """execute_stumble_recovery should adjust multiple joints."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        active._recovery_behavior = RecoveryBehavior.STUMBLE
        active._recovery_time = 0.0
        active._execute_stumble_recovery()
        # Legs should have reduced strength
        assert active._controllers[BodyPartType.UPPER_LEG_L].strength < 1.0

    def test_execute_brace_recovery(self):
        """execute_brace_recovery should stiffen arms."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        active.activate()
        active.set_global_strength(0.5)  # Lower starting strength
        active._recovery_behavior = RecoveryBehavior.BRACE
        active._execute_brace_recovery()
        # Arms should have full strength
        assert active._controllers[BodyPartType.UPPER_ARM_L].strength == 1.0
        assert active._controllers[BodyPartType.UPPER_ARM_R].strength == 1.0


class TestActiveRagdollControllers:
    """Tests for joint controller initialization."""

    def test_spine_controllers_initialized(self):
        """Spine controllers should be initialized with high strength."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        assert BodyPartType.PELVIS in active._controllers
        assert BodyPartType.SPINE_LOWER in active._controllers
        assert BodyPartType.SPINE_UPPER in active._controllers
        assert BodyPartType.CHEST in active._controllers

    def test_head_controllers_initialized(self):
        """Head controllers should be initialized."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        assert BodyPartType.NECK in active._controllers
        assert BodyPartType.HEAD in active._controllers

    def test_arm_controllers_initialized(self):
        """Arm controllers should be initialized."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        assert BodyPartType.SHOULDER_L in active._controllers
        assert BodyPartType.UPPER_ARM_L in active._controllers
        assert BodyPartType.LOWER_ARM_L in active._controllers

    def test_leg_controllers_initialized(self):
        """Leg controllers should be initialized."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        assert BodyPartType.UPPER_LEG_L in active._controllers
        assert BodyPartType.LOWER_LEG_L in active._controllers
        assert BodyPartType.FOOT_L in active._controllers

    def test_leg_controllers_high_strength(self):
        """Leg controllers should have high strength for balance."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        assert active._controllers[BodyPartType.UPPER_LEG_L].strength == 1.0
        assert active._controllers[BodyPartType.FOOT_L].strength == 1.0

    def test_arm_controllers_lower_strength(self):
        """Arm controllers should have lower strength."""
        ragdoll = MockRagdoll()
        physics = MockActiveRagdollPhysics()
        active = ActiveRagdoll(ragdoll, physics)
        assert active._controllers[BodyPartType.UPPER_ARM_L].strength < 1.0
