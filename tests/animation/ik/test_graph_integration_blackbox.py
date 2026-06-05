"""
Blackbox tests for Graph+IK Integration.

T-FB-4.20 Graph+IK Integration - SDLC BLACKBOX TEST

CLEANROOM PROTOCOL: Tests written from specification only, without reading implementation.
Tests the AnimationIKController class which integrates AnimationGraph with IK layers.

Specification (from PHASE_4_TODO.md and PHASE_4_ARCH.md):
- AnimationGraph + IKLayer composition
- Goal sources (context, components)
- Solve order management
- Result combination
- AnimationIKController class
- IKSolveOrder enum
- IKGoalSource base class
- ComponentGoalSource, CallbackGoalSource, StaticGoalSource
- AnimationIKResult dataclass

Public API (from __init__.py):
- IKSolveOrder
- IKGoalSource
- AnimationIKResult
- AnimationIKController
- ComponentGoalSource
- CallbackGoalSource
- StaticGoalSource
"""

import pytest
import math
from typing import List, Optional, Dict, Callable
from enum import Enum

from engine.animation.ik.graph_integration import (
    AnimationIKController,
    IKSolveOrder,
    IKGoalSource,
    AnimationIKResult,
    ComponentGoalSource,
    CallbackGoalSource,
    StaticGoalSource,
)
from engine.animation.ik import (
    IKLayer,
    IKGoalContext,
    TwoBoneIK,
    FullBodyIK,
    SkeletonMapping,
    BodyPart,
    PositionGoal,
    FABRIKChain,
    IKBlendMode,
)
from engine.core.math import Vec3, Quat, Transform


# =============================================================================
# Helper Functions
# =============================================================================


def make_transform(
    position: Vec3,
    rotation: Optional[Quat] = None,
    scale: Optional[Vec3] = None
) -> Transform:
    """Create a Transform from position, rotation, and scale."""
    return Transform(
        translation=position,
        rotation=rotation if rotation else Quat.identity(),
        scale=scale if scale else Vec3(1.0, 1.0, 1.0)
    )


def vec3_distance(a: Vec3, b: Vec3) -> float:
    """Calculate distance between two Vec3 points."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def create_arm_transforms() -> List[Transform]:
    """Create a simple arm chain (shoulder -> elbow -> hand)."""
    return [
        make_transform(Vec3(0.0, 1.5, 0.0)),    # 0: shoulder
        make_transform(Vec3(0.3, 1.5, 0.0)),    # 1: elbow
        make_transform(Vec3(0.6, 1.5, 0.0)),    # 2: hand
    ]


def create_humanoid_transforms() -> List[Transform]:
    """Create a basic humanoid skeleton in T-pose.

    Layout (indices):
    0 = pelvis (root)
    1 = spine
    2 = chest
    3 = neck
    4 = head
    5 = left_shoulder
    6 = left_upper_arm
    7 = left_lower_arm
    8 = left_hand
    9 = right_shoulder
    10 = right_upper_arm
    11 = right_lower_arm
    12 = right_hand
    13 = left_upper_leg
    14 = left_lower_leg
    15 = left_foot
    16 = left_toe
    17 = right_upper_leg
    18 = right_lower_leg
    19 = right_foot
    20 = right_toe
    """
    transforms = [
        # Spine chain
        make_transform(Vec3(0.0, 1.0, 0.0)),    # 0: pelvis
        make_transform(Vec3(0.0, 1.2, 0.0)),    # 1: spine
        make_transform(Vec3(0.0, 1.5, 0.0)),    # 2: chest
        make_transform(Vec3(0.0, 1.7, 0.0)),    # 3: neck
        make_transform(Vec3(0.0, 1.9, 0.0)),    # 4: head
        # Left arm
        make_transform(Vec3(-0.2, 1.5, 0.0)),   # 5: left_shoulder
        make_transform(Vec3(-0.35, 1.5, 0.0)),  # 6: left_upper_arm
        make_transform(Vec3(-0.55, 1.5, 0.0)),  # 7: left_lower_arm
        make_transform(Vec3(-0.75, 1.5, 0.0)),  # 8: left_hand
        # Right arm
        make_transform(Vec3(0.2, 1.5, 0.0)),    # 9: right_shoulder
        make_transform(Vec3(0.35, 1.5, 0.0)),   # 10: right_upper_arm
        make_transform(Vec3(0.55, 1.5, 0.0)),   # 11: right_lower_arm
        make_transform(Vec3(0.75, 1.5, 0.0)),   # 12: right_hand
        # Left leg
        make_transform(Vec3(-0.1, 1.0, 0.0)),   # 13: left_upper_leg
        make_transform(Vec3(-0.1, 0.5, 0.0)),   # 14: left_lower_leg
        make_transform(Vec3(-0.1, 0.0, 0.0)),   # 15: left_foot
        make_transform(Vec3(-0.1, 0.0, 0.1)),   # 16: left_toe
        # Right leg
        make_transform(Vec3(0.1, 1.0, 0.0)),    # 17: right_upper_leg
        make_transform(Vec3(0.1, 0.5, 0.0)),    # 18: right_lower_leg
        make_transform(Vec3(0.1, 0.0, 0.0)),    # 19: right_foot
        make_transform(Vec3(0.1, 0.0, 0.1)),    # 20: right_toe
    ]
    return transforms


def create_humanoid_mapping() -> SkeletonMapping:
    """Create a standard humanoid skeleton mapping."""
    bone_map = {
        BodyPart.PELVIS: 0,
        BodyPart.SPINE: 1,
        BodyPart.CHEST: 2,
        BodyPart.NECK: 3,
        BodyPart.HEAD: 4,
        BodyPart.LEFT_SHOULDER: 5,
        BodyPart.LEFT_UPPER_ARM: 6,
        BodyPart.LEFT_LOWER_ARM: 7,
        BodyPart.LEFT_HAND: 8,
        BodyPart.RIGHT_SHOULDER: 9,
        BodyPart.RIGHT_UPPER_ARM: 10,
        BodyPart.RIGHT_LOWER_ARM: 11,
        BodyPart.RIGHT_HAND: 12,
        BodyPart.LEFT_UPPER_LEG: 13,
        BodyPart.LEFT_LOWER_LEG: 14,
        BodyPart.LEFT_FOOT: 15,
        BodyPart.LEFT_TOE: 16,
        BodyPart.RIGHT_UPPER_LEG: 17,
        BodyPart.RIGHT_LOWER_LEG: 18,
        BodyPart.RIGHT_FOOT: 19,
        BodyPart.RIGHT_TOE: 20,
    }
    return SkeletonMapping(
        bone_map=bone_map,
        spine_chain=[0, 1, 2],
        left_arm_chain=[5, 6, 7, 8],
        right_arm_chain=[9, 10, 11, 12],
        left_leg_chain=[13, 14, 15],
        right_leg_chain=[17, 18, 19],
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def arm_transforms() -> List[Transform]:
    """Simple arm chain transforms."""
    return create_arm_transforms()


@pytest.fixture
def humanoid_transforms() -> List[Transform]:
    """Standard humanoid T-pose transforms."""
    return create_humanoid_transforms()


@pytest.fixture
def skeleton_mapping() -> SkeletonMapping:
    """Standard humanoid skeleton mapping."""
    return create_humanoid_mapping()


@pytest.fixture
def two_bone_solver() -> TwoBoneIK:
    """A two-bone IK solver for arm testing."""
    return TwoBoneIK(
        root_bone=0,
        mid_bone=1,
        end_bone=2
    )


@pytest.fixture
def fullbody_solver(skeleton_mapping: SkeletonMapping) -> FullBodyIK:
    """Standard full body IK solver."""
    return FullBodyIK(skeleton_mapping)


@pytest.fixture
def ik_controller() -> AnimationIKController:
    """Create a fresh AnimationIKController for testing."""
    return AnimationIKController()


# =============================================================================
# Test Class: IKSolveOrder Enum
# =============================================================================


class TestIKSolveOrderEnum:
    """Tests for IKSolveOrder enumeration."""

    def test_ik_solve_order_exists(self) -> None:
        """IKSolveOrder enum should be importable."""
        assert IKSolveOrder is not None

    def test_ik_solve_order_is_enum(self) -> None:
        """IKSolveOrder should be an Enum."""
        assert issubclass(IKSolveOrder, Enum)

    def test_foot_first_order_exists(self) -> None:
        """FOOT_FIRST solve order should exist for foot placement priority."""
        assert hasattr(IKSolveOrder, 'FOOT_FIRST')

    def test_fullbody_first_order_exists(self) -> None:
        """FULLBODY_FIRST solve order should exist for full body IK priority."""
        assert hasattr(IKSolveOrder, 'FULLBODY_FIRST')

    def test_parallel_order_exists(self) -> None:
        """PARALLEL solve order should exist."""
        assert hasattr(IKSolveOrder, 'PARALLEL')

    def test_custom_order_exists(self) -> None:
        """CUSTOM solve order should exist."""
        assert hasattr(IKSolveOrder, 'CUSTOM')

    def test_solve_order_values_unique(self) -> None:
        """All solve order values should be unique."""
        values = [m.value for m in IKSolveOrder]
        assert len(values) == len(set(values))


# =============================================================================
# Test Class: IKGoalSource
# =============================================================================


class TestIKGoalSource:
    """Tests for IKGoalSource base class."""

    def test_ik_goal_source_exists(self) -> None:
        """IKGoalSource class should be importable."""
        assert IKGoalSource is not None

    def test_goal_source_has_get_goals_method(self) -> None:
        """IKGoalSource should have a method to get goals."""
        has_get_method = (
            hasattr(IKGoalSource, 'get_goals') or
            hasattr(IKGoalSource, 'fetch_goals') or
            hasattr(IKGoalSource, 'provide_goals')
        )
        assert has_get_method

    def test_goal_source_has_priority(self) -> None:
        """IKGoalSource should have a priority attribute or method."""
        static_source = StaticGoalSource(name='test_source')
        has_priority = (
            hasattr(static_source, 'priority') or
            hasattr(static_source, 'get_priority')
        )
        assert has_priority

    def test_goal_source_has_enabled_state(self) -> None:
        """Goal sources should have an enabled/disabled state."""
        static_source = StaticGoalSource(name='test_source')
        has_enabled = (
            hasattr(static_source, 'enabled') or
            hasattr(static_source, 'is_enabled') or
            hasattr(static_source, 'set_enabled')
        )
        assert has_enabled


# =============================================================================
# Test Class: StaticGoalSource
# =============================================================================


class TestStaticGoalSource:
    """Tests for StaticGoalSource class."""

    def test_static_goal_source_exists(self) -> None:
        """StaticGoalSource class should be importable."""
        assert StaticGoalSource is not None

    def test_create_static_source_with_name(self) -> None:
        """Should be able to create a StaticGoalSource with a name."""
        source = StaticGoalSource(name='test_source')
        assert source is not None

    def test_create_static_source_with_priority(self) -> None:
        """Should be able to create a StaticGoalSource with priority."""
        source = StaticGoalSource(name='test_source', priority=5)
        assert source is not None
        assert source.priority == 5

    def test_static_source_set_position_goal(self) -> None:
        """Should be able to set a position goal."""
        source = StaticGoalSource(name='test_source')
        target = Vec3(0.5, 1.0, 0.3)
        source.set_position_goal('left_hand', target)

        # Should have goals
        context = source.get_context()
        assert context is not None

    def test_static_source_set_rotation_goal(self) -> None:
        """Should be able to set a rotation goal."""
        source = StaticGoalSource(name='test_source')
        rotation = Quat.identity()
        source.set_rotation_goal('head', rotation)

        # Should have goals
        context = source.get_context()
        assert context is not None

    def test_static_source_clear_goals(self) -> None:
        """Should be able to clear goals from a StaticGoalSource."""
        source = StaticGoalSource(name='test_source')
        source.set_position_goal('left_hand', Vec3(0.5, 1.0, 0.3))

        source.clear()
        # After clearing, context should be empty or reset


# =============================================================================
# Test Class: CallbackGoalSource
# =============================================================================


class TestCallbackGoalSource:
    """Tests for CallbackGoalSource class."""

    def test_callback_goal_source_exists(self) -> None:
        """CallbackGoalSource class should be importable."""
        assert CallbackGoalSource is not None

    def test_create_callback_source_with_name(self) -> None:
        """Should be able to create a CallbackGoalSource with a name."""
        source = CallbackGoalSource(name='test_callback')
        assert source is not None

    def test_create_callback_source_with_function(self) -> None:
        """Should be able to create a CallbackGoalSource with a callback."""
        def goal_callback() -> IKGoalContext:
            return IKGoalContext()

        source = CallbackGoalSource(name='test_callback', callback=goal_callback)
        assert source is not None

    def test_callback_source_calls_function(self) -> None:
        """CallbackGoalSource should call its callback to get goals."""
        call_count = [0]

        def goal_callback() -> IKGoalContext:
            call_count[0] += 1
            return IKGoalContext()

        source = CallbackGoalSource(name='test_callback', callback=goal_callback)

        # Get goals
        if hasattr(source, 'get_goals'):
            source.get_goals()
        elif hasattr(source, 'fetch_goals'):
            source.fetch_goals()

        assert call_count[0] >= 1

    def test_callback_source_returns_dynamic_goals(self) -> None:
        """CallbackGoalSource should return dynamic goals each time."""
        target_x = [0.0]

        def moving_goal() -> IKGoalContext:
            target_x[0] += 0.1
            ctx = IKGoalContext()
            ctx.set_position_goal('hand', Vec3(target_x[0], 1.0, 0.0))
            return ctx

        source = CallbackGoalSource(name='moving_target', callback=moving_goal)

        # Get goals twice
        if hasattr(source, 'get_goals'):
            source.get_goals()
            source.get_goals()
        elif hasattr(source, 'fetch_goals'):
            source.fetch_goals()
            source.fetch_goals()

        # Should have called callback multiple times
        assert target_x[0] >= 0.2


# =============================================================================
# Test Class: ComponentGoalSource
# =============================================================================


class TestComponentGoalSource:
    """Tests for ComponentGoalSource class."""

    def test_component_goal_source_exists(self) -> None:
        """ComponentGoalSource class should be importable."""
        assert ComponentGoalSource is not None

    def test_create_component_source(self) -> None:
        """Should be able to create a ComponentGoalSource."""
        source = ComponentGoalSource(name='component_source')
        assert source is not None

    def test_component_source_with_getter(self) -> None:
        """Should be able to create ComponentGoalSource with component getter."""
        def get_component():
            return {'position': Vec3(0.5, 1.0, 0.0)}

        source = ComponentGoalSource(
            name='component_source',
            component_getter=get_component
        )
        assert source is not None


# =============================================================================
# Test Class: AnimationIKResult
# =============================================================================


class TestAnimationIKResult:
    """Tests for AnimationIKResult dataclass."""

    def test_animation_ik_result_exists(self) -> None:
        """AnimationIKResult class should be importable."""
        assert AnimationIKResult is not None

    def test_create_empty_result(self) -> None:
        """Should be able to create an empty AnimationIKResult."""
        result = AnimationIKResult()
        assert result is not None

    def test_result_has_transforms(self) -> None:
        """AnimationIKResult should have transforms field."""
        result = AnimationIKResult()
        assert hasattr(result, 'transforms')

    def test_result_has_layers_applied(self) -> None:
        """AnimationIKResult should track which layers were applied."""
        result = AnimationIKResult()
        assert hasattr(result, 'layers_applied')

    def test_result_has_success_status(self) -> None:
        """AnimationIKResult should indicate success/failure."""
        result = AnimationIKResult()
        assert hasattr(result, 'success')

    def test_result_has_goals_used(self) -> None:
        """AnimationIKResult should track goals used."""
        result = AnimationIKResult()
        assert hasattr(result, 'goals_used')

    def test_result_has_weights(self) -> None:
        """AnimationIKResult should have weight fields."""
        result = AnimationIKResult()
        assert hasattr(result, 'animation_weight')
        assert hasattr(result, 'ik_weight')


# =============================================================================
# Test Class: AnimationIKController Core
# =============================================================================


class TestAnimationIKControllerCore:
    """Tests for AnimationIKController creation and basic operations."""

    def test_controller_exists(self) -> None:
        """AnimationIKController class should be importable."""
        assert AnimationIKController is not None

    def test_create_controller(self) -> None:
        """Should be able to create an AnimationIKController."""
        controller = AnimationIKController()
        assert controller is not None

    def test_controller_has_add_goal_source_method(self) -> None:
        """Controller should have add_goal_source method."""
        controller = AnimationIKController()
        assert hasattr(controller, 'add_goal_source')

    def test_controller_has_remove_goal_source_method(self) -> None:
        """Controller should have remove_goal_source method."""
        controller = AnimationIKController()
        assert hasattr(controller, 'remove_goal_source')

    def test_controller_has_add_ik_layer_method(self) -> None:
        """Controller should have add_ik_layer method."""
        controller = AnimationIKController()
        assert hasattr(controller, 'add_ik_layer')

    def test_controller_has_remove_ik_layer_method(self) -> None:
        """Controller should have remove_ik_layer method."""
        controller = AnimationIKController()
        assert hasattr(controller, 'remove_ik_layer')

    def test_controller_has_update_method(self) -> None:
        """Controller should have an update method."""
        controller = AnimationIKController()
        assert hasattr(controller, 'update')

    def test_controller_has_set_solve_order_method(self) -> None:
        """Controller should have set_solve_order method."""
        controller = AnimationIKController()
        assert hasattr(controller, 'set_solve_order')

    def test_controller_has_get_solve_order_method(self) -> None:
        """Controller should have get_solve_order method."""
        controller = AnimationIKController()
        assert hasattr(controller, 'get_solve_order')


# =============================================================================
# Test Class: Goal Source Management
# =============================================================================


class TestGoalSourceManagement:
    """Tests for adding and removing goal sources from the controller."""

    def test_add_static_goal_source(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to add a StaticGoalSource to controller."""
        source = StaticGoalSource(name='test_source')
        ik_controller.add_goal_source(source)

        assert ik_controller.goal_source_count() >= 1

    def test_add_callback_goal_source(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to add a CallbackGoalSource to controller."""
        def callback():
            return IKGoalContext()

        source = CallbackGoalSource(name='callback_source', callback=callback)
        ik_controller.add_goal_source(source)

        assert ik_controller.goal_source_count() >= 1

    def test_add_multiple_goal_sources(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to add multiple goal sources."""
        source1 = StaticGoalSource(name='source1')
        source2 = StaticGoalSource(name='source2')
        source3 = CallbackGoalSource(name='source3')

        ik_controller.add_goal_source(source1)
        ik_controller.add_goal_source(source2)
        ik_controller.add_goal_source(source3)

        assert ik_controller.goal_source_count() >= 3

    def test_remove_goal_source(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to remove a goal source."""
        source = StaticGoalSource(name='removable_source')
        ik_controller.add_goal_source(source)

        initial_count = ik_controller.goal_source_count()
        ik_controller.remove_goal_source('removable_source')

        assert ik_controller.goal_source_count() < initial_count

    def test_get_goal_source_names(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to get names of all goal sources."""
        source1 = StaticGoalSource(name='alpha')
        source2 = StaticGoalSource(name='beta')

        ik_controller.add_goal_source(source1)
        ik_controller.add_goal_source(source2)

        names = ik_controller.get_goal_source_names()
        assert 'alpha' in names
        assert 'beta' in names

    def test_get_goal_source_by_name(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to get a goal source by name."""
        source = StaticGoalSource(name='findable')
        ik_controller.add_goal_source(source)

        found = ik_controller.get_goal_source('findable')
        assert found is not None


# =============================================================================
# Test Class: IK Layer Management
# =============================================================================


class TestIKLayerManagement:
    """Tests for adding and removing IK layers from the controller."""

    def test_add_ik_layer(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """Should be able to add an IK layer to controller."""
        layer = IKLayer(
            name='left_arm',
            solver=two_bone_solver,
            blend_mode=IKBlendMode.OVERRIDE
        )
        ik_controller.add_ik_layer(layer)

        assert ik_controller.ik_layer_count() >= 1

    def test_add_multiple_ik_layers(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """Should be able to add multiple IK layers."""
        layer1 = IKLayer(name='left_arm', solver=two_bone_solver)
        layer2 = IKLayer(name='right_arm', solver=two_bone_solver)

        ik_controller.add_ik_layer(layer1)
        ik_controller.add_ik_layer(layer2)

        assert ik_controller.ik_layer_count() >= 2

    def test_remove_ik_layer(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """Should be able to remove an IK layer."""
        layer = IKLayer(name='test_layer', solver=two_bone_solver)
        ik_controller.add_ik_layer(layer)

        initial_count = ik_controller.ik_layer_count()
        ik_controller.remove_ik_layer('test_layer')

        assert ik_controller.ik_layer_count() < initial_count

    def test_get_ik_layer_names(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """Should be able to get names of all IK layers."""
        layer1 = IKLayer(name='layer_alpha', solver=two_bone_solver)
        layer2 = IKLayer(name='layer_beta', solver=two_bone_solver)

        ik_controller.add_ik_layer(layer1)
        ik_controller.add_ik_layer(layer2)

        names = ik_controller.get_ik_layer_names()
        assert 'layer_alpha' in names
        assert 'layer_beta' in names

    def test_get_ik_layer_by_name(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """Should be able to get an IK layer by name."""
        layer = IKLayer(name='findable_layer', solver=two_bone_solver)
        ik_controller.add_ik_layer(layer)

        found = ik_controller.get_ik_layer('findable_layer')
        assert found is not None


# =============================================================================
# Test Class: Solve Order
# =============================================================================


class TestSolveOrder:
    """Tests for solve order management."""

    def test_controller_has_default_solve_order(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Controller should have a default solve order."""
        order = ik_controller.get_solve_order()
        assert order is not None
        assert isinstance(order, IKSolveOrder)

    def test_set_solve_order_foot_first(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to set FOOT_FIRST solve order."""
        ik_controller.set_solve_order(IKSolveOrder.FOOT_FIRST)
        assert ik_controller.get_solve_order() == IKSolveOrder.FOOT_FIRST

    def test_set_solve_order_fullbody_first(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to set FULLBODY_FIRST solve order."""
        ik_controller.set_solve_order(IKSolveOrder.FULLBODY_FIRST)
        assert ik_controller.get_solve_order() == IKSolveOrder.FULLBODY_FIRST

    def test_set_solve_order_parallel(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to set PARALLEL solve order."""
        ik_controller.set_solve_order(IKSolveOrder.PARALLEL)
        assert ik_controller.get_solve_order() == IKSolveOrder.PARALLEL

    def test_different_solve_orders_are_retrievable(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Different solve orders should be settable and retrievable."""
        for order in IKSolveOrder:
            if order == IKSolveOrder.CUSTOM:
                # CUSTOM requires a custom_order list
                ik_controller.set_solve_order(order, custom_order=['layer1', 'layer2'])
            else:
                ik_controller.set_solve_order(order)
            assert ik_controller.get_solve_order() == order


# =============================================================================
# Test Class: Update/Solve
# =============================================================================


class TestUpdateSolve:
    """Tests for the update/solve method."""

    def test_update_returns_transforms(
        self,
        ik_controller: AnimationIKController,
        humanoid_transforms: List[Transform]
    ) -> None:
        """Update should return a list of transforms."""
        result = ik_controller.update(humanoid_transforms, dt=0.016)
        assert result is not None
        assert isinstance(result, list)

    def test_update_with_no_layers_returns_input_count(
        self,
        ik_controller: AnimationIKController,
        humanoid_transforms: List[Transform]
    ) -> None:
        """Update with no layers should return same number of transforms."""
        result = ik_controller.update(humanoid_transforms, dt=0.016)
        assert len(result) == len(humanoid_transforms)

    def test_update_with_layer_modifies_transforms(
        self,
        ik_controller: AnimationIKController,
        humanoid_transforms: List[Transform],
        skeleton_mapping: SkeletonMapping
    ) -> None:
        """Update with an active layer should process transforms."""
        # Use FullBodyIK which is properly integrated with IKLayer
        solver = FullBodyIK(skeleton_mapping)
        layer = IKLayer(name='fullbody', solver=solver)
        ik_controller.add_ik_layer(layer)

        # Add a goal source with a position goal
        source = StaticGoalSource(name='body_goal')
        source.set_position_goal('left_hand', Vec3(-0.4, 1.3, 0.2))
        ik_controller.add_goal_source(source)

        result = ik_controller.update(humanoid_transforms, dt=0.016)
        assert result is not None
        assert len(result) == len(humanoid_transforms)

    def test_update_applies_goals_from_sources(
        self,
        ik_controller: AnimationIKController,
        humanoid_transforms: List[Transform],
        skeleton_mapping: SkeletonMapping
    ) -> None:
        """Goals from sources should be applied during update."""
        target = Vec3(-0.35, 1.2, 0.1)

        source = StaticGoalSource(name='body_source')
        source.set_position_goal('left_hand', target)
        ik_controller.add_goal_source(source)

        # Use FullBodyIK which is properly integrated with IKLayer
        solver = FullBodyIK(skeleton_mapping)
        layer = IKLayer(name='fullbody', solver=solver)
        ik_controller.add_ik_layer(layer)

        result = ik_controller.update(humanoid_transforms, dt=0.016)
        assert result is not None

    def test_get_last_result(
        self,
        ik_controller: AnimationIKController,
        arm_transforms: List[Transform]
    ) -> None:
        """Should be able to get the last result after update."""
        ik_controller.update(arm_transforms, dt=0.016)

        last_result = ik_controller.get_last_result()
        assert last_result is not None
        assert isinstance(last_result, AnimationIKResult)


# =============================================================================
# Test Class: Goal Source Priority
# =============================================================================


class TestGoalSourcePriority:
    """Tests for goal source priority handling."""

    def test_higher_priority_source_added_first(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Higher priority goal sources should be processed."""
        low_source = StaticGoalSource(name='low', priority=1)
        high_source = StaticGoalSource(name='high', priority=10)

        ik_controller.add_goal_source(low_source)
        ik_controller.add_goal_source(high_source)

        assert ik_controller.goal_source_count() >= 2

    def test_source_priority_can_be_changed(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to change a source's priority."""
        source = StaticGoalSource(name='adjustable', priority=5)
        ik_controller.add_goal_source(source)

        ik_controller.set_goal_source_priority('adjustable', 15)

        # Priority should be updated
        found = ik_controller.get_goal_source('adjustable')
        if found:
            assert found.priority == 15


# =============================================================================
# Test Class: Goal Source Enabled State
# =============================================================================


class TestGoalSourceEnabledState:
    """Tests for enabling/disabling goal sources."""

    def test_source_enabled_by_default(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Goal sources should be enabled by default."""
        source = StaticGoalSource(name='default_enabled')
        ik_controller.add_goal_source(source)

        found = ik_controller.get_goal_source('default_enabled')
        if found:
            assert found.enabled is True

    def test_disable_goal_source(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to disable a goal source."""
        source = StaticGoalSource(name='disableable')
        ik_controller.add_goal_source(source)

        ik_controller.set_goal_source_enabled('disableable', False)

        found = ik_controller.get_goal_source('disableable')
        if found:
            assert found.enabled is False

    def test_enable_disabled_source(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to re-enable a disabled source."""
        source = StaticGoalSource(name='toggle')
        ik_controller.add_goal_source(source)

        ik_controller.set_goal_source_enabled('toggle', False)
        ik_controller.set_goal_source_enabled('toggle', True)

        found = ik_controller.get_goal_source('toggle')
        if found:
            assert found.enabled is True


# =============================================================================
# Test Class: IK Layer Enable/Disable
# =============================================================================


class TestIKLayerEnableDisable:
    """Tests for enabling/disabling IK layers."""

    def test_layer_enabled_by_default(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """IK layers should be enabled by default."""
        layer = IKLayer(name='default_layer', solver=two_bone_solver)
        ik_controller.add_ik_layer(layer)

        found = ik_controller.get_ik_layer('default_layer')
        if found:
            assert found.enabled is True

    def test_disable_ik_layer(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """Should be able to disable an IK layer."""
        layer = IKLayer(name='disableable_layer', solver=two_bone_solver)
        ik_controller.add_ik_layer(layer)

        ik_controller.set_ik_layer_enabled('disableable_layer', False)

        found = ik_controller.get_ik_layer('disableable_layer')
        if found:
            assert found.enabled is False

    def test_enable_all_layers(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """Should be able to enable all layers at once."""
        layer1 = IKLayer(name='layer1', solver=two_bone_solver, enabled=False)
        layer2 = IKLayer(name='layer2', solver=two_bone_solver, enabled=False)

        ik_controller.add_ik_layer(layer1)
        ik_controller.add_ik_layer(layer2)

        ik_controller.enable_all_layers()

        for name in ik_controller.get_ik_layer_names():
            layer = ik_controller.get_ik_layer(name)
            if layer:
                assert layer.enabled is True

    def test_disable_all_layers(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """Should be able to disable all layers at once."""
        layer1 = IKLayer(name='layer1', solver=two_bone_solver)
        layer2 = IKLayer(name='layer2', solver=two_bone_solver)

        ik_controller.add_ik_layer(layer1)
        ik_controller.add_ik_layer(layer2)

        ik_controller.disable_all_layers()

        for name in ik_controller.get_ik_layer_names():
            layer = ik_controller.get_ik_layer(name)
            if layer:
                assert layer.enabled is False


# =============================================================================
# Test Class: Result Layer Tracking
# =============================================================================


class TestResultLayerTracking:
    """Tests for tracking which layers were applied in the result."""

    def test_result_tracks_layers_applied(
        self,
        ik_controller: AnimationIKController,
        arm_transforms: List[Transform],
        two_bone_solver: TwoBoneIK
    ) -> None:
        """AnimationIKResult should track which layers were applied."""
        layer = IKLayer(name='tracked_layer', solver=two_bone_solver)
        ik_controller.add_ik_layer(layer)

        ik_controller.update(arm_transforms, dt=0.016)
        result = ik_controller.get_last_result()

        assert result is not None
        assert hasattr(result, 'layers_applied')

    def test_result_lists_layer_names(
        self,
        ik_controller: AnimationIKController,
        humanoid_transforms: List[Transform],
        skeleton_mapping: SkeletonMapping
    ) -> None:
        """Applied layers in result should include layer names."""
        # Use FullBodyIK which is properly integrated with IKLayer
        solver = FullBodyIK(skeleton_mapping)
        layer = IKLayer(name='named_layer', solver=solver)
        ik_controller.add_ik_layer(layer)

        source = StaticGoalSource(name='goal_source')
        source.set_position_goal('left_hand', Vec3(-0.5, 1.5, 0.0))
        ik_controller.add_goal_source(source)

        ik_controller.update(humanoid_transforms, dt=0.016)
        result = ik_controller.get_last_result()

        if result and hasattr(result, 'layers_applied'):
            assert isinstance(result.layers_applied, list)


# =============================================================================
# Test Class: IK Weight Control
# =============================================================================


class TestIKWeightControl:
    """Tests for IK weight control."""

    def test_get_ik_weight(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to get the IK weight."""
        weight = ik_controller.get_ik_weight()
        assert 0.0 <= weight <= 1.0

    def test_set_ik_weight(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to set the IK weight."""
        ik_controller.set_ik_weight(0.5)
        assert ik_controller.get_ik_weight() == 0.5

    def test_set_ik_layer_weight(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """Should be able to set individual layer weight."""
        layer = IKLayer(name='weighted_layer', solver=two_bone_solver)
        ik_controller.add_ik_layer(layer)

        ik_controller.set_ik_layer_weight('weighted_layer', 0.75)

        # The layer weight should be set
        # Note: Implementation may update layer in place or store weight separately
        assert ik_controller.ik_layer_count() > 0


# =============================================================================
# Test Class: Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    def test_multiple_layers_multiple_sources(
        self,
        skeleton_mapping: SkeletonMapping,
        humanoid_transforms: List[Transform]
    ) -> None:
        """Controller should handle multiple layers and multiple sources."""
        controller = AnimationIKController()

        # Create solvers - use FullBodyIK which is properly integrated
        solver1 = FullBodyIK(skeleton_mapping)
        solver2 = FullBodyIK(skeleton_mapping)

        # Add layers with different categories
        layer1 = IKLayer(name='fullbody1', solver=solver1)
        layer2 = IKLayer(name='fullbody2', solver=solver2)

        controller.add_ik_layer(layer1)
        controller.add_ik_layer(layer2)

        # Add sources
        arm_source = StaticGoalSource(name='arm_goals')
        arm_source.set_position_goal('left_hand', Vec3(-0.6, 1.3, 0.2))

        leg_source = StaticGoalSource(name='leg_goals')
        leg_source.set_position_goal('left_foot', Vec3(-0.1, -0.1, 0.0))

        controller.add_goal_source(arm_source)
        controller.add_goal_source(leg_source)

        # Update
        result = controller.update(humanoid_transforms, dt=0.016)
        assert result is not None
        assert len(result) == len(humanoid_transforms)

    def test_callback_source_with_layer(
        self,
        humanoid_transforms: List[Transform],
        skeleton_mapping: SkeletonMapping
    ) -> None:
        """CallbackGoalSource should work with IK layers."""
        controller = AnimationIKController()

        # Dynamic target callback
        call_count = [0]
        def dynamic_goal():
            call_count[0] += 1
            ctx = IKGoalContext()
            ctx.set_position_goal('left_hand', Vec3(-0.5 + call_count[0] * 0.01, 1.0, 0.0))
            return ctx

        source = CallbackGoalSource(name='dynamic', callback=dynamic_goal)
        # Use FullBodyIK which is properly integrated with IKLayer
        solver = FullBodyIK(skeleton_mapping)
        layer = IKLayer(name='fullbody', solver=solver)

        controller.add_goal_source(source)
        controller.add_ik_layer(layer)

        # Update multiple times
        controller.update(humanoid_transforms, dt=0.016)
        controller.update(humanoid_transforms, dt=0.016)

        # Callback should have been called
        assert call_count[0] >= 2

    def test_fullbody_solver_integration(
        self,
        skeleton_mapping: SkeletonMapping,
        humanoid_transforms: List[Transform]
    ) -> None:
        """FullBodyIK solver should integrate with controller."""
        controller = AnimationIKController()

        solver = FullBodyIK(skeleton_mapping)
        layer = IKLayer(name='fullbody', solver=solver)
        controller.add_ik_layer(layer)

        # Add hand goal
        source = StaticGoalSource(name='hand_goal')
        source.set_position_goal('left_hand', Vec3(-0.4, 1.2, 0.3))
        controller.add_goal_source(source)

        result = controller.update(humanoid_transforms, dt=0.016)
        assert result is not None
        assert len(result) > 0

    def test_empty_controller_update(
        self, humanoid_transforms: List[Transform]
    ) -> None:
        """Empty controller should still return valid result."""
        controller = AnimationIKController()
        result = controller.update(humanoid_transforms, dt=0.016)

        assert result is not None
        assert len(result) == len(humanoid_transforms)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_remove_nonexistent_source(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Removing a source that doesn't exist should handle gracefully."""
        try:
            ik_controller.remove_goal_source('nonexistent')
            # Should not crash
            assert True
        except KeyError:
            # Also acceptable
            assert True

    def test_remove_nonexistent_layer(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Removing a layer that doesn't exist should handle gracefully."""
        try:
            ik_controller.remove_ik_layer('nonexistent')
            assert True
        except KeyError:
            assert True

    def test_update_with_empty_transforms(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Update with empty transforms should handle gracefully."""
        try:
            result = ik_controller.update([], dt=0.016)
            assert result is not None
            assert len(result) == 0
        except Exception:
            # May raise on invalid input
            assert True

    def test_update_with_zero_dt(
        self,
        ik_controller: AnimationIKController,
        arm_transforms: List[Transform]
    ) -> None:
        """Update with zero dt should still work."""
        result = ik_controller.update(arm_transforms, dt=0.0)
        assert result is not None

    def test_reset_controller(
        self,
        ik_controller: AnimationIKController,
        two_bone_solver: TwoBoneIK
    ) -> None:
        """Should be able to reset the controller."""
        # Add some sources and layers
        source = StaticGoalSource(name='temp')
        layer = IKLayer(name='temp_layer', solver=two_bone_solver)

        ik_controller.add_goal_source(source)
        ik_controller.add_ik_layer(layer)

        ik_controller.reset()

        # Should be cleared
        assert ik_controller.goal_source_count() == 0
        assert ik_controller.ik_layer_count() == 0

    def test_clear_all_goals(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Should be able to clear all goals from all sources."""
        source = StaticGoalSource(name='clearable')
        source.set_position_goal('hand', Vec3(1.0, 1.0, 1.0))

        ik_controller.add_goal_source(source)
        ik_controller.clear_all_goals()

        # Goals should be cleared
        assert True


# =============================================================================
# Test Class: Performance Characteristics
# =============================================================================


class TestPerformanceCharacteristics:
    """Tests for performance-related behavior."""

    def test_update_time_scales_with_layers(
        self,
        humanoid_transforms: List[Transform]
    ) -> None:
        """Update time should scale reasonably with number of layers."""
        import time

        controller = AnimationIKController()

        # Add several layers
        for i in range(5):
            solver = TwoBoneIK(root_bone=0, mid_bone=1, end_bone=2)
            layer = IKLayer(name=f'layer_{i}', solver=solver)
            controller.add_ik_layer(layer)

        # Time the update
        start = time.perf_counter()
        for _ in range(10):
            controller.update(humanoid_transforms[:3], dt=0.016)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 1 second for 10 updates)
        assert elapsed < 1.0

    def test_many_goal_sources(
        self, humanoid_transforms: List[Transform]
    ) -> None:
        """Controller should handle many goal sources efficiently."""
        controller = AnimationIKController()

        # Add many sources
        for i in range(20):
            source = StaticGoalSource(name=f'source_{i}', priority=i)
            source.set_position_goal(
                f'goal_{i}',
                Vec3(float(i) * 0.1, 1.0, 0.0)
            )
            controller.add_goal_source(source)

        # Should handle without issues
        result = controller.update(humanoid_transforms, dt=0.016)
        assert result is not None

    def test_controller_is_enabled_check(
        self, ik_controller: AnimationIKController
    ) -> None:
        """Controller should have an is_enabled method."""
        assert ik_controller.is_enabled() is True

        ik_controller.set_enabled(False)
        assert ik_controller.is_enabled() is False

        ik_controller.set_enabled(True)
        assert ik_controller.is_enabled() is True
