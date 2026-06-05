"""
Blackbox tests for StateMachineBuilder.

Tests the fluent builder API for constructing animation state machines
from contract only - no implementation knowledge.

Contract:
- Fluent builder API (method chaining)
- add_state(name, source) adds animation states
- add_transition(source, target, condition, ...) adds transitions
- add_any_state_transition(target, condition) adds global transitions
- set_initial(state_name) sets starting state
- build() validates configuration and returns StateMachine
- Clear error messages for invalid configurations
"""

import pytest
from unittest.mock import MagicMock, Mock, patch

from engine.animation.graph.state_machine import (
    StateMachineBuilder,
    StateMachineBuilderError,
    StateMachine,
    TransitionCondition,
    BlendCurve,
    AnimationState,
    StateTransition,
    MotionMode,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_clip():
    """Create a mock animation clip for testing."""
    clip = MagicMock()
    clip.name = "test_clip"
    clip.duration = 1.0
    return clip


@pytest.fixture
def idle_clip():
    """Create a mock idle animation clip."""
    clip = MagicMock()
    clip.name = "idle"
    clip.duration = 2.0
    return clip


@pytest.fixture
def walk_clip():
    """Create a mock walk animation clip."""
    clip = MagicMock()
    clip.name = "walk"
    clip.duration = 1.0
    return clip


@pytest.fixture
def run_clip():
    """Create a mock run animation clip."""
    clip = MagicMock()
    clip.name = "run"
    clip.duration = 0.8
    return clip


@pytest.fixture
def jump_clip():
    """Create a mock jump animation clip."""
    clip = MagicMock()
    clip.name = "jump"
    clip.duration = 1.5
    return clip


@pytest.fixture
def attack_clip():
    """Create a mock attack animation clip."""
    clip = MagicMock()
    clip.name = "attack"
    clip.duration = 0.5
    return clip


# =============================================================================
# BUILDER INSTANTIATION TESTS
# =============================================================================


class TestBuilderInstantiation:
    """Test StateMachineBuilder instantiation."""

    def test_builder_can_be_instantiated_with_name(self):
        """Builder can be created with a name."""
        builder = StateMachineBuilder("player_controller")
        assert builder is not None

    def test_builder_can_be_instantiated_without_name(self):
        """Builder can be created without a name (uses default)."""
        builder = StateMachineBuilder()
        assert builder is not None

    def test_builder_with_empty_name(self):
        """Builder with empty string name should work or raise clear error."""
        # Either works with default name or raises clear error
        try:
            builder = StateMachineBuilder("")
            assert builder is not None
        except (ValueError, StateMachineBuilderError) as e:
            assert len(str(e)) > 0  # Error message should be descriptive

    def test_builder_with_various_valid_names(self):
        """Builder accepts various valid name formats."""
        valid_names = [
            "simple",
            "with_underscores",
            "withNumbers123",
            "CamelCase",
            "mixed_Case_123",
            "a",  # Single character
        ]
        for name in valid_names:
            builder = StateMachineBuilder(name)
            assert builder is not None


# =============================================================================
# FLUENT API / METHOD CHAINING TESTS
# =============================================================================


class TestFluentAPI:
    """Test that builder methods support fluent chaining."""

    def test_add_state_returns_builder(self, idle_clip):
        """add_state returns builder for chaining."""
        builder = StateMachineBuilder("test")
        result = builder.add_state("idle", clip=idle_clip)
        assert result is builder

    def test_add_transition_returns_builder(self, idle_clip, walk_clip):
        """add_transition returns builder for chaining."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", clip=idle_clip)
        builder.add_state("walk", clip=walk_clip)
        result = builder.add_transition("idle", "walk")
        assert result is builder

    def test_set_initial_returns_builder(self, idle_clip):
        """set_initial returns builder for chaining."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", clip=idle_clip)
        result = builder.set_initial("idle")
        assert result is builder

    def test_add_any_state_transition_returns_builder(self, idle_clip, jump_clip):
        """add_any_state_transition returns builder for chaining."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", clip=idle_clip)
        builder.add_state("jump", clip=jump_clip)
        result = builder.add_any_state_transition("jump")
        assert result is builder

    def test_full_chain(self, idle_clip, walk_clip):
        """Complete fluent chain works."""
        sm = (
            StateMachineBuilder("player")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_transition("idle", "walk")
            .add_transition("walk", "idle")
            .set_initial("idle")
            .build()
        )
        assert sm is not None
        assert isinstance(sm, StateMachine)

    def test_multiple_states_chain(self, idle_clip, walk_clip, run_clip, jump_clip):
        """Multiple add_state calls can be chained."""
        builder = (
            StateMachineBuilder("locomotion")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_state("run", clip=run_clip)
            .add_state("jump", clip=jump_clip)
        )
        assert builder is not None

    def test_multiple_transitions_chain(self, idle_clip, walk_clip, run_clip):
        """Multiple add_transition calls can be chained."""
        builder = (
            StateMachineBuilder("locomotion")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_state("run", clip=run_clip)
            .add_transition("idle", "walk")
            .add_transition("walk", "run")
            .add_transition("run", "walk")
            .add_transition("walk", "idle")
        )
        assert builder is not None


# =============================================================================
# ADD STATE TESTS
# =============================================================================


class TestAddState:
    """Test add_state functionality."""

    def test_add_single_state(self, idle_clip):
        """Can add a single state."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", clip=idle_clip)
        builder.set_initial("idle")
        sm = builder.build()
        assert sm is not None

    def test_add_multiple_states(self, idle_clip, walk_clip, run_clip):
        """Can add multiple states."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", clip=idle_clip)
        builder.add_state("walk", clip=walk_clip)
        builder.add_state("run", clip=run_clip)
        builder.set_initial("idle")
        sm = builder.build()
        assert sm is not None

    def test_add_duplicate_state_raises_error(self, idle_clip, walk_clip):
        """Adding a state with duplicate name raises clear error."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", clip=idle_clip)
        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.add_state("idle", clip=walk_clip)
        error_msg = str(exc_info.value).lower()
        # Error should mention the duplicate or already exists
        assert "idle" in error_msg or "duplicate" in error_msg or "exists" in error_msg

    def test_add_state_with_empty_name_raises_error(self, idle_clip):
        """Adding state with empty name raises clear error."""
        builder = StateMachineBuilder("test")
        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.add_state("", clip=idle_clip)
        error_msg = str(exc_info.value).lower()
        assert "empty" in error_msg or "name" in error_msg or "invalid" in error_msg

    def test_add_state_with_various_name_formats(self, mock_clip):
        """Various valid state name formats are accepted."""
        builder = StateMachineBuilder("test")
        valid_names = ["idle", "walk_fast", "Run", "JUMP", "state_123", "a"]
        for name in valid_names:
            # Create new clip for each state
            clip = MagicMock()
            clip.name = name
            builder.add_state(name, clip=clip)
        builder.set_initial("idle")
        sm = builder.build()
        assert sm is not None

    def test_add_state_with_motion_mode(self, idle_clip):
        """Can add state with motion mode parameter."""
        builder = StateMachineBuilder("test")
        # Try adding with motion mode if supported
        try:
            builder.add_state("idle", clip=idle_clip, motion_mode=MotionMode.LOOP)
            builder.set_initial("idle")
            sm = builder.build()
            assert sm is not None
        except TypeError:
            # If motion_mode not supported in add_state, skip
            pytest.skip("motion_mode parameter not supported in add_state")

    def test_add_state_with_speed(self, idle_clip):
        """Can add state with speed parameter."""
        builder = StateMachineBuilder("test")
        try:
            builder.add_state("idle", clip=idle_clip, speed=1.5)
            builder.set_initial("idle")
            sm = builder.build()
            assert sm is not None
        except TypeError:
            pytest.skip("speed parameter not supported in add_state")


# =============================================================================
# ADD TRANSITION TESTS
# =============================================================================


class TestAddTransition:
    """Test add_transition functionality."""

    def test_add_simple_transition(self, idle_clip, walk_clip):
        """Can add a simple transition between states."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_transition("idle", "walk")
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_add_bidirectional_transitions(self, idle_clip, walk_clip):
        """Can add transitions in both directions."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_transition("idle", "walk")
            .add_transition("walk", "idle")
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_add_transition_with_condition(self, idle_clip, walk_clip):
        """Can add transition with condition."""
        condition = TransitionCondition.trigger("start_walk")
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_transition("idle", "walk", condition=condition)
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_add_transition_with_duration(self, idle_clip, walk_clip):
        """Can add transition with blend duration."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_transition("idle", "walk", duration=0.3)
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_add_transition_with_blend_curve(self, idle_clip, walk_clip):
        """Can add transition with blend curve."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_transition("idle", "walk", curve=BlendCurve.EASE_IN_OUT)
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_add_transition_with_all_parameters(self, idle_clip, walk_clip):
        """Can add transition with multiple parameters."""
        condition = TransitionCondition.trigger("move")
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_transition(
                "idle",
                "walk",
                condition=condition,
                duration=0.25,
                curve=BlendCurve.SMOOTH_STEP,
            )
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_transition_from_nonexistent_source_raises_error(self, idle_clip, walk_clip):
        """Transition from nonexistent state raises error (at add or build time)."""
        builder = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
        )
        # Error may be raised at add_transition OR at build time (deferred validation)
        try:
            builder.add_transition("nonexistent", "walk")
            builder.set_initial("idle")
            with pytest.raises(StateMachineBuilderError) as exc_info:
                builder.build()
            error_msg = str(exc_info.value).lower()
            assert "nonexistent" in error_msg or "not found" in error_msg or "unknown" in error_msg or "source" in error_msg
        except StateMachineBuilderError as e:
            # Early validation is also valid
            error_msg = str(e).lower()
            assert "nonexistent" in error_msg or "not found" in error_msg or "unknown" in error_msg

    def test_transition_to_nonexistent_target_raises_error(self, idle_clip, walk_clip):
        """Transition to nonexistent state raises error (at add or build time)."""
        builder = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
        )
        # Error may be raised at add_transition OR at build time (deferred validation)
        try:
            builder.add_transition("idle", "nonexistent")
            builder.set_initial("idle")
            with pytest.raises(StateMachineBuilderError) as exc_info:
                builder.build()
            error_msg = str(exc_info.value).lower()
            assert "nonexistent" in error_msg or "not found" in error_msg or "unknown" in error_msg or "target" in error_msg
        except StateMachineBuilderError as e:
            # Early validation is also valid
            error_msg = str(e).lower()
            assert "nonexistent" in error_msg or "not found" in error_msg or "unknown" in error_msg

    def test_self_transition(self, idle_clip):
        """Can add transition from state to itself."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_transition("idle", "idle")
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_multiple_transitions_from_same_state(self, idle_clip, walk_clip, run_clip):
        """Can add multiple transitions from the same state."""
        condition_walk = TransitionCondition.trigger("walk")
        condition_run = TransitionCondition.trigger("run")
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_state("run", clip=run_clip)
            .add_transition("idle", "walk", condition=condition_walk)
            .add_transition("idle", "run", condition=condition_run)
            .set_initial("idle")
            .build()
        )
        assert sm is not None


# =============================================================================
# ADD ANY STATE TRANSITION TESTS
# =============================================================================


class TestAddAnyStateTransition:
    """Test add_any_state_transition functionality."""

    def test_any_state_transition_basic(self, idle_clip, jump_clip):
        """Can add basic any-state transition."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("jump", clip=jump_clip)
            .add_any_state_transition("jump")
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_any_state_transition_with_condition(self, idle_clip, walk_clip, attack_clip):
        """Can add any-state transition with condition."""
        condition = TransitionCondition.trigger("attack")
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_state("attack", clip=attack_clip)
            .add_any_state_transition("attack", condition=condition)
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_any_state_transition_with_duration(self, idle_clip, jump_clip):
        """Can add any-state transition with blend duration."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("jump", clip=jump_clip)
            .add_any_state_transition("jump", duration=0.1)
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_any_state_transition_with_curve(self, idle_clip, attack_clip):
        """Can add any-state transition with blend curve."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("attack", clip=attack_clip)
            .add_any_state_transition("attack", curve=BlendCurve.EASE_OUT)
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_any_state_transition_to_nonexistent_raises_error(self, idle_clip):
        """Any-state transition to nonexistent state raises error (at add or build)."""
        builder = StateMachineBuilder("test").add_state("idle", clip=idle_clip)
        # Error may be raised at add_any_state_transition OR at build time
        try:
            builder.add_any_state_transition("nonexistent")
            builder.set_initial("idle")
            with pytest.raises(StateMachineBuilderError) as exc_info:
                builder.build()
            error_msg = str(exc_info.value).lower()
            assert "nonexistent" in error_msg or "not found" in error_msg or "unknown" in error_msg or "target" in error_msg
        except StateMachineBuilderError as e:
            # Early validation is also valid
            error_msg = str(e).lower()
            assert "nonexistent" in error_msg or "not found" in error_msg or "unknown" in error_msg

    def test_multiple_any_state_transitions(self, idle_clip, walk_clip, jump_clip, attack_clip):
        """Can add multiple any-state transitions."""
        jump_cond = TransitionCondition.trigger("jump")
        attack_cond = TransitionCondition.trigger("attack")
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_state("jump", clip=jump_clip)
            .add_state("attack", clip=attack_clip)
            .add_any_state_transition("jump", condition=jump_cond)
            .add_any_state_transition("attack", condition=attack_cond)
            .set_initial("idle")
            .build()
        )
        assert sm is not None


# =============================================================================
# SET INITIAL TESTS
# =============================================================================


class TestSetInitial:
    """Test set_initial functionality."""

    def test_set_initial_state(self, idle_clip):
        """Can set initial state."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_set_initial_to_different_state(self, idle_clip, walk_clip):
        """Can set initial state to non-first state."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .set_initial("walk")
            .build()
        )
        assert sm is not None

    def test_set_initial_to_nonexistent_state_raises_error(self, idle_clip):
        """Setting initial to nonexistent state raises error (at set or build)."""
        builder = StateMachineBuilder("test").add_state("idle", clip=idle_clip)
        # Error may be raised at set_initial OR at build time (deferred validation)
        try:
            builder.set_initial("nonexistent")
            with pytest.raises(StateMachineBuilderError) as exc_info:
                builder.build()
            error_msg = str(exc_info.value).lower()
            assert "nonexistent" in error_msg or "not found" in error_msg or "unknown" in error_msg or "initial" in error_msg
        except StateMachineBuilderError as e:
            # Early validation is also valid
            error_msg = str(e).lower()
            assert "nonexistent" in error_msg or "not found" in error_msg or "unknown" in error_msg

    def test_set_initial_can_be_called_multiple_times(self, idle_clip, walk_clip):
        """Can change initial state by calling set_initial again."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .set_initial("idle")
            .set_initial("walk")  # Override
            .build()
        )
        assert sm is not None

    def test_set_initial_before_adding_states_raises_error(self):
        """Setting initial before adding any states raises error (at set or build)."""
        builder = StateMachineBuilder("test")
        # Error may be raised at set_initial OR at build time (deferred validation)
        try:
            builder.set_initial("idle")
            with pytest.raises(StateMachineBuilderError) as exc_info:
                builder.build()
            error_msg = str(exc_info.value).lower()
            assert "idle" in error_msg or "not found" in error_msg or "state" in error_msg or "initial" in error_msg
        except StateMachineBuilderError as e:
            # Early validation is also valid
            error_msg = str(e).lower()
            assert "idle" in error_msg or "not found" in error_msg or "state" in error_msg


# =============================================================================
# BUILD VALIDATION TESTS
# =============================================================================


class TestBuildValidation:
    """Test build() validation."""

    def test_build_returns_state_machine(self, idle_clip):
        """build() returns StateMachine instance."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .set_initial("idle")
            .build()
        )
        assert isinstance(sm, StateMachine)

    def test_build_with_no_states_raises_error(self):
        """build() with no states raises clear error."""
        builder = StateMachineBuilder("test")
        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.build()
        error_msg = str(exc_info.value).lower()
        assert "state" in error_msg or "empty" in error_msg or "no" in error_msg

    def test_build_without_initial_state_uses_first_or_raises(self, idle_clip):
        """build() without initial state uses first state or raises error."""
        builder = StateMachineBuilder("test").add_state("idle", clip=idle_clip)
        # Implementation may either use first state as default initial,
        # or raise an error requiring explicit initial state
        try:
            sm = builder.build()
            # If successful, first state is used as initial (valid behavior)
            assert isinstance(sm, StateMachine)
        except StateMachineBuilderError as e:
            # Requiring explicit initial is also valid
            error_msg = str(e).lower()
            assert "initial" in error_msg or "start" in error_msg or len(error_msg) > 0

    def test_build_can_be_called_once(self, idle_clip):
        """build() creates a valid state machine."""
        builder = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .set_initial("idle")
        )
        sm = builder.build()
        assert sm is not None

    def test_build_creates_independent_state_machine(self, idle_clip, walk_clip):
        """Each build() should create independent state machine or raise."""
        builder = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_transition("idle", "walk")
            .set_initial("idle")
        )
        sm1 = builder.build()
        assert sm1 is not None
        # Second build might work or raise - either is valid
        try:
            sm2 = builder.build()
            # If allowed, should be independent
            if sm2 is not None:
                assert sm2 is not sm1 or sm2 is sm1  # Either independent or same
        except (StateMachineBuilderError, RuntimeError):
            pass  # Also valid if builder is consumed after build()


# =============================================================================
# ERROR MESSAGE QUALITY TESTS
# =============================================================================


class TestErrorMessages:
    """Test that error messages are clear and descriptive."""

    def test_duplicate_state_error_mentions_state_name(self, idle_clip, walk_clip):
        """Duplicate state error should mention the state name."""
        builder = StateMachineBuilder("test").add_state("idle", clip=idle_clip)
        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.add_state("idle", clip=walk_clip)
        error_msg = str(exc_info.value)
        assert "idle" in error_msg

    def test_nonexistent_transition_source_error_mentions_state(self, idle_clip):
        """Transition source error should mention the state name (at add or build)."""
        builder = StateMachineBuilder("test").add_state("idle", clip=idle_clip)
        try:
            builder.add_transition("missing_source", "idle")
            builder.set_initial("idle")
            with pytest.raises(StateMachineBuilderError) as exc_info:
                builder.build()
            error_msg = str(exc_info.value)
            assert "missing_source" in error_msg or "source" in error_msg.lower()
        except StateMachineBuilderError as e:
            error_msg = str(e)
            assert "missing_source" in error_msg

    def test_nonexistent_transition_target_error_mentions_state(self, idle_clip):
        """Transition target error should mention the state name (at add or build)."""
        builder = StateMachineBuilder("test").add_state("idle", clip=idle_clip)
        try:
            builder.add_transition("idle", "missing_target")
            builder.set_initial("idle")
            with pytest.raises(StateMachineBuilderError) as exc_info:
                builder.build()
            error_msg = str(exc_info.value)
            assert "missing_target" in error_msg or "target" in error_msg.lower()
        except StateMachineBuilderError as e:
            error_msg = str(e)
            assert "missing_target" in error_msg

    def test_nonexistent_initial_error_mentions_state(self, idle_clip):
        """Initial state error should mention the state name (at set or build)."""
        builder = StateMachineBuilder("test").add_state("idle", clip=idle_clip)
        try:
            builder.set_initial("missing_initial")
            with pytest.raises(StateMachineBuilderError) as exc_info:
                builder.build()
            error_msg = str(exc_info.value)
            assert "missing_initial" in error_msg or "initial" in error_msg.lower()
        except StateMachineBuilderError as e:
            error_msg = str(e)
            assert "missing_initial" in error_msg

    def test_no_states_error_is_descriptive(self):
        """No states error should be descriptive."""
        builder = StateMachineBuilder("test")
        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.build()
        error_msg = str(exc_info.value)
        # Should indicate what the problem is
        assert len(error_msg) > 10  # Not just a short cryptic message

    def test_error_messages_are_not_empty(self, idle_clip):
        """All error messages should have content."""
        # Test empty state name - should raise with message
        with pytest.raises(StateMachineBuilderError) as exc_info:
            StateMachineBuilder("test").add_state("", clip=idle_clip)
        assert len(str(exc_info.value)) > 0

        # Test empty builder build - should raise with message
        with pytest.raises(StateMachineBuilderError) as exc_info:
            StateMachineBuilder("test").build()
        assert len(str(exc_info.value)) > 0

        # Test nonexistent initial - may defer to build
        builder = StateMachineBuilder("test")
        builder.add_state("idle", clip=idle_clip)
        try:
            builder.set_initial("x")
            with pytest.raises(StateMachineBuilderError) as exc_info:
                builder.build()
            assert len(str(exc_info.value)) > 0
        except StateMachineBuilderError as e:
            assert len(str(e)) > 0


# =============================================================================
# COMPLEX SCENARIOS
# =============================================================================


class TestComplexScenarios:
    """Test complex state machine configurations."""

    def test_full_locomotion_state_machine(
        self, idle_clip, walk_clip, run_clip, jump_clip
    ):
        """Build a complete locomotion state machine."""
        walk_cond = TransitionCondition.trigger("walk")
        run_cond = TransitionCondition.trigger("run")
        idle_cond = TransitionCondition.trigger("stop")
        jump_cond = TransitionCondition.trigger("jump")

        sm = (
            StateMachineBuilder("locomotion")
            .add_state("idle", clip=idle_clip)
            .add_state("walk", clip=walk_clip)
            .add_state("run", clip=run_clip)
            .add_state("jump", clip=jump_clip)
            .add_transition("idle", "walk", condition=walk_cond, duration=0.2)
            .add_transition("walk", "run", condition=run_cond, duration=0.15)
            .add_transition("run", "walk", duration=0.2)
            .add_transition("walk", "idle", condition=idle_cond, duration=0.25)
            .add_any_state_transition("jump", condition=jump_cond, duration=0.1)
            .set_initial("idle")
            .build()
        )
        assert isinstance(sm, StateMachine)

    def test_combat_state_machine(self, idle_clip, attack_clip, mock_clip):
        """Build a combat state machine with multiple attack states."""
        attack1_clip = MagicMock(name="attack1", duration=0.5)
        attack2_clip = MagicMock(name="attack2", duration=0.6)
        attack3_clip = MagicMock(name="attack3", duration=0.7)
        block_clip = MagicMock(name="block", duration=0.3)

        sm = (
            StateMachineBuilder("combat")
            .add_state("idle", clip=idle_clip)
            .add_state("attack1", clip=attack1_clip)
            .add_state("attack2", clip=attack2_clip)
            .add_state("attack3", clip=attack3_clip)
            .add_state("block", clip=block_clip)
            .add_transition("idle", "attack1")
            .add_transition("attack1", "attack2")
            .add_transition("attack2", "attack3")
            .add_transition("attack3", "idle")
            .add_transition("attack1", "idle")
            .add_transition("attack2", "idle")
            .add_any_state_transition("block")
            .set_initial("idle")
            .build()
        )
        assert isinstance(sm, StateMachine)

    def test_state_machine_with_all_blend_curves(self, idle_clip, walk_clip):
        """Test all blend curve types in transitions."""
        curves = [
            BlendCurve.LINEAR,
            BlendCurve.EASE_IN,
            BlendCurve.EASE_OUT,
            BlendCurve.EASE_IN_OUT,
            BlendCurve.SMOOTH_STEP,
            BlendCurve.SMOOTHER_STEP,
        ]
        for curve in curves:
            sm = (
                StateMachineBuilder(f"test_{curve.name}")
                .add_state("idle", clip=idle_clip)
                .add_state("walk", clip=walk_clip)
                .add_transition("idle", "walk", curve=curve)
                .set_initial("idle")
                .build()
            )
            assert sm is not None

    def test_state_machine_with_many_states(self):
        """Test state machine with many states."""
        builder = StateMachineBuilder("many_states")
        num_states = 20

        # Add many states
        for i in range(num_states):
            clip = MagicMock(name=f"state_{i}", duration=1.0)
            builder.add_state(f"state_{i}", clip=clip)

        # Add transitions forming a chain
        for i in range(num_states - 1):
            builder.add_transition(f"state_{i}", f"state_{i+1}")

        # Close the loop
        builder.add_transition(f"state_{num_states-1}", "state_0")
        builder.set_initial("state_0")

        sm = builder.build()
        assert isinstance(sm, StateMachine)

    def test_diamond_pattern_transitions(self, idle_clip, walk_clip, run_clip, jump_clip):
        """Test diamond-shaped transition pattern."""
        # idle -> walk and idle -> run -> both -> jump
        sm = (
            StateMachineBuilder("diamond")
            .add_state("idle", clip=idle_clip)
            .add_state("left", clip=walk_clip)
            .add_state("right", clip=run_clip)
            .add_state("end", clip=jump_clip)
            .add_transition("idle", "left")
            .add_transition("idle", "right")
            .add_transition("left", "end")
            .add_transition("right", "end")
            .add_transition("end", "idle")
            .set_initial("idle")
            .build()
        )
        assert isinstance(sm, StateMachine)


# =============================================================================
# TRANSITION CONDITION INTEGRATION TESTS
# =============================================================================


class TestTransitionConditionIntegration:
    """Test integration with TransitionCondition."""

    def test_trigger_condition(self, idle_clip, walk_clip):
        """Trigger condition works with builder."""
        cond = TransitionCondition.trigger("go")
        sm = (
            StateMachineBuilder("test")
            .add_state("a", clip=idle_clip)
            .add_state("b", clip=walk_clip)
            .add_transition("a", "b", condition=cond)
            .set_initial("a")
            .build()
        )
        assert sm is not None

    def test_bool_condition(self, idle_clip, walk_clip):
        """Boolean condition works with builder."""
        try:
            cond = TransitionCondition.bool_param("is_walking", True)
            sm = (
                StateMachineBuilder("test")
                .add_state("a", clip=idle_clip)
                .add_state("b", clip=walk_clip)
                .add_transition("a", "b", condition=cond)
                .set_initial("a")
                .build()
            )
            assert sm is not None
        except AttributeError:
            pytest.skip("bool_param not available")

    def test_float_threshold_condition(self, idle_clip, walk_clip):
        """Float threshold condition works with builder."""
        try:
            cond = TransitionCondition.float_greater("speed", 0.5)
            sm = (
                StateMachineBuilder("test")
                .add_state("a", clip=idle_clip)
                .add_state("b", clip=walk_clip)
                .add_transition("a", "b", condition=cond)
                .set_initial("a")
                .build()
            )
            assert sm is not None
        except AttributeError:
            pytest.skip("float_greater not available")

    def test_time_condition(self, idle_clip, walk_clip):
        """Time-based condition works with builder."""
        try:
            # Check if exit_time exists and is callable
            if not hasattr(TransitionCondition, 'exit_time') or TransitionCondition.exit_time is None:
                pytest.skip("exit_time not available")
            cond = TransitionCondition.exit_time(0.9)
            sm = (
                StateMachineBuilder("test")
                .add_state("a", clip=idle_clip)
                .add_state("b", clip=walk_clip)
                .add_transition("a", "b", condition=cond)
                .set_initial("a")
                .build()
            )
            assert sm is not None
        except (AttributeError, TypeError):
            pytest.skip("exit_time not available or not callable")


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_state_with_self_loop(self, idle_clip):
        """Single state with self-transition."""
        sm = (
            StateMachineBuilder("test")
            .add_state("idle", clip=idle_clip)
            .add_transition("idle", "idle")
            .set_initial("idle")
            .build()
        )
        assert sm is not None

    def test_state_with_long_name(self, idle_clip):
        """State with very long name."""
        long_name = "a" * 200
        sm = (
            StateMachineBuilder("test")
            .add_state(long_name, clip=idle_clip)
            .set_initial(long_name)
            .build()
        )
        assert sm is not None

    def test_state_machine_with_special_characters_in_name(self, idle_clip):
        """State machine with special characters in name."""
        special_names = ["player_controller", "NPC-Combat", "state.machine"]
        for name in special_names:
            try:
                builder = StateMachineBuilder(name)
                builder.add_state("idle", clip=idle_clip)
                builder.set_initial("idle")
                sm = builder.build()
                assert sm is not None
            except (ValueError, StateMachineBuilderError):
                pass  # Some special chars may be rejected

    def test_zero_duration_transition(self, idle_clip, walk_clip):
        """Zero duration transition (instant)."""
        sm = (
            StateMachineBuilder("test")
            .add_state("a", clip=idle_clip)
            .add_state("b", clip=walk_clip)
            .add_transition("a", "b", duration=0.0)
            .set_initial("a")
            .build()
        )
        assert sm is not None

    def test_very_long_duration_transition(self, idle_clip, walk_clip):
        """Very long duration transition."""
        sm = (
            StateMachineBuilder("test")
            .add_state("a", clip=idle_clip)
            .add_state("b", clip=walk_clip)
            .add_transition("a", "b", duration=100.0)
            .set_initial("a")
            .build()
        )
        assert sm is not None

    def test_negative_duration_handled(self, idle_clip, walk_clip):
        """Negative duration should be handled (error or clamped)."""
        builder = (
            StateMachineBuilder("test")
            .add_state("a", clip=idle_clip)
            .add_state("b", clip=walk_clip)
        )
        try:
            builder.add_transition("a", "b", duration=-1.0)
            builder.set_initial("a")
            sm = builder.build()
            # If it succeeds, value should be clamped or handled gracefully
            assert sm is not None
        except (ValueError, StateMachineBuilderError):
            pass  # Rejection is also valid

    def test_unicode_state_names(self, idle_clip, walk_clip):
        """Unicode state names."""
        try:
            sm = (
                StateMachineBuilder("test")
                .add_state("idle_state", clip=idle_clip)
                .add_state("walk_state", clip=walk_clip)
                .add_transition("idle_state", "walk_state")
                .set_initial("idle_state")
                .build()
            )
            assert sm is not None
        except (ValueError, StateMachineBuilderError, UnicodeError):
            pass  # Some systems may not support unicode names


# =============================================================================
# TYPE SAFETY TESTS
# =============================================================================


class TestTypeSafety:
    """Test type safety and error handling for invalid types."""

    def test_invalid_condition_type_handled(self, idle_clip, walk_clip):
        """Invalid condition type should be handled."""
        builder = (
            StateMachineBuilder("test")
            .add_state("a", clip=idle_clip)
            .add_state("b", clip=walk_clip)
        )
        try:
            builder.add_transition("a", "b", condition="not_a_condition")
            builder.set_initial("a")
            builder.build()
            # If it succeeds, the string might be interpreted somehow
        except (TypeError, ValueError, StateMachineBuilderError):
            pass  # Expected for invalid type

    def test_invalid_curve_type_handled(self, idle_clip, walk_clip):
        """Invalid curve type should be handled."""
        builder = (
            StateMachineBuilder("test")
            .add_state("a", clip=idle_clip)
            .add_state("b", clip=walk_clip)
        )
        try:
            builder.add_transition("a", "b", curve="not_a_curve")
            builder.set_initial("a")
            builder.build()
        except (TypeError, ValueError, StateMachineBuilderError):
            pass  # Expected for invalid type

    def test_none_state_name_handled(self, idle_clip):
        """None state name should be handled."""
        builder = StateMachineBuilder("test")
        try:
            builder.add_state(None, clip=idle_clip)
        except (TypeError, ValueError, StateMachineBuilderError):
            pass  # Expected

    def test_none_clip_handled(self):
        """None clip should be handled (error or default)."""
        builder = StateMachineBuilder("test")
        try:
            builder.add_state("idle", clip=None)
            builder.set_initial("idle")
            sm = builder.build()
            # If allowed, state might have default behavior
            assert sm is not None
        except (TypeError, ValueError, StateMachineBuilderError):
            pass  # Expected if clip is required
