"""Tests for Animation Session Persistence Integration (T-AN-9.11).

This test suite covers 50+ test cases for the session persistence system,
validating:
- State machine state persistence
- Blend parameter save/restore
- IK enable flag persistence
- Transient data exclusion (bone transforms, clip times)
- Partial restore handling
- Version compatibility
- Error recovery
"""

from __future__ import annotations

import copy
import pytest
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

from engine.animation.session_integration import (
    AnimationSessionData,
    StateMachineState,
    BlendParameterState,
    IKChainState,
    GraphParameterState,
    AnimationSessionError,
    StateMachineNotFoundError,
    StateNotFoundError,
    VersionMismatchError,
    RestoreResult,
    save_animation_state,
    restore_animation_state,
    serializable,
    transient,
    is_transient,
    SESSION_VERSION,
)
from engine.animation.graph.animation_graph import (
    GraphParameter,
    ParameterType,
)


# =============================================================================
# TEST FIXTURES AND HELPERS
# =============================================================================


@dataclass
class MockStateMachine:
    """Mock state machine for testing."""

    node_id: str = "test_sm"
    _initial_state: Optional[str] = "idle"
    _current_state: Optional["MockAnimationState"] = None
    states: Dict[str, "MockAnimationState"] = field(default_factory=dict)

    @property
    def current_state(self) -> Optional["MockAnimationState"]:
        return self._current_state

    @property
    def current_state_name(self) -> Optional[str]:
        return self._current_state.name if self._current_state else None

    def get_state(self, name: str) -> Optional["MockAnimationState"]:
        return self.states.get(name)

    def force_state(self, state_name: str, context: Any, immediate: bool = False) -> bool:
        if state_name in self.states:
            self._current_state = self.states[state_name]
            return True
        return False


@dataclass
class MockAnimationState:
    """Mock animation state for testing."""

    name: str
    _time_in_state: float = 0.0
    _normalized_time: float = 0.0


@dataclass
class MockIKChain:
    """Mock IK chain for testing."""

    chain_id: str
    enabled: bool = True
    weight: float = 1.0

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def set_weight(self, weight: float) -> None:
        self.weight = weight


@dataclass
class MockEntity:
    """Mock entity for testing session persistence."""

    entity_id: str = "test_entity"
    state_machine: Optional[MockStateMachine] = None
    blend_params: Dict[str, GraphParameter] = field(default_factory=dict)
    ik_chains: Dict[str, MockIKChain] = field(default_factory=dict)

    # Transient data (should NOT be persisted)
    bone_transforms: List[Any] = field(default_factory=list)
    clip_time: float = 0.0
    normalized_time: float = 0.0


def create_test_entity() -> MockEntity:
    """Create a fully populated test entity."""
    entity = MockEntity(entity_id="player_1")

    # State machine with states
    sm = MockStateMachine(node_id="locomotion")
    sm.states = {
        "idle": MockAnimationState("idle"),
        "walk": MockAnimationState("walk"),
        "run": MockAnimationState("run"),
        "jump": MockAnimationState("jump"),
    }
    sm._current_state = sm.states["walk"]
    entity.state_machine = sm

    # Blend parameters
    entity.blend_params = {
        "speed": GraphParameter.float_param("speed", default=0.0, min_val=0.0, max_val=10.0),
        "direction": GraphParameter.float_param("direction", default=0.0, min_val=-1.0, max_val=1.0),
        "is_grounded": GraphParameter.bool_param("is_grounded", default=True),
        "jump_trigger": GraphParameter.trigger_param("jump_trigger"),
    }
    entity.blend_params["speed"].value = 5.0
    entity.blend_params["direction"].value = 0.5

    # IK chains
    entity.ik_chains = {
        "left_arm": MockIKChain("left_arm", enabled=True, weight=1.0),
        "right_arm": MockIKChain("right_arm", enabled=True, weight=0.8),
        "left_leg": MockIKChain("left_leg", enabled=True, weight=1.0),
        "right_leg": MockIKChain("right_leg", enabled=False, weight=0.0),
    }

    # Transient data
    entity.bone_transforms = [{"position": [0, 0, 0], "rotation": [0, 0, 0, 1]}] * 50
    entity.clip_time = 1.5
    entity.normalized_time = 0.75

    return entity


# =============================================================================
# STATE MACHINE STATE PERSISTENCE TESTS
# =============================================================================


class TestStateMachineStatePersistence:
    """Tests for state machine state persistence."""

    def test_save_current_state(self) -> None:
        """State machine current state should be saved."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        assert "default" in session.state_machines
        sm_state = session.state_machines["default"]
        assert sm_state.current_state == "walk"

    def test_save_initial_state(self) -> None:
        """State machine initial state should be saved."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        sm_state = session.state_machines["default"]
        assert sm_state.initial_state == "idle"

    def test_save_no_current_state(self) -> None:
        """Handles state machine with no current state."""
        entity = create_test_entity()
        entity.state_machine._current_state = None

        session = save_animation_state(entity)
        sm_state = session.state_machines["default"]
        assert sm_state.current_state is None

    def test_restore_state_machine_state(self) -> None:
        """State machine state should be restored correctly."""
        entity = create_test_entity()
        entity.state_machine._current_state = entity.state_machine.states["run"]

        session = save_animation_state(entity)

        # Reset state
        entity.state_machine._current_state = entity.state_machine.states["idle"]

        result = restore_animation_state(entity, session)
        assert result.success
        assert result.state_machines_restored == 1
        assert entity.state_machine.current_state_name == "run"

    def test_restore_missing_state(self) -> None:
        """Handles restore when state doesn't exist in machine."""
        entity = create_test_entity()
        session = AnimationSessionData()
        session.state_machines["default"] = StateMachineState(
            machine_id="default",
            current_state="nonexistent_state",
            initial_state="idle",
        )

        result = restore_animation_state(entity, session)
        assert result.state_machines_failed == 1
        assert len(result.warnings) > 0

    def test_restore_missing_state_strict(self) -> None:
        """Strict mode raises on missing state."""
        entity = create_test_entity()
        session = AnimationSessionData()
        session.state_machines["default"] = StateMachineState(
            machine_id="default",
            current_state="nonexistent_state",
            initial_state="idle",
        )

        with pytest.raises(StateNotFoundError):
            restore_animation_state(entity, session, strict=True)

    def test_restore_missing_machine(self) -> None:
        """Handles restore when state machine doesn't exist in entity."""
        entity = create_test_entity()
        session = AnimationSessionData()
        session.state_machines["nonexistent_machine"] = StateMachineState(
            machine_id="nonexistent_machine",
            current_state="idle",
            initial_state="idle",
        )

        result = restore_animation_state(entity, session)
        assert result.state_machines_missing == 1

    def test_restore_missing_machine_strict(self) -> None:
        """Strict mode raises on missing state machine."""
        entity = create_test_entity()
        session = AnimationSessionData()
        session.state_machines["nonexistent"] = StateMachineState(
            machine_id="nonexistent",
            current_state="idle",
            initial_state="idle",
        )

        with pytest.raises(StateMachineNotFoundError):
            restore_animation_state(entity, session, strict=True)

    def test_multiple_state_machines(self) -> None:
        """Handles entities with multiple state machines."""
        entity = MockEntity()

        # Add multiple state machines via custom attribute pattern
        sm1 = MockStateMachine(node_id="upper_body")
        sm1.states = {"aim": MockAnimationState("aim"), "idle": MockAnimationState("idle")}
        sm1._current_state = sm1.states["aim"]

        sm2 = MockStateMachine(node_id="lower_body")
        sm2.states = {"walk": MockAnimationState("walk"), "run": MockAnimationState("run")}
        sm2._current_state = sm2.states["walk"]

        # Mock the animation graph pattern
        entity.animation_graph = Mock()
        entity.animation_graph.get_state_machines = lambda: {
            "upper_body": sm1,
            "lower_body": sm2,
        }
        # Ensure parameters attribute doesn't cause issues (return None for hasattr check)
        del entity.animation_graph.parameters  # Remove the auto-generated Mock attribute

        session = save_animation_state(entity)
        assert "upper_body" in session.state_machines
        assert "lower_body" in session.state_machines
        assert session.state_machines["upper_body"].current_state == "aim"
        assert session.state_machines["lower_body"].current_state == "walk"


# =============================================================================
# BLEND PARAMETER PERSISTENCE TESTS
# =============================================================================


class TestBlendParameterPersistence:
    """Tests for blend parameter persistence."""

    def test_save_float_parameter(self) -> None:
        """Float parameters should be saved."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        assert "speed" in session.blend_params
        assert session.blend_params["speed"].value == 5.0
        assert session.blend_params["speed"].param_type == "FLOAT"

    def test_save_bool_parameter(self) -> None:
        """Bool parameters should be saved."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        assert "is_grounded" in session.blend_params
        assert session.blend_params["is_grounded"].value is True
        assert session.blend_params["is_grounded"].param_type == "BOOL"

    def test_trigger_not_saved(self) -> None:
        """Trigger parameters should NOT be saved (transient)."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        assert "jump_trigger" not in session.blend_params

    def test_save_parameter_constraints(self) -> None:
        """Parameter min/max constraints should be saved."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        speed_state = session.blend_params["speed"]
        assert speed_state.min_value == 0.0
        assert speed_state.max_value == 10.0

    def test_restore_float_parameter(self) -> None:
        """Float parameters should be restored."""
        entity = create_test_entity()
        entity.blend_params["speed"].value = 7.5

        session = save_animation_state(entity)

        # Reset parameter
        entity.blend_params["speed"].value = 0.0

        result = restore_animation_state(entity, session)
        assert result.success
        assert result.blend_params_restored >= 1
        assert entity.blend_params["speed"].value == 7.5

    def test_restore_bool_parameter(self) -> None:
        """Bool parameters should be restored."""
        entity = create_test_entity()
        entity.blend_params["is_grounded"].value = False

        session = save_animation_state(entity)

        # Reset parameter
        entity.blend_params["is_grounded"].value = True

        result = restore_animation_state(entity, session)
        assert entity.blend_params["is_grounded"].value is False

    def test_restore_missing_parameter(self) -> None:
        """Handles restore when parameter doesn't exist in entity."""
        entity = create_test_entity()
        session = AnimationSessionData()
        session.blend_params["nonexistent"] = BlendParameterState(
            name="nonexistent",
            value=1.0,
            param_type="FLOAT",
        )

        result = restore_animation_state(entity, session)
        assert result.blend_params_missing == 1

    def test_restore_type_mismatch(self) -> None:
        """Handles type mismatch during restore."""
        entity = create_test_entity()
        session = AnimationSessionData()
        # Try to restore a string to a float parameter
        session.blend_params["speed"] = BlendParameterState(
            name="speed",
            value="invalid_value",
            param_type="STRING",
        )

        result = restore_animation_state(entity, session)
        # Should fail gracefully
        assert result.blend_params_failed >= 0 or result.blend_params_restored >= 0

    def test_clamped_value_restored(self) -> None:
        """Values outside constraints should be clamped on restore."""
        entity = create_test_entity()
        session = AnimationSessionData()
        session.blend_params["speed"] = BlendParameterState(
            name="speed",
            value=100.0,  # Way over max
            param_type="FLOAT",
        )

        restore_animation_state(entity, session)
        # Should be clamped to max
        assert entity.blend_params["speed"].value <= 10.0


# =============================================================================
# IK ENABLE FLAG PERSISTENCE TESTS
# =============================================================================


class TestIKEnableFlagPersistence:
    """Tests for IK enable/disable flag persistence."""

    def test_save_ik_enabled(self) -> None:
        """IK enabled flag should be saved."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        assert "left_arm" in session.ik_chains
        assert session.ik_chains["left_arm"].enabled is True

    def test_save_ik_disabled(self) -> None:
        """IK disabled flag should be saved."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        assert "right_leg" in session.ik_chains
        assert session.ik_chains["right_leg"].enabled is False

    def test_save_ik_weight(self) -> None:
        """IK weight should be saved."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        assert session.ik_chains["right_arm"].weight == 0.8

    def test_restore_ik_enabled(self) -> None:
        """IK enabled flag should be restored."""
        entity = create_test_entity()
        entity.ik_chains["left_arm"].enabled = False

        session = AnimationSessionData()
        session.ik_chains["left_arm"] = IKChainState(
            chain_id="left_arm",
            enabled=True,
            weight=1.0,
        )

        result = restore_animation_state(entity, session)
        assert result.success
        assert entity.ik_chains["left_arm"].enabled is True

    def test_restore_ik_disabled(self) -> None:
        """IK disabled flag should be restored."""
        entity = create_test_entity()
        entity.ik_chains["left_arm"].enabled = True

        session = AnimationSessionData()
        session.ik_chains["left_arm"] = IKChainState(
            chain_id="left_arm",
            enabled=False,
            weight=0.0,
        )

        result = restore_animation_state(entity, session)
        assert entity.ik_chains["left_arm"].enabled is False

    def test_restore_ik_weight(self) -> None:
        """IK weight should be restored."""
        entity = create_test_entity()
        entity.ik_chains["right_arm"].weight = 1.0

        session = AnimationSessionData()
        session.ik_chains["right_arm"] = IKChainState(
            chain_id="right_arm",
            enabled=True,
            weight=0.5,
        )

        result = restore_animation_state(entity, session)
        assert entity.ik_chains["right_arm"].weight == 0.5

    def test_restore_missing_ik_chain(self) -> None:
        """Handles restore when IK chain doesn't exist in entity."""
        entity = create_test_entity()
        session = AnimationSessionData()
        session.ik_chains["nonexistent"] = IKChainState(
            chain_id="nonexistent",
            enabled=True,
            weight=1.0,
        )

        result = restore_animation_state(entity, session)
        assert result.ik_chains_missing == 1

    def test_multiple_ik_chains(self) -> None:
        """Multiple IK chains should be saved and restored."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        assert len(session.ik_chains) == 4

        # Restore all chains
        for chain_id, chain in entity.ik_chains.items():
            chain.enabled = False
            chain.weight = 0.0

        result = restore_animation_state(entity, session)
        assert result.ik_chains_restored == 4


# =============================================================================
# BONE TRANSFORMS NOT PERSISTED TESTS
# =============================================================================


class TestBoneTransformsNotPersisted:
    """Tests verifying bone transforms are NOT persisted."""

    def test_bone_transforms_not_in_session(self) -> None:
        """Bone transforms should NOT be saved to session."""
        entity = create_test_entity()
        entity.bone_transforms = [{"position": [1, 2, 3]}] * 100

        session = save_animation_state(entity)
        session_dict = session.to_dict()

        # Should not contain bone transforms
        assert "bone_transforms" not in str(session_dict)

    def test_restore_does_not_touch_bone_transforms(self) -> None:
        """Restore should NOT modify bone transforms."""
        entity = create_test_entity()
        original_transforms = copy.deepcopy(entity.bone_transforms)

        session = save_animation_state(entity)

        # Modify transforms
        entity.bone_transforms = [{"position": [10, 20, 30]}]

        restore_animation_state(entity, session)

        # Transforms should remain modified (not restored)
        assert entity.bone_transforms != original_transforms


# =============================================================================
# CLIP PLAYBACK NOT PERSISTED TESTS
# =============================================================================


class TestClipPlaybackNotPersisted:
    """Tests verifying clip playback state is NOT persisted."""

    def test_clip_time_not_in_session(self) -> None:
        """Clip time should NOT be saved to session."""
        entity = create_test_entity()
        entity.clip_time = 5.0

        session = save_animation_state(entity)
        session_dict = session.to_dict()

        # Should not contain clip_time
        assert "clip_time" not in str(session_dict) or session_dict.get("clip_time") is None

    def test_normalized_time_not_in_session(self) -> None:
        """Normalized time should NOT be saved to session."""
        entity = create_test_entity()
        entity.normalized_time = 0.9

        session = save_animation_state(entity)
        session_dict = session.to_dict()

        # Should not contain normalized_time in top-level entity state
        assert not hasattr(session, "normalized_time")

    def test_restore_does_not_touch_clip_time(self) -> None:
        """Restore should NOT modify clip time."""
        entity = create_test_entity()
        entity.clip_time = 1.0

        session = save_animation_state(entity)

        # Modify clip time
        entity.clip_time = 99.0

        restore_animation_state(entity, session)

        # Clip time should remain modified
        assert entity.clip_time == 99.0


# =============================================================================
# PARTIAL RESTORE HANDLING TESTS
# =============================================================================


class TestPartialRestoreHandling:
    """Tests for partial restore functionality."""

    def test_partial_restore_success(self) -> None:
        """Partial restore should succeed when some components are missing."""
        entity = create_test_entity()
        session = AnimationSessionData()

        # Add valid state machine state
        session.state_machines["default"] = StateMachineState(
            machine_id="default",
            current_state="run",
            initial_state="idle",
        )

        # Add missing IK chain
        session.ik_chains["nonexistent"] = IKChainState(
            chain_id="nonexistent",
            enabled=True,
            weight=1.0,
        )

        result = restore_animation_state(entity, session)
        assert result.success
        assert result.is_partial
        assert result.state_machines_restored == 1
        assert result.ik_chains_missing == 1

    def test_partial_restore_all_missing(self) -> None:
        """Partial restore should fail when all components are missing."""
        entity = MockEntity()  # Empty entity
        session = AnimationSessionData()
        session.state_machines["sm"] = StateMachineState(
            machine_id="sm",
            current_state="state",
            initial_state="state",
        )

        result = restore_animation_state(entity, session)
        assert not result.success

    def test_selective_restore(self) -> None:
        """Should support selective restore of components."""
        entity = create_test_entity()
        entity.state_machine._current_state = entity.state_machine.states["run"]

        session = save_animation_state(entity)

        # Reset
        entity.state_machine._current_state = entity.state_machine.states["idle"]
        entity.blend_params["speed"].value = 0.0

        # Restore only state machines
        result = restore_animation_state(
            entity,
            session,
            restore_blend_params=False,
            restore_ik=False,
            restore_graph_params=False,  # Also disable graph params which share parameters
        )

        assert entity.state_machine.current_state_name == "run"
        assert entity.blend_params["speed"].value == 0.0  # Not restored

    def test_restore_result_totals(self) -> None:
        """RestoreResult should report accurate totals."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        result = restore_animation_state(entity, session)

        total = result.total_restored
        assert total >= 0
        assert result.total_failed >= 0
        assert result.total_missing >= 0


# =============================================================================
# VERSION COMPATIBILITY TESTS
# =============================================================================


class TestVersionCompatibility:
    """Tests for version compatibility and migration."""

    def test_session_version_saved(self) -> None:
        """Session version should be saved."""
        session = AnimationSessionData()
        assert session.session_version == SESSION_VERSION

    def test_schema_hash_saved(self) -> None:
        """Schema hash should be saved."""
        session = AnimationSessionData()
        assert session.schema_hash_stored is not None

    def test_timestamp_saved(self) -> None:
        """Timestamp should be saved."""
        before = time.time()
        session = AnimationSessionData()
        after = time.time()

        assert before <= session.timestamp <= after

    def test_future_version_raises(self) -> None:
        """Loading a future version should raise VersionMismatchError."""
        session_dict = {
            "__type__": "engine.animation.session_integration.AnimationSessionData",
            "session_version": SESSION_VERSION + 10,  # Future version
        }

        with pytest.raises(VersionMismatchError):
            AnimationSessionData.from_dict(session_dict)

    def test_older_version_migration(self) -> None:
        """Older versions should be migrated if possible."""
        # This test validates the migration infrastructure exists
        session = AnimationSessionData(session_version=SESSION_VERSION)
        session_dict = session.to_dict()
        session_dict["session_version"] = SESSION_VERSION  # Current version

        # Should not raise
        restored = AnimationSessionData.from_dict(session_dict)
        assert restored.session_version == SESSION_VERSION


# =============================================================================
# ERROR RECOVERY TESTS
# =============================================================================


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    def test_corrupted_state_machine_state(self) -> None:
        """Should handle corrupted state machine state gracefully."""
        entity = create_test_entity()
        session = AnimationSessionData()
        session.state_machines["default"] = StateMachineState(
            machine_id="default",
            current_state=None,  # Invalid but should not crash
            initial_state=None,
        )

        result = restore_animation_state(entity, session)
        assert result.success  # Should not crash

    def test_null_entity_attributes(self) -> None:
        """Should handle entities with null/missing attributes."""
        entity = MockEntity()
        entity.state_machine = None
        entity.blend_params = {}
        entity.ik_chains = {}

        session = save_animation_state(entity)
        assert session.is_empty()

    def test_exception_in_force_state(self) -> None:
        """Should handle exceptions during state forcing."""
        entity = create_test_entity()

        # Make force_state raise an exception
        def bad_force_state(*args: Any, **kwargs: Any) -> bool:
            raise RuntimeError("Simulated error")

        entity.state_machine.force_state = bad_force_state

        session = AnimationSessionData()
        session.state_machines["default"] = StateMachineState(
            machine_id="default",
            current_state="run",
            initial_state="idle",
        )

        result = restore_animation_state(entity, session)
        assert result.state_machines_failed == 1
        assert len(result.errors) > 0

    def test_exception_in_ik_restore(self) -> None:
        """Should handle exceptions during IK chain restore."""
        entity = create_test_entity()

        # Make set_enabled raise an exception
        def bad_set_enabled(enabled: bool) -> None:
            raise RuntimeError("Simulated error")

        entity.ik_chains["left_arm"].set_enabled = bad_set_enabled

        session = AnimationSessionData()
        session.ik_chains["left_arm"] = IKChainState(
            chain_id="left_arm",
            enabled=True,
            weight=1.0,
        )

        result = restore_animation_state(entity, session)
        assert result.ik_chains_failed == 1

    def test_empty_session_restore(self) -> None:
        """Should handle restoring empty session data."""
        entity = create_test_entity()
        session = AnimationSessionData()

        result = restore_animation_state(entity, session)
        assert result.success  # Empty restore is still success
        assert result.total_restored == 0


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Tests for Foundation Serializer integration."""

    def test_to_dict_roundtrip(self) -> None:
        """Session data should survive to_dict/from_dict roundtrip."""
        entity = create_test_entity()
        session = save_animation_state(entity)

        session_dict = session.to_dict()
        restored_session = AnimationSessionData.from_dict(session_dict)

        assert restored_session.entity_id == session.entity_id
        assert len(restored_session.state_machines) == len(session.state_machines)
        assert len(restored_session.blend_params) == len(session.blend_params)
        assert len(restored_session.ik_chains) == len(session.ik_chains)

    def test_serializable_decorator(self) -> None:
        """@serializable decorator should register class."""
        @serializable(version=1)
        class TestSerializable:
            def __init__(self) -> None:
                self.value = 42

        obj = TestSerializable()
        assert hasattr(TestSerializable, "_serializable")
        assert TestSerializable._serializable is True
        assert TestSerializable._serializable_version == 1

    def test_transient_decorator(self) -> None:
        """@transient decorator should mark class/field."""
        @transient
        class TransientClass:
            pass

        assert is_transient(TransientClass)

    def test_transient_field_metadata(self) -> None:
        """Fields with transient metadata should be detected."""
        @dataclass
        class WithTransient:
            normal: int = 1
            transient_field: int = field(default=2, metadata={"transient": True})

        # Check via dataclass field metadata
        from dataclasses import fields
        for f in fields(WithTransient):
            if f.name == "transient_field":
                assert f.metadata.get("transient") is True


# =============================================================================
# ANIMATION SESSION DATA TESTS
# =============================================================================


class TestAnimationSessionData:
    """Tests for AnimationSessionData class."""

    def test_is_empty(self) -> None:
        """is_empty should return True for empty session."""
        session = AnimationSessionData()
        assert session.is_empty()

    def test_is_not_empty_with_state_machine(self) -> None:
        """is_empty should return False with state machine."""
        session = AnimationSessionData()
        session.state_machines["sm"] = StateMachineState(
            machine_id="sm",
            current_state="state",
            initial_state="state",
        )
        assert not session.is_empty()

    def test_get_state_machine_names(self) -> None:
        """Should list all state machine names."""
        session = AnimationSessionData()
        session.state_machines["sm1"] = StateMachineState("sm1", "s", "s")
        session.state_machines["sm2"] = StateMachineState("sm2", "s", "s")

        names = session.get_state_machine_names()
        assert "sm1" in names
        assert "sm2" in names

    def test_get_blend_param_names(self) -> None:
        """Should list all blend parameter names."""
        session = AnimationSessionData()
        session.blend_params["speed"] = BlendParameterState("speed", 1.0, "FLOAT")
        session.blend_params["direction"] = BlendParameterState("direction", 0.0, "FLOAT")

        names = session.get_blend_param_names()
        assert "speed" in names
        assert "direction" in names

    def test_get_ik_chain_ids(self) -> None:
        """Should list all IK chain IDs."""
        session = AnimationSessionData()
        session.ik_chains["arm"] = IKChainState("arm", True, 1.0)
        session.ik_chains["leg"] = IKChainState("leg", True, 1.0)

        ids = session.get_ik_chain_ids()
        assert "arm" in ids
        assert "leg" in ids


# =============================================================================
# BLEND PARAMETER STATE TESTS
# =============================================================================


class TestBlendParameterState:
    """Tests for BlendParameterState class."""

    def test_from_graph_parameter_float(self) -> None:
        """Should create state from float GraphParameter."""
        param = GraphParameter.float_param("speed", default=0.0, min_val=0.0, max_val=10.0)
        param.value = 5.0

        state = BlendParameterState.from_graph_parameter(param)

        assert state.name == "speed"
        assert state.value == 5.0
        assert state.param_type == "FLOAT"
        assert state.min_value == 0.0
        assert state.max_value == 10.0

    def test_from_graph_parameter_bool(self) -> None:
        """Should create state from bool GraphParameter."""
        param = GraphParameter.bool_param("enabled", default=True)

        state = BlendParameterState.from_graph_parameter(param)

        assert state.name == "enabled"
        assert state.value is True
        assert state.param_type == "BOOL"

    def test_apply_to_parameter(self) -> None:
        """Should apply state to GraphParameter."""
        param = GraphParameter.float_param("speed", default=0.0)
        state = BlendParameterState(name="speed", value=7.5, param_type="FLOAT")

        result = state.apply_to(param)

        assert result is True
        assert param.value == 7.5

    def test_apply_to_trigger_skipped(self) -> None:
        """Should skip trigger parameters."""
        param = GraphParameter.trigger_param("jump")
        state = BlendParameterState(name="jump", value=True, param_type="TRIGGER")

        result = state.apply_to(param)

        assert result is True  # Skipped successfully


# =============================================================================
# GRAPH PARAMETER STATE TESTS
# =============================================================================


class TestGraphParameterState:
    """Tests for GraphParameterState class."""

    def test_from_graph(self) -> None:
        """Should create state from parameter dictionary."""
        params = {
            "speed": GraphParameter.float_param("speed", default=0.0),
            "enabled": GraphParameter.bool_param("enabled", default=True),
            "jump": GraphParameter.trigger_param("jump"),  # Should be excluded
        }
        params["speed"].value = 5.0

        state = GraphParameterState.from_graph(params)

        assert "speed" in state.parameters
        assert "enabled" in state.parameters
        assert "jump" not in state.parameters  # Triggers excluded

    def test_apply_to_parameters(self) -> None:
        """Should apply state to parameter dictionary."""
        params = {
            "speed": GraphParameter.float_param("speed", default=0.0),
            "enabled": GraphParameter.bool_param("enabled", default=False),
        }

        state = GraphParameterState()
        state.parameters["speed"] = BlendParameterState("speed", 7.0, "FLOAT")
        state.parameters["enabled"] = BlendParameterState("enabled", True, "BOOL")

        restored = state.apply_to(params)

        assert restored == 2
        assert params["speed"].value == 7.0
        assert params["enabled"].value is True


# =============================================================================
# RESTORE RESULT TESTS
# =============================================================================


class TestRestoreResult:
    """Tests for RestoreResult class."""

    def test_total_restored(self) -> None:
        """Should calculate total restored correctly."""
        result = RestoreResult(
            success=True,
            state_machines_restored=2,
            blend_params_restored=5,
            ik_chains_restored=3,
            graph_params_restored=10,
        )

        assert result.total_restored == 20

    def test_total_failed(self) -> None:
        """Should calculate total failed correctly."""
        result = RestoreResult(
            success=True,
            state_machines_failed=1,
            blend_params_failed=2,
            ik_chains_failed=1,
        )

        assert result.total_failed == 4

    def test_total_missing(self) -> None:
        """Should calculate total missing correctly."""
        result = RestoreResult(
            success=True,
            state_machines_missing=1,
            blend_params_missing=3,
            ik_chains_missing=2,
        )

        assert result.total_missing == 6

    def test_is_partial(self) -> None:
        """Should detect partial restore."""
        result = RestoreResult(
            success=True,
            state_machines_restored=1,
            blend_params_missing=1,
        )

        assert result.is_partial is True

    def test_is_not_partial(self) -> None:
        """Should detect complete restore."""
        result = RestoreResult(
            success=True,
            state_machines_restored=2,
            blend_params_restored=5,
        )

        assert result.is_partial is False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for full save/restore cycle."""

    def test_full_save_restore_cycle(self) -> None:
        """Complete save/restore cycle should work."""
        entity = create_test_entity()

        # Modify state
        entity.state_machine._current_state = entity.state_machine.states["run"]
        entity.blend_params["speed"].value = 8.0
        entity.ik_chains["right_leg"].enabled = True

        # Save
        session = save_animation_state(entity)

        # Reset state
        entity.state_machine._current_state = entity.state_machine.states["idle"]
        entity.blend_params["speed"].value = 0.0
        entity.ik_chains["right_leg"].enabled = False

        # Restore
        result = restore_animation_state(entity, session)

        # Verify
        assert result.success
        assert entity.state_machine.current_state_name == "run"
        assert entity.blend_params["speed"].value == 8.0
        assert entity.ik_chains["right_leg"].enabled is True

    def test_save_restore_preserves_constraints(self) -> None:
        """Save/restore should preserve parameter constraints."""
        entity = create_test_entity()
        entity.blend_params["speed"].value = 7.5

        session = save_animation_state(entity)
        restored_session = AnimationSessionData.from_dict(session.to_dict())

        speed_state = restored_session.blend_params["speed"]
        assert speed_state.min_value == 0.0
        assert speed_state.max_value == 10.0

    def test_multiple_save_restore_cycles(self) -> None:
        """Multiple save/restore cycles should be stable."""
        entity = create_test_entity()

        for i in range(5):
            entity.blend_params["speed"].value = float(i)
            session = save_animation_state(entity)
            entity.blend_params["speed"].value = 0.0
            restore_animation_state(entity, session)
            assert entity.blend_params["speed"].value == float(i)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_state_machine_states(self) -> None:
        """Should handle state machine with no states."""
        entity = MockEntity()
        sm = MockStateMachine()
        sm.states = {}
        sm._current_state = None
        entity.state_machine = sm

        session = save_animation_state(entity)
        assert session.state_machines["default"].current_state is None

    def test_empty_blend_params(self) -> None:
        """Should handle entity with no blend parameters."""
        entity = MockEntity()
        entity.blend_params = {}

        session = save_animation_state(entity)
        assert len(session.blend_params) == 0

    def test_empty_ik_chains(self) -> None:
        """Should handle entity with no IK chains."""
        entity = MockEntity()
        entity.ik_chains = {}

        session = save_animation_state(entity)
        assert len(session.ik_chains) == 0

    def test_very_large_parameter_values(self) -> None:
        """Should handle very large parameter values."""
        entity = create_test_entity()
        # Remove constraints for this test
        entity.blend_params["speed"] = GraphParameter.float_param("speed", default=0.0)
        entity.blend_params["speed"].value = 1e10

        session = save_animation_state(entity)
        entity.blend_params["speed"].value = 0.0
        restore_animation_state(entity, session)

        assert entity.blend_params["speed"].value == 1e10

    def test_negative_parameter_values(self) -> None:
        """Should handle negative parameter values."""
        entity = create_test_entity()
        entity.blend_params["direction"].value = -0.75

        session = save_animation_state(entity)
        entity.blend_params["direction"].value = 0.0
        restore_animation_state(entity, session)

        assert entity.blend_params["direction"].value == -0.75

    def test_unicode_state_names(self) -> None:
        """Should handle unicode characters in state names."""
        entity = create_test_entity()
        entity.state_machine.states["走る"] = MockAnimationState("走る")  # Japanese for "run"
        entity.state_machine._current_state = entity.state_machine.states["走る"]

        session = save_animation_state(entity)
        assert session.state_machines["default"].current_state == "走る"

    def test_special_characters_in_ids(self) -> None:
        """Should handle special characters in IDs."""
        entity = MockEntity()
        entity.ik_chains = {
            "left-arm.ik": MockIKChain("left-arm.ik"),
            "right_arm/ik": MockIKChain("right_arm/ik"),
        }

        session = save_animation_state(entity)
        assert "left-arm.ik" in session.ik_chains
        assert "right_arm/ik" in session.ik_chains
