"""
Tests for data flow decorators (data_flow.py).

Tests the 4 data flow decorators built on Ops:
    @serializable, @networked, @snapshot, @versioned

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
5. Generated methods exist and work
6. Decorator dependencies are enforced
"""

import pytest

from trinity.decorators.data_flow import (
    VALID_INTERPOLATION_MODES,
    VALID_NETWORK_AUTHORITY,
    VALID_NETWORK_RELEVANCE,
    VALID_SERIALIZATION_FORMATS,
    NetworkedConfig,
    SerializableConfig,
    SnapshotConfig,
    VersionedConfig,
    networked,
    serializable,
    snapshot,
    versioned,
)
from trinity.decorators.ops import Op, decompose, expand


# =============================================================================
# @serializable
# =============================================================================


class TestSerializable:
    def test_default_params(self):
        @serializable()
        class Foo:
            pass

        assert Foo._serializable is True
        config = Foo._tags.get("serializable_config")
        assert isinstance(config, SerializableConfig)
        assert Foo._serializable_format == "binary"
        assert Foo._serializable_version == 1

    def test_custom_format_and_version(self):
        @serializable(format="json", version=2)
        class Bar:
            pass

        assert Bar._serializable_format == "json"
        assert Bar._serializable_version == 2

    def test_msgpack_format(self):
        @serializable(format="msgpack", version=3)
        class Baz:
            pass

        assert Baz._serializable_format == "msgpack"
        assert Baz._serializable_version == 3

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="invalid format"):

            @serializable(format="xml")
            class Bad:
                pass

    def test_invalid_version_negative(self):
        with pytest.raises(ValueError, match="version must be positive integer"):

            @serializable(version=-1)
            class Bad:
                pass

    def test_invalid_version_zero(self):
        with pytest.raises(ValueError, match="version must be positive integer"):

            @serializable(version=0)
            class Bad:
                pass

    def test_applied_decorators(self):
        @serializable()
        class C:
            pass

        assert "serializable" in C._applied_decorators

    def test_steps_recorded(self):
        @serializable()
        class C:
            pass

        assert hasattr(C, "_applied_steps")
        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used
        assert Op.DESCRIBE in ops_used

    def test_decompose(self):
        steps = decompose(serializable)
        assert len(steps) > 0

    def test_no_parens(self):
        @serializable
        class C:
            pass

        assert C._serializable is True

    def test_serializable_fields_from_annotations(self):
        @serializable()
        class Player:
            x: float
            y: float
            name: str

        assert "x" in Player._serializable_fields
        assert "y" in Player._serializable_fields
        assert "name" in Player._serializable_fields

    def test_serializable_fields_no_annotations(self):
        @serializable()
        class NoAnnotations:
            pass

        assert NoAnnotations._serializable_fields == []

    def test_serialize_method_exists(self):
        @serializable()
        class C:
            pass

        assert hasattr(C, "serialize")
        assert callable(C.serialize)

    def test_deserialize_method_exists(self):
        @serializable()
        class C:
            pass

        assert hasattr(C, "deserialize")
        assert callable(C.deserialize)

    def test_serialize_deserialize_roundtrip(self):
        @serializable(format="json")
        class Position:
            x: float
            y: float

            def __init__(self, x: float, y: float):
                self.x = x
                self.y = y

        pos = Position(10.5, 20.3)
        data = Position.serialize(pos)
        assert data["__type__"] == "Position"
        assert data["__version__"] == 1
        assert data["x"] == 10.5
        assert data["y"] == 20.3

        restored = Position.deserialize(data)
        assert restored.x == 10.5
        assert restored.y == 20.3

    def test_serialize_wrong_type(self):
        @serializable()
        class A:
            pass

        @serializable()
        class B:
            pass

        b = B()
        with pytest.raises(TypeError, match="Expected A"):
            A.serialize(b)

    def test_deserialize_wrong_type(self):
        @serializable()
        class A:
            pass

        data = {"__type__": "WrongType", "__version__": 1}
        with pytest.raises(ValueError, match="Type mismatch"):
            A.deserialize(data)


# =============================================================================
# @networked
# =============================================================================


class TestNetworked:
    def test_default_params(self):
        @networked()
        class Entity:
            pass

        assert Entity._networked is True
        config = Entity._tags.get("networked_config")
        assert isinstance(config, NetworkedConfig)
        assert Entity._networked_relevance == "spatial"
        assert Entity._networked_authority == "server"
        assert Entity._networked_priority == 0
        assert Entity._networked_unreliable is False
        assert Entity._networked_delta is False
        assert Entity._networked_predicted is False
        assert Entity._networked_interpolated == "none"

    def test_custom_relevance(self):
        @networked(relevance="global")
        class GlobalEntity:
            pass

        assert GlobalEntity._networked_relevance == "global"

    def test_owner_relevance(self):
        @networked(relevance="owner", authority="owner")
        class OwnedEntity:
            pass

        assert OwnedEntity._networked_relevance == "owner"
        assert OwnedEntity._networked_authority == "owner"

    def test_custom_priority(self):
        @networked(priority=10)
        class HighPriority:
            pass

        assert HighPriority._networked_priority == 10

    def test_unreliable_and_delta(self):
        @networked(unreliable=True, delta=True)
        class FastUpdate:
            pass

        assert FastUpdate._networked_unreliable is True
        assert FastUpdate._networked_delta is True

    def test_predicted_entity(self):
        @networked(predicted=True, authority="client")
        class Predicted:
            pass

        assert Predicted._networked_predicted is True

    def test_linear_interpolation(self):
        @networked(interpolated="linear")
        class Smooth:
            pass

        assert Smooth._networked_interpolated == "linear"

    def test_hermite_interpolation(self):
        @networked(interpolated="hermite")
        class VerySmooth:
            pass

        assert VerySmooth._networked_interpolated == "hermite"

    def test_invalid_relevance(self):
        with pytest.raises(ValueError, match="invalid relevance"):

            @networked(relevance="regional")
            class Bad:
                pass

    def test_invalid_authority(self):
        with pytest.raises(ValueError, match="invalid authority"):

            @networked(authority="peer")
            class Bad:
                pass

    def test_invalid_priority_type(self):
        with pytest.raises(ValueError, match="priority must be integer"):

            @networked(priority="high")
            class Bad:
                pass

    def test_invalid_interpolation(self):
        with pytest.raises(ValueError, match="invalid interpolated"):

            @networked(interpolated="cubic")
            class Bad:
                pass

    def test_applied_decorators(self):
        @networked()
        class C:
            pass

        assert "networked" in C._applied_decorators

    def test_steps_recorded(self):
        @networked()
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used

    def test_serialize_net_method_exists(self):
        @networked()
        class C:
            pass

        assert hasattr(C, "_serialize_net")

    def test_deserialize_net_method_exists(self):
        @networked()
        class C:
            pass

        assert hasattr(C, "_deserialize_net")

    def test_serialize_net_works(self):
        @serializable()
        @networked()
        class NetEntity:
            x: int

            def __init__(self, x: int):
                self.x = x

        entity = NetEntity(42)
        data = entity._serialize_net()
        assert data["__type__"] == "NetEntity"
        assert data["x"] == 42

    def test_deserialize_net_works(self):
        @serializable()
        @networked()
        class NetEntity:
            x: int

            def __init__(self):
                self.x = 0

        entity = NetEntity()
        entity._deserialize_net({"x": 99})
        assert entity.x == 99


# =============================================================================
# @snapshot
# =============================================================================


class TestSnapshot:
    def test_requires_serializable(self):
        """@snapshot without @serializable should fail."""
        with pytest.raises(TypeError, match="requires @serializable"):

            @snapshot()
            class NoSerializable:
                pass

    def test_default_history(self):
        @snapshot()
        @serializable()
        class Snapshotted:
            pass

        assert Snapshotted._snapshot is True
        config = Snapshotted._tags.get("snapshot_config")
        assert isinstance(config, SnapshotConfig)
        assert Snapshotted._snapshot_history_frames == 60

    def test_custom_history(self):
        @snapshot(history_frames=120)
        @serializable()
        class LongHistory:
            pass

        assert LongHistory._snapshot_history_frames == 120

    def test_invalid_history_negative(self):
        with pytest.raises(ValueError, match="history_frames must be positive integer"):

            @snapshot(history_frames=-10)
            @serializable()
            class Bad:
                pass

    def test_invalid_history_zero(self):
        with pytest.raises(ValueError, match="history_frames must be positive integer"):

            @snapshot(history_frames=0)
            @serializable()
            class Bad:
                pass

    def test_applied_decorators(self):
        @snapshot()
        @serializable()
        class C:
            pass

        assert "snapshot" in C._applied_decorators
        assert "serializable" in C._applied_decorators

    def test_steps_recorded(self):
        @snapshot()
        @serializable()
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used

    def test_snapshot_save_method_exists(self):
        @snapshot()
        @serializable()
        class C:
            pass

        assert hasattr(C, "snapshot_save")

    def test_snapshot_restore_method_exists(self):
        @snapshot()
        @serializable()
        class C:
            pass

        assert hasattr(C, "snapshot_restore")

    def test_snapshot_save_restore_works(self):
        @snapshot(history_frames=5)
        @serializable()
        class GameState:
            score: int

            def __init__(self, score: int):
                self.score = score

        state = GameState(100)
        frame0 = state.snapshot_save()
        assert frame0 == 0

        state.score = 200
        frame1 = state.snapshot_save()
        assert frame1 == 1

        state.score = 300
        state.snapshot_save()

        # Restore to frame 1
        result = state.snapshot_restore(1)
        assert result is True
        assert state.score == 200

    def test_snapshot_restore_invalid_frame(self):
        @snapshot()
        @serializable()
        class State:
            x: int

            def __init__(self):
                self.x = 0

        state = State()
        state.snapshot_save()

        # Out of range frame
        result = state.snapshot_restore(999)
        assert result is False

    def test_snapshot_ring_buffer(self):
        @snapshot(history_frames=3)
        @serializable()
        class State:
            val: int

            def __init__(self, val: int):
                self.val = val

        state = State(1)
        state.snapshot_save()  # frame 0
        state.val = 2
        state.snapshot_save()  # frame 1
        state.val = 3
        state.snapshot_save()  # frame 2
        state.val = 4
        state.snapshot_save()  # frame 3, should evict frame 0

        # Only 3 frames should be in history
        assert len(state._snapshot_history) == 3


# =============================================================================
# @versioned
# =============================================================================


class TestVersioned:
    def test_requires_serializable(self):
        """@versioned without @serializable should fail."""
        with pytest.raises(TypeError, match="requires @serializable"):

            @versioned(version=1)
            class NoSerializable:
                pass

    def test_default_version(self):
        @versioned(version=1)
        @serializable()
        class V1:
            pass

        assert V1._versioned is True
        config = V1._tags.get("versioned_config")
        assert isinstance(config, VersionedConfig)
        assert V1._versioned_version == 1
        assert V1._versioned_migrations == {}

    def test_custom_version(self):
        @versioned(version=5)
        @serializable()
        class V5:
            pass

        assert V5._versioned_version == 5

    def test_with_migrations(self):
        migrations = {
            1: lambda data: data,
            2: lambda data: {**data, "new_field": None},
        }

        @versioned(version=2, migrations=migrations)
        @serializable()
        class Versioned:
            pass

        assert Versioned._versioned_migrations == migrations

    def test_invalid_version_negative(self):
        with pytest.raises(ValueError, match="version must be positive integer"):

            @versioned(version=-1)
            @serializable()
            class Bad:
                pass

    def test_invalid_version_zero(self):
        with pytest.raises(ValueError, match="version must be positive integer"):

            @versioned(version=0)
            @serializable()
            class Bad:
                pass

    def test_invalid_migrations_type(self):
        with pytest.raises(ValueError, match="migrations must be dict"):

            @versioned(version=1, migrations=[])
            @serializable()
            class Bad:
                pass

    def test_applied_decorators(self):
        @versioned(version=1)
        @serializable()
        class C:
            pass

        assert "versioned" in C._applied_decorators
        assert "serializable" in C._applied_decorators

    def test_steps_recorded(self):
        @versioned(version=1)
        @serializable()
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used
        assert Op.VALIDATE in ops_used

    def test_validate_step_present(self):
        @versioned(version=1)
        @serializable()
        class C:
            pass

        validate_steps = [s for s in C._applied_steps if s.op is Op.VALIDATE]
        assert len(validate_steps) > 0
        assert any(
            s.args.get("check") == "requires_serializable" for s in validate_steps
        )


# =============================================================================
# STACKING TESTS
# =============================================================================


class TestDataFlowStacking:
    def test_serializable_and_networked(self):
        @serializable(format="json")
        @networked(relevance="global")
        class NetworkedData:
            pass

        assert NetworkedData._serializable is True
        assert NetworkedData._networked is True

    def test_serializable_snapshot_versioned(self):
        @versioned(version=2)
        @snapshot(history_frames=30)
        @serializable()
        class FullStack:
            pass

        assert FullStack._serializable is True
        assert FullStack._snapshot is True
        assert FullStack._versioned is True

    def test_all_four_decorators(self):
        @versioned(version=1)
        @snapshot(history_frames=60)
        @networked(relevance="spatial", authority="server")
        @serializable(format="msgpack", version=1)
        class CompleteEntity:
            pass

        assert CompleteEntity._serializable is True
        assert CompleteEntity._networked is True
        assert CompleteEntity._snapshot is True
        assert CompleteEntity._versioned is True


# =============================================================================
# INTROSPECTION (all decorators decompose)
# =============================================================================


class TestDataFlowIntrospection:
    @pytest.mark.parametrize("dec", [serializable, networked, snapshot, versioned])
    def test_decompose_returns_steps(self, dec):
        steps = decompose(dec)
        assert isinstance(steps, list)

    @pytest.mark.parametrize("dec", [serializable, networked, snapshot, versioned])
    def test_expand_returns_string(self, dec):
        result = expand(dec)
        assert isinstance(result, str)

    def test_all_register_data_flow(self):
        """Every data flow decorator should have a REGISTER step for 'data_flow'."""
        for dec in [serializable, networked, snapshot, versioned]:
            steps = decompose(dec)
            reg_steps = [s for s in steps if s.op is Op.REGISTER]
            assert any(
                s.args.get("registry") == "data_flow" for s in reg_steps
            ), f"{dec.__name__} missing REGISTER(data_flow) step"

    def test_serializable_has_describe_step(self):
        steps = decompose(serializable)
        describe_steps = [s for s in steps if s.op is Op.DESCRIBE]
        assert len(describe_steps) > 0

    def test_versioned_has_validate_step(self):
        steps = decompose(versioned)
        validate_steps = [s for s in steps if s.op is Op.VALIDATE]
        assert len(validate_steps) > 0


# =============================================================================
# CONFIG CLASSES
# =============================================================================


class TestConfigClasses:
    def test_serializable_config_defaults(self):
        config = SerializableConfig()
        assert config.format == "binary"
        assert config.version == 1

    def test_networked_config_defaults(self):
        config = NetworkedConfig()
        assert config.relevance == "spatial"
        assert config.authority == "server"
        assert config.priority == 0
        assert config.unreliable is False
        assert config.delta is False
        assert config.predicted is False
        assert config.interpolated == "none"

    def test_snapshot_config_defaults(self):
        config = SnapshotConfig()
        assert config.history_frames == 60

    def test_versioned_config_defaults(self):
        config = VersionedConfig()
        assert config.version == 1
        assert config.migrations == {}

    def test_config_frozen(self):
        config = SerializableConfig(format="json", version=2)
        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
            config.format = "binary"


# =============================================================================
# VALID VALUES
# =============================================================================


class TestValidValues:
    def test_serialization_formats(self):
        assert "binary" in VALID_SERIALIZATION_FORMATS
        assert "json" in VALID_SERIALIZATION_FORMATS
        assert "msgpack" in VALID_SERIALIZATION_FORMATS

    def test_network_relevance(self):
        assert "global" in VALID_NETWORK_RELEVANCE
        assert "spatial" in VALID_NETWORK_RELEVANCE
        assert "owner" in VALID_NETWORK_RELEVANCE

    def test_network_authority(self):
        assert "server" in VALID_NETWORK_AUTHORITY
        assert "client" in VALID_NETWORK_AUTHORITY
        assert "owner" in VALID_NETWORK_AUTHORITY

    def test_interpolation_modes(self):
        assert "linear" in VALID_INTERPOLATION_MODES
        assert "hermite" in VALID_INTERPOLATION_MODES
        assert "none" in VALID_INTERPOLATION_MODES
