"""T-NET-1.7: Registration edge cases for ProtocolMeta.

Tests cover:
- Registry query edge cases (non-existent lookups)
- Inheritance chains with registered protocols
- Message ID boundary values (0, negative, large int)
- Version decoder error propagation
- Message type decode error handling
- Negotiation edge cases (duplicates, wraparound)
- Thread safety during concurrent registration
"""

from __future__ import annotations

import threading

import pytest

from trinity.metaclasses import ProtocolMeta


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    ProtocolMeta.clear_registry()
    yield
    ProtocolMeta.clear_registry()


# ---------------------------------------------------------------------------
# Registry query edge cases
# ---------------------------------------------------------------------------

class TestRegistryQueries:
    """Edge cases for get_by_id, get_by_name, all_protocols."""

    def test_get_by_id_nonexistent(self):
        """get_by_id returns None for a protocol ID that was never created."""
        result = ProtocolMeta.get_by_id(999)
        assert result is None

    def test_get_by_name_nonexistent(self):
        """get_by_name returns None for a name that was never registered."""
        result = ProtocolMeta.get_by_name("nonexistent.module.FakeProto")
        assert result is None

    def test_get_by_id_after_clear(self):
        """get_by_id returns None after clear_registry removes the entry."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        proto_id = Proto._protocol_id
        assert ProtocolMeta.get_by_id(proto_id) is Proto

        ProtocolMeta.clear_registry()
        assert ProtocolMeta.get_by_id(proto_id) is None

    def test_get_by_name_after_clear(self):
        """get_by_name returns None after clear_registry removes the entry."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        name = Proto._protocol_qualified_name
        assert ProtocolMeta.get_by_name(name) is Proto

        ProtocolMeta.clear_registry()
        assert ProtocolMeta.get_by_name(name) is None

    def test_all_protocols_empty_initially(self):
        """all_protocols returns empty list when no protocols are registered."""
        assert ProtocolMeta.all_protocols() == []

    def test_all_protocols_after_clear(self):
        """all_protocols returns empty list after clear_registry."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        assert len(ProtocolMeta.all_protocols()) == 1
        ProtocolMeta.clear_registry()
        assert ProtocolMeta.all_protocols() == []


# ---------------------------------------------------------------------------
# Inheritance chain with registered protocols
# ---------------------------------------------------------------------------

class TestInheritanceRegistration:
    """ProtocolMeta correctly handles inheritance chains."""

    def test_subclass_of_registered_protocol(self):
        """Subclass of a registered protocol is independently registered."""

        class BaseProto(metaclass=ProtocolMeta):
            _protocol_version = 1

        class ChildProto(BaseProto):
            _protocol_version = 2

        all_protos = ProtocolMeta.all_protocols()
        assert BaseProto in all_protos
        assert ChildProto in all_protos
        assert ChildProto._protocol_id != BaseProto._protocol_id
        assert ChildProto._protocol_id > BaseProto._protocol_id

    def test_subclass_inherits_min_version(self):
        """Subclass inherits _protocol_min_version from parent if not overridden."""

        class BaseProto(metaclass=ProtocolMeta):
            _protocol_version = 5
            _protocol_min_version = 3

        class ChildProto(BaseProto):
            _protocol_version = 7

        assert ChildProto._protocol_min_version == 3
        assert ChildProto._protocol_version == 7

    def test_subclass_overrides_min_version(self):
        """Subclass can override _protocol_min_version."""

        class BaseProto(metaclass=ProtocolMeta):
            _protocol_version = 5
            _protocol_min_version = 3

        class ChildProto(BaseProto):
            _protocol_version = 7
            _protocol_min_version = 6

        assert ChildProto._protocol_min_version == 6

    def test_subclass_defaults_min_version_to_own_version(self):
        """When parent has no min_version, subclass defaults to own version."""

        class BaseProto(metaclass=ProtocolMeta):
            _protocol_version = 5

        class ChildProto(BaseProto):
            _protocol_version = 3

        assert ChildProto._protocol_min_version == 3

    def test_deep_inheritance_chain(self):
        """Three-level inheritance: grandchild is registered independently."""

        class Grandparent(metaclass=ProtocolMeta):
            _protocol_version = 1

        class Parent(Grandparent):
            _protocol_version = 2

        class Child(Parent):
            _protocol_version = 3

        all_ids = [
            Grandparent._protocol_id,
            Parent._protocol_id,
            Child._protocol_id,
        ]
        assert len(set(all_ids)) == 3
        assert sorted(all_ids) == all_ids  # sequential ascending


# ---------------------------------------------------------------------------
# Message ID boundary values
# ---------------------------------------------------------------------------

class TestMessageRegistrationBoundaries:
    """Edge cases for register_message with boundary message IDs."""

    def test_register_message_id_zero(self):
        """Message ID 0 is valid."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        class Msg:
            pass

        ProtocolMeta.register_message(Proto, 0, Msg)
        assert ProtocolMeta.get_message_type(Proto, 0) is Msg

    def test_register_message_id_negative(self):
        """Negative message IDs are valid (no validation against negatives)."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        class Msg:
            pass

        ProtocolMeta.register_message(Proto, -1, Msg)
        assert ProtocolMeta.get_message_type(Proto, -1) is Msg

    def test_register_message_large_id(self):
        """Large message ID (max int) is valid."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        class Msg:
            pass

        ProtocolMeta.register_message(Proto, 2**31 - 1, Msg)
        assert ProtocolMeta.get_message_type(Proto, 2**31 - 1) is Msg

    def test_register_message_zero_and_positive_are_distinct(self):
        """Message IDs 0 and 1 are distinct entries."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        class Msg0:
            pass

        class Msg1:
            pass

        ProtocolMeta.register_message(Proto, 0, Msg0)
        ProtocolMeta.register_message(Proto, 1, Msg1)

        assert ProtocolMeta.get_message_type(Proto, 0) is Msg0
        assert ProtocolMeta.get_message_type(Proto, 1) is Msg1

    def test_register_message_on_proto_without_messages_dict(self):
        """register_message initialises _protocol_messages if absent."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        # Delete the auto-created dict to force lazy init path
        del Proto._protocol_messages

        class Msg:
            pass

        ProtocolMeta.register_message(Proto, 10, Msg)
        assert ProtocolMeta.get_message_type(Proto, 10) is Msg


# ---------------------------------------------------------------------------
# Version decoder and decode_message edge cases
# ---------------------------------------------------------------------------

class TestVersionDecoderEdgeCases:
    """Edge cases for register_version_decoder and decode_message."""

    def test_register_decoder_unregistered_protocol_raises(self):
        """register_version_decoder raises ValueError when protocol has no _protocol_id."""

        class FakeProto:
            pass

        def decoder(msg_id, data):
            return data

        with pytest.raises(ValueError, match="no _protocol_id"):
            ProtocolMeta.register_version_decoder(FakeProto, 1, decoder)

    def test_decode_message_decoder_raises_exception(self):
        """When the decoder function raises, the exception propagates."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        def broken_decoder(msg_id, data):
            raise RuntimeError("decoder failure")

        ProtocolMeta.register_version_decoder(Proto, 1, broken_decoder)

        with pytest.raises(RuntimeError, match="decoder failure"):
            ProtocolMeta.decode_message(Proto, 1, 42, b"data")

    def test_decode_message_message_type_decode_raises(self):
        """When message_type.decode raises, the exception propagates."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        class BrokenMessage:
            @staticmethod
            def decode(data):
                raise RuntimeError("decode failure")

        ProtocolMeta.register_message(Proto, 10, BrokenMessage)

        with pytest.raises(RuntimeError, match="decode failure"):
            ProtocolMeta.decode_message(Proto, 1, 10, b"data")

    def test_decode_message_message_type_no_decode(self):
        """When message type has no decode method, returns None gracefully."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        class NoDecodeMessage:
            pass

        ProtocolMeta.register_message(Proto, 10, NoDecodeMessage)

        result = ProtocolMeta.decode_message(Proto, 1, 10, b"data")
        assert result is None

    def test_decode_message_for_version_different_from_current(self):
        """Decode with a version that differs from _protocol_version uses its own decoder."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 2

        def v1_decoder(msg_id, data):
            return f"v1:{data.decode()}"

        def v2_decoder(msg_id, data):
            return f"v2:{data.decode()}"

        ProtocolMeta.register_version_decoder(Proto, 1, v1_decoder)
        ProtocolMeta.register_version_decoder(Proto, 2, v2_decoder)

        assert ProtocolMeta.decode_message(Proto, 1, 5, b"hello") == "v1:hello"
        assert ProtocolMeta.decode_message(Proto, 2, 5, b"hello") == "v2:hello"

    def test_register_duplicate_decoder_distinct_protocols(self):
        """Same version can have different decoders for different protocols."""

        class ProtoA(metaclass=ProtocolMeta):
            _protocol_version = 1

        class ProtoB(metaclass=ProtocolMeta):
            _protocol_version = 1

        def dec_a(msg_id, data):
            return "a"

        def dec_b(msg_id, data):
            return "b"

        ProtocolMeta.register_version_decoder(ProtoA, 1, dec_a)
        ProtocolMeta.register_version_decoder(ProtoB, 1, dec_b)

        assert ProtocolMeta.decode_message(ProtoA, 1, 0, b"") == "a"
        assert ProtocolMeta.decode_message(ProtoB, 1, 0, b"") == "b"

    def test_clear_registry_clears_decoders(self):
        """clear_registry removes all version decoders."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 1

        def decoder(msg_id, data):
            return "decoded"

        ProtocolMeta.register_version_decoder(Proto, 1, decoder)

        ProtocolMeta.clear_registry()

        # After clear, the protocol is no longer in the registry,
        # so decode_message will raise about no _protocol_id.
        with pytest.raises(ValueError, match="no _protocol_id"):
            ProtocolMeta.decode_message(Proto, 1, 0, b"")


# ---------------------------------------------------------------------------
# Negotiation edge cases
# ---------------------------------------------------------------------------

class TestNegotiationEdgeCases:
    """Edge cases for negotiate_version."""

    def test_negotiate_duplicate_offered_versions(self):
        """Duplicate versions in offered list are handled."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 5
            _protocol_min_version = 3

        result = ProtocolMeta.negotiate_version(Proto, [3, 4, 4, 5, 5])
        assert result == 5

    def test_negotiate_single_compatible_version(self):
        """Single-element list with a compatible version returns that version."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 5
            _protocol_min_version = 3

        result = ProtocolMeta.negotiate_version(Proto, [4])
        assert result == 4

    def test_negotiate_single_incompatible_version(self):
        """Single-element list with an incompatible version returns None."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 5
            _protocol_min_version = 3

        result = ProtocolMeta.negotiate_version(Proto, [2])
        assert result is None

        result = ProtocolMeta.negotiate_version(Proto, [6])
        assert result is None

    def test_negotiate_exact_min_and_max(self):
        """Boundary versions exactly at min and max are both compatible."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 5
            _protocol_min_version = 3

        result = ProtocolMeta.negotiate_version(Proto, [3, 5])
        assert result == 5


# ---------------------------------------------------------------------------
# Thread safety during concurrent registration
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Concurrent protocol registration does not corrupt registry state."""

    def test_concurrent_registration(self):
        """Multiple threads creating protocols do not cause data loss."""

        results: list[bool] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def create_proto(n: int) -> None:
            try:
                namespace = {}

                # Dynamically create a class with unique name
                cls = ProtocolMeta(
                    f"ConcurrentProto{n}",
                    (),
                    {"_protocol_version": 1},
                )
                with lock:
                    results.append(cls._protocol_id is not None)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=create_proto, args=(i,))
            for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert all(results)
        assert len(ProtocolMeta.all_protocols()) == 20

    def test_concurrent_registration_unique_ids(self):
        """Concurrent threads produce unique protocol IDs."""

        ids: list[int] = []
        ids_lock = threading.Lock()

        def create_and_record(n: int) -> None:
            cls = ProtocolMeta(
                f"UniqueProto{n}",
                (),
                {"_protocol_version": 1},
            )
            with ids_lock:
                ids.append(cls._protocol_id)

        threads = [
            threading.Thread(target=create_and_record, args=(i,))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(ids)) == 10  # All IDs are unique
        assert sorted(ids) == ids  # In order (lock ensures sequential IDs)


# ---------------------------------------------------------------------------
# get_migration_path edge cases
# ---------------------------------------------------------------------------

class TestMigrationPathEdgeCases:
    """Edge cases for get_migration_path."""

    def test_migration_single_step_forward(self):
        """Migration from version 1 to 2 returns [1, 2]."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 5
            _protocol_min_version = 1

        path = ProtocolMeta.get_migration_path(Proto, 1, 2)
        assert path == [1, 2]

    def test_migration_single_step_backward(self):
        """Migration from version 5 to 4 returns [5, 4]."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 5
            _protocol_min_version = 1

        path = ProtocolMeta.get_migration_path(Proto, 5, 4)
        assert path == [5, 4]

    def test_migration_full_range_forward(self):
        """Migration across the full compatible range."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 10
            _protocol_min_version = 1

        path = ProtocolMeta.get_migration_path(Proto, 1, 10)
        assert path == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert len(path) == 10

    def test_migration_full_range_backward(self):
        """Migration backward across the full compatible range."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 10
            _protocol_min_version = 1

        path = ProtocolMeta.get_migration_path(Proto, 10, 1)
        assert path == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        assert len(path) == 10

    def test_migration_from_min_above_supported_returns_empty(self):
        """from_version < min_version returns empty list."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 10
            _protocol_min_version = 5

        path = ProtocolMeta.get_migration_path(Proto, 1, 5)
        assert path == []

    def test_migration_to_above_supported_returns_empty(self):
        """to_version > max_version returns empty list."""

        class Proto(metaclass=ProtocolMeta):
            _protocol_version = 10
            _protocol_min_version = 5

        path = ProtocolMeta.get_migration_path(Proto, 10, 11)
        assert path == []
