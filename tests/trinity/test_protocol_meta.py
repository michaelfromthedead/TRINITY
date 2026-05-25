"""
Comprehensive tests for ProtocolMeta - Metaclass for network protocol definitions.

Tests cover:
- Protocol ID assignment
- Version validation (must be positive int)
- min_version validation (can't exceed version)
- is_compatible version range check
- negotiate_version (picks highest compatible)
- register_message / get_message_type
- Duplicate message ID raises ValueError
- Registry clearing
"""
import pytest

from trinity.metaclasses import ProtocolMeta


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    ProtocolMeta.clear_registry()
    yield
    ProtocolMeta.clear_registry()


def test_protocol_id_assignment():
    """Test that protocol IDs are assigned sequentially."""

    class Proto1(metaclass=ProtocolMeta):
        _protocol_version = 1

    class Proto2(metaclass=ProtocolMeta):
        _protocol_version = 1

    class Proto3(metaclass=ProtocolMeta):
        _protocol_version = 1

    assert Proto1._protocol_id == 1
    assert Proto2._protocol_id == 2
    assert Proto3._protocol_id == 3


def test_protocol_qualified_name():
    """Test that protocol qualified name includes module."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    assert "." in TestProto._protocol_qualified_name
    assert TestProto._protocol_qualified_name.endswith(".TestProto")


def test_version_required():
    """Test that _protocol_version is required."""

    with pytest.raises(TypeError, match="must define _protocol_version"):

        class NoVersion(metaclass=ProtocolMeta):
            pass


def test_version_must_be_positive_int():
    """Test that _protocol_version must be a positive integer."""

    with pytest.raises(TypeError, match="must be a positive integer"):

        class ZeroVersion(metaclass=ProtocolMeta):
            _protocol_version = 0

    with pytest.raises(TypeError, match="must be a positive integer"):

        class NegativeVersion(metaclass=ProtocolMeta):
            _protocol_version = -1

    with pytest.raises(TypeError, match="must be a positive integer"):

        class FloatVersion(metaclass=ProtocolMeta):
            _protocol_version = 1.5


def test_min_version_default():
    """Test that _protocol_min_version defaults to _protocol_version."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5

    assert TestProto._protocol_min_version == 5


def test_min_version_custom():
    """Test that _protocol_min_version can be set."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 3

    assert TestProto._protocol_min_version == 3


def test_min_version_cannot_exceed_version():
    """Test that min_version > version raises TypeError."""

    with pytest.raises(TypeError, match="cannot be greater than"):

        class BadProto(metaclass=ProtocolMeta):
            _protocol_version = 3
            _protocol_min_version = 5


def test_protocol_name_default():
    """Test that _protocol_name defaults to class name."""

    class MyProtocol(metaclass=ProtocolMeta):
        _protocol_version = 1

    assert MyProtocol._protocol_name == "MyProtocol"


def test_protocol_name_custom():
    """Test that _protocol_name can be set."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1
        _protocol_name = "CustomName"

    assert TestProto._protocol_name == "CustomName"


def test_messages_default():
    """Test that _protocol_messages defaults to empty dict."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    assert TestProto._protocol_messages == {}


def test_is_compatible_in_range():
    """Test is_compatible with version in range."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 3

    assert ProtocolMeta.is_compatible(TestProto, 3) is True
    assert ProtocolMeta.is_compatible(TestProto, 4) is True
    assert ProtocolMeta.is_compatible(TestProto, 5) is True


def test_is_compatible_out_of_range():
    """Test is_compatible with version out of range."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 3

    assert ProtocolMeta.is_compatible(TestProto, 2) is False
    assert ProtocolMeta.is_compatible(TestProto, 6) is False


def test_is_compatible_exact_version():
    """Test is_compatible when min_version == version."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 3
        # min_version defaults to version (3)

    assert ProtocolMeta.is_compatible(TestProto, 3) is True
    assert ProtocolMeta.is_compatible(TestProto, 2) is False
    assert ProtocolMeta.is_compatible(TestProto, 4) is False


def test_negotiate_version_best_match():
    """Test negotiate_version picks highest compatible version."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 3

    # Offered versions: 2, 4, 6
    # Compatible: 4
    # Should pick 4 (highest compatible)
    result = ProtocolMeta.negotiate_version(TestProto, [2, 4, 6])
    assert result == 4


def test_negotiate_version_multiple_compatible():
    """Test negotiate_version with multiple compatible versions."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 3

    # Offered: 3, 4, 5
    # All compatible, should pick 5 (highest)
    result = ProtocolMeta.negotiate_version(TestProto, [3, 4, 5])
    assert result == 5


def test_negotiate_version_no_compatible():
    """Test negotiate_version with no compatible versions."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 3

    # None are in range [3, 5]
    result = ProtocolMeta.negotiate_version(TestProto, [1, 2, 6, 7])
    assert result is None


def test_negotiate_version_empty_list():
    """Test negotiate_version with empty offered list."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5

    result = ProtocolMeta.negotiate_version(TestProto, [])
    assert result is None


def test_register_message_basic():
    """Test registering a message type."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    class TestMessage:
        pass

    ProtocolMeta.register_message(TestProto, 1, TestMessage)

    assert TestProto._protocol_messages[1] is TestMessage


def test_register_message_multiple():
    """Test registering multiple messages."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    class Msg1:
        pass

    class Msg2:
        pass

    ProtocolMeta.register_message(TestProto, 1, Msg1)
    ProtocolMeta.register_message(TestProto, 2, Msg2)

    assert TestProto._protocol_messages[1] is Msg1
    assert TestProto._protocol_messages[2] is Msg2


def test_register_message_duplicate_id():
    """Test that registering duplicate message ID raises ValueError."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    class Msg1:
        pass

    class Msg2:
        pass

    ProtocolMeta.register_message(TestProto, 1, Msg1)

    with pytest.raises(ValueError, match="already registered"):
        ProtocolMeta.register_message(TestProto, 1, Msg2)


def test_get_message_type():
    """Test retrieving message type by ID."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    class TestMessage:
        pass

    ProtocolMeta.register_message(TestProto, 42, TestMessage)

    retrieved = ProtocolMeta.get_message_type(TestProto, 42)
    assert retrieved is TestMessage


def test_get_message_type_not_found():
    """Test get_message_type with non-existent ID."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    result = ProtocolMeta.get_message_type(TestProto, 999)
    assert result is None


def test_get_by_id():
    """Test retrieving protocol by ID."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    retrieved = ProtocolMeta.get_by_id(TestProto._protocol_id)
    assert retrieved is TestProto


def test_get_by_name():
    """Test retrieving protocol by qualified name."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    retrieved = ProtocolMeta.get_by_name(TestProto._protocol_qualified_name)
    assert retrieved is TestProto


def test_all_protocols():
    """Test all_protocols returns all registered protocols."""

    class Proto1(metaclass=ProtocolMeta):
        _protocol_version = 1

    class Proto2(metaclass=ProtocolMeta):
        _protocol_version = 1

    all_protos = ProtocolMeta.all_protocols()

    assert len(all_protos) == 2
    assert Proto1 in all_protos
    assert Proto2 in all_protos


def test_clear_registry():
    """Test that clear_registry removes all protocols."""

    class Proto1(metaclass=ProtocolMeta):
        _protocol_version = 1

    class Proto2(metaclass=ProtocolMeta):
        _protocol_version = 1

    assert len(ProtocolMeta.all_protocols()) == 2

    ProtocolMeta.clear_registry()

    assert len(ProtocolMeta.all_protocols()) == 0


def test_clear_registry_resets_id():
    """Test that clear_registry resets ID counter."""

    class Proto1(metaclass=ProtocolMeta):
        _protocol_version = 1

    assert Proto1._protocol_id == 1

    ProtocolMeta.clear_registry()

    class Proto2(metaclass=ProtocolMeta):
        _protocol_version = 1

    assert Proto2._protocol_id == 1


def test_base_protocol_class_skipped():
    """Test that base Protocol class is not registered."""

    class Protocol(metaclass=ProtocolMeta):
        _protocol_version = 1

    assert len(ProtocolMeta.all_protocols()) == 0


def test_get_migration_path_forward():
    """Test get_migration_path for forward migration."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 2

    path = ProtocolMeta.get_migration_path(TestProto, 2, 5)

    assert path == [2, 3, 4, 5]


def test_get_migration_path_backward():
    """Test get_migration_path for backward migration."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 2

    path = ProtocolMeta.get_migration_path(TestProto, 5, 2)

    assert path == [5, 4, 3, 2]


def test_get_migration_path_same_version():
    """Test get_migration_path when from == to."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 2

    path = ProtocolMeta.get_migration_path(TestProto, 3, 3)

    assert path == [3]


def test_get_migration_path_out_of_range():
    """Test get_migration_path with version out of range."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 3

    # From version too low
    path = ProtocolMeta.get_migration_path(TestProto, 2, 5)
    assert path == []

    # To version too high
    path = ProtocolMeta.get_migration_path(TestProto, 3, 6)
    assert path == []


def test_version_decoder_registration():
    """Test register_version_decoder."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    def decoder(msg_id, data):
        return f"decoded_{msg_id}"

    ProtocolMeta.register_version_decoder(TestProto, 1, decoder)

    # Check it was registered (internal API)
    key = (TestProto._protocol_id, 1)
    assert key in ProtocolMeta._version_decoders


def test_version_decoder_duplicate():
    """Test that registering duplicate decoder raises ValueError."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    def decoder1(msg_id, data):
        return "decoder1"

    def decoder2(msg_id, data):
        return "decoder2"

    ProtocolMeta.register_version_decoder(TestProto, 1, decoder1)

    with pytest.raises(ValueError, match="already registered"):
        ProtocolMeta.register_version_decoder(TestProto, 1, decoder2)


def test_decode_message_with_unregistered_protocol():
    """Test decode_message with protocol that has no _protocol_id."""

    class FakeProto:
        """Fake protocol without proper registration."""

        pass

    # Should raise ValueError for unregistered protocol
    with pytest.raises(ValueError, match="has no _protocol_id"):
        ProtocolMeta.decode_message(FakeProto, 1, 1, b"data")


def test_decode_message_with_no_decoder():
    """Test decode_message when no decoder is available returns None."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    # No decoder registered, no message type registered
    result = ProtocolMeta.decode_message(TestProto, 1, 999, b"data")

    # Should return None and log warning (not raise)
    assert result is None


def test_decode_message_with_version_decoder():
    """Test decode_message uses version-specific decoder."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 2

    def custom_decoder(msg_id, data):
        return f"v2_decoded_{msg_id}_{data.decode()}"

    ProtocolMeta.register_version_decoder(TestProto, 2, custom_decoder)

    result = ProtocolMeta.decode_message(TestProto, 2, 42, b"test")

    assert result == "v2_decoded_42_test"


def test_decode_message_fallback_to_message_type():
    """Test decode_message falls back to message type decode method."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 1

    class TestMessage:
        @staticmethod
        def decode(data):
            return f"decoded_{data.decode()}"

    ProtocolMeta.register_message(TestProto, 10, TestMessage)

    result = ProtocolMeta.decode_message(TestProto, 1, 10, b"payload")

    assert result == "decoded_payload"


def test_get_migration_path_invalid_from_version():
    """Test get_migration_path with invalid from_version raises ValueError."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 3

    # Negative version
    with pytest.raises(ValueError, match="must be a positive integer"):
        ProtocolMeta.get_migration_path(TestProto, -1, 5)

    # Zero version
    with pytest.raises(ValueError, match="must be a positive integer"):
        ProtocolMeta.get_migration_path(TestProto, 0, 5)

    # Non-integer
    with pytest.raises(ValueError, match="must be a positive integer"):
        ProtocolMeta.get_migration_path(TestProto, 3.5, 5)


def test_get_migration_path_invalid_to_version():
    """Test get_migration_path with invalid to_version raises ValueError."""

    class TestProto(metaclass=ProtocolMeta):
        _protocol_version = 5
        _protocol_min_version = 3

    # Negative version
    with pytest.raises(ValueError, match="must be a positive integer"):
        ProtocolMeta.get_migration_path(TestProto, 3, -1)

    # String instead of int
    with pytest.raises(ValueError, match="must be a positive integer"):
        ProtocolMeta.get_migration_path(TestProto, 3, "5")
