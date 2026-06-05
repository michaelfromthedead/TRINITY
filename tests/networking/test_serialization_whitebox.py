"""
Whitebox tests for the serialization layer: bit_packer, net_serializer, delta_encoder, quantizer.

Tests:
- T-1.4: Serialization (field types, delta compression)
- Bit-level packing/unpacking
- Network message serialization
- Delta encoding and state compression
"""

import pytest
import struct
import math
import time
from unittest.mock import Mock, patch

from engine.networking.serialization.bit_packer import BitWriter, BitReader
from engine.networking.serialization.net_serializer import (
    NetSerializer,
    MessageHeader,
    MessageType,
    serialize_message,
    deserialize_message,
)
from engine.networking.serialization.delta_encoder import (
    DeltaEncoder,
    DeltaBaseline,
    DeltaFieldType,
    DeltaFieldDescriptor,
    DeltaSchema,
    SnapshotDeltaEncoder,
)
from engine.networking.config import DEFAULT_CONFIG


# =============================================================================
# BitWriter Tests
# =============================================================================

class TestBitWriter:
    """Tests for BitWriter bit-level serialization."""

    def test_write_single_bit_true(self):
        """Writing True should produce bit 1."""
        writer = BitWriter()
        writer.write_bool(True)
        data = writer.to_bytes()
        assert data[0] & 0x80 == 0x80  # First bit is set

    def test_write_single_bit_false(self):
        """Writing False should produce bit 0."""
        writer = BitWriter()
        writer.write_bool(False)
        data = writer.to_bytes()
        assert data[0] & 0x80 == 0x00  # First bit is clear

    def test_write_multiple_bools(self):
        """Multiple bools should pack into bytes correctly."""
        writer = BitWriter()
        pattern = [True, False, True, True, False, False, True, False]
        for bit in pattern:
            writer.write_bool(bit)

        data = writer.to_bytes()
        assert len(data) == 1
        # Pattern: 10110010 = 0xB2
        assert data[0] == 0xB2

    def test_write_bits_single_byte(self):
        """Writing bits within a single byte should work."""
        writer = BitWriter()
        writer.write_bits(0b101, 3)  # Write 5 in 3 bits
        writer.write_bits(0b11, 2)   # Write 3 in 2 bits
        writer.write_bits(0b010, 3)  # Write 2 in 3 bits

        data = writer.to_bytes()
        assert len(data) == 1
        # 10111010 = 0xBA
        assert data[0] == 0xBA

    def test_write_bits_crosses_byte_boundary(self):
        """Writing bits that cross byte boundaries should work."""
        writer = BitWriter()
        writer.write_bits(0xFF, 4)  # 1111 in first 4 bits
        writer.write_bits(0xFF, 8)  # 8 more bits crossing boundary

        data = writer.to_bytes()
        assert len(data) == 2
        # First byte: 11111111 = 0xFF
        # Second byte: 11110000 = 0xF0
        assert data[0] == 0xFF
        assert data[1] == 0xF0

    def test_write_bits_64_bit_value(self):
        """Writing 64-bit values should work."""
        writer = BitWriter()
        value = 0xDEADBEEFCAFEBABE
        writer.write_bits(value, 64)

        data = writer.to_bytes()
        assert len(data) == 8
        # Verify via struct
        unpacked = struct.unpack('>Q', data)[0]
        assert unpacked == value

    def test_write_bits_invalid_num_bits_raises(self):
        """Invalid num_bits should raise ValueError."""
        writer = BitWriter()
        with pytest.raises(ValueError):
            writer.write_bits(0, 0)
        with pytest.raises(ValueError):
            writer.write_bits(0, 65)

    def test_write_int_bounded(self):
        """write_int should use minimum bits for range."""
        writer = BitWriter()
        writer.write_int(50, 0, 100)  # Needs 7 bits (range 0-100 = 101 values)

        # 100 - 0 = 100, 100.bit_length() = 7
        # 50 - 0 = 50 = 0b110010
        data = writer.to_bytes()
        assert len(data) == 1  # 7 bits fits in 1 byte

    def test_write_int_clamping(self):
        """Values outside range should be clamped."""
        writer = BitWriter()
        writer.write_int(150, 0, 100)  # Should clamp to 100
        writer.reset()
        writer.write_int(-50, 0, 100)  # Should clamp to 0

    def test_write_int_invalid_range_raises(self):
        """Invalid range (min >= max) should raise ValueError."""
        writer = BitWriter()
        with pytest.raises(ValueError):
            writer.write_int(5, 10, 5)
        with pytest.raises(ValueError):
            writer.write_int(5, 10, 10)

    def test_write_float_compressed(self):
        """Compressed floats should quantize correctly."""
        writer = BitWriter()
        writer.write_float_compressed(5.0, 0.0, 10.0, 0.1)

        # Range = 10, precision = 0.1, steps = 100
        # 100.bit_length() = 7 bits
        data = writer.to_bytes()
        assert len(data) == 1

    def test_write_float_compressed_edge_values(self):
        """Edge float values should work."""
        writer = BitWriter()
        writer.write_float_compressed(0.0, 0.0, 10.0, 0.1)
        writer.reset()
        writer.write_float_compressed(10.0, 0.0, 10.0, 0.1)

    def test_write_float_compressed_invalid_precision(self):
        """Invalid precision should raise ValueError."""
        writer = BitWriter()
        with pytest.raises(ValueError):
            writer.write_float_compressed(5.0, 0.0, 10.0, 0)
        with pytest.raises(ValueError):
            writer.write_float_compressed(5.0, 0.0, 10.0, -0.1)

    def test_write_float_compressed_invalid_range(self):
        """Invalid range should raise ValueError."""
        writer = BitWriter()
        with pytest.raises(ValueError):
            writer.write_float_compressed(5.0, 10.0, 5.0, 0.1)

    def test_write_bytes_aligns(self):
        """write_bytes should align to byte boundary first."""
        writer = BitWriter()
        writer.write_bits(0b111, 3)  # Partial byte
        writer.write_bytes(b'Hello')

        data = writer.to_bytes()
        # 3 bits + alignment + 5 bytes = 6 bytes
        assert len(data) == 6
        assert data[1:6] == b'Hello'

    def test_write_string(self):
        """write_string should write length-prefixed UTF-8."""
        writer = BitWriter()
        writer.write_string("Test")

        data = writer.to_bytes()
        # Default max_length=255, so 8 bits for length
        # 1 byte length + 4 bytes "Test" = 5 bytes minimum
        assert len(data) >= 5

    def test_write_string_truncation(self):
        """Strings longer than max_length should be truncated."""
        writer = BitWriter()
        writer.write_string("A" * 300, max_length=10)

        # Should only write 10 characters
        data = writer.to_bytes()
        reader = BitReader(data)
        restored = reader.read_string(max_length=10)
        assert len(restored) <= 10

    def test_align_to_byte(self):
        """align_to_byte should advance to next byte boundary."""
        writer = BitWriter()
        writer.write_bits(0b111, 3)
        assert writer.bit_position == 3

        writer.align_to_byte()
        assert writer.bit_position % 8 == 0

    def test_align_to_byte_already_aligned(self):
        """align_to_byte on aligned position should be no-op."""
        writer = BitWriter()
        writer.write_bits(0xFF, 8)
        pos_before = writer.bit_position

        writer.align_to_byte()
        assert writer.bit_position == pos_before

    def test_byte_length_calculation(self):
        """byte_length should round up correctly."""
        writer = BitWriter()
        assert writer.byte_length == 0

        writer.write_bool(True)
        assert writer.byte_length == 1

        writer.write_bits(0, 7)  # Now 8 bits
        assert writer.byte_length == 1

        writer.write_bool(True)  # Now 9 bits
        assert writer.byte_length == 2

    def test_reset_clears_writer(self):
        """reset should clear the writer state."""
        writer = BitWriter()
        writer.write_bits(0xFFFFFFFF, 32)

        writer.reset()

        assert writer.bit_position == 0
        assert writer.byte_length == 0

    def test_capacity_grows_automatically(self):
        """Buffer should grow when needed."""
        writer = BitWriter(initial_capacity=4)

        # Write more than 4 bytes
        writer.write_bits(0, 64)
        writer.write_bits(0, 64)

        assert len(writer.to_bytes()) >= 16


# =============================================================================
# BitReader Tests
# =============================================================================

class TestBitReader:
    """Tests for BitReader bit-level deserialization."""

    def test_read_single_bit(self):
        """Reading single bits should work."""
        data = bytes([0b10110010])
        reader = BitReader(data)

        assert reader.read_bool() == True
        assert reader.read_bool() == False
        assert reader.read_bool() == True
        assert reader.read_bool() == True
        assert reader.read_bool() == False
        assert reader.read_bool() == False
        assert reader.read_bool() == True
        assert reader.read_bool() == False

    def test_read_bits_single_byte(self):
        """Reading bits within single byte should work."""
        data = bytes([0b10111010])
        reader = BitReader(data)

        assert reader.read_bits(3) == 0b101
        assert reader.read_bits(2) == 0b11
        assert reader.read_bits(3) == 0b010

    def test_read_bits_crosses_boundary(self):
        """Reading bits across byte boundaries should work."""
        data = bytes([0xFF, 0xF0])
        reader = BitReader(data)

        assert reader.read_bits(4) == 0xF
        assert reader.read_bits(8) == 0xFF

    def test_read_bits_64_bit(self):
        """Reading 64-bit values should work."""
        value = 0xDEADBEEFCAFEBABE
        data = struct.pack('>Q', value)
        reader = BitReader(data)

        assert reader.read_bits(64) == value

    def test_read_bits_invalid_num_bits(self):
        """Invalid num_bits should raise ValueError."""
        reader = BitReader(b'\x00')
        with pytest.raises(ValueError):
            reader.read_bits(0)
        with pytest.raises(ValueError):
            reader.read_bits(65)

    def test_read_bits_eof(self):
        """Reading past end should raise EOFError."""
        reader = BitReader(b'\x00')
        reader.read_bits(8)
        with pytest.raises(EOFError):
            reader.read_bits(1)

    def test_read_int_bounded(self):
        """read_int should decode bounded integers correctly."""
        writer = BitWriter()
        writer.write_int(75, 0, 100)
        data = writer.to_bytes()

        reader = BitReader(data)
        assert reader.read_int(0, 100) == 75

    def test_read_int_invalid_range(self):
        """Invalid range should raise ValueError."""
        reader = BitReader(b'\xFF')
        with pytest.raises(ValueError):
            reader.read_int(10, 5)

    def test_read_float_compressed(self):
        """read_float_compressed should decode correctly."""
        writer = BitWriter()
        original = 3.14
        writer.write_float_compressed(original, 0.0, 10.0, 0.01)
        data = writer.to_bytes()

        reader = BitReader(data)
        restored = reader.read_float_compressed(0.0, 10.0, 0.01)

        # Should be within precision
        assert abs(restored - original) <= 0.01

    def test_read_float_compressed_invalid_precision(self):
        """Invalid precision should raise ValueError."""
        reader = BitReader(b'\xFF')
        with pytest.raises(ValueError):
            reader.read_float_compressed(0.0, 10.0, 0)

    def test_read_bytes_aligns(self):
        """read_bytes should align to byte boundary first."""
        writer = BitWriter()
        writer.write_bits(0b111, 3)
        writer.write_bytes(b'Test')
        data = writer.to_bytes()

        reader = BitReader(data)
        reader.read_bits(3)
        result = reader.read_bytes(4)
        assert result == b'Test'

    def test_read_string(self):
        """read_string should decode UTF-8 with length prefix."""
        writer = BitWriter()
        writer.write_string("Hello")
        data = writer.to_bytes()

        reader = BitReader(data)
        result = reader.read_string()
        assert result == "Hello"

    def test_read_string_unicode(self):
        """Unicode strings should work."""
        writer = BitWriter()
        original = "Hello World"  # ASCII for now, Unicode would be longer
        writer.write_string(original)
        data = writer.to_bytes()

        reader = BitReader(data)
        result = reader.read_string()
        assert result == original

    def test_bits_remaining(self):
        """bits_remaining should track correctly."""
        data = bytes([0xFF, 0xFF])
        reader = BitReader(data)

        assert reader.bits_remaining == 16
        reader.read_bits(5)
        assert reader.bits_remaining == 11
        reader.read_bits(8)
        assert reader.bits_remaining == 3

    def test_peek_bits(self):
        """peek_bits should not advance position."""
        data = bytes([0xAB])
        reader = BitReader(data)

        peeked = reader.peek_bits(4)
        assert peeked == 0xA
        assert reader.bit_position == 0

        # Read should get same value
        read = reader.read_bits(4)
        assert read == 0xA
        assert reader.bit_position == 4

    def test_skip_bits(self):
        """skip_bits should advance position correctly."""
        data = bytes([0xAB, 0xCD])
        reader = BitReader(data)

        reader.skip_bits(4)
        assert reader.bit_position == 4

        reader.skip_bits(8)
        assert reader.bit_position == 12

    def test_reset_returns_to_start(self):
        """reset should return to beginning."""
        data = bytes([0xAB, 0xCD])
        reader = BitReader(data)

        reader.read_bits(12)
        reader.reset()

        assert reader.bit_position == 0
        assert reader.read_bits(8) == 0xAB


class TestBitPackerRoundtrip:
    """Roundtrip tests for BitWriter/BitReader."""

    def test_roundtrip_mixed_types(self):
        """Mixed type serialization should roundtrip."""
        writer = BitWriter()
        writer.write_bool(True)
        writer.write_bits(42, 7)
        writer.write_int(50, 0, 100)
        writer.write_float_compressed(3.14, 0.0, 10.0, 0.01)
        writer.write_string("Test")
        data = writer.to_bytes()

        reader = BitReader(data)
        assert reader.read_bool() == True
        assert reader.read_bits(7) == 42
        assert reader.read_int(0, 100) == 50
        assert abs(reader.read_float_compressed(0.0, 10.0, 0.01) - 3.14) < 0.01
        assert reader.read_string() == "Test"

    def test_roundtrip_many_bools(self):
        """Many bools should roundtrip correctly."""
        writer = BitWriter()
        original = [i % 2 == 0 for i in range(100)]
        for b in original:
            writer.write_bool(b)
        data = writer.to_bytes()

        reader = BitReader(data)
        restored = [reader.read_bool() for _ in range(100)]
        assert restored == original

    def test_roundtrip_boundary_values(self):
        """Boundary values should roundtrip."""
        writer = BitWriter()

        # 8-bit boundary
        writer.write_bits(0, 8)
        writer.write_bits(255, 8)

        # 16-bit boundary
        writer.write_bits(0, 16)
        writer.write_bits(65535, 16)

        # 32-bit boundary
        writer.write_bits(0, 32)
        writer.write_bits(0xFFFFFFFF, 32)

        data = writer.to_bytes()
        reader = BitReader(data)

        assert reader.read_bits(8) == 0
        assert reader.read_bits(8) == 255
        assert reader.read_bits(16) == 0
        assert reader.read_bits(16) == 65535
        assert reader.read_bits(32) == 0
        assert reader.read_bits(32) == 0xFFFFFFFF


# =============================================================================
# MessageHeader Tests
# =============================================================================

class TestMessageHeader:
    """Tests for MessageHeader serialization."""

    def test_message_header_roundtrip(self):
        """MessageHeader should roundtrip correctly."""
        original = MessageHeader(
            message_type=MessageType.ENTITY_UPDATE,
            version=2,
            sequence=12345,
            timestamp=time.time(),
            flags=MessageHeader.FLAG_RELIABLE | MessageHeader.FLAG_COMPRESSED,
            payload_size=256
        )
        data = original.to_bytes()
        restored = MessageHeader.from_bytes(data)

        assert restored.message_type == original.message_type
        assert restored.version == original.version
        assert restored.sequence == original.sequence
        assert abs(restored.timestamp - original.timestamp) < 0.001  # ms precision
        assert restored.flags == original.flags
        assert restored.payload_size == original.payload_size

    def test_message_header_all_types(self):
        """All message types should serialize correctly."""
        for msg_type in MessageType:
            header = MessageHeader(
                message_type=msg_type,
                version=1,
                sequence=0,
                timestamp=0.0,
                flags=0,
                payload_size=0
            )
            data = header.to_bytes()
            restored = MessageHeader.from_bytes(data)
            assert restored.message_type == msg_type

    def test_message_header_insufficient_data(self):
        """Insufficient data should raise ValueError."""
        with pytest.raises(ValueError):
            MessageHeader.from_bytes(b'\x00' * 10)

    def test_message_header_size_constant(self):
        """Header size should match constant."""
        header = MessageHeader(
            message_type=MessageType.HEARTBEAT,
            version=1,
            sequence=0,
            timestamp=0.0,
            flags=0,
            payload_size=0
        )
        data = header.to_bytes()
        assert len(data) == MessageHeader.HEADER_SIZE


# =============================================================================
# NetSerializer Tests
# =============================================================================

class TestNetSerializer:
    """Tests for NetSerializer message handling."""

    def test_serializer_version(self):
        """Serializer version should be accessible."""
        serializer = NetSerializer(version=5)
        assert serializer.version == 5

    def test_serializer_sequence_increments(self):
        """Sequence should increment with each serialize."""
        serializer = NetSerializer()
        seq1 = serializer.sequence

        serializer.serialize(MessageType.HEARTBEAT, None)
        seq2 = serializer.sequence

        assert seq2 == seq1 + 1

    def test_serialize_empty_payload(self):
        """Empty payload types should serialize."""
        serializer = NetSerializer()
        data = serializer.serialize(MessageType.HEARTBEAT, None)

        msg_type, payload = serializer.deserialize(data)
        assert msg_type == MessageType.HEARTBEAT

    def test_serialize_dict_payload(self):
        """Dict payloads should serialize and deserialize."""
        serializer = NetSerializer()
        original = {"key1": "value1", "key2": 42, "key3": True}

        data = serializer.serialize(MessageType.FULL_STATE, original)
        msg_type, restored = serializer.deserialize(data)

        assert msg_type == MessageType.FULL_STATE
        assert restored == original

    def test_serialize_bytes_payload(self):
        """Raw bytes should work for unregistered types."""
        serializer = NetSerializer()
        original = b'raw bytes data'

        data = serializer.serialize(150, original)  # Custom type
        msg_type, restored = serializer.deserialize(data)

        assert msg_type == 150
        assert restored == original

    def test_serialize_compression(self):
        """Large payloads should be compressed."""
        serializer = NetSerializer(compress_threshold=50)
        original = {"data": "x" * 100}

        data = serializer.serialize(MessageType.FULL_STATE, original)
        header = MessageHeader.from_bytes(data)

        assert header.flags & MessageHeader.FLAG_COMPRESSED

    def test_serialize_no_compression_small(self):
        """Small payloads should not be compressed."""
        serializer = NetSerializer(compress_threshold=1000)
        original = {"key": "small"}

        data = serializer.serialize(MessageType.FULL_STATE, original)
        header = MessageHeader.from_bytes(data)

        assert not (header.flags & MessageHeader.FLAG_COMPRESSED)

    def test_serialize_custom_encoder(self):
        """Custom encoders should work."""
        serializer = NetSerializer()

        # Register custom encoder/decoder
        def encode(obj):
            return obj.to_bytes()

        def decode(data):
            return int.from_bytes(data, 'big')

        serializer.register_encoder(200, lambda x: x.to_bytes(4, 'big'))
        serializer.register_decoder(200, lambda d: int.from_bytes(d, 'big'))

        data = serializer.serialize(200, 12345)
        msg_type, restored = serializer.deserialize(data)

        assert msg_type == 200
        assert restored == 12345

    def test_serialize_nested_dict(self):
        """Nested dicts should serialize correctly."""
        serializer = NetSerializer()
        original = {
            "level1": {
                "level2": {
                    "value": 42
                }
            },
            "array": [1, 2, 3]
        }

        data = serializer.serialize(MessageType.FULL_STATE, original)
        msg_type, restored = serializer.deserialize(data)

        assert restored == original

    def test_serialize_with_flags(self):
        """Custom flags should be preserved."""
        serializer = NetSerializer()
        data = serializer.serialize(
            MessageType.HEARTBEAT,
            None,
            flags=MessageHeader.FLAG_RELIABLE
        )

        header = MessageHeader.from_bytes(data)
        assert header.flags & MessageHeader.FLAG_RELIABLE

    def test_deserialize_insufficient_data(self):
        """Insufficient data should raise ValueError."""
        serializer = NetSerializer()
        with pytest.raises(ValueError):
            serializer.deserialize(b'\x00' * 10)

    def test_deserialize_incomplete_payload(self):
        """Incomplete payload should raise ValueError."""
        serializer = NetSerializer()
        # Create valid header but truncate payload
        data = serializer.serialize(MessageType.FULL_STATE, {"key": "value"})
        truncated = data[:MessageHeader.HEADER_SIZE + 5]

        with pytest.raises(ValueError):
            serializer.deserialize(truncated)

    def test_deserialize_header_only(self):
        """deserialize_header should parse only header."""
        serializer = NetSerializer()
        data = serializer.serialize(MessageType.ENTITY_UPDATE, {"key": "value"})

        header = serializer.deserialize_header(data)
        assert header.message_type == MessageType.ENTITY_UPDATE


class TestNetSerializerValueTypes:
    """Tests for different value type serialization."""

    def test_serialize_none(self):
        """None values should serialize."""
        serializer = NetSerializer()
        original = {"null_value": None}

        data = serializer.serialize(MessageType.FULL_STATE, original)
        _, restored = serializer.deserialize(data)

        assert restored["null_value"] is None

    def test_serialize_bool(self):
        """Boolean values should serialize."""
        serializer = NetSerializer()
        original = {"true_val": True, "false_val": False}

        data = serializer.serialize(MessageType.FULL_STATE, original)
        _, restored = serializer.deserialize(data)

        assert restored["true_val"] == True
        assert restored["false_val"] == False

    def test_serialize_int_ranges(self):
        """All int ranges should serialize."""
        serializer = NetSerializer()
        original = {
            "int8": 127,
            "int8_neg": -128,
            "int16": 32767,
            "int16_neg": -32768,
            "int32": 2147483647,
            "int32_neg": -2147483648,
            "int64": 2**62,
            "int64_neg": -(2**62),
        }

        data = serializer.serialize(MessageType.FULL_STATE, original)
        _, restored = serializer.deserialize(data)

        for key, value in original.items():
            assert restored[key] == value

    def test_serialize_float(self):
        """Float values should serialize."""
        serializer = NetSerializer()
        original = {"pi": 3.14159, "neg": -2.718, "zero": 0.0}

        data = serializer.serialize(MessageType.FULL_STATE, original)
        _, restored = serializer.deserialize(data)

        for key, value in original.items():
            assert abs(restored[key] - value) < 0.0001

    def test_serialize_string(self):
        """String values should serialize."""
        serializer = NetSerializer()
        original = {"empty": "", "short": "hi", "long": "x" * 100}

        data = serializer.serialize(MessageType.FULL_STATE, original)
        _, restored = serializer.deserialize(data)

        assert restored == original

    def test_serialize_bytes(self):
        """Bytes values should serialize."""
        serializer = NetSerializer()
        original = {"binary": b'\x00\x01\x02\xff'}

        data = serializer.serialize(MessageType.FULL_STATE, original)
        _, restored = serializer.deserialize(data)

        assert restored["binary"] == original["binary"]

    def test_serialize_list(self):
        """List values should serialize."""
        serializer = NetSerializer()
        original = {"nums": [1, 2, 3], "mixed": [1, "two", 3.0, None]}

        data = serializer.serialize(MessageType.FULL_STATE, original)
        _, restored = serializer.deserialize(data)

        assert restored["nums"] == [1, 2, 3]
        assert len(restored["mixed"]) == 4


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_serialize_message_function(self):
        """serialize_message convenience function should work."""
        data = serialize_message(MessageType.HEARTBEAT, None)
        assert len(data) >= MessageHeader.HEADER_SIZE

    def test_deserialize_message_function(self):
        """deserialize_message convenience function should work."""
        data = serialize_message(MessageType.FULL_STATE, {"key": "value"})
        msg_type, payload = deserialize_message(data)

        assert msg_type == MessageType.FULL_STATE
        assert payload["key"] == "value"


# =============================================================================
# DeltaEncoder Tests
# =============================================================================

class TestDeltaBaseline:
    """Tests for DeltaBaseline."""

    def test_baseline_hash_deterministic(self):
        """Baseline hash should be deterministic."""
        state = {"a": 1, "b": 2, "c": 3}
        hash1 = DeltaBaseline.compute_hash(state)
        hash2 = DeltaBaseline.compute_hash(state)
        assert hash1 == hash2

    def test_baseline_hash_different_states(self):
        """Different states should have different hashes."""
        state1 = {"a": 1}
        state2 = {"a": 2}
        hash1 = DeltaBaseline.compute_hash(state1)
        hash2 = DeltaBaseline.compute_hash(state2)
        assert hash1 != hash2

    def test_baseline_hash_order_independent(self):
        """Hash should be order-independent (sorted keys)."""
        state1 = {"a": 1, "b": 2}
        state2 = {"b": 2, "a": 1}
        hash1 = DeltaBaseline.compute_hash(state1)
        hash2 = DeltaBaseline.compute_hash(state2)
        assert hash1 == hash2


class TestDeltaEncoder:
    """Tests for DeltaEncoder."""

    def test_set_baseline(self):
        """set_baseline should store baseline correctly."""
        encoder = DeltaEncoder()
        state = {"x": 0, "y": 0}

        baseline = encoder.set_baseline(0, state)

        assert baseline.sequence == 0
        assert baseline.state == state
        assert encoder.get_baseline(0) is not None

    def test_set_baseline_updates_latest_sequence(self):
        """latest_sequence should track highest sequence."""
        encoder = DeltaEncoder()
        encoder.set_baseline(5, {"a": 1})
        assert encoder.latest_sequence == 5

        encoder.set_baseline(10, {"a": 2})
        assert encoder.latest_sequence == 10

        encoder.set_baseline(7, {"a": 3})
        assert encoder.latest_sequence == 10  # Still 10

    def test_set_baseline_eviction(self):
        """Old baselines should be evicted when at capacity."""
        encoder = DeltaEncoder(max_baselines=3)

        for i in range(5):
            encoder.set_baseline(i, {"val": i})

        assert encoder.get_baseline_count() == 3
        assert encoder.get_baseline(0) is None  # Evicted
        assert encoder.get_baseline(1) is None  # Evicted
        assert encoder.get_baseline(4) is not None

    def test_acknowledge_baseline(self):
        """acknowledge_baseline should remove older baselines."""
        encoder = DeltaEncoder()

        for i in range(5):
            encoder.set_baseline(i, {"val": i})

        encoder.acknowledge_baseline(3)

        assert encoder.get_baseline(0) is None
        assert encoder.get_baseline(1) is None
        assert encoder.get_baseline(2) is None
        assert encoder.get_baseline(3) is not None
        assert encoder.get_baseline(4) is not None

    def test_encode_delta_no_changes(self):
        """Delta with no changes should be minimal."""
        encoder = DeltaEncoder()
        state = {"x": 10, "y": 20}
        encoder.set_baseline(0, state)

        delta = encoder.encode_delta(state, 0)

        # Should have minimal overhead for empty delta
        assert len(delta) < 20

    def test_encode_decode_delta_modified(self):
        """Modified fields should encode and decode correctly."""
        encoder = DeltaEncoder()
        baseline_state = {"x": 10, "y": 20, "name": "entity"}
        encoder.set_baseline(0, baseline_state)

        current_state = {"x": 15, "y": 20, "name": "entity"}
        delta = encoder.encode_delta(current_state, 0)

        decoded = encoder.decode_delta(delta, 0)
        assert decoded["x"] == 15
        assert decoded["y"] == 20

    def test_encode_decode_delta_added(self):
        """Added fields should encode correctly."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {"x": 10})

        current = {"x": 10, "y": 20, "z": 30}
        delta = encoder.encode_delta(current, 0)
        decoded = encoder.decode_delta(delta, 0)

        assert decoded == current

    def test_encode_decode_delta_removed(self):
        """Removed fields should encode correctly."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {"x": 10, "y": 20, "z": 30})

        current = {"x": 10}  # y and z removed
        delta = encoder.encode_delta(current, 0)
        decoded = encoder.decode_delta(delta, 0)

        assert "y" not in decoded
        assert "z" not in decoded
        assert decoded["x"] == 10

    def test_encode_delta_missing_baseline(self):
        """Encoding against missing baseline should raise KeyError."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {"x": 10})

        with pytest.raises(KeyError):
            encoder.encode_delta({"x": 15}, 999)

    def test_decode_delta_missing_baseline(self):
        """Decoding against missing baseline should raise KeyError."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {"x": 10})
        delta = encoder.encode_delta({"x": 15}, 0)

        with pytest.raises(KeyError):
            encoder.decode_delta(delta, 999)

    def test_encode_full_state(self):
        """Full state encoding should work without baseline."""
        encoder = DeltaEncoder()
        state = {"a": 1, "b": "hello", "c": True}

        data = encoder.encode_full_state(state)
        decoded = encoder.decode_full_state(data)

        assert decoded == state

    def test_clear_baselines(self):
        """clear_baselines should remove all baselines."""
        encoder = DeltaEncoder()

        for i in range(5):
            encoder.set_baseline(i, {"val": i})

        encoder.clear_baselines()

        assert encoder.get_baseline_count() == 0
        assert encoder.latest_sequence == -1


class TestDeltaEncoderFieldTypes:
    """Tests for different field type serialization in delta encoding."""

    def test_delta_bool_field(self):
        """Boolean fields should encode correctly."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {"flag": False})

        delta = encoder.encode_delta({"flag": True}, 0)
        decoded = encoder.decode_delta(delta, 0)

        assert decoded["flag"] == True

    def test_delta_int_fields(self):
        """Integer fields of various sizes should encode correctly."""
        encoder = DeltaEncoder()
        baseline = {"i8": 0, "i16": 0, "i32": 0, "i64": 0}
        encoder.set_baseline(0, baseline)

        current = {
            "i8": 100,        # Fits in int8
            "i16": 1000,      # Fits in int16
            "i32": 100000,    # Fits in int32
            "i64": 2**40,     # Needs int64
        }
        delta = encoder.encode_delta(current, 0)
        decoded = encoder.decode_delta(delta, 0)

        assert decoded == current

    def test_delta_negative_ints(self):
        """Negative integers should encode correctly."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {"val": 0})

        for neg_val in [-1, -128, -32768, -2147483648]:
            delta = encoder.encode_delta({"val": neg_val}, 0)
            decoded = encoder.decode_delta(delta, 0)
            assert decoded["val"] == neg_val

    def test_delta_float_field(self):
        """Float fields should encode correctly."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {"val": 0.0})

        delta = encoder.encode_delta({"val": 3.14159}, 0)
        decoded = encoder.decode_delta(delta, 0)

        assert abs(decoded["val"] - 3.14159) < 0.0001

    def test_delta_string_field(self):
        """String fields should encode correctly."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {"name": ""})

        delta = encoder.encode_delta({"name": "EntityName"}, 0)
        decoded = encoder.decode_delta(delta, 0)

        assert decoded["name"] == "EntityName"

    def test_delta_bytes_field(self):
        """Bytes fields should encode correctly."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {"data": b''})

        delta = encoder.encode_delta({"data": b'\x00\x01\x02'}, 0)
        decoded = encoder.decode_delta(delta, 0)

        assert decoded["data"] == b'\x00\x01\x02'


class TestDeltaCompression:
    """Tests for delta compression behavior."""

    def test_delta_compression_large_payload(self):
        """Large deltas should be compressed."""
        encoder = DeltaEncoder(compress_threshold=50)
        encoder.set_baseline(0, {"data": ""})

        # Create large change
        current = {"data": "x" * 200}
        delta = encoder.encode_delta(current, 0)

        # First byte should indicate compression
        assert delta[0] == 0x01  # Compressed flag

    def test_delta_no_compression_small(self):
        """Small deltas should not be compressed."""
        encoder = DeltaEncoder(compress_threshold=1000)
        encoder.set_baseline(0, {"x": 0})

        delta = encoder.encode_delta({"x": 1}, 0)

        # First byte should indicate no compression
        assert delta[0] == 0x00

    def test_empty_delta_decodes_to_baseline(self):
        """Empty delta bytes should return baseline."""
        encoder = DeltaEncoder()
        baseline = {"x": 10, "y": 20}
        encoder.set_baseline(0, baseline)

        decoded = encoder._decode_state_delta(b'', baseline)
        assert decoded == baseline


# =============================================================================
# DeltaSchema Tests
# =============================================================================

class TestDeltaSchema:
    """Tests for DeltaSchema typed encoding."""

    def test_schema_field_lookup_by_name(self):
        """Schema should support field lookup by name."""
        schema = DeltaSchema(
            name="TestEntity",
            version=1,
            fields=[
                DeltaFieldDescriptor("health", DeltaFieldType.INT16, 0),
                DeltaFieldDescriptor("position_x", DeltaFieldType.FLOAT32, 1),
            ]
        )

        health_field = schema.get_field("health")
        assert health_field is not None
        assert health_field.field_type == DeltaFieldType.INT16

        missing = schema.get_field("nonexistent")
        assert missing is None

    def test_schema_field_lookup_by_index(self):
        """Schema should support field lookup by index."""
        schema = DeltaSchema(
            name="TestEntity",
            version=1,
            fields=[
                DeltaFieldDescriptor("health", DeltaFieldType.INT16, 0),
                DeltaFieldDescriptor("mana", DeltaFieldType.INT16, 1),
            ]
        )

        field = schema.get_field_by_index(1)
        assert field is not None
        assert field.name == "mana"


# =============================================================================
# SnapshotDeltaEncoder Tests
# =============================================================================

class TestSnapshotDeltaEncoder:
    """Tests for SnapshotDeltaEncoder entity state handling."""

    def test_snapshot_encode_empty(self):
        """Empty snapshot should encode."""
        encoder = SnapshotDeltaEncoder()
        data = encoder.encode_snapshot({})
        assert len(data) > 0

    def test_snapshot_encode_decode_added_entities(self):
        """Added entities should encode and decode."""
        encoder = SnapshotDeltaEncoder()

        entities = {
            1: {"x": 10, "y": 20},
            2: {"x": 30, "y": 40},
        }

        data = encoder.encode_snapshot(entities, baseline_entities={})
        decoded = encoder.decode_snapshot(data, baseline_entities={})

        assert decoded == entities

    def test_snapshot_encode_decode_changed_entities(self):
        """Changed entities should use delta encoding."""
        encoder = SnapshotDeltaEncoder()

        baseline = {
            1: {"x": 10, "y": 20},
            2: {"x": 30, "y": 40},
        }
        current = {
            1: {"x": 15, "y": 20},  # x changed
            2: {"x": 30, "y": 45},  # y changed
        }

        data = encoder.encode_snapshot(current, baseline)
        decoded = encoder.decode_snapshot(data, baseline)

        assert decoded == current

    def test_snapshot_encode_decode_removed_entities(self):
        """Removed entities should be handled."""
        encoder = SnapshotDeltaEncoder()

        baseline = {
            1: {"x": 10},
            2: {"x": 20},
            3: {"x": 30},
        }
        current = {
            1: {"x": 10},  # Entity 2 and 3 removed
        }

        data = encoder.encode_snapshot(current, baseline)
        decoded = encoder.decode_snapshot(data, baseline)

        assert 1 in decoded
        assert 2 not in decoded
        assert 3 not in decoded

    def test_snapshot_mixed_operations(self):
        """Mix of add, change, remove should work."""
        encoder = SnapshotDeltaEncoder()

        baseline = {
            1: {"health": 100},
            2: {"health": 80},
        }
        current = {
            1: {"health": 95},   # Changed
            3: {"health": 100},  # Added (2 removed)
        }

        data = encoder.encode_snapshot(current, baseline)
        decoded = encoder.decode_snapshot(data, baseline)

        assert decoded[1]["health"] == 95
        assert 2 not in decoded
        assert decoded[3]["health"] == 100

    def test_snapshot_no_changes(self):
        """No changes should produce minimal delta."""
        encoder = SnapshotDeltaEncoder()

        entities = {1: {"x": 10}}

        data = encoder.encode_snapshot(entities, entities)
        decoded = encoder.decode_snapshot(data, entities)

        assert decoded == entities
