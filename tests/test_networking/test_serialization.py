"""
Tests for the networking serialization module.

Tests bit packing, quantization, delta compression, and message serialization.
"""

import math
import pytest
import struct

from engine.networking.serialization.bit_packer import BitWriter, BitReader
from engine.networking.serialization.quantizer import (
    quantize_float,
    dequantize_float,
    quantize_vector3,
    dequantize_vector3,
    quantize_quaternion,
    dequantize_quaternion,
    quantize_angle,
    dequantize_angle,
    Vector3,
    Quaternion,
)
from engine.networking.serialization.delta_encoder import (
    DeltaEncoder,
    SnapshotDeltaEncoder,
    DeltaBaseline,
)
from engine.networking.serialization.net_serializer import (
    NetSerializer,
    MessageType,
    MessageHeader,
    serialize_message,
    deserialize_message,
)
from engine.networking.config import DEFAULT_CONFIG


class TestBitWriter:
    """Tests for BitWriter class."""

    def test_write_single_bit(self):
        """Test writing a single bit."""
        writer = BitWriter()
        writer.write_bits(1, 1)
        data = writer.to_bytes()
        assert len(data) == 1
        assert data[0] == 0x80  # 1 in MSB

    def test_write_multiple_bits(self):
        """Test writing multiple bits."""
        writer = BitWriter()
        writer.write_bits(0b10110, 5)
        data = writer.to_bytes()
        assert len(data) == 1
        assert data[0] == 0b10110000

    def test_write_bool(self):
        """Test writing boolean values."""
        writer = BitWriter()
        writer.write_bool(True)
        writer.write_bool(False)
        writer.write_bool(True)
        data = writer.to_bytes()
        assert data[0] == 0b10100000

    def test_write_int_bounded(self):
        """Test writing bounded integers."""
        writer = BitWriter()
        writer.write_int(50, 0, 100)
        # Range 0-100 needs 7 bits
        assert writer.bit_position == 7

    def test_write_float_compressed(self):
        """Test writing compressed floats."""
        writer = BitWriter()
        writer.write_float_compressed(0.5, 0.0, 1.0, 0.01)
        data = writer.to_bytes()
        assert len(data) >= 1

    def test_roundtrip_bits(self):
        """Test roundtrip of bit values."""
        writer = BitWriter()
        values = [1, 0, 1, 1, 0, 1, 0, 1]
        for v in values:
            writer.write_bits(v, 1)

        reader = BitReader(writer.to_bytes())
        for expected in values:
            assert reader.read_bits(1) == expected

    def test_byte_alignment(self):
        """Test byte alignment."""
        writer = BitWriter()
        writer.write_bits(1, 1)
        writer.align_to_byte()
        writer.write_bits(0xFF, 8)
        data = writer.to_bytes()
        assert len(data) == 2
        assert data[1] == 0xFF

    def test_write_string(self):
        """Test writing strings."""
        writer = BitWriter()
        writer.write_string("Hello")
        reader = BitReader(writer.to_bytes())
        assert reader.read_string() == "Hello"

    def test_write_bytes(self):
        """Test writing raw bytes."""
        writer = BitWriter()
        writer.write_bytes(b'\x12\x34\x56')
        reader = BitReader(writer.to_bytes())
        assert reader.read_bytes(3) == b'\x12\x34\x56'

    def test_reset(self):
        """Test reset functionality."""
        writer = BitWriter()
        writer.write_bits(0xFF, 8)
        writer.reset()
        assert writer.bit_position == 0
        assert writer.byte_length == 0


class TestBitReader:
    """Tests for BitReader class."""

    def test_read_single_bit(self):
        """Test reading a single bit."""
        reader = BitReader(bytes([0x80]))
        assert reader.read_bits(1) == 1

    def test_read_bool(self):
        """Test reading boolean values."""
        reader = BitReader(bytes([0b10100000]))
        assert reader.read_bool() is True
        assert reader.read_bool() is False
        assert reader.read_bool() is True

    def test_read_int_bounded(self):
        """Test reading bounded integers."""
        writer = BitWriter()
        writer.write_int(75, 0, 100)
        reader = BitReader(writer.to_bytes())
        assert reader.read_int(0, 100) == 75

    def test_read_float_compressed(self):
        """Test reading compressed floats."""
        writer = BitWriter()
        original = 0.75
        writer.write_float_compressed(original, 0.0, 1.0, 0.01)
        reader = BitReader(writer.to_bytes())
        result = reader.read_float_compressed(0.0, 1.0, 0.01)
        assert abs(result - original) < 0.02  # Within precision

    def test_bits_remaining(self):
        """Test bits remaining calculation."""
        reader = BitReader(bytes([0xFF, 0xFF]))
        assert reader.bits_remaining == 16
        reader.read_bits(5)
        assert reader.bits_remaining == 11

    def test_peek_bits(self):
        """Test peeking without advancing."""
        reader = BitReader(bytes([0xAB]))
        peeked = reader.peek_bits(4)
        assert peeked == 0xA
        # Position should not change
        assert reader.bit_position == 0

    def test_skip_bits(self):
        """Test skipping bits."""
        reader = BitReader(bytes([0xAB, 0xCD]))
        reader.skip_bits(4)
        assert reader.read_bits(4) == 0xB

    def test_eof_error(self):
        """Test EOF error on reading past end."""
        reader = BitReader(bytes([0x00]))
        with pytest.raises(EOFError):
            reader.read_bits(16)


class TestQuantizer:
    """Tests for quantization functions."""

    def test_quantize_float_basic(self):
        """Test basic float quantization."""
        quantized = quantize_float(0.5, 0.0, 1.0, 8)
        assert 0 <= quantized <= 255

    def test_dequantize_float_basic(self):
        """Test basic float dequantization."""
        result = dequantize_float(128, 0.0, 1.0, 8)
        assert abs(result - 0.5) < 0.01

    def test_quantize_roundtrip(self):
        """Test quantize/dequantize roundtrip."""
        original = 3.14159
        min_val, max_val = 0.0, 10.0
        bits = 16

        quantized = quantize_float(original, min_val, max_val, bits)
        result = dequantize_float(quantized, min_val, max_val, bits)

        # Should be within precision
        precision = (max_val - min_val) / (2**bits - 1)
        assert abs(result - original) <= precision * 2

    def test_quantize_clamps_value(self):
        """Test that quantization clamps out-of-range values."""
        # Value below minimum
        quantized = quantize_float(-5.0, 0.0, 1.0, 8)
        assert quantized == 0

        # Value above maximum
        quantized = quantize_float(5.0, 0.0, 1.0, 8)
        assert quantized == 255

    def test_quantize_vector3_8bit(self):
        """Test 8-bit vector quantization."""
        vec = Vector3(10.5, -20.0, 5.25)
        data = quantize_vector3(vec, precision=8)
        assert len(data) == 3

        result = dequantize_vector3(data, precision=8)
        assert abs(result.x - vec.x) < 10  # Lower precision
        assert abs(result.y - vec.y) < 10
        assert abs(result.z - vec.z) < 10

    def test_quantize_vector3_16bit(self):
        """Test 16-bit vector quantization."""
        vec = (100.0, -50.0, 25.0)
        data = quantize_vector3(vec, precision=16)
        assert len(data) == 6

        result = dequantize_vector3(data, precision=16)
        assert abs(result.x - vec[0]) < 0.1
        assert abs(result.y - vec[1]) < 0.1
        assert abs(result.z - vec[2]) < 0.1

    def test_quantize_quaternion_identity(self):
        """Test quaternion quantization with identity rotation."""
        quat = Quaternion(0.0, 0.0, 0.0, 1.0)
        data = quantize_quaternion(quat)
        assert len(data) == 4

        result = dequantize_quaternion(data)
        assert abs(result.w - 1.0) < 0.01
        assert abs(result.x) < 0.01
        assert abs(result.y) < 0.01
        assert abs(result.z) < 0.01

    def test_quantize_quaternion_rotation(self):
        """Test quaternion quantization with actual rotation."""
        # 90 degree rotation around Y axis
        angle = math.pi / 2
        quat = Quaternion(0.0, math.sin(angle/2), 0.0, math.cos(angle/2))
        data = quantize_quaternion(quat)

        result = dequantize_quaternion(data)
        # Dot product should be close to 1 for similar quaternions
        dot = (quat.x * result.x + quat.y * result.y +
               quat.z * result.z + quat.w * result.w)
        assert abs(abs(dot) - 1.0) < 0.01

    def test_quantize_angle(self):
        """Test angle quantization."""
        original = math.pi / 4  # 45 degrees
        quantized = quantize_angle(original, bits=8)
        result = dequantize_angle(quantized, bits=8)
        assert abs(result - original) < (2 * math.pi / 256)

    def test_invalid_bits_raises(self):
        """Test that invalid bit counts raise errors."""
        with pytest.raises(ValueError):
            quantize_float(0.5, 0.0, 1.0, 0)
        with pytest.raises(ValueError):
            quantize_float(0.5, 0.0, 1.0, 33)


class TestDeltaEncoder:
    """Tests for DeltaEncoder class."""

    def test_set_baseline(self):
        """Test setting a baseline."""
        encoder = DeltaEncoder()
        state = {'x': 0.0, 'y': 0.0, 'health': 100}
        baseline = encoder.set_baseline(0, state)

        assert baseline.sequence == 0
        assert baseline.state == state

    def test_encode_decode_delta(self):
        """Test encoding and decoding a delta."""
        encoder = DeltaEncoder()
        baseline = {'x': 0.0, 'y': 0.0, 'health': 100}
        current = {'x': 10.5, 'y': 0.0, 'health': 95}

        encoder.set_baseline(0, baseline)
        delta = encoder.encode_delta(current, baseline_seq=0)

        # Decode
        result = encoder.decode_delta(delta, baseline_seq=0)
        assert result['x'] == current['x']
        assert result['y'] == current['y']
        assert result['health'] == current['health']

    def test_delta_with_new_field(self):
        """Test delta with a new field added."""
        encoder = DeltaEncoder()
        baseline = {'x': 0.0}
        current = {'x': 0.0, 'y': 10.0}

        encoder.set_baseline(0, baseline)
        delta = encoder.encode_delta(current, baseline_seq=0)
        result = encoder.decode_delta(delta, baseline_seq=0)

        assert 'y' in result
        assert result['y'] == 10.0

    def test_delta_with_removed_field(self):
        """Test delta with a field removed."""
        encoder = DeltaEncoder()
        baseline = {'x': 0.0, 'y': 10.0}
        current = {'x': 0.0}

        encoder.set_baseline(0, baseline)
        delta = encoder.encode_delta(current, baseline_seq=0)
        result = encoder.decode_delta(delta, baseline_seq=0)

        assert 'y' not in result

    def test_multiple_baselines(self):
        """Test handling multiple baselines."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {'x': 0.0})
        encoder.set_baseline(1, {'x': 5.0})
        encoder.set_baseline(2, {'x': 10.0})

        assert encoder.get_baseline_count() == 3
        assert encoder.get_baseline(1).state['x'] == 5.0

    def test_acknowledge_baseline(self):
        """Test acknowledging baselines removes old ones."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {'x': 0.0})
        encoder.set_baseline(1, {'x': 5.0})
        encoder.set_baseline(2, {'x': 10.0})

        encoder.acknowledge_baseline(2)

        assert encoder.get_baseline(0) is None
        assert encoder.get_baseline(1) is None
        assert encoder.get_baseline(2) is not None

    def test_baseline_hash(self):
        """Test baseline hash computation."""
        state = {'a': 1, 'b': 2}
        hash1 = DeltaBaseline.compute_hash(state)
        hash2 = DeltaBaseline.compute_hash(state)
        hash3 = DeltaBaseline.compute_hash({'a': 1, 'b': 3})

        assert hash1 == hash2
        assert hash1 != hash3

    def test_full_state_encoding(self):
        """Test encoding full state without baseline."""
        encoder = DeltaEncoder()
        state = {'x': 10.0, 'name': 'test', 'active': True}

        data = encoder.encode_full_state(state)
        result = encoder.decode_full_state(data)

        assert result == state

    def test_compression_for_large_delta(self):
        """Test that large deltas get compressed."""
        encoder = DeltaEncoder(compress_threshold=10)
        baseline = {}
        current = {f'field_{i}': f'value_{i}' * 10 for i in range(50)}

        encoder.set_baseline(0, baseline)
        delta = encoder.encode_delta(current, baseline_seq=0)

        # First byte is compression flag
        assert delta[0] == 0x01  # Compressed

    def test_various_data_types(self):
        """Test encoding various data types."""
        encoder = DeltaEncoder()
        baseline = {}
        current = {
            'int_small': 42,
            'int_large': 100000,
            'float_val': 3.14159,
            'bool_val': True,
            'string_val': 'hello',
            'bytes_val': b'\x00\x01\x02',
        }

        encoder.set_baseline(0, baseline)
        delta = encoder.encode_delta(current, baseline_seq=0)
        result = encoder.decode_delta(delta, baseline_seq=0)

        assert result['int_small'] == 42
        assert result['int_large'] == 100000
        assert abs(result['float_val'] - 3.14159) < 0.0001
        assert result['bool_val'] is True
        assert result['string_val'] == 'hello'
        assert result['bytes_val'] == b'\x00\x01\x02'


class TestSnapshotDeltaEncoder:
    """Tests for SnapshotDeltaEncoder class."""

    def test_encode_snapshot_added_entities(self):
        """Test encoding snapshot with new entities."""
        encoder = SnapshotDeltaEncoder()
        baseline = {}
        current = {
            1: {'x': 10.0, 'y': 20.0},
            2: {'x': 30.0, 'y': 40.0},
        }

        data = encoder.encode_snapshot(current, baseline)
        result = encoder.decode_snapshot(data, baseline)

        assert len(result) == 2
        assert result[1]['x'] == 10.0
        assert result[2]['y'] == 40.0

    def test_encode_snapshot_removed_entities(self):
        """Test encoding snapshot with removed entities."""
        encoder = SnapshotDeltaEncoder()
        baseline = {
            1: {'x': 10.0},
            2: {'x': 20.0},
        }
        current = {
            1: {'x': 10.0},
        }

        data = encoder.encode_snapshot(current, baseline)
        result = encoder.decode_snapshot(data, baseline)

        assert 1 in result
        assert 2 not in result

    def test_encode_snapshot_changed_entities(self):
        """Test encoding snapshot with changed entities."""
        encoder = SnapshotDeltaEncoder()
        baseline = {
            1: {'x': 10.0, 'y': 20.0},
        }
        current = {
            1: {'x': 15.0, 'y': 20.0},
        }

        data = encoder.encode_snapshot(current, baseline)
        result = encoder.decode_snapshot(data, baseline)

        assert result[1]['x'] == 15.0
        assert result[1]['y'] == 20.0


class TestNetSerializer:
    """Tests for NetSerializer class."""

    def test_serialize_deserialize_dict(self):
        """Test serializing dictionary payloads."""
        serializer = NetSerializer()
        payload = {'x': 10, 'y': 20, 'name': 'test'}

        data = serializer.serialize(MessageType.FULL_STATE, payload)
        msg_type, result = serializer.deserialize(data)

        assert msg_type == MessageType.FULL_STATE
        assert result == payload

    def test_serialize_heartbeat(self):
        """Test serializing heartbeat messages."""
        serializer = NetSerializer()

        data = serializer.serialize(MessageType.HEARTBEAT, None)
        msg_type, result = serializer.deserialize(data)

        assert msg_type == MessageType.HEARTBEAT
        assert result is None

    def test_message_header(self):
        """Test message header serialization."""
        header = MessageHeader(
            message_type=MessageType.FULL_STATE,
            version=1,
            sequence=100,
            timestamp=12345.678,
            flags=0,
            payload_size=50
        )

        data = header.to_bytes()
        assert len(data) == MessageHeader.HEADER_SIZE

        restored = MessageHeader.from_bytes(data)
        assert restored.message_type == header.message_type
        assert restored.sequence == header.sequence
        assert restored.payload_size == header.payload_size

    def test_compression(self):
        """Test automatic compression of large payloads."""
        serializer = NetSerializer(compress_threshold=50)
        payload = {'data': 'x' * 200}

        data = serializer.serialize(MessageType.FULL_STATE, payload)
        header = serializer.deserialize_header(data)

        assert header.flags & MessageHeader.FLAG_COMPRESSED

    def test_sequence_incrementing(self):
        """Test sequence number incrementing."""
        serializer = NetSerializer()

        data1 = serializer.serialize(MessageType.HEARTBEAT, None)
        data2 = serializer.serialize(MessageType.HEARTBEAT, None)

        header1 = MessageHeader.from_bytes(data1)
        header2 = MessageHeader.from_bytes(data2)

        assert header2.sequence == header1.sequence + 1

    def test_custom_encoder_decoder(self):
        """Test registering custom encoder/decoder."""
        serializer = NetSerializer()

        def encode_custom(payload):
            return struct.pack('!i', payload['value'])

        def decode_custom(data):
            value, = struct.unpack('!i', data)
            return {'value': value}

        serializer.register_encoder(MessageType.CUSTOM_START, encode_custom)
        serializer.register_decoder(MessageType.CUSTOM_START, decode_custom)

        data = serializer.serialize(MessageType.CUSTOM_START, {'value': 42})
        msg_type, result = serializer.deserialize(data)

        assert msg_type == MessageType.CUSTOM_START
        assert result['value'] == 42

    def test_convenience_functions(self):
        """Test convenience serialize/deserialize functions."""
        payload = {'test': 123}
        data = serialize_message(MessageType.FULL_STATE, payload)
        msg_type, result = deserialize_message(data)

        assert msg_type == MessageType.FULL_STATE
        assert result == payload

    def test_nested_data_structures(self):
        """Test serializing nested data structures."""
        serializer = NetSerializer()
        payload = {
            'list': [1, 2, 3],
            'nested': {'a': 1, 'b': 2},
            'deep': {'x': {'y': {'z': 42}}},
        }

        data = serializer.serialize(MessageType.FULL_STATE, payload)
        msg_type, result = serializer.deserialize(data)

        assert result['list'] == [1, 2, 3]
        assert result['nested'] == {'a': 1, 'b': 2}
        assert result['deep']['x']['y']['z'] == 42

    def test_version_field(self):
        """Test protocol version field."""
        serializer = NetSerializer(version=5)
        data = serializer.serialize(MessageType.HEARTBEAT, None)
        header = serializer.deserialize_header(data)

        assert header.version == 5


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_data(self):
        """Test handling empty data."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {})
        delta = encoder.encode_delta({}, baseline_seq=0)
        result = encoder.decode_delta(delta, baseline_seq=0)
        assert result == {}
        assert isinstance(result, dict)

    def test_invalid_baseline_sequence(self):
        """Test error on invalid baseline sequence."""
        encoder = DeltaEncoder()
        encoder.set_baseline(0, {'x': 0})

        with pytest.raises(KeyError) as exc_info:
            encoder.encode_delta({'x': 1}, baseline_seq=99)
        assert "99" in str(exc_info.value)

    def test_unicode_strings(self):
        """Test Unicode string handling."""
        encoder = DeltaEncoder()
        state = {'name': 'Test with unicode: \u00e9\u00e8\u00ea'}
        encoder.set_baseline(0, {})
        delta = encoder.encode_delta(state, baseline_seq=0)
        result = encoder.decode_delta(delta, baseline_seq=0)
        assert result['name'] == state['name']
        # Verify the actual unicode characters are preserved
        assert '\u00e9' in result['name']
        assert '\u00e8' in result['name']
        assert '\u00ea' in result['name']

    def test_large_integers(self):
        """Test large integer handling."""
        encoder = DeltaEncoder()
        state = {
            'small': 127,
            'medium': 32767,
            'large': 2147483647,
            'huge': 9223372036854775807,
        }
        encoder.set_baseline(0, {})
        delta = encoder.encode_delta(state, baseline_seq=0)
        result = encoder.decode_delta(delta, baseline_seq=0)

        # Verify each value matches exactly
        assert result['small'] == 127
        assert result['medium'] == 32767
        assert result['large'] == 2147483647
        assert result['huge'] == 9223372036854775807

    def test_negative_integers(self):
        """Test negative integer handling."""
        encoder = DeltaEncoder()
        state = {'neg': -128, 'neg_large': -2147483648}
        encoder.set_baseline(0, {})
        delta = encoder.encode_delta(state, baseline_seq=0)
        result = encoder.decode_delta(delta, baseline_seq=0)

        assert result['neg'] == -128
        assert result['neg_large'] == -2147483648

    def test_bit_writer_invalid_num_bits(self):
        """Test BitWriter raises on invalid bit counts."""
        writer = BitWriter()
        with pytest.raises(ValueError) as exc_info:
            writer.write_bits(0, 0)
        assert "1-64" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            writer.write_bits(0, 65)
        assert "1-64" in str(exc_info.value)

    def test_bit_reader_invalid_num_bits(self):
        """Test BitReader raises on invalid bit counts."""
        reader = BitReader(bytes([0xFF]))
        with pytest.raises(ValueError) as exc_info:
            reader.read_bits(0)
        assert "1-64" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            reader.read_bits(65)
        assert "1-64" in str(exc_info.value)

    def test_quantize_float_invalid_range(self):
        """Test quantize_float raises on invalid range."""
        with pytest.raises(ValueError) as exc_info:
            quantize_float(0.5, 1.0, 0.0, 8)  # min > max
        assert "min_value" in str(exc_info.value)

    def test_quantize_vector3_invalid_precision(self):
        """Test quantize_vector3 raises on invalid precision."""
        with pytest.raises(ValueError) as exc_info:
            quantize_vector3((0, 0, 0), precision=10)
        assert "precision" in str(exc_info.value)

    def test_message_header_short_data(self):
        """Test MessageHeader raises on insufficient data."""
        with pytest.raises(ValueError) as exc_info:
            MessageHeader.from_bytes(bytes(10))
        assert "bytes" in str(exc_info.value)

    def test_net_serializer_short_data(self):
        """Test NetSerializer raises on insufficient data."""
        serializer = NetSerializer()
        with pytest.raises(ValueError) as exc_info:
            serializer.deserialize(bytes(5))
        assert "too short" in str(exc_info.value)

    def test_baseline_capacity_limit(self):
        """Test baseline capacity is enforced."""
        encoder = DeltaEncoder(max_baselines=3)

        # Add 4 baselines
        for i in range(4):
            encoder.set_baseline(i, {'x': i})

        # Oldest should be removed
        assert encoder.get_baseline(0) is None
        assert encoder.get_baseline(1) is not None
        assert encoder.get_baseline(2) is not None
        assert encoder.get_baseline(3) is not None
        assert encoder.get_baseline_count() == 3

    def test_config_constants_used(self):
        """Test that config constants are properly used."""
        # Verify constants match expected values
        assert DEFAULT_CONFIG.MTU == 1200
        assert DEFAULT_CONFIG.PACKET_HEADER_SIZE == 12
        assert DEFAULT_CONFIG.COMPRESS_THRESHOLD == 128
        assert DEFAULT_CONFIG.MAX_BASELINES == 64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
