"""
BLACKBOX tests for packet primitives (Phase 1, Task 1.1).

Tests the PUBLIC API of the packet module:
  - PacketHeader serialization roundtrip
  - Packet creation with all PacketType values
  - Sequence number comparison with wraparound
  - FragmentHeader serialization roundtrip
  - Packet fragmentation and reassembly (in-order and out-of-order)

These tests verify ACCEPTANCE CRITERIA from PHASE_1_TODO.md -- not internals.
No _pending_fragments or other implementation details are accessed.
"""

from __future__ import annotations

import struct

import pytest

from engine.networking.transport import (
    Packet,
    PacketHeader,
    PacketType,
    MTU,
    MAX_PAYLOAD_SIZE,
)

from engine.networking.transport.packet import (
    PacketFlags,
    FragmentHeader,
    PacketFragmenter,
    HEADER_SIZE,
    sequence_greater_than,
)

from engine.networking.config import DEFAULT_CONFIG


# =============================================================================
# 1. PacketHeader Serialization Roundtrip
# =============================================================================

class TestPacketHeaderRoundtrip:
    """PacketHeader to_bytes/from_bytes roundtrip preserves all fields."""

    def test_header_roundtrip_preserves_all_fields(self):
        """All PacketHeader fields survive to_bytes -> from_bytes roundtrip."""
        header = PacketHeader(
            packet_type=PacketType.DATA,
            flags=PacketFlags.RELIABLE | PacketFlags.ORDERED,
            sequence=42,
            ack=10,
            ack_bits=0xABCD1234,
            size=100,
        )
        data = header.to_bytes()
        assert len(data) == HEADER_SIZE, f"Expected {HEADER_SIZE} bytes, got {len(data)}"

        restored = PacketHeader.from_bytes(data)
        assert restored.packet_type == PacketType.DATA
        assert restored.flags == (PacketFlags.RELIABLE | PacketFlags.ORDERED)
        assert restored.sequence == 42
        assert restored.ack == 10
        assert restored.ack_bits == 0xABCD1234
        assert restored.size == 100

    def test_header_roundtrip_with_max_values(self):
        """PacketHeader roundtrips at maximum field values."""
        header = PacketHeader(
            packet_type=PacketType.SEQUENCED_DATA,
            flags=0xFF,
            sequence=0xFFFF,
            ack=0xFFFF,
            ack_bits=0xFFFFFFFF,
            size=MAX_PAYLOAD_SIZE,
        )
        data = header.to_bytes()
        restored = PacketHeader.from_bytes(data)
        assert restored.packet_type == PacketType.SEQUENCED_DATA
        assert restored.flags == 0xFF
        assert restored.sequence == 0xFFFF
        assert restored.ack == 0xFFFF
        assert restored.ack_bits == 0xFFFFFFFF
        assert restored.size == MAX_PAYLOAD_SIZE

    def test_header_roundtrip_with_zero_values(self):
        """PacketHeader roundtrips with all-zero fields."""
        header = PacketHeader(
            packet_type=PacketType.DATA,
            flags=0,
            sequence=0,
            ack=0,
            ack_bits=0,
            size=0,
        )
        data = header.to_bytes()
        restored = PacketHeader.from_bytes(data)
        assert restored.packet_type == PacketType.DATA
        assert restored.flags == 0
        assert restored.sequence == 0
        assert restored.ack == 0
        assert restored.ack_bits == 0
        assert restored.size == 0

    def test_header_from_bytes_rejects_short_data(self):
        """from_bytes raises ValueError when fewer than HEADER_SIZE bytes."""
        with pytest.raises(ValueError, match="Need .* bytes for header"):
            PacketHeader.from_bytes(b"\x00" * (HEADER_SIZE - 1))

    def test_header_from_bytes_rejects_empty_data(self):
        """from_bytes raises ValueError on empty data."""
        with pytest.raises(ValueError):
            PacketHeader.from_bytes(b"")


# =============================================================================
# 2. Packet Creation with All PacketType Values
# =============================================================================

class TestPacketCreation:
    """Packet creation succeeds with all PacketType values."""

    def test_all_packet_types_can_be_created(self):
        """Every PacketType value can be used to create a Packet."""
        for pt in PacketType:
            packet = Packet.create(pt, payload=b"test", sequence=1)
            assert packet.header.packet_type == pt
            assert packet.payload == b"test"
            assert packet.header.sequence == 1

    def test_all_packet_types_survive_serialization(self):
        """Every PacketType value survives to_bytes -> from_bytes roundtrip."""
        for pt in PacketType:
            original = Packet.create(pt, payload=b"payload", sequence=42)
            data = original.to_bytes()
            restored = Packet.from_bytes(data)
            assert restored.header.packet_type == pt
            assert restored.payload == b"payload"
            assert restored.header.sequence == 42

    def test_create_data_packet_with_payload(self):
        """DATA packet can be created with a payload and serialized back."""
        payload = b"hello network world"
        packet = Packet.create(PacketType.DATA, payload, sequence=5)
        assert packet.header.packet_type == PacketType.DATA
        assert packet.payload == payload
        assert packet.header.sequence == 5
        assert packet.header.size == len(payload)

        data = packet.to_bytes()
        restored = Packet.from_bytes(data)
        assert restored.payload == payload
        assert restored.header.size == len(payload)

    def test_create_ack_packet(self):
        """ACK packet can be created with ack sequence and bitfield."""
        packet = Packet.create_ack(ack_sequence=100, ack_bits=0xFFFF)
        assert packet.header.packet_type == PacketType.ACK
        assert packet.header.ack == 100
        assert packet.header.ack_bits == 0xFFFF
        assert packet.payload == b""

    def test_create_heartbeat_packet(self):
        """HEARTBEAT packet can be created with sequence number."""
        packet = Packet.create_heartbeat(sequence=7)
        assert packet.header.packet_type == PacketType.HEARTBEAT
        assert packet.header.sequence == 7
        assert packet.payload == b""

    def test_packet_with_empty_payload(self):
        """Packet with empty payload roundtrips correctly."""
        packet = Packet.create(PacketType.RELIABLE_DATA, b"", sequence=0)
        assert packet.total_size == HEADER_SIZE
        data = packet.to_bytes()
        restored = Packet.from_bytes(data)
        assert restored.payload == b""
        assert restored.header.size == 0

    def test_packet_with_max_payload(self):
        """Packet with MAX_PAYLOAD_SIZE payload roundtrips correctly."""
        payload = b"x" * MAX_PAYLOAD_SIZE
        packet = Packet.create(PacketType.DATA, payload)
        assert packet.header.size == MAX_PAYLOAD_SIZE
        assert packet.total_size == HEADER_SIZE + MAX_PAYLOAD_SIZE
        data = packet.to_bytes()
        restored = Packet.from_bytes(data)
        assert len(restored.payload) == MAX_PAYLOAD_SIZE

    def test_packet_total_size_property(self):
        """total_size property equals HEADER_SIZE + payload length."""
        for size in (0, 1, 100, 500):
            packet = Packet.create(PacketType.DATA, b"x" * size)
            assert packet.total_size == HEADER_SIZE + size


# =============================================================================
# 3. Sequence Number Wraparound Comparison
# =============================================================================

class TestSequenceComparison:
    """sequence_greater_than handles wraparound correctly."""

    def test_basic_greater_than(self):
        """sequence_greater_than(100, 50) returns True."""
        result = sequence_greater_than(100, 50)
        assert result is True

    def test_basic_less_than(self):
        """sequence_greater_than(50, 100) returns False."""
        result = sequence_greater_than(50, 100)
        assert result is False

    def test_equal_returns_false(self):
        """sequence_greater_than(n, n) returns False."""
        assert sequence_greater_than(0, 0) is False
        assert sequence_greater_than(32768, 32768) is False
        assert sequence_greater_than(65535, 65535) is False

    def test_wraparound_forward(self):
        """sequence_greater_than(10, 65530) returns True (wraparound)."""
        result = sequence_greater_than(10, 65530)
        assert result is True

    def test_wraparound_backward(self):
        """sequence_greater_than(65530, 10) returns False (wraparound)."""
        result = sequence_greater_than(65530, 10)
        assert result is False

    def test_wraparound_at_max_sequence(self):
        """sequence_greater_than handles edge at MAX_SEQUENCE boundary."""
        max_seq = DEFAULT_CONFIG.MAX_SEQUENCE
        # s1 just wrapped past max, s2 is near max
        assert sequence_greater_than(0, max_seq) is True
        assert sequence_greater_than(1, max_seq) is True
        # s1 is near max, s2 just wrapped
        assert sequence_greater_than(max_seq, 0) is False
        assert sequence_greater_than(max_seq, 1) is False

    def test_wraparound_half_boundary(self):
        """sequence_greater_than correctly handles boundary at half of range."""
        max_seq = DEFAULT_CONFIG.MAX_SEQUENCE
        half = max_seq // 2

        # s1 > s2 and difference <= half: True
        assert sequence_greater_than(half, 0) is True
        # s1 < s2 and difference > half: True (wraparound)
        assert sequence_greater_than(0, half + 1) is True
        # s1 > s2 but difference > half: False (actually wraparound in reverse)
        # e.g. s1=50000, s2=100: diff=49900 > half, so s1 < s2 in wraparound space
        assert sequence_greater_than(50000, 100) is False


# =============================================================================
# 4. FragmentHeader Serialization Roundtrip
# =============================================================================

class TestFragmentHeaderRoundtrip:
    """FragmentHeader to_bytes/from_bytes preserves all fields."""

    def test_fragment_header_roundtrip(self):
        """FragmentHeader roundtrips through to_bytes/from_bytes."""
        fh = FragmentHeader(
            fragment_id=42,
            fragment_index=0,
            fragment_total=3,
        )
        data = fh.to_bytes()
        assert len(data) == FragmentHeader.SIZE, \
            f"Expected {FragmentHeader.SIZE} bytes, got {len(data)}"

        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 42
        assert restored.fragment_index == 0
        assert restored.fragment_total == 3

    def test_fragment_header_max_values(self):
        """FragmentHeader roundtrips at maximum field values."""
        fh = FragmentHeader(
            fragment_id=0xFFFF,
            fragment_index=0xFF,
            fragment_total=0xFF,
        )
        data = fh.to_bytes()
        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 0xFFFF
        assert restored.fragment_index == 0xFF
        assert restored.fragment_total == 0xFF

    def test_fragment_header_zero_values(self):
        """FragmentHeader roundtrips with all-zero fields."""
        fh = FragmentHeader(fragment_id=0, fragment_index=0, fragment_total=0)
        data = fh.to_bytes()
        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 0
        assert restored.fragment_index == 0
        assert restored.fragment_total == 0

    def test_fragment_header_middle_fragment(self):
        """FragmentHeader roundtrips for a middle fragment (index=1 of 5)."""
        fh = FragmentHeader(fragment_id=100, fragment_index=1, fragment_total=5)
        data = fh.to_bytes()
        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 100
        assert restored.fragment_index == 1
        assert restored.fragment_total == 5

    def test_fragment_header_last_fragment(self):
        """FragmentHeader roundtrips for the last fragment (index=total-1)."""
        fh = FragmentHeader(fragment_id=200, fragment_index=9, fragment_total=10)
        data = fh.to_bytes()
        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 200
        assert restored.fragment_index == 9
        assert restored.fragment_total == 10

    def test_fragment_header_size_constant(self):
        """FragmentHeader.SIZE is 4 bytes."""
        assert FragmentHeader.SIZE == 4

    def test_fragment_header_short_data_raises(self):
        """from_bytes raises ValueError for data shorter than SIZE."""
        with pytest.raises(ValueError, match="Need .* bytes for fragment header"):
            FragmentHeader.from_bytes(b"\x00" * (FragmentHeader.SIZE - 1))


# =============================================================================
# 5. Packet Fragmentation and Reassembly
# =============================================================================

class TestPacketFragmenter:
    """Packet fragmentation at MTU boundary and reassembly."""

    FRAG_PAYLOAD_SIZE = MAX_PAYLOAD_SIZE - FragmentHeader.SIZE

    def test_small_payload_not_fragmented(self):
        """Payload <= MAX_PAYLOAD_SIZE returns a single DATA packet."""
        fragmenter = PacketFragmenter()
        payload = b"x" * 100
        packets = fragmenter.fragment(payload)
        assert len(packets) == 1
        assert packets[0].header.packet_type == PacketType.DATA
        assert packets[0].payload == payload

    def test_exact_max_payload_not_fragmented(self):
        """Payload exactly MAX_PAYLOAD_SIZE is not fragmented."""
        fragmenter = PacketFragmenter()
        payload = b"x" * MAX_PAYLOAD_SIZE
        packets = fragmenter.fragment(payload)
        assert len(packets) == 1

    def test_one_byte_over_max_is_fragmented(self):
        """Payload MAX_PAYLOAD_SIZE + 1 produces 2 fragments."""
        fragmenter = PacketFragmenter()
        payload = b"x" * (MAX_PAYLOAD_SIZE + 1)
        packets = fragmenter.fragment(payload)
        assert len(packets) == 2
        for p in packets:
            assert p.header.packet_type == PacketType.FRAGMENT
            assert p.is_fragmented()

    def test_fragmentation_at_mtu_boundary(self):
        """Payload exactly 2x fragment payload size produces 2 full fragments."""
        fragmenter = PacketFragmenter()
        payload = b"x" * (self.FRAG_PAYLOAD_SIZE * 2)
        packets = fragmenter.fragment(payload)
        assert len(packets) == 2

    def test_fragmentation_with_partial_last_fragment(self):
        """Payload 2x + 1 produces 3 fragments (last is partial)."""
        fragmenter = PacketFragmenter()
        payload = b"x" * (self.FRAG_PAYLOAD_SIZE * 2 + 1)
        packets = fragmenter.fragment(payload)
        assert len(packets) == 3

    def test_reassembly_in_order(self):
        """Fragments reassemble to original payload when added in order."""
        fragmenter = PacketFragmenter()
        original = b"hello world fragmented payload test " * 100
        packets = fragmenter.fragment(original)
        assert len(packets) > 1, "Payload should produce multiple fragments"

        result = None
        for p in packets:
            result = fragmenter.add_fragment(p)
        assert result == original, "Reassembled payload must match original"

    def test_reassembly_out_of_order(self):
        """Fragments reassemble correctly when added in reverse order."""
        fragmenter = PacketFragmenter()
        original = b"out of order reassembly test payload " * 80
        packets = fragmenter.fragment(original)
        assert len(packets) > 1

        result = None
        for p in reversed(packets):
            result = fragmenter.add_fragment(p)
        assert result == original

    def test_reassembly_scrambled_order(self):
        """Fragments reassemble correctly when added in non-sequential order."""
        fragmenter = PacketFragmenter()
        payload_size = self.FRAG_PAYLOAD_SIZE * 4 + 50
        original = b"scrambled order reassembly " * (
            payload_size // len(b"scrambled order reassembly ") + 1
        )
        original = original[:payload_size]
        packets = fragmenter.fragment(original)
        assert len(packets) == 5

        # Add fragments in order: 0, 2, 4, 1, 3
        result = None
        for idx in (0, 2, 4, 1, 3):
            result = fragmenter.add_fragment(packets[idx])
        assert result == original

    def test_incomplete_reassembly_returns_none(self):
        """Adding only some fragments returns None (incomplete)."""
        fragmenter = PacketFragmenter()
        payload = b"x" * (MAX_PAYLOAD_SIZE + 100)
        packets = fragmenter.fragment(payload)
        assert len(packets) >= 2

        # Only add the first fragment
        result = fragmenter.add_fragment(packets[0])
        assert result is None, "Incomplete reassembly must return None"

    def test_non_fragment_packet_passes_through(self):
        """add_fragment returns payload as-is for non-FRAGMENT packets."""
        fragmenter = PacketFragmenter()
        packet = Packet.create(PacketType.DATA, b"passthrough", sequence=1)
        result = fragmenter.add_fragment(packet)
        assert result == b"passthrough"

    def test_consecutive_fragment_groups(self):
        """Multiple fragment groups can be processed sequentially."""
        fragmenter = PacketFragmenter()
        payload_a = b"A" * (MAX_PAYLOAD_SIZE + 50)
        payload_b = b"B" * (MAX_PAYLOAD_SIZE + 50)

        packets_a = fragmenter.fragment(payload_a)
        packets_b = fragmenter.fragment(payload_b)

        # Reassemble first group
        result_a = None
        for p in packets_a:
            result_a = fragmenter.add_fragment(p)
        assert result_a == payload_a

        # Reassemble second group (same fragmenter)
        result_b = None
        for p in packets_b:
            result_b = fragmenter.add_fragment(p)
        assert result_b == payload_b


# =============================================================================
# 6. Flag Operations
# =============================================================================

class TestPacketFlags:
    """PacketFlags set, check, and clear operations."""

    def test_flag_set_and_check(self):
        """A flag can be set and checked on a PacketHeader."""
        header = PacketHeader(packet_type=PacketType.DATA)
        assert not header.has_flag(PacketFlags.COMPRESSED)
        header.set_flag(PacketFlags.COMPRESSED)
        assert header.has_flag(PacketFlags.COMPRESSED)

    def test_flag_clear(self):
        """A set flag can be cleared."""
        header = PacketHeader(packet_type=PacketType.DATA)
        header.set_flag(PacketFlags.RELIABLE)
        assert header.has_flag(PacketFlags.RELIABLE)
        header.clear_flag(PacketFlags.RELIABLE)
        assert not header.has_flag(PacketFlags.RELIABLE)

    def test_multiple_flags_combined(self):
        """Multiple flags can be set and checked independently."""
        header = PacketHeader(packet_type=PacketType.DATA)
        header.set_flag(PacketFlags.RELIABLE)
        header.set_flag(PacketFlags.ORDERED)
        header.set_flag(PacketFlags.PRIORITY_HIGH)
        assert header.has_flag(PacketFlags.RELIABLE)
        assert header.has_flag(PacketFlags.ORDERED)
        assert header.has_flag(PacketFlags.PRIORITY_HIGH)
        assert not header.has_flag(PacketFlags.COMPRESSED)
        assert not header.has_flag(PacketFlags.FRAGMENTED)

    def test_flags_survive_serialization(self):
        """Flags set on a header survive to_bytes/from_bytes roundtrip."""
        header = PacketHeader(packet_type=PacketType.RELIABLE_DATA)
        header.set_flag(PacketFlags.RELIABLE)
        header.set_flag(PacketFlags.PRIORITY_HIGH)

        data = header.to_bytes()
        restored = PacketHeader.from_bytes(data)
        assert restored.has_flag(PacketFlags.RELIABLE)
        assert restored.has_flag(PacketFlags.PRIORITY_HIGH)

    def test_is_reliable_property(self):
        """Packet.is_reliable() reflects the RELIABLE flag."""
        reliable = Packet.create(PacketType.DATA, flags=PacketFlags.RELIABLE)
        assert reliable.is_reliable()
        unreliable = Packet.create(PacketType.DATA)
        assert not unreliable.is_reliable()

    def test_is_fragmented_property(self):
        """Packet.is_fragmented() reflects the FRAGMENTED flag."""
        frag = Packet.create(PacketType.DATA, flags=PacketFlags.FRAGMENTED)
        assert frag.is_fragmented()
        normal = Packet.create(PacketType.DATA)
        assert not normal.is_fragmented()
