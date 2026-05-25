"""
White-box tests for packet header edge cases.

Tests PacketHeader serialization, Packet creation, flag operations,
FragmentHeader, PacketFragmenter, and sequence number utilities.
"""

from __future__ import annotations

import pytest

from engine.networking.transport.packet import (
    PacketHeader, Packet, PacketType, PacketFlags,
    FragmentHeader, PacketFragmenter,
    HEADER_SIZE, MAX_PAYLOAD_SIZE,
    sequence_greater_than, sequence_difference,
)
from engine.networking.config import DEFAULT_CONFIG


class TestPacketHeaderEdgeCases:
    """Packet header serialization and edge cases."""

    def test_header_roundtrip(self):
        """PacketHeader roundtrip through to_bytes/from_bytes."""
        header = PacketHeader(
            packet_type=PacketType.DATA,
            flags=PacketFlags.RELIABLE,
            sequence=42,
            ack=10,
            ack_bits=0xABCD,
            size=100
        )
        data = header.to_bytes()
        assert len(data) == HEADER_SIZE
        restored = PacketHeader.from_bytes(data)
        assert restored.packet_type == PacketType.DATA
        assert restored.flags == PacketFlags.RELIABLE
        assert restored.sequence == 42
        assert restored.ack == 10
        assert restored.ack_bits == 0xABCD
        assert restored.size == 100

    def test_from_bytes_too_short(self):
        """from_bytes raises ValueError for too-short data."""
        with pytest.raises(ValueError, match=r"Need 12 bytes"):
            PacketHeader.from_bytes(b'\x00' * 4)

    def test_from_bytes_empty(self):
        """from_bytes raises ValueError for empty data."""
        with pytest.raises(ValueError):
            PacketHeader.from_bytes(b'')

    def test_all_packet_types_roundtrip(self):
        """All PacketType values survive serialization roundtrip."""
        for pt in PacketType:
            header = PacketHeader(packet_type=pt, sequence=1)
            data = header.to_bytes()
            restored = PacketHeader.from_bytes(data)
            assert restored.packet_type == pt

    def test_sequence_wraparound(self):
        """Sequence wraparound at 0xFFFF is handled."""
        header = PacketHeader(packet_type=PacketType.DATA, sequence=0xFFFF)
        data = header.to_bytes()
        restored = PacketHeader.from_bytes(data)
        assert restored.sequence == 0xFFFF

    def test_unknown_packet_type_falls_back(self):
        """Unknown packet type falls back to DATA."""
        raw = bytearray(HEADER_SIZE)
        raw[0] = 0xFF
        header = PacketHeader.from_bytes(bytes(raw))
        assert header.packet_type == PacketType.DATA

    def test_packet_create_data(self):
        """Packet.create() creates DATA packet with payload and flags."""
        packet = Packet.create(PacketType.DATA, b'hello', sequence=5, flags=PacketFlags.RELIABLE)
        assert packet.header.packet_type == PacketType.DATA
        assert packet.payload == b'hello'
        assert packet.header.sequence == 5
        assert packet.header.has_flag(PacketFlags.RELIABLE)
        assert packet.header.size == 5

    def test_packet_create_ack(self):
        """Packet.create_ack() creates ACK packet."""
        packet = Packet.create_ack(100, 0xFFFF)
        assert packet.header.packet_type == PacketType.ACK
        assert packet.header.ack == 100
        assert packet.header.ack_bits == 0xFFFF

    def test_packet_to_bytes_from_bytes(self):
        """Packet roundtrip through to_bytes/from_bytes."""
        original = Packet.create(PacketType.DATA, b'hello', sequence=42)
        data = original.to_bytes()
        restored = Packet.from_bytes(data)
        assert restored.header.packet_type == PacketType.DATA
        assert restored.payload == b'hello'
        assert restored.header.sequence == 42

    def test_packet_total_size(self):
        """total_size includes header and payload."""
        packet = Packet.create(PacketType.DATA, b'A' * 100)
        assert packet.total_size == HEADER_SIZE + 100

    def test_flags_set_and_check(self):
        """PacketFlags can be set, checked, and cleared."""
        header = PacketHeader(packet_type=PacketType.DATA)
        assert not header.has_flag(PacketFlags.RELIABLE)
        header.set_flag(PacketFlags.RELIABLE)
        assert header.has_flag(PacketFlags.RELIABLE)
        header.clear_flag(PacketFlags.RELIABLE)
        assert not header.has_flag(PacketFlags.RELIABLE)

    def test_flags_combination(self):
        """Multiple flags can be set simultaneously."""
        header = PacketHeader(packet_type=PacketType.DATA)
        header.set_flag(PacketFlags.RELIABLE)
        header.set_flag(PacketFlags.ORDERED)
        assert header.has_flag(PacketFlags.RELIABLE)
        assert header.has_flag(PacketFlags.ORDERED)
        header.clear_flag(PacketFlags.RELIABLE)
        assert not header.has_flag(PacketFlags.RELIABLE)
        assert header.has_flag(PacketFlags.ORDERED)

    def test_packet_is_reliable(self):
        """is_reliable() checks RELIABLE flag."""
        reliable = Packet.create(PacketType.DATA, flags=PacketFlags.RELIABLE)
        assert reliable.is_reliable()
        unreliable = Packet.create(PacketType.DATA)
        assert not unreliable.is_reliable()

    def test_packet_is_fragmented(self):
        """is_fragmented() checks FRAGMENTED flag."""
        frag = Packet.create(PacketType.DATA, flags=PacketFlags.FRAGMENTED)
        assert frag.is_fragmented()
        normal = Packet.create(PacketType.DATA)
        assert not normal.is_fragmented()


class TestFragmentHeader:
    """FragmentHeader serialization and edge cases."""

    def test_fragment_header_roundtrip(self):
        """FragmentHeader roundtrip through to_bytes/from_bytes."""
        fh = FragmentHeader(fragment_id=42, fragment_index=0, fragment_total=3)
        data = fh.to_bytes()
        assert len(data) == FragmentHeader.SIZE
        assert FragmentHeader.SIZE == 4
        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 42
        assert restored.fragment_index == 0
        assert restored.fragment_total == 3

    def test_fragment_header_invalid_short(self):
        """FragmentHeader.from_bytes raises on short data."""
        with pytest.raises(ValueError, match=r"Need 4 bytes"):
            FragmentHeader.from_bytes(b'\x00\x00')

    def test_fragment_header_max_values(self):
        """FragmentHeader handles max values."""
        fh = FragmentHeader(fragment_id=0xFFFF, fragment_index=0xFF, fragment_total=0xFF)
        data = fh.to_bytes()
        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 0xFFFF
        assert restored.fragment_index == 0xFF
        assert restored.fragment_total == 0xFF


class TestPacketFragmenter:
    """PacketFragmenter edge cases."""

    def test_no_fragmentation_needed(self):
        """Fragmenter returns single packet for small payload."""
        fragmenter = PacketFragmenter()
        payload = b'A' * 100
        packets = fragmenter.fragment(payload)
        assert len(packets) == 1
        assert packets[0].payload == payload

    def test_fragmentation_at_max_payload(self):
        """Fragmenter splits payload > MAX_PAYLOAD_SIZE."""
        fragmenter = PacketFragmenter()
        payload = b'A' * (MAX_PAYLOAD_SIZE + 1)
        packets = fragmenter.fragment(payload)
        assert len(packets) >= 2
        for p in packets:
            assert p.header.has_flag(PacketFlags.FRAGMENTED)
            assert p.header.has_flag(PacketFlags.RELIABLE)

    def test_reassemble_fragments(self):
        """Fragmenter reassembles fragments correctly."""
        fragmenter = PacketFragmenter()
        original = b'X' * (MAX_PAYLOAD_SIZE + 100)
        packets = fragmenter.fragment(original)
        assert len(packets) >= 2
        for i, packet in enumerate(packets):
            result = fragmenter.add_fragment(packet)
            if i < len(packets) - 1:
                assert result is None
            else:
                assert result == original

    def test_fragment_id_increments(self):
        """Fragment ID increments between fragment calls."""
        fragmenter = PacketFragmenter()
        payload = b'A' * (MAX_PAYLOAD_SIZE + 1)
        packets1 = fragmenter.fragment(payload)
        packets2 = fragmenter.fragment(payload)
        import struct
        fh1 = struct.unpack('!HBB', packets1[0].payload[:4])
        fh2 = struct.unpack('!HBB', packets2[0].payload[:4])
        assert fh2[0] != fh1[0]

    def test_add_fragment_non_fragment(self):
        """add_fragment() returns packet.payload for non-FRAGMENT packets."""
        fragmenter = PacketFragmenter()
        packet = Packet.create(PacketType.DATA, b'hello', sequence=1)
        result = fragmenter.add_fragment(packet)
        assert result == b'hello'

    def test_fragment_payload_size_constant(self):
        """FRAGMENT_PAYLOAD_SIZE is correctly calculated."""
        from engine.networking.transport.packet import FragmentHeader
        expected = MAX_PAYLOAD_SIZE - FragmentHeader.SIZE
        assert PacketFragmenter.FRAGMENT_PAYLOAD_SIZE == expected

    def test_clear_pending_all(self):
        """clear_pending() clears all pending fragments."""
        fragmenter = PacketFragmenter()
        payload = b'X' * (MAX_PAYLOAD_SIZE + 1)
        packets = fragmenter.fragment(payload)
        # Add only first fragment so it stays pending (incomplete)
        fragmenter.add_fragment(packets[0])
        assert len(fragmenter._pending_fragments) > 0
        fragmenter.clear_pending()
        assert len(fragmenter._pending_fragments) == 0

    def test_clear_pending_specific(self):
        """clear_pending(frag_id) clears specific fragment group."""
        fragmenter = PacketFragmenter()
        fragmenter._pending_fragments[42] = {0: b'data'}
        fragmenter._fragment_totals[42] = 1
        fragmenter.clear_pending(42)
        assert 42 not in fragmenter._pending_fragments
        assert 42 not in fragmenter._fragment_totals


class TestSequenceUtils:
    """Sequence comparison utility edge cases."""

    def test_sequence_greater_than_basic(self):
        """sequence_greater_than works for normal values."""
        assert sequence_greater_than(10, 5)
        assert not sequence_greater_than(5, 10)
        assert not sequence_greater_than(5, 5)

    def test_sequence_greater_than_wraparound_forward(self):
        """sequence_greater_than wraparound: 1 > 65535."""
        assert sequence_greater_than(1, 65535)

    def test_sequence_greater_than_wraparound_backward(self):
        """sequence_greater_than wraparound: 65535 < 1."""
        assert not sequence_greater_than(65535, 1)

    def test_sequence_difference_basic(self):
        """sequence_difference returns correct difference."""
        assert sequence_difference(10, 5) == 5
        assert sequence_difference(5, 10) == -5

    def test_sequence_difference_wraparound(self):
        """sequence_difference handles wraparound."""
        diff = sequence_difference(1, 65535)
        assert diff == 2 or diff == -65534

    def test_sequence_difference_zero(self):
        """sequence_difference of equal numbers is 0."""
        assert sequence_difference(42, 42) == 0

    def test_headder_size_constant(self):
        """HEADER_SIZE matches config."""
        assert HEADER_SIZE == DEFAULT_CONFIG.PACKET_HEADER_SIZE
        assert HEADER_SIZE == 12
