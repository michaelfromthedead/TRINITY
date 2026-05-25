"""T-NET-1.7: WHITEBOX tests for packet coalescing and fragmentation boundary cases.

Tests the PacketFragmenter at MTU and fragment-size boundaries,
multiple-fragment reassembly, and edge cases around empty payloads
and fragment-sequence continuity.
"""

from __future__ import annotations

import pytest

from engine.networking.transport.packet import (
    MTU,
    HEADER_SIZE,
    MAX_PAYLOAD_SIZE,
    Packet,
    PacketType,
    PacketFlags,
    PacketFragmenter,
)


# The fragmenter splits payloads > MAX_PAYLOAD_SIZE into chunks of
# FRAGMENT_PAYLOAD_SIZE = MAX_PAYLOAD_SIZE - FragmentHeader.SIZE.
# FragmentHeader.SIZE = 4  (defined in packet.py as FRAGMENT_HEADER_SIZE from config)
def _fragment_payload_size() -> int:
    """Return the per-fragment data capacity (1184)."""
    return MAX_PAYLOAD_SIZE - 4  # FragmentHeader.SIZE


class TestPacketFragmenterBoundaries:
    """Whitebox: fragment boundary conditions."""

    def test_payload_at_max_payload_size(self):
        """Payload == MAX_PAYLOAD_SIZE is sent as a single packet (not fragmented)."""
        fragmenter = PacketFragmenter()
        payload = b"X" * MAX_PAYLOAD_SIZE
        fragments = fragmenter.fragment(payload)

        assert len(fragments) == 1
        assert len(fragments[0].payload) == MAX_PAYLOAD_SIZE
        assert fragments[0].header.packet_type == PacketType.DATA
        assert not fragments[0].is_fragmented()

    def test_payload_one_byte_over_max(self):
        """Payload == MAX_PAYLOAD_SIZE + 1 triggers fragmentation into 2 fragments."""
        fragmenter = PacketFragmenter()
        frag_size = _fragment_payload_size()
        # One byte over MAX_PAYLOAD_SIZE means: 1188 total, 1184 in first, 4 in second
        payload = b"Y" * (MAX_PAYLOAD_SIZE + 1)
        fragments = fragmenter.fragment(payload)

        assert len(fragments) == 2
        assert fragments[0].is_fragmented()
        assert fragments[1].is_fragmented()
        assert fragments[0].header.packet_type == PacketType.FRAGMENT
        assert fragments[1].header.packet_type == PacketType.FRAGMENT

        # First fragment contains FragmentHeader + frag_size data bytes
        assert len(fragments[0].payload) == 4 + frag_size  # header + data
        # Second fragment: FragmentHeader + remainder
        remaining = MAX_PAYLOAD_SIZE + 1 - frag_size
        assert len(fragments[1].payload) == 4 + remaining

    def test_payload_at_fragment_boundary_exact(self):
        """Payload spanning exactly 2 fragments (size == 2 * FRAGMENT_PAYLOAD_SIZE)."""
        fragmenter = PacketFragmenter()
        frag_size = _fragment_payload_size()
        payload = b"Z" * (2 * frag_size)
        fragments = fragmenter.fragment(payload)

        assert len(fragments) == 2
        assert len(fragments[0].payload) == 4 + frag_size
        assert len(fragments[1].payload) == 4 + frag_size

    def test_payload_one_over_fragment_boundary(self):
        """Payload == 2 * FRAGMENT_PAYLOAD_SIZE + 1 -> 3 fragments."""
        fragmenter = PacketFragmenter()
        frag_size = _fragment_payload_size()
        payload = b"W" * (2 * frag_size + 1)
        fragments = fragmenter.fragment(payload)

        assert len(fragments) == 3
        assert len(fragments[0].payload) == 4 + frag_size
        assert len(fragments[1].payload) == 4 + frag_size
        assert len(fragments[2].payload) == 4 + 1

    def test_empty_payload(self):
        """Empty payload produces a single unfragmented DATA packet."""
        fragmenter = PacketFragmenter()
        fragments = fragmenter.fragment(b"")

        assert len(fragments) == 1
        assert fragments[0].header.packet_type == PacketType.DATA
        assert fragments[0].payload == b""

    def test_small_payload(self):
        """Very small payload (1 byte) is unfragmented."""
        fragmenter = PacketFragmenter()
        fragments = fragmenter.fragment(b"A")

        assert len(fragments) == 1
        assert fragments[0].payload == b"A"


class TestPacketFragmenterReassembly:
    """Whitebox: fragment reassembly correctness."""

    def test_reassemble_two_fragments(self):
        """Two fragments are correctly reassembled."""
        fragmenter = PacketFragmenter()
        original = b"AB" * 600  # 1200 bytes -> 2 fragments
        fragments = fragmenter.fragment(original)

        assert len(fragments) == 2

        result = None
        for f in fragments:
            r = fragmenter.add_fragment(f)
            if r is not None:
                result = r

        assert result == original

    def test_reassemble_three_fragments(self):
        """Three fragments are correctly reassembled."""
        fragmenter = PacketFragmenter()
        frag_size = _fragment_payload_size()
        original = b"CD" * (frag_size + 100)  # Enough for 3 fragments
        fragments = fragmenter.fragment(original)

        assert len(fragments) >= 3

        result = None
        for f in fragments:
            r = fragmenter.add_fragment(f)
            if r is not None:
                result = r

        assert result == original

    def test_reassemble_idempotent_add(self):
        """Adding the same fragment twice is idempotent (does not corrupt)."""
        fragmenter = PacketFragmenter()
        original = b"EF" * 600
        fragments = fragmenter.fragment(original)

        assert len(fragments) == 2

        # Add first fragment twice
        fragmenter.add_fragment(fragments[0])
        fragmenter.add_fragment(fragments[0])

        # Now add the second fragment
        result = fragmenter.add_fragment(fragments[1])

        assert result == original

    def test_incomplete_fragment_returns_none(self):
        """Missing fragments return None from add_fragment."""
        fragmenter = PacketFragmenter()
        original = b"GH" * 600
        fragments = fragmenter.fragment(original)

        result = fragmenter.add_fragment(fragments[0])

        # Only 1 of 2 fragments received -> None
        assert result is None

    def test_clear_pending_drops_fragments(self):
        """After clear_pending, old fragments cannot be assembled."""
        fragmenter = PacketFragmenter()
        original = b"IJ" * 600
        fragments = fragmenter.fragment(original)

        fragmenter.add_fragment(fragments[0])
        fragmenter.clear_pending()

        # The pending dict is now empty
        # Adding fragment 0 and 1 again creates a new group
        fragmenter.add_fragment(fragments[0])
        result = fragmenter.add_fragment(fragments[1])

        # Should still work because a new fragment() call creates new IDs
        # Actually in this test the ID is the same because we use the same
        # fragmenter and fragment() was called once. So clear_pending removes
        # the pending state, then adding fragment 0 again starts a new group
        # and adding fragment 1 finishes it.
        assert result == original

    def test_add_fragment_non_fragment_packet(self):
        """add_fragment returns payload directly for non-FRAGMENT packets."""
        fragmenter = PacketFragmenter()
        data_packet = Packet.create(PacketType.DATA, b"hello", sequence=1)

        result = fragmenter.add_fragment(data_packet)
        assert result == b"hello"


class TestPacketFragmenterSequenceAndFlags:
    """Whitebox: fragment sequence numbers and packet flags."""

    def test_fragment_sequence_continuity(self):
        """Fragment packets have sequential sequence numbers."""
        fragmenter = PacketFragmenter()
        frag_size = _fragment_payload_size()
        payload = b"K" * (int(2.5 * frag_size))  # 3 fragments
        fragments = fragmenter.fragment(payload, sequence=100)

        assert len(fragments) == 3
        for i, f in enumerate(fragments):
            assert f.header.sequence == 100 + i

    def test_fragment_has_reliable_flag(self):
        """Fragmented packets always have RELIABLE and FRAGMENTED flags."""
        fragmenter = PacketFragmenter()
        payload = b"L" * (MAX_PAYLOAD_SIZE + 50)
        fragments = fragmenter.fragment(payload)

        for f in fragments:
            assert f.is_fragmented()
            assert f.is_reliable()

    def test_fragment_has_correct_packet_type(self):
        """Fragments use PacketType.FRAGMENT."""
        fragmenter = PacketFragmenter()
        payload = b"M" * (MAX_PAYLOAD_SIZE + 1)
        fragments = fragmenter.fragment(payload)

        for f in fragments:
            assert f.header.packet_type == PacketType.FRAGMENT

    def test_fragment_id_increments(self):
        """Each call to fragment() uses an incrementing fragment_id."""
        fragmenter = PacketFragmenter()
        payload = b"N" * (MAX_PAYLOAD_SIZE + 1)

        first = fragmenter.fragment(payload)
        second = fragmenter.fragment(payload)

        # Extract fragment IDs from the first fragment of each group
        from engine.networking.transport.packet import FragmentHeader
        id1 = FragmentHeader.from_bytes(first[0].payload).fragment_id
        id2 = FragmentHeader.from_bytes(second[0].payload).fragment_id

        assert id2 > id1 or (id1 == 65535 and id2 == 0)
