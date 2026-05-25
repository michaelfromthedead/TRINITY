"""
WHITEBOX tests for engine/networking/transport/packet.py.

WHITEBOX coverage plan (gaps NOT covered by blackbox or test_packet_header_edge_cases):
  [PacketHeader.to_bytes overflow truncation]
    Path A1:  values exceeding field bit widths are masked during serialization
              -> test_header_truncates_overflow_via_bitmask

  [Packet.from_bytes edge cases]
    Path B1:  header.size > actual remaining data -> truncated payload (Python slices silently)
              -> test_from_bytes_truncated_payload
    Path B2:  extra trailing bytes beyond HEADER_SIZE + header.size are ignored
              -> test_from_bytes_ignores_trailing_bytes

  [Packet dataclass defaults]
    Path C1:  timestamp is set on creation (float, recent)
              -> test_packet_timestamp_set_on_create
    Path C2:  retransmit_count defaults to 0
              -> test_retransmit_count_defaults_to_zero

  [PacketFragmenter.fragment sequence numbering]
    Path D1:  fragments receive sequential sequence numbers from base
              -> test_fragment_sequences_sequential
    Path D2:  fragment_id wraps at 0xFFFF -> 0x0000
              -> test_fragment_id_wraps_at_0xFFFF

  [PacketFragmenter.add_fragment internal paths]
    Path E1:  duplicate fragment index overwrites (last-write-wins in dict)
              -> test_duplicate_fragment_index_overwrites
    Path E2:  missing middle fragment (indices 0 and 2 present, 1 absent)
              -> test_missing_middle_fragment_returns_none_until_complete
    Path E3:  FRAGMENT packet with payload too short for FragmentHeader -> ValueError
              -> test_add_fragment_invalid_header_raises

  [sequence_difference internal branches]
    Path F1:  diff > half  (s1 >> s2, wraparound forward)
              -> test_sequence_difference_wraparound_forward
    Path F2:  diff < -half (s1 << s2, wraparound backward)
              -> test_sequence_difference_wraparound_backward
    Path F3:  custom max_value parameter propagates correctly
              -> test_sequence_difference_custom_max_value
    Path F4:  diff exactly at half boundary
              -> test_sequence_difference_half_boundary

  [PacketHeader struct format invariant]
    Path G1:  _FORMAT struct size matches HEADER_SIZE constant
              -> test_struct_format_size_matches_header_size
"""

from __future__ import annotations

import struct
import time

import pytest

from engine.networking.transport.packet import (
    Packet,
    PacketHeader,
    PacketType,
    PacketFlags,
    FragmentHeader,
    PacketFragmenter,
    HEADER_SIZE,
    MAX_PAYLOAD_SIZE,
    sequence_greater_than,
    sequence_difference,
)
from engine.networking.config import DEFAULT_CONFIG


# =============================================================================
# A. PacketHeader overflow truncation (to_bytes bitmask behavior)
# =============================================================================

class TestHeaderOverflowTruncation:
    """PacketHeader.to_bytes masks field values exceeding wire-format width."""

    def test_header_truncates_overflow_via_bitmask(self):
        """Values > field width are truncated by to_bytes bitmasks, not wrapped
        in the struct pack call itself -- the bitmasks are the guard."""
        header = PacketHeader(
            packet_type=PacketType.DATA,
            flags=0,
            sequence=0x1FFFF,      # exceeds 16 bits
            ack=0x2FFFF,           # exceeds 16 bits
            ack_bits=0x1FFFFFFFF,  # exceeds 32 bits
            size=0x3FFFF,          # exceeds 16 bits
        )
        data = header.to_bytes()
        restored = PacketHeader.from_bytes(data)

        # Each field was &-masked by to_bytes before packing:
        #   0x1FFFF & 0xFFFF = 0xFFFF,
        #   0x2FFFF & 0xFFFF = 0xFFFF,
        #   0x1FFFFFFFF & 0xFFFFFFFF = 0xFFFFFFFF,
        #   0x3FFFF & 0xFFFF = 0xFFFF
        assert restored.sequence == 0xFFFF
        assert restored.ack == 0xFFFF
        assert restored.ack_bits == 0xFFFFFFFF
        assert restored.size == 0xFFFF

    def test_header_truncates_negative_values(self):
        """Negative Python ints produce defined truncation through bitmask."""
        header = PacketHeader(
            packet_type=PacketType.DATA,
            flags=0,
            sequence=-1,
            ack=-1,
            ack_bits=-1,
            size=-1,
        )
        data = header.to_bytes()
        restored = PacketHeader.from_bytes(data)

        # -1 in Python is infinite 1-bits; -1 & 0xFFFF == 0xFFFF, etc.
        assert restored.sequence == 0xFFFF
        assert restored.ack == 0xFFFF
        assert restored.ack_bits == 0xFFFFFFFF
        assert restored.size == 0xFFFF


# =============================================================================
# B. Packet.from_bytes edge cases (payload slicing behavior)
# =============================================================================

class TestPacketFromBytesEdgeCases:
    """Packet.from_bytes internal payload-slicing edge cases."""

    def test_from_bytes_truncated_payload(self):
        """When header.size exceeds remaining bytes, payload is silently truncated."""
        # Construct a header claiming 100 bytes of payload
        header = PacketHeader(
            packet_type=PacketType.DATA,
            size=100,
        )
        # Only append 10 bytes of actual payload
        header_bytes = header.to_bytes()
        truncated_bytes = header_bytes + b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a'

        packet = Packet.from_bytes(truncated_bytes)

        # Payload should be whatever bytes were available (not 100)
        assert packet.header.size == 100
        assert len(packet.payload) == 10
        assert packet.payload == b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a'

    def test_from_bytes_ignores_trailing_bytes(self):
        """Extra bytes beyond HEADER_SIZE + header.size are ignored."""
        header = PacketHeader(
            packet_type=PacketType.DATA,
            size=5,
        )
        header_bytes = header.to_bytes()
        # Append 5 payload bytes + 10 trailing garbage bytes
        full_bytes = header_bytes + b'hello' + b'TRAILING_GARBAGE'

        packet = Packet.from_bytes(full_bytes)

        assert packet.header.size == 5
        assert packet.payload == b'hello'
        # Trailing bytes are simply not consumed
        assert len(packet.payload) == 5

    def test_from_bytes_zero_size_payload_edge(self):
        """header.size = 0 with zero further bytes after header is valid."""
        header = PacketHeader(packet_type=PacketType.DATA, size=0)
        header_bytes = header.to_bytes()
        assert len(header_bytes) == HEADER_SIZE

        packet = Packet.from_bytes(header_bytes)

        assert packet.header.size == 0
        assert packet.payload == b''


# =============================================================================
# C. Packet dataclass field defaults
# =============================================================================

class TestPacketDataclassDefaults:
    """Packet fields that are NOT set by create() factory methods."""

    def test_packet_timestamp_set_on_create(self):
        """Packet.timestamp is populated on construction (not zero)."""
        before = time.time()
        packet = Packet.create(PacketType.DATA, b'x')
        after = time.time()

        assert before <= packet.timestamp <= after, \
            f"timestamp {packet.timestamp} should be in [{before}, {after}]"

    def test_retransmit_count_defaults_to_zero(self):
        """Packet.retransmit_count starts at 0."""
        packet = Packet.create(PacketType.DATA, b'x')
        assert packet.retransmit_count == 0

    def test_retransmit_count_mutable(self):
        """retransmit_count can be incremented after creation."""
        packet = Packet.create(PacketType.DATA, b'x')
        packet.retransmit_count += 1
        assert packet.retransmit_count == 1
        packet.retransmit_count += 1
        assert packet.retransmit_count == 2

    def test_payload_defaults_to_empty(self):
        """Packet payload defaults to b'' when not provided."""
        packet = Packet.create(PacketType.DATA)
        assert packet.payload == b''


# =============================================================================
# D. PacketFragmenter.fragment internal behavior
# =============================================================================

class TestFragmenterFragmentInternals:
    """Internal behavior of PacketFragmenter.fragment() beyond contract tests."""

    def test_fragment_sequences_sequential(self):
        """Fragments receive sequential sequence numbers starting from base."""
        fragmenter = PacketFragmenter()
        payload = b'x' * (MAX_PAYLOAD_SIZE + 50)
        packets = fragmenter.fragment(payload, sequence=100)

        assert len(packets) >= 2
        for i, p in enumerate(packets):
            assert p.header.sequence == 100 + i, \
                f"Fragment {i} expected seq {100 + i}, got {p.header.sequence}"

    def test_fragment_sequences_zero_base(self):
        """Fragments with sequence=0 produce sequences 0, 1, 2, ..."""
        fragmenter = PacketFragmenter()
        payload = b'x' * (MAX_PAYLOAD_SIZE + 50)
        packets = fragmenter.fragment(payload, sequence=0)

        assert len(packets) >= 2
        for i, p in enumerate(packets):
            assert p.header.sequence == i

    def test_fragment_type_and_flags_set(self):
        """Each fragment has type FRAGMENT and flags FRAGMENTED | RELIABLE."""
        fragmenter = PacketFragmenter()
        payload = b'x' * (MAX_PAYLOAD_SIZE + 1)
        packets = fragmenter.fragment(payload)

        for p in packets:
            assert p.header.packet_type == PacketType.FRAGMENT
            assert p.is_fragmented()
            assert p.is_reliable()

    def test_fragment_id_wraps_at_0xFFFF(self):
        """_fragment_id wraps from 0xFFFF to 0x0000."""
        fragmenter = PacketFragmenter()
        # Force _fragment_id to just below wraparound
        fragmenter._fragment_id = 0xFFFF

        payload = b'x' * (MAX_PAYLOAD_SIZE + 1)
        packets = fragmenter.fragment(payload)

        # Extract fragment_id from fragment header
        fh = FragmentHeader.from_bytes(packets[0].payload)
        assert fh.fragment_id == 0x0000, \
            f"Expected fragment_id 0 after wraparound, got {fh.fragment_id}"

    def test_fragment_id_increments_sequentially(self):
        """Each call to fragment() gets the next fragment_id."""
        fragmenter = PacketFragmenter()
        payload = b'x' * (MAX_PAYLOAD_SIZE + 1)

        ids = []
        for _ in range(5):
            packets = fragmenter.fragment(payload)
            fh = FragmentHeader.from_bytes(packets[0].payload)
            ids.append(fh.fragment_id)

        # Each call should get a unique, monotonically increasing id
        assert len(set(ids)) == 5, "fragment_ids should be unique across calls"
        for i in range(1, len(ids)):
            assert ids[i] > ids[i - 1], \
                f"fragment_id should increase: {ids[i]} <= {ids[i - 1]}"

    def test_fragment_payload_starts_with_fragment_header(self):
        """Fragment packet payload begins with a valid FragmentHeader,
        followed by the chunk data."""
        fragmenter = PacketFragmenter()
        payload = b'x' * (MAX_PAYLOAD_SIZE + 50)
        packets = fragmenter.fragment(payload)

        for i, p in enumerate(packets):
            # First FragmentHeader.SIZE bytes should parse as FragmentHeader
            fh = FragmentHeader.from_bytes(p.payload[:FragmentHeader.SIZE])
            assert fh.fragment_index == i
            assert fh.fragment_total == len(packets)

            # The rest should be non-empty data
            chunk = p.payload[FragmentHeader.SIZE:]
            assert len(chunk) > 0

            # Total reassembled chunks should equal original
            original_chunks = [
                payload[i * PacketFragmenter.FRAGMENT_PAYLOAD_SIZE:
                        (i + 1) * PacketFragmenter.FRAGMENT_PAYLOAD_SIZE]
                for i in range(len(packets))
            ]
            # Last chunk may be shorter
            expected_chunk = original_chunks[i]
            assert chunk == expected_chunk, \
                f"Fragment {i} data mismatch: {len(chunk)} vs {len(expected_chunk)}"


# =============================================================================
# E. PacketFragmenter.add_fragment internal paths
# =============================================================================

class TestFragmenterAddFragmentInternals:
    """Internal code paths of add_fragment that contract tests don't cover."""

    def test_duplicate_fragment_index_overwrites(self):
        """Adding a duplicate fragment index overwrites the previous data
        (last-write-wins in the dict store)."""
        fragmenter = PacketFragmenter()
        # Need enough payload for 3+ fragments.
        # FRAGMENT_PAYLOAD_SIZE = MAX_PAYLOAD_SIZE - FragmentHeader.SIZE = 1184.
        # 2 * 1184 = 2368; payload > 2368 gives 3 fragments.
        payload = b'X' * 3000
        packets = fragmenter.fragment(payload)
        assert len(packets) >= 3, f"Need 3+ fragments, got {len(packets)}"

        # First, reassemble normally
        fragmenter2 = PacketFragmenter()
        original_result = None
        for p in packets:
            original_result = fragmenter2.add_fragment(p)
        assert original_result == payload

        # Now, re-fragment the same payload and corrupt the middle fragment
        packets2 = fragmenter.fragment(payload)
        # Replace fragment index 1 with a different value
        frag_header = FragmentHeader.from_bytes(packets2[1].payload)
        chunk = packets2[1].payload[FragmentHeader.SIZE:]
        corrupted_chunk = b'CORRUPTED_' + chunk[10:]
        corrupted_payload = frag_header.to_bytes() + corrupted_chunk
        corrupted_packet = Packet(
            header=PacketHeader(
                packet_type=PacketType.FRAGMENT,
                flags=PacketFlags.FRAGMENTED | PacketFlags.RELIABLE,
                size=len(corrupted_payload),
            ),
            payload=corrupted_payload,
        )

        # Reassemble: add fragment 0, then corrupted fragment 1, then normal fragment 1
        result = fragmenter.add_fragment(packets2[0])
        assert result is None  # incomplete

        result = fragmenter.add_fragment(corrupted_packet)
        assert result is None  # still incomplete (needs fragment 2)

        # Overwrite fragment 1 with the original data
        result = fragmenter.add_fragment(packets2[1])
        assert result is None  # still incomplete (needs fragment 2)

        # Now add fragment 2
        result = fragmenter.add_fragment(packets2[2])
        assert result == payload  # correct payload because fragment 1 was overwritten

    def test_missing_middle_fragment_returns_none_until_complete(self):
        """Adding fragments 0 and 2 (skipping 1) returns None. Adding
        fragment 1 then completes reassembly."""
        fragmenter = PacketFragmenter()
        # FRAGMENT_PAYLOAD_SIZE = 1184. Need payload > 2*1184 = 2368 for 3+ fragments
        payload = b'M' * 3000
        packets = fragmenter.fragment(payload)
        assert len(packets) >= 3, \
            f"Need at least 3 fragments, got {len(packets)} (payload={len(payload)}, " \
            f"FRAGMENT_PAYLOAD_SIZE={PacketFragmenter.FRAGMENT_PAYLOAD_SIZE})"

        n_frags = len(packets)

        # Add first fragment
        result = fragmenter.add_fragment(packets[0])
        assert result is None

        # Add LAST fragment (skip middle ones)
        result = fragmenter.add_fragment(packets[-1])
        assert result is None

        # Reassembler still has 2 of n_frags
        frag_id = FragmentHeader.from_bytes(packets[0].payload).fragment_id
        assert len(fragmenter._pending_fragments[frag_id]) == 2
        assert fragmenter._fragment_totals[frag_id] == n_frags

        # Now add all remaining middle fragments in reverse order
        for i in range(n_frags - 2, 0, -1):
            result = fragmenter.add_fragment(packets[i])
            if i == 1:
                # Last missing piece — should complete
                assert result == payload, \
                    f"Reassembly should complete at fragment 1, got result length {len(result) if result else 0}"
            else:
                assert result is None

    def test_add_fragment_invalid_header_raises(self):
        """FRAGMENT packet with payload shorter than FragmentHeader.SIZE
        raises ValueError (no try/except in add_fragment)."""
        fragmenter = PacketFragmenter()

        # Packet of type FRAGMENT but payload is only 2 bytes (need 4)
        short_payload = b'\x00\x01'
        packet = Packet(
            header=PacketHeader(
                packet_type=PacketType.FRAGMENT,
                size=len(short_payload),
            ),
            payload=short_payload,
        )

        with pytest.raises(ValueError, match="Need 4 bytes for fragment header"):
            fragmenter.add_fragment(packet)

    def test_add_fragment_empty_payload_raises(self):
        """FRAGMENT packet with empty payload raises ValueError."""
        fragmenter = PacketFragmenter()

        packet = Packet(
            header=PacketHeader(
                packet_type=PacketType.FRAGMENT,
                size=0,
            ),
            payload=b'',
        )

        with pytest.raises(ValueError, match="Need 4 bytes for fragment header"):
            fragmenter.add_fragment(packet)


# =============================================================================
# F. sequence_difference internal branches
# =============================================================================

class TestSequenceDifferenceInternals:
    """sequence_difference internal branch coverage."""

    def test_sequence_difference_basic_positive(self):
        """Normal case: s1 > s2 returns positive difference."""
        assert sequence_difference(100, 50) == 50

    def test_sequence_difference_basic_negative(self):
        """Normal case: s1 < s2 returns negative difference."""
        assert sequence_difference(50, 100) == -50

    def test_sequence_difference_equal(self):
        """Equal sequence numbers return 0."""
        assert sequence_difference(0, 0) == 0
        assert sequence_difference(32768, 32768) == 0
        assert sequence_difference(DEFAULT_CONFIG.MAX_SEQUENCE, DEFAULT_CONFIG.MAX_SEQUENCE) == 0

    def test_sequence_difference_wraparound_forward(self):
        """diff > half: s1 wraps past max to a small value while s2 is near max.
        E.g., s1=100, s2=50000: diff=-49900 < -half => corrected to +15636.
        This tests the `elif diff < -half` branch."""
        # s1=100, s2=50000: diff = -49900 < -32767 => diff += 65536 = 15636
        diff = sequence_difference(100, 50000)
        assert diff == 15636, f"Expected 15636, got {diff}"

    def test_sequence_difference_wraparound_backward(self):
        """diff < -half: s1 is near max while s2 wrapped past.
        E.g., s1=50000, s2=100: diff=49900 > 32767 => diff -= 65536 = -15636.
        This tests the `if diff > half` branch."""
        # s1=50000, s2=100: diff = 49900 > 32767 => diff -= 65536 = -15636
        diff = sequence_difference(50000, 100)
        assert diff == -15636, f"Expected -15636, got {diff}"

    def test_sequence_difference_custom_max_value(self):
        """Custom max_value parameter is used instead of default."""
        max_val = 1023  # 10-bit sequence space
        half = max_val // 2  # 511

        # s1=600, s2=100: diff=500 < half=511 => no correction, result=500
        assert sequence_difference(600, 100, max_value=max_val) == 500

        # s1=100, s2=900: diff=-800 < -half=-511 => diff += 1024 = 224
        assert sequence_difference(100, 900, max_value=max_val) == 224

        # s1=900, s2=100: diff=800 > half=511 => diff -= 1024 = -224
        assert sequence_difference(900, 100, max_value=max_val) == -224

        # s1=0, s2=1023: diff=-1023 < -511 => diff += 1024 = 1
        assert sequence_difference(0, 1023, max_value=max_val) == 1

    def test_sequence_difference_half_boundary(self):
        """Behavior when diff is exactly at the half boundary."""
        half = DEFAULT_CONFIG.MAX_SEQUENCE // 2  # 32767

        # diff == half: not > half, not < -half => raw diff returned
        # s1=half, s2=0: diff = 32767
        assert sequence_difference(half, 0) == half

        # diff == -half: not > half, not < -half => raw diff returned
        # s1=0, s2=half: diff = -32767
        assert sequence_difference(0, half) == -half

        # Just over half triggers wraparound
        # s1=half+1, s2=0: diff=32768 > 32767 => diff -= 65536 = -32768
        assert sequence_difference(half + 1, 0) == -32768

        # Just under -half does NOT trigger (boundary is strict)
        # s1=0, s2=half: diff=-32767, not < -half => stays -32767
        assert sequence_difference(0, half) == -32767


# =============================================================================
# G. Structural invariants
# =============================================================================

class TestStructuralInvariants:
    """Compile-time invariants the code silently relies on."""

    def test_struct_format_size_matches_header_size(self):
        """PacketHeader._FORMAT '!BBHHIH' must produce exactly HEADER_SIZE bytes."""
        # B=1, B=1, H=2, H=2, I=4, H=2 = 12 bytes
        expected_size = struct.calcsize('!BBHHIH')
        assert expected_size == HEADER_SIZE, \
            f"struct format calcsize {expected_size} != HEADER_SIZE {HEADER_SIZE}"
        assert HEADER_SIZE == DEFAULT_CONFIG.PACKET_HEADER_SIZE

    def test_fragment_payload_size_positive(self):
        """FRAGMENT_PAYLOAD_SIZE must be positive for fragmentation to work."""
        assert PacketFragmenter.FRAGMENT_PAYLOAD_SIZE > 0, \
            "FRAGMENT_PAYLOAD_SIZE must be > 0"
        assert PacketFragmenter.FRAGMENT_PAYLOAD_SIZE == MAX_PAYLOAD_SIZE - FragmentHeader.SIZE

    def test_sequence_difference_default_identity(self):
        """sequence_difference and sequence_greater_than agree:
        sequence_greater_than(s1, s2) iff sequence_difference(s1, s2) > 0."""
        test_pairs = [
            (0, 0), (1, 0), (0, 1),
            (65535, 0), (0, 65535), (65535, 65535),
            (32768, 0), (0, 32768),
            (50000, 100), (100, 50000),
            (1, 65535), (65535, 1),
        ]
        for s1, s2 in test_pairs:
            diff = sequence_difference(s1, s2)
            gt = sequence_greater_than(s1, s2)
            assert (diff > 0) == gt, \
                f"Mismatch for ({s1}, {s2}): diff={diff}, greater_than={gt}"
