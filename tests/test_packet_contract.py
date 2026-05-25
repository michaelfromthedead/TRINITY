"""
Contract tests for packet primitives (Phase 1, Task 1.1).

Tests the PUBLIC CONTRACT of the packet module from a cleanroom perspective:
  - PacketType enum completeness and type safety
  - PacketHeader field access and binary serialization roundtrip
  - Packet creation, serialization, and payload integrity
  - Sequence number comparison with 16-bit wraparound semantics
  - PacketFlags enumeration and bitwise combination
  - FragmentHeader serialization and fragmentation/reassembly guarantees
  - Physical constants relationship (HEADER_SIZE, MTU, MAX_PAYLOAD_SIZE)

This is a CLEANROOM artifact. It tests the contract, not the implementation.
No internal state or private methods are accessed.
"""

from __future__ import annotations

import enum
import struct
import random

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
    HEADER_SIZE,
    sequence_greater_than,
    FragmentHeader,
    PacketFragmenter,
)


# =============================================================================
# 1. PacketType contract
# =============================================================================

class TestPacketTypeContract:
    """PacketType enum members, types, and value properties."""

    def test_packet_type_is_enum(self):
        """PacketType is an enum type (Enum or IntEnum)."""
        assert issubclass(PacketType, enum.Enum)

    def test_packet_type_core_members_exist(self):
        """PacketType exposes the required set of members."""
        expected = {"DATA", "ACK", "HEARTBEAT", "FRAGMENT",
                    "RELIABLE_DATA", "SEQUENCED_DATA"}
        actual = {m.name for m in PacketType}
        missing = expected - actual
        assert not missing, f"PacketType missing expected members: {missing}"

    def test_packet_type_values_are_valid(self):
        """Each PacketType member has a small non-negative integer value."""
        for member in PacketType:
            assert isinstance(member.value, int), f"{member.name}.value is not int"
            assert 0 <= member.value <= 255, f"{member.name}.value out of uint8 range"

    def test_packet_type_values_unique(self):
        """No two PacketType members share the same value."""
        values = [m.value for m in PacketType]
        assert len(values) == len(set(values)), "Duplicate PacketType values"

    def test_packet_type_roundtrip_all_members(self):
        """Every PacketType value survives PacketHeader serialization roundtrip."""
        for pt in PacketType:
            h = PacketHeader(packet_type=pt, sequence=1, size=10)
            restored = PacketHeader.from_bytes(h.to_bytes())
            assert restored.packet_type == pt, f"PacketType {pt} lost in roundtrip"

    def test_packet_type_create_all_members(self):
        """Every PacketType can create a Packet that roundtrips."""
        for pt in PacketType:
            p = Packet.create(pt, b"data", sequence=1)
            restored = Packet.from_bytes(p.to_bytes())
            assert restored.header.packet_type == pt
            assert restored.payload == b"data"


# =============================================================================
# 2. PacketHeader contract
# =============================================================================

class TestPacketHeaderContract:
    """PacketHeader construction, field contract, binary serialization."""

    def test_header_has_required_fields(self):
        """PacketHeader provides all documented fields."""
        h = PacketHeader(packet_type=PacketType.DATA)
        for field in ("packet_type", "flags", "sequence", "ack", "ack_bits", "size"):
            assert hasattr(h, field), f"Missing field: {field}"

    def test_header_field_defaults(self):
        """Numeric fields default to zero."""
        h = PacketHeader(packet_type=PacketType.DATA)
        assert h.flags == 0
        assert h.sequence == 0
        assert h.ack == 0
        assert h.ack_bits == 0
        assert h.size == 0

    def test_header_field_assignment(self):
        """All fields accept and return assigned values."""
        h = PacketHeader(
            packet_type=PacketType.ACK,
            flags=6,
            sequence=42000,
            ack=100,
            ack_bits=0xAABBCCDD,
            size=200,
        )
        assert h.packet_type == PacketType.ACK
        assert h.flags == 6
        assert h.sequence == 42000
        assert h.ack == 100
        assert h.ack_bits == 0xAABBCCDD
        assert h.size == 200

    def test_header_full_roundtrip(self):
        """to_bytes then from_bytes preserves every field exactly."""
        fields = dict(
            packet_type=PacketType.SEQUENCED_DATA,
            flags=0x1F,
            sequence=0xABCD,
            ack=0x1234,
            ack_bits=0x87654321,
            size=0xFFFF,
        )
        h = PacketHeader(**fields)
        data = h.to_bytes()
        assert len(data) == HEADER_SIZE, f"Expected {HEADER_SIZE}, got {len(data)}"
        restored = PacketHeader.from_bytes(data)
        for name, val in fields.items():
            assert getattr(restored, name) == val, f"{name} differs after roundtrip"

    def test_header_serialization_deterministic(self):
        """Same header always produces identical bytes."""
        h = PacketHeader(packet_type=PacketType.DATA, sequence=99, size=50)
        assert h.to_bytes() == h.to_bytes()

    def test_header_roundtrip_idempotent(self):
        """Serializing twice (roundtrip -> to_bytes) yields identical bytes."""
        h = PacketHeader(
            packet_type=PacketType.RELIABLE_DATA,
            flags=0x07,
            sequence=12345,
            ack=6789,
            ack_bits=0xDEADBEEF,
            size=500,
        )
        data_first = h.to_bytes()
        data_second = PacketHeader.from_bytes(data_first).to_bytes()
        assert data_first == data_second, "Idempotency violated"

    def test_header_size_constant_is_accurate(self):
        """HEADER_SIZE matches actual serialized byte length."""
        h = PacketHeader(packet_type=PacketType.DATA)
        assert len(h.to_bytes()) == HEADER_SIZE

    def test_header_from_bytes_rejects_short(self):
        """from_bytes raises ValueError for data shorter than HEADER_SIZE."""
        for n in [0, 1, HEADER_SIZE - 1, HEADER_SIZE // 2]:
            with pytest.raises(ValueError):
                PacketHeader.from_bytes(b"\x00" * n)

    def test_header_from_bytes_accepts_exact_length(self):
        """from_bytes succeeds with exactly HEADER_SIZE bytes."""
        h = PacketHeader.from_bytes(b"\x00" * HEADER_SIZE)
        assert isinstance(h, PacketHeader)

    def test_header_from_bytes_extra_bytes_ignored(self):
        """from_bytes reads only HEADER_SIZE bytes from longer input."""
        h = PacketHeader.from_bytes(b"\x00" * HEADER_SIZE + b"extra")
        assert isinstance(h, PacketHeader)


# =============================================================================
# 3. Packet contract
# =============================================================================

class TestPacketContract:
    """Packet creation, serialization, payload integrity."""

    def test_create_minimal(self):
        """Packet.create with only type and payload succeeds."""
        p = Packet.create(PacketType.DATA, b"hello")
        assert isinstance(p, Packet)
        assert isinstance(p.header, PacketHeader)
        assert p.payload == b"hello"

    def test_create_with_sequence(self):
        """Packet.create accepts a sequence parameter."""
        p = Packet.create(PacketType.DATA, b"data", sequence=999)
        assert p.header.sequence == 999

    def test_payload_integrity_various_sizes(self):
        """Payloads of various sizes survive to_bytes/from_bytes unchanged."""
        for size in [0, 1, 10, 100, 500, 1000, MAX_PAYLOAD_SIZE]:
            payload = b"\xAB" * size
            p = Packet.create(PacketType.DATA, payload)
            restored = Packet.from_bytes(p.to_bytes())
            assert restored.payload == payload, f"Failed at payload size {size}"

    def test_empty_payload(self):
        """Empty payload is preserved through roundtrip."""
        p = Packet.create(PacketType.RELIABLE_DATA, b"")
        restored = Packet.from_bytes(p.to_bytes())
        assert restored.payload == b""

    def test_all_packet_types_serialize(self):
        """Every PacketType creates a packet that survives roundtrip."""
        for pt in PacketType:
            p = Packet.create(pt, b"data", sequence=1)
            restored = Packet.from_bytes(p.to_bytes())
            assert restored.header.packet_type == pt
            assert restored.payload == b"data"

    def test_triple_roundtrip_idempotent(self):
        """Three consecutive roundtrips produce identical payload."""
        p = Packet.create(PacketType.RELIABLE_DATA, b"triple", sequence=5)
        for _ in range(3):
            p = Packet.from_bytes(p.to_bytes())
        assert p.payload == b"triple"

    def test_from_bytes_rejects_empty(self):
        """Packet.from_bytes raises on empty input."""
        with pytest.raises((ValueError, EOFError, struct.error)):
            Packet.from_bytes(b"")

    def test_from_bytes_rejects_too_short(self):
        """Packet.from_bytes raises on input shorter than HEADER_SIZE."""
        with pytest.raises((ValueError, EOFError, struct.error)):
            Packet.from_bytes(b"\x00" * (HEADER_SIZE - 1))


# =============================================================================
# 4. Sequence number comparison contract
# =============================================================================

class TestSequenceComparisonContract:
    """sequence_greater_than with 16-bit wraparound semantics."""

    SEQ_MAX = 65535
    HALF_RANGE = 32768

    def test_normal_ordering(self):
        """Follows natural order for sequences in the forward window."""
        assert sequence_greater_than(100, 50) is True
        assert sequence_greater_than(50, 100) is False

    def test_equal_returns_false_universally(self):
        """sequence_greater_than(n, n) is False for every n."""
        for n in [0, 1, self.HALF_RANGE, self.SEQ_MAX]:
            assert sequence_greater_than(n, n) is False, f"Failed at n={n}"

    def test_wraparound_forward(self):
        """After wraparound, small numbers are > large ones."""
        assert sequence_greater_than(0, self.SEQ_MAX) is True
        assert sequence_greater_than(1, self.SEQ_MAX) is True
        # diff between 65534 and 32767 is 32767 < half-range, not wraparound
        assert sequence_greater_than(0, self.SEQ_MAX - 100) is True
        assert sequence_greater_than(100, self.SEQ_MAX - 50) is True

    def test_wraparound_backward(self):
        """Before wraparound, large numbers are NOT > small ones."""
        assert sequence_greater_than(self.SEQ_MAX, 0) is False
        assert sequence_greater_than(self.SEQ_MAX, 1) is False
        # 40000 > 100? forward diff = 39900 > half-range, so NO — wraparound
        assert sequence_greater_than(40000, 100) is False

    def test_antisymmetry_for_clear_order(self):
        """For distinct sequences: a>b XOR b>a (one and only one holds)."""
        pairs = [
            (100, 50),
            (1, self.SEQ_MAX),
            (50000, 100),
            (0, self.SEQ_MAX - 1),
        ]
        for a, b in pairs:
            gt_ab = sequence_greater_than(a, b)
            gt_ba = sequence_greater_than(b, a)
            if a != b:
                assert gt_ab != gt_ba, f"Antisymmetry violated: ({a}, {b})"

    def test_half_boundary_deterministic(self):
        """At the half-range boundary, the answer is deterministic."""
        first = sequence_greater_than(self.HALF_RANGE, 0)
        for _ in range(20):
            assert sequence_greater_than(self.HALF_RANGE, 0) == first

    def test_type_stability(self):
        """Returns bool, never int or None."""
        for a, b in [(10, 5), (5, 10), (0, self.SEQ_MAX), (self.SEQ_MAX, 0)]:
            result = sequence_greater_than(a, b)
            assert isinstance(result, bool), f"Expected bool, got {type(result)}"


# =============================================================================
# 5. PacketFlags contract
# =============================================================================

class TestPacketFlagsContract:
    """PacketFlags enumeration contract."""

    def test_expected_flag_members_exist(self):
        """All flag constants are accessible."""
        for name in ("RELIABLE", "ORDERED", "PRIORITY_HIGH", "COMPRESSED", "FRAGMENTED"):
            assert hasattr(PacketFlags, name), f"Missing PacketFlags.{name}"

    def test_flags_are_powers_of_two(self):
        """Each flag (excluding NONE) is a power-of-two bitmask."""
        for flag in PacketFlags:
            v = int(flag)
            if v == 0:
                continue  # NONE=0 is a null flag, not a bitmask
            assert v > 0, f"Flag {flag} has non-positive value"
            assert (v & (v - 1)) == 0, f"Flag {flag} value {v} is not a power of two"

    def test_set_flag_and_check(self):
        """set_flag makes has_flag return True (NONE=0 is idempotent)."""
        h = PacketHeader(packet_type=PacketType.DATA)
        for flag in PacketFlags:
            if int(flag) == 0:
                continue  # NONE=0 is a no-op flag
            h.set_flag(flag)
            assert h.has_flag(flag), f"Flag {flag} not detected after set"

    def test_clear_flag_after_set(self):
        """clear_flag reverses set_flag."""
        h = PacketHeader(packet_type=PacketType.DATA)
        for flag in PacketFlags:
            h.set_flag(flag)
            h.clear_flag(flag)
            assert not h.has_flag(flag), f"Flag {flag} still set after clear"

    def test_multiple_flags_independent(self):
        """Setting/clearing one flag does not affect others."""
        h = PacketHeader(packet_type=PacketType.DATA)
        h.set_flag(PacketFlags.RELIABLE)
        h.set_flag(PacketFlags.ORDERED)
        h.set_flag(PacketFlags.COMPRESSED)
        h.clear_flag(PacketFlags.RELIABLE)
        assert h.has_flag(PacketFlags.ORDERED)
        assert h.has_flag(PacketFlags.COMPRESSED)
        assert not h.has_flag(PacketFlags.RELIABLE)

    def test_flags_survive_header_roundtrip(self):
        """All non-zero flags survive header to_bytes/from_bytes serialization."""
        for flag in PacketFlags:
            if int(flag) == 0:
                continue  # NONE=0 is always absent
            h = PacketHeader(packet_type=PacketType.DATA)
            h.set_flag(flag)
            restored = PacketHeader.from_bytes(h.to_bytes())
            assert restored.has_flag(flag), f"Flag {flag} lost in roundtrip"

    def test_flag_combinations_use_bitwise_or(self):
        """Flag values support bitwise OR for combination."""
        combined = PacketFlags.RELIABLE.value | PacketFlags.ORDERED.value
        assert combined & PacketFlags.RELIABLE.value
        assert combined & PacketFlags.ORDERED.value

    def test_packet_is_reliable_contract(self):
        """is_reliable() reflects the RELIABLE flag."""
        r = Packet.create(PacketType.DATA, flags=PacketFlags.RELIABLE)
        assert r.is_reliable() is True
        u = Packet.create(PacketType.DATA)
        assert u.is_reliable() is False

    def test_packet_is_fragmented_contract(self):
        """is_fragmented() reflects the FRAGMENTED flag."""
        f = Packet.create(PacketType.DATA, flags=PacketFlags.FRAGMENTED)
        assert f.is_fragmented() is True
        n = Packet.create(PacketType.DATA)
        assert n.is_fragmented() is False


# =============================================================================
# 6. FragmentHeader contract
# =============================================================================

class TestFragmentHeaderContract:
    """FragmentHeader serialization and field contract."""

    def test_fragment_header_has_fields(self):
        """FragmentHeader provides all documented fields."""
        fh = FragmentHeader(fragment_id=0, fragment_index=0, fragment_total=0)
        for field in ("fragment_id", "fragment_index", "fragment_total"):
            assert hasattr(fh, field), f"Missing field: {field}"

    def test_fragment_header_roundtrip(self):
        """FragmentHeader to_bytes/from_bytes preserves all fields."""
        fh = FragmentHeader(fragment_id=42, fragment_index=0, fragment_total=3)
        data = fh.to_bytes()
        assert len(data) == FragmentHeader.SIZE
        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 42
        assert restored.fragment_index == 0
        assert restored.fragment_total == 3

    def test_fragment_header_middle_and_last(self):
        """Middle and last fragment indices roundtrip correctly."""
        middle = FragmentHeader(fragment_id=100, fragment_index=2, fragment_total=5)
        data = middle.to_bytes()
        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 100
        assert restored.fragment_index == 2
        assert restored.fragment_total == 5

        last = FragmentHeader(fragment_id=200, fragment_index=9, fragment_total=10)
        restored = FragmentHeader.from_bytes(last.to_bytes())
        assert restored.fragment_index == 9
        assert restored.fragment_total == 10

    def test_fragment_header_max_values(self):
        """FragmentHeader handles maximum field values."""
        fh = FragmentHeader(
            fragment_id=0xFFFF, fragment_index=0xFF, fragment_total=0xFF,
        )
        data = fh.to_bytes()
        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 0xFFFF
        assert restored.fragment_index == 0xFF
        assert restored.fragment_total == 0xFF

    def test_fragment_header_zero_values(self):
        """FragmentHeader with all-zero fields works."""
        fh = FragmentHeader(fragment_id=0, fragment_index=0, fragment_total=0)
        data = fh.to_bytes()
        restored = FragmentHeader.from_bytes(data)
        assert restored.fragment_id == 0
        assert restored.fragment_index == 0
        assert restored.fragment_total == 0

    def test_fragment_header_short_data_rejected(self):
        """from_bytes raises ValueError for data shorter than SIZE."""
        for n in [0, 1, FragmentHeader.SIZE - 1]:
            with pytest.raises(ValueError):
                FragmentHeader.from_bytes(b"\x00" * n)

    def test_fragment_header_size_constant(self):
        """FragmentHeader.SIZE is a small positive integer."""
        assert isinstance(FragmentHeader.SIZE, int)
        assert FragmentHeader.SIZE > 0
        assert FragmentHeader.SIZE <= 8


# =============================================================================
# 7. Fragmentation and reassembly contract
# =============================================================================

class TestFragmentationContract:
    """Packet fragmentation and payload reassembly invariants."""

    def test_empty_payload_not_fragmented(self):
        """Empty payload produces a single unfragmented packet."""
        f = PacketFragmenter()
        packets = f.fragment(b"")
        assert len(packets) == 1

    def test_small_payload_not_fragmented(self):
        """Payload up to MAX_PAYLOAD_SIZE is not fragmented."""
        for size in [1, 100, MAX_PAYLOAD_SIZE]:
            f = PacketFragmenter()
            payload = b"x" * size
            packets = f.fragment(payload)
            assert len(packets) == 1, f"Failed at size {size}"
            assert packets[0].payload == payload

    def test_one_byte_over_max_produces_two_fragments(self):
        """Exactly MAX_PAYLOAD_SIZE + 1 produces at least 2 fragments."""
        f = PacketFragmenter()
        packets = f.fragment(b"x" * (MAX_PAYLOAD_SIZE + 1))
        assert len(packets) >= 2

    def test_reassembly_in_order(self):
        """In-order fragment reassembly reproduces original payload."""
        f = PacketFragmenter()
        original = b"in order reassembly verification " * 200
        packets = f.fragment(original)
        result = None
        for p in packets:
            result = f.add_fragment(p)
        assert result == original, "In-order reassembly failed"

    def test_reassembly_out_of_order(self):
        """Reverse-order fragment reassembly reproduces original."""
        f = PacketFragmenter()
        original = b"reverse order reassembly check " * 150
        packets = f.fragment(original)
        result = None
        for p in reversed(packets):
            result = f.add_fragment(p)
        assert result == original, "Reverse-order reassembly failed"

    def test_reassembly_random_order(self):
        """Random-order fragment reassembly reproduces original."""
        f = PacketFragmenter()
        original = b"shuffled reassembly payload pattern " * 80
        packets = f.fragment(original)
        rng = random.Random(42)
        shuffled = list(packets)
        rng.shuffle(shuffled)
        result = None
        for p in shuffled:
            result = f.add_fragment(p)
        assert result == original, "Random-order reassembly failed"

    def test_incomplete_reassembly_returns_none(self):
        """Partial fragment set returns None, not partial payload."""
        f = PacketFragmenter()
        original = b"x" * (MAX_PAYLOAD_SIZE * 2 + 1)
        packets = f.fragment(original)
        assert len(packets) > 2
        for p in packets[:2]:
            assert f.add_fragment(p) is None, "Partial set must return None"

    def test_full_reassembly_after_partial_returns_original(self):
        """Completing fragments after a partial add returns the full payload."""
        f = PacketFragmenter()
        original = b"x" * (MAX_PAYLOAD_SIZE + 100)
        packets = f.fragment(original)
        # Add first fragment only (incomplete)
        f.add_fragment(packets[0])
        # Now add the rest
        result = None
        for p in packets[1:]:
            result = f.add_fragment(p)
        assert result == original, "Full reassembly after partial failed"

    def test_non_fragment_packet_passthrough(self):
        """add_fragment returns payload immediately for non-FRAGMENT packets."""
        f = PacketFragmenter()
        p = Packet.create(PacketType.DATA, b"passthrough")
        assert f.add_fragment(p) == b"passthrough"

    def test_heartbeat_packet_passthrough(self):
        """add_fragment returns empty payload for HEARTBEAT (non-FRAGMENT)."""
        f = PacketFragmenter()
        hb = Packet.create(PacketType.HEARTBEAT, b"", sequence=1)
        assert f.add_fragment(hb) == b""

    def test_consecutive_fragment_groups_reassemble(self):
        """Multiple fragment groups processed sequentially with one fragmenter."""
        f = PacketFragmenter()
        for char, size_offset in [(b"A", 50), (b"B", 100), (b"C", 150)]:
            original = char * (MAX_PAYLOAD_SIZE + size_offset)
            packets = f.fragment(original)
            result = None
            for p in packets:
                result = f.add_fragment(p)
            assert result == original, f"Group {char} failed"

    def test_various_payload_sizes_reassemble(self):
        """Fragmentation works correctly at various payload sizes."""
        f = PacketFragmenter()
        sizes = [
            MAX_PAYLOAD_SIZE + 1,
            MAX_PAYLOAD_SIZE * 2,
            MAX_PAYLOAD_SIZE * 2 + 1,
            MAX_PAYLOAD_SIZE * 3 - 1,
            MAX_PAYLOAD_SIZE * 3,
        ]
        for size in sizes:
            fragmenter = PacketFragmenter()
            original = b"x" * size
            packets = fragmenter.fragment(original)
            result = None
            for p in packets:
                result = fragmenter.add_fragment(p)
            assert result == original, f"Reassembly failed at size {size}"


# =============================================================================
# 8. Physical constants contract
# =============================================================================

class TestConstantsContract:
    """Published constants satisfy documented relationships."""

    def test_header_size_reasonable(self):
        """HEADER_SIZE is small (8-24 bytes typical)."""
        assert isinstance(HEADER_SIZE, int)
        assert 8 <= HEADER_SIZE <= 24

    def test_mtu_reasonable(self):
        """MTU is in typical network range (500-1500)."""
        assert isinstance(MTU, int)
        assert 500 <= MTU <= 1500

    def test_max_payload_size_positive_and_less_than_mtu(self):
        """MAX_PAYLOAD_SIZE is positive and fits within MTU."""
        assert isinstance(MAX_PAYLOAD_SIZE, int)
        assert MAX_PAYLOAD_SIZE > 0
        assert MAX_PAYLOAD_SIZE < MTU

    def test_total_size_relationship(self):
        """Packet.total_size == HEADER_SIZE + len(payload)."""
        for plen in (0, 1, 50, 500, MAX_PAYLOAD_SIZE):
            p = Packet.create(PacketType.DATA, b"x" * plen)
            assert p.total_size == HEADER_SIZE + plen, f"Failed at payload size {plen}"
