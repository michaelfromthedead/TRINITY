"""Tests for PCG64 deterministic RNG (T-CC-0.15)."""

import pytest

from trinity.types import Fixed16, Fixed32, PCG64


class TestPCG64Creation:
    """Test PCG64 creation and initialization."""

    def test_default_seed(self):
        """Test default seed produces valid output."""
        rng = PCG64()
        val = rng.next_u32()
        assert 0 <= val <= 0xFFFFFFFF

    def test_explicit_seed(self):
        """Test explicit seed."""
        rng = PCG64(seed=12345)
        val = rng.next_u32()
        assert 0 <= val <= 0xFFFFFFFF

    def test_with_stream(self):
        """Test stream parameter produces different sequence."""
        rng1 = PCG64(seed=42, stream=1)
        rng2 = PCG64(seed=42, stream=2)
        assert rng1.next_u32() != rng2.next_u32()

    def test_from_seeds_single(self):
        """Test from_seeds with single seed."""
        rng = PCG64.from_seeds(42)
        val = rng.next_u32()
        assert 0 <= val <= 0xFFFFFFFF

    def test_from_seeds_multiple(self):
        """Test from_seeds combines multiple seeds."""
        rng1 = PCG64.from_seeds(1, 2, 3)
        rng2 = PCG64.from_seeds(1, 2, 4)
        assert rng1.next_u32() != rng2.next_u32()

    def test_from_state(self):
        """Test state serialization roundtrip."""
        rng1 = PCG64(seed=999)
        rng1.next_u32()
        rng1.next_u32()
        state = rng1.state

        rng2 = PCG64.from_state(state)
        assert rng1.next_u32() == rng2.next_u32()
        assert rng1.next_u32() == rng2.next_u32()


class TestPCG64Determinism:
    """Test PCG64 produces deterministic sequences."""

    def test_same_seed_same_sequence(self):
        """Test same seed produces identical sequence."""
        rng1 = PCG64(seed=42)
        rng2 = PCG64(seed=42)

        for _ in range(100):
            assert rng1.next_u32() == rng2.next_u32()

    def test_different_seeds_different_sequence(self):
        """Test different seeds produce different sequences."""
        rng1 = PCG64(seed=1)
        rng2 = PCG64(seed=2)

        vals1 = [rng1.next_u32() for _ in range(10)]
        vals2 = [rng2.next_u32() for _ in range(10)]
        assert vals1 != vals2

    def test_sequence_reproducible(self):
        """Test sequence is reproducible across runs."""

        def get_sequence(seed):
            rng = PCG64(seed)
            return [rng.next_u32() for _ in range(10)]

        seq1 = get_sequence(12345)
        seq2 = get_sequence(12345)
        assert seq1 == seq2

    def test_long_sequence_determinism(self):
        """Test determinism over long sequence."""
        rng1 = PCG64(seed=0xDEADBEEF)
        rng2 = PCG64(seed=0xDEADBEEF)

        for _ in range(10000):
            rng1.next_u32()
            rng2.next_u32()

        assert rng1.next_u32() == rng2.next_u32()


class TestPCG64U32:
    """Test 32-bit unsigned integer generation."""

    def test_next_u32_range(self):
        """Test next_u32 produces values in valid range."""
        rng = PCG64(seed=42)
        for _ in range(1000):
            val = rng.next_u32()
            assert 0 <= val <= 0xFFFFFFFF

    def test_next_u32_distribution(self):
        """Test next_u32 produces varied distribution."""
        rng = PCG64(seed=42)
        vals = [rng.next_u32() for _ in range(1000)]
        unique = len(set(vals))
        assert unique > 900  # High variance expected


class TestPCG64U64:
    """Test 64-bit unsigned integer generation."""

    def test_next_u64_range(self):
        """Test next_u64 produces values in valid range."""
        rng = PCG64(seed=42)
        for _ in range(100):
            val = rng.next_u64()
            assert 0 <= val <= 0xFFFFFFFFFFFFFFFF

    def test_next_u64_uses_two_u32(self):
        """Test next_u64 combines two u32 values."""
        rng1 = PCG64(seed=42)
        rng2 = PCG64(seed=42)

        u64 = rng1.next_u64()
        high = rng2.next_u32()
        low = rng2.next_u32()
        assert u64 == (high << 32) | low


class TestPCG64Float:
    """Test floating point generation."""

    def test_next_float_range(self):
        """Test next_float produces values in [0, 1)."""
        rng = PCG64(seed=42)
        for _ in range(1000):
            val = rng.next_float()
            assert 0.0 <= val < 1.0

    def test_next_float64_range(self):
        """Test next_float64 produces values in [0, 1)."""
        rng = PCG64(seed=42)
        for _ in range(1000):
            val = rng.next_float64()
            assert 0.0 <= val < 1.0

    def test_next_float64_higher_precision(self):
        """Test next_float64 has more precision than next_float."""
        rng = PCG64(seed=42)
        floats = {rng.next_float() for _ in range(10000)}

        rng = PCG64(seed=42)
        float64s = {rng.next_float64() for _ in range(10000)}

        # float64 should have more unique values due to higher precision
        assert len(float64s) >= len(floats)


class TestPCG64Int:
    """Test bounded integer generation."""

    def test_next_int_range(self):
        """Test next_int produces values in specified range."""
        rng = PCG64(seed=42)
        for _ in range(1000):
            val = rng.next_int(10, 20)
            assert 10 <= val <= 20

    def test_next_int_single_value(self):
        """Test next_int with low == high."""
        rng = PCG64(seed=42)
        for _ in range(10):
            assert rng.next_int(5, 5) == 5

    def test_next_int_negative_range(self):
        """Test next_int with negative values."""
        rng = PCG64(seed=42)
        for _ in range(100):
            val = rng.next_int(-10, -5)
            assert -10 <= val <= -5

    def test_next_int_crossing_zero(self):
        """Test next_int crossing zero."""
        rng = PCG64(seed=42)
        for _ in range(100):
            val = rng.next_int(-5, 5)
            assert -5 <= val <= 5

    def test_next_int_invalid_range(self):
        """Test next_int raises for invalid range."""
        rng = PCG64(seed=42)
        with pytest.raises(ValueError, match="low.*must be <= high"):
            rng.next_int(10, 5)

    def test_next_int_uniform_distribution(self):
        """Test next_int has reasonably uniform distribution."""
        rng = PCG64(seed=42)
        counts = {i: 0 for i in range(10)}

        for _ in range(10000):
            val = rng.next_int(0, 9)
            counts[val] += 1

        # Each bucket should have ~1000, allow ±200
        for count in counts.values():
            assert 800 < count < 1200


class TestPCG64Bool:
    """Test boolean generation."""

    def test_next_bool_values(self):
        """Test next_bool produces both True and False."""
        rng = PCG64(seed=42)
        vals = [rng.next_bool() for _ in range(100)]
        assert True in vals
        assert False in vals

    def test_next_bool_distribution(self):
        """Test next_bool has roughly 50/50 distribution."""
        rng = PCG64(seed=42)
        trues = sum(rng.next_bool() for _ in range(10000))
        assert 4500 < trues < 5500


class TestPCG64FixedPoint:
    """Test fixed-point number generation."""

    def test_next_fixed16_range(self):
        """Test next_fixed16 produces values in [0, 1)."""
        rng = PCG64(seed=42)
        for _ in range(100):
            val = rng.next_fixed16()
            assert isinstance(val, Fixed16)
            assert 0 <= val.as_float < 1.0

    def test_next_fixed32_range(self):
        """Test next_fixed32 produces values in [0, 1)."""
        rng = PCG64(seed=42)
        for _ in range(100):
            val = rng.next_fixed32()
            assert isinstance(val, Fixed32)
            assert 0 <= val.as_float < 1.0

    def test_fixed_determinism(self):
        """Test fixed-point generation is deterministic."""
        rng1 = PCG64(seed=42)
        rng2 = PCG64(seed=42)

        for _ in range(100):
            assert rng1.next_fixed16().raw == rng2.next_fixed16().raw
            assert rng1.next_fixed32().raw == rng2.next_fixed32().raw


class TestPCG64Shuffle:
    """Test list shuffling."""

    def test_shuffle_changes_order(self):
        """Test shuffle changes list order."""
        rng = PCG64(seed=42)
        items = list(range(10))
        original = items.copy()
        rng.shuffle(items)
        assert items != original

    def test_shuffle_preserves_elements(self):
        """Test shuffle preserves all elements."""
        rng = PCG64(seed=42)
        items = list(range(10))
        rng.shuffle(items)
        assert sorted(items) == list(range(10))

    def test_shuffle_deterministic(self):
        """Test shuffle is deterministic."""
        items1 = list(range(10))
        items2 = list(range(10))

        PCG64(seed=42).shuffle(items1)
        PCG64(seed=42).shuffle(items2)

        assert items1 == items2

    def test_shuffle_empty_list(self):
        """Test shuffle handles empty list."""
        rng = PCG64(seed=42)
        items = []
        rng.shuffle(items)
        assert items == []

    def test_shuffle_single_element(self):
        """Test shuffle handles single element."""
        rng = PCG64(seed=42)
        items = [1]
        rng.shuffle(items)
        assert items == [1]


class TestPCG64Choice:
    """Test random choice."""

    def test_choice_returns_element(self):
        """Test choice returns element from list."""
        rng = PCG64(seed=42)
        items = ["a", "b", "c", "d"]
        for _ in range(100):
            val = rng.choice(items)
            assert val in items

    def test_choice_deterministic(self):
        """Test choice is deterministic."""
        items = [1, 2, 3, 4, 5]
        rng1 = PCG64(seed=42)
        rng2 = PCG64(seed=42)

        for _ in range(100):
            assert rng1.choice(items) == rng2.choice(items)

    def test_choice_empty_list_raises(self):
        """Test choice raises for empty list."""
        rng = PCG64(seed=42)
        with pytest.raises(ValueError, match="empty"):
            rng.choice([])


class TestPCG64Fork:
    """Test RNG forking for parallel streams."""

    def test_fork_produces_different_sequence(self):
        """Test forked RNG produces different sequence."""
        rng = PCG64(seed=42)
        child = rng.fork(0)
        assert rng.next_u32() != child.next_u32()

    def test_fork_is_deterministic(self):
        """Test forking is deterministic."""
        rng1 = PCG64(seed=42)
        rng2 = PCG64(seed=42)

        child1 = rng1.fork(0)
        child2 = rng2.fork(0)

        for _ in range(100):
            assert child1.next_u32() == child2.next_u32()

    def test_different_fork_ids_different_sequences(self):
        """Test different fork IDs produce different sequences."""
        rng = PCG64(seed=42)
        child1 = rng.fork(1)
        child2 = rng.fork(2)

        vals1 = [child1.next_u32() for _ in range(10)]
        vals2 = [child2.next_u32() for _ in range(10)]
        assert vals1 != vals2


class TestPCG64Jump:
    """Test state jumping for parallel partitioning."""

    def test_jump_advances_state(self):
        """Test jump advances state by specified steps."""
        rng1 = PCG64(seed=42)
        rng2 = PCG64(seed=42)

        for _ in range(100):
            rng1.next_u32()

        rng2.jump(100)

        assert rng1.next_u32() == rng2.next_u32()

    def test_jump_deterministic(self):
        """Test jump is deterministic."""
        rng1 = PCG64(seed=42)
        rng2 = PCG64(seed=42)

        rng1.jump(1000)
        rng2.jump(1000)

        assert rng1.next_u32() == rng2.next_u32()

    def test_jump_zero(self):
        """Test jump(0) doesn't change state."""
        rng1 = PCG64(seed=42)
        rng2 = PCG64(seed=42)

        rng1.jump(0)

        assert rng1.next_u32() == rng2.next_u32()


class TestPCG64State:
    """Test state management."""

    def test_state_property(self):
        """Test state property returns tuple."""
        rng = PCG64(seed=42)
        state = rng.state
        assert isinstance(state, tuple)
        assert len(state) == 2
        assert isinstance(state[0], int)
        assert isinstance(state[1], int)

    def test_state_changes_on_advance(self):
        """Test state changes after generating values."""
        rng = PCG64(seed=42)
        state1 = rng.state
        rng.next_u32()
        state2 = rng.state
        assert state1 != state2


class TestPCG64Repr:
    """Test string representation."""

    def test_repr(self):
        """Test repr shows state in hex."""
        rng = PCG64(seed=42)
        rep = repr(rng)
        assert "PCG64" in rep
        assert "state=0x" in rep
        assert "inc=0x" in rep
