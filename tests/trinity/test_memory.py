"""
Tests for Memory decorators (memory.py) — built on Ops.

Tests the 12 memory decorators:
    @pooled, @packed, @aligned, @arena, @flyweight, @intern,
    @generations, @copy_on_write, @inline_array, @budget,
    @allocator, @atomic

Each test verifies:
1. Steps are applied (_applied_steps populated with correct Ops)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Runtime behavior (methods, state) works
5. Introspection via decompose/expand
"""

import pytest

from trinity.decorators.memory import (
    AlignedConfig,
    AllocatorConfig,
    ArenaConfig,
    BudgetConfig,
    InlineArrayConfig,
    PackedConfig,
    PoolConfig,
    aligned,
    allocator,
    arena,
    atomic,
    budget,
    copy_on_write,
    flyweight,
    generations,
    inline_array,
    intern,
    packed,
    pooled,
)
from trinity.decorators.ops import Op, Step, decompose, expand

# =============================================================================
# STEP COMPOSITION TESTS (Ops introspection)
# =============================================================================


class TestStepCompositions:
    """Every memory decorator must decompose into correct Ops."""

    def test_pooled_steps(self):
        steps = decompose(pooled)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.HOOK in ops
        assert Op.REGISTER in ops
        hooks = [s for s in steps if s.op is Op.HOOK]
        assert len(hooks) == 2  # on_create + on_destroy

    def test_packed_steps(self):
        steps = decompose(packed)
        assert len(steps) == 1
        assert steps[0].op is Op.TAG

    def test_aligned_steps(self):
        steps = decompose(aligned)
        assert len(steps) == 1
        assert steps[0].op is Op.TAG

    def test_arena_steps(self):
        steps = decompose(arena)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.HOOK in ops
        assert len(steps) == 2

    def test_flyweight_steps(self):
        steps = decompose(flyweight)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.INTERCEPT in ops
        assert Op.REGISTER in ops
        assert len(steps) == 3

    def test_intern_steps(self):
        steps = decompose(intern)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.INTERCEPT in ops
        assert len(steps) == 2

    def test_generations_steps(self):
        steps = decompose(generations)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.TRACK in ops
        assert len(steps) == 2

    def test_copy_on_write_steps(self):
        steps = decompose(copy_on_write)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.INTERCEPT in ops
        assert len(steps) == 2

    def test_inline_array_steps(self):
        steps = decompose(inline_array)
        assert len(steps) == 1
        assert steps[0].op is Op.TAG

    def test_budget_steps(self):
        steps = decompose(budget)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.VALIDATE in ops
        assert len(steps) == 2

    def test_allocator_steps(self):
        steps = decompose(allocator)
        assert len(steps) == 1
        assert steps[0].op is Op.TAG

    def test_atomic_steps(self):
        steps = decompose(atomic)
        ops = [s.op for s in steps]
        assert Op.INTERCEPT in ops
        assert Op.TAG in ops
        assert len(steps) == 2


class TestExpandReadable:
    """expand() should produce human-readable step strings."""

    def test_expand_pooled(self):
        result = expand(pooled)
        assert "TAG" in result
        assert "HOOK" in result
        assert "REGISTER" in result

    def test_expand_atomic(self):
        result = expand(atomic)
        assert "INTERCEPT" in result
        assert "TAG" in result

    def test_expand_generations(self):
        result = expand(generations)
        assert "TRACK" in result


# =============================================================================
# FUNCTIONAL TESTS — @pooled
# =============================================================================


class TestPooledFunctional:
    def test_marks_class(self):
        @pooled()
        class Particle:
            pass

        assert Particle._pooled is True
        assert isinstance(Particle._pool_config, PoolConfig)
        assert "pooled" in Particle._applied_decorators

    def test_custom_config(self):
        @pooled(initial_size=500, grow_factor=1.5, max_size=10000)
        class Bullet:
            pass

        assert Bullet._pool_config.initial_size == 500
        assert Bullet._pool_config.grow_factor == 1.5
        assert Bullet._pool_config.max_size == 10000

    def test_adds_release_method(self):
        @pooled()
        class Thing:
            pass

        t = Thing()
        t.release()  # no-op without runtime pool, shouldn't error

    def test_steps_recorded(self):
        @pooled()
        class P:
            pass

        assert hasattr(P, "_applied_steps")
        ops = {s.op for s in P._applied_steps}
        assert Op.TAG in ops
        assert Op.HOOK in ops
        assert Op.REGISTER in ops

    def test_tags_set(self):
        @pooled()
        class P:
            pass

        assert "pool" in P._tags
        assert "PoolManager" in P._registries
        assert "on_create" in P._hooks
        assert "on_destroy" in P._hooks


# =============================================================================
# FUNCTIONAL TESTS — @packed
# =============================================================================


class TestPackedFunctional:
    def test_default_soa(self):
        @packed()
        class Transform:
            pass

        assert Transform._packed is True
        assert Transform._packed_layout == "soa"

    def test_aos_layout(self):
        @packed(layout="aos")
        class Data:
            pass

        assert Data._packed_layout == "aos"

    def test_hybrid_layout(self):
        @packed(layout="hybrid")
        class Mix:
            pass

        assert Mix._packed_layout == "hybrid"

    def test_tags_set(self):
        @packed(layout="aos")
        class P:
            pass

        assert P._tags["memory"] == {"layout": "aos"}

    def test_steps_recorded(self):
        @packed()
        class P:
            pass

        assert len(P._applied_steps) == 1
        assert P._applied_steps[0].op is Op.TAG


# =============================================================================
# FUNCTIONAL TESTS — @aligned
# =============================================================================


class TestAlignedFunctional:
    def test_default_alignment(self):
        @aligned()
        class Rigid:
            pass

        assert Rigid._aligned is True
        assert Rigid._aligned_bytes == 64

    def test_custom_alignment(self):
        @aligned(bytes=32)
        class Simd:
            pass

        assert Simd._aligned_bytes == 32

    def test_rejects_non_power_of_2(self):
        with pytest.raises(ValueError, match="power of 2"):

            @aligned(bytes=33)
            class Bad:
                pass

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="power of 2"):

            @aligned(bytes=0)
            class Bad:
                pass

    def test_tags_set(self):
        @aligned(bytes=16)
        class A:
            pass

        assert A._tags["memory"] == {"alignment": 16}


# =============================================================================
# FUNCTIONAL TESTS — @arena
# =============================================================================


class TestArenaFunctional:
    def test_marks_class(self):
        @arena(name="frame")
        class DebugLine:
            pass

        assert DebugLine._arena is True
        assert DebugLine._arena_name == "frame"
        assert isinstance(DebugLine._arena_config, ArenaConfig)

    def test_tags_set(self):
        @arena(name="level")
        class A:
            pass

        assert A._tags["memory"] == {"allocator": "arena"}
        assert "on_create" in A._hooks


# =============================================================================
# FUNCTIONAL TESTS — @flyweight
# =============================================================================


class TestFlyweightFunctional:
    def test_auto_registers(self):
        @flyweight
        class MeshData:
            pass

        m1 = MeshData()
        m2 = MeshData()
        assert m1._flyweight_id == 0
        assert m2._flyweight_id == 1
        assert MeshData.get_by_id(0) is m1
        assert MeshData.get_by_id(1) is m2

    def test_unregister(self):
        @flyweight
        class Tex:
            pass

        t = Tex()
        fid = t._flyweight_id
        assert Tex.get_by_id(fid) is t
        t.unregister()
        assert Tex.get_by_id(fid) is None

    def test_tags_set(self):
        @flyweight
        class F:
            pass

        assert F._tags["memory"] == {"shared": True}
        assert "FlyweightCache" in F._registries
        assert len(F._intercepts) > 0

    def test_steps_recorded(self):
        @flyweight
        class F:
            pass

        ops = {s.op for s in F._applied_steps}
        assert Op.TAG in ops
        assert Op.INTERCEPT in ops
        assert Op.REGISTER in ops


# =============================================================================
# FUNCTIONAL TESTS — @intern
# =============================================================================


class TestInternFunctional:
    def test_string_interning(self):
        @intern
        class Path:
            pass

        s1 = Path.intern_string("assets/player.png")
        s2 = Path.intern_string("assets/player.png")
        assert s1 is s2

    def test_different_strings(self):
        @intern
        class Name:
            pass

        s1 = Name.intern_string("alice")
        s2 = Name.intern_string("bob")
        assert s1 is not s2

    def test_tags_set(self):
        @intern
        class I:
            pass

        assert I._tags["memory"] == {"interned": True}
        assert len(I._intercepts) > 0


# =============================================================================
# FUNCTIONAL TESTS — @generations
# =============================================================================


class TestGenerationsFunctional:
    def test_marks_class(self):
        @generations
        class EntityRef:
            pass

        assert EntityRef._generations is True
        assert "generations" in EntityRef._applied_decorators

    def test_tags_set(self):
        @generations
        class G:
            pass

        assert G._tags["memory"] == {"generational": True}
        assert G._tracked is True

    def test_has_validation_method(self):
        @generations
        class G:
            pass

        g = G()
        # No counters set yet, so any index is invalid
        assert g.is_generation_valid(0, 0) is False


# =============================================================================
# FUNCTIONAL TESTS — @copy_on_write
# =============================================================================


class TestCopyOnWriteFunctional:
    def test_marks_class(self):
        @copy_on_write
        class Level:
            pass

        assert Level._copy_on_write is True

    def test_clone_method(self):
        @copy_on_write
        class State:
            pass

        s = State()
        s.val = 42
        clone = s.cow_clone()
        assert clone._cow_shared is True

    def test_tags_set(self):
        @copy_on_write
        class C:
            pass

        assert C._tags["memory"] == {"cow": True}
        assert len(C._intercepts) > 0


# =============================================================================
# FUNCTIONAL TESTS — @inline_array
# =============================================================================


class TestInlineArrayFunctional:
    def test_marks_class(self):
        @inline_array(size=32)
        class Inventory:
            pass

        assert Inventory._inline_array is True
        assert Inventory._inline_array_size == 32
        assert isinstance(Inventory._inline_array_config, InlineArrayConfig)

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="positive"):

            @inline_array(size=0)
            class Bad:
                pass

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="positive"):

            @inline_array(size=-1)
            class Bad:
                pass

    def test_tags_set(self):
        @inline_array(size=16)
        class I:
            pass

        assert I._tags["memory"] == {"inline": True, "size": 16}


# =============================================================================
# FUNCTIONAL TESTS — @budget
# =============================================================================


class TestBudgetFunctional:
    def test_marks_class(self):
        @budget(category="particles", max_bytes=10_000_000)
        class Particle:
            pass

        assert Particle._budget is True
        assert Particle._budget_category == "particles"
        assert Particle._budget_max_bytes == 10_000_000
        assert isinstance(Particle._budget_config, BudgetConfig)

    def test_rejects_bad_warn_at(self):
        with pytest.raises(ValueError, match="0.0-1.0"):

            @budget(category="x", warn_at=1.5)
            class Bad:
                pass

    def test_tags_and_constraints(self):
        @budget(category="fx", max_bytes=1000)
        class B:
            pass

        assert "resource" in B._tags
        assert any(c["constraint"] == "budget_limit" for c in B._constraints)


# =============================================================================
# FUNCTIONAL TESTS — @allocator
# =============================================================================


class TestAllocatorFunctional:
    def test_marks_class(self):
        @allocator(type="linear", size=1024)
        class FrameAlloc:
            pass

        assert FrameAlloc._allocator is True
        assert FrameAlloc._allocator_type == "linear"
        assert FrameAlloc._allocator_size == 1024
        assert isinstance(FrameAlloc._allocator_config, AllocatorConfig)

    def test_thread_safe(self):
        @allocator(type="pool", size=64, thread_safe=True)
        class ThreadAlloc:
            pass

        assert ThreadAlloc._allocator_thread_safe is True

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError, match="positive"):

            @allocator(type="pool", size=0)
            class Bad:
                pass

    def test_tags_set(self):
        @allocator(type="buddy", size=4096)
        class A:
            pass

        assert A._tags["memory"] == {"allocator": "buddy"}


# =============================================================================
# FUNCTIONAL TESTS — @atomic
# =============================================================================


class TestAtomicFunctional:
    def test_marks_class(self):
        @atomic
        class Counter:
            pass

        assert Counter._atomic is True
        assert "atomic" in Counter._applied_decorators

    def test_atomic_operations(self):
        @atomic
        class Counter:
            pass

        c = Counter()
        c.value = 0

        old = c.fetch_add(10)
        assert old == 0
        assert c.value == 10

        old = c.fetch_sub(3)
        assert old == 10
        assert c.value == 7

    def test_compare_exchange(self):
        @atomic
        class Flag:
            pass

        f = Flag()
        f.value = False

        assert f.compare_exchange(False, True) is True
        assert f.value is True
        assert f.compare_exchange(False, True) is False

    def test_atomic_exchange(self):
        @atomic
        class Val:
            pass

        v = Val()
        v.value = "old"
        old = v.atomic_exchange("new")
        assert old == "old"
        assert v.value == "new"

    def test_atomic_load_store(self):
        @atomic
        class A:
            pass

        a = A()
        a.value = 42
        assert a.atomic_load() == 42
        a.atomic_store(99)
        assert a.atomic_load() == 99

    def test_tags_and_intercepts(self):
        @atomic
        class A:
            pass

        assert A._tags["thread_safe"] is True
        assert any(i.get("set") == "atomic" for i in A._intercepts)


# =============================================================================
# APPLIED STEPS TESTS (all decorators record steps)
# =============================================================================


class TestAppliedStepsAtRuntime:
    """Every decorator should populate _applied_steps with correct Ops."""

    def test_pooled(self):
        @pooled()
        class P:
            pass

        assert len(P._applied_steps) == 4

    def test_packed(self):
        @packed()
        class P:
            pass

        assert len(P._applied_steps) == 1

    def test_aligned(self):
        @aligned()
        class A:
            pass

        assert len(A._applied_steps) == 1

    def test_arena(self):
        @arena(name="x")
        class A:
            pass

        assert len(A._applied_steps) == 2

    def test_flyweight(self):
        @flyweight
        class F:
            pass

        assert len(F._applied_steps) == 3

    def test_intern(self):
        @intern
        class I:
            pass

        assert len(I._applied_steps) == 2

    def test_generations(self):
        @generations
        class G:
            pass

        assert len(G._applied_steps) == 2

    def test_copy_on_write(self):
        @copy_on_write
        class C:
            pass

        assert len(C._applied_steps) == 2

    def test_inline_array(self):
        @inline_array(size=8)
        class I:
            pass

        assert len(I._applied_steps) == 1

    def test_budget(self):
        @budget(category="test")
        class B:
            pass

        assert len(B._applied_steps) == 2

    def test_allocator(self):
        @allocator(type="linear", size=100)
        class A:
            pass

        assert len(A._applied_steps) == 1

    def test_atomic(self):
        @atomic
        class A:
            pass

        assert len(A._applied_steps) == 2


# =============================================================================
# INTROSPECTION
# =============================================================================


class TestMemoryIntrospection:
    ALL_DECORATORS = [
        pooled,
        packed,
        aligned,
        arena,
        flyweight,
        intern,
        generations,
        copy_on_write,
        inline_array,
        budget,
        allocator,
        atomic,
    ]

    @pytest.mark.parametrize("dec", ALL_DECORATORS)
    def test_decompose_returns_list(self, dec):
        steps = decompose(dec)
        assert isinstance(steps, list)
        assert len(steps) > 0

    @pytest.mark.parametrize("dec", ALL_DECORATORS)
    def test_expand_returns_string(self, dec):
        result = expand(dec)
        assert isinstance(result, str)
        assert "+" in result or result.isupper() or "TAG" in result

    @pytest.mark.parametrize("dec", ALL_DECORATORS)
    def test_all_steps_have_op(self, dec):
        steps = decompose(dec)
        for s in steps:
            assert isinstance(s, Step)
            assert isinstance(s.op, Op)
