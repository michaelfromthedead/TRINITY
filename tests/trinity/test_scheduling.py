"""Tests for trinity.decorators.scheduling — Ops-based scheduling decorators."""

from __future__ import annotations

import asyncio

import pytest

from trinity.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MIN_BATCH,
    DEFAULT_PHYSICS_HZ,
    DEFAULT_STACK_SIZE,
)
from trinity.decorators.ops import Op, Step, decompose, expand
from trinity.decorators.scheduling import (
    after,
    async_system,
    before,
    chain,
    deferred,
    exclusive,
    fixed,
    job,
    parallel,
    phase,
    run_if,
    throttle,
)

# ── helpers ──────────────────────────────────────────────────────────────


def dummy_system():
    pass


def another_system():
    pass


def third_system():
    pass


def always_true():
    return True


def is_paused():
    return False


async def async_handler():
    pass


# ── @phase ───────────────────────────────────────────────────────────────


class TestPhase:
    def test_basic(self):
        @phase(name="update")
        def sys():
            pass

        assert sys._phase is True
        assert sys._phase_name == "update"

    def test_after_before(self):
        @phase(name="render", after=("update",), before=("cleanup",))
        def sys():
            pass

        assert sys._phase_after == ("update",)
        assert sys._phase_before == ("cleanup",)

    def test_defaults(self):
        @phase(name="init")
        def sys():
            pass

        assert sys._phase_after == ()
        assert sys._phase_before == ()

    def test_steps_on_target(self):
        @phase(name="x")
        def sys():
            pass

        steps = sys._applied_steps
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_decompose_factory(self):
        steps = decompose(phase)
        assert len(steps) > 0

    def test_expand_readable(self):
        text = expand(phase)
        assert isinstance(text, str)
        assert "TAG" in text or "phase" in text.lower()


# ── @parallel ────────────────────────────────────────────────────────────


class TestParallel:
    def test_defaults(self):
        @parallel()
        def sys():
            pass

        assert sys._parallel is True
        assert sys._parallel_chunk_size == DEFAULT_CHUNK_SIZE
        assert sys._parallel_min_batch == DEFAULT_MIN_BATCH

    def test_custom(self):
        @parallel(chunk_size=128, min_batch=512)
        def sys():
            pass

        assert sys._parallel_chunk_size == 128
        assert sys._parallel_min_batch == 512

    def test_steps_count(self):
        @parallel()
        def sys():
            pass

        steps = sys._applied_steps
        assert len(steps) >= 3  # parallel, chunk_size, min_batch, REGISTER


# ── @exclusive ───────────────────────────────────────────────────────────


class TestExclusive:
    def test_basic(self):
        @exclusive()
        def sys():
            pass

        assert sys._exclusive is True

    def test_steps(self):
        @exclusive()
        def sys():
            pass

        steps = sys._applied_steps
        tags = [s for s in steps if s.op == Op.TAG]
        assert any(s.args.get("key") == "exclusive" for s in tags)


# ── @after ───────────────────────────────────────────────────────────────


class TestAfter:
    def test_single(self):
        @after(systems=(dummy_system,))
        def sys():
            pass

        assert dummy_system in sys._after
        assert "dummy_system" in sys._after_names

    def test_multiple(self):
        @after(systems=(dummy_system, another_system))
        def sys():
            pass

        assert len(sys._after) == 2

    def test_accumulate(self):
        @after(systems=(another_system,))
        @after(systems=(dummy_system,))
        def sys():
            pass

        assert dummy_system in sys._after
        assert another_system in sys._after

    def test_steps(self):
        @after(systems=(dummy_system,))
        def sys():
            pass

        steps = sys._applied_steps
        assert any(s.args.get("key") == "after" for s in steps if s.op == Op.TAG)


# ── @before ──────────────────────────────────────────────────────────────


class TestBefore:
    def test_single(self):
        @before(systems=(dummy_system,))
        def sys():
            pass

        assert dummy_system in sys._before
        assert "dummy_system" in sys._before_names

    def test_accumulate(self):
        @before(systems=(another_system,))
        @before(systems=(dummy_system,))
        def sys():
            pass

        assert len(sys._before) == 2


# ── @run_if ──────────────────────────────────────────────────────────────


class TestRunIf:
    def test_basic(self):
        @run_if(condition=always_true)
        def sys():
            pass

        assert sys._run_if is always_true
        assert sys._run_if_name == "always_true"

    def test_conditions_list(self):
        @run_if(condition=is_paused)
        @run_if(condition=always_true)
        def sys():
            pass

        assert len(sys._run_if_conditions) == 2
        assert always_true in sys._run_if_conditions
        assert is_paused in sys._run_if_conditions

    def test_steps(self):
        @run_if(condition=always_true)
        def sys():
            pass

        steps = sys._applied_steps
        assert any(s.args.get("key") == "run_if" for s in steps if s.op == Op.TAG)


# ── @fixed ───────────────────────────────────────────────────────────────


class TestFixed:
    def test_defaults(self):
        @fixed()
        def sys():
            pass

        assert sys._fixed is True
        assert sys._fixed_hz == DEFAULT_PHYSICS_HZ
        assert sys._fixed_delta == pytest.approx(1.0 / DEFAULT_PHYSICS_HZ)

    def test_custom_hz(self):
        @fixed(hz=120)
        def sys():
            pass

        assert sys._fixed_hz == 120
        assert sys._fixed_delta == pytest.approx(1.0 / 120)

    def test_steps(self):
        @fixed(hz=30)
        def sys():
            pass

        steps = sys._applied_steps
        tags = {s.args["key"]: s.args["value"] for s in steps if s.op == Op.TAG}
        assert tags["fixed_hz"] == 30


# ── @job ─────────────────────────────────────────────────────────────────


class TestJob:
    def test_defaults(self):
        @job()
        def sys():
            pass

        assert sys._job is True
        assert sys._job_priority == 0
        assert sys._job_affinity == "any"
        assert sys._job_stack_size == DEFAULT_STACK_SIZE

    def test_custom(self):
        @job(priority=10, affinity="main", stack_size=131072)
        def sys():
            pass

        assert sys._job_priority == 10
        assert sys._job_affinity == "main"
        assert sys._job_stack_size == 131072

    def test_steps(self):
        @job(priority=5)
        def sys():
            pass

        steps = sys._applied_steps
        tags = {s.args["key"]: s.args["value"] for s in steps if s.op == Op.TAG}
        assert tags["job_priority"] == 5


# ── @async_system ────────────────────────────────────────────────────────


class TestAsyncSystem:
    def test_sync_function(self):
        @async_system()
        def sys():
            pass

        assert sys._async_system is True
        assert sys._is_coroutine is False

    def test_async_function(self):
        decorated = async_system()(async_handler)
        assert decorated._async_system is True
        assert decorated._is_coroutine is True

    def test_steps(self):
        @async_system()
        def sys():
            pass

        steps = sys._applied_steps
        assert any(s.args.get("key") == "async_system" for s in steps if s.op == Op.TAG)


# ── @throttle ────────────────────────────────────────────────────────────


class TestThrottle:
    def test_max_hz(self):
        @throttle(max_hz=30.0)
        def sys():
            pass

        assert sys._throttle is True
        assert sys._throttle_max_hz == 30.0

    def test_max_ms(self):
        @throttle(max_ms=2.0)
        def sys():
            pass

        assert sys._throttle_max_ms == 2.0

    def test_both(self):
        @throttle(max_hz=60.0, max_ms=1.0)
        def sys():
            pass

        assert sys._throttle_max_hz == 60.0
        assert sys._throttle_max_ms == 1.0

    def test_validation_neither(self):
        with pytest.raises(ValueError, match="at least one"):

            @throttle()
            def sys():
                pass

    def test_tracking_attrs(self):
        @throttle(max_hz=10.0)
        def sys():
            pass

        assert sys._throttle_last_run == 0.0
        assert sys._throttle_accumulated_work is None

    def test_steps(self):
        @throttle(max_hz=60.0)
        def sys():
            pass

        steps = sys._applied_steps
        tags = {s.args["key"]: s.args["value"] for s in steps if s.op == Op.TAG}
        assert tags["throttle_max_hz"] == 60.0


# ── @deferred ────────────────────────────────────────────────────────────


class TestDeferred:
    def test_basic(self):
        @deferred()
        def sys():
            pass

        assert sys._deferred is True

    def test_steps(self):
        @deferred()
        def sys():
            pass

        steps = sys._applied_steps
        assert any(s.args.get("key") == "deferred" for s in steps if s.op == Op.TAG)


# ── @chain ───────────────────────────────────────────────────────────────


class TestChain:
    def test_basic(self):
        @chain(systems=(dummy_system, another_system, third_system))
        def pipeline():
            pass

        assert pipeline._chain is True
        assert len(pipeline._chain_systems) == 3
        assert pipeline._chain_names == (
            "dummy_system",
            "another_system",
            "third_system",
        )

    def test_member_marking(self):
        @chain(systems=(dummy_system, another_system))
        def pipeline():
            pass

        assert dummy_system._chain_member is True
        assert another_system._chain_member is True
        assert dummy_system._chain_index == 0
        assert another_system._chain_index == 1

    def test_implicit_ordering(self):
        @chain(systems=(dummy_system, another_system, third_system))
        def pipeline():
            pass

        # second system should have first as _after
        assert dummy_system in getattr(another_system, "_after", ())
        # third should have second
        assert another_system in getattr(third_system, "_after", ())

    def test_steps(self):
        @chain(systems=(dummy_system,))
        def pipeline():
            pass

        steps = pipeline._applied_steps
        assert any(s.args.get("key") == "chain" for s in steps if s.op == Op.TAG)


# ── cross-cutting ────────────────────────────────────────────────────────


class TestCrossCutting:
    def test_all_have_register_step(self):
        """Every scheduling decorator should produce a REGISTER(scheduling) step."""
        fns = []

        @phase(name="t")
        def s1():
            pass

        fns.append(s1)

        @parallel()
        def s2():
            pass

        fns.append(s2)

        @exclusive()
        def s3():
            pass

        fns.append(s3)

        @fixed()
        def s4():
            pass

        fns.append(s4)

        @job()
        def s5():
            pass

        fns.append(s5)

        @async_system()
        def s6():
            pass

        fns.append(s6)

        @deferred()
        def s7():
            pass

        fns.append(s7)

        @throttle(max_hz=10.0)
        def s8():
            pass

        fns.append(s8)

        for fn in fns:
            steps = fn._applied_steps
            reg = [s for s in steps if s.op == Op.REGISTER]
            assert len(reg) >= 1, f"{fn.__name__} missing REGISTER step"
            assert reg[0].args["registry"] == "scheduling"

    def test_expand_returns_string(self):
        result = expand(parallel)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decorator_preserves_function(self):
        """Decorated function should still be callable."""

        @fixed(hz=60)
        def my_system():
            return 42

        assert my_system() == 42

    def test_stacking(self):
        """Multiple scheduling decorators can stack."""

        @throttle(max_hz=30.0)
        @parallel(chunk_size=32)
        def sys():
            pass

        assert sys._parallel is True
        assert sys._throttle is True
