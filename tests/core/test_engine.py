"""Tests for engine.core.engine -- Engine class lifecycle and game loop."""

import threading
import time

import pytest

from engine.core.engine import Engine
from engine.core.frame import FramePhase, FrameContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RecordingScheduler:
    """Fake system scheduler that records phase calls."""

    def __init__(self):
        self.phases_run: list[FramePhase] = []

    def run_phase(self, phase: FramePhase, dt: float) -> None:
        self.phases_run.append(phase)


class _FakeTaskScheduler:
    def __init__(self):
        self.shut_down = False

    def shutdown(self) -> None:
        self.shut_down = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEngineLifecycle:
    def test_initialize_sets_flag(self):
        engine = Engine()
        assert not engine.is_running
        engine.initialize()
        assert engine._initialized is True
        # Prove it actually initialized by stepping without error
        engine.step()
        engine.shutdown()

    def test_double_initialize_is_safe(self):
        engine = Engine()
        engine.initialize()
        engine.initialize()  # Should warn but not raise
        engine.shutdown()

    def test_shutdown_without_init(self):
        engine = Engine()
        engine.shutdown()  # Should be a no-op

    def test_shutdown_stops_task_scheduler(self):
        ts = _FakeTaskScheduler()
        engine = Engine(task_scheduler=ts)
        engine.initialize()
        engine.shutdown()
        assert ts.shut_down

    def test_on_init_hook_called(self):
        called = []

        class MyEngine(Engine):
            def on_init(self):
                called.append(True)

        e = MyEngine()
        e.initialize()
        e.shutdown()
        assert called == [True]

    def test_on_shutdown_hook_called(self):
        called = []

        class MyEngine(Engine):
            def on_shutdown(self):
                called.append(True)

        e = MyEngine()
        e.initialize()
        e.shutdown()
        assert called == [True]


class TestEngineStep:
    def test_single_step_increments_frame_count(self):
        engine = Engine()
        engine.initialize()
        engine.step()
        assert engine.frame_count == 1
        engine.step()
        assert engine.frame_count == 2
        engine.shutdown()

    def test_step_invokes_all_phases_in_order(self):
        sched = _RecordingScheduler()
        engine = Engine(system_scheduler=sched)
        engine.initialize()
        engine.step()
        assert sched.phases_run == list(FramePhase)
        engine.shutdown()

    def test_phase_callbacks_receive_context(self):
        contexts: list[FrameContext] = []
        engine = Engine()
        engine.initialize()
        engine.add_phase_callback(FramePhase.UPDATE, lambda ctx: contexts.append(ctx))
        engine.step()
        assert len(contexts) == 1
        assert contexts[0].phase == FramePhase.UPDATE
        assert contexts[0].frame_number == 0  # frame_count before end_frame
        engine.shutdown()

    def test_time_tracking(self):
        engine = Engine()
        engine.initialize()
        engine.step()
        time.sleep(0.005)
        engine.step()
        time.sleep(0.005)
        engine.step()
        assert engine.delta_time > 0.0
        assert engine.total_time > 0.0
        engine.shutdown()


class TestEngineRun:
    def test_request_shutdown_stops_loop(self):
        engine = Engine()
        engine.initialize()
        # Schedule shutdown after first step
        engine.add_phase_callback(
            FramePhase.POST_RENDER,
            lambda ctx: engine.request_shutdown(),
        )
        engine._running = True
        engine.run()
        assert not engine.is_running
        assert engine.frame_count >= 1

    def test_run_auto_initializes(self):
        engine = Engine()
        engine.add_phase_callback(
            FramePhase.POST_RENDER,
            lambda ctx: engine.request_shutdown(),
        )
        engine.run()
        assert engine.frame_count >= 1


class TestEngineDependencies:
    def test_world_property(self):
        sentinel = object()
        engine = Engine(world=sentinel)
        assert engine.world is sentinel

    def test_system_scheduler_property(self):
        sched = _RecordingScheduler()
        engine = Engine(system_scheduler=sched)
        assert engine.system_scheduler is sched

    def test_task_scheduler_property(self):
        ts = _FakeTaskScheduler()
        engine = Engine(task_scheduler=ts)
        assert engine.task_scheduler is ts
