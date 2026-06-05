"""Tests for TickScheduler (T-CC-0.17)."""
from dataclasses import dataclass

import pytest

from trinity.types import Fixed32, PCG64, SystemPhase
from engine.core.ecs import World, DeterministicCommandBuffer
from engine.core.tick_scheduler import (
    TickScheduler,
    SchedulerConfig,
    PhaseContext,
    TickResult,
    TickSchedulerBuilder,
)


@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Counter:
    value: int = 0


class TestSystemPhase:
    """Test the 13-phase enum."""

    def test_phase_count(self):
        """Test there are exactly 13 phases."""
        assert len(SystemPhase) == 13

    def test_phase_order(self):
        """Test phases are in correct order."""
        phases = list(SystemPhase)
        for i, phase in enumerate(phases):
            assert phase.value == i

    def test_phase_names(self):
        """Test all required phases exist."""
        names = {phase.name for phase in SystemPhase}
        required = {
            "PRE_INPUT", "INPUT", "POST_INPUT",
            "PRE_PHYSICS", "PHYSICS", "POST_PHYSICS",
            "PRE_UPDATE", "UPDATE", "POST_UPDATE",
            "PRE_RENDER", "RENDER", "POST_RENDER",
            "LATE",
        }
        assert names == required


class TestSchedulerConfig:
    """Test SchedulerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SchedulerConfig()
        assert abs(config.fixed_timestep.as_float - (1.0 / 60.0)) < 0.001
        assert config.max_ticks_per_frame == 4
        assert abs(config.accumulator_cap.as_float - 0.25) < 0.001

    def test_custom_config(self):
        """Test custom configuration."""
        config = SchedulerConfig(
            fixed_timestep=Fixed32(1.0 / 30.0),
            max_ticks_per_frame=2,
        )
        assert abs(config.fixed_timestep.as_float - (1.0 / 30.0)) < 0.001
        assert config.max_ticks_per_frame == 2


class TestTickSchedulerCreation:
    """Test TickScheduler creation."""

    def test_default_creation(self):
        """Test default scheduler creation."""
        scheduler = TickScheduler()
        assert scheduler.current_tick == 0
        assert not scheduler.paused

    def test_with_config(self):
        """Test scheduler with custom config."""
        config = SchedulerConfig(max_ticks_per_frame=8)
        scheduler = TickScheduler(config=config)
        assert scheduler.config.max_ticks_per_frame == 8

    def test_with_seed(self):
        """Test scheduler with custom seed."""
        scheduler = TickScheduler(seed=12345)
        assert scheduler.current_tick == 0


class TestSystemRegistration:
    """Test system registration."""

    def test_register_system(self):
        """Test registering a system."""
        scheduler = TickScheduler()
        calls = []

        def my_system(ctx: PhaseContext) -> None:
            calls.append(ctx.phase)

        scheduler.register(SystemPhase.UPDATE, my_system)

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert SystemPhase.UPDATE in calls

    def test_unregister_system(self):
        """Test unregistering a system."""
        scheduler = TickScheduler()
        calls = []

        def my_system(ctx: PhaseContext) -> None:
            calls.append(1)

        scheduler.register(SystemPhase.UPDATE, my_system)
        scheduler.unregister(SystemPhase.UPDATE, my_system)

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert len(calls) == 0

    def test_clear_phase(self):
        """Test clearing all systems from a phase."""
        scheduler = TickScheduler()

        def sys1(ctx: PhaseContext) -> None: pass
        def sys2(ctx: PhaseContext) -> None: pass

        scheduler.register(SystemPhase.UPDATE, sys1)
        scheduler.register(SystemPhase.UPDATE, sys2)
        scheduler.clear_phase(SystemPhase.UPDATE)

        # No way to check directly, but should not error
        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

    def test_clear_all(self):
        """Test clearing all systems."""
        scheduler = TickScheduler()

        def sys1(ctx: PhaseContext) -> None: pass
        def sys2(ctx: PhaseContext) -> None: pass

        scheduler.register(SystemPhase.UPDATE, sys1)
        scheduler.register(SystemPhase.PHYSICS, sys2)
        scheduler.clear_all()

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)


class TestPhaseExecution:
    """Test phase execution order."""

    def test_phases_execute_in_order(self):
        """Test all 13 phases execute in correct order."""
        scheduler = TickScheduler()
        execution_order = []

        for phase in SystemPhase:
            def capture_phase(ctx: PhaseContext, p=phase) -> None:
                execution_order.append(p)
            scheduler.register(phase, capture_phase)

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert execution_order == list(SystemPhase)

    def test_multiple_systems_per_phase(self):
        """Test multiple systems in same phase execute in order."""
        scheduler = TickScheduler()
        calls = []

        def sys1(ctx: PhaseContext) -> None:
            calls.append(1)

        def sys2(ctx: PhaseContext) -> None:
            calls.append(2)

        scheduler.register(SystemPhase.UPDATE, sys1)
        scheduler.register(SystemPhase.UPDATE, sys2)

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert calls == [1, 2]


class TestPhaseContext:
    """Test PhaseContext provided to systems."""

    def test_context_tick(self):
        """Test context provides correct tick."""
        scheduler = TickScheduler()
        received_ticks = []

        def capture_tick(ctx: PhaseContext) -> None:
            received_ticks.append(ctx.tick)

        scheduler.register(SystemPhase.UPDATE, capture_tick)

        world = World()
        buf = DeterministicCommandBuffer()

        scheduler.update(Fixed32(1.0 / 60.0), world, buf)
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert received_ticks == [0, 1]

    def test_context_phase(self):
        """Test context provides correct phase."""
        scheduler = TickScheduler()
        received_phases = []

        def capture_phase(ctx: PhaseContext) -> None:
            received_phases.append(ctx.phase)

        scheduler.register(SystemPhase.UPDATE, capture_phase)
        scheduler.register(SystemPhase.PHYSICS, capture_phase)

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert SystemPhase.PHYSICS in received_phases
        assert SystemPhase.UPDATE in received_phases

    def test_context_delta_time(self):
        """Test context provides correct delta time."""
        config = SchedulerConfig(fixed_timestep=Fixed32(1.0 / 30.0))
        scheduler = TickScheduler(config=config)
        received_dt = []

        def capture_dt(ctx: PhaseContext) -> None:
            received_dt.append(ctx.delta_time.as_float)

        scheduler.register(SystemPhase.UPDATE, capture_dt)

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 30.0), world, buf)

        assert abs(received_dt[0] - (1.0 / 30.0)) < 0.001

    def test_context_rng_deterministic(self):
        """Test context RNG is deterministic across runs."""
        def run_simulation(seed):
            scheduler = TickScheduler(seed=seed)
            values = []

            def capture_rng(ctx: PhaseContext) -> None:
                values.append(ctx.rng.next_u32())

            scheduler.register(SystemPhase.UPDATE, capture_rng)

            world = World()
            buf = DeterministicCommandBuffer()
            for _ in range(5):
                scheduler.update(Fixed32(1.0 / 60.0), world, buf)

            return values

        vals1 = run_simulation(42)
        vals2 = run_simulation(42)
        vals3 = run_simulation(99)

        assert vals1 == vals2
        assert vals1 != vals3

    def test_context_fork_rng(self):
        """Test context provides fork_rng method."""
        scheduler = TickScheduler(seed=42)
        forked_values = []

        def use_fork(ctx: PhaseContext) -> None:
            fork = ctx.fork_rng(0)
            forked_values.append(fork.next_u32())

        scheduler.register(SystemPhase.UPDATE, use_fork)

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert len(forked_values) == 1
        assert forked_values[0] > 0


class TestFixedTimestep:
    """Test fixed timestep behavior."""

    def test_accumulator_single_tick(self):
        """Test single tick executes with exact timestep."""
        config = SchedulerConfig(fixed_timestep=Fixed32(1.0 / 60.0))
        scheduler = TickScheduler(config=config)
        tick_count = [0]

        def count_ticks(ctx: PhaseContext) -> None:
            tick_count[0] += 1

        scheduler.register(SystemPhase.UPDATE, count_ticks)

        world = World()
        buf = DeterministicCommandBuffer()
        results = scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert len(results) == 1
        assert tick_count[0] == 1

    def test_accumulator_multiple_ticks(self):
        """Test multiple ticks execute when time accumulated."""
        config = SchedulerConfig(fixed_timestep=Fixed32(1.0 / 60.0))
        scheduler = TickScheduler(config=config)
        tick_count = [0]

        def count_ticks(ctx: PhaseContext) -> None:
            tick_count[0] += 1

        scheduler.register(SystemPhase.UPDATE, count_ticks)

        world = World()
        buf = DeterministicCommandBuffer()
        results = scheduler.update(Fixed32(3.0 / 60.0), world, buf)

        assert len(results) == 3
        assert tick_count[0] == 3

    def test_accumulator_capped(self):
        """Test accumulator is capped to prevent spiral of death."""
        config = SchedulerConfig(
            fixed_timestep=Fixed32(1.0 / 60.0),
            max_ticks_per_frame=4,
            accumulator_cap=Fixed32(0.25),
        )
        scheduler = TickScheduler(config=config)

        world = World()
        buf = DeterministicCommandBuffer()
        results = scheduler.update(Fixed32(1.0), world, buf)  # 1 second

        assert len(results) == 4  # Capped at max_ticks_per_frame

    def test_no_tick_on_insufficient_time(self):
        """Test no tick executes if not enough time accumulated."""
        config = SchedulerConfig(fixed_timestep=Fixed32(1.0 / 60.0))
        scheduler = TickScheduler(config=config)
        tick_count = [0]

        def count_ticks(ctx: PhaseContext) -> None:
            tick_count[0] += 1

        scheduler.register(SystemPhase.UPDATE, count_ticks)

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(0.001), world, buf)  # Much less than 1/60

        assert tick_count[0] == 0


class TestPauseResume:
    """Test pause/resume functionality."""

    def test_pause_stops_ticks(self):
        """Test pausing stops tick execution."""
        scheduler = TickScheduler()
        tick_count = [0]

        def count_ticks(ctx: PhaseContext) -> None:
            tick_count[0] += 1

        scheduler.register(SystemPhase.UPDATE, count_ticks)
        scheduler.pause()

        world = World()
        buf = DeterministicCommandBuffer()
        results = scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert len(results) == 0
        assert tick_count[0] == 0

    def test_unpause_resumes_ticks(self):
        """Test unpausing resumes tick execution."""
        scheduler = TickScheduler()
        tick_count = [0]

        def count_ticks(ctx: PhaseContext) -> None:
            tick_count[0] += 1

        scheduler.register(SystemPhase.UPDATE, count_ticks)
        scheduler.pause()
        scheduler.unpause()

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert tick_count[0] == 1


class TestTickResults:
    """Test tick result tracking."""

    def test_result_tick_number(self):
        """Test result contains correct tick number."""
        scheduler = TickScheduler()

        world = World()
        buf = DeterministicCommandBuffer()
        results = scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert results[0].tick == 0

    def test_result_checksum(self):
        """Test result contains checksum."""
        scheduler = TickScheduler()

        world = World()
        buf = DeterministicCommandBuffer()
        results = scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert isinstance(results[0].checksum, int)

    def test_get_tick_checksum(self):
        """Test retrieving checksum by tick."""
        scheduler = TickScheduler()

        world = World()
        buf = DeterministicCommandBuffer()
        results = scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        expected = results[0].checksum
        assert scheduler.get_tick_checksum(0) == expected

    def test_get_nonexistent_checksum(self):
        """Test getting checksum for non-executed tick."""
        scheduler = TickScheduler()
        assert scheduler.get_tick_checksum(999) is None


class TestInterpolation:
    """Test interpolation alpha calculation."""

    def test_interpolation_alpha_at_start(self):
        """Test alpha is 0 at tick boundary."""
        scheduler = TickScheduler()
        alpha = scheduler.get_interpolation_alpha()
        assert alpha.as_float == 0.0

    def test_interpolation_alpha_mid_frame(self):
        """Test alpha increases as time accumulates."""
        config = SchedulerConfig(fixed_timestep=Fixed32(1.0 / 60.0))
        scheduler = TickScheduler(config=config)

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(0.5 / 60.0), world, buf)  # Half a tick

        alpha = scheduler.get_interpolation_alpha()
        assert 0.4 < alpha.as_float < 0.6


class TestReset:
    """Test scheduler reset functionality."""

    def test_reset_tick(self):
        """Test reset resets tick counter."""
        scheduler = TickScheduler()

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        scheduler.reset()
        assert scheduler.current_tick == 0

    def test_reset_with_new_seed(self):
        """Test reset with new seed."""
        scheduler = TickScheduler(seed=42)
        values1 = []

        def capture(ctx: PhaseContext) -> None:
            values1.append(ctx.rng.next_u32())

        scheduler.register(SystemPhase.UPDATE, capture)

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        scheduler.reset(seed=99)
        values2 = []

        def capture2(ctx: PhaseContext) -> None:
            values2.append(ctx.rng.next_u32())

        scheduler.clear_all()
        scheduler.register(SystemPhase.UPDATE, capture2)

        world2 = World()
        buf2 = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world2, buf2)

        assert values1 != values2


class TestBuilder:
    """Test TickSchedulerBuilder."""

    def test_builder_with_seed(self):
        """Test builder with_seed."""
        scheduler = TickSchedulerBuilder().with_seed(42).build()
        assert scheduler.current_tick == 0

    def test_builder_with_timestep(self):
        """Test builder with_timestep."""
        scheduler = TickSchedulerBuilder().with_timestep(30).build()
        assert abs(scheduler.config.fixed_timestep.as_float - (1.0 / 30.0)) < 0.001

    def test_builder_add_system(self):
        """Test builder add_system."""
        calls = []

        def my_sys(ctx: PhaseContext) -> None:
            calls.append(1)

        scheduler = (
            TickSchedulerBuilder()
            .add_system(SystemPhase.UPDATE, my_sys)
            .build()
        )

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert calls == [1]

    def test_builder_chaining(self):
        """Test builder method chaining."""
        calls = []

        def sys1(ctx: PhaseContext) -> None:
            calls.append(1)

        def sys2(ctx: PhaseContext) -> None:
            calls.append(2)

        scheduler = (
            TickSchedulerBuilder()
            .with_seed(42)
            .with_timestep(60)
            .add_system(SystemPhase.PHYSICS, sys1)
            .add_system(SystemPhase.UPDATE, sys2)
            .build()
        )

        world = World()
        buf = DeterministicCommandBuffer()
        scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert 1 in calls
        assert 2 in calls
