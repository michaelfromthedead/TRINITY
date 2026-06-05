"""Deterministic 13-phase tick scheduler for lockstep simulation (T-CC-0.17)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

from trinity.types import Fixed32, PCG64, SystemPhase

if TYPE_CHECKING:
    from engine.core.ecs import DeterministicCommandBuffer, World

__all__ = [
    "TickScheduler",
    "PhaseContext",
    "SystemCallback",
    "SchedulerConfig",
    "TickResult",
]


class SystemCallback(Protocol):
    """Protocol for system update callbacks."""

    def __call__(self, ctx: "PhaseContext") -> None: ...


@dataclass(slots=True, frozen=True)
class SchedulerConfig:
    """Configuration for the tick scheduler."""

    fixed_timestep: Fixed32 = field(default_factory=lambda: Fixed32(1.0 / 60.0))
    max_ticks_per_frame: int = 4
    accumulator_cap: Fixed32 = field(default_factory=lambda: Fixed32(0.25))


@dataclass(slots=True)
class PhaseContext:
    """
    Context passed to systems during phase execution.

    Provides access to tick state, delta time, command buffer, and RNG.
    """

    tick: int
    phase: SystemPhase
    delta_time: Fixed32
    rng: PCG64
    command_buffer: "DeterministicCommandBuffer"
    world: "World"

    def fork_rng(self, stream_id: int = 0) -> PCG64:
        """Fork RNG for parallel or sub-system use."""
        return self.rng.fork(stream_id)


@dataclass(slots=True)
class TickResult:
    """Result of a single tick execution."""

    tick: int
    checksum: int
    phase_times_ns: dict[SystemPhase, int] = field(default_factory=dict)


class TickScheduler:
    """
    13-phase tick scheduler for deterministic simulation.

    Executes registered systems in strict phase order with fixed timestep.
    Each tick produces a checksum for replay verification.
    """

    __slots__ = (
        "_config",
        "_systems",
        "_tick",
        "_accumulator",
        "_master_rng",
        "_paused",
        "_tick_results",
    )

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        seed: int = 0,
    ) -> None:
        self._config = config or SchedulerConfig()
        self._systems: dict[SystemPhase, list[SystemCallback]] = {
            phase: [] for phase in SystemPhase
        }
        self._tick = 0
        self._accumulator = Fixed32(0)
        self._master_rng = PCG64(seed)
        self._paused = False
        self._tick_results: list[TickResult] = []

    @property
    def current_tick(self) -> int:
        """Current tick number."""
        return self._tick

    @property
    def config(self) -> SchedulerConfig:
        """Scheduler configuration (read-only)."""
        return self._config

    @property
    def paused(self) -> bool:
        """Whether the scheduler is paused."""
        return self._paused

    def pause(self) -> None:
        """Pause tick execution."""
        self._paused = True

    def unpause(self) -> None:
        """Resume tick execution."""
        self._paused = False

    def register(self, phase: SystemPhase, callback: SystemCallback) -> None:
        """Register a system callback for a phase."""
        self._systems[phase].append(callback)

    def unregister(self, phase: SystemPhase, callback: SystemCallback) -> None:
        """Unregister a system callback from a phase."""
        try:
            self._systems[phase].remove(callback)
        except ValueError:
            pass

    def clear_phase(self, phase: SystemPhase) -> None:
        """Remove all systems from a phase."""
        self._systems[phase].clear()

    def clear_all(self) -> None:
        """Remove all registered systems."""
        for phase in SystemPhase:
            self._systems[phase].clear()

    def update(
        self,
        frame_delta: Fixed32,
        world: "World",
        command_buffer: "DeterministicCommandBuffer",
    ) -> list[TickResult]:
        """
        Update scheduler with frame delta time.

        Accumulates time and executes fixed-timestep ticks as needed.
        Returns list of tick results for this frame.
        """
        if self._paused:
            return []

        results: list[TickResult] = []

        # Accumulate and cap
        self._accumulator = self._accumulator + frame_delta
        if self._accumulator > self._config.accumulator_cap:
            self._accumulator = self._config.accumulator_cap

        # Execute ticks
        ticks_executed = 0
        while self._accumulator >= self._config.fixed_timestep:
            if ticks_executed >= self._config.max_ticks_per_frame:
                break

            result = self._execute_tick(world, command_buffer)
            results.append(result)
            self._tick_results.append(result)
            ticks_executed += 1

            self._accumulator = self._accumulator - self._config.fixed_timestep
            self._tick += 1
            command_buffer.advance_tick()

        return results

    def _execute_tick(
        self,
        world: "World",
        command_buffer: "DeterministicCommandBuffer",
    ) -> TickResult:
        """Execute a single tick through all 13 phases."""
        import time

        phase_times: dict[SystemPhase, int] = {}

        # Fork RNG for this tick (deterministic)
        tick_rng = self._master_rng.fork(self._tick)

        for phase in SystemPhase:
            start = time.perf_counter_ns()

            # Create phase context
            ctx = PhaseContext(
                tick=self._tick,
                phase=phase,
                delta_time=self._config.fixed_timestep,
                rng=tick_rng.fork(phase.value),
                command_buffer=command_buffer,
                world=world,
            )

            # Execute all systems in this phase
            for callback in self._systems[phase]:
                callback(ctx)

            phase_times[phase] = time.perf_counter_ns() - start

        # Flush command buffer and get checksum
        checksum = command_buffer.flush(world)

        return TickResult(
            tick=self._tick,
            checksum=checksum,
            phase_times_ns=phase_times,
        )

    def get_tick_checksum(self, tick: int) -> int | None:
        """Get checksum for a completed tick."""
        for result in self._tick_results:
            if result.tick == tick:
                return result.checksum
        return None

    def get_interpolation_alpha(self) -> Fixed32:
        """
        Get interpolation alpha for rendering between ticks.

        Returns value in [0, 1) representing progress toward next tick.
        """
        if self._config.fixed_timestep.raw == 0:
            return Fixed32(0)
        return self._accumulator / self._config.fixed_timestep

    def reset(self, seed: int | None = None) -> None:
        """Reset scheduler to initial state."""
        self._tick = 0
        self._accumulator = Fixed32(0)
        self._tick_results.clear()
        if seed is not None:
            self._master_rng = PCG64(seed)

    def set_tick(self, tick: int) -> None:
        """Set current tick (for replay/rollback)."""
        self._tick = tick

    def get_phase_budget(self, phase: SystemPhase) -> int:
        """Get average time spent in phase (nanoseconds)."""
        times = [
            r.phase_times_ns.get(phase, 0)
            for r in self._tick_results[-60:]  # Last 60 ticks
            if phase in r.phase_times_ns
        ]
        return sum(times) // len(times) if times else 0

    def get_tick_budget(self) -> int:
        """Get average time per tick (nanoseconds)."""
        budgets = [
            sum(r.phase_times_ns.values())
            for r in self._tick_results[-60:]
        ]
        return sum(budgets) // len(budgets) if budgets else 0


class TickSchedulerBuilder:
    """Builder for constructing a TickScheduler with systems."""

    __slots__ = ("_config", "_seed", "_systems")

    def __init__(self) -> None:
        self._config: SchedulerConfig | None = None
        self._seed = 0
        self._systems: list[tuple[SystemPhase, SystemCallback]] = []

    def with_config(self, config: SchedulerConfig) -> "TickSchedulerBuilder":
        """Set scheduler configuration."""
        self._config = config
        return self

    def with_seed(self, seed: int) -> "TickSchedulerBuilder":
        """Set master RNG seed."""
        self._seed = seed
        return self

    def with_timestep(self, hz: int) -> "TickSchedulerBuilder":
        """Set tick rate in Hz."""
        self._config = SchedulerConfig(
            fixed_timestep=Fixed32(1.0 / hz),
        )
        return self

    def add_system(self, phase: SystemPhase, callback: SystemCallback) -> "TickSchedulerBuilder":
        """Add a system to a phase."""
        self._systems.append((phase, callback))
        return self

    def build(self) -> TickScheduler:
        """Build the scheduler."""
        scheduler = TickScheduler(config=self._config, seed=self._seed)
        for phase, callback in self._systems:
            scheduler.register(phase, callback)
        return scheduler
