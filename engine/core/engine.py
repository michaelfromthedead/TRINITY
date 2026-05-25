"""Main Engine class: initialization, game loop, shutdown."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol

from engine.core.constants import DEFAULT_TARGET_FPS, DEFAULT_FIXED_TIMESTEP
from engine.core.frame import (
    FramePhase,
    FrameTimer,
    FrameAllocator,
    FrameContext,
    FixedTimestepAccumulator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency protocols (structural typing so we don't couple to concrete impls)
# ---------------------------------------------------------------------------

class WorldLike(Protocol):
    """Minimal interface expected from the World dependency."""
    def flush_commands(self) -> None: ...


class SystemSchedulerLike(Protocol):
    """Minimal interface expected from the SystemScheduler dependency."""
    def run_phase(self, phase: FramePhase, dt: float) -> None: ...


class TaskSchedulerLike(Protocol):
    """Minimal interface expected from the TaskScheduler dependency."""
    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    """Top-level engine that owns the game loop and coordinates subsystems.

    Parameters
    ----------
    target_fps:
        Target frames per second (used for fixed timestep and optional sleep).
    fixed_timestep:
        The delta used for fixed-rate simulation ticks.
    world:
        Optional ECS World instance.
    system_scheduler:
        Optional system scheduler for phase-based execution.
    task_scheduler:
        Optional parallel task scheduler.
    """

    __slots__ = (
        "_running",
        "_target_fps",
        "_frame_timer",
        "_frame_allocator",
        "_fixed_accumulator",
        "_world",
        "_system_scheduler",
        "_task_scheduler",
        "_phase_callbacks",
        "_initialized",
    )

    def __init__(
        self,
        target_fps: int = DEFAULT_TARGET_FPS,
        fixed_timestep: float = DEFAULT_FIXED_TIMESTEP,
        world: Optional[Any] = None,
        system_scheduler: Optional[Any] = None,
        task_scheduler: Optional[Any] = None,
    ) -> None:
        self._running: bool = False
        self._target_fps: int = target_fps
        self._frame_timer: FrameTimer = FrameTimer()
        self._frame_allocator: FrameAllocator = FrameAllocator()
        self._fixed_accumulator: FixedTimestepAccumulator = FixedTimestepAccumulator(fixed_timestep)
        self._world: Optional[Any] = world
        self._system_scheduler: Optional[Any] = system_scheduler
        self._task_scheduler: Optional[Any] = task_scheduler
        self._phase_callbacks: dict[FramePhase, list[Callable[[FrameContext], None]]] = {
            phase: [] for phase in FramePhase
        }
        self._initialized: bool = False

    # -- Properties ----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_timer(self) -> FrameTimer:
        return self._frame_timer

    @property
    def frame_allocator(self) -> FrameAllocator:
        return self._frame_allocator

    @property
    def delta_time(self) -> float:
        return self._frame_timer.delta_time

    @property
    def total_time(self) -> float:
        return self._frame_timer.total_time

    @property
    def frame_count(self) -> int:
        return self._frame_timer.frame_count

    @property
    def fps(self) -> float:
        return self._frame_timer.fps

    @property
    def world(self) -> Optional[Any]:
        return self._world

    @property
    def system_scheduler(self) -> Optional[Any]:
        return self._system_scheduler

    @property
    def task_scheduler(self) -> Optional[Any]:
        return self._task_scheduler

    # -- Phase callbacks -----------------------------------------------------

    def add_phase_callback(self, phase: FramePhase, callback: Callable[[FrameContext], None]) -> None:
        """Register a callback to be invoked during *phase* each frame."""
        self._phase_callbacks[phase].append(callback)

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self) -> None:
        """Initialize the engine and all subsystems."""
        if self._initialized:
            logger.warning("Engine.initialize called more than once")
            return
        logger.info("Engine initializing")
        self._initialized = True
        self.on_init()
        logger.info("Engine initialized")

    def shutdown(self) -> None:
        """Shut down the engine and release resources."""
        if not self._initialized:
            return
        logger.info("Engine shutting down")
        self._running = False
        self.on_shutdown()
        if self._task_scheduler is not None:
            try:
                self._task_scheduler.shutdown()
            except Exception:
                logger.exception("Error shutting down task scheduler")
        self._initialized = False
        logger.info("Engine shut down")

    def request_shutdown(self) -> None:
        """Request a graceful shutdown at the end of the current frame."""
        logger.info("Shutdown requested")
        self._running = False

    # -- Game loop -----------------------------------------------------------

    def run(self) -> None:
        """Run the main game loop until shutdown is requested."""
        if not self._initialized:
            self.initialize()
        self._running = True
        logger.info("Engine entering main loop")
        try:
            while self._running:
                self.step()
        finally:
            self.shutdown()

    def step(self) -> None:
        """Execute a single frame.

        This is the main per-frame entry point.  It can also be called
        manually for testing or editor-driven stepping.
        """
        # -- Begin frame --
        self._frame_timer.begin_frame()
        self._frame_allocator.reset()

        dt = self._frame_timer.delta_time

        # -- Fixed timestep ticks --
        self._fixed_accumulator.accumulate(self._frame_timer.unscaled_delta_time)
        while self._fixed_accumulator.should_tick():
            self._run_fixed_update(self._fixed_accumulator.fixed_dt)
            self._fixed_accumulator.consume_tick()

        # -- Variable-rate phases --
        for phase in FramePhase:
            ctx = FrameContext(
                frame_number=self._frame_timer.frame_count,
                delta_time=dt,
                total_time=self._frame_timer.total_time,
                time_scale=self._frame_timer.time_scale,
                phase=phase,
            )
            # System scheduler integration
            if self._system_scheduler is not None:
                try:
                    self._system_scheduler.run_phase(phase, dt)
                except Exception:
                    logger.exception("Error in system scheduler phase %s", phase.name)

            # User-registered callbacks
            for cb in self._phase_callbacks[phase]:
                try:
                    cb(ctx)
                except Exception:
                    logger.exception("Error in phase callback %s", phase.name)

        # -- End frame --
        self._frame_timer.end_frame()

    # -- Hooks for subclasses ------------------------------------------------

    def on_init(self) -> None:
        """Override in subclasses for custom initialization."""

    def on_shutdown(self) -> None:
        """Override in subclasses for custom shutdown logic."""

    # -- Internal ------------------------------------------------------------

    def _run_fixed_update(self, fixed_dt: float) -> None:
        """Execute one fixed-rate simulation tick."""
        # Placeholder: SystemScheduler would run fixed-rate systems here.
        logging.debug("Fixed update not yet connected to system scheduler")
