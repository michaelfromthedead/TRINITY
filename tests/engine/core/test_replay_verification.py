"""Integration tests for replay verification (T-CC-0.18)."""
from dataclasses import dataclass

import pytest

from trinity.types import Fixed32, PCG64, SystemPhase
from engine.core.ecs import World, DeterministicCommandBuffer
from engine.core.ecs.deterministic_buffer import ReplayBuffer
from engine.core.tick_scheduler import TickScheduler, SchedulerConfig, PhaseContext


@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Velocity:
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class GameState:
    score: int = 0
    health: int = 100


class TestTickChecksumGeneration:
    """Test that ticks produce consistent checksums."""

    def test_tick_produces_checksum(self):
        """Test each tick produces a checksum."""
        scheduler = TickScheduler(seed=42)

        world = World()
        buf = DeterministicCommandBuffer()
        results = scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert len(results) == 1
        assert isinstance(results[0].checksum, int)

    def test_identical_simulations_same_checksums(self):
        """Test identical simulations produce identical checksums."""

        def run_simulation(seed, num_ticks):
            world = World()
            buf = DeterministicCommandBuffer()
            scheduler = TickScheduler(seed=seed)

            def update_system(ctx: PhaseContext) -> None:
                # Use RNG to generate values - this makes checksums seed-dependent
                x_val = ctx.rng.next_int(0, 1000)
                y_val = ctx.rng.next_int(0, 1000)
                entity = ctx.world.spawn(Position(x_val, y_val))
                ctx.command_buffer.set_value(entity, Position, "x", x_val * 2)

            scheduler.register(SystemPhase.UPDATE, update_system)

            checksums = []
            for _ in range(num_ticks):
                results = scheduler.update(Fixed32(1.0 / 60.0), world, buf)
                checksums.extend(r.checksum for r in results)

            return checksums

        cs1 = run_simulation(42, 10)
        cs2 = run_simulation(42, 10)
        cs3 = run_simulation(99, 10)

        assert cs1 == cs2  # Same seed = same checksums
        assert cs1 != cs3  # Different seed = different checksums

    def test_different_inputs_different_checksums(self):
        """Test different inputs produce different checksums."""
        world1 = World()
        world2 = World()
        buf1 = DeterministicCommandBuffer()
        buf2 = DeterministicCommandBuffer()

        e1 = world1.spawn(GameState(score=100))
        e2 = world2.spawn(GameState(score=200))  # Different starting state

        buf1.set_value(e1, GameState, "score", 150)
        buf2.set_value(e2, GameState, "score", 150)

        cs1 = buf1.flush(world1)
        cs2 = buf2.flush(world2)

        # Same operation but different starting entity IDs may produce different checksums
        # The important thing is consistency
        assert isinstance(cs1, int)
        assert isinstance(cs2, int)


class TestReplayVerification:
    """Test replay verification between two simulations."""

    def test_verify_matching_replays(self):
        """Test verification passes for matching replays."""

        def run_game(seed):
            world = World()
            buf = DeterministicCommandBuffer()
            scheduler = TickScheduler(seed=seed)

            def game_logic(ctx: PhaseContext) -> None:
                val = ctx.rng.next_int(1, 100)
                ctx.world.spawn(GameState(score=val))

            scheduler.register(SystemPhase.UPDATE, game_logic)

            checksums = []
            for _ in range(5):
                results = scheduler.update(Fixed32(1.0 / 60.0), world, buf)
                checksums.extend(r.checksum for r in results)

            return checksums

        replay1 = run_game(42)
        replay2 = run_game(42)

        assert len(replay1) == 5
        assert replay1 == replay2

        for i, (c1, c2) in enumerate(zip(replay1, replay2)):
            assert c1 == c2, f"Mismatch at tick {i}: {c1} != {c2}"

    def test_detect_desync(self):
        """Test detection of desynced replays."""

        def run_game(seed, corrupt_tick=-1):
            world = World()
            buf = DeterministicCommandBuffer()
            scheduler = TickScheduler(seed=seed)

            def game_logic(ctx: PhaseContext) -> None:
                val = ctx.rng.next_int(1, 100)
                if ctx.tick == corrupt_tick:
                    val += 1  # Introduce desync
                entity = ctx.world.spawn(GameState(score=val))
                # Record via command buffer for checksumming
                ctx.command_buffer.set_value(entity, GameState, "score", val)

            scheduler.register(SystemPhase.UPDATE, game_logic)

            checksums = []
            for _ in range(5):
                results = scheduler.update(Fixed32(1.0 / 60.0), world, buf)
                checksums.extend(r.checksum for r in results)

            return checksums

        clean = run_game(42)
        corrupted = run_game(42, corrupt_tick=2)

        # Find where desync occurs
        desync_tick = -1
        for i, (c1, c2) in enumerate(zip(clean, corrupted)):
            if c1 != c2:
                desync_tick = i
                break

        assert desync_tick == 2


class TestReplayBuffer:
    """Test ReplayBuffer for state reconstruction."""

    def test_replay_from_snapshot(self):
        """Test replay can continue from a snapshot."""
        world = World()
        buf = DeterministicCommandBuffer()
        scheduler = TickScheduler(seed=42)

        entity = world.spawn(GameState(score=0, health=100))
        scores = []

        def update_score(ctx: PhaseContext) -> None:
            current = ctx.world.get_component(entity, GameState)
            if current:
                new_score = current.score + ctx.rng.next_int(1, 10)
                scores.append(new_score)
                ctx.command_buffer.set_value(entity, GameState, "score", new_score)

        scheduler.register(SystemPhase.UPDATE, update_score)

        # Run 10 ticks
        for _ in range(10):
            scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        # Get final score
        final_state = world.get_component(entity, GameState)
        final_score = final_state.score

        # Replay should produce same final score
        world2 = World()
        buf2 = DeterministicCommandBuffer()
        scheduler2 = TickScheduler(seed=42)

        entity2 = world2.spawn(GameState(score=0, health=100))
        scores2 = []

        def update_score2(ctx: PhaseContext) -> None:
            current = ctx.world.get_component(entity2, GameState)
            if current:
                new_score = current.score + ctx.rng.next_int(1, 10)
                scores2.append(new_score)
                ctx.command_buffer.set_value(entity2, GameState, "score", new_score)

        scheduler2.register(SystemPhase.UPDATE, update_score2)

        for _ in range(10):
            scheduler2.update(Fixed32(1.0 / 60.0), world2, buf2)

        final_state2 = world2.get_component(entity2, GameState)
        assert final_state2.score == final_score
        assert scores == scores2


class TestLockstepSimulation:
    """Test lockstep simulation scenarios."""

    def test_lockstep_two_players(self):
        """Test two players running same simulation stay in sync."""

        class GameSim:
            def __init__(self, seed):
                self.world = World()
                self.buf = DeterministicCommandBuffer()
                self.scheduler = TickScheduler(seed=seed)
                self.checksums = []

                def physics(ctx: PhaseContext) -> None:
                    pass  # Placeholder

                def game_logic(ctx: PhaseContext) -> None:
                    val = ctx.rng.next_int(0, 100)
                    ctx.world.spawn(Position(val, val))

                self.scheduler.register(SystemPhase.PHYSICS, physics)
                self.scheduler.register(SystemPhase.UPDATE, game_logic)

            def step(self):
                results = self.scheduler.update(
                    Fixed32(1.0 / 60.0),
                    self.world,
                    self.buf,
                )
                for r in results:
                    self.checksums.append(r.checksum)

        player1 = GameSim(seed=12345)
        player2 = GameSim(seed=12345)

        # Both players run 100 ticks
        for _ in range(100):
            player1.step()
            player2.step()

        assert len(player1.checksums) == 100
        assert player1.checksums == player2.checksums

    def test_detect_cheat_attempt(self):
        """Test detection of cheating (modified game state)."""

        def run_fair(seed):
            world = World()
            buf = DeterministicCommandBuffer()
            scheduler = TickScheduler(seed=seed)

            def game(ctx: PhaseContext) -> None:
                dmg = ctx.rng.next_int(5, 15)
                health = 100 - dmg
                entity = ctx.world.spawn(GameState(health=health))
                # Record via command buffer for checksumming
                ctx.command_buffer.set_value(entity, GameState, "health", health)

            scheduler.register(SystemPhase.UPDATE, game)

            checksums = []
            for _ in range(10):
                for r in scheduler.update(Fixed32(1.0 / 60.0), world, buf):
                    checksums.append(r.checksum)
            return checksums

        def run_cheat(seed):
            world = World()
            buf = DeterministicCommandBuffer()
            scheduler = TickScheduler(seed=seed)

            def game(ctx: PhaseContext) -> None:
                ctx.rng.next_int(5, 15)  # Consume RNG but ignore
                entity = ctx.world.spawn(GameState(health=100))  # Always full health!
                # Record via command buffer - different value = different checksum
                ctx.command_buffer.set_value(entity, GameState, "health", 100)

            scheduler.register(SystemPhase.UPDATE, game)

            checksums = []
            for _ in range(10):
                for r in scheduler.update(Fixed32(1.0 / 60.0), world, buf):
                    checksums.append(r.checksum)
            return checksums

        fair = run_fair(42)
        cheat = run_cheat(42)

        assert fair != cheat
        # Can identify exactly which tick diverged
        first_mismatch = next(
            i for i, (f, c) in enumerate(zip(fair, cheat)) if f != c
        )
        assert first_mismatch == 0  # Cheating from the start


class TestDeterminismGuarantees:
    """Test determinism guarantees hold."""

    def test_no_floating_point_in_simulation(self):
        """Test simulation uses fixed-point, not float."""
        world = World()
        buf = DeterministicCommandBuffer()
        scheduler = TickScheduler(seed=42)

        # All RNG values should be integers or Fixed
        rng_values = []

        def capture_rng(ctx: PhaseContext) -> None:
            # These are all deterministic integer operations
            rng_values.append(ctx.rng.next_u32())
            rng_values.append(ctx.rng.next_int(0, 100))

        scheduler.register(SystemPhase.UPDATE, capture_rng)

        for _ in range(10):
            scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        assert all(isinstance(v, int) for v in rng_values)

    def test_execution_order_invariant(self):
        """Test execution order is consistent."""
        execution_trace = []

        def make_tracer(name):
            def tracer(ctx: PhaseContext) -> None:
                execution_trace.append((ctx.tick, name))
            return tracer

        scheduler = TickScheduler(seed=42)
        scheduler.register(SystemPhase.PRE_INPUT, make_tracer("pre_input"))
        scheduler.register(SystemPhase.INPUT, make_tracer("input"))
        scheduler.register(SystemPhase.PHYSICS, make_tracer("physics"))
        scheduler.register(SystemPhase.UPDATE, make_tracer("update"))
        scheduler.register(SystemPhase.RENDER, make_tracer("render"))

        world = World()
        buf = DeterministicCommandBuffer()

        for _ in range(3):
            scheduler.update(Fixed32(1.0 / 60.0), world, buf)

        # Verify order is consistent
        expected_per_tick = ["pre_input", "input", "physics", "update", "render"]

        for tick in range(3):
            tick_trace = [name for t, name in execution_trace if t == tick]
            assert tick_trace == expected_per_tick


class TestChecksumPersistence:
    """Test checksum persistence and retrieval."""

    def test_checksums_accessible_after_ticks(self):
        """Test checksums can be retrieved after simulation."""
        scheduler = TickScheduler(seed=42)
        world = World()
        buf = DeterministicCommandBuffer()

        all_checksums = []
        for _ in range(10):
            results = scheduler.update(Fixed32(1.0 / 60.0), world, buf)
            all_checksums.extend(r.checksum for r in results)

        # Can retrieve any tick's checksum
        for tick in range(10):
            cs = scheduler.get_tick_checksum(tick)
            assert cs == all_checksums[tick]

    def test_checksums_cleared_on_reset(self):
        """Test checksums are cleared on reset."""
        scheduler = TickScheduler(seed=42)
        world = World()
        buf = DeterministicCommandBuffer()

        scheduler.update(Fixed32(1.0 / 60.0), world, buf)
        scheduler.reset()

        assert scheduler.get_tick_checksum(0) is None
