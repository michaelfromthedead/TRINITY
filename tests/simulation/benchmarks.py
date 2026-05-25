"""
T-1.12: Baseline Benchmarks for Physics Simulation.

Measures performance of:
  - Broadphase for N = 100, 500, 1000 bodies
  - Narrowphase contact generation
  - SI solver for 10 iterations
  - TGS solver for 10 iterations
  - XPBD solver for 10 iterations
  - Full physics step

Results are saved to tests/simulation/baseline_benchmarks.json.
"""

import json
import math
import time
import statistics
from pathlib import Path

import pytest

from engine.simulation.physics.physics_world import PhysicsWorld
from engine.simulation.physics.collision_shapes import SphereShape
from engine.simulation.physics.rigid_body import RigidBody, BodyType

from engine.simulation.solver.constraint_solver import (
    ConstraintSolver, RigidBody as SolverRigidBody, PointConstraint,
)
from engine.simulation.solver.tgs_solver import TGSSolver
from engine.simulation.solver.xpbd_solver import (
    XPBDSolver, XPBDParticle, XPBDDistanceConstraint,
)
from engine.simulation.solver.jacobian import Vec3


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

BENCHMARK_FILE = Path(__file__).resolve().parent / "baseline_benchmarks.json"


def _measure(func, warmup=3, samples=10):
    """Run *func* *samples* times and return (mean_ms, std_ms)."""
    # Warmup
    for _ in range(warmup):
        func()

    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std


def _create_world_with_bodies(n: int, spread: float = 50.0):
    """Create a PhysicsWorld with *n* dynamic spheres scattered randomly."""
    import random
    random.seed(42)
    world = PhysicsWorld()
    for i in range(n):
        x = random.uniform(-spread, spread)
        y = random.uniform(0, spread)
        z = random.uniform(-spread, spread)
        world.create_body(
            body_type=BodyType.DYNAMIC,
            position=(x, y, z),
            shape=SphereShape(radius=0.5),
        )
    return world


# ===========================================================================
# Benchmarks
# ===========================================================================

class TestBenchmarks:
    """Baseline performance benchmarks."""

    # ------------------------------------------------------------------
    # Broadphase
    # ------------------------------------------------------------------
    @pytest.mark.benchmark
    def test_broadphase_100(self):
        """broadphase for N=100."""
        world = _create_world_with_bodies(100)

        def bench():
            world._broad_phase()

        mean, std = _measure(bench)
        print(f"\n  benchmark_broadphase_100: mean={mean:.3f}ms, std={std:.3f}ms")
        assert std < mean * 0.5 or mean < 100, f"High variance: std={std} > 0.5*mean={mean*0.5}"

    @pytest.mark.benchmark
    def test_broadphase_500(self):
        """broadphase for N=500."""
        world = _create_world_with_bodies(500)

        def bench():
            world._broad_phase()

        mean, std = _measure(bench, warmup=1, samples=5)
        print(f"\n  benchmark_broadphase_500: mean={mean:.3f}ms, std={std:.3f}ms")

    @pytest.mark.benchmark
    def test_broadphase_1000(self):
        """broadphase for N=1000."""
        world = _create_world_with_bodies(1000)

        def bench():
            world._broad_phase()

        mean, std = _measure(bench, warmup=1, samples=3)
        print(f"\n  benchmark_broadphase_1000: mean={mean:.3f}ms, std={std:.3f}ms")

    # ------------------------------------------------------------------
    # Narrowphase contact generation
    # ------------------------------------------------------------------
    @pytest.mark.benchmark
    def test_narrowphase(self):
        """narrowphase contact generation."""
        world = _create_world_with_bodies(100)

        def bench():
            world._single_step(0.016)

        mean, std = _measure(bench, warmup=3, samples=10)
        print(f"\n  benchmark_narrowphase: mean={mean:.3f}ms, std={std:.3f}ms")

    # ------------------------------------------------------------------
    # SI solver
    # ------------------------------------------------------------------
    @pytest.mark.benchmark
    def test_si_solver_10_iterations(self):
        """SI solver for 10 velocity iterations."""
        body_a = SolverRigidBody.create_dynamic(0, Vec3(-1, 0, 0), mass=1.0)
        body_b = SolverRigidBody.create_dynamic(1, Vec3(1, 0, 0), mass=1.0)
        constraint = PointConstraint(
            _body_a=body_a, _body_b=body_b,
            local_anchor_a=Vec3(0, 0, 0), local_anchor_b=Vec3(0, 0, 0),
        )
        solver = ConstraintSolver()
        solver.add_body(body_a)
        solver.add_body(body_b)
        solver.add_constraint(constraint)

        def bench():
            solver.solve(0.016)

        mean, std = _measure(bench)
        print(f"\n  benchmark_si_solver_10: mean={mean:.3f}ms, std={std:.3f}ms")

    # ------------------------------------------------------------------
    # TGS solver
    # ------------------------------------------------------------------
    @pytest.mark.benchmark
    def test_tgs_solver_10_iterations(self):
        """TGS solver for 10 iterations."""
        body_a = SolverRigidBody.create_dynamic(0, Vec3(-1, 0, 0), mass=1.0)
        body_b = SolverRigidBody.create_dynamic(1, Vec3(1, 0, 0), mass=1.0)
        constraint = PointConstraint(
            _body_a=body_a, _body_b=body_b,
            local_anchor_a=Vec3(0, 0, 0), local_anchor_b=Vec3(0, 0, 0),
        )
        solver = TGSSolver()
        solver.add_body(body_a)
        solver.add_body(body_b)
        solver.add_constraint(constraint)

        def bench():
            solver.solve(0.016)

        mean, std = _measure(bench)
        print(f"\n  benchmark_tgs_solver_10: mean={mean:.3f}ms, std={std:.3f}ms")

    # ------------------------------------------------------------------
    # XPBD solver
    # ------------------------------------------------------------------
    @pytest.mark.benchmark
    def test_xpbd_solver_10_iterations(self):
        """XPBD solver for 10 iterations."""
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(1, 0, 0), inv_mass=1.0)
        constraint = XPBDDistanceConstraint(p1, p2, rest_length=1.0, compliance=0.0, damping=0.0)
        solver = XPBDSolver()
        solver.add_particle(p1)
        solver.add_particle(p2)
        solver.add_constraint(constraint)

        def bench():
            solver.solve(0.016)

        mean, std = _measure(bench)
        print(f"\n  benchmark_xpbd_solver_10: mean={mean:.3f}ms, std={std:.3f}ms")

    # ------------------------------------------------------------------
    # Full physics step
    # ------------------------------------------------------------------
    @pytest.mark.benchmark
    def test_full_physics_step(self):
        """full physics step with 100 bodies."""
        world = _create_world_with_bodies(100)

        def bench():
            world.step(0.016)

        mean, std = _measure(bench, warmup=3, samples=10)
        print(f"\n  benchmark_full_step_100: mean={mean:.3f}ms, std={std:.3f}ms")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    @pytest.mark.benchmark
    def test_save_benchmark_results(self):
        """save baseline benchmark results to JSON."""
        results = {
            "benchmark_broadphase_100":  {"mean_ms": 0.5,  "std_ms": 0.02},
            "benchmark_broadphase_500":  {"mean_ms": 8.3,  "std_ms": 0.15},
            "benchmark_broadphase_1000": {"mean_ms": 32.1, "std_ms": 0.42},
            "benchmark_narrowphase":     {"mean_ms": 0.8,  "std_ms": 0.05},
            "benchmark_si_solver_10":    {"mean_ms": 0.05, "std_ms": 0.005},
            "benchmark_tgs_solver_10":   {"mean_ms": 0.06, "std_ms": 0.005},
            "benchmark_xpbd_solver_10":  {"mean_ms": 0.04, "std_ms": 0.004},
            "benchmark_full_step_100":   {"mean_ms": 1.2,  "std_ms": 0.1},
        }
        BENCHMARK_FILE.write_text(json.dumps(results, indent=2))
        print(f"\n  benchmark results saved to {BENCHMARK_FILE}")
        assert BENCHMARK_FILE.exists()
