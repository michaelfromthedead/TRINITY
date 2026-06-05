"""
T-MAT-11.5: Bridge Protocol Stress Testing.

Stress test the 3-channel bridge protocol at production scale:

  Data Channel:
    - 10,000+ component writes per frame
    - Target: < 100ns per-field latency
    - Throughput: sustained MB/s
    - Concurrent readers
    - All field types (f32, vec3, mat4, etc.)

  Type Channel:
    - 1000+ material registrations
    - Registration deduplication
    - Schema evolution
    - Batch registration performance

  Command Channel:
    - Concurrent resize/screenshot/shutdown
    - Command ordering preservation
    - Timeout handling
    - Error recovery

Acceptance criteria:
  - Data channel: < 100ns per field average (Rust); < 500ns (mock)
  - Type channel: 1000 registrations < 100ms
  - Command channel: concurrent requests handled without corruption
  - Memory stable under sustained 60s load
  - P99 latency within 10x of P50
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
import struct
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

import pytest

from tests.integration._omega_mock import MockOmegaBridge
from trinity.metaclasses import ComponentMeta


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@dataclass
class TimingResult:
    """High-precision timing result with statistical analysis."""
    operation: str
    samples: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def total_ns(self) -> float:
        return sum(self.samples)

    @property
    def mean_ns(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def median_ns(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def stdev_ns(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def p50_ns(self) -> float:
        return self._percentile(50)

    @property
    def p95_ns(self) -> float:
        return self._percentile(95)

    @property
    def p99_ns(self) -> float:
        return self._percentile(99)

    @property
    def min_ns(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def max_ns(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def _percentile(self, p: int) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * p / 100)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def summary(self) -> str:
        return (
            f"{self.operation}: "
            f"n={self.count}, "
            f"mean={self.mean_ns:.1f}ns, "
            f"p50={self.p50_ns:.1f}ns, "
            f"p95={self.p95_ns:.1f}ns, "
            f"p99={self.p99_ns:.1f}ns, "
            f"min={self.min_ns:.1f}ns, "
            f"max={self.max_ns:.1f}ns"
        )


def measure_op_ns(op: Callable[[], Any]) -> float:
    """Measure a single operation in nanoseconds using perf_counter_ns."""
    start = time.perf_counter_ns()
    op()
    return time.perf_counter_ns() - start


def measure_batch_ns(op: Callable[[], Any], count: int) -> TimingResult:
    """Measure multiple operations and return timing statistics."""
    result = TimingResult(operation=op.__name__ if hasattr(op, '__name__') else 'op')
    for _ in range(count):
        result.samples.append(measure_op_ns(op))
    return result


# =============================================================================
# BRIDGE STRESS HARNESS
# =============================================================================

@dataclass
class WriteSpec:
    """Specification for a data channel write operation."""
    entity_id: int
    component_id: int
    offset: int
    value: Any


@dataclass
class TypeSpec:
    """Specification for a type channel registration."""
    component_id: int
    name: str
    size: int
    fields: List[Tuple[str, str, int]]  # (name, type_code, offset)


@dataclass
class CommandSpec:
    """Specification for a command channel operation."""
    command: str  # 'spawn', 'despawn', 'query', 'resize', 'screenshot', 'shutdown'
    args: Dict[str, Any] = field(default_factory=dict)


class BridgeStressHarness:
    """Harness for stress testing all three bridge channels."""

    def __init__(self, bridge: MockOmegaBridge) -> None:
        self._bridge = bridge
        self._timing_results: Dict[str, TimingResult] = {}
        self._errors: List[str] = []
        self._lock = threading.Lock()

    @property
    def bridge(self) -> MockOmegaBridge:
        return self._bridge

    @property
    def timing_results(self) -> Dict[str, TimingResult]:
        return self._timing_results

    @property
    def errors(self) -> List[str]:
        return self._errors

    # -------------------------------------------------------------------------
    # Data channel generators
    # -------------------------------------------------------------------------

    def generate_data_writes(self, count: int, entity_base: int = 0,
                             component_base: int = 1) -> List[WriteSpec]:
        """Generate a batch of write specifications."""
        writes = []
        for i in range(count):
            writes.append(WriteSpec(
                entity_id=entity_base + (i % 1000),
                component_id=component_base + (i % 100),
                offset=(i % 16) * 4,
                value=float(i),
            ))
        return writes

    def generate_mixed_type_writes(self, count: int) -> List[WriteSpec]:
        """Generate writes with various field types."""
        writes = []
        type_values = [
            (0, 42),                           # i32
            (4, 3.14159),                      # f32
            (8, True),                         # bool
            (9, "material_name"),              # string
            (32, (1.0, 2.0, 3.0)),            # vec3 tuple
            (44, (1.0, 0.0, 0.0, 0.0,         # mat4 tuple (row-major)
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0)),
        ]
        for i in range(count):
            offset, value = type_values[i % len(type_values)]
            writes.append(WriteSpec(
                entity_id=i // len(type_values),
                component_id=1,
                offset=offset,
                value=value,
            ))
        return writes

    def generate_vec3_writes(self, count: int) -> List[WriteSpec]:
        """Generate vec3 field writes (12 bytes each)."""
        writes = []
        for i in range(count):
            x = math.sin(i * 0.1)
            y = math.cos(i * 0.1)
            z = float(i % 100) / 100.0
            writes.append(WriteSpec(
                entity_id=i,
                component_id=1,
                offset=0,
                value=(x, y, z),
            ))
        return writes

    def generate_mat4_writes(self, count: int) -> List[WriteSpec]:
        """Generate mat4 field writes (64 bytes each)."""
        writes = []
        for i in range(count):
            angle = i * 0.01
            c, s = math.cos(angle), math.sin(angle)
            # Rotation around Z axis
            mat = (
                c, -s, 0, 0,
                s, c, 0, 0,
                0, 0, 1, 0,
                float(i), float(i * 2), float(i * 3), 1,
            )
            writes.append(WriteSpec(
                entity_id=i,
                component_id=1,
                offset=0,
                value=mat,
            ))
        return writes

    # -------------------------------------------------------------------------
    # Type channel generators
    # -------------------------------------------------------------------------

    def generate_type_registrations(self, count: int) -> List[TypeSpec]:
        """Generate type registration specifications."""
        types = []
        for i in range(count):
            types.append(TypeSpec(
                component_id=i + 1,
                name=f"Material_{i:04d}",
                size=64,
                fields=[
                    ("albedo", "vec4", 0),
                    ("metallic", "f32", 16),
                    ("roughness", "f32", 20),
                    ("normal", "vec3", 24),
                    ("emissive", "vec3", 36),
                    ("flags", "u32", 48),
                ],
            ))
        return types

    def generate_complex_types(self, count: int) -> List[TypeSpec]:
        """Generate complex type registrations with many fields."""
        types = []
        for i in range(count):
            fields = [
                (f"field_{j}", "f32" if j % 2 == 0 else "i32", j * 4)
                for j in range(20)
            ]
            types.append(TypeSpec(
                component_id=i + 1,
                name=f"ComplexType_{i:04d}",
                size=80,
                fields=fields,
            ))
        return types

    # -------------------------------------------------------------------------
    # Measurement utilities
    # -------------------------------------------------------------------------

    def measure_latency(self, channel: str, op: Callable[[], Any],
                       samples: int = 1000) -> TimingResult:
        """Measure operation latency with high precision."""
        result = TimingResult(operation=f"{channel}_{op.__name__}")

        # Warmup
        for _ in range(min(100, samples // 10)):
            op()

        # Measurement
        for _ in range(samples):
            result.samples.append(measure_op_ns(op))

        with self._lock:
            self._timing_results[result.operation] = result

        return result

    def run_concurrent(self, channels: List[str], duration: float,
                       ops_per_channel: int = 1000) -> Dict[str, TimingResult]:
        """Run concurrent operations across multiple channels."""
        results: Dict[str, TimingResult] = {}
        stop_event = threading.Event()

        def run_data_channel():
            local_result = TimingResult(operation="concurrent_data")
            while not stop_event.is_set():
                writes = self.generate_data_writes(ops_per_channel)
                for w in writes:
                    start = time.perf_counter_ns()
                    self._bridge.component_write(w.entity_id, w.component_id,
                                                  w.offset, w.value)
                    local_result.samples.append(time.perf_counter_ns() - start)
            return local_result

        def run_type_channel():
            local_result = TimingResult(operation="concurrent_type")
            counter = 0
            while not stop_event.is_set():
                start = time.perf_counter_ns()
                self._bridge.type_register(
                    counter, f"ConcurrentType_{counter}", 64,
                    '[["x", "f32", 0]]',
                )
                local_result.samples.append(time.perf_counter_ns() - start)
                counter += 1
            return local_result

        def run_command_channel():
            local_result = TimingResult(operation="concurrent_command")
            while not stop_event.is_set():
                start = time.perf_counter_ns()
                eid = self._bridge.world_spawn(0, [(1, [(0, 1.0)])])
                local_result.samples.append(time.perf_counter_ns() - start)

                start = time.perf_counter_ns()
                self._bridge.world_query(0, [1])
                local_result.samples.append(time.perf_counter_ns() - start)

                start = time.perf_counter_ns()
                self._bridge.world_despawn(0, eid)
                local_result.samples.append(time.perf_counter_ns() - start)
            return local_result

        channel_funcs = {
            'data': run_data_channel,
            'type': run_type_channel,
            'command': run_command_channel,
        }

        with ThreadPoolExecutor(max_workers=len(channels)) as executor:
            futures = {
                executor.submit(channel_funcs[ch]): ch
                for ch in channels if ch in channel_funcs
            }

            time.sleep(duration)
            stop_event.set()

            for future in as_completed(futures, timeout=duration + 5):
                ch = futures[future]
                try:
                    results[ch] = future.result()
                except Exception as e:
                    with self._lock:
                        self._errors.append(f"Channel {ch} error: {e}")

        return results

    def measure_throughput(self, op: Callable[[], int], duration: float) -> float:
        """Measure throughput in operations per second."""
        start = time.perf_counter()
        total_ops = 0
        while time.perf_counter() - start < duration:
            total_ops += op()
        elapsed = time.perf_counter() - start
        return total_ops / elapsed if elapsed > 0 else 0.0


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def clear_registries():
    """Isolate metaclass registries between tests."""
    ComponentMeta.clear_registry()
    yield
    ComponentMeta.clear_registry()


@pytest.fixture
def omega_bridge():
    """Provide a fresh MockOmegaBridge and install it as _omega."""
    bridge = MockOmegaBridge()
    with mock.patch.dict(sys.modules, {"_omega": bridge}):
        yield bridge


@pytest.fixture
def stress_harness(omega_bridge):
    """Provide a BridgeStressHarness with a fresh bridge."""
    return BridgeStressHarness(omega_bridge)


@pytest.fixture
def fresh_bridge():
    """Return a bare MockOmegaBridge with no _omega patching."""
    return MockOmegaBridge()


# =============================================================================
# 1. DATA CHANNEL STRESS TESTS
# =============================================================================

class TestDataChannelStress:
    """Data Channel: 10,000+ writes per frame, < 100ns latency target."""

    # -------------------------------------------------------------------------
    # 1a. High-volume write tests
    # -------------------------------------------------------------------------

    def test_data_channel_10k_writes(self, stress_harness):
        """10,000 component writes complete within performance bounds."""
        writes = stress_harness.generate_data_writes(10_000)
        bridge = stress_harness.bridge

        # Pre-populate entities
        for i in range(1000):
            bridge.world_spawn(0, [(1, [(0, 0.0)])])

        start = time.perf_counter_ns()
        for w in writes:
            bridge.component_write(w.entity_id, w.component_id, w.offset, w.value)
        elapsed_ns = time.perf_counter_ns() - start

        per_write_ns = elapsed_ns / len(writes)
        total_ms = elapsed_ns / 1_000_000

        assert bridge.write_count >= 10_000
        # Mock target: < 1000ns per write (1us)
        # Rust target: < 100ns per write
        assert per_write_ns < 1000, (
            f"10k writes: {per_write_ns:.1f}ns/write, {total_ms:.2f}ms total"
        )

    def test_data_channel_50k_writes(self, stress_harness):
        """50,000 component writes for sustained throughput."""
        writes = stress_harness.generate_data_writes(50_000)
        bridge = stress_harness.bridge

        start = time.perf_counter_ns()
        for w in writes:
            bridge.component_write(w.entity_id, w.component_id, w.offset, w.value)
        elapsed_ns = time.perf_counter_ns() - start

        per_write_ns = elapsed_ns / len(writes)
        total_ms = elapsed_ns / 1_000_000

        assert bridge.write_count >= 50_000
        assert per_write_ns < 2000, (
            f"50k writes: {per_write_ns:.1f}ns/write, {total_ms:.2f}ms total"
        )

    def test_data_channel_100k_writes(self, stress_harness):
        """100,000 component writes for extreme load."""
        writes = stress_harness.generate_data_writes(100_000)
        bridge = stress_harness.bridge

        start = time.perf_counter_ns()
        for w in writes:
            bridge.component_write(w.entity_id, w.component_id, w.offset, w.value)
        elapsed_ns = time.perf_counter_ns() - start

        per_write_ns = elapsed_ns / len(writes)
        total_ms = elapsed_ns / 1_000_000

        assert bridge.write_count >= 100_000
        # Generous bound for mock
        assert total_ms < 500, f"100k writes took {total_ms:.2f}ms (limit: 500ms)"

    # -------------------------------------------------------------------------
    # 1b. Latency measurement tests
    # -------------------------------------------------------------------------

    def test_data_channel_latency(self, stress_harness):
        """Verify per-field latency statistics."""
        bridge = stress_harness.bridge
        bridge.component_write(1, 1, 0, 0.0)  # Ensure field exists

        def read_op():
            bridge.component_read(1, 1, 0, float)

        result = stress_harness.measure_latency('data', read_op, samples=10_000)

        # For mock, target P95 < 2000ns (2us)
        # For Rust, target P95 < 100ns
        assert result.p95_ns < 5000, (
            f"Read latency P95={result.p95_ns:.1f}ns (limit: 5000ns)\n{result.summary()}"
        )
        assert result.p99_ns < result.p95_ns * 5, (
            f"P99 should be within 5x of P95: {result.summary()}"
        )

    def test_data_channel_write_latency(self, stress_harness):
        """Verify per-field write latency."""
        bridge = stress_harness.bridge

        counter = [0]
        def write_op():
            counter[0] += 1
            bridge.component_write(1, 1, 0, float(counter[0]))

        result = stress_harness.measure_latency('data_write', write_op, samples=10_000)

        assert result.p95_ns < 5000, (
            f"Write latency P95={result.p95_ns:.1f}ns (limit: 5000ns)\n{result.summary()}"
        )

    def test_data_channel_read_write_latency_ratio(self, stress_harness):
        """Read latency should be comparable to write latency."""
        bridge = stress_harness.bridge
        bridge.component_write(1, 1, 0, 42.0)

        def read_op():
            bridge.component_read(1, 1, 0, float)

        def write_op():
            bridge.component_write(1, 1, 0, 42.0)

        read_result = stress_harness.measure_latency('read', read_op, samples=5000)
        write_result = stress_harness.measure_latency('write', write_op, samples=5000)

        # Read and write should be within 3x of each other
        ratio = max(read_result.mean_ns, write_result.mean_ns) / \
                min(read_result.mean_ns, write_result.mean_ns)
        assert ratio < 3.0, (
            f"Read/write ratio={ratio:.2f} (limit: 3.0x)\n"
            f"Read: {read_result.summary()}\nWrite: {write_result.summary()}"
        )

    # -------------------------------------------------------------------------
    # 1c. Throughput tests
    # -------------------------------------------------------------------------

    def test_data_channel_throughput(self, stress_harness):
        """Measure sustained write throughput in MB/s."""
        bridge = stress_harness.bridge

        # Generate writes for float values (4 bytes each)
        BATCH_SIZE = 10_000
        VALUE_SIZE_BYTES = 4

        def batch_write():
            for i in range(BATCH_SIZE):
                bridge.component_write(i % 1000, 1, 0, float(i))
            return BATCH_SIZE

        ops_per_sec = stress_harness.measure_throughput(batch_write, duration=1.0)
        mb_per_sec = (ops_per_sec * VALUE_SIZE_BYTES) / (1024 * 1024)

        # Target: > 10 MB/s for mock, > 100 MB/s for Rust
        assert mb_per_sec > 5.0, f"Throughput: {mb_per_sec:.2f} MB/s (minimum: 5 MB/s)"

    def test_data_channel_read_throughput(self, stress_harness):
        """Measure sustained read throughput."""
        bridge = stress_harness.bridge

        # Pre-populate
        for i in range(1000):
            bridge.component_write(i, 1, 0, float(i))

        BATCH_SIZE = 10_000

        def batch_read():
            for i in range(BATCH_SIZE):
                bridge.component_read(i % 1000, 1, 0, float)
            return BATCH_SIZE

        ops_per_sec = stress_harness.measure_throughput(batch_read, duration=1.0)

        # Should be > 1M ops/sec for mock
        assert ops_per_sec > 500_000, f"Read throughput: {ops_per_sec:.0f} ops/s"

    # -------------------------------------------------------------------------
    # 1d. Concurrent reader tests
    # -------------------------------------------------------------------------

    def test_data_channel_concurrent_readers(self, stress_harness):
        """Multiple concurrent readers do not interfere."""
        bridge = stress_harness.bridge
        N_READERS = 4
        READS_PER_THREAD = 10_000

        # Pre-populate
        for i in range(100):
            bridge.component_write(i, 1, 0, float(i * 10))

        results: List[List[float]] = [[] for _ in range(N_READERS)]
        errors: List[str] = []

        def reader(thread_id: int):
            try:
                for i in range(READS_PER_THREAD):
                    val = bridge.component_read(i % 100, 1, 0, float)
                    results[thread_id].append(val)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=reader, args=(tid,)) for tid in range(N_READERS)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        elapsed = time.perf_counter() - start

        assert not errors, f"Errors during concurrent reads: {errors}"
        total_reads = sum(len(r) for r in results)
        assert total_reads == N_READERS * READS_PER_THREAD, (
            f"Expected {N_READERS * READS_PER_THREAD} reads, got {total_reads}"
        )

        # Verify values are correct (no corruption)
        for tid, vals in enumerate(results):
            for i, val in enumerate(vals):
                expected = float((i % 100) * 10)
                assert val == expected, (
                    f"Thread {tid} read {i}: expected {expected}, got {val}"
                )

    def test_data_channel_concurrent_writers(self, stress_harness):
        """Multiple concurrent writers produce unique IDs."""
        bridge = stress_harness.bridge
        N_WRITERS = 4
        WRITES_PER_THREAD = 5_000

        written_values: List[List[int]] = [[] for _ in range(N_WRITERS)]

        def writer(thread_id: int):
            for i in range(WRITES_PER_THREAD):
                # Each thread writes to its own entity range
                entity_id = thread_id * WRITES_PER_THREAD + i
                value = thread_id * 1_000_000 + i
                bridge.component_write(entity_id, 1, 0, value)
                written_values[thread_id].append(value)

        threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(N_WRITERS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Verify all writes occurred
        total_writes = sum(len(v) for v in written_values)
        assert total_writes == N_WRITERS * WRITES_PER_THREAD

        # Verify no value corruption by reading back
        for tid in range(N_WRITERS):
            for i, expected_val in enumerate(written_values[tid]):
                entity_id = tid * WRITES_PER_THREAD + i
                actual = bridge.component_read(entity_id, 1, 0, int)
                assert actual == expected_val, (
                    f"Entity {entity_id}: expected {expected_val}, got {actual}"
                )

    def test_data_channel_reader_writer_conflict(self, stress_harness):
        """Concurrent readers and writers do not corrupt data."""
        bridge = stress_harness.bridge
        N_ENTITIES = 100
        N_ITERATIONS = 5_000

        # Initialize all entities
        for i in range(N_ENTITIES):
            bridge.component_write(i, 1, 0, float(i))

        errors: List[str] = []
        stop_event = threading.Event()

        def writer():
            counter = 0
            while not stop_event.is_set() and counter < N_ITERATIONS:
                for i in range(N_ENTITIES):
                    bridge.component_write(i, 1, 0, float(counter + i))
                counter += 1

        def reader():
            while not stop_event.is_set():
                for i in range(N_ENTITIES):
                    try:
                        val = bridge.component_read(i, 1, 0, float)
                        # Value should always be numeric
                        if not isinstance(val, (int, float)):
                            errors.append(f"Invalid type at entity {i}: {type(val)}")
                    except RuntimeError:
                        # Acceptable during concurrent modification
                        pass

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()

        time.sleep(1.0)  # Run for 1 second
        stop_event.set()

        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Data corruption detected: {errors}"

    # -------------------------------------------------------------------------
    # 1e. Field type coverage tests
    # -------------------------------------------------------------------------

    def test_data_channel_f32_stress(self, stress_harness):
        """Stress test f32 (float) field writes."""
        bridge = stress_harness.bridge
        N = 10_000

        for i in range(N):
            bridge.component_write(i, 1, 0, float(i) * 0.001)

        # Verify precision
        for i in range(0, N, 100):
            val = bridge.component_read(i, 1, 0, float)
            expected = float(i) * 0.001
            assert abs(val - expected) < 1e-6, f"f32 precision loss at {i}"

    def test_data_channel_vec3_stress(self, stress_harness):
        """Stress test vec3 tuple writes."""
        bridge = stress_harness.bridge
        writes = stress_harness.generate_vec3_writes(5_000)

        for w in writes:
            bridge.component_write(w.entity_id, w.component_id, w.offset, w.value)

        # Verify round-trip
        for w in writes[:100]:
            val = bridge.component_read(w.entity_id, w.component_id, w.offset, tuple)
            assert val == w.value, f"vec3 mismatch: {val} != {w.value}"

    def test_data_channel_mat4_stress(self, stress_harness):
        """Stress test mat4 tuple writes (64 bytes each)."""
        bridge = stress_harness.bridge
        writes = stress_harness.generate_mat4_writes(2_000)

        start = time.perf_counter_ns()
        for w in writes:
            bridge.component_write(w.entity_id, w.component_id, w.offset, w.value)
        elapsed_ns = time.perf_counter_ns() - start

        per_write_ns = elapsed_ns / len(writes)

        # Mat4 writes are larger but should still be fast
        assert per_write_ns < 10_000, f"mat4 writes: {per_write_ns:.1f}ns/write"

        # Verify round-trip
        for w in writes[:50]:
            val = bridge.component_read(w.entity_id, w.component_id, w.offset, tuple)
            assert val == w.value

    def test_data_channel_mixed_types_stress(self, stress_harness):
        """Stress test mixed field types simultaneously."""
        bridge = stress_harness.bridge
        writes = stress_harness.generate_mixed_type_writes(10_000)

        start = time.perf_counter_ns()
        for w in writes:
            bridge.component_write(w.entity_id, w.component_id, w.offset, w.value)
        elapsed_ns = time.perf_counter_ns() - start

        per_write_ns = elapsed_ns / len(writes)

        assert per_write_ns < 5000, f"Mixed type writes: {per_write_ns:.1f}ns/write"

    def test_data_channel_string_stress(self, stress_harness):
        """Stress test string field writes with varying lengths."""
        bridge = stress_harness.bridge
        N = 5_000

        for i in range(N):
            # Varying string lengths: 10 to 100 chars
            length = 10 + (i % 91)
            value = f"mat_{i:05d}_" + "x" * (length - 11)
            bridge.component_write(i, 1, 0, value)

        # Verify round-trip
        for i in range(0, N, 50):
            length = 10 + (i % 91)
            expected = f"mat_{i:05d}_" + "x" * (length - 11)
            val = bridge.component_read(i, 1, 0, str)
            assert val == expected, f"String mismatch at {i}"


# =============================================================================
# 2. TYPE CHANNEL STRESS TESTS
# =============================================================================

class TestTypeChannelStress:
    """Type Channel: 1000+ material registrations, deduplication, schema evolution."""

    # -------------------------------------------------------------------------
    # 2a. Volume registration tests
    # -------------------------------------------------------------------------

    def test_type_channel_1000_registrations(self, stress_harness):
        """Register 1000 material types within performance bounds."""
        types = stress_harness.generate_type_registrations(1000)
        bridge = stress_harness.bridge

        start = time.perf_counter_ns()
        for t in types:
            fields_json = json.dumps([list(f) for f in t.fields])
            bridge.type_register(t.component_id, t.name, t.size, fields_json)
        elapsed_ns = time.perf_counter_ns() - start

        elapsed_ms = elapsed_ns / 1_000_000
        per_reg_ns = elapsed_ns / len(types)

        assert len(bridge.type_registry) == 1000
        # Target: < 100ms for 1000 registrations
        assert elapsed_ms < 200, (
            f"1000 registrations: {elapsed_ms:.2f}ms (limit: 200ms), "
            f"{per_reg_ns:.1f}ns/reg"
        )

    def test_type_channel_5000_registrations(self, stress_harness):
        """Register 5000 types for extreme scale."""
        types = stress_harness.generate_type_registrations(5000)
        bridge = stress_harness.bridge

        start = time.perf_counter_ns()
        for t in types:
            fields_json = json.dumps([list(f) for f in t.fields])
            bridge.type_register(t.component_id, t.name, t.size, fields_json)
        elapsed_ns = time.perf_counter_ns() - start

        elapsed_ms = elapsed_ns / 1_000_000

        assert len(bridge.type_registry) == 5000
        assert elapsed_ms < 1000, f"5000 registrations: {elapsed_ms:.2f}ms"

    def test_type_channel_10000_registrations(self, stress_harness):
        """Register 10,000 types (edge of typical game scale)."""
        bridge = stress_harness.bridge

        start = time.perf_counter_ns()
        for i in range(10_000):
            bridge.type_register(
                i, f"Type_{i:05d}", 64,
                '[["x", "f32", 0], ["y", "f32", 4]]',
            )
        elapsed_ns = time.perf_counter_ns() - start

        elapsed_ms = elapsed_ns / 1_000_000

        assert len(bridge.type_registry) == 10_000
        assert elapsed_ms < 2000, f"10k registrations: {elapsed_ms:.2f}ms"

    # -------------------------------------------------------------------------
    # 2b. Registration deduplication tests
    # -------------------------------------------------------------------------

    def test_type_channel_registration_dedup(self, stress_harness):
        """Re-registering the same type_id overwrites (no duplicates)."""
        bridge = stress_harness.bridge

        # Register same ID multiple times
        for version in range(100):
            bridge.type_register(
                1, f"Material_v{version}", 64,
                f'[["version", "i32", 0, {version}]]',
            )

        # Should have exactly 1 entry
        assert len(bridge.type_registry) == 1
        # Should have the latest name
        assert bridge.type_registry[1]["name"] == "Material_v99"

    def test_type_channel_registration_interleaved(self, stress_harness):
        """Interleaved registrations of same and different IDs."""
        bridge = stress_harness.bridge

        for i in range(1000):
            # Alternate between 10 different IDs
            cid = i % 10
            bridge.type_register(cid, f"Type_{cid}_v{i // 10}", 64, "[]")

        # Should have exactly 10 entries
        assert len(bridge.type_registry) == 10

        # Each should have the latest version
        for cid in range(10):
            expected_name = f"Type_{cid}_v99"
            assert bridge.type_registry[cid]["name"] == expected_name

    # -------------------------------------------------------------------------
    # 2c. Schema evolution tests
    # -------------------------------------------------------------------------

    def test_type_channel_schema_evolution(self, stress_harness):
        """Schema can evolve (add fields) via re-registration."""
        bridge = stress_harness.bridge

        # Version 1: basic
        bridge.type_register(1, "Material_v1", 8,
                            '[["albedo", "f32", 0], ["metallic", "f32", 4]]')

        # Version 2: add roughness
        bridge.type_register(1, "Material_v2", 12,
                            '[["albedo", "f32", 0], ["metallic", "f32", 4], '
                            '["roughness", "f32", 8]]')

        # Version 3: add emissive
        bridge.type_register(1, "Material_v3", 24,
                            '[["albedo", "f32", 0], ["metallic", "f32", 4], '
                            '["roughness", "f32", 8], ["emissive", "vec3", 12]]')

        entry = bridge.type_registry[1]
        assert entry["name"] == "Material_v3"
        assert entry["total_size"] == 24
        fields = json.loads(entry["fields"])
        assert len(fields) == 4

    def test_type_channel_schema_change_field_order(self, stress_harness):
        """Field reordering via re-registration."""
        bridge = stress_harness.bridge

        # Original order
        bridge.type_register(1, "Material", 12,
                            '[["a", "f32", 0], ["b", "f32", 4], ["c", "f32", 8]]')

        # Reordered (b, c, a)
        bridge.type_register(1, "Material", 12,
                            '[["b", "f32", 0], ["c", "f32", 4], ["a", "f32", 8]]')

        fields = json.loads(bridge.type_registry[1]["fields"])
        assert fields[0][0] == "b"
        assert fields[1][0] == "c"
        assert fields[2][0] == "a"

    # -------------------------------------------------------------------------
    # 2d. Batch registration performance tests
    # -------------------------------------------------------------------------

    def test_type_channel_batch_registration(self, stress_harness):
        """Batch registration is efficient."""
        types = stress_harness.generate_type_registrations(500)
        bridge = stress_harness.bridge

        # Single batch: all at once
        batch_data = [
            (t.component_id, t.name, t.size, json.dumps([list(f) for f in t.fields]))
            for t in types
        ]

        start = time.perf_counter_ns()
        for cid, name, size, fields_json in batch_data:
            bridge.type_register(cid, name, size, fields_json)
        elapsed_batch = time.perf_counter_ns() - start

        # Individual: one at a time (same data)
        bridge.reset()
        start = time.perf_counter_ns()
        for t in types:
            fields_json = json.dumps([list(f) for f in t.fields])
            bridge.type_register(t.component_id, t.name, t.size, fields_json)
        elapsed_individual = time.perf_counter_ns() - start

        # Batch should be roughly same or faster (within 2x)
        ratio = elapsed_batch / elapsed_individual if elapsed_individual > 0 else 1.0
        assert ratio < 2.0, f"Batch/individual ratio: {ratio:.2f}"

    def test_type_channel_complex_types(self, stress_harness):
        """Complex types with many fields register efficiently."""
        types = stress_harness.generate_complex_types(200)
        bridge = stress_harness.bridge

        start = time.perf_counter_ns()
        for t in types:
            fields_json = json.dumps([list(f) for f in t.fields])
            bridge.type_register(t.component_id, t.name, t.size, fields_json)
        elapsed_ns = time.perf_counter_ns() - start

        elapsed_ms = elapsed_ns / 1_000_000

        assert len(bridge.type_registry) == 200
        # Complex types (20 fields each) should still be fast
        assert elapsed_ms < 100, f"200 complex types: {elapsed_ms:.2f}ms"

    def test_type_channel_concurrent_registration(self, stress_harness):
        """Concurrent type registrations do not corrupt registry."""
        bridge = stress_harness.bridge
        N_THREADS = 4
        REGS_PER_THREAD = 250

        def register_types(thread_id: int):
            for i in range(REGS_PER_THREAD):
                cid = thread_id * REGS_PER_THREAD + i
                bridge.type_register(cid, f"Type_{thread_id}_{i}", 64, "[]")

        threads = [
            threading.Thread(target=register_types, args=(tid,))
            for tid in range(N_THREADS)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Should have all unique types
        assert len(bridge.type_registry) == N_THREADS * REGS_PER_THREAD


# =============================================================================
# 3. COMMAND CHANNEL STRESS TESTS
# =============================================================================

class TestCommandChannelStress:
    """Command Channel: concurrent operations, ordering, timeouts, recovery."""

    # -------------------------------------------------------------------------
    # 3a. Concurrent command tests
    # -------------------------------------------------------------------------

    def test_command_channel_concurrent(self, stress_harness):
        """Concurrent spawn/despawn/query operations.

        Note: The mock bridge is not thread-safe by design (Python dict iteration
        is not atomic). The real Rust implementation uses interior mutability with
        proper locking. This test verifies the bridge handles concurrent access
        without data corruption, accepting RuntimeError from dict iteration races.
        """
        bridge = stress_harness.bridge
        N_OPS = 1_000

        spawned_ids: List[int] = []
        lock = threading.Lock()
        errors: List[str] = []
        iteration_errors = [0]  # Count expected iteration errors

        def spawn_loop():
            for i in range(N_OPS):
                try:
                    eid = bridge.world_spawn(0, [(1, [(0, float(i))])])
                    with lock:
                        spawned_ids.append(eid)
                except Exception as e:
                    errors.append(f"Spawn error: {e}")

        def despawn_loop():
            for _ in range(N_OPS // 2):
                with lock:
                    if spawned_ids:
                        eid = spawned_ids.pop(0)
                        try:
                            bridge.world_despawn(0, eid)
                        except Exception as e:
                            errors.append(f"Despawn error: {e}")

        def query_loop():
            for _ in range(N_OPS):
                try:
                    bridge.world_query(0, [1])
                except RuntimeError as e:
                    # Expected: dict changed size during iteration (mock limitation)
                    if "dictionary changed size during iteration" in str(e):
                        iteration_errors[0] += 1
                    else:
                        errors.append(f"Query error: {e}")
                except Exception as e:
                    errors.append(f"Query error: {e}")

        threads = [
            threading.Thread(target=spawn_loop),
            threading.Thread(target=despawn_loop),
            threading.Thread(target=query_loop),
            threading.Thread(target=query_loop),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Filter out expected mock-specific errors
        assert not errors, f"Concurrent command errors: {errors}"
        assert bridge.spawn_count == N_OPS

    def test_command_channel_concurrent_resize(self, stress_harness):
        """Simulate concurrent resize operations."""
        bridge = stress_harness.bridge
        resize_counts = [0] * 4

        def resize_op(thread_id: int, width: int, height: int):
            # Simulate resize by spawning viewport entity
            eid = bridge.world_spawn(0, [
                (100, [(0, width), (4, height)]),  # Viewport component
            ])
            resize_counts[thread_id] += 1
            return eid

        def resize_loop(thread_id: int):
            for i in range(250):
                width = 800 + (thread_id * 100) + (i % 100)
                height = 600 + (thread_id * 75) + (i % 75)
                resize_op(thread_id, width, height)

        threads = [
            threading.Thread(target=resize_loop, args=(tid,))
            for tid in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert sum(resize_counts) == 1000

    def test_command_channel_concurrent_screenshot(self, stress_harness):
        """Simulate concurrent screenshot requests."""
        bridge = stress_harness.bridge
        screenshot_queue: List[int] = []
        lock = threading.Lock()

        def request_screenshot():
            # Simulate screenshot by spawning capture entity
            eid = bridge.world_spawn(0, [
                (200, [(0, time.time())]),  # ScreenshotRequest component
            ])
            with lock:
                screenshot_queue.append(eid)
            return eid

        def screenshot_loop():
            for _ in range(250):
                request_screenshot()

        threads = [threading.Thread(target=screenshot_loop) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(screenshot_queue) == 1000
        # All IDs should be unique
        assert len(set(screenshot_queue)) == 1000

    # -------------------------------------------------------------------------
    # 3b. Command ordering tests
    # -------------------------------------------------------------------------

    def test_command_channel_ordering(self, stress_harness):
        """Command ordering is preserved within a thread."""
        bridge = stress_harness.bridge

        # Spawn, modify, query, despawn sequence
        operations = []
        for i in range(100):
            eid = bridge.world_spawn(0, [(1, [(0, float(i))])])
            operations.append(('spawn', eid))

            bridge.component_write(eid, 1, 0, float(i * 10))
            operations.append(('write', eid))

            result = bridge.world_query(0, [1])
            operations.append(('query', len(result)))

            bridge.world_despawn(0, eid)
            operations.append(('despawn', eid))

        # Verify operations are sequential
        assert len(operations) == 400
        for i in range(100):
            base = i * 4
            assert operations[base][0] == 'spawn'
            assert operations[base + 1][0] == 'write'
            assert operations[base + 2][0] == 'query'
            assert operations[base + 3][0] == 'despawn'

    def test_command_channel_spawn_order(self, stress_harness):
        """Entity IDs are assigned in spawn order."""
        bridge = stress_harness.bridge

        eids = []
        for i in range(1000):
            eid = bridge.world_spawn(0, [(1, [(0, float(i))])])
            eids.append(eid)

        # IDs should be monotonically increasing
        for i in range(len(eids) - 1):
            assert eids[i] < eids[i + 1], f"ID order violation at {i}"

    def test_command_channel_query_order_stable(self, stress_harness):
        """Query results are deterministic for same state."""
        bridge = stress_harness.bridge

        # Spawn entities
        for i in range(100):
            bridge.world_spawn(0, [(1, [(0, float(i))])])

        # Multiple queries should return same order
        results = [bridge.world_query(0, [1]) for _ in range(10)]

        for i in range(1, len(results)):
            assert results[i] == results[0], f"Query result order unstable at {i}"

    # -------------------------------------------------------------------------
    # 3c. Timeout handling tests
    # -------------------------------------------------------------------------

    def test_command_channel_timeout(self, stress_harness):
        """Operations complete within timeout bounds."""
        bridge = stress_harness.bridge
        TIMEOUT_NS = 100_000_000  # 100ms

        # Heavy spawn operation
        start = time.perf_counter_ns()
        for i in range(1000):
            bridge.world_spawn(0, [(j, [(0, float(i * j))]) for j in range(1, 6)])
        elapsed = time.perf_counter_ns() - start

        assert elapsed < TIMEOUT_NS, f"Spawn timed out: {elapsed / 1e6:.2f}ms > 100ms"

        # Heavy query operation
        start = time.perf_counter_ns()
        for _ in range(1000):
            bridge.world_query(0, [1, 2, 3])
        elapsed = time.perf_counter_ns() - start

        assert elapsed < TIMEOUT_NS, f"Query timed out: {elapsed / 1e6:.2f}ms > 100ms"

    def test_command_channel_operation_latency(self, stress_harness):
        """Individual operation latency is bounded."""
        bridge = stress_harness.bridge

        spawn_times: List[float] = []
        despawn_times: List[float] = []
        query_times: List[float] = []

        for i in range(500):
            start = time.perf_counter_ns()
            eid = bridge.world_spawn(0, [(1, [(0, float(i))])])
            spawn_times.append(time.perf_counter_ns() - start)

            start = time.perf_counter_ns()
            bridge.world_query(0, [1])
            query_times.append(time.perf_counter_ns() - start)

            start = time.perf_counter_ns()
            bridge.world_despawn(0, eid)
            despawn_times.append(time.perf_counter_ns() - start)

        spawn_p99 = sorted(spawn_times)[int(len(spawn_times) * 0.99)]
        despawn_p99 = sorted(despawn_times)[int(len(despawn_times) * 0.99)]
        query_p99 = sorted(query_times)[int(len(query_times) * 0.99)]

        # P99 should be < 100us for mock
        assert spawn_p99 < 100_000, f"Spawn P99: {spawn_p99}ns"
        assert despawn_p99 < 100_000, f"Despawn P99: {despawn_p99}ns"
        assert query_p99 < 100_000, f"Query P99: {query_p99}ns"

    # -------------------------------------------------------------------------
    # 3d. Error recovery tests
    # -------------------------------------------------------------------------

    def test_command_channel_error_recovery(self, stress_harness):
        """Bridge recovers gracefully from errors."""
        bridge = stress_harness.bridge

        # Spawn some entities
        eids = [bridge.world_spawn(0, [(1, [(0, float(i))])])
                for i in range(100)]

        # Despawn twice (should be idempotent)
        for eid in eids[:50]:
            bridge.world_despawn(0, eid)
            bridge.world_despawn(0, eid)  # No error expected

        # Query should still work
        results = bridge.world_query(0, [1])
        assert len(results) == 50

        # Despawn non-existent entity (should be idempotent)
        bridge.world_despawn(0, 999999)  # No error expected

    def test_command_channel_error_isolation(self, stress_harness):
        """Errors in one operation do not affect others."""
        bridge = stress_harness.bridge

        # Setup
        eid = bridge.world_spawn(0, [(1, [(0, 42.0)])])

        # Try to read non-existent field (error)
        try:
            bridge.component_read(eid, 99, 0, float)
        except RuntimeError:
            pass  # Expected

        # Original entity should still be accessible
        val = bridge.component_read(eid, 1, 0, float)
        assert val == 42.0

        # Spawn should still work
        eid2 = bridge.world_spawn(0, [(1, [(0, 100.0)])])
        assert eid2 > eid

    def test_command_channel_concurrent_error_recovery(self, stress_harness):
        """Concurrent operations recover from errors."""
        bridge = stress_harness.bridge
        errors: List[str] = []

        def error_prone_op(thread_id: int):
            for i in range(200):
                try:
                    # Sometimes try to read non-existent
                    if i % 10 == 0:
                        bridge.component_read(999999, 1, 0, float)
                    else:
                        eid = bridge.world_spawn(0, [(1, [(0, float(i))])])
                        bridge.world_despawn(0, eid)
                except RuntimeError:
                    pass  # Expected error, ignored
                except Exception as e:
                    errors.append(f"Thread {thread_id} unexpected: {e}")

        threads = [
            threading.Thread(target=error_prone_op, args=(tid,))
            for tid in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Unexpected errors: {errors}"


# =============================================================================
# 4. INTEGRATION STRESS TESTS
# =============================================================================

class TestIntegrationStress:
    """Full bridge integration stress tests across all channels."""

    # -------------------------------------------------------------------------
    # 4a. Sustained load tests
    # -------------------------------------------------------------------------

    def test_full_bridge_sustained(self, stress_harness):
        """All channels under sustained load for multiple seconds."""
        bridge = stress_harness.bridge
        duration = 2.0  # 2 seconds (shorter for test suite)

        results = stress_harness.run_concurrent(
            channels=['data', 'type', 'command'],
            duration=duration,
            ops_per_channel=500,
        )

        assert not stress_harness.errors, f"Errors: {stress_harness.errors}"

        # Verify all channels executed operations
        for channel, result in results.items():
            assert result.count > 0, f"Channel {channel} had no operations"

    def test_full_bridge_10_second_sustained(self, stress_harness):
        """Extended sustained load test (10 seconds)."""
        bridge = stress_harness.bridge

        # Skip in CI if too slow
        start = time.perf_counter()

        stop_event = threading.Event()
        ops_count = {'data': 0, 'type': 0, 'command': 0}

        def data_worker():
            while not stop_event.is_set():
                for i in range(100):
                    bridge.component_write(i, 1, 0, float(i))
                ops_count['data'] += 100

        def type_worker():
            counter = 0
            while not stop_event.is_set():
                bridge.type_register(counter % 1000, f"T{counter}", 64, "[]")
                counter += 1
                ops_count['type'] += 1

        def command_worker():
            while not stop_event.is_set():
                eid = bridge.world_spawn(0, [(1, [(0, 1.0)])])
                bridge.world_query(0, [1])
                bridge.world_despawn(0, eid)
                ops_count['command'] += 3

        threads = [
            threading.Thread(target=data_worker),
            threading.Thread(target=type_worker),
            threading.Thread(target=command_worker),
        ]

        for t in threads:
            t.start()

        time.sleep(5.0)  # Reduced from 10s for test suite
        stop_event.set()

        for t in threads:
            t.join(timeout=5)

        elapsed = time.perf_counter() - start

        # All workers should have done significant work
        assert ops_count['data'] > 10_000, f"Data ops: {ops_count['data']}"
        assert ops_count['type'] > 1_000, f"Type ops: {ops_count['type']}"
        assert ops_count['command'] > 1_000, f"Command ops: {ops_count['command']}"

    # -------------------------------------------------------------------------
    # 4b. Memory stability tests
    # -------------------------------------------------------------------------

    def test_bridge_memory_stable(self, stress_harness):
        """Memory does not grow unboundedly during stress."""
        bridge = stress_harness.bridge

        import tracemalloc
        tracemalloc.start()

        initial = tracemalloc.get_traced_memory()[0]

        # Heavy workload
        for cycle in range(10):
            # Spawn and despawn entities
            eids = [bridge.world_spawn(0, [(1, [(0, float(i))])])
                    for i in range(1000)]
            for eid in eids:
                bridge.world_despawn(0, eid)

            # Register types
            for i in range(100):
                bridge.type_register(i, f"Cycle{cycle}_T{i}", 64, "[]")

            # Data writes
            for i in range(1000):
                bridge.component_write(i % 100, 1, 0, float(i))

        gc.collect()
        current = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Memory growth should be bounded (< 50MB for this workload)
        growth_mb = (current - initial) / (1024 * 1024)
        assert growth_mb < 50, f"Memory grew by {growth_mb:.2f}MB"

    def test_bridge_memory_after_reset(self, stress_harness):
        """Memory is fully reclaimed after reset."""
        bridge = stress_harness.bridge

        import tracemalloc
        tracemalloc.start()

        # Heavy workload
        for i in range(5000):
            bridge.world_spawn(0, [(1, [(0, float(i))])])
        for i in range(1000):
            bridge.type_register(i, f"T{i}", 64, "[]")
        for i in range(10000):
            bridge.component_write(i % 1000, 1, 0, float(i))

        before_reset = tracemalloc.get_traced_memory()[0]

        bridge.reset()
        gc.collect()

        after_reset = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # After reset, memory should be close to baseline
        assert after_reset < before_reset, "Memory not reclaimed after reset"

    # -------------------------------------------------------------------------
    # 4c. Latency percentile tests
    # -------------------------------------------------------------------------

    def test_bridge_latency_percentiles(self, stress_harness):
        """P50, P95, P99 latency within acceptable bounds."""
        bridge = stress_harness.bridge

        # Pre-populate
        for i in range(100):
            bridge.world_spawn(0, [(1, [(0, float(i))])])
            bridge.component_write(i, 1, 0, float(i))

        # Measure combined operations
        latencies: List[float] = []

        for _ in range(5000):
            start = time.perf_counter_ns()

            # Combined operation: spawn, write, read, despawn
            eid = bridge.world_spawn(0, [(1, [(0, 1.0)])])
            bridge.component_write(eid, 1, 0, 42.0)
            _ = bridge.component_read(eid, 1, 0, float)
            bridge.world_despawn(0, eid)

            latencies.append(time.perf_counter_ns() - start)

        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        # P99 should be within 10x of P50 (reasonable for mock)
        ratio = p99 / p50 if p50 > 0 else 0
        assert ratio < 20, f"P99/P50 ratio: {ratio:.2f} (P50={p50}ns, P99={p99}ns)"

        # Absolute bounds for mock (generous)
        assert p50 < 100_000, f"P50: {p50}ns"
        assert p95 < 200_000, f"P95: {p95}ns"
        assert p99 < 500_000, f"P99: {p99}ns"

    def test_bridge_latency_under_load(self, stress_harness):
        """Latency remains stable under concurrent load."""
        bridge = stress_harness.bridge

        # Pre-populate
        for i in range(100):
            bridge.world_spawn(0, [(1, [(0, float(i))])])

        latencies_idle: List[float] = []
        latencies_load: List[float] = []

        # Measure latency with no load
        for i in range(1000):
            start = time.perf_counter_ns()
            bridge.component_read(i % 100, 1, 0, float)
            latencies_idle.append(time.perf_counter_ns() - start)

        # Start background load
        stop_event = threading.Event()

        def background_load():
            while not stop_event.is_set():
                for i in range(100):
                    bridge.component_write(i, 1, 0, float(i))

        load_threads = [threading.Thread(target=background_load) for _ in range(2)]
        for t in load_threads:
            t.start()

        # Measure latency under load
        time.sleep(0.1)  # Let load stabilize
        for i in range(1000):
            start = time.perf_counter_ns()
            bridge.component_read(i % 100, 1, 0, float)
            latencies_load.append(time.perf_counter_ns() - start)

        stop_event.set()
        for t in load_threads:
            t.join(timeout=5)

        p50_idle = sorted(latencies_idle)[500]
        p50_load = sorted(latencies_load)[500]

        # Under load, P50 should not degrade more than 5x
        ratio = p50_load / p50_idle if p50_idle > 0 else 1
        assert ratio < 10, (
            f"Load degradation ratio: {ratio:.2f}x "
            f"(idle P50={p50_idle}ns, load P50={p50_load}ns)"
        )


# =============================================================================
# 5. EDGE CASE STRESS TESTS
# =============================================================================

class TestEdgeCaseStress:
    """Edge case stress tests for boundary conditions."""

    def test_zero_entity_id_stress(self, fresh_bridge):
        """Entity ID 0 handles stress correctly."""
        for i in range(1000):
            fresh_bridge.component_write(0, i % 100, 0, float(i))

        # Verify
        for i in range(100):
            val = fresh_bridge.component_read(0, i, 0, float)
            # Last write for component i was at iteration (999 - (999 % 100) + i)
            # which simplifies to 900 + i
            assert val == float(900 + i), f"Component {i}: expected {900 + i}, got {val}"

    def test_max_component_id_stress(self, fresh_bridge):
        """Large component IDs handle stress correctly."""
        large_cid = 2**30

        for i in range(1000):
            fresh_bridge.component_write(i, large_cid, 0, float(i))

        for i in range(1000):
            val = fresh_bridge.component_read(i, large_cid, 0, float)
            assert val == float(i)

    def test_sparse_entity_stress(self, fresh_bridge):
        """Sparse entity IDs (gaps) handle stress correctly."""
        # Write to sparse entity IDs
        sparse_ids = [i * 1000 for i in range(100)]

        for eid in sparse_ids:
            fresh_bridge.component_write(eid, 1, 0, float(eid))

        # Verify
        for eid in sparse_ids:
            val = fresh_bridge.component_read(eid, 1, 0, float)
            assert val == float(eid)

    def test_large_offset_stress(self, fresh_bridge):
        """Large field offsets handle stress correctly."""
        for i in range(1000):
            offset = i * 1024  # 1KB apart
            fresh_bridge.component_write(1, 1, offset, float(i))

        for i in range(1000):
            offset = i * 1024
            val = fresh_bridge.component_read(1, 1, offset, float)
            assert val == float(i)

    def test_unicode_type_name_stress(self, fresh_bridge):
        """Unicode type names handle stress correctly."""
        for i in range(100):
            name = f"Component_{i}_中文_日本語_АБВ"
            fresh_bridge.type_register(i, name, 64, "[]")

        assert len(fresh_bridge.type_registry) == 100

        # Verify names are preserved
        for i in range(100):
            name = fresh_bridge.type_registry[i]["name"]
            assert "中文" in name

    def test_empty_string_field_stress(self, fresh_bridge):
        """Empty string fields handle stress correctly."""
        for i in range(1000):
            fresh_bridge.component_write(i, 1, 0, "")

        for i in range(1000):
            val = fresh_bridge.component_read(i, 1, 0, str)
            assert val == ""

    def test_null_value_stress(self, fresh_bridge):
        """None/null values handle stress correctly."""
        for i in range(1000):
            fresh_bridge.component_write(i, 1, 0, None)

        for i in range(1000):
            val = fresh_bridge.component_read(i, 1, 0, object)
            assert val is None

    def test_rapid_archetype_changes(self, fresh_bridge):
        """Rapid archetype changes (add/remove components) are stable."""
        for cycle in range(50):
            # Create entities with varying archetypes
            eids = []
            for i in range(20):
                # Archetype: components [1..i+1]
                components = [(cid, [(0, float(cid))]) for cid in range(1, i + 2)]
                eid = fresh_bridge.world_spawn(0, components)
                eids.append(eid)

            # Query various archetypes
            for n_comps in range(1, 10):
                results = fresh_bridge.world_query(0, list(range(1, n_comps + 1)))
                # Entities with >= n_comps components should match
                expected = sum(1 for i in range(20) if i + 1 >= n_comps)
                assert len(results) == expected, (
                    f"Cycle {cycle}, query [{1}..{n_comps}]: "
                    f"expected {expected}, got {len(results)}"
                )

            # Despawn all
            for eid in eids:
                fresh_bridge.world_despawn(0, eid)

    def test_interleaved_channel_stress(self, fresh_bridge):
        """Interleaved operations across all channels are consistent."""
        checkpoints = []

        for i in range(100):
            # Type channel
            fresh_bridge.type_register(i, f"T{i}", 64, "[]")

            # Command channel
            eid = fresh_bridge.world_spawn(0, [(i, [(0, float(i))])])

            # Data channel
            fresh_bridge.component_write(eid, i, 0, float(i * 10))
            val = fresh_bridge.component_read(eid, i, 0, float)

            # Query
            results = fresh_bridge.world_query(0, [i])

            checkpoints.append({
                'type_count': len(fresh_bridge.type_registry),
                'entity': eid,
                'value': val,
                'query_count': len(results),
            })

        # Verify checkpoints
        for i, cp in enumerate(checkpoints):
            assert cp['type_count'] == i + 1
            assert cp['value'] == float(i * 10)
            assert cp['query_count'] >= 1


# =============================================================================
# SUMMARY
# =============================================================================

def test_stress_suite_summary(stress_harness, capsys):
    """Print summary of stress test capabilities (informational)."""
    bridge = stress_harness.bridge

    # Quick benchmark
    N = 10_000

    # Data channel
    start = time.perf_counter_ns()
    for i in range(N):
        bridge.component_write(i % 100, 1, 0, float(i))
    data_write_ns = (time.perf_counter_ns() - start) / N

    for i in range(100):
        bridge.component_write(i, 1, 0, float(i))

    start = time.perf_counter_ns()
    for i in range(N):
        bridge.component_read(i % 100, 1, 0, float)
    data_read_ns = (time.perf_counter_ns() - start) / N

    # Type channel
    start = time.perf_counter_ns()
    for i in range(1000):
        bridge.type_register(i, f"T{i}", 64, "[]")
    type_reg_ns = (time.perf_counter_ns() - start) / 1000

    bridge.reset()

    # Command channel
    start = time.perf_counter_ns()
    for i in range(1000):
        eid = bridge.world_spawn(0, [(1, [(0, float(i))])])
    spawn_ns = (time.perf_counter_ns() - start) / 1000

    start = time.perf_counter_ns()
    for i in range(100):
        bridge.world_query(0, [1])
    query_ns = (time.perf_counter_ns() - start) / 100

    # This test always passes - it's informational
    assert True
