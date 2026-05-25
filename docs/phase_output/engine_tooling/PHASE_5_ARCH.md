# PHASE 5 ARCHITECTURE: Development Support Tools

## Phase Overview

Phase 5 implements the development support infrastructure: profiling, replay system, debug tools, testing framework, and VCS integration.

## Components

### 1. Profiling System (engine/tooling/profiling/)

**Purpose**: Comprehensive performance analysis across CPU, GPU, memory, and network

**Architecture**:
```
ProfilerSubsystem
    |
    +-- CPUProfiler
    |       +-- CPUProfileSample (name, timing, thread)
    |       +-- CallTreeNode (parent/children, stats)
    |       +-- FlameGraphData (for visualization)
    |       +-- HotPath (critical path analysis)
    |       +-- @profile decorator
    |
    +-- GPUProfiler
    |       +-- GPUProfileSample (pass, timing)
    |       +-- GPUTimestampQuery (placeholder for backend)
    |       +-- DrawCallStats (triangles, vertices, batches)
    |       +-- ShaderStats (invocations, time)
    |       +-- GPUMemoryStats (VRAM, bandwidth)
    |       +-- RenderPassTiming
    |       +-- @gpu_profile decorator
    |
    +-- MemoryProfiler
    |       +-- AllocationRecord (size, category, stack trace)
    |       +-- MemorySnapshot (point-in-time state)
    |       +-- SnapshotDiff (compare snapshots)
    |       +-- LeakReport (long-lived allocations)
    |       +-- FragmentationStats
    |       +-- Category budgets
    |
    +-- NetworkProfiler
    |       +-- PacketRecord (direction, size, type, channel)
    |       +-- BandwidthSample (KB/s)
    |       +-- LatencyGraph (RTT, jitter)
    |       +-- Per-channel/actor statistics
    |
    +-- FrameProfiler
    |       +-- FrameData (phases, timing)
    |       +-- FrameTimeline (per-frame breakdown)
    |       +-- SpikeDetector (adaptive threshold)
    |       +-- FrameBudget (per-phase limits)
    |       +-- Phase categories (Input, Physics, AI, Render)
    |
    +-- ProfilerOverlay
    |       +-- Configurable panels
    |       +-- Real-time stats display
    |       +-- Render callback hook
    |
    +-- ProfilerExporter
    |       +-- ChromeTraceExporter (chrome://tracing)
    |       +-- CSVExporter (spreadsheet)
    |       +-- JSONExporter (custom tools)
    |
    +-- ProfilerComparator
            +-- SessionDiff (baseline vs current)
            +-- MetricComparison (delta, percentage)
            +-- RegressionDetector (severity levels)
```

**Frame Phases**:
| Phase | Description |
|-------|-------------|
| Input | Input polling and processing |
| Physics | Physics simulation step |
| AI | AI decision making |
| Animation | Animation evaluation |
| Rendering | Draw call submission |
| Audio | Audio mixing |
| Network | Network synchronization |
| Custom | User-defined phases |

### 2. Replay System (engine/tooling/replay/)

**Purpose**: Game session recording, playback, and analysis

**Architecture**:
```
ReplaySystem
    |
    +-- InputRecorder
    |       +-- RecordedInput (type, data, timestamp)
    |       +-- InputType (keyboard, mouse, gamepad, touch)
    |       +-- High-precision timing (perf_counter)
    |       +-- Mouse move deduplication
    |       +-- SHA-256 input hashing
    |
    +-- StateRecorder
    |       +-- StateSnapshot (keyframe, full state)
    |       +-- StateDelta (incremental changes)
    |       +-- Compression (ZLIB variants)
    |       +-- SHA-256 checksums
    |       +-- Recursive diff computation
    |
    +-- ReplayPlayback
    |       +-- Speed control (0.1x to 10x)
    |       +-- SeekMode (frame, time, percentage, keyframe, marker)
    |       +-- Frame stepping (forward/backward)
    |       +-- Loop range configuration
    |       +-- State restoration on seek
    |
    +-- ReplayFile
    |       +-- Binary format (magic "RPLY")
    |       +-- Section-based layout
    |       +-- Optional per-section compression
    |       +-- Fast metadata-only loading
    |
    +-- GhostSystem
    |       +-- GhostRenderMode (solid, transparent, outline, trail)
    |       +-- SLERP interpolation for rotation
    |       +-- Time offset for comparison
    |       +-- GhostComparison (time diff, distance, lead changes)
    |
    +-- DeterminismChecker
    |       +-- Configurable tolerances
    |       +-- Path-specific overrides
    |       +-- DriftSeverity (MINOR to CRITICAL)
    |       +-- Snapshot chain verification
    |
    +-- ReplayTimeline
    |       +-- MarkerType (bookmark, event, keyframe, highlight)
    |       +-- Event (with duration)
    |       +-- Segment (grouping)
    |       +-- Track (multi-layer)
    |
    +-- ReplayBrowser
    |       +-- Multi-criteria filtering
    |       +-- Sort orders (date, duration, score)
    |       +-- Pagination
    |       +-- Collection statistics
    |
    +-- ReplayExport
            +-- Video formats (MP4, WEBM, AVI, GIF)
            +-- Image sequences (PNG, JPEG)
            +-- Codec selection
            +-- Progress tracking
```

### 3. Debug Tools (engine/tooling/debug/)

**Purpose**: Runtime visualization and inspection

**Architecture**:
```
DebugTools
    |
    +-- DebugDraw
    |       +-- DrawPrimitive (15 types)
    |       +-- DrawCommand (expiration, category)
    |       +-- DebugDrawBatch (efficient submission)
    |       +-- Immediate and persistent modes
    |       +-- Category filtering
    |       +-- @debug_draw decorator
    |
    +-- DebugMenu
    |       +-- MenuItem types (Toggle, Slider, Action, Dropdown)
    |       +-- SubMenu hierarchy
    |       +-- Default menus (Rendering, Physics, AI, Perf)
    |       +-- Keyboard shortcuts
    |
    +-- GameplayDebugger
    |       +-- AIVisualization (path, perception, state)
    |       +-- NavMeshDisplay (polygons, costs)
    |       +-- TriggerVolumeVisualizer
    |
    +-- PhysicsDebugger
    |       +-- CollisionShapeVisualizer
    |       +-- ContactPointDisplay
    |       +-- RaycastVisualizer
    |       +-- Collision layer naming
    |
    +-- RenderDebugger
    |       +-- WireframeMode (Off, Overlay, Only, XRay)
    |       +-- BoundingBoxDisplay (AABB, OBB, Sphere)
    |       +-- LODVisualization (level coloring)
    |       +-- OverdrawHeatmap
    |
    +-- DebugCamera
    |       +-- CameraMode (Game, FreeFly, Orbit, Fixed, Path)
    |       +-- FreeFlyCamera (WASD, mouse look)
    |       +-- OrbitCamera (target tracking)
    |       +-- Smooth transitions
    |
    +-- WatchWindow
    |       +-- WatchVariable (history, format)
    |       +-- Breakpoint (condition, hit count)
    |       +-- ConditionalWatch (change triggers)
    |       +-- VariableTracker (weakref-based)
    |
    +-- DebugConsole
            +-- CommandCategory
            +-- ConsoleCommand (args, permissions)
            +-- Built-in commands (help, clear, history)
            +-- @cheat decorator
```

### 4. Testing Framework (engine/tooling/testing/)

**Purpose**: Comprehensive test infrastructure for game systems

**Architecture**:
```
TestFramework
    |
    +-- TestFramework
    |       +-- @test decorator (tags, priority, timeout)
    |       +-- @bench decorator (iterations, warmup)
    |       +-- @skip / @skip_if decorators
    |       +-- @expected_failure decorator
    |       +-- @parametrize decorator
    |       +-- TestCase base class
    |       +-- BenchmarkSuite
    |
    +-- TestRunner
    |       +-- Test discovery (directory scanning)
    |       +-- TestFilter (patterns, tags, regex)
    |       +-- Sequential execution
    |       +-- Hook system (before/after test/suite)
    |
    +-- ParallelTestRunner
    |       +-- ThreadPoolExecutor
    |       +-- ProcessPoolExecutor
    |       +-- Fail-fast option
    |
    +-- TestMocking
    |       +-- Mock class (call tracking, return values)
    |       +-- MockEntity (components, tags)
    |       +-- MockComponent (dirty tracking)
    |       +-- MockSystem (update counting)
    |       +-- MockWorld (full ECS mock)
    |       +-- patch(), spy(), stub()
    |
    +-- TestAssertions
    |       +-- Vector assertions (equal, near)
    |       +-- Transform assertions (quaternion equivalence)
    |       +-- ECS assertions (has_component, entity_count)
    |       +-- Performance assertions (memory leaks, frame time)
    |       +-- GameAssertions mixin
    |
    +-- TestReporting
    |       +-- TestReport (results, timing, environment)
    |       +-- JUnitReporter (XML for CI)
    |       +-- HTMLReporter (styled, interactive)
    |       +-- ConsoleReporter (ANSI colors)
    |       +-- JSONReporter
    |
    +-- TestFixtures
            +-- Fixture (factory, scope, dependencies)
            +-- FixtureManager (resolution, caching)
            +-- @fixture decorator
            +-- @setup / @teardown decorators
            +-- GameWorldFixture
            +-- EntityFixture
            +-- ResourceFixture (temp files)
```

### 5. VCS Integration (engine/tooling/vcs/)

**Purpose**: Version control abstraction for Git and Perforce

**Architecture**:
```
VCSIntegration
    |
    +-- VCSProvider (ABC)
    |       +-- connect(), disconnect()
    |       +-- status(), commit(), revert()
    |       +-- branch operations
    |       +-- merge operations
    |       +-- blame, log, diff
    |
    +-- GitProvider
    |       +-- subprocess.run integration
    |       +-- Porcelain format parsing
    |       +-- Status code mapping
    |       +-- Branch/tag management
    |       +-- Remote operations
    |       +-- Stash operations
    |
    +-- PerforceProvider
    |       +-- p4 command integration
    |       +-- Changelist management
    |       +-- Stream-based branching
    |       +-- Label operations
    |       +-- Client spec parsing
    |
    +-- LockManager
    |       +-- LockType (EXCLUSIVE, SHARED, INTENT)
    |       +-- LockState (LOCKED, UNLOCKED, PENDING)
    |       +-- Binary file detection (65+ extensions)
    |       +-- Git LFS integration
    |       +-- Lock persistence (JSON)
    |
    +-- MergeResolver
    |       +-- ThreeWayMerge (difflib-based)
    |       +-- MergeStrategy (ours, theirs, union, auto)
    |       +-- ConflictRegion (ours/theirs/base content)
    |       +-- Conflict marker parsing
    |
    +-- FileOperations
    |       +-- FileStatusInfo
    |       +-- DiffViewer
    |       +-- Unified diff parsing
    |
    +-- VCSProviderRegistry
            +-- Auto-detection (.git, .svn, P4CLIENT)
            +-- Factory function
```

## Data Flow

### Profiling Data Flow
```
@profile decorator
    -> CPUProfiler.scope(name)
    -> Record start time
    -> Execute function
    -> Record end time
    -> Create CPUProfileSample
    -> Add to hierarchical tree
    -> Export to ChromeTrace
```

### Replay Data Flow
```
Recording:
    Input event -> InputRecorder.record()
    State change -> StateRecorder.record_delta()
    Periodic -> StateRecorder.take_snapshot()
    Save -> ReplayFile.save()

Playback:
    ReplayFile.load() -> Load metadata
    Seek to position -> Restore nearest snapshot
    Play forward -> Apply deltas + inject inputs
```

### Test Execution Flow
```
TestRunner.discover()
    -> Find test files
    -> Import test modules
    -> Collect @test decorated functions
    -> Apply filters

TestRunner.run()
    -> For each test:
        -> Setup fixtures
        -> Run setup
        -> Execute test
        -> Handle timeout
        -> Catch exceptions
        -> Run teardown
        -> Record result
    -> Generate report
```

## Integration Points

### Renderer Integration
- DebugDraw needs render callback
- OverdrawHeatmap needs pixel data
- LODVisualization needs LOD info

### Engine Integration
- Profiler markers around engine systems
- Replay hooks for input/state
- Test fixtures need engine initialization

### Platform Integration
- VCS subprocess calls
- File system monitoring for hot reload
- GPU timestamp queries

## Thread Safety

| Component | Strategy |
|-----------|----------|
| CPUProfiler | Thread-local samples |
| MemoryProfiler | Global lock on record |
| ReplayRecorder | Lock-free queue |
| TestRunner | Sequential by default |
| ParallelTestRunner | Per-worker isolation |
| LockManager | File-based locking |

## Configuration

### Profiler Configuration
```python
ProfilerConfig(
    cpu_enabled=True,
    gpu_enabled=True,
    memory_enabled=True,
    network_enabled=True,
    frame_budget_ms=16.67,
    spike_threshold=2.0,  # 2x budget
)
```

### Replay Configuration
```python
ReplayConfig(
    keyframe_interval=60,  # frames
    compression=CompressionMethod.ZLIB,
    max_state_size_mb=10,
    max_inputs=100000,
)
```

### Test Configuration
```python
TestConfig(
    parallel=False,
    workers=4,
    timeout_seconds=60,
    fail_fast=True,
    capture_output=True,
)
```

## Testing Strategy

### Unit Tests
- Profile sample aggregation
- Replay serialization
- Mock behavior verification
- Assertion correctness

### Integration Tests
- Full profiling session
- Replay record/playback cycle
- Test discovery and execution
- VCS operations (with mock repo)

### Performance Tests
- Profiler overhead (<1%)
- Replay seek latency
- Test runner scaling
