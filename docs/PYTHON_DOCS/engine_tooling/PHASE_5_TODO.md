# PHASE 5 TODO: Development Support Tools

## Overview

Phase 5 implements development support infrastructure including profiling, replay, debugging, testing, and version control integration.

---

## 1. Profiling System

### 1.1 CPU Profiler
- [ ] **T1.1.1**: Test hierarchical timing
  - Acceptance: Parent-child relationships correct
  - Acceptance: Timing sums match
  - File: `engine/tooling/profiling/cpu_profiler.py`

- [ ] **T1.1.2**: Test flame graph generation
  - Acceptance: Data suitable for flame graph visualization
  - Acceptance: Stack depths correct

- [ ] **T1.1.3**: Test hot path detection
  - Acceptance: Critical path identified
  - Acceptance: Percentage calculations correct

### 1.2 GPU Profiler
- [ ] **T1.2.1**: Implement GPU timestamp query integration
  - Acceptance: Queries submitted to GPU
  - Acceptance: Results retrieved correctly
  - Integration: Requires GPU backend
  - File: `engine/tooling/profiling/gpu_profiler.py`

- [ ] **T1.2.2**: Test render pass timing
  - Acceptance: Per-pass timing recorded
  - Acceptance: Category aggregation works

- [ ] **T1.2.3**: Implement VRAM tracking
  - Acceptance: Current VRAM usage tracked
  - Acceptance: Per-category breakdown
  - Integration: Requires GPU backend

### 1.3 Memory Profiler
- [ ] **T1.3.1**: Test allocation tracking
  - Acceptance: Allocations recorded with stack trace
  - Acceptance: Frees matched to allocations
  - File: `engine/tooling/profiling/memory_profiler.py`

- [ ] **T1.3.2**: Test snapshot diffing
  - Acceptance: New allocations identified
  - Acceptance: Freed allocations identified

- [ ] **T1.3.3**: Test leak detection
  - Acceptance: Long-lived allocations flagged
  - Acceptance: Threshold configurable

- [ ] **T1.3.4**: Test GC integration
  - Acceptance: gc.collect() triggers tracked
  - Acceptance: Generation info recorded

### 1.4 Network Profiler
- [ ] **T1.4.1**: Test bandwidth sampling
  - Acceptance: KB/s calculated correctly
  - Acceptance: Direction separated (send/receive)
  - File: `engine/tooling/profiling/network_profiler.py`

- [ ] **T1.4.2**: Test RTT tracking
  - Acceptance: RTT recorded per sample
  - Acceptance: Jitter calculated

- [ ] **T1.4.3**: Test per-actor breakdown
  - Acceptance: Bandwidth per actor ID
  - Acceptance: Top consumers identified

### 1.5 Frame Profiler
- [ ] **T1.5.1**: Test phase tracking
  - Acceptance: All phases recorded
  - Acceptance: Custom phases work
  - File: `engine/tooling/profiling/frame_profiler.py`

- [ ] **T1.5.2**: Test spike detection
  - Acceptance: Frames exceeding threshold flagged
  - Acceptance: Adaptive threshold works

- [ ] **T1.5.3**: Test budget warnings
  - Acceptance: Over-budget phases flagged
  - Acceptance: Budget configurable per phase

### 1.6 Export
- [ ] **T1.6.1**: Test Chrome Trace export
  - Acceptance: Valid JSON for chrome://tracing
  - Acceptance: All events included
  - File: `engine/tooling/profiling/profiler_export.py`

- [ ] **T1.6.2**: Test CSV export
  - Acceptance: Valid CSV format
  - Acceptance: All columns present

### 1.7 Comparison
- [ ] **T1.7.1**: Test session diff
  - Acceptance: Metrics compared correctly
  - Acceptance: Delta percentages calculated
  - File: `engine/tooling/profiling/profiler_compare.py`

- [ ] **T1.7.2**: Test regression detection
  - Acceptance: Severity levels assigned
  - Acceptance: Thresholds configurable

---

## 2. Replay System

### 2.1 Input Recording
- [ ] **T2.1.1**: Test input capture
  - Acceptance: All input types recorded
  - Acceptance: Timestamps accurate
  - File: `engine/tooling/replay/input_recorder.py`

- [ ] **T2.1.2**: Test mouse deduplication
  - Acceptance: Redundant moves filtered
  - Acceptance: Significant moves preserved

- [ ] **T2.1.3**: Test input hashing
  - Acceptance: SHA-256 hash computed
  - Acceptance: Same inputs = same hash

### 2.2 State Recording
- [ ] **T2.2.1**: Test snapshot capture
  - Acceptance: Full state serialized
  - Acceptance: Keyframe interval respected
  - File: `engine/tooling/replay/state_recorder.py`

- [ ] **T2.2.2**: Test delta compression
  - Acceptance: Only changes stored
  - Acceptance: Apply/reverse works

- [ ] **T2.2.3**: Test compression
  - Acceptance: ZLIB compression works
  - Acceptance: Size reduction significant

### 2.3 Playback
- [ ] **T2.3.1**: Test speed control
  - Acceptance: 0.1x to 10x speeds work
  - Acceptance: Timing remains accurate
  - File: `engine/tooling/replay/replay_playback.py`

- [ ] **T2.3.2**: Test seeking
  - Acceptance: Seek to frame works
  - Acceptance: Seek to time works
  - Acceptance: Seek to marker works

- [ ] **T2.3.3**: Test frame stepping
  - Acceptance: Step forward works
  - Acceptance: Step backward works

### 2.4 File Format
- [ ] **T2.4.1**: Test save/load cycle
  - Acceptance: All data preserved
  - Acceptance: Checksum verified
  - File: `engine/tooling/replay/replay_file.py`

- [ ] **T2.4.2**: Test metadata-only load
  - Acceptance: Fast metadata access
  - Acceptance: Full data loadable later

### 2.5 Ghost System
- [ ] **T2.5.1**: Test ghost playback
  - Acceptance: Position interpolation works
  - Acceptance: SLERP rotation correct
  - File: `engine/tooling/replay/ghost_system.py`

- [ ] **T2.5.2**: Test comparison metrics
  - Acceptance: Time difference tracked
  - Acceptance: Lead changes counted

### 2.6 Determinism
- [ ] **T2.6.1**: Test state verification
  - Acceptance: Identical inputs = identical state
  - Acceptance: Drifts detected and reported
  - File: `engine/tooling/replay/determinism_checker.py`

- [ ] **T2.6.2**: Test tolerance handling
  - Acceptance: Float tolerance works
  - Acceptance: Path-specific tolerance works

### 2.7 Export
- [ ] **T2.7.1**: Integrate video encoding
  - Acceptance: MP4 export works
  - Acceptance: GIF export works
  - Integration: Requires FFmpeg or similar
  - File: `engine/tooling/replay/replay_export.py`

---

## 3. Debug Tools

### 3.1 Debug Draw
- [ ] **T3.1.1**: Test all primitives
  - Acceptance: Line, sphere, box, arrow work
  - Acceptance: Text rendering works
  - File: `engine/tooling/debug/debug_draw.py`

- [ ] **T3.1.2**: Test lifetime expiration
  - Acceptance: Timed commands expire
  - Acceptance: Persistent commands stay

- [ ] **T3.1.3**: Test category filtering
  - Acceptance: Disabled categories hidden
  - Acceptance: Toggle works

### 3.2 Debug Menu
- [ ] **T3.2.1**: Test menu navigation
  - Acceptance: Submenus open/close
  - Acceptance: Items selectable
  - File: `engine/tooling/debug/debug_menu.py`

- [ ] **T3.2.2**: Test item types
  - Acceptance: Toggle changes value
  - Acceptance: Slider changes value
  - Acceptance: Action triggers callback

### 3.3 Gameplay Debugger
- [ ] **T3.3.1**: Test AI visualization
  - Acceptance: AI state shown
  - Acceptance: Path drawn
  - Acceptance: Perception radius shown
  - File: `engine/tooling/debug/gameplay_debug.py`

- [ ] **T3.3.2**: Test NavMesh display
  - Acceptance: Polygons rendered
  - Acceptance: Costs displayed

### 3.4 Physics Debugger
- [ ] **T3.4.1**: Test collision shape display
  - Acceptance: All shape types rendered
  - Acceptance: Body type colors correct
  - File: `engine/tooling/debug/physics_debug.py`

- [ ] **T3.4.2**: Test contact point display
  - Acceptance: Contact points shown
  - Acceptance: Normal arrows visible

### 3.5 Debug Camera
- [ ] **T3.5.1**: Test free-fly camera
  - Acceptance: WASD movement works
  - Acceptance: Mouse look works
  - Acceptance: Sprint/slow modifiers work
  - File: `engine/tooling/debug/debug_camera.py`

- [ ] **T3.5.2**: Test orbit camera
  - Acceptance: Orbit around target
  - Acceptance: Zoom works

- [ ] **T3.5.3**: Test camera transitions
  - Acceptance: Smooth transition between cameras
  - Acceptance: Easing correct

### 3.6 Watch Window
- [ ] **T3.6.1**: Test variable watching
  - Acceptance: Values updated
  - Acceptance: History tracked
  - File: `engine/tooling/debug/watch_variables.py`

- [ ] **T3.6.2**: Test breakpoints
  - Acceptance: Condition evaluated
  - Acceptance: Hit count tracked

---

## 4. Testing Framework

### 4.1 Test Framework
- [ ] **T4.1.1**: Test @test decorator
  - Acceptance: Tests discovered
  - Acceptance: Tags applied
  - Acceptance: Priority ordering works
  - File: `engine/tooling/testing/test_framework.py`

- [ ] **T4.1.2**: Test @bench decorator
  - Acceptance: Benchmarks run
  - Acceptance: Statistics calculated
  - Acceptance: Memory tracked

- [ ] **T4.1.3**: Test @parametrize
  - Acceptance: Test runs for each parameter
  - Acceptance: Parameters passed correctly

### 4.2 Test Runner
- [ ] **T4.2.1**: Test discovery
  - Acceptance: Tests found in directory
  - Acceptance: Filters applied
  - File: `engine/tooling/testing/test_runner.py`

- [ ] **T4.2.2**: Test timeout handling
  - Acceptance: Long tests cancelled
  - Acceptance: Timeout configurable

- [ ] **T4.2.3**: Test parallel execution
  - Acceptance: Tests run concurrently
  - Acceptance: Results collected correctly

### 4.3 Mocking
- [ ] **T4.3.1**: Test Mock class
  - Acceptance: Call tracking works
  - Acceptance: Return value configurable
  - Acceptance: Side effects work
  - File: `engine/tooling/testing/test_mocking.py`

- [ ] **T4.3.2**: Test MockWorld
  - Acceptance: Entity CRUD works
  - Acceptance: Component management works
  - Acceptance: Event queue works

### 4.4 Assertions
- [ ] **T4.4.1**: Test vector assertions
  - Acceptance: assert_vector_equal works
  - Acceptance: assert_vector_near works (tolerance)
  - File: `engine/tooling/testing/test_assertions.py`

- [ ] **T4.4.2**: Test ECS assertions
  - Acceptance: assert_entity_has_component works
  - Acceptance: assert_event_fired works

- [ ] **T4.4.3**: Test performance assertions
  - Acceptance: assert_no_memory_leaks works
  - Acceptance: assert_frame_time works

### 4.5 Reporting
- [ ] **T4.5.1**: Test JUnit output
  - Acceptance: Valid XML for CI systems
  - Acceptance: All test info included
  - File: `engine/tooling/testing/test_reporting.py`

- [ ] **T4.5.2**: Test HTML output
  - Acceptance: Styled report generated
  - Acceptance: Interactive elements work

### 4.6 Fixtures
- [ ] **T4.6.1**: Test fixture resolution
  - Acceptance: Dependencies resolved
  - Acceptance: Caching works per scope
  - File: `engine/tooling/testing/test_fixtures.py`

- [ ] **T4.6.2**: Test generator fixtures
  - Acceptance: Setup before yield
  - Acceptance: Teardown after yield

---

## 5. VCS Integration

### 5.1 Git Provider
- [ ] **T5.1.1**: Test status parsing
  - Acceptance: Porcelain format parsed
  - Acceptance: All status codes mapped
  - File: `engine/tooling/vcs/git_provider.py`

- [ ] **T5.1.2**: Test commit operations
  - Acceptance: Commit with message works
  - Acceptance: Staged files committed

- [ ] **T5.1.3**: Test branch operations
  - Acceptance: Create/delete branches
  - Acceptance: Checkout works
  - Acceptance: Merge works

- [ ] **T5.1.4**: Test blame parsing
  - Acceptance: Author info extracted
  - Acceptance: Line ranges work

### 5.2 Perforce Provider
- [ ] **T5.2.1**: Test changelist operations
  - Acceptance: Create/submit changelists
  - Acceptance: Shelve/unshelve works
  - File: `engine/tooling/vcs/perforce_provider.py`

- [ ] **T5.2.2**: Test sync operations
  - Acceptance: Sync to head
  - Acceptance: Sync to revision

### 5.3 Lock Manager
- [ ] **T5.3.1**: Test file locking
  - Acceptance: Exclusive lock works
  - Acceptance: Shared lock works
  - File: `engine/tooling/vcs/lock_manager.py`

- [ ] **T5.3.2**: Test binary detection
  - Acceptance: All 65+ extensions detected
  - Acceptance: Custom extensions work

- [ ] **T5.3.3**: Test Git LFS integration
  - Acceptance: LFS lock/unlock works

### 5.4 Merge Tools
- [ ] **T5.4.1**: Test 3-way merge
  - Acceptance: Clean merges work
  - Acceptance: Conflicts detected
  - File: `engine/tooling/vcs/merge_tools.py`

- [ ] **T5.4.2**: Test conflict resolution
  - Acceptance: Ours strategy works
  - Acceptance: Theirs strategy works
  - Acceptance: Union strategy works

### 5.5 File Operations
- [ ] **T5.5.1**: Test diff parsing
  - Acceptance: Unified diff parsed
  - Acceptance: Hunks extracted
  - File: `engine/tooling/vcs/file_operations.py`

---

## Integration Tests

### I1. Full Profiling Session
- [ ] **I1.1**: Profile frame, export trace
  - Steps: Start profiling, run frames, export Chrome Trace
  - Acceptance: Valid trace file

### I2. Replay Cycle
- [ ] **I2.1**: Record, save, load, playback
  - Steps: Record session, save file, load file, playback
  - Acceptance: Playback matches recording

### I3. Test Suite
- [ ] **I3.1**: Discover and run tests
  - Steps: Discover tests, apply filters, run, report
  - Acceptance: All expected tests run

### I4. VCS Workflow
- [ ] **I4.1**: Status, commit, push
  - Steps: Check status, commit changes, push to remote
  - Acceptance: Changes visible in remote

---

## Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| Profiler overhead | < 1% | Benchmark with/without |
| Replay seek | < 100ms | Benchmark seek operations |
| Test discovery | < 1s (1000 tests) | Benchmark |
| Git status | < 500ms | Benchmark on large repo |

---

## Dependencies

### Required Before Phase 5
- Phase 1: Logging, console
- Engine initialization for testing

### External Tools
- Git CLI for Git operations
- P4 CLI for Perforce operations
- FFmpeg for video export

### Blocks Phase 6
- Localization may use VCS for string management
