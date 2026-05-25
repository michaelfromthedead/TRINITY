# PHASE 3 TODO: Prediction and Lag Compensation

## Overview

Phase 3 implements client-side prediction and server-side lag compensation. All tasks assume the existing implementation is production-ready; these TODOs focus on testing, verification, and identified gaps.

---

## 1. Client Prediction Tasks

### 1.1 Unit Tests: Input Buffer

**File**: `tests/blackbox_input_buffer.py`

**Acceptance Criteria**:
- [ ] `push()` adds input to buffer
- [ ] Buffer size capped at max_size
- [ ] `pop_confirmed()` removes inputs up to sequence
- [ ] `pop_confirmed()` returns confirmed inputs list
- [ ] `get_unconfirmed()` returns remaining inputs
- [ ] `get_input_at_sequence()` returns correct input
- [ ] `get_inputs_after_sequence()` returns subsequent inputs
- [ ] Sequence wraparound handled correctly

---

### 1.2 Unit Tests: Prediction State

**File**: `tests/blackbox_prediction_state.py`

**Acceptance Criteria**:
- [ ] `apply_input()` updates position based on movement
- [ ] Friction applied to velocity
- [ ] Jump only triggers when on ground
- [ ] Gravity accumulates each frame
- [ ] Ground clamp prevents falling through y=0
- [ ] `clone()` produces independent copy
- [ ] `distance_to()` calculates Euclidean distance
- [ ] Sequence increments after apply_input

---

### 1.3 Unit Tests: Client Predictor

**File**: `tests/blackbox_client_predictor.py`

**Acceptance Criteria**:
- [ ] `predict()` applies input and returns new state
- [ ] `store_input()` adds to buffer with predicted state
- [ ] `get_prediction_error()` compares to server state
- [ ] Prediction accuracy tracking updates on server state
- [ ] Misprediction counter increments on error > threshold

---

## 2. Server Reconciliation Tasks

### 2.1 Unit Tests: State Comparison

**File**: `tests/blackbox_state_comparison.py`

**Acceptance Criteria**:
- [ ] Identical states return MATCH
- [ ] Small position error returns MISMATCH_SMALL
- [ ] Large position error returns MISMATCH_LARGE
- [ ] Velocity error weighted lower than position
- [ ] Rotation error weighted appropriately
- [ ] Quaternion angle calculation correct

---

### 2.2 Unit Tests: Rollback and Replay

**File**: `tests/blackbox_reconciliation.py`

**Acceptance Criteria**:
- [ ] `rollback_to_server_state()` resets state to server
- [ ] `replay_inputs()` applies all unconfirmed inputs
- [ ] Final state matches prediction if inputs identical
- [ ] Statistics updated on each reconciliation
- [ ] History records frame for debugging

---

### 2.3 Unit Tests: Reconciliation Statistics

**File**: `tests/blackbox_reconciliation_stats.py`

**Acceptance Criteria**:
- [ ] `record()` increments total count
- [ ] MATCH result increments match count
- [ ] MISMATCH_SMALL increments mismatch count
- [ ] MISMATCH_LARGE increments snap count
- [ ] `get_mismatch_rate()` calculates correctly

---

## 3. Entity Interpolation Tasks

### 3.1 Unit Tests: Linear Interpolation

**File**: `tests/blackbox_interpolation.py`

**Acceptance Criteria**:
- [ ] `lerp_position()` at t=0 returns start
- [ ] `lerp_position()` at t=1 returns end
- [ ] `lerp_position()` at t=0.5 returns midpoint
- [ ] `lerp_position()` clamps t to [0,1]

---

### 3.2 Unit Tests: Spherical Interpolation

**File**: `tests/blackbox_slerp.py`

**Acceptance Criteria**:
- [ ] `slerp_rotation()` at t=0 returns q1
- [ ] `slerp_rotation()` at t=1 returns q2
- [ ] Result is unit quaternion
- [ ] Shorter path taken (dot < 0 handling)
- [ ] Near-parallel quaternions use nlerp fallback

---

### 3.3 Unit Tests: Hermite Interpolation

**File**: `tests/blackbox_hermite.py`

**Acceptance Criteria**:
- [ ] `hermite_interpolate()` at t=0 returns p0
- [ ] `hermite_interpolate()` at t=1 returns p1
- [ ] Velocity tangents affect curve shape
- [ ] Smooth acceleration/deceleration at endpoints

---

### 3.4 Unit Tests: Interpolation Buffer

**File**: `tests/blackbox_interpolation_buffer.py`

**Acceptance Criteria**:
- [ ] `push_snapshot()` adds in sorted order
- [ ] Out-of-order snapshots inserted correctly
- [ ] `get_interpolated()` returns interpolated state
- [ ] Extrapolation triggered when render_time > newest
- [ ] Extrapolation capped at limit
- [ ] `is_extrapolated` flag set correctly

---

### 3.5 Unit Tests: Entity Interpolator

**File**: `tests/blackbox_entity_interpolator.py`

**Acceptance Criteria**:
- [ ] `add_snapshot()` populates buffer
- [ ] `update()` returns state at render_time - delay
- [ ] Interpolation delay configurable
- [ ] Multiple entities interpolated independently

---

## 4. Correction Smoothing Tasks

### 4.1 Unit Tests: Smoothing Methods

**File**: `tests/blackbox_smoothing.py`

**Acceptance Criteria**:
- [ ] SNAP method returns target immediately
- [ ] INTERPOLATE method blends over time
- [ ] EXPONENTIAL method converges asymptotically
- [ ] THRESHOLD auto-selects based on error magnitude

---

### 4.2 Unit Tests: Correction Smoother

**File**: `tests/blackbox_correction_smoother.py`

**Acceptance Criteria**:
- [ ] `apply_correction()` with large error snaps
- [ ] `apply_correction()` with small error starts blend
- [ ] `update()` advances blend progress
- [ ] Correction completes at blend_duration
- [ ] Blend time calculated from error and speed

---

### 4.3 Unit Tests: Visual Smoother

**File**: `tests/blackbox_visual_smoother.py`

**Acceptance Criteria**:
- [ ] `set_simulation_state()` updates authoritative position
- [ ] `update()` moves visual toward simulation
- [ ] Exponential smoothing converges over time
- [ ] `snap_to_simulation()` immediately syncs

---

## 5. Lag Compensation Tasks

### 5.1 Unit Tests: Bounds

**File**: `tests/blackbox_bounds.py`

**Acceptance Criteria**:
- [ ] Intersection test: overlapping bounds return True
- [ ] Intersection test: non-overlapping return False
- [ ] Containment test: point inside returns True
- [ ] Containment test: point outside returns False
- [ ] Translation shifts bounds correctly
- [ ] Center and extents calculated correctly

---

### 5.2 Unit Tests: Hitbox History

**File**: `tests/blackbox_hitbox_history.py`

**Acceptance Criteria**:
- [ ] `record()` stores snapshot by tick
- [ ] `get_hitbox_at_time()` returns closest snapshot
- [ ] `get_hitbox_at_tick()` returns exact tick snapshot
- [ ] `get_interpolated_hitbox()` interpolates position
- [ ] Bounds not interpolated (assumed stable)
- [ ] Frame cache enables O(1) tick lookup
- [ ] Old snapshots pruned at max_frames

---

### 5.3 Unit Tests: Rewind Manager

**File**: `tests/blackbox_rewind_manager.py`

**Acceptance Criteria**:
- [ ] `record_frame()` deep-copies world state
- [ ] `get_frame_at_time()` returns closest frame
- [ ] `get_frame_at_tick()` returns exact tick frame
- [ ] `rewind_to()` marks manager as rewound
- [ ] `rewind_to()` while rewound raises RuntimeError
- [ ] `restore_to_current()` clears rewind flag
- [ ] `can_rewind_to()` checks history bounds
- [ ] Interpolated frame calculates position/velocity lerp

---

### 5.4 Unit Tests: View Time Calculator

**File**: `tests/blackbox_view_time.py`

**Acceptance Criteria**:
- [ ] `add_rtt_sample()` updates running stats
- [ ] RTT smoothing converges to sample mean
- [ ] Jitter calculated from RTT variance
- [ ] `get_interpolated_view_time()` uses jitter buffer
- [ ] `get_conservative_view_time()` uses min RTT
- [ ] `get_liberal_view_time()` uses max RTT + jitter
- [ ] `get_view_time_range()` returns (conservative, liberal)
- [ ] Max lag compensation clamping applied

---

### 5.5 Unit Tests: Lag Compensation Validator

**File**: `tests/blackbox_lag_validator.py`

**Acceptance Criteria**:
- [ ] `register_client()` creates calculator
- [ ] `validate_view_time_claim()` accepts valid claim
- [ ] `validate_view_time_claim()` rejects invalid claim
- [ ] Violation count incremented on invalid
- [ ] `is_suspicious()` returns True above threshold
- [ ] Corrected view time returned on invalid claim

---

## 6. Integration Tests

### 6.1 Prediction-Reconciliation Cycle

**File**: `tests/integration_prediction.py`

**Acceptance Criteria**:
- [ ] Client predicts movement locally
- [ ] Server validates and sends authoritative state
- [ ] Client reconciles with server state
- [ ] Minor prediction errors smoothed
- [ ] Major prediction errors snapped
- [ ] Inputs replayed correctly after rollback

---

### 6.2 Interpolation for Remote Entities

**File**: `tests/integration_interpolation.py`

**Acceptance Criteria**:
- [ ] Remote entity snapshots buffered
- [ ] Interpolation delay hides jitter
- [ ] Entity moves smoothly between updates
- [ ] Extrapolation kicks in when updates late
- [ ] Extrapolation limited to prevent overshoot

---

### 6.3 Lag Compensation Hit Detection

**File**: `tests/integration_lag_compensation.py`

**Acceptance Criteria**:
- [ ] Server records world state each tick
- [ ] On weapon fire, rewind to client view time
- [ ] Hit detection at historical positions
- [ ] State restored after hit check
- [ ] Damage applied in current state
- [ ] Multiple simultaneous shots handled

---

### 6.4 Anti-Cheat Validation

**File**: `tests/integration_anti_cheat.py`

**Acceptance Criteria**:
- [ ] Valid view time claims accepted
- [ ] Invalid view time claims rejected and corrected
- [ ] Repeated violations flagged as suspicious
- [ ] Suspicious clients reported for action

---

## 7. Gap Tasks

### 7.1 Gap: Rotation Interpolation

**File**: `engine/networking/lag_compensation/rewind_manager.py` (modify)

**Background**: Code comments note rotation interpolation is "skipped for simplicity."

**Acceptance Criteria**:
- [ ] Quaternion slerp for rotation interpolation
- [ ] Fast-rotating entities track correctly
- [ ] Performance overhead < 5%

---

### 7.2 Gap: Extrapolation Using Velocity

**File**: `engine/networking/lag_compensation/rewind_manager.py` (modify)

**Background**: If view time is ahead of newest frame, system clamps. Could extrapolate.

**Acceptance Criteria**:
- [ ] Extrapolate position using last known velocity
- [ ] Extrapolation limit prevents unrealistic overshoot
- [ ] Flag extrapolated results for debugging

---

### 7.3 Gap: Per-Bone Hitboxes

**File**: `engine/networking/lag_compensation/hitbox_history.py` (modify)

**Background**: Current Bounds is a single AABB. Skeletal hitboxes need pose history.

**Acceptance Criteria**:
- [ ] Support multiple named hitboxes per entity
- [ ] Hitbox transforms relative to root
- [ ] Animation pose history for skeletal entities
- [ ] Performance: < 2x overhead vs single AABB

---

### 7.4 Gap: Multi-Hit Reconciliation

**File**: `engine/networking/lag_compensation/` (new)

**Background**: Each shot validated independently. Burst weapons may benefit from batching.

**Acceptance Criteria**:
- [ ] Batch validation API for multiple shots
- [ ] Single rewind for all shots in burst
- [ ] Results returned as list
- [ ] Performance: O(1) rewind cost for batch

---

## 8. Performance Tasks

### 8.1 Benchmark: Prediction Throughput

**File**: `benchmarks/prediction_throughput.py`

**Acceptance Criteria**:
- [ ] Input buffer operations: > 100,000/second
- [ ] State prediction (physics): > 50,000/second
- [ ] Reconciliation cycle: > 10,000/second

---

### 8.2 Benchmark: Interpolation Performance

**File**: `benchmarks/interpolation_performance.py`

**Acceptance Criteria**:
- [ ] Linear lerp: > 1,000,000 vectors/second
- [ ] Quaternion slerp: > 500,000 quaternions/second
- [ ] Hermite interpolation: > 200,000 vectors/second
- [ ] 100 entities interpolated: < 0.5ms per frame

---

### 8.3 Benchmark: Lag Compensation

**File**: `benchmarks/lag_compensation_performance.py`

**Acceptance Criteria**:
- [ ] World state recording: < 0.1ms per tick
- [ ] Rewind to historical state: < 0.05ms
- [ ] Hit detection at rewound state: < 0.1ms per entity
- [ ] Full hit validation cycle: < 1ms

---

## 9. Test Data Requirements

### 9.1 Synthetic RTT Samples

**Acceptance Criteria**:
- [ ] Generate RTT samples with configurable mean/variance
- [ ] Simulate jitter spikes
- [ ] Simulate packet loss patterns

---

### 9.2 Movement Input Sequences

**Acceptance Criteria**:
- [ ] Generate WASD movement sequences
- [ ] Include jump timing
- [ ] Simulate various play patterns (strafe, bunny hop)

---

### 9.3 Historical State Fixtures

**Acceptance Criteria**:
- [ ] Pre-recorded world state sequences
- [ ] Multiple entity configurations
- [ ] Various tick rates (20, 60, 128 Hz)

---

## 10. Documentation Tasks

### 10.1 Prediction System Guide

**Acceptance Criteria**:
- [ ] Input buffer sizing recommendations
- [ ] Physics parameter tuning
- [ ] Misprediction debugging guide

---

### 10.2 Lag Compensation Guide

**Acceptance Criteria**:
- [ ] View time calculation explanation
- [ ] Rewind depth configuration
- [ ] Anti-cheat threshold tuning
- [ ] Hitbox history sizing
