# PHASE 3 ARCHITECTURE: Prediction and Lag Compensation

## Phase Overview

Phase 3 implements client-side prediction for responsive controls and server-side lag compensation for fair hit detection. These systems work together to hide network latency from players.

---

## 1. Client-Side Prediction Architecture

### 1.1 Component Overview

```
ClientPredictor
    |
    +-- InputBuffer
    |       - Circular buffer of unconfirmed inputs
    |       - BufferedInput: sequence, input_data, timestamp, predicted_state
    |       - pop_confirmed(last_confirmed_seq)
    |       - get_unconfirmed()
    |
    +-- PredictionState
    |       - position, velocity, rotation (quaternion)
    |       - sequence, timestamp, custom_data
    |       - apply_input(input_data, delta_time)
    |       - clone(), distance_to(other)
    |
    +-- Accuracy Tracking
            - predictions_total, mispredictions
            - prediction_accuracy property
```

### 1.2 Prediction State Physics

```python
def apply_input(self, input_data: dict, delta_time: float) -> 'PredictionState':
    # Extract movement input
    move_x = input_data.get('move_x', 0.0) * MOVE_SPEED * delta_time
    move_z = input_data.get('move_z', 0.0) * MOVE_SPEED * delta_time
    
    # Apply friction
    new_vx = (self.velocity[0] + move_x) * FRICTION
    new_vz = (self.velocity[2] + move_z) * FRICTION
    
    # Jump handling
    new_vy = self.velocity[1]
    if input_data.get('jump') and abs(self.position[1]) < GROUND_CHECK_TOLERANCE:
        new_vy = JUMP_VELOCITY
    
    # Gravity
    new_vy += GRAVITY * delta_time
    
    # Position integration
    new_position = (
        self.position[0] + new_vx * delta_time,
        max(0.0, self.position[1] + new_vy * delta_time),  # Ground clamp
        self.position[2] + new_vz * delta_time,
    )
    
    return PredictionState(
        position=new_position,
        velocity=(new_vx, new_vy, new_vz),
        rotation=self.rotation,
        sequence=self.sequence + 1,
        timestamp=self.timestamp + delta_time,
    )
```

### 1.3 Input Buffer Management

```python
class InputBuffer:
    def __init__(self, max_size=DEFAULT_INPUT_BUFFER_SIZE):
        self._buffer: deque[BufferedInput] = deque(maxlen=max_size)
        self._sequence_to_index: dict[int, int] = {}
    
    def push(self, input_data: dict, sequence: int, predicted_state: PredictionState):
        entry = BufferedInput(
            sequence=sequence,
            input_data=input_data,
            timestamp=time.time(),
            predicted_state=predicted_state
        )
        self._buffer.append(entry)
        self._sequence_to_index[sequence] = len(self._buffer) - 1
    
    def pop_confirmed(self, last_confirmed_seq: int) -> list[BufferedInput]:
        """Remove all inputs up to and including last_confirmed_seq."""
        confirmed = []
        while self._buffer and self._buffer[0].sequence <= last_confirmed_seq:
            entry = self._buffer.popleft()
            del self._sequence_to_index[entry.sequence]
            confirmed.append(entry)
        self._rebuild_index()
        return confirmed
    
    def get_unconfirmed(self) -> list[BufferedInput]:
        """Return all remaining unconfirmed inputs."""
        return list(self._buffer)
```

---

## 2. Server Reconciliation Architecture

### 2.1 Component Overview

```
ServerReconciler
    |
    +-- State Comparison
    |       - compare_states(predicted, authoritative)
    |       - ReconciliationResult: MATCH, MISMATCH_SMALL, MISMATCH_LARGE, ERROR
    |
    +-- Rollback
    |       - rollback_to_server_state(server_state)
    |
    +-- Replay
    |       - replay_inputs(input_buffer)
    |
    +-- Statistics
    |       - ReconciliationStats: total, matches, mismatches, snaps
    |
    +-- History (Debug)
            - ReconciliationHistory
            - ReconciliationFrame: predicted, authoritative, result, timestamp
```

### 2.2 State Comparison Algorithm

```python
def compare_states(self, predicted: PredictionState, authoritative: PredictionState) -> ReconciliationResult:
    # Position error
    position_error = euclidean_distance(predicted.position, authoritative.position)
    
    # Velocity error (weighted lower)
    velocity_error = euclidean_distance(predicted.velocity, authoritative.velocity)
    
    # Rotation error (quaternion angle)
    dot = quaternion_dot(predicted.rotation, authoritative.rotation)
    rotation_error = 2.0 * acos(abs(dot))  # radians
    
    # Total weighted error
    total_error = (
        position_error +
        velocity_error * VELOCITY_WEIGHT +
        rotation_error * ROTATION_WEIGHT
    )
    
    if total_error < MATCH_THRESHOLD:
        return ReconciliationResult.MATCH
    elif total_error < SNAP_THRESHOLD:
        return ReconciliationResult.MISMATCH_SMALL
    else:
        return ReconciliationResult.MISMATCH_LARGE
```

### 2.3 Rollback and Replay

```python
def reconcile(self, server_state: PredictionState, input_buffer: InputBuffer) -> PredictionState:
    # Get predicted state at server's sequence
    predicted = input_buffer.get_input_at_sequence(server_state.sequence)
    if not predicted:
        return server_state  # No prediction to compare
    
    result = self.compare_states(predicted.predicted_state, server_state)
    self._stats.record(result)
    
    if result == ReconciliationResult.MATCH:
        # Prediction correct, continue normally
        input_buffer.pop_confirmed(server_state.sequence)
        return None  # No correction needed
    
    # Rollback to server state
    current_state = self.rollback_to_server_state(server_state)
    
    # Confirm processed inputs
    input_buffer.pop_confirmed(server_state.sequence)
    
    # Replay unconfirmed inputs
    for buffered in input_buffer.get_unconfirmed():
        current_state = current_state.apply_input(
            buffered.input_data,
            DEFAULT_DELTA_TIME
        )
    
    return current_state
```

---

## 3. Entity Interpolation Architecture

### 3.1 Component Overview

```
EntityInterpolator
    |
    +-- InterpolationBuffer
    |       - Sorted deque by timestamp
    |       - push_snapshot(snapshot)
    |       - get_interpolated(render_time)
    |
    +-- Interpolation Modes
    |       - LINEAR: Simple lerp
    |       - HERMITE: Cubic spline with velocity
    |       - CATMULL_ROM: (placeholder)
    |
    +-- Extrapolation
            - Linear using last velocity
            - Clamped to extrapolation_limit
```

### 3.2 Interpolation Algorithms

**Linear Interpolation**:

```python
def lerp_position(a: Vector3, b: Vector3, t: float) -> Vector3:
    t = clamp(t, 0.0, 1.0)
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )
```

**Spherical Linear Interpolation (Quaternions)**:

```python
def slerp_rotation(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    dot = quaternion_dot(q1, q2)
    
    # Take shorter path
    if dot < 0:
        q2 = (-q2[0], -q2[1], -q2[2], -q2[3])
        dot = -dot
    
    # Use nlerp for nearly parallel quaternions
    if dot > QUATERNION_LERP_THRESHOLD:
        result = lerp_quaternion(q1, q2, t)
        return normalize(result)
    
    # Full slerp
    theta_0 = acos(dot)
    theta = theta_0 * t
    sin_theta = sin(theta)
    sin_theta_0 = sin(theta_0)
    
    s0 = cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (
        q1[0] * s0 + q2[0] * s1,
        q1[1] * s0 + q2[1] * s1,
        q1[2] * s0 + q2[2] * s1,
        q1[3] * s0 + q2[3] * s1,
    )
```

**Hermite Interpolation**:

```python
def hermite_interpolate(p0, p1, v0, v1, t: float, duration: float) -> Vector3:
    # Tangents scaled by duration
    m0 = (v0[0] * duration, v0[1] * duration, v0[2] * duration)
    m1 = (v1[0] * duration, v1[1] * duration, v1[2] * duration)
    
    # Hermite basis functions
    t2 = t * t
    t3 = t2 * t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    
    return (
        h00*p0[0] + h10*m0[0] + h01*p1[0] + h11*m1[0],
        h00*p0[1] + h10*m0[1] + h01*p1[1] + h11*m1[1],
        h00*p0[2] + h10*m0[2] + h01*p1[2] + h11*m1[2],
    )
```

### 3.3 Interpolation Buffer

```python
class InterpolationBuffer:
    def __init__(self, max_size=DEFAULT_INTERPOLATION_BUFFER_SIZE):
        self._snapshots: deque[Snapshot] = deque(maxlen=max_size)
        self._extrapolation_limit = DEFAULT_EXTRAPOLATION_LIMIT
    
    def push_snapshot(self, snapshot: Snapshot):
        """Insert snapshot in sorted order by timestamp."""
        if not self._snapshots or snapshot.timestamp > self._snapshots[-1].timestamp:
            self._snapshots.append(snapshot)
        else:
            # Insert in correct position (out-of-order handling)
            for i, existing in enumerate(self._snapshots):
                if snapshot.timestamp < existing.timestamp:
                    self._snapshots.insert(i, snapshot)
                    break
    
    def get_interpolated(self, render_time: float) -> InterpolatedState:
        """Get interpolated state at render_time."""
        if len(self._snapshots) < 2:
            return None
        
        # Find surrounding snapshots
        before, after = None, None
        for i in range(len(self._snapshots) - 1):
            if self._snapshots[i].timestamp <= render_time <= self._snapshots[i+1].timestamp:
                before = self._snapshots[i]
                after = self._snapshots[i+1]
                break
        
        if before and after:
            t = (render_time - before.timestamp) / (after.timestamp - before.timestamp)
            return self._interpolate(before, after, t)
        
        # Extrapolate if needed
        if render_time > self._snapshots[-1].timestamp:
            return self._extrapolate(render_time)
        
        return None
```

---

## 4. Correction Smoothing Architecture

### 4.1 Component Overview

```
CorrectionSmoother
    |
    +-- SmoothingMethod
    |       - SNAP: Immediate teleport
    |       - INTERPOLATE: Linear blend over time
    |       - THRESHOLD: Auto-select based on error
    |       - EXPONENTIAL: Fast start, slow convergence
    |
    +-- CorrectionState
    |       - start_position, target_position
    |       - elapsed_time, blend_duration
    |
    +-- VisualSmoother (separate visual/simulation positions)
            - simulation_position (authoritative)
            - visual_position (smoothed)
```

### 4.2 Smoothing Algorithms

**Linear Interpolation**:

```python
def smooth_position(start: Vector3, target: Vector3, t: float) -> Vector3:
    t = clamp(t, 0.0, 1.0)
    return lerp_position(start, target, t)
```

**Exponential Smoothing**:

```python
def exponential_smooth(current: float, target: float, factor: float, dt: float) -> float:
    blend = 1.0 - exp(-factor * dt)
    return current + (target - current) * blend

def exponential_smooth_vector(current: Vector3, target: Vector3, factor: float, dt: float) -> Vector3:
    blend = 1.0 - exp(-factor * dt)
    return (
        current[0] + (target[0] - current[0]) * blend,
        current[1] + (target[1] - current[1]) * blend,
        current[2] + (target[2] - current[2]) * blend,
    )
```

### 4.3 Correction Flow

```python
class CorrectionSmoother:
    def apply_correction(self, new_target: Vector3) -> Vector3:
        error = distance(self._current_position, new_target)
        
        if error > SNAP_THRESHOLD:
            # Large error: snap immediately
            self._current_position = new_target
            return new_target
        
        # Start smooth correction
        self._correction = CorrectionState(
            start_position=self._current_position,
            target_position=new_target,
            elapsed_time=0.0,
            blend_duration=self._calculate_blend_time(error)
        )
        return self._current_position
    
    def update(self, delta_time: float) -> Vector3:
        if not self._correction:
            return self._current_position
        
        self._correction.elapsed_time += delta_time
        t = self._correction.elapsed_time / self._correction.blend_duration
        
        if t >= 1.0:
            # Correction complete
            self._current_position = self._correction.target_position
            self._correction = None
        else:
            self._current_position = smooth_position(
                self._correction.start_position,
                self._correction.target_position,
                t
            )
        
        return self._current_position
```

---

## 5. Lag Compensation Architecture

### 5.1 Component Overview

```
Lag Compensation System
    |
    +-- HitboxHistory
    |       - Per-entity hitbox snapshots
    |       - Bounds: AABB with intersection tests
    |       - HitboxSnapshot: position, bounds, tick, active
    |       - Frame cache for tick-based lookup
    |
    +-- RewindManager
    |       - Full world state history
    |       - EntityState: position, rotation, velocity, custom
    |       - WorldState: all entities at one tick
    |       - HistoryFrame: tick + timestamp + WorldState
    |
    +-- ViewTimeCalculator
    |       - Per-client RTT tracking
    |       - RTT smoothing (EWMA)
    |       - View time calculation: server_time - RTT/2 - interpolation_delay
    |       - Jitter buffer support
    |
    +-- LagCompensationValidator
            - Anti-cheat validation
            - View time claim verification
            - Violation tracking
```

### 5.2 View Time Calculation

```python
def calculate_client_view_time(server_time: float, rtt: float, interpolation_delay: float) -> float:
    """Calculate what time the client was perceiving."""
    return server_time - (rtt / 2.0) - interpolation_delay

class ViewTimeCalculator:
    def get_interpolated_view_time(self, server_time: float) -> float:
        """View time with jitter compensation."""
        rtt = self._get_smoothed_rtt()
        jitter = self._get_jitter() * JITTER_STANDARD_DEVIATIONS
        
        view_time = server_time - (rtt / 2.0) - self._interpolation_delay - jitter
        
        # Clamp to max lag compensation
        min_view_time = server_time - self._max_lag_compensation
        return max(view_time, min_view_time)
    
    def get_view_time_range(self, server_time: float) -> tuple[float, float]:
        """Return (conservative, liberal) view time for uncertainty window."""
        conservative = self.get_conservative_view_time(server_time)  # Uses min RTT
        liberal = self.get_liberal_view_time(server_time)            # Uses max RTT + jitter
        return (conservative, liberal)
```

### 5.3 Hit Detection Flow

```
Client fires weapon
        |
        v
ViewTimeCalculator.get_interpolated_view_time(server_time)
        |
        +----------------------+
        |                      |
        v                      v
RewindManager            HitboxHistory
.rewind_to(view_time)    .get_interpolated_hitbox(view_time)
        |                      |
        +----------------------+
                    |
                    v
        Perform hit detection at historical state
                    |
                    v
        RewindManager.restore_to_current()
                    |
                    v
        Apply damage/effects in live state
```

### 5.4 Hitbox History

```python
class HitboxHistory:
    def __init__(self, max_frames=DEFAULT_HITBOX_HISTORY_FRAMES):
        self._entities: dict[int, EntityHitboxHistory] = {}
        self._frame_cache: dict[int, dict[int, HitboxSnapshot]] = {}
    
    def record(self, entity_id: int, position: Vector3, bounds: Bounds, tick: int, timestamp: float):
        """Record hitbox snapshot for entity."""
        snapshot = HitboxSnapshot(position, bounds, tick, timestamp, active=True)
        
        if entity_id not in self._entities:
            self._entities[entity_id] = EntityHitboxHistory(max_frames)
        
        self._entities[entity_id].add(snapshot)
        
        # Cache by tick for fast lookup
        if tick not in self._frame_cache:
            self._frame_cache[tick] = {}
        self._frame_cache[tick][entity_id] = snapshot
    
    def get_interpolated_hitbox(self, entity_id: int, timestamp: float) -> Optional[HitboxSnapshot]:
        """Get hitbox interpolated to exact timestamp."""
        history = self._entities.get(entity_id)
        if not history:
            return None
        
        before, after = history.get_surrounding(timestamp)
        if not before or not after:
            return before or after
        
        # Interpolate position only (bounds assumed stable)
        t = (timestamp - before.timestamp) / (after.timestamp - before.timestamp)
        interpolated_position = lerp_position(before.position, after.position, t)
        
        return HitboxSnapshot(
            position=interpolated_position,
            bounds=before.bounds,  # Use before's bounds
            tick=before.tick,
            timestamp=timestamp,
            active=before.active
        )
```

### 5.5 Rewind Manager

```python
class RewindManager:
    def __init__(self, max_rewind_time_ms=DEFAULT_MAX_REWIND_TIME_MS):
        self._max_frames = calculate_max_history_frames(max_rewind_time_ms)
        self._history: deque[HistoryFrame] = deque(maxlen=self._max_frames)
        self._is_rewound = False
        self._rewound_frame: Optional[HistoryFrame] = None
    
    def record_frame(self, world_state: WorldState, tick: int):
        """Record current world state (deep copy)."""
        frame = HistoryFrame(
            tick=tick,
            timestamp=world_state.timestamp,
            world_state=world_state.copy()  # Deep copy!
        )
        self._history.append(frame)
    
    def rewind_to(self, target_time: float) -> Optional[WorldState]:
        """Rewind to historical state. Must call restore_to_current() after."""
        if self._is_rewound:
            raise RuntimeError("Already rewound. Call restore_to_current() first.")
        
        frame = self._get_frame_at_time(target_time)
        if not frame:
            return None
        
        self._is_rewound = True
        self._rewound_frame = frame
        return frame.world_state
    
    def restore_to_current(self) -> None:
        """Restore to current state after rewind operation."""
        if not self._is_rewound:
            raise RuntimeError("Not currently rewound.")
        self._is_rewound = False
        self._rewound_frame = None
```

### 5.6 Anti-Cheat Validation

```python
class LagCompensationValidator:
    def validate_view_time_claim(self, client_id: int, claimed_view_time: float, server_time: float) -> tuple[bool, float]:
        """Validate client's claimed view time against expected range."""
        calculator = self._calculators.get(client_id)
        if not calculator:
            return False, server_time
        
        expected = calculator.get_interpolated_view_time(server_time)
        deviation = abs(claimed_view_time - expected)
        
        if deviation > MAX_VIEW_TIME_DEVIATION:
            self._violation_counts[client_id] = self._violation_counts.get(client_id, 0) + 1
            return False, expected  # Correct to server-calculated value
        
        return True, claimed_view_time
    
    def is_suspicious(self, client_id: int) -> bool:
        """Check if client has accumulated too many violations."""
        return self._violation_counts.get(client_id, 0) > SUSPICIOUS_THRESHOLD
```

---

## 6. Data Flow Summary

```
Client Frame
    |
    +-- Input captured
    |
    +-- ClientPredictor.predict(input)
    |       - InputBuffer.push(input, sequence, predicted_state)
    |       - PredictionState.apply_input()
    |
    +-- Send input to server
    |
    v
Server receives input
    |
    +-- Validate input
    +-- Apply to authoritative state
    +-- Send state update with sequence
    |
    v
Client receives server state
    |
    +-- ServerReconciler.reconcile(server_state, input_buffer)
    |       - compare_states()
    |       - if MISMATCH: rollback_to_server_state()
    |       - replay_inputs()
    |
    +-- CorrectionSmoother.apply_correction(corrected_state)
    |
    +-- EntityInterpolator.update(render_time) [for remote entities]
    |
    v
Render frame
```
