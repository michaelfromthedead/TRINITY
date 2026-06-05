# Phase 6 Architecture -- Prediction & Reconciliation

> **Cluster**: GAPSET_16_NETWORKING
> **Module**: `engine/networking/prediction/`

---

## Overview

The prediction system implements client-side prediction with server reconciliation for responsive gameplay under network latency. It also provides entity interpolation for smooth rendering of non-predicted entities and configurable correction smoothing modes.

---

## File Map

| File | LOC | Role |
|------|-----|------|
| `client_prediction.py` | 559 | InputBuffer, PredictionState, ClientPredictor |
| `server_reconciliation.py` | 431 | ServerReconciler with compare/rollback/replay/smooth |
| `entity_interpolation.py` | 588 | InterpolationBuffer, lerp/slerp/hermite, EntityInterpolator |
| `smoothing.py` | 553 | CorrectionSmoother with 4 modes, VisualSmoother |

---

## Architecture

### Client Prediction (client_prediction.py)

**InputBuffer**: Stores timed input commands with sequence numbers. Configurable capacity (default 128). Supports:
- `add_input(input, sequence)`: Append new input
- `get_inputs_since(time)`: Retrieve inputs for replay
- `ack_inputs(sequence)`: Remove inputs confirmed by server

**PredictionState**: Snapshot of predicted entity state at a point in time. Contains position, rotation, velocity, and configurable custom fields.

**ClientPredictor**: Core prediction loop:
```
1. Apply local input to entity state immediately
2. Store predicted state in history buffer
3. On server state arrival:
   a. Compare predicted vs server state
   b. If mismatch -> reconciliation
4. Return predicted state for rendering
```

**Thread Safety**: Thread-safe via careful state management (no shared locks needed -- input and state owned by caller).

### Server Reconciliation (server_reconciliation.py)

**ServerReconciler**: Full reconciliation pipeline:

```
1. COMPARE: Compare predicted state at server time vs received server state
2. ROLLBACK: If difference exceeds threshold:
   a. Revert entity to server state
   b. Clear prediction history after server time
3. REAPPLY: Re-apply all buffered inputs since server time
4. SMOOTH: Apply correction smoothing to avoid visual snaps
```

**Configurable Thresholds**: Position error (default 0.1 units), rotation error (default 0.01 rad), velocity error. Only triggers rollback when error exceeds threshold to avoid jitter on tiny discrepancies.

**Replay**: Iterates InputBuffer from server time to present, applying each input in sequence to the corrected server state. Rebuilds prediction history from the corrected trajectory.

### Entity Interpolation (entity_interpolation.py)

**InterpolationBuffer**: Timed ring buffer of entity states. Supports:
- Push new states with timestamps
- Query interpolated state at arbitrary time
- Auto-cleanup of old states

**Interpolation Methods**:
- `lerp`: Linear interpolation for position/scalar values
- `slerp`: Spherical linear interpolation for quaternions
- `hermite`: Cubic Hermite spline for smooth trajectory (requires velocity)

**EntityInterpolator**: Coordinates interpolation for all non-predicted entities. Handles:
- Buffer management per entity
- Render time calculation (server_time - half_RTT)
- Smooth blending between states

**Extrapolation**: Configurable extrapolation when buffer underflows (default: hold last state).

### Smoothing (smoothing.py)

**CorrectionSmoother** (4 modes):

| Mode | Behavior | Use Case |
|------|----------|----------|
| SNAP | Immediate correction | Small errors, authority-critical |
| INTERPOLATE | Linear blend over time | General purpose |
| EXPONENTIAL | Exponential decay towards target | Smooth convergence |
| THRESHOLD | Snap only above threshold | Jerk reduction for small errors |

**VisualSmoother**: Client-only smoothing that maintains visual continuity during corrections. Applies velocity-matched blending and minimizes visual discontinuities.

---

## Missing Components

1. **@server_reconcile decorator**: Described in NETWORKING_CONTEXT.md but not implemented. Would mark methods as reconciliation-triggering.
2. **Dedicated test file**: No tests for the prediction system (~2,200 LOC untested).
3. **Foundation integration**: PredictedDescriptor, InterpolatedDescriptor described but not implemented.

---

## Reality Status

- ClientPredictor (input buffer, prediction, history): **[x]** Complete
- ServerReconciler (compare/rollback/replay/smooth): **[x]** Complete
- EntityInterpolator (lerp/slerp/hermite, buffers): **[x]** Complete
- CorrectionSmoother (4 modes): **[x]** Complete
- @server_reconcile decorator: **[-]** Not implemented
- Tests: **[-]** Not implemented

---

*End of PHASE_6_ARCH.md*
