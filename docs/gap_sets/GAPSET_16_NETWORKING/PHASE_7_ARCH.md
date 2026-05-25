# Phase 7 Architecture -- Lag Compensation

> **Cluster**: GAPSET_16_NETWORKING
> **Module**: `engine/networking/lag_compensation/`

---

## Overview

Lag compensation enables fair hit detection for shooters by rewinding server state to the client's view time when evaluating hits. This prevents high-ping players from being penalized due to latency.

---

## File Map

| File | LOC | Role |
|------|-----|------|
| `rewind_manager.py` | 490 | Ring buffer of world frames, binary search rewind |
| `hitbox_history.py` | 520 | Per-entity AABB hitbox snapshots keyed by NetGUID |
| `view_time.py` | 439 | Client view time extraction with jitter compensation |

---

## Architecture

### Rewind Manager (rewind_manager.py)

**HistoryFrame**: Snapshot of all entity hitboxes at a given server time. Stored in a ring buffer with configurable capacity (default 64 frames) and duration (default 1 second).

```
[Frame 0] [Frame 1] ... [Frame N-1]  (ring buffer)
   |          |              |
 time=0     time=1         time=N-1
```

**Rewind Operation**: Binary search over frame timestamps to find the frame pair bracketing the target time:

```
1. Binary search in frame history for nearest frame
2. If exact match: use that frame
3. If between frames: interpolate hitboxes
4. If no match: return None (outside history window)
```

**Interpolation Fallback**: When target time falls between two history frames, linearly interpolates hitbox positions for smooth rewinding.

**Thread Safety**: RLock around history frame operations (read, write, rewind).

### Hitbox History (hitbox_history.py)

**HitboxSnapshot**: Collection of entity hitboxes at a moment in time. Each hitbox is defined as an **AABB (Axis-Aligned Bounding Box)** with: center (Vector3), half-extents (Vector3), and rotation (quaternion for rotated AABB support).

**EntityHitboxHistory**: Per-entity ring buffer of hitbox snapshots keyed by NetGUID + timestamp.

**HitboxHistory**: Global container mapping entity IDs to their hitbox histories. Supports:

```
record_hitboxes(entity_id, hitboxes, timestamp)
get_hitboxes_at(entity_id, timestamp) -> interpolated AABBs
```

**AABB Operations**:
- Contains point check
- AABB vs AABB intersection test
- Ray-AABB intersection (for hitscan weapons)
- Interpolation between two AABBs

### View Time Calculator (view_time.py)

**ViewTimeCalculator**: Extracts the client's view time for lag compensation:

```
Client view time = RPC timestamp                (if client provides explicit timestamp)
                 = server_time - RTT / 2         (fallback)
                 = server_time - RTT / 2 - jitter(compensation for jitter)
```

**Strategies**:
1. **Explicit timestamp**: Client includes timestamp in RPC call (most accurate)
2. **RTT/2 estimate**: Uses connection RTT as fallback (standard approach)
3. **Jitter-compensated**: Subtracts additional jitter margin for stability

**LagCompensationValidator**: Sanity checks on rewind results:
- Validates target time is within history window
- Clamps rewind time to available range
- Reports if rewind failed (outside window)
- Logs statistics for debugging

---

## Flow

```
1. Client fires weapon -> sends RPC with view timestamp
2. Server receives RPC:
   a. Extract view time (RPC timestamp or RTT/2 estimate)
   b. Rewind world state to view time
   c. Evaluate hit detection against rewound hitboxes
   d. Restore current world state
3. Return hit result to client
```

---

## Missing Components

1. **Dedicated test file**: No tests for lag compensation (~1,500 LOC untested).
2. **Hitbox visualization**: No debug rendering for rewound hitboxes.
3. **Networked hitbox updates**: Assumes hitboxes arrive via replication -- no explicit hitbox replication channel.

---

## Reality Status

- RewindManager (ring buffer, binary search, interpolation): **[x]** Complete
- HitboxHistory (AABB, per-entity, ray intersection): **[x]** Complete
- ViewTimeCalculator (RTT/2, jitter-compensated): **[x]** Complete
- LagCompensationValidator: **[x]** Complete
- Tests: **[-]** Not implemented

---

*End of PHASE_7_ARCH.md*
