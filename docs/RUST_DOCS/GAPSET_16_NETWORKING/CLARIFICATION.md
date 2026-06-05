# CLARIFICATION.md -- Gap Analysis and Discrepancies

> **Cluster**: GAPSET_16_NETWORKING
> **Purpose**: Document discrepancies between PHASE_N_TODO.md assumptions and actual source code reality.

---

## 1. Compression Library

| TODO Specifies | Reality | Impact |
|---------------|---------|--------|
| LZ4 compression | zlib compression throughout | zlib provides better ratios but 2-5x slower throughput. Standard library dependency (no external lib needed). |

**Files affected**: `serialization/net_serializer.py`, `serialization/delta_encoder.py`
**Action**: Accept zlib as-is unless performance profiling shows it as a bottleneck. LZ4 would require adding a dependency.

---

## 2. Quality Level System

| TODO Specifies | Reality |
|---------------|---------|
| 3 quality levels (Good/Degraded/Poor) | 5 quality levels (EXCELLENT/GOOD/FAIR/POOR/CRITICAL) |

The codebase implements a more granular system via `QualityLevel` enum with hysteresis-gated transitions. AdaptiveSettings provides bandwidth-aware rate calculation with 5 presets, each defining max send rate, update frequency, and whether to skip non-essential updates.

**Action**: Update TODO to reflect 5-level system. No changes needed to code.

---

## 3. Connection States

| TODO Specifies | Reality |
|---------------|---------|
| 4 connection states | 5 states including FAILED |

`ConnectionState.FAILED = 4` is implemented in `transport/connection.py` and reached on connect timeout and idle timeout. The code correctly transitions to FAILED (not straight to DISCONNECTED) so the caller can distinguish intentional disconnection from failure.

**Action**: Update TODO to include FAILED state. No code changes needed.

---

## 4. NetGUID Size

| TODO Specifies | Reality |
|---------------|---------|
| 64-bit NetGUID | 32-bit NetGUID with authority bit |

`replication/net_guid.py` uses 32-bit IDs: bit 31 = authority flag, bits 16-30 = client ID (32K clients), bits 0-15 = object index (64K objects per client). MAX_CLIENTS is configurable. The 32-bit design is intentional for bandwidth efficiency.

**Action**: Update TODO to specify 32-bit design. No code changes needed.

---

## 5. Relevancy Strategies

| TODO Specifies | Reality |
|---------------|---------|
| 4 relevancy strategies | 6 strategies (Always/Owner/Radius/Grid/Custom/Composite) |

`replication/relevancy.py` implements 6 strategies plus `CompositeRelevancy` for combining multiple strategies. `RelevancyContext` provides spatial, priority, and visibility context.

**Action**: Update TODO to reflect 6 strategies. No code changes needed.

---

## 6. Replication Conditions

| TODO Specifies | Reality |
|---------------|---------|
| 6 conditions as separate conditions.py | 7 conditions as ReplicationCondition enum inline |

Conditions are implemented as `ReplicationCondition` enum (ALL, OWNER_ONLY, SIMULATED_PROXY, AUTONOMOUS_PROXY, SERVER_ONLY, DORMANT, NEVER) in `replication/property_replication.py`. There is no separate `conditions.py` file.

**Action**: Accept inline enum approach. No refactoring needed unless decorator integration is desired.

---

## 7. Security Anomaly Types

| TODO Specifies | Reality |
|---------------|---------|
| ~4 anomaly types (approximation) | 10 AnomalyType values |

The TODO significantly underestimates the security system. The codebase implements: SPEED_HACK, TELEPORT, FLY_HACK, AIMBOT, NO_SPREAD, NO_RECOIL, RAPID_FIRE, INPUT_BURST, TIMING_ANOMALY, CUSTOM. Each has configurable thresholds and severity levels.

**Action**: Update TODO with full anomaly catalog. No code changes needed.

---

## 8. Security Severity Tiers

| TODO Specifies | Reality |
|---------------|---------|
| 3-tier severity | 4-tier severity (LOW/MEDIUM/HIGH/CRITICAL) |

The response system uses 4 severity levels mapped to escalation rules, not 3 as the TODO implies.

**Action**: Update TODO. No code changes needed.

---

## 9. Connection Handshake

| TODO Specifies | Reality |
|---------------|---------|
| Cryptographic challenge in handshake | Simple 2-packet handshake (CONNECT/CONNECT_ACK) |

The connection handshake does not include cryptographic challenge-response. This is a security gap but standard for game UDP networking where encryption is typically handled at a higher layer (DTLS or application-level).

**Action**: Document as known gap. Consider adding DTLS or crypto challenge if security requirements escalate.

---

## 10. Foundation Decorator Integration

| TODO Specifies | Reality |
|---------------|---------|
| @networked, @rpc, @interest, @serializable, etc. integrated | Python layer is standalone with its own decorators |

This is the largest architectural gap. NETWORKING_CONTEXT.md describes Foundation decorator integration but none is implemented. The Python code has its own `@rpc` decorator in `rpc_manager.py` and ReplicatedProperty class in `property_replication.py`, but they do not interface with Foundation's Tracker, EventLog, or Mirror systems.

**Action**: Architectural decision needed. Options:
1. Bridge Python layer to Foundation via adapter pattern
2. Keep standalone — the Python layer is fully functional without it
3. Re-implement in Rust/WASM for Foundation-native integration

---

## 11. Phase 8 Deterministic Lockstep

| TODO Specifies | Reality |
|---------------|---------|
| 7 tasks described | 0 tasks implemented (empty __init__.py) |

The entire lockstep phase is missing. It blocks on Grail 1 (S15 Core Systems) which provides: fixed-point math, command-based mutation, 13-phase tick, deterministic RNG, hierarchical checksums.

**Action**: Do not start until Grail 1 completes. Document as explicit dependency.

---

## 12. Missing Files vs. Inline Implementation

| Missing File | Inline Location | Status |
|-------------|----------------|--------|
| `conditions.py` | `replication/property_replication.py` (ReplicationCondition enum) | Inline acceptable |
| `replication/__init__.py` | Not created | Does not break imports (namespace package) |
| `prediction/__init__.py` | Not created | Does not break imports (namespace package) |
| `lag_compensation/__init__.py` | Not created | Does not break imports (namespace package) |

**Action**: Create __init__.py files if explicit package exports are desired. Current namespace package approach works.

---

## 13. Test Coverage Reality

| TODO Specifies | Reality |
|---------------|---------|
| No mention of tests | 920-line test_security.py with 16 test classes |

Only the security module has tests. All other phases lack test coverage entirely.

**Action**: Add test files for transport, serialization, replication, RPC, prediction, and lag compensation (6 new test files minimum).

---

## 14. TCP Transport Missing

The TODO lists TCP transport as planned but no implementation exists. The current code only supports UDP. This blocks asset download channels and REST API integration.

**Action**: Implement as needed for non-realtime use cases. Medium priority.

---

## 15. Server Browser Missing

The TODO lists server browser functionality for the social module. The code has matchmaking, lobby, party, voice, and text chat, but no server discovery mechanism.

**Action**: Implement via UDP broadcast + LAN discovery as described in TODO. Medium priority.

---

*End of CLARIFICATION.md*
