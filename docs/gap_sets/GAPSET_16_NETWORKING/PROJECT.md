# PROJECT.md -- GAPSET_16_NETWORKING Implementation Reference

> **Purpose**: Single authoritative reference for the engine/networking/ layer.
> **Scope**: All 65 tasks across 10 phases with Reality annotations.
> **Key Files**: All paths relative to `engine/networking/`.

---

## 1. Project Overview

The networking layer provides multiplayer connectivity for the Trinity engine. It implements a client-server architecture with UDP transport, property replication, RPCs, client-side prediction with server reconciliation, lag compensation, anti-cheat, and social services. The codebase is written in Python with typed interfaces and is approximately 75-80% complete.

### Source Tree (48 Python files)

```
transport/          -- 5 files (4 implemented, 2 missing)
serialization/      -- 4 files (4 implemented)
replication/        -- 6 files (6 implemented, conditions inline)
rpc/                -- 3 files (3 implemented)
prediction/         -- 4 files (4 implemented)
lag_compensation/   -- 3 files (3 implemented)
security/           -- 7 files (7 implemented, 1 test file)
social/             -- 8 files (7 implemented, 1 missing)
tests/              -- 1 test file (test_security.py)
determinism/network/ -- empty init (lockstep not started)
```

---

## 2. Phase Architecture

### Phase 1: Transport Foundation

| File | Lines | Status | Key Classes |
|------|-------|--------|-------------|
| `transport/udp_transport.py` | 534 | [x] | UDPTransport, TransportConfig, TransportStats, TransportEvent, TransportEventData |
| `transport/tcp_transport.py` | -- | [-] | NOT IMPLEMENTED |
| `transport/channel.py` | 578 | [x] | Channel, UnreliableChannel, ReliableChannel, ReliableOrderedChannel, SequencedChannel, ChannelConfig, ChannelManager |
| `transport/packet.py` | 437 | [x] | Packet, PacketHeader, PacketType (12), PacketFlags (7), Fragmenter |
| `transport/quality.py` | 518 | [x] | QualityLevel (5), QualityMetrics, QualityMonitor, AdaptiveSettings, NetworkQualityAdapter |
| `transport/__init__.py` | 69 | [x] | Exports all transport classes |
| `transport/connection.py` | 532 | [x] | Connection, ConnectionState (5), ConnectionConfig, ConnectionStats |

**Architecture**: UDPTransport manages a single DGRAM socket, routes packets to Connection objects by (host,port). Each Connection owns a ChannelManager with 4 default channels. Channels provide different reliability/ordering semantics. QualityMonitor tracks RTT/jitter/loss/bandwidth with EWMA smoothing. NetworkQualityAdapter provides hysteresis-gated adaptation across 5 quality levels.

**Missing**: TCP transport stream wrapper, NAT traversal (STUN/hole punch/TURN), ProtocolMeta decorator integration.

### Phase 2: Connection Management

*Implemented within transport/ files above* (no separate directory).

| Component | Lines | Status | Key Features |
|-----------|-------|--------|--------------|
| Connection | 532 | [x] | 5-state machine, 4-way handshake, heartbeats, idle timeout, sequence tracking, ACK processing |
| QualityMonitor | 518 | [x] | EWMA RTT, variance jitter, loss window, bandwidth tracking |
| NetworkQualityAdapter | inline | [x] | 5 presets, hysteresis, bandwidth-aware rate calculation |
| __init__ wiring | 69 | [x] | Public API exports |

**Missing**: NAT traversal (STUN/hole punch/TURN relay), cryptographic challenge in handshake, explicit reconnect with sequence recovery.

### Phase 3: Serialization

| File | Lines | Status | Key Classes |
|------|-------|--------|-------------|
| `serialization/net_serializer.py` | 513 | [x] | NetSerializer (23 MessageTypes), MessageHeader (20-byte), custom type registration |
| `serialization/delta_encoder.py` | 580 | [x] | DeltaEncoder, DeltaSchema (5 field type IDs), SnapshotDeltaEncoder, baseline tracking |
| `serialization/quantizer.py` | 437 | [x] | quantize/dequantize_float (8/12/16/24-bit), quantize/dequantize_vector3, quantize/dequantize_quaternion (smallest-three), quantize_angle, unit_float, signed_unit_float |
| `serialization/bit_packer.py` | 442 | [x] | BitWriter (write_bits/int/float_compressed/bytes/string), BitReader (read_bits/int/float_compressed/peek/skip) |

**Architecture**: Serialization pipeline: NetSerializer (schema + types) -> DeltaEncoder (changed fields) -> Quantizer (precision reduction) -> BitPacker (bit-aligned) -> zlib compression. Uses zlib throughout (not LZ4 as TODO specifies).

**Missing**: Dedicated test file.

### Phase 4: Replication

| File | Lines | Status | Key Classes |
|------|-------|--------|-------------|
| `replication/replication_manager.py` | 748 | [x] | ReplicationManager, ReplicationRole (4), EntityState (4), 4 packet types |
| `replication/net_guid.py` | 399 | [x] | NetGUID (32-bit with authority), NetGUIDManager |
| `replication/actor_channel.py` | 681 | [x] | ActorChannel (4 states, 6 message types) |
| `replication/relevancy.py` | 572 | [x] | RelevancyStrategy (Always/Owner/Radius/Grid/Custom), CompositeRelevancy, RelevancyContext |
| `replication/bandwidth.py` | 580 | [x] | BandwidthManager (5 priorities), TokenBucket, SendQueue with anti-starvation |
| `replication/property_replication.py` | 437 | [x] | ReplicatedProperty, ReplicationCondition (7), NotifyMode (3) |

**Architecture**: ReplicationManager orchestrates the per-frame cycle: collect dirty properties -> filter by relevancy -> prioritize by bandwidth -> serialize -> send. NetGUIDManager assigns 32-bit unique IDs. ActorChannel manages per-entity, per-client state streams. Relevancy uses spatial/grid/custom strategies. BandwidthManager uses token bucket with priority queue and anti-starvation.

**Notes**: Conditions are implemented inline in property_replication.py as ReplicationCondition enum (not a separate conditions.py). No direct Foundation decorator integration (@interest, @bandwidth_priority, @networked).

### Phase 5: RPC Framework

| File | Lines | Status | Key Classes |
|------|-------|--------|-------------|
| `rpc/rpc_manager.py` | 615 | [x] | RPCManager, @rpc decorator, RPCType (4), RPCConfig |
| `rpc/rpc_channel.py` | 593 | [x] | RPCChannel, RPCMessage (4 types), ordered delivery, retransmission |
| `rpc/rpc_validation.py` | 539 | [x] | RateLimiter (sliding window + token bucket), RPCValidator |

**Architecture**: RPCManager auto-discovers @rpc-decorated methods, dispatches by type (SERVER/CLIENT/OWNER/MULTICAST) and reliability. RPCChannel provides ordered delivery with sequence numbers and retransmission queue. RPCValidator enforces authority, rate limits, and parameter bounds.

**Missing**: Dedicated test file, Foundation EventMeta integration.

### Phase 6: Prediction & Reconciliation

| File | Lines | Status | Key Classes |
|------|-------|--------|-------------|
| `prediction/client_prediction.py` | 559 | [x] | InputBuffer, PredictionState, ClientPredictor |
| `prediction/server_reconciliation.py` | 431 | [x] | ServerReconciler (compare/rollback/replay/smooth) |
| `prediction/entity_interpolation.py` | 588 | [x] | InterpolationBuffer, EntityInterpolator, lerp/slerp/hermite |
| `prediction/smoothing.py` | 553 | [x] | CorrectionSmoother (4 modes), VisualSmoother |

**Architecture**: ClientPredictor applies inputs locally and stores prediction states. ServerReconciler compares predicted vs authoritative states, triggers rollback on mismatch, re-applies buffered inputs. EntityInterpolator provides buffered interpolation for non-predicted entities with lerp/slerp/hermite. CorrectionSmoother supports snap/interpolate/exponential/threshold modes.

**Missing**: @server_reconcile decorator wiring (described in NETWORKING_CONTEXT.md but not implemented), dedicated test file.

### Phase 7: Lag Compensation

| File | Lines | Status | Key Classes |
|------|-------|--------|-------------|
| `lag_compensation/rewind_manager.py` | 490 | [x] | RewindManager, HistoryFrame (ring buffer), binary search |
| `lag_compensation/hitbox_history.py` | 520 | [x] | Bounds (AABB), HitboxSnapshot, EntityHitboxHistory, HitboxHistory |
| `lag_compensation/view_time.py` | 439 | [x] | ViewTimeCalculator, LagCompensationValidator |

**Architecture**: RewindManager maintains a ring buffer of world state frames, supports rewinding to any timestamp via binary search (with interpolated frame fallback). HitboxHistory tracks per-entity hitbox transforms keyed by NetGUID + timestamp. ViewTimeCalculator extracts client view time from RPC timestamps or server_time - RTT/2 fallback with jitter compensation.

**Missing**: Dedicated test file.

### Phase 8: Deterministic Lockstep

| File | Lines | Status |
|------|-------|--------|
| `determinism/network/__init__.py` | 1 | [-] Empty file |
| All 7 TODO tasks | -- | [-] NOT IMPLEMENTED |

**Status**: Entirely missing. Blocks on Grail 1 (S15 Core Systems: fixed-point math, command-based mutation, 13-phase tick, deterministic RNG, hierarchical checksums). Should not be started until Grail 1 completes.

### Phase 9: Anti-Cheat & Security

| File | Lines | Status | Key Classes |
|------|-------|--------|-------------|
| `security/authority_validator.py` | 388 | [x] | AuthorityValidator, Caller, Entity, Authority, EntityAuthority, FieldAuthority |
| `security/input_validator.py` | 524 | [x] | InputValidator, PlayerState, ValidationResult (8), Vector3, InputBounds |
| `security/rate_limiter.py` | 478 | [x] | TokenBucket, RateLimiter, AdaptiveRateLimiter, RateLimitConfig, RateLimitResult |
| `security/anomaly_detector.py` | 687 | [x] | AnomalyDetector, AnomalyType (10), AnomalySeverity (4), PlayerStats, AnomalyReport |
| `security/response.py` | 597 | [x] | ResponseManager, EscalationRule, BanRecord, CheatResponse, ResponseSeverity (5) |
| `security/config.py` | 208 | [x] | 7 frozen dataclass configs |
| `security/__init__.py` | ~60 | [x] | Comprehensive exports |

**Architecture**: Full security pipeline: InputValidator (sanity checks) -> RateLimiter (token buckets) -> AnomalyDetector (statistical/heuristic) -> ResponseManager (escalation). AuthorityValidator provides field-level write permissions. AdaptiveRateLimiter responds to server load.

**Test file**: `tests/test_networking/test_security.py` (920 lines, 16 test classes, unit + adversarial + thread-safety + integration + bypass tests). **The only test file in the entire networking layer.**

### Phase 10: Social Services

| File | Lines | Status | Key Classes |
|------|-------|--------|-------------|
| `social/matchmaking.py` | 547 | [x] | MatchmakingQueue, MatchmakingService |
| `social/skill_rating.py` | 662 | [x] | Elo (K-factor), Glicko-2, MMRManager |
| `social/lobby.py` | 847 | [x] | Lobby, LobbyManager |
| `social/party.py` | 921 | [x] | Party, PartyManager |
| `social/server_browser.py` | -- | [-] | NOT IMPLEMENTED |
| `social/voice_chat.py` | 764 | [x] | VoiceChatManager, ProximityVoice |
| `social/text_chat.py` | 848 | [x] | ChatManager, ProfanityFilter |
| `social/config.py` | 173 | [x] | SocialConfig (6 nested configs) |

**Missing**: Server browser for server discovery via UDP broadcast.

---

## 3. Cross-Cutting Concerns

### Thread Safety
- AuthorityValidator: RLock around all write/validate operations
- AnomalyDetector: RLock around event record/analyze
- RateLimiter: RLock around token operations
- ResponseManager: RLock around ban/violation operations
- NetGUIDManager: RLock around allocate/release
- ClientPredictor: Thread-safe via careful state management

### Foundation Integration (NOT implemented)
The NETWORKING_CONTEXT.md describes integration with Foundation decorators (@networked, @rpc, @serializable, @tracked, @interest, @bandwidth_priority, etc.), descriptors (NetworkedDescriptor, PredictedDescriptor, InterpolatedDescriptor, ThrottledNetworkDescriptor), and systems (Tracker, EventLog, Capabilities, Mirror, DeltaSync, Bridge). **None of these integrations are implemented in the Python networking layer.** The networking code is standalone Python with its own decorators and class hierarchies.

### Compression
The codebase uses **zlib** throughout for compression (not LZ4 as specified in the TODO). This affects performance characteristics: zlib provides better compression ratios but slower throughput than LZ4.

### Configuration
All configuration values are centralized in `config.py` (603 lines) with a frozen `NetworkConfig` dataclass. Sub-modules have their own configs:
- `security/config.py`: 7 frozen dataclass configs
- `social/config.py`: SocialConfig with 6 nested frozen dataclass configs

---

## 4. File Inventory (48 Python files)

| Module | Files | Lines Range | Total LOC |
|--------|-------|-------------|-----------|
| transport/ | 5 .py + __init__ | 437-578 | ~2,600 |
| serialization/ | 4 .py + __init__ | 437-580 | ~2,000 |
| replication/ | 6 .py + __init__ | 399-748 | ~3,800 |
| rpc/ | 3 .py + __init__ | 539-615 | ~1,800 |
| prediction/ | 4 .py + __init__ | 431-588 | ~2,200 |
| lag_compensation/ | 3 .py + __init__ | 439-520 | ~1,500 |
| security/ | 6 .py + __init__ + config | 208-687 | ~2,800 |
| social/ | 6 .py + __init__ + config | 173-921 | ~4,100 |
| root/ | __init__.py + config.py | 14-603 | ~600 |
| tests/ | test_security.py | 920 | ~920 |
| **Total** | **48 files** | -- | **~22,320** |

---

*End of PROJECT.md*
