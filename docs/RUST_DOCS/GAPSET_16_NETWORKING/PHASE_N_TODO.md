# PHASE_N_TODO.md — GAPSET_16_NETWORKING Tasks

> **Cluster**: GAPSET_16_NETWORKING
> **TASK_ID Format**: T-NET-{PHASE}.{N}
> **Total Tasks**: 65
> **Implementation Status**: 0% (All tasks unchecked)

---

## Phase 1: Transport Foundation (7 tasks)

**Dependencies**: S15 Core Systems (memory, sockets), S14 Platform RHI

- [ ] **T-NET-1.1** — Implement `udp_transport.py`: UDP socket creation, bind, send, receive with non-blocking I/O. Wire into platform socket abstraction.
  - **Acceptance**: Socket created, packets sent/received between two processes on localhost. MTU handling works.
  - **Dependencies**: S15-G4 (memory.rs allocators), S14 platform sockets
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-1.2** — Implement `tcp_transport.py`: TCP socket with stream encapsulation, connect/listen/accept loop. Message framing (4-byte length prefix).
  - **Acceptance**: TCP stream messages sent/received correctly between processes. Framing handles partial reads.
  - **Dependencies**: S15-G4
  - **Effort**: Medium (2-4 days)

- [ ] **T-NET-1.3** — Implement `channel.py`: 4 channel types (Unreliable, Reliable Ordered, Reliable Unordered, Sequenced). ACK-based retransmission for reliable channels. Send window (32 packets), receive buffer for reordering. Sequence tracking for sequenced channel.
  - **Acceptance**: All 4 channels deliver messages correctly under simulated packet loss (5%, 10%, 25%). Reliable channels retransmit lost packets. Sequenced drops out-of-order.
  - **Dependencies**: T-NET-1.1
  - **Effort**: High (5-8 days)

- [ ] **T-NET-1.4** — Implement `packet.py`: Packet header (type, size, sequence, ack mask, CRC32). Coalescing: batch multiple messages within MTU. 4-priority packing (Critical > High > Normal > Low). Fragmentation for messages exceeding MTU.
  - **Acceptance**: Packets created, coalesced, fragmented, reassembled correctly. CRC32 validates integrity. Priority packing verified.
  - **Dependencies**: T-NET-1.1, T-NET-1.3
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-1.5** — Wire `ProtocolMeta` to transport: Protocol class auto-registration, `get_wire_format()`, `get_version()`, `validate_message(data)`.
  - **Acceptance**: ProtocolMeta-derived classes auto-register in protocol registry. Wire format and version queries work. Message validation passes/fails correctly.
  - **Dependencies**: T-NET-1.4
  - **Effort**: Low (1-2 days)

- [ ] **T-NET-1.6** — Implement `__init__.py` for transport module: public API exports.
  - **Acceptance**: `from trinity.network.transport import *` provides all transport classes.
  - **Dependencies**: T-NET-1.1 through T-NET-1.5
  - **Effort**: Low (0.5 days)

- [ ] **T-NET-1.7** — Write transport unit tests: socket send/receive, channel reliability under loss, packet coalescing, ProtocolMeta registration.
  - **Acceptance**: 30+ test cases passing. Coverage >80% for transport module.
  - **Dependencies**: T-NET-1.1 through T-NET-1.6
  - **Effort**: Medium (2-3 days)

---

## Phase 2: Connection Management (6 tasks)

**Dependencies**: Phase 1 complete

- [ ] **T-NET-2.1** — Implement `connection.py`: 4-state connection machine (DISCONNECTED -> CONNECTING -> CONNECTED -> DISCONNECTING). SYN/Challenge/Response/ACK 4-way handshake with cryptographic challenge. Heartbeat at configurable interval (default 1s). Timeout detection (default 10s). Reconnect with sequence recovery.
  - **Acceptance**: Connection established between client/server processes. Handshake completes within expected round trips. Heartbeat maintains connection. Timeout detected. Reconnect recovers sequence state.
  - **Dependencies**: T-NET-1.1, T-NET-1.4
  - **Effort**: High (5-8 days)

- [ ] **T-NET-2.2** — Implement `quality.py`: RTT computation from ACK timestamps. Jitter as RTT standard deviation. Packet loss % from sequence gaps. Bandwidth measurement (bytes/sec sent/received). 3-state quality adaptation (Good -> Degraded -> Poor) with graduated response.
  - **Acceptance**: Metrics computed correctly under simulated network conditions. Adaptation triggers at correct thresholds. Metrics exposed via public API.
  - **Dependencies**: T-NET-1.3, T-NET-2.1
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-2.3** — Implement `nat.py`: STUN client (RFC 3489) for NAT type discovery. UDP hole punching for cone NAT. TURN relay fallback for symmetric NAT.
  - **Acceptance**: NAT type determined correctly. Hole punching succeeds for cone NAT. TURN relay established as fallback.
  - **Dependencies**: T-NET-1.1
  - **Effort**: High (5-7 days)

- [ ] **T-NET-2.4** — Implement connection quality adaptation: reduce update rate under degradation, increase compression, drop low-priority updates. Graceful degradation chain.
  - **Acceptance**: Under simulated degradation, update rate reduces, compression increases, low-priority updates dropped. Recovery when quality improves.
  - **Dependencies**: T-NET-2.2
  - **Effort**: Medium (2-3 days)

- [ ] **T-NET-2.5** — Wire connection into transport `__init__.py`: public API for connection lifecycle.
  - **Acceptance**: Connection lifecycle methods exposed. Existing tests pass.
  - **Dependencies**: T-NET-2.1 through T-NET-2.4
  - **Effort**: Low (0.5 days)

- [ ] **T-NET-2.6** — Write connection management tests: handshake, heartbeat, timeout, quality metrics, NAT traversal.
  - **Acceptance**: 25+ test cases. Coverage >80%.
  - **Dependencies**: T-NET-2.1 through T-NET-2.5
  - **Effort**: Medium (2-3 days)

---

## Phase 3: Serialization (5 tasks)

**Dependencies**: Phase 1 complete

- [ ] **T-NET-3.1** — Implement `net_serializer.py`: Binary serialization schema with field types (int, float, bool, Vec2/3/4, Quat, string, byte array). Schema registration, versioned serialization. Schema hash for protocol validation.
  - **Acceptance**: All field types serialize/deserialize correctly cross-process. Schema hash matches on both ends. Version mismatch detected.
  - **Dependencies**: T-NET-1.4, Foundation @serializable
  - **Effort**: High (5-7 days)

- [ ] **T-NET-3.2** — Implement `delta_encoder.py`: Changed-fields-only compression. Track field dirty state per frame. Encode only changed fields with field index references. Baseline snapshot + incremental deltas.
  - **Acceptance**: Delta encoding produces smaller payloads than full state for partially-changed objects. Full state sent on first transmission. Deltas correct across multiple frames.
  - **Dependencies**: T-NET-3.1
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-3.3** — Implement `quantizer.py`: Float32 -> Fixed16 Q8.8 quantization. Vec3 -> 4-component compressed representation. Quat -> smallest-three (drop largest component). Configurable quantization levels per field.
  - **Acceptance**: Quantized values stay within acceptable error bounds (Float32 -> Fixed16: <0.004). Quat reconstruction valid (unit length maintained). Vec3 error < 0.01 world units.
  - **Dependencies**: T-NET-3.1
  - **Effort**: Medium (3-4 days)

- [ ] **T-NET-3.4** — Implement `bit_packer.py`: Bit-level field packing. Variable-length integer encoding. Bit-aligned field reads/writes. LZ4 compression pass on packed payload.
  - **Acceptance**: Bit-packed payloads smaller than byte-aligned. All field values round-trip correctly. LZ4 compression reduces payload size measurably.
  - **Dependencies**: T-NET-3.1
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-3.5** — Write serialization tests: round-trip all types, delta encoding correctness, quantization error bounds, bit packing edge cases.
  - **Acceptance**: 30+ test cases. Coverage >85%. Quantization error within specified bounds.
  - **Dependencies**: T-NET-3.1 through T-NET-3.4
  - **Effort**: Medium (2-3 days)

---

## Phase 4: Replication System (9 tasks)

**Dependencies**: Phases 1, 3 complete

- [ ] **T-NET-4.1** — Implement `replication_manager.py`: Central coordinator for property replication. Per-frame cycle: collect dirty networked fields -> filter by relevancy -> prioritize by bandwidth -> serialize -> send over channels. Wire Foundation Tracker.all_dirty() for change detection.
  - **Acceptance**: Replication cycle runs per frame. Dirty fields detected, filtered, prioritized, serialized, and sent correctly. Tracker integration verified.
  - **Dependencies**: T-NET-3.1, Foundation Tracker
  - **Effort**: High (5-7 days)

- [ ] **T-NET-4.2** — Implement `net_guid.py`: 64-bit globally unique network ID assignment. GUID generation (unique per process). GUID lookup table. GUID-to-entity mapping.
  - **Acceptance**: Every entity gets unique GUID. Lookup returns correct entity. No collisions in multi-entity test.
  - **Dependencies**: None
  - **Effort**: Low (1-2 days)

- [ ] **T-NET-4.3** — Implement `actor_channel.py`: Per-entity, per-client replication channel. Open on spawn/relevancy-enter. Close on irrelevancy/destroy. Sequenced message delivery within channel. Gap detection and resync.
  - **Acceptance**: Channel opens on entity spawn, closes on destroy channels. Messages delivered in sequence. Gap detection triggers resync.
  - **Dependencies**: T-NET-1.3, T-NET-4.2
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-4.4** — Implement `relevancy.py`: Interest management with 4 strategies (radius spatial, grid spatial, custom predicate, always-relevant, owner-always). Per-client relevancy evaluation. Relevancy transition triggers (enter -> open channel, leave -> close channel).
  - **Acceptance**: Entities correctly filtered per-client by distance. Grid spatial correctly assigns cells. Custom predicate evaluated per entity. Transitions fire correctly.
  - **Dependencies**: T-NET-4.3, Foundation @interest
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-4.5** — Implement `conditions.py`: 6 replication conditions (Always, OnChange, InitialOnly, OwnerOnly, SkipOwner, Custom). Per-condition evaluation per field per client.
  - **Acceptance**: Each condition correctly filters field replication. Custom predicate evaluated. Combinations work.
  - **Dependencies**: T-NET-4.1
  - **Effort**: Medium (2-4 days)

- [ ] **T-NET-4.6** — Implement `bandwidth.py`: Per-actor bandwidth limits (max_bps). Prioritized send queue (bandwidth_priority + relevancy weight). Saturation handling: drop lowest-priority when budget exceeded. Per-client bandwidth budget.
  - **Acceptance**: Higher-priority entities sent before lower. Budget enforcement drops lowest-priority under saturation. Per-client budgets independent.
  - **Dependencies**: T-NET-4.1, Foundation @bandwidth_priority
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-4.7** — Implement `property_replication.py`: Per-field change detection via NetworkedDescriptor dirty queue. Serialize changed fields. Attach field metadata (authority, priority, change mode). Integration with net_serializer for wire format.
  - **Acceptance**: Field changes detected by descriptor. Changes serialized with metadata. 3 change notification modes work (None, RepNotify, WithPrevious).
  - **Dependencies**: T-NET-3.1, T-NET-4.1, Foundation NetworkedDescriptor
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-4.8** — Wire all replication modules into `__init__.py`.
  - **Acceptance**: Public replication API exposed. All existing tests pass.
  - **Dependencies**: T-NET-4.1 through T-NET-4.7
  - **Effort**: Low (0.5 days)

- [ ] **T-NET-4.9** — Write replication tests: full replication cycle, interest management, bandwidth enforcement, property change detection.
  - **Acceptance**: 35+ test cases. Coverage >75%.
  - **Dependencies**: T-NET-4.1 through T-NET-4.8
  - **Effort**: Medium (3-4 days)

---

## Phase 5: RPC Framework (5 tasks)

**Dependencies**: Phases 1, 2 complete

- [ ] **T-NET-5.1** — Implement `rpc_manager.py`: RPC registration (auto-discover @rpc methods via EventMeta). RPC dispatch by type. Serialize/deserialize RPC arguments. Return value handling for reliable RPCs.
  - **Acceptance**: @rpc methods auto-registered. RPC dispatched correctly by type. Arguments round-trip correctly. Return values delivered for reliable RPCs.
  - **Dependencies**: T-NET-3.1, Foundation EventMeta, @rpc
  - **Effort**: High (5-7 days)

- [ ] **T-NET-5.2** — Implement `rpc_channel.py`: Map 4 RPC types to 2 transport channels. Server/Client/Safe RPCs use Reliable Ordered. Multicast uses Unreliable. Per-channel ordering.
  - **Acceptance**: RPC types correctly mapped to channels. Server/Client/Safe RPCs delivered reliably. Multicast RPCs delivered unreliably. Ordering preserved per channel.
  - **Dependencies**: T-NET-1.3, T-NET-5.1
  - **Effort**: Medium (2-4 days)

- [ ] **T-NET-5.3** — Implement `rpc_validation.py`: 4-step validation pipeline (authority -> rate limit -> parameter bounds -> execute/reject). Per-player, per-RPC-type rate counters with token bucket replenishment. Parameter type and range validation.
  - **Acceptance**: Authority check rejects unauthorized callers. Rate limits enforced per-player per-type. Invalid parameters rejected. Valid calls pass through.
  - **Dependencies**: T-NET-5.1, Foundation Capabilities
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-5.4** — Wire RPC into networking `__init__.py`.
  - **Acceptance**: RPC API exposed. Existing tests pass.
  - **Dependencies**: T-NET-5.1 through T-NET-5.3
  - **Effort**: Low (0.5 days)

- [ ] **T-NET-5.5** — Write RPC tests: registration, dispatch, validation, rate limiting, authority.
  - **Acceptance**: 25+ test cases. Coverage >80%.
  - **Dependencies**: T-NET-5.1 through T-NET-5.4
  - **Effort**: Medium (2-3 days)

---

## Phase 6: Prediction & Reconciliation (7 tasks)

**Dependencies**: Phases 1, 3, 4, 5 complete

- [ ] **T-NET-6.1** — Implement `client_prediction.py`: Input buffer (ordered by sequence number). Local input application. Sequence tracking. Wire PredictedDescriptor for history.
  - **Acceptance**: Inputs buffered in order. Local prediction applies inputs immediately. Sequence tracking matches server. PredictedDescriptor records history.
  - **Dependencies**: T-NET-4.1, Foundation PredictedDescriptor
  - **Effort**: High (5-7 days)

- [ ] **T-NET-6.2** — Implement `server_reconciliation.py`: Receive server authoritative state. Compare predicted vs. actual at last_processed_sequence. On match: discard confirmed inputs. On mismatch: PredictedDescriptor.rollback -> apply server state -> re-apply unconfirmed inputs.
  - **Acceptance**: Matching state correctly discards inputs. Mismatch triggers rollback + replay. State converges to server authority within tolerance.
  - **Dependencies**: T-NET-6.1, Foundation Tracker.undo(), EventLog
  - **Effort**: High (5-8 days)

- [ ] **T-NET-6.3** — Implement `entity_interpolation.py`: Snapshot buffer (INTERPOLATION_BUFFER_SIZE=3). Render at time = server_time - interp_delay_ms. Linear and Hermite interpolation between snapshots. Extrapolation with velocity projection (with limit).
  - **Acceptance**: Interpolation produces smooth movement between snapshots. Extrapolation within bounds. Buffer management correct (no underflow/overflow).
  - **Dependencies**: T-NET-4.1, Foundation InterpolatedDescriptor
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-6.4** — Implement `smoothing.py`: Snap (error > snap_threshold). Interpolate (error < snap_threshold). Threshold-based dynamic selection. Configurable snap_threshold per entity.
  - **Acceptance**: Large errors snap. Small errors interpolate. Smoothing visually acceptable across various error magnitudes.
  - **Dependencies**: T-NET-6.2
  - **Effort**: Low (2-3 days)

- [ ] **T-NET-6.5** — Wire @server_reconcile decorator: configure reconciliation strategy per entity (max_reconcile_frames, snap_threshold).
  - **Acceptance**: `@server_reconcile(max_reconcile_frames=10, snap_threshold=0.5)` correctly configures reconciliation parameters per entity.
  - **Dependencies**: T-NET-6.2, Foundation @server_reconcile
  - **Effort**: Low (1-2 days)

- [ ] **T-NET-6.6** — Wire prediction into networking `__init__.py`.
  - **Acceptance**: Prediction API exposed. Existing tests pass.
  - **Dependencies**: T-NET-6.1 through T-NET-6.5
  - **Effort**: Low (0.5 days)

- [ ] **T-NET-6.7** — Write prediction/reconciliation tests: input buffer, rollback, reconciliation, interpolation, smoothing.
  - **Acceptance**: 30+ test cases. Coverage >80%.
  - **Dependencies**: T-NET-6.1 through T-NET-6.6
  - **Effort**: Medium (3-4 days)

---

## Phase 7: Lag Compensation (5 tasks)

**Dependencies**: Phases 3, 4, 5 complete

- [ ] **T-NET-7.1** — Implement `rewind_manager.py`: World state snapshot ring buffer. Rewind to timestamp (restore all entities to historical positions). Restore to current. Maximum rewind window = MAX_LAG_COMPENSATION_MS (default 500ms).
  - **Acceptance**: World state snapshotted per frame. Rewind restores correct historical positions. Restore returns to current. Rewind window limit enforced.
  - **Dependencies**: T-NET-3.2, Foundation EventLog, @snapshot
  - **Effort**: High (5-7 days)

- [ ] **T-NET-7.2** — Implement `hitbox_history.py`: Per-entity hitbox transform circular buffer. Keyed by Net GUID + frame timestamp. Reconstruction: interpolate between nearest frames if exact time unavailable.
  - **Acceptance**: Hitbox transforms recorded per frame. Retrieval by GUID + timestamp returns correct data. Interpolation for non-exact timestamps produces valid hitboxes.
  - **Dependencies**: T-NET-4.2, T-NET-7.1
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-7.3** — Implement `view_time.py`: Extract client view time from RPC timestamp (preferred) or compute as server_time - RTT/2 (fallback). Jitter compensation: smoothed RTT moving average.
  - **Acceptance**: View time extraction works correctly with both timestamped and non-timestamped RPCs. Jitter compensation reduces variance.
  - **Dependencies**: T-NET-2.2, T-NET-5.1
  - **Effort**: Medium (2-4 days)

- [ ] **T-NET-7.4** — Wire lag compensation into networking `__init__.py`.
  - **Acceptance**: Lag compensation API exposed. Existing tests pass.
  - **Dependencies**: T-NET-7.1 through T-NET-7.3
  - **Effort**: Low (0.5 days)

- [ ] **T-NET-7.5** — Write lag compensation tests: rewind/restore correctness, hitbox history, view time extraction, rewind limit enforcement.
  - **Acceptance**: 20+ test cases. Coverage >80%.
  - **Dependencies**: T-NET-7.1 through T-NET-7.4
  - **Effort**: Medium (2-3 days)

---

## Phase 8: Deterministic Lockstep Networking (7 tasks)

**Hard Dependencies**: Grail 1 Implementation (S15 Core Systems — fixed-point math, command-based mutation, 13-phase tick, deterministic RNG, hierarchical checksums)

- [ ] **T-NET-8.1** — Implement `lockstep_manager.py`: Frame-based simulation synchronization. Collect all clients' inputs for frame N. When all inputs received (or timeout): execute frame N locally. Proceed to frame N+1.
  - **Acceptance**: Simulation advances in lockstep. Frame N executed only when all inputs for N are available. Timeout with "no input" substitution works. Deterministic execution verified.
  - **Dependencies**: Grail 1 (Fixed16, Fixed32, command mutation), T-NET-1.1
  - **Effort**: High (7-10 days)

- [ ] **T-NET-8.2** — Implement `input_queue.py`: Per-client ordered input packet queue. Push (from network), pop (for simulation frame), peek next. Sequence validation. Gap detection.
  - **Acceptance**: Inputs queued per-client. Inputs retrieved in sequence for correct frame. Gap detection flags missing sequences.
  - **Dependencies**: T-NET-8.1
  - **Effort**: Medium (2-4 days)

- [ ] **T-NET-8.3** — Implement `frame_sync.py`: Frame lock protocol. Round-trip time estimation. Timeout + "no input" substitution. Roll-forward after timeout. Re-sync after divergence.
  - **Acceptance**: Frame lock/unlock works correctly. Timeout substitutes "no input" after threshold. Roll-forward produces valid simulation. Re-sync restores consistency.
  - **Dependencies**: T-NET-8.1, T-NET-2.1
  - **Effort**: High (5-7 days)

- [ ] **T-NET-8.4** — Implement `checksum_verifier.py`: Hierarchical checksum computation per Grail 1 spec. Periodic checksum exchange between clients (configurable interval, default every 10 frames). Divergence detection: checksum mismatch triggers pause + state dump. Re-sync protocol on divergence.
  - **Acceptance**: Checksums computed correctly at each hierarchy level. Checksum exchange succeeds. Mismatch detected, pause triggered, state dumped. Re-sync restores consistency.
  - **Dependencies**: Grail 1 (hierarchical checksums), T-NET-8.1
  - **Effort**: High (5-7 days)

- [ ] **T-NET-8.5** — Integrate lockstep with Grail 1 deterministic pipeline: command-based mutation, 13-phase tick, PCG-based deterministic RNG. Verify bit-identical simulation across processes.
  - **Acceptance**: Two independent processes with identical inputs produce identical final state (bit-for-bit). Checksums match at every frame. Deterministic RNG produces identical sequences.
  - **Dependencies**: T-NET-8.1 through T-NET-8.4, Grail 1 (full)
  - **Effort**: Critical (10-14 days)

- [ ] **T-NET-8.6** — Wire lockstep into networking `__init__.py`.
  - **Acceptance**: Lockstep API exposed. Existing tests pass.
  - **Dependencies**: T-NET-8.1 through T-NET-8.5
  - **Effort**: Low (0.5 days)

- [ ] **T-NET-8.7** — Write lockstep tests: bit-identical simulation across processes, checksum verification, divergence detection/recovery, timeout handling.
  - **Acceptance**: 25+ test cases including multi-process determinism verification. Coverage >80%.
  - **Dependencies**: T-NET-8.1 through T-NET-8.6
  - **Effort**: High (4-6 days)

---

## Phase 9: Anti-Cheat & Security (7 tasks)

**Dependencies**: Phases 2, 4, 5 complete

- [ ] **T-NET-9.1** — Implement `authority_validator.py`: Server authority enforcement. Validate entity write permissions (server-owned, client-owned, owner, simulated proxy). Wire Foundation Capabilities system. Reject unauthorized property writes.
  - **Acceptance**: Unauthorized writes rejected. Server-owned fields writable only by server. Client-owned fields writable by owning client. Capabilities integration verified.
  - **Dependencies**: T-NET-4.1, Foundation Capabilities, @server_authoritative
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-9.2** — Implement `input_validator.py`: Input sanity checks (speed limits, acceleration bounds, direction change rate). Physics bounds verification (position within world limits). Sequence validation (monotonically increasing, no gaps, no replay).
  - **Acceptance**: Speed hack detected. Teleportation detected. Out-of-bounds position rejected. Replay attack detected. Valid inputs pass through.
  - **Dependencies**: T-NET-6.1
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-9.3** — Implement `rate_limiter.py`: Per-player rate limiting. Per-command-type token buckets. Configurable fill rate and burst capacity. Integration with @rate_limited decorator.
  - **Acceptance**: Rate limits enforced per-player per-command. Burst allowed within capacity. @rate_limited correctly configures limits.
  - **Dependencies**: T-NET-5.3, Foundation @rate_limited
  - **Effort**: Medium (2-4 days)

- [ ] **T-NET-9.4** — Implement `anomaly_detector.py`: Statistical detection (reaction time distribution, accuracy deviation). Heuristic rules (impossible movements, illegal state transitions). ML anomaly detection stub (model architecture specified, training pipeline deferred). Per-player baseline deviation scoring.
  - **Acceptance**: Statistical detection flags abnormal patterns. Heuristic rules catch impossible states. ML stub returns unmet dependency error with clear message. Deviation scores computed correctly.
  - **Dependencies**: T-NET-9.2
  - **Effort**: High (5-8 days)

- [ ] **T-NET-9.5** — Implement `response.py`: 4-tier response system (Warning -> Kick -> Temporary Ban -> Permanent Ban + Shadow Ban). Shadow ban: isolated instance with other shadow-banned players. Response escalation per configurable thresholds.
  - **Acceptance**: Warning delivered to client. Kick disconnects player. Temporary ban enforced for duration. Shadow ban isolates player. Escalation triggers correctly.
  - **Dependencies**: T-NET-2.1
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-9.6** — Wire anti-cheat into networking `__init__.py`.
  - **Acceptance**: Anti-cheat API exposed. Existing tests pass.
  - **Dependencies**: T-NET-9.1 through T-NET-9.5
  - **Effort**: Low (0.5 days)

- [ ] **T-NET-9.7** — Write anti-cheat tests: authority validation, input validation, rate limiting, anomaly detection, response enforcement.
  - **Acceptance**: 30+ test cases. Coverage >80%.
  - **Dependencies**: T-NET-9.1 through T-NET-9.6
  - **Effort**: Medium (3-4 days)

---

## Phase 10: Social Services (7 tasks)

**Dependencies**: Phases 1, 2 complete

- [ ] **T-NET-10.1** — Implement `matchmaking.py`: Queue (join, leave, timeout). Match criteria configuration (mode, map, region, skill range). Search expansion over time. Skill rating integration (Elo/Glicko-2/TrueSkill/MMR). Match formation (select players, select server, connect all).
  - **Acceptance**: Players queue and match correctly by skill. Search expansion works. Server selected by ping. Match formed and connections initiated.
  - **Dependencies**: T-NET-2.1
  - **Effort**: High (7-10 days)

- [ ] **T-NET-10.2** — Implement `lobby.py`: Create lobby (public/private/invite-only parameters). Join by ID. Ready-up system (all ready -> countdown -> match start). Lobby settings (mode, map, max players, teams). Lobby state machine: Forming -> Ready -> Countdown -> InGame -> Closed.
  - **Acceptance**: Lobby created with correct visibility. Players join by ID. Ready-up triggers countdown when all ready. Lobby transitions through states correctly.
  - **Dependencies**: T-NET-2.1
  - **Effort**: Medium (4-6 days)

- [ ] **T-NET-10.3** — Implement `party.py`: Create party (leader). Invite system (send/accept/decline). Join as group. Leader controls matchmaking. Party state machine: Forming -> Ready -> Queuing -> InMatch -> Disbanded.
  - **Acceptance**: Party created. Invites sent and accepted. Party queues as group. Leader controls match criteria. Party states transition correctly.
  - **Dependencies**: T-NET-10.1
  - **Effort**: Medium (3-5 days)

- [ ] **T-NET-10.4** — Implement `server_browser.py`: Server list (UDP broadcast discovery). Server metadata (name, map, mode, players, ping). Filter (mode, map, region, player count). Favorites list (persistent).
  - **Acceptance**: Servers discovered via broadcast. Metadata displayed correctly. Filters work. Favorites persist across sessions.
  - **Dependencies**: T-NET-1.1, T-NET-2.1
  - **Effort**: Medium (4-6 days)

- [ ] **T-NET-10.5** — Implement `voice_chat.py`: Voice capture (mic input). Voice Activity Detection (VAD). Noise reduction pipeline. Opus encode/decode. Transmit over unreliable channel. Receive with jitter buffer. 3D positional audio (integration with audio subsystem). Voice channels: Team, Squad, Proximity, Global.
  - **Acceptance**: Voice captured, encoded, transmitted, received, decoded, and played. VAD correctly gates transmission. 3D positional audio works. Channel scoping correct.
  - **Dependencies**: T-NET-1.1, Audio subsystem (spatial audio), Opus library
  - **Effort**: High (7-10 days)

- [ ] **T-NET-10.6** — Implement `text_chat.py`: Text channels (Global, Team, Whisper, System). Message routing per channel scoping. Profanity filter (configurable word list). Rate limiting (messages/sec). Report system (flag message for moderation). Ban/timeout per chat channel.
  - **Acceptance**: Messages routed to correct channel. Profanity filter blocks configured words. Rate limits enforced. Reports logged. Channel bans enforced.
  - **Dependencies**: T-NET-5.1, T-NET-9.3
  - **Effort**: Medium (4-6 days)

- [ ] **T-NET-10.7** — Wire social services into networking `__init__.py`.
  - **Acceptance**: Social API exposed. Existing tests pass.
  - **Dependencies**: T-NET-10.1 through T-NET-10.6
  - **Effort**: Low (0.5 days)

---

## Task Summary

| Phase | Tasks | Effort Estimate |
|-------|-------|-----------------|
| 1: Transport Foundation | T-NET-1.1 through T-NET-1.7 (7 tasks) | 3-5 medium, 1 high, 1 low = ~20-30 days |
| 2: Connection Management | T-NET-2.1 through T-NET-2.6 (6 tasks) | 1 high, 3 medium, 1 low = ~18-26 days |
| 3: Serialization | T-NET-3.1 through T-NET-3.5 (5 tasks) | 1 high, 4 medium = ~16-24 days |
| 4: Replication System | T-NET-4.1 through T-NET-4.9 (9 tasks) | 1 high, 6 medium, 1 low = ~24-39 days |
| 5: RPC Framework | T-NET-5.1 through T-NET-5.5 (5 tasks) | 1 high, 2 medium, 1 low = ~13-20 days |
| 6: Prediction & Reconciliation | T-NET-6.1 through T-NET-6.7 (7 tasks) | 2 high, 3 medium, 1 low = ~20-32 days |
| 7: Lag Compensation | T-NET-7.1 through T-NET-7.5 (5 tasks) | 1 high, 2 medium, 1 low = ~13-20 days |
| 8: Deterministic Lockstep | T-NET-8.1 through T-NET-8.7 (7 tasks) | 4 high, 1 medium, 1 low = ~33-49 days |
| 9: Anti-Cheat & Security | T-NET-9.1 through T-NET-9.7 (7 tasks) | 1 high, 4 medium, 1 low = ~18-28 days |
| 10: Social Services | T-NET-10.1 through T-NET-10.7 (7 tasks) | 2 high, 4 medium, 1 low = ~27-40 days |
| **Total** | **65 tasks** | **~202-308 days (~0.8-1.2 developer-years)** |

---

## Dependency Summary for Scheduling

**Phase 8 (Deterministic Lockstep) blocks on Grail 1** — it cannot start until S15 Core Systems provides fixed-point math, command-based mutation, 13-phase tick execution, and deterministic RNG. All other phases are dependent on Phases 1-3 but are otherwise independent and can be parallelized:

```
Grail 1 (S15)
     │
     ▼
Phase 8 ──── (parallelizable with Phases 4-7, 9-10 once Phases 1-3 complete)
     │
     ▼
Phase 1 ──► Phase 2 ──► Phase 3 ──┬──► Phase 4 ──► Phase 6 ──► Phase 7
                                   │                                   │
                                   ├──► Phase 5 ──► Phase 9 ────────────┤
                                   │                                   │
                                   └──► Phase 10 ───────────────────────┘
```

---

*End of PHASE_N_TODO.md*
