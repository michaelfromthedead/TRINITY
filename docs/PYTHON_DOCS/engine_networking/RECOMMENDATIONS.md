# RECOMMENDATIONS: engine_networking

## Rust Bridge Requirements

### High Priority

| Component | Rationale | Estimated Effort |
|-----------|-----------|------------------|
| BitWriter/BitReader | Hot path for all packet serialization, CPU-bound | 2-3 days |
| Delta Encoder compression | zlib calls are blocking, significant CPU | 2 days |
| Packet.to_bytes()/from_bytes() | Every packet touches this code | 1-2 days |
| HitboxHistory.get_interpolated_hitbox() | Called per-shot in lag compensation | 1 day |
| TokenBucket.try_consume() | Called per-packet for rate limiting | 1 day |

### Medium Priority

| Component | Rationale | Estimated Effort |
|-----------|-----------|------------------|
| RewindManager world state storage | Memory-intensive deque of snapshots | 2 days |
| Connection state machine | Correctness-critical, can parallelize | 2 days |
| Channel reliability (ACK processing) | Timing-sensitive bitmap operations | 2 days |
| GUID allocation | Lock contention point on server | 1 day |

### Low Priority

| Component | Rationale | Estimated Effort |
|-----------|-----------|------------------|
| Matchmaking algorithms | Runs infrequently (1/sec) | 3 days |
| RPC serialization | Pickle is problematic anyway, needs redesign | 3 days |
| Relevancy spatial hashing | Already efficient in Python | 2 days |
| Social features | Not performance-critical | 5 days |

## Integration Strategy

### Phase 1: Foundation (Week 1)
1. Create Rust crate `trinity_networking`
2. Define FFI types for:
   - `Packet` (header + payload bytes)
   - `Vector3`, `Quaternion` (shared math types)
   - `Bounds` (AABB for hitboxes)
3. Implement `BitWriter`/`BitReader` in Rust
4. PyO3 bindings exposing same API as Python

### Phase 2: Hot Paths (Week 2)
1. Port `delta_encoder` compression logic
2. Port `Packet` serialization
3. Port `TokenBucket` rate limiting
4. Benchmark Python vs Rust paths

### Phase 3: State Management (Week 3)
1. Port `HitboxHistory` with interpolation
2. Port `RewindManager` world state storage
3. Implement Rust-side connection tracking
4. Memory profiling and optimization

### Phase 4: Full Integration (Week 4)
1. Bridge replication entities to `ComponentStore`
2. Wire RPC parameter types to `TypeRegistry`
3. Expose network-interpolated transforms to renderer
4. Integration testing across Python/Rust boundary

## Testing Strategy

### Unit Tests (Rust)
- Bit packing round-trip
- Delta encoding correctness
- ACK bitmap processing
- Hitbox intersection math
- Rate limiter timing

### Integration Tests (Python + Rust)
- Packet serialization across FFI boundary
- Delta compression/decompression
- Connection establishment/teardown
- Replication spawn/update/destroy cycle

### Stress Tests
- 64 concurrent connections
- 1000 packets/second throughput
- 100ms artificial latency
- 5% simulated packet loss

### Fuzz Tests
- Malformed packet handling
- Invalid delta state
- Out-of-bounds rate limit abuse

## Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pickle RPC serialization | Security vulnerability (RCE) | Replace with MessagePack or Protobuf |
| No encryption | Network sniffing, tampering | Add DTLS before production |
| Single test file | Regressions undetected | Write tests before Rust port |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Threading model mismatch | Rust async vs Python GIL | Use PyO3 GIL release patterns |
| Memory layout differences | Incorrect state sharing | Define C-compatible structs |
| Time precision | Lag compensation errors | Use monotonic clocks consistently |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| API compatibility | Python code changes needed | Maintain same method signatures |
| Performance regression | Slower than Python | Unlikely, but benchmark continuously |

## Immediate Actions

1. **Create blackbox test suite** for networking before any Rust port
2. **Replace pickle** in RPC serialization with safe alternative
3. **Add DTLS** or document that encryption is out of scope
4. **Profile** current Python implementation to establish baseline
5. **Document** the network protocol format for Rust reimplementation
