# Phase 5 Architecture -- RPC Framework

> **Cluster**: GAPSET_16_NETWORKING
> **Module**: `engine/networking/rpc/`

---

## Overview

The RPC framework provides remote procedure call functionality with 4 authority types, 3 reliability modes, ordered delivery, retransmission, rate limiting, and parameter validation.

---

## File Map

| File | LOC | Role |
|------|-----|------|
| `rpc_manager.py` | 615 | @rpc decorator, method discovery, dispatch by type/reliability |
| `rpc_channel.py` | 593 | Ordered delivery, sequence numbers, retransmission queue |
| `rpc_validation.py` | 539 | Rate limiting, authority checks, parameter bounds |

---

## Architecture

### RPCManager (rpc_manager.py)

**@rpc Decorator**: Auto-discovers decorated methods on registration. Decorator parameters:

```python
@rpc(
    rpc_type=RPCType.SERVER,     # Who handles this RPC
    reliable=True,                # Reliability guarantee
    ordered=False,                # Ordering guarantee
    channel=0,                    # Transport channel
    rate_limit=60,                # Max calls per second
    timeout=30.0                  # Call timeout in seconds
)
def some_method(self, param1, param2):
    pass
```

**RPCType** (4 authority types):
| Type | Called On | Use Case |
|------|-----------|----------|
| SERVER | Server | Client -> Server requests |
| CLIENT | Specific client | Server -> Client updates |
| OWNER | Entity owner | Server -> Owning client notifications |
| MULTICAST | Multiple clients | Server -> Group broadcasts |

**Dispatch Flow**:
```
1. Parse RPC call (method name, args, type)
2. Validate authority (server checks SERVER RPCs, etc.)
3. Rate limit check
4. Route to RPCChannel for delivery
5. On receipt: validate -> deserialize -> invoke target method
```

**Method Registry**: Class-level mapping of method name -> RPCConfig. Supports inheritance via base class scanning.

### RPCChannel (rpc_channel.py)

**RPCMessage** (4 types):
- `CALL`: Remote function invocation
- `RESPONSE`: Return value delivery
- `ERROR`: Exception propagation
- `CANCEL`: Cancel pending call

**Ordered Delivery**: Sequence numbers per channel, receiver buffers out-of-order messages and delivers in sequence. Dropped messages trigger retransmit request.

**Retransmission**: Unacknowledged messages stored in pending queue with timeout. On timeout, retransmit up to max_retries.

**Pending Calls**: Tracks outstanding calls with timestamp, timeout, and callback. Calls that timeout fire error callback.

### RPCValidation (rpc_validation.py)

**RPCValidator**: Multi-stage validation pipeline:
1. **Authority check**: Verify caller has permission for this RPC type
2. **Rate limit**: Sliding window + token bucket per-method
3. **Parameter bounds**: Type checking, value range validation, length limits
4. **Cooldown**: Minimum interval between repeated calls

**RateLimiter**: Dual mechanism:
- **Sliding window**: Tracks call timestamps, rejects if window exceeded
- **Token bucket**: Per-call cost with configurable refill rate and burst capacity

**Configurable Per-Method Limits**: Each RPC method can have independent rate limits, cooldown periods, and parameter validation rules.

---

## Missing Components

1. **Dedicated test file**: No tests for RPC framework (~1,800 LOC untested).
2. **Foundation EventMeta integration**: NETWORKING_CONTEXT.md describes event-sourced RPC via Foundation EventMeta, not implemented.
3. **Streaming RPCs**: No support for long-running streaming calls or server-sent events.

---

## Reality Status

- @rpc decorator (4 types, configurable): **[x]** Complete
- RPCChannel (ordered delivery, retransmission): **[x]** Complete
- RPCValidator (authority, rate limits, parameter bounds): **[x]** Complete
- Tests: **[-]** Not implemented
- Foundation EventMeta integration: **[-]** Not implemented

---

*End of PHASE_5_ARCH.md*
