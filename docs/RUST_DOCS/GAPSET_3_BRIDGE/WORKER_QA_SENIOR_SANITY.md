# SENIOR_QA SANITY -- T-FG-4.5 Barrier Descriptors

**Reviewer:** SENIOR_QA  
**Scope:** T-FG-4.5 — Barrier descriptor types and generation (mod.rs lines 1830-2115)  
**Method:** Direct source-code inspection of `frame_graph/mod.rs` lines 1830-2260, `blackbox_barriers.rs`, and architecture docs  
**Date:** 2026-05-22  
**Stance:** JUDICIAL — each finding evaluated on code correctness, scope relevance, and severity  

---

## VOTE SUMMARY

| Finding | Level | JUNIOR Verdict | SENIOR Verdict |
|---------|-------|---------------|----------------|
| C9: `generate_barriers()` not called by `compile()` | CRITICAL | Dead code | **OVERZEALOUS** |
| C11: `or_insert()` loses multi-resource boundaries | CRITICAL | Correctness bug | **REAL** |
| H2a: `compute_barriers()` drops ResourceHandle | HIGH | Data loss | **REAL** |
| H2b: Dead `_passes` param in `generate_barriers()` | HIGH | Dead parameter | **OVERZEALOUS** |
| H2c: 7 pre-existing WHITEBOX test failures | HIGH | Preexisting | **OVERZEALOUS** |
| M2: Missing doc on `resource_desc_is_texture()` | MEDIUM | Missing doc | **OVERZEALOUS** |
| M5: Panic msg doesn't list valid states | MEDIUM | Poor UX | **REAL** |
| M3: No multi-resource same-boundary test | MEDIUM | Coverage gap | **REAL** |
| L1: String descriptors instead of wgpu types | LOW | Architecture question | **OVERZEALOUS** |
| L2: DepthStencilReadOnly -> TextureBinding | LOW | Possibly wrong | **OVERZEALOUS** |

**Totals: 4 REAL, 6 OVERZEALOUS**

---

## CRITICAL FINDINGS

### C9: `generate_barriers()` not called by `compile()`

**Cited code:** `generate_barriers()` (mod.rs:2055-2115), `compile()` (mod.rs:2210-2251)

**Analysis:**
`compile()` calls `compute_barriers()` (line 2227) and stores the raw barrier tuples in `self.barriers`. It does NOT call `generate_barriers()`. However, this is **by architectural design**, not an oversight:

1. The architecture separates **compilation** (producing the barrier plan) from **execution** (materializing barrier commands). The doc comment on `generate_barriers` explicitly says: *"Produced by `generate_barriers` and consumed by the runtime to record `wgpu::CommandEncoder::insert_barriers` calls at the correct point in the command stream."*

2. `generate_barriers` is `pub fn` with full doc comments, tests in both whitebox (`mod.rs` #\[cfg(test)\] module) and blackbox (`blackbox_barriers.rs`). It is intentionally public API for the GPU runtime backend.

3. The `CompiledFrameGraph` stores raw tuples (`self.barriers: Vec<(PassIndex, PassIndex, ResourceState, ResourceState)>`) because these are the portable, serializable plan. `generate_barriers` resolves these tuples into concrete `BarrierCommand` objects — a step the runtime backend performs when executing the graph.

4. Similar pattern: `compute_lifetimes()` is called from `compile()` and its result is stored via `let _lifetimes = ...` (line 2224, underscore-prefixed, not stored on the struct). This is purely informational/planned-for-future-use, yet no one flags it as dead code.

**Verdict: OVERZEALOUS.** `generate_barriers` is forward-looking public API for the runtime backend, not dead code. The compilation/execution boundary is a valid architectural separation.

---

### C11: `or_insert()` in `edge_resource` loses multi-resource boundaries

**Cited code:** `generate_barriers()` lines 2063-2069:
```rust
let edge_resource: HashMap<(PassIndex, PassIndex), ResourceHandle> = {
    let mut m = HashMap::new();
    for e in edges {
        m.entry((e.from, e.to)).or_insert(e.resource);
    }
    m
};
```

**Analysis:**
This is a genuine correctness bug. The `edge_resource` HashMap maps `(from, to)` pass boundaries to a **single** `ResourceHandle`. When multiple resources have edges between the same two passes (e.g., P0 writes both texture A and buffer B, P1 reads both), `or_insert` keeps only the **first** resource handle.

In the barrier processing loop (lines 2079-2107), every barrier tuple at that boundary looks up `edge_resource.get(&(from, to))` — and ALL get the same (first) resource handle. This means:
- **Wrong resource handle assigned** to the second and subsequent barriers at the same boundary.
- **Wrong barrier type classification** if a texture barrier gets a buffer resource handle or vice versa.
- **No panic** — silently produces incorrect barrier commands.

The root cause is that the barrier tuple type `(PassIndex, PassIndex, ResourceState, ResourceState)` does not carry a `ResourceHandle`, forcing `generate_barriers` to reconstruct it from edges. This reconstruction is lossy for multi-resource boundaries.

**Verdict: REAL.** This is a correctness bug that produces silent data corruption when multiple resources share a pass boundary. The fix requires either (a) adding `ResourceHandle` to the barrier tuple, or (b) changing `edge_resource` to `HashMap<(PassIndex, PassIndex), Vec<ResourceHandle>>` and matching barriers to handles by state.

---

## HIGH FINDINGS

### H2a: `compute_barriers()` drops `ResourceHandle` from return tuple

**Cited code:** `compute_barriers()` signature (mod.rs:1842-1846):
```rust
pub fn compute_barriers(
    ordered_passes: &[PassIndex],
    passes: &[IrPass],
    edges: &[IrEdge],
) -> Vec<(PassIndex, PassIndex, ResourceState, ResourceState)>
```

**Analysis:**
The return type `(PassIndex, PassIndex, ResourceState, ResourceState)` omits `ResourceHandle`. This is factually accurate — the handle is not in the tuple. The function processes edges which DO carry a resource handle, but this information is discarded from the output.

This is the **direct root cause of C11**. Because `compute_barriers` doesn't include the handle in its output, `generate_barriers` must reconstruct it. The reconstruction (`edge_resource` HashMap) is lossy for multi-resource boundaries.

However, this only becomes a bug when multiple resources share a boundary. For the single-resource-per-boundary case (which is what the current tests exercise), the handle can be correctly reconstructed. The JUNIOR_QA correctly identified this as the root cause.

**Verdict: REAL.** The barrier tuple should carry a `ResourceHandle` (making it a 5-tuple), or the entire pipeline should pass `(from, to, resource, before, after)` through from `compute_barriers` to `generate_barriers`. This fix also resolves C11.

---

### H2b: Dead `_passes` parameter in `generate_barriers()`

**Cited code:** mod.rs:2057:
```rust
    _passes: &[IrPass],
```

**Analysis:**
The `_passes` parameter is prefixed with an underscore, which is the Rust convention for "intentionally unused." Removing it would change the public API signature and break all callers (including blackbox tests). It was likely included for forward-looking use (e.g., pass-level validation or metadata extraction during barrier generation).

Characterizing an underscore-prefixed parameter as "dead" is factually inaccurate — Rust's `_` prefix explicitly signals "this is not dead, I know it's unused, that's intentional."

**Verdict: OVERZEALOUS.** Standard Rust convention. Not a defect.

---

### H2c: 7 pre-existing WHITEBOX test failures (not in barrier code)

**Cited code:** N/A — pre-existing, not in T-FG-4.5 scope.

**Analysis:**
Pre-existing test failures outside the barrier descriptor code are not findings against T-FG-4.5. Filing them here dilutes the signal-to-noise ratio. They should be tracked as a separate infrastructure concern.

**Verdict: OVERZEALOUS.** Out of scope for T-FG-4.5 barrier descriptor review.

---

## MEDIUM FINDINGS

### M2: Missing doc on `resource_desc_is_texture()`

**Cited code:** mod.rs:2009-2015:
```rust
/// Returns `true` if `desc` describes any kind of texture resource.
fn resource_desc_is_texture(desc: &ResourceDesc) -> bool {
    matches!(
        desc,
        ResourceDesc::Texture2D(_) | ResourceDesc::Texture3D(_) | ResourceDesc::TextureCube(_),
    )
}
```

**Analysis:**
The function **has** a doc comment: *"Returns true if desc describes any kind of texture resource."* It is a private (`fn`, not `pub fn`) 3-line helper with a self-documenting name (`resource_desc_is_texture`). The doc is present, correct, and sufficient for a private utility function.

**Verdict: OVERZEALOUS.** The doc exists and is adequate for the function's scope.

---

### M5: Panic message doesn't list valid states

**Cited code:** mod.rs:1981-1984:
```rust
_ => panic!(
    "resource_state_to_texture_usage: {:?} has no texture counterpart",
    state,
),
```

**Analysis:**
The panic message tells the developer which invalid state was passed but does not enumerate which states ARE valid. A developer hitting this panic must read the source code to learn the valid states. Compare to `resource_state_to_buffer_usage` which has the same pattern (line 2002-2005).

This is a legitimate developer-experience gap. Including the valid states in the message (e.g., *"Valid states for texture: ColorAttachment, DepthStencilAttachment, ..."*) would eliminate the need to consult source code.

**Verdict: REAL.** The panic should list valid states for developer-friendliness. Low severity, valid MEDIUM.

---

### M3: BLACKBOX doesn't test multi-resource same-boundary

**Analysis:**
The blackbox test `generate_barriers_mixed_texture_and_buffer()` (blackbox_barriers.rs:841) tests two resources at **different** boundaries (P0->P1 is a texture, P1->P2 is a buffer). The test `generate_barriers_multiple_boundaries_produce_multiple_commands()` (line 888) tests the same resource at two boundaries. Neither test exercises **two resources at the same boundary**.

Given that C11 is a real bug in multi-resource same-boundary handling, this coverage gap means the bug is not caught by tests.

**Verdict: REAL.** This test coverage gap directly corresponds to a confirmed correctness bug (C11).

---

## LOW FINDINGS

### L1: String descriptors instead of wgpu types (by design?)

**Cited code:** `TextureDesc.format: String` throughout.

**Analysis:**
The use of `String` for format identifiers (e.g., `"rgba8unorm"`) instead of wgpu enum types is an explicit architectural decision. The CLARIFICATION.md (Section 5, descriptor chain model) and the code comments consistently describe these as "data-descriptor layer, no wgpu dependency." The entire module is designed to be wgpu-agnostic so that the frame graph compiler can be tested and used without a GPU backend.

Changing these to wgpu types would couple the entire barrier descriptor system to wgpu, violating the separation of concerns documented in the architecture.

**Verdict: OVERZEALOUS.** This is by-design wgpu-agnosticism, not an oversight.

---

### L2: DepthStencilReadOnly -> TextureBinding mapping (possibly wrong)

**Cited code:** mod.rs:1975:
```rust
ResourceState::DepthStencilReadOnly | ResourceState::ShaderRead => "TextureBinding",
```

**Analysis:**
In wgpu, when a depth-stencil texture is used as read-only in a shader, the `wgpu::TextureUsage` flag is indeed `TextureBinding` — the texture is bound as a read-only shader resource. The alternative would be `RenderAttachment` (wrong — that implies write access) or a dedicated depth-stencil flag (wgpu does not distinguish depth-stencil read-only at the TextureUsage level; the distinction is made at the bind-group layout level via `TextureViewDescriptor.aspect`).

The mapping is correct. DepthStencilReadOnly maps to TextureBinding because that is the correct wgpu usage flag for read-only depth access in shaders.

**Verdict: OVERZEALOUS.** The mapping is correct per wgpu semantics.

---

## ROOT CAUSE CHAIN

```
compute_barriers() drops ResourceHandle from output tuple
    └── H2a: REAL (data model issue)
        └── forces generate_barriers() to reconstruct handle from edges
            └── edge_resource uses or_insert() — keeps only one handle per boundary
                └── C11: REAL (multi-resource boundaries produce wrong barriers)
                    └── M3: REAL (no test coverage for this case)
```

C9 (`generate_barriers` not called by `compile`) is **not** part of this chain — it is an unrelated architectural separation.

---

## RECOMMENDATIONS

1. **Fix H2a + C11 (combined):** Add `ResourceHandle` to the barrier tuple, changing from 4-tuple to 5-tuple `(PassIndex, PassIndex, ResourceHandle, ResourceState, ResourceState)`. This eliminates the lossy `edge_resource` reconstruction entirely. Update `compute_barriers` return type, `CompiledFrameGraph::barriers` field type, `generate_barriers` signature, and all JSON serialization accordingly.

2. **Add M3 test:** Write a BLACKBOX test where two resources (e.g., one texture, one buffer) share the same pass boundary (P0->P1 with two edges). Verify both barriers appear in the same `BarrierCommand` with correct resource handles.

3. **Fix M5:** Update panic messages in `resource_state_to_texture_usage` and `resource_state_to_buffer_usage` to list valid states for each function.

4. **Leave C9, H2b, M2, L1, L2 as-is.** These are either intentional, conventional, or out of scope.

---

*Generated by SENIOR_QA_SANITY — 2026-05-22*
