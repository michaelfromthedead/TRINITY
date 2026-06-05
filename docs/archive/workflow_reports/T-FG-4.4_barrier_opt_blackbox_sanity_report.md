# T-FG-4.4 Barrier Optimizer -- Blackbox Sanity Report

**Reviewer**: Senior QA (Sanity)
**Date**: 2026-05-23
**Task**: Verify DEV fixes to 6 blackbox test files (~106 original errors)

---

## Execution Summary

| File | Original Errors | Remaining Errors | Verdict |
|------|:-:|:-:|:-:|
| `blackbox_async2.rs` | ~20 | **0** | **PASS -- REAL fix** |
| `blackbox_regression.rs` | ~20 | **0** | **PASS -- REAL fix** |
| `blackbox_compiler.rs` | ~15 | **5** | **DNP -- OVERZEALOUS** |
| `blackbox_alias_policy.rs` | ~15 | **11** | **DNP -- OVERZEALOUS** |
| `blackbox_integration.rs` | ~18 | **18** | **DNP -- OVERZEALOUS** |
| `blackbox_fix7_7.rs` | ~18 | **23** | **DNP -- OVERZEALOUS** |

**Key**: PASS = compiles clean. DNP = Did Not Pass (compilation errors remain).

---

## File-by-File Analysis

### 1. `blackbox_async2.rs` -- PASS (0 errors) -- REAL

Compiles clean against the public API. All async scheduling tests are correctly structured. DEV correctly discovered the API surface.

**Verdict: REAL fix. No issues found.**

---

### 2. `blackbox_regression.rs` -- PASS (0 errors) -- REAL

Compiles clean. Barrier 5-tuple integrity, BarrierOptimizer standalone, TextureCube round-trips -- all tests use valid public types and functions.

**Verdict: REAL fix. No issues found.**

---

### 3. `blackbox_compiler.rs` -- 5 errors -- OVERZEALOUS

#### Error breakdown:

| # | Error | Line(s) | Root Cause | Classification |
|---|-------|---------|------------|---------------|
| 1 | `E0609`: no field `texture_barriers` on `&(PassIndex, PassIndex, ResourceState, ResourceState)` | 601 | `compiled.barriers` is `Vec<(PassIndex, PassIndex, ResourceState, ResourceState)>` (4-tuples). DEV assumed `BarrierCommand` struct with `.texture_barriers`. | OVERZEALOUS |
| 2 | `E0308`: mismatched types | ~602 | Cascading from same barrier-tuple-vs-struct mismatch. | OVERZEALOUS |
| 3 | `E0277`: `CompiledFrameGraph` does not implement `Debug` | 951 | `format!("{:?}", compiled)` not available. DEV assumed `#[derive(Debug)]`. | OVERZEALOUS |
| 4 | `E0609`: no field `texture_barriers` on tuple | 984-988 | Same barrier-type confusion in `barrier_command_fields_verify_shape`. | OVERZEALOUS |
| 5 | `E0609`: no field `buffer_barriers` on tuple | 994-997 | Same barrier-type confusion. | OVERZEALOUS |

**Root cause**: `CompiledFrameGraph.barriers` is `Vec<(PassIndex, PassIndex, ResourceState, ResourceState)>` (line 3085 of `mod.rs`), a flat 4-tuple without sub-fields. DEV wrote tests as if it were `Vec<BarrierCommand>` which has `.texture_barriers` and `.buffer_barriers`. The `BarrierCommand` type exists but is only produced by `generate_barriers()`, not stored on the compiled graph.

---

### 4. `blackbox_alias_policy.rs` -- 11 errors -- OVERZEALOUS

#### Error breakdown:

| # | Error | Line(s) | Root Cause | Classification |
|---|-------|---------|------------|---------------|
| 1-3 | `E0432`: unresolved imports `apply_aliasing`, `AliasMapping`, `AliasPolicy` | 72-74 | These types/functions do not appear in the public API of `renderer_backend::frame_graph`. | OVERZEALOUS |
| 4-10 | `E0609`: no field `is_transient` on `IrResource` | 599, 600, 649, 673, 674, 698, 699 | `IrResource` has `is_history: bool` (line 938), not `is_transient`. The `Transient` vs `Imported` distinction is via `lifetime: ResourceLifetime`. | OVERZEALOUS |
| 11 | `E0308`: mismatched types | multiple | Cascading from non-existent function/type imports. | OVERZEALOUS |

**Root cause**: DEV invented a non-existent aliasing API (`apply_aliasing`, `AliasPolicy`, `AliasMapping`) and used a `is_transient` field name that doesn't exist on `IrResource`. The actual `IrResource` lifetime is tracked via `lifetime: ResourceLifetime` enum.

---

### 5. `blackbox_integration.rs` -- 18 errors -- OVERZEALOUS

#### Error breakdown:

| # | Error | Line(s) | Root Cause | Classification |
|---|-------|---------|------------|---------------|
| 1-3 | `E0432`: unresolved imports `deserialize_from_json`, `execute`, `round_trip_test` | 48-50 | These functions are not exported from `frame_graph` module. `round_trip_test` exists as a private test helper. `execute` and `deserialize_from_json` are not present. | OVERZEALOUS |
| 4-7 | `E0609`: no field `perf_counters` on `CompiledFrameGraph` | 362-373 | DEV assumed `CompiledFrameGraph.perf_counters` with sub-fields `total_us`, `dag_build_us`, `topo_sort_us`, etc. No such field exists. Fields on `CompiledFrameGraph` that store timing are `compilation_time_us: u64` (single u64). | OVERZEALOUS |
| 8 | `E0609`: no field `compilation_time_us` on `&CullStats` | 362 (cascade) | DEV accessed `compiled.cull_stats.compilation_time_us` but `compilation_time_us` is on `CompiledFrameGraph` directly (line 3139), not on `CullStats`. | OVERZEALOUS |
| 9-12 | `E0609`: no field `barriers_total` on `CullStats` | 352, 771, 773 | `CullStats` (lines 2707-2735) has `passes_total`, `passes_eliminated`, `culled_pass_count`, `resources_freed`, `bytes_saved`, `live_pass_count`, `estimated_gpu_time_saved_ms`, `dynamically_skipped`. No barrier-related fields exist. | OVERZEALOUS |
| 13-16 | `E0609`: no field `barriers_optimized` on `CullStats` | 352, 771, 773 | Same as above -- not a CullStats field. | OVERZEALOUS |
| 17 | `E0599`: no method `compile` on `Result<CompiledFrameGraph, String>` | 960 | `FrameGraphCompiler::new()` (alias for `CompiledFrameGraph::new()`) returns `Result`, not a builder. DEV wrote `compiler.compile()` but `compiler` is already a `Result`. Should be `FrameGraphCompiler::new(passes, resources).expect("...")`. | OVERZEALOUS |
| 18 | `E0308`: mismatched types | multiple | Cascading from import/type failures. | OVERZEALOUS |

**Root cause**: Three categories of overzealous assumptions:
1. Three functions imported that are not in the public API
2. `perf_counters` struct with sub-fields assumed but doesn't exist
3. `barriers_total` / `barriers_optimized` on `CullStats` but these fields don't exist anywhere on the compiled graph
4. Wrong API pattern for `FrameGraphCompiler`

---

### 6. `blackbox_fix7_7.rs` -- 23 errors -- OVERZEALOUS

#### Error breakdown:

| # | Error | Line(s) | Root Cause | Classification |
|---|-------|---------|------------|---------------|
| 1 | `E0432`: unresolved import `JsonExporter` | 38 | `JsonExporter` is not a public type in `frame_graph`. Bridge JSON is exported via `CompiledFrameGraph` methods. | OVERZEALOUS |
| 2-18 | `E0599`: no method `compile` on `Result<CompiledFrameGraph, String>` | 57, 88, 131, 212, 263, 290, 367, 401, 463, 530, 602, 630, 657, 683, 708, 719, 841, 873 | DEV wrote: `let compiler = FrameGraphCompiler::new(passes, resources); compiler.compile().expect(...)` but `FrameGraphCompiler::new()` (alias for `CompiledFrameGraph::new()`) returns `Result<Self, String>` directly. There is no two-step builder pattern. | OVERZEALOUS |
| 19-20 | `E0308`: `optimize` expects `&[BarrierTuple]` but got `Vec<...>` | 766, 807 | DEV passed `Vec` directly instead of `&vec`. Additionally, the tuple shape is wrong -- `BarrierTuple` is `(PassIndex, PassIndex, ResourceHandle, EdgeType, ResourceState, ResourceState)` (6-tuple with EdgeType), but DEV's input tuples are missing `EdgeType`. | OVERZEALOUS |
| 21-22 | Compile errors in `barriers_optimized` related access | 349, 381 | Cascading from the `compiler.compile()` Result issue. | OVERZEALOUS |

**Root cause**: Three categories:
1. `JsonExporter` is not a public type -- DEV assumed it exists
2. `FrameGraphCompiler` API is misunderstood -- it's a type alias for `CompiledFrameGraph` whose `new()` directly calls `compile()` and returns `Result`. No separate `compile()` method call exists.
3. `BarrierOptimizer::optimize(&self, barriers: &[BarrierTuple])` takes a slice reference, not an owned Vec, and `BarrierTuple` includes `EdgeType` which DEV omitted.

---

## Rollup by Error Category

| Category | Count | Files Affected |
|----------|:-----:|----------------|
| **Wrong API shape (struct fields / tuple types)** | 14 | compiler, alias_policy, integration, fix7_7 |
| **Nonexistent imports** | 7 | alias_policy, integration, fix7_7 |
| **Wrong API pattern (two-step compile)** | 17 | integration, fix7_7 |
| **Nonexistent fields on existing structs** | 16 | compiler, alias_policy, integration |
| **`Debug` not derived** | 1 | compiler |

---

## Conclusion

**2 of 6 files are REAL fixes** (compile clean, no issues).

**4 of 6 files have remaining errors, all classified OVERZEALOUS.** Every remaining error stems from DEV assuming a type, field, function, or API pattern that does not match the actual public API surface of `renderer_backend::frame_graph`. Specifically:

- DEV assumed `CompiledFrameGraph.barriers` is `Vec<BarrierCommand>` with `.texture_barriers`/`.buffer_barriers` sub-fields. It is `Vec<(PassIndex, PassIndex, ResourceState, ResourceState)>` (4-tuple).
- DEV assumed `CullStats` has barrier-count fields (`barriers_total`, `barriers_optimized`). It does not.
- DEV assumed `CompiledFrameGraph` has a `perf_counters` field. It does not.
- DEV assumed `FrameGraphCompiler::new()` returns a builder with `.compile()`. It returns `Result<CompiledFrameGraph, String>` directly (`FrameGraphCompiler` is a type alias for `CompiledFrameGraph`).
- DEV assumed `apply_aliasing`, `AliasPolicy`, `AliasMapping`, `JsonExporter`, `deserialize_from_json`, `execute`, `round_trip_test` are public API exports. They are not.
- DEV assumed `IrResource.is_transient`. The field is `is_history`.
- DEV assumed `CompiledFrameGraph` derives `Debug`. It does not.

**Recommendation**: The two clean-compiling files (blackbox_async2.rs, blackbox_regression.rs) are merge-ready. The four failing files need a second DEV pass to adjust their API assumptions to match reality. Estimated effort: mechanical, not architectural.
