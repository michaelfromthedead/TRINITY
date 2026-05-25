#!/usr/bin/env python3
"""Port T-FG-7.7 changes from GRAPHICS to TRINITY.

Edits:
1. Add memory_savings_percent to CullStats + Display impl
2. Add CompileError struct before CompiledFrameGraph
3. Add errors: Vec<CompileError> to CompiledFrameGraph
4. Add errors: vec![] to compile() return + memory_savings_percent computation
5. Add memory_savings_percent to cull_stats JSON in emit_bridge_json
6. Add errors and memory_savings_percent to main JSON output
7. Update test CullStats literal
"""

import os

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mod.rs")

with open(FILE, "r") as f:
    content = f.read()

changes = 0

# ---- Edit 1a: Add memory_savings_percent field to CullStats ----
old = '''    pub estimated_gpu_time_saved_ms: f32,
}

/// A cross-timeline synchronisation point'''
new = '''    pub estimated_gpu_time_saved_ms: f32,
    /// Memory savings as a percentage of total resource footprint.
    ///
    /// Computed as `(bytes_saved + alias_bytes_saved) / total_resource_bytes * 100.0`.
    /// A value of 0.0 means no savings or zero total resources.
    pub memory_savings_percent: f32,
}

/// A cross-timeline synchronisation point'''
assert old in content, "EDIT 1a FAILED"
content = content.replace(old, new, 1)
changes += 1
print(f"Edit 1a OK: added memory_savings_percent to CullStats")

# ---- Edit 1b: Update Display impl (if present) ----
old_disp = '''    pub estimated_gpu_time_saved_ms: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType'''
# Only update Display if the old format exists
if old_disp in content:
    new_disp = old_disp.replace(
        "pub estimated_gpu_time_saved_ms: f32,",
        "pub estimated_gpu_time_saved_ms: f32,\n    pub memory_savings_percent: f32,"
    )
    # Actually no, this is a separate CullStats in the test module or somewhere else
    # Let's handle Display separately

# The Display impl I need to find and update
disp_idx = content.find("impl fmt::Display for CullStats {")
if disp_idx >= 0:
    disp_end = content.find("}", disp_idx)
    disp_section = content[disp_idx:disp_end+1]
    # Find the format string
    if "gpu_time_saved={}ms)" in disp_section:
        old_fmt = '''            "CullStats(passes_total={}, eliminated={}, resources_freed={}, bytes_saved={}, live={}, culled={}, gpu_time_saved={}ms)",
            self.passes_total, self.passes_eliminated, self.resources_freed, self.bytes_saved,
            self.live_pass_count, self.culled_pass_count, self.estimated_gpu_time_saved_ms,'''
        new_fmt = '''            "CullStats(passes_total={}, eliminated={}, resources_freed={}, bytes_saved={}, live={}, culled={}, gpu_time_saved={}ms, mem_savings={:.1}%)",
            self.passes_total, self.passes_eliminated, self.resources_freed, self.bytes_saved,
            self.live_pass_count, self.culled_pass_count, self.estimated_gpu_time_saved_ms, self.memory_savings_percent,'''
        assert old_fmt in content, "EDIT 1b FAILED: Display fmt not found"
        content = content.replace(old_fmt, new_fmt, 1)
        changes += 1
        print(f"Edit 1b OK: updated Display impl")
    else:
        print(f"WARNING: Display impl format string not found in expected shape")
else:
    print(f"WARNING: Display impl not found")

# ---- Edit 2: Add CompileError struct before SyncPoint ----
# Note: There's a stray SyncPoint between CullStats and CompiledFrameGraph
old_sync = '''pub struct SyncPoint {'''
new_sync = '''/// A non-fatal compilation error or warning produced during frame graph
/// compilation.
///
/// Unlike the `Result::Err` path (which halts compilation), `CompileError`
/// records issues that the compiler can recover from -- for example, a pass
/// that references a non-existent resource handle, or a barrier that violates
/// the expected resource state machine.  The caller can inspect these after
/// compilation and decide whether to abort execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileError {
    /// Name of the pass that triggered the error.
    pub pass_name: String,
    /// Compiler phase or validation step that produced the error
    /// (e.g. `"Phase 4"`, `"BridgeValidator"`).
    pub phase: String,
    /// Human-readable error message.
    pub message: String,
}

pub struct SyncPoint {'''

assert old_sync in content, "EDIT 2 FAILED: SyncPoint not found"
content = content.replace(old_sync, new_sync, 1)
changes += 1
print(f"Edit 2 OK: added CompileError struct")

# ---- Edit 3: Add errors: Vec<CompileError> to CompiledFrameGraph ----
old_cfg = '''    pub runtime_features: FeatureSet,
}

/// A cross-timeline synchronisation point'''
new_cfg = '''    pub runtime_features: FeatureSet,
    /// Non-fatal compilation errors / warnings (T-FG-7.7).
    ///
    /// Populated during compilation phases that encounter recoverable
    /// issues.  Empty on successful compilation.
    pub errors: Vec<CompileError>,
}

/// A cross-timeline synchronisation point'''

assert old_cfg in content, "EDIT 3 FAILED: CompiledFrameGraph end not found"
content = content.replace(old_cfg, new_cfg, 1)
changes += 1
print(f"Edit 3 OK: added errors field to CompiledFrameGraph")

# ---- Edit 4: Add memory savings computation + errors to compile() return ----
old_compile_ret = '''        Ok(CompiledFrameGraph {
            passes,
            resources,
            edges,
            order: order.to_vec(),
            depths,
            barriers,
            scheduled_passes,
            async_passes,
            sync_points,
            eliminated_passes: eliminated,
            cull_stats,
            parallel_regions,
            interference_graph,
        })'''
new_compile_ret = '''        // Compute memory savings percentage for T-FG-7.7.
        let total_resource_bytes: u64 = resources.iter().map(|r| r.desc.estimated_bytes()).sum();
        let alias_bytes_saved: u64 = 0; // Placeholder until Phase 3b aliasing is implemented.
        let memory_savings_percent = if total_resource_bytes > 0 {
            let total_saved = (cull_stats.bytes_saved as u64) + alias_bytes_saved;
            (total_saved as f64 / total_resource_bytes as f64) * 100.0
        } else {
            0.0
        };

        Ok(CompiledFrameGraph {
            passes,
            resources,
            edges,
            order: order.to_vec(),
            depths,
            barriers,
            scheduled_passes,
            async_passes,
            sync_points,
            eliminated_passes: eliminated,
            cull_stats,
            parallel_regions,
            interference_graph,
            errors: vec![],
        })'''

assert old_compile_ret in content, "EDIT 4 FAILED: compile return not found"
content = content.replace(old_compile_ret, new_compile_ret, 1)
changes += 1
print(f"Edit 4 OK: updated compile() return with memory savings + errors")

# ---- Edit 5: Add memory_savings_percent to cull_stats JSON ----
old_cull_json = '''        let cull_stats = serde_json::json!({
            "passes_total": self.cull_stats.passes_total,
            "passes_eliminated": self.cull_stats.passes_eliminated,
            "resources_freed": self.cull_stats.resources_freed,
            "bytes_saved": self.cull_stats.bytes_saved,
            "live_pass_count": self.cull_stats.live_pass_count,
            "culled_pass_count": self.cull_stats.culled_pass_count,
            "estimated_gpu_time_saved_ms": self.cull_stats.estimated_gpu_time_saved_ms,
        });'''
new_cull_json = '''        let cull_stats = serde_json::json!({
            "passes_total": self.cull_stats.passes_total,
            "passes_eliminated": self.cull_stats.passes_eliminated,
            "resources_freed": self.cull_stats.resources_freed,
            "bytes_saved": self.cull_stats.bytes_saved,
            "live_pass_count": self.cull_stats.live_pass_count,
            "culled_pass_count": self.cull_stats.culled_pass_count,
            "estimated_gpu_time_saved_ms": self.cull_stats.estimated_gpu_time_saved_ms,
            "memory_savings_percent": self.cull_stats.memory_savings_percent,
        });'''

assert old_cull_json in content, "EDIT 5 FAILED: cull_stats JSON not found"
content = content.replace(old_cull_json, new_cull_json, 1)
changes += 1
print(f"Edit 5 OK: added memory_savings_percent to JSON")

# ---- Edit 6: Add errors and memory_savings_percent to main JSON output ----
old_main_json = '''        serde_json::json!({
            "passes": passes,
            "resources": resources,
            "barriers": barriers,
            "async_passes": async_passes,
            "parallel_regions": parallel_regions,
            "depths": depths,
            "cull_stats": cull_stats,
            "validation": validation,
        })'''
new_main_json = '''        serde_json::json!({
            "passes": passes,
            "resources": resources,
            "barriers": barriers,
            "async_passes": async_passes,
            "parallel_regions": parallel_regions,
            "depths": depths,
            "cull_stats": cull_stats,
            "validation": validation,
            "errors": self.errors,
            "memory_savings_percent": self.cull_stats.memory_savings_percent,
        })'''

assert old_main_json in content, "EDIT 6 FAILED: main JSON not found"
content = content.replace(old_main_json, new_main_json, 1)
changes += 1
print(f"Edit 6 OK: added errors and memory_savings_percent to main JSON")

# ---- Edit 7: Update test CullStats literal ----
old_test = '''        let stats = CullStats {
            passes_total: 10,
            passes_eliminated: 3,
            resources_freed: 5,
            bytes_saved: 65536,
            live_pass_count: 7,
            culled_pass_count: 3,
            estimated_gpu_time_saved_ms: 1.5,
        };'''
new_test = '''        let stats = CullStats {
            passes_total: 10,
            passes_eliminated: 3,
            resources_freed: 5,
            bytes_saved: 65536,
            live_pass_count: 7,
            culled_pass_count: 3,
            estimated_gpu_time_saved_ms: 1.5,
            memory_savings_percent: 0.0,
        };'''

assert old_test in content, "EDIT 7 FAILED: test CullStats literal not found"
content = content.replace(old_test, new_test, 1)
changes += 1
print(f"Edit 7 OK: updated test CullStats literal")

# Write back atomically
with open(FILE, "w") as f:
    f.write(content)

print(f"\nAll {changes} edits applied successfully!")
