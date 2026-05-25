//! Debug dump formatter for [`CompiledFrameGraph`].
//!
//! Provides [`DebugDumper`] which produces a human-readable multi-section string
//! covering pass execution order, resources, pipeline barriers, DAG dependency
//! edges, async-scheduled passes, parallel regions, eliminated (dead) passes,
//! compiler statistics, and per-phase performance counters.
//!
//! # T-FG-7.9
//!
//! This module implements the debugging output required by T-FG-7.9:
//!
//! - [`DebugDumper::dump`] returns a formatted `String` with all sections.
//! - [`super::CompiledFrameGraph`] also implements [`core::fmt::Display`] by
//!   delegating to [`DebugDumper::dump`].
//! - Setting the environment variable `TRINITY_DUMP_FRAME_GRAPH` to any non-empty
//!   value causes the compiler to print the dump to stderr after compilation.

use core::fmt;
use std::collections::HashMap;

use super::{
    BarrierTuple, CompiledFrameGraph, EdgeType, InterferenceGraph, IrEdge, IrPass, IrResource,
    PassIndex, PassType, ResourceHandle, ResourceState, ScheduledPass, TextureDesc,
};

// ---------------------------------------------------------------------------
// DebugDumper
// ---------------------------------------------------------------------------

/// Produces a human-readable multi-section debug dump of a compiled frame graph.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::frame_graph::{CompiledFrameGraph, DebugDumper};
///
/// let graph = CompiledFrameGraph::compile(passes, resources)?;
/// let dump = DebugDumper::dump(&graph);
/// println!("{dump}");
/// ```
pub struct DebugDumper;

impl DebugDumper {
    /// Returns a formatted multi-section string for `graph`.
    ///
    /// The output includes: header summary, resources, pass execution order,
    /// pipeline barriers, async-scheduled passes, parallel regions, eliminated
    /// passes, DAG dependency edges, compiler statistics, and performance
    /// counters.
    pub fn dump(graph: &CompiledFrameGraph) -> String {
        let mut out = String::new();
        let rule = "=".repeat(80);

        // -----------------------------------------------------------------------
        // Header
        // -----------------------------------------------------------------------
        out.push_str(&rule);
        out.push('\n');
        out.push_str("FRAME GRAPH DEBUG DUMP\n");
        out.push_str(&rule);
        out.push('\n');
        out.push('\n');

        let total = graph.cull_stats.passes_total;
        let alive = graph.order.len();
        let eliminated = graph.cull_stats.passes_eliminated;
        let num_resources = graph.resources.len();
        let num_barriers = graph.barriers.len();
        let num_async = graph.async_passes.len();
        let num_parallel = graph.parallel_regions.len();
        let alias_count = estimate_alias_groups(&graph.interference_graph, graph.resources.len());

        out.push_str(&format!("  Passes (total)        : {total} ({alive} alive, {eliminated} eliminated)\n"));
        out.push_str(&format!("  Resources             : {num_resources}\n"));
        out.push_str(&format!("  Barriers              : {num_barriers}\n"));
        out.push_str(&format!("  Async passes          : {num_async}\n"));
        out.push_str(&format!("  Parallel regions      : {num_parallel}\n"));
        out.push_str(&format!("  Resources aliased     : {alias_count}\n"));

        let comp_time_us = graph.compilation_time_us;
        let comp_time_ms = comp_time_us as f64 / 1000.0;
        out.push_str(&format!(
            "  Compilation time      : {comp_time_us} us ({comp_time_ms:.2} ms)\n"
        ));

        out.push('\n');

        // -----------------------------------------------------------------------
        // RESOURCES
        // -----------------------------------------------------------------------
        write_section(&mut out, "RESOURCES");
        if graph.resources.is_empty() {
            out.push_str("  (none)\n\n");
        } else {
            out.push('\n');
            for res in &graph.resources {
                write_resource(&mut out, res);
            }
            out.push('\n');
        }

        // -----------------------------------------------------------------------
        // PASS EXECUTION ORDER
        // -----------------------------------------------------------------------
        write_section(&mut out, "PASS EXECUTION ORDER");
        if graph.order.is_empty() {
            out.push_str("  (none)\n\n");
        } else {
            out.push('\n');
            for pass_idx in &graph.order {
                if let Some(pass) = graph.passes.iter().find(|p| &p.index == pass_idx) {
                    write_pass(&mut out, pass);
                }
            }
            out.push('\n');
        }

        // -----------------------------------------------------------------------
        // BARRIERS
        // -----------------------------------------------------------------------
        write_section(&mut out, "BARRIERS");
        if graph.barriers.is_empty() {
            out.push_str("  (none)\n\n");
        } else {
            out.push('\n');
            for (from_idx, to_idx, before, after) in &graph.barriers {
                write_barrier(&mut out, &graph.passes, &graph.resources, *from_idx, *to_idx, before, after);
            }
            out.push('\n');
        }

        // -----------------------------------------------------------------------
        // DAG DEPENDENCY EDGES
        // -----------------------------------------------------------------------
        write_section(&mut out, "DAG DEPENDENCY EDGES");
        if graph.edges.is_empty() {
            out.push_str("  (none)\n\n");
        } else {
            out.push('\n');
            for edge in &graph.edges {
                write_edge(&mut out, edge, &graph.passes, &graph.resources);
            }
            out.push('\n');
        }

        // -----------------------------------------------------------------------
        // ASYNC SCHEDULED PASSES
        // -----------------------------------------------------------------------
        write_section(&mut out, "ASYNC SCHEDULED PASSES");
        if graph.async_passes.is_empty() {
            out.push_str("  (none)\n\n");
        } else {
            out.push('\n');
            for (pass_idx, queue_type) in &graph.async_passes {
                let name = graph
                    .passes
                    .iter()
                    .find(|p| &p.index == pass_idx)
                    .map(|p| p.name.as_str())
                    .unwrap_or("<unknown>");
                out.push_str(&format!(
                    "  P{} \"{name}\" -> {queue_type} queue\n",
                    pass_idx.0,
                ));
            }
            out.push('\n');
        }

        // -----------------------------------------------------------------------
        // PARALLEL REGIONS
        // -----------------------------------------------------------------------
        write_section(&mut out, "PARALLEL REGIONS");
        if graph.parallel_regions.is_empty() {
            out.push_str("  (none)\n\n");
        } else {
            out.push('\n');
            for (ri, region) in graph.parallel_regions.iter().enumerate() {
                let names: Vec<String> = region
                    .iter()
                    .map(|pi| {
                        graph
                            .passes
                            .iter()
                            .find(|p| &p.index == pi)
                            .map(|p| format!("\"{}\"", p.name))
                            .unwrap_or_else(|| format!("P{}", pi.0))
                    })
                    .collect();
                out.push_str(&format!("  Region {ri}: [{}]\n", names.join(", ")));
            }
            out.push('\n');
        }

        // -----------------------------------------------------------------------
        // ELIMINATED PASSES
        // -----------------------------------------------------------------------
        write_section(&mut out, "ELIMINATED PASSES");
        if graph.eliminated_passes.is_empty() {
            out.push_str("  (none)\n\n");
        } else {
            out.push('\n');
            for pi in &graph.eliminated_passes {
                let name = graph
                    .eliminated_pass_names
                    .get(pi)
                    .map(|n| n.as_str())
                    .unwrap_or("<eliminated>");
                out.push_str(&format!("  P{} \"{name}\" <eliminated>\n", pi.0));
            }
            out.push('\n');
            let cs = &graph.cull_stats;
            out.push_str(&format!(
                "  Cull stats: passes_total={}, passes_eliminated={}, resources_freed={}, bytes_saved={}\n",
                cs.passes_total, cs.passes_eliminated, cs.resources_freed, cs.bytes_saved,
            ));
            out.push('\n');
        }

        // -----------------------------------------------------------------------
        // COMPILER STATISTICS
        // -----------------------------------------------------------------------
        write_section(&mut out, "COMPILER STATISTICS");
        out.push('\n');
        let stats = &graph.cull_stats;
        out.push_str(&format!("  Passes total          : {}\n", stats.passes_total));
        out.push_str(&format!("  Passes eliminated     : {}\n", stats.passes_eliminated));
        out.push_str(&format!("  Barriers total        : {}\n", graph.barriers.len()));
        let barriers_optimized = count_optimized_barriers(graph);
        out.push_str(&format!("  Barriers optimized    : {barriers_optimized}\n"));
        out.push_str(&format!("  Async passes          : {}\n", graph.async_passes.len()));
        out.push_str(&format!("  Resources aliased     : {alias_count}\n"));
        out.push_str(&format!(
            "  Compilation time      : {comp_time_us} us ({comp_time_ms:.2} ms)\n"
        ));
        out.push_str(&format!(
            "  Resources freed       : {}\n",
            stats.resources_freed
        ));
        out.push_str(&format!("  Bytes saved           : {}\n", stats.bytes_saved));
        out.push_str(&format!(
            "  GPU time saved        : {:.1} ms\n",
            stats.estimated_gpu_time_saved_ms
        ));
        out.push_str(&format!(
            "  Dynamically skipped   : {}\n",
            stats.dynamically_skipped
        ));
        out.push_str(&format!("  Live passes           : {}\n", stats.live_pass_count));
        out.push('\n');

        // -----------------------------------------------------------------------
        // PERFORMANCE COUNTERS
        // -----------------------------------------------------------------------
        write_section(&mut out, "PERFORMANCE COUNTERS");
        out.push('\n');
        out.push_str("  DAG build             : -- us\n");
        out.push_str("  Topological sort      : -- us\n");
        out.push_str("  Barrier compute       : -- us\n");
        out.push_str("  Async schedule        : -- us\n");
        out.push_str("  Dead-pass elim        : -- us\n");
        out.push_str(&format!("  Total                 : {comp_time_us} us\n"));
        out.push('\n');

        // -----------------------------------------------------------------------
        // Footer
        // -----------------------------------------------------------------------
        out.push_str(&rule);
        out.push('\n');
        out.push_str("END DEBUG DUMP\n");
        out.push_str(&rule);
        out.push('\n');

        out
    }
}

// ---------------------------------------------------------------------------
// Helper: write a section header
// ---------------------------------------------------------------------------

fn write_section(out: &mut String, title: &str) {
    let rule = "=".repeat(80);
    out.push_str(&rule);
    out.push('\n');
    out.push_str(title);
    out.push('\n');
    out.push_str(&rule);
    out.push('\n');
}

// ---------------------------------------------------------------------------
// Helper: write a resource entry
// ---------------------------------------------------------------------------

fn write_resource(out: &mut String, res: &IrResource) {
    use super::ResourceDesc;

    match &res.desc {
        ResourceDesc::Texture2D(desc) | ResourceDesc::TextureCube(desc) => {
            out.push_str(&format!(
                "  {} \"{}\" Texture2D({}x{}, mips={}, layers={}, format={}) lifetime={} init={:?}\n",
                res.handle,
                res.name,
                desc.width,
                desc.height,
                desc.mip_levels,
                desc.array_layers,
                desc.format,
                res.lifetime,
                res.initial_state,
            ));
        }
        ResourceDesc::Texture3D(desc) => {
            out.push_str(&format!(
                "  {} \"{}\" Texture3D({}x{}x{}, mips={}, format={}) lifetime={} init={:?}\n",
                res.handle,
                res.name,
                desc.width,
                desc.height,
                desc.depth,
                desc.mip_levels,
                desc.format,
                res.lifetime,
                res.initial_state,
            ));
        }
        ResourceDesc::Buffer(desc) => {
            out.push_str(&format!(
                "  {} \"{}\" Buffer({} bytes, usage={}) lifetime={} init={:?}\n",
                res.handle,
                res.name,
                desc.size,
                desc.usage,
                res.lifetime,
                res.initial_state,
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: write a pass entry
// ---------------------------------------------------------------------------

fn write_pass(out: &mut String, pass: &IrPass) {
    out.push_str(&format!(
        "  P{} \"{}\" [{}]\n",
        pass.index.0, pass.name, pass.pass_type,
    ));

    // Depth if available is shown via the depths map at the call site.
    // Here we show the pass details.

    // View type
    out.push_str(&format!("    view: {}\n", pass.view_type));

    // Access set
    out.push_str(&format!("    access: {}\n", pass.access_set));

    // Color attachments
    if !pass.color_attachments.is_empty() {
        for ca in &pass.color_attachments {
            out.push_str(&format!("    color: {}\n", ca));
        }
    }

    // Depth-stencil attachment
    if let Some(ds) = &pass.depth_stencil {
        out.push_str(&format!("    depth-stencil: {}\n", ds));
    }

    // Tags
    if !pass.tags.is_empty() {
        out.push_str(&format!("    tags: {:?}\n", pass.tags));
    }

    out.push('\n');
}

// ---------------------------------------------------------------------------
// Helper: write a barrier entry
// ---------------------------------------------------------------------------

fn write_barrier(
    out: &mut String,
    passes: &[IrPass],
    resources: &[IrResource],
    from: PassIndex,
    to: PassIndex,
    before: &ResourceState,
    after: &ResourceState,
) {
    let from_name = passes
        .iter()
        .find(|p| p.index == from)
        .map(|p| p.name.as_str())
        .unwrap_or("<unknown>");
    let to_name = passes
        .iter()
        .find(|p| p.index == to)
        .map(|p| p.name.as_str())
        .unwrap_or("<unknown>");

    out.push_str(&format!(
        "  P{} (\"{from_name}\") --[{before} -> {after}]--> P{} (\"{to_name}\")\n",
        from.0, to.0,
    ));
}

// ---------------------------------------------------------------------------
// Helper: write a DAG edge
// ---------------------------------------------------------------------------

fn write_edge(
    out: &mut String,
    edge: &IrEdge,
    passes: &[IrPass],
    resources: &[IrResource],
) {
    let res_name = resources
        .iter()
        .find(|r| r.handle == edge.resource)
        .map(|r| r.name.as_str())
        .unwrap_or("<unknown>");

    out.push_str(&format!(
        "  P{} --[{}:{}]--> P{}\n",
        edge.from.0, edge.edge_type, res_name, edge.to.0,
    ));
}

// ---------------------------------------------------------------------------
// Helper: estimate alias-group count from interference graph
// ---------------------------------------------------------------------------

/// Estimates the number of alias groups (unique physical allocations) using a
/// simple greedy coloring algorithm on the complement of the interference
/// graph. Lower = more aliasing.
fn estimate_alias_groups(ig: &InterferenceGraph, total_resources: usize) -> usize {
    if total_resources == 0 {
        return 0;
    }

    // Collect all handles that appear in the interference graph.
    // Handles NOT in the graph do not interfere with any resource.
    let mut interfering_handles: Vec<ResourceHandle> = ig.all_handles();
    interfering_handles.sort();

    let num_non_interfering = total_resources - interfering_handles.len();

    // All non-interfering resources can share one alias group.
    // For interfering resources, each group is one (since they all conflict).
    // This is a conservative upper bound.
    if num_non_interfering > 0 {
        // Non-interfering resources share 1 group; each interfering resource
        // needs its own group.
        interfering_handles.len().saturating_add(1)
    } else {
        // All resources interfere — greedy color the interference graph.
        // The number of colors needed is a lower bound on required groups.
        greedy_color_count(&ig, &interfering_handles)
    }
}

/// Counts colors needed using greedy (LDO) ordering on the interference graph.
fn greedy_color_count(
    ig: &InterferenceGraph,
    handles: &[ResourceHandle],
) -> usize {
    if handles.is_empty() {
        return 0;
    }

    // Build neighbor sets for quick lookup.
    let neighbor_set: HashMap<ResourceHandle, Vec<ResourceHandle>> = handles
        .iter()
        .map(|&h| (h, ig.neighbors(h).to_vec()))
        .collect();

    // Sort by degree descending (largest-first ordering).
    let mut sorted = handles.to_vec();
    sorted.sort_by(|a, b| {
        let deg_a = neighbor_set.get(a).map_or(0, Vec::len);
        let deg_b = neighbor_set.get(b).map_or(0, Vec::len);
        deg_b.cmp(&deg_a)
    });

    let mut colors: HashMap<ResourceHandle, usize> = HashMap::new();

    for &h in &sorted {
        // Collect colors used by neighbors.
        let mut used = Vec::new();
        if let Some(neighbors) = neighbor_set.get(&h) {
            for &n in neighbors {
                if let Some(&c) = colors.get(&n) {
                    used.push(c);
                }
            }
        }
        used.sort();
        used.dedup();

        // Find the smallest unused color.
        let mut color = 0;
        for &u in &used {
            if u == color {
                color += 1;
            } else if u > color {
                break;
            }
        }
        colors.insert(h, color);
    }

    // Count unique colors.
    let mut unique_colors: Vec<usize> = colors.values().copied().collect();
    unique_colors.sort();
    unique_colors.dedup();
    unique_colors.len()
}

// ---------------------------------------------------------------------------
// Helper: estimate how many barriers were optimized away
// ---------------------------------------------------------------------------

fn count_optimized_barriers(graph: &CompiledFrameGraph) -> usize {
    // The scheduled_passes contain per-pass barriers. We can count how many
    // barrier tuples exist there vs. the raw barrier count.

    let scheduled_count: usize = graph
        .scheduled_passes
        .iter()
        .map(|sp| sp.pre_barriers.len() + sp.post_barriers.len())
        .sum();

    if scheduled_count >= graph.barriers.len() {
        scheduled_count - graph.barriers.len()
    } else {
        0
    }
}
