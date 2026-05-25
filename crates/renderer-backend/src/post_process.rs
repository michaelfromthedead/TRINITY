//! Post-processing pass builders for the TRINITY frame graph.
//!
//! Provides factory functions that construct [`IrPass`] and [`IrResource`]
//! instances for common post-processing effects: ACES tonemapping, Gaussian
//! bloom, and temporal anti-aliasing (TAA).  The [`create_post_process_chain`]
//! helper wires the three effects together with transient intermediate
//! resources.

use crate::frame_graph::{
    DispatchSource, IrPass, IrResource, PassIndex, ResourceDesc, ResourceHandle,
    ResourceLifetime, ResourceState, TextureDesc, ViewType,
};

// ---------------------------------------------------------------------------
// Tonemapping (ACES)
// ---------------------------------------------------------------------------

/// Build an IR pass that applies ACES filmic tonemapping.
///
/// `input`  — the HDR colour resource to be tonemapped.
/// `output` — the LDR (tone-mapped) output resource.
///
/// The pass type is `Compute`.  Dispatch dimensions are conservative (1×1×1);
/// the backend should re-derive them from the output resolution at recording
/// time.
pub fn create_tonemap_pass(
    index: PassIndex,
    input: ResourceHandle,
    output: ResourceHandle,
) -> IrPass {
    let mut pass = IrPass::compute(
        index,
        "tonemap",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.reads.push(input);
    pass.access_set.writes.push(output);
    pass.tags.push("post-process".into());
    pass.tags.push("tonemap".into());
    pass
}

// ---------------------------------------------------------------------------
// Bloom (Gaussian pyramid)
// ---------------------------------------------------------------------------

/// Build an IR pass that applies a Gaussian bloom (downsample + upsample).
///
/// `input`  — the source texture.
/// `output` — the bloom-composite output.
/// `width`  / `height` — the full-resolution size used to derive dispatch
/// counts for the downsample / upsample chain.
pub fn create_bloom_pass(
    index: PassIndex,
    input: ResourceHandle,
    output: ResourceHandle,
    width: u32,
    height: u32,
) -> IrPass {
    let mut pass = IrPass::compute(
        index,
        "bloom",
        DispatchSource::Direct {
            group_count_x: (width / 8).max(1),
            group_count_y: (height / 8).max(1),
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.reads.push(input);
    pass.access_set.writes.push(output);
    pass.tags.push("post-process".into());
    pass.tags.push("bloom".into());
    pass
}

// ---------------------------------------------------------------------------
// Temporal Anti-Aliasing (TAA)
// ---------------------------------------------------------------------------

/// Build an IR pass that applies temporal anti-aliasing.
///
/// `input`   — the current-frame colour input.
/// `history` — the previous-frame colour (read + written back for the next
///             frame).
/// `output`  — the TAA-resolved output.
///
/// The history buffer is both read and written so it can be updated in place
/// for subsequent frames.
pub fn create_taa_pass(
    index: PassIndex,
    input: ResourceHandle,
    history: ResourceHandle,
    output: ResourceHandle,
) -> IrPass {
    let mut pass = IrPass::compute(
        index,
        "taa",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.reads.push(input);
    pass.access_set.reads.push(history);
    pass.access_set.writes.push(output);
    pass.access_set.writes.push(history); // history is updated for the next frame
    pass.tags.push("post-process".into());
    pass.tags.push("taa".into());
    pass
}

// ---------------------------------------------------------------------------
// Full post-process chain
// ---------------------------------------------------------------------------

/// Build the full post-process pipeline: **tonemap → bloom → TAA**.
///
/// Returns a `(Vec<IrPass>, Vec<IrResource>)` tuple containing three passes
/// and three transient intermediate resources:
///
/// | Index | Pass     | Reads                  | Writes                     |
/// |-------|----------|------------------------|----------------------------|
/// | 0     | Tonemap  | `hdr_input`            | intermediate (1)           |
/// | 1     | Bloom    | intermediate (1)       | intermediate (2)           |
/// | 2     | TAA      | intermediate (2) + history | `ldr_output` + history |
///
/// The three transient resources are:
///
/// 1. `tonemap_output`  — rg16f, full-res, after ACES.
/// 2. `bloom_output`    — rg16f, full-res, after bloom composite.
/// 3. `taa_history`     — rg16f, full-res, accumulated TAA history.
pub fn create_post_process_chain(
    start_index: PassIndex,
    hdr_input: ResourceHandle,
    ldr_output: ResourceHandle,
    width: u32,
    height: u32,
) -> (Vec<IrPass>, Vec<IrResource>) {
    let mut resources = Vec::new();
    let mut passes = Vec::new();

    // ── Offsets for the three transient resources ───────────────────────
    // Use a reserved range so they do not collide with application handles.
    const TONEMAP_OUT: ResourceHandle = ResourceHandle(0xFF00);
    const BLOOM_OUT: ResourceHandle = ResourceHandle(0xFF01);
    const TAA_HISTORY: ResourceHandle = ResourceHandle(0xFF02);

    // ── Transient intermediate resources ───────────────────────────────
    let tex_desc = TextureDesc {
        width,
        height,
        mip_levels: 1,
        array_layers: 1,
        format: "rgba16float".into(),
    };

    resources.push(IrResource::new(
        TONEMAP_OUT,
        "tonemap_output",
        ResourceDesc::Texture2D(tex_desc.clone()),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    ));

    resources.push(IrResource::new(
        BLOOM_OUT,
        "bloom_output",
        ResourceDesc::Texture2D(tex_desc.clone()),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    ));

    resources.push(IrResource::new(
        TAA_HISTORY,
        "taa_history",
        ResourceDesc::Texture2D(tex_desc),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    ));

    // ── Passes ─────────────────────────────────────────────────────────
    passes.push(create_tonemap_pass(
        PassIndex(start_index.0),
        hdr_input,
        TONEMAP_OUT,
    ));

    passes.push(create_bloom_pass(
        PassIndex(start_index.0 + 1),
        TONEMAP_OUT,
        BLOOM_OUT,
        width,
        height,
    ));

    passes.push(create_taa_pass(
        PassIndex(start_index.0 + 2),
        BLOOM_OUT,
        TAA_HISTORY,
        ldr_output,
    ));

    (passes, resources)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::PassType;

    // ── Tonemap ─────────────────────────────────────────────────────────

    #[test]
    fn test_tonemap_pass_type() {
        let pass = create_tonemap_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert_eq!(pass.pass_type, PassType::Compute);
        assert_eq!(pass.name, "tonemap");
    }

    #[test]
    fn test_tonemap_pass_access_set() {
        let pass = create_tonemap_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert!(
            pass.access_set.reads.contains(&ResourceHandle(1)),
            "tonemap must read the HDR input"
        );
        assert!(
            pass.access_set.writes.contains(&ResourceHandle(2)),
            "tonemap must write the LDR output"
        );
    }

    #[test]
    fn test_tonemap_pass_tags() {
        let pass = create_tonemap_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert!(pass.tags.contains(&"post-process".into()));
        assert!(pass.tags.contains(&"tonemap".into()));
    }

    // ── Bloom ──────────────────────────────────────────────────────────

    #[test]
    fn test_bloom_pass_type() {
        let pass = create_bloom_pass(PassIndex(1), ResourceHandle(2), ResourceHandle(3), 1920, 1080);
        assert_eq!(pass.pass_type, PassType::Compute);
        assert_eq!(pass.name, "bloom");
    }

    #[test]
    fn test_bloom_pass_access_set() {
        let pass = create_bloom_pass(PassIndex(1), ResourceHandle(2), ResourceHandle(3), 1920, 1080);
        assert!(pass.access_set.reads.contains(&ResourceHandle(2)));
        assert!(pass.access_set.writes.contains(&ResourceHandle(3)));
    }

    #[test]
    fn test_bloom_pass_tags() {
        let pass = create_bloom_pass(PassIndex(1), ResourceHandle(2), ResourceHandle(3), 1920, 1080);
        assert!(pass.tags.contains(&"bloom".into()));
        assert!(pass.tags.contains(&"post-process".into()));
    }

    #[test]
    fn test_bloom_pass_dispatch_derived_from_resolution() {
        let pass = create_bloom_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2), 1920, 1080);
        if let Some(DispatchSource::Direct { group_count_x, group_count_y, .. }) = pass.dispatch_source {
            assert_eq!(group_count_x, 1920 / 8);
            assert_eq!(group_count_y, 1080 / 8);
        } else {
            panic!("bloom pass must have a direct dispatch source");
        }
    }

    #[test]
    fn test_bloom_pass_dispatch_minimum_one() {
        // Very small resolution should still yield at least 1 workgroup.
        let pass = create_bloom_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2), 1, 1);
        if let Some(DispatchSource::Direct { group_count_x, group_count_y, .. }) = pass.dispatch_source {
            assert!(group_count_x >= 1);
            assert!(group_count_y >= 1);
        } else {
            panic!("bloom pass must have a direct dispatch source");
        }
    }

    // ── TAA ────────────────────────────────────────────────────────────

    #[test]
    fn test_taa_pass_type() {
        let pass = create_taa_pass(PassIndex(2), ResourceHandle(3), ResourceHandle(4), ResourceHandle(5));
        assert_eq!(pass.pass_type, PassType::Compute);
        assert_eq!(pass.name, "taa");
    }

    #[test]
    fn test_taa_pass_reads_input_and_history() {
        let pass = create_taa_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2), ResourceHandle(3));
        assert!(pass.access_set.reads.contains(&ResourceHandle(1)), "TAA must read the current frame input");
        assert!(pass.access_set.reads.contains(&ResourceHandle(2)), "TAA must read the history buffer");
    }

    #[test]
    fn test_taa_pass_writes_output_and_history() {
        let pass = create_taa_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2), ResourceHandle(3));
        assert!(pass.access_set.writes.contains(&ResourceHandle(3)), "TAA must write the resolved output");
        assert!(pass.access_set.writes.contains(&ResourceHandle(2)), "TAA must update the history buffer");
    }

    #[test]
    fn test_taa_pass_tags() {
        let pass = create_taa_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2), ResourceHandle(3));
        assert!(pass.tags.contains(&"taa".into()));
        assert!(pass.tags.contains(&"post-process".into()));
    }

    // ── Chain ──────────────────────────────────────────────────────────

    #[test]
    fn test_chain_returns_three_passes() {
        let (passes, _resources) =
            create_post_process_chain(PassIndex(0), ResourceHandle(10), ResourceHandle(20), 1920, 1080);
        assert_eq!(passes.len(), 3, "chain must produce three passes");
    }

    #[test]
    fn test_chain_returns_three_resources() {
        let (_passes, resources) =
            create_post_process_chain(PassIndex(0), ResourceHandle(10), ResourceHandle(20), 1920, 1080);
        assert_eq!(resources.len(), 3, "chain must produce three transient resources");
    }

    #[test]
    fn test_chain_passes_have_consecutive_indices() {
        let (passes, _) =
            create_post_process_chain(PassIndex(5), ResourceHandle(10), ResourceHandle(20), 1920, 1080);
        assert_eq!(passes[0].index, PassIndex(5));
        assert_eq!(passes[1].index, PassIndex(6));
        assert_eq!(passes[2].index, PassIndex(7));
    }

    #[test]
    fn test_chain_passes_names() {
        let (passes, _) =
            create_post_process_chain(PassIndex(0), ResourceHandle(10), ResourceHandle(20), 1920, 1080);
        assert_eq!(passes[0].name, "tonemap");
        assert_eq!(passes[1].name, "bloom");
        assert_eq!(passes[2].name, "taa");
    }

    #[test]
    fn test_chain_tonemap_reads_hdr_input() {
        let (passes, _) =
            create_post_process_chain(PassIndex(0), ResourceHandle(10), ResourceHandle(20), 1920, 1080);
        assert!(passes[0].access_set.reads.contains(&ResourceHandle(10)));
    }

    #[test]
    fn test_chain_taa_writes_ldr_output() {
        let (passes, _) =
            create_post_process_chain(PassIndex(0), ResourceHandle(10), ResourceHandle(20), 1920, 1080);
        assert!(passes[2].access_set.writes.contains(&ResourceHandle(20)));
    }

    #[test]
    fn test_chain_internal_wiring_tonemap_to_bloom() {
        // Tonemap's output should be bloom's input.
        let (passes, resources) =
            create_post_process_chain(PassIndex(0), ResourceHandle(10), ResourceHandle(20), 1920, 1080);

        // Find the tonemap output handle from the resources.
        let tonemap_out_h = resources
            .iter()
            .find(|r| r.name == "tonemap_output")
            .map(|r| r.handle)
            .expect("tonemap_output resource must exist");

        assert!(
            passes[0].access_set.writes.contains(&tonemap_out_h),
            "tonemap must write tonemap_output"
        );
        assert!(
            passes[1].access_set.reads.contains(&tonemap_out_h),
            "bloom must read tonemap_output"
        );
    }

    #[test]
    fn test_chain_internal_wiring_bloom_to_taa() {
        let (passes, resources) =
            create_post_process_chain(PassIndex(0), ResourceHandle(10), ResourceHandle(20), 1920, 1080);

        let bloom_out_h = resources
            .iter()
            .find(|r| r.name == "bloom_output")
            .map(|r| r.handle)
            .expect("bloom_output resource must exist");

        assert!(
            passes[1].access_set.writes.contains(&bloom_out_h),
            "bloom must write bloom_output"
        );
        assert!(
            passes[2].access_set.reads.contains(&bloom_out_h),
            "TAA must read bloom_output"
        );
    }

    #[test]
    fn test_chain_all_resources_are_transient() {
        let (_, resources) =
            create_post_process_chain(PassIndex(0), ResourceHandle(10), ResourceHandle(20), 1920, 1080);
        for r in &resources {
            assert_eq!(
                r.lifetime,
                ResourceLifetime::Transient,
                "all chain resources must be transient: {}",
                r.name
            );
        }
    }

    #[test]
    fn test_chain_resource_formats() {
        let (_, resources) =
            create_post_process_chain(PassIndex(0), ResourceHandle(10), ResourceHandle(20), 1920, 1080);
        for r in &resources {
            match &r.desc {
                ResourceDesc::Texture2D(desc) => {
                    assert_eq!(desc.format, "rgba16float");
                    assert_eq!(desc.mip_levels, 1);
                    assert_eq!(desc.array_layers, 1);
                }
                other => panic!("expected Texture2D, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_chain_zero_start_index() {
        let (passes, _) =
            create_post_process_chain(PassIndex(0), ResourceHandle(1), ResourceHandle(2), 800, 600);
        assert_eq!(passes[0].index, PassIndex(0));
        assert_eq!(passes[2].index, PassIndex(2));
    }

    #[test]
    fn test_chain_high_start_index() {
        let (passes, _) =
            create_post_process_chain(PassIndex(100), ResourceHandle(1), ResourceHandle(2), 800, 600);
        assert_eq!(passes[0].index, PassIndex(100));
        assert_eq!(passes[2].index, PassIndex(102));
    }

    #[test]
    fn test_chain_resource_resolution() {
        let (_, resources) =
            create_post_process_chain(PassIndex(0), ResourceHandle(1), ResourceHandle(2), 640, 480);
        for r in &resources {
            match &r.desc {
                ResourceDesc::Texture2D(desc) => {
                    assert_eq!(desc.width, 640);
                    assert_eq!(desc.height, 480);
                }
                _ => panic!("unexpected resource type"),
            }
        }
    }
}
