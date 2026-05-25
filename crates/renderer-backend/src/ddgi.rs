//! Dynamic Diffuse Global Illumination (DDGI) pass builders for the TRINITY
//! frame graph.
//!
//! Provides factory functions that construct [`IrPass`] instances for
//! DDGI probe-volume update and sample passes:
//!
//! - **Update** — traces rays from each probe to gather irradiance and depth.
//! - **Sample** — interpolates probe data at shading points to produce
//!   indirect diffuse lighting.

use crate::frame_graph::{DispatchSource, IrPass, PassIndex, ResourceHandle, ViewType};

// ---------------------------------------------------------------------------
// DDGI probe volume descriptor
// ---------------------------------------------------------------------------

/// CPU-side descriptor for a DDGI probe volume.
///
/// The volume is a 3D grid of probes, each storing a ray-traced irradiance
/// and depth history.  Probes are placed at regular intervals within the
/// axis-aligned box defined by `origin` and `extents`.
#[derive(Clone, Debug, PartialEq)]
pub struct DDGIProbeVolume {
    /// World-space origin of the volume (minimum corner).
    pub origin: [f32; 3],
    /// Extents (width, height, depth) of the volume in world space.
    pub extents: [f32; 3],
    /// Number of probes along each axis (X, Y, Z).
    pub probe_count: [u32; 3],
    /// Spacing between adjacent probes (used by the shader to compute
    /// world-space probe positions).
    pub probe_spacing: f32,
}

impl DDGIProbeVolume {
    /// Total number of probes in the volume.
    pub fn total_probes(&self) -> u32 {
        self.probe_count[0] * self.probe_count[1] * self.probe_count[2]
    }
}

impl Default for DDGIProbeVolume {
    fn default() -> Self {
        Self {
            origin: [0.0, 0.0, 0.0],
            extents: [20.0, 10.0, 20.0],
            probe_count: [8, 4, 8],
            probe_spacing: 2.5,
        }
    }
}

// ---------------------------------------------------------------------------
// DDGI update pass
// ---------------------------------------------------------------------------

/// Build a compute pass that updates probe irradiance and depth.
///
/// For each probe in the volume, one workgroup traces a set of rays and
/// accumulates the result into `irradiance` and `depth` textures.
///
/// `irradiance` — 3D texture (probe_count × 6 faces) storing accumulated
///                irradiance colour per probe.
/// `depth`      — 3D texture storing average hit distance per probe.
pub fn create_ddgi_update_pass(
    index: PassIndex,
    volume: &DDGIProbeVolume,
    irradiance: ResourceHandle,
    depth: ResourceHandle,
) -> IrPass {
    let total = volume.total_probes();
    let mut pass = IrPass::compute(
        index,
        "ddgi_update",
        DispatchSource::Direct {
            group_count_x: total,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.writes.push(irradiance);
    pass.access_set.writes.push(depth);
    pass.tags.push("gi".into());
    pass.tags.push("ddgi".into());
    pass
}

// ---------------------------------------------------------------------------
// DDGI sample pass
// ---------------------------------------------------------------------------

/// Build a compute pass that samples DDGI probes at every shading point.
///
/// Reads the `irradiance` probe volume and writes the indirect diffuse
/// contribution into `output`.
pub fn create_ddgi_sample_pass(
    index: PassIndex,
    irradiance: ResourceHandle,
    output: ResourceHandle,
) -> IrPass {
    let mut pass = IrPass::compute(
        index,
        "ddgi_sample",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.reads.push(irradiance);
    pass.access_set.writes.push(output);
    pass.tags.push("gi".into());
    pass.tags.push("ddgi".into());
    pass
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::PassType;

    // ── DDGIProbeVolume ─────────────────────────────────────────────────

    #[test]
    fn test_ddgi_probe_volume_default() {
        let vol = DDGIProbeVolume::default();
        assert_eq!(vol.origin, [0.0, 0.0, 0.0]);
        assert_eq!(vol.extents, [20.0, 10.0, 20.0]);
        assert_eq!(vol.probe_count, [8, 4, 8]);
        assert!((vol.probe_spacing - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ddgi_probe_volume_total_probes_default() {
        let vol = DDGIProbeVolume::default();
        assert_eq!(vol.total_probes(), 8 * 4 * 8);
    }

    #[test]
    fn test_ddgi_probe_volume_total_probes_custom() {
        let vol = DDGIProbeVolume {
            probe_count: [16, 8, 16],
            ..Default::default()
        };
        assert_eq!(vol.total_probes(), 16 * 8 * 16);
    }

    #[test]
    fn test_ddgi_probe_volume_non_uniform_counts() {
        let vol = DDGIProbeVolume {
            probe_count: [7, 3, 11],
            ..Default::default()
        };
        assert_eq!(vol.total_probes(), 7 * 3 * 11);
    }

    #[test]
    fn test_ddgi_probe_volume_single_probe() {
        let vol = DDGIProbeVolume {
            probe_count: [1, 1, 1],
            ..Default::default()
        };
        assert_eq!(vol.total_probes(), 1);
    }

    #[test]
    fn test_ddgi_probe_volume_custom_origin() {
        let vol = DDGIProbeVolume {
            origin: [-10.0, 0.0, -10.0],
            ..Default::default()
        };
        assert_eq!(vol.origin, [-10.0, 0.0, -10.0]);
    }

    #[test]
    fn test_ddgi_probe_volume_custom_spacing() {
        let vol = DDGIProbeVolume {
            probe_spacing: 1.0,
            ..Default::default()
        };
        assert!((vol.probe_spacing - 1.0).abs() < f32::EPSILON);
    }

    // ── Update pass ─────────────────────────────────────────────────────

    #[test]
    fn test_ddgi_update_pass_type() {
        let vol = DDGIProbeVolume::default();
        let pass = create_ddgi_update_pass(PassIndex(0), &vol, ResourceHandle(1), ResourceHandle(2));
        assert_eq!(pass.pass_type, PassType::Compute);
        assert_eq!(pass.name, "ddgi_update");
    }

    #[test]
    fn test_ddgi_update_pass_writes_irradiance_and_depth() {
        let vol = DDGIProbeVolume::default();
        let pass = create_ddgi_update_pass(PassIndex(0), &vol, ResourceHandle(1), ResourceHandle(2));
        assert!(pass.access_set.writes.contains(&ResourceHandle(1)), "must write irradiance");
        assert!(pass.access_set.writes.contains(&ResourceHandle(2)), "must write depth");
    }

    #[test]
    fn test_ddgi_update_pass_no_reads() {
        let vol = DDGIProbeVolume::default();
        let pass = create_ddgi_update_pass(PassIndex(0), &vol, ResourceHandle(1), ResourceHandle(2));
        assert!(
            pass.access_set.reads.is_empty(),
            "update pass should not read any resources (writes only)"
        );
    }

    #[test]
    fn test_ddgi_update_pass_dispatch_matches_probe_count() {
        let vol = DDGIProbeVolume {
            probe_count: [16, 8, 16],
            ..Default::default()
        };
        let pass = create_ddgi_update_pass(PassIndex(0), &vol, ResourceHandle(1), ResourceHandle(2));
        if let Some(DispatchSource::Direct { group_count_x, .. }) = pass.dispatch_source {
            assert_eq!(group_count_x, 16 * 8 * 16);
        } else {
            panic!("expected direct dispatch source");
        }
    }

    #[test]
    fn test_ddgi_update_pass_tags() {
        let vol = DDGIProbeVolume::default();
        let pass = create_ddgi_update_pass(PassIndex(0), &vol, ResourceHandle(1), ResourceHandle(2));
        assert!(pass.tags.contains(&"gi".into()));
        assert!(pass.tags.contains(&"ddgi".into()));
    }

    // ── Sample pass ─────────────────────────────────────────────────────

    #[test]
    fn test_ddgi_sample_pass_type() {
        let pass = create_ddgi_sample_pass(PassIndex(1), ResourceHandle(1), ResourceHandle(2));
        assert_eq!(pass.pass_type, PassType::Compute);
        assert_eq!(pass.name, "ddgi_sample");
    }

    #[test]
    fn test_ddgi_sample_pass_reads_irradiance_writes_output() {
        let pass = create_ddgi_sample_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert!(pass.access_set.reads.contains(&ResourceHandle(1)), "must read irradiance volume");
        assert!(pass.access_set.writes.contains(&ResourceHandle(2)), "must write output");
    }

    #[test]
    fn test_ddgi_sample_pass_tags() {
        let pass = create_ddgi_sample_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert!(pass.tags.contains(&"gi".into()));
        assert!(pass.tags.contains(&"ddgi".into()));
    }

    // ── Pipeline integration ────────────────────────────────────────────

    #[test]
    fn test_ddgi_pipeline_pass_indices() {
        let vol = DDGIProbeVolume::default();
        let irr = ResourceHandle(10);
        let dep = ResourceHandle(11);
        let out = ResourceHandle(12);

        let update = create_ddgi_update_pass(PassIndex(0), &vol, irr, dep);
        let sample = create_ddgi_sample_pass(PassIndex(1), irr, out);

        assert_eq!(update.index, PassIndex(0));
        assert_eq!(sample.index, PassIndex(1));
    }

    #[test]
    fn test_ddgi_pipeline_irradiance_shared() {
        let vol = DDGIProbeVolume::default();
        let irr = ResourceHandle(10);

        let update = create_ddgi_update_pass(PassIndex(0), &vol, irr, ResourceHandle(11));
        let sample = create_ddgi_sample_pass(PassIndex(1), irr, ResourceHandle(12));

        assert!(update.access_set.writes.contains(&irr), "update must write irradiance");
        assert!(sample.access_set.reads.contains(&irr), "sample must read irradiance");
    }

    #[test]
    fn test_ddgi_update_pass_view_type() {
        let vol = DDGIProbeVolume::default();
        let pass = create_ddgi_update_pass(PassIndex(0), &vol, ResourceHandle(1), ResourceHandle(2));
        assert_eq!(pass.view_type, ViewType::Storage);
    }

    #[test]
    fn test_ddgi_sample_pass_view_type() {
        let pass = create_ddgi_sample_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert_eq!(pass.view_type, ViewType::Storage);
    }
}
