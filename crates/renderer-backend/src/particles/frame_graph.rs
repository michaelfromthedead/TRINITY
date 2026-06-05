//! Particle system pass builders for the TRINITY frame graph.
//!
//! Provides factory functions that construct [`IrPass`] instances for the
//! four phases of GPU-driven particle simulation:
//!
//! 1. **Spawn**  — emit new particles into a GPU particle buffer.
//! 2. **Update** — integrate velocity / lifetime per particle.
//! 3. **Render** — draw camera-facing quads at each particle position.
//! 4. **Compact** — remove dead particles (pack the buffer).

use crate::frame_graph::{
    AttachmentLoadOp, AttachmentStoreOp, ColorAttachment, DispatchSource, InstanceSource, IrPass,
    PassIndex, ResourceHandle, ViewType,
};

// ---------------------------------------------------------------------------
// Particle emitter descriptor
// ---------------------------------------------------------------------------

/// CPU-side descriptor for a GPU-driven particle emitter.
///
/// The emitter defines the spatial origin, emission rate, maximum number of
/// live particles, and per-particle lifetime.  These values are consumed by
/// the spawn compute shader at runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleEmitter {
    /// World-space position of the emitter origin.
    pub position: [f32; 3],
    /// Particles emitted per second.
    pub rate: f32,
    /// Maximum number of particles alive simultaneously.
    pub max_particles: u32,
    /// Per-particle lifetime in seconds.
    pub lifetime: f32,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rate: 100.0,
            max_particles: 4096,
            lifetime: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Spawn pass
// ---------------------------------------------------------------------------

/// Build a compute pass that spawns new particles into `particle_buffer`.
///
/// The pass reads the emitter descriptor via a uniform (not tracked in the
/// access set) and appends new particle records to the GPU particle buffer.
pub fn create_particle_spawn_pass(
    index: PassIndex,
    _emitter: &ParticleEmitter,
    particle_buffer: ResourceHandle,
) -> IrPass {
    let mut pass = IrPass::compute(
        index,
        "particle_spawn",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::StorageBuffer,
    );
    pass.access_set.writes.push(particle_buffer);
    pass.tags.push("particle".into());
    pass.tags.push("spawn".into());
    pass
}

// ---------------------------------------------------------------------------
// Update pass
// ---------------------------------------------------------------------------

/// Build a compute pass that integrates particle physics and lifetimes.
///
/// Reads the current particle buffer, advances positions / velocities by
/// `delta_time`, and marks expired particles as dead (so the subsequent
/// compact pass can remove them).
///
/// The pass performs a read-modify-write on `particle_buffer`.
pub fn create_particle_update_pass(
    index: PassIndex,
    particle_buffer: ResourceHandle,
    _delta_time: f32,
) -> IrPass {
    let mut pass = IrPass::compute(
        index,
        "particle_update",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::StorageBuffer,
    );
    pass.access_set.reads.push(particle_buffer);
    pass.access_set.writes.push(particle_buffer);
    pass.tags.push("particle".into());
    pass.tags.push("update".into());
    pass
}

// ---------------------------------------------------------------------------
// Render pass
// ---------------------------------------------------------------------------

/// Build a graphics pass that renders camera-facing quads for every live
/// particle.
///
/// `particle_buffer` is bound as a storage (read-only) buffer so the vertex
/// shader can read per-particle positions.  The pass renders onto `output`
/// (the current render-target colour attachment) with `Load`+`Store`
/// semantics.
pub fn create_particle_render_pass(
    index: PassIndex,
    particle_buffer: ResourceHandle,
    output: ResourceHandle,
) -> IrPass {
    let mut pass = IrPass::graphics(
        index,
        "particle_render",
        vec![ColorAttachment {
            resource: output,
            load_op: AttachmentLoadOp::Load,
            store_op: AttachmentStoreOp::Store,
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 6,     // two triangles per quad
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );
    // The particle buffer is read by the vertex shader — it is not a colour
    // or depth-stencil attachment, so we must add it to the access set
    // manually.
    pass.access_set.reads.push(particle_buffer);
    pass.tags.push("particle".into());
    pass.tags.push("render".into());
    pass
}

// ---------------------------------------------------------------------------
// Compact pass
// ---------------------------------------------------------------------------

/// Build a compute pass that removes dead particles from the buffer (prefix
/// sum compaction).
///
/// The pass reads the current particle buffer, identifies live particles,
/// and packs them into a contiguous region, writing back the compacted
/// particle count.
pub fn create_particle_compact_pass(
    index: PassIndex,
    particle_buffer: ResourceHandle,
) -> IrPass {
    let mut pass = IrPass::compute(
        index,
        "particle_compact",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::StorageBuffer,
    );
    pass.access_set.reads.push(particle_buffer);
    pass.access_set.writes.push(particle_buffer);
    pass.tags.push("particle".into());
    pass.tags.push("compact".into());
    pass
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::{PassType, ResourceHandle};

    // ── ParticleEmitter ─────────────────────────────────────────────────

    #[test]
    fn test_particle_emitter_default() {
        let emitter = ParticleEmitter::default();
        assert_eq!(emitter.position, [0.0, 0.0, 0.0]);
        assert!((emitter.rate - 100.0).abs() < f32::EPSILON);
        assert_eq!(emitter.max_particles, 4096);
        assert!((emitter.lifetime - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_particle_emitter_custom() {
        let emitter = ParticleEmitter {
            position: [1.0, 2.0, 3.0],
            rate: 250.0,
            max_particles: 8192,
            lifetime: 5.0,
        };
        assert_eq!(emitter.position, [1.0, 2.0, 3.0]);
        assert!((emitter.rate - 250.0).abs() < f32::EPSILON);
        assert_eq!(emitter.max_particles, 8192);
        assert!((emitter.lifetime - 5.0).abs() < f32::EPSILON);
    }

    // ── Spawn pass ──────────────────────────────────────────────────────

    #[test]
    fn test_spawn_pass_type() {
        let emitter = ParticleEmitter::default();
        let pass = create_particle_spawn_pass(PassIndex(0), &emitter, ResourceHandle(1));
        assert_eq!(pass.pass_type, PassType::Compute);
        assert_eq!(pass.name, "particle_spawn");
    }

    #[test]
    fn test_spawn_pass_writes_particle_buffer() {
        let emitter = ParticleEmitter::default();
        let pass = create_particle_spawn_pass(PassIndex(0), &emitter, ResourceHandle(1));
        assert!(pass.access_set.writes.contains(&ResourceHandle(1)));
        assert!(
            !pass.access_set.reads.contains(&ResourceHandle(1)),
            "spawn should not read the particle buffer"
        );
    }

    #[test]
    fn test_spawn_pass_tags() {
        let emitter = ParticleEmitter::default();
        let pass = create_particle_spawn_pass(PassIndex(0), &emitter, ResourceHandle(1));
        assert!(pass.tags.contains(&"particle".into()));
        assert!(pass.tags.contains(&"spawn".into()));
    }

    #[test]
    fn test_spawn_pass_view_type() {
        let emitter = ParticleEmitter::default();
        let pass = create_particle_spawn_pass(PassIndex(0), &emitter, ResourceHandle(1));
        assert_eq!(pass.view_type, ViewType::StorageBuffer);
    }

    // ── Update pass ─────────────────────────────────────────────────────

    #[test]
    fn test_update_pass_type() {
        let pass = create_particle_update_pass(PassIndex(1), ResourceHandle(1), 1.0 / 60.0);
        assert_eq!(pass.pass_type, PassType::Compute);
        assert_eq!(pass.name, "particle_update");
    }

    #[test]
    fn test_update_pass_read_write_particle_buffer() {
        let pass = create_particle_update_pass(PassIndex(1), ResourceHandle(1), 1.0 / 60.0);
        assert!(
            pass.access_set.reads.contains(&ResourceHandle(1)),
            "update must read the particle buffer"
        );
        assert!(
            pass.access_set.writes.contains(&ResourceHandle(1)),
            "update must write the particle buffer (read-modify-write)"
        );
    }

    #[test]
    fn test_update_pass_tags() {
        let pass = create_particle_update_pass(PassIndex(0), ResourceHandle(1), 0.016);
        assert!(pass.tags.contains(&"update".into()));
        assert!(pass.tags.contains(&"particle".into()));
    }

    #[test]
    fn test_update_pass_view_type() {
        let pass = create_particle_update_pass(PassIndex(0), ResourceHandle(1), 0.016);
        assert_eq!(pass.view_type, ViewType::StorageBuffer);
    }

    // ── Render pass ─────────────────────────────────────────────────────

    #[test]
    fn test_render_pass_type() {
        let pass = create_particle_render_pass(PassIndex(2), ResourceHandle(1), ResourceHandle(2));
        assert_eq!(pass.pass_type, PassType::Graphics);
        assert_eq!(pass.name, "particle_render");
    }

    #[test]
    fn test_render_pass_color_attachment() {
        let pass = create_particle_render_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert_eq!(pass.color_attachments.len(), 1);
        assert_eq!(pass.color_attachments[0].resource, ResourceHandle(2));
        assert_eq!(pass.color_attachments[0].load_op, AttachmentLoadOp::Load);
        assert_eq!(pass.color_attachments[0].store_op, AttachmentStoreOp::Store);
    }

    #[test]
    fn test_render_pass_reads_particle_buffer() {
        let pass = create_particle_render_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert!(
            pass.access_set.reads.contains(&ResourceHandle(1)),
            "render pass must read the particle buffer for vertex data"
        );
    }

    #[test]
    fn test_render_pass_writes_output() {
        let pass = create_particle_render_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert!(
            pass.access_set.writes.contains(&ResourceHandle(2)),
            "render pass must write the output colour attachment"
        );
    }

    #[test]
    fn test_render_pass_no_depth_stencil() {
        let pass = create_particle_render_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert!(pass.depth_stencil.is_none());
    }

    #[test]
    fn test_render_pass_tags() {
        let pass = create_particle_render_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        assert!(pass.tags.contains(&"render".into()));
        assert!(pass.tags.contains(&"particle".into()));
    }

    // ── Compact pass ────────────────────────────────────────────────────

    #[test]
    fn test_compact_pass_type() {
        let pass = create_particle_compact_pass(PassIndex(3), ResourceHandle(1));
        assert_eq!(pass.pass_type, PassType::Compute);
        assert_eq!(pass.name, "particle_compact");
    }

    #[test]
    fn test_compact_pass_read_write_particle_buffer() {
        let pass = create_particle_compact_pass(PassIndex(0), ResourceHandle(1));
        assert!(
            pass.access_set.reads.contains(&ResourceHandle(1)),
            "compact must read the particle buffer"
        );
        assert!(
            pass.access_set.writes.contains(&ResourceHandle(1)),
            "compact must write the particle buffer"
        );
    }

    #[test]
    fn test_compact_pass_tags() {
        let pass = create_particle_compact_pass(PassIndex(0), ResourceHandle(1));
        assert!(pass.tags.contains(&"compact".into()));
        assert!(pass.tags.contains(&"particle".into()));
    }

    #[test]
    fn test_compact_pass_view_type() {
        let pass = create_particle_compact_pass(PassIndex(0), ResourceHandle(1));
        assert_eq!(pass.view_type, ViewType::StorageBuffer);
    }

    // ── Cross-pass integration ──────────────────────────────────────────

    #[test]
    fn test_particle_pipeline_indices_are_distinct() {
        let emitter = ParticleEmitter::default();
        let buf = ResourceHandle(100);
        let out = ResourceHandle(200);

        let spawn = create_particle_spawn_pass(PassIndex(0), &emitter, buf);
        let update = create_particle_update_pass(PassIndex(1), buf, 0.016);
        let render = create_particle_render_pass(PassIndex(2), buf, out);
        let compact = create_particle_compact_pass(PassIndex(3), buf);

        let indices: Vec<PassIndex> =
            vec![&spawn, &update, &render, &compact].iter().map(|p| p.index).collect();
        assert_eq!(indices[0], PassIndex(0));
        assert_eq!(indices[1], PassIndex(1));
        assert_eq!(indices[2], PassIndex(2));
        assert_eq!(indices[3], PassIndex(3));
    }

    #[test]
    fn test_particle_pipeline_all_share_buffer() {
        let emitter = ParticleEmitter::default();
        let buf = ResourceHandle(42);

        let spawn = create_particle_spawn_pass(PassIndex(0), &emitter, buf);
        let update = create_particle_update_pass(PassIndex(1), buf, 0.016);
        let render = create_particle_render_pass(PassIndex(2), buf, ResourceHandle(99));
        let compact = create_particle_compact_pass(PassIndex(3), buf);

        // Every pass must touch the shared particle buffer.
        for pass in [&spawn, &update, &render, &compact] {
            assert!(
                pass.access_set.contains(buf),
                "pass '{}' must touch particle buffer",
                pass.name
            );
        }
    }

    #[test]
    fn test_render_pass_instance_source() {
        let pass = create_particle_render_pass(PassIndex(0), ResourceHandle(1), ResourceHandle(2));
        match pass.instance_source {
            InstanceSource::Direct {
                index_count,
                instance_count,
                ..
            } => {
                assert_eq!(index_count, 6, "particle quad = 2 triangles = 6 indices");
                assert_eq!(instance_count, 1);
            }
            _ => panic!("expected Direct instance source"),
        }
    }
}
