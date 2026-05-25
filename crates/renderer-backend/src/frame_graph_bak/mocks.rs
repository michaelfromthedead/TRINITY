//! Mock constructors for Rust-side frame graph compiler testing.
//!
//! Provides builder-style helpers that wrap the IR types with ergonomic
//! defaults so that compiler phases can be tested without Python or a
//! full bridge setup.
//!
//! # Types
//!
//! - [`MockResourceDesc`] — builds [`IrResource`] with auto-assigned unique
//!   handles from a global atomic counter.
//! - [`MockPassNode`] — builds [`IrPass`] with a chainable builder
//!   (`.reads()`, `.writes()`, `.color_attachment()`, `.depth_stencil()`).
//! - [`reset_mock_handles`] — resets the global handle counter so that
//!   tests produce deterministic handle sequences.
//!
//! [`IrPass`]: super::IrPass
//! [`IrResource`]: super::IrResource

use core::sync::atomic::{AtomicU32, Ordering};

use super::{
    AttachmentLoadOp, AttachmentStoreOp, ColorAttachment, DepthStencilAttachment,
    DispatchSource, InstanceSource, IrPass, IrResource, PassIndex, PassType, ResourceDesc,
    ResourceHandle, ResourceLifetime, ResourceState, TextureDesc, ViewType,
};

/// Global atomic counter for unique resource handles.
static NEXT_MOCK_HANDLE: AtomicU32 = AtomicU32::new(1);

/// Resets the global mock handle counter to 1.
///
/// Call before each test that needs deterministic handle values.
pub fn reset_mock_handles() {
    NEXT_MOCK_HANDLE.store(1, Ordering::SeqCst);
}

/// Allocates the next unique handle.
fn next_handle() -> ResourceHandle {
    let raw = NEXT_MOCK_HANDLE.fetch_add(1, Ordering::SeqCst);
    debug_assert!(
        raw < u32::MAX,
        "mock handle counter wrapped past NONE sentinel"
    );
    ResourceHandle(raw)
}

// ---------------------------------------------------------------------------
// MockResourceDesc
// ---------------------------------------------------------------------------

/// Builder for [`IrResource`] that auto-assigns a unique [`ResourceHandle`].
///
/// # Examples
///
/// ```ignore
/// use frame_graph::mocks::MockResourceDesc;
///
/// let desc = MockResourceDesc::texture_2d("color_rt", 1920, 1080);
/// let handle = desc.handle();          // peek at the assigned handle
/// let resource: IrResource = desc.build();
/// ```
pub struct MockResourceDesc {
    handle: ResourceHandle,
    name: String,
    desc: ResourceDesc,
}

impl MockResourceDesc {
    /// Creates a new builder for a 2D texture resource.
    ///
    /// Defaults: `mip_levels = 1`, `array_layers = 1`, format = `"rgba8unorm"`,
    /// lifetime = `Transient`, initial_state = `Uninitialized`.
    pub fn texture_2d(name: &str, width: u32, height: u32) -> Self {
        Self {
            handle: next_handle(),
            name: name.to_owned(),
            desc: ResourceDesc::Texture2D(TextureDesc {
                width,
                height,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
        }
    }

    /// Creates a new builder for a buffer resource.
    ///
    /// Defaults: usage = `"storage | indirect"`, is_indirect_arg = `false`,
    /// lifetime = `Transient`, initial_state = `Uninitialized`.
    pub fn buffer(name: &str, size: u64) -> Self {
        Self {
            handle: next_handle(),
            name: name.to_owned(),
            desc: ResourceDesc::Buffer(super::BufferDesc {
                size,
                usage: "storage | indirect".into(),
                is_indirect_arg: false,
            }),
        }
    }

    /// Returns the [`ResourceHandle`] that will be assigned to the built
    /// resource. Useful for wiring handles into [`MockPassNode`] before
    /// building the resource.
    pub fn handle(&self) -> ResourceHandle {
        self.handle
    }

    /// Consumes the builder and produces an [`IrResource`].
    pub fn build(self) -> IrResource {
        IrResource::new(
            self.handle,
            self.name,
            self.desc,
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        )
    }
}

// ---------------------------------------------------------------------------
// MockPassNode
// ---------------------------------------------------------------------------

/// Builder for [`IrPass`] with chainable access set and attachment helpers.
///
/// # Examples
///
/// ```ignore
/// use frame_graph::mocks::MockPassNode;
///
/// let pass = MockPassNode::compute("test")
///     .reads(&[h0])
///     .writes(&[h1])
///     .build();
/// ```
pub struct MockPassNode {
    pass_type: PassType,
    name: String,
    reads: Vec<ResourceHandle>,
    writes: Vec<ResourceHandle>,
    color_attachments: Vec<ColorAttachment>,
    depth_stencil: Option<DepthStencilAttachment>,
}

impl MockPassNode {
    /// Creates a builder for a graphics pass.
    ///
    /// Default view type is `ColorAttachment`.
    pub fn graphics(name: &str) -> Self {
        Self {
            pass_type: PassType::Graphics,
            name: name.to_owned(),
            reads: Vec::new(),
            writes: Vec::new(),
            color_attachments: Vec::new(),
            depth_stencil: None,
        }
    }

    /// Creates a builder for a compute pass.
    ///
    /// Default view type is `Storage`.
    pub fn compute(name: &str) -> Self {
        Self {
            pass_type: PassType::Compute,
            name: name.to_owned(),
            reads: Vec::new(),
            writes: Vec::new(),
            color_attachments: Vec::new(),
            depth_stencil: None,
        }
    }

    /// Creates a builder for a copy pass.
    ///
    /// Default view type is `StorageBuffer`.
    pub fn copy(name: &str) -> Self {
        Self {
            pass_type: PassType::Copy,
            name: name.to_owned(),
            reads: Vec::new(),
            writes: Vec::new(),
            color_attachments: Vec::new(),
            depth_stencil: None,
        }
    }

    /// Adds resource handles to the read set.
    pub fn reads(mut self, handles: &[ResourceHandle]) -> Self {
        self.reads.extend_from_slice(handles);
        self
    }

    /// Adds resource handles to the write set.
    pub fn writes(mut self, handles: &[ResourceHandle]) -> Self {
        self.writes.extend_from_slice(handles);
        self
    }

    /// Adds a colour attachment with default Clear→Store semantics.
    ///
    /// The attachment targets the given resource at mip 0, layer 0.
    pub fn color_attachment(mut self, resource: ResourceHandle) -> Self {
        self.color_attachments.push(ColorAttachment {
            resource,
            mip_level: 0,
            array_layer: 0,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 0.0],
        });
        self
    }

    /// Adds a depth-stencil attachment with default Load→Store semantics.
    pub fn depth_stencil(mut self, resource: ResourceHandle) -> Self {
        self.depth_stencil = Some(DepthStencilAttachment {
            resource,
            depth_load_op: AttachmentLoadOp::Load,
            depth_store_op: AttachmentStoreOp::Store,
            stencil_load_op: AttachmentLoadOp::Load,
            stencil_store_op: AttachmentStoreOp::DontCare,
            clear_depth: 1.0,
            clear_stencil: 0,
            depth_test_enabled: true,
            depth_write_enabled: true,
        });
        self
    }

    /// Consumes the builder and produces an [`IrPass`].
    ///
    /// For graphics passes, the access set is synchronised from colour and
    /// depth-stencil attachments.  Callers may also push additional entries
    /// into the access set fields after construction.
    pub fn build(self) -> IrPass {
        let view_type = match self.pass_type {
            PassType::Graphics => ViewType::ColorAttachment,
            PassType::Compute => ViewType::Storage,
            PassType::Copy | PassType::RayTracing => ViewType::StorageBuffer,
        };

        // For graphics passes, construct via IrPass::graphics which runs
        // sync_access_set_from_attachments internally.
        if self.pass_type == PassType::Graphics {
            let mut pass = IrPass::graphics(
                PassIndex(0),
                self.name,
                self.color_attachments,
                self.depth_stencil,
                InstanceSource::Direct {
                    index_count: 6,
                    instance_count: 1,
                    base_vertex: 0,
                    first_index: 0,
                    first_instance: 0,
                },
                view_type,
            );
            // Merge manually added reads/writes.
            pass.access_set.reads.extend(self.reads);
            pass.access_set.writes.extend(self.writes);
            return pass;
        }

        // Compute or copy.
        let mut pass = match self.pass_type {
            PassType::Compute => IrPass::compute(
                PassIndex(0),
                self.name,
                DispatchSource::Direct {
                    group_count_x: 1,
                    group_count_y: 1,
                    group_count_z: 1,
                },
                view_type,
            ),
            PassType::Copy => IrPass::copy(PassIndex(0), self.name),
            _ => unreachable!(),
        };

        pass.access_set.reads = self.reads;
        pass.access_set.writes = self.writes;
        pass
    }
}

// ---------------------------------------------------------------------------
// Whitebox tests — T-FG-1.7 MockPassNode + MockResourceDesc
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // MockResourceDesc tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mock_resource_desc_texture_2d_creates_texture() {
        let desc = MockResourceDesc::texture_2d("color_rt", 1920, 1080);
        let resource = desc.build();
        assert!(resource.handle.0 >= 1, "handle should be non-NONE");
        assert_eq!(resource.name, "color_rt");
        match &resource.desc {
            ResourceDesc::Texture2D(t) => {
                assert_eq!(t.width, 1920);
                assert_eq!(t.height, 1080);
                assert_eq!(t.mip_levels, 1);
                assert_eq!(t.array_layers, 1);
            }
            _ => panic!("expected Texture2D variant"),
        }
    }

    #[test]
    fn test_mock_resource_desc_buffer_creates_buffer() {
        let desc = MockResourceDesc::buffer("ssbo", 65536);
        let resource = desc.build();
        assert!(resource.handle.0 >= 1);
        assert_eq!(resource.name, "ssbo");
        match &resource.desc {
            ResourceDesc::Buffer(b) => {
                assert_eq!(b.size, 65536);
                assert!(b.usage.contains("storage"));
            }
            _ => panic!("expected Buffer variant"),
        }
    }

    #[test]
    fn test_mock_resource_desc_unique_handles() {
        reset_mock_handles();
        let a = MockResourceDesc::texture_2d("a", 100, 100).build();
        let b = MockResourceDesc::texture_2d("b", 200, 200).build();
        assert_ne!(a.handle, b.handle, "handles must be unique");
    }

    #[test]
    fn test_mock_resource_desc_handle_peek() {
        reset_mock_handles();
        let builder = MockResourceDesc::texture_2d("peek", 64, 64);
        let h = builder.handle();
        let resource = builder.build();
        assert_eq!(h, resource.handle, "peek handle must match build handle");
    }

    #[test]
    fn test_reset_mock_handles_determinism() {
        reset_mock_handles();
        let a1 = MockResourceDesc::texture_2d("x", 1, 1).build();
        reset_mock_handles();
        let a2 = MockResourceDesc::texture_2d("x", 1, 1).build();
        assert_eq!(
            a1.handle, a2.handle,
            "reset should produce deterministic handle sequences"
        );
    }

    // -----------------------------------------------------------------------
    // MockPassNode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mock_pass_node_graphics_type() {
        let pass = MockPassNode::graphics("gfx_pass").build();
        assert_eq!(pass.name, "gfx_pass");
        assert_eq!(pass.pass_type, PassType::Graphics);
    }

    #[test]
    fn test_mock_pass_node_compute_type() {
        let pass = MockPassNode::compute("compute_pass").build();
        assert_eq!(pass.name, "compute_pass");
        assert_eq!(pass.pass_type, PassType::Compute);
    }

    #[test]
    fn test_mock_pass_node_copy_type() {
        let pass = MockPassNode::copy("copy_pass").build();
        assert_eq!(pass.name, "copy_pass");
        assert_eq!(pass.pass_type, PassType::Copy);
    }

    #[test]
    fn test_mock_pass_node_reads_writes() {
        reset_mock_handles();
        let r_src = MockResourceDesc::buffer("src", 1024);
        let r_dst = MockResourceDesc::buffer("dst", 1024);
        let pass = MockPassNode::compute("rw_test")
            .reads(&[r_src.handle()])
            .writes(&[r_dst.handle()])
            .build();
        assert!(pass.access_set.reads.contains(&r_src.handle()));
        assert!(pass.access_set.writes.contains(&r_dst.handle()));
    }

    #[test]
    fn test_mock_pass_node_color_attachment() {
        reset_mock_handles();
        let rt = MockResourceDesc::texture_2d("rt", 800, 600);
        let pass = MockPassNode::graphics("color_pass")
            .color_attachment(rt.handle())
            .build();
        assert_eq!(pass.color_attachments.len(), 1);
        assert_eq!(pass.color_attachments[0].resource, rt.handle());
        // The attachment should also appear in the write set via sync.
        assert!(pass.access_set.writes.contains(&rt.handle()));
    }

    #[test]
    fn test_mock_pass_node_depth_stencil() {
        reset_mock_handles();
        let ds = MockResourceDesc::texture_2d("depth", 800, 600);
        let pass = MockPassNode::graphics("depth_pass")
            .depth_stencil(ds.handle())
            .build();
        assert!(pass.depth_stencil.is_some());
        let ds_att = pass.depth_stencil.as_ref().unwrap();
        assert_eq!(ds_att.resource, ds.handle());
        // Depth-stencil resource should be in the write set.
        assert!(pass.access_set.writes.contains(&ds.handle()));
    }

    #[test]
    fn test_mock_pass_node_chained_builder() {
        reset_mock_handles();
        let r1 = MockResourceDesc::texture_2d("in", 100, 100);
        let r2 = MockResourceDesc::texture_2d("out", 100, 100);
        let ds = MockResourceDesc::texture_2d("depth", 100, 100);
        let pass = MockPassNode::graphics("chained")
            .reads(&[r1.handle()])
            .color_attachment(r2.handle())
            .depth_stencil(ds.handle())
            .build();
        assert!(pass.access_set.reads.contains(&r1.handle()));
        assert!(pass.access_set.writes.contains(&r2.handle()));
        assert!(pass.access_set.writes.contains(&ds.handle()));
        assert_eq!(pass.color_attachments.len(), 1);
        assert!(pass.depth_stencil.is_some());
    }
}
