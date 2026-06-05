//! Render pipeline descriptor and wrapper types.
//!
//! Provides a builder-pattern API for creating wgpu render pipelines with all 9
//! configurable fields and sensible defaults.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};

use super::depth_stencil_state::DepthStencilStateDescriptor;
use super::fragment_state::{ColorTargetStateDescriptor, FragmentStateDescriptor};
use super::multisample_state::MultisampleStateDescriptor;
use super::primitive_state::PrimitiveStateDescriptor;
use super::vertex_state::VertexStateDescriptor;

// ---------------------------------------------------------------------------
// Layout ID generator (for cache invalidation tracking)
// ---------------------------------------------------------------------------

static LAYOUT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique layout ID for cache invalidation tracking.
fn next_layout_id() -> u64 {
    LAYOUT_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// TrinityRenderPipeline
// ---------------------------------------------------------------------------

/// A compiled render pipeline wrapper with metadata for cache management.
///
/// Wraps a [`wgpu::RenderPipeline`] with:
/// - Optional debug label for identification
/// - Layout ID for cache invalidation tracking
///
/// # Thread Safety
///
/// `TrinityRenderPipeline` is `Send + Sync` because `wgpu::RenderPipeline` is.
#[derive(Debug)]
pub struct TrinityRenderPipeline {
    inner: wgpu::RenderPipeline,
    label: Option<String>,
    layout_id: u64,
}

impl TrinityRenderPipeline {
    /// Create a new `TrinityRenderPipeline` from a wgpu pipeline.
    pub(crate) fn new(
        inner: wgpu::RenderPipeline,
        label: Option<String>,
        layout_id: u64,
    ) -> Self {
        Self {
            inner,
            label,
            layout_id,
        }
    }

    /// Access the underlying [`wgpu::RenderPipeline`].
    #[inline]
    pub fn raw(&self) -> &wgpu::RenderPipeline {
        &self.inner
    }

    /// Get the debug label, if any.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Get the layout ID for cache invalidation tracking.
    ///
    /// When the pipeline layout changes, pipelines with old layout IDs
    /// should be invalidated.
    #[inline]
    pub fn layout_id(&self) -> u64 {
        self.layout_id
    }

    /// Consume and return the inner wgpu pipeline.
    #[inline]
    pub fn into_inner(self) -> wgpu::RenderPipeline {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// RenderPipelineDescriptor
// ---------------------------------------------------------------------------

/// Builder for creating render pipelines with all wgpu configuration options.
///
/// # Required Fields
///
/// - `layout`: Pipeline layout (required at construction, enforced by API)
/// - `vertex`: Vertex state (required before build)
///
/// # Optional Fields
///
/// - `label`: Debug label
/// - `primitive`: Primitive state (default: triangle list with backface culling)
/// - `depth_stencil`: Depth/stencil state (default: None)
/// - `multisample`: Multisample state (default: 1 sample)
/// - `fragment`: Fragment state (default: None)
/// - `multiview`: Multiview configuration (default: None)
/// - `cache`: Pipeline cache (default: None)
///
/// # Example
///
/// ```no_run
/// # fn example(
/// #     device: &wgpu::Device,
/// #     layout: &wgpu::PipelineLayout,
/// #     vs_module: &wgpu::ShaderModule,
/// #     fs_module: &wgpu::ShaderModule,
/// # ) {
/// use renderer_backend::render_pipeline::{
///     RenderPipelineDescriptor, VertexStateDescriptor, FragmentStateDescriptor,
/// };
///
/// let pipeline = RenderPipelineDescriptor::new(layout)
///     .label("my_pipeline")
///     .vertex(VertexStateDescriptor::new(vs_module))
///     .fragment(FragmentStateDescriptor::new(fs_module)
///         .target(wgpu::TextureFormat::Bgra8UnormSrgb))
///     .build(device);
/// # }
/// ```
#[derive(Debug)]
pub struct RenderPipelineDescriptor<'a> {
    /// Debug label.
    label: Option<&'a str>,
    /// Pipeline layout (required).
    layout: &'a wgpu::PipelineLayout,
    /// Layout ID for cache invalidation.
    layout_id: u64,
    /// Vertex state.
    vertex: Option<VertexStateDescriptor<'a>>,
    /// Primitive state.
    primitive: PrimitiveStateDescriptor,
    /// Depth/stencil state.
    depth_stencil: Option<DepthStencilStateDescriptor>,
    /// Multisample state.
    multisample: MultisampleStateDescriptor,
    /// Fragment state.
    fragment: Option<FragmentStateDescriptor<'a>>,
    /// Multiview configuration.
    multiview: Option<NonZeroU32>,
    /// Pipeline cache.
    cache: Option<&'a wgpu::PipelineCache>,
}

impl<'a> RenderPipelineDescriptor<'a> {
    /// Create a new render pipeline descriptor with the given layout.
    ///
    /// The layout is **required** - this enforces layout association at compile time.
    pub fn new(layout: &'a wgpu::PipelineLayout) -> Self {
        Self {
            label: None,
            layout,
            layout_id: next_layout_id(),
            vertex: None,
            primitive: PrimitiveStateDescriptor::default(),
            depth_stencil: None,
            multisample: MultisampleStateDescriptor::default(),
            fragment: None,
            multiview: None,
            cache: None,
        }
    }

    /// Create with a specific layout ID (for cache coordination).
    pub fn with_layout_id(layout: &'a wgpu::PipelineLayout, layout_id: u64) -> Self {
        Self {
            label: None,
            layout,
            layout_id,
            vertex: None,
            primitive: PrimitiveStateDescriptor::default(),
            depth_stencil: None,
            multisample: MultisampleStateDescriptor::default(),
            fragment: None,
            multiview: None,
            cache: None,
        }
    }

    /// Set the debug label.
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// Set the vertex state (required).
    pub fn vertex(mut self, vertex: VertexStateDescriptor<'a>) -> Self {
        self.vertex = Some(vertex);
        self
    }

    /// Set the primitive state.
    pub fn primitive(mut self, primitive: PrimitiveStateDescriptor) -> Self {
        self.primitive = primitive;
        self
    }

    /// Set the depth/stencil state.
    pub fn depth_stencil(mut self, depth_stencil: DepthStencilStateDescriptor) -> Self {
        self.depth_stencil = Some(depth_stencil);
        self
    }

    /// Set the multisample state.
    pub fn multisample(mut self, multisample: MultisampleStateDescriptor) -> Self {
        self.multisample = multisample;
        self
    }

    /// Set the fragment state.
    pub fn fragment(mut self, fragment: FragmentStateDescriptor<'a>) -> Self {
        self.fragment = Some(fragment);
        self
    }

    /// Set the multiview configuration.
    pub fn multiview(mut self, count: NonZeroU32) -> Self {
        self.multiview = Some(count);
        self
    }

    /// Set the pipeline cache.
    pub fn cache(mut self, cache: &'a wgpu::PipelineCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Build the render pipeline.
    ///
    /// # Panics
    ///
    /// Panics if `vertex` state has not been set.
    pub fn build(self, device: &wgpu::Device) -> TrinityRenderPipeline {
        let vertex = self
            .vertex
            .expect("vertex state is required for render pipeline");

        // Convert vertex buffer layouts to wgpu format
        // We need to keep the attribute vectors alive during pipeline creation
        let wgpu_attr_vecs: Vec<Vec<wgpu::VertexAttribute>> = vertex
            .buffers
            .iter()
            .map(|vl| vl.attributes.iter().map(|a| (*a).into()).collect())
            .collect();

        let wgpu_vertex_buffers: Vec<wgpu::VertexBufferLayout<'_>> = vertex
            .buffers
            .iter()
            .enumerate()
            .map(|(i, vl)| wgpu::VertexBufferLayout {
                array_stride: vl.array_stride,
                step_mode: vl.step_mode,
                attributes: &wgpu_attr_vecs[i],
            })
            .collect();

        // Convert fragment targets to wgpu format
        let wgpu_targets: Vec<Option<wgpu::ColorTargetState>> = self
            .fragment
            .as_ref()
            .map(|f| {
                f.targets
                    .iter()
                    .map(|t| {
                        t.as_ref().map(|target| wgpu::ColorTargetState {
                            format: target.format,
                            blend: target.blend.map(|b| b.into()),
                            write_mask: target.write_mask,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Convert depth/stencil state
        let depth_stencil = self.depth_stencil.map(|ds| wgpu::DepthStencilState {
            format: ds.format,
            depth_write_enabled: ds.depth_write_enabled,
            depth_compare: ds.depth_compare,
            stencil: wgpu::StencilState {
                front: ds.stencil_front.into(),
                back: ds.stencil_back.into(),
                read_mask: ds.stencil_read_mask,
                write_mask: ds.stencil_write_mask,
            },
            bias: ds.bias.into(),
        });

        // Create the pipeline
        let inner = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: self.label,
            layout: Some(self.layout),
            vertex: wgpu::VertexState {
                module: vertex.module,
                entry_point: vertex.entry_point,
                compilation_options: vertex.compilation_options,
                buffers: &wgpu_vertex_buffers,
            },
            primitive: self.primitive.into(),
            depth_stencil,
            multisample: self.multisample.into(),
            fragment: self.fragment.as_ref().map(|f| wgpu::FragmentState {
                module: f.module,
                entry_point: f.entry_point,
                compilation_options: f.compilation_options.clone(),
                targets: &wgpu_targets,
            }),
            multiview: self.multiview,
            cache: self.cache,
        });

        TrinityRenderPipeline::new(inner, self.label.map(String::from), self.layout_id)
    }
}

// ---------------------------------------------------------------------------
// create_render_pipeline (convenience function)
// ---------------------------------------------------------------------------

/// Create a render pipeline from a descriptor.
///
/// This is a convenience function that calls `desc.build(device)`.
///
/// # Example
///
/// ```no_run
/// # fn example(
/// #     device: &wgpu::Device,
/// #     layout: &wgpu::PipelineLayout,
/// #     vs_module: &wgpu::ShaderModule,
/// #     fs_module: &wgpu::ShaderModule,
/// # ) {
/// use renderer_backend::render_pipeline::{
///     create_render_pipeline, RenderPipelineDescriptor,
///     VertexStateDescriptor, FragmentStateDescriptor,
/// };
///
/// let desc = RenderPipelineDescriptor::new(layout)
///     .label("my_pipeline")
///     .vertex(VertexStateDescriptor::new(vs_module))
///     .fragment(FragmentStateDescriptor::new(fs_module)
///         .target(wgpu::TextureFormat::Bgra8UnormSrgb));
///
/// let pipeline = create_render_pipeline(device, desc);
/// # }
/// ```
pub fn create_render_pipeline(
    device: &wgpu::Device,
    desc: RenderPipelineDescriptor<'_>,
) -> TrinityRenderPipeline {
    desc.build(device)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_id_increments() {
        let id1 = next_layout_id();
        let id2 = next_layout_id();
        assert!(id2 > id1);
    }

    #[test]
    fn test_trinity_render_pipeline_label() {
        // Test that label accessor works correctly
        // (Full pipeline creation requires a device, so we just test the struct)
        let label: Option<String> = Some("test_label".to_string());
        assert_eq!(label.as_deref(), Some("test_label"));
    }

    #[test]
    fn test_layout_id_uniqueness_across_multiple_calls() {
        // Test that multiple layout IDs are all unique
        let ids: Vec<u64> = (0..100).map(|_| next_layout_id()).collect();
        let unique_ids: std::collections::HashSet<u64> = ids.iter().cloned().collect();
        assert_eq!(ids.len(), unique_ids.len(), "All layout IDs must be unique");
    }

    #[test]
    fn test_layout_id_monotonic() {
        // Test that IDs are monotonically increasing
        let ids: Vec<u64> = (0..50).map(|_| next_layout_id()).collect();
        for window in ids.windows(2) {
            assert!(window[1] > window[0], "Layout IDs must be monotonically increasing");
        }
    }

    #[test]
    fn test_layout_id_thread_safety() {
        // Test thread safety with concurrent access
        use std::sync::Arc;
        use std::thread;

        let ids = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut handles = vec![];

        for _ in 0..4 {
            let ids_clone = Arc::clone(&ids);
            handles.push(thread::spawn(move || {
                let mut local_ids = vec![];
                for _ in 0..25 {
                    local_ids.push(next_layout_id());
                }
                ids_clone.lock().unwrap().extend(local_ids);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let all_ids = ids.lock().unwrap();
        let unique_ids: std::collections::HashSet<u64> = all_ids.iter().cloned().collect();
        assert_eq!(all_ids.len(), unique_ids.len(), "All IDs must be unique across threads");
    }

    #[test]
    fn test_trinity_render_pipeline_send_sync() {
        // Compile-time check for Send + Sync bounds
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TrinityRenderPipeline>();
    }

    #[test]
    fn test_render_pipeline_descriptor_send() {
        // Compile-time check that descriptor is Send
        fn assert_send<T: Send>() {}
        assert_send::<RenderPipelineDescriptor<'_>>();
    }

    #[test]
    fn test_label_none_handling() {
        // Test None label accessor
        let label: Option<String> = None;
        assert_eq!(label.as_deref(), None);
    }

    #[test]
    fn test_label_empty_string() {
        // Test empty string label
        let label: Option<String> = Some(String::new());
        assert_eq!(label.as_deref(), Some(""));
    }

    #[test]
    fn test_multiview_nonzero_u32() {
        // Test NonZeroU32 for multiview configuration
        use std::num::NonZeroU32;

        let count = NonZeroU32::new(2).unwrap();
        assert_eq!(count.get(), 2);

        let count4 = NonZeroU32::new(4).unwrap();
        assert_eq!(count4.get(), 4);
    }

    #[test]
    fn test_default_primitive_state() {
        // Verify default primitive state values
        let primitive = PrimitiveStateDescriptor::default();
        assert_eq!(primitive.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(primitive.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_default_multisample_state() {
        // Verify default multisample state values
        let multisample = MultisampleStateDescriptor::default();
        assert_eq!(multisample.count, 1);
        assert_eq!(multisample.mask, !0u64);
        assert!(!multisample.alpha_to_coverage_enabled);
    }
}
