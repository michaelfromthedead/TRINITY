//! RHI command recording layer.
//!
//! Thin wrappers around the wgpu command encoding primitives:
//!
//! | Rust type | wgpu counterpart |
//! |---|---|
//! | [`RhiCommandEncoder`] | [`wgpu::CommandEncoder`] |
//! | [`RhiRenderPass`] | [`wgpu::RenderPass`] |
//! | [`RhiComputePass`] | [`wgpu::ComputePass`] |
//! | [`RhiCommandBuffer`] | [`wgpu::CommandBuffer`] |
//!
//! Together with helpers for creating, finishing, and submitting command
//! buffers via the [`RhiDevice`] queue.
//!
//! # Workflow
//!
//! ```ignore
//! let mut encoder = create_command_encoder(&device);
//!
//! {
//!     let mut rp = RhiRenderPass::begin(&mut encoder, &render_pass_desc);
//!     rp.set_pipeline(&render_pipeline);
//!     rp.set_bind_group(0, &bind_group, &[]);
//!     rp.set_vertex_buffer(0, vertex_buffer.slice(..));
//!     rp.draw(0..3, 0..1);
//!     rp.end();
//! }
//!
//! let cb = finish_encoder(encoder);
//! submit(&rhi_device, vec![cb]);
//! ```
//!
//! [`RhiDevice`]: crate::rhi_device::RhiDevice

use crate::rhi_device::RhiDevice;

// ---------------------------------------------------------------------------
// RhiCommandEncoder
// ---------------------------------------------------------------------------

/// Thin wrapper around a [`wgpu::CommandEncoder`].
///
/// Records GPU commands (render passes, compute passes, copies) into a
/// command buffer that can later be submitted to the GPU queue.
pub struct RhiCommandEncoder {
    /// The underlying wgpu command encoder.
    inner: wgpu::CommandEncoder,
}

impl RhiCommandEncoder {
    /// Access the inner [`wgpu::CommandEncoder`] by shared reference.
    pub fn inner(&self) -> &wgpu::CommandEncoder {
        &self.inner
    }

    /// Access the inner [`wgpu::CommandEncoder`] by mutable reference.
    pub fn inner_mut(&mut self) -> &mut wgpu::CommandEncoder {
        &mut self.inner
    }
}

// ---------------------------------------------------------------------------
// RhiRenderPass
// ---------------------------------------------------------------------------

/// Thin wrapper around a [`wgpu::RenderPass`].
///
/// Created via [`RhiRenderPass::begin`] and ended via [`end`](Self::end).
/// While alive, the underlying [`RhiCommandEncoder`] is mutably borrowed so
/// no other pass can be recorded concurrently.
pub struct RhiRenderPass<'a> {
    /// The underlying wgpu render pass.
    inner: wgpu::RenderPass<'a>,
}

impl<'encoder> RhiRenderPass<'encoder> {
    /// Begin a render pass on the given encoder.
    ///
    /// Borrows the encoder mutably until [`end`](Self::end) is called (or
    /// this pass is dropped).
    ///
    /// # Parameters
    ///
    /// * `encoder` — The command encoder to record into.
    /// * `desc` — Descriptor specifying colour / depth-stencil attachments,
    ///   load/store operations, and optional timestamp writes.
    pub fn begin(
        encoder: &'encoder mut RhiCommandEncoder,
        desc: &wgpu::RenderPassDescriptor<'_>,
    ) -> Self {
        let inner = encoder.inner.begin_render_pass(desc);
        Self { inner }
    }

    /// End the render pass.
    ///
    /// Consumes the pass, releasing the mutable borrow on the encoder so it
    /// can be used for further recording or finishing.
    pub fn end(self) {
        // The inner wgpu::RenderPass is dropped here, ending the pass.
    }

    /// Set the active render pipeline for subsequent draw calls.
    pub fn set_pipeline(&mut self, pipeline: &wgpu::RenderPipeline) {
        self.inner.set_pipeline(pipeline);
    }

    /// Set a bind group for a given binding index.
    ///
    /// # Parameters
    ///
    /// * `group_index` — The bind group index (0-based) matching the
    ///   shader's `@group(N)` attribute.
    /// * `bind_group` — The bind group to bind.
    /// * `dynamic_offsets` — Dynamic uniform / storage buffer offsets.
    pub fn set_bind_group(
        &mut self,
        group_index: u32,
        bind_group: &wgpu::BindGroup,
        dynamic_offsets: &[u32],
    ) {
        self.inner.set_bind_group(group_index, bind_group, dynamic_offsets);
    }

    /// Set the vertex buffer for a given slot.
    ///
    /// # Parameters
    ///
    /// * `slot` — The vertex buffer slot index (matches shader `@location`
    ///   or `@attribute` indices).
    /// * `buffer_slice` — A slice view of the vertex buffer (e.g.
    ///   `buffer.slice(..)`).
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: wgpu::BufferSlice<'_>) {
        self.inner.set_vertex_buffer(slot, buffer_slice);
    }

    /// Set the index buffer for indexed drawing.
    ///
    /// # Parameters
    ///
    /// * `buffer_slice` — A slice view of the index buffer.
    /// * `format` — The index format (`Uint16` or `Uint32`).
    pub fn set_index_buffer(
        &mut self,
        buffer_slice: wgpu::BufferSlice<'_>,
        format: wgpu::IndexFormat,
    ) {
        self.inner.set_index_buffer(buffer_slice, format);
    }

    /// Issue a non-indexed draw call.
    ///
    /// # Parameters
    ///
    /// * `vertices` — Range of vertex indices to draw.
    /// * `instances` — Range of instance indices.
    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        self.inner.draw(vertices, instances);
    }
}

// ---------------------------------------------------------------------------
// RhiComputePass
// ---------------------------------------------------------------------------

/// Thin wrapper around a [`wgpu::ComputePass`].
///
/// Created via [`RhiComputePass::begin`] and ended via [`end`](Self::end).
/// While alive, the underlying [`RhiCommandEncoder`] is mutably borrowed.
pub struct RhiComputePass<'a> {
    /// The underlying wgpu compute pass.
    inner: wgpu::ComputePass<'a>,
}

impl<'encoder> RhiComputePass<'encoder> {
    /// Begin a compute pass on the given encoder.
    ///
    /// Borrows the encoder mutably until [`end`](Self::end) is called.
    ///
    /// # Parameters
    ///
    /// * `encoder` — The command encoder to record into.
    /// * `desc` — Descriptor with an optional label and timestamp writes.
    pub fn begin(
        encoder: &'encoder mut RhiCommandEncoder,
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> Self {
        let inner = encoder.inner.begin_compute_pass(desc);
        Self { inner }
    }

    /// End the compute pass.
    ///
    /// Consumes the pass, releasing the mutable borrow on the encoder.
    pub fn end(self) {
        // The inner wgpu::ComputePass is dropped here, ending the pass.
    }

    /// Dispatch a compute shader workgroup grid.
    ///
    /// # Parameters
    ///
    /// * `workgroup_x` — Number of workgroups in the X dimension.
    /// * `workgroup_y` — Number of workgroups in the Y dimension.
    /// * `workgroup_z` — Number of workgroups in the Z dimension.
    pub fn dispatch(&mut self, workgroup_x: u32, workgroup_y: u32, workgroup_z: u32) {
        self.inner.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
    }
}

// ---------------------------------------------------------------------------
// RhiCommandBuffer
// ---------------------------------------------------------------------------

/// A finished command buffer, ready for GPU submission.
///
/// Produced by [`finish_encoder`] and consumed by [`submit`].
pub struct RhiCommandBuffer {
    /// The underlying wgpu command buffer.
    inner: wgpu::CommandBuffer,
}

impl RhiCommandBuffer {
    /// Access the inner [`wgpu::CommandBuffer`] by shared reference.
    pub fn inner(&self) -> &wgpu::CommandBuffer {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Create a new [`RhiCommandEncoder`] from a [`wgpu::Device`].
///
/// The encoder is used to record render passes, compute passes, and copy
/// operations.  After recording is complete, call [`finish_encoder`] to
/// obtain a [`RhiCommandBuffer`] and then [`submit`] to send it to the GPU.
///
/// # Example
///
/// ```ignore
/// let mut encoder = create_command_encoder(&device);
/// ```
pub fn create_command_encoder(device: &wgpu::Device) -> RhiCommandEncoder {
    let inner = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("RHI Command Encoder"),
    });
    RhiCommandEncoder { inner }
}

/// Finalise a command encoder and produce a [`RhiCommandBuffer`].
///
/// Consumes the encoder.  The returned command buffer can be submitted to
/// the GPU queue via [`submit`].
///
/// # Panics
///
/// Panics if there is an outstanding (un-ended) render or compute pass.
pub fn finish_encoder(encoder: RhiCommandEncoder) -> RhiCommandBuffer {
    let inner = encoder.inner.finish();
    RhiCommandBuffer { inner }
}

/// Submit command buffers to the GPU queue via an [`RhiDevice`].
///
/// Consumes the command buffers.  Returns the submission index, which can
/// be used with [`RhiDevice::current_submission_index`] or a [`WgpuFence`]
/// to synchronise CPU-side waits.
///
/// [`WgpuFence`]: crate::rhi_device::WgpuFence
///
/// # Example
///
/// ```ignore
/// let index = submit(&rhi_device, vec![command_buffer]);
/// ```
pub fn submit(device: &RhiDevice, command_buffers: Vec<RhiCommandBuffer>) -> u64 {
    let index = device.next_submission_index();
    let bufs: Vec<wgpu::CommandBuffer> =
        command_buffers.into_iter().map(|cb| cb.inner).collect();
    device.queue().submit(bufs);
    index
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhi_device::{
        create_instance, request_device, FeatureFlags, QualityTier,
    };

    /// Helper: attempt to obtain a test device.  Returns `None` when no GPU
    /// adapter is available (headless / CI).
    fn test_device() -> Option<RhiDevice> {
        let instance = create_instance();
        let adapter = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                }),
        )?;
        Some(request_device(&adapter, FeatureFlags::empty(), QualityTier::Low))
    }

    // -- RhiCommandEncoder + helpers ------------------------------------------

    #[test]
    fn test_create_command_encoder() {
        let Some(device) = test_device() else {
            return;
        };
        let encoder = create_command_encoder(device.device());
        // The encoder should be valid (no panic on finish).
        let _cb = finish_encoder(encoder);
    }

    #[test]
    fn test_create_and_finish_encoder() {
        let Some(device) = test_device() else {
            return;
        };
        let encoder = create_command_encoder(device.device());
        let cb = finish_encoder(encoder);
        // A finished encoder produces a command buffer.
        let _ = cb.inner();
    }

    #[test]
    fn test_encoder_inner_accessors() {
        let Some(device) = test_device() else {
            return;
        };
        let mut encoder = create_command_encoder(device.device());
        // inner() and inner_mut() should both return a reference.
        let _shared: &wgpu::CommandEncoder = encoder.inner();
        let _mutable: &mut wgpu::CommandEncoder = encoder.inner_mut();
        let _cb = finish_encoder(encoder);
    }

    // -- RhiCommandBuffer -----------------------------------------------------

    #[test]
    fn test_command_buffer_inner() {
        let Some(device) = test_device() else {
            return;
        };
        let encoder = create_command_encoder(device.device());
        let cb = finish_encoder(encoder);
        let inner: &wgpu::CommandBuffer = cb.inner();
        let _ = inner;
    }

    // -- RhiRenderPass --------------------------------------------------------

    #[test]
    fn test_render_pass_begin_end() {
        let Some(device) = test_device() else {
            return;
        };
        let dev = device.device();

        // Create a minimal 1x1 render target texture.
        let texture = dev.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-render-target"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = create_command_encoder(dev);

        {
            let mut rp = RhiRenderPass::begin(
                &mut encoder,
                &wgpu::RenderPassDescriptor {
                    label: Some("test-render-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                },
            );

            // Render pass methods should be callable.
            rp.set_bind_group(0, &dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("test-bind-group"),
                layout: &dev.create_bind_group_layout(
                    &wgpu::BindGroupLayoutDescriptor {
                        label: Some("test-bgl"),
                        entries: &[],
                    },
                ),
                entries: &[],
            }), &[]);

            rp.draw(0..0, 0..0);
            rp.end();
        }

        let _cb = finish_encoder(encoder);
    }

    #[test]
    fn test_render_pass_methods_no_panic() {
        let Some(device) = test_device() else {
            return;
        };
        let dev = device.device();

        let texture = dev.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-rt"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = create_command_encoder(dev);

        let mut rp = RhiRenderPass::begin(
            &mut encoder,
            &wgpu::RenderPassDescriptor {
                label: Some("test-methods"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            },
        );

        // All methods should work without panicking.
        let _ = &mut rp; // ensure rp is still alive

        // Minimal vertex/index buffer calls (empty buffers).
        let vb = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-vb"),
            size: 4,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let ib = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-ib"),
            size: 4,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });

        rp.set_vertex_buffer(0, vb.slice(..));
        rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);

        rp.end();
        let _cb = finish_encoder(encoder);
    }

    // -- RhiComputePass -------------------------------------------------------

    #[test]
    fn test_compute_pass_begin_end() {
        let Some(device) = test_device() else {
            return;
        };
        let dev = device.device();

        let mut encoder = create_command_encoder(dev);

        {
            let mut cp = RhiComputePass::begin(
                &mut encoder,
                &wgpu::ComputePassDescriptor {
                    label: Some("test-compute-pass"),
                    timestamp_writes: None,
                },
            );

            cp.dispatch(1, 1, 1);
            cp.end();
        }

        let _cb = finish_encoder(encoder);
    }

    #[test]
    fn test_compute_pass_dispatch_zero() {
        let Some(device) = test_device() else {
            return;
        };
        let dev = device.device();

        let mut encoder = create_command_encoder(dev);

        let mut cp = RhiComputePass::begin(
            &mut encoder,
            &wgpu::ComputePassDescriptor {
                label: Some("zero-dispatch"),
                timestamp_writes: None,
            },
        );

        // Dispatching 0 in any dimension is legal (no-op).
        cp.dispatch(0, 0, 0);
        cp.dispatch(1, 0, 1);
        cp.end();

        let _cb = finish_encoder(encoder);
    }

    // -- submit ---------------------------------------------------------------

    #[test]
    fn test_submit_noop_buffer() {
        let Some(device) = test_device() else {
            return;
        };
        let encoder = create_command_encoder(device.device());
        let cb = finish_encoder(encoder);

        // Submit an empty command buffer -- should not panic.
        let index = submit(&device, vec![cb]);
        // The submission index should advance.
        assert_eq!(index, 0, "first submission index should be 0");

        device.wait_idle();
    }

    #[test]
    fn test_submit_multiple_buffers() {
        let Some(device) = test_device() else {
            return;
        };

        let cb1 = {
            let e = create_command_encoder(device.device());
            finish_encoder(e)
        };
        let cb2 = {
            let e = create_command_encoder(device.device());
            finish_encoder(e)
        };

        // Submit two empty command buffers.
        let _index = submit(&device, vec![cb1, cb2]);
        device.wait_idle();
    }

    #[test]
    fn test_submit_submission_index_advances() {
        let Some(device) = test_device() else {
            return;
        };

        assert_eq!(device.current_submission_index(), 0);

        let idx1 = {
            let e = create_command_encoder(device.device());
            let cb = finish_encoder(e);
            submit(&device, vec![cb])
        };
        assert_eq!(idx1, 0, "first submit returns index 0");
        assert_eq!(device.current_submission_index(), 1);

        let idx2 = {
            let e = create_command_encoder(device.device());
            let cb = finish_encoder(e);
            submit(&device, vec![cb])
        };
        assert_eq!(idx2, 1, "second submit returns index 1");
        assert_eq!(device.current_submission_index(), 2);

        device.wait_idle();
    }

    // -- Integration: render + submit -----------------------------------------

    #[test]
    fn test_render_pass_and_submit() {
        let Some(device) = test_device() else {
            return;
        };
        let dev = device.device();

        let texture = dev.create_texture(&wgpu::TextureDescriptor {
            label: Some("integration-rt"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = create_command_encoder(dev);

        let rp = RhiRenderPass::begin(
            &mut encoder,
            &wgpu::RenderPassDescriptor {
                label: Some("integration-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            },
        );
        rp.end();

        let cb = finish_encoder(encoder);
        let index = submit(&device, vec![cb]);
        assert_eq!(index, 0);

        device.wait_idle();
    }

    #[test]
    fn test_compute_pass_and_submit() {
        let Some(device) = test_device() else {
            return;
        };

        let mut encoder = create_command_encoder(device.device());

        let mut cp = RhiComputePass::begin(
            &mut encoder,
            &wgpu::ComputePassDescriptor {
                label: Some("integration-compute"),
                timestamp_writes: None,
            },
        );
        cp.dispatch(1, 1, 1);
        cp.end();

        let cb = finish_encoder(encoder);
        submit(&device, vec![cb]);
        device.wait_idle();
    }
}
