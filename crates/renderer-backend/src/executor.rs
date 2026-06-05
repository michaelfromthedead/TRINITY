//! Frame Graph Executor.
//!
//! Executes a compiled frame graph on the GPU by:
//! 1. Allocating resources (textures and buffers) from the IR descriptions
//! 2. Recording GPU commands for each pass in topological order
//! 3. Submitting commands to the queue

use crate::frame_graph::{
    AttachmentLoadOp, AttachmentStoreOp, ColorAttachment, CompiledFrameGraph,
    DepthStencilAttachment, DispatchSource, IrPass, IrResource, PassType, ResourceDesc,
    ResourceHandle, TextureDesc,
};
use std::collections::HashMap;

/// Executes a compiled frame graph on the GPU.
pub struct FrameGraphExecutor<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    compiled: &'a CompiledFrameGraph,

    /// Allocated texture views indexed by ResourceHandle.
    textures: HashMap<ResourceHandle, AllocatedTexture>,
    /// Allocated buffers indexed by ResourceHandle.
    buffers: HashMap<ResourceHandle, wgpu::Buffer>,
}

/// An allocated GPU texture with its view.
struct AllocatedTexture {
    #[allow(dead_code)]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
}

impl<'a> FrameGraphExecutor<'a> {
    /// Create a new executor for a compiled frame graph.
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        compiled: &'a CompiledFrameGraph,
    ) -> Self {
        Self {
            device,
            queue,
            compiled,
            textures: HashMap::new(),
            buffers: HashMap::new(),
        }
    }

    /// Allocate all GPU resources described in the frame graph.
    pub fn allocate_resources(&mut self) {
        for resource in &self.compiled.resources {
            self.allocate_resource(resource);
        }
    }

    /// Allocate a single resource.
    fn allocate_resource(&mut self, resource: &IrResource) {
        match &resource.desc {
            ResourceDesc::Texture2D(desc) => {
                let format = parse_texture_format(&desc.format);
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&resource.name),
                    size: wgpu::Extent3d {
                        width: desc.width,
                        height: desc.height,
                        depth_or_array_layers: desc.array_layers.max(1),
                    },
                    mip_level_count: desc.mip_levels.max(1),
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                self.textures.insert(
                    resource.handle,
                    AllocatedTexture {
                        texture,
                        view,
                        format,
                    },
                );
            }
            ResourceDesc::Texture3D(desc) => {
                let format = wgpu::TextureFormat::Rgba8Unorm; // Default for 3D
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&resource.name),
                    size: wgpu::Extent3d {
                        width: desc.width,
                        height: desc.height,
                        depth_or_array_layers: desc.depth,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                self.textures.insert(
                    resource.handle,
                    AllocatedTexture {
                        texture,
                        view,
                        format,
                    },
                );
            }
            ResourceDesc::Buffer(desc) => {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&resource.name),
                    size: desc.size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.buffers.insert(resource.handle, buffer);
            }
            ResourceDesc::TextureCube(desc) => {
                // Cube maps are stored as 2D arrays with 6 layers
                let format = parse_texture_format(&desc.format);
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&resource.name),
                    size: wgpu::Extent3d {
                        width: desc.width,
                        height: desc.height,
                        depth_or_array_layers: 6,
                    },
                    mip_level_count: desc.mip_levels.max(1),
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                    ..Default::default()
                });
                self.textures.insert(
                    resource.handle,
                    AllocatedTexture {
                        texture,
                        view,
                        format,
                    },
                );
            }
        }
    }

    /// Execute the frame graph, rendering to the provided output texture.
    pub fn execute(&self, output: &wgpu::TextureView) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Graph Executor"),
            });

        // Execute passes in topological order
        for pass_idx in &self.compiled.order {
            if let Some(pass) = self.compiled.passes.iter().find(|p| p.index == *pass_idx) {
                self.execute_pass(&mut encoder, pass, output);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Execute a single pass.
    fn execute_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass: &IrPass,
        output: &wgpu::TextureView,
    ) {
        match pass.pass_type {
            PassType::Graphics => self.execute_graphics_pass(encoder, pass, output),
            PassType::Compute => self.execute_compute_pass(encoder, pass),
            PassType::Copy => self.execute_copy_pass(encoder, pass),
            PassType::RayTracing => {
                // Ray tracing not yet supported in wgpu stable
            }
        }
    }

    /// Execute a graphics render pass.
    fn execute_graphics_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass: &IrPass,
        output: &wgpu::TextureView,
    ) {
        // Build depth-stencil attachment first (needs to outlive render pass)
        let depth_stencil_attachment = pass.depth_stencil.as_ref().and_then(|ds| {
            let allocated = self.textures.get(&ds.resource)?;
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: &allocated.view,
                depth_ops: Some(wgpu::Operations {
                    load: convert_depth_load_op(ds.depth_load_op, ds.clear_depth),
                    store: convert_store_op(ds.depth_store_op),
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: convert_stencil_load_op(ds.stencil_load_op, ds.clear_stencil),
                    store: convert_store_op(ds.stencil_store_op),
                }),
            })
        });

        // Build color attachments
        if pass.color_attachments.is_empty() {
            // Default: render to output with clear
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: output,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];

            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&pass.name),
                color_attachments: &color_attachments,
                depth_stencil_attachment,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            // TODO: Bind pipeline, set bind groups, draw
        } else {
            // Build color attachments with proper lifetimes
            // We need to collect views first to ensure they live long enough
            let views: Vec<&wgpu::TextureView> = pass
                .color_attachments
                .iter()
                .map(|ca| {
                    if ca.resource == ResourceHandle::NONE {
                        output
                    } else {
                        self.textures
                            .get(&ca.resource)
                            .map(|t| &t.view)
                            .unwrap_or(output)
                    }
                })
                .collect();

            let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = pass
                .color_attachments
                .iter()
                .zip(views.iter())
                .map(|(ca, view)| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: convert_load_op(ca.load_op, ca.clear_color),
                            store: convert_store_op(ca.store_op),
                        },
                    })
                })
                .collect();

            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&pass.name),
                color_attachments: &color_attachments,
                depth_stencil_attachment,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            // TODO: Bind pipeline, set bind groups, draw
        }
    }

    /// Execute a compute pass.
    fn execute_compute_pass(&self, encoder: &mut wgpu::CommandEncoder, pass: &IrPass) {
        {
            let _cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&pass.name),
                timestamp_writes: None,
            });

            // TODO: Bind compute pipeline, set bind groups, dispatch
            if let Some(dispatch) = &pass.dispatch_source {
                match dispatch {
                    DispatchSource::Direct {
                        group_count_x,
                        group_count_y,
                        group_count_z,
                    } => {
                        // _cpass.dispatch_workgroups(*group_count_x, *group_count_y, *group_count_z);
                        let _ = (*group_count_x, *group_count_y, *group_count_z); // Placeholder
                    }
                    DispatchSource::Indirect { buffer: _, offset: _ } => {
                        // Indirect dispatch from buffer
                    }
                }
            }
        }
    }

    /// Execute a copy pass.
    fn execute_copy_pass(&self, encoder: &mut wgpu::CommandEncoder, pass: &IrPass) {
        // Copy passes read source and write destination from access_set
        // For now, this is a placeholder
        let _ = encoder;
        let _ = pass;
    }

    /// Get the allocated texture view for a resource handle.
    pub fn get_texture_view(&self, handle: ResourceHandle) -> Option<&wgpu::TextureView> {
        self.textures.get(&handle).map(|t| &t.view)
    }

    /// Get the allocated buffer for a resource handle.
    pub fn get_buffer(&self, handle: ResourceHandle) -> Option<&wgpu::Buffer> {
        self.buffers.get(&handle)
    }

    /// Get the number of passes that will be executed.
    pub fn pass_count(&self) -> usize {
        self.compiled.order.len()
    }

    /// Get the names of passes in execution order.
    pub fn pass_names(&self) -> Vec<&str> {
        self.compiled
            .order
            .iter()
            .filter_map(|idx| {
                self.compiled
                    .passes
                    .iter()
                    .find(|p| p.index == *idx)
                    .map(|p| p.name.as_str())
            })
            .collect()
    }
}

/// Parse a texture format string to wgpu::TextureFormat.
fn parse_texture_format(format: &str) -> wgpu::TextureFormat {
    match format.to_lowercase().as_str() {
        "rgba8unorm" | "rgba8" => wgpu::TextureFormat::Rgba8Unorm,
        "rgba8snorm" => wgpu::TextureFormat::Rgba8Snorm,
        "rgba8uint" => wgpu::TextureFormat::Rgba8Uint,
        "rgba8sint" => wgpu::TextureFormat::Rgba8Sint,
        "bgra8unorm" | "bgra8" => wgpu::TextureFormat::Bgra8Unorm,
        "bgra8unorm-srgb" => wgpu::TextureFormat::Bgra8UnormSrgb,
        "rgba16float" | "rgba16f" => wgpu::TextureFormat::Rgba16Float,
        "rgba32float" | "rgba32f" => wgpu::TextureFormat::Rgba32Float,
        "r8unorm" | "r8" => wgpu::TextureFormat::R8Unorm,
        "r16float" | "r16f" => wgpu::TextureFormat::R16Float,
        "r32float" | "r32f" => wgpu::TextureFormat::R32Float,
        "rg8unorm" | "rg8" => wgpu::TextureFormat::Rg8Unorm,
        "rg16float" | "rg16f" => wgpu::TextureFormat::Rg16Float,
        "rg32float" | "rg32f" => wgpu::TextureFormat::Rg32Float,
        "depth32float" | "depth32" => wgpu::TextureFormat::Depth32Float,
        "depth24plus" => wgpu::TextureFormat::Depth24Plus,
        "depth24plus-stencil8" => wgpu::TextureFormat::Depth24PlusStencil8,
        "depth32float-stencil8" => wgpu::TextureFormat::Depth32FloatStencil8,
        _ => wgpu::TextureFormat::Rgba8Unorm, // Default fallback
    }
}

/// Convert IR load op to wgpu load op for color attachments.
fn convert_load_op(op: AttachmentLoadOp, clear_color: [f32; 4]) -> wgpu::LoadOp<wgpu::Color> {
    match op {
        AttachmentLoadOp::Load => wgpu::LoadOp::Load,
        AttachmentLoadOp::Clear => wgpu::LoadOp::Clear(wgpu::Color {
            r: clear_color[0] as f64,
            g: clear_color[1] as f64,
            b: clear_color[2] as f64,
            a: clear_color[3] as f64,
        }),
        AttachmentLoadOp::DontCare => wgpu::LoadOp::Clear(wgpu::Color::BLACK),
    }
}

/// Convert IR load op to wgpu load op for depth attachments.
fn convert_depth_load_op(op: AttachmentLoadOp, clear_depth: f32) -> wgpu::LoadOp<f32> {
    match op {
        AttachmentLoadOp::Load => wgpu::LoadOp::Load,
        AttachmentLoadOp::Clear => wgpu::LoadOp::Clear(clear_depth),
        AttachmentLoadOp::DontCare => wgpu::LoadOp::Clear(1.0),
    }
}

/// Convert IR load op to wgpu load op for stencil attachments.
fn convert_stencil_load_op(op: AttachmentLoadOp, clear_stencil: u32) -> wgpu::LoadOp<u32> {
    match op {
        AttachmentLoadOp::Load => wgpu::LoadOp::Load,
        AttachmentLoadOp::Clear => wgpu::LoadOp::Clear(clear_stencil),
        AttachmentLoadOp::DontCare => wgpu::LoadOp::Clear(0),
    }
}

/// Convert IR store op to wgpu store op.
fn convert_store_op(op: AttachmentStoreOp) -> wgpu::StoreOp {
    match op {
        AttachmentStoreOp::Store => wgpu::StoreOp::Store,
        AttachmentStoreOp::DontCare => wgpu::StoreOp::Discard,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::{PassIndex, ResourceLifetime, ResourceState, TextureDesc};
    use crate::rhi_device::RhiDevice;

    fn make_simple_frame_graph() -> CompiledFrameGraph {
        let pass = IrPass::graphics(
            PassIndex(0),
            "main",
            vec![],
            None,
            crate::frame_graph::InstanceSource::Direct {
                index_count: 3,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            crate::frame_graph::ViewType::ColorAttachment,
        );

        let resource = IrResource {
            handle: ResourceHandle(0),
            name: "output".into(),
            desc: ResourceDesc::Texture2D(TextureDesc {
                width: 800,
                height: 600,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            lifetime: ResourceLifetime::Transient,
            initial_state: ResourceState::Uninitialized,
            view_format_override: None,
        };

        CompiledFrameGraph {
            passes: vec![pass],
            resources: vec![resource],
            edges: vec![],
            order: vec![PassIndex(0)],
            depths: std::collections::HashMap::new(),
            barriers: vec![],
            async_passes: vec![],
            async_timeline: Some(vec![]),
            eliminated_passes: vec![],
            cull_stats: crate::frame_graph::CullStats::default(),
            parallel_regions: vec![vec![PassIndex(0)]],
            stats: crate::frame_graph::CompilerStats::default(),
            compilation_time_us: 0,
            perf_counters: crate::frame_graph::PerfCounters::default(),
            sync_points: Vec::new(),
        }
    }

    #[test]
    fn test_parse_texture_format() {
        assert_eq!(
            parse_texture_format("rgba8unorm"),
            wgpu::TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            parse_texture_format("RGBA8"),
            wgpu::TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            parse_texture_format("depth32float"),
            wgpu::TextureFormat::Depth32Float
        );
        assert_eq!(
            parse_texture_format("bgra8unorm-srgb"),
            wgpu::TextureFormat::Bgra8UnormSrgb
        );
    }

    #[test]
    fn test_executor_creation() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let compiled = make_simple_frame_graph();
            let executor = FrameGraphExecutor::new(&device.device, &device.queue, &compiled);

            assert_eq!(executor.pass_count(), 1);
            assert_eq!(executor.pass_names(), vec!["main"]);
        }
    }

    #[test]
    fn test_resource_allocation() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let compiled = make_simple_frame_graph();
            let mut executor = FrameGraphExecutor::new(&device.device, &device.queue, &compiled);

            executor.allocate_resources();

            // Should have allocated the texture
            assert!(executor.get_texture_view(ResourceHandle(0)).is_some());
        }
    }

    #[test]
    fn test_frame_graph_execution() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let compiled = make_simple_frame_graph();
            let mut executor = FrameGraphExecutor::new(&device.device, &device.queue, &compiled);

            executor.allocate_resources();

            // Create output texture
            let output = device.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Output"),
                size: wgpu::Extent3d {
                    width: 800,
                    height: 600,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());

            // Execute should not panic
            executor.execute(&output_view);

            println!("Executed {} passes: {:?}", executor.pass_count(), executor.pass_names());
        } else {
            println!("Skipping test: no GPU available");
        }
    }

    #[test]
    fn test_multi_pass_execution() {
        if let Some(device) = RhiDevice::try_new_headless() {
            // Create a multi-pass frame graph
            let shadow_pass = IrPass::graphics(
                PassIndex(0),
                "shadow",
                vec![],
                Some(DepthStencilAttachment {
                    resource: ResourceHandle(1),
                    depth_load_op: AttachmentLoadOp::Clear,
                    depth_store_op: AttachmentStoreOp::Store,
                    stencil_load_op: AttachmentLoadOp::DontCare,
                    stencil_store_op: AttachmentStoreOp::DontCare,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                    depth_test_enabled: true,
                    depth_write_enabled: true,
                }),
                crate::frame_graph::InstanceSource::Direct {
                    index_count: 0,
                    instance_count: 1,
                    base_vertex: 0,
                    first_index: 0,
                    first_instance: 0,
                },
                crate::frame_graph::ViewType::ColorAttachment,
            );

            let main_pass = IrPass::graphics(
                PassIndex(1),
                "main",
                vec![ColorAttachment {
                    resource: ResourceHandle(0),
                    mip_level: 0,
                    array_layer: 0,
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_color: [0.1, 0.2, 0.3, 1.0],
                }],
                None,
                crate::frame_graph::InstanceSource::Direct {
                    index_count: 3,
                    instance_count: 1,
                    base_vertex: 0,
                    first_index: 0,
                    first_instance: 0,
                },
                crate::frame_graph::ViewType::ColorAttachment,
            );

            let color_resource = IrResource {
                handle: ResourceHandle(0),
                name: "color".into(),
                desc: ResourceDesc::Texture2D(TextureDesc {
                    width: 800,
                    height: 600,
                    mip_levels: 1,
                    array_layers: 1,
                    format: "rgba8unorm".into(),
                }),
                lifetime: ResourceLifetime::Transient,
                initial_state: ResourceState::Uninitialized,
                view_format_override: None,
            };

            let depth_resource = IrResource {
                handle: ResourceHandle(1),
                name: "depth".into(),
                desc: ResourceDesc::Texture2D(TextureDesc {
                    width: 2048,
                    height: 2048,
                    mip_levels: 1,
                    array_layers: 1,
                    format: "depth32float".into(),
                }),
                lifetime: ResourceLifetime::Transient,
                initial_state: ResourceState::Uninitialized,
                view_format_override: None,
            };

            let compiled = CompiledFrameGraph {
                passes: vec![shadow_pass, main_pass],
                resources: vec![color_resource, depth_resource],
                edges: vec![],
                order: vec![PassIndex(0), PassIndex(1)],
                depths: std::collections::HashMap::new(),
                barriers: vec![],
                async_passes: vec![],
                async_timeline: Some(vec![]),
                eliminated_passes: vec![],
                cull_stats: crate::frame_graph::CullStats::default(),
                parallel_regions: vec![vec![PassIndex(0)], vec![PassIndex(1)]],
                stats: crate::frame_graph::CompilerStats::default(),
                compilation_time_us: 0,
                perf_counters: crate::frame_graph::PerfCounters::default(),
                sync_points: Vec::new(),
            };

            let mut executor = FrameGraphExecutor::new(&device.device, &device.queue, &compiled);
            executor.allocate_resources();

            // Create output texture
            let output = device.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Output"),
                size: wgpu::Extent3d {
                    width: 800,
                    height: 600,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());

            executor.execute(&output_view);

            println!(
                "Executed {} passes: {:?}",
                executor.pass_count(),
                executor.pass_names()
            );
        } else {
            println!("Skipping test: no GPU available");
        }
    }
}
