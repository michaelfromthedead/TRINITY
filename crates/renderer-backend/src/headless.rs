//! Headless rendering infrastructure.
//!
//! Provides offscreen rendering capabilities for testing, CI, and server-side
//! rendering scenarios. This module enables rendering to textures without
//! requiring a window or display.

use crate::rhi_device::RhiDevice;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

/// Render target abstraction that supports both surfaces and offscreen textures.
pub enum RenderTarget {
    /// A window surface for presenting to a display.
    Surface(wgpu::Surface<'static>),
    /// An offscreen texture for headless rendering.
    Texture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    },
}

impl RenderTarget {
    /// Create a new texture-based render target for offscreen rendering.
    pub fn new_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Headless Render Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        RenderTarget::Texture {
            texture,
            view,
            width,
            height,
            format,
        }
    }

    /// Get the texture view for rendering.
    pub fn view(&self) -> &wgpu::TextureView {
        match self {
            RenderTarget::Surface(_) => {
                panic!("Cannot get view from Surface target; use get_current_texture() instead")
            }
            RenderTarget::Texture { view, .. } => view,
        }
    }

    /// Get the dimensions of the render target.
    pub fn size(&self) -> (u32, u32) {
        match self {
            RenderTarget::Surface(_) => panic!("Surface size not directly available"),
            RenderTarget::Texture { width, height, .. } => (*width, *height),
        }
    }

    /// Get the texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        match self {
            RenderTarget::Surface(_) => wgpu::TextureFormat::Rgba8Unorm, // typical default
            RenderTarget::Texture { format, .. } => *format,
        }
    }

    /// Read back the rendered pixels from a texture target.
    ///
    /// Returns the raw pixel data as RGBA8 bytes (4 bytes per pixel).
    /// Only valid for `RenderTarget::Texture` variants.
    ///
    /// # Panics
    ///
    /// Panics if called on a `Surface` target.
    pub fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        match self {
            RenderTarget::Surface(_) => {
                panic!("Cannot readback from Surface target")
            }
            RenderTarget::Texture {
                texture,
                width,
                height,
                ..
            } => {
                let bytes_per_pixel = 4u32; // RGBA8
                let unpadded_bytes_per_row = *width * bytes_per_pixel;
                let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
                let buffer_size = padded_bytes_per_row * *height;

                let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Readback Staging Buffer"),
                    size: buffer_size as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Readback Encoder"),
                    });

                encoder.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &staging_buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(padded_bytes_per_row),
                            rows_per_image: Some(*height),
                        },
                    },
                    wgpu::Extent3d {
                        width: *width,
                        height: *height,
                        depth_or_array_layers: 1,
                    },
                );

                queue.submit(std::iter::once(encoder.finish()));

                let buffer_slice = staging_buffer.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
                device.poll(wgpu::Maintain::Wait);
                rx.recv().unwrap().expect("Failed to map staging buffer");

                let data = buffer_slice.get_mapped_range();

                // Remove row padding if present
                let mut pixels = Vec::with_capacity((*width * *height * bytes_per_pixel) as usize);
                for row in 0..*height {
                    let start = (row * padded_bytes_per_row) as usize;
                    let end = start + (unpadded_bytes_per_row as usize);
                    pixels.extend_from_slice(&data[start..end]);
                }

                pixels
            }
        }
    }
}

/// Vertex type for the headless triangle renderer.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

/// Default triangle vertices for testing.
const TRIANGLE_VERTICES: [Vertex; 3] = [
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

/// Inline WGSL shader for headless rendering.
const SHADER_SRC: &str = r#"
struct Uniforms {
    transform: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) fragment_color: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = uniforms.transform * vec4<f32>(input.position, 1.0);
    output.fragment_color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.fragment_color, 1.0);
}
"#;

/// Headless renderer for offscreen rendering.
///
/// This renderer can render to texture targets without requiring a window,
/// making it suitable for testing, CI, and server-side rendering.
pub struct HeadlessRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: RenderTarget,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl HeadlessRenderer {
    /// Create a new headless renderer with the specified dimensions.
    pub fn new(rhi_device: RhiDevice, width: u32, height: u32) -> Self {
        let device = rhi_device.device;
        let queue = rhi_device.queue;
        let format = wgpu::TextureFormat::Rgba8Unorm;

        let target = RenderTarget::new_texture(&device, width, height, format);

        // Create vertex buffer
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Headless Vertex Buffer"),
            contents: bytemuck::cast_slice(&TRIANGLE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create uniform buffer (identity matrix)
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Headless Uniform Buffer"),
            contents: bytemuck::cast_slice(&identity),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Headless Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(64),
                },
                count: None,
            }],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Headless Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: NonZeroU64::new(64),
                }),
            }],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Headless Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Headless Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        // Vertex buffer layout
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                },
            ],
        };

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Headless Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            device,
            queue,
            target,
            render_pipeline,
            vertex_buffer,
            uniform_buffer,
            bind_group,
        }
    }

    /// Render the triangle to the offscreen target.
    pub fn render_triangle(&mut self) {
        let view = self.target.view();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Headless Render Encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Headless Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Read back the rendered pixels from the target texture.
    ///
    /// Returns RGBA8 pixel data (4 bytes per pixel).
    pub fn readback(&self) -> Vec<u8> {
        self.target.readback(&self.device, &self.queue)
    }

    /// Get the render target dimensions.
    pub fn size(&self) -> (u32, u32) {
        self.target.size()
    }

    /// Access the underlying device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access the underlying queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl Drop for HeadlessRenderer {
    fn drop(&mut self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_target_new_texture() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let target = RenderTarget::new_texture(
                &device.device,
                800,
                600,
                wgpu::TextureFormat::Rgba8Unorm,
            );

            assert_eq!(target.size(), (800, 600));
            assert_eq!(target.format(), wgpu::TextureFormat::Rgba8Unorm);
        }
    }

    #[test]
    fn test_headless_renderer_creation() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HeadlessRenderer::new(device, 800, 600);
            assert_eq!(renderer.size(), (800, 600));
        }
    }

    #[test]
    fn test_headless_triangle_render() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HeadlessRenderer::new(device, 800, 600);
            renderer.render_triangle();

            let pixels = renderer.readback();

            // Verify we got the expected number of bytes (800 * 600 * 4 = 1,920,000)
            assert_eq!(pixels.len(), 800 * 600 * 4);

            // Verify we have non-black pixels (the triangle should render something)
            let has_color = pixels.chunks(4).any(|pixel| {
                // Check if any pixel has non-background color
                // Background is (0.1, 0.2, 0.3) = (25, 51, 76) in u8
                pixel[0] > 30 || pixel[1] > 60 || pixel[2] > 80
            });
            assert!(has_color, "Rendered frame should have non-black pixels from the triangle");

            println!(
                "Rendered {}x{} frame, {} bytes",
                800,
                600,
                pixels.len()
            );
        } else {
            println!("Skipping test: no GPU available");
        }
    }

    #[test]
    fn test_readback_pixel_format() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HeadlessRenderer::new(device, 100, 100);
            renderer.render_triangle();

            let pixels = renderer.readback();

            // Each pixel should be 4 bytes (RGBA)
            assert_eq!(pixels.len() % 4, 0);

            // Sample a corner pixel (should be close to clear color)
            // Clear color is (0.1, 0.2, 0.3) * 255 = (25, 51, 76)
            let corner_r = pixels[0];
            let corner_g = pixels[1];
            let corner_b = pixels[2];

            // Allow some tolerance for float->u8 conversion
            assert!(corner_r < 35, "Red channel should be near clear color");
            assert!(corner_g < 65, "Green channel should be near clear color");
            assert!(corner_b < 90, "Blue channel should be near clear color");
        }
    }
}
