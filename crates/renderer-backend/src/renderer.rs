//! Wgpu renderer skeleton with triangle rendering.
//!
//! Provides a [`Renderer`] struct that wraps the core wgpu objects needed
//! for GPU-accelerated rendering: instance, adapter, device, queue, surface,
//! and a render pipeline for a coloured triangle.  This is the runtime entry
//! point for the TRINITY rendering backend.

use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

use crate::component_store::ComponentStore;
use crate::gpu_driven::mesh_table::{MeshTable, MeshTableEntry};

// ---------------------------------------------------------------------------
// Vertex types
// ---------------------------------------------------------------------------

/// A single vertex: position (`vec3<f32>`) + colour (`vec3<f32>`).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) color: [f32; 3],
}

/// Three vertices forming a right-angle triangle with per-vertex colours.
const VERTICES: [Vertex; 3] = [
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

// ---------------------------------------------------------------------------
// Inline WGSL shader
// ---------------------------------------------------------------------------

/// Vertex shader: transforms by a uniform matrix.
/// Fragment shader: outputs the interpolated vertex colour with full alpha.
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

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

/// Core wgpu renderer.
///
/// Owns the full wgpu stack -- instance, adapter, device, queue, surface --
/// together with a minimal pipeline that renders a single coloured triangle.
pub struct Renderer {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub render_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub size: (u32, u32),
    /// Uniform buffer holding the 4x4 transform matrix (identity by default).
    /// Only used indirectly via `bind_group` -- suppress the dead-code lint.
    #[allow(dead_code)]
    uniform_buffer: wgpu::Buffer,
    /// Bind group connecting `uniform_buffer` to shader binding 0.
    bind_group: wgpu::BindGroup,
}

impl Renderer {
    /// Create a new `Renderer` from a window handle.
    ///
    /// Initialises the wgpu runtime, creates a surface, picks an adapter,
    /// builds a device, and sets up a render pipeline for a coloured triangle.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The window handle is invalid.
    /// - No compatible GPU adapter is available.
    /// - Device creation fails.
    pub fn new(
        window: &(impl wgpu::rwh::HasWindowHandle + wgpu::rwh::HasDisplayHandle),
        width: u32,
        height: u32,
    ) -> Self {
        // -- Instance ----------------------------------------------------------
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // -- Surface (via raw handles to get `'static` lifetime) --------------
        let wh = window.window_handle().expect("valid window handle");
        let dh = window.display_handle().expect("valid display handle");
        let surface = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_window_handle: wh.as_raw(),
                    raw_display_handle: dh.as_raw(),
                })
                .expect("surface creation failed")
        };

        // -- Adapter -----------------------------------------------------------
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }),
        )
        .expect("no suitable GPU adapter found");

        // -- Device + Queue ----------------------------------------------------
        let (device, queue) = pollster::block_on(
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("TRINITY Renderer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None, // trace_path
            ),
        )
        .expect("device creation failed");

        // -- Surface configuration --------------------------------------------
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // -- Vertex buffer ----------------------------------------------------
        let vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Triangle Vertex Buffer"),
                contents: bytemuck::cast_slice(&VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });

        // -- Uniform buffer (identity matrix) ---------------------------------
        // Column-major 4x4 identity: {c0, c1, c2, c3}.
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Transform Uniform Buffer"),
                contents: bytemuck::cast_slice(&identity),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // -- Bind group --------------------------------------------------------
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Triangle Bind Group Layout"),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Triangle Bind Group"),
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

        // -- Pipeline layout --------------------------------------------------
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Triangle Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // -- Shader module ----------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Triangle Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        // -- Vertex buffer layout ---------------------------------------------
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

        // -- Render pipeline --------------------------------------------------
        let render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Triangle Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[vertex_layout],
                    compilation_options:
                        wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
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
            instance,
            adapter,
            device,
            queue,
            surface,
            config,
            render_pipeline,
            vertex_buffer,
            uniform_buffer,
            bind_group,
            size: (width, height),
        }
    }

    /// Resize the swap chain to a new resolution.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.size = (width, height);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Render a single frame.
    ///
    /// Acquires the next swap-chain texture, clears it to a dark blue-grey,
    /// draws the triangle, and submits the command buffer.
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
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
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }

    /// Render all renderable entities from an ECS [`ComponentStore`].
    ///
    /// Queries the store for entities with both a Transform and a MeshHandle
    /// component, reads per-entity mesh handles, and would generate draw
    /// commands (currently a placeholder -- collects entity counts without
    /// emitting GPU commands).
    ///
    /// This is the primary integration point between the ECS data channel
    /// (Phase 2) and the wgpu renderer (Phase 4).
    pub fn render_from_store(&mut self, store: &ComponentStore) {
        let renderable = collect_renderable_entities(store);
        // Placeholder: in a full implementation this would:
        //   1. Update per-instance transform data from the store's Transform
        //      components into the GPU instance buffer.
        //   2. For each unique mesh_handle, issue an indirect draw command
        //      referencing the MeshTable entry via the bindless table.
        //   3. Submit the indirect draw command buffer.
        let _count = renderable.len();
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Block until the GPU finishes all outstanding work so that all
        // GPU resources can be safely freed.
        self.device.poll(wgpu::Maintain::Wait);
    }
}

// =========================================================================
// Component ID constants (scene rendering)
// =========================================================================

/// Component type ID for the transform (mat4x4<f32>, 64 bytes).
pub const COMPONENT_ID_TRANSFORM: u32 = 100;

/// Component type ID for the mesh handle (u32 index into MeshTable, 4 bytes).
pub const COMPONENT_ID_MESH_HANDLE: u32 = 101;

/// Byte size of a full 4x4 column-major transform matrix.
pub const TRANSFORM_SIZE: usize = 64;

/// Byte size of a mesh handle (a single u32 pointing into the MeshTable).
pub const MESH_HANDLE_SIZE: usize = 4;

// =========================================================================
// MeshRegistry
// =========================================================================

/// Bindless mesh registry wrapping a [`MeshTable`].
///
/// Provides AssetId (u32) indexed access to mesh entries.  The underlying
/// table can be staged to GPU via [`MeshTable::stage`] for bindless rendering.
///
/// This is the primary CPU-side interface for managing mesh data that will
/// be consumed by GPU shaders through the bindless mesh table.
///
/// # Example
///
/// ```
/// use renderer_backend::gpu_driven::mesh_table::MeshTableEntry;
/// use renderer_backend::renderer::MeshRegistry;
///
/// let mut registry = MeshRegistry::new();
/// let asset_id = registry.register(MeshTableEntry::new(0, 0, 3, 3, 0, 1));
/// assert_eq!(asset_id, 0);
/// assert_eq!(registry.live_count(), 1);
/// ```
pub struct MeshRegistry {
    /// The underlying bindless mesh table.
    table: MeshTable,
}

impl MeshRegistry {
    /// Create a new empty mesh registry with the default capacity.
    pub fn new() -> Self {
        Self {
            table: MeshTable::new(),
        }
    }

    /// Create a new mesh registry with the given initial table capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            table: MeshTable::with_capacity(capacity),
        }
    }

    /// Register a mesh entry and return its AssetId (u32 index into the table).
    ///
    /// The returned AssetId is used by shaders and by [`collect_renderable_entities`]
    /// to reference this mesh.
    pub fn register(&mut self, entry: MeshTableEntry) -> u32 {
        self.table.add(entry)
    }

    /// Get a shared reference to a mesh entry by its AssetId.
    ///
    /// Returns `None` if the AssetId is out of range or the slot is a hole
    /// (previously removed).
    pub fn get(&self, asset_id: u32) -> Option<&MeshTableEntry> {
        let entry = self.table.get(asset_id)?;
        if entry.is_zero() {
            return None;
        }
        Some(entry)
    }

    /// Get a mutable reference to a mesh entry by its AssetId.
    ///
    /// Returns `None` if the AssetId is out of range or the slot is a hole.
    pub fn get_mut(&mut self, asset_id: u32) -> Option<&mut MeshTableEntry> {
        let entry = self.table.get_mut(asset_id)?;
        if entry.is_zero() {
            return None;
        }
        Some(entry)
    }

    /// Remove a mesh entry by its AssetId.
    ///
    /// Returns `true` if the entry was found and removed, `false` if the
    /// AssetId is out of range or already a hole.
    pub fn remove(&mut self, asset_id: u32) -> bool {
        use crate::gpu_driven::RemoveResult;
        matches!(self.table.remove(asset_id), RemoveResult::Removed)
    }

    /// Access the underlying [`MeshTable`] (for GPU staging, etc.).
    pub fn table(&self) -> &MeshTable {
        &self.table
    }

    /// Access the underlying [`MeshTable`] mutably.
    pub fn table_mut(&mut self) -> &mut MeshTable {
        &mut self.table
    }

    /// Total number of entries in the table (including zeroed holes).
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Number of live (non-hole) entries.
    pub fn live_count(&self) -> usize {
        self.table.live_count()
    }

    /// Returns `true` if there are no live entries.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

impl Default for MeshRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// ECS-to-renderer wiring
// =========================================================================

/// Collect renderable entities from a [`ComponentStore`].
///
/// Finds all entities that have both a `Transform` and a `MeshHandle`
/// component, reads each entity's mesh handle from the store, and returns
/// the `(entity_id, mesh_handle)` pairs.
///
/// The returned `mesh_handle` values are indices into the [`MeshTable`]
/// (or equivalently, [`MeshRegistry`]) and can be used to look up the
/// corresponding vertex/index data for drawing.
pub fn collect_renderable_entities(store: &ComponentStore) -> Vec<(u64, u32)> {
    let entities = store.query(&[COMPONENT_ID_TRANSFORM, COMPONENT_ID_MESH_HANDLE]);
    let mut result = Vec::with_capacity(entities.len());

    for entity_id in entities {
        if let Some(mesh_bytes) = store.read_field(entity_id, COMPONENT_ID_MESH_HANDLE, 0, 4) {
            let mesh_handle = u32::from_le_bytes(
                mesh_bytes[..4].try_into().expect("mesh handle must be 4 bytes"),
            );
            result.push((entity_id, mesh_handle));
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component_store::ComponentStore;
    use crate::type_registry::{ComponentTypeInfo, TypeRegistry};
    use std::sync::Arc;

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Create a TypeRegistry with Transform and MeshHandle components
    /// registered, ready for ECS-to-renderer tests.
    fn make_renderable_registry() -> Arc<TypeRegistry> {
        let registry = Arc::new(TypeRegistry::new());

        // Transform: 4x4 matrix (16 f32s = 64 bytes).
        registry.register(ComponentTypeInfo {
            id: COMPONENT_ID_TRANSFORM,
            name: "Transform".into(),
            size: TRANSFORM_SIZE,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });

        // MeshHandle: single u32 (4 bytes).
        registry.register(ComponentTypeInfo {
            id: COMPONENT_ID_MESH_HANDLE,
            name: "MeshHandle".into(),
            size: MESH_HANDLE_SIZE,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });

        registry
    }

    // ── Existing tests (preserved) ───────────────────────────────────────

    #[test]
    fn test_instance_creation() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        // Success if we reach here without panicking.
        let _ = instance;
    }

    #[test]
    fn test_adapter_request() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }),
        );

        // On headless / CI systems the adapter may be `None`; the request
        // itself should not crash.
        if let Some(_adapter) = adapter {
            // Adapter found -- a GPU is available.
        }
    }

    #[test]
    fn test_vertex_struct_size() {
        // The Vertex struct should be exactly 6 × f32 = 24 bytes.
        assert_eq!(std::mem::size_of::<Vertex>(), 24);
        // bytemuck traits must compile.
        let _bytes: &[u8] = bytemuck::cast_slice(&VERTICES);
    }

    #[test]
    fn test_shader_source_parses() {
        let module = naga::front::wgsl::parse_str(SHADER_SRC);
        assert!(
            module.is_ok(),
            "WGSL shader should parse: {:?}",
            module.err()
        );
    }

    // ── T-BRG-5.1: MeshRegistry tests ───────────────────────────────────

    #[test]
    fn test_mesh_registry_new_is_empty() {
        let registry = MeshRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.live_count(), 0);
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_mesh_registry_register_returns_increasing_ids() {
        let mut registry = MeshRegistry::new();
        let a = registry.register(MeshTableEntry::new(0, 0, 3, 3, 0, 1));
        let b = registry.register(MeshTableEntry::new(0, 0, 6, 6, 1, 1));
        let c = registry.register(MeshTableEntry::new(0, 0, 9, 9, 2, 1));
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert_eq!(registry.live_count(), 3);
    }

    #[test]
    fn test_mesh_registry_get_by_asset_id() {
        let mut registry = MeshRegistry::new();
        let id = registry.register(MeshTableEntry::new(0, 0, 100, 50, 3, 1));
        let entry = registry.get(id).expect("entry must exist");
        assert_eq!(entry.index_count, 100);
        assert_eq!(entry.vertex_count, 50);
        assert_eq!(entry.material_id, 3);
        assert_eq!(entry.flags, 1);
    }

    #[test]
    fn test_mesh_registry_get_nonexistent() {
        let registry = MeshRegistry::new();
        assert!(registry.get(0).is_none());
        assert!(registry.get(999).is_none());
    }

    #[test]
    fn test_mesh_registry_remove() {
        let mut registry = MeshRegistry::new();
        let id = registry.register(MeshTableEntry::new(0, 0, 3, 3, 0, 1));
        assert_eq!(registry.live_count(), 1);

        assert!(registry.remove(id));
        assert_eq!(registry.live_count(), 0);
        assert!(registry.get(id).is_none());

        // Remove again should return false (already a hole).
        assert!(!registry.remove(id));
    }

    #[test]
    fn test_mesh_registry_remove_nonexistent() {
        let mut registry = MeshRegistry::new();
        assert!(!registry.remove(0));
        assert!(!registry.remove(999));
    }

    #[test]
    fn test_mesh_registry_get_mut() {
        let mut registry = MeshRegistry::new();
        let id = registry.register(MeshTableEntry::new(0, 0, 3, 3, 0, 1));

        {
            let entry = registry.get_mut(id).unwrap();
            entry.flags = 0xFF;
        }

        assert_eq!(registry.get(id).unwrap().flags, 0xFF);
    }

    #[test]
    fn test_mesh_registry_underlying_table() {
        let mut registry = MeshRegistry::new();
        registry.register(MeshTableEntry::new(0, 0, 3, 3, 0, 1));

        let table = registry.table();
        assert_eq!(table.live_count(), 1);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_mesh_registry_default() {
        let registry: MeshRegistry = Default::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_mesh_registry_with_capacity() {
        let registry = MeshRegistry::with_capacity(2048);
        assert!(registry.is_empty());
        // Capacity is initially 0 for Vec, but reserve ensures room.
        assert!(registry.len() == 0);
    }

    #[test]
    fn test_mesh_registry_get_mut_nonexistent() {
        let mut registry = MeshRegistry::new();
        assert!(registry.get_mut(0).is_none());
    }

    // ── T-BRG-5.3: ECS-to-renderer wiring tests ─────────────────────────

    #[test]
    fn test_collect_renderable_entities_returns_matching() {
        let registry = make_renderable_registry();

        let mut store = ComponentStore::new(registry);

        // Entity 100: has both Transform and MeshHandle -> renderable.
        let mut transform_data = vec![0u8; TRANSFORM_SIZE];
        // Identity matrix (column-major).
        transform_data[0] = 0x80; // Non-zero so we can distinguish.
        transform_data[1] = 0x3F; // f32 1.0 in little-endian.
        let mesh_data: Vec<u8> = 5u32.to_le_bytes().to_vec(); // mesh_handle = 5

        store.spawn(
            100,
            &[COMPONENT_ID_TRANSFORM, COMPONENT_ID_MESH_HANDLE],
            &[
                (COMPONENT_ID_TRANSFORM, transform_data),
                (COMPONENT_ID_MESH_HANDLE, mesh_data),
            ],
        );

        // Entity 200: has only Transform -> not renderable.
        store.spawn(200, &[COMPONENT_ID_TRANSFORM], &[(COMPONENT_ID_TRANSFORM, vec![0u8; TRANSFORM_SIZE])]);

        // Entity 300: has only MeshHandle -> not renderable.
        store.spawn(300, &[COMPONENT_ID_MESH_HANDLE], &[(COMPONENT_ID_MESH_HANDLE, vec![0u8; MESH_HANDLE_SIZE])]);

        let renderable = collect_renderable_entities(&store);
        assert_eq!(renderable.len(), 1, "only entity 100 has both components");
        assert_eq!(renderable[0], (100, 5), "entity 100 has mesh handle 5");
    }

    #[test]
    fn test_collect_renderable_entities_empty_store() {
        let registry = make_renderable_registry();
        let store = ComponentStore::new(registry);
        let renderable = collect_renderable_entities(&store);
        assert!(renderable.is_empty());
    }

    #[test]
    fn test_collect_renderable_entities_multiple_matching() {
        let registry = make_renderable_registry();
        let mut store = ComponentStore::new(registry);

        // Spawn 5 entities all with both components.
        for i in 0..5u64 {
            store.spawn(
                i,
                &[COMPONENT_ID_TRANSFORM, COMPONENT_ID_MESH_HANDLE],
                &[
                    (COMPONENT_ID_TRANSFORM, vec![0u8; TRANSFORM_SIZE]),
                    (COMPONENT_ID_MESH_HANDLE, (i as u32).to_le_bytes().to_vec()),
                ],
            );
        }

        let renderable = collect_renderable_entities(&store);
        assert_eq!(renderable.len(), 5);
        assert!(renderable.contains(&(0, 0)));
        assert!(renderable.contains(&(4, 4)));
    }

    #[test]
    fn test_collect_renderable_entities_excludes_despawned() {
        let registry = make_renderable_registry();
        let mut store = ComponentStore::new(registry);

        store.spawn(
            100,
            &[COMPONENT_ID_TRANSFORM, COMPONENT_ID_MESH_HANDLE],
            &[
                (COMPONENT_ID_TRANSFORM, vec![0u8; TRANSFORM_SIZE]),
                (COMPONENT_ID_MESH_HANDLE, 42u32.to_le_bytes().to_vec()),
            ],
        );
        store.spawn(
            200,
            &[COMPONENT_ID_TRANSFORM, COMPONENT_ID_MESH_HANDLE],
            &[
                (COMPONENT_ID_TRANSFORM, vec![0u8; TRANSFORM_SIZE]),
                (COMPONENT_ID_MESH_HANDLE, 99u32.to_le_bytes().to_vec()),
            ],
        );

        store.despawn(100);

        let renderable = collect_renderable_entities(&store);
        assert_eq!(renderable.len(), 1);
        assert_eq!(renderable[0], (200, 99));
    }

    #[test]
    fn test_collect_renderable_entities_mesh_handle_value() {
        let registry = make_renderable_registry();
        let mut store = ComponentStore::new(registry);

        // Different mesh handles.
        let mesh_ids = [7u32, 13, 42, 255, 65535, u32::MAX];
        for (i, &mesh_id) in mesh_ids.iter().enumerate() {
            store.spawn(
                i as u64,
                &[COMPONENT_ID_TRANSFORM, COMPONENT_ID_MESH_HANDLE],
                &[
                    (COMPONENT_ID_TRANSFORM, vec![0u8; TRANSFORM_SIZE]),
                    (COMPONENT_ID_MESH_HANDLE, mesh_id.to_le_bytes().to_vec()),
                ],
            );
        }

        let renderable = collect_renderable_entities(&store);
        assert_eq!(renderable.len(), mesh_ids.len());

        for (i, expected_mesh_id) in mesh_ids.iter().enumerate() {
            assert_eq!(renderable[i].0, i as u64);
            assert_eq!(renderable[i].1, *expected_mesh_id);
        }
    }
}
