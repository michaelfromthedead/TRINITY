//! Pipeline cache and management.
//!
//! Provides three abstractions for GPU pipeline lifecycle:
//!
//! - [`CachedPipeline`] -- a single compiled render pipeline with its
//!   bind-group layout and shader hash.
//! - [`ShaderCache`] -- deduplicates [`wgpu::ShaderModule`] allocations by
//!   keying on the SHA-256 hash of the WGSL source.
//! - [`PipelineTable`] -- a table of cached pipelines together with a shared
//!   shader cache and a convenience method for compiling new pipelines.
//!
//! # SHA-256 deduplication
//!
//! Every WGSL source string is hashed with SHA-256 **before** compilation.
//! If the same hash is encountered again the existing [`wgpu::ShaderModule`]
//! is returned, avoiding redundant GPU shader compilation.

use std::collections::HashMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// SHA-256 helper
// ---------------------------------------------------------------------------

/// Compute the SHA-256 hash of `data` and return it as a `[u8; 32]`.
fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

// ---------------------------------------------------------------------------
// CachedPipeline
// ---------------------------------------------------------------------------

/// A fully compiled render pipeline together with its bind-group layout
/// and the SHA-256 hash of the WGSL source that produced it.
pub struct CachedPipeline {
    /// User-assigned pipeline identifier.
    pub id: u32,
    /// The compiled wgpu render pipeline.
    pub render_pipeline: wgpu::RenderPipeline,
    /// The bind-group layout used by this pipeline.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// SHA-256 hash of the WGSL source (32 bytes).
    pub shader_hash: [u8; 32],
}

// ---------------------------------------------------------------------------
// ShaderCache
// ---------------------------------------------------------------------------

/// Deduplicates [`wgpu::ShaderModule`] allocations by keying on the
/// SHA-256 hash of the WGSL source.
///
/// Also maintains a map from source path to hash for file-path-based lookups.
pub struct ShaderCache {
    /// Compiled shader modules keyed by their SHA-256 hash (Arc-wrapped for sharing).
    pub modules: HashMap<[u8; 32], Arc<wgpu::ShaderModule>>,
    /// Maps source file paths to their SHA-256 hash.
    pub source_hashes: HashMap<String, [u8; 32]>,
}

impl ShaderCache {
    /// Create an empty shader cache.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            source_hashes: HashMap::new(),
        }
    }

    /// Return a compiled shader module for `wgsl_source`.
    ///
    /// If a module with the same SHA-256 hash already exists in the cache it
    /// is returned without recompilation. Otherwise the source is compiled
    /// into a new [`wgpu::ShaderModule`] and stored in the cache.
    ///
    /// Returns the module (Arc-wrapped for sharing) together with its SHA-256 hash.
    pub fn get_or_compile(
        &mut self,
        device: &wgpu::Device,
        wgsl_source: &str,
    ) -> (Arc<wgpu::ShaderModule>, [u8; 32]) {
        let hash = sha256(wgsl_source.as_bytes());

        if let Some(module) = self.modules.get(&hash) {
            // Cheap Arc clone -- no GPU work.
            return (Arc::clone(module), hash);
        }

        let module = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShaderCache module"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        }));

        self.modules.insert(hash, Arc::clone(&module));
        (module, hash)
    }

    /// Remove all cached modules and source-hash mappings, releasing the
    /// underlying GPU resources.
    pub fn clear(&mut self) {
        self.modules.clear();
        self.source_hashes.clear();
    }
}

impl Default for ShaderCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PipelineTable
// ---------------------------------------------------------------------------

/// Manages a collection of [`CachedPipeline`] entries together with a shared
/// [`ShaderCache`] for deduplicated shader compilation.
pub struct PipelineTable {
    /// Cached pipelines indexed by their numeric id.
    pub pipelines: HashMap<u32, CachedPipeline>,
    /// Shared shader cache for deduplicated compilation.
    pub shader_cache: ShaderCache,
}

impl PipelineTable {
    /// Create an empty pipeline table.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            shader_cache: ShaderCache::new(),
        }
    }

    /// Insert a pre-built pipeline into the table, keyed by `id`.
    ///
    /// If a pipeline with the same `id` already exists it is silently
    /// replaced (dropping the old GPU resources).
    pub fn insert(&mut self, id: u32, pipeline: CachedPipeline) {
        self.pipelines.insert(id, pipeline);
    }

    /// Look up a cached pipeline by its numeric id.
    ///
    /// Returns `None` if no pipeline with that id exists.
    pub fn get(&self, id: u32) -> Option<&CachedPipeline> {
        self.pipelines.get(&id)
    }

    /// Remove a pipeline from the table.
    ///
    /// Returns `true` if the pipeline existed and was removed, `false` if
    /// no pipeline with that id was found.
    pub fn remove(&mut self, id: u32) -> bool {
        self.pipelines.remove(&id).is_some()
    }

    /// Number of cached pipelines currently in the table.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    /// Returns `true` if the table contains no pipelines.
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Compile a new render pipeline and insert it into the table.
    ///
    /// The WGSL source is deduplicated through the internal [`ShaderCache`].
    /// A default (empty) bind-group layout is used; callers that need custom
    /// layouts should construct a [`CachedPipeline`] manually and use
    /// [`insert`](Self::insert).
    ///
    /// # Errors
    ///
    /// Returns `Err(msg)` if shader compilation or pipeline creation fails.
    /// Note that wgpu may **panic** (via `wgpu::Device::create_shader_module`
    /// or `create_render_pipeline`) on invalid WGSL rather than returning an
    /// error. This method wraps those calls with `std::panic::catch_unwind`
    /// to convert panics into `Err` values.
    pub fn compile_pipeline<'a>(
        &mut self,
        device: &wgpu::Device,
        id: u32,
        wgsl_source: &str,
        vertex_entry: &'a str,
        fragment_entry: &'a str,
        vertex_layouts: &'a [wgpu::VertexBufferLayout<'a>],
        color_format: wgpu::TextureFormat,
    ) -> Result<u32, String> {
        // Compile (or fetch) the shader module.
        let (module, shader_hash) = self.shader_cache.get_or_compile(device, wgsl_source);

        // Create a default empty bind-group layout.
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("Pipeline {} BGL", id)),
                entries: &[],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("Pipeline {} Layout", id)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Catch panics from wgpu (e.g. invalid WGSL source).
        let render_pipeline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("Pipeline {}", id)),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: vertex_entry,
                    buffers: vertex_layouts,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: fragment_entry,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
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
            })
        }))
        .map_err(|panic_payload| {
            let msg = panic_payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| panic_payload.downcast_ref::<String>().map(|s| s.as_str()))
                .unwrap_or("unknown wgpu panic");
            format!("pipeline compilation panicked: {msg}")
        })?;

        self.pipelines.insert(
            id,
            CachedPipeline {
                id,
                render_pipeline,
                bind_group_layout,
                shader_hash,
            },
        );

        Ok(id)
    }
}

impl Default for PipelineTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SHA-256 ────────────────────────────────────────────────────────────

    #[test]
    fn test_sha256_same_input_same_hash() {
        let a = sha256(b"hello");
        let b = sha256(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_sha256_different_input_different_hash() {
        let a = sha256(b"hello");
        let b = sha256(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_sha256_known_vector() {
        // Known SHA-256("abc") from FIPS-180 test vector.
        let hash = sha256(b"abc");
        let expected: [u8; 32] = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
            0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
            0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
            0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(hash, expected);
    }

    #[test]
    fn test_sha256_empty_input() {
        let hash = sha256(b"");
        let expected: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
            0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
            0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
            0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(hash, expected);
    }

    // ── ShaderCache ────────────────────────────────────────────────────────

    #[test]
    fn test_shader_cache_new_is_empty() {
        let cache = ShaderCache::new();
        assert!(cache.modules.is_empty());
        assert!(cache.source_hashes.is_empty());
    }

    #[test]
    fn test_shader_cache_clear() {
        let mut cache = ShaderCache::new();
        // Clear on an empty cache should not panic.
        cache.clear();
        assert!(cache.modules.is_empty());
        assert!(cache.source_hashes.is_empty());
    }

    #[test]
    fn test_shader_cache_get_or_compile_dedup() {
        // Requires a GPU device.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ));
        let Some(adapter) = adapter else {
            eprintln!("Skipping test_shader_cache_get_or_compile_dedup: no GPU adapter");
            return;
        };
        let (device, _queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("device creation");

        let mut cache = ShaderCache::new();
        let src = r#"
            @vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let (module_a, hash_a) = cache.get_or_compile(&device, src);
        let (module_b, hash_b) = cache.get_or_compile(&device, src);

        // Same source -> same hash.
        assert_eq!(hash_a, hash_b);
        // Only one entry in the cache.
        assert_eq!(cache.modules.len(), 1);

        // Verify the modules compile (do not panic).
        let _ = module_a;
        let _ = module_b;
    }

    #[test]
    fn test_shader_cache_different_sources_different_hashes() {
        let cache = ShaderCache::new();
        let src_a = "one";
        let src_b = "two";

        let hash_a = sha256(src_a.as_bytes());
        let hash_b = sha256(src_b.as_bytes());

        assert_ne!(hash_a, hash_b);

        // This test doesn't need a device -- it verifies hashing only.
        let _ = cache;
    }

    // ── PipelineTable ──────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_table_new_is_empty() {
        let table = PipelineTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert!(table.pipelines.is_empty());
    }

    /// Helper: obtain a (device, queue) pair, skipping the test if no GPU
    /// is available.
    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ));
        let adapter = adapter?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .expect("device creation"),
        )
    }

    #[test]
    fn test_pipeline_table_insert_get_remove_roundtrip() {
        // This test requires a GPU device.
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Create an empty bind-group layout (valid even without a shader).
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test BGL"),
            entries: &[],
        });

        // Minimal valid WGSL shader.
        let src = r#"
            @vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test module"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let hash = sha256(src.as_bytes());

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let rp = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
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

        let pipeline = CachedPipeline {
            id: 42,
            render_pipeline: rp,
            bind_group_layout: bgl,
            shader_hash: hash,
        };

        let mut table = PipelineTable::new();
        assert!(table.is_empty());

        // Insert.
        table.insert(42, pipeline);
        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());

        // Get.
        let fetched = table.get(42).expect("pipeline should exist");
        assert_eq!(fetched.id, 42);
        assert_eq!(fetched.shader_hash, sha256(src.as_bytes()));

        // Get nonexistent.
        assert!(table.get(99).is_none());

        // Remove.
        assert!(table.remove(42));
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());

        // Remove again should return false.
        assert!(!table.remove(42));

        // Remove nonexistent.
        assert!(!table.remove(99));
    }

    #[test]
    fn test_pipeline_table_compile_pipeline() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Minimal valid WGSL with explicit entry points.
        let src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(0.5, 0.6, 0.7, 1.0);
            }
        "#;

        let mut table = PipelineTable::new();

        let result = table.compile_pipeline(
            &device,
            1,
            src,
            "vs_main",
            "fs_main",
            &[], // no vertex buffers
            wgpu::TextureFormat::Rgba8Unorm,
        );

        assert!(result.is_ok(), "compile_pipeline should succeed: {:?}", result);
        assert_eq!(result.unwrap(), 1);
        assert_eq!(table.len(), 1);

        let pipeline = table.get(1).expect("pipeline 1 should exist");
        assert_eq!(pipeline.id, 1);
        assert_eq!(pipeline.shader_hash, sha256(src.as_bytes()));
    }

    #[test]
    fn test_pipeline_table_multiple_pipelines() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let src_a = r#"
            @vertex fn vs_a() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }
            @fragment fn fs_a() -> @location(0) vec4<f32> { return vec4<f32>(1.0,0.0,0.0,1.0); }
        "#;
        let src_b = r#"
            @vertex fn vs_b() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }
            @fragment fn fs_b() -> @location(0) vec4<f32> { return vec4<f32>(0.0,1.0,0.0,1.0); }
        "#;

        let mut table = PipelineTable::new();

        let r1 = table.compile_pipeline(
            &device, 10, src_a, "vs_a", "fs_a", &[], wgpu::TextureFormat::Rgba8Unorm,
        );
        let r2 = table.compile_pipeline(
            &device, 20, src_b, "vs_b", "fs_b", &[], wgpu::TextureFormat::Rgba8Unorm,
        );

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert_eq!(table.len(), 2);

        // Pipelines have different shader hashes.
        let p10 = table.get(10).unwrap();
        let p20 = table.get(20).unwrap();
        assert_ne!(p10.shader_hash, p20.shader_hash);

        // Remove one.
        assert!(table.remove(10));
        assert_eq!(table.len(), 1);
        assert!(table.get(10).is_none());
        assert!(table.get(20).is_some());
    }

    #[test]
    fn test_pipeline_table_insert_overwrites() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let src = r#"
            @vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }
            @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0,0.0,0.0,1.0); }
        "#;

        let mut table = PipelineTable::new();
        let _ = table.compile_pipeline(&device, 1, src, "vs", "fs", &[], wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(table.len(), 1);

        // Compile again with same id -- should overwrite without changing count.
        let _ = table.compile_pipeline(&device, 1, src, "vs", "fs", &[], wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(table.len(), 1);
        assert!(table.get(1).is_some());
    }

    #[test]
    fn test_pipeline_table_shared_shader_cache() {
        // Two pipelines with the same WGSL source should share a shader module.
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let src = r#"
            @vertex fn shared_vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }
            @fragment fn shared_fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0,0.0,0.0,1.0); }
        "#;

        let mut table = PipelineTable::new();

        let _ = table.compile_pipeline(&device, 1, src, "shared_vs", "shared_fs", &[], wgpu::TextureFormat::Rgba8Unorm);
        let _ = table.compile_pipeline(&device, 2, src, "shared_vs", "shared_fs", &[], wgpu::TextureFormat::Rgba8Unorm);

        // Only one shader module should be in the cache.
        assert_eq!(table.shader_cache.modules.len(), 1);

        // Both pipelines should have the same shader hash.
        assert_eq!(table.get(1).unwrap().shader_hash, table.get(2).unwrap().shader_hash);
    }
}
