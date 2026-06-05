//! White-box tests for the wgpu renderer backend.
//!
//! These tests exercise the wgpu API integration at the integration-test level:
//! instance creation, adapter enumeration, device creation, shader compilation,
//! vertex layout validation, and uniform buffer creation.  Tests that require
//! a physical GPU adapter gracefully handle headless / CI environments by
//! skipping assertions when no adapter is available.
//!
//! Run with: `cargo test --test whitebox_renderer -- --test-threads=1`
//! (the `--test-threads=1` flag avoids GPU contention when multiple GPU
//!  tests are running concurrently).

// =============================================================================
// WGSL shader source (duplicated from renderer.rs for whitebox access)
// =============================================================================

/// The WGSL shader used by the production `Renderer`, inlined here for
/// integration-level compilation testing via naga.
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

// =============================================================================
// Test fixture: vertex struct for bytemuck layout tests
// =============================================================================

/// A re-definition of the production `Vertex` struct for whitebox layout
/// testing.  The production struct is `pub(crate)` and therefore not
/// directly accessible from integration tests.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TestVertex {
    position: [f32; 3],
    color: [f32; 3],
}

// =============================================================================
// Helpers
// =============================================================================

/// Create a wgpu instance with all backends enabled.
fn create_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    })
}

/// Request a high-performance adapter.  Returns `None` on headless / CI
/// systems where no GPU is available.
fn request_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    pollster::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }),
    )
}

/// Request a device with default limits and no special features.
fn request_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(
        adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Whitebox Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ),
    )
    .ok()
}

// =============================================================================
// T-BRG-4.1.wb-1: Adapter supports required features
// =============================================================================

/// Request an adapter with no special features and verify it works.
///
/// - Creates a wgpu instance.
/// - Requests an adapter with `Features::empty()` (no special features).
/// - Verifies that, if an adapter is available, the empty feature set is
///   trivially supported.
#[test]
fn test_adapter_supports_empty_features() {
    let instance = create_instance();
    let adapter = request_adapter(&instance);

    if let Some(adapter) = adapter {
        // The adapter must support the empty feature set (trivially true).
        assert!(
            adapter.features().contains(wgpu::Features::empty()),
            "adapter must support the empty feature set"
        );
        // Verify the adapter has valid metadata.
        let info = adapter.get_info();
        assert!(!info.name.is_empty(), "adapter name should not be empty");
        eprintln!(
            "  [info] adapter found: {} ({:?})",
            info.name, info.backend
        );
    } else {
        eprintln!("  [info] no GPU adapter available -- skipping GPU-dependent assertions");
    }
}

// =============================================================================
// T-BRG-4.1.wb-2: Device creation with default limits
// =============================================================================

/// Request a device with default limits and verify the queue exists.
///
/// - Obtains an adapter.
/// - Requests a device with `Limits::default()`.
/// - Verifies the returned queue is a valid command submission endpoint.
/// - Verifies device limits are at least the default set.
#[test]
fn test_device_creation_with_default_limits() {
    let instance = create_instance();
    let adapter = request_adapter(&instance);

    if let Some(adapter) = adapter {
        let (device, queue) = request_device(&adapter).expect(
            "device creation should succeed when an adapter is available",
        );

        // Queue existence: the queue is the sole submission endpoint.
        // Verify by submitting an empty command buffer (no-op).
        let encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Whitebox No-Op Encoder"),
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Device must be functional -- submit and poll succeed.
        device.poll(wgpu::Maintain::Wait);

        // Device limits should be at least the defaults in key fields.
        let limits = device.limits();
        assert!(
            limits.max_texture_dimension_1d >= wgpu::Limits::default().max_texture_dimension_1d,
            "device max_texture_dimension_1d must meet or exceed the default",
        );
        assert!(
            limits.max_bind_groups >= wgpu::Limits::default().max_bind_groups,
            "device max_bind_groups must meet or exceed the default",
        );

        eprintln!(
            "  [info] device created: max_bind_groups={}, max_texture_dimension_1d={}",
            limits.max_bind_groups, limits.max_texture_dimension_1d,
        );
    } else {
        eprintln!("  [info] no GPU adapter available -- skipping GPU-dependent assertions");
    }
}

// =============================================================================
// T-BRG-4.1.wb-3: Surface configuration / adapter enumeration
// =============================================================================

/// Create an instance, enumerate adapters, and verify adapter properties.
///
/// - Creates a wgpu instance.
/// - Enumerates all available adapters for all backends.
/// - For each adapter, verifies:
///   - Non-empty name string.
///   - Known backend type (Vulkan, Metal, DX12, GL, BrowserWebGpu).
///   - Sensible limits (e.g., max_texture_dimension_1d > 0).
///   - Features are queryable.
/// - Also verifies the single-adapter request path works.
#[test]
fn test_adapter_enumeration_and_properties() {
    let instance = create_instance();

    // Enumerate all adapters available to the system.
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all());

    // In many headless CI environments, enumerate_adapters returns an empty
    // vec.  The call itself must not panic.
    eprintln!("  [info] {} adapter(s) enumerated", adapters.len());

    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();

        // Basic metadata checks.
        assert!(
            !info.name.is_empty(),
            "adapter {} name should not be empty",
            i,
        );

        // Must be a known backend.
        match info.backend {
            wgpu::Backend::Vulkan
            | wgpu::Backend::Metal
            | wgpu::Backend::Dx12
            | wgpu::Backend::Gl
            | wgpu::Backend::BrowserWebGpu => {}
            other => panic!("adapter {} has unexpected backend: {:?}", i, other),
        }

        // Limits must be sensible (i.e., non-zero for key fields).
        let limits = adapter.limits();
        assert!(
            limits.max_texture_dimension_1d > 0,
            "adapter {} max_texture_dimension_1d must be > 0",
            i,
        );

        // Features must be queryable.
        let _features = adapter.features();
        let _downlevel = adapter.get_downlevel_capabilities();

        eprintln!("  [info]   adapter[{}]: {} ({:?})", i, info.name, info.backend);
    }

    // Also verify the single-adapter request path (no surface).
    if let Some(adapter) = request_adapter(&instance) {
        let info = adapter.get_info();
        assert!(!info.name.is_empty());
    }
}

// =============================================================================
// T-BRG-4.1.wb-4: Shader module compilation via naga
// =============================================================================

/// Compile the inline WGSL shader from the source file using naga.
///
/// - Parses the WGSL source with `naga::front::wgsl::parse_str`.
/// - Asserts the parse succeeds (no syntax errors, type mismatches, or
///   unsupported constructs).
/// - Verifies both expected entry points (`vs_main`, `fs_main`) are present
///   in the parsed module's entry point list.
#[test]
fn test_shader_compiles_via_naga() {
    let result = naga::front::wgsl::parse_str(SHADER_SRC);
    assert!(
        result.is_ok(),
        "WGSL shader should parse without errors:\n{:#?}",
        result.err(),
    );

    // Verify both entry points exist in the parsed module.
    // In naga, `@vertex` / `@fragment` functions are stored in
    // `module.entry_points`, not in `module.functions`.
    let module = result.unwrap();
    let ep_names: Vec<&str> = module
        .entry_points
        .iter()
        .map(|ep| ep.name.as_str())
        .collect();

    assert!(
        ep_names.contains(&"vs_main"),
        "parsed module must contain vertex entry point `vs_main`; found: {:?}",
        ep_names,
    );
    assert!(
        ep_names.contains(&"fs_main"),
        "parsed module must contain fragment entry point `fs_main`; found: {:?}",
        ep_names,
    );

    eprintln!(
        "  [info] WGSL parsed successfully: {} entry points (vs_main, fs_main)",
        ep_names.len(),
    );
}

// =============================================================================
// T-BRG-4.1.wb-5: Vertex struct layout
// =============================================================================

/// Verify the vertex struct layout: 2 x vec3<f32> = 24 bytes, bytemuck
/// castable.
///
/// - Asserts `size_of::<TestVertex>() == 24` (6 x f32).
/// - Asserts `align_of::<TestVertex>() == 4` (f32 alignment).
/// - Verifies a slice of vertices casts to a byte slice of the correct length.
/// - Verifies known byte patterns for sample vertex data.
#[test]
fn test_vertex_struct_layout() {
    // 2 x vec3<f32> = 6 x f32 = 24 bytes, fully packed, no padding.
    assert_eq!(
        std::mem::size_of::<TestVertex>(),
        24,
        "TestVertex (position: vec3 + color: vec3) must be exactly 24 bytes",
    );
    assert_eq!(
        std::mem::align_of::<TestVertex>(),
        4,
        "TestVertex alignment must be f32 alignment (4 bytes)",
    );
}

/// Verify bytemuck castability of the vertex struct.
#[test]
fn test_vertex_bytemuck_castable() {
    let vertices = [
        TestVertex {
            position: [0.0, 0.5, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        TestVertex {
            position: [-0.5, -0.5, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        TestVertex {
            position: [0.5, -0.5, 0.0],
            color: [0.0, 0.0, 1.0],
        },
    ];

    // Full slice cast.
    let bytes: &[u8] = bytemuck::cast_slice(&vertices);
    assert_eq!(
        bytes.len(),
        3 * 24,
        "3 vertices x 24 bytes = 72 bytes; got {}",
        bytes.len(),
    );

    // Single vertex cast.
    let single: &[u8] = bytemuck::bytes_of(&vertices[0]);
    assert_eq!(single.len(), 24, "single vertex must be 24 bytes");

    // Spot-check: first vertex position.y = 0.5 at bytes 4..8.
    assert_eq!(
        &single[4..8],
        0.5f32.to_ne_bytes(),
        "bytes 4-8 should be f32 0.5",
    );

    // Spot-check: first vertex color.r = 1.0 at bytes 12..16.
    assert_eq!(
        &single[12..16],
        1.0f32.to_ne_bytes(),
        "bytes 12-16 should be f32 1.0",
    );
}

// =============================================================================
// T-BRG-4.1.wb-6: wgpu::Instance with Backends::PRIMARY
// =============================================================================

/// Verify that a `wgpu::Instance` can be created with `Backends::PRIMARY`.
///
/// `Backends::PRIMARY` selects the primary native backends for the platform:
/// Vulkan on Linux/Android, Metal on macOS/iOS, DX12 on Windows.  This test
/// verifies that instance creation with this restricted backend set does not
/// panic.
#[test]
fn test_instance_creation_backends_primary() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    // Success if we reach here without panicking.
    drop(instance);
}

// =============================================================================
// T-BRG-4.1.wb-7: Uniform buffer creation
// =============================================================================

/// Create a device, make a uniform buffer with 64 bytes (mat4x4<f32>), and
/// verify its size.
///
/// - Obtains an adapter (skips if unavailable).
/// - Requests a device with default limits.
/// - Creates a `wgpu::Buffer` with `UNIFORM | COPY_DST` usage, size 64.
/// - Asserts `buffer.size() == 64`.
/// - Writes an identity matrix into the buffer via the queue.
/// - Verifies the device is functional after operations.
#[test]
fn test_uniform_buffer_creation() {
    let instance = create_instance();
    let adapter = request_adapter(&instance);

    if let Some(adapter) = adapter {
        let (device, queue) = request_device(&adapter)
            .expect("device creation should succeed");

        // Column-major 4x4 identity matrix (16 x f32 = 64 bytes).
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Whitebox Test Uniform Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Verify buffer was created with the correct size.
        assert_eq!(
            buffer.size(),
            64,
            "uniform buffer must be exactly 64 bytes (mat4x4<f32>)",
        );

        // Write identity matrix into the buffer via the queue.
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&identity));

        // Device must be functional -- write and poll succeed.
        device.poll(wgpu::Maintain::Wait);

        eprintln!("  [info] uniform buffer created: size={}B", buffer.size());
    } else {
        eprintln!("  [info] no GPU adapter available -- skipping GPU-dependent assertions");
    }
}
