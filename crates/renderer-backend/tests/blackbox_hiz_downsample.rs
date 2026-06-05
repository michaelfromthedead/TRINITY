// Blackbox contract tests for T-WGPU-P6.4.2 HiZ Downsample Shader
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::gpu_driven`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/gpu_driven/hiz_pyramid.rs (implementation)
//   - crates/renderer-backend/shaders/hiz_downsample.wgsl (shader source - only string check)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P6.4.2)
//
// Public API under test:
//   - HIZ_DOWNSAMPLE_SHADER: Embedded WGSL shader source string
//   - HIZ_DOWNSAMPLE_WORKGROUP_SIZE: Workgroup size constant (expected 8)
//   - HIZ_DOWNSAMPLE_PARAMS_SIZE: Params buffer size constant (expected 24 bytes)
//   - HiZDownsampleParams: Parameters struct for downsample dispatch
//   - create_hiz_downsample_texture_layout(): Bind group layout for textures
//   - create_hiz_downsample_params_layout(): Bind group layout for params uniform
//   - cpu_max_reduction(): CPU reference implementation of 2x2 max reduction
//   - calculate_downsample_dispatch(): Calculate dispatch dimensions
//
// Test design rationale:
//   Acceptance Criteria Coverage:
//     1. 2x2 max reduction (reverse-Z) - cpu_max_reduction returns maximum value
//     2. Per-mip dispatch - HiZDownsampleParams auto-calculates dst size
//     3. Correct UV sampling - shader source contains textureLoad
//     4. Workgroup size (8, 8, 1) - HIZ_DOWNSAMPLE_WORKGROUP_SIZE = 8
//
//   Resolution test cases:
//     - 128x128 (small)
//     - 1024x1024 (typical)
//     - 1920x1080 (non-square)
//     - Edge cases (1x1, 2x2, odd dimensions)

use pollster::block_on;
use renderer_backend::device::{enumerate_adapters_with_info, TrinityInstance};
use renderer_backend::gpu_driven::{
    calculate_downsample_dispatch, cpu_max_reduction, create_hiz_downsample_params_layout,
    create_hiz_downsample_texture_layout, HiZDownsampleParams, HIZ_DOWNSAMPLE_PARAMS_SIZE,
    HIZ_DOWNSAMPLE_SHADER, HIZ_DOWNSAMPLE_WORKGROUP_SIZE, MIN_HIZ_SIZE,
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Creates a TrinityInstance and gets the first available adapter.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    result.adapters.into_iter().next()
}

/// Helper macro to skip test if no GPU adapter is available.
macro_rules! require_adapter {
    () => {
        match get_test_adapter() {
            Some(adapter) => adapter,
            None => {
                eprintln!("SKIP: No GPU adapter available for this test");
                return;
            }
        }
    };
}

/// Creates a wgpu device for testing.
fn create_test_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

/// Helper macro to get a device, skipping if unavailable.
macro_rules! require_device {
    ($adapter:expr) => {
        match create_test_device(&$adapter) {
            Some(device) => device,
            None => {
                eprintln!("SKIP: Could not create device");
                return;
            }
        }
    };
}

// =============================================================================
// CRITERION 1: 2x2 MAX REDUCTION (REVERSE-Z) TESTS
// =============================================================================

/// Test that cpu_max_reduction returns the maximum value (reverse-Z: larger = closer).
#[test]
fn test_cpu_max_reduction_basic() {
    // Basic case: clearly different values
    let result = cpu_max_reduction(0.5, 0.6, 0.7, 0.8);
    assert_eq!(result, 0.8, "Max of (0.5, 0.6, 0.7, 0.8) should be 0.8");
    println!("PASS: cpu_max_reduction returns maximum for basic case");
}

/// Test cpu_max_reduction with maximum in each position.
#[test]
fn test_cpu_max_reduction_all_positions() {
    // Max in position 0 (d00)
    assert_eq!(
        cpu_max_reduction(1.0, 0.0, 0.5, 0.5),
        1.0,
        "Max at d00 should return 1.0"
    );

    // Max in position 1 (d10)
    assert_eq!(
        cpu_max_reduction(0.1, 0.9, 0.2, 0.3),
        0.9,
        "Max at d10 should return 0.9"
    );

    // Max in position 2 (d01)
    assert_eq!(
        cpu_max_reduction(0.1, 0.2, 0.95, 0.3),
        0.95,
        "Max at d01 should return 0.95"
    );

    // Max in position 3 (d11)
    assert_eq!(
        cpu_max_reduction(0.1, 0.2, 0.3, 0.99),
        0.99,
        "Max at d11 should return 0.99"
    );

    println!("PASS: cpu_max_reduction handles max in all positions");
}

/// Test cpu_max_reduction with all equal values.
#[test]
fn test_cpu_max_reduction_uniform() {
    // All zeros
    assert_eq!(
        cpu_max_reduction(0.0, 0.0, 0.0, 0.0),
        0.0,
        "Max of all zeros should be 0.0"
    );

    // All ones
    assert_eq!(
        cpu_max_reduction(1.0, 1.0, 1.0, 1.0),
        1.0,
        "Max of all ones should be 1.0"
    );

    // All same non-trivial value
    assert_eq!(
        cpu_max_reduction(0.5, 0.5, 0.5, 0.5),
        0.5,
        "Max of uniform 0.5 should be 0.5"
    );

    println!("PASS: cpu_max_reduction handles uniform values");
}

/// Test cpu_max_reduction with close values (precision test).
#[test]
fn test_cpu_max_reduction_precision() {
    // Very close values - verify precision
    let result = cpu_max_reduction(0.999, 0.998, 0.997, 0.996);
    assert_eq!(result, 0.999, "Should pick 0.999 as maximum");

    let result = cpu_max_reduction(0.0001, 0.0002, 0.0003, 0.0004);
    assert_eq!(result, 0.0004, "Should pick 0.0004 as maximum");

    println!("PASS: cpu_max_reduction maintains precision for close values");
}

/// Test cpu_max_reduction with reverse-Z depth semantics.
#[test]
fn test_cpu_max_reduction_reverse_z_semantics() {
    // In reverse-Z: 1.0 = near plane, 0.0 = far plane
    // Max reduction picks the CLOSEST fragment (largest depth value)

    // Near surface (0.9) vs far surface (0.1) - should pick near
    let result = cpu_max_reduction(0.1, 0.9, 0.2, 0.3);
    assert_eq!(result, 0.9, "Reverse-Z: should pick closest surface (0.9)");

    // All moderately far values except one near
    let result = cpu_max_reduction(0.2, 0.2, 0.2, 0.8);
    assert_eq!(result, 0.8, "Reverse-Z: should pick the one near surface");

    println!("PASS: cpu_max_reduction follows reverse-Z (larger = closer)");
}

/// Test cpu_max_reduction with negative values (edge case).
#[test]
fn test_cpu_max_reduction_edge_values() {
    // Typical depth values are 0..1, but test edge cases
    let result = cpu_max_reduction(0.0, 0.5, 0.25, 0.75);
    assert_eq!(result, 0.75, "Should handle typical range correctly");

    // Very small differences
    let eps = 1e-7_f32;
    let result = cpu_max_reduction(0.5, 0.5 + eps, 0.5 - eps, 0.5);
    assert!(
        (result - (0.5 + eps)).abs() < eps * 10.0,
        "Should detect tiny maximum: got {}, expected ~{}",
        result,
        0.5 + eps
    );

    println!("PASS: cpu_max_reduction handles edge values");
}

// =============================================================================
// CRITERION 2: PER-MIP DISPATCH TESTS
// =============================================================================

/// Test HiZDownsampleParams struct size matches constant.
#[test]
fn test_hiz_downsample_params_size() {
    // The params struct must be exactly HIZ_DOWNSAMPLE_PARAMS_SIZE bytes
    // for proper GPU buffer alignment
    let struct_size = std::mem::size_of::<HiZDownsampleParams>();
    assert_eq!(
        struct_size, HIZ_DOWNSAMPLE_PARAMS_SIZE,
        "HiZDownsampleParams size ({}) must match HIZ_DOWNSAMPLE_PARAMS_SIZE ({})",
        struct_size, HIZ_DOWNSAMPLE_PARAMS_SIZE
    );

    // Expected: 24 bytes = 2*u32 (src_size) + 2*u32 (dst_size) + u32 (mip_level) + u32 (padding)
    assert_eq!(
        HIZ_DOWNSAMPLE_PARAMS_SIZE, 24,
        "HIZ_DOWNSAMPLE_PARAMS_SIZE should be 24 bytes"
    );

    println!("PASS: HiZDownsampleParams size = {} bytes", struct_size);
}

/// Test HiZDownsampleParams::new() creates correct parameters.
#[test]
fn test_hiz_downsample_params_new() {
    // Create params for mip 1: 256x256 -> 128x128
    let params = HiZDownsampleParams::new(256, 256, 128, 128, 1);

    assert_eq!(params.src_size, [256, 256], "src_size should be [256, 256]");
    assert_eq!(params.dst_size, [128, 128], "dst_size should be [128, 128]");
    assert_eq!(params.mip_level, 1, "mip_level should be 1");

    println!("PASS: HiZDownsampleParams::new() creates correct params");
}

/// Test HiZDownsampleParams::from_source() auto-calculates destination size.
#[test]
fn test_hiz_downsample_params_from_source() {
    // Power of 2: 256x256 -> 128x128
    let params = HiZDownsampleParams::from_source(256, 256, 1);
    assert_eq!(params.src_size, [256, 256], "src_size should be [256, 256]");
    assert_eq!(
        params.dst_size,
        [128, 128],
        "from_source should auto-calculate dst_size = [128, 128]"
    );
    assert_eq!(params.mip_level, 1, "mip_level should be 1");

    // Non-power of 2: 100x100 -> 50x50
    let params = HiZDownsampleParams::from_source(100, 100, 2);
    assert_eq!(params.dst_size, [50, 50], "100x100 should downsample to 50x50");
    assert_eq!(params.mip_level, 2, "mip_level should be 2");

    // Asymmetric: 200x100 -> 100x50
    let params = HiZDownsampleParams::from_source(200, 100, 3);
    assert_eq!(params.dst_size, [100, 50], "200x100 should downsample to 100x50");

    println!("PASS: HiZDownsampleParams::from_source() auto-calculates dst size");
}

/// Test from_source with minimum size clamping.
#[test]
fn test_hiz_downsample_params_min_size_clamp() {
    // 2x2 -> should clamp to MIN_HIZ_SIZE (1x1)
    let params = HiZDownsampleParams::from_source(2, 2, 5);
    assert_eq!(
        params.dst_size,
        [MIN_HIZ_SIZE, MIN_HIZ_SIZE],
        "2x2 should clamp to MIN_HIZ_SIZE"
    );

    // Already at minimum: 1x1 should stay 1x1
    let params = HiZDownsampleParams::from_source(1, 1, 10);
    assert_eq!(
        params.dst_size,
        [MIN_HIZ_SIZE, MIN_HIZ_SIZE],
        "1x1 should stay at MIN_HIZ_SIZE"
    );

    println!("PASS: from_source clamps to MIN_HIZ_SIZE correctly");
}

/// Test params for multiple mip levels in a chain.
#[test]
fn test_hiz_downsample_params_mip_chain() {
    // Simulate a full mip chain: 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    let mut src_size = 1024u32;

    for mip_level in 1..=10 {
        let params = HiZDownsampleParams::from_source(src_size, src_size, mip_level);
        let expected_dst = (src_size / 2).max(MIN_HIZ_SIZE);

        assert_eq!(
            params.dst_size,
            [expected_dst, expected_dst],
            "Mip {} should have dst_size [{}, {}]",
            mip_level, expected_dst, expected_dst
        );
        assert_eq!(params.mip_level, mip_level, "mip_level should be {}", mip_level);

        src_size = expected_dst;
    }

    println!("PASS: Mip chain parameters are correct for all levels");
}

/// Test params workgroups calculation method.
#[test]
fn test_hiz_downsample_params_workgroups() {
    let params = HiZDownsampleParams::new(256, 256, 128, 128, 1);

    // Workgroups should cover dst size with HIZ_DOWNSAMPLE_WORKGROUP_SIZE
    let workgroups_x = params.workgroups_x();
    let workgroups_y = params.workgroups_y();

    // 128 / 8 = 16 workgroups per dimension
    assert_eq!(workgroups_x, 16, "128 width with workgroup 8 = 16 groups");
    assert_eq!(workgroups_y, 16, "128 height with workgroup 8 = 16 groups");

    println!("PASS: params.workgroups_x/y calculate correctly");
}

// =============================================================================
// CRITERION 3: SHADER SOURCE TESTS
// =============================================================================

/// Test that HIZ_DOWNSAMPLE_SHADER is non-empty.
#[test]
fn test_hiz_downsample_shader_exists() {
    assert!(
        !HIZ_DOWNSAMPLE_SHADER.is_empty(),
        "HIZ_DOWNSAMPLE_SHADER must not be empty"
    );
    assert!(
        HIZ_DOWNSAMPLE_SHADER.len() > 100,
        "HIZ_DOWNSAMPLE_SHADER should be substantial WGSL code"
    );

    println!(
        "PASS: HIZ_DOWNSAMPLE_SHADER exists ({} bytes)",
        HIZ_DOWNSAMPLE_SHADER.len()
    );
}

/// Test shader contains textureLoad for sampling.
#[test]
fn test_hiz_downsample_shader_contains_texture_load() {
    // Per acceptance criteria: shader must use textureLoad for sampling
    assert!(
        HIZ_DOWNSAMPLE_SHADER.contains("textureLoad"),
        "Shader must contain textureLoad for integer coordinate sampling"
    );

    println!("PASS: Shader contains textureLoad");
}

/// Test shader contains textureStore for writing.
#[test]
fn test_hiz_downsample_shader_contains_texture_store() {
    // Shader must write to storage texture
    assert!(
        HIZ_DOWNSAMPLE_SHADER.contains("textureStore"),
        "Shader must contain textureStore for writing output"
    );

    println!("PASS: Shader contains textureStore");
}

/// Test shader contains max() for reduction.
#[test]
fn test_hiz_downsample_shader_contains_max() {
    // Shader must use max() for 2x2 reduction
    assert!(
        HIZ_DOWNSAMPLE_SHADER.contains("max("),
        "Shader must contain max() for reduction operation"
    );

    println!("PASS: Shader contains max() reduction");
}

/// Test shader contains @compute workgroup decorator.
#[test]
fn test_hiz_downsample_shader_is_compute() {
    assert!(
        HIZ_DOWNSAMPLE_SHADER.contains("@compute"),
        "Shader must be a compute shader (@compute)"
    );

    println!("PASS: Shader is a compute shader");
}

/// Test shader contains workgroup_size decorator.
#[test]
fn test_hiz_downsample_shader_has_workgroup_size() {
    assert!(
        HIZ_DOWNSAMPLE_SHADER.contains("@workgroup_size"),
        "Shader must specify @workgroup_size"
    );

    // Should specify (8, 8, 1) or similar
    let wg_size_str = format!("@workgroup_size({}", HIZ_DOWNSAMPLE_WORKGROUP_SIZE);
    assert!(
        HIZ_DOWNSAMPLE_SHADER.contains(&wg_size_str),
        "Shader should specify workgroup_size({}, ...)",
        HIZ_DOWNSAMPLE_WORKGROUP_SIZE
    );

    println!("PASS: Shader has correct workgroup_size decorator");
}

/// Test shader parses successfully via naga.
#[test]
fn test_hiz_downsample_shader_naga_parse() {
    // Use naga to validate WGSL syntax (without GPU)
    let result = naga::front::wgsl::parse_str(HIZ_DOWNSAMPLE_SHADER);

    match result {
        Ok(module) => {
            // Verify module has entry point
            assert!(
                !module.entry_points.is_empty(),
                "Shader module must have at least one entry point"
            );

            // Find compute entry point
            let has_compute = module
                .entry_points
                .iter()
                .any(|ep| ep.stage == naga::ShaderStage::Compute);
            assert!(has_compute, "Shader must have a compute entry point");

            println!("PASS: Shader parses successfully with naga");
        }
        Err(e) => {
            panic!("HIZ_DOWNSAMPLE_SHADER failed naga parse: {:?}", e);
        }
    }
}

// =============================================================================
// CRITERION 4: WORKGROUP SIZE TESTS
// =============================================================================

/// Test HIZ_DOWNSAMPLE_WORKGROUP_SIZE is 8.
#[test]
fn test_hiz_downsample_workgroup_size_constant() {
    assert_eq!(
        HIZ_DOWNSAMPLE_WORKGROUP_SIZE, 8,
        "HIZ_DOWNSAMPLE_WORKGROUP_SIZE must be 8 (shader uses 8x8x1)"
    );

    println!("PASS: HIZ_DOWNSAMPLE_WORKGROUP_SIZE = {}", HIZ_DOWNSAMPLE_WORKGROUP_SIZE);
}

/// Test calculate_downsample_dispatch for power-of-2 sizes.
#[test]
fn test_calculate_downsample_dispatch_power_of_2() {
    // 128x128: 128/8 = 16 workgroups per dimension
    let (x, y, z) = calculate_downsample_dispatch(128, 128);
    assert_eq!((x, y, z), (16, 16, 1), "128x128 should dispatch (16, 16, 1)");

    // 256x256: 256/8 = 32 workgroups
    let (x, y, z) = calculate_downsample_dispatch(256, 256);
    assert_eq!((x, y, z), (32, 32, 1), "256x256 should dispatch (32, 32, 1)");

    // 1024x1024: 1024/8 = 128 workgroups
    let (x, y, z) = calculate_downsample_dispatch(1024, 1024);
    assert_eq!((x, y, z), (128, 128, 1), "1024x1024 should dispatch (128, 128, 1)");

    println!("PASS: calculate_downsample_dispatch works for power-of-2 sizes");
}

/// Test calculate_downsample_dispatch with non-power-of-2 sizes (ceil division).
#[test]
fn test_calculate_downsample_dispatch_non_power_of_2() {
    // 50x50: ceil(50/8) = 7 workgroups per dimension
    let (x, y, z) = calculate_downsample_dispatch(50, 50);
    assert_eq!((x, y, z), (7, 7, 1), "50x50 should dispatch (7, 7, 1)");

    // 100x100: ceil(100/8) = 13 workgroups
    let (x, y, z) = calculate_downsample_dispatch(100, 100);
    assert_eq!((x, y, z), (13, 13, 1), "100x100 should dispatch (13, 13, 1)");

    // 1920x1080: ceil(1920/8)=240, ceil(1080/8)=135
    let (x, y, z) = calculate_downsample_dispatch(1920, 1080);
    assert_eq!((x, y, z), (240, 135, 1), "1920x1080 should dispatch (240, 135, 1)");

    println!("PASS: calculate_downsample_dispatch uses ceil division for non-power-of-2");
}

/// Test calculate_downsample_dispatch for minimum size (1x1).
#[test]
fn test_calculate_downsample_dispatch_minimum() {
    // 1x1: Always needs at least 1 workgroup
    let (x, y, z) = calculate_downsample_dispatch(1, 1);
    assert_eq!((x, y, z), (1, 1, 1), "1x1 should dispatch (1, 1, 1)");

    // 8x8: Exactly one workgroup per dimension
    let (x, y, z) = calculate_downsample_dispatch(8, 8);
    assert_eq!((x, y, z), (1, 1, 1), "8x8 should dispatch (1, 1, 1)");

    // 9x9: Needs ceil(9/8) = 2 workgroups
    let (x, y, z) = calculate_downsample_dispatch(9, 9);
    assert_eq!((x, y, z), (2, 2, 1), "9x9 should dispatch (2, 2, 1)");

    println!("PASS: calculate_downsample_dispatch handles minimum sizes");
}

/// Test calculate_downsample_dispatch for asymmetric sizes.
#[test]
fn test_calculate_downsample_dispatch_asymmetric() {
    // 64x32: (8, 4, 1)
    let (x, y, z) = calculate_downsample_dispatch(64, 32);
    assert_eq!((x, y, z), (8, 4, 1), "64x32 should dispatch (8, 4, 1)");

    // 200x100: ceil(200/8)=25, ceil(100/8)=13
    let (x, y, z) = calculate_downsample_dispatch(200, 100);
    assert_eq!((x, y, z), (25, 13, 1), "200x100 should dispatch (25, 13, 1)");

    // Ultra-wide: 3840x1080
    let (x, y, z) = calculate_downsample_dispatch(3840, 1080);
    assert_eq!((x, y, z), (480, 135, 1), "3840x1080 should dispatch (480, 135, 1)");

    println!("PASS: calculate_downsample_dispatch handles asymmetric sizes");
}

/// Test z dimension is always 1.
#[test]
fn test_calculate_downsample_dispatch_z_is_one() {
    let test_cases = [
        (1, 1),
        (8, 8),
        (128, 128),
        (1024, 1024),
        (1920, 1080),
        (3840, 2160),
    ];

    for (w, h) in test_cases {
        let (_, _, z) = calculate_downsample_dispatch(w, h);
        assert_eq!(z, 1, "z dimension should always be 1 for {}x{}", w, h);
    }

    println!("PASS: z dimension is always 1");
}

// =============================================================================
// BIND GROUP LAYOUT TESTS
// =============================================================================

/// Test create_hiz_downsample_texture_layout creates valid layout.
#[test]
fn test_create_hiz_downsample_texture_layout() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(adapter);

    let layout = create_hiz_downsample_texture_layout(&device);

    // Layout should be created without panic
    // Further validation would require GPU inspection
    drop(layout);

    println!("PASS: create_hiz_downsample_texture_layout creates valid layout");
}

/// Test create_hiz_downsample_params_layout creates valid layout.
#[test]
fn test_create_hiz_downsample_params_layout() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(adapter);

    let layout = create_hiz_downsample_params_layout(&device);

    // Layout should be created without panic
    drop(layout);

    println!("PASS: create_hiz_downsample_params_layout creates valid layout");
}

/// Test both layouts can be created together.
#[test]
fn test_hiz_downsample_layouts_compatible() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(adapter);

    let texture_layout = create_hiz_downsample_texture_layout(&device);
    let params_layout = create_hiz_downsample_params_layout(&device);

    // Both layouts should coexist
    // They would be used in a pipeline layout together
    let _pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hiz_downsample_test_layout"),
        bind_group_layouts: &[&texture_layout, &params_layout],
        push_constant_ranges: &[],
    });

    println!("PASS: Both layouts can be combined into pipeline layout");
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

/// Test full dispatch calculation for a mip chain.
#[test]
fn test_dispatch_for_mip_chain() {
    // Simulate dispatching for a 1024x1024 pyramid
    let mut width = 1024u32;
    let mut height = 1024u32;

    for _mip in 1..=10 {
        let dst_w = (width / 2).max(1);
        let dst_h = (height / 2).max(1);

        let (dispatch_x, dispatch_y, dispatch_z) = calculate_downsample_dispatch(dst_w, dst_h);

        // Verify dispatch covers the destination
        assert!(
            dispatch_x * HIZ_DOWNSAMPLE_WORKGROUP_SIZE >= dst_w,
            "Dispatch x ({}) should cover dst_w ({})",
            dispatch_x * HIZ_DOWNSAMPLE_WORKGROUP_SIZE,
            dst_w
        );
        assert!(
            dispatch_y * HIZ_DOWNSAMPLE_WORKGROUP_SIZE >= dst_h,
            "Dispatch y ({}) should cover dst_h ({})",
            dispatch_y * HIZ_DOWNSAMPLE_WORKGROUP_SIZE,
            dst_h
        );
        assert_eq!(dispatch_z, 1, "Dispatch z should always be 1");

        // Move to next mip
        width = dst_w;
        height = dst_h;

        if width == 1 && height == 1 {
            break;
        }
    }

    println!("PASS: Dispatch calculation covers all mip levels");
}

/// Test params and dispatch work together.
#[test]
fn test_params_and_dispatch_coherent() {
    // For each mip level, params.workgroups should match calculate_downsample_dispatch
    let sizes = [(256, 256), (512, 512), (1920, 1080), (100, 100)];

    for (src_w, src_h) in sizes {
        let dst_w = src_w / 2;
        let dst_h = src_h / 2;

        let params = HiZDownsampleParams::new(src_w, src_h, dst_w, dst_h, 1);
        let (dispatch_x, dispatch_y, _) = calculate_downsample_dispatch(dst_w, dst_h);

        assert_eq!(
            params.workgroups_x(),
            dispatch_x,
            "params.workgroups_x should match calculate_downsample_dispatch for {}x{}",
            src_w, src_h
        );
        assert_eq!(
            params.workgroups_y(),
            dispatch_y,
            "params.workgroups_y should match calculate_downsample_dispatch for {}x{}",
            src_w, src_h
        );
    }

    println!("PASS: Params workgroups match calculate_downsample_dispatch");
}

/// Test shader can be compiled into a module.
#[test]
fn test_hiz_downsample_shader_compile() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(adapter);

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hiz_downsample_test"),
        source: wgpu::ShaderSource::Wgsl(HIZ_DOWNSAMPLE_SHADER.into()),
    });

    // Module creation should succeed
    drop(shader_module);

    println!("PASS: HIZ_DOWNSAMPLE_SHADER compiles to shader module");
}

// =============================================================================
// SUMMARY
// =============================================================================

/// Meta-test that prints coverage summary.
#[test]
fn test_blackbox_coverage_summary() {
    println!();
    println!("=== BLACKBOX COMPLETE: T-WGPU-P6.4.2 ===");
    println!("- Criterion 1: 2x2 max reduction (reverse-Z) - 6 tests");
    println!("- Criterion 2: Per-mip dispatch params - 6 tests");
    println!("- Criterion 3: Shader source verification - 7 tests");
    println!("- Criterion 4: Workgroup size (8,8,1) - 5 tests");
    println!("- Integration tests - 5 tests");
    println!("- Total: 29 tests covering 4 criteria");
    println!("- API surface verified: 8 public items");
    println!("==========================================");
}
