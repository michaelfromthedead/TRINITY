// SPDX-License-Identifier: MIT
//
// blackbox_dispatch_commands.rs -- Blackbox tests for T-WGPU-P3.9.4 Dispatch Commands.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - DispatchError -- Enum representing dispatch validation errors
//   - DispatchLimits -- Device dispatch limit configuration
//   - calculate_workgroups -- 1D workgroup calculation
//   - calculate_workgroups_2d -- 2D workgroup calculation
//   - calculate_workgroups_3d -- 3D workgroup calculation
//   - calculate_workgroups_validated -- 1D with limit validation
//   - calculate_workgroups_2d_validated -- 2D with limit validation
//   - calculate_workgroups_3d_validated -- 3D with limit validation
//   - ComputePass::dispatch_workgroups -- Direct dispatch method
//   - ComputePass::dispatch_workgroups_indirect -- Indirect dispatch method
//
// PUBLIC API METHODS:
//   DispatchLimits:
//     - new(max_x, max_y, max_z) -> Self
//     - uniform(max_per_dimension) -> Self
//     - from_wgpu_limits(limits) -> Self
//     - validate(x, y, z) -> Result<(), DispatchError>
//     - is_valid(x, y, z) -> bool
//     - default() -> Self (max 65535 per dimension)
//
//   DispatchError (variants):
//     - ExceedsMaxX { requested, limit }
//     - ExceedsMaxY { requested, limit }
//     - ExceedsMaxZ { requested, limit }
//     - MultipleExceeded { dimensions, details }
//     - TotalOverflow { x, y, z }
//     - ZeroWorkgroups { dimension }
//
//   Free functions:
//     - calculate_workgroups(total, workgroup_size) -> u32
//     - calculate_workgroups_2d(w, h, (size_x, size_y)) -> (u32, u32)
//     - calculate_workgroups_3d(w, h, d, (sx, sy, sz)) -> (u32, u32, u32)
//     - calculate_workgroups_validated(total, size, limits) -> Result<u32>
//     - calculate_workgroups_2d_validated(w, h, size, limits) -> Result<(u32, u32)>
//     - calculate_workgroups_3d_validated(w, h, d, size, limits) -> Result<(u32, u32, u32)>
//
// ACCEPTANCE CRITERIA (T-WGPU-P3.9.4):
//   1. dispatch_workgroups(x, y, z)
//   2. dispatch_workgroups_indirect(buffer, offset)
//   3. Workgroup count calculation helper
//   4. Limit validation
//
// TEST CATEGORIES:
//   1. API Tests (10 tests) - Public interface, types exist
//   2. DispatchLimits (16 tests) - Construction, validation, queries
//   3. DispatchError (10 tests) - Variants, Display, Error trait
//   4. calculate_workgroups (10 tests) - 1D calculations
//   5. calculate_workgroups_2d (10 tests) - 2D calculations
//   6. calculate_workgroups_3d (10 tests) - 3D calculations
//   7. Validated Functions (12 tests) - Functions with limit checking
//   8. Real-world Scenarios (12 tests) - Particle systems, image processing, etc.
//   9. Edge Cases (10 tests) - Boundaries, zero, max values
//   10. Type Bounds (6 tests) - Send, Sync, Clone, etc.
//
// Total target: 96 tests

use renderer_backend::compute_pass::{
    calculate_workgroups, calculate_workgroups_2d, calculate_workgroups_2d_validated,
    calculate_workgroups_3d, calculate_workgroups_3d_validated, calculate_workgroups_validated,
    DispatchError, DispatchLimits,
};

// ---------------------------------------------------------------------------
// Category 1: API Tests (10 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_error_type_exists() {
    // DispatchError enum exists and can be constructed
    let _: fn() -> DispatchError = || DispatchError::ZeroWorkgroups { dimension: 'X' };
}

#[test]
fn test_dispatch_limits_type_exists() {
    // DispatchLimits struct exists and can be constructed
    let limits = DispatchLimits::default();
    let _ = limits;
}

#[test]
fn test_calculate_workgroups_function_exists() {
    // Function exists with correct signature
    let result = calculate_workgroups(100, 64);
    assert!(result > 0);
}

#[test]
fn test_calculate_workgroups_2d_function_exists() {
    // Function exists with correct signature
    let (x, y) = calculate_workgroups_2d(100, 100, (16, 16));
    assert!(x > 0 && y > 0);
}

#[test]
fn test_calculate_workgroups_3d_function_exists() {
    // Function exists with correct signature
    let (x, y, z) = calculate_workgroups_3d(100, 100, 100, (8, 8, 8));
    assert!(x > 0 && y > 0 && z > 0);
}

#[test]
fn test_calculate_workgroups_validated_function_exists() {
    // Function exists with correct signature
    let limits = DispatchLimits::default();
    let result = calculate_workgroups_validated(100, 64, &limits);
    assert!(result.is_ok());
}

#[test]
fn test_calculate_workgroups_2d_validated_function_exists() {
    // Function exists with correct signature
    let limits = DispatchLimits::default();
    let result = calculate_workgroups_2d_validated(100, 100, (16, 16), &limits);
    assert!(result.is_ok());
}

#[test]
fn test_calculate_workgroups_3d_validated_function_exists() {
    // Function exists with correct signature
    let limits = DispatchLimits::default();
    let result = calculate_workgroups_3d_validated(100, 100, 100, (8, 8, 8), &limits);
    assert!(result.is_ok());
}

#[test]
fn test_dispatch_limits_validate_method_exists() {
    let limits = DispatchLimits::default();
    let result = limits.validate(10, 10, 10);
    assert!(result.is_ok());
}

#[test]
fn test_dispatch_limits_is_valid_method_exists() {
    let limits = DispatchLimits::default();
    let valid = limits.is_valid(10, 10, 10);
    assert!(valid);
}

// ---------------------------------------------------------------------------
// Category 2: DispatchLimits (16 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_limits_default() {
    let limits = DispatchLimits::default();
    // Default should have conservative limits (65535)
    assert!(limits.is_valid(65535, 65535, 65535));
    assert!(!limits.is_valid(65536, 1, 1));
}

#[test]
fn test_dispatch_limits_new() {
    let limits = DispatchLimits::new(100, 200, 300);
    assert!(limits.is_valid(100, 200, 300));
    assert!(!limits.is_valid(101, 200, 300));
    assert!(!limits.is_valid(100, 201, 300));
    assert!(!limits.is_valid(100, 200, 301));
}

#[test]
fn test_dispatch_limits_uniform() {
    let limits = DispatchLimits::uniform(1000);
    assert!(limits.is_valid(1000, 1000, 1000));
    assert!(!limits.is_valid(1001, 1, 1));
    assert!(!limits.is_valid(1, 1001, 1));
    assert!(!limits.is_valid(1, 1, 1001));
}

#[test]
fn test_dispatch_limits_validate_valid() {
    let limits = DispatchLimits::uniform(100);
    assert!(limits.validate(50, 50, 50).is_ok());
    assert!(limits.validate(100, 100, 100).is_ok());
    assert!(limits.validate(1, 1, 1).is_ok());
}

#[test]
fn test_dispatch_limits_validate_exceeds_x() {
    let limits = DispatchLimits::uniform(100);
    let result = limits.validate(101, 50, 50);
    assert!(result.is_err());
    match result.unwrap_err() {
        DispatchError::ExceedsMaxX { requested, limit } => {
            assert_eq!(requested, 101);
            assert_eq!(limit, 100);
        }
        _ => panic!("Expected ExceedsMaxX error"),
    }
}

#[test]
fn test_dispatch_limits_validate_exceeds_y() {
    let limits = DispatchLimits::uniform(100);
    let result = limits.validate(50, 101, 50);
    assert!(result.is_err());
    match result.unwrap_err() {
        DispatchError::ExceedsMaxY { requested, limit } => {
            assert_eq!(requested, 101);
            assert_eq!(limit, 100);
        }
        _ => panic!("Expected ExceedsMaxY error"),
    }
}

#[test]
fn test_dispatch_limits_validate_exceeds_z() {
    let limits = DispatchLimits::uniform(100);
    let result = limits.validate(50, 50, 101);
    assert!(result.is_err());
    match result.unwrap_err() {
        DispatchError::ExceedsMaxZ { requested, limit } => {
            assert_eq!(requested, 101);
            assert_eq!(limit, 100);
        }
        _ => panic!("Expected ExceedsMaxZ error"),
    }
}

#[test]
fn test_dispatch_limits_validate_zero_x() {
    let limits = DispatchLimits::default();
    let result = limits.validate(0, 50, 50);
    assert!(result.is_err());
    match result.unwrap_err() {
        DispatchError::ZeroWorkgroups { dimension } => {
            assert_eq!(dimension, 'X');
        }
        _ => panic!("Expected ZeroWorkgroups error"),
    }
}

#[test]
fn test_dispatch_limits_validate_zero_y() {
    let limits = DispatchLimits::default();
    let result = limits.validate(50, 0, 50);
    assert!(result.is_err());
    match result.unwrap_err() {
        DispatchError::ZeroWorkgroups { dimension } => {
            assert_eq!(dimension, 'Y');
        }
        _ => panic!("Expected ZeroWorkgroups error"),
    }
}

#[test]
fn test_dispatch_limits_validate_zero_z() {
    let limits = DispatchLimits::default();
    let result = limits.validate(50, 50, 0);
    assert!(result.is_err());
    match result.unwrap_err() {
        DispatchError::ZeroWorkgroups { dimension } => {
            assert_eq!(dimension, 'Z');
        }
        _ => panic!("Expected ZeroWorkgroups error"),
    }
}

#[test]
fn test_dispatch_limits_validate_multiple_exceeded() {
    let limits = DispatchLimits::uniform(100);
    let result = limits.validate(101, 101, 50);
    assert!(result.is_err());
    match result.unwrap_err() {
        DispatchError::MultipleExceeded { dimensions, .. } => {
            assert!(dimensions.contains('X'));
            assert!(dimensions.contains('Y'));
        }
        _ => panic!("Expected MultipleExceeded error"),
    }
}

#[test]
fn test_dispatch_limits_is_valid_true() {
    let limits = DispatchLimits::uniform(100);
    assert!(limits.is_valid(1, 1, 1));
    assert!(limits.is_valid(50, 50, 50));
    assert!(limits.is_valid(100, 100, 100));
}

#[test]
fn test_dispatch_limits_is_valid_false() {
    let limits = DispatchLimits::uniform(100);
    assert!(!limits.is_valid(101, 1, 1));
    assert!(!limits.is_valid(1, 101, 1));
    assert!(!limits.is_valid(1, 1, 101));
    assert!(!limits.is_valid(0, 1, 1));
}

#[test]
fn test_dispatch_limits_asymmetric() {
    let limits = DispatchLimits::new(1000, 2000, 500);
    assert!(limits.is_valid(1000, 2000, 500));
    assert!(!limits.is_valid(1001, 1, 1));
    assert!(!limits.is_valid(1, 2001, 1));
    assert!(!limits.is_valid(1, 1, 501));
}

#[test]
fn test_dispatch_limits_minimum() {
    let limits = DispatchLimits::uniform(1);
    assert!(limits.is_valid(1, 1, 1));
    assert!(!limits.is_valid(2, 1, 1));
}

#[test]
fn test_dispatch_limits_public_fields() {
    // Public fields should be accessible
    let limits = DispatchLimits::new(100, 200, 300);
    assert_eq!(limits.max_workgroups_x, 100);
    assert_eq!(limits.max_workgroups_y, 200);
    assert_eq!(limits.max_workgroups_z, 300);
}

// ---------------------------------------------------------------------------
// Category 3: DispatchError (10 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_error_display_exceeds_max_x() {
    let error = DispatchError::ExceedsMaxX {
        requested: 100,
        limit: 50,
    };
    let msg = format!("{}", error);
    assert!(msg.contains("100"));
    assert!(msg.contains("50"));
}

#[test]
fn test_dispatch_error_display_exceeds_max_y() {
    let error = DispatchError::ExceedsMaxY {
        requested: 200,
        limit: 100,
    };
    let msg = format!("{}", error);
    assert!(msg.contains("200"));
    assert!(msg.contains("100"));
}

#[test]
fn test_dispatch_error_display_exceeds_max_z() {
    let error = DispatchError::ExceedsMaxZ {
        requested: 300,
        limit: 150,
    };
    let msg = format!("{}", error);
    assert!(msg.contains("300"));
    assert!(msg.contains("150"));
}

#[test]
fn test_dispatch_error_display_zero_workgroups() {
    let error = DispatchError::ZeroWorkgroups { dimension: 'X' };
    let msg = format!("{}", error);
    assert!(msg.contains("zero") || msg.contains("Zero") || msg.contains('X'));
}

#[test]
fn test_dispatch_error_display_multiple_exceeded() {
    let error = DispatchError::MultipleExceeded {
        dimensions: "X, Y".to_string(),
        details: "X and Y exceeded".to_string(),
    };
    let msg = format!("{}", error);
    assert!(msg.len() > 0);
}

#[test]
fn test_dispatch_error_display_total_overflow() {
    let error = DispatchError::TotalOverflow {
        x: u32::MAX,
        y: u32::MAX,
        z: u32::MAX,
    };
    let msg = format!("{}", error);
    assert!(msg.len() > 0);
}

#[test]
fn test_dispatch_error_implements_std_error() {
    fn assert_error<T: std::error::Error>() {}
    assert_error::<DispatchError>();
}

#[test]
fn test_dispatch_error_implements_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<DispatchError>();
}

#[test]
fn test_dispatch_error_exceeds_max_x_fields() {
    let error = DispatchError::ExceedsMaxX {
        requested: 100,
        limit: 50,
    };
    match error {
        DispatchError::ExceedsMaxX { requested, limit } => {
            assert_eq!(requested, 100);
            assert_eq!(limit, 50);
        }
        _ => panic!("Variant mismatch"),
    }
}

#[test]
fn test_dispatch_error_zero_workgroups_dimension() {
    for dim in ['X', 'Y', 'Z'] {
        let error = DispatchError::ZeroWorkgroups { dimension: dim };
        match error {
            DispatchError::ZeroWorkgroups { dimension } => {
                assert_eq!(dimension, dim);
            }
            _ => panic!("Variant mismatch"),
        }
    }
}

// ---------------------------------------------------------------------------
// Category 4: calculate_workgroups (10 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_calculate_workgroups_exact_multiple() {
    // 64 elements / 64 workgroup size = 1 workgroup
    assert_eq!(calculate_workgroups(64, 64), 1);
    // 128 elements / 64 workgroup size = 2 workgroups
    assert_eq!(calculate_workgroups(128, 64), 2);
    // 256 elements / 64 workgroup size = 4 workgroups
    assert_eq!(calculate_workgroups(256, 64), 4);
}

#[test]
fn test_calculate_workgroups_with_remainder() {
    // 100 elements / 64 workgroup size = ceil(100/64) = 2 workgroups
    assert_eq!(calculate_workgroups(100, 64), 2);
    // 65 elements / 64 workgroup size = ceil(65/64) = 2 workgroups
    assert_eq!(calculate_workgroups(65, 64), 2);
    // 63 elements / 64 workgroup size = ceil(63/64) = 1 workgroup
    assert_eq!(calculate_workgroups(63, 64), 1);
}

#[test]
fn test_calculate_workgroups_single_element() {
    // 1 element / any size = 1 workgroup
    assert_eq!(calculate_workgroups(1, 64), 1);
    assert_eq!(calculate_workgroups(1, 256), 1);
    assert_eq!(calculate_workgroups(1, 1), 1);
}

#[test]
fn test_calculate_workgroups_workgroup_size_one() {
    // When workgroup size is 1, result equals total elements
    assert_eq!(calculate_workgroups(100, 1), 100);
    assert_eq!(calculate_workgroups(1000, 1), 1000);
}

#[test]
fn test_calculate_workgroups_large_values() {
    // Large element counts
    assert_eq!(calculate_workgroups(1_000_000, 256), 3907);
    assert_eq!(calculate_workgroups(10_000_000, 256), 39063);
}

#[test]
fn test_calculate_workgroups_particle_system() {
    // 1 million particles with workgroup size 256
    let workgroups = calculate_workgroups(1_000_000, 256);
    // ceil(1000000 / 256) = 3907
    assert_eq!(workgroups, 3907);
}

#[test]
fn test_calculate_workgroups_common_sizes() {
    // Common workgroup sizes: 64, 128, 256
    assert_eq!(calculate_workgroups(1000, 64), 16);
    assert_eq!(calculate_workgroups(1000, 128), 8);
    assert_eq!(calculate_workgroups(1000, 256), 4);
}

#[test]
fn test_calculate_workgroups_zero_elements() {
    // 0 elements should result in 0 workgroups
    assert_eq!(calculate_workgroups(0, 64), 0);
}

#[test]
fn test_calculate_workgroups_max_u32() {
    // Test with maximum u32 value
    let result = calculate_workgroups(u32::MAX, 256);
    // Should not overflow
    assert!(result > 0);
}

#[test]
#[should_panic]
fn test_calculate_workgroups_zero_workgroup_size_panics() {
    // Workgroup size of 0 should panic
    let _ = calculate_workgroups(100, 0);
}

// ---------------------------------------------------------------------------
// Category 5: calculate_workgroups_2d (10 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_calculate_workgroups_2d_exact_multiple() {
    let (x, y) = calculate_workgroups_2d(64, 64, (16, 16));
    assert_eq!(x, 4);
    assert_eq!(y, 4);
}

#[test]
fn test_calculate_workgroups_2d_with_remainder() {
    let (x, y) = calculate_workgroups_2d(100, 100, (16, 16));
    // ceil(100/16) = 7
    assert_eq!(x, 7);
    assert_eq!(y, 7);
}

#[test]
fn test_calculate_workgroups_2d_asymmetric_dimensions() {
    let (x, y) = calculate_workgroups_2d(1920, 1080, (16, 16));
    // ceil(1920/16) = 120, ceil(1080/16) = 68
    assert_eq!(x, 120);
    assert_eq!(y, 68);
}

#[test]
fn test_calculate_workgroups_2d_asymmetric_workgroup() {
    let (x, y) = calculate_workgroups_2d(1920, 1080, (32, 8));
    // ceil(1920/32) = 60, ceil(1080/8) = 135
    assert_eq!(x, 60);
    assert_eq!(y, 135);
}

#[test]
fn test_calculate_workgroups_2d_single_pixel() {
    let (x, y) = calculate_workgroups_2d(1, 1, (16, 16));
    assert_eq!(x, 1);
    assert_eq!(y, 1);
}

#[test]
fn test_calculate_workgroups_2d_4k_image() {
    // 4K resolution: 3840x2160
    let (x, y) = calculate_workgroups_2d(3840, 2160, (16, 16));
    // ceil(3840/16) = 240, ceil(2160/16) = 135
    assert_eq!(x, 240);
    assert_eq!(y, 135);
}

#[test]
fn test_calculate_workgroups_2d_8x8_workgroup() {
    let (x, y) = calculate_workgroups_2d(1024, 1024, (8, 8));
    // 1024/8 = 128 exactly
    assert_eq!(x, 128);
    assert_eq!(y, 128);
}

#[test]
fn test_calculate_workgroups_2d_zero_width() {
    let (x, y) = calculate_workgroups_2d(0, 100, (16, 16));
    assert_eq!(x, 0);
    assert_eq!(y, 7);
}

#[test]
fn test_calculate_workgroups_2d_zero_height() {
    let (x, y) = calculate_workgroups_2d(100, 0, (16, 16));
    assert_eq!(x, 7);
    assert_eq!(y, 0);
}

#[test]
#[should_panic]
fn test_calculate_workgroups_2d_zero_workgroup_size_panics() {
    let _ = calculate_workgroups_2d(100, 100, (0, 16));
}

// ---------------------------------------------------------------------------
// Category 6: calculate_workgroups_3d (10 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_calculate_workgroups_3d_exact_multiple() {
    let (x, y, z) = calculate_workgroups_3d(64, 64, 64, (8, 8, 8));
    assert_eq!(x, 8);
    assert_eq!(y, 8);
    assert_eq!(z, 8);
}

#[test]
fn test_calculate_workgroups_3d_with_remainder() {
    let (x, y, z) = calculate_workgroups_3d(100, 100, 100, (8, 8, 8));
    // ceil(100/8) = 13
    assert_eq!(x, 13);
    assert_eq!(y, 13);
    assert_eq!(z, 13);
}

#[test]
fn test_calculate_workgroups_3d_volume_256_cubed() {
    // Common volume size: 256^3
    let (x, y, z) = calculate_workgroups_3d(256, 256, 256, (8, 8, 8));
    // 256/8 = 32 exactly
    assert_eq!(x, 32);
    assert_eq!(y, 32);
    assert_eq!(z, 32);
}

#[test]
fn test_calculate_workgroups_3d_asymmetric_dimensions() {
    let (x, y, z) = calculate_workgroups_3d(512, 256, 128, (8, 8, 8));
    assert_eq!(x, 64);
    assert_eq!(y, 32);
    assert_eq!(z, 16);
}

#[test]
fn test_calculate_workgroups_3d_asymmetric_workgroup() {
    let (x, y, z) = calculate_workgroups_3d(256, 256, 256, (4, 8, 16));
    assert_eq!(x, 64);
    assert_eq!(y, 32);
    assert_eq!(z, 16);
}

#[test]
fn test_calculate_workgroups_3d_single_voxel() {
    let (x, y, z) = calculate_workgroups_3d(1, 1, 1, (8, 8, 8));
    assert_eq!(x, 1);
    assert_eq!(y, 1);
    assert_eq!(z, 1);
}

#[test]
fn test_calculate_workgroups_3d_voxel_terrain() {
    // Minecraft-style: 16x256x16 chunks with 4x4x4 workgroups
    let (x, y, z) = calculate_workgroups_3d(16, 256, 16, (4, 4, 4));
    assert_eq!(x, 4);
    assert_eq!(y, 64);
    assert_eq!(z, 4);
}

#[test]
fn test_calculate_workgroups_3d_zero_width() {
    let (x, y, z) = calculate_workgroups_3d(0, 100, 100, (8, 8, 8));
    assert_eq!(x, 0);
    assert_eq!(y, 13);
    assert_eq!(z, 13);
}

#[test]
fn test_calculate_workgroups_3d_zero_depth() {
    let (x, y, z) = calculate_workgroups_3d(100, 100, 0, (8, 8, 8));
    assert_eq!(x, 13);
    assert_eq!(y, 13);
    assert_eq!(z, 0);
}

#[test]
#[should_panic]
fn test_calculate_workgroups_3d_zero_workgroup_size_panics() {
    let _ = calculate_workgroups_3d(100, 100, 100, (8, 0, 8));
}

// ---------------------------------------------------------------------------
// Category 7: Validated Functions (12 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_calculate_workgroups_validated_ok() {
    let limits = DispatchLimits::default();
    let result = calculate_workgroups_validated(1000, 64, &limits);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 16);
}

#[test]
fn test_calculate_workgroups_validated_exceeds_limit() {
    let limits = DispatchLimits::uniform(10);
    // 1000 / 64 = 16 workgroups, exceeds limit of 10
    let result = calculate_workgroups_validated(1000, 64, &limits);
    assert!(result.is_err());
}

#[test]
fn test_calculate_workgroups_validated_at_limit() {
    let limits = DispatchLimits::uniform(16);
    let result = calculate_workgroups_validated(1000, 64, &limits);
    assert!(result.is_ok());
}

#[test]
fn test_calculate_workgroups_2d_validated_ok() {
    let limits = DispatchLimits::default();
    let result = calculate_workgroups_2d_validated(1920, 1080, (16, 16), &limits);
    assert!(result.is_ok());
    let (x, y) = result.unwrap();
    assert_eq!(x, 120);
    assert_eq!(y, 68);
}

#[test]
fn test_calculate_workgroups_2d_validated_exceeds_x() {
    let limits = DispatchLimits::new(100, 1000, 1000);
    // 1920/16 = 120 > 100
    let result = calculate_workgroups_2d_validated(1920, 1080, (16, 16), &limits);
    assert!(result.is_err());
}

#[test]
fn test_calculate_workgroups_2d_validated_exceeds_y() {
    let limits = DispatchLimits::new(1000, 50, 1000);
    // 1080/16 = 68 > 50
    let result = calculate_workgroups_2d_validated(1920, 1080, (16, 16), &limits);
    assert!(result.is_err());
}

#[test]
fn test_calculate_workgroups_3d_validated_ok() {
    let limits = DispatchLimits::default();
    let result = calculate_workgroups_3d_validated(256, 256, 256, (8, 8, 8), &limits);
    assert!(result.is_ok());
    let (x, y, z) = result.unwrap();
    assert_eq!(x, 32);
    assert_eq!(y, 32);
    assert_eq!(z, 32);
}

#[test]
fn test_calculate_workgroups_3d_validated_exceeds_z() {
    let limits = DispatchLimits::new(1000, 1000, 10);
    // 256/8 = 32 > 10 for Z
    let result = calculate_workgroups_3d_validated(256, 256, 256, (8, 8, 8), &limits);
    assert!(result.is_err());
}

#[test]
fn test_calculate_workgroups_validated_large_workgroup_size() {
    // Using larger workgroup size to reduce workgroups
    let limits = DispatchLimits::uniform(10);
    // 1000 / 256 = 4 workgroups, within limit
    let result = calculate_workgroups_validated(1000, 256, &limits);
    assert!(result.is_ok());
}

#[test]
fn test_calculate_workgroups_2d_validated_larger_workgroup() {
    let limits = DispatchLimits::uniform(100);
    // 1920/32=60, 1080/32=34, both within 100
    let result = calculate_workgroups_2d_validated(1920, 1080, (32, 32), &limits);
    assert!(result.is_ok());
}

#[test]
fn test_calculate_workgroups_3d_validated_larger_workgroup() {
    let limits = DispatchLimits::uniform(20);
    // 256/16=16, within limit of 20
    let result = calculate_workgroups_3d_validated(256, 256, 256, (16, 16, 16), &limits);
    assert!(result.is_ok());
}

#[test]
fn test_validated_functions_return_same_values_as_unvalidated() {
    let limits = DispatchLimits::default();

    // 1D
    let val1d = calculate_workgroups(1000, 64);
    let val1d_validated = calculate_workgroups_validated(1000, 64, &limits).unwrap();
    assert_eq!(val1d, val1d_validated);

    // 2D
    let val2d = calculate_workgroups_2d(1920, 1080, (16, 16));
    let val2d_validated = calculate_workgroups_2d_validated(1920, 1080, (16, 16), &limits).unwrap();
    assert_eq!(val2d, val2d_validated);

    // 3D
    let val3d = calculate_workgroups_3d(256, 256, 256, (8, 8, 8));
    let val3d_validated =
        calculate_workgroups_3d_validated(256, 256, 256, (8, 8, 8), &limits).unwrap();
    assert_eq!(val3d, val3d_validated);
}

// ---------------------------------------------------------------------------
// Category 8: Real-world Scenarios (12 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_particle_system_1m_particles() {
    // 1 million particles with 256 threads per workgroup
    let workgroups = calculate_workgroups(1_000_000, 256);
    assert_eq!(workgroups, 3907);

    let limits = DispatchLimits::default();
    assert!(limits.is_valid(workgroups, 1, 1));
}

#[test]
fn test_particle_system_10m_particles() {
    // 10 million particles
    let workgroups = calculate_workgroups(10_000_000, 256);
    assert_eq!(workgroups, 39063);

    let limits = DispatchLimits::default();
    assert!(limits.is_valid(workgroups, 1, 1));
}

#[test]
fn test_image_processing_1080p() {
    // 1920x1080 with 16x16 workgroup
    let (x, y) = calculate_workgroups_2d(1920, 1080, (16, 16));
    assert_eq!(x, 120);
    assert_eq!(y, 68);

    let limits = DispatchLimits::default();
    assert!(limits.is_valid(x, y, 1));
}

#[test]
fn test_image_processing_4k() {
    // 3840x2160 with 16x16 workgroup
    let (x, y) = calculate_workgroups_2d(3840, 2160, (16, 16));
    assert_eq!(x, 240);
    assert_eq!(y, 135);

    let limits = DispatchLimits::default();
    assert!(limits.is_valid(x, y, 1));
}

#[test]
fn test_image_processing_8k() {
    // 7680x4320 with 16x16 workgroup
    let (x, y) = calculate_workgroups_2d(7680, 4320, (16, 16));
    assert_eq!(x, 480);
    assert_eq!(y, 270);

    let limits = DispatchLimits::default();
    assert!(limits.is_valid(x, y, 1));
}

#[test]
fn test_volume_rendering_256_cubed() {
    // 256^3 volume with 8x8x8 workgroup
    let (x, y, z) = calculate_workgroups_3d(256, 256, 256, (8, 8, 8));
    assert_eq!(x, 32);
    assert_eq!(y, 32);
    assert_eq!(z, 32);

    let limits = DispatchLimits::default();
    assert!(limits.is_valid(x, y, z));
}

#[test]
fn test_volume_rendering_512_cubed() {
    // 512^3 volume with 8x8x8 workgroup
    let (x, y, z) = calculate_workgroups_3d(512, 512, 512, (8, 8, 8));
    assert_eq!(x, 64);
    assert_eq!(y, 64);
    assert_eq!(z, 64);

    let limits = DispatchLimits::default();
    assert!(limits.is_valid(x, y, z));
}

#[test]
fn test_fluid_simulation_grid() {
    // 128x64x128 fluid grid with 4x4x4 workgroups
    let (x, y, z) = calculate_workgroups_3d(128, 64, 128, (4, 4, 4));
    assert_eq!(x, 32);
    assert_eq!(y, 16);
    assert_eq!(z, 32);

    let limits = DispatchLimits::default();
    assert!(limits.is_valid(x, y, z));
}

#[test]
fn test_compute_shader_prefix_sum() {
    // Prefix sum on 1M elements with 1024-element workgroups
    // First pass: 1M/1024 = 977 workgroups
    let workgroups = calculate_workgroups(1_000_000, 1024);
    assert_eq!(workgroups, 977);
}

#[test]
fn test_physics_broadphase() {
    // Collision broadphase: 10000 objects with 64 threads
    let workgroups = calculate_workgroups(10_000, 64);
    assert_eq!(workgroups, 157);
}

#[test]
fn test_skinned_mesh_animation() {
    // 50000 vertices, 256 threads per workgroup
    let workgroups = calculate_workgroups(50_000, 256);
    assert_eq!(workgroups, 196);
}

#[test]
fn test_terrain_heightmap() {
    // 4096x4096 heightmap with 16x16 workgroup
    let (x, y) = calculate_workgroups_2d(4096, 4096, (16, 16));
    assert_eq!(x, 256);
    assert_eq!(y, 256);

    let limits = DispatchLimits::default();
    assert!(limits.is_valid(x, y, 1));
}

// ---------------------------------------------------------------------------
// Category 9: Edge Cases (10 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_edge_case_workgroup_equals_total() {
    // When workgroup size equals total elements
    assert_eq!(calculate_workgroups(256, 256), 1);
    assert_eq!(calculate_workgroups(64, 64), 1);
}

#[test]
fn test_edge_case_workgroup_larger_than_total() {
    // When workgroup size is larger than total elements
    assert_eq!(calculate_workgroups(10, 256), 1);
    assert_eq!(calculate_workgroups(1, 1024), 1);
}

#[test]
fn test_edge_case_one_less_than_multiple() {
    // 63 elements / 64 workgroup = 1 workgroup
    assert_eq!(calculate_workgroups(63, 64), 1);
    // 127 / 64 = 2 workgroups
    assert_eq!(calculate_workgroups(127, 64), 2);
}

#[test]
fn test_edge_case_one_more_than_multiple() {
    // 65 elements / 64 workgroup = 2 workgroups
    assert_eq!(calculate_workgroups(65, 64), 2);
    // 129 / 64 = 3 workgroups
    assert_eq!(calculate_workgroups(129, 64), 3);
}

#[test]
fn test_edge_case_limits_at_boundary() {
    let limits = DispatchLimits::uniform(100);

    // Exactly at limit
    assert!(limits.is_valid(100, 100, 100));
    assert!(limits.validate(100, 100, 100).is_ok());

    // One over limit
    assert!(!limits.is_valid(101, 100, 100));
    assert!(limits.validate(101, 100, 100).is_err());
}

#[test]
fn test_edge_case_near_u32_max() {
    // Large values near u32::MAX
    let result = calculate_workgroups(u32::MAX - 255, 256);
    // Should complete without overflow
    assert!(result > 0);
}

#[test]
fn test_edge_case_2d_asymmetric_zero() {
    // Mixed zero and non-zero
    let (x, y) = calculate_workgroups_2d(0, 1920, (16, 16));
    assert_eq!(x, 0);
    assert_eq!(y, 120);
}

#[test]
fn test_edge_case_3d_one_dimension_large() {
    // One very large dimension, others small
    let (x, y, z) = calculate_workgroups_3d(10000, 1, 1, (8, 8, 8));
    assert_eq!(x, 1250);
    assert_eq!(y, 1);
    assert_eq!(z, 1);
}

#[test]
fn test_edge_case_power_of_two() {
    // Power of two dimensions
    assert_eq!(calculate_workgroups(1024, 64), 16);
    assert_eq!(calculate_workgroups(2048, 64), 32);
    assert_eq!(calculate_workgroups(4096, 64), 64);
}

#[test]
fn test_edge_case_prime_numbers() {
    // Prime number total elements
    assert_eq!(calculate_workgroups(997, 64), 16); // ceil(997/64) = 16
    assert_eq!(calculate_workgroups(1009, 64), 16); // ceil(1009/64) = 16
}

// ---------------------------------------------------------------------------
// Category 10: Type Bounds (6 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_limits_is_clone() {
    fn assert_clone<T: Clone>() {}
    assert_clone::<DispatchLimits>();
}

#[test]
fn test_dispatch_limits_is_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<DispatchLimits>();
}

#[test]
fn test_dispatch_limits_clone_works() {
    let limits = DispatchLimits::new(100, 200, 300);
    let cloned = limits.clone();
    assert_eq!(cloned.max_workgroups_x, 100);
    assert_eq!(cloned.max_workgroups_y, 200);
    assert_eq!(cloned.max_workgroups_z, 300);
}

#[test]
fn test_dispatch_limits_debug_format() {
    let limits = DispatchLimits::new(100, 200, 300);
    let debug = format!("{:?}", limits);
    assert!(debug.contains("DispatchLimits"));
}

#[test]
fn test_dispatch_error_is_clone() {
    fn assert_clone<T: Clone>() {}
    assert_clone::<DispatchError>();
}

#[test]
fn test_dispatch_error_clone_works() {
    let error = DispatchError::ExceedsMaxX {
        requested: 100,
        limit: 50,
    };
    let cloned = error.clone();
    match cloned {
        DispatchError::ExceedsMaxX { requested, limit } => {
            assert_eq!(requested, 100);
            assert_eq!(limit, 50);
        }
        _ => panic!("Clone mismatch"),
    }
}
