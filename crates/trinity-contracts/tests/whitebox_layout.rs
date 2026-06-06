//! Whitebox tests for layout contracts.

use trinity_contracts::layout::{
    check_layout, get_layout, gpu_sizes, LayoutError, LayoutResult, LayoutSpec, MirrorRegistry,
    WgslMirror,
};

// ==================== LayoutSpec ====================

#[test]
fn test_layout_spec_new() {
    let spec = LayoutSpec::new();
    assert!(spec.size.is_none());
    assert!(spec.align.is_none());
    assert!(!spec.packed);
}

#[test]
fn test_layout_spec_size() {
    let spec = LayoutSpec::new().size(16);
    assert_eq!(spec.size, Some(16));
}

#[test]
fn test_layout_spec_align() {
    let spec = LayoutSpec::new().align(8);
    assert_eq!(spec.align, Some(8));
}

#[test]
fn test_layout_spec_packed() {
    let spec = LayoutSpec::new().packed();
    assert!(spec.packed);
}

#[test]
fn test_layout_spec_check_size() {
    let spec = LayoutSpec::new().size(16);
    assert!(spec.check_size(16));
    assert!(!spec.check_size(8));
}

#[test]
fn test_layout_spec_check_align() {
    let spec = LayoutSpec::new().align(8);
    assert!(spec.check_align(8));
    assert!(!spec.check_align(4));
}

#[test]
fn test_layout_spec_check_ok() {
    let spec = LayoutSpec::new().size(16).align(8);
    let result = spec.check(16, 8);
    assert!(result.is_ok());
}

#[test]
fn test_layout_spec_check_mismatch() {
    let spec = LayoutSpec::new().size(16).align(8);
    let result = spec.check(32, 4);
    assert!(!result.is_ok());
}

// ==================== LayoutResult ====================

#[test]
fn test_layout_result_ok() {
    let result = LayoutResult::Ok;
    assert!(result.is_ok());
    assert!(result.error_message().is_none());
}

#[test]
fn test_layout_result_mismatch() {
    let result = LayoutResult::Mismatch {
        expected_size: Some(16),
        actual_size: 32,
        expected_align: Some(8),
        actual_align: 4,
    };
    assert!(!result.is_ok());
    let msg = result.error_message().unwrap();
    assert!(msg.contains("size"));
    assert!(msg.contains("alignment"));
}

// ==================== WgslMirror ====================

#[test]
fn test_wgsl_mirror_new() {
    let mirror = WgslMirror::new("MyStruct", "my_struct");
    assert_eq!(mirror.rust_type, "MyStruct");
    assert_eq!(mirror.wgsl_type, "my_struct");
}

#[test]
fn test_wgsl_mirror_layout() {
    let mirror = WgslMirror::new("Vec4", "vec4<f32>")
        .layout(LayoutSpec::new().size(16).align(16));
    assert_eq!(mirror.layout.size, Some(16));
}

#[test]
fn test_wgsl_mirror_verify_ok() {
    let mirror = WgslMirror::new("Vec4", "vec4<f32>")
        .layout(LayoutSpec::new().size(16).align(16));
    let result = mirror.verify(16, 16);
    assert!(result.is_ok());
}

#[test]
fn test_wgsl_mirror_verify_fail() {
    let mirror = WgslMirror::new("Vec4", "vec4<f32>")
        .layout(LayoutSpec::new().size(16));
    let result = mirror.verify(12, 4);
    assert!(!result.is_ok());
}

// ==================== MirrorRegistry ====================

#[test]
fn test_registry_new() {
    let registry = MirrorRegistry::new();
    assert!(registry.is_empty());
}

#[test]
fn test_registry_register() {
    let mut registry = MirrorRegistry::new();
    registry.register(WgslMirror::new("Test", "test"));
    assert_eq!(registry.len(), 1);
}

#[test]
fn test_registry_get() {
    let mut registry = MirrorRegistry::new();
    registry.register(WgslMirror::new("Test", "test"));
    assert!(registry.get("Test").is_some());
    assert!(registry.get("Other").is_none());
}

#[test]
fn test_registry_types() {
    let mut registry = MirrorRegistry::new();
    registry.register(WgslMirror::new("A", "a"));
    registry.register(WgslMirror::new("B", "b"));
    let types = registry.types();
    assert_eq!(types.len(), 2);
}

// ==================== gpu_sizes ====================

#[test]
fn test_gpu_sizes() {
    assert_eq!(gpu_sizes::VEC2_F32, 8);
    assert_eq!(gpu_sizes::VEC3_F32, 12);
    assert_eq!(gpu_sizes::VEC4_F32, 16);
    assert_eq!(gpu_sizes::MAT4X4_F32, 64);
}

// ==================== check_layout / get_layout ====================

#[test]
fn test_get_layout() {
    let (size, align) = get_layout::<u32>();
    assert_eq!(size, 4);
    assert_eq!(align, 4);
}

#[test]
fn test_check_layout() {
    let spec = LayoutSpec::new().size(4).align(4);
    let result = check_layout::<u32>(&spec);
    assert!(result.is_ok());
}

// ==================== assert_layout macro ====================

#[test]
fn test_assert_layout_macro() {
    // These are compile-time checks
    trinity_contracts::assert_layout!(u32, size = 4);
    trinity_contracts::assert_layout!(u64, align = 8);
    trinity_contracts::assert_layout!(u32, size = 4, align = 4);
}
