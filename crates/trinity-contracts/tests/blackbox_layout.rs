//! Blackbox tests for layout contracts with GPU-like structs.

use trinity_contracts::layout::{
    check_layout, get_layout, gpu_sizes, LayoutSpec, MirrorRegistry, WgslMirror,
};

// Simulated GPU structs
#[repr(C)]
struct Vec2 {
    x: f32,
    y: f32,
}

#[repr(C)]
struct Vec4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[repr(C)]
struct Mat4x4 {
    cols: [Vec4; 4],
}

#[repr(C)]
struct GpuVertex {
    position: Vec4,
    normal: Vec4,
    uv: Vec2,
}

#[test]
fn test_vec2_layout() {
    let spec = LayoutSpec::new().size(gpu_sizes::VEC2_F32);
    let result = check_layout::<Vec2>(&spec);
    assert!(result.is_ok());
}

#[test]
fn test_vec4_layout() {
    let spec = LayoutSpec::new().size(gpu_sizes::VEC4_F32).align(16);
    let (size, _align) = get_layout::<Vec4>();
    assert_eq!(size, 16);
}

#[test]
fn test_mat4x4_layout() {
    let spec = LayoutSpec::new().size(gpu_sizes::MAT4X4_F32);
    let result = check_layout::<Mat4x4>(&spec);
    assert!(result.is_ok());
}

#[test]
fn test_gpu_vertex_registry() {
    let mut registry = MirrorRegistry::new();

    registry.register(
        WgslMirror::new("Vec2", "vec2<f32>")
            .layout(LayoutSpec::new().size(8)),
    );
    registry.register(
        WgslMirror::new("Vec4", "vec4<f32>")
            .layout(LayoutSpec::new().size(16)),
    );
    registry.register(
        WgslMirror::new("Mat4x4", "mat4x4<f32>")
            .layout(LayoutSpec::new().size(64)),
    );

    assert_eq!(registry.len(), 3);

    // Verify Vec4
    let mirror = registry.get("Vec4").unwrap();
    let result = mirror.verify(16, 4);
    assert!(result.is_ok());
}

#[test]
fn test_verify_all_mirrors() {
    let mut registry = MirrorRegistry::new();

    registry.register(
        WgslMirror::new("u32", "u32")
            .layout(LayoutSpec::new().size(4).align(4)),
    );
    registry.register(
        WgslMirror::new("f32", "f32")
            .layout(LayoutSpec::new().size(4).align(4)),
    );

    let errors = registry.verify_all(|name| {
        match name {
            "u32" => Some((4, 4)),
            "f32" => Some((4, 4)),
            _ => None,
        }
    });

    assert!(errors.is_empty());
}

#[test]
fn test_detect_layout_mismatch() {
    let mut registry = MirrorRegistry::new();

    registry.register(
        WgslMirror::new("BadStruct", "bad_struct")
            .layout(LayoutSpec::new().size(32)),  // Wrong size
    );

    let errors = registry.verify_all(|name| {
        match name {
            "BadStruct" => Some((16, 4)),  // Actual is 16, expected 32
            _ => None,
        }
    });

    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("size"));
}

#[test]
fn test_packed_struct_simulation() {
    // Simulating packed struct verification
    let spec = LayoutSpec::new().size(12).packed();

    // Check that packed flag is set
    assert!(spec.packed);

    // Size should match exactly
    assert!(spec.check_size(12));
}
