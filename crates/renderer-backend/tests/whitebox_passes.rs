// WHITEBOX tests for T-WGPU-P7.5.5 (Render Pass Declaration)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/frame_graph/passes.rs
//   - PassLoadOp (Clear, Load, DontCare)
//   - PassStoreOp (Store, Discard)
//   - PassViewport
//   - PassColorAttachment
//   - PassDepthAttachment
//   - RenderPassConfig
//   - RenderPassBuilder
//   - PassExecutor trait
//   - NoOpExecutor
//   - FnExecutor
//   - RenderPassNode
//
// WHITEBOX coverage plan:
//   - PassLoadOp: All variants, predicates, default, Clone/Copy, PartialEq, Eq, Hash, Display, wgpu conversions
//   - PassStoreOp: All variants, predicates, default, Clone/Copy, PartialEq, Eq, Hash, Display, wgpu conversion
//   - PassViewport: Construction, defaults, validation, aspect ratio, depth range, Display
//   - PassColorAttachment: Construction patterns, MSAA resolve, transient, referenced_resources, Display
//   - PassDepthAttachment: Construction patterns, read-only, stencil, write predicates, Display
//   - RenderPassConfig: Construction patterns, validation, resource tracking, Display
//   - RenderPassBuilder: Fluent API, dependency tracking, build variants
//   - PassExecutor: NoOpExecutor, FnExecutor with name
//   - RenderPassNode: Construction, resource tracking, Display/Debug

use renderer_backend::frame_graph::graph::{PassId, RenderContext, ResourceId};
use renderer_backend::frame_graph::passes::{
    FnExecutor, NoOpExecutor, PassColorAttachment, PassDepthAttachment, PassExecutor, PassLoadOp,
    PassStoreOp, PassViewport, RenderPassBuilder, RenderPassConfig, RenderPassNode,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a test resource ID
fn resource(id: u64) -> ResourceId {
    ResourceId::new(id)
}

/// Create a test pass ID
fn pass(id: u64) -> PassId {
    PassId::new(id)
}

/// Hash a value for testing Hash trait
fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// Section 1: PassLoadOp Tests (20+ tests)
// ============================================================================

// --- 1.1: Variant Construction and Identity ---

#[test]
fn test_load_op_clear_variant() {
    let op = PassLoadOp::Clear;
    assert!(op.is_clear());
    assert!(!op.is_load());
    assert!(!op.is_dont_care());
}

#[test]
fn test_load_op_load_variant() {
    let op = PassLoadOp::Load;
    assert!(!op.is_clear());
    assert!(op.is_load());
    assert!(!op.is_dont_care());
}

#[test]
fn test_load_op_dont_care_variant() {
    let op = PassLoadOp::DontCare;
    assert!(!op.is_clear());
    assert!(!op.is_load());
    assert!(op.is_dont_care());
}

#[test]
fn test_load_op_default_is_clear() {
    let op = PassLoadOp::default();
    assert_eq!(op, PassLoadOp::Clear);
    assert!(op.is_clear());
}

// --- 1.2: Clone/Copy Semantics ---

#[test]
fn test_load_op_clone() {
    let op = PassLoadOp::Load;
    let cloned = op.clone();
    assert_eq!(op, cloned);
}

#[test]
fn test_load_op_copy() {
    let op = PassLoadOp::DontCare;
    let copied = op; // Copy
    let original = op; // Still valid because Copy
    assert_eq!(copied, original);
}

// --- 1.3: Equality and Hashing ---

#[test]
fn test_load_op_partial_eq_same() {
    assert_eq!(PassLoadOp::Clear, PassLoadOp::Clear);
    assert_eq!(PassLoadOp::Load, PassLoadOp::Load);
    assert_eq!(PassLoadOp::DontCare, PassLoadOp::DontCare);
}

#[test]
fn test_load_op_partial_eq_different() {
    assert_ne!(PassLoadOp::Clear, PassLoadOp::Load);
    assert_ne!(PassLoadOp::Clear, PassLoadOp::DontCare);
    assert_ne!(PassLoadOp::Load, PassLoadOp::DontCare);
}

#[test]
fn test_load_op_hash_distinct() {
    let h1 = hash_value(&PassLoadOp::Clear);
    let h2 = hash_value(&PassLoadOp::Load);
    let h3 = hash_value(&PassLoadOp::DontCare);

    // Different variants should have different hashes (with high probability)
    assert_ne!(h1, h2);
    assert_ne!(h1, h3);
    assert_ne!(h2, h3);
}

#[test]
fn test_load_op_hash_consistent() {
    let h1 = hash_value(&PassLoadOp::Clear);
    let h2 = hash_value(&PassLoadOp::Clear);
    assert_eq!(h1, h2);
}

// --- 1.4: Debug and Display ---

#[test]
fn test_load_op_debug_format() {
    let debug_clear = format!("{:?}", PassLoadOp::Clear);
    let debug_load = format!("{:?}", PassLoadOp::Load);
    let debug_dont_care = format!("{:?}", PassLoadOp::DontCare);

    assert!(debug_clear.contains("Clear"));
    assert!(debug_load.contains("Load"));
    assert!(debug_dont_care.contains("DontCare"));
}

#[test]
fn test_load_op_display_format() {
    assert_eq!(format!("{}", PassLoadOp::Clear), "Clear");
    assert_eq!(format!("{}", PassLoadOp::Load), "Load");
    assert_eq!(format!("{}", PassLoadOp::DontCare), "DontCare");
}

// --- 1.5: WGPU Conversions ---

#[test]
fn test_load_op_to_wgpu_color_clear_with_color() {
    let clear_color = Some([1.0_f32, 0.5, 0.25, 0.75]);
    let wgpu_op = PassLoadOp::Clear.to_wgpu_color(clear_color);

    match wgpu_op {
        wgpu::LoadOp::Clear(color) => {
            assert!((color.r - 1.0).abs() < 0.001);
            assert!((color.g - 0.5).abs() < 0.001);
            assert!((color.b - 0.25).abs() < 0.001);
            assert!((color.a - 0.75).abs() < 0.001);
        }
        _ => panic!("Expected Clear operation"),
    }
}

#[test]
fn test_load_op_to_wgpu_color_clear_default() {
    let wgpu_op = PassLoadOp::Clear.to_wgpu_color(None);

    match wgpu_op {
        wgpu::LoadOp::Clear(color) => {
            // Default is black with alpha 1.0
            assert!((color.r - 0.0).abs() < 0.001);
            assert!((color.g - 0.0).abs() < 0.001);
            assert!((color.b - 0.0).abs() < 0.001);
            assert!((color.a - 1.0).abs() < 0.001);
        }
        _ => panic!("Expected Clear operation"),
    }
}

#[test]
fn test_load_op_to_wgpu_color_load() {
    let wgpu_op = PassLoadOp::Load.to_wgpu_color(None);
    assert!(matches!(wgpu_op, wgpu::LoadOp::Load));
}

#[test]
fn test_load_op_to_wgpu_color_dont_care() {
    // DontCare maps to Load in wgpu (no direct DontCare for color)
    let wgpu_op = PassLoadOp::DontCare.to_wgpu_color(None);
    assert!(matches!(wgpu_op, wgpu::LoadOp::Load));
}

#[test]
fn test_load_op_to_wgpu_depth_clear() {
    let wgpu_op = PassLoadOp::Clear.to_wgpu_depth(0.5);
    match wgpu_op {
        wgpu::LoadOp::Clear(depth) => {
            assert!((depth - 0.5).abs() < 0.001);
        }
        _ => panic!("Expected Clear operation"),
    }
}

#[test]
fn test_load_op_to_wgpu_depth_load() {
    let wgpu_op = PassLoadOp::Load.to_wgpu_depth(1.0);
    assert!(matches!(wgpu_op, wgpu::LoadOp::Load));
}

#[test]
fn test_load_op_to_wgpu_stencil_clear() {
    let wgpu_op = PassLoadOp::Clear.to_wgpu_stencil(128);
    match wgpu_op {
        wgpu::LoadOp::Clear(stencil) => {
            assert_eq!(stencil, 128);
        }
        _ => panic!("Expected Clear operation"),
    }
}

#[test]
fn test_load_op_to_wgpu_stencil_load() {
    let wgpu_op = PassLoadOp::Load.to_wgpu_stencil(0);
    assert!(matches!(wgpu_op, wgpu::LoadOp::Load));
}

// ============================================================================
// Section 2: PassStoreOp Tests (15+ tests)
// ============================================================================

// --- 2.1: Variant Construction and Identity ---

#[test]
fn test_store_op_store_variant() {
    let op = PassStoreOp::Store;
    assert!(op.is_store());
    assert!(!op.is_discard());
}

#[test]
fn test_store_op_discard_variant() {
    let op = PassStoreOp::Discard;
    assert!(!op.is_store());
    assert!(op.is_discard());
}

#[test]
fn test_store_op_default_is_store() {
    let op = PassStoreOp::default();
    assert_eq!(op, PassStoreOp::Store);
    assert!(op.is_store());
}

// --- 2.2: Clone/Copy Semantics ---

#[test]
fn test_store_op_clone() {
    let op = PassStoreOp::Discard;
    let cloned = op.clone();
    assert_eq!(op, cloned);
}

#[test]
fn test_store_op_copy() {
    let op = PassStoreOp::Store;
    let copied = op;
    let original = op;
    assert_eq!(copied, original);
}

// --- 2.3: Equality and Hashing ---

#[test]
fn test_store_op_partial_eq_same() {
    assert_eq!(PassStoreOp::Store, PassStoreOp::Store);
    assert_eq!(PassStoreOp::Discard, PassStoreOp::Discard);
}

#[test]
fn test_store_op_partial_eq_different() {
    assert_ne!(PassStoreOp::Store, PassStoreOp::Discard);
}

#[test]
fn test_store_op_hash_distinct() {
    let h1 = hash_value(&PassStoreOp::Store);
    let h2 = hash_value(&PassStoreOp::Discard);
    assert_ne!(h1, h2);
}

#[test]
fn test_store_op_hash_consistent() {
    let h1 = hash_value(&PassStoreOp::Store);
    let h2 = hash_value(&PassStoreOp::Store);
    assert_eq!(h1, h2);
}

// --- 2.4: Debug and Display ---

#[test]
fn test_store_op_debug_format() {
    let debug_store = format!("{:?}", PassStoreOp::Store);
    let debug_discard = format!("{:?}", PassStoreOp::Discard);

    assert!(debug_store.contains("Store"));
    assert!(debug_discard.contains("Discard"));
}

#[test]
fn test_store_op_display_format() {
    assert_eq!(format!("{}", PassStoreOp::Store), "Store");
    assert_eq!(format!("{}", PassStoreOp::Discard), "Discard");
}

// --- 2.5: WGPU Conversion ---

#[test]
fn test_store_op_to_wgpu_store() {
    let wgpu_op = PassStoreOp::Store.to_wgpu();
    assert_eq!(wgpu_op, wgpu::StoreOp::Store);
}

#[test]
fn test_store_op_to_wgpu_discard() {
    let wgpu_op = PassStoreOp::Discard.to_wgpu();
    assert_eq!(wgpu_op, wgpu::StoreOp::Discard);
}

#[test]
fn test_store_op_to_wgpu_is_const() {
    // Verify const-ness by using in const context
    const STORE_OP: wgpu::StoreOp = PassStoreOp::Store.to_wgpu();
    assert_eq!(STORE_OP, wgpu::StoreOp::Store);
}

// ============================================================================
// Section 3: PassViewport Tests (25+ tests)
// ============================================================================

// --- 3.1: Default Construction ---

#[test]
fn test_viewport_default_dimensions() {
    let vp = PassViewport::default();
    assert_eq!(vp.x, 0.0);
    assert_eq!(vp.y, 0.0);
    assert_eq!(vp.width, 1920.0);
    assert_eq!(vp.height, 1080.0);
}

#[test]
fn test_viewport_default_depth_range() {
    let vp = PassViewport::default();
    assert_eq!(vp.min_depth, 0.0);
    assert_eq!(vp.max_depth, 1.0);
}

#[test]
fn test_viewport_default_is_valid() {
    let vp = PassViewport::default();
    assert!(vp.is_valid());
}

// --- 3.2: Custom Construction ---

#[test]
fn test_viewport_new_with_offset() {
    let vp = PassViewport::new(100.0, 50.0, 800.0, 600.0);
    assert_eq!(vp.x, 100.0);
    assert_eq!(vp.y, 50.0);
    assert_eq!(vp.width, 800.0);
    assert_eq!(vp.height, 600.0);
    assert_eq!(vp.min_depth, 0.0);
    assert_eq!(vp.max_depth, 1.0);
}

#[test]
fn test_viewport_with_size() {
    let vp = PassViewport::with_size(1280.0, 720.0);
    assert_eq!(vp.x, 0.0);
    assert_eq!(vp.y, 0.0);
    assert_eq!(vp.width, 1280.0);
    assert_eq!(vp.height, 720.0);
}

#[test]
fn test_viewport_with_depth_range_builder() {
    let vp = PassViewport::new(0.0, 0.0, 640.0, 480.0).with_depth_range(0.1, 0.9);
    assert_eq!(vp.min_depth, 0.1);
    assert_eq!(vp.max_depth, 0.9);
}

#[test]
fn test_viewport_chained_builders() {
    let vp = PassViewport::with_size(1024.0, 768.0).with_depth_range(0.0, 0.5);
    assert_eq!(vp.width, 1024.0);
    assert_eq!(vp.height, 768.0);
    assert_eq!(vp.max_depth, 0.5);
    assert!(vp.is_valid());
}

// --- 3.3: Validation ---

#[test]
fn test_viewport_valid_positive_dimensions() {
    let vp = PassViewport::new(0.0, 0.0, 1.0, 1.0);
    assert!(vp.is_valid());
}

#[test]
fn test_viewport_invalid_zero_width() {
    let vp = PassViewport::new(0.0, 0.0, 0.0, 100.0);
    assert!(!vp.is_valid());
}

#[test]
fn test_viewport_invalid_zero_height() {
    let vp = PassViewport::new(0.0, 0.0, 100.0, 0.0);
    assert!(!vp.is_valid());
}

#[test]
fn test_viewport_invalid_negative_width() {
    let vp = PassViewport {
        x: 0.0,
        y: 0.0,
        width: -100.0,
        height: 100.0,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    assert!(!vp.is_valid());
}

#[test]
fn test_viewport_invalid_negative_height() {
    let vp = PassViewport {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: -100.0,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    assert!(!vp.is_valid());
}

#[test]
fn test_viewport_invalid_inverted_depth_range() {
    let vp = PassViewport::new(0.0, 0.0, 100.0, 100.0).with_depth_range(1.0, 0.0);
    assert!(!vp.is_valid());
}

#[test]
fn test_viewport_valid_equal_depth_range() {
    // min_depth == max_depth is valid (thin depth slice)
    let vp = PassViewport::new(0.0, 0.0, 100.0, 100.0).with_depth_range(0.5, 0.5);
    assert!(vp.is_valid());
}

#[test]
fn test_viewport_valid_full_depth_range() {
    let vp = PassViewport::new(0.0, 0.0, 100.0, 100.0).with_depth_range(0.0, 1.0);
    assert!(vp.is_valid());
}

// --- 3.4: Aspect Ratio ---

#[test]
fn test_viewport_aspect_ratio_16_9() {
    let vp = PassViewport::with_size(1920.0, 1080.0);
    let ratio = vp.aspect_ratio();
    assert!((ratio - (16.0 / 9.0)).abs() < 0.001);
}

#[test]
fn test_viewport_aspect_ratio_4_3() {
    let vp = PassViewport::with_size(1024.0, 768.0);
    let ratio = vp.aspect_ratio();
    assert!((ratio - (4.0 / 3.0)).abs() < 0.001);
}

#[test]
fn test_viewport_aspect_ratio_square() {
    let vp = PassViewport::with_size(512.0, 512.0);
    let ratio = vp.aspect_ratio();
    assert!((ratio - 1.0).abs() < 0.001);
}

#[test]
fn test_viewport_aspect_ratio_zero_height_returns_one() {
    let vp = PassViewport {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 0.0,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    // Division by zero protection returns 1.0
    assert_eq!(vp.aspect_ratio(), 1.0);
}

// --- 3.5: Clone and PartialEq ---

#[test]
fn test_viewport_clone() {
    let vp = PassViewport::new(10.0, 20.0, 800.0, 600.0).with_depth_range(0.1, 0.9);
    let cloned = vp.clone();
    assert_eq!(vp, cloned);
}

#[test]
fn test_viewport_partial_eq_same() {
    let vp1 = PassViewport::default();
    let vp2 = PassViewport::default();
    assert_eq!(vp1, vp2);
}

#[test]
fn test_viewport_partial_eq_different_position() {
    let vp1 = PassViewport::new(0.0, 0.0, 100.0, 100.0);
    let vp2 = PassViewport::new(1.0, 0.0, 100.0, 100.0);
    assert_ne!(vp1, vp2);
}

#[test]
fn test_viewport_partial_eq_different_size() {
    let vp1 = PassViewport::with_size(100.0, 100.0);
    let vp2 = PassViewport::with_size(100.0, 101.0);
    assert_ne!(vp1, vp2);
}

// --- 3.6: Display ---

#[test]
fn test_viewport_display_format() {
    let vp = PassViewport::new(10.0, 20.0, 800.0, 600.0).with_depth_range(0.0, 1.0);
    let display = format!("{}", vp);

    assert!(display.contains("Viewport"));
    assert!(display.contains("10"));
    assert!(display.contains("20"));
    assert!(display.contains("800"));
    assert!(display.contains("600"));
}

// ============================================================================
// Section 4: PassColorAttachment Tests (35+ tests)
// ============================================================================

// --- 4.1: Default Construction ---

#[test]
fn test_color_attachment_new_defaults() {
    let att = PassColorAttachment::new(resource(1));

    assert_eq!(att.resource, resource(1));
    assert_eq!(att.load_op, PassLoadOp::Clear);
    assert_eq!(att.store_op, PassStoreOp::Store);
    assert_eq!(att.clear_color, Some([0.0, 0.0, 0.0, 1.0]));
    assert!(att.resolve_target.is_none());
}

#[test]
fn test_color_attachment_new_no_resolve() {
    let att = PassColorAttachment::new(resource(42));
    assert!(!att.has_resolve());
}

// --- 4.2: Static Constructors ---

#[test]
fn test_color_attachment_load_constructor() {
    let att = PassColorAttachment::load(resource(5));

    assert_eq!(att.resource, resource(5));
    assert_eq!(att.load_op, PassLoadOp::Load);
    assert_eq!(att.store_op, PassStoreOp::Store);
    assert!(att.clear_color.is_none());
    assert!(att.resolve_target.is_none());
}

#[test]
fn test_color_attachment_clear_constructor() {
    let color = [1.0, 0.5, 0.25, 0.75];
    let att = PassColorAttachment::clear(resource(10), color);

    assert_eq!(att.resource, resource(10));
    assert_eq!(att.load_op, PassLoadOp::Clear);
    assert_eq!(att.store_op, PassStoreOp::Store);
    assert_eq!(att.clear_color, Some(color));
}

#[test]
fn test_color_attachment_transient_constructor() {
    let att = PassColorAttachment::transient(resource(15));

    assert_eq!(att.resource, resource(15));
    assert_eq!(att.load_op, PassLoadOp::DontCare);
    assert_eq!(att.store_op, PassStoreOp::Discard);
    assert!(att.clear_color.is_none());
    assert!(att.resolve_target.is_none());
}

// --- 4.3: Builder Methods ---

#[test]
fn test_color_attachment_with_resolve() {
    let att = PassColorAttachment::new(resource(1)).with_resolve(resource(2));

    assert!(att.has_resolve());
    assert_eq!(att.resolve_target, Some(resource(2)));
}

#[test]
fn test_color_attachment_with_clear_color() {
    let color = [0.5, 0.6, 0.7, 0.8];
    let att = PassColorAttachment::load(resource(1)).with_clear_color(color);

    // with_clear_color also sets load_op to Clear
    assert_eq!(att.load_op, PassLoadOp::Clear);
    assert_eq!(att.clear_color, Some(color));
}

#[test]
fn test_color_attachment_chained_builders() {
    let att = PassColorAttachment::new(resource(1))
        .with_clear_color([1.0, 1.0, 1.0, 1.0])
        .with_resolve(resource(2));

    assert_eq!(att.clear_color, Some([1.0, 1.0, 1.0, 1.0]));
    assert_eq!(att.resolve_target, Some(resource(2)));
}

// --- 4.4: Clear Color Values ---

#[test]
fn test_color_attachment_clear_color_black() {
    let att = PassColorAttachment::clear(resource(1), [0.0, 0.0, 0.0, 1.0]);
    let color = att.clear_color.unwrap();
    assert_eq!(color, [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_color_attachment_clear_color_white() {
    let att = PassColorAttachment::clear(resource(1), [1.0, 1.0, 1.0, 1.0]);
    let color = att.clear_color.unwrap();
    assert_eq!(color, [1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_color_attachment_clear_color_transparent() {
    let att = PassColorAttachment::clear(resource(1), [0.0, 0.0, 0.0, 0.0]);
    let color = att.clear_color.unwrap();
    assert_eq!(color[3], 0.0);
}

#[test]
fn test_color_attachment_clear_color_hdr_values() {
    // HDR values can exceed 1.0
    let att = PassColorAttachment::clear(resource(1), [2.0, 3.0, 4.0, 1.0]);
    let color = att.clear_color.unwrap();
    assert_eq!(color[0], 2.0);
    assert_eq!(color[1], 3.0);
    assert_eq!(color[2], 4.0);
}

// --- 4.5: Resource References ---

#[test]
fn test_color_attachment_referenced_resources_single() {
    let att = PassColorAttachment::new(resource(10));
    let refs = att.referenced_resources();

    assert_eq!(refs.len(), 1);
    assert!(refs.contains(&resource(10)));
}

#[test]
fn test_color_attachment_referenced_resources_with_resolve() {
    let att = PassColorAttachment::new(resource(10)).with_resolve(resource(20));
    let refs = att.referenced_resources();

    assert_eq!(refs.len(), 2);
    assert!(refs.contains(&resource(10)));
    assert!(refs.contains(&resource(20)));
}

#[test]
fn test_color_attachment_has_resolve_false() {
    let att = PassColorAttachment::new(resource(1));
    assert!(!att.has_resolve());
}

#[test]
fn test_color_attachment_has_resolve_true() {
    let att = PassColorAttachment::new(resource(1)).with_resolve(resource(2));
    assert!(att.has_resolve());
}

// --- 4.6: Load/Store Combinations ---

#[test]
fn test_color_attachment_clear_store() {
    let att = PassColorAttachment::new(resource(1));
    assert_eq!(att.load_op, PassLoadOp::Clear);
    assert_eq!(att.store_op, PassStoreOp::Store);
}

#[test]
fn test_color_attachment_load_store() {
    let att = PassColorAttachment::load(resource(1));
    assert_eq!(att.load_op, PassLoadOp::Load);
    assert_eq!(att.store_op, PassStoreOp::Store);
}

#[test]
fn test_color_attachment_dont_care_discard() {
    let att = PassColorAttachment::transient(resource(1));
    assert_eq!(att.load_op, PassLoadOp::DontCare);
    assert_eq!(att.store_op, PassStoreOp::Discard);
}

#[test]
fn test_color_attachment_custom_ops() {
    let att = PassColorAttachment {
        resource: resource(1),
        load_op: PassLoadOp::Load,
        store_op: PassStoreOp::Discard,
        clear_color: None,
        resolve_target: None,
    };
    assert_eq!(att.load_op, PassLoadOp::Load);
    assert_eq!(att.store_op, PassStoreOp::Discard);
}

// --- 4.7: Clone and PartialEq ---

#[test]
fn test_color_attachment_clone() {
    let att = PassColorAttachment::new(resource(5)).with_resolve(resource(10));
    let cloned = att.clone();
    assert_eq!(att, cloned);
}

#[test]
fn test_color_attachment_partial_eq_same() {
    let att1 = PassColorAttachment::new(resource(1));
    let att2 = PassColorAttachment::new(resource(1));
    assert_eq!(att1, att2);
}

#[test]
fn test_color_attachment_partial_eq_different_resource() {
    let att1 = PassColorAttachment::new(resource(1));
    let att2 = PassColorAttachment::new(resource(2));
    assert_ne!(att1, att2);
}

#[test]
fn test_color_attachment_partial_eq_different_clear_color() {
    let att1 = PassColorAttachment::clear(resource(1), [0.0, 0.0, 0.0, 1.0]);
    let att2 = PassColorAttachment::clear(resource(1), [1.0, 0.0, 0.0, 1.0]);
    assert_ne!(att1, att2);
}

// --- 4.8: Display ---

#[test]
fn test_color_attachment_display_basic() {
    let att = PassColorAttachment::new(resource(42));
    let display = format!("{}", att);

    assert!(display.contains("ColorAttachment"));
    assert!(display.contains("42"));
    assert!(display.contains("Clear"));
    assert!(display.contains("Store"));
}

#[test]
fn test_color_attachment_display_with_resolve() {
    let att = PassColorAttachment::new(resource(1)).with_resolve(resource(2));
    let display = format!("{}", att);

    assert!(display.contains("resolve"));
    assert!(display.contains("2") || display.contains("Some"));
}

// ============================================================================
// Section 5: PassDepthAttachment Tests (30+ tests)
// ============================================================================

// --- 5.1: Default Construction ---

#[test]
fn test_depth_attachment_new_defaults() {
    let att = PassDepthAttachment::new(resource(1));

    assert_eq!(att.resource, resource(1));
    assert_eq!(att.depth_load_op, PassLoadOp::Clear);
    assert_eq!(att.depth_store_op, PassStoreOp::Store);
    assert_eq!(att.stencil_load_op, PassLoadOp::DontCare);
    assert_eq!(att.stencil_store_op, PassStoreOp::Discard);
    assert_eq!(att.clear_depth, 1.0);
    assert_eq!(att.clear_stencil, 0);
    assert!(!att.read_only);
}

// --- 5.2: Static Constructors ---

#[test]
fn test_depth_attachment_load_constructor() {
    let att = PassDepthAttachment::load(resource(5));

    assert_eq!(att.resource, resource(5));
    assert_eq!(att.depth_load_op, PassLoadOp::Load);
    assert_eq!(att.depth_store_op, PassStoreOp::Store);
    assert!(!att.read_only);
}

#[test]
fn test_depth_attachment_read_only_constructor() {
    let att = PassDepthAttachment::read_only(resource(10));

    assert_eq!(att.resource, resource(10));
    assert_eq!(att.depth_load_op, PassLoadOp::Load);
    assert!(att.read_only);
}

#[test]
fn test_depth_attachment_with_stencil_constructor() {
    let att = PassDepthAttachment::with_stencil(resource(15));

    assert_eq!(att.resource, resource(15));
    assert_eq!(att.depth_load_op, PassLoadOp::Clear);
    assert_eq!(att.depth_store_op, PassStoreOp::Store);
    assert_eq!(att.stencil_load_op, PassLoadOp::Clear);
    assert_eq!(att.stencil_store_op, PassStoreOp::Store);
    assert!(!att.read_only);
}

// --- 5.3: Builder Methods ---

#[test]
fn test_depth_attachment_with_clear_depth() {
    let att = PassDepthAttachment::new(resource(1)).with_clear_depth(0.5);

    assert_eq!(att.clear_depth, 0.5);
    assert_eq!(att.depth_load_op, PassLoadOp::Clear);
}

#[test]
fn test_depth_attachment_with_clear_stencil() {
    let att = PassDepthAttachment::with_stencil(resource(1)).with_clear_stencil(128);

    assert_eq!(att.clear_stencil, 128);
    assert_eq!(att.stencil_load_op, PassLoadOp::Clear);
}

#[test]
fn test_depth_attachment_make_read_only() {
    let att = PassDepthAttachment::new(resource(1)).make_read_only();

    assert!(att.read_only);
}

#[test]
fn test_depth_attachment_chained_builders() {
    let att = PassDepthAttachment::with_stencil(resource(1))
        .with_clear_depth(0.0)
        .with_clear_stencil(255)
        .make_read_only();

    assert_eq!(att.clear_depth, 0.0);
    assert_eq!(att.clear_stencil, 255);
    assert!(att.read_only);
}

// --- 5.4: Write Predicates ---

#[test]
fn test_depth_attachment_writes_depth_default() {
    let att = PassDepthAttachment::new(resource(1));
    assert!(att.writes_depth());
}

#[test]
fn test_depth_attachment_writes_depth_read_only() {
    let att = PassDepthAttachment::read_only(resource(1));
    assert!(!att.writes_depth());
}

#[test]
fn test_depth_attachment_writes_depth_discard() {
    let att = PassDepthAttachment {
        resource: resource(1),
        depth_load_op: PassLoadOp::Clear,
        depth_store_op: PassStoreOp::Discard,
        stencil_load_op: PassLoadOp::DontCare,
        stencil_store_op: PassStoreOp::Discard,
        clear_depth: 1.0,
        clear_stencil: 0,
        read_only: false,
    };
    assert!(!att.writes_depth());
}

#[test]
fn test_depth_attachment_writes_stencil_default() {
    let att = PassDepthAttachment::new(resource(1));
    // Default stencil_store_op is Discard
    assert!(!att.writes_stencil());
}

#[test]
fn test_depth_attachment_writes_stencil_with_stencil() {
    let att = PassDepthAttachment::with_stencil(resource(1));
    assert!(att.writes_stencil());
}

#[test]
fn test_depth_attachment_writes_stencil_read_only() {
    let att = PassDepthAttachment::with_stencil(resource(1)).make_read_only();
    assert!(!att.writes_stencil());
}

// --- 5.5: Clear Values ---

#[test]
fn test_depth_attachment_clear_depth_zero() {
    let att = PassDepthAttachment::new(resource(1)).with_clear_depth(0.0);
    assert_eq!(att.clear_depth, 0.0);
}

#[test]
fn test_depth_attachment_clear_depth_one() {
    let att = PassDepthAttachment::new(resource(1));
    assert_eq!(att.clear_depth, 1.0);
}

#[test]
fn test_depth_attachment_clear_depth_mid() {
    let att = PassDepthAttachment::new(resource(1)).with_clear_depth(0.5);
    assert!((att.clear_depth - 0.5).abs() < 0.001);
}

#[test]
fn test_depth_attachment_clear_stencil_zero() {
    let att = PassDepthAttachment::new(resource(1));
    assert_eq!(att.clear_stencil, 0);
}

#[test]
fn test_depth_attachment_clear_stencil_max() {
    let att = PassDepthAttachment::with_stencil(resource(1)).with_clear_stencil(255);
    assert_eq!(att.clear_stencil, 255);
}

#[test]
fn test_depth_attachment_clear_stencil_mid() {
    let att = PassDepthAttachment::with_stencil(resource(1)).with_clear_stencil(128);
    assert_eq!(att.clear_stencil, 128);
}

// --- 5.6: Load/Store Combinations ---

#[test]
fn test_depth_attachment_clear_store() {
    let att = PassDepthAttachment::new(resource(1));
    assert_eq!(att.depth_load_op, PassLoadOp::Clear);
    assert_eq!(att.depth_store_op, PassStoreOp::Store);
}

#[test]
fn test_depth_attachment_load_store() {
    let att = PassDepthAttachment::load(resource(1));
    assert_eq!(att.depth_load_op, PassLoadOp::Load);
    assert_eq!(att.depth_store_op, PassStoreOp::Store);
}

#[test]
fn test_depth_attachment_stencil_dont_care_discard() {
    let att = PassDepthAttachment::new(resource(1));
    assert_eq!(att.stencil_load_op, PassLoadOp::DontCare);
    assert_eq!(att.stencil_store_op, PassStoreOp::Discard);
}

// --- 5.7: Clone and PartialEq ---

#[test]
fn test_depth_attachment_clone() {
    let att = PassDepthAttachment::with_stencil(resource(5))
        .with_clear_depth(0.5)
        .with_clear_stencil(100);
    let cloned = att.clone();
    assert_eq!(att, cloned);
}

#[test]
fn test_depth_attachment_partial_eq_same() {
    let att1 = PassDepthAttachment::new(resource(1));
    let att2 = PassDepthAttachment::new(resource(1));
    assert_eq!(att1, att2);
}

#[test]
fn test_depth_attachment_partial_eq_different_resource() {
    let att1 = PassDepthAttachment::new(resource(1));
    let att2 = PassDepthAttachment::new(resource(2));
    assert_ne!(att1, att2);
}

#[test]
fn test_depth_attachment_partial_eq_different_read_only() {
    let att1 = PassDepthAttachment::new(resource(1));
    let att2 = PassDepthAttachment::new(resource(1)).make_read_only();
    assert_ne!(att1, att2);
}

// --- 5.8: Display ---

#[test]
fn test_depth_attachment_display_basic() {
    let att = PassDepthAttachment::new(resource(42));
    let display = format!("{}", att);

    assert!(display.contains("DepthAttachment"));
    assert!(display.contains("42"));
    assert!(display.contains("depth"));
}

#[test]
fn test_depth_attachment_display_read_only() {
    let att = PassDepthAttachment::read_only(resource(1));
    let display = format!("{}", att);

    assert!(display.contains("read_only=true"));
}

// ============================================================================
// Section 6: RenderPassConfig Tests (25+ tests)
// ============================================================================

// --- 6.1: Construction ---

#[test]
fn test_render_pass_config_new() {
    let config = RenderPassConfig::new("test_pass");

    assert_eq!(config.name, "test_pass");
    assert!(config.color_attachments.is_empty());
    assert!(config.depth_attachment.is_none());
    assert_eq!(config.sample_count, 1);
    assert!(config.viewport.is_none());
}

#[test]
fn test_render_pass_config_default() {
    let config = RenderPassConfig::default();

    assert_eq!(config.name, "unnamed_pass");
    assert!(config.color_attachments.is_empty());
}

#[test]
fn test_render_pass_config_with_color() {
    let color = PassColorAttachment::new(resource(1));
    let config = RenderPassConfig::with_color("color_pass", color);

    assert_eq!(config.name, "color_pass");
    assert_eq!(config.color_attachments.len(), 1);
    assert!(config.depth_attachment.is_none());
}

#[test]
fn test_render_pass_config_with_color_and_depth() {
    let color = PassColorAttachment::new(resource(1));
    let depth = PassDepthAttachment::new(resource(2));
    let config = RenderPassConfig::with_color_and_depth("full_pass", color, depth);

    assert_eq!(config.name, "full_pass");
    assert_eq!(config.color_attachments.len(), 1);
    assert!(config.depth_attachment.is_some());
}

// --- 6.2: Predicates ---

#[test]
fn test_render_pass_config_has_color_true() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    assert!(config.has_color());
}

#[test]
fn test_render_pass_config_has_color_false() {
    let config = RenderPassConfig::new("test");
    assert!(!config.has_color());
}

#[test]
fn test_render_pass_config_has_depth_true() {
    let config = RenderPassConfig::with_color_and_depth(
        "test",
        PassColorAttachment::new(resource(1)),
        PassDepthAttachment::new(resource(2)),
    );
    assert!(config.has_depth());
}

#[test]
fn test_render_pass_config_has_depth_false() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    assert!(!config.has_depth());
}

#[test]
fn test_render_pass_config_is_multisampled_false() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    assert!(!config.is_multisampled());
}

#[test]
fn test_render_pass_config_is_multisampled_true() {
    let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    config.sample_count = 4;
    assert!(config.is_multisampled());
}

#[test]
fn test_render_pass_config_color_attachment_count() {
    let mut config = RenderPassConfig::new("test");
    assert_eq!(config.color_attachment_count(), 0);

    config
        .color_attachments
        .push(PassColorAttachment::new(resource(1)));
    assert_eq!(config.color_attachment_count(), 1);

    config
        .color_attachments
        .push(PassColorAttachment::new(resource(2)));
    assert_eq!(config.color_attachment_count(), 2);
}

// --- 6.3: Validation ---

#[test]
fn test_render_pass_config_validate_empty_invalid() {
    let config = RenderPassConfig::new("empty");
    let error = config.validate();

    assert!(error.is_some());
    assert!(error.unwrap().contains("at least one attachment"));
}

#[test]
fn test_render_pass_config_validate_color_only_valid() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    assert!(config.validate().is_none());
}

#[test]
fn test_render_pass_config_validate_depth_only_valid() {
    let mut config = RenderPassConfig::new("depth_only");
    config.depth_attachment = Some(PassDepthAttachment::new(resource(1)));
    assert!(config.validate().is_none());
}

#[test]
fn test_render_pass_config_validate_sample_count_1() {
    let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    config.sample_count = 1;
    assert!(config.validate().is_none());
}

#[test]
fn test_render_pass_config_validate_sample_count_2() {
    let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    config.sample_count = 2;
    assert!(config.validate().is_none());
}

#[test]
fn test_render_pass_config_validate_sample_count_4() {
    let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    config.sample_count = 4;
    assert!(config.validate().is_none());
}

#[test]
fn test_render_pass_config_validate_sample_count_8() {
    let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    config.sample_count = 8;
    assert!(config.validate().is_none());
}

#[test]
fn test_render_pass_config_validate_sample_count_16() {
    let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    config.sample_count = 16;
    assert!(config.validate().is_none());
}

#[test]
fn test_render_pass_config_validate_sample_count_0_invalid() {
    let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    config.sample_count = 0;
    let error = config.validate();

    assert!(error.is_some());
    assert!(error.unwrap().contains("sample count"));
}

#[test]
fn test_render_pass_config_validate_sample_count_3_invalid() {
    let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    config.sample_count = 3;
    assert!(config.validate().is_some());
}

#[test]
fn test_render_pass_config_validate_sample_count_32_invalid() {
    let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    config.sample_count = 32;
    assert!(config.validate().is_some());
}

#[test]
fn test_render_pass_config_validate_too_many_colors() {
    let mut config = RenderPassConfig::new("too_many");
    for i in 0..9 {
        config
            .color_attachments
            .push(PassColorAttachment::new(resource(i)));
    }

    let error = config.validate();
    assert!(error.is_some());
    assert!(error.unwrap().contains("Too many color attachments"));
}

#[test]
fn test_render_pass_config_validate_max_colors_valid() {
    let mut config = RenderPassConfig::new("max_colors");
    for i in 0..8 {
        config
            .color_attachments
            .push(PassColorAttachment::new(resource(i)));
    }

    assert!(config.validate().is_none());
}

#[test]
fn test_render_pass_config_validate_invalid_viewport() {
    let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    config.viewport = Some(PassViewport::new(0.0, 0.0, 0.0, 100.0)); // Zero width

    let error = config.validate();
    assert!(error.is_some());
    assert!(error.unwrap().contains("viewport"));
}

// --- 6.4: Resource Tracking ---

#[test]
fn test_render_pass_config_written_resources_color() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(10)));
    let writes = config.written_resources();

    assert!(writes.contains(&resource(10)));
}

#[test]
fn test_render_pass_config_written_resources_color_discard() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::transient(resource(10)));
    let writes = config.written_resources();

    // Transient uses Discard, so not in written resources
    assert!(!writes.contains(&resource(10)));
}

#[test]
fn test_render_pass_config_written_resources_resolve() {
    let att = PassColorAttachment::new(resource(1)).with_resolve(resource(2));
    let config = RenderPassConfig::with_color("test", att);
    let writes = config.written_resources();

    assert!(writes.contains(&resource(1)));
    assert!(writes.contains(&resource(2)));
}

#[test]
fn test_render_pass_config_written_resources_depth() {
    let config = RenderPassConfig::with_color_and_depth(
        "test",
        PassColorAttachment::new(resource(1)),
        PassDepthAttachment::new(resource(2)),
    );
    let writes = config.written_resources();

    assert!(writes.contains(&resource(1)));
    assert!(writes.contains(&resource(2)));
}

#[test]
fn test_render_pass_config_written_resources_depth_read_only() {
    let config = RenderPassConfig::with_color_and_depth(
        "test",
        PassColorAttachment::new(resource(1)),
        PassDepthAttachment::read_only(resource(2)),
    );
    let writes = config.written_resources();

    assert!(writes.contains(&resource(1)));
    assert!(!writes.contains(&resource(2)));
}

#[test]
fn test_render_pass_config_read_resources_load() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::load(resource(10)));
    let reads = config.read_resources();

    assert!(reads.contains(&resource(10)));
}

#[test]
fn test_render_pass_config_read_resources_clear() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(10)));
    let reads = config.read_resources();

    // Clear doesn't read, it overwrites
    assert!(!reads.contains(&resource(10)));
}

#[test]
fn test_render_pass_config_read_resources_depth_load() {
    let config = RenderPassConfig::with_color_and_depth(
        "test",
        PassColorAttachment::new(resource(1)),
        PassDepthAttachment::load(resource(2)),
    );
    let reads = config.read_resources();

    assert!(reads.contains(&resource(2)));
}

#[test]
fn test_render_pass_config_read_resources_depth_read_only() {
    let config = RenderPassConfig::with_color_and_depth(
        "test",
        PassColorAttachment::new(resource(1)),
        PassDepthAttachment::read_only(resource(2)),
    );
    let reads = config.read_resources();

    assert!(reads.contains(&resource(2)));
}

// --- 6.5: Clone and PartialEq ---

#[test]
fn test_render_pass_config_clone() {
    let config = RenderPassConfig::with_color_and_depth(
        "test",
        PassColorAttachment::new(resource(1)),
        PassDepthAttachment::new(resource(2)),
    );
    let cloned = config.clone();
    assert_eq!(config, cloned);
}

#[test]
fn test_render_pass_config_partial_eq_same() {
    let config1 = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    let config2 = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    assert_eq!(config1, config2);
}

#[test]
fn test_render_pass_config_partial_eq_different_name() {
    let config1 = RenderPassConfig::with_color("test1", PassColorAttachment::new(resource(1)));
    let config2 = RenderPassConfig::with_color("test2", PassColorAttachment::new(resource(1)));
    assert_ne!(config1, config2);
}

// --- 6.6: Display ---

#[test]
fn test_render_pass_config_display() {
    let config = RenderPassConfig::with_color_and_depth(
        "my_pass",
        PassColorAttachment::new(resource(1)),
        PassDepthAttachment::new(resource(2)),
    );
    let display = format!("{}", config);

    assert!(display.contains("RenderPassConfig"));
    assert!(display.contains("my_pass"));
    assert!(display.contains("colors=1"));
    assert!(display.contains("depth=true"));
}

// ============================================================================
// Section 7: RenderPassBuilder Tests (25+ tests)
// ============================================================================

// --- 7.1: Construction ---

#[test]
fn test_render_pass_builder_new() {
    let builder = RenderPassBuilder::new("test_builder");
    let config = builder.build();

    assert_eq!(config.name, "test_builder");
    assert!(config.color_attachments.is_empty());
}

#[test]
fn test_render_pass_builder_default() {
    let builder = RenderPassBuilder::default();
    let config = builder.build();

    assert_eq!(config.name, "unnamed_pass");
}

// --- 7.2: Fluent API ---

#[test]
fn test_render_pass_builder_add_color_attachment() {
    let config = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .build();

    assert_eq!(config.color_attachments.len(), 1);
    assert_eq!(config.color_attachments[0].resource, resource(1));
}

#[test]
fn test_render_pass_builder_multiple_color_attachments() {
    let config = RenderPassBuilder::new("mrt")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .add_color_attachment(PassColorAttachment::new(resource(2)))
        .add_color_attachment(PassColorAttachment::new(resource(3)))
        .build();

    assert_eq!(config.color_attachments.len(), 3);
}

#[test]
fn test_render_pass_builder_set_depth_attachment() {
    let config = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .set_depth_attachment(PassDepthAttachment::new(resource(2)))
        .build();

    assert!(config.depth_attachment.is_some());
    assert_eq!(config.depth_attachment.unwrap().resource, resource(2));
}

#[test]
fn test_render_pass_builder_sample_count() {
    let config = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .sample_count(4)
        .build();

    assert_eq!(config.sample_count, 4);
}

#[test]
fn test_render_pass_builder_viewport() {
    let vp = PassViewport::with_size(1280.0, 720.0);
    let config = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .viewport(vp.clone())
        .build();

    assert_eq!(config.viewport, Some(vp));
}

#[test]
fn test_render_pass_builder_full_chain() {
    let config = RenderPassBuilder::new("full_pass")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .add_color_attachment(PassColorAttachment::new(resource(2)))
        .set_depth_attachment(PassDepthAttachment::new(resource(3)))
        .sample_count(8)
        .viewport(PassViewport::with_size(1920.0, 1080.0))
        .build();

    assert_eq!(config.name, "full_pass");
    assert_eq!(config.color_attachments.len(), 2);
    assert!(config.depth_attachment.is_some());
    assert_eq!(config.sample_count, 8);
    assert!(config.viewport.is_some());
    assert!(config.validate().is_none());
}

// --- 7.3: Explicit Resource Dependencies ---

#[test]
fn test_render_pass_builder_read_resource() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .read_resource(resource(10));

    assert!(builder.get_reads().contains(&resource(10)));
}

#[test]
fn test_render_pass_builder_write_resource() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .write_resource(resource(20));

    assert!(builder.get_writes().contains(&resource(20)));
}

#[test]
fn test_render_pass_builder_read_resource_dedup() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .read_resource(resource(10))
        .read_resource(resource(10)); // Duplicate

    // Should only appear once
    assert_eq!(builder.get_reads().iter().filter(|&&r| r == resource(10)).count(), 1);
}

#[test]
fn test_render_pass_builder_write_resource_dedup() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .write_resource(resource(20))
        .write_resource(resource(20)); // Duplicate

    assert_eq!(builder.get_writes().iter().filter(|&&r| r == resource(20)).count(), 1);
}

// --- 7.4: Automatic Dependency Tracking ---

#[test]
fn test_render_pass_builder_tracks_color_load_as_read() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::load(resource(1)));

    assert!(builder.get_reads().contains(&resource(1)));
}

#[test]
fn test_render_pass_builder_tracks_color_store_as_write() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1))); // Store by default

    assert!(builder.get_writes().contains(&resource(1)));
}

#[test]
fn test_render_pass_builder_tracks_resolve_as_write() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)).with_resolve(resource(2)));

    assert!(builder.get_writes().contains(&resource(2)));
}

#[test]
fn test_render_pass_builder_tracks_depth_load_as_read() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .set_depth_attachment(PassDepthAttachment::load(resource(2)));

    assert!(builder.get_reads().contains(&resource(2)));
}

#[test]
fn test_render_pass_builder_tracks_depth_read_only_as_read() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .set_depth_attachment(PassDepthAttachment::read_only(resource(2)));

    assert!(builder.get_reads().contains(&resource(2)));
}

#[test]
fn test_render_pass_builder_tracks_depth_write() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .set_depth_attachment(PassDepthAttachment::new(resource(2)));

    assert!(builder.get_writes().contains(&resource(2)));
}

// --- 7.5: Build Variants ---

#[test]
fn test_render_pass_builder_build() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)));

    let config = builder.build();
    assert_eq!(config.name, "test");
}

#[test]
fn test_render_pass_builder_build_with_deps() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::load(resource(1)))
        .read_resource(resource(10))
        .write_resource(resource(20));

    let (config, reads, writes) = builder.build_with_deps();

    assert_eq!(config.name, "test");
    assert!(reads.contains(&resource(1)));
    assert!(reads.contains(&resource(10)));
    assert!(writes.contains(&resource(20)));
}

// --- 7.6: Clone and Debug ---

#[test]
fn test_render_pass_builder_clone() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)))
        .sample_count(4);

    let cloned = builder.clone();
    let config1 = builder.build();
    let config2 = cloned.build();

    assert_eq!(config1, config2);
}

#[test]
fn test_render_pass_builder_debug() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)));

    let debug = format!("{:?}", builder);
    assert!(debug.contains("RenderPassBuilder"));
}

// ============================================================================
// Section 8: PassExecutor Tests (15+ tests)
// ============================================================================

// --- 8.1: NoOpExecutor ---

#[test]
fn test_no_op_executor_default() {
    let executor = NoOpExecutor::default();
    assert_eq!(executor.name(), "NoOpExecutor");
}

#[test]
fn test_no_op_executor_clone() {
    let executor = NoOpExecutor;
    let cloned = executor.clone();
    assert_eq!(executor.name(), cloned.name());
}

#[test]
fn test_no_op_executor_debug() {
    let executor = NoOpExecutor;
    let debug = format!("{:?}", executor);
    assert!(debug.contains("NoOpExecutor"));
}

#[test]
fn test_no_op_executor_name() {
    let executor = NoOpExecutor;
    assert_eq!(executor.name(), "NoOpExecutor");
}

// --- 8.2: FnExecutor ---

#[test]
fn test_fn_executor_new() {
    let executor = FnExecutor::new(|_ctx, _pass| {
        // No-op closure
    });
    assert_eq!(executor.name(), "FnExecutor");
}

#[test]
fn test_fn_executor_named() {
    let executor = FnExecutor::named("custom_executor", |_ctx, _pass| {
        // No-op closure
    });
    assert_eq!(executor.name(), "custom_executor");
}

#[test]
fn test_fn_executor_name_default() {
    let executor = FnExecutor::new(|_ctx, _pass| {});
    assert_eq!(executor.name(), "FnExecutor");
}

#[test]
fn test_fn_executor_name_custom() {
    let executor = FnExecutor::named("my_pass_executor", |_ctx, _pass| {});
    assert_eq!(executor.name(), "my_pass_executor");
}

// --- 8.3: PassExecutor Trait ---

#[test]
fn test_pass_executor_trait_default_name() {
    struct CustomExecutor;
    impl PassExecutor for CustomExecutor {
        fn execute(&self, _ctx: &mut RenderContext, _encoder: &mut wgpu::RenderPass) {}
    }

    let executor = CustomExecutor;
    assert_eq!(executor.name(), "PassExecutor");
}

#[test]
fn test_pass_executor_trait_custom_name() {
    struct NamedExecutor;
    impl PassExecutor for NamedExecutor {
        fn execute(&self, _ctx: &mut RenderContext, _encoder: &mut wgpu::RenderPass) {}
        fn name(&self) -> &str {
            "NamedExecutor"
        }
    }

    let executor = NamedExecutor;
    assert_eq!(executor.name(), "NamedExecutor");
}

// ============================================================================
// Section 9: RenderPassNode Tests (20+ tests)
// ============================================================================

// --- 9.1: Construction ---

#[test]
fn test_render_pass_node_empty() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    let node = RenderPassNode::empty(pass(42), config);

    assert_eq!(node.id, pass(42));
    assert_eq!(node.name(), "test");
    assert_eq!(node.executor.name(), "NoOpExecutor");
}

#[test]
fn test_render_pass_node_new() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    let executor = Box::new(NoOpExecutor);
    let node = RenderPassNode::new(pass(10), config, executor);

    assert_eq!(node.id, pass(10));
    assert_eq!(node.name(), "test");
}

#[test]
fn test_render_pass_node_with_fn() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    let node = RenderPassNode::with_fn(pass(5), config, |_ctx, _pass| {
        // Custom render logic
    });

    assert_eq!(node.id, pass(5));
    assert_eq!(node.executor.name(), "FnExecutor");
}

// --- 9.2: Name Access ---

#[test]
fn test_render_pass_node_name() {
    let config = RenderPassConfig::with_color("my_pass", PassColorAttachment::new(resource(1)));
    let node = RenderPassNode::empty(pass(1), config);

    assert_eq!(node.name(), "my_pass");
}

// --- 9.3: Resource Tracking ---

#[test]
fn test_render_pass_node_written_resources() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(10)));
    let node = RenderPassNode::empty(pass(1), config);
    let writes = node.written_resources();

    assert!(writes.contains(&resource(10)));
}

#[test]
fn test_render_pass_node_written_resources_multiple() {
    let mut config = RenderPassConfig::new("test");
    config.color_attachments.push(PassColorAttachment::new(resource(1)));
    config.color_attachments.push(PassColorAttachment::new(resource(2)));
    config.depth_attachment = Some(PassDepthAttachment::new(resource(3)));

    let node = RenderPassNode::empty(pass(1), config);
    let writes = node.written_resources();

    assert!(writes.contains(&resource(1)));
    assert!(writes.contains(&resource(2)));
    assert!(writes.contains(&resource(3)));
}

#[test]
fn test_render_pass_node_read_resources() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::load(resource(10)));
    let node = RenderPassNode::empty(pass(1), config);
    let reads = node.read_resources();

    assert!(reads.contains(&resource(10)));
}

#[test]
fn test_render_pass_node_read_resources_depth_read_only() {
    let mut config = RenderPassConfig::new("test");
    config.color_attachments.push(PassColorAttachment::new(resource(1)));
    config.depth_attachment = Some(PassDepthAttachment::read_only(resource(2)));

    let node = RenderPassNode::empty(pass(1), config);
    let reads = node.read_resources();

    assert!(reads.contains(&resource(2)));
}

// --- 9.4: Debug and Display ---

#[test]
fn test_render_pass_node_debug() {
    let config = RenderPassConfig::with_color("debug_pass", PassColorAttachment::new(resource(1)));
    let node = RenderPassNode::empty(pass(99), config);
    let debug = format!("{:?}", node);

    assert!(debug.contains("RenderPassNode"));
    assert!(debug.contains("99"));
    assert!(debug.contains("debug_pass"));
    assert!(debug.contains("NoOpExecutor"));
}

#[test]
fn test_render_pass_node_display() {
    let config = RenderPassConfig::with_color("display_pass", PassColorAttachment::new(resource(1)));
    let node = RenderPassNode::empty(pass(77), config);
    let display = format!("{}", node);

    assert!(display.contains("RenderPassNode"));
    assert!(display.contains("display_pass"));
    assert!(display.contains("NoOpExecutor"));
}

// --- 9.5: ID Access ---

#[test]
fn test_render_pass_node_id() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    let node = RenderPassNode::empty(pass(123), config);

    assert_eq!(node.id, pass(123));
    assert_eq!(node.id.raw(), 123);
}

// --- 9.6: Config Access ---

#[test]
fn test_render_pass_node_config_access() {
    let config = RenderPassConfig::with_color_and_depth(
        "config_test",
        PassColorAttachment::new(resource(1)),
        PassDepthAttachment::new(resource(2)),
    );
    let node = RenderPassNode::empty(pass(1), config);

    assert!(node.config.has_color());
    assert!(node.config.has_depth());
    assert_eq!(node.config.sample_count, 1);
}

// ============================================================================
// Section 10: Edge Cases and Integration Tests (15+ tests)
// ============================================================================

// --- 10.1: Boundary Values ---

#[test]
fn test_resource_id_zero() {
    let att = PassColorAttachment::new(resource(0));
    assert_eq!(att.resource.raw(), 0);
}

#[test]
fn test_resource_id_max_minus_one() {
    // u64::MAX - 1 is valid, u64::MAX is INVALID
    let att = PassColorAttachment::new(resource(u64::MAX - 1));
    assert_eq!(att.resource.raw(), u64::MAX - 1);
    assert!(!att.resource.is_invalid());
}

#[test]
fn test_viewport_tiny_dimensions() {
    let vp = PassViewport::new(0.0, 0.0, 0.001, 0.001);
    assert!(vp.is_valid());
}

#[test]
fn test_viewport_large_dimensions() {
    let vp = PassViewport::with_size(16384.0, 16384.0);
    assert!(vp.is_valid());
}

#[test]
fn test_clear_color_negative_values() {
    // Negative clear colors are technically valid (for signed formats)
    let att = PassColorAttachment::clear(resource(1), [-1.0, -0.5, 0.0, 1.0]);
    let color = att.clear_color.unwrap();
    assert_eq!(color[0], -1.0);
}

// --- 10.2: Empty/None Cases ---

#[test]
fn test_render_pass_config_empty_name() {
    let config = RenderPassConfig::new("");
    assert_eq!(config.name, "");
}

#[test]
fn test_render_pass_config_no_viewport() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
    assert!(config.viewport.is_none());
}

#[test]
fn test_color_attachment_no_clear_color() {
    let att = PassColorAttachment::load(resource(1));
    assert!(att.clear_color.is_none());
}

// --- 10.3: Builder Reset/Reuse ---

#[test]
fn test_render_pass_builder_build_consumes() {
    let builder = RenderPassBuilder::new("test")
        .add_color_attachment(PassColorAttachment::new(resource(1)));

    let _config = builder.build();
    // builder is consumed, cannot be used again (compile-time check)
}

// --- 10.4: Multiple Operations ---

#[test]
fn test_render_pass_config_mrt_8_colors() {
    let mut config = RenderPassConfig::new("mrt_8");
    for i in 0..8 {
        config
            .color_attachments
            .push(PassColorAttachment::new(resource(i)));
    }

    assert_eq!(config.color_attachment_count(), 8);
    assert!(config.validate().is_none());
}

#[test]
fn test_render_pass_config_all_transient() {
    let mut config = RenderPassConfig::new("all_transient");
    for i in 0..4 {
        config
            .color_attachments
            .push(PassColorAttachment::transient(resource(i)));
    }

    // All transient = no writes
    let writes = config.written_resources();
    assert!(writes.is_empty());
}

#[test]
fn test_render_pass_config_mixed_ops() {
    let mut config = RenderPassConfig::new("mixed");
    config.color_attachments.push(PassColorAttachment::new(resource(1))); // Clear/Store
    config.color_attachments.push(PassColorAttachment::load(resource(2))); // Load/Store
    config.color_attachments.push(PassColorAttachment::transient(resource(3))); // DontCare/Discard

    let writes = config.written_resources();
    let reads = config.read_resources();

    assert!(writes.contains(&resource(1)));
    assert!(writes.contains(&resource(2)));
    assert!(!writes.contains(&resource(3))); // Transient

    assert!(!reads.contains(&resource(1))); // Clear
    assert!(reads.contains(&resource(2))); // Load
    assert!(!reads.contains(&resource(3))); // DontCare
}

// --- 10.5: Depth/Stencil Combinations ---

#[test]
fn test_depth_only_pass() {
    let mut config = RenderPassConfig::new("depth_only");
    config.depth_attachment = Some(PassDepthAttachment::new(resource(1)));

    assert!(!config.has_color());
    assert!(config.has_depth());
    assert!(config.validate().is_none());
}

#[test]
fn test_depth_and_stencil_combined() {
    let depth = PassDepthAttachment::with_stencil(resource(1))
        .with_clear_depth(0.0)
        .with_clear_stencil(255);

    assert_eq!(depth.depth_load_op, PassLoadOp::Clear);
    assert_eq!(depth.stencil_load_op, PassLoadOp::Clear);
    assert!(depth.writes_depth());
    assert!(depth.writes_stencil());
}

#[test]
fn test_stencil_only_writes() {
    let depth = PassDepthAttachment {
        resource: resource(1),
        depth_load_op: PassLoadOp::Load,
        depth_store_op: PassStoreOp::Discard, // Don't store depth
        stencil_load_op: PassLoadOp::Clear,
        stencil_store_op: PassStoreOp::Store, // Store stencil
        clear_depth: 1.0,
        clear_stencil: 0,
        read_only: false,
    };

    assert!(!depth.writes_depth());
    assert!(depth.writes_stencil());
}

// ============================================================================
// Section 11: Stress Tests (5+ tests)
// ============================================================================

#[test]
fn test_many_render_pass_nodes() {
    let nodes: Vec<RenderPassNode> = (0..100)
        .map(|i| {
            let config = RenderPassConfig::with_color(
                format!("pass_{}", i),
                PassColorAttachment::new(resource(i)),
            );
            RenderPassNode::empty(pass(i), config)
        })
        .collect();

    assert_eq!(nodes.len(), 100);
    for (i, node) in nodes.iter().enumerate() {
        assert_eq!(node.id.raw(), i as u64);
    }
}

#[test]
fn test_many_color_attachments_validation() {
    // Test validation catches too many attachments
    let mut config = RenderPassConfig::new("stress");
    for i in 0..100 {
        config.color_attachments.push(PassColorAttachment::new(resource(i)));
    }

    let error = config.validate();
    assert!(error.is_some());
    assert!(error.unwrap().contains("Too many"));
}

#[test]
fn test_builder_many_reads_writes() {
    let mut builder = RenderPassBuilder::new("stress")
        .add_color_attachment(PassColorAttachment::new(resource(0)));

    for i in 1..50 {
        builder = builder.read_resource(resource(i));
    }
    for i in 50..100 {
        builder = builder.write_resource(resource(i));
    }

    let (_, reads, writes) = builder.build_with_deps();
    assert!(reads.len() >= 49);
    assert!(writes.len() >= 50);
}

#[test]
fn test_viewport_many_depth_ranges() {
    for i in 0..100 {
        let min = i as f32 / 200.0;
        let max = 0.5 + (i as f32 / 200.0);
        let vp = PassViewport::new(0.0, 0.0, 100.0, 100.0).with_depth_range(min, max);
        assert!(vp.is_valid());
    }
}

#[test]
fn test_sample_count_all_valid_values() {
    let valid_counts = [1, 2, 4, 8, 16];
    for count in valid_counts {
        let mut config = RenderPassConfig::with_color("test", PassColorAttachment::new(resource(1)));
        config.sample_count = count;
        assert!(
            config.validate().is_none(),
            "Sample count {} should be valid",
            count
        );
    }
}
