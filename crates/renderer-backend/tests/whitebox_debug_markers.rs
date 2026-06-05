//! Whitebox tests for GPU Debug Marker System (T-WGPU-P7.3.1)
//!
//! This test module provides comprehensive coverage of the debug marker system
//! including DebugLabel, DebugGroup, DebugMarkerStack, colors module, and edge cases.

use renderer_backend::debug::markers::{
    colors, DebugContextOps, DebugGroup, DebugLabel, DebugMarkerStack,
};
use std::borrow::Cow;
use std::collections::HashSet;
// Note: Hash and Hasher not needed for these tests
use std::thread;
use std::time::Duration;

// ============================================================================
// CATEGORY 1: DebugLabel Construction Tests (20+ tests)
// ============================================================================

#[test]
fn test_debug_label_new_simple_string() {
    let label = DebugLabel::new("Simple Label");
    assert_eq!(label.as_wgpu_label(), "Simple Label");
    assert!(label.color.is_none());
}

#[test]
fn test_debug_label_new_owned_string() {
    let s = String::from("Owned Label");
    let label = DebugLabel::new(s);
    assert_eq!(label.as_wgpu_label(), "Owned Label");
}

#[test]
fn test_debug_label_new_cow_borrowed() {
    let cow: Cow<'static, str> = Cow::Borrowed("Borrowed Cow");
    let label = DebugLabel::new(cow);
    assert_eq!(label.as_wgpu_label(), "Borrowed Cow");
}

#[test]
fn test_debug_label_new_cow_owned() {
    let cow: Cow<'static, str> = Cow::Owned(String::from("Owned Cow"));
    let label = DebugLabel::new(cow);
    assert_eq!(label.as_wgpu_label(), "Owned Cow");
}

#[test]
fn test_debug_label_new_static_zero_alloc() {
    let label = DebugLabel::new_static("Static Zero Alloc");
    assert_eq!(label.as_wgpu_label(), "Static Zero Alloc");
    // Verify it's actually borrowed (no heap allocation)
    match &label.name {
        Cow::Borrowed(_) => {}
        Cow::Owned(_) => panic!("Expected borrowed string for new_static"),
    }
}

#[test]
fn test_debug_label_new_static_const_context() {
    // Verify new_static can be used in const context
    const LABEL: DebugLabel = DebugLabel::new_static("Const Label");
    assert_eq!(LABEL.as_wgpu_label(), "Const Label");
}

#[test]
fn test_debug_label_with_color_basic() {
    let label = DebugLabel::with_color("Colored", [1.0, 0.5, 0.25, 0.75]);
    assert_eq!(label.as_wgpu_label(), "Colored");
    assert_eq!(label.color, Some([1.0, 0.5, 0.25, 0.75]));
}

#[test]
fn test_debug_label_with_color_red() {
    let label = DebugLabel::with_color("Red", [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(label.color.unwrap()[0], 1.0);
    assert_eq!(label.color.unwrap()[1], 0.0);
    assert_eq!(label.color.unwrap()[2], 0.0);
    assert_eq!(label.color.unwrap()[3], 1.0);
}

#[test]
fn test_debug_label_with_color_transparent() {
    let label = DebugLabel::with_color("Transparent", [0.5, 0.5, 0.5, 0.0]);
    assert_eq!(label.color.unwrap()[3], 0.0);
}

#[test]
fn test_debug_label_with_static_color_const() {
    // Verify with_static_color can be used in const context
    const COLORED: DebugLabel = DebugLabel::with_static_color("Static Colored", [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(COLORED.as_wgpu_label(), "Static Colored");
    assert!(COLORED.color.is_some());
}

#[test]
fn test_debug_label_with_static_color_borrowed() {
    let label = DebugLabel::with_static_color("Static Color", [1.0, 1.0, 1.0, 1.0]);
    match &label.name {
        Cow::Borrowed(_) => {}
        Cow::Owned(_) => panic!("Expected borrowed string"),
    }
}

#[test]
fn test_debug_label_from_static_str_trait() {
    let label: DebugLabel = "From Static Str".into();
    assert_eq!(label.as_wgpu_label(), "From Static Str");
    assert!(label.color.is_none());
}

#[test]
fn test_debug_label_from_string_trait() {
    let s = String::from("From String Trait");
    let label: DebugLabel = s.into();
    assert_eq!(label.as_wgpu_label(), "From String Trait");
}

#[test]
fn test_debug_label_default_is_empty() {
    let label = DebugLabel::default();
    assert_eq!(label.as_wgpu_label(), "");
    assert!(label.color.is_none());
}

#[test]
fn test_debug_label_multiple_constructors_consistency() {
    let l1 = DebugLabel::new("Test");
    let l2 = DebugLabel::new_static("Test");
    let l3: DebugLabel = "Test".into();

    assert_eq!(l1.as_wgpu_label(), l2.as_wgpu_label());
    assert_eq!(l2.as_wgpu_label(), l3.as_wgpu_label());
}

#[test]
fn test_debug_label_with_color_owned_string() {
    let s = String::from("Owned With Color");
    let label = DebugLabel::with_color(s, [0.1, 0.2, 0.3, 0.4]);
    assert_eq!(label.as_wgpu_label(), "Owned With Color");
    assert_eq!(label.color.unwrap()[0], 0.1);
}

#[test]
fn test_debug_label_color_boundary_values_zero() {
    let label = DebugLabel::with_color("Zero", [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(label.color.unwrap(), [0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_debug_label_color_boundary_values_one() {
    let label = DebugLabel::with_color("One", [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(label.color.unwrap(), [1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_debug_label_color_negative_values() {
    // Out of range values should still be stored (validation is caller's responsibility)
    let label = DebugLabel::with_color("Negative", [-0.5, -1.0, -2.0, -0.1]);
    assert_eq!(label.color.unwrap()[0], -0.5);
}

#[test]
fn test_debug_label_color_values_above_one() {
    let label = DebugLabel::with_color("Above", [1.5, 2.0, 3.0, 10.0]);
    assert_eq!(label.color.unwrap()[0], 1.5);
}

// ============================================================================
// CATEGORY 2: DebugLabel Color Helpers Tests (15+ tests)
// ============================================================================

#[test]
fn test_debug_label_has_color_true() {
    let label = DebugLabel::with_color("Has Color", [0.5, 0.5, 0.5, 1.0]);
    assert!(label.has_color());
}

#[test]
fn test_debug_label_has_color_false() {
    let label = DebugLabel::new("No Color");
    assert!(!label.has_color());
}

#[test]
fn test_debug_label_rgb_extraction() {
    let label = DebugLabel::with_color("RGB", [0.1, 0.2, 0.3, 0.9]);
    let rgb = label.rgb().unwrap();
    assert_eq!(rgb, [0.1, 0.2, 0.3]);
}

#[test]
fn test_debug_label_rgb_none_when_no_color() {
    let label = DebugLabel::new("No Color");
    assert!(label.rgb().is_none());
}

#[test]
fn test_debug_label_rgb_ignores_alpha() {
    let l1 = DebugLabel::with_color("A1", [0.5, 0.5, 0.5, 0.0]);
    let l2 = DebugLabel::with_color("A2", [0.5, 0.5, 0.5, 1.0]);
    assert_eq!(l1.rgb(), l2.rgb());
}

#[test]
fn test_debug_label_rgba_u8_conversion() {
    let label = DebugLabel::with_color("U8", [1.0, 0.5, 0.0, 0.5]);
    let rgba = label.rgba_u8().unwrap();
    assert_eq!(rgba[0], 255);
    assert_eq!(rgba[1], 127);
    assert_eq!(rgba[2], 0);
    assert_eq!(rgba[3], 127);
}

#[test]
fn test_debug_label_rgba_u8_zero() {
    let label = DebugLabel::with_color("Zero", [0.0, 0.0, 0.0, 0.0]);
    let rgba = label.rgba_u8().unwrap();
    assert_eq!(rgba, [0, 0, 0, 0]);
}

#[test]
fn test_debug_label_rgba_u8_max() {
    let label = DebugLabel::with_color("Max", [1.0, 1.0, 1.0, 1.0]);
    let rgba = label.rgba_u8().unwrap();
    assert_eq!(rgba, [255, 255, 255, 255]);
}

#[test]
fn test_debug_label_rgba_u8_none_when_no_color() {
    let label = DebugLabel::new("No Color");
    assert!(label.rgba_u8().is_none());
}

#[test]
fn test_debug_label_rgba_u8_clamping_negative() {
    let label = DebugLabel::with_color("Clamp", [-0.5, -1.0, 0.0, 0.5]);
    let rgba = label.rgba_u8().unwrap();
    assert_eq!(rgba[0], 0);
    assert_eq!(rgba[1], 0);
}

#[test]
fn test_debug_label_rgba_u8_clamping_above_one() {
    let label = DebugLabel::with_color("Clamp", [1.5, 2.0, 0.5, 0.5]);
    let rgba = label.rgba_u8().unwrap();
    assert_eq!(rgba[0], 255);
    assert_eq!(rgba[1], 255);
}

#[test]
fn test_debug_label_rgba_u8_precise_quarter() {
    let label = DebugLabel::with_color("Quarter", [0.25, 0.25, 0.25, 0.25]);
    let rgba = label.rgba_u8().unwrap();
    // 0.25 * 255 = 63.75 -> 63
    assert_eq!(rgba[0], 63);
}

#[test]
fn test_debug_label_rgba_u8_precise_half() {
    let label = DebugLabel::with_color("Half", [0.5, 0.5, 0.5, 0.5]);
    let rgba = label.rgba_u8().unwrap();
    // 0.5 * 255 = 127.5 -> 127
    assert_eq!(rgba[0], 127);
}

#[test]
fn test_debug_label_rgba_u8_precise_three_quarters() {
    let label = DebugLabel::with_color("3/4", [0.75, 0.75, 0.75, 0.75]);
    let rgba = label.rgba_u8().unwrap();
    // 0.75 * 255 = 191.25 -> 191
    assert_eq!(rgba[0], 191);
}

#[test]
fn test_debug_label_rgba_u8_each_channel_independent() {
    let label = DebugLabel::with_color("Channels", [0.0, 0.5, 1.0, 0.25]);
    let rgba = label.rgba_u8().unwrap();
    assert_eq!(rgba[0], 0);
    assert_eq!(rgba[1], 127);
    assert_eq!(rgba[2], 255);
    assert_eq!(rgba[3], 63);
}

// ============================================================================
// CATEGORY 3: DebugLabel Child/Nesting Tests (12+ tests)
// ============================================================================

#[test]
fn test_debug_label_child_basic() {
    let parent = DebugLabel::new("Parent");
    let child = parent.child("Child");
    assert_eq!(child.as_wgpu_label(), "Parent/Child");
}

#[test]
fn test_debug_label_child_preserves_color() {
    let parent = DebugLabel::with_color("Parent", [1.0, 0.0, 0.0, 1.0]);
    let child = parent.child("Child");
    assert_eq!(child.color, Some([1.0, 0.0, 0.0, 1.0]));
}

#[test]
fn test_debug_label_child_no_color_preserves_none() {
    let parent = DebugLabel::new("Parent");
    let child = parent.child("Child");
    assert!(child.color.is_none());
}

#[test]
fn test_debug_label_child_chain() {
    let l1 = DebugLabel::new("A");
    let l2 = l1.child("B");
    let l3 = l2.child("C");
    assert_eq!(l3.as_wgpu_label(), "A/B/C");
}

#[test]
fn test_debug_label_child_deep_nesting() {
    let mut label = DebugLabel::new("Root");
    for i in 1..=10 {
        label = label.child(&format!("Level{}", i));
    }
    assert!(label.as_wgpu_label().starts_with("Root/Level1/Level2"));
    assert!(label.as_wgpu_label().contains("Level10"));
}

#[test]
fn test_debug_label_child_empty_suffix() {
    let parent = DebugLabel::new("Parent");
    let child = parent.child("");
    assert_eq!(child.as_wgpu_label(), "Parent/");
}

#[test]
fn test_debug_label_child_empty_parent() {
    let parent = DebugLabel::new("");
    let child = parent.child("Child");
    assert_eq!(child.as_wgpu_label(), "/Child");
}

#[test]
fn test_debug_label_child_both_empty() {
    let parent = DebugLabel::new("");
    let child = parent.child("");
    assert_eq!(child.as_wgpu_label(), "/");
}

#[test]
fn test_debug_label_child_with_slashes_in_suffix() {
    let parent = DebugLabel::new("Parent");
    let child = parent.child("A/B/C");
    assert_eq!(child.as_wgpu_label(), "Parent/A/B/C");
}

#[test]
fn test_debug_label_child_unicode() {
    let parent = DebugLabel::new("Pass");
    let child = parent.child("Cascade");
    assert_eq!(child.as_wgpu_label(), "Pass/Cascade");
}

#[test]
fn test_debug_label_child_special_chars() {
    let parent = DebugLabel::new("Pass");
    let child = parent.child("[0]");
    assert_eq!(child.as_wgpu_label(), "Pass/[0]");
}

#[test]
fn test_debug_label_child_does_not_modify_parent() {
    let parent = DebugLabel::new("Parent");
    let _child = parent.child("Child");
    assert_eq!(parent.as_wgpu_label(), "Parent");
}

// ============================================================================
// CATEGORY 4: DebugLabel Trait Implementations (15+ tests)
// ============================================================================

#[test]
fn test_debug_label_clone_simple() {
    let original = DebugLabel::new("Clone Me");
    let cloned = original.clone();
    assert_eq!(original.as_wgpu_label(), cloned.as_wgpu_label());
}

#[test]
fn test_debug_label_clone_with_color() {
    let original = DebugLabel::with_color("Clone", [0.5, 0.5, 0.5, 0.5]);
    let cloned = original.clone();
    assert_eq!(original.color, cloned.color);
}

#[test]
fn test_debug_label_clone_independence() {
    let original = DebugLabel::new("Original");
    let mut cloned = original.clone();
    cloned = cloned.child("Modified");
    assert_eq!(original.as_wgpu_label(), "Original");
    assert_eq!(cloned.as_wgpu_label(), "Original/Modified");
}

#[test]
fn test_debug_label_debug_format() {
    let label = DebugLabel::new("Debug Format");
    let debug_str = format!("{:?}", label);
    assert!(debug_str.contains("DebugLabel"));
    assert!(debug_str.contains("Debug Format"));
}

#[test]
fn test_debug_label_debug_format_with_color() {
    let label = DebugLabel::with_color("Debug", [1.0, 0.0, 0.0, 1.0]);
    let debug_str = format!("{:?}", label);
    assert!(debug_str.contains("Some"));
}

#[test]
fn test_debug_label_display() {
    let label = DebugLabel::new("Display Test");
    assert_eq!(format!("{}", label), "Display Test");
}

#[test]
fn test_debug_label_display_ignores_color() {
    let label = DebugLabel::with_color("Display", [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(format!("{}", label), "Display");
}

#[test]
fn test_debug_label_partial_eq_same() {
    let l1 = DebugLabel::new("Same");
    let l2 = DebugLabel::new("Same");
    assert_eq!(l1, l2);
}

#[test]
fn test_debug_label_partial_eq_different_names() {
    let l1 = DebugLabel::new("Name1");
    let l2 = DebugLabel::new("Name2");
    assert_ne!(l1, l2);
}

#[test]
fn test_debug_label_partial_eq_different_colors() {
    let l1 = DebugLabel::with_color("Same", [1.0, 0.0, 0.0, 1.0]);
    let l2 = DebugLabel::with_color("Same", [0.0, 1.0, 0.0, 1.0]);
    assert_ne!(l1, l2);
}

#[test]
fn test_debug_label_partial_eq_color_vs_no_color() {
    let l1 = DebugLabel::new("Same");
    let l2 = DebugLabel::with_color("Same", [1.0, 0.0, 0.0, 1.0]);
    assert_ne!(l1, l2);
}

#[test]
fn test_debug_label_partial_eq_same_with_color() {
    let l1 = DebugLabel::with_color("Same", [0.5, 0.5, 0.5, 1.0]);
    let l2 = DebugLabel::with_color("Same", [0.5, 0.5, 0.5, 1.0]);
    assert_eq!(l1, l2);
}

#[test]
fn test_debug_label_from_static_str_is_borrowed() {
    let label: DebugLabel = "Static".into();
    match &label.name {
        Cow::Borrowed(_) => {}
        Cow::Owned(_) => panic!("Expected borrowed for &'static str"),
    }
}

#[test]
fn test_debug_label_from_string_is_owned() {
    let s = String::from("Owned");
    let label: DebugLabel = s.into();
    match &label.name {
        Cow::Owned(_) => {}
        Cow::Borrowed(_) => panic!("Expected owned for String"),
    }
}

#[test]
fn test_debug_label_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DebugLabel>();
}

// ============================================================================
// CATEGORY 5: DebugGroup Tests (20+ tests)
// ============================================================================

#[test]
fn test_debug_group_new_basic() {
    let group = DebugGroup::new(DebugLabel::new("Basic Group"));
    assert_eq!(group.label.as_wgpu_label(), "Basic Group");
    assert_eq!(group.depth, 0);
    assert!(group.start_time.is_none());
}

#[test]
fn test_debug_group_new_default_depth() {
    let group = DebugGroup::new(DebugLabel::new("Test"));
    assert_eq!(group.depth, 0);
}

#[test]
fn test_debug_group_new_no_profiling() {
    let group = DebugGroup::new(DebugLabel::new("Test"));
    assert!(!group.has_profiling());
    assert!(group.start_time.is_none());
}

#[test]
fn test_debug_group_with_depth_zero() {
    let group = DebugGroup::with_depth(DebugLabel::new("Zero"), 0);
    assert_eq!(group.depth, 0);
}

#[test]
fn test_debug_group_with_depth_nonzero() {
    let group = DebugGroup::with_depth(DebugLabel::new("Deep"), 5);
    assert_eq!(group.depth, 5);
}

#[test]
fn test_debug_group_with_depth_max() {
    let group = DebugGroup::with_depth(DebugLabel::new("Max"), u32::MAX);
    assert_eq!(group.depth, u32::MAX);
}

#[test]
fn test_debug_group_with_depth_no_profiling() {
    let group = DebugGroup::with_depth(DebugLabel::new("Test"), 3);
    assert!(!group.has_profiling());
}

#[test]
fn test_debug_group_with_profiling_enabled() {
    let group = DebugGroup::with_profiling(DebugLabel::new("Profiled"));
    assert!(group.has_profiling());
    assert!(group.start_time.is_some());
}

#[test]
fn test_debug_group_with_profiling_default_depth() {
    let group = DebugGroup::with_profiling(DebugLabel::new("Test"));
    assert_eq!(group.depth, 0);
}

#[test]
fn test_debug_group_with_depth_and_profiling() {
    let group = DebugGroup::with_depth_and_profiling(DebugLabel::new("Both"), 7);
    assert_eq!(group.depth, 7);
    assert!(group.has_profiling());
}

#[test]
fn test_debug_group_elapsed_some() {
    let group = DebugGroup::with_profiling(DebugLabel::new("Timed"));
    thread::sleep(Duration::from_millis(1));
    let elapsed = group.elapsed();
    assert!(elapsed.is_some());
    assert!(elapsed.unwrap().as_micros() > 0);
}

#[test]
fn test_debug_group_elapsed_none_without_profiling() {
    let group = DebugGroup::new(DebugLabel::new("Not Timed"));
    assert!(group.elapsed().is_none());
}

#[test]
fn test_debug_group_elapsed_ms_some() {
    let group = DebugGroup::with_profiling(DebugLabel::new("Timed"));
    thread::sleep(Duration::from_millis(5));
    let ms = group.elapsed_ms();
    assert!(ms.is_some());
    assert!(ms.unwrap() >= 0.0);
}

#[test]
fn test_debug_group_elapsed_ms_none_without_profiling() {
    let group = DebugGroup::new(DebugLabel::new("Not Timed"));
    assert!(group.elapsed_ms().is_none());
}

#[test]
fn test_debug_group_elapsed_increases_over_time() {
    let group = DebugGroup::with_profiling(DebugLabel::new("Increasing"));
    let e1 = group.elapsed().unwrap();
    thread::sleep(Duration::from_millis(2));
    let e2 = group.elapsed().unwrap();
    assert!(e2 > e1);
}

#[test]
fn test_debug_group_has_profiling_true() {
    let group = DebugGroup::with_profiling(DebugLabel::new("Profiled"));
    assert!(group.has_profiling());
}

#[test]
fn test_debug_group_has_profiling_false() {
    let group = DebugGroup::new(DebugLabel::new("Not Profiled"));
    assert!(!group.has_profiling());
}

#[test]
fn test_debug_group_clone() {
    let group = DebugGroup::with_depth_and_profiling(DebugLabel::new("Clone"), 3);
    let cloned = group.clone();
    assert_eq!(cloned.label.as_wgpu_label(), "Clone");
    assert_eq!(cloned.depth, 3);
    assert!(cloned.has_profiling());
}

#[test]
fn test_debug_group_debug_format() {
    let group = DebugGroup::new(DebugLabel::new("Debug"));
    let debug_str = format!("{:?}", group);
    assert!(debug_str.contains("DebugGroup"));
}

#[test]
fn test_debug_group_label_with_color() {
    let label = DebugLabel::with_color("Colored", [1.0, 0.0, 0.0, 1.0]);
    let group = DebugGroup::new(label);
    assert!(group.label.has_color());
}

#[test]
fn test_debug_group_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DebugGroup>();
}

// ============================================================================
// CATEGORY 6: DebugMarkerStack Basic Tests (20+ tests)
// ============================================================================

#[test]
fn test_marker_stack_new_empty() {
    let stack = DebugMarkerStack::new();
    assert!(stack.is_empty());
}

#[test]
fn test_marker_stack_new_depth_zero() {
    let stack = DebugMarkerStack::new();
    assert_eq!(stack.current_depth(), 0);
}

#[test]
fn test_marker_stack_new_default_max_depth() {
    let stack = DebugMarkerStack::new();
    assert_eq!(stack.max_depth(), DebugMarkerStack::DEFAULT_MAX_DEPTH);
    assert_eq!(stack.max_depth(), 16);
}

#[test]
fn test_marker_stack_with_max_depth() {
    let stack = DebugMarkerStack::with_max_depth(4);
    assert_eq!(stack.max_depth(), 4);
}

#[test]
fn test_marker_stack_with_max_depth_zero() {
    let stack = DebugMarkerStack::with_max_depth(0);
    assert_eq!(stack.max_depth(), 0);
    assert!(!stack.can_push());
}

#[test]
fn test_marker_stack_with_max_depth_one() {
    let mut stack = DebugMarkerStack::with_max_depth(1);
    assert!(stack.push_group(DebugLabel::new("Only")));
    assert!(!stack.push_group(DebugLabel::new("Too Many")));
}

#[test]
fn test_marker_stack_with_max_depth_large() {
    let stack = DebugMarkerStack::with_max_depth(1000);
    assert_eq!(stack.max_depth(), 1000);
}

#[test]
fn test_marker_stack_with_profiling() {
    let stack = DebugMarkerStack::with_profiling();
    assert!(stack.profiling_enabled());
}

#[test]
fn test_marker_stack_profiling_default_false() {
    let stack = DebugMarkerStack::new();
    assert!(!stack.profiling_enabled());
}

#[test]
fn test_marker_stack_set_profiling_enable() {
    let mut stack = DebugMarkerStack::new();
    stack.set_profiling(true);
    assert!(stack.profiling_enabled());
}

#[test]
fn test_marker_stack_set_profiling_disable() {
    let mut stack = DebugMarkerStack::with_profiling();
    stack.set_profiling(false);
    assert!(!stack.profiling_enabled());
}

#[test]
fn test_marker_stack_default_impl() {
    let stack: DebugMarkerStack = Default::default();
    assert!(stack.is_empty());
    assert_eq!(stack.max_depth(), DebugMarkerStack::DEFAULT_MAX_DEPTH);
}

#[test]
fn test_marker_stack_can_push_empty() {
    let stack = DebugMarkerStack::new();
    assert!(stack.can_push());
}

#[test]
fn test_marker_stack_can_push_at_max() {
    let mut stack = DebugMarkerStack::with_max_depth(2);
    stack.push_group(DebugLabel::new("One"));
    stack.push_group(DebugLabel::new("Two"));
    assert!(!stack.can_push());
}

#[test]
fn test_marker_stack_can_push_below_max() {
    let mut stack = DebugMarkerStack::with_max_depth(3);
    stack.push_group(DebugLabel::new("One"));
    stack.push_group(DebugLabel::new("Two"));
    assert!(stack.can_push());
}

#[test]
fn test_marker_stack_clone() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("Group"));
    let cloned = stack.clone();
    assert_eq!(cloned.current_depth(), 1);
}

#[test]
fn test_marker_stack_debug_format() {
    let stack = DebugMarkerStack::new();
    let debug_str = format!("{:?}", stack);
    assert!(debug_str.contains("DebugMarkerStack"));
}

#[test]
fn test_marker_stack_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DebugMarkerStack>();
}

#[test]
fn test_marker_stack_initial_markers_at_depth() {
    let stack = DebugMarkerStack::new();
    assert_eq!(stack.markers_at_depth(), 0);
}

#[test]
fn test_marker_stack_groups_empty() {
    let stack = DebugMarkerStack::new();
    assert!(stack.groups().is_empty());
}

// ============================================================================
// CATEGORY 7: DebugMarkerStack Push/Pop Tests (20+ tests)
// ============================================================================

#[test]
fn test_marker_stack_push_single() {
    let mut stack = DebugMarkerStack::new();
    assert!(stack.push_group(DebugLabel::new("First")));
    assert_eq!(stack.current_depth(), 1);
}

#[test]
fn test_marker_stack_push_multiple() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("One"));
    stack.push_group(DebugLabel::new("Two"));
    stack.push_group(DebugLabel::new("Three"));
    assert_eq!(stack.current_depth(), 3);
}

#[test]
fn test_marker_stack_push_at_max_depth_fails() {
    let mut stack = DebugMarkerStack::with_max_depth(2);
    assert!(stack.push_group(DebugLabel::new("One")));
    assert!(stack.push_group(DebugLabel::new("Two")));
    assert!(!stack.push_group(DebugLabel::new("Three")));
    assert_eq!(stack.current_depth(), 2);
}

#[test]
fn test_marker_stack_push_returns_true_success() {
    let mut stack = DebugMarkerStack::new();
    assert!(stack.push_group(DebugLabel::new("Success")));
}

#[test]
fn test_marker_stack_push_returns_false_failure() {
    let mut stack = DebugMarkerStack::with_max_depth(0);
    assert!(!stack.push_group(DebugLabel::new("Fail")));
}

#[test]
fn test_marker_stack_pop_single() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("Pop Me"));
    let group = stack.pop_group().unwrap();
    assert_eq!(group.label.as_wgpu_label(), "Pop Me");
}

#[test]
fn test_marker_stack_pop_returns_none_when_empty() {
    let mut stack = DebugMarkerStack::new();
    assert!(stack.pop_group().is_none());
}

#[test]
fn test_marker_stack_pop_lifo_order() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("First"));
    stack.push_group(DebugLabel::new("Second"));
    stack.push_group(DebugLabel::new("Third"));

    assert_eq!(stack.pop_group().unwrap().label.as_wgpu_label(), "Third");
    assert_eq!(stack.pop_group().unwrap().label.as_wgpu_label(), "Second");
    assert_eq!(stack.pop_group().unwrap().label.as_wgpu_label(), "First");
}

#[test]
fn test_marker_stack_pop_depth_tracking() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("A"));
    stack.push_group(DebugLabel::new("B"));

    let b = stack.pop_group().unwrap();
    assert_eq!(b.depth, 1);

    let a = stack.pop_group().unwrap();
    assert_eq!(a.depth, 0);
}

#[test]
fn test_marker_stack_push_pop_push() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("First"));
    stack.pop_group();
    stack.push_group(DebugLabel::new("Second"));
    assert_eq!(stack.current_depth(), 1);
    assert_eq!(
        stack.current_group().unwrap().label.as_wgpu_label(),
        "Second"
    );
}

#[test]
fn test_marker_stack_push_with_profiling() {
    let mut stack = DebugMarkerStack::with_profiling();
    stack.push_group(DebugLabel::new("Profiled"));
    let group = stack.current_group().unwrap();
    assert!(group.has_profiling());
}

#[test]
fn test_marker_stack_push_without_profiling() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("Not Profiled"));
    let group = stack.current_group().unwrap();
    assert!(!group.has_profiling());
}

#[test]
fn test_marker_stack_is_empty_after_pop() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("Only"));
    stack.pop_group();
    assert!(stack.is_empty());
}

#[test]
fn test_marker_stack_not_empty_with_groups() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("Present"));
    assert!(!stack.is_empty());
}

#[test]
fn test_marker_stack_depth_matches_count() {
    let mut stack = DebugMarkerStack::new();
    for i in 0..10 {
        stack.push_group(DebugLabel::new(format!("Group{}", i)));
        assert_eq!(stack.current_depth(), i + 1);
    }
}

#[test]
fn test_marker_stack_depth_decreases_on_pop() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("A"));
    stack.push_group(DebugLabel::new("B"));
    stack.push_group(DebugLabel::new("C"));

    stack.pop_group();
    assert_eq!(stack.current_depth(), 2);
    stack.pop_group();
    assert_eq!(stack.current_depth(), 1);
    stack.pop_group();
    assert_eq!(stack.current_depth(), 0);
}

#[test]
fn test_marker_stack_multiple_pop_on_empty() {
    let mut stack = DebugMarkerStack::new();
    assert!(stack.pop_group().is_none());
    assert!(stack.pop_group().is_none());
    assert!(stack.pop_group().is_none());
    assert_eq!(stack.current_depth(), 0);
}

#[test]
fn test_marker_stack_push_resets_markers_at_depth() {
    let mut stack = DebugMarkerStack::new();
    stack.insert_marker(&DebugLabel::new("Marker"));
    assert_eq!(stack.markers_at_depth(), 1);
    stack.push_group(DebugLabel::new("Group"));
    assert_eq!(stack.markers_at_depth(), 0);
}

#[test]
fn test_marker_stack_pop_resets_markers_at_depth() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("Group"));
    stack.insert_marker(&DebugLabel::new("Marker"));
    assert_eq!(stack.markers_at_depth(), 1);
    stack.pop_group();
    assert_eq!(stack.markers_at_depth(), 0);
}

#[test]
fn test_marker_stack_push_fill_to_max() {
    let mut stack = DebugMarkerStack::with_max_depth(16);
    for i in 0..16 {
        assert!(stack.push_group(DebugLabel::new(format!("G{}", i))));
    }
    assert_eq!(stack.current_depth(), 16);
    assert!(!stack.can_push());
}

// ============================================================================
// CATEGORY 8: DebugMarkerStack Accessors Tests (15+ tests)
// ============================================================================

#[test]
fn test_marker_stack_current_group_none_when_empty() {
    let stack = DebugMarkerStack::new();
    assert!(stack.current_group().is_none());
}

#[test]
fn test_marker_stack_current_group_some_after_push() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("Current"));
    assert!(stack.current_group().is_some());
}

#[test]
fn test_marker_stack_current_group_returns_last() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("First"));
    stack.push_group(DebugLabel::new("Second"));
    assert_eq!(
        stack.current_group().unwrap().label.as_wgpu_label(),
        "Second"
    );
}

#[test]
fn test_marker_stack_groups_slice_length() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("A"));
    stack.push_group(DebugLabel::new("B"));
    stack.push_group(DebugLabel::new("C"));
    assert_eq!(stack.groups().len(), 3);
}

#[test]
fn test_marker_stack_groups_slice_order() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("First"));
    stack.push_group(DebugLabel::new("Second"));
    let groups = stack.groups();
    assert_eq!(groups[0].label.as_wgpu_label(), "First");
    assert_eq!(groups[1].label.as_wgpu_label(), "Second");
}

#[test]
fn test_marker_stack_path_empty() {
    let stack = DebugMarkerStack::new();
    assert_eq!(stack.path(), "");
}

#[test]
fn test_marker_stack_path_single() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("Only"));
    assert_eq!(stack.path(), "Only");
}

#[test]
fn test_marker_stack_path_multiple() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("A"));
    stack.push_group(DebugLabel::new("B"));
    stack.push_group(DebugLabel::new("C"));
    assert_eq!(stack.path(), "A/B/C");
}

#[test]
fn test_marker_stack_path_with_empty_labels() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new(""));
    stack.push_group(DebugLabel::new("B"));
    assert_eq!(stack.path(), "/B");
}

#[test]
fn test_marker_stack_max_depth_accessor() {
    let stack = DebugMarkerStack::with_max_depth(42);
    assert_eq!(stack.max_depth(), 42);
}

#[test]
fn test_marker_stack_insert_marker_increments_count() {
    let mut stack = DebugMarkerStack::new();
    assert_eq!(stack.markers_at_depth(), 0);
    stack.insert_marker(&DebugLabel::new("M1"));
    assert_eq!(stack.markers_at_depth(), 1);
    stack.insert_marker(&DebugLabel::new("M2"));
    assert_eq!(stack.markers_at_depth(), 2);
}

#[test]
fn test_marker_stack_insert_marker_at_different_depths() {
    let mut stack = DebugMarkerStack::new();
    stack.insert_marker(&DebugLabel::new("Root Marker"));
    assert_eq!(stack.markers_at_depth(), 1);

    stack.push_group(DebugLabel::new("Level1"));
    assert_eq!(stack.markers_at_depth(), 0);
    stack.insert_marker(&DebugLabel::new("L1 Marker"));
    assert_eq!(stack.markers_at_depth(), 1);
}

#[test]
fn test_marker_stack_clear_empty_stack() {
    let mut stack = DebugMarkerStack::new();
    stack.clear();
    assert!(stack.is_empty());
}

#[test]
fn test_marker_stack_clear_with_groups() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("A"));
    stack.push_group(DebugLabel::new("B"));
    stack.clear();
    assert!(stack.is_empty());
    assert_eq!(stack.current_depth(), 0);
}

#[test]
fn test_marker_stack_clear_resets_markers() {
    let mut stack = DebugMarkerStack::new();
    stack.insert_marker(&DebugLabel::new("Marker"));
    stack.clear();
    assert_eq!(stack.markers_at_depth(), 0);
}

// ============================================================================
// CATEGORY 9: Colors Module Tests (15+ tests)
// ============================================================================

#[test]
fn test_colors_geometry_valid() {
    let c = colors::GEOMETRY;
    assert!(c.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_colors_shadow_valid() {
    let c = colors::SHADOW;
    assert!(c.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_colors_lighting_valid() {
    let c = colors::LIGHTING;
    assert!(c.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_colors_post_process_valid() {
    let c = colors::POST_PROCESS;
    assert!(c.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_colors_compute_valid() {
    let c = colors::COMPUTE;
    assert!(c.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_colors_ui_valid() {
    let c = colors::UI;
    assert!(c.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_colors_debug_valid() {
    let c = colors::DEBUG;
    assert!(c.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_colors_transparent_valid() {
    let c = colors::TRANSPARENT;
    assert!(c.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_colors_raytracing_valid() {
    let c = colors::RAYTRACING;
    assert!(c.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_colors_transfer_valid() {
    let c = colors::TRANSFER;
    assert!(c.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_colors_all_have_alpha_one() {
    let all = [
        colors::GEOMETRY,
        colors::SHADOW,
        colors::LIGHTING,
        colors::POST_PROCESS,
        colors::COMPUTE,
        colors::UI,
        colors::DEBUG,
        colors::TRANSPARENT,
        colors::RAYTRACING,
        colors::TRANSFER,
    ];
    for c in all {
        assert_eq!(c[3], 1.0, "Alpha should be 1.0 for visibility");
    }
}

#[test]
fn test_colors_are_distinct() {
    let all = [
        colors::GEOMETRY,
        colors::SHADOW,
        colors::LIGHTING,
        colors::POST_PROCESS,
        colors::COMPUTE,
        colors::UI,
        colors::DEBUG,
        colors::TRANSPARENT,
        colors::RAYTRACING,
        colors::TRANSFER,
    ];
    // Check at least the RGB portions are distinct
    let mut seen = HashSet::new();
    for c in all {
        let rgb = (
            (c[0] * 100.0) as i32,
            (c[1] * 100.0) as i32,
            (c[2] * 100.0) as i32,
        );
        seen.insert(rgb);
    }
    // All should be unique
    assert_eq!(seen.len(), 10);
}

#[test]
fn test_colors_geometry_is_orange() {
    let c = colors::GEOMETRY;
    assert!(c[0] > 0.8); // High red
    assert!(c[1] > 0.4 && c[1] < 0.8); // Medium green
    assert!(c[2] < 0.4); // Low blue
}

#[test]
fn test_colors_shadow_is_dark_blue() {
    let c = colors::SHADOW;
    assert!(c[0] < 0.4); // Low red
    assert!(c[1] < 0.5); // Low-medium green
    assert!(c[2] > 0.4); // Higher blue
}

#[test]
fn test_colors_debug_is_red() {
    let c = colors::DEBUG;
    assert!(c[0] > 0.8); // High red
    assert!(c[1] < 0.4); // Low green
    assert!(c[2] < 0.4); // Low blue
}

// ============================================================================
// CATEGORY 10: Edge Cases - Empty Strings (10+ tests)
// ============================================================================

#[test]
fn test_edge_empty_label_new() {
    let label = DebugLabel::new("");
    assert_eq!(label.as_wgpu_label(), "");
}

#[test]
fn test_edge_empty_label_static() {
    let label = DebugLabel::new_static("");
    assert_eq!(label.as_wgpu_label(), "");
}

#[test]
fn test_edge_empty_label_with_color() {
    let label = DebugLabel::with_color("", [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(label.as_wgpu_label(), "");
    assert!(label.has_color());
}

#[test]
fn test_edge_empty_label_display() {
    let label = DebugLabel::new("");
    assert_eq!(format!("{}", label), "");
}

#[test]
fn test_edge_empty_label_child() {
    let parent = DebugLabel::new("Parent");
    let child = parent.child("");
    assert_eq!(child.as_wgpu_label(), "Parent/");
}

#[test]
fn test_edge_empty_parent_child() {
    let parent = DebugLabel::new("");
    let child = parent.child("Child");
    assert_eq!(child.as_wgpu_label(), "/Child");
}

#[test]
fn test_edge_empty_group_label() {
    let group = DebugGroup::new(DebugLabel::new(""));
    assert_eq!(group.label.as_wgpu_label(), "");
}

#[test]
fn test_edge_empty_stack_push() {
    let mut stack = DebugMarkerStack::new();
    assert!(stack.push_group(DebugLabel::new("")));
    assert_eq!(stack.current_depth(), 1);
}

#[test]
fn test_edge_empty_stack_path_single_empty() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new(""));
    assert_eq!(stack.path(), "");
}

#[test]
fn test_edge_empty_stack_path_mixed() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("A"));
    stack.push_group(DebugLabel::new(""));
    stack.push_group(DebugLabel::new("C"));
    assert_eq!(stack.path(), "A//C");
}

// ============================================================================
// CATEGORY 11: Edge Cases - Unicode (10+ tests)
// ============================================================================

#[test]
fn test_edge_unicode_emoji() {
    let label = DebugLabel::new("Pass ");
    assert_eq!(label.as_wgpu_label(), "Pass ");
}

#[test]
fn test_edge_unicode_cjk() {
    let label = DebugLabel::new("Chinese Characters");
    assert!(label.as_wgpu_label().contains(""));
}

#[test]
fn test_edge_unicode_arabic() {
    let label = DebugLabel::new("");
    assert_eq!(label.as_wgpu_label(), "");
}

#[test]
fn test_edge_unicode_combining_chars() {
    let label = DebugLabel::new("e\u{0301}"); // e with acute accent
    assert_eq!(label.as_wgpu_label().chars().count(), 2);
}

#[test]
fn test_edge_unicode_zero_width() {
    let label = DebugLabel::new("A\u{200B}B"); // zero-width space
    assert!(label.as_wgpu_label().len() > 2);
}

#[test]
fn test_edge_unicode_rtl() {
    let label = DebugLabel::new("");
    assert_eq!(label.as_wgpu_label(), "");
}

#[test]
fn test_edge_unicode_mixed_scripts() {
    let label = DebugLabel::new("ABC123");
    assert!(label.as_wgpu_label().contains("ABC"));
    assert!(label.as_wgpu_label().contains("123"));
}

#[test]
fn test_edge_unicode_in_child() {
    let parent = DebugLabel::new("Parent");
    let child = parent.child("Child");
    assert!(child.as_wgpu_label().contains(""));
}

#[test]
fn test_edge_unicode_full_width() {
    // Full-width characters (U+FF21, U+FF22, U+FF23) are 3 bytes each in UTF-8
    let label = DebugLabel::new("\u{FF21}\u{FF22}\u{FF23}"); // Full-width ABC
    // 3 chars * 3 bytes = 9 bytes
    assert_eq!(label.as_wgpu_label().len(), 9);
    assert_eq!(label.as_wgpu_label().chars().count(), 3);
}

#[test]
fn test_edge_unicode_stack_path() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("A"));
    stack.push_group(DebugLabel::new("B"));
    assert_eq!(stack.path(), "A/B");
}

// ============================================================================
// CATEGORY 12: Edge Cases - Long Strings (8+ tests)
// ============================================================================

#[test]
fn test_edge_long_string_1k() {
    let long = "X".repeat(1000);
    let label = DebugLabel::new(long.clone());
    assert_eq!(label.as_wgpu_label().len(), 1000);
}

#[test]
fn test_edge_long_string_10k() {
    let long = "Y".repeat(10000);
    let label = DebugLabel::new(long.clone());
    assert_eq!(label.as_wgpu_label().len(), 10000);
}

#[test]
fn test_edge_long_string_with_color() {
    let long = "Z".repeat(5000);
    let label = DebugLabel::with_color(long.clone(), [0.5, 0.5, 0.5, 1.0]);
    assert_eq!(label.as_wgpu_label().len(), 5000);
    assert!(label.has_color());
}

#[test]
fn test_edge_long_string_child() {
    let long = "A".repeat(500);
    let parent = DebugLabel::new(long.clone());
    let child = parent.child(&"B".repeat(500));
    assert_eq!(child.as_wgpu_label().len(), 1001); // 500 + "/" + 500
}

#[test]
fn test_edge_long_string_in_group() {
    let long = "G".repeat(2000);
    let group = DebugGroup::new(DebugLabel::new(long.clone()));
    assert_eq!(group.label.as_wgpu_label().len(), 2000);
}

#[test]
fn test_edge_long_string_in_stack() {
    let mut stack = DebugMarkerStack::new();
    let long = "S".repeat(1000);
    stack.push_group(DebugLabel::new(long));
    assert_eq!(stack.current_group().unwrap().label.as_wgpu_label().len(), 1000);
}

#[test]
fn test_edge_long_path() {
    let mut stack = DebugMarkerStack::new();
    for i in 0..10 {
        stack.push_group(DebugLabel::new(format!("Level{:03}", i)));
    }
    let path = stack.path();
    assert!(path.len() > 70);
    assert!(path.starts_with("Level000"));
}

#[test]
fn test_edge_very_long_display() {
    let long = "D".repeat(10000);
    let label = DebugLabel::new(long.clone());
    let display = format!("{}", label);
    assert_eq!(display.len(), 10000);
}

// ============================================================================
// CATEGORY 13: Edge Cases - Special Characters (10+ tests)
// ============================================================================

#[test]
fn test_edge_special_newline() {
    let label = DebugLabel::new("Line1\nLine2");
    assert!(label.as_wgpu_label().contains('\n'));
}

#[test]
fn test_edge_special_tab() {
    let label = DebugLabel::new("Col1\tCol2");
    assert!(label.as_wgpu_label().contains('\t'));
}

#[test]
fn test_edge_special_carriage_return() {
    let label = DebugLabel::new("Line\rOver");
    assert!(label.as_wgpu_label().contains('\r'));
}

#[test]
fn test_edge_special_null() {
    let label = DebugLabel::new("Before\0After");
    assert!(label.as_wgpu_label().contains('\0'));
}

#[test]
fn test_edge_special_backslash() {
    let label = DebugLabel::new("Path\\To\\File");
    assert!(label.as_wgpu_label().contains('\\'));
}

#[test]
fn test_edge_special_quotes() {
    let label = DebugLabel::new("\"Quoted\"");
    assert!(label.as_wgpu_label().contains('"'));
}

#[test]
fn test_edge_special_brackets() {
    let label = DebugLabel::new("[Index]");
    assert!(label.as_wgpu_label().starts_with('['));
}

#[test]
fn test_edge_special_angle_brackets() {
    let label = DebugLabel::new("<Generic>");
    assert!(label.as_wgpu_label().contains('<'));
}

#[test]
fn test_edge_special_pipe() {
    let label = DebugLabel::new("A|B|C");
    assert!(label.as_wgpu_label().contains('|'));
}

#[test]
fn test_edge_special_asterisk() {
    let label = DebugLabel::new("*pointer");
    assert!(label.as_wgpu_label().starts_with('*'));
}

// ============================================================================
// CATEGORY 14: Deep Nesting Tests (10+ tests)
// ============================================================================

#[test]
fn test_deep_nesting_default_max() {
    let mut stack = DebugMarkerStack::new();
    for i in 0..16 {
        assert!(stack.push_group(DebugLabel::new(format!("Level{}", i))));
    }
    assert_eq!(stack.current_depth(), 16);
    assert!(!stack.push_group(DebugLabel::new("TooDeep")));
}

#[test]
fn test_deep_nesting_custom_max() {
    let mut stack = DebugMarkerStack::with_max_depth(100);
    for i in 0..100 {
        assert!(stack.push_group(DebugLabel::new(format!("L{}", i))));
    }
    assert_eq!(stack.current_depth(), 100);
    assert!(!stack.can_push());
}

#[test]
fn test_deep_nesting_path_accuracy() {
    let mut stack = DebugMarkerStack::with_max_depth(5);
    stack.push_group(DebugLabel::new("A"));
    stack.push_group(DebugLabel::new("B"));
    stack.push_group(DebugLabel::new("C"));
    stack.push_group(DebugLabel::new("D"));
    stack.push_group(DebugLabel::new("E"));
    assert_eq!(stack.path(), "A/B/C/D/E");
}

#[test]
fn test_deep_nesting_depth_values() {
    let mut stack = DebugMarkerStack::with_max_depth(5);
    for i in 0..5 {
        stack.push_group(DebugLabel::new(format!("Level{}", i)));
    }
    let groups = stack.groups();
    for i in 0..5 {
        assert_eq!(groups[i].depth, i as u32);
    }
}

#[test]
fn test_deep_nesting_pop_all() {
    let mut stack = DebugMarkerStack::new();
    for i in 0..10 {
        stack.push_group(DebugLabel::new(format!("G{}", i)));
    }
    for _ in 0..10 {
        assert!(stack.pop_group().is_some());
    }
    assert!(stack.is_empty());
}

#[test]
fn test_deep_nesting_child_labels() {
    let mut label = DebugLabel::new("Root");
    for i in 1..=20 {
        label = label.child(&format!("L{}", i));
    }
    assert!(label.as_wgpu_label().starts_with("Root/L1/L2"));
    assert!(label.as_wgpu_label().ends_with("L20"));
}

#[test]
fn test_deep_nesting_with_profiling() {
    let mut stack = DebugMarkerStack::with_profiling();
    for i in 0..8 {
        stack.push_group(DebugLabel::new(format!("Profiled{}", i)));
    }
    for group in stack.groups() {
        assert!(group.has_profiling());
    }
}

#[test]
fn test_deep_nesting_markers_at_each_level() {
    let mut stack = DebugMarkerStack::new();
    for i in 0..5 {
        stack.push_group(DebugLabel::new(format!("L{}", i)));
        stack.insert_marker(&DebugLabel::new(format!("M{}", i)));
        assert_eq!(stack.markers_at_depth(), 1);
    }
}

#[test]
fn test_deep_nesting_interleaved_push_pop() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("A"));
    stack.push_group(DebugLabel::new("B"));
    stack.pop_group();
    stack.push_group(DebugLabel::new("C"));
    stack.push_group(DebugLabel::new("D"));
    assert_eq!(stack.path(), "A/C/D");
}

#[test]
fn test_deep_nesting_stress() {
    let mut stack = DebugMarkerStack::with_max_depth(1000);
    for i in 0..1000 {
        assert!(stack.push_group(DebugLabel::new(format!("X{}", i))));
    }
    assert_eq!(stack.current_depth(), 1000);
}

// ============================================================================
// CATEGORY 15: Rapid Push/Pop Sequences (10+ tests)
// ============================================================================

#[test]
fn test_rapid_push_pop_simple() {
    let mut stack = DebugMarkerStack::new();
    for _ in 0..100 {
        stack.push_group(DebugLabel::new("Rapid"));
        stack.pop_group();
    }
    assert!(stack.is_empty());
}

#[test]
fn test_rapid_push_pop_alternating() {
    let mut stack = DebugMarkerStack::new();
    for i in 0..50 {
        stack.push_group(DebugLabel::new(format!("A{}", i)));
        stack.push_group(DebugLabel::new(format!("B{}", i)));
        stack.pop_group();
        stack.pop_group();
    }
    assert!(stack.is_empty());
}

#[test]
fn test_rapid_push_burst() {
    let mut stack = DebugMarkerStack::with_max_depth(100);
    for i in 0..100 {
        stack.push_group(DebugLabel::new(format!("Burst{}", i)));
    }
    assert_eq!(stack.current_depth(), 100);
}

#[test]
fn test_rapid_pop_burst() {
    let mut stack = DebugMarkerStack::with_max_depth(100);
    for i in 0..100 {
        stack.push_group(DebugLabel::new(format!("B{}", i)));
    }
    for _ in 0..100 {
        stack.pop_group();
    }
    assert!(stack.is_empty());
}

#[test]
fn test_rapid_markers_between_groups() {
    let mut stack = DebugMarkerStack::new();
    for _ in 0..20 {
        stack.push_group(DebugLabel::new("G"));
        stack.insert_marker(&DebugLabel::new("M1"));
        stack.insert_marker(&DebugLabel::new("M2"));
        stack.pop_group();
    }
    assert!(stack.is_empty());
}

#[test]
fn test_rapid_nested_burst() {
    let mut stack = DebugMarkerStack::with_max_depth(10);
    for outer in 0..10 {
        for inner in 0..10 {
            stack.push_group(DebugLabel::new(format!("O{}I{}", outer, inner)));
        }
        for _ in 0..10 {
            stack.pop_group();
        }
    }
    assert!(stack.is_empty());
}

#[test]
fn test_rapid_depth_oscillation() {
    let mut stack = DebugMarkerStack::new();
    for _ in 0..50 {
        stack.push_group(DebugLabel::new("Up"));
        stack.push_group(DebugLabel::new("Up"));
        stack.push_group(DebugLabel::new("Up"));
        stack.pop_group();
        stack.pop_group();
        stack.pop_group();
    }
    assert!(stack.is_empty());
}

#[test]
fn test_rapid_with_profiling() {
    let mut stack = DebugMarkerStack::with_profiling();
    for _ in 0..50 {
        stack.push_group(DebugLabel::new("Profiled"));
        stack.pop_group();
    }
    assert!(stack.is_empty());
}

#[test]
fn test_rapid_clear_cycles() {
    let mut stack = DebugMarkerStack::new();
    for _ in 0..10 {
        for j in 0..5 {
            stack.push_group(DebugLabel::new(format!("G{}", j)));
        }
        stack.clear();
    }
    assert!(stack.is_empty());
}

#[test]
fn test_rapid_mixed_operations() {
    let mut stack = DebugMarkerStack::new();
    for i in 0..100 {
        if i % 3 == 0 {
            stack.push_group(DebugLabel::new("Push"));
        } else if i % 3 == 1 && !stack.is_empty() {
            stack.pop_group();
        } else {
            stack.insert_marker(&DebugLabel::new("Mark"));
        }
    }
    // Final state depends on sequence, just verify no panic
    assert!(stack.current_depth() <= 100);
}

// ============================================================================
// CATEGORY 16: DebugContextOps Trait Tests (5+ tests)
// ============================================================================

// Note: We can only test DebugContextOps on DebugMarkerStack-based types
// RenderPassDebugContext, ComputePassDebugContext, CommandEncoderDebugContext
// require actual wgpu types, so we test the trait interface conceptually

#[test]
fn test_debug_context_ops_trait_exists() {
    // Verify the trait is properly defined and accessible
    fn assert_debug_context_ops<T: DebugContextOps>(_t: &T) {}

    // We can't instantiate the context types without wgpu,
    // but we can verify the trait bounds exist
    let _marker: fn(&dyn DebugContextOps) = |_| {};
}

#[test]
fn test_debug_context_ops_methods() {
    // Verify all expected methods exist on the trait
    fn check_methods<T: DebugContextOps>(t: &mut T) {
        let _: bool = t.push_debug_group(DebugLabel::new("Test"));
        let _: Option<DebugGroup> = t.pop_debug_group();
        t.insert_debug_marker(DebugLabel::new("Marker"));
        let _: usize = t.current_depth();
        let _: bool = t.has_active_groups();
    }
    // Compilation proves the methods exist
}

#[test]
fn test_debug_context_ops_object_safety() {
    // Verify the trait is object-safe
    fn take_dyn(_ctx: &mut dyn DebugContextOps) {}
    // Compilation success proves object safety
}

#[test]
fn test_debug_label_into_debug_label() {
    // Verify Into<DebugLabel> works for the trait
    fn accept_into<T: Into<DebugLabel>>(_t: T) {}
    accept_into("Static");
    accept_into(String::from("Owned"));
    accept_into(DebugLabel::new("Label"));
}

#[test]
fn test_debug_marker_stack_not_impl_debug_context_ops() {
    // DebugMarkerStack intentionally does NOT implement DebugContextOps
    // because it doesn't wrap a wgpu type. Verify this by type checking.
    // (This test is conceptual - we're documenting the design choice)
    let mut stack = DebugMarkerStack::new();
    // These are similar but NOT DebugContextOps methods:
    let _: bool = stack.push_group(DebugLabel::new("Test"));
    let _: Option<DebugGroup> = stack.pop_group();
    stack.insert_marker(&DebugLabel::new("Marker"));
    let _: usize = stack.current_depth();
    let _: bool = !stack.is_empty();
}

// ============================================================================
// CATEGORY 17: Thread Safety Tests (5+ tests)
// ============================================================================

#[test]
fn test_debug_label_send() {
    fn assert_send<T: Send>() {}
    assert_send::<DebugLabel>();
}

#[test]
fn test_debug_label_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<DebugLabel>();
}

#[test]
fn test_debug_group_send() {
    fn assert_send<T: Send>() {}
    assert_send::<DebugGroup>();
}

#[test]
fn test_debug_group_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<DebugGroup>();
}

#[test]
fn test_debug_marker_stack_send() {
    fn assert_send<T: Send>() {}
    assert_send::<DebugMarkerStack>();
}

#[test]
fn test_debug_marker_stack_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<DebugMarkerStack>();
}

#[test]
fn test_debug_label_cross_thread() {
    let label = DebugLabel::new("Cross Thread");
    let handle = thread::spawn(move || {
        assert_eq!(label.as_wgpu_label(), "Cross Thread");
    });
    handle.join().unwrap();
}

#[test]
fn test_debug_group_cross_thread() {
    let group = DebugGroup::new(DebugLabel::new("Cross Thread Group"));
    let handle = thread::spawn(move || {
        assert_eq!(group.label.as_wgpu_label(), "Cross Thread Group");
    });
    handle.join().unwrap();
}

#[test]
fn test_debug_marker_stack_cross_thread() {
    let mut stack = DebugMarkerStack::new();
    stack.push_group(DebugLabel::new("In Main"));

    let handle = thread::spawn(move || {
        assert_eq!(stack.current_depth(), 1);
        stack.push_group(DebugLabel::new("In Thread"));
        assert_eq!(stack.current_depth(), 2);
        stack
    });

    let stack = handle.join().unwrap();
    assert_eq!(stack.current_depth(), 2);
}

// ============================================================================
// CATEGORY 18: Memory and Performance Characteristics (5+ tests)
// ============================================================================

#[test]
fn test_debug_label_static_no_allocation() {
    // new_static should use borrowed string (no heap allocation)
    let label = DebugLabel::new_static("No Alloc");
    match &label.name {
        Cow::Borrowed(_) => {}
        Cow::Owned(_) => panic!("Static label should not allocate"),
    }
}

#[test]
fn test_debug_label_owned_allocates() {
    // new with String should use owned (heap allocation)
    let s = String::from("Allocated");
    let label = DebugLabel::new(s);
    match &label.name {
        Cow::Owned(_) => {}
        Cow::Borrowed(_) => panic!("String should be owned"),
    }
}

#[test]
fn test_debug_marker_stack_initial_empty() {
    // Stack should start empty
    let stack = DebugMarkerStack::new();
    assert!(stack.groups().is_empty());
}

#[test]
fn test_debug_marker_stack_large_max_depth() {
    let stack = DebugMarkerStack::with_max_depth(1000);
    // Verify max depth is set correctly
    assert_eq!(stack.max_depth(), 1000);
    assert!(stack.groups().is_empty());
}

#[test]
fn test_debug_group_profiling_instant() {
    // Verify profiling uses Instant (not SystemTime)
    let group = DebugGroup::with_profiling(DebugLabel::new("Instant"));
    // Instant::now() is monotonic, elapsed should always be positive
    let e1 = group.elapsed().unwrap();
    let e2 = group.elapsed().unwrap();
    assert!(e2 >= e1);
}

// ============================================================================
// CATEGORY 19: Const Context Tests (5+ tests)
// ============================================================================

#[test]
fn test_const_debug_label_static() {
    const LABEL: DebugLabel = DebugLabel::new_static("Const Static");
    assert_eq!(LABEL.as_wgpu_label(), "Const Static");
}

#[test]
fn test_const_debug_label_with_static_color() {
    const COLORED: DebugLabel = DebugLabel::with_static_color("Const Colored", [1.0, 0.5, 0.0, 1.0]);
    assert_eq!(COLORED.as_wgpu_label(), "Const Colored");
    assert!(COLORED.color.is_some());
}

#[test]
fn test_const_colors_geometry() {
    const C: [f32; 4] = colors::GEOMETRY;
    assert!(C[0] > 0.0);
}

#[test]
fn test_const_colors_all() {
    const ALL: [[f32; 4]; 10] = [
        colors::GEOMETRY,
        colors::SHADOW,
        colors::LIGHTING,
        colors::POST_PROCESS,
        colors::COMPUTE,
        colors::UI,
        colors::DEBUG,
        colors::TRANSPARENT,
        colors::RAYTRACING,
        colors::TRANSFER,
    ];
    for c in ALL {
        for v in c {
            assert!((0.0..=1.0).contains(&v));
        }
    }
}

#[test]
fn test_const_default_max_depth() {
    const MAX: usize = DebugMarkerStack::DEFAULT_MAX_DEPTH;
    assert_eq!(MAX, 16);
}

// ============================================================================
// CATEGORY 20: Integration-Style Tests (5+ tests)
// ============================================================================

#[test]
fn test_full_workflow_simple() {
    let mut stack = DebugMarkerStack::new();

    // Frame start
    stack.push_group(DebugLabel::with_color("Frame", colors::GEOMETRY));

    // Shadow pass
    stack.push_group(DebugLabel::with_color("Shadows", colors::SHADOW));
    stack.insert_marker(&DebugLabel::new("Cascade 0"));
    stack.insert_marker(&DebugLabel::new("Cascade 1"));
    stack.pop_group();

    // Lighting
    stack.push_group(DebugLabel::with_color("Lighting", colors::LIGHTING));
    stack.insert_marker(&DebugLabel::new("Point Lights"));
    stack.pop_group();

    // Frame end
    stack.pop_group();

    assert!(stack.is_empty());
}

#[test]
fn test_full_workflow_with_profiling() {
    let mut stack = DebugMarkerStack::with_profiling();

    stack.push_group(DebugLabel::new("Profiled Frame"));
    thread::sleep(Duration::from_millis(1));
    stack.push_group(DebugLabel::new("Profiled Pass"));
    thread::sleep(Duration::from_millis(1));

    let inner = stack.pop_group().unwrap();
    assert!(inner.elapsed_ms().unwrap() >= 0.0);

    let outer = stack.pop_group().unwrap();
    assert!(outer.elapsed_ms().unwrap() >= inner.elapsed_ms().unwrap_or(0.0));
}

#[test]
fn test_full_workflow_path_tracking() {
    let mut stack = DebugMarkerStack::new();

    stack.push_group(DebugLabel::new("Render"));
    assert_eq!(stack.path(), "Render");

    stack.push_group(DebugLabel::new("GBuffer"));
    assert_eq!(stack.path(), "Render/GBuffer");

    stack.push_group(DebugLabel::new("Albedo"));
    assert_eq!(stack.path(), "Render/GBuffer/Albedo");

    stack.pop_group();
    assert_eq!(stack.path(), "Render/GBuffer");

    stack.pop_group();
    assert_eq!(stack.path(), "Render");

    stack.pop_group();
    assert_eq!(stack.path(), "");
}

#[test]
fn test_full_workflow_label_hierarchy() {
    let frame = DebugLabel::with_color("Frame", [0.5, 0.5, 0.5, 1.0]);
    let shadows = frame.child("Shadows");
    let cascade0 = shadows.child("Cascade0");
    let objects = cascade0.child("Objects");

    assert_eq!(objects.as_wgpu_label(), "Frame/Shadows/Cascade0/Objects");
    assert_eq!(objects.color, frame.color);
}

#[test]
fn test_full_workflow_mixed_labels() {
    let mut stack = DebugMarkerStack::new();

    // Mix of label creation methods
    stack.push_group(DebugLabel::new("Owned".to_string()));
    stack.push_group(DebugLabel::new_static("Static"));
    stack.push_group(DebugLabel::with_color("Colored", [1.0, 0.0, 0.0, 1.0]));
    stack.push_group("Converted".into());

    assert_eq!(stack.current_depth(), 4);
    assert_eq!(stack.path(), "Owned/Static/Colored/Converted");
}

// ============================================================================
// Test count verification
// ============================================================================

#[test]
fn test_count_verification() {
    // This test exists to document the test count
    // Category 1: 20+ tests (DebugLabel Construction)
    // Category 2: 15+ tests (DebugLabel Color Helpers)
    // Category 3: 12+ tests (DebugLabel Child/Nesting)
    // Category 4: 15+ tests (DebugLabel Traits)
    // Category 5: 20+ tests (DebugGroup)
    // Category 6: 20+ tests (DebugMarkerStack Basic)
    // Category 7: 20+ tests (DebugMarkerStack Push/Pop)
    // Category 8: 15+ tests (DebugMarkerStack Accessors)
    // Category 9: 15+ tests (Colors Module)
    // Category 10: 10+ tests (Edge: Empty Strings)
    // Category 11: 10+ tests (Edge: Unicode)
    // Category 12: 8+ tests (Edge: Long Strings)
    // Category 13: 10+ tests (Edge: Special Characters)
    // Category 14: 10+ tests (Deep Nesting)
    // Category 15: 10+ tests (Rapid Push/Pop)
    // Category 16: 5+ tests (DebugContextOps)
    // Category 17: 9+ tests (Thread Safety)
    // Category 18: 5+ tests (Memory/Performance)
    // Category 19: 5+ tests (Const Context)
    // Category 20: 5+ tests (Integration)
    // Total: 200+ tests
}
