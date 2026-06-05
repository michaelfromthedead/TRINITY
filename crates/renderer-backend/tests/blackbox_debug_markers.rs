// Blackbox contract tests for GPU Debug Marker System (T-WGPU-P7.3.1).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::debug::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria:
//   1.  Label creation with static/owned strings and colors
//   2.  Debug group lifecycle with depth tracking
//   3.  Marker stack push/pop with overflow/underflow protection
//   4.  Path generation for nested groups
//   5.  Profiling timing accuracy
//   6.  Predefined color schemes validation
//   7.  Real-world frame debug structure patterns
//   8.  Edge cases: empty strings, unicode, special characters, long strings

use renderer_backend::debug::{
    colors, DebugContextOps, DebugGroup, DebugLabel, DebugMarkerStack, DebugScopeGuard,
};
use std::borrow::Cow;
use std::time::Duration;

// =============================================================================
// MODULE: DebugLabel Creation Workflows
// =============================================================================

mod label_creation {
    use super::*;

    // -------------------------------------------------------------------------
    // Static String Labels
    // -------------------------------------------------------------------------

    #[test]
    fn create_label_with_static_string_literal() {
        let label = DebugLabel::new_static("GBuffer Pass");
        assert_eq!(label.as_wgpu_label(), "GBuffer Pass");
        assert!(!label.has_color());
    }

    #[test]
    fn static_string_label_uses_borrowed_cow() {
        let label = DebugLabel::new_static("Static Label");
        // Verify it's a borrowed string (no allocation)
        match &label.name {
            Cow::Borrowed(s) => assert_eq!(*s, "Static Label"),
            Cow::Owned(_) => panic!("Expected borrowed Cow for static label"),
        }
    }

    #[test]
    fn multiple_static_labels_share_string_references() {
        let label1 = DebugLabel::new_static("Shared");
        let label2 = DebugLabel::new_static("Shared");
        // Both should be borrowed
        assert!(matches!(&label1.name, Cow::Borrowed(_)));
        assert!(matches!(&label2.name, Cow::Borrowed(_)));
    }

    #[test]
    fn const_static_label_compiles() {
        const LABEL: DebugLabel = DebugLabel::new_static("Compile-Time Label");
        assert_eq!(LABEL.as_wgpu_label(), "Compile-Time Label");
    }

    // -------------------------------------------------------------------------
    // Owned String Labels
    // -------------------------------------------------------------------------

    #[test]
    fn create_label_with_owned_string() {
        let name = String::from("Dynamic Label");
        let label = DebugLabel::new(name);
        assert_eq!(label.as_wgpu_label(), "Dynamic Label");
    }

    #[test]
    fn create_label_from_string_type() {
        let label: DebugLabel = String::from("From String").into();
        assert_eq!(label.as_wgpu_label(), "From String");
    }

    #[test]
    fn create_label_from_format_macro() {
        let index = 42;
        let label = DebugLabel::new(format!("Draw Call #{}", index));
        assert_eq!(label.as_wgpu_label(), "Draw Call #42");
    }

    #[test]
    fn create_label_from_str_reference() {
        let s = "Reference";
        let label = DebugLabel::new(s);
        assert_eq!(label.as_wgpu_label(), "Reference");
    }

    // -------------------------------------------------------------------------
    // Labels with Colors
    // -------------------------------------------------------------------------

    #[test]
    fn create_label_with_rgba_color() {
        let label = DebugLabel::with_color("Colored Pass", [1.0, 0.5, 0.0, 1.0]);
        assert!(label.has_color());
        assert_eq!(label.color, Some([1.0, 0.5, 0.0, 1.0]));
    }

    #[test]
    fn create_static_label_with_color() {
        let label = DebugLabel::with_static_color("Static Colored", [0.2, 0.4, 0.6, 0.8]);
        assert!(label.has_color());
        assert_eq!(label.color, Some([0.2, 0.4, 0.6, 0.8]));
    }

    #[test]
    fn const_static_colored_label_compiles() {
        const COLORED: DebugLabel =
            DebugLabel::with_static_color("Compile-Time Colored", [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(COLORED.color, Some([1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn label_color_extraction_rgb() {
        let label = DebugLabel::with_color("Test", [0.1, 0.2, 0.3, 0.9]);
        let rgb = label.rgb().unwrap();
        assert_eq!(rgb, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn label_color_extraction_rgba_u8() {
        let label = DebugLabel::with_color("Test", [1.0, 0.5, 0.0, 0.5]);
        let rgba = label.rgba_u8().unwrap();
        assert_eq!(rgba[0], 255); // 1.0 -> 255
        assert_eq!(rgba[1], 127); // 0.5 -> 127
        assert_eq!(rgba[2], 0); // 0.0 -> 0
        assert_eq!(rgba[3], 127); // 0.5 -> 127
    }

    #[test]
    fn label_without_color_returns_none_for_rgb() {
        let label = DebugLabel::new("No Color");
        assert!(label.rgb().is_none());
    }

    #[test]
    fn label_without_color_returns_none_for_rgba_u8() {
        let label = DebugLabel::new("No Color");
        assert!(label.rgba_u8().is_none());
    }

    #[test]
    fn color_clamping_handles_overflow() {
        let label = DebugLabel::with_color("Overflow", [2.0, -0.5, 1.5, 1.0]);
        let rgba = label.rgba_u8().unwrap();
        // Values should be clamped to 0-255 range
        assert_eq!(rgba[0], 255); // 2.0 clamped to 255
        assert_eq!(rgba[1], 0); // -0.5 clamped to 0
        assert_eq!(rgba[2], 255); // 1.5 clamped to 255
    }

    // -------------------------------------------------------------------------
    // Child Labels for Hierarchy
    // -------------------------------------------------------------------------

    #[test]
    fn create_child_label_appends_suffix() {
        let parent = DebugLabel::new("Shadow Pass");
        let child = parent.child("Cascade 0");
        assert_eq!(child.as_wgpu_label(), "Shadow Pass/Cascade 0");
    }

    #[test]
    fn child_label_inherits_color_from_parent() {
        let parent = DebugLabel::with_color("Parent", [1.0, 0.0, 0.0, 1.0]);
        let child = parent.child("Child");
        assert_eq!(child.color, Some([1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn nested_child_labels_build_path() {
        let root = DebugLabel::new("Frame");
        let level1 = root.child("Geometry");
        let level2 = level1.child("Opaque");
        let level3 = level2.child("Static Meshes");
        assert_eq!(
            level3.as_wgpu_label(),
            "Frame/Geometry/Opaque/Static Meshes"
        );
    }

    #[test]
    fn child_label_from_static_parent() {
        let parent = DebugLabel::new_static("Static Parent");
        let child = parent.child("Dynamic Child");
        assert_eq!(child.as_wgpu_label(), "Static Parent/Dynamic Child");
        // Child should be owned since we concatenate
        assert!(matches!(&child.name, Cow::Owned(_)));
    }

    // -------------------------------------------------------------------------
    // wgpu Format Conversion
    // -------------------------------------------------------------------------

    #[test]
    fn as_wgpu_label_returns_name_string() {
        let label = DebugLabel::new("Test Label");
        let wgpu_label: &str = label.as_wgpu_label();
        assert_eq!(wgpu_label, "Test Label");
    }

    #[test]
    fn display_trait_shows_name() {
        let label = DebugLabel::new("Display Test");
        assert_eq!(format!("{}", label), "Display Test");
    }

    #[test]
    fn debug_trait_shows_full_struct() {
        let label = DebugLabel::with_color("Debug", [1.0, 0.0, 0.0, 1.0]);
        let debug_str = format!("{:?}", label);
        assert!(debug_str.contains("Debug"));
        assert!(debug_str.contains("1.0"));
    }

    // -------------------------------------------------------------------------
    // From Traits
    // -------------------------------------------------------------------------

    #[test]
    fn from_static_str_creates_borrowed_label() {
        let label: DebugLabel = "Static Str".into();
        assert_eq!(label.as_wgpu_label(), "Static Str");
    }

    #[test]
    fn from_string_creates_owned_label() {
        let s = String::from("Owned String");
        let label: DebugLabel = s.into();
        assert_eq!(label.as_wgpu_label(), "Owned String");
    }

    // -------------------------------------------------------------------------
    // Default and Clone
    // -------------------------------------------------------------------------

    #[test]
    fn default_label_is_empty() {
        let label = DebugLabel::default();
        assert_eq!(label.as_wgpu_label(), "");
        assert!(!label.has_color());
    }

    #[test]
    fn clone_preserves_all_fields() {
        let original = DebugLabel::with_color("Original", [0.5, 0.5, 0.5, 0.5]);
        let cloned = original.clone();
        assert_eq!(original, cloned);
        assert_eq!(cloned.as_wgpu_label(), "Original");
        assert_eq!(cloned.color, Some([0.5, 0.5, 0.5, 0.5]));
    }

    #[test]
    fn equality_checks_name_and_color() {
        let a = DebugLabel::with_color("Same", [1.0, 0.0, 0.0, 1.0]);
        let b = DebugLabel::with_color("Same", [1.0, 0.0, 0.0, 1.0]);
        let c = DebugLabel::with_color("Same", [0.0, 1.0, 0.0, 1.0]); // different color
        let d = DebugLabel::with_color("Different", [1.0, 0.0, 0.0, 1.0]); // different name

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }
}

// =============================================================================
// MODULE: DebugGroup Lifecycle
// =============================================================================

mod debug_group_lifecycle {
    use super::*;

    // -------------------------------------------------------------------------
    // Basic Group Creation
    // -------------------------------------------------------------------------

    #[test]
    fn create_group_without_profiling() {
        let group = DebugGroup::new(DebugLabel::new("Test Group"));
        assert_eq!(group.label.as_wgpu_label(), "Test Group");
        assert_eq!(group.depth, 0);
        assert!(!group.has_profiling());
        assert!(group.start_time.is_none());
    }

    #[test]
    fn create_group_with_specified_depth() {
        let group = DebugGroup::with_depth(DebugLabel::new("Nested"), 5);
        assert_eq!(group.depth, 5);
        assert!(!group.has_profiling());
    }

    #[test]
    fn create_group_with_profiling() {
        let group = DebugGroup::with_profiling(DebugLabel::new("Profiled"));
        assert!(group.has_profiling());
        assert!(group.start_time.is_some());
        assert_eq!(group.depth, 0);
    }

    #[test]
    fn create_group_with_depth_and_profiling() {
        let group = DebugGroup::with_depth_and_profiling(DebugLabel::new("Full"), 3);
        assert_eq!(group.depth, 3);
        assert!(group.has_profiling());
    }

    // -------------------------------------------------------------------------
    // Profiling Timing
    // -------------------------------------------------------------------------

    #[test]
    fn profiling_elapsed_increases_over_time() {
        let group = DebugGroup::with_profiling(DebugLabel::new("Timed"));
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = group.elapsed().unwrap();
        assert!(elapsed >= Duration::from_millis(5));
    }

    #[test]
    fn profiling_elapsed_ms_returns_milliseconds() {
        let group = DebugGroup::with_profiling(DebugLabel::new("Timed"));
        std::thread::sleep(Duration::from_millis(10));
        let ms = group.elapsed_ms().unwrap();
        assert!(ms >= 10.0, "Expected >= 10ms, got {}ms", ms);
    }

    #[test]
    fn non_profiled_group_returns_none_for_elapsed() {
        let group = DebugGroup::new(DebugLabel::new("No Timing"));
        assert!(group.elapsed().is_none());
        assert!(group.elapsed_ms().is_none());
    }

    #[test]
    fn profiling_captures_start_time_immediately() {
        let before = std::time::Instant::now();
        let group = DebugGroup::with_profiling(DebugLabel::new("Immediate"));
        let after = std::time::Instant::now();

        let start = group.start_time.unwrap();
        assert!(start >= before);
        assert!(start <= after);
    }

    // -------------------------------------------------------------------------
    // Group Clone
    // -------------------------------------------------------------------------

    #[test]
    fn clone_group_preserves_all_fields() {
        let original = DebugGroup::with_depth_and_profiling(DebugLabel::new("Clone Me"), 7);
        let cloned = original.clone();

        assert_eq!(cloned.label.as_wgpu_label(), "Clone Me");
        assert_eq!(cloned.depth, 7);
        assert!(cloned.has_profiling());
    }

    #[test]
    fn clone_group_preserves_start_time() {
        let group = DebugGroup::with_profiling(DebugLabel::new("Timed Clone"));
        std::thread::sleep(Duration::from_millis(1));
        let cloned = group.clone();

        // Both should report similar elapsed times
        let orig_ms = group.elapsed_ms().unwrap();
        let clone_ms = cloned.elapsed_ms().unwrap();
        assert!((orig_ms - clone_ms).abs() < 1.0); // Within 1ms
    }
}

// =============================================================================
// MODULE: DebugMarkerStack Operations
// =============================================================================

mod marker_stack_operations {
    use super::*;

    // -------------------------------------------------------------------------
    // Stack Construction
    // -------------------------------------------------------------------------

    #[test]
    fn new_stack_is_empty() {
        let stack = DebugMarkerStack::new();
        assert!(stack.is_empty());
        assert_eq!(stack.current_depth(), 0);
    }

    #[test]
    fn new_stack_has_default_max_depth() {
        let stack = DebugMarkerStack::new();
        assert_eq!(stack.max_depth(), DebugMarkerStack::DEFAULT_MAX_DEPTH);
        assert_eq!(stack.max_depth(), 16);
    }

    #[test]
    fn custom_max_depth_stack() {
        let stack = DebugMarkerStack::with_max_depth(8);
        assert_eq!(stack.max_depth(), 8);
    }

    #[test]
    fn profiling_enabled_stack() {
        let stack = DebugMarkerStack::with_profiling();
        assert!(stack.profiling_enabled());
    }

    #[test]
    fn default_stack_has_profiling_disabled() {
        let stack = DebugMarkerStack::new();
        assert!(!stack.profiling_enabled());
    }

    // -------------------------------------------------------------------------
    // Push/Pop Balanced Operations
    // -------------------------------------------------------------------------

    #[test]
    fn push_increases_depth() {
        let mut stack = DebugMarkerStack::new();
        assert!(stack.push_group(DebugLabel::new("Level 1")));
        assert_eq!(stack.current_depth(), 1);

        assert!(stack.push_group(DebugLabel::new("Level 2")));
        assert_eq!(stack.current_depth(), 2);

        assert!(stack.push_group(DebugLabel::new("Level 3")));
        assert_eq!(stack.current_depth(), 3);
    }

    #[test]
    fn pop_decreases_depth() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("A"));
        stack.push_group(DebugLabel::new("B"));
        stack.push_group(DebugLabel::new("C"));
        assert_eq!(stack.current_depth(), 3);

        stack.pop_group();
        assert_eq!(stack.current_depth(), 2);

        stack.pop_group();
        assert_eq!(stack.current_depth(), 1);

        stack.pop_group();
        assert_eq!(stack.current_depth(), 0);
        assert!(stack.is_empty());
    }

    #[test]
    fn pop_returns_correct_group() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("First"));
        stack.push_group(DebugLabel::new("Second"));
        stack.push_group(DebugLabel::new("Third"));

        let third = stack.pop_group().unwrap();
        assert_eq!(third.label.as_wgpu_label(), "Third");
        assert_eq!(third.depth, 2);

        let second = stack.pop_group().unwrap();
        assert_eq!(second.label.as_wgpu_label(), "Second");
        assert_eq!(second.depth, 1);

        let first = stack.pop_group().unwrap();
        assert_eq!(first.label.as_wgpu_label(), "First");
        assert_eq!(first.depth, 0);
    }

    #[test]
    fn balanced_push_pop_sequence() {
        let mut stack = DebugMarkerStack::new();

        // Simulate nested rendering passes
        stack.push_group(DebugLabel::new("Frame"));
        stack.push_group(DebugLabel::new("Shadow"));
        stack.pop_group();
        stack.push_group(DebugLabel::new("GBuffer"));
        stack.push_group(DebugLabel::new("Opaque"));
        stack.pop_group();
        stack.push_group(DebugLabel::new("Transparent"));
        stack.pop_group();
        stack.pop_group(); // GBuffer
        stack.push_group(DebugLabel::new("Lighting"));
        stack.pop_group();
        stack.pop_group(); // Frame

        assert!(stack.is_empty());
    }

    // -------------------------------------------------------------------------
    // Stack Overflow Protection
    // -------------------------------------------------------------------------

    #[test]
    fn push_fails_at_max_depth() {
        let mut stack = DebugMarkerStack::with_max_depth(3);

        assert!(stack.push_group(DebugLabel::new("Level 1")));
        assert!(stack.push_group(DebugLabel::new("Level 2")));
        assert!(stack.push_group(DebugLabel::new("Level 3")));
        assert!(!stack.push_group(DebugLabel::new("Level 4"))); // Should fail

        assert_eq!(stack.current_depth(), 3);
    }

    #[test]
    fn can_push_returns_false_at_max_depth() {
        let mut stack = DebugMarkerStack::with_max_depth(2);

        assert!(stack.can_push());
        stack.push_group(DebugLabel::new("One"));
        assert!(stack.can_push());
        stack.push_group(DebugLabel::new("Two"));
        assert!(!stack.can_push());
    }

    #[test]
    fn can_push_becomes_true_after_pop() {
        let mut stack = DebugMarkerStack::with_max_depth(2);
        stack.push_group(DebugLabel::new("One"));
        stack.push_group(DebugLabel::new("Two"));
        assert!(!stack.can_push());

        stack.pop_group();
        assert!(stack.can_push());
    }

    #[test]
    fn default_max_depth_allows_16_levels() {
        let mut stack = DebugMarkerStack::new();

        for i in 0..16 {
            assert!(
                stack.push_group(DebugLabel::new(format!("Level {}", i))),
                "Failed to push level {}",
                i
            );
        }
        assert_eq!(stack.current_depth(), 16);
        assert!(!stack.push_group(DebugLabel::new("Level 16")));
    }

    // -------------------------------------------------------------------------
    // Stack Underflow Handling
    // -------------------------------------------------------------------------

    #[test]
    fn pop_empty_stack_returns_none() {
        let mut stack = DebugMarkerStack::new();
        assert!(stack.pop_group().is_none());
    }

    #[test]
    fn multiple_pops_on_empty_stack_return_none() {
        let mut stack = DebugMarkerStack::new();
        assert!(stack.pop_group().is_none());
        assert!(stack.pop_group().is_none());
        assert!(stack.pop_group().is_none());
    }

    #[test]
    fn pop_after_emptying_stack_returns_none() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("Only"));
        stack.pop_group();
        assert!(stack.pop_group().is_none());
    }

    // -------------------------------------------------------------------------
    // Insert Markers at Various Depths
    // -------------------------------------------------------------------------

    #[test]
    fn insert_marker_at_root_level() {
        let mut stack = DebugMarkerStack::new();
        stack.insert_marker(&DebugLabel::new("Root Marker"));
        assert_eq!(stack.markers_at_depth(), 1);
    }

    #[test]
    fn insert_multiple_markers_at_same_depth() {
        let mut stack = DebugMarkerStack::new();
        stack.insert_marker(&DebugLabel::new("Marker 1"));
        stack.insert_marker(&DebugLabel::new("Marker 2"));
        stack.insert_marker(&DebugLabel::new("Marker 3"));
        assert_eq!(stack.markers_at_depth(), 3);
    }

    #[test]
    fn markers_reset_on_push() {
        let mut stack = DebugMarkerStack::new();
        stack.insert_marker(&DebugLabel::new("Before Push"));
        assert_eq!(stack.markers_at_depth(), 1);

        stack.push_group(DebugLabel::new("New Group"));
        assert_eq!(stack.markers_at_depth(), 0);
    }

    #[test]
    fn markers_reset_on_pop() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("Group"));
        stack.insert_marker(&DebugLabel::new("Inside"));
        stack.insert_marker(&DebugLabel::new("Inside 2"));
        assert_eq!(stack.markers_at_depth(), 2);

        stack.pop_group();
        assert_eq!(stack.markers_at_depth(), 0);
    }

    #[test]
    fn markers_at_different_depths() {
        let mut stack = DebugMarkerStack::new();

        // Root level markers
        stack.insert_marker(&DebugLabel::new("Root 1"));
        assert_eq!(stack.markers_at_depth(), 1);

        // Level 1 markers
        stack.push_group(DebugLabel::new("Level 1"));
        stack.insert_marker(&DebugLabel::new("L1 Marker 1"));
        stack.insert_marker(&DebugLabel::new("L1 Marker 2"));
        assert_eq!(stack.markers_at_depth(), 2);

        // Level 2 markers
        stack.push_group(DebugLabel::new("Level 2"));
        stack.insert_marker(&DebugLabel::new("L2 Only"));
        assert_eq!(stack.markers_at_depth(), 1);
    }

    // -------------------------------------------------------------------------
    // Path Generation for Nested Groups
    // -------------------------------------------------------------------------

    #[test]
    fn path_empty_stack_returns_empty_string() {
        let stack = DebugMarkerStack::new();
        assert_eq!(stack.path(), "");
    }

    #[test]
    fn path_single_group() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("Frame"));
        assert_eq!(stack.path(), "Frame");
    }

    #[test]
    fn path_nested_groups() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("Frame"));
        stack.push_group(DebugLabel::new("Geometry"));
        stack.push_group(DebugLabel::new("Opaque"));
        assert_eq!(stack.path(), "Frame/Geometry/Opaque");
    }

    #[test]
    fn path_updates_on_push_pop() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("A"));
        assert_eq!(stack.path(), "A");

        stack.push_group(DebugLabel::new("B"));
        assert_eq!(stack.path(), "A/B");

        stack.pop_group();
        assert_eq!(stack.path(), "A");

        stack.push_group(DebugLabel::new("C"));
        assert_eq!(stack.path(), "A/C");
    }

    // -------------------------------------------------------------------------
    // Current Group Access
    // -------------------------------------------------------------------------

    #[test]
    fn current_group_empty_stack_returns_none() {
        let stack = DebugMarkerStack::new();
        assert!(stack.current_group().is_none());
    }

    #[test]
    fn current_group_returns_innermost() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("Outer"));
        assert_eq!(stack.current_group().unwrap().label.as_wgpu_label(), "Outer");

        stack.push_group(DebugLabel::new("Inner"));
        assert_eq!(stack.current_group().unwrap().label.as_wgpu_label(), "Inner");

        stack.pop_group();
        assert_eq!(stack.current_group().unwrap().label.as_wgpu_label(), "Outer");
    }

    // -------------------------------------------------------------------------
    // Groups Slice Access
    // -------------------------------------------------------------------------

    #[test]
    fn groups_returns_all_active_groups() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("First"));
        stack.push_group(DebugLabel::new("Second"));
        stack.push_group(DebugLabel::new("Third"));

        let groups = stack.groups();
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].label.as_wgpu_label(), "First");
        assert_eq!(groups[1].label.as_wgpu_label(), "Second");
        assert_eq!(groups[2].label.as_wgpu_label(), "Third");
    }

    #[test]
    fn groups_order_is_outermost_first() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("Root"));
        stack.push_group(DebugLabel::new("Child"));
        stack.push_group(DebugLabel::new("Grandchild"));

        let groups = stack.groups();
        assert_eq!(groups[0].depth, 0); // Root at depth 0
        assert_eq!(groups[1].depth, 1); // Child at depth 1
        assert_eq!(groups[2].depth, 2); // Grandchild at depth 2
    }

    // -------------------------------------------------------------------------
    // Clear Operation
    // -------------------------------------------------------------------------

    #[test]
    fn clear_empties_stack() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("A"));
        stack.push_group(DebugLabel::new("B"));
        stack.insert_marker(&DebugLabel::new("M"));

        stack.clear();

        assert!(stack.is_empty());
        assert_eq!(stack.current_depth(), 0);
        assert_eq!(stack.markers_at_depth(), 0);
    }

    // -------------------------------------------------------------------------
    // Profiling Mode
    // -------------------------------------------------------------------------

    #[test]
    fn set_profiling_enables_timestamps() {
        let mut stack = DebugMarkerStack::new();
        assert!(!stack.profiling_enabled());

        stack.set_profiling(true);
        assert!(stack.profiling_enabled());

        stack.push_group(DebugLabel::new("Profiled Group"));
        assert!(stack.current_group().unwrap().has_profiling());
    }

    #[test]
    fn set_profiling_can_be_toggled() {
        let mut stack = DebugMarkerStack::new();

        stack.set_profiling(true);
        stack.push_group(DebugLabel::new("Profiled"));
        assert!(stack.current_group().unwrap().has_profiling());

        stack.set_profiling(false);
        stack.push_group(DebugLabel::new("Not Profiled"));
        assert!(!stack.current_group().unwrap().has_profiling());
    }

    #[test]
    fn with_profiling_constructor_enables_timestamps() {
        let mut stack = DebugMarkerStack::with_profiling();
        stack.push_group(DebugLabel::new("Auto Profiled"));

        let group = stack.current_group().unwrap();
        assert!(group.has_profiling());
        assert!(group.start_time.is_some());
    }

    // -------------------------------------------------------------------------
    // Default Trait
    // -------------------------------------------------------------------------

    #[test]
    fn default_stack_is_empty() {
        let stack: DebugMarkerStack = Default::default();
        assert!(stack.is_empty());
        assert_eq!(stack.max_depth(), DebugMarkerStack::DEFAULT_MAX_DEPTH);
    }
}

// =============================================================================
// MODULE: Color Schemes
// =============================================================================

mod color_schemes {
    use super::*;

    // -------------------------------------------------------------------------
    // Predefined Colors Validation
    // -------------------------------------------------------------------------

    #[test]
    fn geometry_color_is_orange() {
        let color = colors::GEOMETRY;
        assert!((color[0] - 1.0).abs() < 0.01); // R ~ 1.0
        assert!((color[1] - 0.6).abs() < 0.01); // G ~ 0.6
        assert!((color[2] - 0.2).abs() < 0.01); // B ~ 0.2
        assert!((color[3] - 1.0).abs() < 0.01); // A = 1.0
    }

    #[test]
    fn shadow_color_is_dark_blue() {
        let color = colors::SHADOW;
        assert!(color[0] < 0.5); // R low
        assert!(color[1] < 0.5); // G low
        assert!(color[2] > 0.5); // B high (relatively)
    }

    #[test]
    fn lighting_color_is_yellow() {
        let color = colors::LIGHTING;
        assert!(color[0] > 0.9); // R high
        assert!(color[1] > 0.8); // G high
        assert!(color[2] < 0.5); // B low
    }

    #[test]
    fn all_predefined_colors_have_valid_components() {
        let all_colors = [
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

        for color in all_colors {
            for (i, component) in color.iter().enumerate() {
                assert!(
                    *component >= 0.0 && *component <= 1.0,
                    "Color component {} = {} is out of [0.0, 1.0] range",
                    i,
                    component
                );
            }
        }
    }

    #[test]
    fn all_predefined_colors_are_fully_opaque_or_semi_transparent() {
        // All should have alpha >= 0.5
        let all_colors = [
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

        for color in all_colors {
            assert!(color[3] >= 0.5, "Alpha {} is too low", color[3]);
        }
    }

    // -------------------------------------------------------------------------
    // Custom Colors
    // -------------------------------------------------------------------------

    #[test]
    fn custom_color_with_full_alpha() {
        let label = DebugLabel::with_color("Custom", [0.25, 0.5, 0.75, 1.0]);
        let rgba = label.rgba_u8().unwrap();
        assert_eq!(rgba[0], 63); // 0.25 * 255
        assert_eq!(rgba[1], 127); // 0.5 * 255
        assert_eq!(rgba[2], 191); // 0.75 * 255
        assert_eq!(rgba[3], 255); // 1.0 * 255
    }

    #[test]
    fn transparent_color() {
        let label = DebugLabel::with_color("Transparent", [1.0, 1.0, 1.0, 0.0]);
        let rgba = label.rgba_u8().unwrap();
        assert_eq!(rgba[3], 0); // Fully transparent
    }

    // -------------------------------------------------------------------------
    // Using Colors with Labels
    // -------------------------------------------------------------------------

    #[test]
    fn use_predefined_color_with_label() {
        let label = DebugLabel::with_color("Shadow Pass", colors::SHADOW);
        assert_eq!(label.color, Some(colors::SHADOW));
    }

    #[test]
    fn different_pass_types_have_distinct_colors() {
        assert_ne!(colors::GEOMETRY, colors::SHADOW);
        assert_ne!(colors::SHADOW, colors::LIGHTING);
        assert_ne!(colors::LIGHTING, colors::POST_PROCESS);
        assert_ne!(colors::POST_PROCESS, colors::COMPUTE);
        assert_ne!(colors::COMPUTE, colors::UI);
    }
}

// =============================================================================
// MODULE: Real-World Usage Patterns
// =============================================================================

mod real_world_patterns {
    use super::*;

    // -------------------------------------------------------------------------
    // Frame Debug Structure
    // -------------------------------------------------------------------------

    #[test]
    fn typical_frame_structure() {
        let mut stack = DebugMarkerStack::with_profiling();

        // BeginFrame
        assert!(stack.push_group(DebugLabel::with_color("Frame", [1.0, 1.0, 1.0, 1.0])));
        assert_eq!(stack.path(), "Frame");

        // Shadow Pass
        assert!(stack.push_group(DebugLabel::with_color("Shadow Pass", colors::SHADOW)));
        stack.insert_marker(&DebugLabel::new("Cascade 0"));
        stack.insert_marker(&DebugLabel::new("Cascade 1"));
        stack.insert_marker(&DebugLabel::new("Cascade 2"));
        assert_eq!(stack.markers_at_depth(), 3);
        stack.pop_group();

        // Geometry Pass
        assert!(stack.push_group(DebugLabel::with_color("Geometry Pass", colors::GEOMETRY)));
        stack.push_group(DebugLabel::new("Opaque"));
        stack.insert_marker(&DebugLabel::new("Static Meshes"));
        stack.insert_marker(&DebugLabel::new("Skinned Meshes"));
        stack.pop_group();
        stack.push_group(DebugLabel::new("Transparent"));
        stack.insert_marker(&DebugLabel::new("Sorted Draw"));
        stack.pop_group();
        stack.pop_group();

        // Lighting Pass
        assert!(stack.push_group(DebugLabel::with_color("Lighting Pass", colors::LIGHTING)));
        stack.insert_marker(&DebugLabel::new("Directional Light"));
        stack.insert_marker(&DebugLabel::new("Point Lights"));
        stack.insert_marker(&DebugLabel::new("Spot Lights"));
        stack.pop_group();

        // Post-Process Pass
        assert!(stack.push_group(DebugLabel::with_color("Post-Process", colors::POST_PROCESS)));
        stack.insert_marker(&DebugLabel::new("Bloom"));
        stack.insert_marker(&DebugLabel::new("Tone Mapping"));
        stack.insert_marker(&DebugLabel::new("FXAA"));
        stack.pop_group();

        // UI Pass
        assert!(stack.push_group(DebugLabel::with_color("UI", colors::UI)));
        stack.insert_marker(&DebugLabel::new("ImGui"));
        stack.pop_group();

        // EndFrame
        stack.pop_group();
        assert!(stack.is_empty());
    }

    #[test]
    fn cascaded_shadow_map_pattern() {
        let mut stack = DebugMarkerStack::new();

        stack.push_group(DebugLabel::with_color("CSM", colors::SHADOW));
        assert_eq!(stack.path(), "CSM");

        for cascade in 0..4 {
            stack.push_group(DebugLabel::new(format!("Cascade {}", cascade)));
            assert_eq!(stack.path(), format!("CSM/Cascade {}", cascade));

            stack.insert_marker(&DebugLabel::new("Setup Viewport"));
            stack.insert_marker(&DebugLabel::new("Draw Static"));
            stack.insert_marker(&DebugLabel::new("Draw Skinned"));

            stack.pop_group();
        }

        stack.pop_group();
        assert!(stack.is_empty());
    }

    #[test]
    fn deferred_rendering_pipeline() {
        let mut stack = DebugMarkerStack::new();

        // GBuffer Pass
        stack.push_group(DebugLabel::with_color("GBuffer", colors::GEOMETRY));
        stack.insert_marker(&DebugLabel::new("Albedo"));
        stack.insert_marker(&DebugLabel::new("Normal"));
        stack.insert_marker(&DebugLabel::new("Material"));
        stack.insert_marker(&DebugLabel::new("Depth"));
        stack.pop_group();

        // SSAO
        stack.push_group(DebugLabel::with_color("SSAO", colors::COMPUTE));
        stack.insert_marker(&DebugLabel::new("Generate"));
        stack.insert_marker(&DebugLabel::new("Blur"));
        stack.pop_group();

        // Deferred Lighting
        stack.push_group(DebugLabel::with_color("Deferred Lighting", colors::LIGHTING));
        stack.insert_marker(&DebugLabel::new("Light Culling"));
        stack.insert_marker(&DebugLabel::new("Shade"));
        stack.pop_group();

        // Forward Transparent
        stack.push_group(DebugLabel::with_color("Forward", colors::TRANSPARENT));
        stack.insert_marker(&DebugLabel::new("Particles"));
        stack.insert_marker(&DebugLabel::new("Decals"));
        stack.pop_group();

        assert!(stack.is_empty());
    }

    // -------------------------------------------------------------------------
    // GPU Capture Tool Patterns (RenderDoc/PIX)
    // -------------------------------------------------------------------------

    #[test]
    fn renderdoc_compatible_naming() {
        // RenderDoc/PIX prefer descriptive hierarchical names
        let mut stack = DebugMarkerStack::new();

        stack.push_group(DebugLabel::new("Scene: Main"));
        stack.push_group(DebugLabel::new("Pass: Shadow"));
        stack.push_group(DebugLabel::new("Light: Sun"));
        stack.insert_marker(&DebugLabel::new("Draw: Terrain"));
        stack.insert_marker(&DebugLabel::new("Draw: Buildings"));

        // Path should be readable
        assert_eq!(stack.path(), "Scene: Main/Pass: Shadow/Light: Sun");
    }

    #[test]
    fn pix_timing_capture_pattern() {
        // PIX uses debug groups for timing
        let mut stack = DebugMarkerStack::with_profiling();

        stack.push_group(DebugLabel::new("GPU Frame"));
        std::thread::sleep(Duration::from_millis(1));

        stack.push_group(DebugLabel::new("Render"));
        std::thread::sleep(Duration::from_millis(1));
        let render_group = stack.pop_group().unwrap();
        assert!(render_group.elapsed_ms().unwrap() >= 1.0);

        stack.push_group(DebugLabel::new("Post"));
        std::thread::sleep(Duration::from_millis(1));
        let post_group = stack.pop_group().unwrap();
        assert!(post_group.elapsed_ms().unwrap() >= 1.0);

        let frame_group = stack.pop_group().unwrap();
        assert!(frame_group.elapsed_ms().unwrap() >= 2.0);
    }

    // -------------------------------------------------------------------------
    // Compute Shader Patterns
    // -------------------------------------------------------------------------

    #[test]
    fn compute_pass_structure() {
        let mut stack = DebugMarkerStack::new();

        stack.push_group(DebugLabel::with_color("Compute: Particle Sim", colors::COMPUTE));
        stack.insert_marker(&DebugLabel::new("Update Positions"));
        stack.insert_marker(&DebugLabel::new("Apply Forces"));
        stack.insert_marker(&DebugLabel::new("Collision Detection"));
        stack.insert_marker(&DebugLabel::new("Update Velocities"));
        stack.pop_group();

        assert!(stack.is_empty());
    }

    #[test]
    fn raytracing_pass_structure() {
        let mut stack = DebugMarkerStack::new();

        stack.push_group(DebugLabel::with_color("Ray Tracing", colors::RAYTRACING));
        stack.push_group(DebugLabel::new("Build TLAS"));
        stack.pop_group();
        stack.push_group(DebugLabel::new("Trace Primary"));
        stack.pop_group();
        stack.push_group(DebugLabel::new("Trace Shadows"));
        stack.pop_group();
        stack.push_group(DebugLabel::new("Denoise"));
        stack.pop_group();
        stack.pop_group();

        assert!(stack.is_empty());
    }

    // -------------------------------------------------------------------------
    // Transfer Operations
    // -------------------------------------------------------------------------

    #[test]
    fn transfer_operation_markers() {
        let mut stack = DebugMarkerStack::new();

        stack.push_group(DebugLabel::with_color("Transfer", colors::TRANSFER));
        stack.insert_marker(&DebugLabel::new("Upload Vertex Data"));
        stack.insert_marker(&DebugLabel::new("Upload Textures"));
        stack.insert_marker(&DebugLabel::new("Readback Query Results"));
        stack.pop_group();

        assert!(stack.is_empty());
    }
}

// =============================================================================
// MODULE: Edge Cases and Negative Tests
// =============================================================================

mod edge_cases {
    use super::*;

    // -------------------------------------------------------------------------
    // Empty Strings
    // -------------------------------------------------------------------------

    #[test]
    fn empty_string_label() {
        let label = DebugLabel::new("");
        assert_eq!(label.as_wgpu_label(), "");
    }

    #[test]
    fn empty_static_label() {
        let label = DebugLabel::new_static("");
        assert_eq!(label.as_wgpu_label(), "");
    }

    #[test]
    fn empty_label_child() {
        let parent = DebugLabel::new("");
        let child = parent.child("Child");
        assert_eq!(child.as_wgpu_label(), "/Child");
    }

    #[test]
    fn child_with_empty_suffix() {
        let parent = DebugLabel::new("Parent");
        let child = parent.child("");
        assert_eq!(child.as_wgpu_label(), "Parent/");
    }

    // -------------------------------------------------------------------------
    // Unicode and Special Characters
    // -------------------------------------------------------------------------

    #[test]
    fn unicode_label() {
        let label = DebugLabel::new("Pass");
        assert_eq!(label.as_wgpu_label(), "Pass");
    }

    #[test]
    fn chinese_characters() {
        let label = DebugLabel::new("Pass");
        assert_eq!(label.as_wgpu_label(), "Pass");
    }

    #[test]
    fn emoji_in_label() {
        let label = DebugLabel::new("Frame Start");
        assert_eq!(label.as_wgpu_label(), "Frame Start");
    }

    #[test]
    fn special_characters() {
        let label = DebugLabel::new("Test/with\\special[chars]{}()");
        assert_eq!(label.as_wgpu_label(), "Test/with\\special[chars]{}()");
    }

    #[test]
    fn newlines_and_tabs() {
        let label = DebugLabel::new("Line1\nLine2\tTabbed");
        assert_eq!(label.as_wgpu_label(), "Line1\nLine2\tTabbed");
    }

    #[test]
    fn null_characters_in_string() {
        let label = DebugLabel::new("Before\0After");
        assert_eq!(label.as_wgpu_label(), "Before\0After");
    }

    // -------------------------------------------------------------------------
    // Very Long Strings
    // -------------------------------------------------------------------------

    #[test]
    fn very_long_label_name() {
        let long_name = "A".repeat(10000);
        let label = DebugLabel::new(long_name.clone());
        assert_eq!(label.as_wgpu_label(), long_name);
    }

    #[test]
    fn deeply_nested_child_path() {
        let mut label = DebugLabel::new("Root");
        for i in 0..100 {
            label = label.child(&format!("Level{}", i));
        }
        assert!(label.as_wgpu_label().starts_with("Root/Level0/Level1"));
        assert!(label.as_wgpu_label().ends_with("Level99"));
    }

    // -------------------------------------------------------------------------
    // Boundary Values
    // -------------------------------------------------------------------------

    #[test]
    fn max_depth_zero() {
        let mut stack = DebugMarkerStack::with_max_depth(0);
        assert!(!stack.push_group(DebugLabel::new("Any")));
        assert!(stack.is_empty());
    }

    #[test]
    fn max_depth_one() {
        let mut stack = DebugMarkerStack::with_max_depth(1);
        assert!(stack.push_group(DebugLabel::new("Only")));
        assert!(!stack.push_group(DebugLabel::new("Second")));
        assert_eq!(stack.current_depth(), 1);
    }

    #[test]
    fn color_boundary_values() {
        // Min values
        let min_label = DebugLabel::with_color("Min", [0.0, 0.0, 0.0, 0.0]);
        let min_rgba = min_label.rgba_u8().unwrap();
        assert_eq!(min_rgba, [0, 0, 0, 0]);

        // Max values
        let max_label = DebugLabel::with_color("Max", [1.0, 1.0, 1.0, 1.0]);
        let max_rgba = max_label.rgba_u8().unwrap();
        assert_eq!(max_rgba, [255, 255, 255, 255]);
    }

    #[test]
    fn color_just_below_one() {
        let label = DebugLabel::with_color("Near Max", [0.999, 0.999, 0.999, 0.999]);
        let rgba = label.rgba_u8().unwrap();
        // 0.999 * 255 = 254.745, rounds to 254
        assert!(rgba[0] >= 254);
    }

    // -------------------------------------------------------------------------
    // Stress Tests
    // -------------------------------------------------------------------------

    #[test]
    fn many_push_pop_cycles() {
        let mut stack = DebugMarkerStack::new();

        for i in 0..1000 {
            stack.push_group(DebugLabel::new(format!("Cycle {}", i)));
            stack.insert_marker(&DebugLabel::new(format!("Marker {}", i)));
            stack.pop_group();
        }

        assert!(stack.is_empty());
    }

    #[test]
    fn rapid_profiling_timestamps() {
        let mut stack = DebugMarkerStack::with_profiling();

        for i in 0..100 {
            stack.push_group(DebugLabel::new(format!("Fast {}", i)));
            let group = stack.pop_group().unwrap();
            // Should complete without timing issues
            assert!(group.elapsed().is_some());
        }
    }

    #[test]
    fn alternating_profiling_mode() {
        let mut stack = DebugMarkerStack::new();

        for i in 0..50 {
            stack.set_profiling(i % 2 == 0);
            stack.push_group(DebugLabel::new(format!("Alternate {}", i)));
            let group = stack.pop_group().unwrap();

            if i % 2 == 0 {
                assert!(group.has_profiling());
            } else {
                assert!(!group.has_profiling());
            }
        }
    }

    // -------------------------------------------------------------------------
    // Memory/Clone Safety
    // -------------------------------------------------------------------------

    #[test]
    fn cloned_label_is_independent() {
        let original = DebugLabel::new(String::from("Mutable"));
        let cloned = original.clone();

        // Modifying the source string before clone won't affect it
        assert_eq!(original.as_wgpu_label(), cloned.as_wgpu_label());
    }

    #[test]
    fn cloned_stack_is_independent() {
        let mut original = DebugMarkerStack::new();
        original.push_group(DebugLabel::new("Original"));

        let cloned = original.clone();

        original.push_group(DebugLabel::new("Only In Original"));

        assert_eq!(original.current_depth(), 2);
        assert_eq!(cloned.current_depth(), 1);
    }
}

// =============================================================================
// MODULE: Profiling Scenarios
// =============================================================================

mod profiling_scenarios {
    use super::*;

    #[test]
    fn profiling_overhead_is_minimal() {
        // Non-profiled push/pop
        let start_non_profiled = std::time::Instant::now();
        let mut stack = DebugMarkerStack::new();
        for _ in 0..1000 {
            stack.push_group(DebugLabel::new("Test"));
            stack.pop_group();
        }
        let non_profiled_duration = start_non_profiled.elapsed();

        // Profiled push/pop
        let start_profiled = std::time::Instant::now();
        let mut stack = DebugMarkerStack::with_profiling();
        for _ in 0..1000 {
            stack.push_group(DebugLabel::new("Test"));
            stack.pop_group();
        }
        let profiled_duration = start_profiled.elapsed();

        // Profiling should not add more than 10x overhead (very generous)
        assert!(
            profiled_duration < non_profiled_duration * 10,
            "Profiling overhead too high: {:?} vs {:?}",
            profiled_duration,
            non_profiled_duration
        );
    }

    #[test]
    fn elapsed_time_accumulates_correctly() {
        let group = DebugGroup::with_profiling(DebugLabel::new("Accumulator"));

        let t1 = group.elapsed_ms().unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let t2 = group.elapsed_ms().unwrap();

        assert!(t2 > t1, "Time should increase: {} <= {}", t2, t1);
        assert!(t2 - t1 >= 5.0, "Should show ~5ms increase");
    }

    #[test]
    fn profiling_works_at_all_depths() {
        let mut stack = DebugMarkerStack::with_profiling();

        // Push to max depth
        for i in 0..16 {
            stack.push_group(DebugLabel::new(format!("Depth {}", i)));
        }

        // Pop all and verify each had profiling
        for _ in 0..16 {
            let group = stack.pop_group().unwrap();
            assert!(group.has_profiling());
            assert!(group.elapsed_ms().unwrap() >= 0.0);
        }
    }

    #[test]
    fn nested_timing_is_hierarchical() {
        let mut stack = DebugMarkerStack::with_profiling();

        stack.push_group(DebugLabel::new("Outer"));
        std::thread::sleep(Duration::from_millis(5));

        stack.push_group(DebugLabel::new("Inner"));
        std::thread::sleep(Duration::from_millis(5));
        let inner = stack.pop_group().unwrap();

        std::thread::sleep(Duration::from_millis(5));
        let outer = stack.pop_group().unwrap();

        // Inner should be ~5ms, Outer should be ~15ms
        assert!(inner.elapsed_ms().unwrap() >= 5.0);
        assert!(outer.elapsed_ms().unwrap() >= 15.0);
        assert!(outer.elapsed_ms().unwrap() > inner.elapsed_ms().unwrap());
    }
}

// =============================================================================
// MODULE: DebugContextOps Trait
// =============================================================================

mod debug_context_ops {
    use super::*;

    // Since RenderPassDebugContext, ComputePassDebugContext, and
    // CommandEncoderDebugContext require wgpu types which need GPU,
    // we test the trait implementation through DebugMarkerStack instead
    // which provides the same logical operations.

    #[test]
    fn marker_stack_as_debug_context_substitute() {
        // DebugMarkerStack provides the same operations as DebugContextOps
        let mut stack = DebugMarkerStack::new();

        // push_group (analogous to push_debug_group)
        assert!(stack.push_group(DebugLabel::new("Group 1")));
        assert_eq!(stack.current_depth(), 1);

        // pop_group (analogous to pop_debug_group)
        let group = stack.pop_group();
        assert!(group.is_some());
        assert_eq!(stack.current_depth(), 0);

        // insert_marker (analogous to insert_debug_marker)
        stack.insert_marker(&DebugLabel::new("Marker"));
        assert_eq!(stack.markers_at_depth(), 1);

        // has_active_groups
        assert!(stack.is_empty()); // No active groups
        stack.push_group(DebugLabel::new("Active"));
        assert!(!stack.is_empty()); // Now has active group
    }
}

// =============================================================================
// MODULE: Concurrent Access (Stack is not thread-safe, but Clone is)
// =============================================================================

mod concurrent_patterns {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn label_can_be_shared_across_threads() {
        let label = Arc::new(DebugLabel::with_color("Shared Label", colors::GEOMETRY));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let label = Arc::clone(&label);
                thread::spawn(move || {
                    assert_eq!(label.as_wgpu_label(), "Shared Label");
                    assert!(label.has_color());
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn each_thread_gets_own_stack() {
        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    let mut stack = DebugMarkerStack::new();
                    stack.push_group(DebugLabel::new(format!("Thread {}", i)));
                    assert_eq!(stack.current_depth(), 1);
                    stack.pop_group();
                    assert!(stack.is_empty());
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn cloned_stack_per_thread() {
        let base_stack = DebugMarkerStack::with_profiling();

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let mut stack = base_stack.clone();
                thread::spawn(move || {
                    stack.push_group(DebugLabel::new(format!("Thread {} Group", i)));
                    stack.insert_marker(&DebugLabel::new(format!("Marker {}", i)));
                    let group = stack.pop_group().unwrap();
                    assert!(group.has_profiling());
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

// =============================================================================
// MODULE: Integration Patterns
// =============================================================================

mod integration_patterns {
    use super::*;

    #[test]
    fn builder_pattern_for_complex_label() {
        // Demonstrate building complex labels step by step
        let base = DebugLabel::with_color("Base Pass", colors::GEOMETRY);
        let specialized = base.child("Specialized");

        assert_eq!(specialized.as_wgpu_label(), "Base Pass/Specialized");
        assert_eq!(specialized.color, Some(colors::GEOMETRY));
    }

    #[test]
    fn frame_timing_summary() {
        let mut stack = DebugMarkerStack::with_profiling();

        // Simulate frame timing collection
        stack.push_group(DebugLabel::new("Frame"));
        std::thread::sleep(Duration::from_millis(1));

        let passes = ["Shadow", "GBuffer", "Lighting", "Post"];
        let mut timings = Vec::new();

        for pass in &passes {
            stack.push_group(DebugLabel::new(*pass));
            std::thread::sleep(Duration::from_millis(1));
            let group = stack.pop_group().unwrap();
            timings.push((pass.to_string(), group.elapsed_ms().unwrap()));
        }

        let frame_group = stack.pop_group().unwrap();

        // Verify we collected timing data
        assert_eq!(timings.len(), 4);
        for (name, ms) in &timings {
            assert!(
                *ms >= 1.0,
                "Pass {} took only {}ms, expected >= 1ms",
                name,
                ms
            );
        }

        // Frame time should be sum of all passes plus overhead
        let total_pass_time: f64 = timings.iter().map(|(_, ms)| ms).sum();
        assert!(frame_group.elapsed_ms().unwrap() >= total_pass_time);
    }

    #[test]
    fn conditional_debug_groups() {
        let debug_enabled = true;
        let mut stack = DebugMarkerStack::new();

        if debug_enabled {
            stack.push_group(DebugLabel::new("Debug Only"));
        }

        // Work happens here...

        if debug_enabled && !stack.is_empty() {
            stack.pop_group();
        }

        assert!(stack.is_empty());
    }

    #[test]
    fn error_recovery_with_clear() {
        let mut stack = DebugMarkerStack::new();

        // Normal operation
        stack.push_group(DebugLabel::new("A"));
        stack.push_group(DebugLabel::new("B"));
        stack.push_group(DebugLabel::new("C"));

        // Simulated error - need to abort and reset
        // In real code, this would be used when an error occurs mid-frame
        stack.clear();

        assert!(stack.is_empty());
        assert_eq!(stack.current_depth(), 0);

        // Can continue with new frame
        stack.push_group(DebugLabel::new("New Frame"));
        assert_eq!(stack.current_depth(), 1);
    }
}

// =============================================================================
// MODULE: API Surface Completeness
// =============================================================================

mod api_completeness {
    use super::*;

    // Ensure all public API methods are exercised

    #[test]
    fn debug_label_api_coverage() {
        // Construction
        let _ = DebugLabel::new("owned");
        let _ = DebugLabel::new_static("static");
        let _ = DebugLabel::with_color("colored", [1.0, 0.0, 0.0, 1.0]);
        let _ = DebugLabel::with_static_color("static colored", [0.0, 1.0, 0.0, 1.0]);
        let _ = DebugLabel::default();

        // Accessors
        let label = DebugLabel::with_color("test", [0.5, 0.5, 0.5, 0.5]);
        let _ = label.as_wgpu_label();
        let _ = label.has_color();
        let _ = label.rgb();
        let _ = label.rgba_u8();
        let _ = label.child("suffix");

        // Traits
        let _ = format!("{}", label);
        let _ = format!("{:?}", label);
        let cloned = label.clone();
        assert_eq!(label, cloned);

        // From impls
        let _: DebugLabel = "str".into();
        let _: DebugLabel = String::from("String").into();
    }

    #[test]
    fn debug_group_api_coverage() {
        // Construction
        let _ = DebugGroup::new(DebugLabel::new("basic"));
        let _ = DebugGroup::with_depth(DebugLabel::new("depth"), 5);
        let _ = DebugGroup::with_profiling(DebugLabel::new("profiled"));
        let _ = DebugGroup::with_depth_and_profiling(DebugLabel::new("both"), 3);

        // Accessors
        let group = DebugGroup::with_profiling(DebugLabel::new("test"));
        let _ = group.elapsed();
        let _ = group.elapsed_ms();
        let _ = group.has_profiling();
        let _ = group.label.as_wgpu_label();
        let _ = group.depth;
        let _ = group.start_time;

        // Clone
        let cloned = group.clone();
        assert_eq!(cloned.label.as_wgpu_label(), "test");
    }

    #[test]
    fn debug_marker_stack_api_coverage() {
        // Construction
        let _ = DebugMarkerStack::new();
        let _ = DebugMarkerStack::with_max_depth(8);
        let _ = DebugMarkerStack::with_profiling();
        let _: DebugMarkerStack = Default::default();

        // Mutation
        let mut stack = DebugMarkerStack::new();
        stack.set_profiling(true);
        let _ = stack.push_group(DebugLabel::new("group"));
        stack.insert_marker(&DebugLabel::new("marker"));
        let _ = stack.pop_group();
        stack.clear();

        // Accessors
        let stack = DebugMarkerStack::with_profiling();
        let _ = stack.current_depth();
        let _ = stack.is_empty();
        let _ = stack.max_depth();
        let _ = stack.current_group();
        let _ = stack.markers_at_depth();
        let _ = stack.groups();
        let _ = stack.can_push();
        let _ = stack.path();
        let _ = stack.profiling_enabled();

        // Clone
        let cloned = stack.clone();
        assert!(cloned.is_empty());
    }

    #[test]
    fn colors_module_coverage() {
        // All predefined colors
        let _ = colors::GEOMETRY;
        let _ = colors::SHADOW;
        let _ = colors::LIGHTING;
        let _ = colors::POST_PROCESS;
        let _ = colors::COMPUTE;
        let _ = colors::UI;
        let _ = colors::DEBUG;
        let _ = colors::TRANSPARENT;
        let _ = colors::RAYTRACING;
        let _ = colors::TRANSFER;
    }
}

// =============================================================================
// MODULE: Documentation Examples
// =============================================================================

mod documentation_examples {
    use super::*;

    // These tests verify the examples from the module documentation work

    #[test]
    fn example_debug_label_creation() {
        // Simple label
        let label = DebugLabel::new("GBuffer Pass");
        assert_eq!(label.as_wgpu_label(), "GBuffer Pass");

        // Label with color (RGBA)
        let colored = DebugLabel::with_color("Shadow Pass", [0.2, 0.4, 0.8, 1.0]);
        assert!(colored.has_color());

        // Static label (no allocation)
        let static_label = DebugLabel::new_static("Lighting Pass");
        assert!(matches!(&static_label.name, Cow::Borrowed(_)));
    }

    #[test]
    fn example_debug_group_creation() {
        // Basic group
        let group = DebugGroup::new(DebugLabel::new("My Pass"));
        assert!(!group.has_profiling());

        // Group with profiling enabled
        let profiled = DebugGroup::with_profiling(DebugLabel::new("Timed Pass"));
        assert!(profiled.start_time.is_some());
    }

    #[test]
    fn example_marker_stack_usage() {
        let mut stack = DebugMarkerStack::new();

        // Push groups
        stack.push_group(DebugLabel::new("Outer"));
        assert_eq!(stack.current_depth(), 1);

        stack.push_group(DebugLabel::new("Inner"));
        assert_eq!(stack.current_depth(), 2);

        // Pop groups
        let inner = stack.pop_group();
        assert!(inner.is_some());
        assert_eq!(stack.current_depth(), 1);

        let outer = stack.pop_group();
        assert!(outer.is_some());
        assert!(stack.is_empty());
    }

    #[test]
    fn example_child_labels() {
        let parent = DebugLabel::new("Shadow Pass");
        let child = parent.child("Cascade 0");
        assert_eq!(child.as_wgpu_label(), "Shadow Pass/Cascade 0");
    }

    #[test]
    fn example_path_hierarchy() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("Shadows"));
        stack.push_group(DebugLabel::new("Cascade0"));
        assert_eq!(stack.path(), "Shadows/Cascade0");
    }
}
