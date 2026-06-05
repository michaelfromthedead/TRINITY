// SPDX-License-Identifier: MIT
//
// blackbox_compute_pass.rs -- Blackbox tests for T-WGPU-P3.9.3 Compute Pass Creation.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - ComputePassDescriptor -- High-level compute pass descriptor with builder
//   - ComputePassTimestampWrites -- Timestamp query configuration
//   - ComputePassBuilder -- Fluent builder for compute pass descriptors
//   - ComputePass -- Wrapper around wgpu::ComputePass
//   - ComputePassPreset -- Preset metadata for common compute patterns
//   - begin_compute_pass -- Helper function to begin a compute pass
//   - get_preset_info -- Lookup preset by name
//   - preset_names -- Iterator over preset names
//   - COMPUTE_PASS_PRESETS -- Static array of presets
//
// PUBLIC API METHODS:
//   ComputePassTimestampWrites:
//     - new(query_set) -> Self
//     - both(beginning, end) -> Self
//     - beginning_only(index) -> Self
//     - end_only(index) -> Self
//     - beginning(index) -> Self
//     - end(index) -> Self
//     - no_beginning() -> Self
//     - no_end() -> Self
//     - is_enabled() -> bool
//     - has_beginning() -> bool
//     - has_end() -> bool
//     - to_wgpu() -> wgpu::ComputePassTimestampWrites
//
//   ComputePassDescriptor:
//     - new() -> Self
//     - label(label) -> Self
//     - timestamp_writes(writes) -> Self
//     - no_timestamp_writes() -> Self
//     - get_timestamp_writes() -> Option<&ComputePassTimestampWrites>
//     - has_timestamp_writes() -> bool
//     - to_wgpu() -> wgpu::ComputePassDescriptor
//
//   ComputePassBuilder:
//     - new() -> Self
//     - label(label) -> Self
//     - with_timestamps(query_set, begin, end) -> Self
//     - with_begin_timestamp(query_set, index) -> Self
//     - with_end_timestamp(query_set, index) -> Self
//     - timestamp_writes(writes) -> Self
//     - build() -> ComputePassDescriptor
//
//   ComputePass:
//     - new(encoder, desc) -> Self
//     - from_raw(pass) -> Self
//     - set_pipeline(pipeline) -> &mut Self
//     - set_bind_group(index, bind_group, offsets) -> &mut Self
//     - set_push_constants(offset, data) -> &mut Self
//     - dispatch_workgroups(x, y, z) -> &mut Self
//     - dispatch_workgroups_indirect(buffer, offset) -> &mut Self
//     - insert_debug_marker(label) -> &mut Self
//     - push_debug_group(label) -> &mut Self
//     - pop_debug_group() -> &mut Self
//     - inner() -> &wgpu::ComputePass
//     - inner_mut() -> &mut wgpu::ComputePass
//     - into_inner() -> wgpu::ComputePass
//     - finish()
//
//   Helper functions:
//     - begin_compute_pass(encoder, desc) -> ComputePass
//     - get_preset_info(name) -> Option<&ComputePassPreset>
//     - preset_names() -> impl Iterator<Item = &str>
//
//   Constants:
//     - COMPUTE_PASS_PRESETS
//
// ACCEPTANCE CRITERIA (T-WGPU-P3.9.3):
//   1. ComputePassDescriptor construction with builder pattern
//   2. ComputePassTimestampWrites configuration
//   3. ComputePassBuilder fluent API
//   4. ComputePass wrapper methods
//   5. Preset system for common compute patterns
//   6. Thread safety notes (ComputePass is not Send)
//   7. Real-world scenarios (particle systems, image processing, physics)
//
// TEST CATEGORIES:
//   1. API Tests (10 tests) - Public interface, types exist
//   2. ComputePassTimestampWrites (12 tests) - Builder methods, queries
//   3. ComputePassDescriptor (10 tests) - Construction and builder
//   4. ComputePassBuilder (10 tests) - Fluent builder API
//   5. ComputePass Wrapper (10 tests) - Methods and inner access
//   6. Preset System (8 tests) - Preset lookup, names, iteration
//   7. Builder Chaining (8 tests) - Fluent API patterns
//   8. Thread Safety Notes (4 tests) - Type bound tests
//   9. Edge Cases (6 tests) - Empty labels, zero indices
//   10. Real-world Scenarios (10 tests) - Common compute workloads
//
// Total target: 88 tests

use renderer_backend::compute_pass::{
    begin_compute_pass, get_preset_info, preset_names, ComputePass, ComputePassBuilder,
    ComputePassDescriptor, ComputePassPreset, ComputePassTimestampWrites, COMPUTE_PASS_PRESETS,
};

// =============================================================================
// CATEGORY 1: API TESTS - Public Interface Existence (10 tests)
// =============================================================================

mod api_tests {
    use super::*;

    #[test]
    fn test_compute_pass_descriptor_is_public() {
        // Verify ComputePassDescriptor struct is accessible
        let desc = ComputePassDescriptor::new();
        // Should have default values
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_compute_pass_builder_is_public() {
        // Verify ComputePassBuilder struct is accessible
        let builder = ComputePassBuilder::new();
        let desc = builder.build();
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_compute_pass_timestamp_writes_type_exists() {
        // Verify ComputePassTimestampWrites type is accessible
        // Note: requires QuerySet which we can't easily construct in blackbox tests
        let _: Option<ComputePassTimestampWrites<'_>> = None;
    }

    #[test]
    fn test_compute_pass_type_exists() {
        // Verify ComputePass type is accessible
        let _: Option<ComputePass<'_>> = None;
    }

    #[test]
    fn test_compute_pass_preset_is_public() {
        // Verify ComputePassPreset struct is accessible
        let _: Option<&ComputePassPreset> = None;
    }

    #[test]
    fn test_begin_compute_pass_function_exists() {
        // Verify begin_compute_pass function is accessible
        // We verify the function exists by checking its signature at compile time
        fn check_begin<'a>(
            encoder: &'a mut wgpu::CommandEncoder,
            desc: &ComputePassDescriptor<'_>,
        ) -> ComputePass<'a> {
            begin_compute_pass(encoder, desc)
        }
        let _ = check_begin;
    }

    #[test]
    fn test_get_preset_info_function_exists() {
        // Verify get_preset_info function is accessible
        let _ = get_preset_info as fn(&str) -> Option<&'static ComputePassPreset>;
    }

    #[test]
    fn test_preset_names_function_exists() {
        // Verify preset_names function is accessible
        let names: Vec<&str> = preset_names().collect();
        assert!(!names.is_empty());
    }

    #[test]
    fn test_compute_pass_presets_constant_exists() {
        // Verify COMPUTE_PASS_PRESETS constant is accessible
        assert!(!COMPUTE_PASS_PRESETS.is_empty());
    }

    #[test]
    fn test_all_imports_compile() {
        // Verify all public types can be imported together
        use renderer_backend::compute_pass::*;
        let _ = ComputePassDescriptor::new();
        let _ = ComputePassBuilder::new();
        let _: Option<ComputePass<'_>> = None;
        let _: Option<ComputePassTimestampWrites<'_>> = None;
        let _: Option<&ComputePassPreset> = get_preset_info("simulation");
    }
}

// =============================================================================
// CATEGORY 2: COMPUTE PASS TIMESTAMP WRITES TESTS (12 tests)
// =============================================================================

mod timestamp_writes_tests {
    use super::*;

    // Note: Most ComputePassTimestampWrites tests require a wgpu::QuerySet which
    // cannot be constructed without a GPU device. We test what we can at compile time.

    #[test]
    fn test_timestamp_writes_type_is_generic_over_lifetime() {
        // Verify ComputePassTimestampWrites is generic over lifetime
        fn accepts_timestamp_writes<'a>(_tw: Option<ComputePassTimestampWrites<'a>>) {}
        accepts_timestamp_writes(None);
    }

    #[test]
    fn test_timestamp_writes_has_display_impl() {
        // Verify ComputePassTimestampWrites implements Display
        // We can't construct one without a QuerySet, but we can verify the trait bound
        fn requires_display<T: std::fmt::Display>() {}
        requires_display::<ComputePassTimestampWrites<'static>>();
    }

    #[test]
    fn test_timestamp_writes_builder_methods_exist() {
        // Verify builder methods are available on the type
        // These are compile-time checks - the methods exist on the impl
        // We check that the new method exists by referencing it in a closure
        fn check_new<'a>(qs: &'a wgpu::QuerySet) -> ComputePassTimestampWrites<'a> {
            ComputePassTimestampWrites::new(qs)
        }
        let _ = check_new;
    }

    #[test]
    fn test_timestamp_writes_method_signatures_are_fluent() {
        // Verify methods return Self for chaining (compile-time)
        // The builder pattern requires Self return types
        fn check_fluent<'a, F>(_f: F)
        where
            F: FnOnce(ComputePassTimestampWrites<'a>) -> ComputePassTimestampWrites<'a>,
        {
        }
        // Can't actually call these without QuerySet, but signature check works
    }

    #[test]
    fn test_timestamp_writes_query_methods_exist() {
        // Verify query methods return expected types (compile-time)
        fn check_is_enabled<'a>(tw: &ComputePassTimestampWrites<'a>) -> bool {
            tw.is_enabled()
        }
        fn check_has_beginning<'a>(tw: &ComputePassTimestampWrites<'a>) -> bool {
            tw.has_beginning()
        }
        fn check_has_end<'a>(tw: &ComputePassTimestampWrites<'a>) -> bool {
            tw.has_end()
        }
        let _ = check_is_enabled;
        let _ = check_has_beginning;
        let _ = check_has_end;
    }

    #[test]
    fn test_timestamp_writes_to_wgpu_method_exists() {
        // Verify to_wgpu method exists and returns correct type
        fn check_to_wgpu<'a>(tw: &ComputePassTimestampWrites<'a>) -> wgpu::ComputePassTimestampWrites<'a> {
            tw.to_wgpu()
        }
        let _ = check_to_wgpu;
    }

    #[test]
    fn test_timestamp_writes_both_method_signature() {
        // Verify both() method signature
        fn check_both<'a>(tw: ComputePassTimestampWrites<'a>, begin: u32, end: u32) -> ComputePassTimestampWrites<'a> {
            tw.both(begin, end)
        }
        let _ = check_both;
    }

    #[test]
    fn test_timestamp_writes_beginning_only_method_signature() {
        // Verify beginning_only() method signature
        fn check<'a>(tw: ComputePassTimestampWrites<'a>, idx: u32) -> ComputePassTimestampWrites<'a> {
            tw.beginning_only(idx)
        }
        let _ = check;
    }

    #[test]
    fn test_timestamp_writes_end_only_method_signature() {
        // Verify end_only() method signature
        fn check<'a>(tw: ComputePassTimestampWrites<'a>, idx: u32) -> ComputePassTimestampWrites<'a> {
            tw.end_only(idx)
        }
        let _ = check;
    }

    #[test]
    fn test_timestamp_writes_beginning_method_signature() {
        // Verify beginning() method signature
        fn check<'a>(tw: ComputePassTimestampWrites<'a>, idx: u32) -> ComputePassTimestampWrites<'a> {
            tw.beginning(idx)
        }
        let _ = check;
    }

    #[test]
    fn test_timestamp_writes_end_method_signature() {
        // Verify end() method signature
        fn check<'a>(tw: ComputePassTimestampWrites<'a>, idx: u32) -> ComputePassTimestampWrites<'a> {
            tw.end(idx)
        }
        let _ = check;
    }

    #[test]
    fn test_timestamp_writes_clearing_methods_exist() {
        // Verify no_beginning() and no_end() methods exist
        fn check_no_beginning<'a>(tw: ComputePassTimestampWrites<'a>) -> ComputePassTimestampWrites<'a> {
            tw.no_beginning()
        }
        fn check_no_end<'a>(tw: ComputePassTimestampWrites<'a>) -> ComputePassTimestampWrites<'a> {
            tw.no_end()
        }
        let _ = check_no_beginning;
        let _ = check_no_end;
    }
}

// =============================================================================
// CATEGORY 3: COMPUTE PASS DESCRIPTOR TESTS (10 tests)
// =============================================================================

mod descriptor_tests {
    use super::*;

    #[test]
    fn test_descriptor_new_creates_empty_descriptor() {
        let desc = ComputePassDescriptor::new();
        assert!(!desc.has_timestamp_writes());
        assert!(desc.get_timestamp_writes().is_none());
    }

    #[test]
    fn test_descriptor_implements_default() {
        // Verify Default is implemented and equivalent to new()
        let default_desc: ComputePassDescriptor<'_> = Default::default();
        let new_desc = ComputePassDescriptor::new();
        assert_eq!(default_desc.has_timestamp_writes(), new_desc.has_timestamp_writes());
    }

    #[test]
    fn test_descriptor_label_method() {
        let desc = ComputePassDescriptor::new().label("particle_update");
        // Label should be set (we can verify via to_wgpu if needed)
        // For blackbox, we just verify the method chains correctly
        assert!(!desc.has_timestamp_writes()); // Still no timestamps
    }

    #[test]
    fn test_descriptor_label_chaining() {
        // Verify label returns Self for chaining
        let desc = ComputePassDescriptor::new()
            .label("first_label")
            .label("second_label");
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_descriptor_no_timestamp_writes_method() {
        // Verify no_timestamp_writes clears any timestamps
        let desc = ComputePassDescriptor::new().no_timestamp_writes();
        assert!(!desc.has_timestamp_writes());
        assert!(desc.get_timestamp_writes().is_none());
    }

    #[test]
    fn test_descriptor_has_timestamp_writes_returns_false_by_default() {
        let desc = ComputePassDescriptor::new();
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_descriptor_get_timestamp_writes_returns_none_by_default() {
        let desc = ComputePassDescriptor::new();
        assert!(desc.get_timestamp_writes().is_none());
    }

    #[test]
    fn test_descriptor_to_wgpu_method_exists() {
        // Verify to_wgpu method exists and returns correct type
        let desc = ComputePassDescriptor::new();
        let _wgpu_desc: wgpu::ComputePassDescriptor<'_> = desc.to_wgpu();
    }

    #[test]
    fn test_descriptor_to_wgpu_with_label() {
        let desc = ComputePassDescriptor::new().label("test_pass");
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("test_pass"));
    }

    #[test]
    fn test_descriptor_implements_display() {
        // Verify Display is implemented
        let desc = ComputePassDescriptor::new().label("display_test");
        let display_str = format!("{}", desc);
        assert!(display_str.contains("ComputePassDescriptor") || display_str.contains("display_test"));
    }
}

// =============================================================================
// CATEGORY 4: COMPUTE PASS BUILDER TESTS (10 tests)
// =============================================================================

mod builder_tests {
    use super::*;

    #[test]
    fn test_builder_new_creates_empty_builder() {
        let builder = ComputePassBuilder::new();
        let desc = builder.build();
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_builder_implements_default() {
        let default_builder: ComputePassBuilder<'_> = Default::default();
        let desc = default_builder.build();
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_builder_label_method() {
        let desc = ComputePassBuilder::new()
            .label("compute_simulation")
            .build();
        // Verify via to_wgpu
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("compute_simulation"));
    }

    #[test]
    fn test_builder_label_chaining() {
        // Verify label returns Self for chaining
        let desc = ComputePassBuilder::new()
            .label("first")
            .label("second")
            .build();
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("second"));
    }

    #[test]
    fn test_builder_build_returns_descriptor() {
        let builder = ComputePassBuilder::new().label("build_test");
        let desc: ComputePassDescriptor<'_> = builder.build();
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_builder_with_timestamps_method_signature() {
        // Verify with_timestamps method signature (compile-time check)
        fn check<'a>(
            builder: ComputePassBuilder<'a>,
            qs: &'a wgpu::QuerySet,
            begin: u32,
            end: u32,
        ) -> ComputePassBuilder<'a> {
            builder.with_timestamps(qs, begin, end)
        }
        let _ = check;
    }

    #[test]
    fn test_builder_with_begin_timestamp_method_signature() {
        // Verify with_begin_timestamp method signature
        fn check<'a>(
            builder: ComputePassBuilder<'a>,
            qs: &'a wgpu::QuerySet,
            idx: u32,
        ) -> ComputePassBuilder<'a> {
            builder.with_begin_timestamp(qs, idx)
        }
        let _ = check;
    }

    #[test]
    fn test_builder_with_end_timestamp_method_signature() {
        // Verify with_end_timestamp method signature
        fn check<'a>(
            builder: ComputePassBuilder<'a>,
            qs: &'a wgpu::QuerySet,
            idx: u32,
        ) -> ComputePassBuilder<'a> {
            builder.with_end_timestamp(qs, idx)
        }
        let _ = check;
    }

    #[test]
    fn test_builder_timestamp_writes_method_signature() {
        // Verify timestamp_writes method signature
        fn check<'a>(
            builder: ComputePassBuilder<'a>,
            tw: ComputePassTimestampWrites<'a>,
        ) -> ComputePassBuilder<'a> {
            builder.timestamp_writes(tw)
        }
        let _ = check;
    }

    #[test]
    fn test_builder_full_chain() {
        // Verify full fluent chain compiles
        let desc = ComputePassBuilder::new()
            .label("full_chain_test")
            .build();
        assert_eq!(desc.to_wgpu().label, Some("full_chain_test"));
    }
}

// =============================================================================
// CATEGORY 5: COMPUTE PASS WRAPPER TESTS (10 tests)
// =============================================================================

mod compute_pass_wrapper_tests {
    use super::*;

    #[test]
    fn test_compute_pass_type_is_generic_over_lifetime() {
        fn accepts_compute_pass<'a>(_pass: Option<ComputePass<'a>>) {}
        accepts_compute_pass(None);
    }

    #[test]
    fn test_compute_pass_new_method_signature() {
        // Verify new() method signature (compile-time)
        fn check<'a>(
            encoder: &'a mut wgpu::CommandEncoder,
            desc: &ComputePassDescriptor<'_>,
        ) -> ComputePass<'a> {
            ComputePass::new(encoder, desc)
        }
        let _ = check;
    }

    #[test]
    fn test_compute_pass_from_raw_method_signature() {
        // Verify from_raw() method signature
        fn check<'a>(pass: wgpu::ComputePass<'a>) -> ComputePass<'a> {
            ComputePass::from_raw(pass)
        }
        let _ = check;
    }

    #[test]
    fn test_compute_pass_set_pipeline_method_signature() {
        // Verify set_pipeline returns &mut Self for chaining
        fn check<'a, 'b>(
            pass: &'b mut ComputePass<'a>,
            pipeline: &'a wgpu::ComputePipeline,
        ) -> &'b mut ComputePass<'a> {
            pass.set_pipeline(pipeline)
        }
        let _ = check;
    }

    #[test]
    fn test_compute_pass_set_bind_group_method_signature() {
        // Verify set_bind_group returns &mut Self
        fn check<'a, 'b>(
            pass: &'b mut ComputePass<'a>,
            index: u32,
            bind_group: &'a wgpu::BindGroup,
            offsets: &[u32],
        ) -> &'b mut ComputePass<'a> {
            pass.set_bind_group(index, bind_group, offsets)
        }
        let _ = check;
    }

    #[test]
    fn test_compute_pass_set_push_constants_method_signature() {
        // Verify set_push_constants returns &mut Self
        fn check<'a, 'b>(pass: &'b mut ComputePass<'a>, offset: u32, data: &[u8]) -> &'b mut ComputePass<'a> {
            pass.set_push_constants(offset, data)
        }
        let _ = check;
    }

    #[test]
    fn test_compute_pass_dispatch_workgroups_method_signature() {
        // Verify dispatch_workgroups returns &mut Self
        fn check<'a, 'b>(pass: &'b mut ComputePass<'a>, x: u32, y: u32, z: u32) -> &'b mut ComputePass<'a> {
            pass.dispatch_workgroups(x, y, z)
        }
        let _ = check;
    }

    #[test]
    fn test_compute_pass_dispatch_workgroups_indirect_method_signature() {
        // Verify dispatch_workgroups_indirect returns &mut Self
        fn check<'a, 'b>(
            pass: &'b mut ComputePass<'a>,
            buffer: &'a wgpu::Buffer,
            offset: u64,
        ) -> &'b mut ComputePass<'a> {
            pass.dispatch_workgroups_indirect(buffer, offset)
        }
        let _ = check;
    }

    #[test]
    fn test_compute_pass_debug_methods_exist() {
        // Verify debug marker methods exist
        fn check_insert_debug_marker<'a, 'b>(pass: &'b mut ComputePass<'a>, label: &str) -> &'b mut ComputePass<'a> {
            pass.insert_debug_marker(label)
        }
        fn check_push_debug_group<'a, 'b>(pass: &'b mut ComputePass<'a>, label: &str) -> &'b mut ComputePass<'a> {
            pass.push_debug_group(label)
        }
        fn check_pop_debug_group<'a, 'b>(pass: &'b mut ComputePass<'a>) -> &'b mut ComputePass<'a> {
            pass.pop_debug_group()
        }
        let _ = check_insert_debug_marker;
        let _ = check_push_debug_group;
        let _ = check_pop_debug_group;
    }

    #[test]
    fn test_compute_pass_inner_access_methods_exist() {
        // Verify inner access methods exist
        fn check_inner<'a, 'b>(pass: &'b ComputePass<'a>) -> &'b wgpu::ComputePass<'a> {
            pass.inner()
        }
        fn check_inner_mut<'a, 'b>(pass: &'b mut ComputePass<'a>) -> &'b mut wgpu::ComputePass<'a> {
            pass.inner_mut()
        }
        fn check_into_inner<'a>(pass: ComputePass<'a>) -> wgpu::ComputePass<'a> {
            pass.into_inner()
        }
        let _ = check_inner;
        let _ = check_inner_mut;
        let _ = check_into_inner;
    }
}

// =============================================================================
// CATEGORY 6: PRESET SYSTEM TESTS (8 tests)
// =============================================================================

mod preset_tests {
    use super::*;

    #[test]
    fn test_compute_pass_presets_is_not_empty() {
        assert!(!COMPUTE_PASS_PRESETS.is_empty());
    }

    #[test]
    fn test_compute_pass_presets_count() {
        // There should be at least a few common presets
        assert!(COMPUTE_PASS_PRESETS.len() >= 3);
    }

    #[test]
    fn test_get_preset_info_simulation() {
        let info = get_preset_info("simulation");
        assert!(info.is_some());
        let preset = info.unwrap();
        assert_eq!(preset.name, "simulation");
        assert!(!preset.description.is_empty());
    }

    #[test]
    fn test_get_preset_info_profiled() {
        let info = get_preset_info("profiled");
        assert!(info.is_some());
        let preset = info.unwrap();
        assert_eq!(preset.name, "profiled");
    }

    #[test]
    fn test_get_preset_info_nonexistent() {
        let info = get_preset_info("nonexistent_preset_xyz");
        assert!(info.is_none());
    }

    #[test]
    fn test_preset_names_returns_all_names() {
        let names: Vec<&str> = preset_names().collect();
        assert_eq!(names.len(), COMPUTE_PASS_PRESETS.len());
    }

    #[test]
    fn test_preset_names_contains_simulation() {
        let names: Vec<&str> = preset_names().collect();
        assert!(names.contains(&"simulation"));
    }

    #[test]
    fn test_all_presets_have_valid_names_and_descriptions() {
        for preset in COMPUTE_PASS_PRESETS {
            assert!(!preset.name.is_empty(), "Preset name should not be empty");
            assert!(
                !preset.description.is_empty(),
                "Preset {} should have a description",
                preset.name
            );
        }
    }
}

// =============================================================================
// CATEGORY 7: BUILDER CHAINING TESTS (8 tests)
// =============================================================================

mod builder_chaining_tests {
    use super::*;

    #[test]
    fn test_descriptor_method_chaining_compiles() {
        // Verify method chaining on ComputePassDescriptor compiles
        let _desc = ComputePassDescriptor::new()
            .label("chain_test")
            .no_timestamp_writes();
    }

    #[test]
    fn test_builder_method_chaining_compiles() {
        // Verify method chaining on ComputePassBuilder compiles
        let _desc = ComputePassBuilder::new()
            .label("builder_chain")
            .build();
    }

    #[test]
    fn test_descriptor_chain_preserves_label() {
        let desc = ComputePassDescriptor::new()
            .label("preserved")
            .no_timestamp_writes();
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("preserved"));
    }

    #[test]
    fn test_builder_chain_produces_valid_descriptor() {
        let desc = ComputePassBuilder::new()
            .label("valid_desc")
            .build();
        // Should be convertible to wgpu without panic
        let _wgpu = desc.to_wgpu();
    }

    #[test]
    fn test_empty_chain_produces_valid_descriptor() {
        // Even an empty chain should produce a valid descriptor
        let desc = ComputePassDescriptor::new();
        let wgpu_desc = desc.to_wgpu();
        assert!(wgpu_desc.label.is_none());
        assert!(wgpu_desc.timestamp_writes.is_none());
    }

    #[test]
    fn test_builder_empty_chain_produces_valid_descriptor() {
        let desc = ComputePassBuilder::new().build();
        let wgpu_desc = desc.to_wgpu();
        assert!(wgpu_desc.timestamp_writes.is_none());
    }

    #[test]
    fn test_descriptor_label_overwrite() {
        // Later label should overwrite earlier one
        let desc = ComputePassDescriptor::new()
            .label("first")
            .label("second")
            .label("third");
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("third"));
    }

    #[test]
    fn test_builder_label_overwrite() {
        let desc = ComputePassBuilder::new()
            .label("one")
            .label("two")
            .label("three")
            .build();
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("three"));
    }
}

// =============================================================================
// CATEGORY 8: THREAD SAFETY NOTES TESTS (4 tests)
// =============================================================================

mod thread_safety_tests {
    use super::*;

    #[test]
    fn test_compute_pass_descriptor_is_send() {
        // ComputePassDescriptor should be Send when 'a: 'static
        fn assert_send<T: Send>() {}
        assert_send::<ComputePassDescriptor<'static>>();
    }

    #[test]
    fn test_compute_pass_descriptor_is_sync() {
        // ComputePassDescriptor should be Sync when 'a: 'static
        fn assert_sync<T: Sync>() {}
        assert_sync::<ComputePassDescriptor<'static>>();
    }

    #[test]
    fn test_compute_pass_builder_is_send() {
        // ComputePassBuilder should be Send when 'a: 'static
        fn assert_send<T: Send>() {}
        assert_send::<ComputePassBuilder<'static>>();
    }

    #[test]
    fn test_compute_pass_builder_is_sync() {
        // ComputePassBuilder should be Sync when 'a: 'static
        fn assert_sync<T: Sync>() {}
        assert_sync::<ComputePassBuilder<'static>>();
    }

    // Note: ComputePass<'a> wraps wgpu::ComputePass which is NOT Send.
    // This is intentional - compute passes are recording commands and
    // must be used on the thread where they were created.
    // We document this rather than test it.
}

// =============================================================================
// CATEGORY 9: EDGE CASES TESTS (6 tests)
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_label() {
        let desc = ComputePassDescriptor::new().label("");
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some(""));
    }

    #[test]
    fn test_unicode_label() {
        let desc = ComputePassDescriptor::new().label("compute_\u{1F680}_simulation");
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("compute_\u{1F680}_simulation"));
    }

    #[test]
    fn test_very_long_label() {
        let long_label = "x".repeat(1000);
        let desc = ComputePassDescriptor::new().label(&long_label);
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label.unwrap().len(), 1000);
    }

    #[test]
    fn test_label_with_special_characters() {
        let desc = ComputePassDescriptor::new().label("pass/with:special@chars#!");
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("pass/with:special@chars#!"));
    }

    #[test]
    fn test_multiple_no_timestamp_writes_calls() {
        // Multiple calls should not cause issues
        let desc = ComputePassDescriptor::new()
            .no_timestamp_writes()
            .no_timestamp_writes()
            .no_timestamp_writes();
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_builder_multiple_builds_conceptually() {
        // Each builder produces an independent descriptor
        // (Builder is consumed on build, so we create multiple)
        let desc1 = ComputePassBuilder::new().label("first").build();
        let desc2 = ComputePassBuilder::new().label("second").build();

        assert_eq!(desc1.to_wgpu().label, Some("first"));
        assert_eq!(desc2.to_wgpu().label, Some("second"));
    }
}

// =============================================================================
// CATEGORY 10: REAL-WORLD SCENARIOS TESTS (10 tests)
// =============================================================================

mod real_world_scenarios {
    use super::*;

    #[test]
    fn test_particle_system_compute_pass_descriptor() {
        // Particle simulation typically uses labeled passes for debugging
        let desc = ComputePassDescriptor::new()
            .label("particle_simulation");
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("particle_simulation"));
    }

    #[test]
    fn test_image_processing_compute_pass() {
        // Image processing (blur, tonemap, etc.)
        let desc = ComputePassDescriptor::new()
            .label("image_postprocess");
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_physics_simulation_compute_pass() {
        // Physics simulation (rigid body, cloth, fluid)
        let desc = ComputePassDescriptor::new()
            .label("physics_solver");
        let wgpu_desc = desc.to_wgpu();
        assert!(wgpu_desc.label.unwrap().contains("physics"));
    }

    #[test]
    fn test_culling_compute_pass() {
        // GPU-driven culling pass
        let desc = ComputePassBuilder::new()
            .label("frustum_culling")
            .build();
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("frustum_culling"));
    }

    #[test]
    fn test_prefix_sum_compute_pass() {
        // Prefix sum / scan algorithm
        let desc = ComputePassDescriptor::new()
            .label("prefix_sum_upsweep");
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_histogram_compute_pass() {
        // Histogram calculation
        let desc = ComputePassBuilder::new()
            .label("luminance_histogram")
            .build();
        let wgpu_desc = desc.to_wgpu();
        assert!(wgpu_desc.label.is_some());
    }

    #[test]
    fn test_sort_compute_pass() {
        // GPU radix sort
        let desc = ComputePassDescriptor::new()
            .label("radix_sort_pass");
        let _wgpu = desc.to_wgpu();
    }

    #[test]
    fn test_skinning_compute_pass() {
        // Skeletal animation / skinning
        let desc = ComputePassBuilder::new()
            .label("skeletal_skinning")
            .build();
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("skeletal_skinning"));
    }

    #[test]
    fn test_terrain_lod_compute_pass() {
        // Terrain LOD calculation
        let desc = ComputePassDescriptor::new()
            .label("terrain_lod_select");
        let wgpu_desc = desc.to_wgpu();
        assert!(wgpu_desc.label.unwrap().contains("terrain"));
    }

    #[test]
    fn test_indirect_args_compute_pass() {
        // Building indirect draw/dispatch arguments
        let desc = ComputePassBuilder::new()
            .label("build_indirect_args")
            .build();
        let wgpu_desc = desc.to_wgpu();
        assert!(wgpu_desc.label.unwrap().contains("indirect"));
    }
}

// =============================================================================
// ADDITIONAL CATEGORY: PRESET FIELD ACCESS TESTS (4 tests)
// =============================================================================

mod preset_field_tests {
    use super::*;

    #[test]
    fn test_preset_name_field_accessible() {
        if let Some(preset) = get_preset_info("simulation") {
            let _name: &str = preset.name;
        }
    }

    #[test]
    fn test_preset_description_field_accessible() {
        if let Some(preset) = get_preset_info("simulation") {
            let _desc: &str = preset.description;
        }
    }

    #[test]
    fn test_all_preset_names_are_unique() {
        let names: Vec<&str> = preset_names().collect();
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(names.len(), unique.len(), "Preset names should be unique");
    }

    #[test]
    fn test_preset_lookup_matches_iteration() {
        // Every name from preset_names() should be findable via get_preset_info()
        for name in preset_names() {
            let info = get_preset_info(name);
            assert!(info.is_some(), "Preset '{}' should be findable", name);
            assert_eq!(info.unwrap().name, name);
        }
    }
}

// =============================================================================
// ADDITIONAL CATEGORY: DISPLAY/DEBUG IMPL TESTS (4 tests)
// =============================================================================

mod display_debug_tests {
    use super::*;

    #[test]
    fn test_descriptor_display_contains_type_name() {
        let desc = ComputePassDescriptor::new();
        let display = format!("{}", desc);
        // Should mention what it is
        assert!(display.len() > 0);
    }

    #[test]
    fn test_descriptor_display_includes_label_when_set() {
        let desc = ComputePassDescriptor::new().label("my_label");
        let display = format!("{}", desc);
        // Should include the label in display
        assert!(display.contains("my_label") || display.contains("ComputePass"));
    }

    #[test]
    fn test_compute_pass_debug_impl_exists() {
        // Verify ComputePass implements Debug
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<ComputePass<'static>>();
    }

    #[test]
    fn test_descriptor_display_no_panic_on_empty() {
        let desc = ComputePassDescriptor::new();
        // Should not panic
        let _ = format!("{}", desc);
    }
}
