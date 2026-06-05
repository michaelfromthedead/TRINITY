//! Blackbox tests for T-WGPU-P4.7.1 — Resource State Tracking
//!
//! CLEANROOM: Tests written against public API only.
//! Tests cover:
//! 1. PipelineStage enum (vertex, fragment, compute, transfer, etc.)
//! 2. AccessFlags bitflags (read, write, combinations)
//! 3. TextureLayout enum (8+ states)
//! 4. HashMap<ResourceId, ResourceState> for tracking
//! 5. State query methods

use renderer_backend::resource_state::{
    AccessFlags, PipelineStage, ResourceId, ResourceState, ResourceStateTracker, TextureLayout,
};

// =============================================================================
// CRITERION 1: PipelineStage enum
// =============================================================================

mod pipeline_stage_tests {
    use super::*;

    #[test]
    fn test_pipeline_stage_variants_exist() {
        // Verify all expected pipeline stages exist
        let stages = [
            PipelineStage::None,
            PipelineStage::VertexInput,
            PipelineStage::VertexShader,
            PipelineStage::FragmentShader,
            PipelineStage::EarlyDepth,
            PipelineStage::LateDepth,
            PipelineStage::ColorOutput,
            PipelineStage::ComputeShader,
            PipelineStage::Transfer,
            PipelineStage::Host,
            PipelineStage::AllGraphics,
            PipelineStage::AllCommands,
        ];

        // Just accessing them verifies they exist
        assert!(stages.len() >= 10, "Expected at least 10 pipeline stages");
    }

    #[test]
    fn test_pipeline_stage_is_graphics() {
        // Graphics stages should return true
        assert!(PipelineStage::VertexInput.is_graphics());
        assert!(PipelineStage::VertexShader.is_graphics());
        assert!(PipelineStage::FragmentShader.is_graphics());
        assert!(PipelineStage::ColorOutput.is_graphics());
        assert!(PipelineStage::AllGraphics.is_graphics());

        // Non-graphics stages should return false
        assert!(!PipelineStage::None.is_graphics());
        assert!(!PipelineStage::ComputeShader.is_graphics());
        assert!(!PipelineStage::Transfer.is_graphics());
        assert!(!PipelineStage::Host.is_graphics());
    }

    #[test]
    fn test_pipeline_stage_is_compute() {
        assert!(PipelineStage::ComputeShader.is_compute());
        assert!(!PipelineStage::VertexShader.is_compute());
        assert!(!PipelineStage::FragmentShader.is_compute());
        assert!(!PipelineStage::Transfer.is_compute());
    }

    #[test]
    fn test_pipeline_stage_is_transfer() {
        assert!(PipelineStage::Transfer.is_transfer());
        assert!(!PipelineStage::VertexShader.is_transfer());
        assert!(!PipelineStage::ComputeShader.is_transfer());
    }

    #[test]
    fn test_pipeline_stage_is_shader_stage() {
        // Shader stages
        assert!(PipelineStage::VertexShader.is_shader_stage());
        assert!(PipelineStage::FragmentShader.is_shader_stage());
        assert!(PipelineStage::ComputeShader.is_shader_stage());

        // Non-shader stages
        assert!(!PipelineStage::None.is_shader_stage());
        assert!(!PipelineStage::Transfer.is_shader_stage());
        assert!(!PipelineStage::Host.is_shader_stage());
        assert!(!PipelineStage::VertexInput.is_shader_stage());
    }

    #[test]
    fn test_pipeline_stage_order_index() {
        // Verify stages have logical ordering
        let none_idx = PipelineStage::None.order_index();
        let vertex_input_idx = PipelineStage::VertexInput.order_index();
        let vertex_shader_idx = PipelineStage::VertexShader.order_index();
        let fragment_shader_idx = PipelineStage::FragmentShader.order_index();
        let color_output_idx = PipelineStage::ColorOutput.order_index();

        // Graphics pipeline should follow logical order
        assert!(none_idx <= vertex_input_idx);
        assert!(vertex_input_idx <= vertex_shader_idx);
        assert!(vertex_shader_idx <= fragment_shader_idx);
        assert!(fragment_shader_idx <= color_output_idx);
    }

    #[test]
    fn test_pipeline_stage_comes_before() {
        // VertexInput comes before VertexShader
        assert!(PipelineStage::VertexInput.comes_before(&PipelineStage::VertexShader));

        // VertexShader comes before FragmentShader
        assert!(PipelineStage::VertexShader.comes_before(&PipelineStage::FragmentShader));

        // FragmentShader does not come before VertexShader
        assert!(!PipelineStage::FragmentShader.comes_before(&PipelineStage::VertexShader));

        // Same stage does not come before itself
        assert!(!PipelineStage::VertexShader.comes_before(&PipelineStage::VertexShader));
    }

    #[test]
    fn test_pipeline_stage_clone_and_eq() {
        let stage = PipelineStage::FragmentShader;
        let cloned = stage.clone();
        assert_eq!(stage, cloned);
        assert_ne!(PipelineStage::VertexShader, PipelineStage::FragmentShader);
    }

    #[test]
    fn test_pipeline_stage_debug() {
        // Verify Debug is implemented
        let debug_str = format!("{:?}", PipelineStage::VertexShader);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_pipeline_stage_default() {
        let default_stage = PipelineStage::default();
        assert_eq!(default_stage, PipelineStage::None);
    }
}

// =============================================================================
// CRITERION 2: AccessFlags bitflags
// =============================================================================

mod access_flags_tests {
    use super::*;

    #[test]
    fn test_access_flags_basic_flags() {
        // Verify fundamental access flags exist
        let _ = AccessFlags::READ;
        let _ = AccessFlags::WRITE;
        let _ = AccessFlags::SHADER_READ;
        let _ = AccessFlags::SHADER_WRITE;
        let _ = AccessFlags::TRANSFER_READ;
        let _ = AccessFlags::TRANSFER_WRITE;
        let _ = AccessFlags::HOST_READ;
        let _ = AccessFlags::HOST_WRITE;
    }

    #[test]
    fn test_access_flags_combinations() {
        // Test bitwise OR combinations
        let read_write = AccessFlags::READ | AccessFlags::WRITE;
        assert!(read_write.contains(AccessFlags::READ));
        assert!(read_write.contains(AccessFlags::WRITE));

        let shader_all = AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE;
        assert!(shader_all.contains(AccessFlags::SHADER_READ));
        assert!(shader_all.contains(AccessFlags::SHADER_WRITE));
    }

    #[test]
    fn test_access_flags_has_read() {
        assert!(AccessFlags::READ.has_read());
        assert!(AccessFlags::SHADER_READ.has_read());
        assert!(AccessFlags::TRANSFER_READ.has_read());
        assert!(AccessFlags::HOST_READ.has_read());

        // Combined flags with read
        let combined = AccessFlags::READ | AccessFlags::WRITE;
        assert!(combined.has_read());
    }

    #[test]
    fn test_access_flags_has_write() {
        assert!(AccessFlags::WRITE.has_write());
        assert!(AccessFlags::SHADER_WRITE.has_write());
        assert!(AccessFlags::TRANSFER_WRITE.has_write());
        assert!(AccessFlags::HOST_WRITE.has_write());

        // Combined flags with write
        let combined = AccessFlags::READ | AccessFlags::WRITE;
        assert!(combined.has_write());
    }

    #[test]
    fn test_access_flags_is_read_only() {
        assert!(AccessFlags::READ.is_read_only());
        assert!(AccessFlags::SHADER_READ.is_read_only());

        // Not read-only if write is included
        let read_write = AccessFlags::READ | AccessFlags::WRITE;
        assert!(!read_write.is_read_only());

        assert!(!AccessFlags::WRITE.is_read_only());
    }

    #[test]
    fn test_access_flags_is_write_only() {
        assert!(AccessFlags::WRITE.is_write_only());
        assert!(AccessFlags::SHADER_WRITE.is_write_only());

        // Not write-only if read is included
        let read_write = AccessFlags::READ | AccessFlags::WRITE;
        assert!(!read_write.is_write_only());

        assert!(!AccessFlags::READ.is_write_only());
    }

    #[test]
    fn test_access_flags_conflicts_with() {
        // Write conflicts with read (potential hazard)
        assert!(AccessFlags::WRITE.conflicts_with(AccessFlags::READ));
        assert!(AccessFlags::READ.conflicts_with(AccessFlags::WRITE));

        // Write conflicts with write
        assert!(AccessFlags::WRITE.conflicts_with(AccessFlags::WRITE));

        // Read does not conflict with read (multiple readers allowed)
        assert!(!AccessFlags::READ.conflicts_with(AccessFlags::READ));
    }

    #[test]
    fn test_access_flags_requires_barrier_to() {
        // Write to read requires barrier
        assert!(AccessFlags::WRITE.requires_barrier_to(AccessFlags::READ));

        // Write to write requires barrier
        assert!(AccessFlags::WRITE.requires_barrier_to(AccessFlags::WRITE));

        // Read to write requires barrier
        assert!(AccessFlags::READ.requires_barrier_to(AccessFlags::WRITE));
    }

    #[test]
    fn test_access_flags_empty() {
        let empty = AccessFlags::empty();
        assert!(empty.is_empty());
        assert!(!empty.has_read());
        assert!(!empty.has_write());
    }

    #[test]
    fn test_access_flags_all() {
        let all = AccessFlags::all();
        assert!(!all.is_empty());
        assert!(all.has_read());
        assert!(all.has_write());
    }

    #[test]
    fn test_access_flags_bitwise_operations() {
        let a = AccessFlags::READ | AccessFlags::SHADER_READ;
        let b = AccessFlags::WRITE | AccessFlags::SHADER_WRITE;

        // Union
        let union = a | b;
        assert!(union.contains(AccessFlags::READ));
        assert!(union.contains(AccessFlags::WRITE));
        assert!(union.contains(AccessFlags::SHADER_READ));
        assert!(union.contains(AccessFlags::SHADER_WRITE));

        // Intersection
        let c = AccessFlags::READ | AccessFlags::WRITE;
        let d = AccessFlags::READ | AccessFlags::SHADER_READ;
        let intersection = c & d;
        assert!(intersection.contains(AccessFlags::READ));
        assert!(!intersection.contains(AccessFlags::WRITE));

        // Difference
        let diff = c - d;
        assert!(diff.contains(AccessFlags::WRITE));
        assert!(!diff.contains(AccessFlags::READ));
    }

    #[test]
    fn test_access_flags_debug() {
        let flags = AccessFlags::READ | AccessFlags::WRITE;
        let debug_str = format!("{:?}", flags);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_access_flags_clone_and_eq() {
        let flags = AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE;
        let cloned = flags.clone();
        assert_eq!(flags, cloned);
    }
}

// =============================================================================
// CRITERION 3: TextureLayout enum (8+ states)
// =============================================================================

mod texture_layout_tests {
    use super::*;

    #[test]
    fn test_texture_layout_has_at_least_8_states() {
        // Verify at least 8 texture layout states exist
        let layouts = [
            TextureLayout::Undefined,
            TextureLayout::General,
            TextureLayout::ColorAttachment,
            TextureLayout::DepthStencilAttachment,
            TextureLayout::DepthStencilReadOnly,
            TextureLayout::ShaderReadOnly,
            TextureLayout::TransferSrc,
            TextureLayout::TransferDst,
            TextureLayout::Present,
        ];

        assert!(layouts.len() >= 8, "Expected at least 8 texture layout states");
    }

    #[test]
    fn test_texture_layout_supports_shader_read() {
        assert!(TextureLayout::General.supports_shader_read());
        assert!(TextureLayout::ShaderReadOnly.supports_shader_read());
        assert!(TextureLayout::DepthStencilReadOnly.supports_shader_read());

        // Typically transfer layouts don't support shader read
        assert!(!TextureLayout::TransferDst.supports_shader_read());
    }

    #[test]
    fn test_texture_layout_supports_shader_write() {
        assert!(TextureLayout::General.supports_shader_write());

        // Read-only layouts should not support write
        assert!(!TextureLayout::ShaderReadOnly.supports_shader_write());
        assert!(!TextureLayout::DepthStencilReadOnly.supports_shader_write());
    }

    #[test]
    fn test_texture_layout_supports_color_attachment() {
        assert!(TextureLayout::ColorAttachment.supports_color_attachment());
        assert!(TextureLayout::General.supports_color_attachment());

        assert!(!TextureLayout::DepthStencilAttachment.supports_color_attachment());
        assert!(!TextureLayout::ShaderReadOnly.supports_color_attachment());
    }

    #[test]
    fn test_texture_layout_supports_depth_stencil() {
        assert!(TextureLayout::DepthStencilAttachment.supports_depth_stencil());
        assert!(TextureLayout::DepthStencilReadOnly.supports_depth_stencil());

        assert!(!TextureLayout::ColorAttachment.supports_depth_stencil());
        assert!(!TextureLayout::ShaderReadOnly.supports_depth_stencil());
    }

    #[test]
    fn test_texture_layout_supports_transfer_read() {
        assert!(TextureLayout::TransferSrc.supports_transfer_read());
        assert!(TextureLayout::General.supports_transfer_read());

        assert!(!TextureLayout::TransferDst.supports_transfer_read());
    }

    #[test]
    fn test_texture_layout_supports_transfer_write() {
        assert!(TextureLayout::TransferDst.supports_transfer_write());
        assert!(TextureLayout::General.supports_transfer_write());

        assert!(!TextureLayout::TransferSrc.supports_transfer_write());
        assert!(!TextureLayout::ShaderReadOnly.supports_transfer_write());
    }

    #[test]
    fn test_texture_layout_requires_transition_to() {
        // Different layouts require transition
        assert!(TextureLayout::Undefined.requires_transition_to(TextureLayout::ShaderReadOnly));
        assert!(TextureLayout::ColorAttachment.requires_transition_to(TextureLayout::ShaderReadOnly));

        // Same layout does not require transition
        assert!(!TextureLayout::General.requires_transition_to(TextureLayout::General));
        assert!(!TextureLayout::ShaderReadOnly.requires_transition_to(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_texture_layout_optimal_for_access() {
        // Shader read should suggest shader read-only layout
        let layout = TextureLayout::optimal_for_access(AccessFlags::SHADER_READ);
        assert!(layout.supports_shader_read());

        // Transfer write should suggest transfer dst layout
        let layout = TextureLayout::optimal_for_access(AccessFlags::TRANSFER_WRITE);
        assert!(layout.supports_transfer_write());

        // Transfer read should suggest transfer src layout
        let layout = TextureLayout::optimal_for_access(AccessFlags::TRANSFER_READ);
        assert!(layout.supports_transfer_read());
    }

    #[test]
    fn test_texture_layout_clone_and_eq() {
        let layout = TextureLayout::ColorAttachment;
        let cloned = layout.clone();
        assert_eq!(layout, cloned);
        assert_ne!(TextureLayout::ColorAttachment, TextureLayout::DepthStencilAttachment);
    }

    #[test]
    fn test_texture_layout_debug() {
        let debug_str = format!("{:?}", TextureLayout::Present);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_texture_layout_default() {
        let default_layout = TextureLayout::default();
        assert_eq!(default_layout, TextureLayout::Undefined);
    }
}

// =============================================================================
// CRITERION 4: HashMap<ResourceId, ResourceState> tracking
// =============================================================================

mod resource_state_tracker_tests {
    use super::*;

    #[test]
    fn test_tracker_new() {
        let tracker = ResourceStateTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_tracker_with_capacity() {
        let tracker = ResourceStateTracker::with_capacity(100);
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_tracker_set_and_get() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 42;

        let state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ);
        tracker.set(id, state);

        let retrieved = tracker.get(id);
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.stage, PipelineStage::VertexShader);
        assert_eq!(retrieved.access, AccessFlags::READ);
    }

    #[test]
    fn test_tracker_get_nonexistent() {
        let tracker = ResourceStateTracker::new();
        assert!(tracker.get(999).is_none());
    }

    #[test]
    fn test_tracker_update() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 1;

        // Set initial state
        tracker.set(id, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));

        // Update to new stage and access
        tracker.update(id, PipelineStage::FragmentShader, AccessFlags::SHADER_READ);

        let state = tracker.get(id).unwrap();
        assert_eq!(state.stage, PipelineStage::FragmentShader);
        assert_eq!(state.access, AccessFlags::SHADER_READ);
    }

    #[test]
    fn test_tracker_update_nonexistent_creates_state() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 100;

        // Update on nonexistent resource should create it
        tracker.update(id, PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE);

        assert!(tracker.contains(id));
        let state = tracker.get(id).unwrap();
        assert_eq!(state.stage, PipelineStage::ComputeShader);
        assert_eq!(state.access, AccessFlags::SHADER_WRITE);
    }

    #[test]
    fn test_tracker_update_layout() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 2;

        // Set initial texture state
        tracker.set(
            id,
            ResourceState::texture(
                PipelineStage::FragmentShader,
                AccessFlags::SHADER_READ,
                TextureLayout::Undefined,
            ),
        );

        // Update layout
        tracker.update_layout(id, TextureLayout::ShaderReadOnly);

        let state = tracker.get(id).unwrap();
        assert_eq!(state.layout, Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_tracker_remove() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 3;

        tracker.set(id, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));

        let removed = tracker.remove(id);
        assert!(removed.is_some());
        assert!(!tracker.contains(id));
        assert!(tracker.get(id).is_none());
    }

    #[test]
    fn test_tracker_remove_nonexistent() {
        let mut tracker = ResourceStateTracker::new();
        let removed = tracker.remove(999);
        assert!(removed.is_none());
    }

    #[test]
    fn test_tracker_clear() {
        let mut tracker = ResourceStateTracker::new();

        tracker.set(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        tracker.set(2, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::READ));
        tracker.set(3, ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::WRITE));

        assert_eq!(tracker.len(), 3);

        tracker.clear();

        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_tracker_len_and_is_empty() {
        let mut tracker = ResourceStateTracker::new();

        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);

        tracker.set(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        assert!(!tracker.is_empty());
        assert_eq!(tracker.len(), 1);

        tracker.set(2, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::READ));
        assert_eq!(tracker.len(), 2);

        tracker.remove(1);
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn test_tracker_contains() {
        let mut tracker = ResourceStateTracker::new();

        assert!(!tracker.contains(1));

        tracker.set(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        assert!(tracker.contains(1));
        assert!(!tracker.contains(2));
    }

    #[test]
    fn test_tracker_ids_iterator() {
        let mut tracker = ResourceStateTracker::new();

        tracker.set(10, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        tracker.set(20, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::READ));
        tracker.set(30, ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::WRITE));

        let ids: Vec<ResourceId> = tracker.ids().copied().collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&10));
        assert!(ids.contains(&20));
        assert!(ids.contains(&30));
    }

    #[test]
    fn test_tracker_states_iterator() {
        let mut tracker = ResourceStateTracker::new();

        tracker.set(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        tracker.set(2, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::SHADER_READ));

        let states: Vec<_> = tracker.states().collect();
        assert_eq!(states.len(), 2);

        for (id, state) in states {
            if *id == 1 {
                assert_eq!(state.stage, PipelineStage::VertexShader);
            } else if *id == 2 {
                assert_eq!(state.stage, PipelineStage::FragmentShader);
            }
        }
    }

    #[test]
    fn test_tracker_states_mut_iterator() {
        let mut tracker = ResourceStateTracker::new();

        tracker.set(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        tracker.set(2, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::READ));

        // Modify all states via mutable iterator
        for (_id, state) in tracker.states_mut() {
            state.access = AccessFlags::WRITE;
        }

        // Verify modifications
        for (_id, state) in tracker.states() {
            assert_eq!(state.access, AccessFlags::WRITE);
        }
    }

    #[test]
    fn test_tracker_multiple_resources() {
        let mut tracker = ResourceStateTracker::new();

        // Add many resources
        for i in 0..100 {
            tracker.set(
                i,
                ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE),
            );
        }

        assert_eq!(tracker.len(), 100);

        // Verify all can be retrieved
        for i in 0..100 {
            assert!(tracker.contains(i));
            assert!(tracker.get(i).is_some());
        }
    }

    #[test]
    fn test_tracker_overwrite_existing() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 42;

        tracker.set(id, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));

        // Overwrite with different state
        tracker.set(id, ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::WRITE));

        assert_eq!(tracker.len(), 1); // Still only one entry
        let state = tracker.get(id).unwrap();
        assert_eq!(state.stage, PipelineStage::ComputeShader);
        assert_eq!(state.access, AccessFlags::WRITE);
    }

    #[test]
    fn test_tracker_transition() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 1;

        // Set initial state
        tracker.set(
            id,
            ResourceState::texture(
                PipelineStage::Transfer,
                AccessFlags::TRANSFER_WRITE,
                TextureLayout::TransferDst,
            ),
        );

        // Transition to shader read state
        let target = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );

        // transition returns Some((old, new)) if barrier needed, None otherwise
        let barrier_info = tracker.transition(id, target);
        assert!(barrier_info.is_some(), "Expected barrier to be needed");

        // Verify new state
        let state = tracker.get(id).unwrap();
        assert_eq!(state.stage, PipelineStage::FragmentShader);
        assert_eq!(state.access, AccessFlags::SHADER_READ);
        assert_eq!(state.layout, Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_tracker_merge() {
        let mut tracker1 = ResourceStateTracker::new();
        let mut tracker2 = ResourceStateTracker::new();

        tracker1.set(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        tracker1.set(2, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::READ));

        tracker2.set(3, ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::WRITE));
        tracker2.set(4, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_READ));

        tracker1.merge(&tracker2);

        assert_eq!(tracker1.len(), 4);
        assert!(tracker1.contains(1));
        assert!(tracker1.contains(2));
        assert!(tracker1.contains(3));
        assert!(tracker1.contains(4));
    }

    #[test]
    fn test_tracker_merge_overwrites() {
        let mut tracker1 = ResourceStateTracker::new();
        let mut tracker2 = ResourceStateTracker::new();

        tracker1.set(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        tracker2.set(1, ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::WRITE));

        tracker1.merge(&tracker2);

        // tracker2's state should overwrite tracker1's
        let state = tracker1.get(1).unwrap();
        assert_eq!(state.stage, PipelineStage::ComputeShader);
        assert_eq!(state.access, AccessFlags::WRITE);
    }
}

// =============================================================================
// CRITERION 5: State query methods
// =============================================================================

mod resource_state_tests {
    use super::*;

    #[test]
    fn test_resource_state_buffer_constructor() {
        let state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ);

        assert_eq!(state.stage, PipelineStage::VertexShader);
        assert_eq!(state.access, AccessFlags::READ);
        assert!(state.layout.is_none());
        assert!(state.is_buffer());
        assert!(!state.is_texture());
    }

    #[test]
    fn test_resource_state_texture_constructor() {
        let state = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );

        assert_eq!(state.stage, PipelineStage::FragmentShader);
        assert_eq!(state.access, AccessFlags::SHADER_READ);
        assert_eq!(state.layout, Some(TextureLayout::ShaderReadOnly));
        assert!(!state.is_buffer());
        assert!(state.is_texture());
    }

    #[test]
    fn test_resource_state_undefined() {
        let state = ResourceState::undefined();

        assert_eq!(state.stage, PipelineStage::None);
        assert!(state.access.is_empty());
    }

    #[test]
    fn test_resource_state_is_buffer() {
        let buffer = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ);
        assert!(buffer.is_buffer());

        let texture = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::General,
        );
        assert!(!texture.is_buffer());
    }

    #[test]
    fn test_resource_state_is_texture() {
        let texture = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::General,
        );
        assert!(texture.is_texture());

        let buffer = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ);
        assert!(!buffer.is_texture());
    }

    #[test]
    fn test_resource_state_requires_barrier_to_different_stage() {
        let src = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::WRITE);
        let dst = ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::READ);

        assert!(src.requires_barrier_to(&dst));
    }

    #[test]
    fn test_resource_state_requires_barrier_write_to_read() {
        let src = ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE);
        let dst = ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_READ);

        assert!(src.requires_barrier_to(&dst));
    }

    #[test]
    fn test_resource_state_requires_barrier_layout_change() {
        let src = ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        );
        let dst = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );

        assert!(src.requires_barrier_to(&dst));
    }

    #[test]
    fn test_resource_state_clone_and_eq() {
        let state = ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::WRITE);
        let cloned = state.clone();

        assert_eq!(state.stage, cloned.stage);
        assert_eq!(state.access, cloned.access);
        assert_eq!(state.layout, cloned.layout);
    }

    #[test]
    fn test_resource_state_debug() {
        let state = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        let debug_str = format!("{:?}", state);
        assert!(!debug_str.is_empty());
    }
}

// =============================================================================
// EDGE CASES
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_resource_id_zero() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 0;

        tracker.set(id, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        assert!(tracker.contains(id));
        assert!(tracker.get(id).is_some());
    }

    #[test]
    fn test_resource_id_max() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = u64::MAX;

        tracker.set(id, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        assert!(tracker.contains(id));
        assert!(tracker.get(id).is_some());
    }

    #[test]
    fn test_empty_access_flags() {
        let empty = AccessFlags::empty();
        assert!(empty.is_empty());
        assert!(!empty.has_read());
        assert!(!empty.has_write());
        assert!(!empty.is_read_only()); // Empty is neither read-only nor write-only
        assert!(!empty.is_write_only());
    }

    #[test]
    fn test_all_access_flags_combined() {
        let all = AccessFlags::all();
        assert!(all.has_read());
        assert!(all.has_write());
        assert!(!all.is_read_only());
        assert!(!all.is_write_only());
    }

    #[test]
    fn test_texture_layout_undefined_transitions() {
        // Undefined should require transition to any other layout
        assert!(TextureLayout::Undefined.requires_transition_to(TextureLayout::General));
        assert!(TextureLayout::Undefined.requires_transition_to(TextureLayout::ColorAttachment));
        assert!(TextureLayout::Undefined.requires_transition_to(TextureLayout::ShaderReadOnly));

        // But not to itself
        assert!(!TextureLayout::Undefined.requires_transition_to(TextureLayout::Undefined));
    }

    #[test]
    fn test_pipeline_stage_all_commands_properties() {
        // AllCommands should have high order index
        let all_idx = PipelineStage::AllCommands.order_index();
        let vertex_idx = PipelineStage::VertexShader.order_index();
        let compute_idx = PipelineStage::ComputeShader.order_index();

        assert!(all_idx >= vertex_idx);
        assert!(all_idx >= compute_idx);
    }

    #[test]
    fn test_tracker_remove_then_add_same_id() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 42;

        tracker.set(id, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        tracker.remove(id);
        tracker.set(id, ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::WRITE));

        let state = tracker.get(id).unwrap();
        assert_eq!(state.stage, PipelineStage::ComputeShader);
        assert_eq!(state.access, AccessFlags::WRITE);
    }

    #[test]
    fn test_tracker_clear_then_add() {
        let mut tracker = ResourceStateTracker::new();

        tracker.set(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));
        tracker.set(2, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::READ));
        tracker.clear();

        assert!(tracker.is_empty());

        tracker.set(3, ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::WRITE));
        assert_eq!(tracker.len(), 1);
        assert!(tracker.contains(3));
        assert!(!tracker.contains(1));
        assert!(!tracker.contains(2));
    }

    #[test]
    fn test_complex_bitflag_operations() {
        // Test complex combinations
        let a = AccessFlags::READ | AccessFlags::SHADER_READ | AccessFlags::HOST_READ;
        let b = AccessFlags::WRITE | AccessFlags::SHADER_WRITE | AccessFlags::HOST_WRITE;

        let union = a | b;
        assert!(union.contains(AccessFlags::READ));
        assert!(union.contains(AccessFlags::WRITE));
        assert!(union.contains(AccessFlags::SHADER_READ));
        assert!(union.contains(AccessFlags::SHADER_WRITE));
        assert!(union.contains(AccessFlags::HOST_READ));
        assert!(union.contains(AccessFlags::HOST_WRITE));

        // Toggle
        let toggled = a ^ AccessFlags::READ;
        assert!(!toggled.contains(AccessFlags::READ));
        assert!(toggled.contains(AccessFlags::SHADER_READ));
        assert!(toggled.contains(AccessFlags::HOST_READ));
    }

    #[test]
    fn test_transition_same_state() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 1;

        let state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ);
        tracker.set(id, state.clone());

        // Transition to same state
        let barrier_info = tracker.transition(id, state);

        // No barrier needed for same state - returns None
        assert!(barrier_info.is_none(), "No barrier should be needed for same state");
    }

    #[test]
    fn test_merge_empty_trackers() {
        let mut tracker1 = ResourceStateTracker::new();
        let tracker2 = ResourceStateTracker::new();

        tracker1.merge(&tracker2);
        assert!(tracker1.is_empty());
    }

    #[test]
    fn test_merge_into_empty_tracker() {
        let mut tracker1 = ResourceStateTracker::new();
        let mut tracker2 = ResourceStateTracker::new();

        tracker2.set(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));

        tracker1.merge(&tracker2);
        assert_eq!(tracker1.len(), 1);
        assert!(tracker1.contains(1));
    }

    #[test]
    fn test_update_layout_on_buffer_state() {
        let mut tracker = ResourceStateTracker::new();
        let id: ResourceId = 1;

        // Set as buffer (no layout)
        tracker.set(id, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::READ));

        // Update layout - should add layout to buffer state
        tracker.update_layout(id, TextureLayout::General);

        let state = tracker.get(id).unwrap();
        assert_eq!(state.layout, Some(TextureLayout::General));
    }

    #[test]
    fn test_concurrent_read_access_no_conflict() {
        let read1 = AccessFlags::SHADER_READ;
        let read2 = AccessFlags::SHADER_READ;

        // Multiple reads should not conflict
        assert!(!read1.conflicts_with(read2));
    }

    #[test]
    fn test_texture_layout_present() {
        // Present layout is for swapchain presentation
        let present = TextureLayout::Present;

        // Present typically doesn't support shader operations
        assert!(!present.supports_shader_write());
        assert!(!present.supports_transfer_write());
    }
}
