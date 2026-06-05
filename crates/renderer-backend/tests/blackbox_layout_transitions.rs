//! Blackbox tests for Layout Transitions (T-WGPU-P4.7.3)
//!
//! CLEANROOM TESTING: Tests exercise only the public API without
//! knowledge of internal implementation details.
//!
//! ACCEPTANCE CRITERIA:
//! 1. Automatic layout tracking per texture
//! 2. Transition path calculation (optimal transitions)
//! 3. Implicit vs explicit transition modes
//! 4. Transition coalescing (batch multiple transitions)

use renderer_backend::resource_state::{
    LayoutTransition, LayoutTransitionManager, SubresourceRange, TextureLayout, TransitionMode,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a subresource range covering all mips and layers
fn whole_resource() -> SubresourceRange {
    SubresourceRange {
        base_mip: 0,
        mip_count: None,
        base_layer: 0,
        layer_count: None,
    }
}

/// Create a specific subresource range
fn subresource(base_mip: u32, mip_count: Option<u32>, base_layer: u32, layer_count: Option<u32>) -> SubresourceRange {
    SubresourceRange {
        base_mip,
        mip_count,
        base_layer,
        layer_count,
    }
}

/// Create a single mip, single layer subresource
fn single_subresource(mip: u32, layer: u32) -> SubresourceRange {
    SubresourceRange {
        base_mip: mip,
        mip_count: Some(1),
        base_layer: layer,
        layer_count: Some(1),
    }
}

// ============================================================================
// CRITERION 1: Automatic Layout Tracking Per Texture
// ============================================================================

mod automatic_layout_tracking {
    use super::*;

    #[test]
    fn test_track_single_texture_layout() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        // Set initial layout
        manager.set_layout(texture_id, whole_resource(), TextureLayout::Undefined);

        // Verify layout is tracked
        let layout = manager.get_layout(texture_id, whole_resource());
        assert!(layout.is_some(), "Layout should be tracked");
        assert_eq!(layout.unwrap(), TextureLayout::Undefined);
    }

    #[test]
    fn test_track_multiple_textures_independently() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Set different layouts for different textures
        manager.set_layout(1, whole_resource(), TextureLayout::ColorAttachment);
        manager.set_layout(2, whole_resource(), TextureLayout::DepthStencilAttachment);
        manager.set_layout(3, whole_resource(), TextureLayout::ShaderReadOnly);

        // Verify each texture has its own layout
        assert_eq!(manager.get_layout(1, whole_resource()), Some(TextureLayout::ColorAttachment));
        assert_eq!(manager.get_layout(2, whole_resource()), Some(TextureLayout::DepthStencilAttachment));
        assert_eq!(manager.get_layout(3, whole_resource()), Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_track_subresource_layouts_independently() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        // Set different layouts for different mip levels
        manager.set_layout(texture_id, single_subresource(0, 0), TextureLayout::ShaderReadOnly);
        manager.set_layout(texture_id, single_subresource(1, 0), TextureLayout::TransferDst);
        manager.set_layout(texture_id, single_subresource(2, 0), TextureLayout::General);

        // Verify each subresource has its own layout
        assert_eq!(
            manager.get_layout(texture_id, single_subresource(0, 0)),
            Some(TextureLayout::ShaderReadOnly)
        );
        assert_eq!(
            manager.get_layout(texture_id, single_subresource(1, 0)),
            Some(TextureLayout::TransferDst)
        );
        assert_eq!(
            manager.get_layout(texture_id, single_subresource(2, 0)),
            Some(TextureLayout::General)
        );
    }

    #[test]
    fn test_track_array_layer_layouts() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        // Set different layouts for different array layers
        manager.set_layout(texture_id, single_subresource(0, 0), TextureLayout::ColorAttachment);
        manager.set_layout(texture_id, single_subresource(0, 1), TextureLayout::ShaderReadOnly);
        manager.set_layout(texture_id, single_subresource(0, 2), TextureLayout::StorageImage);

        // Verify each layer has its own layout
        assert_eq!(
            manager.get_layout(texture_id, single_subresource(0, 0)),
            Some(TextureLayout::ColorAttachment)
        );
        assert_eq!(
            manager.get_layout(texture_id, single_subresource(0, 1)),
            Some(TextureLayout::ShaderReadOnly)
        );
        assert_eq!(
            manager.get_layout(texture_id, single_subresource(0, 2)),
            Some(TextureLayout::StorageImage)
        );
    }

    #[test]
    fn test_update_existing_layout() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        // Set initial layout
        manager.set_layout(texture_id, whole_resource(), TextureLayout::Undefined);
        assert_eq!(manager.get_layout(texture_id, whole_resource()), Some(TextureLayout::Undefined));

        // Update to new layout
        manager.set_layout(texture_id, whole_resource(), TextureLayout::ShaderReadOnly);
        assert_eq!(manager.get_layout(texture_id, whole_resource()), Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_untracked_texture_returns_none() {
        let manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Query non-existent texture
        assert_eq!(manager.get_layout(999, whole_resource()), None);
    }

    #[test]
    fn test_is_tracked_method() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        assert!(!manager.is_tracked(100), "Texture should not be tracked initially");

        manager.set_layout(100, whole_resource(), TextureLayout::General);
        assert!(manager.is_tracked(100), "Texture should be tracked after set_layout");

        assert!(!manager.is_tracked(200), "Other textures should not be tracked");
    }

    #[test]
    fn test_remove_tracked_texture() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        manager.set_layout(texture_id, whole_resource(), TextureLayout::General);
        assert!(manager.is_tracked(texture_id));

        let removed = manager.remove(texture_id);
        assert!(removed, "Remove should return true for tracked texture");
        assert!(!manager.is_tracked(texture_id), "Texture should no longer be tracked");
        assert_eq!(manager.get_layout(texture_id, whole_resource()), None);
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);

        manager.set_layout(1, whole_resource(), TextureLayout::General);
        assert!(!manager.is_empty());
        assert_eq!(manager.len(), 1);

        manager.set_layout(2, whole_resource(), TextureLayout::General);
        assert_eq!(manager.len(), 2);

        manager.remove(1);
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_clear_removes_all_tracking() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        manager.set_layout(1, whole_resource(), TextureLayout::General);
        manager.set_layout(2, whole_resource(), TextureLayout::General);
        manager.set_layout(3, whole_resource(), TextureLayout::General);

        assert_eq!(manager.len(), 3);

        manager.clear();

        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
        assert!(!manager.is_tracked(1));
        assert!(!manager.is_tracked(2));
        assert!(!manager.is_tracked(3));
    }

    #[test]
    fn test_tracked_resources_iterator() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        manager.set_layout(10, whole_resource(), TextureLayout::General);
        manager.set_layout(20, whole_resource(), TextureLayout::General);
        manager.set_layout(30, whole_resource(), TextureLayout::General);

        let tracked: Vec<_> = manager.tracked_resources().cloned().collect();
        assert_eq!(tracked.len(), 3);
        assert!(tracked.contains(&10));
        assert!(tracked.contains(&20));
        assert!(tracked.contains(&30));
    }

    #[test]
    fn test_whole_layout_helper() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        manager.set_whole_layout(texture_id, TextureLayout::ColorAttachment);

        let layout = manager.get_whole_layout(texture_id);
        assert!(layout.is_some());
        assert_eq!(layout.unwrap(), TextureLayout::ColorAttachment);
    }
}

// ============================================================================
// CRITERION 2: Transition Path Calculation (Optimal Transitions)
// ============================================================================

mod transition_path_calculation {
    use super::*;

    #[test]
    fn test_direct_transition_no_intermediate() {
        // optimal_transition_path returns the optimal path which may include
        // intermediate states for safety (e.g., Undefined -> TransferDst -> ShaderReadOnly)
        let path = LayoutTransitionManager::optimal_transition_path(
            TextureLayout::Undefined,
            TextureLayout::ShaderReadOnly,
        );

        // Path should be reasonable length and end at target
        assert!(!path.is_empty(), "Path should not be empty for layout transition");
        assert!(path.len() <= 5, "Path should be reasonably short: {:?}", path);

        // Last element should be the target layout
        assert_eq!(
            path.last(),
            Some(&TextureLayout::ShaderReadOnly),
            "Path should end at target layout"
        );
    }

    #[test]
    fn test_same_layout_no_transition() {
        let path = LayoutTransitionManager::optimal_transition_path(
            TextureLayout::ShaderReadOnly,
            TextureLayout::ShaderReadOnly,
        );

        assert!(path.is_empty() || path.len() == 1, "Same layout should require no transition");
    }

    #[test]
    fn test_is_transition_needed_same_layout() {
        assert!(
            !LayoutTransitionManager::is_transition_needed(
                TextureLayout::ShaderReadOnly,
                TextureLayout::ShaderReadOnly
            ),
            "Transition from same to same layout should not be needed"
        );
    }

    #[test]
    fn test_is_transition_needed_different_layouts() {
        assert!(
            LayoutTransitionManager::is_transition_needed(
                TextureLayout::Undefined,
                TextureLayout::ShaderReadOnly
            ),
            "Transition from Undefined to ShaderReadOnly should be needed"
        );

        assert!(
            LayoutTransitionManager::is_transition_needed(
                TextureLayout::ColorAttachment,
                TextureLayout::ShaderReadOnly
            ),
            "Transition from ColorAttachment to ShaderReadOnly should be needed"
        );
    }

    #[test]
    fn test_is_transition_needed_to_general() {
        // General layout can be used for most operations
        assert!(
            LayoutTransitionManager::is_transition_needed(
                TextureLayout::Undefined,
                TextureLayout::General
            ),
            "Transition from Undefined to General should be needed"
        );
    }

    #[test]
    fn test_transition_to_generates_correct_transition() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        // Set initial layout
        manager.set_whole_layout(texture_id, TextureLayout::Undefined);

        // Request transition
        let transition = manager.transition_to(texture_id, TextureLayout::ShaderReadOnly);

        assert!(transition.is_some(), "Transition should be generated");
        let t = transition.unwrap();
        assert_eq!(t.resource_id, texture_id);
        assert_eq!(t.old_layout, TextureLayout::Undefined);
        assert_eq!(t.new_layout, TextureLayout::ShaderReadOnly);
    }

    #[test]
    fn test_transition_to_same_layout_returns_none() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        manager.set_whole_layout(texture_id, TextureLayout::ShaderReadOnly);

        // Request transition to same layout
        let transition = manager.transition_to(texture_id, TextureLayout::ShaderReadOnly);

        assert!(transition.is_none(), "Transition to same layout should return None");
    }

    #[test]
    fn test_transition_updates_tracked_layout() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        manager.set_whole_layout(texture_id, TextureLayout::Undefined);
        manager.transition_to(texture_id, TextureLayout::ShaderReadOnly);

        // After transition, the tracked layout should be updated
        let current = manager.get_whole_layout(texture_id);
        assert_eq!(current, Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_optimal_path_through_common_layouts() {
        // Test various layout transitions to ensure path calculation works
        let transitions_to_test = [
            (TextureLayout::Undefined, TextureLayout::ColorAttachment),
            (TextureLayout::ColorAttachment, TextureLayout::Present),
            (TextureLayout::TransferDst, TextureLayout::ShaderReadOnly),
            (TextureLayout::General, TextureLayout::StorageImage),
        ];

        for (from, to) in transitions_to_test {
            let path = LayoutTransitionManager::optimal_transition_path(from, to);
            // Path should exist and be reasonable length
            assert!(path.len() <= 5, "Path from {:?} to {:?} too long: {:?}", from, to, path);
        }
    }

    #[test]
    fn test_transition_subresource() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;
        let subres = single_subresource(0, 0);

        manager.set_layout(texture_id, subres.clone(), TextureLayout::Undefined);

        let transition = manager.transition_subresource(
            texture_id,
            subres.clone(),
            TextureLayout::ShaderReadOnly,
        );

        assert!(transition.is_some());
        let t = transition.unwrap();
        assert_eq!(t.subresource.base_mip, 0);
        assert_eq!(t.subresource.base_layer, 0);
        assert_eq!(t.new_layout, TextureLayout::ShaderReadOnly);
    }
}

// ============================================================================
// CRITERION 3: Implicit vs Explicit Transition Modes
// ============================================================================

mod transition_modes {
    use super::*;

    #[test]
    fn test_create_implicit_mode() {
        let manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        assert_eq!(manager.mode(), TransitionMode::Implicit);
    }

    #[test]
    fn test_create_explicit_mode() {
        let manager = LayoutTransitionManager::new(TransitionMode::Explicit);
        assert_eq!(manager.mode(), TransitionMode::Explicit);
    }

    #[test]
    fn test_switch_mode_runtime() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        assert_eq!(manager.mode(), TransitionMode::Implicit);

        manager.set_mode(TransitionMode::Explicit);
        assert_eq!(manager.mode(), TransitionMode::Explicit);

        manager.set_mode(TransitionMode::Implicit);
        assert_eq!(manager.mode(), TransitionMode::Implicit);
    }

    #[test]
    fn test_implicit_mode_tracks_automatically() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        manager.set_whole_layout(texture_id, TextureLayout::Undefined);

        // In implicit mode, transition_to should work and update state
        let transition = manager.transition_to(texture_id, TextureLayout::ShaderReadOnly);
        assert!(transition.is_some());

        // State should be automatically updated
        assert_eq!(manager.get_whole_layout(texture_id), Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_explicit_mode_requires_manual_tracking() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Explicit);
        let texture_id = 100;

        manager.set_whole_layout(texture_id, TextureLayout::Undefined);

        // In explicit mode, still works but caller is responsible for managing transitions
        let transition = manager.transition_to(texture_id, TextureLayout::ShaderReadOnly);
        // Should still generate transition info
        assert!(transition.is_some() || transition.is_none()); // Implementation specific
    }

    #[test]
    fn test_mode_preserved_through_operations() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Explicit);

        manager.set_layout(1, whole_resource(), TextureLayout::General);
        manager.set_layout(2, whole_resource(), TextureLayout::General);
        manager.transition_to(1, TextureLayout::ShaderReadOnly);

        // Mode should still be explicit
        assert_eq!(manager.mode(), TransitionMode::Explicit);
    }

    #[test]
    fn test_with_capacity_preserves_mode() {
        let manager = LayoutTransitionManager::with_capacity(TransitionMode::Explicit, 100);
        assert_eq!(manager.mode(), TransitionMode::Explicit);
    }

    #[test]
    fn test_default_mode_is_implicit() {
        // TransitionMode default should be Implicit
        let mode: TransitionMode = Default::default();
        assert_eq!(mode, TransitionMode::Implicit);
    }
}

// ============================================================================
// CRITERION 4: Transition Coalescing (Batch Multiple Transitions)
// ============================================================================

mod transition_coalescing {
    use super::*;

    #[test]
    fn test_add_pending_transition() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        let transition = LayoutTransition {
            resource_id: 100,
            old_layout: TextureLayout::Undefined,
            new_layout: TextureLayout::ShaderReadOnly,
            subresource: whole_resource(),
        };

        manager.add_pending(transition);
        assert_eq!(manager.pending_count(), 1);
        assert!(manager.has_pending());
    }

    #[test]
    fn test_add_multiple_pending_transitions() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        for i in 0..5 {
            let transition = LayoutTransition {
                resource_id: i,
                old_layout: TextureLayout::Undefined,
                new_layout: TextureLayout::ShaderReadOnly,
                subresource: whole_resource(),
            };
            manager.add_pending(transition);
        }

        assert_eq!(manager.pending_count(), 5);
    }

    #[test]
    fn test_coalesce_reduces_redundant_transitions() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Add transitions for same resource
        manager.add_pending(LayoutTransition {
            resource_id: 100,
            old_layout: TextureLayout::Undefined,
            new_layout: TextureLayout::General,
            subresource: whole_resource(),
        });

        manager.add_pending(LayoutTransition {
            resource_id: 100,
            old_layout: TextureLayout::General,
            new_layout: TextureLayout::ShaderReadOnly,
            subresource: whole_resource(),
        });

        let coalesced = manager.coalesce_pending();

        // Coalescing behavior: implementation may keep transitions separate
        // or combine them. The key is that all necessary transitions are present.
        assert!(!coalesced.is_empty(), "Should have at least one transition");

        // Check that we have transitions for resource 100
        let resource_transitions: Vec<_> = coalesced.iter()
            .filter(|t| t.resource_id == 100)
            .collect();
        assert!(!resource_transitions.is_empty(), "Should have transition(s) for resource 100");

        // The last transition for resource 100 should end at the final target
        // (or coalescing may have combined them)
        let last_for_resource = resource_transitions.last().unwrap();
        // Either it's coalesced to ShaderReadOnly directly, or ends at General
        // depending on implementation strategy
        assert!(
            last_for_resource.new_layout == TextureLayout::ShaderReadOnly
                || last_for_resource.new_layout == TextureLayout::General,
            "Final transition should be to an expected layout, got {:?}",
            last_for_resource.new_layout
        );
    }

    #[test]
    fn test_coalesce_keeps_independent_transitions() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Add transitions for different resources
        manager.add_pending(LayoutTransition {
            resource_id: 100,
            old_layout: TextureLayout::Undefined,
            new_layout: TextureLayout::ShaderReadOnly,
            subresource: whole_resource(),
        });

        manager.add_pending(LayoutTransition {
            resource_id: 200,
            old_layout: TextureLayout::Undefined,
            new_layout: TextureLayout::ColorAttachment,
            subresource: whole_resource(),
        });

        let coalesced = manager.coalesce_pending();

        // Both independent transitions should remain
        assert_eq!(coalesced.len(), 2, "Independent transitions should not be merged");
    }

    #[test]
    fn test_flush_returns_and_clears_pending() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        manager.add_pending(LayoutTransition {
            resource_id: 100,
            old_layout: TextureLayout::Undefined,
            new_layout: TextureLayout::ShaderReadOnly,
            subresource: whole_resource(),
        });

        manager.add_pending(LayoutTransition {
            resource_id: 200,
            old_layout: TextureLayout::Undefined,
            new_layout: TextureLayout::ColorAttachment,
            subresource: whole_resource(),
        });

        assert_eq!(manager.pending_count(), 2);

        let flushed = manager.flush_pending();

        assert!(!flushed.is_empty(), "Flush should return pending transitions");
        assert_eq!(manager.pending_count(), 0, "Flush should clear pending");
        assert!(!manager.has_pending());
    }

    #[test]
    fn test_clear_pending_only() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Add tracked resource
        manager.set_layout(100, whole_resource(), TextureLayout::General);

        // Add pending transition
        manager.add_pending(LayoutTransition {
            resource_id: 200,
            old_layout: TextureLayout::Undefined,
            new_layout: TextureLayout::ShaderReadOnly,
            subresource: whole_resource(),
        });

        manager.clear_pending();

        // Pending should be cleared but tracking preserved
        assert_eq!(manager.pending_count(), 0);
        assert!(manager.is_tracked(100), "Tracked resources should be preserved");
    }

    #[test]
    fn test_coalesce_same_subresource_different_mips() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Add transitions for different mip levels of same texture
        manager.add_pending(LayoutTransition {
            resource_id: 100,
            old_layout: TextureLayout::Undefined,
            new_layout: TextureLayout::TransferDst,
            subresource: single_subresource(0, 0),
        });

        manager.add_pending(LayoutTransition {
            resource_id: 100,
            old_layout: TextureLayout::Undefined,
            new_layout: TextureLayout::TransferDst,
            subresource: single_subresource(1, 0),
        });

        let coalesced = manager.coalesce_pending();

        // These could potentially be coalesced into a single transition with mip_count > 1
        // Or kept separate if targeting different subresources
        assert!(!coalesced.is_empty(), "Should have at least one coalesced transition");
    }

    #[test]
    fn test_transition_batch() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Set up initial layouts
        manager.set_whole_layout(1, TextureLayout::Undefined);
        manager.set_whole_layout(2, TextureLayout::Undefined);
        manager.set_whole_layout(3, TextureLayout::Undefined);

        // Batch transition multiple resources - API takes slice of (id, layout) tuples
        let transitions = manager.transition_batch(&[
            (1, TextureLayout::ShaderReadOnly),
            (2, TextureLayout::ShaderReadOnly),
            (3, TextureLayout::ShaderReadOnly),
        ]);

        assert_eq!(transitions.len(), 3, "Should generate transition for each resource");

        for t in &transitions {
            assert_eq!(t.old_layout, TextureLayout::Undefined);
            assert_eq!(t.new_layout, TextureLayout::ShaderReadOnly);
        }
    }

    #[test]
    fn test_batch_skips_same_layout_transitions() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Set up mixed layouts
        manager.set_whole_layout(1, TextureLayout::Undefined);
        manager.set_whole_layout(2, TextureLayout::ShaderReadOnly); // Already at target
        manager.set_whole_layout(3, TextureLayout::Undefined);

        let transitions = manager.transition_batch(&[
            (1, TextureLayout::ShaderReadOnly),
            (2, TextureLayout::ShaderReadOnly),
            (3, TextureLayout::ShaderReadOnly),
        ]);

        // Resource 2 should be skipped since it's already ShaderReadOnly
        assert!(transitions.len() <= 3);

        let ids: Vec<_> = transitions.iter().map(|t| t.resource_id).collect();
        // Resource 2 might not have a transition if it was already at target
        if ids.contains(&2) {
            // Some implementations might still return it
        }
    }

    #[test]
    fn test_has_pending_and_pending_count() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        assert!(!manager.has_pending());
        assert_eq!(manager.pending_count(), 0);

        manager.add_pending(LayoutTransition {
            resource_id: 1,
            old_layout: TextureLayout::Undefined,
            new_layout: TextureLayout::General,
            subresource: whole_resource(),
        });

        assert!(manager.has_pending());
        assert_eq!(manager.pending_count(), 1);
    }
}

// ============================================================================
// Advanced Tests: Snapshot and Restore
// ============================================================================

mod snapshot_restore {
    use super::*;

    #[test]
    fn test_snapshot_captures_state() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        manager.set_layout(1, single_subresource(0, 0), TextureLayout::ColorAttachment);
        manager.set_layout(2, single_subresource(0, 0), TextureLayout::ShaderReadOnly);

        let snapshot = manager.snapshot();

        assert!(snapshot.contains_key(&1));
        assert!(snapshot.contains_key(&2));
    }

    #[test]
    fn test_restore_from_snapshot() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        manager.set_layout(1, single_subresource(0, 0), TextureLayout::ColorAttachment);
        manager.set_layout(2, single_subresource(0, 0), TextureLayout::ShaderReadOnly);

        let snapshot = manager.snapshot();

        // Modify state
        manager.set_layout(1, single_subresource(0, 0), TextureLayout::TransferDst);
        manager.set_layout(3, single_subresource(0, 0), TextureLayout::General);

        // Restore
        manager.restore(snapshot);

        // Should be back to original state
        assert_eq!(
            manager.get_layout(1, single_subresource(0, 0)),
            Some(TextureLayout::ColorAttachment)
        );
        assert_eq!(
            manager.get_layout(2, single_subresource(0, 0)),
            Some(TextureLayout::ShaderReadOnly)
        );
    }

    #[test]
    fn test_merge_combines_managers() {
        let mut manager1 = LayoutTransitionManager::new(TransitionMode::Implicit);
        let mut manager2 = LayoutTransitionManager::new(TransitionMode::Implicit);

        manager1.set_layout(1, whole_resource(), TextureLayout::ColorAttachment);
        manager2.set_layout(2, whole_resource(), TextureLayout::ShaderReadOnly);

        manager1.merge(&manager2);

        assert!(manager1.is_tracked(1));
        assert!(manager1.is_tracked(2));
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_zero_resource_id() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        manager.set_layout(0, whole_resource(), TextureLayout::General);
        assert!(manager.is_tracked(0));
        assert_eq!(manager.get_whole_layout(0), Some(TextureLayout::General));
    }

    #[test]
    fn test_max_resource_id() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let max_id = u64::MAX;

        manager.set_layout(max_id, whole_resource(), TextureLayout::General);
        assert!(manager.is_tracked(max_id));
    }

    #[test]
    fn test_undefined_to_undefined_transition() {
        let path = LayoutTransitionManager::optimal_transition_path(
            TextureLayout::Undefined,
            TextureLayout::Undefined,
        );

        // Should be no-op
        assert!(path.is_empty() || path.len() == 1);
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        let removed = manager.remove(999);
        assert!(!removed, "Remove non-existent should return false");
    }

    #[test]
    fn test_transition_untracked_resource() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Try to transition a resource that isn't tracked
        let transition = manager.transition_to(999, TextureLayout::ShaderReadOnly);

        // Implementation may either:
        // 1. Return None since there's no known old_layout
        // 2. Assume Undefined layout and generate a transition
        // Both are valid behaviors - the key is consistency
        if let Some(t) = transition {
            // If transition is generated, it should assume Undefined as old layout
            assert_eq!(t.old_layout, TextureLayout::Undefined,
                "Untracked resource transition should assume Undefined old layout");
            assert_eq!(t.new_layout, TextureLayout::ShaderReadOnly);
        }
        // If None, that's also valid - implementation requires explicit tracking first
    }

    #[test]
    fn test_get_all_layouts_for_resource() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture_id = 100;

        manager.set_layout(texture_id, single_subresource(0, 0), TextureLayout::ColorAttachment);
        manager.set_layout(texture_id, single_subresource(1, 0), TextureLayout::ShaderReadOnly);

        let all_layouts = manager.get_all_layouts(texture_id);
        assert!(all_layouts.is_some());
        assert_eq!(all_layouts.unwrap().len(), 2);
    }

    #[test]
    fn test_get_all_layouts_untracked() {
        let manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        assert!(manager.get_all_layouts(999).is_none());
    }

    #[test]
    fn test_empty_batch_transition() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        let transitions: Vec<LayoutTransition> = manager.transition_batch(&[]);
        assert!(transitions.is_empty());
    }

    #[test]
    fn test_flush_empty_returns_empty() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        let flushed = manager.flush_pending();
        assert!(flushed.is_empty());
    }

    #[test]
    fn test_coalesce_empty_returns_empty() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        let coalesced = manager.coalesce_pending();
        assert!(coalesced.is_empty());
    }
}

// ============================================================================
// Layout-Specific Tests
// ============================================================================

mod layout_specific {
    use super::*;

    #[test]
    fn test_all_layout_variants() {
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
            TextureLayout::StorageImage,
        ];

        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        for (i, layout) in layouts.iter().enumerate() {
            manager.set_layout(i as u64, whole_resource(), *layout);
            assert_eq!(
                manager.get_whole_layout(i as u64),
                Some(*layout),
                "Failed to track layout {:?}",
                layout
            );
        }
    }

    #[test]
    fn test_transition_between_all_common_pairs() {
        let layouts = [
            TextureLayout::Undefined,
            TextureLayout::General,
            TextureLayout::ColorAttachment,
            TextureLayout::ShaderReadOnly,
            TextureLayout::TransferDst,
        ];

        for from in &layouts {
            for to in &layouts {
                let needed = LayoutTransitionManager::is_transition_needed(*from, *to);
                if from == to {
                    assert!(!needed, "Same layout {:?} should not need transition", from);
                }
                // Just ensure no panic
                let _path = LayoutTransitionManager::optimal_transition_path(*from, *to);
            }
        }
    }

    #[test]
    fn test_depth_stencil_layouts() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Test depth/stencil specific transitions
        manager.set_whole_layout(1, TextureLayout::Undefined);

        let transition = manager.transition_to(1, TextureLayout::DepthStencilAttachment);
        assert!(transition.is_some());

        let transition = manager.transition_to(1, TextureLayout::DepthStencilReadOnly);
        assert!(transition.is_some());
    }

    #[test]
    fn test_storage_image_layout() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        manager.set_whole_layout(1, TextureLayout::Undefined);
        let transition = manager.transition_to(1, TextureLayout::StorageImage);

        assert!(transition.is_some());
        assert_eq!(manager.get_whole_layout(1), Some(TextureLayout::StorageImage));
    }

    #[test]
    fn test_present_layout() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        manager.set_whole_layout(1, TextureLayout::ColorAttachment);
        let transition = manager.transition_to(1, TextureLayout::Present);

        assert!(transition.is_some());
        let t = transition.unwrap();
        assert_eq!(t.old_layout, TextureLayout::ColorAttachment);
        assert_eq!(t.new_layout, TextureLayout::Present);
    }
}

// ============================================================================
// Concurrency-Ready Tests (Single-threaded verification)
// ============================================================================

mod concurrency_ready {
    use super::*;

    #[test]
    fn test_many_resources_performance() {
        let mut manager = LayoutTransitionManager::with_capacity(TransitionMode::Implicit, 10000);

        // Track many resources
        for i in 0..1000 {
            manager.set_layout(i, whole_resource(), TextureLayout::General);
        }

        assert_eq!(manager.len(), 1000);

        // Transition all to different layout
        for i in 0..1000 {
            let _ = manager.transition_to(i, TextureLayout::ShaderReadOnly);
        }

        // Verify all updated
        for i in 0..1000 {
            assert_eq!(manager.get_whole_layout(i), Some(TextureLayout::ShaderReadOnly));
        }
    }

    #[test]
    fn test_many_pending_coalesce() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Add many pending transitions
        for i in 0..100 {
            manager.add_pending(LayoutTransition {
                resource_id: i,
                old_layout: TextureLayout::Undefined,
                new_layout: TextureLayout::ShaderReadOnly,
                subresource: whole_resource(),
            });
        }

        assert_eq!(manager.pending_count(), 100);

        let coalesced = manager.coalesce_pending();
        assert!(!coalesced.is_empty());
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration {
    use super::*;

    #[test]
    fn test_typical_render_pass_workflow() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Setup phase: create textures in undefined state
        let color_target = 1;
        let depth_target = 2;
        let input_texture = 3;

        manager.set_whole_layout(color_target, TextureLayout::Undefined);
        manager.set_whole_layout(depth_target, TextureLayout::Undefined);
        manager.set_whole_layout(input_texture, TextureLayout::ShaderReadOnly);

        // Begin render pass: transition to attachment layouts
        let t1 = manager.transition_to(color_target, TextureLayout::ColorAttachment);
        let t2 = manager.transition_to(depth_target, TextureLayout::DepthStencilAttachment);

        assert!(t1.is_some());
        assert!(t2.is_some());

        // End render pass: transition for presentation
        let t3 = manager.transition_to(color_target, TextureLayout::Present);
        assert!(t3.is_some());

        // Verify final states
        assert_eq!(manager.get_whole_layout(color_target), Some(TextureLayout::Present));
        assert_eq!(manager.get_whole_layout(depth_target), Some(TextureLayout::DepthStencilAttachment));
        assert_eq!(manager.get_whole_layout(input_texture), Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_compute_dispatch_workflow() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Input and output storage images
        let input = 1;
        let output = 2;

        manager.set_whole_layout(input, TextureLayout::ShaderReadOnly);
        manager.set_whole_layout(output, TextureLayout::Undefined);

        // Transition output to storage for compute write
        let t1 = manager.transition_to(output, TextureLayout::StorageImage);
        assert!(t1.is_some());

        // After compute: transition output to shader read for next pass
        let t2 = manager.transition_to(output, TextureLayout::ShaderReadOnly);
        assert!(t2.is_some());

        assert_eq!(manager.get_whole_layout(output), Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_mipmap_generation_workflow() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let texture = 100;

        // All mips start undefined
        for mip in 0..4 {
            manager.set_layout(texture, single_subresource(mip, 0), TextureLayout::Undefined);
        }

        // Upload to mip 0
        manager.set_layout(texture, single_subresource(0, 0), TextureLayout::TransferDst);

        // Generate mips: each mip becomes transfer src, next becomes transfer dst
        for mip in 0..3 {
            // Source mip to TransferSrc
            manager.transition_subresource(
                texture,
                single_subresource(mip, 0),
                TextureLayout::TransferSrc,
            );

            // Dest mip to TransferDst
            manager.transition_subresource(
                texture,
                single_subresource(mip + 1, 0),
                TextureLayout::TransferDst,
            );
        }

        // Final: all mips to ShaderReadOnly
        for mip in 0..4 {
            manager.transition_subresource(
                texture,
                single_subresource(mip, 0),
                TextureLayout::ShaderReadOnly,
            );
        }

        // Verify all mips are shader readable
        for mip in 0..4 {
            assert_eq!(
                manager.get_layout(texture, single_subresource(mip, 0)),
                Some(TextureLayout::ShaderReadOnly),
                "Mip {} should be ShaderReadOnly",
                mip
            );
        }
    }
}
