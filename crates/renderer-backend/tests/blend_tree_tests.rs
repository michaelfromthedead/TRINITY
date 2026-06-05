//! Integration tests for blend_tree module (T-AN-5.3).

use renderer_backend::blend_tree::{
    AdditiveBlendSpace, BlendTree1D, BlendTree2D, DirectionalBlendSpace,
    BLEND_EPSILON, angle_difference, barycentric_coords, normalize_angle,
};
use std::f32::consts::{PI, TAU};

// ---------------------------------------------------------------------------
// BlendTree1D Tests
// ---------------------------------------------------------------------------

#[test]
fn test_blend_tree_1d_new() {
    let tree = BlendTree1D::new("speed");
    assert_eq!(tree.parameter, "speed");
    assert!(tree.clips.is_empty());
}

#[test]
fn test_blend_tree_1d_add_clip() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(0.0, 0);
    tree.add_clip(5.0, 1);
    tree.add_clip(10.0, 2);

    assert_eq!(tree.clip_count(), 3);
}

#[test]
fn test_blend_tree_1d_threshold_range() {
    let mut tree = BlendTree1D::new("speed");
    assert!(tree.threshold_range().is_none());

    tree.add_clip(5.0, 0);
    tree.add_clip(0.0, 1);
    tree.add_clip(10.0, 2);

    let (min, max) = tree.threshold_range().unwrap();
    assert!((min - 0.0).abs() < BLEND_EPSILON);
    assert!((max - 10.0).abs() < BLEND_EPSILON);
}

#[test]
fn test_blend_tree_1d_single_clip() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(5.0, 0);

    let weights = tree.evaluate(0.0);
    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0], (0, 1.0));

    let weights = tree.evaluate(10.0);
    assert_eq!(weights[0], (0, 1.0));
}

#[test]
fn test_blend_tree_1d_at_threshold() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(0.0, 0);
    tree.add_clip(5.0, 1);
    tree.add_clip(10.0, 2);

    // Exactly at threshold
    let weights = tree.evaluate(5.0);
    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0], (1, 1.0));
}

#[test]
fn test_blend_tree_1d_between_clips() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(0.0, 0);
    tree.add_clip(10.0, 1);

    // Midpoint
    let weights = tree.evaluate(5.0);
    assert_eq!(weights.len(), 2);
    assert!((weights[0].1 - 0.5).abs() < 0.01);
    assert!((weights[1].1 - 0.5).abs() < 0.01);
}

#[test]
fn test_blend_tree_1d_below_min() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(5.0, 0);
    tree.add_clip(10.0, 1);

    let weights = tree.evaluate(0.0);
    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0].0, 0);
}

#[test]
fn test_blend_tree_1d_above_max() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(0.0, 0);
    tree.add_clip(5.0, 1);

    let weights = tree.evaluate(10.0);
    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0].0, 1);
}

#[test]
fn test_blend_tree_1d_unsorted_clips() {
    let mut tree = BlendTree1D::new("speed");
    // Add out of order
    tree.add_clip(10.0, 2);
    tree.add_clip(0.0, 0);
    tree.add_clip(5.0, 1);

    // Should still work correctly
    let weights = tree.evaluate(2.5);
    assert_eq!(weights.len(), 2);
    assert_eq!(weights[0].0, 0); // First clip
    assert_eq!(weights[1].0, 1); // Second clip
}

#[test]
fn test_blend_tree_1d_weight_normalization() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(0.0, 0);
    tree.add_clip(10.0, 1);

    for param in [0.0, 2.5, 5.0, 7.5, 10.0] {
        let weights = tree.evaluate(param);
        let total: f32 = weights.iter().map(|(_, w)| w).sum();
        assert!((total - 1.0).abs() < 0.01, "weights should sum to 1.0 at {}", param);
    }
}

#[test]
fn test_blend_tree_1d_empty() {
    let mut tree = BlendTree1D::new("speed");
    let weights = tree.evaluate(5.0);
    assert!(weights.is_empty());
}

#[test]
fn test_blend_tree_1d_validate() {
    let tree = BlendTree1D::new("speed");
    assert!(tree.validate().is_err());

    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(0.0, 0);
    assert!(tree.validate().is_ok());
}

// ---------------------------------------------------------------------------
// BlendTree2D Tests
// ---------------------------------------------------------------------------

#[test]
fn test_blend_tree_2d_new() {
    let tree = BlendTree2D::new("vel_x", "vel_z");
    assert_eq!(tree.param_x, "vel_x");
    assert_eq!(tree.param_y, "vel_z");
    assert!(tree.clips.is_empty());
}

#[test]
fn test_blend_tree_2d_add_clip() {
    let mut tree = BlendTree2D::new("vel_x", "vel_z");
    tree.add_clip(0.0, 0.0, 0);
    tree.add_clip(1.0, 0.0, 1);
    tree.add_clip(0.0, 1.0, 2);

    assert_eq!(tree.clip_count(), 3);
}

#[test]
fn test_blend_tree_2d_bounds() {
    let mut tree = BlendTree2D::new("x", "y");
    assert!(tree.bounds().is_none());

    tree.add_clip(-1.0, -1.0, 0);
    tree.add_clip(1.0, 2.0, 1);

    let ((min_x, min_y), (max_x, max_y)) = tree.bounds().unwrap();
    assert!((min_x - (-1.0)).abs() < BLEND_EPSILON);
    assert!((min_y - (-1.0)).abs() < BLEND_EPSILON);
    assert!((max_x - 1.0).abs() < BLEND_EPSILON);
    assert!((max_y - 2.0).abs() < BLEND_EPSILON);
}

#[test]
fn test_blend_tree_2d_triangulation() {
    let mut tree = BlendTree2D::new("x", "y");
    tree.add_clip(0.0, 0.0, 0);
    tree.add_clip(1.0, 0.0, 1);
    tree.add_clip(0.5, 1.0, 2);

    assert!(!tree.is_triangulated());
    tree.triangulate();
    assert!(tree.is_triangulated());
    assert_eq!(tree.triangles.len(), 1);
}

#[test]
fn test_blend_tree_2d_triangulation_square() {
    let mut tree = BlendTree2D::new("x", "y");
    tree.add_clip(0.0, 0.0, 0);
    tree.add_clip(1.0, 0.0, 1);
    tree.add_clip(1.0, 1.0, 2);
    tree.add_clip(0.0, 1.0, 3);

    tree.triangulate();
    // A square should produce 2 triangles
    assert_eq!(tree.triangles.len(), 2);
}

#[test]
fn test_blend_tree_2d_barycentric_center() {
    let weights = barycentric_coords(
        1.0 / 3.0, 1.0 / 3.0, // center of unit triangle
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
    );

    let w = weights.unwrap();
    // All three weights should be equal at center
    assert!((w[0] - 1.0 / 3.0).abs() < 0.01);
    assert!((w[1] - 1.0 / 3.0).abs() < 0.01);
    assert!((w[2] - 1.0 / 3.0).abs() < 0.01);
}

#[test]
fn test_blend_tree_2d_barycentric_vertex() {
    // Point at vertex 0
    let w = barycentric_coords(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
    assert!((w[0] - 1.0).abs() < BLEND_EPSILON);
    assert!(w[1].abs() < BLEND_EPSILON);
    assert!(w[2].abs() < BLEND_EPSILON);

    // Point at vertex 1
    let w = barycentric_coords(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
    assert!(w[0].abs() < BLEND_EPSILON);
    assert!((w[1] - 1.0).abs() < BLEND_EPSILON);
    assert!(w[2].abs() < BLEND_EPSILON);
}

#[test]
fn test_blend_tree_2d_single_clip() {
    let mut tree = BlendTree2D::new("x", "y");
    tree.add_clip(0.0, 0.0, 5);

    let weights = tree.evaluate(1.0, 1.0);
    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0], (5, 1.0));
}

#[test]
fn test_blend_tree_2d_two_clips() {
    let mut tree = BlendTree2D::new("x", "y");
    tree.add_clip(0.0, 0.0, 0);
    tree.add_clip(2.0, 0.0, 1);

    // Midpoint
    let weights = tree.evaluate(1.0, 0.0);
    assert_eq!(weights.len(), 2);
    assert!((weights[0].1 - 0.5).abs() < 0.01);
    assert!((weights[1].1 - 0.5).abs() < 0.01);
}

#[test]
fn test_blend_tree_2d_evaluate_inside() {
    let mut tree = BlendTree2D::new("x", "y");
    tree.add_clip(0.0, 0.0, 0);
    tree.add_clip(1.0, 0.0, 1);
    tree.add_clip(0.5, 1.0, 2);
    tree.triangulate();

    // Center of triangle
    let weights = tree.evaluate(0.5, 0.33);
    assert!(!weights.is_empty());

    let total: f32 = weights.iter().map(|(_, w)| w).sum();
    assert!((total - 1.0).abs() < 0.01);
}

#[test]
fn test_blend_tree_2d_evaluate_outside() {
    let mut tree = BlendTree2D::new("x", "y");
    tree.add_clip(0.0, 0.0, 0);
    tree.add_clip(1.0, 0.0, 1);
    tree.add_clip(0.5, 1.0, 2);
    tree.triangulate();

    // Outside triangle - should return nearest
    let weights = tree.evaluate(10.0, 10.0);
    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0].0, 2); // Nearest to (0.5, 1.0)
}

#[test]
fn test_blend_tree_2d_validate() {
    let tree = BlendTree2D::new("x", "y");
    assert!(tree.validate().is_err());

    let mut tree = BlendTree2D::new("x", "y");
    tree.add_clip(0.0, 0.0, 0);
    tree.add_clip(1.0, 0.0, 1);
    tree.add_clip(0.5, 1.0, 2);
    assert!(tree.validate().is_ok());
}

// ---------------------------------------------------------------------------
// DirectionalBlendSpace Tests
// ---------------------------------------------------------------------------

#[test]
fn test_directional_new() {
    let space = DirectionalBlendSpace::new("direction", "speed");
    assert_eq!(space.direction_param, "direction");
    assert_eq!(space.speed_param, "speed");
    assert!(space.clips.is_empty());
}

#[test]
fn test_directional_add_clip() {
    let mut space = DirectionalBlendSpace::new("dir", "spd");
    space.add_clip(0.0, 0.0, 0);
    space.add_clip(PI / 2.0, 5.0, 1);
    space.add_clip(-PI, 10.0, 2);

    assert_eq!(space.clip_count(), 3);
}

#[test]
fn test_directional_angle_normalization() {
    let mut space = DirectionalBlendSpace::new("dir", "spd");
    space.add_clip(-PI, 0.0, 0);
    space.add_clip(3.0 * PI, 0.0, 1);

    // Both should be normalized to [0, 2*PI)
    assert!(space.clips[0].0 >= 0.0 && space.clips[0].0 < TAU);
    assert!(space.clips[1].0 >= 0.0 && space.clips[1].0 < TAU);
}

#[test]
fn test_directional_wrap_around() {
    // Test that angle difference handles wrap-around correctly
    let diff1 = angle_difference(0.1, TAU - 0.1);
    assert!((diff1 - 0.2).abs() < 0.01);

    let diff2 = angle_difference(TAU - 0.1, 0.1);
    assert!((diff2 - 0.2).abs() < 0.01);
}

#[test]
fn test_directional_evaluate_single() {
    let mut space = DirectionalBlendSpace::new("dir", "spd");
    space.add_clip(0.0, 5.0, 0);

    let weights = space.evaluate(0.0, 5.0);
    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0], (0, 1.0));
}

#[test]
fn test_directional_evaluate_direction_match() {
    let mut space = DirectionalBlendSpace::new("dir", "spd");
    space.set_angle_tolerance(PI / 4.0);
    space.add_clip(0.0, 0.0, 0);
    space.add_clip(0.0, 10.0, 1);

    // Same direction, different speeds
    let weights = space.evaluate(0.0, 5.0);
    assert_eq!(weights.len(), 2);
}

#[test]
fn test_directional_evaluate_no_direction_match() {
    let mut space = DirectionalBlendSpace::new("dir", "spd");
    space.set_angle_tolerance(PI / 8.0);
    space.add_clip(0.0, 5.0, 0);
    space.add_clip(PI, 5.0, 1);

    // Direction doesn't match any clip
    let weights = space.evaluate(PI / 2.0, 5.0);
    // Should return nearest
    assert_eq!(weights.len(), 1);
}

#[test]
fn test_directional_cardinal_directions() {
    let mut space = DirectionalBlendSpace::new("dir", "spd");
    space.set_angle_tolerance(PI / 4.0);

    // Cardinal directions
    space.add_clip(0.0, 5.0, 0);         // Forward
    space.add_clip(PI / 2.0, 5.0, 1);    // Right
    space.add_clip(PI, 5.0, 2);          // Back
    space.add_clip(3.0 * PI / 2.0, 5.0, 3); // Left

    // Test each cardinal direction
    let w = space.evaluate(0.0, 5.0);
    assert!(w.iter().any(|(idx, _)| *idx == 0));

    let w = space.evaluate(PI / 2.0, 5.0);
    assert!(w.iter().any(|(idx, _)| *idx == 1));
}

#[test]
fn test_directional_validate() {
    let space = DirectionalBlendSpace::new("dir", "spd");
    assert!(space.validate().is_err());

    let mut space = DirectionalBlendSpace::new("dir", "spd");
    space.add_clip(0.0, 0.0, 0);
    assert!(space.validate().is_ok());
}

// ---------------------------------------------------------------------------
// AdditiveBlendSpace Tests
// ---------------------------------------------------------------------------

#[test]
fn test_additive_new() {
    let space = AdditiveBlendSpace::new(0, "intensity");
    assert_eq!(space.base_clip, 0);
    assert_eq!(space.additive_param, "intensity");
    assert!(space.additive_clips.is_empty());
}

#[test]
fn test_additive_add_clip() {
    let mut space = AdditiveBlendSpace::new(0, "intensity");
    space.add_additive_clip(0.0, 1);
    space.add_additive_clip(1.0, 2);

    assert_eq!(space.additive_clip_count(), 2);
}

#[test]
fn test_additive_bone_mask() {
    let mut space = AdditiveBlendSpace::new(0, "intensity");
    space.set_bone_mask(vec![true, true, false, false, true]);

    assert!(space.is_bone_masked(0));
    assert!(space.is_bone_masked(1));
    assert!(!space.is_bone_masked(2));
    assert!(!space.is_bone_masked(3));
    assert!(space.is_bone_masked(4));

    // Out of bounds should return true
    assert!(space.is_bone_masked(100));

    space.clear_bone_mask();
    assert!(space.is_bone_masked(2)); // Now all bones affected
}

#[test]
fn test_additive_evaluate_no_additives() {
    let mut space = AdditiveBlendSpace::new(0, "intensity");

    let (base, additives) = space.evaluate(0.5);
    assert_eq!(base, 0);
    assert!(additives.is_empty());
}

#[test]
fn test_additive_evaluate_at_min() {
    let mut space = AdditiveBlendSpace::new(0, "intensity");
    space.add_additive_clip(0.0, 1);
    space.add_additive_clip(1.0, 2);

    let (base, additives) = space.evaluate(0.0);
    assert_eq!(base, 0);
    // At minimum threshold, additive weight should be 0
    assert!(additives.is_empty());
}

#[test]
fn test_additive_evaluate_at_max() {
    let mut space = AdditiveBlendSpace::new(0, "intensity");
    space.add_additive_clip(0.0, 1);
    space.add_additive_clip(1.0, 2);

    let (base, additives) = space.evaluate(1.0);
    assert_eq!(base, 0);
    assert_eq!(additives.len(), 1);
    assert_eq!(additives[0].0, 2);
    assert!((additives[0].1 - 1.0).abs() < 0.01);
}

#[test]
fn test_additive_evaluate_between() {
    let mut space = AdditiveBlendSpace::new(0, "intensity");
    space.add_additive_clip(0.0, 1);
    space.add_additive_clip(1.0, 2);

    let (base, additives) = space.evaluate(0.5);
    assert_eq!(base, 0);
    // Should have both additive clips blended
    assert_eq!(additives.len(), 2);

    // Total additive weight should be 0.5 (halfway through the range)
    let total: f32 = additives.iter().map(|(_, w)| w).sum();
    assert!((total - 0.5).abs() < 0.01);
}

#[test]
fn test_additive_max_weight() {
    let mut space = AdditiveBlendSpace::new(0, "intensity");
    space.set_max_weight(0.5);
    space.add_additive_clip(0.0, 1);
    space.add_additive_clip(1.0, 2);

    let (_, additives) = space.evaluate(1.0);
    let total: f32 = additives.iter().map(|(_, w)| w).sum();
    assert!((total - 0.5).abs() < 0.01);
}

#[test]
fn test_additive_single_clip() {
    let mut space = AdditiveBlendSpace::new(0, "intensity");
    space.add_additive_clip(0.5, 1);

    // Below threshold
    let (base, additives) = space.evaluate(0.0);
    assert_eq!(base, 0);
    assert!(additives.is_empty());

    // At threshold
    let (base, additives) = space.evaluate(0.5);
    assert_eq!(base, 0);
    // Single clip at its threshold
    assert_eq!(additives.len(), 1);
}

#[test]
fn test_additive_validate() {
    let space = AdditiveBlendSpace::new(0, "intensity");
    // Valid even with no additive clips
    assert!(space.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Utility Function Tests
// ---------------------------------------------------------------------------

#[test]
fn test_normalize_angle() {
    assert!((normalize_angle(0.0) - 0.0).abs() < BLEND_EPSILON);
    assert!((normalize_angle(TAU) - 0.0).abs() < BLEND_EPSILON);
    assert!((normalize_angle(-PI) - PI).abs() < BLEND_EPSILON);
    assert!((normalize_angle(3.0 * PI) - PI).abs() < BLEND_EPSILON);
}

#[test]
fn test_angle_difference_same() {
    assert!(angle_difference(0.0, 0.0).abs() < BLEND_EPSILON);
    assert!(angle_difference(PI, PI).abs() < BLEND_EPSILON);
}

#[test]
fn test_angle_difference_opposite() {
    let diff = angle_difference(0.0, PI);
    assert!((diff - PI).abs() < BLEND_EPSILON);
}

#[test]
fn test_angle_difference_wrap() {
    // Small difference across 0/2PI boundary
    let diff = angle_difference(0.1, TAU - 0.1);
    assert!((diff - 0.2).abs() < 0.01);
}

#[test]
fn test_barycentric_degenerate() {
    // Collinear points
    let w = barycentric_coords(0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0);
    assert!(w.is_none());
}

#[test]
fn test_barycentric_edge() {
    // Point on edge between v0 and v1
    let w = barycentric_coords(0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
    assert!((w[2]).abs() < BLEND_EPSILON); // v2 weight should be 0
    assert!((w[0] + w[1] - 1.0).abs() < BLEND_EPSILON);
}

// ---------------------------------------------------------------------------
// Edge Case Tests
// ---------------------------------------------------------------------------

#[test]
fn test_edge_case_duplicate_threshold_1d() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(5.0, 0);
    tree.add_clip(5.0, 1); // Same threshold

    // Should not crash
    let weights = tree.evaluate(5.0);
    assert!(!weights.is_empty());
}

#[test]
fn test_edge_case_same_clip_index() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(0.0, 0);
    tree.add_clip(10.0, 0); // Same clip at different thresholds

    let weights = tree.evaluate(5.0);
    // Should consolidate to single clip
    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0].0, 0);
}

#[test]
fn test_edge_case_negative_param() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(-10.0, 0);
    tree.add_clip(0.0, 1);
    tree.add_clip(10.0, 2);

    let weights = tree.evaluate(-5.0);
    assert_eq!(weights.len(), 2);
}

#[test]
fn test_edge_case_very_small_range() {
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(0.0, 0);
    tree.add_clip(0.0001, 1);

    // Should handle very small ranges
    let weights = tree.evaluate(0.00005);
    assert!(!weights.is_empty());
}

#[test]
fn test_edge_case_collinear_2d() {
    let mut tree = BlendTree2D::new("x", "y");
    tree.add_clip(0.0, 0.0, 0);
    tree.add_clip(1.0, 0.0, 1);
    tree.add_clip(2.0, 0.0, 2);

    tree.triangulate();
    // Collinear points produce no triangles
    assert!(tree.triangles.is_empty());

    // Should still evaluate via nearest
    let weights = tree.evaluate(0.5, 0.0);
    assert!(!weights.is_empty());
}

#[test]
fn test_edge_case_directional_zero_speed() {
    let mut space = DirectionalBlendSpace::new("dir", "spd");
    space.add_clip(0.0, 0.0, 0);
    space.add_clip(0.0, 10.0, 1);

    let weights = space.evaluate(0.0, 0.0);
    assert!(!weights.is_empty());
}

#[test]
fn test_edge_case_additive_negative_param() {
    let mut space = AdditiveBlendSpace::new(0, "intensity");
    space.add_additive_clip(0.0, 1);
    space.add_additive_clip(1.0, 2);

    let (base, additives) = space.evaluate(-1.0);
    assert_eq!(base, 0);
    assert!(additives.is_empty());
}

// ---------------------------------------------------------------------------
// Integration Tests
// ---------------------------------------------------------------------------

#[test]
fn test_locomotion_blend_tree() {
    // Simulate a typical locomotion setup
    let mut tree = BlendTree1D::new("speed");
    tree.add_clip(0.0, 0);   // idle
    tree.add_clip(1.5, 1);   // walk
    tree.add_clip(3.0, 2);   // jog
    tree.add_clip(6.0, 3);   // run
    tree.add_clip(10.0, 4);  // sprint

    // Test various speeds
    let test_cases = [
        (0.0, vec![(0, 1.0)]),          // idle
        (1.5, vec![(1, 1.0)]),          // walk
        (4.5, vec![(2, 0.5), (3, 0.5)]), // jog/run blend
    ];

    for (speed, expected) in test_cases {
        let weights = tree.evaluate(speed);
        assert_eq!(weights.len(), expected.len(), "speed={}", speed);
        for (i, (clip, weight)) in expected.into_iter().enumerate() {
            assert_eq!(weights[i].0, clip, "speed={}, clip mismatch", speed);
            assert!((weights[i].1 - weight).abs() < 0.01, "speed={}, weight mismatch", speed);
        }
    }
}

#[test]
fn test_directional_locomotion() {
    let mut space = DirectionalBlendSpace::new("direction", "speed");
    space.set_angle_tolerance(PI / 4.0);

    // 8-directional setup: forward=0, NE=1, right=2, SE=3, back=4, SW=5, left=6, NW=7
    // Angles: 0, PI/4, PI/2, 3PI/4, PI, 5PI/4, 3PI/2, 7PI/4
    for i in 0..8 {
        let angle = (i as f32) * PI / 4.0;
        space.add_clip(angle, 5.0, i);
    }

    // Test cardinal directions - forward is clip 0
    let forward_weights = space.evaluate(0.0, 5.0);
    assert!(forward_weights.iter().any(|(idx, _)| *idx == 0), "forward should match clip 0");

    // Right is at PI/2 which is clip 2
    let right_weights = space.evaluate(PI / 2.0, 5.0);
    // Check that we get a result (may be clip 2 or nearby due to tolerance)
    assert!(!right_weights.is_empty(), "right direction should have a match");
}

#[test]
fn test_upper_body_additive() {
    let mut space = AdditiveBlendSpace::new(0, "aim_weight");

    // Upper body mask (bones 0-10)
    let mut mask = vec![false; 50];
    for i in 0..10 {
        mask[i] = true;
    }
    space.set_bone_mask(mask);

    space.add_additive_clip(0.0, 1); // aim down
    space.add_additive_clip(0.5, 2); // aim center
    space.add_additive_clip(1.0, 3); // aim up

    // Test aim blend
    let (base, additives) = space.evaluate(0.75);
    assert_eq!(base, 0);
    assert!(!additives.is_empty());

    // Check bone masking
    assert!(space.is_bone_masked(5));   // Upper body
    assert!(!space.is_bone_masked(30)); // Lower body
}
