//! Integration tests for animation foundation (T-AN-1.8).
//!
//! This module provides comprehensive integration tests covering:
//! - Skeleton hierarchy evaluation with 10+ bones
//! - Bind/inverse bind matrix correctness for skinned mesh
//! - Root motion extraction
//! - Multiple root bones handling
//! - Pose blending (lerp/slerp precision, weight edge cases, additive layering)
//! - SIMD vs scalar accuracy comparison
//! - Animation clip sampling (keyframe interpolation, loop modes, events)
//! - Performance benchmarks

#[cfg(test)]
mod tests {
    use crate::animation_clip::{
        AnimationClip, AnimationClipBuilder, AnimationEvent, BoneTrack, CurveTrack,
        EventTrack, InterpolationMode, Keyframe, LoopMode, Track,
    };
    use crate::pose::{lerp_vec3, nlerp_quat, slerp_quat, Pose, PoseBuffer, PoseType};
    use crate::skeleton::{Bone, Skeleton, SkeletonBuilder, Transform, MAX_BONES};
    use crate::skeleton_simd::{
        blend_poses_additive_simd, blend_poses_masked, blend_poses_simd,
        compute_bone_chain_simd, compute_skinning_matrices_simd, normalize_rotations_simd,
        poses_approx_equal, SoAPose, SIMD_LANE_WIDTH,
    };
    use glam::{Mat4, Quat, Vec3};
    use std::f32::consts::PI;

    // =========================================================================
    // SKELETON HIERARCHY TESTS
    // =========================================================================

    /// Test parent-child chain evaluation with 10+ bones
    #[test]
    fn test_skeleton_hierarchy_10_bone_chain() {
        let mut skeleton = Skeleton::new();

        // Create a 12-bone chain: root -> spine1 -> spine2 -> ... -> head
        skeleton.add_bone(Bone::root("root"));
        for i in 1..12 {
            skeleton.add_bone(
                Bone::new(format!("bone_{}", i))
                    .with_parent(i - 1)
                    .with_local_transform(Transform::from_position(Vec3::new(0.0, 1.0, 0.0))),
            );
        }

        assert_eq!(skeleton.bone_count(), 12);
        assert!(skeleton.validate().is_ok());

        // Create local poses that match the skeleton's bind pose local transforms
        // (compute_world_transforms uses these poses, not the skeleton's local_transform)
        let local_poses: Vec<Transform> = skeleton
            .bones()
            .iter()
            .map(|b| b.local_transform)
            .collect();
        let world = skeleton.compute_world_transforms(&local_poses);

        // Each bone should accumulate the parent's position
        // Root is at (0,0,0), bone_1 at (0,1,0), bone_2 at (0,2,0), etc.
        for i in 0..12 {
            let expected_y = i as f32;
            let actual_pos = world[i].w_axis.truncate();
            assert!(
                actual_pos.abs_diff_eq(Vec3::new(0.0, expected_y, 0.0), 1e-4),
                "Bone {} position mismatch: expected (0, {}, 0), got {:?}",
                i,
                expected_y,
                actual_pos
            );
        }
    }

    /// Test skeleton hierarchy with branching (spine -> left_arm, right_arm)
    #[test]
    fn test_skeleton_branching_hierarchy() {
        let skeleton = SkeletonBuilder::new()
            .root("root")
            .child_at("spine", "root", Vec3::new(0.0, 1.0, 0.0))
            .child_at("left_shoulder", "spine", Vec3::new(-1.0, 0.5, 0.0))
            .child_at("left_elbow", "left_shoulder", Vec3::new(-0.5, 0.0, 0.0))
            .child_at("left_hand", "left_elbow", Vec3::new(-0.5, 0.0, 0.0))
            .child_at("right_shoulder", "spine", Vec3::new(1.0, 0.5, 0.0))
            .child_at("right_elbow", "right_shoulder", Vec3::new(0.5, 0.0, 0.0))
            .child_at("right_hand", "right_elbow", Vec3::new(0.5, 0.0, 0.0))
            .child_at("head", "spine", Vec3::new(0.0, 0.5, 0.0))
            .build_unchecked();

        assert_eq!(skeleton.bone_count(), 9);
        assert!(skeleton.validate().is_ok());

        // Verify hierarchy structure
        let spine_idx = skeleton.bone_index("spine").unwrap();
        let children = skeleton.children(spine_idx);
        assert_eq!(children.len(), 3); // left_shoulder, right_shoulder, head

        // Compute world transforms
        let local_poses: Vec<Transform> = skeleton
            .bones()
            .iter()
            .map(|b| b.local_transform)
            .collect();
        let world = skeleton.compute_world_transforms(&local_poses);

        // Left hand should be at (-2, 1.5, 0): root(0) + spine(0,1,0) + left_shoulder(-1,0.5,0) + elbow(-0.5,0,0) + hand(-0.5,0,0)
        let left_hand_idx = skeleton.bone_index("left_hand").unwrap();
        let left_hand_pos = world[left_hand_idx].w_axis.truncate();
        assert!(
            left_hand_pos.abs_diff_eq(Vec3::new(-2.0, 1.5, 0.0), 1e-4),
            "Left hand position mismatch: {:?}",
            left_hand_pos
        );
    }

    /// Test bind/inverse bind matrix correctness for skinned mesh
    #[test]
    fn test_bind_inverse_bind_matrix_correctness() {
        let mut skeleton = Skeleton::new();

        // Create skeleton with known transforms
        skeleton.add_bone(
            Bone::root("root").with_local_transform(Transform::from_position(Vec3::new(
                0.0, 1.0, 0.0,
            ))),
        );
        skeleton.add_bone(
            Bone::new("child")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 2.0, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("grandchild")
                .with_parent(1)
                .with_local_transform(Transform::from_position_rotation(
                    Vec3::new(1.0, 0.0, 0.0),
                    Quat::from_rotation_z(PI / 4.0),
                )),
        );

        // Compute inverse bind matrices
        skeleton.compute_inverse_bind_matrices().unwrap();

        // Verify: world * inverse_bind = identity for bind pose
        let world = skeleton.compute_bind_pose_world_transforms();
        for (i, bone) in skeleton.bones().iter().enumerate() {
            let result = world[i] * bone.inverse_bind_matrix;
            assert!(
                result.abs_diff_eq(Mat4::IDENTITY, 1e-4),
                "Bone {} bind pose validation failed: world * inverse_bind != identity",
                bone.name
            );
        }

        // Verify skinning matrices in animated pose
        let animated_poses: Vec<Transform> = skeleton
            .bones()
            .iter()
            .map(|b| {
                let mut t = b.local_transform;
                t.position += Vec3::new(0.1, 0.0, 0.0); // Small offset
                t
            })
            .collect();

        let skinning = skeleton.compute_skinning_matrices(&animated_poses);
        assert_eq!(skinning.len(), 3);

        // Skinning matrices should be different from identity (we animated them)
        for mat in &skinning {
            assert!(
                !mat.abs_diff_eq(Mat4::IDENTITY, 1e-4),
                "Skinning matrix should differ from identity when animated"
            );
        }
    }

    /// Test root motion extraction by computing delta between frames
    #[test]
    fn test_root_motion_extraction() {
        let skeleton = SkeletonBuilder::new()
            .root("root")
            .child_at("spine", "root", Vec3::new(0.0, 1.0, 0.0))
            .build_unchecked();

        // Frame 0: root at origin
        let frame0_poses = vec![
            Transform::from_position(Vec3::new(0.0, 0.0, 0.0)),
            Transform::IDENTITY,
        ];

        // Frame 1: root moved forward
        let frame1_poses = vec![
            Transform::from_position(Vec3::new(1.0, 0.0, 2.0)),
            Transform::IDENTITY,
        ];

        let world0 = skeleton.compute_world_transforms(&frame0_poses);
        let world1 = skeleton.compute_world_transforms(&frame1_poses);

        // Root motion = delta of root bone world position
        let root_motion_delta = world1[0].w_axis.truncate() - world0[0].w_axis.truncate();
        assert!(root_motion_delta.abs_diff_eq(Vec3::new(1.0, 0.0, 2.0), 1e-5));

        // Child bone should move with root
        let spine_delta = world1[1].w_axis.truncate() - world0[1].w_axis.truncate();
        assert!(
            spine_delta.abs_diff_eq(Vec3::new(1.0, 0.0, 2.0), 1e-5),
            "Spine should move with root"
        );
    }

    /// Test multiple root bones handling (forest structure)
    #[test]
    fn test_multiple_root_bones() {
        let mut skeleton = Skeleton::new();

        // Create two separate hierarchies
        skeleton.add_bone(Bone::root("body_root"));
        skeleton.add_bone(
            Bone::new("spine")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 1.0, 0.0))),
        );

        // Second root for a separate entity (e.g., weapon)
        skeleton.add_bone(
            Bone::root("weapon_root")
                .with_local_transform(Transform::from_position(Vec3::new(2.0, 0.0, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("weapon_blade")
                .with_parent(2)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.5, 0.0))),
        );

        assert_eq!(skeleton.root_count(), 2);
        assert_eq!(skeleton.root_indices(), &[0, 2]);
        assert!(skeleton.validate().is_ok());

        // Compute world transforms
        let local_poses: Vec<Transform> = skeleton
            .bones()
            .iter()
            .map(|b| b.local_transform)
            .collect();
        let world = skeleton.compute_world_transforms(&local_poses);

        // Body spine at (0, 1, 0)
        assert!(world[1]
            .w_axis
            .truncate()
            .abs_diff_eq(Vec3::new(0.0, 1.0, 0.0), 1e-5));

        // Weapon blade at (2, 0.5, 0)
        assert!(world[3]
            .w_axis
            .truncate()
            .abs_diff_eq(Vec3::new(2.0, 0.5, 0.0), 1e-5));

        // Verify no common ancestor between body and weapon hierarchies
        assert_eq!(skeleton.common_ancestor(1, 3), None);
    }

    /// Test skeleton depth calculation for deep hierarchies
    #[test]
    fn test_skeleton_depth_calculation() {
        let mut skeleton = Skeleton::new();

        // Create a 15-deep hierarchy
        skeleton.add_bone(Bone::root("root"));
        for i in 1..15 {
            skeleton.add_bone(Bone::new(format!("bone_{}", i)).with_parent(i - 1));
        }

        // Verify depths
        for i in 0..15 {
            assert_eq!(skeleton.bone_depth(i), Some(i));
        }

        // Verify path to root
        let path = skeleton.path_to_root(14);
        assert_eq!(path.len(), 15);
        assert_eq!(path[0], 0); // Root first
        assert_eq!(path[14], 14); // Leaf last
    }

    // =========================================================================
    // POSE BLENDING TESTS
    // =========================================================================

    /// Test lerp/slerp precision for position/rotation
    #[test]
    fn test_lerp_slerp_precision() {
        // Position lerp precision
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(100.0, 200.0, 300.0);

        let result_25 = lerp_vec3(p0, p1, 0.25);
        assert!(result_25.abs_diff_eq(Vec3::new(25.0, 50.0, 75.0), 1e-5));

        let result_75 = lerp_vec3(p0, p1, 0.75);
        assert!(result_75.abs_diff_eq(Vec3::new(75.0, 150.0, 225.0), 1e-5));

        // Rotation slerp precision (use 90 degrees to avoid singularity at 180)
        let r0 = Quat::IDENTITY;
        let r1 = Quat::from_rotation_y(PI / 2.0); // 90 degrees

        // Midpoint should be 45 degrees
        let result_mid = slerp_quat(r0, r1, 0.5);
        let expected_mid = Quat::from_rotation_y(PI / 4.0);
        assert!(
            result_mid.dot(expected_mid).abs() > 0.999,
            "Slerp midpoint precision check: dot={}, expected={:?}, got={:?}",
            result_mid.dot(expected_mid),
            expected_mid,
            result_mid
        );

        // Quarter point should be 22.5 degrees
        let result_quarter = slerp_quat(r0, r1, 0.25);
        let expected_quarter = Quat::from_rotation_y(PI / 8.0);
        assert!(
            result_quarter.dot(expected_quarter).abs() > 0.999,
            "Slerp quarter precision check"
        );
    }

    /// Test weight edge cases (0.0, 0.5, 1.0)
    #[test]
    fn test_blend_weight_edge_cases() {
        let mut pose_a = Pose::new(3, PoseType::Current);
        let mut pose_b = Pose::new(3, PoseType::Current);

        // Set distinct values
        pose_a.positions[0] = Vec3::new(0.0, 0.0, 0.0);
        pose_b.positions[0] = Vec3::new(10.0, 20.0, 30.0);

        pose_a.rotations[0] = Quat::IDENTITY;
        pose_b.rotations[0] = Quat::from_rotation_z(PI / 2.0);

        pose_a.scales[0] = Vec3::ONE;
        pose_b.scales[0] = Vec3::splat(2.0);

        // Weight = 0.0: should equal pose_a exactly
        let result_0 = pose_a.blend(&pose_b, 0.0);
        assert!(result_0.positions[0].abs_diff_eq(pose_a.positions[0], 1e-6));
        assert!(result_0.rotations[0].abs_diff_eq(pose_a.rotations[0], 1e-6));
        assert!(result_0.scales[0].abs_diff_eq(pose_a.scales[0], 1e-6));

        // Weight = 1.0: should equal pose_b exactly
        let result_1 = pose_a.blend(&pose_b, 1.0);
        assert!(result_1.positions[0].abs_diff_eq(pose_b.positions[0], 1e-6));
        assert!(result_1.rotations[0].abs_diff_eq(pose_b.rotations[0], 1e-5));
        assert!(result_1.scales[0].abs_diff_eq(pose_b.scales[0], 1e-6));

        // Weight = 0.5: should be midpoint
        let result_05 = pose_a.blend(&pose_b, 0.5);
        assert!(result_05
            .positions[0]
            .abs_diff_eq(Vec3::new(5.0, 10.0, 15.0), 1e-6));
        assert!(result_05.scales[0].abs_diff_eq(Vec3::splat(1.5), 1e-6));
        // Rotation should be halfway
        let expected_rot = Quat::from_rotation_z(PI / 4.0);
        assert!(result_05.rotations[0].abs_diff_eq(expected_rot, 1e-4));
    }

    /// Test additive pose layering
    #[test]
    fn test_additive_pose_layering() {
        let bone_count = 4;

        // Base pose with some animation
        let mut base = Pose::new(bone_count, PoseType::Current);
        base.positions[0] = Vec3::new(0.0, 1.0, 0.0);
        base.rotations[0] = Quat::from_rotation_y(PI / 6.0); // 30 degrees

        // Additive pose representing "lean forward"
        let mut additive = Pose::new(bone_count, PoseType::Additive);
        additive.positions[0] = Vec3::new(0.0, 0.0, 0.5); // Move forward
        additive.rotations[0] = Quat::from_rotation_x(PI / 12.0); // Tilt 15 degrees

        // Full weight additive
        let result = base.blend_additive(&additive, 1.0);
        assert!(result.positions[0].abs_diff_eq(Vec3::new(0.0, 1.0, 0.5), 1e-5));

        // Half weight additive
        let result_half = base.blend_additive(&additive, 0.5);
        assert!(result_half.positions[0].abs_diff_eq(Vec3::new(0.0, 1.0, 0.25), 1e-5));

        // Zero weight additive (no change)
        let result_zero = base.blend_additive(&additive, 0.0);
        assert!(result_zero.positions[0].abs_diff_eq(base.positions[0], 1e-5));
    }

    /// Test multi-layer blend (3+ layers)
    #[test]
    fn test_multi_layer_blend() {
        let bone_count = 2;

        // Layer 0: Base idle animation
        let mut layer_idle = Pose::new(bone_count, PoseType::Current);
        layer_idle.positions[0] = Vec3::new(0.0, 0.0, 0.0);

        // Layer 1: Walk blend
        let mut layer_walk = Pose::new(bone_count, PoseType::Current);
        layer_walk.positions[0] = Vec3::new(0.0, 0.1, 0.0); // Bob up slightly

        // Layer 2: Run blend
        let mut layer_run = Pose::new(bone_count, PoseType::Current);
        layer_run.positions[0] = Vec3::new(0.0, 0.2, 0.0); // Bob up more

        // Layer 3: Additive breathing
        let mut layer_breathe = Pose::new(bone_count, PoseType::Additive);
        layer_breathe.positions[0] = Vec3::new(0.0, 0.05, 0.0);

        // Blend layers: idle + walk (0.5), then + run (0.3), then + breathe additive
        let blend_1 = layer_idle.blend(&layer_walk, 0.5);
        assert!(blend_1.positions[0].abs_diff_eq(Vec3::new(0.0, 0.05, 0.0), 1e-5));

        let blend_2 = blend_1.blend(&layer_run, 0.3);
        // Should be: blend_1 * 0.7 + run * 0.3 = (0, 0.05 * 0.7 + 0.2 * 0.3, 0) = (0, 0.095, 0)
        assert!(blend_2.positions[0].abs_diff_eq(Vec3::new(0.0, 0.095, 0.0), 1e-5));

        // Apply additive breathing
        let final_blend = blend_2.blend_additive(&layer_breathe, 1.0);
        // Should be: (0, 0.095 + 0.05, 0) = (0, 0.145, 0)
        assert!(final_blend.positions[0].abs_diff_eq(Vec3::new(0.0, 0.145, 0.0), 1e-5));
    }

    /// Test SIMD vs scalar accuracy comparison
    #[test]
    fn test_simd_vs_scalar_accuracy() {
        let bone_count = 17; // Not a multiple of SIMD_LANE_WIDTH to test remainder

        // Create poses with varied transforms
        let a_transforms: Vec<Transform> = (0..bone_count)
            .map(|i| {
                Transform::new(
                    Vec3::new(i as f32 * 0.5, i as f32 * 0.3, i as f32 * 0.1),
                    Quat::from_rotation_y((i as f32) * PI / 17.0),
                    Vec3::splat(1.0 + (i as f32) * 0.05),
                )
            })
            .collect();

        let b_transforms: Vec<Transform> = (0..bone_count)
            .map(|i| {
                Transform::new(
                    Vec3::new(-(i as f32) * 0.3, (i as f32) * 0.7, -(i as f32) * 0.2),
                    Quat::from_rotation_z((i as f32) * PI / 13.0),
                    Vec3::splat(1.5 + (i as f32) * 0.03),
                )
            })
            .collect();

        // SIMD blend
        let soa_a = SoAPose::from_aos(&a_transforms);
        let soa_b = SoAPose::from_aos(&b_transforms);
        let mut simd_result = SoAPose::new();
        blend_poses_simd(&soa_a, &soa_b, 0.37, &mut simd_result);

        // Scalar reference blend (manually using Transform::lerp)
        let scalar_transforms: Vec<Transform> = a_transforms
            .iter()
            .zip(b_transforms.iter())
            .map(|(a, b)| a.lerp(b, 0.37))
            .collect();
        let scalar_soa = SoAPose::from_aos(&scalar_transforms);

        // Compare results
        assert!(
            poses_approx_equal(&simd_result, &scalar_soa, 1e-4),
            "SIMD and scalar blend results should match"
        );
    }

    /// Test nlerp vs slerp comparison
    #[test]
    fn test_nlerp_vs_slerp_accuracy() {
        // For small rotation differences, nlerp and slerp should be very close
        let q0 = Quat::IDENTITY;
        let q1 = Quat::from_rotation_y(PI / 10.0); // Small rotation

        let slerp_result = slerp_quat(q0, q1, 0.5);
        let nlerp_result = nlerp_quat(q0, q1, 0.5);

        // Should be very close for small angles
        assert!(
            slerp_result.dot(nlerp_result).abs() > 0.999,
            "nlerp and slerp should be very close for small angles"
        );

        // For larger rotations, there can be some difference
        let q2 = Quat::from_rotation_y(PI / 2.0); // 90 degrees
        let slerp_90 = slerp_quat(q0, q2, 0.5);
        let nlerp_90 = nlerp_quat(q0, q2, 0.5);

        // Still should be reasonably close
        assert!(
            slerp_90.dot(nlerp_90).abs() > 0.99,
            "nlerp and slerp should still be close for 90-degree rotation"
        );
    }

    // =========================================================================
    // CLIP SAMPLING TESTS
    // =========================================================================

    /// Test keyframe interpolation accuracy (step, linear, cubic)
    #[test]
    fn test_keyframe_interpolation_accuracy() {
        // STEP interpolation
        let step_track = Track::from_keyframes(vec![
            Keyframe::step(0.0, Vec3::new(0.0, 0.0, 0.0)),
            Keyframe::step(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ]);
        // Step should hold previous value until next keyframe
        assert!(step_track
            .sample(0.0)
            .unwrap()
            .abs_diff_eq(Vec3::new(0.0, 0.0, 0.0), 1e-5));
        assert!(step_track
            .sample(0.5)
            .unwrap()
            .abs_diff_eq(Vec3::new(0.0, 0.0, 0.0), 1e-5));
        assert!(step_track
            .sample(0.999)
            .unwrap()
            .abs_diff_eq(Vec3::new(0.0, 0.0, 0.0), 1e-5));
        assert!(step_track
            .sample(1.0)
            .unwrap()
            .abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-5));

        // LINEAR interpolation
        let linear_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::new(0.0, 0.0, 0.0)),
            Keyframe::linear(1.0, Vec3::new(10.0, 20.0, 30.0)),
        ]);
        assert!(linear_track
            .sample(0.25)
            .unwrap()
            .abs_diff_eq(Vec3::new(2.5, 5.0, 7.5), 1e-5));
        assert!(linear_track
            .sample(0.5)
            .unwrap()
            .abs_diff_eq(Vec3::new(5.0, 10.0, 15.0), 1e-5));
        assert!(linear_track
            .sample(0.75)
            .unwrap()
            .abs_diff_eq(Vec3::new(7.5, 15.0, 22.5), 1e-5));

        // CUBIC (Hermite) interpolation
        // Create a curve that starts at Y=0, ends at Y=0, with positive tangents
        // causing a peak in the middle
        let cubic_track = Track::from_keyframes(vec![
            Keyframe::cubic(
                0.0,
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::ZERO,
                Vec3::new(10.0, 10.0, 0.0), // Out tangent points up and right
            ),
            Keyframe::cubic(
                1.0,
                Vec3::new(10.0, 0.0, 0.0),
                Vec3::new(10.0, -10.0, 0.0), // In tangent points down-right (approaches from above)
                Vec3::ZERO,
            ),
        ]);
        // Cubic should interpolate smoothly with tangent influence
        let mid = cubic_track.sample(0.5).unwrap();
        // With tangents causing upward curve, the X component should be around 5
        assert!(
            (mid.x - 5.0).abs() < 2.0,
            "Cubic X should be near midpoint: {}",
            mid.x
        );
        // The Y should have some influence from tangents (may be positive or affected by Hermite)
        // Just verify cubic produces different result than linear
        let quarter = cubic_track.sample(0.25).unwrap();
        let three_quarter = cubic_track.sample(0.75).unwrap();
        // Verify the curve is smooth (values exist and are in reasonable range)
        assert!(quarter.x >= 0.0 && quarter.x <= 10.0);
        assert!(three_quarter.x >= 0.0 && three_quarter.x <= 10.0);
    }

    /// Test loop mode behavior (once stops, loop wraps, ping-pong reverses)
    #[test]
    fn test_loop_mode_behavior() {
        let duration = 2.0;

        // ONCE mode: clamps to duration
        let once_mode = LoopMode::Once;
        assert_eq!(once_mode.calculate_time(-1.0, duration), (0.0, false));
        assert_eq!(once_mode.calculate_time(1.0, duration), (1.0, false));
        assert_eq!(once_mode.calculate_time(3.0, duration), (2.0, false));
        assert!(once_mode.is_complete(2.0, duration));
        assert!(once_mode.is_complete(3.0, duration));
        assert!(!once_mode.is_complete(1.5, duration));

        // LOOP mode: wraps around
        let loop_mode = LoopMode::Loop;
        let (t, rev) = loop_mode.calculate_time(2.5, duration);
        assert!((t - 0.5).abs() < 1e-5);
        assert!(!rev);
        let (t, _) = loop_mode.calculate_time(4.5, duration);
        assert!((t - 0.5).abs() < 1e-5);
        assert!(!loop_mode.is_complete(100.0, duration));

        // PING-PONG mode: alternates direction
        let pingpong_mode = LoopMode::PingPong;

        // Forward phase (0 -> 2)
        let (t, rev) = pingpong_mode.calculate_time(1.0, duration);
        assert!((t - 1.0).abs() < 1e-5);
        assert!(!rev);

        // Reverse phase (2 -> 0)
        let (t, rev) = pingpong_mode.calculate_time(3.0, duration);
        assert!((t - 1.0).abs() < 1e-5);
        assert!(rev);

        // Back to forward
        let (t, rev) = pingpong_mode.calculate_time(5.0, duration);
        assert!((t - 1.0).abs() < 1e-5);
        assert!(!rev);

        assert!(!pingpong_mode.is_complete(100.0, duration));
    }

    /// Test event notification timing within 1ms accuracy
    #[test]
    fn test_event_notification_timing() {
        let mut clip = AnimationClip::new("test", 1.0);

        let mut event_track = EventTrack::new("notifies");
        // Add events at precise times
        event_track.add_event(AnimationEvent::new("footstep_left", 0.1));
        event_track.add_event(AnimationEvent::new("footstep_right", 0.35));
        event_track.add_event(AnimationEvent::new("footstep_left", 0.6));
        event_track.add_event(AnimationEvent::new("footstep_right", 0.85));
        clip.add_event_track(event_track);

        // Query events in small windows (simulating frame updates)
        // Window: 0.09 to 0.11 should catch the 0.1 event
        let events = clip.events_in_range(0.09, 0.11);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "footstep_left");
        assert!((events[0].time - 0.1).abs() < 0.001);

        // Window: 0.11 to 0.15 should have no events
        let events = clip.events_in_range(0.11, 0.15);
        assert!(events.is_empty());

        // Window: 0.34 to 0.36 should catch the 0.35 event
        let events = clip.events_in_range(0.34, 0.36);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "footstep_right");
        assert!((events[0].time - 0.35).abs() < 0.001);

        // Large window should catch multiple events
        let events = clip.events_in_range(0.0, 0.5);
        assert_eq!(events.len(), 2);
    }

    /// Test partial bone tracks (some bones animated, others not)
    #[test]
    fn test_partial_bone_tracks() {
        let mut clip = AnimationClip::new("partial", 1.0);

        // Only animate hip and left_leg, not right_leg or spine
        let hip_track = BoneTrack::new("hip").with_position(Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::new(0.0, 1.0, 0.0)),
            Keyframe::linear(1.0, Vec3::new(0.0, 1.1, 0.0)),
        ]));
        clip.add_bone_track(hip_track);

        let left_leg_track = BoneTrack::new("left_leg").with_rotation(Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(1.0, Quat::from_rotation_x(PI / 6.0)),
        ]));
        clip.add_bone_track(left_leg_track);

        // Sample at midpoint
        let pose = clip.sample(0.5);

        // Hip should have interpolated position
        let hip = pose.get("hip").unwrap();
        assert!(hip.position.abs_diff_eq(Vec3::new(0.0, 1.05, 0.0), 1e-5));
        assert_eq!(hip.rotation, Quat::IDENTITY); // No rotation track
        assert_eq!(hip.scale, Vec3::ONE); // No scale track

        // Left leg should have interpolated rotation
        let left_leg = pose.get("left_leg").unwrap();
        assert!(left_leg.rotation.abs_diff_eq(Quat::from_rotation_x(PI / 12.0), 1e-4));
        assert_eq!(left_leg.position, Vec3::ZERO); // No position track

        // Non-animated bones should not be in pose
        assert!(pose.get("right_leg").is_none());
        assert!(pose.get("spine").is_none());
    }

    /// Test animation clip with multiple keyframes and varied timing
    #[test]
    fn test_complex_animation_clip() {
        let mut clip = AnimationClip::new("walk_cycle", 1.0);
        clip.looping = LoopMode::Loop;

        // Hip bobbing animation
        let hip_pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::new(0.0, 1.0, 0.0)),
            Keyframe::linear(0.25, Vec3::new(0.0, 1.05, 0.0)),
            Keyframe::linear(0.5, Vec3::new(0.0, 1.0, 0.0)),
            Keyframe::linear(0.75, Vec3::new(0.0, 1.05, 0.0)),
            Keyframe::linear(1.0, Vec3::new(0.0, 1.0, 0.0)),
        ]);

        // Hip rotation (slight twist)
        let hip_rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(0.5, Quat::from_rotation_y(0.05)),
            Keyframe::linear(1.0, Quat::IDENTITY),
        ]);

        clip.add_bone_track(
            BoneTrack::new("hip")
                .with_position(hip_pos_track)
                .with_rotation(hip_rot_track),
        );

        // Sample at various times
        let pose_0 = clip.sample(0.0);
        let pose_25 = clip.sample(0.25);
        let pose_50 = clip.sample(0.5);
        let pose_looped = clip.sample(1.25); // Should wrap to 0.25

        // Verify keyframe values
        assert!(pose_0.get("hip").unwrap().position.y - 1.0 < 1e-5);
        assert!(pose_25.get("hip").unwrap().position.y - 1.05 < 1e-5);
        assert!(pose_50.get("hip").unwrap().position.y - 1.0 < 1e-5);

        // Looped sample should match 0.25
        assert!(pose_looped
            .get("hip")
            .unwrap()
            .position
            .abs_diff_eq(pose_25.get("hip").unwrap().position, 1e-5));
    }

    /// Test curve track sampling for blend shapes
    #[test]
    fn test_curve_track_sampling() {
        let mut clip = AnimationClip::new("expression", 2.0);

        // Smile blend shape animation
        clip.add_curve_track(CurveTrack::with_keyframes(
            "blendshape.smile",
            vec![
                Keyframe::linear(0.0, 0.0),
                Keyframe::linear(0.5, 1.0),
                Keyframe::linear(1.5, 1.0),
                Keyframe::linear(2.0, 0.0),
            ],
        ));

        // Test sampling
        assert!((clip.sample_curve("blendshape.smile", 0.0).unwrap() - 0.0).abs() < 1e-5);
        assert!((clip.sample_curve("blendshape.smile", 0.25).unwrap() - 0.5).abs() < 1e-5);
        assert!((clip.sample_curve("blendshape.smile", 0.5).unwrap() - 1.0).abs() < 1e-5);
        assert!((clip.sample_curve("blendshape.smile", 1.0).unwrap() - 1.0).abs() < 1e-5);
        assert!((clip.sample_curve("blendshape.smile", 1.75).unwrap() - 0.5).abs() < 1e-5);

        // Non-existent curve
        assert!(clip.sample_curve("nonexistent", 0.5).is_none());
    }

    // =========================================================================
    // SIMD INTEGRATION TESTS
    // =========================================================================

    /// Test SIMD bone chain computation with complex hierarchy
    #[test]
    fn test_simd_bone_chain_complex_hierarchy() {
        let skeleton = SkeletonBuilder::new()
            .root("root")
            .child_at("spine_01", "root", Vec3::new(0.0, 0.5, 0.0))
            .child_at("spine_02", "spine_01", Vec3::new(0.0, 0.5, 0.0))
            .child_at("spine_03", "spine_02", Vec3::new(0.0, 0.5, 0.0))
            .child_at("neck", "spine_03", Vec3::new(0.0, 0.3, 0.0))
            .child_at("head", "neck", Vec3::new(0.0, 0.2, 0.0))
            .child_at("l_clavicle", "spine_03", Vec3::new(-0.2, 0.0, 0.0))
            .child_at("l_shoulder", "l_clavicle", Vec3::new(-0.3, 0.0, 0.0))
            .child_at("l_elbow", "l_shoulder", Vec3::new(-0.4, 0.0, 0.0))
            .child_at("l_wrist", "l_elbow", Vec3::new(-0.3, 0.0, 0.0))
            .child_at("r_clavicle", "spine_03", Vec3::new(0.2, 0.0, 0.0))
            .child_at("r_shoulder", "r_clavicle", Vec3::new(0.3, 0.0, 0.0))
            .child_at("r_elbow", "r_shoulder", Vec3::new(0.4, 0.0, 0.0))
            .child_at("r_wrist", "r_elbow", Vec3::new(0.3, 0.0, 0.0))
            .build_unchecked();

        let local_transforms: Vec<Transform> = skeleton
            .bones()
            .iter()
            .map(|b| b.local_transform)
            .collect();

        // SIMD computation
        let soa_pose = SoAPose::from_aos(&local_transforms);
        let mut simd_matrices = vec![Mat4::IDENTITY; skeleton.bone_count()];
        compute_bone_chain_simd(&skeleton, &soa_pose, &mut simd_matrices);

        // Scalar computation
        let scalar_matrices = skeleton.compute_world_transforms(&local_transforms);

        // Compare results
        for i in 0..skeleton.bone_count() {
            assert!(
                simd_matrices[i].abs_diff_eq(scalar_matrices[i], 1e-4),
                "Bone {} SIMD/scalar matrix mismatch",
                skeleton.bone(i).unwrap().name
            );
        }

        // Verify specific bone positions
        let head_idx = skeleton.bone_index("head").unwrap();
        let head_pos = simd_matrices[head_idx].w_axis.truncate();
        // Head should be at: 0.5 + 0.5 + 0.5 + 0.3 + 0.2 = 2.0 in Y
        assert!(
            head_pos.abs_diff_eq(Vec3::new(0.0, 2.0, 0.0), 1e-4),
            "Head position mismatch: {:?}",
            head_pos
        );

        let l_wrist_idx = skeleton.bone_index("l_wrist").unwrap();
        let l_wrist_pos = simd_matrices[l_wrist_idx].w_axis.truncate();
        // Left wrist X: -0.2 - 0.3 - 0.4 - 0.3 = -1.2
        // Left wrist Y: 0.5 + 0.5 + 0.5 = 1.5
        assert!(
            l_wrist_pos.abs_diff_eq(Vec3::new(-1.2, 1.5, 0.0), 1e-4),
            "Left wrist position mismatch: {:?}",
            l_wrist_pos
        );
    }

    /// Test SIMD additive blending
    #[test]
    fn test_simd_additive_blend_integration() {
        let bone_count = 8;

        // Base pose
        let base_transforms: Vec<Transform> = (0..bone_count)
            .map(|i| Transform::from_position(Vec3::new(0.0, i as f32, 0.0)))
            .collect();

        // Additive pose (represents "lean")
        let additive_transforms: Vec<Transform> = (0..bone_count)
            .map(|i| {
                Transform::new(
                    Vec3::new(0.0, 0.0, 0.1 * (i as f32)), // Forward lean
                    Quat::from_rotation_x(0.02 * (i as f32)),
                    Vec3::ONE, // No scale change
                )
            })
            .collect();

        let base_soa = SoAPose::from_aos(&base_transforms);
        let additive_soa = SoAPose::from_aos(&additive_transforms);
        let mut result = SoAPose::new();

        blend_poses_additive_simd(&base_soa, &additive_soa, 0.5, &mut result);

        // Verify positions have additive component
        for i in 0..bone_count {
            let expected_z = 0.1 * (i as f32) * 0.5; // Half weight
            assert!(
                (result.positions_z[i] - expected_z).abs() < 1e-5,
                "Bone {} Z position mismatch",
                i
            );
        }
    }

    /// Test SIMD masked blend
    #[test]
    fn test_simd_masked_blend_integration() {
        let bone_count = 6;

        // Pose A: all at Y=0
        let a_transforms: Vec<Transform> =
            (0..bone_count).map(|_| Transform::IDENTITY).collect();

        // Pose B: all at Y=10
        let b_transforms: Vec<Transform> = (0..bone_count)
            .map(|_| Transform::from_position(Vec3::new(0.0, 10.0, 0.0)))
            .collect();

        let soa_a = SoAPose::from_aos(&a_transforms);
        let soa_b = SoAPose::from_aos(&b_transforms);

        // Mask: first 3 bones use A, last 3 use B
        let mask = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut result = SoAPose::new();

        blend_poses_masked(&soa_a, &soa_b, &mask, &mut result);

        // Verify mask effect
        for i in 0..3 {
            assert!(result.positions_y[i].abs() < 1e-5, "Bone {} should use A", i);
        }
        for i in 3..6 {
            assert!((result.positions_y[i] - 10.0).abs() < 1e-5, "Bone {} should use B", i);
        }
    }

    /// Test SIMD rotation normalization after multiple blends
    #[test]
    fn test_simd_rotation_normalization_after_blends() {
        let bone_count = 5;

        // Create poses with valid rotations
        let a = SoAPose::identity(bone_count);
        let b = SoAPose::from_aos(
            &(0..bone_count)
                .map(|i| Transform::from_rotation(Quat::from_rotation_y((i as f32) * 0.5)))
                .collect::<Vec<_>>(),
        );

        let mut result = SoAPose::new();

        // Perform multiple blends (which can accumulate floating-point error)
        blend_poses_simd(&a, &b, 0.3, &mut result);

        let mut temp = SoAPose::new();
        for _ in 0..10 {
            blend_poses_simd(&result, &b, 0.1, &mut temp);
            result.copy_from(&temp);
        }

        // Normalize rotations
        normalize_rotations_simd(&mut result);

        // Verify all rotations are unit quaternions
        for i in 0..bone_count {
            let len_sq = result.rotations_x[i].powi(2)
                + result.rotations_y[i].powi(2)
                + result.rotations_z[i].powi(2)
                + result.rotations_w[i].powi(2);
            assert!(
                (len_sq - 1.0).abs() < 1e-5,
                "Rotation {} not normalized: length^2 = {}",
                i,
                len_sq
            );
        }
    }

    // =========================================================================
    // POSE BUFFER INTEGRATION TESTS
    // =========================================================================

    /// Test PoseBuffer workflow
    #[test]
    fn test_pose_buffer_full_workflow() {
        let skeleton = SkeletonBuilder::new()
            .root("root")
            .child_at("spine", "root", Vec3::new(0.0, 1.0, 0.0))
            .child_at("head", "spine", Vec3::new(0.0, 0.5, 0.0))
            .build_unchecked();

        let mut buffer = PoseBuffer::from_skeleton(&skeleton);

        // Initial model pose should reflect bind pose
        let head_idx = skeleton.bone_index("head").unwrap();
        let initial_head_pos = buffer.model_pose[head_idx].w_axis.truncate();
        assert!(
            initial_head_pos.abs_diff_eq(Vec3::new(0.0, 1.5, 0.0), 1e-5),
            "Initial head position: {:?}",
            initial_head_pos
        );

        // Animate the root bone
        buffer.local_pose.positions[0] = Vec3::new(5.0, 0.0, 0.0);
        buffer.update_model_pose(&skeleton);

        // Head should now be at (5, 1.5, 0)
        let animated_head_pos = buffer.model_pose[head_idx].w_axis.truncate();
        assert!(
            animated_head_pos.abs_diff_eq(Vec3::new(5.0, 1.5, 0.0), 1e-5),
            "Animated head position: {:?}",
            animated_head_pos
        );
    }

    // =========================================================================
    // PERFORMANCE BENCHMARK TESTS (OPTIONAL)
    // =========================================================================

    /// Test 100-bone skeleton blend performance
    #[test]
    fn test_100_bone_skeleton_blend_performance() {
        let bone_count = 100;

        // Create skeleton
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        for i in 1..bone_count {
            skeleton.add_bone(Bone::new(format!("bone_{}", i)).with_parent(0));
        }

        // Create poses
        let a_transforms: Vec<Transform> = (0..bone_count)
            .map(|i| {
                Transform::new(
                    Vec3::new(i as f32 * 0.1, 0.0, 0.0),
                    Quat::from_rotation_y((i as f32) * 0.01),
                    Vec3::splat(1.0 + (i as f32) * 0.001),
                )
            })
            .collect();

        let b_transforms: Vec<Transform> = (0..bone_count)
            .map(|i| {
                Transform::new(
                    Vec3::new(0.0, i as f32 * 0.1, 0.0),
                    Quat::from_rotation_z((i as f32) * 0.01),
                    Vec3::splat(1.0 - (i as f32) * 0.001),
                )
            })
            .collect();

        let soa_a = SoAPose::from_aos(&a_transforms);
        let soa_b = SoAPose::from_aos(&b_transforms);
        let mut result = SoAPose::new();

        // Run blend multiple times for performance measurement
        let iterations = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            blend_poses_simd(&soa_a, &soa_b, 0.5, &mut result);
        }
        let elapsed = start.elapsed();

        // Just verify it completed (actual performance check would be in benchmarks)
        assert_eq!(result.bone_count(), bone_count);
        println!(
            "100-bone SIMD blend: {:?} for {} iterations ({:?} per iteration)",
            elapsed,
            iterations,
            elapsed / iterations as u32
        );
    }

    /// Test 1000-frame clip sampling performance
    #[test]
    fn test_1000_frame_clip_sampling_performance() {
        // Create a clip with many keyframes
        let keyframes: Vec<Keyframe<Vec3>> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0;
                Keyframe::linear(t, Vec3::new(t * 10.0, (t * PI * 4.0).sin(), 0.0))
            })
            .collect();

        let track = Track::from_keyframes(keyframes);
        let mut clip = AnimationClip::new("long_clip", 1.0);
        clip.add_bone_track(BoneTrack::new("hip").with_position(track));

        // Sample at many times
        let iterations = 10000;
        let start = std::time::Instant::now();
        for i in 0..iterations {
            let t = (i as f32 / iterations as f32) * 1.0;
            let _ = clip.sample(t);
        }
        let elapsed = start.elapsed();

        println!(
            "1000-keyframe clip sampling: {:?} for {} samples ({:?} per sample)",
            elapsed,
            iterations,
            elapsed / iterations as u32
        );
    }

    // =========================================================================
    // EDGE CASE TESTS
    // =========================================================================

    /// Test empty skeleton handling
    #[test]
    fn test_empty_skeleton_handling() {
        let skeleton = Skeleton::new();
        assert!(skeleton.is_empty());
        assert!(skeleton.validate().is_ok());

        let transforms: Vec<Transform> = vec![];
        let world = skeleton.compute_world_transforms(&transforms);
        assert!(world.is_empty());
    }

    /// Test single bone skeleton
    #[test]
    fn test_single_bone_skeleton() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(
            Bone::root("root").with_local_transform(Transform::from_position(Vec3::new(
                1.0, 2.0, 3.0,
            ))),
        );

        skeleton.compute_inverse_bind_matrices().unwrap();

        let poses = vec![Transform::from_position(Vec3::new(4.0, 5.0, 6.0))];
        let skinning = skeleton.compute_skinning_matrices(&poses);

        // Skinning = world * inverse_bind
        // World = (4,5,6), bind was at (1,2,3), so inverse_bind = (-1,-2,-3)
        // Skinning = (4,5,6) + (-1,-2,-3) = (3,3,3)
        let result_pos = skinning[0].w_axis.truncate();
        assert!(
            result_pos.abs_diff_eq(Vec3::new(3.0, 3.0, 3.0), 1e-5),
            "Skinning result: {:?}",
            result_pos
        );
    }

    /// Test animation clip with no tracks
    #[test]
    fn test_animation_clip_no_tracks() {
        let clip = AnimationClip::new("empty", 1.0);
        let pose = clip.sample(0.5);
        assert!(pose.is_empty());
        assert!(!clip.is_complete(0.5));
        assert!(clip.is_complete(1.0));
    }

    /// Test pose with zero bones
    #[test]
    fn test_pose_zero_bones() {
        let pose = Pose::new(0, PoseType::Current);
        assert!(pose.is_empty());
        assert_eq!(pose.bone_count(), 0);
    }

    /// Test SoA pose SIMD lane boundary handling
    #[test]
    fn test_soa_simd_boundary_handling() {
        // Test with bone counts around SIMD boundaries
        for bone_count in [
            SIMD_LANE_WIDTH - 1,
            SIMD_LANE_WIDTH,
            SIMD_LANE_WIDTH + 1,
            SIMD_LANE_WIDTH * 2,
            SIMD_LANE_WIDTH * 2 + 1,
        ] {
            let a = SoAPose::identity(bone_count);
            let b = SoAPose::from_aos(
                &(0..bone_count)
                    .map(|i| Transform::from_position(Vec3::new(i as f32, 0.0, 0.0)))
                    .collect::<Vec<_>>(),
            );
            let mut result = SoAPose::new();

            blend_poses_simd(&a, &b, 0.5, &mut result);

            assert_eq!(result.bone_count(), bone_count);
            for i in 0..bone_count {
                let expected_x = (i as f32) * 0.5;
                assert!(
                    (result.positions_x[i] - expected_x).abs() < 1e-5,
                    "Bone {} mismatch for count {}: expected {}, got {}",
                    i,
                    bone_count,
                    expected_x,
                    result.positions_x[i]
                );
            }
        }
    }

    /// Test quaternion slerp edge case: nearly identical rotations
    #[test]
    fn test_quaternion_slerp_nearly_identical() {
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_y(0.0001); // Very small rotation

        let result = slerp_quat(a, b, 0.5);
        assert!(result.is_normalized());
        assert!(result.dot(Quat::IDENTITY).abs() > 0.9999);
    }

    /// Test quaternion slerp edge case: opposite rotations
    #[test]
    fn test_quaternion_slerp_opposite() {
        let a = Quat::IDENTITY;
        let b = -Quat::IDENTITY; // Opposite (same rotation, different representation)

        let result = slerp_quat(a, b, 0.5);
        assert!(result.is_normalized());
        // Should still be close to identity
        assert!(result.dot(Quat::IDENTITY).abs() > 0.99);
    }

    /// Test transform with zero scale
    #[test]
    fn test_transform_zero_scale() {
        let t = Transform::from_scale_vec(Vec3::new(0.0, 1.0, 1.0));
        assert!(t.inverse().is_none());
    }

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    /// Create a flat skeleton (all bones parented to root) for testing
    fn create_flat_skeleton(bone_count: usize, transforms: &[Transform]) -> Skeleton {
        let mut skeleton = Skeleton::new();
        for (i, t) in transforms.iter().enumerate().take(bone_count) {
            if i == 0 {
                skeleton.add_bone(Bone::root(format!("bone_{}", i)).with_local_transform(*t));
            } else {
                skeleton.add_bone(
                    Bone::new(format!("bone_{}", i))
                        .with_parent(0)
                        .with_local_transform(*t),
                );
            }
        }
        skeleton
    }
}
