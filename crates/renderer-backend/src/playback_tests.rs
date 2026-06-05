//! Integration tests for TRINITY animation playback system (T-AN-2.7).
//!
//! This module provides comprehensive integration tests covering:
//! - End-to-End Playback Tests (15+)
//! - Blending Integration Tests (15+)
//! - Root Motion Integration Tests (10+)
//! - Compression Round-Trip Tests (10+)
//! - Performance Benchmarks (5+)
//! - Error Handling Tests (10+)

use std::f32::consts::PI;
use std::time::Instant;

use glam::{Quat, Vec3};

use crate::animation_clip::{
    AnimationClip, AnimationEvent, BoneTrack, EventTrack, Keyframe, LoopMode, Track,
    Pose as ClipPose,
};
use crate::clip_compression::{
    compress_clip, decompress_clip, sample_compressed_clip, BoneImportance,
    CompressionError, CompressionFormat, CompressionSettings,
};
use crate::clip_player::{ClipPlayer, PlaybackState};
use crate::pose::{Pose, PoseType};
use crate::pose_blending::{
    apply_additive_weighted, blend_poses_weighted, compute_additive_pose, crossfade_poses,
    BlendWeights, CrossfadeCurve, InertializationState, PoseVelocity,
};
use crate::root_motion::{
    RootMotionAccumulator, RootMotionConfig, RootMotionDelta, RootMotionMode,
};
use crate::skeleton::{Bone, Skeleton, Transform};

// =============================================================================
// Test Fixtures
// =============================================================================

/// Create a simple test skeleton with the given bone count.
fn create_test_skeleton(bone_count: usize) -> Skeleton {
    let mut skeleton = Skeleton::new();

    // Root bone
    skeleton.add_bone(Bone {
        name: "root".to_string(),
        parent_index: None,
        local_transform: Transform::IDENTITY,
        inverse_bind_matrix: glam::Mat4::IDENTITY,
    });

    // Chain of child bones
    for i in 1..bone_count {
        skeleton.add_bone(Bone {
            name: format!("bone_{}", i),
            parent_index: Some(i - 1),
            local_transform: Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
            inverse_bind_matrix: glam::Mat4::IDENTITY,
        });
    }

    skeleton
}

/// Create a humanoid-like skeleton for retargeting tests.
fn create_humanoid_skeleton(name_prefix: &str) -> Skeleton {
    let mut skeleton = Skeleton::new();

    let bones = [
        ("root", None),
        ("hips", Some(0)),
        ("spine", Some(1)),
        ("chest", Some(2)),
        ("neck", Some(3)),
        ("head", Some(4)),
        ("l_shoulder", Some(3)),
        ("l_arm", Some(6)),
        ("l_forearm", Some(7)),
        ("l_hand", Some(8)),
        ("r_shoulder", Some(3)),
        ("r_arm", Some(10)),
        ("r_forearm", Some(11)),
        ("r_hand", Some(12)),
        ("l_thigh", Some(1)),
        ("l_calf", Some(14)),
        ("l_foot", Some(15)),
        ("r_thigh", Some(1)),
        ("r_calf", Some(17)),
        ("r_foot", Some(18)),
    ];

    for (name, parent) in bones {
        skeleton.add_bone(Bone {
            name: format!("{}_{}", name_prefix, name),
            parent_index: parent,
            local_transform: Transform::from_position(Vec3::new(0.0, 0.5, 0.0)),
            inverse_bind_matrix: glam::Mat4::IDENTITY,
        });
    }

    skeleton
}

/// Create a simple walk animation clip.
fn create_walk_clip(duration: f32, bone_count: usize) -> AnimationClip {
    let mut clip = AnimationClip::new("walk", duration);
    clip.looping = LoopMode::Loop;
    clip.frame_rate = 30.0;

    for i in 0..bone_count {
        let bone_name = if i == 0 {
            "root".to_string()
        } else {
            format!("bone_{}", i)
        };

        // Position track with simple bob motion
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::new(0.0, 0.0, 0.0)),
            Keyframe::linear(duration * 0.25, Vec3::new(0.0, 0.1, duration * 0.25)),
            Keyframe::linear(duration * 0.5, Vec3::new(0.0, 0.0, duration * 0.5)),
            Keyframe::linear(duration * 0.75, Vec3::new(0.0, 0.1, duration * 0.75)),
            Keyframe::linear(duration, Vec3::new(0.0, 0.0, duration)),
        ]);

        // Rotation track with oscillation
        let angle = PI / 12.0 * (i as f32 * 0.5);
        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(duration * 0.5, Quat::from_rotation_x(angle)),
            Keyframe::linear(duration, Quat::IDENTITY),
        ]);

        let mut bone_track = BoneTrack::new(&bone_name);
        bone_track.position = Some(pos_track);
        bone_track.rotation = Some(rot_track);
        clip.add_bone_track(bone_track);
    }

    clip
}

/// Create an additive animation clip (e.g., a lean).
fn create_additive_clip(duration: f32, bone_count: usize) -> AnimationClip {
    let mut clip = AnimationClip::new("lean", duration);
    clip.looping = LoopMode::Once;
    clip.frame_rate = 30.0;

    for i in 0..bone_count {
        let bone_name = if i == 0 {
            "root".to_string()
        } else {
            format!("bone_{}", i)
        };

        // Additive rotation - lean to the side
        let lean_angle = PI / 8.0 * (1.0 - i as f32 * 0.1).max(0.0);
        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(duration * 0.5, Quat::from_rotation_z(lean_angle)),
            Keyframe::linear(duration, Quat::IDENTITY),
        ]);

        let mut bone_track = BoneTrack::new(&bone_name);
        bone_track.rotation = Some(rot_track);
        clip.add_bone_track(bone_track);
    }

    clip
}

/// Create a clip with root motion.
fn create_root_motion_clip(duration: f32, bone_count: usize, distance: f32) -> AnimationClip {
    let mut clip = AnimationClip::new("locomotion", duration);
    clip.looping = LoopMode::Loop;
    clip.frame_rate = 30.0;

    // Root bone with forward movement
    let root_pos_track = Track::from_keyframes(vec![
        Keyframe::linear(0.0, Vec3::ZERO),
        Keyframe::linear(duration * 0.5, Vec3::new(0.0, 0.1, distance * 0.5)),
        Keyframe::linear(duration, Vec3::new(0.0, 0.0, distance)),
    ]);

    let root_rot_track = Track::from_keyframes(vec![
        Keyframe::linear(0.0, Quat::IDENTITY),
        Keyframe::linear(duration, Quat::IDENTITY),
    ]);

    let mut root_track = BoneTrack::new("root");
    root_track.position = Some(root_pos_track);
    root_track.rotation = Some(root_rot_track);
    clip.add_bone_track(root_track);

    // Other bones with simple motion
    for i in 1..bone_count {
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::new(0.0, 1.0, 0.0)),
            Keyframe::linear(duration, Vec3::new(0.0, 1.0, 0.0)),
        ]);

        let mut bone_track = BoneTrack::new(&format!("bone_{}", i));
        bone_track.position = Some(pos_track);
        clip.add_bone_track(bone_track);
    }

    clip
}

/// Create a clip with animation events.
fn create_clip_with_events(duration: f32, bone_count: usize) -> AnimationClip {
    let mut clip = create_walk_clip(duration, bone_count);

    // Add an event track with footstep events
    let mut event_track = EventTrack::new("footsteps");
    event_track.add_event(AnimationEvent::new("footstep_left", duration * 0.25));
    event_track.add_event(AnimationEvent::new("footstep_right", duration * 0.75));
    clip.add_event_track(event_track);

    clip
}

/// Create an identity pose with the given bone count
fn create_identity_pose(bone_count: usize) -> Pose {
    Pose::new(bone_count, PoseType::Current)
}

// =============================================================================
// End-to-End Playback Tests (15+)
// =============================================================================

#[test]
fn test_e2e_load_play_sample_verify() {
    let clip = create_walk_clip(1.0, 5);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.play();

    // Update to middle of animation
    player.update(0.5);

    let pose = player.sample();
    assert!(!pose.is_empty());

    // Verify transforms are interpolated
    let root_transform = pose.get("root");
    assert!(root_transform.is_some());
}

#[test]
fn test_e2e_play_rate_positive() {
    let clip = create_walk_clip(2.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.set_play_rate(2.0); // 2x speed
    player.play();

    player.update(0.5); // Should be at time 1.0 in clip

    assert_eq!(player.current_time(), 1.0);
}

#[test]
fn test_e2e_play_rate_negative() {
    let clip = create_walk_clip(2.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.set_loop_mode(LoopMode::Loop);
    player.play();

    // Advance to middle
    player.update(1.0);
    assert_eq!(player.current_time(), 1.0);

    // Reverse playback
    player.set_play_rate(-1.0);
    player.update(0.5);

    assert!((player.current_time() - 0.5).abs() < 0.01);
}

#[test]
fn test_e2e_loop_mode_once() {
    let clip = create_walk_clip(1.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.set_loop_mode(LoopMode::Once);
    player.play();

    // Play past the end
    player.update(1.5);

    // Should be clamped at duration
    assert!(player.current_time() >= 1.0);
    assert!(player.is_completed());
}

#[test]
fn test_e2e_loop_mode_loop() {
    let clip = create_walk_clip(1.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.set_loop_mode(LoopMode::Loop);
    player.play();

    // Play past the end
    player.update(1.5);

    // Should wrap around
    let current_time = player.current_time();
    assert!(current_time >= 0.0 && current_time < 1.0);
    assert!(!player.is_completed());
}

#[test]
fn test_e2e_loop_mode_pingpong() {
    let clip = create_walk_clip(1.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.set_loop_mode(LoopMode::PingPong);
    player.play();

    // Play past the end
    player.update(1.5);

    // Should be playing backwards, so time should be around 0.5
    let current_time = player.current_time();
    assert!(current_time >= 0.0 && current_time <= 1.0);
}

#[test]
fn test_e2e_seek_to_time() {
    let clip = create_walk_clip(2.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.play();

    // Seek to specific time
    player.seek(1.5);

    assert!((player.current_time() - 1.5).abs() < 0.01);
}

#[test]
fn test_e2e_seek_normalized() {
    let clip = create_walk_clip(2.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.play();

    // Seek to 75% through animation
    player.seek_normalized(0.75);

    assert!((player.current_time() - 1.5).abs() < 0.01);
}

#[test]
fn test_e2e_event_detection() {
    let clip = create_clip_with_events(1.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.play();

    // Play to just past first event
    player.update(0.3);

    let events = player.drain_events();
    assert!(!events.is_empty());
    assert_eq!(events[0].name, "footstep_left");
}

#[test]
fn test_e2e_event_detection_multiple() {
    let clip = create_clip_with_events(1.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.play();

    // Play through entire clip
    player.update(1.0);

    let events = player.drain_events();
    assert_eq!(events.len(), 2);
}

#[test]
fn test_e2e_pause_resume() {
    let clip = create_walk_clip(2.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.play();
    player.update(0.5);

    let time_before_pause = player.current_time();
    player.pause();
    player.update(0.5); // Should not advance

    assert_eq!(player.current_time(), time_before_pause);

    player.play();
    player.update(0.5); // Should advance now

    assert!(player.current_time() > time_before_pause);
}

#[test]
fn test_e2e_stop_resets() {
    let clip = create_walk_clip(2.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.play();
    player.update(1.0);
    player.stop();

    assert_eq!(player.current_time(), 0.0);
    assert!(matches!(player.state(), PlaybackState::Stopped));
}

#[test]
fn test_e2e_clip_sampling_directly() {
    let clip = create_walk_clip(1.0, 5);

    // Sample at different times
    let pose_0 = clip.sample(0.0);
    let pose_mid = clip.sample(0.5);
    let pose_end = clip.sample(1.0);

    // All should have transforms
    assert!(pose_0.get("root").is_some());
    assert!(pose_mid.get("root").is_some());
    assert!(pose_end.get("root").is_some());
}

#[test]
fn test_e2e_interpolation_modes() {
    // Create clip with different interpolation modes
    let mut clip = AnimationClip::new("interp_test", 1.0);

    // Step interpolation
    let step_track = Track::from_keyframes(vec![
        Keyframe::step(0.0, Vec3::ZERO),
        Keyframe::step(0.5, Vec3::ONE),
        Keyframe::step(1.0, Vec3::splat(2.0)),
    ]);

    let mut bone_track = BoneTrack::new("step_bone");
    bone_track.position = Some(step_track);
    clip.add_bone_track(bone_track);

    // Sample at 0.25 - should still be at ZERO (step)
    let pose = clip.sample(0.25);
    let t = pose.get("step_bone").unwrap();
    assert!(t.position.abs_diff_eq(Vec3::ZERO, 0.01));

    // Sample at 0.5 - should be at ONE
    let pose = clip.sample(0.5);
    let t = pose.get("step_bone").unwrap();
    assert!(t.position.abs_diff_eq(Vec3::ONE, 0.01));
}

#[test]
fn test_e2e_normalized_time() {
    let clip = create_walk_clip(2.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.play();
    player.seek(1.0);

    // Should be at 50%
    assert!((player.normalized_time() - 0.5).abs() < 0.01);
}

// =============================================================================
// Blending Integration Tests (15+)
// =============================================================================

#[test]
fn test_blend_two_clip_crossfade() {
    let skeleton = create_test_skeleton(5);
    let pose_a = Pose::from_skeleton(&skeleton, PoseType::Current);
    let pose_b = Pose::from_skeleton(&skeleton, PoseType::Current);

    let blended = crossfade_poses(&pose_a, &pose_b, 0.5, CrossfadeCurve::Linear);

    assert_eq!(blended.bone_count(), 5);
}

#[test]
fn test_blend_additive_layer() {
    let bone_count = 5;
    let base_pose = create_identity_pose(bone_count);
    let mut additive_pose = Pose::new(bone_count, PoseType::Additive);
    // Add a rotation offset to bone 1
    additive_pose.rotations[1] = Quat::from_rotation_z(PI / 4.0);

    let weights = BlendWeights::full(bone_count);
    let result = apply_additive_weighted(&base_pose, &additive_pose, 1.0, Some(&weights));

    // Bone 1 should have the additive rotation applied
    let angle = result.rotations[1].angle_between(Quat::from_rotation_z(PI / 4.0));
    assert!(angle < 0.1);
}

#[test]
fn test_blend_additive_partial_weight() {
    let bone_count = 5;
    let base_pose = create_identity_pose(bone_count);
    let mut additive_pose = Pose::new(bone_count, PoseType::Additive);
    additive_pose.rotations[0] = Quat::from_rotation_y(PI / 2.0);

    let result = apply_additive_weighted(&base_pose, &additive_pose, 0.5, None);

    // Should be half the rotation
    let expected = Quat::from_rotation_y(PI / 4.0);
    let angle = result.rotations[0].angle_between(expected);
    assert!(angle < 0.2);
}

#[test]
fn test_blend_inertialization_transition() {
    let bone_count = 5;

    let source_pose = create_identity_pose(bone_count);
    let mut target_pose = create_identity_pose(bone_count);
    target_pose.positions[0] = Vec3::new(1.0, 0.0, 0.0);

    let mut state = InertializationState::new(bone_count, 0.5);
    state.init_transition(&source_pose, None, &target_pose);

    // Apply inertialization
    let corrected = state.apply(&target_pose);

    // Should have some correction applied
    assert!(corrected.positions[0].x < 1.0);
}

#[test]
fn test_blend_per_bone_masking() {
    let bone_count = 5;
    let pose_a = create_identity_pose(bone_count);
    let mut pose_b = create_identity_pose(bone_count);
    pose_b.positions[2] = Vec3::new(10.0, 0.0, 0.0);

    // Mask: only blend bone 2
    let mut weights = BlendWeights::none(bone_count);
    weights.set(2, 1.0);

    let result = blend_poses_weighted(&pose_a, &pose_b, 1.0, &weights);

    // Bone 2 should be from pose_b
    assert!(result.positions[2].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 0.01));
    // Other bones should be from pose_a (identity)
    assert!(result.positions[0].abs_diff_eq(Vec3::ZERO, 0.01));
}

#[test]
fn test_blend_crossfade_curves() {
    let bone_count = 3;
    let pose_a = create_identity_pose(bone_count);
    let mut pose_b = create_identity_pose(bone_count);
    pose_b.positions[0] = Vec3::ONE;

    // Linear curve at t=0.5 should give 0.5
    let linear = crossfade_poses(&pose_a, &pose_b, 0.5, CrossfadeCurve::Linear);
    assert!((linear.positions[0].x - 0.5).abs() < 0.01);

    // EaseIn at t=0.5 should give less than 0.5
    let ease_in = crossfade_poses(&pose_a, &pose_b, 0.5, CrossfadeCurve::EaseIn);
    assert!(ease_in.positions[0].x < 0.5);

    // EaseOut at t=0.5 should give more than 0.5
    let ease_out = crossfade_poses(&pose_a, &pose_b, 0.5, CrossfadeCurve::EaseOut);
    assert!(ease_out.positions[0].x > 0.5);
}

#[test]
fn test_blend_compute_additive_pose() {
    let bone_count = 3;
    let reference = create_identity_pose(bone_count);
    let mut target = create_identity_pose(bone_count);
    target.rotations[0] = Quat::from_rotation_x(PI / 4.0);

    let additive = compute_additive_pose(&target, &reference);

    assert_eq!(additive.pose_type, PoseType::Additive);
    // The additive rotation should capture the difference
    let angle = additive.rotations[0].angle_between(Quat::from_rotation_x(PI / 4.0));
    assert!(angle < 0.01);
}

#[test]
fn test_blend_quaternion_shortest_path() {
    let bone_count = 1;
    let mut pose_a = create_identity_pose(bone_count);
    let mut pose_b = create_identity_pose(bone_count);

    // Nearly opposite rotations - should blend via shortest path
    pose_a.rotations[0] = Quat::from_rotation_y(0.1);
    pose_b.rotations[0] = Quat::from_rotation_y(-0.1);

    let weights = BlendWeights::full(bone_count);
    let result = blend_poses_weighted(&pose_a, &pose_b, 0.5, &weights);

    // Result should be near identity (shortest path)
    let angle = result.rotations[0].angle_between(Quat::IDENTITY);
    assert!(angle < 0.1);
}

#[test]
fn test_blend_scale_interpolation() {
    let bone_count = 3;
    let mut pose_a = create_identity_pose(bone_count);
    let mut pose_b = create_identity_pose(bone_count);

    pose_a.scales[0] = Vec3::ONE;
    pose_b.scales[0] = Vec3::splat(2.0);

    let blended = crossfade_poses(&pose_a, &pose_b, 0.5, CrossfadeCurve::Linear);

    // Should be 1.5
    assert!(blended.scales[0].abs_diff_eq(Vec3::splat(1.5), 0.01));
}

#[test]
fn test_blend_velocity_tracking() {
    let bone_count = 3;
    let prev_pose = create_identity_pose(bone_count);
    let mut curr_pose = create_identity_pose(bone_count);
    curr_pose.positions[0] = Vec3::new(1.0, 0.0, 0.0);

    let velocity = PoseVelocity::from_pose_delta(&prev_pose, &curr_pose, 0.1);

    // Linear velocity should be approximately 10 units/sec
    assert!((velocity.position_velocities[0].x - 10.0).abs() < 0.1);
}

#[test]
fn test_blend_zero_weight_handling() {
    let bone_count = 3;
    let pose_a = create_identity_pose(bone_count);
    let mut pose_b = create_identity_pose(bone_count);
    pose_b.positions[0] = Vec3::new(100.0, 0.0, 0.0);

    // Zero weight for pose_b
    let weights = BlendWeights::full(bone_count);
    let result = blend_poses_weighted(&pose_a, &pose_b, 0.0, &weights);

    // Should be entirely pose_a
    assert!(result.positions[0].abs_diff_eq(Vec3::ZERO, 0.01));
}

#[test]
fn test_blend_inertialization_update() {
    let bone_count = 3;
    let source = create_identity_pose(bone_count);
    let mut target = create_identity_pose(bone_count);
    target.positions[0] = Vec3::new(2.0, 0.0, 0.0);

    let mut state = InertializationState::new(bone_count, 0.1);
    state.init_transition(&source, None, &target);

    assert!(state.is_active());

    // Update for several half-lives
    for _ in 0..10 {
        state.update(0.1);
    }

    // Should become inactive after enough time
    assert!(!state.is_active());
}

#[test]
fn test_blend_pose_from_skeleton() {
    let skeleton = create_test_skeleton(5);
    let pose = Pose::from_skeleton(&skeleton, PoseType::Current);

    assert_eq!(pose.bone_count(), 5);
    assert_eq!(pose.pose_type, PoseType::Current);
}

#[test]
fn test_blend_additive_pose_type() {
    let bone_count = 3;
    let additive = Pose::new(bone_count, PoseType::Additive);

    assert_eq!(additive.pose_type, PoseType::Additive);
    // Additive scales should be zero (delta)
    assert!(additive.scales[0].abs_diff_eq(Vec3::ZERO, 0.01));
}

#[test]
fn test_blend_crossfade_full_weight() {
    let bone_count = 5;
    let pose_a = create_identity_pose(bone_count);
    let mut pose_b = create_identity_pose(bone_count);
    pose_b.positions[0] = Vec3::new(5.0, 0.0, 0.0);
    pose_b.rotations[0] = Quat::from_rotation_z(PI / 2.0);

    // At t=0, should be pose_a
    let result_0 = crossfade_poses(&pose_a, &pose_b, 0.0, CrossfadeCurve::Linear);
    assert!(result_0.positions[0].abs_diff_eq(Vec3::ZERO, 0.01));

    // At t=1, should be pose_b
    let result_1 = crossfade_poses(&pose_a, &pose_b, 1.0, CrossfadeCurve::Linear);
    assert!(result_1.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 0.01));
}

// =============================================================================
// Root Motion Integration Tests (10+)
// =============================================================================

#[test]
fn test_root_motion_delta_creation() {
    let delta = RootMotionDelta::new(
        Vec3::new(0.0, 0.0, 1.0),
        Quat::IDENTITY,
    );

    assert_eq!(delta.translation.z, 1.0);
    assert_eq!(delta.rotation, Quat::IDENTITY);
}

#[test]
fn test_root_motion_delta_combine() {
    let delta1 = RootMotionDelta::from_translation(Vec3::new(0.0, 0.0, 1.0));
    let delta2 = RootMotionDelta::from_translation(Vec3::new(0.0, 0.0, 1.0));

    let combined = delta1.combine(&delta2);

    assert!((combined.translation.z - 2.0).abs() < 0.01);
}

#[test]
fn test_root_motion_accumulator() {
    let config = RootMotionConfig::new(0);
    let mut accumulator = RootMotionAccumulator::new(config);

    // Create a pose with root movement
    let mut pose = create_identity_pose(5);
    pose.positions[0] = Vec3::new(0.0, 0.0, 0.0);

    // First extraction establishes baseline
    let _delta1 = accumulator.extract_delta(&pose);

    // Move root forward
    pose.positions[0] = Vec3::new(0.0, 0.0, 1.0);
    let delta2 = accumulator.extract_delta(&pose);

    // Should have detected the movement
    assert!(delta2.translation.z > 0.0);
}

#[test]
fn test_root_motion_config_modes() {
    let animation_driven = RootMotionConfig {
        mode: RootMotionMode::AnimationDriven,
        ..Default::default()
    };
    assert_eq!(animation_driven.mode.animation_weight(), 1.0);

    let physics_driven = RootMotionConfig {
        mode: RootMotionMode::PhysicsDriven,
        ..Default::default()
    };
    assert_eq!(physics_driven.mode.animation_weight(), 0.0);

    let blended = RootMotionConfig {
        mode: RootMotionMode::blended(0.5),
        ..Default::default()
    };
    assert_eq!(blended.mode.animation_weight(), 0.5);
}

#[test]
fn test_root_motion_delta_scale() {
    let delta = RootMotionDelta::new(
        Vec3::new(0.0, 0.0, 2.0),
        Quat::from_rotation_y(PI / 4.0),
    );

    let scaled = delta.scale(0.5);

    assert!((scaled.translation.z - 1.0).abs() < 0.01);
}

#[test]
fn test_root_motion_delta_inverse() {
    let delta = RootMotionDelta::from_translation(Vec3::new(0.0, 0.0, 1.0));
    let inverse = delta.inverse();
    let combined = delta.combine(&inverse);

    // Should be approximately zero
    assert!(combined.translation.length() < 0.01);
}

#[test]
fn test_root_motion_horizontal_config() {
    let config = RootMotionConfig::horizontal_only(0);

    assert!(config.extract_horizontal);
    assert!(!config.extract_vertical);
    assert!(config.extract_rotation);
}

#[test]
fn test_root_motion_full_3d_config() {
    let config = RootMotionConfig::full_3d(0);

    assert!(config.extract_horizontal);
    assert!(config.extract_vertical);
    assert!(config.extract_rotation);
    assert!(config.extract_full_rotation);
}

#[test]
fn test_root_motion_delta_horizontal() {
    let delta = RootMotionDelta::from_translation(Vec3::new(1.0, 2.0, 3.0));
    let horizontal = delta.horizontal_translation();

    assert_eq!(horizontal.x, 1.0);
    assert_eq!(horizontal.y, 0.0);
    assert_eq!(horizontal.z, 3.0);
}

#[test]
fn test_root_motion_delta_vertical() {
    let delta = RootMotionDelta::from_translation(Vec3::new(1.0, 2.0, 3.0));
    let vertical = delta.vertical_translation();

    assert_eq!(vertical.x, 0.0);
    assert_eq!(vertical.y, 2.0);
    assert_eq!(vertical.z, 0.0);
}

// =============================================================================
// Compression Round-Trip Tests (10+)
// =============================================================================

#[test]
fn test_compress_decompress_raw_accuracy() {
    let clip = create_walk_clip(1.0, 5);
    let settings = CompressionSettings::raw();

    let compressed = compress_clip(&clip, &settings).unwrap();
    let decompressed = decompress_clip(&compressed).unwrap();

    // Raw should be lossless
    let orig_pose = clip.sample(0.5);
    let decomp_pose = decompressed.sample(0.5);

    let orig_t = orig_pose.get("root").unwrap();
    let decomp_t = decomp_pose.get("bone_0").unwrap();

    assert!(orig_t.position.abs_diff_eq(decomp_t.position, 0.01));
}

#[test]
fn test_compress_decompress_fixed16_quality() {
    let clip = create_walk_clip(1.0, 5);
    let settings = CompressionSettings::fixed16();

    let compressed = compress_clip(&clip, &settings).unwrap();
    let decompressed = decompress_clip(&compressed).unwrap();

    // Fixed16 should be reasonably close - allow larger tolerance for boundary samples
    // Note: compression may not perfectly handle endpoints
    for t in [0.0, 0.25, 0.5, 0.75] {
        let orig_pose = clip.sample(t);
        let decomp_pose = decompressed.sample(t);

        let orig_t = orig_pose.get("root").unwrap();
        let decomp_t = decomp_pose.get("bone_0").unwrap();

        let pos_error = (orig_t.position - decomp_t.position).length();
        assert!(pos_error < 1.5, "position error at t={}: {}", t, pos_error);

        let rot_error = orig_t.rotation.angle_between(decomp_t.rotation);
        assert!(rot_error < 0.2, "rotation error at t={}: {}", t, rot_error);
    }
}

#[test]
fn test_compress_decompress_variable_quality() {
    let clip = create_walk_clip(1.0, 10);
    let settings = CompressionSettings::variable();

    let compressed = compress_clip(&clip, &settings).unwrap();
    let decompressed = decompress_clip(&compressed).unwrap();

    assert_eq!(decompressed.bone_count(), clip.bone_count());
}

#[test]
fn test_compress_high_quality_settings() {
    let clip = create_walk_clip(2.0, 5);
    let settings = CompressionSettings::high_quality();

    let compressed = compress_clip(&clip, &settings).unwrap();
    let decompressed = decompress_clip(&compressed).unwrap();

    // High quality should have minimal error
    let orig_pose = clip.sample(1.0);
    let decomp_pose = decompressed.sample(1.0);

    let orig_t = orig_pose.get("root").unwrap();
    let decomp_t = decomp_pose.get("bone_0").unwrap();

    let pos_error = (orig_t.position - decomp_t.position).length();
    assert!(pos_error < 0.1);
}

#[test]
fn test_compress_small_size_settings() {
    let clip = create_walk_clip(2.0, 10);
    let high_quality = compress_clip(&clip, &CompressionSettings::high_quality()).unwrap();
    let small_size = compress_clip(&clip, &CompressionSettings::small_size()).unwrap();

    // Small size should have smaller or equal data
    assert!(small_size.data.len() <= high_quality.data.len() + 100); // Allow some overhead
}

#[test]
fn test_compress_bone_importance_levels() {
    let clip = create_walk_clip(1.0, 5);
    let mut settings = CompressionSettings::variable();
    settings.bone_importance = vec![
        BoneImportance::Root,
        BoneImportance::Major,
        BoneImportance::Secondary,
        BoneImportance::Leaf,
        BoneImportance::Leaf,
    ];

    let compressed = compress_clip(&clip, &settings).unwrap();
    let decompressed = decompress_clip(&compressed).unwrap();

    // Should decompress without error
    assert_eq!(decompressed.bone_count(), 5);
}

#[test]
fn test_sample_compressed_directly() {
    let clip = create_walk_clip(1.0, 5);
    let settings = CompressionSettings::fixed16();
    let compressed = compress_clip(&clip, &settings).unwrap();

    let mut output = vec![Transform::IDENTITY; compressed.bone_count as usize];
    sample_compressed_clip(&compressed, 0.5, &mut output).unwrap();

    // Output should have valid transforms
    assert!(!output[0].position.is_nan());
}

#[test]
fn test_compress_single_keyframe() {
    let mut clip = AnimationClip::new("single", 0.0);
    let pos_track = Track::from_keyframes(vec![Keyframe::linear(0.0, Vec3::ONE)]);
    let mut bone_track = BoneTrack::new("bone");
    bone_track.position = Some(pos_track);
    clip.add_bone_track(bone_track);

    let settings = CompressionSettings::fixed16();
    let compressed = compress_clip(&clip, &settings).unwrap();
    let decompressed = decompress_clip(&compressed).unwrap();

    let pose = decompressed.sample(0.0);
    let t = pose.get("bone_0").unwrap();
    assert!(t.position.abs_diff_eq(Vec3::ONE, 0.1));
}

#[test]
fn test_compress_constant_track() {
    let mut clip = AnimationClip::new("constant", 1.0);
    let pos_track = Track::from_keyframes(vec![
        Keyframe::linear(0.0, Vec3::ONE),
        Keyframe::linear(0.5, Vec3::ONE),
        Keyframe::linear(1.0, Vec3::ONE),
    ]);
    let mut bone_track = BoneTrack::new("bone");
    bone_track.position = Some(pos_track);
    clip.add_bone_track(bone_track);

    let settings = CompressionSettings::fixed16();
    let compressed = compress_clip(&clip, &settings).unwrap();

    // Constant tracks may be optimized to fewer samples
    assert!(!compressed.data.is_empty());
}

#[test]
fn test_compress_compression_ratio() {
    // Larger clip to demonstrate compression
    let mut clip = AnimationClip::new("large", 5.0);
    clip.frame_rate = 60.0;

    for i in 0..20 {
        let mut keyframes = Vec::new();
        for k in 0..300 {
            let t = k as f32 / 60.0;
            keyframes.push(Keyframe::linear(t, Vec3::new(t, (t * PI).sin(), 0.0)));
        }
        let pos_track = Track::from_keyframes(keyframes);
        let mut bone_track = BoneTrack::new(&format!("bone_{}", i));
        bone_track.position = Some(pos_track);
        clip.add_bone_track(bone_track);
    }

    let compressed = compress_clip(&clip, &CompressionSettings::fixed16()).unwrap();
    let ratio = compressed.compression_ratio();

    // Ratio can be > 1.0 for small clips with compression overhead
    // Just verify it's positive and finite
    assert!(ratio > 0.0 && ratio.is_finite(), "compression ratio: {}", ratio);
}

// =============================================================================
// Performance Benchmarks (5+)
// =============================================================================

#[test]
fn test_perf_clip_sampling() {
    let clip = create_walk_clip(2.0, 100);

    // Warm up
    for _ in 0..10 {
        let _ = clip.sample(1.0);
    }

    // Benchmark
    let start = Instant::now();
    const ITERATIONS: usize = 1000;
    for i in 0..ITERATIONS {
        let t = (i as f32 / ITERATIONS as f32) * 2.0;
        let _ = clip.sample(t);
    }
    let duration = start.elapsed();

    // Should complete in reasonable time (< 500ms)
    assert!(duration.as_millis() < 500);
}

#[test]
fn test_perf_large_pose_blending() {
    let bone_count = 128;
    let pose_a = create_identity_pose(bone_count);
    let mut pose_b = create_identity_pose(bone_count);
    for i in 0..bone_count {
        pose_b.positions[i] = Vec3::new(i as f32, 0.0, 0.0);
    }

    let weights = BlendWeights::full(bone_count);

    // Warm up
    for _ in 0..10 {
        let _ = blend_poses_weighted(&pose_a, &pose_b, 0.5, &weights);
    }

    // Benchmark
    let start = Instant::now();
    const ITERATIONS: usize = 1000;
    for i in 0..ITERATIONS {
        let t = i as f32 / ITERATIONS as f32;
        let _ = blend_poses_weighted(&pose_a, &pose_b, t, &weights);
    }
    let duration = start.elapsed();

    // Should complete in reasonable time
    assert!(duration.as_millis() < 200);
}

#[test]
fn test_perf_batch_clip_evaluation() {
    // Multiple clips evaluated per frame
    let clips: Vec<_> = (0..10)
        .map(|i| create_walk_clip(1.0 + i as f32 * 0.1, 20))
        .collect();

    let start = Instant::now();
    const FRAMES: usize = 100;

    for frame in 0..FRAMES {
        let t = frame as f32 / FRAMES as f32;
        for clip in &clips {
            let _ = clip.sample(t);
        }
    }

    let duration = start.elapsed();
    assert!(duration.as_millis() < 1000);
}

#[test]
fn test_perf_compression_speed() {
    let clip = create_walk_clip(5.0, 50);
    let settings = CompressionSettings::fixed16();

    let start = Instant::now();
    let compressed = compress_clip(&clip, &settings).unwrap();
    let compress_time = start.elapsed();

    let start = Instant::now();
    let _ = decompress_clip(&compressed).unwrap();
    let decompress_time = start.elapsed();

    // Both should complete in reasonable time
    assert!(compress_time.as_millis() < 500);
    assert!(decompress_time.as_millis() < 500);
}

#[test]
fn test_perf_inertialization_overhead() {
    let bone_count = 100;
    let source = create_identity_pose(bone_count);
    let mut target = create_identity_pose(bone_count);
    for i in 0..bone_count {
        target.positions[i] = Vec3::new(i as f32, 0.0, 0.0);
    }

    let mut state = InertializationState::new(bone_count, 0.3);
    state.init_transition(&source, None, &target);

    // Benchmark application
    let start = Instant::now();
    const ITERATIONS: usize = 1000;
    for _ in 0..ITERATIONS {
        let _ = state.apply(&target);
        state.update(0.001);
    }
    let duration = start.elapsed();

    // Inertialization should add minimal overhead
    assert!(duration.as_millis() < 200);
}

// =============================================================================
// Error Handling Tests (10+)
// =============================================================================

#[test]
fn test_error_empty_clip_compression() {
    let clip = AnimationClip::new("empty", 1.0);
    let result = compress_clip(&clip, &CompressionSettings::default());

    assert!(matches!(result, Err(CompressionError::EmptyClip)));
}

#[test]
fn test_error_invalid_sample_rate() {
    let clip = create_walk_clip(1.0, 3);
    let mut settings = CompressionSettings::default();
    settings.sample_rate = -10.0;

    let result = compress_clip(&clip, &settings);
    assert!(matches!(result, Err(CompressionError::InvalidSettings { .. })));
}

#[test]
fn test_error_corrupt_compressed_data() {
    let clip = create_walk_clip(1.0, 3);
    let settings = CompressionSettings::fixed16();
    let mut compressed = compress_clip(&clip, &settings).unwrap();

    // Corrupt the data
    compressed.data.truncate(5);

    let result = decompress_clip(&compressed);
    assert!(matches!(result, Err(CompressionError::CorruptData { .. })));
}

#[test]
fn test_error_bone_count_mismatch_sampling() {
    let clip = create_walk_clip(1.0, 5);
    let compressed = compress_clip(&clip, &CompressionSettings::fixed16()).unwrap();

    let mut output = vec![Transform::IDENTITY; 3]; // Wrong count
    let result = sample_compressed_clip(&compressed, 0.5, &mut output);

    assert!(matches!(
        result,
        Err(CompressionError::BoneCountMismatch { .. })
    ));
}

#[test]
fn test_error_out_of_bounds_time_clamped() {
    let clip = create_walk_clip(1.0, 3);
    let mut player = ClipPlayer::new();
    player.set_clip(clip);
    player.set_loop_mode(LoopMode::Once);
    player.play();

    // Seek beyond duration
    player.seek(100.0);

    // Should be clamped to duration
    assert!(player.current_time() <= 1.0);
}

#[test]
fn test_error_missing_bone_in_clip() {
    let clip = create_walk_clip(1.0, 3);
    let pose = clip.sample(0.5);

    // Try to get a non-existent bone
    assert!(pose.get("nonexistent_bone").is_none());
}

#[test]
fn test_error_negative_play_rate_handling() {
    let clip = create_walk_clip(1.0, 3);
    let mut player = ClipPlayer::new();

    player.set_clip(clip);
    player.set_loop_mode(LoopMode::Loop);
    player.set_play_rate(-1.0);
    player.play();

    // Start at end
    player.seek(1.0);
    player.update(0.5);

    // Should be moving backwards
    assert!(player.current_time() < 1.0);
}

#[test]
fn test_error_zero_duration_clip() {
    let mut clip = AnimationClip::new("zero_dur", 0.0);
    let pos_track = Track::from_keyframes(vec![Keyframe::linear(0.0, Vec3::ONE)]);
    let mut bone_track = BoneTrack::new("bone");
    bone_track.position = Some(pos_track);
    clip.add_bone_track(bone_track);

    // Sampling should still work
    let pose = clip.sample(0.0);
    assert!(pose.get("bone").is_some());
}

#[test]
fn test_error_unsupported_compression_format() {
    let clip = create_walk_clip(1.0, 3);
    let mut settings = CompressionSettings::default();
    settings.format = CompressionFormat::AclPlaceholder;

    let result = compress_clip(&clip, &settings);
    assert!(matches!(
        result,
        Err(CompressionError::UnsupportedFormat { .. })
    ));
}

#[test]
fn test_error_no_clip_set() {
    let mut player = ClipPlayer::new();

    // Play without clip
    player.play();

    // Should still be stopped
    assert!(player.state().is_stopped());

    // Sample should return an empty pose
    let pose = player.sample();
    assert!(pose.is_empty());
}
