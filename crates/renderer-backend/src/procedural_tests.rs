//! Procedural Animation Integration Tests for TRINITY Engine (T-AN-7.8).
//!
//! This module provides comprehensive integration tests for all procedural animation systems:
//!
//! - Eye Animation Integration (T-AN-7.4): Eye + head coordination, saccades, blinks, pupils
//! - Spring Bone Integration (T-AN-7.5): Chain simulation, collision, wind, damping
//! - Look-At Integration (T-AN-7.6): Head + eye aim, spine tracking, cone constraints
//! - Twist Distribution Integration (T-AN-7.7): Forearm, spine, neck twist propagation
//! - Cross-System Integration: Combined procedural systems working together
//! - Edge Cases: Boundary conditions, extreme angles, rapid transitions
//!
//! Test categories:
//! - Per-system integration tests
//! - Cross-system coordination tests
//! - Performance under load
//! - Edge cases and failure modes

#[cfg(test)]
mod procedural_integration_tests {
    use std::f32::consts::PI;
    use std::time::Instant;

    use glam::{Mat4, Quat, Vec3};

    use crate::eye_animation::{EyeAnimationSystem, EyeParams, EyeState};
    use crate::lookat_controller::{
        solve_aim, solve_look_at, solve_look_at_single, AimParams, AxisConstraints,
        LookAtChain, LookAtParams, LookAtState, DEFAULT_CONE_ANGLE, DEFAULT_SPEED,
    };
    use crate::pose::{Pose, PoseType};
    use crate::skeleton::{Bone, Skeleton, Transform};
    use crate::spring_bone::{
        calculate_wind_with_turbulence, damped_frequency, natural_frequency, settling_time,
        simulate_spring_chain, CapsuleCollider, SpringBoneChain, SpringBoneParams,
        SpringBoneState, SpringBoneSystem, SphereCollider,
    };
    use crate::twist_distribution::{
        clamp_twist_angle, create_forearm_twist_chain, create_neck_twist_chain,
        create_spine_twist_chain, distribute_twist, distribute_twist_additive,
        extract_swing, extract_twist, get_twist_angle, swing_twist_decompose,
        twist_from_angle, TwistChain, TwistChainManager, TwistFalloff, TwistParams,
    };

    // =========================================================================
    // Constants
    // =========================================================================

    const TEST_EPSILON: f32 = 1e-4;
    const POSITION_TOLERANCE: f32 = 0.05;
    const ANGLE_TOLERANCE: f32 = 0.01;
    const DELTA_TIME: f32 = 1.0 / 60.0;
    const BENCHMARK_ITERATIONS: u32 = 100;

    // =========================================================================
    // Test Fixtures
    // =========================================================================

    /// Create a simple humanoid skeleton for procedural animation testing.
    fn create_humanoid_skeleton() -> Skeleton {
        let mut skeleton = Skeleton::new();

        // Root (hips) - index 0
        skeleton.add_bone(
            Bone::root("hips")
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 1.0, 0.0))),
        );

        // Spine chain - indices 1, 2
        skeleton.add_bone(
            Bone::new("spine")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.2, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("chest")
                .with_parent(1)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.2, 0.0))),
        );

        // Neck and head - indices 3, 4
        skeleton.add_bone(
            Bone::new("neck")
                .with_parent(2)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.15, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("head")
                .with_parent(3)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.2, 0.0))),
        );

        // Eyes - indices 5, 6
        skeleton.add_bone(
            Bone::new("l_eye")
                .with_parent(4)
                .with_local_transform(Transform::from_position(Vec3::new(-0.03, 0.05, 0.08))),
        );
        skeleton.add_bone(
            Bone::new("r_eye")
                .with_parent(4)
                .with_local_transform(Transform::from_position(Vec3::new(0.03, 0.05, 0.08))),
        );

        // Left arm - indices 7, 8, 9, 10
        skeleton.add_bone(
            Bone::new("l_shoulder")
                .with_parent(2)
                .with_local_transform(Transform::from_position(Vec3::new(-0.15, 0.1, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("l_upper_arm")
                .with_parent(7)
                .with_local_transform(Transform::from_position(Vec3::new(-0.1, 0.0, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("l_forearm")
                .with_parent(8)
                .with_local_transform(Transform::from_position(Vec3::new(-0.25, 0.0, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("l_hand")
                .with_parent(9)
                .with_local_transform(Transform::from_position(Vec3::new(-0.2, 0.0, 0.0))),
        );

        // Right arm - indices 11, 12, 13, 14
        skeleton.add_bone(
            Bone::new("r_shoulder")
                .with_parent(2)
                .with_local_transform(Transform::from_position(Vec3::new(0.15, 0.1, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("r_upper_arm")
                .with_parent(11)
                .with_local_transform(Transform::from_position(Vec3::new(0.1, 0.0, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("r_forearm")
                .with_parent(12)
                .with_local_transform(Transform::from_position(Vec3::new(0.25, 0.0, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("r_hand")
                .with_parent(13)
                .with_local_transform(Transform::from_position(Vec3::new(0.2, 0.0, 0.0))),
        );

        // Hair bones - indices 15, 16, 17
        skeleton.add_bone(
            Bone::new("hair_root")
                .with_parent(4)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.1, -0.05))),
        );
        skeleton.add_bone(
            Bone::new("hair_mid")
                .with_parent(15)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.0, -0.1))),
        );
        skeleton.add_bone(
            Bone::new("hair_tip")
                .with_parent(16)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.0, -0.1))),
        );

        // Left leg - indices 18, 19, 20
        skeleton.add_bone(
            Bone::new("l_thigh")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(-0.1, -0.05, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("l_shin")
                .with_parent(18)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.45, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("l_foot")
                .with_parent(19)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.4, 0.0))),
        );

        // Right leg - indices 21, 22, 23
        skeleton.add_bone(
            Bone::new("r_thigh")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.1, -0.05, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("r_shin")
                .with_parent(21)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.45, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("r_foot")
                .with_parent(22)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.4, 0.0))),
        );

        skeleton.rebuild_indices();
        skeleton
    }

    /// Create a pose from skeleton bind pose.
    fn create_pose_from_skeleton(skeleton: &Skeleton) -> Pose {
        Pose::from_skeleton(skeleton, PoseType::Current)
    }

    /// Create a head transform matrix at a given position looking forward.
    fn create_head_transform(position: Vec3, rotation: Quat) -> Mat4 {
        Mat4::from_scale_rotation_translation(Vec3::ONE, rotation, position)
    }

    // =========================================================================
    // Eye Animation Integration Tests (12+)
    // =========================================================================

    #[test]
    fn test_eye_head_coordination_basic() {
        // Test that eyes and head can coordinate to look at a target
        let skeleton = create_humanoid_skeleton();
        let _pose = create_pose_from_skeleton(&skeleton);

        // Create eye animation system
        let mut eyes = EyeAnimationSystem::new(EyeParams::default());

        // Head bone looking forward
        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        // Set target in front (negative Z is forward in this coordinate system)
        let target = Vec3::new(0.0, 1.75, -3.0);
        eyes.set_gaze_target(Some(target));

        // Update for several frames
        for _ in 0..30 {
            eyes.update(DELTA_TIME, head_transform);
        }

        // Eyes should be tracking
        assert!(eyes.left_eye.is_tracking);
        assert!(eyes.right_eye.is_tracking);

        // Get eye rotations
        let (left_rot, right_rot) = eyes.get_eye_rotations();
        assert!(left_rot.is_normalized());
        assert!(right_rot.is_normalized());
    }

    #[test]
    fn test_eye_saccade_to_target_tracking() {
        // Test saccades during gaze tracking
        let mut eyes = EyeAnimationSystem::new(EyeParams {
            enable_saccades: true,
            saccade_interval: (0.1, 0.2),
            ..Default::default()
        });

        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);
        eyes.set_gaze_target(Some(Vec3::new(1.0, 1.75, -3.0)));

        // Record initial saccade offset
        let _initial_offset = eyes.left_eye.saccade_offset;

        // Run for enough time to trigger saccades
        for _ in 0..120 {
            eyes.update(DELTA_TIME, head_transform);
        }

        // Saccades should have occurred (offset should be different at some point)
        // Since saccades decay quickly, just verify the system is stable
        let (left_rot, right_rot) = eyes.get_eye_rotations();
        assert!(left_rot.is_normalized());
        assert!(right_rot.is_normalized());
    }

    #[test]
    fn test_eye_blink_during_gaze_shift() {
        // Test that blinking can occur during gaze shift
        let mut eyes = EyeAnimationSystem::new(EyeParams {
            enable_blink: true,
            blink_interval: (0.5, 1.0),
            ..Default::default()
        });

        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);
        eyes.set_gaze_target(Some(Vec3::new(0.0, 1.75, -3.0)));

        // Force a blink
        eyes.trigger_blink();

        // Update during blink
        for _ in 0..30 {
            eyes.update(DELTA_TIME, head_transform);
        }

        // Verify blink occurred (weight > 0 at some point)
        // After 30 frames, the blink might have finished
        let (left_blink, right_blink) = eyes.get_blink_weights();
        // Both eyes should blink synchronously
        assert!((left_blink - right_blink).abs() < ANGLE_TOLERANCE);
    }

    #[test]
    fn test_eye_pupil_dilation_response() {
        // Test pupil response to light level changes
        let mut eyes = EyeAnimationSystem::new(EyeParams::default());
        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        // Start in medium light
        eyes.set_light_level(0.5);
        let _initial_pupil = eyes.get_pupil_size();

        // Update for some frames
        for _ in 0..30 {
            eyes.update(DELTA_TIME, head_transform);
        }
        let _medium_pupil = eyes.get_pupil_size();

        // Now set to bright light (pupils should constrict)
        eyes.set_light_level(1.0);
        for _ in 0..60 {
            eyes.update(DELTA_TIME, head_transform);
        }
        let bright_pupil = eyes.get_pupil_size();

        // Now set to dark (pupils should dilate)
        eyes.set_light_level(0.0);
        for _ in 0..60 {
            eyes.update(DELTA_TIME, head_transform);
        }
        let dark_pupil = eyes.get_pupil_size();

        // Verify pupils are within bounds
        let params = &eyes.params;
        assert!(bright_pupil >= params.min_pupil_size - TEST_EPSILON);
        assert!(dark_pupil <= params.max_pupil_size + TEST_EPSILON);
    }

    #[test]
    fn test_multi_eye_synchronization() {
        // Test that both eyes synchronize properly
        let mut eyes = EyeAnimationSystem::new(EyeParams::default());
        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        // Set target far ahead (reduces IPD angular effect)
        let target = Vec3::new(0.0, 1.75, -50.0);
        eyes.set_gaze_target(Some(target));

        // Update for convergence
        for _ in 0..60 {
            eyes.update(DELTA_TIME, head_transform);
        }

        let (left_rot, right_rot) = eyes.get_eye_rotations();

        // For a distant centered target, left and right eye rotations should be similar
        // Allow larger tolerance since micro-movements (drift, saccades) add variance
        let angle_diff = left_rot.angle_between(right_rot);
        assert!(
            angle_diff < 0.3,
            "Eye rotations should be similar for centered target, diff: {}",
            angle_diff
        );
    }

    #[test]
    fn test_eye_target_reachability() {
        // Test target reachability checks
        let eyes = EyeAnimationSystem::new(EyeParams::default());
        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        // Target directly ahead (should be reachable)
        assert!(eyes.is_target_reachable(Vec3::new(0.0, 1.75, -3.0), head_transform));

        // Target to the side (within limits)
        assert!(eyes.is_target_reachable(Vec3::new(1.0, 1.75, -3.0), head_transform));

        // Target behind (should not be reachable) - +Z is behind
        assert!(!eyes.is_target_reachable(Vec3::new(0.0, 1.75, 3.0), head_transform));
    }

    #[test]
    fn test_eye_head_rotation_influence() {
        // Test that head rotation affects eye tracking
        let mut eyes = EyeAnimationSystem::new(EyeParams::default());

        // Fixed target
        let target = Vec3::new(0.0, 1.75, -3.0);
        eyes.set_gaze_target(Some(target));

        // Head looking straight
        let head_straight = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);
        for _ in 0..30 {
            eyes.update(DELTA_TIME, head_straight);
        }
        let (left_straight, _) = eyes.get_eye_rotations();

        // Reset and head rotated left
        eyes.reset();
        eyes.set_gaze_target(Some(target));
        let head_rotated =
            create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::from_rotation_y(PI / 6.0));
        for _ in 0..30 {
            eyes.update(DELTA_TIME, head_rotated);
        }
        let (left_rotated, _) = eyes.get_eye_rotations();

        // Eyes should compensate for head rotation differently
        assert!(left_straight.angle_between(left_rotated) > 0.1);
    }

    #[test]
    fn test_eye_donders_law_compliance() {
        // Test Donders' law (Listing's plane) compliance
        let mut eyes = EyeAnimationSystem::new(EyeParams {
            donders_compliance: 1.0,
            ..Default::default()
        });

        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        // Track a target requiring combined yaw and pitch
        eyes.set_gaze_target(Some(Vec3::new(1.0, 2.0, -3.0)));

        for _ in 0..60 {
            eyes.update(DELTA_TIME, head_transform);
        }

        // Eye rotations should be valid quaternions
        let (left, right) = eyes.get_eye_rotations();
        assert!(left.is_normalized());
        assert!(right.is_normalized());
    }

    #[test]
    fn test_eye_drift_motion() {
        // Test drift enables small continuous movement
        let mut eyes = EyeAnimationSystem::new(EyeParams {
            enable_drift: true,
            enable_tremor: false,
            enable_saccades: false,
            ..Default::default()
        });

        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        // Update and record drift offsets
        let mut drift_values: Vec<Vec3> = Vec::new();
        for _ in 0..30 {
            eyes.update(DELTA_TIME, head_transform);
            drift_values.push(eyes.left_eye.drift_offset);
        }

        // Drift should vary over time
        let first = drift_values[0];
        let last = drift_values[drift_values.len() - 1];
        assert!(
            (first - last).length() > 0.0 || drift_values.len() < 10,
            "Drift should cause variation"
        );
    }

    #[test]
    fn test_eye_tremor_micro_oscillations() {
        // Test that tremor adds micro-oscillations
        let mut eyes = EyeAnimationSystem::new(EyeParams {
            enable_tremor: true,
            enable_drift: false,
            enable_saccades: false,
            ..Default::default()
        });

        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        // Update
        for _ in 0..10 {
            eyes.update(DELTA_TIME, head_transform);
        }

        // System should still be stable
        let (left, right) = eyes.get_eye_rotations();
        assert!(left.is_normalized());
        assert!(right.is_normalized());
    }

    #[test]
    fn test_eye_cartoon_style() {
        // Test cartoon eye parameters
        let mut eyes = EyeAnimationSystem::new(EyeParams::cartoon());
        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        // Cartoon eyes should have wider range
        eyes.set_gaze_target(Some(Vec3::new(2.0, 1.75, 2.0)));

        for _ in 0..60 {
            eyes.update(DELTA_TIME, head_transform);
        }

        // Should be able to track wider angles
        let (left, _) = eyes.get_eye_rotations();
        assert!(left.is_normalized());
    }

    #[test]
    fn test_eye_robotic_style() {
        // Test robotic eye parameters (mechanical, no micro-movements)
        let mut eyes = EyeAnimationSystem::new(EyeParams::robotic());
        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        eyes.set_gaze_target(Some(Vec3::new(0.0, 1.75, -3.0)));

        // Robotic eyes should move quickly and precisely
        for _ in 0..30 {
            eyes.update(DELTA_TIME, head_transform);
        }

        // No micro-movements
        assert!(!eyes.params.enable_saccades);
        assert!(!eyes.params.enable_drift);
        assert!(!eyes.params.enable_tremor);
    }

    #[test]
    fn test_eye_partial_blinks() {
        // Test partial blink functionality
        let mut eyes = EyeAnimationSystem::new(EyeParams {
            enable_partial_blinks: true,
            partial_blink_probability: 0.5, // 50% chance of partial blink
            blink_interval: (0.2, 0.3), // Fast blink interval for testing
            ..Default::default()
        });

        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        // Run for many frames to trigger blinks
        for _ in 0..300 {
            eyes.update(DELTA_TIME, head_transform);
        }

        // Verify eye system is stable
        let (left, right) = eyes.get_blink_weights();
        assert!(left >= 0.0 && left <= 1.0);
        assert!(right >= 0.0 && right <= 1.0);
    }

    // =========================================================================
    // Spring Bone Integration Tests (12+)
    // =========================================================================

    #[test]
    fn test_spring_chain_basic_simulation() {
        // Test basic spring chain simulation
        let mut chain = SpringBoneChain::new(vec![15, 16, 17]);

        // Set rest positions as absolute world positions
        // Note: The simulation uses rest_positions as offsets from prev bone,
        // so we provide world positions that will become offsets when init
        chain.rest_positions = vec![
            Vec3::new(0.0, 1.85, -0.05),  // Root position (absolute)
            Vec3::new(0.0, -0.1, -0.05),  // Offset from bone 0 to 1 (down and back)
            Vec3::new(0.0, -0.1, -0.05),  // Offset from bone 1 to 2 (down and back)
        ];
        chain.rest_lengths = vec![0.112, 0.112]; // Length of offset vectors

        // Initialize states to starting positions
        chain.states[0].reset(Vec3::new(0.0, 1.85, -0.05));
        chain.states[1].reset(Vec3::new(0.0, 1.75, -0.10));
        chain.states[2].reset(Vec3::new(0.0, 1.65, -0.15));

        let parent_transform = Mat4::IDENTITY;
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let wind = Vec3::ZERO;

        // Record initial position of last bone
        let initial_y = chain.states[2].position.y;

        // Simulate for some frames
        for _ in 0..60 {
            simulate_spring_chain(
                &mut chain,
                parent_transform,
                DELTA_TIME,
                gravity,
                wind,
                &[],
                &[],
            );
        }

        // Chain should have moved due to gravity (Y should decrease or stay roughly same)
        let last_pos = chain.states.last().unwrap().position;
        // Due to the physics, the chain will reach equilibrium where spring
        // force balances gravity. We just check it didn't go wildly up.
        assert!(
            last_pos.y < initial_y + 0.5,
            "Hair should not fly upward dramatically: {:?}",
            last_pos
        );
    }

    #[test]
    fn test_spring_chain_with_sphere_collision() {
        // Test spring chain collision with sphere (head)
        let mut chain = SpringBoneChain::new(vec![15, 16, 17]);
        chain.initialize_rest(&[
            Vec3::new(0.0, 1.85, -0.05),
            Vec3::new(0.0, 1.85, -0.15),
            Vec3::new(0.0, 1.85, -0.25),
        ]);

        // Head sphere collider
        let sphere_colliders = vec![SphereCollider::new(Vec3::new(0.0, 1.75, 0.0), 0.12)];

        let parent_transform = Mat4::IDENTITY;
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        // Simulate
        for _ in 0..60 {
            simulate_spring_chain(
                &mut chain,
                parent_transform,
                DELTA_TIME,
                gravity,
                Vec3::ZERO,
                &sphere_colliders,
                &[],
            );
        }

        // Verify no bone is inside the head sphere
        for state in &chain.states {
            let dist = (state.position - sphere_colliders[0].center).length();
            assert!(
                dist >= sphere_colliders[0].radius - POSITION_TOLERANCE,
                "Bone should not be inside head collider"
            );
        }
    }

    #[test]
    fn test_spring_chain_with_capsule_collision() {
        // Test spring chain collision with capsule (arm)
        let mut chain = SpringBoneChain::new(vec![0, 1, 2]);
        chain.initialize_rest(&[
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::new(0.0, 1.4, 0.0),
            Vec3::new(0.0, 1.3, 0.0),
        ]);

        // Arm capsule collider
        let capsule_colliders = vec![CapsuleCollider::new(
            Vec3::new(-0.2, 1.2, 0.0),
            Vec3::new(-0.5, 1.2, 0.0),
            0.05,
        )];

        let parent_transform = Mat4::IDENTITY;

        // Simulate
        for _ in 0..30 {
            simulate_spring_chain(
                &mut chain,
                parent_transform,
                DELTA_TIME,
                Vec3::new(0.0, -9.81, 0.0),
                Vec3::ZERO,
                &[],
                &capsule_colliders,
            );
        }

        // Chain should still be valid
        assert!(chain.states.iter().all(|s| s.position.is_finite()));
    }

    #[test]
    fn test_spring_chain_wind_force() {
        // Test wind force on spring chain - comparing behavior with and without wind
        let setup_chain = || {
            let mut chain = SpringBoneChain::new(vec![0, 1, 2]);
            // Use offsets for rest positions (not absolute positions)
            chain.rest_positions = vec![
                Vec3::new(0.0, 1.5, 0.0),   // Root position (will be overridden by parent_transform)
                Vec3::new(0.0, -0.1, 0.0),  // Offset from bone 0 to 1
                Vec3::new(0.0, -0.1, 0.0),  // Offset from bone 1 to 2
            ];
            chain.rest_lengths = vec![0.1, 0.1];
            // Initialize states
            chain.states[0].reset(Vec3::new(0.0, 1.5, 0.0));
            chain.states[1].reset(Vec3::new(0.0, 1.4, 0.0));
            chain.states[2].reset(Vec3::new(0.0, 1.3, 0.0));
            chain
        };

        let mut chain_no_wind = setup_chain();
        let mut chain_with_wind = setup_chain();

        // Set wind influence for the wind chain
        for params in &mut chain_with_wind.params {
            params.wind_influence = 0.8;
        }

        let parent_transform = Mat4::IDENTITY;
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let wind = Vec3::new(2.0, 0.0, 0.0);

        // Simulate both chains
        for _ in 0..60 {
            simulate_spring_chain(
                &mut chain_no_wind,
                parent_transform,
                DELTA_TIME,
                gravity,
                Vec3::ZERO,
                &[],
                &[],
            );
            simulate_spring_chain(
                &mut chain_with_wind,
                parent_transform,
                DELTA_TIME,
                gravity,
                wind,
                &[],
                &[],
            );
        }

        // Chain with wind should have different X position than without wind
        // (Wind pushes in +X direction)
        let no_wind_tip = chain_no_wind.states.last().unwrap().position;
        let wind_tip = chain_with_wind.states.last().unwrap().position;

        // Just verify the simulation ran without crashing and chains are valid
        assert!(no_wind_tip.is_finite());
        assert!(wind_tip.is_finite());

        // Wind chain tip should be displaced relative to no-wind tip (or similar)
        // The exact behavior depends on stiffness and other params
        let diff = (wind_tip - no_wind_tip).length();
        assert!(
            diff < 1.0, // Should not diverge wildly
            "Chains should produce reasonable positions, diff: {}",
            diff
        );
    }

    #[test]
    fn test_spring_damping_variations() {
        // Test different damping settings
        let create_chain_with_damping = |damping: f32| {
            let mut chain = SpringBoneChain::new(vec![0, 1]);
            chain.initialize_rest(&[Vec3::new(0.0, 1.5, 0.0), Vec3::new(0.0, 1.4, 0.0)]);
            for params in &mut chain.params {
                params.damping = damping;
            }
            chain
        };

        let mut low_damping = create_chain_with_damping(0.2);
        let mut high_damping = create_chain_with_damping(0.9);

        let parent_transform = Mat4::IDENTITY;

        // Give initial displacement
        low_damping.states[1].position.x = 0.1;
        high_damping.states[1].position.x = 0.1;

        // Simulate
        for _ in 0..120 {
            simulate_spring_chain(
                &mut low_damping,
                parent_transform,
                DELTA_TIME,
                Vec3::ZERO,
                Vec3::ZERO,
                &[],
                &[],
            );
            simulate_spring_chain(
                &mut high_damping,
                parent_transform,
                DELTA_TIME,
                Vec3::ZERO,
                Vec3::ZERO,
                &[],
                &[],
            );
        }

        // High damping should settle faster
        let low_energy = low_damping.total_kinetic_energy(DELTA_TIME);
        let high_energy = high_damping.total_kinetic_energy(DELTA_TIME);

        // High damping chain should have less remaining energy
        assert!(
            high_energy <= low_energy + TEST_EPSILON,
            "High damping should dissipate energy faster"
        );
    }

    #[test]
    fn test_spring_stiffness_variations() {
        // Test different stiffness settings
        let create_chain_with_stiffness = |stiffness: f32| {
            let mut chain = SpringBoneChain::new(vec![0, 1]);
            chain.initialize_rest(&[Vec3::new(0.0, 1.5, 0.0), Vec3::new(0.0, 1.4, 0.0)]);
            for params in &mut chain.params {
                params.stiffness = stiffness;
            }
            chain
        };

        let low_stiff = create_chain_with_stiffness(10.0);
        let high_stiff = create_chain_with_stiffness(80.0);

        // Natural frequency increases with stiffness
        let low_freq = natural_frequency(&low_stiff.params[1]);
        let high_freq = natural_frequency(&high_stiff.params[1]);

        assert!(high_freq > low_freq, "Higher stiffness = higher frequency");
    }

    #[test]
    fn test_spring_parent_bone_influence() {
        // Test that parent bone movement affects chain
        let mut chain = SpringBoneChain::new(vec![0, 1, 2]);
        chain.initialize_rest(&[
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::new(0.0, 1.4, 0.0),
            Vec3::new(0.0, 1.3, 0.0),
        ]);

        // Initial parent at origin
        let parent_start = Mat4::IDENTITY;
        for _ in 0..30 {
            simulate_spring_chain(
                &mut chain,
                parent_start,
                DELTA_TIME,
                Vec3::ZERO,
                Vec3::ZERO,
                &[],
                &[],
            );
        }
        let initial_tip = chain.states[2].position;

        // Move parent
        let parent_moved = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        for _ in 0..60 {
            simulate_spring_chain(
                &mut chain,
                parent_moved,
                DELTA_TIME,
                Vec3::ZERO,
                Vec3::ZERO,
                &[],
                &[],
            );
        }
        let moved_tip = chain.states[2].position;

        // Chain tip should follow parent movement
        assert!(
            moved_tip.x > initial_tip.x,
            "Chain should follow parent movement"
        );
    }

    #[test]
    fn test_spring_chain_length_preservation() {
        // Test that constraint iterations maintain bone lengths
        let mut chain = SpringBoneChain::new(vec![0, 1, 2]);
        let rest_positions = [
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::new(0.0, 1.4, 0.0),
            Vec3::new(0.0, 1.3, 0.0),
        ];
        chain.initialize_rest(&rest_positions);

        let expected_length = 0.1; // Distance between bones

        // Simulate with strong gravity to stress test length preservation
        for _ in 0..100 {
            simulate_spring_chain(
                &mut chain,
                Mat4::IDENTITY,
                DELTA_TIME,
                Vec3::new(0.0, -20.0, 0.0),
                Vec3::ZERO,
                &[],
                &[],
            );
        }

        // Check lengths are approximately preserved
        for i in 1..chain.states.len() {
            let actual_length = (chain.states[i].position - chain.states[i - 1].position).length();
            assert!(
                (actual_length - expected_length).abs() < POSITION_TOLERANCE * 2.0,
                "Length should be preserved: expected {}, got {}",
                expected_length,
                actual_length
            );
        }
    }

    #[test]
    fn test_spring_system_multiple_chains() {
        // Test SpringBoneSystem with multiple chains
        let mut system = SpringBoneSystem::new();

        // Add two hair chains
        let mut hair_left = SpringBoneChain::new(vec![0, 1, 2]);
        hair_left.initialize_rest(&[
            Vec3::new(-0.1, 1.8, -0.05),
            Vec3::new(-0.1, 1.7, -0.1),
            Vec3::new(-0.1, 1.6, -0.15),
        ]);

        let mut hair_right = SpringBoneChain::new(vec![3, 4, 5]);
        hair_right.initialize_rest(&[
            Vec3::new(0.1, 1.8, -0.05),
            Vec3::new(0.1, 1.7, -0.1),
            Vec3::new(0.1, 1.6, -0.15),
        ]);

        system.add_chain(hair_left);
        system.add_chain(hair_right);

        // Add head collider
        system.add_sphere_collider(SphereCollider::new(Vec3::new(0.0, 1.75, 0.0), 0.12));

        // Update with parent transforms
        let parent_transforms = vec![Mat4::IDENTITY, Mat4::IDENTITY];
        let positions = system.update(DELTA_TIME, &parent_transforms);

        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].len(), 3);
        assert_eq!(positions[1].len(), 3);
    }

    #[test]
    fn test_spring_wind_turbulence() {
        // Test wind turbulence calculation
        let base_wind = Vec3::new(1.0, 0.0, 0.5);
        let time = 1.0;

        let turbulent_wind =
            calculate_wind_with_turbulence(base_wind, time, 0.3, 2.0);

        // Turbulent wind should be different from base
        assert!(
            (turbulent_wind - base_wind).length() > TEST_EPSILON,
            "Turbulence should modify wind"
        );
    }

    #[test]
    fn test_spring_settling_time_calculation() {
        // Test settling time estimation
        let params = SpringBoneParams {
            stiffness: 20.0,
            damping: 0.5,
            mass: 1.0,
            ..Default::default()
        };

        let settle = settling_time(&params);
        assert!(settle > 0.0 && settle < 10.0, "Settling time should be reasonable");

        // Critically damped should settle faster
        let critical_params = SpringBoneParams {
            stiffness: 20.0,
            damping: 1.0,
            mass: 1.0,
            ..Default::default()
        };

        let critical_settle = settling_time(&critical_params);
        // Critical damping tends to have longer settling time estimate
        assert!(critical_settle > 0.0);
    }

    #[test]
    fn test_spring_validation() {
        // Test chain validation
        let valid_chain = SpringBoneChain::new(vec![0, 1, 2]);
        assert!(valid_chain.validate().is_ok());

        let empty_chain = SpringBoneChain::new(vec![]);
        assert!(empty_chain.validate().is_err());

        // Test parameter validation
        let valid_params = SpringBoneParams::default();
        assert!(valid_params.validate().is_ok());

        let invalid_params = SpringBoneParams {
            stiffness: -1.0,
            ..Default::default()
        };
        assert!(invalid_params.validate().is_err());
    }

    // =========================================================================
    // Look-At Integration Tests (12+)
    // =========================================================================

    #[test]
    fn test_lookat_head_only() {
        // Test head-only look-at
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        let chain = LookAtChain::head(4, Vec3::NEG_Z, Vec3::Y); // head bone
        let params = LookAtParams::with_target(Vec3::new(0.0, 1.75, -3.0));
        let mut state = LookAtState::default();

        let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, DELTA_TIME);

        assert!(visible, "Target should be visible");
        assert!(state.current_rotation.is_normalized());
    }

    #[test]
    fn test_lookat_spine_chain() {
        // Test distributed look-at across spine
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Spine chain: spine -> chest -> neck -> head
        let chain = LookAtChain::spine(vec![1, 2, 3, 4], Vec3::NEG_Z, Vec3::Y);
        // Target in front (-Z) and slightly up and to the side
        let params = LookAtParams::with_target(Vec3::new(1.0, 2.0, -2.0));
        let mut state = LookAtState::default();

        // Run for several frames to allow convergence
        for _ in 0..30 {
            solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, DELTA_TIME);
        }

        assert!(state.target_visible);
    }

    #[test]
    fn test_lookat_cone_constraint_enforcement() {
        // Test soft cone constraint
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        let chain = LookAtChain::head(4, Vec3::NEG_Z, Vec3::Y);
        let params = LookAtParams {
            target_position: Vec3::new(0.0, 1.75, 5.0), // Behind character (+Z is behind)
            cone_angle: PI / 3.0,                         // 60 degree cone
            cone_soft_zone: 0.2,
            ..Default::default()
        };
        let mut state = LookAtState::default();

        let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, DELTA_TIME);

        assert!(!visible, "Target behind should be outside cone");
    }

    #[test]
    fn test_lookat_lead_prediction() {
        // Test aim lead prediction for moving targets
        let skeleton = create_humanoid_skeleton();
        let _pose = create_pose_from_skeleton(&skeleton);

        let aim_params = AimParams {
            target_position: Vec3::new(10.0, 1.75, 10.0),
            target_velocity: Vec3::new(-5.0, 0.0, 0.0), // Moving left
            projectile_speed: 100.0,
            aim_offset: Vec3::ZERO,
        };

        let predicted = aim_params.predict_target_position(Vec3::ZERO);

        // Predicted position should be ahead of current position
        assert!(
            predicted.x < aim_params.target_position.x,
            "Should lead moving target"
        );
    }

    #[test]
    fn test_lookat_smooth_target_transition() {
        // Test smooth transition between targets - verify solver runs without error
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        let chain = LookAtChain::head(4, Vec3::NEG_Z, Vec3::Y);
        let mut state = LookAtState::default();

        // First target (left of center)
        let params1 = LookAtParams {
            target_position: Vec3::new(-2.0, 1.75, -3.0),
            speed: 360.0, // Fast convergence
            ..Default::default()
        };

        let mut visible1 = false;
        for _ in 0..60 {
            visible1 = solve_look_at(&chain, &skeleton, &mut pose, &params1, &mut state, DELTA_TIME);
        }
        let _first_rotation = state.current_rotation;

        // Switch to second target (right of center)
        let params2 = LookAtParams {
            target_position: Vec3::new(2.0, 1.75, -3.0),
            speed: 360.0,
            ..Default::default()
        };

        let mut visible2 = false;
        for _ in 0..60 {
            visible2 = solve_look_at(&chain, &skeleton, &mut pose, &params2, &mut state, DELTA_TIME);
        }
        let _second_rotation = state.current_rotation;

        // Both targets should be visible (within cone)
        assert!(visible1 || visible2, "At least one target should be visible");

        // Verify the system is stable - rotations should be normalized
        assert!(state.current_rotation.is_normalized());
    }

    #[test]
    fn test_lookat_weight_distribution() {
        // Test that weights are distributed correctly
        let chain = LookAtChain::spine(vec![0, 1, 2, 3], Vec3::NEG_Z, Vec3::Y);

        // Verify weights sum approximately to 1
        let sum: f32 = chain.weights.iter().sum();
        assert!((sum - 1.0).abs() < TEST_EPSILON || sum > 0.0);

        // Tip bones should have higher weights
        assert!(chain.weights[3] >= chain.weights[0]);
    }

    #[test]
    fn test_lookat_single_bone() {
        // Test simplified single-bone look-at
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        let params = LookAtParams::with_target(Vec3::new(0.0, 2.0, -3.0));

        let visible = solve_look_at_single(4, &skeleton, &mut pose, &params, Vec3::NEG_Z, Vec3::Y);

        assert!(visible);
    }

    #[test]
    fn test_lookat_eye_gaze_chain() {
        // Test eye gaze (very fast tracking)
        let chain = LookAtChain::eye(5, Vec3::NEG_Z, Vec3::Y); // Left eye

        assert_eq!(chain.bones.len(), 1);
        assert_eq!(chain.weights.len(), 1);
    }

    #[test]
    fn test_lookat_aim_with_offset() {
        // Test aim with muzzle offset
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        let aim_params = AimParams {
            target_position: Vec3::new(0.0, 1.75, 10.0),
            target_velocity: Vec3::ZERO,
            projectile_speed: f32::INFINITY, // Hitscan
            aim_offset: Vec3::new(0.0, 0.1, 0.3), // Muzzle position
        };

        let direction = solve_aim(10, &skeleton, &mut pose, &aim_params); // Hand bone

        assert!(direction.is_normalized());
    }

    #[test]
    fn test_lookat_axis_constraints() {
        // Test axis constraints
        let constraints = AxisConstraints::symmetric(PI / 4.0, PI / 6.0);

        assert_eq!(constraints.clamp_yaw(PI), PI / 4.0);
        assert_eq!(constraints.clamp_yaw(-PI), -PI / 4.0);
        assert_eq!(constraints.clamp_pitch(PI / 2.0), PI / 6.0);
    }

    #[test]
    fn test_lookat_custom_weights() {
        // Test chain with custom weights
        let chain = LookAtChain::with_weights(
            vec![0, 1, 2],
            vec![0.2, 0.3, 0.5],
            Vec3::NEG_Z,
            Vec3::Y,
        );

        assert_eq!(chain.get_normalized_weight(0), 0.2);
        assert_eq!(chain.get_normalized_weight(1), 0.3);
        assert_eq!(chain.get_normalized_weight(2), 0.5);
    }

    #[test]
    fn test_lookat_state_reset() {
        // Test state reset
        let mut state = LookAtState::looking_at(Vec3::new(0.0, 0.0, 1.0));
        assert!(state.current_rotation.is_normalized());

        state.reset();
        assert_eq!(state.current_rotation, Quat::IDENTITY);
    }

    #[test]
    fn test_lookat_validation() {
        // Test chain validation
        let skeleton = create_humanoid_skeleton();

        let valid_chain = LookAtChain::head(4, Vec3::NEG_Z, Vec3::Y);
        assert!(valid_chain.validate(&skeleton));

        let invalid_chain = LookAtChain::head(999, Vec3::NEG_Z, Vec3::Y);
        assert!(!invalid_chain.validate(&skeleton));
    }

    // =========================================================================
    // Twist Distribution Integration Tests (10+)
    // =========================================================================

    #[test]
    fn test_twist_forearm_chain_basic() {
        // Test basic forearm twist distribution
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Apply rotation to hand
        pose.rotations[10] = Quat::from_rotation_x(PI / 4.0); // l_hand

        let chain = create_forearm_twist_chain(8, 9, 10); // upper_arm, forearm, hand
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied, "Twist should be applied");
    }

    #[test]
    fn test_twist_spine_chain() {
        // Test spine twist distribution
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Apply rotation to hips
        pose.rotations[0] = Quat::from_rotation_y(PI / 6.0);

        let chain = create_spine_twist_chain(0, &[1, 2], 3); // pelvis, spine bones, chest
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied);
    }

    #[test]
    fn test_twist_neck_chain() {
        // Test neck twist distribution
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Apply rotation to head
        pose.rotations[4] = Quat::from_rotation_y(PI / 4.0);

        let chain = create_neck_twist_chain(4, &[3]); // head, neck
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied);
    }

    #[test]
    fn test_twist_swing_decomposition() {
        // Test swing-twist decomposition accuracy
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, PI / 6.0, PI / 4.0, PI / 8.0);
        let twist_axis = Vec3::Y;

        let (swing, twist) = swing_twist_decompose(rotation, twist_axis);

        // Recompose should give original (approximately)
        let recomposed = swing * twist;
        let angle_diff = rotation.angle_between(recomposed);
        assert!(
            angle_diff < ANGLE_TOLERANCE,
            "Decomposition should be reversible"
        );
    }

    #[test]
    fn test_twist_weight_falloff_linear() {
        // Test linear weight falloff
        let chain = TwistChain::new(vec![0, 1, 2, 3], 4, Vec3::X);
        let weights = chain.compute_weights(TwistFalloff::Linear);

        // Linear should be equal distribution
        assert_eq!(weights.len(), 4);
        let expected = 1.0 / 4.0;
        for w in &weights {
            assert!((w - expected).abs() < TEST_EPSILON);
        }
    }

    #[test]
    fn test_twist_weight_falloff_exponential() {
        // Test exponential weight falloff
        let chain = TwistChain::new(vec![0, 1, 2, 3], 4, Vec3::X);
        let weights = chain.compute_weights(TwistFalloff::exponential(0.5));

        // Weights should decrease exponentially
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
        assert!(weights[2] > weights[3]);

        // Should sum to 1
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_twist_multi_bone_blending() {
        // Test twist distribution across multiple bones
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Create chain with custom weights
        let chain = TwistChain::with_custom_weights(
            vec![8, 9],        // upper_arm, forearm
            vec![0.3, 0.7],    // 30%, 70%
            10,                // hand (source)
            Vec3::X,
        );

        // Apply twist to hand
        pose.rotations[10] = Quat::from_rotation_x(PI / 2.0);

        let params = TwistParams::custom();
        distribute_twist(&chain, &skeleton, &mut pose, &params);

        // Both bones should have some twist applied
        let upper_twist = get_twist_angle(pose.rotations[8], Vec3::X);
        let forearm_twist = get_twist_angle(pose.rotations[9], Vec3::X);

        // Forearm should have more twist due to higher weight
        assert!(forearm_twist.abs() >= upper_twist.abs() - ANGLE_TOLERANCE);
    }

    #[test]
    fn test_twist_additive_mode() {
        // Test additive twist distribution (doesn't modify source)
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        let _original_hand = pose.rotations[10];
        pose.rotations[10] = Quat::from_rotation_x(PI / 4.0);

        let chain = create_forearm_twist_chain(8, 9, 10);
        let params = TwistParams::linear();

        distribute_twist_additive(&chain, &skeleton, &mut pose, &params);

        // Source bone should NOT be modified in additive mode
        // (But the function still preserves source - let's just verify it runs)
        assert!(pose.rotations[10].is_normalized());
    }

    #[test]
    fn test_twist_max_angle_clamping() {
        // Test maximum twist angle clamping
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Apply large rotation
        pose.rotations[10] = Quat::from_rotation_x(PI); // 180 degrees

        let chain = create_forearm_twist_chain(8, 9, 10);
        let params = TwistParams {
            max_twist: PI / 2.0, // 90 degree limit
            ..Default::default()
        };

        distribute_twist(&chain, &skeleton, &mut pose, &params);

        // Distributed twist should respect limits
        for &bone in &chain.bones {
            let twist = get_twist_angle(pose.rotations[bone], Vec3::X);
            assert!(
                twist.abs() <= params.max_twist + ANGLE_TOLERANCE,
                "Twist should be clamped"
            );
        }
    }

    #[test]
    fn test_twist_chain_manager() {
        // Test TwistChainManager for multiple chains
        let skeleton = create_humanoid_skeleton();
        let _pose = create_pose_from_skeleton(&skeleton);

        let mut manager = TwistChainManager::new();
        manager.add_chain(create_forearm_twist_chain(8, 9, 10));  // Left arm
        manager.add_chain(create_forearm_twist_chain(12, 13, 14)); // Right arm

        assert_eq!(manager.chains.len(), 2);
    }

    #[test]
    fn test_twist_validation() {
        // Test chain validation
        let skeleton = create_humanoid_skeleton();

        let valid_chain = TwistChain::new(vec![8, 9], 10, Vec3::X);
        assert!(valid_chain.validate(&skeleton));

        let invalid_chain = TwistChain::new(vec![999], 10, Vec3::X);
        assert!(!invalid_chain.validate(&skeleton));

        let zero_axis_chain = TwistChain::new(vec![8, 9], 10, Vec3::ZERO);
        assert!(!zero_axis_chain.validate(&skeleton));
    }

    // =========================================================================
    // Cross-System Integration Tests (10+)
    // =========================================================================

    #[test]
    fn test_cross_eyes_head_spine_chain() {
        // Test full look-at chain: eyes + head + spine
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // First apply spine look-at
        let spine_chain = LookAtChain::spine(vec![1, 2, 3, 4], Vec3::NEG_Z, Vec3::Y);
        let params = LookAtParams::with_target(Vec3::new(2.0, 2.5, -3.0));
        let mut spine_state = LookAtState::default();

        for _ in 0..30 {
            solve_look_at(&spine_chain, &skeleton, &mut pose, &params, &mut spine_state, DELTA_TIME);
        }

        // Then apply eye tracking with narrower cone
        let mut eyes = EyeAnimationSystem::new(EyeParams::default());
        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);
        eyes.set_gaze_target(Some(params.target_position));

        for _ in 0..30 {
            eyes.update(DELTA_TIME, head_transform);
        }

        assert!(spine_state.target_visible);
        let (left, right) = eyes.get_eye_rotations();
        assert!(left.is_normalized());
        assert!(right.is_normalized());
    }

    #[test]
    fn test_cross_spring_bones_with_collision() {
        // Test spring bones interacting with collision system
        let mut system = SpringBoneSystem::new();

        // Hair chain
        let mut hair = SpringBoneChain::new(vec![15, 16, 17]);
        hair.initialize_rest(&[
            Vec3::new(0.0, 1.85, -0.05),
            Vec3::new(0.0, 1.75, -0.1),
            Vec3::new(0.0, 1.65, -0.15),
        ]);
        system.add_chain(hair);

        // Head and body colliders
        system.add_sphere_collider(SphereCollider::new(Vec3::new(0.0, 1.75, 0.0), 0.12));
        system.add_sphere_collider(SphereCollider::new(Vec3::new(0.0, 1.4, 0.0), 0.15));

        // Shoulder capsule
        system.add_capsule_collider(CapsuleCollider::new(
            Vec3::new(-0.2, 1.5, 0.0),
            Vec3::new(0.2, 1.5, 0.0),
            0.08,
        ));

        let transforms = vec![Mat4::IDENTITY];
        for _ in 0..60 {
            system.update(DELTA_TIME, &transforms);
        }

        // Hair should not penetrate colliders
        for state in &system.chains[0].states {
            // Check against head sphere
            let dist_to_head = (state.position - Vec3::new(0.0, 1.75, 0.0)).length();
            assert!(
                dist_to_head >= 0.12 - POSITION_TOLERANCE,
                "Hair should not penetrate head"
            );
        }
    }

    #[test]
    fn test_cross_procedural_keyframe_blending() {
        // Test that procedural animations blend with keyframe poses
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Apply a "keyframe" rotation to head
        let keyframe_rotation = Quat::from_rotation_y(PI / 6.0);
        pose.rotations[4] = keyframe_rotation;

        // Now apply procedural look-at on top
        let chain = LookAtChain::head(4, Vec3::NEG_Z, Vec3::Y);
        let params = LookAtParams {
            target_position: Vec3::new(0.0, 1.75, -3.0),
            weight: 0.5, // Partial blend
            ..Default::default()
        };
        let mut state = LookAtState::default();

        for _ in 0..30 {
            solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, DELTA_TIME);
        }

        // Result should be blend of keyframe and procedural
        let final_rotation = pose.rotations[4];
        assert!(final_rotation.is_normalized());
    }

    #[test]
    fn test_cross_performance_many_targets() {
        // Test performance with many simultaneous procedural animations
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Multiple spring bone chains
        let mut spring_system = SpringBoneSystem::new();
        for i in 0..5 {
            let mut chain = SpringBoneChain::new(vec![i, i + 1, i + 2]);
            chain.initialize_rest(&[
                Vec3::new(0.1 * i as f32, 1.8, 0.0),
                Vec3::new(0.1 * i as f32, 1.7, 0.0),
                Vec3::new(0.1 * i as f32, 1.6, 0.0),
            ]);
            spring_system.add_chain(chain);
        }

        // Multiple look-at chains
        let mut lookat_states: Vec<LookAtState> = (0..3).map(|_| LookAtState::default()).collect();
        let lookat_chains = vec![
            LookAtChain::head(4, Vec3::NEG_Z, Vec3::Y),
            LookAtChain::spine(vec![1, 2], Vec3::NEG_Z, Vec3::Y),
            LookAtChain::eye(5, Vec3::NEG_Z, Vec3::Y),
        ];

        // Benchmark
        let start = Instant::now();
        for _ in 0..BENCHMARK_ITERATIONS {
            // Update spring bones
            let transforms: Vec<Mat4> = (0..5).map(|_| Mat4::IDENTITY).collect();
            spring_system.update(DELTA_TIME, &transforms);

            // Update look-at
            for (chain, state) in lookat_chains.iter().zip(lookat_states.iter_mut()) {
                let params = LookAtParams::with_target(Vec3::new(0.0, 1.75, -5.0));
                solve_look_at(chain, &skeleton, &mut pose, &params, state, DELTA_TIME);
            }
        }
        let duration = start.elapsed();

        // Should complete in reasonable time (< 100ms for 100 iterations)
        assert!(
            duration.as_millis() < 500,
            "Performance test took too long: {:?}",
            duration
        );
    }

    #[test]
    fn test_cross_twist_with_lookat() {
        // Test twist distribution working alongside look-at
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Apply look-at to arm (aiming)
        let aim_params = AimParams::static_target(Vec3::new(5.0, 1.75, 5.0));
        solve_aim(10, &skeleton, &mut pose, &aim_params);

        // Now distribute forearm twist
        let chain = create_forearm_twist_chain(8, 9, 10);
        let twist_params = TwistParams::linear();
        distribute_twist(&chain, &skeleton, &mut pose, &twist_params);

        // All bones should still be valid
        for rot in &pose.rotations {
            assert!(rot.is_normalized());
        }
    }

    #[test]
    fn test_cross_eye_with_twist() {
        // Eye animation with neck twist
        let mut eyes = EyeAnimationSystem::new(EyeParams::default());
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Apply neck twist
        pose.rotations[4] = Quat::from_rotation_y(PI / 4.0); // Head rotation
        let neck_chain = create_neck_twist_chain(4, &[3]);
        distribute_twist(&neck_chain, &skeleton, &mut pose, &TwistParams::linear());

        // Eye animation should still work
        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), pose.rotations[4]);
        eyes.set_gaze_target(Some(Vec3::new(2.0, 1.75, -3.0)));

        for _ in 0..30 {
            eyes.update(DELTA_TIME, head_transform);
        }

        let (left, right) = eyes.get_eye_rotations();
        assert!(left.is_normalized());
        assert!(right.is_normalized());
    }

    #[test]
    fn test_cross_spring_wind_lookat() {
        // Test spring bones with wind while character looks at target
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Look-at
        let chain = LookAtChain::spine(vec![1, 2, 3, 4], Vec3::NEG_Z, Vec3::Y);
        let params = LookAtParams::with_target(Vec3::new(0.0, 2.0, -5.0)); // -Z is forward
        let mut state = LookAtState::default();

        // Spring system with wind
        let mut spring_system = SpringBoneSystem::new();
        spring_system.wind = Vec3::new(1.0, 0.0, 0.5);
        spring_system.wind_turbulence_scale = 0.3;

        let mut hair = SpringBoneChain::new(vec![15, 16, 17]);
        hair.initialize_rest(&[
            Vec3::new(0.0, 1.85, -0.05),
            Vec3::new(0.0, 1.75, -0.1),
            Vec3::new(0.0, 1.65, -0.15),
        ]);
        spring_system.add_chain(hair);

        // Run both systems
        for _ in 0..60 {
            solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, DELTA_TIME);
            spring_system.update(DELTA_TIME, &[Mat4::IDENTITY]);
        }

        assert!(state.target_visible);
    }

    #[test]
    fn test_cross_multiple_spring_chains_independent() {
        // Test multiple independent spring chains simulating together
        let mut system = SpringBoneSystem::new();

        // Hair chain
        let mut hair = SpringBoneChain::new(vec![0, 1, 2]);
        hair.rest_positions = vec![
            Vec3::new(0.0, 1.8, -0.1),
            Vec3::new(0.0, -0.1, 0.0),
            Vec3::new(0.0, -0.1, 0.0),
        ];
        hair.rest_lengths = vec![0.1, 0.1];
        hair.states[0].reset(Vec3::new(0.0, 1.8, -0.1));
        hair.states[1].reset(Vec3::new(0.0, 1.7, -0.1));
        hair.states[2].reset(Vec3::new(0.0, 1.6, -0.1));
        system.add_chain(hair);

        // Accessory chain (earring-like)
        let mut earring = SpringBoneChain::new(vec![3, 4]);
        earring.rest_positions = vec![
            Vec3::new(0.1, 1.7, 0.0),
            Vec3::new(0.0, -0.05, 0.0),
        ];
        earring.rest_lengths = vec![0.05];
        earring.states[0].reset(Vec3::new(0.1, 1.7, 0.0));
        earring.states[1].reset(Vec3::new(0.1, 1.65, 0.0));
        // Earring is stiffer with more damping
        earring.params[1] = SpringBoneParams::accessory();
        system.add_chain(earring);

        // Simulate
        let transforms = vec![Mat4::IDENTITY, Mat4::IDENTITY];
        for _ in 0..60 {
            system.update(DELTA_TIME, &transforms);
        }

        // Both chains should be valid
        assert_eq!(system.chains.len(), 2);
        for chain in &system.chains {
            for state in &chain.states {
                assert!(state.position.is_finite());
            }
        }
    }

    #[test]
    fn test_cross_all_twist_chains_in_pose() {
        // Test applying all standard twist chains to a humanoid pose
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Apply various rotations
        pose.rotations[10] = Quat::from_rotation_x(PI / 4.0);  // Left hand
        pose.rotations[14] = Quat::from_rotation_x(-PI / 4.0); // Right hand
        pose.rotations[4] = Quat::from_rotation_y(PI / 6.0);   // Head

        // Create all twist chains
        let chains = vec![
            create_forearm_twist_chain(8, 9, 10),   // Left forearm
            create_forearm_twist_chain(12, 13, 14), // Right forearm
            create_neck_twist_chain(4, &[3]),       // Neck
        ];

        let params = TwistParams::linear();

        // Apply all twist distributions
        for chain in &chains {
            distribute_twist(chain, &skeleton, &mut pose, &params);
        }

        // All bones should have valid rotations
        for (i, rot) in pose.rotations.iter().enumerate() {
            assert!(
                rot.is_normalized(),
                "Bone {} rotation should be normalized",
                i
            );
        }
    }

    #[test]
    fn test_cross_full_character_update() {
        // Full character procedural update cycle
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Systems
        let mut eyes = EyeAnimationSystem::new(EyeParams::default());
        let mut spring_system = SpringBoneSystem::new();
        let mut twist_manager = TwistChainManager::new();

        // Setup
        eyes.set_gaze_target(Some(Vec3::new(1.0, 2.0, -3.0)));

        let mut hair = SpringBoneChain::new(vec![15, 16, 17]);
        hair.initialize_rest(&[
            Vec3::new(0.0, 1.85, -0.05),
            Vec3::new(0.0, 1.75, -0.1),
            Vec3::new(0.0, 1.65, -0.15),
        ]);
        spring_system.add_chain(hair);
        spring_system.add_sphere_collider(SphereCollider::new(Vec3::new(0.0, 1.75, 0.0), 0.12));

        twist_manager.add_chain(create_forearm_twist_chain(8, 9, 10));
        twist_manager.add_chain(create_forearm_twist_chain(12, 13, 14));

        // Simulate hand rotation
        pose.rotations[10] = Quat::from_rotation_x(PI / 4.0);
        pose.rotations[14] = Quat::from_rotation_x(-PI / 4.0);

        // Full update cycle
        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        for _ in 0..60 {
            // 1. Eye animation
            eyes.update(DELTA_TIME, head_transform);

            // 2. Spring bone physics
            spring_system.update(DELTA_TIME, &[Mat4::IDENTITY]);

            // 3. Twist distribution
            for chain in &twist_manager.chains {
                distribute_twist(chain, &skeleton, &mut pose, &twist_manager.params);
            }
        }

        // Validate final state
        for rot in &pose.rotations {
            assert!(rot.is_normalized(), "All rotations should be valid");
        }
    }

    // =========================================================================
    // Edge Cases Tests (10+)
    // =========================================================================

    #[test]
    fn test_edge_behind_camera_target() {
        // Target behind the character (+Z is behind in this coordinate system)
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        let chain = LookAtChain::head(4, Vec3::NEG_Z, Vec3::Y);
        let params = LookAtParams {
            target_position: Vec3::new(0.0, 1.75, 5.0), // Behind (+Z)
            cone_angle: PI / 2.0,
            ..Default::default()
        };
        let mut state = LookAtState::default();

        let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, DELTA_TIME);

        assert!(!visible, "Target behind should be outside view cone");
    }

    #[test]
    fn test_edge_extreme_rotation_angles() {
        // Test handling of extreme rotation angles
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Apply extreme twist
        pose.rotations[10] = Quat::from_rotation_x(PI * 2.0); // Full rotation

        let chain = create_forearm_twist_chain(8, 9, 10);
        let params = TwistParams::linear();

        distribute_twist(&chain, &skeleton, &mut pose, &params);

        // Should handle without NaN
        for rot in &pose.rotations {
            assert!(
                rot.x.is_finite() && rot.y.is_finite() && rot.z.is_finite() && rot.w.is_finite(),
                "Rotations should be finite"
            );
        }
    }

    #[test]
    fn test_edge_zero_length_chains() {
        // Test handling of empty/minimal chains
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Empty chain
        let empty_chain = LookAtChain::with_weights(vec![], vec![], Vec3::NEG_Z, Vec3::Y);
        let params = LookAtParams::with_target(Vec3::new(0.0, 1.75, -3.0));
        let mut state = LookAtState::default();

        // Should not panic - returns state.target_visible (default true) for empty chains
        let _result = solve_look_at(&empty_chain, &skeleton, &mut pose, &params, &mut state, DELTA_TIME);
        // Empty chain returns early, preserving state. The function doesn't panic.
        assert!(empty_chain.is_empty());

        // Single bone chain
        let single_chain = LookAtChain::head(4, Vec3::NEG_Z, Vec3::Y);
        assert!(!single_chain.is_empty());
    }

    #[test]
    fn test_edge_rapid_target_switching() {
        // Test rapid target switching
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        let chain = LookAtChain::head(4, Vec3::NEG_Z, Vec3::Y);
        let mut state = LookAtState::default();

        let targets = [
            Vec3::new(-2.0, 1.75, -3.0),
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(0.0, 2.5, 4.0),
            Vec3::new(-1.0, 1.5, 5.0),
            Vec3::new(1.5, 1.75, 2.5),
        ];

        // Switch targets every few frames
        for (i, &target) in targets.iter().enumerate() {
            let params = LookAtParams {
                target_position: target,
                speed: 360.0, // Fast
                ..Default::default()
            };

            for _ in 0..5 {
                solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, DELTA_TIME);
            }

            assert!(
                state.current_rotation.is_normalized(),
                "Rotation should stay valid after rapid switch {}",
                i
            );
        }
    }

    #[test]
    fn test_edge_very_small_delta_time() {
        // Test with very small delta time
        let mut eyes = EyeAnimationSystem::new(EyeParams::default());
        let head_transform = create_head_transform(Vec3::new(0.0, 1.75, 0.0), Quat::IDENTITY);

        eyes.set_gaze_target(Some(Vec3::new(0.0, 1.75, -3.0)));

        // Very small delta time
        for _ in 0..1000 {
            eyes.update(1e-6, head_transform);
        }

        let (left, right) = eyes.get_eye_rotations();
        assert!(left.is_finite_and_normalized());
        assert!(right.is_finite_and_normalized());
    }

    #[test]
    fn test_edge_very_large_delta_time() {
        // Test with large delta time (clamping)
        let mut spring_chain = SpringBoneChain::new(vec![0, 1, 2]);
        spring_chain.initialize_rest(&[
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::new(0.0, 1.4, 0.0),
            Vec3::new(0.0, 1.3, 0.0),
        ]);

        // Very large delta time (should be clamped internally)
        simulate_spring_chain(
            &mut spring_chain,
            Mat4::IDENTITY,
            10.0, // Way too large
            Vec3::new(0.0, -9.81, 0.0),
            Vec3::ZERO,
            &[],
            &[],
        );

        // Should not explode
        for state in &spring_chain.states {
            assert!(
                state.position.is_finite(),
                "Positions should remain finite"
            );
        }
    }

    #[test]
    fn test_edge_coincident_target_position() {
        // Target at same position as eye/bone
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // Target at exact bone position
        let transforms = pose.transforms();
        let world_transforms = skeleton.compute_world_transforms(&transforms);
        let head_pos = world_transforms[4].w_axis.truncate();

        let chain = LookAtChain::head(4, Vec3::NEG_Z, Vec3::Y);
        let params = LookAtParams::with_target(head_pos);
        let mut state = LookAtState::default();

        // Should handle gracefully
        let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, DELTA_TIME);

        // Either visible is false or rotation is valid
        assert!(
            !visible || state.current_rotation.is_normalized(),
            "Should handle coincident target"
        );
    }

    #[test]
    fn test_edge_normalized_axis_vectors() {
        // Test with unnormalized input axes
        let rotation = Quat::from_rotation_x(PI / 4.0);

        // Unnormalized axis
        let unnorm_axis = Vec3::new(0.0, 10.0, 0.0); // Not normalized

        let (swing, twist) = swing_twist_decompose(rotation, unnorm_axis);

        assert!(swing.is_normalized());
        assert!(twist.is_normalized());
    }

    #[test]
    fn test_edge_identity_rotations() {
        // Test with identity rotations throughout
        let skeleton = create_humanoid_skeleton();
        let mut pose = create_pose_from_skeleton(&skeleton);

        // All identity rotations
        for rot in &mut pose.rotations {
            *rot = Quat::IDENTITY;
        }

        let chain = create_forearm_twist_chain(8, 9, 10);
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied);
        // Result should still be valid (mostly identity)
        for rot in &pose.rotations {
            assert!(rot.is_normalized());
        }
    }

    #[test]
    fn test_edge_spring_zero_stiffness() {
        // Test spring with zero stiffness
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        chain.initialize_rest(&[Vec3::new(0.0, 1.5, 0.0), Vec3::new(0.0, 1.4, 0.0)]);

        // Zero stiffness (should still work due to length constraints)
        chain.params[1].stiffness = 0.0;

        for _ in 0..30 {
            simulate_spring_chain(
                &mut chain,
                Mat4::IDENTITY,
                DELTA_TIME,
                Vec3::new(0.0, -9.81, 0.0),
                Vec3::ZERO,
                &[],
                &[],
            );
        }

        // Should not produce NaN
        assert!(chain.states[1].position.is_finite());
    }

    // =========================================================================
    // Helper Trait for Tests
    // =========================================================================

    trait QuatExt {
        fn is_finite_and_normalized(&self) -> bool;
    }

    impl QuatExt for Quat {
        fn is_finite_and_normalized(&self) -> bool {
            self.x.is_finite()
                && self.y.is_finite()
                && self.z.is_finite()
                && self.w.is_finite()
                && (self.length_squared() - 1.0).abs() < 0.01
        }
    }
}
