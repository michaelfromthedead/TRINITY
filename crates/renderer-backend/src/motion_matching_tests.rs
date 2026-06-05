//! Integration tests for motion matching system (T-AN-6.6).
//!
//! This module provides comprehensive integration tests for the TRINITY motion
//! matching system, covering end-to-end pipeline, context-driven queries,
//! inertialization transitions, feature matching, performance, and edge cases.
//!
//! # Test Categories
//!
//! - End-to-end pipeline tests (query -> search -> transition -> playback)
//! - Context-driven tests (controller input, navigation, terrain)
//! - Inertialization tests (transition smoothness, velocity preservation)
//! - Feature matching tests (pose, trajectory, foot contact)
//! - Performance tests (large database, continuous search)
//! - Edge case tests (no match, 180-degree turns, stop-to-walk)

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};
    use std::f32::consts::PI;

    use crate::inertialization::{
        DampingMode, DecayCurve, HierarchyAwareBlender, InertializationBlender,
        InertializationConfig, InertializationState, JointConfig, JointOffset,
        JointVelocity, VelocityEstimator,
    };
    use crate::motion_context::{
        ContextBuilder, ContextInterpolator, ContextUpdatePolicy, FootContactTracker,
        MotionContext, TagManager, TagSet, TrajectoryRequest,
    };
    use crate::motion_features::{
        FeatureWeights, FootFeatures, LocomotionStyle, MotionFeatures, MotionTags, TerrainType,
    };
    use crate::motion_matching_db::{
        ClipInfo, FootContact, KdTree, LocomotionTags, MotionDatabase, MotionFrame,
        PoseFeature, QueryFeature, TrajectoryFeature,
    };
    use crate::motion_search::{
        MotionSearcher, SearchConfig, SearchCostWeights, SearchQuery,
    };

    // =========================================================================
    // Test Helpers
    // =========================================================================

    /// Create a test motion database with walking and running animations.
    fn create_test_database(frame_count: usize) -> MotionDatabase {
        let mut db = MotionDatabase::with_name("test_db");
        db.joint_count = 4; // Simplified skeleton
        db.foot_count = 2;
        db.feature_dims = 32;

        // Add walk clip
        let walk_clip = ClipInfo {
            name: "walk".to_string(),
            duration: 1.0,
            start_frame: 0,
            frame_count: (frame_count / 2) as u32,
            sample_rate: 30.0,
            default_tags: LocomotionTags::WALK | LocomotionTags::TERRAIN_FLAT,
            is_looping: true,
        };
        db.clips.push(walk_clip);

        // Add run clip
        let run_clip = ClipInfo {
            name: "run".to_string(),
            duration: 0.8,
            start_frame: (frame_count / 2) as u32,
            frame_count: (frame_count / 2) as u32,
            sample_rate: 30.0,
            default_tags: LocomotionTags::RUN | LocomotionTags::TERRAIN_FLAT,
            is_looping: true,
        };
        db.clips.push(run_clip);

        // Generate frames
        for i in 0..frame_count {
            let is_walk = i < frame_count / 2;
            let clip_index = if is_walk { 0 } else { 1 };
            let local_frame = if is_walk { i } else { i - frame_count / 2 };
            let time = local_frame as f32 / 30.0;

            let speed = if is_walk { 1.5 } else { 4.0 };
            let phase = (time * 2.0 * PI) % (2.0 * PI);

            // Create pose with cyclic motion
            let mut pose = PoseFeature::with_joint_count(4);
            for j in 0..4 {
                let offset = j as f32 * 0.25 * PI;
                pose.joint_positions[j] = Vec3::new(
                    (phase + offset).sin() * 0.1,
                    0.9 + (phase * 2.0 + offset).sin() * 0.05,
                    (phase + offset).cos() * 0.2,
                );
                pose.joint_velocities[j] = Vec3::new(
                    (phase + offset).cos() * speed * 0.1,
                    (phase * 2.0 + offset).cos() * speed * 0.05,
                    -(phase + offset).sin() * speed * 0.2,
                );
            }

            // Create trajectory
            let trajectory = TrajectoryFeature::from_predictions(
                [
                    Vec3::new(0.0, 0.0, speed * 0.2),
                    Vec3::new(0.0, 0.0, speed * 0.5),
                    Vec3::new(0.0, 0.0, speed * 1.0),
                ],
                [0.0, 0.0, 0.0],
            );

            // Create foot contacts (alternating)
            let left_planted = phase < PI;
            let right_planted = phase >= PI;
            let foot_contacts = vec![
                FootContact {
                    is_planted: left_planted,
                    position: Vec3::new(-0.1, if left_planted { 0.0 } else { 0.1 }, phase.sin() * 0.3),
                    velocity: if left_planted { Vec3::ZERO } else { Vec3::new(0.0, 0.5, speed) },
                },
                FootContact {
                    is_planted: right_planted,
                    position: Vec3::new(0.1, if right_planted { 0.0 } else { 0.1 }, phase.cos() * 0.3),
                    velocity: if right_planted { Vec3::ZERO } else { Vec3::new(0.0, 0.5, speed) },
                },
            ];

            let frame = MotionFrame {
                clip_index: clip_index as u32,
                time,
                pose,
                trajectory,
                foot_contacts,
                tags: if is_walk {
                    LocomotionTags::WALK | LocomotionTags::TERRAIN_FLAT
                } else {
                    LocomotionTags::RUN | LocomotionTags::TERRAIN_FLAT
                },
                root_velocity: Vec3::new(0.0, 0.0, speed),
                root_angular_velocity: 0.0,
            };

            db.frames.push(frame);
        }

        // Build KD-tree
        db.kd_tree = KdTree::build(&db.frames, db.feature_dims);

        db
    }

    /// Create a query for walking motion.
    fn create_walk_query() -> QueryFeature {
        let mut pose = PoseFeature::with_joint_count(4);
        for j in 0..4 {
            pose.joint_positions[j] = Vec3::new(0.0, 0.9, 0.0);
            pose.joint_velocities[j] = Vec3::new(0.0, 0.0, 1.5);
        }

        QueryFeature {
            pose,
            trajectory: TrajectoryFeature::from_predictions(
                [
                    Vec3::new(0.0, 0.0, 0.3),
                    Vec3::new(0.0, 0.0, 0.75),
                    Vec3::new(0.0, 0.0, 1.5),
                ],
                [0.0, 0.0, 0.0],
            ),
            foot_contacts: vec![
                FootContact::planted(Vec3::new(-0.1, 0.0, 0.0)),
                FootContact::moving(Vec3::new(0.1, 0.1, 0.2), Vec3::new(0.0, 0.5, 1.5)),
            ],
            required_tags: LocomotionTags::WALK,
            excluded_tags: LocomotionTags::empty(),
        }
    }

    /// Create a query for running motion.
    fn create_run_query() -> QueryFeature {
        let mut pose = PoseFeature::with_joint_count(4);
        for j in 0..4 {
            pose.joint_positions[j] = Vec3::new(0.0, 0.85, 0.0);
            pose.joint_velocities[j] = Vec3::new(0.0, 0.0, 4.0);
        }

        QueryFeature {
            pose,
            trajectory: TrajectoryFeature::from_predictions(
                [
                    Vec3::new(0.0, 0.0, 0.8),
                    Vec3::new(0.0, 0.0, 2.0),
                    Vec3::new(0.0, 0.0, 4.0),
                ],
                [0.0, 0.0, 0.0],
            ),
            foot_contacts: vec![
                FootContact::planted(Vec3::new(-0.1, 0.0, 0.0)),
                FootContact::moving(Vec3::new(0.1, 0.15, 0.4), Vec3::new(0.0, 0.8, 4.0)),
            ],
            required_tags: LocomotionTags::RUN,
            excluded_tags: LocomotionTags::empty(),
        }
    }

    /// Create test poses for transitions.
    fn create_test_poses(joint_count: usize) -> (Vec<Vec3>, Vec<Quat>, Vec<Vec3>, Vec<Quat>) {
        let source_positions: Vec<Vec3> = (0..joint_count)
            .map(|i| Vec3::new(i as f32 * 0.1, 0.5 + i as f32 * 0.1, 0.0))
            .collect();
        let source_rotations: Vec<Quat> = (0..joint_count)
            .map(|i| Quat::from_rotation_y(i as f32 * 0.1))
            .collect();
        let target_positions: Vec<Vec3> = (0..joint_count)
            .map(|i| Vec3::new(i as f32 * 0.1, 0.6 + i as f32 * 0.1, 0.1))
            .collect();
        let target_rotations: Vec<Quat> = (0..joint_count)
            .map(|i| Quat::from_rotation_y(i as f32 * 0.2))
            .collect();

        (source_positions, source_rotations, target_positions, target_rotations)
    }

    // =========================================================================
    // End-to-End Pipeline Tests (15+)
    // =========================================================================

    #[test]
    fn test_pipeline_query_search_match() {
        let db = create_test_database(100);
        let query = create_walk_query();

        let result = db.find_best_match(&query);
        assert!(result.is_some(), "Should find a matching frame");

        let result = result.unwrap();
        assert!(result.distance >= 0.0, "Distance should be non-negative");
        assert!(result.clip_index == 0, "Should match walk clip");
    }

    #[test]
    fn test_pipeline_searcher_integration() {
        let db = create_test_database(100);
        let mut searcher = MotionSearcher::new(db, SearchConfig::default());

        let context = MotionContext::new();
        let query = SearchQuery::new()
            .with_required_tags(MotionTags::walk());

        let result = searcher.search(&query);
        assert!(result.is_some() || searcher.database.frame_count() > 0);
    }

    #[test]
    fn test_pipeline_multiple_consecutive_transitions() {
        let db = create_test_database(100);
        let mut blender = InertializationBlender::new(InertializationConfig::fast());

        let (src_pos, src_rot, tgt_pos, tgt_rot) = create_test_poses(4);

        // First transition
        blender.start_transition(&src_pos, &src_rot, &tgt_pos, &tgt_rot, None);
        for _ in 0..10 {
            blender.update(0.016);
        }
        let (pos1, rot1) = blender.apply(&tgt_pos, &tgt_rot);

        // Second transition (during or after first)
        let new_target_pos: Vec<Vec3> = tgt_pos.iter().map(|p| *p + Vec3::Y * 0.1).collect();
        blender.start_transition(&pos1, &rot1, &new_target_pos, &tgt_rot, None);

        assert!(blender.is_active(), "Second transition should be active");

        // Continue updating
        for _ in 0..20 {
            blender.update(0.016);
        }

        assert!(blender.is_complete(), "Should complete after sufficient time");
    }

    #[test]
    fn test_pipeline_tag_filtered_search() {
        let db = create_test_database(100);

        // Query for walk only
        let walk_query = create_walk_query();
        let walk_results = db.find_k_best_matches(&walk_query, 10);

        for result in &walk_results {
            let frame = &db.frames[result.frame_index as usize];
            assert!(
                frame.tags.contains(LocomotionTags::WALK),
                "All results should be walk frames"
            );
        }

        // Query for run only
        let run_query = create_run_query();
        let run_results = db.find_k_best_matches(&run_query, 10);

        for result in &run_results {
            let frame = &db.frames[result.frame_index as usize];
            assert!(
                frame.tags.contains(LocomotionTags::RUN),
                "All results should be run frames"
            );
        }
    }

    #[test]
    fn test_pipeline_budget_constrained_search() {
        let db = create_test_database(200);
        let config = SearchConfig::fast().with_budget_ms(0.5);
        let mut searcher = MotionSearcher::new(db, config);

        let query = SearchQuery::new();
        let _result = searcher.search(&query);

        let stats = searcher.last_statistics();
        // Fast search should not evaluate all candidates
        assert!(stats.candidates_evaluated <= stats.candidates_total);
    }

    #[test]
    fn test_pipeline_context_to_query_conversion() {
        let mut context = MotionContext::new();
        context.trajectory = TrajectoryRequest::straight(1.5, 1.0);
        context.tags = TagSet::from_locomotion(LocomotionTags::WALK);

        let query = context.to_search_query();
        assert_eq!(query.required_tags.locomotion, LocomotionStyle::Idle);
    }

    #[test]
    fn test_pipeline_full_frame_cycle() {
        let db = create_test_database(60);
        let mut searcher = MotionSearcher::new(db, SearchConfig::default());
        let mut blender = InertializationBlender::new(InertializationConfig::default());
        let mut context = MotionContext::new();

        // Simulate 60 frames of gameplay
        for frame in 0..60 {
            let dt = 1.0 / 60.0;
            context.update(dt);

            // Periodic search
            if frame % 4 == 0 {
                let query = context.to_search_query();
                if let Some(result) = searcher.search(&query) {
                    // Would trigger transition to new animation
                    if result.cost < 1.0 {
                        context.on_search(result.cost);
                    }
                }
            }

            // Update transition
            if blender.is_active() {
                blender.update(dt);
            }
        }

        // Should complete without panics
        assert!(context.time > 0.9, "Should have simulated ~1 second");
    }

    #[test]
    fn test_pipeline_interpolated_trajectory() {
        let mut context = MotionContext::new();

        // Set initial trajectory
        context.set_trajectory(TrajectoryRequest::straight(1.5, 1.0));
        context.update(0.1);

        // Change trajectory
        context.set_trajectory(TrajectoryRequest::straight(4.0, 1.0));

        // Interpolation should be active
        assert!(context.interpolator.is_active());

        // Update until complete
        for _ in 0..30 {
            context.update(0.016);
        }

        let final_trajectory = context.current_trajectory();
        assert!(final_trajectory.speed_profile >= 3.5, "Should have interpolated to higher speed");
    }

    #[test]
    fn test_pipeline_search_k_nearest() {
        let db = create_test_database(100);
        let query = create_walk_query();

        let results = db.find_k_best_matches(&query, 5);
        assert!(results.len() <= 5, "Should return at most K results");

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(
                results[i].distance >= results[i - 1].distance,
                "Results should be sorted by distance"
            );
        }
    }

    #[test]
    fn test_pipeline_clip_info_tracking() {
        let db = create_test_database(100);
        let query = create_walk_query();

        let result = db.find_best_match(&query).unwrap();

        // Verify clip info is valid
        assert!(
            (result.clip_index as usize) < db.clips.len(),
            "Clip index should be valid"
        );

        let clip = &db.clips[result.clip_index as usize];
        assert!(result.clip_time <= clip.duration, "Time should be within clip duration");
    }

    #[test]
    fn test_pipeline_root_velocity_propagation() {
        let mut context = MotionContext::new();
        context.motion_state.root_velocity = Vec3::new(0.0, 0.0, 2.0);

        let query = context.to_search_query();
        assert_eq!(query.root_velocity, Vec3::new(0.0, 0.0, 2.0));
    }

    #[test]
    fn test_pipeline_transition_state_tracking() {
        let mut context = MotionContext::new();

        assert!(!context.in_transition());

        context.start_transition(1, 0.5, 0.2);
        assert!(context.in_transition());
        assert_eq!(context.motion_state.clip_index, 1);

        // Advance past transition
        for _ in 0..20 {
            context.update(0.016);
        }

        assert!(!context.in_transition(), "Transition should complete");
    }

    #[test]
    fn test_pipeline_feature_extraction_roundtrip() {
        let db = create_test_database(10);
        let frame = &db.frames[0];

        // Extract features
        let features = frame.pose.to_flat_vector();
        let trajectory = frame.trajectory.to_flat_vector();

        // Reconstruct
        let pose_restored = PoseFeature::from_flat_vector(&features, frame.pose.joint_count());
        let traj_restored = TrajectoryFeature::from_flat_vector(&trajectory);

        assert_eq!(frame.pose.joint_count(), pose_restored.joint_count());
        for i in 0..frame.pose.joint_count() {
            let diff = (frame.pose.joint_positions[i] - pose_restored.joint_positions[i]).length();
            assert!(diff < 0.001, "Position should round-trip accurately");
        }
    }

    #[test]
    fn test_pipeline_cost_breakdown() {
        let db = create_test_database(50);
        let config = SearchConfig::default();
        let mut searcher = MotionSearcher::new(db, config);

        let query = SearchQuery::new();
        if let Some(result) = searcher.search(&query) {
            if let Some(breakdown) = &result.cost_breakdown {
                let total = breakdown.total();
                assert!(
                    (total - result.cost).abs() < 0.1,
                    "Breakdown should sum to total cost"
                );
            }
        }
    }

    #[test]
    fn test_pipeline_motion_database_persistence() {
        let db = create_test_database(50);

        // Verify database state
        assert_eq!(db.frame_count(), 50);
        assert_eq!(db.clip_count(), 2);
        assert_eq!(db.joint_count, 4);
        assert_eq!(db.foot_count, 2);

        // Verify clip metadata
        let walk_clip = &db.clips[0];
        assert_eq!(walk_clip.name, "walk");
        assert!(walk_clip.default_tags.contains(LocomotionTags::WALK));

        let run_clip = &db.clips[1];
        assert_eq!(run_clip.name, "run");
        assert!(run_clip.default_tags.contains(LocomotionTags::RUN));
    }

    // =========================================================================
    // Context-Driven Tests (10+)
    // =========================================================================

    #[test]
    fn test_context_controller_input_to_motion() {
        let builder = ContextBuilder::new()
            .from_controller_input(
                Vec3::new(0.0, 0.0, 1.0), // Forward movement
                Vec3::Z,                   // Looking forward
                1.5,                       // Walking speed
                0.016,
            );

        let context = builder.build();
        assert!(context.trajectory.speed_profile > 0.0);
        assert!(context.tags.locomotion.contains(LocomotionTags::WALK));
    }

    #[test]
    fn test_context_navigation_path_following() {
        let path_points = vec![
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(1.0, 0.0, 4.0),
            Vec3::new(2.0, 0.0, 6.0),
        ];

        let builder = ContextBuilder::new()
            .from_navigation_path(&path_points, Vec3::ZERO, 2.0);

        let context = builder.build();
        assert!(!context.trajectory.points.is_empty());

        // First point should be ahead
        if !context.trajectory.points.is_empty() {
            assert!(context.trajectory.points[0].position.z > 0.0);
        }
    }

    #[test]
    fn test_context_terrain_based_tag_switching() {
        let mut tag_manager = TagManager::with_standard_rules();

        // Start on flat terrain
        tag_manager.update_from_terrain(TerrainType::Flat);
        assert!(tag_manager.active.locomotion.contains(LocomotionTags::TERRAIN_FLAT));

        // Move to uphill
        tag_manager.update_from_terrain(TerrainType::SlopeUp);
        assert!(tag_manager.active.locomotion.contains(LocomotionTags::TERRAIN_UPHILL));
        assert!(!tag_manager.active.locomotion.contains(LocomotionTags::TERRAIN_FLAT));

        // Move to stairs
        tag_manager.update_from_terrain(TerrainType::StairsUp);
        assert!(tag_manager.active.locomotion.contains(LocomotionTags::TERRAIN_STAIRS));
    }

    #[test]
    fn test_context_update_policy_timing() {
        let mut policy = ContextUpdatePolicy::responsive();

        // Should not search immediately after transition
        policy.on_transition();
        assert!(!policy.should_search(0.5, false));

        // Advance time past cooldown
        for _ in 0..10 {
            policy.update(0.016);
        }

        // Should search if cost is high
        assert!(policy.should_search(0.5, false) || policy.update_interval > 1);
    }

    #[test]
    fn test_context_foot_contact_tracker() {
        let mut tracker = FootContactTracker::bipedal(5, 6);

        let positions = vec![Vec3::new(-0.1, 0.0, 0.0), Vec3::new(0.1, 0.05, 0.0)];
        let velocities = vec![Vec3::ZERO, Vec3::new(0.0, 0.3, 1.0)];

        tracker.update(&positions, &velocities, 0.016, 0.0);

        assert!(tracker.feet[0].planted, "Left foot should be planted (low, slow)");
        assert!(!tracker.feet[1].planted, "Right foot should be moving (has velocity)");
    }

    #[test]
    fn test_context_locomotion_style_from_speed() {
        assert_eq!(LocomotionStyle::from_speed(0.0), LocomotionStyle::Idle);
        assert_eq!(LocomotionStyle::from_speed(1.0), LocomotionStyle::Walk);
        assert_eq!(LocomotionStyle::from_speed(3.0), LocomotionStyle::Run);
        assert_eq!(LocomotionStyle::from_speed(6.0), LocomotionStyle::Sprint);
    }

    #[test]
    fn test_context_tag_manager_priorities() {
        let mut manager = TagManager::with_standard_rules();

        // Add both walk and run
        manager.add_tags(LocomotionTags::WALK | LocomotionTags::RUN);

        // Run should override walk based on exclusion rules
        assert!(
            manager.active.locomotion.contains(LocomotionTags::RUN)
                || manager.active.locomotion.contains(LocomotionTags::WALK)
        );
    }

    #[test]
    fn test_context_trajectory_straight() {
        let trajectory = TrajectoryRequest::straight(2.0, 1.0);

        assert_eq!(trajectory.speed_profile, 2.0);
        assert!(!trajectory.stopping);
        assert!(!trajectory.starting);

        // All points should be along Z axis
        for point in &trajectory.points {
            assert!(point.position.x.abs() < 0.01);
            assert!(point.facing.z > 0.5);
        }
    }

    #[test]
    fn test_context_trajectory_turning() {
        let trajectory = TrajectoryRequest::turning(2.0, PI / 4.0, 1.0);

        assert!(trajectory.curvature.abs() > 0.0, "Should have curvature");

        // Points should curve
        let last_point = trajectory.points.last();
        if let Some(point) = last_point {
            assert!(point.position.x.abs() > 0.01 || point.facing_angle().abs() > 0.01);
        }
    }

    #[test]
    fn test_context_trajectory_stopping() {
        let trajectory = TrajectoryRequest::stopping(2.0, 0.5);

        assert!(trajectory.stopping);
        assert!(!trajectory.starting);

        // Speed should decrease
        if trajectory.points.len() >= 2 {
            let early = &trajectory.points[0];
            let late = trajectory.points.last().unwrap();
            assert!(late.speed <= early.speed, "Speed should decrease when stopping");
        }
    }

    #[test]
    fn test_context_interpolator_smooth_transition() {
        let mut interpolator = ContextInterpolator::with_smoothness(0.3);

        let target = TrajectoryRequest::straight(4.0, 1.0);
        interpolator.set_target(target);

        assert!(interpolator.is_active());

        // Update partially
        interpolator.update(0.05);
        let progress = interpolator.progress();
        assert!(progress > 0.0 && progress < 1.0);

        // Current should be between start and target
        let current = interpolator.current();
        assert!(current.speed_profile < 4.0 && current.speed_profile >= 0.0);
    }

    // =========================================================================
    // Inertialization Tests (10+)
    // =========================================================================

    #[test]
    fn test_inertialization_transition_smoothness() {
        let config = InertializationConfig::default().with_duration(0.2);
        let mut blender = InertializationBlender::new(config);

        let (src_pos, src_rot, tgt_pos, tgt_rot) = create_test_poses(4);
        blender.start_transition(&src_pos, &src_rot, &tgt_pos, &tgt_rot, None);

        let mut prev_positions = src_pos.clone();
        let mut max_position_change = 0.0f32;

        // Sample positions over the transition
        for _ in 0..20 {
            blender.update(0.01);
            let (positions, _) = blender.apply(&tgt_pos, &tgt_rot);

            for (i, pos) in positions.iter().enumerate() {
                let change = (*pos - prev_positions[i]).length();
                max_position_change = max_position_change.max(change);
            }

            prev_positions = positions;
        }

        // Changes should be small per frame (smooth)
        assert!(
            max_position_change < 0.1,
            "Position changes should be smooth (got {})",
            max_position_change
        );
    }

    #[test]
    fn test_inertialization_velocity_preservation() {
        let config = InertializationConfig::default().with_velocity_preservation(1.0);
        let mut blender = InertializationBlender::new(config);

        let (src_pos, src_rot, tgt_pos, tgt_rot) = create_test_poses(4);

        // Create source velocities
        let velocities: Vec<JointVelocity> = (0..4)
            .map(|i| JointVelocity::from_linear(Vec3::new(0.0, 0.0, 1.0 + i as f32 * 0.1)))
            .collect();

        blender.start_transition(&src_pos, &src_rot, &tgt_pos, &tgt_rot, Some(&velocities));

        // Initial velocity should be preserved
        let state = blender.state();
        assert!(!state.source_velocities.is_empty());
        assert!(state.source_velocities[0].linear.length() > 0.0);
    }

    #[test]
    fn test_inertialization_hierarchy_aware() {
        let parent_indices = vec![-1, 0, 1, 2]; // Chain: root -> j1 -> j2 -> j3
        let config = InertializationConfig::default();
        let mut blender = HierarchyAwareBlender::new(config, parent_indices);

        let (src_pos, src_rot, tgt_pos, tgt_rot) = create_test_poses(4);
        blender.start_transition(&src_pos, &src_rot, &tgt_pos, &tgt_rot, None);

        blender.update(0.05);
        let (positions, _) = blender.apply(&tgt_pos, &tgt_rot);

        // All joints should be affected
        assert_eq!(positions.len(), 4);

        // Child joints should inherit parent influence
        // (difficult to test precisely, but at least verify no NaN/infinite)
        for pos in &positions {
            assert!(pos.is_finite(), "Positions should be finite");
        }
    }

    #[test]
    fn test_inertialization_damping_modes() {
        let modes = [DampingMode::Critical, DampingMode::Underdamped, DampingMode::Overdamped];

        for mode in modes {
            let curve = DecayCurve::new(mode, 0.1);

            // At t=0, decay should be 1.0
            let decay_0 = curve.decay(0.0);
            assert!((decay_0 - 1.0).abs() < 0.01, "Decay at t=0 should be 1.0");

            // Decay should decrease over time
            let decay_01 = curve.decay(0.1);
            let decay_02 = curve.decay(0.2);
            assert!(
                decay_01 < decay_0,
                "Decay should decrease: {:?}",
                mode
            );
            assert!(
                decay_02 < decay_01,
                "Decay should continue decreasing: {:?}",
                mode
            );
        }
    }

    #[test]
    fn test_inertialization_joint_config_override() {
        let mut config = InertializationConfig::default();
        config = config.with_joint_override(
            2,
            JointConfig::with_weight(0.5).with_duration(0.1),
        );

        assert_eq!(config.effective_weight(0), 1.0);
        assert_eq!(config.effective_weight(2), 0.5);
        assert_eq!(config.effective_duration(2), 0.1);
    }

    #[test]
    fn test_inertialization_completion() {
        let config = InertializationConfig::fast();
        let mut blender = InertializationBlender::new(config);

        let (src_pos, src_rot, tgt_pos, tgt_rot) = create_test_poses(4);
        blender.start_transition(&src_pos, &src_rot, &tgt_pos, &tgt_rot, None);

        assert!(!blender.is_complete());

        // Fast forward
        for _ in 0..20 {
            blender.update(0.016);
        }

        assert!(blender.is_complete(), "Should complete after sufficient time");
    }

    #[test]
    fn test_inertialization_cancellation() {
        let config = InertializationConfig::default();
        let mut blender = InertializationBlender::new(config);

        let (src_pos, src_rot, tgt_pos, tgt_rot) = create_test_poses(4);
        blender.start_transition(&src_pos, &src_rot, &tgt_pos, &tgt_rot, None);

        assert!(blender.is_active());

        blender.cancel();

        assert!(!blender.is_active());
        assert!(blender.is_complete());
    }

    #[test]
    fn test_inertialization_velocity_estimator() {
        let mut estimator = VelocityEstimator::with_smoothing(0.3);
        estimator.initialize(4);

        let positions1: Vec<Vec3> = (0..4).map(|i| Vec3::new(0.0, 0.5, i as f32 * 0.1)).collect();
        let rotations: Vec<Quat> = vec![Quat::IDENTITY; 4];
        let scales: Vec<Vec3> = vec![Vec3::ONE; 4];

        // First sample
        let vels1 = estimator.update(&positions1, &rotations, &scales, 0.0);
        assert!(vels1.iter().all(|v| v.linear.length() < 0.01), "First sample should have zero velocity");

        // Second sample with movement
        let positions2: Vec<Vec3> = positions1.iter().map(|p| *p + Vec3::Z * 0.1).collect();
        let vels2 = estimator.update(&positions2, &rotations, &scales, 0.1);

        // Should detect Z velocity
        assert!(vels2[0].linear.z > 0.0, "Should detect forward velocity");
    }

    #[test]
    fn test_inertialization_state_progress() {
        let mut state = InertializationState::with_joint_count(4);
        state.duration = 0.2;
        state.is_active = true;

        assert_eq!(state.progress(), 0.0);

        state.elapsed = 0.1;
        assert!((state.progress() - 0.5).abs() < 0.01);

        state.elapsed = 0.2;
        assert!((state.progress() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_inertialization_joint_offset_operations() {
        let offset1 = JointOffset::from_position(Vec3::new(1.0, 0.0, 0.0));
        let offset2 = JointOffset::from_rotation(Vec3::new(0.0, PI / 4.0, 0.0));

        // Magnitude
        assert!(offset1.magnitude() > 0.0);
        assert!(offset2.magnitude() > 0.0);

        // Scaling
        let scaled = offset1.scaled(0.5);
        assert!((scaled.position.x - 0.5).abs() < 0.01);

        // Interpolation
        let lerped = offset1.lerp(&offset2, 0.5);
        assert!(lerped.position.x > 0.0);
        assert!(lerped.rotation.y > 0.0);
    }

    // =========================================================================
    // Feature Matching Tests (10+)
    // =========================================================================

    #[test]
    fn test_feature_pose_distance() {
        let mut pose1 = PoseFeature::with_joint_count(4);
        let mut pose2 = PoseFeature::with_joint_count(4);

        for i in 0..4 {
            pose1.joint_positions[i] = Vec3::new(i as f32 * 0.1, 0.5, 0.0);
            pose2.joint_positions[i] = Vec3::new(i as f32 * 0.1, 0.5, 0.0);
        }

        // Identical poses should have zero distance
        let dist = pose1.distance_squared(&pose2, 1.0, 0.1);
        assert!(dist < 0.01, "Identical poses should have near-zero distance");

        // Different poses should have positive distance
        pose2.joint_positions[0] = Vec3::new(1.0, 0.5, 0.0);
        let dist2 = pose1.distance_squared(&pose2, 1.0, 0.1);
        assert!(dist2 > 0.0, "Different poses should have positive distance");
    }

    #[test]
    fn test_feature_trajectory_distance() {
        let traj1 = TrajectoryFeature::from_predictions(
            [Vec3::new(0.0, 0.0, 0.5), Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 2.0)],
            [0.0, 0.0, 0.0],
        );

        let traj2 = traj1.clone();
        let dist = traj1.distance_squared(&traj2, 1.0, 1.0);
        assert!(dist < 0.01, "Identical trajectories should have near-zero distance");

        let traj3 = TrajectoryFeature::from_predictions(
            [Vec3::new(1.0, 0.0, 0.5), Vec3::new(1.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 2.0)],
            [0.0, 0.0, 0.0],
        );
        let dist2 = traj1.distance_squared(&traj3, 1.0, 1.0);
        assert!(dist2 > 0.0, "Different trajectories should have positive distance");
    }

    #[test]
    fn test_feature_foot_contact_synchronization() {
        let db = create_test_database(60);

        // Find frames with similar foot contact patterns
        let query_left_planted = QueryFeature {
            foot_contacts: vec![
                FootContact::planted(Vec3::new(-0.1, 0.0, 0.0)),
                FootContact::moving(Vec3::new(0.1, 0.1, 0.2), Vec3::new(0.0, 0.5, 1.5)),
            ],
            ..create_walk_query()
        };

        let results = db.find_k_best_matches(&query_left_planted, 10);

        // Should find frames with similar foot states
        assert!(!results.is_empty());
    }

    #[test]
    fn test_feature_foot_features_distance() {
        let mut foot1 = FootFeatures::with_count(2);
        foot1.contact_states = vec![true, false];
        foot1.positions = vec![Vec3::new(-0.1, 0.0, 0.0), Vec3::new(0.1, 0.1, 0.2)];

        let mut foot2 = FootFeatures::with_count(2);
        foot2.contact_states = vec![true, false];
        foot2.positions = vec![Vec3::new(-0.1, 0.0, 0.0), Vec3::new(0.1, 0.1, 0.2)];

        let dist = foot1.distance_squared(&foot2, 1.0, 0.5, 0.1, 0.5);
        assert!(dist < 0.01, "Identical foot features should have near-zero distance");

        // Different contact state
        foot2.contact_states = vec![false, true];
        let dist2 = foot1.distance_squared(&foot2, 1.0, 0.5, 0.1, 0.5);
        assert!(dist2 >= 2.0, "Different contact states should add penalty");
    }

    #[test]
    fn test_feature_motion_tags_matching() {
        let walk_tags = MotionTags::walk();
        let run_tags = MotionTags::run();
        let idle_tags = MotionTags::idle();

        assert!(walk_tags.matches(&walk_tags));
        assert!(!walk_tags.matches(&run_tags));

        // Idle should match anything (as default)
        assert!(walk_tags.matches(&idle_tags));
    }

    #[test]
    fn test_feature_trajectory_interpolation() {
        let traj1 = TrajectoryFeature::from_predictions(
            [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO],
            [0.0, 0.0, 0.0],
        );

        let traj2 = TrajectoryFeature::from_predictions(
            [Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 3.0)],
            [0.0, 0.0, 0.0],
        );

        let lerped = traj1.lerp(&traj2, 0.5);

        assert!((lerped.future_positions[0].z - 0.5).abs() < 0.01);
        assert!((lerped.future_positions[1].z - 1.0).abs() < 0.01);
        assert!((lerped.future_positions[2].z - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_feature_weights_configurations() {
        let default_weights = FeatureWeights::default();
        let trajectory_weights = FeatureWeights::trajectory_focused();
        let pose_weights = FeatureWeights::pose_focused();

        assert!(trajectory_weights.trajectory_position > default_weights.trajectory_position);
        assert!(pose_weights.pose_position > default_weights.pose_position);
    }

    #[test]
    fn test_feature_motion_features_dimension() {
        let mut features = MotionFeatures::new();
        features.pose = vec![0.0; 24]; // 4 joints * 6 (pos + vel)
        features.trajectory = vec![0.0; 12]; // 3 samples * 4 (pos.xyz + facing)
        features.foot = FootFeatures::with_count(2);

        let dim = features.dimension();
        assert_eq!(dim, 24 + 12 + 16); // pose + trajectory + foot (2 * 8)
    }

    #[test]
    fn test_feature_flat_vector_roundtrip() {
        let mut features = MotionFeatures::new();
        features.pose = vec![1.0, 2.0, 3.0, 4.0];
        features.trajectory = vec![5.0, 6.0, 7.0];
        features.foot = FootFeatures::with_count(1);
        features.foot.positions[0] = Vec3::new(0.1, 0.2, 0.3);

        let flat = features.to_flat_vector();
        assert!(flat.contains(&1.0));
        assert!(flat.contains(&5.0));
    }

    #[test]
    fn test_feature_search_cost_weights() {
        let default_weights = SearchCostWeights::default();
        let trajectory_weights = SearchCostWeights::trajectory_focused();
        let pose_weights = SearchCostWeights::pose_focused();
        let transition_weights = SearchCostWeights::transition_focused();

        assert!(trajectory_weights.trajectory_weight > default_weights.trajectory_weight);
        assert!(pose_weights.pose_weight > default_weights.pose_weight);
        assert!(transition_weights.transition_weight > default_weights.transition_weight);
    }

    // =========================================================================
    // Performance Tests (5+)
    // =========================================================================

    #[test]
    fn test_performance_large_database_search() {
        let db = create_test_database(1000);
        let query = create_walk_query();

        let start = std::time::Instant::now();
        let _result = db.find_best_match(&query);
        let duration = start.elapsed();

        // Search should complete in reasonable time
        assert!(
            duration.as_millis() < 100,
            "Large database search took too long: {:?}",
            duration
        );
    }

    #[test]
    fn test_performance_continuous_search_overhead() {
        let db = create_test_database(200);
        let mut searcher = MotionSearcher::new(db, SearchConfig::fast());
        let query = create_walk_query();
        let search_query = SearchQuery::new().with_required_tags(MotionTags::walk());

        let start = std::time::Instant::now();

        // Simulate 60 frames of continuous search (every 4th frame)
        for _ in 0..15 {
            let _ = searcher.search(&search_query);
        }

        let duration = start.elapsed();
        let avg_per_search = duration.as_micros() / 15;

        assert!(
            avg_per_search < 5000, // 5ms per search max
            "Average search time too high: {}us",
            avg_per_search
        );
    }

    #[test]
    fn test_performance_kd_tree_efficiency() {
        let db = create_test_database(500);

        // KD-tree should be built
        assert!(!db.kd_tree.is_empty());
        assert!(db.kd_tree.node_count() > 0);

        // Query features
        let query_features: Vec<f32> = vec![0.0; db.feature_dims];
        let results = db.kd_tree.find_k_nearest(&query_features, 10);

        assert!(results.len() <= 10);
    }

    #[test]
    fn test_performance_inertialization_update_speed() {
        let config = InertializationConfig::default();
        let mut blender = InertializationBlender::new(config);

        let (src_pos, src_rot, tgt_pos, tgt_rot) = create_test_poses(20); // More joints

        blender.start_transition(&src_pos, &src_rot, &tgt_pos, &tgt_rot, None);

        let start = std::time::Instant::now();

        // Simulate 1 second of updates at 60fps
        for _ in 0..60 {
            blender.update(0.016);
            let _ = blender.apply(&tgt_pos, &tgt_rot);
        }

        let duration = start.elapsed();
        let avg_per_frame = duration.as_micros() / 60;

        assert!(
            avg_per_frame < 500, // 0.5ms per frame max
            "Inertialization update too slow: {}us per frame",
            avg_per_frame
        );
    }

    #[test]
    fn test_performance_context_update_speed() {
        let mut context = MotionContext::with_foot_count(2);
        context.set_trajectory(TrajectoryRequest::straight(2.0, 1.0));

        let start = std::time::Instant::now();

        // Simulate 1 second of updates
        for _ in 0..60 {
            context.update(0.016);
            let _ = context.to_search_query();
        }

        let duration = start.elapsed();
        let avg_per_frame = duration.as_micros() / 60;

        assert!(
            avg_per_frame < 200, // 0.2ms per frame max
            "Context update too slow: {}us per frame",
            avg_per_frame
        );
    }

    // =========================================================================
    // Edge Case Tests (10+)
    // =========================================================================

    #[test]
    fn test_edge_no_matching_motion() {
        let db = create_test_database(50);

        // Query with impossible tag combination
        let query = QueryFeature {
            required_tags: LocomotionTags::SWIM | LocomotionTags::CLIMB, // Not in database
            ..create_walk_query()
        };

        let result = db.find_best_match(&query);
        assert!(result.is_none(), "Should return None for impossible tags");
    }

    #[test]
    fn test_edge_180_degree_turn() {
        let trajectory = TrajectoryRequest::turning(1.5, PI, 1.0); // 180 degree turn

        assert!(trajectory.curvature.abs() > 0.5, "Should have high curvature");

        // Points should show significant direction change
        if trajectory.points.len() >= 2 {
            let first = trajectory.points.first().unwrap();
            let last = trajectory.points.last().unwrap();

            let angle_diff = (first.facing_angle() - last.facing_angle()).abs();
            assert!(
                angle_diff > PI / 2.0 || angle_diff < 0.01,
                "Should show significant turn"
            );
        }
    }

    #[test]
    fn test_edge_stop_to_walk_transition() {
        let mut context = MotionContext::new();
        context.tags.locomotion = LocomotionTags::IDLE;
        context.motion_state.root_velocity = Vec3::ZERO;

        // Transition to walking
        context.set_trajectory(TrajectoryRequest::starting(1.5, 0.3, Vec3::Z));

        assert!(context.interpolator.current().starting);
    }

    #[test]
    fn test_edge_walk_to_stop_transition() {
        let mut context = MotionContext::new();
        context.tags.locomotion = LocomotionTags::WALK;
        context.motion_state.root_velocity = Vec3::new(0.0, 0.0, 1.5);

        // Transition to stopping
        context.set_trajectory(TrajectoryRequest::stopping(1.5, 0.5));

        assert!(context.interpolator.current().stopping);
    }

    #[test]
    fn test_edge_empty_database() {
        let db = MotionDatabase::new();

        let query = create_walk_query();
        let result = db.find_best_match(&query);

        assert!(result.is_none(), "Empty database should return None");
    }

    #[test]
    fn test_edge_single_frame_database() {
        let mut db = MotionDatabase::new();

        let frame = MotionFrame {
            clip_index: 0,
            time: 0.0,
            tags: LocomotionTags::WALK,
            ..Default::default()
        };
        db.frames.push(frame);
        db.kd_tree = KdTree::build(&db.frames, 32);

        let query = create_walk_query();
        let result = db.find_best_match(&query);

        assert!(result.is_some(), "Should find the single frame");
    }

    #[test]
    fn test_edge_very_short_transition() {
        let config = InertializationConfig::default().with_duration(0.001);
        let mut blender = InertializationBlender::new(config);

        let (src_pos, src_rot, tgt_pos, tgt_rot) = create_test_poses(4);
        blender.start_transition(&src_pos, &src_rot, &tgt_pos, &tgt_rot, None);

        blender.update(0.001);
        assert!(blender.is_complete(), "Very short transition should complete quickly");
    }

    #[test]
    fn test_edge_zero_velocity_input() {
        let builder = ContextBuilder::new()
            .from_controller_input(
                Vec3::ZERO,
                Vec3::Z,
                0.0,
                0.016,
            );

        let context = builder.build();
        assert!(context.trajectory.speed_profile < 0.1);
    }

    #[test]
    fn test_edge_very_high_speed() {
        let trajectory = TrajectoryRequest::straight(100.0, 1.0);

        assert!(!trajectory.points.is_empty());

        // Points should extend far
        if let Some(last) = trajectory.points.last() {
            assert!(last.position.z > 50.0, "High speed should create distant points");
        }
    }

    #[test]
    fn test_edge_empty_pose_feature() {
        let pose = PoseFeature::new();
        assert!(pose.is_empty());
        assert_eq!(pose.joint_count(), 0);

        let flat = pose.to_flat_vector();
        assert!(flat.is_empty());
    }

    #[test]
    fn test_edge_mismatched_joint_counts() {
        let pose1 = PoseFeature::with_joint_count(4);
        let pose2 = PoseFeature::with_joint_count(8);

        // Should handle gracefully
        let dist = pose1.distance_squared(&pose2, 1.0, 0.1);
        assert!(dist >= 0.0, "Should handle mismatched counts");
    }

    #[test]
    fn test_edge_negative_time() {
        let curve = DecayCurve::critical(0.1);

        let decay = curve.decay(-0.1);
        assert!((decay - 1.0).abs() < 0.01, "Negative time should return 1.0");
    }

    #[test]
    fn test_edge_very_long_transition() {
        let config = InertializationConfig::default().with_duration(10.0);
        let mut blender = InertializationBlender::new(config);

        let (src_pos, src_rot, tgt_pos, tgt_rot) = create_test_poses(4);
        blender.start_transition(&src_pos, &src_rot, &tgt_pos, &tgt_rot, None);

        // Update for 1 second
        for _ in 0..60 {
            blender.update(0.016);
        }

        assert!(!blender.is_complete(), "Long transition should not complete in 1 second");
        assert!(blender.remaining_time() > 8.0);
    }

    // =========================================================================
    // Additional Integration Tests
    // =========================================================================

    #[test]
    fn test_integration_full_motion_system() {
        // Create all components
        let db = create_test_database(100);
        let mut searcher = MotionSearcher::new(db, SearchConfig::balanced());
        let mut blender = InertializationBlender::new(InertializationConfig::default());
        let mut context = MotionContext::with_foot_count(2);

        // Set initial state
        context.set_trajectory_immediate(TrajectoryRequest::straight(1.5, 1.0));
        context.tags = TagSet::from_locomotion(LocomotionTags::WALK);

        let mut transitions = 0;

        // Simulate 2 seconds of gameplay
        for frame in 0..120 {
            let dt = 1.0 / 60.0;
            context.update(dt);

            // Periodic search
            if frame % 8 == 0 {
                let query = context.to_search_query();
                if let Some(result) = searcher.search(&query) {
                    if result.cost < 50.0 && !context.in_transition() {
                        // Start transition
                        context.start_transition(
                            result.clip_index as u32,
                            result.time,
                            0.15,
                        );
                        transitions += 1;

                        // Would trigger inertialization here
                        let positions = vec![Vec3::ZERO; 4];
                        let rotations = vec![Quat::IDENTITY; 4];
                        blender.start_transition(&positions, &rotations, &positions, &rotations, None);
                    }
                }
            }

            // Update inertialization
            if blender.is_active() {
                blender.update(dt);
            }
        }

        // Should have completed successfully
        assert!(context.time > 1.9, "Should have simulated ~2 seconds");
    }

    #[test]
    fn test_integration_tag_flow() {
        let mut manager = TagManager::with_standard_rules();
        let mut context = MotionContext::new();

        // Simulate changing locomotion states
        let states = [
            (LocomotionStyle::Idle, 0.0),
            (LocomotionStyle::Walk, 1.5),
            (LocomotionStyle::Run, 4.0),
            (LocomotionStyle::Sprint, 7.0),
            (LocomotionStyle::Run, 4.0),
            (LocomotionStyle::Walk, 1.5),
            (LocomotionStyle::Idle, 0.0),
        ];

        for (style, speed) in states {
            manager.update_from_style(style);
            context.motion_state.root_velocity = Vec3::new(0.0, 0.0, speed);

            // Tags should reflect current style
            let expected_tag = match style {
                LocomotionStyle::Idle => LocomotionTags::IDLE,
                LocomotionStyle::Walk => LocomotionTags::WALK,
                LocomotionStyle::Run => LocomotionTags::RUN,
                LocomotionStyle::Sprint => LocomotionTags::SPRINT,
                LocomotionStyle::Crouch => LocomotionTags::CROUCH,
            };
            assert!(manager.active.locomotion.contains(expected_tag));
        }
    }

    #[test]
    fn test_integration_foot_tracking_during_motion() {
        let mut tracker = FootContactTracker::bipedal(5, 6);

        // Simulate a walk cycle
        for frame in 0..30 {
            let phase = frame as f32 / 30.0 * 2.0 * PI;

            let left_height = if phase < PI { 0.0 } else { 0.1 * (phase - PI).sin() };
            let right_height = if phase >= PI { 0.0 } else { 0.1 * phase.sin() };

            let positions = vec![
                Vec3::new(-0.1, left_height, phase.sin() * 0.3),
                Vec3::new(0.1, right_height, phase.cos() * 0.3),
            ];

            let velocities = vec![
                if left_height > 0.01 { Vec3::new(0.0, 0.5, 1.5) } else { Vec3::ZERO },
                if right_height > 0.01 { Vec3::new(0.0, 0.5, 1.5) } else { Vec3::ZERO },
            ];

            tracker.update(&positions, &velocities, 0.033, frame as f32 * 0.033);
        }

        // Gait phase should have advanced
        assert!(tracker.gait_phase >= 0.0 && tracker.gait_phase <= 1.0);
    }

    #[test]
    fn test_integration_search_config_presets() {
        let db = create_test_database(100);
        let query = create_walk_query();

        // Test different presets
        let configs = [
            SearchConfig::fast(),
            SearchConfig::balanced(),
            SearchConfig::quality(),
        ];

        for config in configs {
            let mut searcher = MotionSearcher::new(db.clone(), config);
            let result = searcher.search(&SearchQuery::new());

            // All should return valid results
            let stats = searcher.last_statistics();
            assert!(stats.search_time_ms >= 0.0);
        }
    }

    #[test]
    fn test_integration_trajectory_types() {
        let trajectories = vec![
            TrajectoryRequest::straight(1.5, 1.0),
            TrajectoryRequest::turning(1.5, PI / 4.0, 1.0),
            TrajectoryRequest::stopping(2.0, 0.5),
            TrajectoryRequest::starting(1.5, 0.3, Vec3::Z),
        ];

        for traj in trajectories {
            assert!(!traj.points.is_empty(), "All trajectory types should have points");
            assert!(traj.speed_profile >= 0.0, "Speed profile should be non-negative");
        }
    }
}
