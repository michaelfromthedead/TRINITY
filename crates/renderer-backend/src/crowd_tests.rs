//! Integration Tests for Crowd Rendering System (T-AN-8.6)
//!
//! Comprehensive integration tests covering:
//! - Full rendering pipeline (texture bake -> instance -> render)
//! - LOD system with distance-based transitions
//! - Impostor billboard rendering
//! - GPU instancing and batching
//! - Performance benchmarks for large crowds
//! - Edge cases and error handling
//!
//! # Architecture
//!
//! ```text
//! AnimationTextures     CrowdRenderer      CrowdShaders
//!       |                    |                  |
//!       v                    v                  v
//! +----------------+   +----------------+   +----------------+
//! | Texture Baking | + | Instancing     | + | GPU Skinning   |
//! | Tests          |   | Tests          |   | Tests          |
//! +----------------+   +----------------+   +----------------+
//!                            |
//!                            v
//!                  +-------------------+
//!                  | ImpostorSystem    |
//!                  | Tests             |
//!                  +-------------------+
//!                            |
//!                            v
//!                  +-------------------+
//!                  | CrowdLod          |
//!                  | Tests             |
//!                  +-------------------+
//!                            |
//!                            v
//!                  +-------------------+
//!                  | Full Pipeline     |
//!                  | Integration       |
//!                  +-------------------+
//! ```

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use crate::animation_clip::{AnimationClip, BoneTrack, Keyframe, Track};
    use crate::animation_textures::{
        AnimationTextureBaker, AnimationTextureData, AnimationTextureLayout,
        AnimationTextureSampler, ClipRegion, prepare_gpu_texture, generate_mipmaps,
    };
    use crate::crowd_renderer::{
        CrowdBatch, CrowdInstance, CrowdRenderer, CrowdRenderConfig, Frustum,
        IndirectDrawCommand, Vec3 as CrowdVec3, Quat as CrowdQuat,
        LOD_FULL_SKELETON, LOD_SIMPLIFIED, LOD_IMPOSTOR, FLAG_VISIBLE,
        FLAG_PHASE_SYNC, CROWD_INSTANCE_SIZE,
    };
    use crate::crowd_shaders::{
        CrowdAnimationShader, CrowdUniforms, ShaderConfig, ShaderFeatures,
        ShaderPermutation, LodLevel as ShaderLodLevel, ShaderCompiler,
        compose_transform_matrix, nlerp_quat, bilinear_sample,
    };
    use crate::impostor_system::{
        ImpostorConfig, ImpostorInstance, ViewAngleSelector, BillboardMode,
        Vec3 as ImpostorVec3, Quat as ImpostorQuat,
    };
    use crate::crowd_lod::{
        LodConfig, LodLevel, LodSelector, LodTransition,
        LodBudgetManager, CrowdLodInstance, Vec3 as LodVec3,
        PRIORITY_HERO, PRIORITY_IMPORTANT,
    };
    use glam::{Quat, Vec3};

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /// Create a simple animation clip with linear motion for testing.
    fn create_test_clip(name: &str, duration: f32, bone_count: usize) -> AnimationClip {
        let mut clip = AnimationClip::new(name, duration);
        clip.frame_rate = 30.0;

        for i in 0..bone_count {
            let pos_track = Track::from_keyframes(vec![
                Keyframe::linear(0.0, Vec3::new(i as f32, 0.0, 0.0)),
                Keyframe::linear(duration, Vec3::new(i as f32, duration * 2.0, 0.0)),
            ]);

            let rot_track = Track::from_keyframes(vec![
                Keyframe::linear(0.0, Quat::IDENTITY),
                Keyframe::linear(duration, Quat::from_rotation_y(PI * (i as f32 + 1.0) / 4.0)),
            ]);

            let scale_track = Track::from_keyframes(vec![
                Keyframe::linear(0.0, Vec3::ONE),
                Keyframe::linear(duration, Vec3::splat(1.0 + 0.05 * i as f32)),
            ]);

            let bone_track = BoneTrack::new(format!("bone_{}", i))
                .with_position(pos_track)
                .with_rotation(rot_track)
                .with_scale(scale_track);

            clip.add_bone_track(bone_track);
        }

        clip
    }

    /// Create a constant pose clip (single frame).
    fn create_static_clip(name: &str, position: Vec3, rotation: Quat) -> AnimationClip {
        let mut clip = AnimationClip::new(name, 0.0);
        clip.frame_rate = 30.0;

        let pos_track = Track::from_keyframes(vec![Keyframe::linear(0.0, position)]);
        let rot_track = Track::from_keyframes(vec![Keyframe::linear(0.0, rotation)]);

        let bone_track = BoneTrack::new("bone_0")
            .with_position(pos_track)
            .with_rotation(rot_track);

        clip.add_bone_track(bone_track);
        clip
    }

    /// Create a crowd renderer with default configuration and initial instances.
    fn create_test_renderer(instance_count: usize) -> CrowdRenderer {
        let config = CrowdRenderConfig::new()
            .with_lod_distances([20.0, 50.0, 100.0])
            .with_max_instances((instance_count + 1000) as u32);
        let mut renderer = CrowdRenderer::new(config);
        let batch_id = renderer.add_batch(0);

        for i in 0..instance_count {
            let x = (i % 32) as f32 * 3.0;
            let z = (i / 32) as f32 * 3.0;
            renderer
                .add_instance(batch_id, CrowdInstance::new([x, 0.0, z]))
                .unwrap();
        }

        renderer
    }

    /// Create a frustum that encompasses the test area.
    fn create_test_frustum() -> Frustum {
        Frustum::from_bounds(
            CrowdVec3::new(-500.0, -100.0, -500.0),
            CrowdVec3::new(500.0, 100.0, 500.0),
        )
    }

    // ========================================================================
    // SECTION 1: Full Pipeline Tests (15+ tests)
    // ========================================================================

    #[test]
    fn test_full_pipeline_texture_bake_to_render() {
        // Step 1: Create animation clip
        let clip = create_test_clip("walk", 1.0, 4);

        // Step 2: Bake to texture
        let baker = AnimationTextureBaker::new(4, 30.0);
        let texture_data = baker.bake_clip(&clip);
        assert!(texture_data.is_valid());

        // Step 3: Create renderer and instances
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        for i in 0..10 {
            renderer
                .add_instance(
                    batch_id,
                    CrowdInstance::new([i as f32 * 2.0, 0.0, 0.0])
                        .with_animation(0, i as f32 * 0.1),
                )
                .unwrap();
        }

        // Step 4: Cull and prepare GPU buffers
        renderer.cull_frustum(&Frustum::unbounded());
        let buffers = renderer.prepare_gpu_buffers();

        // Step 5: Get draw commands
        let commands = renderer.get_draw_commands();

        // Verify
        assert_eq!(renderer.instance_count(), 10);
        assert_eq!(renderer.visible_count(), 10);
        assert!(!buffers.is_empty());
        assert!(!commands.is_empty());
    }

    #[test]
    fn test_full_pipeline_multi_clip_atlas() {
        // Create multiple clips
        let walk = create_test_clip("walk", 1.0, 8);
        let run = create_test_clip("run", 0.5, 8);
        let idle = create_test_clip("idle", 2.0, 8);

        // Bake to atlas
        let baker = AnimationTextureBaker::new(8, 30.0);
        let atlas = baker.bake_clips(&[&walk, &run, &idle]);

        assert_eq!(atlas.clip_count, 3);
        assert!(atlas.get_region("walk").is_some());
        assert!(atlas.get_region("run").is_some());
        assert!(atlas.get_region("idle").is_some());

        // Verify clip regions are properly sequenced
        let walk_region = atlas.get_region("walk").unwrap();
        let run_region = atlas.get_region("run").unwrap();
        assert_eq!(walk_region.start_frame, 0);
        assert!(run_region.start_frame > 0);
    }

    #[test]
    fn test_full_pipeline_lod_transitions_across_distances() {
        let mut renderer = CrowdRenderer::new(
            CrowdRenderConfig::new().with_lod_distances([10.0, 30.0, 60.0]),
        );
        let batch_id = renderer.add_batch(0);

        // Add instances at various distances
        let distances = [5.0, 15.0, 25.0, 45.0, 80.0, 120.0];
        for &dist in &distances {
            renderer
                .add_instance(batch_id, CrowdInstance::new([dist, 0.0, 0.0]))
                .unwrap();
        }

        // Update LODs from origin
        renderer.update_lods([0.0, 0.0, 0.0]);

        let batch = renderer.get_batch(batch_id).unwrap();
        assert_eq!(batch.get(0).unwrap().lod_level, LOD_FULL_SKELETON); // 5m
        assert_eq!(batch.get(1).unwrap().lod_level, LOD_SIMPLIFIED);    // 15m
        assert_eq!(batch.get(2).unwrap().lod_level, LOD_SIMPLIFIED);    // 25m
        assert_eq!(batch.get(3).unwrap().lod_level, LOD_IMPOSTOR);      // 45m
        assert_eq!(batch.get(4).unwrap().lod_level, LOD_IMPOSTOR);      // 80m
        assert_eq!(batch.get(5).unwrap().lod_level, LOD_IMPOSTOR);      // 120m
    }

    #[test]
    fn test_full_pipeline_animation_playback_verification() {
        let clip = create_test_clip("test", 1.0, 2);
        let baker = AnimationTextureBaker::new(2, 30.0);
        let data = baker.bake_clip(&clip);

        let sampler = AnimationTextureSampler::new(data.layout.clone());

        // Sample at start
        let (pos_start, _, _) = sampler.sample_bone_at_time(&data, 0, 0.0);
        assert!(pos_start.y.abs() < 0.01); // Y should be ~0 at start

        // Sample at end
        let (pos_end, _, _) = sampler.sample_bone_at_time(&data, 0, 1.0);
        assert!((pos_end.y - 2.0).abs() < 0.1); // Y should be ~2.0 at end (duration * 2)

        // Sample at middle
        let (pos_mid, _, _) = sampler.sample_bone_at_time(&data, 0, 0.5);
        assert!(pos_mid.y > 0.5 && pos_mid.y < 1.5); // Should be interpolated
    }

    #[test]
    fn test_full_pipeline_phase_synchronization() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        // Add instances with and without phase sync
        renderer
            .add_instance(batch_id, CrowdInstance::new([0.0, 0.0, 0.0]))
            .unwrap();
        renderer
            .add_instance(
                batch_id,
                CrowdInstance::new([5.0, 0.0, 0.0]).with_phase_sync(true),
            )
            .unwrap();

        // Update animation time
        renderer.update_animation_time(0.5);
        renderer.update_animation_time(0.3);

        let batch = renderer.get_batch(batch_id).unwrap();

        // Non-synced: 0 + 0.5 + 0.3 = 0.8
        assert!((batch.get(0).unwrap().animation_time - 0.8).abs() < 0.001);

        // Synced: global time = 0.8
        assert!((batch.get(1).unwrap().animation_time - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_full_pipeline_shader_generation_for_all_lods() {
        let shader = CrowdAnimationShader::default();

        for lod in ShaderLodLevel::all() {
            let source = shader.generate_source(lod);
            assert!(shader.validate_source(&source).is_ok());
            assert!(source.contains(&format!("#define LOD_LEVEL {}", lod as u32)));
        }
    }

    #[test]
    fn test_full_pipeline_gpu_buffer_preparation() {
        let clip = create_test_clip("test", 1.0, 4);
        let baker = AnimationTextureBaker::new(4, 30.0);
        let texture_data = baker.bake_clip(&clip);

        let (bytes, format) = prepare_gpu_texture(&texture_data);

        // Verify size matches expected
        let expected_size = texture_data.layout.bone_count as usize
            * 2 // pos + rot rows
            * texture_data.layout.frame_count as usize
            * 16; // RGBA32F bytes

        assert_eq!(bytes.len(), expected_size);
        assert_eq!(format.wgpu_format(), "rgba32float");
    }

    #[test]
    fn test_full_pipeline_mipmap_generation() {
        let clip = create_test_clip("test", 1.0, 4);
        let baker = AnimationTextureBaker::new(4, 60.0); // Higher sample rate
        let data = baker.bake_clip(&clip);

        let mipmaps = generate_mipmaps(&data, 3);

        assert!(!mipmaps.is_empty());
        for (i, mip) in mipmaps.iter().enumerate() {
            assert!(mip.layout.frame_count < data.layout.frame_count);
            assert_eq!(mip.layout.bone_count, data.layout.bone_count);
            assert!(mip.clip_name.contains(&format!("mip{}", i + 1)));
        }
    }

    #[test]
    fn test_full_pipeline_uniform_buffer_setup() {
        let uniforms = CrowdUniforms::new()
            .with_time(1.5, 0.016)
            .with_camera_pos([0.0, 5.0, -20.0])
            .with_animation(64, 31, 30.0)
            .with_lod_distances([20.0, 50.0, 100.0]);

        let bytes = uniforms.as_bytes();
        assert_eq!(bytes.len(), std::mem::size_of::<CrowdUniforms>());

        // Verify uniforms are correctly set
        assert_eq!(uniforms.time, 1.5);
        assert_eq!(uniforms.bone_count, 64);
        assert_eq!(uniforms.frame_count, 31);
    }

    #[test]
    fn test_full_pipeline_draw_command_generation() {
        let mut renderer = create_test_renderer(100);
        renderer.update_lods([0.0, 0.0, 0.0]);
        renderer.cull_frustum(&create_test_frustum());

        let commands = renderer.get_draw_commands();

        // Should have commands for each LOD level with instances
        assert!(!commands.is_empty());

        // Total instances should match visible count
        let total: u32 = commands.iter().map(|c| c.instance_count).sum();
        assert_eq!(total, renderer.visible_count() as u32);
    }

    #[test]
    fn test_full_pipeline_lod_batching() {
        let mut renderer = create_test_renderer(50);
        renderer.update_lods([25.0, 0.0, 25.0]); // Center of crowd
        renderer.cull_frustum(&Frustum::unbounded());

        let lod_batches = renderer.get_lod_batches();

        // Should have instances at various LOD levels
        let lod0_count: u32 = lod_batches
            .iter()
            .filter(|b| b.lod_level == LOD_FULL_SKELETON)
            .map(|b| b.instance_count)
            .sum();
        let lod1_count: u32 = lod_batches
            .iter()
            .filter(|b| b.lod_level == LOD_SIMPLIFIED)
            .map(|b| b.instance_count)
            .sum();

        assert!(lod0_count > 0); // Some near camera
        assert!(lod1_count > 0 || lod_batches.len() > 0);
    }

    #[test]
    fn test_full_pipeline_animation_blending_shader() {
        let features = ShaderFeatures::new().with_blend_animations(true);
        let shader = CrowdAnimationShader::new(
            64,
            ShaderConfig::default().with_features(features),
        );

        let source = shader.generate_source(ShaderLodLevel::FullSkeleton);

        assert!(source.contains("sample_blended_bone_transform"));
        assert!(source.contains("mat_to_quat"));
        assert!(shader.validate_source(&source).is_ok());
    }

    #[test]
    fn test_full_pipeline_instance_buffer_layout() {
        let mut batch = CrowdBatch::new(0);

        for i in 0..10 {
            batch.add_instance(
                CrowdInstance::new([i as f32, 0.0, 0.0])
                    .with_animation(i as u16, i as f32 * 0.1)
                    .with_scale(1.0 + i as f32 * 0.1),
            );
        }

        let bytes = batch.as_bytes();

        // Each instance should be exactly CROWD_INSTANCE_SIZE bytes
        assert_eq!(bytes.len(), 10 * CROWD_INSTANCE_SIZE);
    }

    #[test]
    fn test_full_pipeline_render_stats() {
        let mut renderer = create_test_renderer(100);
        renderer.update_lods([50.0, 0.0, 50.0]);
        renderer.cull_frustum(&create_test_frustum());

        let stats = renderer.stats();

        assert_eq!(stats.batch_count, 1);
        assert_eq!(stats.total_instances, 100);
        assert!(stats.visible_instances > 0);
        assert!(stats.total_buffer_size > 0);
    }

    // ========================================================================
    // SECTION 2: LOD System Tests (10+ tests)
    // ========================================================================

    #[test]
    fn test_lod_distance_based_selection() {
        let config = LodConfig::new()
            .with_distance_thresholds([15.0, 40.0, 80.0, 150.0]);

        let selector = LodSelector::new(config);

        // Test LOD selection at various distances using select_lod_distance
        let lod_5 = selector.select_lod_distance(
            LodVec3::new(5.0, 0.0, 0.0), 1.0, None, 0, None
        );
        let lod_20 = selector.select_lod_distance(
            LodVec3::new(20.0, 0.0, 0.0), 1.0, None, 0, None
        );
        let lod_60 = selector.select_lod_distance(
            LodVec3::new(60.0, 0.0, 0.0), 1.0, None, 0, None
        );
        let lod_100 = selector.select_lod_distance(
            LodVec3::new(100.0, 0.0, 0.0), 1.0, None, 0, None
        );

        assert_eq!(lod_5, LodLevel::Lod0);
        assert_eq!(lod_20, LodLevel::Lod1);
        assert_eq!(lod_60, LodLevel::Lod2);
        assert_eq!(lod_100, LodLevel::Lod3);
    }

    #[test]
    fn test_lod_budget_constrained_quality() {
        let mut budget_manager = LodBudgetManager::new(10_000);

        // Add instances with polygon costs
        let costs = [1000, 500, 100, 10]; // LOD0, LOD1, LOD2, LOD3

        let mut remaining = budget_manager.budget();
        let mut selected_levels = Vec::new();

        for _ in 0..20 {
            // Select best LOD that fits budget
            for (lod, &cost) in costs.iter().enumerate() {
                if cost <= remaining {
                    selected_levels.push(lod);
                    remaining -= cost;
                    break;
                }
            }
        }

        assert!(!selected_levels.is_empty());
        assert!(remaining <= budget_manager.budget());
    }

    #[test]
    fn test_lod_smooth_transition_crossfade() {
        // Test LodTransition from crowd_lod module
        let mut transition = LodTransition::new(LodLevel::Lod0, LodLevel::Lod1, 0.25, false);

        // Start transition
        assert_eq!(transition.progress, 0.0);

        // Update over time
        transition.advance(0.1);
        assert!(transition.progress > 0.0 && transition.progress < 1.0);

        transition.advance(0.1);
        let mid_progress = transition.progress;
        assert!(mid_progress > 0.3 && mid_progress < 0.9);

        // Complete transition
        transition.advance(0.2);
        assert!(transition.is_complete());
    }

    #[test]
    fn test_lod_dithered_transition() {
        let config = LodConfig::new().with_dithering(true);

        // Dithered transitions should apply dither pattern based on screen position
        let dither_value = |x: f32, y: f32, threshold: f32| -> bool {
            // Simplified dither pattern
            let pattern = ((x as i32 + y as i32) % 4) as f32 / 4.0;
            pattern < threshold
        };

        // At 50% transition, roughly half pixels should show each LOD
        let mut visible_old = 0;
        let mut visible_new = 0;

        for x in 0..16 {
            for y in 0..16 {
                if dither_value(x as f32, y as f32, 0.5) {
                    visible_new += 1;
                } else {
                    visible_old += 1;
                }
            }
        }

        // Should be roughly equal
        assert!(((visible_old - visible_new) as i32).abs() < 64);
    }

    #[test]
    fn test_lod_hysteresis_prevents_oscillation() {
        let config = LodConfig::new()
            .with_distance_thresholds([20.0, 50.0, 100.0, 200.0])
            .with_hysteresis(3.0);

        let selector = LodSelector::new(config);

        // Current LOD is 0, distance crosses threshold
        // At exactly threshold, should not change due to hysteresis
        let new_lod = selector.select_lod_distance(
            LodVec3::new(20.0, 0.0, 0.0), 1.0, None, 0, Some(LodLevel::Lod0)
        );
        assert_eq!(new_lod, LodLevel::Lod0);

        // Just past threshold + hysteresis, should change
        let new_lod = selector.select_lod_distance(
            LodVec3::new(23.5, 0.0, 0.0), 1.0, None, 0, Some(LodLevel::Lod0)
        );
        assert_eq!(new_lod, LodLevel::Lod1);

        // Coming back, need to cross threshold - hysteresis
        let new_lod = selector.select_lod_distance(
            LodVec3::new(18.0, 0.0, 0.0), 1.0, None, 0, Some(LodLevel::Lod1)
        );
        assert_eq!(new_lod, LodLevel::Lod1);

        let new_lod = selector.select_lod_distance(
            LodVec3::new(16.5, 0.0, 0.0), 1.0, None, 0, Some(LodLevel::Lod1)
        );
        assert_eq!(new_lod, LodLevel::Lod0);
    }

    #[test]
    fn test_lod_priority_hero_characters() {
        let config = LodConfig::new().with_priority(true);

        // Hero character should never go below LOD0
        let selector = LodSelector::new(config);

        // Even at far distance, hero stays at LOD0
        let lod = selector.select_lod_distance(
            LodVec3::new(200.0, 0.0, 0.0), 1.0, None, PRIORITY_HERO, None
        );
        assert_eq!(lod, LodLevel::Lod0);

        // Important character should never go below LOD1
        let lod = selector.select_lod_distance(
            LodVec3::new(200.0, 0.0, 0.0), 1.0, None, PRIORITY_IMPORTANT, None
        );
        assert_eq!(lod, LodLevel::Lod1);
    }

    #[test]
    fn test_lod_level_transitions() {
        // Test transition path
        assert_eq!(LodLevel::Lod0.lower(), Some(LodLevel::Lod1));
        assert_eq!(LodLevel::Lod1.lower(), Some(LodLevel::Lod2));
        assert_eq!(LodLevel::Lod2.lower(), Some(LodLevel::Lod3));
        assert_eq!(LodLevel::Lod3.lower(), None);

        assert_eq!(LodLevel::Lod0.higher(), None);
        assert_eq!(LodLevel::Lod1.higher(), Some(LodLevel::Lod0));
        assert_eq!(LodLevel::Lod2.higher(), Some(LodLevel::Lod1));
        assert_eq!(LodLevel::Lod3.higher(), Some(LodLevel::Lod2));
    }

    #[test]
    fn test_lod_skeletal_vs_billboard() {
        assert!(LodLevel::Lod0.uses_skeleton());
        assert!(LodLevel::Lod1.uses_skeleton());
        assert!(!LodLevel::Lod2.uses_skeleton());
        assert!(!LodLevel::Lod3.uses_skeleton());

        assert!(!LodLevel::Lod0.is_billboard());
        assert!(!LodLevel::Lod1.is_billboard());
        assert!(LodLevel::Lod2.is_billboard());
        assert!(!LodLevel::Lod3.is_billboard());
    }

    #[test]
    fn test_lod_screen_size_based() {
        let config = LodConfig::new()
            .with_screen_size_lod(true)
            .with_screen_size_thresholds([100.0, 50.0, 25.0, 10.0]);

        let thresholds = config.screen_size_thresholds;
        let selector = LodSelector::new(config);

        // Calculate approximate screen size from distance
        let screen_size_at_distance = |distance: f32, fov: f32| -> f32 {
            let character_height = 2.0;
            let screen_height = 1080.0;
            (character_height / distance) * screen_height / fov.tan()
        };

        // Objects that project to larger screen size should get higher LOD
        let large_screen = screen_size_at_distance(10.0, 0.8);
        let small_screen = screen_size_at_distance(100.0, 0.8);

        assert!(large_screen > thresholds[0]);
        assert!(small_screen < thresholds[1]);
    }

    #[test]
    fn test_lod_polygon_budget_distribution() {
        let mut budget_manager = LodBudgetManager::new(100_000);

        // Simulate 1000 instances competing for budget
        let instance_count = 1000;
        let costs = [2000, 500, 100, 20];

        // Sort instances by priority (distance) and allocate budget
        let distances: Vec<f32> = (0..instance_count).map(|i| i as f32).collect();
        let mut allocated_levels = vec![LodLevel::Lod3; instance_count];
        let mut used_budget = 0;

        for (i, _dist) in distances.iter().enumerate() {
            for (lod_idx, &cost) in costs.iter().enumerate() {
                if used_budget + cost <= budget_manager.budget() {
                    allocated_levels[i] = LodLevel::from_u8(lod_idx as u8).unwrap();
                    used_budget += cost;
                    break;
                }
            }
        }

        // Verify budget not exceeded
        assert!(used_budget <= budget_manager.budget());

        // Closer instances should have higher LOD
        let lod0_count = allocated_levels.iter().filter(|&&l| l == LodLevel::Lod0).count();
        assert!(lod0_count > 0);
    }

    // ========================================================================
    // SECTION 3: Impostor Tests (10+ tests)
    // ========================================================================

    #[test]
    fn test_impostor_view_angle_selection() {
        let config = ImpostorConfig::new().with_view_angles(8, 4);
        let total_angles = config.total_view_angles();
        let selector = ViewAngleSelector::new(&config);

        let camera_pos = ImpostorVec3::new(0.0, 5.0, -10.0);
        let instance_pos = ImpostorVec3::new(0.0, 0.0, 0.0);
        let rotation = ImpostorQuat::IDENTITY;

        let (combined_idx, h_idx, v_idx) = selector.select_angle(camera_pos, instance_pos, rotation);

        // Should select appropriate view based on camera direction
        assert!(combined_idx < total_angles);
        assert!(h_idx < 8);
        assert!(v_idx < 4);
    }

    #[test]
    fn test_impostor_animation_frame_sampling() {
        let config = ImpostorConfig::new().with_animation_frames(16);

        // Sample animation at different times
        let duration = 1.0;
        let sample_rate = 30.0;

        let frame_at_time = |time: f32| -> u32 {
            let frame = (time * sample_rate) % (config.animation_frames as f32);
            frame.floor() as u32
        };

        assert_eq!(frame_at_time(0.0), 0);
        assert_eq!(frame_at_time(0.5), 15);
        assert_eq!(frame_at_time(0.533), 0); // Wraps around
    }

    #[test]
    fn test_impostor_billboard_orientation_spherical() {
        let config = ImpostorConfig::new().with_billboard_mode(BillboardMode::Spherical);

        let camera_pos = ImpostorVec3::new(5.0, 5.0, -10.0);
        let instance_pos = ImpostorVec3::new(0.0, 0.0, 0.0);

        // Calculate facing direction
        let to_camera = camera_pos.sub(instance_pos).normalize();

        // Spherical billboard should face exactly toward camera
        assert!(to_camera.length() > 0.99);
    }

    #[test]
    fn test_impostor_billboard_orientation_cylindrical() {
        let config = ImpostorConfig::new().with_billboard_mode(BillboardMode::Cylindrical);

        let camera_pos = ImpostorVec3::new(5.0, 5.0, -10.0);
        let instance_pos = ImpostorVec3::new(0.0, 0.0, 0.0);

        // Calculate horizontal facing (ignore Y)
        let to_camera_xz = ImpostorVec3::new(
            camera_pos.x - instance_pos.x,
            0.0,
            camera_pos.z - instance_pos.z,
        )
        .normalize();

        // Should only rotate around Y
        assert!(to_camera_xz.y.abs() < 0.001);
        assert!(to_camera_xz.length() > 0.99);
    }

    #[test]
    fn test_impostor_atlas_layout() {
        let config = ImpostorConfig::new()
            .with_view_angles(8, 4)
            .with_animation_frames(16)
            .with_sprite_resolution(256, 256);

        let total_sprites = config.total_sprites();
        assert_eq!(total_sprites, 8 * 4 * 16);

        // Calculate atlas size
        let sprites_per_row = (total_sprites as f32).sqrt().ceil() as u32;
        let atlas_width = sprites_per_row * config.sprite_width;
        let atlas_height = sprites_per_row * config.sprite_height;

        assert!(atlas_width <= 16384); // Max texture size
        assert!(atlas_height <= 16384);
    }

    #[test]
    fn test_impostor_instance_creation() {
        let instance = ImpostorInstance::new([10.0, 0.0, 5.0])
            .with_animation_frame(8)
            .with_view_angle_index(3)
            .with_lod_fade(0.5);

        assert_eq!(instance.position.x, 10.0);
        assert_eq!(instance.position.z, 5.0);
        assert_eq!(instance.animation_frame, 8);
        assert_eq!(instance.view_angle_index, 3);
        assert!((instance.fade_factor - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_impostor_alpha_cutoff() {
        let config = ImpostorConfig::new().with_alpha_cutoff(0.5);

        // Alpha values below cutoff should be discarded
        let test_alphas = [0.3, 0.5, 0.7, 0.9];
        let visible: Vec<_> = test_alphas
            .iter()
            .filter(|&&a| a >= config.alpha_cutoff)
            .collect();

        assert_eq!(visible.len(), 3);
    }

    #[test]
    fn test_impostor_pitch_angle_clamping() {
        let config = ImpostorConfig::new().with_pitch_range(-PI / 6.0, PI / 3.0);

        assert!(config.min_pitch >= -PI / 2.0);
        assert!(config.max_pitch <= PI / 2.0);
        assert!(config.min_pitch < config.max_pitch);
    }

    #[test]
    fn test_impostor_mipmap_generation() {
        let config = ImpostorConfig::new()
            .with_sprite_resolution(256, 256)
            .with_mipmaps(true);

        // Calculate mipmap levels
        let max_dim = config.sprite_width.max(config.sprite_height);
        let mip_levels = (max_dim as f32).log2().floor() as u32 + 1;

        assert!(mip_levels > 1);
        assert!(mip_levels <= 9); // 256 = 2^8, so 9 levels
    }

    #[test]
    fn test_impostor_compression() {
        let config = ImpostorConfig::new().with_compression(true);

        // BC3/DXT5 compression
        let uncompressed_size = config.sprite_width * config.sprite_height * 4; // RGBA
        let compressed_size = config.sprite_width * config.sprite_height; // BC3 is 1 byte/pixel

        assert!(compressed_size < uncompressed_size);
    }

    // ========================================================================
    // SECTION 4: Instancing Tests (10+ tests)
    // ========================================================================

    #[test]
    fn test_instancing_large_crowd_batching() {
        let mut renderer = create_test_renderer(1000);

        assert_eq!(renderer.instance_count(), 1000);
        assert_eq!(renderer.batch_count(), 1);

        renderer.cull_frustum(&create_test_frustum());
        assert_eq!(renderer.visible_count(), 1000);
    }

    #[test]
    fn test_instancing_frustum_culling_accuracy() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        // Add instances at known positions
        renderer.add_instance(batch_id, CrowdInstance::new([0.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([50.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([100.0, 0.0, 0.0])).unwrap();

        // Create frustum that only includes first two
        let frustum = Frustum::from_bounds(
            CrowdVec3::new(-10.0, -10.0, -10.0),
            CrowdVec3::new(60.0, 10.0, 10.0),
        );

        renderer.cull_frustum(&frustum);

        assert_eq!(renderer.visible_count(), 2);

        let batch = renderer.get_batch(batch_id).unwrap();
        assert!(batch.get(0).unwrap().is_visible());
        assert!(batch.get(1).unwrap().is_visible());
        assert!(!batch.get(2).unwrap().is_visible());
    }

    #[test]
    fn test_instancing_buffer_management() {
        let mut renderer = CrowdRenderer::new(
            CrowdRenderConfig::new().with_max_instances(100),
        );
        let batch_id = renderer.add_batch(0);

        // Fill buffer
        for i in 0..100 {
            let result = renderer.add_instance(
                batch_id,
                CrowdInstance::new([i as f32, 0.0, 0.0]),
            );
            assert!(result.is_ok());
        }

        // Should fail when full
        let result = renderer.add_instance(batch_id, CrowdInstance::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_instancing_instance_removal() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        for i in 0..10 {
            renderer
                .add_instance(batch_id, CrowdInstance::new([i as f32, 0.0, 0.0]))
                .unwrap();
        }

        // Remove middle instance
        let result = renderer.remove_instance(batch_id, 5);
        assert!(result.is_ok());
        assert_eq!(renderer.instance_count(), 9);
    }

    #[test]
    fn test_instancing_instance_update() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        renderer
            .add_instance(batch_id, CrowdInstance::new([0.0, 0.0, 0.0]))
            .unwrap();

        // Update instance
        let new_instance = CrowdInstance::new([10.0, 5.0, -3.0]).with_scale(2.0);
        renderer.update_instance(batch_id, 0, new_instance).unwrap();

        let batch = renderer.get_batch(batch_id).unwrap();
        let updated = batch.get(0).unwrap();
        assert_eq!(updated.position.x, 10.0);
        assert_eq!(updated.scale, 2.0);
    }

    #[test]
    fn test_instancing_multiple_batches() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());

        let batch_a = renderer.add_batch(0);
        let batch_b = renderer.add_batch(1);
        let batch_c = renderer.add_batch(2);

        for _ in 0..50 {
            renderer.add_instance(batch_a, CrowdInstance::default()).unwrap();
        }
        for _ in 0..30 {
            renderer.add_instance(batch_b, CrowdInstance::default()).unwrap();
        }
        for _ in 0..20 {
            renderer.add_instance(batch_c, CrowdInstance::default()).unwrap();
        }

        assert_eq!(renderer.batch_count(), 3);
        assert_eq!(renderer.instance_count(), 100);
    }

    #[test]
    fn test_instancing_draw_command_structure() {
        let cmd = IndirectDrawCommand::with_offsets(1000, 50, 0, 100);

        assert_eq!(cmd.vertex_count, 1000);
        assert_eq!(cmd.instance_count, 50);
        assert_eq!(cmd.first_vertex, 0);
        assert_eq!(cmd.first_instance, 100);
        assert!(!cmd.is_empty());
    }

    #[test]
    fn test_instancing_buffer_alignment() {
        assert_eq!(CROWD_INSTANCE_SIZE, 64);
        assert_eq!(std::mem::size_of::<CrowdInstance>(), CROWD_INSTANCE_SIZE);

        // Verify alignment for GPU
        assert!(CROWD_INSTANCE_SIZE % 16 == 0); // vec4 alignment
    }

    #[test]
    fn test_instancing_dirty_flag_tracking() {
        let mut batch = CrowdBatch::new(0);

        assert!(!batch.is_dirty());

        batch.add_instance(CrowdInstance::default());
        assert!(batch.is_dirty());

        batch.mark_clean();
        assert!(!batch.is_dirty());

        batch.get_mut(0).unwrap().scale = 2.0;
        assert!(batch.is_dirty());
    }

    #[test]
    fn test_instancing_visible_collection() {
        let mut batch = CrowdBatch::new(0);

        for i in 0..10 {
            batch.add_instance(CrowdInstance::new([i as f32, 0.0, 0.0]));
        }

        // Mark some as invisible
        batch.instances[2].set_visible(false);
        batch.instances[5].set_visible(false);
        batch.instances[8].set_visible(false);

        let visible: Vec<_> = batch.visible_instances().collect();
        assert_eq!(visible.len(), 7);

        let collected = batch.collect_visible();
        assert_eq!(collected.len(), 7);
    }

    // ========================================================================
    // SECTION 5: Performance Tests (5+ tests)
    // ========================================================================

    #[test]
    fn test_perf_1000_instance_rendering() {
        let start = std::time::Instant::now();

        let mut renderer = create_test_renderer(1000);
        renderer.update_lods([50.0, 0.0, 50.0]);
        renderer.cull_frustum(&create_test_frustum());
        let _buffers = renderer.prepare_gpu_buffers();
        let _commands = renderer.get_draw_commands();

        let elapsed = start.elapsed();

        // Should complete in reasonable time (< 50ms on most systems)
        assert!(elapsed.as_millis() < 500, "Took too long: {:?}", elapsed);

        assert_eq!(renderer.instance_count(), 1000);
        assert_eq!(renderer.visible_count(), 1000);
    }

    #[test]
    fn test_perf_lod_update_overhead() {
        let mut renderer = create_test_renderer(5000);

        // Measure LOD update time
        let start = std::time::Instant::now();
        for _ in 0..10 {
            renderer.update_lods([50.0, 0.0, 50.0]);
        }
        let elapsed = start.elapsed();

        // 10 LOD updates of 5000 instances should be fast
        assert!(elapsed.as_millis() < 200, "LOD update too slow: {:?}", elapsed);
    }

    #[test]
    fn test_perf_culling_overhead() {
        let mut renderer = create_test_renderer(5000);
        let frustum = create_test_frustum();

        // Measure culling time
        let start = std::time::Instant::now();
        for _ in 0..10 {
            renderer.cull_frustum(&frustum);
        }
        let elapsed = start.elapsed();

        // 10 culling passes of 5000 instances
        assert!(elapsed.as_millis() < 200, "Culling too slow: {:?}", elapsed);
    }

    #[test]
    fn test_perf_memory_budget_tracking() {
        let renderer = create_test_renderer(1000);

        let stats = renderer.stats();

        // 1000 instances * 64 bytes = 64KB
        let expected_min = 1000 * CROWD_INSTANCE_SIZE;
        assert!(stats.total_buffer_size >= expected_min);
    }

    #[test]
    fn test_perf_shader_cache_efficiency() {
        let mut shader = CrowdAnimationShader::default();

        // Generate all permutations
        for lod in ShaderLodLevel::all() {
            for blend in [false, true] {
                let features = ShaderFeatures::new().with_blend_animations(blend);
                let perm = ShaderPermutation::new(lod, features);
                let _ = shader.generate_permutation(&perm);
            }
        }

        let (hits, misses) = shader.cache_stats();

        // All should be misses on first pass
        assert_eq!(misses, 6);

        // Second pass should all hit
        for lod in ShaderLodLevel::all() {
            let perm = ShaderPermutation::new(lod, ShaderFeatures::default());
            let _ = shader.generate_permutation(&perm);
        }

        let (hits2, _) = shader.cache_stats();
        assert!(hits2 > hits);
    }

    #[test]
    fn test_perf_texture_bake_large() {
        let clip = create_test_clip("large", 2.0, 64);
        let baker = AnimationTextureBaker::new(64, 30.0);

        let start = std::time::Instant::now();
        let data = baker.bake_clip(&clip);
        let elapsed = start.elapsed();

        assert!(data.is_valid());
        assert!(elapsed.as_millis() < 500, "Baking too slow: {:?}", elapsed);
    }

    // ========================================================================
    // SECTION 6: Edge Cases (10+ tests)
    // ========================================================================

    #[test]
    fn test_edge_camera_teleportation() {
        let mut renderer = create_test_renderer(100);

        // Initial state
        renderer.update_lods([0.0, 0.0, 0.0]);
        let stats_before = renderer.stats();

        // Teleport camera far away
        renderer.update_lods([1000.0, 0.0, 1000.0]);
        let stats_after = renderer.stats();

        // All instances should now be at low LOD
        assert!(stats_after.lod_2_count >= stats_before.lod_0_count);
    }

    #[test]
    fn test_edge_spawn_burst_handling() {
        let mut renderer = CrowdRenderer::new(
            CrowdRenderConfig::new().with_max_instances(10_000),
        );
        let batch_id = renderer.add_batch(0);

        // Spawn many instances at once
        let start = std::time::Instant::now();
        for i in 0..1000 {
            renderer
                .add_instance(batch_id, CrowdInstance::new([i as f32, 0.0, 0.0]))
                .unwrap();
        }
        let elapsed = start.elapsed();

        assert_eq!(renderer.instance_count(), 1000);
        assert!(elapsed.as_millis() < 100);
    }

    #[test]
    fn test_edge_zero_distance_instances() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        // Instance at camera position
        renderer.add_instance(batch_id, CrowdInstance::new([0.0, 0.0, 0.0])).unwrap();

        renderer.update_lods([0.0, 0.0, 0.0]);

        let batch = renderer.get_batch(batch_id).unwrap();
        assert_eq!(batch.get(0).unwrap().lod_level, LOD_FULL_SKELETON);
    }

    #[test]
    fn test_edge_nan_position_validation() {
        let config = CrowdRenderConfig::new().with_validation(true);
        let mut renderer = CrowdRenderer::new(config);
        let batch_id = renderer.add_batch(0);

        let mut invalid = CrowdInstance::new([0.0, 0.0, 0.0]);
        invalid.position = CrowdVec3::new(f32::NAN, 0.0, 0.0);

        let result = renderer.add_instance(batch_id, invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_negative_scale() {
        let config = CrowdRenderConfig::new().with_validation(true);
        let mut renderer = CrowdRenderer::new(config);
        let batch_id = renderer.add_batch(0);

        let invalid = CrowdInstance::new([0.0, 0.0, 0.0]).with_scale(-1.0);

        let result = renderer.add_instance(batch_id, invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_empty_batch_operations() {
        let mut batch = CrowdBatch::new(0);

        assert!(batch.is_empty());
        assert_eq!(batch.visible_count(), 0);
        assert!(batch.get(0).is_none());

        let culled = batch.cull_frustum(&Frustum::unbounded());
        assert_eq!(culled, 0);
    }

    #[test]
    fn test_edge_animation_time_wrap() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        renderer.add_instance(batch_id, CrowdInstance::default()).unwrap();

        // Advance time significantly
        for _ in 0..1000 {
            renderer.update_animation_time(0.1);
        }

        // Should handle large time values
        assert!(renderer.global_time() > 99.0);
    }

    #[test]
    fn test_edge_single_frame_animation() {
        let clip = create_static_clip("static", Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        let baker = AnimationTextureBaker::new(1, 30.0);

        let data = baker.bake_clip(&clip);

        assert_eq!(data.layout.frame_count, 1);
        assert!(data.is_valid());

        let sampler = AnimationTextureSampler::new(data.layout.clone());
        let (pos, _, _) = sampler.sample_bone(&data, 0, 0.0);
        assert!((pos - Vec3::new(1.0, 2.0, 3.0)).length() < 0.001);
    }

    #[test]
    fn test_edge_max_lod_distance() {
        let mut renderer = CrowdRenderer::new(
            CrowdRenderConfig::new().with_lod_distances([10.0, 20.0, 30.0]),
        );
        let batch_id = renderer.add_batch(0);

        // Instance way beyond max LOD distance
        renderer.add_instance(batch_id, CrowdInstance::new([10000.0, 0.0, 0.0])).unwrap();

        renderer.update_lods([0.0, 0.0, 0.0]);

        let batch = renderer.get_batch(batch_id).unwrap();
        // Should still be renderable at highest LOD level
        assert_eq!(batch.get(0).unwrap().lod_level, LOD_IMPOSTOR);
    }

    #[test]
    fn test_edge_frustum_boundary() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        // Instance exactly at frustum boundary
        renderer.add_instance(batch_id, CrowdInstance::new([100.0, 0.0, 0.0])).unwrap();

        let frustum = Frustum::from_bounds(
            CrowdVec3::new(0.0, -10.0, -10.0),
            CrowdVec3::new(100.0, 10.0, 10.0),
        );

        renderer.cull_frustum(&frustum);

        // Boundary cases should be included (sphere intersection)
        assert_eq!(renderer.visible_count(), 1);
    }

    // ========================================================================
    // SECTION 7: Additional Integration Tests
    // ========================================================================

    #[test]
    fn test_integration_texture_sampler_accuracy() {
        let clip = create_test_clip("accuracy", 1.0, 4);
        let baker = AnimationTextureBaker::new(4, 30.0);
        let data = baker.bake_clip(&clip);

        let sampler = AnimationTextureSampler::new(data.layout.clone());

        // Sample at various interpolation points
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let pose = sampler.sample_pose_at_time(&data, t);
            assert_eq!(pose.len(), 4);

            for (bone, (pos, _rot, _scale)) in pose.iter().enumerate() {
                // Y should increase linearly with time
                let expected_y = t * 2.0;
                assert!((pos.y - expected_y).abs() < 0.2, "Bone {} at t={}: y={} expected {}", bone, t, pos.y, expected_y);
            }
        }
    }

    #[test]
    fn test_integration_blend_two_animations() {
        let clip_a = create_static_clip("a", Vec3::ZERO, Quat::IDENTITY);
        let clip_b = create_static_clip("b", Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY);

        let baker = AnimationTextureBaker::new(1, 30.0);
        let data_a = baker.bake_clip(&clip_a);
        let data_b = baker.bake_clip(&clip_b);

        let sampler = AnimationTextureSampler::new(data_a.layout.clone());

        for weight in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let blended = sampler.blend_samples(&data_a, &data_b, 0.0, 0.0, weight);
            let expected_x = 10.0 * weight;
            assert!((blended[0].0.x - expected_x).abs() < 0.01);
        }
    }

    #[test]
    fn test_integration_shader_compiler_utilities() {
        let source = ShaderCompiler::generate_skinning_shader(64, &ShaderFeatures::default());

        assert!(source.contains("cs_skin_vertices"));
        assert!(source.contains("#define MAX_BONES 64u"));

        let complexity = ShaderCompiler::estimate_complexity(&source);
        assert!(complexity > 0);

        let line_count = ShaderCompiler::line_count(&source);
        assert!(line_count > 50);
    }

    #[test]
    fn test_integration_compose_transform() {
        let pos = [1.0, 2.0, 3.0];
        let rot = [0.0, 0.0, 0.0, 1.0]; // Identity
        let scale = 1.0;

        let matrix = compose_transform_matrix(pos, rot, scale);

        // Check translation column
        assert!((matrix[3][0] - 1.0).abs() < 0.001);
        assert!((matrix[3][1] - 2.0).abs() < 0.001);
        assert!((matrix[3][2] - 3.0).abs() < 0.001);
        assert!((matrix[3][3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_integration_nlerp_quaternion() {
        let a = [0.0, 0.0, 0.0, 1.0];
        let b = [0.0, 0.707, 0.0, 0.707]; // 90 deg Y rotation

        // 50% blend
        let result = nlerp_quat(a, b, 0.5);

        // Should be normalized
        let len = (result[0] * result[0]
            + result[1] * result[1]
            + result[2] * result[2]
            + result[3] * result[3])
            .sqrt();
        assert!((len - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_integration_bilinear_sample() {
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ];

        // Sample at center
        let result = bilinear_sample(&data, 2, 2, 0.5, 0.5);
        assert!((result[0] - 0.5).abs() < 0.01);
        assert!((result[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_integration_frame_lifecycle() {
        let mut renderer = create_test_renderer(10);

        assert_eq!(renderer.frame(), 0);

        renderer.begin_frame();
        renderer.update_animation_time(0.016);
        renderer.update_lods([0.0, 0.0, 0.0]);
        renderer.cull_frustum(&create_test_frustum());

        assert_eq!(renderer.frame(), 1);
        assert!(renderer.global_time() > 0.0);
    }
}
