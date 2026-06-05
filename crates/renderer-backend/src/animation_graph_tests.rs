//! Integration tests for the TRINITY animation graph system (T-AN-5.8).
//!
//! These tests verify cross-component interactions between:
//! - Animation graphs and their layers
//! - State machines and blend trees
//! - Parameter systems and sync groups
//! - Layer evaluation and blending

use crate::animation_graph::{
    AnimationGraph, AnimationLayer, AnimationNode, AnimationNodeType,
    AnimationParameter, ComparisonOp, LayerBlendMode, ParameterValue,
    StateTransition, TransitionCondition,
};
use crate::animation_layers::{LayerMask, LayerStack, MaskPreset};
use crate::pose::{Pose, PoseType};
use crate::skeleton::{Skeleton, Transform};
use crate::state_machine::{
    AnimationState, BlendCurve, CompareOp, ConditionValue, EvaluationContext,
    ParameterSet, StateContent, StateMachine, SyncMode, Transition,
    TransitionCondition as SMTransitionCondition,
};
use crate::sync_groups::{
    FootSyncState, MarkerType, SyncGroup, SyncGroupId, SyncGroupManager,
    SyncGroupMode, SyncMarker, SyncMember, SyncTrack,
};
use std::time::Instant;

// ============================================================================
// Test Helpers
// ============================================================================

fn create_test_skeleton(num_bones: usize) -> Skeleton {
    let mut skeleton = Skeleton::new();
    for i in 0..num_bones {
        let bone = crate::skeleton::Bone {
            name: format!("bone_{}", i),
            parent_index: if i == 0 { None } else { Some(i - 1) },
            local_transform: Transform::IDENTITY,
            inverse_bind_matrix: glam::Mat4::IDENTITY,
        };
        skeleton.add_bone(bone);
    }
    skeleton
}

fn create_test_pose(num_bones: usize) -> Pose {
    Pose::new(num_bones, PoseType::Current)
}

// ============================================================================
// GRAPH CONSTRUCTION TESTS (15+)
// ============================================================================

#[test]
fn test_graph_construction_empty() {
    let graph = AnimationGraph::new();
    assert!(graph.parameters.is_empty());
    assert!(graph.layers.is_empty());
    assert!(graph.nodes.is_empty());
}

#[test]
fn test_graph_construction_with_bone_count() {
    let graph = AnimationGraph::with_bone_count(60);
    assert_eq!(graph.bone_count, 60);
}

#[test]
fn test_graph_construction_single_layer() {
    let mut graph = AnimationGraph::new();
    let layer = AnimationLayer::new("base", LayerBlendMode::Override);
    graph.layers.push(layer);

    assert_eq!(graph.layers.len(), 1);
    assert_eq!(graph.layers[0].name, "base");
    assert_eq!(graph.layers[0].blend_mode, LayerBlendMode::Override);
}

#[test]
fn test_graph_construction_multiple_layers() {
    let mut graph = AnimationGraph::new();

    graph.layers.push(AnimationLayer::new("base", LayerBlendMode::Override));
    graph.layers.push(AnimationLayer::new("additive", LayerBlendMode::Additive));
    graph.layers.push(AnimationLayer::new("masked", LayerBlendMode::MaskedAdditive));

    assert_eq!(graph.layers.len(), 3);
    assert_eq!(graph.layers[0].blend_mode, LayerBlendMode::Override);
    assert_eq!(graph.layers[1].blend_mode, LayerBlendMode::Additive);
    assert_eq!(graph.layers[2].blend_mode, LayerBlendMode::MaskedAdditive);
}

#[test]
fn test_graph_construction_clip_node() {
    let mut graph = AnimationGraph::new();

    let node = AnimationNode::clip(0);
    graph.nodes.push(node);

    assert_eq!(graph.nodes.len(), 1);
    match &graph.nodes[0].node_type {
        AnimationNodeType::Clip { clip_index, .. } => {
            assert_eq!(*clip_index, 0);
        }
        _ => panic!("Expected Clip node"),
    }
}

#[test]
fn test_graph_construction_blend1d_node() {
    let mut graph = AnimationGraph::new();

    graph.add_parameter("speed", ParameterValue::Float(0.5));

    let node = AnimationNode::blend_1d(
        vec![0, 1, 2],
        vec![0.0, 0.5, 1.0],
        "speed",
    );
    graph.nodes.push(node);

    match &graph.nodes[0].node_type {
        AnimationNodeType::Blend1D { children, positions, parameter } => {
            assert_eq!(children.len(), 3);
            assert_eq!(positions.len(), 3);
            assert_eq!(parameter, "speed");
        }
        _ => panic!("Expected Blend1D node"),
    }
}

#[test]
fn test_graph_construction_blend2d_node() {
    let mut graph = AnimationGraph::new();

    graph.add_parameter("velocity_x", ParameterValue::Float(0.0));
    graph.add_parameter("velocity_z", ParameterValue::Float(0.0));

    let node = AnimationNode::blend_2d(
        vec![0, 1, 2, 3],
        vec![(0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        "velocity_x",
        "velocity_z",
    );
    graph.nodes.push(node);

    match &graph.nodes[0].node_type {
        AnimationNodeType::Blend2D { children, positions, parameter_x, parameter_y } => {
            assert_eq!(children.len(), 4);
            assert_eq!(positions.len(), 4);
            assert_eq!(parameter_x, "velocity_x");
            assert_eq!(parameter_y, "velocity_z");
        }
        _ => panic!("Expected Blend2D node"),
    }
}

#[test]
fn test_graph_construction_state_machine_node() {
    let mut graph = AnimationGraph::new();

    let transition = StateTransition::new(0, 1)
        .with_trigger("jump")
        .with_duration(0.2);

    let node = AnimationNode::state_machine(vec![0, 1], vec![transition]);
    graph.nodes.push(node);

    match &graph.nodes[0].node_type {
        AnimationNodeType::StateMachine { states, transitions, .. } => {
            assert_eq!(states.len(), 2);
            assert_eq!(transitions.len(), 1);
        }
        _ => panic!("Expected StateMachine node"),
    }
}

#[test]
fn test_graph_construction_additive_blend_node() {
    let mut graph = AnimationGraph::new();

    let node = AnimationNode::additive_blend(0, 1, 0.5);
    graph.nodes.push(node);

    match &graph.nodes[0].node_type {
        AnimationNodeType::AdditiveBlend { base, additive, weight } => {
            assert_eq!(*base, 0);
            assert_eq!(*additive, 1);
            assert!((*weight - 0.5).abs() < 0.001);
        }
        _ => panic!("Expected AdditiveBlend node"),
    }
}

#[test]
fn test_graph_construction_identity_node() {
    let mut graph = AnimationGraph::new();

    let node = AnimationNode::identity();
    graph.nodes.push(node);

    assert!(matches!(graph.nodes[0].node_type, AnimationNodeType::Identity));
}

#[test]
fn test_graph_construction_complex_hierarchy() {
    let mut graph = AnimationGraph::with_bone_count(60);

    // Parameters for all systems
    graph.add_parameter("speed", ParameterValue::Float(0.0));
    graph.add_parameter("direction", ParameterValue::Float(0.0));
    graph.add_parameter("is_airborne", ParameterValue::Bool(false));
    graph.add_parameter("aim_weight", ParameterValue::Float(0.0));

    // Base locomotion layer
    graph.layers.push(AnimationLayer::base("base"));

    // Additive aim layer
    graph.layers.push(AnimationLayer::additive("aim"));

    // Masked hit reaction layer
    let mask = vec![false; 30].into_iter().chain(vec![true; 30]).collect();
    graph.layers.push(AnimationLayer::masked_additive("hit_reaction", mask));

    assert_eq!(graph.layers.len(), 3);
    assert_eq!(graph.parameters.len(), 4);
}

#[test]
fn test_graph_construction_parameter_wiring() {
    let mut graph = AnimationGraph::new();

    // Add various parameter types
    graph.add_parameter("float_param", ParameterValue::Float(1.0));
    graph.add_parameter("int_param", ParameterValue::Int(42));
    graph.add_parameter("bool_param", ParameterValue::Bool(true));
    graph.add_parameter("vec3_param", ParameterValue::Vec3(glam::Vec3::new(1.0, 2.0, 3.0)));

    // Verify all parameters exist
    assert!(matches!(
        graph.get_parameter("float_param"),
        Some(ParameterValue::Float(v)) if (*v - 1.0).abs() < 0.001
    ));
    assert!(matches!(
        graph.get_parameter("int_param"),
        Some(ParameterValue::Int(42))
    ));
    assert!(matches!(
        graph.get_parameter("bool_param"),
        Some(ParameterValue::Bool(true))
    ));
}

#[test]
fn test_graph_construction_layer_weights() {
    let mut graph = AnimationGraph::new();

    graph.layers.push(AnimationLayer::new("base", LayerBlendMode::Override).with_weight(1.0));
    graph.layers.push(AnimationLayer::new("overlay", LayerBlendMode::Additive).with_weight(0.5));

    assert!((graph.layers[0].weight - 1.0).abs() < 0.001);
    assert!((graph.layers[1].weight - 0.5).abs() < 0.001);
}

#[test]
fn test_graph_construction_deep_layer_stack() {
    let mut graph = AnimationGraph::new();

    // Create 10 layers
    for i in 0..10 {
        let mode = match i % 3 {
            0 => LayerBlendMode::Override,
            1 => LayerBlendMode::Additive,
            _ => LayerBlendMode::MaskedAdditive,
        };
        graph.layers.push(AnimationLayer::new(&format!("layer_{}", i), mode));
    }

    assert_eq!(graph.layers.len(), 10);
}

#[test]
fn test_graph_construction_node_naming() {
    let mut graph = AnimationGraph::new();

    let node = AnimationNode::clip(0).with_name("idle_clip");
    graph.nodes.push(node);

    assert_eq!(graph.nodes[0].name, Some("idle_clip".to_string()));
}

#[test]
fn test_graph_construction_trigger_parameter() {
    let mut graph = AnimationGraph::new();

    let idx = graph.add_parameter("jump", ParameterValue::Trigger);

    assert_eq!(idx, 0);
    assert!(matches!(
        graph.get_parameter("jump"),
        Some(ParameterValue::Trigger)
    ));
}

// ============================================================================
// EVALUATION PIPELINE TESTS (15+)
// ============================================================================

#[test]
fn test_evaluation_dirty_flag_initial() {
    let graph = AnimationGraph::new();
    assert!(graph.dirty); // New graphs start dirty
}

#[test]
fn test_evaluation_dirty_flag_on_parameter_change() {
    let mut graph = AnimationGraph::new();
    graph.add_parameter("speed", ParameterValue::Float(0.0));
    graph.dirty = false; // Clear dirty

    graph.set_parameter("speed", ParameterValue::Float(1.0));
    assert!(graph.dirty);
}

#[test]
fn test_evaluation_parameter_no_change_no_dirty() {
    let mut graph = AnimationGraph::new();
    graph.add_parameter("speed", ParameterValue::Float(1.0));
    graph.dirty = false;

    // Setting same value should not mark dirty
    graph.set_parameter("speed", ParameterValue::Float(1.0));
    assert!(!graph.dirty);
}

#[test]
fn test_evaluation_trigger_firing() {
    let mut graph = AnimationGraph::new();
    graph.add_parameter("jump", ParameterValue::Trigger);

    let triggered = graph.trigger("jump");
    assert!(triggered);
    assert!(graph.dirty);
}

#[test]
fn test_evaluation_trigger_not_found() {
    let mut graph = AnimationGraph::new();

    let triggered = graph.trigger("nonexistent");
    assert!(!triggered);
}

#[test]
fn test_evaluation_layer_affects_bone() {
    let layer = AnimationLayer::new("base", LayerBlendMode::Override);

    assert!(layer.affects_bone(0));
    assert!(layer.affects_bone(100));
}

#[test]
fn test_evaluation_masked_layer_affects_bone() {
    let mask = vec![true, true, false, false, true];
    let layer = AnimationLayer::masked_additive("masked", mask);

    assert!(layer.affects_bone(0));
    assert!(layer.affects_bone(1));
    assert!(!layer.affects_bone(2));
    assert!(!layer.affects_bone(3));
    assert!(layer.affects_bone(4));
}

#[test]
fn test_evaluation_effective_weight() {
    let layer = AnimationLayer::new("base", LayerBlendMode::Override).with_weight(0.7);

    assert!((layer.effective_weight(0) - 0.7).abs() < 0.001);
}

#[test]
fn test_evaluation_disabled_layer_weight() {
    let layer = AnimationLayer::new("disabled", LayerBlendMode::Override)
        .with_weight(1.0)
        .disabled();

    assert!((layer.effective_weight(0) - 0.0).abs() < 0.001);
}

#[test]
fn test_evaluation_node_cache_invalidation() {
    let mut node = AnimationNode::clip(0);
    node.cached_pose = Some(Pose::new(10, PoseType::Current));

    node.invalidate_cache();
    assert!(node.cached_pose.is_none());
}

#[test]
fn test_evaluation_blend_mode_additive_check() {
    assert!(LayerBlendMode::Additive.is_additive());
    assert!(LayerBlendMode::MaskedAdditive.is_additive());
    assert!(!LayerBlendMode::Override.is_additive());
}

#[test]
fn test_evaluation_blend_mode_uses_mask() {
    assert!(LayerBlendMode::MaskedAdditive.uses_mask());
    assert!(!LayerBlendMode::Additive.uses_mask());
    assert!(!LayerBlendMode::Override.uses_mask());
}

#[test]
fn test_evaluation_layer_with_root_node() {
    let layer = AnimationLayer::new("base", LayerBlendMode::Override)
        .with_root_node(5);

    assert_eq!(layer.root_node, 5);
}

#[test]
fn test_evaluation_frame_counter() {
    let mut graph = AnimationGraph::new();
    assert_eq!(graph.frame, 0);

    graph.frame = 100;
    assert_eq!(graph.frame, 100);
}

#[test]
fn test_evaluation_graph_with_name() {
    let graph = AnimationGraph::new().with_name("character_controller");
    assert_eq!(graph.name, Some("character_controller".to_string()));
}

// ============================================================================
// PARAMETER SYSTEM TESTS (10+)
// ============================================================================

#[test]
fn test_parameter_float_operations() {
    let mut graph = AnimationGraph::new();

    graph.add_parameter("value", ParameterValue::Float(0.0));

    graph.set_parameter("value", ParameterValue::Float(0.5));
    assert!(matches!(
        graph.get_parameter("value"),
        Some(ParameterValue::Float(v)) if (*v - 0.5).abs() < 0.001
    ));

    graph.set_parameter("value", ParameterValue::Float(1.0));
    assert!(matches!(
        graph.get_parameter("value"),
        Some(ParameterValue::Float(v)) if (*v - 1.0).abs() < 0.001
    ));
}

#[test]
fn test_parameter_int_operations() {
    let mut graph = AnimationGraph::new();

    graph.add_parameter("count", ParameterValue::Int(0));

    graph.set_parameter("count", ParameterValue::Int(10));
    assert!(matches!(
        graph.get_parameter("count"),
        Some(ParameterValue::Int(10))
    ));

    graph.set_parameter("count", ParameterValue::Int(-5));
    assert!(matches!(
        graph.get_parameter("count"),
        Some(ParameterValue::Int(-5))
    ));
}

#[test]
fn test_parameter_bool_operations() {
    let mut graph = AnimationGraph::new();

    graph.add_parameter("active", ParameterValue::Bool(false));

    assert!(matches!(
        graph.get_parameter("active"),
        Some(ParameterValue::Bool(false))
    ));

    graph.set_parameter("active", ParameterValue::Bool(true));
    assert!(matches!(
        graph.get_parameter("active"),
        Some(ParameterValue::Bool(true))
    ));
}

#[test]
fn test_parameter_vec3_operations() {
    let mut graph = AnimationGraph::new();

    graph.add_parameter("velocity", ParameterValue::Vec3(glam::Vec3::ZERO));

    graph.set_parameter("velocity", ParameterValue::Vec3(glam::Vec3::new(1.0, 2.0, 3.0)));

    match graph.get_parameter("velocity") {
        Some(ParameterValue::Vec3(v)) => {
            assert!((v.x - 1.0).abs() < 0.001);
            assert!((v.y - 2.0).abs() < 0.001);
            assert!((v.z - 3.0).abs() < 0.001);
        }
        _ => panic!("Expected Vec3 parameter"),
    }
}

#[test]
fn test_parameter_nonexistent_get() {
    let graph = AnimationGraph::new();
    assert!(graph.get_parameter("nonexistent").is_none());
}

#[test]
fn test_parameter_find_by_name() {
    let mut graph = AnimationGraph::new();

    graph.add_parameter("first", ParameterValue::Float(1.0));
    graph.add_parameter("second", ParameterValue::Float(2.0));
    graph.add_parameter("third", ParameterValue::Float(3.0));

    assert_eq!(graph.find_parameter("first"), Some(0));
    assert_eq!(graph.find_parameter("second"), Some(1));
    assert_eq!(graph.find_parameter("third"), Some(2));
    assert_eq!(graph.find_parameter("fourth"), None);
}

#[test]
fn test_parameter_get_by_index() {
    let mut graph = AnimationGraph::new();

    graph.add_parameter("value", ParameterValue::Float(42.0));

    assert!(matches!(
        graph.get_parameter_by_index(0),
        Some(ParameterValue::Float(v)) if (*v - 42.0).abs() < 0.001
    ));
    assert!(graph.get_parameter_by_index(1).is_none());
}

#[test]
fn test_parameter_set_by_index() {
    let mut graph = AnimationGraph::new();

    graph.add_parameter("value", ParameterValue::Float(0.0));

    let result = graph.set_parameter_by_index(0, ParameterValue::Float(5.0));
    assert!(result);

    assert!(matches!(
        graph.get_parameter_by_index(0),
        Some(ParameterValue::Float(v)) if (*v - 5.0).abs() < 0.001
    ));
}

#[test]
fn test_parameter_set_nonexistent_returns_false() {
    let mut graph = AnimationGraph::new();

    let result = graph.set_parameter("nonexistent", ParameterValue::Float(1.0));
    assert!(!result);
}

#[test]
fn test_parameter_many_parameters() {
    let mut graph = AnimationGraph::new();

    for i in 0..100 {
        graph.add_parameter(&format!("param_{}", i), ParameterValue::Float(i as f32));
    }

    assert_eq!(graph.parameters.len(), 100);

    // Verify random access
    assert!(matches!(
        graph.get_parameter("param_50"),
        Some(ParameterValue::Float(v)) if (*v - 50.0).abs() < 0.001
    ));
}

#[test]
fn test_parameter_value_type_name() {
    assert_eq!(ParameterValue::Float(0.0).type_name(), "float");
    assert_eq!(ParameterValue::Int(0).type_name(), "int");
    assert_eq!(ParameterValue::Bool(false).type_name(), "bool");
    assert_eq!(ParameterValue::Vec3(glam::Vec3::ZERO).type_name(), "vec3");
    assert_eq!(ParameterValue::Trigger.type_name(), "trigger");
}

#[test]
fn test_parameter_is_trigger() {
    assert!(ParameterValue::Trigger.is_trigger());
    assert!(!ParameterValue::Float(0.0).is_trigger());
    assert!(!ParameterValue::Bool(false).is_trigger());
}

// ============================================================================
// STATE MACHINE INTEGRATION TESTS (10+)
// ============================================================================

#[test]
fn test_state_machine_basic_construction() {
    let mut machine = StateMachine::new();

    machine.add_state(AnimationState::clip("idle", 0));
    machine.add_state(AnimationState::clip("walk", 1));

    assert_eq!(machine.state_count(), 2);
}

#[test]
fn test_state_machine_state_content_types() {
    let clip_state = AnimationState::clip("clip", 0);
    let tree_state = AnimationState::blend_tree("tree", 0);
    let sub_state = AnimationState::sub_graph("sub", 0);
    let empty_state = AnimationState::empty("empty");

    assert!(matches!(clip_state.clip_or_tree, StateContent::Clip(0)));
    assert!(matches!(tree_state.clip_or_tree, StateContent::BlendTree(0)));
    assert!(matches!(sub_state.clip_or_tree, StateContent::SubGraph(0)));
    assert!(matches!(empty_state.clip_or_tree, StateContent::Empty));
}

#[test]
fn test_state_machine_state_callbacks() {
    let state = AnimationState::clip("idle", 0)
        .with_on_enter("on_idle_enter")
        .with_on_exit("on_idle_exit");

    assert_eq!(state.on_enter, Some("on_idle_enter".to_string()));
    assert_eq!(state.on_exit, Some("on_idle_exit".to_string()));
}

#[test]
fn test_state_machine_state_speed() {
    let state = AnimationState::clip("run", 0).with_speed(1.5);
    assert!((state.speed_multiplier - 1.5).abs() < 0.001);
}

#[test]
fn test_state_machine_state_tags() {
    let state = AnimationState::clip("idle", 0)
        .with_tag("grounded")
        .with_tag("locomotion");

    assert!(state.has_tag("grounded"));
    assert!(state.has_tag("locomotion"));
    assert!(!state.has_tag("airborne"));
}

#[test]
fn test_state_machine_transition_construction() {
    let transition = Transition::new(Some(0), 1)
        .with_blend_time(0.3)
        .with_blend_curve(BlendCurve::EaseInOut);

    assert_eq!(transition.from_state, Some(0));
    assert_eq!(transition.to_state, 1);
    assert!((transition.blend_time - 0.3).abs() < 0.001);
    assert_eq!(transition.blend_curve, BlendCurve::EaseInOut);
}

#[test]
fn test_state_machine_wildcard_transition() {
    let transition = Transition::wildcard(5);

    assert!(transition.from_state.is_none());
    assert_eq!(transition.to_state, 5);
    assert!(transition.can_interrupt); // Wildcards can interrupt by default
}

#[test]
fn test_state_machine_transition_condition() {
    let condition = SMTransitionCondition::float_param("speed", CompareOp::Greater, 5.0);

    let mut params = ParameterSet::new();
    params.set_float("speed", 6.0);

    let context = EvaluationContext::new(0);
    assert!(condition.evaluate(&params, &context));

    params.set_float("speed", 4.0);
    assert!(!condition.evaluate(&params, &context));
}

#[test]
fn test_state_machine_compound_condition_and() {
    let condition = SMTransitionCondition::and(
        SMTransitionCondition::float_param("speed", CompareOp::Greater, 5.0),
        SMTransitionCondition::bool_param("is_grounded", true),
    );

    let mut params = ParameterSet::new();
    params.set_float("speed", 6.0);
    params.set_bool("is_grounded", true);

    let context = EvaluationContext::new(0);
    assert!(condition.evaluate(&params, &context));

    params.set_bool("is_grounded", false);
    assert!(!condition.evaluate(&params, &context));
}

#[test]
fn test_state_machine_compound_condition_or() {
    let condition = SMTransitionCondition::or(
        SMTransitionCondition::trigger("jump"),
        SMTransitionCondition::trigger("attack"),
    );

    let mut params = ParameterSet::new();
    params.fire_trigger("jump");

    let context = EvaluationContext::new(0);
    assert!(condition.evaluate(&params, &context));
}

#[test]
fn test_state_machine_condition_not() {
    let condition = SMTransitionCondition::not(
        SMTransitionCondition::bool_param("is_active", true),
    );

    let mut params = ParameterSet::new();
    params.set_bool("is_active", false);

    let context = EvaluationContext::new(0);
    assert!(condition.evaluate(&params, &context));
}

#[test]
fn test_state_machine_blend_curves() {
    let curves = vec![
        BlendCurve::Linear,
        BlendCurve::EaseIn,
        BlendCurve::EaseOut,
        BlendCurve::EaseInOut,
        BlendCurve::SmoothStep,
        BlendCurve::SmootherStep,
        BlendCurve::Instant,
    ];

    for curve in curves {
        // Test that apply works for all curves
        let at_0 = curve.apply(0.0);
        let at_half = curve.apply(0.5);
        let at_1 = curve.apply(1.0);

        // All curves should map 0 -> 0ish and 1 -> 1
        assert!(at_0 >= 0.0 && at_0 <= 0.1);
        assert!(at_1 >= 0.9 && at_1 <= 1.0);
        assert!(at_half >= 0.0 && at_half <= 1.0);
    }
}

#[test]
fn test_state_machine_sync_modes() {
    let modes = vec![
        SyncMode::FreezeSource,
        SyncMode::SyncToSource,
        SyncMode::Crossfade,
        SyncMode::Independent,
        SyncMode::FootSync,
    ];

    for mode in modes {
        // Verify each mode has a name
        assert!(!mode.name().is_empty());
    }
}

// ============================================================================
// PERFORMANCE TESTS (5+)
// ============================================================================

#[test]
fn test_performance_parameter_lookup() {
    let mut graph = AnimationGraph::new();

    // Add 100 parameters
    for i in 0..100 {
        graph.add_parameter(&format!("param_{}", i), ParameterValue::Float(i as f32));
    }

    let start = Instant::now();
    let iterations = 10000;

    for i in 0..iterations {
        let idx = i % 100;
        let _ = graph.get_parameter(&format!("param_{}", idx));
    }

    let elapsed = start.elapsed();
    let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;

    // Should be under 10us per lookup
    assert!(avg_us < 10.0, "Parameter lookup too slow: {:.3}us", avg_us);
}

#[test]
fn test_performance_parameter_updates() {
    let mut graph = AnimationGraph::new();

    for i in 0..100 {
        graph.add_parameter(&format!("param_{}", i), ParameterValue::Float(0.0));
    }

    let start = Instant::now();
    let iterations = 10000;

    for i in 0..iterations {
        let idx = i % 100;
        graph.set_parameter(&format!("param_{}", idx), ParameterValue::Float(i as f32));
    }

    let elapsed = start.elapsed();
    let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;

    assert!(avg_us < 15.0, "Parameter update too slow: {:.3}us", avg_us);
}

#[test]
fn test_performance_layer_weight_computation() {
    let mut layers = Vec::new();
    for i in 0..50 {
        layers.push(
            AnimationLayer::new(&format!("layer_{}", i), LayerBlendMode::Additive)
                .with_weight(0.5),
        );
    }

    let start = Instant::now();
    let iterations = 100000;

    for _ in 0..iterations {
        for (idx, layer) in layers.iter().enumerate() {
            let _ = layer.effective_weight(idx % 60);
        }
    }

    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() as f64 / (iterations * 50) as f64;

    assert!(avg_ns < 100.0, "Layer weight computation too slow: {:.1}ns", avg_ns);
}

#[test]
fn test_performance_condition_evaluation() {
    let condition = SMTransitionCondition::and(
        SMTransitionCondition::float_param("speed", CompareOp::Greater, 5.0),
        SMTransitionCondition::and(
            SMTransitionCondition::bool_param("is_grounded", true),
            SMTransitionCondition::not(
                SMTransitionCondition::bool_param("is_attacking", true),
            ),
        ),
    );

    let mut params = ParameterSet::new();
    params.set_float("speed", 6.0);
    params.set_bool("is_grounded", true);
    params.set_bool("is_attacking", false);

    let context = EvaluationContext::new(0);

    let start = Instant::now();
    let iterations = 100000;

    for _ in 0..iterations {
        let _ = condition.evaluate(&params, &context);
    }

    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() as f64 / iterations as f64;

    assert!(avg_ns < 500.0, "Condition evaluation too slow: {:.1}ns", avg_ns);
}

#[test]
fn test_performance_blend_curve_application() {
    let curves = vec![
        BlendCurve::Linear,
        BlendCurve::EaseIn,
        BlendCurve::EaseOut,
        BlendCurve::EaseInOut,
        BlendCurve::SmoothStep,
        BlendCurve::SmootherStep,
    ];

    let start = Instant::now();
    let iterations = 100000;

    for i in 0..iterations {
        let t = (i as f32 / iterations as f32) % 1.0;
        for curve in &curves {
            let _ = curve.apply(t);
        }
    }

    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() as f64 / (iterations * curves.len()) as f64;

    assert!(avg_ns < 50.0, "Blend curve application too slow: {:.1}ns", avg_ns);
}

#[test]
fn test_performance_sync_group_operations() {
    let mut manager = SyncGroupManager::new();

    // Create multiple sync groups with members
    for i in 0..10 {
        let mut group = SyncGroup::new(format!("group_{}", i));
        for j in 0..10 {
            let member = SyncMember::new(j);
            group.add_member(member);
        }
        manager.add_group(group);
    }

    let start = Instant::now();
    let iterations = 10000;

    for _ in 0..iterations {
        manager.update(0.016);
    }

    let elapsed = start.elapsed();
    let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;

    assert!(avg_us < 50.0, "Sync group update too slow: {:.3}us", avg_us);
}

// ============================================================================
// ERROR HANDLING TESTS (10+)
// ============================================================================

#[test]
fn test_error_invalid_parameter_name() {
    let graph = AnimationGraph::new();

    assert!(graph.get_parameter("nonexistent").is_none());
    assert!(graph.find_parameter("nonexistent").is_none());
}

#[test]
fn test_error_invalid_parameter_index() {
    let graph = AnimationGraph::new();

    assert!(graph.get_parameter_by_index(0).is_none());
    assert!(graph.get_parameter_by_index(100).is_none());
}

#[test]
fn test_error_set_nonexistent_parameter() {
    let mut graph = AnimationGraph::new();

    let result = graph.set_parameter("nonexistent", ParameterValue::Float(1.0));
    assert!(!result);
}

#[test]
fn test_error_set_nonexistent_parameter_by_index() {
    let mut graph = AnimationGraph::new();

    let result = graph.set_parameter_by_index(0, ParameterValue::Float(1.0));
    assert!(!result);
}

#[test]
fn test_error_trigger_nonexistent() {
    let mut graph = AnimationGraph::new();

    let result = graph.trigger("nonexistent");
    assert!(!result);
}

#[test]
fn test_error_trigger_non_trigger_parameter() {
    let mut graph = AnimationGraph::new();
    graph.add_parameter("value", ParameterValue::Float(0.0));

    // Should fail - not a trigger
    let result = graph.trigger("value");
    assert!(!result);
}

#[test]
fn test_error_layer_mask_out_of_bounds() {
    let mask = vec![true, true, false];
    let layer = AnimationLayer::masked_additive("masked", mask);

    // Out of bounds access should return false
    assert!(!layer.affects_bone(100));
}

#[test]
fn test_error_weight_clamping() {
    let layer = AnimationLayer::new("test", LayerBlendMode::Override)
        .with_weight(2.0); // Should clamp to 1.0

    assert!((layer.weight - 1.0).abs() < 0.001);

    let layer2 = AnimationLayer::new("test", LayerBlendMode::Override)
        .with_weight(-0.5); // Should clamp to 0.0

    assert!((layer2.weight - 0.0).abs() < 0.001);
}

#[test]
fn test_error_compare_op_with_mismatched_types() {
    let condition = SMTransitionCondition::float_param("speed", CompareOp::Greater, 5.0);

    let mut params = ParameterSet::new();
    params.set_bool("speed", true); // Wrong type

    let context = EvaluationContext::new(0);

    // Should return false for type mismatch (or handle gracefully)
    let result = condition.evaluate(&params, &context);
    // Just verify it doesn't panic
    let _ = result;
}

#[test]
fn test_error_empty_blend_positions() {
    // Verify that creating a blend node with empty arrays doesn't panic
    let node = AnimationNode::blend_1d(vec![], vec![], "param");

    match &node.node_type {
        AnimationNodeType::Blend1D { children, positions, .. } => {
            assert!(children.is_empty());
            assert!(positions.is_empty());
        }
        _ => panic!("Wrong node type"),
    }
}

#[test]
fn test_error_state_content_checks() {
    let empty = StateContent::Empty;
    let clip = StateContent::Clip(0);
    let tree = StateContent::BlendTree(1);
    let sub = StateContent::SubGraph(2);

    assert!(empty.is_empty());
    assert!(!clip.is_empty());

    assert_eq!(clip.clip_index(), Some(0));
    assert_eq!(tree.clip_index(), None);

    assert_eq!(tree.blend_tree_index(), Some(1));
    assert_eq!(clip.blend_tree_index(), None);

    assert_eq!(sub.sub_graph_index(), Some(2));
    assert_eq!(clip.sub_graph_index(), None);
}

#[test]
fn test_error_evaluation_context_defaults() {
    let context = EvaluationContext::default();

    assert!(!context.state_complete);
    assert!((context.current_time - 0.0).abs() < 0.001);
    assert!((context.normalized_time - 0.0).abs() < 0.001);
    assert_eq!(context.current_state, 0);
}

#[test]
fn test_error_transition_condition_always_never() {
    let always = SMTransitionCondition::always();
    let never = SMTransitionCondition::never();

    let params = ParameterSet::new();
    let context = EvaluationContext::new(0);

    assert!(always.evaluate(&params, &context));
    assert!(!never.evaluate(&params, &context));
}

// ============================================================================
// ADDITIONAL INTEGRATION TESTS
// ============================================================================

#[test]
fn test_integration_layer_stack_construction() {
    let mut stack = LayerStack::new(60);

    stack.add_layer(
        crate::animation_layers::AnimationLayer::new("base", crate::animation_layers::LayerBlendMode::Override)
    );
    stack.add_layer(
        crate::animation_layers::AnimationLayer::new("additive", crate::animation_layers::LayerBlendMode::Additive)
    );

    assert_eq!(stack.layer_count(), 2);
}

#[test]
fn test_integration_layer_mask_preset() {
    let upper_body = MaskPreset::UpperBody;
    let lower_body = MaskPreset::LowerBody;
    let full_body = MaskPreset::FullBody;

    // Just verify the presets exist and can be used
    assert!(!upper_body.name().is_empty());
    assert!(!lower_body.name().is_empty());
    assert!(!full_body.name().is_empty());
}

#[test]
fn test_integration_sync_group_manager() {
    let mut manager = SyncGroupManager::new();

    let group = SyncGroup::new("locomotion").with_mode(SyncGroupMode::PhaseLocked);
    let group_id = manager.add_group(group);

    // Verify group was added
    assert!(manager.get_group(group_id).is_some());

    // Add a sync track
    let mut track = SyncTrack::new(0).with_name("walk");
    track.add_marker(SyncMarker::foot_contact("left_foot_down", 0.0));
    track.add_marker(SyncMarker::foot_contact("right_foot_down", 0.5));
    manager.add_track(track);

    assert!(manager.has_track(0));
}

#[test]
fn test_integration_sync_track_markers() {
    let mut track = SyncTrack::new(0).with_duration(1.0);

    track.add_marker(SyncMarker::new("start", 0.0));
    track.add_marker(SyncMarker::foot_contact("left_foot", 0.25));
    track.add_marker(SyncMarker::foot_contact("right_foot", 0.75));
    track.add_marker(SyncMarker::event("end", 1.0));

    assert_eq!(track.marker_count(), 4);

    let left = track.find_marker("left_foot");
    assert!(left.is_some());
    assert!((left.unwrap().normalized_time - 0.25).abs() < 0.001);

    let foot_markers = track.foot_contacts();
    assert_eq!(foot_markers.len(), 2);
}

#[test]
fn test_integration_comparison_op_conversions() {
    // Test ComparisonOp from animation_graph
    let ops = vec![
        ComparisonOp::Less,
        ComparisonOp::LessEqual,
        ComparisonOp::Greater,
        ComparisonOp::GreaterEqual,
        ComparisonOp::Equal,
        ComparisonOp::NotEqual,
    ];

    for op in ops {
        assert!(!op.symbol().is_empty());
    }
}

#[test]
fn test_integration_full_graph_setup() {
    // Create a complete animation graph setup similar to what a game would use
    let mut graph = AnimationGraph::with_bone_count(60);
    graph.name = Some("character".to_string());

    // Add locomotion parameters
    graph.add_parameter("speed", ParameterValue::Float(0.0));
    graph.add_parameter("direction", ParameterValue::Float(0.0));
    graph.add_parameter("is_grounded", ParameterValue::Bool(true));

    // Add action parameters
    graph.add_parameter("jump", ParameterValue::Trigger);
    graph.add_parameter("attack", ParameterValue::Trigger);

    // Base locomotion layer
    graph.layers.push(AnimationLayer::base("locomotion"));

    // Upper body overlay layer
    let upper_mask = vec![false; 30].into_iter().chain(vec![true; 30]).collect();
    graph.layers.push(AnimationLayer::masked_additive("upper_body", upper_mask));

    // Add locomotion blend tree
    let locomotion_node = AnimationNode::blend_1d(
        vec![0, 1, 2], // idle, walk, run clip indices
        vec![0.0, 0.5, 1.0],
        "speed",
    ).with_name("locomotion_blend");
    graph.nodes.push(locomotion_node);

    // Set up layer root nodes
    graph.layers[0].root_node = 0;

    // Verify graph structure
    assert_eq!(graph.parameters.len(), 5);
    assert_eq!(graph.layers.len(), 2);
    assert_eq!(graph.nodes.len(), 1);
    assert_eq!(graph.bone_count, 60);
}

#[test]
fn test_integration_state_machine_full_setup() {
    let mut machine = StateMachine::new();

    // Add states
    machine.add_state(
        AnimationState::clip("idle", 0)
            .with_tag("grounded")
            .with_looping(true),
    );
    machine.add_state(
        AnimationState::clip("walk", 1)
            .with_tag("grounded")
            .with_tag("locomotion"),
    );
    machine.add_state(
        AnimationState::clip("run", 2)
            .with_tag("grounded")
            .with_tag("locomotion")
            .with_speed(1.2),
    );
    machine.add_state(
        AnimationState::clip("jump", 3)
            .with_tag("airborne")
            .with_looping(false)
            .with_on_enter("on_jump_start")
            .with_on_exit("on_jump_end"),
    );

    // Add transitions
    machine.add_transition(
        Transition::direct(0, 1)
            .with_condition(SMTransitionCondition::float_param("speed", CompareOp::Greater, 0.1))
            .with_blend_time(0.2)
            .with_blend_curve(BlendCurve::EaseInOut),
    );

    machine.add_transition(
        Transition::direct(1, 2)
            .with_condition(SMTransitionCondition::float_param("speed", CompareOp::Greater, 5.0))
            .with_blend_time(0.3),
    );

    machine.add_transition(
        Transition::wildcard(3)
            .with_condition(SMTransitionCondition::trigger("jump")),
    );

    assert_eq!(machine.state_count(), 4);
    assert_eq!(machine.transition_count(), 3);
}

#[test]
fn test_integration_pose_creation() {
    let pose = Pose::new(60, PoseType::Current);
    assert_eq!(pose.bone_count(), 60);
    assert_eq!(pose.pose_type, PoseType::Current);
}

#[test]
fn test_integration_pose_types() {
    assert!(!PoseType::Bind.is_additive());
    assert!(!PoseType::Reference.is_additive());
    assert!(!PoseType::Current.is_additive());
    assert!(PoseType::Additive.is_additive());
}

#[test]
fn test_integration_transition_state_complete() {
    let condition = SMTransitionCondition::state_complete();

    let params = ParameterSet::new();

    let context_incomplete = EvaluationContext::new(0);
    assert!(!condition.evaluate(&params, &context_incomplete));

    let context_complete = EvaluationContext::new(0).with_complete(true);
    assert!(condition.evaluate(&params, &context_complete));
}

#[test]
fn test_integration_transition_time_exceeds() {
    let condition = SMTransitionCondition::time_exceeds(1.0);

    let params = ParameterSet::new();

    let context_before = EvaluationContext::new(0).with_time(0.5, 0.5);
    assert!(!condition.evaluate(&params, &context_before));

    let context_after = EvaluationContext::new(0).with_time(1.5, 0.75);
    assert!(condition.evaluate(&params, &context_after));
}

#[test]
fn test_integration_foot_sync_state() {
    let mut foot_state = FootSyncState::new();

    // Initial state: left foot planted
    assert!(foot_state.left_planted);
    assert!(!foot_state.right_planted);

    // Update to move to mid-cycle
    foot_state.set_phase(0.5);

    // At 0.5, right foot should be planted
    assert!(!foot_state.left_planted);
    assert!(foot_state.right_planted);
}

#[test]
fn test_integration_sync_group_modes() {
    let modes = vec![
        SyncGroupMode::Leader,
        SyncGroupMode::Follower,
        SyncGroupMode::PhaseLocked,
        SyncGroupMode::MarkerAligned,
        SyncGroupMode::Independent,
    ];

    for mode in modes {
        assert!(!mode.name().is_empty());
    }

    assert!(SyncGroupMode::Leader.requires_master());
    assert!(!SyncGroupMode::Independent.requires_master());

    assert!(SyncGroupMode::PhaseLocked.synchronizes_phase());
    assert!(!SyncGroupMode::Independent.synchronizes_phase());
}

#[test]
fn test_integration_marker_types() {
    let types = vec![
        MarkerType::Generic,
        MarkerType::FootContact,
        MarkerType::FootLift,
        MarkerType::Beat,
        MarkerType::Event,
        MarkerType::LoopPoint,
    ];

    for marker_type in types {
        assert!(!marker_type.name().is_empty());
    }

    assert!(MarkerType::FootContact.is_foot_marker());
    assert!(MarkerType::FootLift.is_foot_marker());
    assert!(!MarkerType::Beat.is_foot_marker());
}
