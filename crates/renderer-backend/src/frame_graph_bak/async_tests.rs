use super::*;

// =========================================================================
// AsyncExecutionPlan -- whitebox tests (T-FG-5.1)
// =========================================================================

// -- Structure & traits ---------------------------------------------------

#[test]
fn async_execution_plan_has_public_fields() {
    let plan = AsyncExecutionPlan {
        graphics_queue: vec![PassIndex(0)],
        async_queues: HashMap::from([("compute".into(), vec![PassIndex(1)])]),
    };
    assert_eq!(plan.graphics_queue, vec![PassIndex(0)]);
    assert_eq!(
        plan.async_queues.get("compute"),
        Some(&vec![PassIndex(1)])
    );
}

#[test]
fn async_execution_plan_default_is_empty() {
    let plan = AsyncExecutionPlan::default();
    assert!(plan.graphics_queue.is_empty());
    assert!(plan.async_queues.is_empty());
}

#[test]
fn async_execution_plan_clone_produces_independent_copy() {
    let plan = AsyncExecutionPlan {
        graphics_queue: vec![PassIndex(0), PassIndex(1)],
        async_queues: HashMap::from([("compute".into(), vec![PassIndex(2)])]),
    };
    let cloned = plan.clone();
    assert_eq!(plan, cloned, "clone must be equal to original");
    let mut cloned = cloned;
    cloned.graphics_queue.push(PassIndex(99));
    assert_ne!(plan.graphics_queue.len(), cloned.graphics_queue.len());
}

#[test]
fn async_execution_plan_debug_output() {
    let plan = AsyncExecutionPlan {
        graphics_queue: vec![PassIndex(0)],
        async_queues: HashMap::from([("compute".into(), vec![PassIndex(1)])]),
    };
    let dbg = format!("{:?}", plan);
    assert!(dbg.contains("graphics_queue"));
    assert!(dbg.contains("async_queues"));
    assert!(dbg.contains("compute"));
}

#[test]
fn async_execution_plan_partial_eq() {
    let a = AsyncExecutionPlan {
        graphics_queue: vec![PassIndex(0)],
        async_queues: HashMap::new(),
    };
    let b = AsyncExecutionPlan {
        graphics_queue: vec![PassIndex(0)],
        async_queues: HashMap::new(),
    };
    assert_eq!(a, b, "identical plans must be equal");
    let c = AsyncExecutionPlan {
        graphics_queue: vec![PassIndex(0)],
        async_queues: HashMap::from([("compute".into(), vec![PassIndex(1)])]),
    };
    assert_ne!(a, c, "different async_queues must not be equal");
}

// -- build_async_plan internal queue construction ------------------------

#[test]
fn build_async_plan_uses_hashset_for_async_membership() {
    let order = vec![PassIndex(0), PassIndex(1)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(0), "compute".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);
    let compute_queue = plan.async_queues.get("compute").unwrap();
    assert_eq!(compute_queue.len(), 1,
        "HashSet dedup means each pass appears at most once per queue");
    assert_eq!(compute_queue[0], PassIndex(0));
}

#[test]
fn build_async_plan_iterates_order_exactly_once() {
    let order = vec![PassIndex(10), PassIndex(20), PassIndex(30), PassIndex(40)];
    let async_passes = vec![
        (PassIndex(20), "compute".to_string()),
        (PassIndex(40), "copy".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);
    let total: usize = plan.graphics_queue.len()
        + plan.async_queues.values().map(|v| v.len()).sum::<usize>();
    assert_eq!(total, order.len(), "every input pass goes to exactly one queue");
}

#[test]
fn build_async_plan_async_passes_not_in_order_are_ignored() {
    let order = vec![PassIndex(0)];
    let async_passes = vec![(PassIndex(99), "compute".to_string())];
    let plan = build_async_plan(&order, &async_passes);
    assert_eq!(plan.graphics_queue, vec![PassIndex(0)]);
    assert!(!plan.async_queues.contains_key("compute"),
        "queue for non-existent pass must not be created");
}

#[test]
fn build_async_plan_multiple_same_type_queues_share_an_entry() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3), PassIndex(4)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "compute".to_string()),
        (PassIndex(4), "compute".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);
    assert_eq!(plan.async_queues.len(), 1, "only one queue type");
    let compute_queue = plan.async_queues.get("compute").unwrap();
    assert_eq!(compute_queue.len(), 3);
}

#[test]
fn build_async_plan_preserves_graphics_queue_order() {
    let order = vec![PassIndex(5), PassIndex(3), PassIndex(1), PassIndex(4), PassIndex(2)];
    let async_passes = vec![
        (PassIndex(3), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);
    let expected_gfx = vec![PassIndex(5), PassIndex(1), PassIndex(4)];
    assert_eq!(plan.graphics_queue, expected_gfx,
        "graphics_queue must preserve input order of non-async passes");
}

#[test]
fn build_async_plan_preserves_async_queue_order() {
    let order = vec![PassIndex(7), PassIndex(2), PassIndex(5), PassIndex(1), PassIndex(9)];
    let async_passes = vec![
        (PassIndex(2), "compute".to_string()),
        (PassIndex(1), "compute".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);
    let compute_queue = plan.async_queues.get("compute").unwrap();
    assert_eq!(compute_queue, &vec![PassIndex(2), PassIndex(1)],
        "async queue must preserve input order");
}

#[test]
fn build_async_plan_with_all_non_async_passes() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes: Vec<(PassIndex, String)> = vec![];
    let plan = build_async_plan(&order, &async_passes);
    assert_eq!(plan.graphics_queue, order, "all passes stay in graphics");
    assert!(plan.async_queues.is_empty(), "no async queues created");
}

#[test]
fn build_async_plan_with_all_async_passes() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "compute".to_string()),
        (PassIndex(2), "compute".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);
    assert!(plan.graphics_queue.is_empty(), "graphics_queue is empty");
    assert_eq!(plan.async_queues.len(), 1, "one async queue");
    assert_eq!(plan.async_queues.get("compute").unwrap().len(), 3,
        "all three passes in compute");
}

// -- Multiple queue types (compute + copy) ------------------------------

#[test]
fn build_async_plan_two_queue_types_separate_passes() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);
    assert_eq!(plan.async_queues.len(), 2, "two queue types");
    assert_eq!(plan.async_queues.get("compute").unwrap(), &vec![PassIndex(0)]);
    assert_eq!(plan.async_queues.get("copy").unwrap(), &vec![PassIndex(2)]);
}

#[test]
fn build_async_plan_three_queue_types() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
        (PassIndex(3), "transfer".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);
    assert_eq!(plan.async_queues.len(), 3);
    assert!(plan.async_queues.contains_key("compute"));
    assert!(plan.async_queues.contains_key("copy"));
    assert!(plan.async_queues.contains_key("transfer"));
}

#[test]
fn build_async_plan_interleaved_queue_types() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3), PassIndex(4)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
        (PassIndex(3), "compute".to_string()),
        (PassIndex(4), "copy".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);
    assert_eq!(plan.async_queues.get("compute").unwrap(), &vec![PassIndex(0), PassIndex(3)],
        "compute queue preserves interleaved order");
    assert_eq!(plan.async_queues.get("copy").unwrap(), &vec![PassIndex(1), PassIndex(4)],
        "copy queue preserves interleaved order");
    assert_eq!(plan.graphics_queue, vec![PassIndex(2)]);
}

// -- is_async_pass: edge cases ------------------------------------------

#[test]
fn is_async_pass_with_empty_list() {
    let empty: Vec<(PassIndex, String)> = vec![];
    assert!(!is_async_pass(PassIndex(0), &empty));
    assert!(!is_async_pass(PassIndex(1), &empty));
    assert!(!is_async_pass(PassIndex(usize::MAX), &empty));
}

#[test]
fn is_async_pass_with_non_existent_index() {
    let passes = vec![
        (PassIndex(10), "compute".to_string()),
        (PassIndex(20), "compute".to_string()),
    ];
    assert!(!is_async_pass(PassIndex(0), &passes), "index 0 not in list");
    assert!(!is_async_pass(PassIndex(15), &passes), "index 15 not in list");
    assert!(!is_async_pass(PassIndex(30), &passes), "index 30 not in list");
}

#[test]
fn is_async_pass_ignores_queue_type() {
    let passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
        (PassIndex(2), "".to_string()),
        (PassIndex(3), "arbitrary_weird_type".to_string()),
    ];
    assert!(is_async_pass(PassIndex(0), &passes));
    assert!(is_async_pass(PassIndex(1), &passes));
    assert!(is_async_pass(PassIndex(2), &passes));
    assert!(is_async_pass(PassIndex(3), &passes));
    assert!(!is_async_pass(PassIndex(4), &passes));
}

#[test]
fn is_async_pass_with_duplicate_entries() {
    let passes = vec![
        (PassIndex(5), "compute".to_string()),
        (PassIndex(5), "copy".to_string()),
    ];
    assert!(is_async_pass(PassIndex(5), &passes));
}

#[test]
fn is_async_pass_large_index() {
    let passes = vec![(PassIndex(999_999), "compute".to_string())];
    assert!(is_async_pass(PassIndex(999_999), &passes));
    assert!(!is_async_pass(PassIndex(1_000_000), &passes));
}

#[test]
fn is_async_pass_zero_index() {
    let passes = vec![(PassIndex(0), "compute".to_string())];
    assert!(is_async_pass(PassIndex(0), &passes));
}

// -- build_async_plan: edge cases ---------------------------------------

#[test]
fn build_async_plan_empty_order_and_empty_async() {
    let plan = build_async_plan(&[], &[]);
    assert!(plan.graphics_queue.is_empty());
    assert!(plan.async_queues.is_empty());
}

#[test]
fn build_async_plan_single_non_async_pass() {
    let plan = build_async_plan(&[PassIndex(42)], &[]);
    assert_eq!(plan.graphics_queue, vec![PassIndex(42)]);
    assert!(plan.async_queues.is_empty());
}

#[test]
fn build_async_plan_single_async_pass() {
    let plan = build_async_plan(&[PassIndex(7)], &[(PassIndex(7), "compute".to_string())]);
    assert!(plan.graphics_queue.is_empty());
    assert_eq!(plan.async_queues.get("compute"), Some(&vec![PassIndex(7)]));
}

#[test]
fn build_async_plan_large_order_preserves_order() {
    let order: Vec<PassIndex> = (0..100).map(PassIndex).collect();
    let async_passes: Vec<(PassIndex, String)> = (0..100)
        .filter(|i| i % 2 == 0)
        .map(|i| (PassIndex(i), "compute".to_string()))
        .collect();
    let plan = build_async_plan(&order, &async_passes);
    let expected_gfx: Vec<PassIndex> = (0..100).filter(|i| i % 2 != 0).map(PassIndex).collect();
    assert_eq!(plan.graphics_queue, expected_gfx);
    let expected_comp: Vec<PassIndex> = (0..100).filter(|i| i % 2 == 0).map(PassIndex).collect();
    assert_eq!(plan.async_queues.get("compute").unwrap(), &expected_comp);
}

#[test]
fn build_async_plan_with_various_queue_type_strings() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "COMPUTE".to_string()),
        (PassIndex(2), " compute ".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);
    assert_eq!(plan.async_queues.len(), 3);
    assert_eq!(plan.async_queues.get("compute"), Some(&vec![PassIndex(0)]));
    assert_eq!(plan.async_queues.get("COMPUTE"), Some(&vec![PassIndex(1)]));
    assert_eq!(plan.async_queues.get(" compute "), Some(&vec![PassIndex(2)]));
    assert!(plan.graphics_queue.is_empty());
}

#[test]
fn build_async_plan_no_async_passes_does_not_create_queues() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let plan = build_async_plan(&order, &[]);
    assert!(plan.async_queues.is_empty(), "no entries in async_queues");
    assert_eq!(plan.graphics_queue.len(), 3);
}

#[test]
fn build_async_plan_or_default_creates_new_entry_on_first_encounter() {
    let plan = build_async_plan(&[PassIndex(0)], &[(PassIndex(0), "compute".to_string())]);
    let queue = plan.async_queues.get("compute");
    assert!(queue.is_some(), "compute key must exist");
    assert!(!queue.unwrap().is_empty(), "compute queue must be non-empty");
}

#[test]
fn build_async_plan_capacity_hint_does_not_affect_correctness() {
    let order: Vec<PassIndex> = (0..10_000).map(PassIndex).collect();
    let async_passes: Vec<(PassIndex, String)> = vec![];
    let plan = build_async_plan(&order, &async_passes);
    assert_eq!(plan.graphics_queue.len(), 10_000);
    assert_eq!(plan.graphics_queue, order);
}

#[test]
fn build_async_plan_never_panics_any_input() {
    let ap0: [(PassIndex, String); 0] = [];
    let ap1 = [(PassIndex(0), "compute".to_string())];
    let ap2 = [(PassIndex(0), "".to_string())];
    let ap3 = [
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
    ];
    let ap4 = [(PassIndex(usize::MAX), "compute".to_string())];
    let cases: [(&[PassIndex], &[(PassIndex, String)]); 6] = [
        (&[], &ap0),
        (&[PassIndex(0)], &[]),
        (&[PassIndex(0)], &ap1),
        (&[PassIndex(0), PassIndex(1)], &ap2),
        (&[PassIndex(0), PassIndex(1)], &ap3),
        (&[PassIndex(usize::MAX)], &ap4),
    ];
    for (order, async_passes) in cases {
        let plan = build_async_plan(order, async_passes);
        let total: usize = plan.graphics_queue.len()
            + plan.async_queues.values().map(|v| v.len()).sum::<usize>();
        assert_eq!(total, order.len(), "total must equal order.len()");
    }
}

// -- Doc-test-like verification -----------------------------------------

#[test]
fn is_async_pass_doc_example() {
    let passes = vec![(PassIndex(0), "compute".to_string())];
    assert!(is_async_pass(PassIndex(0), &passes));
    assert!(!is_async_pass(PassIndex(1), &passes));
}


// =========================================================================
// async_schedule integration tests (T-FG-5.1)
// =========================================================================

#[test]
fn async_schedule_compute_only_no_blocking() {
    let passes = vec![
        IrPass::compute(PassIndex(0), "c0", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
        IrPass::compute(PassIndex(1), "c1", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];
    let result = async_schedule(&order, &passes, &[]);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], (PassIndex(0), "compute".to_string()));
    assert_eq!(result[1], (PassIndex(1), "compute".to_string()));
}

#[test]
fn async_schedule_graphics_blocks_compute_via_raw() {
    let passes = vec![
        IrPass::graphics(PassIndex(0), "g0", vec![ColorAttachment::default()], None,
            InstanceSource::Direct { index_count: 0, instance_count: 1, base_vertex: 0, first_index: 0, first_instance: 0 },
            ViewType::Texture2D),
        IrPass::compute(PassIndex(1), "c1", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW)];
    let result = async_schedule(&order, &passes, &edges);
    assert!(result.is_empty(), "compute pass blocked by graphics RAW edge");
}

#[test]
fn async_schedule_mixed_blocked_and_free() {
    let passes = vec![
        IrPass::graphics(PassIndex(0), "g0", vec![ColorAttachment::default()], None,
            InstanceSource::Direct { index_count: 0, instance_count: 1, base_vertex: 0, first_index: 0, first_instance: 0 },
            ViewType::Texture2D),
        IrPass::compute(PassIndex(1), "c1", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
        IrPass::compute(PassIndex(2), "c2", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW)];
    let result = async_schedule(&order, &passes, &edges);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], (PassIndex(2), "compute".to_string()));
}

#[test]
fn async_schedule_copy_async() {
    let passes = vec![
        IrPass::copy(PassIndex(0), "cp0"),
        IrPass::copy(PassIndex(1), "cp1"),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];
    let result = async_schedule(&order, &passes, &[]);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].1, "copy");
    assert_eq!(result[1].1, "copy");
}

#[test]
fn async_schedule_raytracing_blocks_compute() {
    let passes = vec![
        IrPass::ray_tracing(PassIndex(0), "r0", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }),
        IrPass::compute(PassIndex(1), "c1", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW)];
    let result = async_schedule(&order, &passes, &edges);
    assert!(result.is_empty(), "compute pass blocked by raytracing RAW edge");
}

#[test]
fn async_schedule_all_pass_types() {
    let passes = vec![
        IrPass::graphics(PassIndex(0), "g0", vec![ColorAttachment::default()], None,
            InstanceSource::Direct { index_count: 0, instance_count: 1, base_vertex: 0, first_index: 0, first_instance: 0 },
            ViewType::Texture2D),
        IrPass::compute(PassIndex(1), "c1", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
        IrPass::copy(PassIndex(2), "cp2"),
        IrPass::ray_tracing(PassIndex(3), "r3", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let result = async_schedule(&order, &passes, &[]);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], (PassIndex(1), "compute".to_string()));
    assert_eq!(result[1], (PassIndex(2), "copy".to_string()));
}

#[test]
fn async_schedule_empty_input() {
    let result = async_schedule(&[], &[], &[]);
    assert!(result.is_empty());
}

#[test]
fn async_schedule_graphics_skipped() {
    let passes = vec![
        IrPass::graphics(PassIndex(0), "g0", vec![ColorAttachment::default()], None,
            InstanceSource::Direct { index_count: 0, instance_count: 1, base_vertex: 0, first_index: 0, first_instance: 0 },
            ViewType::Texture2D),
        IrPass::graphics(PassIndex(1), "g1", vec![ColorAttachment::default()], None,
            InstanceSource::Direct { index_count: 0, instance_count: 1, base_vertex: 0, first_index: 0, first_instance: 0 },
            ViewType::Texture2D),
    ];
    let result = async_schedule(&[PassIndex(0), PassIndex(1)], &passes, &[]);
    assert!(result.is_empty(), "graphics passes are never async");
}

#[test]
fn async_schedule_raytracing_skipped() {
    let passes = vec![
        IrPass::ray_tracing(PassIndex(0), "r0", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }),
    ];
    let result = async_schedule(&[PassIndex(0)], &passes, &[]);
    assert!(result.is_empty(), "raytracing passes are never async");
}

#[test]
fn async_schedule_non_raw_edges_dont_block() {
    let passes = vec![
        IrPass::graphics(PassIndex(0), "g0", vec![ColorAttachment::default()], None,
            InstanceSource::Direct { index_count: 0, instance_count: 1, base_vertex: 0, first_index: 0, first_instance: 0 },
            ViewType::Texture2D),
        IrPass::compute(PassIndex(1), "c1", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::WAR),
        IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(2), EdgeType::WAW),
    ];
    let result = async_schedule(&order, &passes, &edges);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], (PassIndex(1), "compute".to_string()));
}

#[test]
fn async_schedule_raw_from_compute_does_not_block() {
    let passes = vec![
        IrPass::compute(PassIndex(0), "c0", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
        IrPass::compute(PassIndex(1), "c1", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW)];
    let result = async_schedule(&order, &passes, &edges);
    assert_eq!(result.len(), 2, "compute-to-compute RAW does not block");
}

#[test]
fn async_schedule_raw_from_copy_does_not_block() {
    let passes = vec![
        IrPass::copy(PassIndex(0), "cp0"),
        IrPass::compute(PassIndex(1), "c1", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW)];
    let result = async_schedule(&order, &passes, &edges);
    assert_eq!(result.len(), 2, "copy-to-compute RAW does not block");
}

#[test]
fn async_schedule_round_trip_with_build_async_plan() {
    let passes = vec![
        IrPass::graphics(PassIndex(0), "g0", vec![ColorAttachment::default()], None,
            InstanceSource::Direct { index_count: 0, instance_count: 1, base_vertex: 0, first_index: 0, first_instance: 0 },
            ViewType::Texture2D),
        IrPass::compute(PassIndex(1), "c1", DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 }, ViewType::Storage),
        IrPass::copy(PassIndex(2), "cp2"),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let edges = vec![];
    let async_passes = async_schedule(&order, &passes, &edges);
    let plan = build_async_plan(&order, &async_passes);
    assert_eq!(plan.graphics_queue, vec![PassIndex(0)]);
    assert_eq!(plan.async_queues.get("compute").unwrap(), &vec![PassIndex(1)]);
    assert_eq!(plan.async_queues.get("copy").unwrap(), &vec![PassIndex(2)]);
}

// =========================================================================
// Additional internal queue construction tests (T-FG-5.1)
// =========================================================================

#[test]
fn build_async_plan_async_set_membership_controls_queue_assignment() {
    // Verifies the internal HashSet membership check: passes NOT in the
    // async_set go to graphics_queue, those IN async_set go to their
    // respective async queue with type resolved by find().
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    // Only indices 1 and 3 are in the async set.
    let async_passes = vec![
        (PassIndex(1), "compute".to_string()),
        (PassIndex(3), "compute".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(
        plan.graphics_queue,
        vec![PassIndex(0), PassIndex(2)],
        "indices not in async_set remain on graphics queue",
    );
    let q = plan.async_queues.get("compute").unwrap();
    assert_eq!(q, &vec![PassIndex(1), PassIndex(3)]);
}

#[test]
fn build_async_plan_same_index_two_queues_uses_first_match() {
    // Regression guard: when the same pass index appears in async_passes
    // with two different queue types, the HashSet dedup means it enters
    // the loop body once, and `find()` returns ONLY the first match.
    // The pass is pushed to the first queue type only.
    let order = vec![PassIndex(0)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(0), "copy".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);

    // The pass goes to "compute" (first match) -- never to "copy".
    assert_eq!(
        plan.async_queues.get("compute"),
        Some(&vec![PassIndex(0)]),
        "pass goes to first-matched queue type",
    );
    assert_eq!(
        plan.async_queues.get("copy"),
        None,
        "second queue type must NOT receive the pass",
    );
    assert!(
        plan.graphics_queue.is_empty(),
        "async-indexed pass must NOT remain on graphics queue",
    );
}

#[test]
fn build_async_plan_or_default_does_not_share_entries_across_queues() {
    // Tests that `entry(queue_type).or_default()` creates a NEW Vec for
    // each distinct queue type string, so mutating one queue never
    // affects another.
    let order = vec![PassIndex(0), PassIndex(1)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);

    let compute_ptr: *const Vec<PassIndex> = &plan.async_queues["compute"];
    let copy_ptr: *const Vec<PassIndex> = &plan.async_queues["copy"];
    assert_ne!(
        compute_ptr, copy_ptr,
        "each queue type must have its own independent Vec allocation",
    );
}

// =========================================================================
// Additional multi-queue separation tests (T-FG-5.1)
// =========================================================================

#[test]
fn build_async_plan_four_queue_types_strict_separation() {
    // Each queue type gets only its own passes; no cross-contamination.
    let order = vec![
        PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3),
        PassIndex(4), PassIndex(5), PassIndex(6), PassIndex(7),
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(3), "copy".to_string()),
        (PassIndex(5), "transfer".to_string()),
        (PassIndex(7), "video".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(plan.async_queues.len(), 4, "four distinct queue types");
    assert_eq!(plan.async_queues["compute"], vec![PassIndex(0)]);
    assert_eq!(plan.async_queues["copy"],    vec![PassIndex(3)]);
    assert_eq!(plan.async_queues["transfer"], vec![PassIndex(5)]);
    assert_eq!(plan.async_queues["video"],   vec![PassIndex(7)]);

    // Graphics queue contains the remaining non-async passes in order.
    assert_eq!(
        plan.graphics_queue,
        vec![PassIndex(1), PassIndex(2), PassIndex(4), PassIndex(6)],
    );
}

#[test]
fn build_async_plan_separate_queues_no_cross_contamination_on_push() {
    // Pushing to one queue must not affect the contents of another queue.
    let order = vec![
        PassIndex(0), // compute
        PassIndex(1), // copy
        PassIndex(2), // compute
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
        (PassIndex(2), "compute".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(
        plan.async_queues.get("compute"),
        Some(&vec![PassIndex(0), PassIndex(2)]),
        "compute queue holds both compute passes",
    );
    assert_eq!(
        plan.async_queues.get("copy"),
        Some(&vec![PassIndex(1)]),
        "copy queue must NOT contain compute passes",
    );
}

// =========================================================================
// Additional order preservation tests (T-FG-5.1)
// =========================================================================

#[test]
fn build_async_plan_preserves_order_across_three_queue_types() {
    // Verifies order preservation within each queue when three different
    // async queue types are interleaved with graphics passes.
    let order = vec![
        PassIndex(0), // graphics (sync)
        PassIndex(1), // compute
        PassIndex(2), // copy
        PassIndex(3), // transfer
        PassIndex(4), // compute
        PassIndex(5), // copy
        PassIndex(6), // graphics (sync)
        PassIndex(7), // compute
    ];
    let async_passes = vec![
        (PassIndex(1), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
        (PassIndex(3), "transfer".to_string()),
        (PassIndex(4), "compute".to_string()),
        (PassIndex(5), "copy".to_string()),
        (PassIndex(7), "compute".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);

    // Graphics queue: sync passes in original order.
    assert_eq!(
        plan.graphics_queue,
        vec![PassIndex(0), PassIndex(6)],
        "graphics queue preserves sync-pass order",
    );

    // Compute queue: passes 1, 4, 7 in original order.
    assert_eq!(
        plan.async_queues.get("compute").unwrap(),
        &vec![PassIndex(1), PassIndex(4), PassIndex(7)],
        "compute queue preserves original topological order",
    );

    // Copy queue: passes 2, 5 in original order.
    assert_eq!(
        plan.async_queues.get("copy").unwrap(),
        &vec![PassIndex(2), PassIndex(5)],
        "copy queue preserves original topological order",
    );

    // Transfer queue: pass 3 only.
    assert_eq!(
        plan.async_queues.get("transfer").unwrap(),
        &vec![PassIndex(3)],
        "transfer queue has single pass",
    );

    // Total invariant check.
    let total: usize = plan.graphics_queue.len()
        + plan.async_queues.values().map(|q| q.len()).sum::<usize>();
    assert_eq!(total, order.len(), "every input pass appears exactly once");
}

#[test]
fn build_async_plan_order_preserved_when_first_and_last_are_async() {
    // Edge case: the first and last passes in the topological order are
    // async, while middle passes are sync.
    let order = vec![
        PassIndex(0), // async (compute)
        PassIndex(1), // sync
        PassIndex(2), // sync
        PassIndex(3), // async (copy)
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(3), "copy".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(
        plan.graphics_queue,
        vec![PassIndex(1), PassIndex(2)],
        "middle sync passes preserve order",
    );
    assert_eq!(
        plan.async_queues.get("compute").unwrap(),
        &vec![PassIndex(0)],
    );
    assert_eq!(
        plan.async_queues.get("copy").unwrap(),
        &vec![PassIndex(3)],
    );
}

#[test]
fn build_async_plan_order_preserved_reversed_async_types() {
    // Two adjacent async passes of different types: compute then copy.
    let order = vec![
        PassIndex(0), // compute
        PassIndex(1), // copy
        PassIndex(2), // compute
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
        (PassIndex(2), "compute".to_string()),
    ];
    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(
        plan.async_queues.get("compute").unwrap(),
        &vec![PassIndex(0), PassIndex(2)],
        "compute order: 0 before 2, no copy pass leaked in",
    );
    assert_eq!(
        plan.async_queues.get("copy").unwrap(),
        &vec![PassIndex(1)],
        "copy order: single copy pass in its own queue",
    );
}

#[test]
fn build_async_plan_all_async_preserves_original_full_order() {
    // When ALL passes are async on the same queue, the async queue
    // must exactly match the input order.
    let order = vec![
        PassIndex(5), PassIndex(3), PassIndex(1), PassIndex(4), PassIndex(2),
    ];
    let async_passes: Vec<(PassIndex, String)> = order
        .iter()
        .map(|&idx| (idx, "compute".to_string()))
        .collect();
    let plan = build_async_plan(&order, &async_passes);

    assert!(
        plan.graphics_queue.is_empty(),
        "graphics queue empty when all passes are async",
    );
    assert_eq!(
        plan.async_queues.get("compute").unwrap(),
        &order,
        "single async queue exactly matches input order",
    );
}

// =========================================================================
// is_async_pass structural tests (T-FG-5.1)
// =========================================================================

#[test]
fn is_async_pass_first_match_semantics_match_build_async_plan() {
    // is_async_pass uses .any() over async_passes; build_async_plan
    // uses .find(). Both should agree: any() == find().is_some().
    // This test validates the inherent consistency between the two
    // approaches.
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
        (PassIndex(2), "compute".to_string()),
    ];
    for i in 0..3 {
        assert!(
            is_async_pass(PassIndex(i), &async_passes),
            "pass {i} must be async per is_async_pass",
        );
    }
    assert!(!is_async_pass(PassIndex(3), &async_passes));
}

#[test]
fn is_async_pass_empty_string_queue_type() {
    // Edge case: empty queue type string still marks the pass as async.
    let passes = vec![(PassIndex(0), "".to_string())];
    assert!(is_async_pass(PassIndex(0), &passes));
}

#[test]
fn is_async_pass_whitespace_only_queue_type() {
    // Edge case: whitespace-only string is still a valid (non-empty)
    // queue type — the pass is still async.
    let passes = vec![(PassIndex(0), "   ".to_string())];
    assert!(is_async_pass(PassIndex(0), &passes));
}

// =========================================================================
// AsyncExecutionPlan derived trait correctness (T-FG-5.1)
// =========================================================================

#[test]
fn async_execution_plan_debug_contains_queues_when_empty() {
    let plan = AsyncExecutionPlan::default();
    let dbg = format!("{:?}", plan);
    assert!(dbg.contains("graphics_queue"), "debug must show graphics_queue field name");
    assert!(dbg.contains("async_queues"), "debug must show async_queues field name");
}

#[test]
fn async_execution_plan_debug_shows_index_values() {
    let plan = AsyncExecutionPlan {
        graphics_queue: vec![PassIndex(10), PassIndex(20)],
        async_queues: HashMap::from([("compute".into(), vec![PassIndex(30)])]),
    };
    let dbg = format!("{:?}", plan);
    assert!(dbg.contains("PassIndex(10)"), "debug shows PassIndex(10)");
    assert!(dbg.contains("PassIndex(20)"), "debug shows PassIndex(20)");
    assert!(dbg.contains("PassIndex(30)"), "debug shows PassIndex(30)");
    assert!(dbg.contains("compute"), "debug shows queue type label");
}

#[test]
fn async_execution_plan_clone_equality_via_derived() {
    let a = AsyncExecutionPlan {
        graphics_queue: vec![PassIndex(1), PassIndex(2)],
        async_queues: HashMap::from([
            ("compute".into(), vec![PassIndex(3)]),
            ("copy".into(), vec![PassIndex(4), PassIndex(5)]),
        ]),
    };
    let b = a.clone();
    assert_eq!(a, b, "clone must be equal via derived PartialEq");
}

#[test]
fn async_execution_plan_partial_eq_full_mismatch() {
    // Both fields differ.
    let a = AsyncExecutionPlan {
        graphics_queue: vec![PassIndex(0)],
        async_queues: HashMap::new(),
    };
    let b = AsyncExecutionPlan {
        graphics_queue: vec![PassIndex(1)],
        async_queues: HashMap::from([("copy".into(), vec![PassIndex(2)])]),
    };
    assert_ne!(a, b, "fully differing plans must not be equal");
    assert_ne!(b, a, "asymmetry must be symmetric");
}

// =========================================================================
// QueueType -- whitebox tests (T-FG-5.2)
// =========================================================================

#[test]
fn queue_type_variants_are_distinct() {
    assert_ne!(QueueType::Graphics, QueueType::Compute);
    assert_ne!(QueueType::Graphics, QueueType::Copy);
    assert_ne!(QueueType::Compute, QueueType::Copy);
}

#[test]
fn queue_type_clone_produces_equal_copy() {
    let original = QueueType::Compute;
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn queue_type_debug_output() {
    assert_eq!(format!("{:?}", QueueType::Graphics), "Graphics");
    assert_eq!(format!("{:?}", QueueType::Compute), "Compute");
    assert_eq!(format!("{:?}", QueueType::Copy), "Copy");
}

#[test]
fn queue_type_graphics_partial_eq() {
    assert_eq!(QueueType::Graphics, QueueType::Graphics);
    assert_ne!(QueueType::Graphics, QueueType::Compute);
}

#[test]
fn queue_type_copy_partial_eq() {
    assert_eq!(QueueType::Copy, QueueType::Copy);
    assert_ne!(QueueType::Copy, QueueType::Graphics);
}

// =========================================================================
// AsyncTimeline -- whitebox tests (T-FG-5.2)
// =========================================================================

#[test]
fn async_timeline_direct_construction() {
    let timeline = AsyncTimeline {
        queue_type: QueueType::Compute,
        passes: vec![PassIndex(0), PassIndex(1)],
        submit_order: 1,
    };
    assert_eq!(timeline.queue_type, QueueType::Compute);
    assert_eq!(timeline.passes, vec![PassIndex(0), PassIndex(1)]);
    assert_eq!(timeline.submit_order, 1);
}

#[test]
fn async_timeline_graphics_queue_construction() {
    let timeline = AsyncTimeline {
        queue_type: QueueType::Graphics,
        passes: vec![PassIndex(10)],
        submit_order: 0,
    };
    assert_eq!(timeline.queue_type, QueueType::Graphics);
    assert_eq!(timeline.passes, vec![PassIndex(10)]);
}

#[test]
fn async_timeline_copy_queue_construction() {
    let timeline = AsyncTimeline {
        queue_type: QueueType::Copy,
        passes: vec![PassIndex(5), PassIndex(6), PassIndex(7)],
        submit_order: 2,
    };
    assert_eq!(timeline.queue_type, QueueType::Copy);
    assert_eq!(timeline.passes.len(), 3);
    assert_eq!(timeline.submit_order, 2);
}

#[test]
fn async_timeline_empty_passes() {
    let timeline = AsyncTimeline {
        queue_type: QueueType::Compute,
        passes: vec![],
        submit_order: 0,
    };
    assert!(timeline.passes.is_empty());
}

#[test]
fn async_timeline_debug_output() {
    let timeline = AsyncTimeline {
        queue_type: QueueType::Compute,
        passes: vec![PassIndex(0)],
        submit_order: 1,
    };
    let dbg = format!("{:?}", timeline);
    assert!(dbg.contains("AsyncTimeline"));
    assert!(dbg.contains("Compute"));
    assert!(dbg.contains("PassIndex(0)"));
    assert!(dbg.contains("submit_order"));
}

#[test]
fn async_timeline_clone_produces_independent_copy() {
    let timeline = AsyncTimeline {
        queue_type: QueueType::Compute,
        passes: vec![PassIndex(0), PassIndex(1)],
        submit_order: 1,
    };
    let cloned = timeline.clone();
    assert_eq!(timeline, cloned, "clone must be equal to original");
    let mut cloned = cloned;
    cloned.passes.push(PassIndex(99));
    assert_ne!(timeline.passes.len(), cloned.passes.len(),
        "mutating clone must not affect original");
}

#[test]
fn async_timeline_partial_eq_equal() {
    let a = AsyncTimeline {
        queue_type: QueueType::Copy,
        passes: vec![PassIndex(0)],
        submit_order: 1,
    };
    let b = AsyncTimeline {
        queue_type: QueueType::Copy,
        passes: vec![PassIndex(0)],
        submit_order: 1,
    };
    assert_eq!(a, b);
}

#[test]
fn async_timeline_partial_eq_different_queue_type() {
    let a = AsyncTimeline {
        queue_type: QueueType::Compute,
        passes: vec![PassIndex(0)],
        submit_order: 1,
    };
    let b = AsyncTimeline {
        queue_type: QueueType::Graphics,
        passes: vec![PassIndex(0)],
        submit_order: 1,
    };
    assert_ne!(a, b);
}

#[test]
fn async_timeline_partial_eq_different_passes() {
    let a = AsyncTimeline {
        queue_type: QueueType::Compute,
        passes: vec![PassIndex(0)],
        submit_order: 1,
    };
    let b = AsyncTimeline {
        queue_type: QueueType::Compute,
        passes: vec![PassIndex(1)],
        submit_order: 1,
    };
    assert_ne!(a, b);
}

#[test]
fn async_timeline_partial_eq_different_submit_order() {
    let a = AsyncTimeline {
        queue_type: QueueType::Compute,
        passes: vec![PassIndex(0)],
        submit_order: 1,
    };
    let b = AsyncTimeline {
        queue_type: QueueType::Compute,
        passes: vec![PassIndex(0)],
        submit_order: 2,
    };
    assert_ne!(a, b);
}

// =========================================================================
// TimelineBuilder -- whitebox tests (T-FG-5.2)
// =========================================================================

#[test]
fn timeline_builder_new_creates_empty_builder() {
    let builder = TimelineBuilder::new(QueueType::Compute);
    assert_eq!(builder.queue_type, QueueType::Compute);
    assert!(builder.passes.is_empty(), "new builder should have no passes");
}

#[test]
fn timeline_builder_new_graphics_queue() {
    let builder = TimelineBuilder::new(QueueType::Graphics);
    assert_eq!(builder.queue_type, QueueType::Graphics);
    assert!(builder.passes.is_empty());
}

#[test]
fn timeline_builder_new_copy_queue() {
    let builder = TimelineBuilder::new(QueueType::Copy);
    assert_eq!(builder.queue_type, QueueType::Copy);
    assert!(builder.passes.is_empty());
}

#[test]
fn timeline_builder_add_pass_appends_pass() {
    let mut builder = TimelineBuilder::new(QueueType::Compute);
    builder.add_pass(PassIndex(0));
    assert_eq!(builder.passes.len(), 1);
    assert_eq!(builder.passes[0], PassIndex(0));
}

#[test]
fn timeline_builder_add_pass_returns_self_ref() {
    let mut builder = TimelineBuilder::new(QueueType::Compute);
    let ret = builder.add_pass(PassIndex(0));
    // The returned reference points to the same builder.
    assert_eq!(ret.passes.len(), 1);
    assert_eq!(ret.passes[0], PassIndex(0));
}

#[test]
fn timeline_builder_add_pass_multiple_passes() {
    let mut builder = TimelineBuilder::new(QueueType::Copy);
    builder
        .add_pass(PassIndex(0))
        .add_pass(PassIndex(1))
        .add_pass(PassIndex(2));
    assert_eq!(builder.passes.len(), 3);
    assert_eq!(builder.passes, vec![PassIndex(0), PassIndex(1), PassIndex(2)]);
}

#[test]
fn timeline_builder_add_pass_chaining_preserves_queue_type() {
    let mut builder = TimelineBuilder::new(QueueType::Compute);
    builder.add_pass(PassIndex(5)).add_pass(PassIndex(6));
    assert_eq!(builder.queue_type, QueueType::Compute);
}

#[test]
fn timeline_builder_add_pass_out_of_order_indices() {
    let mut builder = TimelineBuilder::new(QueueType::Compute);
    builder
        .add_pass(PassIndex(3))
        .add_pass(PassIndex(1))
        .add_pass(PassIndex(2));
    // Whitebox: builder does not reorder — it preserves insertion order.
    assert_eq!(builder.passes, vec![PassIndex(3), PassIndex(1), PassIndex(2)]);
}

#[test]
fn timeline_builder_finalize_consumes_and_returns_timeline() {
    let mut builder = TimelineBuilder::new(QueueType::Compute);
    builder.add_pass(PassIndex(0)).add_pass(PassIndex(1));
    let timeline = builder.finalize(42);
    assert_eq!(timeline.queue_type, QueueType::Compute);
    assert_eq!(timeline.passes, vec![PassIndex(0), PassIndex(1)]);
    assert_eq!(timeline.submit_order, 42);
}

#[test]
fn timeline_builder_finalize_empty_builder() {
    let builder = TimelineBuilder::new(QueueType::Graphics);
    let timeline = builder.finalize(0);
    assert_eq!(timeline.queue_type, QueueType::Graphics);
    assert!(timeline.passes.is_empty());
    assert_eq!(timeline.submit_order, 0);
}

#[test]
fn timeline_builder_finalize_submit_order_zero() {
    let mut builder = TimelineBuilder::new(QueueType::Copy);
    builder.add_pass(PassIndex(0));
    let timeline = builder.finalize(0);
    assert_eq!(timeline.submit_order, 0);
}

#[test]
fn timeline_builder_finalize_submit_order_max() {
    let mut builder = TimelineBuilder::new(QueueType::Compute);
    builder.add_pass(PassIndex(0));
    let timeline = builder.finalize(u32::MAX);
    assert_eq!(timeline.submit_order, u32::MAX);
}

#[test]
fn timeline_builder_finalize_submit_order_gap() {
    let mut builder = TimelineBuilder::new(QueueType::Graphics);
    builder.add_pass(PassIndex(0));
    let timeline = builder.finalize(100);
    assert_eq!(timeline.submit_order, 100);
}

#[test]
fn timeline_builder_finalize_transfers_pass_ownership() {
    let mut builder = TimelineBuilder::new(QueueType::Compute);
    builder.add_pass(PassIndex(0)).add_pass(PassIndex(1));
    let timeline = builder.finalize(1);
    // Whitebox: finalize moves passes out of the builder via self.
    // After finalize the builder is consumed, so we only verify the timeline.
    assert_eq!(timeline.passes, vec![PassIndex(0), PassIndex(1)]);
    assert_eq!(timeline.queue_type, QueueType::Compute);
    assert_eq!(timeline.submit_order, 1);
}

#[test]
fn timeline_builder_finalize_multiple_builders_different_submit_orders() {
    let mut compute_builder = TimelineBuilder::new(QueueType::Compute);
    compute_builder.add_pass(PassIndex(0));
    let compute_timeline = compute_builder.finalize(1);

    let mut copy_builder = TimelineBuilder::new(QueueType::Copy);
    copy_builder.add_pass(PassIndex(1));
    let copy_timeline = copy_builder.finalize(2);

    assert_eq!(compute_timeline.submit_order, 1);
    assert_eq!(copy_timeline.submit_order, 2);
    assert!(compute_timeline.submit_order < copy_timeline.submit_order);
    assert_eq!(compute_timeline.passes, vec![PassIndex(0)]);
    assert_eq!(copy_timeline.passes, vec![PassIndex(1)]);
}

#[test]
fn timeline_builder_debug_output() {
    let mut builder = TimelineBuilder::new(QueueType::Compute);
    builder.add_pass(PassIndex(0));
    let dbg = format!("{:?}", builder);
    assert!(dbg.contains("TimelineBuilder"));
    assert!(dbg.contains("Compute"));
    assert!(dbg.contains("PassIndex(0)"));
}

#[test]
fn timeline_builder_clone_produces_independent_copy() {
    let mut builder = TimelineBuilder::new(QueueType::Compute);
    builder.add_pass(PassIndex(0));
    let mut cloned = builder.clone();
    cloned.add_pass(PassIndex(99));
    assert_eq!(builder.passes.len(), 1,
        "original passes must be unchanged after clone's mutation");
    assert_eq!(cloned.passes.len(), 2,
        "clone must have its own passes vec");
    assert_eq!(builder.queue_type, cloned.queue_type);
}
