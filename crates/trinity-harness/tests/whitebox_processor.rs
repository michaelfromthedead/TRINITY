//! Whitebox tests for event processor.

use trinity_harness::daemon::{
    process_batch, BatchResult, DaemonEvent, EventProcessor, ProcessorConfig, ProcessResult,
};
use trinity_harness::graph::NodeId;
use trinity_harness::runners::{NodeState, StateTracker};

// ==================== ProcessorConfig ====================

#[test]
fn test_config_default() {
    let config = ProcessorConfig::default();
    assert!(config.propagate_staleness);
    assert_eq!(config.max_propagation_depth, 10);
}

// ==================== EventProcessor ====================

#[test]
fn test_processor_new() {
    let config = ProcessorConfig::default();
    let processor = EventProcessor::new(config);

    assert_eq!(processor.events_processed(), 0);
    assert_eq!(processor.registered_files(), 0);
}

#[test]
fn test_processor_register_file() {
    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);

    processor.register_file("src/lib.rs", NodeId(0));
    processor.register_file("src/lib.rs", NodeId(1));
    processor.register_file("src/main.rs", NodeId(2));

    assert_eq!(processor.registered_files(), 2);
}

#[test]
fn test_processor_register_dependency() {
    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);

    processor.register_dependency(NodeId(0), NodeId(1));
    processor.register_dependency(NodeId(0), NodeId(2));

    assert_eq!(processor.registered_dependencies(), 2);
}

#[test]
fn test_processor_process_file_modified() {
    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);
    let mut tracker = StateTracker::new();

    processor.register_file("src/lib.rs", NodeId(0));
    tracker.set_state(NodeId(0), NodeState::Green);

    let event = DaemonEvent::FileModified {
        path: "src/lib.rs".to_string(),
    };

    let result = processor.process_event(&event, &mut tracker);

    assert!(result.is_some());
    let r = result.unwrap();
    assert_eq!(r.directly_affected, 1);
    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Dirty);
}

#[test]
fn test_processor_process_file_created() {
    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);
    let mut tracker = StateTracker::new();

    processor.register_file("src/new.rs", NodeId(0));

    let event = DaemonEvent::FileCreated {
        path: "src/new.rs".to_string(),
    };

    let result = processor.process_event(&event, &mut tracker);

    assert!(result.is_some());
}

#[test]
fn test_processor_process_file_deleted() {
    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);
    let mut tracker = StateTracker::new();

    processor.register_file("src/old.rs", NodeId(0));
    tracker.set_state(NodeId(0), NodeState::Green);

    let event = DaemonEvent::FileDeleted {
        path: "src/old.rs".to_string(),
    };

    let result = processor.process_event(&event, &mut tracker);

    assert!(result.is_some());
    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Untested);
}

#[test]
fn test_processor_staleness_propagation() {
    let mut config = ProcessorConfig::default();
    config.propagate_staleness = true;

    let mut processor = EventProcessor::new(config);
    let mut tracker = StateTracker::new();

    // Register: node 0 depends on node 1
    processor.register_file("src/lib.rs", NodeId(0));
    processor.register_dependency(NodeId(0), NodeId(1));

    // Both green initially
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.set_state(NodeId(1), NodeState::Green);

    // Modify node 0
    let event = DaemonEvent::FileModified {
        path: "src/lib.rs".to_string(),
    };

    let result = processor.process_event(&event, &mut tracker);

    assert!(result.is_some());
    let r = result.unwrap();
    assert!(r.indirectly_affected >= 1);
    // Node 1 should be dirty too
    assert_eq!(tracker.get_state(NodeId(1)), NodeState::Dirty);
}

#[test]
fn test_processor_no_propagation() {
    let mut config = ProcessorConfig::default();
    config.propagate_staleness = false;

    let mut processor = EventProcessor::new(config);
    let mut tracker = StateTracker::new();

    processor.register_file("src/lib.rs", NodeId(0));
    processor.register_dependency(NodeId(0), NodeId(1));

    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.set_state(NodeId(1), NodeState::Green);

    let event = DaemonEvent::FileModified {
        path: "src/lib.rs".to_string(),
    };

    let result = processor.process_event(&event, &mut tracker);

    assert!(result.is_some());
    // Node 1 should still be green (no propagation)
    assert_eq!(tracker.get_state(NodeId(1)), NodeState::Green);
}

#[test]
fn test_processor_events_count() {
    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);
    let mut tracker = StateTracker::new();

    processor.register_file("src/a.rs", NodeId(0));
    processor.register_file("src/b.rs", NodeId(1));

    processor.process_event(&DaemonEvent::FileModified {
        path: "src/a.rs".to_string(),
    }, &mut tracker);

    processor.process_event(&DaemonEvent::FileModified {
        path: "src/b.rs".to_string(),
    }, &mut tracker);

    assert_eq!(processor.events_processed(), 2);
}

// ==================== ProcessResult ====================

#[test]
fn test_result_new() {
    let result = ProcessResult::new();
    assert_eq!(result.directly_affected, 0);
    assert_eq!(result.indirectly_affected, 0);
    assert!(result.nodes_marked_dirty.is_empty());
}

#[test]
fn test_result_total_affected() {
    let mut result = ProcessResult::new();
    result.directly_affected = 2;
    result.indirectly_affected = 3;

    assert_eq!(result.total_affected(), 5);
}

// ==================== process_batch ====================

#[test]
fn test_process_batch() {
    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);
    let mut tracker = StateTracker::new();

    processor.register_file("src/a.rs", NodeId(0));
    processor.register_file("src/b.rs", NodeId(1));

    let events = vec![
        DaemonEvent::FileModified { path: "src/a.rs".to_string() },
        DaemonEvent::FileModified { path: "src/b.rs".to_string() },
    ];

    let result = process_batch(&mut processor, &events, &mut tracker);

    assert_eq!(result.events_processed, 2);
}

#[test]
fn test_batch_result_new() {
    let result = BatchResult::new();
    assert_eq!(result.events_processed, 0);
    assert_eq!(result.total_affected, 0);
}
