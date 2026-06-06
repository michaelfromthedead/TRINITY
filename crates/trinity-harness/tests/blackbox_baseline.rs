//! Blackbox tests for baseline recording with persistence.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::NodeId;
use trinity_harness::runners::{Baseline, NodeState, StateTracker, record_baseline};

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

#[test]
fn test_baseline_save_load() {
    let dir = create_test_dir();
    let path = dir.path().join("baseline.json");

    let mut baseline = Baseline::new("Test baseline");
    baseline.record_node(NodeId(0), "src/lib.rs", "func_a", NodeState::Green);
    baseline.record_node(NodeId(1), "src/lib.rs", "func_b", NodeState::Red);
    baseline.record_failure("test_b", Some(NodeId(1)), "assertion failed");
    baseline.compute_summary();

    // Save
    baseline.save(&path).expect("Failed to save");

    // Load
    let loaded = Baseline::load(&path).expect("Failed to load");

    assert_eq!(loaded.description, "Test baseline");
    assert_eq!(loaded.node_states.len(), 2);
    assert_eq!(loaded.failures.len(), 1);
    assert_eq!(loaded.summary.green, 1);
    assert_eq!(loaded.summary.red, 1);
}

#[test]
fn test_baseline_persistence_roundtrip() {
    let dir = create_test_dir();
    let path = dir.path().join("baseline.json");

    // Create and populate
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.set_state(NodeId(1), NodeState::Green);
    tracker.set_state(NodeId(2), NodeState::Red);

    let node_info = vec![
        (NodeId(0), "a.rs".to_string(), "func_a".to_string()),
        (NodeId(1), "b.rs".to_string(), "func_b".to_string()),
        (NodeId(2), "c.rs".to_string(), "func_c".to_string()),
    ];

    let baseline = record_baseline(&tracker, "Roundtrip test", &node_info);
    baseline.save(&path).expect("Failed to save");

    // Reload
    let loaded = Baseline::load(&path).expect("Failed to load");

    assert_eq!(loaded.summary.total_nodes, 3);
    assert_eq!(loaded.summary.green, 2);
    assert_eq!(loaded.summary.red, 1);
}

#[test]
fn test_baseline_json_format() {
    let dir = create_test_dir();
    let path = dir.path().join("baseline.json");

    let mut baseline = Baseline::new("JSON test");
    baseline.record_node(NodeId(0), "src/lib.rs", "func", NodeState::Green);
    baseline.compute_summary();

    baseline.save(&path).expect("Failed to save");

    // Read raw JSON
    let json = std::fs::read_to_string(&path).expect("Failed to read");

    assert!(json.contains("\"description\""));
    assert!(json.contains("JSON test"));
    assert!(json.contains("node_states"));
    assert!(json.contains("GREEN"));
}

#[test]
fn test_baseline_multiple_saves() {
    let dir = create_test_dir();

    for i in 0..3 {
        let path = dir.path().join(format!("baseline_{}.json", i));
        let baseline = Baseline::new(format!("Baseline {}", i));
        baseline.save(&path).expect("Failed to save");

        assert!(path.exists());
    }
}

#[test]
fn test_baseline_load_nonexistent() {
    let result = Baseline::load(Path::new("/nonexistent/baseline.json"));
    assert!(result.is_err());
}
