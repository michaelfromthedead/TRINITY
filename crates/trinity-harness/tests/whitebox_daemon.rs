//! Whitebox tests for HarnessDaemon.

use std::sync::atomic::Ordering;
use trinity_harness::daemon::{DaemonConfig, DaemonEvent, DaemonStatus, HarnessDaemon};
use trinity_harness::graph::NodeId;
use trinity_harness::runners::NodeState;

// ==================== DaemonConfig ====================

#[test]
fn test_config_default() {
    let config = DaemonConfig::default();
    assert_eq!(config.project_root, ".");
    assert_eq!(config.poll_interval_ms, 1000);
    assert_eq!(config.debounce_ms, 100);
    assert!(!config.verbose);
}

#[test]
fn test_config_new() {
    let config = DaemonConfig::new("/project");
    assert_eq!(config.project_root, "/project");
}

#[test]
fn test_config_builder() {
    let config = DaemonConfig::new(".")
        .poll_interval(500)
        .debounce(50)
        .verbose();

    assert_eq!(config.poll_interval_ms, 500);
    assert_eq!(config.debounce_ms, 50);
    assert!(config.verbose);
}

// ==================== HarnessDaemon ====================

#[test]
fn test_daemon_new() {
    let config = DaemonConfig::default();
    let daemon = HarnessDaemon::new(config);

    assert!(!daemon.is_running());
}

#[test]
fn test_daemon_stop_handle() {
    let config = DaemonConfig::default();
    let daemon = HarnessDaemon::new(config);

    let handle = daemon.stop_handle();
    assert!(!handle.load(Ordering::SeqCst));
}

#[test]
fn test_daemon_tracker() {
    let config = DaemonConfig::default();
    let daemon = HarnessDaemon::new(config);

    let tracker = daemon.tracker();
    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Untested);
}

#[test]
fn test_daemon_tracker_mut() {
    let config = DaemonConfig::default();
    let mut daemon = HarnessDaemon::new(config);

    daemon.tracker_mut().set_state(NodeId(0), NodeState::Green);
    assert_eq!(daemon.tracker().get_state(NodeId(0)), NodeState::Green);
}

#[test]
fn test_daemon_handle_state_change() {
    let config = DaemonConfig::default();
    let mut daemon = HarnessDaemon::new(config);

    daemon.handle_state_change(NodeId(0), NodeState::Green);
    assert_eq!(daemon.tracker().get_state(NodeId(0)), NodeState::Green);
}

#[test]
fn test_daemon_mark_dirty() {
    let config = DaemonConfig::default();
    let mut daemon = HarnessDaemon::new(config);

    // First set to green
    daemon.handle_state_change(NodeId(0), NodeState::Green);
    
    // Then mark dirty
    daemon.mark_dirty(&[NodeId(0)]);
    assert_eq!(daemon.tracker().get_state(NodeId(0)), NodeState::Dirty);
}

#[test]
fn test_daemon_needs_testing() {
    let config = DaemonConfig::default();
    let mut daemon = HarnessDaemon::new(config);

    daemon.handle_state_change(NodeId(0), NodeState::Green);
    daemon.handle_state_change(NodeId(1), NodeState::Red);
    daemon.handle_state_change(NodeId(2), NodeState::Untested);
    daemon.handle_state_change(NodeId(3), NodeState::Dirty);

    let all_nodes = vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)];
    let needs_test = daemon.needs_testing(&all_nodes);

    // Green doesn't need testing, others do
    assert_eq!(needs_test.len(), 3);
    assert!(!needs_test.contains(&NodeId(0)));
    assert!(needs_test.contains(&NodeId(1)));
    assert!(needs_test.contains(&NodeId(2)));
    assert!(needs_test.contains(&NodeId(3)));
}

#[test]
fn test_daemon_queue_file_event() {
    let config = DaemonConfig::default();
    let mut daemon = HarnessDaemon::new(config);

    daemon.queue_file_event(DaemonEvent::FileModified {
        path: "src/lib.rs".to_string(),
    });

    // Tick processes events
    daemon.tick();
}

#[test]
fn test_daemon_tick() {
    let config = DaemonConfig::default();
    let mut daemon = HarnessDaemon::new(config);

    daemon.tick();
    // Should not panic
}

#[test]
fn test_daemon_stop() {
    let config = DaemonConfig::default();
    let daemon = HarnessDaemon::new(config);

    daemon.stop();
    assert!(!daemon.is_running());
}

// ==================== DaemonStatus ====================

#[test]
fn test_status_from_daemon() {
    let config = DaemonConfig::default();
    let mut daemon = HarnessDaemon::new(config);

    daemon.handle_state_change(NodeId(0), NodeState::Green);
    daemon.handle_state_change(NodeId(1), NodeState::Red);

    let all_nodes = vec![NodeId(0), NodeId(1)];
    let status = DaemonStatus::from_daemon(&daemon, &all_nodes);

    assert!(!status.running);
    assert_eq!(status.total_nodes, 2);
    assert_eq!(status.needs_testing, 1); // Red needs testing
}

// ==================== DaemonEvent ====================

#[test]
fn test_daemon_event_variants() {
    let events = vec![
        DaemonEvent::FileCreated { path: "a.rs".to_string() },
        DaemonEvent::FileModified { path: "b.rs".to_string() },
        DaemonEvent::FileDeleted { path: "c.rs".to_string() },
        DaemonEvent::StateChanged { 
            node_id: NodeId(0), 
            old: NodeState::Untested, 
            new: NodeState::Green 
        },
        DaemonEvent::Started,
        DaemonEvent::Stopped,
        DaemonEvent::Error { message: "test".to_string() },
    ];

    assert_eq!(events.len(), 7);
}
