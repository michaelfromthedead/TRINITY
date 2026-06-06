//! Blackbox tests for HarnessDaemon.

use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;
use trinity_harness::daemon::{DaemonConfig, DaemonEvent, HarnessDaemon};
use trinity_harness::graph::NodeId;
use trinity_harness::runners::NodeState;

#[test]
fn test_daemon_with_callbacks() {
    let config = DaemonConfig::default();
    let mut daemon = HarnessDaemon::new(config);

    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();

    daemon.on_event(Box::new(move |event| {
        let msg = match event {
            DaemonEvent::StateChanged { node_id, old, new } => {
                format!("state:{}:{:?}->{:?}", node_id.0, old, new)
            }
            _ => format!("{:?}", event),
        };
        events_clone.lock().unwrap().push(msg);
    }));

    daemon.handle_state_change(NodeId(0), NodeState::Green);

    let captured = events.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(captured[0].contains("state:0"));
}

#[test]
fn test_daemon_workflow() {
    let config = DaemonConfig::new(".").verbose();
    let mut daemon = HarnessDaemon::new(config);

    // Initial state
    assert!(!daemon.is_running());

    // Set up some nodes
    let nodes = vec![NodeId(0), NodeId(1), NodeId(2)];
    for &id in &nodes {
        daemon.handle_state_change(id, NodeState::Untested);
    }

    // All need testing initially
    assert_eq!(daemon.needs_testing(&nodes).len(), 3);

    // Simulate tests passing
    daemon.handle_state_change(NodeId(0), NodeState::Green);
    daemon.handle_state_change(NodeId(1), NodeState::Green);

    // One still needs testing
    let needs_test = daemon.needs_testing(&nodes);
    assert_eq!(needs_test.len(), 1);
    assert!(needs_test.contains(&NodeId(2)));

    // File modified - mark dirty
    daemon.queue_file_event(DaemonEvent::FileModified {
        path: "src/node_0.rs".to_string(),
    });
    daemon.mark_dirty(&[NodeId(0)]);

    // Now two need testing
    let needs_test = daemon.needs_testing(&nodes);
    assert_eq!(needs_test.len(), 2);
}

#[test]
fn test_daemon_stop_via_handle() {
    let config = DaemonConfig::default();
    let daemon = HarnessDaemon::new(config);

    let handle = daemon.stop_handle();

    // Simulate external stop
    handle.store(false, Ordering::SeqCst);

    // Daemon should not be running
    assert!(!daemon.is_running());
}

#[test]
fn test_daemon_process_events() {
    let config = DaemonConfig::default();
    let mut daemon = HarnessDaemon::new(config);

    // Queue multiple events
    for i in 0..5 {
        daemon.queue_file_event(DaemonEvent::FileModified {
            path: format!("file_{}.rs", i),
        });
    }

    // Process them
    daemon.tick();

    // Events should be processed
}

#[test]
fn test_daemon_status_reflects_state() {
    let config = DaemonConfig::default();
    let mut daemon = HarnessDaemon::new(config);

    let nodes = vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(4)];

    // Set states
    daemon.handle_state_change(NodeId(0), NodeState::Green);
    daemon.handle_state_change(NodeId(1), NodeState::Green);
    daemon.handle_state_change(NodeId(2), NodeState::Red);
    daemon.handle_state_change(NodeId(3), NodeState::Dirty);
    // NodeId(4) remains Untested

    let status = trinity_harness::daemon::DaemonStatus::from_daemon(&daemon, &nodes);

    assert!(!status.running);
    assert_eq!(status.total_nodes, 5);
    assert_eq!(status.needs_testing, 3); // Red, Dirty, Untested
}
