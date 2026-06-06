//! Blackbox tests for notification service with daemon integration.

use std::sync::{Arc, Mutex};
use trinity_harness::daemon::{
    DaemonEvent, Notification, NotifyKind, NotifyService, TransitionLogger,
};
use trinity_harness::graph::NodeId;
use trinity_harness::runners::NodeState;

#[test]
fn test_full_notification_workflow() {
    let service = NotifyService::new();
    let messages: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let messages_clone = messages.clone();
    
    // Subscribe to state changes
    service.subscribe("state", Box::new(move |n| {
        messages_clone.lock().unwrap().push(n.message.clone());
    }));
    
    // Simulate state changes
    let events = vec![
        DaemonEvent::StateChanged {
            node_id: NodeId(0),
            old: NodeState::Untested,
            new: NodeState::Green,
        },
        DaemonEvent::StateChanged {
            node_id: NodeId(1),
            old: NodeState::Untested,
            new: NodeState::Red,
        },
        DaemonEvent::StateChanged {
            node_id: NodeId(1),
            old: NodeState::Red,
            new: NodeState::Green,
        },
    ];
    
    for event in &events {
        let notif = Notification::from_event(event);
        service.publish("state", &notif);
    }
    
    let msgs = messages.lock().unwrap();
    assert_eq!(msgs.len(), 3);
}

#[test]
fn test_multiple_channels() {
    let service = NotifyService::new();
    
    let state_count = Arc::new(Mutex::new(0));
    let file_count = Arc::new(Mutex::new(0));
    
    let state_clone = state_count.clone();
    let file_clone = file_count.clone();
    
    service.subscribe("state", Box::new(move |_| {
        *state_clone.lock().unwrap() += 1;
    }));
    
    service.subscribe("files", Box::new(move |_| {
        *file_clone.lock().unwrap() += 1;
    }));
    
    // Publish to different channels
    service.publish("state", &Notification::new(NotifyKind::StateChange, "state1"));
    service.publish("state", &Notification::new(NotifyKind::StateChange, "state2"));
    service.publish("files", &Notification::new(NotifyKind::FileChange, "file1"));
    
    assert_eq!(*state_count.lock().unwrap(), 2);
    assert_eq!(*file_count.lock().unwrap(), 1);
}

#[test]
fn test_transition_logging_workflow() {
    let logger = TransitionLogger::new();
    
    // Simulate a node going through test cycle
    let node = NodeId(42);
    
    // Initial state
    logger.log(node, NodeState::Untested, NodeState::Dirty);
    
    // Tests run
    logger.log(node, NodeState::Dirty, NodeState::Green);
    
    // Code modified
    logger.log(node, NodeState::Green, NodeState::Dirty);
    
    // Tests fail
    logger.log(node, NodeState::Dirty, NodeState::Red);
    
    // Tests fixed
    logger.log(node, NodeState::Red, NodeState::Green);
    
    let history = logger.get_for_node(node);
    assert_eq!(history.len(), 5);
    
    // Verify final state
    assert_eq!(history.last().unwrap().to, NodeState::Green);
}

#[test]
fn test_notification_with_logger() {
    let service = NotifyService::new();
    let logger = TransitionLogger::new();
    
    let logger_clone = Arc::new(Mutex::new(Vec::new()));
    let logger_ref = logger_clone.clone();
    
    service.subscribe("state", Box::new(move |n| {
        logger_ref.lock().unwrap().push(n.message.clone());
    }));
    
    // Log and notify together
    let node = NodeId(0);
    let old_state = NodeState::Green;
    let new_state = NodeState::Red;
    
    logger.log(node, old_state, new_state);
    
    let event = DaemonEvent::StateChanged {
        node_id: node,
        old: old_state,
        new: new_state,
    };
    service.publish("state", &Notification::from_event(&event));
    
    assert_eq!(logger.count(), 1);
    assert_eq!(logger_clone.lock().unwrap().len(), 1);
}

#[test]
fn test_recovery_detection() {
    let events = vec![
        (NodeState::Green, NodeState::Red, NotifyKind::Error),
        (NodeState::Red, NodeState::Green, NotifyKind::Recovery),
        (NodeState::Green, NodeState::Dirty, NotifyKind::StateChange),
    ];
    
    for (from, to, expected) in events {
        let event = DaemonEvent::StateChanged {
            node_id: NodeId(0),
            old: from,
            new: to,
        };
        let notif = Notification::from_event(&event);
        assert_eq!(notif.kind, expected);
    }
}
