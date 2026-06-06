//! Whitebox tests for notification service.

use std::sync::{Arc, Mutex};
use trinity_harness::daemon::{
    DaemonEvent, Notification, NotifyKind, NotifyService, Transition, TransitionLogger,
};
use trinity_harness::graph::NodeId;
use trinity_harness::runners::NodeState;

// ==================== Notification ====================

#[test]
fn test_notification_new() {
    let notif = Notification::new(NotifyKind::Info, "test message");
    assert_eq!(notif.kind, NotifyKind::Info);
    assert_eq!(notif.message, "test message");
    assert!(notif.timestamp > 0);
}

#[test]
fn test_notification_from_state_change() {
    let event = DaemonEvent::StateChanged {
        node_id: NodeId(0),
        old: NodeState::Untested,
        new: NodeState::Green,
    };
    let notif = Notification::from_event(&event);
    
    assert_eq!(notif.kind, NotifyKind::StateChange);
    assert!(notif.message.contains("Node 0"));
}

#[test]
fn test_notification_from_error() {
    let event = DaemonEvent::Error {
        message: "test error".to_string(),
    };
    let notif = Notification::from_event(&event);
    
    assert_eq!(notif.kind, NotifyKind::Error);
    assert!(notif.message.contains("test error"));
}

#[test]
fn test_notification_from_file_change() {
    let event = DaemonEvent::FileModified {
        path: "src/lib.rs".to_string(),
    };
    let notif = Notification::from_event(&event);
    
    assert_eq!(notif.kind, NotifyKind::FileChange);
    assert!(notif.message.contains("src/lib.rs"));
}

// ==================== NotifyKind ====================

#[test]
fn test_notify_kind_variants() {
    let kinds = vec![
        NotifyKind::Info,
        NotifyKind::StateChange,
        NotifyKind::FileChange,
        NotifyKind::Error,
        NotifyKind::Recovery,
    ];
    assert_eq!(kinds.len(), 5);
}

// ==================== NotifyService ====================

#[test]
fn test_service_new() {
    let service = NotifyService::new();
    assert!(service.channels().is_empty());
    assert!(service.get_log().is_empty());
}

#[test]
fn test_service_subscribe() {
    let service = NotifyService::new();
    
    service.subscribe("test", Box::new(|_| {}));
    
    assert_eq!(service.subscriber_count("test"), 1);
}

#[test]
fn test_service_publish() {
    let service = NotifyService::new();
    let received = Arc::new(Mutex::new(Vec::new()));
    let received_clone = received.clone();
    
    service.subscribe("test", Box::new(move |n| {
        received_clone.lock().unwrap().push(n.message.clone());
    }));
    
    let notif = Notification::new(NotifyKind::Info, "hello");
    service.publish("test", &notif);
    
    let msgs = received.lock().unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0], "hello");
}

#[test]
fn test_service_log() {
    let service = NotifyService::new();
    
    service.publish("test", &Notification::new(NotifyKind::Info, "msg1"));
    service.publish("test", &Notification::new(NotifyKind::Info, "msg2"));
    
    let log = service.get_log();
    assert_eq!(log.len(), 2);
}

#[test]
fn test_service_get_recent() {
    let service = NotifyService::new();
    
    for i in 0..5 {
        service.publish("test", &Notification::new(NotifyKind::Info, format!("msg{}", i)));
    }
    
    let recent = service.get_recent(3);
    assert_eq!(recent.len(), 3);
}

#[test]
fn test_service_clear_log() {
    let service = NotifyService::new();
    
    service.publish("test", &Notification::new(NotifyKind::Info, "msg"));
    service.clear_log();
    
    assert!(service.get_log().is_empty());
}

#[test]
fn test_service_with_webhook() {
    let service = NotifyService::new()
        .with_webhook("http://example.com/webhook");
    
    // Just verify it builds
    service.publish("test", &Notification::new(NotifyKind::Info, "msg"));
}

// ==================== TransitionLogger ====================

#[test]
fn test_logger_new() {
    let logger = TransitionLogger::new();
    assert_eq!(logger.count(), 0);
}

#[test]
fn test_logger_log() {
    let logger = TransitionLogger::new();
    
    logger.log(NodeId(0), NodeState::Untested, NodeState::Green);
    
    assert_eq!(logger.count(), 1);
}

#[test]
fn test_logger_get_all() {
    let logger = TransitionLogger::new();
    
    logger.log(NodeId(0), NodeState::Untested, NodeState::Green);
    logger.log(NodeId(1), NodeState::Green, NodeState::Red);
    
    let all = logger.get_all();
    assert_eq!(all.len(), 2);
}

#[test]
fn test_logger_get_for_node() {
    let logger = TransitionLogger::new();
    
    logger.log(NodeId(0), NodeState::Untested, NodeState::Green);
    logger.log(NodeId(1), NodeState::Green, NodeState::Red);
    logger.log(NodeId(0), NodeState::Green, NodeState::Dirty);
    
    let node0 = logger.get_for_node(NodeId(0));
    assert_eq!(node0.len(), 2);
}

#[test]
fn test_logger_get_recent() {
    let logger = TransitionLogger::new();
    
    for i in 0..5 {
        logger.log(NodeId(i), NodeState::Untested, NodeState::Green);
    }
    
    let recent = logger.get_recent(3);
    assert_eq!(recent.len(), 3);
}

#[test]
fn test_logger_clear() {
    let logger = TransitionLogger::new();
    
    logger.log(NodeId(0), NodeState::Untested, NodeState::Green);
    logger.clear();
    
    assert_eq!(logger.count(), 0);
}

// ==================== Transition ====================

#[test]
fn test_transition_fields() {
    let logger = TransitionLogger::new();
    logger.log(NodeId(42), NodeState::Green, NodeState::Red);
    
    let transitions = logger.get_all();
    let t = &transitions[0];
    
    assert_eq!(t.node_id, NodeId(42));
    assert_eq!(t.from, NodeState::Green);
    assert_eq!(t.to, NodeState::Red);
    assert!(t.timestamp > 0);
}
