//! Notification service for state change events.
//!
//! Provides pub/sub notifications and optional webhook support.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::graph::NodeId;
use crate::runners::NodeState;

use super::DaemonEvent;

/// Type alias for notification handlers.
pub type NotifyHandler = Box<dyn Fn(&Notification) + Send + Sync>;

/// A notification about a state change.
#[derive(Debug, Clone)]
pub struct Notification {
    /// Type of notification.
    pub kind: NotifyKind,
    /// Timestamp (Unix epoch seconds).
    pub timestamp: u64,
    /// Associated message.
    pub message: String,
}

impl Notification {
    /// Create a new notification.
    pub fn new(kind: NotifyKind, message: impl Into<String>) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            kind,
            timestamp,
            message: message.into(),
        }
    }

    /// Create from a daemon event.
    pub fn from_event(event: &DaemonEvent) -> Self {
        match event {
            DaemonEvent::StateChanged { node_id, old, new } => {
                let kind = if matches!(new, NodeState::Red) {
                    NotifyKind::Error
                } else if matches!(old, NodeState::Red) && matches!(new, NodeState::Green) {
                    NotifyKind::Recovery
                } else {
                    NotifyKind::StateChange
                };
                Self::new(
                    kind,
                    format!("Node {} changed from {:?} to {:?}", node_id.0, old, new),
                )
            }
            DaemonEvent::Error { message } => {
                Self::new(NotifyKind::Error, message.clone())
            }
            DaemonEvent::Started => {
                Self::new(NotifyKind::Info, "Daemon started")
            }
            DaemonEvent::Stopped => {
                Self::new(NotifyKind::Info, "Daemon stopped")
            }
            DaemonEvent::FileModified { path } => {
                Self::new(NotifyKind::FileChange, format!("Modified: {}", path))
            }
            DaemonEvent::FileCreated { path } => {
                Self::new(NotifyKind::FileChange, format!("Created: {}", path))
            }
            DaemonEvent::FileDeleted { path } => {
                Self::new(NotifyKind::FileChange, format!("Deleted: {}", path))
            }
        }
    }
}

/// Kind of notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotifyKind {
    /// Informational message.
    Info,
    /// State changed.
    StateChange,
    /// File changed.
    FileChange,
    /// Error occurred.
    Error,
    /// Recovery from error.
    Recovery,
}

/// Notification service with pub/sub.
pub struct NotifyService {
    /// Subscribers by channel.
    subscribers: Arc<Mutex<HashMap<String, Vec<NotifyHandler>>>>,
    /// Notification log.
    log: Arc<Mutex<Vec<Notification>>>,
    /// Maximum log size.
    max_log_size: usize,
    /// Webhook URL (optional).
    webhook_url: Option<String>,
}

impl NotifyService {
    /// Create a new notification service.
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(Mutex::new(HashMap::new())),
            log: Arc::new(Mutex::new(Vec::new())),
            max_log_size: 1000,
            webhook_url: None,
        }
    }

    /// Set the webhook URL.
    pub fn with_webhook(mut self, url: impl Into<String>) -> Self {
        self.webhook_url = Some(url.into());
        self
    }

    /// Set maximum log size.
    pub fn with_max_log(mut self, size: usize) -> Self {
        self.max_log_size = size;
        self
    }

    /// Subscribe to a channel.
    pub fn subscribe(&self, channel: &str, handler: NotifyHandler) {
        let mut subs = self.subscribers.lock().unwrap();
        subs.entry(channel.to_string())
            .or_default()
            .push(handler);
    }

    /// Publish a notification to a channel.
    pub fn publish(&self, channel: &str, notification: &Notification) {
        // Log the notification
        {
            let mut log = self.log.lock().unwrap();
            log.push(notification.clone());
            if log.len() > self.max_log_size {
                log.remove(0);
            }
        }

        // Notify subscribers
        let subs = self.subscribers.lock().unwrap();
        if let Some(handlers) = subs.get(channel) {
            for handler in handlers {
                handler(notification);
            }
        }

        // Webhook (if configured)
        if let Some(ref _url) = self.webhook_url {
            // In production: send HTTP POST to webhook URL
            // For now, just log that we would send it
        }
    }

    /// Publish to all channels.
    pub fn broadcast(&self, notification: &Notification) {
        let channels: Vec<String> = {
            self.subscribers.lock().unwrap().keys().cloned().collect()
        };

        for channel in channels {
            self.publish(&channel, notification);
        }
    }

    /// Get notification log.
    pub fn get_log(&self) -> Vec<Notification> {
        self.log.lock().unwrap().clone()
    }

    /// Get recent notifications.
    pub fn get_recent(&self, count: usize) -> Vec<Notification> {
        let log = self.log.lock().unwrap();
        log.iter().rev().take(count).cloned().collect()
    }

    /// Clear the log.
    pub fn clear_log(&self) {
        self.log.lock().unwrap().clear();
    }

    /// Get subscriber count for a channel.
    pub fn subscriber_count(&self, channel: &str) -> usize {
        self.subscribers
            .lock()
            .unwrap()
            .get(channel)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get all channels.
    pub fn channels(&self) -> Vec<String> {
        self.subscribers.lock().unwrap().keys().cloned().collect()
    }
}

impl Default for NotifyService {
    fn default() -> Self {
        Self::new()
    }
}

/// Transition logger for state changes.
pub struct TransitionLogger {
    /// Logged transitions.
    transitions: Arc<Mutex<Vec<Transition>>>,
    /// Maximum size.
    max_size: usize,
}

/// A state transition.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Node ID.
    pub node_id: NodeId,
    /// Old state.
    pub from: NodeState,
    /// New state.
    pub to: NodeState,
    /// Timestamp.
    pub timestamp: u64,
}

impl TransitionLogger {
    /// Create a new logger.
    pub fn new() -> Self {
        Self {
            transitions: Arc::new(Mutex::new(Vec::new())),
            max_size: 10000,
        }
    }

    /// Log a transition.
    pub fn log(&self, node_id: NodeId, from: NodeState, to: NodeState) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut transitions = self.transitions.lock().unwrap();
        transitions.push(Transition {
            node_id,
            from,
            to,
            timestamp,
        });

        if transitions.len() > self.max_size {
            transitions.remove(0);
        }
    }

    /// Get all transitions.
    pub fn get_all(&self) -> Vec<Transition> {
        self.transitions.lock().unwrap().clone()
    }

    /// Get transitions for a node.
    pub fn get_for_node(&self, node_id: NodeId) -> Vec<Transition> {
        self.transitions
            .lock()
            .unwrap()
            .iter()
            .filter(|t| t.node_id == node_id)
            .cloned()
            .collect()
    }

    /// Get recent transitions.
    pub fn get_recent(&self, count: usize) -> Vec<Transition> {
        let transitions = self.transitions.lock().unwrap();
        transitions.iter().rev().take(count).cloned().collect()
    }

    /// Clear all transitions.
    pub fn clear(&self) {
        self.transitions.lock().unwrap().clear();
    }

    /// Get count.
    pub fn count(&self) -> usize {
        self.transitions.lock().unwrap().len()
    }
}

impl Default for TransitionLogger {
    fn default() -> Self {
        Self::new()
    }
}
