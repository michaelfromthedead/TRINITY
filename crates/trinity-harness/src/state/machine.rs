//! State machine implementation for workflow management.

use std::collections::HashMap;

/// A state in the state machine.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct State {
    pub name: String,
}

impl State {
    /// Create a new state.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// A transition between states.
#[derive(Debug, Clone)]
pub struct Transition {
    pub from: State,
    pub to: State,
    pub event: String,
}

impl Transition {
    /// Create a new transition.
    pub fn new(from: State, to: State, event: impl Into<String>) -> Self {
        Self {
            from,
            to,
            event: event.into(),
        }
    }
}

/// A simple state machine.
pub struct StateMachine {
    current: State,
    transitions: HashMap<(State, String), State>,
}

impl StateMachine {
    /// Create a new state machine with an initial state.
    pub fn new(initial: State) -> Self {
        Self {
            current: initial,
            transitions: HashMap::new(),
        }
    }

    /// Add a transition to the state machine.
    pub fn add_transition(&mut self, transition: Transition) {
        self.transitions
            .insert((transition.from, transition.event), transition.to);
    }

    /// Get the current state.
    pub fn current(&self) -> &State {
        &self.current
    }

    /// Attempt to transition on an event.
    pub fn transition(&mut self, event: &str) -> bool {
        let key = (self.current.clone(), event.to_string());
        if let Some(next) = self.transitions.get(&key) {
            self.current = next.clone();
            true
        } else {
            false
        }
    }
}
