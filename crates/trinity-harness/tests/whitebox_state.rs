//! Whitebox tests for trinity-harness state module.
//!
//! WHITEBOX coverage plan:
//!   - State::new() with String
//!   - State::new() with &str
//!   - State equality and hashing
//!   - Transition::new() creates transition with correct fields
//!   - StateMachine::new() with initial state
//!   - StateMachine::current() returns current state
//!   - add_transition() stores transition
//!   - transition() success path (valid event)
//!   - transition() failure path (invalid event)
//!   - transition() failure path (wrong state)
//!   - Multiple transitions from same state
//!   - Chain of transitions

use std::collections::HashSet;
use trinity_harness::state::{State, StateMachine, Transition};

#[test]
fn test_state_new_with_string() {
    let state = State::new(String::from("idle"));
    assert_eq!(state.name, "idle");
}

#[test]
fn test_state_new_with_str() {
    let state = State::new("running");
    assert_eq!(state.name, "running");
}

#[test]
fn test_state_equality() {
    let s1 = State::new("active");
    let s2 = State::new("active");
    let s3 = State::new("inactive");

    assert_eq!(s1, s2);
    assert_ne!(s1, s3);
}

#[test]
fn test_state_hash() {
    let mut set = HashSet::new();
    set.insert(State::new("a"));
    set.insert(State::new("b"));
    set.insert(State::new("a")); // duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&State::new("a")));
    assert!(set.contains(&State::new("b")));
}

#[test]
fn test_state_clone() {
    let s1 = State::new("original");
    let s2 = s1.clone();

    assert_eq!(s1, s2);
    assert_eq!(s1.name, s2.name);
}

#[test]
fn test_state_debug() {
    let state = State::new("test");
    let debug_str = format!("{:?}", state);
    assert!(debug_str.contains("test"), "Debug should include state name");
}

#[test]
fn test_transition_new_sets_fields() {
    let from = State::new("idle");
    let to = State::new("running");
    let transition = Transition::new(from.clone(), to.clone(), "start");

    assert_eq!(transition.from, from);
    assert_eq!(transition.to, to);
    assert_eq!(transition.event, "start");
}

#[test]
fn test_transition_new_with_string_event() {
    let transition = Transition::new(
        State::new("a"),
        State::new("b"),
        String::from("event_name"),
    );
    assert_eq!(transition.event, "event_name");
}

#[test]
fn test_transition_clone() {
    let transition = Transition::new(State::new("x"), State::new("y"), "go");
    let cloned = transition.clone();

    assert_eq!(transition.from, cloned.from);
    assert_eq!(transition.to, cloned.to);
    assert_eq!(transition.event, cloned.event);
}

#[test]
fn test_state_machine_new_sets_initial_state() {
    let initial = State::new("init");
    let machine = StateMachine::new(initial.clone());

    assert_eq!(machine.current(), &initial);
}

#[test]
fn test_state_machine_current_returns_reference() {
    let machine = StateMachine::new(State::new("start"));
    let current = machine.current();

    assert_eq!(current.name, "start");
}

#[test]
fn test_add_transition_enables_transition() {
    let mut machine = StateMachine::new(State::new("off"));

    machine.add_transition(Transition::new(
        State::new("off"),
        State::new("on"),
        "turn_on",
    ));

    // The transition should now be available
    let success = machine.transition("turn_on");
    assert!(success, "transition should succeed after adding");
    assert_eq!(machine.current().name, "on");
}

#[test]
fn test_transition_success_returns_true() {
    let mut machine = StateMachine::new(State::new("a"));
    machine.add_transition(Transition::new(State::new("a"), State::new("b"), "go"));

    let result = machine.transition("go");
    assert!(result, "valid transition should return true");
}

#[test]
fn test_transition_failure_invalid_event_returns_false() {
    let mut machine = StateMachine::new(State::new("a"));
    machine.add_transition(Transition::new(State::new("a"), State::new("b"), "go"));

    let result = machine.transition("unknown_event");
    assert!(!result, "invalid event should return false");
    assert_eq!(machine.current().name, "a", "state should not change");
}

#[test]
fn test_transition_failure_wrong_state_returns_false() {
    let mut machine = StateMachine::new(State::new("a"));
    machine.add_transition(Transition::new(State::new("b"), State::new("c"), "go"));

    // We're in state "a", but the transition is from "b"
    let result = machine.transition("go");
    assert!(!result, "transition from wrong state should return false");
    assert_eq!(machine.current().name, "a", "state should not change");
}

#[test]
fn test_multiple_transitions_from_same_state() {
    let mut machine = StateMachine::new(State::new("idle"));

    machine.add_transition(Transition::new(
        State::new("idle"),
        State::new("running"),
        "start",
    ));
    machine.add_transition(Transition::new(
        State::new("idle"),
        State::new("error"),
        "fail",
    ));

    // Can transition to running
    assert!(machine.transition("start"));
    assert_eq!(machine.current().name, "running");

    // Reset for next test
    let mut machine2 = StateMachine::new(State::new("idle"));
    machine2.add_transition(Transition::new(
        State::new("idle"),
        State::new("running"),
        "start",
    ));
    machine2.add_transition(Transition::new(
        State::new("idle"),
        State::new("error"),
        "fail",
    ));

    // Can also transition to error
    assert!(machine2.transition("fail"));
    assert_eq!(machine2.current().name, "error");
}

#[test]
fn test_chain_of_transitions() {
    let mut machine = StateMachine::new(State::new("idle"));

    machine.add_transition(Transition::new(
        State::new("idle"),
        State::new("starting"),
        "init",
    ));
    machine.add_transition(Transition::new(
        State::new("starting"),
        State::new("running"),
        "ready",
    ));
    machine.add_transition(Transition::new(
        State::new("running"),
        State::new("stopped"),
        "stop",
    ));

    assert_eq!(machine.current().name, "idle");

    assert!(machine.transition("init"));
    assert_eq!(machine.current().name, "starting");

    assert!(machine.transition("ready"));
    assert_eq!(machine.current().name, "running");

    assert!(machine.transition("stop"));
    assert_eq!(machine.current().name, "stopped");
}

#[test]
fn test_transition_no_effect_when_no_transitions_defined() {
    let mut machine = StateMachine::new(State::new("alone"));

    let result = machine.transition("any_event");
    assert!(!result, "should return false when no transitions defined");
    assert_eq!(machine.current().name, "alone");
}

#[test]
fn test_self_transition() {
    let mut machine = StateMachine::new(State::new("loop"));
    machine.add_transition(Transition::new(
        State::new("loop"),
        State::new("loop"),
        "cycle",
    ));

    assert!(machine.transition("cycle"));
    assert_eq!(machine.current().name, "loop");

    assert!(machine.transition("cycle"));
    assert_eq!(machine.current().name, "loop");
}

#[test]
fn test_transition_updates_current_state() {
    let mut machine = StateMachine::new(State::new("before"));
    machine.add_transition(Transition::new(
        State::new("before"),
        State::new("after"),
        "change",
    ));

    let before = machine.current().clone();
    machine.transition("change");
    let after = machine.current().clone();

    assert_ne!(before, after);
    assert_eq!(before.name, "before");
    assert_eq!(after.name, "after");
}
