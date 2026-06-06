//! Blackbox tests for rollout tracking with real scenarios.

use trinity_contracts::rollout::{
    create_initial_tracker, validate_expansion, AdoptionStatus,
    RolloutTracker, TrackedFunction,
};

#[test]
fn test_full_rollout_workflow() {
    let mut tracker = RolloutTracker::new();
    tracker.set_phase(1);

    // Phase 1: Add high-priority functions
    let high_value = vec![
        TrackedFunction::new("safe_div", "math").priority(10),
        TrackedFunction::new("safe_sqrt", "math").priority(9),
        TrackedFunction::new("alloc", "memory").priority(10),
    ];

    for func in high_value {
        tracker.track(func);
    }

    // Check initial state
    let stats = tracker.stats();
    assert_eq!(stats.total, 3);
    assert_eq!(stats.pending, 3);

    // Annotate functions
    tracker.update_status("math::safe_div", AdoptionStatus::Annotated);
    tracker.update_status("math::safe_sqrt", AdoptionStatus::Annotated);

    let stats = tracker.stats();
    assert_eq!(stats.annotated, 2);

    // Validate functions
    tracker.update_status("math::safe_div", AdoptionStatus::Validated);

    let stats = tracker.stats();
    assert_eq!(stats.validated, 1);
    assert!(stats.adoption_rate() > 60.0); // 2/3 adopted
}

#[test]
fn test_phased_rollout() {
    let mut tracker = RolloutTracker::new();

    // Phase 1: Core functions
    tracker.set_phase(1);
    tracker.track(TrackedFunction::new("core_fn", "core").priority(10));
    
    // Phase 2: Extended functions
    tracker.set_phase(2);
    tracker.track(TrackedFunction::new("ext_fn", "ext").priority(5));

    assert_eq!(tracker.phase(), 2);
    assert_eq!(tracker.stats().total, 2);
}

#[test]
fn test_skip_unsuitable_functions() {
    let mut tracker = RolloutTracker::new();

    tracker.track(
        TrackedFunction::new("macro_fn", "macros")
            .status(AdoptionStatus::Skipped)
            .notes("Macros don't support contracts"),
    );

    let stats = tracker.stats();
    assert_eq!(stats.skipped, 1);
    
    // Skipped functions shouldn't count toward adoption
    assert_eq!(stats.adoption_rate(), 0.0);
}

#[test]
fn test_priority_based_selection() {
    let mut tracker = RolloutTracker::new();

    tracker.track(TrackedFunction::new("low1", "mod").priority(2));
    tracker.track(TrackedFunction::new("low2", "mod").priority(3));
    tracker.track(TrackedFunction::new("medium", "mod").priority(5));
    tracker.track(TrackedFunction::new("high1", "mod").priority(8));
    tracker.track(TrackedFunction::new("high2", "mod").priority(9));
    tracker.track(TrackedFunction::new("critical", "mod").priority(10));

    // Get high priority only
    let high = tracker.pending_high_priority(8);
    assert_eq!(high.len(), 3);

    // Get critical only
    let critical = tracker.pending_high_priority(10);
    assert_eq!(critical.len(), 1);
}

#[test]
fn test_initial_tracker_is_complete() {
    let tracker = create_initial_tracker();
    let stats = tracker.stats();

    // All 10 high-value functions should be validated
    assert_eq!(stats.total, 10);
    assert_eq!(stats.validated, 10);
    assert_eq!(stats.pending, 0);
    assert!((stats.validation_rate() - 100.0).abs() < 0.1);
}

#[test]
fn test_validate_contract_expansion() {
    // Valid expansion with requires
    let original = r#"
        #[contract]
        #[requires(x > 0)]
        fn positive(x: i32) -> i32 { x }
    "#;
    let expanded = r#"
        fn positive(x: i32) -> i32 {
            debug_assert!(x > 0);
            x
        }
    "#;
    let result = validate_expansion(original, expanded);
    assert!(result.is_valid());

    // Valid expansion with ensures
    let original_ensures = r#"
        #[contract]
        #[ensures(result > 0)]
        fn double(x: i32) -> i32 { x * 2 }
    "#;
    let expanded_ensures = r#"
        fn double(x: i32) -> i32 {
            let __contract_result = x * 2;
            debug_assert!(__contract_result > 0);
            __contract_result
        }
    "#;
    let result = validate_expansion(original_ensures, expanded_ensures);
    assert!(result.is_valid());
}

#[test]
fn test_stats_formatting() {
    let tracker = create_initial_tracker();
    let stats = tracker.stats();
    let summary = stats.summary();

    assert!(summary.contains("Total: 10"));
    assert!(summary.contains("Validated: 10"));
    assert!(summary.contains("100.0% validated"));
}
