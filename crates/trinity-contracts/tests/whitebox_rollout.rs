//! Whitebox tests for rollout tracking.

use trinity_contracts::rollout::{
    create_initial_tracker, high_value_functions, validate_expansion, AdoptionStatus,
    RolloutStats, RolloutTracker, TrackedFunction, ValidationResult,
};

// ==================== TrackedFunction ====================

#[test]
fn test_tracked_function_new() {
    let func = TrackedFunction::new("test_func", "test_mod");
    assert_eq!(func.name, "test_func");
    assert_eq!(func.module, "test_mod");
    assert_eq!(func.status, AdoptionStatus::Pending);
}

#[test]
fn test_tracked_function_priority() {
    let func = TrackedFunction::new("func", "mod").priority(10);
    assert_eq!(func.priority, 10);
}

#[test]
fn test_tracked_function_status() {
    let func = TrackedFunction::new("func", "mod").status(AdoptionStatus::Validated);
    assert_eq!(func.status, AdoptionStatus::Validated);
}

#[test]
fn test_tracked_function_notes() {
    let func = TrackedFunction::new("func", "mod").notes("test note");
    assert_eq!(func.notes, Some("test note".to_string()));
}

#[test]
fn test_tracked_function_fqn() {
    let func = TrackedFunction::new("func", "module");
    assert_eq!(func.fqn(), "module::func");
}

// ==================== RolloutTracker ====================

#[test]
fn test_tracker_new() {
    let tracker = RolloutTracker::new();
    assert_eq!(tracker.phase(), 0);
}

#[test]
fn test_tracker_track() {
    let mut tracker = RolloutTracker::new();
    tracker.track(TrackedFunction::new("func", "mod"));
    assert!(tracker.get("mod::func").is_some());
}

#[test]
fn test_tracker_update_status() {
    let mut tracker = RolloutTracker::new();
    tracker.track(TrackedFunction::new("func", "mod"));
    assert!(tracker.update_status("mod::func", AdoptionStatus::Validated));
    assert_eq!(tracker.get("mod::func").unwrap().status, AdoptionStatus::Validated);
}

#[test]
fn test_tracker_by_status() {
    let mut tracker = RolloutTracker::new();
    tracker.track(TrackedFunction::new("a", "mod").status(AdoptionStatus::Pending));
    tracker.track(TrackedFunction::new("b", "mod").status(AdoptionStatus::Validated));
    tracker.track(TrackedFunction::new("c", "mod").status(AdoptionStatus::Validated));
    
    assert_eq!(tracker.by_status(AdoptionStatus::Pending).len(), 1);
    assert_eq!(tracker.by_status(AdoptionStatus::Validated).len(), 2);
}

#[test]
fn test_tracker_pending_high_priority() {
    let mut tracker = RolloutTracker::new();
    tracker.track(TrackedFunction::new("low", "mod").priority(3));
    tracker.track(TrackedFunction::new("high", "mod").priority(9));
    tracker.track(TrackedFunction::new("done", "mod").priority(10).status(AdoptionStatus::Validated));
    
    let high = tracker.pending_high_priority(8);
    assert_eq!(high.len(), 1);
    assert_eq!(high[0].name, "high");
}

#[test]
fn test_tracker_set_phase() {
    let mut tracker = RolloutTracker::new();
    tracker.set_phase(2);
    assert_eq!(tracker.phase(), 2);
}

// ==================== RolloutStats ====================

#[test]
fn test_stats_default() {
    let stats = RolloutStats::default();
    assert_eq!(stats.total, 0);
}

#[test]
fn test_stats_adoption_rate() {
    let stats = RolloutStats {
        total: 10,
        pending: 5,
        annotated: 3,
        validated: 2,
        skipped: 0,
    };
    assert!((stats.adoption_rate() - 50.0).abs() < 0.1);
}

#[test]
fn test_stats_validation_rate() {
    let stats = RolloutStats {
        total: 10,
        pending: 5,
        annotated: 3,
        validated: 2,
        skipped: 0,
    };
    assert!((stats.validation_rate() - 20.0).abs() < 0.1);
}

#[test]
fn test_stats_summary() {
    let stats = RolloutStats {
        total: 10,
        pending: 5,
        annotated: 3,
        validated: 2,
        skipped: 0,
    };
    let summary = stats.summary();
    assert!(summary.contains("Total: 10"));
}

// ==================== ValidationResult ====================

#[test]
fn test_validation_result_new() {
    let result = ValidationResult::new();
    assert!(result.is_valid());
    assert!(result.is_clean());
}

#[test]
fn test_validation_result_add_error() {
    let mut result = ValidationResult::new();
    result.add_error("test error");
    assert!(!result.is_valid());
}

#[test]
fn test_validation_result_add_warning() {
    let mut result = ValidationResult::new();
    result.add_warning("test warning");
    assert!(result.is_valid());
    assert!(!result.is_clean());
}

// ==================== high_value_functions ====================

#[test]
fn test_high_value_functions() {
    let funcs = high_value_functions();
    assert_eq!(funcs.len(), 10);
    
    // All should be validated
    for func in &funcs {
        assert_eq!(func.status, AdoptionStatus::Validated);
    }
}

// ==================== create_initial_tracker ====================

#[test]
fn test_create_initial_tracker() {
    let tracker = create_initial_tracker();
    assert_eq!(tracker.phase(), 1);
    
    let stats = tracker.stats();
    assert_eq!(stats.total, 10);
    assert_eq!(stats.validated, 10);
}

// ==================== validate_expansion ====================

#[test]
fn test_validate_expansion_valid() {
    let original = "#[contract]\n#[requires(x > 0)]\nfn test(x: i32) {}";
    let expanded = "fn test(x: i32) { debug_assert!(x > 0); }";
    let result = validate_expansion(original, expanded);
    assert!(result.is_valid());
}

#[test]
fn test_validate_expansion_missing_fn() {
    let original = "#[contract]\nfn test() {}";
    let expanded = "invalid";
    let result = validate_expansion(original, expanded);
    assert!(!result.is_valid());
}
