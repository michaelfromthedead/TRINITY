//! GPU Resource Leak Detection for wgpu 25.x
//!
//! This module provides comprehensive leak detection capabilities for GPU resources,
//! including severity classification, configurable thresholds, and per-frame tracking.
//!
//! # Overview
//!
//! - **LeakSeverity**: Classify leaks by age (Info, Warning, Critical)
//! - **LeakCandidate**: Individual potential leak with metadata
//! - **LeakThresholds**: Configurable timing thresholds for detection
//! - **AllocationTracker**: Internal tracking of allocations with timestamps
//! - **LeakDetector**: Main interface for leak detection
//! - **LeakStats**: Aggregate statistics about leak detection
//! - **LeakReport**: Full report with candidates and statistics
//! - **FrameLeakChecker**: Per-frame allocation tracking
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::profiling::leaks::{LeakDetector, LeakThresholds};
//!
//! // Create detector with default thresholds
//! let mut detector = LeakDetector::with_default_thresholds();
//!
//! // Track allocations
//! detector.track_allocation(1, "Vertex Buffer", 1024 * 1024);
//! detector.track_allocation(2, "Index Buffer", 256 * 1024);
//!
//! // Mark long-lived resources as expected
//! detector.mark_expected(1);
//!
//! // Check for leaks
//! let candidates = detector.check();
//! for candidate in &candidates {
//!     println!("Potential leak: {:?} ({} bytes, age: {:?})",
//!         candidate.label, candidate.size_bytes, candidate.age());
//! }
//!
//! // Release allocations
//! detector.release_allocation(1);
//! detector.release_allocation(2);
//! ```

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use super::memory::ResourceType;

// ============================================================================
// LeakSeverity
// ============================================================================

/// Severity level for potential memory leaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LeakSeverity {
    /// Informational - allocation exists but may be intentional.
    Info,
    /// Warning - allocation has exceeded warning threshold.
    Warning,
    /// Critical - allocation has exceeded critical threshold.
    Critical,
}

impl LeakSeverity {
    /// Determine severity based on allocation age and thresholds.
    ///
    /// # Arguments
    /// * `age_secs` - Age of the allocation in seconds
    /// * `warning_threshold` - Seconds before warning severity
    /// * `critical_threshold` - Seconds before critical severity
    #[must_use]
    pub fn from_age_secs(age_secs: u64, warning_threshold: u64, critical_threshold: u64) -> Self {
        if age_secs >= critical_threshold {
            LeakSeverity::Critical
        } else if age_secs >= warning_threshold {
            LeakSeverity::Warning
        } else {
            LeakSeverity::Info
        }
    }

    /// Get ANSI color code for terminal display.
    ///
    /// Returns escape sequences for colored output:
    /// - Info: Cyan
    /// - Warning: Yellow
    /// - Critical: Red
    #[must_use]
    pub fn display_color(&self) -> &'static str {
        match self {
            LeakSeverity::Info => "\x1b[36m",     // Cyan
            LeakSeverity::Warning => "\x1b[33m",  // Yellow
            LeakSeverity::Critical => "\x1b[31m", // Red
        }
    }

    /// Get ANSI reset code for terminal display.
    #[must_use]
    pub fn reset_color() -> &'static str {
        "\x1b[0m"
    }

    /// Get display name for this severity level.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            LeakSeverity::Info => "INFO",
            LeakSeverity::Warning => "WARNING",
            LeakSeverity::Critical => "CRITICAL",
        }
    }
}

impl std::fmt::Display for LeakSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ============================================================================
// LeakCandidate
// ============================================================================

/// A potential memory leak candidate with allocation details.
#[derive(Debug, Clone)]
pub struct LeakCandidate {
    /// Unique identifier for this allocation.
    pub allocation_id: u64,
    /// Type of GPU resource.
    pub resource_type: ResourceType,
    /// Size of the allocation in bytes.
    pub size_bytes: u64,
    /// Optional label for debugging.
    pub label: Option<String>,
    /// When this allocation was created.
    pub allocated_at: Instant,
}

impl LeakCandidate {
    /// Create a new leak candidate.
    #[must_use]
    pub fn new(
        allocation_id: u64,
        resource_type: ResourceType,
        size_bytes: u64,
        label: Option<String>,
        allocated_at: Instant,
    ) -> Self {
        Self {
            allocation_id,
            resource_type,
            size_bytes,
            label,
            allocated_at,
        }
    }

    /// Get the age of this allocation.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.allocated_at.elapsed()
    }

    /// Get the age in seconds.
    #[must_use]
    pub fn age_secs(&self) -> u64 {
        self.age().as_secs()
    }

    /// Determine severity based on thresholds.
    #[must_use]
    pub fn severity(&self, thresholds: &LeakThresholds) -> LeakSeverity {
        LeakSeverity::from_age_secs(
            self.age_secs(),
            thresholds.warning_secs,
            thresholds.critical_secs,
        )
    }

    /// Format as a colored string for terminal output.
    #[must_use]
    pub fn format_colored(&self, thresholds: &LeakThresholds) -> String {
        let severity = self.severity(thresholds);
        let label = self.label.as_deref().unwrap_or("<unlabeled>");
        format!(
            "{}[{}]{} {} ({}, {} bytes, age: {:?})",
            severity.display_color(),
            severity,
            LeakSeverity::reset_color(),
            label,
            self.resource_type,
            self.size_bytes,
            self.age()
        )
    }
}

// ============================================================================
// LeakThresholds
// ============================================================================

/// Configurable thresholds for leak detection.
#[derive(Debug, Clone, Copy)]
pub struct LeakThresholds {
    /// Seconds before an allocation is considered warning-level.
    pub warning_secs: u64,
    /// Seconds before an allocation is considered critical.
    pub critical_secs: u64,
    /// Minimum size in bytes to consider for leak detection.
    /// Allocations smaller than this are ignored.
    pub min_size_bytes: u64,
}

impl Default for LeakThresholds {
    fn default() -> Self {
        Self {
            warning_secs: 30,
            critical_secs: 120,
            min_size_bytes: 1024,
        }
    }
}

impl LeakThresholds {
    /// Create thresholds with reasonable defaults.
    ///
    /// - Warning: 30 seconds
    /// - Critical: 120 seconds (2 minutes)
    /// - Min size: 1024 bytes (1 KB)
    #[must_use]
    pub fn default_thresholds() -> Self {
        Self::default()
    }

    /// Create strict thresholds for aggressive leak detection.
    ///
    /// - Warning: 5 seconds
    /// - Critical: 30 seconds
    /// - Min size: 0 bytes (track everything)
    #[must_use]
    pub fn strict() -> Self {
        Self {
            warning_secs: 5,
            critical_secs: 30,
            min_size_bytes: 0,
        }
    }

    /// Create relaxed thresholds for long-lived resources.
    ///
    /// - Warning: 300 seconds (5 minutes)
    /// - Critical: 600 seconds (10 minutes)
    /// - Min size: 4096 bytes (4 KB)
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            warning_secs: 300,
            critical_secs: 600,
            min_size_bytes: 4096,
        }
    }

    /// Create custom thresholds.
    #[must_use]
    pub fn custom(warning_secs: u64, critical_secs: u64, min_size_bytes: u64) -> Self {
        Self {
            warning_secs,
            critical_secs,
            min_size_bytes,
        }
    }
}

// ============================================================================
// AllocationTracker
// ============================================================================

/// Internal tracking structure for allocations.
///
/// Tracks allocation timestamps, labels, sizes, and categorization
/// (expected long-lived vs temporary).
#[derive(Debug, Default)]
pub struct AllocationTracker {
    /// Active allocations: id -> (timestamp, label, size)
    allocations: HashMap<u64, (Instant, String, u64)>,
    /// IDs of allocations expected to be long-lived.
    expected_long_lived: HashSet<u64>,
    /// IDs of allocations marked as temporary (stricter checking).
    temporary: HashSet<u64>,
    /// Resource types for allocations.
    resource_types: HashMap<u64, ResourceType>,
}

impl AllocationTracker {
    /// Create a new allocation tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Track a new allocation.
    pub fn track(&mut self, id: u64, label: impl Into<String>, size: u64) {
        self.allocations.insert(id, (Instant::now(), label.into(), size));
    }

    /// Track a new allocation with resource type.
    pub fn track_with_type(
        &mut self,
        id: u64,
        label: impl Into<String>,
        size: u64,
        resource_type: ResourceType,
    ) {
        self.allocations.insert(id, (Instant::now(), label.into(), size));
        self.resource_types.insert(id, resource_type);
    }

    /// Stop tracking an allocation.
    ///
    /// Returns `true` if the allocation was being tracked.
    pub fn untrack(&mut self, id: u64) -> bool {
        let was_tracked = self.allocations.remove(&id).is_some();
        self.expected_long_lived.remove(&id);
        self.temporary.remove(&id);
        self.resource_types.remove(&id);
        was_tracked
    }

    /// Mark an allocation as expected to be long-lived.
    ///
    /// Long-lived allocations are excluded from leak detection.
    pub fn mark_expected(&mut self, id: u64) {
        if self.allocations.contains_key(&id) {
            self.expected_long_lived.insert(id);
            self.temporary.remove(&id);
        }
    }

    /// Mark an allocation as temporary (stricter checking).
    ///
    /// Temporary allocations should be released quickly.
    pub fn mark_temporary(&mut self, id: u64) {
        if self.allocations.contains_key(&id) {
            self.temporary.insert(id);
            self.expected_long_lived.remove(&id);
        }
    }

    /// Check if an allocation is marked as expected long-lived.
    #[must_use]
    pub fn is_expected(&self, id: u64) -> bool {
        self.expected_long_lived.contains(&id)
    }

    /// Check if an allocation is marked as temporary.
    #[must_use]
    pub fn is_temporary(&self, id: u64) -> bool {
        self.temporary.contains(&id)
    }

    /// Get allocation info by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&(Instant, String, u64)> {
        self.allocations.get(&id)
    }

    /// Get the resource type for an allocation.
    #[must_use]
    pub fn get_resource_type(&self, id: u64) -> ResourceType {
        self.resource_types.get(&id).copied().unwrap_or(ResourceType::Other)
    }

    /// Get the number of tracked allocations.
    #[must_use]
    pub fn count(&self) -> usize {
        self.allocations.len()
    }

    /// Get the number of expected long-lived allocations.
    #[must_use]
    pub fn expected_count(&self) -> usize {
        self.expected_long_lived.len()
    }

    /// Get the number of temporary allocations.
    #[must_use]
    pub fn temporary_count(&self) -> usize {
        self.temporary.len()
    }

    /// Iterate over all allocations.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &(Instant, String, u64))> {
        self.allocations.iter().map(|(k, v)| (*k, v))
    }

    /// Clear all tracked allocations.
    pub fn clear(&mut self) {
        self.allocations.clear();
        self.expected_long_lived.clear();
        self.temporary.clear();
        self.resource_types.clear();
    }

    /// Get total tracked bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.allocations.values().map(|(_, _, size)| *size).sum()
    }
}

// ============================================================================
// LeakStats
// ============================================================================

/// Statistics about leak detection activities.
#[derive(Debug, Clone, Default)]
pub struct LeakStats {
    /// Total allocations ever tracked.
    pub total_tracked: u64,
    /// Total allocations released.
    pub total_released: u64,
    /// Currently tracked allocations.
    pub current_tracked: u64,
    /// Allocations marked as expected long-lived.
    pub expected_long_lived: u64,
    /// Number of leak checks performed.
    pub checks_performed: u64,
    /// Total leaks ever detected.
    pub leaks_detected: u64,
    /// Critical leaks detected.
    pub critical_leaks: u64,
}

impl LeakStats {
    /// Create new empty stats.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the leak rate as a percentage of total tracked.
    #[must_use]
    pub fn leak_rate(&self) -> f64 {
        if self.total_tracked == 0 {
            0.0
        } else {
            (self.leaks_detected as f64 / self.total_tracked as f64) * 100.0
        }
    }

    /// Get the critical leak rate as a percentage of all leaks.
    #[must_use]
    pub fn critical_rate(&self) -> f64 {
        if self.leaks_detected == 0 {
            0.0
        } else {
            (self.critical_leaks as f64 / self.leaks_detected as f64) * 100.0
        }
    }
}

// ============================================================================
// LeakDetector
// ============================================================================

/// Main interface for GPU resource leak detection.
///
/// Tracks allocations, detects potential leaks based on configurable
/// thresholds, and provides statistics and reports.
#[derive(Debug)]
pub struct LeakDetector {
    /// Internal allocation tracker.
    tracker: AllocationTracker,
    /// Detection thresholds.
    thresholds: LeakThresholds,
    /// Number of checks performed.
    check_count: u64,
    /// When the last check was performed.
    last_check: Option<Instant>,
    /// Running statistics.
    stats: LeakStats,
}

impl LeakDetector {
    /// Create a new leak detector with custom thresholds.
    #[must_use]
    pub fn new(thresholds: LeakThresholds) -> Self {
        Self {
            tracker: AllocationTracker::new(),
            thresholds,
            check_count: 0,
            last_check: None,
            stats: LeakStats::new(),
        }
    }

    /// Create a leak detector with default thresholds.
    #[must_use]
    pub fn with_default_thresholds() -> Self {
        Self::new(LeakThresholds::default())
    }

    /// Create a leak detector with strict thresholds.
    #[must_use]
    pub fn with_strict_thresholds() -> Self {
        Self::new(LeakThresholds::strict())
    }

    /// Create a leak detector with relaxed thresholds.
    #[must_use]
    pub fn with_relaxed_thresholds() -> Self {
        Self::new(LeakThresholds::relaxed())
    }

    /// Get the current thresholds.
    #[must_use]
    pub fn thresholds(&self) -> &LeakThresholds {
        &self.thresholds
    }

    /// Set new thresholds.
    pub fn set_thresholds(&mut self, thresholds: LeakThresholds) {
        self.thresholds = thresholds;
    }

    /// Track a new allocation.
    pub fn track_allocation(&mut self, id: u64, label: impl Into<String>, size: u64) {
        self.tracker.track(id, label, size);
        self.stats.total_tracked += 1;
        self.stats.current_tracked = self.tracker.count() as u64;
    }

    /// Track a new allocation with resource type.
    pub fn track_allocation_typed(
        &mut self,
        id: u64,
        label: impl Into<String>,
        size: u64,
        resource_type: ResourceType,
    ) {
        self.tracker.track_with_type(id, label, size, resource_type);
        self.stats.total_tracked += 1;
        self.stats.current_tracked = self.tracker.count() as u64;
    }

    /// Release (untrack) an allocation.
    ///
    /// Returns `true` if the allocation was being tracked.
    pub fn release_allocation(&mut self, id: u64) -> bool {
        let was_tracked = self.tracker.untrack(id);
        if was_tracked {
            self.stats.total_released += 1;
            self.stats.current_tracked = self.tracker.count() as u64;
        }
        was_tracked
    }

    /// Mark an allocation as expected to be long-lived.
    pub fn mark_expected(&mut self, id: u64) {
        self.tracker.mark_expected(id);
        self.stats.expected_long_lived = self.tracker.expected_count() as u64;
    }

    /// Mark an allocation as temporary (stricter checking).
    pub fn mark_temporary(&mut self, id: u64) {
        self.tracker.mark_temporary(id);
    }

    /// Check for potential leaks.
    ///
    /// Returns all allocations that exceed the warning threshold
    /// and are not marked as expected long-lived.
    #[must_use]
    pub fn check(&mut self) -> Vec<LeakCandidate> {
        self.check_count += 1;
        self.last_check = Some(Instant::now());
        self.stats.checks_performed = self.check_count;

        let mut candidates = Vec::new();

        for (id, (allocated_at, label, size)) in self.tracker.iter() {
            // Skip expected long-lived allocations
            if self.tracker.is_expected(id) {
                continue;
            }

            // Skip allocations below minimum size
            if *size < self.thresholds.min_size_bytes {
                continue;
            }

            let age_secs = allocated_at.elapsed().as_secs();

            // For temporary allocations, use stricter thresholds
            let effective_warning = if self.tracker.is_temporary(id) {
                self.thresholds.warning_secs / 2
            } else {
                self.thresholds.warning_secs
            };

            if age_secs >= effective_warning {
                let candidate = LeakCandidate::new(
                    id,
                    self.tracker.get_resource_type(id),
                    *size,
                    Some(label.clone()),
                    *allocated_at,
                );

                // Update stats
                let severity = candidate.severity(&self.thresholds);
                self.stats.leaks_detected += 1;
                if severity == LeakSeverity::Critical {
                    self.stats.critical_leaks += 1;
                }

                candidates.push(candidate);
            }
        }

        candidates
    }

    /// Check for critical leaks only.
    ///
    /// Returns only allocations that exceed the critical threshold.
    #[must_use]
    pub fn check_critical_only(&mut self) -> Vec<LeakCandidate> {
        self.check()
            .into_iter()
            .filter(|c| c.severity(&self.thresholds) == LeakSeverity::Critical)
            .collect()
    }

    /// Get current statistics.
    #[must_use]
    pub fn stats(&self) -> LeakStats {
        LeakStats {
            total_tracked: self.stats.total_tracked,
            total_released: self.stats.total_released,
            current_tracked: self.tracker.count() as u64,
            expected_long_lived: self.tracker.expected_count() as u64,
            checks_performed: self.check_count,
            leaks_detected: self.stats.leaks_detected,
            critical_leaks: self.stats.critical_leaks,
        }
    }

    /// Clear all tracked allocations and reset statistics.
    pub fn clear(&mut self) {
        self.tracker.clear();
        self.check_count = 0;
        self.last_check = None;
        self.stats = LeakStats::new();
    }

    /// Get the number of currently tracked allocations.
    #[must_use]
    pub fn tracked_count(&self) -> usize {
        self.tracker.count()
    }

    /// Get total tracked bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.tracker.total_bytes()
    }

    /// Get the time since the last check.
    #[must_use]
    pub fn time_since_last_check(&self) -> Option<Duration> {
        self.last_check.map(|t| t.elapsed())
    }

    /// Generate a full leak report.
    #[must_use]
    pub fn report(&mut self) -> LeakReport {
        let candidates = self.check();
        LeakReport {
            candidates,
            stats: self.stats(),
            timestamp: Instant::now(),
        }
    }
}

// ============================================================================
// LeakReport
// ============================================================================

/// Full report of leak detection results.
#[derive(Debug)]
pub struct LeakReport {
    /// All leak candidates found.
    pub candidates: Vec<LeakCandidate>,
    /// Statistics at report time.
    pub stats: LeakStats,
    /// When the report was generated.
    pub timestamp: Instant,
}

impl LeakReport {
    /// Check if any critical leaks were found.
    #[must_use]
    pub fn has_critical(&self) -> bool {
        self.stats.critical_leaks > 0
    }

    /// Get total bytes potentially leaked.
    #[must_use]
    pub fn total_leaked_bytes(&self) -> u64 {
        self.candidates.iter().map(|c| c.size_bytes).sum()
    }

    /// Generate a summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        let total_bytes = self.total_leaked_bytes();
        let kb = total_bytes as f64 / 1024.0;

        format!(
            "Leak Report: {} candidates ({:.2} KB total), {} critical\n\
             Stats: {}/{} tracked/released, {} checks",
            self.candidates.len(),
            kb,
            self.stats.critical_leaks,
            self.stats.current_tracked,
            self.stats.total_released,
            self.stats.checks_performed
        )
    }

    /// Categorize candidates by severity.
    ///
    /// Returns (info, warning, critical) vectors.
    pub fn by_severity(
        &self,
        thresholds: &LeakThresholds,
    ) -> (Vec<&LeakCandidate>, Vec<&LeakCandidate>, Vec<&LeakCandidate>) {
        let mut info = Vec::new();
        let mut warning = Vec::new();
        let mut critical = Vec::new();

        for candidate in &self.candidates {
            match candidate.severity(thresholds) {
                LeakSeverity::Info => info.push(candidate),
                LeakSeverity::Warning => warning.push(candidate),
                LeakSeverity::Critical => critical.push(candidate),
            }
        }

        (info, warning, critical)
    }

    /// Get the number of candidates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Check if there are no candidates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

// ============================================================================
// FrameLeakChecker
// ============================================================================

/// Per-frame leak detection for transient allocations.
///
/// Tracks allocations made during a frame and ensures they are
/// released before the frame ends.
#[derive(Debug, Default)]
pub struct FrameLeakChecker {
    /// Allocations made during the current frame.
    frame_allocations: Vec<u64>,
    /// Frame number for debugging.
    frame_number: u64,
}

impl FrameLeakChecker {
    /// Create a new frame leak checker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a new frame.
    ///
    /// Clears the list of frame allocations.
    pub fn begin_frame(&mut self) {
        self.frame_allocations.clear();
        self.frame_number += 1;
    }

    /// Track an allocation made during this frame.
    pub fn track(&mut self, id: u64) {
        self.frame_allocations.push(id);
    }

    /// Mark an allocation as released during this frame.
    pub fn release(&mut self, id: u64) {
        self.frame_allocations.retain(|&x| x != id);
    }

    /// End the frame and check for unreleased allocations.
    ///
    /// Returns the IDs of allocations that were made this frame
    /// but not released.
    #[must_use]
    pub fn end_frame(&mut self) -> Vec<u64> {
        let unreleased = self.frame_allocations.clone();
        self.frame_allocations.clear();
        unreleased
    }

    /// Check if all frame allocations have been released.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.frame_allocations.is_empty()
    }

    /// Get the current frame number.
    #[must_use]
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Get the number of unreleased allocations.
    #[must_use]
    pub fn unreleased_count(&self) -> usize {
        self.frame_allocations.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // LeakSeverity Tests
    // ========================================================================

    #[test]
    fn test_leak_severity_from_age_info() {
        let severity = LeakSeverity::from_age_secs(5, 30, 120);
        assert_eq!(severity, LeakSeverity::Info);
    }

    #[test]
    fn test_leak_severity_from_age_warning() {
        let severity = LeakSeverity::from_age_secs(30, 30, 120);
        assert_eq!(severity, LeakSeverity::Warning);

        let severity = LeakSeverity::from_age_secs(60, 30, 120);
        assert_eq!(severity, LeakSeverity::Warning);
    }

    #[test]
    fn test_leak_severity_from_age_critical() {
        let severity = LeakSeverity::from_age_secs(120, 30, 120);
        assert_eq!(severity, LeakSeverity::Critical);

        let severity = LeakSeverity::from_age_secs(200, 30, 120);
        assert_eq!(severity, LeakSeverity::Critical);
    }

    #[test]
    fn test_leak_severity_display_color() {
        assert_eq!(LeakSeverity::Info.display_color(), "\x1b[36m");
        assert_eq!(LeakSeverity::Warning.display_color(), "\x1b[33m");
        assert_eq!(LeakSeverity::Critical.display_color(), "\x1b[31m");
    }

    #[test]
    fn test_leak_severity_display_name() {
        assert_eq!(LeakSeverity::Info.display_name(), "INFO");
        assert_eq!(LeakSeverity::Warning.display_name(), "WARNING");
        assert_eq!(LeakSeverity::Critical.display_name(), "CRITICAL");
    }

    #[test]
    fn test_leak_severity_ordering() {
        assert!(LeakSeverity::Info < LeakSeverity::Warning);
        assert!(LeakSeverity::Warning < LeakSeverity::Critical);
    }

    // ========================================================================
    // LeakCandidate Tests
    // ========================================================================

    #[test]
    fn test_leak_candidate_new() {
        let now = Instant::now();
        let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, Some("Test".to_string()), now);

        assert_eq!(candidate.allocation_id, 1);
        assert_eq!(candidate.resource_type, ResourceType::Buffer);
        assert_eq!(candidate.size_bytes, 1024);
        assert_eq!(candidate.label, Some("Test".to_string()));
    }

    #[test]
    fn test_leak_candidate_age() {
        let now = Instant::now();
        let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);

        // Age should be very small
        assert!(candidate.age().as_millis() < 100);
    }

    #[test]
    fn test_leak_candidate_severity() {
        let now = Instant::now();
        let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);
        let thresholds = LeakThresholds::default();

        // Fresh allocation should be Info
        assert_eq!(candidate.severity(&thresholds), LeakSeverity::Info);
    }

    // ========================================================================
    // LeakThresholds Tests
    // ========================================================================

    #[test]
    fn test_leak_thresholds_default() {
        let thresholds = LeakThresholds::default();
        assert_eq!(thresholds.warning_secs, 30);
        assert_eq!(thresholds.critical_secs, 120);
        assert_eq!(thresholds.min_size_bytes, 1024);
    }

    #[test]
    fn test_leak_thresholds_strict() {
        let thresholds = LeakThresholds::strict();
        assert_eq!(thresholds.warning_secs, 5);
        assert_eq!(thresholds.critical_secs, 30);
        assert_eq!(thresholds.min_size_bytes, 0);
    }

    #[test]
    fn test_leak_thresholds_relaxed() {
        let thresholds = LeakThresholds::relaxed();
        assert_eq!(thresholds.warning_secs, 300);
        assert_eq!(thresholds.critical_secs, 600);
        assert_eq!(thresholds.min_size_bytes, 4096);
    }

    #[test]
    fn test_leak_thresholds_custom() {
        let thresholds = LeakThresholds::custom(10, 60, 512);
        assert_eq!(thresholds.warning_secs, 10);
        assert_eq!(thresholds.critical_secs, 60);
        assert_eq!(thresholds.min_size_bytes, 512);
    }

    // ========================================================================
    // AllocationTracker Tests
    // ========================================================================

    #[test]
    fn test_allocation_tracker_new() {
        let tracker = AllocationTracker::new();
        assert_eq!(tracker.count(), 0);
    }

    #[test]
    fn test_allocation_tracker_track() {
        let mut tracker = AllocationTracker::new();
        tracker.track(1, "Buffer1", 1024);

        assert_eq!(tracker.count(), 1);
        assert!(tracker.get(1).is_some());
    }

    #[test]
    fn test_allocation_tracker_untrack() {
        let mut tracker = AllocationTracker::new();
        tracker.track(1, "Buffer1", 1024);

        assert!(tracker.untrack(1));
        assert_eq!(tracker.count(), 0);
        assert!(!tracker.untrack(1)); // Already removed
    }

    #[test]
    fn test_allocation_tracker_mark_expected() {
        let mut tracker = AllocationTracker::new();
        tracker.track(1, "LongLived", 1024);
        tracker.mark_expected(1);

        assert!(tracker.is_expected(1));
        assert_eq!(tracker.expected_count(), 1);
    }

    #[test]
    fn test_allocation_tracker_mark_temporary() {
        let mut tracker = AllocationTracker::new();
        tracker.track(1, "Temp", 1024);
        tracker.mark_temporary(1);

        assert!(tracker.is_temporary(1));
        assert_eq!(tracker.temporary_count(), 1);
    }

    #[test]
    fn test_allocation_tracker_clear() {
        let mut tracker = AllocationTracker::new();
        tracker.track(1, "Buffer1", 1024);
        tracker.track(2, "Buffer2", 2048);
        tracker.mark_expected(1);

        tracker.clear();

        assert_eq!(tracker.count(), 0);
        assert_eq!(tracker.expected_count(), 0);
    }

    #[test]
    fn test_allocation_tracker_total_bytes() {
        let mut tracker = AllocationTracker::new();
        tracker.track(1, "Buffer1", 1024);
        tracker.track(2, "Buffer2", 2048);

        assert_eq!(tracker.total_bytes(), 3072);
    }

    // ========================================================================
    // LeakStats Tests
    // ========================================================================

    #[test]
    fn test_leak_stats_new() {
        let stats = LeakStats::new();
        assert_eq!(stats.total_tracked, 0);
        assert_eq!(stats.leaks_detected, 0);
    }

    #[test]
    fn test_leak_stats_leak_rate() {
        let mut stats = LeakStats::new();
        stats.total_tracked = 100;
        stats.leaks_detected = 10;

        assert!((stats.leak_rate() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_leak_stats_critical_rate() {
        let mut stats = LeakStats::new();
        stats.leaks_detected = 20;
        stats.critical_leaks = 5;

        assert!((stats.critical_rate() - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_leak_stats_zero_division() {
        let stats = LeakStats::new();
        assert_eq!(stats.leak_rate(), 0.0);
        assert_eq!(stats.critical_rate(), 0.0);
    }

    // ========================================================================
    // LeakDetector Tests
    // ========================================================================

    #[test]
    fn test_leak_detector_new() {
        let detector = LeakDetector::with_default_thresholds();
        assert_eq!(detector.tracked_count(), 0);
    }

    #[test]
    fn test_leak_detector_track_allocation() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer1", 1024);

        assert_eq!(detector.tracked_count(), 1);
        assert_eq!(detector.total_bytes(), 1024);
    }

    #[test]
    fn test_leak_detector_release_allocation() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer1", 1024);

        assert!(detector.release_allocation(1));
        assert_eq!(detector.tracked_count(), 0);
        assert!(!detector.release_allocation(1));
    }

    #[test]
    fn test_leak_detector_mark_expected() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Static", 1024);
        detector.mark_expected(1);

        let stats = detector.stats();
        assert_eq!(stats.expected_long_lived, 1);
    }

    #[test]
    fn test_leak_detector_check_no_leaks() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer1", 1024);

        // Fresh allocation shouldn't be a leak
        let candidates = detector.check();
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_leak_detector_check_respects_min_size() {
        let thresholds = LeakThresholds::custom(0, 1, 2048); // Immediate warning, min 2KB
        let mut detector = LeakDetector::new(thresholds);
        detector.track_allocation(1, "TinyBuffer", 1024); // Below min size

        let candidates = detector.check();
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_leak_detector_check_respects_expected() {
        let thresholds = LeakThresholds::custom(0, 1, 0); // Immediate detection
        let mut detector = LeakDetector::new(thresholds);
        detector.track_allocation(1, "Static", 1024);
        detector.mark_expected(1);

        let candidates = detector.check();
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_leak_detector_clear() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer1", 1024);
        let _ = detector.check();

        detector.clear();

        assert_eq!(detector.tracked_count(), 0);
        let stats = detector.stats();
        assert_eq!(stats.checks_performed, 0);
    }

    #[test]
    fn test_leak_detector_stats() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer1", 1024);
        detector.track_allocation(2, "Buffer2", 2048);
        detector.release_allocation(1);

        let stats = detector.stats();
        assert_eq!(stats.total_tracked, 2);
        assert_eq!(stats.total_released, 1);
        assert_eq!(stats.current_tracked, 1);
    }

    // ========================================================================
    // LeakReport Tests
    // ========================================================================

    #[test]
    fn test_leak_report_has_critical() {
        let report = LeakReport {
            candidates: vec![],
            stats: LeakStats {
                critical_leaks: 1,
                ..Default::default()
            },
            timestamp: Instant::now(),
        };

        assert!(report.has_critical());
    }

    #[test]
    fn test_leak_report_total_leaked_bytes() {
        let now = Instant::now();
        let report = LeakReport {
            candidates: vec![
                LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now),
                LeakCandidate::new(2, ResourceType::Buffer, 2048, None, now),
            ],
            stats: LeakStats::default(),
            timestamp: now,
        };

        assert_eq!(report.total_leaked_bytes(), 3072);
    }

    #[test]
    fn test_leak_report_summary() {
        let report = LeakReport {
            candidates: vec![],
            stats: LeakStats::default(),
            timestamp: Instant::now(),
        };

        let summary = report.summary();
        assert!(summary.contains("Leak Report"));
    }

    #[test]
    fn test_leak_report_len_and_is_empty() {
        let report = LeakReport {
            candidates: vec![],
            stats: LeakStats::default(),
            timestamp: Instant::now(),
        };

        assert_eq!(report.len(), 0);
        assert!(report.is_empty());
    }

    // ========================================================================
    // FrameLeakChecker Tests
    // ========================================================================

    #[test]
    fn test_frame_leak_checker_new() {
        let checker = FrameLeakChecker::new();
        assert!(checker.is_clean());
        assert_eq!(checker.frame_number(), 0);
    }

    #[test]
    fn test_frame_leak_checker_begin_frame() {
        let mut checker = FrameLeakChecker::new();
        checker.begin_frame();

        assert_eq!(checker.frame_number(), 1);
        assert!(checker.is_clean());
    }

    #[test]
    fn test_frame_leak_checker_track() {
        let mut checker = FrameLeakChecker::new();
        checker.begin_frame();
        checker.track(1);
        checker.track(2);

        assert!(!checker.is_clean());
        assert_eq!(checker.unreleased_count(), 2);
    }

    #[test]
    fn test_frame_leak_checker_release() {
        let mut checker = FrameLeakChecker::new();
        checker.begin_frame();
        checker.track(1);
        checker.track(2);
        checker.release(1);

        assert_eq!(checker.unreleased_count(), 1);
    }

    #[test]
    fn test_frame_leak_checker_end_frame() {
        let mut checker = FrameLeakChecker::new();
        checker.begin_frame();
        checker.track(1);
        checker.track(2);
        checker.release(1);

        let unreleased = checker.end_frame();

        assert_eq!(unreleased, vec![2]);
        assert!(checker.is_clean());
    }

    #[test]
    fn test_frame_leak_checker_clean_frame() {
        let mut checker = FrameLeakChecker::new();
        checker.begin_frame();
        checker.track(1);
        checker.release(1);

        let unreleased = checker.end_frame();

        assert!(unreleased.is_empty());
        assert!(checker.is_clean());
    }

    #[test]
    fn test_frame_leak_checker_multiple_frames() {
        let mut checker = FrameLeakChecker::new();

        for i in 0..3 {
            checker.begin_frame();
            assert_eq!(checker.frame_number(), i + 1);
            checker.track(i);
            checker.release(i);
            let unreleased = checker.end_frame();
            assert!(unreleased.is_empty());
        }

        assert_eq!(checker.frame_number(), 3);
    }
}
