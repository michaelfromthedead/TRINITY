//! Resource Leak Detection for GPU Resources (T-WGPU-P7.4.3).
//!
//! This module provides leak detection capabilities for GPU resources,
//! with RAII-based tracking and detailed reporting.
//!
//! # Overview
//!
//! - [`ResourceType`]: Types of GPU resources that can be tracked
//! - [`TrackedResource`]: Information about a tracked resource
//! - [`LeakReport`]: Report of detected leaks
//! - [`LeakDetector`]: Main interface for leak detection
//! - [`LeakDetectorStats`]: Statistics about the detector
//! - [`LeakScope`]: RAII guard for automatic resource tracking
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::profiling::leak_detector::{LeakDetector, ResourceType, LeakScope};
//!
//! # fn example() {
//! let mut detector = LeakDetector::new();
//!
//! // Track resources manually
//! detector.track(1, ResourceType::Buffer, "Vertex Buffer", 1024 * 1024);
//! detector.track(2, ResourceType::Texture, "Diffuse Map", 4 * 1024 * 1024);
//!
//! // Or use RAII scope
//! {
//!     let _scope = LeakScope::new(&mut detector, 3, ResourceType::Buffer, "Temp Buffer", 4096);
//!     // ... use buffer ...
//! } // Automatically untracked
//!
//! // Check for leaks
//! let report = detector.check_leaks();
//! if report.has_leaks() {
//!     println!("Detected {} potential leaks!", report.leak_count());
//! }
//!
//! // Untrack when done
//! detector.untrack(1);
//! detector.untrack(2);
//! # }
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// ResourceType
// ============================================================================

/// Types of GPU resources that can be tracked for leaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// GPU buffer (vertex, index, uniform, storage).
    Buffer,
    /// Texture resource.
    Texture,
    /// Texture view.
    TextureView,
    /// Sampler.
    Sampler,
    /// Bind group.
    BindGroup,
    /// Bind group layout.
    BindGroupLayout,
    /// Pipeline layout.
    PipelineLayout,
    /// Render pipeline.
    RenderPipeline,
    /// Compute pipeline.
    ComputePipeline,
    /// Shader module.
    ShaderModule,
    /// Query set.
    QuerySet,
    /// Command buffer.
    CommandBuffer,
    /// Surface.
    Surface,
    /// Unknown or custom resource.
    Unknown,
}

impl ResourceType {
    /// Get the name of this resource type.
    pub fn name(&self) -> &'static str {
        match self {
            ResourceType::Buffer => "Buffer",
            ResourceType::Texture => "Texture",
            ResourceType::TextureView => "TextureView",
            ResourceType::Sampler => "Sampler",
            ResourceType::BindGroup => "BindGroup",
            ResourceType::BindGroupLayout => "BindGroupLayout",
            ResourceType::PipelineLayout => "PipelineLayout",
            ResourceType::RenderPipeline => "RenderPipeline",
            ResourceType::ComputePipeline => "ComputePipeline",
            ResourceType::ShaderModule => "ShaderModule",
            ResourceType::QuerySet => "QuerySet",
            ResourceType::CommandBuffer => "CommandBuffer",
            ResourceType::Surface => "Surface",
            ResourceType::Unknown => "Unknown",
        }
    }

    /// Check if this is a long-lived resource type.
    ///
    /// Returns true for pipelines, layouts, and other resources
    /// that are typically kept for the lifetime of the application.
    pub fn is_long_lived(&self) -> bool {
        matches!(
            self,
            ResourceType::RenderPipeline
                | ResourceType::ComputePipeline
                | ResourceType::PipelineLayout
                | ResourceType::BindGroupLayout
                | ResourceType::ShaderModule
        )
    }
}

impl std::fmt::Display for ResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for ResourceType {
    fn default() -> Self {
        ResourceType::Unknown
    }
}

// ============================================================================
// TrackedResource
// ============================================================================

/// Information about a tracked resource.
#[derive(Debug, Clone)]
pub struct TrackedResource {
    /// Unique ID of the resource.
    pub id: u64,
    /// Type of the resource.
    pub resource_type: ResourceType,
    /// Optional label/name for debugging.
    pub label: Option<String>,
    /// Size in bytes (if known).
    pub size_bytes: u64,
    /// Time when tracking started.
    pub created_at: Instant,
    /// Stack trace or creation context (optional).
    pub creation_context: Option<String>,
    /// Whether this resource is expected to be long-lived.
    pub expected_long_lived: bool,
}

impl TrackedResource {
    /// Create a new tracked resource.
    pub fn new(id: u64, resource_type: ResourceType, label: Option<String>, size_bytes: u64) -> Self {
        Self {
            id,
            resource_type,
            label,
            size_bytes,
            created_at: Instant::now(),
            creation_context: None,
            expected_long_lived: resource_type.is_long_lived(),
        }
    }

    /// Get the age of this resource.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get the age in seconds.
    pub fn age_secs(&self) -> f64 {
        self.age().as_secs_f64()
    }

    /// Set the creation context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.creation_context = Some(context.into());
        self
    }

    /// Mark as expected long-lived.
    pub fn mark_long_lived(mut self) -> Self {
        self.expected_long_lived = true;
        self
    }
}

impl std::fmt::Display for TrackedResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = self.label.as_deref().unwrap_or("unnamed");
        write!(
            f,
            "{} '{}' (id={}, {} bytes, age={:.1}s)",
            self.resource_type,
            label,
            self.id,
            self.size_bytes,
            self.age_secs()
        )
    }
}

// ============================================================================
// LeakReport
// ============================================================================

/// Report of detected resource leaks.
#[derive(Debug, Clone)]
pub struct LeakReport {
    /// Resources that may be leaking.
    pub potential_leaks: Vec<TrackedResource>,
    /// Total bytes potentially leaked.
    pub total_bytes: u64,
    /// Time when the report was generated.
    pub generated_at: Instant,
    /// Warning threshold used (seconds).
    pub warning_threshold_secs: f64,
    /// Critical threshold used (seconds).
    pub critical_threshold_secs: f64,
}

impl Default for LeakReport {
    fn default() -> Self {
        Self {
            potential_leaks: Vec::new(),
            total_bytes: 0,
            generated_at: Instant::now(),
            warning_threshold_secs: 5.0,
            critical_threshold_secs: 30.0,
        }
    }
}

impl LeakReport {
    /// Create an empty report.
    pub fn empty() -> Self {
        Self {
            generated_at: Instant::now(),
            ..Default::default()
        }
    }

    /// Check if there are any potential leaks.
    pub fn has_leaks(&self) -> bool {
        !self.potential_leaks.is_empty()
    }

    /// Get the number of potential leaks.
    pub fn leak_count(&self) -> usize {
        self.potential_leaks.len()
    }

    /// Get leaks by resource type.
    pub fn leaks_by_type(&self, resource_type: ResourceType) -> Vec<&TrackedResource> {
        self.potential_leaks
            .iter()
            .filter(|r| r.resource_type == resource_type)
            .collect()
    }

    /// Get the oldest leak.
    pub fn oldest_leak(&self) -> Option<&TrackedResource> {
        self.potential_leaks
            .iter()
            .max_by(|a, b| a.age().cmp(&b.age()))
    }

    /// Get critical leaks (above critical threshold).
    pub fn critical_leaks(&self) -> Vec<&TrackedResource> {
        self.potential_leaks
            .iter()
            .filter(|r| r.age_secs() >= self.critical_threshold_secs)
            .collect()
    }

    /// Get warning leaks (above warning but below critical).
    pub fn warning_leaks(&self) -> Vec<&TrackedResource> {
        self.potential_leaks
            .iter()
            .filter(|r| {
                let age = r.age_secs();
                age >= self.warning_threshold_secs && age < self.critical_threshold_secs
            })
            .collect()
    }
}

impl std::fmt::Display for LeakReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.has_leaks() {
            writeln!(
                f,
                "LeakReport: {} potential leaks, {} bytes",
                self.leak_count(),
                self.total_bytes
            )?;
            for leak in &self.potential_leaks {
                writeln!(f, "  - {}", leak)?;
            }
            Ok(())
        } else {
            write!(f, "LeakReport: No leaks detected")
        }
    }
}

// ============================================================================
// LeakDetectorStats
// ============================================================================

/// Statistics about the leak detector.
#[derive(Debug, Clone, Default)]
pub struct LeakDetectorStats {
    /// Total resources tracked since creation.
    pub total_tracked: u64,
    /// Total resources untracked since creation.
    pub total_untracked: u64,
    /// Currently tracked resource count.
    pub currently_tracked: usize,
    /// Total bytes currently tracked.
    pub total_bytes: u64,
    /// Number of leak checks performed.
    pub checks_performed: u64,
    /// Number of leaks detected across all checks.
    pub total_leaks_detected: u64,
}

impl LeakDetectorStats {
    /// Check if there are currently tracked resources.
    pub fn has_tracked_resources(&self) -> bool {
        self.currently_tracked > 0
    }

    /// Get the tracking balance (tracked - untracked).
    pub fn tracking_balance(&self) -> i64 {
        self.total_tracked as i64 - self.total_untracked as i64
    }
}

// ============================================================================
// LeakDetector
// ============================================================================

/// Detects potential resource leaks.
#[derive(Debug)]
pub struct LeakDetector {
    /// Tracked resources by ID.
    resources: HashMap<u64, TrackedResource>,
    /// Warning threshold in seconds.
    warning_threshold_secs: f64,
    /// Critical threshold in seconds.
    critical_threshold_secs: f64,
    /// Whether detection is enabled.
    enabled: bool,
    /// Statistics.
    stats: LeakDetectorStats,
    /// IDs to ignore (expected long-lived).
    ignored_ids: std::collections::HashSet<u64>,
}

impl LeakDetector {
    /// Create a new leak detector with default thresholds.
    ///
    /// Default thresholds:
    /// - Warning: 5 seconds
    /// - Critical: 30 seconds
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            warning_threshold_secs: 5.0,
            critical_threshold_secs: 30.0,
            enabled: true,
            stats: LeakDetectorStats::default(),
            ignored_ids: std::collections::HashSet::new(),
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(warning_secs: f64, critical_secs: f64) -> Self {
        Self {
            warning_threshold_secs: warning_secs,
            critical_threshold_secs: critical_secs,
            ..Self::new()
        }
    }

    /// Track a resource.
    pub fn track(
        &mut self,
        id: u64,
        resource_type: ResourceType,
        label: impl Into<String>,
        size_bytes: u64,
    ) {
        if !self.enabled {
            return;
        }

        let resource = TrackedResource::new(id, resource_type, Some(label.into()), size_bytes);
        self.stats.total_tracked += 1;
        self.stats.total_bytes += size_bytes;
        self.stats.currently_tracked += 1;
        self.resources.insert(id, resource);
    }

    /// Track a resource with full details.
    pub fn track_resource(&mut self, resource: TrackedResource) {
        if !self.enabled {
            return;
        }

        self.stats.total_tracked += 1;
        self.stats.total_bytes += resource.size_bytes;
        self.stats.currently_tracked += 1;
        self.resources.insert(resource.id, resource);
    }

    /// Untrack a resource.
    pub fn untrack(&mut self, id: u64) -> Option<TrackedResource> {
        if !self.enabled {
            return None;
        }

        if let Some(resource) = self.resources.remove(&id) {
            self.stats.total_untracked += 1;
            self.stats.total_bytes = self.stats.total_bytes.saturating_sub(resource.size_bytes);
            self.stats.currently_tracked = self.stats.currently_tracked.saturating_sub(1);
            self.ignored_ids.remove(&id);
            Some(resource)
        } else {
            None
        }
    }

    /// Mark a resource as expected long-lived (won't show up as leak).
    pub fn mark_expected(&mut self, id: u64) {
        self.ignored_ids.insert(id);
        if let Some(resource) = self.resources.get_mut(&id) {
            resource.expected_long_lived = true;
        }
    }

    /// Check for potential leaks.
    pub fn check_leaks(&mut self) -> LeakReport {
        self.stats.checks_performed += 1;

        let mut report = LeakReport {
            warning_threshold_secs: self.warning_threshold_secs,
            critical_threshold_secs: self.critical_threshold_secs,
            generated_at: Instant::now(),
            ..Default::default()
        };

        for resource in self.resources.values() {
            // Skip ignored resources
            if self.ignored_ids.contains(&resource.id) {
                continue;
            }

            // Skip expected long-lived resources
            if resource.expected_long_lived {
                continue;
            }

            // Check if age exceeds warning threshold
            if resource.age_secs() >= self.warning_threshold_secs {
                report.total_bytes += resource.size_bytes;
                report.potential_leaks.push(resource.clone());
            }
        }

        self.stats.total_leaks_detected += report.leak_count() as u64;

        report
    }

    /// Get a tracked resource by ID.
    pub fn get(&self, id: u64) -> Option<&TrackedResource> {
        self.resources.get(&id)
    }

    /// Check if a resource is tracked.
    pub fn is_tracked(&self, id: u64) -> bool {
        self.resources.contains_key(&id)
    }

    /// Get all currently tracked resources.
    pub fn all_resources(&self) -> impl Iterator<Item = &TrackedResource> {
        self.resources.values()
    }

    /// Get statistics.
    pub fn stats(&self) -> &LeakDetectorStats {
        &self.stats
    }

    /// Enable detection.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable detection.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Clear all tracked resources.
    pub fn clear(&mut self) {
        self.resources.clear();
        self.ignored_ids.clear();
        self.stats.currently_tracked = 0;
        self.stats.total_bytes = 0;
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = LeakDetectorStats::default();
        self.stats.currently_tracked = self.resources.len();
        self.stats.total_bytes = self.resources.values().map(|r| r.size_bytes).sum();
    }

    /// Get the number of currently tracked resources.
    pub fn tracked_count(&self) -> usize {
        self.resources.len()
    }

    /// Set warning threshold.
    pub fn set_warning_threshold(&mut self, secs: f64) {
        self.warning_threshold_secs = secs;
    }

    /// Set critical threshold.
    pub fn set_critical_threshold(&mut self, secs: f64) {
        self.critical_threshold_secs = secs;
    }
}

impl Default for LeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LeakScope
// ============================================================================

/// RAII guard for automatic resource tracking.
///
/// Tracks the resource when created, untracks when dropped.
pub struct LeakScope<'a> {
    detector: &'a mut LeakDetector,
    id: u64,
}

impl<'a> LeakScope<'a> {
    /// Create a new leak scope and track the resource.
    pub fn new(
        detector: &'a mut LeakDetector,
        id: u64,
        resource_type: ResourceType,
        label: impl Into<String>,
        size_bytes: u64,
    ) -> Self {
        detector.track(id, resource_type, label, size_bytes);
        Self { detector, id }
    }

    /// Create a scope from an existing tracked resource.
    pub fn from_resource(detector: &'a mut LeakDetector, resource: TrackedResource) -> Self {
        let id = resource.id;
        detector.track_resource(resource);
        Self { detector, id }
    }

    /// Get the tracked resource ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the tracked resource.
    pub fn resource(&self) -> Option<&TrackedResource> {
        self.detector.get(self.id)
    }

    /// Mark as expected long-lived (prevents leak warning on drop).
    pub fn mark_long_lived(&mut self) {
        self.detector.mark_expected(self.id);
    }

    /// Release without untracking (transfers ownership elsewhere).
    pub fn release(self) -> u64 {
        let id = self.id;
        std::mem::forget(self);
        id
    }
}

impl<'a> Drop for LeakScope<'a> {
    fn drop(&mut self) {
        self.detector.untrack(self.id);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_type_name() {
        assert_eq!(ResourceType::Buffer.name(), "Buffer");
        assert_eq!(ResourceType::Texture.name(), "Texture");
        assert_eq!(ResourceType::RenderPipeline.name(), "RenderPipeline");
    }

    #[test]
    fn test_resource_type_is_long_lived() {
        assert!(ResourceType::RenderPipeline.is_long_lived());
        assert!(ResourceType::ComputePipeline.is_long_lived());
        assert!(ResourceType::ShaderModule.is_long_lived());
        assert!(!ResourceType::Buffer.is_long_lived());
        assert!(!ResourceType::Texture.is_long_lived());
    }

    #[test]
    fn test_resource_type_display() {
        assert_eq!(ResourceType::Buffer.to_string(), "Buffer");
    }

    #[test]
    fn test_tracked_resource() {
        let resource = TrackedResource::new(1, ResourceType::Buffer, Some("Test".to_string()), 1024);
        assert_eq!(resource.id, 1);
        assert_eq!(resource.resource_type, ResourceType::Buffer);
        assert_eq!(resource.label.as_deref(), Some("Test"));
        assert_eq!(resource.size_bytes, 1024);
    }

    #[test]
    fn test_tracked_resource_age() {
        let resource = TrackedResource::new(1, ResourceType::Buffer, None, 0);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(resource.age_secs() >= 0.01);
    }

    #[test]
    fn test_leak_report_empty() {
        let report = LeakReport::empty();
        assert!(!report.has_leaks());
        assert_eq!(report.leak_count(), 0);
    }

    #[test]
    fn test_leak_detector_track_untrack() {
        let mut detector = LeakDetector::new();
        detector.track(1, ResourceType::Buffer, "Test", 1024);

        assert!(detector.is_tracked(1));
        assert_eq!(detector.tracked_count(), 1);

        let resource = detector.untrack(1);
        assert!(resource.is_some());
        assert!(!detector.is_tracked(1));
        assert_eq!(detector.tracked_count(), 0);
    }

    #[test]
    fn test_leak_detector_stats() {
        let mut detector = LeakDetector::new();
        detector.track(1, ResourceType::Buffer, "Test", 1024);
        detector.track(2, ResourceType::Texture, "Tex", 4096);

        let stats = detector.stats();
        assert_eq!(stats.total_tracked, 2);
        assert_eq!(stats.currently_tracked, 2);
        assert_eq!(stats.total_bytes, 5120);

        detector.untrack(1);
        let stats = detector.stats();
        assert_eq!(stats.total_untracked, 1);
        assert_eq!(stats.currently_tracked, 1);
    }

    #[test]
    fn test_leak_detector_check_no_leaks() {
        let mut detector = LeakDetector::new();
        detector.track(1, ResourceType::Buffer, "Test", 1024);
        detector.untrack(1);

        let report = detector.check_leaks();
        assert!(!report.has_leaks());
    }

    #[test]
    fn test_leak_detector_mark_expected() {
        let mut detector = LeakDetector::with_thresholds(0.0, 0.0); // Immediate detection
        detector.track(1, ResourceType::Buffer, "Test", 1024);
        detector.mark_expected(1);

        let report = detector.check_leaks();
        assert!(!report.has_leaks()); // Marked as expected, not a leak
    }

    #[test]
    fn test_leak_detector_long_lived_resources() {
        let mut detector = LeakDetector::with_thresholds(0.0, 0.0);
        detector.track(1, ResourceType::RenderPipeline, "Pipeline", 1024);

        let report = detector.check_leaks();
        // Pipelines are auto-marked as long-lived, shouldn't show as leak
        assert!(!report.has_leaks());
    }

    #[test]
    fn test_leak_detector_disabled() {
        let mut detector = LeakDetector::new();
        detector.disable();
        detector.track(1, ResourceType::Buffer, "Test", 1024);

        assert!(!detector.is_tracked(1));
        assert!(!detector.is_enabled());
    }

    #[test]
    fn test_leak_detector_clear() {
        let mut detector = LeakDetector::new();
        detector.track(1, ResourceType::Buffer, "A", 1024);
        detector.track(2, ResourceType::Buffer, "B", 2048);
        detector.clear();

        assert_eq!(detector.tracked_count(), 0);
        assert_eq!(detector.stats().currently_tracked, 0);
    }

    #[test]
    fn test_leak_scope() {
        let mut detector = LeakDetector::new();

        // Verify not tracked before scope
        assert!(!detector.is_tracked(1));

        {
            let _scope = LeakScope::new(&mut detector, 1, ResourceType::Buffer, "Test", 1024);
            // Cannot call is_tracked here due to borrow conflict - _scope holds &mut detector
            // The scope creation itself tracks the resource
        }

        // After scope drops, resource should be untracked
        assert!(!detector.is_tracked(1));
    }

    #[test]
    fn test_leak_scope_release() {
        let mut detector = LeakDetector::new();

        let id = {
            let scope = LeakScope::new(&mut detector, 1, ResourceType::Buffer, "Test", 1024);
            scope.release()
        };

        assert_eq!(id, 1);
        assert!(detector.is_tracked(1)); // Still tracked after release
    }

    #[test]
    fn test_leak_report_by_type() {
        let mut detector = LeakDetector::with_thresholds(0.0, 0.0);
        detector.track(1, ResourceType::Buffer, "Buf1", 1024);
        detector.track(2, ResourceType::Buffer, "Buf2", 1024);
        detector.track(3, ResourceType::Texture, "Tex1", 4096);

        // Make buffers not long-lived for testing
        if let Some(r) = detector.resources.get_mut(&1) {
            r.expected_long_lived = false;
        }
        if let Some(r) = detector.resources.get_mut(&2) {
            r.expected_long_lived = false;
        }
        if let Some(r) = detector.resources.get_mut(&3) {
            r.expected_long_lived = false;
        }

        let report = detector.check_leaks();
        let buffer_leaks = report.leaks_by_type(ResourceType::Buffer);
        let texture_leaks = report.leaks_by_type(ResourceType::Texture);

        assert_eq!(buffer_leaks.len(), 2);
        assert_eq!(texture_leaks.len(), 1);
    }

    #[test]
    fn test_leak_detector_get() {
        let mut detector = LeakDetector::new();
        detector.track(42, ResourceType::Buffer, "Test", 1024);

        let resource = detector.get(42);
        assert!(resource.is_some());
        assert_eq!(resource.unwrap().id, 42);

        assert!(detector.get(999).is_none());
    }

    #[test]
    fn test_tracked_resource_with_context() {
        let resource = TrackedResource::new(1, ResourceType::Buffer, None, 0)
            .with_context("Created in render_frame()");
        assert_eq!(
            resource.creation_context.as_deref(),
            Some("Created in render_frame()")
        );
    }

    #[test]
    fn test_leak_detector_stats_balance() {
        let mut detector = LeakDetector::new();
        detector.track(1, ResourceType::Buffer, "A", 100);
        detector.track(2, ResourceType::Buffer, "B", 200);
        detector.untrack(1);

        let stats = detector.stats();
        assert_eq!(stats.tracking_balance(), 1);
    }

    #[test]
    fn test_leak_report_oldest_leak() {
        let mut detector = LeakDetector::with_thresholds(0.0, 0.0);

        // Track first resource
        detector.track(1, ResourceType::Buffer, "First", 100);
        if let Some(r) = detector.resources.get_mut(&1) {
            r.expected_long_lived = false;
        }

        std::thread::sleep(std::time::Duration::from_millis(10));

        // Track second resource
        detector.track(2, ResourceType::Buffer, "Second", 200);
        if let Some(r) = detector.resources.get_mut(&2) {
            r.expected_long_lived = false;
        }

        let report = detector.check_leaks();
        let oldest = report.oldest_leak().unwrap();
        assert_eq!(oldest.id, 1);
    }

    #[test]
    fn test_leak_scope_mark_long_lived() {
        let mut detector = LeakDetector::with_thresholds(0.0, 0.0);

        {
            let mut scope = LeakScope::new(&mut detector, 1, ResourceType::Buffer, "Test", 100);
            scope.mark_long_lived();
        }

        // After scope ends, check that it was properly cleaned up
        assert!(!detector.is_tracked(1));
    }

    #[test]
    fn test_leak_detector_set_thresholds() {
        let mut detector = LeakDetector::new();
        detector.set_warning_threshold(10.0);
        detector.set_critical_threshold(60.0);

        let report = detector.check_leaks();
        assert_eq!(report.warning_threshold_secs, 10.0);
        assert_eq!(report.critical_threshold_secs, 60.0);
    }

    #[test]
    fn test_resource_type_default() {
        assert_eq!(ResourceType::default(), ResourceType::Unknown);
    }

    #[test]
    fn test_leak_detector_all_resources() {
        let mut detector = LeakDetector::new();
        detector.track(1, ResourceType::Buffer, "A", 100);
        detector.track(2, ResourceType::Texture, "B", 200);

        let resources: Vec<_> = detector.all_resources().collect();
        assert_eq!(resources.len(), 2);
    }

    #[test]
    fn test_leak_detector_reset_stats() {
        let mut detector = LeakDetector::new();
        detector.track(1, ResourceType::Buffer, "Test", 1024);
        detector.check_leaks();
        detector.check_leaks();

        assert_eq!(detector.stats().checks_performed, 2);

        detector.reset_stats();

        assert_eq!(detector.stats().checks_performed, 0);
        assert_eq!(detector.stats().currently_tracked, 1); // Still tracking
    }
}
