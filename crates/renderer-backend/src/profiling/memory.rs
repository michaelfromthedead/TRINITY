//! GPU Memory Tracking and Statistics for wgpu 25.x Resource Management.
//!
//! This module provides comprehensive GPU memory tracking capabilities for
//! monitoring resource allocations, detecting memory leaks, and analyzing
//! GPU memory usage patterns.
//!
//! # Overview
//!
//! The memory tracker supports:
//! - **AllocationInfo**: Individual allocation metadata with timestamps
//! - **AllocationStats**: Aggregate statistics for all allocations
//! - **MemoryBudget**: GPU memory budget tracking from adapter limits
//! - **MemoryTracker**: Main interface for tracking allocations
//! - **MemorySnapshot**: Point-in-time memory state capture
//! - **MemoryDiff**: Comparison between snapshots
//! - **LeakDetector**: Detection of long-lived allocations
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::profiling::memory::{MemoryTracker, ResourceType};
//!
//! # fn example(adapter: &wgpu::Adapter, device: &wgpu::Device) {
//! // Create tracker with adapter limits
//! let mut tracker = MemoryTracker::new(adapter);
//!
//! // Track buffer allocation
//! let buffer = device.create_buffer(&wgpu::BufferDescriptor {
//!     label: Some("Vertex Buffer"),
//!     size: 1024 * 1024, // 1MB
//!     usage: wgpu::BufferUsages::VERTEX,
//!     mapped_at_creation: false,
//! });
//! let id = tracker.track_buffer(&buffer, 1024 * 1024, Some("Vertex Buffer"));
//!
//! // Check memory usage
//! println!("Memory utilization: {:.1}%", tracker.budget().utilization() * 100.0);
//! println!("{}", tracker.summary());
//!
//! // Untrack when resource is dropped
//! tracker.untrack(id);
//! # }
//! ```
//!
//! # Leak Detection
//!
//! ```no_run
//! use renderer_backend::profiling::memory::{MemoryTracker, LeakDetector};
//!
//! # fn example(adapter: &wgpu::Adapter) {
//! let tracker = MemoryTracker::new(adapter);
//! let mut leak_detector = LeakDetector::new();
//!
//! // Check for allocations older than 60 seconds
//! let potential_leaks = leak_detector.check_leaks(&tracker, 60.0);
//! for leak in potential_leaks {
//!     eprintln!("Potential leak: {:?} ({} bytes)", leak.label, leak.size_bytes);
//! }
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// MemoryType
// ============================================================================

/// Types of GPU memory based on visibility and caching characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Device-local memory (fastest, GPU-only access).
    DeviceLocal,
    /// Host-visible memory (CPU can access, slower GPU access).
    HostVisible,
    /// Host-coherent memory (no explicit flush required).
    HostCoherent,
    /// Host-cached memory (faster CPU reads, may need invalidation).
    HostCached,
}

impl MemoryType {
    /// Infer memory type from wgpu buffer usage flags.
    ///
    /// This makes a best-effort guess based on buffer usage:
    /// - MAP_READ or MAP_WRITE implies host-visible memory
    /// - COPY_DST with MAP_READ implies staging (host-coherent)
    /// - Otherwise assumes device-local for performance
    #[must_use]
    pub fn from_buffer_usage(usage: wgpu::BufferUsages) -> Self {
        if usage.contains(wgpu::BufferUsages::MAP_READ) {
            // Readback buffers typically use coherent memory
            MemoryType::HostCoherent
        } else if usage.contains(wgpu::BufferUsages::MAP_WRITE) {
            // Upload buffers may be cached for write combining
            MemoryType::HostCached
        } else {
            // Default to device-local for best GPU performance
            MemoryType::DeviceLocal
        }
    }

    /// Infer memory type from wgpu texture usage flags.
    ///
    /// Textures are almost always device-local for performance.
    #[must_use]
    pub fn from_texture_usage(usage: wgpu::TextureUsages) -> Self {
        if usage.contains(wgpu::TextureUsages::COPY_SRC) {
            // Textures that will be read back might be in visible memory
            MemoryType::HostVisible
        } else {
            MemoryType::DeviceLocal
        }
    }

    /// Check if this memory type can be mapped by the CPU.
    #[must_use]
    pub fn is_mappable(&self) -> bool {
        matches!(
            self,
            MemoryType::HostVisible | MemoryType::HostCoherent | MemoryType::HostCached
        )
    }
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryType::DeviceLocal => write!(f, "Device Local"),
            MemoryType::HostVisible => write!(f, "Host Visible"),
            MemoryType::HostCoherent => write!(f, "Host Coherent"),
            MemoryType::HostCached => write!(f, "Host Cached"),
        }
    }
}

// ============================================================================
// ResourceType
// ============================================================================

/// Types of GPU resources that can be tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// GPU buffer (vertex, index, uniform, storage, etc.).
    Buffer,
    /// GPU texture (2D, 3D, cube, array).
    Texture,
    /// Query set for occlusion or timestamp queries.
    QuerySet,
    /// Bind group containing resource bindings.
    BindGroup,
    /// Render or compute pipeline.
    Pipeline,
    /// Other unclassified resources.
    Other,
}

impl ResourceType {
    /// Get a human-readable display name for this resource type.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            ResourceType::Buffer => "Buffer",
            ResourceType::Texture => "Texture",
            ResourceType::QuerySet => "Query Set",
            ResourceType::BindGroup => "Bind Group",
            ResourceType::Pipeline => "Pipeline",
            ResourceType::Other => "Other",
        }
    }
}

impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ============================================================================
// AllocationInfo
// ============================================================================

/// Information about a single GPU memory allocation.
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Unique identifier for this allocation.
    pub id: u64,
    /// Type of resource being tracked.
    pub resource_type: ResourceType,
    /// Type of memory backing this allocation.
    pub memory_type: MemoryType,
    /// Size of the allocation in bytes.
    pub size_bytes: u64,
    /// Optional label for debugging.
    pub label: Option<String>,
    /// Timestamp when this allocation was created.
    pub timestamp: Instant,
}

impl AllocationInfo {
    /// Create a new allocation info entry.
    #[must_use]
    pub fn new(
        id: u64,
        resource_type: ResourceType,
        memory_type: MemoryType,
        size_bytes: u64,
    ) -> Self {
        Self {
            id,
            resource_type,
            memory_type,
            size_bytes,
            label: None,
            timestamp: Instant::now(),
        }
    }

    /// Create a new allocation info entry with a label.
    #[must_use]
    pub fn with_label(
        id: u64,
        resource_type: ResourceType,
        memory_type: MemoryType,
        size_bytes: u64,
        label: impl Into<String>,
    ) -> Self {
        Self {
            id,
            resource_type,
            memory_type,
            size_bytes,
            label: Some(label.into()),
            timestamp: Instant::now(),
        }
    }

    /// Get the age of this allocation in seconds.
    #[must_use]
    pub fn age_secs(&self) -> f64 {
        self.timestamp.elapsed().as_secs_f64()
    }
}

// ============================================================================
// AllocationStats
// ============================================================================

/// Aggregate statistics for GPU memory allocations.
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    /// Total number of allocations ever made.
    pub total_allocations: u64,
    /// Total number of deallocations ever made.
    pub total_deallocations: u64,
    /// Current number of active allocations.
    pub current_allocations: u64,
    /// Peak number of concurrent allocations.
    pub peak_allocations: u64,
    /// Total bytes ever allocated.
    pub total_bytes_allocated: u64,
    /// Current bytes in use.
    pub current_bytes: u64,
    /// Peak bytes ever in use at once.
    pub peak_bytes: u64,
    /// Breakdown of current bytes by resource type.
    pub bytes_by_type: HashMap<ResourceType, u64>,
}

impl AllocationStats {
    /// Create new empty statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new allocation.
    pub fn record_allocation(&mut self, info: &AllocationInfo) {
        self.total_allocations += 1;
        self.current_allocations += 1;
        self.total_bytes_allocated += info.size_bytes;
        self.current_bytes += info.size_bytes;

        // Update peaks
        if self.current_allocations > self.peak_allocations {
            self.peak_allocations = self.current_allocations;
        }
        if self.current_bytes > self.peak_bytes {
            self.peak_bytes = self.current_bytes;
        }

        // Update type breakdown
        *self.bytes_by_type.entry(info.resource_type).or_insert(0) += info.size_bytes;
    }

    /// Record a deallocation.
    pub fn record_deallocation(&mut self, info: &AllocationInfo) {
        self.total_deallocations += 1;
        self.current_allocations = self.current_allocations.saturating_sub(1);
        self.current_bytes = self.current_bytes.saturating_sub(info.size_bytes);

        // Update type breakdown
        if let Some(bytes) = self.bytes_by_type.get_mut(&info.resource_type) {
            *bytes = bytes.saturating_sub(info.size_bytes);
        }
    }

    /// Reset peak statistics to current values.
    pub fn reset_peak(&mut self) {
        self.peak_allocations = self.current_allocations;
        self.peak_bytes = self.current_bytes;
    }

    /// Get the allocation rate (allocations - deallocations).
    #[must_use]
    pub fn net_allocations(&self) -> i64 {
        self.total_allocations as i64 - self.total_deallocations as i64
    }

    /// Get current memory in human-readable format.
    #[must_use]
    pub fn current_bytes_formatted(&self) -> String {
        format_bytes(self.current_bytes)
    }

    /// Get peak memory in human-readable format.
    #[must_use]
    pub fn peak_bytes_formatted(&self) -> String {
        format_bytes(self.peak_bytes)
    }
}

// ============================================================================
// MemoryBudget
// ============================================================================

/// GPU memory budget information from adapter limits.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Budget for device-local memory in bytes.
    pub device_local_budget: u64,
    /// Budget for host-visible memory in bytes.
    pub host_visible_budget: u64,
    /// Total memory budget in bytes.
    pub total_budget: u64,
    /// Current device-local memory usage in bytes.
    pub device_local_used: u64,
    /// Current host-visible memory usage in bytes.
    pub host_visible_used: u64,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        // Conservative defaults when adapter info isn't available
        Self {
            device_local_budget: 2 * 1024 * 1024 * 1024, // 2 GB
            host_visible_budget: 512 * 1024 * 1024,      // 512 MB
            total_budget: 2 * 1024 * 1024 * 1024 + 512 * 1024 * 1024,
            device_local_used: 0,
            host_visible_used: 0,
        }
    }
}

impl MemoryBudget {
    /// Create a new memory budget from adapter limits.
    ///
    /// Note: wgpu doesn't expose detailed memory info, so this uses
    /// the maximum buffer size as a proxy for available memory.
    #[must_use]
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let limits = adapter.limits();

        // Use max buffer size as a rough estimate of device memory
        // Real memory budgeting would require backend-specific queries
        let device_local = limits.max_buffer_size;
        let host_visible = limits.max_buffer_size / 4; // Conservative estimate

        Self {
            device_local_budget: device_local,
            host_visible_budget: host_visible,
            total_budget: device_local + host_visible,
            device_local_used: 0,
            host_visible_used: 0,
        }
    }

    /// Get overall memory utilization as a fraction (0.0 to 1.0+).
    #[must_use]
    pub fn utilization(&self) -> f32 {
        if self.total_budget == 0 {
            return 0.0;
        }
        let total_used = self.device_local_used + self.host_visible_used;
        total_used as f32 / self.total_budget as f32
    }

    /// Get device-local memory utilization.
    #[must_use]
    pub fn device_local_utilization(&self) -> f32 {
        if self.device_local_budget == 0 {
            return 0.0;
        }
        self.device_local_used as f32 / self.device_local_budget as f32
    }

    /// Get host-visible memory utilization.
    #[must_use]
    pub fn host_visible_utilization(&self) -> f32 {
        if self.host_visible_budget == 0 {
            return 0.0;
        }
        self.host_visible_used as f32 / self.host_visible_budget as f32
    }

    /// Get remaining total memory in bytes.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        let total_used = self.device_local_used + self.host_visible_used;
        self.total_budget.saturating_sub(total_used)
    }

    /// Check if memory usage exceeds budget.
    #[must_use]
    pub fn is_over_budget(&self) -> bool {
        self.device_local_used > self.device_local_budget
            || self.host_visible_used > self.host_visible_budget
    }

    /// Record memory usage from an allocation.
    pub fn record_allocation(&mut self, memory_type: MemoryType, size: u64) {
        match memory_type {
            MemoryType::DeviceLocal => self.device_local_used += size,
            MemoryType::HostVisible | MemoryType::HostCoherent | MemoryType::HostCached => {
                self.host_visible_used += size;
            }
        }
    }

    /// Record memory freed from a deallocation.
    pub fn record_deallocation(&mut self, memory_type: MemoryType, size: u64) {
        match memory_type {
            MemoryType::DeviceLocal => {
                self.device_local_used = self.device_local_used.saturating_sub(size);
            }
            MemoryType::HostVisible | MemoryType::HostCoherent | MemoryType::HostCached => {
                self.host_visible_used = self.host_visible_used.saturating_sub(size);
            }
        }
    }
}

// ============================================================================
// MemoryTracker
// ============================================================================

/// Main interface for GPU memory tracking.
///
/// Tracks all GPU resource allocations and provides statistics,
/// budget monitoring, and leak detection capabilities.
pub struct MemoryTracker {
    /// Active allocations indexed by ID.
    allocations: HashMap<u64, AllocationInfo>,
    /// Aggregate statistics.
    stats: AllocationStats,
    /// Memory budget information.
    budget: MemoryBudget,
    /// Next allocation ID generator.
    next_id: AtomicU64,
}

impl MemoryTracker {
    /// Create a new memory tracker with adapter limits.
    #[must_use]
    pub fn new(adapter: &wgpu::Adapter) -> Self {
        Self {
            allocations: HashMap::new(),
            stats: AllocationStats::new(),
            budget: MemoryBudget::from_adapter(adapter),
            next_id: AtomicU64::new(1),
        }
    }

    /// Create a new memory tracker with default budget.
    #[must_use]
    pub fn with_default_budget() -> Self {
        Self {
            allocations: HashMap::new(),
            stats: AllocationStats::new(),
            budget: MemoryBudget::default(),
            next_id: AtomicU64::new(1),
        }
    }

    /// Track a buffer allocation.
    ///
    /// Returns the allocation ID for later untracking.
    pub fn track_buffer(
        &mut self,
        _buffer: &wgpu::Buffer,
        size: u64,
        label: Option<&str>,
    ) -> u64 {
        self.track_buffer_with_usage(size, wgpu::BufferUsages::empty(), label)
    }

    /// Track a buffer allocation with usage flags for memory type inference.
    pub fn track_buffer_with_usage(
        &mut self,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> u64 {
        let memory_type = MemoryType::from_buffer_usage(usage);
        self.track_resource(ResourceType::Buffer, memory_type, size, label)
    }

    /// Track a texture allocation.
    ///
    /// Returns the allocation ID for later untracking.
    pub fn track_texture(
        &mut self,
        _texture: &wgpu::Texture,
        size: u64,
        label: Option<&str>,
    ) -> u64 {
        self.track_texture_with_usage(size, wgpu::TextureUsages::empty(), label)
    }

    /// Track a texture allocation with usage flags for memory type inference.
    pub fn track_texture_with_usage(
        &mut self,
        size: u64,
        usage: wgpu::TextureUsages,
        label: Option<&str>,
    ) -> u64 {
        let memory_type = MemoryType::from_texture_usage(usage);
        self.track_resource(ResourceType::Texture, memory_type, size, label)
    }

    /// Track a query set allocation.
    pub fn track_query_set(&mut self, size: u64, label: Option<&str>) -> u64 {
        self.track_resource(ResourceType::QuerySet, MemoryType::DeviceLocal, size, label)
    }

    /// Track a bind group (typically minimal memory).
    pub fn track_bind_group(&mut self, label: Option<&str>) -> u64 {
        // Bind groups use negligible memory, track for completeness
        self.track_resource(ResourceType::BindGroup, MemoryType::DeviceLocal, 256, label)
    }

    /// Track a pipeline (typically minimal memory).
    pub fn track_pipeline(&mut self, label: Option<&str>) -> u64 {
        // Pipelines use driver-dependent memory
        self.track_resource(ResourceType::Pipeline, MemoryType::DeviceLocal, 4096, label)
    }

    /// Track a generic resource allocation.
    pub fn track_resource(
        &mut self,
        resource_type: ResourceType,
        memory_type: MemoryType,
        size: u64,
        label: Option<&str>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let info = if let Some(label) = label {
            AllocationInfo::with_label(id, resource_type, memory_type, size, label)
        } else {
            AllocationInfo::new(id, resource_type, memory_type, size)
        };

        self.stats.record_allocation(&info);
        self.budget.record_allocation(memory_type, size);
        self.allocations.insert(id, info);

        id
    }

    /// Stop tracking an allocation.
    ///
    /// Returns true if the allocation was found and removed.
    pub fn untrack(&mut self, id: u64) -> bool {
        if let Some(info) = self.allocations.remove(&id) {
            self.stats.record_deallocation(&info);
            self.budget.record_deallocation(info.memory_type, info.size_bytes);
            true
        } else {
            false
        }
    }

    /// Get information about a specific allocation.
    #[must_use]
    pub fn get_allocation(&self, id: u64) -> Option<&AllocationInfo> {
        self.allocations.get(&id)
    }

    /// Get all current allocations.
    #[must_use]
    pub fn allocations(&self) -> &HashMap<u64, AllocationInfo> {
        &self.allocations
    }

    /// Get aggregate statistics.
    #[must_use]
    pub fn stats(&self) -> &AllocationStats {
        &self.stats
    }

    /// Get mutable reference to statistics (for peak reset).
    pub fn stats_mut(&mut self) -> &mut AllocationStats {
        &mut self.stats
    }

    /// Get memory budget information.
    #[must_use]
    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Generate a human-readable summary of memory usage.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== GPU Memory Summary ===\n");
        s.push_str(&format!(
            "Current: {} ({} allocations)\n",
            format_bytes(self.stats.current_bytes),
            self.stats.current_allocations
        ));
        s.push_str(&format!(
            "Peak: {} ({} allocations)\n",
            format_bytes(self.stats.peak_bytes),
            self.stats.peak_allocations
        ));
        s.push_str(&format!(
            "Budget: {:.1}% used ({} remaining)\n",
            self.budget.utilization() * 100.0,
            format_bytes(self.budget.remaining())
        ));

        if !self.stats.bytes_by_type.is_empty() {
            s.push_str("\nBy Resource Type:\n");
            for (resource_type, bytes) in &self.stats.bytes_by_type {
                if *bytes > 0 {
                    s.push_str(&format!("  {}: {}\n", resource_type, format_bytes(*bytes)));
                }
            }
        }

        s
    }

    /// Clear all tracked allocations.
    pub fn clear(&mut self) {
        self.allocations.clear();
        self.stats = AllocationStats::new();
        self.budget.device_local_used = 0;
        self.budget.host_visible_used = 0;
    }

    /// Take a snapshot of current memory state.
    #[must_use]
    pub fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            allocations: self.allocations.values().cloned().collect(),
            stats: self.stats.clone(),
            budget: self.budget.clone(),
            timestamp: Instant::now(),
        }
    }
}

impl fmt::Debug for MemoryTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryTracker")
            .field("allocations_count", &self.allocations.len())
            .field("current_bytes", &self.stats.current_bytes)
            .field("peak_bytes", &self.stats.peak_bytes)
            .finish()
    }
}

// ============================================================================
// MemorySnapshot
// ============================================================================

/// Point-in-time snapshot of memory state.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// All allocations at snapshot time.
    pub allocations: Vec<AllocationInfo>,
    /// Statistics at snapshot time.
    pub stats: AllocationStats,
    /// Budget information at snapshot time.
    pub budget: MemoryBudget,
    /// When the snapshot was taken.
    pub timestamp: Instant,
}

impl MemorySnapshot {
    /// Compare this snapshot with another to find differences.
    #[must_use]
    pub fn diff(&self, other: &MemorySnapshot) -> MemoryDiff {
        let self_ids: std::collections::HashSet<u64> =
            self.allocations.iter().map(|a| a.id).collect();
        let other_ids: std::collections::HashSet<u64> =
            other.allocations.iter().map(|a| a.id).collect();

        let added: Vec<AllocationInfo> = other
            .allocations
            .iter()
            .filter(|a| !self_ids.contains(&a.id))
            .cloned()
            .collect();

        let removed: Vec<u64> = self_ids
            .iter()
            .filter(|id| !other_ids.contains(id))
            .copied()
            .collect();

        let bytes_delta = other.stats.current_bytes as i64 - self.stats.current_bytes as i64;

        MemoryDiff {
            added,
            removed,
            bytes_delta,
        }
    }

    /// Get the age of this snapshot in seconds.
    #[must_use]
    pub fn age_secs(&self) -> f64 {
        self.timestamp.elapsed().as_secs_f64()
    }
}

// ============================================================================
// MemoryDiff
// ============================================================================

/// Difference between two memory snapshots.
#[derive(Debug, Clone)]
pub struct MemoryDiff {
    /// Allocations added since the previous snapshot.
    pub added: Vec<AllocationInfo>,
    /// IDs of allocations removed since the previous snapshot.
    pub removed: Vec<u64>,
    /// Net change in bytes (positive = increased, negative = decreased).
    pub bytes_delta: i64,
}

impl MemoryDiff {
    /// Check if there were no changes between snapshots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }

    /// Get the number of allocations added.
    #[must_use]
    pub fn added_count(&self) -> usize {
        self.added.len()
    }

    /// Get the number of allocations removed.
    #[must_use]
    pub fn removed_count(&self) -> usize {
        self.removed.len()
    }

    /// Get total bytes added.
    #[must_use]
    pub fn bytes_added(&self) -> u64 {
        self.added.iter().map(|a| a.size_bytes).sum()
    }
}

impl fmt::Display for MemoryDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryDiff {{ +{} allocs, -{} allocs, {} bytes }}",
            self.added.len(),
            self.removed.len(),
            if self.bytes_delta >= 0 {
                format!("+{}", format_bytes(self.bytes_delta as u64))
            } else {
                format!("-{}", format_bytes((-self.bytes_delta) as u64))
            }
        )
    }
}

// ============================================================================
// LeakDetector
// ============================================================================

/// Detector for potential memory leaks.
///
/// Tracks allocations that persist longer than expected thresholds
/// and can exclude known long-lived resources.
#[derive(Debug, Default)]
pub struct LeakDetector {
    /// IDs of allocations expected to be long-lived (not leaks).
    expected_long_lived: std::collections::HashSet<u64>,
}

impl LeakDetector {
    /// Create a new leak detector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark an allocation as expected to be long-lived.
    ///
    /// Long-lived allocations (like static meshes) won't be reported as leaks.
    pub fn mark_expected(&mut self, id: u64) {
        self.expected_long_lived.insert(id);
    }

    /// Remove an allocation from the expected set.
    pub fn unmark_expected(&mut self, id: u64) {
        self.expected_long_lived.remove(&id);
    }

    /// Clear all expected allocations.
    pub fn clear_expected(&mut self) {
        self.expected_long_lived.clear();
    }

    /// Check for potential memory leaks.
    ///
    /// Returns allocations that have existed longer than the threshold
    /// and are not marked as expected long-lived resources.
    #[must_use]
    pub fn check_leaks(&self, tracker: &MemoryTracker, threshold_secs: f64) -> Vec<AllocationInfo> {
        tracker
            .allocations
            .values()
            .filter(|info| {
                info.age_secs() > threshold_secs && !self.expected_long_lived.contains(&info.id)
            })
            .cloned()
            .collect()
    }

    /// Check for leaks of a specific resource type.
    #[must_use]
    pub fn check_leaks_by_type(
        &self,
        tracker: &MemoryTracker,
        resource_type: ResourceType,
        threshold_secs: f64,
    ) -> Vec<AllocationInfo> {
        tracker
            .allocations
            .values()
            .filter(|info| {
                info.resource_type == resource_type
                    && info.age_secs() > threshold_secs
                    && !self.expected_long_lived.contains(&info.id)
            })
            .cloned()
            .collect()
    }

    /// Get the number of expected long-lived allocations.
    #[must_use]
    pub fn expected_count(&self) -> usize {
        self.expected_long_lived.len()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Format bytes into a human-readable string.
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // MemoryType Tests
    // ========================================================================

    #[test]
    fn test_memory_type_from_buffer_usage_device_local() {
        let usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::DeviceLocal);
    }

    #[test]
    fn test_memory_type_from_buffer_usage_map_read() {
        let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::HostCoherent);
    }

    #[test]
    fn test_memory_type_from_buffer_usage_map_write() {
        let usage = wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::HostCached);
    }

    #[test]
    fn test_memory_type_is_mappable() {
        assert!(!MemoryType::DeviceLocal.is_mappable());
        assert!(MemoryType::HostVisible.is_mappable());
        assert!(MemoryType::HostCoherent.is_mappable());
        assert!(MemoryType::HostCached.is_mappable());
    }

    #[test]
    fn test_memory_type_display() {
        assert_eq!(format!("{}", MemoryType::DeviceLocal), "Device Local");
        assert_eq!(format!("{}", MemoryType::HostVisible), "Host Visible");
    }

    // ========================================================================
    // ResourceType Tests
    // ========================================================================

    #[test]
    fn test_resource_type_display_name() {
        assert_eq!(ResourceType::Buffer.display_name(), "Buffer");
        assert_eq!(ResourceType::Texture.display_name(), "Texture");
        assert_eq!(ResourceType::QuerySet.display_name(), "Query Set");
        assert_eq!(ResourceType::BindGroup.display_name(), "Bind Group");
        assert_eq!(ResourceType::Pipeline.display_name(), "Pipeline");
        assert_eq!(ResourceType::Other.display_name(), "Other");
    }

    #[test]
    fn test_resource_type_display() {
        assert_eq!(format!("{}", ResourceType::Buffer), "Buffer");
    }

    // ========================================================================
    // AllocationInfo Tests
    // ========================================================================

    #[test]
    fn test_allocation_info_new() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        assert_eq!(info.id, 1);
        assert_eq!(info.resource_type, ResourceType::Buffer);
        assert_eq!(info.memory_type, MemoryType::DeviceLocal);
        assert_eq!(info.size_bytes, 1024);
        assert!(info.label.is_none());
    }

    #[test]
    fn test_allocation_info_with_label() {
        let info = AllocationInfo::with_label(
            2,
            ResourceType::Texture,
            MemoryType::DeviceLocal,
            4096,
            "Albedo Texture",
        );
        assert_eq!(info.id, 2);
        assert_eq!(info.label, Some("Albedo Texture".to_string()));
    }

    #[test]
    fn test_allocation_info_age() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        // Age should be very small (just created)
        assert!(info.age_secs() < 1.0);
    }

    // ========================================================================
    // AllocationStats Tests
    // ========================================================================

    #[test]
    fn test_allocation_stats_new() {
        let stats = AllocationStats::new();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.current_bytes, 0);
    }

    #[test]
    fn test_allocation_stats_record_allocation() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);

        stats.record_allocation(&info);

        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.current_allocations, 1);
        assert_eq!(stats.current_bytes, 1024);
        assert_eq!(stats.peak_allocations, 1);
        assert_eq!(stats.peak_bytes, 1024);
        assert_eq!(stats.bytes_by_type.get(&ResourceType::Buffer), Some(&1024));
    }

    #[test]
    fn test_allocation_stats_record_deallocation() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);

        stats.record_allocation(&info);
        stats.record_deallocation(&info);

        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.current_allocations, 0);
        assert_eq!(stats.current_bytes, 0);
        // Peak should remain at previous high
        assert_eq!(stats.peak_bytes, 1024);
    }

    #[test]
    fn test_allocation_stats_peak_tracking() {
        let mut stats = AllocationStats::new();

        // Allocate three buffers
        for i in 0..3 {
            let info = AllocationInfo::new(i, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
            stats.record_allocation(&info);
        }

        assert_eq!(stats.peak_allocations, 3);
        assert_eq!(stats.peak_bytes, 3072);

        // Deallocate one
        let info = AllocationInfo::new(0, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_deallocation(&info);

        // Peak should remain unchanged
        assert_eq!(stats.peak_allocations, 3);
        assert_eq!(stats.peak_bytes, 3072);
        assert_eq!(stats.current_allocations, 2);
    }

    #[test]
    fn test_allocation_stats_reset_peak() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);

        stats.record_allocation(&info);
        stats.record_deallocation(&info);
        stats.reset_peak();

        assert_eq!(stats.peak_allocations, 0);
        assert_eq!(stats.peak_bytes, 0);
    }

    // ========================================================================
    // MemoryBudget Tests
    // ========================================================================

    #[test]
    fn test_memory_budget_default() {
        let budget = MemoryBudget::default();
        assert!(budget.device_local_budget > 0);
        assert!(budget.total_budget > 0);
        assert_eq!(budget.device_local_used, 0);
    }

    #[test]
    fn test_memory_budget_utilization() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = budget.device_local_budget / 2;

        let util = budget.utilization();
        assert!(util > 0.0 && util < 1.0);
    }

    #[test]
    fn test_memory_budget_remaining() {
        let mut budget = MemoryBudget::default();
        let initial_remaining = budget.remaining();

        budget.device_local_used = 1000;
        assert_eq!(budget.remaining(), initial_remaining - 1000);
    }

    #[test]
    fn test_memory_budget_over_budget() {
        let mut budget = MemoryBudget::default();
        assert!(!budget.is_over_budget());

        budget.device_local_used = budget.device_local_budget + 1;
        assert!(budget.is_over_budget());
    }

    #[test]
    fn test_memory_budget_record_allocation() {
        let mut budget = MemoryBudget::default();
        budget.record_allocation(MemoryType::DeviceLocal, 1000);
        assert_eq!(budget.device_local_used, 1000);

        budget.record_allocation(MemoryType::HostVisible, 500);
        assert_eq!(budget.host_visible_used, 500);
    }

    // ========================================================================
    // MemoryTracker Tests
    // ========================================================================

    #[test]
    fn test_memory_tracker_with_default_budget() {
        let tracker = MemoryTracker::with_default_budget();
        assert_eq!(tracker.stats().current_allocations, 0);
    }

    #[test]
    fn test_memory_tracker_track_resource() {
        let mut tracker = MemoryTracker::with_default_budget();

        let id = tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            Some("Test Buffer"),
        );

        assert!(id > 0);
        assert_eq!(tracker.stats().current_allocations, 1);
        assert_eq!(tracker.stats().current_bytes, 1024);

        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.label, Some("Test Buffer".to_string()));
    }

    #[test]
    fn test_memory_tracker_untrack() {
        let mut tracker = MemoryTracker::with_default_budget();

        let id = tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            None,
        );

        assert!(tracker.untrack(id));
        assert_eq!(tracker.stats().current_allocations, 0);
        assert_eq!(tracker.stats().current_bytes, 0);

        // Untracking again should return false
        assert!(!tracker.untrack(id));
    }

    #[test]
    fn test_memory_tracker_track_buffer_with_usage() {
        let mut tracker = MemoryTracker::with_default_budget();

        let id = tracker.track_buffer_with_usage(
            2048,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("Readback"),
        );

        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.memory_type, MemoryType::HostCoherent);
    }

    #[test]
    fn test_memory_tracker_summary() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 4096, None);

        let summary = tracker.summary();
        assert!(summary.contains("GPU Memory Summary"));
        assert!(summary.contains("Buffer"));
        assert!(summary.contains("Texture"));
    }

    #[test]
    fn test_memory_tracker_clear() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);

        tracker.clear();

        assert_eq!(tracker.stats().current_allocations, 0);
        assert_eq!(tracker.allocations().len(), 0);
    }

    // ========================================================================
    // MemorySnapshot Tests
    // ========================================================================

    #[test]
    fn test_memory_snapshot() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.allocations.len(), 1);
        assert_eq!(snapshot.stats.current_bytes, 1024);
    }

    #[test]
    fn test_memory_snapshot_diff() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id1 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);

        let snapshot1 = tracker.snapshot();

        tracker.untrack(id1);
        tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 2048, None);

        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);

        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.bytes_delta, 1024); // 2048 - 1024
    }

    // ========================================================================
    // MemoryDiff Tests
    // ========================================================================

    #[test]
    fn test_memory_diff_is_empty() {
        let diff = MemoryDiff {
            added: vec![],
            removed: vec![],
            bytes_delta: 0,
        };
        assert!(diff.is_empty());
    }

    #[test]
    fn test_memory_diff_display() {
        let diff = MemoryDiff {
            added: vec![AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024)],
            removed: vec![2],
            bytes_delta: 512,
        };

        let s = format!("{}", diff);
        assert!(s.contains("+1 allocs"));
        assert!(s.contains("-1 allocs"));
    }

    // ========================================================================
    // LeakDetector Tests
    // ========================================================================

    #[test]
    fn test_leak_detector_new() {
        let detector = LeakDetector::new();
        assert_eq!(detector.expected_count(), 0);
    }

    #[test]
    fn test_leak_detector_mark_expected() {
        let mut detector = LeakDetector::new();
        detector.mark_expected(1);
        detector.mark_expected(2);

        assert_eq!(detector.expected_count(), 2);

        detector.unmark_expected(1);
        assert_eq!(detector.expected_count(), 1);
    }

    #[test]
    fn test_leak_detector_check_leaks_empty() {
        let tracker = MemoryTracker::with_default_budget();
        let detector = LeakDetector::new();

        let leaks = detector.check_leaks(&tracker, 0.0);
        assert!(leaks.is_empty());
    }

    #[test]
    fn test_leak_detector_respects_expected() {
        let mut tracker = MemoryTracker::with_default_budget();
        let mut detector = LeakDetector::new();

        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        detector.mark_expected(id);

        // Even with threshold of 0, expected allocations aren't leaks
        let leaks = detector.check_leaks(&tracker, 0.0);
        assert!(leaks.is_empty());
    }

    // ========================================================================
    // Utility Tests
    // ========================================================================

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(1536), "1.50 KB");
    }

    #[test]
    fn test_format_bytes_large() {
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }
}
