//! GPU Memory Tracker for Resource Monitoring.
//!
//! This module provides fine-grained GPU memory tracking with category-based
//! allocation monitoring, budget enforcement, and statistics collection.
//!
//! # Overview
//!
//! The memory tracker supports:
//! - **MemoryCategory**: Fine-grained resource categories (Texture, Buffer, Staging, etc.)
//! - **MemoryAllocation**: Individual allocation metadata with timestamps
//! - **MemoryStats**: Aggregate statistics with per-category breakdown
//! - **MemoryBudget**: Configurable memory budgets with per-category limits
//! - **MemoryTracker**: Main interface for tracking allocations
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::profiling::memory_tracker::{MemoryTracker, MemoryCategory, MemoryBudget, MemoryStats};
//! use std::collections::HashMap;
//!
//! // Create tracker with default settings
//! let mut tracker = MemoryTracker::new();
//!
//! // Track allocations by category
//! let vertex_id = tracker.track(MemoryCategory::Vertex, 1024 * 1024, Some("Main Vertex Buffer"));
//! let texture_id = tracker.track(MemoryCategory::Texture, 4 * 1024 * 1024, Some("Albedo Texture"));
//!
//! // Check memory stats
//! let stats = tracker.stats();
//! println!("Total usage: {} bytes", stats.total_bytes);
//! println!("Peak usage: {} bytes", stats.peak_bytes);
//!
//! // Per-category usage
//! for (category, bytes, pct) in stats.usage_by_category() {
//!     println!("{}: {} ({:.1}%)", category.name(), MemoryStats::format_bytes(bytes), pct * 100.0);
//! }
//!
//! // Untrack when resources are released
//! tracker.untrack(vertex_id);
//! ```
//!
//! # Budget Tracking
//!
//! ```no_run
//! use renderer_backend::profiling::memory_tracker::{MemoryTracker, MemoryCategory, MemoryBudget};
//! use std::collections::HashMap;
//!
//! let mut per_category = HashMap::new();
//! per_category.insert(MemoryCategory::Texture, 512 * 1024 * 1024); // 512MB for textures
//! per_category.insert(MemoryCategory::Vertex, 128 * 1024 * 1024);  // 128MB for vertices
//!
//! let budget = MemoryBudget {
//!     total_budget: 1024 * 1024 * 1024, // 1GB total
//!     per_category,
//!     warning_threshold: 0.8, // Warn at 80%
//! };
//!
//! let mut tracker = MemoryTracker::with_budget(budget);
//!
//! // Check budget status
//! println!("Budget usage: {:.1}%", tracker.budget_usage() * 100.0);
//! if tracker.is_over_budget() {
//!     eprintln!("WARNING: Over memory budget!");
//! }
//! ```

use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// MemoryCategory
// ============================================================================

/// Memory allocation category for fine-grained resource tracking.
///
/// Categories are more specific than generic resource types, allowing
/// detailed analysis of memory usage patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryCategory {
    /// 2D, 3D, or cube textures (diffuse, normal, specular, etc.).
    Texture,
    /// Generic GPU buffers.
    Buffer,
    /// Staging buffers for CPU-GPU data transfer.
    Staging,
    /// Render target textures (framebuffer attachments).
    RenderTarget,
    /// Depth/stencil buffer attachments.
    DepthStencil,
    /// Uniform buffers (constant data for shaders).
    Uniform,
    /// Storage buffers (compute shader read/write).
    Storage,
    /// Index buffers for indexed drawing.
    Index,
    /// Vertex buffers containing mesh data.
    Vertex,
    /// Uncategorized or miscellaneous resources.
    Other,
}

impl MemoryCategory {
    /// Get the human-readable name of this category.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            MemoryCategory::Texture => "Texture",
            MemoryCategory::Buffer => "Buffer",
            MemoryCategory::Staging => "Staging",
            MemoryCategory::RenderTarget => "RenderTarget",
            MemoryCategory::DepthStencil => "DepthStencil",
            MemoryCategory::Uniform => "Uniform",
            MemoryCategory::Storage => "Storage",
            MemoryCategory::Index => "Index",
            MemoryCategory::Vertex => "Vertex",
            MemoryCategory::Other => "Other",
        }
    }

    /// Check if this category is GPU-only (not accessible from CPU).
    ///
    /// GPU-only resources are typically device-local and offer best performance.
    #[must_use]
    pub fn is_gpu_only(&self) -> bool {
        match self {
            MemoryCategory::Texture => true,
            MemoryCategory::Buffer => false, // Generic buffer may be mappable
            MemoryCategory::Staging => false, // Staging is CPU-accessible by design
            MemoryCategory::RenderTarget => true,
            MemoryCategory::DepthStencil => true,
            MemoryCategory::Uniform => false, // Often updated from CPU
            MemoryCategory::Storage => false, // May be read back
            MemoryCategory::Index => true,    // Typically device-local
            MemoryCategory::Vertex => true,   // Typically device-local
            MemoryCategory::Other => false,
        }
    }

    /// Get all category variants.
    #[must_use]
    pub fn all() -> &'static [MemoryCategory] {
        &[
            MemoryCategory::Texture,
            MemoryCategory::Buffer,
            MemoryCategory::Staging,
            MemoryCategory::RenderTarget,
            MemoryCategory::DepthStencil,
            MemoryCategory::Uniform,
            MemoryCategory::Storage,
            MemoryCategory::Index,
            MemoryCategory::Vertex,
            MemoryCategory::Other,
        ]
    }
}

impl std::fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// MemoryAllocation
// ============================================================================

/// A tracked memory allocation with metadata.
#[derive(Clone, Debug)]
pub struct MemoryAllocation {
    /// Unique identifier for this allocation.
    pub id: u64,
    /// Category of the allocation.
    pub category: MemoryCategory,
    /// Size of the allocation in bytes.
    pub size_bytes: u64,
    /// Optional descriptive name for debugging.
    pub name: Option<String>,
    /// Timestamp when the allocation was created.
    pub timestamp: Instant,
}

impl MemoryAllocation {
    /// Create a new memory allocation.
    #[must_use]
    pub fn new(id: u64, category: MemoryCategory, size_bytes: u64) -> Self {
        Self {
            id,
            category,
            size_bytes,
            name: None,
            timestamp: Instant::now(),
        }
    }

    /// Create a new memory allocation with a name.
    #[must_use]
    pub fn with_name(id: u64, category: MemoryCategory, size_bytes: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            category,
            size_bytes,
            name: Some(name.into()),
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
// MemoryStats
// ============================================================================

/// Memory usage statistics with per-category breakdown.
#[derive(Clone, Debug, Default)]
pub struct MemoryStats {
    /// Total bytes currently allocated.
    pub total_bytes: u64,
    /// Peak bytes ever allocated at once.
    pub peak_bytes: u64,
    /// Number of active allocations.
    pub allocation_count: usize,
    /// Bytes allocated by category.
    pub by_category: HashMap<MemoryCategory, u64>,
}

impl MemoryStats {
    /// Create new empty statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get per-category usage as (category, bytes, percentage of total).
    #[must_use]
    pub fn usage_by_category(&self) -> Vec<(MemoryCategory, u64, f32)> {
        let mut result: Vec<_> = self
            .by_category
            .iter()
            .filter(|(_, &bytes)| bytes > 0)
            .map(|(&category, &bytes)| {
                let pct = if self.total_bytes > 0 {
                    bytes as f32 / self.total_bytes as f32
                } else {
                    0.0
                };
                (category, bytes, pct)
            })
            .collect();

        // Sort by bytes descending
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// Format bytes into human-readable string.
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

    /// Get total bytes formatted.
    #[must_use]
    pub fn total_formatted(&self) -> String {
        Self::format_bytes(self.total_bytes)
    }

    /// Get peak bytes formatted.
    #[must_use]
    pub fn peak_formatted(&self) -> String {
        Self::format_bytes(self.peak_bytes)
    }
}

// ============================================================================
// MemoryBudget
// ============================================================================

/// Memory budget configuration with per-category limits.
#[derive(Clone, Debug)]
pub struct MemoryBudget {
    /// Total memory budget in bytes.
    pub total_budget: u64,
    /// Per-category budget limits.
    pub per_category: HashMap<MemoryCategory, u64>,
    /// Warning threshold as fraction (0.0 - 1.0).
    pub warning_threshold: f32,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            total_budget: 2 * 1024 * 1024 * 1024, // 2 GB default
            per_category: HashMap::new(),
            warning_threshold: 0.8, // Warn at 80%
        }
    }
}

impl MemoryBudget {
    /// Create a new budget with total limit only.
    #[must_use]
    pub fn new(total_budget: u64) -> Self {
        Self {
            total_budget,
            per_category: HashMap::new(),
            warning_threshold: 0.8,
        }
    }

    /// Create a budget with per-category limits.
    #[must_use]
    pub fn with_categories(total_budget: u64, per_category: HashMap<MemoryCategory, u64>) -> Self {
        Self {
            total_budget,
            per_category,
            warning_threshold: 0.8,
        }
    }

    /// Set the warning threshold (0.0 - 1.0).
    pub fn set_warning_threshold(&mut self, threshold: f32) {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get the budget for a specific category, or None if not set.
    #[must_use]
    pub fn category_budget(&self, category: MemoryCategory) -> Option<u64> {
        self.per_category.get(&category).copied()
    }

    /// Check if a category is over its specific budget.
    #[must_use]
    pub fn is_category_over_budget(&self, category: MemoryCategory, used: u64) -> bool {
        if let Some(budget) = self.per_category.get(&category) {
            used > *budget
        } else {
            false
        }
    }

    /// Get category budget usage as fraction (may exceed 1.0).
    #[must_use]
    pub fn category_usage(&self, category: MemoryCategory, used: u64) -> Option<f32> {
        self.per_category.get(&category).map(|&budget| {
            if budget == 0 {
                0.0
            } else {
                used as f32 / budget as f32
            }
        })
    }
}

// ============================================================================
// MemoryTracker
// ============================================================================

/// Tracks GPU memory allocations with category-based organization.
///
/// Provides allocation tracking, statistics collection, and budget monitoring.
pub struct MemoryTracker {
    /// Active allocations indexed by ID.
    allocations: HashMap<u64, MemoryAllocation>,
    /// Next allocation ID.
    next_id: u64,
    /// Current total usage in bytes.
    current_usage: u64,
    /// Peak usage in bytes.
    peak_usage: u64,
    /// Optional memory budget.
    budget: Option<MemoryBudget>,
    /// Whether tracking is enabled.
    enabled: bool,
}

impl MemoryTracker {
    /// Create a new memory tracker without budget limits.
    #[must_use]
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            next_id: 1,
            current_usage: 0,
            peak_usage: 0,
            budget: None,
            enabled: true,
        }
    }

    /// Create a new memory tracker with budget configuration.
    #[must_use]
    pub fn with_budget(budget: MemoryBudget) -> Self {
        Self {
            allocations: HashMap::new(),
            next_id: 1,
            current_usage: 0,
            peak_usage: 0,
            budget: Some(budget),
            enabled: true,
        }
    }

    /// Track a new memory allocation.
    ///
    /// Returns the allocation ID for later reference.
    pub fn track(&mut self, category: MemoryCategory, size: u64, name: Option<&str>) -> u64 {
        if !self.enabled {
            return 0;
        }

        let id = self.next_id;
        self.next_id += 1;

        let allocation = if let Some(name) = name {
            MemoryAllocation::with_name(id, category, size, name)
        } else {
            MemoryAllocation::new(id, category, size)
        };

        self.allocations.insert(id, allocation);
        self.current_usage += size;

        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }

        id
    }

    /// Stop tracking an allocation and return it if found.
    pub fn untrack(&mut self, id: u64) -> Option<MemoryAllocation> {
        if !self.enabled {
            return None;
        }

        if let Some(allocation) = self.allocations.remove(&id) {
            self.current_usage = self.current_usage.saturating_sub(allocation.size_bytes);
            Some(allocation)
        } else {
            None
        }
    }

    /// Update the size of an existing allocation.
    ///
    /// Returns true if the allocation was found and updated.
    pub fn update(&mut self, id: u64, new_size: u64) -> bool {
        if !self.enabled {
            return false;
        }

        if let Some(allocation) = self.allocations.get_mut(&id) {
            let old_size = allocation.size_bytes;
            allocation.size_bytes = new_size;

            if new_size >= old_size {
                self.current_usage += new_size - old_size;
            } else {
                self.current_usage = self.current_usage.saturating_sub(old_size - new_size);
            }

            if self.current_usage > self.peak_usage {
                self.peak_usage = self.current_usage;
            }

            true
        } else {
            false
        }
    }

    /// Get an allocation by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&MemoryAllocation> {
        self.allocations.get(&id)
    }

    /// Get current memory statistics.
    #[must_use]
    pub fn stats(&self) -> MemoryStats {
        let mut by_category: HashMap<MemoryCategory, u64> = HashMap::new();

        for allocation in self.allocations.values() {
            *by_category.entry(allocation.category).or_insert(0) += allocation.size_bytes;
        }

        MemoryStats {
            total_bytes: self.current_usage,
            peak_bytes: self.peak_usage,
            allocation_count: self.allocations.len(),
            by_category,
        }
    }

    /// Get current memory usage in bytes.
    #[must_use]
    pub fn current_usage(&self) -> u64 {
        self.current_usage
    }

    /// Get peak memory usage in bytes.
    #[must_use]
    pub fn peak_usage(&self) -> u64 {
        self.peak_usage
    }

    /// Get all allocations of a specific category.
    #[must_use]
    pub fn allocations_by_category(&self, category: MemoryCategory) -> Vec<&MemoryAllocation> {
        self.allocations
            .values()
            .filter(|a| a.category == category)
            .collect()
    }

    /// Get the largest allocations, sorted by size descending.
    #[must_use]
    pub fn largest_allocations(&self, count: usize) -> Vec<&MemoryAllocation> {
        let mut allocations: Vec<_> = self.allocations.values().collect();
        allocations.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
        allocations.truncate(count);
        allocations
    }

    /// Check if current usage exceeds the budget.
    #[must_use]
    pub fn is_over_budget(&self) -> bool {
        if let Some(budget) = &self.budget {
            if self.current_usage > budget.total_budget {
                return true;
            }

            // Check per-category budgets
            let stats = self.stats();
            for (category, &limit) in &budget.per_category {
                if let Some(&used) = stats.by_category.get(category) {
                    if used > limit {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Get budget usage as fraction (0.0 - 1.0+).
    #[must_use]
    pub fn budget_usage(&self) -> f32 {
        if let Some(budget) = &self.budget {
            if budget.total_budget == 0 {
                return 0.0;
            }
            self.current_usage as f32 / budget.total_budget as f32
        } else {
            0.0
        }
    }

    /// Reset peak usage to current usage.
    pub fn reset_peak(&mut self) {
        self.peak_usage = self.current_usage;
    }

    /// Clear all tracked allocations.
    pub fn clear(&mut self) {
        self.allocations.clear();
        self.next_id = 1;
        self.current_usage = 0;
        // Note: peak is NOT reset by clear
    }

    /// Enable memory tracking.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable memory tracking.
    ///
    /// When disabled, track/untrack/update operations are no-ops.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if tracking is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the memory budget, if configured.
    #[must_use]
    pub fn budget(&self) -> Option<&MemoryBudget> {
        self.budget.as_ref()
    }

    /// Set or replace the memory budget.
    pub fn set_budget(&mut self, budget: MemoryBudget) {
        self.budget = Some(budget);
    }

    /// Remove the memory budget.
    pub fn clear_budget(&mut self) {
        self.budget = None;
    }

    /// Get all allocations.
    #[must_use]
    pub fn allocations(&self) -> &HashMap<u64, MemoryAllocation> {
        &self.allocations
    }

    /// Get the number of active allocations.
    #[must_use]
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Generate a summary report of memory usage.
    #[must_use]
    pub fn summary(&self) -> String {
        let stats = self.stats();
        let mut s = String::new();

        s.push_str("=== Memory Tracker Summary ===\n");
        s.push_str(&format!(
            "Current: {} ({} allocations)\n",
            stats.total_formatted(),
            stats.allocation_count
        ));
        s.push_str(&format!("Peak: {}\n", stats.peak_formatted()));

        if let Some(budget) = &self.budget {
            s.push_str(&format!(
                "Budget: {:.1}% of {}\n",
                self.budget_usage() * 100.0,
                MemoryStats::format_bytes(budget.total_budget)
            ));

            if self.is_over_budget() {
                s.push_str("WARNING: Over budget!\n");
            }
        }

        s.push_str("\nBy Category:\n");
        for (category, bytes, pct) in stats.usage_by_category() {
            s.push_str(&format!(
                "  {}: {} ({:.1}%)\n",
                category.name(),
                MemoryStats::format_bytes(bytes),
                pct * 100.0
            ));
        }

        s
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MemoryTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryTracker")
            .field("allocation_count", &self.allocations.len())
            .field("current_usage", &self.current_usage)
            .field("peak_usage", &self.peak_usage)
            .field("enabled", &self.enabled)
            .field("has_budget", &self.budget.is_some())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // MemoryCategory Tests
    // ========================================================================

    #[test]
    fn test_memory_category_name() {
        assert_eq!(MemoryCategory::Texture.name(), "Texture");
        assert_eq!(MemoryCategory::Buffer.name(), "Buffer");
        assert_eq!(MemoryCategory::Staging.name(), "Staging");
        assert_eq!(MemoryCategory::RenderTarget.name(), "RenderTarget");
        assert_eq!(MemoryCategory::DepthStencil.name(), "DepthStencil");
        assert_eq!(MemoryCategory::Uniform.name(), "Uniform");
        assert_eq!(MemoryCategory::Storage.name(), "Storage");
        assert_eq!(MemoryCategory::Index.name(), "Index");
        assert_eq!(MemoryCategory::Vertex.name(), "Vertex");
        assert_eq!(MemoryCategory::Other.name(), "Other");
    }

    #[test]
    fn test_memory_category_is_gpu_only() {
        assert!(MemoryCategory::Texture.is_gpu_only());
        assert!(!MemoryCategory::Buffer.is_gpu_only());
        assert!(!MemoryCategory::Staging.is_gpu_only());
        assert!(MemoryCategory::RenderTarget.is_gpu_only());
        assert!(MemoryCategory::DepthStencil.is_gpu_only());
        assert!(!MemoryCategory::Uniform.is_gpu_only());
        assert!(!MemoryCategory::Storage.is_gpu_only());
        assert!(MemoryCategory::Index.is_gpu_only());
        assert!(MemoryCategory::Vertex.is_gpu_only());
        assert!(!MemoryCategory::Other.is_gpu_only());
    }

    #[test]
    fn test_memory_category_all() {
        let all = MemoryCategory::all();
        assert_eq!(all.len(), 10);
        assert!(all.contains(&MemoryCategory::Texture));
        assert!(all.contains(&MemoryCategory::Vertex));
    }

    #[test]
    fn test_memory_category_display() {
        assert_eq!(format!("{}", MemoryCategory::Texture), "Texture");
        assert_eq!(format!("{}", MemoryCategory::DepthStencil), "DepthStencil");
    }

    #[test]
    fn test_memory_category_clone_copy() {
        let cat = MemoryCategory::Texture;
        let cat2 = cat;
        let cat3 = cat.clone();
        assert_eq!(cat, cat2);
        assert_eq!(cat, cat3);
    }

    #[test]
    fn test_memory_category_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MemoryCategory::Texture);
        set.insert(MemoryCategory::Buffer);
        set.insert(MemoryCategory::Texture); // Duplicate
        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // MemoryAllocation Tests
    // ========================================================================

    #[test]
    fn test_memory_allocation_new() {
        let alloc = MemoryAllocation::new(1, MemoryCategory::Texture, 4096);
        assert_eq!(alloc.id, 1);
        assert_eq!(alloc.category, MemoryCategory::Texture);
        assert_eq!(alloc.size_bytes, 4096);
        assert!(alloc.name.is_none());
    }

    #[test]
    fn test_memory_allocation_with_name() {
        let alloc = MemoryAllocation::with_name(2, MemoryCategory::Vertex, 1024, "Main VBO");
        assert_eq!(alloc.id, 2);
        assert_eq!(alloc.category, MemoryCategory::Vertex);
        assert_eq!(alloc.size_bytes, 1024);
        assert_eq!(alloc.name, Some("Main VBO".to_string()));
    }

    #[test]
    fn test_memory_allocation_age() {
        let alloc = MemoryAllocation::new(1, MemoryCategory::Buffer, 1024);
        // Just created, age should be near zero
        assert!(alloc.age_secs() < 1.0);
    }

    #[test]
    fn test_memory_allocation_clone() {
        let alloc = MemoryAllocation::with_name(1, MemoryCategory::Texture, 2048, "Test");
        let cloned = alloc.clone();
        assert_eq!(cloned.id, alloc.id);
        assert_eq!(cloned.size_bytes, alloc.size_bytes);
        assert_eq!(cloned.name, alloc.name);
    }

    // ========================================================================
    // MemoryStats Tests
    // ========================================================================

    #[test]
    fn test_memory_stats_new() {
        let stats = MemoryStats::new();
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.peak_bytes, 0);
        assert_eq!(stats.allocation_count, 0);
        assert!(stats.by_category.is_empty());
    }

    #[test]
    fn test_memory_stats_format_bytes() {
        assert_eq!(MemoryStats::format_bytes(0), "0 B");
        assert_eq!(MemoryStats::format_bytes(512), "512 B");
        assert_eq!(MemoryStats::format_bytes(1024), "1.00 KB");
        assert_eq!(MemoryStats::format_bytes(1536), "1.50 KB");
        assert_eq!(MemoryStats::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(MemoryStats::format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_memory_stats_usage_by_category() {
        let mut by_category = HashMap::new();
        by_category.insert(MemoryCategory::Texture, 3000);
        by_category.insert(MemoryCategory::Vertex, 1000);

        let stats = MemoryStats {
            total_bytes: 4000,
            peak_bytes: 4000,
            allocation_count: 2,
            by_category,
        };

        let usage = stats.usage_by_category();
        assert_eq!(usage.len(), 2);
        // Should be sorted by bytes descending
        assert_eq!(usage[0].0, MemoryCategory::Texture);
        assert_eq!(usage[0].1, 3000);
        assert!((usage[0].2 - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_memory_stats_usage_by_category_empty() {
        let stats = MemoryStats::new();
        let usage = stats.usage_by_category();
        assert!(usage.is_empty());
    }

    #[test]
    fn test_memory_stats_formatted() {
        let stats = MemoryStats {
            total_bytes: 1024 * 1024,
            peak_bytes: 2 * 1024 * 1024,
            allocation_count: 1,
            by_category: HashMap::new(),
        };

        assert_eq!(stats.total_formatted(), "1.00 MB");
        assert_eq!(stats.peak_formatted(), "2.00 MB");
    }

    // ========================================================================
    // MemoryBudget Tests
    // ========================================================================

    #[test]
    fn test_memory_budget_default() {
        let budget = MemoryBudget::default();
        assert_eq!(budget.total_budget, 2 * 1024 * 1024 * 1024);
        assert!((budget.warning_threshold - 0.8).abs() < 0.001);
        assert!(budget.per_category.is_empty());
    }

    #[test]
    fn test_memory_budget_new() {
        let budget = MemoryBudget::new(1024 * 1024 * 1024);
        assert_eq!(budget.total_budget, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_budget_with_categories() {
        let mut per_cat = HashMap::new();
        per_cat.insert(MemoryCategory::Texture, 512 * 1024 * 1024);
        per_cat.insert(MemoryCategory::Vertex, 128 * 1024 * 1024);

        let budget = MemoryBudget::with_categories(1024 * 1024 * 1024, per_cat);
        assert_eq!(budget.category_budget(MemoryCategory::Texture), Some(512 * 1024 * 1024));
        assert_eq!(budget.category_budget(MemoryCategory::Vertex), Some(128 * 1024 * 1024));
        assert_eq!(budget.category_budget(MemoryCategory::Buffer), None);
    }

    #[test]
    fn test_memory_budget_set_warning_threshold() {
        let mut budget = MemoryBudget::default();
        budget.set_warning_threshold(0.9);
        assert!((budget.warning_threshold - 0.9).abs() < 0.001);

        // Clamp test
        budget.set_warning_threshold(1.5);
        assert!((budget.warning_threshold - 1.0).abs() < 0.001);

        budget.set_warning_threshold(-0.5);
        assert!((budget.warning_threshold - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_budget_is_category_over_budget() {
        let mut per_cat = HashMap::new();
        per_cat.insert(MemoryCategory::Texture, 1000);

        let budget = MemoryBudget::with_categories(10000, per_cat);

        assert!(!budget.is_category_over_budget(MemoryCategory::Texture, 500));
        assert!(!budget.is_category_over_budget(MemoryCategory::Texture, 1000));
        assert!(budget.is_category_over_budget(MemoryCategory::Texture, 1001));
        // No budget set for Vertex
        assert!(!budget.is_category_over_budget(MemoryCategory::Vertex, 999999));
    }

    #[test]
    fn test_memory_budget_category_usage() {
        let mut per_cat = HashMap::new();
        per_cat.insert(MemoryCategory::Texture, 1000);

        let budget = MemoryBudget::with_categories(10000, per_cat);

        assert_eq!(budget.category_usage(MemoryCategory::Texture, 500), Some(0.5));
        assert_eq!(budget.category_usage(MemoryCategory::Vertex, 500), None);
    }

    // ========================================================================
    // MemoryTracker Basic Tests
    // ========================================================================

    #[test]
    fn test_memory_tracker_new() {
        let tracker = MemoryTracker::new();
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.peak_usage(), 0);
        assert_eq!(tracker.allocation_count(), 0);
        assert!(tracker.is_enabled());
        assert!(tracker.budget().is_none());
    }

    #[test]
    fn test_memory_tracker_with_budget() {
        let budget = MemoryBudget::new(1000);
        let tracker = MemoryTracker::with_budget(budget);
        assert!(tracker.budget().is_some());
        assert_eq!(tracker.budget().unwrap().total_budget, 1000);
    }

    #[test]
    fn test_memory_tracker_track() {
        let mut tracker = MemoryTracker::new();

        let id = tracker.track(MemoryCategory::Texture, 1024, Some("Test Texture"));

        assert!(id > 0);
        assert_eq!(tracker.current_usage(), 1024);
        assert_eq!(tracker.allocation_count(), 1);

        let alloc = tracker.get(id).unwrap();
        assert_eq!(alloc.category, MemoryCategory::Texture);
        assert_eq!(alloc.size_bytes, 1024);
        assert_eq!(alloc.name, Some("Test Texture".to_string()));
    }

    #[test]
    fn test_memory_tracker_track_no_name() {
        let mut tracker = MemoryTracker::new();
        let id = tracker.track(MemoryCategory::Buffer, 512, None);

        let alloc = tracker.get(id).unwrap();
        assert!(alloc.name.is_none());
    }

    #[test]
    fn test_memory_tracker_untrack() {
        let mut tracker = MemoryTracker::new();
        let id = tracker.track(MemoryCategory::Vertex, 2048, None);

        assert_eq!(tracker.current_usage(), 2048);

        let removed = tracker.untrack(id);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().size_bytes, 2048);
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.allocation_count(), 0);

        // Untracking again returns None
        assert!(tracker.untrack(id).is_none());
    }

    #[test]
    fn test_memory_tracker_update() {
        let mut tracker = MemoryTracker::new();
        let id = tracker.track(MemoryCategory::Buffer, 1000, None);

        // Increase size
        assert!(tracker.update(id, 2000));
        assert_eq!(tracker.current_usage(), 2000);
        assert_eq!(tracker.get(id).unwrap().size_bytes, 2000);

        // Decrease size
        assert!(tracker.update(id, 500));
        assert_eq!(tracker.current_usage(), 500);

        // Update non-existent
        assert!(!tracker.update(999, 100));
    }

    #[test]
    fn test_memory_tracker_update_underflow_protection() {
        let mut tracker = MemoryTracker::new();
        let id1 = tracker.track(MemoryCategory::Buffer, 1000, None);
        let id2 = tracker.track(MemoryCategory::Buffer, 500, None);

        assert_eq!(tracker.current_usage(), 1500);

        // Even if we shrink by more than the total, it shouldn't underflow
        tracker.update(id1, 0);
        assert_eq!(tracker.current_usage(), 500);

        tracker.untrack(id2);
        assert_eq!(tracker.current_usage(), 0);
    }

    // ========================================================================
    // MemoryTracker Stats Tests
    // ========================================================================

    #[test]
    fn test_memory_tracker_stats() {
        let mut tracker = MemoryTracker::new();
        tracker.track(MemoryCategory::Texture, 1000, None);
        tracker.track(MemoryCategory::Texture, 500, None);
        tracker.track(MemoryCategory::Vertex, 2000, None);

        let stats = tracker.stats();
        assert_eq!(stats.total_bytes, 3500);
        assert_eq!(stats.allocation_count, 3);
        assert_eq!(stats.by_category.get(&MemoryCategory::Texture), Some(&1500));
        assert_eq!(stats.by_category.get(&MemoryCategory::Vertex), Some(&2000));
    }

    #[test]
    fn test_memory_tracker_peak_tracking() {
        let mut tracker = MemoryTracker::new();

        let id1 = tracker.track(MemoryCategory::Buffer, 1000, None);
        let id2 = tracker.track(MemoryCategory::Buffer, 2000, None);
        assert_eq!(tracker.peak_usage(), 3000);

        tracker.untrack(id1);
        assert_eq!(tracker.current_usage(), 2000);
        assert_eq!(tracker.peak_usage(), 3000); // Peak unchanged

        tracker.untrack(id2);
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.peak_usage(), 3000); // Peak still unchanged
    }

    #[test]
    fn test_memory_tracker_reset_peak() {
        let mut tracker = MemoryTracker::new();
        let id = tracker.track(MemoryCategory::Buffer, 5000, None);
        tracker.untrack(id);

        assert_eq!(tracker.peak_usage(), 5000);

        tracker.reset_peak();
        assert_eq!(tracker.peak_usage(), 0);

        // After new allocation, peak should track new value
        tracker.track(MemoryCategory::Buffer, 1000, None);
        assert_eq!(tracker.peak_usage(), 1000);
    }

    // ========================================================================
    // MemoryTracker Budget Tests
    // ========================================================================

    #[test]
    fn test_memory_tracker_is_over_budget_total() {
        let budget = MemoryBudget::new(1000);
        let mut tracker = MemoryTracker::with_budget(budget);

        tracker.track(MemoryCategory::Buffer, 500, None);
        assert!(!tracker.is_over_budget());

        tracker.track(MemoryCategory::Buffer, 600, None);
        assert!(tracker.is_over_budget());
    }

    #[test]
    fn test_memory_tracker_is_over_budget_category() {
        let mut per_cat = HashMap::new();
        per_cat.insert(MemoryCategory::Texture, 500);

        let budget = MemoryBudget::with_categories(10000, per_cat);
        let mut tracker = MemoryTracker::with_budget(budget);

        tracker.track(MemoryCategory::Buffer, 5000, None);
        assert!(!tracker.is_over_budget());

        tracker.track(MemoryCategory::Texture, 600, None);
        assert!(tracker.is_over_budget());
    }

    #[test]
    fn test_memory_tracker_budget_usage() {
        let budget = MemoryBudget::new(1000);
        let mut tracker = MemoryTracker::with_budget(budget);

        assert!((tracker.budget_usage() - 0.0).abs() < 0.001);

        tracker.track(MemoryCategory::Buffer, 500, None);
        assert!((tracker.budget_usage() - 0.5).abs() < 0.001);

        tracker.track(MemoryCategory::Buffer, 500, None);
        assert!((tracker.budget_usage() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_tracker_budget_usage_no_budget() {
        let tracker = MemoryTracker::new();
        assert_eq!(tracker.budget_usage(), 0.0);
    }

    // ========================================================================
    // MemoryTracker Filter/Query Tests
    // ========================================================================

    #[test]
    fn test_memory_tracker_allocations_by_category() {
        let mut tracker = MemoryTracker::new();
        tracker.track(MemoryCategory::Texture, 1000, Some("Tex1"));
        tracker.track(MemoryCategory::Texture, 2000, Some("Tex2"));
        tracker.track(MemoryCategory::Vertex, 500, Some("Vert"));

        let textures = tracker.allocations_by_category(MemoryCategory::Texture);
        assert_eq!(textures.len(), 2);

        let vertices = tracker.allocations_by_category(MemoryCategory::Vertex);
        assert_eq!(vertices.len(), 1);

        let buffers = tracker.allocations_by_category(MemoryCategory::Buffer);
        assert!(buffers.is_empty());
    }

    #[test]
    fn test_memory_tracker_largest_allocations() {
        let mut tracker = MemoryTracker::new();
        tracker.track(MemoryCategory::Buffer, 100, Some("Small"));
        tracker.track(MemoryCategory::Buffer, 1000, Some("Medium"));
        tracker.track(MemoryCategory::Buffer, 10000, Some("Large"));
        tracker.track(MemoryCategory::Buffer, 500, Some("MedSmall"));

        let largest = tracker.largest_allocations(2);
        assert_eq!(largest.len(), 2);
        assert_eq!(largest[0].size_bytes, 10000);
        assert_eq!(largest[1].size_bytes, 1000);

        let all = tracker.largest_allocations(10);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_memory_tracker_largest_allocations_empty() {
        let tracker = MemoryTracker::new();
        let largest = tracker.largest_allocations(5);
        assert!(largest.is_empty());
    }

    // ========================================================================
    // MemoryTracker Enable/Disable Tests
    // ========================================================================

    #[test]
    fn test_memory_tracker_enable_disable() {
        let mut tracker = MemoryTracker::new();
        assert!(tracker.is_enabled());

        tracker.disable();
        assert!(!tracker.is_enabled());

        tracker.enable();
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_memory_tracker_disabled_track() {
        let mut tracker = MemoryTracker::new();
        tracker.disable();

        let id = tracker.track(MemoryCategory::Buffer, 1000, None);
        assert_eq!(id, 0);
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.allocation_count(), 0);
    }

    #[test]
    fn test_memory_tracker_disabled_untrack() {
        let mut tracker = MemoryTracker::new();
        let id = tracker.track(MemoryCategory::Buffer, 1000, None);

        tracker.disable();
        let removed = tracker.untrack(id);
        assert!(removed.is_none());
        // Allocation still exists (disabled doesn't remove)
        assert_eq!(tracker.allocation_count(), 1);
    }

    #[test]
    fn test_memory_tracker_disabled_update() {
        let mut tracker = MemoryTracker::new();
        let id = tracker.track(MemoryCategory::Buffer, 1000, None);

        tracker.disable();
        assert!(!tracker.update(id, 2000));
        // Size unchanged
        assert_eq!(tracker.current_usage(), 1000);
    }

    // ========================================================================
    // MemoryTracker Clear Tests
    // ========================================================================

    #[test]
    fn test_memory_tracker_clear() {
        let mut tracker = MemoryTracker::new();
        tracker.track(MemoryCategory::Texture, 1000, None);
        tracker.track(MemoryCategory::Buffer, 2000, None);

        assert_eq!(tracker.peak_usage(), 3000);

        tracker.clear();

        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.allocation_count(), 0);
        // Peak is NOT cleared by clear()
        assert_eq!(tracker.peak_usage(), 3000);
    }

    #[test]
    fn test_memory_tracker_clear_resets_id() {
        let mut tracker = MemoryTracker::new();
        tracker.track(MemoryCategory::Buffer, 100, None);
        tracker.track(MemoryCategory::Buffer, 100, None);

        tracker.clear();

        // After clear, IDs start fresh
        let id = tracker.track(MemoryCategory::Buffer, 100, None);
        assert_eq!(id, 1);
    }

    // ========================================================================
    // MemoryTracker Budget Management Tests
    // ========================================================================

    #[test]
    fn test_memory_tracker_set_budget() {
        let mut tracker = MemoryTracker::new();
        assert!(tracker.budget().is_none());

        tracker.set_budget(MemoryBudget::new(5000));
        assert!(tracker.budget().is_some());
        assert_eq!(tracker.budget().unwrap().total_budget, 5000);
    }

    #[test]
    fn test_memory_tracker_clear_budget() {
        let mut tracker = MemoryTracker::with_budget(MemoryBudget::new(5000));
        assert!(tracker.budget().is_some());

        tracker.clear_budget();
        assert!(tracker.budget().is_none());
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_memory_tracker_zero_size_allocation() {
        let mut tracker = MemoryTracker::new();
        let id = tracker.track(MemoryCategory::Buffer, 0, None);

        assert!(id > 0);
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.allocation_count(), 1);
    }

    #[test]
    fn test_memory_tracker_large_allocation() {
        let mut tracker = MemoryTracker::new();
        let large_size = 16 * 1024 * 1024 * 1024u64; // 16GB
        let id = tracker.track(MemoryCategory::Texture, large_size, None);

        assert_eq!(tracker.current_usage(), large_size);
        assert_eq!(tracker.get(id).unwrap().size_bytes, large_size);
    }

    #[test]
    fn test_memory_tracker_many_allocations() {
        let mut tracker = MemoryTracker::new();
        let count = 1000;

        for i in 0..count {
            tracker.track(MemoryCategory::Buffer, i as u64, None);
        }

        assert_eq!(tracker.allocation_count(), count);
        // Sum of 0..999 = 499500
        assert_eq!(tracker.current_usage(), 499500);
    }

    #[test]
    fn test_memory_tracker_get_nonexistent() {
        let tracker = MemoryTracker::new();
        assert!(tracker.get(999).is_none());
    }

    #[test]
    fn test_memory_tracker_summary() {
        let budget = MemoryBudget::new(10000);
        let mut tracker = MemoryTracker::with_budget(budget);
        tracker.track(MemoryCategory::Texture, 2000, Some("Tex"));
        tracker.track(MemoryCategory::Vertex, 1000, Some("Vert"));

        let summary = tracker.summary();
        assert!(summary.contains("Memory Tracker Summary"));
        assert!(summary.contains("Texture"));
        assert!(summary.contains("Vertex"));
        assert!(summary.contains("Budget"));
    }

    #[test]
    fn test_memory_tracker_summary_over_budget() {
        let budget = MemoryBudget::new(100);
        let mut tracker = MemoryTracker::with_budget(budget);
        tracker.track(MemoryCategory::Buffer, 200, None);

        let summary = tracker.summary();
        assert!(summary.contains("WARNING: Over budget!"));
    }

    #[test]
    fn test_memory_tracker_debug() {
        let tracker = MemoryTracker::new();
        let debug = format!("{:?}", tracker);
        assert!(debug.contains("MemoryTracker"));
        assert!(debug.contains("allocation_count"));
        assert!(debug.contains("enabled"));
    }

    #[test]
    fn test_memory_tracker_default() {
        let tracker = MemoryTracker::default();
        assert!(tracker.is_enabled());
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn test_memory_tracker_allocations_accessor() {
        let mut tracker = MemoryTracker::new();
        let id1 = tracker.track(MemoryCategory::Buffer, 100, None);
        let id2 = tracker.track(MemoryCategory::Texture, 200, None);

        let allocs = tracker.allocations();
        assert_eq!(allocs.len(), 2);
        assert!(allocs.contains_key(&id1));
        assert!(allocs.contains_key(&id2));
    }
}
