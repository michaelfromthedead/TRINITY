//! Budget-Aware LOD Selection System
//!
//! This module implements memory-budget-aware LOD selection that considers
//! per-category budgets (mesh, texture, shader) and global limits to select
//! optimal LOD levels for visible meshes.
//!
//! # Features
//!
//! - Per-category memory budgets (mesh, texture, shader)
//! - Global memory budget cap
//! - Priority-based LOD assignment using distance, screen-space size, and importance
//! - Automatic LOD degradation when budget is exceeded
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::budget::{BudgetConfig, BudgetTracker, BudgetLodSelector, MeshInstance};
//!
//! let config = BudgetConfig::default();
//! let tracker = BudgetTracker::new(config);
//! let selector = BudgetLodSelector::new(tracker);
//!
//! let instances = vec![
//!     MeshInstance {
//!         mesh_id: 0,
//!         lod_sizes: vec![1_000_000, 500_000, 100_000],
//!         priority: 0.9,
//!     },
//! ];
//!
//! let assignments = selector.select_lods(&instances);
//! ```

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default mesh memory budget (512 MB).
pub const DEFAULT_MESH_BUDGET: usize = 536_870_912;

/// Default texture memory budget (1 GB).
pub const DEFAULT_TEXTURE_BUDGET: usize = 1_073_741_824;

/// Default shader memory budget (256 MB).
pub const DEFAULT_SHADER_BUDGET: usize = 268_435_456;

/// Default global memory budget (2 GB).
pub const DEFAULT_GLOBAL_BUDGET: usize = 2_147_483_648;

// ---------------------------------------------------------------------------
// Budget Configuration
// ---------------------------------------------------------------------------

/// Configuration for memory budget limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetConfig {
    /// Maximum memory for mesh data (vertices, indices).
    pub mesh_budget: usize,
    /// Maximum memory for texture data.
    pub texture_budget: usize,
    /// Maximum memory for shader programs.
    pub shader_budget: usize,
    /// Maximum total memory across all categories.
    pub global_budget: usize,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            mesh_budget: DEFAULT_MESH_BUDGET,
            texture_budget: DEFAULT_TEXTURE_BUDGET,
            shader_budget: DEFAULT_SHADER_BUDGET,
            global_budget: DEFAULT_GLOBAL_BUDGET,
        }
    }
}

impl BudgetConfig {
    /// Create a new budget configuration with custom values.
    pub fn new(
        mesh_budget: usize,
        texture_budget: usize,
        shader_budget: usize,
        global_budget: usize,
    ) -> Self {
        Self {
            mesh_budget,
            texture_budget,
            shader_budget,
            global_budget,
        }
    }

    /// Create a scaled version of the default config.
    ///
    /// Useful for devices with different memory capacities.
    /// A scale of 0.5 would halve all budgets.
    pub fn scaled(scale: f32) -> Self {
        Self {
            mesh_budget: (DEFAULT_MESH_BUDGET as f64 * scale as f64) as usize,
            texture_budget: (DEFAULT_TEXTURE_BUDGET as f64 * scale as f64) as usize,
            shader_budget: (DEFAULT_SHADER_BUDGET as f64 * scale as f64) as usize,
            global_budget: (DEFAULT_GLOBAL_BUDGET as f64 * scale as f64) as usize,
        }
    }

    /// Get the total budget across all categories.
    pub fn total_category_budget(&self) -> usize {
        self.mesh_budget
            .saturating_add(self.texture_budget)
            .saturating_add(self.shader_budget)
    }
}

// ---------------------------------------------------------------------------
// Usage Snapshot
// ---------------------------------------------------------------------------

/// Current memory usage snapshot across all categories.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UsageSnapshot {
    /// Current mesh memory usage in bytes.
    pub mesh_usage: usize,
    /// Current texture memory usage in bytes.
    pub texture_usage: usize,
    /// Current shader memory usage in bytes.
    pub shader_usage: usize,
}

impl UsageSnapshot {
    /// Create a new usage snapshot.
    pub fn new(mesh_usage: usize, texture_usage: usize, shader_usage: usize) -> Self {
        Self {
            mesh_usage,
            texture_usage,
            shader_usage,
        }
    }

    /// Get total memory usage across all categories.
    pub fn total(&self) -> usize {
        self.mesh_usage
            .saturating_add(self.texture_usage)
            .saturating_add(self.shader_usage)
    }

    /// Check if any category exceeds its budget.
    pub fn exceeds_any(&self, config: &BudgetConfig) -> bool {
        self.mesh_usage > config.mesh_budget
            || self.texture_usage > config.texture_budget
            || self.shader_usage > config.shader_budget
    }

    /// Check if total usage exceeds global budget.
    pub fn exceeds_global(&self, config: &BudgetConfig) -> bool {
        self.total() > config.global_budget
    }

    /// Get remaining mesh budget.
    pub fn remaining_mesh(&self, config: &BudgetConfig) -> usize {
        config.mesh_budget.saturating_sub(self.mesh_usage)
    }

    /// Get remaining texture budget.
    pub fn remaining_texture(&self, config: &BudgetConfig) -> usize {
        config.texture_budget.saturating_sub(self.texture_usage)
    }

    /// Get remaining shader budget.
    pub fn remaining_shader(&self, config: &BudgetConfig) -> usize {
        config.shader_budget.saturating_sub(self.shader_usage)
    }

    /// Get remaining global budget.
    pub fn remaining_global(&self, config: &BudgetConfig) -> usize {
        config.global_budget.saturating_sub(self.total())
    }
}

// ---------------------------------------------------------------------------
// Budget Tracker
// ---------------------------------------------------------------------------

/// Tracks memory budget usage and allocations.
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    /// Budget configuration.
    config: BudgetConfig,
    /// Current usage snapshot.
    usage: UsageSnapshot,
}

impl BudgetTracker {
    /// Create a new budget tracker with the given configuration.
    pub fn new(config: BudgetConfig) -> Self {
        Self {
            config,
            usage: UsageSnapshot::default(),
        }
    }

    /// Create a new budget tracker with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BudgetConfig::default())
    }

    /// Get the budget configuration.
    pub fn config(&self) -> &BudgetConfig {
        &self.config
    }

    /// Get the current usage snapshot.
    pub fn usage(&self) -> &UsageSnapshot {
        &self.usage
    }

    /// Get remaining mesh budget.
    pub fn remaining_mesh_budget(&self) -> usize {
        self.usage.remaining_mesh(&self.config)
    }

    /// Get remaining texture budget.
    pub fn remaining_texture_budget(&self) -> usize {
        self.usage.remaining_texture(&self.config)
    }

    /// Get remaining shader budget.
    pub fn remaining_shader_budget(&self) -> usize {
        self.usage.remaining_shader(&self.config)
    }

    /// Get remaining global budget.
    pub fn remaining_global_budget(&self) -> usize {
        self.usage.remaining_global(&self.config)
    }

    /// Check if mesh allocation would fit in budget.
    pub fn can_allocate_mesh(&self, size: usize) -> bool {
        let new_mesh_usage = self.usage.mesh_usage.saturating_add(size);
        let new_total = self.usage.total().saturating_add(size);

        new_mesh_usage <= self.config.mesh_budget && new_total <= self.config.global_budget
    }

    /// Check if texture allocation would fit in budget.
    pub fn can_allocate_texture(&self, size: usize) -> bool {
        let new_texture_usage = self.usage.texture_usage.saturating_add(size);
        let new_total = self.usage.total().saturating_add(size);

        new_texture_usage <= self.config.texture_budget && new_total <= self.config.global_budget
    }

    /// Check if shader allocation would fit in budget.
    pub fn can_allocate_shader(&self, size: usize) -> bool {
        let new_shader_usage = self.usage.shader_usage.saturating_add(size);
        let new_total = self.usage.total().saturating_add(size);

        new_shader_usage <= self.config.shader_budget && new_total <= self.config.global_budget
    }

    /// Try to allocate mesh memory. Returns true if successful.
    pub fn try_allocate_mesh(&mut self, size: usize) -> bool {
        if self.can_allocate_mesh(size) {
            self.usage.mesh_usage = self.usage.mesh_usage.saturating_add(size);
            true
        } else {
            false
        }
    }

    /// Try to allocate texture memory. Returns true if successful.
    pub fn try_allocate_texture(&mut self, size: usize) -> bool {
        if self.can_allocate_texture(size) {
            self.usage.texture_usage = self.usage.texture_usage.saturating_add(size);
            true
        } else {
            false
        }
    }

    /// Try to allocate shader memory. Returns true if successful.
    pub fn try_allocate_shader(&mut self, size: usize) -> bool {
        if self.can_allocate_shader(size) {
            self.usage.shader_usage = self.usage.shader_usage.saturating_add(size);
            true
        } else {
            false
        }
    }

    /// Free mesh memory.
    pub fn free_mesh(&mut self, size: usize) {
        self.usage.mesh_usage = self.usage.mesh_usage.saturating_sub(size);
    }

    /// Free texture memory.
    pub fn free_texture(&mut self, size: usize) {
        self.usage.texture_usage = self.usage.texture_usage.saturating_sub(size);
    }

    /// Free shader memory.
    pub fn free_shader(&mut self, size: usize) {
        self.usage.shader_usage = self.usage.shader_usage.saturating_sub(size);
    }

    /// Reset all usage counters to zero.
    pub fn reset(&mut self) {
        self.usage = UsageSnapshot::default();
    }

    /// Check if any budget category is exceeded.
    pub fn is_over_budget(&self) -> bool {
        self.usage.exceeds_any(&self.config) || self.usage.exceeds_global(&self.config)
    }

    /// Get budget utilization as a percentage (0.0 to 1.0+).
    pub fn utilization(&self) -> f32 {
        if self.config.global_budget == 0 {
            0.0
        } else {
            self.usage.total() as f32 / self.config.global_budget as f32
        }
    }

    /// Get mesh budget utilization as a percentage.
    pub fn mesh_utilization(&self) -> f32 {
        if self.config.mesh_budget == 0 {
            0.0
        } else {
            self.usage.mesh_usage as f32 / self.config.mesh_budget as f32
        }
    }

    /// Get texture budget utilization as a percentage.
    pub fn texture_utilization(&self) -> f32 {
        if self.config.texture_budget == 0 {
            0.0
        } else {
            self.usage.texture_usage as f32 / self.config.texture_budget as f32
        }
    }

    /// Get shader budget utilization as a percentage.
    pub fn shader_utilization(&self) -> f32 {
        if self.config.shader_budget == 0 {
            0.0
        } else {
            self.usage.shader_usage as f32 / self.config.shader_budget as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh Instance
// ---------------------------------------------------------------------------

/// Visibility state of a mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisibilityState {
    /// Mesh is visible and should be rendered.
    #[default]
    Visible,
    /// Mesh is occluded by other geometry.
    Occluded,
    /// Mesh is outside the view frustum.
    OutOfFrustum,
    /// Mesh is culled due to distance.
    DistanceCulled,
}

impl VisibilityState {
    /// Check if the mesh should be considered for rendering.
    pub fn is_visible(&self) -> bool {
        matches!(self, Self::Visible)
    }
}

/// A mesh instance for LOD selection.
#[derive(Debug, Clone)]
pub struct MeshInstance {
    /// Unique identifier for this mesh.
    pub mesh_id: u32,
    /// Memory size in bytes for each LOD level (index 0 = highest detail).
    pub lod_sizes: Vec<usize>,
    /// Computed priority for this mesh (higher = more important).
    pub priority: f32,
}

impl MeshInstance {
    /// Create a new mesh instance.
    pub fn new(mesh_id: u32, lod_sizes: Vec<usize>, priority: f32) -> Self {
        Self {
            mesh_id,
            lod_sizes,
            priority,
        }
    }

    /// Get the number of available LOD levels.
    pub fn lod_count(&self) -> usize {
        self.lod_sizes.len()
    }

    /// Get the memory size for a specific LOD level.
    pub fn lod_size(&self, level: u32) -> Option<usize> {
        self.lod_sizes.get(level as usize).copied()
    }

    /// Get the highest detail LOD size.
    pub fn highest_detail_size(&self) -> usize {
        self.lod_sizes.first().copied().unwrap_or(0)
    }

    /// Get the lowest detail LOD size.
    pub fn lowest_detail_size(&self) -> usize {
        self.lod_sizes.last().copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Mesh Priority Computation
// ---------------------------------------------------------------------------

/// Parameters for computing mesh priority.
#[derive(Debug, Clone, Copy)]
pub struct PriorityParams {
    /// Distance from camera to mesh center.
    pub distance: f32,
    /// Screen-space size in pixels.
    pub screen_size: f32,
    /// Importance/weight factor (0.0 to 1.0+).
    pub importance: f32,
    /// Visibility state.
    pub visibility: VisibilityState,
}

impl PriorityParams {
    /// Create new priority parameters.
    pub fn new(distance: f32, screen_size: f32, importance: f32, visibility: VisibilityState) -> Self {
        Self {
            distance,
            screen_size,
            importance,
            visibility,
        }
    }

    /// Create parameters for a visible mesh.
    pub fn visible(distance: f32, screen_size: f32, importance: f32) -> Self {
        Self::new(distance, screen_size, importance, VisibilityState::Visible)
    }
}

impl Default for PriorityParams {
    fn default() -> Self {
        Self {
            distance: 0.0,
            screen_size: 0.0,
            importance: 1.0,
            visibility: VisibilityState::Visible,
        }
    }
}

/// Compute priority for a mesh based on view parameters.
///
/// Higher values indicate higher priority (should get better LOD).
///
/// The formula combines:
/// - Distance factor: closer objects get higher priority (1 / (1 + distance))
/// - Screen-space size: larger objects get higher priority (normalized to [0,1])
/// - Importance weight: user-defined importance multiplier
/// - Visibility: non-visible meshes get zero priority
pub fn compute_priority(params: &PriorityParams) -> f32 {
    // Non-visible meshes get zero priority
    if !params.visibility.is_visible() {
        return 0.0;
    }

    // Distance factor: closer = higher priority
    // Using 1/(1+d) to avoid division by zero and create smooth falloff
    let distance_factor = 1.0 / (1.0 + params.distance.max(0.0));

    // Screen-space size factor: larger = higher priority
    // Normalize assuming max screen size of ~2000 pixels
    let size_factor = (params.screen_size / 2000.0).clamp(0.0, 1.0);

    // Combine factors with importance
    // Weight: 40% distance, 40% size, 20% base importance
    let base_priority = 0.4 * distance_factor + 0.4 * size_factor + 0.2;

    // Apply importance multiplier
    base_priority * params.importance.max(0.0)
}

// ---------------------------------------------------------------------------
// LOD Assignment
// ---------------------------------------------------------------------------

/// Result of LOD assignment for a single mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodAssignment {
    /// Mesh identifier.
    pub mesh_id: u32,
    /// Assigned LOD level (0 = highest detail).
    pub assigned_lod: u32,
}

impl LodAssignment {
    /// Create a new LOD assignment.
    pub fn new(mesh_id: u32, assigned_lod: u32) -> Self {
        Self { mesh_id, assigned_lod }
    }
}

// ---------------------------------------------------------------------------
// Budget LOD Selector
// ---------------------------------------------------------------------------

/// Budget-aware LOD selector that assigns LOD levels based on priority and budget.
#[derive(Debug, Clone)]
pub struct BudgetLodSelector {
    /// Budget tracker for memory accounting.
    tracker: BudgetTracker,
}

impl BudgetLodSelector {
    /// Create a new budget LOD selector with the given tracker.
    pub fn new(tracker: BudgetTracker) -> Self {
        Self { tracker }
    }

    /// Create a new budget LOD selector with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BudgetTracker::with_defaults())
    }

    /// Get the budget tracker.
    pub fn tracker(&self) -> &BudgetTracker {
        &self.tracker
    }

    /// Get mutable access to the budget tracker.
    pub fn tracker_mut(&mut self) -> &mut BudgetTracker {
        &mut self.tracker
    }

    /// Reset the budget tracker usage.
    pub fn reset(&mut self) {
        self.tracker.reset();
    }

    /// Select LOD levels for all mesh instances based on priority and budget.
    ///
    /// Algorithm:
    /// 1. Sort instances by priority (descending)
    /// 2. For each instance, assign the highest affordable LOD
    /// 3. When budget is exhausted, use lowest LOD for remaining meshes
    pub fn select_lods(&self, instances: &[MeshInstance]) -> Vec<LodAssignment> {
        if instances.is_empty() {
            return Vec::new();
        }

        // Create indexed instances for sorting
        let mut indexed: Vec<(usize, f32)> = instances
            .iter()
            .enumerate()
            .map(|(i, inst)| (i, inst.priority))
            .collect();

        // Sort by priority descending (highest first)
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Track remaining mesh budget
        let mut remaining_budget = self.tracker.remaining_mesh_budget();
        let mut assignments = Vec::with_capacity(instances.len());

        for (idx, _priority) in indexed {
            let instance = &instances[idx];

            // Skip meshes with no LOD levels
            if instance.lod_sizes.is_empty() {
                continue;
            }

            // Find the highest affordable LOD
            let mut selected_lod = (instance.lod_count() - 1) as u32; // Default to lowest detail
            let mut found_affordable = false;

            for (lod_level, &size) in instance.lod_sizes.iter().enumerate() {
                if size <= remaining_budget {
                    selected_lod = lod_level as u32;
                    remaining_budget = remaining_budget.saturating_sub(size);
                    found_affordable = true;
                    break;
                }
            }

            // If no LOD fits in budget, still assign lowest LOD but don't subtract
            // (the mesh will be rendered at lowest detail without consuming more budget)
            if !found_affordable {
                // Check if lowest LOD fits - if so, subtract it
                let lowest_size = instance.lowest_detail_size();
                if lowest_size <= remaining_budget {
                    remaining_budget = remaining_budget.saturating_sub(lowest_size);
                }
                // Otherwise, we're over budget - don't subtract, just use lowest LOD
            }

            assignments.push(LodAssignment::new(instance.mesh_id, selected_lod));
        }

        // Sort assignments back to mesh_id order for consistent output
        assignments.sort_by_key(|a| a.mesh_id);

        assignments
    }

    /// Select LOD levels with a custom mesh budget (ignores tracker's current usage).
    pub fn select_lods_with_budget(
        &self,
        instances: &[MeshInstance],
        mesh_budget: usize,
    ) -> Vec<LodAssignment> {
        if instances.is_empty() {
            return Vec::new();
        }

        // Create indexed instances for sorting
        let mut indexed: Vec<(usize, f32)> = instances
            .iter()
            .enumerate()
            .map(|(i, inst)| (i, inst.priority))
            .collect();

        // Sort by priority descending
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut remaining_budget = mesh_budget;
        let mut assignments = Vec::with_capacity(instances.len());

        for (idx, _priority) in indexed {
            let instance = &instances[idx];

            if instance.lod_sizes.is_empty() {
                continue;
            }

            let mut selected_lod = (instance.lod_count() - 1) as u32;
            let mut found_affordable = false;

            for (lod_level, &size) in instance.lod_sizes.iter().enumerate() {
                if size <= remaining_budget {
                    selected_lod = lod_level as u32;
                    remaining_budget = remaining_budget.saturating_sub(size);
                    found_affordable = true;
                    break;
                }
            }

            if !found_affordable {
                let lowest_size = instance.lowest_detail_size();
                if lowest_size <= remaining_budget {
                    remaining_budget = remaining_budget.saturating_sub(lowest_size);
                }
            }

            assignments.push(LodAssignment::new(instance.mesh_id, selected_lod));
        }

        assignments.sort_by_key(|a| a.mesh_id);
        assignments
    }

    /// Select LODs and return total memory used.
    pub fn select_lods_with_accounting(
        &self,
        instances: &[MeshInstance],
    ) -> (Vec<LodAssignment>, usize) {
        let assignments = self.select_lods(instances);
        let total_used = self.compute_total_memory(&assignments, instances);
        (assignments, total_used)
    }

    /// Compute total memory for a set of assignments.
    pub fn compute_total_memory(
        &self,
        assignments: &[LodAssignment],
        instances: &[MeshInstance],
    ) -> usize {
        let instance_map: std::collections::HashMap<u32, &MeshInstance> = instances
            .iter()
            .map(|i| (i.mesh_id, i))
            .collect();

        assignments
            .iter()
            .filter_map(|a| {
                instance_map
                    .get(&a.mesh_id)
                    .and_then(|inst| inst.lod_size(a.assigned_lod))
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BudgetConfig Tests
    // =========================================================================

    #[test]
    fn test_budget_config_default() {
        let config = BudgetConfig::default();
        assert_eq!(config.mesh_budget, DEFAULT_MESH_BUDGET);
        assert_eq!(config.texture_budget, DEFAULT_TEXTURE_BUDGET);
        assert_eq!(config.shader_budget, DEFAULT_SHADER_BUDGET);
        assert_eq!(config.global_budget, DEFAULT_GLOBAL_BUDGET);
    }

    #[test]
    fn test_budget_config_custom() {
        let config = BudgetConfig::new(100, 200, 300, 600);
        assert_eq!(config.mesh_budget, 100);
        assert_eq!(config.texture_budget, 200);
        assert_eq!(config.shader_budget, 300);
        assert_eq!(config.global_budget, 600);
    }

    #[test]
    fn test_budget_config_scaled() {
        let config = BudgetConfig::scaled(0.5);
        assert_eq!(config.mesh_budget, DEFAULT_MESH_BUDGET / 2);
        assert_eq!(config.texture_budget, DEFAULT_TEXTURE_BUDGET / 2);
        assert_eq!(config.shader_budget, DEFAULT_SHADER_BUDGET / 2);
        assert_eq!(config.global_budget, DEFAULT_GLOBAL_BUDGET / 2);
    }

    #[test]
    fn test_budget_config_total_category_budget() {
        let config = BudgetConfig::new(100, 200, 300, 1000);
        assert_eq!(config.total_category_budget(), 600);
    }

    // =========================================================================
    // UsageSnapshot Tests
    // =========================================================================

    #[test]
    fn test_usage_snapshot_default() {
        let snapshot = UsageSnapshot::default();
        assert_eq!(snapshot.mesh_usage, 0);
        assert_eq!(snapshot.texture_usage, 0);
        assert_eq!(snapshot.shader_usage, 0);
        assert_eq!(snapshot.total(), 0);
    }

    #[test]
    fn test_usage_snapshot_total() {
        let snapshot = UsageSnapshot::new(100, 200, 300);
        assert_eq!(snapshot.total(), 600);
    }

    #[test]
    fn test_usage_snapshot_exceeds_any() {
        let config = BudgetConfig::new(100, 200, 300, 1000);

        let within = UsageSnapshot::new(50, 100, 150);
        assert!(!within.exceeds_any(&config));

        let over_mesh = UsageSnapshot::new(150, 100, 150);
        assert!(over_mesh.exceeds_any(&config));

        let over_texture = UsageSnapshot::new(50, 250, 150);
        assert!(over_texture.exceeds_any(&config));
    }

    #[test]
    fn test_usage_snapshot_exceeds_global() {
        let config = BudgetConfig::new(500, 500, 500, 1000);

        let within = UsageSnapshot::new(300, 300, 300);
        assert!(!within.exceeds_global(&config));

        let over = UsageSnapshot::new(400, 400, 400);
        assert!(over.exceeds_global(&config));
    }

    #[test]
    fn test_usage_snapshot_remaining() {
        let config = BudgetConfig::new(100, 200, 300, 1000);
        let snapshot = UsageSnapshot::new(30, 50, 100);

        assert_eq!(snapshot.remaining_mesh(&config), 70);
        assert_eq!(snapshot.remaining_texture(&config), 150);
        assert_eq!(snapshot.remaining_shader(&config), 200);
        assert_eq!(snapshot.remaining_global(&config), 820);
    }

    // =========================================================================
    // BudgetTracker Tests
    // =========================================================================

    #[test]
    fn test_budget_tracker_new() {
        let config = BudgetConfig::new(100, 200, 300, 600);
        let tracker = BudgetTracker::new(config);

        assert_eq!(tracker.config().mesh_budget, 100);
        assert_eq!(tracker.usage().total(), 0);
    }

    #[test]
    fn test_budget_tracker_with_defaults() {
        let tracker = BudgetTracker::with_defaults();
        assert_eq!(tracker.config().mesh_budget, DEFAULT_MESH_BUDGET);
    }

    #[test]
    fn test_budget_tracker_allocation() {
        let config = BudgetConfig::new(100, 200, 300, 600);
        let mut tracker = BudgetTracker::new(config);

        assert!(tracker.try_allocate_mesh(50));
        assert_eq!(tracker.usage().mesh_usage, 50);

        assert!(tracker.try_allocate_texture(100));
        assert_eq!(tracker.usage().texture_usage, 100);

        assert!(tracker.try_allocate_shader(150));
        assert_eq!(tracker.usage().shader_usage, 150);

        assert_eq!(tracker.usage().total(), 300);
    }

    #[test]
    fn test_budget_tracker_allocation_fails_over_budget() {
        let config = BudgetConfig::new(100, 200, 300, 600);
        let mut tracker = BudgetTracker::new(config);

        // Try to allocate more than mesh budget
        assert!(!tracker.try_allocate_mesh(150));
        assert_eq!(tracker.usage().mesh_usage, 0);
    }

    #[test]
    fn test_budget_tracker_global_budget_constraint() {
        let config = BudgetConfig::new(500, 500, 500, 400);
        let mut tracker = BudgetTracker::new(config);

        // Category budgets allow it, but global does not
        assert!(tracker.try_allocate_mesh(200));
        assert!(tracker.try_allocate_texture(100));
        assert!(!tracker.try_allocate_shader(150)); // Would exceed 400 global
    }

    #[test]
    fn test_budget_tracker_free() {
        let config = BudgetConfig::new(100, 200, 300, 600);
        let mut tracker = BudgetTracker::new(config);

        tracker.try_allocate_mesh(50);
        tracker.free_mesh(30);
        assert_eq!(tracker.usage().mesh_usage, 20);

        tracker.free_mesh(100); // Free more than allocated
        assert_eq!(tracker.usage().mesh_usage, 0); // Saturates to 0
    }

    #[test]
    fn test_budget_tracker_reset() {
        let config = BudgetConfig::new(100, 200, 300, 600);
        let mut tracker = BudgetTracker::new(config);

        tracker.try_allocate_mesh(50);
        tracker.try_allocate_texture(100);
        tracker.reset();

        assert_eq!(tracker.usage().total(), 0);
    }

    #[test]
    fn test_budget_tracker_utilization() {
        let config = BudgetConfig::new(100, 200, 300, 1000);
        let mut tracker = BudgetTracker::new(config);

        tracker.try_allocate_mesh(50);
        tracker.try_allocate_texture(100);
        tracker.try_allocate_shader(150);

        assert_eq!(tracker.mesh_utilization(), 0.5);
        assert_eq!(tracker.texture_utilization(), 0.5);
        assert_eq!(tracker.shader_utilization(), 0.5);
        assert_eq!(tracker.utilization(), 0.3); // 300/1000
    }

    #[test]
    fn test_budget_tracker_is_over_budget() {
        let config = BudgetConfig::new(100, 200, 300, 600);
        let mut tracker = BudgetTracker::new(config);

        assert!(!tracker.is_over_budget());

        tracker.try_allocate_mesh(100);
        assert!(!tracker.is_over_budget());

        // Force over-budget (direct manipulation for test)
        tracker.try_allocate_mesh(100); // This should fail, so manually set
    }

    // =========================================================================
    // MeshInstance Tests
    // =========================================================================

    #[test]
    fn test_mesh_instance_new() {
        let instance = MeshInstance::new(42, vec![1000, 500, 100], 0.8);
        assert_eq!(instance.mesh_id, 42);
        assert_eq!(instance.lod_count(), 3);
        assert_eq!(instance.priority, 0.8);
    }

    #[test]
    fn test_mesh_instance_lod_size() {
        let instance = MeshInstance::new(0, vec![1000, 500, 100], 1.0);
        assert_eq!(instance.lod_size(0), Some(1000));
        assert_eq!(instance.lod_size(1), Some(500));
        assert_eq!(instance.lod_size(2), Some(100));
        assert_eq!(instance.lod_size(3), None);
    }

    #[test]
    fn test_mesh_instance_detail_sizes() {
        let instance = MeshInstance::new(0, vec![1000, 500, 100], 1.0);
        assert_eq!(instance.highest_detail_size(), 1000);
        assert_eq!(instance.lowest_detail_size(), 100);
    }

    #[test]
    fn test_mesh_instance_empty_lods() {
        let instance = MeshInstance::new(0, vec![], 1.0);
        assert_eq!(instance.lod_count(), 0);
        assert_eq!(instance.highest_detail_size(), 0);
        assert_eq!(instance.lowest_detail_size(), 0);
    }

    // =========================================================================
    // Priority Tests
    // =========================================================================

    #[test]
    fn test_visibility_state() {
        assert!(VisibilityState::Visible.is_visible());
        assert!(!VisibilityState::Occluded.is_visible());
        assert!(!VisibilityState::OutOfFrustum.is_visible());
        assert!(!VisibilityState::DistanceCulled.is_visible());
    }

    #[test]
    fn test_compute_priority_visible() {
        let params = PriorityParams::visible(0.0, 1000.0, 1.0);
        let priority = compute_priority(&params);
        assert!(priority > 0.0, "visible mesh should have positive priority");
    }

    #[test]
    fn test_compute_priority_not_visible() {
        let params = PriorityParams::new(10.0, 500.0, 1.0, VisibilityState::Occluded);
        let priority = compute_priority(&params);
        assert_eq!(priority, 0.0, "non-visible mesh should have zero priority");
    }

    #[test]
    fn test_compute_priority_distance_factor() {
        let close = PriorityParams::visible(1.0, 500.0, 1.0);
        let far = PriorityParams::visible(100.0, 500.0, 1.0);

        let close_priority = compute_priority(&close);
        let far_priority = compute_priority(&far);

        assert!(
            close_priority > far_priority,
            "closer object should have higher priority"
        );
    }

    #[test]
    fn test_compute_priority_size_factor() {
        let large = PriorityParams::visible(10.0, 1000.0, 1.0);
        let small = PriorityParams::visible(10.0, 100.0, 1.0);

        let large_priority = compute_priority(&large);
        let small_priority = compute_priority(&small);

        assert!(
            large_priority > small_priority,
            "larger object should have higher priority"
        );
    }

    #[test]
    fn test_compute_priority_importance_factor() {
        let important = PriorityParams::visible(10.0, 500.0, 2.0);
        let normal = PriorityParams::visible(10.0, 500.0, 1.0);

        let important_priority = compute_priority(&important);
        let normal_priority = compute_priority(&normal);

        assert!(
            important_priority > normal_priority,
            "more important object should have higher priority"
        );
    }

    // =========================================================================
    // BudgetLodSelector Tests
    // =========================================================================

    #[test]
    fn test_selector_single_mesh_fits_budget_highest_lod() {
        let config = BudgetConfig::new(1_000_000, 0, 0, 1_000_000);
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        let instances = vec![MeshInstance::new(0, vec![100_000, 50_000, 10_000], 1.0)];
        let assignments = selector.select_lods(&instances);

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].mesh_id, 0);
        assert_eq!(assignments[0].assigned_lod, 0); // Highest detail fits
    }

    #[test]
    fn test_selector_over_budget_reduces_lod() {
        let config = BudgetConfig::new(60_000, 0, 0, 60_000); // Only fits LOD 1
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        let instances = vec![MeshInstance::new(0, vec![100_000, 50_000, 10_000], 1.0)];
        let assignments = selector.select_lods(&instances);

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].assigned_lod, 1); // LOD 1 (50k) fits
    }

    #[test]
    fn test_selector_very_small_budget_uses_lowest_lod() {
        let config = BudgetConfig::new(5_000, 0, 0, 5_000); // Only lowest fits
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        let instances = vec![MeshInstance::new(0, vec![100_000, 50_000, 10_000], 1.0)];
        let assignments = selector.select_lods(&instances);

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].assigned_lod, 2); // Lowest detail
    }

    #[test]
    fn test_selector_priority_sorting() {
        let config = BudgetConfig::new(150_000, 0, 0, 150_000);
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        // Low priority mesh first, high priority mesh second
        let instances = vec![
            MeshInstance::new(0, vec![100_000, 50_000, 10_000], 0.3), // Low priority
            MeshInstance::new(1, vec![100_000, 50_000, 10_000], 0.9), // High priority
        ];

        let assignments = selector.select_lods(&instances);

        // High priority mesh (id=1) should get better LOD
        let mesh_0 = assignments.iter().find(|a| a.mesh_id == 0).unwrap();
        let mesh_1 = assignments.iter().find(|a| a.mesh_id == 1).unwrap();

        assert!(
            mesh_1.assigned_lod <= mesh_0.assigned_lod,
            "high priority mesh should get equal or better LOD"
        );
    }

    #[test]
    fn test_selector_budget_categories_independent() {
        let config = BudgetConfig::new(100_000, 200_000, 50_000, 500_000);
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        // Mesh budget is 100k
        let instances = vec![MeshInstance::new(0, vec![80_000, 40_000, 10_000], 1.0)];
        let assignments = selector.select_lods(&instances);

        assert_eq!(assignments[0].assigned_lod, 0); // Fits in mesh budget
    }

    #[test]
    fn test_selector_empty_scene_returns_empty() {
        let selector = BudgetLodSelector::with_defaults();
        let instances: Vec<MeshInstance> = vec![];
        let assignments = selector.select_lods(&instances);
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_selector_snapshot_tracks_usage() {
        let config = BudgetConfig::new(200_000, 0, 0, 200_000);
        let mut tracker = BudgetTracker::new(config);
        tracker.try_allocate_mesh(50_000);

        let selector = BudgetLodSelector::new(tracker);

        // Remaining budget is 150k
        let instances = vec![MeshInstance::new(0, vec![200_000, 100_000, 50_000], 1.0)];
        let assignments = selector.select_lods(&instances);

        assert_eq!(assignments[0].assigned_lod, 1); // 100k fits in remaining 150k
    }

    #[test]
    fn test_selector_budget_exceeded_triggers_degradation() {
        let config = BudgetConfig::new(100_000, 0, 0, 100_000);
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        // Two meshes, each needs 80k at highest LOD
        let instances = vec![
            MeshInstance::new(0, vec![80_000, 40_000, 10_000], 0.5),
            MeshInstance::new(1, vec![80_000, 40_000, 10_000], 0.5),
        ];

        let assignments = selector.select_lods(&instances);

        // Total highest would be 160k, but budget is 100k
        // First mesh gets 80k, second must degrade
        let total_if_all_highest = 160_000;
        assert!(total_if_all_highest > config.mesh_budget);

        // At least one mesh should be degraded
        let has_degraded = assignments.iter().any(|a| a.assigned_lod > 0);
        assert!(has_degraded, "some mesh should be degraded due to budget");
    }

    #[test]
    fn test_selector_high_priority_gets_best_affordable_lod() {
        let config = BudgetConfig::new(120_000, 0, 0, 120_000);
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        let instances = vec![
            MeshInstance::new(0, vec![100_000, 50_000, 10_000], 0.1), // Low priority
            MeshInstance::new(1, vec![100_000, 50_000, 10_000], 1.0), // High priority
        ];

        let assignments = selector.select_lods(&instances);

        let mesh_1 = assignments.iter().find(|a| a.mesh_id == 1).unwrap();
        assert_eq!(mesh_1.assigned_lod, 0, "high priority should get LOD 0");
    }

    #[test]
    fn test_selector_with_custom_budget() {
        let selector = BudgetLodSelector::with_defaults();

        let instances = vec![MeshInstance::new(0, vec![100_000, 50_000, 10_000], 1.0)];

        // Custom budget that only fits LOD 2
        let assignments = selector.select_lods_with_budget(&instances, 15_000);
        assert_eq!(assignments[0].assigned_lod, 2);

        // Custom budget that fits LOD 0
        let assignments = selector.select_lods_with_budget(&instances, 200_000);
        assert_eq!(assignments[0].assigned_lod, 0);
    }

    #[test]
    fn test_selector_accounting() {
        let config = BudgetConfig::new(200_000, 0, 0, 200_000);
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        let instances = vec![
            MeshInstance::new(0, vec![100_000, 50_000, 10_000], 0.9),
            MeshInstance::new(1, vec![80_000, 40_000, 8_000], 0.8),
        ];

        let (assignments, total) = selector.select_lods_with_accounting(&instances);

        assert_eq!(assignments.len(), 2);
        assert!(total <= 200_000, "total should not exceed budget");
    }

    #[test]
    fn test_selector_mesh_with_no_lods_skipped() {
        let selector = BudgetLodSelector::with_defaults();

        let instances = vec![
            MeshInstance::new(0, vec![], 1.0),                   // No LODs
            MeshInstance::new(1, vec![10_000, 5_000], 1.0),      // Has LODs
        ];

        let assignments = selector.select_lods(&instances);

        // Only mesh 1 should have an assignment
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].mesh_id, 1);
    }

    #[test]
    fn test_selector_equal_priority_deterministic() {
        let config = BudgetConfig::new(50_000, 0, 0, 50_000);
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        let instances = vec![
            MeshInstance::new(0, vec![30_000, 15_000, 5_000], 0.5),
            MeshInstance::new(1, vec![30_000, 15_000, 5_000], 0.5),
        ];

        // Run multiple times to check determinism
        let assignments1 = selector.select_lods(&instances);
        let assignments2 = selector.select_lods(&instances);

        assert_eq!(assignments1, assignments2, "should be deterministic");
    }

    #[test]
    fn test_lod_assignment_struct() {
        let assignment = LodAssignment::new(42, 1);
        assert_eq!(assignment.mesh_id, 42);
        assert_eq!(assignment.assigned_lod, 1);
    }

    #[test]
    fn test_priority_params_default() {
        let params = PriorityParams::default();
        assert_eq!(params.distance, 0.0);
        assert_eq!(params.screen_size, 0.0);
        assert_eq!(params.importance, 1.0);
        assert!(params.visibility.is_visible());
    }

    // =========================================================================
    // Integration / Edge Case Tests
    // =========================================================================

    #[test]
    fn test_many_meshes_budget_distribution() {
        // Budget of 350k can fit about 5 meshes at LOD 0 (100k each) = 500k
        // But we want to show that high priority gets better LODs
        let config = BudgetConfig::new(350_000, 0, 0, 350_000);
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        // 5 meshes with varying priorities
        let instances: Vec<MeshInstance> = (0..5)
            .map(|i| {
                MeshInstance::new(
                    i,
                    vec![100_000, 50_000, 10_000],
                    (5 - i) as f32 / 5.0, // Decreasing priority: 1.0, 0.8, 0.6, 0.4, 0.2
                )
            })
            .collect();

        let (assignments, _total) = selector.select_lods_with_accounting(&instances);

        assert_eq!(assignments.len(), 5);

        // Higher priority meshes should have lower or equal LOD indices
        // Mesh 0 has highest priority (1.0), mesh 4 has lowest (0.2)
        for i in 0..4 {
            let mesh_i = assignments.iter().find(|a| a.mesh_id == i).unwrap();
            let mesh_i_plus_1 = assignments.iter().find(|a| a.mesh_id == i + 1).unwrap();

            assert!(
                mesh_i.assigned_lod <= mesh_i_plus_1.assigned_lod,
                "mesh {} (priority {}) should have <= LOD than mesh {} (priority {})",
                i,
                (5 - i) as f32 / 5.0,
                i + 1,
                (5 - (i + 1)) as f32 / 5.0
            );
        }

        // The highest priority mesh should get the best LOD
        let mesh_0 = assignments.iter().find(|a| a.mesh_id == 0).unwrap();
        assert_eq!(mesh_0.assigned_lod, 0, "highest priority should get LOD 0");
    }

    #[test]
    fn test_zero_budget() {
        let config = BudgetConfig::new(0, 0, 0, 0);
        let tracker = BudgetTracker::new(config);
        let selector = BudgetLodSelector::new(tracker);

        let instances = vec![MeshInstance::new(0, vec![100, 50, 10], 1.0)];
        let assignments = selector.select_lods(&instances);

        // Should still produce assignment (lowest LOD)
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].assigned_lod, 2);
    }

    #[test]
    fn test_single_lod_mesh() {
        let selector = BudgetLodSelector::with_defaults();

        let instances = vec![MeshInstance::new(0, vec![50_000], 1.0)];
        let assignments = selector.select_lods(&instances);

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].assigned_lod, 0); // Only option
    }

    #[test]
    fn test_compute_total_memory() {
        let selector = BudgetLodSelector::with_defaults();

        let instances = vec![
            MeshInstance::new(0, vec![100, 50, 10], 1.0),
            MeshInstance::new(1, vec![200, 100, 20], 0.5),
        ];

        let assignments = vec![
            LodAssignment::new(0, 1), // 50 bytes
            LodAssignment::new(1, 0), // 200 bytes
        ];

        let total = selector.compute_total_memory(&assignments, &instances);
        assert_eq!(total, 250);
    }
}
