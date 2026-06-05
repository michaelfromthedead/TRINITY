// SPDX-License-Identifier: MIT
//
// streaming/budget.rs -- GPU Memory Budget System (T-AS-5.3)
//
// Provides budget management for streaming assets with:
// - Per-type budgets (mesh, texture, shader)
// - Global memory cap
// - I/O bandwidth budget with dynamic adjustment
// - Pre-load estimation
// - LRU eviction with cost-benefit scoring
// - @unloadable integration (min_age, save_state)
//
// # Architecture
//
// The budget manager tracks GPU memory usage across asset types and enforces
// configurable limits. When budget is exceeded, it selects eviction candidates
// using a cost-benefit formula that prioritizes:
// - Low-priority assets
// - Large footprint assets
// - Old assets (high age)
//
// # Eviction Formula
//
// ```text
// score = (1 / priority) * footprint * age_secs
// ```
//
// Higher scores are evicted first. Pinned assets and assets below min_age
// are never evicted.
//
// # Example
//
// ```ignore
// use renderer_backend::streaming::budget::{BudgetManager, BudgetConfig, AssetFootprint, AssetType};
//
// let config = BudgetConfig::default();
// let mut manager = BudgetManager::new(config);
//
// // Check before loading
// let footprint = AssetFootprint {
//     asset_type: AssetType::Texture,
//     gpu_bytes: 4 * 1024 * 1024, // 4MB
//     priority: 1.0,
//     min_age_secs: 5.0,
//     save_state: false,
// };
//
// if manager.can_load(&footprint) {
//     // Load asset...
//     manager.register_asset(asset_id, footprint);
// } else {
//     // Need to evict first
//     let candidates = manager.select_eviction_candidates(footprint.gpu_bytes);
//     for id in candidates {
//         // Evict asset...
//         manager.unregister_asset(id);
//     }
// }
// ```

use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Type Definitions
// ---------------------------------------------------------------------------

/// Unique identifier for assets in the budget system.
pub type AssetId = u64;

/// Asset type categories for per-type budget tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetType {
    /// Mesh/geometry data (vertex buffers, index buffers).
    Mesh,
    /// Texture data (2D, 3D, cubemaps, mipmaps).
    Texture,
    /// Shader programs (compiled pipelines, PSOs).
    Shader,
}

impl AssetType {
    /// Get all asset types for iteration.
    pub fn all() -> &'static [AssetType] {
        &[AssetType::Mesh, AssetType::Texture, AssetType::Shader]
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default mesh budget: 512 MB.
pub const DEFAULT_MESH_BUDGET: u64 = 512 * 1024 * 1024;

/// Default texture budget: 1 GB.
pub const DEFAULT_TEXTURE_BUDGET: u64 = 1024 * 1024 * 1024;

/// Default shader budget: 256 MB.
pub const DEFAULT_SHADER_BUDGET: u64 = 256 * 1024 * 1024;

/// Default global budget: 2 GB.
pub const DEFAULT_GLOBAL_BUDGET: u64 = 2 * 1024 * 1024 * 1024;

/// Default I/O budget per frame: 16 MB.
pub const DEFAULT_IO_BUDGET_PER_FRAME: u64 = 16 * 1024 * 1024;

/// Default target frame time: 16.67 ms (60 FPS).
pub const DEFAULT_TARGET_FRAME_TIME_MS: f32 = 16.67;

/// Minimum I/O budget multiplier when frame time is exceeded.
pub const MIN_IO_BUDGET_MULTIPLIER: f32 = 0.25;

/// Maximum I/O budget multiplier when frame time is under target.
pub const MAX_IO_BUDGET_MULTIPLIER: f32 = 2.0;

/// Budget configuration for the streaming system.
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    /// Maximum GPU memory for mesh assets (bytes).
    pub mesh_budget: u64,
    /// Maximum GPU memory for texture assets (bytes).
    pub texture_budget: u64,
    /// Maximum GPU memory for shader assets (bytes).
    pub shader_budget: u64,
    /// Maximum total GPU memory across all types (bytes).
    pub global_budget: u64,
    /// Maximum I/O bytes per frame (dynamically adjusted).
    pub io_budget_per_frame: u64,
    /// Target frame time in milliseconds for I/O budget adjustment.
    pub target_frame_time_ms: f32,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            mesh_budget: DEFAULT_MESH_BUDGET,
            texture_budget: DEFAULT_TEXTURE_BUDGET,
            shader_budget: DEFAULT_SHADER_BUDGET,
            global_budget: DEFAULT_GLOBAL_BUDGET,
            io_budget_per_frame: DEFAULT_IO_BUDGET_PER_FRAME,
            target_frame_time_ms: DEFAULT_TARGET_FRAME_TIME_MS,
        }
    }
}

impl BudgetConfig {
    /// Create a budget config with custom mesh budget.
    pub fn with_mesh_budget(mut self, bytes: u64) -> Self {
        self.mesh_budget = bytes;
        self
    }

    /// Create a budget config with custom texture budget.
    pub fn with_texture_budget(mut self, bytes: u64) -> Self {
        self.texture_budget = bytes;
        self
    }

    /// Create a budget config with custom shader budget.
    pub fn with_shader_budget(mut self, bytes: u64) -> Self {
        self.shader_budget = bytes;
        self
    }

    /// Create a budget config with custom global budget.
    pub fn with_global_budget(mut self, bytes: u64) -> Self {
        self.global_budget = bytes;
        self
    }

    /// Create a budget config with custom I/O budget per frame.
    pub fn with_io_budget_per_frame(mut self, bytes: u64) -> Self {
        self.io_budget_per_frame = bytes;
        self
    }

    /// Create a budget config with custom target frame time.
    pub fn with_target_frame_time(mut self, ms: f32) -> Self {
        self.target_frame_time_ms = ms;
        self
    }

    /// Get the budget limit for a specific asset type.
    pub fn budget_for_type(&self, asset_type: AssetType) -> u64 {
        match asset_type {
            AssetType::Mesh => self.mesh_budget,
            AssetType::Texture => self.texture_budget,
            AssetType::Shader => self.shader_budget,
        }
    }

    /// Validate the configuration.
    ///
    /// Returns an error message if invalid, None if valid.
    pub fn validate(&self) -> Option<&'static str> {
        // Per-type budgets should not exceed global budget
        let sum = self.mesh_budget + self.texture_budget + self.shader_budget;
        if sum > 0 && self.global_budget > 0 && sum > self.global_budget * 2 {
            // Allow some overlap since assets rarely max out all categories
            // but warn if sum is way over
            return Some("per-type budget sum greatly exceeds global budget");
        }

        if self.target_frame_time_ms <= 0.0 {
            return Some("target frame time must be positive");
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Asset Footprint
// ---------------------------------------------------------------------------

/// Describes the GPU memory footprint and eviction constraints of an asset.
#[derive(Debug, Clone)]
pub struct AssetFootprint {
    /// Type of asset for per-type budget tracking.
    pub asset_type: AssetType,
    /// GPU memory usage in bytes.
    pub gpu_bytes: u64,
    /// Priority level (higher = more important, less likely to evict).
    /// Must be > 0. Default is 1.0.
    pub priority: f32,
    /// Minimum age in seconds before asset can be evicted (@unloadable min_age).
    /// 0.0 means immediately evictable.
    pub min_age_secs: f32,
    /// Whether to serialize state before GPU memory is freed (@unloadable save_state).
    pub save_state: bool,
}

impl Default for AssetFootprint {
    fn default() -> Self {
        Self {
            asset_type: AssetType::Texture,
            gpu_bytes: 0,
            priority: 1.0,
            min_age_secs: 0.0,
            save_state: false,
        }
    }
}

impl AssetFootprint {
    /// Create a new footprint for a mesh asset.
    pub fn mesh(gpu_bytes: u64) -> Self {
        Self {
            asset_type: AssetType::Mesh,
            gpu_bytes,
            ..Default::default()
        }
    }

    /// Create a new footprint for a texture asset.
    pub fn texture(gpu_bytes: u64) -> Self {
        Self {
            asset_type: AssetType::Texture,
            gpu_bytes,
            ..Default::default()
        }
    }

    /// Create a new footprint for a shader asset.
    pub fn shader(gpu_bytes: u64) -> Self {
        Self {
            asset_type: AssetType::Shader,
            gpu_bytes,
            ..Default::default()
        }
    }

    /// Set the priority level.
    pub fn with_priority(mut self, priority: f32) -> Self {
        self.priority = priority.max(0.001); // Prevent division by zero
        self
    }

    /// Set the minimum age constraint.
    pub fn with_min_age(mut self, min_age_secs: f32) -> Self {
        self.min_age_secs = min_age_secs.max(0.0);
        self
    }

    /// Set the save_state flag.
    pub fn with_save_state(mut self, save: bool) -> Self {
        self.save_state = save;
        self
    }
}

// ---------------------------------------------------------------------------
// Budget Usage Tracking
// ---------------------------------------------------------------------------

/// Current budget usage statistics.
#[derive(Debug, Clone, Default)]
pub struct BudgetUsage {
    /// Bytes used by mesh assets.
    pub mesh_bytes: u64,
    /// Bytes used by texture assets.
    pub texture_bytes: u64,
    /// Bytes used by shader assets.
    pub shader_bytes: u64,
    /// Total bytes used across all types.
    pub total_bytes: u64,
    /// Current I/O budget (dynamically adjusted).
    pub current_io_budget: u64,
    /// Number of registered assets.
    pub asset_count: usize,
    /// Number of pinned assets.
    pub pinned_count: usize,
}

impl BudgetUsage {
    /// Get usage for a specific asset type.
    pub fn usage_for_type(&self, asset_type: AssetType) -> u64 {
        match asset_type {
            AssetType::Mesh => self.mesh_bytes,
            AssetType::Texture => self.texture_bytes,
            AssetType::Shader => self.shader_bytes,
        }
    }

    /// Get remaining budget for a specific type given config limits.
    pub fn remaining_for_type(&self, asset_type: AssetType, config: &BudgetConfig) -> u64 {
        let used = self.usage_for_type(asset_type);
        let limit = config.budget_for_type(asset_type);
        limit.saturating_sub(used)
    }

    /// Get remaining global budget.
    pub fn remaining_global(&self, config: &BudgetConfig) -> u64 {
        config.global_budget.saturating_sub(self.total_bytes)
    }
}

// ---------------------------------------------------------------------------
// Tracked Asset
// ---------------------------------------------------------------------------

/// Internal representation of a tracked asset.
#[derive(Debug, Clone)]
struct TrackedAsset {
    /// Asset footprint and constraints.
    footprint: AssetFootprint,
    /// When the asset was loaded (for age calculation).
    load_time: Instant,
    /// Whether the asset is pinned (cannot be evicted).
    pinned: bool,
}

impl TrackedAsset {
    /// Get the age of this asset in seconds.
    fn age_secs(&self) -> f32 {
        self.load_time.elapsed().as_secs_f32()
    }

    /// Check if this asset can be evicted (not pinned and above min age).
    fn can_evict(&self) -> bool {
        !self.pinned && self.age_secs() >= self.footprint.min_age_secs
    }

    /// Compute eviction score: (1/priority) * footprint * age
    ///
    /// Higher scores are evicted first.
    fn eviction_score(&self) -> f64 {
        let inv_priority = 1.0 / self.footprint.priority.max(0.001) as f64;
        let footprint = self.footprint.gpu_bytes as f64;
        let age = self.age_secs() as f64;

        inv_priority * footprint * age
    }
}

// ---------------------------------------------------------------------------
// Eviction Candidate
// ---------------------------------------------------------------------------

/// Information about an asset selected for eviction.
#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    /// Asset ID.
    pub id: AssetId,
    /// Eviction score (higher = evicted first).
    pub score: f64,
    /// GPU bytes that will be freed.
    pub gpu_bytes: u64,
    /// Whether state should be saved before eviction.
    pub save_state: bool,
    /// Asset type.
    pub asset_type: AssetType,
}

// ---------------------------------------------------------------------------
// Budget Manager
// ---------------------------------------------------------------------------

/// Manages GPU memory budgets for streaming assets.
///
/// Tracks per-type and global memory usage, provides pre-load estimation,
/// and selects eviction candidates using cost-benefit scoring.
pub struct BudgetManager {
    /// Budget configuration.
    config: BudgetConfig,
    /// Current usage statistics.
    usage: BudgetUsage,
    /// Tracked assets by ID.
    assets: HashMap<AssetId, TrackedAsset>,
    /// Base I/O budget (before dynamic adjustment).
    base_io_budget: u64,
}

impl BudgetManager {
    /// Create a new budget manager with the given configuration.
    pub fn new(config: BudgetConfig) -> Self {
        let base_io_budget = config.io_budget_per_frame;
        Self {
            config,
            usage: BudgetUsage {
                current_io_budget: base_io_budget,
                ..Default::default()
            },
            assets: HashMap::new(),
            base_io_budget,
        }
    }

    /// Create a budget manager with default configuration.
    pub fn default_config() -> Self {
        Self::new(BudgetConfig::default())
    }

    /// Get the current configuration.
    pub fn config(&self) -> &BudgetConfig {
        &self.config
    }

    /// Get current budget usage statistics.
    pub fn usage(&self) -> &BudgetUsage {
        &self.usage
    }

    /// Check if an asset with the given footprint can be loaded without exceeding budgets.
    ///
    /// Returns true if:
    /// - Per-type budget has space for the asset
    /// - Global budget has space for the asset
    pub fn can_load(&self, footprint: &AssetFootprint) -> bool {
        // Check per-type budget
        let type_used = self.usage.usage_for_type(footprint.asset_type);
        let type_limit = self.config.budget_for_type(footprint.asset_type);
        if type_used + footprint.gpu_bytes > type_limit {
            return false;
        }

        // Check global budget
        if self.usage.total_bytes + footprint.gpu_bytes > self.config.global_budget {
            return false;
        }

        true
    }

    /// Check how much budget would remain after loading an asset.
    ///
    /// Returns (per_type_remaining, global_remaining) in bytes.
    /// Negative values indicate over-budget.
    pub fn remaining_after_load(&self, footprint: &AssetFootprint) -> (i64, i64) {
        let type_used = self.usage.usage_for_type(footprint.asset_type);
        let type_limit = self.config.budget_for_type(footprint.asset_type);
        let type_remaining =
            type_limit as i64 - type_used as i64 - footprint.gpu_bytes as i64;

        let global_remaining =
            self.config.global_budget as i64 - self.usage.total_bytes as i64 - footprint.gpu_bytes as i64;

        (type_remaining, global_remaining)
    }

    /// Register an asset in the budget system.
    ///
    /// Updates usage statistics and starts tracking the asset for eviction.
    /// If an asset with the same ID exists, it will be replaced.
    pub fn register_asset(&mut self, id: AssetId, footprint: AssetFootprint) {
        // Unregister existing asset if present
        if self.assets.contains_key(&id) {
            self.unregister_asset(id);
        }

        // Update usage
        match footprint.asset_type {
            AssetType::Mesh => self.usage.mesh_bytes += footprint.gpu_bytes,
            AssetType::Texture => self.usage.texture_bytes += footprint.gpu_bytes,
            AssetType::Shader => self.usage.shader_bytes += footprint.gpu_bytes,
        }
        self.usage.total_bytes += footprint.gpu_bytes;
        self.usage.asset_count += 1;

        // Track asset
        self.assets.insert(id, TrackedAsset {
            footprint,
            load_time: Instant::now(),
            pinned: false,
        });
    }

    /// Unregister an asset from the budget system.
    ///
    /// Updates usage statistics and stops tracking the asset.
    /// Returns the footprint if the asset existed.
    pub fn unregister_asset(&mut self, id: AssetId) -> Option<AssetFootprint> {
        let tracked = self.assets.remove(&id)?;

        // Update usage
        match tracked.footprint.asset_type {
            AssetType::Mesh => {
                self.usage.mesh_bytes = self.usage.mesh_bytes.saturating_sub(tracked.footprint.gpu_bytes);
            }
            AssetType::Texture => {
                self.usage.texture_bytes = self.usage.texture_bytes.saturating_sub(tracked.footprint.gpu_bytes);
            }
            AssetType::Shader => {
                self.usage.shader_bytes = self.usage.shader_bytes.saturating_sub(tracked.footprint.gpu_bytes);
            }
        }
        self.usage.total_bytes = self.usage.total_bytes.saturating_sub(tracked.footprint.gpu_bytes);
        self.usage.asset_count = self.usage.asset_count.saturating_sub(1);

        if tracked.pinned {
            self.usage.pinned_count = self.usage.pinned_count.saturating_sub(1);
        }

        Some(tracked.footprint)
    }

    /// Select eviction candidates to free at least `required_bytes`.
    ///
    /// Returns assets sorted by eviction score (highest first), excluding:
    /// - Pinned assets
    /// - Assets below min_age threshold
    ///
    /// The returned list contains enough assets to free at least `required_bytes`,
    /// or all evictable assets if total evictable < required.
    pub fn select_eviction_candidates(&self, required_bytes: u64) -> Vec<EvictionCandidate> {
        // Collect evictable assets with scores
        let mut candidates: Vec<_> = self.assets
            .iter()
            .filter(|(_, tracked)| tracked.can_evict())
            .map(|(&id, tracked)| EvictionCandidate {
                id,
                score: tracked.eviction_score(),
                gpu_bytes: tracked.footprint.gpu_bytes,
                save_state: tracked.footprint.save_state,
                asset_type: tracked.footprint.asset_type,
            })
            .collect();

        // Sort by score descending (highest score = evict first)
        candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select enough candidates to meet required_bytes
        let mut selected = Vec::new();
        let mut freed = 0u64;

        for candidate in candidates {
            if freed >= required_bytes {
                break;
            }
            freed += candidate.gpu_bytes;
            selected.push(candidate);
        }

        selected
    }

    /// Select eviction candidates for a specific asset type.
    ///
    /// Only considers assets of the given type.
    pub fn select_eviction_candidates_for_type(
        &self,
        asset_type: AssetType,
        required_bytes: u64,
    ) -> Vec<EvictionCandidate> {
        let mut candidates: Vec<_> = self.assets
            .iter()
            .filter(|(_, tracked)| {
                tracked.can_evict() && tracked.footprint.asset_type == asset_type
            })
            .map(|(&id, tracked)| EvictionCandidate {
                id,
                score: tracked.eviction_score(),
                gpu_bytes: tracked.footprint.gpu_bytes,
                save_state: tracked.footprint.save_state,
                asset_type: tracked.footprint.asset_type,
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected = Vec::new();
        let mut freed = 0u64;

        for candidate in candidates {
            if freed >= required_bytes {
                break;
            }
            freed += candidate.gpu_bytes;
            selected.push(candidate);
        }

        selected
    }

    /// Update the priority of an asset.
    ///
    /// Returns true if the asset exists and was updated.
    pub fn update_priority(&mut self, id: AssetId, priority: f32) -> bool {
        if let Some(tracked) = self.assets.get_mut(&id) {
            tracked.footprint.priority = priority.max(0.001);
            true
        } else {
            false
        }
    }

    /// Pin an asset to prevent eviction.
    ///
    /// Returns true if the asset exists and was pinned.
    pub fn pin_asset(&mut self, id: AssetId) -> bool {
        if let Some(tracked) = self.assets.get_mut(&id) {
            if !tracked.pinned {
                tracked.pinned = true;
                self.usage.pinned_count += 1;
            }
            true
        } else {
            false
        }
    }

    /// Unpin an asset to allow eviction.
    ///
    /// Returns true if the asset exists and was unpinned.
    pub fn unpin_asset(&mut self, id: AssetId) -> bool {
        if let Some(tracked) = self.assets.get_mut(&id) {
            if tracked.pinned {
                tracked.pinned = false;
                self.usage.pinned_count = self.usage.pinned_count.saturating_sub(1);
            }
            true
        } else {
            false
        }
    }

    /// Check if an asset is pinned.
    pub fn is_pinned(&self, id: AssetId) -> Option<bool> {
        self.assets.get(&id).map(|t| t.pinned)
    }

    /// Get the age of an asset in seconds.
    pub fn asset_age(&self, id: AssetId) -> Option<f32> {
        self.assets.get(&id).map(|t| t.age_secs())
    }

    /// Get the footprint of a registered asset.
    pub fn asset_footprint(&self, id: AssetId) -> Option<&AssetFootprint> {
        self.assets.get(&id).map(|t| &t.footprint)
    }

    /// Adjust I/O budget based on frame time.
    ///
    /// - If frame_time > target: reduce budget (down to MIN_IO_BUDGET_MULTIPLIER)
    /// - If frame_time < target: increase budget (up to MAX_IO_BUDGET_MULTIPLIER)
    ///
    /// Uses exponential smoothing to avoid oscillation.
    pub fn adjust_io_budget(&mut self, frame_time_ms: f32) {
        let target = self.config.target_frame_time_ms;

        // Calculate multiplier based on frame time ratio
        let ratio = target / frame_time_ms.max(0.1);
        let multiplier = ratio.clamp(MIN_IO_BUDGET_MULTIPLIER, MAX_IO_BUDGET_MULTIPLIER);

        // Apply exponential smoothing (blend current and new)
        let current_multiplier = self.usage.current_io_budget as f32 / self.base_io_budget as f32;
        let smoothed = current_multiplier * 0.7 + multiplier * 0.3;

        self.usage.current_io_budget = (self.base_io_budget as f32 * smoothed) as u64;
    }

    /// Reset I/O budget to base value.
    pub fn reset_io_budget(&mut self) {
        self.usage.current_io_budget = self.base_io_budget;
    }

    /// Get the current (dynamically adjusted) I/O budget.
    pub fn current_io_budget(&self) -> u64 {
        self.usage.current_io_budget
    }

    /// Get total bytes that can be evicted (excluding pinned and min_age assets).
    pub fn evictable_bytes(&self) -> u64 {
        self.assets
            .values()
            .filter(|t| t.can_evict())
            .map(|t| t.footprint.gpu_bytes)
            .sum()
    }

    /// Get the number of evictable assets.
    pub fn evictable_count(&self) -> usize {
        self.assets.values().filter(|t| t.can_evict()).count()
    }

    /// Clear all tracked assets and reset usage.
    pub fn clear(&mut self) {
        self.assets.clear();
        self.usage = BudgetUsage {
            current_io_budget: self.base_io_budget,
            ..Default::default()
        };
    }

    /// Get a snapshot of all tracked asset IDs.
    pub fn asset_ids(&self) -> Vec<AssetId> {
        self.assets.keys().copied().collect()
    }
}

impl Default for BudgetManager {
    fn default() -> Self {
        Self::default_config()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // ========================================================================
    // Configuration tests
    // ========================================================================

    #[test]
    fn test_default_config() {
        let config = BudgetConfig::default();
        assert_eq!(config.mesh_budget, DEFAULT_MESH_BUDGET);
        assert_eq!(config.texture_budget, DEFAULT_TEXTURE_BUDGET);
        assert_eq!(config.shader_budget, DEFAULT_SHADER_BUDGET);
        assert_eq!(config.global_budget, DEFAULT_GLOBAL_BUDGET);
        assert_eq!(config.io_budget_per_frame, DEFAULT_IO_BUDGET_PER_FRAME);
        assert!((config.target_frame_time_ms - DEFAULT_TARGET_FRAME_TIME_MS).abs() < 0.01);
    }

    #[test]
    fn test_config_builder() {
        let config = BudgetConfig::default()
            .with_mesh_budget(100)
            .with_texture_budget(200)
            .with_shader_budget(50)
            .with_global_budget(400)
            .with_io_budget_per_frame(1000)
            .with_target_frame_time(33.33);

        assert_eq!(config.mesh_budget, 100);
        assert_eq!(config.texture_budget, 200);
        assert_eq!(config.shader_budget, 50);
        assert_eq!(config.global_budget, 400);
        assert_eq!(config.io_budget_per_frame, 1000);
        assert!((config.target_frame_time_ms - 33.33).abs() < 0.01);
    }

    #[test]
    fn test_config_budget_for_type() {
        let config = BudgetConfig::default()
            .with_mesh_budget(100)
            .with_texture_budget(200)
            .with_shader_budget(50);

        assert_eq!(config.budget_for_type(AssetType::Mesh), 100);
        assert_eq!(config.budget_for_type(AssetType::Texture), 200);
        assert_eq!(config.budget_for_type(AssetType::Shader), 50);
    }

    #[test]
    fn test_config_validate() {
        let valid = BudgetConfig::default();
        assert!(valid.validate().is_none());

        let invalid = BudgetConfig::default().with_target_frame_time(0.0);
        assert!(invalid.validate().is_some());

        let invalid_neg = BudgetConfig::default().with_target_frame_time(-1.0);
        assert!(invalid_neg.validate().is_some());
    }

    // ========================================================================
    // Per-type budget enforcement tests (5 tests)
    // ========================================================================

    #[test]
    fn test_mesh_budget_enforcement() {
        let config = BudgetConfig::default()
            .with_mesh_budget(1000)
            .with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        // Can load mesh within budget
        let fp = AssetFootprint::mesh(500);
        assert!(manager.can_load(&fp));
        manager.register_asset(1, fp.clone());

        // Can load another small mesh
        let fp2 = AssetFootprint::mesh(400);
        assert!(manager.can_load(&fp2));

        // Cannot load mesh that exceeds remaining budget
        let fp3 = AssetFootprint::mesh(600);
        assert!(!manager.can_load(&fp3));
    }

    #[test]
    fn test_texture_budget_enforcement() {
        let config = BudgetConfig::default()
            .with_texture_budget(2000)
            .with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        let fp1 = AssetFootprint::texture(1500);
        assert!(manager.can_load(&fp1));
        manager.register_asset(1, fp1);

        assert_eq!(manager.usage().texture_bytes, 1500);

        let fp2 = AssetFootprint::texture(600);
        assert!(!manager.can_load(&fp2));
    }

    #[test]
    fn test_shader_budget_enforcement() {
        let config = BudgetConfig::default()
            .with_shader_budget(500)
            .with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        let fp1 = AssetFootprint::shader(250);
        assert!(manager.can_load(&fp1));
        manager.register_asset(1, fp1);

        let fp2 = AssetFootprint::shader(250);
        assert!(manager.can_load(&fp2));
        manager.register_asset(2, fp2);

        // Budget exactly full
        assert_eq!(manager.usage().shader_bytes, 500);

        // Cannot add more
        let fp3 = AssetFootprint::shader(1);
        assert!(!manager.can_load(&fp3));
    }

    #[test]
    fn test_mixed_type_budgets() {
        let config = BudgetConfig::default()
            .with_mesh_budget(100)
            .with_texture_budget(200)
            .with_shader_budget(50)
            .with_global_budget(1000);
        let mut manager = BudgetManager::new(config);

        // Load one of each type
        manager.register_asset(1, AssetFootprint::mesh(80));
        manager.register_asset(2, AssetFootprint::texture(150));
        manager.register_asset(3, AssetFootprint::shader(40));

        assert_eq!(manager.usage().mesh_bytes, 80);
        assert_eq!(manager.usage().texture_bytes, 150);
        assert_eq!(manager.usage().shader_bytes, 40);
        assert_eq!(manager.usage().total_bytes, 270);

        // Can still add more textures (within texture budget)
        assert!(manager.can_load(&AssetFootprint::texture(50)));

        // Cannot exceed mesh budget
        assert!(!manager.can_load(&AssetFootprint::mesh(30)));
    }

    #[test]
    fn test_type_budget_unregister_frees_space() {
        let config = BudgetConfig::default()
            .with_mesh_budget(100)
            .with_global_budget(1000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::mesh(100));
        assert!(!manager.can_load(&AssetFootprint::mesh(1)));

        manager.unregister_asset(1);
        assert_eq!(manager.usage().mesh_bytes, 0);
        assert!(manager.can_load(&AssetFootprint::mesh(100)));
    }

    // ========================================================================
    // Global budget enforcement tests (3 tests)
    // ========================================================================

    #[test]
    fn test_global_budget_enforcement() {
        let config = BudgetConfig::default()
            .with_mesh_budget(1000)
            .with_texture_budget(1000)
            .with_shader_budget(1000)
            .with_global_budget(500);
        let mut manager = BudgetManager::new(config);

        // Each type budget is 1000, but global is only 500
        let fp = AssetFootprint::mesh(400);
        assert!(manager.can_load(&fp));
        manager.register_asset(1, fp);

        // Per-type has room (600 remaining), but global only has 100
        let fp2 = AssetFootprint::texture(200);
        assert!(!manager.can_load(&fp2));

        let fp3 = AssetFootprint::texture(100);
        assert!(manager.can_load(&fp3));
    }

    #[test]
    fn test_global_budget_across_types() {
        let config = BudgetConfig::default()
            .with_mesh_budget(500)
            .with_texture_budget(500)
            .with_shader_budget(500)
            .with_global_budget(300);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::mesh(100));
        manager.register_asset(2, AssetFootprint::texture(100));
        manager.register_asset(3, AssetFootprint::shader(100));

        assert_eq!(manager.usage().total_bytes, 300);

        // All per-type budgets have room, but global is full
        assert!(!manager.can_load(&AssetFootprint::mesh(1)));
        assert!(!manager.can_load(&AssetFootprint::texture(1)));
        assert!(!manager.can_load(&AssetFootprint::shader(1)));
    }

    #[test]
    fn test_remaining_after_load() {
        let config = BudgetConfig::default()
            .with_mesh_budget(1000)
            .with_global_budget(800);
        let manager = BudgetManager::new(config);

        let (type_rem, global_rem) = manager.remaining_after_load(&AssetFootprint::mesh(500));
        assert_eq!(type_rem, 500);
        assert_eq!(global_rem, 300);

        // Over global budget
        let (type_rem2, global_rem2) = manager.remaining_after_load(&AssetFootprint::mesh(900));
        assert_eq!(type_rem2, 100);
        assert_eq!(global_rem2, -100);
    }

    // ========================================================================
    // I/O bandwidth adjustment tests (4 tests)
    // ========================================================================

    #[test]
    fn test_io_budget_default() {
        let config = BudgetConfig::default().with_io_budget_per_frame(1000);
        let manager = BudgetManager::new(config);
        assert_eq!(manager.current_io_budget(), 1000);
    }

    #[test]
    fn test_io_budget_reduces_on_slow_frame() {
        let config = BudgetConfig::default()
            .with_io_budget_per_frame(1000)
            .with_target_frame_time(16.67);
        let mut manager = BudgetManager::new(config);

        // Frame took 33ms (2x target) - should reduce budget
        manager.adjust_io_budget(33.34);

        // With smoothing, won't drop immediately to 50%, but should decrease
        assert!(manager.current_io_budget() < 1000);
    }

    #[test]
    fn test_io_budget_increases_on_fast_frame() {
        let config = BudgetConfig::default()
            .with_io_budget_per_frame(1000)
            .with_target_frame_time(16.67);
        let mut manager = BudgetManager::new(config);

        // Simulate a few slow frames to reduce budget first
        for _ in 0..10 {
            manager.adjust_io_budget(50.0);
        }
        let reduced = manager.current_io_budget();

        // Now fast frames should increase it
        for _ in 0..10 {
            manager.adjust_io_budget(8.0);
        }

        assert!(manager.current_io_budget() > reduced);
    }

    #[test]
    fn test_io_budget_reset() {
        let config = BudgetConfig::default().with_io_budget_per_frame(1000);
        let mut manager = BudgetManager::new(config);

        manager.adjust_io_budget(100.0); // Very slow frame
        assert!(manager.current_io_budget() < 1000);

        manager.reset_io_budget();
        assert_eq!(manager.current_io_budget(), 1000);
    }

    // ========================================================================
    // Eviction candidate selection tests (5 tests)
    // ========================================================================

    #[test]
    fn test_eviction_selects_lowest_priority_first() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100).with_priority(10.0));
        manager.register_asset(2, AssetFootprint::texture(100).with_priority(1.0)); // Lower priority
        manager.register_asset(3, AssetFootprint::texture(100).with_priority(5.0));

        // Wait a moment for age
        thread::sleep(Duration::from_millis(10));

        let candidates = manager.select_eviction_candidates(100);
        assert!(!candidates.is_empty());

        // Lower priority asset should be first
        assert_eq!(candidates[0].id, 2);
    }

    #[test]
    fn test_eviction_selects_larger_footprint_with_same_priority() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100).with_priority(1.0));
        manager.register_asset(2, AssetFootprint::texture(500).with_priority(1.0)); // Larger
        manager.register_asset(3, AssetFootprint::texture(200).with_priority(1.0));

        thread::sleep(Duration::from_millis(10));

        let candidates = manager.select_eviction_candidates(100);
        assert!(!candidates.is_empty());

        // Larger asset should be first (higher score)
        assert_eq!(candidates[0].id, 2);
    }

    #[test]
    fn test_eviction_accumulates_enough_bytes() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        for i in 0..10 {
            manager.register_asset(i, AssetFootprint::texture(100).with_priority(1.0));
        }

        thread::sleep(Duration::from_millis(10));

        // Need 350 bytes - should select 4 assets (400 bytes)
        let candidates = manager.select_eviction_candidates(350);
        let total: u64 = candidates.iter().map(|c| c.gpu_bytes).sum();
        assert!(total >= 350);
        assert_eq!(candidates.len(), 4);
    }

    #[test]
    fn test_eviction_excludes_pinned() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100).with_priority(0.1)); // Would be first
        manager.register_asset(2, AssetFootprint::texture(100).with_priority(1.0));

        manager.pin_asset(1);

        thread::sleep(Duration::from_millis(10));

        let candidates = manager.select_eviction_candidates(100);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, 2); // Pinned asset excluded
    }

    #[test]
    fn test_eviction_for_specific_type() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::mesh(100));
        manager.register_asset(2, AssetFootprint::texture(100));
        manager.register_asset(3, AssetFootprint::texture(100));

        thread::sleep(Duration::from_millis(10));

        // Only evict textures
        let candidates = manager.select_eviction_candidates_for_type(AssetType::Texture, 150);
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().all(|c| c.asset_type == AssetType::Texture));
    }

    // ========================================================================
    // min_age constraint tests (3 tests)
    // ========================================================================

    #[test]
    fn test_min_age_prevents_eviction() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        // Asset with 10 second min_age
        manager.register_asset(1, AssetFootprint::texture(100).with_min_age(10.0));

        // Immediately try to evict - should not be evictable
        let candidates = manager.select_eviction_candidates(100);
        assert!(candidates.is_empty());

        // Asset is tracked but not evictable
        assert_eq!(manager.evictable_count(), 0);
    }

    #[test]
    fn test_min_age_allows_eviction_after_threshold() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        // Very short min_age for testing
        manager.register_asset(1, AssetFootprint::texture(100).with_min_age(0.01));

        // Wait longer than min_age
        thread::sleep(Duration::from_millis(20));

        let candidates = manager.select_eviction_candidates(100);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, 1);
    }

    #[test]
    fn test_min_age_mixed_assets() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        // One immediately evictable, one with min_age
        manager.register_asset(1, AssetFootprint::texture(100).with_min_age(0.0));
        manager.register_asset(2, AssetFootprint::texture(100).with_min_age(100.0));

        thread::sleep(Duration::from_millis(10));

        let candidates = manager.select_eviction_candidates(200);
        // Only asset 1 should be evictable
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, 1);
        assert_eq!(manager.evictable_count(), 1);
    }

    // ========================================================================
    // Pin/unpin behavior tests (3 tests)
    // ========================================================================

    #[test]
    fn test_pin_prevents_eviction() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100));
        manager.pin_asset(1);

        thread::sleep(Duration::from_millis(10));

        assert_eq!(manager.evictable_count(), 0);
        assert_eq!(manager.usage().pinned_count, 1);

        let candidates = manager.select_eviction_candidates(100);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_unpin_allows_eviction() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100));
        manager.pin_asset(1);
        assert!(manager.is_pinned(1).unwrap());

        manager.unpin_asset(1);
        assert!(!manager.is_pinned(1).unwrap());
        assert_eq!(manager.usage().pinned_count, 0);

        thread::sleep(Duration::from_millis(10));

        let candidates = manager.select_eviction_candidates(100);
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_pin_unpin_idempotent() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100));

        // Multiple pins should only count once
        manager.pin_asset(1);
        manager.pin_asset(1);
        manager.pin_asset(1);
        assert_eq!(manager.usage().pinned_count, 1);

        // Multiple unpins should not underflow
        manager.unpin_asset(1);
        manager.unpin_asset(1);
        assert_eq!(manager.usage().pinned_count, 0);
    }

    // ========================================================================
    // Cost-benefit scoring tests (2 tests)
    // ========================================================================

    #[test]
    fn test_eviction_score_formula() {
        // score = (1/priority) * footprint * age
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        // Low priority, large footprint = high score = evict first
        manager.register_asset(1, AssetFootprint::texture(1000).with_priority(0.5));
        // High priority, small footprint = low score = evict last
        manager.register_asset(2, AssetFootprint::texture(100).with_priority(10.0));

        thread::sleep(Duration::from_millis(10));

        let candidates = manager.select_eviction_candidates(2000);
        assert_eq!(candidates.len(), 2);

        // Asset 1 should be first (higher score)
        assert_eq!(candidates[0].id, 1);
        assert_eq!(candidates[1].id, 2);

        // Verify relative scores
        assert!(candidates[0].score > candidates[1].score);
    }

    #[test]
    fn test_eviction_score_age_factor() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        // Same priority and footprint, but different ages
        manager.register_asset(1, AssetFootprint::texture(100).with_priority(1.0));
        thread::sleep(Duration::from_millis(50));
        manager.register_asset(2, AssetFootprint::texture(100).with_priority(1.0));

        thread::sleep(Duration::from_millis(10));

        let candidates = manager.select_eviction_candidates(200);
        assert_eq!(candidates.len(), 2);

        // Older asset (1) should have higher score and be first
        assert_eq!(candidates[0].id, 1);
    }

    // ========================================================================
    // save_state flag tests
    // ========================================================================

    #[test]
    fn test_save_state_flag_in_eviction_candidates() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100).with_save_state(true));
        manager.register_asset(2, AssetFootprint::texture(100).with_save_state(false));

        thread::sleep(Duration::from_millis(10));

        let candidates = manager.select_eviction_candidates(200);

        let c1 = candidates.iter().find(|c| c.id == 1).unwrap();
        let c2 = candidates.iter().find(|c| c.id == 2).unwrap();

        assert!(c1.save_state);
        assert!(!c2.save_state);
    }

    // ========================================================================
    // Asset lifecycle tests
    // ========================================================================

    #[test]
    fn test_register_updates_usage() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        assert_eq!(manager.usage().asset_count, 0);
        assert_eq!(manager.usage().total_bytes, 0);

        manager.register_asset(1, AssetFootprint::mesh(100));
        assert_eq!(manager.usage().asset_count, 1);
        assert_eq!(manager.usage().total_bytes, 100);
        assert_eq!(manager.usage().mesh_bytes, 100);

        manager.register_asset(2, AssetFootprint::texture(200));
        assert_eq!(manager.usage().asset_count, 2);
        assert_eq!(manager.usage().total_bytes, 300);
    }

    #[test]
    fn test_unregister_returns_footprint() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        let original = AssetFootprint::texture(500).with_priority(2.0);
        manager.register_asset(1, original.clone());

        let removed = manager.unregister_asset(1).unwrap();
        assert_eq!(removed.gpu_bytes, 500);
        assert!((removed.priority - 2.0).abs() < 0.01);

        // Non-existent returns None
        assert!(manager.unregister_asset(1).is_none());
    }

    #[test]
    fn test_register_replaces_existing() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100));
        assert_eq!(manager.usage().total_bytes, 100);

        // Re-register with different size
        manager.register_asset(1, AssetFootprint::texture(200));
        assert_eq!(manager.usage().total_bytes, 200);
        assert_eq!(manager.usage().asset_count, 1);
    }

    #[test]
    fn test_clear_resets_all() {
        let config = BudgetConfig::default()
            .with_global_budget(10000)
            .with_io_budget_per_frame(1000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100));
        manager.register_asset(2, AssetFootprint::mesh(200));
        manager.pin_asset(1);
        manager.adjust_io_budget(100.0);

        manager.clear();

        assert_eq!(manager.usage().asset_count, 0);
        assert_eq!(manager.usage().total_bytes, 0);
        assert_eq!(manager.usage().pinned_count, 0);
        assert_eq!(manager.current_io_budget(), 1000);
    }

    #[test]
    fn test_asset_ids() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(10, AssetFootprint::texture(100));
        manager.register_asset(20, AssetFootprint::mesh(100));
        manager.register_asset(30, AssetFootprint::shader(100));

        let mut ids = manager.asset_ids();
        ids.sort();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn test_update_priority() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100).with_priority(1.0));

        assert!(manager.update_priority(1, 5.0));
        assert!((manager.asset_footprint(1).unwrap().priority - 5.0).abs() < 0.01);

        // Non-existent asset
        assert!(!manager.update_priority(999, 1.0));
    }

    #[test]
    fn test_evictable_bytes() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100));
        manager.register_asset(2, AssetFootprint::texture(200));
        manager.register_asset(3, AssetFootprint::texture(300).with_min_age(1000.0));

        thread::sleep(Duration::from_millis(10));

        // Assets 1 and 2 are evictable, 3 is not (min_age)
        assert_eq!(manager.evictable_bytes(), 300);
        assert_eq!(manager.evictable_count(), 2);

        manager.pin_asset(1);
        assert_eq!(manager.evictable_bytes(), 200);
        assert_eq!(manager.evictable_count(), 1);
    }

    #[test]
    fn test_asset_age() {
        let config = BudgetConfig::default().with_global_budget(10000);
        let mut manager = BudgetManager::new(config);

        manager.register_asset(1, AssetFootprint::texture(100));

        let age1 = manager.asset_age(1).unwrap();
        thread::sleep(Duration::from_millis(50));
        let age2 = manager.asset_age(1).unwrap();

        assert!(age2 > age1);
        assert!(age2 >= 0.05);

        // Non-existent
        assert!(manager.asset_age(999).is_none());
    }
}
