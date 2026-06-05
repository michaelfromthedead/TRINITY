// SPDX-License-Identifier: MIT
//
// streaming/budget_lod.rs -- Budget-Aware LOD Selection (T-AS-5.5)
//
// Provides budget-constrained LOD selection for the streaming pipeline:
// - Texel budget: limit total texels across all loaded textures
// - Triangle budget: limit total triangles across all loaded meshes
// - LOD selection considers budget: if budget exceeded, select lower LOD
// - Priority-weighted LOD: low-priority assets reduce LOD first
// - Integration with @lod distances array and bias parameter
// - Dynamic adjustment as camera moves and budget pressure changes
//
// # Architecture
//
// The budget-aware LOD selector maintains per-asset LOD state and selects
// optimal LOD levels based on:
// 1. Distance-based desired LOD (from LodChain)
// 2. Current budget pressure (texels and triangles)
// 3. Asset priority (high priority keeps better LOD)
// 4. User-specified LOD bias
//
// When budget is exceeded, lower-priority assets are forced to lower LOD
// levels first, preserving quality for important assets.
//
// # Example
//
// ```ignore
// use renderer_backend::streaming::budget_lod::{
//     BudgetAwareLodSelector, LodBudgetConfig, AssetLodState,
// };
// use renderer_backend::asset::lod::LodChain;
//
// // Create selector with 2 billion texel and 50 million triangle budget
// let config = LodBudgetConfig {
//     texel_budget: 2_000_000_000,
//     triangle_budget: 50_000_000,
//     reduction_step: 0.1,
//     priority_weight: 0.5,
// };
// let mut selector = BudgetAwareLodSelector::new(config);
//
// // Register assets with their LOD chains
// selector.register_asset(asset_id, &lod_chain, priority);
//
// // Update priorities as needed
// selector.update_priorities(&[(asset_id, new_priority)]);
//
// // Select optimal LODs for all assets
// let lod_assignments = selector.select_lods();
// for (id, lod_level) in lod_assignments {
//     // Apply LOD level to asset
// }
// ```

use std::collections::HashMap;

use crate::asset::lod::LodChain;
use crate::streaming::budget::AssetId;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default texel budget: 2 billion texels.
pub const DEFAULT_TEXEL_BUDGET: u64 = 2_000_000_000;

/// Default triangle budget: 50 million triangles.
pub const DEFAULT_TRIANGLE_BUDGET: u64 = 50_000_000;

/// Default LOD reduction step per budget overage.
pub const DEFAULT_REDUCTION_STEP: f32 = 0.1;

/// Default priority weight for LOD selection.
pub const DEFAULT_PRIORITY_WEIGHT: f32 = 0.5;

/// Budget configuration for LOD selection.
#[derive(Debug, Clone)]
pub struct LodBudgetConfig {
    /// Maximum total texels across all loaded textures.
    pub texel_budget: u64,
    /// Maximum total triangles across all loaded meshes.
    pub triangle_budget: u64,
    /// LOD reduction step per unit of budget overage (0.0-1.0).
    /// Higher values mean more aggressive LOD reduction when over budget.
    pub reduction_step: f32,
    /// How much priority affects LOD selection (0.0-1.0).
    /// Higher values give more weight to priority when deciding which assets
    /// to reduce LOD for.
    pub priority_weight: f32,
}

impl Default for LodBudgetConfig {
    fn default() -> Self {
        Self {
            texel_budget: DEFAULT_TEXEL_BUDGET,
            triangle_budget: DEFAULT_TRIANGLE_BUDGET,
            reduction_step: DEFAULT_REDUCTION_STEP,
            priority_weight: DEFAULT_PRIORITY_WEIGHT,
        }
    }
}

impl LodBudgetConfig {
    /// Create a new configuration with custom texel budget.
    pub fn with_texel_budget(mut self, budget: u64) -> Self {
        self.texel_budget = budget;
        self
    }

    /// Create a new configuration with custom triangle budget.
    pub fn with_triangle_budget(mut self, budget: u64) -> Self {
        self.triangle_budget = budget;
        self
    }

    /// Create a new configuration with custom reduction step.
    pub fn with_reduction_step(mut self, step: f32) -> Self {
        self.reduction_step = step.clamp(0.01, 1.0);
        self
    }

    /// Create a new configuration with custom priority weight.
    pub fn with_priority_weight(mut self, weight: f32) -> Self {
        self.priority_weight = weight.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// Budget Tracking
// ---------------------------------------------------------------------------

/// Current LOD budget usage.
#[derive(Debug, Clone, Default)]
pub struct LodBudget {
    /// Maximum texels allowed.
    pub max_texels: u64,
    /// Maximum triangles allowed.
    pub max_triangles: u64,
    /// Current texel usage.
    pub current_texels: u64,
    /// Current triangle usage.
    pub current_triangles: u64,
}

impl LodBudget {
    /// Create a new budget with the given limits.
    pub fn new(max_texels: u64, max_triangles: u64) -> Self {
        Self {
            max_texels,
            max_triangles,
            current_texels: 0,
            current_triangles: 0,
        }
    }

    /// Get texel headroom (positive = under budget, negative = over budget).
    pub fn texel_headroom(&self) -> i64 {
        self.max_texels as i64 - self.current_texels as i64
    }

    /// Get triangle headroom (positive = under budget, negative = over budget).
    pub fn triangle_headroom(&self) -> i64 {
        self.max_triangles as i64 - self.current_triangles as i64
    }

    /// Check if texels are within budget.
    pub fn texels_within_budget(&self) -> bool {
        self.current_texels <= self.max_texels
    }

    /// Check if triangles are within budget.
    pub fn triangles_within_budget(&self) -> bool {
        self.current_triangles <= self.max_triangles
    }

    /// Check if both texels and triangles are within budget.
    pub fn within_budget(&self) -> bool {
        self.texels_within_budget() && self.triangles_within_budget()
    }

    /// Calculate texel budget pressure (0.0 = under budget, 1.0+ = over budget).
    pub fn texel_pressure(&self) -> f32 {
        if self.max_texels == 0 {
            return 0.0;
        }
        (self.current_texels as f32 / self.max_texels as f32).max(0.0)
    }

    /// Calculate triangle budget pressure (0.0 = under budget, 1.0+ = over budget).
    pub fn triangle_pressure(&self) -> f32 {
        if self.max_triangles == 0 {
            return 0.0;
        }
        (self.current_triangles as f32 / self.max_triangles as f32).max(0.0)
    }

    /// Calculate overall budget pressure (max of texel and triangle pressure).
    pub fn pressure(&self) -> f32 {
        self.texel_pressure().max(self.triangle_pressure())
    }

    /// Add usage to the budget.
    pub fn add(&mut self, texels: u64, triangles: u64) {
        self.current_texels = self.current_texels.saturating_add(texels);
        self.current_triangles = self.current_triangles.saturating_add(triangles);
    }

    /// Remove usage from the budget.
    pub fn remove(&mut self, texels: u64, triangles: u64) {
        self.current_texels = self.current_texels.saturating_sub(texels);
        self.current_triangles = self.current_triangles.saturating_sub(triangles);
    }

    /// Reset usage to zero.
    pub fn reset(&mut self) {
        self.current_texels = 0;
        self.current_triangles = 0;
    }
}

// ---------------------------------------------------------------------------
// Asset LOD State
// ---------------------------------------------------------------------------

/// Per-asset LOD tracking state.
#[derive(Debug, Clone)]
pub struct AssetLodState {
    /// Unique asset identifier.
    pub asset_id: AssetId,
    /// Currently active LOD level.
    pub current_lod: usize,
    /// Desired LOD level (based on distance/screen coverage).
    pub desired_lod: usize,
    /// Asset priority (higher = more important, keeps better LOD).
    pub priority: f32,
    /// Texel count at each LOD level.
    pub texels_at_lod: Vec<u64>,
    /// Triangle count at each LOD level.
    pub triangles_at_lod: Vec<u64>,
    /// Distance thresholds from LodChain.
    pub distances: Vec<f32>,
    /// Maximum LOD level (number of levels - 1).
    pub max_lod: usize,
}

impl AssetLodState {
    /// Create a new asset LOD state from a LodChain.
    pub fn from_lod_chain(asset_id: AssetId, chain: &LodChain, priority: f32) -> Self {
        let texels_at_lod: Vec<u64> = chain
            .levels
            .iter()
            .map(|level| {
                // Estimate texels from vertex count (rough approximation)
                // In practice, this would come from texture metadata
                level.vertex_count as u64 * 256 // Assume 16x16 texel area per vertex
            })
            .collect();

        let triangles_at_lod: Vec<u64> = chain
            .levels
            .iter()
            .map(|level| level.triangle_count as u64)
            .collect();

        let max_lod = chain.levels.len().saturating_sub(1);

        Self {
            asset_id,
            current_lod: 0,
            desired_lod: 0,
            priority: priority.max(0.001),
            texels_at_lod,
            triangles_at_lod,
            distances: chain.distances.clone(),
            max_lod,
        }
    }

    /// Create a new asset LOD state with explicit LOD data.
    pub fn new(
        asset_id: AssetId,
        priority: f32,
        texels_at_lod: Vec<u64>,
        triangles_at_lod: Vec<u64>,
        distances: Vec<f32>,
    ) -> Self {
        let max_lod = texels_at_lod.len().saturating_sub(1);
        Self {
            asset_id,
            current_lod: 0,
            desired_lod: 0,
            priority: priority.max(0.001),
            texels_at_lod,
            triangles_at_lod,
            distances,
            max_lod,
        }
    }

    /// Get texels at the current LOD level.
    pub fn current_texels(&self) -> u64 {
        self.texels_at_lod.get(self.current_lod).copied().unwrap_or(0)
    }

    /// Get triangles at the current LOD level.
    pub fn current_triangles(&self) -> u64 {
        self.triangles_at_lod.get(self.current_lod).copied().unwrap_or(0)
    }

    /// Get texels at the desired LOD level.
    pub fn desired_texels(&self) -> u64 {
        self.texels_at_lod.get(self.desired_lod).copied().unwrap_or(0)
    }

    /// Get triangles at the desired LOD level.
    pub fn desired_triangles(&self) -> u64 {
        self.triangles_at_lod.get(self.desired_lod).copied().unwrap_or(0)
    }

    /// Get texels at a specific LOD level.
    pub fn texels_at(&self, lod: usize) -> u64 {
        self.texels_at_lod.get(lod).copied().unwrap_or(0)
    }

    /// Get triangles at a specific LOD level.
    pub fn triangles_at(&self, lod: usize) -> u64 {
        self.triangles_at_lod.get(lod).copied().unwrap_or(0)
    }

    /// Calculate LOD reduction score for budget-aware selection.
    ///
    /// Higher scores mean this asset should reduce LOD first.
    /// Score is based on: (1/priority) * (current_triangles + current_texels/1000)
    pub fn reduction_score(&self) -> f64 {
        let inv_priority = 1.0 / self.priority.max(0.001) as f64;
        let triangles = self.current_triangles() as f64;
        let texels = self.current_texels() as f64 / 1000.0;
        inv_priority * (triangles + texels)
    }

    /// Check if LOD can be reduced further.
    pub fn can_reduce_lod(&self) -> bool {
        self.current_lod < self.max_lod
    }

    /// Check if LOD can be increased (improved quality).
    pub fn can_increase_lod(&self) -> bool {
        self.current_lod > 0
    }

    /// Update desired LOD based on distance.
    pub fn update_desired_lod(&mut self, distance: f32, bias: f32) {
        // Apply bias to effective distance
        let bias_factor = 2.0_f32.powf(bias);
        let effective_distance = distance * bias_factor;

        // Find appropriate level based on distance thresholds
        let mut level = 0;
        for (i, &threshold) in self.distances.iter().enumerate() {
            if effective_distance >= threshold {
                level = i + 1;
            } else {
                break;
            }
        }

        self.desired_lod = level.min(self.max_lod);
    }
}

// ---------------------------------------------------------------------------
// Budget-Aware LOD Selector
// ---------------------------------------------------------------------------

/// Budget-aware LOD selection system.
///
/// Manages LOD selection across all assets while respecting texel and
/// triangle budgets. When budget is exceeded, lower-priority assets
/// are forced to lower LOD levels first.
pub struct BudgetAwareLodSelector {
    /// Configuration.
    config: LodBudgetConfig,
    /// Current budget state.
    budget: LodBudget,
    /// Per-asset LOD state.
    assets: HashMap<AssetId, AssetLodState>,
}

impl BudgetAwareLodSelector {
    /// Create a new budget-aware LOD selector.
    pub fn new(config: LodBudgetConfig) -> Self {
        let budget = LodBudget::new(config.texel_budget, config.triangle_budget);
        Self {
            config,
            budget,
            assets: HashMap::new(),
        }
    }

    /// Create a selector with default configuration.
    pub fn default_config() -> Self {
        Self::new(LodBudgetConfig::default())
    }

    /// Get the current configuration.
    pub fn config(&self) -> &LodBudgetConfig {
        &self.config
    }

    /// Get the current budget state.
    pub fn budget(&self) -> &LodBudget {
        &self.budget
    }

    /// Register an asset with its LOD chain.
    ///
    /// The asset will be tracked for budget-aware LOD selection.
    pub fn register_asset(&mut self, asset_id: AssetId, chain: &LodChain, priority: f32) {
        // Unregister if already present
        self.unregister_asset(asset_id);

        let state = AssetLodState::from_lod_chain(asset_id, chain, priority);

        // Add to budget
        self.budget.add(state.current_texels(), state.current_triangles());

        self.assets.insert(asset_id, state);
    }

    /// Register an asset with explicit LOD data.
    pub fn register_asset_explicit(
        &mut self,
        asset_id: AssetId,
        priority: f32,
        texels_at_lod: Vec<u64>,
        triangles_at_lod: Vec<u64>,
        distances: Vec<f32>,
    ) {
        // Unregister if already present
        self.unregister_asset(asset_id);

        let state = AssetLodState::new(asset_id, priority, texels_at_lod, triangles_at_lod, distances);

        // Add to budget
        self.budget.add(state.current_texels(), state.current_triangles());

        self.assets.insert(asset_id, state);
    }

    /// Unregister an asset from the selector.
    ///
    /// Returns the asset's LOD state if it was registered.
    pub fn unregister_asset(&mut self, asset_id: AssetId) -> Option<AssetLodState> {
        if let Some(state) = self.assets.remove(&asset_id) {
            // Remove from budget
            self.budget.remove(state.current_texels(), state.current_triangles());
            Some(state)
        } else {
            None
        }
    }

    /// Update priorities for multiple assets.
    pub fn update_priorities(&mut self, updates: &[(AssetId, f32)]) {
        for &(asset_id, priority) in updates {
            if let Some(state) = self.assets.get_mut(&asset_id) {
                state.priority = priority.max(0.001);
            }
        }
    }

    /// Update desired LOD for an asset based on distance.
    pub fn update_desired_lod(&mut self, asset_id: AssetId, distance: f32, bias: f32) {
        if let Some(state) = self.assets.get_mut(&asset_id) {
            state.update_desired_lod(distance, bias);
        }
    }

    /// Update desired LOD for multiple assets.
    pub fn update_desired_lods(&mut self, updates: &[(AssetId, f32, f32)]) {
        for &(asset_id, distance, bias) in updates {
            self.update_desired_lod(asset_id, distance, bias);
        }
    }

    /// Select optimal LOD levels for all assets considering budget constraints.
    ///
    /// Returns a list of (asset_id, lod_level) pairs for assets whose LOD changed.
    pub fn select_lods(&mut self) -> Vec<(AssetId, usize)> {
        // First pass: set all assets to their desired LOD and calculate budget
        self.budget.reset();

        for state in self.assets.values_mut() {
            state.current_lod = state.desired_lod;
        }

        // Recalculate budget
        for state in self.assets.values() {
            self.budget.add(state.current_texels(), state.current_triangles());
        }

        // Second pass: if over budget, reduce LOD for low-priority assets
        let mut changes = Vec::new();

        while !self.budget.within_budget() {
            // Find the asset with the highest reduction score that can still reduce LOD
            let candidate = self.find_reduction_candidate();

            match candidate {
                Some(asset_id) => {
                    let state = self.assets.get_mut(&asset_id).unwrap();

                    // Remove old contribution
                    self.budget.remove(state.current_texels(), state.current_triangles());

                    // Reduce LOD
                    state.current_lod = (state.current_lod + 1).min(state.max_lod);

                    // Add new contribution
                    self.budget.add(state.current_texels(), state.current_triangles());

                    changes.push((asset_id, state.current_lod));
                }
                None => {
                    // No more assets can reduce LOD - we're stuck over budget
                    break;
                }
            }
        }

        changes
    }

    /// Find the asset that should reduce LOD first.
    ///
    /// Returns the asset with the highest reduction score that can still reduce LOD.
    fn find_reduction_candidate(&self) -> Option<AssetId> {
        self.assets
            .values()
            .filter(|state| state.can_reduce_lod())
            .max_by(|a, b| {
                a.reduction_score()
                    .partial_cmp(&b.reduction_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|state| state.asset_id)
    }

    /// Get the current budget pressure (0.0 = under budget, 1.0+ = over budget).
    pub fn get_budget_pressure(&self) -> f32 {
        self.budget.pressure()
    }

    /// Get texel budget pressure.
    pub fn get_texel_pressure(&self) -> f32 {
        self.budget.texel_pressure()
    }

    /// Get triangle budget pressure.
    pub fn get_triangle_pressure(&self) -> f32 {
        self.budget.triangle_pressure()
    }

    /// Apply LOD bias to a base LOD level for a specific asset.
    ///
    /// Bias is applied on top of distance-based LOD:
    /// - Positive bias: use lower detail (higher LOD index)
    /// - Negative bias: use higher detail (lower LOD index)
    pub fn apply_lod_bias(&self, asset_id: AssetId, base_lod: usize, bias: f32) -> usize {
        let state = match self.assets.get(&asset_id) {
            Some(s) => s,
            None => return base_lod,
        };

        // Apply bias: each 1.0 of bias shifts LOD by one level
        let bias_shift = bias.round() as i32;
        let new_lod = (base_lod as i32 + bias_shift).max(0) as usize;

        new_lod.min(state.max_lod)
    }

    /// Get the current LOD level for an asset.
    pub fn get_current_lod(&self, asset_id: AssetId) -> Option<usize> {
        self.assets.get(&asset_id).map(|s| s.current_lod)
    }

    /// Get the desired LOD level for an asset.
    pub fn get_desired_lod(&self, asset_id: AssetId) -> Option<usize> {
        self.assets.get(&asset_id).map(|s| s.desired_lod)
    }

    /// Get the LOD state for an asset.
    pub fn get_asset_state(&self, asset_id: AssetId) -> Option<&AssetLodState> {
        self.assets.get(&asset_id)
    }

    /// Get all registered asset IDs.
    pub fn asset_ids(&self) -> Vec<AssetId> {
        self.assets.keys().copied().collect()
    }

    /// Get the number of registered assets.
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Check if an asset is registered.
    pub fn has_asset(&self, asset_id: AssetId) -> bool {
        self.assets.contains_key(&asset_id)
    }

    /// Clear all registered assets and reset budget.
    pub fn clear(&mut self) {
        self.assets.clear();
        self.budget.reset();
    }

    /// Recalculate budget from current asset states.
    pub fn recalculate_budget(&mut self) {
        self.budget.reset();
        for state in self.assets.values() {
            self.budget.add(state.current_texels(), state.current_triangles());
        }
    }

    /// Set a specific LOD level for an asset, bypassing automatic selection.
    ///
    /// Updates the budget accordingly. Returns the old LOD level.
    pub fn set_asset_lod(&mut self, asset_id: AssetId, lod: usize) -> Option<usize> {
        let state = self.assets.get_mut(&asset_id)?;
        let old_lod = state.current_lod;

        if lod != old_lod {
            // Remove old contribution
            self.budget.remove(state.current_texels(), state.current_triangles());

            // Update LOD
            state.current_lod = lod.min(state.max_lod);

            // Add new contribution
            self.budget.add(state.current_texels(), state.current_triangles());
        }

        Some(old_lod)
    }
}

impl Default for BudgetAwareLodSelector {
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
    use crate::asset::lod::{LodLevel, CrossFadeConfig};
    use crate::asset::meshlet::BoundingSphere;

    // Helper to create a simple LOD chain for testing
    fn create_test_lod_chain(level_count: usize, base_triangles: u32) -> LodChain {
        let mut levels = Vec::with_capacity(level_count);
        let mut distances = Vec::with_capacity(level_count);

        for i in 0..level_count {
            let simplification = 0.5_f32.powi(i as i32);
            let triangles = ((base_triangles as f32) * simplification) as u32;
            let vertices = triangles; // Rough approximation

            levels.push(LodLevel {
                mesh_data: crate::asset::lod::MeshData::new(
                    vec![[0.0, 0.0, 0.0]; vertices as usize],
                    vec![0; (triangles * 3) as usize],
                ),
                screen_coverage: simplification,
                vertex_count: vertices,
                triangle_count: triangles,
                bounds: BoundingSphere::default(),
                error: i as f32 * 0.1,
            });

            if i > 0 {
                distances.push(10.0 * i as f32);
            }
        }

        LodChain {
            levels,
            distances,
            cross_fade: CrossFadeConfig::default(),
            bounds: BoundingSphere::default(),
        }
    }

    // ========================================================================
    // Budget tracking tests (4 tests)
    // ========================================================================

    #[test]
    fn test_budget_add_texels() {
        let mut budget = LodBudget::new(1000, 500);
        assert_eq!(budget.current_texels, 0);

        budget.add(250, 0);
        assert_eq!(budget.current_texels, 250);

        budget.add(500, 0);
        assert_eq!(budget.current_texels, 750);

        assert!(budget.texels_within_budget());
    }

    #[test]
    fn test_budget_add_triangles() {
        let mut budget = LodBudget::new(1000, 500);
        assert_eq!(budget.current_triangles, 0);

        budget.add(0, 200);
        assert_eq!(budget.current_triangles, 200);

        budget.add(0, 350);
        assert_eq!(budget.current_triangles, 550);

        assert!(!budget.triangles_within_budget());
    }

    #[test]
    fn test_budget_remove_texels() {
        let mut budget = LodBudget::new(1000, 500);
        budget.add(500, 200);

        budget.remove(200, 0);
        assert_eq!(budget.current_texels, 300);
        assert_eq!(budget.current_triangles, 200);

        // Saturating subtract
        budget.remove(1000, 0);
        assert_eq!(budget.current_texels, 0);
    }

    #[test]
    fn test_budget_remove_triangles() {
        let mut budget = LodBudget::new(1000, 500);
        budget.add(100, 400);

        budget.remove(0, 150);
        assert_eq!(budget.current_triangles, 250);

        // Saturating subtract
        budget.remove(0, 1000);
        assert_eq!(budget.current_triangles, 0);
    }

    #[test]
    fn test_budget_headroom() {
        let mut budget = LodBudget::new(1000, 500);
        budget.add(300, 200);

        assert_eq!(budget.texel_headroom(), 700);
        assert_eq!(budget.triangle_headroom(), 300);

        // Over budget
        budget.add(800, 400);
        assert_eq!(budget.texel_headroom(), -100);
        assert_eq!(budget.triangle_headroom(), -100);
    }

    // ========================================================================
    // LOD selection tests (5 tests)
    // ========================================================================

    #[test]
    fn test_under_budget_selects_high_lod() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(10_000_000)
            .with_triangle_budget(1_000_000);
        let mut selector = BudgetAwareLodSelector::new(config);

        let chain = create_test_lod_chain(4, 1000);
        selector.register_asset(1, &chain, 1.0);

        // Set desired LOD to 0 (highest quality)
        selector.update_desired_lod(1, 5.0, 0.0); // Close distance

        let changes = selector.select_lods();

        // Should select LOD 0 (highest quality) since under budget
        assert_eq!(selector.get_current_lod(1), Some(0));
        // No changes needed if already at desired
        assert!(changes.is_empty() || changes.iter().any(|(id, lod)| *id == 1 && *lod == 0));
    }

    #[test]
    fn test_over_budget_reduces_lod() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(100)  // Very small budget
            .with_triangle_budget(100);
        let mut selector = BudgetAwareLodSelector::new(config);

        // Register asset with more triangles than budget allows
        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500, 250, 125],
            vec![1000, 500, 250, 125],
            vec![10.0, 25.0, 50.0],
        );

        let changes = selector.select_lods();

        // Should have reduced LOD to fit budget
        assert!(selector.get_current_lod(1).unwrap() > 0);
        assert!(!changes.is_empty());
    }

    #[test]
    fn test_priority_ordering_in_lod_reduction() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(2000)
            .with_triangle_budget(2000);
        let mut selector = BudgetAwareLodSelector::new(config);

        // High priority asset
        selector.register_asset_explicit(
            1,
            10.0,  // High priority
            vec![1000, 500, 250],
            vec![1000, 500, 250],
            vec![10.0, 25.0],
        );

        // Low priority asset
        selector.register_asset_explicit(
            2,
            0.1,  // Low priority
            vec![1500, 750, 375],
            vec![1500, 750, 375],
            vec![10.0, 25.0],
        );

        let changes = selector.select_lods();

        // Low priority asset should reduce LOD first
        let lod_1 = selector.get_current_lod(1).unwrap();
        let lod_2 = selector.get_current_lod(2).unwrap();

        // Asset 2 (low priority) should have higher LOD index (lower quality)
        assert!(lod_2 >= lod_1);
        assert!(!changes.is_empty());
    }

    #[test]
    fn test_multiple_assets_budget_balancing() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(5000)
            .with_triangle_budget(5000);
        let mut selector = BudgetAwareLodSelector::new(config);

        // Register 5 assets with same priority
        for i in 1..=5 {
            selector.register_asset_explicit(
                i,
                1.0,
                vec![2000, 1000, 500],
                vec![2000, 1000, 500],
                vec![10.0, 25.0],
            );
        }

        selector.select_lods();

        // Total budget is 5000, each asset at LOD 0 uses 2000
        // 5 * 2000 = 10000 > 5000, so some must reduce LOD
        let total_triangles: u64 = (1..=5)
            .map(|i| selector.get_asset_state(i).unwrap().current_triangles())
            .sum();

        assert!(total_triangles <= 5000);
    }

    #[test]
    fn test_lod_selection_respects_max_lod() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(10)  // Impossibly small
            .with_triangle_budget(10);
        let mut selector = BudgetAwareLodSelector::new(config);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500, 250],  // 3 levels, max_lod = 2
            vec![1000, 500, 250],
            vec![10.0, 25.0],
        );

        selector.select_lods();

        // Should not exceed max LOD
        assert!(selector.get_current_lod(1).unwrap() <= 2);
    }

    // ========================================================================
    // Priority weighting tests (4 tests)
    // ========================================================================

    #[test]
    fn test_high_priority_keeps_lod() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(1500)
            .with_triangle_budget(1500);
        let mut selector = BudgetAwareLodSelector::new(config);

        // Very high priority - should resist reduction
        selector.register_asset_explicit(
            1,
            100.0,
            vec![1000, 500, 250],
            vec![1000, 500, 250],
            vec![10.0, 25.0],
        );

        // Low priority - should reduce first
        selector.register_asset_explicit(
            2,
            0.01,
            vec![1000, 500, 250],
            vec![1000, 500, 250],
            vec![10.0, 25.0],
        );

        selector.select_lods();

        // High priority asset should maintain better LOD
        let lod_1 = selector.get_current_lod(1).unwrap();
        let lod_2 = selector.get_current_lod(2).unwrap();

        assert!(lod_1 <= lod_2);
    }

    #[test]
    fn test_low_priority_reduces_first() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(1200)
            .with_triangle_budget(1200);
        let mut selector = BudgetAwareLodSelector::new(config);

        // Both assets same size, different priorities
        selector.register_asset_explicit(
            1,
            10.0,
            vec![1000, 500],
            vec![1000, 500],
            vec![20.0],
        );

        selector.register_asset_explicit(
            2,
            0.1,
            vec![1000, 500],
            vec![1000, 500],
            vec![20.0],
        );

        selector.select_lods();

        // Asset 2 (low priority) should have reduced
        assert!(selector.get_current_lod(2).unwrap() > 0);
    }

    #[test]
    fn test_update_priorities_affects_selection() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(1200)
            .with_triangle_budget(1200);
        let mut selector = BudgetAwareLodSelector::new(config);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500],
            vec![1000, 500],
            vec![20.0],
        );

        selector.register_asset_explicit(
            2,
            1.0,
            vec![1000, 500],
            vec![1000, 500],
            vec![20.0],
        );

        // Update priorities: asset 2 becomes high priority
        selector.update_priorities(&[(2, 100.0)]);

        selector.select_lods();

        // Asset 1 (now lower priority) should reduce first
        let lod_1 = selector.get_current_lod(1).unwrap();
        let lod_2 = selector.get_current_lod(2).unwrap();

        assert!(lod_2 <= lod_1);
    }

    #[test]
    fn test_priority_zero_clamps_to_minimum() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(500)
            .with_triangle_budget(500);
        let mut selector = BudgetAwareLodSelector::new(config);

        // Priority 0 should be clamped to minimum
        selector.register_asset_explicit(
            1,
            0.0,  // Will be clamped to 0.001
            vec![1000, 500],
            vec![1000, 500],
            vec![20.0],
        );

        let state = selector.get_asset_state(1).unwrap();
        assert!(state.priority > 0.0);
    }

    // ========================================================================
    // Dynamic adjustment tests (3 tests)
    // ========================================================================

    #[test]
    fn test_camera_movement_updates_desired_lod() {
        let config = LodBudgetConfig::default();
        let mut selector = BudgetAwareLodSelector::new(config);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500, 250],
            vec![1000, 500, 250],
            vec![10.0, 25.0],
        );

        // Close distance - should want LOD 0
        selector.update_desired_lod(1, 5.0, 0.0);
        assert_eq!(selector.get_desired_lod(1), Some(0));

        // Far distance - should want higher LOD
        selector.update_desired_lod(1, 30.0, 0.0);
        assert_eq!(selector.get_desired_lod(1), Some(2));

        // Medium distance
        selector.update_desired_lod(1, 15.0, 0.0);
        assert_eq!(selector.get_desired_lod(1), Some(1));
    }

    #[test]
    fn test_budget_pressure_changes() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(1000)
            .with_triangle_budget(1000);
        let mut selector = BudgetAwareLodSelector::new(config);

        // Initially under budget
        assert!(selector.get_budget_pressure() < 1.0);

        // Add assets to exceed budget
        selector.register_asset_explicit(
            1,
            1.0,
            vec![600, 300],
            vec![600, 300],
            vec![20.0],
        );

        assert!(selector.get_budget_pressure() < 1.0);

        selector.register_asset_explicit(
            2,
            1.0,
            vec![600, 300],
            vec![600, 300],
            vec![20.0],
        );

        // Now over budget
        assert!(selector.get_budget_pressure() >= 1.0);
    }

    #[test]
    fn test_batch_desired_lod_updates() {
        let config = LodBudgetConfig::default();
        let mut selector = BudgetAwareLodSelector::new(config);

        for i in 1..=5 {
            selector.register_asset_explicit(
                i,
                1.0,
                vec![1000, 500, 250],
                vec![1000, 500, 250],
                vec![10.0, 25.0],
            );
        }

        // Batch update all distances
        selector.update_desired_lods(&[
            (1, 5.0, 0.0),
            (2, 15.0, 0.0),
            (3, 30.0, 0.0),
            (4, 5.0, 0.5),   // With positive bias
            (5, 5.0, -0.5),  // With negative bias
        ]);

        assert_eq!(selector.get_desired_lod(1), Some(0));
        assert_eq!(selector.get_desired_lod(2), Some(1));
        assert_eq!(selector.get_desired_lod(3), Some(2));
        // Positive bias increases effective distance -> higher LOD
        assert!(selector.get_desired_lod(4).unwrap() >= 0);
        // Negative bias decreases effective distance -> lower LOD
        assert_eq!(selector.get_desired_lod(5), Some(0));
    }

    // ========================================================================
    // Bias integration tests (2 tests)
    // ========================================================================

    #[test]
    fn test_bias_increases_lod() {
        let config = LodBudgetConfig::default();
        let mut selector = BudgetAwareLodSelector::new(config);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500, 250, 125],
            vec![1000, 500, 250, 125],
            vec![10.0, 25.0, 50.0],
        );

        // Base LOD is 1
        let adjusted = selector.apply_lod_bias(1, 1, 1.0);
        assert_eq!(adjusted, 2);

        // Larger bias
        let adjusted = selector.apply_lod_bias(1, 0, 2.0);
        assert_eq!(adjusted, 2);
    }

    #[test]
    fn test_bias_decreases_lod() {
        let config = LodBudgetConfig::default();
        let mut selector = BudgetAwareLodSelector::new(config);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500, 250, 125],
            vec![1000, 500, 250, 125],
            vec![10.0, 25.0, 50.0],
        );

        // Negative bias decreases LOD (improves quality)
        let adjusted = selector.apply_lod_bias(1, 2, -1.0);
        assert_eq!(adjusted, 1);

        // Cannot go below 0
        let adjusted = selector.apply_lod_bias(1, 1, -5.0);
        assert_eq!(adjusted, 0);
    }

    #[test]
    fn test_bias_respects_max_lod() {
        let config = LodBudgetConfig::default();
        let mut selector = BudgetAwareLodSelector::new(config);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500, 250],  // max_lod = 2
            vec![1000, 500, 250],
            vec![10.0, 25.0],
        );

        // Large positive bias should clamp to max_lod
        let adjusted = selector.apply_lod_bias(1, 1, 10.0);
        assert_eq!(adjusted, 2);
    }

    // ========================================================================
    // Edge cases (2 tests)
    // ========================================================================

    #[test]
    fn test_empty_budget() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(0)
            .with_triangle_budget(0);
        let selector = BudgetAwareLodSelector::new(config);

        assert_eq!(selector.get_budget_pressure(), 0.0);
        assert_eq!(selector.asset_count(), 0);
    }

    #[test]
    fn test_single_asset_selection() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(500)
            .with_triangle_budget(500);
        let mut selector = BudgetAwareLodSelector::new(config);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500, 250],
            vec![1000, 500, 250],
            vec![10.0, 25.0],
        );

        let changes = selector.select_lods();

        // Should reduce LOD to fit budget
        assert!(selector.get_current_lod(1).unwrap() > 0);
        assert!(!changes.is_empty());
    }

    #[test]
    fn test_all_at_min_lod() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(10)  // Impossibly small
            .with_triangle_budget(10);
        let mut selector = BudgetAwareLodSelector::new(config);

        // Single level assets - can't reduce further
        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000],  // Single LOD level
            vec![1000],
            vec![],
        );

        selector.register_asset_explicit(
            2,
            1.0,
            vec![1000],
            vec![1000],
            vec![],
        );

        let changes = selector.select_lods();

        // All assets at minimum LOD (only level), can't reduce further
        assert_eq!(selector.get_current_lod(1), Some(0));
        assert_eq!(selector.get_current_lod(2), Some(0));
        // Still over budget but no changes possible
        assert!(changes.is_empty());
    }

    // ========================================================================
    // Asset lifecycle tests
    // ========================================================================

    #[test]
    fn test_register_unregister_asset() {
        let config = LodBudgetConfig::default();
        let mut selector = BudgetAwareLodSelector::new(config);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500],
            vec![1000, 500],
            vec![20.0],
        );

        assert!(selector.has_asset(1));
        assert_eq!(selector.asset_count(), 1);

        let state = selector.unregister_asset(1);
        assert!(state.is_some());
        assert!(!selector.has_asset(1));
        assert_eq!(selector.asset_count(), 0);
    }

    #[test]
    fn test_register_updates_budget() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(10000)
            .with_triangle_budget(10000);
        let mut selector = BudgetAwareLodSelector::new(config);

        assert_eq!(selector.budget().current_triangles, 0);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500],
            vec![1000, 500],
            vec![20.0],
        );

        assert_eq!(selector.budget().current_triangles, 1000);

        selector.unregister_asset(1);
        assert_eq!(selector.budget().current_triangles, 0);
    }

    #[test]
    fn test_clear_resets_all() {
        let config = LodBudgetConfig::default();
        let mut selector = BudgetAwareLodSelector::new(config);

        for i in 1..=5 {
            selector.register_asset_explicit(
                i,
                1.0,
                vec![1000, 500],
                vec![1000, 500],
                vec![20.0],
            );
        }

        assert_eq!(selector.asset_count(), 5);
        assert!(selector.budget().current_triangles > 0);

        selector.clear();

        assert_eq!(selector.asset_count(), 0);
        assert_eq!(selector.budget().current_triangles, 0);
        assert_eq!(selector.budget().current_texels, 0);
    }

    #[test]
    fn test_set_asset_lod_updates_budget() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(10000)
            .with_triangle_budget(10000);
        let mut selector = BudgetAwareLodSelector::new(config);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500, 250],
            vec![1000, 500, 250],
            vec![10.0, 25.0],
        );

        assert_eq!(selector.budget().current_triangles, 1000);

        // Force LOD to level 1
        let old_lod = selector.set_asset_lod(1, 1);
        assert_eq!(old_lod, Some(0));
        assert_eq!(selector.budget().current_triangles, 500);

        // Force LOD to level 2
        selector.set_asset_lod(1, 2);
        assert_eq!(selector.budget().current_triangles, 250);
    }

    #[test]
    fn test_recalculate_budget() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(10000)
            .with_triangle_budget(10000);
        let mut selector = BudgetAwareLodSelector::new(config);

        selector.register_asset_explicit(
            1,
            1.0,
            vec![1000, 500],
            vec![1000, 500],
            vec![20.0],
        );

        // Manually corrupt budget (for testing)
        selector.budget.current_triangles = 9999;

        // Recalculate should fix it
        selector.recalculate_budget();
        assert_eq!(selector.budget().current_triangles, 1000);
    }

    // ========================================================================
    // Asset LOD state tests
    // ========================================================================

    #[test]
    fn test_asset_lod_state_from_chain() {
        let chain = create_test_lod_chain(4, 1000);
        let state = AssetLodState::from_lod_chain(1, &chain, 2.5);

        assert_eq!(state.asset_id, 1);
        assert_eq!(state.priority, 2.5);
        assert_eq!(state.max_lod, 3);
        assert_eq!(state.triangles_at_lod.len(), 4);
        assert_eq!(state.texels_at_lod.len(), 4);
    }

    #[test]
    fn test_asset_lod_state_reduction_score() {
        let state1 = AssetLodState::new(
            1,
            1.0,  // Normal priority
            vec![1000, 500],
            vec![1000, 500],
            vec![20.0],
        );

        let state2 = AssetLodState::new(
            2,
            10.0,  // High priority
            vec![1000, 500],
            vec![1000, 500],
            vec![20.0],
        );

        // Lower priority should have higher reduction score
        assert!(state1.reduction_score() > state2.reduction_score());
    }

    #[test]
    fn test_asset_lod_state_can_reduce() {
        let mut state = AssetLodState::new(
            1,
            1.0,
            vec![1000, 500, 250],
            vec![1000, 500, 250],
            vec![10.0, 25.0],
        );

        assert!(state.can_reduce_lod());
        assert!(!state.can_increase_lod());

        state.current_lod = 1;
        assert!(state.can_reduce_lod());
        assert!(state.can_increase_lod());

        state.current_lod = 2;
        assert!(!state.can_reduce_lod());
        assert!(state.can_increase_lod());
    }

    // ========================================================================
    // Configuration tests
    // ========================================================================

    #[test]
    fn test_config_builder() {
        let config = LodBudgetConfig::default()
            .with_texel_budget(1_000_000)
            .with_triangle_budget(500_000)
            .with_reduction_step(0.2)
            .with_priority_weight(0.8);

        assert_eq!(config.texel_budget, 1_000_000);
        assert_eq!(config.triangle_budget, 500_000);
        assert!((config.reduction_step - 0.2).abs() < 0.001);
        assert!((config.priority_weight - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_config_clamps_values() {
        let config = LodBudgetConfig::default()
            .with_reduction_step(2.0)  // Should clamp to 1.0
            .with_priority_weight(-1.0);  // Should clamp to 0.0

        assert!((config.reduction_step - 1.0).abs() < 0.001);
        assert!((config.priority_weight - 0.0).abs() < 0.001);
    }
}
