//! Ray Budget Management
//!
//! Prevents frame-time spikes by limiting total rays per frame and providing
//! graceful degradation when budget is exceeded.

use std::collections::HashMap;

/// Types of ray-traced effects that consume budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RayEffect {
    /// Shadow rays for hard/soft shadows
    Shadows,
    /// Reflection rays for mirror-like surfaces
    Reflections,
    /// Global illumination rays for indirect lighting
    GlobalIllumination,
    /// Ambient occlusion rays for contact shadows
    AmbientOcclusion,
}

impl RayEffect {
    /// Returns all ray effect variants in priority order (highest first by default config).
    pub fn all() -> &'static [RayEffect] {
        &[
            RayEffect::Shadows,
            RayEffect::Reflections,
            RayEffect::GlobalIllumination,
            RayEffect::AmbientOcclusion,
        ]
    }
}

/// Configuration for the ray budget system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RayBudgetConfig {
    /// Maximum rays allowed per frame across all effects.
    pub max_rays_per_frame: u64,
    /// Priority for shadow rays (higher = more important, gets budget first).
    pub shadow_priority: u32,
    /// Priority for reflection rays.
    pub reflection_priority: u32,
    /// Priority for global illumination rays.
    pub gi_priority: u32,
    /// Priority for ambient occlusion rays.
    pub ao_priority: u32,
}

impl Default for RayBudgetConfig {
    fn default() -> Self {
        Self {
            max_rays_per_frame: 50_000_000, // 50M rays
            shadow_priority: 100,
            reflection_priority: 80,
            gi_priority: 60,
            ao_priority: 40,
        }
    }
}

impl RayBudgetConfig {
    /// Creates a new config with custom maximum rays per frame.
    pub fn with_max_rays(max_rays: u64) -> Self {
        Self {
            max_rays_per_frame: max_rays,
            ..Default::default()
        }
    }

    /// Returns the priority for a given effect.
    pub fn priority_for(&self, effect: RayEffect) -> u32 {
        match effect {
            RayEffect::Shadows => self.shadow_priority,
            RayEffect::Reflections => self.reflection_priority,
            RayEffect::GlobalIllumination => self.gi_priority,
            RayEffect::AmbientOcclusion => self.ao_priority,
        }
    }

    /// Returns effects sorted by priority (highest first).
    pub fn effects_by_priority(&self) -> Vec<RayEffect> {
        let mut effects: Vec<RayEffect> = RayEffect::all().to_vec();
        effects.sort_by(|a, b| self.priority_for(*b).cmp(&self.priority_for(*a)));
        effects
    }
}

/// Result of a ray budget allocation request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayAllocation {
    /// Number of rays actually granted.
    pub rays_granted: u64,
    /// Whether the allocation was degraded from the request.
    pub is_degraded: bool,
    /// Degradation factor: 1.0 = full allocation, 0.5 = half-res, 0.0 = disabled.
    pub degradation_factor: f32,
}

impl RayAllocation {
    /// Creates a full allocation (no degradation).
    pub fn full(rays: u64) -> Self {
        Self {
            rays_granted: rays,
            is_degraded: false,
            degradation_factor: 1.0,
        }
    }

    /// Creates a degraded allocation.
    pub fn degraded(rays_granted: u64, requested: u64) -> Self {
        let factor = if requested > 0 {
            rays_granted as f32 / requested as f32
        } else {
            1.0
        };
        Self {
            rays_granted,
            is_degraded: true,
            degradation_factor: factor,
        }
    }

    /// Creates a zero allocation (budget exhausted).
    pub fn exhausted() -> Self {
        Self {
            rays_granted: 0,
            is_degraded: true,
            degradation_factor: 0.0,
        }
    }

    /// Returns true if this allocation suggests using half resolution.
    pub fn should_use_half_res(&self) -> bool {
        self.degradation_factor > 0.0 && self.degradation_factor <= 0.5
    }

    /// Returns true if this effect should be disabled entirely.
    pub fn should_disable(&self) -> bool {
        self.degradation_factor == 0.0
    }
}

/// Manages ray budget allocation across effects per frame.
#[derive(Clone, Debug)]
pub struct RayBudget {
    config: RayBudgetConfig,
    allocated: HashMap<RayEffect, u64>,
    remaining: u64,
    frame: u64,
}

impl RayBudget {
    /// Creates a new ray budget manager with the given configuration.
    pub fn new(config: RayBudgetConfig) -> Self {
        let remaining = config.max_rays_per_frame;
        Self {
            config,
            allocated: HashMap::new(),
            remaining,
            frame: 0,
        }
    }

    /// Resets the budget for a new frame.
    pub fn reset(&mut self) {
        self.allocated.clear();
        self.remaining = self.config.max_rays_per_frame;
        self.frame = self.frame.wrapping_add(1);
    }

    /// Attempts to allocate rays for an effect.
    ///
    /// Returns a `RayAllocation` describing how many rays were granted
    /// and whether degradation is needed.
    pub fn allocate(&mut self, effect: RayEffect, requested_rays: u64) -> RayAllocation {
        if requested_rays == 0 {
            return RayAllocation::full(0);
        }

        let allocation = if self.remaining >= requested_rays {
            // Full allocation
            RayAllocation::full(requested_rays)
        } else if self.remaining > 0 {
            // Partial allocation - degraded
            RayAllocation::degraded(self.remaining, requested_rays)
        } else {
            // Budget exhausted
            RayAllocation::exhausted()
        };

        // Track allocation
        let granted = allocation.rays_granted;
        *self.allocated.entry(effect).or_insert(0) += granted;
        self.remaining = self.remaining.saturating_sub(granted);

        allocation
    }

    /// Returns the remaining ray budget for this frame.
    pub fn remaining_budget(&self) -> u64 {
        self.remaining
    }

    /// Returns rays allocated for a specific effect this frame.
    pub fn allocated_for(&self, effect: RayEffect) -> u64 {
        self.allocated.get(&effect).copied().unwrap_or(0)
    }

    /// Returns total rays allocated across all effects this frame.
    pub fn total_allocated(&self) -> u64 {
        self.allocated.values().sum()
    }

    /// Returns true if the budget is completely exhausted.
    pub fn budget_exhausted(&self) -> bool {
        self.remaining == 0
    }

    /// Returns the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.frame
    }

    /// Returns the configuration.
    pub fn config(&self) -> &RayBudgetConfig {
        &self.config
    }

    /// Updates the configuration. Does not reset current allocations.
    pub fn set_config(&mut self, config: RayBudgetConfig) {
        self.config = config;
    }

    /// Returns the percentage of budget used (0.0 to 1.0).
    pub fn utilization(&self) -> f32 {
        if self.config.max_rays_per_frame == 0 {
            return 0.0;
        }
        self.total_allocated() as f32 / self.config.max_rays_per_frame as f32
    }

    /// Suggests which effect to degrade next based on priorities.
    ///
    /// Returns the lowest-priority effect that still has allocations,
    /// or None if nothing can be degraded.
    pub fn suggest_degradation_target(&self) -> Option<RayEffect> {
        let effects_by_priority = self.config.effects_by_priority();
        // Return lowest priority effect that has allocations
        for effect in effects_by_priority.iter().rev() {
            if self.allocated_for(*effect) > 0 {
                return Some(*effect);
            }
        }
        None
    }

    /// Computes a degradation cascade based on budget pressure.
    ///
    /// Returns a map of effects to their suggested degradation factors
    /// based on how much over budget we are.
    pub fn compute_degradation_cascade(&self, pressure: f32) -> HashMap<RayEffect, f32> {
        let mut cascade = HashMap::new();
        let effects = self.config.effects_by_priority();

        // Pressure 0.0 = no degradation needed
        // Pressure 1.0+ = heavy degradation needed
        for (i, effect) in effects.iter().rev().enumerate() {
            let threshold = (i as f32 + 1.0) * 0.25; // 0.25, 0.5, 0.75, 1.0
            let factor = if pressure >= threshold {
                // Effect should be degraded
                let excess = pressure - threshold;
                (1.0 - excess.min(0.25) * 4.0).max(0.0)
            } else {
                1.0
            };
            cascade.insert(*effect, factor);
        }

        cascade
    }
}

impl Default for RayBudget {
    fn default() -> Self {
        Self::new(RayBudgetConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        let config = RayBudgetConfig::default();
        assert_eq!(config.max_rays_per_frame, 50_000_000);
        assert_eq!(config.shadow_priority, 100);
        assert_eq!(config.reflection_priority, 80);
        assert_eq!(config.gi_priority, 60);
        assert_eq!(config.ao_priority, 40);
    }

    #[test]
    fn test_config_with_max_rays() {
        let config = RayBudgetConfig::with_max_rays(10_000_000);
        assert_eq!(config.max_rays_per_frame, 10_000_000);
        // Other values should be default
        assert_eq!(config.shadow_priority, 100);
    }

    #[test]
    fn test_priority_for_effect() {
        let config = RayBudgetConfig::default();
        assert_eq!(config.priority_for(RayEffect::Shadows), 100);
        assert_eq!(config.priority_for(RayEffect::Reflections), 80);
        assert_eq!(config.priority_for(RayEffect::GlobalIllumination), 60);
        assert_eq!(config.priority_for(RayEffect::AmbientOcclusion), 40);
    }

    #[test]
    fn test_effects_by_priority() {
        let config = RayBudgetConfig::default();
        let effects = config.effects_by_priority();
        assert_eq!(effects[0], RayEffect::Shadows);
        assert_eq!(effects[1], RayEffect::Reflections);
        assert_eq!(effects[2], RayEffect::GlobalIllumination);
        assert_eq!(effects[3], RayEffect::AmbientOcclusion);
    }

    #[test]
    fn test_allocate_within_budget() {
        let mut budget = RayBudget::new(RayBudgetConfig::with_max_rays(1_000_000));
        let alloc = budget.allocate(RayEffect::Shadows, 500_000);

        assert_eq!(alloc.rays_granted, 500_000);
        assert!(!alloc.is_degraded);
        assert_eq!(alloc.degradation_factor, 1.0);
        assert_eq!(budget.remaining_budget(), 500_000);
    }

    #[test]
    fn test_allocate_exceeds_budget_degraded() {
        let mut budget = RayBudget::new(RayBudgetConfig::with_max_rays(1_000_000));
        let alloc = budget.allocate(RayEffect::Shadows, 1_500_000);

        assert_eq!(alloc.rays_granted, 1_000_000);
        assert!(alloc.is_degraded);
        // 1M / 1.5M = 0.666...
        assert!((alloc.degradation_factor - 0.6666667).abs() < 0.001);
        assert_eq!(budget.remaining_budget(), 0);
    }

    #[test]
    fn test_allocate_budget_exhausted() {
        let mut budget = RayBudget::new(RayBudgetConfig::with_max_rays(1_000_000));
        budget.allocate(RayEffect::Shadows, 1_000_000);
        let alloc = budget.allocate(RayEffect::Reflections, 500_000);

        assert_eq!(alloc.rays_granted, 0);
        assert!(alloc.is_degraded);
        assert_eq!(alloc.degradation_factor, 0.0);
        assert!(budget.budget_exhausted());
    }

    #[test]
    fn test_reset_clears_allocations() {
        let mut budget = RayBudget::new(RayBudgetConfig::with_max_rays(1_000_000));
        budget.allocate(RayEffect::Shadows, 500_000);
        budget.allocate(RayEffect::Reflections, 300_000);

        assert_eq!(budget.total_allocated(), 800_000);
        assert_eq!(budget.remaining_budget(), 200_000);

        budget.reset();

        assert_eq!(budget.total_allocated(), 0);
        assert_eq!(budget.remaining_budget(), 1_000_000);
        assert_eq!(budget.allocated_for(RayEffect::Shadows), 0);
    }

    #[test]
    fn test_multiple_effects_compete() {
        let mut budget = RayBudget::new(RayBudgetConfig::with_max_rays(1_000_000));

        let shadow_alloc = budget.allocate(RayEffect::Shadows, 400_000);
        let reflection_alloc = budget.allocate(RayEffect::Reflections, 400_000);
        let gi_alloc = budget.allocate(RayEffect::GlobalIllumination, 400_000);

        assert_eq!(shadow_alloc.rays_granted, 400_000);
        assert!(!shadow_alloc.is_degraded);

        assert_eq!(reflection_alloc.rays_granted, 400_000);
        assert!(!reflection_alloc.is_degraded);

        // GI only gets remaining 200K
        assert_eq!(gi_alloc.rays_granted, 200_000);
        assert!(gi_alloc.is_degraded);
        assert_eq!(gi_alloc.degradation_factor, 0.5);
    }

    #[test]
    fn test_allocation_tracking_per_effect() {
        let mut budget = RayBudget::new(RayBudgetConfig::with_max_rays(10_000_000));

        budget.allocate(RayEffect::Shadows, 1_000_000);
        budget.allocate(RayEffect::Shadows, 500_000);
        budget.allocate(RayEffect::Reflections, 2_000_000);

        assert_eq!(budget.allocated_for(RayEffect::Shadows), 1_500_000);
        assert_eq!(budget.allocated_for(RayEffect::Reflections), 2_000_000);
        assert_eq!(budget.allocated_for(RayEffect::GlobalIllumination), 0);
        assert_eq!(budget.total_allocated(), 3_500_000);
    }

    #[test]
    fn test_budget_exhaustion_detection() {
        let mut budget = RayBudget::new(RayBudgetConfig::with_max_rays(1_000));

        assert!(!budget.budget_exhausted());
        budget.allocate(RayEffect::Shadows, 500);
        assert!(!budget.budget_exhausted());
        budget.allocate(RayEffect::Reflections, 500);
        assert!(budget.budget_exhausted());
    }

    #[test]
    fn test_degradation_factor_half_res() {
        let alloc = RayAllocation::degraded(500_000, 1_000_000);
        assert_eq!(alloc.degradation_factor, 0.5);
        assert!(alloc.should_use_half_res());
        assert!(!alloc.should_disable());
    }

    #[test]
    fn test_degradation_factor_quarter_res() {
        let alloc = RayAllocation::degraded(250_000, 1_000_000);
        assert_eq!(alloc.degradation_factor, 0.25);
        assert!(alloc.should_use_half_res());
        assert!(!alloc.should_disable());
    }

    #[test]
    fn test_degradation_exhausted_should_disable() {
        let alloc = RayAllocation::exhausted();
        assert_eq!(alloc.degradation_factor, 0.0);
        assert!(!alloc.should_use_half_res());
        assert!(alloc.should_disable());
    }

    #[test]
    fn test_frame_counter_increments() {
        let mut budget = RayBudget::default();
        assert_eq!(budget.current_frame(), 0);
        budget.reset();
        assert_eq!(budget.current_frame(), 1);
        budget.reset();
        assert_eq!(budget.current_frame(), 2);
    }

    #[test]
    fn test_utilization_calculation() {
        let mut budget = RayBudget::new(RayBudgetConfig::with_max_rays(1_000_000));
        assert_eq!(budget.utilization(), 0.0);

        budget.allocate(RayEffect::Shadows, 500_000);
        assert_eq!(budget.utilization(), 0.5);

        budget.allocate(RayEffect::Reflections, 500_000);
        assert_eq!(budget.utilization(), 1.0);
    }

    #[test]
    fn test_suggest_degradation_target() {
        let mut budget = RayBudget::default();

        // No allocations - nothing to degrade
        assert!(budget.suggest_degradation_target().is_none());

        budget.allocate(RayEffect::Shadows, 1000);
        budget.allocate(RayEffect::GlobalIllumination, 1000);

        // Should suggest GI (lower priority than shadows)
        assert_eq!(
            budget.suggest_degradation_target(),
            Some(RayEffect::GlobalIllumination)
        );
    }

    #[test]
    fn test_degradation_cascade_no_pressure() {
        let budget = RayBudget::default();
        let cascade = budget.compute_degradation_cascade(0.0);

        assert_eq!(cascade.get(&RayEffect::Shadows), Some(&1.0));
        assert_eq!(cascade.get(&RayEffect::Reflections), Some(&1.0));
        assert_eq!(cascade.get(&RayEffect::GlobalIllumination), Some(&1.0));
        assert_eq!(cascade.get(&RayEffect::AmbientOcclusion), Some(&1.0));
    }

    #[test]
    fn test_degradation_cascade_moderate_pressure() {
        let budget = RayBudget::default();
        let cascade = budget.compute_degradation_cascade(0.5);

        // AO and GI should be degraded, shadows and reflections should be fine
        assert_eq!(cascade.get(&RayEffect::Shadows), Some(&1.0));
        assert_eq!(cascade.get(&RayEffect::Reflections), Some(&1.0));
        // GI is threshold 0.5, so at pressure 0.5 it starts degrading
        let gi_factor = *cascade.get(&RayEffect::GlobalIllumination).unwrap();
        assert!(gi_factor <= 1.0 && gi_factor >= 0.0);
        // AO is threshold 0.25, so at pressure 0.5 it should be heavily degraded
        let ao_factor = *cascade.get(&RayEffect::AmbientOcclusion).unwrap();
        assert!(ao_factor < 1.0);
    }

    #[test]
    fn test_zero_ray_allocation() {
        let mut budget = RayBudget::default();
        let alloc = budget.allocate(RayEffect::Shadows, 0);

        assert_eq!(alloc.rays_granted, 0);
        assert!(!alloc.is_degraded);
        assert_eq!(alloc.degradation_factor, 1.0);
    }

    #[test]
    fn test_ray_effect_all_variants() {
        let all = RayEffect::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&RayEffect::Shadows));
        assert!(all.contains(&RayEffect::Reflections));
        assert!(all.contains(&RayEffect::GlobalIllumination));
        assert!(all.contains(&RayEffect::AmbientOcclusion));
    }

    #[test]
    fn test_set_config() {
        let mut budget = RayBudget::default();
        let new_config = RayBudgetConfig::with_max_rays(100_000_000);
        budget.set_config(new_config);
        assert_eq!(budget.config().max_rays_per_frame, 100_000_000);
    }

    #[test]
    fn test_allocation_full_helper() {
        let alloc = RayAllocation::full(1_000_000);
        assert_eq!(alloc.rays_granted, 1_000_000);
        assert!(!alloc.is_degraded);
        assert_eq!(alloc.degradation_factor, 1.0);
    }

    #[test]
    fn test_config_equality() {
        let a = RayBudgetConfig::default();
        let b = RayBudgetConfig::default();
        assert_eq!(a, b);

        let c = RayBudgetConfig::with_max_rays(100);
        assert_ne!(a, c);
    }
}
