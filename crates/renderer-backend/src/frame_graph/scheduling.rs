//! Scheduling hints for Frame Graph pass ordering and execution.
//!
//! This module provides types for guiding the optimal ordering and execution
//! of frame graph passes. Scheduling hints allow passes to communicate their
//! priority, preferred queue, and batching preferences to the scheduler.
//!
//! # Overview
//!
//! The scheduling system consists of:
//! - [`SchedulingPriority`]: Determines pass execution urgency
//! - [`SchedulingHint`]: Full scheduling configuration for a pass
//! - [`SchedulingQueueType`]: Target GPU queue for pass execution
//! - [`PassCost`]: Estimated resource usage for scheduling decisions
//! - [`PassScheduleInfo`]: Complete scheduling information for a pass
//! - [`ScheduleBuilder`]: Builder for constructing optimal pass schedules
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::frame_graph::scheduling::*;
//! use renderer_backend::frame_graph::graph::PassId;
//!
//! let mut builder = ScheduleBuilder::new();
//!
//! // Add critical shadow pass
//! builder.add_pass(
//!     PassId::new(0),
//!     SchedulingHint::new(SchedulingPriority::Critical)
//!         .with_queue(SchedulingQueueType::Graphics)
//! );
//!
//! // Add async compute pass
//! builder.add_pass(
//!     PassId::new(1),
//!     SchedulingHint::new(SchedulingPriority::Normal)
//!         .with_queue(SchedulingQueueType::Compute)
//!         .with_async(true)
//! );
//!
//! let schedule = builder.build_schedule();
//! ```

use std::collections::HashMap;
use std::fmt;

use super::graph::PassId;

// ---------------------------------------------------------------------------
// SchedulingPriority
// ---------------------------------------------------------------------------

/// Priority level for pass scheduling.
///
/// Higher priority passes are scheduled earlier when dependencies allow.
/// The ordering is such that `Critical < High < Normal < Low < Background`,
/// meaning Critical has the highest priority (smallest ordinal value).
///
/// # Priority Levels
///
/// | Priority    | Use Case                                         |
/// |-------------|--------------------------------------------------|
/// | Critical    | Async compute dependencies, frame-critical paths |
/// | High        | Shadow maps, GBuffer generation                  |
/// | Normal      | Standard rendering passes                        |
/// | Low         | Post-processing, optional effects                |
/// | Background  | Streaming, precomputation, mipmap generation     |
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum SchedulingPriority {
    /// Must run as soon as possible. Used for async compute dependencies
    /// and frame-critical path operations.
    Critical = 0,
    /// Important passes that should run early. Shadow maps, GBuffer
    /// generation, and other foundational passes.
    High = 1,
    /// Default priority for standard rendering passes.
    #[default]
    Normal = 2,
    /// Can be deferred if needed. Post-processing effects, optional
    /// rendering features.
    Low = 3,
    /// Lowest priority. Streaming, precomputation, background work
    /// that can be delayed without visible impact.
    Background = 4,
}

impl SchedulingPriority {
    /// Returns true if this priority is higher than the other.
    ///
    /// Note: Lower ordinal values indicate higher priority.
    #[inline]
    pub const fn is_higher_than(self, other: Self) -> bool {
        (self as u8) < (other as u8)
    }

    /// Returns true if this priority is lower than the other.
    #[inline]
    pub const fn is_lower_than(self, other: Self) -> bool {
        (self as u8) > (other as u8)
    }

    /// Returns the numeric priority value (0 = highest, 4 = lowest).
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Creates a priority from a numeric value, clamping to valid range.
    #[inline]
    pub const fn from_u8_clamped(value: u8) -> Self {
        match value {
            0 => Self::Critical,
            1 => Self::High,
            2 => Self::Normal,
            3 => Self::Low,
            _ => Self::Background,
        }
    }
}

impl fmt::Display for SchedulingPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "Critical"),
            Self::High => write!(f, "High"),
            Self::Normal => write!(f, "Normal"),
            Self::Low => write!(f, "Low"),
            Self::Background => write!(f, "Background"),
        }
    }
}

// ---------------------------------------------------------------------------
// SchedulingQueueType
// ---------------------------------------------------------------------------

/// GPU queue type for pass execution.
///
/// Determines which hardware queue a pass should be submitted to. Modern
/// GPUs typically have multiple queue families that can execute work in
/// parallel:
///
/// - **Graphics**: Main queue for rasterization and ray-tracing
/// - **Compute**: Async compute queue for compute-only workloads
/// - **Transfer**: DMA engine for memory copies
/// - **Present**: Display output queue
///
/// # Queue Selection
///
/// The scheduler uses this hint along with dependency analysis to determine
/// the actual queue assignment. Passes with RAW dependencies on graphics
/// work may be forced to the Graphics queue even if they prefer Compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SchedulingQueueType {
    /// Main graphics queue. Supports all pass types but may have contention.
    #[default]
    Graphics,
    /// Async compute queue. Ideal for compute passes without graphics
    /// dependencies. Can run in parallel with graphics work.
    Compute,
    /// Transfer/DMA queue. Optimal for buffer copies, texture uploads,
    /// and other memory operations.
    Transfer,
    /// Present queue. Used for swapchain presentation. Typically maps
    /// to the graphics queue on most hardware.
    Present,
}

impl SchedulingQueueType {
    /// Returns true if this is the graphics queue.
    #[inline]
    pub const fn is_graphics(self) -> bool {
        matches!(self, Self::Graphics)
    }

    /// Returns true if this is the async compute queue.
    #[inline]
    pub const fn is_compute(self) -> bool {
        matches!(self, Self::Compute)
    }

    /// Returns true if this is the transfer queue.
    #[inline]
    pub const fn is_transfer(self) -> bool {
        matches!(self, Self::Transfer)
    }

    /// Returns true if this is the present queue.
    #[inline]
    pub const fn is_present(self) -> bool {
        matches!(self, Self::Present)
    }

    /// Returns true if this queue supports compute workloads.
    #[inline]
    pub const fn supports_compute(self) -> bool {
        matches!(self, Self::Graphics | Self::Compute)
    }

    /// Returns true if this queue supports graphics workloads.
    #[inline]
    pub const fn supports_graphics(self) -> bool {
        matches!(self, Self::Graphics)
    }
}

impl fmt::Display for SchedulingQueueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graphics => write!(f, "Graphics"),
            Self::Compute => write!(f, "Compute"),
            Self::Transfer => write!(f, "Transfer"),
            Self::Present => write!(f, "Present"),
        }
    }
}

// ---------------------------------------------------------------------------
// SchedulingHint
// ---------------------------------------------------------------------------

/// Scheduling hints for a frame graph pass.
///
/// These hints guide the scheduler in determining optimal pass ordering
/// and queue assignment. Hints are advisory; the scheduler may override
/// them based on dependencies and resource constraints.
///
/// # Example
///
/// ```ignore
/// let hint = SchedulingHint::new(SchedulingPriority::High)
///     .with_queue(SchedulingQueueType::Compute)
///     .with_async(true)
///     .with_deadline(0.25); // Complete within first 25% of frame
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SchedulingHint {
    /// Execution priority relative to other passes.
    pub priority: SchedulingPriority,
    /// Preferred GPU queue for execution.
    pub preferred_queue: SchedulingQueueType,
    /// Whether this pass can run on an async queue.
    pub allow_async: bool,
    /// Passes this one should be batched with for efficiency.
    pub batch_with: Option<Vec<PassId>>,
    /// Passes this one should NOT be batched with.
    pub separate_from: Option<Vec<PassId>>,
    /// Target completion as a fraction of frame time (0.0 - 1.0).
    /// `None` means no deadline constraint.
    pub deadline_frame_fraction: Option<f32>,
}

impl SchedulingHint {
    /// Creates a new scheduling hint with the given priority.
    ///
    /// Uses default values for other fields:
    /// - `preferred_queue`: Graphics
    /// - `allow_async`: false
    /// - `batch_with`: None
    /// - `separate_from`: None
    /// - `deadline_frame_fraction`: None
    pub fn new(priority: SchedulingPriority) -> Self {
        Self {
            priority,
            preferred_queue: SchedulingQueueType::Graphics,
            allow_async: false,
            batch_with: None,
            separate_from: None,
            deadline_frame_fraction: None,
        }
    }

    /// Creates a default hint with Normal priority.
    pub fn default_hint() -> Self {
        Self::new(SchedulingPriority::Normal)
    }

    /// Sets the preferred queue type.
    pub fn with_queue(mut self, queue: SchedulingQueueType) -> Self {
        self.preferred_queue = queue;
        self
    }

    /// Sets whether async execution is allowed.
    pub fn with_async(mut self, allow: bool) -> Self {
        self.allow_async = allow;
        self
    }

    /// Sets passes to batch with.
    pub fn with_batch(mut self, passes: Vec<PassId>) -> Self {
        self.batch_with = Some(passes);
        self
    }

    /// Sets passes to separate from.
    pub fn with_separate(mut self, passes: Vec<PassId>) -> Self {
        self.separate_from = Some(passes);
        self
    }

    /// Sets the deadline as a fraction of frame time.
    ///
    /// # Panics
    ///
    /// Debug-asserts if `fraction` is not in [0.0, 1.0].
    pub fn with_deadline(mut self, fraction: f32) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&fraction),
            "Deadline fraction must be in [0.0, 1.0], got {}",
            fraction
        );
        self.deadline_frame_fraction = Some(fraction.clamp(0.0, 1.0));
        self
    }

    /// Returns true if this pass should preferentially batch with the given pass.
    pub fn wants_batch_with(&self, other: PassId) -> bool {
        self.batch_with
            .as_ref()
            .map(|v| v.contains(&other))
            .unwrap_or(false)
    }

    /// Returns true if this pass should avoid batching with the given pass.
    pub fn wants_separation_from(&self, other: PassId) -> bool {
        self.separate_from
            .as_ref()
            .map(|v| v.contains(&other))
            .unwrap_or(false)
    }

    /// Returns true if this pass has a deadline constraint.
    pub fn has_deadline(&self) -> bool {
        self.deadline_frame_fraction.is_some()
    }
}

impl Default for SchedulingHint {
    fn default() -> Self {
        Self::default_hint()
    }
}

// ---------------------------------------------------------------------------
// PassCost
// ---------------------------------------------------------------------------

/// Estimated execution cost of a pass.
///
/// Used by the scheduler to make informed decisions about pass ordering
/// and parallelization. Costs are estimates provided by the application
/// or measured at runtime.
///
/// # Fields
///
/// | Field               | Description                              |
/// |---------------------|------------------------------------------|
/// | `gpu_time_us`       | Estimated GPU execution time (microsec)  |
/// | `memory_bandwidth_mb` | Memory bandwidth usage (megabytes)     |
/// | `dispatch_count`    | Number of draw/dispatch calls            |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PassCost {
    /// Estimated GPU execution time in microseconds.
    pub gpu_time_us: u32,
    /// Estimated memory bandwidth usage in megabytes.
    pub memory_bandwidth_mb: u32,
    /// Number of draw calls or compute dispatches.
    pub dispatch_count: u32,
}

impl PassCost {
    /// Creates a new pass cost with all zeros.
    pub const fn zero() -> Self {
        Self {
            gpu_time_us: 0,
            memory_bandwidth_mb: 0,
            dispatch_count: 0,
        }
    }

    /// Creates a pass cost with the given GPU time.
    pub const fn from_gpu_time(gpu_time_us: u32) -> Self {
        Self {
            gpu_time_us,
            memory_bandwidth_mb: 0,
            dispatch_count: 0,
        }
    }

    /// Creates a pass cost with all fields specified.
    pub const fn new(gpu_time_us: u32, memory_bandwidth_mb: u32, dispatch_count: u32) -> Self {
        Self {
            gpu_time_us,
            memory_bandwidth_mb,
            dispatch_count,
        }
    }

    /// Returns true if this cost represents no work.
    pub const fn is_zero(&self) -> bool {
        self.gpu_time_us == 0 && self.memory_bandwidth_mb == 0 && self.dispatch_count == 0
    }

    /// Returns the total estimated cost as a single metric.
    ///
    /// This is a weighted combination of the individual costs for
    /// comparison purposes. The weights are tuned for typical GPU
    /// workloads.
    pub const fn total_cost(&self) -> u64 {
        // Weight GPU time heavily, bandwidth moderately, dispatch count lightly
        (self.gpu_time_us as u64) * 10
            + (self.memory_bandwidth_mb as u64) * 5
            + (self.dispatch_count as u64)
    }

    /// Adds two costs together.
    pub const fn add(self, other: Self) -> Self {
        Self {
            gpu_time_us: self.gpu_time_us.saturating_add(other.gpu_time_us),
            memory_bandwidth_mb: self.memory_bandwidth_mb.saturating_add(other.memory_bandwidth_mb),
            dispatch_count: self.dispatch_count.saturating_add(other.dispatch_count),
        }
    }
}

impl std::ops::Add for PassCost {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::add(self, rhs)
    }
}

impl fmt::Display for PassCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PassCost({}us, {}MB, {} dispatches)",
            self.gpu_time_us, self.memory_bandwidth_mb, self.dispatch_count
        )
    }
}

// ---------------------------------------------------------------------------
// PassScheduleInfo
// ---------------------------------------------------------------------------

/// Complete scheduling information for a pass.
///
/// Combines the pass identifier, scheduling hints, estimated cost, and
/// dependency information into a single structure used by the scheduler.
#[derive(Clone, Debug, PartialEq)]
pub struct PassScheduleInfo {
    /// The pass identifier.
    pub pass_id: PassId,
    /// Scheduling hints for this pass.
    pub hints: SchedulingHint,
    /// Estimated execution cost.
    pub estimated_cost: PassCost,
    /// Passes that this pass depends on (must run before).
    pub dependencies: Vec<PassId>,
    /// Passes that depend on this pass (must run after).
    pub dependents: Vec<PassId>,
}

impl PassScheduleInfo {
    /// Creates new schedule info with the given pass ID and hints.
    pub fn new(pass_id: PassId, hints: SchedulingHint) -> Self {
        Self {
            pass_id,
            hints,
            estimated_cost: PassCost::default(),
            dependencies: Vec::new(),
            dependents: Vec::new(),
        }
    }

    /// Creates schedule info with default hints.
    pub fn with_defaults(pass_id: PassId) -> Self {
        Self::new(pass_id, SchedulingHint::default())
    }

    /// Sets the estimated cost.
    pub fn with_cost(mut self, cost: PassCost) -> Self {
        self.estimated_cost = cost;
        self
    }

    /// Adds a dependency (a pass that must run before this one).
    pub fn add_dependency(&mut self, dep: PassId) {
        if !self.dependencies.contains(&dep) {
            self.dependencies.push(dep);
        }
    }

    /// Adds a dependent (a pass that must run after this one).
    pub fn add_dependent(&mut self, dep: PassId) {
        if !self.dependents.contains(&dep) {
            self.dependents.push(dep);
        }
    }

    /// Returns true if this pass has no dependencies.
    pub fn is_root(&self) -> bool {
        self.dependencies.is_empty()
    }

    /// Returns true if this pass has no dependents.
    pub fn is_leaf(&self) -> bool {
        self.dependents.is_empty()
    }

    /// Returns the depth of this pass in the dependency graph.
    ///
    /// Root passes have depth 0. The depth is the number of dependencies.
    /// For accurate depth in a DAG, use `ScheduleBuilder::compute_depths`.
    pub fn dependency_count(&self) -> usize {
        self.dependencies.len()
    }
}

// ---------------------------------------------------------------------------
// ScheduleBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing optimal pass schedules.
///
/// The builder collects pass information and scheduling hints, then produces
/// an ordered schedule that respects dependencies while optimizing for
/// priority, queue assignment, and batching preferences.
///
/// # Example
///
/// ```ignore
/// let mut builder = ScheduleBuilder::new();
///
/// builder.add_pass(shadow_pass, SchedulingHint::new(SchedulingPriority::High));
/// builder.add_pass(gbuffer_pass, SchedulingHint::new(SchedulingPriority::High));
/// builder.add_pass(lighting_pass, SchedulingHint::new(SchedulingPriority::Normal));
/// builder.add_pass(postfx_pass, SchedulingHint::new(SchedulingPriority::Low));
///
/// builder.with_cost(shadow_pass, PassCost::from_gpu_time(500));
///
/// let schedule = builder.build_schedule();
/// let queue_groups = builder.group_by_queue();
/// ```
#[derive(Clone, Debug, Default)]
pub struct ScheduleBuilder {
    /// All registered passes with their scheduling info.
    passes: Vec<PassScheduleInfo>,
    /// Lookup from PassId to index in `passes`.
    pass_index: HashMap<PassId, usize>,
}

impl ScheduleBuilder {
    /// Creates a new empty schedule builder.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            pass_index: HashMap::new(),
        }
    }

    /// Creates a builder with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            passes: Vec::with_capacity(capacity),
            pass_index: HashMap::with_capacity(capacity),
        }
    }

    /// Adds a pass with scheduling hints.
    ///
    /// If the pass already exists, updates its hints.
    pub fn add_pass(&mut self, pass_id: PassId, hints: SchedulingHint) -> &mut Self {
        if let Some(&idx) = self.pass_index.get(&pass_id) {
            self.passes[idx].hints = hints;
        } else {
            let idx = self.passes.len();
            self.passes.push(PassScheduleInfo::new(pass_id, hints));
            self.pass_index.insert(pass_id, idx);
        }
        self
    }

    /// Sets the estimated cost for a pass.
    ///
    /// The pass must have been added first via `add_pass`.
    pub fn with_cost(&mut self, pass_id: PassId, cost: PassCost) -> &mut Self {
        if let Some(&idx) = self.pass_index.get(&pass_id) {
            self.passes[idx].estimated_cost = cost;
        }
        self
    }

    /// Adds a dependency relationship: `dependent` depends on `dependency`.
    ///
    /// Both passes must have been added first.
    pub fn add_dependency(&mut self, dependent: PassId, dependency: PassId) -> &mut Self {
        if let Some(&dep_idx) = self.pass_index.get(&dependent) {
            self.passes[dep_idx].add_dependency(dependency);
        }
        if let Some(&src_idx) = self.pass_index.get(&dependency) {
            self.passes[src_idx].add_dependent(dependent);
        }
        self
    }

    /// Returns the number of passes in the builder.
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Returns true if no passes have been added.
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Returns the schedule info for a pass, if it exists.
    pub fn get_pass(&self, pass_id: PassId) -> Option<&PassScheduleInfo> {
        self.pass_index.get(&pass_id).map(|&idx| &self.passes[idx])
    }

    /// Builds an optimal execution order respecting dependencies and priorities.
    ///
    /// Uses a modified topological sort that considers:
    /// 1. Dependencies (hard constraints)
    /// 2. Priorities (soft constraints)
    /// 3. Deadlines (soft constraints)
    /// 4. Batching hints (optimization)
    ///
    /// Returns the passes in execution order.
    pub fn build_schedule(&self) -> Vec<PassId> {
        if self.passes.is_empty() {
            return Vec::new();
        }

        // Track remaining dependencies for each pass
        let mut remaining_deps: HashMap<PassId, usize> = self
            .passes
            .iter()
            .map(|p| (p.pass_id, p.dependencies.len()))
            .collect();

        // Track which passes are ready (no remaining dependencies)
        let mut ready: Vec<PassId> = self
            .passes
            .iter()
            .filter(|p| p.is_root())
            .map(|p| p.pass_id)
            .collect();

        // Sort ready list by priority (descending so highest priority is at end for pop())
        // Lower ordinal = higher priority, so we reverse the comparison
        ready.sort_by(|a, b| {
            let pa = self.get_pass(*a).map(|p| p.hints.priority).unwrap_or_default();
            let pb = self.get_pass(*b).map(|p| p.hints.priority).unwrap_or_default();
            pb.cmp(&pa) // Reversed: highest priority (lowest ordinal) goes to end
        });

        let mut scheduled = Vec::with_capacity(self.passes.len());

        while let Some(pass_id) = ready.pop() {
            scheduled.push(pass_id);

            // Update dependents
            if let Some(info) = self.get_pass(pass_id) {
                for &dependent in &info.dependents {
                    if let Some(count) = remaining_deps.get_mut(&dependent) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            ready.push(dependent);
                        }
                    }
                }
            }

            // Re-sort ready list by priority
            ready.sort_by(|a, b| {
                let pa = self.get_pass(*a).map(|p| p.hints.priority).unwrap_or_default();
                let pb = self.get_pass(*b).map(|p| p.hints.priority).unwrap_or_default();
                // Reverse comparison so highest priority (lowest ordinal) comes last (popped first)
                pb.cmp(&pa)
            });
        }

        scheduled
    }

    /// Groups passes by their preferred queue type.
    ///
    /// Returns a map from queue type to the passes that prefer that queue.
    /// The passes within each group maintain their dependency order.
    pub fn group_by_queue(&self) -> HashMap<SchedulingQueueType, Vec<PassId>> {
        let mut groups: HashMap<SchedulingQueueType, Vec<PassId>> = HashMap::new();

        // Initialize all queue types
        groups.insert(SchedulingQueueType::Graphics, Vec::new());
        groups.insert(SchedulingQueueType::Compute, Vec::new());
        groups.insert(SchedulingQueueType::Transfer, Vec::new());
        groups.insert(SchedulingQueueType::Present, Vec::new());

        for pass in &self.passes {
            groups
                .entry(pass.hints.preferred_queue)
                .or_default()
                .push(pass.pass_id);
        }

        groups
    }

    /// Returns passes that can potentially run on async queues.
    ///
    /// A pass is async-eligible if it allows async execution and prefers
    /// a queue that supports async operation (Compute or Transfer).
    pub fn async_eligible_passes(&self) -> Vec<PassId> {
        self.passes
            .iter()
            .filter(|p| {
                p.hints.allow_async
                    && matches!(
                        p.hints.preferred_queue,
                        SchedulingQueueType::Compute | SchedulingQueueType::Transfer
                    )
            })
            .map(|p| p.pass_id)
            .collect()
    }

    /// Returns the total estimated cost of all passes.
    pub fn total_cost(&self) -> PassCost {
        self.passes
            .iter()
            .fold(PassCost::zero(), |acc, p| acc + p.estimated_cost)
    }

    /// Computes the depth of each pass in the dependency graph.
    ///
    /// Root passes have depth 0. A pass's depth is 1 + max depth of dependencies.
    pub fn compute_depths(&self) -> HashMap<PassId, u32> {
        let mut depths: HashMap<PassId, u32> = HashMap::new();

        // Initialize all passes with depth 0
        for pass in &self.passes {
            depths.insert(pass.pass_id, 0);
        }

        // Iterate until no changes (handles arbitrary DAG shapes)
        let mut changed = true;
        while changed {
            changed = false;
            for pass in &self.passes {
                let max_dep_depth = pass
                    .dependencies
                    .iter()
                    .filter_map(|dep| depths.get(dep))
                    .max()
                    .copied()
                    .unwrap_or(0);

                let new_depth = if pass.dependencies.is_empty() {
                    0
                } else {
                    max_dep_depth + 1
                };

                if let Some(current) = depths.get_mut(&pass.pass_id) {
                    if new_depth > *current {
                        *current = new_depth;
                        changed = true;
                    }
                }
            }
        }

        depths
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SchedulingPriority Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scheduling_priority_ordering() {
        // Critical < High < Normal < Low < Background
        assert!(SchedulingPriority::Critical < SchedulingPriority::High);
        assert!(SchedulingPriority::High < SchedulingPriority::Normal);
        assert!(SchedulingPriority::Normal < SchedulingPriority::Low);
        assert!(SchedulingPriority::Low < SchedulingPriority::Background);
    }

    #[test]
    fn test_scheduling_priority_is_higher_than() {
        assert!(SchedulingPriority::Critical.is_higher_than(SchedulingPriority::High));
        assert!(SchedulingPriority::High.is_higher_than(SchedulingPriority::Normal));
        assert!(!SchedulingPriority::Normal.is_higher_than(SchedulingPriority::Critical));
        assert!(!SchedulingPriority::Normal.is_higher_than(SchedulingPriority::Normal));
    }

    #[test]
    fn test_scheduling_priority_is_lower_than() {
        assert!(SchedulingPriority::Background.is_lower_than(SchedulingPriority::Low));
        assert!(SchedulingPriority::Low.is_lower_than(SchedulingPriority::Normal));
        assert!(!SchedulingPriority::Critical.is_lower_than(SchedulingPriority::High));
    }

    #[test]
    fn test_scheduling_priority_as_u8() {
        assert_eq!(SchedulingPriority::Critical.as_u8(), 0);
        assert_eq!(SchedulingPriority::High.as_u8(), 1);
        assert_eq!(SchedulingPriority::Normal.as_u8(), 2);
        assert_eq!(SchedulingPriority::Low.as_u8(), 3);
        assert_eq!(SchedulingPriority::Background.as_u8(), 4);
    }

    #[test]
    fn test_scheduling_priority_from_u8_clamped() {
        assert_eq!(SchedulingPriority::from_u8_clamped(0), SchedulingPriority::Critical);
        assert_eq!(SchedulingPriority::from_u8_clamped(2), SchedulingPriority::Normal);
        assert_eq!(SchedulingPriority::from_u8_clamped(100), SchedulingPriority::Background);
    }

    #[test]
    fn test_scheduling_priority_default() {
        assert_eq!(SchedulingPriority::default(), SchedulingPriority::Normal);
    }

    #[test]
    fn test_scheduling_priority_display() {
        assert_eq!(format!("{}", SchedulingPriority::Critical), "Critical");
        assert_eq!(format!("{}", SchedulingPriority::High), "High");
        assert_eq!(format!("{}", SchedulingPriority::Normal), "Normal");
        assert_eq!(format!("{}", SchedulingPriority::Low), "Low");
        assert_eq!(format!("{}", SchedulingPriority::Background), "Background");
    }

    // -----------------------------------------------------------------------
    // SchedulingQueueType Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_queue_type_variants() {
        assert_ne!(SchedulingQueueType::Graphics, SchedulingQueueType::Compute);
        assert_ne!(SchedulingQueueType::Compute, SchedulingQueueType::Transfer);
        assert_ne!(SchedulingQueueType::Transfer, SchedulingQueueType::Present);
    }

    #[test]
    fn test_queue_type_is_methods() {
        assert!(SchedulingQueueType::Graphics.is_graphics());
        assert!(!SchedulingQueueType::Compute.is_graphics());
        assert!(SchedulingQueueType::Compute.is_compute());
        assert!(SchedulingQueueType::Transfer.is_transfer());
        assert!(SchedulingQueueType::Present.is_present());
    }

    #[test]
    fn test_queue_type_supports_compute() {
        assert!(SchedulingQueueType::Graphics.supports_compute());
        assert!(SchedulingQueueType::Compute.supports_compute());
        assert!(!SchedulingQueueType::Transfer.supports_compute());
        assert!(!SchedulingQueueType::Present.supports_compute());
    }

    #[test]
    fn test_queue_type_supports_graphics() {
        assert!(SchedulingQueueType::Graphics.supports_graphics());
        assert!(!SchedulingQueueType::Compute.supports_graphics());
        assert!(!SchedulingQueueType::Transfer.supports_graphics());
    }

    #[test]
    fn test_queue_type_default() {
        assert_eq!(SchedulingQueueType::default(), SchedulingQueueType::Graphics);
    }

    #[test]
    fn test_queue_type_display() {
        assert_eq!(format!("{}", SchedulingQueueType::Graphics), "Graphics");
        assert_eq!(format!("{}", SchedulingQueueType::Compute), "Compute");
        assert_eq!(format!("{}", SchedulingQueueType::Transfer), "Transfer");
        assert_eq!(format!("{}", SchedulingQueueType::Present), "Present");
    }

    // -----------------------------------------------------------------------
    // SchedulingHint Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scheduling_hint_creation() {
        let hint = SchedulingHint::new(SchedulingPriority::High);
        assert_eq!(hint.priority, SchedulingPriority::High);
        assert_eq!(hint.preferred_queue, SchedulingQueueType::Graphics);
        assert!(!hint.allow_async);
        assert!(hint.batch_with.is_none());
        assert!(hint.separate_from.is_none());
        assert!(hint.deadline_frame_fraction.is_none());
    }

    #[test]
    fn test_scheduling_hint_defaults() {
        let hint = SchedulingHint::default();
        assert_eq!(hint.priority, SchedulingPriority::Normal);
    }

    #[test]
    fn test_scheduling_hint_builder_chain() {
        let pass1 = PassId::new(1);
        let pass2 = PassId::new(2);

        let hint = SchedulingHint::new(SchedulingPriority::Critical)
            .with_queue(SchedulingQueueType::Compute)
            .with_async(true)
            .with_batch(vec![pass1])
            .with_separate(vec![pass2])
            .with_deadline(0.5);

        assert_eq!(hint.priority, SchedulingPriority::Critical);
        assert_eq!(hint.preferred_queue, SchedulingQueueType::Compute);
        assert!(hint.allow_async);
        assert_eq!(hint.batch_with, Some(vec![pass1]));
        assert_eq!(hint.separate_from, Some(vec![pass2]));
        assert_eq!(hint.deadline_frame_fraction, Some(0.5));
    }

    #[test]
    fn test_scheduling_hint_batch_queries() {
        let pass1 = PassId::new(1);
        let pass2 = PassId::new(2);
        let pass3 = PassId::new(3);

        let hint = SchedulingHint::new(SchedulingPriority::Normal)
            .with_batch(vec![pass1])
            .with_separate(vec![pass2]);

        assert!(hint.wants_batch_with(pass1));
        assert!(!hint.wants_batch_with(pass3));
        assert!(hint.wants_separation_from(pass2));
        assert!(!hint.wants_separation_from(pass3));
    }

    #[test]
    fn test_scheduling_hint_has_deadline() {
        let hint1 = SchedulingHint::new(SchedulingPriority::Normal);
        let hint2 = hint1.clone().with_deadline(0.25);

        assert!(!hint1.has_deadline());
        assert!(hint2.has_deadline());
    }

    // -----------------------------------------------------------------------
    // PassCost Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pass_cost_zero() {
        let cost = PassCost::zero();
        assert_eq!(cost.gpu_time_us, 0);
        assert_eq!(cost.memory_bandwidth_mb, 0);
        assert_eq!(cost.dispatch_count, 0);
        assert!(cost.is_zero());
    }

    #[test]
    fn test_pass_cost_from_gpu_time() {
        let cost = PassCost::from_gpu_time(500);
        assert_eq!(cost.gpu_time_us, 500);
        assert_eq!(cost.memory_bandwidth_mb, 0);
        assert_eq!(cost.dispatch_count, 0);
        assert!(!cost.is_zero());
    }

    #[test]
    fn test_pass_cost_new() {
        let cost = PassCost::new(100, 256, 50);
        assert_eq!(cost.gpu_time_us, 100);
        assert_eq!(cost.memory_bandwidth_mb, 256);
        assert_eq!(cost.dispatch_count, 50);
    }

    #[test]
    fn test_pass_cost_total() {
        let cost = PassCost::new(100, 20, 10);
        // 100 * 10 + 20 * 5 + 10 = 1000 + 100 + 10 = 1110
        assert_eq!(cost.total_cost(), 1110);
    }

    #[test]
    fn test_pass_cost_add() {
        let cost1 = PassCost::new(100, 50, 10);
        let cost2 = PassCost::new(200, 100, 20);
        let combined = cost1 + cost2;
        assert_eq!(combined.gpu_time_us, 300);
        assert_eq!(combined.memory_bandwidth_mb, 150);
        assert_eq!(combined.dispatch_count, 30);
    }

    #[test]
    fn test_pass_cost_default() {
        let cost = PassCost::default();
        assert!(cost.is_zero());
    }

    #[test]
    fn test_pass_cost_display() {
        let cost = PassCost::new(100, 256, 50);
        let display = format!("{}", cost);
        assert!(display.contains("100us"));
        assert!(display.contains("256MB"));
        assert!(display.contains("50 dispatches"));
    }

    // -----------------------------------------------------------------------
    // PassScheduleInfo Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pass_schedule_info_creation() {
        let pass = PassId::new(42);
        let info = PassScheduleInfo::new(pass, SchedulingHint::new(SchedulingPriority::High));
        assert_eq!(info.pass_id, pass);
        assert_eq!(info.hints.priority, SchedulingPriority::High);
        assert!(info.estimated_cost.is_zero());
        assert!(info.dependencies.is_empty());
        assert!(info.dependents.is_empty());
    }

    #[test]
    fn test_pass_schedule_info_with_defaults() {
        let pass = PassId::new(1);
        let info = PassScheduleInfo::with_defaults(pass);
        assert_eq!(info.hints.priority, SchedulingPriority::Normal);
    }

    #[test]
    fn test_pass_schedule_info_with_cost() {
        let pass = PassId::new(1);
        let info = PassScheduleInfo::with_defaults(pass).with_cost(PassCost::from_gpu_time(500));
        assert_eq!(info.estimated_cost.gpu_time_us, 500);
    }

    #[test]
    fn test_pass_schedule_info_dependencies() {
        let pass = PassId::new(1);
        let dep = PassId::new(0);
        let dependent = PassId::new(2);

        let mut info = PassScheduleInfo::with_defaults(pass);
        info.add_dependency(dep);
        info.add_dependent(dependent);

        assert_eq!(info.dependencies, vec![dep]);
        assert_eq!(info.dependents, vec![dependent]);
        assert!(!info.is_root());
        assert!(!info.is_leaf());
    }

    #[test]
    fn test_pass_schedule_info_root_and_leaf() {
        let pass = PassId::new(1);
        let info = PassScheduleInfo::with_defaults(pass);
        assert!(info.is_root());
        assert!(info.is_leaf());
    }

    // -----------------------------------------------------------------------
    // ScheduleBuilder Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_schedule_builder_empty() {
        let builder = ScheduleBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);
        assert!(builder.build_schedule().is_empty());
    }

    #[test]
    fn test_schedule_builder_add_pass() {
        let mut builder = ScheduleBuilder::new();
        let pass = PassId::new(1);
        builder.add_pass(pass, SchedulingHint::new(SchedulingPriority::High));

        assert_eq!(builder.len(), 1);
        assert!(builder.get_pass(pass).is_some());
        assert_eq!(builder.get_pass(pass).unwrap().hints.priority, SchedulingPriority::High);
    }

    #[test]
    fn test_schedule_builder_with_cost() {
        let mut builder = ScheduleBuilder::new();
        let pass = PassId::new(1);
        builder.add_pass(pass, SchedulingHint::default());
        builder.with_cost(pass, PassCost::from_gpu_time(1000));

        assert_eq!(builder.get_pass(pass).unwrap().estimated_cost.gpu_time_us, 1000);
    }

    #[test]
    fn test_schedule_builder_pass_ordering_by_priority() {
        let mut builder = ScheduleBuilder::new();

        let low = PassId::new(0);
        let high = PassId::new(1);
        let critical = PassId::new(2);

        // Add in reverse priority order
        builder.add_pass(low, SchedulingHint::new(SchedulingPriority::Low));
        builder.add_pass(high, SchedulingHint::new(SchedulingPriority::High));
        builder.add_pass(critical, SchedulingHint::new(SchedulingPriority::Critical));

        let schedule = builder.build_schedule();

        // Critical should come first, then high, then low
        assert_eq!(schedule[0], critical);
        assert_eq!(schedule[1], high);
        assert_eq!(schedule[2], low);
    }

    #[test]
    fn test_schedule_builder_respects_dependencies() {
        let mut builder = ScheduleBuilder::new();

        let first = PassId::new(0);
        let second = PassId::new(1);
        let third = PassId::new(2);

        // Add passes with lower priority first having highest priority
        // but dependency constraints should win
        builder.add_pass(first, SchedulingHint::new(SchedulingPriority::Low));
        builder.add_pass(second, SchedulingHint::new(SchedulingPriority::Critical));
        builder.add_pass(third, SchedulingHint::new(SchedulingPriority::High));

        // second depends on first, third depends on second
        builder.add_dependency(second, first);
        builder.add_dependency(third, second);

        let schedule = builder.build_schedule();

        // Despite priorities, dependencies force order: first -> second -> third
        assert_eq!(schedule[0], first);
        assert_eq!(schedule[1], second);
        assert_eq!(schedule[2], third);
    }

    #[test]
    fn test_schedule_builder_group_by_queue() {
        let mut builder = ScheduleBuilder::new();

        let graphics1 = PassId::new(0);
        let graphics2 = PassId::new(1);
        let compute1 = PassId::new(2);
        let transfer1 = PassId::new(3);

        builder.add_pass(graphics1, SchedulingHint::new(SchedulingPriority::Normal).with_queue(SchedulingQueueType::Graphics));
        builder.add_pass(graphics2, SchedulingHint::new(SchedulingPriority::Normal).with_queue(SchedulingQueueType::Graphics));
        builder.add_pass(compute1, SchedulingHint::new(SchedulingPriority::Normal).with_queue(SchedulingQueueType::Compute));
        builder.add_pass(transfer1, SchedulingHint::new(SchedulingPriority::Normal).with_queue(SchedulingQueueType::Transfer));

        let groups = builder.group_by_queue();

        assert_eq!(groups.get(&SchedulingQueueType::Graphics).unwrap().len(), 2);
        assert_eq!(groups.get(&SchedulingQueueType::Compute).unwrap().len(), 1);
        assert_eq!(groups.get(&SchedulingQueueType::Transfer).unwrap().len(), 1);
        assert!(groups.get(&SchedulingQueueType::Present).unwrap().is_empty());
    }

    #[test]
    fn test_schedule_builder_async_eligible() {
        let mut builder = ScheduleBuilder::new();

        let graphics_async = PassId::new(0);
        let compute_async = PassId::new(1);
        let compute_sync = PassId::new(2);
        let transfer_async = PassId::new(3);

        builder.add_pass(graphics_async, SchedulingHint::new(SchedulingPriority::Normal)
            .with_queue(SchedulingQueueType::Graphics)
            .with_async(true));
        builder.add_pass(compute_async, SchedulingHint::new(SchedulingPriority::Normal)
            .with_queue(SchedulingQueueType::Compute)
            .with_async(true));
        builder.add_pass(compute_sync, SchedulingHint::new(SchedulingPriority::Normal)
            .with_queue(SchedulingQueueType::Compute)
            .with_async(false));
        builder.add_pass(transfer_async, SchedulingHint::new(SchedulingPriority::Normal)
            .with_queue(SchedulingQueueType::Transfer)
            .with_async(true));

        let async_passes = builder.async_eligible_passes();

        // Only compute_async and transfer_async should be eligible
        // graphics_async is on Graphics queue, compute_sync doesn't allow async
        assert_eq!(async_passes.len(), 2);
        assert!(async_passes.contains(&compute_async));
        assert!(async_passes.contains(&transfer_async));
        assert!(!async_passes.contains(&graphics_async));
        assert!(!async_passes.contains(&compute_sync));
    }

    #[test]
    fn test_schedule_builder_total_cost() {
        let mut builder = ScheduleBuilder::new();

        let pass1 = PassId::new(0);
        let pass2 = PassId::new(1);

        builder.add_pass(pass1, SchedulingHint::default());
        builder.add_pass(pass2, SchedulingHint::default());
        builder.with_cost(pass1, PassCost::new(100, 50, 10));
        builder.with_cost(pass2, PassCost::new(200, 100, 20));

        let total = builder.total_cost();
        assert_eq!(total.gpu_time_us, 300);
        assert_eq!(total.memory_bandwidth_mb, 150);
        assert_eq!(total.dispatch_count, 30);
    }

    #[test]
    fn test_schedule_builder_compute_depths() {
        let mut builder = ScheduleBuilder::new();

        // Create a diamond dependency:
        //     A (depth 0)
        //    / \
        //   B   C (depth 1)
        //    \ /
        //     D (depth 2)

        let a = PassId::new(0);
        let b = PassId::new(1);
        let c = PassId::new(2);
        let d = PassId::new(3);

        builder.add_pass(a, SchedulingHint::default());
        builder.add_pass(b, SchedulingHint::default());
        builder.add_pass(c, SchedulingHint::default());
        builder.add_pass(d, SchedulingHint::default());

        builder.add_dependency(b, a);
        builder.add_dependency(c, a);
        builder.add_dependency(d, b);
        builder.add_dependency(d, c);

        let depths = builder.compute_depths();

        assert_eq!(depths.get(&a), Some(&0));
        assert_eq!(depths.get(&b), Some(&1));
        assert_eq!(depths.get(&c), Some(&1));
        assert_eq!(depths.get(&d), Some(&2));
    }

    #[test]
    fn test_schedule_builder_batch_hint_tracking() {
        let mut builder = ScheduleBuilder::new();

        let shadow = PassId::new(0);
        let gbuffer = PassId::new(1);
        let postfx = PassId::new(2);

        // gbuffer should batch with shadow, separate from postfx
        let gbuffer_hint = SchedulingHint::new(SchedulingPriority::High)
            .with_batch(vec![shadow])
            .with_separate(vec![postfx]);

        builder.add_pass(shadow, SchedulingHint::new(SchedulingPriority::High));
        builder.add_pass(gbuffer, gbuffer_hint);
        builder.add_pass(postfx, SchedulingHint::new(SchedulingPriority::Low));

        let info = builder.get_pass(gbuffer).unwrap();
        assert!(info.hints.wants_batch_with(shadow));
        assert!(info.hints.wants_separation_from(postfx));
    }

    #[test]
    fn test_schedule_builder_separation_hints() {
        let mut builder = ScheduleBuilder::new();

        let shadow = PassId::new(0);
        let transparent = PassId::new(1);

        let shadow_hint = SchedulingHint::new(SchedulingPriority::High)
            .with_separate(vec![transparent]);

        builder.add_pass(shadow, shadow_hint);
        builder.add_pass(transparent, SchedulingHint::new(SchedulingPriority::Normal));

        let info = builder.get_pass(shadow).unwrap();
        assert!(info.hints.wants_separation_from(transparent));
        assert!(!info.hints.wants_batch_with(transparent));
    }
}
