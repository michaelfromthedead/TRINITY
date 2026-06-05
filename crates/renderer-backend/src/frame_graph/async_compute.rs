//! Async Compute Overlap for Frame Graph Parallel Queue Execution (T-WGPU-P7.5.12)
//!
//! This module provides types and analysis for enabling parallel GPU queue execution
//! between graphics and compute workloads. It identifies opportunities for async
//! compute overlap and generates the necessary synchronization points.
//!
//! # Architecture
//!
//! Async compute allows compute-only passes to execute in parallel with graphics
//! passes on a separate hardware queue. This requires:
//! 1. **Hint System**: Passes declare their async compute preferences
//! 2. **Dependency Analysis**: Identify which passes can safely run in parallel
//! 3. **Overlap Detection**: Find regions where graphics and compute can overlap
//! 4. **Sync Point Generation**: Create cross-queue synchronization primitives
//!
//! # Usage
//!
//! ```rust,ignore
//! use renderer_backend::frame_graph::async_compute::*;
//! use renderer_backend::frame_graph::graph::{FrameGraph, PassId};
//!
//! let mut analyzer = AsyncComputeAnalyzer::new();
//!
//! // Set hints for compute passes
//! analyzer.set_hint(particle_pass, AsyncComputeHint::Preferred);
//! analyzer.set_hint(culling_pass, AsyncComputeHint::Required);
//! analyzer.set_hint(shadow_pass, AsyncComputeHint::Disabled);
//!
//! // Analyze the frame graph
//! let overlap_info = analyzer.analyze(&frame_graph);
//!
//! // Get sync points for cross-queue dependencies
//! let sync_points = analyzer.compute_sync_points(&overlap_info);
//! ```
//!
//! # Queue Types
//!
//! | Queue    | Supported Operations          | Async Overlap |
//! |----------|-------------------------------|---------------|
//! | Graphics | Render, Compute, Transfer     | Primary       |
//! | Compute  | Compute, Transfer             | Secondary     |
//! | Transfer | Transfer only                 | Tertiary      |
//!
//! # Timeline Semaphores
//!
//! Cross-queue synchronization uses timeline semaphores (Vulkan) or equivalent
//! primitives. Each queue has a monotonically increasing timeline value that
//! passes signal and wait on.

use std::collections::HashMap;
use std::fmt;

use super::graph::{FrameGraph, PassId, PassType, ResourceId};
use super::scheduling::SchedulingQueueType;

// ---------------------------------------------------------------------------
// QueueType (for async compute context)
// ---------------------------------------------------------------------------

/// GPU queue type for async compute scheduling.
///
/// This enum represents the hardware queue families available for parallel
/// execution. Not all GPUs support all queue types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum QueueType {
    /// Main graphics queue. Supports all operations.
    #[default]
    Graphics,
    /// Async compute queue. Supports compute and transfer operations.
    Compute,
    /// Transfer/DMA queue. Supports only memory transfer operations.
    Transfer,
}

impl QueueType {
    /// Returns true if this is the graphics queue.
    #[inline]
    pub const fn is_graphics(self) -> bool {
        matches!(self, Self::Graphics)
    }

    /// Returns true if this is the compute queue.
    #[inline]
    pub const fn is_compute(self) -> bool {
        matches!(self, Self::Compute)
    }

    /// Returns true if this is the transfer queue.
    #[inline]
    pub const fn is_transfer(self) -> bool {
        matches!(self, Self::Transfer)
    }

    /// Returns true if this queue supports compute operations.
    #[inline]
    pub const fn supports_compute(self) -> bool {
        matches!(self, Self::Graphics | Self::Compute)
    }

    /// Returns true if this queue supports graphics operations.
    #[inline]
    pub const fn supports_graphics(self) -> bool {
        matches!(self, Self::Graphics)
    }

    /// Returns true if this queue supports transfer operations.
    #[inline]
    pub const fn supports_transfer(self) -> bool {
        // All queues support transfer
        true
    }

    /// Converts from SchedulingQueueType.
    pub fn from_scheduling_queue(queue: SchedulingQueueType) -> Self {
        match queue {
            SchedulingQueueType::Graphics | SchedulingQueueType::Present => Self::Graphics,
            SchedulingQueueType::Compute => Self::Compute,
            SchedulingQueueType::Transfer => Self::Transfer,
        }
    }
}

impl fmt::Display for QueueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graphics => write!(f, "Graphics"),
            Self::Compute => write!(f, "Compute"),
            Self::Transfer => write!(f, "Transfer"),
        }
    }
}

// ---------------------------------------------------------------------------
// AsyncComputeHint
// ---------------------------------------------------------------------------

/// Hint for async compute execution preference.
///
/// Passes use this hint to indicate their preference for async compute
/// execution. The scheduler considers these hints along with dependency
/// analysis to make the final queue assignment.
///
/// # Hint Levels
///
/// | Hint          | Behavior                                          |
/// |---------------|---------------------------------------------------|
/// | Disabled      | Always run on graphics queue                      |
/// | Preferred     | Prefer async compute if dependencies allow        |
/// | Required      | Must run on async compute (error if blocked)      |
/// | Opportunistic | Run async if no explicit dependencies block it    |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AsyncComputeHint {
    /// The pass must run on the graphics queue. No async compute.
    Disabled,
    /// Prefer async compute if available and dependencies allow.
    /// Falls back to graphics queue if async is not possible.
    Preferred,
    /// The pass must run on async compute. Scheduling will fail if
    /// this cannot be satisfied.
    Required,
    /// Run on async compute if there are no blocking dependencies.
    /// More aggressive than Preferred but less strict than Required.
    #[default]
    Opportunistic,
}

impl AsyncComputeHint {
    /// Returns true if async compute is allowed.
    #[inline]
    pub const fn allows_async(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// Returns true if async compute is required.
    #[inline]
    pub const fn requires_async(self) -> bool {
        matches!(self, Self::Required)
    }

    /// Returns true if this is opportunistic mode.
    #[inline]
    pub const fn is_opportunistic(self) -> bool {
        matches!(self, Self::Opportunistic)
    }

    /// Returns true if async compute is preferred (but not required).
    #[inline]
    pub const fn prefers_async(self) -> bool {
        matches!(self, Self::Preferred)
    }

    /// Returns true if the pass should be disabled from async compute.
    #[inline]
    pub const fn is_disabled(self) -> bool {
        matches!(self, Self::Disabled)
    }
}

impl fmt::Display for AsyncComputeHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "Disabled"),
            Self::Preferred => write!(f, "Preferred"),
            Self::Required => write!(f, "Required"),
            Self::Opportunistic => write!(f, "Opportunistic"),
        }
    }
}

// ---------------------------------------------------------------------------
// QueueSyncPoint
// ---------------------------------------------------------------------------

/// A synchronization point on a specific GPU queue.
///
/// Represents a point in the queue's timeline that can be signaled by one
/// pass and waited on by another. Uses timeline semaphore semantics where
/// each sync point has a monotonically increasing value.
///
/// # Timeline Values
///
/// Timeline values increase monotonically per queue. A pass signals its
/// completion by incrementing the timeline. Other passes can wait for
/// a specific value to ensure ordering.
///
/// ```text
/// Queue Timeline:
///   Pass A signals value 10
///   Pass B waits for value 10
///   Pass B signals value 11
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QueueSyncPoint {
    /// The queue this sync point belongs to.
    pub queue: QueueType,
    /// The timeline semaphore value.
    pub value: u64,
    /// The pass that signals or waits on this sync point.
    pub pass_id: PassId,
}

impl QueueSyncPoint {
    /// Creates a new queue sync point.
    pub fn new(queue: QueueType, value: u64, pass_id: PassId) -> Self {
        Self {
            queue,
            value,
            pass_id,
        }
    }

    /// Creates a sync point for the graphics queue.
    pub fn graphics(value: u64, pass_id: PassId) -> Self {
        Self::new(QueueType::Graphics, value, pass_id)
    }

    /// Creates a sync point for the compute queue.
    pub fn compute(value: u64, pass_id: PassId) -> Self {
        Self::new(QueueType::Compute, value, pass_id)
    }

    /// Creates a sync point for the transfer queue.
    pub fn transfer(value: u64, pass_id: PassId) -> Self {
        Self::new(QueueType::Transfer, value, pass_id)
    }

    /// Returns true if this sync point is on the graphics queue.
    #[inline]
    pub fn is_graphics(&self) -> bool {
        self.queue.is_graphics()
    }

    /// Returns true if this sync point is on the compute queue.
    #[inline]
    pub fn is_compute(&self) -> bool {
        self.queue.is_compute()
    }

    /// Returns true if this sync point is on the transfer queue.
    #[inline]
    pub fn is_transfer(&self) -> bool {
        self.queue.is_transfer()
    }
}

impl fmt::Display for QueueSyncPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SyncPoint({} @{} for {})",
            self.queue, self.value, self.pass_id
        )
    }
}

// ---------------------------------------------------------------------------
// CrossQueueDependency
// ---------------------------------------------------------------------------

/// A dependency between passes on different GPU queues.
///
/// Represents the need for synchronization when a pass on one queue depends
/// on resources produced by a pass on another queue. The destination pass
/// must wait for the source pass to complete before starting.
///
/// # Resource Tracking
///
/// The dependency tracks which resources require synchronization. This enables
/// the barrier resolver to insert appropriate memory barriers along with the
/// semaphore wait.
#[derive(Clone, Debug)]
pub struct CrossQueueDependency {
    /// The pass that produces the resource (signals).
    pub source: QueueSyncPoint,
    /// The pass that consumes the resource (waits).
    pub dest: QueueSyncPoint,
    /// Resources that require cross-queue synchronization.
    pub resources: Vec<ResourceId>,
}

impl CrossQueueDependency {
    /// Creates a new cross-queue dependency.
    pub fn new(
        source: QueueSyncPoint,
        dest: QueueSyncPoint,
        resources: Vec<ResourceId>,
    ) -> Self {
        Self {
            source,
            dest,
            resources,
        }
    }

    /// Creates a dependency with a single resource.
    pub fn with_resource(source: QueueSyncPoint, dest: QueueSyncPoint, resource: ResourceId) -> Self {
        Self::new(source, dest, vec![resource])
    }

    /// Creates a dependency with no resources (pure execution barrier).
    pub fn execution_only(source: QueueSyncPoint, dest: QueueSyncPoint) -> Self {
        Self::new(source, dest, Vec::new())
    }

    /// Returns the source queue type.
    #[inline]
    pub fn source_queue(&self) -> QueueType {
        self.source.queue
    }

    /// Returns the destination queue type.
    #[inline]
    pub fn dest_queue(&self) -> QueueType {
        self.dest.queue
    }

    /// Returns true if this is a graphics-to-compute dependency.
    #[inline]
    pub fn is_graphics_to_compute(&self) -> bool {
        self.source.queue.is_graphics() && self.dest.queue.is_compute()
    }

    /// Returns true if this is a compute-to-graphics dependency.
    #[inline]
    pub fn is_compute_to_graphics(&self) -> bool {
        self.source.queue.is_compute() && self.dest.queue.is_graphics()
    }

    /// Returns the number of resources in this dependency.
    #[inline]
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Returns true if this dependency has resources.
    #[inline]
    pub fn has_resources(&self) -> bool {
        !self.resources.is_empty()
    }

    /// Adds a resource to this dependency.
    pub fn add_resource(&mut self, resource: ResourceId) {
        if !self.resources.contains(&resource) {
            self.resources.push(resource);
        }
    }
}

impl fmt::Display for CrossQueueDependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CrossQueueDep({} -> {}, {} resources)",
            self.source, self.dest, self.resources.len()
        )
    }
}

// ---------------------------------------------------------------------------
// OverlapRegion
// ---------------------------------------------------------------------------

/// A region where graphics and compute passes can execute in parallel.
///
/// Identifies a contiguous range of passes on each queue that can overlap.
/// The estimated overlap time helps prioritize which overlaps are most
/// beneficial for performance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlapRegion {
    /// Index range of graphics passes in this overlap (start, end exclusive).
    pub graphics_range: (usize, usize),
    /// Index range of compute passes in this overlap (start, end exclusive).
    pub compute_range: (usize, usize),
    /// Estimated time saved by overlapping (in microseconds).
    pub estimated_overlap_time_us: u32,
}

impl OverlapRegion {
    /// Creates a new overlap region.
    pub fn new(
        graphics_range: (usize, usize),
        compute_range: (usize, usize),
        estimated_overlap_time_us: u32,
    ) -> Self {
        Self {
            graphics_range,
            compute_range,
            estimated_overlap_time_us,
        }
    }

    /// Returns the number of graphics passes in this region.
    #[inline]
    pub fn graphics_pass_count(&self) -> usize {
        self.graphics_range.1.saturating_sub(self.graphics_range.0)
    }

    /// Returns the number of compute passes in this region.
    #[inline]
    pub fn compute_pass_count(&self) -> usize {
        self.compute_range.1.saturating_sub(self.compute_range.0)
    }

    /// Returns the total number of passes in this region.
    #[inline]
    pub fn total_pass_count(&self) -> usize {
        self.graphics_pass_count() + self.compute_pass_count()
    }

    /// Returns true if this region is empty (no passes).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.graphics_pass_count() == 0 && self.compute_pass_count() == 0
    }

    /// Returns true if the estimated overlap time exceeds a threshold.
    #[inline]
    pub fn is_significant(&self, threshold_us: u32) -> bool {
        self.estimated_overlap_time_us >= threshold_us
    }
}

impl fmt::Display for OverlapRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OverlapRegion(graphics=[{}..{}), compute=[{}..{}), {}us)",
            self.graphics_range.0,
            self.graphics_range.1,
            self.compute_range.0,
            self.compute_range.1,
            self.estimated_overlap_time_us
        )
    }
}

// ---------------------------------------------------------------------------
// AsyncOverlapInfo
// ---------------------------------------------------------------------------

/// Complete information about async compute overlap opportunities.
///
/// Contains the analysis results for a frame graph, including which passes
/// run on which queue, identified overlap regions, and synchronization points.
#[derive(Clone, Debug, Default)]
pub struct AsyncOverlapInfo {
    /// Passes assigned to the graphics queue.
    pub graphics_passes: Vec<PassId>,
    /// Passes assigned to the async compute queue.
    pub compute_passes: Vec<PassId>,
    /// Regions where graphics and compute can overlap.
    pub overlap_regions: Vec<OverlapRegion>,
    /// Cross-queue synchronization points.
    pub sync_points: Vec<CrossQueueDependency>,
}

impl AsyncOverlapInfo {
    /// Creates an empty async overlap info.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates overlap info with pre-allocated capacity.
    pub fn with_capacity(graphics: usize, compute: usize) -> Self {
        Self {
            graphics_passes: Vec::with_capacity(graphics),
            compute_passes: Vec::with_capacity(compute),
            overlap_regions: Vec::new(),
            sync_points: Vec::new(),
        }
    }

    /// Returns the total number of passes.
    #[inline]
    pub fn total_pass_count(&self) -> usize {
        self.graphics_passes.len() + self.compute_passes.len()
    }

    /// Returns true if there are any compute passes for async execution.
    #[inline]
    pub fn has_async_compute(&self) -> bool {
        !self.compute_passes.is_empty()
    }

    /// Returns true if there are any overlap regions identified.
    #[inline]
    pub fn has_overlaps(&self) -> bool {
        !self.overlap_regions.is_empty()
    }

    /// Returns true if cross-queue synchronization is needed.
    #[inline]
    pub fn needs_sync(&self) -> bool {
        !self.sync_points.is_empty()
    }

    /// Returns the total estimated overlap time in microseconds.
    pub fn total_overlap_time_us(&self) -> u32 {
        self.overlap_regions
            .iter()
            .map(|r| r.estimated_overlap_time_us)
            .sum()
    }

    /// Adds a graphics pass.
    pub fn add_graphics_pass(&mut self, pass: PassId) {
        if !self.graphics_passes.contains(&pass) {
            self.graphics_passes.push(pass);
        }
    }

    /// Adds a compute pass.
    pub fn add_compute_pass(&mut self, pass: PassId) {
        if !self.compute_passes.contains(&pass) {
            self.compute_passes.push(pass);
        }
    }

    /// Adds an overlap region.
    pub fn add_overlap_region(&mut self, region: OverlapRegion) {
        self.overlap_regions.push(region);
    }

    /// Adds a sync point.
    pub fn add_sync_point(&mut self, sync: CrossQueueDependency) {
        self.sync_points.push(sync);
    }
}

impl fmt::Display for AsyncOverlapInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AsyncOverlapInfo(graphics={}, compute={}, overlaps={}, syncs={})",
            self.graphics_passes.len(),
            self.compute_passes.len(),
            self.overlap_regions.len(),
            self.sync_points.len()
        )
    }
}

// ---------------------------------------------------------------------------
// AsyncComputeAnalyzer
// ---------------------------------------------------------------------------

/// Analyzer for async compute overlap opportunities in a frame graph.
///
/// The analyzer examines pass dependencies, async compute hints, and resource
/// access patterns to identify which passes can safely execute in parallel
/// on separate GPU queues.
///
/// # Analysis Process
///
/// 1. **Collect Hints**: Gather async compute preferences from passes
/// 2. **Classify Passes**: Assign passes to graphics or compute queues
/// 3. **Find Overlaps**: Identify regions where queues can overlap
/// 4. **Generate Syncs**: Create synchronization points for dependencies
///
/// # Example
///
/// ```rust,ignore
/// let mut analyzer = AsyncComputeAnalyzer::new();
///
/// analyzer.set_hint(compute_culling, AsyncComputeHint::Required);
/// analyzer.set_hint(particle_sim, AsyncComputeHint::Preferred);
///
/// let info = analyzer.analyze(&frame_graph);
/// println!("Found {} overlap regions", info.overlap_regions.len());
/// ```
#[derive(Clone, Debug, Default)]
pub struct AsyncComputeAnalyzer {
    /// Async compute hints per pass.
    hints: HashMap<PassId, AsyncComputeHint>,
    /// Timeline values per queue (for sync point generation).
    timeline_values: HashMap<QueueType, u64>,
}

impl AsyncComputeAnalyzer {
    /// Creates a new async compute analyzer.
    pub fn new() -> Self {
        let mut timeline_values = HashMap::new();
        timeline_values.insert(QueueType::Graphics, 0);
        timeline_values.insert(QueueType::Compute, 0);
        timeline_values.insert(QueueType::Transfer, 0);

        Self {
            hints: HashMap::new(),
            timeline_values,
        }
    }

    /// Creates an analyzer with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut analyzer = Self::new();
        analyzer.hints.reserve(capacity);
        analyzer
    }

    /// Sets the async compute hint for a pass.
    pub fn set_hint(&mut self, pass: PassId, hint: AsyncComputeHint) {
        self.hints.insert(pass, hint);
    }

    /// Gets the async compute hint for a pass.
    ///
    /// Returns `Opportunistic` if no hint is set.
    pub fn get_hint(&self, pass: PassId) -> AsyncComputeHint {
        self.hints.get(&pass).copied().unwrap_or(AsyncComputeHint::Opportunistic)
    }

    /// Returns the number of passes with hints.
    pub fn hint_count(&self) -> usize {
        self.hints.len()
    }

    /// Clears all hints.
    pub fn clear_hints(&mut self) {
        self.hints.clear();
    }

    /// Removes the hint for a specific pass.
    pub fn remove_hint(&mut self, pass: PassId) -> Option<AsyncComputeHint> {
        self.hints.remove(&pass)
    }

    /// Returns true if a pass has a hint set.
    pub fn has_hint(&self, pass: PassId) -> bool {
        self.hints.contains_key(&pass)
    }

    /// Analyzes the frame graph for async compute opportunities.
    ///
    /// Returns complete overlap information including pass assignments,
    /// overlap regions, and synchronization points.
    pub fn analyze(&self, graph: &FrameGraph) -> AsyncOverlapInfo {
        let mut info = AsyncOverlapInfo::new();

        // Phase 1: Classify passes into queues
        for pass in graph.passes() {
            let hint = self.get_hint(pass.id);
            let can_async = self.can_run_async(pass.id, graph);

            if can_async && hint.allows_async() {
                info.add_compute_pass(pass.id);
            } else {
                info.add_graphics_pass(pass.id);
            }
        }

        // Phase 2: Find overlap opportunities
        info.overlap_regions = self.find_overlap_opportunities(graph);

        // Phase 3: Compute sync points
        info.sync_points = self.compute_sync_points(&info);

        info
    }

    /// Finds regions where graphics and compute passes can overlap.
    ///
    /// Identifies contiguous ranges of passes on each queue that can
    /// execute in parallel without violating dependencies.
    pub fn find_overlap_opportunities(&self, graph: &FrameGraph) -> Vec<OverlapRegion> {
        let mut regions = Vec::new();

        // Collect compute-eligible passes
        let compute_passes: Vec<PassId> = graph
            .passes()
            .filter(|p| {
                let hint = self.get_hint(p.id);
                p.pass_type.is_compute() && hint.allows_async()
            })
            .map(|p| p.id)
            .collect();

        // Collect graphics passes
        let graphics_passes: Vec<PassId> = graph
            .passes()
            .filter(|p| p.pass_type.is_graphics())
            .map(|p| p.id)
            .collect();

        if compute_passes.is_empty() || graphics_passes.is_empty() {
            return regions;
        }

        // Simple heuristic: create overlap regions based on execution order
        // In a real implementation, this would do proper dependency analysis
        let graphics_count = graphics_passes.len();
        let compute_count = compute_passes.len();

        // Create one overlap region covering all overlappable passes
        // Estimate 100us overlap per compute pass (placeholder)
        let estimated_time = (compute_count as u32) * 100;

        regions.push(OverlapRegion::new(
            (0, graphics_count),
            (0, compute_count),
            estimated_time,
        ));

        regions
    }

    /// Computes synchronization points for cross-queue dependencies.
    ///
    /// Analyzes which resources are shared between queues and generates
    /// the necessary semaphore signal/wait pairs.
    pub fn compute_sync_points(&self, overlap: &AsyncOverlapInfo) -> Vec<CrossQueueDependency> {
        let mut sync_points = Vec::new();

        // If there's no async compute, no sync needed
        if !overlap.has_async_compute() {
            return sync_points;
        }

        // Generate sync points at region boundaries
        // In a real implementation, this would analyze actual resource dependencies
        let mut graphics_timeline = 0u64;
        let mut compute_timeline = 0u64;

        for (i, region) in overlap.overlap_regions.iter().enumerate() {
            // Sync at start of overlap region: graphics -> compute
            if region.compute_pass_count() > 0 && i == 0 {
                if let Some(&first_compute) = overlap.compute_passes.first() {
                    if let Some(&last_graphics_before) = overlap.graphics_passes.first() {
                        graphics_timeline += 1;
                        compute_timeline += 1;

                        let source = QueueSyncPoint::graphics(graphics_timeline, last_graphics_before);
                        let dest = QueueSyncPoint::compute(compute_timeline, first_compute);
                        sync_points.push(CrossQueueDependency::execution_only(source, dest));
                    }
                }
            }

            // Sync at end of overlap region: compute -> graphics
            if region.compute_pass_count() > 0 {
                if let Some(&last_compute) = overlap.compute_passes.last() {
                    if let Some(&first_graphics_after) = overlap.graphics_passes.last() {
                        compute_timeline += 1;
                        graphics_timeline += 1;

                        let source = QueueSyncPoint::compute(compute_timeline, last_compute);
                        let dest = QueueSyncPoint::graphics(graphics_timeline, first_graphics_after);
                        sync_points.push(CrossQueueDependency::execution_only(source, dest));
                    }
                }
            }
        }

        sync_points
    }

    /// Determines if a pass can run on async compute.
    ///
    /// A pass can run async if:
    /// 1. It's a compute pass (not graphics)
    /// 2. Its hint allows async execution
    /// 3. Its dependencies don't force it to the graphics queue
    pub fn can_run_async(&self, pass: PassId, graph: &FrameGraph) -> bool {
        // Check pass exists and is compute type
        let Some(pass_node) = graph.get_pass(pass) else {
            return false;
        };

        // Only compute passes can run on async compute
        if !pass_node.pass_type.is_compute() {
            return false;
        }

        // Check hint
        let hint = self.get_hint(pass);
        if hint.is_disabled() {
            return false;
        }

        // Check dependencies: if any read resource was written by a graphics pass,
        // we may need to stay on graphics queue (simplified check)
        // In a full implementation, this would be more sophisticated
        let reads = pass_node.read_resources();
        for resource_id in reads {
            let writers = graph.find_writers(resource_id);
            for writer in writers {
                if let Some(writer_node) = graph.get_pass(writer) {
                    // If a graphics pass writes a resource we read, check if we need sync
                    if writer_node.pass_type.is_graphics() {
                        // Required hint can still run async with sync points
                        if !hint.requires_async() && !hint.prefers_async() && !hint.is_opportunistic() {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    /// Resets timeline values for a new frame.
    pub fn reset_timelines(&mut self) {
        for value in self.timeline_values.values_mut() {
            *value = 0;
        }
    }

    /// Gets the current timeline value for a queue.
    pub fn timeline_value(&self, queue: QueueType) -> u64 {
        self.timeline_values.get(&queue).copied().unwrap_or(0)
    }

    /// Advances the timeline for a queue and returns the new value.
    pub fn advance_timeline(&mut self, queue: QueueType) -> u64 {
        let value = self.timeline_values.entry(queue).or_insert(0);
        *value += 1;
        *value
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::graph::{
        FrameGraph, GraphResourceLifetime, PassType, ResourceType,
    };

    // -----------------------------------------------------------------------
    // QueueType Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_queue_type_variants() {
        assert_ne!(QueueType::Graphics, QueueType::Compute);
        assert_ne!(QueueType::Compute, QueueType::Transfer);
        assert_ne!(QueueType::Transfer, QueueType::Graphics);
    }

    #[test]
    fn test_queue_type_is_methods() {
        assert!(QueueType::Graphics.is_graphics());
        assert!(!QueueType::Graphics.is_compute());
        assert!(!QueueType::Graphics.is_transfer());

        assert!(!QueueType::Compute.is_graphics());
        assert!(QueueType::Compute.is_compute());
        assert!(!QueueType::Compute.is_transfer());

        assert!(!QueueType::Transfer.is_graphics());
        assert!(!QueueType::Transfer.is_compute());
        assert!(QueueType::Transfer.is_transfer());
    }

    #[test]
    fn test_queue_type_supports_compute() {
        assert!(QueueType::Graphics.supports_compute());
        assert!(QueueType::Compute.supports_compute());
        assert!(!QueueType::Transfer.supports_compute());
    }

    #[test]
    fn test_queue_type_supports_graphics() {
        assert!(QueueType::Graphics.supports_graphics());
        assert!(!QueueType::Compute.supports_graphics());
        assert!(!QueueType::Transfer.supports_graphics());
    }

    #[test]
    fn test_queue_type_supports_transfer() {
        assert!(QueueType::Graphics.supports_transfer());
        assert!(QueueType::Compute.supports_transfer());
        assert!(QueueType::Transfer.supports_transfer());
    }

    #[test]
    fn test_queue_type_default() {
        assert_eq!(QueueType::default(), QueueType::Graphics);
    }

    #[test]
    fn test_queue_type_display() {
        assert_eq!(format!("{}", QueueType::Graphics), "Graphics");
        assert_eq!(format!("{}", QueueType::Compute), "Compute");
        assert_eq!(format!("{}", QueueType::Transfer), "Transfer");
    }

    #[test]
    fn test_queue_type_from_scheduling_queue() {
        assert_eq!(
            QueueType::from_scheduling_queue(SchedulingQueueType::Graphics),
            QueueType::Graphics
        );
        assert_eq!(
            QueueType::from_scheduling_queue(SchedulingQueueType::Compute),
            QueueType::Compute
        );
        assert_eq!(
            QueueType::from_scheduling_queue(SchedulingQueueType::Transfer),
            QueueType::Transfer
        );
        assert_eq!(
            QueueType::from_scheduling_queue(SchedulingQueueType::Present),
            QueueType::Graphics
        );
    }

    // -----------------------------------------------------------------------
    // AsyncComputeHint Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_async_compute_hint_variants() {
        assert_ne!(AsyncComputeHint::Disabled, AsyncComputeHint::Preferred);
        assert_ne!(AsyncComputeHint::Preferred, AsyncComputeHint::Required);
        assert_ne!(AsyncComputeHint::Required, AsyncComputeHint::Opportunistic);
    }

    #[test]
    fn test_async_compute_hint_allows_async() {
        assert!(!AsyncComputeHint::Disabled.allows_async());
        assert!(AsyncComputeHint::Preferred.allows_async());
        assert!(AsyncComputeHint::Required.allows_async());
        assert!(AsyncComputeHint::Opportunistic.allows_async());
    }

    #[test]
    fn test_async_compute_hint_requires_async() {
        assert!(!AsyncComputeHint::Disabled.requires_async());
        assert!(!AsyncComputeHint::Preferred.requires_async());
        assert!(AsyncComputeHint::Required.requires_async());
        assert!(!AsyncComputeHint::Opportunistic.requires_async());
    }

    #[test]
    fn test_async_compute_hint_is_opportunistic() {
        assert!(!AsyncComputeHint::Disabled.is_opportunistic());
        assert!(!AsyncComputeHint::Preferred.is_opportunistic());
        assert!(!AsyncComputeHint::Required.is_opportunistic());
        assert!(AsyncComputeHint::Opportunistic.is_opportunistic());
    }

    #[test]
    fn test_async_compute_hint_prefers_async() {
        assert!(!AsyncComputeHint::Disabled.prefers_async());
        assert!(AsyncComputeHint::Preferred.prefers_async());
        assert!(!AsyncComputeHint::Required.prefers_async());
        assert!(!AsyncComputeHint::Opportunistic.prefers_async());
    }

    #[test]
    fn test_async_compute_hint_is_disabled() {
        assert!(AsyncComputeHint::Disabled.is_disabled());
        assert!(!AsyncComputeHint::Preferred.is_disabled());
        assert!(!AsyncComputeHint::Required.is_disabled());
        assert!(!AsyncComputeHint::Opportunistic.is_disabled());
    }

    #[test]
    fn test_async_compute_hint_default() {
        assert_eq!(AsyncComputeHint::default(), AsyncComputeHint::Opportunistic);
    }

    #[test]
    fn test_async_compute_hint_display() {
        assert_eq!(format!("{}", AsyncComputeHint::Disabled), "Disabled");
        assert_eq!(format!("{}", AsyncComputeHint::Preferred), "Preferred");
        assert_eq!(format!("{}", AsyncComputeHint::Required), "Required");
        assert_eq!(format!("{}", AsyncComputeHint::Opportunistic), "Opportunistic");
    }

    // -----------------------------------------------------------------------
    // QueueSyncPoint Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_queue_sync_point_creation() {
        let pass = PassId::new(42);
        let sync = QueueSyncPoint::new(QueueType::Graphics, 10, pass);
        assert_eq!(sync.queue, QueueType::Graphics);
        assert_eq!(sync.value, 10);
        assert_eq!(sync.pass_id, pass);
    }

    #[test]
    fn test_queue_sync_point_convenience_constructors() {
        let pass = PassId::new(1);

        let gfx = QueueSyncPoint::graphics(5, pass);
        assert!(gfx.is_graphics());
        assert_eq!(gfx.value, 5);

        let comp = QueueSyncPoint::compute(10, pass);
        assert!(comp.is_compute());
        assert_eq!(comp.value, 10);

        let xfer = QueueSyncPoint::transfer(15, pass);
        assert!(xfer.is_transfer());
        assert_eq!(xfer.value, 15);
    }

    #[test]
    fn test_queue_sync_point_is_methods() {
        let pass = PassId::new(1);

        let gfx = QueueSyncPoint::graphics(1, pass);
        assert!(gfx.is_graphics());
        assert!(!gfx.is_compute());
        assert!(!gfx.is_transfer());

        let comp = QueueSyncPoint::compute(1, pass);
        assert!(!comp.is_graphics());
        assert!(comp.is_compute());
        assert!(!comp.is_transfer());
    }

    #[test]
    fn test_queue_sync_point_display() {
        let pass = PassId::new(42);
        let sync = QueueSyncPoint::graphics(10, pass);
        let display = format!("{}", sync);
        assert!(display.contains("Graphics"));
        assert!(display.contains("10"));
    }

    #[test]
    fn test_queue_sync_point_equality() {
        let pass = PassId::new(1);
        let sync1 = QueueSyncPoint::graphics(10, pass);
        let sync2 = QueueSyncPoint::graphics(10, pass);
        let sync3 = QueueSyncPoint::graphics(20, pass);
        assert_eq!(sync1, sync2);
        assert_ne!(sync1, sync3);
    }

    // -----------------------------------------------------------------------
    // CrossQueueDependency Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_queue_dependency_creation() {
        let pass1 = PassId::new(1);
        let pass2 = PassId::new(2);
        let res = ResourceId::new(100);

        let source = QueueSyncPoint::graphics(5, pass1);
        let dest = QueueSyncPoint::compute(6, pass2);
        let dep = CrossQueueDependency::new(source.clone(), dest.clone(), vec![res]);

        assert_eq!(dep.source, source);
        assert_eq!(dep.dest, dest);
        assert_eq!(dep.resources, vec![res]);
    }

    #[test]
    fn test_cross_queue_dependency_with_resource() {
        let pass1 = PassId::new(1);
        let pass2 = PassId::new(2);
        let res = ResourceId::new(100);

        let source = QueueSyncPoint::graphics(1, pass1);
        let dest = QueueSyncPoint::compute(2, pass2);
        let dep = CrossQueueDependency::with_resource(source, dest, res);

        assert_eq!(dep.resource_count(), 1);
        assert!(dep.has_resources());
    }

    #[test]
    fn test_cross_queue_dependency_execution_only() {
        let pass1 = PassId::new(1);
        let pass2 = PassId::new(2);

        let source = QueueSyncPoint::graphics(1, pass1);
        let dest = QueueSyncPoint::compute(2, pass2);
        let dep = CrossQueueDependency::execution_only(source, dest);

        assert_eq!(dep.resource_count(), 0);
        assert!(!dep.has_resources());
    }

    #[test]
    fn test_cross_queue_dependency_queue_types() {
        let pass1 = PassId::new(1);
        let pass2 = PassId::new(2);

        let source = QueueSyncPoint::graphics(1, pass1);
        let dest = QueueSyncPoint::compute(2, pass2);
        let dep = CrossQueueDependency::execution_only(source, dest);

        assert_eq!(dep.source_queue(), QueueType::Graphics);
        assert_eq!(dep.dest_queue(), QueueType::Compute);
        assert!(dep.is_graphics_to_compute());
        assert!(!dep.is_compute_to_graphics());
    }

    #[test]
    fn test_cross_queue_dependency_add_resource() {
        let pass1 = PassId::new(1);
        let pass2 = PassId::new(2);

        let source = QueueSyncPoint::graphics(1, pass1);
        let dest = QueueSyncPoint::compute(2, pass2);
        let mut dep = CrossQueueDependency::execution_only(source, dest);

        let res1 = ResourceId::new(100);
        let res2 = ResourceId::new(200);

        dep.add_resource(res1);
        assert_eq!(dep.resource_count(), 1);

        dep.add_resource(res2);
        assert_eq!(dep.resource_count(), 2);

        // Duplicate should not be added
        dep.add_resource(res1);
        assert_eq!(dep.resource_count(), 2);
    }

    #[test]
    fn test_cross_queue_dependency_display() {
        let pass1 = PassId::new(1);
        let pass2 = PassId::new(2);

        let source = QueueSyncPoint::graphics(1, pass1);
        let dest = QueueSyncPoint::compute(2, pass2);
        let dep = CrossQueueDependency::execution_only(source, dest);

        let display = format!("{}", dep);
        assert!(display.contains("CrossQueueDep"));
        assert!(display.contains("0 resources"));
    }

    // -----------------------------------------------------------------------
    // OverlapRegion Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_overlap_region_creation() {
        let region = OverlapRegion::new((0, 5), (0, 3), 500);
        assert_eq!(region.graphics_range, (0, 5));
        assert_eq!(region.compute_range, (0, 3));
        assert_eq!(region.estimated_overlap_time_us, 500);
    }

    #[test]
    fn test_overlap_region_pass_counts() {
        let region = OverlapRegion::new((2, 7), (1, 4), 100);
        assert_eq!(region.graphics_pass_count(), 5);
        assert_eq!(region.compute_pass_count(), 3);
        assert_eq!(region.total_pass_count(), 8);
    }

    #[test]
    fn test_overlap_region_is_empty() {
        let empty = OverlapRegion::new((0, 0), (0, 0), 0);
        assert!(empty.is_empty());

        let non_empty = OverlapRegion::new((0, 1), (0, 0), 0);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_overlap_region_is_significant() {
        let region = OverlapRegion::new((0, 5), (0, 3), 500);
        assert!(region.is_significant(100));
        assert!(region.is_significant(500));
        assert!(!region.is_significant(501));
    }

    #[test]
    fn test_overlap_region_display() {
        let region = OverlapRegion::new((0, 5), (0, 3), 500);
        let display = format!("{}", region);
        assert!(display.contains("OverlapRegion"));
        assert!(display.contains("500us"));
    }

    // -----------------------------------------------------------------------
    // AsyncOverlapInfo Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_async_overlap_info_new() {
        let info = AsyncOverlapInfo::new();
        assert!(info.graphics_passes.is_empty());
        assert!(info.compute_passes.is_empty());
        assert!(info.overlap_regions.is_empty());
        assert!(info.sync_points.is_empty());
    }

    #[test]
    fn test_async_overlap_info_with_capacity() {
        let info = AsyncOverlapInfo::with_capacity(10, 5);
        assert!(info.graphics_passes.capacity() >= 10);
        assert!(info.compute_passes.capacity() >= 5);
    }

    #[test]
    fn test_async_overlap_info_add_passes() {
        let mut info = AsyncOverlapInfo::new();

        let gfx = PassId::new(1);
        let comp = PassId::new(2);

        info.add_graphics_pass(gfx);
        info.add_compute_pass(comp);

        assert_eq!(info.graphics_passes.len(), 1);
        assert_eq!(info.compute_passes.len(), 1);
        assert_eq!(info.total_pass_count(), 2);
    }

    #[test]
    fn test_async_overlap_info_no_duplicates() {
        let mut info = AsyncOverlapInfo::new();
        let pass = PassId::new(1);

        info.add_graphics_pass(pass);
        info.add_graphics_pass(pass);
        assert_eq!(info.graphics_passes.len(), 1);
    }

    #[test]
    fn test_async_overlap_info_has_async_compute() {
        let mut info = AsyncOverlapInfo::new();
        assert!(!info.has_async_compute());

        info.add_compute_pass(PassId::new(1));
        assert!(info.has_async_compute());
    }

    #[test]
    fn test_async_overlap_info_has_overlaps() {
        let mut info = AsyncOverlapInfo::new();
        assert!(!info.has_overlaps());

        info.add_overlap_region(OverlapRegion::new((0, 1), (0, 1), 100));
        assert!(info.has_overlaps());
    }

    #[test]
    fn test_async_overlap_info_needs_sync() {
        let mut info = AsyncOverlapInfo::new();
        assert!(!info.needs_sync());

        let source = QueueSyncPoint::graphics(1, PassId::new(1));
        let dest = QueueSyncPoint::compute(2, PassId::new(2));
        info.add_sync_point(CrossQueueDependency::execution_only(source, dest));
        assert!(info.needs_sync());
    }

    #[test]
    fn test_async_overlap_info_total_overlap_time() {
        let mut info = AsyncOverlapInfo::new();
        info.add_overlap_region(OverlapRegion::new((0, 1), (0, 1), 100));
        info.add_overlap_region(OverlapRegion::new((1, 2), (1, 2), 200));
        assert_eq!(info.total_overlap_time_us(), 300);
    }

    #[test]
    fn test_async_overlap_info_display() {
        let info = AsyncOverlapInfo::new();
        let display = format!("{}", info);
        assert!(display.contains("AsyncOverlapInfo"));
    }

    // -----------------------------------------------------------------------
    // AsyncComputeAnalyzer Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_async_compute_analyzer_new() {
        let analyzer = AsyncComputeAnalyzer::new();
        assert_eq!(analyzer.hint_count(), 0);
    }

    #[test]
    fn test_async_compute_analyzer_with_capacity() {
        let analyzer = AsyncComputeAnalyzer::with_capacity(100);
        assert_eq!(analyzer.hint_count(), 0);
    }

    #[test]
    fn test_async_compute_analyzer_set_get_hint() {
        let mut analyzer = AsyncComputeAnalyzer::new();
        let pass = PassId::new(42);

        // Default is Opportunistic
        assert_eq!(analyzer.get_hint(pass), AsyncComputeHint::Opportunistic);

        analyzer.set_hint(pass, AsyncComputeHint::Required);
        assert_eq!(analyzer.get_hint(pass), AsyncComputeHint::Required);
    }

    #[test]
    fn test_async_compute_analyzer_has_hint() {
        let mut analyzer = AsyncComputeAnalyzer::new();
        let pass = PassId::new(1);

        assert!(!analyzer.has_hint(pass));
        analyzer.set_hint(pass, AsyncComputeHint::Disabled);
        assert!(analyzer.has_hint(pass));
    }

    #[test]
    fn test_async_compute_analyzer_remove_hint() {
        let mut analyzer = AsyncComputeAnalyzer::new();
        let pass = PassId::new(1);

        analyzer.set_hint(pass, AsyncComputeHint::Preferred);
        let removed = analyzer.remove_hint(pass);
        assert_eq!(removed, Some(AsyncComputeHint::Preferred));
        assert!(!analyzer.has_hint(pass));
    }

    #[test]
    fn test_async_compute_analyzer_clear_hints() {
        let mut analyzer = AsyncComputeAnalyzer::new();

        analyzer.set_hint(PassId::new(1), AsyncComputeHint::Required);
        analyzer.set_hint(PassId::new(2), AsyncComputeHint::Preferred);
        assert_eq!(analyzer.hint_count(), 2);

        analyzer.clear_hints();
        assert_eq!(analyzer.hint_count(), 0);
    }

    #[test]
    fn test_async_compute_analyzer_timelines() {
        let mut analyzer = AsyncComputeAnalyzer::new();

        assert_eq!(analyzer.timeline_value(QueueType::Graphics), 0);
        assert_eq!(analyzer.timeline_value(QueueType::Compute), 0);

        let new_val = analyzer.advance_timeline(QueueType::Graphics);
        assert_eq!(new_val, 1);
        assert_eq!(analyzer.timeline_value(QueueType::Graphics), 1);

        analyzer.reset_timelines();
        assert_eq!(analyzer.timeline_value(QueueType::Graphics), 0);
    }

    #[test]
    fn test_async_compute_analyzer_can_run_async_graphics_pass() {
        let analyzer = AsyncComputeAnalyzer::new();
        let mut graph = FrameGraph::new();

        let render_pass = graph.add_pass("render", PassType::Render);

        // Graphics passes cannot run on async compute
        assert!(!analyzer.can_run_async(render_pass, &graph));
    }

    #[test]
    fn test_async_compute_analyzer_can_run_async_compute_pass() {
        let analyzer = AsyncComputeAnalyzer::new();
        let mut graph = FrameGraph::new();

        let compute_pass = graph.add_pass("compute", PassType::Compute);

        // Compute passes can run on async compute (default is Opportunistic)
        assert!(analyzer.can_run_async(compute_pass, &graph));
    }

    #[test]
    fn test_async_compute_analyzer_can_run_async_disabled() {
        let mut analyzer = AsyncComputeAnalyzer::new();
        let mut graph = FrameGraph::new();

        let compute_pass = graph.add_pass("compute", PassType::Compute);
        analyzer.set_hint(compute_pass, AsyncComputeHint::Disabled);

        // Disabled hint prevents async
        assert!(!analyzer.can_run_async(compute_pass, &graph));
    }

    #[test]
    fn test_async_compute_analyzer_analyze_empty_graph() {
        let analyzer = AsyncComputeAnalyzer::new();
        let graph = FrameGraph::new();

        let info = analyzer.analyze(&graph);
        assert!(info.graphics_passes.is_empty());
        assert!(info.compute_passes.is_empty());
        assert!(info.overlap_regions.is_empty());
    }

    #[test]
    fn test_async_compute_analyzer_analyze_graphics_only() {
        let analyzer = AsyncComputeAnalyzer::new();
        let mut graph = FrameGraph::new();

        graph.add_pass("render1", PassType::Render);
        graph.add_pass("render2", PassType::Render);

        let info = analyzer.analyze(&graph);
        assert_eq!(info.graphics_passes.len(), 2);
        assert!(info.compute_passes.is_empty());
    }

    #[test]
    fn test_async_compute_analyzer_analyze_mixed() {
        let analyzer = AsyncComputeAnalyzer::new();
        let mut graph = FrameGraph::new();

        graph.add_pass("render", PassType::Render);
        graph.add_pass("compute", PassType::Compute);

        let info = analyzer.analyze(&graph);
        assert_eq!(info.graphics_passes.len(), 1);
        assert_eq!(info.compute_passes.len(), 1);
    }

    #[test]
    fn test_async_compute_analyzer_find_overlap_opportunities() {
        let analyzer = AsyncComputeAnalyzer::new();
        let mut graph = FrameGraph::new();

        graph.add_pass("render1", PassType::Render);
        graph.add_pass("render2", PassType::Render);
        graph.add_pass("compute1", PassType::Compute);
        graph.add_pass("compute2", PassType::Compute);

        let overlaps = analyzer.find_overlap_opportunities(&graph);
        assert!(!overlaps.is_empty());
    }

    #[test]
    fn test_async_compute_analyzer_compute_sync_points() {
        let analyzer = AsyncComputeAnalyzer::new();
        let mut graph = FrameGraph::new();

        graph.add_pass("render", PassType::Render);
        graph.add_pass("compute", PassType::Compute);

        let info = analyzer.analyze(&graph);
        let sync_points = analyzer.compute_sync_points(&info);

        // Should have sync points for cross-queue dependencies
        if info.has_async_compute() && !info.graphics_passes.is_empty() {
            assert!(!sync_points.is_empty());
        }
    }
}
