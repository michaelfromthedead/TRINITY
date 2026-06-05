//! Memory aliasing support for the Frame Graph system.
//!
//! This module provides types and algorithms for enabling efficient transient
//! resource memory reuse through aliasing. Resources with non-overlapping
//! lifetimes can share the same underlying GPU memory, reducing peak memory
//! usage by 30-50% in typical rendering scenarios.
//!
//! # Overview
//!
//! The aliasing system works in three phases:
//! 1. **Lifetime Analysis**: Determine when each resource is first used and
//!    last used within the frame graph execution.
//! 2. **Alias Group Formation**: Group resources with non-overlapping lifetimes
//!    that can share memory.
//! 3. **Memory Assignment**: Assign memory offsets to alias groups, computing
//!    actual memory savings.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::frame_graph::aliasing::*;
//! use renderer_backend::frame_graph::graph::{FrameGraph, ResourceId, PassId};
//!
//! // Analyze lifetimes from the compiled frame graph
//! let mut analyzer = AliasAnalyzer::new();
//! analyzer.analyze_lifetimes(&graph);
//!
//! // Find alias groups and compute memory savings
//! let groups = analyzer.find_alias_groups();
//! let alias_info = analyzer.compute_aliasing(heap_size);
//! println!("Memory savings: {} bytes", analyzer.total_savings());
//! ```

use std::collections::HashMap;
use std::fmt;

use super::graph::{PassId, ResourceId, FrameGraph};

// ---------------------------------------------------------------------------
// AliasPolicy
// ---------------------------------------------------------------------------

/// Policy controlling how a resource can be aliased with others.
///
/// Different policies provide trade-offs between memory savings and
/// synchronization complexity:
/// - `Never`: Safest, no aliasing allowed
/// - `SamePass`: Allows aliasing only within the same pass
/// - `NonOverlapping`: Default, aliases when lifetimes don't overlap
/// - `Aggressive`: Maximum savings, may require extra barriers
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AliasPolicy {
    /// Never alias this resource. Use for resources that must maintain
    /// their contents across frame boundaries or have special requirements.
    Never,

    /// Can alias within the same pass only. Useful for intermediate
    /// computation results that are produced and consumed within a single pass.
    SamePass,

    /// Can alias if lifetimes don't overlap. This is the default and safest
    /// policy for transient resources, providing good memory savings without
    /// additional synchronization complexity.
    #[default]
    NonOverlapping,

    /// Aggressive aliasing. May alias even with partially overlapping lifetimes
    /// if proper barriers are inserted. Provides maximum memory savings but
    /// may increase barrier overhead.
    Aggressive,
}

impl AliasPolicy {
    /// Returns true if this policy allows any form of aliasing.
    #[inline]
    pub const fn allows_aliasing(&self) -> bool {
        !matches!(self, Self::Never)
    }

    /// Returns true if this policy requires strict lifetime checking.
    #[inline]
    pub const fn requires_strict_lifetimes(&self) -> bool {
        matches!(self, Self::NonOverlapping)
    }

    /// Returns true if this is the most restrictive policy.
    #[inline]
    pub const fn is_never(&self) -> bool {
        matches!(self, Self::Never)
    }

    /// Returns true if this is the most aggressive policy.
    #[inline]
    pub const fn is_aggressive(&self) -> bool {
        matches!(self, Self::Aggressive)
    }
}

impl fmt::Display for AliasPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Never => write!(f, "Never"),
            Self::SamePass => write!(f, "SamePass"),
            Self::NonOverlapping => write!(f, "NonOverlapping"),
            Self::Aggressive => write!(f, "Aggressive"),
        }
    }
}

// ---------------------------------------------------------------------------
// AliasMapping (T-FG-3.6 API compatibility)
// ---------------------------------------------------------------------------

use super::{IrPass, IrResource, PassIndex, ResourceHandle};

/// Result of alias analysis mapping logical resources to physical slots.
///
/// Each logical resource in the frame graph receives a physical slot index.
/// Resources with non-overlapping lifetimes may share the same physical slot,
/// reducing peak memory usage.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AliasMapping {
    /// Mapping from ResourceHandle to physical slot index.
    /// Vec<(ResourceHandle, slot_index)> in resource order.
    pub entries: Vec<(ResourceHandle, usize)>,
    /// Total number of unique physical slots consumed.
    pub slots_used: usize,
}

impl AliasMapping {
    /// Creates a new empty alias mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a mapping with the given entries and slot count.
    pub fn with_entries(entries: Vec<(ResourceHandle, usize)>, slots_used: usize) -> Self {
        Self { entries, slots_used }
    }

    /// Returns true if the mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of logical resources mapped.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Gets the physical slot for a resource handle.
    pub fn get_slot(&self, handle: ResourceHandle) -> Option<usize> {
        self.entries.iter()
            .find(|(h, _)| *h == handle)
            .map(|(_, slot)| *slot)
    }

    /// Returns memory savings as a ratio (0.0 to 1.0).
    /// Higher is better - 0.5 means 50% memory saved through aliasing.
    pub fn savings_ratio(&self) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let logical_count = self.entries.len();
        let physical_count = self.slots_used.max(1);
        1.0 - (physical_count as f32 / logical_count as f32)
    }
}

impl fmt::Display for AliasMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AliasMapping({} resources -> {} slots, {:.1}% savings)",
            self.entries.len(),
            self.slots_used,
            self.savings_ratio() * 100.0
        )
    }
}

/// Apply aliasing policy to determine physical slot assignments.
///
/// This function analyzes resource lifetimes based on pass execution order
/// and the aliasing policy to determine which resources can share physical
/// memory slots.
///
/// # Arguments
///
/// * `policy` - The aliasing policy controlling memory sharing behavior
/// * `order` - Topologically sorted pass execution order
/// * `passes` - All passes in the frame graph
/// * `resources` - All resources in the frame graph
///
/// # Returns
///
/// An `AliasMapping` containing the physical slot assignment for each resource.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::frame_graph::aliasing::*;
///
/// let mapping = apply_aliasing(
///     AliasPolicy::NonOverlapping,
///     &pass_order,
///     &passes,
///     &resources,
/// );
/// println!("Using {} physical slots for {} resources",
///     mapping.slots_used, mapping.len());
/// ```
pub fn apply_aliasing(
    policy: AliasPolicy,
    order: &[PassIndex],
    passes: &[IrPass],
    resources: &[IrResource],
) -> AliasMapping {
    if resources.is_empty() {
        return AliasMapping::new();
    }

    // If policy is Never, each resource gets its own slot
    if policy.is_never() {
        let entries: Vec<_> = resources.iter()
            .enumerate()
            .map(|(i, r)| (r.handle, i))
            .collect();
        let slots_used = entries.len();
        return AliasMapping::with_entries(entries, slots_used);
    }

    // Compute resource lifetimes from pass access patterns
    let mut first_use: HashMap<ResourceHandle, usize> = HashMap::new();
    let mut last_use: HashMap<ResourceHandle, usize> = HashMap::new();

    for (exec_idx, pass_idx) in order.iter().enumerate() {
        if pass_idx.0 >= passes.len() {
            continue;
        }
        let pass = &passes[pass_idx.0];

        // Track all resources accessed by this pass (reads and writes)
        for &handle in pass.access_set.reads.iter().chain(pass.access_set.writes.iter()) {
            first_use.entry(handle).or_insert(exec_idx);
            last_use.insert(handle, exec_idx);
        }
    }

    // Also ensure all resources have entries (some may not be accessed)
    for res in resources {
        first_use.entry(res.handle).or_insert(0);
        last_use.entry(res.handle).or_insert(0);
    }

    // Greedy interval coloring for slot assignment
    // Sort resources by first_use for better packing
    let mut sorted_resources: Vec<_> = resources.iter().collect();
    sorted_resources.sort_by_key(|r| first_use.get(&r.handle).copied().unwrap_or(0));

    let mut slot_end_times: Vec<usize> = Vec::new(); // When each slot becomes free
    let mut entries: Vec<(ResourceHandle, usize)> = Vec::new();

    for res in sorted_resources {
        let start = first_use.get(&res.handle).copied().unwrap_or(0);
        let end = last_use.get(&res.handle).copied().unwrap_or(0);

        // Find a slot that's free at this resource's start time
        let slot = if policy.allows_aliasing() {
            slot_end_times.iter()
                .position(|&slot_end| slot_end < start)
        } else {
            None
        };

        let assigned_slot = match slot {
            Some(s) => {
                slot_end_times[s] = end;
                s
            }
            None => {
                slot_end_times.push(end);
                slot_end_times.len() - 1
            }
        };

        entries.push((res.handle, assigned_slot));
    }

    // Re-sort entries by original resource order for determinism
    entries.sort_by_key(|(h, _)| h.0);

    AliasMapping::with_entries(entries, slot_end_times.len())
}

// ---------------------------------------------------------------------------
// AliasingLifetime
// ---------------------------------------------------------------------------

/// Tracks the lifetime of a resource within the frame graph.
///
/// A resource's lifetime is defined by the range of passes in which it is
/// used. Resources with non-overlapping lifetimes can potentially share
/// the same GPU memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AliasingLifetime {
    /// The resource identifier.
    pub resource: ResourceId,
    /// The first pass that uses this resource.
    pub first_use: PassId,
    /// The last pass that uses this resource.
    pub last_use: PassId,
    /// Number of times the resource is accessed across all passes.
    pub usage_count: u32,
}

impl AliasingLifetime {
    /// Creates a new resource lifetime.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource identifier.
    /// * `first_use` - The first pass that uses this resource.
    /// * `last_use` - The last pass that uses this resource.
    /// * `usage_count` - Number of accesses to this resource.
    pub fn new(
        resource: ResourceId,
        first_use: PassId,
        last_use: PassId,
        usage_count: u32,
    ) -> Self {
        Self {
            resource,
            first_use,
            last_use,
            usage_count,
        }
    }

    /// Creates a new resource lifetime with a single use.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource identifier.
    /// * `pass` - The pass that uses this resource.
    pub fn single_use(resource: ResourceId, pass: PassId) -> Self {
        Self {
            resource,
            first_use: pass,
            last_use: pass,
            usage_count: 1,
        }
    }

    /// Returns true if this resource's lifetime overlaps with another.
    ///
    /// Lifetimes overlap if: `A.first_use <= B.last_use && B.first_use <= A.last_use`
    ///
    /// # Arguments
    ///
    /// * `other` - The other resource lifetime to check against.
    pub fn overlaps(&self, other: &AliasingLifetime) -> bool {
        // Pass IDs are ordered by execution order
        self.first_use.raw() <= other.last_use.raw()
            && other.first_use.raw() <= self.last_use.raw()
    }

    /// Returns true if this resource is used in the given pass.
    ///
    /// # Arguments
    ///
    /// * `pass` - The pass to check.
    pub fn contains(&self, pass: PassId) -> bool {
        pass.raw() >= self.first_use.raw() && pass.raw() <= self.last_use.raw()
    }

    /// Returns the duration (number of passes) this resource is live.
    ///
    /// A resource that is only used in one pass has a duration of 1.
    pub fn duration(&self) -> u32 {
        (self.last_use.raw() - self.first_use.raw() + 1) as u32
    }

    /// Returns true if this resource is only used in a single pass.
    #[inline]
    pub fn is_single_pass(&self) -> bool {
        self.first_use == self.last_use
    }

    /// Extends the lifetime to include a new pass.
    ///
    /// # Arguments
    ///
    /// * `pass` - The pass to include in the lifetime.
    pub fn extend_to(&mut self, pass: PassId) {
        if pass.raw() < self.first_use.raw() {
            self.first_use = pass;
        }
        if pass.raw() > self.last_use.raw() {
            self.last_use = pass;
        }
        self.usage_count += 1;
    }
}

impl fmt::Display for AliasingLifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Lifetime({}: passes {}..{}, {} uses)",
            self.resource,
            self.first_use.raw(),
            self.last_use.raw(),
            self.usage_count
        )
    }
}

// ---------------------------------------------------------------------------
// AliasCandidate
// ---------------------------------------------------------------------------

/// Represents a group of resources that can potentially share memory.
///
/// An alias candidate contains resources with compatible properties and
/// non-overlapping lifetimes (depending on alias policy).
#[derive(Clone, Debug, PartialEq)]
pub struct AliasCandidate {
    /// Resources in this alias group.
    pub resources: Vec<ResourceId>,
    /// Memory offset within the aliased heap.
    pub memory_offset: u64,
    /// Size of the memory region (max size of all members).
    pub memory_size: u64,
    /// Whether the resources are compatible for aliasing.
    pub compatible: bool,
}

impl AliasCandidate {
    /// Creates a new empty alias candidate.
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            memory_offset: 0,
            memory_size: 0,
            compatible: true,
        }
    }

    /// Creates a new alias candidate with a single resource.
    ///
    /// # Arguments
    ///
    /// * `resource` - The initial resource.
    /// * `size` - The size of the resource in bytes.
    pub fn with_resource(resource: ResourceId, size: u64) -> Self {
        Self {
            resources: vec![resource],
            memory_offset: 0,
            memory_size: size,
            compatible: true,
        }
    }

    /// Attempts to add a resource to this alias candidate.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource to add.
    /// * `size` - The size of the resource in bytes.
    /// * `lifetime` - The lifetime of the resource.
    /// * `existing_lifetimes` - Map of existing resource lifetimes.
    ///
    /// # Returns
    ///
    /// True if the resource was added, false if it cannot be aliased.
    pub fn try_add(
        &mut self,
        resource: ResourceId,
        size: u64,
        lifetime: &AliasingLifetime,
        existing_lifetimes: &HashMap<ResourceId, AliasingLifetime>,
    ) -> bool {
        // Check for overlap with any existing resource
        for &existing in &self.resources {
            if let Some(existing_lifetime) = existing_lifetimes.get(&existing) {
                if lifetime.overlaps(existing_lifetime) {
                    return false;
                }
            }
        }

        // No overlap, add to group
        self.resources.push(resource);
        self.memory_size = self.memory_size.max(size);
        true
    }

    /// Returns the number of resources in this candidate.
    #[inline]
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Returns true if this candidate is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Sets the memory offset for this candidate.
    pub fn set_offset(&mut self, offset: u64) {
        self.memory_offset = offset;
    }

    /// Marks this candidate as incompatible for aliasing.
    pub fn mark_incompatible(&mut self) {
        self.compatible = false;
    }
}

impl Default for AliasCandidate {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AliasCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AliasCandidate({} resources, offset={}, size={}, compatible={})",
            self.resources.len(),
            self.memory_offset,
            self.memory_size,
            self.compatible
        )
    }
}

// ---------------------------------------------------------------------------
// MemoryAliasInfo
// ---------------------------------------------------------------------------

/// Information about a resolved memory aliasing assignment.
///
/// After the aliasing analysis is complete, each alias group is assigned
/// a concrete memory region. This struct captures that assignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryAliasInfo {
    /// Resources sharing this memory region.
    pub aliased_resources: Vec<ResourceId>,
    /// Index into the memory heap.
    pub memory_block: u64,
    /// Byte offset within the memory block.
    pub offset: u64,
    /// Size of the memory region in bytes.
    pub size: u64,
    /// Bytes saved by aliasing (compared to individual allocations).
    pub savings_bytes: u64,
}

impl MemoryAliasInfo {
    /// Creates a new memory alias info for a single resource (no aliasing).
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource.
    /// * `memory_block` - The memory block index.
    /// * `offset` - Byte offset within the block.
    /// * `size` - Size of the memory region.
    pub fn single(resource: ResourceId, memory_block: u64, offset: u64, size: u64) -> Self {
        Self {
            aliased_resources: vec![resource],
            memory_block,
            offset,
            size,
            savings_bytes: 0,
        }
    }

    /// Creates a new memory alias info for multiple aliased resources.
    ///
    /// # Arguments
    ///
    /// * `resources` - The aliased resources.
    /// * `sizes` - Individual sizes of each resource.
    /// * `memory_block` - The memory block index.
    /// * `offset` - Byte offset within the block.
    pub fn aliased(
        resources: Vec<ResourceId>,
        sizes: &[u64],
        memory_block: u64,
        offset: u64,
    ) -> Self {
        let size = sizes.iter().copied().max().unwrap_or(0);
        let total_individual = sizes.iter().sum::<u64>();
        let savings_bytes = total_individual.saturating_sub(size);

        Self {
            aliased_resources: resources,
            memory_block,
            offset,
            size,
            savings_bytes,
        }
    }

    /// Returns the number of aliased resources.
    #[inline]
    pub fn resource_count(&self) -> usize {
        self.aliased_resources.len()
    }

    /// Returns true if this represents an actual aliasing (2+ resources).
    #[inline]
    pub fn is_aliased(&self) -> bool {
        self.aliased_resources.len() > 1
    }

    /// Returns the aliasing efficiency (savings / total potential).
    ///
    /// Returns 0.0 if not aliased or no savings.
    pub fn efficiency(&self) -> f32 {
        if self.aliased_resources.len() <= 1 || self.savings_bytes == 0 {
            0.0
        } else {
            let total = self.size + self.savings_bytes;
            self.savings_bytes as f32 / total as f32
        }
    }
}

impl fmt::Display for MemoryAliasInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryAliasInfo(block={}, offset={}, size={}, {} resources, saved={})",
            self.memory_block,
            self.offset,
            self.size,
            self.aliased_resources.len(),
            self.savings_bytes
        )
    }
}

// ---------------------------------------------------------------------------
// AliasAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes resource lifetimes and computes aliasing opportunities.
///
/// The analyzer tracks resource lifetimes across passes and determines
/// which resources can share GPU memory through aliasing.
#[derive(Clone, Debug)]
pub struct AliasAnalyzer {
    /// Resource lifetime information.
    lifetimes: HashMap<ResourceId, AliasingLifetime>,
    /// Per-resource alias policies.
    policies: HashMap<ResourceId, AliasPolicy>,
    /// Cached alias groups (populated after find_alias_groups).
    alias_groups: Vec<AliasCandidate>,
    /// Cached memory info (populated after compute_aliasing).
    memory_info: Vec<MemoryAliasInfo>,
    /// Total memory savings from aliasing.
    total_savings: u64,
}

impl AliasAnalyzer {
    /// Creates a new alias analyzer.
    pub fn new() -> Self {
        Self {
            lifetimes: HashMap::new(),
            policies: HashMap::new(),
            alias_groups: Vec::new(),
            memory_info: Vec::new(),
            total_savings: 0,
        }
    }

    /// Analyzes resource lifetimes from a compiled frame graph.
    ///
    /// This method examines each pass's resource accesses and builds
    /// lifetime information for all resources used in the graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - The compiled frame graph.
    pub fn analyze_lifetimes(&mut self, graph: &FrameGraph) {
        self.lifetimes.clear();

        // Get execution order for proper lifetime tracking
        let execution_order = graph.execution_order();

        // Build a map from pass order to PassId for quick lookup
        let pass_order: HashMap<PassId, usize> = execution_order
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // Analyze each pass's resource accesses
        for (order_idx, &pass_id) in execution_order.iter().enumerate() {
            if let Some(pass) = graph.get_pass(pass_id) {
                // Create a virtual PassId based on execution order for lifetime tracking
                let ordered_pass_id = PassId::new(order_idx as u64);

                // Track inputs (reads)
                for usage in &pass.inputs {
                    self.record_access(usage.resource, ordered_pass_id);
                }

                // Track outputs (writes)
                for usage in &pass.outputs {
                    self.record_access(usage.resource, ordered_pass_id);
                }
            }
        }
    }

    /// Records a resource access at a given pass.
    fn record_access(&mut self, resource: ResourceId, pass: PassId) {
        self.lifetimes
            .entry(resource)
            .and_modify(|lt| lt.extend_to(pass))
            .or_insert_with(|| AliasingLifetime::single_use(resource, pass));
    }

    /// Sets the alias policy for a specific resource.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource identifier.
    /// * `policy` - The alias policy to apply.
    pub fn set_policy(&mut self, resource: ResourceId, policy: AliasPolicy) {
        self.policies.insert(resource, policy);
    }

    /// Gets the alias policy for a resource.
    ///
    /// Returns `AliasPolicy::NonOverlapping` if no policy is explicitly set.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource identifier.
    pub fn get_policy(&self, resource: ResourceId) -> AliasPolicy {
        self.policies.get(&resource).copied().unwrap_or_default()
    }

    /// Gets the lifetime for a specific resource.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource identifier.
    ///
    /// # Returns
    ///
    /// The resource lifetime, or None if not tracked.
    pub fn get_lifetime(&self, resource: ResourceId) -> Option<&AliasingLifetime> {
        self.lifetimes.get(&resource)
    }

    /// Checks if two resources can be aliased based on their lifetimes and policies.
    ///
    /// # Arguments
    ///
    /// * `a` - First resource identifier.
    /// * `b` - Second resource identifier.
    ///
    /// # Returns
    ///
    /// True if the resources can share memory.
    pub fn can_alias(&self, a: ResourceId, b: ResourceId) -> bool {
        // Check policies
        let policy_a = self.get_policy(a);
        let policy_b = self.get_policy(b);

        // If either forbids aliasing, return false
        if !policy_a.allows_aliasing() || !policy_b.allows_aliasing() {
            return false;
        }

        // Get lifetimes
        let lifetime_a = match self.lifetimes.get(&a) {
            Some(lt) => lt,
            None => return false,
        };
        let lifetime_b = match self.lifetimes.get(&b) {
            Some(lt) => lt,
            None => return false,
        };

        // Check lifetime overlap based on most restrictive policy
        match (policy_a, policy_b) {
            // SamePass requires both to be single-pass in the same pass
            (AliasPolicy::SamePass, _) | (_, AliasPolicy::SamePass) => {
                lifetime_a.is_single_pass()
                    && lifetime_b.is_single_pass()
                    && lifetime_a.first_use == lifetime_b.first_use
            }
            // NonOverlapping requires no overlap
            (AliasPolicy::NonOverlapping, _) | (_, AliasPolicy::NonOverlapping) => {
                !lifetime_a.overlaps(lifetime_b)
            }
            // Aggressive allows even with overlap (caller must handle barriers)
            (AliasPolicy::Aggressive, AliasPolicy::Aggressive) => true,
            // Never should have been caught above
            (AliasPolicy::Never, _) | (_, AliasPolicy::Never) => false,
        }
    }

    /// Finds all alias groups among the tracked resources.
    ///
    /// Uses a greedy first-fit algorithm to pack resources into groups.
    /// Resources in the same group have non-overlapping lifetimes and
    /// compatible alias policies.
    ///
    /// # Returns
    ///
    /// A vector of alias candidates, each representing a group of resources
    /// that can share memory.
    pub fn find_alias_groups(&mut self) -> Vec<AliasCandidate> {
        let mut groups: Vec<AliasCandidate> = Vec::new();

        // Sort resources by first use for better packing
        let mut resources: Vec<_> = self.lifetimes.keys().copied().collect();
        resources.sort_by_key(|r| {
            self.lifetimes
                .get(r)
                .map(|lt| lt.first_use.raw())
                .unwrap_or(u64::MAX)
        });

        for resource in resources {
            let policy = self.get_policy(resource);

            // Skip resources that don't allow aliasing
            if !policy.allows_aliasing() {
                let mut candidate = AliasCandidate::with_resource(resource, 0);
                candidate.mark_incompatible();
                groups.push(candidate);
                continue;
            }

            // Try to fit into an existing group
            let mut added = false;
            for group in &mut groups {
                if !group.compatible {
                    continue;
                }

                // Check if this resource can alias with all members
                let can_alias_all = group
                    .resources
                    .iter()
                    .all(|&existing| self.can_alias(resource, existing));

                if can_alias_all {
                    group.resources.push(resource);
                    added = true;
                    break;
                }
            }

            // Create a new group if needed
            if !added {
                groups.push(AliasCandidate::with_resource(resource, 0));
            }
        }

        self.alias_groups = groups.clone();
        groups
    }

    /// Computes memory aliasing assignments for a given heap size.
    ///
    /// Assigns memory offsets to each alias group and computes the total
    /// memory savings achieved through aliasing.
    ///
    /// # Arguments
    ///
    /// * `heap_size` - The total size of the memory heap available.
    ///
    /// # Returns
    ///
    /// A vector of memory alias info structures.
    pub fn compute_aliasing(&mut self, _heap_size: u64) -> Vec<MemoryAliasInfo> {
        if self.alias_groups.is_empty() {
            self.find_alias_groups();
        }

        let mut memory_info = Vec::new();
        let mut current_offset = 0u64;
        let mut total_savings = 0u64;

        for (block_idx, group) in self.alias_groups.iter().enumerate() {
            if group.resources.is_empty() {
                continue;
            }

            if group.resources.len() == 1 {
                // Single resource, no aliasing
                let info = MemoryAliasInfo::single(
                    group.resources[0],
                    block_idx as u64,
                    current_offset,
                    group.memory_size,
                );
                current_offset += group.memory_size;
                memory_info.push(info);
            } else {
                // Multiple resources aliasing
                // For now, use uniform size since we don't track individual sizes here
                let sizes: Vec<u64> = group.resources.iter().map(|_| group.memory_size).collect();
                let info = MemoryAliasInfo::aliased(
                    group.resources.clone(),
                    &sizes,
                    block_idx as u64,
                    current_offset,
                );
                total_savings += info.savings_bytes;
                current_offset += info.size;
                memory_info.push(info);
            }
        }

        self.total_savings = total_savings;
        self.memory_info = memory_info.clone();
        memory_info
    }

    /// Returns the total memory savings from aliasing.
    ///
    /// Must be called after `compute_aliasing()` to get valid results.
    pub fn total_savings(&self) -> u64 {
        self.total_savings
    }

    /// Returns an iterator over all tracked lifetimes.
    pub fn lifetimes(&self) -> impl Iterator<Item = (&ResourceId, &AliasingLifetime)> {
        self.lifetimes.iter()
    }

    /// Returns the number of tracked resources.
    pub fn resource_count(&self) -> usize {
        self.lifetimes.len()
    }

    /// Returns the number of alias groups.
    ///
    /// Must be called after `find_alias_groups()` to get valid results.
    pub fn group_count(&self) -> usize {
        self.alias_groups.len()
    }

    /// Clears all tracked data.
    pub fn clear(&mut self) {
        self.lifetimes.clear();
        self.policies.clear();
        self.alias_groups.clear();
        self.memory_info.clear();
        self.total_savings = 0;
    }
}

impl Default for AliasAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AliasAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AliasAnalyzer({} resources, {} groups, {} bytes saved)",
            self.lifetimes.len(),
            self.alias_groups.len(),
            self.total_savings
        )
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- AliasPolicy Tests --

    #[test]
    fn test_alias_policy_default() {
        let policy = AliasPolicy::default();
        assert_eq!(policy, AliasPolicy::NonOverlapping);
    }

    #[test]
    fn test_alias_policy_allows_aliasing() {
        assert!(!AliasPolicy::Never.allows_aliasing());
        assert!(AliasPolicy::SamePass.allows_aliasing());
        assert!(AliasPolicy::NonOverlapping.allows_aliasing());
        assert!(AliasPolicy::Aggressive.allows_aliasing());
    }

    #[test]
    fn test_alias_policy_requires_strict_lifetimes() {
        assert!(!AliasPolicy::Never.requires_strict_lifetimes());
        assert!(!AliasPolicy::SamePass.requires_strict_lifetimes());
        assert!(AliasPolicy::NonOverlapping.requires_strict_lifetimes());
        assert!(!AliasPolicy::Aggressive.requires_strict_lifetimes());
    }

    #[test]
    fn test_alias_policy_is_never() {
        assert!(AliasPolicy::Never.is_never());
        assert!(!AliasPolicy::SamePass.is_never());
        assert!(!AliasPolicy::NonOverlapping.is_never());
        assert!(!AliasPolicy::Aggressive.is_never());
    }

    #[test]
    fn test_alias_policy_is_aggressive() {
        assert!(!AliasPolicy::Never.is_aggressive());
        assert!(!AliasPolicy::SamePass.is_aggressive());
        assert!(!AliasPolicy::NonOverlapping.is_aggressive());
        assert!(AliasPolicy::Aggressive.is_aggressive());
    }

    #[test]
    fn test_alias_policy_display() {
        assert_eq!(format!("{}", AliasPolicy::Never), "Never");
        assert_eq!(format!("{}", AliasPolicy::SamePass), "SamePass");
        assert_eq!(format!("{}", AliasPolicy::NonOverlapping), "NonOverlapping");
        assert_eq!(format!("{}", AliasPolicy::Aggressive), "Aggressive");
    }

    // -- AliasingLifetime Tests --

    #[test]
    fn test_resource_lifetime_new() {
        let resource = ResourceId::new(1);
        let first = PassId::new(0);
        let last = PassId::new(5);
        let lifetime = AliasingLifetime::new(resource, first, last, 3);

        assert_eq!(lifetime.resource, resource);
        assert_eq!(lifetime.first_use, first);
        assert_eq!(lifetime.last_use, last);
        assert_eq!(lifetime.usage_count, 3);
    }

    #[test]
    fn test_resource_lifetime_single_use() {
        let resource = ResourceId::new(2);
        let pass = PassId::new(3);
        let lifetime = AliasingLifetime::single_use(resource, pass);

        assert_eq!(lifetime.first_use, pass);
        assert_eq!(lifetime.last_use, pass);
        assert_eq!(lifetime.usage_count, 1);
        assert!(lifetime.is_single_pass());
    }

    #[test]
    fn test_resource_lifetime_overlaps() {
        let lt1 = AliasingLifetime::new(ResourceId::new(1), PassId::new(0), PassId::new(5), 1);
        let lt2 = AliasingLifetime::new(ResourceId::new(2), PassId::new(3), PassId::new(8), 1);
        let lt3 = AliasingLifetime::new(ResourceId::new(3), PassId::new(6), PassId::new(10), 1);

        // lt1 [0-5] and lt2 [3-8] overlap
        assert!(lt1.overlaps(&lt2));
        assert!(lt2.overlaps(&lt1));

        // lt1 [0-5] and lt3 [6-10] don't overlap
        assert!(!lt1.overlaps(&lt3));
        assert!(!lt3.overlaps(&lt1));

        // lt2 [3-8] and lt3 [6-10] overlap
        assert!(lt2.overlaps(&lt3));
    }

    #[test]
    fn test_resource_lifetime_contains() {
        let lifetime = AliasingLifetime::new(ResourceId::new(1), PassId::new(2), PassId::new(7), 1);

        assert!(!lifetime.contains(PassId::new(1)));
        assert!(lifetime.contains(PassId::new(2)));
        assert!(lifetime.contains(PassId::new(5)));
        assert!(lifetime.contains(PassId::new(7)));
        assert!(!lifetime.contains(PassId::new(8)));
    }

    #[test]
    fn test_resource_lifetime_duration() {
        let lt1 = AliasingLifetime::new(ResourceId::new(1), PassId::new(0), PassId::new(0), 1);
        assert_eq!(lt1.duration(), 1);

        let lt2 = AliasingLifetime::new(ResourceId::new(2), PassId::new(2), PassId::new(7), 1);
        assert_eq!(lt2.duration(), 6);
    }

    #[test]
    fn test_resource_lifetime_extend_to() {
        let mut lifetime = AliasingLifetime::single_use(ResourceId::new(1), PassId::new(5));

        lifetime.extend_to(PassId::new(3));
        assert_eq!(lifetime.first_use, PassId::new(3));
        assert_eq!(lifetime.last_use, PassId::new(5));

        lifetime.extend_to(PassId::new(8));
        assert_eq!(lifetime.first_use, PassId::new(3));
        assert_eq!(lifetime.last_use, PassId::new(8));

        assert_eq!(lifetime.usage_count, 3);
    }

    #[test]
    fn test_resource_lifetime_display() {
        let lifetime = AliasingLifetime::new(ResourceId::new(42), PassId::new(1), PassId::new(5), 3);
        let display = format!("{}", lifetime);
        assert!(display.contains("42"));
        assert!(display.contains("1"));
        assert!(display.contains("5"));
        assert!(display.contains("3 uses"));
    }

    // -- AliasCandidate Tests --

    #[test]
    fn test_alias_candidate_new() {
        let candidate = AliasCandidate::new();
        assert!(candidate.is_empty());
        assert_eq!(candidate.len(), 0);
        assert!(candidate.compatible);
    }

    #[test]
    fn test_alias_candidate_with_resource() {
        let resource = ResourceId::new(5);
        let candidate = AliasCandidate::with_resource(resource, 1024);

        assert_eq!(candidate.len(), 1);
        assert_eq!(candidate.resources[0], resource);
        assert_eq!(candidate.memory_size, 1024);
        assert!(candidate.compatible);
    }

    #[test]
    fn test_alias_candidate_try_add_non_overlapping() {
        let mut lifetimes = HashMap::new();
        lifetimes.insert(
            ResourceId::new(1),
            AliasingLifetime::new(ResourceId::new(1), PassId::new(0), PassId::new(2), 1),
        );
        lifetimes.insert(
            ResourceId::new(2),
            AliasingLifetime::new(ResourceId::new(2), PassId::new(4), PassId::new(6), 1),
        );

        let mut candidate = AliasCandidate::with_resource(ResourceId::new(1), 1024);
        let lt2 = &lifetimes[&ResourceId::new(2)];

        // Should succeed - no overlap
        assert!(candidate.try_add(ResourceId::new(2), 2048, lt2, &lifetimes));
        assert_eq!(candidate.len(), 2);
        assert_eq!(candidate.memory_size, 2048); // Max of 1024 and 2048
    }

    #[test]
    fn test_alias_candidate_try_add_overlapping() {
        let mut lifetimes = HashMap::new();
        lifetimes.insert(
            ResourceId::new(1),
            AliasingLifetime::new(ResourceId::new(1), PassId::new(0), PassId::new(5), 1),
        );
        lifetimes.insert(
            ResourceId::new(2),
            AliasingLifetime::new(ResourceId::new(2), PassId::new(3), PassId::new(8), 1),
        );

        let mut candidate = AliasCandidate::with_resource(ResourceId::new(1), 1024);
        let lt2 = &lifetimes[&ResourceId::new(2)];

        // Should fail - overlap
        assert!(!candidate.try_add(ResourceId::new(2), 2048, lt2, &lifetimes));
        assert_eq!(candidate.len(), 1);
    }

    #[test]
    fn test_alias_candidate_set_offset() {
        let mut candidate = AliasCandidate::new();
        candidate.set_offset(4096);
        assert_eq!(candidate.memory_offset, 4096);
    }

    #[test]
    fn test_alias_candidate_mark_incompatible() {
        let mut candidate = AliasCandidate::new();
        assert!(candidate.compatible);
        candidate.mark_incompatible();
        assert!(!candidate.compatible);
    }

    #[test]
    fn test_alias_candidate_display() {
        let candidate = AliasCandidate::with_resource(ResourceId::new(1), 2048);
        let display = format!("{}", candidate);
        assert!(display.contains("1 resources"));
        assert!(display.contains("2048"));
    }

    // -- MemoryAliasInfo Tests --

    #[test]
    fn test_memory_alias_info_single() {
        let info = MemoryAliasInfo::single(ResourceId::new(1), 0, 0, 4096);

        assert_eq!(info.resource_count(), 1);
        assert!(!info.is_aliased());
        assert_eq!(info.savings_bytes, 0);
        assert_eq!(info.efficiency(), 0.0);
    }

    #[test]
    fn test_memory_alias_info_aliased() {
        let resources = vec![ResourceId::new(1), ResourceId::new(2), ResourceId::new(3)];
        let sizes = vec![1024, 2048, 1536];
        let info = MemoryAliasInfo::aliased(resources, &sizes, 0, 0);

        assert_eq!(info.resource_count(), 3);
        assert!(info.is_aliased());
        assert_eq!(info.size, 2048); // Max size
        // Total individual = 1024 + 2048 + 1536 = 4608
        // Savings = 4608 - 2048 = 2560
        assert_eq!(info.savings_bytes, 2560);
    }

    #[test]
    fn test_memory_alias_info_efficiency() {
        let resources = vec![ResourceId::new(1), ResourceId::new(2)];
        let sizes = vec![1000, 1000];
        let info = MemoryAliasInfo::aliased(resources, &sizes, 0, 0);

        // Savings = 2000 - 1000 = 1000
        // Efficiency = 1000 / 2000 = 0.5
        assert!((info.efficiency() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_memory_alias_info_display() {
        let info = MemoryAliasInfo::single(ResourceId::new(42), 1, 1024, 4096);
        let display = format!("{}", info);
        assert!(display.contains("block=1"));
        assert!(display.contains("offset=1024"));
        assert!(display.contains("size=4096"));
    }

    // -- AliasAnalyzer Tests --

    #[test]
    fn test_alias_analyzer_new() {
        let analyzer = AliasAnalyzer::new();
        assert_eq!(analyzer.resource_count(), 0);
        assert_eq!(analyzer.group_count(), 0);
        assert_eq!(analyzer.total_savings(), 0);
    }

    #[test]
    fn test_alias_analyzer_set_get_policy() {
        let mut analyzer = AliasAnalyzer::new();
        let resource = ResourceId::new(1);

        // Default policy
        assert_eq!(analyzer.get_policy(resource), AliasPolicy::NonOverlapping);

        // Set policy
        analyzer.set_policy(resource, AliasPolicy::Never);
        assert_eq!(analyzer.get_policy(resource), AliasPolicy::Never);
    }

    #[test]
    fn test_alias_analyzer_can_alias_never_policy() {
        let mut analyzer = AliasAnalyzer::new();
        let r1 = ResourceId::new(1);
        let r2 = ResourceId::new(2);

        // Add non-overlapping lifetimes
        analyzer.lifetimes.insert(
            r1,
            AliasingLifetime::new(r1, PassId::new(0), PassId::new(2), 1),
        );
        analyzer.lifetimes.insert(
            r2,
            AliasingLifetime::new(r2, PassId::new(4), PassId::new(6), 1),
        );

        // Without Never policy, they can alias
        assert!(analyzer.can_alias(r1, r2));

        // With Never policy on one, they cannot alias
        analyzer.set_policy(r1, AliasPolicy::Never);
        assert!(!analyzer.can_alias(r1, r2));
    }

    #[test]
    fn test_alias_analyzer_can_alias_overlapping() {
        let mut analyzer = AliasAnalyzer::new();
        let r1 = ResourceId::new(1);
        let r2 = ResourceId::new(2);

        // Add overlapping lifetimes
        analyzer.lifetimes.insert(
            r1,
            AliasingLifetime::new(r1, PassId::new(0), PassId::new(5), 1),
        );
        analyzer.lifetimes.insert(
            r2,
            AliasingLifetime::new(r2, PassId::new(3), PassId::new(8), 1),
        );

        // With NonOverlapping policy, they cannot alias
        assert!(!analyzer.can_alias(r1, r2));

        // With Aggressive policy, they can alias
        analyzer.set_policy(r1, AliasPolicy::Aggressive);
        analyzer.set_policy(r2, AliasPolicy::Aggressive);
        assert!(analyzer.can_alias(r1, r2));
    }

    #[test]
    fn test_alias_analyzer_can_alias_same_pass() {
        let mut analyzer = AliasAnalyzer::new();
        let r1 = ResourceId::new(1);
        let r2 = ResourceId::new(2);
        let r3 = ResourceId::new(3);

        // Single-pass lifetimes in the same pass
        analyzer.lifetimes.insert(
            r1,
            AliasingLifetime::single_use(r1, PassId::new(5)),
        );
        analyzer.lifetimes.insert(
            r2,
            AliasingLifetime::single_use(r2, PassId::new(5)),
        );
        // Different pass
        analyzer.lifetimes.insert(
            r3,
            AliasingLifetime::single_use(r3, PassId::new(6)),
        );

        analyzer.set_policy(r1, AliasPolicy::SamePass);
        analyzer.set_policy(r2, AliasPolicy::SamePass);
        analyzer.set_policy(r3, AliasPolicy::SamePass);

        // Same pass - can alias
        assert!(analyzer.can_alias(r1, r2));

        // Different passes - cannot alias
        assert!(!analyzer.can_alias(r1, r3));
    }

    #[test]
    fn test_alias_analyzer_find_alias_groups() {
        let mut analyzer = AliasAnalyzer::new();

        // Create non-overlapping resources
        analyzer.lifetimes.insert(
            ResourceId::new(1),
            AliasingLifetime::new(ResourceId::new(1), PassId::new(0), PassId::new(2), 1),
        );
        analyzer.lifetimes.insert(
            ResourceId::new(2),
            AliasingLifetime::new(ResourceId::new(2), PassId::new(4), PassId::new(6), 1),
        );
        analyzer.lifetimes.insert(
            ResourceId::new(3),
            AliasingLifetime::new(ResourceId::new(3), PassId::new(8), PassId::new(10), 1),
        );

        let groups = analyzer.find_alias_groups();

        // All three should be in the same group (non-overlapping)
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].resources.len(), 3);
    }

    #[test]
    fn test_alias_analyzer_find_alias_groups_with_overlap() {
        let mut analyzer = AliasAnalyzer::new();

        // Create overlapping resources
        analyzer.lifetimes.insert(
            ResourceId::new(1),
            AliasingLifetime::new(ResourceId::new(1), PassId::new(0), PassId::new(5), 1),
        );
        analyzer.lifetimes.insert(
            ResourceId::new(2),
            AliasingLifetime::new(ResourceId::new(2), PassId::new(3), PassId::new(8), 1),
        );

        let groups = analyzer.find_alias_groups();

        // Each in its own group due to overlap
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_alias_analyzer_compute_aliasing() {
        let mut analyzer = AliasAnalyzer::new();

        // Create non-overlapping resources
        analyzer.lifetimes.insert(
            ResourceId::new(1),
            AliasingLifetime::new(ResourceId::new(1), PassId::new(0), PassId::new(2), 1),
        );
        analyzer.lifetimes.insert(
            ResourceId::new(2),
            AliasingLifetime::new(ResourceId::new(2), PassId::new(4), PassId::new(6), 1),
        );

        let info = analyzer.compute_aliasing(1024 * 1024);

        // Should have grouped them together
        assert!(!info.is_empty());
        assert!(analyzer.group_count() > 0);
    }

    #[test]
    fn test_alias_analyzer_clear() {
        let mut analyzer = AliasAnalyzer::new();
        analyzer.lifetimes.insert(
            ResourceId::new(1),
            AliasingLifetime::single_use(ResourceId::new(1), PassId::new(0)),
        );
        analyzer.set_policy(ResourceId::new(1), AliasPolicy::Never);

        analyzer.clear();

        assert_eq!(analyzer.resource_count(), 0);
        assert_eq!(analyzer.group_count(), 0);
        assert_eq!(analyzer.total_savings(), 0);
    }

    #[test]
    fn test_alias_analyzer_display() {
        let analyzer = AliasAnalyzer::new();
        let display = format!("{}", analyzer);
        assert!(display.contains("AliasAnalyzer"));
        assert!(display.contains("0 resources"));
    }
}
