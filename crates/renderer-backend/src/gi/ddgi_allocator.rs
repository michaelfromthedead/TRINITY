//! Fixed-size GPU buffer allocation for DDGI probe volumes.
//!
//! This module provides pre-allocated GPU buffers sized for different quality
//! presets. The allocator creates all necessary buffers upfront to avoid
//! runtime allocations and ensure consistent memory layout.
//!
//! # Quality Presets
//!
//! | Preset | Probes        | Spacing | GPU Memory |
//! |--------|---------------|---------|------------|
//! | Low    | 16x16x4 (1K)  | 4.0m    | ~200KB     |
//! | Medium | 24x24x6 (3.4K)| 3.0m    | ~650KB     |
//! | High   | 32x32x8 (8K)  | 2.0m    | ~1.5MB     |
//! | Ultra  | 48x48x12 (27K)| 1.5m    | ~5MB       |
//!
//! # Memory Layout
//!
//! Each allocation contains:
//! - Grid uniform buffer (64 bytes): [`ProbeGridGpu`]
//! - Irradiance buffer: `probe_count * 192` bytes ([`ProbeSH`])
//! - Visibility buffer: `probe_count * 16` bytes ([`ProbeVis`])

use crate::gi::probe_grid::{ProbeGridGpu, ProbeSH, ProbeVis};
use bytemuck::Pod;
use std::fmt;

// ============================================================================
// Quality Presets
// ============================================================================

/// DDGI quality preset enumeration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DDGIQuality {
    /// Low quality: 16x16x4 probes, 4m spacing (~200KB)
    Low = 0,
    /// Medium quality: 24x24x6 probes, 3m spacing (~650KB)
    Medium = 1,
    /// High quality: 32x32x8 probes, 2m spacing (~1.5MB)
    #[default]
    High = 2,
    /// Ultra quality: 48x48x12 probes, 1.5m spacing (~5MB)
    Ultra = 3,
}

impl DDGIQuality {
    /// Get probe grid dimensions for this quality level.
    pub const fn dimensions(self) -> [u32; 3] {
        match self {
            DDGIQuality::Low => [16, 16, 4],
            DDGIQuality::Medium => [24, 24, 6],
            DDGIQuality::High => [32, 32, 8],
            DDGIQuality::Ultra => [48, 48, 12],
        }
    }

    /// Get probe spacing in meters for this quality level.
    pub const fn spacing(self) -> f32 {
        match self {
            DDGIQuality::Low => 4.0,
            DDGIQuality::Medium => 3.0,
            DDGIQuality::High => 2.0,
            DDGIQuality::Ultra => 1.5,
        }
    }

    /// Get rays per probe for this quality level.
    pub const fn rays_per_probe(self) -> u32 {
        match self {
            DDGIQuality::Low => 32,
            DDGIQuality::Medium => 48,
            DDGIQuality::High => 64,
            DDGIQuality::Ultra => 128,
        }
    }

    /// Get total number of probes for this quality level.
    pub const fn total_probes(self) -> u32 {
        let dims = self.dimensions();
        dims[0] * dims[1] * dims[2]
    }

    /// Estimate total GPU memory usage in bytes.
    pub const fn estimated_memory_bytes(self) -> usize {
        let probe_count = self.total_probes() as usize;
        let grid_uniform = std::mem::size_of::<ProbeGridGpu>();
        let irradiance = probe_count * std::mem::size_of::<ProbeSH>();
        let visibility = probe_count * std::mem::size_of::<ProbeVis>();
        grid_uniform + irradiance + visibility
    }

    /// Get grid extents in world units (dimensions * spacing).
    pub fn world_extents(self) -> [f32; 3] {
        let dims = self.dimensions();
        let spacing = self.spacing();
        [
            (dims[0] - 1) as f32 * spacing,
            (dims[1] - 1) as f32 * spacing,
            (dims[2] - 1) as f32 * spacing,
        ]
    }
}

impl fmt::Display for DDGIQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DDGIQuality::Low => write!(f, "Low (16x16x4)"),
            DDGIQuality::Medium => write!(f, "Medium (24x24x6)"),
            DDGIQuality::High => write!(f, "High (32x32x8)"),
            DDGIQuality::Ultra => write!(f, "Ultra (48x48x12)"),
        }
    }
}

// ============================================================================
// Buffer Handles
// ============================================================================

/// Opaque handle to a GPU buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u32);

impl BufferHandle {
    /// Invalid/null buffer handle.
    pub const INVALID: Self = BufferHandle(u32::MAX);

    /// Check if handle is valid.
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

impl Default for BufferHandle {
    fn default() -> Self {
        Self::INVALID
    }
}

// ============================================================================
// DDGI Allocation
// ============================================================================

/// Fixed-size DDGI buffer allocation.
///
/// Contains all GPU buffers needed for a DDGI probe volume at a specific
/// quality level. Buffers are pre-allocated with fixed sizes.
#[derive(Clone, Debug)]
pub struct DDGIAllocation {
    /// Quality preset used for this allocation.
    pub quality: DDGIQuality,
    /// Probe grid dimensions (X, Y, Z).
    pub probe_count: [u32; 3],
    /// Probe spacing in world units.
    pub spacing: f32,
    /// Handle to irradiance buffer (ProbeSH array).
    pub irradiance_buffer: BufferHandle,
    /// Handle to visibility buffer (ProbeVis array).
    pub visibility_buffer: BufferHandle,
    /// Handle to grid uniform buffer (ProbeGridGpu).
    pub grid_uniform: BufferHandle,
    /// Total GPU memory allocated in bytes.
    pub allocated_bytes: usize,
}

impl DDGIAllocation {
    /// Get total number of probes.
    pub fn total_probes(&self) -> u32 {
        self.probe_count[0] * self.probe_count[1] * self.probe_count[2]
    }

    /// Get irradiance buffer size in bytes.
    pub fn irradiance_buffer_size(&self) -> usize {
        self.total_probes() as usize * std::mem::size_of::<ProbeSH>()
    }

    /// Get visibility buffer size in bytes.
    pub fn visibility_buffer_size(&self) -> usize {
        self.total_probes() as usize * std::mem::size_of::<ProbeVis>()
    }

    /// Get grid uniform buffer size in bytes.
    pub const fn grid_uniform_size(&self) -> usize {
        std::mem::size_of::<ProbeGridGpu>()
    }

    /// Check if allocation is valid.
    pub fn is_valid(&self) -> bool {
        self.irradiance_buffer.is_valid()
            && self.visibility_buffer.is_valid()
            && self.grid_uniform.is_valid()
    }
}

impl Default for DDGIAllocation {
    fn default() -> Self {
        Self {
            quality: DDGIQuality::default(),
            probe_count: [0, 0, 0],
            spacing: 0.0,
            irradiance_buffer: BufferHandle::INVALID,
            visibility_buffer: BufferHandle::INVALID,
            grid_uniform: BufferHandle::INVALID,
            allocated_bytes: 0,
        }
    }
}

// ============================================================================
// Allocation Result
// ============================================================================

/// Result of a DDGI buffer allocation.
#[derive(Debug)]
pub enum AllocationResult {
    /// Allocation succeeded.
    Success(DDGIAllocation),
    /// Device does not have enough memory.
    OutOfMemory { required: usize, available: usize },
    /// Quality preset is not supported.
    UnsupportedQuality(DDGIQuality),
    /// Device error during allocation.
    DeviceError(String),
}

impl AllocationResult {
    /// Check if allocation succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self, AllocationResult::Success(_))
    }

    /// Unwrap the allocation, panicking on failure.
    pub fn unwrap(self) -> DDGIAllocation {
        match self {
            AllocationResult::Success(alloc) => alloc,
            other => panic!("DDGI allocation failed: {:?}", other),
        }
    }

    /// Get allocation if successful, None otherwise.
    pub fn ok(self) -> Option<DDGIAllocation> {
        match self {
            AllocationResult::Success(alloc) => Some(alloc),
            _ => None,
        }
    }
}

// ============================================================================
// Allocator Trait
// ============================================================================

/// Trait for DDGI buffer allocation.
///
/// Implementations create GPU buffers for a specific graphics API.
pub trait DDGIAllocator {
    /// Allocate DDGI buffers for the given quality level.
    fn allocate(&mut self, quality: DDGIQuality) -> AllocationResult;

    /// Free previously allocated DDGI buffers.
    fn free(&mut self, allocation: &DDGIAllocation);

    /// Get maximum supported quality level.
    fn max_supported_quality(&self) -> DDGIQuality;

    /// Check if a quality level is supported.
    fn is_quality_supported(&self, quality: DDGIQuality) -> bool {
        quality as u8 <= self.max_supported_quality() as u8
    }
}

// ============================================================================
// Mock Allocator (for testing)
// ============================================================================

/// Mock allocator for testing without a GPU.
#[derive(Default)]
pub struct MockDDGIAllocator {
    next_handle: u32,
    allocations: Vec<DDGIAllocation>,
    memory_limit: Option<usize>,
}

impl MockDDGIAllocator {
    /// Create a new mock allocator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a mock allocator with a memory limit.
    pub fn with_memory_limit(limit: usize) -> Self {
        Self {
            memory_limit: Some(limit),
            ..Default::default()
        }
    }

    /// Get number of active allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Get total allocated memory.
    pub fn total_allocated(&self) -> usize {
        self.allocations.iter().map(|a| a.allocated_bytes).sum()
    }

    fn next_handle(&mut self) -> BufferHandle {
        let handle = BufferHandle(self.next_handle);
        self.next_handle += 1;
        handle
    }
}

impl DDGIAllocator for MockDDGIAllocator {
    fn allocate(&mut self, quality: DDGIQuality) -> AllocationResult {
        let required = quality.estimated_memory_bytes();

        // Check memory limit
        if let Some(limit) = self.memory_limit {
            let current = self.total_allocated();
            if current + required > limit {
                return AllocationResult::OutOfMemory {
                    required,
                    available: limit.saturating_sub(current),
                };
            }
        }

        let dims = quality.dimensions();
        let allocation = DDGIAllocation {
            quality,
            probe_count: dims,
            spacing: quality.spacing(),
            irradiance_buffer: self.next_handle(),
            visibility_buffer: self.next_handle(),
            grid_uniform: self.next_handle(),
            allocated_bytes: required,
        };

        self.allocations.push(allocation.clone());
        AllocationResult::Success(allocation)
    }

    fn free(&mut self, allocation: &DDGIAllocation) {
        self.allocations
            .retain(|a| a.irradiance_buffer != allocation.irradiance_buffer);
    }

    fn max_supported_quality(&self) -> DDGIQuality {
        DDGIQuality::Ultra
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create initial GPU data for a probe grid.
///
/// Returns a `ProbeGridGpu` struct initialized with the given quality settings.
pub fn create_probe_grid_gpu(quality: DDGIQuality, origin: [f32; 3]) -> ProbeGridGpu {
    let dims = quality.dimensions();
    let spacing = quality.spacing();

    ProbeGridGpu {
        origin,
        _pad0: 0.0,
        cell_size: [spacing, spacing, spacing],
        _pad1: 0.0,
        dimensions: dims,
        total_probes: dims[0] * dims[1] * dims[2],
        scroll_offset: [0, 0, 0],
        frame_index: 0,
    }
}

/// Create initial irradiance buffer data (zeroed).
pub fn create_irradiance_buffer(quality: DDGIQuality) -> Vec<ProbeSH> {
    let count = quality.total_probes() as usize;
    vec![ProbeSH::ZERO; count]
}

/// Create initial visibility buffer data (default values).
pub fn create_visibility_buffer(quality: DDGIQuality) -> Vec<ProbeVis> {
    let count = quality.total_probes() as usize;
    vec![ProbeVis::default(); count]
}

/// Compute bytes needed for a Pod type array.
pub fn buffer_bytes<T: Pod>(data: &[T]) -> &[u8] {
    bytemuck::cast_slice(data)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Quality preset tests ────────────────────────────────────────────────

    #[test]
    fn test_quality_dimensions_low() {
        assert_eq!(DDGIQuality::Low.dimensions(), [16, 16, 4]);
    }

    #[test]
    fn test_quality_dimensions_medium() {
        assert_eq!(DDGIQuality::Medium.dimensions(), [24, 24, 6]);
    }

    #[test]
    fn test_quality_dimensions_high() {
        assert_eq!(DDGIQuality::High.dimensions(), [32, 32, 8]);
    }

    #[test]
    fn test_quality_dimensions_ultra() {
        assert_eq!(DDGIQuality::Ultra.dimensions(), [48, 48, 12]);
    }

    #[test]
    fn test_quality_spacing_low() {
        assert!((DDGIQuality::Low.spacing() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_spacing_medium() {
        assert!((DDGIQuality::Medium.spacing() - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_spacing_high() {
        assert!((DDGIQuality::High.spacing() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_spacing_ultra() {
        assert!((DDGIQuality::Ultra.spacing() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_total_probes_low() {
        assert_eq!(DDGIQuality::Low.total_probes(), 16 * 16 * 4);
    }

    #[test]
    fn test_quality_total_probes_medium() {
        assert_eq!(DDGIQuality::Medium.total_probes(), 24 * 24 * 6);
    }

    #[test]
    fn test_quality_total_probes_high() {
        assert_eq!(DDGIQuality::High.total_probes(), 32 * 32 * 8);
    }

    #[test]
    fn test_quality_total_probes_ultra() {
        assert_eq!(DDGIQuality::Ultra.total_probes(), 48 * 48 * 12);
    }

    #[test]
    fn test_quality_rays_per_probe() {
        assert_eq!(DDGIQuality::Low.rays_per_probe(), 32);
        assert_eq!(DDGIQuality::Medium.rays_per_probe(), 48);
        assert_eq!(DDGIQuality::High.rays_per_probe(), 64);
        assert_eq!(DDGIQuality::Ultra.rays_per_probe(), 128);
    }

    // ── Memory estimation tests ─────────────────────────────────────────────

    #[test]
    fn test_memory_estimation_low() {
        // LOW: 1024 probes * (192 + 16) + 64 = 212,992 bytes (~208KB)
        let mem = DDGIQuality::Low.estimated_memory_bytes();
        assert!(mem > 200_000);
        assert!(mem < 250_000);
    }

    #[test]
    fn test_memory_estimation_medium() {
        // MEDIUM: 3456 probes * (192 + 16) + 64 = 718,912 bytes (~702KB)
        let mem = DDGIQuality::Medium.estimated_memory_bytes();
        assert!(mem > 600_000);
        assert!(mem < 800_000);
    }

    #[test]
    fn test_memory_estimation_high() {
        // HIGH: 8192 probes * (192 + 16) + 64 = 1,703,936 bytes (~1.6MB)
        let mem = DDGIQuality::High.estimated_memory_bytes();
        assert!(mem > 1_500_000);
        assert!(mem < 2_000_000);
    }

    #[test]
    fn test_memory_estimation_ultra() {
        // ULTRA: 27648 probes * (192 + 16) + 64 = 5,750,848 bytes (~5.5MB)
        let mem = DDGIQuality::Ultra.estimated_memory_bytes();
        assert!(mem > 5_000_000);
        assert!(mem < 6_000_000);
    }

    // ── World extents tests ─────────────────────────────────────────────────

    #[test]
    fn test_world_extents_low() {
        let extents = DDGIQuality::Low.world_extents();
        // (16-1) * 4 = 60, (16-1) * 4 = 60, (4-1) * 4 = 12
        assert!((extents[0] - 60.0).abs() < f32::EPSILON);
        assert!((extents[1] - 60.0).abs() < f32::EPSILON);
        assert!((extents[2] - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_world_extents_high() {
        let extents = DDGIQuality::High.world_extents();
        // (32-1) * 2 = 62, (32-1) * 2 = 62, (8-1) * 2 = 14
        assert!((extents[0] - 62.0).abs() < f32::EPSILON);
        assert!((extents[1] - 62.0).abs() < f32::EPSILON);
        assert!((extents[2] - 14.0).abs() < f32::EPSILON);
    }

    // ── Buffer handle tests ─────────────────────────────────────────────────

    #[test]
    fn test_buffer_handle_invalid() {
        let handle = BufferHandle::INVALID;
        assert!(!handle.is_valid());
    }

    #[test]
    fn test_buffer_handle_valid() {
        let handle = BufferHandle(0);
        assert!(handle.is_valid());
    }

    #[test]
    fn test_buffer_handle_default() {
        let handle = BufferHandle::default();
        assert_eq!(handle, BufferHandle::INVALID);
    }

    // ── Allocation tests ────────────────────────────────────────────────────

    #[test]
    fn test_allocation_default() {
        let alloc = DDGIAllocation::default();
        assert!(!alloc.is_valid());
        assert_eq!(alloc.total_probes(), 0);
    }

    #[test]
    fn test_allocation_buffer_sizes() {
        let alloc = DDGIAllocation {
            quality: DDGIQuality::High,
            probe_count: [32, 32, 8],
            spacing: 2.0,
            irradiance_buffer: BufferHandle(0),
            visibility_buffer: BufferHandle(1),
            grid_uniform: BufferHandle(2),
            allocated_bytes: DDGIQuality::High.estimated_memory_bytes(),
        };

        assert!(alloc.is_valid());
        assert_eq!(alloc.total_probes(), 8192);
        assert_eq!(alloc.irradiance_buffer_size(), 8192 * 192);
        assert_eq!(alloc.visibility_buffer_size(), 8192 * 16);
        assert_eq!(alloc.grid_uniform_size(), 64);
    }

    // ── Mock allocator tests ────────────────────────────────────────────────

    #[test]
    fn test_mock_allocator_allocate() {
        let mut alloc = MockDDGIAllocator::new();
        let result = alloc.allocate(DDGIQuality::High);

        assert!(result.is_success());
        let allocation = result.unwrap();
        assert!(allocation.is_valid());
        assert_eq!(allocation.probe_count, [32, 32, 8]);
    }

    #[test]
    fn test_mock_allocator_multiple_allocations() {
        let mut alloc = MockDDGIAllocator::new();

        let r1 = alloc.allocate(DDGIQuality::Low);
        let r2 = alloc.allocate(DDGIQuality::Medium);

        assert!(r1.is_success());
        assert!(r2.is_success());
        assert_eq!(alloc.allocation_count(), 2);
    }

    #[test]
    fn test_mock_allocator_free() {
        let mut alloc = MockDDGIAllocator::new();

        let allocation = alloc.allocate(DDGIQuality::High).unwrap();
        assert_eq!(alloc.allocation_count(), 1);

        alloc.free(&allocation);
        assert_eq!(alloc.allocation_count(), 0);
    }

    #[test]
    fn test_mock_allocator_memory_limit() {
        let mut alloc = MockDDGIAllocator::with_memory_limit(100_000);

        let result = alloc.allocate(DDGIQuality::High);

        match result {
            AllocationResult::OutOfMemory { required, available } => {
                assert!(required > 100_000);
                assert!(available <= 100_000);
            }
            _ => panic!("Expected OutOfMemory"),
        }
    }

    #[test]
    fn test_mock_allocator_total_allocated() {
        let mut alloc = MockDDGIAllocator::new();

        let a1 = alloc.allocate(DDGIQuality::Low).unwrap();
        let total_after_low = alloc.total_allocated();

        let _a2 = alloc.allocate(DDGIQuality::Medium).unwrap();
        let total_after_both = alloc.total_allocated();

        assert!(total_after_both > total_after_low);

        alloc.free(&a1);
        assert!(alloc.total_allocated() < total_after_both);
    }

    #[test]
    fn test_mock_allocator_max_quality() {
        let alloc = MockDDGIAllocator::new();
        assert_eq!(alloc.max_supported_quality(), DDGIQuality::Ultra);
    }

    #[test]
    fn test_mock_allocator_quality_supported() {
        let alloc = MockDDGIAllocator::new();
        assert!(alloc.is_quality_supported(DDGIQuality::Low));
        assert!(alloc.is_quality_supported(DDGIQuality::Ultra));
    }

    // ── Convenience function tests ──────────────────────────────────────────

    #[test]
    fn test_create_probe_grid_gpu() {
        let grid = create_probe_grid_gpu(DDGIQuality::High, [1.0, 2.0, 3.0]);

        assert_eq!(grid.origin, [1.0, 2.0, 3.0]);
        assert_eq!(grid.cell_size, [2.0, 2.0, 2.0]);
        assert_eq!(grid.dimensions, [32, 32, 8]);
        assert_eq!(grid.total_probes, 8192);
        assert_eq!(grid.scroll_offset, [0, 0, 0]);
        assert_eq!(grid.frame_index, 0);
    }

    #[test]
    fn test_create_irradiance_buffer() {
        let buffer = create_irradiance_buffer(DDGIQuality::Low);
        assert_eq!(buffer.len(), 1024);
    }

    #[test]
    fn test_create_visibility_buffer() {
        let buffer = create_visibility_buffer(DDGIQuality::Low);
        assert_eq!(buffer.len(), 1024);
        // Check default values
        assert_eq!(buffer[0].mean_distance, f32::MAX);
        assert_eq!(buffer[0].confidence, 0.0);
    }

    #[test]
    fn test_buffer_bytes() {
        let grid = create_probe_grid_gpu(DDGIQuality::Low, [0.0; 3]);
        let bytes = buffer_bytes(std::slice::from_ref(&grid));
        assert_eq!(bytes.len(), 64);
    }

    // ── AllocationResult tests ──────────────────────────────────────────────

    #[test]
    fn test_allocation_result_success() {
        let result = AllocationResult::Success(DDGIAllocation::default());
        assert!(result.is_success());
    }

    #[test]
    fn test_allocation_result_out_of_memory() {
        let result = AllocationResult::OutOfMemory {
            required: 1000,
            available: 500,
        };
        assert!(!result.is_success());
    }

    #[test]
    fn test_allocation_result_ok() {
        let result = AllocationResult::Success(DDGIAllocation::default());
        assert!(result.ok().is_some());

        let result = AllocationResult::OutOfMemory {
            required: 1000,
            available: 500,
        };
        assert!(result.ok().is_none());
    }

    // ── Display tests ───────────────────────────────────────────────────────

    #[test]
    fn test_quality_display() {
        assert_eq!(format!("{}", DDGIQuality::Low), "Low (16x16x4)");
        assert_eq!(format!("{}", DDGIQuality::Medium), "Medium (24x24x6)");
        assert_eq!(format!("{}", DDGIQuality::High), "High (32x32x8)");
        assert_eq!(format!("{}", DDGIQuality::Ultra), "Ultra (48x48x12)");
    }

    // ── Default tests ───────────────────────────────────────────────────────

    #[test]
    fn test_quality_default() {
        let quality: DDGIQuality = Default::default();
        assert_eq!(quality, DDGIQuality::High);
    }
}
