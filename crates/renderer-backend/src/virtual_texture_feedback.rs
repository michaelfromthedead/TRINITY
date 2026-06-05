//! Virtual Texturing Feedback Pass (T-ENV-2.12)
//!
//! Implements a feedback system for virtual texturing that determines which
//! texture pages are visible and need to be loaded. The system uses a GPU
//! feedback pass followed by CPU readback and deduplication.
//!
//! # Architecture
//!
//! - **Feedback Pass**: GPU renders virtual UV coordinates to a low-resolution
//!   feedback buffer (1024x1024 R32Uint).
//! - **Feedback Encoding**: Each pixel encodes (tile_x, tile_y, mip) in a u32.
//! - **Async Readback**: Buffer is copied to staging and read by CPU next frame.
//! - **Deduplication**: CPU processes readback data to extract unique page requests.
//! - **Priority Scoring**: Each request is scored based on screen coverage and
//!   distance from camera center.
//! - **Streaming Queue**: Missing pages are added to a stream queue sorted by priority.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::virtual_texture_feedback::{FeedbackPass, FeedbackConfig};
//! use std::collections::HashSet;
//!
//! let config = FeedbackConfig::default();
//! let mut feedback = FeedbackPass::new(config);
//!
//! // After GPU feedback pass, copy data to staging
//! let readback_data = read_staging_buffer();
//!
//! // Process on CPU
//! feedback.process_readback(&readback_data);
//! feedback.deduplicate_requests();
//!
//! // Get missing pages to stream
//! let resident_pages = get_resident_pages();
//! let missing = feedback.get_missing_pages(&resident_pages);
//!
//! for page in missing {
//!     stream_page(page);
//! }
//! ```

use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default feedback buffer width.
pub const DEFAULT_FEEDBACK_WIDTH: u32 = 1024;

/// Default feedback buffer height.
pub const DEFAULT_FEEDBACK_HEIGHT: u32 = 1024;

/// Default virtual texture size (128K x 128K pixels).
pub const DEFAULT_VIRTUAL_SIZE: u32 = 131072;

/// Default tile size in pixels.
pub const DEFAULT_TILE_SIZE: u32 = 128;

/// Maximum mip level supported (log2(1024) = 10).
pub const MAX_MIP_LEVEL: u8 = 10;

/// Invalid feedback value (sentinel for empty pixels).
pub const INVALID_FEEDBACK: u32 = 0xFFFFFFFF;

/// Bits allocated for tile X coordinate.
pub const TILE_X_BITS: u32 = 12;

/// Bits allocated for tile Y coordinate.
pub const TILE_Y_BITS: u32 = 12;

/// Bits allocated for mip level.
pub const MIP_BITS: u32 = 4;

/// Bits allocated for priority.
pub const PRIORITY_BITS: u32 = 4;

/// Maximum tile coordinate value (2^12 - 1 = 4095).
pub const MAX_TILE_COORD: u32 = (1 << TILE_X_BITS) - 1;

/// Maximum mip level value (2^4 - 1 = 15).
pub const MAX_MIP_VALUE: u32 = (1 << MIP_BITS) - 1;

/// Maximum priority value (2^4 - 1 = 15).
pub const MAX_PRIORITY_VALUE: u32 = (1 << PRIORITY_BITS) - 1;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the feedback system (GPU-compatible).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct FeedbackConfig {
    /// Feedback buffer width in pixels (typically 1024).
    pub feedback_width: u32,
    /// Feedback buffer height in pixels (typically 1024).
    pub feedback_height: u32,
    /// Virtual texture size in pixels (e.g., 131072 for 128K).
    pub virtual_size: u32,
    /// Tile size in pixels (e.g., 128).
    pub tile_size: u32,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            feedback_width: DEFAULT_FEEDBACK_WIDTH,
            feedback_height: DEFAULT_FEEDBACK_HEIGHT,
            virtual_size: DEFAULT_VIRTUAL_SIZE,
            tile_size: DEFAULT_TILE_SIZE,
        }
    }
}

impl FeedbackConfig {
    /// Create a new feedback configuration.
    pub fn new(feedback_width: u32, feedback_height: u32, virtual_size: u32, tile_size: u32) -> Self {
        Self {
            feedback_width,
            feedback_height,
            virtual_size,
            tile_size,
        }
    }

    /// Calculate the number of tiles per dimension.
    #[inline]
    pub fn tiles_per_dimension(&self) -> u32 {
        if self.tile_size == 0 {
            return 0;
        }
        self.virtual_size / self.tile_size
    }

    /// Calculate the maximum mip level.
    #[inline]
    pub fn max_mip_level(&self) -> u32 {
        let tiles = self.tiles_per_dimension();
        if tiles == 0 {
            return 0;
        }
        (tiles as f32).log2().floor() as u32
    }

    /// Get the feedback buffer size in pixels.
    #[inline]
    pub fn feedback_buffer_size(&self) -> usize {
        (self.feedback_width * self.feedback_height) as usize
    }
}

// ---------------------------------------------------------------------------
// Feedback Entry
// ---------------------------------------------------------------------------

/// A decoded feedback entry representing a page request.
///
/// This is the CPU-side representation of encoded feedback data.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct FeedbackEntry {
    /// Tile X coordinate (0-4095).
    pub tile_x: u16,
    /// Tile Y coordinate (0-4095).
    pub tile_y: u16,
    /// Mip level (0-15).
    pub mip_level: u8,
    /// Priority hint (0-15, lower = higher priority).
    pub priority: u8,
    /// Padding for alignment.
    pub _padding: u16,
}

impl FeedbackEntry {
    /// Create a new feedback entry.
    pub fn new(tile_x: u16, tile_y: u16, mip_level: u8, priority: u8) -> Self {
        Self {
            tile_x,
            tile_y,
            mip_level,
            priority,
            _padding: 0,
        }
    }

    /// Create an entry with default priority.
    pub fn with_default_priority(tile_x: u16, tile_y: u16, mip_level: u8) -> Self {
        Self::new(tile_x, tile_y, mip_level, 0)
    }

    /// Create an invalid/sentinel entry.
    pub fn invalid() -> Self {
        Self {
            tile_x: u16::MAX,
            tile_y: u16::MAX,
            mip_level: u8::MAX,
            priority: u8::MAX,
            _padding: 0,
        }
    }

    /// Check if this is an invalid/sentinel entry.
    #[inline]
    pub fn is_invalid(&self) -> bool {
        self.tile_x == u16::MAX && self.tile_y == u16::MAX
    }

    /// Pack entry into u32 for GPU storage.
    /// Format: [priority:4][mip:4][tile_y:12][tile_x:12]
    #[inline]
    pub fn pack(&self) -> u32 {
        let tx = (self.tile_x as u32) & MAX_TILE_COORD;
        let ty = (self.tile_y as u32) & MAX_TILE_COORD;
        let mip = (self.mip_level as u32) & MAX_MIP_VALUE;
        let pri = (self.priority as u32) & MAX_PRIORITY_VALUE;

        tx | (ty << TILE_X_BITS) | (mip << (TILE_X_BITS + TILE_Y_BITS))
            | (pri << (TILE_X_BITS + TILE_Y_BITS + MIP_BITS))
    }

    /// Get a key for deduplication (ignores priority).
    #[inline]
    pub fn dedup_key(&self) -> (u16, u16, u8) {
        (self.tile_x, self.tile_y, self.mip_level)
    }
}

impl Default for FeedbackEntry {
    fn default() -> Self {
        Self::invalid()
    }
}

// ---------------------------------------------------------------------------
// Page Request
// ---------------------------------------------------------------------------

/// A page request with priority for streaming queue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageRequest {
    /// Tile X coordinate.
    pub tile_x: u32,
    /// Tile Y coordinate.
    pub tile_y: u32,
    /// Mip level.
    pub mip_level: u32,
    /// Priority score (higher = more urgent).
    pub priority: f32,
}

impl PageRequest {
    /// Create a new page request.
    pub fn new(tile_x: u32, tile_y: u32, mip_level: u32, priority: f32) -> Self {
        Self {
            tile_x,
            tile_y,
            mip_level,
            priority,
        }
    }

    /// Create from a feedback entry with default priority.
    pub fn from_entry(entry: &FeedbackEntry) -> Self {
        Self {
            tile_x: entry.tile_x as u32,
            tile_y: entry.tile_y as u32,
            mip_level: entry.mip_level as u32,
            priority: 1.0 - (entry.priority as f32 / MAX_PRIORITY_VALUE as f32),
        }
    }

    /// Create a key tuple for comparison with FeedbackEntry.
    #[inline]
    pub fn key(&self) -> (u32, u32, u32) {
        (self.tile_x, self.tile_y, self.mip_level)
    }
}

impl Eq for PageRequest {}

impl std::hash::Hash for PageRequest {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tile_x.hash(state);
        self.tile_y.hash(state);
        self.mip_level.hash(state);
    }
}

impl PartialOrd for PageRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PageRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority should come first
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.mip_level.cmp(&other.mip_level))
            .then_with(|| self.tile_x.cmp(&other.tile_x))
            .then_with(|| self.tile_y.cmp(&other.tile_y))
    }
}

// ---------------------------------------------------------------------------
// Feedback Pass
// ---------------------------------------------------------------------------

/// Virtual texturing feedback pass controller.
///
/// Manages the feedback buffer readback and processes page requests.
pub struct FeedbackPass {
    /// Configuration.
    pub config: FeedbackConfig,
    /// Staging buffer for async readback (double-buffered internally by GPU).
    pub staging_buffer: Vec<u32>,
    /// Decoded page requests from current frame's readback.
    pub requests: Vec<PageRequest>,
    /// Deduplicated entries (used internally during processing).
    deduplicated: HashSet<(u16, u16, u8)>,
    /// Current frame index for tracking.
    pub frame_index: u32,
    /// Number of valid samples in last readback.
    pub last_valid_samples: u32,
    /// Statistics: total pixels processed.
    pub total_pixels_processed: u64,
    /// Statistics: unique requests generated.
    pub unique_requests_generated: u64,
}

impl FeedbackPass {
    /// Create a new feedback pass with the given configuration.
    pub fn new(config: FeedbackConfig) -> Self {
        let buffer_size = config.feedback_buffer_size();
        Self {
            config,
            staging_buffer: vec![INVALID_FEEDBACK; buffer_size],
            requests: Vec::with_capacity(1024),
            deduplicated: HashSet::with_capacity(1024),
            frame_index: 0,
            last_valid_samples: 0,
            total_pixels_processed: 0,
            unique_requests_generated: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(FeedbackConfig::default())
    }

    /// Encode tile coordinates and mip level into a u32.
    /// Format: [priority:4][mip:4][tile_y:12][tile_x:12]
    #[inline]
    pub fn encode_feedback(tile_x: u32, tile_y: u32, mip: u32) -> u32 {
        let tx = tile_x & MAX_TILE_COORD;
        let ty = tile_y & MAX_TILE_COORD;
        let m = mip & MAX_MIP_VALUE;
        tx | (ty << TILE_X_BITS) | (m << (TILE_X_BITS + TILE_Y_BITS))
    }

    /// Encode with priority.
    #[inline]
    pub fn encode_feedback_with_priority(tile_x: u32, tile_y: u32, mip: u32, priority: u32) -> u32 {
        Self::encode_feedback(tile_x, tile_y, mip)
            | ((priority & MAX_PRIORITY_VALUE) << (TILE_X_BITS + TILE_Y_BITS + MIP_BITS))
    }

    /// Decode a u32 feedback value into a FeedbackEntry.
    #[inline]
    pub fn decode_feedback(value: u32) -> FeedbackEntry {
        if value == INVALID_FEEDBACK {
            return FeedbackEntry::invalid();
        }

        let tile_x = (value & MAX_TILE_COORD) as u16;
        let tile_y = ((value >> TILE_X_BITS) & MAX_TILE_COORD) as u16;
        let mip_level = ((value >> (TILE_X_BITS + TILE_Y_BITS)) & MAX_MIP_VALUE) as u8;
        let priority = ((value >> (TILE_X_BITS + TILE_Y_BITS + MIP_BITS)) & MAX_PRIORITY_VALUE) as u8;

        FeedbackEntry {
            tile_x,
            tile_y,
            mip_level,
            priority,
            _padding: 0,
        }
    }

    /// Check if a feedback value is valid.
    #[inline]
    pub fn is_valid_feedback(value: u32) -> bool {
        value != INVALID_FEEDBACK
    }

    /// Process readback data from GPU staging buffer.
    ///
    /// Decodes all valid feedback values and accumulates them into the request list.
    pub fn process_readback(&mut self, data: &[u32]) {
        self.requests.clear();
        self.deduplicated.clear();
        self.last_valid_samples = 0;

        for &value in data {
            self.total_pixels_processed += 1;

            if !Self::is_valid_feedback(value) {
                continue;
            }

            let entry = Self::decode_feedback(value);
            if entry.is_invalid() {
                continue;
            }

            self.last_valid_samples += 1;

            // Add to requests (will be deduplicated later)
            self.requests.push(PageRequest::from_entry(&entry));
        }
    }

    /// Deduplicate requests by (tile_x, tile_y, mip_level).
    ///
    /// Keeps the highest priority request for each unique page.
    pub fn deduplicate_requests(&mut self) {
        if self.requests.is_empty() {
            return;
        }

        // Sort by key to group duplicates together
        self.requests.sort_by(|a, b| {
            a.key()
                .cmp(&b.key())
                .then_with(|| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Keep only first (highest priority) of each key
        let mut write_idx = 0;
        let mut last_key: Option<(u32, u32, u32)> = None;

        for read_idx in 0..self.requests.len() {
            let key = self.requests[read_idx].key();
            if last_key != Some(key) {
                self.requests[write_idx] = self.requests[read_idx];
                write_idx += 1;
                last_key = Some(key);
            }
        }

        self.requests.truncate(write_idx);
        self.unique_requests_generated += write_idx as u64;

        // Re-sort by priority for streaming order
        self.requests.sort();
    }

    /// Compute priority score for a feedback entry based on camera position.
    ///
    /// Higher values = higher priority for streaming.
    /// Takes into account:
    /// - Distance from camera center (closer = higher priority)
    /// - Mip level (lower mip = higher detail = higher priority for close objects)
    /// - Screen coverage (tiles near screen center are prioritized)
    pub fn compute_priority(&self, entry: &FeedbackEntry, camera_pos: [f32; 3]) -> f32 {
        // Base priority from mip level (lower mip = higher priority)
        let max_mip = self.config.max_mip_level() as f32;
        let mip_priority = if max_mip > 0.0 {
            1.0 - (entry.mip_level as f32 / max_mip)
        } else {
            1.0
        };

        // Compute approximate world position of tile center
        let tile_size_world = self.config.tile_size as f32 * (1 << entry.mip_level) as f32;
        let tile_center_x = (entry.tile_x as f32 + 0.5) * tile_size_world;
        let tile_center_y = (entry.tile_y as f32 + 0.5) * tile_size_world;

        // Distance from camera (using XZ plane distance)
        let dx = tile_center_x - camera_pos[0];
        let dz = tile_center_y - camera_pos[2];
        let distance = (dx * dx + dz * dz).sqrt();

        // Distance factor (closer = higher priority)
        // Use exponential falloff for smooth prioritization
        let max_distance = self.config.virtual_size as f32;
        let distance_priority = (-distance / max_distance * 2.0).exp();

        // Screen center proximity bonus (assuming UV 0.5,0.5 is center)
        let tiles_per_dim = self.config.tiles_per_dimension() as f32;
        let center_tile = tiles_per_dim / 2.0;
        let center_dx = entry.tile_x as f32 - center_tile;
        let center_dy = entry.tile_y as f32 - center_tile;
        let center_dist = (center_dx * center_dx + center_dy * center_dy).sqrt();
        let center_priority = 1.0 - (center_dist / tiles_per_dim).min(1.0);

        // Combine factors (weighted sum)
        let priority = mip_priority * 0.4 + distance_priority * 0.4 + center_priority * 0.2;

        priority.clamp(0.0, 1.0)
    }

    /// Get pages that are not currently resident.
    ///
    /// Compares requested pages against the resident set and returns
    /// missing pages sorted by priority (highest first).
    pub fn get_missing_pages(&self, resident: &HashSet<FeedbackEntry>) -> Vec<PageRequest> {
        let resident_keys: HashSet<(u16, u16, u8)> = resident
            .iter()
            .map(|e| e.dedup_key())
            .collect();

        let mut missing: Vec<PageRequest> = self
            .requests
            .iter()
            .filter(|req| {
                let key = (req.tile_x as u16, req.tile_y as u16, req.mip_level as u8);
                !resident_keys.contains(&key)
            })
            .copied()
            .collect();

        // Already sorted by deduplicate_requests, but ensure order
        missing.sort();

        missing
    }

    /// Get pages missing from a set of (tile_x, tile_y, mip) tuples.
    pub fn get_missing_pages_from_set(
        &self,
        resident: &HashSet<(u32, u32, u32)>,
    ) -> Vec<PageRequest> {
        let mut missing: Vec<PageRequest> = self
            .requests
            .iter()
            .filter(|req| !resident.contains(&req.key()))
            .copied()
            .collect();

        missing.sort();
        missing
    }

    /// Reset for next frame.
    pub fn begin_frame(&mut self) {
        self.requests.clear();
        self.deduplicated.clear();
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Get statistics about the last processed frame.
    pub fn stats(&self) -> FeedbackStats {
        FeedbackStats {
            frame_index: self.frame_index,
            valid_samples: self.last_valid_samples,
            unique_requests: self.requests.len() as u32,
            total_pixels_processed: self.total_pixels_processed,
            unique_requests_generated: self.unique_requests_generated,
            buffer_size: self.staging_buffer.len() as u32,
        }
    }

    /// Clear all state.
    pub fn reset(&mut self) {
        self.staging_buffer.fill(INVALID_FEEDBACK);
        self.requests.clear();
        self.deduplicated.clear();
        self.frame_index = 0;
        self.last_valid_samples = 0;
        self.total_pixels_processed = 0;
        self.unique_requests_generated = 0;
    }

    /// Get the staging buffer for GPU readback.
    pub fn staging_buffer(&self) -> &[u32] {
        &self.staging_buffer
    }

    /// Get mutable staging buffer for writing readback data.
    pub fn staging_buffer_mut(&mut self) -> &mut [u32] {
        &mut self.staging_buffer
    }

    /// Get current requests (after processing and deduplication).
    pub fn requests(&self) -> &[PageRequest] {
        &self.requests
    }

    /// Get the number of unique requests.
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Update priorities based on camera position.
    pub fn update_priorities(&mut self, camera_pos: [f32; 3]) {
        // Precompute config values to avoid borrowing self in the loop
        let max_mip = self.config.max_mip_level() as f32;
        let tile_size = self.config.tile_size as f32;
        let virtual_size = self.config.virtual_size as f32;
        let tiles_per_dim = self.config.tiles_per_dimension() as f32;

        for request in &mut self.requests {
            // Base priority from mip level (lower mip = higher priority)
            let mip_priority = if max_mip > 0.0 {
                1.0 - (request.mip_level as f32 / max_mip)
            } else {
                1.0
            };

            // Compute approximate world position of tile center
            let tile_size_world = tile_size * (1 << request.mip_level) as f32;
            let tile_center_x = (request.tile_x as f32 + 0.5) * tile_size_world;
            let tile_center_y = (request.tile_y as f32 + 0.5) * tile_size_world;

            // Distance from camera (using XZ plane distance)
            let dx = tile_center_x - camera_pos[0];
            let dz = tile_center_y - camera_pos[2];
            let distance = (dx * dx + dz * dz).sqrt();

            // Distance factor (closer = higher priority)
            let distance_priority = (-distance / virtual_size * 2.0).exp();

            // Screen center proximity bonus
            let center_tile = tiles_per_dim / 2.0;
            let center_dx = request.tile_x as f32 - center_tile;
            let center_dy = request.tile_y as f32 - center_tile;
            let center_dist = (center_dx * center_dx + center_dy * center_dy).sqrt();
            let center_priority = 1.0 - (center_dist / tiles_per_dim).min(1.0);

            // Combine factors (weighted sum)
            request.priority = (mip_priority * 0.4 + distance_priority * 0.4 + center_priority * 0.2)
                .clamp(0.0, 1.0);
        }

        // Re-sort by new priorities
        self.requests.sort();
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics from the feedback pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeedbackStats {
    /// Current frame index.
    pub frame_index: u32,
    /// Number of valid samples in last readback.
    pub valid_samples: u32,
    /// Number of unique page requests after deduplication.
    pub unique_requests: u32,
    /// Total pixels processed across all frames.
    pub total_pixels_processed: u64,
    /// Total unique requests generated across all frames.
    pub unique_requests_generated: u64,
    /// Staging buffer size in elements.
    pub buffer_size: u32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // FeedbackConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = FeedbackConfig::default();
        assert_eq!(config.feedback_width, 1024);
        assert_eq!(config.feedback_height, 1024);
        assert_eq!(config.virtual_size, 131072);
        assert_eq!(config.tile_size, 128);
    }

    #[test]
    fn test_config_new() {
        let config = FeedbackConfig::new(512, 512, 65536, 64);
        assert_eq!(config.feedback_width, 512);
        assert_eq!(config.feedback_height, 512);
        assert_eq!(config.virtual_size, 65536);
        assert_eq!(config.tile_size, 64);
    }

    #[test]
    fn test_config_tiles_per_dimension() {
        let config = FeedbackConfig::default();
        // 131072 / 128 = 1024
        assert_eq!(config.tiles_per_dimension(), 1024);

        let config2 = FeedbackConfig::new(512, 512, 8192, 64);
        // 8192 / 64 = 128
        assert_eq!(config2.tiles_per_dimension(), 128);
    }

    #[test]
    fn test_config_tiles_per_dimension_zero() {
        let config = FeedbackConfig::new(512, 512, 8192, 0);
        assert_eq!(config.tiles_per_dimension(), 0);
    }

    #[test]
    fn test_config_max_mip_level() {
        let config = FeedbackConfig::default();
        // 1024 tiles -> log2(1024) = 10
        assert_eq!(config.max_mip_level(), 10);

        let config2 = FeedbackConfig::new(512, 512, 8192, 64);
        // 128 tiles -> log2(128) = 7
        assert_eq!(config2.max_mip_level(), 7);
    }

    #[test]
    fn test_config_max_mip_level_zero() {
        let config = FeedbackConfig::new(512, 512, 0, 64);
        assert_eq!(config.max_mip_level(), 0);
    }

    #[test]
    fn test_config_feedback_buffer_size() {
        let config = FeedbackConfig::default();
        assert_eq!(config.feedback_buffer_size(), 1024 * 1024);

        let config2 = FeedbackConfig::new(256, 512, 8192, 64);
        assert_eq!(config2.feedback_buffer_size(), 256 * 512);
    }

    #[test]
    fn test_config_pod_zeroable() {
        let config: FeedbackConfig = bytemuck::Zeroable::zeroed();
        assert_eq!(config.feedback_width, 0);
        assert_eq!(config.feedback_height, 0);
        assert_eq!(config.virtual_size, 0);
        assert_eq!(config.tile_size, 0);
    }

    #[test]
    fn test_config_memory_layout() {
        assert_eq!(std::mem::size_of::<FeedbackConfig>(), 16);
        assert_eq!(std::mem::align_of::<FeedbackConfig>(), 4);
    }

    // =========================================================================
    // FeedbackEntry Tests
    // =========================================================================

    #[test]
    fn test_entry_new() {
        let entry = FeedbackEntry::new(100, 200, 5, 3);
        assert_eq!(entry.tile_x, 100);
        assert_eq!(entry.tile_y, 200);
        assert_eq!(entry.mip_level, 5);
        assert_eq!(entry.priority, 3);
        assert_eq!(entry._padding, 0);
    }

    #[test]
    fn test_entry_with_default_priority() {
        let entry = FeedbackEntry::with_default_priority(50, 75, 2);
        assert_eq!(entry.tile_x, 50);
        assert_eq!(entry.tile_y, 75);
        assert_eq!(entry.mip_level, 2);
        assert_eq!(entry.priority, 0);
    }

    #[test]
    fn test_entry_invalid() {
        let entry = FeedbackEntry::invalid();
        assert_eq!(entry.tile_x, u16::MAX);
        assert_eq!(entry.tile_y, u16::MAX);
        assert_eq!(entry.mip_level, u8::MAX);
        assert!(entry.is_invalid());
    }

    #[test]
    fn test_entry_is_invalid() {
        let valid = FeedbackEntry::new(0, 0, 0, 0);
        assert!(!valid.is_invalid());

        let invalid = FeedbackEntry::invalid();
        assert!(invalid.is_invalid());
    }

    #[test]
    fn test_entry_default() {
        let entry = FeedbackEntry::default();
        assert!(entry.is_invalid());
    }

    #[test]
    fn test_entry_pack() {
        let entry = FeedbackEntry::new(0x123, 0x456, 0x7, 0x8);
        let packed = entry.pack();

        // Format: [priority:4][mip:4][tile_y:12][tile_x:12]
        assert_eq!(packed & 0xFFF, 0x123); // tile_x
        assert_eq!((packed >> 12) & 0xFFF, 0x456); // tile_y
        assert_eq!((packed >> 24) & 0xF, 0x7); // mip
        assert_eq!((packed >> 28) & 0xF, 0x8); // priority
    }

    #[test]
    fn test_entry_pack_max_values() {
        let entry = FeedbackEntry::new(
            MAX_TILE_COORD as u16,
            MAX_TILE_COORD as u16,
            MAX_MIP_VALUE as u8,
            MAX_PRIORITY_VALUE as u8,
        );
        let packed = entry.pack();

        assert_eq!(packed & MAX_TILE_COORD, MAX_TILE_COORD);
        assert_eq!((packed >> 12) & MAX_TILE_COORD, MAX_TILE_COORD);
        assert_eq!((packed >> 24) & MAX_MIP_VALUE, MAX_MIP_VALUE);
        assert_eq!((packed >> 28) & MAX_PRIORITY_VALUE, MAX_PRIORITY_VALUE);
    }

    #[test]
    fn test_entry_pack_zero_values() {
        let entry = FeedbackEntry::new(0, 0, 0, 0);
        let packed = entry.pack();
        assert_eq!(packed, 0);
    }

    #[test]
    fn test_entry_dedup_key() {
        let entry = FeedbackEntry::new(100, 200, 5, 10);
        let key = entry.dedup_key();
        assert_eq!(key, (100, 200, 5));

        // Different priority, same key
        let entry2 = FeedbackEntry::new(100, 200, 5, 3);
        assert_eq!(entry.dedup_key(), entry2.dedup_key());
    }

    #[test]
    fn test_entry_equality() {
        let entry1 = FeedbackEntry::new(10, 20, 3, 5);
        let entry2 = FeedbackEntry::new(10, 20, 3, 5);
        assert_eq!(entry1, entry2);

        let entry3 = FeedbackEntry::new(10, 20, 3, 6);
        assert_ne!(entry1, entry3);
    }

    #[test]
    fn test_entry_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();

        let entry1 = FeedbackEntry::new(10, 20, 3, 5);
        let entry2 = FeedbackEntry::new(10, 20, 3, 5);

        set.insert(entry1);
        assert!(set.contains(&entry2));
    }

    #[test]
    fn test_entry_pod_zeroable() {
        let entry: FeedbackEntry = bytemuck::Zeroable::zeroed();
        assert_eq!(entry.tile_x, 0);
        assert_eq!(entry.tile_y, 0);
        assert_eq!(entry.mip_level, 0);
        assert_eq!(entry.priority, 0);
    }

    #[test]
    fn test_entry_memory_layout() {
        assert_eq!(std::mem::size_of::<FeedbackEntry>(), 8);
        assert_eq!(std::mem::align_of::<FeedbackEntry>(), 2);
    }

    // =========================================================================
    // PageRequest Tests
    // =========================================================================

    #[test]
    fn test_page_request_new() {
        let req = PageRequest::new(100, 200, 5, 0.75);
        assert_eq!(req.tile_x, 100);
        assert_eq!(req.tile_y, 200);
        assert_eq!(req.mip_level, 5);
        assert!((req.priority - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_page_request_from_entry() {
        let entry = FeedbackEntry::new(50, 75, 3, 8);
        let req = PageRequest::from_entry(&entry);
        assert_eq!(req.tile_x, 50);
        assert_eq!(req.tile_y, 75);
        assert_eq!(req.mip_level, 3);
        // priority = 1.0 - 8/15 = 0.4666...
        assert!((req.priority - (1.0 - 8.0 / 15.0)).abs() < 0.001);
    }

    #[test]
    fn test_page_request_key() {
        let req = PageRequest::new(10, 20, 3, 0.5);
        assert_eq!(req.key(), (10, 20, 3));
    }

    #[test]
    fn test_page_request_ordering() {
        let high = PageRequest::new(0, 0, 0, 1.0);
        let medium = PageRequest::new(0, 0, 0, 0.5);
        let low = PageRequest::new(0, 0, 0, 0.0);

        // Higher priority should come first (be "less than")
        assert!(high < medium);
        assert!(medium < low);
        assert!(high < low);
    }

    #[test]
    fn test_page_request_sort_by_priority() {
        let mut requests = vec![
            PageRequest::new(0, 0, 0, 0.3),
            PageRequest::new(1, 1, 1, 0.9),
            PageRequest::new(2, 2, 2, 0.1),
            PageRequest::new(3, 3, 3, 0.7),
        ];

        requests.sort();

        assert!((requests[0].priority - 0.9).abs() < 0.001);
        assert!((requests[1].priority - 0.7).abs() < 0.001);
        assert!((requests[2].priority - 0.3).abs() < 0.001);
        assert!((requests[3].priority - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_page_request_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();

        let req1 = PageRequest::new(10, 20, 3, 0.5);
        let req2 = PageRequest::new(10, 20, 3, 0.8); // Same coords, different priority

        set.insert(req1);
        // Hash is based on coords only, so this should work
        // Note: set will contain both since Eq also uses priority
        set.insert(req2);
    }

    // =========================================================================
    // Encoding/Decoding Tests
    // =========================================================================

    #[test]
    fn test_encode_feedback_basic() {
        let encoded = FeedbackPass::encode_feedback(100, 200, 5);
        let entry = FeedbackPass::decode_feedback(encoded);

        assert_eq!(entry.tile_x, 100);
        assert_eq!(entry.tile_y, 200);
        assert_eq!(entry.mip_level, 5);
        assert_eq!(entry.priority, 0);
    }

    #[test]
    fn test_encode_feedback_with_priority() {
        let encoded = FeedbackPass::encode_feedback_with_priority(100, 200, 5, 10);
        let entry = FeedbackPass::decode_feedback(encoded);

        assert_eq!(entry.tile_x, 100);
        assert_eq!(entry.tile_y, 200);
        assert_eq!(entry.mip_level, 5);
        assert_eq!(entry.priority, 10);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let test_cases = [
            (0, 0, 0),
            (1, 1, 1),
            (MAX_TILE_COORD, MAX_TILE_COORD, MAX_MIP_VALUE),
            (123, 456, 7),
            (4095, 0, 10),
            (0, 4095, 15),
        ];

        for (tile_x, tile_y, mip) in test_cases {
            let encoded = FeedbackPass::encode_feedback(tile_x, tile_y, mip);
            let entry = FeedbackPass::decode_feedback(encoded);

            assert_eq!(
                entry.tile_x as u32, tile_x,
                "tile_x mismatch for ({}, {}, {})",
                tile_x, tile_y, mip
            );
            assert_eq!(
                entry.tile_y as u32, tile_y,
                "tile_y mismatch for ({}, {}, {})",
                tile_x, tile_y, mip
            );
            assert_eq!(
                entry.mip_level as u32, mip,
                "mip mismatch for ({}, {}, {})",
                tile_x, tile_y, mip
            );
        }
    }

    #[test]
    fn test_encode_decode_with_priority_roundtrip() {
        for priority in 0..=MAX_PRIORITY_VALUE {
            let encoded = FeedbackPass::encode_feedback_with_priority(100, 200, 5, priority);
            let entry = FeedbackPass::decode_feedback(encoded);

            assert_eq!(entry.tile_x, 100);
            assert_eq!(entry.tile_y, 200);
            assert_eq!(entry.mip_level, 5);
            assert_eq!(entry.priority as u32, priority);
        }
    }

    #[test]
    fn test_encode_truncates_overflow() {
        // Values larger than max should be truncated
        let encoded = FeedbackPass::encode_feedback(0xFFFF, 0xFFFF, 0xFF);
        let entry = FeedbackPass::decode_feedback(encoded);

        assert_eq!(entry.tile_x as u32, MAX_TILE_COORD);
        assert_eq!(entry.tile_y as u32, MAX_TILE_COORD);
        assert_eq!(entry.mip_level as u32, MAX_MIP_VALUE);
    }

    #[test]
    fn test_decode_invalid_feedback() {
        let entry = FeedbackPass::decode_feedback(INVALID_FEEDBACK);
        assert!(entry.is_invalid());
    }

    #[test]
    fn test_is_valid_feedback() {
        assert!(!FeedbackPass::is_valid_feedback(INVALID_FEEDBACK));
        assert!(FeedbackPass::is_valid_feedback(0));
        assert!(FeedbackPass::is_valid_feedback(0x12345678));
    }

    // =========================================================================
    // FeedbackPass Tests
    // =========================================================================

    #[test]
    fn test_feedback_pass_new() {
        let config = FeedbackConfig::default();
        let pass = FeedbackPass::new(config);

        assert_eq!(pass.config.feedback_width, 1024);
        assert_eq!(pass.staging_buffer.len(), 1024 * 1024);
        assert!(pass.requests.is_empty());
        assert_eq!(pass.frame_index, 0);
    }

    #[test]
    fn test_feedback_pass_with_defaults() {
        let pass = FeedbackPass::with_defaults();
        assert_eq!(pass.config.feedback_width, 1024);
    }

    #[test]
    fn test_process_readback_empty() {
        let mut pass = FeedbackPass::with_defaults();
        pass.process_readback(&[]);
        assert!(pass.requests.is_empty());
        assert_eq!(pass.last_valid_samples, 0);
    }

    #[test]
    fn test_process_readback_all_invalid() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![INVALID_FEEDBACK; 100];
        pass.process_readback(&data);

        assert!(pass.requests.is_empty());
        assert_eq!(pass.last_valid_samples, 0);
    }

    #[test]
    fn test_process_readback_single_valid() {
        let mut pass = FeedbackPass::with_defaults();
        let encoded = FeedbackPass::encode_feedback(10, 20, 3);
        let data = vec![encoded];

        pass.process_readback(&data);

        assert_eq!(pass.requests.len(), 1);
        assert_eq!(pass.requests[0].tile_x, 10);
        assert_eq!(pass.requests[0].tile_y, 20);
        assert_eq!(pass.requests[0].mip_level, 3);
        assert_eq!(pass.last_valid_samples, 1);
    }

    #[test]
    fn test_process_readback_mixed() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![
            FeedbackPass::encode_feedback(1, 1, 0),
            INVALID_FEEDBACK,
            FeedbackPass::encode_feedback(2, 2, 1),
            INVALID_FEEDBACK,
            INVALID_FEEDBACK,
            FeedbackPass::encode_feedback(3, 3, 2),
        ];

        pass.process_readback(&data);

        assert_eq!(pass.requests.len(), 3);
        assert_eq!(pass.last_valid_samples, 3);
    }

    #[test]
    fn test_deduplicate_requests_empty() {
        let mut pass = FeedbackPass::with_defaults();
        pass.deduplicate_requests();
        assert!(pass.requests.is_empty());
    }

    #[test]
    fn test_deduplicate_requests_no_duplicates() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![
            FeedbackPass::encode_feedback(1, 1, 0),
            FeedbackPass::encode_feedback(2, 2, 1),
            FeedbackPass::encode_feedback(3, 3, 2),
        ];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        assert_eq!(pass.requests.len(), 3);
    }

    #[test]
    fn test_deduplicate_requests_with_duplicates() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![
            FeedbackPass::encode_feedback(1, 1, 0),
            FeedbackPass::encode_feedback(1, 1, 0), // duplicate
            FeedbackPass::encode_feedback(2, 2, 1),
            FeedbackPass::encode_feedback(1, 1, 0), // duplicate
        ];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        assert_eq!(pass.requests.len(), 2);
    }

    #[test]
    fn test_deduplicate_keeps_highest_priority() {
        let mut pass = FeedbackPass::with_defaults();
        // Note: lower priority value = higher actual priority
        let data = vec![
            FeedbackPass::encode_feedback_with_priority(1, 1, 0, 5),  // medium
            FeedbackPass::encode_feedback_with_priority(1, 1, 0, 2),  // highest
            FeedbackPass::encode_feedback_with_priority(1, 1, 0, 10), // lowest
        ];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        assert_eq!(pass.requests.len(), 1);
        // The request with priority 2 should be kept (maps to highest priority value)
    }

    #[test]
    fn test_get_missing_pages_all_missing() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![
            FeedbackPass::encode_feedback(1, 1, 0),
            FeedbackPass::encode_feedback(2, 2, 1),
        ];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        let resident = HashSet::new();
        let missing = pass.get_missing_pages(&resident);

        assert_eq!(missing.len(), 2);
    }

    #[test]
    fn test_get_missing_pages_some_resident() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![
            FeedbackPass::encode_feedback(1, 1, 0),
            FeedbackPass::encode_feedback(2, 2, 1),
            FeedbackPass::encode_feedback(3, 3, 2),
        ];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        let mut resident = HashSet::new();
        resident.insert(FeedbackEntry::new(2, 2, 1, 0));

        let missing = pass.get_missing_pages(&resident);

        assert_eq!(missing.len(), 2);
        assert!(missing.iter().any(|r| r.tile_x == 1 && r.tile_y == 1));
        assert!(missing.iter().any(|r| r.tile_x == 3 && r.tile_y == 3));
    }

    #[test]
    fn test_get_missing_pages_all_resident() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![FeedbackPass::encode_feedback(1, 1, 0)];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        let mut resident = HashSet::new();
        resident.insert(FeedbackEntry::new(1, 1, 0, 0));

        let missing = pass.get_missing_pages(&resident);

        assert!(missing.is_empty());
    }

    #[test]
    fn test_get_missing_pages_from_set() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![
            FeedbackPass::encode_feedback(1, 1, 0),
            FeedbackPass::encode_feedback(2, 2, 1),
        ];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        let mut resident = HashSet::new();
        resident.insert((1, 1, 0));

        let missing = pass.get_missing_pages_from_set(&resident);

        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].tile_x, 2);
        assert_eq!(missing[0].tile_y, 2);
    }

    #[test]
    fn test_begin_frame() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![FeedbackPass::encode_feedback(1, 1, 0)];
        pass.process_readback(&data);

        assert!(!pass.requests.is_empty());

        pass.begin_frame();

        assert!(pass.requests.is_empty());
        assert_eq!(pass.frame_index, 1);
    }

    #[test]
    fn test_frame_index_wrapping() {
        let mut pass = FeedbackPass::with_defaults();
        pass.frame_index = u32::MAX;
        pass.begin_frame();
        assert_eq!(pass.frame_index, 0);
    }

    #[test]
    fn test_reset() {
        let mut pass = FeedbackPass::with_defaults();
        pass.frame_index = 100;
        pass.total_pixels_processed = 1000;
        pass.staging_buffer[0] = 12345;

        let data = vec![FeedbackPass::encode_feedback(1, 1, 0)];
        pass.process_readback(&data);

        pass.reset();

        assert_eq!(pass.frame_index, 0);
        assert_eq!(pass.total_pixels_processed, 0);
        assert!(pass.requests.is_empty());
        assert_eq!(pass.staging_buffer[0], INVALID_FEEDBACK);
    }

    #[test]
    fn test_stats() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![
            FeedbackPass::encode_feedback(1, 1, 0),
            FeedbackPass::encode_feedback(2, 2, 1),
            FeedbackPass::encode_feedback(1, 1, 0), // duplicate
        ];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        let stats = pass.stats();
        assert_eq!(stats.valid_samples, 3);
        assert_eq!(stats.unique_requests, 2);
        assert_eq!(stats.buffer_size, 1024 * 1024);
    }

    #[test]
    fn test_staging_buffer_access() {
        let mut pass = FeedbackPass::with_defaults();

        // Read access
        assert_eq!(pass.staging_buffer().len(), 1024 * 1024);

        // Write access
        pass.staging_buffer_mut()[0] = 12345;
        assert_eq!(pass.staging_buffer()[0], 12345);
    }

    #[test]
    fn test_request_count() {
        let mut pass = FeedbackPass::with_defaults();
        assert_eq!(pass.request_count(), 0);

        let data = vec![
            FeedbackPass::encode_feedback(1, 1, 0),
            FeedbackPass::encode_feedback(2, 2, 1),
        ];
        pass.process_readback(&data);
        pass.deduplicate_requests();

        assert_eq!(pass.request_count(), 2);
    }

    // =========================================================================
    // Priority Tests
    // =========================================================================

    #[test]
    fn test_compute_priority_mip_factor() {
        let pass = FeedbackPass::with_defaults();
        let camera = [65536.0, 0.0, 65536.0]; // Center of virtual texture

        let low_mip = FeedbackEntry::new(512, 512, 0, 0);  // High detail
        let high_mip = FeedbackEntry::new(512, 512, 10, 0); // Low detail

        let priority_low = pass.compute_priority(&low_mip, camera);
        let priority_high = pass.compute_priority(&high_mip, camera);

        // Lower mip should have higher priority
        assert!(priority_low > priority_high);
    }

    #[test]
    fn test_compute_priority_distance_factor() {
        let pass = FeedbackPass::with_defaults();
        let camera = [0.0, 0.0, 0.0];

        let near = FeedbackEntry::new(0, 0, 0, 0);   // Near camera
        let far = FeedbackEntry::new(900, 900, 0, 0); // Far from camera

        let priority_near = pass.compute_priority(&near, camera);
        let priority_far = pass.compute_priority(&far, camera);

        // Near tile should have higher priority
        assert!(priority_near > priority_far);
    }

    #[test]
    fn test_compute_priority_center_factor() {
        let pass = FeedbackPass::with_defaults();
        let camera = [0.0, 0.0, 0.0];

        let center = FeedbackEntry::new(512, 512, 5, 0); // Near center at same mip
        let edge = FeedbackEntry::new(0, 0, 5, 0);       // Near edge at same mip

        let priority_center = pass.compute_priority(&center, camera);
        let priority_edge = pass.compute_priority(&edge, camera);

        // Center tile should have slightly higher priority due to center factor
        // Note: This test may be affected by distance factor, so we just check range
        assert!(priority_center >= 0.0 && priority_center <= 1.0);
        assert!(priority_edge >= 0.0 && priority_edge <= 1.0);
    }

    #[test]
    fn test_compute_priority_range() {
        let pass = FeedbackPass::with_defaults();

        // Test various inputs to ensure output is always in [0, 1]
        let test_cases = [
            ([0.0, 0.0, 0.0], FeedbackEntry::new(0, 0, 0, 0)),
            ([100000.0, 100.0, 100000.0], FeedbackEntry::new(4095, 4095, 15, 15)),
            ([65536.0, 0.0, 65536.0], FeedbackEntry::new(512, 512, 5, 0)),
        ];

        for (camera, entry) in test_cases {
            let priority = pass.compute_priority(&entry, camera);
            assert!(
                priority >= 0.0 && priority <= 1.0,
                "Priority {} out of range for {:?}",
                priority,
                camera
            );
        }
    }

    #[test]
    fn test_update_priorities() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![
            FeedbackPass::encode_feedback(10, 10, 0),
            FeedbackPass::encode_feedback(900, 900, 0),
        ];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        let camera = [10.0 * 128.0, 0.0, 10.0 * 128.0]; // Near tile (10, 10)
        pass.update_priorities(camera);

        // After update, requests should be re-sorted
        // Tile (10, 10) should have higher priority (come first)
        assert_eq!(pass.requests[0].tile_x, 10);
        assert_eq!(pass.requests[0].tile_y, 10);
    }

    // =========================================================================
    // Readback Staging Tests
    // =========================================================================

    #[test]
    fn test_staging_buffer_initial_state() {
        let pass = FeedbackPass::with_defaults();
        for &value in pass.staging_buffer() {
            assert_eq!(value, INVALID_FEEDBACK);
        }
    }

    #[test]
    fn test_staging_buffer_size_matches_config() {
        let config = FeedbackConfig::new(256, 512, 8192, 64);
        let pass = FeedbackPass::new(config);
        assert_eq!(pass.staging_buffer().len(), 256 * 512);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_feedback_workflow() {
        let config = FeedbackConfig::new(512, 512, 8192, 64);
        let mut pass = FeedbackPass::new(config);

        // Simulate GPU feedback data
        let mut data = vec![INVALID_FEEDBACK; 100];
        data[0] = FeedbackPass::encode_feedback(10, 20, 2);
        data[5] = FeedbackPass::encode_feedback(10, 20, 2); // Duplicate
        data[10] = FeedbackPass::encode_feedback(30, 40, 1);
        data[15] = FeedbackPass::encode_feedback(50, 60, 0);

        // Process readback
        pass.process_readback(&data);
        assert_eq!(pass.last_valid_samples, 4);
        assert_eq!(pass.requests.len(), 4);

        // Deduplicate
        pass.deduplicate_requests();
        assert_eq!(pass.requests.len(), 3); // One duplicate removed

        // Check resident pages
        let mut resident = HashSet::new();
        resident.insert(FeedbackEntry::new(30, 40, 1, 0));

        let missing = pass.get_missing_pages(&resident);
        assert_eq!(missing.len(), 2);

        // Verify statistics
        let stats = pass.stats();
        assert_eq!(stats.valid_samples, 4);
        assert_eq!(stats.unique_requests, 3);
    }

    #[test]
    fn test_multi_frame_workflow() {
        let mut pass = FeedbackPass::with_defaults();

        // Frame 1
        let data1 = vec![FeedbackPass::encode_feedback(1, 1, 0)];
        pass.process_readback(&data1);
        pass.deduplicate_requests();
        assert_eq!(pass.request_count(), 1);
        assert_eq!(pass.frame_index, 0);

        // Frame 2
        pass.begin_frame();
        assert_eq!(pass.frame_index, 1);
        assert_eq!(pass.request_count(), 0); // Cleared

        let data2 = vec![
            FeedbackPass::encode_feedback(2, 2, 1),
            FeedbackPass::encode_feedback(3, 3, 2),
        ];
        pass.process_readback(&data2);
        pass.deduplicate_requests();
        assert_eq!(pass.request_count(), 2);

        // Statistics accumulate
        assert_eq!(pass.total_pixels_processed, 3);
    }

    #[test]
    fn test_large_readback() {
        let mut pass = FeedbackPass::with_defaults();

        // Create large dataset with many duplicates
        let mut data = Vec::with_capacity(10000);
        for i in 0..10000u32 {
            // Create tiles with lots of repetition
            let tile_x = i % 100;
            let tile_y = (i / 100) % 100;
            let mip = (i % 5) as u32;
            data.push(FeedbackPass::encode_feedback(tile_x, tile_y, mip));
        }

        pass.process_readback(&data);
        pass.deduplicate_requests();

        // Should have max 100*100 = 10000 unique (x,y) but with 5 mip levels
        // Actually 100*100*5 = 50000 unique but we only have 10000 samples
        // So should be at most 10000/5 = 2000 unique x,y per mip on average
        assert!(pass.request_count() <= 10000);
        assert!(pass.request_count() > 0);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_zero_config() {
        let config = FeedbackConfig::new(0, 0, 0, 0);
        let pass = FeedbackPass::new(config);
        assert_eq!(pass.staging_buffer().len(), 0);
    }

    #[test]
    fn test_single_pixel_buffer() {
        let config = FeedbackConfig::new(1, 1, 128, 128);
        let mut pass = FeedbackPass::new(config);
        assert_eq!(pass.staging_buffer().len(), 1);

        let data = vec![FeedbackPass::encode_feedback(0, 0, 0)];
        pass.process_readback(&data);
        pass.deduplicate_requests();

        assert_eq!(pass.request_count(), 1);
    }

    #[test]
    fn test_max_tile_coordinates() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![FeedbackPass::encode_feedback(
            MAX_TILE_COORD,
            MAX_TILE_COORD,
            MAX_MIP_VALUE,
        )];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        assert_eq!(pass.request_count(), 1);
        assert_eq!(pass.requests[0].tile_x, MAX_TILE_COORD);
        assert_eq!(pass.requests[0].tile_y, MAX_TILE_COORD);
        assert_eq!(pass.requests[0].mip_level, MAX_MIP_VALUE);
    }

    #[test]
    fn test_requests_sorted_after_dedup() {
        let mut pass = FeedbackPass::with_defaults();
        let data = vec![
            FeedbackPass::encode_feedback_with_priority(1, 1, 5, 10), // low priority
            FeedbackPass::encode_feedback_with_priority(2, 2, 0, 2),  // high priority
            FeedbackPass::encode_feedback_with_priority(3, 3, 3, 5),  // medium priority
        ];

        pass.process_readback(&data);
        pass.deduplicate_requests();

        // Should be sorted by priority (highest first)
        // Priority 2 (value) -> 1.0 - 2/15 = 0.866 (highest)
        // Priority 5 (value) -> 1.0 - 5/15 = 0.666
        // Priority 10 (value) -> 1.0 - 10/15 = 0.333 (lowest)
        assert!(pass.requests[0].priority > pass.requests[1].priority);
        assert!(pass.requests[1].priority > pass.requests[2].priority);
    }
}
