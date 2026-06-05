//! Virtual Texturing Page Extraction (T-AS-2.6)
//!
//! Provides virtual texturing page extraction for streaming large textures
//! efficiently. Supports page-aligned tiling, mip chain organization, page
//! table generation, and content-hash deduplication.
//!
//! # Features
//!
//! - Extract 128x128 pixel pages from source textures
//! - Support all mip levels with boundary handling
//! - Page table initialization data for GPU residency
//! - Morton curve ordering for cache-efficient PAK archives
//! - Content hash deduplication via ContentStore
//! - Cook-time operation with configurable page size
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::virtual_texture::{
//!     VirtualTextureConfig, extract_pages, generate_page_table, pack_for_archive,
//! };
//! use renderer_backend::asset::mipmap::{MipmapConfig, generate_mipmaps};
//!
//! // Generate mips first
//! let mips = generate_mipmaps(&texture_asset, &MipmapConfig::default())?;
//!
//! // Extract virtual texture pages
//! let config = VirtualTextureConfig::default();
//! let vt_data = extract_pages(&texture_data, &mips, &config);
//!
//! // Generate GPU page table
//! let page_table = generate_page_table(&vt_data.pages, &config);
//!
//! // Pack for archive with Morton ordering
//! let archive_data = pack_for_archive(&vt_data);
//! ```

use std::collections::HashMap;
use std::fmt;

use super::content_hash::ContentHash;
use super::cubemap::TextureData;
use super::mipmap::MipLevel;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default page size (128x128 pixels).
pub const DEFAULT_PAGE_SIZE: u32 = 128;

/// Minimum supported page size.
pub const MIN_PAGE_SIZE: u32 = 16;

/// Maximum supported page size.
pub const MAX_PAGE_SIZE: u32 = 512;

/// Invalid physical page index (used in page table for non-resident pages).
pub const INVALID_PHYSICAL_PAGE: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors during virtual texture operations.
#[derive(Debug, Clone)]
pub enum VirtualTextureError {
    /// Invalid page size (must be power of 2 and within limits).
    InvalidPageSize(u32),
    /// Invalid texture dimensions.
    InvalidDimensions { width: u32, height: u32 },
    /// Invalid mip level data.
    InvalidMipData { level: u32, expected: usize, actual: usize },
    /// No mip levels provided.
    NoMipLevels,
    /// Page extraction failed.
    PageExtractionFailed(String),
}

impl fmt::Display for VirtualTextureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VirtualTextureError::InvalidPageSize(size) => {
                write!(f, "invalid page size {}: must be power of 2, {}-{}", size, MIN_PAGE_SIZE, MAX_PAGE_SIZE)
            }
            VirtualTextureError::InvalidDimensions { width, height } => {
                write!(f, "invalid texture dimensions: {}x{}", width, height)
            }
            VirtualTextureError::InvalidMipData { level, expected, actual } => {
                write!(f, "invalid mip level {} data: expected {} bytes, got {}", level, expected, actual)
            }
            VirtualTextureError::NoMipLevels => {
                write!(f, "no mip levels provided")
            }
            VirtualTextureError::PageExtractionFailed(msg) => {
                write!(f, "page extraction failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for VirtualTextureError {}

// ---------------------------------------------------------------------------
// Virtual Texture Page
// ---------------------------------------------------------------------------

/// A single page extracted from a virtual texture.
#[derive(Debug, Clone)]
pub struct VirtualTexturePage {
    /// Mip level this page belongs to.
    pub mip_level: u32,
    /// Page X coordinate within the mip level.
    pub page_x: u32,
    /// Page Y coordinate within the mip level.
    pub page_y: u32,
    /// Content hash of this page's pixel data.
    pub content_hash: ContentHash,
    /// Offset in the packed archive data.
    pub data_offset: u64,
    /// Size of page data in bytes.
    pub data_size: u32,
    /// Raw page pixel data (before archive packing).
    pub data: Vec<u8>,
}

impl VirtualTexturePage {
    /// Create a new virtual texture page.
    pub fn new(
        mip_level: u32,
        page_x: u32,
        page_y: u32,
        data: Vec<u8>,
    ) -> Self {
        let content_hash = ContentHash::from_bytes(&data);
        let data_size = data.len() as u32;
        Self {
            mip_level,
            page_x,
            page_y,
            content_hash,
            data_offset: 0,
            data_size,
            data,
        }
    }

    /// Get the Morton index for this page (for spatial locality ordering).
    pub fn morton_index(&self) -> u32 {
        compute_morton_index(self.page_x, self.page_y)
    }
}

// ---------------------------------------------------------------------------
// Page Table Entry
// ---------------------------------------------------------------------------

/// An entry in the GPU page table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageTableEntry {
    /// Physical page index in the texture cache.
    pub physical_page: u32,
    /// Whether this page is currently resident in GPU memory.
    pub resident: bool,
}

impl PageTableEntry {
    /// Create a resident page table entry.
    pub fn resident(physical_page: u32) -> Self {
        Self {
            physical_page,
            resident: true,
        }
    }

    /// Create a non-resident page table entry.
    pub fn non_resident() -> Self {
        Self {
            physical_page: INVALID_PHYSICAL_PAGE,
            resident: false,
        }
    }

    /// Check if this entry is valid (resident with valid page).
    pub fn is_valid(&self) -> bool {
        self.resident && self.physical_page != INVALID_PHYSICAL_PAGE
    }
}

impl Default for PageTableEntry {
    fn default() -> Self {
        Self::non_resident()
    }
}

// ---------------------------------------------------------------------------
// Virtual Texture Configuration
// ---------------------------------------------------------------------------

/// Configuration for virtual texture page extraction.
#[derive(Debug, Clone)]
pub struct VirtualTextureConfig {
    /// Page size in pixels (default: 128).
    pub page_size: u32,
    /// Maximum mip levels to process (0 = all).
    pub max_mip_levels: u32,
    /// Use Morton curve ordering for spatial locality.
    pub use_morton_order: bool,
    /// Enable content hash deduplication.
    pub deduplicate: bool,
    /// Border size for bilinear filtering (typically 1-4 pixels).
    pub border_size: u32,
}

impl Default for VirtualTextureConfig {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            max_mip_levels: 0,
            use_morton_order: true,
            deduplicate: true,
            border_size: 4,
        }
    }
}

impl VirtualTextureConfig {
    /// Create configuration with custom page size.
    pub fn with_page_size(page_size: u32) -> Self {
        Self {
            page_size,
            ..Default::default()
        }
    }

    /// Create configuration without deduplication.
    pub fn without_deduplication() -> Self {
        Self {
            deduplicate: false,
            ..Default::default()
        }
    }

    /// Create configuration without Morton ordering.
    pub fn without_morton_order() -> Self {
        Self {
            use_morton_order: false,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), VirtualTextureError> {
        if !self.page_size.is_power_of_two() {
            return Err(VirtualTextureError::InvalidPageSize(self.page_size));
        }
        if self.page_size < MIN_PAGE_SIZE || self.page_size > MAX_PAGE_SIZE {
            return Err(VirtualTextureError::InvalidPageSize(self.page_size));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Virtual Texture Data
// ---------------------------------------------------------------------------

/// Complete virtual texture data after page extraction.
#[derive(Debug, Clone)]
pub struct VirtualTextureData {
    /// All extracted pages.
    pub pages: Vec<VirtualTexturePage>,
    /// Page table entries for GPU.
    pub page_table: Vec<PageTableEntry>,
    /// Width in pages at mip level 0.
    pub width_pages: u32,
    /// Height in pages at mip level 0.
    pub height_pages: u32,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Number of unique pages (after deduplication).
    pub unique_pages: usize,
    /// Total number of pages (before deduplication).
    pub total_pages: usize,
    /// Page size used.
    pub page_size: u32,
    /// Original texture width.
    pub texture_width: u32,
    /// Original texture height.
    pub texture_height: u32,
}

impl VirtualTextureData {
    /// Calculate the deduplication ratio.
    pub fn dedup_ratio(&self) -> f32 {
        if self.total_pages == 0 {
            return 1.0;
        }
        self.unique_pages as f32 / self.total_pages as f32
    }

    /// Calculate total data size in bytes.
    pub fn total_data_size(&self) -> usize {
        self.pages.iter().map(|p| p.data.len()).sum()
    }

    /// Get page at specific mip level and coordinates.
    pub fn get_page(&self, mip_level: u32, page_x: u32, page_y: u32) -> Option<&VirtualTexturePage> {
        self.pages.iter().find(|p| {
            p.mip_level == mip_level && p.page_x == page_x && p.page_y == page_y
        })
    }

    /// Get pages for a specific mip level.
    pub fn pages_at_mip(&self, mip_level: u32) -> Vec<&VirtualTexturePage> {
        self.pages.iter().filter(|p| p.mip_level == mip_level).collect()
    }
}

// ---------------------------------------------------------------------------
// Mip Level Page Info
// ---------------------------------------------------------------------------

/// Information about pages at a specific mip level.
#[derive(Debug, Clone)]
pub struct MipLevelPageInfo {
    /// Mip level index.
    pub level: u32,
    /// Width in pages.
    pub width_pages: u32,
    /// Height in pages.
    pub height_pages: u32,
    /// Total pages at this level.
    pub total_pages: u32,
    /// Width in pixels.
    pub width_pixels: u32,
    /// Height in pixels.
    pub height_pixels: u32,
}

// ---------------------------------------------------------------------------
// Morton Index Computation
// ---------------------------------------------------------------------------

/// Compute Morton (Z-order) index from 2D coordinates.
///
/// Morton indexing provides spatial locality, improving cache efficiency
/// when streaming pages.
pub fn compute_morton_index(x: u32, y: u32) -> u32 {
    let mut result = 0u32;
    for i in 0..16 {
        result |= ((x >> i) & 1) << (2 * i);
        result |= ((y >> i) & 1) << (2 * i + 1);
    }
    result
}

/// Decode Morton index back to 2D coordinates.
pub fn decode_morton_index(morton: u32) -> (u32, u32) {
    let mut x = 0u32;
    let mut y = 0u32;
    for i in 0..16 {
        x |= ((morton >> (2 * i)) & 1) << i;
        y |= ((morton >> (2 * i + 1)) & 1) << i;
    }
    (x, y)
}

// ---------------------------------------------------------------------------
// Page Extraction
// ---------------------------------------------------------------------------

/// Extract virtual texture pages from a texture and its mip chain.
pub fn extract_pages(
    texture: &TextureData,
    mips: &[MipLevel],
    config: &VirtualTextureConfig,
) -> Result<VirtualTextureData, VirtualTextureError> {
    config.validate()?;

    if mips.is_empty() {
        return Err(VirtualTextureError::NoMipLevels);
    }

    if texture.width == 0 || texture.height == 0 {
        return Err(VirtualTextureError::InvalidDimensions {
            width: texture.width,
            height: texture.height,
        });
    }

    let page_size = config.page_size;
    let bytes_per_pixel = texture.format.bytes_per_pixel();

    // Calculate pages at mip level 0
    let width_pages = (texture.width + page_size - 1) / page_size;
    let height_pages = (texture.height + page_size - 1) / page_size;

    // Determine mip levels to process
    let max_mips = if config.max_mip_levels > 0 {
        config.max_mip_levels.min(mips.len() as u32)
    } else {
        mips.len() as u32
    };

    let mut all_pages: Vec<VirtualTexturePage> = Vec::new();
    let mut hash_to_page: HashMap<ContentHash, usize> = HashMap::new();
    let mut unique_count = 0usize;
    let mut total_count = 0usize;

    // Extract pages from each mip level
    for (level_idx, mip) in mips.iter().take(max_mips as usize).enumerate() {
        let level = level_idx as u32;

        // Calculate page dimensions at this mip level
        let mip_width_pages = (mip.width + page_size - 1) / page_size;
        let mip_height_pages = (mip.height + page_size - 1) / page_size;

        for py in 0..mip_height_pages {
            for px in 0..mip_width_pages {
                // Extract page data
                let page_data = extract_single_page(
                    mip,
                    px,
                    py,
                    page_size,
                    bytes_per_pixel,
                );

                total_count += 1;

                // Create page
                let mut page = VirtualTexturePage::new(level, px, py, page_data);

                // Handle deduplication
                if config.deduplicate {
                    if let Some(&existing_idx) = hash_to_page.get(&page.content_hash) {
                        // This page is a duplicate - still add it but mark the relationship
                        // In a real implementation, we'd store a reference instead
                        page.data_offset = all_pages[existing_idx].data_offset;
                    } else {
                        hash_to_page.insert(page.content_hash, all_pages.len());
                        unique_count += 1;
                    }
                } else {
                    unique_count += 1;
                }

                all_pages.push(page);
            }
        }
    }

    // Sort by Morton order if configured
    if config.use_morton_order {
        all_pages.sort_by(|a, b| {
            // Primary sort by mip level, secondary by Morton index
            match a.mip_level.cmp(&b.mip_level) {
                std::cmp::Ordering::Equal => a.morton_index().cmp(&b.morton_index()),
                other => other,
            }
        });
    }

    // Assign data offsets
    let mut current_offset = 0u64;
    if config.deduplicate {
        let mut assigned: HashMap<ContentHash, u64> = HashMap::new();
        for page in &mut all_pages {
            if let Some(&offset) = assigned.get(&page.content_hash) {
                page.data_offset = offset;
            } else {
                page.data_offset = current_offset;
                assigned.insert(page.content_hash, current_offset);
                current_offset += page.data_size as u64;
            }
        }
    } else {
        for page in &mut all_pages {
            page.data_offset = current_offset;
            current_offset += page.data_size as u64;
        }
    }

    // Generate page table
    let page_table = generate_page_table(&all_pages, config);

    Ok(VirtualTextureData {
        pages: all_pages,
        page_table,
        width_pages,
        height_pages,
        mip_levels: max_mips,
        unique_pages: unique_count,
        total_pages: total_count,
        page_size,
        texture_width: texture.width,
        texture_height: texture.height,
    })
}

/// Extract a single page from a mip level.
fn extract_single_page(
    mip: &MipLevel,
    page_x: u32,
    page_y: u32,
    page_size: u32,
    bytes_per_pixel: usize,
) -> Vec<u8> {
    let page_bytes = (page_size * page_size) as usize * bytes_per_pixel;
    let mut page_data = vec![0u8; page_bytes];

    let start_x = page_x * page_size;
    let start_y = page_y * page_size;

    for py in 0..page_size {
        for px in 0..page_size {
            let src_x = start_x + px;
            let src_y = start_y + py;

            // Handle boundary cases - clamp to edge
            let src_x = src_x.min(mip.width.saturating_sub(1));
            let src_y = src_y.min(mip.height.saturating_sub(1));

            let pixel = mip.get_pixel(src_x, src_y);

            let dst_idx = (py * page_size + px) as usize * bytes_per_pixel;
            match bytes_per_pixel {
                1 => {
                    page_data[dst_idx] = pixel[0];
                }
                2 => {
                    page_data[dst_idx] = pixel[0];
                    page_data[dst_idx + 1] = pixel[3];
                }
                3 => {
                    page_data[dst_idx] = pixel[0];
                    page_data[dst_idx + 1] = pixel[1];
                    page_data[dst_idx + 2] = pixel[2];
                }
                4 => {
                    page_data[dst_idx] = pixel[0];
                    page_data[dst_idx + 1] = pixel[1];
                    page_data[dst_idx + 2] = pixel[2];
                    page_data[dst_idx + 3] = pixel[3];
                }
                _ => {}
            }
        }
    }

    page_data
}

// ---------------------------------------------------------------------------
// Page Table Generation
// ---------------------------------------------------------------------------

/// Generate page table initialization data for GPU.
///
/// The page table maps virtual page coordinates to physical page indices.
pub fn generate_page_table(
    pages: &[VirtualTexturePage],
    config: &VirtualTextureConfig,
) -> Vec<PageTableEntry> {
    if pages.is_empty() {
        return Vec::new();
    }

    // Find max dimensions
    let max_x = pages.iter().map(|p| p.page_x).max().unwrap_or(0);
    let max_y = pages.iter().map(|p| p.page_y).max().unwrap_or(0);
    let max_mip = pages.iter().map(|p| p.mip_level).max().unwrap_or(0);

    // Calculate total table size (all mip levels)
    let mut total_entries = 0usize;
    let mut level_offsets = Vec::with_capacity((max_mip + 1) as usize);

    for level in 0..=max_mip {
        level_offsets.push(total_entries);
        let level_width = (max_x + 1) >> level;
        let level_height = (max_y + 1) >> level;
        let level_width = level_width.max(1);
        let level_height = level_height.max(1);
        total_entries += (level_width * level_height) as usize;
    }

    let mut page_table = vec![PageTableEntry::non_resident(); total_entries];

    // Populate page table
    let mut physical_page_idx = 0u32;
    let mut hash_to_physical: HashMap<ContentHash, u32> = HashMap::new();

    for page in pages {
        let level_width = ((max_x + 1) >> page.mip_level).max(1);
        let entry_idx = level_offsets[page.mip_level as usize]
            + (page.page_y * level_width + page.page_x) as usize;

        if entry_idx < page_table.len() {
            // Handle deduplication - shared pages get same physical index
            let physical_idx = if config.deduplicate {
                if let Some(&existing) = hash_to_physical.get(&page.content_hash) {
                    existing
                } else {
                    let idx = physical_page_idx;
                    hash_to_physical.insert(page.content_hash, idx);
                    physical_page_idx += 1;
                    idx
                }
            } else {
                let idx = physical_page_idx;
                physical_page_idx += 1;
                idx
            };

            page_table[entry_idx] = PageTableEntry::resident(physical_idx);
        }
    }

    page_table
}

// ---------------------------------------------------------------------------
// Archive Packing
// ---------------------------------------------------------------------------

/// Pack virtual texture data for archive storage.
///
/// Returns a byte array with pages laid out for efficient streaming.
/// Pages are ordered by Morton curve for spatial locality when enabled.
pub fn pack_for_archive(vt: &VirtualTextureData) -> Vec<u8> {
    if vt.pages.is_empty() {
        return Vec::new();
    }

    // Calculate total size needed
    let total_size: usize = vt.pages.iter().map(|p| p.data.len()).sum();
    let mut archive_data = Vec::with_capacity(total_size);

    // Track which content hashes we've already written (for deduplication)
    let mut written_hashes: HashMap<ContentHash, u64> = HashMap::new();

    for page in &vt.pages {
        // Skip duplicates - they share data via offset
        if written_hashes.contains_key(&page.content_hash) {
            continue;
        }

        written_hashes.insert(page.content_hash, archive_data.len() as u64);
        archive_data.extend_from_slice(&page.data);
    }

    archive_data
}

/// Pack with header metadata for self-describing archive.
pub fn pack_with_header(vt: &VirtualTextureData) -> Vec<u8> {
    // Header format:
    // [0..4]   magic: "VTEX"
    // [4..8]   version: u32
    // [8..12]  page_size: u32
    // [12..16] texture_width: u32
    // [16..20] texture_height: u32
    // [20..24] mip_levels: u32
    // [24..28] page_count: u32
    // [28..32] unique_pages: u32
    // [32..]   page data

    let header_size = 32usize;
    let page_data = pack_for_archive(vt);
    let mut archive = Vec::with_capacity(header_size + page_data.len());

    // Magic
    archive.extend_from_slice(b"VTEX");
    // Version
    archive.extend_from_slice(&1u32.to_le_bytes());
    // Page size
    archive.extend_from_slice(&vt.page_size.to_le_bytes());
    // Texture dimensions
    archive.extend_from_slice(&vt.texture_width.to_le_bytes());
    archive.extend_from_slice(&vt.texture_height.to_le_bytes());
    // Mip levels
    archive.extend_from_slice(&vt.mip_levels.to_le_bytes());
    // Page counts
    archive.extend_from_slice(&(vt.total_pages as u32).to_le_bytes());
    archive.extend_from_slice(&(vt.unique_pages as u32).to_le_bytes());
    // Page data
    archive.extend_from_slice(&page_data);

    archive
}

// ---------------------------------------------------------------------------
// Mip Level Information
// ---------------------------------------------------------------------------

/// Calculate page information for all mip levels.
pub fn calculate_mip_page_info(
    texture_width: u32,
    texture_height: u32,
    page_size: u32,
    max_mips: u32,
) -> Vec<MipLevelPageInfo> {
    let mut info = Vec::new();
    let mut width = texture_width;
    let mut height = texture_height;

    for level in 0..max_mips {
        let width_pages = (width + page_size - 1) / page_size;
        let height_pages = (height + page_size - 1) / page_size;

        info.push(MipLevelPageInfo {
            level,
            width_pages,
            height_pages,
            total_pages: width_pages * height_pages,
            width_pixels: width,
            height_pixels: height,
        });

        width = (width / 2).max(1);
        height = (height / 2).max(1);

        // Stop when we have 1x1 pages
        if width_pages == 1 && height_pages == 1 {
            break;
        }
    }

    info
}

// ---------------------------------------------------------------------------
// Content Store Integration
// ---------------------------------------------------------------------------

/// Page deduplication statistics.
#[derive(Debug, Clone, Default)]
pub struct DeduplicationStats {
    /// Total pages before deduplication.
    pub total_pages: usize,
    /// Unique pages after deduplication.
    pub unique_pages: usize,
    /// Bytes saved by deduplication.
    pub bytes_saved: usize,
    /// Most common duplicate count.
    pub max_duplicates: usize,
}

impl DeduplicationStats {
    /// Calculate deduplication ratio (0.0 = no duplicates, 1.0 = all duplicates).
    pub fn dedup_ratio(&self) -> f32 {
        if self.total_pages == 0 {
            return 0.0;
        }
        1.0 - (self.unique_pages as f32 / self.total_pages as f32)
    }
}

/// Calculate deduplication statistics for a virtual texture.
pub fn calculate_dedup_stats(vt: &VirtualTextureData) -> DeduplicationStats {
    let mut hash_counts: HashMap<ContentHash, usize> = HashMap::new();

    for page in &vt.pages {
        *hash_counts.entry(page.content_hash).or_insert(0) += 1;
    }

    let unique_pages = hash_counts.len();
    let max_duplicates = hash_counts.values().copied().max().unwrap_or(0);

    // Calculate bytes saved
    let page_size = if let Some(page) = vt.pages.first() {
        page.data.len()
    } else {
        0
    };

    let original_size = vt.total_pages * page_size;
    let deduped_size = unique_pages * page_size;
    let bytes_saved = original_size.saturating_sub(deduped_size);

    DeduplicationStats {
        total_pages: vt.total_pages,
        unique_pages,
        bytes_saved,
        max_duplicates,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::mipmap::{MipmapConfig, generate_mipmaps};
    use super::super::texture_importer::{GpuTextureFormat, TextureAsset, TextureMetadata, TextureState, SourceFormat};

    fn create_test_texture_data(width: u32, height: u32) -> TextureData {
        let bpp = 4;
        let data: Vec<u8> = (0..(width * height))
            .flat_map(|i| {
                let x = i % width;
                let y = i / width;
                let r = ((x * 255) / width.max(1)) as u8;
                let g = ((y * 255) / height.max(1)) as u8;
                let b = 128u8;
                let a = 255u8;
                vec![r, g, b, a]
            })
            .collect();

        TextureData::new(width, height, GpuTextureFormat::R8G8B8A8Unorm, data, false)
    }

    fn create_test_texture_asset(width: u32, height: u32) -> TextureAsset {
        let bpp = 4;
        let data: Vec<u8> = (0..(width * height))
            .flat_map(|i| {
                let x = i % width;
                let y = i / width;
                let r = ((x * 255) / width.max(1)) as u8;
                let g = ((y * 255) / height.max(1)) as u8;
                let b = 128u8;
                let a = 255u8;
                vec![r, g, b, a]
            })
            .collect();

        TextureAsset {
            id: 1,
            metadata: TextureMetadata {
                width,
                height,
                format: GpuTextureFormat::R8G8B8A8Unorm,
                memory_size: data.len(),
                is_srgb: false,
                source_format: SourceFormat::Png,
                source_bit_depth: 8,
                source_channels: 4,
            },
            data,
            state: TextureState::Pending,
        }
    }

    fn create_uniform_texture(width: u32, height: u32, color: [u8; 4]) -> TextureData {
        let data: Vec<u8> = (0..(width * height))
            .flat_map(|_| color.iter().copied())
            .collect();
        TextureData::new(width, height, GpuTextureFormat::R8G8B8A8Unorm, data, false)
    }

    // ---------------------------------------------------------------------------
    // Page Extraction Boundary Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_page_extraction_exact_boundary() {
        // Texture exactly 1 page (128x128)
        let texture = create_test_texture_data(128, 128);
        let asset = create_test_texture_asset(128, 128);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        assert_eq!(vt.width_pages, 1);
        assert_eq!(vt.height_pages, 1);
        assert!(!vt.pages.is_empty());
    }

    #[test]
    fn test_page_extraction_partial_boundary() {
        // Texture 150x150 - requires 2x2 pages at mip 0
        let texture = create_test_texture_data(150, 150);
        let asset = create_test_texture_asset(150, 150);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        assert_eq!(vt.width_pages, 2);
        assert_eq!(vt.height_pages, 2);
    }

    #[test]
    fn test_page_extraction_large_texture_boundary() {
        // Texture 513x513 - 5 pages wide/tall
        let texture = create_test_texture_data(513, 513);
        let asset = create_test_texture_asset(513, 513);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        assert_eq!(vt.width_pages, 5); // ceil(513/128) = 5
        assert_eq!(vt.height_pages, 5);
    }

    #[test]
    fn test_page_extraction_asymmetric_boundary() {
        // Asymmetric texture 256x64
        let texture = create_test_texture_data(256, 64);
        let asset = create_test_texture_asset(256, 64);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        assert_eq!(vt.width_pages, 2);  // 256/128 = 2
        assert_eq!(vt.height_pages, 1); // 64/128 = 1 (rounded up)
    }

    // ---------------------------------------------------------------------------
    // Mip Level Page Organization Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_mip_level_pages_count() {
        let texture = create_test_texture_data(256, 256);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // Mip 0: 2x2 = 4 pages (256/128)
        // Mip 1: 1x1 = 1 page (128/128)
        // Mip 2+: 1x1 pages each

        let mip0_pages = vt.pages_at_mip(0);
        let mip1_pages = vt.pages_at_mip(1);

        assert_eq!(mip0_pages.len(), 4);
        assert_eq!(mip1_pages.len(), 1);
    }

    #[test]
    fn test_mip_level_page_coordinates() {
        let texture = create_test_texture_data(256, 256);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig {
            use_morton_order: false, // Disable for predictable ordering
            ..Default::default()
        };
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // Check mip 0 pages have correct coordinates
        let mip0_pages = vt.pages_at_mip(0);
        let coords: Vec<(u32, u32)> = mip0_pages.iter()
            .map(|p| (p.page_x, p.page_y))
            .collect();

        assert!(coords.contains(&(0, 0)));
        assert!(coords.contains(&(1, 0)));
        assert!(coords.contains(&(0, 1)));
        assert!(coords.contains(&(1, 1)));
    }

    #[test]
    fn test_mip_level_page_size_reduction() {
        let texture = create_test_texture_data(512, 512);
        let asset = create_test_texture_asset(512, 512);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // Each mip level should have roughly 1/4 the pages of the previous
        let mip0 = vt.pages_at_mip(0).len();
        let mip1 = vt.pages_at_mip(1).len();
        let mip2 = vt.pages_at_mip(2).len();

        // 512/128 = 4x4 = 16 pages
        // 256/128 = 2x2 = 4 pages
        // 128/128 = 1x1 = 1 page
        assert_eq!(mip0, 16);
        assert_eq!(mip1, 4);
        assert_eq!(mip2, 1);
    }

    #[test]
    fn test_max_mip_levels_limit() {
        let texture = create_test_texture_data(512, 512);
        let asset = create_test_texture_asset(512, 512);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig {
            max_mip_levels: 2,
            ..Default::default()
        };
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // Should only have mip 0 and mip 1
        assert_eq!(vt.mip_levels, 2);
        assert!(vt.pages_at_mip(0).len() > 0);
        assert!(vt.pages_at_mip(1).len() > 0);
        assert_eq!(vt.pages_at_mip(2).len(), 0);
    }

    // ---------------------------------------------------------------------------
    // Morton Ordering Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_morton_index_basic() {
        assert_eq!(compute_morton_index(0, 0), 0);
        assert_eq!(compute_morton_index(1, 0), 1);
        assert_eq!(compute_morton_index(0, 1), 2);
        assert_eq!(compute_morton_index(1, 1), 3);
    }

    #[test]
    fn test_morton_index_larger() {
        // (2, 0) -> binary: x=10, y=00 -> interleaved: 0100 = 4
        assert_eq!(compute_morton_index(2, 0), 4);
        // (0, 2) -> binary: x=00, y=10 -> interleaved: 1000 = 8
        assert_eq!(compute_morton_index(0, 2), 8);
    }

    #[test]
    fn test_morton_decode_roundtrip() {
        for x in 0..16 {
            for y in 0..16 {
                let morton = compute_morton_index(x, y);
                let (dx, dy) = decode_morton_index(morton);
                assert_eq!((x, y), (dx, dy), "roundtrip failed for ({}, {})", x, y);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Page Table Generation Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_page_table_basic() {
        let texture = create_test_texture_data(256, 256);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        assert!(!vt.page_table.is_empty());

        // All pages should be marked resident
        let resident_count = vt.page_table.iter().filter(|e| e.resident).count();
        assert!(resident_count > 0);
    }

    #[test]
    fn test_page_table_entries_valid() {
        let texture = create_test_texture_data(256, 256);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // Check that resident entries have valid physical indices
        for entry in &vt.page_table {
            if entry.resident {
                assert!(entry.is_valid());
                assert_ne!(entry.physical_page, INVALID_PHYSICAL_PAGE);
            }
        }
    }

    #[test]
    fn test_page_table_without_dedup() {
        let texture = create_uniform_texture(256, 256, [128, 128, 128, 255]);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::without_deduplication();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // Without dedup, each page should have unique physical index
        let physical_indices: Vec<u32> = vt.page_table.iter()
            .filter(|e| e.resident)
            .map(|e| e.physical_page)
            .collect();

        let unique_indices: std::collections::HashSet<_> = physical_indices.iter().collect();
        assert_eq!(physical_indices.len(), unique_indices.len());
    }

    #[test]
    fn test_page_table_entry_types() {
        let resident = PageTableEntry::resident(5);
        assert!(resident.resident);
        assert_eq!(resident.physical_page, 5);
        assert!(resident.is_valid());

        let non_resident = PageTableEntry::non_resident();
        assert!(!non_resident.resident);
        assert_eq!(non_resident.physical_page, INVALID_PHYSICAL_PAGE);
        assert!(!non_resident.is_valid());
    }

    // ---------------------------------------------------------------------------
    // Content Hash Deduplication Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_dedup_identical_pages() {
        // Create uniform texture - all pages should be identical
        let texture = create_uniform_texture(256, 256, [100, 100, 100, 255]);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // With deduplication, unique pages should be much less than total
        assert!(vt.unique_pages < vt.total_pages);
    }

    #[test]
    fn test_dedup_varied_pages() {
        // Gradient texture - most pages should be unique
        let texture = create_test_texture_data(256, 256);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // Most pages should be unique for gradient
        assert!(vt.unique_pages > vt.total_pages / 2);
    }

    #[test]
    fn test_dedup_stats() {
        let texture = create_uniform_texture(256, 256, [50, 50, 50, 255]);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        let stats = calculate_dedup_stats(&vt);

        assert!(stats.dedup_ratio() > 0.0);
        assert!(stats.bytes_saved > 0);
    }

    // ---------------------------------------------------------------------------
    // Archive Packing Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_archive_pack_basic() {
        let texture = create_test_texture_data(256, 256);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        let archive = pack_for_archive(&vt);
        assert!(!archive.is_empty());
    }

    #[test]
    fn test_archive_pack_with_header() {
        let texture = create_test_texture_data(256, 256);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        let archive = pack_with_header(&vt);

        // Check magic
        assert_eq!(&archive[0..4], b"VTEX");

        // Check version
        let version = u32::from_le_bytes([archive[4], archive[5], archive[6], archive[7]]);
        assert_eq!(version, 1);

        // Check page size
        let page_size = u32::from_le_bytes([archive[8], archive[9], archive[10], archive[11]]);
        assert_eq!(page_size, DEFAULT_PAGE_SIZE);
    }

    // ---------------------------------------------------------------------------
    // Configuration Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_config_validate_valid() {
        let config = VirtualTextureConfig::default();
        assert!(config.validate().is_ok());

        let config = VirtualTextureConfig::with_page_size(64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_page_size() {
        let mut config = VirtualTextureConfig::default();
        config.page_size = 100; // Not power of 2
        assert!(config.validate().is_err());

        config.page_size = 8; // Too small
        assert!(config.validate().is_err());

        config.page_size = 1024; // Too large
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_custom_page_size() {
        let texture = create_test_texture_data(256, 256);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::with_page_size(64);
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // With 64 pixel pages, 256/64 = 4 pages per dimension
        assert_eq!(vt.width_pages, 4);
        assert_eq!(vt.height_pages, 4);
        assert_eq!(vt.page_size, 64);
    }

    // ---------------------------------------------------------------------------
    // Edge Case Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_empty_mips() {
        let texture = create_test_texture_data(128, 128);
        let config = VirtualTextureConfig::default();

        let mips: Vec<MipLevel> = Vec::new();
        let result = extract_pages(&texture, &mips, &config);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VirtualTextureError::NoMipLevels));
    }

    #[test]
    fn test_zero_dimension_texture() {
        let texture = TextureData::new(0, 128, GpuTextureFormat::R8G8B8A8Unorm, vec![], false);
        let asset = create_test_texture_asset(128, 128);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let result = extract_pages(&texture, &mips, &config);

        assert!(result.is_err());
    }

    #[test]
    fn test_small_texture_single_page() {
        let texture = create_test_texture_data(64, 64);
        let asset = create_test_texture_asset(64, 64);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // 64x64 fits in one 128x128 page
        assert_eq!(vt.width_pages, 1);
        assert_eq!(vt.height_pages, 1);
    }

    #[test]
    fn test_mip_page_info_calculation() {
        let info = calculate_mip_page_info(512, 512, 128, 10);

        assert_eq!(info[0].width_pages, 4);
        assert_eq!(info[0].height_pages, 4);
        assert_eq!(info[0].total_pages, 16);

        assert_eq!(info[1].width_pages, 2);
        assert_eq!(info[1].height_pages, 2);
        assert_eq!(info[1].total_pages, 4);
    }

    // ---------------------------------------------------------------------------
    // Virtual Texture Data Methods Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_vt_data_dedup_ratio() {
        let texture = create_uniform_texture(256, 256, [128, 128, 128, 255]);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        let ratio = vt.dedup_ratio();
        assert!(ratio > 0.0 && ratio <= 1.0);
    }

    #[test]
    fn test_vt_data_total_size() {
        let texture = create_test_texture_data(256, 256);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        let total_size = vt.total_data_size();
        assert!(total_size > 0);
    }

    #[test]
    fn test_vt_data_get_page() {
        let texture = create_test_texture_data(256, 256);
        let asset = create_test_texture_asset(256, 256);
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // Should find page at (0, 0) mip 0
        let page = vt.get_page(0, 0, 0);
        assert!(page.is_some());

        // Should not find non-existent page
        let no_page = vt.get_page(0, 100, 100);
        assert!(no_page.is_none());
    }

    // ---------------------------------------------------------------------------
    // Integration Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_full_pipeline() {
        // Create texture
        let texture = create_test_texture_data(512, 512);
        let asset = create_test_texture_asset(512, 512);

        // Generate mips
        let mips = generate_mipmaps(&asset, &MipmapConfig::default()).unwrap();

        // Extract pages
        let config = VirtualTextureConfig::default();
        let vt = extract_pages(&texture, &mips, &config).unwrap();

        // Verify data integrity
        assert!(vt.total_pages > 0);
        assert!(vt.unique_pages <= vt.total_pages);
        assert_eq!(vt.page_size, DEFAULT_PAGE_SIZE);

        // Pack for archive
        let archive = pack_for_archive(&vt);
        assert!(!archive.is_empty());

        // Calculate stats
        let stats = calculate_dedup_stats(&vt);
        assert_eq!(stats.total_pages, vt.total_pages);
    }
}
