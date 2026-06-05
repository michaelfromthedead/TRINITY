//! Parameter-only Material Hot-Reload (no shader recompilation)
//!
//! This module provides real-time material parameter updates without requiring
//! shader recompilation or pipeline rebuilds. Parameters flow through the Bridge
//! Data channel and are applied via uniform buffer updates on the next frame.
//!
//! # Architecture
//!
//! - **Bridge Data Channel**: Parameter changes arrive from the Python frontend
//!   via the bridge module and are queued for processing.
//! - **Uniform Buffer Updates**: Changed parameters are batched into buffer writes
//!   that update the GPU-side material table.
//! - **DepGraph Integration**: Structural material changes (not just parameters)
//!   trigger DepGraph updates, while parameter-only changes skip this overhead.
//!
//! # Performance
//!
//! - Parameter changes update rendering within 1 frame
//! - No pipeline rebuild on parameter-only changes
//! - Batch processing for multiple simultaneous parameter updates
//! - Triple-buffered staging prevents CPU-GPU sync stalls
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::param_hot_reload::{
//!     ParamHotReloader, MaterialParamUpdate, ParamValue, UniformBufferPool,
//! };
//! use std::time::Instant;
//!
//! let pool = UniformBufferPool::new(1024 * 64);
//! let mut reloader = ParamHotReloader::new(pool);
//!
//! // Queue a parameter change
//! reloader.queue_update(MaterialParamUpdate {
//!     material_id: MaterialId(42),
//!     param_name: "base_color".to_string(),
//!     new_value: ParamValue::Vec4([1.0, 0.0, 0.0, 1.0]),
//!     timestamp: Instant::now(),
//! });
//!
//! // On next frame, flush and apply updates
//! let writes = reloader.flush_updates();
//! for write in &writes {
//!     reloader.apply_to_uniform_buffer(write, &mut gpu_buffer);
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::gpu_driven::buffers::{AcquireResult, BufferRegistry, SubmitResult};
use crate::gpu_driven::material_table::{
    MaterialTable, MaterialTableEntry, MATERIAL_FLAG_DIRTY, MATERIAL_TABLE_ENTRY_SIZE,
};
use crate::material_dep_graph::DepGraph;

// ---------------------------------------------------------------------------
// Type Aliases and Identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a material instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

impl MaterialId {
    /// Create a new MaterialId.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw u32 value.
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for MaterialId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<MaterialId> for u32 {
    fn from(id: MaterialId) -> u32 {
        id.0
    }
}

/// Unique identifier for a texture resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u32);

impl TextureId {
    /// Create a new TextureId.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw u32 value.
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Sentinel value for "no texture bound".
    pub const INVALID: Self = Self(u32::MAX);
}

impl From<u32> for TextureId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<TextureId> for u32 {
    fn from(id: TextureId) -> u32 {
        id.0
    }
}

// ---------------------------------------------------------------------------
// Parameter Value Types
// ---------------------------------------------------------------------------

/// A parameter value that can be applied to a material.
///
/// Each variant maps to a specific field or field component in
/// `MaterialTableEntry`. The memory layout matches the GPU uniform buffer.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// Single f32 scalar (e.g., metallic, roughness).
    Float(f32),
    /// 2-component vector (e.g., UV offset).
    Vec2([f32; 2]),
    /// 3-component vector (e.g., RGB color without alpha).
    Vec3([f32; 3]),
    /// 4-component vector (e.g., base_color, emissive).
    Vec4([f32; 4]),
    /// RGBA color (semantically same as Vec4, but may trigger color space conversion).
    Color([f32; 4]),
    /// Texture reference update.
    Texture(TextureId),
}

impl ParamValue {
    /// Returns the size in bytes of this parameter value.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Float(_) => 4,
            Self::Vec2(_) => 8,
            Self::Vec3(_) => 12,
            Self::Vec4(_) | Self::Color(_) => 16,
            Self::Texture(_) => 4,
        }
    }

    /// Serialize this value to bytes (little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Float(v) => v.to_le_bytes().to_vec(),
            Self::Vec2(v) => {
                let mut bytes = Vec::with_capacity(8);
                bytes.extend_from_slice(&v[0].to_le_bytes());
                bytes.extend_from_slice(&v[1].to_le_bytes());
                bytes
            }
            Self::Vec3(v) => {
                let mut bytes = Vec::with_capacity(12);
                bytes.extend_from_slice(&v[0].to_le_bytes());
                bytes.extend_from_slice(&v[1].to_le_bytes());
                bytes.extend_from_slice(&v[2].to_le_bytes());
                bytes
            }
            Self::Vec4(v) | Self::Color(v) => {
                let mut bytes = Vec::with_capacity(16);
                bytes.extend_from_slice(&v[0].to_le_bytes());
                bytes.extend_from_slice(&v[1].to_le_bytes());
                bytes.extend_from_slice(&v[2].to_le_bytes());
                bytes.extend_from_slice(&v[3].to_le_bytes());
                bytes
            }
            Self::Texture(id) => id.raw().to_le_bytes().to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// Material Parameter Update
// ---------------------------------------------------------------------------

/// A single material parameter update request.
#[derive(Debug, Clone)]
pub struct MaterialParamUpdate {
    /// The material to update.
    pub material_id: MaterialId,
    /// Name of the parameter (e.g., "base_color", "metallic").
    pub param_name: String,
    /// The new value to apply.
    pub new_value: ParamValue,
    /// When this update was requested (for ordering and deduplication).
    pub timestamp: Instant,
}

impl MaterialParamUpdate {
    /// Create a new parameter update.
    pub fn new(
        material_id: MaterialId,
        param_name: impl Into<String>,
        new_value: ParamValue,
    ) -> Self {
        Self {
            material_id,
            param_name: param_name.into(),
            new_value,
            timestamp: Instant::now(),
        }
    }

    /// Create an update with a specific timestamp.
    pub fn with_timestamp(
        material_id: MaterialId,
        param_name: impl Into<String>,
        new_value: ParamValue,
        timestamp: Instant,
    ) -> Self {
        Self {
            material_id,
            param_name: param_name.into(),
            new_value,
            timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer Write Descriptor
// ---------------------------------------------------------------------------

/// Describes a write operation to be applied to a uniform buffer.
#[derive(Debug, Clone)]
pub struct BufferWrite {
    /// Offset in bytes from the start of the buffer.
    pub offset: usize,
    /// Data to write at that offset.
    pub data: Vec<u8>,
    /// Material ID this write belongs to (for debugging/tracking).
    pub material_id: MaterialId,
    /// Parameter name (for debugging/tracking).
    pub param_name: String,
}

impl BufferWrite {
    /// Create a new buffer write.
    pub fn new(offset: usize, data: Vec<u8>, material_id: MaterialId, param_name: String) -> Self {
        Self {
            offset,
            data,
            material_id,
            param_name,
        }
    }

    /// Returns the end offset (offset + data.len()).
    pub fn end_offset(&self) -> usize {
        self.offset + self.data.len()
    }
}

// ---------------------------------------------------------------------------
// Parameter Offset Map
// ---------------------------------------------------------------------------

/// Maps parameter names to their byte offsets within `MaterialTableEntry`.
///
/// This struct provides O(1) lookup for parameter offsets, enabling efficient
/// buffer write generation without runtime string parsing.
#[derive(Debug, Clone)]
pub struct ParamOffsetMap {
    offsets: HashMap<&'static str, usize>,
}

impl ParamOffsetMap {
    /// Create a new offset map with the standard MaterialTableEntry layout.
    pub fn new() -> Self {
        let mut offsets = HashMap::new();

        // base_color: [f32; 4] at offset 0
        offsets.insert("base_color", 0);
        offsets.insert("base_color.r", 0);
        offsets.insert("base_color.g", 4);
        offsets.insert("base_color.b", 8);
        offsets.insert("base_color.a", 12);

        // emissive: [f32; 4] at offset 16
        offsets.insert("emissive", 16);
        offsets.insert("emissive.r", 16);
        offsets.insert("emissive.g", 20);
        offsets.insert("emissive.b", 24);
        offsets.insert("emissive_intensity", 28);

        // metallic: f32 at offset 32
        offsets.insert("metallic", 32);

        // roughness: f32 at offset 36
        offsets.insert("roughness", 36);

        // occlusion: f32 at offset 40
        offsets.insert("occlusion", 40);

        // normal_scale: f32 at offset 44
        offsets.insert("normal_scale", 44);

        // albedo_texture_id: u32 at offset 48
        offsets.insert("albedo_texture", 48);
        offsets.insert("albedo_texture_id", 48);

        // normal_texture_id: u32 at offset 52
        offsets.insert("normal_texture", 52);
        offsets.insert("normal_texture_id", 52);

        // metallic_roughness_texture_id: u32 at offset 56
        offsets.insert("metallic_roughness_texture", 56);
        offsets.insert("metallic_roughness_texture_id", 56);

        // emissive_texture_id: u32 at offset 60
        offsets.insert("emissive_texture", 60);
        offsets.insert("emissive_texture_id", 60);

        // flags: u32 at offset 64
        offsets.insert("flags", 64);

        // alpha_cutoff: f32 at offset 68
        offsets.insert("alpha_cutoff", 68);

        Self { offsets }
    }

    /// Look up the byte offset for a parameter name.
    pub fn get(&self, param_name: &str) -> Option<usize> {
        self.offsets.get(param_name).copied()
    }

    /// Check if a parameter name is valid.
    pub fn contains(&self, param_name: &str) -> bool {
        self.offsets.contains_key(param_name)
    }
}

impl Default for ParamOffsetMap {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Uniform Buffer Pool
// ---------------------------------------------------------------------------

/// A pool of staging buffers for uniform buffer updates.
///
/// Wraps `BufferRegistry` to provide a uniform-buffer-specific interface
/// for hot-reload operations.
pub struct UniformBufferPool {
    registry: BufferRegistry,
}

impl UniformBufferPool {
    /// Create a new pool with the given default capacity per slot.
    pub fn new(default_capacity: usize) -> Self {
        Self {
            registry: BufferRegistry::new(default_capacity),
        }
    }

    /// Get the underlying buffer registry.
    pub fn registry(&self) -> &BufferRegistry {
        &self.registry
    }

    /// Get a mutable reference to the underlying buffer registry.
    pub fn registry_mut(&mut self) -> &mut BufferRegistry {
        &mut self.registry
    }

    /// Check if the pool is stalled (no free staging slots).
    pub fn is_stalled(&self) -> bool {
        self.registry.is_stalled()
    }

    /// Acquire a staging slot for writing.
    pub fn acquire(&mut self) -> Option<usize> {
        match self.registry.acquire_staging() {
            AcquireResult::Acquired { slot_index } => Some(slot_index),
            AcquireResult::NoSlotAvailable => None,
        }
    }

    /// Submit a staging slot with the given byte count.
    pub fn submit(&mut self, slot_index: usize, byte_count: usize) -> bool {
        matches!(
            self.registry.submit_staging(slot_index, byte_count),
            SubmitResult::Submitted
        )
    }

    /// Release a staging slot back to the pool.
    pub fn release(&mut self, slot_index: usize) -> bool {
        matches!(
            self.registry.release_staging(slot_index),
            crate::gpu_driven::buffers::ReleaseResult::Released
        )
    }

    /// Reset the pool (for device-loss recovery).
    pub fn reset(&mut self) {
        self.registry.reset();
    }
}

// ---------------------------------------------------------------------------
// Param Hot Reloader
// ---------------------------------------------------------------------------

/// Manages parameter-only material hot-reloading.
///
/// This is the main entry point for parameter hot-reload. It:
/// 1. Queues parameter updates from the Bridge Data channel
/// 2. Deduplicates and batches updates per material
/// 3. Generates buffer writes for GPU upload
/// 4. Tracks which materials have pending updates
///
/// # Thread Safety
///
/// `ParamHotReloader` is not `Sync`. In a multi-threaded scenario, wrap it
/// in a `Mutex` or use message-passing to funnel updates to a single thread.
pub struct ParamHotReloader {
    /// Pending updates keyed by (material_id, param_name).
    /// Later updates supersede earlier ones.
    pending_updates: HashMap<(MaterialId, String), MaterialParamUpdate>,

    /// Ordered queue of material IDs with pending updates (for batch ordering).
    update_order: VecDeque<MaterialId>,

    /// Set of materials that have pending updates (for dedup in update_order).
    materials_with_updates: std::collections::HashSet<MaterialId>,

    /// Pool for staging uniform buffer updates.
    uniform_buffer_pool: UniformBufferPool,

    /// Parameter offset lookup table.
    param_offsets: ParamOffsetMap,

    /// Statistics.
    stats: HotReloadStats,
}

/// Statistics for parameter hot-reload operations.
#[derive(Debug, Clone, Default)]
pub struct HotReloadStats {
    /// Total updates queued.
    pub updates_queued: usize,
    /// Total updates flushed (applied).
    pub updates_flushed: usize,
    /// Total buffer writes generated.
    pub buffer_writes: usize,
    /// Number of updates that were superseded (deduped).
    pub updates_superseded: usize,
    /// Number of flush cycles.
    pub flush_cycles: usize,
}

impl ParamHotReloader {
    /// Create a new ParamHotReloader with the given buffer pool.
    pub fn new(uniform_buffer_pool: UniformBufferPool) -> Self {
        Self {
            pending_updates: HashMap::new(),
            update_order: VecDeque::new(),
            materials_with_updates: std::collections::HashSet::new(),
            uniform_buffer_pool,
            param_offsets: ParamOffsetMap::new(),
            stats: HotReloadStats::default(),
        }
    }

    /// Queue a parameter update.
    ///
    /// If an update for the same (material_id, param_name) is already pending,
    /// the newer update supersedes the older one.
    pub fn queue_update(&mut self, update: MaterialParamUpdate) {
        let key = (update.material_id, update.param_name.clone());

        // Track if this is a new material
        if !self.materials_with_updates.contains(&update.material_id) {
            self.update_order.push_back(update.material_id);
            self.materials_with_updates.insert(update.material_id);
        }

        // Insert or replace
        if self.pending_updates.insert(key, update).is_some() {
            self.stats.updates_superseded += 1;
        }

        self.stats.updates_queued += 1;
    }

    /// Queue multiple updates at once.
    pub fn queue_updates(&mut self, updates: impl IntoIterator<Item = MaterialParamUpdate>) {
        for update in updates {
            self.queue_update(update);
        }
    }

    /// Check if there are any pending updates.
    pub fn has_pending(&self) -> bool {
        !self.pending_updates.is_empty()
    }

    /// Get the number of pending updates.
    pub fn pending_count(&self) -> usize {
        self.pending_updates.len()
    }

    /// Get the number of materials with pending updates.
    pub fn materials_pending(&self) -> usize {
        self.materials_with_updates.len()
    }

    /// Flush all pending updates and generate buffer writes.
    ///
    /// Returns a vector of `BufferWrite` descriptors that should be applied
    /// to the GPU uniform buffer. After calling this, the pending queue is cleared.
    pub fn flush_updates(&mut self) -> Vec<BufferWrite> {
        if self.pending_updates.is_empty() {
            return Vec::new();
        }

        let mut writes = Vec::with_capacity(self.pending_updates.len());

        // Process updates in the order materials were first touched
        for &material_id in &self.update_order {
            // Find all updates for this material
            let material_updates: Vec<_> = self
                .pending_updates
                .iter()
                .filter(|((mid, _), _)| *mid == material_id)
                .map(|(_, update)| update.clone())
                .collect();

            for update in material_updates {
                if let Some(offset) = self.param_offsets.get(&update.param_name) {
                    // Calculate the absolute offset in the material table
                    let material_offset = material_id.raw() as usize * MATERIAL_TABLE_ENTRY_SIZE;
                    let absolute_offset = material_offset + offset;

                    writes.push(BufferWrite::new(
                        absolute_offset,
                        update.new_value.to_bytes(),
                        material_id,
                        update.param_name.clone(),
                    ));

                    self.stats.buffer_writes += 1;
                }
                // Silently ignore unknown parameter names (or could log a warning)
            }
        }

        // Update stats
        self.stats.updates_flushed += self.pending_updates.len();
        self.stats.flush_cycles += 1;

        // Clear pending state
        self.pending_updates.clear();
        self.update_order.clear();
        self.materials_with_updates.clear();

        writes
    }

    /// Apply buffer writes directly to a byte buffer (uniform buffer memory).
    ///
    /// # Arguments
    ///
    /// * `writes` - The buffer writes to apply
    /// * `buffer` - The destination buffer (must be large enough for all writes)
    ///
    /// # Panics
    ///
    /// Panics if any write exceeds the buffer bounds.
    pub fn apply_to_uniform_buffer(writes: &[BufferWrite], buffer: &mut [u8]) {
        for write in writes {
            let end = write.end_offset();
            assert!(
                end <= buffer.len(),
                "BufferWrite exceeds buffer bounds: offset={}, len={}, buffer_len={}",
                write.offset,
                write.data.len(),
                buffer.len()
            );
            buffer[write.offset..end].copy_from_slice(&write.data);
        }
    }

    /// Apply buffer writes to a MaterialTable directly.
    ///
    /// This updates the CPU-side MaterialTable and marks affected entries dirty.
    pub fn apply_to_material_table(&self, writes: &[BufferWrite], table: &mut MaterialTable) {
        for write in writes {
            // Get the material entry and update it
            if let Some(entry) = table.get_mut(write.material_id.raw()) {
                // Apply the write to the entry's bytes
                // This is a simplified version; a full implementation would
                // parse the param_name and update the specific field.
                Self::apply_write_to_entry(write, entry);
            }
        }
    }

    /// Apply a single write to a MaterialTableEntry.
    fn apply_write_to_entry(write: &BufferWrite, entry: &mut MaterialTableEntry) {
        // Get the offset within the entry (not the absolute offset)
        let entry_offset = write.offset % MATERIAL_TABLE_ENTRY_SIZE;

        match entry_offset {
            0..=15 => {
                // base_color
                if let Some(arr) = Self::try_read_vec4(&write.data) {
                    if entry_offset == 0 && write.data.len() == 16 {
                        entry.base_color = arr;
                    } else {
                        // Partial update
                        let idx = entry_offset / 4;
                        if let Some(f) = Self::try_read_f32(&write.data) {
                            entry.base_color[idx] = f;
                        }
                    }
                }
            }
            16..=31 => {
                // emissive
                if entry_offset == 16 && write.data.len() == 16 {
                    if let Some(arr) = Self::try_read_vec4(&write.data) {
                        entry.emissive = arr;
                    }
                } else {
                    let idx = (entry_offset - 16) / 4;
                    if let Some(f) = Self::try_read_f32(&write.data) {
                        entry.emissive[idx] = f;
                    }
                }
            }
            32 => {
                if let Some(f) = Self::try_read_f32(&write.data) {
                    entry.metallic = f;
                }
            }
            36 => {
                if let Some(f) = Self::try_read_f32(&write.data) {
                    entry.roughness = f;
                }
            }
            40 => {
                if let Some(f) = Self::try_read_f32(&write.data) {
                    entry.occlusion = f;
                }
            }
            44 => {
                if let Some(f) = Self::try_read_f32(&write.data) {
                    entry.normal_scale = f;
                }
            }
            48 => {
                if let Some(u) = Self::try_read_u32(&write.data) {
                    entry.albedo_texture_id = u;
                }
            }
            52 => {
                if let Some(u) = Self::try_read_u32(&write.data) {
                    entry.normal_texture_id = u;
                }
            }
            56 => {
                if let Some(u) = Self::try_read_u32(&write.data) {
                    entry.metallic_roughness_texture_id = u;
                }
            }
            60 => {
                if let Some(u) = Self::try_read_u32(&write.data) {
                    entry.emissive_texture_id = u;
                }
            }
            64 => {
                if let Some(u) = Self::try_read_u32(&write.data) {
                    entry.flags = u | MATERIAL_FLAG_DIRTY;
                }
            }
            68 => {
                if let Some(f) = Self::try_read_f32(&write.data) {
                    entry.alpha_cutoff = f;
                }
            }
            _ => {
                // Unknown offset - ignore or log warning
            }
        }
    }

    fn try_read_f32(data: &[u8]) -> Option<f32> {
        if data.len() >= 4 {
            Some(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
        } else {
            None
        }
    }

    fn try_read_u32(data: &[u8]) -> Option<u32> {
        if data.len() >= 4 {
            Some(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
        } else {
            None
        }
    }

    fn try_read_vec4(data: &[u8]) -> Option<[f32; 4]> {
        if data.len() >= 16 {
            Some([
                f32::from_le_bytes([data[0], data[1], data[2], data[3]]),
                f32::from_le_bytes([data[4], data[5], data[6], data[7]]),
                f32::from_le_bytes([data[8], data[9], data[10], data[11]]),
                f32::from_le_bytes([data[12], data[13], data[14], data[15]]),
            ])
        } else {
            None
        }
    }

    /// Get hot-reload statistics.
    pub fn stats(&self) -> &HotReloadStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = HotReloadStats::default();
    }

    /// Get a reference to the uniform buffer pool.
    pub fn buffer_pool(&self) -> &UniformBufferPool {
        &self.uniform_buffer_pool
    }

    /// Get a mutable reference to the uniform buffer pool.
    pub fn buffer_pool_mut(&mut self) -> &mut UniformBufferPool {
        &mut self.uniform_buffer_pool
    }

    /// Get a reference to the parameter offset map.
    pub fn param_offsets(&self) -> &ParamOffsetMap {
        &self.param_offsets
    }
}

// ---------------------------------------------------------------------------
// Bridge Data Channel Integration
// ---------------------------------------------------------------------------

/// Message type for parameter updates arriving via the Bridge Data channel.
#[derive(Debug, Clone)]
pub struct BridgeParamMessage {
    /// Material ID to update.
    pub material_id: u32,
    /// Parameter name.
    pub param_name: String,
    /// Parameter value (serialized).
    pub value: BridgeValue,
}

/// Value types that can arrive from the Python bridge.
#[derive(Debug, Clone)]
pub enum BridgeValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Int(i32),
    UInt(u32),
}

impl BridgeValue {
    /// Convert to a ParamValue, inferring the best type based on context.
    pub fn to_param_value(&self, param_name: &str) -> ParamValue {
        match self {
            Self::Float(v) => ParamValue::Float(*v),
            Self::Vec2(v) => ParamValue::Vec2(*v),
            Self::Vec3(v) => ParamValue::Vec3(*v),
            Self::Vec4(v) => {
                // Check if this is a color parameter
                if param_name.contains("color") || param_name.contains("emissive") {
                    ParamValue::Color(*v)
                } else {
                    ParamValue::Vec4(*v)
                }
            }
            Self::Int(v) => ParamValue::Texture(TextureId::new(*v as u32)),
            Self::UInt(v) => ParamValue::Texture(TextureId::new(*v)),
        }
    }
}

/// Process messages from the Bridge Data channel.
pub fn process_bridge_messages(
    reloader: &mut ParamHotReloader,
    messages: impl IntoIterator<Item = BridgeParamMessage>,
) {
    for msg in messages {
        let param_value = msg.value.to_param_value(&msg.param_name);
        reloader.queue_update(MaterialParamUpdate::new(
            MaterialId::new(msg.material_id),
            msg.param_name,
            param_value,
        ));
    }
}

// ---------------------------------------------------------------------------
// DepGraph Integration
// ---------------------------------------------------------------------------

/// Determines if a parameter change requires DepGraph update.
///
/// Parameter-only changes (colors, floats) do NOT require DepGraph update.
/// Structural changes (texture bindings) MAY require DepGraph update if the
/// texture itself changed shaders.
pub fn requires_dep_graph_update(update: &MaterialParamUpdate) -> bool {
    // Texture changes might require DepGraph update if shader code changed
    // For pure parameter changes, skip DepGraph overhead
    matches!(update.new_value, ParamValue::Texture(_))
}

/// Notify the DepGraph of structural material changes.
pub fn notify_dep_graph(dep_graph: &mut DepGraph, material_id: MaterialId, param_name: &str) {
    // For texture parameter changes, we might need to invalidate dependent materials
    // This is a simplified implementation; a full one would check if the texture
    // change affects shader compilation.
    if param_name.contains("texture") {
        // Mark this material as potentially needing shader recompilation
        // The actual invalidation happens in hot_reload.rs
        dep_graph.add_include(material_id.raw(), format!("param:{}", param_name));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Test helpers ----

    fn make_reloader() -> ParamHotReloader {
        let pool = UniformBufferPool::new(64 * 1024);
        ParamHotReloader::new(pool)
    }

    fn make_update(material_id: u32, param_name: &str, value: ParamValue) -> MaterialParamUpdate {
        MaterialParamUpdate::new(MaterialId::new(material_id), param_name, value)
    }

    // ---- Test 1: Parameter change queued correctly ----

    #[test]
    fn test_parameter_change_queued_correctly() {
        let mut reloader = make_reloader();

        let update = make_update(1, "metallic", ParamValue::Float(0.5));
        reloader.queue_update(update);

        assert!(reloader.has_pending());
        assert_eq!(reloader.pending_count(), 1);
        assert_eq!(reloader.materials_pending(), 1);
    }

    #[test]
    fn test_multiple_parameters_same_material() {
        let mut reloader = make_reloader();

        reloader.queue_update(make_update(1, "metallic", ParamValue::Float(0.5)));
        reloader.queue_update(make_update(1, "roughness", ParamValue::Float(0.3)));
        reloader.queue_update(make_update(1, "base_color", ParamValue::Vec4([1.0, 0.0, 0.0, 1.0])));

        assert_eq!(reloader.pending_count(), 3);
        assert_eq!(reloader.materials_pending(), 1);
    }

    #[test]
    fn test_same_parameter_superseded() {
        let mut reloader = make_reloader();

        reloader.queue_update(make_update(1, "metallic", ParamValue::Float(0.5)));
        reloader.queue_update(make_update(1, "metallic", ParamValue::Float(0.8)));

        assert_eq!(reloader.pending_count(), 1); // Only one pending
        assert_eq!(reloader.stats().updates_superseded, 1);

        let writes = reloader.flush_updates();
        assert_eq!(writes.len(), 1);

        // Verify it's the newer value (0.8)
        let bytes = &writes[0].data;
        let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert!((value - 0.8).abs() < f32::EPSILON);
    }

    // ---- Test 2: Flush returns pending updates ----

    #[test]
    fn test_flush_returns_pending_updates() {
        let mut reloader = make_reloader();

        reloader.queue_update(make_update(1, "metallic", ParamValue::Float(0.5)));
        reloader.queue_update(make_update(2, "roughness", ParamValue::Float(0.3)));

        let writes = reloader.flush_updates();

        assert_eq!(writes.len(), 2);
        assert!(!reloader.has_pending());
        assert_eq!(reloader.pending_count(), 0);
    }

    #[test]
    fn test_flush_empty_returns_empty() {
        let mut reloader = make_reloader();

        let writes = reloader.flush_updates();
        assert!(writes.is_empty());
    }

    // ---- Test 3: Uniform buffer updated within 1 frame ----

    #[test]
    fn test_uniform_buffer_updated_within_1_frame() {
        let mut reloader = make_reloader();
        let mut buffer = vec![0u8; MATERIAL_TABLE_ENTRY_SIZE * 4];

        // Material 0, metallic at offset 32
        reloader.queue_update(make_update(0, "metallic", ParamValue::Float(0.75)));

        // Flush and apply
        let writes = reloader.flush_updates();
        ParamHotReloader::apply_to_uniform_buffer(&writes, &mut buffer);

        // Verify the value was written at the correct offset
        let value = f32::from_le_bytes([buffer[32], buffer[33], buffer[34], buffer[35]]);
        assert!((value - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_multiple_materials_buffer_update() {
        let mut reloader = make_reloader();
        let mut buffer = vec![0u8; MATERIAL_TABLE_ENTRY_SIZE * 4];

        reloader.queue_update(make_update(0, "metallic", ParamValue::Float(0.1)));
        reloader.queue_update(make_update(1, "metallic", ParamValue::Float(0.2)));
        reloader.queue_update(make_update(2, "metallic", ParamValue::Float(0.3)));

        let writes = reloader.flush_updates();
        ParamHotReloader::apply_to_uniform_buffer(&writes, &mut buffer);

        // Material 0 metallic at offset 32
        let v0 = f32::from_le_bytes([buffer[32], buffer[33], buffer[34], buffer[35]]);
        assert!((v0 - 0.1).abs() < f32::EPSILON);

        // Material 1 metallic at offset 80 + 32 = 112
        let v1 = f32::from_le_bytes([buffer[112], buffer[113], buffer[114], buffer[115]]);
        assert!((v1 - 0.2).abs() < f32::EPSILON);

        // Material 2 metallic at offset 160 + 32 = 192
        let v2 = f32::from_le_bytes([buffer[192], buffer[193], buffer[194], buffer[195]]);
        assert!((v2 - 0.3).abs() < f32::EPSILON);
    }

    // ---- Test 4: No pipeline rebuild triggered ----

    #[test]
    fn test_no_pipeline_rebuild_for_float_params() {
        let update = make_update(1, "metallic", ParamValue::Float(0.5));
        assert!(!requires_dep_graph_update(&update));
    }

    #[test]
    fn test_no_pipeline_rebuild_for_vec4_params() {
        let update = make_update(1, "base_color", ParamValue::Vec4([1.0, 0.0, 0.0, 1.0]));
        assert!(!requires_dep_graph_update(&update));
    }

    #[test]
    fn test_texture_may_require_dep_graph_update() {
        let update = make_update(1, "albedo_texture", ParamValue::Texture(TextureId::new(5)));
        assert!(requires_dep_graph_update(&update));
    }

    // ---- Test 5: Multiple parameter changes batched ----

    #[test]
    fn test_multiple_parameter_changes_batched() {
        let mut reloader = make_reloader();

        // Queue many updates
        for i in 0..10 {
            reloader.queue_update(make_update(i, "metallic", ParamValue::Float(i as f32 / 10.0)));
        }

        assert_eq!(reloader.pending_count(), 10);
        assert_eq!(reloader.materials_pending(), 10);

        let writes = reloader.flush_updates();
        assert_eq!(writes.len(), 10);
        assert_eq!(reloader.stats().flush_cycles, 1);
    }

    #[test]
    fn test_batch_preserves_order() {
        let mut reloader = make_reloader();

        reloader.queue_update(make_update(2, "metallic", ParamValue::Float(0.2)));
        reloader.queue_update(make_update(0, "metallic", ParamValue::Float(0.0)));
        reloader.queue_update(make_update(1, "metallic", ParamValue::Float(0.1)));

        let writes = reloader.flush_updates();

        // Materials should appear in the order they were first touched
        assert_eq!(writes[0].material_id, MaterialId::new(2));
        assert_eq!(writes[1].material_id, MaterialId::new(0));
        assert_eq!(writes[2].material_id, MaterialId::new(1));
    }

    // ---- Test 6: Texture parameter updates work ----

    #[test]
    fn test_texture_parameter_updates() {
        let mut reloader = make_reloader();
        let mut buffer = vec![0u8; MATERIAL_TABLE_ENTRY_SIZE * 2];

        reloader.queue_update(make_update(0, "albedo_texture", ParamValue::Texture(TextureId::new(42))));

        let writes = reloader.flush_updates();
        ParamHotReloader::apply_to_uniform_buffer(&writes, &mut buffer);

        // albedo_texture_id at offset 48
        let tex_id = u32::from_le_bytes([buffer[48], buffer[49], buffer[50], buffer[51]]);
        assert_eq!(tex_id, 42);
    }

    #[test]
    fn test_multiple_texture_parameters() {
        let mut reloader = make_reloader();
        let mut buffer = vec![0u8; MATERIAL_TABLE_ENTRY_SIZE * 2];

        reloader.queue_update(make_update(0, "albedo_texture", ParamValue::Texture(TextureId::new(1))));
        reloader.queue_update(make_update(0, "normal_texture", ParamValue::Texture(TextureId::new(2))));
        reloader.queue_update(make_update(0, "emissive_texture", ParamValue::Texture(TextureId::new(3))));

        let writes = reloader.flush_updates();
        ParamHotReloader::apply_to_uniform_buffer(&writes, &mut buffer);

        assert_eq!(u32::from_le_bytes([buffer[48], buffer[49], buffer[50], buffer[51]]), 1);
        assert_eq!(u32::from_le_bytes([buffer[52], buffer[53], buffer[54], buffer[55]]), 2);
        assert_eq!(u32::from_le_bytes([buffer[60], buffer[61], buffer[62], buffer[63]]), 3);
    }

    // ---- Additional unit tests ----

    #[test]
    fn test_param_value_size_bytes() {
        assert_eq!(ParamValue::Float(0.0).size_bytes(), 4);
        assert_eq!(ParamValue::Vec2([0.0; 2]).size_bytes(), 8);
        assert_eq!(ParamValue::Vec3([0.0; 3]).size_bytes(), 12);
        assert_eq!(ParamValue::Vec4([0.0; 4]).size_bytes(), 16);
        assert_eq!(ParamValue::Color([0.0; 4]).size_bytes(), 16);
        assert_eq!(ParamValue::Texture(TextureId::new(0)).size_bytes(), 4);
    }

    #[test]
    fn test_param_value_to_bytes() {
        let float_val = ParamValue::Float(1.5);
        let bytes = float_val.to_bytes();
        assert_eq!(bytes.len(), 4);
        let decoded = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert!((decoded - 1.5).abs() < f32::EPSILON);

        let vec2_val = ParamValue::Vec2([1.0, 2.0]);
        let bytes = vec2_val.to_bytes();
        assert_eq!(bytes.len(), 8);

        let tex_val = ParamValue::Texture(TextureId::new(123));
        let bytes = tex_val.to_bytes();
        assert_eq!(bytes.len(), 4);
        let decoded = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(decoded, 123);
    }

    #[test]
    fn test_param_offset_map() {
        let map = ParamOffsetMap::new();

        assert_eq!(map.get("base_color"), Some(0));
        assert_eq!(map.get("emissive"), Some(16));
        assert_eq!(map.get("metallic"), Some(32));
        assert_eq!(map.get("roughness"), Some(36));
        assert_eq!(map.get("occlusion"), Some(40));
        assert_eq!(map.get("normal_scale"), Some(44));
        assert_eq!(map.get("albedo_texture"), Some(48));
        assert_eq!(map.get("normal_texture"), Some(52));
        assert_eq!(map.get("alpha_cutoff"), Some(68));

        assert!(map.contains("metallic"));
        assert!(!map.contains("nonexistent"));
        assert_eq!(map.get("nonexistent"), None);
    }

    #[test]
    fn test_buffer_write_end_offset() {
        let write = BufferWrite::new(100, vec![0u8; 16], MaterialId::new(0), "test".to_string());
        assert_eq!(write.end_offset(), 116);
    }

    #[test]
    fn test_material_id_conversions() {
        let id = MaterialId::new(42);
        assert_eq!(id.raw(), 42);

        let from_u32: MaterialId = 123u32.into();
        assert_eq!(from_u32.raw(), 123);

        let to_u32: u32 = id.into();
        assert_eq!(to_u32, 42);
    }

    #[test]
    fn test_texture_id_invalid() {
        assert_eq!(TextureId::INVALID.raw(), u32::MAX);
    }

    #[test]
    fn test_bridge_value_to_param_value() {
        let float_val = BridgeValue::Float(0.5);
        assert!(matches!(float_val.to_param_value("metallic"), ParamValue::Float(v) if (v - 0.5).abs() < f32::EPSILON));

        let color_val = BridgeValue::Vec4([1.0, 0.0, 0.0, 1.0]);
        assert!(matches!(color_val.to_param_value("base_color"), ParamValue::Color(_)));

        let vec4_val = BridgeValue::Vec4([1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(vec4_val.to_param_value("some_other"), ParamValue::Vec4(_)));

        let uint_val = BridgeValue::UInt(5);
        assert!(matches!(uint_val.to_param_value("albedo_texture"), ParamValue::Texture(id) if id.raw() == 5));
    }

    #[test]
    fn test_process_bridge_messages() {
        let mut reloader = make_reloader();

        let messages = vec![
            BridgeParamMessage {
                material_id: 1,
                param_name: "metallic".to_string(),
                value: BridgeValue::Float(0.5),
            },
            BridgeParamMessage {
                material_id: 2,
                param_name: "base_color".to_string(),
                value: BridgeValue::Vec4([1.0, 0.0, 0.0, 1.0]),
            },
        ];

        process_bridge_messages(&mut reloader, messages);

        assert_eq!(reloader.pending_count(), 2);
    }

    #[test]
    fn test_stats_tracking() {
        let mut reloader = make_reloader();

        reloader.queue_update(make_update(1, "metallic", ParamValue::Float(0.5)));
        reloader.queue_update(make_update(1, "metallic", ParamValue::Float(0.6)));
        reloader.queue_update(make_update(2, "roughness", ParamValue::Float(0.3)));

        let _ = reloader.flush_updates();

        assert_eq!(reloader.stats().updates_queued, 3);
        assert_eq!(reloader.stats().updates_superseded, 1);
        assert_eq!(reloader.stats().updates_flushed, 2);
        assert_eq!(reloader.stats().buffer_writes, 2);
        assert_eq!(reloader.stats().flush_cycles, 1);
    }

    #[test]
    fn test_reset_stats() {
        let mut reloader = make_reloader();

        reloader.queue_update(make_update(1, "metallic", ParamValue::Float(0.5)));
        let _ = reloader.flush_updates();

        assert!(reloader.stats().updates_queued > 0);

        reloader.reset_stats();

        assert_eq!(reloader.stats().updates_queued, 0);
        assert_eq!(reloader.stats().flush_cycles, 0);
    }

    #[test]
    fn test_uniform_buffer_pool() {
        let mut pool = UniformBufferPool::new(1024);

        assert!(!pool.is_stalled());

        let slot = pool.acquire();
        assert!(slot.is_some());

        // Submit and release
        assert!(pool.submit(slot.unwrap(), 64));
        assert!(pool.release(slot.unwrap()));
    }

    #[test]
    fn test_vec4_parameter_update() {
        let mut reloader = make_reloader();
        let mut buffer = vec![0u8; MATERIAL_TABLE_ENTRY_SIZE * 2];

        reloader.queue_update(make_update(0, "base_color", ParamValue::Vec4([0.5, 0.6, 0.7, 1.0])));

        let writes = reloader.flush_updates();
        ParamHotReloader::apply_to_uniform_buffer(&writes, &mut buffer);

        // base_color at offset 0
        let r = f32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        let g = f32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
        let b = f32::from_le_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]);
        let a = f32::from_le_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]);

        assert!((r - 0.5).abs() < f32::EPSILON);
        assert!((g - 0.6).abs() < f32::EPSILON);
        assert!((b - 0.7).abs() < f32::EPSILON);
        assert!((a - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_emissive_parameter_update() {
        let mut reloader = make_reloader();
        let mut buffer = vec![0u8; MATERIAL_TABLE_ENTRY_SIZE * 2];

        reloader.queue_update(make_update(0, "emissive", ParamValue::Vec4([1.0, 0.5, 0.0, 2.0])));

        let writes = reloader.flush_updates();
        ParamHotReloader::apply_to_uniform_buffer(&writes, &mut buffer);

        // emissive at offset 16
        let r = f32::from_le_bytes([buffer[16], buffer[17], buffer[18], buffer[19]]);
        let g = f32::from_le_bytes([buffer[20], buffer[21], buffer[22], buffer[23]]);
        let b = f32::from_le_bytes([buffer[24], buffer[25], buffer[26], buffer[27]]);
        let intensity = f32::from_le_bytes([buffer[28], buffer[29], buffer[30], buffer[31]]);

        assert!((r - 1.0).abs() < f32::EPSILON);
        assert!((g - 0.5).abs() < f32::EPSILON);
        assert!((b - 0.0).abs() < f32::EPSILON);
        assert!((intensity - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_alpha_cutoff_update() {
        let mut reloader = make_reloader();
        let mut buffer = vec![0u8; MATERIAL_TABLE_ENTRY_SIZE * 2];

        reloader.queue_update(make_update(0, "alpha_cutoff", ParamValue::Float(0.5)));

        let writes = reloader.flush_updates();
        ParamHotReloader::apply_to_uniform_buffer(&writes, &mut buffer);

        // alpha_cutoff at offset 68
        let value = f32::from_le_bytes([buffer[68], buffer[69], buffer[70], buffer[71]]);
        assert!((value - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_to_material_table() {
        let mut reloader = make_reloader();
        let mut table = MaterialTable::with_capacity(4);

        // Add a material
        let entry = MaterialTableEntry {
            base_color: [0.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            ..MaterialTableEntry::zeroed()
        };
        let idx = table.add(entry);
        table.mark_clean();

        // Queue an update
        reloader.queue_update(MaterialParamUpdate::new(
            MaterialId::new(idx),
            "metallic",
            ParamValue::Float(0.8),
        ));

        let writes = reloader.flush_updates();
        reloader.apply_to_material_table(&writes, &mut table);

        // Verify the update was applied
        let updated = table.get(idx).unwrap();
        assert!((updated.metallic - 0.8).abs() < f32::EPSILON);
        assert!(table.any_dirty());
    }

    #[test]
    fn test_apply_buffer_out_of_bounds_panics() {
        let writes = vec![BufferWrite::new(
            100,
            vec![0u8; 16],
            MaterialId::new(0),
            "test".to_string(),
        )];
        let mut buffer = vec![0u8; 50]; // Too small

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ParamHotReloader::apply_to_uniform_buffer(&writes, &mut buffer);
        }));

        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_param_silently_ignored() {
        let mut reloader = make_reloader();

        reloader.queue_update(make_update(0, "nonexistent_param", ParamValue::Float(0.5)));

        let writes = reloader.flush_updates();

        // No buffer write generated for unknown param
        assert!(writes.is_empty());
    }

    #[test]
    fn test_dep_graph_integration() {
        let mut dep_graph = DepGraph::new();
        let material_id = MaterialId::new(1);

        // Notify about a texture change
        notify_dep_graph(&mut dep_graph, material_id, "albedo_texture");

        // The material should now have a dependency on the param include
        assert!(dep_graph.materials_to_includes.contains_key(&material_id.raw()));
    }
}
