//! GPU-driven rendering subsystem.
//!
//! Implements efficient GPU-driven data management for the renderer backend,
//! including buffer staging, indirect draw management, GPU-side culling
//! support, the bindless mesh table, the bindless material table, and the
//! bindless texture table.
//!
//! Sub-modules:
//! - `buffers`         -- BufferRegistry with triple-buffered staging
//! - `mesh_table`      -- Bindless Mesh Table (`array<MeshTableEntry>`) with
//!                        CPU-side manager for load-time population
//! - `material_table`  -- Bindless Material Table (`array<MaterialTableEntry>`)
//!                        with CPU-side manager and dirty-flag tracking
//! - `texture_table`   -- Bindless Texture Table (`texture_2d_array<f32>`) with
//!                        CPU-side manager and free-list allocation

pub mod buffers;
pub mod material_table;
pub mod mesh_table;
pub mod texture_table;

pub use buffers::{
    AcquireResult, BufferRegistry, BufferSlot, ReleaseResult, SlotState,
    StagingBufferDesc, SubmitResult, NUM_STAGING_SLOTS,
};

pub use material_table::{
    AddEntry as MaterialAddEntry, MaterialTable, MaterialTableEntry,
    RemoveResult as MaterialRemoveResult,
    DEFAULT_MATERIAL_TABLE_CAPACITY, MATERIAL_FLAG_DIRTY, MATERIAL_FLAG_VISIBLE,
    MATERIAL_TABLE_ENTRY_SIZE,
};

pub use mesh_table::{
    AddEntry, MeshTable, MeshTableEntry, RemoveResult,
    DEFAULT_MESH_TABLE_CAPACITY, MESH_TABLE_ENTRY_SIZE,
};
pub use texture_table::{
    AddEntry as TextureAddEntry, TextureTable, TextureTableEntry,
    RemoveResult as TextureRemoveResult,
    DEFAULT_TEXTURE_TABLE_CAPACITY, MAX_BINDLESS_TEXTURES, TEXTURE_TABLE_ENTRY_SIZE,
};
