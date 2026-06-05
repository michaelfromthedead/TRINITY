//! Asset pipeline module for TRINITY renderer.
//!
//! Provides content hashing, glTF parsing, texture import, and asset management
//! for the TRINITY engine.
//!
//! # Modules
//!
//! - [`content_hash`] - BLAKE3/SHA-256 content hashing with migration support
//! - [`content_store`] - Streaming content-addressed storage with seeking support
//! - [`integrity`] - CRC-32C integrity verification with replication (T-AS-4.6)
//! - [`sqlite_metadata`] - SQLite-backed metadata queries for content store (T-AS-4.4)
//! - [`provenance`] - Provenance chain tracking for reproducibility and debugging (T-AS-4.7)
//! - [`gltf_parser`] - glTF 2.0 parser with progressive loading and streaming support
//! - [`draco`] - KHR_draco_mesh_compression support for glTF (T-AS-1.7)
//! - [`texture_importer`] - PNG/JPEG/TGA/BMP texture import with sRGB detection
//! - [`hdr_formats`] - EXR/HDR/TIFF/PSD HDR format import with tone mapping
//! - [`vertex_format`] - Vertex format conversion with compression support
//! - [`compressed_formats`] - KTX v1/DDS container parsing with GPU passthrough (T-AS-2.7)
//! - [`ktx2_parser`] - KTX2 container parsing with Basis Universal transcoding (T-AS-2.3)
//! - [`cubemap`] - Cubemap and texture array assembly (T-AS-2.5)
//! - [`virtual_texture`] - Virtual texturing page extraction for streaming (T-AS-2.6)
//! - [`delta_sync`] - Incremental asset synchronization with O(differences) proofs (T-AS-4.8)
//! - [`blas`] - BLAS baking pipeline for ray tracing (T-AS-1.5)
//! - [`lod`] - LOD generation with QEM simplification and cross-fade (T-AS-1.6)

pub mod blas;
pub mod compressed_formats;
pub mod content_hash;
pub mod content_store;
pub mod cubemap;
pub mod delta_sync;
pub mod draco;
pub mod gltf_parser;
pub mod hdr_formats;
pub mod index_buffer;
pub mod integrity;
pub mod ktx2_parser;
pub mod lod;
pub mod meshlet;
pub mod mipmap;
pub mod provenance;
pub mod sqlite_metadata;
pub mod texture_importer;
pub mod vertex_format;
pub mod virtual_texture;

pub use content_hash::{
    AssetManifestEntry, ContentHashWrapper, ContentHasher, DualHash, ExtendedHash,
    HashOutputLength,
};

pub use content_store::{
    ContentStore, ContentStoreConfig, ContentStoreError, FileContentStore,
    MemoryContentStore, SeekableReader, StreamingHashReader, StreamingHashWriter,
    copy_with_hash, hash_reader, verify_content,
    DEFAULT_BUFFER_SIZE, MAX_BUFFER_SIZE, MIN_BUFFER_SIZE,
    // LRU eviction support (T-AS-4.3)
    LruContentStore, LruConfig, EvictionEvent, EvictionReason, EvictionCallback,
    // Multi-level sharding support (T-AS-4.5)
    ShardingDepth, ShardedStoreConfig, ShardedContentStore, ShardedStoreStats,
    ShardingMigrator, MigrationProgress, MigrationProgressCallback, MigrationResult,
};

pub use gltf_parser::{
    Aabb, BoundsResult, GltfDocument, GltfNode, GltfParser, GltfScene, GltfSkin,
    LoadStage, Mat4, StreamingBufferReader, ValidationError, ValidationResult,
    ValidationSeverity, load_gltf_streaming,
};

pub use draco::{
    // Draco extension types (T-AS-1.7)
    DracoError, DracoResult, DracoExtension, DracoDecompressResult,
    DracoBufferView, DracoGeometryType, DracoFallbackConfig, DracoFallbackResult,
    // Detection functions
    detect_draco_extension, is_draco_required, is_decoder_available,
    // Parsing functions
    parse_draco_extension, parse_draco_extension_value, get_draco_from_extensions,
    extract_buffer_view, extract_draco_data,
    // Decompression
    validate_draco_header, decompress_draco, decompress_draco_with_buffers,
    decompress_primitive_json, decompress_primitive_if_draco,
    check_draco_fallback, draco_result_to_primitive,
    // Utilities
    semantic_from_string, semantic_to_string, expected_draco_attr_type, estimate_decompressed_size,
    // Constants
    DRACO_EXTENSION_NAME, DRACO_ATTR_POSITION, DRACO_ATTR_NORMAL,
    DRACO_ATTR_COLOR, DRACO_ATTR_TEX_COORD, DRACO_ATTR_GENERIC,
};

pub use texture_importer::{
    GpuTextureFormat, MemoryBudgetTracker, TextureAsset, TextureImportError, TextureImporter,
    TextureMetadata, TextureState, SrgbHint, SourceFormat,
};

pub use hdr_formats::{
    // HDR format types
    HdrSourceFormat, HdrGpuFormat, HdrTextureAsset, HdrTextureMetadata, HdrTextureImporter,
    HdrImportConfig,
    // Tone mapping
    ToneMapOperator, DynamicRange, ColorPrimaries, ExrCompression,
    // GPU format selection
    select_hdr_gpu_format, convert_to_rgb10a2, convert_to_r11g11b10,
};

pub use vertex_format::{
    // Core types
    VertexFormatConverter, VertexFormatError, ImportSettings,
    // Layout types
    VertexLayout, VertexAttributeDescriptor, OutputFormat,
    // Output buffers
    InterleavedBuffer, SplitBuffers, CompressedBuffer, MergedMesh, MergedIndices, PrimitiveOffset,
    // Axis conversion
    AxisConversion,
    // Compression settings
    CompressionSettings, PositionCompression, NormalCompression, UvCompression, ColorCompression,
    // Compression utilities
    f32_to_f16, f16_to_f32, encode_octahedral, decode_octahedral,
    pack_10_10_10_2, unpack_10_10_10_2, pack_position_10_10_10_2, unpack_position_10_10_10_2,
    f32_to_snorm8, snorm8_to_f32, f32_to_unorm8, unorm8_to_f32, f32_to_unorm16, unorm16_to_f32,
};

pub use integrity::{
    // Configuration (T-AS-4.6)
    IntegrityConfig, CrcMetadata,
    // Error types
    IntegrityError, IntegrityResult,
    // CRC computation utilities
    compute_crc32c, Crc32cReader, Crc32cWriter,
    // Stores with CRC integrity
    MemoryIntegrityStore, FileIntegrityStore,
};

pub use sqlite_metadata::{
    // SQLite metadata backend (T-AS-4.4)
    SqliteMetadataStore, AssetMetadata, AssetType,
    MetadataError, MetadataResult, MetadataStats, PoolStats,
    // Optional backend with fallback
    OptionalMetadataBackend,
};

pub use compressed_formats::{
    // Compressed formats (T-AS-2.7)
    CompressedFormat, TextureType, MipLevelData, FaceData, CompressedTextureAsset,
    // KTX v1 parser
    KtxParser, KtxHeader, KtxTexture,
    // DDS parser
    DdsParser, DdsHeader, DdsDx10Header, DdsPixelFormat, DdsTexture,
    // Extension detection
    is_ktx_extension, is_dds_extension, detect_container_format,
};

pub use ktx2_parser::{
    // KTX2 parser (T-AS-2.3)
    Ktx2Parser, Ktx2Header, Ktx2Texture, Ktx2LevelIndex, Ktx2Dfd, Ktx2KeyValue,
    Ktx2TextureAsset,
    // Vulkan formats
    VkFormat, SupercompressionScheme,
    // Basis Universal
    BasisMode, BasisGlobalData, BasisTranscoder, TranscodeTarget,
    // Extension detection
    is_ktx2_extension, detect_ktx2_format,
    // Constants
    KTX2_IDENTIFIER,
};

pub use provenance::{
    // Provenance chain types (T-AS-4.7)
    ProvenanceChain, ProvenanceError, ProvenanceResult,
    ProcessingStep, ProcessingParam, ProcessingParams,
    // Cook configuration
    CookConfig, Platform, QualityLevel,
    // Dependencies
    DependencyRef, DependencyType,
    // Tree storage
    ProvenanceTreeNode, chain_to_tree,
    // Diffing
    ProvenanceDiff, diff_provenance,
    // Rebuild detection
    RebuildDecision, RebuildReason,
    // Constants
    PROVENANCE_SENTINEL, PROVENANCE_VERSION,
};

pub use mipmap::{
    // Mipmap generation (T-AS-2.4)
    FilterType, MipmapConfig, MipLevel, MipmapError,
    generate_mipmaps, calculate_mip_count, is_power_of_two, next_power_of_two,
    // Block compression
    CompressionFormat, CompressionQuality, CompressionConfig,
    CompressedTexture, compress_texture,
    // Cooked texture
    CookedTexture, cook_texture,
    // @cook decorator integration
    CookDecoratorParams,
};

pub use index_buffer::{
    // Index buffer optimization (T-AS-1.3)
    IndexType, IndexBufferConfig, IndexBufferError, OptimizedIndices, ReorderedMesh, StripResult,
    // Core functions
    select_index_type, select_index_type_min, optimize_vertex_cache, optimize_vertex_cache_with_size,
    compute_acmr, compute_acmr_lru, reorder_vertices, reorder_vertices_with_mapping,
    stripify, optimize_index_buffer, encode_indices, decode_indices,
    // Validation
    validate_indices, validate_triangle_count,
};

pub use cubemap::{
    // Cubemap and texture array assembly (T-AS-2.5)
    CubemapError, CubemapLayout, CubemapFace, CubemapConfig, CubemapTexture, CubemapFaceData,
    TextureData, EdgeDirection,
    // Layout detection
    detect_cubemap_layout,
    // Face extraction
    extract_faces_from_cross, extract_single_face,
    // Assembly
    assemble_cubemap, reorder_faces_to_gpu,
    // Seam-aware mips
    generate_seam_aware_mips,
    // KTX cubemap
    parse_ktx_cubemap,
    // Texture arrays
    TextureArrayConfig, TextureArray,
    create_texture_array, add_array_layer, generate_array_mips,
    // GPU flags
    VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
};

pub use delta_sync::{
    // Delta sync types (T-AS-4.8)
    DeltaSync, DeltaSyncConfig, DeltaSyncError, DeltaSyncResult,
    // Conflict resolution
    ConflictStrategy,
    // Sync manifest
    SyncManifest,
    // Delta proof
    DeltaProof,
    // Transfer types
    TransferChunk, TransferState,
    // Version vector for distributed sync
    VersionVector,
    // Utility functions
    compute_set_delta, estimate_transfer_size,
    // Constants
    DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, DEFAULT_MAX_CONCURRENT,
};

pub use virtual_texture::{
    // Virtual texturing (T-AS-2.6)
    VirtualTextureConfig, VirtualTextureData, VirtualTexturePage, VirtualTextureError,
    PageTableEntry, MipLevelPageInfo, DeduplicationStats,
    // Core functions
    extract_pages, generate_page_table, pack_for_archive, pack_with_header,
    // Morton ordering
    compute_morton_index, decode_morton_index,
    // Utilities
    calculate_mip_page_info, calculate_dedup_stats,
    // Constants
    DEFAULT_PAGE_SIZE, MIN_PAGE_SIZE, MAX_PAGE_SIZE, INVALID_PHYSICAL_PAGE,
};

pub use meshlet::{
    // Meshlet generation types (T-AS-1.4)
    Meshlet, MeshletMesh, MeshletConfig, MeshletError,
    // Bounding primitives
    BoundingSphere, NormalCone,
    // Core functions
    generate_meshlets, generate_meshlets_validated,
    compute_bounding_sphere, compute_bounding_sphere_indexed,
    compute_normal_cone,
    // Depth bounds
    compute_depth_bounds_indexed,
    // Cache reordering
    reorder_meshlet_for_cache,
    // Utility functions
    compute_centroid, normalize, dot, cross,
};

pub use blas::{
    // BLAS baking types (T-AS-1.5)
    BlasError, BlasResult, BlasBuildFlags, BlasGeometry, GeometryFlags,
    BlasConfig, BlasBuildResult, SerializedBlas,
    // Ray tracing decorator types
    RayTracingDecoratorParams, DecoratorValue,
    // Core functions
    prepare_blas_input, prepare_blas_input_raw,
    compute_blas_sizes, query_compacted_size,
    // Serialization
    serialize_blas, serialize_blas_with_metadata,
    deserialize_blas, deserialize_blas_from_bytes, deserialize_blas_with_metadata,
    // Skinned mesh support
    is_skinned_mesh, configure_for_skinned, update_skinned_positions,
    // Decorator parsing
    parse_ray_tracing_decorator, decorator_to_blas_config,
    // Multi-LOD support
    BlasLodConfig, create_multi_lod_blas,
    // Validation
    validate_geometry, validate_config,
    // Utility
    calculate_compaction_savings,
    // Constants
    BLAS_SERIALIZATION_VERSION, BLAS_MAGIC,
};

pub use lod::{
    // LOD types (T-AS-1.6)
    LodError, LodResult, MeshData, LodLevel, LodStrategy, LodConfig, LodChain,
    // Cross-fade
    DitherPattern, CrossFadeConfig,
    // Hierarchical LOD
    LodTree, LodTreeNode,
    // Nanite-style DAG
    ClusterDag, ClusterNode,
    // Quadric error metric
    Quadric, MeshSimplifier,
    // Core functions
    generate_lod_chain, simplify_mesh, compute_quadric_error,
    select_lod_level, compute_cross_fade_alpha, build_hierarchical_lod,
    // Decorator parsing
    LodDecoratorParams, parse_lod_decorator, parse_decorator_params, decorator_params_to_config,
};
