//! Virtual Geometry Subsystem for TRINITY Engine.
//!
//! Implements Nanite-style virtual geometry rendering with hierarchical
//! cluster culling and LOD selection. This subsystem enables rendering
//! of extremely high-polygon models (billions of triangles) at real-time
//! framerates by:
//!
//! 1. **Hierarchical Clustering**: Decomposing meshes into clusters (64-128 triangles)
//!    organized in a DAG hierarchy
//! 2. **Screen-Space Error Metrics**: Selecting appropriate LOD based on projected
//!    geometric error
//! 3. **GPU-Driven Culling**: Frustum, occlusion, and backface culling per cluster
//! 4. **Streaming**: Loading cluster data on demand based on visibility
//!
//! # Architecture
//!
//! ```text
//! +-------------------+
//! |   Mesh Import     | -> Build cluster hierarchy from source geometry
//! +-------------------+
//!          |
//!          v
//! +-------------------+
//! |   ClusterDAG      | -> CPU-side cluster hierarchy management
//! +-------------------+
//!          |
//!          v
//! +-------------------+
//! |  ClusterCull      | -> GPU cluster visibility and LOD selection
//! +-------------------+
//!          |
//!          v
//! +-------------------+
//! |  ClusterRender    | -> Rasterize visible clusters
//! +-------------------+
//! ```
//!
//! # Sub-modules
//!
//! - `cluster_cull` - Hierarchical cluster culling and LOD selection (T-GPU-8.1)
//! - `sw_rasterizer` - Software rasterizer in compute shaders (T-GPU-8.2)
//! - `virtual_lod` - Virtual geometry LOD system with streaming priority (T-GPU-8.3)
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::virtual_geometry::{
//!     ClusterDAG, ClusterNode, ClusterCullParams, ClusterCullPipeline,
//!     ClusterCullResources, cpu_traverse_dag,
//! };
//!
//! // Build cluster hierarchy
//! let mut dag = ClusterDAG::new();
//! let root = dag.add_root([0.0, 0.0, 0.0], 10.0, 1.0);
//! dag.add_child(root, [2.0, 0.0, 0.0], 5.0, 0.5);
//! dag.add_child(root, [-2.0, 0.0, 0.0], 5.0, 0.5);
//!
//! // Create GPU resources
//! let resources = ClusterCullResources::new(&device, dag.len() as u32);
//! let pipeline = ClusterCullPipeline::new(&device, &shader_source);
//!
//! // Each frame: upload data and dispatch
//! resources.upload_clusters(&queue, dag.clusters());
//! resources.upload_params(&queue, &params);
//! resources.clear_count(&queue);
//!
//! // Read visible clusters
//! let visible = cpu_traverse_dag(&dag, &planes, camera, screen_height, fov_tan, threshold, true, true);
//! ```
//!
//! # Performance
//!
//! - Target: <0.2ms for 10M clusters on modern GPUs
//! - Memory: 64 bytes per cluster node
//! - Workgroup size: 64 threads (persistent threads pattern)

pub mod cluster_cull;
pub mod sw_rasterizer;
pub mod virtual_lod;

// Re-exports for convenient access
pub use cluster_cull::{
    // Constants
    DEFAULT_ERROR_THRESHOLD, DEFAULT_MAX_CLUSTERS, FLAG_ACTIVE, FLAG_FORCE_DRAW,
    FLAG_LEAF, FLAG_ROOT, FLAG_STREAMING, INVALID_INDEX, MAX_CHILDREN,
    NUM_FRUSTUM_PLANES, WORKGROUP_SIZE,
    // Types
    ClusterCullParams, ClusterCullPipeline, ClusterCullResources, ClusterDAG,
    ClusterNode, FrustumPlane, VisibleCluster,
    // CPU reference implementations
    cpu_cluster_error, cpu_cone_cull, cpu_frustum_cull_sphere,
    cpu_select_cluster_lod, cpu_traverse_dag,
};

pub use sw_rasterizer::{
    // Constants
    DEFAULT_MAX_TRIANGLES, DEFAULT_TILE_SIZE, DEPTH_CLEAR_VALUE,
    FLAG_BACKFACE_CULL, FLAG_DEPTH_TEST, INVALID_INSTANCE_ID as SW_INVALID_INSTANCE_ID,
    INVALID_PRIMITIVE_ID as SW_INVALID_PRIMITIVE_ID, LINEAR_WORKGROUP_SIZE,
    MIN_TRIANGLE_AREA, TILE_WORKGROUP_SIZE,
    // Types
    BoundingBox, ClipSpaceTriangle, RasterizerParams, RasterizerResources,
    RasterizerStats, RasterizerTile, SoftwareRasterizerPipeline, TriangleMeta,
    // CPU reference implementations
    cpu_compute_barycentrics, cpu_decode_depth, cpu_depth_test, cpu_edge_function,
    cpu_encode_depth, cpu_interpolate_depth, cpu_interpolate_depth_perspective,
    cpu_is_backfacing, cpu_pack_visibility, cpu_project_vertex, cpu_rasterize_triangle,
    cpu_triangle_area, cpu_triangle_bbox, cpu_unpack_instance_id, cpu_unpack_primitive_id,
};

pub use virtual_lod::{
    // Parameters
    VirtualLODParams,
    // LOD level descriptor
    LODLevel,
    // Virtual mesh
    VirtualMesh,
    // Streaming priority
    StreamingPriority,
    // LOD result
    LODResult,
    // Page residency
    PageResidency,
    // Resources and pipeline
    VirtualLODResources,
    VirtualLODPipeline,
    // CPU reference functions
    cpu_screen_space_error,
    cpu_select_lod,
    cpu_lod_blend_factor,
    cpu_streaming_priority,
    // Constants
    WORKGROUP_SIZE as VIRTUAL_LOD_WORKGROUP_SIZE,
    MAX_LOD_LEVELS,
    MAX_INLINE_LODS,
    INVALID_LOD,
    FLAG_USE_DITHER,
    FLAG_FORCE_LOD,
    FLAG_DISABLE_STREAMING,
    FLAG_PAGE_TRACKING,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    PRIORITY_LOW,
    RESULT_FLAG_NEEDS_STREAMING,
    RESULT_FLAG_PAGE_RESIDENT,
};
