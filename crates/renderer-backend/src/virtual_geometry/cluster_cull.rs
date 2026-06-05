//! Nanite-Style Hierarchical Cluster Culling (T-GPU-8.1).
//!
//! Implements the core culling pass for virtual geometry rendering, providing:
//!
//! 1. **Hierarchical DAG Traversal**: BVH-like cluster hierarchy for LOD selection
//! 2. **Screen-Space Error Metric**: Project geometric error to pixels
//! 3. **Frustum Culling**: Per-cluster sphere bounds test
//! 4. **Occlusion Culling**: HZB-based visibility test
//! 5. **Normal Cone Culling**: Backface rejection using precomputed cones
//!
//! # Nanite-Style Virtual Geometry
//!
//! Virtual geometry systems like Nanite decompose high-polygon meshes into
//! clusters (groups of ~128 triangles) organized in a hierarchical DAG.
//! At runtime, the system selects which clusters to render based on:
//!
//! - **Screen-space error**: How much geometric detail is visible at current distance
//! - **Visibility**: Whether the cluster is in view and not occluded
//!
//! # DAG Structure
//!
//! ```text
//!                  [Root Cluster]           LOD 0 (coarsest)
//!                 /     |      \
//!         [Child]   [Child]   [Child]       LOD 1
//!        /    \        |
//!    [Leaf] [Leaf]  [Leaf]                  LOD 2 (finest)
//! ```
//!
//! - Root clusters cover entire meshes at low detail
//! - Leaf clusters contain the highest detail geometry
//! - Each level provides 2-4x more detail than the parent
//!
//! # Error Metric
//!
//! The error metric determines when to switch LOD levels:
//!
//! ```text
//! screen_error = (geometric_error * screen_height) / (2 * distance * tan(fov/2))
//! ```
//!
//! If `screen_error < threshold`, render this cluster.
//! If `screen_error >= threshold`, try children (if any).
//!
//! # Performance
//!
//! - Work complexity: O(visible_clusters)
//! - Target: <0.2ms for 10M clusters
//! - Memory: 64 bytes per cluster node
//!
//! # Usage
//!
//! ```ignore
//! // Create DAG and pipeline
//! let dag = ClusterDAG::new(&device, 1_000_000);
//! let pipeline = ClusterCullPipeline::new(&device, &shader_source);
//!
//! // Each frame
//! dag.upload_clusters(&queue, &clusters);
//! dag.upload_params(&queue, &params);
//! pipeline.dispatch(&mut encoder, &dag, num_clusters);
//!
//! // Read visible clusters
//! let visible = dag.read_visible_clusters(&device, &queue);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 64;

/// Maximum children per cluster node.
pub const MAX_CHILDREN: u32 = 4;

/// Number of frustum planes.
pub const NUM_FRUSTUM_PLANES: usize = 6;

/// Default maximum clusters.
pub const DEFAULT_MAX_CLUSTERS: u32 = 1_000_000;

/// Default error threshold in pixels.
pub const DEFAULT_ERROR_THRESHOLD: f32 = 1.0;

/// Cluster flag: Active/valid cluster.
pub const FLAG_ACTIVE: u32 = 1;

/// Cluster flag: Leaf node (no children).
pub const FLAG_LEAF: u32 = 2;

/// Cluster flag: Root node (no parent).
pub const FLAG_ROOT: u32 = 4;

/// Cluster flag: Data is streaming.
pub const FLAG_STREAMING: u32 = 8;

/// Cluster flag: Force draw regardless of LOD.
pub const FLAG_FORCE_DRAW: u32 = 16;

/// Invalid index sentinel value.
pub const INVALID_INDEX: u32 = 0xFFFFFFFF;

// ---------------------------------------------------------------------------
// ClusterNode
// ---------------------------------------------------------------------------

/// A node in the cluster DAG hierarchy.
///
/// Each cluster represents a group of triangles (typically 64-128) with
/// associated bounding information for culling and LOD selection.
///
/// # Memory Layout
///
/// 64 bytes, 16-byte aligned:
/// | Offset | Field             | Size | Description                        |
/// |--------|-------------------|------|------------------------------------|
/// | 0      | bounds_center     | 12   | Bounding sphere center             |
/// | 12     | bounds_radius     | 4    | Bounding sphere radius             |
/// | 16     | normal_cone_axis  | 12   | Normal cone axis                   |
/// | 28     | normal_cone_cutoff| 4    | Normal cone cutoff                 |
/// | 32     | error_metric      | 4    | Geometric error (object space)     |
/// | 36     | parent_index      | 4    | Parent cluster (-1 for root)       |
/// | 40     | first_child       | 4    | Index of first child               |
/// | 44     | child_count       | 4    | Number of children (0-4)           |
/// | 48     | lod_level         | 4    | LOD level (0 = coarsest)           |
/// | 52     | flags             | 4    | Status flags                       |
/// | 56     | _pad              | 8    | Padding to 64 bytes                |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterNode {
    /// Center of bounding sphere in world/object space.
    pub bounds_center: [f32; 3],
    /// Radius of bounding sphere.
    pub bounds_radius: f32,
    /// Normal cone axis (average normal direction, normalized).
    pub normal_cone_axis: [f32; 3],
    /// Normal cone cutoff: cos(half_angle). Values > 1.0 disable cone test.
    pub normal_cone_cutoff: f32,
    /// Geometric error in object space. Represents the maximum deviation
    /// from the full-resolution geometry when using this cluster's LOD.
    pub error_metric: f32,
    /// Parent cluster index. -1 for root clusters.
    pub parent_index: i32,
    /// Index of first child cluster. Only valid if child_count > 0.
    pub first_child: u32,
    /// Number of children (0-4). 0 indicates a leaf cluster.
    pub child_count: u32,
    /// LOD level: 0 = coarsest (root), increases with detail level.
    pub lod_level: u32,
    /// Status flags (FLAG_ACTIVE, FLAG_LEAF, FLAG_ROOT, etc.).
    pub flags: u32,
    /// Padding for 64-byte alignment.
    pub _pad0: u32,
    pub _pad1: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ClusterNode>() == 64);

impl ClusterNode {
    /// Create a new cluster node.
    pub fn new(
        bounds_center: [f32; 3],
        bounds_radius: f32,
        error_metric: f32,
        lod_level: u32,
    ) -> Self {
        Self {
            bounds_center,
            bounds_radius,
            normal_cone_axis: [0.0, 0.0, 1.0],
            normal_cone_cutoff: 2.0, // Disabled
            error_metric,
            parent_index: -1,
            first_child: INVALID_INDEX,
            child_count: 0,
            lod_level,
            flags: FLAG_ACTIVE | FLAG_LEAF | FLAG_ROOT,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create a root cluster.
    pub fn root(bounds_center: [f32; 3], bounds_radius: f32, error_metric: f32) -> Self {
        let mut node = Self::new(bounds_center, bounds_radius, error_metric, 0);
        node.flags = FLAG_ACTIVE | FLAG_ROOT;
        node
    }

    /// Create a leaf cluster.
    pub fn leaf(
        bounds_center: [f32; 3],
        bounds_radius: f32,
        error_metric: f32,
        parent_index: i32,
        lod_level: u32,
    ) -> Self {
        Self {
            bounds_center,
            bounds_radius,
            normal_cone_axis: [0.0, 0.0, 1.0],
            normal_cone_cutoff: 2.0,
            error_metric,
            parent_index,
            first_child: INVALID_INDEX,
            child_count: 0,
            lod_level,
            flags: FLAG_ACTIVE | FLAG_LEAF,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create an internal (non-leaf) cluster.
    pub fn internal(
        bounds_center: [f32; 3],
        bounds_radius: f32,
        error_metric: f32,
        parent_index: i32,
        first_child: u32,
        child_count: u32,
        lod_level: u32,
    ) -> Self {
        Self {
            bounds_center,
            bounds_radius,
            normal_cone_axis: [0.0, 0.0, 1.0],
            normal_cone_cutoff: 2.0,
            error_metric,
            parent_index,
            first_child,
            child_count,
            lod_level,
            flags: FLAG_ACTIVE,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Set the normal cone for backface culling.
    pub fn with_normal_cone(mut self, axis: [f32; 3], cutoff: f32) -> Self {
        self.normal_cone_axis = normalize_vec3(axis);
        self.normal_cone_cutoff = cutoff;
        self
    }

    /// Check if this is a root cluster.
    #[inline]
    pub fn is_root(&self) -> bool {
        (self.flags & FLAG_ROOT) != 0
    }

    /// Check if this is a leaf cluster.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        (self.flags & FLAG_LEAF) != 0 || self.child_count == 0
    }

    /// Check if this cluster is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        (self.flags & FLAG_ACTIVE) != 0
    }

    /// Get child indices as a slice of valid indices.
    pub fn children(&self) -> impl Iterator<Item = u32> {
        let first = self.first_child;
        let count = self.child_count;
        (0..count).map(move |i| first + i)
    }
}

// ---------------------------------------------------------------------------
// ClusterCullParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for cluster culling parameters.
///
/// # Memory Layout
///
/// 128 bytes, std140 compatible:
/// | Offset | Field              | Size |
/// |--------|--------------------|----- |
/// | 0      | view_proj          | 64   |
/// | 64     | camera_position    | 12   |
/// | 76     | screen_height      | 4    |
/// | 80     | fov_y_half_tan     | 4    |
/// | 84     | error_threshold    | 4    |
/// | 88     | hzb_width          | 4    |
/// | 92     | hzb_height         | 4    |
/// | 96     | num_mips           | 4    |
/// | 100    | num_clusters       | 4    |
/// | 104    | enable_frustum     | 4    |
/// | 108    | enable_occlusion   | 4    |
/// | 112    | enable_cone        | 4    |
/// | 116    | _pad               | 12   |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterCullParams {
    /// Combined view-projection matrix (column-major).
    pub view_proj: [[f32; 4]; 4],
    /// Camera position in world space.
    pub camera_position: [f32; 3],
    /// Screen height in pixels.
    pub screen_height: f32,
    /// Half vertical FOV tangent: tan(fov_y / 2).
    pub fov_y_half_tan: f32,
    /// Error threshold in pixels.
    pub error_threshold: f32,
    /// HZB texture width (mip 0).
    pub hzb_width: u32,
    /// HZB texture height (mip 0).
    pub hzb_height: u32,
    /// Number of HZB mip levels.
    pub num_mips: u32,
    /// Total number of clusters.
    pub num_clusters: u32,
    /// Enable frustum culling (1 = enabled).
    pub enable_frustum: u32,
    /// Enable HZB occlusion culling (1 = enabled).
    pub enable_occlusion: u32,
    /// Enable normal cone culling (1 = enabled).
    pub enable_cone: u32,
    /// Padding.
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ClusterCullParams>() == 128);

impl Default for ClusterCullParams {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_position: [0.0, 0.0, 0.0],
            screen_height: 1080.0,
            fov_y_half_tan: (45.0_f32.to_radians() / 2.0).tan(),
            error_threshold: DEFAULT_ERROR_THRESHOLD,
            hzb_width: 0,
            hzb_height: 0,
            num_mips: 0,
            num_clusters: 0,
            enable_frustum: 1,
            enable_occlusion: 0,
            enable_cone: 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

impl ClusterCullParams {
    /// Create culling parameters.
    pub fn new(
        view_proj: &[[f32; 4]; 4],
        camera_position: [f32; 3],
        screen_height: f32,
        fov_y_radians: f32,
        num_clusters: u32,
    ) -> Self {
        Self {
            view_proj: *view_proj,
            camera_position,
            screen_height,
            fov_y_half_tan: (fov_y_radians / 2.0).tan(),
            error_threshold: DEFAULT_ERROR_THRESHOLD,
            hzb_width: 0,
            hzb_height: 0,
            num_mips: 0,
            num_clusters,
            enable_frustum: 1,
            enable_occlusion: 0,
            enable_cone: 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    /// Create parameters with HZB occlusion culling enabled.
    pub fn with_hzb(
        view_proj: &[[f32; 4]; 4],
        camera_position: [f32; 3],
        screen_height: f32,
        fov_y_radians: f32,
        num_clusters: u32,
        hzb_width: u32,
        hzb_height: u32,
        num_mips: u32,
    ) -> Self {
        Self {
            view_proj: *view_proj,
            camera_position,
            screen_height,
            fov_y_half_tan: (fov_y_radians / 2.0).tan(),
            error_threshold: DEFAULT_ERROR_THRESHOLD,
            hzb_width,
            hzb_height,
            num_mips,
            num_clusters,
            enable_frustum: 1,
            enable_occlusion: 1,
            enable_cone: 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    /// Set the error threshold in pixels.
    pub fn set_error_threshold(&mut self, threshold: f32) {
        self.error_threshold = threshold;
    }

    /// Enable or disable frustum culling.
    pub fn set_frustum_cull(&mut self, enabled: bool) {
        self.enable_frustum = if enabled { 1 } else { 0 };
    }

    /// Enable or disable occlusion culling.
    pub fn set_occlusion_cull(&mut self, enabled: bool) {
        self.enable_occlusion = if enabled { 1 } else { 0 };
    }

    /// Enable or disable cone culling.
    pub fn set_cone_cull(&mut self, enabled: bool) {
        self.enable_cone = if enabled { 1 } else { 0 };
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_clusters + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

// ---------------------------------------------------------------------------
// FrustumPlane
// ---------------------------------------------------------------------------

/// A frustum plane in Hessian normal form.
///
/// # Memory Layout
///
/// 16 bytes, vec4 aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrustumPlane {
    /// Normalized plane normal (pointing into frustum).
    pub normal: [f32; 3],
    /// Signed distance from origin.
    pub distance: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<FrustumPlane>() == 16);

impl FrustumPlane {
    /// Create a new frustum plane.
    pub fn new(normal: [f32; 3], distance: f32) -> Self {
        let normalized = normalize_vec3(normal);
        let len = vec3_length(normal);
        let dist = if len > 1e-8 { distance / len } else { 0.0 };
        Self {
            normal: normalized,
            distance: dist,
        }
    }

    /// Compute signed distance from a point to the plane.
    #[inline]
    pub fn distance_to_point(&self, point: [f32; 3]) -> f32 {
        self.normal[0] * point[0]
            + self.normal[1] * point[1]
            + self.normal[2] * point[2]
            + self.distance
    }
}

// ---------------------------------------------------------------------------
// VisibleCluster
// ---------------------------------------------------------------------------

/// Output entry for a visible cluster.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisibleCluster {
    /// Index of the visible cluster.
    pub cluster_index: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VisibleCluster>() == 4);

// ---------------------------------------------------------------------------
// ClusterDAG
// ---------------------------------------------------------------------------

/// Manages the cluster hierarchy DAG.
///
/// Provides CPU-side management for the cluster hierarchy including:
/// - Cluster storage and indexing
/// - Parent-child relationship tracking
/// - LOD level organization
pub struct ClusterDAG {
    /// All cluster nodes.
    clusters: Vec<ClusterNode>,
    /// Root cluster indices.
    roots: Vec<u32>,
    /// Maximum LOD level in the DAG.
    max_lod_level: u32,
}

impl ClusterDAG {
    /// Create a new empty cluster DAG.
    pub fn new() -> Self {
        Self {
            clusters: Vec::new(),
            roots: Vec::new(),
            max_lod_level: 0,
        }
    }

    /// Create a DAG with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            clusters: Vec::with_capacity(capacity),
            roots: Vec::new(),
            max_lod_level: 0,
        }
    }

    /// Add a cluster and return its index.
    pub fn add_cluster(&mut self, cluster: ClusterNode) -> u32 {
        let index = self.clusters.len() as u32;
        if cluster.is_root() {
            self.roots.push(index);
        }
        self.max_lod_level = self.max_lod_level.max(cluster.lod_level);
        self.clusters.push(cluster);
        index
    }

    /// Add a root cluster.
    pub fn add_root(&mut self, bounds_center: [f32; 3], bounds_radius: f32, error_metric: f32) -> u32 {
        let cluster = ClusterNode::root(bounds_center, bounds_radius, error_metric);
        self.add_cluster(cluster)
    }

    /// Add a child cluster to an existing parent.
    pub fn add_child(
        &mut self,
        parent_index: u32,
        bounds_center: [f32; 3],
        bounds_radius: f32,
        error_metric: f32,
    ) -> u32 {
        let parent_lod = self.clusters[parent_index as usize].lod_level;
        let child = ClusterNode::leaf(
            bounds_center,
            bounds_radius,
            error_metric,
            parent_index as i32,
            parent_lod + 1,
        );
        let child_index = self.add_cluster(child);

        // Update parent's child info
        let parent = &mut self.clusters[parent_index as usize];
        if parent.child_count == 0 {
            parent.first_child = child_index;
            parent.flags &= !FLAG_LEAF; // No longer a leaf
        }
        parent.child_count += 1;

        child_index
    }

    /// Get cluster at index.
    pub fn get(&self, index: u32) -> Option<&ClusterNode> {
        self.clusters.get(index as usize)
    }

    /// Get mutable cluster at index.
    pub fn get_mut(&mut self, index: u32) -> Option<&mut ClusterNode> {
        self.clusters.get_mut(index as usize)
    }

    /// Get all clusters as a slice.
    pub fn clusters(&self) -> &[ClusterNode] {
        &self.clusters
    }

    /// Get root cluster indices.
    pub fn roots(&self) -> &[u32] {
        &self.roots
    }

    /// Get the number of clusters.
    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    /// Check if the DAG is empty.
    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }

    /// Get maximum LOD level.
    pub fn max_lod_level(&self) -> u32 {
        self.max_lod_level
    }

    /// Clear all clusters.
    pub fn clear(&mut self) {
        self.clusters.clear();
        self.roots.clear();
        self.max_lod_level = 0;
    }

    /// Traverse the DAG from roots, calling visitor for each cluster.
    pub fn traverse<F>(&self, mut visitor: F)
    where
        F: FnMut(u32, &ClusterNode, Option<u32>), // (index, cluster, parent_index)
    {
        let mut stack: Vec<(u32, Option<u32>)> = self.roots.iter().map(|&r| (r, None)).collect();

        while let Some((index, parent)) = stack.pop() {
            if let Some(cluster) = self.get(index) {
                visitor(index, cluster, parent);

                // Add children to stack
                for child_idx in cluster.children() {
                    stack.push((child_idx, Some(index)));
                }
            }
        }
    }
}

impl Default for ClusterDAG {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ClusterCullResources
// ---------------------------------------------------------------------------

/// GPU resources for cluster culling.
pub struct ClusterCullResources {
    /// Uniform buffer for culling parameters.
    pub params_buffer: wgpu::Buffer,
    /// Storage buffer for cluster nodes.
    pub clusters_buffer: wgpu::Buffer,
    /// Storage buffer for frustum planes.
    pub planes_buffer: wgpu::Buffer,
    /// Storage buffer for visible cluster output.
    pub visible_buffer: wgpu::Buffer,
    /// Storage buffer for visible count atomic.
    pub count_buffer: wgpu::Buffer,
    /// Staging buffer for reading visible count.
    pub count_staging: wgpu::Buffer,
    /// Staging buffer for reading visible clusters.
    pub visible_staging: wgpu::Buffer,
    /// Maximum number of clusters.
    pub max_clusters: u32,
}

impl ClusterCullResources {
    /// Create cluster culling resources.
    pub fn new(device: &wgpu::Device, max_clusters: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_cull_params"),
            size: mem::size_of::<ClusterCullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let clusters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_cull_clusters"),
            size: (max_clusters as u64) * (mem::size_of::<ClusterNode>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let planes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_cull_planes"),
            size: (NUM_FRUSTUM_PLANES as u64) * (mem::size_of::<FrustumPlane>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_cull_visible"),
            size: (max_clusters as u64) * (mem::size_of::<VisibleCluster>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_cull_count"),
            size: 4, // Single atomic u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_cull_count_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_cull_visible_staging"),
            size: (max_clusters as u64) * (mem::size_of::<VisibleCluster>() as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            clusters_buffer,
            planes_buffer,
            visible_buffer,
            count_buffer,
            count_staging,
            visible_staging,
            max_clusters,
        }
    }

    /// Upload culling parameters.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &ClusterCullParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload cluster nodes.
    pub fn upload_clusters(&self, queue: &wgpu::Queue, clusters: &[ClusterNode]) {
        assert!(clusters.len() <= self.max_clusters as usize);
        queue.write_buffer(&self.clusters_buffer, 0, bytemuck::cast_slice(clusters));
    }

    /// Upload frustum planes.
    pub fn upload_planes(&self, queue: &wgpu::Queue, planes: &[FrustumPlane; 6]) {
        queue.write_buffer(&self.planes_buffer, 0, bytemuck::cast_slice(planes));
    }

    /// Clear the visible count before dispatch.
    pub fn clear_count(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.count_buffer, 0, &[0u8; 4]);
    }
}

// ---------------------------------------------------------------------------
// ClusterCullPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for cluster culling.
pub struct ClusterCullPipeline {
    /// Main culling pipeline.
    pub pipeline: wgpu::ComputePipeline,
    /// Roots-only pipeline.
    pub pipeline_roots: wgpu::ComputePipeline,
    /// Frustum-only pipeline.
    pub pipeline_frustum_only: wgpu::ComputePipeline,
    /// LOD selection only pipeline.
    pub pipeline_lod_select: wgpu::ComputePipeline,
    /// Hierarchical traversal pipeline.
    pub pipeline_hierarchical: wgpu::ComputePipeline,
    /// Bind group layout.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout with HZB.
    pub bind_group_layout_with_hzb: wgpu::BindGroupLayout,
}

impl ClusterCullPipeline {
    /// Create the cluster culling pipeline.
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster_cull_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Bind group layout without HZB
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster_cull_bind_group_layout"),
            entries: &[
                // @binding(0) params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<ClusterCullParams>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) clusters
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2) frustum_planes
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3) visible_clusters
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(4) visible_count
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Bind group layout with HZB
        let bind_group_layout_with_hzb =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cluster_cull_bind_group_layout_with_hzb"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    mem::size_of::<ClusterCullParams>() as u64,
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(5) hzb_texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cluster_cull_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_layout_with_hzb =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cluster_cull_pipeline_layout_with_hzb"),
                bind_group_layouts: &[&bind_group_layout_with_hzb],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cluster_cull_pipeline"),
            layout: Some(&pipeline_layout_with_hzb),
            module: &shader_module,
            entry_point: "cluster_cull",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_roots = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cluster_cull_pipeline_roots"),
            layout: Some(&pipeline_layout_with_hzb),
            module: &shader_module,
            entry_point: "cluster_cull_roots",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_frustum_only =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cluster_cull_pipeline_frustum_only"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "cluster_cull_frustum_only",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let pipeline_lod_select =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cluster_cull_pipeline_lod_select"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "cluster_select_lod",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let pipeline_hierarchical =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cluster_cull_pipeline_hierarchical"),
                layout: Some(&pipeline_layout_with_hzb),
                module: &shader_module,
                entry_point: "cluster_cull_hierarchical",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            pipeline,
            pipeline_roots,
            pipeline_frustum_only,
            pipeline_lod_select,
            pipeline_hierarchical,
            bind_group_layout,
            bind_group_layout_with_hzb,
        }
    }

    /// Create a bind group for the given resources (without HZB).
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &ClusterCullResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster_cull_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.planes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.visible_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create a bind group with HZB texture.
    pub fn create_bind_group_with_hzb(
        &self,
        device: &wgpu::Device,
        resources: &ClusterCullResources,
        hzb_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster_cull_bind_group_with_hzb"),
            layout: &self.bind_group_layout_with_hzb,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.planes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.visible_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(hzb_view),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// Compute screen-space error in pixels.
///
/// Formula:
/// ```text
/// screen_error = (object_error * screen_height) / (2 * distance * tan(fov_y/2))
/// ```
pub fn cpu_cluster_error(
    object_error: f32,
    cluster_center: [f32; 3],
    camera_position: [f32; 3],
    screen_height: f32,
    fov_y_half_tan: f32,
) -> f32 {
    let dx = cluster_center[0] - camera_position[0];
    let dy = cluster_center[1] - camera_position[1];
    let dz = cluster_center[2] - camera_position[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();

    if distance < 1e-6 {
        return object_error * screen_height; // Very close
    }

    object_error * screen_height / (2.0 * distance * fov_y_half_tan)
}

/// Select whether a cluster should be rendered based on error metric.
///
/// Returns `true` if this cluster's error is acceptable (should render).
pub fn cpu_select_cluster_lod(
    cluster: &ClusterNode,
    camera_position: [f32; 3],
    screen_height: f32,
    fov_y_half_tan: f32,
    error_threshold: f32,
) -> bool {
    // Force draw always passes
    if (cluster.flags & FLAG_FORCE_DRAW) != 0 {
        return true;
    }

    let screen_error = cpu_cluster_error(
        cluster.error_metric,
        cluster.bounds_center,
        camera_position,
        screen_height,
        fov_y_half_tan,
    );

    screen_error < error_threshold
}

/// Frustum cull a bounding sphere.
pub fn cpu_frustum_cull_sphere(
    center: [f32; 3],
    radius: f32,
    planes: &[FrustumPlane; 6],
) -> bool {
    for plane in planes {
        let dist = plane.distance_to_point(center);
        if dist < -radius {
            return false;
        }
    }
    true
}

/// Normal cone backface culling.
///
/// Returns `true` if cluster should be CULLED (is backfacing).
pub fn cpu_cone_cull(
    center: [f32; 3],
    cone_axis: [f32; 3],
    cone_cutoff: f32,
    camera_pos: [f32; 3],
) -> bool {
    if cone_cutoff >= 1.0 {
        return false;
    }

    let to_camera = [
        camera_pos[0] - center[0],
        camera_pos[1] - center[1],
        camera_pos[2] - center[2],
    ];

    let dist_sq = to_camera[0] * to_camera[0]
        + to_camera[1] * to_camera[1]
        + to_camera[2] * to_camera[2];

    if dist_sq < 1e-8 {
        return false;
    }

    let inv_dist = 1.0 / dist_sq.sqrt();
    let view_dir = [
        to_camera[0] * inv_dist,
        to_camera[1] * inv_dist,
        to_camera[2] * inv_dist,
    ];

    let cone_dot =
        view_dir[0] * cone_axis[0] + view_dir[1] * cone_axis[1] + view_dir[2] * cone_axis[2];

    cone_dot < -cone_cutoff
}

/// Traverse the DAG and select visible clusters.
///
/// This CPU reference implementation mirrors the GPU algorithm:
/// 1. Start from roots
/// 2. For each cluster, check visibility (frustum, cone)
/// 3. If visible and error is acceptable, add to output
/// 4. If visible but error too high, recurse to children
pub fn cpu_traverse_dag(
    dag: &ClusterDAG,
    planes: &[FrustumPlane; 6],
    camera_position: [f32; 3],
    screen_height: f32,
    fov_y_half_tan: f32,
    error_threshold: f32,
    enable_frustum: bool,
    enable_cone: bool,
) -> Vec<u32> {
    let mut visible = Vec::new();
    let mut stack: Vec<u32> = dag.roots().to_vec();

    while let Some(index) = stack.pop() {
        let cluster = match dag.get(index) {
            Some(c) => c,
            None => continue,
        };

        // Check if active
        if !cluster.is_active() {
            continue;
        }

        // Frustum culling
        if enable_frustum {
            if !cpu_frustum_cull_sphere(cluster.bounds_center, cluster.bounds_radius, planes) {
                continue;
            }
        }

        // Cone culling
        if enable_cone {
            if cpu_cone_cull(
                cluster.bounds_center,
                cluster.normal_cone_axis,
                cluster.normal_cone_cutoff,
                camera_position,
            ) {
                continue;
            }
        }

        // LOD selection
        let should_render = cpu_select_cluster_lod(
            cluster,
            camera_position,
            screen_height,
            fov_y_half_tan,
            error_threshold,
        );

        if cluster.is_leaf() || should_render {
            visible.push(index);
        } else {
            // Recurse to children
            for child_idx in cluster.children() {
                stack.push(child_idx);
            }
        }
    }

    visible
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Normalize a 3D vector.
fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let len = vec3_length(v);
    if len < 1e-8 {
        return [0.0, 0.0, 1.0];
    }
    let inv = 1.0 / len;
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Compute length of a 3D vector.
fn vec3_length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // Helper: Create a test frustum (camera at origin, looking down -Z, 90-degree FOV)
    fn make_test_frustum() -> [FrustumPlane; 6] {
        [
            FrustumPlane::new([1.0, 0.0, -1.0], 0.0),   // Left
            FrustumPlane::new([-1.0, 0.0, -1.0], 0.0),  // Right
            FrustumPlane::new([0.0, 1.0, -1.0], 0.0),   // Bottom
            FrustumPlane::new([0.0, -1.0, -1.0], 0.0),  // Top
            FrustumPlane::new([0.0, 0.0, -1.0], -1.0),  // Near
            FrustumPlane::new([0.0, 0.0, 1.0], 100.0),  // Far
        ]
    }

    // -------------------------------------------------------------------------
    // Struct Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cluster_node_size() {
        assert_eq!(
            mem::size_of::<ClusterNode>(),
            64,
            "ClusterNode must be 64 bytes"
        );
    }

    #[test]
    fn test_cluster_cull_params_size() {
        assert_eq!(
            mem::size_of::<ClusterCullParams>(),
            128,
            "ClusterCullParams must be 128 bytes"
        );
    }

    #[test]
    fn test_frustum_plane_size() {
        assert_eq!(
            mem::size_of::<FrustumPlane>(),
            16,
            "FrustumPlane must be 16 bytes"
        );
    }

    #[test]
    fn test_visible_cluster_size() {
        assert_eq!(
            mem::size_of::<VisibleCluster>(),
            4,
            "VisibleCluster must be 4 bytes"
        );
    }

    // -------------------------------------------------------------------------
    // ClusterNode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cluster_node_new() {
        let node = ClusterNode::new([0.0, 0.0, 0.0], 1.0, 0.1, 0);

        assert_eq!(node.bounds_center, [0.0, 0.0, 0.0]);
        assert_eq!(node.bounds_radius, 1.0);
        assert_eq!(node.error_metric, 0.1);
        assert_eq!(node.lod_level, 0);
        assert!(node.is_active());
        assert!(node.is_leaf());
        assert!(node.is_root());
    }

    #[test]
    fn test_cluster_node_root() {
        let root = ClusterNode::root([1.0, 2.0, 3.0], 5.0, 0.5);

        assert!(root.is_root());
        // A root with no children is considered a leaf (child_count == 0)
        assert!(root.is_leaf());
        assert!((root.flags & FLAG_LEAF) == 0); // But FLAG_LEAF is not explicitly set
        assert!(root.is_active());
        assert_eq!(root.parent_index, -1);
        assert_eq!(root.lod_level, 0);
    }

    #[test]
    fn test_cluster_node_leaf() {
        let leaf = ClusterNode::leaf([0.0, 0.0, -10.0], 0.5, 0.01, 0, 2);

        assert!(leaf.is_leaf());
        assert!(!leaf.is_root());
        assert_eq!(leaf.parent_index, 0);
        assert_eq!(leaf.lod_level, 2);
        assert_eq!(leaf.child_count, 0);
    }

    #[test]
    fn test_cluster_node_internal() {
        let internal = ClusterNode::internal(
            [0.0, 0.0, -5.0],
            2.0,
            0.2,
            0,
            5,
            3,
            1,
        );

        assert!(!internal.is_leaf());
        assert!(!internal.is_root());
        assert_eq!(internal.parent_index, 0);
        assert_eq!(internal.first_child, 5);
        assert_eq!(internal.child_count, 3);
        assert_eq!(internal.lod_level, 1);
    }

    #[test]
    fn test_cluster_node_with_normal_cone() {
        let node = ClusterNode::new([0.0, 0.0, 0.0], 1.0, 0.1, 0)
            .with_normal_cone([0.0, 1.0, 0.0], 0.5);

        assert!((node.normal_cone_axis[1] - 1.0).abs() < 1e-6);
        assert_eq!(node.normal_cone_cutoff, 0.5);
    }

    #[test]
    fn test_cluster_node_children() {
        let node = ClusterNode::internal([0.0, 0.0, 0.0], 1.0, 0.1, -1, 10, 4, 0);
        let children: Vec<u32> = node.children().collect();

        assert_eq!(children, vec![10, 11, 12, 13]);
    }

    #[test]
    fn test_cluster_node_no_children() {
        let leaf = ClusterNode::leaf([0.0, 0.0, 0.0], 1.0, 0.1, 0, 1);
        let children: Vec<u32> = leaf.children().collect();

        assert!(children.is_empty());
    }

    // -------------------------------------------------------------------------
    // ClusterCullParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_params_default() {
        let params = ClusterCullParams::default();

        assert_eq!(params.screen_height, 1080.0);
        assert_eq!(params.error_threshold, DEFAULT_ERROR_THRESHOLD);
        assert_eq!(params.enable_frustum, 1);
        assert_eq!(params.enable_occlusion, 0);
        assert_eq!(params.enable_cone, 1);
    }

    #[test]
    fn test_params_new() {
        let vp = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let params = ClusterCullParams::new(&vp, [0.0, 0.0, 0.0], 720.0, PI / 2.0, 1000);

        assert_eq!(params.screen_height, 720.0);
        assert_eq!(params.num_clusters, 1000);
        assert!((params.fov_y_half_tan - 1.0).abs() < 0.01); // tan(45deg) ~ 1
    }

    #[test]
    fn test_params_with_hzb() {
        let vp = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let params = ClusterCullParams::with_hzb(&vp, [0.0, 0.0, 0.0], 1080.0, PI / 2.0, 1000, 1920, 1080, 10);

        assert_eq!(params.enable_occlusion, 1);
        assert_eq!(params.hzb_width, 1920);
        assert_eq!(params.hzb_height, 1080);
        assert_eq!(params.num_mips, 10);
    }

    #[test]
    fn test_params_set_flags() {
        let mut params = ClusterCullParams::default();

        params.set_frustum_cull(false);
        params.set_occlusion_cull(true);
        params.set_cone_cull(false);
        params.set_error_threshold(2.0);

        assert_eq!(params.enable_frustum, 0);
        assert_eq!(params.enable_occlusion, 1);
        assert_eq!(params.enable_cone, 0);
        assert_eq!(params.error_threshold, 2.0);
    }

    #[test]
    fn test_params_num_workgroups() {
        let mut params = ClusterCullParams::default();

        params.num_clusters = 64;
        assert_eq!(params.num_workgroups(), 1);

        params.num_clusters = 65;
        assert_eq!(params.num_workgroups(), 2);

        params.num_clusters = 128;
        assert_eq!(params.num_workgroups(), 2);

        params.num_clusters = 129;
        assert_eq!(params.num_workgroups(), 3);
    }

    // -------------------------------------------------------------------------
    // FrustumPlane Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frustum_plane_new() {
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 5.0);

        assert!((plane.normal[2] - 1.0).abs() < 1e-6);
        assert!((plane.distance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_frustum_plane_distance_to_point() {
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);

        assert!((plane.distance_to_point([0.0, 0.0, 5.0]) - 5.0).abs() < 1e-6);
        assert!((plane.distance_to_point([0.0, 0.0, -3.0]) - (-3.0)).abs() < 1e-6);
        assert!(plane.distance_to_point([0.0, 0.0, 0.0]).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // ClusterDAG Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dag_new() {
        let dag = ClusterDAG::new();

        assert!(dag.is_empty());
        assert_eq!(dag.len(), 0);
        assert!(dag.roots().is_empty());
    }

    #[test]
    fn test_dag_add_root() {
        let mut dag = ClusterDAG::new();
        let idx = dag.add_root([0.0, 0.0, 0.0], 10.0, 1.0);

        assert_eq!(idx, 0);
        assert_eq!(dag.len(), 1);
        assert_eq!(dag.roots(), &[0]);

        let cluster = dag.get(0).unwrap();
        assert!(cluster.is_root());
    }

    #[test]
    fn test_dag_add_child() {
        let mut dag = ClusterDAG::new();
        let root = dag.add_root([0.0, 0.0, 0.0], 10.0, 1.0);
        let child1 = dag.add_child(root, [1.0, 0.0, 0.0], 5.0, 0.5);
        let child2 = dag.add_child(root, [-1.0, 0.0, 0.0], 5.0, 0.5);

        assert_eq!(dag.len(), 3);
        assert_eq!(child1, 1);
        assert_eq!(child2, 2);

        let root_node = dag.get(root).unwrap();
        assert!(!root_node.is_leaf());
        assert_eq!(root_node.first_child, 1);
        assert_eq!(root_node.child_count, 2);

        let child_node = dag.get(child1).unwrap();
        assert!(child_node.is_leaf());
        assert_eq!(child_node.parent_index, root as i32);
        assert_eq!(child_node.lod_level, 1);
    }

    #[test]
    fn test_dag_max_lod_level() {
        let mut dag = ClusterDAG::new();
        let root = dag.add_root([0.0, 0.0, 0.0], 10.0, 1.0);
        let child = dag.add_child(root, [0.0, 0.0, 0.0], 5.0, 0.5);
        dag.add_child(child, [0.0, 0.0, 0.0], 2.5, 0.25);

        assert_eq!(dag.max_lod_level(), 2);
    }

    #[test]
    fn test_dag_traverse() {
        let mut dag = ClusterDAG::new();
        let root = dag.add_root([0.0, 0.0, 0.0], 10.0, 1.0);
        dag.add_child(root, [1.0, 0.0, 0.0], 5.0, 0.5);
        dag.add_child(root, [-1.0, 0.0, 0.0], 5.0, 0.5);

        let mut visited = Vec::new();
        dag.traverse(|idx, _cluster, _parent| {
            visited.push(idx);
        });

        assert_eq!(visited.len(), 3);
        assert!(visited.contains(&0));
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
    }

    #[test]
    fn test_dag_clear() {
        let mut dag = ClusterDAG::new();
        dag.add_root([0.0, 0.0, 0.0], 10.0, 1.0);
        dag.add_root([10.0, 0.0, 0.0], 10.0, 1.0);

        assert_eq!(dag.len(), 2);
        dag.clear();
        assert!(dag.is_empty());
        assert!(dag.roots().is_empty());
    }

    // -------------------------------------------------------------------------
    // Error Metric Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_cluster_error() {
        let camera = [0.0, 0.0, 0.0];
        let screen_height = 1080.0;
        let fov_half_tan = 1.0; // 90 degree FOV

        // Cluster at distance 10
        let error = cpu_cluster_error(1.0, [0.0, 0.0, -10.0], camera, screen_height, fov_half_tan);
        // Expected: 1.0 * 1080 / (2 * 10 * 1.0) = 54
        assert!((error - 54.0).abs() < 0.1);

        // Cluster at distance 20 (half the error)
        let error2 = cpu_cluster_error(1.0, [0.0, 0.0, -20.0], camera, screen_height, fov_half_tan);
        assert!((error2 - 27.0).abs() < 0.1);
    }

    #[test]
    fn test_cpu_cluster_error_close() {
        let camera = [0.0, 0.0, 0.0];
        let screen_height = 1080.0;
        let fov_half_tan = 1.0;

        // Cluster very close to camera
        let error = cpu_cluster_error(1.0, [0.0, 0.0, -0.1], camera, screen_height, fov_half_tan);
        assert!(error > 1000.0, "Close cluster should have large error");
    }

    #[test]
    fn test_cpu_cluster_error_at_camera() {
        let camera = [5.0, 5.0, 5.0];
        let screen_height = 1080.0;
        let fov_half_tan = 1.0;

        let error = cpu_cluster_error(1.0, camera, camera, screen_height, fov_half_tan);
        assert!(error > screen_height * 0.9);
    }

    #[test]
    fn test_cpu_select_cluster_lod() {
        let camera = [0.0, 0.0, 0.0];
        let screen_height = 1080.0;
        let fov_half_tan = 1.0;
        let threshold = 1.0;

        // Far cluster with low error - should render
        let far_cluster = ClusterNode::new([0.0, 0.0, -1000.0], 1.0, 0.01, 0);
        assert!(cpu_select_cluster_lod(&far_cluster, camera, screen_height, fov_half_tan, threshold));

        // Close cluster with high error - should NOT render (needs children)
        let close_cluster = ClusterNode::new([0.0, 0.0, -1.0], 1.0, 1.0, 0);
        assert!(!cpu_select_cluster_lod(&close_cluster, camera, screen_height, fov_half_tan, threshold));
    }

    #[test]
    fn test_cpu_select_cluster_lod_force_draw() {
        let camera = [0.0, 0.0, 0.0];

        // Force draw cluster always passes
        let mut cluster = ClusterNode::new([0.0, 0.0, -1.0], 1.0, 100.0, 0); // High error
        cluster.flags |= FLAG_FORCE_DRAW;

        assert!(cpu_select_cluster_lod(&cluster, camera, 1080.0, 1.0, 1.0));
    }

    // -------------------------------------------------------------------------
    // Frustum Culling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_frustum_cull_sphere_inside() {
        let planes = make_test_frustum();
        let visible = cpu_frustum_cull_sphere([0.0, 0.0, -10.0], 1.0, &planes);
        assert!(visible, "Cluster inside frustum should be visible");
    }

    #[test]
    fn test_cpu_frustum_cull_sphere_outside() {
        let planes = make_test_frustum();
        let visible = cpu_frustum_cull_sphere([100.0, 0.0, -10.0], 1.0, &planes);
        assert!(!visible, "Cluster outside frustum should be culled");
    }

    #[test]
    fn test_cpu_frustum_cull_sphere_behind() {
        let planes = make_test_frustum();
        let visible = cpu_frustum_cull_sphere([0.0, 0.0, 10.0], 1.0, &planes);
        assert!(!visible, "Cluster behind camera should be culled");
    }

    #[test]
    fn test_cpu_frustum_cull_sphere_intersecting() {
        let planes = make_test_frustum();
        // Large sphere that intersects frustum
        let visible = cpu_frustum_cull_sphere([50.0, 0.0, -10.0], 100.0, &planes);
        assert!(visible, "Large sphere intersecting frustum should be visible");
    }

    // -------------------------------------------------------------------------
    // Cone Culling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_cone_cull_backfacing() {
        let camera = [0.0, 0.0, 0.0];
        let center = [0.0, 0.0, -10.0];
        let cone_axis = [0.0, 0.0, -1.0]; // Pointing away
        let cone_cutoff = 0.0;

        let culled = cpu_cone_cull(center, cone_axis, cone_cutoff, camera);
        assert!(culled, "Backfacing cluster should be culled");
    }

    #[test]
    fn test_cpu_cone_cull_frontfacing() {
        let camera = [0.0, 0.0, 0.0];
        let center = [0.0, 0.0, -10.0];
        let cone_axis = [0.0, 0.0, 1.0]; // Pointing toward camera
        let cone_cutoff = 0.0;

        let culled = cpu_cone_cull(center, cone_axis, cone_cutoff, camera);
        assert!(!culled, "Frontfacing cluster should not be culled");
    }

    #[test]
    fn test_cpu_cone_cull_sideways() {
        let camera = [0.0, 0.0, 0.0];
        let center = [0.0, 0.0, -10.0];
        let cone_axis = [1.0, 0.0, 0.0]; // Pointing sideways
        let cone_cutoff = 0.0;

        let culled = cpu_cone_cull(center, cone_axis, cone_cutoff, camera);
        assert!(!culled, "Sideways cluster should not be culled");
    }

    #[test]
    fn test_cpu_cone_cull_disabled() {
        let camera = [0.0, 0.0, 0.0];
        let center = [0.0, 0.0, -10.0];
        let cone_axis = [0.0, 0.0, -1.0]; // Backfacing
        let cone_cutoff = 2.0; // Disabled (>= 1.0)

        let culled = cpu_cone_cull(center, cone_axis, cone_cutoff, camera);
        assert!(!culled, "Disabled cone should not cull");
    }

    // -------------------------------------------------------------------------
    // DAG Traversal Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_traverse_dag_simple() {
        let mut dag = ClusterDAG::new();
        dag.add_root([0.0, 0.0, -10.0], 1.0, 0.001); // Small error = render

        let planes = make_test_frustum();
        let camera = [0.0, 0.0, 0.0];

        let visible = cpu_traverse_dag(&dag, &planes, camera, 1080.0, 1.0, 1.0, true, true);

        assert_eq!(visible, vec![0]);
    }

    #[test]
    fn test_cpu_traverse_dag_lod_selection() {
        let mut dag = ClusterDAG::new();

        // Root with high error - should recurse to children
        let root = dag.add_root([0.0, 0.0, -10.0], 5.0, 10.0);

        // Children with lower error - should render
        dag.add_child(root, [1.0, 0.0, -10.0], 2.0, 0.001);
        dag.add_child(root, [-1.0, 0.0, -10.0], 2.0, 0.001);

        let planes = make_test_frustum();
        let camera = [0.0, 0.0, 0.0];

        let visible = cpu_traverse_dag(&dag, &planes, camera, 1080.0, 1.0, 1.0, true, false);

        // Should select children, not root
        assert_eq!(visible.len(), 2);
        assert!(!visible.contains(&0)); // Root not selected
        assert!(visible.contains(&1));
        assert!(visible.contains(&2));
    }

    #[test]
    fn test_cpu_traverse_dag_frustum_cull() {
        let mut dag = ClusterDAG::new();

        // Visible root
        dag.add_root([0.0, 0.0, -10.0], 1.0, 0.001);

        // Root outside frustum
        dag.add_root([100.0, 0.0, -10.0], 1.0, 0.001);

        let planes = make_test_frustum();
        let camera = [0.0, 0.0, 0.0];

        let visible = cpu_traverse_dag(&dag, &planes, camera, 1080.0, 1.0, 1.0, true, false);

        assert_eq!(visible, vec![0]); // Only first root visible
    }

    #[test]
    fn test_cpu_traverse_dag_cone_cull() {
        let mut dag = ClusterDAG::new();

        // Frontfacing cluster
        let c1 = ClusterNode::new([0.0, 0.0, -10.0], 1.0, 0.001, 0)
            .with_normal_cone([0.0, 0.0, 1.0], 0.0);
        dag.add_cluster(c1);

        // Backfacing cluster
        let c2 = ClusterNode::new([5.0, 0.0, -10.0], 1.0, 0.001, 0)
            .with_normal_cone([0.0, 0.0, -1.0], 0.0);
        dag.add_cluster(c2);

        let planes = make_test_frustum();
        let camera = [0.0, 0.0, 0.0];

        let visible = cpu_traverse_dag(&dag, &planes, camera, 1080.0, 1.0, 1.0, true, true);

        assert_eq!(visible, vec![0]); // Only frontfacing visible
    }

    #[test]
    fn test_cpu_traverse_dag_inactive() {
        let mut dag = ClusterDAG::new();

        // Active cluster
        dag.add_root([0.0, 0.0, -10.0], 1.0, 0.001);

        // Inactive cluster
        let mut inactive = ClusterNode::root([5.0, 0.0, -10.0], 1.0, 0.001);
        inactive.flags &= !FLAG_ACTIVE;
        dag.add_cluster(inactive);

        let planes = make_test_frustum();
        let camera = [0.0, 0.0, 0.0];

        let visible = cpu_traverse_dag(&dag, &planes, camera, 1080.0, 1.0, 1.0, false, false);

        assert_eq!(visible, vec![0]); // Only active cluster
    }

    #[test]
    fn test_cpu_traverse_dag_deep_hierarchy() {
        let mut dag = ClusterDAG::new();

        // Build 4-level hierarchy
        let root = dag.add_root([0.0, 0.0, -100.0], 50.0, 100.0);
        let l1 = dag.add_child(root, [0.0, 0.0, -100.0], 25.0, 10.0);
        let l2 = dag.add_child(l1, [0.0, 0.0, -100.0], 12.0, 1.0);
        dag.add_child(l2, [0.0, 0.0, -100.0], 6.0, 0.001);

        let planes = make_test_frustum();
        let camera = [0.0, 0.0, 0.0];

        // With threshold 1.0, should recurse to leaf
        let visible = cpu_traverse_dag(&dag, &planes, camera, 1080.0, 1.0, 1.0, true, false);

        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0], 3); // Leaf
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normalize_vec3() {
        let v = normalize_vec3([3.0, 4.0, 0.0]);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        assert!(v[2].abs() < 1e-6);
    }

    #[test]
    fn test_normalize_vec3_zero() {
        let v = normalize_vec3([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 0.0, 1.0]); // Default direction
    }

    #[test]
    fn test_vec3_length() {
        assert!((vec3_length([3.0, 4.0, 0.0]) - 5.0).abs() < 1e-6);
        assert!((vec3_length([1.0, 1.0, 1.0]) - 3.0_f32.sqrt()).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Bytemuck Compatibility Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cluster_node_bytemuck() {
        let node = ClusterNode::new([1.0, 2.0, 3.0], 4.0, 5.0, 6);
        let bytes: &[u8] = bytemuck::bytes_of(&node);
        assert_eq!(bytes.len(), 64);

        let reconstructed: ClusterNode = *bytemuck::from_bytes(bytes);
        assert_eq!(reconstructed.bounds_center, node.bounds_center);
        assert_eq!(reconstructed.bounds_radius, node.bounds_radius);
    }

    #[test]
    fn test_cluster_cull_params_bytemuck() {
        let params = ClusterCullParams::default();
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 128);
    }

    #[test]
    fn test_frustum_plane_bytemuck() {
        let plane = FrustumPlane::new([0.0, 1.0, 0.0], 5.0);
        let bytes: &[u8] = bytemuck::bytes_of(&plane);
        assert_eq!(bytes.len(), 16);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_dag_traversal() {
        let dag = ClusterDAG::new();
        let planes = make_test_frustum();
        let visible = cpu_traverse_dag(&dag, &planes, [0.0, 0.0, 0.0], 1080.0, 1.0, 1.0, true, true);
        assert!(visible.is_empty());
    }

    #[test]
    fn test_single_cluster_dag() {
        let mut dag = ClusterDAG::new();
        dag.add_root([0.0, 0.0, -10.0], 1.0, 0.001);

        let planes = make_test_frustum();
        let visible = cpu_traverse_dag(&dag, &planes, [0.0, 0.0, 0.0], 1080.0, 1.0, 1.0, true, true);

        assert_eq!(visible.len(), 1);
    }

    #[test]
    fn test_all_culled() {
        let mut dag = ClusterDAG::new();

        // All clusters behind camera
        dag.add_root([0.0, 0.0, 10.0], 1.0, 0.001);
        dag.add_root([10.0, 0.0, 10.0], 1.0, 0.001);

        let planes = make_test_frustum();
        let visible = cpu_traverse_dag(&dag, &planes, [0.0, 0.0, 0.0], 1080.0, 1.0, 1.0, true, false);

        assert!(visible.is_empty());
    }

    #[test]
    fn test_very_large_error_threshold() {
        let mut dag = ClusterDAG::new();
        let root = dag.add_root([0.0, 0.0, -10.0], 5.0, 100.0);
        dag.add_child(root, [0.0, 0.0, -10.0], 2.0, 10.0);
        dag.add_child(root, [0.0, 0.0, -10.0], 2.0, 10.0);

        let planes = make_test_frustum();

        // Very high threshold - root should be selected
        let visible = cpu_traverse_dag(
            &dag, &planes, [0.0, 0.0, 0.0], 1080.0, 1.0, 10000.0, true, false
        );

        assert_eq!(visible, vec![0]); // Only root
    }

    #[test]
    fn test_very_small_error_threshold() {
        let mut dag = ClusterDAG::new();
        let root = dag.add_root([0.0, 0.0, -10.0], 5.0, 0.1);
        dag.add_child(root, [0.0, 0.0, -10.0], 2.0, 0.01);

        let planes = make_test_frustum();

        // Very small threshold - should recurse to finest detail
        let visible = cpu_traverse_dag(
            &dag, &planes, [0.0, 0.0, 0.0], 1080.0, 1.0, 0.0001, true, false
        );

        assert!(visible.contains(&1)); // Child should be selected
    }

    // -------------------------------------------------------------------------
    // WGSL Shader Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_naga_validate_cluster_cull_shader() {
        // Load the shader source
        let shader_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/virtual_geometry/cluster_cull.comp.wgsl"
        );
        let shader_source = std::fs::read_to_string(shader_path)
            .expect("Failed to read cluster_cull.comp.wgsl shader");

        // Parse with naga
        let module = naga::front::wgsl::parse_str(&shader_source)
            .expect("Failed to parse cluster_cull.comp.wgsl");

        // Validate the module
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module)
            .expect("cluster_cull.comp.wgsl validation failed");

        // Verify expected entry points exist
        let entry_points: Vec<&str> = module.entry_points.iter()
            .map(|ep| ep.name.as_str())
            .collect();

        assert!(entry_points.contains(&"cluster_cull"),
            "Missing entry point: cluster_cull");
        assert!(entry_points.contains(&"cluster_cull_roots"),
            "Missing entry point: cluster_cull_roots");
        assert!(entry_points.contains(&"cluster_cull_frustum_only"),
            "Missing entry point: cluster_cull_frustum_only");
        assert!(entry_points.contains(&"cluster_select_lod"),
            "Missing entry point: cluster_select_lod");
        assert!(entry_points.contains(&"cluster_cull_hierarchical"),
            "Missing entry point: cluster_cull_hierarchical");

        // Verify all are compute shaders
        for ep in &module.entry_points {
            assert_eq!(ep.stage, naga::ShaderStage::Compute,
                "Entry point {} should be a compute shader", ep.name);
        }
    }
}
