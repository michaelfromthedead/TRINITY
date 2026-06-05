// SPDX-License-Identifier: MIT
//
// reduce_sum.wgsl - Parallel sum reduction shader (T-WGPU-P3.10.1).
//
// Performs parallel sum reduction using tree reduction in workgroup shared memory.
// Supports f32 arrays with sequential addressing to avoid shared memory bank conflicts.
//
// Algorithm:
//   1. Each thread loads 2 elements from global memory, sums them, stores to shared
//   2. Tree reduction in shared memory with workgroupBarrier() between steps
//   3. Thread 0 writes final partial sum to output
//
// For arrays > WORKGROUP_SIZE * 2, multiple passes are required:
//   - Pass 1: Reduce input to N / (WORKGROUP_SIZE * 2) partial sums
//   - Pass 2+: Reduce partial sums until single value remains
//
// Workgroup size: 256 threads (configurable via override constant)
// Each workgroup reduces 512 elements to 1 partial sum.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

override WORKGROUP_SIZE: u32 = 256u;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct ReductionParams {
    input_size: u32,    // Total number of elements in input buffer
    output_offset: u32, // Offset into output buffer for this pass
    _pad0: u32,
    _pad1: u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: ReductionParams;

// ---------------------------------------------------------------------------
// Shared Memory
// ---------------------------------------------------------------------------

var<workgroup> shared_data: array<f32, 256>;

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// Parallel sum reduction kernel.
///
/// Each thread loads 2 elements, performs local addition, then participates
/// in tree reduction within the workgroup. Thread 0 writes the final sum
/// for this workgroup to the output buffer.
@compute @workgroup_size(256, 1, 1)
fn reduce_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_idx = lid.x;
    let workgroup_idx = wid.x;

    // Calculate indices for coalesced global memory access
    // Each thread loads 2 elements: at index 2*gid.x and 2*gid.x + 1
    let idx0 = workgroup_idx * WORKGROUP_SIZE * 2u + local_idx;
    let idx1 = idx0 + WORKGROUP_SIZE;

    // Load and sum two elements (with bounds checking)
    var sum: f32 = 0.0;
    if (idx0 < params.input_size) {
        sum = input[idx0];
    }
    if (idx1 < params.input_size) {
        sum = sum + input[idx1];
    }

    // Store to shared memory
    shared_data[local_idx] = sum;
    workgroupBarrier();

    // Tree reduction in shared memory
    // Sequential addressing pattern avoids bank conflicts
    // stride: 128, 64, 32, 16, 8, 4, 2, 1
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride) {
            shared_data[local_idx] = shared_data[local_idx] + shared_data[local_idx + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the final partial sum for this workgroup
    if (local_idx == 0u) {
        output[params.output_offset + workgroup_idx] = shared_data[0];
    }
}
