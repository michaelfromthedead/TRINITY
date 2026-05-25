# Phase 3: Skinning Compute Shaders -- Architecture

## Status: 1 [x] 0 [~] 4 [-]

## Module: `engine/animation/skeletal/skinning.py`

### Files
| File | Lines | Purpose |
|------|-------|---------|
| skinning.py | 797 | CPU skinning orchestrator |
| *(missing)* shaders/skinning/ | 0 | **Does not exist** |

### Architecture

**Skinning** (`skinning.py`):
- `SkinningMethod`: LINEAR (LBS), DUAL_QUATERNION (DQS)
- `VertexWeight`: bone_index + weight pair
- `SkinningData`: vertex weights, bind_poses, bone_names, max_influences
- `DualQuaternion`: real + dual Quat, conversion from Mat4, transform_vector
- `LinearBlendSkinning`: weighted sum of skinning matrices
- `DualQuaternionSkinning`: DQS blend with antipodality handling
- `GPUSkinningData`: texture-size arrays prepared for shader upload
- `SkinningCache`: LRU cache keyed by (pose_hash, lod_level)
- `prepare_gpu_skinning_data()`: layout bone matrices as flattened array
- `skin_mesh()`: convenience function for single-call skinning

**GPU data flow** (`systems/skinning_system.py`):
- `SkinnedMeshComponent`: mesh ref, skinning_data, method, output buffers
- `SkinningSystem.update()`: computes skinning matrices, dispatches to LBS/DQS/GPU
  - GPU path: calls `component.prepare_gpu_buffer()` for SSBO upload
  - LBS path: inline CPU vertex transform
  - DQS path: dual quaternion CPU vertex transform
- Bone influence reduction: full (4) -> simplified (2) -> single (1) via LOD

### Missing (All 4 WGSL tasks)
- T-AN-3.2: `shaders/skinning/skinning_lbs.comp.wgsl` -- LBS compute shader
- T-AN-3.3: `shaders/skinning/skinning_dqs.comp.wgsl` -- DQS compute shader
- T-AN-3.4: `shaders/skinning/skinning_vert.wgsl` -- vertex shader fallback
- T-AN-3.5: Skinning tests

### Key Design Decisions
- CPU path is fully implemented (LBS + DQS with antipodality check)
- `GPUSkinningData` provides Python-side data structure ready for WGSL consumption
- Dual quaternion skinning uses hemisphere consistency check for correct blending
- `SkinningCache` avoids redundant skinning when pose unchanged
- Pipeline integration: skinning system reads pose data from animation graph system output
