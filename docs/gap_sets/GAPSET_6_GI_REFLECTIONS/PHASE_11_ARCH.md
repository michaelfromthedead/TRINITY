# Phase 11: Advanced GI Research -- Architecture

## Overview

Phase 11 covers research and prototyping for advanced GI techniques: adaptive DDGI probe placement, sparse voxel octree implementation, and a Lumen-Lite feasibility study. These are research deliverables (documents, prototypes, go/no-go decisions) rather than production code.

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P11.1 | [-] | DDGI adaptive probe placement: research and prototype |
| T-GIR-P11.2 | [-] | Sparse Voxel Octree implementation: research and prototype |
| T-GIR-P11.3 | [-] | Lumen-Lite feasibility study |

## Current State: NOT PRODUCED

No research documents, algorithm specifications, or decision documents exist for any of these topics.

## Required Architecture (Research Deliverables)

### Adaptive DDGI Probe Placement (T-GIR-P11.1)

**Research goal**: Replace uniform grid with adaptive placement for optimal quality-per-probe.

```
Research Topics:

1. Error-driven placement:
   - Detect regions with high SH reconstruction error
   - Add probes where error exceeds threshold
   - Remove probes where error is consistently low

2. Visibility-driven placement:
   - Analyze probe visibility to surfaces
   - Place probes near geometrically complex regions
   - Reduce probe density in open, simple areas

3. Temporal stability:
   - Smooth probe count transitions (add/remove gradually)
   - Prevent probe thrashing near moving objects
   - Maintain consistent quality across frames

4. Deliverable:
   - Research document: survey of adaptive placement techniques
   - Algorithm specification: error metric, placement rules, transition logic
   - Pseudocode for GPU implementation
   - Go/no-go recommendation with quality/performance trade-off analysis
```

### Sparse Voxel Octree (T-GIR-P11.2)

**Research goal**: Replace uniform 3D voxel grid with sparse voxel octree for memory-efficient voxel GI.

```
Research Topics:

1. SVO Data Structure:
   - Pointer-based vs. pointerless (heap) octree
   - Node structure: child pointers, brick index, occupancy bits
   - Memory budgeting: how many bricks fit in GPU memory

2. Voxelisation to Octree:
   - Direct octree voxelisation vs. uniform grid -> octree conversion
   - Mipmap generation via parent averaging
   - Streaming: upload visible bricks only

3. Cone Tracing in SVO:
   - March through octree nodes rather than uniform grid
   - Empty space skipping (don't traverse empty nodes)
   - LOD selection: use octree depth for cone footprint

4. Comparison with Uniform Grid:
   - Memory: SVO vs. uniform (256^3 = 64 MB)
   - Performance: traversal overhead vs. less memory bandwidth
   - Quality: SVO adaptive detail vs. uniform resolution

5. Deliverable:
   - Research document: SVO data structure survey
   - Memory budget analysis (uniform vs. SVO)
   - Pseudocode for SVO traversal cone tracing
   - Go/no-go recommendation
```

### Lumen-Lite Feasibility Study (T-GIR-P11.3)

**Research goal**: Assess feasibility of implementing a "Lumen-lite" system combining DDGI, SSGI, and Voxel GI with software ray tracing.

```
Research Topics:

1. Lumen Architecture Overview (Epic Games, Siggraph 2022):
   - Software ray tracing against distance field + mesh SDF representation
   - Screen-space tracing fallback
   - Probe-based indirect lighting (temporal accumulation)

2. Feasibility Assessment:
   - What does Trinity already have? (DDGI probes, SH math, frame graph)
   - What's missing? (distance fields, mesh SDFs, software tracing)
   - Implementation stages: minimal viable lumen -> full lumen

3. Cost-Benefit Analysis:
   - Quality improvement over separate DDGI/SSGI/Voxel GI
   - Implementation effort (estimated in person-weeks)
   - Performance impact vs. current targeted approach

4. Integration Path:
   - Can Lumen-lite replace DDGI + SSGI + Voxel GI?
   - Or should it augment the existing pipeline?
   - Transition strategy: build alongside, swap when ready

5. Deliverable:
   - Research document: Lumen architecture survey
   - Trinity readiness assessment
   - Implementation roadmap with phases
   - Go/no-go decision with justification
```

## Dependencies

- T-GIR-P11.1: T-GIR-P2.1 (uniform probe grid as baseline)
- T-GIR-P11.2: T-GIR-P7.1 (uniform voxel grid as baseline)
- T-GIR-P11.3: T-GIR-P2.4, T-GIR-P2.7 (DDGI update + radiance cache as prerequisites)

## Files to Create

| File | Purpose |
|------|---------|
| `docs/gap_sets/GAPSET_6_GI_REFLECTIONS/research/P11.1_adaptive_ddgi.md` | Adaptive DDGI research |
| `docs/gap_sets/GAPSET_6_GI_REFLECTIONS/research/P11.2_svo_research.md` | SVO research |
| `docs/gap_sets/GAPSET_6_GI_REFLECTIONS/research/P11.3_lumen_lite.md` | Lumen-Lite feasibility |

## Acceptance Criteria (All Failing)

| Criterion | Status |
|-----------|--------|
| Adaptive placement research document exists | Failing -- not produced |
| SVO research document with memory analysis exists | Failing -- not produced |
| Lumen-Lite feasibility study exists with go/no-go | Failing -- not produced |
| Prototype code for adaptive placement exists | Failing -- not built |
| Algorithm specifications are detailed enough for implementation | Failing -- not produced |
