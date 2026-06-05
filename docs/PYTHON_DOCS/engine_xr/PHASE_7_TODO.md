# PHASE 7 TODO: Spatial AR Features

## Overview

Phase 7 integrates and validates the spatial AR features. The core implementation is production-ready; this phase focuses on platform integration, cloud services, and real-world validation.

## Tasks

### T-XR-7.1: Spatial Anchor Integration

**Priority**: Critical
**Effort**: Large (24 hours)
**Dependencies**: T-XR-1.1 (OpenXR bindings)

**Description**: Connect spatial anchor system to platform SDK.

**Subtasks**:
- [ ] T-XR-7.1.1: Create anchors via OpenXR spatial anchor extension
- [ ] T-XR-7.1.2: Update anchor poses from runtime
- [ ] T-XR-7.1.3: Test tracking state transitions
- [ ] T-XR-7.1.4: Test confidence decay over time
- [ ] T-XR-7.1.5: Test persistent anchor save/load
- [ ] T-XR-7.1.6: Test spatial queries (get_anchors_near)

**Acceptance Criteria**:
- [ ] Anchors persist when app restarts
- [ ] Tracking state reflects real tracking quality
- [ ] Confidence decays when anchor not visible
- [ ] Anchors restored on session restart

**Files**:
- `engine/xr/spatial/anchor.py`

---

### T-XR-7.2: Cloud Anchor Service

**Priority**: High
**Effort**: Large (32 hours)
**Dependencies**: T-XR-7.1, cloud backend

**Description**: Implement cloud anchor save/resolve for shared AR.

**Subtasks**:
- [ ] T-XR-7.2.1: Design cloud anchor API contract
- [ ] T-XR-7.2.2: Implement save_to_cloud() with HTTP upload
- [ ] T-XR-7.2.3: Implement resolve_cloud_anchor() with HTTP download
- [ ] T-XR-7.2.4: Handle resolve latency (async with callback)
- [ ] T-XR-7.2.5: Test multi-user anchor sharing
- [ ] T-XR-7.2.6: Test anchor expiration

**Acceptance Criteria**:
- [ ] Anchor saved to cloud within 5 seconds
- [ ] Anchor resolved on different device
- [ ] Multiple users see same anchor position
- [ ] Expired anchors cleaned up

**Files**:
- `engine/xr/spatial/anchor.py`
- `engine/xr/spatial/cloud_anchor_service.py` (new)

---

### T-XR-7.3: Plane Detection Validation

**Priority**: Critical
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-1.1

**Description**: Validate plane detection against real-world surfaces.

**Subtasks**:
- [ ] T-XR-7.3.1: Test horizontal plane detection (floor, table)
- [ ] T-XR-7.3.2: Test vertical plane detection (walls)
- [ ] T-XR-7.3.3: Validate plane classification accuracy
- [ ] T-XR-7.3.4: Test plane merging behavior
- [ ] T-XR-7.3.5: Test plane raycasting for placement
- [ ] T-XR-7.3.6: Measure detection latency

**Acceptance Criteria**:
- [ ] Floor detected within 3 seconds
- [ ] Walls detected within 5 seconds
- [ ] Classification 80%+ accurate
- [ ] Raycast returns valid placement points

**Files**:
- `engine/xr/spatial/plane_detection.py`

---

### T-XR-7.4: Mesh Mapping Integration

**Priority**: High
**Effort**: Large (24 hours)
**Dependencies**: T-XR-1.1, depth sensor support

**Description**: Integrate mesh mapping with depth data.

**Subtasks**:
- [ ] T-XR-7.4.1: Receive mesh blocks from runtime
- [ ] T-XR-7.4.2: Test incremental mesh updates
- [ ] T-XR-7.4.3: Test LOD distance scaling
- [ ] T-XR-7.4.4: Test distant block cleanup
- [ ] T-XR-7.4.5: Extract physics collision mesh
- [ ] T-XR-7.4.6: Extract occlusion mesh

**Acceptance Criteria**:
- [ ] Mesh covers visible environment
- [ ] Incremental updates maintain consistency
- [ ] LOD reduces triangle count at distance
- [ ] Physics mesh enables collision detection

**Files**:
- `engine/xr/spatial/mesh_mapping.py`

---

### T-XR-7.5: Scene Understanding Testing

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-7.3, T-XR-7.4

**Description**: Test scene understanding classification and helpers.

**Subtasks**:
- [ ] T-XR-7.5.1: Test room classification heuristics
- [ ] T-XR-7.5.2: Test semantic region labeling
- [ ] T-XR-7.5.3: Test light estimation accuracy
- [ ] T-XR-7.5.4: Test placement helpers (find_floor_position)
- [ ] T-XR-7.5.5: Test human segmentation (if available)
- [ ] T-XR-7.5.6: Prepare ML model integration hooks

**Acceptance Criteria**:
- [ ] Room type classified correctly for test rooms
- [ ] Semantic labels match visible objects
- [ ] Light direction matches real light source
- [ ] Placement helpers return valid positions

**Files**:
- `engine/xr/spatial/scene_understanding.py`

---

### T-XR-7.6: Image Tracking Testing

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-1.1

**Description**: Test image marker tracking system.

**Subtasks**:
- [ ] T-XR-7.6.1: Add reference image to database
- [ ] T-XR-7.6.2: Test image detection and tracking
- [ ] T-XR-7.6.3: Verify physical size scaling
- [ ] T-XR-7.6.4: Test multi-image tracking
- [ ] T-XR-7.6.5: Test extended tracking when image occluded
- [ ] T-XR-7.6.6: Measure tracking accuracy

**Acceptance Criteria**:
- [ ] Image detected within 1 second of visibility
- [ ] Pose stable when image stationary
- [ ] Physical size matches real marker size
- [ ] Extended tracking maintains pose briefly

**Files**:
- `engine/xr/spatial/image_tracking.py`

---

### T-XR-7.7: Object Tracking Testing

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-1.1

**Description**: Test 3D object recognition and tracking.

**Subtasks**:
- [ ] T-XR-7.7.1: Add reference object to database
- [ ] T-XR-7.7.2: Test object detection and tracking
- [ ] T-XR-7.7.3: Verify scale estimation
- [ ] T-XR-7.7.4: Test object occlusion support
- [ ] T-XR-7.7.5: Measure tracking accuracy
- [ ] T-XR-7.7.6: Test with different object sizes

**Acceptance Criteria**:
- [ ] Object detected when visible
- [ ] Scale matches reference object
- [ ] Occlusion mesh generated correctly
- [ ] Tracking stable when object stationary

**Files**:
- `engine/xr/spatial/object_tracking.py`

---

### T-XR-7.8: Physics Integration

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-7.4, physics system

**Description**: Connect spatial mesh to physics collision.

**Subtasks**:
- [ ] T-XR-7.8.1: Extract physics mesh from spatial mesh
- [ ] T-XR-7.8.2: Update collision shapes on mesh update
- [ ] T-XR-7.8.3: Test virtual object collision with real surfaces
- [ ] T-XR-7.8.4: Test raycasting against spatial mesh
- [ ] T-XR-7.8.5: Optimize collision mesh complexity

**Acceptance Criteria**:
- [ ] Virtual objects collide with real floor
- [ ] Virtual objects blocked by real walls
- [ ] Raycasts hit real surfaces
- [ ] Physics performance within budget

**Files**:
- `engine/xr/spatial/mesh_mapping.py`
- `engine/physics/` integration

---

### T-XR-7.9: Spatial Unit Tests

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: None

**Description**: Add unit tests for spatial algorithms.

**Subtasks**:
- [ ] T-XR-7.9.1: Test anchor confidence decay
- [ ] T-XR-7.9.2: Test plane area calculation (shoelace)
- [ ] T-XR-7.9.3: Test point-in-polygon (ray casting)
- [ ] T-XR-7.9.4: Test ray-mesh intersection (Moller-Trumbore)
- [ ] T-XR-7.9.5: Test room classification heuristics
- [ ] T-XR-7.9.6: Test bounds containment

**Acceptance Criteria**:
- [ ] >85% code coverage on spatial core
- [ ] Geometric algorithms verified mathematically
- [ ] Classification logic verified with test cases

**Files**:
- `engine/xr/spatial/tests/` (new directory)

---

## Phase 7 Completion Criteria

- [ ] Spatial anchors persist across sessions
- [ ] Cloud anchors enable shared AR experiences
- [ ] Plane detection classifies real surfaces
- [ ] Mesh mapping reconstructs environment
- [ ] Scene understanding labels semantic regions
- [ ] Image tracking stable on markers
- [ ] Object tracking recognizes 3D objects
- [ ] Physics collision with real surfaces
- [ ] Unit tests cover spatial algorithms

## Estimated Total Effort

| Task | Effort |
|------|--------|
| T-XR-7.1: Spatial Anchor Integration | 24 hours |
| T-XR-7.2: Cloud Anchor Service | 32 hours |
| T-XR-7.3: Plane Detection | 16 hours |
| T-XR-7.4: Mesh Mapping | 24 hours |
| T-XR-7.5: Scene Understanding | 16 hours |
| T-XR-7.6: Image Tracking | 16 hours |
| T-XR-7.7: Object Tracking | 16 hours |
| T-XR-7.8: Physics Integration | 16 hours |
| T-XR-7.9: Unit Tests | 16 hours |
| **Total** | **176 hours** |
