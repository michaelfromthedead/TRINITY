# Phase X5: XR Spatial Understanding and AR — Architecture

**Tasks:** T-XR-5.1 through T-XR-5.5 (5 tasks)
**Effort:** 14-19 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase X5 implements spatial understanding for AR: spatial anchors, plane detection, mesh mapping, and image/object tracking.

---

## 2. Spatial Anchors (`spatial/anchor.py`)

### SpatialAnchor Component
```python
class SpatialAnchor:
    anchor_id: ImmutableDescriptor[str]
    
    # Runtime-refined pose
    position: TrackedDescriptor[Vec3]
    orientation: TrackedDescriptor[Quat]
    
    # Persistence
    cloud_uuid: ImmutableDescriptor[str?]  # For cloud anchors
    
    # Tracking state
    tracking_state: StateMeta  # unknown/tracking/lost
```

### Anchor Types
| Type | Persistence | Use Case |
|------|-------------|----------|
| Local | Session only | Temporary placement |
| Cloud | Cross-device | Shared AR experiences |

Cloud anchor resolve for multi-user scenarios.

---

## 3. Plane Detection (`spatial/plane_detection.py`)

### PlaneDetection Component
```python
class PlaneDetection:
    classification: ImmutableDescriptor[PlaneType]
    # FLOOR, CEILING, WALL, TABLE, SEAT
    
    center: TrackedDescriptor[Vec3]
    normal: TrackedDescriptor[Vec3]
    polygon_bounds: TrackedDescriptor[list[Vec2]]
    width: TrackedDescriptor[float]
    height: TrackedDescriptor[float]
    
    is_tracked: ObservableDescriptor[bool]
```

Filtering by plane type supported.

---

## 4. Spatial Mesh Mapping (`spatial/mesh_mapping.py`)

### Real-Time Environment Mesh
- Incremental vertex updates (not full rebuild)
- Per-vertex confidence weighting
- Configurable update rate (1Hz default)

### Mesh LOD
| Level | Distance | Vertex Density |
|-------|----------|----------------|
| 0 | <2m | Full |
| 1 | 2-5m | 50% |
| 2 | >5m | 25% |

### Uses
- Physics collision
- Occlusion rendering (virtual hidden behind real)

---

## 5. Image Tracking (`spatial/image_tracking.py`)

### ImageTarget Component
```python
class ImageTarget:
    reference_image_id: ImmutableDescriptor[str]
    physical_size: ImmutableDescriptor[Vec2]  # meters
    
    is_tracked: ObservableDescriptor[bool]
    position: TrackedDescriptor[Vec3]
    orientation: TrackedDescriptor[Quat]
    tracked_size: TrackedDescriptor[Vec2]
```

### Tracking Modes
- One-shot: Detect once, stop tracking
- Continuous: Real-time tracking updates

---

## 6. Object Tracking (`spatial/object_tracking.py`)

Similar to image tracking but for 3D reference objects (CAD models, scanned objects).

---

## 7. Decorators

| Decorator | Configuration |
|-----------|---------------|
| `@spatial_anchor` | anchor_type (local/cloud), persistent |
| `@ar_trackable` | trackable_type (image/object), reference_id |

---

## 8. Dependencies

- Phase X1: XR Runtime (OpenXR spatial extensions)
- OpenXR: XR_MSFT_scene_understanding, XR_MSFT_spatial_anchor
