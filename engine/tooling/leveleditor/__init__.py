"""
Level Editor - Comprehensive editor tooling for scene and level construction.

This module provides a complete level editing toolkit including:
- Object placement with multiple modes (single, paint, scatter, foliage, spline)
- Snapping systems (grid, surface, vertex, edge, pivot)
- Scene hierarchy management with drag-drop, grouping, and layers
- Alignment and distribution tools
- Prefab system with nested prefabs and overrides
- Layer management with visibility, locking, and color coding
- Camera bookmarks for quick navigation
- Measurement tools for distance, angle, and area

All editor-only classes use the @editor decorator and integrate with
Foundation's Tracker for full undo/redo support.
"""

from .placement import (
    PlacementMode,
    PlacementTool,
    PlacementResult,
    ScatterSettings,
    FoliageSettings,
    SplineSettings,
    BrushSettings,
)
from .snapping import (
    SnapMode,
    SnapSettings,
    SnapResult,
    GridSnap,
    SurfaceSnap,
    VertexSnap,
    EdgeSnap,
    PivotSnap,
    SnapManager,
)
from .hierarchy import (
    HierarchyNode,
    HierarchyTree,
    HierarchyFolder,
    HierarchyGroup,
    DragDropOperation,
    HierarchyFilter,
)
from .alignment import (
    AlignAxis,
    AlignReference,
    AlignmentTool,
)
from .distribution import (
    DistributionMode,
    DistributionTool,
    SpacingSettings,
)
from .prefabs import (
    PrefabAsset,
    PrefabInstance,
    PrefabOverride,
    PrefabVariant,
    PrefabManager,
)
from .layers import (
    Layer,
    LayerSettings,
    LayerManager,
    LayerColor,
)
from .bookmarks import (
    CameraBookmark,
    BookmarkManager,
    BookmarkCategory,
)
from .measurements import (
    MeasurementUnit,
    MeasurementType,
    MeasurementTool,
    MeasurementResult,
    DistanceMeasurement,
    AngleMeasurement,
    AreaMeasurement,
)

__all__ = [
    # Placement
    "PlacementMode",
    "PlacementTool",
    "PlacementResult",
    "ScatterSettings",
    "FoliageSettings",
    "SplineSettings",
    "BrushSettings",
    # Snapping
    "SnapMode",
    "SnapSettings",
    "SnapResult",
    "GridSnap",
    "SurfaceSnap",
    "VertexSnap",
    "EdgeSnap",
    "PivotSnap",
    "SnapManager",
    # Hierarchy
    "HierarchyNode",
    "HierarchyTree",
    "HierarchyFolder",
    "HierarchyGroup",
    "DragDropOperation",
    "HierarchyFilter",
    # Alignment
    "AlignAxis",
    "AlignReference",
    "AlignmentTool",
    # Distribution
    "DistributionMode",
    "DistributionTool",
    "SpacingSettings",
    # Prefabs
    "PrefabAsset",
    "PrefabInstance",
    "PrefabOverride",
    "PrefabVariant",
    "PrefabManager",
    # Layers
    "Layer",
    "LayerSettings",
    "LayerManager",
    "LayerColor",
    # Bookmarks
    "CameraBookmark",
    "BookmarkManager",
    "BookmarkCategory",
    # Measurements
    "MeasurementUnit",
    "MeasurementType",
    "MeasurementTool",
    "MeasurementResult",
    "DistanceMeasurement",
    "AngleMeasurement",
    "AreaMeasurement",
]
