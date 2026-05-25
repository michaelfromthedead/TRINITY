"""
HLOD (Hierarchical Level of Detail) System.

Provides mesh generation, layer management, and smooth transitions
for efficient distant rendering in the world partition system.

Modules:
    constants: Centralized HLOD system constants
    generator: HLOD mesh generation (merging, simplification, impostor, proxy)
    layers: HLOD layer and cell management
    transitions: LOD transition handling and visibility
"""

from .constants import (
    FloatingPointConstants,
    SimplificationConstants,
    MergeConstants,
    ImpostorConstants,
    MethodSelectionConstants,
    LayerConstants,
    TransitionConstantsConfig,
    ValidationConstants,
)

from .generator import (
    # Constants
    HLODConstants,
    # Enums
    HLODGenerationMethod,
    # Settings
    SimplificationSettings,
    ImpostorSettings,
    MergeSettings,
    # Math types
    Vec3,
    Vec2,
    AABB,
    # Data structures
    MeshData,
    HLODMeshData,
    ImpostorData,
    Edge,
    # Generators
    MeshMerger,
    MeshSimplifier,
    ImpostorGenerator,
    ProxyMeshGenerator,
    HLODGenerator,
)

from .layers import (
    # Constants
    HLODLayerConstants,
    # Enums
    HLODCellState,
    # Configuration
    HLODLayerConfig,
    # Core classes
    HLODLayer,
    HLODCell,
    HLODLayerManager,
    HLODCluster,
    HLODHierarchyManager,
)

from .transitions import (
    # Constants
    TransitionConstants,
    # Enums
    TransitionMode,
    TransitionState,
    # Settings
    TransitionSettings,
    # Core classes
    LODTransition,
    TransitionCalculator,
    ScreenSpaceError,
    HLODTransitionManager,
    # Visibility
    VisibilityResult,
    HLODVisibilitySystem,
)

__all__ = [
    # === Constants Module ===
    "FloatingPointConstants",
    "SimplificationConstants",
    "MergeConstants",
    "ImpostorConstants",
    "MethodSelectionConstants",
    "LayerConstants",
    "TransitionConstantsConfig",
    "ValidationConstants",
    # === Generator Module ===
    # Constants
    "HLODConstants",
    # Enums
    "HLODGenerationMethod",
    # Settings
    "SimplificationSettings",
    "ImpostorSettings",
    "MergeSettings",
    # Math types
    "Vec3",
    "Vec2",
    "AABB",
    # Data structures
    "MeshData",
    "HLODMeshData",
    "ImpostorData",
    "Edge",
    # Generators
    "MeshMerger",
    "MeshSimplifier",
    "ImpostorGenerator",
    "ProxyMeshGenerator",
    "HLODGenerator",
    # === Layers Module ===
    # Constants
    "HLODLayerConstants",
    # Enums
    "HLODCellState",
    # Configuration
    "HLODLayerConfig",
    # Core classes
    "HLODLayer",
    "HLODCell",
    "HLODLayerManager",
    "HLODCluster",
    "HLODHierarchyManager",
    # === Transitions Module ===
    # Constants
    "TransitionConstants",
    # Enums
    "TransitionMode",
    "TransitionState",
    # Settings
    "TransitionSettings",
    # Core classes
    "LODTransition",
    "TransitionCalculator",
    "ScreenSpaceError",
    "HLODTransitionManager",
    # Visibility
    "VisibilityResult",
    "HLODVisibilitySystem",
]
