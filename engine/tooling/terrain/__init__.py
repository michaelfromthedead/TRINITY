"""
Terrain Tools subsystem for the AI Game Engine.

Provides comprehensive terrain editing capabilities including:
- Sculpting: Raise, Lower, Smooth, Flatten, Noise, Level, Stamp
- Painting: Material layers with height/slope/noise masks
- Import/Export: Heightmaps (RAW, PNG), terrain data
- Erosion: Hydraulic and thermal erosion simulation
- Foliage: Placement, density painting, LOD settings
- LOD: Chunk-based terrain streaming
- Materials: Layer management with blending
"""

from .sculpt_tools import (
    SculptMode,
    BrushShape,
    FalloffCurve,
    TerrainBrush,
    SculptOperation,
    TerrainSculptTool,
)

from .paint_tools import (
    PaintMode,
    LayerBlendMode,
    TerrainMask,
    HeightMask,
    SlopeMask,
    NoiseMask,
    PaintBrush,
    TerrainPaintTool,
)

from .terrain_import import (
    HeightmapFormat,
    TerrainExportFormat,
    HeightmapImporter,
    TerrainExporter,
    TerrainImportExport,
)

from .erosion_tools import (
    ErosionType,
    ErosionParams,
    HydraulicErosionParams,
    ThermalErosionParams,
    ErosionSimulator,
)

from .foliage_tools import (
    FoliageType,
    FoliageLODLevel,
    FoliageInstance,
    FoliageLayer,
    FoliageDensityBrush,
    FoliagePlacementTool,
)

from .terrain_lod import (
    LODLevel,
    TerrainChunk,
    ChunkState,
    TerrainLODSettings,
    TerrainChunkManager,
    TerrainLODSystem,
)

from .terrain_materials import (
    BlendMode,
    TerrainMaterialLayer,
    MaterialBlendSettings,
    TerrainMaterialStack,
    TerrainMaterialManager,
)

__all__ = [
    # Sculpt tools
    "SculptMode",
    "BrushShape",
    "FalloffCurve",
    "TerrainBrush",
    "SculptOperation",
    "TerrainSculptTool",
    # Paint tools
    "PaintMode",
    "LayerBlendMode",
    "TerrainMask",
    "HeightMask",
    "SlopeMask",
    "NoiseMask",
    "PaintBrush",
    "TerrainPaintTool",
    # Import/Export
    "HeightmapFormat",
    "TerrainExportFormat",
    "HeightmapImporter",
    "TerrainExporter",
    "TerrainImportExport",
    # Erosion
    "ErosionType",
    "ErosionParams",
    "HydraulicErosionParams",
    "ThermalErosionParams",
    "ErosionSimulator",
    # Foliage
    "FoliageType",
    "FoliageLODLevel",
    "FoliageInstance",
    "FoliageLayer",
    "FoliageDensityBrush",
    "FoliagePlacementTool",
    # LOD
    "LODLevel",
    "TerrainChunk",
    "ChunkState",
    "TerrainLODSettings",
    "TerrainChunkManager",
    "TerrainLODSystem",
    # Materials
    "BlendMode",
    "TerrainMaterialLayer",
    "MaterialBlendSettings",
    "TerrainMaterialStack",
    "TerrainMaterialManager",
]
