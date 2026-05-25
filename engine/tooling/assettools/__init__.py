"""
Asset Tools - Content browser, import pipeline, and asset management for the AI Game Engine.

This module provides comprehensive asset management capabilities including:
- ContentBrowser: Asset browser with thumbnails, filters, search, favorites
- ImportPipeline: Import system for FBX, OBJ, glTF, textures, audio
- ReferenceManager: Track asset references, find usages, redirect references
- AssetValidator: Validate assets with configurable rules
- AssetProcessor: Batch processing, compression, format conversion
- ThumbnailGenerator: Generate thumbnails asynchronously
- AssetMetadata: Metadata editing, tagging, custom properties
- AssetCollection: Asset collections for organization
- AssetSearch: Advanced search with filters, saved searches

Integrates with Foundation's ContentStore for deduplication and Provenance for lineage.
"""

from engine.tooling.assettools.content_browser import (
    ContentBrowser,
    BrowserItem,
    BrowserFilter,
    BrowserFavorites,
    BrowserHistory,
    DragDropPayload,
)
from engine.tooling.assettools.import_pipeline import (
    ImportPipeline,
    ImportSettings,
    ImportPreset,
    FBXImportSettings,
    OBJImportSettings,
    GLTFImportSettings,
    TextureImportSettings,
    AudioImportSettings,
    ImportResult,
    ImportFormat,
)
from engine.tooling.assettools.reference_manager import (
    ReferenceManager,
    AssetReference,
    ReferenceGraph,
    BrokenReference,
    ReferenceRedirect,
)
from engine.tooling.assettools.asset_validation import (
    AssetValidator,
    ValidationRule,
    ValidationResult,
    ValidationSeverity,
    TextureValidationRule,
    MeshValidationRule,
    MaterialValidationRule,
    NamingConventionRule,
)
from engine.tooling.assettools.asset_processor import (
    AssetProcessor,
    ProcessingTask,
    ProcessingResult,
    CompressionSettings,
    FormatConversionSettings,
    BatchProcessor,
)
from engine.tooling.assettools.thumbnail_generator import (
    ThumbnailGenerator,
    ThumbnailCache,
    ThumbnailRequest,
    ThumbnailResult,
    ThumbnailSize,
)
from engine.tooling.assettools.metadata import (
    AssetMetadata,
    MetadataProperty,
    MetadataTag,
    MetadataSchema,
    MetadataEditor,
)
from engine.tooling.assettools.collections import (
    AssetCollection,
    CollectionManager,
    SmartCollection,
    CollectionQuery,
)
from engine.tooling.assettools.search import (
    AssetSearch,
    SearchQuery,
    SearchFilter,
    SearchResult,
    SavedSearch,
    SearchOperator,
)

__all__ = [
    # Content Browser
    "ContentBrowser",
    "BrowserItem",
    "BrowserFilter",
    "BrowserFavorites",
    "BrowserHistory",
    "DragDropPayload",
    # Import Pipeline
    "ImportPipeline",
    "ImportSettings",
    "ImportPreset",
    "FBXImportSettings",
    "OBJImportSettings",
    "GLTFImportSettings",
    "TextureImportSettings",
    "AudioImportSettings",
    "ImportResult",
    "ImportFormat",
    # Reference Manager
    "ReferenceManager",
    "AssetReference",
    "ReferenceGraph",
    "BrokenReference",
    "ReferenceRedirect",
    # Asset Validation
    "AssetValidator",
    "ValidationRule",
    "ValidationResult",
    "ValidationSeverity",
    "TextureValidationRule",
    "MeshValidationRule",
    "MaterialValidationRule",
    "NamingConventionRule",
    # Asset Processor
    "AssetProcessor",
    "ProcessingTask",
    "ProcessingResult",
    "CompressionSettings",
    "FormatConversionSettings",
    "BatchProcessor",
    # Thumbnail Generator
    "ThumbnailGenerator",
    "ThumbnailCache",
    "ThumbnailRequest",
    "ThumbnailResult",
    "ThumbnailSize",
    # Metadata
    "AssetMetadata",
    "MetadataProperty",
    "MetadataTag",
    "MetadataSchema",
    "MetadataEditor",
    # Collections
    "AssetCollection",
    "CollectionManager",
    "SmartCollection",
    "CollectionQuery",
    # Search
    "AssetSearch",
    "SearchQuery",
    "SearchFilter",
    "SearchResult",
    "SavedSearch",
    "SearchOperator",
]
