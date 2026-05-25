"""
Localization subsystem for the AI Game Engine.

Provides comprehensive localization support including:
- String table management with keys, contexts, and plural forms
- Translation workflow (extract, translate, import, validate)
- Translation memory for consistency
- Preview with pseudo-localization
- Dashboard for progress tracking
"""

from .string_table import (
    PluralForm,
    PluralRule,
    StringEntry,
    StringContext,
    StringTable,
    StringTableManager,
)

from .text_extraction import (
    ExtractionSource,
    ExtractedString,
    ExtractionPattern,
    CodeExtractor,
    AssetExtractor,
    TextExtractionTool,
)

from .translation_memory import (
    TMEntry,
    TMMatch,
    TMMatchType,
    TranslationMemory,
    TranslationMemoryManager,
)

from .loc_workflow import (
    WorkflowState,
    WorkflowStep,
    TranslationTask,
    ValidationResult,
    LocalizationWorkflow,
)

from .loc_preview import (
    PreviewMode,
    PseudoLocSettings,
    LocalizationPreview,
    LanguageSwitcher,
)

from .loc_dashboard import (
    LanguageProgress,
    TranslationStats,
    MissingString,
    LocalizationDashboard,
)

__all__ = [
    # String table
    "PluralForm",
    "PluralRule",
    "StringEntry",
    "StringContext",
    "StringTable",
    "StringTableManager",
    # Text extraction
    "ExtractionSource",
    "ExtractedString",
    "ExtractionPattern",
    "CodeExtractor",
    "AssetExtractor",
    "TextExtractionTool",
    # Translation memory
    "TMEntry",
    "TMMatch",
    "TMMatchType",
    "TranslationMemory",
    "TranslationMemoryManager",
    # Workflow
    "WorkflowState",
    "WorkflowStep",
    "TranslationTask",
    "ValidationResult",
    "LocalizationWorkflow",
    # Preview
    "PreviewMode",
    "PseudoLocSettings",
    "LocalizationPreview",
    "LanguageSwitcher",
    # Dashboard
    "LanguageProgress",
    "TranslationStats",
    "MissingString",
    "LocalizationDashboard",
]
