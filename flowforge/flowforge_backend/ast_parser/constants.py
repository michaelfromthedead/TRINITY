"""Centralized constants for the FlowForge AST parser.

This module contains all magic numbers, configuration values, and shared
constants used across the ast_parser package. Import from here to avoid
duplication and ensure consistency.
"""

from __future__ import annotations

# =============================================================================
# Node ID Generation
# =============================================================================

# Length of the truncated hash used for node IDs
NODE_ID_HASH_LENGTH: int = 12

# Prefix for generated node IDs
NODE_ID_PREFIX: str = "node_"


# =============================================================================
# Trinity Framework Detection
# =============================================================================

# Module name for Trinity imports
TRINITY_MODULE: str = "trinity"

# Valid Trinity decorator names
TRINITY_DECORATOR_NAMES: frozenset[str] = frozenset({
    "component",
    "system",
    "resource",
    "event",
})

# Note: DECORATOR_TYPE_MAP is defined in visitor.py because it maps to
# TrinityDecoratorType enum values which are specific to that module.

# Base classes that indicate Trinity types (for inheritance detection)
COMPONENT_BASE_CLASSES: frozenset[str] = frozenset({"Component"})
SYSTEM_BASE_CLASSES: frozenset[str] = frozenset({"System"})
RESOURCE_BASE_CLASSES: frozenset[str] = frozenset({"Resource"})
EVENT_BASE_CLASSES: frozenset[str] = frozenset({"Event"})


# =============================================================================
# Type Annotation Parsing
# =============================================================================

# Types that should NOT be treated as class references when building edges
BUILTIN_TYPES: frozenset[str] = frozenset({
    # Python builtins
    "int", "float", "str", "bool", "bytes", "bytearray",
    "list", "dict", "set", "frozenset", "tuple",
    "None", "NoneType", "type", "object",
    # Typing module types
    "List", "Dict", "Set", "FrozenSet", "Tuple",
    "Optional", "Union", "Any", "Callable", "Type",
    "Sequence", "Mapping", "MutableMapping", "Iterable",
    "Iterator", "Generator", "Coroutine", "AsyncGenerator",
    "ClassVar", "Final", "Literal", "TypeVar", "Generic",
    "Protocol", "TypedDict", "NamedTuple",
    # Common third-party types
    "Self",
    # Trinity Query type (handled separately)
    "Query",
})

# Regex pattern string for extracting generic type parameters
# Matches patterns like List[Player], Optional[Position], Query[A, B]
GENERIC_TYPE_PATTERN: str = r"(\w+)\[([^\]]+)\]"


# =============================================================================
# Default/Placeholder Values
# =============================================================================

# Default source file name when parsing from string
DEFAULT_SOURCE_NAME: str = "<string>"

# Placeholder for unknown type annotations
UNKNOWN_ANNOTATION: str = "<unknown>"

# Placeholder for unparseable default values
UNKNOWN_DEFAULT: str = "<default>"

# Placeholder for unknown decorator names
UNKNOWN_DECORATOR: str = "<decorator>"


# =============================================================================
# Layout Constants
# =============================================================================

# Node dimensions in pixels (for visual layout)
NODE_WIDTH: int = 200
NODE_HEIGHT: int = 100

# Spacing between nodes
HORIZONTAL_SPACING: int = 50
VERTICAL_SPACING: int = 80

# Spacing between columns of different node types
COLUMN_SPACING: int = 300

# Default starting position for layouts
DEFAULT_START_X: float = 100.0
DEFAULT_START_Y: float = 100.0


# =============================================================================
# Project Scanning
# =============================================================================

# Directory/file patterns to exclude when scanning projects
DEFAULT_EXCLUDE_PATTERNS: frozenset[str] = frozenset({
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "build",
    "dist",
    "*.egg-info",
})

# File extension for Python files
PYTHON_FILE_EXTENSION: str = ".py"
