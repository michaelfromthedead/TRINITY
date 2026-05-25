"""
Text extraction for localization.

Provides tools to extract localizable strings from code and assets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, Callable
import re
import os


class ExtractionSource(Enum):
    """Source types for string extraction."""
    CODE_PYTHON = auto()
    CODE_CPP = auto()
    CODE_CSHARP = auto()
    CODE_JAVASCRIPT = auto()
    ASSET_JSON = auto()
    ASSET_XML = auto()
    ASSET_YAML = auto()
    ASSET_DIALOGUE = auto()
    ASSET_UI = auto()


@dataclass(slots=True)
class ExtractedString:
    """
    A string extracted from source code or assets.

    Contains the string content and metadata about its location.
    """
    text: str
    source_file: str
    source_line: int
    source_type: ExtractionSource
    key_suggestion: str = ""
    context_hint: str = ""
    is_plural: bool = False
    plural_variants: list[str] = field(default_factory=list)
    confidence: float = 1.0  # How confident we are this should be localized

    def generate_key(self, prefix: str = "") -> str:
        """
        Generate a suggested key for this string.

        Args:
            prefix: Optional key prefix

        Returns:
            Suggested key
        """
        if self.key_suggestion:
            return self.key_suggestion

        # Generate from text
        # Remove special characters, convert to snake_case
        key = self.text.lower()
        key = re.sub(r'[^\w\s]', '', key)
        key = re.sub(r'\s+', '_', key)
        key = key[:50]  # Truncate long strings

        if prefix:
            return f"{prefix}.{key}"
        return key


@dataclass(slots=True)
class ExtractionPattern:
    """
    Pattern for extracting localizable strings.

    Defines what to look for in source files.
    """
    name: str
    regex: str
    source_type: ExtractionSource
    group_index: int = 1  # Which regex group contains the string
    context_group: int = 0  # Group for context (0 = none)
    is_plural_pattern: bool = False


class CodeExtractor:
    """
    Extracts localizable strings from source code.

    Supports multiple programming languages with configurable patterns.
    """
    __slots__ = ("_patterns", "_ignored_patterns")

    def __init__(self):
        """Initialize code extractor with default patterns."""
        self._patterns: list[ExtractionPattern] = []
        self._ignored_patterns: list[re.Pattern] = []

        # Add default patterns
        self._add_default_patterns()

    def _add_default_patterns(self) -> None:
        """Add default extraction patterns."""
        # Python patterns
        self._patterns.extend([
            ExtractionPattern(
                name="python_gettext",
                regex=r'_\(["\'](.+?)["\']\)',
                source_type=ExtractionSource.CODE_PYTHON,
            ),
            ExtractionPattern(
                name="python_localize",
                regex=r'localize\(["\'](.+?)["\']\)',
                source_type=ExtractionSource.CODE_PYTHON,
            ),
            ExtractionPattern(
                name="python_tr",
                regex=r'tr\(["\'](.+?)["\']\)',
                source_type=ExtractionSource.CODE_PYTHON,
            ),
            ExtractionPattern(
                name="python_ngettext",
                regex=r'ngettext\(["\'](.+?)["\'],\s*["\'](.+?)["\']\)',
                source_type=ExtractionSource.CODE_PYTHON,
                is_plural_pattern=True,
            ),
        ])

        # C++ patterns
        self._patterns.extend([
            ExtractionPattern(
                name="cpp_tr",
                regex=r'TR\(["\'](.+?)["\']\)',
                source_type=ExtractionSource.CODE_CPP,
            ),
            ExtractionPattern(
                name="cpp_loctext",
                regex=r'LOCTEXT\(["\'](\w+)["\'],\s*["\'](.+?)["\']\)',
                source_type=ExtractionSource.CODE_CPP,
                group_index=2,
                context_group=1,
            ),
        ])

        # JavaScript/TypeScript patterns
        self._patterns.extend([
            ExtractionPattern(
                name="js_t",
                regex=r't\(["\'](.+?)["\']\)',
                source_type=ExtractionSource.CODE_JAVASCRIPT,
            ),
            ExtractionPattern(
                name="js_i18n",
                regex=r'i18n\(["\'](.+?)["\']\)',
                source_type=ExtractionSource.CODE_JAVASCRIPT,
            ),
        ])

        # Default ignored patterns (code comments, etc.)
        self._ignored_patterns = [
            re.compile(r'^\s*#'),  # Python comments
            re.compile(r'^\s*//'),  # C-style comments
            re.compile(r'^\s*/\*'),  # Block comments
        ]

    def add_pattern(self, pattern: ExtractionPattern) -> None:
        """Add an extraction pattern."""
        self._patterns.append(pattern)

    def add_ignored_pattern(self, regex: str) -> None:
        """Add a pattern to ignore."""
        self._ignored_patterns.append(re.compile(regex))

    def _detect_source_type(self, filepath: str) -> ExtractionSource:
        """Detect source type from file extension."""
        ext = os.path.splitext(filepath)[1].lower()

        type_map = {
            ".py": ExtractionSource.CODE_PYTHON,
            ".cpp": ExtractionSource.CODE_CPP,
            ".cc": ExtractionSource.CODE_CPP,
            ".h": ExtractionSource.CODE_CPP,
            ".hpp": ExtractionSource.CODE_CPP,
            ".cs": ExtractionSource.CODE_CSHARP,
            ".js": ExtractionSource.CODE_JAVASCRIPT,
            ".ts": ExtractionSource.CODE_JAVASCRIPT,
            ".jsx": ExtractionSource.CODE_JAVASCRIPT,
            ".tsx": ExtractionSource.CODE_JAVASCRIPT,
        }

        return type_map.get(ext, ExtractionSource.CODE_PYTHON)

    def extract_from_content(
        self,
        content: str,
        filepath: str,
        source_type: Optional[ExtractionSource] = None
    ) -> list[ExtractedString]:
        """
        Extract strings from source content.

        Args:
            content: Source code content
            filepath: Source file path
            source_type: Override source type detection

        Returns:
            List of extracted strings
        """
        if source_type is None:
            source_type = self._detect_source_type(filepath)

        results: list[ExtractedString] = []
        lines = content.split('\n')

        for pattern in self._patterns:
            if pattern.source_type != source_type:
                continue

            regex = re.compile(pattern.regex)

            for line_num, line in enumerate(lines, 1):
                # Skip ignored lines
                if any(p.match(line) for p in self._ignored_patterns):
                    continue

                for match in regex.finditer(line):
                    text = match.group(pattern.group_index)
                    context = ""

                    if pattern.context_group > 0:
                        try:
                            context = match.group(pattern.context_group)
                        except IndexError:
                            pass

                    extracted = ExtractedString(
                        text=text,
                        source_file=filepath,
                        source_line=line_num,
                        source_type=source_type,
                        context_hint=context,
                        is_plural=pattern.is_plural_pattern,
                    )

                    if pattern.is_plural_pattern:
                        # Try to get all groups as plural variants
                        extracted.plural_variants = list(match.groups())

                    results.append(extracted)

        return results

    def extract_from_file(self, filepath: str) -> list[ExtractedString]:
        """
        Extract strings from a file.

        Args:
            filepath: Path to source file

        Returns:
            List of extracted strings
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.extract_from_content(content, filepath)
        except (IOError, UnicodeDecodeError):
            return []

    def extract_from_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: Optional[list[str]] = None
    ) -> list[ExtractedString]:
        """
        Extract strings from all files in a directory.

        Args:
            directory: Directory path
            recursive: Search recursively
            extensions: File extensions to process

        Returns:
            List of extracted strings
        """
        if extensions is None:
            extensions = ['.py', '.cpp', '.h', '.js', '.ts', '.cs']

        results: list[ExtractedString] = []

        if recursive:
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    if any(filename.endswith(ext) for ext in extensions):
                        filepath = os.path.join(root, filename)
                        results.extend(self.extract_from_file(filepath))
        else:
            for filename in os.listdir(directory):
                if any(filename.endswith(ext) for ext in extensions):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        results.extend(self.extract_from_file(filepath))

        return results


class AssetExtractor:
    """
    Extracts localizable strings from game assets.

    Supports JSON, XML, YAML, and custom asset formats.
    """
    __slots__ = ("_json_keys", "_xml_tags", "_yaml_keys")

    def __init__(self):
        """Initialize asset extractor."""
        # Keys in JSON that contain localizable text
        self._json_keys = {
            "text", "label", "title", "description", "message",
            "tooltip", "hint", "name", "display_name", "dialogue",
        }
        # XML tags that contain localizable text
        self._xml_tags = {
            "Text", "Label", "Title", "Description", "Message",
            "Tooltip", "Hint", "DisplayName", "Dialogue",
        }
        # YAML keys that contain localizable text
        self._yaml_keys = self._json_keys

    def add_json_key(self, key: str) -> None:
        """Add a JSON key to extract."""
        self._json_keys.add(key)

    def add_xml_tag(self, tag: str) -> None:
        """Add an XML tag to extract."""
        self._xml_tags.add(tag)

    def extract_from_json(
        self,
        content: str,
        filepath: str
    ) -> list[ExtractedString]:
        """
        Extract strings from JSON content.

        Args:
            content: JSON content
            filepath: Source file path

        Returns:
            List of extracted strings
        """
        import json

        results: list[ExtractedString] = []

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return results

        def extract_recursive(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key

                    if key.lower() in self._json_keys and isinstance(value, str):
                        results.append(ExtractedString(
                            text=value,
                            source_file=filepath,
                            source_line=0,
                            source_type=ExtractionSource.ASSET_JSON,
                            key_suggestion=new_path,
                            context_hint=key,
                        ))
                    else:
                        extract_recursive(value, new_path)

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{path}[{i}]")

        extract_recursive(data)
        return results

    def extract_from_xml(
        self,
        content: str,
        filepath: str
    ) -> list[ExtractedString]:
        """
        Extract strings from XML content.

        Args:
            content: XML content
            filepath: Source file path

        Returns:
            List of extracted strings
        """
        results: list[ExtractedString] = []

        # Simple regex-based XML extraction
        for tag in self._xml_tags:
            pattern = re.compile(
                f'<{tag}[^>]*>([^<]+)</{tag}>',
                re.IGNORECASE
            )

            for match in pattern.finditer(content):
                text = match.group(1).strip()
                if text:
                    results.append(ExtractedString(
                        text=text,
                        source_file=filepath,
                        source_line=content[:match.start()].count('\n') + 1,
                        source_type=ExtractionSource.ASSET_XML,
                        context_hint=tag,
                    ))

        return results

    def extract_from_file(self, filepath: str) -> list[ExtractedString]:
        """
        Extract strings from an asset file.

        Args:
            filepath: Path to asset file

        Returns:
            List of extracted strings
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError):
            return []

        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.json':
            return self.extract_from_json(content, filepath)
        elif ext == '.xml':
            return self.extract_from_xml(content, filepath)

        return []


class TextExtractionTool:
    """
    Unified text extraction tool.

    Combines code and asset extraction into a single interface.
    """
    __slots__ = (
        "_code_extractor",
        "_asset_extractor",
        "_extracted_strings",
        "_duplicate_filter",
    )

    def __init__(self):
        """Initialize extraction tool."""
        self._code_extractor = CodeExtractor()
        self._asset_extractor = AssetExtractor()
        self._extracted_strings: list[ExtractedString] = []
        self._duplicate_filter: set[str] = set()

    @property
    def code_extractor(self) -> CodeExtractor:
        """Get code extractor."""
        return self._code_extractor

    @property
    def asset_extractor(self) -> AssetExtractor:
        """Get asset extractor."""
        return self._asset_extractor

    def extract_from_project(
        self,
        project_root: str,
        code_dirs: Optional[list[str]] = None,
        asset_dirs: Optional[list[str]] = None,
        deduplicate: bool = True
    ) -> list[ExtractedString]:
        """
        Extract all localizable strings from a project.

        Args:
            project_root: Root project directory
            code_dirs: Subdirectories containing code
            asset_dirs: Subdirectories containing assets
            deduplicate: Remove duplicate strings

        Returns:
            List of extracted strings
        """
        self._extracted_strings.clear()
        self._duplicate_filter.clear()

        # Default directories
        if code_dirs is None:
            code_dirs = ['src', 'source', 'scripts']
        if asset_dirs is None:
            asset_dirs = ['assets', 'data', 'content']

        # Extract from code
        for code_dir in code_dirs:
            full_path = os.path.join(project_root, code_dir)
            if os.path.isdir(full_path):
                strings = self._code_extractor.extract_from_directory(full_path)
                self._add_strings(strings, deduplicate)

        # Extract from assets
        for asset_dir in asset_dirs:
            full_path = os.path.join(project_root, asset_dir)
            if os.path.isdir(full_path):
                for root, dirs, files in os.walk(full_path):
                    for filename in files:
                        if filename.endswith(('.json', '.xml')):
                            filepath = os.path.join(root, filename)
                            strings = self._asset_extractor.extract_from_file(filepath)
                            self._add_strings(strings, deduplicate)

        return self._extracted_strings

    def _add_strings(
        self,
        strings: list[ExtractedString],
        deduplicate: bool
    ) -> None:
        """Add strings to extracted list with optional deduplication."""
        for s in strings:
            if deduplicate:
                if s.text in self._duplicate_filter:
                    continue
                self._duplicate_filter.add(s.text)

            self._extracted_strings.append(s)

    def get_extracted_strings(self) -> list[ExtractedString]:
        """Get all extracted strings."""
        return self._extracted_strings.copy()

    def get_strings_by_file(self) -> dict[str, list[ExtractedString]]:
        """Get extracted strings grouped by source file."""
        result: dict[str, list[ExtractedString]] = {}

        for s in self._extracted_strings:
            if s.source_file not in result:
                result[s.source_file] = []
            result[s.source_file].append(s)

        return result

    def get_unique_string_count(self) -> int:
        """Get count of unique strings."""
        return len(set(s.text for s in self._extracted_strings))

    def generate_string_table_entries(
        self,
        prefix: str = ""
    ) -> list[tuple[str, str, str]]:
        """
        Generate string table entries from extracted strings.

        Returns:
            List of (key, text, context) tuples
        """
        entries = []
        used_keys: set[str] = set()

        for s in self._extracted_strings:
            key = s.generate_key(prefix)

            # Ensure unique keys
            base_key = key
            counter = 1
            while key in used_keys:
                key = f"{base_key}_{counter}"
                counter += 1

            used_keys.add(key)
            entries.append((key, s.text, s.context_hint))

        return entries

    def export_for_translation(self, output_format: str = "json") -> str:
        """
        Export extracted strings for external translation.

        Args:
            output_format: Output format ("json" or "csv")

        Returns:
            Formatted export data
        """
        if output_format == "json":
            import json
            data = {
                "strings": [
                    {
                        "key": s.generate_key(),
                        "text": s.text,
                        "source_file": s.source_file,
                        "source_line": s.source_line,
                        "context": s.context_hint,
                        "is_plural": s.is_plural,
                    }
                    for s in self._extracted_strings
                ]
            }
            return json.dumps(data, indent=2)

        elif output_format == "csv":
            lines = ["key,text,source_file,context"]
            for s in self._extracted_strings:
                key = s.generate_key()
                # Escape quotes in CSV
                text = s.text.replace('"', '""')
                context = s.context_hint.replace('"', '""')
                lines.append(f'"{key}","{text}","{s.source_file}","{context}"')
            return '\n'.join(lines)

        return ""
