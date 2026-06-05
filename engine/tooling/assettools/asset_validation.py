"""
AssetValidator - Validate assets with configurable rules.

Provides comprehensive asset validation:
- Texture validation (power-of-2, size limits, format checks)
- Mesh validation (vertex limits, UV checks, material slots)
- Material validation (shader compatibility, texture references)
- Naming convention enforcement
- Custom validation rules
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union

from trinity.decorators.dev import editor


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = auto()  # Informational, not an issue
    WARNING = auto()  # Potential issue, should review
    ERROR = auto()  # Definite issue, should fix
    CRITICAL = auto()  # Blocking issue, must fix


@dataclass
class ValidationIssue:
    """Represents a validation issue.

    Attributes:
        rule_name: Name of the rule that found the issue
        severity: Severity of the issue
        message: Human-readable description
        path: Path to the asset with the issue
        location: Specific location within the asset
        suggestion: Suggested fix
        auto_fixable: Whether this can be auto-fixed
        metadata: Additional issue data
    """

    rule_name: str
    severity: ValidationSeverity
    message: str
    path: Optional[Path] = None
    location: Optional[str] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validating an asset.

    Attributes:
        path: Path to the validated asset
        passed: Whether validation passed (no errors or critical)
        issues: List of validation issues
        validation_time_ms: Time taken to validate
        rules_checked: Number of rules checked
        metadata: Additional result data
    """

    path: Path
    passed: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    validation_time_ms: float = 0.0
    rules_checked: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        """Count of errors and critical issues."""
        return sum(
            1 for i in self.issues
            if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        """Count of warnings."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)
        if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.passed = False


class ValidationRule(ABC):
    """Base class for validation rules.

    Attributes:
        name: Rule name
        description: Rule description
        severity: Default severity for issues from this rule
        enabled: Whether the rule is enabled
        extensions: File extensions this rule applies to (empty = all)
    """

    name: str = "base_rule"
    description: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True
    extensions: set[str] = set()

    def applies_to(self, path: Path) -> bool:
        """Check if this rule applies to a path."""
        if not self.enabled:
            return False
        if not self.extensions:
            return True
        return path.suffix.lstrip(".").lower() in self.extensions

    @abstractmethod
    def validate(self, path: Path, context: dict[str, Any]) -> list[ValidationIssue]:
        """Validate an asset.

        Args:
            path: Path to the asset
            context: Validation context (asset data, settings, etc)

        Returns:
            List of validation issues
        """
        pass

    def auto_fix(self, path: Path, issue: ValidationIssue) -> bool:
        """Attempt to auto-fix an issue.

        Args:
            path: Path to the asset
            issue: The issue to fix

        Returns:
            True if fixed successfully
        """
        return False


class TextureValidationRule(ValidationRule):
    """Validation rule for texture assets.

    Attributes:
        max_width: Maximum allowed width
        max_height: Maximum allowed height
        require_power_of_two: Require power-of-2 dimensions
        allowed_formats: Allowed file formats
        max_file_size_mb: Maximum file size in MB
        min_file_size_bytes: Minimum file size in bytes (files smaller are likely corrupt)
        require_mipmaps: Require mipmap data
    """

    name = "texture_validation"
    description = "Validates texture assets"
    extensions = {"png", "jpg", "jpeg", "tga", "dds", "exr", "hdr", "bmp", "tiff"}

    # Minimum valid sizes for texture headers (approximate)
    MIN_TEXTURE_SIZES = {
        "png": 67,   # PNG minimum with IHDR chunk
        "jpg": 107,  # JPEG minimum with SOI/APP0/SOF/EOI
        "jpeg": 107,
        "tga": 18,   # TGA header minimum
        "dds": 128,  # DDS header
        "exr": 100,  # OpenEXR minimum
        "hdr": 50,   # Radiance HDR minimum
        "bmp": 54,   # BMP header minimum
        "tiff": 8,   # TIFF header minimum
    }

    def __init__(
        self,
        max_width: int = 4096,
        max_height: int = 4096,
        require_power_of_two: bool = True,
        allowed_formats: Optional[set[str]] = None,
        max_file_size_mb: float = 50.0,
        min_file_size_bytes: Optional[int] = None,
        require_mipmaps: bool = False,
    ) -> None:
        self.max_width = max_width
        self.max_height = max_height
        self.require_power_of_two = require_power_of_two
        self.allowed_formats = allowed_formats or self.extensions
        self.max_file_size_mb = max_file_size_mb
        self.min_file_size_bytes = min_file_size_bytes
        self.require_mipmaps = require_mipmaps

    def validate(self, path: Path, context: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Check file exists
        if not path.exists():
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.CRITICAL,
                message="Texture file not found",
                path=path,
            ))
            return issues

        # Check format
        ext = path.suffix.lstrip(".").lower()
        if ext not in self.allowed_formats:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Format '{ext}' not in allowed formats: {self.allowed_formats}",
                path=path,
                suggestion=f"Convert to one of: {', '.join(self.allowed_formats)}",
            ))

        # Check file size
        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Check minimum size (files too small are likely corrupt or invalid)
        min_size = self.min_file_size_bytes
        if min_size is None:
            # Use format-specific minimum or a general minimum
            min_size = self.MIN_TEXTURE_SIZES.get(ext, 10)
        if file_size_bytes < min_size:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"File size {file_size_bytes} bytes is below minimum {min_size} bytes for {ext} format",
                path=path,
                suggestion="File may be corrupt or invalid",
            ))

        # Check maximum size
        if file_size_mb > self.max_file_size_mb:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"File size {file_size_mb:.1f}MB exceeds max {self.max_file_size_mb}MB",
                path=path,
                suggestion="Compress or resize the texture",
                auto_fixable=True,
            ))

        # Get texture dimensions from context or parse file
        width = context.get("width", 0)
        height = context.get("height", 0)

        if width > 0 and height > 0:
            # Check dimensions
            if width > self.max_width:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Width {width} exceeds maximum {self.max_width}",
                    path=path,
                    suggestion=f"Resize to max width {self.max_width}",
                    auto_fixable=True,
                ))

            if height > self.max_height:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Height {height} exceeds maximum {self.max_height}",
                    path=path,
                    suggestion=f"Resize to max height {self.max_height}",
                    auto_fixable=True,
                ))

            # Check power of two
            if self.require_power_of_two:
                if not self._is_power_of_two(width):
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Width {width} is not a power of 2",
                        path=path,
                        suggestion=f"Resize to {self._nearest_power_of_two(width)}",
                        auto_fixable=True,
                    ))

                if not self._is_power_of_two(height):
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Height {height} is not a power of 2",
                        path=path,
                        suggestion=f"Resize to {self._nearest_power_of_two(height)}",
                        auto_fixable=True,
                    ))

        return issues

    def _is_power_of_two(self, n: int) -> bool:
        """Check if a number is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def _nearest_power_of_two(self, n: int) -> int:
        """Get the nearest power of 2."""
        if n <= 0:
            return 1
        # Round to nearest power of 2
        lower = 1 << (n - 1).bit_length() - 1
        upper = 1 << (n - 1).bit_length()
        if n - lower < upper - n:
            return lower
        return upper


class MeshValidationRule(ValidationRule):
    """Validation rule for mesh assets.

    Attributes:
        max_vertices: Maximum vertex count per mesh
        max_triangles: Maximum triangle count per mesh
        max_materials: Maximum material slots
        require_uvs: Require UV coordinates
        require_normals: Require vertex normals
        require_tangents: Require tangent space
        max_bones: Maximum bones per skinned mesh
        max_bone_weights: Maximum bone influences per vertex
    """

    name = "mesh_validation"
    description = "Validates mesh assets"
    extensions = {"fbx", "obj", "gltf", "glb", "dae"}

    def __init__(
        self,
        max_vertices: int = 100000,
        max_triangles: int = 100000,
        max_materials: int = 8,
        require_uvs: bool = True,
        require_normals: bool = True,
        require_tangents: bool = False,
        max_bones: int = 256,
        max_bone_weights: int = 4,
    ) -> None:
        self.max_vertices = max_vertices
        self.max_triangles = max_triangles
        self.max_materials = max_materials
        self.require_uvs = require_uvs
        self.require_normals = require_normals
        self.require_tangents = require_tangents
        self.max_bones = max_bones
        self.max_bone_weights = max_bone_weights

    def validate(self, path: Path, context: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Check file exists
        if not path.exists():
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.CRITICAL,
                message="Mesh file not found",
                path=path,
            ))
            return issues

        # Get mesh data from context
        vertex_count = context.get("vertex_count", 0)
        triangle_count = context.get("triangle_count", 0)
        material_count = context.get("material_count", 0)
        has_uvs = context.get("has_uvs", True)
        has_normals = context.get("has_normals", True)
        has_tangents = context.get("has_tangents", False)
        bone_count = context.get("bone_count", 0)

        # Vertex count
        if vertex_count > self.max_vertices:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Vertex count {vertex_count} exceeds max {self.max_vertices}",
                path=path,
                suggestion="Reduce polygon count or split into multiple meshes",
            ))

        # Triangle count
        if triangle_count > self.max_triangles:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Triangle count {triangle_count} exceeds max {self.max_triangles}",
                path=path,
                suggestion="Reduce polygon count",
            ))

        # Material slots
        if material_count > self.max_materials:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message=f"Material count {material_count} exceeds recommended max {self.max_materials}",
                path=path,
                suggestion="Consider combining materials or splitting mesh",
            ))

        # UVs
        if self.require_uvs and not has_uvs:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message="Mesh missing UV coordinates",
                path=path,
                suggestion="Add UV mapping in modeling software",
            ))

        # Normals
        if self.require_normals and not has_normals:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message="Mesh missing vertex normals",
                path=path,
                suggestion="Compute normals during import",
                auto_fixable=True,
            ))

        # Tangents
        if self.require_tangents and not has_tangents:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message="Mesh missing tangent space",
                path=path,
                suggestion="Compute tangents during import",
                auto_fixable=True,
            ))

        # Bones
        if bone_count > self.max_bones:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Bone count {bone_count} exceeds max {self.max_bones}",
                path=path,
                suggestion="Reduce skeleton complexity",
            ))

        return issues


class MaterialValidationRule(ValidationRule):
    """Validation rule for material assets.

    Attributes:
        allowed_shaders: List of allowed shader names
        required_textures: Required texture slots
        max_texture_slots: Maximum texture slots
        validate_texture_refs: Validate that referenced textures exist
    """

    name = "material_validation"
    description = "Validates material assets"
    extensions = {"mat", "mtl"}

    def __init__(
        self,
        allowed_shaders: Optional[list[str]] = None,
        required_textures: Optional[list[str]] = None,
        max_texture_slots: int = 16,
        validate_texture_refs: bool = True,
    ) -> None:
        self.allowed_shaders = allowed_shaders
        self.required_textures = required_textures or []
        self.max_texture_slots = max_texture_slots
        self.validate_texture_refs = validate_texture_refs

    def validate(self, path: Path, context: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Check file exists
        if not path.exists():
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.CRITICAL,
                message="Material file not found",
                path=path,
            ))
            return issues

        # Get material data from context
        shader_name = context.get("shader_name", "")
        texture_slots = context.get("texture_slots", {})
        texture_refs = context.get("texture_refs", [])

        # Shader validation
        if self.allowed_shaders and shader_name:
            if shader_name not in self.allowed_shaders:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Shader '{shader_name}' not in allowed list",
                    path=path,
                    suggestion=f"Use one of: {', '.join(self.allowed_shaders)}",
                ))

        # Required textures
        for required in self.required_textures:
            if required not in texture_slots:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing required texture slot: {required}",
                    path=path,
                    suggestion=f"Add texture to '{required}' slot",
                ))

        # Texture slot count
        if len(texture_slots) > self.max_texture_slots:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message=f"Texture slot count {len(texture_slots)} exceeds max {self.max_texture_slots}",
                path=path,
            ))

        # Validate texture references exist
        if self.validate_texture_refs:
            for tex_path in texture_refs:
                if not Path(tex_path).exists():
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Referenced texture not found: {tex_path}",
                        path=path,
                        location=tex_path,
                    ))

        return issues


class NamingConventionRule(ValidationRule):
    """Validation rule for naming conventions.

    Attributes:
        patterns: Dict mapping extensions to regex patterns
        case_style: Required case style (snake_case, PascalCase, etc)
        forbidden_chars: Characters not allowed in names
        max_length: Maximum filename length
        require_prefix: Required prefix patterns by folder
    """

    name = "naming_convention"
    description = "Validates asset naming conventions"
    severity = ValidationSeverity.WARNING

    def __init__(
        self,
        patterns: Optional[dict[str, str]] = None,
        case_style: Optional[str] = None,
        forbidden_chars: str = " !@#$%^&*()+=[]{}|;:'\",<>?",
        max_length: int = 64,
        require_prefix: Optional[dict[str, str]] = None,
    ) -> None:
        self.patterns = patterns or {}
        self.case_style = case_style
        self.forbidden_chars = forbidden_chars
        self.max_length = max_length
        self.require_prefix = require_prefix or {}

    def validate(self, path: Path, context: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        name = path.stem  # Filename without extension
        ext = path.suffix.lstrip(".").lower()

        # Length check
        if len(name) > self.max_length:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=self.severity,
                message=f"Filename '{name}' exceeds max length {self.max_length}",
                path=path,
                suggestion=f"Shorten to {self.max_length} characters",
            ))

        # Forbidden characters
        for char in self.forbidden_chars:
            if char in name:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Filename contains forbidden character: '{char}'",
                    path=path,
                    suggestion="Remove or replace forbidden characters",
                    auto_fixable=True,
                ))
                break

        # Pattern matching
        if ext in self.patterns:
            pattern = self.patterns[ext]
            if not re.match(pattern, name):
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Filename '{name}' doesn't match pattern: {pattern}",
                    path=path,
                    suggestion="Rename to match expected pattern",
                ))

        # Case style
        if self.case_style:
            if self.case_style == "snake_case" and not self._is_snake_case(name):
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Filename '{name}' is not snake_case",
                    path=path,
                    suggestion=f"Rename to: {self._to_snake_case(name)}",
                    auto_fixable=True,
                ))
            elif self.case_style == "PascalCase" and not self._is_pascal_case(name):
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Filename '{name}' is not PascalCase",
                    path=path,
                    suggestion=f"Rename to: {self._to_pascal_case(name)}",
                    auto_fixable=True,
                ))

        # Prefix requirements
        for folder_pattern, prefix in self.require_prefix.items():
            if folder_pattern in str(path.parent):
                if not name.startswith(prefix):
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Filename should start with '{prefix}' prefix",
                        path=path,
                        suggestion=f"Rename to: {prefix}{name}",
                        auto_fixable=True,
                    ))

        return issues

    def _is_snake_case(self, name: str) -> bool:
        """Check if name is snake_case."""
        return bool(re.match(r'^[a-z][a-z0-9]*(_[a-z0-9]+)*$', name))

    def _is_pascal_case(self, name: str) -> bool:
        """Check if name is PascalCase."""
        return bool(re.match(r'^[A-Z][a-zA-Z0-9]*$', name))

    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case."""
        # Insert underscore before uppercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _to_pascal_case(self, name: str) -> str:
        """Convert to PascalCase."""
        # Split on underscores and spaces
        parts = re.split(r'[_\s]+', name)
        return ''.join(p.capitalize() for p in parts)


@editor(category="Assets")
class AssetValidator:
    """Validates assets using configurable rules.

    Provides:
    - Rule-based validation
    - Batch validation
    - Auto-fix capabilities
    - Validation profiles

    Attributes:
        rules: List of active validation rules
        profiles: Named validation profiles
        _results_cache: Cache of recent validation results
    """

    def __init__(self) -> None:
        """Initialize the asset validator."""
        self.rules: list[ValidationRule] = []
        self.profiles: dict[str, list[ValidationRule]] = {}
        self._results_cache: dict[Path, ValidationResult] = {}

        # Add default rules
        self._add_default_rules()

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                return True
        return False

    def get_rule(self, rule_name: str) -> Optional[ValidationRule]:
        """Get a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None

    def enable_rule(self, rule_name: str) -> bool:
        """Enable a rule by name."""
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = True
            return True
        return False

    def disable_rule(self, rule_name: str) -> bool:
        """Disable a rule by name."""
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = False
            return True
        return False

    def validate(
        self,
        path: Union[str, Path],
        context: Optional[dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> ValidationResult:
        """Validate an asset.

        Args:
            path: Path to the asset
            context: Additional context for validation
            use_cache: Whether to use cached results

        Returns:
            ValidationResult with all issues
        """
        path = Path(path)
        context = context or {}

        # Check cache
        if use_cache and path in self._results_cache:
            cached = self._results_cache[path]
            # Invalidate if file modified
            if path.exists():
                if path.stat().st_mtime <= cached.metadata.get("mtime", 0):
                    return cached

        # Create result
        result = ValidationResult(path=path)
        start_time = time.perf_counter()

        # Run applicable rules
        for rule in self.rules:
            if rule.applies_to(path):
                try:
                    issues = rule.validate(path, context)
                    for issue in issues:
                        issue.path = path
                        result.add_issue(issue)
                    result.rules_checked += 1
                except Exception as e:
                    result.add_issue(ValidationIssue(
                        rule_name=rule.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Rule failed with error: {e}",
                        path=path,
                    ))

        # Calculate time
        result.validation_time_ms = (time.perf_counter() - start_time) * 1000

        # Store in cache
        if path.exists():
            result.metadata["mtime"] = path.stat().st_mtime
        self._results_cache[path] = result

        return result

    def validate_batch(
        self,
        paths: list[Union[str, Path]],
        context: Optional[dict[str, Any]] = None,
    ) -> list[ValidationResult]:
        """Validate multiple assets.

        Args:
            paths: List of asset paths
            context: Shared context for all validations

        Returns:
            List of ValidationResults
        """
        return [self.validate(path, context) for path in paths]

    def validate_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        context: Optional[dict[str, Any]] = None,
    ) -> list[ValidationResult]:
        """Validate all assets in a directory.

        Args:
            directory: Directory to validate
            recursive: Whether to validate subdirectories
            context: Shared context for all validations

        Returns:
            List of ValidationResults
        """
        directory = Path(directory)
        results: list[ValidationResult] = []

        if recursive:
            paths = list(directory.rglob("*"))
        else:
            paths = list(directory.iterdir())

        for path in paths:
            if path.is_file():
                results.append(self.validate(path, context))

        return results

    def auto_fix(
        self,
        result: ValidationResult,
        dry_run: bool = False,
    ) -> list[tuple[ValidationIssue, bool]]:
        """Attempt to auto-fix issues.

        Args:
            result: Validation result with issues
            dry_run: If True, don't actually apply fixes

        Returns:
            List of (issue, fixed) tuples
        """
        fixed: list[tuple[ValidationIssue, bool]] = []

        for issue in result.issues:
            if not issue.auto_fixable:
                continue

            rule = self.get_rule(issue.rule_name)
            if not rule:
                continue

            if dry_run:
                fixed.append((issue, True))
            else:
                success = rule.auto_fix(result.path, issue)
                fixed.append((issue, success))

        return fixed

    def save_profile(self, name: str) -> None:
        """Save current rules as a profile."""
        self.profiles[name] = self.rules.copy()

    def load_profile(self, name: str) -> bool:
        """Load a saved profile."""
        if name in self.profiles:
            self.rules = self.profiles[name].copy()
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        results = list(self._results_cache.values())

        total_issues = sum(len(r.issues) for r in results)
        errors = sum(r.error_count for r in results)
        warnings = sum(r.warning_count for r in results)
        passed = sum(1 for r in results if r.passed)

        return {
            "total_validated": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "total_issues": total_issues,
            "errors": errors,
            "warnings": warnings,
            "rules_enabled": sum(1 for r in self.rules if r.enabled),
            "rules_total": len(self.rules),
        }

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._results_cache.clear()

    def _add_default_rules(self) -> None:
        """Add default validation rules."""
        self.rules.append(TextureValidationRule())
        self.rules.append(MeshValidationRule())
        self.rules.append(MaterialValidationRule())
        self.rules.append(NamingConventionRule())


__all__ = [
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "ValidationRule",
    "TextureValidationRule",
    "MeshValidationRule",
    "MaterialValidationRule",
    "NamingConventionRule",
    "AssetValidator",
]
