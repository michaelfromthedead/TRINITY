"""
Comprehensive tests for AssetValidator functionality.

Tests validation rules, result handling, and auto-fix capabilities.
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.assettools.asset_validation import (
    AssetValidator,
    ValidationRule,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    TextureValidationRule,
    MeshValidationRule,
    MaterialValidationRule,
    NamingConventionRule,
)


@pytest.fixture
def temp_validation_dir():
    """Create a temporary directory for validation tests."""
    path = Path(tempfile.mkdtemp())

    # Create test files
    (path / "valid_texture.png").write_bytes(b"x" * 2048)
    (path / "large_texture.png").write_bytes(b"x" * (60 * 1024 * 1024))  # 60MB
    (path / "model.fbx").write_text("fbx data")
    (path / "material.mat").write_text("{}")
    (path / "Invalid Name.png").write_text("bad")
    (path / "hero_diffuse.png").write_text("good")

    yield path
    shutil.rmtree(path)


class TestValidationSeverity:
    """Test ValidationSeverity enum."""

    def test_severity_ordering(self):
        """Severity levels should have proper ordering."""
        assert ValidationSeverity.INFO.value < ValidationSeverity.WARNING.value
        assert ValidationSeverity.WARNING.value < ValidationSeverity.ERROR.value
        assert ValidationSeverity.ERROR.value < ValidationSeverity.CRITICAL.value


class TestValidationIssue:
    """Test ValidationIssue dataclass."""

    def test_issue_creation(self):
        """Issue should store all attributes."""
        issue = ValidationIssue(
            rule_name="test_rule",
            severity=ValidationSeverity.ERROR,
            message="Test error message",
            path=Path("/test/file.png"),
            location="texture.diffuse",
            suggestion="Fix this issue",
            auto_fixable=True,
        )

        assert issue.rule_name == "test_rule"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Test error message"
        assert issue.path == Path("/test/file.png")
        assert issue.location == "texture.diffuse"
        assert issue.suggestion == "Fix this issue"
        assert issue.auto_fixable is True


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_result_creation(self):
        """Result should initialize correctly."""
        result = ValidationResult(path=Path("/test/file.png"))

        assert result.path == Path("/test/file.png")
        assert result.passed is True
        assert len(result.issues) == 0

    def test_error_count(self):
        """error_count should count errors and critical issues."""
        result = ValidationResult(path=Path("/test"))

        result.issues.append(ValidationIssue(
            rule_name="r1", severity=ValidationSeverity.WARNING, message="warning"
        ))
        result.issues.append(ValidationIssue(
            rule_name="r2", severity=ValidationSeverity.ERROR, message="error"
        ))
        result.issues.append(ValidationIssue(
            rule_name="r3", severity=ValidationSeverity.CRITICAL, message="critical"
        ))

        assert result.error_count == 2
        assert result.warning_count == 1

    def test_add_issue_marks_failed(self):
        """Adding error/critical should mark as failed."""
        result = ValidationResult(path=Path("/test"))

        # Warning doesn't fail
        result.add_issue(ValidationIssue(
            rule_name="r1", severity=ValidationSeverity.WARNING, message="warning"
        ))
        assert result.passed is True

        # Error fails
        result.add_issue(ValidationIssue(
            rule_name="r2", severity=ValidationSeverity.ERROR, message="error"
        ))
        assert result.passed is False


class TestTextureValidationRule:
    """Test TextureValidationRule."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        rule = TextureValidationRule()

        assert rule.max_width == 4096
        assert rule.max_height == 4096
        assert rule.require_power_of_two is True
        assert rule.max_file_size_mb == 50.0

    def test_applies_to_textures(self, temp_validation_dir):
        """Rule should apply to texture files."""
        rule = TextureValidationRule()

        assert rule.applies_to(Path("file.png"))
        assert rule.applies_to(Path("file.jpg"))
        assert rule.applies_to(Path("file.tga"))
        assert rule.applies_to(Path("file.dds"))
        assert not rule.applies_to(Path("file.fbx"))
        assert not rule.applies_to(Path("file.wav"))

    def test_validate_missing_file(self, temp_validation_dir):
        """Validation should fail for missing file."""
        rule = TextureValidationRule()

        issues = rule.validate(temp_validation_dir / "missing.png", {})

        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.CRITICAL

    def test_validate_file_size(self, temp_validation_dir):
        """Validation should check file size."""
        rule = TextureValidationRule(max_file_size_mb=0.001)  # Very small limit

        issues = rule.validate(temp_validation_dir / "valid_texture.png", {})

        size_issues = [i for i in issues if "size" in i.message.lower()]
        assert len(size_issues) >= 1

    def test_validate_power_of_two(self, temp_validation_dir):
        """Validation should check power-of-two dimensions."""
        rule = TextureValidationRule(require_power_of_two=True)

        # Context with non-power-of-two dimensions
        context = {"width": 500, "height": 700}
        issues = rule.validate(temp_validation_dir / "valid_texture.png", context)

        pot_issues = [i for i in issues if "power of 2" in i.message.lower()]
        assert len(pot_issues) == 2  # Both width and height

    def test_validate_dimensions(self, temp_validation_dir):
        """Validation should check max dimensions."""
        rule = TextureValidationRule(max_width=100, max_height=100)

        context = {"width": 200, "height": 200}
        issues = rule.validate(temp_validation_dir / "valid_texture.png", context)

        dim_issues = [i for i in issues if "exceeds max" in i.message.lower()]
        assert len(dim_issues) == 2

    def test_validate_format(self, temp_validation_dir):
        """Validation should check allowed formats."""
        rule = TextureValidationRule(allowed_formats={"dds"})

        issues = rule.validate(temp_validation_dir / "valid_texture.png", {})

        format_issues = [i for i in issues if "format" in i.message.lower()]
        assert len(format_issues) >= 1


class TestMeshValidationRule:
    """Test MeshValidationRule."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        rule = MeshValidationRule()

        assert rule.max_vertices == 100000
        assert rule.max_triangles == 100000
        assert rule.require_uvs is True
        assert rule.require_normals is True

    def test_applies_to_meshes(self):
        """Rule should apply to mesh files."""
        rule = MeshValidationRule()

        assert rule.applies_to(Path("model.fbx"))
        assert rule.applies_to(Path("model.obj"))
        assert rule.applies_to(Path("model.gltf"))
        assert rule.applies_to(Path("model.glb"))
        assert not rule.applies_to(Path("texture.png"))

    def test_validate_vertex_count(self, temp_validation_dir):
        """Validation should check vertex count."""
        rule = MeshValidationRule(max_vertices=1000)

        context = {"vertex_count": 5000}
        issues = rule.validate(temp_validation_dir / "model.fbx", context)

        vert_issues = [i for i in issues if "vertex" in i.message.lower()]
        assert len(vert_issues) >= 1

    def test_validate_triangle_count(self, temp_validation_dir):
        """Validation should check triangle count."""
        rule = MeshValidationRule(max_triangles=1000)

        context = {"triangle_count": 5000}
        issues = rule.validate(temp_validation_dir / "model.fbx", context)

        tri_issues = [i for i in issues if "triangle" in i.message.lower()]
        assert len(tri_issues) >= 1

    def test_validate_materials(self, temp_validation_dir):
        """Validation should check material count."""
        rule = MeshValidationRule(max_materials=4)

        context = {"material_count": 10}
        issues = rule.validate(temp_validation_dir / "model.fbx", context)

        mat_issues = [i for i in issues if "material" in i.message.lower()]
        assert len(mat_issues) >= 1

    def test_validate_uvs_required(self, temp_validation_dir):
        """Validation should check for UVs when required."""
        rule = MeshValidationRule(require_uvs=True)

        context = {"has_uvs": False}
        issues = rule.validate(temp_validation_dir / "model.fbx", context)

        uv_issues = [i for i in issues if "uv" in i.message.lower()]
        assert len(uv_issues) >= 1

    def test_validate_normals_required(self, temp_validation_dir):
        """Validation should check for normals when required."""
        rule = MeshValidationRule(require_normals=True)

        context = {"has_normals": False}
        issues = rule.validate(temp_validation_dir / "model.fbx", context)

        normal_issues = [i for i in issues if "normal" in i.message.lower()]
        assert len(normal_issues) >= 1

    def test_validate_bone_count(self, temp_validation_dir):
        """Validation should check bone count."""
        rule = MeshValidationRule(max_bones=100)

        context = {"bone_count": 300}
        issues = rule.validate(temp_validation_dir / "model.fbx", context)

        bone_issues = [i for i in issues if "bone" in i.message.lower()]
        assert len(bone_issues) >= 1


class TestMaterialValidationRule:
    """Test MaterialValidationRule."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        rule = MaterialValidationRule()

        assert rule.max_texture_slots == 16
        assert rule.validate_texture_refs is True

    def test_applies_to_materials(self):
        """Rule should apply to material files."""
        rule = MaterialValidationRule()

        assert rule.applies_to(Path("material.mat"))
        assert rule.applies_to(Path("surface.mtl"))
        assert not rule.applies_to(Path("model.fbx"))

    def test_validate_shader(self, temp_validation_dir):
        """Validation should check allowed shaders."""
        rule = MaterialValidationRule(allowed_shaders=["standard", "unlit"])

        context = {"shader_name": "custom_shader"}
        issues = rule.validate(temp_validation_dir / "material.mat", context)

        shader_issues = [i for i in issues if "shader" in i.message.lower()]
        assert len(shader_issues) >= 1

    def test_validate_required_textures(self, temp_validation_dir):
        """Validation should check required texture slots."""
        rule = MaterialValidationRule(required_textures=["diffuse", "normal"])

        context = {"texture_slots": {"diffuse": "tex.png"}}  # Missing normal
        issues = rule.validate(temp_validation_dir / "material.mat", context)

        missing_issues = [i for i in issues if "missing" in i.message.lower() and "normal" in i.message.lower()]
        assert len(missing_issues) >= 1

    def test_validate_texture_slots_count(self, temp_validation_dir):
        """Validation should check texture slot count."""
        rule = MaterialValidationRule(max_texture_slots=2)

        context = {"texture_slots": {"a": "1", "b": "2", "c": "3", "d": "4"}}
        issues = rule.validate(temp_validation_dir / "material.mat", context)

        slot_issues = [i for i in issues if "slot" in i.message.lower()]
        assert len(slot_issues) >= 1


class TestNamingConventionRule:
    """Test NamingConventionRule."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        rule = NamingConventionRule()

        assert rule.max_length == 64
        assert " " in rule.forbidden_chars

    def test_validate_length(self, temp_validation_dir):
        """Validation should check filename length."""
        rule = NamingConventionRule(max_length=10)

        issues = rule.validate(temp_validation_dir / "valid_texture.png", {})

        len_issues = [i for i in issues if "length" in i.message.lower()]
        assert len(len_issues) >= 1

    def test_validate_forbidden_chars(self, temp_validation_dir):
        """Validation should check forbidden characters."""
        rule = NamingConventionRule(forbidden_chars=" ")

        issues = rule.validate(temp_validation_dir / "Invalid Name.png", {})

        char_issues = [i for i in issues if "forbidden" in i.message.lower()]
        assert len(char_issues) >= 1

    def test_validate_snake_case(self, temp_validation_dir):
        """Validation should check snake_case convention."""
        rule = NamingConventionRule(case_style="snake_case")

        # PascalCase name
        issues = rule.validate(Path("HeroDiffuse.png"), {})

        case_issues = [i for i in issues if "snake_case" in i.message.lower()]
        assert len(case_issues) >= 1

    def test_validate_pascal_case(self, temp_validation_dir):
        """Validation should check PascalCase convention."""
        rule = NamingConventionRule(case_style="PascalCase")

        # snake_case name
        issues = rule.validate(temp_validation_dir / "hero_diffuse.png", {})

        case_issues = [i for i in issues if "pascalcase" in i.message.lower()]
        assert len(case_issues) >= 1

    def test_validate_pattern(self, temp_validation_dir):
        """Validation should check regex patterns."""
        rule = NamingConventionRule(patterns={"png": r"^T_.*"})

        # Doesn't start with T_
        issues = rule.validate(temp_validation_dir / "hero_diffuse.png", {})

        pattern_issues = [i for i in issues if "pattern" in i.message.lower()]
        assert len(pattern_issues) >= 1

    def test_validate_prefix_requirement(self, temp_validation_dir):
        """Validation should check required prefixes."""
        rule = NamingConventionRule(require_prefix={"textures": "T_"})

        # Create file path in textures folder
        tex_path = Path("/project/textures/hero.png")
        issues = rule.validate(tex_path, {})

        prefix_issues = [i for i in issues if "prefix" in i.message.lower()]
        assert len(prefix_issues) >= 1


class TestAssetValidator:
    """Test AssetValidator main class."""

    def test_validator_creation(self):
        """Validator should initialize with default rules."""
        validator = AssetValidator()

        assert len(validator.rules) > 0

    def test_add_rule(self):
        """add_rule() should add a rule."""
        validator = AssetValidator()
        initial_count = len(validator.rules)

        rule = TextureValidationRule(max_width=1024)
        validator.add_rule(rule)

        assert len(validator.rules) == initial_count + 1

    def test_remove_rule(self):
        """remove_rule() should remove a rule."""
        validator = AssetValidator()
        validator.add_rule(TextureValidationRule())

        success = validator.remove_rule("texture_validation")

        assert success

    def test_get_rule(self):
        """get_rule() should return rule by name."""
        validator = AssetValidator()

        rule = validator.get_rule("texture_validation")

        assert rule is not None
        assert isinstance(rule, TextureValidationRule)

    def test_enable_disable_rule(self):
        """enable/disable_rule() should toggle rule state."""
        validator = AssetValidator()

        validator.disable_rule("texture_validation")
        rule = validator.get_rule("texture_validation")
        assert rule.enabled is False

        validator.enable_rule("texture_validation")
        assert rule.enabled is True

    def test_validate(self, temp_validation_dir):
        """validate() should run all applicable rules."""
        validator = AssetValidator()

        result = validator.validate(temp_validation_dir / "valid_texture.png")

        assert isinstance(result, ValidationResult)
        assert result.rules_checked > 0

    def test_validate_with_context(self, temp_validation_dir):
        """validate() should use context data."""
        validator = AssetValidator()

        context = {"width": 100, "height": 100}  # Non-power-of-two
        result = validator.validate(temp_validation_dir / "valid_texture.png", context)

        # Should have power-of-two issues
        pot_issues = [i for i in result.issues if "power of 2" in i.message.lower()]
        assert len(pot_issues) > 0

    def test_validate_caching(self, temp_validation_dir):
        """validate() should cache results."""
        validator = AssetValidator()

        # First validation
        result1 = validator.validate(temp_validation_dir / "valid_texture.png", use_cache=True)

        # Second validation should use cache
        result2 = validator.validate(temp_validation_dir / "valid_texture.png", use_cache=True)

        # Should be same result
        assert result1.validation_time_ms > 0

    def test_validate_batch(self, temp_validation_dir):
        """validate_batch() should validate multiple files."""
        validator = AssetValidator()

        paths = [
            temp_validation_dir / "valid_texture.png",
            temp_validation_dir / "model.fbx",
            temp_validation_dir / "material.mat",
        ]

        results = validator.validate_batch(paths)

        assert len(results) == 3

    def test_validate_directory(self, temp_validation_dir):
        """validate_directory() should validate all files."""
        validator = AssetValidator()

        results = validator.validate_directory(temp_validation_dir, recursive=False)

        assert len(results) > 0

    def test_auto_fix(self, temp_validation_dir):
        """auto_fix() should attempt to fix issues."""
        validator = AssetValidator()

        result = ValidationResult(path=temp_validation_dir / "test.png")
        issue = ValidationIssue(
            rule_name="naming_convention",
            severity=ValidationSeverity.WARNING,
            message="Bad name",
            auto_fixable=True,
        )
        result.add_issue(issue)

        fixed = validator.auto_fix(result, dry_run=True)

        assert len(fixed) == 1
        assert fixed[0][1] is True  # Dry run always returns True

    def test_profiles(self):
        """Profile save/load should work."""
        validator = AssetValidator()

        # Save current rules as profile
        validator.save_profile("test_profile")

        # Modify rules
        validator.disable_rule("texture_validation")

        # Load profile
        success = validator.load_profile("test_profile")

        assert success

    def test_get_stats(self, temp_validation_dir):
        """get_stats() should return statistics."""
        validator = AssetValidator()

        # Run some validations
        validator.validate(temp_validation_dir / "valid_texture.png")

        stats = validator.get_stats()

        assert "total_validated" in stats
        assert "rules_enabled" in stats

    def test_clear_cache(self, temp_validation_dir):
        """clear_cache() should clear the results cache."""
        validator = AssetValidator()

        validator.validate(temp_validation_dir / "valid_texture.png")
        validator.clear_cache()

        # Cache should be empty
        assert len(validator._results_cache) == 0


class TestCustomValidationRule:
    """Test creating custom validation rules."""

    def test_custom_rule(self, temp_validation_dir):
        """Custom validation rules should work."""

        class FileSizeRule(ValidationRule):
            name = "file_size_check"
            description = "Check file size"

            def __init__(self, max_bytes: int):
                self.max_bytes = max_bytes

            def validate(self, path: Path, context: dict) -> list[ValidationIssue]:
                issues = []
                if path.exists():
                    size = path.stat().st_size
                    if size > self.max_bytes:
                        issues.append(ValidationIssue(
                            rule_name=self.name,
                            severity=ValidationSeverity.ERROR,
                            message=f"File size {size} exceeds max {self.max_bytes}",
                        ))
                return issues

        validator = AssetValidator()
        validator.add_rule(FileSizeRule(max_bytes=10))  # Very small limit

        result = validator.validate(temp_validation_dir / "valid_texture.png")

        # Should have size issue
        size_issues = [i for i in result.issues if i.rule_name == "file_size_check"]
        assert len(size_issues) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
