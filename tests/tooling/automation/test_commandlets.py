"""
Tests for commandlets.py - All commandlets (cook, build, test, validate).
"""

import pytest

from engine.tooling.automation.commandlets import (
    BuildCommandlet,
    CleanCommandlet,
    Commandlet,
    CommandletResult,
    CommandletRunner,
    CommandletStatus,
    CookCommandlet,
    PackageCommandlet,
    TestCommandlet,
    ValidateCommandlet,
    run_commandlet,
)


class TestCommandletStatus:
    """Tests for CommandletStatus enum."""

    def test_status_values_exist(self):
        assert CommandletStatus.SUCCESS
        assert CommandletStatus.FAILED
        assert CommandletStatus.CANCELLED
        assert CommandletStatus.TIMEOUT


class TestCommandletResult:
    """Tests for CommandletResult dataclass."""

    def test_create_result(self):
        result = CommandletResult(status=CommandletStatus.SUCCESS)
        assert result.status == CommandletStatus.SUCCESS
        assert result.success is True

    def test_success_property(self):
        success = CommandletResult(status=CommandletStatus.SUCCESS)
        failed = CommandletResult(status=CommandletStatus.FAILED)

        assert success.success is True
        assert failed.success is False

    def test_to_dict(self):
        result = CommandletResult(
            status=CommandletStatus.SUCCESS,
            duration=10.5,
            message="Build complete",
        )
        data = result.to_dict()

        assert data["status"] == "SUCCESS"
        assert data["duration"] == 10.5
        assert data["message"] == "Build complete"


class TestCookCommandlet:
    """Tests for CookCommandlet."""

    def test_create_commandlet(self, tmp_path):
        commandlet = CookCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )
        assert commandlet.name == "cook"

    def test_execute_dry_run(self, tmp_path):
        # Create minimal project structure
        (tmp_path / "Content").mkdir()
        (tmp_path / "Config").mkdir()

        commandlet = CookCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
            verbose=True,
        )

        result = commandlet.execute(platform="Windows", config="Shipping")
        assert result.status == CommandletStatus.SUCCESS

    def test_execute_invalid_project(self, tmp_path):
        # No Content or Config directories
        commandlet = CookCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )

        result = commandlet.execute()
        assert result.status == CommandletStatus.FAILED


class TestBuildCommandlet:
    """Tests for BuildCommandlet."""

    def test_create_commandlet(self, tmp_path):
        commandlet = BuildCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )
        assert commandlet.name == "build"

    def test_execute_dry_run(self, tmp_path):
        commandlet = BuildCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )

        result = commandlet.execute(platform="Windows", config="Development")
        assert result.status == CommandletStatus.SUCCESS

    def test_execute_with_clean(self, tmp_path):
        commandlet = BuildCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )

        result = commandlet.execute(clean=True)
        assert result.status == CommandletStatus.SUCCESS


class TestTestCommandlet:
    """Tests for TestCommandlet."""

    def test_create_commandlet(self, tmp_path):
        commandlet = TestCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )
        assert commandlet.name == "test"

    def test_execute_dry_run(self, tmp_path):
        commandlet = TestCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )

        result = commandlet.execute()
        # With no tests found, should still succeed
        assert result.status == CommandletStatus.SUCCESS


class TestValidateCommandlet:
    """Tests for ValidateCommandlet."""

    def test_create_commandlet(self, tmp_path):
        commandlet = ValidateCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )
        assert commandlet.name == "validate"

    def test_execute_dry_run(self, tmp_path):
        (tmp_path / "Content").mkdir()

        commandlet = ValidateCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )

        result = commandlet.execute()
        assert result.status == CommandletStatus.SUCCESS


class TestCleanCommandlet:
    """Tests for CleanCommandlet."""

    def test_create_commandlet(self, tmp_path):
        commandlet = CleanCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )
        assert commandlet.name == "clean"

    def test_execute_dry_run(self, tmp_path):
        commandlet = CleanCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )

        result = commandlet.execute()
        assert result.status == CommandletStatus.SUCCESS

    def test_execute_with_all_flag(self, tmp_path):
        commandlet = CleanCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )

        result = commandlet.execute(all=True)
        assert result.status == CommandletStatus.SUCCESS


class TestPackageCommandlet:
    """Tests for PackageCommandlet."""

    def test_create_commandlet(self, tmp_path):
        commandlet = PackageCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )
        assert commandlet.name == "package"

    def test_execute_dry_run(self, tmp_path):
        commandlet = PackageCommandlet(
            project_path=str(tmp_path),
            dry_run=True,
        )

        result = commandlet.execute(platform="Windows")
        assert result.status == CommandletStatus.SUCCESS


class TestCommandletRunner:
    """Tests for CommandletRunner class."""

    def test_list_commandlets(self):
        commandlets = CommandletRunner.list_commandlets()

        assert "cook" in commandlets
        assert "build" in commandlets
        assert "test" in commandlets
        assert "validate" in commandlets
        assert "clean" in commandlets
        assert "package" in commandlets

    def test_get_commandlet(self):
        cook_cls = CommandletRunner.get_commandlet("cook")
        assert cook_cls == CookCommandlet

        unknown = CommandletRunner.get_commandlet("unknown")
        assert unknown is None

    def test_run_commandlet(self, tmp_path):
        result = CommandletRunner.run(
            "clean",
            project_path=str(tmp_path),
            dry_run=True,
        )
        assert result.status == CommandletStatus.SUCCESS

    def test_run_unknown_commandlet(self):
        result = CommandletRunner.run("nonexistent")
        assert result.status == CommandletStatus.FAILED
        assert "Unknown commandlet" in result.message

    def test_register_custom_commandlet(self, tmp_path):
        class CustomCommandlet(Commandlet):
            name = "custom"
            description = "Custom commandlet"

            def execute(self, **kwargs):
                return CommandletResult(
                    status=CommandletStatus.SUCCESS,
                    message="Custom executed",
                )

        CommandletRunner.register(CustomCommandlet)

        assert "custom" in CommandletRunner.list_commandlets()

        result = CommandletRunner.run("custom", project_path=str(tmp_path))
        assert result.status == CommandletStatus.SUCCESS


class TestRunCommandletFunction:
    """Tests for run_commandlet convenience function."""

    def test_run_commandlet(self, tmp_path):
        result = run_commandlet("clean", project_path=str(tmp_path), dry_run=True)
        assert result.status == CommandletStatus.SUCCESS


class TestCommandletBase:
    """Tests for Commandlet base class functionality."""

    def test_log_method(self, tmp_path):
        commandlet = CleanCommandlet(
            project_path=str(tmp_path),
            verbose=True,
        )

        commandlet.log("Test message")
        assert "Test message" in commandlet._output

    def test_error_method(self, tmp_path):
        commandlet = CleanCommandlet(
            project_path=str(tmp_path),
        )

        commandlet.error("Error message")
        assert "Error message" in commandlet._errors

    def test_warning_method(self, tmp_path):
        commandlet = CleanCommandlet(
            project_path=str(tmp_path),
            verbose=True,
        )

        commandlet.warning("Warning message")
        assert "Warning message" in commandlet._warnings
