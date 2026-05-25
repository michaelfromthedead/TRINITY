"""
Tests for python_api.py - Python API for scripting automation.
"""

import os

import pytest

from engine.tooling.automation.python_api import (
    AssetAPI,
    AutomationAPI,
    BuildAPI,
    DeployAPI,
    ScriptContext,
    TestAPI,
    execute_command,
    run_script,
)


class TestScriptContext:
    """Tests for ScriptContext class."""

    def test_create_context(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        assert ctx.project_path == tmp_path

    def test_get_config_nested(self, tmp_path):
        ctx = ScriptContext(
            project_path=tmp_path,
            config={"build": {"platform": "Windows"}},
        )

        assert ctx.get_config("build.platform") == "Windows"
        assert ctx.get_config("build.missing", "default") == "default"

    def test_set_and_get_variable(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        ctx.set_variable("test_var", 123)

        assert ctx.get_variable("test_var") == 123
        assert ctx.get_variable("missing", "default") == "default"

    def test_resolve_path_relative(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        resolved = ctx.resolve_path("Content/Maps")

        assert resolved == tmp_path / "Content" / "Maps"

    def test_resolve_path_absolute(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        abs_path = "/absolute/path"
        resolved = ctx.resolve_path(abs_path)

        assert str(resolved) == abs_path


class TestAutomationAPI:
    """Tests for AutomationAPI class."""

    def test_create_api(self, tmp_path):
        api = AutomationAPI(project_path=str(tmp_path))

        assert api.context.project_path == tmp_path
        assert api.build is not None
        assert api.test is not None
        assert api.asset is not None
        assert api.deploy is not None

    def test_run_commandlet(self, tmp_path):
        api = AutomationAPI(project_path=str(tmp_path))
        result = api.run_commandlet("clean", dry_run=True)

        assert result.success

    def test_set_environment(self, tmp_path):
        api = AutomationAPI(project_path=str(tmp_path))
        api.set_environment("TEST_VAR", "test_value")

        assert os.environ.get("TEST_VAR") == "test_value"
        assert api.context.environment.get("TEST_VAR") == "test_value"

    def test_load_config(self, tmp_path):
        import json

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"setting": "value"}))

        api = AutomationAPI(project_path=str(tmp_path))
        config = api.load_config("config.json")

        assert config["setting"] == "value"
        assert api.context.config["setting"] == "value"


class TestBuildAPI:
    """Tests for BuildAPI class."""

    def test_create_api(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        build = BuildAPI(ctx)

        assert build is not None

    def test_build(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        build = BuildAPI(ctx)

        # Should use dry_run internally
        result = build.build(platform="Windows", dry_run=True)
        assert result is not None

    def test_cook(self, tmp_path):
        # Create minimal project structure
        (tmp_path / "Content").mkdir()
        (tmp_path / "Config").mkdir()

        ctx = ScriptContext(project_path=tmp_path)
        build = BuildAPI(ctx)

        result = build.cook(platform="Windows", dry_run=True)
        assert result is not None

    def test_clean(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        build = BuildAPI(ctx)

        result = build.clean()
        assert result.success


class TestTestAPI:
    """Tests for TestAPI class."""

    def test_create_api(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        test = TestAPI(ctx)

        assert test is not None

    def test_run_tests(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        test = TestAPI(ctx)

        result = test.run_tests()
        assert result is not None

    def test_run_unit_tests(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        test = TestAPI(ctx)

        result = test.run_unit_tests()
        assert result is not None

    def test_validate(self, tmp_path):
        (tmp_path / "Content").mkdir()

        ctx = ScriptContext(project_path=tmp_path)
        test = TestAPI(ctx)

        result = test.validate()
        assert result is not None


class TestAssetAPI:
    """Tests for AssetAPI class."""

    def test_create_api(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        asset = AssetAPI(ctx)

        assert asset is not None

    def test_import_asset(self, tmp_path):
        # Create source file
        source = tmp_path / "source.txt"
        source.write_text("content")

        ctx = ScriptContext(project_path=tmp_path)
        asset = AssetAPI(ctx)

        result = asset.import_asset(str(source), "Content/imported.txt")
        assert result is True

    def test_import_nonexistent_asset(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        asset = AssetAPI(ctx)

        result = asset.import_asset("/nonexistent/file.txt", "Content/file.txt")
        assert result is False

    def test_validate_asset(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        asset = AssetAPI(ctx)

        result = asset.validate_asset("Content/test.asset")
        assert "valid" in result

    def test_get_asset_info(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        asset = AssetAPI(ctx)

        info = asset.get_asset_info("Content/test.asset")
        assert "path" in info
        assert "type" in info

    def test_find_assets(self, tmp_path):
        # Create content directory with files
        content_dir = tmp_path / "Content"
        content_dir.mkdir()
        (content_dir / "file1.txt").write_text("1")
        (content_dir / "file2.txt").write_text("2")

        ctx = ScriptContext(project_path=tmp_path)
        asset = AssetAPI(ctx)

        files = asset.find_assets("*.txt")
        assert len(files) == 2


class TestDeployAPI:
    """Tests for DeployAPI class."""

    def test_create_api(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        deploy = DeployAPI(ctx)

        assert deploy is not None

    def test_deploy(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        deploy = DeployAPI(ctx)

        result = deploy.deploy("staging")
        assert result is True

    def test_notify(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        deploy = DeployAPI(ctx)

        result = deploy.notify("Build complete")
        assert result is True

    def test_create_release(self, tmp_path):
        ctx = ScriptContext(project_path=tmp_path)
        deploy = DeployAPI(ctx)

        result = deploy.create_release("1.0.0", notes="Release notes")
        assert result["version"] == "1.0.0"
        assert result["created"] is True


class TestExecuteCommand:
    """Tests for execute_command function."""

    def test_execute_simple_command(self):
        result = execute_command("echo hello", check=False)
        assert result.returncode == 0

    def test_execute_command_list(self):
        result = execute_command(["echo", "hello"], check=False)
        assert result.returncode == 0

    def test_execute_with_cwd(self, tmp_path):
        result = execute_command("pwd", cwd=str(tmp_path), check=False)
        assert result.returncode == 0


class TestRunScript:
    """Tests for run_script function."""

    def test_run_simple_script(self, tmp_path):
        script = tmp_path / "test_script.py"
        script.write_text("""
result = api.run_commandlet("clean", dry_run=True)
""")

        run_script(str(script), project_path=str(tmp_path))
        # Should not raise

    def test_script_has_access_to_apis(self, tmp_path):
        script = tmp_path / "test_script.py"
        script.write_text("""
assert build is not None
assert test is not None
assert asset is not None
assert deploy is not None
assert context is not None
""")

        run_script(str(script), project_path=str(tmp_path))
        # Should not raise
