"""
Python API for scripting automation.

Provides a comprehensive Python API for automating build,
test, and deployment workflows.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .commandlets import CommandletResult, CommandletRunner, CommandletStatus


@dataclass
class ScriptContext:
    """
    Context for automation scripts.

    Provides access to project information, environment,
    and helper functions.
    """

    project_path: Path
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = False

    def __post_init__(self):
        self.project_path = Path(self.project_path)
        if not self.environment:
            self.environment = dict(os.environ)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def set_variable(self, name: str, value: Any) -> None:
        """Set a script variable."""
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a script variable."""
        return self.variables.get(name, default)

    def resolve_path(self, path: str) -> Path:
        """Resolve a path relative to project."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.project_path / p

    def log(self, message: str) -> None:
        """Log a message."""
        if self.verbose:
            print(f"[Script] {message}")


class AutomationAPI:
    """
    Main automation API for scripting.

    Provides high-level methods for common automation tasks.
    """

    def __init__(
        self,
        project_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.context = ScriptContext(
            project_path=Path(project_path) if project_path else Path.cwd(),
            verbose=verbose,
        )
        self.build = BuildAPI(self.context)
        self.test = TestAPI(self.context)
        self.asset = AssetAPI(self.context)
        self.deploy = DeployAPI(self.context)

    def run_commandlet(self, name: str, **kwargs) -> CommandletResult:
        """Run a commandlet."""
        return CommandletRunner.run(
            name,
            project_path=str(self.context.project_path),
            verbose=self.context.verbose,
            **kwargs,
        )

    def execute(self, command: str, **kwargs) -> subprocess.CompletedProcess:
        """Execute a shell command."""
        return execute_command(
            command,
            cwd=str(self.context.project_path),
            env=self.context.environment,
            **kwargs,
        )

    def load_config(self, path: str) -> Dict[str, Any]:
        """Load a configuration file."""
        import json

        config_path = self.context.resolve_path(path)
        with open(config_path) as f:
            config = json.load(f)
        self.context.config.update(config)
        return config

    def set_environment(self, key: str, value: str) -> None:
        """Set an environment variable."""
        self.context.environment[key] = value
        os.environ[key] = value

    def log(self, message: str) -> None:
        """Log a message."""
        self.context.log(message)


class BuildAPI:
    """API for build operations."""

    def __init__(self, context: ScriptContext):
        self.context = context

    def build(
        self,
        platform: str = "Windows",
        config: str = "Development",
        clean: bool = False,
        **kwargs,
    ) -> CommandletResult:
        """Build the game."""
        return CommandletRunner.run(
            "build",
            project_path=str(self.context.project_path),
            verbose=self.context.verbose,
            platform=platform,
            config=config,
            clean=clean,
            **kwargs,
        )

    def cook(
        self,
        platform: str = "Windows",
        config: str = "Shipping",
        maps: Optional[List[str]] = None,
        **kwargs,
    ) -> CommandletResult:
        """Cook assets."""
        return CommandletRunner.run(
            "cook",
            project_path=str(self.context.project_path),
            verbose=self.context.verbose,
            platform=platform,
            config=config,
            maps=maps,
            **kwargs,
        )

    def package(
        self,
        platform: str = "Windows",
        output: Optional[str] = None,
        **kwargs,
    ) -> CommandletResult:
        """Package the game."""
        return CommandletRunner.run(
            "package",
            project_path=str(self.context.project_path),
            verbose=self.context.verbose,
            platform=platform,
            output=output,
            **kwargs,
        )

    def clean(self, all: bool = False) -> CommandletResult:
        """Clean build artifacts."""
        return CommandletRunner.run(
            "clean",
            project_path=str(self.context.project_path),
            verbose=self.context.verbose,
            all=all,
        )

    def rebuild(
        self,
        platform: str = "Windows",
        config: str = "Development",
        **kwargs,
    ) -> CommandletResult:
        """Clean and rebuild."""
        self.clean()
        return self.build(platform=platform, config=config, **kwargs)


class TestAPI:
    """API for testing operations."""

    def __init__(self, context: ScriptContext):
        self.context = context

    def run_tests(
        self,
        filter: Optional[str] = None,
        category: Optional[str] = None,
        parallel: bool = False,
        report: Optional[str] = None,
        **kwargs,
    ) -> CommandletResult:
        """Run tests."""
        return CommandletRunner.run(
            "test",
            project_path=str(self.context.project_path),
            verbose=self.context.verbose,
            filter=filter,
            category=category,
            parallel=parallel,
            report=report,
            **kwargs,
        )

    def run_unit_tests(self, **kwargs) -> CommandletResult:
        """Run unit tests."""
        return self.run_tests(category="unit", **kwargs)

    def run_integration_tests(self, **kwargs) -> CommandletResult:
        """Run integration tests."""
        return self.run_tests(category="integration", **kwargs)

    def run_automation_tests(self, **kwargs) -> CommandletResult:
        """Run automation tests."""
        return self.run_tests(category="automation", **kwargs)

    def validate(
        self,
        content: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> CommandletResult:
        """Validate assets."""
        return CommandletRunner.run(
            "validate",
            project_path=str(self.context.project_path),
            verbose=self.context.verbose,
            content=content,
            strict=strict,
            **kwargs,
        )


class AssetAPI:
    """API for asset operations."""

    def __init__(self, context: ScriptContext):
        self.context = context

    def import_asset(
        self,
        source: str,
        destination: str,
        **kwargs,
    ) -> bool:
        """Import an asset."""
        src = Path(source)
        dst = self.context.resolve_path(destination)

        if not src.exists():
            return False

        dst.parent.mkdir(parents=True, exist_ok=True)

        # Asset import would happen here
        self.context.log(f"Importing {src} to {dst}")
        return True

    def export_asset(
        self,
        source: str,
        destination: str,
        format: str = "fbx",
        **kwargs,
    ) -> bool:
        """Export an asset."""
        src = self.context.resolve_path(source)
        dst = Path(destination)

        if not src.exists():
            return False

        # Asset export would happen here
        self.context.log(f"Exporting {src} to {dst}")
        return True

    def validate_asset(self, path: str) -> Dict[str, Any]:
        """Validate a single asset."""
        asset_path = self.context.resolve_path(path)
        # Asset validation would happen here
        return {"valid": True, "warnings": [], "errors": []}

    def get_asset_info(self, path: str) -> Dict[str, Any]:
        """Get asset information."""
        asset_path = self.context.resolve_path(path)
        # Asset info retrieval would happen here
        return {
            "path": str(asset_path),
            "type": "unknown",
            "size": 0,
            "dependencies": [],
        }

    def find_assets(
        self,
        pattern: str,
        path: Optional[str] = None,
    ) -> List[str]:
        """Find assets matching a pattern."""
        search_path = self.context.resolve_path(path) if path else self.context.project_path / "Content"
        return [str(p) for p in search_path.glob(pattern)]


class DeployAPI:
    """API for deployment operations."""

    def __init__(self, context: ScriptContext):
        self.context = context

    def deploy(
        self,
        target: str,
        source: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Deploy to a target."""
        self.context.log(f"Deploying to {target}")
        # Deployment would happen here
        return True

    def upload(
        self,
        source: str,
        destination: str,
        **kwargs,
    ) -> bool:
        """Upload files to remote location."""
        self.context.log(f"Uploading {source} to {destination}")
        # Upload would happen here
        return True

    def notify(
        self,
        message: str,
        channels: Optional[List[str]] = None,
        **kwargs,
    ) -> bool:
        """Send notification."""
        self.context.log(f"Notifying: {message}")
        # Notification would happen here
        return True

    def create_release(
        self,
        version: str,
        notes: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a release."""
        self.context.log(f"Creating release {version}")
        # Release creation would happen here
        return {"version": version, "created": True}


def run_script(
    script_path: str,
    project_path: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Run an automation script.

    Args:
        script_path: Path to Python script
        project_path: Project path
        **kwargs: Additional arguments passed to script

    Returns:
        Script return value
    """
    api = AutomationAPI(project_path=project_path, **kwargs)

    # Load and execute script
    script_globals = {
        "api": api,
        "build": api.build,
        "test": api.test,
        "asset": api.asset,
        "deploy": api.deploy,
        "context": api.context,
        "execute": api.execute,
        "log": api.log,
        "__name__": "__automation__",
    }

    with open(script_path) as f:
        code = compile(f.read(), script_path, "exec")
        exec(code, script_globals)

    return script_globals.get("result")


def execute_command(
    command: Union[str, List[str]],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Execute a shell command.

    Args:
        command: Command to execute
        cwd: Working directory
        env: Environment variables
        timeout: Command timeout
        capture: Capture output
        check: Raise on error

    Returns:
        Completed process result
    """
    if isinstance(command, str):
        command = command.split()

    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        timeout=timeout,
        capture_output=capture,
        text=True,
        check=check,
    )
