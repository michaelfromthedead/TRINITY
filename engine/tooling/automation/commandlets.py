"""
Command-line utilities for build, cook, test, and validation.

Provides commandlets that can be run from the command line
or invoked programmatically for automation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union


class CommandletStatus(Enum):
    """Status of a commandlet execution."""

    SUCCESS = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class CommandletResult:
    """Result of a commandlet execution."""

    status: CommandletStatus
    exit_code: int = 0
    duration: float = 0.0
    message: str = ""
    output: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if commandlet succeeded."""
        return self.status == CommandletStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.name,
            "exit_code": self.exit_code,
            "duration": self.duration,
            "message": self.message,
            "errors": self.errors,
            "warnings": self.warnings,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


class Commandlet(ABC):
    """
    Base class for commandlets.

    Commandlets are self-contained command-line utilities that
    perform specific build, test, or automation tasks.
    """

    name: str = "commandlet"
    description: str = "Base commandlet"

    def __init__(
        self,
        project_path: Optional[str] = None,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.verbose = verbose
        self.dry_run = dry_run
        self._output: List[str] = []
        self._errors: List[str] = []
        self._warnings: List[str] = []

    @abstractmethod
    def execute(self, **kwargs) -> CommandletResult:
        """Execute the commandlet."""
        pass

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add commandlet-specific arguments to parser."""
        pass

    def log(self, message: str) -> None:
        """Log a message."""
        self._output.append(message)
        if self.verbose:
            print(f"[{self.name}] {message}")

    def error(self, message: str) -> None:
        """Log an error."""
        self._errors.append(message)
        print(f"[{self.name}] ERROR: {message}", file=sys.stderr)

    def warning(self, message: str) -> None:
        """Log a warning."""
        self._warnings.append(message)
        if self.verbose:
            print(f"[{self.name}] WARNING: {message}")

    def run_command(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run an external command."""
        if self.dry_run:
            self.log(f"Would run: {' '.join(command)}")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.log(f"Running: {' '.join(command)}")

        result = subprocess.run(
            command,
            cwd=cwd or str(self.project_path),
            capture_output=capture,
            text=True,
            timeout=timeout,
        )

        if result.stdout:
            self._output.append(result.stdout)
        if result.stderr:
            self._errors.append(result.stderr)

        return result


class CookCommandlet(Commandlet):
    """
    Cook commandlet for processing and packaging assets.

    Converts raw assets into optimized, platform-specific formats.
    """

    name = "cook"
    description = "Cook assets for target platform"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--platform",
            "-p",
            default="Windows",
            help="Target platform (Windows, Linux, Android, iOS)",
        )
        parser.add_argument(
            "--config",
            "-c",
            default="Shipping",
            choices=["Development", "Shipping", "Debug"],
            help="Build configuration",
        )
        parser.add_argument(
            "--maps",
            "-m",
            nargs="*",
            help="Specific maps to cook",
        )
        parser.add_argument(
            "--iterate",
            "-i",
            action="store_true",
            help="Enable iterative cooking",
        )
        parser.add_argument(
            "--compressed",
            action="store_true",
            help="Enable compression",
        )

    def execute(
        self,
        platform: str = "Windows",
        config: str = "Shipping",
        maps: Optional[List[str]] = None,
        iterate: bool = False,
        compressed: bool = True,
        **kwargs,
    ) -> CommandletResult:
        """Execute the cook commandlet."""
        start_time = time.perf_counter()
        artifacts = []

        self.log(f"Cooking for {platform} ({config})")

        # Validate project
        if not self._validate_project():
            return CommandletResult(
                status=CommandletStatus.FAILED,
                message="Invalid project structure",
                errors=self._errors,
            )

        # Determine cook output path
        cook_path = self.project_path / "Saved" / "Cooked" / platform
        if not self.dry_run:
            cook_path.mkdir(parents=True, exist_ok=True)

        # Cook assets
        try:
            # Process shaders
            self.log("Compiling shaders...")
            self._cook_shaders(platform, config)

            # Process textures
            self.log("Processing textures...")
            self._cook_textures(platform, compressed)

            # Process meshes
            self.log("Processing meshes...")
            self._cook_meshes(platform)

            # Process audio
            self.log("Processing audio...")
            self._cook_audio(platform)

            # Process maps
            if maps:
                for map_name in maps:
                    self.log(f"Cooking map: {map_name}")
                    self._cook_map(map_name, platform)
            else:
                self.log("Cooking all maps...")
                self._cook_all_maps(platform)

            # Generate asset registry
            self.log("Generating asset registry...")
            registry_path = cook_path / "AssetRegistry.bin"
            artifacts.append(str(registry_path))

            duration = time.perf_counter() - start_time
            self.log(f"Cook completed in {duration:.2f}s")

            return CommandletResult(
                status=CommandletStatus.SUCCESS,
                duration=duration,
                message=f"Successfully cooked for {platform}",
                output="\n".join(self._output),
                warnings=self._warnings,
                artifacts=artifacts,
            )

        except Exception as e:
            return CommandletResult(
                status=CommandletStatus.FAILED,
                duration=time.perf_counter() - start_time,
                message=f"Cook failed: {e}",
                output="\n".join(self._output),
                errors=self._errors + [str(e)],
            )

    def _validate_project(self) -> bool:
        """Validate project structure."""
        required_dirs = ["Content", "Config"]
        for dir_name in required_dirs:
            if not (self.project_path / dir_name).exists():
                self.error(f"Missing required directory: {dir_name}")
                return False
        return True

    def _cook_shaders(self, platform: str, config: str) -> None:
        """Cook shaders for platform."""
        if self.dry_run:
            self.log(f"Would cook shaders for {platform}")
            return
        # Shader cooking implementation would go here

    def _cook_textures(self, platform: str, compressed: bool) -> None:
        """Cook textures for platform."""
        if self.dry_run:
            self.log(f"Would cook textures for {platform}")
            return
        # Texture cooking implementation would go here

    def _cook_meshes(self, platform: str) -> None:
        """Cook meshes for platform."""
        if self.dry_run:
            self.log(f"Would cook meshes for {platform}")
            return
        # Mesh cooking implementation would go here

    def _cook_audio(self, platform: str) -> None:
        """Cook audio for platform."""
        if self.dry_run:
            self.log(f"Would cook audio for {platform}")
            return
        # Audio cooking implementation would go here

    def _cook_map(self, map_name: str, platform: str) -> None:
        """Cook a specific map."""
        if self.dry_run:
            self.log(f"Would cook map {map_name} for {platform}")
            return
        # Map cooking implementation would go here

    def _cook_all_maps(self, platform: str) -> None:
        """Cook all maps."""
        content_path = self.project_path / "Content" / "Maps"
        if content_path.exists():
            for map_file in content_path.glob("*.umap"):
                self._cook_map(map_file.stem, platform)


class BuildCommandlet(Commandlet):
    """
    Build commandlet for compiling the game.

    Compiles source code and links the executable.
    """

    name = "build"
    description = "Build game executable"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--platform",
            "-p",
            default="Windows",
            help="Target platform",
        )
        parser.add_argument(
            "--config",
            "-c",
            default="Development",
            choices=["Debug", "Development", "Shipping"],
            help="Build configuration",
        )
        parser.add_argument(
            "--clean",
            action="store_true",
            help="Clean before building",
        )
        parser.add_argument(
            "--jobs",
            "-j",
            type=int,
            default=0,
            help="Number of parallel jobs (0 = auto)",
        )

    def execute(
        self,
        platform: str = "Windows",
        config: str = "Development",
        clean: bool = False,
        jobs: int = 0,
        **kwargs,
    ) -> CommandletResult:
        """Execute the build commandlet."""
        start_time = time.perf_counter()
        artifacts = []

        self.log(f"Building for {platform} ({config})")

        # Clean if requested
        if clean:
            self.log("Cleaning intermediate files...")
            self._clean_build(platform)

        try:
            # Generate project files
            self.log("Generating project files...")
            self._generate_project_files()

            # Compile
            self.log("Compiling...")
            compile_result = self._compile(platform, config, jobs)

            if compile_result != 0:
                return CommandletResult(
                    status=CommandletStatus.FAILED,
                    exit_code=compile_result,
                    duration=time.perf_counter() - start_time,
                    message="Compilation failed",
                    output="\n".join(self._output),
                    errors=self._errors,
                )

            # Link
            self.log("Linking...")
            link_result = self._link(platform, config)

            if link_result != 0:
                return CommandletResult(
                    status=CommandletStatus.FAILED,
                    exit_code=link_result,
                    duration=time.perf_counter() - start_time,
                    message="Linking failed",
                    output="\n".join(self._output),
                    errors=self._errors,
                )

            # Collect artifacts
            output_path = self._get_output_path(platform, config)
            if output_path.exists():
                artifacts.append(str(output_path))

            duration = time.perf_counter() - start_time
            self.log(f"Build completed in {duration:.2f}s")

            return CommandletResult(
                status=CommandletStatus.SUCCESS,
                duration=duration,
                message=f"Successfully built for {platform}",
                output="\n".join(self._output),
                warnings=self._warnings,
                artifacts=artifacts,
            )

        except Exception as e:
            return CommandletResult(
                status=CommandletStatus.FAILED,
                duration=time.perf_counter() - start_time,
                message=f"Build failed: {e}",
                errors=self._errors + [str(e)],
            )

    def _clean_build(self, platform: str) -> None:
        """Clean build artifacts."""
        if self.dry_run:
            self.log(f"Would clean build for {platform}")
            return

        intermediate_path = self.project_path / "Intermediate" / platform
        if intermediate_path.exists():
            shutil.rmtree(intermediate_path)

    def _generate_project_files(self) -> None:
        """Generate project files."""
        if self.dry_run:
            self.log("Would generate project files")
            return
        # Project file generation would go here

    def _compile(self, platform: str, config: str, jobs: int) -> int:
        """Compile source code."""
        if self.dry_run:
            self.log(f"Would compile for {platform}")
            return 0
        # Compilation would go here
        return 0

    def _link(self, platform: str, config: str) -> int:
        """Link executable."""
        if self.dry_run:
            self.log(f"Would link for {platform}")
            return 0
        # Linking would go here
        return 0

    def _get_output_path(self, platform: str, config: str) -> Path:
        """Get output executable path."""
        return self.project_path / "Binaries" / platform / f"Game-{platform}-{config}"


class TestCommandlet(Commandlet):
    """
    Test commandlet for running automated tests.

    Runs unit tests, integration tests, and automation tests.
    """

    name = "test"
    description = "Run automated tests"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--filter",
            "-f",
            help="Test name filter pattern",
        )
        parser.add_argument(
            "--category",
            "-c",
            help="Test category to run",
        )
        parser.add_argument(
            "--parallel",
            action="store_true",
            help="Run tests in parallel",
        )
        parser.add_argument(
            "--report",
            "-r",
            help="Output report path",
        )
        parser.add_argument(
            "--format",
            choices=["console", "junit", "html"],
            default="console",
            help="Report format",
        )

    def execute(
        self,
        filter: Optional[str] = None,
        category: Optional[str] = None,
        parallel: bool = False,
        report: Optional[str] = None,
        format: str = "console",
        **kwargs,
    ) -> CommandletResult:
        """Execute the test commandlet."""
        start_time = time.perf_counter()

        self.log("Running tests...")

        try:
            # Discover tests
            self.log("Discovering tests...")
            tests = self._discover_tests(filter, category)

            if not tests:
                return CommandletResult(
                    status=CommandletStatus.SUCCESS,
                    duration=time.perf_counter() - start_time,
                    message="No tests found",
                )

            self.log(f"Found {len(tests)} tests")

            # Run tests
            passed = 0
            failed = 0
            test_results = []

            for test in tests:
                self.log(f"Running: {test}")
                result = self._run_test(test)
                test_results.append(result)

                if result["passed"]:
                    passed += 1
                else:
                    failed += 1
                    self.error(f"Test failed: {test}")

            # Generate report
            if report:
                self.log(f"Generating {format} report...")
                self._generate_report(test_results, report, format)

            duration = time.perf_counter() - start_time

            if failed > 0:
                return CommandletResult(
                    status=CommandletStatus.FAILED,
                    duration=duration,
                    message=f"{failed}/{passed + failed} tests failed",
                    output="\n".join(self._output),
                    errors=self._errors,
                    metadata={"passed": passed, "failed": failed},
                )

            return CommandletResult(
                status=CommandletStatus.SUCCESS,
                duration=duration,
                message=f"All {passed} tests passed",
                output="\n".join(self._output),
                metadata={"passed": passed, "failed": failed},
            )

        except Exception as e:
            return CommandletResult(
                status=CommandletStatus.FAILED,
                duration=time.perf_counter() - start_time,
                message=f"Test execution failed: {e}",
                errors=[str(e)],
            )

    def _discover_tests(
        self,
        filter: Optional[str],
        category: Optional[str],
    ) -> List[str]:
        """Discover tests to run."""
        # Test discovery implementation would go here
        return []

    def _run_test(self, test_name: str) -> Dict[str, Any]:
        """Run a single test."""
        # Test execution implementation would go here
        return {"name": test_name, "passed": True, "duration": 0.0}

    def _generate_report(
        self,
        results: List[Dict],
        path: str,
        format: str,
    ) -> None:
        """Generate test report."""
        # Report generation would go here
        pass


class ValidateCommandlet(Commandlet):
    """
    Validate commandlet for validating assets and content.

    Checks for errors, warnings, and best practices violations.
    """

    name = "validate"
    description = "Validate assets and content"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--content",
            "-c",
            help="Content path to validate",
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help="Treat warnings as errors",
        )
        parser.add_argument(
            "--fix",
            action="store_true",
            help="Attempt to fix issues automatically",
        )

    def execute(
        self,
        content: Optional[str] = None,
        strict: bool = False,
        fix: bool = False,
        **kwargs,
    ) -> CommandletResult:
        """Execute the validate commandlet."""
        start_time = time.perf_counter()

        content_path = Path(content) if content else self.project_path / "Content"
        self.log(f"Validating: {content_path}")

        issues = []

        try:
            # Validate textures
            self.log("Validating textures...")
            texture_issues = self._validate_textures(content_path)
            issues.extend(texture_issues)

            # Validate meshes
            self.log("Validating meshes...")
            mesh_issues = self._validate_meshes(content_path)
            issues.extend(mesh_issues)

            # Validate blueprints
            self.log("Validating blueprints...")
            bp_issues = self._validate_blueprints(content_path)
            issues.extend(bp_issues)

            # Validate references
            self.log("Validating asset references...")
            ref_issues = self._validate_references(content_path)
            issues.extend(ref_issues)

            # Fix issues if requested
            if fix:
                self.log("Attempting to fix issues...")
                fixed = self._fix_issues(issues)
                self.log(f"Fixed {fixed} issues")

            # Categorize issues
            errors = [i for i in issues if i["severity"] == "error"]
            warnings = [i for i in issues if i["severity"] == "warning"]

            duration = time.perf_counter() - start_time

            if errors or (strict and warnings):
                self._errors.extend([i["message"] for i in errors])
                if strict:
                    self._errors.extend([i["message"] for i in warnings])

                return CommandletResult(
                    status=CommandletStatus.FAILED,
                    duration=duration,
                    message=f"Validation failed: {len(errors)} errors, {len(warnings)} warnings",
                    errors=self._errors,
                    warnings=[i["message"] for i in warnings],
                )

            return CommandletResult(
                status=CommandletStatus.SUCCESS,
                duration=duration,
                message=f"Validation passed with {len(warnings)} warnings",
                warnings=[i["message"] for i in warnings],
            )

        except Exception as e:
            return CommandletResult(
                status=CommandletStatus.FAILED,
                duration=time.perf_counter() - start_time,
                message=f"Validation failed: {e}",
                errors=[str(e)],
            )

    def _validate_textures(self, path: Path) -> List[Dict]:
        """Validate texture assets."""
        issues = []
        # Texture validation would go here
        return issues

    def _validate_meshes(self, path: Path) -> List[Dict]:
        """Validate mesh assets."""
        issues = []
        # Mesh validation would go here
        return issues

    def _validate_blueprints(self, path: Path) -> List[Dict]:
        """Validate blueprint assets."""
        issues = []
        # Blueprint validation would go here
        return issues

    def _validate_references(self, path: Path) -> List[Dict]:
        """Validate asset references."""
        issues = []
        # Reference validation would go here
        return issues

    def _fix_issues(self, issues: List[Dict]) -> int:
        """Attempt to fix issues."""
        fixed = 0
        # Issue fixing would go here
        return fixed


class CleanCommandlet(Commandlet):
    """Clean commandlet for removing build artifacts."""

    name = "clean"
    description = "Clean build artifacts"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--all",
            "-a",
            action="store_true",
            help="Clean all artifacts including cached data",
        )

    def execute(self, all: bool = False, **kwargs) -> CommandletResult:
        """Execute the clean commandlet."""
        start_time = time.perf_counter()

        self.log("Cleaning build artifacts...")

        dirs_to_clean = [
            "Intermediate",
            "Binaries",
            "Saved/Cooked",
        ]

        if all:
            dirs_to_clean.extend([
                "Saved/Logs",
                "DerivedDataCache",
            ])

        cleaned = 0
        for dir_name in dirs_to_clean:
            dir_path = self.project_path / dir_name
            if dir_path.exists():
                self.log(f"Cleaning: {dir_path}")
                if not self.dry_run:
                    shutil.rmtree(dir_path)
                cleaned += 1

        return CommandletResult(
            status=CommandletStatus.SUCCESS,
            duration=time.perf_counter() - start_time,
            message=f"Cleaned {cleaned} directories",
        )


class PackageCommandlet(Commandlet):
    """Package commandlet for creating distributable packages."""

    name = "package"
    description = "Create distributable package"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--platform",
            "-p",
            default="Windows",
            help="Target platform",
        )
        parser.add_argument(
            "--output",
            "-o",
            help="Output directory",
        )
        parser.add_argument(
            "--archive",
            choices=["zip", "tar", "none"],
            default="zip",
            help="Archive format",
        )

    def execute(
        self,
        platform: str = "Windows",
        output: Optional[str] = None,
        archive: str = "zip",
        **kwargs,
    ) -> CommandletResult:
        """Execute the package commandlet."""
        start_time = time.perf_counter()

        output_path = Path(output) if output else self.project_path / "Package"
        self.log(f"Packaging for {platform} to {output_path}")

        try:
            # Create output directory
            if not self.dry_run:
                output_path.mkdir(parents=True, exist_ok=True)

            # Copy binaries
            self.log("Copying binaries...")
            # Binary copying would go here

            # Copy cooked content
            self.log("Copying cooked content...")
            # Content copying would go here

            # Create archive
            if archive != "none":
                self.log(f"Creating {archive} archive...")
                # Archive creation would go here

            return CommandletResult(
                status=CommandletStatus.SUCCESS,
                duration=time.perf_counter() - start_time,
                message=f"Package created at {output_path}",
                artifacts=[str(output_path)],
            )

        except Exception as e:
            return CommandletResult(
                status=CommandletStatus.FAILED,
                duration=time.perf_counter() - start_time,
                message=f"Packaging failed: {e}",
                errors=[str(e)],
            )


class CommandletRunner:
    """Runner for executing commandlets."""

    _commandlets: Dict[str, Type[Commandlet]] = {
        "cook": CookCommandlet,
        "build": BuildCommandlet,
        "test": TestCommandlet,
        "validate": ValidateCommandlet,
        "clean": CleanCommandlet,
        "package": PackageCommandlet,
    }

    @classmethod
    def register(cls, commandlet_class: Type[Commandlet]) -> None:
        """Register a commandlet class."""
        cls._commandlets[commandlet_class.name] = commandlet_class

    @classmethod
    def get_commandlet(cls, name: str) -> Optional[Type[Commandlet]]:
        """Get a commandlet class by name."""
        return cls._commandlets.get(name)

    @classmethod
    def list_commandlets(cls) -> List[str]:
        """List all registered commandlets."""
        return list(cls._commandlets.keys())

    @classmethod
    def run(
        cls,
        name: str,
        project_path: Optional[str] = None,
        verbose: bool = False,
        dry_run: bool = False,
        **kwargs,
    ) -> CommandletResult:
        """Run a commandlet by name."""
        commandlet_class = cls._commandlets.get(name)
        if not commandlet_class:
            return CommandletResult(
                status=CommandletStatus.FAILED,
                message=f"Unknown commandlet: {name}",
            )

        commandlet = commandlet_class(
            project_path=project_path,
            verbose=verbose,
            dry_run=dry_run,
        )

        return commandlet.execute(**kwargs)


def run_commandlet(
    name: str,
    project_path: Optional[str] = None,
    **kwargs,
) -> CommandletResult:
    """Convenience function to run a commandlet."""
    return CommandletRunner.run(name, project_path=project_path, **kwargs)
