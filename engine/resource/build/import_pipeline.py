"""Import pipeline — first stage of the build pipeline, reads raw source files."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from engine.resource.constants import DEFAULT_IMPORT_OUTPUT_PATH


@dataclass(slots=True)
class ImportSettings:
    """Settings for importing an asset."""

    source_path: str
    output_path: str
    options: dict = field(default_factory=dict)


@dataclass(slots=True)
class ImportResult:
    """Result of an import operation."""

    success: bool
    output_path: Optional[str]
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Importer(ABC):
    """Abstract base for asset importers."""

    __slots__ = ()

    @abstractmethod
    def can_import(self, path: str) -> bool:
        """Return True if this importer can handle the given path."""

    @abstractmethod
    def import_asset(self, settings: ImportSettings) -> ImportResult:
        """Import the asset described by *settings*."""


class ImporterRegistry:
    """Registry of available importers."""

    __slots__ = ("_importers",)

    def __init__(self) -> None:
        self._importers: list[Importer] = []

    def register(self, importer: Importer) -> None:
        """Register an importer."""
        self._importers.append(importer)

    def find_importer(self, path: str) -> Optional[Importer]:
        """Find the first importer that can handle *path*."""
        for importer in self._importers:
            if importer.can_import(path):
                return importer
        return None


class ImportPipeline:
    """Runs the import stage of the build pipeline."""

    __slots__ = ("_registry", "_output_path")

    def __init__(self, registry: ImporterRegistry, output_path: str = DEFAULT_IMPORT_OUTPUT_PATH) -> None:
        self._registry = registry
        self._output_path = output_path

    def run(self, source_path: str) -> ImportResult:
        """Import the asset at *source_path*."""
        importer = self._registry.find_importer(source_path)
        if importer is None:
            return ImportResult(
                success=False,
                output_path=None,
                errors=[f"No importer found for '{source_path}'"],
            )
        settings = ImportSettings(
            source_path=source_path,
            output_path=self._output_path,
        )
        try:
            return importer.import_asset(settings)
        except Exception as exc:  # noqa: BLE001
            return ImportResult(
                success=False,
                output_path=None,
                errors=[f"Import failed: {exc}"],
            )
