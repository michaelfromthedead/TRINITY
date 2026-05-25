"""Tests for the import pipeline."""
import pytest

from engine.resource.build.import_pipeline import (
    Importer,
    ImporterRegistry,
    ImportPipeline,
    ImportResult,
    ImportSettings,
)


class PngImporter(Importer):
    def can_import(self, path: str) -> bool:
        return path.endswith(".png")

    def import_asset(self, settings: ImportSettings) -> ImportResult:
        return ImportResult(
            success=True,
            output_path=f"{settings.output_path}/imported.png",
            metadata={"format": "png", "source": settings.source_path},
        )


class FailingImporter(Importer):
    def can_import(self, path: str) -> bool:
        return path.endswith(".bad")

    def import_asset(self, settings: ImportSettings) -> ImportResult:
        raise RuntimeError("disk error")


class TestImporterRegistry:
    def test_register_and_find(self) -> None:
        registry = ImporterRegistry()
        importer = PngImporter()
        registry.register(importer)
        assert registry.find_importer("texture.png") is importer

    def test_find_returns_none_for_unknown(self) -> None:
        registry = ImporterRegistry()
        assert registry.find_importer("file.xyz") is None

    def test_first_match_wins(self) -> None:
        registry = ImporterRegistry()
        first = PngImporter()
        second = PngImporter()
        registry.register(first)
        registry.register(second)
        assert registry.find_importer("a.png") is first


class TestImportPipeline:
    def test_successful_import(self) -> None:
        registry = ImporterRegistry()
        registry.register(PngImporter())
        pipeline = ImportPipeline(registry)
        result = pipeline.run("art/hero.png")
        assert result.success is True
        assert result.output_path is not None
        assert result.errors == []
        assert result.metadata["format"] == "png"

    def test_no_importer_found(self) -> None:
        registry = ImporterRegistry()
        pipeline = ImportPipeline(registry)
        result = pipeline.run("unknown.dat")
        assert result.success is False
        assert "No importer found" in result.errors[0]

    def test_importer_exception_handled(self) -> None:
        registry = ImporterRegistry()
        registry.register(FailingImporter())
        pipeline = ImportPipeline(registry)
        result = pipeline.run("crash.bad")
        assert result.success is False
        assert "Import failed" in result.errors[0]


class TestImportSettings:
    def test_defaults(self) -> None:
        s = ImportSettings(source_path="a.png", output_path="out/")
        assert s.options == {}

    def test_slots(self) -> None:
        s = ImportSettings(source_path="a.png", output_path="out/")
        assert not hasattr(s, "__dict__")
