"""Tests for the process pipeline."""
from typing import Any

from engine.resource.build.process_pipeline import (
    ProcessContext,
    ProcessPipeline,
    ProcessResult,
    ProcessStage,
    QualityLevel,
)


class UpperStage(ProcessStage):
    @property
    def name(self) -> str:
        return "upper"

    def process(self, data: Any, context: ProcessContext) -> ProcessResult:
        return ProcessResult(success=True, data=str(data).upper())


class PrefixStage(ProcessStage):
    @property
    def name(self) -> str:
        return "prefix"

    def process(self, data: Any, context: ProcessContext) -> ProcessResult:
        return ProcessResult(success=True, data=f"[{context.platform}]{data}")


class FailStage(ProcessStage):
    @property
    def name(self) -> str:
        return "fail"

    def process(self, data: Any, context: ProcessContext) -> ProcessResult:
        return ProcessResult(success=False, data=data, errors=["stage failed"])


class CrashStage(ProcessStage):
    @property
    def name(self) -> str:
        return "crash"

    def process(self, data: Any, context: ProcessContext) -> ProcessResult:
        raise ValueError("boom")


CTX = ProcessContext(platform="windows", quality=QualityLevel.HIGH)


class TestProcessPipeline:
    def test_add_and_run_single_stage(self) -> None:
        p = ProcessPipeline()
        p.add_stage(UpperStage())
        result = p.run("hello", CTX)
        assert result.success is True
        assert result.data == "HELLO"

    def test_sequential_execution(self) -> None:
        p = ProcessPipeline()
        p.add_stage(UpperStage())
        p.add_stage(PrefixStage())
        result = p.run("hello", CTX)
        assert result.data == "[windows]HELLO"

    def test_remove_stage(self) -> None:
        p = ProcessPipeline()
        p.add_stage(UpperStage())
        assert p.remove_stage("upper") is True
        assert p.remove_stage("nonexistent") is False
        assert len(p.stages) == 0

    def test_stage_failure_stops_pipeline(self) -> None:
        p = ProcessPipeline()
        p.add_stage(FailStage())
        p.add_stage(UpperStage())
        result = p.run("data", CTX)
        assert result.success is False
        assert "stage failed" in result.errors

    def test_stage_exception_handled(self) -> None:
        p = ProcessPipeline()
        p.add_stage(CrashStage())
        result = p.run("data", CTX)
        assert result.success is False
        assert any("boom" in e for e in result.errors)

    def test_empty_pipeline(self) -> None:
        p = ProcessPipeline()
        result = p.run("unchanged", CTX)
        assert result.success is True
        assert result.data == "unchanged"

    def test_quality_levels(self) -> None:
        for name in ("LOW", "MEDIUM", "HIGH", "ULTRA"):
            assert hasattr(QualityLevel, name), f"QualityLevel.{name} missing"
        assert QualityLevel.LOW.value < QualityLevel.MEDIUM.value < QualityLevel.HIGH.value < QualityLevel.ULTRA.value
