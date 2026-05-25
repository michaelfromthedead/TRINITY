"""Process pipeline — transforms imported assets through ordered stages."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QualityLevel(Enum):
    """Asset quality levels."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


@dataclass(slots=True)
class ProcessContext:
    """Context passed to every processing stage."""

    platform: str
    quality: QualityLevel
    debug: bool = False


@dataclass(slots=True)
class ProcessResult:
    """Result from a processing stage or the full pipeline."""

    success: bool
    data: Any
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ProcessStage(ABC):
    """Abstract base for a single processing stage."""

    __slots__ = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this stage."""

    @abstractmethod
    def process(self, data: Any, context: ProcessContext) -> ProcessResult:
        """Process *data* within *context* and return a result."""


class ProcessPipeline:
    """Runs an ordered sequence of processing stages."""

    __slots__ = ("_stages",)

    def __init__(self) -> None:
        self._stages: list[ProcessStage] = []

    def add_stage(self, stage: ProcessStage) -> None:
        """Append a stage to the pipeline."""
        self._stages.append(stage)

    def remove_stage(self, name: str) -> bool:
        """Remove a stage by name. Returns True if found."""
        for i, stage in enumerate(self._stages):
            if stage.name == name:
                self._stages.pop(i)
                return True
        return False

    @property
    def stages(self) -> list[ProcessStage]:
        """Return a copy of the current stage list."""
        return list(self._stages)

    def run(self, data: Any, context: ProcessContext) -> ProcessResult:
        """Run all stages sequentially, passing data through each."""
        all_warnings: list[str] = []
        all_errors: list[str] = []
        current_data = data

        for stage in self._stages:
            try:
                result = stage.process(current_data, context)
            except Exception as exc:  # noqa: BLE001
                return ProcessResult(
                    success=False,
                    data=current_data,
                    warnings=all_warnings,
                    errors=[*all_errors, f"Stage '{stage.name}' raised: {exc}"],
                )
            all_warnings.extend(result.warnings)
            all_errors.extend(result.errors)
            if not result.success:
                return ProcessResult(
                    success=False,
                    data=result.data,
                    warnings=all_warnings,
                    errors=all_errors,
                )
            current_data = result.data

        return ProcessResult(
            success=True,
            data=current_data,
            warnings=all_warnings,
            errors=all_errors,
        )
