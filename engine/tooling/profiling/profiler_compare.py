"""
Profiler Comparison for the AI Game Engine.

Provides comparison and diff analysis between profiling sessions:
- Session comparison
- Performance regression detection
- Diff reports
- Trend analysis
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


class ComparisonResult(Enum):
    """Result of a comparison."""
    IMPROVED = auto()
    REGRESSED = auto()
    UNCHANGED = auto()
    NEW = auto()
    REMOVED = auto()


class RegressionSeverity(Enum):
    """Severity of a regression."""
    NONE = auto()
    MINOR = auto()
    MODERATE = auto()
    SEVERE = auto()
    CRITICAL = auto()


@dataclass
class MetricComparison:
    """Comparison of a single metric between two sessions."""
    name: str
    baseline_value: float
    current_value: float
    delta: float
    delta_percentage: float
    result: ComparisonResult
    severity: RegressionSeverity = RegressionSeverity.NONE

    @classmethod
    def from_values(
        cls,
        name: str,
        baseline: Optional[float],
        current: Optional[float],
        regression_thresholds: Optional[Dict[RegressionSeverity, float]] = None,
        higher_is_worse: bool = True,
    ) -> "MetricComparison":
        """Create a comparison from two values."""
        if baseline is None and current is not None:
            return cls(
                name=name,
                baseline_value=0.0,
                current_value=current,
                delta=current,
                delta_percentage=100.0,
                result=ComparisonResult.NEW,
            )

        if baseline is not None and current is None:
            return cls(
                name=name,
                baseline_value=baseline,
                current_value=0.0,
                delta=-baseline,
                delta_percentage=-100.0,
                result=ComparisonResult.REMOVED,
            )

        if baseline is None or current is None:
            return cls(
                name=name,
                baseline_value=0.0,
                current_value=0.0,
                delta=0.0,
                delta_percentage=0.0,
                result=ComparisonResult.UNCHANGED,
            )

        delta = current - baseline
        delta_percentage = (delta / baseline * 100.0) if baseline != 0 else 0.0

        # Determine result
        threshold = 1.0  # 1% threshold for "unchanged"
        if abs(delta_percentage) <= threshold:
            result = ComparisonResult.UNCHANGED
        elif (delta > 0 and higher_is_worse) or (delta < 0 and not higher_is_worse):
            result = ComparisonResult.REGRESSED
        else:
            result = ComparisonResult.IMPROVED

        # Determine severity
        severity = RegressionSeverity.NONE
        if result == ComparisonResult.REGRESSED and regression_thresholds:
            abs_pct = abs(delta_percentage)
            if abs_pct >= regression_thresholds.get(RegressionSeverity.CRITICAL, 100.0):
                severity = RegressionSeverity.CRITICAL
            elif abs_pct >= regression_thresholds.get(RegressionSeverity.SEVERE, 50.0):
                severity = RegressionSeverity.SEVERE
            elif abs_pct >= regression_thresholds.get(RegressionSeverity.MODERATE, 20.0):
                severity = RegressionSeverity.MODERATE
            elif abs_pct >= regression_thresholds.get(RegressionSeverity.MINOR, 5.0):
                severity = RegressionSeverity.MINOR

        return cls(
            name=name,
            baseline_value=baseline,
            current_value=current,
            delta=delta,
            delta_percentage=delta_percentage,
            result=result,
            severity=severity,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "delta": self.delta,
            "delta_percentage": self.delta_percentage,
            "result": self.result.name,
            "severity": self.severity.name,
        }


@dataclass
class SessionDiff:
    """Difference between two profiling sessions."""
    baseline_id: str
    current_id: str
    timestamp: float = field(default_factory=time.time)
    cpu_comparisons: List[MetricComparison] = field(default_factory=list)
    gpu_comparisons: List[MetricComparison] = field(default_factory=list)
    memory_comparisons: List[MetricComparison] = field(default_factory=list)
    frame_comparisons: List[MetricComparison] = field(default_factory=list)
    overall_result: ComparisonResult = ComparisonResult.UNCHANGED
    regression_count: int = 0
    improvement_count: int = 0

    def add_cpu_comparison(self, comparison: MetricComparison) -> None:
        """Add a CPU metric comparison."""
        self.cpu_comparisons.append(comparison)
        self._update_counts(comparison)

    def add_gpu_comparison(self, comparison: MetricComparison) -> None:
        """Add a GPU metric comparison."""
        self.gpu_comparisons.append(comparison)
        self._update_counts(comparison)

    def add_memory_comparison(self, comparison: MetricComparison) -> None:
        """Add a memory metric comparison."""
        self.memory_comparisons.append(comparison)
        self._update_counts(comparison)

    def add_frame_comparison(self, comparison: MetricComparison) -> None:
        """Add a frame metric comparison."""
        self.frame_comparisons.append(comparison)
        self._update_counts(comparison)

    def _update_counts(self, comparison: MetricComparison) -> None:
        """Update regression/improvement counts."""
        if comparison.result == ComparisonResult.REGRESSED:
            self.regression_count += 1
        elif comparison.result == ComparisonResult.IMPROVED:
            self.improvement_count += 1

        # Update overall result
        if self.regression_count > 0:
            self.overall_result = ComparisonResult.REGRESSED
        elif self.improvement_count > 0:
            self.overall_result = ComparisonResult.IMPROVED

    def get_regressions(
        self,
        min_severity: RegressionSeverity = RegressionSeverity.MINOR,
    ) -> List[MetricComparison]:
        """Get all regressions above a severity threshold."""
        all_comparisons = (
            self.cpu_comparisons
            + self.gpu_comparisons
            + self.memory_comparisons
            + self.frame_comparisons
        )
        return [
            c for c in all_comparisons
            if c.result == ComparisonResult.REGRESSED
            and c.severity.value >= min_severity.value
        ]

    def get_improvements(self) -> List[MetricComparison]:
        """Get all improvements."""
        all_comparisons = (
            self.cpu_comparisons
            + self.gpu_comparisons
            + self.memory_comparisons
            + self.frame_comparisons
        )
        return [c for c in all_comparisons if c.result == ComparisonResult.IMPROVED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_id": self.baseline_id,
            "current_id": self.current_id,
            "timestamp": self.timestamp,
            "overall_result": self.overall_result.name,
            "regression_count": self.regression_count,
            "improvement_count": self.improvement_count,
            "cpu_comparisons": [c.to_dict() for c in self.cpu_comparisons],
            "gpu_comparisons": [c.to_dict() for c in self.gpu_comparisons],
            "memory_comparisons": [c.to_dict() for c in self.memory_comparisons],
            "frame_comparisons": [c.to_dict() for c in self.frame_comparisons],
        }


@dataclass
class ComparisonReport:
    """Detailed comparison report."""
    title: str
    generated_at: float = field(default_factory=time.time)
    session_diff: Optional[SessionDiff] = None
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    charts: Dict[str, Any] = field(default_factory=dict)

    def generate_summary(self) -> str:
        """Generate a text summary of the comparison."""
        if not self.session_diff:
            return "No comparison data available."

        diff = self.session_diff
        lines = [
            f"Performance Comparison Report",
            f"=============================",
            f"",
            f"Baseline: {diff.baseline_id}",
            f"Current:  {diff.current_id}",
            f"",
            f"Overall Result: {diff.overall_result.name}",
            f"",
            f"Statistics:",
            f"  - Regressions:  {diff.regression_count}",
            f"  - Improvements: {diff.improvement_count}",
            f"",
        ]

        # Add regressions
        regressions = diff.get_regressions()
        if regressions:
            lines.append("Regressions:")
            for reg in sorted(regressions, key=lambda x: -abs(x.delta_percentage)):
                lines.append(
                    f"  [{reg.severity.name}] {reg.name}: "
                    f"{reg.baseline_value:.2f} -> {reg.current_value:.2f} "
                    f"({reg.delta_percentage:+.1f}%)"
                )
            lines.append("")

        # Add improvements
        improvements = diff.get_improvements()
        if improvements:
            lines.append("Improvements:")
            for imp in sorted(improvements, key=lambda x: x.delta_percentage):
                lines.append(
                    f"  {imp.name}: "
                    f"{imp.baseline_value:.2f} -> {imp.current_value:.2f} "
                    f"({imp.delta_percentage:+.1f}%)"
                )
            lines.append("")

        self.summary = "\n".join(lines)
        return self.summary

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation to the report."""
        self.recommendations.append(recommendation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "session_diff": self.session_diff.to_dict() if self.session_diff else None,
            "summary": self.summary,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        """Generate a Markdown report."""
        lines = [
            f"# {self.title}",
            f"",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.generated_at))}",
            f"",
        ]

        if self.session_diff:
            diff = self.session_diff
            lines.extend([
                f"## Summary",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Baseline | `{diff.baseline_id}` |",
                f"| Current | `{diff.current_id}` |",
                f"| Overall Result | **{diff.overall_result.name}** |",
                f"| Regressions | {diff.regression_count} |",
                f"| Improvements | {diff.improvement_count} |",
                f"",
            ])

            # Regressions
            regressions = diff.get_regressions()
            if regressions:
                lines.extend([
                    f"## Regressions",
                    f"",
                    f"| Metric | Baseline | Current | Delta | Severity |",
                    f"|--------|----------|---------|-------|----------|",
                ])
                for reg in sorted(regressions, key=lambda x: -abs(x.delta_percentage)):
                    lines.append(
                        f"| {reg.name} | {reg.baseline_value:.2f} | "
                        f"{reg.current_value:.2f} | {reg.delta_percentage:+.1f}% | "
                        f"{reg.severity.name} |"
                    )
                lines.append("")

            # Improvements
            improvements = diff.get_improvements()
            if improvements:
                lines.extend([
                    f"## Improvements",
                    f"",
                    f"| Metric | Baseline | Current | Delta |",
                    f"|--------|----------|---------|-------|",
                ])
                for imp in sorted(improvements, key=lambda x: x.delta_percentage):
                    lines.append(
                        f"| {imp.name} | {imp.baseline_value:.2f} | "
                        f"{imp.current_value:.2f} | {imp.delta_percentage:+.1f}% |"
                    )
                lines.append("")

        # Recommendations
        if self.recommendations:
            lines.extend([
                f"## Recommendations",
                f"",
            ])
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class RegressionThresholds:
    """Thresholds for regression detection."""
    minor: float = 5.0  # 5% regression
    moderate: float = 20.0  # 20% regression
    severe: float = 50.0  # 50% regression
    critical: float = 100.0  # 100% regression (2x worse)

    def to_dict(self) -> Dict[RegressionSeverity, float]:
        """Convert to severity dictionary."""
        return {
            RegressionSeverity.MINOR: self.minor,
            RegressionSeverity.MODERATE: self.moderate,
            RegressionSeverity.SEVERE: self.severe,
            RegressionSeverity.CRITICAL: self.critical,
        }


class RegressionDetector:
    """Detects performance regressions between sessions."""

    def __init__(
        self,
        thresholds: Optional[RegressionThresholds] = None,
    ) -> None:
        """
        Initialize the regression detector.

        Args:
            thresholds: Regression severity thresholds
        """
        self.thresholds = thresholds or RegressionThresholds()

    def detect(
        self,
        baseline: Dict[str, float],
        current: Dict[str, float],
        higher_is_worse: bool = True,
    ) -> List[MetricComparison]:
        """
        Detect regressions between two metric sets.

        Args:
            baseline: Baseline metrics
            current: Current metrics
            higher_is_worse: Whether higher values indicate worse performance

        Returns:
            List of comparisons with detected regressions
        """
        all_keys = set(baseline.keys()) | set(current.keys())
        comparisons = []

        for key in all_keys:
            comparison = MetricComparison.from_values(
                name=key,
                baseline=baseline.get(key),
                current=current.get(key),
                regression_thresholds=self.thresholds.to_dict(),
                higher_is_worse=higher_is_worse,
            )
            comparisons.append(comparison)

        return comparisons

    def has_severe_regression(
        self,
        comparisons: List[MetricComparison],
        min_severity: RegressionSeverity = RegressionSeverity.SEVERE,
    ) -> bool:
        """Check if there are any severe regressions."""
        return any(
            c.result == ComparisonResult.REGRESSED
            and c.severity.value >= min_severity.value
            for c in comparisons
        )


class ProfilerComparator:
    """
    Compares profiling sessions and generates diff reports.

    Usage:
        comparator = ProfilerComparator()
        diff = comparator.compare(baseline_data, current_data)
        report = comparator.generate_report(diff, "Performance Report")
    """

    def __init__(
        self,
        thresholds: Optional[RegressionThresholds] = None,
    ) -> None:
        """
        Initialize the comparator.

        Args:
            thresholds: Regression severity thresholds
        """
        self.thresholds = thresholds or RegressionThresholds()
        self.detector = RegressionDetector(thresholds)

    def compare(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
        baseline_id: str = "baseline",
        current_id: str = "current",
    ) -> SessionDiff:
        """
        Compare two profiling sessions.

        Args:
            baseline: Baseline session data
            current: Current session data
            baseline_id: Identifier for baseline
            current_id: Identifier for current

        Returns:
            Session diff object
        """
        diff = SessionDiff(
            baseline_id=baseline_id,
            current_id=current_id,
        )

        # Compare CPU metrics
        if "cpu" in baseline and "cpu" in current:
            cpu_diff = self._compare_cpu(
                baseline.get("cpu", {}),
                current.get("cpu", {}),
            )
            for comparison in cpu_diff:
                diff.add_cpu_comparison(comparison)

        # Compare GPU metrics
        if "gpu" in baseline and "gpu" in current:
            gpu_diff = self._compare_gpu(
                baseline.get("gpu", {}),
                current.get("gpu", {}),
            )
            for comparison in gpu_diff:
                diff.add_gpu_comparison(comparison)

        # Compare memory metrics
        if "memory" in baseline and "memory" in current:
            memory_diff = self._compare_memory(
                baseline.get("memory", {}),
                current.get("memory", {}),
            )
            for comparison in memory_diff:
                diff.add_memory_comparison(comparison)

        # Compare frame metrics
        if "frames" in baseline and "frames" in current:
            frame_diff = self._compare_frames(
                baseline.get("frames", {}),
                current.get("frames", {}),
            )
            for comparison in frame_diff:
                diff.add_frame_comparison(comparison)

        return diff

    def _compare_cpu(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> List[MetricComparison]:
        """Compare CPU profiler data."""
        comparisons = []

        # Compare stats
        baseline_stats = baseline.get("stats", {})
        current_stats = current.get("stats", {})

        for name in set(baseline_stats.keys()) | set(current_stats.keys()):
            b_stat = baseline_stats.get(name, {})
            c_stat = current_stats.get(name, {})

            # Compare average time
            comparisons.append(
                MetricComparison.from_values(
                    name=f"cpu.{name}.avg_ms",
                    baseline=b_stat.get("avg_time_ms"),
                    current=c_stat.get("avg_time_ms"),
                    regression_thresholds=self.thresholds.to_dict(),
                    higher_is_worse=True,
                )
            )

        return comparisons

    def _compare_gpu(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> List[MetricComparison]:
        """Compare GPU profiler data."""
        comparisons = []

        # Compare pass timings
        baseline_passes = baseline.get("pass_timings", {})
        current_passes = current.get("pass_timings", {})

        for name in set(baseline_passes.keys()) | set(current_passes.keys()):
            b_pass = baseline_passes.get(name, {})
            c_pass = current_passes.get(name, {})

            comparisons.append(
                MetricComparison.from_values(
                    name=f"gpu.{name}.time_ms",
                    baseline=b_pass.get("gpu_time_ms"),
                    current=c_pass.get("gpu_time_ms"),
                    regression_thresholds=self.thresholds.to_dict(),
                    higher_is_worse=True,
                )
            )

        # Compare draw stats
        b_draws = baseline.get("draw_stats", {})
        c_draws = current.get("draw_stats", {})

        comparisons.append(
            MetricComparison.from_values(
                name="gpu.draw_calls",
                baseline=b_draws.get("total_draw_calls"),
                current=c_draws.get("total_draw_calls"),
                regression_thresholds=self.thresholds.to_dict(),
                higher_is_worse=True,
            )
        )

        return comparisons

    def _compare_memory(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> List[MetricComparison]:
        """Compare memory profiler data."""
        comparisons = []

        comparisons.append(
            MetricComparison.from_values(
                name="memory.current_mb",
                baseline=baseline.get("current_usage_mb"),
                current=current.get("current_usage_mb"),
                regression_thresholds=self.thresholds.to_dict(),
                higher_is_worse=True,
            )
        )

        comparisons.append(
            MetricComparison.from_values(
                name="memory.peak_mb",
                baseline=baseline.get("peak_usage_mb"),
                current=current.get("peak_usage_mb"),
                regression_thresholds=self.thresholds.to_dict(),
                higher_is_worse=True,
            )
        )

        return comparisons

    def _compare_frames(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> List[MetricComparison]:
        """Compare frame profiler data."""
        comparisons = []

        b_stats = baseline.get("stats", {})
        c_stats = current.get("stats", {})

        comparisons.append(
            MetricComparison.from_values(
                name="frame.avg_time_ms",
                baseline=b_stats.get("avg_frame_time_ms"),
                current=c_stats.get("avg_frame_time_ms"),
                regression_thresholds=self.thresholds.to_dict(),
                higher_is_worse=True,
            )
        )

        comparisons.append(
            MetricComparison.from_values(
                name="frame.spike_count",
                baseline=b_stats.get("spike_count"),
                current=c_stats.get("spike_count"),
                regression_thresholds=self.thresholds.to_dict(),
                higher_is_worse=True,
            )
        )

        # FPS is better when higher
        comparisons.append(
            MetricComparison.from_values(
                name="frame.avg_fps",
                baseline=b_stats.get("current_fps"),
                current=c_stats.get("current_fps"),
                regression_thresholds=self.thresholds.to_dict(),
                higher_is_worse=False,
            )
        )

        return comparisons

    def generate_report(
        self,
        diff: SessionDiff,
        title: str = "Performance Comparison Report",
    ) -> ComparisonReport:
        """
        Generate a comparison report from a session diff.

        Args:
            diff: Session diff object
            title: Report title

        Returns:
            Comparison report
        """
        report = ComparisonReport(
            title=title,
            session_diff=diff,
        )

        # Generate summary
        report.generate_summary()

        # Generate recommendations
        regressions = diff.get_regressions(RegressionSeverity.MODERATE)

        for reg in regressions:
            if "cpu" in reg.name:
                report.add_recommendation(
                    f"Optimize {reg.name}: Consider profiling to identify hotspots"
                )
            elif "gpu" in reg.name:
                report.add_recommendation(
                    f"Investigate GPU regression in {reg.name}: "
                    "Check shader complexity and draw calls"
                )
            elif "memory" in reg.name:
                report.add_recommendation(
                    f"Memory usage increased ({reg.delta_percentage:+.1f}%): "
                    "Check for memory leaks or increased allocations"
                )
            elif "frame" in reg.name:
                report.add_recommendation(
                    f"Frame time regression: Run detailed frame analysis"
                )

        return report

    def compare_files(
        self,
        baseline_path: Union[str, Path],
        current_path: Union[str, Path],
    ) -> SessionDiff:
        """
        Compare two profiling session files.

        Args:
            baseline_path: Path to baseline session file
            current_path: Path to current session file

        Returns:
            Session diff
        """
        baseline_path = Path(baseline_path)
        current_path = Path(current_path)

        with open(baseline_path, "r") as f:
            baseline = json.load(f)

        with open(current_path, "r") as f:
            current = json.load(f)

        return self.compare(
            baseline=baseline,
            current=current,
            baseline_id=baseline_path.name,
            current_id=current_path.name,
        )
