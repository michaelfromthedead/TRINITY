"""
Crash analytics and grouping.

Provides analytics capabilities for crash reports including
grouping similar crashes, pattern detection, and trend analysis.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .crash_reporter import CrashReport, CrashSeverity


@dataclass
class CrashGroup:
    """
    A group of similar crashes.

    Crashes are grouped by their fingerprint to identify
    recurring issues.
    """

    id: str
    fingerprint: str
    exception_type: str
    exception_message_pattern: str
    first_seen: float
    last_seen: float
    count: int = 0
    crash_ids: List[str] = field(default_factory=list)
    affected_versions: Set[str] = field(default_factory=set)
    affected_platforms: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    status: str = "open"  # open, investigating, resolved, ignored
    assignee: Optional[str] = None
    notes: str = ""

    def add_crash(self, report: CrashReport) -> None:
        """Add a crash to this group."""
        self.count += 1
        self.crash_ids.append(report.id)
        self.last_seen = max(self.last_seen, report.timestamp)

        if report.context.build_version:
            self.affected_versions.add(report.context.build_version)

        if report.system_info.os_name:
            self.affected_platforms.add(report.system_info.os_name)

        self.tags.update(report.context.tags)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "exception_type": self.exception_type,
            "exception_message_pattern": self.exception_message_pattern,
            "first_seen": self.first_seen,
            "first_seen_formatted": datetime.fromtimestamp(self.first_seen).isoformat(),
            "last_seen": self.last_seen,
            "last_seen_formatted": datetime.fromtimestamp(self.last_seen).isoformat(),
            "count": self.count,
            "affected_versions": list(self.affected_versions),
            "affected_platforms": list(self.affected_platforms),
            "tags": list(self.tags),
            "status": self.status,
            "assignee": self.assignee,
            "notes": self.notes,
        }


@dataclass
class CrashPattern:
    """
    A pattern identified in crash data.

    Patterns can indicate common causes or conditions
    that lead to crashes.
    """

    id: str
    name: str
    description: str
    pattern_type: str  # stack_trace, exception, context, temporal
    criteria: Dict[str, Any] = field(default_factory=dict)
    matched_crash_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    severity: str = "medium"  # low, medium, high, critical

    def matches(self, report: CrashReport) -> bool:
        """Check if a crash matches this pattern."""
        # Override in subclasses for specific pattern types
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pattern_type": self.pattern_type,
            "criteria": self.criteria,
            "match_count": len(self.matched_crash_ids),
            "confidence": self.confidence,
            "severity": self.severity,
        }


@dataclass
class CrashTrend:
    """
    Trend information for crash data.

    Tracks how crash rates change over time.
    """

    period: str  # hourly, daily, weekly
    start_time: float
    end_time: float
    total_crashes: int = 0
    unique_issues: int = 0
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_version: Dict[str, int] = field(default_factory=dict)
    by_platform: Dict[str, int] = field(default_factory=dict)
    top_groups: List[str] = field(default_factory=list)
    trend_direction: str = "stable"  # increasing, decreasing, stable
    change_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period": self.period,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_crashes": self.total_crashes,
            "unique_issues": self.unique_issues,
            "by_severity": self.by_severity,
            "by_version": self.by_version,
            "by_platform": self.by_platform,
            "top_groups": self.top_groups,
            "trend_direction": self.trend_direction,
            "change_percent": self.change_percent,
        }


class CrashAnalytics:
    """
    Analytics engine for crash data.

    Features:
    - Crash grouping by fingerprint
    - Pattern detection
    - Trend analysis
    - Statistical reporting
    """

    def __init__(self):
        self._crashes: Dict[str, CrashReport] = {}
        self._groups: Dict[str, CrashGroup] = {}
        self._patterns: Dict[str, CrashPattern] = {}
        self._fingerprint_to_group: Dict[str, str] = {}

    def add_crash(self, report: CrashReport) -> CrashGroup:
        """
        Add a crash report and return its group.

        Args:
            report: Crash report to add

        Returns:
            The crash group this report belongs to
        """
        self._crashes[report.id] = report

        # Find or create group
        fingerprint = report.fingerprint
        group_id = self._fingerprint_to_group.get(fingerprint)

        if group_id and group_id in self._groups:
            group = self._groups[group_id]
        else:
            # Create new group
            group = CrashGroup(
                id=f"group_{fingerprint[:8]}",
                fingerprint=fingerprint,
                exception_type=report.exception_info.exception_type,
                exception_message_pattern=self._extract_message_pattern(
                    report.exception_info.exception_message
                ),
                first_seen=report.timestamp,
                last_seen=report.timestamp,
            )
            self._groups[group.id] = group
            self._fingerprint_to_group[fingerprint] = group.id

        group.add_crash(report)

        # Check patterns
        for pattern in self._patterns.values():
            if pattern.matches(report):
                pattern.matched_crash_ids.append(report.id)

        return group

    def _extract_message_pattern(self, message: str) -> str:
        """Extract a generalized pattern from an error message."""
        # Replace specific values with placeholders
        pattern = message

        # Replace hex addresses
        pattern = re.sub(r"0x[0-9a-fA-F]+", "<addr>", pattern)

        # Replace numbers
        pattern = re.sub(r"\b\d+\b", "<num>", pattern)

        # Replace quoted strings
        pattern = re.sub(r'"[^"]*"', '"<str>"', pattern)
        pattern = re.sub(r"'[^']*'", "'<str>'", pattern)

        # Replace file paths
        pattern = re.sub(r"[A-Za-z]:\\[^\s]+", "<path>", pattern)
        pattern = re.sub(r"/[^\s]+", "<path>", pattern)

        return pattern

    def get_group(self, group_id: str) -> Optional[CrashGroup]:
        """Get a crash group by ID."""
        return self._groups.get(group_id)

    def get_groups(
        self,
        status: Optional[str] = None,
        min_count: int = 0,
        limit: int = 100,
    ) -> List[CrashGroup]:
        """
        Get crash groups with optional filtering.

        Args:
            status: Filter by status
            min_count: Minimum crash count
            limit: Maximum groups to return

        Returns:
            List of crash groups
        """
        groups = list(self._groups.values())

        if status:
            groups = [g for g in groups if g.status == status]

        groups = [g for g in groups if g.count >= min_count]

        # Sort by count descending
        groups.sort(key=lambda g: g.count, reverse=True)

        return groups[:limit]

    def analyze_trends(
        self,
        period: str = "daily",
        days: int = 7,
    ) -> List[CrashTrend]:
        """
        Analyze crash trends over time.

        Args:
            period: Trend period (hourly, daily, weekly)
            days: Number of days to analyze

        Returns:
            List of trend data points
        """
        now = time.time()
        end_time = now
        trends = []

        if period == "hourly":
            interval = 3600
            num_periods = min(days * 24, 168)  # Max 1 week of hours
        elif period == "daily":
            interval = 86400
            num_periods = days
        else:  # weekly
            interval = 86400 * 7
            num_periods = max(1, days // 7)

        for i in range(num_periods):
            period_end = end_time - (i * interval)
            period_start = period_end - interval

            trend = self._compute_trend_for_period(
                period, period_start, period_end
            )
            trends.append(trend)

        # Calculate trend direction for each period
        for i in range(len(trends) - 1):
            current = trends[i].total_crashes
            previous = trends[i + 1].total_crashes

            if previous > 0:
                change = (current - previous) / previous * 100
                trends[i].change_percent = change

                if change > 10:
                    trends[i].trend_direction = "increasing"
                elif change < -10:
                    trends[i].trend_direction = "decreasing"
                else:
                    trends[i].trend_direction = "stable"

        trends.reverse()  # Chronological order
        return trends

    def _compute_trend_for_period(
        self,
        period: str,
        start_time: float,
        end_time: float,
    ) -> CrashTrend:
        """Compute trend data for a specific period."""
        crashes_in_period = [
            c for c in self._crashes.values()
            if start_time <= c.timestamp < end_time
        ]

        trend = CrashTrend(
            period=period,
            start_time=start_time,
            end_time=end_time,
            total_crashes=len(crashes_in_period),
        )

        unique_fingerprints = set()
        for crash in crashes_in_period:
            unique_fingerprints.add(crash.fingerprint)

            # By severity
            severity = crash.severity.name
            trend.by_severity[severity] = trend.by_severity.get(severity, 0) + 1

            # By version
            version = crash.context.build_version or "unknown"
            trend.by_version[version] = trend.by_version.get(version, 0) + 1

            # By platform
            platform = crash.system_info.os_name or "unknown"
            trend.by_platform[platform] = trend.by_platform.get(platform, 0) + 1

        trend.unique_issues = len(unique_fingerprints)

        # Top groups
        group_counts: Dict[str, int] = defaultdict(int)
        for crash in crashes_in_period:
            group_id = self._fingerprint_to_group.get(crash.fingerprint)
            if group_id:
                group_counts[group_id] += 1

        top_groups = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)
        trend.top_groups = [g[0] for g in top_groups[:5]]

        return trend

    def detect_patterns(self) -> List[CrashPattern]:
        """
        Detect patterns in crash data.

        Returns:
            List of detected patterns
        """
        patterns = []

        # Detect repeated stack patterns
        stack_patterns = self._detect_stack_patterns()
        patterns.extend(stack_patterns)

        # Detect version-specific patterns
        version_patterns = self._detect_version_patterns()
        patterns.extend(version_patterns)

        # Detect time-based patterns
        time_patterns = self._detect_time_patterns()
        patterns.extend(time_patterns)

        return patterns

    def _detect_stack_patterns(self) -> List[CrashPattern]:
        """Detect common stack trace patterns."""
        patterns = []

        # Count function occurrences in crashes
        function_counts: Dict[str, List[str]] = defaultdict(list)

        for crash in self._crashes.values():
            for frame in crash.exception_info.stack_trace:
                key = f"{frame.filename}:{frame.function_name}"
                function_counts[key].append(crash.id)

        # Find functions that appear in many crashes
        for key, crash_ids in function_counts.items():
            if len(crash_ids) >= 3:  # Threshold
                pattern = CrashPattern(
                    id=f"stack_{hashlib.md5(key.encode()).hexdigest()[:8]}",
                    name=f"Crashes involving {key.split(':')[1]}",
                    description=f"Multiple crashes have {key} in stack trace",
                    pattern_type="stack_trace",
                    criteria={"function": key},
                    matched_crash_ids=crash_ids,
                    confidence=min(1.0, len(crash_ids) / 10),
                )
                patterns.append(pattern)

        return patterns

    def _detect_version_patterns(self) -> List[CrashPattern]:
        """Detect version-specific crash patterns."""
        patterns = []

        # Count crashes by version
        version_counts: Dict[str, List[str]] = defaultdict(list)

        for crash in self._crashes.values():
            version = crash.context.build_version or "unknown"
            version_counts[version].append(crash.id)

        # Find versions with disproportionate crashes
        total = len(self._crashes)
        if total < 10:
            return patterns

        for version, crash_ids in version_counts.items():
            ratio = len(crash_ids) / total
            if ratio > 0.3 and len(crash_ids) >= 5:  # 30%+ of crashes
                pattern = CrashPattern(
                    id=f"version_{hashlib.md5(version.encode()).hexdigest()[:8]}",
                    name=f"High crash rate in version {version}",
                    description=f"{ratio*100:.1f}% of crashes occur in version {version}",
                    pattern_type="context",
                    criteria={"version": version, "ratio": ratio},
                    matched_crash_ids=crash_ids,
                    confidence=ratio,
                    severity="high" if ratio > 0.5 else "medium",
                )
                patterns.append(pattern)

        return patterns

    def _detect_time_patterns(self) -> List[CrashPattern]:
        """Detect time-based crash patterns."""
        patterns = []

        # Group crashes by hour of day
        hourly_counts: Dict[int, List[str]] = defaultdict(list)

        for crash in self._crashes.values():
            hour = datetime.fromtimestamp(crash.timestamp).hour
            hourly_counts[hour].append(crash.id)

        # Find peak hours
        total = len(self._crashes)
        if total < 24:
            return patterns

        avg_per_hour = total / 24

        for hour, crash_ids in hourly_counts.items():
            if len(crash_ids) > avg_per_hour * 2:  # 2x average
                pattern = CrashPattern(
                    id=f"time_hour_{hour}",
                    name=f"Crash spike at {hour:02d}:00",
                    description=f"Disproportionate crashes at hour {hour}",
                    pattern_type="temporal",
                    criteria={"hour": hour, "count": len(crash_ids)},
                    matched_crash_ids=crash_ids,
                    confidence=len(crash_ids) / total,
                )
                patterns.append(pattern)

        return patterns

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall crash statistics.

        Returns:
            Statistics dictionary
        """
        if not self._crashes:
            return {
                "total_crashes": 0,
                "unique_issues": 0,
                "open_issues": 0,
            }

        crashes = list(self._crashes.values())
        timestamps = [c.timestamp for c in crashes]

        by_severity = defaultdict(int)
        by_platform = defaultdict(int)
        by_version = defaultdict(int)

        for crash in crashes:
            by_severity[crash.severity.name] += 1
            by_platform[crash.system_info.os_name or "unknown"] += 1
            by_version[crash.context.build_version or "unknown"] += 1

        open_groups = len([g for g in self._groups.values() if g.status == "open"])

        return {
            "total_crashes": len(crashes),
            "unique_issues": len(self._groups),
            "open_issues": open_groups,
            "first_crash": min(timestamps),
            "last_crash": max(timestamps),
            "by_severity": dict(by_severity),
            "by_platform": dict(by_platform),
            "by_version": dict(by_version),
            "patterns_detected": len(self._patterns),
        }


# Global instance and convenience functions

_analytics: Optional[CrashAnalytics] = None


def get_analytics() -> CrashAnalytics:
    """Get or create the global analytics instance."""
    global _analytics
    if _analytics is None:
        _analytics = CrashAnalytics()
    return _analytics


def analyze_crash(report: CrashReport) -> CrashGroup:
    """
    Analyze a crash and add it to analytics.

    Args:
        report: Crash report

    Returns:
        The crash group
    """
    return get_analytics().add_crash(report)


def group_crashes(
    crashes: List[CrashReport],
) -> Dict[str, CrashGroup]:
    """
    Group multiple crashes.

    Args:
        crashes: List of crash reports

    Returns:
        Dictionary of fingerprint to group
    """
    analytics = get_analytics()
    groups = {}

    for crash in crashes:
        group = analytics.add_crash(crash)
        groups[crash.fingerprint] = group

    return groups


def get_crash_statistics() -> Dict[str, Any]:
    """
    Get crash statistics.

    Returns:
        Statistics dictionary
    """
    return get_analytics().get_statistics()


def detect_patterns() -> List[CrashPattern]:
    """
    Detect patterns in crash data.

    Returns:
        List of patterns
    """
    return get_analytics().detect_patterns()
