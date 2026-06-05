"""
Quality tier manager (T-CC-0.3).

Manages quality tier selection, per-subsystem overrides, and dynamic adjustment.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from trinity.types import QualityTier

from .capability_scorer import AdapterInfo, CapabilityScorer


@dataclass
class QualityOverride:
    """Override settings for a specific subsystem."""

    tier: Optional[QualityTier] = None
    locked: bool = False  # Prevent auto-adjustment


@dataclass
class QualityManagerConfig:
    """Configuration for quality manager."""

    default_tier: QualityTier = QualityTier.HIGH
    auto_adjust: bool = True
    frame_budget_ms: float = 16.67  # 60 FPS target
    budget_violation_threshold: int = 10  # Consecutive frames before downgrade
    budget_recovery_threshold: int = 60  # Consecutive frames before upgrade


class QualityManager:
    """
    Manages quality tier selection and dynamic adjustment.

    Responsibilities:
    - Select initial tier based on hardware capabilities
    - Allow per-subsystem tier overrides
    - Monitor frame budget and adjust tiers dynamically
    - Notify listeners of tier changes
    """

    def __init__(
        self,
        adapter_info: Optional[AdapterInfo] = None,
        config: Optional[QualityManagerConfig] = None,
    ):
        self._config = config or QualityManagerConfig()
        self._adapter_info = adapter_info
        self._scorer = CapabilityScorer(adapter_info) if adapter_info else None

        # Current tier (computed or overridden)
        self._current_tier = self._compute_initial_tier()
        self._base_tier = self._current_tier  # Before any dynamic adjustment

        # Per-subsystem overrides
        self._overrides: dict[str, QualityOverride] = {}

        # Budget monitoring
        self._violation_count = 0
        self._recovery_count = 0

        # Change listeners
        self._listeners: list[Callable[[QualityTier, QualityTier], None]] = []

    def _compute_initial_tier(self) -> QualityTier:
        """Compute initial tier from hardware capabilities."""
        if self._scorer:
            score = self._scorer.score()
            return QualityTier.from_score(score)
        return self._config.default_tier

    @property
    def current_tier(self) -> QualityTier:
        """Get the current global quality tier."""
        return self._current_tier

    @property
    def base_tier(self) -> QualityTier:
        """Get the base tier before dynamic adjustment."""
        return self._base_tier

    def get_tier(self, subsystem: str) -> QualityTier:
        """Get effective tier for a subsystem (may be overridden)."""
        if subsystem in self._overrides:
            override = self._overrides[subsystem]
            if override.tier is not None:
                return override.tier
        return self._current_tier

    def set_override(
        self, subsystem: str, tier: Optional[QualityTier], locked: bool = False
    ) -> None:
        """Set tier override for a subsystem."""
        self._overrides[subsystem] = QualityOverride(tier=tier, locked=locked)

    def clear_override(self, subsystem: str) -> None:
        """Remove tier override for a subsystem."""
        self._overrides.pop(subsystem, None)

    def is_locked(self, subsystem: str) -> bool:
        """Check if subsystem tier is locked from auto-adjustment."""
        if subsystem in self._overrides:
            return self._overrides[subsystem].locked
        return False

    def set_tier(self, tier: QualityTier) -> None:
        """Manually set the global tier."""
        old_tier = self._current_tier
        self._current_tier = tier
        self._base_tier = tier
        self._violation_count = 0
        self._recovery_count = 0
        if old_tier != tier:
            self._notify_listeners(old_tier, tier)

    def record_frame_time(self, frame_time_ms: float) -> None:
        """
        Record frame time for dynamic tier adjustment.

        Call this each frame with the measured frame time.
        """
        if not self._config.auto_adjust:
            return

        budget = self._config.frame_budget_ms

        if frame_time_ms > budget * 1.2:  # 20% over budget
            self._violation_count += 1
            self._recovery_count = 0

            if self._violation_count >= self._config.budget_violation_threshold:
                self._try_downgrade()
        elif frame_time_ms < budget * 0.8:  # 20% under budget
            self._recovery_count += 1
            self._violation_count = 0

            if self._recovery_count >= self._config.budget_recovery_threshold:
                self._try_upgrade()
        else:
            # Within acceptable range
            self._violation_count = max(0, self._violation_count - 1)
            self._recovery_count = max(0, self._recovery_count - 1)

    def _try_downgrade(self) -> None:
        """Attempt to downgrade to a lower tier."""
        if self._current_tier == QualityTier.LOW:
            return  # Already at lowest

        old_tier = self._current_tier
        new_tier = QualityTier(self._current_tier.value - 1)
        self._current_tier = new_tier
        self._violation_count = 0
        self._notify_listeners(old_tier, new_tier)

    def _try_upgrade(self) -> None:
        """Attempt to upgrade to a higher tier."""
        if self._current_tier >= self._base_tier:
            return  # Don't upgrade beyond base tier

        old_tier = self._current_tier
        new_tier = QualityTier(self._current_tier.value + 1)
        self._current_tier = new_tier
        self._recovery_count = 0
        self._notify_listeners(old_tier, new_tier)

    def add_listener(
        self, listener: Callable[[QualityTier, QualityTier], None]
    ) -> None:
        """Add listener for tier changes. Called with (old_tier, new_tier)."""
        self._listeners.append(listener)

    def remove_listener(
        self, listener: Callable[[QualityTier, QualityTier], None]
    ) -> None:
        """Remove tier change listener."""
        self._listeners.remove(listener)

    def _notify_listeners(
        self, old_tier: QualityTier, new_tier: QualityTier
    ) -> None:
        """Notify all listeners of tier change."""
        for listener in self._listeners:
            listener(old_tier, new_tier)

    @property
    def adapter_info(self) -> Optional[AdapterInfo]:
        """Get adapter info used for scoring."""
        return self._adapter_info

    @property
    def capability_score(self) -> float:
        """Get the computed capability score (0.0 to 1.0)."""
        if self._scorer:
            return self._scorer.score()
        return 0.5

    def explain_score(self) -> dict[str, float]:
        """Get detailed breakdown of capability score."""
        if self._scorer:
            return self._scorer.explain()
        return {"total": 0.5}
