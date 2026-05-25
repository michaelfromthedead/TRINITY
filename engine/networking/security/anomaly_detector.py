"""
Anomaly detection system for anti-cheat.

This module analyzes player behavior to detect cheating patterns
such as aimbot, speed hacks, wallhacks, and impossible reactions.

Thread-safety: All public methods are thread-safe.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set
import statistics
import time
import threading

from engine.networking.security.config import (
    ANOMALY_DETECTION,
    VALIDATION_LIMITS,
)


class AnomalyType(Enum):
    """Types of detectable anomalies."""
    AIMBOT = auto()
    SPEED_HACK = auto()
    TELEPORT = auto()
    IMPOSSIBLE_REACTION = auto()
    WALL_HACK_SUSPECT = auto()
    DAMAGE_HACK = auto()
    RAPID_FIRE = auto()
    NO_RECOIL = auto()
    GOD_MODE = auto()
    RESOURCE_HACK = auto()


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AnomalyThresholds:
    """
    Configurable thresholds for anomaly detection.

    All defaults loaded from security config to avoid magic numbers.

    Attributes:
        accuracy_threshold: Accuracy above this is suspicious (0.0-1.0)
        accuracy_sample_size: Minimum shots to analyze accuracy
        headshot_rate_threshold: Headshot rate above this is suspicious
        min_reaction_time_ms: Reactions faster than this are suspicious
        reaction_sample_size: Minimum reactions to analyze
        speed_variance_threshold: High variance in movement speed
        consecutive_kills_threshold: Kills without deaths to flag
        damage_multiplier_threshold: Damage above expected
        recoil_variance_threshold: Low recoil variance is suspicious
        wall_hit_rate_threshold: Rate of hitting invisible targets
    """
    accuracy_threshold: float = ANOMALY_DETECTION.ACCURACY_THRESHOLD
    accuracy_sample_size: int = ANOMALY_DETECTION.ACCURACY_SAMPLE_SIZE
    headshot_rate_threshold: float = ANOMALY_DETECTION.HEADSHOT_RATE_THRESHOLD
    min_reaction_time_ms: float = ANOMALY_DETECTION.MIN_REACTION_TIME_MS
    reaction_sample_size: int = ANOMALY_DETECTION.REACTION_SAMPLE_SIZE
    speed_variance_threshold: float = ANOMALY_DETECTION.SPEED_VARIANCE_THRESHOLD
    consecutive_kills_threshold: int = ANOMALY_DETECTION.CONSECUTIVE_KILLS_THRESHOLD
    damage_multiplier_threshold: float = ANOMALY_DETECTION.DAMAGE_MULTIPLIER_THRESHOLD
    recoil_variance_threshold: float = ANOMALY_DETECTION.RECOIL_VARIANCE_THRESHOLD
    wall_hit_rate_threshold: float = ANOMALY_DETECTION.WALL_HIT_RATE_THRESHOLD


@dataclass
class ShotEvent:
    """Record of a shot fired by a player."""
    timestamp: float
    hit: bool
    headshot: bool
    distance: float
    target_visible: bool
    target_id: Optional[str] = None


@dataclass
class KillEvent:
    """Record of a kill by a player."""
    timestamp: float
    victim_id: str
    weapon: str
    distance: float
    time_to_kill: float
    headshot: bool


@dataclass
class MovementEvent:
    """Record of player movement."""
    timestamp: float
    speed: float
    position_delta: float
    time_delta: float


@dataclass
class ReactionEvent:
    """Record of player reaction time."""
    timestamp: float
    reaction_time_ms: float
    stimulus_type: str


@dataclass
class DamageEvent:
    """Record of damage dealt by a player."""
    timestamp: float
    target_id: str
    damage_dealt: float
    expected_damage: float
    weapon: str


@dataclass
class PlayerStats:
    """
    Statistical data for a player.

    Tracks all relevant events for anomaly analysis.
    """
    player_id: str
    shots: List[ShotEvent] = field(default_factory=list)
    kills: List[KillEvent] = field(default_factory=list)
    deaths: List[float] = field(default_factory=list)  # Timestamps
    movements: List[MovementEvent] = field(default_factory=list)
    reactions: List[ReactionEvent] = field(default_factory=list)
    damages: List[DamageEvent] = field(default_factory=list)

    # Computed statistics (cached)
    _accuracy: Optional[float] = None
    _headshot_rate: Optional[float] = None
    _avg_reaction_time: Optional[float] = None

    # Rolling window for recent stats (from config)
    recent_window_seconds: float = ANOMALY_DETECTION.RECENT_WINDOW_SECONDS

    def add_shot(self, event: ShotEvent) -> None:
        """Add a shot event with bounds checking."""
        if len(self.shots) < VALIDATION_LIMITS.MAX_EVENTS_PER_PLAYER:
            self.shots.append(event)
        else:
            # Remove oldest event to make room (FIFO)
            self.shots.pop(0)
            self.shots.append(event)
        self._accuracy = None  # Invalidate cache

    def add_kill(self, event: KillEvent) -> None:
        """Add a kill event with bounds checking."""
        if len(self.kills) < VALIDATION_LIMITS.MAX_EVENTS_PER_PLAYER:
            self.kills.append(event)
        else:
            self.kills.pop(0)
            self.kills.append(event)
        self._headshot_rate = None

    def add_death(self, timestamp: float) -> None:
        """Add a death event with bounds checking."""
        if len(self.deaths) < VALIDATION_LIMITS.MAX_EVENTS_PER_PLAYER:
            self.deaths.append(timestamp)
        else:
            self.deaths.pop(0)
            self.deaths.append(timestamp)

    def add_movement(self, event: MovementEvent) -> None:
        """Add a movement event with bounds checking."""
        if len(self.movements) < VALIDATION_LIMITS.MAX_EVENTS_PER_PLAYER:
            self.movements.append(event)
        else:
            self.movements.pop(0)
            self.movements.append(event)

    def add_reaction(self, event: ReactionEvent) -> None:
        """Add a reaction event with bounds checking."""
        if len(self.reactions) < VALIDATION_LIMITS.MAX_EVENTS_PER_PLAYER:
            self.reactions.append(event)
        else:
            self.reactions.pop(0)
            self.reactions.append(event)
        self._avg_reaction_time = None

    def add_damage(self, event: DamageEvent) -> None:
        """Add a damage event with bounds checking."""
        if len(self.damages) < VALIDATION_LIMITS.MAX_EVENTS_PER_PLAYER:
            self.damages.append(event)
        else:
            self.damages.pop(0)
            self.damages.append(event)

    def _get_recent_items(self, items: list, timestamp_attr: str = "timestamp") -> list:
        """Filter items to recent window."""
        cutoff = time.time() - self.recent_window_seconds
        return [
            item for item in items
            if (getattr(item, timestamp_attr, item) if hasattr(item, timestamp_attr) else item) > cutoff
        ]

    @property
    def accuracy(self) -> Optional[float]:
        """Calculate hit accuracy."""
        if self._accuracy is not None:
            return self._accuracy

        recent_shots = self._get_recent_items(self.shots)
        if not recent_shots:
            return None

        hits = sum(1 for s in recent_shots if s.hit)
        self._accuracy = hits / len(recent_shots)
        return self._accuracy

    @property
    def headshot_rate(self) -> Optional[float]:
        """Calculate headshot rate among hits."""
        recent_shots = [s for s in self._get_recent_items(self.shots) if s.hit]
        if not recent_shots:
            return None

        headshots = sum(1 for s in recent_shots if s.headshot)
        return headshots / len(recent_shots)

    @property
    def average_reaction_time(self) -> Optional[float]:
        """Calculate average reaction time in ms."""
        if self._avg_reaction_time is not None:
            return self._avg_reaction_time

        recent_reactions = self._get_recent_items(self.reactions)
        if not recent_reactions:
            return None

        self._avg_reaction_time = statistics.mean(r.reaction_time_ms for r in recent_reactions)
        return self._avg_reaction_time

    @property
    def reaction_times(self) -> List[float]:
        """Get list of recent reaction times."""
        return [r.reaction_time_ms for r in self._get_recent_items(self.reactions)]

    @property
    def kill_distances(self) -> List[float]:
        """Get list of recent kill distances."""
        return [k.distance for k in self._get_recent_items(self.kills)]

    @property
    def consecutive_kills(self) -> int:
        """Get current consecutive kills without dying."""
        if not self.kills:
            return 0

        recent_kills = self._get_recent_items(self.kills)
        recent_deaths = self._get_recent_items(self.deaths, "timestamp" if hasattr(self.deaths[0] if self.deaths else 0, "timestamp") else None)

        if not recent_deaths:
            return len(recent_kills)

        last_death = max(recent_deaths) if isinstance(recent_deaths[0], float) else max(recent_deaths)
        return sum(1 for k in recent_kills if k.timestamp > last_death)

    def get_speed_variance(self) -> Optional[float]:
        """Calculate variance in movement speeds."""
        recent_movements = self._get_recent_items(self.movements)
        if len(recent_movements) < 2:
            return None

        speeds = [m.speed for m in recent_movements]
        return statistics.variance(speeds) if len(speeds) > 1 else 0.0

    def get_damage_ratio(self) -> Optional[float]:
        """Calculate ratio of actual to expected damage."""
        recent_damages = self._get_recent_items(self.damages)
        if not recent_damages:
            return None

        total_dealt = sum(d.damage_dealt for d in recent_damages)
        total_expected = sum(d.expected_damage for d in recent_damages)

        if total_expected == 0:
            return None

        return total_dealt / total_expected

    def get_wall_hit_rate(self) -> Optional[float]:
        """Calculate rate of hitting targets through walls."""
        recent_shots = [s for s in self._get_recent_items(self.shots) if s.hit]
        if not recent_shots:
            return None

        wall_hits = sum(1 for s in recent_shots if not s.target_visible)
        return wall_hits / len(recent_shots)


@dataclass
class AnomalyReport:
    """Report of a detected anomaly."""
    player_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float  # 0.0 to 1.0
    timestamp: float
    details: str
    evidence: Dict[str, float] = field(default_factory=dict)


class AnomalyDetector:
    """
    Detects cheating anomalies by analyzing player behavior.

    Thread-safe implementation for concurrent event recording and analysis.
    """

    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        """
        Initialize the anomaly detector.

        Args:
            thresholds: Configuration for detection thresholds
        """
        self._thresholds = thresholds or AnomalyThresholds()
        self._player_stats: Dict[str, PlayerStats] = {}
        self._lock = threading.RLock()
        self._custom_detectors: List[Callable[[PlayerStats, AnomalyThresholds], Optional[AnomalyReport]]] = []
        self._anomaly_history: Dict[str, List[AnomalyReport]] = {}

    @property
    def thresholds(self) -> AnomalyThresholds:
        """Get current thresholds."""
        return self._thresholds

    def set_thresholds(self, thresholds: AnomalyThresholds) -> None:
        """Update detection thresholds."""
        self._thresholds = thresholds

    def _get_player_stats(self, player_id: str) -> PlayerStats:
        """Get or create player stats."""
        if player_id not in self._player_stats:
            self._player_stats[player_id] = PlayerStats(player_id=player_id)
        return self._player_stats[player_id]

    def record_event(
        self,
        player_id: str,
        event_type: str,
        data: dict
    ) -> None:
        """
        Record a game event for analysis.

        Args:
            player_id: The player's unique identifier
            event_type: Type of event (shot, kill, movement, reaction, damage)
            data: Event data
        """
        with self._lock:
            stats = self._get_player_stats(player_id)
            timestamp = data.get("timestamp", time.time())

            if event_type == "shot":
                stats.add_shot(ShotEvent(
                    timestamp=timestamp,
                    hit=data.get("hit", False),
                    headshot=data.get("headshot", False),
                    distance=data.get("distance", 0.0),
                    target_visible=data.get("target_visible", True),
                    target_id=data.get("target_id")
                ))

            elif event_type == "kill":
                stats.add_kill(KillEvent(
                    timestamp=timestamp,
                    victim_id=data.get("victim_id", ""),
                    weapon=data.get("weapon", ""),
                    distance=data.get("distance", 0.0),
                    time_to_kill=data.get("time_to_kill", 0.0),
                    headshot=data.get("headshot", False)
                ))

            elif event_type == "death":
                stats.add_death(timestamp)

            elif event_type == "movement":
                stats.add_movement(MovementEvent(
                    timestamp=timestamp,
                    speed=data.get("speed", 0.0),
                    position_delta=data.get("position_delta", 0.0),
                    time_delta=data.get("time_delta", 0.0)
                ))

            elif event_type == "reaction":
                stats.add_reaction(ReactionEvent(
                    timestamp=timestamp,
                    reaction_time_ms=data.get("reaction_time_ms", 0.0),
                    stimulus_type=data.get("stimulus_type", "")
                ))

            elif event_type == "damage":
                stats.add_damage(DamageEvent(
                    timestamp=timestamp,
                    target_id=data.get("target_id", ""),
                    damage_dealt=data.get("damage_dealt", 0.0),
                    expected_damage=data.get("expected_damage", 0.0),
                    weapon=data.get("weapon", "")
                ))

    def register_custom_detector(
        self,
        detector: Callable[[PlayerStats, AnomalyThresholds], Optional[AnomalyReport]]
    ) -> None:
        """
        Register a custom anomaly detector function.

        Args:
            detector: Function that takes player stats and thresholds,
                     returns AnomalyReport if anomaly detected
        """
        with self._lock:
            self._custom_detectors.append(detector)

    def analyze_player(self, player_id: str) -> List[AnomalyReport]:
        """
        Analyze a player for anomalies.

        Args:
            player_id: The player's unique identifier

        Returns:
            List of detected anomalies
        """
        with self._lock:
            if player_id not in self._player_stats:
                return []

            stats = self._player_stats[player_id]
            anomalies = []

            # Check aimbot (high accuracy + high headshot rate)
            aimbot_report = self._check_aimbot(stats)
            if aimbot_report:
                anomalies.append(aimbot_report)

            # Check speed hack
            speed_report = self._check_speed_hack(stats)
            if speed_report:
                anomalies.append(speed_report)

            # Check impossible reaction times
            reaction_report = self._check_impossible_reaction(stats)
            if reaction_report:
                anomalies.append(reaction_report)

            # Check wallhack suspicion
            wallhack_report = self._check_wallhack(stats)
            if wallhack_report:
                anomalies.append(wallhack_report)

            # Check damage hack
            damage_report = self._check_damage_hack(stats)
            if damage_report:
                anomalies.append(damage_report)

            # Run custom detectors
            for detector in self._custom_detectors:
                report = detector(stats, self._thresholds)
                if report:
                    anomalies.append(report)

            # Store in history
            if anomalies:
                if player_id not in self._anomaly_history:
                    self._anomaly_history[player_id] = []
                self._anomaly_history[player_id].extend(anomalies)

            return anomalies

    def _check_aimbot(self, stats: PlayerStats) -> Optional[AnomalyReport]:
        """Check for aimbot indicators."""
        # Need enough samples
        recent_shots = stats._get_recent_items(stats.shots)
        if len(recent_shots) < self._thresholds.accuracy_sample_size:
            return None

        accuracy = stats.accuracy
        headshot_rate = stats.headshot_rate

        if accuracy is None or headshot_rate is None:
            return None

        # Both accuracy and headshot rate must be suspiciously high
        if accuracy >= self._thresholds.accuracy_threshold:
            if headshot_rate >= self._thresholds.headshot_rate_threshold:
                confidence = min(1.0, (accuracy + headshot_rate) / 2)
                return AnomalyReport(
                    player_id=stats.player_id,
                    anomaly_type=AnomalyType.AIMBOT,
                    severity=AnomalySeverity.CRITICAL,
                    confidence=confidence,
                    timestamp=time.time(),
                    details=f"Suspiciously high accuracy ({accuracy:.2%}) and headshot rate ({headshot_rate:.2%})",
                    evidence={
                        "accuracy": accuracy,
                        "headshot_rate": headshot_rate,
                        "sample_size": len(recent_shots)
                    }
                )

        return None

    def _check_speed_hack(self, stats: PlayerStats) -> Optional[AnomalyReport]:
        """Check for speed hack indicators."""
        recent_movements = stats._get_recent_items(stats.movements)
        if len(recent_movements) < ANOMALY_DETECTION.MIN_MOVEMENT_SAMPLES:
            return None

        # Check for abnormally low speed variance (constant max speed)
        variance = stats.get_speed_variance()
        if variance is None:
            return None

        if variance < self._thresholds.speed_variance_threshold:
            avg_speed = statistics.mean(m.speed for m in recent_movements)
            # Only flag if also moving fast (use config constant)
            if avg_speed > ANOMALY_DETECTION.SPEED_HACK_MIN_SPEED:
                return AnomalyReport(
                    player_id=stats.player_id,
                    anomaly_type=AnomalyType.SPEED_HACK,
                    severity=AnomalySeverity.HIGH,
                    confidence=0.7,
                    timestamp=time.time(),
                    details=f"Abnormally consistent movement speed (variance: {variance:.4f})",
                    evidence={
                        "speed_variance": variance,
                        "average_speed": avg_speed,
                        "sample_size": len(recent_movements)
                    }
                )

        return None

    def _check_impossible_reaction(self, stats: PlayerStats) -> Optional[AnomalyReport]:
        """Check for impossible reaction times."""
        reaction_times = stats.reaction_times
        if len(reaction_times) < self._thresholds.reaction_sample_size:
            return None

        avg_reaction = stats.average_reaction_time
        if avg_reaction is None:
            return None

        # Count suspiciously fast reactions
        fast_count = sum(1 for rt in reaction_times if rt < self._thresholds.min_reaction_time_ms)
        fast_ratio = fast_count / len(reaction_times)

        # Use config constant for ratio threshold
        if fast_ratio > ANOMALY_DETECTION.IMPOSSIBLE_REACTION_RATIO:
            return AnomalyReport(
                player_id=stats.player_id,
                anomaly_type=AnomalyType.IMPOSSIBLE_REACTION,
                severity=AnomalySeverity.CRITICAL,
                confidence=fast_ratio,
                timestamp=time.time(),
                details=f"Impossible reaction times: {fast_ratio:.1%} under {self._thresholds.min_reaction_time_ms}ms",
                evidence={
                    "avg_reaction_time_ms": avg_reaction,
                    "fast_reaction_ratio": fast_ratio,
                    "min_reaction_time_ms": min(reaction_times),
                    "sample_size": len(reaction_times)
                }
            )

        return None

    def _check_wallhack(self, stats: PlayerStats) -> Optional[AnomalyReport]:
        """Check for wallhack indicators."""
        wall_hit_rate = stats.get_wall_hit_rate()
        if wall_hit_rate is None:
            return None

        if wall_hit_rate > self._thresholds.wall_hit_rate_threshold:
            return AnomalyReport(
                player_id=stats.player_id,
                anomaly_type=AnomalyType.WALL_HACK_SUSPECT,
                severity=AnomalySeverity.MEDIUM,
                confidence=min(1.0, wall_hit_rate / ANOMALY_DETECTION.WALLHACK_CONFIDENCE_DIVISOR),
                timestamp=time.time(),
                details=f"High rate of hitting non-visible targets ({wall_hit_rate:.2%})",
                evidence={
                    "wall_hit_rate": wall_hit_rate
                }
            )

        return None

    def _check_damage_hack(self, stats: PlayerStats) -> Optional[AnomalyReport]:
        """Check for damage modification."""
        damage_ratio = stats.get_damage_ratio()
        if damage_ratio is None:
            return None

        if damage_ratio > self._thresholds.damage_multiplier_threshold:
            return AnomalyReport(
                player_id=stats.player_id,
                anomaly_type=AnomalyType.DAMAGE_HACK,
                severity=AnomalySeverity.CRITICAL,
                confidence=min(1.0, damage_ratio / ANOMALY_DETECTION.DAMAGE_CONFIDENCE_DIVISOR),
                timestamp=time.time(),
                details=f"Damage dealt {damage_ratio:.1f}x expected",
                evidence={
                    "damage_ratio": damage_ratio
                }
            )

        return None

    def get_anomaly_history(self, player_id: str) -> List[AnomalyReport]:
        """Get historical anomaly reports for a player."""
        with self._lock:
            return list(self._anomaly_history.get(player_id, []))

    def get_player_risk_score(self, player_id: str) -> float:
        """
        Calculate an overall risk score for a player.

        Args:
            player_id: The player's unique identifier

        Returns:
            Risk score from 0.0 (clean) to 1.0 (definitely cheating)
        """
        # Import here to avoid circular import
        from engine.networking.security.config import RESPONSE_CONFIG

        with self._lock:
            history = self._anomaly_history.get(player_id, [])
            if not history:
                return 0.0

            # Weight anomalies by severity and recency
            current_time = time.time()
            weighted_score = 0.0
            decay_seconds = RESPONSE_CONFIG.RISK_SCORE_DECAY_HOURS * 3600

            for report in history:
                age = current_time - report.timestamp
                # Use config constants for decay calculation
                age_factor = max(
                    RESPONSE_CONFIG.RISK_SCORE_MIN_AGE_FACTOR,
                    1.0 - (age / decay_seconds)
                )

                # Use config constants for severity weights
                severity_weight = {
                    AnomalySeverity.LOW: RESPONSE_CONFIG.SEVERITY_WEIGHT_LOW,
                    AnomalySeverity.MEDIUM: RESPONSE_CONFIG.SEVERITY_WEIGHT_MEDIUM,
                    AnomalySeverity.HIGH: RESPONSE_CONFIG.SEVERITY_WEIGHT_HIGH,
                    AnomalySeverity.CRITICAL: RESPONSE_CONFIG.SEVERITY_WEIGHT_CRITICAL
                }[report.severity]

                weighted_score += severity_weight * report.confidence * age_factor

            # Normalize to 0-1 range using config constant
            return min(1.0, weighted_score / RESPONSE_CONFIG.RISK_SCORE_NORMALIZATION)

    def clear_player_data(self, player_id: str) -> None:
        """Clear all data for a player."""
        with self._lock:
            self._player_stats.pop(player_id, None)
            self._anomaly_history.pop(player_id, None)

    def get_all_suspicious_players(self, min_risk_score: float = 0.5) -> List[str]:
        """Get list of players above a risk threshold."""
        with self._lock:
            suspicious = []
            for player_id in self._player_stats.keys():
                if self.get_player_risk_score(player_id) >= min_risk_score:
                    suspicious.append(player_id)
            return suspicious
