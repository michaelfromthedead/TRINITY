"""
Response management system for anti-cheat actions.

This module handles the escalation and enforcement of anti-cheat responses,
from warnings to permanent bans.

Thread-safety: All public methods are thread-safe.
Security: Uses secrets module for any security-sensitive random values.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set
import hashlib
import secrets
import time
import threading

from engine.networking.security.config import (
    RESPONSE_CONFIG,
    VALIDATION_LIMITS,
)


class ResponseSeverity(Enum):
    """Severity levels for anti-cheat responses."""
    WARNING = 1
    KICK = 2
    TEMP_BAN = 3
    PERMANENT_BAN = 4
    SHADOW_BAN = 5  # Player can play but only with other cheaters


@dataclass
class CheatResponse:
    """
    Response to a detected cheat.

    Attributes:
        severity: How severe the response is
        reason: Human-readable reason for the response
        duration: Duration in seconds (0 for permanent, None for non-bans)
        message_to_player: Message to show the player (if any)
    """
    severity: ResponseSeverity
    reason: str
    duration: Optional[float] = None
    message_to_player: Optional[str] = None

    def is_ban(self) -> bool:
        """Check if this response is a ban."""
        return self.severity in (
            ResponseSeverity.TEMP_BAN,
            ResponseSeverity.PERMANENT_BAN,
            ResponseSeverity.SHADOW_BAN
        )


@dataclass
class BanRecord:
    """
    Record of a player ban.

    Attributes:
        player_id: The banned player's ID
        hwid: Hardware ID for hardware bans
        ip_address: IP address for IP bans
        ban_type: Type of ban
        reason: Reason for the ban
        created_at: When the ban was created
        expires_at: When the ban expires (None for permanent)
        issuer_id: ID of the admin/system that issued the ban
        appeal_count: Number of appeals made
    """
    player_id: str
    ban_type: ResponseSeverity
    reason: str
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hwid: Optional[str] = None
    ip_address: Optional[str] = None
    issuer_id: str = "anti-cheat-system"
    appeal_count: int = 0

    def is_active(self) -> bool:
        """Check if the ban is still active."""
        if self.expires_at is None:
            return True  # Permanent
        return time.time() < self.expires_at

    def is_permanent(self) -> bool:
        """Check if this is a permanent ban."""
        return self.expires_at is None

    def remaining_time(self) -> Optional[float]:
        """Get remaining ban time in seconds."""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0, remaining)


@dataclass
class ViolationRecord:
    """Record of a single violation."""
    timestamp: float
    anomaly_type: str
    severity: int
    confidence: float
    details: str


@dataclass
class PlayerViolationHistory:
    """Tracks violation history for a player."""
    player_id: str
    violations: List[ViolationRecord] = field(default_factory=list)
    warnings_issued: int = 0
    kicks_issued: int = 0
    temp_bans_issued: int = 0
    last_response_time: float = 0.0
    first_violation_time: Optional[float] = None


@dataclass
class EscalationRule:
    """
    Rule for escalating responses based on violation count.

    Attributes:
        min_violations: Minimum violations to trigger this rule
        min_kicks: Minimum kicks before this rule applies
        min_warnings: Minimum warnings before this rule applies
        response: The response to issue
        cooldown: Time before same response can be issued again
    """
    response: CheatResponse
    min_violations: int = 0
    min_warnings: int = 0
    min_kicks: int = 0
    min_temp_bans: int = 0
    cooldown: float = 0.0


# Default escalation rules (using config constants)
DEFAULT_ESCALATION_RULES: List[EscalationRule] = [
    # First violation: warning
    EscalationRule(
        response=CheatResponse(
            severity=ResponseSeverity.WARNING,
            reason="Suspicious activity detected",
            message_to_player="Warning: Suspicious activity has been detected. Continued violations may result in action."
        ),
        min_violations=1,
        cooldown=RESPONSE_CONFIG.WARNING_COOLDOWN
    ),
    # After configured warnings: kick
    EscalationRule(
        response=CheatResponse(
            severity=ResponseSeverity.KICK,
            reason="Multiple warnings for suspicious activity",
            message_to_player="You have been kicked due to repeated suspicious activity."
        ),
        min_warnings=RESPONSE_CONFIG.WARNINGS_BEFORE_KICK,
        cooldown=RESPONSE_CONFIG.KICK_COOLDOWN
    ),
    # After configured kicks: temp ban (1 hour)
    EscalationRule(
        response=CheatResponse(
            severity=ResponseSeverity.TEMP_BAN,
            reason="Repeated kicks for suspicious activity",
            duration=RESPONSE_CONFIG.FIRST_TEMP_BAN_DURATION,
            message_to_player="You have been temporarily banned for 1 hour."
        ),
        min_kicks=RESPONSE_CONFIG.KICKS_BEFORE_TEMP_BAN,
        cooldown=RESPONSE_CONFIG.BAN_COOLDOWN
    ),
    # After configured temp bans: longer temp ban (24 hours)
    EscalationRule(
        response=CheatResponse(
            severity=ResponseSeverity.TEMP_BAN,
            reason="Multiple temporary bans",
            duration=RESPONSE_CONFIG.SECOND_TEMP_BAN_DURATION,
            message_to_player="You have been temporarily banned for 24 hours."
        ),
        min_temp_bans=RESPONSE_CONFIG.TEMP_BANS_BEFORE_LONGER_BAN,
        cooldown=RESPONSE_CONFIG.BAN_COOLDOWN
    ),
    # After configured temp bans: permanent ban
    EscalationRule(
        response=CheatResponse(
            severity=ResponseSeverity.PERMANENT_BAN,
            reason="Continued cheating after multiple bans",
            message_to_player="You have been permanently banned for cheating."
        ),
        min_temp_bans=RESPONSE_CONFIG.TEMP_BANS_BEFORE_PERMANENT,
        cooldown=RESPONSE_CONFIG.BAN_COOLDOWN
    ),
]


class ResponseManager:
    """
    Manages anti-cheat responses and ban enforcement.

    Thread-safe implementation for concurrent access.
    """

    def __init__(
        self,
        escalation_rules: Optional[List[EscalationRule]] = None,
        on_response: Optional[Callable[[str, CheatResponse], None]] = None
    ):
        """
        Initialize the response manager.

        Args:
            escalation_rules: Custom escalation rules (uses defaults if None)
            on_response: Callback when a response is issued
        """
        self._escalation_rules = escalation_rules or DEFAULT_ESCALATION_RULES.copy()
        self._player_histories: Dict[str, PlayerViolationHistory] = {}
        self._ban_records: Dict[str, BanRecord] = {}
        self._hwid_bans: Dict[str, BanRecord] = {}
        self._ip_bans: Dict[str, BanRecord] = {}
        self._shadow_banned: Set[str] = set()
        self._on_response = on_response
        self._lock = threading.RLock()

    def record_violation(
        self,
        player_id: str,
        anomaly_type: str,
        severity: int = 1,
        confidence: float = 1.0,
        details: str = ""
    ) -> Optional[CheatResponse]:
        """
        Record a violation and determine the appropriate response.

        Args:
            player_id: The player's unique identifier
            anomaly_type: Type of anomaly detected
            severity: Severity level (1-4)
            confidence: Confidence in the detection (0.0-1.0)
            details: Additional details

        Returns:
            The response to take, if any
        """
        with self._lock:
            # Get or create player history
            if player_id not in self._player_histories:
                self._player_histories[player_id] = PlayerViolationHistory(player_id=player_id)

            history = self._player_histories[player_id]

            # Record the violation
            violation = ViolationRecord(
                timestamp=time.time(),
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=confidence,
                details=details
            )
            history.violations.append(violation)

            if history.first_violation_time is None:
                history.first_violation_time = violation.timestamp

            # Determine response
            response = self._determine_response(history)

            if response:
                self._apply_response(player_id, response, history)

            return response

    def _determine_response(self, history: PlayerViolationHistory) -> Optional[CheatResponse]:
        """Determine the appropriate response based on history."""
        current_time = time.time()

        # Find the most severe applicable rule
        best_response: Optional[CheatResponse] = None

        for rule in self._escalation_rules:
            # Check if rule requirements are met
            if (len(history.violations) >= rule.min_violations and
                history.warnings_issued >= rule.min_warnings and
                history.kicks_issued >= rule.min_kicks and
                history.temp_bans_issued >= rule.min_temp_bans):

                # Check cooldown
                if current_time - history.last_response_time >= rule.cooldown:
                    # This rule is applicable
                    if best_response is None or rule.response.severity.value > best_response.severity.value:
                        best_response = rule.response

        return best_response

    def _apply_response(
        self,
        player_id: str,
        response: CheatResponse,
        history: PlayerViolationHistory
    ) -> None:
        """Apply a response and update history."""
        history.last_response_time = time.time()

        if response.severity == ResponseSeverity.WARNING:
            history.warnings_issued += 1
        elif response.severity == ResponseSeverity.KICK:
            history.kicks_issued += 1
        elif response.severity == ResponseSeverity.TEMP_BAN:
            history.temp_bans_issued += 1
            self._create_ban(player_id, response)
        elif response.severity == ResponseSeverity.PERMANENT_BAN:
            self._create_ban(player_id, response)
        elif response.severity == ResponseSeverity.SHADOW_BAN:
            self._shadow_banned.add(player_id)

        # Invoke callback if set
        if self._on_response:
            self._on_response(player_id, response)

    def _create_ban(
        self,
        player_id: str,
        response: CheatResponse,
        hwid: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> BanRecord:
        """Create a ban record."""
        expires_at = None
        if response.duration is not None and response.duration > 0:
            expires_at = time.time() + response.duration

        ban = BanRecord(
            player_id=player_id,
            ban_type=response.severity,
            reason=response.reason,
            expires_at=expires_at,
            hwid=hwid,
            ip_address=ip_address
        )

        self._ban_records[player_id] = ban

        # Also ban by hardware ID and IP if provided
        if hwid:
            self._hwid_bans[hwid] = ban
        if ip_address:
            self._ip_bans[ip_address] = ban

        return ban

    def get_response(self, player_id: str) -> Optional[CheatResponse]:
        """
        Get the current response status for a player.

        Args:
            player_id: The player's unique identifier

        Returns:
            The current response in effect, if any
        """
        with self._lock:
            # Check bans first
            if player_id in self._ban_records:
                ban = self._ban_records[player_id]
                if ban.is_active():
                    return CheatResponse(
                        severity=ban.ban_type,
                        reason=ban.reason,
                        duration=ban.remaining_time(),
                        message_to_player=f"Banned: {ban.reason}"
                    )

            # Check shadow ban
            if player_id in self._shadow_banned:
                return CheatResponse(
                    severity=ResponseSeverity.SHADOW_BAN,
                    reason="Shadow banned for cheating",
                    message_to_player=None  # Don't tell them
                )

            return None

    def is_banned(self, player_id: str) -> bool:
        """Check if a player is banned."""
        with self._lock:
            if player_id in self._ban_records:
                return self._ban_records[player_id].is_active()
            return False

    def is_shadow_banned(self, player_id: str) -> bool:
        """Check if a player is shadow banned."""
        with self._lock:
            return player_id in self._shadow_banned

    def check_hwid_ban(self, hwid: str) -> Optional[BanRecord]:
        """Check if a hardware ID is banned."""
        with self._lock:
            if hwid in self._hwid_bans:
                ban = self._hwid_bans[hwid]
                if ban.is_active():
                    return ban
            return None

    def check_ip_ban(self, ip_address: str) -> Optional[BanRecord]:
        """Check if an IP address is banned."""
        with self._lock:
            if ip_address in self._ip_bans:
                ban = self._ip_bans[ip_address]
                if ban.is_active():
                    return ban
            return None

    def get_ban_record(self, player_id: str) -> Optional[BanRecord]:
        """Get the ban record for a player."""
        with self._lock:
            return self._ban_records.get(player_id)

    def get_violation_history(self, player_id: str) -> Optional[PlayerViolationHistory]:
        """Get the violation history for a player."""
        with self._lock:
            return self._player_histories.get(player_id)

    def lift_ban(self, player_id: str) -> bool:
        """
        Lift a ban for a player.

        Args:
            player_id: The player's unique identifier

        Returns:
            True if a ban was lifted
        """
        with self._lock:
            if player_id in self._ban_records:
                ban = self._ban_records.pop(player_id)
                # Also remove associated bans
                if ban.hwid and ban.hwid in self._hwid_bans:
                    del self._hwid_bans[ban.hwid]
                if ban.ip_address and ban.ip_address in self._ip_bans:
                    del self._ip_bans[ban.ip_address]
                return True
            return False

    def remove_shadow_ban(self, player_id: str) -> bool:
        """Remove a shadow ban."""
        with self._lock:
            if player_id in self._shadow_banned:
                self._shadow_banned.remove(player_id)
                return True
            return False

    def clear_violations(self, player_id: str) -> None:
        """Clear all violation history for a player (does not lift bans)."""
        with self._lock:
            if player_id in self._player_histories:
                del self._player_histories[player_id]

    def add_manual_ban(
        self,
        player_id: str,
        reason: str,
        duration: Optional[float] = None,
        hwid: Optional[str] = None,
        ip_address: Optional[str] = None,
        issuer_id: str = "admin"
    ) -> BanRecord:
        """
        Manually add a ban.

        Args:
            player_id: The player to ban
            reason: Reason for the ban
            duration: Ban duration in seconds (None for permanent)
            hwid: Hardware ID for hardware ban
            ip_address: IP for IP ban
            issuer_id: ID of the issuer

        Returns:
            The created ban record

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not player_id or not isinstance(player_id, str):
            raise ValueError("player_id must be a non-empty string")
        if not reason or not isinstance(reason, str):
            raise ValueError("reason must be a non-empty string")
        if duration is not None:
            if duration <= 0:
                raise ValueError("duration must be positive")
            if duration > VALIDATION_LIMITS.MAX_BAN_DURATION_SECONDS:
                raise ValueError(
                    f"duration exceeds maximum ({VALIDATION_LIMITS.MAX_BAN_DURATION_SECONDS} seconds)"
                )

        with self._lock:
            severity = ResponseSeverity.PERMANENT_BAN if duration is None else ResponseSeverity.TEMP_BAN
            response = CheatResponse(
                severity=severity,
                reason=reason,
                duration=duration
            )
            ban = self._create_ban(player_id, response, hwid, ip_address)
            ban.issuer_id = issuer_id
            return ban

    def get_all_active_bans(self) -> List[BanRecord]:
        """Get all active bans."""
        with self._lock:
            return [ban for ban in self._ban_records.values() if ban.is_active()]

    def get_all_shadow_banned(self) -> Set[str]:
        """Get all shadow banned player IDs."""
        with self._lock:
            return set(self._shadow_banned)

    def cleanup_expired_bans(self) -> int:
        """
        Remove expired bans from records.

        Returns:
            Number of bans cleaned up
        """
        with self._lock:
            expired_players = [
                pid for pid, ban in self._ban_records.items()
                if not ban.is_active()
            ]

            for player_id in expired_players:
                self.lift_ban(player_id)

            return len(expired_players)

    def get_statistics(self) -> Dict[str, int]:
        """Get response manager statistics."""
        with self._lock:
            total_violations = sum(
                len(h.violations) for h in self._player_histories.values()
            )
            return {
                "total_players_tracked": len(self._player_histories),
                "total_violations": total_violations,
                "active_bans": len([b for b in self._ban_records.values() if b.is_active()]),
                "shadow_banned": len(self._shadow_banned),
                "total_warnings_issued": sum(h.warnings_issued for h in self._player_histories.values()),
                "total_kicks_issued": sum(h.kicks_issued for h in self._player_histories.values()),
                "total_temp_bans_issued": sum(h.temp_bans_issued for h in self._player_histories.values()),
            }


def generate_hwid_hash(components: List[str], salt: Optional[str] = None) -> str:
    """
    Generate a hardware ID hash from system components.

    Uses SHA-256 with optional salt for additional security.

    Args:
        components: List of hardware component strings
        salt: Optional salt to add randomness (use secrets.token_hex() for secure salt)

    Returns:
        SHA-256 hash of the components

    Raises:
        ValueError: If components is empty
    """
    if not components:
        raise ValueError("components must be a non-empty list")

    combined = "|".join(sorted(components))
    if salt:
        combined = f"{salt}|{combined}"
    return hashlib.sha256(combined.encode()).hexdigest()


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.

    Uses secrets module for security-sensitive random values.

    Args:
        length: Length of the token in bytes (default 32)

    Returns:
        Hex-encoded secure random token
    """
    return secrets.token_hex(length)
