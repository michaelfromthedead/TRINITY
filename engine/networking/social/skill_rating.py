"""
Skill Rating System Module.

Provides implementations of Elo, Glicko-2, and MMR systems for tracking
player skill levels and calculating rating changes after matches.
"""

from dataclasses import dataclass, field
from typing import Optional
from threading import Lock
import logging
import math
import time

from .config import SOCIAL_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SkillRating:
    """
    Represents a player's skill rating with uncertainty.

    Attributes:
        rating: The player's current rating (e.g., 1500 for Elo).
        uncertainty: The rating deviation/uncertainty (higher = less certain).
        games_played: Total number of rated games played.
        last_updated: Timestamp of last rating update.
    """
    rating: float = SOCIAL_CONFIG.SkillRating.DEFAULT_RATING
    uncertainty: float = SOCIAL_CONFIG.SkillRating.DEFAULT_UNCERTAINTY
    games_played: int = 0
    last_updated: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate rating values."""
        if self.rating < 0:
            raise ValueError("Rating cannot be negative")
        if self.uncertainty < 0:
            raise ValueError("Uncertainty cannot be negative")
        if self.games_played < 0:
            raise ValueError("Games played cannot be negative")


@dataclass
class MatchOutcome:
    """Represents the outcome of a match for rating calculations."""
    opponent_id: str
    opponent_rating: SkillRating
    result: float  # 1.0 = win, 0.5 = draw, 0.0 = loss


class EloCalculator:
    """
    Classic Elo rating calculator.

    The Elo system was originally designed for chess and provides
    a simple yet effective way to rate players in two-player games.

    Formula:
        Expected score: E = 1 / (1 + 10^((opponent_rating - player_rating) / 400))
        New rating: R' = R + K * (actual - expected)
    """

    def __init__(
        self,
        k_factor: Optional[int] = None,
        floor_rating: Optional[float] = None
    ) -> None:
        """
        Initialize the Elo calculator.

        Args:
            k_factor: The K-factor determining how much ratings change.
                     Higher K = more volatile ratings.
                     Typical values: 16 (master), 24 (intermediate), 32 (new).
            floor_rating: Minimum rating a player can have.
        """
        self.k_factor = k_factor if k_factor is not None else SOCIAL_CONFIG.SkillRating.ELO_K_FACTOR_DEFAULT
        self.floor_rating = floor_rating if floor_rating is not None else SOCIAL_CONFIG.SkillRating.MIN_RATING

    def calculate_expected(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate the expected score for player A against player B.

        Args:
            rating_a: Player A's rating.
            rating_b: Player B's rating.

        Returns:
            Expected score between 0 and 1.
        """
        scale_divisor = SOCIAL_CONFIG.SkillRating.ELO_SCALE_DIVISOR
        exponent = (rating_b - rating_a) / scale_divisor
        return 1.0 / (1.0 + math.pow(10, exponent))

    def update_ratings(
        self,
        winner_rating: float,
        loser_rating: float,
        is_draw: bool = False
    ) -> tuple[float, float]:
        """
        Calculate new ratings after a match.

        Args:
            winner_rating: Winner's current rating.
            loser_rating: Loser's current rating.
            is_draw: If True, treat as a draw.

        Returns:
            Tuple of (new_winner_rating, new_loser_rating).
        """
        expected_winner = self.calculate_expected(winner_rating, loser_rating)
        expected_loser = 1.0 - expected_winner

        if is_draw:
            actual_winner = 0.5
            actual_loser = 0.5
        else:
            actual_winner = 1.0
            actual_loser = 0.0

        new_winner = winner_rating + self.k_factor * (actual_winner - expected_winner)
        new_loser = loser_rating + self.k_factor * (actual_loser - expected_loser)

        # Apply floor
        new_winner = max(self.floor_rating, new_winner)
        new_loser = max(self.floor_rating, new_loser)

        return new_winner, new_loser

    def get_dynamic_k_factor(
        self,
        rating: float,
        games_played: int,
        provisional_games: Optional[int] = None
    ) -> int:
        """
        Get a dynamic K-factor based on player experience and rating.

        Args:
            rating: Player's current rating.
            games_played: Number of games the player has played.
            provisional_games: Games considered "provisional" period.

        Returns:
            Dynamic K-factor.
        """
        if provisional_games is None:
            provisional_games = SOCIAL_CONFIG.SkillRating.ELO_PROVISIONAL_GAMES

        # New players have higher K for faster calibration
        if games_played < provisional_games:
            return SOCIAL_CONFIG.SkillRating.ELO_K_FACTOR_NEW_PLAYER

        # High-rated players have lower K for stability
        if rating >= SOCIAL_CONFIG.SkillRating.ELO_HIGH_RATING_THRESHOLD:
            return SOCIAL_CONFIG.SkillRating.ELO_K_FACTOR_HIGH_RATED

        return self.k_factor


class Glicko2Calculator:
    """
    Simplified Glicko-2 rating calculator.

    Glicko-2 extends Elo by tracking rating uncertainty (deviation)
    and volatility, providing more accurate ratings especially for
    players who play infrequently.

    Key improvements over Elo:
    - Tracks uncertainty in ratings
    - Handles inactivity (uncertainty increases over time)
    - Considers opponent strength more accurately
    """

    # System constants (use config values)
    TAU: float = SOCIAL_CONFIG.SkillRating.GLICKO2_TAU
    EPSILON: float = SOCIAL_CONFIG.SkillRating.GLICKO2_EPSILON
    MAX_ITERATIONS: int = SOCIAL_CONFIG.SkillRating.GLICKO2_MAX_ITERATIONS

    # Scaling constants (Glicko-2 uses a different scale internally)
    SCALE_FACTOR: float = SOCIAL_CONFIG.SkillRating.GLICKO2_SCALE_FACTOR

    def __init__(
        self,
        default_rating: Optional[float] = None,
        default_deviation: Optional[float] = None,
        default_volatility: Optional[float] = None,
        inactivity_decay_days: Optional[float] = None
    ) -> None:
        """
        Initialize the Glicko-2 calculator.

        Args:
            default_rating: Starting rating for new players.
            default_deviation: Starting rating deviation.
            default_volatility: Starting volatility (typical 0.03-0.09).
            inactivity_decay_days: Days before uncertainty starts increasing.
        """
        self.default_rating = default_rating if default_rating is not None else SOCIAL_CONFIG.SkillRating.DEFAULT_RATING
        self.default_deviation = default_deviation if default_deviation is not None else SOCIAL_CONFIG.SkillRating.DEFAULT_UNCERTAINTY
        self.default_volatility = default_volatility if default_volatility is not None else SOCIAL_CONFIG.SkillRating.GLICKO2_DEFAULT_VOLATILITY
        self.inactivity_decay_days = inactivity_decay_days if inactivity_decay_days is not None else SOCIAL_CONFIG.SkillRating.GLICKO2_INACTIVITY_DECAY_DAYS

    def _to_glicko2_scale(self, rating: float) -> float:
        """Convert from Elo scale to Glicko-2 scale."""
        return (rating - self.default_rating) / self.SCALE_FACTOR

    def _from_glicko2_scale(self, rating: float) -> float:
        """Convert from Glicko-2 scale to Elo scale."""
        return rating * self.SCALE_FACTOR + self.default_rating

    def _to_glicko2_deviation(self, deviation: float) -> float:
        """Convert deviation to Glicko-2 scale."""
        return deviation / self.SCALE_FACTOR

    def _from_glicko2_deviation(self, deviation: float) -> float:
        """Convert deviation from Glicko-2 scale."""
        return deviation * self.SCALE_FACTOR

    def _g(self, phi: float) -> float:
        """The g function from Glicko-2."""
        return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))

    def _e(self, mu: float, mu_j: float, phi_j: float) -> float:
        """The E function (expected outcome) from Glicko-2."""
        return 1.0 / (1.0 + math.exp(-self._g(phi_j) * (mu - mu_j)))

    def apply_inactivity_decay(
        self,
        rating: SkillRating,
        days_inactive: float
    ) -> SkillRating:
        """
        Apply uncertainty decay due to inactivity.

        Ratings become less certain when a player hasn't played recently.

        Args:
            rating: Current rating.
            days_inactive: Days since last rated game.

        Returns:
            Updated rating with increased uncertainty.
        """
        if days_inactive <= self.inactivity_decay_days:
            return rating

        # Increase deviation based on inactivity
        inactive_periods = (days_inactive - self.inactivity_decay_days) / 30.0
        phi = self._to_glicko2_deviation(rating.uncertainty)

        # Standard Glicko-2 deviation increase formula
        # sigma is volatility, simplified here
        sigma = 0.06  # Default volatility
        new_phi = math.sqrt(phi * phi + inactive_periods * sigma * sigma)

        # Cap deviation at default (max uncertainty)
        new_phi = min(new_phi, self._to_glicko2_deviation(self.default_deviation))

        return SkillRating(
            rating=rating.rating,
            uncertainty=self._from_glicko2_deviation(new_phi),
            games_played=rating.games_played,
            last_updated=rating.last_updated
        )

    def update_rating(
        self,
        player: SkillRating,
        outcomes: list[MatchOutcome]
    ) -> SkillRating:
        """
        Update a player's rating based on match outcomes.

        This is a simplified Glicko-2 implementation that handles
        the common case of updating ratings after a rating period.

        Args:
            player: The player's current rating.
            outcomes: List of match outcomes in the rating period.

        Returns:
            Updated SkillRating.
        """
        if not outcomes:
            # No games played, just apply time-based uncertainty increase
            return player

        # Convert to Glicko-2 scale
        mu = self._to_glicko2_scale(player.rating)
        phi = self._to_glicko2_deviation(player.uncertainty)

        # Step 3: Compute variance
        variance_inv = 0.0
        delta_sum = 0.0

        for outcome in outcomes:
            mu_j = self._to_glicko2_scale(outcome.opponent_rating.rating)
            phi_j = self._to_glicko2_deviation(outcome.opponent_rating.uncertainty)

            g_j = self._g(phi_j)
            e_j = self._e(mu, mu_j, phi_j)

            variance_inv += g_j * g_j * e_j * (1 - e_j)
            delta_sum += g_j * (outcome.result - e_j)

        if variance_inv == 0:
            return player

        variance = 1.0 / variance_inv
        delta = variance * delta_sum

        # Step 4: Update volatility (simplified - using fixed volatility)
        sigma = 0.06

        # Step 5 & 6: Update rating deviation
        phi_star = math.sqrt(phi * phi + sigma * sigma)
        phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / variance)

        # Step 7: Update rating
        mu_new = mu + phi_new * phi_new * delta_sum

        # Convert back to Elo scale
        new_rating = self._from_glicko2_scale(mu_new)
        new_uncertainty = self._from_glicko2_deviation(phi_new)

        # Ensure minimum uncertainty for stability
        new_uncertainty = max(new_uncertainty, SOCIAL_CONFIG.SkillRating.MIN_UNCERTAINTY)

        return SkillRating(
            rating=max(SOCIAL_CONFIG.SkillRating.MIN_RATING, new_rating),
            uncertainty=new_uncertainty,
            games_played=player.games_played + len(outcomes),
            last_updated=time.time()
        )


class MMRManager:
    """
    High-level manager for player skill ratings.

    Provides a unified interface for storing, retrieving, and updating
    player ratings using either Elo or Glicko-2 systems.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        use_glicko: bool = True,
        default_rating: Optional[float] = None,
        default_uncertainty: Optional[float] = None
    ) -> None:
        """
        Initialize the MMR manager.

        Args:
            use_glicko: If True, use Glicko-2. If False, use Elo.
            default_rating: Starting rating for new players.
            default_uncertainty: Starting uncertainty for new players.
        """
        self._ratings: dict[str, SkillRating] = {}
        self._lock = Lock()

        self.use_glicko = use_glicko
        self.default_rating = default_rating if default_rating is not None else SOCIAL_CONFIG.SkillRating.DEFAULT_RATING
        self.default_uncertainty = default_uncertainty if default_uncertainty is not None else SOCIAL_CONFIG.SkillRating.DEFAULT_UNCERTAINTY

        self._elo = EloCalculator()
        self._glicko = Glicko2Calculator(
            default_rating=self.default_rating,
            default_deviation=self.default_uncertainty
        )

    def get_rating(self, player_id: str) -> SkillRating:
        """
        Get a player's current skill rating.

        Creates a default rating if the player doesn't exist.

        Args:
            player_id: The player's unique identifier.

        Returns:
            The player's SkillRating.
        """
        with self._lock:
            if player_id not in self._ratings:
                self._ratings[player_id] = SkillRating(
                    rating=self.default_rating,
                    uncertainty=self.default_uncertainty,
                    games_played=0
                )
            return self._ratings[player_id]

    def set_rating(self, player_id: str, rating: SkillRating) -> None:
        """
        Directly set a player's rating.

        Args:
            player_id: The player's unique identifier.
            rating: The new rating to set.
        """
        with self._lock:
            self._ratings[player_id] = rating

    def _get_rating_internal(self, player_id: str) -> SkillRating:
        """Get rating without acquiring lock (caller must hold lock)."""
        if player_id not in self._ratings:
            self._ratings[player_id] = SkillRating(
                rating=self.default_rating,
                uncertainty=self.default_uncertainty,
                games_played=0
            )
        return self._ratings[player_id]

    def update_after_match_simple(
        self,
        winner_id: str,
        loser_id: str,
        is_draw: bool = False
    ) -> tuple[SkillRating, SkillRating]:
        """
        Update ratings after a simple 1v1 match.

        Args:
            winner_id: Winner's player ID (or any player if draw).
            loser_id: Loser's player ID (or other player if draw).
            is_draw: Whether the match was a draw.

        Returns:
            Tuple of (winner's new rating, loser's new rating).
        """
        with self._lock:
            winner_rating = self._get_rating_internal(winner_id)
            loser_rating = self._get_rating_internal(loser_id)

            if self.use_glicko:
                # Use Glicko-2
                winner_result = 1.0 if not is_draw else 0.5
                loser_result = 0.0 if not is_draw else 0.5

                winner_outcomes = [MatchOutcome(
                    opponent_id=loser_id,
                    opponent_rating=loser_rating,
                    result=winner_result
                )]
                loser_outcomes = [MatchOutcome(
                    opponent_id=winner_id,
                    opponent_rating=winner_rating,
                    result=loser_result
                )]

                new_winner = self._glicko.update_rating(winner_rating, winner_outcomes)
                new_loser = self._glicko.update_rating(loser_rating, loser_outcomes)
            else:
                # Use Elo
                new_winner_val, new_loser_val = self._elo.update_ratings(
                    winner_rating.rating,
                    loser_rating.rating,
                    is_draw
                )

                new_winner = SkillRating(
                    rating=new_winner_val,
                    uncertainty=winner_rating.uncertainty,
                    games_played=winner_rating.games_played + 1,
                    last_updated=time.time()
                )
                new_loser = SkillRating(
                    rating=new_loser_val,
                    uncertainty=loser_rating.uncertainty,
                    games_played=loser_rating.games_played + 1,
                    last_updated=time.time()
                )

            self._ratings[winner_id] = new_winner
            self._ratings[loser_id] = new_loser

            return new_winner, new_loser

    def update_after_match(
        self,
        match_result: dict[str, float]
    ) -> dict[str, SkillRating]:
        """
        Update ratings after a match with multiple players.

        Uses a round-robin approach where each player is compared
        against every other player based on their relative scores.

        Args:
            match_result: Dict mapping player_id to their score.
                         Higher scores are better.

        Returns:
            Dict mapping player_id to their new SkillRating.
        """
        with self._lock:
            if len(match_result) < 2:
                return {}

            player_ids = list(match_result.keys())
            scores = match_result

            # Get current ratings
            ratings = {pid: self._get_rating_internal(pid) for pid in player_ids}
            new_ratings: dict[str, SkillRating] = {}

            # Process each player
            for player_id in player_ids:
                player_rating = ratings[player_id]
                player_score = scores[player_id]
                outcomes: list[MatchOutcome] = []

                # Compare against all other players
                for opponent_id in player_ids:
                    if opponent_id == player_id:
                        continue

                    opponent_rating = ratings[opponent_id]
                    opponent_score = scores[opponent_id]

                    # Determine result
                    if player_score > opponent_score:
                        result = 1.0
                    elif player_score < opponent_score:
                        result = 0.0
                    else:
                        result = 0.5

                    outcomes.append(MatchOutcome(
                        opponent_id=opponent_id,
                        opponent_rating=opponent_rating,
                        result=result
                    ))

                if self.use_glicko:
                    new_ratings[player_id] = self._glicko.update_rating(
                        player_rating, outcomes
                    )
                else:
                    # For Elo, sum the individual changes
                    total_change = 0.0
                    for outcome in outcomes:
                        expected = self._elo.calculate_expected(
                            player_rating.rating,
                            outcome.opponent_rating.rating
                        )
                        total_change += self._elo.k_factor * (outcome.result - expected)

                    # Average the change across all opponents
                    avg_change = total_change / len(outcomes)

                    new_ratings[player_id] = SkillRating(
                        rating=max(SOCIAL_CONFIG.SkillRating.MIN_RATING, player_rating.rating + avg_change),
                        uncertainty=player_rating.uncertainty,
                        games_played=player_rating.games_played + 1,
                        last_updated=time.time()
                    )

            # Update stored ratings
            for player_id, new_rating in new_ratings.items():
                self._ratings[player_id] = new_rating

            return new_ratings

    def get_leaderboard(
        self,
        limit: Optional[int] = None,
        min_games: Optional[int] = None
    ) -> list[tuple[str, SkillRating]]:
        """
        Get the top players by rating.

        Args:
            limit: Maximum number of players to return.
            min_games: Minimum games required to be on leaderboard.

        Returns:
            List of (player_id, rating) tuples sorted by rating.
        """
        if limit is None:
            limit = SOCIAL_CONFIG.SkillRating.LEADERBOARD_DEFAULT_LIMIT
        if min_games is None:
            min_games = SOCIAL_CONFIG.SkillRating.LEADERBOARD_MIN_GAMES

        with self._lock:
            qualified = [
                (pid, rating) for pid, rating in self._ratings.items()
                if rating.games_played >= min_games
            ]
            qualified.sort(key=lambda x: x[1].rating, reverse=True)
            return qualified[:limit]

    def get_percentile(self, player_id: str) -> float:
        """
        Get a player's percentile ranking (0-100).

        Args:
            player_id: The player's unique identifier.

        Returns:
            Percentile ranking (100 = top player).
        """
        with self._lock:
            if player_id not in self._ratings:
                return 50.0

            player_rating = self._ratings[player_id].rating
            total_players = len(self._ratings)

            if total_players <= 1:
                return 50.0

            players_below = sum(
                1 for rating in self._ratings.values()
                if rating.rating < player_rating
            )

            return (players_below / (total_players - 1)) * 100.0

    def decay_inactive_ratings(self, max_inactive_days: Optional[float] = None) -> int:
        """
        Apply decay to inactive players' uncertainty.

        Args:
            max_inactive_days: Maximum days to apply decay for.

        Returns:
            Number of ratings updated.
        """
        if max_inactive_days is None:
            max_inactive_days = SOCIAL_CONFIG.SkillRating.MAX_INACTIVE_DAYS

        if not self.use_glicko:
            return 0  # Elo doesn't have uncertainty

        with self._lock:
            updated = 0
            current_time = time.time()
            seconds_per_day = SOCIAL_CONFIG.SkillRating.SECONDS_PER_DAY

            for player_id, rating in list(self._ratings.items()):
                days_inactive = (current_time - rating.last_updated) / seconds_per_day

                if days_inactive > 0:
                    days_to_apply = min(days_inactive, max_inactive_days)
                    new_rating = self._glicko.apply_inactivity_decay(
                        rating, days_to_apply
                    )

                    if new_rating.uncertainty != rating.uncertainty:
                        self._ratings[player_id] = new_rating
                        updated += 1

            return updated
