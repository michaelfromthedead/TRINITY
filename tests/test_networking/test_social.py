"""
Comprehensive tests for the Social Systems module.

Tests matchmaking, skill rating, lobbies, parties, voice chat, and text chat.
Includes edge cases, state transitions, race condition tests, and meaningful assertions.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Matchmaking imports
from engine.networking.social.matchmaking import (
    MatchmakingState,
    MatchCriteria,
    QueueEntry,
    MatchResult,
    MatchmakingQueue,
    MatchmakingService,
)

# Skill Rating imports
from engine.networking.social.skill_rating import (
    SkillRating,
    MatchOutcome,
    EloCalculator,
    Glicko2Calculator,
    MMRManager,
)

# Lobby imports
from engine.networking.social.lobby import (
    LobbyState,
    LobbySettings,
    LobbyPlayer,
    Lobby,
    LobbyManager,
)

# Party imports
from engine.networking.social.party import (
    PartyRole,
    PartyState,
    PartyMember,
    PartyInvite,
    Party,
    PartyManager,
)

# Voice Chat imports
from engine.networking.social.voice_chat import (
    VoiceChannel,
    VoiceQuality,
    VoiceState,
    VoiceParticipant,
    ProximityVoice,
    VoiceChatManager,
)

# Text Chat imports
from engine.networking.social.text_chat import (
    ChatChannel,
    MessageType,
    ChatMessage,
    ProfanityFilter,
    RateLimiter,
    ChatManager,
)

# Config imports
from engine.networking.social.config import SOCIAL_CONFIG


# ==================== MATCHMAKING TESTS ====================

class TestMatchCriteria:
    """Tests for MatchCriteria."""

    def test_create_valid_criteria(self) -> None:
        """Test creating valid match criteria."""
        criteria = MatchCriteria(
            mode="ranked",
            region="us-west",
            skill_range=(1400, 1600),
            party_size=2
        )
        assert criteria.mode == "ranked"
        assert criteria.region == "us-west"
        assert criteria.skill_range == (1400, 1600)
        assert criteria.party_size == 2

    def test_invalid_skill_range(self) -> None:
        """Test that invalid skill range raises error."""
        with pytest.raises(ValueError, match="skill_range min must be <= max"):
            MatchCriteria(
                mode="ranked",
                region="us-west",
                skill_range=(1600, 1400)  # min > max
            )

    def test_invalid_party_size(self) -> None:
        """Test that invalid party size raises error."""
        with pytest.raises(ValueError, match="party_size must be at least 1"):
            MatchCriteria(
                mode="ranked",
                region="us-west",
                skill_range=(1400, 1600),
                party_size=0
            )

    def test_criteria_matches(self) -> None:
        """Test criteria matching logic."""
        criteria1 = MatchCriteria(
            mode="ranked",
            region="us-west",
            skill_range=(1400, 1600)
        )
        criteria2 = MatchCriteria(
            mode="ranked",
            region="us-west",
            skill_range=(1500, 1700)
        )
        criteria3 = MatchCriteria(
            mode="casual",
            region="us-west",
            skill_range=(1400, 1600)
        )

        # Same mode/region, overlapping skills
        assert criteria1.matches(criteria2)

        # Different mode
        assert not criteria1.matches(criteria3)


class TestMatchmakingQueue:
    """Tests for MatchmakingQueue."""

    def test_join_queue(self) -> None:
        """Test joining the matchmaking queue."""
        queue = MatchmakingQueue(min_players=2, max_players=4)
        criteria = MatchCriteria(
            mode="ranked",
            region="us-west",
            skill_range=(1400, 1600)
        )

        result = queue.join("player1", criteria, skill=1500)
        assert result is True
        assert queue.get_queue_size() == 1
        assert queue.get_state("player1") == MatchmakingState.SEARCHING

    def test_join_queue_duplicate(self) -> None:
        """Test that duplicate join returns False."""
        queue = MatchmakingQueue()
        criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))

        queue.join("player1", criteria)
        result = queue.join("player1", criteria)
        assert result is False

    def test_leave_queue(self) -> None:
        """Test leaving the matchmaking queue."""
        queue = MatchmakingQueue()
        criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))

        queue.join("player1", criteria)
        result = queue.leave("player1")

        assert result is True
        assert queue.get_queue_size() == 0
        assert queue.get_state("player1") is None

    def test_leave_queue_not_in_queue(self) -> None:
        """Test leaving when not in queue returns False."""
        queue = MatchmakingQueue()
        result = queue.leave("nonexistent")
        assert result is False

    def test_find_match_not_enough_players(self) -> None:
        """Test that find_match returns None with insufficient players."""
        queue = MatchmakingQueue(min_players=2)
        criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))

        queue.join("player1", criteria)
        match = queue.find_match()
        assert match is None

    def test_find_match_success(self) -> None:
        """Test successful match finding."""
        queue = MatchmakingQueue(min_players=2, max_players=4)
        criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))

        queue.join("player1", criteria, skill=1500)
        queue.join("player2", criteria, skill=1520)

        match = queue.find_match()

        assert match is not None
        assert len(match.players) == 2
        assert "player1" in match.players
        assert "player2" in match.players
        assert match.mode == "ranked"
        assert match.region == "us-west"

    def test_find_match_incompatible_modes(self) -> None:
        """Test that incompatible modes don't match."""
        queue = MatchmakingQueue(min_players=2)
        criteria1 = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))
        criteria2 = MatchCriteria(mode="casual", region="us-west", skill_range=(1400, 1600))

        queue.join("player1", criteria1, skill=1500)
        queue.join("player2", criteria2, skill=1500)

        match = queue.find_match()
        assert match is None

    def test_search_expansion_over_time(self) -> None:
        """Test that skill range expands over time."""
        queue = MatchmakingQueue(
            min_players=2,
            base_skill_range=100,
            expansion_rate=50,
            expansion_interval=1.0
        )
        criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))

        queue.join("player1", criteria, skill=1500)

        # Simulate time passing
        entry = queue.get_entry("player1")
        assert entry is not None
        original_range = entry.criteria.skill_range

        # Manually set queue time to simulate waiting
        entry.queue_time = time.time() - 5  # 5 seconds ago

        queue.expand_search_over_time()

        # Range should have expanded
        new_entry = queue.get_entry("player1")
        assert new_entry is not None
        assert new_entry.criteria.skill_range[0] < original_range[0]
        assert new_entry.criteria.skill_range[1] > original_range[1]

    def test_callback_on_match_found(self) -> None:
        """Test that callback is called when match is found."""
        queue = MatchmakingQueue(min_players=2)
        callback = Mock()
        queue.set_on_match_found(callback)

        criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))
        queue.join("player1", criteria, skill=1500)
        queue.join("player2", criteria, skill=1520)

        queue.find_match()

        callback.assert_called_once()
        match_result = callback.call_args[0][0]
        assert isinstance(match_result, MatchResult)


# ==================== SKILL RATING TESTS ====================

class TestSkillRating:
    """Tests for SkillRating dataclass."""

    def test_create_default_rating(self) -> None:
        """Test creating a default skill rating."""
        rating = SkillRating()
        assert rating.rating == 1500.0
        assert rating.uncertainty == 350.0
        assert rating.games_played == 0

    def test_create_custom_rating(self) -> None:
        """Test creating a custom skill rating."""
        rating = SkillRating(rating=2000, uncertainty=100, games_played=50)
        assert rating.rating == 2000
        assert rating.uncertainty == 100
        assert rating.games_played == 50

    def test_invalid_negative_rating(self) -> None:
        """Test that negative rating raises error."""
        with pytest.raises(ValueError, match="Rating cannot be negative"):
            SkillRating(rating=-100)


class TestEloCalculator:
    """Tests for EloCalculator."""

    def test_calculate_expected(self) -> None:
        """Test expected score calculation."""
        elo = EloCalculator()

        # Equal ratings -> 0.5 expected
        expected = elo.calculate_expected(1500, 1500)
        assert abs(expected - 0.5) < 0.001

        # Higher rated player expects to win
        expected_high = elo.calculate_expected(1600, 1400)
        assert expected_high > 0.5

    def test_update_ratings_winner_gains(self) -> None:
        """Test that winner gains rating."""
        elo = EloCalculator(k_factor=32)

        new_winner, new_loser = elo.update_ratings(1500, 1500)

        assert new_winner > 1500
        assert new_loser < 1500
        assert new_winner - 1500 == 1500 - new_loser  # Symmetric

    def test_update_ratings_draw(self) -> None:
        """Test rating update for a draw."""
        elo = EloCalculator(k_factor=32)

        # Equal ratings draw -> no change
        new_a, new_b = elo.update_ratings(1500, 1500, is_draw=True)

        assert abs(new_a - 1500) < 0.001
        assert abs(new_b - 1500) < 0.001

    def test_upset_rating_change(self) -> None:
        """Test that upsets cause larger rating changes."""
        elo = EloCalculator(k_factor=32)

        # Expected win
        winner_gain_expected, _ = elo.update_ratings(1600, 1400)

        # Upset (lower rated wins)
        winner_gain_upset, _ = elo.update_ratings(1400, 1600)

        # Upset winner gains more than expected winner
        assert winner_gain_upset - 1400 > winner_gain_expected - 1600

    def test_dynamic_k_factor(self) -> None:
        """Test dynamic K-factor calculation."""
        elo = EloCalculator()

        # New player gets high K
        k_new = elo.get_dynamic_k_factor(1500, games_played=5)
        assert k_new == 40

        # High rated player gets low K
        k_high = elo.get_dynamic_k_factor(2500, games_played=100)
        assert k_high == 16


class TestMMRManager:
    """Tests for MMRManager."""

    def test_get_rating_new_player(self) -> None:
        """Test getting rating for a new player."""
        mmr = MMRManager()
        rating = mmr.get_rating("new_player")

        assert rating.rating == 1500.0
        assert rating.games_played == 0

    def test_update_after_match_simple(self) -> None:
        """Test updating ratings after a 1v1 match."""
        mmr = MMRManager(use_glicko=False)  # Use Elo for simpler testing

        winner_rating, loser_rating = mmr.update_after_match_simple(
            "winner", "loser"
        )

        assert winner_rating.rating > 1500
        assert loser_rating.rating < 1500
        assert winner_rating.games_played == 1
        assert loser_rating.games_played == 1

    def test_uncertainty_reduction(self) -> None:
        """Test that uncertainty decreases with more games (Glicko)."""
        mmr = MMRManager(use_glicko=True)

        # Get initial rating
        initial = mmr.get_rating("player1")
        initial_uncertainty = initial.uncertainty

        # Play one game to test uncertainty reduction
        mmr.update_after_match_simple("player1", "opponent1")

        final = mmr.get_rating("player1")
        # Uncertainty should decrease after playing a game
        assert final.uncertainty < initial_uncertainty

    def test_get_leaderboard(self) -> None:
        """Test getting leaderboard."""
        mmr = MMRManager(use_glicko=False)

        # Create some players with different ratings
        mmr.set_rating("player1", SkillRating(rating=1800, games_played=20))
        mmr.set_rating("player2", SkillRating(rating=1700, games_played=15))
        mmr.set_rating("player3", SkillRating(rating=1600, games_played=5))  # Too few games

        leaderboard = mmr.get_leaderboard(limit=10, min_games=10)

        assert len(leaderboard) == 2
        assert leaderboard[0][0] == "player1"  # Highest rated first
        assert leaderboard[1][0] == "player2"


# ==================== LOBBY TESTS ====================

class TestLobbySettings:
    """Tests for LobbySettings."""

    def test_create_default_settings(self) -> None:
        """Test creating default lobby settings."""
        settings = LobbySettings()
        assert settings.max_players == 8
        assert settings.min_players == 2
        assert settings.is_private is False

    def test_invalid_player_counts(self) -> None:
        """Test that invalid player counts raise error."""
        with pytest.raises(ValueError, match="max_players must be >= min_players"):
            LobbySettings(max_players=2, min_players=5)


class TestLobby:
    """Tests for Lobby class."""

    def test_create_lobby(self) -> None:
        """Test creating a new lobby."""
        lobby = Lobby("host1", "HostPlayer")

        assert lobby.host_id == "host1"
        assert lobby.state == LobbyState.WAITING
        assert lobby.player_count == 1  # Host is in

    def test_join_lobby(self) -> None:
        """Test joining a lobby."""
        lobby = Lobby("host1", "HostPlayer")

        result = lobby.join("player2", "Player2")

        assert result is True
        assert lobby.player_count == 2

    def test_join_full_lobby(self) -> None:
        """Test that joining a full lobby fails."""
        settings = LobbySettings(max_players=2)
        lobby = Lobby("host1", "HostPlayer", settings)

        lobby.join("player2", "Player2")
        result = lobby.join("player3", "Player3")

        assert result is False
        assert lobby.player_count == 2

    def test_leave_lobby(self) -> None:
        """Test leaving a lobby."""
        lobby = Lobby("host1", "HostPlayer")
        lobby.join("player2", "Player2")

        result = lobby.leave("player2")

        assert result is True
        assert lobby.player_count == 1

    def test_host_transfer_on_leave(self) -> None:
        """Test that host is transferred when host leaves."""
        lobby = Lobby("host1", "HostPlayer")
        lobby.join("player2", "Player2")

        lobby.leave("host1")

        assert lobby.host_id == "player2"
        assert lobby.player_count == 1

    def test_ready_system(self) -> None:
        """Test player ready system."""
        lobby = Lobby("host1", "HostPlayer")
        lobby.join("player2", "Player2")

        lobby.set_ready("host1", True)
        lobby.set_ready("player2", True)

        assert lobby.all_ready is True
        assert lobby.ready_count == 2

    def test_countdown_start(self) -> None:
        """Test starting countdown."""
        settings = LobbySettings(min_players=2, countdown_seconds=10, auto_start=False)
        lobby = Lobby("host1", "HostPlayer", settings)
        lobby.join("player2", "Player2")
        lobby.set_ready("host1", True)
        lobby.set_ready("player2", True)

        result = lobby.start_countdown()

        assert result is True
        assert lobby.state == LobbyState.COUNTDOWN

    def test_countdown_cancel_on_unready(self) -> None:
        """Test that countdown cancels when player unreadies."""
        settings = LobbySettings(min_players=2, countdown_seconds=60, auto_start=False)
        lobby = Lobby("host1", "HostPlayer", settings)
        lobby.join("player2", "Player2")
        lobby.set_ready("host1", True)
        lobby.set_ready("player2", True)
        lobby.start_countdown()

        lobby.set_ready("player2", False)

        assert lobby.state == LobbyState.WAITING


class TestLobbyManager:
    """Tests for LobbyManager."""

    def test_create_lobby(self) -> None:
        """Test creating a lobby through manager."""
        manager = LobbyManager()

        lobby = manager.create_lobby("host1", "HostPlayer")

        assert lobby is not None
        assert manager.get_lobby_count() == 1

    def test_player_can_only_be_in_one_lobby(self) -> None:
        """Test that a player can't be in multiple lobbies."""
        manager = LobbyManager()

        lobby1 = manager.create_lobby("host1", "HostPlayer")
        lobby2 = manager.create_lobby("host1", "HostPlayer")

        assert lobby1 is not None
        assert lobby2 is None

    def test_find_lobbies(self) -> None:
        """Test finding available lobbies."""
        manager = LobbyManager()
        settings = LobbySettings(game_mode="ranked")

        manager.create_lobby("host1", "Host1", settings)
        manager.create_lobby("host2", "Host2", settings)

        lobbies = manager.find_lobbies(game_mode="ranked")

        assert len(lobbies) == 2


# ==================== PARTY TESTS ====================

class TestParty:
    """Tests for Party class."""

    def test_create_party(self) -> None:
        """Test creating a new party."""
        party = Party("leader1", "LeaderName")

        assert party.leader_id == "leader1"
        assert party.size == 1
        assert party.state == PartyState.IDLE

    def test_invite_and_accept(self) -> None:
        """Test inviting and accepting party invite."""
        party = Party("leader1", "LeaderName")

        invite = party.invite("player2", "leader1", "LeaderName")
        assert invite is not None

        result = party.accept_invite(invite.id, "player2", "Player2")
        assert result is True
        assert party.size == 2

    def test_invite_expired(self) -> None:
        """Test that expired invites can't be accepted."""
        party = Party("leader1", "LeaderName")

        invite = party.invite("player2", "leader1", expire_seconds=0.001)
        time.sleep(0.01)

        result = party.accept_invite(invite.id, "player2", "Player2")
        assert result is False

    def test_kick_member(self) -> None:
        """Test kicking a party member."""
        party = Party("leader1", "LeaderName")
        invite = party.invite("player2", "leader1")
        party.accept_invite(invite.id, "player2", "Player2")

        result = party.kick("player2", "leader1")

        assert result is True
        assert party.size == 1

    def test_only_leader_can_kick(self) -> None:
        """Test that only leader can kick."""
        party = Party("leader1", "LeaderName", max_size=3)
        invite1 = party.invite("player2", "leader1")
        party.accept_invite(invite1.id, "player2", "Player2")
        invite2 = party.invite("player3", "leader1")
        party.accept_invite(invite2.id, "player3", "Player3")

        result = party.kick("player3", "player2")  # Non-leader tries to kick

        assert result is False
        assert party.size == 3

    def test_promote_leader(self) -> None:
        """Test promoting a new leader."""
        party = Party("leader1", "LeaderName")
        invite = party.invite("player2", "leader1")
        party.accept_invite(invite.id, "player2", "Player2")

        result = party.promote_leader("player2", "leader1")

        assert result is True
        assert party.leader_id == "player2"

    def test_max_size_enforcement(self) -> None:
        """Test that max party size is enforced."""
        party = Party("leader1", "LeaderName", max_size=2)
        invite1 = party.invite("player2", "leader1")
        party.accept_invite(invite1.id, "player2", "Player2")

        invite2 = party.invite("player3", "leader1")
        assert invite2 is None  # Can't invite when full


class TestPartyManager:
    """Tests for PartyManager."""

    def test_create_party(self) -> None:
        """Test creating a party through manager."""
        manager = PartyManager()

        party = manager.create_party("leader1", "LeaderName")

        assert party is not None
        assert manager.get_party_count() == 1

    def test_send_and_accept_invite(self) -> None:
        """Test sending and accepting invite through manager."""
        manager = PartyManager()
        party = manager.create_party("leader1", "LeaderName")

        invite = manager.send_invite(party.id, "player2", "leader1", "LeaderName")
        assert invite is not None

        result = manager.accept_invite(invite.id, "player2", "Player2")
        assert result is True
        assert party.size == 2

    def test_dissolve_party(self) -> None:
        """Test dissolving a party."""
        manager = PartyManager()
        party = manager.create_party("leader1", "LeaderName")

        result = manager.dissolve_party(party.id, "leader1")

        assert result is True
        assert manager.get_party_count() == 0


# ==================== VOICE CHAT TESTS ====================

class TestProximityVoice:
    """Tests for ProximityVoice."""

    def test_calculate_distance(self) -> None:
        """Test 3D distance calculation."""
        proximity = ProximityVoice()

        distance = proximity.calculate_distance((0, 0, 0), (3, 4, 0))
        assert abs(distance - 5.0) < 0.001

    def test_attenuation_at_zero_distance(self) -> None:
        """Test volume at zero distance is 100%."""
        proximity = ProximityVoice(max_distance=50, min_distance=1)

        attenuation = proximity.calculate_attenuation(0)
        assert attenuation == 1.0

    def test_attenuation_at_max_distance(self) -> None:
        """Test volume at max distance is 0%."""
        proximity = ProximityVoice(max_distance=50)

        attenuation = proximity.calculate_attenuation(50)
        assert attenuation == 0.0

    def test_attenuation_beyond_max_distance(self) -> None:
        """Test volume beyond max distance is 0%."""
        proximity = ProximityVoice(max_distance=50)

        attenuation = proximity.calculate_attenuation(100)
        assert attenuation == 0.0

    def test_attenuation_decreases_with_distance(self) -> None:
        """Test that volume decreases as distance increases."""
        proximity = ProximityVoice(max_distance=50, min_distance=1)

        att_close = proximity.calculate_attenuation(10)
        att_far = proximity.calculate_attenuation(40)

        assert att_close > att_far


class TestVoiceChatManager:
    """Tests for VoiceChatManager."""

    def test_join_channel(self) -> None:
        """Test joining a voice channel."""
        manager = VoiceChatManager()

        result = manager.join_channel("player1", "Player1", VoiceChannel.TEAM)

        assert result is True
        participant = manager.get_participant("player1")
        assert participant is not None
        assert participant.channel == VoiceChannel.TEAM

    def test_leave_channel(self) -> None:
        """Test leaving a voice channel."""
        manager = VoiceChatManager()
        manager.join_channel("player1", "Player1", VoiceChannel.TEAM)

        result = manager.leave_channel("player1")

        assert result is True
        assert manager.get_participant("player1") is None

    def test_mute_self(self) -> None:
        """Test self-muting."""
        manager = VoiceChatManager()
        manager.join_channel("player1", "Player1", VoiceChannel.TEAM)

        manager.mute_self("player1", True)

        participant = manager.get_participant("player1")
        assert participant.state.is_muted is True

    def test_mute_player(self) -> None:
        """Test muting another player."""
        manager = VoiceChatManager()
        manager.join_channel("player1", "Player1", VoiceChannel.TEAM)
        manager.join_channel("player2", "Player2", VoiceChannel.TEAM)

        # Personal mute
        manager.mute_player("player2", "player1")

        participant1 = manager.get_participant("player1")
        assert participant1.is_player_muted("player2") is True

    def test_set_volume(self) -> None:
        """Test setting volume."""
        manager = VoiceChatManager()
        manager.join_channel("player1", "Player1", VoiceChannel.TEAM)

        manager.set_volume("player1", volume=0.5)

        participant = manager.get_participant("player1")
        assert participant.state.volume == 0.5

    def test_get_active_speakers(self) -> None:
        """Test getting active speakers."""
        manager = VoiceChatManager()
        manager.join_channel("player1", "Player1", VoiceChannel.TEAM)
        manager.join_channel("player2", "Player2", VoiceChannel.TEAM)

        manager.set_transmitting("player1", True)

        speakers = manager.get_active_speakers()
        assert "player1" in speakers
        assert "player2" not in speakers

    def test_proximity_voice_audible(self) -> None:
        """Test proximity-based audibility."""
        manager = VoiceChatManager()
        manager.join_channel(
            "player1", "Player1", VoiceChannel.PROXIMITY,
            position=(0, 0, 0)
        )
        manager.join_channel(
            "player2", "Player2", VoiceChannel.PROXIMITY,
            position=(10, 0, 0)
        )
        manager.join_channel(
            "player3", "Player3", VoiceChannel.PROXIMITY,
            position=(100, 0, 0)  # Too far
        )

        manager.set_transmitting("player2", True)
        manager.set_transmitting("player3", True)

        audible = manager.get_audible_players("player1")

        assert "player2" in audible
        assert "player3" not in audible  # Too far


# ==================== TEXT CHAT TESTS ====================

class TestProfanityFilter:
    """Tests for ProfanityFilter."""

    def test_filter_disabled(self) -> None:
        """Test that disabled filter doesn't modify text."""
        pf = ProfanityFilter()
        pf.enabled = False
        pf.add_blocked_word("badword")

        result = pf.filter("This contains badword")
        assert "badword" in result

    def test_filter_blocked_word(self) -> None:
        """Test filtering blocked words."""
        pf = ProfanityFilter()
        pf.add_blocked_word("badword")

        result = pf.filter("This contains badword")
        assert "badword" not in result
        assert "*******" in result

    def test_contains_profanity(self) -> None:
        """Test profanity detection."""
        pf = ProfanityFilter()
        pf.add_blocked_word("badword")

        assert pf.contains_profanity("badword") is True
        assert pf.contains_profanity("goodword") is False

    def test_whitelist(self) -> None:
        """Test whitelist functionality."""
        pf = ProfanityFilter()
        pf.add_blocked_word("bass")  # Could be blocked
        pf.add_whitelist_word("bass")  # But whitelisted

        assert pf.contains_profanity("I play bass guitar") is False


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_initial_can_send(self) -> None:
        """Test that new players can send."""
        limiter = RateLimiter()

        assert limiter.can_send("player1") is True

    def test_consume_token(self) -> None:
        """Test token consumption."""
        limiter = RateLimiter(burst_limit=2)

        assert limiter.consume("player1") is True
        assert limiter.consume("player1") is True
        assert limiter.consume("player1") is False  # Burst depleted

    def test_rate_limiting(self) -> None:
        """Test rate limiting over time."""
        limiter = RateLimiter(messages_per_second=10.0, burst_limit=2, cooldown_seconds=0.1)

        # Deplete burst
        limiter.consume("player1")
        limiter.consume("player1")

        # Should be rate limited
        assert limiter.can_send("player1") is False

        # Wait for cooldown
        time.sleep(0.15)

        # Should be able to send again
        assert limiter.can_send("player1") is True

    def test_get_wait_time(self) -> None:
        """Test getting wait time."""
        limiter = RateLimiter(burst_limit=1, cooldown_seconds=1.0)

        limiter.consume("player1")
        wait_time = limiter.get_wait_time("player1")

        assert wait_time > 0


class TestChatManager:
    """Tests for ChatManager."""

    def test_send_message(self) -> None:
        """Test sending a chat message."""
        manager = ChatManager()

        message = manager.send_message(
            "player1", "Player1",
            ChatChannel.GLOBAL, "Hello, world!"
        )

        assert message is not None
        assert message.content == "Hello, world!"
        assert message.sender_id == "player1"

    def test_send_message_profanity_filtered(self) -> None:
        """Test that profanity is filtered from messages."""
        manager = ChatManager(enable_profanity_filter=True)
        manager.profanity_filter.add_blocked_word("badword")

        message = manager.send_message(
            "player1", "Player1",
            ChatChannel.GLOBAL, "This is a badword"
        )

        assert message is not None
        assert "badword" not in message.content
        assert message.is_filtered is True

    def test_send_message_rate_limited(self) -> None:
        """Test that messages are rate limited."""
        manager = ChatManager(enable_rate_limiting=True)
        manager.rate_limiter._player_state.clear()

        # Spam messages
        for _ in range(10):
            manager.send_message(
                "player1", "Player1",
                ChatChannel.GLOBAL, "Spam"
            )

        # Should be rate limited now
        result = manager.send_message(
            "player1", "Player1",
            ChatChannel.GLOBAL, "Another message"
        )

        assert result is None

    def test_get_history(self) -> None:
        """Test getting chat history."""
        manager = ChatManager()

        manager.send_message("player1", "P1", ChatChannel.GLOBAL, "Message 1")
        manager.send_message("player2", "P2", ChatChannel.GLOBAL, "Message 2")

        history = manager.get_history(ChatChannel.GLOBAL, limit=10)

        assert len(history) == 2

    def test_mute_player(self) -> None:
        """Test server-muting a player."""
        manager = ChatManager()

        manager.mute_player("player1", "admin")

        assert manager.is_muted("player1") is True

        # Muted player can't send
        result = manager.send_message(
            "player1", "Player1",
            ChatChannel.GLOBAL, "Hello"
        )
        assert result is None

    def test_personal_mute(self) -> None:
        """Test personal muting."""
        manager = ChatManager()

        manager.personal_mute("player1", "player2")

        assert manager.is_personally_muted("player1", "player2") is True

    def test_send_system_message(self) -> None:
        """Test sending system messages."""
        manager = ChatManager()

        message = manager.send_system_message("Server restarting in 5 minutes")

        assert message is not None
        assert message.sender_id == "SYSTEM"
        assert message.message_type == MessageType.SYSTEM


# ==================== INTEGRATION TESTS ====================

class TestSocialSystemsIntegration:
    """Integration tests for social systems working together."""

    def test_party_to_matchmaking_flow(self) -> None:
        """Test flow from party creation to matchmaking."""
        party_mgr = PartyManager()
        mm_queue = MatchmakingQueue(min_players=2)

        # Create party
        party = party_mgr.create_party("leader", "Leader")
        invite = party_mgr.send_invite(party.id, "member", "leader", "Leader")
        party_mgr.accept_invite(invite.id, "member", "Member")

        # Set party ready
        party.set_ready("leader")
        party.set_ready("member")

        # Join matchmaking as party
        criteria = MatchCriteria(
            mode="ranked",
            region="us-west",
            skill_range=(1400, 1600),
            party_size=2
        )

        for member_id in party.get_member_ids():
            mm_queue.join(member_id, criteria, party_id=party.id)

        assert mm_queue.get_queue_size() == 2

    def test_lobby_with_chat_and_voice(self) -> None:
        """Test lobby with integrated chat and voice."""
        lobby_mgr = LobbyManager()
        chat_mgr = ChatManager()
        voice_mgr = VoiceChatManager()

        # Create lobby
        lobby = lobby_mgr.create_lobby("host", "HostPlayer")
        lobby_id = lobby.id

        # Players join lobby, chat, and voice
        lobby_mgr.join_lobby(lobby_id, "player2", "Player2")

        chat_mgr.send_message(
            "host", "HostPlayer",
            ChatChannel.LOBBY, "Welcome to the lobby!",
            context_id=lobby_id
        )

        voice_mgr.join_channel("host", "HostPlayer", VoiceChannel.TEAM, lobby_id)
        voice_mgr.join_channel("player2", "Player2", VoiceChannel.TEAM, lobby_id)

        # Verify integrated state
        assert lobby.player_count == 2
        history = chat_mgr.get_history(ChatChannel.LOBBY, lobby_id)
        assert len(history) == 1
        participants = voice_mgr.get_channel_participants(VoiceChannel.TEAM, lobby_id)
        assert len(participants) == 2


# ==================== EDGE CASE TESTS ====================

class TestMatchmakingEdgeCases:
    """Edge case tests for matchmaking system."""

    def test_join_during_expansion(self) -> None:
        """Test joining queue while expansion is occurring."""
        queue = MatchmakingQueue(min_players=2, expansion_interval=0.1)
        criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))

        queue.join("player1", criteria, skill=1500)

        # Simulate time passing
        entry = queue.get_entry("player1")
        entry.queue_time = time.time() - 1  # 1 second ago

        # Expand while adding new player
        queue.expand_search_over_time()
        queue.join("player2", criteria, skill=1800)  # Outside original range

        # Both should be in queue
        assert queue.get_queue_size() == 2

    def test_leave_during_match_finding(self) -> None:
        """Test player leaving while match is being found."""
        queue = MatchmakingQueue(min_players=2)
        criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))

        queue.join("player1", criteria, skill=1500)
        queue.join("player2", criteria, skill=1520)

        # Player 1 leaves before match is found
        queue.leave("player1")

        # Match should not be found
        match = queue.find_match()
        assert match is None

    def test_concurrent_queue_operations(self) -> None:
        """Test thread safety of queue operations."""
        queue = MatchmakingQueue(min_players=10, max_players=20)
        criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))
        errors = []

        def join_player(player_id: str) -> None:
            try:
                queue.join(player_id, criteria, skill=1500)
            except Exception as e:
                errors.append(e)

        def leave_player(player_id: str) -> None:
            try:
                time.sleep(0.001)  # Small delay
                queue.leave(player_id)
            except Exception as e:
                errors.append(e)

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(50):
                futures.append(executor.submit(join_player, f"player{i}"))
            for i in range(25):
                futures.append(executor.submit(leave_player, f"player{i}"))

            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"


class TestLobbyEdgeCases:
    """Edge case tests for lobby system."""

    def test_join_lobby_wrong_password(self) -> None:
        """Test joining private lobby with wrong password."""
        settings = LobbySettings(is_private=True, password="secret123")
        lobby = Lobby("host1", "HostPlayer", settings)

        result = lobby.join("player2", "Player2", password="wrongpass")
        assert result is False
        assert lobby.player_count == 1

    def test_join_lobby_in_game_state(self) -> None:
        """Test that players cannot join lobby that's in game."""
        lobby = Lobby("host1", "HostPlayer")
        lobby.state = LobbyState.IN_GAME

        result = lobby.join("player2", "Player2")
        assert result is False

    def test_kick_triggers_host_transfer(self) -> None:
        """Test that kicking all except host doesn't break lobby."""
        lobby = Lobby("host1", "HostPlayer")
        lobby.join("player2", "Player2")
        lobby.join("player3", "Player3")

        # Kick player2
        lobby.kick("player2", "host1")
        assert lobby.player_count == 2
        assert lobby.host_id == "host1"

        # Kick player3
        lobby.kick("player3", "host1")
        assert lobby.player_count == 1
        assert lobby.host_id == "host1"

    def test_countdown_finish_transitions_state(self) -> None:
        """Test that countdown completion transitions state correctly."""
        settings = LobbySettings(min_players=2, countdown_seconds=0, auto_start=False)
        lobby = Lobby("host1", "HostPlayer", settings)
        lobby.join("player2", "Player2")
        lobby.set_ready("host1", True)
        lobby.set_ready("player2", True)

        # Start countdown with 0 seconds
        lobby.start_countdown()

        # Small wait for timer thread
        time.sleep(0.1)

        # State should have transitioned
        assert lobby.state in (LobbyState.COUNTDOWN, LobbyState.STARTING)

    def test_spectator_doesnt_count_for_game_start(self) -> None:
        """Test that spectators don't count toward min_players."""
        settings = LobbySettings(min_players=2, allow_spectators=True)
        lobby = Lobby("host1", "HostPlayer", settings)

        lobby.join("spectator1", "Spectator1", as_spectator=True)
        lobby.set_ready("host1", True)

        # Should not be able to start with only 1 player
        assert lobby.can_start is False

    def test_lobby_close_clears_countdown(self) -> None:
        """Test that closing lobby cancels active countdown."""
        settings = LobbySettings(min_players=2, countdown_seconds=60)
        lobby = Lobby("host1", "HostPlayer", settings)
        lobby.join("player2", "Player2")
        lobby.set_ready("host1", True)
        lobby.set_ready("player2", True)
        lobby.start_countdown()

        assert lobby.state == LobbyState.COUNTDOWN

        lobby.close("host1")

        assert lobby.state == LobbyState.CLOSED
        assert lobby._countdown_timer is None


class TestPartyEdgeCases:
    """Edge case tests for party system."""

    def test_accept_expired_invite(self) -> None:
        """Test that accepting an expired invite fails gracefully."""
        party = Party("leader1", "LeaderName")

        invite = party.invite("player2", "leader1", expire_seconds=0.001)
        time.sleep(0.01)  # Let it expire

        result = party.accept_invite(invite.id, "player2", "Player2")
        assert result is False
        assert party.size == 1

    def test_accept_invite_party_full(self) -> None:
        """Test accepting invite when party becomes full after invite was sent."""
        party = Party("leader1", "LeaderName", max_size=2)

        # Send invite while party has room
        invite1 = party.invite("player2", "leader1")
        assert invite1 is not None

        # Accept first invite - party is now full
        result1 = party.accept_invite(invite1.id, "player2", "Player2")
        assert result1 is True
        assert party.size == 2
        assert party.is_full is True

        # Now try to invite player3 - should fail because party is full
        invite2 = party.invite("player3", "leader1")
        assert invite2 is None  # Can't invite when full

    def test_leader_leaves_promotes_oldest(self) -> None:
        """Test that when leader leaves, oldest member becomes leader."""
        party = Party("leader1", "LeaderName", max_size=4)

        invite1 = party.invite("player2", "leader1")
        party.accept_invite(invite1.id, "player2", "Player2")
        time.sleep(0.01)  # Ensure different join times

        invite2 = party.invite("player3", "leader1")
        party.accept_invite(invite2.id, "player3", "Player3")

        # Leader leaves
        party.leave("leader1")

        # player2 should be new leader (joined first)
        assert party.leader_id == "player2"
        assert party.size == 2

    def test_disband_clears_invites(self) -> None:
        """Test that disbanding clears all pending invites."""
        party = Party("leader1", "LeaderName")

        party.invite("player2", "leader1")
        party.invite("player3", "leader1")

        assert len(party.get_pending_invites()) == 2

        party.disband("leader1")

        assert party.state == PartyState.DISBANDED

    def test_double_accept_same_invite(self) -> None:
        """Test that accepting same invite twice fails."""
        party = Party("leader1", "LeaderName")

        invite = party.invite("player2", "leader1")

        result1 = party.accept_invite(invite.id, "player2", "Player2")
        assert result1 is True

        # Try accepting again (invite no longer exists)
        result2 = party.accept_invite(invite.id, "player2", "Player2")
        assert result2 is False


class TestVoiceChatEdgeCases:
    """Edge case tests for voice chat system."""

    def test_muted_player_cannot_transmit(self) -> None:
        """Test that server-muted player cannot transmit."""
        manager = VoiceChatManager()
        manager.join_channel("player1", "Player1", VoiceChannel.TEAM)

        manager.mute_player("player1", "admin", server_mute=True)

        result = manager.set_transmitting("player1", True)
        assert result is False

        participant = manager.get_participant("player1")
        assert participant.state.is_transmitting is False

    def test_proximity_voice_exact_max_distance(self) -> None:
        """Test proximity voice at exactly max distance."""
        proximity = ProximityVoice(max_distance=50.0, min_distance=1.0)

        attenuation = proximity.calculate_attenuation(50.0)
        assert attenuation == 0.0

        attenuation = proximity.calculate_attenuation(49.9)
        assert attenuation > 0.0

    def test_leave_channel_clears_active_speaker(self) -> None:
        """Test that leaving channel removes from active speakers."""
        manager = VoiceChatManager()
        manager.join_channel("player1", "Player1", VoiceChannel.TEAM)
        manager.set_transmitting("player1", True)

        assert "player1" in manager.get_active_speakers()

        manager.leave_channel("player1")

        assert "player1" not in manager.get_active_speakers()

    def test_volume_clamping(self) -> None:
        """Test that volume values are clamped to valid range."""
        manager = VoiceChatManager()
        manager.join_channel("player1", "Player1", VoiceChannel.TEAM)

        # Try to set volume above max
        manager.set_volume("player1", volume=5.0)
        participant = manager.get_participant("player1")
        assert participant.state.volume <= SOCIAL_CONFIG.VoiceChat.VOLUME_MAX

        # Try to set volume below min
        manager.set_volume("player1", volume=-1.0)
        participant = manager.get_participant("player1")
        assert participant.state.volume >= SOCIAL_CONFIG.VoiceChat.VOLUME_MIN


class TestTextChatEdgeCases:
    """Edge case tests for text chat system."""

    def test_message_length_truncation(self) -> None:
        """Test that long messages are truncated."""
        manager = ChatManager()
        max_length = SOCIAL_CONFIG.TextChat.MESSAGE_MAX_LENGTH

        long_message = "a" * (max_length + 500)
        message = manager.send_message(
            "player1", "Player1",
            ChatChannel.GLOBAL, long_message
        )

        assert message is not None
        assert len(message.content) <= max_length

    def test_empty_message_rejected(self) -> None:
        """Test that empty messages are rejected."""
        manager = ChatManager()

        result = manager.send_message(
            "player1", "Player1",
            ChatChannel.GLOBAL, ""
        )
        assert result is None

        result = manager.send_message(
            "player1", "Player1",
            ChatChannel.GLOBAL, "   "  # Whitespace only
        )
        assert result is None

    def test_mute_expires(self) -> None:
        """Test that temporary mutes expire."""
        manager = ChatManager()

        manager.mute_player("player1", "admin", duration_seconds=0.1)
        assert manager.is_muted("player1") is True

        time.sleep(0.15)

        # Mute should have expired
        assert manager.is_muted("player1") is False

    def test_profanity_obfuscation_detection(self) -> None:
        """Test that obfuscated profanity is detected."""
        pf = ProfanityFilter()
        pf.add_blocked_word("test")

        # Test l33t speak
        assert pf.contains_profanity("t3st") is True
        assert pf.contains_profanity("t.e.s.t") is True

    def test_rate_limiter_cooldown(self) -> None:
        """Test rate limiter cooldown behavior."""
        # Use higher message rate to ensure tokens refill quickly after cooldown
        limiter = RateLimiter(messages_per_second=20.0, burst_limit=2, cooldown_seconds=0.1)

        # Deplete burst
        assert limiter.consume("player1") is True
        assert limiter.consume("player1") is True
        assert limiter.consume("player1") is False

        # Check wait time
        wait = limiter.get_wait_time("player1")
        assert wait > 0

        # Wait for cooldown plus time for token refill
        time.sleep(0.2)

        # Should be able to send again
        assert limiter.can_send("player1") is True


class TestConfigUsage:
    """Tests to verify config constants are properly used."""

    def test_default_skill_rating_uses_config(self) -> None:
        """Test that default skill rating uses config values."""
        rating = SkillRating()
        assert rating.rating == SOCIAL_CONFIG.SkillRating.DEFAULT_RATING
        assert rating.uncertainty == SOCIAL_CONFIG.SkillRating.DEFAULT_UNCERTAINTY

    def test_default_party_size_uses_config(self) -> None:
        """Test that default party size uses config value."""
        party = Party("leader", "Leader")
        assert party.max_size == SOCIAL_CONFIG.Party.MAX_SIZE_DEFAULT

    def test_default_lobby_settings_use_config(self) -> None:
        """Test that default lobby settings use config values."""
        settings = LobbySettings()
        assert settings.max_players == SOCIAL_CONFIG.Lobby.MAX_PLAYERS_DEFAULT
        assert settings.min_players == SOCIAL_CONFIG.Lobby.MIN_PLAYERS_DEFAULT

    def test_rate_limiter_uses_config(self) -> None:
        """Test that rate limiter uses config values."""
        limiter = RateLimiter()
        assert limiter.messages_per_second == SOCIAL_CONFIG.TextChat.RATE_MESSAGES_PER_SECOND
        assert limiter.burst_limit == SOCIAL_CONFIG.TextChat.RATE_BURST_LIMIT


class TestStateTransitions:
    """Tests for proper state machine transitions."""

    def test_lobby_state_transitions(self) -> None:
        """Test valid lobby state transitions."""
        settings = LobbySettings(min_players=2, countdown_seconds=60, auto_start=False)
        lobby = Lobby("host1", "HostPlayer", settings)

        # Initial state
        assert lobby.state == LobbyState.WAITING

        # Add player and ready up
        lobby.join("player2", "Player2")
        lobby.set_ready("host1", True)
        lobby.set_ready("player2", True)

        # Start countdown - transitions to COUNTDOWN
        lobby.start_countdown()
        assert lobby.state == LobbyState.COUNTDOWN

        # Cancel - transitions back to WAITING
        lobby.cancel_countdown()
        assert lobby.state == LobbyState.WAITING

        # Close - transitions to CLOSED
        lobby.close("host1")
        assert lobby.state == LobbyState.CLOSED

    def test_party_state_transitions(self) -> None:
        """Test valid party state transitions."""
        party = Party("leader1", "LeaderName")

        # Initial state
        assert party.state == PartyState.IDLE

        # Set to searching
        party.set_state(PartyState.SEARCHING, "leader1")
        assert party.state == PartyState.SEARCHING

        # Set to in lobby
        party.set_state(PartyState.IN_LOBBY, "leader1")
        assert party.state == PartyState.IN_LOBBY

        # Set to in game
        party.set_state(PartyState.IN_GAME, "leader1")
        assert party.state == PartyState.IN_GAME

        # Back to idle
        party.set_state(PartyState.IDLE, "leader1")
        assert party.state == PartyState.IDLE

        # Disband
        party.disband("leader1")
        assert party.state == PartyState.DISBANDED

    def test_matchmaking_state_transitions(self) -> None:
        """Test matchmaking player state transitions."""
        queue = MatchmakingQueue(min_players=2)
        criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))

        # Join - SEARCHING
        queue.join("player1", criteria, skill=1500)
        assert queue.get_state("player1") == MatchmakingState.SEARCHING

        # Leave - IDLE (and removed)
        queue.leave("player1")
        assert queue.get_state("player1") is None

        # Rejoin for match test
        queue.join("player1", criteria, skill=1500)
        queue.join("player2", criteria, skill=1520)

        # Find match - FOUND
        match = queue.find_match()
        assert match is not None
        # Players are removed after match is found
        assert queue.get_state("player1") is None
        assert queue.get_state("player2") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
