"""
Tests for Trinity Pattern - Tier 39: SOCIAL Decorators
"""

import pytest

from trinity.decorators.registry import Tier, registry
from trinity.decorators.social import (
    VALID_LEADERBOARD_SORT,
    VALID_LEADERBOARD_UPDATE,
    VALID_PRESENCE_DETAIL,
    leaderboard,
    presence,
    shareable,
    social,
)


# =============================================================================
# @social tests
# =============================================================================


def test_social_basic():
    """Test basic @social application."""

    @social(platform="twitter")
    class TwitterIntegration:
        pass

    assert hasattr(TwitterIntegration, "_social")
    assert TwitterIntegration._social is True
    assert TwitterIntegration._social_platform == "twitter"
    assert "social" in TwitterIntegration._applied_decorators


def test_social_different_platforms():
    """Test @social with different platforms."""

    @social(platform="facebook")
    class FacebookIntegration:
        pass

    @social(platform="discord")
    class DiscordIntegration:
        pass

    assert FacebookIntegration._social_platform == "facebook"
    assert DiscordIntegration._social_platform == "discord"


def test_social_validation_empty_platform():
    """Test @social validation rejects empty platform."""
    with pytest.raises(ValueError, match="'platform' parameter is required"):

        @social(platform="")
        class BadSocial:
            pass


def test_social_tags():
    """Test @social sets correct tags."""

    @social(platform="steam")
    class SteamIntegration:
        pass

    tags = SteamIntegration._tags
    assert tags["social"] is True
    assert tags["social_platform"] == "steam"


def test_social_registry():
    """Test @social registers correctly."""

    @social(platform="test")
    class TestSocial:
        pass

    assert "social" in TestSocial._registries
    spec = registry.get("social")
    assert spec is not None
    assert spec.tier == Tier.SOCIAL


# =============================================================================
# @leaderboard tests
# =============================================================================


def test_leaderboard_basic():
    """Test basic @leaderboard application."""

    @leaderboard(id="high_scores")
    class HighScores:
        pass

    assert hasattr(HighScores, "_leaderboard")
    assert HighScores._leaderboard is True
    assert HighScores._leaderboard_id == "high_scores"
    assert HighScores._leaderboard_sort == "descending"
    assert HighScores._leaderboard_update_frequency == "immediate"
    assert "leaderboard" in HighScores._applied_decorators


def test_leaderboard_ascending():
    """Test @leaderboard with ascending sort."""

    @leaderboard(id="fastest_times", sort="ascending")
    class FastestTimes:
        pass

    assert FastestTimes._leaderboard is True
    assert FastestTimes._leaderboard_sort == "ascending"


def test_leaderboard_daily_update():
    """Test @leaderboard with daily update frequency."""

    @leaderboard(id="daily_scores", update_frequency="daily")
    class DailyScores:
        pass

    assert DailyScores._leaderboard is True
    assert DailyScores._leaderboard_update_frequency == "daily"


def test_leaderboard_weekly_update():
    """Test @leaderboard with weekly update frequency."""

    @leaderboard(id="weekly_challenge", update_frequency="weekly")
    class WeeklyChallenge:
        pass

    assert WeeklyChallenge._leaderboard is True
    assert WeeklyChallenge._leaderboard_update_frequency == "weekly"


def test_leaderboard_validation_empty_id():
    """Test @leaderboard validation rejects empty id."""
    with pytest.raises(ValueError, match="'id' parameter is required"):

        @leaderboard(id="")
        class BadLeaderboard:
            pass


def test_leaderboard_validation_invalid_sort():
    """Test @leaderboard validation rejects invalid sort."""
    with pytest.raises(ValueError, match="invalid sort"):

        @leaderboard(id="test", sort="invalid")
        class BadLeaderboard:
            pass


def test_leaderboard_validation_invalid_update():
    """Test @leaderboard validation rejects invalid update_frequency."""
    with pytest.raises(ValueError, match="invalid update_frequency"):

        @leaderboard(id="test", update_frequency="hourly")
        class BadLeaderboard:
            pass


def test_leaderboard_tags():
    """Test @leaderboard sets correct tags."""

    @leaderboard(id="pvp_ranks", sort="descending", update_frequency="immediate")
    class PVPRanks:
        pass

    tags = PVPRanks._tags
    assert tags["leaderboard"] is True
    assert tags["leaderboard_id"] == "pvp_ranks"
    assert tags["leaderboard_sort"] == "descending"
    assert tags["leaderboard_update_frequency"] == "immediate"


def test_leaderboard_registry():
    """Test @leaderboard registers correctly."""

    @leaderboard(id="test")
    class TestLeaderboard:
        pass

    assert "social" in TestLeaderboard._registries
    spec = registry.get("leaderboard")
    assert spec is not None
    assert spec.tier == Tier.SOCIAL


# =============================================================================
# @shareable tests
# =============================================================================


def test_shareable_basic():
    """Test basic @shareable application with defaults."""

    @shareable()
    class ShareableContent:
        pass

    assert hasattr(ShareableContent, "_shareable")
    assert ShareableContent._shareable is True
    assert ShareableContent._shareable_platforms == frozenset(
        {"twitter", "facebook", "clipboard"}
    )
    assert "shareable" in ShareableContent._applied_decorators


def test_shareable_custom_platforms():
    """Test @shareable with custom platforms."""

    @shareable(platforms={"discord", "reddit"})
    class CustomShare:
        pass

    assert CustomShare._shareable is True
    assert CustomShare._shareable_platforms == frozenset({"discord", "reddit"})


def test_shareable_single_platform():
    """Test @shareable with single platform."""

    @shareable(platforms={"instagram"})
    class InstagramOnly:
        pass

    assert InstagramOnly._shareable_platforms == frozenset({"instagram"})


def test_shareable_function():
    """Test @shareable on function."""

    @shareable(platforms={"twitter"})
    def share_score():
        pass

    assert share_score._shareable is True
    assert share_score._shareable_platforms == frozenset({"twitter"})


def test_shareable_validation_empty_platforms():
    """Test @shareable validation rejects empty platforms."""
    with pytest.raises(ValueError, match="'platforms' must be a non-empty set"):

        @shareable(platforms=set())
        class BadShareable:
            pass


def test_shareable_tags():
    """Test @shareable sets correct tags."""

    @shareable(platforms={"twitter", "discord"})
    class MultiShare:
        pass

    tags = MultiShare._tags
    assert tags["shareable"] is True
    assert tags["shareable_platforms"] == frozenset({"twitter", "discord"})


def test_shareable_registry():
    """Test @shareable registers correctly."""

    @shareable()
    class TestShareable:
        pass

    assert "social" in TestShareable._registries
    spec = registry.get("shareable")
    assert spec is not None
    assert spec.tier == Tier.SOCIAL


def test_shareable_platforms_immutability():
    """Test that platforms are stored as frozenset."""

    @shareable(platforms={"twitter", "facebook"})
    class ImmutableShare:
        pass

    # Should be frozenset (immutable)
    assert isinstance(ImmutableShare._shareable_platforms, frozenset)


# =============================================================================
# @presence tests
# =============================================================================


def test_presence_basic():
    """Test basic @presence application."""

    @presence()
    class OnlineStatus:
        pass

    assert hasattr(OnlineStatus, "_presence")
    assert OnlineStatus._presence is True
    assert OnlineStatus._presence_detail_level == "detailed"
    assert "presence" in OnlineStatus._applied_decorators


def test_presence_minimal():
    """Test @presence with minimal detail level."""

    @presence(detail_level="minimal")
    class MinimalPresence:
        pass

    assert MinimalPresence._presence is True
    assert MinimalPresence._presence_detail_level == "minimal"


def test_presence_rich():
    """Test @presence with rich detail level."""

    @presence(detail_level="rich")
    class RichPresence:
        pass

    assert RichPresence._presence is True
    assert RichPresence._presence_detail_level == "rich"


def test_presence_validation_invalid_detail():
    """Test @presence validation rejects invalid detail_level."""
    with pytest.raises(ValueError, match="invalid detail_level"):

        @presence(detail_level="super_detailed")
        class BadPresence:
            pass


def test_presence_tags():
    """Test @presence sets correct tags."""

    @presence(detail_level="rich")
    class GamePresence:
        pass

    tags = GamePresence._tags
    assert tags["presence"] is True
    assert tags["presence_detail_level"] == "rich"


def test_presence_registry():
    """Test @presence registers correctly."""

    @presence()
    class TestPresence:
        pass

    assert "social" in TestPresence._registries
    spec = registry.get("presence")
    assert spec is not None
    assert spec.tier == Tier.SOCIAL


# =============================================================================
# Decorator composition tests
# =============================================================================


def test_social_with_leaderboard():
    """Test composing @social with @leaderboard."""

    @leaderboard(id="twitter_scores")
    @social(platform="twitter")
    class TwitterLeaderboard:
        pass

    assert TwitterLeaderboard._social is True
    assert TwitterLeaderboard._leaderboard is True
    assert "social" in TwitterLeaderboard._applied_decorators
    assert "leaderboard" in TwitterLeaderboard._applied_decorators


def test_shareable_with_presence():
    """Test composing @shareable with @presence."""

    @presence(detail_level="rich")
    @shareable(platforms={"discord"})
    class ShareablePresence:
        pass

    assert ShareablePresence._shareable is True
    assert ShareablePresence._presence is True


def test_multiple_social_decorators():
    """Test stacking multiple social decorators."""

    @presence(detail_level="detailed")
    @shareable(platforms={"twitter", "facebook"})
    @leaderboard(id="global_ranks", sort="descending")
    @social(platform="steam")
    class FullSocial:
        pass

    assert FullSocial._social is True
    assert FullSocial._leaderboard is True
    assert FullSocial._shareable is True
    assert FullSocial._presence is True
    assert len(FullSocial._applied_decorators) == 4


# =============================================================================
# Edge cases
# =============================================================================


def test_leaderboard_all_options():
    """Test @leaderboard with all options specified."""

    @leaderboard(id="ultimate_board", sort="ascending", update_frequency="weekly")
    class CompleteLeaderboard:
        pass

    assert CompleteLeaderboard._leaderboard is True
    assert CompleteLeaderboard._leaderboard_id == "ultimate_board"
    assert CompleteLeaderboard._leaderboard_sort == "ascending"
    assert CompleteLeaderboard._leaderboard_update_frequency == "weekly"


def test_shareable_no_parens():
    """Test @shareable without parentheses (uses defaults)."""

    @shareable
    class DefaultShareable:
        pass

    assert DefaultShareable._shareable is True
    assert "twitter" in DefaultShareable._shareable_platforms


def test_presence_no_parens():
    """Test @presence without parentheses (uses defaults)."""

    @presence
    class DefaultPresence:
        pass

    assert DefaultPresence._presence is True
    assert DefaultPresence._presence_detail_level == "detailed"


# =============================================================================
# Constant validation tests
# =============================================================================


def test_valid_constants_defined():
    """Test that valid constant sets are defined correctly."""
    assert VALID_LEADERBOARD_SORT == frozenset({"ascending", "descending"})
    assert VALID_LEADERBOARD_UPDATE == frozenset({"immediate", "daily", "weekly"})
    assert VALID_PRESENCE_DETAIL == frozenset({"minimal", "detailed", "rich"})


def test_constants_immutable():
    """Test that valid constants are immutable (frozenset)."""
    assert isinstance(VALID_LEADERBOARD_SORT, frozenset)
    assert isinstance(VALID_LEADERBOARD_UPDATE, frozenset)
    assert isinstance(VALID_PRESENCE_DETAIL, frozenset)
