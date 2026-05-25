"""
Tests for Trinity Pattern - Tier 38: ECONOMY Decorators
"""

import pytest

from trinity.decorators.economy import currency, daily_reward, mtx, transaction
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @currency tests
# =============================================================================


def test_currency_basic():
    """Test basic @currency application."""

    @currency(id="gold")
    class GoldCurrency:
        pass

    assert hasattr(GoldCurrency, "_currency")
    assert GoldCurrency._currency is True
    assert GoldCurrency._currency_id == "gold"
    assert GoldCurrency._currency_premium is False
    assert "currency" in GoldCurrency._applied_decorators


def test_currency_with_premium():
    """Test @currency with premium flag."""

    @currency(id="gems", premium=True)
    class Gems:
        pass

    assert Gems._currency is True
    assert Gems._currency_id == "gems"
    assert Gems._currency_premium is True


def test_currency_with_max_value():
    """Test @currency with max_value."""

    @currency(id="coins", max_value=999999)
    class Coins:
        pass

    assert Coins._currency is True
    assert Coins._currency_id == "coins"
    assert Coins._currency_max_value == 999999


def test_currency_validation_empty_id():
    """Test @currency validation rejects empty id."""
    with pytest.raises(ValueError, match="'id' parameter is required"):

        @currency(id="")
        class BadCurrency:
            pass


def test_currency_validation_negative_max():
    """Test @currency validation rejects non-positive max_value."""
    with pytest.raises(ValueError, match="'max_value' must be > 0"):

        @currency(id="test", max_value=0)
        class BadCurrency:
            pass


def test_currency_tags():
    """Test @currency sets correct tags."""

    @currency(id="stars", premium=True, max_value=100)
    class Stars:
        pass

    tags = Stars._tags
    assert tags["currency"] is True
    assert tags["currency_id"] == "stars"
    assert tags["currency_premium"] is True
    assert tags["currency_max_value"] == 100


def test_currency_registry():
    """Test @currency registers correctly."""

    @currency(id="rubies")
    class Rubies:
        pass

    assert "economy" in Rubies._registries
    spec = registry.get("currency")
    assert spec is not None
    assert spec.tier == Tier.ECONOMY


# =============================================================================
# @transaction tests
# =============================================================================


def test_transaction_basic():
    """Test basic @transaction application."""

    @transaction()
    class Purchase:
        pass

    assert hasattr(Purchase, "_transaction")
    assert Purchase._transaction is True
    assert Purchase._transaction_atomic is True
    assert Purchase._transaction_log is True
    assert "transaction" in Purchase._applied_decorators


def test_transaction_custom_params():
    """Test @transaction with custom parameters."""

    @transaction(atomic=False, log=False)
    def transfer():
        pass

    assert transfer._transaction is True
    assert transfer._transaction_atomic is False
    assert transfer._transaction_log is False


def test_transaction_tags():
    """Test @transaction sets correct tags."""

    @transaction(atomic=True, log=True)
    class Trade:
        pass

    tags = Trade._tags
    assert tags["transaction"] is True
    assert tags["transaction_atomic"] is True
    assert tags["transaction_log"] is True


def test_transaction_registry():
    """Test @transaction registers correctly."""

    @transaction()
    def exchange():
        pass

    assert "economy" in exchange._registries
    spec = registry.get("transaction")
    assert spec is not None
    assert spec.tier == Tier.ECONOMY


# =============================================================================
# @mtx tests
# =============================================================================


def test_mtx_basic():
    """Test basic @mtx application."""

    @mtx(product_id="bundle_001", platforms={"ios": "com.game.bundle1"})
    class StarterBundle:
        pass

    assert hasattr(StarterBundle, "_mtx")
    assert StarterBundle._mtx is True
    assert StarterBundle._mtx_product_id == "bundle_001"
    assert StarterBundle._mtx_platforms == {"ios": "com.game.bundle1"}
    assert "mtx" in StarterBundle._applied_decorators


def test_mtx_multiple_platforms():
    """Test @mtx with multiple platforms."""

    @mtx(
        product_id="mega_pack",
        platforms={
            "ios": "com.game.mega",
            "android": "com.game.android.mega",
            "steam": "steam_mega_001",
        },
    )
    class MegaPack:
        pass

    assert MegaPack._mtx is True
    assert len(MegaPack._mtx_platforms) == 3
    assert MegaPack._mtx_platforms["steam"] == "steam_mega_001"


def test_mtx_validation_empty_product_id():
    """Test @mtx validation rejects empty product_id."""
    with pytest.raises(ValueError, match="'product_id' parameter is required"):

        @mtx(product_id="", platforms={"ios": "test"})
        class BadMTX:
            pass


def test_mtx_validation_empty_platforms():
    """Test @mtx validation rejects empty platforms."""
    with pytest.raises(ValueError, match="'platforms' must be a non-empty dict"):

        @mtx(product_id="test", platforms={})
        class BadMTX:
            pass


def test_mtx_validation_platforms_type():
    """Test @mtx validation rejects non-dict platforms."""
    with pytest.raises(TypeError, match="'platforms' must be a dict"):

        @mtx(product_id="test", platforms=["ios"])
        class BadMTX:
            pass


def test_mtx_tags():
    """Test @mtx sets correct tags."""

    @mtx(product_id="vip_pass", platforms={"ios": "com.game.vip"})
    class VIPPass:
        pass

    tags = VIPPass._tags
    assert tags["mtx"] is True
    assert tags["mtx_product_id"] == "vip_pass"
    assert tags["mtx_platforms"] == {"ios": "com.game.vip"}


def test_mtx_registry():
    """Test @mtx registers correctly."""

    @mtx(product_id="test", platforms={"test": "test"})
    class TestMTX:
        pass

    assert "economy" in TestMTX._registries
    spec = registry.get("mtx")
    assert spec is not None
    assert spec.tier == Tier.ECONOMY


# =============================================================================
# @daily_reward tests
# =============================================================================


def test_daily_reward_basic():
    """Test basic @daily_reward application."""

    @daily_reward()
    class DailyBonus:
        pass

    assert hasattr(DailyBonus, "_daily_reward")
    assert DailyBonus._daily_reward is True
    assert DailyBonus._daily_reward_reset_hour == 0
    assert "daily_reward" in DailyBonus._applied_decorators


def test_daily_reward_custom_hour():
    """Test @daily_reward with custom reset hour."""

    @daily_reward(reset_hour_utc=12)
    class NoonReset:
        pass

    assert NoonReset._daily_reward is True
    assert NoonReset._daily_reward_reset_hour == 12


def test_daily_reward_validation_out_of_range():
    """Test @daily_reward validation rejects out of range hours."""
    with pytest.raises(ValueError, match="'reset_hour_utc' must be 0-23"):

        @daily_reward(reset_hour_utc=24)
        class BadReward:
            pass

    with pytest.raises(ValueError, match="'reset_hour_utc' must be 0-23"):

        @daily_reward(reset_hour_utc=-1)
        class BadReward2:
            pass


def test_daily_reward_validation_type():
    """Test @daily_reward validation rejects non-int hours."""
    with pytest.raises(TypeError, match="'reset_hour_utc' must be an int"):

        @daily_reward(reset_hour_utc="12")
        class BadReward:
            pass


def test_daily_reward_tags():
    """Test @daily_reward sets correct tags."""

    @daily_reward(reset_hour_utc=6)
    class MorningReward:
        pass

    tags = MorningReward._tags
    assert tags["daily_reward"] is True
    assert tags["daily_reward_reset_hour"] == 6


def test_daily_reward_registry():
    """Test @daily_reward registers correctly."""

    @daily_reward()
    class TestReward:
        pass

    assert "economy" in TestReward._registries
    spec = registry.get("daily_reward")
    assert spec is not None
    assert spec.tier == Tier.ECONOMY


# =============================================================================
# Decorator composition tests
# =============================================================================


def test_currency_with_transaction():
    """Test composing @currency with @transaction."""

    @transaction()
    @currency(id="credits")
    class Credits:
        pass

    assert Credits._currency is True
    assert Credits._transaction is True
    assert "currency" in Credits._applied_decorators
    assert "transaction" in Credits._applied_decorators


def test_currency_with_mtx():
    """Test composing @currency with @mtx."""

    @mtx(product_id="premium_currency", platforms={"ios": "com.game.premium"})
    @currency(id="premium_coins", premium=True)
    class PremiumCoins:
        pass

    assert PremiumCoins._currency is True
    assert PremiumCoins._mtx is True
    assert PremiumCoins._currency_premium is True


def test_multiple_economy_decorators():
    """Test stacking multiple economy decorators."""

    @daily_reward(reset_hour_utc=3)
    @transaction(atomic=True)
    @currency(id="tokens", max_value=1000)
    class DailyTokens:
        pass

    assert DailyTokens._currency is True
    assert DailyTokens._transaction is True
    assert DailyTokens._daily_reward is True
    assert len(DailyTokens._applied_decorators) == 3


# =============================================================================
# Edge cases
# =============================================================================


def test_currency_direct_application():
    """Test @currency can be applied without parentheses (if it had no required params)."""
    # Note: currency requires 'id' so must use parentheses

    @currency(id="direct")
    class Direct:
        pass

    assert Direct._currency is True


def test_transaction_no_parens():
    """Test @transaction without parentheses (uses defaults)."""

    @transaction
    class NoParens:
        pass

    assert NoParens._transaction is True
    assert NoParens._transaction_atomic is True


def test_mtx_platforms_immutability():
    """Test that mtx platforms dict is copied."""
    platforms = {"ios": "test"}

    @mtx(product_id="test", platforms=platforms)
    class TestProduct:
        pass

    # Modify original dict
    platforms["android"] = "test2"

    # Should not affect decorated class
    assert "android" not in TestProduct._mtx_platforms
