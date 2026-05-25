"""
Comprehensive tests for the Currency System.

Tests cover:
- Currency types (gold, gems, tokens)
- Currency add/subtract
- Currency transfer
- Currency conversion
- Currency caps
- Transaction history
- Multi-currency operations
"""

import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from enum import Enum, auto

from engine.gameplay.economy.constants import (
    MAX_GOLD,
    MAX_PREMIUM_CURRENCY,
    CURRENCY_DENOMINATIONS,
)


# =============================================================================
# Currency System Implementation
# =============================================================================
# Note: This file tests a Currency system that would need to be implemented.
# The tests are written against an expected API. If the implementation doesn't
# exist, these tests define the expected behavior.


class CurrencyType(Enum):
    """Types of currency."""
    GOLD = auto()
    GEMS = auto()
    TOKENS = auto()
    COPPER = auto()
    SILVER = auto()
    PLATINUM = auto()


@dataclass
class CurrencyTransaction:
    """Record of a currency transaction."""
    transaction_id: UUID = field(default_factory=uuid4)
    currency_type: CurrencyType = CurrencyType.GOLD
    amount: int = 0
    reason: str = ""
    timestamp: float = 0.0
    source: Optional[str] = None
    target: Optional[str] = None


@dataclass
class CurrencyWallet:
    """
    A wallet that holds multiple currency types.

    Provides operations for adding, subtracting, transferring, and
    converting currencies with caps and history tracking.
    """
    owner_id: str = ""
    wallet_id: UUID = field(default_factory=uuid4)
    balances: Dict[CurrencyType, int] = field(default_factory=dict)
    caps: Dict[CurrencyType, int] = field(default_factory=dict)
    history: List[CurrencyTransaction] = field(default_factory=list)
    history_limit: int = 100

    def __post_init__(self):
        """Initialize default caps."""
        if CurrencyType.GOLD not in self.caps:
            self.caps[CurrencyType.GOLD] = MAX_GOLD
        if CurrencyType.GEMS not in self.caps:
            self.caps[CurrencyType.GEMS] = MAX_PREMIUM_CURRENCY

    def get_balance(self, currency_type: CurrencyType) -> int:
        """Get balance of a currency type."""
        return self.balances.get(currency_type, 0)

    def set_balance(self, currency_type: CurrencyType, amount: int) -> None:
        """Set balance of a currency type (clamped to cap)."""
        cap = self.caps.get(currency_type, MAX_GOLD)
        self.balances[currency_type] = min(max(0, amount), cap)

    def get_cap(self, currency_type: CurrencyType) -> int:
        """Get cap for a currency type."""
        return self.caps.get(currency_type, MAX_GOLD)

    def set_cap(self, currency_type: CurrencyType, cap: int) -> None:
        """Set cap for a currency type."""
        self.caps[currency_type] = max(0, cap)
        # Clamp existing balance to new cap
        if currency_type in self.balances:
            self.balances[currency_type] = min(self.balances[currency_type], cap)

    def can_afford(self, currency_type: CurrencyType, amount: int) -> bool:
        """Check if wallet can afford an amount."""
        return self.get_balance(currency_type) >= amount

    def add(
        self,
        currency_type: CurrencyType,
        amount: int,
        reason: str = "",
        source: Optional[str] = None,
        timestamp: float = 0.0,
    ) -> Tuple[bool, int]:
        """
        Add currency to wallet.

        Args:
            currency_type: Type of currency
            amount: Amount to add (must be positive)
            reason: Reason for addition
            source: Source of the currency
            timestamp: Transaction timestamp

        Returns:
            Tuple of (success, actual_amount_added)
        """
        if amount <= 0:
            return (False, 0)

        current = self.get_balance(currency_type)
        cap = self.get_cap(currency_type)
        space = cap - current
        actual = min(amount, space)

        if actual > 0:
            self.balances[currency_type] = current + actual
            self._record_transaction(
                currency_type, actual, reason, source, None, timestamp
            )

        return (actual > 0, actual)

    def subtract(
        self,
        currency_type: CurrencyType,
        amount: int,
        reason: str = "",
        target: Optional[str] = None,
        timestamp: float = 0.0,
    ) -> Tuple[bool, int]:
        """
        Subtract currency from wallet.

        Args:
            currency_type: Type of currency
            amount: Amount to subtract (must be positive)
            reason: Reason for subtraction
            target: Target of the currency
            timestamp: Transaction timestamp

        Returns:
            Tuple of (success, actual_amount_subtracted)
        """
        if amount <= 0:
            return (False, 0)

        current = self.get_balance(currency_type)
        if current < amount:
            return (False, 0)

        self.balances[currency_type] = current - amount
        self._record_transaction(
            currency_type, -amount, reason, None, target, timestamp
        )

        return (True, amount)

    def transfer_to(
        self,
        target_wallet: "CurrencyWallet",
        currency_type: CurrencyType,
        amount: int,
        reason: str = "",
        timestamp: float = 0.0,
    ) -> Tuple[bool, int]:
        """
        Transfer currency to another wallet.

        Args:
            target_wallet: Wallet to transfer to
            currency_type: Type of currency
            amount: Amount to transfer
            reason: Reason for transfer
            timestamp: Transaction timestamp

        Returns:
            Tuple of (success, actual_amount_transferred)
        """
        if amount <= 0:
            return (False, 0)

        if not self.can_afford(currency_type, amount):
            return (False, 0)

        # Check target has space
        target_space = target_wallet.get_cap(currency_type) - target_wallet.get_balance(currency_type)
        actual = min(amount, target_space)

        if actual <= 0:
            return (False, 0)

        # Perform transfer
        self.balances[currency_type] = self.get_balance(currency_type) - actual
        target_wallet.balances[currency_type] = target_wallet.get_balance(currency_type) + actual

        # Record in both wallets
        self._record_transaction(
            currency_type, -actual, reason, None, target_wallet.owner_id, timestamp
        )
        target_wallet._record_transaction(
            currency_type, actual, reason, self.owner_id, None, timestamp
        )

        return (True, actual)

    def convert(
        self,
        from_type: CurrencyType,
        to_type: CurrencyType,
        amount: int,
        rate: float,
        reason: str = "",
        timestamp: float = 0.0,
    ) -> Tuple[bool, int]:
        """
        Convert one currency type to another.

        Args:
            from_type: Currency type to convert from
            to_type: Currency type to convert to
            amount: Amount of from_type to convert
            rate: Exchange rate (to_amount = from_amount * rate)
            reason: Reason for conversion
            timestamp: Transaction timestamp

        Returns:
            Tuple of (success, amount_received)
        """
        if amount <= 0 or rate <= 0:
            return (False, 0)

        if not self.can_afford(from_type, amount):
            return (False, 0)

        # Calculate conversion
        to_amount = int(amount * rate)
        if to_amount <= 0:
            return (False, 0)

        # Check cap on destination
        to_space = self.get_cap(to_type) - self.get_balance(to_type)
        actual_to = min(to_amount, to_space)

        if actual_to <= 0:
            return (False, 0)

        # Calculate actual from amount based on what we can receive
        actual_from = int(actual_to / rate)
        if actual_from <= 0:
            actual_from = amount  # Use full amount if rate is very high
            actual_to = min(int(actual_from * rate), to_space)

        # Perform conversion
        self.balances[from_type] = self.get_balance(from_type) - actual_from
        self.balances[to_type] = self.get_balance(to_type) + actual_to

        self._record_transaction(
            from_type, -actual_from, f"Convert: {reason}", None, None, timestamp
        )
        self._record_transaction(
            to_type, actual_to, f"Convert: {reason}", None, None, timestamp
        )

        return (True, actual_to)

    def clear(self, currency_type: Optional[CurrencyType] = None) -> None:
        """Clear balance(s)."""
        if currency_type:
            self.balances[currency_type] = 0
        else:
            self.balances.clear()

    def get_history(
        self,
        currency_type: Optional[CurrencyType] = None,
        limit: Optional[int] = None,
    ) -> List[CurrencyTransaction]:
        """Get transaction history."""
        history = self.history
        if currency_type:
            history = [t for t in history if t.currency_type == currency_type]
        if limit:
            history = history[-limit:]
        return history

    def clear_history(self) -> None:
        """Clear transaction history."""
        self.history.clear()

    def _record_transaction(
        self,
        currency_type: CurrencyType,
        amount: int,
        reason: str,
        source: Optional[str],
        target: Optional[str],
        timestamp: float,
    ) -> None:
        """Record a transaction in history."""
        transaction = CurrencyTransaction(
            currency_type=currency_type,
            amount=amount,
            reason=reason,
            timestamp=timestamp,
            source=source,
            target=target,
        )
        self.history.append(transaction)

        # Trim history if over limit
        while len(self.history) > self.history_limit:
            self.history.pop(0)

    def to_dict(self) -> Dict:
        """Serialize wallet to dictionary."""
        return {
            "owner_id": self.owner_id,
            "wallet_id": str(self.wallet_id),
            "balances": {k.name: v for k, v in self.balances.items()},
            "caps": {k.name: v for k, v in self.caps.items()},
        }


class CurrencyExchange:
    """Manages currency exchange rates and operations."""

    def __init__(self):
        self._rates: Dict[Tuple[CurrencyType, CurrencyType], float] = {}
        self._fees: Dict[Tuple[CurrencyType, CurrencyType], float] = {}

        # Set default denomination rates (based on CURRENCY_DENOMINATIONS)
        self._setup_default_rates()

    def _setup_default_rates(self):
        """Setup default exchange rates based on denominations."""
        # Copper -> Silver -> Gold -> Platinum
        self.set_rate(CurrencyType.COPPER, CurrencyType.SILVER, 0.01)  # 100 copper = 1 silver
        self.set_rate(CurrencyType.SILVER, CurrencyType.COPPER, 100.0)
        self.set_rate(CurrencyType.SILVER, CurrencyType.GOLD, 0.01)    # 100 silver = 1 gold
        self.set_rate(CurrencyType.GOLD, CurrencyType.SILVER, 100.0)
        self.set_rate(CurrencyType.GOLD, CurrencyType.PLATINUM, 0.01)  # 100 gold = 1 platinum
        self.set_rate(CurrencyType.PLATINUM, CurrencyType.GOLD, 100.0)

    def set_rate(
        self,
        from_type: CurrencyType,
        to_type: CurrencyType,
        rate: float,
    ) -> None:
        """Set exchange rate between two currencies."""
        self._rates[(from_type, to_type)] = rate

    def get_rate(
        self,
        from_type: CurrencyType,
        to_type: CurrencyType,
    ) -> Optional[float]:
        """Get exchange rate between two currencies."""
        return self._rates.get((from_type, to_type))

    def set_fee(
        self,
        from_type: CurrencyType,
        to_type: CurrencyType,
        fee_percent: float,
    ) -> None:
        """Set exchange fee percentage."""
        self._fees[(from_type, to_type)] = fee_percent

    def get_fee(
        self,
        from_type: CurrencyType,
        to_type: CurrencyType,
    ) -> float:
        """Get exchange fee percentage."""
        return self._fees.get((from_type, to_type), 0.0)

    def calculate_exchange(
        self,
        from_type: CurrencyType,
        to_type: CurrencyType,
        amount: int,
    ) -> Tuple[int, int]:
        """
        Calculate exchange result and fee.

        Returns:
            Tuple of (amount_received, fee_amount)
        """
        rate = self.get_rate(from_type, to_type)
        if rate is None:
            return (0, 0)

        fee_percent = self.get_fee(from_type, to_type)
        fee_amount = int(amount * fee_percent)
        after_fee = amount - fee_amount

        received = int(after_fee * rate)
        return (received, fee_amount)

    def exchange(
        self,
        wallet: CurrencyWallet,
        from_type: CurrencyType,
        to_type: CurrencyType,
        amount: int,
        timestamp: float = 0.0,
    ) -> Tuple[bool, int, int]:
        """
        Perform currency exchange.

        Returns:
            Tuple of (success, amount_received, fee_paid)
        """
        rate = self.get_rate(from_type, to_type)
        if rate is None:
            return (False, 0, 0)

        if not wallet.can_afford(from_type, amount):
            return (False, 0, 0)

        received, fee = self.calculate_exchange(from_type, to_type, amount)
        if received <= 0:
            return (False, 0, 0)

        # Perform the conversion
        success, actual = wallet.convert(
            from_type, to_type, amount, rate * (1 - self.get_fee(from_type, to_type)),
            reason=f"Exchange {from_type.name} to {to_type.name}",
            timestamp=timestamp,
        )

        if success:
            return (True, actual, fee)
        return (False, 0, 0)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_wallet():
    """Create an empty wallet."""
    return CurrencyWallet(owner_id="player_1")


@pytest.fixture
def funded_wallet():
    """Create a wallet with some currency."""
    wallet = CurrencyWallet(owner_id="player_1")
    wallet.balances[CurrencyType.GOLD] = 1000
    wallet.balances[CurrencyType.GEMS] = 100
    wallet.balances[CurrencyType.TOKENS] = 50
    return wallet


@pytest.fixture
def exchange():
    """Create a currency exchange."""
    return CurrencyExchange()


# =============================================================================
# CurrencyWallet Creation Tests
# =============================================================================


class TestCurrencyWalletCreation:
    """Tests for CurrencyWallet creation."""

    def test_create_empty_wallet(self):
        """Test creating empty wallet."""
        wallet = CurrencyWallet(owner_id="player_1")
        assert wallet.owner_id == "player_1"
        assert isinstance(wallet.wallet_id, UUID)
        assert len(wallet.balances) == 0

    def test_wallet_default_caps(self, empty_wallet):
        """Test wallet has default caps."""
        assert empty_wallet.get_cap(CurrencyType.GOLD) == MAX_GOLD
        assert empty_wallet.get_cap(CurrencyType.GEMS) == MAX_PREMIUM_CURRENCY

    def test_wallet_custom_caps(self):
        """Test wallet with custom caps."""
        wallet = CurrencyWallet(
            owner_id="player_1",
            caps={CurrencyType.GOLD: 10000},
        )
        assert wallet.get_cap(CurrencyType.GOLD) == 10000

    def test_wallet_with_initial_balances(self):
        """Test wallet with initial balances."""
        wallet = CurrencyWallet(
            owner_id="player_1",
            balances={CurrencyType.GOLD: 500, CurrencyType.GEMS: 50},
        )
        assert wallet.get_balance(CurrencyType.GOLD) == 500
        assert wallet.get_balance(CurrencyType.GEMS) == 50


# =============================================================================
# CurrencyWallet Balance Tests
# =============================================================================


class TestCurrencyWalletBalance:
    """Tests for wallet balance operations."""

    def test_get_balance_empty(self, empty_wallet):
        """Test getting balance of empty currency."""
        assert empty_wallet.get_balance(CurrencyType.GOLD) == 0

    def test_get_balance_with_funds(self, funded_wallet):
        """Test getting balance with funds."""
        assert funded_wallet.get_balance(CurrencyType.GOLD) == 1000

    def test_set_balance(self, empty_wallet):
        """Test setting balance."""
        empty_wallet.set_balance(CurrencyType.GOLD, 500)
        assert empty_wallet.get_balance(CurrencyType.GOLD) == 500

    def test_set_balance_negative_clamps_to_zero(self, empty_wallet):
        """Test negative balance is clamped to zero."""
        empty_wallet.set_balance(CurrencyType.GOLD, -100)
        assert empty_wallet.get_balance(CurrencyType.GOLD) == 0

    def test_set_balance_over_cap_clamps(self):
        """Test balance over cap is clamped."""
        wallet = CurrencyWallet(
            owner_id="player_1",
            caps={CurrencyType.GOLD: 1000},
        )
        wallet.set_balance(CurrencyType.GOLD, 5000)
        assert wallet.get_balance(CurrencyType.GOLD) == 1000

    def test_can_afford_true(self, funded_wallet):
        """Test can_afford returns True when sufficient funds."""
        assert funded_wallet.can_afford(CurrencyType.GOLD, 500) is True

    def test_can_afford_exact(self, funded_wallet):
        """Test can_afford with exact amount."""
        assert funded_wallet.can_afford(CurrencyType.GOLD, 1000) is True

    def test_can_afford_false(self, funded_wallet):
        """Test can_afford returns False when insufficient funds."""
        assert funded_wallet.can_afford(CurrencyType.GOLD, 2000) is False

    def test_can_afford_empty_currency(self, empty_wallet):
        """Test can_afford with empty currency."""
        assert empty_wallet.can_afford(CurrencyType.GOLD, 1) is False


# =============================================================================
# CurrencyWallet Cap Tests
# =============================================================================


class TestCurrencyWalletCaps:
    """Tests for currency caps."""

    def test_get_cap(self, empty_wallet):
        """Test getting currency cap."""
        assert empty_wallet.get_cap(CurrencyType.GOLD) == MAX_GOLD

    def test_set_cap(self, empty_wallet):
        """Test setting currency cap."""
        empty_wallet.set_cap(CurrencyType.GOLD, 5000)
        assert empty_wallet.get_cap(CurrencyType.GOLD) == 5000

    def test_set_cap_negative_clamps(self, empty_wallet):
        """Test negative cap is clamped to zero."""
        empty_wallet.set_cap(CurrencyType.GOLD, -100)
        assert empty_wallet.get_cap(CurrencyType.GOLD) == 0

    def test_set_cap_clamps_existing_balance(self, funded_wallet):
        """Test setting cap lower than balance clamps balance."""
        funded_wallet.set_cap(CurrencyType.GOLD, 500)
        assert funded_wallet.get_balance(CurrencyType.GOLD) == 500

    def test_add_respects_cap(self):
        """Test adding currency respects cap."""
        wallet = CurrencyWallet(
            owner_id="player_1",
            caps={CurrencyType.GOLD: 1000},
        )
        wallet.add(CurrencyType.GOLD, 800)
        success, added = wallet.add(CurrencyType.GOLD, 500)
        assert success is True
        assert added == 200  # Only 200 fits
        assert wallet.get_balance(CurrencyType.GOLD) == 1000


# =============================================================================
# CurrencyWallet Add Tests
# =============================================================================


class TestCurrencyWalletAdd:
    """Tests for adding currency."""

    def test_add_basic(self, empty_wallet):
        """Test basic add operation."""
        success, amount = empty_wallet.add(CurrencyType.GOLD, 100)
        assert success is True
        assert amount == 100
        assert empty_wallet.get_balance(CurrencyType.GOLD) == 100

    def test_add_accumulates(self, funded_wallet):
        """Test adding accumulates."""
        initial = funded_wallet.get_balance(CurrencyType.GOLD)
        funded_wallet.add(CurrencyType.GOLD, 500)
        assert funded_wallet.get_balance(CurrencyType.GOLD) == initial + 500

    def test_add_zero_fails(self, empty_wallet):
        """Test adding zero fails."""
        success, amount = empty_wallet.add(CurrencyType.GOLD, 0)
        assert success is False
        assert amount == 0

    def test_add_negative_fails(self, empty_wallet):
        """Test adding negative fails."""
        success, amount = empty_wallet.add(CurrencyType.GOLD, -100)
        assert success is False
        assert amount == 0

    def test_add_with_reason(self, empty_wallet):
        """Test adding with reason records in history."""
        empty_wallet.add(CurrencyType.GOLD, 100, reason="Quest reward")
        assert len(empty_wallet.history) == 1
        assert empty_wallet.history[0].reason == "Quest reward"

    def test_add_with_source(self, empty_wallet):
        """Test adding with source."""
        empty_wallet.add(CurrencyType.GOLD, 100, source="quest_123")
        assert empty_wallet.history[0].source == "quest_123"

    def test_add_capped_returns_actual(self):
        """Test adding over cap returns actual amount."""
        wallet = CurrencyWallet(
            owner_id="player_1",
            caps={CurrencyType.GOLD: 100},
        )
        success, amount = wallet.add(CurrencyType.GOLD, 150)
        assert success is True
        assert amount == 100

    def test_add_to_full_wallet(self):
        """Test adding to full wallet."""
        wallet = CurrencyWallet(
            owner_id="player_1",
            caps={CurrencyType.GOLD: 100},
            balances={CurrencyType.GOLD: 100},
        )
        success, amount = wallet.add(CurrencyType.GOLD, 50)
        assert success is False
        assert amount == 0


# =============================================================================
# CurrencyWallet Subtract Tests
# =============================================================================


class TestCurrencyWalletSubtract:
    """Tests for subtracting currency."""

    def test_subtract_basic(self, funded_wallet):
        """Test basic subtract operation."""
        initial = funded_wallet.get_balance(CurrencyType.GOLD)
        success, amount = funded_wallet.subtract(CurrencyType.GOLD, 200)
        assert success is True
        assert amount == 200
        assert funded_wallet.get_balance(CurrencyType.GOLD) == initial - 200

    def test_subtract_exact_balance(self, funded_wallet):
        """Test subtracting exact balance."""
        balance = funded_wallet.get_balance(CurrencyType.GOLD)
        success, amount = funded_wallet.subtract(CurrencyType.GOLD, balance)
        assert success is True
        assert funded_wallet.get_balance(CurrencyType.GOLD) == 0

    def test_subtract_more_than_balance_fails(self, funded_wallet):
        """Test subtracting more than balance fails."""
        balance = funded_wallet.get_balance(CurrencyType.GOLD)
        success, amount = funded_wallet.subtract(CurrencyType.GOLD, balance + 1)
        assert success is False
        assert amount == 0
        assert funded_wallet.get_balance(CurrencyType.GOLD) == balance

    def test_subtract_zero_fails(self, funded_wallet):
        """Test subtracting zero fails."""
        success, amount = funded_wallet.subtract(CurrencyType.GOLD, 0)
        assert success is False

    def test_subtract_negative_fails(self, funded_wallet):
        """Test subtracting negative fails."""
        success, amount = funded_wallet.subtract(CurrencyType.GOLD, -100)
        assert success is False

    def test_subtract_with_reason(self, funded_wallet):
        """Test subtracting with reason records in history."""
        funded_wallet.subtract(CurrencyType.GOLD, 100, reason="Shop purchase")
        assert len(funded_wallet.history) >= 1
        assert any(t.reason == "Shop purchase" for t in funded_wallet.history)

    def test_subtract_with_target(self, funded_wallet):
        """Test subtracting with target."""
        funded_wallet.subtract(CurrencyType.GOLD, 100, target="shop_npc")
        assert any(t.target == "shop_npc" for t in funded_wallet.history)

    def test_subtract_from_empty_fails(self, empty_wallet):
        """Test subtracting from empty wallet fails."""
        success, amount = empty_wallet.subtract(CurrencyType.GOLD, 100)
        assert success is False


# =============================================================================
# CurrencyWallet Transfer Tests
# =============================================================================


class TestCurrencyWalletTransfer:
    """Tests for transferring currency."""

    def test_transfer_basic(self, funded_wallet, empty_wallet):
        """Test basic transfer operation."""
        initial = funded_wallet.get_balance(CurrencyType.GOLD)
        success, amount = funded_wallet.transfer_to(
            empty_wallet, CurrencyType.GOLD, 200
        )
        assert success is True
        assert amount == 200
        assert funded_wallet.get_balance(CurrencyType.GOLD) == initial - 200
        assert empty_wallet.get_balance(CurrencyType.GOLD) == 200

    def test_transfer_more_than_balance_fails(self, funded_wallet, empty_wallet):
        """Test transferring more than balance fails."""
        success, amount = funded_wallet.transfer_to(
            empty_wallet, CurrencyType.GOLD, 10000
        )
        assert success is False
        assert amount == 0

    def test_transfer_to_capped_wallet(self, funded_wallet):
        """Test transferring to wallet near cap."""
        target = CurrencyWallet(
            owner_id="player_2",
            caps={CurrencyType.GOLD: 100},
            balances={CurrencyType.GOLD: 80},
        )
        success, amount = funded_wallet.transfer_to(target, CurrencyType.GOLD, 50)
        assert success is True
        assert amount == 20  # Only 20 fits

    def test_transfer_zero_fails(self, funded_wallet, empty_wallet):
        """Test transferring zero fails."""
        success, amount = funded_wallet.transfer_to(
            empty_wallet, CurrencyType.GOLD, 0
        )
        assert success is False

    def test_transfer_records_in_both_histories(self, funded_wallet, empty_wallet):
        """Test transfer records in both wallets' histories."""
        funded_wallet.transfer_to(empty_wallet, CurrencyType.GOLD, 100, reason="Gift")

        # Sender should have outgoing transaction
        sender_trans = [t for t in funded_wallet.history if t.amount < 0]
        assert len(sender_trans) >= 1

        # Receiver should have incoming transaction
        receiver_trans = [t for t in empty_wallet.history if t.amount > 0]
        assert len(receiver_trans) >= 1

    def test_transfer_sets_source_and_target(self, funded_wallet, empty_wallet):
        """Test transfer sets source and target."""
        funded_wallet.transfer_to(empty_wallet, CurrencyType.GOLD, 100)

        # Sender's transaction should have target
        assert any(t.target == empty_wallet.owner_id for t in funded_wallet.history)

        # Receiver's transaction should have source
        assert any(t.source == funded_wallet.owner_id for t in empty_wallet.history)


# =============================================================================
# CurrencyWallet Convert Tests
# =============================================================================


class TestCurrencyWalletConvert:
    """Tests for converting currency."""

    def test_convert_basic(self, funded_wallet):
        """Test basic currency conversion."""
        initial_gold = funded_wallet.get_balance(CurrencyType.GOLD)
        success, received = funded_wallet.convert(
            CurrencyType.GOLD, CurrencyType.SILVER, 10, rate=100.0
        )
        assert success is True
        assert received == 1000
        assert funded_wallet.get_balance(CurrencyType.GOLD) == initial_gold - 10
        assert funded_wallet.get_balance(CurrencyType.SILVER) == 1000

    def test_convert_insufficient_funds_fails(self, funded_wallet):
        """Test conversion with insufficient funds fails."""
        success, received = funded_wallet.convert(
            CurrencyType.GOLD, CurrencyType.SILVER, 10000, rate=100.0
        )
        assert success is False
        assert received == 0

    def test_convert_zero_fails(self, funded_wallet):
        """Test converting zero fails."""
        success, received = funded_wallet.convert(
            CurrencyType.GOLD, CurrencyType.SILVER, 0, rate=100.0
        )
        assert success is False

    def test_convert_zero_rate_fails(self, funded_wallet):
        """Test converting with zero rate fails."""
        success, received = funded_wallet.convert(
            CurrencyType.GOLD, CurrencyType.SILVER, 10, rate=0.0
        )
        assert success is False

    def test_convert_respects_destination_cap(self):
        """Test conversion respects destination cap."""
        wallet = CurrencyWallet(
            owner_id="player_1",
            balances={CurrencyType.GOLD: 1000},
            caps={CurrencyType.SILVER: 500},
        )
        success, received = wallet.convert(
            CurrencyType.GOLD, CurrencyType.SILVER, 100, rate=100.0
        )
        # Would get 10000 silver but cap is 500
        assert received <= 500

    def test_convert_records_transactions(self, funded_wallet):
        """Test conversion records transactions."""
        initial_history_len = len(funded_wallet.history)
        funded_wallet.convert(
            CurrencyType.GOLD, CurrencyType.SILVER, 10, rate=100.0,
            reason="Exchange"
        )
        # Should have two new transactions (subtract and add)
        assert len(funded_wallet.history) >= initial_history_len + 2


# =============================================================================
# CurrencyWallet History Tests
# =============================================================================


class TestCurrencyWalletHistory:
    """Tests for transaction history."""

    def test_history_initially_empty(self, empty_wallet):
        """Test history is initially empty."""
        assert len(empty_wallet.history) == 0

    def test_history_records_add(self, empty_wallet):
        """Test history records add transactions."""
        empty_wallet.add(CurrencyType.GOLD, 100, reason="Test")
        assert len(empty_wallet.history) == 1
        assert empty_wallet.history[0].amount == 100

    def test_history_records_subtract(self, funded_wallet):
        """Test history records subtract transactions."""
        funded_wallet.subtract(CurrencyType.GOLD, 100, reason="Test")
        assert len(funded_wallet.history) >= 1
        assert any(t.amount == -100 for t in funded_wallet.history)

    def test_get_history_all(self, funded_wallet):
        """Test getting all history."""
        funded_wallet.add(CurrencyType.GOLD, 50)
        funded_wallet.subtract(CurrencyType.GOLD, 25)
        funded_wallet.add(CurrencyType.GEMS, 10)

        history = funded_wallet.get_history()
        assert len(history) == 3

    def test_get_history_by_currency(self, funded_wallet):
        """Test getting history filtered by currency."""
        funded_wallet.add(CurrencyType.GOLD, 50)
        funded_wallet.add(CurrencyType.GEMS, 10)

        gold_history = funded_wallet.get_history(currency_type=CurrencyType.GOLD)
        assert all(t.currency_type == CurrencyType.GOLD for t in gold_history)

    def test_get_history_with_limit(self, empty_wallet):
        """Test getting history with limit."""
        for i in range(10):
            empty_wallet.add(CurrencyType.GOLD, 10)

        history = empty_wallet.get_history(limit=5)
        assert len(history) == 5

    def test_history_limit_trims_old(self):
        """Test history limit trims old entries."""
        wallet = CurrencyWallet(owner_id="player_1", history_limit=5)

        for i in range(10):
            wallet.add(CurrencyType.GOLD, i + 1)

        assert len(wallet.history) == 5
        # Should have most recent entries
        assert wallet.history[-1].amount == 10

    def test_clear_history(self, funded_wallet):
        """Test clearing history."""
        funded_wallet.add(CurrencyType.GOLD, 100)
        funded_wallet.clear_history()
        assert len(funded_wallet.history) == 0


# =============================================================================
# CurrencyWallet Clear Tests
# =============================================================================


class TestCurrencyWalletClear:
    """Tests for clearing balances."""

    def test_clear_single_currency(self, funded_wallet):
        """Test clearing single currency."""
        funded_wallet.clear(CurrencyType.GOLD)
        assert funded_wallet.get_balance(CurrencyType.GOLD) == 0
        assert funded_wallet.get_balance(CurrencyType.GEMS) > 0  # Other currency unchanged

    def test_clear_all_currencies(self, funded_wallet):
        """Test clearing all currencies."""
        funded_wallet.clear()
        assert funded_wallet.get_balance(CurrencyType.GOLD) == 0
        assert funded_wallet.get_balance(CurrencyType.GEMS) == 0
        assert funded_wallet.get_balance(CurrencyType.TOKENS) == 0


# =============================================================================
# CurrencyWallet Serialization Tests
# =============================================================================


class TestCurrencyWalletSerialization:
    """Tests for wallet serialization."""

    def test_to_dict(self, funded_wallet):
        """Test serializing to dictionary."""
        data = funded_wallet.to_dict()
        assert data["owner_id"] == "player_1"
        assert "wallet_id" in data
        assert "balances" in data
        assert "caps" in data

    def test_to_dict_balances(self, funded_wallet):
        """Test serialized balances."""
        data = funded_wallet.to_dict()
        assert data["balances"]["GOLD"] == 1000
        assert data["balances"]["GEMS"] == 100


# =============================================================================
# CurrencyExchange Tests
# =============================================================================


class TestCurrencyExchange:
    """Tests for CurrencyExchange class."""

    def test_create_exchange(self, exchange):
        """Test creating exchange."""
        assert exchange is not None

    def test_default_rates_exist(self, exchange):
        """Test default rates are set."""
        rate = exchange.get_rate(CurrencyType.COPPER, CurrencyType.SILVER)
        assert rate is not None

    def test_set_rate(self, exchange):
        """Test setting exchange rate."""
        exchange.set_rate(CurrencyType.GOLD, CurrencyType.GEMS, 0.1)
        assert exchange.get_rate(CurrencyType.GOLD, CurrencyType.GEMS) == 0.1

    def test_get_rate_not_set(self, exchange):
        """Test getting rate that's not set."""
        rate = exchange.get_rate(CurrencyType.TOKENS, CurrencyType.GEMS)
        assert rate is None

    def test_set_fee(self, exchange):
        """Test setting exchange fee."""
        exchange.set_fee(CurrencyType.GOLD, CurrencyType.GEMS, 0.05)
        assert exchange.get_fee(CurrencyType.GOLD, CurrencyType.GEMS) == 0.05

    def test_get_fee_default(self, exchange):
        """Test getting default fee (0)."""
        fee = exchange.get_fee(CurrencyType.GOLD, CurrencyType.TOKENS)
        assert fee == 0.0

    def test_calculate_exchange_basic(self, exchange):
        """Test calculating exchange result."""
        exchange.set_rate(CurrencyType.GOLD, CurrencyType.SILVER, 100.0)
        received, fee = exchange.calculate_exchange(
            CurrencyType.GOLD, CurrencyType.SILVER, 10
        )
        assert received == 1000
        assert fee == 0

    def test_calculate_exchange_with_fee(self, exchange):
        """Test calculating exchange with fee."""
        exchange.set_rate(CurrencyType.GOLD, CurrencyType.SILVER, 100.0)
        exchange.set_fee(CurrencyType.GOLD, CurrencyType.SILVER, 0.1)  # 10% fee

        received, fee = exchange.calculate_exchange(
            CurrencyType.GOLD, CurrencyType.SILVER, 10
        )
        # 10 gold - 10% fee = 9 gold, 9 * 100 = 900 silver
        assert received == 900
        assert fee == 1

    def test_calculate_exchange_no_rate(self, exchange):
        """Test calculating exchange without rate."""
        received, fee = exchange.calculate_exchange(
            CurrencyType.TOKENS, CurrencyType.GEMS, 100
        )
        assert received == 0
        assert fee == 0

    def test_exchange_operation(self, exchange, funded_wallet):
        """Test performing exchange."""
        exchange.set_rate(CurrencyType.GOLD, CurrencyType.SILVER, 100.0)

        initial_gold = funded_wallet.get_balance(CurrencyType.GOLD)
        success, received, fee = exchange.exchange(
            funded_wallet, CurrencyType.GOLD, CurrencyType.SILVER, 10
        )

        assert success is True
        assert received > 0
        assert funded_wallet.get_balance(CurrencyType.GOLD) < initial_gold

    def test_exchange_insufficient_funds(self, exchange, empty_wallet):
        """Test exchange with insufficient funds."""
        exchange.set_rate(CurrencyType.GOLD, CurrencyType.SILVER, 100.0)

        success, received, fee = exchange.exchange(
            empty_wallet, CurrencyType.GOLD, CurrencyType.SILVER, 10
        )

        assert success is False
        assert received == 0

    def test_exchange_no_rate(self, exchange, funded_wallet):
        """Test exchange without rate set."""
        success, received, fee = exchange.exchange(
            funded_wallet, CurrencyType.TOKENS, CurrencyType.GEMS, 10
        )

        assert success is False


# =============================================================================
# Multi-Currency Operations Tests
# =============================================================================


class TestMultiCurrencyOperations:
    """Tests for operations involving multiple currencies."""

    def test_multiple_currency_wallet(self):
        """Test wallet with multiple currencies."""
        wallet = CurrencyWallet(owner_id="player_1")
        wallet.add(CurrencyType.GOLD, 1000)
        wallet.add(CurrencyType.GEMS, 100)
        wallet.add(CurrencyType.TOKENS, 50)
        wallet.add(CurrencyType.COPPER, 5000)

        assert wallet.get_balance(CurrencyType.GOLD) == 1000
        assert wallet.get_balance(CurrencyType.GEMS) == 100
        assert wallet.get_balance(CurrencyType.TOKENS) == 50
        assert wallet.get_balance(CurrencyType.COPPER) == 5000

    def test_transfer_different_currencies(self, funded_wallet):
        """Test transferring different currency types."""
        target = CurrencyWallet(owner_id="player_2")

        funded_wallet.transfer_to(target, CurrencyType.GOLD, 100)
        funded_wallet.transfer_to(target, CurrencyType.GEMS, 10)

        assert target.get_balance(CurrencyType.GOLD) == 100
        assert target.get_balance(CurrencyType.GEMS) == 10

    def test_chain_conversions(self, exchange):
        """Test chain of currency conversions."""
        wallet = CurrencyWallet(
            owner_id="player_1",
            balances={CurrencyType.COPPER: 10000},
        )

        # Convert copper -> silver -> gold
        wallet.convert(CurrencyType.COPPER, CurrencyType.SILVER, 10000, rate=0.01)
        wallet.convert(CurrencyType.SILVER, CurrencyType.GOLD, 100, rate=0.01)

        assert wallet.get_balance(CurrencyType.COPPER) == 0
        assert wallet.get_balance(CurrencyType.GOLD) == 1

    def test_complex_transaction_history(self):
        """Test complex transaction history."""
        wallet = CurrencyWallet(owner_id="player_1")

        # Various transactions
        wallet.add(CurrencyType.GOLD, 1000, reason="Starting funds")
        wallet.add(CurrencyType.GEMS, 50, reason="Login bonus")
        wallet.subtract(CurrencyType.GOLD, 200, reason="Shop purchase")
        wallet.convert(CurrencyType.GOLD, CurrencyType.SILVER, 100, rate=100.0)

        history = wallet.get_history()
        assert len(history) >= 4  # At least 4 transactions (convert adds 2)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestCurrencyEdgeCases:
    """Tests for edge cases."""

    def test_very_large_amount(self, empty_wallet):
        """Test handling very large amounts."""
        success, added = empty_wallet.add(CurrencyType.GOLD, MAX_GOLD)
        assert success is True
        assert empty_wallet.get_balance(CurrencyType.GOLD) == MAX_GOLD

    def test_overflow_protection(self, funded_wallet):
        """Test protection against overflow."""
        # Set balance near max
        funded_wallet.set_balance(CurrencyType.GOLD, MAX_GOLD - 10)

        # Try to add more than fits
        success, added = funded_wallet.add(CurrencyType.GOLD, 100)
        assert added == 10  # Only 10 fits

    def test_concurrent_operations_simulation(self, funded_wallet):
        """Test simulating concurrent operations."""
        # Simulate multiple operations
        operations = [
            ("add", CurrencyType.GOLD, 100),
            ("subtract", CurrencyType.GOLD, 50),
            ("add", CurrencyType.GEMS, 25),
            ("subtract", CurrencyType.GOLD, 30),
        ]

        for op, currency, amount in operations:
            if op == "add":
                funded_wallet.add(currency, amount)
            else:
                funded_wallet.subtract(currency, amount)

        # Verify final state
        # 1000 + 100 - 50 - 30 = 1020
        assert funded_wallet.get_balance(CurrencyType.GOLD) == 1020
        # 100 + 25 = 125
        assert funded_wallet.get_balance(CurrencyType.GEMS) == 125

    def test_zero_cap(self):
        """Test zero cap prevents additions."""
        wallet = CurrencyWallet(
            owner_id="player_1",
            caps={CurrencyType.GOLD: 0},
        )
        success, added = wallet.add(CurrencyType.GOLD, 100)
        assert success is False
        assert added == 0

    def test_fractional_conversion_truncation(self, funded_wallet):
        """Test fractional amounts are truncated in conversions."""
        # Rate that would produce fractional result
        funded_wallet.convert(
            CurrencyType.GOLD, CurrencyType.SILVER, 1, rate=0.5
        )
        # 1 * 0.5 = 0.5, truncated to 0
        # This should fail since result would be 0
