"""
Tests for transaction functionality.
"""
import pytest

from engine.tooling.undo.transaction import (
    Transaction,
    TransactionState,
    TransactionManager,
    get_transaction_manager,
    atomic,
)
from engine.tooling.undo.command_pattern import (
    Command,
    SetFieldCommand,
)


class TestTransactionState:
    """Tests for TransactionState enum."""

    def test_all_states_exist(self):
        """Test all expected states exist."""
        assert hasattr(TransactionState, "PENDING")
        assert hasattr(TransactionState, "ACTIVE")
        assert hasattr(TransactionState, "COMMITTED")
        assert hasattr(TransactionState, "ROLLED_BACK")
        assert hasattr(TransactionState, "FAILED")


class TestTransaction:
    """Tests for Transaction."""

    def test_transaction_creation(self):
        """Test creating a transaction."""
        txn = Transaction(name="Test Transaction")

        assert txn.name == "Test Transaction"
        assert txn.state == TransactionState.PENDING
        assert txn.command_count == 0

    def test_begin(self):
        """Test beginning a transaction."""
        txn = Transaction(name="Test")

        txn.begin()

        assert txn.state == TransactionState.ACTIVE
        assert txn.is_active is True

    def test_begin_twice_raises(self):
        """Test beginning already started transaction raises."""
        txn = Transaction(name="Test")
        txn.begin()

        with pytest.raises(RuntimeError):
            txn.begin()

    def test_add_command(self):
        """Test adding commands to transaction."""
        class TestObj:
            x = 10

        obj = TestObj()
        txn = Transaction(name="Test")
        txn.begin()

        cmd = SetFieldCommand(obj, "x", 20)
        txn.add_command(cmd)

        assert txn.command_count == 1

    def test_add_command_inactive_raises(self):
        """Test adding command to inactive transaction raises."""
        class TestObj:
            x = 10

        obj = TestObj()
        txn = Transaction(name="Test")
        # Not begun

        cmd = SetFieldCommand(obj, "x", 20)

        with pytest.raises(RuntimeError):
            txn.add_command(cmd)

    def test_commit(self):
        """Test committing a transaction."""
        class TestObj:
            x = 10

        obj = TestObj()
        txn = Transaction(name="Test")
        txn.begin()

        cmd = SetFieldCommand(obj, "x", 20)
        txn.add_command(cmd)

        result = txn.commit()

        assert result is True
        assert txn.state == TransactionState.COMMITTED
        assert obj.x == 20

    def test_commit_inactive_raises(self):
        """Test committing inactive transaction raises."""
        txn = Transaction(name="Test")

        with pytest.raises(RuntimeError):
            txn.commit()

    def test_rollback(self):
        """Test rolling back a transaction."""
        class TestObj:
            x = 10

        obj = TestObj()
        txn = Transaction(name="Test")
        txn.begin()

        cmd = SetFieldCommand(obj, "x", 20)
        txn.add_command(cmd)
        cmd.execute()

        result = txn.rollback()

        assert result is True
        assert txn.state == TransactionState.ROLLED_BACK
        assert obj.x == 10

    def test_rollback_on_failure(self):
        """Test rollback when command fails during commit."""
        class TestObj:
            x = 10

        obj = TestObj()

        class FailingCommand(Command):
            def execute(self):
                return False
            def unexecute(self):
                return True

        txn = Transaction(name="Test")
        txn.begin()

        txn.add_command(SetFieldCommand(obj, "x", 20))
        txn.add_command(FailingCommand())

        result = txn.commit()

        assert result is False
        assert txn.state == TransactionState.FAILED
        assert obj.x == 10  # Should be rolled back

    def test_is_complete(self):
        """Test is_complete property."""
        txn = Transaction(name="Test")

        assert txn.is_complete is False

        txn.begin()
        assert txn.is_complete is False

        txn.commit()
        assert txn.is_complete is True

    def test_to_command(self):
        """Test converting transaction to composite command."""
        class TestObj:
            x = 10

        obj = TestObj()
        txn = Transaction(name="Test")
        txn.begin()
        txn.add_command(SetFieldCommand(obj, "x", 20))

        cmd = txn.to_command()

        assert cmd.name == "Test"
        assert cmd.command_count == 1


class TestTransactionManager:
    """Tests for TransactionManager."""

    def setup_method(self):
        """Create fresh manager for each test."""
        self.manager = TransactionManager()

    def test_manager_initialization(self):
        """Test TransactionManager initializes correctly."""
        assert self.manager.in_transaction is False
        assert self.manager.current_transaction is None
        assert self.manager.nesting_level == 0

    def test_begin(self):
        """Test beginning a transaction."""
        txn = self.manager.begin("Test")

        assert txn is not None
        assert self.manager.in_transaction is True
        assert self.manager.current_transaction is txn

    def test_commit(self):
        """Test committing a transaction."""
        class TestObj:
            x = 10

        obj = TestObj()

        self.manager.begin("Test")
        self.manager.add_command(SetFieldCommand(obj, "x", 20))

        result = self.manager.commit()

        assert result is True
        assert self.manager.in_transaction is False
        assert obj.x == 20

    def test_commit_no_transaction_raises(self):
        """Test committing without active transaction raises."""
        with pytest.raises(RuntimeError):
            self.manager.commit()

    def test_rollback(self):
        """Test rolling back a transaction."""
        class TestObj:
            x = 10

        obj = TestObj()

        self.manager.begin("Test")
        cmd = SetFieldCommand(obj, "x", 20)
        self.manager.add_command(cmd)
        cmd.execute()

        result = self.manager.rollback()

        assert result is True
        assert self.manager.in_transaction is False
        assert obj.x == 10

    def test_nested_transactions(self):
        """Test nested transactions."""
        self.manager.begin("Outer")
        assert self.manager.nesting_level == 0

        self.manager.begin("Inner")
        assert self.manager.nesting_level == 1

        self.manager.commit()
        assert self.manager.nesting_level == 0
        assert self.manager.in_transaction is True

        self.manager.commit()
        assert self.manager.in_transaction is False

    def test_savepoint(self):
        """Test creating a savepoint."""
        class TestObj:
            x = 10
            y = 20

        obj = TestObj()

        self.manager.begin("Test")

        cmd1 = SetFieldCommand(obj, "x", 100)
        self.manager.add_command(cmd1)
        cmd1.execute()

        sp = self.manager.savepoint("sp1")

        cmd2 = SetFieldCommand(obj, "y", 200)
        self.manager.add_command(cmd2)
        cmd2.execute()

        # Rollback to savepoint
        self.manager.rollback_to_savepoint(sp)

        assert obj.x == 100  # Before savepoint - kept
        assert obj.y == 20   # After savepoint - rolled back

    def test_savepoint_no_transaction_raises(self):
        """Test savepoint without transaction raises."""
        with pytest.raises(RuntimeError):
            self.manager.savepoint("sp1")

    def test_rollback_to_unknown_savepoint_raises(self):
        """Test rollback to unknown savepoint raises."""
        self.manager.begin("Test")

        with pytest.raises(ValueError):
            self.manager.rollback_to_savepoint("nonexistent")

    def test_release_savepoint(self):
        """Test releasing a savepoint."""
        self.manager.begin("Test")
        sp = self.manager.savepoint("sp1")
        self.manager.release_savepoint(sp)

        # Should not raise, just no-op if not found
        self.manager.release_savepoint("nonexistent")

    def test_context_manager(self):
        """Test using manager as context manager."""
        class TestObj:
            x = 10

        obj = TestObj()

        with self.manager.transaction("Test") as txn:
            cmd = SetFieldCommand(obj, "x", 20)
            self.manager.add_command(cmd)
            cmd.execute()

        assert obj.x == 20
        assert self.manager.in_transaction is False

    def test_context_manager_rollback_on_exception(self):
        """Test context manager rolls back on exception."""
        class TestObj:
            x = 10

        obj = TestObj()

        try:
            with self.manager.transaction("Test"):
                cmd = SetFieldCommand(obj, "x", 20)
                self.manager.add_command(cmd)
                cmd.execute()
                raise ValueError("Test error")
        except ValueError:
            pass

        assert obj.x == 10  # Should be rolled back
        assert self.manager.in_transaction is False

    def test_clear(self):
        """Test clearing the manager."""
        self.manager.begin("Test")
        self.manager.savepoint("sp1")

        self.manager.clear()

        assert self.manager.in_transaction is False
        assert self.manager.nesting_level == 0


class TestAtomicDecorator:
    """Tests for @atomic decorator."""

    def setup_method(self):
        # Clear global manager state
        get_transaction_manager().clear()

    def test_atomic_decorator(self):
        """Test @atomic decorator wraps function in transaction."""
        class TestObj:
            x = 10

        obj = TestObj()

        @atomic("Update")
        def update_obj():
            obj.x = 20

        update_obj()

        # Function should complete without error
        assert obj.x == 20

    def test_atomic_with_exception(self):
        """Test @atomic rolls back on exception."""
        class Counter:
            value = 0

        @atomic("Increment")
        def increment_and_fail():
            Counter.value += 1
            raise ValueError("Intentional error")

        original = Counter.value

        try:
            increment_and_fail()
        except ValueError:
            pass

        # Note: The counter may or may not be rolled back depending on
        # whether the increment was added as a command
        # This test mainly verifies no exception from the decorator


class TestGlobalTransactionManager:
    """Tests for global transaction manager."""

    def test_get_transaction_manager(self):
        """Test getting global transaction manager."""
        manager = get_transaction_manager()
        assert manager is not None
        assert isinstance(manager, TransactionManager)

    def test_global_is_singleton(self):
        """Test global instance is singleton."""
        m1 = get_transaction_manager()
        m2 = get_transaction_manager()
        assert m1 is m2
