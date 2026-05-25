"""Tests for the RPC (Remote Procedure Call) system.

Tests cover:
- RPC registration and invocation
- Authority validation (server/client/owner)
- Rate limiting
- RPC channel communication
- Validation utilities
"""

import pytest
import time
from dataclasses import dataclass

from engine.networking.rpc import (
    # RPC Manager
    RPCManager,
    RPCInfo,
    RPCAuthority,
    RPCReliability,
    RPCCallResult,
    rpc,

    # RPC Channel
    RPCChannel,
    RPCChannelManager,
    RPCChannelState,
    RPCMessage,

    # Validation
    RPCValidator,
    RateLimiter,
    RateLimitConfig,
    ValidationError,
    validate_authority,
    validate_rate_limit,
    validate_param_range,
    validate_param_type,
    validate_param_length,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

@dataclass
class MockCaller:
    """Mock caller for testing."""
    player_id: int = 1
    name: str = "TestPlayer"


def sample_server_rpc(caller, damage: int, target_id: int) -> bool:
    """Sample server RPC for testing."""
    return True


def sample_client_rpc(caller, message: str) -> None:
    """Sample client RPC for testing."""
    pass


# =============================================================================
# RPCInfo Tests
# =============================================================================

class TestRPCInfo:
    """Tests for RPCInfo class."""

    def test_info_creation(self):
        """Test RPCInfo creation."""
        info = RPCInfo(
            name="test_rpc",
            authority=RPCAuthority.SERVER,
            reliability=RPCReliability.RELIABLE
        )

        assert info.name == "test_rpc"
        assert info.authority == RPCAuthority.SERVER
        assert info.reliable

    def test_info_unreliable(self):
        """Test unreliable RPC info."""
        info = RPCInfo(
            name="test_rpc",
            authority=RPCAuthority.SERVER,
            reliability=RPCReliability.UNRELIABLE
        )

        assert not info.reliable

    def test_info_hash(self):
        """Test RPCInfo hash generation."""
        info = RPCInfo(name="test_rpc", authority=RPCAuthority.SERVER)

        hash_val = info.get_hash()

        assert isinstance(hash_val, int)
        assert 0 <= hash_val <= 0xFFFFFFFF

    def test_info_hash_consistency(self):
        """Test hash is consistent for same name."""
        info1 = RPCInfo(name="my_rpc", authority=RPCAuthority.SERVER)
        info2 = RPCInfo(name="my_rpc", authority=RPCAuthority.CLIENT)

        assert info1.get_hash() == info2.get_hash()

    def test_info_immutable(self):
        """Test RPCInfo is immutable."""
        info = RPCInfo(name="test", authority=RPCAuthority.SERVER)

        with pytest.raises(Exception):  # FrozenInstanceError
            info.name = "changed"


# =============================================================================
# RPCManager Tests
# =============================================================================

class TestRPCManager:
    """Tests for RPCManager class."""

    def test_manager_creation_server(self):
        """Test server manager creation."""
        manager = RPCManager(is_server=True)

        assert manager.is_server

    def test_manager_creation_client(self):
        """Test client manager creation."""
        manager = RPCManager(is_server=False)

        assert not manager.is_server

    def test_register_rpc(self):
        """Test RPC registration."""
        manager = RPCManager()

        info = manager.register_rpc(
            sample_server_rpc,
            authority=RPCAuthority.SERVER,
            reliability=RPCReliability.RELIABLE
        )

        assert info.name == "sample_server_rpc"
        assert manager.get_rpc_info("sample_server_rpc") is not None

    def test_register_rpc_custom_name(self):
        """Test RPC registration with custom name."""
        manager = RPCManager()

        info = manager.register_rpc(
            sample_server_rpc,
            authority=RPCAuthority.SERVER,
            name="custom_name"
        )

        assert info.name == "custom_name"
        assert manager.get_rpc_info("custom_name") is not None

    def test_unregister_rpc(self):
        """Test RPC unregistration."""
        manager = RPCManager()

        manager.register_rpc(sample_server_rpc, authority=RPCAuthority.SERVER)
        assert manager.unregister_rpc("sample_server_rpc")
        assert manager.get_rpc_info("sample_server_rpc") is None

    def test_call_rpc_server(self):
        """Test server RPC call."""
        manager = RPCManager(is_server=True)

        sent_data = []

        def on_rpc(data, target, reliable):
            sent_data.append((data, target, reliable))

        manager.set_rpc_callback(on_rpc)

        manager.register_rpc(
            sample_server_rpc,
            authority=RPCAuthority.SERVER,
            reliability=RPCReliability.RELIABLE
        )

        result = manager.call_rpc(
            "sample_server_rpc",
            args=(50, 123),
            target="client1"
        )

        assert result.success
        assert result.sequence > 0
        assert len(sent_data) == 1

    def test_call_rpc_unknown(self):
        """Test calling unknown RPC."""
        manager = RPCManager()

        result = manager.call_rpc(
            "nonexistent_rpc",
            args=(),
            target=None
        )

        assert not result.success
        assert "Unknown" in result.error

    def test_call_rpc_authority_denied_client(self):
        """Test authority denied for client calling server RPC."""
        manager = RPCManager(is_server=False)  # Client

        manager.register_rpc(
            sample_client_rpc,
            authority=RPCAuthority.SERVER  # Server authority
        )

        result = manager.call_rpc(
            "sample_client_rpc",
            args=("hello",),
            target=None
        )

        # Client can't send server RPCs
        assert not result.success
        assert "Authority" in result.error

    def test_receive_rpc(self):
        """Test receiving and executing RPC."""
        manager = RPCManager(is_server=True)

        executed = []

        def handler(caller, damage, target_id):
            executed.append((caller, damage, target_id))
            return True

        manager.register_rpc(
            handler,
            authority=RPCAuthority.CLIENT,
            name="apply_damage"
        )

        # Manually create RPC data
        info = manager.get_rpc_info("apply_damage")
        data = manager._serialize_rpc(info, 1, (50, 123))

        caller = MockCaller()
        success, result = manager.receive_rpc(data, caller, caller_id=1)

        assert success
        assert len(executed) == 1
        assert executed[0][1] == 50  # damage

    def test_acknowledge_rpc(self):
        """Test RPC acknowledgment."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(lambda d, t, r: None)

        manager.register_rpc(
            sample_server_rpc,
            authority=RPCAuthority.SERVER,
            reliability=RPCReliability.RELIABLE
        )

        result = manager.call_rpc(
            "sample_server_rpc",
            args=(50, 123),
            target=None
        )

        # Should be pending
        assert result.sequence in manager._pending_rpcs

        # Acknowledge
        assert manager.acknowledge_rpc(result.sequence)
        assert result.sequence not in manager._pending_rpcs

    def test_pending_retransmits(self):
        """Test getting pending retransmits."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(lambda d, t, r: None)

        manager.register_rpc(
            sample_server_rpc,
            authority=RPCAuthority.SERVER,
            reliability=RPCReliability.RELIABLE
        )

        result = manager.call_rpc(
            "sample_server_rpc",
            args=(50, 123),
            target=None
        )

        # Force timestamp to be old
        manager._pending_rpcs[result.sequence].timestamp = time.time() - 1.0

        retransmits = manager.get_pending_retransmits(timeout=0.5)

        assert len(retransmits) == 1


class TestRPCDecorator:
    """Tests for @rpc decorator."""

    def test_rpc_decorator(self):
        """Test @rpc decorator marks function."""

        @rpc(authority=RPCAuthority.CLIENT, reliable=True, rate_limit=10.0)
        def my_rpc(self, value: int):
            pass

        assert hasattr(my_rpc, '_rpc_info')
        info = my_rpc._rpc_info

        assert info.name == "my_rpc"
        assert info.authority == RPCAuthority.CLIENT
        assert info.reliable
        assert info.rate_limit == 10.0

    def test_rpc_decorator_unreliable(self):
        """Test @rpc decorator with unreliable."""

        @rpc(authority=RPCAuthority.SERVER, reliable=False)
        def explosion_effect(self, position):
            pass

        info = explosion_effect._rpc_info

        assert not info.reliable


# =============================================================================
# RPCChannel Tests
# =============================================================================

class TestRPCChannel:
    """Tests for RPCChannel class."""

    def test_channel_lifecycle(self):
        """Test channel open/close lifecycle."""
        channel = RPCChannel(connection_id=1)

        assert channel.state == RPCChannelState.CLOSED

        assert channel.open()
        assert channel.state == RPCChannelState.OPEN

        assert channel.close()
        # Closes immediately if no pending
        assert channel.state == RPCChannelState.CLOSED

    def test_send_rpc(self):
        """Test sending RPC through channel."""
        channel = RPCChannel(connection_id=1)
        channel.open()

        info = RPCInfo(name="test_rpc", authority=RPCAuthority.SERVER)

        sequence = channel.send_rpc(info, b"test_args")

        assert sequence is not None
        assert sequence > 0

    def test_send_rpc_closed(self):
        """Test can't send on closed channel."""
        channel = RPCChannel(connection_id=1)

        info = RPCInfo(name="test_rpc", authority=RPCAuthority.SERVER)

        sequence = channel.send_rpc(info, b"test_args")

        assert sequence is None

    def test_get_outgoing_data(self):
        """Test getting outgoing data."""
        channel = RPCChannel(connection_id=1)
        channel.open()

        info = RPCInfo(name="test_rpc", authority=RPCAuthority.SERVER)
        channel.send_rpc(info, b"test_args")

        data = channel.get_outgoing_data()

        assert len(data) > 0
        assert channel.outgoing_count == 0  # Queue emptied

    def test_reliable_pending(self):
        """Test reliable messages are tracked."""
        channel = RPCChannel(connection_id=1)
        channel.open()

        info = RPCInfo(
            name="test_rpc",
            authority=RPCAuthority.SERVER,
            reliability=RPCReliability.RELIABLE
        )
        channel.send_rpc(info, b"test_args")
        channel.get_outgoing_data()

        assert channel.pending_count == 1

    def test_acknowledge(self):
        """Test acknowledgment removes pending."""
        channel = RPCChannel(connection_id=1)
        channel.open()

        info = RPCInfo(
            name="test_rpc",
            authority=RPCAuthority.SERVER,
            reliability=RPCReliability.RELIABLE
        )
        sequence = channel.send_rpc(info, b"test_args")
        channel.get_outgoing_data()

        channel.acknowledge(sequence)

        assert channel.pending_count == 0

    def test_receive_rpc(self):
        """Test receiving RPC data."""
        channel = RPCChannel(connection_id=1)
        channel.open()

        info = RPCInfo(name="test_rpc", authority=RPCAuthority.SERVER)

        # Create message manually
        msg = RPCMessage(
            msg_type=0x01,  # RPC_MSG_CALL
            rpc_hash=info.get_hash(),
            sequence=1,
            payload=b"args_data",
            reliable=True
        )
        data = msg.serialize()

        results = channel.receive_rpc(data)

        assert len(results) == 1
        assert results[0][0] == info.get_hash()

    def test_retransmit(self):
        """Test retransmission data generation."""
        channel = RPCChannel(connection_id=1)
        channel.open()

        info = RPCInfo(
            name="test_rpc",
            authority=RPCAuthority.SERVER,
            reliability=RPCReliability.RELIABLE
        )
        channel.send_rpc(info, b"test_args")
        channel.get_outgoing_data()

        # Force old timestamp
        for msg in channel._pending_ack.values():
            msg.timestamp = time.time() - 1.0

        data = channel.get_retransmit_data()

        assert len(data) > 0


class TestRPCChannelManager:
    """Tests for RPCChannelManager class."""

    def test_get_or_create_channel(self):
        """Test channel creation."""
        manager = RPCChannelManager()

        channel = manager.get_or_create_channel(1)

        assert channel is not None
        assert channel.is_open

    def test_get_existing_channel(self):
        """Test getting existing channel."""
        manager = RPCChannelManager()

        channel1 = manager.get_or_create_channel(1)
        channel2 = manager.get_or_create_channel(1)

        assert channel1 is channel2

    def test_send_rpc(self):
        """Test sending RPC through manager."""
        manager = RPCChannelManager()

        info = RPCInfo(name="test_rpc", authority=RPCAuthority.SERVER)

        sequence = manager.send_rpc(1, info, b"args")

        assert sequence is not None

    def test_broadcast_rpc(self):
        """Test broadcasting RPC to all connections."""
        manager = RPCChannelManager()

        # Create multiple channels
        manager.get_or_create_channel(1)
        manager.get_or_create_channel(2)
        manager.get_or_create_channel(3)

        info = RPCInfo(name="test_rpc", authority=RPCAuthority.MULTICAST)

        results = manager.broadcast_rpc(info, b"broadcast_args")

        assert len(results) == 3

    def test_broadcast_rpc_exclude(self):
        """Test broadcasting with exclusions."""
        manager = RPCChannelManager()

        manager.get_or_create_channel(1)
        manager.get_or_create_channel(2)
        manager.get_or_create_channel(3)

        info = RPCInfo(name="test_rpc", authority=RPCAuthority.MULTICAST)

        results = manager.broadcast_rpc(info, b"args", exclude={2})

        assert len(results) == 2
        assert 2 not in results

    def test_remove_channel(self):
        """Test channel removal."""
        manager = RPCChannelManager()

        manager.get_or_create_channel(1)
        assert manager.remove_channel(1)

        # Should create new one
        channel = manager.get_or_create_channel(1)
        assert channel.is_open

    def test_cleanup_closed(self):
        """Test cleanup of closed channels."""
        manager = RPCChannelManager()

        channel = manager.get_or_create_channel(1)
        channel.force_close()

        removed = manager.cleanup_closed_channels()

        assert removed == 1


# =============================================================================
# RPC Validation Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limit_allows(self):
        """Test calls within limit are allowed."""
        limiter = RateLimiter(RateLimitConfig(max_calls=10, window_seconds=1.0))

        for _ in range(10):
            assert limiter.check_rate_limit(1, "test_rpc")

    def test_rate_limit_blocks(self):
        """Test calls over limit are blocked."""
        limiter = RateLimiter(RateLimitConfig(
            max_calls=5,
            window_seconds=1.0,
            burst_allowance=0
        ))

        for _ in range(5):
            limiter.check_rate_limit(1, "test_rpc")

        # 6th call should fail
        assert not limiter.check_rate_limit(1, "test_rpc")

    def test_rate_limit_burst(self):
        """Test burst allowance."""
        limiter = RateLimiter(RateLimitConfig(
            max_calls=5,
            window_seconds=1.0,
            burst_allowance=3
        ))

        # 5 normal + 3 burst = 8 allowed
        for _ in range(8):
            assert limiter.check_rate_limit(1, "test_rpc")

        # 9th should fail
        assert not limiter.check_rate_limit(1, "test_rpc")

    def test_rate_limit_per_rpc(self):
        """Test rate limits are per-RPC."""
        limiter = RateLimiter(RateLimitConfig(
            max_calls=5,
            window_seconds=1.0,
            burst_allowance=0
        ))

        # Max out rpc_a
        for _ in range(5):
            limiter.check_rate_limit(1, "rpc_a")

        # rpc_b should still work
        assert limiter.check_rate_limit(1, "rpc_b")

    def test_get_remaining_calls(self):
        """Test getting remaining call count."""
        limiter = RateLimiter(RateLimitConfig(
            max_calls=10,
            window_seconds=1.0,
            burst_allowance=5
        ))

        assert limiter.get_remaining_calls(1, "test") == 10  # Just normal

        for _ in range(3):
            limiter.check_rate_limit(1, "test")

        # 7 normal + 5 burst remaining
        assert limiter.get_remaining_calls(1, "test") == 12


class TestValidateAuthority:
    """Tests for authority validation."""

    def test_server_authority_server_sending(self):
        """Test server can send server authority RPC."""
        info = RPCInfo(name="test", authority=RPCAuthority.SERVER)

        assert validate_authority(
            caller_id=0,
            rpc_info=info,
            is_server=True
        )

    def test_client_authority_client_sending(self):
        """Test client can send client authority RPC."""
        info = RPCInfo(name="test", authority=RPCAuthority.CLIENT)

        assert validate_authority(
            caller_id=1,
            rpc_info=info,
            is_server=False
        )

    def test_owner_authority_requires_owner_id(self):
        """Test owner authority requires owner_id."""
        info = RPCInfo(name="test", authority=RPCAuthority.OWNER)

        with pytest.raises(ValidationError) as exc_info:
            validate_authority(
                caller_id=1,
                rpc_info=info,
                is_server=True,
                owner_id=None
            )

        assert "owner_id" in str(exc_info.value)

    def test_owner_authority_mismatch(self):
        """Test owner authority with wrong caller."""
        info = RPCInfo(name="test", authority=RPCAuthority.OWNER)

        with pytest.raises(ValidationError) as exc_info:
            validate_authority(
                caller_id=1,
                rpc_info=info,
                is_server=True,
                owner_id=2
            )

        assert "not owner" in str(exc_info.value)

    def test_owner_authority_match(self):
        """Test owner authority with correct caller."""
        info = RPCInfo(name="test", authority=RPCAuthority.OWNER)

        assert validate_authority(
            caller_id=1,
            rpc_info=info,
            is_server=True,
            owner_id=1
        )

    def test_multicast_server_only(self):
        """Test only server can multicast."""
        info = RPCInfo(name="test", authority=RPCAuthority.MULTICAST)

        assert validate_authority(
            caller_id=0,
            rpc_info=info,
            is_server=True
        )

        with pytest.raises(ValidationError):
            validate_authority(
                caller_id=1,
                rpc_info=info,
                is_server=False
            )


class TestRPCValidator:
    """Tests for RPCValidator class."""

    def test_validator_creation(self):
        """Test validator creation."""
        validator = RPCValidator(is_server=True)

        assert validator.is_server

    def test_validate_basic(self):
        """Test basic validation."""
        validator = RPCValidator(is_server=True)

        info = RPCInfo(name="test", authority=RPCAuthority.CLIENT)

        # Should pass - server receiving client RPC
        assert validator.validate(
            caller_id=1,
            rpc_info=info,
            args=()
        )

    def test_validate_with_rate_limit(self):
        """Test validation with rate limiting."""
        validator = RPCValidator(is_server=True)
        validator.rate_limiter = RateLimiter(RateLimitConfig(
            max_calls=2,
            window_seconds=1.0,
            burst_allowance=0
        ))

        info = RPCInfo(
            name="test",
            authority=RPCAuthority.CLIENT,
            rate_limit=2.0
        )

        # First two pass
        validator.validate(1, info, ())
        validator.validate(1, info, ())

        # Third fails
        with pytest.raises(ValidationError):
            validator.validate(1, info, ())

    def test_custom_validator(self):
        """Test custom validator registration."""
        validator = RPCValidator(is_server=True)

        def my_validator(caller_id, entity_id, args):
            return args[0] < 100  # First arg must be < 100

        validator.register_custom_validator("limited_rpc", my_validator)

        info = RPCInfo(name="limited_rpc", authority=RPCAuthority.CLIENT)

        # Should pass
        assert validator.validate(1, info, (50,))

        # Should fail
        with pytest.raises(ValidationError):
            validator.validate(1, info, (150,))

    def test_entity_owner_tracking(self):
        """Test entity owner tracking."""
        validator = RPCValidator(is_server=True)

        validator.set_entity_owner(entity_id=100, owner_id=5)

        assert validator.get_entity_owner(100) == 5

        info = RPCInfo(name="owner_rpc", authority=RPCAuthority.OWNER)

        # Owner should pass
        assert validator.validate(
            caller_id=5,
            rpc_info=info,
            args=(),
            entity_id=100
        )

        # Non-owner should fail
        with pytest.raises(ValidationError):
            validator.validate(
                caller_id=1,
                rpc_info=info,
                args=(),
                entity_id=100
            )


class TestParamValidation:
    """Tests for parameter validation helpers."""

    def test_validate_param_range(self):
        """Test range validation."""
        assert validate_param_range(50, min_val=0, max_val=100)

        with pytest.raises(ValidationError):
            validate_param_range(-5, min_val=0)

        with pytest.raises(ValidationError):
            validate_param_range(150, max_val=100)

    def test_validate_param_type(self):
        """Test type validation."""
        assert validate_param_type(42, int)
        assert validate_param_type("hello", str)

        with pytest.raises(ValidationError):
            validate_param_type("string", int)

    def test_validate_param_length(self):
        """Test length validation."""
        assert validate_param_length([1, 2, 3], min_len=1, max_len=5)
        assert validate_param_length("hello", max_len=10)

        with pytest.raises(ValidationError):
            validate_param_length([], min_len=1)

        with pytest.raises(ValidationError):
            validate_param_length("too long", max_len=5)


# =============================================================================
# Integration Tests
# =============================================================================

class TestRPCIntegration:
    """Integration tests for the complete RPC system."""

    def test_full_rpc_flow_server_to_client(self):
        """Test complete server to client RPC flow."""
        # Server manager
        server = RPCManager(is_server=True)

        sent_data = []
        server.set_rpc_callback(lambda d, t, r: sent_data.append((d, t, r)))

        server.register_rpc(
            lambda caller, msg: None,
            authority=RPCAuthority.SERVER,
            name="notify_client"
        )

        # Server calls RPC
        result = server.call_rpc(
            "notify_client",
            args=("Hello, client!",),
            target=1
        )

        assert result.success
        assert len(sent_data) == 1

        # Client receives
        client = RPCManager(is_server=False)

        received = []

        def client_handler(caller, msg):
            received.append(msg)

        client.register_rpc(
            client_handler,
            authority=RPCAuthority.SERVER,
            name="notify_client"
        )

        data, target, reliable = sent_data[0]
        success, _ = client.receive_rpc(data, MockCaller(), caller_id=0)

        assert success
        assert len(received) == 1
        assert received[0] == "Hello, client!"

    def test_full_rpc_flow_client_to_server(self):
        """Test complete client to server RPC flow."""
        # Client manager
        client = RPCManager(is_server=False)

        sent_data = []
        client.set_rpc_callback(lambda d, t, r: sent_data.append((d, t, r)))

        client.register_rpc(
            lambda caller, action_id: None,
            authority=RPCAuthority.CLIENT,
            name="request_action"
        )

        # Client calls RPC
        result = client.call_rpc(
            "request_action",
            args=(42,),
            target=0  # Server
        )

        assert result.success
        assert len(sent_data) == 1

        # Server receives
        server = RPCManager(is_server=True)

        received = []

        def server_handler(caller, action_id):
            received.append((caller, action_id))
            return True

        server.register_rpc(
            server_handler,
            authority=RPCAuthority.CLIENT,
            name="request_action"
        )

        data, target, reliable = sent_data[0]
        success, result = server.receive_rpc(data, MockCaller(player_id=1), caller_id=1)

        assert success
        assert len(received) == 1
        assert received[0][1] == 42

    def test_rpc_channel_integration(self):
        """Test RPC with channel layer."""
        # Server channel manager
        server_channels = RPCChannelManager()
        server_channel = server_channels.get_or_create_channel(1)

        # Client channel manager
        client_channels = RPCChannelManager()
        client_channel = client_channels.get_or_create_channel(0)

        # Send from server
        info = RPCInfo(name="test_rpc", authority=RPCAuthority.SERVER)
        server_channels.send_rpc(1, info, b"test_payload")

        # Get data
        data = server_channels.get_outgoing_data(1)
        assert len(data) > 0

        # Receive on client
        results = client_channels.receive_data(0, data)
        assert len(results) == 1

    def test_validated_rpc_flow(self):
        """Test RPC with full validation."""
        validator = RPCValidator(is_server=True)
        validator.rate_limiter = RateLimiter(RateLimitConfig(
            max_calls=5,
            window_seconds=1.0,
            burst_allowance=2
        ))

        info = RPCInfo(
            name="attack",
            authority=RPCAuthority.OWNER,
            rate_limit=5.0
        )

        # Set entity owner
        validator.set_entity_owner(entity_id=100, owner_id=1)

        # Valid call
        assert validator.validate(
            caller_id=1,
            rpc_info=info,
            args=(50,),  # damage
            entity_id=100
        )

        # Invalid - wrong owner
        with pytest.raises(ValidationError):
            validator.validate(
                caller_id=2,
                rpc_info=info,
                args=(50,),
                entity_id=100
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
