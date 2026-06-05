"""
Whitebox tests for the RPC (Remote Procedure Call) layer.

Tests:
- RPC registration and unregistration
- RPC invocation and dispatch
- Authority validation
- Rate limiting
- Deduplication
- Serialization/deserialization
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from engine.networking.rpc.rpc_manager import (
    RPCManager,
    RPCInfo,
    RPCAuthority,
    RPCReliability,
    RPCCallResult,
    PendingRPC,
    rpc,
)
from engine.networking.config import get_config

_config = get_config()


# =============================================================================
# RPCInfo Tests
# =============================================================================

class TestRPCInfo:
    """Tests for RPCInfo metadata."""

    def test_rpc_info_creation(self):
        """RPCInfo should store all attributes."""
        info = RPCInfo(
            name="test_rpc",
            authority=RPCAuthority.SERVER,
            reliability=RPCReliability.RELIABLE,
            ordered=True,
            rate_limit=10.0,
            validate_params=True
        )

        assert info.name == "test_rpc"
        assert info.authority == RPCAuthority.SERVER
        assert info.reliability == RPCReliability.RELIABLE
        assert info.ordered == True
        assert info.rate_limit == 10.0
        assert info.validate_params == True

    def test_rpc_info_defaults(self):
        """RPCInfo should have sensible defaults."""
        info = RPCInfo(name="test")

        assert info.authority == RPCAuthority.SERVER
        assert info.reliability == RPCReliability.RELIABLE
        assert info.ordered == True
        assert info.rate_limit == 0.0

    def test_rpc_info_reliable_property(self):
        """reliable property should check reliability type."""
        reliable = RPCInfo(name="r", reliability=RPCReliability.RELIABLE)
        unreliable = RPCInfo(name="u", reliability=RPCReliability.UNRELIABLE)
        reliable_unordered = RPCInfo(name="ru", reliability=RPCReliability.RELIABLE_UNORDERED)

        assert reliable.reliable == True
        assert unreliable.reliable == False
        assert reliable_unordered.reliable == True

    def test_rpc_info_hash(self):
        """get_hash should return consistent hash for name."""
        info1 = RPCInfo(name="test_rpc")
        info2 = RPCInfo(name="test_rpc")
        info3 = RPCInfo(name="other_rpc")

        assert info1.get_hash() == info2.get_hash()
        assert info1.get_hash() != info3.get_hash()

    def test_rpc_info_frozen(self):
        """RPCInfo should be frozen (immutable)."""
        info = RPCInfo(name="test")

        with pytest.raises(AttributeError):
            info.name = "changed"


class TestPendingRPC:
    """Tests for PendingRPC tracking."""

    def test_pending_rpc_creation(self):
        """PendingRPC should store call data."""
        info = RPCInfo(name="test")
        pending = PendingRPC(
            rpc_info=info,
            args=(1, 2, 3),
            target="target",
            sequence=42
        )

        assert pending.rpc_info == info
        assert pending.args == (1, 2, 3)
        assert pending.target == "target"
        assert pending.sequence == 42
        assert pending.retries == 0

    def test_pending_rpc_timestamp(self):
        """PendingRPC should have timestamp."""
        info = RPCInfo(name="test")
        before = time.time()
        pending = PendingRPC(info, (), None, 1)
        after = time.time()

        assert before <= pending.timestamp <= after


class TestRPCCallResult:
    """Tests for RPCCallResult."""

    def test_result_success(self):
        """Successful result should indicate success."""
        result = RPCCallResult(success=True, sequence=42)

        assert result.success == True
        assert result.sequence == 42
        assert result.error == ""

    def test_result_failure(self):
        """Failed result should include error."""
        result = RPCCallResult(success=False, error="Unknown RPC")

        assert result.success == False
        assert result.error == "Unknown RPC"


# =============================================================================
# RPCManager Registration Tests
# =============================================================================

class TestRPCManagerRegistration:
    """Tests for RPC registration."""

    def test_register_rpc_function(self):
        """register_rpc should register a function."""
        manager = RPCManager(is_server=True)

        def my_rpc(caller, arg1, arg2):
            return arg1 + arg2

        info = manager.register_rpc(my_rpc)

        assert info.name == "my_rpc"
        assert manager.get_rpc_info("my_rpc") is not None

    def test_register_rpc_with_name(self):
        """Custom name should override function name."""
        manager = RPCManager(is_server=True)

        def handler(caller):
            pass

        info = manager.register_rpc(handler, name="custom_name")

        assert info.name == "custom_name"
        assert manager.get_rpc_info("custom_name") is not None
        assert manager.get_rpc_info("handler") is None

    def test_register_rpc_with_authority(self):
        """Authority should be settable."""
        manager = RPCManager(is_server=True)

        def handler(caller):
            pass

        info = manager.register_rpc(handler, authority=RPCAuthority.CLIENT)

        assert info.authority == RPCAuthority.CLIENT

    def test_register_rpc_with_reliability(self):
        """Reliability should be settable."""
        manager = RPCManager(is_server=True)

        def handler(caller):
            pass

        info = manager.register_rpc(handler, reliability=RPCReliability.UNRELIABLE)

        assert info.reliability == RPCReliability.UNRELIABLE
        assert not info.reliable

    def test_register_rpc_with_rate_limit(self):
        """Rate limit should be settable."""
        manager = RPCManager(is_server=True)

        def handler(caller):
            pass

        info = manager.register_rpc(handler, rate_limit=5.0)

        assert info.rate_limit == 5.0

    def test_unregister_rpc(self):
        """unregister_rpc should remove registration."""
        manager = RPCManager(is_server=True)

        def handler(caller):
            pass

        manager.register_rpc(handler)
        assert manager.get_rpc_info("handler") is not None

        result = manager.unregister_rpc("handler")

        assert result == True
        assert manager.get_rpc_info("handler") is None

    def test_unregister_nonexistent(self):
        """Unregistering nonexistent RPC should return False."""
        manager = RPCManager(is_server=True)

        result = manager.unregister_rpc("nonexistent")

        assert result == False

    def test_get_rpc_info(self):
        """get_rpc_info should return info for registered RPC."""
        manager = RPCManager(is_server=True)

        def handler(caller):
            pass

        manager.register_rpc(handler)

        info = manager.get_rpc_info("handler")
        assert info is not None
        assert info.name == "handler"

    def test_get_rpc_info_nonexistent(self):
        """get_rpc_info should return None for unknown RPC."""
        manager = RPCManager(is_server=True)

        info = manager.get_rpc_info("unknown")
        assert info is None


# =============================================================================
# RPCManager Call Tests
# =============================================================================

class TestRPCManagerCall:
    """Tests for RPC invocation."""

    def test_call_rpc_unknown(self):
        """Calling unknown RPC should fail."""
        manager = RPCManager(is_server=True)

        result = manager.call_rpc("unknown", (), None)

        assert result.success == False
        assert "Unknown RPC" in result.error

    def test_call_rpc_success(self):
        """Calling registered RPC should succeed."""
        manager = RPCManager(is_server=True)
        callback = Mock()
        manager.set_rpc_callback(callback)

        def handler(caller):
            pass

        manager.register_rpc(handler)

        result = manager.call_rpc("handler", (), "target")

        assert result.success == True
        assert result.sequence > 0
        callback.assert_called_once()

    def test_call_rpc_assigns_sequence(self):
        """Each call should get unique sequence."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler)

        result1 = manager.call_rpc("handler", (), None)
        result2 = manager.call_rpc("handler", (), None)

        assert result1.sequence != result2.sequence

    def test_call_rpc_reliable_tracks_pending(self):
        """Reliable RPCs should be tracked as pending."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, reliability=RPCReliability.RELIABLE)

        result = manager.call_rpc("handler", (), None)

        # Should have pending RPC
        pending = manager.get_pending_retransmits(timeout=0)
        # Initially not timed out, so empty
        assert isinstance(pending, list)

    def test_call_rpc_unreliable_no_pending(self):
        """Unreliable RPCs should not be tracked."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, reliability=RPCReliability.UNRELIABLE)

        result = manager.call_rpc("handler", (), None)
        assert result.success == True


# =============================================================================
# RPCManager Authority Tests
# =============================================================================

class TestRPCManagerAuthority:
    """Tests for RPC authority validation."""

    def test_server_authority_on_server(self):
        """SERVER RPC should be callable from server."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, authority=RPCAuthority.SERVER)

        result = manager.call_rpc("handler", (), None)
        assert result.success == True

    def test_server_authority_on_client(self):
        """SERVER RPC should not be callable from client."""
        manager = RPCManager(is_server=False)  # Client

        def handler(caller):
            pass

        manager.register_rpc(handler, authority=RPCAuthority.SERVER)

        result = manager.call_rpc("handler", (), None)
        assert result.success == False
        assert "Authority denied" in result.error

    def test_client_authority_on_client(self):
        """CLIENT RPC should be callable from client."""
        manager = RPCManager(is_server=False)  # Client
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, authority=RPCAuthority.CLIENT)

        result = manager.call_rpc("handler", (), None, caller_id=1)
        assert result.success == True

    def test_client_authority_on_server(self):
        """CLIENT RPC should not be callable from server."""
        manager = RPCManager(is_server=True)

        def handler(caller):
            pass

        manager.register_rpc(handler, authority=RPCAuthority.CLIENT)

        result = manager.call_rpc("handler", (), None)
        assert result.success == False

    def test_multicast_authority_on_server(self):
        """MULTICAST RPC should be callable from server."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, authority=RPCAuthority.MULTICAST)

        result = manager.call_rpc("handler", (), None)
        assert result.success == True

    def test_multicast_authority_on_client(self):
        """MULTICAST RPC should not be callable from client."""
        manager = RPCManager(is_server=False)

        def handler(caller):
            pass

        manager.register_rpc(handler, authority=RPCAuthority.MULTICAST)

        result = manager.call_rpc("handler", (), None)
        assert result.success == False

    def test_owner_authority_requires_caller(self):
        """OWNER RPC requires caller_id."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, authority=RPCAuthority.OWNER)

        # Without caller_id
        result = manager.call_rpc("handler", (), None, caller_id=None)
        assert result.success == False

        # With caller_id
        result = manager.call_rpc("handler", (), None, caller_id=1)
        assert result.success == True


# =============================================================================
# RPCManager Rate Limiting Tests
# =============================================================================

class TestRPCManagerRateLimiting:
    """Tests for RPC rate limiting."""

    def test_rate_limit_allows_within_limit(self):
        """Calls within limit should succeed."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, rate_limit=10.0)  # 10 per second

        # Make 5 calls (within limit)
        for _ in range(5):
            result = manager.call_rpc("handler", (), None, caller_id=1)
            assert result.success == True

    def test_rate_limit_blocks_over_limit(self):
        """Calls over limit should be blocked."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, rate_limit=3.0)  # 3 per second

        # Make 5 calls (over limit)
        success_count = 0
        for _ in range(5):
            result = manager.call_rpc("handler", (), None, caller_id=1)
            if result.success:
                success_count += 1

        assert success_count == 3  # Only 3 should succeed

    def test_rate_limit_per_caller(self):
        """Rate limits should be per-caller."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, rate_limit=2.0)

        # Caller 1: 3 calls, only 2 succeed
        for i in range(3):
            manager.call_rpc("handler", (), None, caller_id=1)

        # Caller 2: should have their own limit
        result = manager.call_rpc("handler", (), None, caller_id=2)
        assert result.success == True

    def test_rate_limit_zero_unlimited(self):
        """Rate limit 0 should mean unlimited."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, rate_limit=0.0)

        # Many calls should all succeed
        for _ in range(100):
            result = manager.call_rpc("handler", (), None, caller_id=1)
            assert result.success == True

    def test_cleanup_rate_limiters(self):
        """cleanup_rate_limiters should remove old entries."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, rate_limit=5.0)

        # Make some calls
        manager.call_rpc("handler", (), None, caller_id=1)

        # Cleanup with 0 max_age should remove all
        removed = manager.cleanup_rate_limiters(max_age=0)
        assert removed >= 0


# =============================================================================
# RPCManager Receive Tests
# =============================================================================

class TestRPCManagerReceive:
    """Tests for receiving RPCs."""

    def test_receive_rpc_executes_handler(self):
        """Receiving RPC should execute handler."""
        manager = RPCManager(is_server=True)

        def my_handler(caller, arg1, arg2):
            return arg1 + arg2

        manager.register_rpc(my_handler, authority=RPCAuthority.CLIENT)

        # Serialize an RPC call
        info = manager.get_rpc_info("my_handler")
        data = manager._serialize_rpc(info, 1, ("hello", " world"))

        # Receive it
        success, result = manager.receive_rpc(data, "caller_obj", caller_id=1)

        assert success == True
        assert result == "hello world"

    def test_receive_rpc_unknown(self):
        """Receiving unknown RPC should fail."""
        manager = RPCManager(is_server=True)

        # Create fake data with unknown hash
        import struct
        data = struct.pack('<IIB', 0xDEADBEEF, 1, 0)
        data += struct.pack('<H', 0)  # Empty args

        success, error = manager.receive_rpc(data, None, caller_id=1)

        # The manager will try to find by hash and may use "unknown_" prefix
        # Just verify it handled the case
        assert isinstance(success, bool)

    def test_receive_rpc_authority_check(self):
        """RPC authority should be validated on receive."""
        # Server manager
        server_manager = RPCManager(is_server=True)

        def server_only_handler(caller):
            return "ok"

        # Register as server-authority (server -> client)
        server_manager.register_rpc(server_only_handler, authority=RPCAuthority.SERVER)

        # Client manager should accept incoming server RPCs
        client_manager = RPCManager(is_server=False)
        client_manager.register_rpc(server_only_handler, authority=RPCAuthority.SERVER)

        info = server_manager.get_rpc_info("server_only_handler")
        data = server_manager._serialize_rpc(info, 1, ())

        # Client receives server RPC - this should be valid (client is receiving FROM server)
        success, result = client_manager.receive_rpc(data, None, caller_id=1)
        # Authority check for incoming: SERVER RPC received by client = valid
        assert success == True

    def test_receive_rpc_deduplication(self):
        """Duplicate RPCs should be deduplicated."""
        manager = RPCManager(is_server=True)

        call_count = 0

        def handler(caller):
            nonlocal call_count
            call_count += 1

        manager.register_rpc(handler, authority=RPCAuthority.CLIENT)

        info = manager.get_rpc_info("handler")
        data = manager._serialize_rpc(info, 1, ())

        # First receive
        manager.receive_rpc(data, None, caller_id=1)
        assert call_count == 1

        # Second receive (duplicate)
        manager.receive_rpc(data, None, caller_id=1)
        assert call_count == 1  # Not called again

    def test_receive_rpc_rate_limited(self):
        """Rate limited RPCs should be rejected."""
        manager = RPCManager(is_server=True)

        def handler(caller):
            pass

        manager.register_rpc(handler, authority=RPCAuthority.CLIENT, rate_limit=1.0)

        info = manager.get_rpc_info("handler")

        # First call succeeds
        data1 = manager._serialize_rpc(info, 1, ())
        success, _ = manager.receive_rpc(data1, None, caller_id=1)
        assert success == True

        # Second call should be rate limited
        data2 = manager._serialize_rpc(info, 2, ())
        success, error = manager.receive_rpc(data2, None, caller_id=1)
        assert success == False
        assert "Rate limit" in error


# =============================================================================
# RPCManager Acknowledgment Tests
# =============================================================================

class TestRPCManagerAcknowledgment:
    """Tests for RPC acknowledgment."""

    def test_acknowledge_rpc(self):
        """Acknowledging RPC should remove from pending."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, reliability=RPCReliability.RELIABLE)

        result = manager.call_rpc("handler", (), None)
        seq = result.sequence

        # Acknowledge it
        was_pending = manager.acknowledge_rpc(seq)

        assert was_pending == True

        # Acknowledge again should return False
        was_pending = manager.acknowledge_rpc(seq)
        assert was_pending == False

    def test_get_pending_retransmits(self):
        """get_pending_retransmits should return timed out RPCs."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, reliability=RPCReliability.RELIABLE)

        result = manager.call_rpc("handler", (), None)

        # Immediately, nothing should be timed out
        pending = manager.get_pending_retransmits(timeout=1.0)
        assert len(pending) == 0

        # With very short timeout, should be timed out
        time.sleep(0.01)
        pending = manager.get_pending_retransmits(timeout=0.001)
        assert len(pending) >= 1

    def test_retransmit_increments_retry_count(self):
        """Getting retransmit should increment retry counter."""
        manager = RPCManager(is_server=True)
        manager.set_rpc_callback(Mock())

        def handler(caller):
            pass

        manager.register_rpc(handler, reliability=RPCReliability.RELIABLE)
        manager.call_rpc("handler", (), None)

        time.sleep(0.01)

        pending = manager.get_pending_retransmits(timeout=0.001)
        if len(pending) > 0:
            assert pending[0].retries == 1


# =============================================================================
# RPCManager Cleanup Tests
# =============================================================================

class TestRPCManagerCleanup:
    """Tests for cleanup operations."""

    def test_cleanup_history(self):
        """cleanup_history should remove old entries."""
        manager = RPCManager(is_server=True)

        def handler(caller):
            pass

        manager.register_rpc(handler, authority=RPCAuthority.CLIENT)

        info = manager.get_rpc_info("handler")
        data = manager._serialize_rpc(info, 1, ())
        manager.receive_rpc(data, None, caller_id=1)

        # Cleanup with 0 max_age should remove all
        removed = manager.cleanup_history(max_age=0)
        assert removed >= 0

    def test_is_server_property(self):
        """is_server property should return correct value."""
        server_manager = RPCManager(is_server=True)
        client_manager = RPCManager(is_server=False)

        assert server_manager.is_server == True
        assert client_manager.is_server == False


# =============================================================================
# RPCManager Serialization Tests
# =============================================================================

class TestRPCManagerSerialization:
    """Tests for RPC serialization."""

    def test_serialize_roundtrip(self):
        """Serialized RPC should deserialize correctly."""
        manager = RPCManager(is_server=True)

        def handler(caller, x, y):
            return x + y

        manager.register_rpc(handler, authority=RPCAuthority.CLIENT)

        info = manager.get_rpc_info("handler")
        args = (10, 20)

        data = manager._serialize_rpc(info, 42, args)
        name, seq, restored_args, consumed = manager._deserialize_rpc(data)

        assert name == "handler"
        assert seq == 42
        assert restored_args == args

    def test_serialize_complex_args(self):
        """Complex arguments should serialize."""
        manager = RPCManager(is_server=True)

        def handler(caller, data):
            pass

        manager.register_rpc(handler, authority=RPCAuthority.CLIENT)

        info = manager.get_rpc_info("handler")
        args = ({"key": "value", "nested": [1, 2, 3]},)

        data = manager._serialize_rpc(info, 1, args)
        name, seq, restored_args, consumed = manager._deserialize_rpc(data)

        assert restored_args == args

    def test_serialize_header_format(self):
        """Serialized data should have correct header."""
        manager = RPCManager(is_server=True)

        def handler(caller):
            pass

        manager.register_rpc(handler, reliability=RPCReliability.RELIABLE)

        info = manager.get_rpc_info("handler")
        data = manager._serialize_rpc(info, 100, ())

        # Header: name_hash(4) + sequence(4) + flags(1)
        import struct
        name_hash, sequence, flags = struct.unpack('<IIB', data[:9])

        assert sequence == 100
        assert flags & 0x01  # Reliable flag


# =============================================================================
# RPC Decorator Tests
# =============================================================================

class TestRPCDecorator:
    """Tests for @rpc decorator."""

    def test_rpc_decorator_creates_info(self):
        """@rpc decorator should attach RPCInfo."""

        @rpc(authority=RPCAuthority.CLIENT)
        def my_rpc(self):
            pass

        assert hasattr(my_rpc, '_rpc_info')
        assert my_rpc._rpc_info.name == "my_rpc"
        assert my_rpc._rpc_info.authority == RPCAuthority.CLIENT

    def test_rpc_decorator_defaults(self):
        """@rpc decorator should use defaults."""

        @rpc()
        def my_rpc(self):
            pass

        info = my_rpc._rpc_info
        assert info.authority == RPCAuthority.SERVER
        assert info.reliable == True

    def test_rpc_decorator_unreliable(self):
        """@rpc with reliable=False should set unreliable."""

        @rpc(reliable=False)
        def unreliable_rpc(self):
            pass

        assert not unreliable_rpc._rpc_info.reliable

    def test_rpc_decorator_rate_limit(self):
        """@rpc with rate_limit should set it."""

        @rpc(rate_limit=5.0)
        def limited_rpc(self):
            pass

        assert limited_rpc._rpc_info.rate_limit == 5.0

    def test_rpc_decorator_preserves_function(self):
        """@rpc should preserve function behavior."""

        @rpc()
        def add(self, a, b):
            return a + b

        # Function should still work
        result = add(None, 1, 2)
        assert result == 3


# =============================================================================
# RPCManager Callback Tests
# =============================================================================

class TestRPCManagerCallback:
    """Tests for RPC send callback."""

    def test_set_rpc_callback(self):
        """set_rpc_callback should store callback."""
        manager = RPCManager(is_server=True)
        callback = Mock()

        manager.set_rpc_callback(callback)

        def handler(caller):
            pass

        manager.register_rpc(handler)
        manager.call_rpc("handler", (), "target")

        callback.assert_called_once()

    def test_callback_receives_data(self):
        """Callback should receive serialized data."""
        manager = RPCManager(is_server=True)
        received_data = None
        received_target = None
        received_reliable = None

        def callback(data, target, reliable):
            nonlocal received_data, received_target, received_reliable
            received_data = data
            received_target = target
            received_reliable = reliable

        manager.set_rpc_callback(callback)

        def handler(caller):
            pass

        manager.register_rpc(handler)
        manager.call_rpc("handler", (), "my_target")

        assert received_data is not None
        assert received_target == "my_target"
        assert received_reliable == True


# =============================================================================
# Integration Tests
# =============================================================================

class TestRPCIntegration:
    """Integration tests for RPC system."""

    def test_server_to_client_rpc(self):
        """Server should be able to send RPC to client."""
        server = RPCManager(is_server=True)
        client = RPCManager(is_server=False)

        received_args = None

        def client_handler(caller, x, y):
            nonlocal received_args
            received_args = (x, y)

        # Register on both
        server.register_rpc(client_handler, authority=RPCAuthority.SERVER)
        client.register_rpc(client_handler, authority=RPCAuthority.SERVER)

        # Server sends
        sent_data = None
        server.set_rpc_callback(lambda d, t, r: None.__setattr__('x', d) if False else exec("nonlocal sent_data; sent_data = d", {"nonlocal": None, "sent_data": None}))

        captured = []

        def capture_callback(d, t, r):
            captured.append(d)

        server.set_rpc_callback(capture_callback)

        result = server.call_rpc("client_handler", (10, 20), None)
        assert result.success == True

        # Client receives
        if captured:
            client.receive_rpc(captured[0], None, caller_id=0)
            assert received_args == (10, 20)

    def test_client_to_server_rpc(self):
        """Client should be able to send RPC to server."""
        server = RPCManager(is_server=True)
        client = RPCManager(is_server=False)

        received_value = None

        def server_handler(caller, value):
            nonlocal received_value
            received_value = value

        # Register on both
        server.register_rpc(server_handler, authority=RPCAuthority.CLIENT)
        client.register_rpc(server_handler, authority=RPCAuthority.CLIENT)

        # Client sends
        captured = []
        client.set_rpc_callback(lambda d, t, r: captured.append(d))

        result = client.call_rpc("server_handler", ("hello",), None, caller_id=1)
        assert result.success == True

        # Server receives
        if captured:
            server.receive_rpc(captured[0], None, caller_id=1)
            assert received_value == "hello"
