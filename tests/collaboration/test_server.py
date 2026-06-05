"""Tests for T-CC-4.6: Collaboration server with operation log.

Tests cover:
- CollaborationServer: Document management, client management, operations
- ServerOperationLog: Append, query, persistence, snapshots
- ClientSession: State management, subscriptions, reconnection
- SyncProtocol: All message types, error handling
- CollaborationClient: Connection, subscription, push/pull
- UndoManager: Undo/redo stack operations
- Edge cases: Concurrent operations, reconnection, cleanup
"""
import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from engine.collaboration import (
    # CRDT
    CRDTDocument,
    CRDTOperation,
    OperationType,
    VectorClock,
    # Server
    CollaborationServer,
    AsyncCollaborationServer,
    CollaborationClient,
    ClientSession,
    SessionState,
    ServerOperationLog,
    UndoManager,
    UndoEntry,
    SyncProtocol,
    SyncMessage,
    SyncMessageType,
    # Exceptions
    ServerError,
    SessionError,
    SyncError,
    PersistenceError,
    DocumentNotFoundError,
    ClientNotConnectedError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def server():
    """Create a collaboration server."""
    srv = CollaborationServer(server_id="test-server")
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def server_with_doc(server):
    """Create a server with a test document."""
    server.create_document("test-doc", {"title": "Test Document", "count": 0})
    return server


@pytest.fixture
def client(server_with_doc):
    """Create a connected client."""
    client = CollaborationClient(client_id="test-client", server=server_with_doc)
    client.connect()
    yield client
    if client.is_connected:
        client.disconnect()


@pytest.fixture
def temp_persist_path():
    """Create a temporary directory for persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def server_with_persistence(temp_persist_path):
    """Create a server with persistence enabled."""
    srv = CollaborationServer(
        server_id="persist-server",
        persist_path=temp_persist_path,
    )
    srv.start()
    yield srv
    srv.stop()


# =============================================================================
# Test: CollaborationServer - Document Management
# =============================================================================


class TestServerDocumentManagement:
    """Tests for server document management."""

    def test_create_document(self, server):
        """Test creating a document."""
        doc = server.create_document("doc1", {"name": "Test"})

        assert doc is not None
        assert doc.doc_id == "doc1"
        assert doc.get_register("name") == "Test"

    def test_create_document_without_initial_data(self, server):
        """Test creating document without initial data."""
        doc = server.create_document("empty-doc")

        assert doc is not None
        assert doc.doc_id == "empty-doc"

    def test_create_duplicate_document_fails(self, server):
        """Test that creating duplicate document raises error."""
        server.create_document("doc1")

        with pytest.raises(ServerError):
            server.create_document("doc1")

    def test_get_document(self, server_with_doc):
        """Test getting a document."""
        doc = server_with_doc.get_document("test-doc")

        assert doc is not None
        assert doc.doc_id == "test-doc"

    def test_get_nonexistent_document_raises(self, server):
        """Test that getting nonexistent document raises error."""
        with pytest.raises(DocumentNotFoundError):
            server.get_document("nonexistent")

    def test_list_documents(self, server):
        """Test listing documents."""
        server.create_document("doc1")
        server.create_document("doc2")
        server.create_document("doc3")

        docs = server.list_documents()

        assert len(docs) == 3
        assert set(docs) == {"doc1", "doc2", "doc3"}

    def test_delete_document(self, server):
        """Test deleting a document."""
        server.create_document("to-delete")

        assert server.document_exists("to-delete")
        result = server.delete_document("to-delete")

        assert result is True
        assert not server.document_exists("to-delete")

    def test_delete_nonexistent_document(self, server):
        """Test deleting nonexistent document returns False."""
        result = server.delete_document("nonexistent")

        assert result is False

    def test_document_exists(self, server_with_doc):
        """Test document existence check."""
        assert server_with_doc.document_exists("test-doc")
        assert not server_with_doc.document_exists("nonexistent")

    def test_get_document_clock(self, server_with_doc):
        """Test getting document vector clock."""
        clock = server_with_doc.get_document_clock("test-doc")

        assert isinstance(clock, VectorClock)


# =============================================================================
# Test: CollaborationServer - Client Management
# =============================================================================


class TestServerClientManagement:
    """Tests for server client management."""

    def test_connect_client(self, server):
        """Test connecting a client."""
        session = server.connect_client("client1", {"user": "test"})

        assert session is not None
        assert session.client_id == "client1"
        assert session.state == SessionState.CONNECTED
        assert session.metadata["user"] == "test"

    def test_connect_client_multiple_times(self, server):
        """Test connecting same client multiple times."""
        session1 = server.connect_client("client1")
        session2 = server.connect_client("client1")

        # Should return same session
        assert session1.session_id == session2.session_id

    def test_disconnect_client(self, server):
        """Test disconnecting a client."""
        server.connect_client("client1")
        server.disconnect_client("client1")

        session = server.get_session("client1")
        assert session.state == SessionState.DISCONNECTED

    def test_disconnect_nonexistent_client(self, server):
        """Test disconnecting nonexistent client is no-op."""
        # Should not raise
        server.disconnect_client("nonexistent")

    def test_get_session(self, server):
        """Test getting a client session."""
        server.connect_client("client1")

        session = server.get_session("client1")

        assert session is not None
        assert session.client_id == "client1"

    def test_get_session_nonexistent(self, server):
        """Test getting nonexistent session returns None."""
        session = server.get_session("nonexistent")

        assert session is None

    def test_list_clients(self, server):
        """Test listing connected clients."""
        server.connect_client("client1")
        server.connect_client("client2")
        server.connect_client("client3")
        server.disconnect_client("client3")

        clients = server.list_clients()

        assert len(clients) == 2
        assert set(clients) == {"client1", "client2"}

    def test_is_client_connected(self, server):
        """Test checking if client is connected."""
        server.connect_client("client1")

        assert server.is_client_connected("client1")
        assert not server.is_client_connected("nonexistent")


# =============================================================================
# Test: CollaborationServer - Subscriptions
# =============================================================================


class TestServerSubscriptions:
    """Tests for server subscription management."""

    def test_subscribe_client(self, server_with_doc):
        """Test subscribing client to document."""
        server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")

        subs = server_with_doc.get_subscribers("test-doc")

        assert "client1" in subs

    def test_subscribe_not_connected_raises(self, server_with_doc):
        """Test subscribing disconnected client raises error."""
        with pytest.raises(ClientNotConnectedError):
            server_with_doc.subscribe_client("nonexistent", "test-doc")

    def test_subscribe_to_nonexistent_doc_raises(self, server):
        """Test subscribing to nonexistent document raises error."""
        server.connect_client("client1")

        with pytest.raises(DocumentNotFoundError):
            server.subscribe_client("client1", "nonexistent")

    def test_unsubscribe_client(self, server_with_doc):
        """Test unsubscribing client from document."""
        server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")
        server_with_doc.unsubscribe_client("client1", "test-doc")

        subs = server_with_doc.get_subscribers("test-doc")

        assert "client1" not in subs

    def test_get_subscribers(self, server_with_doc):
        """Test getting document subscribers."""
        server_with_doc.connect_client("client1")
        server_with_doc.connect_client("client2")
        server_with_doc.subscribe_client("client1", "test-doc")
        server_with_doc.subscribe_client("client2", "test-doc")

        subs = server_with_doc.get_subscribers("test-doc")

        assert len(subs) == 2
        assert {"client1", "client2"} == subs

    def test_get_subscriptions(self, server_with_doc):
        """Test getting client subscriptions."""
        server_with_doc.create_document("doc2")
        server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")
        server_with_doc.subscribe_client("client1", "doc2")

        subs = server_with_doc.get_subscriptions("client1")

        assert len(subs) == 2
        assert {"test-doc", "doc2"} == subs


# =============================================================================
# Test: CollaborationServer - Operations
# =============================================================================


class TestServerOperations:
    """Tests for server operation handling."""

    def test_apply_operations(self, server_with_doc):
        """Test applying operations to document."""
        doc = server_with_doc.get_document("test-doc")
        clock = VectorClock()
        clock.increment("node1")

        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.title",
            value={"value": "New Title", "timestamp": time.time(), "node_id": "node1"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock,
        )

        applied = server_with_doc.apply_operations("test-doc", [op])

        assert applied == 1
        assert doc.get_register("title") == "New Title"

    def test_apply_duplicate_operation(self, server_with_doc):
        """Test applying duplicate operation is idempotent."""
        clock = VectorClock()
        clock.increment("node1")

        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.title",
            value={"value": "Title", "timestamp": time.time(), "node_id": "node1"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock,
        )

        applied1 = server_with_doc.apply_operations("test-doc", [op])
        applied2 = server_with_doc.apply_operations("test-doc", [op])

        assert applied1 == 1
        assert applied2 == 0  # Duplicate not reapplied

    def test_apply_operations_to_nonexistent_doc(self, server):
        """Test applying operations to nonexistent document raises error."""
        with pytest.raises(DocumentNotFoundError):
            server.apply_operations("nonexistent", [])

    def test_get_operations_since(self, server_with_doc):
        """Test getting operations since a clock state."""
        doc = server_with_doc.get_document("test-doc")
        initial_clock = doc.clock.copy()

        # Apply some operations via server (so they go to ServerOperationLog)
        clock1 = VectorClock()
        clock1.increment("node1")
        op1 = CRDTOperation(
            id="op-since-1",
            type=OperationType.SET,
            path="registers.key1",
            value={"value": "value1", "timestamp": time.time(), "node_id": "node1"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock1,
        )

        clock2 = VectorClock()
        clock2.set("node1", 2)
        op2 = CRDTOperation(
            id="op-since-2",
            type=OperationType.SET,
            path="registers.key2",
            value={"value": "value2", "timestamp": time.time(), "node_id": "node1"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock2,
        )

        server_with_doc.apply_operations("test-doc", [op1, op2])

        # Get operations since initial clock
        ops = server_with_doc.get_operations_since("test-doc", initial_clock)

        assert len(ops) >= 2

    def test_get_operation_log(self, server_with_doc):
        """Test getting operation log for document."""
        log = server_with_doc.get_operation_log("test-doc")

        assert isinstance(log, ServerOperationLog)
        assert log.doc_id == "test-doc"


# =============================================================================
# Test: CollaborationServer - Reconnection
# =============================================================================


class TestServerReconnection:
    """Tests for client reconnection."""

    def test_reconnect_client(self, server_with_doc):
        """Test client reconnection."""
        # Connect and subscribe
        session = server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")
        token = session.reconnect_token
        initial_clock = session.last_sync_clock.copy()

        # Disconnect
        server_with_doc.disconnect_client("client1")

        # Apply operations while disconnected (via server to add to ServerOperationLog)
        clock = VectorClock()
        clock.increment("server")
        op = CRDTOperation(
            id="offline-op-1",
            type=OperationType.SET,
            path="registers.offline_key",
            value={"value": "offline_value", "timestamp": time.time(), "node_id": "server"},
            timestamp=time.time(),
            node_id="server",
            clock=clock,
        )
        server_with_doc.apply_operations("test-doc", [op])

        # Reconnect
        new_session, missed_ops = server_with_doc.reconnect_client(
            "client1",
            token,
            initial_clock,
        )

        assert new_session.state == SessionState.CONNECTED
        assert len(missed_ops) >= 1

    def test_reconnect_with_invalid_token(self, server):
        """Test reconnection with invalid token fails."""
        server.connect_client("client1")
        server.disconnect_client("client1")

        with pytest.raises(SessionError):
            server.reconnect_client("client1", "invalid-token")

    def test_reconnect_no_session(self, server):
        """Test reconnection without existing session fails."""
        with pytest.raises(SessionError):
            server.reconnect_client("nonexistent", "token")


# =============================================================================
# Test: CollaborationServer - Lifecycle
# =============================================================================


class TestServerLifecycle:
    """Tests for server lifecycle management."""

    def test_start_stop(self):
        """Test server start and stop."""
        server = CollaborationServer()

        assert not server.is_running

        server.start()
        assert server.is_running

        server.stop()
        assert not server.is_running

    def test_stop_disconnects_clients(self, server):
        """Test stopping server disconnects all clients."""
        server.connect_client("client1")
        server.connect_client("client2")

        server.stop()

        # Sessions should be disconnected
        assert not server.is_client_connected("client1")
        assert not server.is_client_connected("client2")

    def test_cleanup_stale_sessions(self, server):
        """Test cleanup of stale sessions."""
        session = server.connect_client("client1")
        # Simulate stale heartbeat
        session.last_heartbeat = time.time() - 100

        cleaned = server.cleanup_stale_sessions()

        assert cleaned == 1
        assert not server.is_client_connected("client1")

    def test_get_stats(self, server_with_doc):
        """Test getting server statistics."""
        server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")

        stats = server_with_doc.get_stats()

        assert stats["running"] is True
        assert stats["documents"] == 1
        assert stats["connected_clients"] == 1


# =============================================================================
# Test: ServerOperationLog
# =============================================================================


class TestServerOperationLog:
    """Tests for server operation log."""

    def test_append_operation(self):
        """Test appending operation to log."""
        log = ServerOperationLog("test-doc")
        clock = VectorClock()
        clock.increment("node1")

        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "test"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock,
        )

        seq = log.append(op)

        assert seq == 1
        assert len(log) == 1

    def test_append_batch(self):
        """Test appending batch of operations."""
        log = ServerOperationLog("test-doc")
        ops = []

        for i in range(5):
            clock = VectorClock()
            clock.set("node1", i + 1)
            ops.append(CRDTOperation(
                id=f"op{i}",
                type=OperationType.SET,
                path=f"registers.key{i}",
                value={"value": i},
                timestamp=time.time(),
                node_id="node1",
                clock=clock,
            ))

        seqs = log.append_batch(ops)

        assert len(seqs) == 5
        assert len(log) == 5

    def test_append_duplicate_returns_existing_seq(self):
        """Test appending duplicate operation returns existing sequence."""
        log = ServerOperationLog("test-doc")
        clock = VectorClock()
        clock.increment("node1")

        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "test"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock,
        )

        seq1 = log.append(op)
        seq2 = log.append(op)

        assert seq1 == seq2
        assert len(log) == 1

    def test_get_operation(self):
        """Test getting operation by ID."""
        log = ServerOperationLog("test-doc")
        clock = VectorClock()
        clock.increment("node1")

        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "test"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock,
        )

        log.append(op)
        retrieved = log.get_operation("op1")

        assert retrieved is not None
        assert retrieved.id == "op1"

    def test_get_nonexistent_operation(self):
        """Test getting nonexistent operation returns None."""
        log = ServerOperationLog("test-doc")

        retrieved = log.get_operation("nonexistent")

        assert retrieved is None

    def test_get_operations_since_clock(self):
        """Test getting operations since a clock state."""
        log = ServerOperationLog("test-doc")

        # Add operations
        for i in range(5):
            clock = VectorClock()
            clock.set("node1", i + 1)
            op = CRDTOperation(
                id=f"op{i}",
                type=OperationType.SET,
                path=f"registers.key{i}",
                value={"value": i},
                timestamp=time.time(),
                node_id="node1",
                clock=clock,
            )
            log.append(op)

        # Get operations since clock(2)
        since_clock = VectorClock()
        since_clock.set("node1", 2)

        ops = log.get_operations_since(since_clock)

        # Should get ops with clock > 2
        assert len(ops) >= 2

    def test_get_operations_by_node(self):
        """Test getting operations by node ID."""
        log = ServerOperationLog("test-doc")

        # Add operations from different nodes
        for node in ["node1", "node2", "node1"]:
            clock = VectorClock()
            clock.increment(node)
            op = CRDTOperation(
                id=f"op-{node}-{time.time()}",
                type=OperationType.SET,
                path="registers.key",
                value={"value": "test"},
                timestamp=time.time(),
                node_id=node,
                clock=clock,
            )
            log.append(op)

        ops = log.get_operations_by_node("node1")

        assert len(ops) == 2

    def test_get_operations_by_path(self):
        """Test getting operations by path prefix."""
        log = ServerOperationLog("test-doc")

        # Add operations with different paths
        for path in ["registers.a", "registers.b", "counters.c"]:
            clock = VectorClock()
            clock.increment("node1")
            op = CRDTOperation(
                id=f"op-{path}",
                type=OperationType.SET,
                path=path,
                value={"value": "test"},
                timestamp=time.time(),
                node_id="node1",
                clock=clock,
            )
            log.append(op)

        ops = log.get_operations_by_path("registers")

        assert len(ops) == 2

    def test_get_last_n_operations(self):
        """Test getting last N operations."""
        log = ServerOperationLog("test-doc")

        # Add 10 operations
        for i in range(10):
            clock = VectorClock()
            clock.set("node1", i + 1)
            op = CRDTOperation(
                id=f"op{i}",
                type=OperationType.SET,
                path="registers.key",
                value={"value": i},
                timestamp=time.time(),
                node_id="node1",
                clock=clock,
            )
            log.append(op)

        ops = log.get_last_n_operations(3)

        assert len(ops) == 3
        assert ops[-1].id == "op9"

    def test_clear(self):
        """Test clearing the log."""
        log = ServerOperationLog("test-doc")

        clock = VectorClock()
        clock.increment("node1")
        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "test"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock,
        )
        log.append(op)

        log.clear()

        assert len(log) == 0
        assert log.sequence_number == 0


# =============================================================================
# Test: ServerOperationLog - Persistence
# =============================================================================


class TestServerOperationLogPersistence:
    """Tests for operation log persistence."""

    def test_persist_operations(self, temp_persist_path):
        """Test persisting operations to disk."""
        log = ServerOperationLog("test-doc", persist_path=temp_persist_path)

        clock = VectorClock()
        clock.increment("node1")
        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "test"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock,
        )

        log.append(op)
        log.persist()

        # Check file exists
        ops_file = temp_persist_path / "test-doc_operations.jsonl"
        assert ops_file.exists()

    def test_load_operations(self, temp_persist_path):
        """Test loading operations from disk."""
        # Create and persist operations
        log1 = ServerOperationLog("test-doc", persist_path=temp_persist_path)

        for i in range(5):
            clock = VectorClock()
            clock.set("node1", i + 1)
            op = CRDTOperation(
                id=f"op{i}",
                type=OperationType.SET,
                path=f"registers.key{i}",
                value={"value": i},
                timestamp=time.time(),
                node_id="node1",
                clock=clock,
            )
            log1.append(op)

        log1.persist()

        # Create new log that loads from disk
        log2 = ServerOperationLog("test-doc", persist_path=temp_persist_path)

        assert len(log2) == 5
        assert log2.get_operation("op0") is not None


# =============================================================================
# Test: ClientSession
# =============================================================================


class TestClientSession:
    """Tests for client session management."""

    def test_create_session(self):
        """Test creating a session."""
        session = ClientSession(client_id="client1")

        assert session.client_id == "client1"
        assert session.state == SessionState.DISCONNECTED

    def test_connect(self):
        """Test connecting session."""
        session = ClientSession(client_id="client1")
        session.connect()

        assert session.state == SessionState.CONNECTED
        assert session.connected_at is not None
        assert session.reconnect_token is not None

    def test_disconnect(self):
        """Test disconnecting session."""
        session = ClientSession(client_id="client1")
        session.connect()
        session.disconnect()

        assert session.state == SessionState.DISCONNECTED
        assert session.connected_at is None

    def test_heartbeat(self):
        """Test updating heartbeat."""
        session = ClientSession(client_id="client1")
        session.connect()

        old_heartbeat = session.last_heartbeat
        time.sleep(0.01)
        session.heartbeat()

        assert session.last_heartbeat > old_heartbeat

    def test_is_alive(self):
        """Test checking if session is alive."""
        session = ClientSession(client_id="client1")
        session.connect()

        assert session.is_alive(timeout=30.0)

        # Simulate stale heartbeat
        session.last_heartbeat = time.time() - 60
        assert not session.is_alive(timeout=30.0)

    def test_subscribe_unsubscribe(self):
        """Test subscription management."""
        session = ClientSession(client_id="client1")

        session.subscribe("doc1")
        session.subscribe("doc2")

        assert session.is_subscribed("doc1")
        assert session.is_subscribed("doc2")
        assert not session.is_subscribed("doc3")

        session.unsubscribe("doc1")

        assert not session.is_subscribed("doc1")

    def test_queue_operations(self):
        """Test queuing operations for delivery."""
        session = ClientSession(client_id="client1")

        clock = VectorClock()
        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "test"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock,
        )

        session.queue_operation(op)
        pending = session.get_pending_operations()

        assert len(pending) == 1
        assert pending[0].id == "op1"

    def test_can_reconnect(self):
        """Test reconnection token validation."""
        session = ClientSession(client_id="client1")
        session.connect()
        token = session.reconnect_token

        assert session.can_reconnect(token)
        assert not session.can_reconnect("invalid-token")

    def test_reconnect_attempts_limit(self):
        """Test reconnection attempt limit."""
        session = ClientSession(client_id="client1", max_reconnect_attempts=3)
        session.connect()
        token = session.reconnect_token

        for _ in range(3):
            session.start_reconnect()

        assert not session.can_reconnect(token)

    def test_serialization(self):
        """Test session serialization."""
        session = ClientSession(client_id="client1")
        session.connect()
        session.subscribe("doc1")

        data = session.to_dict()
        restored = ClientSession.from_dict(data)

        assert restored.client_id == session.client_id
        assert restored.session_id == session.session_id
        assert "doc1" in restored.subscribed_docs


# =============================================================================
# Test: SyncMessage
# =============================================================================


class TestSyncMessage:
    """Tests for sync protocol messages."""

    def test_create_message(self):
        """Test creating a sync message."""
        msg = SyncMessage(
            type=SyncMessageType.CONNECT,
            client_id="client1",
            payload={"metadata": {"version": "1.0"}},
        )

        assert msg.type == SyncMessageType.CONNECT
        assert msg.client_id == "client1"
        assert msg.message_id is not None

    def test_message_serialization(self):
        """Test message serialization."""
        msg = SyncMessage(
            type=SyncMessageType.PUSH,
            client_id="client1",
            doc_id="doc1",
            clock=VectorClock({"node1": 5}),
            operations=[],
        )

        data = msg.to_dict()
        restored = SyncMessage.from_dict(data)

        assert restored.type == msg.type
        assert restored.client_id == msg.client_id
        assert restored.doc_id == msg.doc_id

    def test_message_json(self):
        """Test message JSON serialization."""
        msg = SyncMessage(
            type=SyncMessageType.HEARTBEAT,
            client_id="client1",
        )

        json_str = msg.to_json()
        restored = SyncMessage.from_json(json_str)

        assert restored.type == msg.type
        assert restored.client_id == msg.client_id


# =============================================================================
# Test: SyncProtocol
# =============================================================================


class TestSyncProtocol:
    """Tests for sync protocol handling."""

    def test_handle_connect(self, server):
        """Test handling CONNECT message."""
        msg = SyncMessage(
            type=SyncMessageType.CONNECT,
            client_id="client1",
        )

        response = server.handle_message(msg)

        assert response.type == SyncMessageType.WELCOME
        assert "session_id" in response.payload

    def test_handle_disconnect(self, server):
        """Test handling DISCONNECT message."""
        # Connect first
        server.connect_client("client1")

        msg = SyncMessage(
            type=SyncMessageType.DISCONNECT,
            client_id="client1",
        )

        response = server.handle_message(msg)

        assert response.type == SyncMessageType.ACK
        assert response.payload.get("disconnected") is True

    def test_handle_subscribe(self, server_with_doc):
        """Test handling SUBSCRIBE message."""
        server_with_doc.connect_client("client1")

        msg = SyncMessage(
            type=SyncMessageType.SUBSCRIBE,
            client_id="client1",
            doc_id="test-doc",
        )

        response = server_with_doc.handle_message(msg)

        assert response.type == SyncMessageType.FULL_SYNC
        assert response.payload.get("document") is not None

    def test_handle_subscribe_nonexistent_doc(self, server):
        """Test handling SUBSCRIBE to nonexistent document."""
        server.connect_client("client1")

        msg = SyncMessage(
            type=SyncMessageType.SUBSCRIBE,
            client_id="client1",
            doc_id="nonexistent",
        )

        response = server.handle_message(msg)

        assert response.type == SyncMessageType.ERROR

    def test_handle_unsubscribe(self, server_with_doc):
        """Test handling UNSUBSCRIBE message."""
        server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")

        msg = SyncMessage(
            type=SyncMessageType.UNSUBSCRIBE,
            client_id="client1",
            doc_id="test-doc",
        )

        response = server_with_doc.handle_message(msg)

        assert response.type == SyncMessageType.ACK
        assert response.payload.get("unsubscribed") is True

    def test_handle_push(self, server_with_doc):
        """Test handling PUSH message."""
        server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")

        clock = VectorClock()
        clock.increment("client1")
        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.new_key",
            value={"value": "new_value", "timestamp": time.time(), "node_id": "client1"},
            timestamp=time.time(),
            node_id="client1",
            clock=clock,
        )

        msg = SyncMessage(
            type=SyncMessageType.PUSH,
            client_id="client1",
            doc_id="test-doc",
            operations=[op],
        )

        response = server_with_doc.handle_message(msg)

        assert response.type == SyncMessageType.ACK
        assert response.payload.get("applied") == 1

    def test_handle_pull(self, server_with_doc):
        """Test handling PULL message."""
        server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")

        msg = SyncMessage(
            type=SyncMessageType.PULL,
            client_id="client1",
            doc_id="test-doc",
            clock=VectorClock(),
        )

        response = server_with_doc.handle_message(msg)

        assert response.type == SyncMessageType.DELTA

    def test_handle_heartbeat(self, server):
        """Test handling HEARTBEAT message."""
        server.connect_client("client1")

        msg = SyncMessage(
            type=SyncMessageType.HEARTBEAT,
            client_id="client1",
        )

        response = server.handle_message(msg)

        assert response.type == SyncMessageType.ACK
        assert "server_time" in response.payload

    def test_handle_reconnect(self, server_with_doc):
        """Test handling RECONNECT message."""
        session = server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")
        token = session.reconnect_token

        server_with_doc.disconnect_client("client1")

        msg = SyncMessage(
            type=SyncMessageType.RECONNECT,
            client_id="client1",
            clock=VectorClock(),
            payload={"reconnect_token": token},
        )

        response = server_with_doc.handle_message(msg)

        assert response.type == SyncMessageType.DELTA
        assert response.payload.get("reconnected") is True


# =============================================================================
# Test: CollaborationClient
# =============================================================================


class TestCollaborationClient:
    """Tests for collaboration client."""

    def test_connect(self, server):
        """Test client connection."""
        client = CollaborationClient(client_id="client1")
        result = client.connect(server)

        assert result is True
        assert client.is_connected

    def test_disconnect(self, client):
        """Test client disconnection."""
        client.disconnect()

        assert not client.is_connected

    def test_subscribe(self, client):
        """Test subscribing to document."""
        doc = client.subscribe("test-doc")

        assert doc is not None
        assert "test-doc" in client.subscriptions

    def test_subscribe_nonexistent_doc(self, client, server_with_doc):
        """Test subscribing to nonexistent document raises error."""
        with pytest.raises(ServerError):
            client.subscribe("nonexistent")

    def test_unsubscribe(self, client):
        """Test unsubscribing from document."""
        client.subscribe("test-doc")
        client.unsubscribe("test-doc")

        assert "test-doc" not in client.subscriptions

    def test_push_operations(self, client):
        """Test pushing operations to server."""
        client.subscribe("test-doc")

        clock = VectorClock()
        clock.increment("test-client")
        op = CRDTOperation(
            id="push-op1",
            type=OperationType.SET,
            path="registers.pushed",
            value={"value": "pushed_value", "timestamp": time.time(), "node_id": "test-client"},
            timestamp=time.time(),
            node_id="test-client",
            clock=clock,
        )

        applied = client.push("test-doc", [op])

        assert applied == 1

    def test_pull_operations(self, client, server_with_doc):
        """Test pulling operations from server."""
        client.subscribe("test-doc")

        # Make changes on server
        doc = server_with_doc.get_document("test-doc")
        doc.set_register("server_key", "server_value")

        ops = client.pull("test-doc")

        # Should get operations
        assert isinstance(ops, list)

    def test_heartbeat(self, client):
        """Test sending heartbeat."""
        result = client.heartbeat()

        assert result is True

    def test_reconnect(self, client, server_with_doc):
        """Test client reconnection."""
        client.subscribe("test-doc")

        # Simulate disconnect
        client._connected = False

        success, ops = client.reconnect()

        assert success is True
        assert client.is_connected

    def test_get_local_document(self, client):
        """Test getting local document copy."""
        client.subscribe("test-doc")

        local_doc = client.get_local_document("test-doc")

        assert local_doc is not None
        assert local_doc.doc_id == "test-doc"


# =============================================================================
# Test: UndoManager
# =============================================================================


class TestUndoManager:
    """Tests for undo/redo management."""

    def test_push_undo(self):
        """Test pushing to undo stack."""
        mgr = UndoManager("client1")
        clock = VectorClock()

        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "new"},
            timestamp=time.time(),
            node_id="client1",
            clock=clock,
        )
        inverse = CRDTOperation(
            id="op1-inverse",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "old"},
            timestamp=time.time(),
            node_id="client1",
            clock=clock,
        )

        mgr.push(op, inverse, original_value="old")

        assert mgr.can_undo()
        assert mgr.undo_count == 1

    def test_pop_undo(self):
        """Test popping from undo stack."""
        mgr = UndoManager("client1")
        clock = VectorClock()

        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "new"},
            timestamp=time.time(),
            node_id="client1",
            clock=clock,
        )
        inverse = CRDTOperation(
            id="op1-inverse",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "old"},
            timestamp=time.time(),
            node_id="client1",
            clock=clock,
        )

        mgr.push(op, inverse)
        entry = mgr.pop_undo()

        assert entry is not None
        assert entry.operation_id == "op1"
        assert mgr.can_redo()

    def test_pop_redo(self):
        """Test popping from redo stack."""
        mgr = UndoManager("client1")
        clock = VectorClock()

        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "new"},
            timestamp=time.time(),
            node_id="client1",
            clock=clock,
        )
        inverse = CRDTOperation(
            id="op1-inverse",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "old"},
            timestamp=time.time(),
            node_id="client1",
            clock=clock,
        )

        mgr.push(op, inverse)
        mgr.pop_undo()  # Move to redo
        entry = mgr.pop_redo()  # Move back to undo

        assert entry is not None
        assert mgr.can_undo()
        assert not mgr.can_redo()

    def test_new_operation_clears_redo(self):
        """Test that new operation clears redo stack."""
        mgr = UndoManager("client1")
        clock = VectorClock()

        for i in range(3):
            op = CRDTOperation(
                id=f"op{i}",
                type=OperationType.SET,
                path="registers.key",
                value={"value": f"v{i}"},
                timestamp=time.time(),
                node_id="client1",
                clock=clock,
            )
            inverse = CRDTOperation(
                id=f"op{i}-inv",
                type=OperationType.SET,
                path="registers.key",
                value={"value": f"v{i-1}"},
                timestamp=time.time(),
                node_id="client1",
                clock=clock,
            )
            mgr.push(op, inverse)

        # Undo twice
        mgr.pop_undo()
        mgr.pop_undo()
        assert mgr.redo_count == 2

        # New operation clears redo
        new_op = CRDTOperation(
            id="new_op",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "new"},
            timestamp=time.time(),
            node_id="client1",
            clock=clock,
        )
        mgr.push(new_op, new_op)

        assert mgr.redo_count == 0

    def test_undo_depth_limit(self):
        """Test undo stack depth limit."""
        mgr = UndoManager("client1", max_undo_depth=5)
        clock = VectorClock()

        for i in range(10):
            op = CRDTOperation(
                id=f"op{i}",
                type=OperationType.SET,
                path="registers.key",
                value={"value": i},
                timestamp=time.time(),
                node_id="client1",
                clock=clock,
            )
            mgr.push(op, op)

        assert mgr.undo_count == 5

    def test_clear(self):
        """Test clearing undo/redo stacks."""
        mgr = UndoManager("client1")
        clock = VectorClock()

        op = CRDTOperation(
            id="op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "test"},
            timestamp=time.time(),
            node_id="client1",
            clock=clock,
        )
        mgr.push(op, op)
        mgr.pop_undo()

        mgr.clear()

        assert not mgr.can_undo()
        assert not mgr.can_redo()


# =============================================================================
# Test: Concurrent Operations
# =============================================================================


class TestConcurrentOperations:
    """Tests for concurrent operation handling."""

    def test_concurrent_client_operations(self, server_with_doc):
        """Test concurrent operations from multiple clients."""
        # Connect two clients
        client1 = CollaborationClient(client_id="client1")
        client2 = CollaborationClient(client_id="client2")

        client1.connect(server_with_doc)
        client2.connect(server_with_doc)

        client1.subscribe("test-doc")
        client2.subscribe("test-doc")

        # Both clients push operations
        clock1 = VectorClock()
        clock1.increment("client1")
        op1 = CRDTOperation(
            id="op-client1",
            type=OperationType.SET,
            path="registers.client1_key",
            value={"value": "client1_value", "timestamp": time.time(), "node_id": "client1"},
            timestamp=time.time(),
            node_id="client1",
            clock=clock1,
        )

        clock2 = VectorClock()
        clock2.increment("client2")
        op2 = CRDTOperation(
            id="op-client2",
            type=OperationType.SET,
            path="registers.client2_key",
            value={"value": "client2_value", "timestamp": time.time(), "node_id": "client2"},
            timestamp=time.time(),
            node_id="client2",
            clock=clock2,
        )

        client1.push("test-doc", [op1])
        client2.push("test-doc", [op2])

        # Server document should have both
        doc = server_with_doc.get_document("test-doc")
        assert doc.get_register("client1_key") == "client1_value"
        assert doc.get_register("client2_key") == "client2_value"

    def test_threaded_operations(self, server_with_doc):
        """Test operations from multiple threads."""
        results = {"success": 0, "errors": []}
        lock = threading.Lock()

        def client_worker(client_id):
            try:
                client = CollaborationClient(client_id=client_id)
                client.connect(server_with_doc)
                client.subscribe("test-doc")

                clock = VectorClock()
                clock.increment(client_id)
                op = CRDTOperation(
                    id=f"op-{client_id}",
                    type=OperationType.SET,
                    path=f"registers.{client_id}_key",
                    value={"value": f"{client_id}_value", "timestamp": time.time(), "node_id": client_id},
                    timestamp=time.time(),
                    node_id=client_id,
                    clock=clock,
                )
                client.push("test-doc", [op])
                client.disconnect()

                with lock:
                    results["success"] += 1
            except Exception as e:
                with lock:
                    results["errors"].append(str(e))

        threads = [
            threading.Thread(target=client_worker, args=(f"thread-{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["success"] == 5
        assert len(results["errors"]) == 0


# =============================================================================
# Test: AsyncCollaborationServer
# =============================================================================


class TestAsyncCollaborationServer:
    """Tests for async collaboration server."""

    @pytest.mark.asyncio
    async def test_async_create_document(self):
        """Test async document creation."""
        server = AsyncCollaborationServer(server_id="async-server")
        server.start()

        doc = await server.async_create_document("async-doc", {"key": "value"})

        assert doc is not None
        assert doc.doc_id == "async-doc"

        server.stop()

    @pytest.mark.asyncio
    async def test_async_apply_operations(self):
        """Test async operation application."""
        server = AsyncCollaborationServer(server_id="async-server")
        server.start()

        await server.async_create_document("async-doc")

        clock = VectorClock()
        clock.increment("node1")
        op = CRDTOperation(
            id="async-op1",
            type=OperationType.SET,
            path="registers.key",
            value={"value": "test", "timestamp": time.time(), "node_id": "node1"},
            timestamp=time.time(),
            node_id="node1",
            clock=clock,
        )

        applied = await server.async_apply_operations("async-doc", [op])

        assert applied == 1

        server.stop()

    @pytest.mark.asyncio
    async def test_async_handle_message(self):
        """Test async message handling."""
        server = AsyncCollaborationServer(server_id="async-server")
        server.start()

        msg = SyncMessage(
            type=SyncMessageType.CONNECT,
            client_id="async-client",
        )

        response = await server.async_handle_message(msg)

        assert response.type == SyncMessageType.WELCOME

        server.stop()


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_operations_push(self, client):
        """Test pushing empty operations list."""
        client.subscribe("test-doc")
        applied = client.push("test-doc", [])

        assert applied == 0

    def test_operations_without_connection(self, server_with_doc):
        """Test operations without connection raise errors."""
        client = CollaborationClient(client_id="disconnected")

        with pytest.raises(ClientNotConnectedError):
            client.push("test-doc", [])

    def test_pull_without_subscription(self, client):
        """Test pulling from unsubscribed document."""
        # Should work but return empty since not subscribed
        ops = client.pull("test-doc")
        assert isinstance(ops, list)

    def test_server_serialization(self, server_with_doc):
        """Test server state serialization."""
        server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")

        data = server_with_doc.to_dict()

        assert "documents" in data
        assert "sessions" in data
        assert "subscribers" in data
        assert "test-doc" in data["documents"]

    def test_message_handler_not_set(self, server_with_doc):
        """Test server works without message handler."""
        # This should not raise - server handles missing handler gracefully
        server_with_doc.connect_client("client1")
        server_with_doc.subscribe_client("client1", "test-doc")

        doc = server_with_doc.get_document("test-doc")
        doc.set_register("key", "value")

        # Operations should still be applied
        assert doc.get_register("key") == "value"

    def test_callback_registration(self, server):
        """Test event callback registration."""
        connected_clients = []
        disconnected_clients = []
        operations = []

        server.on_client_connect(lambda cid: connected_clients.append(cid))
        server.on_client_disconnect(lambda cid: disconnected_clients.append(cid))
        server.on_operation(lambda doc_id, op: operations.append((doc_id, op)))

        server.create_document("callback-doc")
        server.connect_client("callback-client")
        server.disconnect_client("callback-client")

        assert "callback-client" in connected_clients
        assert "callback-client" in disconnected_clients


# =============================================================================
# Test: Persistence
# =============================================================================


class TestPersistence:
    """Tests for server persistence."""

    def test_save_and_load_state(self, temp_persist_path):
        """Test saving and loading server state."""
        # Create server and add data
        server1 = CollaborationServer(
            server_id="persist-server",
            persist_path=temp_persist_path,
        )
        server1.start()
        server1.create_document("persist-doc", {"key": "value"})
        server1.save_state()
        server1.stop()

        # Create new server that loads state
        server2 = CollaborationServer(
            server_id="persist-server",
            persist_path=temp_persist_path,
        )
        server2.start()

        assert server2.document_exists("persist-doc")
        doc = server2.get_document("persist-doc")
        assert doc.get_register("key") == "value"

        server2.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
