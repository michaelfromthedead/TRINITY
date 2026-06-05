# Viper IDE Architecture Vol. 2: Network & Deployment

A native network layer for remote inspection, multi-machine debugging, and distributed worlds.

---

## Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  NETWORK MEANS:                                                            │
│  ├─ TCP sockets                                                            │
│  ├─ Unix domain sockets                                                    │
│  ├─ Msgpack encoding                                                       │
│  └─ Custom binary protocol                                                 │
│                                                                             │
│  NETWORK DOES NOT MEAN:                                                    │
│  ├─ HTTP                                                                   │
│  ├─ REST                                                                   │
│  ├─ WebSocket                                                              │
│  ├─ JSON                                                                   │
│  ├─ Browsers                                                               │
│  └─ Any web bullshit                                                       │
│                                                                             │
│  The same GUI/TUI that works locally works remotely.                       │
│  The protocol is invisible. The experience is identical.                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Modes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VIPER DEPLOYMENT MODES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LOCAL MODES                                                               │
│  ├─ GUI        │ Dear PyGui + Neovim, same process                         │
│  ├─ TUI        │ Textual + Neovim, same process                            │
│  ├─ Headless   │ Shell/script only, no UI                                  │
│  └─ Library    │ import viper, programmatic access                         │
│                                                                             │
│  NETWORK MODES                                                             │
│  ├─ Serve      │ World runs as daemon, accepts connections                 │
│  ├─ Attach     │ GUI/TUI connects to running world (like gdb attach)       │
│  └─ Tunnel     │ SSH tunnel for remote access (just works™)                │
│                                                                             │
│  DISTRIBUTED MODES (future)                                                │
│  ├─ Cluster    │ Multiple worlds, synchronized via custom protocol         │
│  ├─ Edge       │ World on device, syncs to central server                  │
│  └─ Replicated │ Read replicas for inspection without affecting primary    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  LOCAL (same process)                                                      │
│                                                                             │
│  ┌─────────────┐         ┌─────────────┐                                   │
│  │   GUI/TUI   │────────▶│ IDEProtocol │────────▶ World                    │
│  └─────────────┘  direct └─────────────┘  direct                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  REMOTE (over network)                                                     │
│                                                                             │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────────────┐   │
│  │   GUI/TUI   │────────▶│ IDEProtocol │────────▶│    ViperClient      │   │
│  └─────────────┘  direct └─────────────┘  impl   └──────────┬──────────┘   │
│                          (RemoteProtocol)                   │              │
│                                                             │ TCP/Unix     │
│                                                             │ msgpack      │
│                                                             │              │
│  ════════════════════════════════════════════════════════════════════════  │
│                                 NETWORK                                     │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                             │              │
│                                                             │              │
│                                                  ┌──────────┴──────────┐   │
│                                                  │    ViperServer      │   │
│                                                  └──────────┬──────────┘   │
│                                                             │              │
│                                                  ┌──────────┴──────────┐   │
│                                                  │   IDEProtocol       │   │
│                                                  │   (LocalProtocol)   │   │
│                                                  └──────────┬──────────┘   │
│                                                             │              │
│                                                  ┌──────────┴──────────┐   │
│                                                  │       World         │   │
│                                                  └─────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

KEY INSIGHT: GUI/TUI code is identical.
             Only the IDEProtocol implementation changes.
             LocalProtocol = direct calls.
             RemoteProtocol = calls over network.
```

---

## Wire Protocol

### Message Format

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MESSAGE STRUCTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┬──────────┬──────────────────────────────────────────────┐    │
│  │  LENGTH  │  HEADER  │                   BODY                       │    │
│  │  4 bytes │  msgpack │                  msgpack                     │    │
│  └──────────┴──────────┴──────────────────────────────────────────────┘    │
│                                                                             │
│  LENGTH: uint32, big-endian, total message size (excluding length field)   │
│                                                                             │
│  HEADER: {                                                                 │
│      "op": int,        // Operation code                                   │
│      "seq": int,       // Sequence number for request/response matching    │
│      "flags": int,     // Reserved for future use                          │
│  }                                                                         │
│                                                                             │
│  BODY: Operation-specific msgpack data                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Operations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OPERATIONS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  QUERIES (1-9)                                                             │
│  ├─ 1  GET_ENTITIES      { filter? }           → { entities[] }            │
│  ├─ 2  GET_ENTITY        { id }                → { entity }                │
│  ├─ 3  GET_FIELDS        { id }                → { fields[] }              │
│  ├─ 4  GET_FIELD         { id, field }         → { value }                 │
│  └─ 5  GET_HISTORY       { id, limit? }        → { changes[] }             │
│                                                                             │
│  MUTATIONS (10-19)                                                         │
│  ├─ 10 SET_FIELD         { id, field, value }  → { ok }                    │
│  ├─ 11 SPAWN             { components[] }      → { id }                    │
│  ├─ 12 DESTROY           { id }                → { ok }                    │
│  ├─ 13 ADD_COMPONENT     { id, component }     → { ok }                    │
│  └─ 14 REMOVE_COMPONENT  { id, type }          → { ok }                    │
│                                                                             │
│  SHELL (20-29)                                                             │
│  ├─ 20 EXECUTE           { cmd }               → { result, stdout }        │
│  ├─ 21 COMPLETE          { partial }           → { completions[] }         │
│  └─ 22 VALIDATE          { cmd }               → { valid, errors[] }       │
│                                                                             │
│  TIME (30-39)                                                              │
│  ├─ 30 GET_TICK          { }                   → { tick, min, max }        │
│  ├─ 31 GOTO_TICK         { tick }              → { ok }                    │
│  ├─ 32 STEP              { count? }            → { tick }                  │
│  ├─ 33 UNDO              { }                   → { ok, tick }              │
│  ├─ 34 REDO              { }                   → { ok, tick }              │
│  └─ 35 SNAPSHOT          { name? }             → { id }                    │
│                                                                             │
│  SUBSCRIPTIONS (40-49)                                                     │
│  ├─ 40 SUBSCRIBE         { events[], filter? } → { sub_id }                │
│  ├─ 41 UNSUBSCRIBE       { sub_id }            → { ok }                    │
│  └─ 42 EVENT             { type, data }        ← (server push)             │
│                                                                             │
│  META (50-59)                                                              │
│  ├─ 50 PING              { }                   → PONG { }                  │
│  ├─ 51 PONG              { }                   ← (response to PING)        │
│  ├─ 52 ERROR             { code, message }     ← (error response)          │
│  ├─ 53 INFO              { }                   → { version, world_name }   │
│  └─ 54 CLOSE             { }                   → (connection closes)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Event Types (for subscriptions)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EVENT TYPES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENTITY_SPAWNED      { id, components[] }                                  │
│  ENTITY_DESTROYED    { id }                                                │
│  COMPONENT_ADDED     { entity_id, component }                              │
│  COMPONENT_REMOVED   { entity_id, type }                                   │
│  FIELD_CHANGED       { entity_id, field, old, new, tick }                  │
│  TICK_ADVANCED       { tick }                                              │
│  SNAPSHOT_CREATED    { id, name, tick }                                    │
│  SYSTEM_RAN          { name, duration_us }                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Protocol Core

```python
# viper/net/protocol.py

"""
Viper wire protocol.
TCP or Unix socket. Msgpack encoding. No web bullshit.
"""

import struct
import msgpack
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Op(IntEnum):
    # Queries
    GET_ENTITIES = 1
    GET_ENTITY = 2
    GET_FIELDS = 3
    GET_FIELD = 4
    GET_HISTORY = 5
    
    # Mutations
    SET_FIELD = 10
    SPAWN = 11
    DESTROY = 12
    ADD_COMPONENT = 13
    REMOVE_COMPONENT = 14
    
    # Shell
    EXECUTE = 20
    COMPLETE = 21
    VALIDATE = 22
    
    # Time
    GET_TICK = 30
    GOTO_TICK = 31
    STEP = 32
    UNDO = 33
    REDO = 34
    SNAPSHOT = 35
    
    # Subscriptions
    SUBSCRIBE = 40
    UNSUBSCRIBE = 41
    EVENT = 42
    
    # Meta
    PING = 50
    PONG = 51
    ERROR = 52
    INFO = 53
    CLOSE = 54


@dataclass
class Message:
    op: Op
    seq: int
    data: dict[str, Any]


def encode(msg: Message) -> bytes:
    """Encode message to wire format."""
    header = msgpack.packb({
        'op': int(msg.op),
        'seq': msg.seq,
        'flags': 0,
    })
    body = msgpack.packb(msg.data)
    payload = header + body
    length = struct.pack('>I', len(payload))
    return length + payload


def decode(data: bytes) -> tuple[Message, int]:
    """
    Decode message from wire format.
    Returns (message, bytes_consumed).
    """
    if len(data) < 4:
        raise ValueError("Incomplete message: need length")
    
    length = struct.unpack('>I', data[:4])[0]
    
    if len(data) < 4 + length:
        raise ValueError("Incomplete message: need more data")
    
    payload = data[4:4 + length]
    
    # Decode header and body
    unpacker = msgpack.Unpacker()
    unpacker.feed(payload)
    
    header = next(unpacker)
    body = next(unpacker)
    
    return Message(
        op=Op(header['op']),
        seq=header['seq'],
        data=body,
    ), 4 + length


class MessageReader:
    """Buffered message reader for streaming data."""
    
    def __init__(self):
        self._buffer = bytearray()
    
    def feed(self, data: bytes) -> list[Message]:
        """Feed data, return complete messages."""
        self._buffer.extend(data)
        messages = []
        
        while True:
            try:
                msg, consumed = decode(bytes(self._buffer))
                messages.append(msg)
                self._buffer = self._buffer[consumed:]
            except ValueError:
                break
        
        return messages
```

### Client

```python
# viper/net/client.py

"""
Viper network client.
Connects to remote world, implements IDEProtocol.
"""

import socket
import threading
from typing import Any, Callable, Iterator
from dataclasses import dataclass

from viper.net.protocol import Op, Message, encode, MessageReader
from viper.ide.protocol import IDEProtocol, EntityInfo, FieldInfo, ChangeRecord


class ViperClient(IDEProtocol):
    """
    Connect to remote Viper world.
    
    Implements IDEProtocol, so GUI/TUI can use it transparently.
    """
    
    def __init__(self, address: str):
        """
        address formats:
          - "tcp://host:port"
          - "unix:///path/to/socket"
        """
        self._address = address
        self._socket: socket.socket | None = None
        self._seq = 0
        self._pending: dict[int, threading.Event] = {}
        self._responses: dict[int, Message] = {}
        self._subscriptions: dict[int, Callable] = {}
        self._reader = MessageReader()
        self._running = False
        self._recv_thread: threading.Thread | None = None
    
    def connect(self):
        """Establish connection to server."""
        if self._address.startswith('tcp://'):
            host_port = self._address[6:]
            host, port = host_port.rsplit(':', 1)
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((host, int(port)))
        
        elif self._address.startswith('unix://'):
            path = self._address[7:]
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(path)
        
        else:
            raise ValueError(f"Unknown address format: {self._address}")
        
        # Start receive thread
        self._running = True
        self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._recv_thread.start()
    
    def disconnect(self):
        """Close connection."""
        self._running = False
        if self._socket:
            try:
                self._send(Op.CLOSE, {})
            except:
                pass
            self._socket.close()
            self._socket = None
    
    def _receive_loop(self):
        """Background thread: receive and dispatch messages."""
        while self._running:
            try:
                data = self._socket.recv(65536)
                if not data:
                    break
                
                for msg in self._reader.feed(data):
                    self._handle_message(msg)
            
            except Exception as e:
                if self._running:
                    print(f"Receive error: {e}")
                break
    
    def _handle_message(self, msg: Message):
        """Handle incoming message."""
        if msg.op == Op.EVENT:
            # Dispatch to subscription callbacks
            sub_id = msg.data.get('sub_id')
            if sub_id in self._subscriptions:
                self._subscriptions[sub_id](msg.data)
        else:
            # Response to a request
            if msg.seq in self._pending:
                self._responses[msg.seq] = msg
                self._pending[msg.seq].set()
    
    def _send(self, op: Op, data: dict, wait: bool = True) -> dict:
        """Send message and optionally wait for response."""
        self._seq += 1
        seq = self._seq
        
        msg = Message(op=op, seq=seq, data=data)
        self._socket.sendall(encode(msg))
        
        if not wait:
            return {}
        
        # Wait for response
        event = threading.Event()
        self._pending[seq] = event
        
        if not event.wait(timeout=30):
            del self._pending[seq]
            raise TimeoutError(f"No response for {op.name}")
        
        del self._pending[seq]
        response = self._responses.pop(seq)
        
        if response.op == Op.ERROR:
            raise Exception(response.data.get('message', 'Unknown error'))
        
        return response.data
    
    # ========================================================================
    # IDEProtocol Implementation
    # ========================================================================
    
    def get_entities(self, filter: str = "") -> Iterator[EntityInfo]:
        result = self._send(Op.GET_ENTITIES, {'filter': filter})
        for e in result.get('entities', []):
            yield EntityInfo(
                id=e['id'],
                type_name=e['type_name'],
                components=e['components'],
            )
    
    def get_entity(self, id: int) -> EntityInfo | None:
        try:
            result = self._send(Op.GET_ENTITY, {'id': id})
            e = result.get('entity')
            if e:
                return EntityInfo(
                    id=e['id'],
                    type_name=e['type_name'],
                    components=e['components'],
                )
        except:
            pass
        return None
    
    def get_fields(self, entity_id: int) -> list[FieldInfo]:
        result = self._send(Op.GET_FIELDS, {'id': entity_id})
        return [
            FieldInfo(
                name=f['name'],
                type_name=f['type_name'],
                value=f['value'],
                metadata=f.get('metadata', {}),
            )
            for f in result.get('fields', [])
        ]
    
    def get_field_value(self, entity_id: int, field: str) -> Any:
        result = self._send(Op.GET_FIELD, {'id': entity_id, 'field': field})
        return result.get('value')
    
    def set_field_value(self, entity_id: int, field: str, value: Any) -> None:
        self._send(Op.SET_FIELD, {'id': entity_id, 'field': field, 'value': value})
    
    def get_history(self, entity_id: int, limit: int = 50) -> list[ChangeRecord]:
        result = self._send(Op.GET_HISTORY, {'id': entity_id, 'limit': limit})
        return [
            ChangeRecord(
                tick=c['tick'],
                field=c['field'],
                old_value=c['old'],
                new_value=c['new'],
                cause=c.get('cause'),
            )
            for c in result.get('changes', [])
        ]
    
    def undo(self) -> bool:
        result = self._send(Op.UNDO, {})
        return result.get('ok', False)
    
    def redo(self) -> bool:
        result = self._send(Op.REDO, {})
        return result.get('ok', False)
    
    def get_current_tick(self) -> int:
        result = self._send(Op.GET_TICK, {})
        return result.get('tick', 0)
    
    def get_tick_range(self) -> tuple[int, int]:
        result = self._send(Op.GET_TICK, {})
        return (result.get('min', 0), result.get('max', 0))
    
    def goto_tick(self, tick: int) -> None:
        self._send(Op.GOTO_TICK, {'tick': tick})
    
    def step(self) -> None:
        self._send(Op.STEP, {})
    
    def execute(self, command: str) -> Any:
        result = self._send(Op.EXECUTE, {'cmd': command})
        return result.get('result')
    
    def get_selected_entity(self) -> int | None:
        # Selection is client-side state
        return getattr(self, '_selected', None)
    
    def select_entity(self, id: int | None) -> None:
        self._selected = id
    
    # ========================================================================
    # Subscriptions
    # ========================================================================
    
    def subscribe(self, events: list[str], callback: Callable[[dict], None]) -> int:
        """Subscribe to events, return subscription ID."""
        result = self._send(Op.SUBSCRIBE, {'events': events})
        sub_id = result['sub_id']
        self._subscriptions[sub_id] = callback
        return sub_id
    
    def unsubscribe(self, sub_id: int) -> None:
        """Unsubscribe from events."""
        self._send(Op.UNSUBSCRIBE, {'sub_id': sub_id})
        self._subscriptions.pop(sub_id, None)
```

### Server

```python
# viper/net/server.py

"""
Viper network server.
Serves a World over TCP or Unix socket.
"""

import socket
import threading
import os
from typing import Any
from dataclasses import dataclass, field

from viper.core import World
from viper.ide.protocol import ViperIDEProtocol
from viper.net.protocol import Op, Message, encode, MessageReader


@dataclass
class ClientConnection:
    socket: socket.socket
    address: Any
    subscriptions: dict[int, set[str]] = field(default_factory=dict)
    next_sub_id: int = 1


class ViperServer:
    """
    Serve Viper world over network.
    
    Accepts multiple client connections.
    Broadcasts events to subscribed clients.
    """
    
    def __init__(self, world: World, address: str):
        """
        address formats:
          - "tcp://host:port"
          - "unix:///path/to/socket"
        """
        self._world = world
        self._protocol = ViperIDEProtocol(world)
        self._address = address
        self._socket: socket.socket | None = None
        self._clients: list[ClientConnection] = []
        self._clients_lock = threading.Lock()
        self._running = False
    
    def start(self):
        """Start server, blocking."""
        self._setup_socket()
        self._setup_world_hooks()
        
        self._running = True
        print(f"Viper server listening on {self._address}")
        
        while self._running:
            try:
                client_socket, address = self._socket.accept()
                client = ClientConnection(socket=client_socket, address=address)
                
                with self._clients_lock:
                    self._clients.append(client)
                
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client,),
                    daemon=True
                )
                thread.start()
                
                print(f"Client connected: {address}")
            
            except Exception as e:
                if self._running:
                    print(f"Accept error: {e}")
    
    def stop(self):
        """Stop server."""
        self._running = False
        if self._socket:
            self._socket.close()
        
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.socket.close()
                except:
                    pass
            self._clients.clear()
    
    def _setup_socket(self):
        """Create and bind server socket."""
        if self._address.startswith('tcp://'):
            host_port = self._address[6:]
            host, port = host_port.rsplit(':', 1)
            
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((host, int(port)))
        
        elif self._address.startswith('unix://'):
            path = self._address[7:]
            
            # Remove existing socket file
            if os.path.exists(path):
                os.unlink(path)
            
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.bind(path)
        
        else:
            raise ValueError(f"Unknown address format: {self._address}")
        
        self._socket.listen(16)
    
    def _setup_world_hooks(self):
        """Hook into world events for broadcasting."""
        if hasattr(self._world, 'on_change'):
            self._world.on_change(self._on_world_change)
        
        if hasattr(self._world, 'on_spawn'):
            self._world.on_spawn(self._on_entity_spawned)
        
        if hasattr(self._world, 'on_destroy'):
            self._world.on_destroy(self._on_entity_destroyed)
    
    def _on_world_change(self, entity_id: int, field: str, old: Any, new: Any, tick: int):
        """Broadcast field change to subscribed clients."""
        self._broadcast('FIELD_CHANGED', {
            'entity_id': entity_id,
            'field': field,
            'old': old,
            'new': new,
            'tick': tick,
        })
    
    def _on_entity_spawned(self, entity_id: int, components: list[str]):
        """Broadcast entity spawn."""
        self._broadcast('ENTITY_SPAWNED', {
            'id': entity_id,
            'components': components,
        })
    
    def _on_entity_destroyed(self, entity_id: int):
        """Broadcast entity destruction."""
        self._broadcast('ENTITY_DESTROYED', {
            'id': entity_id,
        })
    
    def _broadcast(self, event_type: str, data: dict):
        """Broadcast event to subscribed clients."""
        with self._clients_lock:
            for client in self._clients:
                for sub_id, events in client.subscriptions.items():
                    if event_type in events or '*' in events:
                        try:
                            msg = Message(
                                op=Op.EVENT,
                                seq=0,
                                data={'sub_id': sub_id, 'type': event_type, **data}
                            )
                            client.socket.sendall(encode(msg))
                        except:
                            pass
    
    def _handle_client(self, client: ClientConnection):
        """Handle client connection."""
        reader = MessageReader()
        
        try:
            while self._running:
                data = client.socket.recv(65536)
                if not data:
                    break
                
                for msg in reader.feed(data):
                    response = self._handle_message(client, msg)
                    client.socket.sendall(encode(response))
        
        except Exception as e:
            print(f"Client error: {e}")
        
        finally:
            with self._clients_lock:
                if client in self._clients:
                    self._clients.remove(client)
            client.socket.close()
            print(f"Client disconnected: {client.address}")
    
    def _handle_message(self, client: ClientConnection, msg: Message) -> Message:
        """Handle incoming message, return response."""
        try:
            match msg.op:
                # Queries
                case Op.GET_ENTITIES:
                    entities = list(self._protocol.get_entities(msg.data.get('filter', '')))
                    return Message(Op.GET_ENTITIES, msg.seq, {
                        'entities': [
                            {'id': e.id, 'type_name': e.type_name, 'components': e.components}
                            for e in entities
                        ]
                    })
                
                case Op.GET_ENTITY:
                    entity = self._protocol.get_entity(msg.data['id'])
                    if entity:
                        return Message(Op.GET_ENTITY, msg.seq, {
                            'entity': {'id': entity.id, 'type_name': entity.type_name, 'components': entity.components}
                        })
                    return Message(Op.ERROR, msg.seq, {'message': 'Entity not found'})
                
                case Op.GET_FIELDS:
                    fields = self._protocol.get_fields(msg.data['id'])
                    return Message(Op.GET_FIELDS, msg.seq, {
                        'fields': [
                            {'name': f.name, 'type_name': f.type_name, 'value': f.value, 'metadata': f.metadata}
                            for f in fields
                        ]
                    })
                
                case Op.GET_FIELD:
                    value = self._protocol.get_field_value(msg.data['id'], msg.data['field'])
                    return Message(Op.GET_FIELD, msg.seq, {'value': value})
                
                case Op.GET_HISTORY:
                    history = self._protocol.get_history(msg.data['id'], msg.data.get('limit', 50))
                    return Message(Op.GET_HISTORY, msg.seq, {
                        'changes': [
                            {'tick': c.tick, 'field': c.field, 'old': c.old_value, 'new': c.new_value, 'cause': c.cause}
                            for c in history
                        ]
                    })
                
                # Mutations
                case Op.SET_FIELD:
                    self._protocol.set_field_value(msg.data['id'], msg.data['field'], msg.data['value'])
                    return Message(Op.SET_FIELD, msg.seq, {'ok': True})
                
                case Op.SPAWN:
                    # TODO: Implement spawn
                    return Message(Op.ERROR, msg.seq, {'message': 'Not implemented'})
                
                case Op.DESTROY:
                    # TODO: Implement destroy
                    return Message(Op.ERROR, msg.seq, {'message': 'Not implemented'})
                
                # Shell
                case Op.EXECUTE:
                    result = self._protocol.execute(msg.data['cmd'])
                    return Message(Op.EXECUTE, msg.seq, {'result': result})
                
                # Time
                case Op.GET_TICK:
                    tick = self._protocol.get_current_tick()
                    min_tick, max_tick = self._protocol.get_tick_range()
                    return Message(Op.GET_TICK, msg.seq, {'tick': tick, 'min': min_tick, 'max': max_tick})
                
                case Op.GOTO_TICK:
                    self._protocol.goto_tick(msg.data['tick'])
                    return Message(Op.GOTO_TICK, msg.seq, {'ok': True})
                
                case Op.STEP:
                    self._protocol.step()
                    return Message(Op.STEP, msg.seq, {'tick': self._protocol.get_current_tick()})
                
                case Op.UNDO:
                    ok = self._protocol.undo()
                    return Message(Op.UNDO, msg.seq, {'ok': ok, 'tick': self._protocol.get_current_tick()})
                
                case Op.REDO:
                    ok = self._protocol.redo()
                    return Message(Op.REDO, msg.seq, {'ok': ok, 'tick': self._protocol.get_current_tick()})
                
                # Subscriptions
                case Op.SUBSCRIBE:
                    sub_id = client.next_sub_id
                    client.next_sub_id += 1
                    client.subscriptions[sub_id] = set(msg.data.get('events', ['*']))
                    return Message(Op.SUBSCRIBE, msg.seq, {'sub_id': sub_id})
                
                case Op.UNSUBSCRIBE:
                    sub_id = msg.data['sub_id']
                    client.subscriptions.pop(sub_id, None)
                    return Message(Op.UNSUBSCRIBE, msg.seq, {'ok': True})
                
                # Meta
                case Op.PING:
                    return Message(Op.PONG, msg.seq, {})
                
                case Op.INFO:
                    return Message(Op.INFO, msg.seq, {
                        'version': '0.1.0',
                        'world_name': getattr(self._world, 'name', 'unnamed'),
                    })
                
                case Op.CLOSE:
                    raise ConnectionAbortedError("Client requested close")
                
                case _:
                    return Message(Op.ERROR, msg.seq, {'message': f'Unknown op: {msg.op}'})
        
        except Exception as e:
            return Message(Op.ERROR, msg.seq, {'message': str(e)})
```

---

## Usage

### Starting a Server

```bash
# TCP
viper serve --address tcp://0.0.0.0:9999

# Unix socket
viper serve --address unix:///tmp/viper.sock

# With specific world file
viper serve --world ./game.world --address tcp://0.0.0.0:9999
```

### Attaching a Client

```bash
# GUI attached to remote
viper ide --mode gui --attach tcp://192.168.1.50:9999

# TUI attached to remote
viper ide --mode tui --attach tcp://localhost:9999

# TUI over Unix socket
viper ide --mode tui --attach unix:///tmp/viper.sock

# Shell only
viper shell --attach tcp://gameserver:9999

# Inspector only (no Neovim)
viper inspect --attach tcp://localhost:9999
```

### SSH Tunneling

```bash
# On local machine: tunnel to remote server
ssh -L 9999:localhost:9999 user@gameserver

# Then attach locally
viper ide --mode gui --attach tcp://localhost:9999

# Or one-liner with ProxyJump
ssh -J jumphost user@gameserver "viper serve --address tcp://localhost:9999" &
viper ide --attach tcp://localhost:9999
```

### Programmatic Usage

```python
from viper.net.client import ViperClient

# Connect
client = ViperClient("tcp://localhost:9999")
client.connect()

# Use like local IDEProtocol
for entity in client.get_entities():
    print(f"Entity {entity.id}: {entity.type_name}")

fields = client.get_fields(42)
for f in fields:
    print(f"  {f.name}: {f.value}")

# Execute shell command
result = client.execute("world.query(Player)")
print(result)

# Subscribe to changes
def on_change(event):
    print(f"Change: {event}")

sub_id = client.subscribe(['FIELD_CHANGED'], on_change)

# Later
client.unsubscribe(sub_id)
client.disconnect()
```

---

## Multi-Machine Debugging

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  SCENARIO: Debug live game server from laptop                              │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ GAME SERVER (192.168.1.100)                                        │    │
│  │                                                                    │    │
│  │   ┌──────────────────────────────────────────────────────────┐    │    │
│  │   │ Game Loop                                                 │    │    │
│  │   │   while running:                                          │    │    │
│  │   │       world.step()                                        │    │    │
│  │   │       render()                                            │    │    │
│  │   └──────────────────────────────────────────────────────────┘    │    │
│  │                        │                                          │    │
│  │                        ▼                                          │    │
│  │   ┌──────────────────────────────────────────────────────────┐    │    │
│  │   │ ViperServer (tcp://0.0.0.0:9999)                         │    │    │
│  │   │   └─ Serves world state                                   │    │    │
│  │   │   └─ Accepts inspection connections                       │    │    │
│  │   │   └─ Broadcasts events                                    │    │    │
│  │   └──────────────────────────────────────────────────────────┘    │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                              TCP :9999                                      │
│                                     │                                       │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ DEVELOPER LAPTOP                                                   │    │
│  │                                                                    │    │
│  │   $ viper ide --mode gui --attach tcp://192.168.1.100:9999        │    │
│  │                                                                    │    │
│  │   ┌───────────────────────────────────────────────────────────┐   │    │
│  │   │ Viper IDE                                                  │   │    │
│  │   │                                                            │   │    │
│  │   │  Browser  │  Neovim  │  Inspector (live server state)     │   │    │
│  │   │           │          │                                     │   │    │
│  │   │           │          │  Player (42)                        │   │    │
│  │   │           │          │  health: 73.5  ← live value         │   │    │
│  │   │           │          │  position: (142, 0, 89)             │   │    │
│  │   │           │          │                                     │   │    │
│  │   │  Shell: >>> player.health = 100                            │   │    │
│  │   │         → health: 73.5 → 100                               │   │    │
│  │   │         (instantly affects live server!)                   │   │    │
│  │   └───────────────────────────────────────────────────────────┘   │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Entry Points

```python
# viper/__main__.py additions

@cli.command()
@click.option('--address', default='tcp://0.0.0.0:9999')
@click.option('--world', type=click.Path())
def serve(address: str, world: str | None):
    """Start Viper server."""
    from viper.core import World
    from viper.net.server import ViperServer
    
    w = World.load(world) if world else World()
    server = ViperServer(w, address)
    
    try:
        server.start()  # Blocking
    except KeyboardInterrupt:
        server.stop()


# Modify 'ide' command to support --attach
@cli.command()
@click.option('--mode', type=click.Choice(['gui', 'tui']), default='gui')
@click.option('--attach', type=str, help='Remote address to attach to')
def ide(mode: str, attach: str | None):
    """Start Viper IDE."""
    from viper.ide.protocol import ViperIDEProtocol
    
    if attach:
        # Remote mode
        from viper.net.client import ViperClient
        protocol = ViperClient(attach)
        protocol.connect()
    else:
        # Local mode
        from viper.core import World
        world = World()
        protocol = ViperIDEProtocol(world)
    
    # Same renderer code works for both!
    if mode == 'gui':
        from viper.ide.renderers.dpg import DearPyGuiRenderer
        renderer = DearPyGuiRenderer(protocol)
        renderer.run()
    else:
        from viper.ide.renderers.textual import run_tui
        run_tui(protocol)
```

---

## File Structure

```
viper/
├── core/                         # ECS
├── foundation/                   # Mirror, Tracker, etc.
├── ide/                          # IDE (from Vol. 1)
│
├── net/
│   ├── __init__.py
│   ├── protocol.py               # Wire protocol, Message, encode/decode (~100 lines)
│   ├── client.py                 # ViperClient (~200 lines)
│   └── server.py                 # ViperServer (~250 lines)
│
└── __main__.py                   # CLI entry points
```

---

## Size Estimate

| Component | Lines |
|-----------|-------|
| Wire protocol | ~100 |
| ViperClient | ~200 |
| ViperServer | ~250 |
| CLI additions | ~50 |
| **Total Net** | **~600** |

Combined with IDE (~2300) and Core (~3000):
**~5900 lines** for full local + network system.

---

## Future: Cluster Mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  CLUSTER MODE (future)                                                     │
│                                                                             │
│  Multiple Viper worlds, synchronized.                                      │
│  Each world can be primary for some entities, replica for others.          │
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │ World A         │     │ World B         │     │ World C         │       │
│  │ (primary: zone1)│◄───▶│ (primary: zone2)│◄───▶│ (primary: zone3)│       │
│  │ (replica: zone2)│     │ (replica: zone1)│     │ (replica: zone2)│       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                             │
│  Sync protocol:                                                            │
│  • Authoritative changes propagate to replicas                             │
│  • Conflict resolution by tick + entity authority                          │
│  • Snapshot sync for new nodes joining cluster                             │
│                                                                             │
│  Use cases:                                                                │
│  • Distributed game servers                                                │
│  • Edge computing (device ↔ cloud sync)                                    │
│  • Read replicas for heavy inspection without affecting primary            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

This is future work. The foundation (Vol. 1 + Vol. 2) makes it possible.
```

---

## Summary

```
WHAT:     Native network layer for Viper
WHY:      Remote debugging, multi-machine development, distributed worlds
HOW:      TCP/Unix sockets, msgpack, custom protocol
NOT:      HTTP, REST, WebSocket, JSON, browsers

MODES:
• serve   → Run world as daemon, accept connections
• attach  → Connect IDE to remote world
• tunnel  → SSH tunneling just works

PROTOCOL:
• Binary msgpack
• Request/response with sequence numbers
• Push events for subscriptions
• ~20 operations covering full IDEProtocol

SIZE:     ~600 lines
ENABLES:  Debug live servers, attach from anywhere, same UI local/remote
```

---

*Volume 2 complete. Network without web. Native all the way down.*
