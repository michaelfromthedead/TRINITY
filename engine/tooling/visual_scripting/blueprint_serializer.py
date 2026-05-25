"""
FlowForge Blueprint Serializer - Save/load blueprints with versioning.

Provides serialization capabilities:
- JSON and binary serialization formats
- Version migration support
- Compression options
- Partial loading for large blueprints
- Asset reference resolution
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from .data_types import TYPE_REGISTRY, BlueprintType
from .graph_editor import BlueprintGraph, Connection
from .node_types import Node, NODE_REGISTRY, NodeCategory, Pin, PinKind


class SerializationFormat(Enum):
    """Serialization format options."""
    JSON = auto()
    JSON_COMPRESSED = auto()
    BINARY = auto()
    BINARY_COMPRESSED = auto()


@dataclass
class SerializationOptions:
    """Options for serialization."""
    format: SerializationFormat = SerializationFormat.JSON
    include_metadata: bool = True
    include_comments: bool = True
    include_layout: bool = True
    pretty_print: bool = False
    compression_level: int = 6  # 1-9 for gzip


@dataclass
class BlueprintHeader:
    """Header information for a serialized blueprint."""
    version: str
    format_version: int
    created_time: float
    modified_time: float
    author: str
    description: str
    checksum: str
    node_count: int
    connection_count: int
    parent_class: str
    is_macro: bool


# Current format version
FORMAT_VERSION = 1

# Version migration handlers
VERSION_MIGRATIONS: Dict[int, Callable[[Dict], Dict]] = {}


def register_migration(from_version: int) -> Callable:
    """Decorator to register a version migration."""
    def decorator(func: Callable[[Dict], Dict]) -> Callable:
        VERSION_MIGRATIONS[from_version] = func
        return func
    return decorator


class BlueprintSerializer:
    """Serializer for blueprint graphs."""

    def __init__(self, options: Optional[SerializationOptions] = None):
        self.options = options or SerializationOptions()

        # Custom serializers for specific node types
        self._node_serializers: Dict[str, Callable[[Node], Dict]] = {}
        self._node_deserializers: Dict[str, Callable[[Dict], Node]] = {}

        # Asset reference resolver
        self._asset_resolver: Optional[Callable[[str], Any]] = None

    def set_asset_resolver(self, resolver: Callable[[str], Any]) -> None:
        """Set a function to resolve asset references."""
        self._asset_resolver = resolver

    def register_node_serializer(
        self,
        node_type: str,
        serializer: Callable[[Node], Dict],
        deserializer: Callable[[Dict], Node]
    ) -> None:
        """Register custom serializer for a node type."""
        self._node_serializers[node_type] = serializer
        self._node_deserializers[node_type] = deserializer

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def serialize(self, graph: BlueprintGraph) -> bytes:
        """Serialize a blueprint graph."""
        data = self._graph_to_dict(graph)

        if self.options.format in (SerializationFormat.JSON, SerializationFormat.JSON_COMPRESSED):
            if self.options.pretty_print:
                json_str = json.dumps(data, indent=2, sort_keys=True)
            else:
                json_str = json.dumps(data, separators=(',', ':'))
            content = json_str.encode('utf-8')
        else:
            content = self._dict_to_binary(data)

        if self.options.format in (SerializationFormat.JSON_COMPRESSED, SerializationFormat.BINARY_COMPRESSED):
            content = gzip.compress(content, compresslevel=self.options.compression_level)

        return content

    def serialize_to_file(self, graph: BlueprintGraph, filepath: str) -> bool:
        """Serialize a blueprint to a file."""
        try:
            data = self.serialize(graph)
            with open(filepath, 'wb') as f:
                f.write(data)
            return True
        except Exception:
            return False

    def _graph_to_dict(self, graph: BlueprintGraph) -> Dict[str, Any]:
        """Convert a graph to a dictionary."""
        current_time = time.time()

        data = {
            "header": {
                "version": "1.0",
                "format_version": FORMAT_VERSION,
                "created_time": current_time,
                "modified_time": current_time,
                "author": "",
                "description": graph.description,
                "checksum": "",
                "node_count": len(graph.nodes),
                "connection_count": len(graph.connections),
                "parent_class": graph.parent_class,
                "is_macro": graph.is_macro
            },
            "graph": {
                "id": graph.id,
                "name": graph.name,
                "category": graph.category,
                "entry_points": graph.entry_points
            },
            "nodes": [],
            "connections": []
        }

        # Serialize nodes
        for node in graph.nodes.values():
            node_data = self._serialize_node(node)
            data["nodes"].append(node_data)

        # Serialize connections
        for conn in graph.connections.values():
            conn_data = self._serialize_connection(conn)
            data["connections"].append(conn_data)

        # Calculate checksum
        data["header"]["checksum"] = self._calculate_checksum(data)

        return data

    def _serialize_node(self, node: Node) -> Dict[str, Any]:
        """Serialize a single node."""
        meta = node.get_metadata()
        node_type = type(node).__name__

        # Check for custom serializer
        if node_type in self._node_serializers:
            base_data = self._node_serializers[node_type](node)
        else:
            base_data = {}

        node_data = {
            "id": node.id,
            "type": node_type,
            "category": meta.category.name,
            **base_data
        }

        if self.options.include_layout:
            node_data["position"] = list(node.position)

        if self.options.include_comments and node.comment:
            node_data["comment"] = node.comment

        if node.is_disabled:
            node_data["disabled"] = True

        # Serialize pins with values
        node_data["input_pins"] = []
        for pin in node.input_pins.values():
            pin_data = self._serialize_pin(pin)
            node_data["input_pins"].append(pin_data)

        node_data["output_pins"] = []
        for pin in node.output_pins.values():
            pin_data = self._serialize_pin(pin)
            node_data["output_pins"].append(pin_data)

        # Serialize node-specific properties
        if hasattr(node, 'variable_name'):
            node_data["variable_name"] = node.variable_name
        if hasattr(node, 'function_name'):
            node_data["function_name"] = node.function_name
        if hasattr(node, 'macro_name'):
            node_data["macro_name"] = node.macro_name
        if hasattr(node, 'action_name'):
            node_data["action_name"] = node.action_name
        if hasattr(node, 'event_name'):
            node_data["event_name"] = node.event_name

        return node_data

    def _serialize_pin(self, pin: Pin) -> Dict[str, Any]:
        """Serialize a pin."""
        pin_data = {
            "id": pin.id,
            "name": pin.name,
            "kind": pin.kind.name,
            "direction": pin.direction.name,
        }

        if pin.data_type:
            pin_data["type"] = pin.data_type.type_name()

        if pin.default_value is not None and pin.kind == PinKind.DATA:
            pin_data["value"] = self._serialize_value(pin.default_value)

        if pin.is_hidden:
            pin_data["hidden"] = True

        return pin_data

    def _serialize_connection(self, conn: Connection) -> Dict[str, Any]:
        """Serialize a connection."""
        return {
            "id": conn.id,
            "source_node": conn.source_node_id,
            "source_pin": conn.source_pin_id,
            "target_node": conn.target_node_id,
            "target_pin": conn.target_pin_id,
            "is_execution": conn.is_execution
        }

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value to JSON-compatible format."""
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, '__dict__'):
            # Dataclass or custom object
            return {
                "__type__": type(value).__name__,
                **{k: self._serialize_value(v) for k, v in value.__dict__.items()}
            }
        return str(value)

    def _calculate_checksum(self, data: Dict) -> str:
        """Calculate a checksum for validation."""
        # Exclude checksum field itself
        data_copy = dict(data)
        if "header" in data_copy:
            data_copy["header"] = dict(data_copy["header"])
            data_copy["header"]["checksum"] = ""

        content = json.dumps(data_copy, sort_keys=True).encode('utf-8')
        return hashlib.sha256(content).hexdigest()[:16]

    # =========================================================================
    # DESERIALIZATION
    # =========================================================================

    def deserialize(self, data: bytes) -> Optional[BlueprintGraph]:
        """Deserialize a blueprint from bytes."""
        try:
            # Detect compression
            if data[:2] == b'\x1f\x8b':  # gzip magic
                data = gzip.decompress(data)

            # Try JSON first
            try:
                dict_data = json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                dict_data = self._binary_to_dict(data)

            return self._dict_to_graph(dict_data)

        except Exception:
            return None

    def deserialize_from_file(self, filepath: str) -> Optional[BlueprintGraph]:
        """Deserialize a blueprint from a file."""
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            return self.deserialize(data)
        except Exception:
            return None

    def _dict_to_graph(self, data: Dict[str, Any]) -> BlueprintGraph:
        """Convert a dictionary to a graph."""
        # Check version and migrate if needed
        format_version = data.get("header", {}).get("format_version", 1)
        while format_version < FORMAT_VERSION:
            if format_version in VERSION_MIGRATIONS:
                data = VERSION_MIGRATIONS[format_version](data)
                format_version += 1
            else:
                raise ValueError(f"No migration for version {format_version}")

        # Validate checksum
        stored_checksum = data.get("header", {}).get("checksum", "")
        calculated_checksum = self._calculate_checksum(data)
        if stored_checksum and stored_checksum != calculated_checksum:
            # Checksum mismatch - data may be corrupted
            pass  # Could raise warning

        # Create graph
        graph_data = data.get("graph", {})
        graph = BlueprintGraph(
            graph_id=graph_data.get("id"),
            name=graph_data.get("name", "Untitled")
        )
        graph.category = graph_data.get("category", "")
        graph.description = data.get("header", {}).get("description", "")
        graph.parent_class = data.get("header", {}).get("parent_class", "")
        graph.is_macro = data.get("header", {}).get("is_macro", False)

        # Deserialize nodes
        node_map: Dict[str, Node] = {}
        for node_data in data.get("nodes", []):
            node = self._deserialize_node(node_data)
            if node:
                graph.add_node(node)
                node_map[node.id] = node

        # Restore entry points
        graph.entry_points = graph_data.get("entry_points", [])

        # Deserialize connections
        for conn_data in data.get("connections", []):
            conn = self._deserialize_connection(conn_data)
            if conn:
                graph.add_connection(conn)

        return graph

    def _deserialize_node(self, data: Dict[str, Any]) -> Optional[Node]:
        """Deserialize a single node."""
        node_type = data.get("type", "")

        # Check for custom deserializer
        if node_type in self._node_deserializers:
            return self._node_deserializers[node_type](data)

        # Get node class from registry
        node_class = NODE_REGISTRY.get(data.get("category", ""))

        # Try to find by type name
        if not node_class:
            for name, cls in NODE_REGISTRY.items():
                if cls.__name__ == node_type:
                    node_class = cls
                    break

        if not node_class:
            # Fallback: create generic node
            from . import node_types
            node_class = getattr(node_types, node_type, None)

        if not node_class:
            return None

        # Create node with position
        position = tuple(data.get("position", [0, 0]))
        node_id = data.get("id")

        # Handle different node constructors
        try:
            if hasattr(node_class, '__init__'):
                import inspect
                sig = inspect.signature(node_class.__init__)
                params = sig.parameters

                kwargs = {"node_id": node_id, "position": position}

                if "variable_name" in params and "variable_name" in data:
                    kwargs["variable_name"] = data["variable_name"]
                if "function_name" in params and "function_name" in data:
                    kwargs["function_name"] = data["function_name"]
                if "macro_name" in params and "macro_name" in data:
                    kwargs["macro_name"] = data["macro_name"]
                if "action_name" in params and "action_name" in data:
                    kwargs["action_name"] = data["action_name"]
                if "event_name" in params and "event_name" in data:
                    kwargs["event_name"] = data["event_name"]

                node = node_class(**kwargs)
            else:
                node = node_class(node_id=node_id, position=position)
        except TypeError:
            node = node_class()
            node.id = node_id
            node.position = position

        # Restore properties
        if data.get("comment"):
            node.comment = data["comment"]
        if data.get("disabled"):
            node.is_disabled = True

        # Restore pin values
        for pin_data in data.get("input_pins", []):
            pin = node.get_input_pin(pin_data.get("name"))
            if pin and "value" in pin_data:
                pin.set_value(self._deserialize_value(pin_data["value"]))

        for pin_data in data.get("output_pins", []):
            pin = node.get_output_pin(pin_data.get("name"))
            if pin and "value" in pin_data:
                pin.set_value(self._deserialize_value(pin_data["value"]))

        return node

    def _deserialize_connection(self, data: Dict[str, Any]) -> Optional[Connection]:
        """Deserialize a connection."""
        return Connection(
            id=data.get("id", str(uuid.uuid4())),
            source_node_id=data.get("source_node", ""),
            source_pin_id=data.get("source_pin", ""),
            target_node_id=data.get("target_node", ""),
            target_pin_id=data.get("target_pin", ""),
            is_execution=data.get("is_execution", False)
        )

    def _deserialize_value(self, data: Any) -> Any:
        """Deserialize a value from JSON format."""
        if data is None:
            return None
        if isinstance(data, (bool, int, float, str)):
            return data
        if isinstance(data, list):
            return [self._deserialize_value(v) for v in data]
        if isinstance(data, dict):
            if "__type__" in data:
                type_name = data["__type__"]
                # Try to reconstruct the object
                from . import data_types
                type_class = getattr(data_types, type_name, None)
                if type_class:
                    kwargs = {k: self._deserialize_value(v) for k, v in data.items() if k != "__type__"}
                    try:
                        return type_class(**kwargs)
                    except TypeError:
                        pass
            return {k: self._deserialize_value(v) for k, v in data.items()}
        return data

    # =========================================================================
    # BINARY FORMAT
    # =========================================================================

    def _dict_to_binary(self, data: Dict) -> bytes:
        """Convert dictionary to binary format."""
        # Simple implementation using JSON internally
        # A production version would use a more compact format
        return json.dumps(data).encode('utf-8')

    def _binary_to_dict(self, data: bytes) -> Dict:
        """Convert binary format to dictionary."""
        return json.loads(data.decode('utf-8'))

    # =========================================================================
    # HEADER INSPECTION
    # =========================================================================

    def read_header(self, data: bytes) -> Optional[BlueprintHeader]:
        """Read just the header without full deserialization."""
        try:
            if data[:2] == b'\x1f\x8b':
                data = gzip.decompress(data)

            dict_data = json.loads(data.decode('utf-8'))
            header = dict_data.get("header", {})

            return BlueprintHeader(
                version=header.get("version", "1.0"),
                format_version=header.get("format_version", 1),
                created_time=header.get("created_time", 0),
                modified_time=header.get("modified_time", 0),
                author=header.get("author", ""),
                description=header.get("description", ""),
                checksum=header.get("checksum", ""),
                node_count=header.get("node_count", 0),
                connection_count=header.get("connection_count", 0),
                parent_class=header.get("parent_class", ""),
                is_macro=header.get("is_macro", False)
            )
        except Exception:
            return None


class IncrementalSerializer:
    """Serializer for incremental/partial updates."""

    def __init__(self, base_serializer: BlueprintSerializer):
        self._base = base_serializer
        self._change_log: List[Dict[str, Any]] = []

    def record_change(
        self,
        change_type: str,
        element_id: str,
        data: Optional[Dict] = None
    ) -> None:
        """Record a change for incremental serialization."""
        self._change_log.append({
            "type": change_type,
            "id": element_id,
            "time": time.time(),
            "data": data
        })

    def get_incremental_update(self) -> bytes:
        """Get serialized incremental update."""
        data = {
            "type": "incremental",
            "changes": self._change_log
        }
        return json.dumps(data).encode('utf-8')

    def apply_incremental_update(
        self,
        graph: BlueprintGraph,
        update: bytes
    ) -> bool:
        """Apply an incremental update to a graph."""
        try:
            data = json.loads(update.decode('utf-8'))
            if data.get("type") != "incremental":
                return False

            for change in data.get("changes", []):
                self._apply_change(graph, change)

            return True
        except Exception:
            return False

    def _apply_change(self, graph: BlueprintGraph, change: Dict) -> None:
        """Apply a single change."""
        change_type = change.get("type")
        element_id = change.get("id")
        data = change.get("data")

        if change_type == "add_node":
            node = self._base._deserialize_node(data)
            if node:
                graph.add_node(node)
        elif change_type == "remove_node":
            graph.remove_node(element_id)
        elif change_type == "move_node":
            node = graph.get_node(element_id)
            if node and data:
                node.position = tuple(data.get("position", [0, 0]))
        elif change_type == "add_connection":
            conn = self._base._deserialize_connection(data)
            if conn:
                graph.add_connection(conn)
        elif change_type == "remove_connection":
            graph.remove_connection(element_id)

    def clear_changes(self) -> None:
        """Clear the change log."""
        self._change_log.clear()


# Convenience functions
def save_blueprint(
    graph: BlueprintGraph,
    filepath: str,
    options: Optional[SerializationOptions] = None
) -> bool:
    """Save a blueprint to a file."""
    serializer = BlueprintSerializer(options)
    return serializer.serialize_to_file(graph, filepath)


def load_blueprint(filepath: str) -> Optional[BlueprintGraph]:
    """Load a blueprint from a file."""
    serializer = BlueprintSerializer()
    return serializer.deserialize_from_file(filepath)


def export_blueprint_json(
    graph: BlueprintGraph,
    pretty: bool = True
) -> str:
    """Export a blueprint as JSON string."""
    options = SerializationOptions(
        format=SerializationFormat.JSON,
        pretty_print=pretty
    )
    serializer = BlueprintSerializer(options)
    return serializer.serialize(graph).decode('utf-8')


def import_blueprint_json(json_str: str) -> Optional[BlueprintGraph]:
    """Import a blueprint from JSON string."""
    serializer = BlueprintSerializer()
    return serializer.deserialize(json_str.encode('utf-8'))
