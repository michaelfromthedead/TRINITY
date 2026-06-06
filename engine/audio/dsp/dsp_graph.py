"""
DSP Graph - Effect Chain and Graph Management

Provides structures for connecting DSP nodes in series (chains) and parallel
configurations to build complex effect processing graphs.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np
import threading
from uuid import uuid4

from .config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    MAX_EFFECT_CHAIN_LENGTH,
    SIMD_ALIGNMENT,
)
from .dsp_node import DSPNode, PassthroughNode, MixNode


class ConnectionType(Enum):
    """Types of connections between DSP nodes."""
    SERIES = auto()      # Output of one feeds input of next
    PARALLEL = auto()    # Nodes process same input, outputs are summed
    SPLIT = auto()       # One output feeds multiple inputs
    MERGE = auto()       # Multiple outputs combine into one


@dataclass
class NodeConnection:
    """Represents a connection between two nodes."""
    source_node_id: str
    source_output: int
    target_node_id: str
    target_input: int
    gain: float = 1.0


@dataclass
class GraphNode:
    """Wrapper for a DSP node in the graph."""
    node_id: str
    dsp_node: DSPNode
    position: int = 0  # For ordering in chains
    is_enabled: bool = True
    connections_in: List[NodeConnection] = field(default_factory=list)
    connections_out: List[NodeConnection] = field(default_factory=list)


class DSPChain(DSPNode):
    """
    A chain of DSP nodes processed in series.

    Audio flows through each node sequentially:
    input -> node1 -> node2 -> ... -> nodeN -> output
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._nodes: List[DSPNode] = []
        self._node_buffers: List[np.ndarray] = []
        self._lock = threading.RLock()
        super().__init__(sample_rate, block_size, num_channels)

    @property
    def nodes(self) -> List[DSPNode]:
        """Get the list of nodes in the chain."""
        return self._nodes.copy()

    @property
    def length(self) -> int:
        """Get the number of nodes in the chain."""
        return len(self._nodes)

    def add_node(self, node: DSPNode, index: Optional[int] = None) -> int:
        """
        Add a node to the chain.

        Args:
            node: The DSP node to add
            index: Position to insert at (None = append to end)

        Returns:
            The index where the node was inserted

        Raises:
            ValueError: If chain is at maximum length
        """
        with self._lock:
            if len(self._nodes) >= MAX_EFFECT_CHAIN_LENGTH:
                raise ValueError(f"Chain at maximum length ({MAX_EFFECT_CHAIN_LENGTH})")

            # Update node settings to match chain
            node.set_sample_rate(self._state.sample_rate)
            node.set_block_size(self._state.block_size)
            node.set_num_channels(self._state.num_channels)

            if index is None:
                self._nodes.append(node)
                index = len(self._nodes) - 1
            else:
                self._nodes.insert(index, node)

            self._update_buffers()
            self._update_latency()
            return index

    def remove_node(self, index: int) -> DSPNode:
        """
        Remove a node from the chain.

        Args:
            index: Index of node to remove

        Returns:
            The removed node
        """
        with self._lock:
            node = self._nodes.pop(index)
            self._update_buffers()
            self._update_latency()
            return node

    def get_node(self, index: int) -> DSPNode:
        """Get a node by index."""
        return self._nodes[index]

    def swap_nodes(self, index1: int, index2: int) -> None:
        """Swap two nodes in the chain."""
        with self._lock:
            self._nodes[index1], self._nodes[index2] = self._nodes[index2], self._nodes[index1]

    def move_node(self, from_index: int, to_index: int) -> None:
        """Move a node from one position to another."""
        with self._lock:
            node = self._nodes.pop(from_index)
            self._nodes.insert(to_index, node)

    def clear(self) -> None:
        """Remove all nodes from the chain."""
        with self._lock:
            self._nodes.clear()
            self._node_buffers.clear()
            self._state.latency_samples = 0

    def _update_buffers(self) -> None:
        """Update internal buffers for node count."""
        num_buffers_needed = max(0, len(self._nodes) - 1)
        while len(self._node_buffers) < num_buffers_needed:
            self._node_buffers.append(
                self._allocate_aligned_buffer(self._state.block_size, self._state.num_channels)
            )
        while len(self._node_buffers) > num_buffers_needed:
            self._node_buffers.pop()

    def _update_latency(self) -> None:
        """Update total chain latency."""
        self._state.latency_samples = sum(node.latency_samples for node in self._nodes)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through the chain."""
        result = sample
        for node in self._nodes:
            if node.is_active and not node.is_bypassed:
                result = node.process_sample(result, channel)
        return result

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block of samples through the chain."""
        if len(self._nodes) == 0:
            np.copyto(output_buffer, input_buffer)
            return

        # Single node - direct processing
        if len(self._nodes) == 1:
            node = self._nodes[0]
            if node.is_active and not node.is_bypassed:
                node.process_block(input_buffer, output_buffer)
            else:
                np.copyto(output_buffer, input_buffer)
            return

        # Multiple nodes - chain through intermediate buffers
        current_input = input_buffer

        for i, node in enumerate(self._nodes):
            # Determine output buffer
            if i == len(self._nodes) - 1:
                current_output = output_buffer
            else:
                # Ensure intermediate buffer matches input shape
                current_output = self._node_buffers[i]
                if current_output.shape != input_buffer.shape:
                    current_output = np.zeros_like(input_buffer)

            # Process or bypass
            if node.is_active and not node.is_bypassed:
                node.process_block(current_input, current_output)
            else:
                np.copyto(current_output, current_input)

            # Next iteration's input is this iteration's output
            current_input = current_output

    def reset(self) -> None:
        """Reset all nodes in the chain."""
        for node in self._nodes:
            node.reset()

    def _on_sample_rate_changed(self) -> None:
        """Update sample rate for all nodes."""
        for node in self._nodes:
            node.set_sample_rate(self._state.sample_rate)

    def _on_block_size_changed(self) -> None:
        """Update block size for all nodes."""
        for node in self._nodes:
            node.set_block_size(self._state.block_size)
        self._update_buffers()

    def _on_channels_changed(self) -> None:
        """Update channel count for all nodes."""
        for node in self._nodes:
            node.set_num_channels(self._state.num_channels)
        self._update_buffers()


class DSPParallel(DSPNode):
    """
    Parallel processing of multiple DSP nodes.

    Same input is fed to all nodes, outputs are summed:
    input -> [node1, node2, ..., nodeN] -> sum -> output
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        normalize_output: bool = True,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._nodes: List[Tuple[DSPNode, float]] = []  # (node, gain)
        self._node_buffers: List[np.ndarray] = []
        self._normalize = normalize_output
        self._lock = threading.RLock()
        super().__init__(sample_rate, block_size, num_channels)

    @property
    def nodes(self) -> List[DSPNode]:
        """Get list of parallel nodes."""
        return [node for node, _ in self._nodes]

    def add_node(self, node: DSPNode, gain: float = 1.0) -> int:
        """Add a node to parallel processing."""
        with self._lock:
            node.set_sample_rate(self._state.sample_rate)
            node.set_block_size(self._state.block_size)
            node.set_num_channels(self._state.num_channels)

            self._nodes.append((node, gain))
            self._node_buffers.append(
                self._allocate_aligned_buffer(self._state.block_size, self._state.num_channels)
            )
            self._update_latency()
            return len(self._nodes) - 1

    def remove_node(self, index: int) -> DSPNode:
        """Remove a node from parallel processing."""
        with self._lock:
            node, _ = self._nodes.pop(index)
            self._node_buffers.pop(index)
            self._update_latency()
            return node

    def set_node_gain(self, index: int, gain: float) -> None:
        """Set the gain for a specific parallel node."""
        node, _ = self._nodes[index]
        self._nodes[index] = (node, gain)

    def _update_latency(self) -> None:
        """Update latency to maximum of all parallel paths."""
        if self._nodes:
            self._state.latency_samples = max(node.latency_samples for node, _ in self._nodes)
        else:
            self._state.latency_samples = 0

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through parallel nodes."""
        if not self._nodes:
            return sample

        result = 0.0
        active_count = 0

        for node, gain in self._nodes:
            if node.is_active and not node.is_bypassed:
                result += node.process_sample(sample, channel) * gain
                active_count += 1

        if self._normalize and active_count > 0:
            result /= active_count

        return result

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block through all parallel nodes and sum."""
        if not self._nodes:
            np.copyto(output_buffer, input_buffer)
            return

        num_channels, num_samples = input_buffer.shape

        # Ensure node buffers are large enough
        if self._node_buffers and self._node_buffers[0].shape != (num_channels, num_samples):
            self._node_buffers = [
                self._allocate_aligned_buffer(num_samples, num_channels)
                for _ in self._nodes
            ]

        output_buffer.fill(0.0)
        active_count = 0

        for i, (node, gain) in enumerate(self._nodes):
            if node.is_active and not node.is_bypassed:
                node.process_block(input_buffer, self._node_buffers[i])
                output_buffer += self._node_buffers[i] * gain
                active_count += 1

        if self._normalize and active_count > 0:
            output_buffer /= active_count

    def reset(self) -> None:
        """Reset all parallel nodes."""
        for node, _ in self._nodes:
            node.reset()

    def _on_sample_rate_changed(self) -> None:
        for node, _ in self._nodes:
            node.set_sample_rate(self._state.sample_rate)

    def _on_block_size_changed(self) -> None:
        for node, _ in self._nodes:
            node.set_block_size(self._state.block_size)
        self._node_buffers = [
            self._allocate_aligned_buffer(self._state.block_size, self._state.num_channels)
            for _ in self._nodes
        ]

    def _on_channels_changed(self) -> None:
        for node, _ in self._nodes:
            node.set_num_channels(self._state.num_channels)
        self._node_buffers = [
            self._allocate_aligned_buffer(self._state.block_size, self._state.num_channels)
            for _ in self._nodes
        ]


class DSPGraph:
    """
    Full DSP graph with arbitrary node connections.

    Supports complex routing including splits, merges, and feedback paths.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._num_channels = num_channels

        self._nodes: Dict[str, GraphNode] = {}
        self._input_node_id: Optional[str] = None
        self._output_node_id: Optional[str] = None
        self._processing_order: List[str] = []
        self._node_buffers: Dict[str, np.ndarray] = {}
        self._lock = threading.RLock()

        # Create default input/output nodes
        self._create_default_io()

    def _create_default_io(self) -> None:
        """Create default input and output nodes."""
        input_node = PassthroughNode(self._sample_rate, self._block_size, self._num_channels)
        output_node = PassthroughNode(self._sample_rate, self._block_size, self._num_channels)

        self._input_node_id = self.add_node(input_node, "input")
        self._output_node_id = self.add_node(output_node, "output")

    def _allocate_buffer(self) -> np.ndarray:
        """Allocate a buffer for node processing."""
        size = self._block_size * self._num_channels
        aligned_size = ((size + SIMD_ALIGNMENT - 1) // SIMD_ALIGNMENT) * SIMD_ALIGNMENT
        buffer = np.zeros(aligned_size, dtype=np.float32)
        return buffer[:size].reshape(self._num_channels, self._block_size)

    def add_node(self, node: DSPNode, node_id: Optional[str] = None) -> str:
        """
        Add a node to the graph.

        Args:
            node: The DSP node to add
            node_id: Optional ID (generated if not provided)

        Returns:
            The node ID
        """
        with self._lock:
            if node_id is None:
                node_id = str(uuid4())

            node.set_sample_rate(self._sample_rate)
            node.set_block_size(self._block_size)
            node.set_num_channels(self._num_channels)

            graph_node = GraphNode(
                node_id=node_id,
                dsp_node=node,
                position=len(self._nodes),
            )

            self._nodes[node_id] = graph_node
            self._node_buffers[node_id] = self._allocate_buffer()
            self._update_processing_order()

            return node_id

    def remove_node(self, node_id: str) -> DSPNode:
        """Remove a node from the graph."""
        with self._lock:
            if node_id in (self._input_node_id, self._output_node_id):
                raise ValueError("Cannot remove input/output nodes")

            graph_node = self._nodes.pop(node_id)
            del self._node_buffers[node_id]

            # Remove all connections involving this node
            for other_node in self._nodes.values():
                other_node.connections_in = [
                    c for c in other_node.connections_in if c.source_node_id != node_id
                ]
                other_node.connections_out = [
                    c for c in other_node.connections_out if c.target_node_id != node_id
                ]

            self._update_processing_order()
            return graph_node.dsp_node

    def connect(
        self,
        source_id: str,
        target_id: str,
        source_output: int = 0,
        target_input: int = 0,
        gain: float = 1.0,
    ) -> None:
        """
        Connect two nodes.

        Args:
            source_id: ID of source node
            target_id: ID of target node
            source_output: Output index on source
            target_input: Input index on target
            gain: Connection gain
        """
        with self._lock:
            if source_id not in self._nodes or target_id not in self._nodes:
                raise ValueError("Invalid node ID")

            connection = NodeConnection(
                source_node_id=source_id,
                source_output=source_output,
                target_node_id=target_id,
                target_input=target_input,
                gain=gain,
            )

            self._nodes[source_id].connections_out.append(connection)
            self._nodes[target_id].connections_in.append(connection)
            self._update_processing_order()

    def disconnect(self, source_id: str, target_id: str) -> None:
        """Disconnect two nodes."""
        with self._lock:
            source_node = self._nodes.get(source_id)
            target_node = self._nodes.get(target_id)

            if source_node:
                source_node.connections_out = [
                    c for c in source_node.connections_out if c.target_node_id != target_id
                ]
            if target_node:
                target_node.connections_in = [
                    c for c in target_node.connections_in if c.source_node_id != source_id
                ]

            self._update_processing_order()

    def _update_processing_order(self) -> None:
        """
        Calculate topological sort of nodes for processing.
        Uses Kahn's algorithm.
        """
        # Build in-degree map
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self._nodes}
        for node in self._nodes.values():
            for conn in node.connections_out:
                in_degree[conn.target_node_id] += 1

        # Start with nodes that have no incoming connections
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        self._processing_order = []

        while queue:
            node_id = queue.pop(0)
            self._processing_order.append(node_id)

            for conn in self._nodes[node_id].connections_out:
                target_id = conn.target_node_id
                in_degree[target_id] -= 1
                if in_degree[target_id] == 0:
                    queue.append(target_id)

        # Check for cycles (not all nodes processed)
        if len(self._processing_order) != len(self._nodes):
            # Fall back to simple ordering for graphs with cycles
            # This handles feedback paths but may not be optimal
            self._processing_order = list(self._nodes.keys())

    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """
        Process audio through the graph.

        Args:
            input_buffer: Input samples, shape (channels, samples)

        Returns:
            Processed samples
        """
        with self._lock:
            # Copy input to input node's buffer
            if self._input_node_id:
                np.copyto(self._node_buffers[self._input_node_id], input_buffer)

            # Process nodes in order
            for node_id in self._processing_order:
                graph_node = self._nodes[node_id]

                if not graph_node.is_enabled:
                    continue

                # Gather inputs
                if graph_node.connections_in:
                    # Sum all incoming connections
                    self._node_buffers[node_id].fill(0.0)
                    for conn in graph_node.connections_in:
                        source_buffer = self._node_buffers[conn.source_node_id]
                        self._node_buffers[node_id] += source_buffer * conn.gain

                # Process through DSP node
                if graph_node.dsp_node.is_active:
                    temp_buffer = self._allocate_buffer()
                    graph_node.dsp_node.process_block(
                        self._node_buffers[node_id],
                        temp_buffer,
                    )
                    np.copyto(self._node_buffers[node_id], temp_buffer)

            # Return output node's buffer
            if self._output_node_id:
                return self._node_buffers[self._output_node_id].copy()
            return input_buffer.copy()

    def get_node(self, node_id: str) -> Optional[DSPNode]:
        """Get a node by ID."""
        graph_node = self._nodes.get(node_id)
        return graph_node.dsp_node if graph_node else None

    def get_all_nodes(self) -> Dict[str, DSPNode]:
        """Get all nodes in the graph."""
        return {node_id: gn.dsp_node for node_id, gn in self._nodes.items()}

    def set_node_enabled(self, node_id: str, enabled: bool) -> None:
        """Enable or disable a node."""
        if node_id in self._nodes:
            self._nodes[node_id].is_enabled = enabled

    def reset(self) -> None:
        """Reset all nodes in the graph."""
        for graph_node in self._nodes.values():
            graph_node.dsp_node.reset()

    def clear(self) -> None:
        """Clear all nodes except input/output."""
        with self._lock:
            nodes_to_remove = [
                node_id for node_id in self._nodes
                if node_id not in (self._input_node_id, self._output_node_id)
            ]
            for node_id in nodes_to_remove:
                self.remove_node(node_id)

    @property
    def input_node_id(self) -> Optional[str]:
        """Get the input node ID."""
        return self._input_node_id

    @property
    def output_node_id(self) -> Optional[str]:
        """Get the output node ID."""
        return self._output_node_id

    @property
    def latency_samples(self) -> int:
        """Calculate total graph latency along the longest path."""
        if not self._nodes:
            return 0

        # Simple calculation: sum of all nodes in processing order
        # For more accurate calculation, would need to trace all paths
        return sum(
            self._nodes[node_id].dsp_node.latency_samples
            for node_id in self._processing_order
        )


class EffectRack(DSPNode):
    """
    A convenient effect rack combining serial and parallel processing.

    Supports:
    - Main chain (series processing)
    - Parallel sends with return mixing
    - Wet/dry control
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._main_chain = DSPChain(sample_rate, block_size, num_channels)
        self._sends: List[Tuple[DSPChain, float]] = []  # (chain, send_level)
        self._send_buffers: List[np.ndarray] = []
        super().__init__(sample_rate, block_size, num_channels)
        self._wet_mix = self.add_parameter('wet', 1.0)
        self._dry_buffer = self._allocate_aligned_buffer(block_size, num_channels)

    @property
    def main_chain(self) -> DSPChain:
        """Get the main effect chain."""
        return self._main_chain

    def add_to_chain(self, node: DSPNode, index: Optional[int] = None) -> int:
        """Add a node to the main chain."""
        return self._main_chain.add_node(node, index)

    def add_send(self, chain: Optional[DSPChain] = None, level: float = 1.0) -> int:
        """
        Add a parallel send chain.

        Args:
            chain: The send chain (created if None)
            level: Send level (0-1)

        Returns:
            Send index
        """
        if chain is None:
            chain = DSPChain(self._state.sample_rate, self._state.block_size, self._state.num_channels)
        else:
            chain.set_sample_rate(self._state.sample_rate)
            chain.set_block_size(self._state.block_size)
            chain.set_num_channels(self._state.num_channels)

        self._sends.append((chain, level))
        self._send_buffers.append(
            self._allocate_aligned_buffer(self._state.block_size, self._state.num_channels)
        )
        return len(self._sends) - 1

    def set_send_level(self, index: int, level: float) -> None:
        """Set the level for a send."""
        chain, _ = self._sends[index]
        self._sends[index] = (chain, level)

    def set_wet_mix(self, wet: float) -> None:
        """Set wet/dry mix (0 = dry, 1 = wet)."""
        self._wet_mix.set_value(max(0.0, min(1.0, wet)))

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through the rack."""
        # Process main chain
        wet = self._main_chain.process_sample(sample, channel)

        # Add sends
        for chain, level in self._sends:
            if chain.is_active:
                wet += chain.process_sample(sample, channel) * level

        # Mix wet/dry
        mix = self._wet_mix.advance()
        return wet * mix + sample * (1.0 - mix)

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block through the rack."""
        num_channels, num_samples = input_buffer.shape

        # Ensure buffers are large enough
        if self._dry_buffer.shape != (num_channels, num_samples):
            self._dry_buffer = self._allocate_aligned_buffer(num_samples, num_channels)
            self._send_buffers = [
                self._allocate_aligned_buffer(num_samples, num_channels)
                for _ in self._sends
            ]

        # Save dry signal
        np.copyto(self._dry_buffer, input_buffer)

        # Process main chain
        self._main_chain.process_block(input_buffer, output_buffer)

        # Add sends
        for i, (chain, level) in enumerate(self._sends):
            if chain.is_active and level > 0:
                chain.process_block(input_buffer, self._send_buffers[i])
                output_buffer += self._send_buffers[i] * level

        # Apply wet/dry mix
        wet_values = self._wet_mix.advance_block(num_samples)
        dry_values = 1.0 - wet_values

        for ch in range(output_buffer.shape[0]):
            output_buffer[ch] = output_buffer[ch] * wet_values + self._dry_buffer[ch] * dry_values

    def reset(self) -> None:
        """Reset the rack and all chains."""
        self._main_chain.reset()
        for chain, _ in self._sends:
            chain.reset()

    def _on_sample_rate_changed(self) -> None:
        self._main_chain.set_sample_rate(self._state.sample_rate)
        for chain, _ in self._sends:
            chain.set_sample_rate(self._state.sample_rate)

    def _on_block_size_changed(self) -> None:
        self._main_chain.set_block_size(self._state.block_size)
        for chain, _ in self._sends:
            chain.set_block_size(self._state.block_size)
        self._dry_buffer = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )
        self._send_buffers = [
            self._allocate_aligned_buffer(self._state.block_size, self._state.num_channels)
            for _ in self._sends
        ]

    def _on_channels_changed(self) -> None:
        self._main_chain.set_num_channels(self._state.num_channels)
        for chain, _ in self._sends:
            chain.set_num_channels(self._state.num_channels)
