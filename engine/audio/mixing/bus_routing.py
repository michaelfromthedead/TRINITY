"""
Bus Routing for audio signal flow.

This module handles how audio signals are routed between buses:
- Parent output: Hierarchical routing through the bus tree
- Aux sends: Parallel routing to effect buses with send levels
- Direct out: Bypass hierarchy and route directly to output

Each bus can have multiple aux sends to different effect buses,
allowing for complex routing scenarios like reverb, delay, etc.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4

from .config import (
    DEFAULT_SEND_LEVEL,
    LOCK_TIMEOUT,
    MAX_AUX_SENDS,
    MAX_SEND_LEVEL,
    MIN_VOLUME_DB,
    clamp,
    db_to_linear,
)
from .mix_bus import MixBus, BusType


class RoutingMode(Enum):
    """How audio is routed from a bus."""
    PARENT = "parent"      # Normal hierarchical routing
    DIRECT = "direct"      # Bypass hierarchy, go to target
    PRE_FADER = "pre"      # Aux send before fader
    POST_FADER = "post"    # Aux send after fader


@dataclass
class AuxSend:
    """
    An auxiliary send from one bus to another.

    Aux sends allow parallel routing of audio to effect buses
    (like reverb or delay) while maintaining the normal signal path.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    source_bus: Optional[MixBus] = None
    target_bus: Optional[MixBus] = None
    send_level_db: float = DEFAULT_SEND_LEVEL
    mode: RoutingMode = RoutingMode.POST_FADER
    enabled: bool = True

    @property
    def send_level_linear(self) -> float:
        """Get send level as linear amplitude."""
        return db_to_linear(self.send_level_db)

    @property
    def target(self) -> Optional[MixBus]:
        """Alias for target_bus for convenience."""
        return self.target_bus

    @property
    def level(self) -> float:
        """Get send level as linear amplitude (alias for send_level_linear)."""
        return self.send_level_linear

    @level.setter
    def level(self, value: float) -> None:
        """Set send level as linear amplitude."""
        if value > 0:
            self.send_level_db = 20.0 * (value if value >= 1.0 else -1.0 / value) if value != 1.0 else 0.0
            # Simplified: use logarithm
            import math
            self.send_level_db = 20.0 * math.log10(max(0.0001, value))
        else:
            self.send_level_db = MIN_VOLUME_DB

    @property
    def pre_fader(self) -> bool:
        """Check if this is a pre-fader send."""
        return self.mode == RoutingMode.PRE_FADER

    @pre_fader.setter
    def pre_fader(self, value: bool) -> None:
        """Set pre-fader mode."""
        self.mode = RoutingMode.PRE_FADER if value else RoutingMode.POST_FADER

    def set_level(self, level_db: float) -> None:
        """Set the send level in dB."""
        self.send_level_db = clamp(level_db, MIN_VOLUME_DB, MAX_SEND_LEVEL)

    def copy(self) -> AuxSend:
        """Create a copy of this aux send."""
        return AuxSend(
            id=self.id,
            source_bus=self.source_bus,
            target_bus=self.target_bus,
            send_level_db=self.send_level_db,
            mode=self.mode,
            enabled=self.enabled,
        )


@dataclass
class DirectOutput:
    """
    A direct output routing that bypasses the bus hierarchy.

    Direct outputs send audio directly to a target without going
    through the normal parent chain.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    source_bus: Optional[MixBus] = None
    target_bus: Optional[MixBus] = None
    level_db: float = 0.0
    enabled: bool = True

    @property
    def level_linear(self) -> float:
        """Get output level as linear amplitude."""
        return db_to_linear(self.level_db)


class BusRouter:
    """
    Manages routing between buses.

    The router tracks:
    - Aux sends from each bus to effect buses
    - Direct outputs that bypass the hierarchy
    - Routing state for snapshot/restore

    Thread Safety:
        All operations are protected by a lock for safe multi-threaded access.
    """

    def __init__(self) -> None:
        """Initialize the bus router."""
        self._lock = threading.RLock()
        self._aux_sends: dict[str, list[AuxSend]] = {}  # source_bus.id -> sends
        self._direct_outputs: dict[str, DirectOutput] = {}  # source_bus.id -> output
        self._aux_buses: dict[str, MixBus] = {}  # Registered aux/effect buses

    # =========================================================================
    # Aux Bus Management
    # =========================================================================

    def register_aux_bus(self, bus: MixBus) -> None:
        """
        Register a bus as an aux/effect bus.

        Args:
            bus: Bus to register as aux target.
        """
        with self._lock:
            self._aux_buses[bus.id] = bus

    def unregister_aux_bus(self, bus: MixBus) -> None:
        """
        Unregister an aux bus and remove all sends to it.

        Args:
            bus: Bus to unregister.
        """
        with self._lock:
            self._aux_buses.pop(bus.id, None)

            # Remove sends targeting this bus
            for source_id, sends in list(self._aux_sends.items()):
                self._aux_sends[source_id] = [
                    s for s in sends if s.target_bus is not bus
                ]

    def get_aux_buses(self) -> list[MixBus]:
        """Get all registered aux buses."""
        with self._lock:
            return list(self._aux_buses.values())

    # =========================================================================
    # Aux Send Management
    # =========================================================================

    def create_send(
        self,
        source: MixBus,
        target: MixBus,
        level_db: float = DEFAULT_SEND_LEVEL,
        mode: RoutingMode = RoutingMode.POST_FADER,
    ) -> AuxSend:
        """
        Create an aux send from source to target bus.

        Args:
            source: Source bus to send from.
            target: Target aux bus to send to.
            level_db: Send level in dB.
            mode: Pre-fader or post-fader send.

        Returns:
            The created AuxSend.

        Raises:
            ValueError: If max sends exceeded or invalid routing.
        """
        with self._lock:
            sends = self._aux_sends.get(source.id, [])

            if len(sends) >= MAX_AUX_SENDS:
                raise ValueError(f"Maximum aux sends ({MAX_AUX_SENDS}) exceeded")

            if source is target:
                raise ValueError("Cannot send bus to itself")

            # Check for existing send to same target
            for existing in sends:
                if existing.target_bus is target:
                    raise ValueError(f"Send to {target.name} already exists")

            send = AuxSend(
                source_bus=source,
                target_bus=target,
                send_level_db=clamp(level_db, MIN_VOLUME_DB, MAX_SEND_LEVEL),
                mode=mode,
            )

            sends.append(send)
            self._aux_sends[source.id] = sends

            return send

    def remove_send(self, send: AuxSend) -> bool:
        """
        Remove an aux send.

        Args:
            send: The send to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if send.source_bus is None:
                return False

            sends = self._aux_sends.get(send.source_bus.id, [])
            if send in sends:
                sends.remove(send)
                return True
            return False

    def remove_all_sends(self, source: MixBus) -> int:
        """
        Remove all sends from a source bus.

        Args:
            source: Source bus to clear sends from.

        Returns:
            Number of sends removed.
        """
        with self._lock:
            sends = self._aux_sends.pop(source.id, [])
            return len(sends)

    def get_sends(self, source: MixBus) -> list[AuxSend]:
        """
        Get all aux sends from a source bus.

        Args:
            source: Source bus.

        Returns:
            List of aux sends (copies).
        """
        with self._lock:
            sends = self._aux_sends.get(source.id, [])
            return [s.copy() for s in sends]

    def get_send_by_id(self, send_id: str) -> Optional[AuxSend]:
        """
        Find an aux send by its ID.

        Args:
            send_id: ID of the send.

        Returns:
            The send if found, None otherwise.
        """
        with self._lock:
            for sends in self._aux_sends.values():
                for send in sends:
                    if send.id == send_id:
                        return send.copy()
            return None

    def set_send_level(self, send: AuxSend, level_db: float) -> None:
        """
        Set the level of an aux send.

        Args:
            send: The send to modify.
            level_db: New level in dB.
        """
        with self._lock:
            if send.source_bus is None:
                return

            sends = self._aux_sends.get(send.source_bus.id, [])
            for s in sends:
                if s.id == send.id:
                    s.set_level(level_db)
                    return

    def enable_send(self, send: AuxSend, enabled: bool = True) -> None:
        """
        Enable or disable an aux send.

        Args:
            send: The send to modify.
            enabled: Whether the send should be enabled.
        """
        with self._lock:
            if send.source_bus is None:
                return

            sends = self._aux_sends.get(send.source_bus.id, [])
            for s in sends:
                if s.id == send.id:
                    s.enabled = enabled
                    return

    # =========================================================================
    # Direct Output Management
    # =========================================================================

    def set_direct_output(
        self,
        source: MixBus,
        target: MixBus,
        level_db: float = 0.0,
    ) -> DirectOutput:
        """
        Set a direct output from source to target, bypassing hierarchy.

        Args:
            source: Source bus.
            target: Target bus (usually master or a submix).
            level_db: Output level in dB.

        Returns:
            The created DirectOutput.

        Raises:
            ValueError: If routing is invalid.
        """
        if source is target:
            raise ValueError("Cannot route bus to itself")

        with self._lock:
            output = DirectOutput(
                source_bus=source,
                target_bus=target,
                level_db=level_db,
            )
            self._direct_outputs[source.id] = output
            return output

    def clear_direct_output(self, source: MixBus) -> bool:
        """
        Clear the direct output from a source bus.

        Args:
            source: Source bus.

        Returns:
            True if output was cleared.
        """
        with self._lock:
            return self._direct_outputs.pop(source.id, None) is not None

    def get_direct_output(self, source: MixBus) -> Optional[DirectOutput]:
        """
        Get the direct output for a source bus.

        Args:
            source: Source bus.

        Returns:
            DirectOutput if set, None otherwise.
        """
        with self._lock:
            output = self._direct_outputs.get(source.id)
            if output:
                return DirectOutput(
                    id=output.id,
                    source_bus=output.source_bus,
                    target_bus=output.target_bus,
                    level_db=output.level_db,
                    enabled=output.enabled,
                )
            return None

    def has_direct_output(self, source: MixBus) -> bool:
        """Check if a bus has a direct output configured."""
        with self._lock:
            return source.id in self._direct_outputs

    # =========================================================================
    # Routing Queries
    # =========================================================================

    def get_effective_routing(self, bus: MixBus) -> dict:
        """
        Get the effective routing configuration for a bus.

        Args:
            bus: Bus to query.

        Returns:
            Dictionary with routing information.
        """
        with self._lock:
            direct = self._direct_outputs.get(bus.id)
            sends = self._aux_sends.get(bus.id, [])

            return {
                "bus_id": bus.id,
                "bus_name": bus.name,
                "parent": bus.parent.name if bus.parent else None,
                "has_direct_output": direct is not None,
                "direct_target": direct.target_bus.name if direct and direct.target_bus else None,
                "aux_sends": [
                    {
                        "id": s.id,
                        "target": s.target_bus.name if s.target_bus else None,
                        "level_db": s.send_level_db,
                        "mode": s.mode.value,
                        "enabled": s.enabled,
                    }
                    for s in sends
                ],
            }

    def get_all_sources_for_target(self, target: MixBus) -> list[tuple[MixBus, AuxSend]]:
        """
        Get all buses that send to a target.

        Args:
            target: Target bus.

        Returns:
            List of (source_bus, send) tuples.
        """
        with self._lock:
            sources = []
            for source_id, sends in self._aux_sends.items():
                for send in sends:
                    if send.target_bus is target and send.enabled:
                        if send.source_bus:
                            sources.append((send.source_bus, send.copy()))
            return sources

    # =========================================================================
    # State Management
    # =========================================================================

    def get_routing_state(self) -> dict:
        """
        Get the complete routing state for snapshot.

        Returns:
            Dictionary containing all routing configuration.
        """
        with self._lock:
            return {
                "aux_sends": {
                    source_id: [
                        {
                            "id": s.id,
                            "source_id": s.source_bus.id if s.source_bus else None,
                            "target_id": s.target_bus.id if s.target_bus else None,
                            "level_db": s.send_level_db,
                            "mode": s.mode.value,
                            "enabled": s.enabled,
                        }
                        for s in sends
                    ]
                    for source_id, sends in self._aux_sends.items()
                },
                "direct_outputs": {
                    source_id: {
                        "id": o.id,
                        "source_id": o.source_bus.id if o.source_bus else None,
                        "target_id": o.target_bus.id if o.target_bus else None,
                        "level_db": o.level_db,
                        "enabled": o.enabled,
                    }
                    for source_id, o in self._direct_outputs.items()
                },
                "aux_bus_ids": list(self._aux_buses.keys()),
            }

    def clear(self) -> None:
        """Clear all routing configuration."""
        with self._lock:
            self._aux_sends.clear()
            self._direct_outputs.clear()
            self._aux_buses.clear()

    def __repr__(self) -> str:
        with self._lock:
            send_count = sum(len(s) for s in self._aux_sends.values())
            direct_count = len(self._direct_outputs)
            return (
                f"BusRouter(aux_sends={send_count}, "
                f"direct_outputs={direct_count}, "
                f"aux_buses={len(self._aux_buses)})"
            )
