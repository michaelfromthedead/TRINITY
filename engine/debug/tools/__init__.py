"""
Debug Tools - Cheats, time control, debug camera, AI/physics/network debugging.

This module provides runtime debugging tools for game development:
- CheatManager: God mode, fly, teleport, spawn, etc.
- TimeController: Pause, slow motion, frame stepping
- DebugCamera: Free cam, orbit, follow modes
- AIDebugger: Perception viz, behavior tree viewer, blackboard display
- PhysicsDebugger: Collision visualization, physics pause/step
- NetworkDebugger: Latency simulation, packet loss, bandwidth limits
"""

from __future__ import annotations

from .cheats import CheatManager, CheatCommand, CheatFlags
from .time_control import TimeController, TimeState
from .debug_camera import DebugCamera, DebugCameraMode
from .ai_debug import AIDebugger, AIDebugState, PerceptionVisual
from .physics_debug import PhysicsDebugger, PhysicsDebugState, BodyInspection
from .network_debug import NetworkDebugger, NetworkStats, NetworkSimulation

__all__ = [
    # Cheats
    "CheatManager",
    "CheatCommand",
    "CheatFlags",
    # Time control
    "TimeController",
    "TimeState",
    # Debug camera
    "DebugCamera",
    "DebugCameraMode",
    # AI debugging
    "AIDebugger",
    "AIDebugState",
    "PerceptionVisual",
    # Physics debugging
    "PhysicsDebugger",
    "PhysicsDebugState",
    "BodyInspection",
    # Network debugging
    "NetworkDebugger",
    "NetworkStats",
    "NetworkSimulation",
]
