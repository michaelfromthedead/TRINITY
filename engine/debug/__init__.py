"""
Debug and diagnostics layer for the game engine.

Provides comprehensive debugging, profiling, and testing infrastructure:
- Logging: Structured, categorized logging with multiple output targets
- Console: In-game console with CVars, commands, and scripting
- Visual: Debug drawing, overlays, and render views
- Profiling: CPU, GPU, memory, and network profiling
- Crash: Assertions, crash handling, and reporting
- Replay: Recording and playback for debugging
- Testing: Unit, integration, and automation testing framework
"""

from . import testing

__all__ = [
    "testing",
]
