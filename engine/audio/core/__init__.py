"""
Audio Core Package.

Core audio engine components including voice management, spatial audio,
and the @voice_priority decorator bridge.
"""

from .voice_priority_bridge import (
    VoicePriorityConfig,
    apply_component_to_allocation,
    configure_source,
    configure_source_from_component,
    extract_voice_config,
    has_voice_priority,
    register_component_rules,
)

__all__ = [
    "VoicePriorityConfig",
    "apply_component_to_allocation",
    "configure_source",
    "configure_source_from_component",
    "extract_voice_config",
    "has_voice_priority",
    "register_component_rules",
]
