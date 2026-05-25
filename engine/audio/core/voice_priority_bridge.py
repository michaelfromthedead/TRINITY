"""
Voice Priority Bridge -- connects @voice_priority decorator to VoiceManager.

This module provides the runtime wiring between the Trinity decorator system
and the engine's voice management.  When a component class is decorated with
@voice_priority, the bridge extracts that configuration and applies it to
the VoiceManager's allocation decisions.

Usage::

    config = extract_voice_config(MyComponent)
    configure_source(source, config)
    register_component_rules(voice_manager, MyComponent)
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PRIORITY_NORMAL, VoiceStealStrategy
from .audio_source import AudioSource
from .voice_manager import VoiceManager


# =============================================================================
# Configuration data class
# =============================================================================


@dataclass
class VoicePriorityConfig:
    """
    Runtime voice priority configuration extracted from
    ``@voice_priority`` decorator attributes.

    Attributes:
        priority:   Voice priority level (0-100, higher = more important).
                    Maps directly to the ``AudioSource.priority`` and
                    ``Voice.priority`` fields that the VoiceManager uses
                    during steal / virtualize decisions.
        virtualize: Whether this voice can be made virtual when the global
                    voice limit is reached.  A virtual voice is tracked but
                    not rendered; it may be promoted back to real when
                    capacity becomes available.
        steal_oldest: Whether this voice's component participates in
                      priority-based stealing.  When True the VoiceManager
                      will ensure its steal strategy allows lower-priority
                      voices to be pre-empted.
    """

    priority: int = PRIORITY_NORMAL
    virtualize: bool = True
    steal_oldest: bool = True


# =============================================================================
# Extraction helpers
# =============================================================================


def extract_voice_config(component_cls: type) -> VoicePriorityConfig:
    """
    Extract voice priority configuration from a decorator-annotated component
    class.

    Reads the private attributes that ``@voice_priority``'s *after_steps*
    callback writes to the target (``_voice_priority``,
    ``_voice_priority_value``, ``_voice_virtualize``,
    ``_voice_steal_oldest``).  Returns defaults if the decorator was not
    applied.

    Args:
        component_cls: A component class (likely decorated with
                       ``@voice_priority``).

    Returns:
        A ``VoicePriorityConfig`` populated from the decorator's parameters
        or default values.
    """
    if getattr(component_cls, "_voice_priority", False):
        return VoicePriorityConfig(
            priority=getattr(component_cls, "_voice_priority_value", PRIORITY_NORMAL),
            virtualize=getattr(component_cls, "_voice_virtualize", True),
            steal_oldest=getattr(component_cls, "_voice_steal_oldest", True),
        )
    return VoicePriorityConfig()


def has_voice_priority(component_cls: type) -> bool:
    """
    Return True if ``component_cls`` has ``@voice_priority`` applied.

    Args:
        component_cls: The component class to inspect.

    Returns:
        True if the decorator was applied to this class.
    """
    return bool(getattr(component_cls, "_voice_priority", False))


# =============================================================================
# AudioSource configuration
# =============================================================================


def configure_source(source: AudioSource, config: VoicePriorityConfig) -> None:
    """
    Apply voice priority configuration to an ``AudioSource``.

    Sets the source's ``priority`` field, which the ``VoiceManager`` reads
    during ``allocate_voice`` to determine steal / virtualize ordering.

    Call this when creating a new ``AudioSource`` that is associated with a
    ``@voice_priority``-decorated component.

    Args:
        source: The ``AudioSource`` to configure.
        config: The ``VoicePriorityConfig`` extracted from the component.

    Example::

        source = AudioSource()
        cfg = extract_voice_config(MyComponent)
        configure_source(source, cfg)
    """
    source.priority = config.priority
    source.is_virtual = config.virtualize


def configure_source_from_component(
    source: AudioSource, component_cls: type
) -> None:
    """
    Convenience: extract config from a component class and apply to source.

    Equivalent to ``configure_source(source, extract_voice_config(cls))``.

    Args:
        source: The ``AudioSource`` to configure.
        component_cls: The decorated component class.
    """
    config = extract_voice_config(component_cls)
    configure_source(source, config)


# =============================================================================
# VoiceManager integration
# =============================================================================


def register_component_rules(
    voice_manager: VoiceManager,
    component_cls: type,
) -> None:
    """
    Register component-level voice rules with the ``VoiceManager``.

    When ``steal_oldest`` is set on the component, this function ensures the
    manager's steal strategy is not ``VoiceStealStrategy.NONE`` so that
    priority-based stealing can take effect.  Components that explicitly
    opt out of stealing (``steal_oldest=False``) leave the strategy
    unchanged.

    This is safe to call repeatedly -- it only transitions from NONE to
    LOWEST_PRIORITY and will not overwrite an explicitly-chosen strategy.

    Args:
        voice_manager: The ``VoiceManager`` instance.
        component_cls: The decorated component class.
    """
    config = extract_voice_config(component_cls)

    if config.steal_oldest:
        current = voice_manager._steal_strategy
        if current == VoiceStealStrategy.NONE:
            voice_manager.set_steal_strategy(VoiceStealStrategy.LOWEST_PRIORITY)


def apply_component_to_allocation(
    voice_manager: VoiceManager,
    source: AudioSource,
    component_cls: type,
) -> None:
    """
    Full wiring: configure an ``AudioSource`` and update the
    ``VoiceManager`` in one call.

    This is the main entry point for wiring ``@voice_priority`` to voice
    management.  Call it before ``voice_manager.allocate_voice(source)``.

    Args:
        voice_manager: The ``VoiceManager`` instance.
        source: The ``AudioSource`` being allocated.
        component_cls: The decorated component class whose @voice_priority
                       settings should be applied.

    Example::

        source = audio_engine.create_source()
        apply_component_to_allocation(vm, source, MyComponent)
        result = vm.allocate_voice(source)
    """
    config = extract_voice_config(component_cls)
    configure_source(source, config)
    if config.steal_oldest:
        current = voice_manager._steal_strategy
        if current == VoiceStealStrategy.NONE:
            voice_manager.set_steal_strategy(VoiceStealStrategy.LOWEST_PRIORITY)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "VoicePriorityConfig",
    "extract_voice_config",
    "has_voice_priority",
    "configure_source",
    "configure_source_from_component",
    "register_component_rules",
    "apply_component_to_allocation",
]
