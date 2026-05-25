"""
Sidechain Bridge -- connects @sidechain decorator to SidechainManager.

This module provides the runtime wiring between the Trinity decorator system
and the engine's sidechain compression.  When a component class is decorated
with @sidechain, the bridge extracts that configuration and creates a
SidechainCompressor registered with the SidechainManager.

Usage::

    config = extract_sidechain_config(MyComponent)
    cfg = create_compressor_config(config, key_bus, target_bus)
    compressor = sidechain_manager.create_compressor(cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from .config import (
    SIDECHAIN_ATTACK_MS,
    SIDECHAIN_KNEE_DB,
    SIDECHAIN_MAKEUP_GAIN_DB,
    SIDECHAIN_RATIO,
    SIDECHAIN_RELEASE_MS,
    SIDECHAIN_THRESHOLD_DB,
)
from .mix_bus import MixBus
from .sidechain import SidechainCompressor, SidechainConfig, SidechainManager


# =============================================================================
# Configuration data class
# =============================================================================


@dataclass
class SidechainBridgeConfig:
    """
    Runtime sidechain configuration extracted from ``@sidechain`` decorator
    attributes.

    Attributes:
        source_bus: Name of the bus whose signal triggers compression (key
                    input).  Mapped to ``SidechainConfig.key_bus`` at
                    registration time.
        attack:     Attack time in seconds.  How quickly compression engages
                    when the key signal exceeds the threshold.
        release:    Release time in seconds.  How quickly compression
                    releases when the key signal falls below the threshold.
        ratio:      Compression ratio (e.g. 4.0 = 4:1).  Higher values mean
                    more aggressive compression.
        threshold_db: Threshold in dB above which compression activates.
                      Defaults to the system-wide ``SIDECHAIN_THRESHOLD_DB``.
        knee_db:    Soft knee width in dB.  A wider knee produces a more
                    gradual compression onset.  Defaults to
                    ``SIDECHAIN_KNEE_DB``.
        makeup_gain_db: Makeup gain in dB applied after compression to
                        compensate for level reduction.
        mix:        Wet/dry mix (0.0 = no compression, 1.0 = full).
    """

    source_bus: str = ""
    attack: float = SIDECHAIN_ATTACK_MS / 1000.0
    release: float = SIDECHAIN_RELEASE_MS / 1000.0
    ratio: float = SIDECHAIN_RATIO
    threshold_db: float = SIDECHAIN_THRESHOLD_DB
    knee_db: float = SIDECHAIN_KNEE_DB
    makeup_gain_db: float = SIDECHAIN_MAKEUP_GAIN_DB
    mix: float = 1.0


# =============================================================================
# Extraction helpers
# =============================================================================


def extract_sidechain_config(component_cls: type) -> SidechainBridgeConfig:
    """
    Extract sidechain configuration from a decorator-annotated component class.

    Reads the private attributes that ``@sidechain``'s *after_steps* callback
    writes to the target (``_sidechain``, ``_sidechain_source_bus``,
    ``_sidechain_attack``, ``_sidechain_release``, ``_sidechain_ratio``).
    Returns defaults if the decorator was not applied.

    Args:
        component_cls: A component class (likely decorated with
                       ``@sidechain``).

    Returns:
        A ``SidechainBridgeConfig`` populated from the decorator's parameters
        or default values.
    """
    if getattr(component_cls, "_sidechain", False):
        return SidechainBridgeConfig(
            source_bus=getattr(component_cls, "_sidechain_source_bus", ""),
            attack=getattr(component_cls, "_sidechain_attack", SIDECHAIN_ATTACK_MS / 1000.0),
            release=getattr(component_cls, "_sidechain_release", SIDECHAIN_RELEASE_MS / 1000.0),
            ratio=getattr(component_cls, "_sidechain_ratio", SIDECHAIN_RATIO),
            threshold_db=getattr(component_cls, "_sidechain_threshold_db", SIDECHAIN_THRESHOLD_DB),
            knee_db=getattr(component_cls, "_sidechain_knee_db", SIDECHAIN_KNEE_DB),
            makeup_gain_db=getattr(component_cls, "_sidechain_makeup_gain_db", SIDECHAIN_MAKEUP_GAIN_DB),
            mix=getattr(component_cls, "_sidechain_mix", 1.0),
        )
    return SidechainBridgeConfig()


def has_sidechain(component_cls: type) -> bool:
    """
    Return True if ``component_cls`` has ``@sidechain`` applied.

    Args:
        component_cls: The component class to inspect.

    Returns:
        True if the decorator was applied to this class.
    """
    return bool(getattr(component_cls, "_sidechain", False))


# =============================================================================
# Compressor config creation
# =============================================================================


def create_compressor_config(
    bridge_config: SidechainBridgeConfig,
    key_bus: MixBus,
    target_bus: MixBus,
    name: str = "",
) -> SidechainConfig:
    """
    Create a ``SidechainConfig`` from a bridge config and resolved buses.

    Maps the decorator's ``source_bus`` name to the actual ``MixBus``
    instance that should act as the key input, and links it to the target
    bus that receives compression.

    Args:
        bridge_config: The ``SidechainBridgeConfig`` extracted from a
                       decorated class.
        key_bus:      The ``MixBus`` instance whose level drives the
                      compressor envelope (the key / sidechain input).
        target_bus:   The ``MixBus`` instance whose gain is attenuated
                      (the target of compression).
        name:         Optional human-readable name for the compressor.

    Returns:
        A ``SidechainConfig`` ready to be passed to
        ``SidechainManager.create_compressor``.
    """
    return SidechainConfig(
        name=name or f"sidechain:{key_bus.name}->{target_bus.name}",
        key_bus=key_bus,
        target_bus=target_bus,
        threshold_db=bridge_config.threshold_db,
        ratio=bridge_config.ratio,
        attack_ms=bridge_config.attack * 1000.0,
        release_ms=bridge_config.release * 1000.0,
        knee_db=bridge_config.knee_db,
        makeup_gain_db=bridge_config.makeup_gain_db,
        enabled=True,
        mix=bridge_config.mix,
    )


# =============================================================================
# SidechainManager integration
# =============================================================================


def register_sidechain(
    sidechain_manager: SidechainManager,
    bridge_config: SidechainBridgeConfig,
    key_bus: MixBus,
    target_bus: MixBus,
    name: str = "",
) -> SidechainCompressor:
    """
    Register a sidechain compressor with the ``SidechainManager``.

    Creates and registers a compressor from the bridge config and resolved
    bus instances.  The compressor begins processing on the next call to
    ``SidechainManager.update()``.

    Args:
        sidechain_manager: The ``SidechainManager`` instance.
        bridge_config:     The ``SidechainBridgeConfig`` from a decorated
                           class.
        key_bus:           The key (sidechain input) ``MixBus``.
        target_bus:        The target ``MixBus`` to compress.
        name:              Optional human-readable name.

    Returns:
        The created ``SidechainCompressor`` instance.
    """
    sc_config = create_compressor_config(bridge_config, key_bus, target_bus, name=name)
    return sidechain_manager.create_compressor(sc_config)


def apply_sidechain(
    sidechain_manager: SidechainManager,
    component_cls: type,
    key_bus: MixBus,
    target_bus: MixBus,
    name: str = "",
) -> Optional[SidechainCompressor]:
    """
    Full wiring: extract config from a component and register with the
    ``SidechainManager`` in one call.

    This is the main entry point for wiring ``@sidechain`` to sidechain
    compression.  Returns ``None`` if the component does not have
    ``@sidechain`` applied.

    Args:
        sidechain_manager: The ``SidechainManager`` instance.
        component_cls:     The decorated component class whose @sidechain
                           settings should be applied.
        key_bus:           The key (sidechain input) ``MixBus``.
        target_bus:        The target ``MixBus`` to compress.
        name:              Optional human-readable name.

    Returns:
        The created ``SidechainCompressor``, or ``None`` if the component
        is not decorated with ``@sidechain``.

    Example::

        mixer = Mixer()
        mixer.initialize()
        kick = mixer.get_bus("kick")
        bass = mixer.get_bus("bass")

        compressor = apply_sidechain(
            mixer.sidechain,
            KickDuckingBass,
            key_bus=kick,
            target_bus=bass,
            name="kick->bass",
        )
    """
    bridge_config = extract_sidechain_config(component_cls)
    if not bridge_config.source_bus:
        return None
    return register_sidechain(
        sidechain_manager,
        bridge_config,
        key_bus,
        target_bus,
        name=name,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SidechainBridgeConfig",
    "extract_sidechain_config",
    "has_sidechain",
    "create_compressor_config",
    "register_sidechain",
    "apply_sidechain",
]
