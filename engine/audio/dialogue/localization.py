"""
Localization Module.

Language switching, audio bank management, and localized content handling
for voice-over and dialogue systems.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .config import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    EVENT_LANGUAGE_CHANGED,
)
from .vo_line import LipSyncData, SubtitleData, VOLine


@dataclass
class LocalizedAsset:
    """
    A localized audio asset with variants for different languages.
    """
    asset_id: str
    base_path: str = ""
    variants: dict[str, str] = field(default_factory=dict)  # language -> path
    durations: dict[str, float] = field(default_factory=dict)  # language -> duration_ms
    subtitles: dict[str, str] = field(default_factory=dict)  # language -> text
    lip_sync: dict[str, LipSyncData] = field(default_factory=dict)  # language -> data

    def get_path(self, language: str) -> str:
        """Get asset path for a language."""
        if language in self.variants:
            return self.variants[language]
        if DEFAULT_LANGUAGE in self.variants:
            return self.variants[DEFAULT_LANGUAGE]
        return self.base_path

    def get_duration(self, language: str) -> float:
        """Get duration for a language."""
        if language in self.durations:
            return self.durations[language]
        if DEFAULT_LANGUAGE in self.durations:
            return self.durations[DEFAULT_LANGUAGE]
        return 0.0

    def get_subtitle(self, language: str) -> str:
        """Get subtitle text for a language."""
        if language in self.subtitles:
            return self.subtitles[language]
        if DEFAULT_LANGUAGE in self.subtitles:
            return self.subtitles[DEFAULT_LANGUAGE]
        return ""

    def get_lip_sync(self, language: str) -> Optional[LipSyncData]:
        """Get lip sync data for a language."""
        if language in self.lip_sync:
            return self.lip_sync[language]
        if DEFAULT_LANGUAGE in self.lip_sync:
            return self.lip_sync[DEFAULT_LANGUAGE]
        return None

    def has_language(self, language: str) -> bool:
        """Check if asset has a variant for the language."""
        return language in self.variants

    def add_variant(
        self,
        language: str,
        path: str,
        duration_ms: float = 0.0,
        subtitle: str = "",
        lip_sync: Optional[LipSyncData] = None,
    ) -> None:
        """Add a language variant."""
        self.variants[language] = path
        if duration_ms > 0:
            self.durations[language] = duration_ms
        if subtitle:
            self.subtitles[language] = subtitle
        if lip_sync:
            self.lip_sync[language] = lip_sync


@dataclass
class AudioBank:
    """
    Collection of audio assets for a specific language and category.
    """
    bank_id: str
    language: str
    category: str = ""  # e.g., "dialogue", "barks", "ambient"
    assets: dict[str, LocalizedAsset] = field(default_factory=dict)
    base_path: str = ""
    is_loaded: bool = field(default=False, init=False)
    _size_bytes: int = field(default=0, init=False)

    def add_asset(self, asset: LocalizedAsset) -> None:
        """Add an asset to the bank."""
        self.assets[asset.asset_id] = asset

    def get_asset(self, asset_id: str) -> Optional[LocalizedAsset]:
        """Get an asset by ID."""
        return self.assets.get(asset_id)

    def remove_asset(self, asset_id: str) -> bool:
        """Remove an asset by ID."""
        if asset_id in self.assets:
            del self.assets[asset_id]
            return True
        return False

    @property
    def asset_count(self) -> int:
        """Get number of assets in bank."""
        return len(self.assets)

    @property
    def size_bytes(self) -> int:
        """Get estimated size of bank in bytes."""
        return self._size_bytes

    def __iter__(self) -> Iterator[LocalizedAsset]:
        return iter(self.assets.values())


class LocalizationManager:
    """
    Manages localization for voice-over content.

    Handles language switching, audio bank loading/unloading,
    and provides localized content access.
    """

    def __init__(
        self,
        default_language: str = DEFAULT_LANGUAGE,
        supported_languages: Optional[tuple[str, ...]] = None,
        on_language_changed: Optional[Callable[[str, str], None]] = None,
        on_bank_loaded: Optional[Callable[[AudioBank], None]] = None,
        on_bank_unloaded: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initialize the localization manager.

        Args:
            default_language: Default language code
            supported_languages: Tuple of supported language codes
            on_language_changed: Callback (old_lang, new_lang)
            on_bank_loaded: Callback when bank is loaded
            on_bank_unloaded: Callback when bank is unloaded
        """
        self._current_language = default_language
        self._supported_languages = supported_languages or SUPPORTED_LANGUAGES
        self._banks: dict[str, AudioBank] = {}  # bank_id -> bank
        self._loaded_banks: set[str] = set()
        self._lock = threading.RLock()

        # Callbacks
        self._on_language_changed = on_language_changed
        self._on_bank_loaded = on_bank_loaded
        self._on_bank_unloaded = on_bank_unloaded

        # Localized assets registry
        self._assets: dict[str, LocalizedAsset] = {}

        # Fallback chain (e.g., pt-BR -> pt -> en)
        self._fallback_chain: dict[str, list[str]] = {}

    @property
    def current_language(self) -> str:
        """Get current language."""
        return self._current_language

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Get supported languages."""
        return self._supported_languages

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in self._supported_languages

    def set_language(self, language: str) -> bool:
        """
        Set the current language.

        Args:
            language: Language code to set

        Returns:
            True if language was changed
        """
        if not self.is_language_supported(language):
            return False

        if language == self._current_language:
            return False

        with self._lock:
            old_language = self._current_language
            self._current_language = language

            if self._on_language_changed:
                self._on_language_changed(old_language, language)

            return True

    def set_fallback_chain(
        self,
        language: str,
        fallbacks: list[str],
    ) -> None:
        """
        Set fallback chain for a language.

        Args:
            language: Primary language
            fallbacks: List of fallback languages in order
        """
        self._fallback_chain[language] = fallbacks

    def get_fallback_chain(self, language: str) -> list[str]:
        """Get fallback chain for a language."""
        return self._fallback_chain.get(language, [DEFAULT_LANGUAGE])

    # =========================================================================
    # Bank Management
    # =========================================================================

    def register_bank(self, bank: AudioBank) -> None:
        """Register an audio bank."""
        with self._lock:
            self._banks[bank.bank_id] = bank

    def unregister_bank(self, bank_id: str) -> bool:
        """Unregister an audio bank."""
        with self._lock:
            if bank_id in self._banks:
                if bank_id in self._loaded_banks:
                    self.unload_bank(bank_id)
                del self._banks[bank_id]
                return True
            return False

    def get_bank(self, bank_id: str) -> Optional[AudioBank]:
        """Get a bank by ID."""
        return self._banks.get(bank_id)

    def get_banks_for_language(self, language: str) -> list[AudioBank]:
        """Get all banks for a specific language."""
        return [b for b in self._banks.values() if b.language == language]

    def load_bank(self, bank_id: str) -> bool:
        """
        Load an audio bank.

        Args:
            bank_id: ID of bank to load

        Returns:
            True if bank was loaded
        """
        with self._lock:
            bank = self._banks.get(bank_id)
            if not bank:
                return False

            if bank_id in self._loaded_banks:
                return True  # Already loaded

            # Simulate loading
            bank.is_loaded = True
            self._loaded_banks.add(bank_id)

            if self._on_bank_loaded:
                self._on_bank_loaded(bank)

            return True

    def unload_bank(self, bank_id: str) -> bool:
        """
        Unload an audio bank.

        Args:
            bank_id: ID of bank to unload

        Returns:
            True if bank was unloaded
        """
        with self._lock:
            if bank_id not in self._loaded_banks:
                return False

            bank = self._banks.get(bank_id)
            if bank:
                bank.is_loaded = False

            self._loaded_banks.remove(bank_id)

            if self._on_bank_unloaded:
                self._on_bank_unloaded(bank_id)

            return True

    def load_language_banks(self, language: str) -> int:
        """
        Load all banks for a language.

        Returns:
            Number of banks loaded
        """
        count = 0
        for bank in self.get_banks_for_language(language):
            if self.load_bank(bank.bank_id):
                count += 1
        return count

    def unload_language_banks(self, language: str) -> int:
        """
        Unload all banks for a language.

        Returns:
            Number of banks unloaded
        """
        count = 0
        for bank in self.get_banks_for_language(language):
            if self.unload_bank(bank.bank_id):
                count += 1
        return count

    def switch_language_banks(
        self,
        from_language: str,
        to_language: str,
    ) -> tuple[int, int]:
        """
        Switch banks from one language to another.

        Returns:
            Tuple of (unloaded_count, loaded_count)
        """
        unloaded = self.unload_language_banks(from_language)
        loaded = self.load_language_banks(to_language)
        return (unloaded, loaded)

    @property
    def loaded_bank_ids(self) -> set[str]:
        """Get IDs of loaded banks."""
        return set(self._loaded_banks)

    # =========================================================================
    # Asset Management
    # =========================================================================

    def register_asset(self, asset: LocalizedAsset) -> None:
        """Register a localized asset."""
        with self._lock:
            self._assets[asset.asset_id] = asset

    def get_asset(self, asset_id: str) -> Optional[LocalizedAsset]:
        """Get a localized asset by ID."""
        return self._assets.get(asset_id)

    def get_localized_path(
        self,
        asset_id: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Get localized asset path.

        Args:
            asset_id: Asset identifier
            language: Language (uses current if not specified)

        Returns:
            Localized asset path
        """
        language = language or self._current_language
        asset = self._assets.get(asset_id)

        if not asset:
            return ""

        # Try exact language
        if asset.has_language(language):
            return asset.get_path(language)

        # Try fallback chain
        for fallback in self.get_fallback_chain(language):
            if asset.has_language(fallback):
                return asset.get_path(fallback)

        # Use default
        return asset.get_path(DEFAULT_LANGUAGE)

    def get_localized_subtitle(
        self,
        asset_id: str,
        language: Optional[str] = None,
    ) -> str:
        """Get localized subtitle text."""
        language = language or self._current_language
        asset = self._assets.get(asset_id)

        if not asset:
            return ""

        return asset.get_subtitle(language)

    def get_localized_duration(
        self,
        asset_id: str,
        language: Optional[str] = None,
    ) -> float:
        """Get localized audio duration."""
        language = language or self._current_language
        asset = self._assets.get(asset_id)

        if not asset:
            return 0.0

        return asset.get_duration(language)

    # =========================================================================
    # Line Localization
    # =========================================================================

    def localize_line(
        self,
        line: VOLine,
        language: Optional[str] = None,
    ) -> VOLine:
        """
        Create a localized copy of a VO line.

        Args:
            line: Original VO line
            language: Target language

        Returns:
            Localized VO line
        """
        language = language or self._current_language

        # Create a copy
        localized = line.clone()
        localized.language = language

        # Get localized asset
        asset = self._assets.get(line.audio_asset)
        if asset:
            localized.audio_asset = asset.get_path(language)
            localized.duration_ms = asset.get_duration(language)
            localized.text = asset.get_subtitle(language)

            # Update lip sync
            lip_sync = asset.get_lip_sync(language)
            if lip_sync:
                localized.lip_sync = lip_sync

            # Update subtitle data
            if localized.subtitle:
                localized.subtitle.text = asset.get_subtitle(language)
                localized.subtitle.end_time_ms = (
                    localized.subtitle.start_time_ms + localized.duration_ms
                )

        return localized

    def localize_lines(
        self,
        lines: list[VOLine],
        language: Optional[str] = None,
    ) -> list[VOLine]:
        """Localize multiple lines."""
        return [self.localize_line(line, language) for line in lines]

    # =========================================================================
    # Statistics
    # =========================================================================

    @property
    def stats(self) -> dict[str, Any]:
        """Get localization statistics."""
        with self._lock:
            return {
                "current_language": self._current_language,
                "supported_languages": list(self._supported_languages),
                "total_banks": len(self._banks),
                "loaded_banks": len(self._loaded_banks),
                "total_assets": len(self._assets),
                "banks_by_language": {
                    lang: len(self.get_banks_for_language(lang))
                    for lang in self._supported_languages
                },
            }


# =============================================================================
# Helper Functions
# =============================================================================


def create_localized_asset(
    asset_id: str,
    variants: dict[str, dict[str, Any]],
) -> LocalizedAsset:
    """
    Create a localized asset from variant data.

    Args:
        asset_id: Unique asset identifier
        variants: Dict of language -> {"path", "duration_ms", "subtitle"}

    Returns:
        LocalizedAsset instance
    """
    asset = LocalizedAsset(asset_id=asset_id)

    for language, data in variants.items():
        asset.add_variant(
            language=language,
            path=data.get("path", ""),
            duration_ms=data.get("duration_ms", 0.0),
            subtitle=data.get("subtitle", ""),
        )

    return asset


def create_audio_bank(
    bank_id: str,
    language: str,
    assets_data: list[dict[str, Any]],
    category: str = "",
    base_path: str = "",
) -> AudioBank:
    """
    Create an audio bank from asset data.

    Args:
        bank_id: Unique bank identifier
        language: Bank language
        assets_data: List of asset definitions
        category: Bank category
        base_path: Base path for assets

    Returns:
        AudioBank instance
    """
    bank = AudioBank(
        bank_id=bank_id,
        language=language,
        category=category,
        base_path=base_path,
    )

    for data in assets_data:
        asset = LocalizedAsset(
            asset_id=data.get("id", ""),
            base_path=data.get("path", ""),
        )

        # Add language variant
        asset.add_variant(
            language=language,
            path=data.get("path", ""),
            duration_ms=data.get("duration_ms", 0.0),
            subtitle=data.get("subtitle", ""),
        )

        bank.add_asset(asset)

    return bank
