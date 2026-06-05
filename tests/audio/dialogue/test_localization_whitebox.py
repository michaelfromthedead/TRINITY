"""
Whitebox tests for Localization module.

Tests LocalizedAsset, AudioBank, LocalizationManager, and helper functions.
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from engine.audio.dialogue.localization import (
    LocalizedAsset,
    AudioBank,
    LocalizationManager,
    create_localized_asset,
    create_audio_bank,
)
from engine.audio.dialogue.vo_line import VOLine, LipSyncData, SubtitleData
from engine.audio.dialogue.config import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
)


# =============================================================================
# LocalizedAsset Tests
# =============================================================================


class TestLocalizedAssetBasic:
    """Basic tests for LocalizedAsset."""

    def test_initialization_defaults(self):
        """Test LocalizedAsset initializes with defaults."""
        asset = LocalizedAsset(asset_id="test_asset")

        assert asset.asset_id == "test_asset"
        assert asset.base_path == ""
        assert asset.variants == {}
        assert asset.durations == {}
        assert asset.subtitles == {}
        assert asset.lip_sync == {}

    def test_initialization_with_values(self):
        """Test LocalizedAsset with explicit values."""
        asset = LocalizedAsset(
            asset_id="test_asset",
            base_path="/audio/default.wav",
            variants={"en": "/audio/en.wav"},
            durations={"en": 1500.0},
            subtitles={"en": "Hello"},
        )

        assert asset.base_path == "/audio/default.wav"
        assert asset.variants["en"] == "/audio/en.wav"
        assert asset.durations["en"] == 1500.0
        assert asset.subtitles["en"] == "Hello"


class TestLocalizedAssetGetters:
    """Tests for LocalizedAsset getter methods."""

    def test_get_path_exact_language(self):
        """Test get_path returns exact language variant."""
        asset = LocalizedAsset(asset_id="test")
        asset.variants = {"en": "/en.wav", "es": "/es.wav"}

        assert asset.get_path("en") == "/en.wav"
        assert asset.get_path("es") == "/es.wav"

    def test_get_path_fallback_to_default(self):
        """Test get_path falls back to default language."""
        asset = LocalizedAsset(asset_id="test")
        asset.variants = {"en": "/en.wav"}

        # If French not found, fall back to English (default)
        assert asset.get_path("fr") == "/en.wav"

    def test_get_path_fallback_to_base(self):
        """Test get_path falls back to base_path."""
        asset = LocalizedAsset(asset_id="test", base_path="/default.wav")

        assert asset.get_path("fr") == "/default.wav"

    def test_get_duration_exact_language(self):
        """Test get_duration returns exact language duration."""
        asset = LocalizedAsset(asset_id="test")
        asset.durations = {"en": 1000.0, "es": 1200.0}

        assert asset.get_duration("en") == 1000.0
        assert asset.get_duration("es") == 1200.0

    def test_get_duration_fallback(self):
        """Test get_duration falls back to default."""
        asset = LocalizedAsset(asset_id="test")
        asset.durations = {"en": 1000.0}

        assert asset.get_duration("fr") == 1000.0

    def test_get_duration_not_found(self):
        """Test get_duration returns 0.0 when not found."""
        asset = LocalizedAsset(asset_id="test")

        assert asset.get_duration("en") == 0.0

    def test_get_subtitle_exact_language(self):
        """Test get_subtitle returns exact language text."""
        asset = LocalizedAsset(asset_id="test")
        asset.subtitles = {"en": "Hello", "es": "Hola"}

        assert asset.get_subtitle("en") == "Hello"
        assert asset.get_subtitle("es") == "Hola"

    def test_get_subtitle_fallback(self):
        """Test get_subtitle falls back to default."""
        asset = LocalizedAsset(asset_id="test")
        asset.subtitles = {"en": "Hello"}

        assert asset.get_subtitle("fr") == "Hello"

    def test_get_subtitle_not_found(self):
        """Test get_subtitle returns empty string when not found."""
        asset = LocalizedAsset(asset_id="test")

        assert asset.get_subtitle("en") == ""

    def test_get_lip_sync_exact_language(self):
        """Test get_lip_sync returns exact language data."""
        asset = LocalizedAsset(asset_id="test")
        lip_sync_en = LipSyncData(phonemes=[("h", 0.0), ("eh", 0.1)])
        lip_sync_es = LipSyncData(phonemes=[("o", 0.0), ("l", 0.1)])
        asset.lip_sync = {"en": lip_sync_en, "es": lip_sync_es}

        assert asset.get_lip_sync("en") is lip_sync_en
        assert asset.get_lip_sync("es") is lip_sync_es

    def test_get_lip_sync_fallback(self):
        """Test get_lip_sync falls back to default."""
        asset = LocalizedAsset(asset_id="test")
        lip_sync_en = LipSyncData(phonemes=[])
        asset.lip_sync = {"en": lip_sync_en}

        assert asset.get_lip_sync("fr") is lip_sync_en

    def test_get_lip_sync_not_found(self):
        """Test get_lip_sync returns None when not found."""
        asset = LocalizedAsset(asset_id="test")

        assert asset.get_lip_sync("en") is None


class TestLocalizedAssetMethods:
    """Tests for LocalizedAsset methods."""

    def test_has_language_true(self):
        """Test has_language returns True for existing variant."""
        asset = LocalizedAsset(asset_id="test")
        asset.variants = {"en": "/en.wav"}

        assert asset.has_language("en") is True

    def test_has_language_false(self):
        """Test has_language returns False for missing variant."""
        asset = LocalizedAsset(asset_id="test")

        assert asset.has_language("en") is False

    def test_add_variant_basic(self):
        """Test add_variant adds path."""
        asset = LocalizedAsset(asset_id="test")

        asset.add_variant("en", "/en.wav")

        assert asset.variants["en"] == "/en.wav"

    def test_add_variant_with_duration(self):
        """Test add_variant with duration."""
        asset = LocalizedAsset(asset_id="test")

        asset.add_variant("en", "/en.wav", duration_ms=1500.0)

        assert asset.durations["en"] == 1500.0

    def test_add_variant_with_zero_duration(self):
        """Test add_variant with zero duration is not added."""
        asset = LocalizedAsset(asset_id="test")

        asset.add_variant("en", "/en.wav", duration_ms=0.0)

        assert "en" not in asset.durations

    def test_add_variant_with_subtitle(self):
        """Test add_variant with subtitle."""
        asset = LocalizedAsset(asset_id="test")

        asset.add_variant("en", "/en.wav", subtitle="Hello")

        assert asset.subtitles["en"] == "Hello"

    def test_add_variant_with_empty_subtitle(self):
        """Test add_variant with empty subtitle is not added."""
        asset = LocalizedAsset(asset_id="test")

        asset.add_variant("en", "/en.wav", subtitle="")

        assert "en" not in asset.subtitles

    def test_add_variant_with_lip_sync(self):
        """Test add_variant with lip sync data."""
        asset = LocalizedAsset(asset_id="test")
        lip_sync = LipSyncData(phonemes=[("h", 0.0)])

        asset.add_variant("en", "/en.wav", lip_sync=lip_sync)

        assert asset.lip_sync["en"] is lip_sync

    def test_add_variant_full(self):
        """Test add_variant with all fields."""
        asset = LocalizedAsset(asset_id="test")
        lip_sync = LipSyncData(phonemes=[])

        asset.add_variant(
            language="en",
            path="/en.wav",
            duration_ms=1500.0,
            subtitle="Hello",
            lip_sync=lip_sync,
        )

        assert asset.variants["en"] == "/en.wav"
        assert asset.durations["en"] == 1500.0
        assert asset.subtitles["en"] == "Hello"
        assert asset.lip_sync["en"] is lip_sync


# =============================================================================
# AudioBank Tests
# =============================================================================


class TestAudioBankBasic:
    """Basic tests for AudioBank."""

    def test_initialization_defaults(self):
        """Test AudioBank initializes with defaults."""
        bank = AudioBank(bank_id="test_bank", language="en")

        assert bank.bank_id == "test_bank"
        assert bank.language == "en"
        assert bank.category == ""
        assert bank.assets == {}
        assert bank.base_path == ""
        assert bank.is_loaded is False

    def test_initialization_with_values(self):
        """Test AudioBank with explicit values."""
        bank = AudioBank(
            bank_id="test_bank",
            language="es",
            category="dialogue",
            base_path="/audio/es/",
        )

        assert bank.language == "es"
        assert bank.category == "dialogue"
        assert bank.base_path == "/audio/es/"


class TestAudioBankAssetManagement:
    """Tests for AudioBank asset management."""

    def test_add_asset(self):
        """Test add_asset adds to bank."""
        bank = AudioBank(bank_id="test", language="en")
        asset = LocalizedAsset(asset_id="asset_1")

        bank.add_asset(asset)

        assert "asset_1" in bank.assets
        assert bank.assets["asset_1"] is asset

    def test_add_multiple_assets(self):
        """Test adding multiple assets."""
        bank = AudioBank(bank_id="test", language="en")
        asset1 = LocalizedAsset(asset_id="asset_1")
        asset2 = LocalizedAsset(asset_id="asset_2")

        bank.add_asset(asset1)
        bank.add_asset(asset2)

        assert len(bank.assets) == 2

    def test_get_asset_found(self):
        """Test get_asset returns existing asset."""
        bank = AudioBank(bank_id="test", language="en")
        asset = LocalizedAsset(asset_id="asset_1")
        bank.add_asset(asset)

        result = bank.get_asset("asset_1")

        assert result is asset

    def test_get_asset_not_found(self):
        """Test get_asset returns None for missing."""
        bank = AudioBank(bank_id="test", language="en")

        result = bank.get_asset("missing")

        assert result is None

    def test_remove_asset_found(self):
        """Test remove_asset removes existing asset."""
        bank = AudioBank(bank_id="test", language="en")
        asset = LocalizedAsset(asset_id="asset_1")
        bank.add_asset(asset)

        result = bank.remove_asset("asset_1")

        assert result is True
        assert "asset_1" not in bank.assets

    def test_remove_asset_not_found(self):
        """Test remove_asset returns False for missing."""
        bank = AudioBank(bank_id="test", language="en")

        result = bank.remove_asset("missing")

        assert result is False


class TestAudioBankProperties:
    """Tests for AudioBank properties."""

    def test_asset_count(self):
        """Test asset_count property."""
        bank = AudioBank(bank_id="test", language="en")
        bank.add_asset(LocalizedAsset(asset_id="a1"))
        bank.add_asset(LocalizedAsset(asset_id="a2"))
        bank.add_asset(LocalizedAsset(asset_id="a3"))

        assert bank.asset_count == 3

    def test_size_bytes(self):
        """Test size_bytes property."""
        bank = AudioBank(bank_id="test", language="en")
        bank._size_bytes = 1024

        assert bank.size_bytes == 1024

    def test_iter(self):
        """Test __iter__ iterates assets."""
        bank = AudioBank(bank_id="test", language="en")
        assets = [
            LocalizedAsset(asset_id="a1"),
            LocalizedAsset(asset_id="a2"),
        ]
        for asset in assets:
            bank.add_asset(asset)

        iterated = list(bank)

        assert len(iterated) == 2


# =============================================================================
# LocalizationManager Tests
# =============================================================================


class TestLocalizationManagerBasic:
    """Basic tests for LocalizationManager."""

    def test_initialization_defaults(self):
        """Test LocalizationManager initializes with defaults."""
        manager = LocalizationManager()

        assert manager.current_language == DEFAULT_LANGUAGE
        assert manager.supported_languages == SUPPORTED_LANGUAGES

    def test_initialization_custom(self):
        """Test LocalizationManager with custom values."""
        callback = MagicMock()
        manager = LocalizationManager(
            default_language="es",
            supported_languages=("en", "es", "fr"),
            on_language_changed=callback,
        )

        assert manager.current_language == "es"
        assert "es" in manager.supported_languages

    def test_is_language_supported_true(self):
        """Test is_language_supported returns True for supported."""
        manager = LocalizationManager()

        assert manager.is_language_supported(DEFAULT_LANGUAGE) is True

    def test_is_language_supported_false(self):
        """Test is_language_supported returns False for unsupported."""
        manager = LocalizationManager(supported_languages=("en",))

        assert manager.is_language_supported("xyz") is False


class TestLocalizationManagerLanguageSwitching:
    """Tests for LocalizationManager language switching."""

    def test_set_language_success(self):
        """Test set_language changes language."""
        manager = LocalizationManager(
            default_language="en",
            supported_languages=("en", "es", "fr"),
        )

        result = manager.set_language("es")

        assert result is True
        assert manager.current_language == "es"

    def test_set_language_unsupported(self):
        """Test set_language fails for unsupported."""
        manager = LocalizationManager(supported_languages=("en",))

        result = manager.set_language("xyz")

        assert result is False
        assert manager.current_language == "en"

    def test_set_language_same(self):
        """Test set_language returns False for same language."""
        manager = LocalizationManager(default_language="en")

        result = manager.set_language("en")

        assert result is False

    def test_set_language_callback(self):
        """Test set_language triggers callback."""
        callback = MagicMock()
        manager = LocalizationManager(
            default_language="en",
            supported_languages=("en", "es"),
            on_language_changed=callback,
        )

        manager.set_language("es")

        callback.assert_called_once_with("en", "es")

    def test_set_fallback_chain(self):
        """Test set_fallback_chain sets chain."""
        manager = LocalizationManager()

        manager.set_fallback_chain("pt-BR", ["pt", "en"])

        assert manager.get_fallback_chain("pt-BR") == ["pt", "en"]

    def test_get_fallback_chain_default(self):
        """Test get_fallback_chain returns default for unknown."""
        manager = LocalizationManager()

        result = manager.get_fallback_chain("xyz")

        assert result == [DEFAULT_LANGUAGE]


# =============================================================================
# LocalizationManager Bank Tests
# =============================================================================


class TestLocalizationManagerBankManagement:
    """Tests for LocalizationManager bank management."""

    def test_register_bank(self):
        """Test register_bank registers bank."""
        manager = LocalizationManager()
        bank = AudioBank(bank_id="test_bank", language="en")

        manager.register_bank(bank)

        assert manager.get_bank("test_bank") is bank

    def test_unregister_bank_found(self):
        """Test unregister_bank removes bank."""
        manager = LocalizationManager()
        bank = AudioBank(bank_id="test_bank", language="en")
        manager.register_bank(bank)

        result = manager.unregister_bank("test_bank")

        assert result is True
        assert manager.get_bank("test_bank") is None

    def test_unregister_bank_not_found(self):
        """Test unregister_bank returns False for missing."""
        manager = LocalizationManager()

        result = manager.unregister_bank("missing")

        assert result is False

    def test_unregister_bank_unloads_first(self):
        """Test unregister_bank unloads bank first."""
        manager = LocalizationManager()
        bank = AudioBank(bank_id="test_bank", language="en")
        manager.register_bank(bank)
        manager.load_bank("test_bank")

        manager.unregister_bank("test_bank")

        assert "test_bank" not in manager.loaded_bank_ids

    def test_get_banks_for_language(self):
        """Test get_banks_for_language filters by language."""
        manager = LocalizationManager()
        bank_en1 = AudioBank(bank_id="en_1", language="en")
        bank_en2 = AudioBank(bank_id="en_2", language="en")
        bank_es = AudioBank(bank_id="es_1", language="es")
        manager.register_bank(bank_en1)
        manager.register_bank(bank_en2)
        manager.register_bank(bank_es)

        result = manager.get_banks_for_language("en")

        assert len(result) == 2
        assert bank_es not in result


class TestLocalizationManagerBankLoading:
    """Tests for LocalizationManager bank loading."""

    def test_load_bank_success(self):
        """Test load_bank loads bank."""
        callback = MagicMock()
        manager = LocalizationManager(on_bank_loaded=callback)
        bank = AudioBank(bank_id="test_bank", language="en")
        manager.register_bank(bank)

        result = manager.load_bank("test_bank")

        assert result is True
        assert bank.is_loaded is True
        assert "test_bank" in manager.loaded_bank_ids
        callback.assert_called_once_with(bank)

    def test_load_bank_already_loaded(self):
        """Test load_bank returns True if already loaded."""
        manager = LocalizationManager()
        bank = AudioBank(bank_id="test_bank", language="en")
        manager.register_bank(bank)
        manager.load_bank("test_bank")

        result = manager.load_bank("test_bank")

        assert result is True

    def test_load_bank_not_found(self):
        """Test load_bank returns False for missing."""
        manager = LocalizationManager()

        result = manager.load_bank("missing")

        assert result is False

    def test_unload_bank_success(self):
        """Test unload_bank unloads bank."""
        callback = MagicMock()
        manager = LocalizationManager(on_bank_unloaded=callback)
        bank = AudioBank(bank_id="test_bank", language="en")
        manager.register_bank(bank)
        manager.load_bank("test_bank")

        result = manager.unload_bank("test_bank")

        assert result is True
        assert bank.is_loaded is False
        assert "test_bank" not in manager.loaded_bank_ids
        callback.assert_called_once_with("test_bank")

    def test_unload_bank_not_loaded(self):
        """Test unload_bank returns False if not loaded."""
        manager = LocalizationManager()
        bank = AudioBank(bank_id="test_bank", language="en")
        manager.register_bank(bank)

        result = manager.unload_bank("test_bank")

        assert result is False

    def test_load_language_banks(self):
        """Test load_language_banks loads all for language."""
        manager = LocalizationManager()
        for i in range(3):
            bank = AudioBank(bank_id=f"en_{i}", language="en")
            manager.register_bank(bank)

        count = manager.load_language_banks("en")

        assert count == 3
        assert len(manager.loaded_bank_ids) == 3

    def test_unload_language_banks(self):
        """Test unload_language_banks unloads all for language."""
        manager = LocalizationManager()
        for i in range(3):
            bank = AudioBank(bank_id=f"en_{i}", language="en")
            manager.register_bank(bank)
            manager.load_bank(f"en_{i}")

        count = manager.unload_language_banks("en")

        assert count == 3
        assert len(manager.loaded_bank_ids) == 0

    def test_switch_language_banks(self):
        """Test switch_language_banks switches banks."""
        manager = LocalizationManager()
        for i in range(2):
            bank_en = AudioBank(bank_id=f"en_{i}", language="en")
            bank_es = AudioBank(bank_id=f"es_{i}", language="es")
            manager.register_bank(bank_en)
            manager.register_bank(bank_es)
            manager.load_bank(f"en_{i}")

        unloaded, loaded = manager.switch_language_banks("en", "es")

        assert unloaded == 2
        assert loaded == 2


# =============================================================================
# LocalizationManager Asset Tests
# =============================================================================


class TestLocalizationManagerAssetManagement:
    """Tests for LocalizationManager asset management."""

    def test_register_asset(self):
        """Test register_asset registers asset."""
        manager = LocalizationManager()
        asset = LocalizedAsset(asset_id="test_asset")

        manager.register_asset(asset)

        assert manager.get_asset("test_asset") is asset

    def test_get_asset_not_found(self):
        """Test get_asset returns None for missing."""
        manager = LocalizationManager()

        result = manager.get_asset("missing")

        assert result is None

    def test_get_localized_path(self):
        """Test get_localized_path returns correct path."""
        manager = LocalizationManager(default_language="en")
        asset = LocalizedAsset(asset_id="test_asset")
        asset.add_variant("en", "/audio/en/test.wav")
        asset.add_variant("es", "/audio/es/test.wav")
        manager.register_asset(asset)

        assert manager.get_localized_path("test_asset") == "/audio/en/test.wav"
        assert manager.get_localized_path("test_asset", "es") == "/audio/es/test.wav"

    def test_get_localized_path_fallback(self):
        """Test get_localized_path uses fallback chain."""
        manager = LocalizationManager(default_language="en")
        manager.set_fallback_chain("pt-BR", ["pt", "en"])
        asset = LocalizedAsset(asset_id="test_asset")
        asset.add_variant("en", "/audio/en/test.wav")
        manager.register_asset(asset)

        result = manager.get_localized_path("test_asset", "pt-BR")

        assert result == "/audio/en/test.wav"

    def test_get_localized_path_not_found(self):
        """Test get_localized_path returns empty for missing asset."""
        manager = LocalizationManager()

        result = manager.get_localized_path("missing")

        assert result == ""

    def test_get_localized_subtitle(self):
        """Test get_localized_subtitle returns correct text."""
        manager = LocalizationManager(default_language="en")
        asset = LocalizedAsset(asset_id="test_asset")
        asset.subtitles = {"en": "Hello", "es": "Hola"}
        manager.register_asset(asset)

        assert manager.get_localized_subtitle("test_asset") == "Hello"
        assert manager.get_localized_subtitle("test_asset", "es") == "Hola"

    def test_get_localized_duration(self):
        """Test get_localized_duration returns correct duration."""
        manager = LocalizationManager(default_language="en")
        asset = LocalizedAsset(asset_id="test_asset")
        asset.durations = {"en": 1000.0, "es": 1200.0}
        manager.register_asset(asset)

        assert manager.get_localized_duration("test_asset") == 1000.0
        assert manager.get_localized_duration("test_asset", "es") == 1200.0


# =============================================================================
# LocalizationManager Line Localization Tests
# =============================================================================


class TestLocalizationManagerLineLocalization:
    """Tests for LocalizationManager line localization."""

    def test_localize_line_basic(self):
        """Test localize_line creates localized copy."""
        manager = LocalizationManager(
            default_language="en",
            supported_languages=("en", "es"),
        )
        asset = LocalizedAsset(asset_id="greeting.wav")
        asset.add_variant("es", "/audio/es/greeting.wav", 1500.0, "Hola")
        manager.register_asset(asset)

        line = VOLine(audio_asset="greeting.wav", text="Hello", language="en")

        localized = manager.localize_line(line, "es")

        assert localized.language == "es"
        assert localized.audio_asset == "/audio/es/greeting.wav"
        assert localized.text == "Hola"
        assert localized.duration_ms == 1500.0

    def test_localize_line_preserves_original(self):
        """Test localize_line does not modify original."""
        manager = LocalizationManager()
        asset = LocalizedAsset(asset_id="greeting.wav")
        asset.add_variant("es", "/es.wav", 1500.0, "Hola")
        manager.register_asset(asset)

        line = VOLine(audio_asset="greeting.wav", text="Hello", language="en")

        manager.localize_line(line, "es")

        assert line.language == "en"
        assert line.text == "Hello"

    def test_localize_line_with_lip_sync(self):
        """Test localize_line updates lip sync."""
        manager = LocalizationManager()
        asset = LocalizedAsset(asset_id="greeting.wav")
        lip_sync_es = LipSyncData(phonemes=[("o", 0.0), ("l", 0.1)])
        asset.add_variant("es", "/es.wav", lip_sync=lip_sync_es)
        manager.register_asset(asset)

        line = VOLine(audio_asset="greeting.wav")

        localized = manager.localize_line(line, "es")

        assert localized.lip_sync is lip_sync_es

    def test_localize_line_updates_subtitle(self):
        """Test localize_line updates subtitle data."""
        manager = LocalizationManager()
        asset = LocalizedAsset(asset_id="greeting.wav")
        asset.add_variant("es", "/es.wav", 1500.0, "Hola")
        manager.register_asset(asset)

        line = VOLine(
            audio_asset="greeting.wav",
            subtitle=SubtitleData(text="Hello", start_time_ms=0.0, end_time_ms=1000.0),
        )

        localized = manager.localize_line(line, "es")

        assert localized.subtitle.text == "Hola"
        assert localized.subtitle.end_time_ms == 1500.0

    def test_localize_line_no_asset(self):
        """Test localize_line without registered asset."""
        manager = LocalizationManager()
        line = VOLine(audio_asset="missing.wav", text="Hello")

        localized = manager.localize_line(line, "es")

        # Should still clone with new language
        assert localized.language == "es"
        assert localized.audio_asset == "missing.wav"

    def test_localize_lines_multiple(self):
        """Test localize_lines localizes multiple."""
        manager = LocalizationManager()
        asset = LocalizedAsset(asset_id="greeting.wav")
        asset.add_variant("es", "/es.wav")
        manager.register_asset(asset)

        lines = [
            VOLine(audio_asset="greeting.wav"),
            VOLine(audio_asset="farewell.wav"),
        ]

        localized = manager.localize_lines(lines, "es")

        assert len(localized) == 2
        assert all(l.language == "es" for l in localized)


# =============================================================================
# LocalizationManager Statistics Tests
# =============================================================================


class TestLocalizationManagerStats:
    """Tests for LocalizationManager statistics."""

    def test_stats_basic(self):
        """Test stats returns correct statistics."""
        manager = LocalizationManager(
            default_language="en",
            supported_languages=("en", "es"),
        )
        bank_en = AudioBank(bank_id="en_1", language="en")
        bank_es = AudioBank(bank_id="es_1", language="es")
        manager.register_bank(bank_en)
        manager.register_bank(bank_es)
        manager.load_bank("en_1")
        manager.register_asset(LocalizedAsset(asset_id="a1"))

        stats = manager.stats

        assert stats["current_language"] == "en"
        assert stats["total_banks"] == 2
        assert stats["loaded_banks"] == 1
        assert stats["total_assets"] == 1


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestCreateLocalizedAsset:
    """Tests for create_localized_asset helper."""

    def test_basic(self):
        """Test create_localized_asset creates asset."""
        variants = {
            "en": {"path": "/en.wav", "duration_ms": 1000.0, "subtitle": "Hello"},
            "es": {"path": "/es.wav", "duration_ms": 1200.0, "subtitle": "Hola"},
        }

        asset = create_localized_asset("test_asset", variants)

        assert asset.asset_id == "test_asset"
        assert asset.variants["en"] == "/en.wav"
        assert asset.variants["es"] == "/es.wav"
        assert asset.durations["en"] == 1000.0
        assert asset.subtitles["es"] == "Hola"

    def test_partial_data(self):
        """Test create_localized_asset with partial data."""
        variants = {
            "en": {"path": "/en.wav"},
        }

        asset = create_localized_asset("test_asset", variants)

        assert asset.variants["en"] == "/en.wav"
        assert asset.durations.get("en") is None
        assert asset.subtitles.get("en") is None


class TestCreateAudioBank:
    """Tests for create_audio_bank helper."""

    def test_basic(self):
        """Test create_audio_bank creates bank."""
        assets_data = [
            {"id": "a1", "path": "/a1.wav", "duration_ms": 1000.0, "subtitle": "Line 1"},
            {"id": "a2", "path": "/a2.wav"},
        ]

        bank = create_audio_bank(
            bank_id="test_bank",
            language="en",
            assets_data=assets_data,
            category="dialogue",
            base_path="/audio/",
        )

        assert bank.bank_id == "test_bank"
        assert bank.language == "en"
        assert bank.category == "dialogue"
        assert bank.asset_count == 2
        assert "a1" in bank.assets


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestLocalizationManagerThreadSafety:
    """Thread safety tests for LocalizationManager."""

    def test_concurrent_language_switches(self):
        """Test concurrent language switching."""
        manager = LocalizationManager(
            default_language="en",
            supported_languages=("en", "es", "fr", "de"),
        )

        languages = ["en", "es", "fr", "de"]
        results = []

        def switch_languages():
            for _ in range(50):
                lang = languages[int(time.time() * 1000) % len(languages)]
                manager.set_language(lang)
                results.append(manager.current_language)
                time.sleep(0.001)

        threads = [threading.Thread(target=switch_languages) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock
        assert len(results) == 150

    def test_concurrent_bank_loading(self):
        """Test concurrent bank loading/unloading."""
        manager = LocalizationManager()
        for i in range(10):
            bank = AudioBank(bank_id=f"bank_{i}", language="en")
            manager.register_bank(bank)

        def load_unload():
            for i in range(50):
                bank_id = f"bank_{i % 10}"
                if i % 2 == 0:
                    manager.load_bank(bank_id)
                else:
                    manager.unload_bank(bank_id)
                time.sleep(0.001)

        threads = [threading.Thread(target=load_unload) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock
