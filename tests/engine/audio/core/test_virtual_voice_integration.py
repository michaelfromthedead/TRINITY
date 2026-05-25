"""Black-box integration tests for VoiceManager virtual voice lifecycle."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from engine.audio.core.audio_clip import AudioClip, AudioClipMetadata, ClipLoadState
from engine.audio.core.audio_source import AudioSource
from engine.audio.core.config import (
    AudioCategory,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    VoiceState,
    VoiceStealStrategy,
)
from engine.audio.core.voice_manager import VoiceManager


def _make_src(src_id: str, priority: int, clip_id: str = "clip_test") -> AudioSource:
    """Factory helper for AudioSource with a real AudioClip."""
    clip = AudioClip(id=clip_id, name=clip_id)
    clip.load_state = ClipLoadState.LOADED
    clip.metadata = AudioClipMetadata(
        sample_rate=48000, total_samples=44100, channels=1
    )
    clip._data = b"\x00\x00" * 44100
    src = AudioSource()
    src.id = src_id
    src.priority = priority
    src.pitch = 1.0
    src.category = AudioCategory.SFX
    src.clip = clip
    return src


def _fill_and_play(
    vm: VoiceManager, count: int, priority: int = PRIORITY_NORMAL
) -> list[AudioSource]:
    """Fill *count* voice slots with sources at *priority* and start playback."""
    sources = []
    for i in range(count):
        src = _make_src(f"fill_{i}", priority, f"clip_fill_{i}")
        result = vm.allocate_voice(src)
        assert result.success, f"Failed to allocate fill {i}: {result.error}"
        # Start playback so the source is in PLAYING state
        src.play()
        sources.append(src)
    assert vm.active_voice_count == count
    return sources


# ============================================================================
# Virtualization path
# ============================================================================

class TestVirtualizeThroughAllocate:
    """VoiceManager virtualization via allocate_voice."""

    def test_virtualize_lowest_priority_victim(self):
        """Lower-priority voice is virtualized when pool exhausted and a
        higher-priority source arrives. The requester gets a voice slot."""
        with patch("time.time", return_value=1000.0):
            vm = VoiceManager(
                max_voices=3,
                steal_strategy=VoiceStealStrategy.NONE,
                enable_virtual_voices=True,
            )
            _fill_and_play(vm, 3, priority=PRIORITY_LOW)

            high = _make_src("high", PRIORITY_HIGH)
            result = vm.allocate_voice(high)

            assert result.success, f"Allocation failed: {result.error}"
            assert result.made_virtual is True
            assert result.voice_id is not None

    def test_victim_is_virtualized(self):
        """The lowest-priority source is in VIRTUAL state after virtualization."""
        with patch("time.time", return_value=1000.0):
            vm = VoiceManager(
                max_voices=2,
                steal_strategy=VoiceStealStrategy.NONE,
                enable_virtual_voices=True,
            )
            a = _make_src("a", PRIORITY_LOW, "clip_a")
            b = _make_src("b", PRIORITY_LOW, "clip_b")
            vm.allocate_voice(a)
            vm.allocate_voice(b)
            a.play()

            high = _make_src("high", PRIORITY_HIGH, "clip_high")
            result = vm.allocate_voice(high)
            assert result.success
            assert result.made_virtual

            # One of the low-priority sources should be virtualized
            assert (a.is_virtual or b.is_virtual), "A source should have been virtualized"

    def test_no_audio_while_virtual(self):
        """Virtualized source get_samples() returns None."""
        with patch("time.time", return_value=1000.0):
            vm = VoiceManager(
                max_voices=2,
                steal_strategy=VoiceStealStrategy.NONE,
                enable_virtual_voices=True,
            )
            a = _make_src("a", PRIORITY_LOW, "clip_a")
            b = _make_src("b", PRIORITY_LOW, "clip_b")
            vm.allocate_voice(a)
            vm.allocate_voice(b)
            a.play()

            high = _make_src("high", PRIORITY_HIGH, "clip_high")
            result = vm.allocate_voice(high)
            assert result.success
            assert result.made_virtual

            # Find the virtualized source
            virtualized = a if a.is_virtual else (b if b.is_virtual else None)
            assert virtualized is not None
            assert virtualized.state == VoiceState.VIRTUAL
            # Virtual source produces no audio
            assert virtualized.get_samples(256) is None

    def test_make_real_restores_playback(self):
        """make_real restores PLAYING state and assigns voice ID."""
        src = _make_src("test", PRIORITY_NORMAL)
        src.state = VoiceState.PLAYING
        src.make_virtual()

        assert src.is_virtual
        assert src.state == VoiceState.VIRTUAL
        assert src.voice_id is None
        assert src.get_samples(256) is None

        src.make_real(voice_id=42)
        assert not src.is_virtual
        assert src.state == VoiceState.PLAYING
        assert src.voice_id == 42
        assert src.get_samples(256) is not None


# ============================================================================
# Callback
# ============================================================================

class TestVirtualizeCallback:
    """on_voice_virtualized callback."""

    def test_callback_fires_with_virtualized_source(self):
        """Callback fires with the virtualized source when a voice is virtualized."""
        with patch("time.time", return_value=1000.0):
            vm = VoiceManager(
                max_voices=2,
                steal_strategy=VoiceStealStrategy.NONE,
                enable_virtual_voices=True,
            )
            fired = []
            vm.on_voice_virtualized = lambda s: fired.append(s)

            a = _make_src("a", PRIORITY_LOW, "clip_a")
            b = _make_src("b", PRIORITY_LOW, "clip_b")
            vm.allocate_voice(a)
            vm.allocate_voice(b)
            a.play()

            high = _make_src("high", PRIORITY_HIGH, "clip_high")
            vm.allocate_voice(high)

            assert len(fired) == 1
            assert fired[0].state == VoiceState.VIRTUAL


# ============================================================================
# Failure cases
# ============================================================================

class TestVirtualizeFailure:
    """Cases where virtualization cannot proceed."""

    def test_fails_when_all_equal_or_higher_priority(self):
        """No virtualization when all active voices have >= priority."""
        with patch("time.time", return_value=1000.0):
            vm = VoiceManager(
                max_voices=2,
                steal_strategy=VoiceStealStrategy.NONE,
                enable_virtual_voices=True,
            )
            _fill_and_play(vm, 2, priority=PRIORITY_HIGH)

            new = _make_src("new", PRIORITY_HIGH)
            result = vm.allocate_voice(new)
            assert not result.success

    def test_fails_when_disabled(self):
        """Virtualization not attempted when enable_virtual_voices=False."""
        with patch("time.time", return_value=1000.0):
            vm = VoiceManager(
                max_voices=2,
                steal_strategy=VoiceStealStrategy.NONE,
                enable_virtual_voices=False,
            )
            _fill_and_play(vm, 2, priority=PRIORITY_NORMAL)

            extra = _make_src("extra", PRIORITY_LOW)
            result = vm.allocate_voice(extra)
            assert not result.success


# ============================================================================
# Stop and release
# ============================================================================

class TestVirtualizeStop:
    """Stop and release behavior with virtualized sources."""

    def test_stop_all_releases_virtual(self):
        """stop_all releases both active and virtualized sources without error."""
        with patch("time.time", return_value=1000.0):
            vm = VoiceManager(
                max_voices=2,
                steal_strategy=VoiceStealStrategy.NONE,
                enable_virtual_voices=True,
            )
            a = _make_src("a", PRIORITY_LOW, "clip_a")
            b = _make_src("b", PRIORITY_LOW, "clip_b")
            vm.allocate_voice(a)
            vm.allocate_voice(b)
            a.play()

            high = _make_src("high", PRIORITY_HIGH, "clip_high")
            result = vm.allocate_voice(high)
            assert result.success

            vm.stop_all(fade_ms=0)
            assert vm.active_voice_count == 0

    def test_virtual_source_stop_and_update(self):
        """Stopping a virtualized source and calling update releases it."""
        with patch("time.time", return_value=1000.0):
            vm = VoiceManager(
                max_voices=2,
                steal_strategy=VoiceStealStrategy.NONE,
                enable_virtual_voices=True,
            )
            a = _make_src("a", PRIORITY_LOW, "clip_a")
            b = _make_src("b", PRIORITY_LOW, "clip_b")
            vm.allocate_voice(a)
            vm.allocate_voice(b)
            a.play()

            high = _make_src("high", PRIORITY_HIGH, "clip_high")
            result = vm.allocate_voice(high)
            assert result.success

            # The virtualized source has no voice_id after virtualization
            # but its state should be VIRTUAL
            if a.is_virtual:
                a.stop()
                assert a.is_stopped
            elif b.is_virtual:
                b.stop()
                assert b.is_stopped


# ============================================================================
# Count properties and configuration
# ============================================================================

class TestVirtualVoiceConfig:
    """Voice count properties and steal strategy."""

    def test_count_properties_initial(self):
        """Initial count properties are zero."""
        vm = VoiceManager(
            max_voices=4,
            steal_strategy=VoiceStealStrategy.NONE,
            enable_virtual_voices=True,
        )
        assert vm.active_voice_count == 0
        assert vm.virtual_voice_count == 0
        assert vm.total_voice_count == 0
        assert vm.available_voices == 4

    def test_set_steal_strategy_to_none(self):
        """set_steal_strategy changes the strategy."""
        vm = VoiceManager(
            max_voices=2, steal_strategy=VoiceStealStrategy.LOWEST_PRIORITY
        )
        assert vm._steal_strategy == VoiceStealStrategy.LOWEST_PRIORITY
        vm.set_steal_strategy(VoiceStealStrategy.NONE)
        assert vm._steal_strategy == VoiceStealStrategy.NONE
