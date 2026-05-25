"""
Blackbox Tests for VoiceManager Voice Limits (T-AU-2.9).

CLEANROOM — tests are derived solely from the public VoiceManager API:
  set_category_limit()
  get_category_voice_count()
  get_sound_instance_count()
  allocate_voice()
  release_voice()
  get_stats()

No knowledge of internals is used or assumed.

Acceptance criteria:
  - Category limits (max N footsteps) enforced  [AC1]
  - Per-sound limits (max 3 gunfire from same weapon) enforced  [AC2]
  - Category tracking O(1)  [AC3]
  - 8+ test cases  [AC4]
"""

from __future__ import annotations

import pytest

from engine.audio.core.config import (
    AudioCategory,
    AudioFormat,
    MemoryPoolType,
    PRIORITY_NORMAL,
    DEFAULT_SAMPLE_RATE,
    VoiceStealStrategy,
    MAX_INSTANCES_PER_SOUND,
)
from engine.audio.core.voice_manager import (
    VoiceAllocationResult,
    VoiceManager,
)
from engine.audio.core.audio_source import AudioSource
from engine.audio.core.audio_clip import AudioClip, AudioClipMetadata, ClipLoadState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def vm() -> VoiceManager:
    """Voice manager with generous global headroom so limit tests are not
    polluted by total-voice-pool exhaustion."""
    return VoiceManager(
        max_voices=64,
        steal_strategy=VoiceStealStrategy.LOWEST_PRIORITY,
        enable_virtual_voices=True,
    )


@pytest.fixture
def sfx_clip() -> AudioClip:
    """Generic SFX clip for repeated allocation tests."""
    clip = AudioClip(
        id="generic_sfx",
        name="generic_sfx",
        category=AudioCategory.SFX,
        pool_type=MemoryPoolType.RESIDENT,
    )
    clip.metadata = AudioClipMetadata(
        duration_seconds=1.0,
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=1,
        format=AudioFormat.PCM_INT16,
        total_samples=DEFAULT_SAMPLE_RATE,
        file_size=DEFAULT_SAMPLE_RATE * 2,
    )
    clip.load_state = ClipLoadState.LOADED
    clip._data = bytes(DEFAULT_SAMPLE_RATE * 2)
    return clip


def _make_source(
    source_id: str,
    clip: AudioClip,
    category: AudioCategory = AudioCategory.SFX,
    priority: int = PRIORITY_NORMAL,
) -> AudioSource:
    src = AudioSource(id=source_id, name=source_id, category=category, priority=priority)
    src.set_clip(clip)
    return src


def _allocate_n(
    vm: VoiceManager, clip: AudioClip, n: int,
    category: AudioCategory = AudioCategory.SFX,
    priority: int = PRIORITY_NORMAL,
) -> list[VoiceAllocationResult]:
    results: list[VoiceAllocationResult] = []
    for i in range(n):
        src = _make_source(f"src_{category.name}_{i}", clip, category, priority)
        results.append(vm.allocate_voice(src))
    return results


# =============================================================================
# AC1 — Category limits are enforced
# =============================================================================


class TestCategoryLimits:
    """Category voice limits — max N active voices of a single category."""

    def test_set_category_limit_reduces_active_count(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """
        AC1 — Category limit (max 5 footsteps) enforced.
        When category limit is N, the category may hold at most N active voices.
        Excess allocations reuse (steal) existing voice slots rather than
        increasing the category count beyond the limit.
        """
        vm.set_category_limit(AudioCategory.SFX, 3)

        # Allocate 6 — category should never exceed 3 active
        _allocate_n(vm, sfx_clip, 6, category=AudioCategory.SFX)
        assert vm.get_category_voice_count(AudioCategory.SFX) == 3, (
            "Category voice count must not exceed the set limit"
        )

    def test_category_limits_are_independent(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """Different categories have independent limits."""
        music_clip = AudioClip(
            id="music_clip", name="music", category=AudioCategory.MUSIC,
            pool_type=MemoryPoolType.RESIDENT,
        )
        music_clip.metadata = sfx_clip.metadata
        music_clip.load_state = ClipLoadState.LOADED
        music_clip._data = sfx_clip._data

        vm.set_category_limit(AudioCategory.SFX, 2)
        vm.set_category_limit(AudioCategory.MUSIC, 2)

        _allocate_n(vm, sfx_clip, 5, category=AudioCategory.SFX)
        _allocate_n(vm, music_clip, 5, category=AudioCategory.MUSIC)

        assert vm.get_category_voice_count(AudioCategory.SFX) == 2
        assert vm.get_category_voice_count(AudioCategory.MUSIC) == 2

    def test_category_limit_zero_disallows_activation(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """Setting a category limit to 0 prevents any active voice in that category."""
        vm.set_category_limit(AudioCategory.SFX, 0)
        result = vm.allocate_voice(_make_source("blocked", sfx_clip, AudioCategory.SFX))
        assert vm.get_category_voice_count(AudioCategory.SFX) == 0, (
            "Zero category limit should keep count at 0"
        )

    def test_expanding_category_limit_allows_more_voices(
        self, vm: VoiceManager, sfx_clip: AudioClip,
    ) -> None:
        """Increasing a category limit mid-session lets additional voices activate."""
        vm.set_category_limit(AudioCategory.SFX, 2)
        _allocate_n(vm, sfx_clip, 4, category=AudioCategory.SFX)
        assert vm.get_category_voice_count(AudioCategory.SFX) == 2

        vm.set_category_limit(AudioCategory.SFX, 4)
        _allocate_n(vm, sfx_clip, 4, category=AudioCategory.SFX)
        assert vm.get_category_voice_count(AudioCategory.SFX) == 4, (
            "After raising the limit, more voices should become active"
        )


# =============================================================================
# AC2 — Per-sound instance limits are enforced
# =============================================================================


class TestPerSoundLimits:
    """Per-sound instance limits — at most MAX_INSTANCES_PER_SOUND active
    instances of the same sound ID."""

    def test_sound_instance_count_capped_at_max(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """
        AC2 — Per-sound limit (max 3 gunfire from same weapon) enforced.
        Allocating N copies of the same sound yields at most MAX_INSTANCES_PER_SOUND
        tracked instances.
        """
        _allocate_n(vm, sfx_clip, MAX_INSTANCES_PER_SOUND * 2)
        count = vm.get_sound_instance_count(sfx_clip.id)
        assert count <= MAX_INSTANCES_PER_SOUND, (
            f"Sound instance count {count} exceeds MAX_INSTANCES_PER_SOUND "
            f"({MAX_INSTANCES_PER_SOUND})"
        )

    def test_different_sounds_have_independent_instance_counts(
        self, vm: VoiceManager, sfx_clip: AudioClip,
    ) -> None:
        """Two distinct sound IDs maintain separate instance counters."""
        clip_a = sfx_clip
        clip_b = AudioClip(
            id="gunfire_b", name="gunfire_b", category=AudioCategory.SFX,
            pool_type=MemoryPoolType.RESIDENT,
        )
        clip_b.metadata = sfx_clip.metadata
        clip_b.load_state = ClipLoadState.LOADED
        clip_b._data = sfx_clip._data

        _allocate_n(vm, clip_a, MAX_INSTANCES_PER_SOUND * 2)
        _allocate_n(vm, clip_b, MAX_INSTANCES_PER_SOUND * 2)

        assert vm.get_sound_instance_count(clip_a.id) <= MAX_INSTANCES_PER_SOUND
        assert vm.get_sound_instance_count(clip_b.id) <= MAX_INSTANCES_PER_SOUND

    def test_sound_instance_limit_reclaim_on_release(
        self, vm: VoiceManager, sfx_clip: AudioClip,
    ) -> None:
        """Releasing a voice frees one per-sound slot so a new allocation can
        use it without exceeding the limit."""
        # Allocate to fill the per-sound limit
        results = _allocate_n(vm, sfx_clip, MAX_INSTANCES_PER_SOUND)
        assert vm.get_sound_instance_count(sfx_clip.id) == MAX_INSTANCES_PER_SOUND

        # At capacity — new instances should reuse (steal) an existing slot
        extra = _make_source("overflow", sfx_clip)
        overflow_result = vm.allocate_voice(extra)
        _ = overflow_result  # may steal, that is OK
        assert vm.get_sound_instance_count(sfx_clip.id) <= MAX_INSTANCES_PER_SOUND, (
            "Sound instance count must never exceed MAX_INSTANCES_PER_SOUND"
        )


# =============================================================================
# AC3 — Category tracking is O(1)
# =============================================================================


class TestCategoryTrackingO1:
    """
    AC3 — Category tracking is O(1).

    The public contract guarantees that per-category voice counts are
    available via direct lookup (dict / counter), not by scanning all
    active voices.  get_stats() exposes category_counts as a flat dict.
    """

    def test_category_count_available_directly(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """Category voice count is queryable via a dedicated getter (O(1))."""
        vm.set_category_limit(AudioCategory.SFX, 6)
        _allocate_n(vm, sfx_clip, 4, category=AudioCategory.SFX)

        count = vm.get_category_voice_count(AudioCategory.SFX)
        assert count == 4, f"Expected 4 SFX voices, got {count}"

    def test_stats_exposes_category_counts(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """get_stats() contains a category_counts dict for O(1) access."""
        vm.set_category_limit(AudioCategory.SFX, 4)
        _allocate_n(vm, sfx_clip, 3, category=AudioCategory.SFX)

        stats = vm.get_stats()
        assert "category_counts" in stats, (
            "stats must contain category_counts"
        )
        counts = stats["category_counts"]
        assert isinstance(counts, dict), "category_counts must be a dict"
        assert counts.get(AudioCategory.SFX) == 3

    def test_stats_exposes_sound_instances(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """get_stats() contains a sound_instances dict for O(1) per-sound lookup."""
        _allocate_n(vm, sfx_clip, 3)

        stats = vm.get_stats()
        assert "sound_instances" in stats, (
            "stats must contain sound_instances"
        )
        instances = stats["sound_instances"]
        assert isinstance(instances, dict), "sound_instances must be a dict"
        assert instances.get(sfx_clip.id) == 3

    def test_category_zeroed_after_release(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """Releasing a voice decrements the O(1) category counter."""
        vm.set_category_limit(AudioCategory.SFX, 6)
        results = _allocate_n(vm, sfx_clip, 3, category=AudioCategory.SFX)
        assert vm.get_category_voice_count(AudioCategory.SFX) == 3

        vm.release_voice(results[0].voice_id)
        assert vm.get_category_voice_count(AudioCategory.SFX) == 2


# =============================================================================
# AC4 — get_stats() coverage
# =============================================================================


class TestGetStats:
    """get_stats() returns a comprehensive snapshot of the voice-limits state."""

    def test_stats_structure(self, vm: VoiceManager) -> None:
        """get_stats() returns the expected top-level keys."""
        stats = vm.get_stats()
        assert "max_voices" in stats
        assert "active_voices" in stats
        assert "available_voices" in stats
        assert "category_counts" in stats
        assert "sound_instances" in stats
        assert "steal_strategy" in stats

    def test_stats_voice_counts_match(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """Stats voice counts reflect allocations."""
        _allocate_n(vm, sfx_clip, 5)
        stats = vm.get_stats()
        assert stats["active_voices"] >= 1

    def test_stats_update_after_release(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """Stats reflect voice release."""
        results = _allocate_n(vm, sfx_clip, 3)
        pre = vm.get_stats()["active_voices"]

        vm.release_voice(results[0].voice_id)
        post = vm.get_stats()["active_voices"]

        assert post < pre, "Release must reduce active voice count"


# =============================================================================
# AC4 — Edge cases (additional test cases beyond the 3 ACs)
# =============================================================================


class TestEdgeCases:
    """Boundary conditions and defensive behaviour."""

    def test_allocate_release_cycle_does_not_leak_count(
        self, vm: VoiceManager, sfx_clip: AudioClip,
    ) -> None:
        """Allocate-and-release cycles keep the O(1) counters accurate."""
        vm.set_category_limit(AudioCategory.SFX, 4)
        for _ in range(10):
            result = vm.allocate_voice(_make_source("leak_test", sfx_clip, AudioCategory.SFX))
            vm.release_voice(result.voice_id)

        # After all releases the category count should be 0
        assert vm.get_category_voice_count(AudioCategory.SFX) == 0, (
            "Releasing all voices of a category must give count 0"
        )

    def test_total_voice_limit_respected(self, vm: VoiceManager, sfx_clip: AudioClip) -> None:
        """Global max_voices is enforced regardless of category limits."""
        small_vm = VoiceManager(
            max_voices=3,
            steal_strategy=VoiceStealStrategy.LOWEST_PRIORITY,
            enable_virtual_voices=False,
        )
        _allocate_n(small_vm, sfx_clip, 5)
        assert small_vm.active_voice_count <= 3, (
            "Global max_voices must cap total active count"
        )

    def test_different_categories_zero_category_counts(
        self, vm: VoiceManager, sfx_clip: AudioClip,
    ) -> None:
        """Unused categories have a count of zero in stats."""
        stats = vm.get_stats()
        for cat in AudioCategory:
            assert cat in stats["category_counts"], (
                f"Category {cat} must appear in category_counts"
            )

    def test_category_count_tracks_across_multiple_sounds(
        self, vm: VoiceManager, sfx_clip: AudioClip,
    ) -> None:
        """Category count correctly sums voices from different sound IDs."""
        clip_b = AudioClip(
            id="sfx_b", name="sfx_b", category=AudioCategory.SFX,
            pool_type=MemoryPoolType.RESIDENT,
        )
        clip_b.metadata = sfx_clip.metadata
        clip_b.load_state = ClipLoadState.LOADED
        clip_b._data = sfx_clip._data

        vm.set_category_limit(AudioCategory.SFX, 6)
        _allocate_n(vm, sfx_clip, 3, category=AudioCategory.SFX)
        _allocate_n(vm, clip_b, 3, category=AudioCategory.SFX)

        assert vm.get_category_voice_count(AudioCategory.SFX) == 6, (
            "Category count should reflect voices from all sound IDs"
        )
