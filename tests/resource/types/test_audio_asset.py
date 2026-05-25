"""Tests for AudioAsset."""
import pytest

from engine.resource.types.audio_asset import AudioAsset, AudioFormat


def _audio(**kw):
    defaults = dict(
        asset_id=50, name="explosion", path="/a.wav", size_bytes=44100,
        sample_rate=44100, channels=2, bit_depth=16,
        duration_seconds=2.0, fmt=AudioFormat.PCM16,
    )
    defaults.update(kw)
    return AudioAsset(**defaults)


class TestAudioAsset:
    def test_creation(self):
        a = _audio()
        assert a.sample_rate == 44100
        assert a.channels == 2
        assert a.bit_depth == 16
        assert a.format is AudioFormat.PCM16

    def test_duration(self):
        a = _audio(duration_seconds=3.5)
        assert a.duration_seconds == pytest.approx(3.5)

    def test_load_unload(self):
        a = _audio()
        assert not a.is_loaded()
        payload = b"\x00" * 100
        a.load(payload)
        assert a.is_loaded()
        assert a.audio_data == payload
        assert a.memory_footprint == 100
        a.unload()
        assert not a.is_loaded()
        assert a.memory_footprint == 0

    def test_all_formats(self):
        for name in ("PCM16", "PCM24", "FLOAT32", "VORBIS", "OPUS"):
            assert hasattr(AudioFormat, name), f"AudioFormat.{name} missing"

    def test_mono_audio(self):
        a = _audio(channels=1)
        assert a.channels == 1
        payload = b"\x00" * 50
        a.load(payload)
        assert a.memory_footprint == 50

    def test_float32_format(self):
        a = _audio(fmt=AudioFormat.FLOAT32, bit_depth=32)
        assert a.format is AudioFormat.FLOAT32
        assert a.bit_depth == 32
        payload = b"\x00" * 200
        a.load(payload)
        assert a.memory_footprint == 200
