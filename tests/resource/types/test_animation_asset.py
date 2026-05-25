"""Tests for AnimationAsset."""
import pytest

from engine.resource.types.animation_asset import (
    AnimationAsset, AnimChannel, Keyframe, InterpolationMode,
)


def _anim(**kw):
    defaults = dict(
        asset_id=40, name="walk", path="/a.anim", size_bytes=512,
        duration_seconds=1.5, frame_count=45,
    )
    defaults.update(kw)
    return AnimationAsset(**defaults)


class TestAnimationAsset:
    def test_creation(self):
        a = _anim()
        assert a.duration_seconds == pytest.approx(1.5)
        assert a.frame_count == 45

    def test_channels(self):
        kf = [Keyframe(0.0, (0.0,)), Keyframe(1.0, (1.0,))]
        ch = AnimChannel(target_path="bone.position", keyframes=kf)
        a = _anim(channels=[ch])
        assert len(a.channels) == 1
        assert a.channels[0].target_path == "bone.position"

    def test_keyframe_interpolation(self):
        kf = Keyframe(0.5, (1.0, 2.0), InterpolationMode.CUBIC)
        assert kf.interpolation is InterpolationMode.CUBIC
        assert kf.time == pytest.approx(0.5)

    def test_keyframe_default_interpolation(self):
        kf = Keyframe(0.0, (0.0,))
        assert kf.interpolation is InterpolationMode.LINEAR

    def test_load_unload(self):
        a = _anim()
        assert not a.is_loaded()
        a.load(b"")
        assert a.is_loaded()
        a.unload()
        assert not a.is_loaded()

    def test_interpolation_modes_count(self):
        for name in ("STEP", "LINEAR", "CUBIC"):
            assert hasattr(InterpolationMode, name), f"InterpolationMode.{name} missing"
