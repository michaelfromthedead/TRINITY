"""Tests for animation montage editor with sections, notifies, and blending."""

import pytest

from engine.tooling.animation_tools.montage_editor import (
    AnimMontage,
    AnimSlot,
    MontageBlendSettings,
    MontageEditor,
    MontagePreview,
    MontageSection,
    SectionLink,
    SectionLoopConfig,
    SlotGroup,
)


# =============================================================================
# BLEND SETTINGS TESTS
# =============================================================================


class TestMontageBlendSettings:
    def test_basic_settings(self):
        settings = MontageBlendSettings()
        assert settings.blend_in == 0.25
        assert settings.blend_out == 0.25
        assert settings.blend_mode == "standard"

    def test_invalid_blend_in_raises(self):
        with pytest.raises(ValueError, match="blend_in must be >= 0"):
            MontageBlendSettings(blend_in=-1.0)

    def test_copy(self):
        settings = MontageBlendSettings(blend_in=0.5, blend_out=0.3)
        copy = settings.copy()
        assert copy.blend_in == settings.blend_in
        assert copy is not settings


# =============================================================================
# SECTION LOOP CONFIG TESTS
# =============================================================================


class TestSectionLoopConfig:
    def test_basic_config(self):
        config = SectionLoopConfig()
        assert not config.enabled

    def test_should_loop_disabled(self):
        config = SectionLoopConfig(enabled=False)
        assert not config.should_loop(0)

    def test_should_loop_infinite(self):
        config = SectionLoopConfig(enabled=True, loop_count=-1)
        assert config.should_loop(0)
        assert config.should_loop(100)

    def test_should_loop_limited(self):
        config = SectionLoopConfig(enabled=True, loop_count=3)
        assert config.should_loop(0)
        assert config.should_loop(2)
        assert not config.should_loop(3)


# =============================================================================
# SECTION LINK TESTS
# =============================================================================


class TestSectionLink:
    def test_basic_link(self):
        link = SectionLink(target_section="section_b")
        assert link.target_section == "section_b"
        assert not link.is_branch

    def test_branch_condition(self):
        link = SectionLink(
            target_section="combat",
            is_branch=True,
            condition_param="in_combat",
            condition_value=True,
        )
        assert link.evaluate_condition({"in_combat": True})
        assert not link.evaluate_condition({"in_combat": False})

    def test_non_branch_always_true(self):
        link = SectionLink(target_section="next", is_branch=False)
        assert link.evaluate_condition({})


# =============================================================================
# MONTAGE SECTION TESTS
# =============================================================================


class TestMontageSection:
    def test_basic_section(self):
        section = MontageSection(name="intro", start_time=0.0, end_time=2.0)
        assert section.name == "intro"
        assert section.duration == 2.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            MontageSection(name="", start_time=0.0, end_time=1.0)

    def test_invalid_times_raises(self):
        with pytest.raises(ValueError):
            MontageSection(name="test", start_time=-1.0, end_time=1.0)
        with pytest.raises(ValueError):
            MontageSection(name="test", start_time=2.0, end_time=1.0)

    def test_contains_time(self):
        section = MontageSection(name="test", start_time=1.0, end_time=3.0)
        assert section.contains_time(1.5)
        assert section.contains_time(1.0)
        assert not section.contains_time(3.0)  # exclusive end
        assert not section.contains_time(0.5)

    def test_get_next_section_default(self):
        section = MontageSection(
            name="test",
            start_time=0.0,
            end_time=1.0,
            next_section="next",
        )
        assert section.get_next_section({}) == "next"

    def test_get_next_section_branch(self):
        section = MontageSection(
            name="test",
            start_time=0.0,
            end_time=1.0,
            next_section="default",
            links=[
                SectionLink(
                    target_section="special",
                    is_branch=True,
                    condition_param="special_mode",
                    condition_value=True,
                )
            ],
        )
        assert section.get_next_section({"special_mode": True}) == "special"
        assert section.get_next_section({"special_mode": False}) == "default"

    def test_copy_section(self):
        section = MontageSection(name="test", start_time=0.0, end_time=1.0)
        copy = section.copy()
        assert copy.name == section.name
        assert copy is not section


# =============================================================================
# ANIM SLOT TESTS
# =============================================================================


class TestAnimSlot:
    def test_basic_slot(self):
        slot = AnimSlot(name="UpperBody")
        assert slot.name == "UpperBody"
        assert slot.blend_weight == 1.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            AnimSlot(name="")

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="blend_weight must be 0-1"):
            AnimSlot(name="test", blend_weight=1.5)

    def test_affects_bone_no_filter(self):
        slot = AnimSlot(name="FullBody")
        assert slot.affects_bone("any_bone")

    def test_affects_bone_with_filter(self):
        slot = AnimSlot(name="UpperBody", bone_filter=["spine", "arm"])
        assert slot.affects_bone("spine")
        assert slot.affects_bone("arm")
        assert not slot.affects_bone("leg")


class TestSlotGroup:
    def test_basic_group(self):
        group = SlotGroup(name="DefaultGroup")
        assert group.name == "DefaultGroup"

    def test_add_slot(self):
        group = SlotGroup(name="Group")
        assert group.add_slot(AnimSlot(name="Slot1"))
        assert group.add_slot(AnimSlot(name="Slot2"))
        assert len(group.slots) == 2

    def test_add_duplicate_rejected(self):
        group = SlotGroup(name="Group")
        group.add_slot(AnimSlot(name="Slot1"))
        assert not group.add_slot(AnimSlot(name="Slot1"))

    def test_remove_slot(self):
        group = SlotGroup(name="Group")
        group.add_slot(AnimSlot(name="Slot1"))
        assert group.remove_slot("Slot1")
        assert len(group.slots) == 0

    def test_get_slot(self):
        group = SlotGroup(name="Group")
        slot = AnimSlot(name="Slot1")
        group.add_slot(slot)
        assert group.get_slot("Slot1") is slot


# =============================================================================
# ANIM MONTAGE TESTS
# =============================================================================


class TestAnimMontage:
    def test_basic_montage(self):
        montage = AnimMontage(name="Attack")
        assert montage.name == "Attack"
        assert montage.section_count == 0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            AnimMontage(name="")

    def test_add_section(self):
        montage = AnimMontage(name="Test")
        section = MontageSection(name="intro", start_time=0.0, end_time=1.0)
        assert montage.add_section(section)
        assert montage.section_count == 1

    def test_duplicate_section_rejected(self):
        montage = AnimMontage(name="Test")
        montage.add_section(MontageSection(name="intro", start_time=0.0, end_time=1.0))
        assert not montage.add_section(MontageSection(name="intro", start_time=1.0, end_time=2.0))

    def test_remove_section(self):
        montage = AnimMontage(name="Test")
        montage.add_section(MontageSection(name="intro", start_time=0.0, end_time=1.0))
        assert montage.remove_section("intro")
        assert montage.section_count == 0

    def test_get_section(self):
        montage = AnimMontage(name="Test")
        section = MontageSection(name="intro", start_time=0.0, end_time=1.0)
        montage.add_section(section)
        assert montage.get_section("intro") is section

    def test_get_section_at_time(self):
        montage = AnimMontage(name="Test")
        montage.add_section(MontageSection(name="intro", start_time=0.0, end_time=1.0))
        montage.add_section(MontageSection(name="main", start_time=1.0, end_time=3.0))
        assert montage.get_section_at_time(0.5).name == "intro"
        assert montage.get_section_at_time(2.0).name == "main"

    def test_rename_section(self):
        montage = AnimMontage(name="Test")
        montage.add_section(MontageSection(name="old", start_time=0.0, end_time=1.0))
        assert montage.rename_section("old", "new")
        assert montage.get_section("new") is not None
        assert montage.get_section("old") is None

    def test_section_link(self):
        montage = AnimMontage(name="Test")
        montage.add_section(MontageSection(name="a", start_time=0.0, end_time=1.0))
        montage.add_section(MontageSection(name="b", start_time=1.0, end_time=2.0))
        assert montage.set_section_link("a", "b")
        assert montage.get_section("a").next_section == "b"

    def test_section_branch(self):
        montage = AnimMontage(name="Test")
        montage.add_section(MontageSection(name="a", start_time=0.0, end_time=1.0))
        montage.add_section(MontageSection(name="b", start_time=1.0, end_time=2.0))
        assert montage.add_section_branch("a", "b", "condition", True)
        section = montage.get_section("a")
        assert len(section.links) == 1

    def test_section_loop(self):
        montage = AnimMontage(name="Test")
        montage.add_section(MontageSection(name="loop", start_time=0.0, end_time=1.0))
        assert montage.set_section_loop("loop", enabled=True, loop_count=3)
        section = montage.get_section("loop")
        assert section.loop_config.enabled
        assert section.loop_config.loop_count == 3

    def test_add_slot(self):
        montage = AnimMontage(name="Test")
        slot = AnimSlot(name="UpperBody")
        assert montage.add_slot(slot)
        assert len(montage.slots) == 1

    def test_playback_play_stop(self):
        montage = AnimMontage(name="Test", duration=5.0)
        montage.add_section(MontageSection(name="intro", start_time=0.0, end_time=2.0))
        montage.play()
        assert montage.is_playing
        montage.stop()
        assert not montage.is_playing
        assert montage.current_time == 0.0

    def test_playback_update(self):
        montage = AnimMontage(name="Test", duration=5.0)
        montage.add_section(MontageSection(name="intro", start_time=0.0, end_time=2.0))
        montage.play()
        montage.update(1.0)
        assert montage.current_time == 1.0

    def test_jump_to_section(self):
        montage = AnimMontage(name="Test")
        montage.add_section(MontageSection(name="a", start_time=0.0, end_time=1.0))
        montage.add_section(MontageSection(name="b", start_time=1.0, end_time=2.0))
        assert montage.jump_to_section("b")
        assert montage.current_section == "b"
        assert montage.current_time == 1.0

    def test_get_section_flow(self):
        montage = AnimMontage(name="Test")
        montage.add_section(MontageSection(name="a", start_time=0.0, end_time=1.0, next_section="b"))
        montage.add_section(MontageSection(name="b", start_time=1.0, end_time=2.0))
        flow = montage.get_section_flow()
        assert len(flow) == 2
        assert flow[0] == ("a", ["b"])


# =============================================================================
# MONTAGE PREVIEW TESTS
# =============================================================================


class TestMontagePreview:
    def test_basic_preview(self):
        preview = MontagePreview()
        assert preview.show_sections

    def test_get_section_color(self):
        preview = MontagePreview()
        color = preview.get_section_color("intro")
        assert len(color) == 3

    def test_set_section_color(self):
        preview = MontagePreview()
        preview.set_section_color("intro", (255, 0, 0))
        assert preview.get_section_color("intro") == (255, 0, 0)


# =============================================================================
# MONTAGE EDITOR TESTS
# =============================================================================


class TestMontageEditor:
    def test_basic_editor(self):
        editor = MontageEditor()
        assert editor.montage is None

    def test_create_new(self):
        editor = MontageEditor()
        montage = editor.create_new("Attack")
        assert montage.name == "Attack"
        assert editor.montage is montage

    def test_load_montage(self):
        editor = MontageEditor()
        montage = AnimMontage(name="Test")
        editor.load(montage)
        assert editor.montage is montage

    def test_clear(self):
        editor = MontageEditor()
        editor.create_new("Test")
        editor.clear()
        assert editor.montage is None

    def test_add_section(self):
        editor = MontageEditor()
        editor.create_new("Test")
        section = editor.add_section("intro", 0.0, 1.0)
        assert section is not None
        assert editor.montage.section_count == 1

    def test_remove_section(self):
        editor = MontageEditor()
        editor.create_new("Test")
        editor.add_section("intro", 0.0, 1.0)
        assert editor.remove_section("intro")
        assert editor.montage.section_count == 0

    def test_update_section_times(self):
        editor = MontageEditor()
        editor.create_new("Test")
        editor.add_section("intro", 0.0, 1.0)
        assert editor.update_section_times("intro", end_time=2.0)
        section = editor.montage.get_section("intro")
        assert section.end_time == 2.0

    def test_link_sections(self):
        editor = MontageEditor()
        editor.create_new("Test")
        editor.add_section("a", 0.0, 1.0)
        editor.add_section("b", 1.0, 2.0)
        assert editor.link_sections("a", "b")

    def test_add_branch(self):
        editor = MontageEditor()
        editor.create_new("Test")
        editor.add_section("a", 0.0, 1.0)
        editor.add_section("b", 1.0, 2.0)
        assert editor.add_branch("a", "b", "condition", True)

    def test_set_loop(self):
        editor = MontageEditor()
        editor.create_new("Test")
        editor.add_section("loop", 0.0, 1.0)
        assert editor.set_loop("loop", enabled=True, loop_count=5)

    def test_add_slot(self):
        editor = MontageEditor()
        editor.create_new("Test")
        assert editor.add_slot("UpperBody")
        assert len(editor.montage.slots) == 1

    def test_set_blend_settings(self):
        editor = MontageEditor()
        editor.create_new("Test")
        editor.set_blend_settings(blend_in=0.5, blend_out=0.3)
        assert editor.montage.blend_settings.blend_in == 0.5
        assert editor.montage.blend_settings.blend_out == 0.3

    def test_validate_empty(self):
        editor = MontageEditor()
        editor.create_new("Test")
        errors = editor.validate()
        assert len(errors) > 0  # Should have "no sections" error

    def test_validate_unreachable(self):
        editor = MontageEditor()
        editor.create_new("Test")
        editor.add_section("a", 0.0, 1.0)
        editor.add_section("b", 1.0, 2.0)
        # b is unreachable since a doesn't link to it
        errors = editor.validate()
        assert any("unreachable" in e for e in errors)

    def test_select_section(self):
        editor = MontageEditor()
        editor.create_new("Test")
        editor.add_section("intro", 0.0, 1.0)
        editor.select_section("intro")
        assert editor.selected_section == "intro"

    def test_on_change_callback(self):
        editor = MontageEditor()
        callback_called = [False]

        def callback():
            callback_called[0] = True

        editor.add_on_change(callback)
        editor.create_new("Test")
        assert callback_called[0]
