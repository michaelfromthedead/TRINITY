"""Animation montage creation with sections, slots, and notifies.

Montages are composite animations that combine multiple animation clips
with sections for branching, looping, and blending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.core.math import Transform


# =============================================================================
# BLEND SETTINGS
# =============================================================================


@dataclass
class MontageBlendSettings:
    """Blend settings for montage transitions.

    Attributes:
        blend_in: Blend in duration in seconds
        blend_out: Blend out duration in seconds
        blend_mode: Blend mode (standard, additive, override)
        blend_profile: Blend curve type
    """

    blend_in: float = 0.25
    blend_out: float = 0.25
    blend_mode: str = "standard"  # standard, additive, override
    blend_profile: str = "ease_in_out"  # linear, ease_in, ease_out, ease_in_out

    def __post_init__(self) -> None:
        if self.blend_in < 0:
            raise ValueError(f"blend_in must be >= 0, got {self.blend_in}")
        if self.blend_out < 0:
            raise ValueError(f"blend_out must be >= 0, got {self.blend_out}")

    def copy(self) -> MontageBlendSettings:
        """Create a copy."""
        return MontageBlendSettings(
            blend_in=self.blend_in,
            blend_out=self.blend_out,
            blend_mode=self.blend_mode,
            blend_profile=self.blend_profile,
        )


# =============================================================================
# SECTIONS
# =============================================================================


@dataclass
class SectionLoopConfig:
    """Configuration for section looping.

    Attributes:
        enabled: Whether looping is enabled
        loop_count: Number of loops (-1 for infinite)
        loop_start_offset: Offset from section start for loop point
    """

    enabled: bool = False
    loop_count: int = -1
    loop_start_offset: float = 0.0

    def should_loop(self, current_loop: int) -> bool:
        """Check if should loop given current loop count."""
        if not self.enabled:
            return False
        if self.loop_count < 0:
            return True  # Infinite
        return current_loop < self.loop_count


@dataclass
class SectionLink:
    """Link between two sections.

    Attributes:
        target_section: Name of target section
        blend_time: Blend time to target
        is_branch: Whether this is a conditional branch
        condition_param: Parameter name for branch condition
        condition_value: Value to match for branch
    """

    target_section: str
    blend_time: float = 0.25
    is_branch: bool = False
    condition_param: str = ""
    condition_value: Any = None

    def evaluate_condition(self, params: Dict[str, Any]) -> bool:
        """Evaluate branch condition."""
        if not self.is_branch:
            return True
        if not self.condition_param:
            return True
        return params.get(self.condition_param) == self.condition_value


@dataclass
class MontageSection:
    """A section within a montage.

    Sections define segments of the montage that can be played, looped,
    or branched to.

    Attributes:
        name: Section name
        start_time: Start time in seconds
        end_time: End time in seconds
        loop_config: Loop configuration
        next_section: Default next section (None to stop)
        links: Additional section links for branching
    """

    name: str
    start_time: float
    end_time: float
    loop_config: SectionLoopConfig = field(default_factory=SectionLoopConfig)
    next_section: Optional[str] = None
    links: List[SectionLink] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Section name cannot be empty")
        if self.start_time < 0:
            raise ValueError(f"start_time must be >= 0, got {self.start_time}")
        if self.end_time < self.start_time:
            raise ValueError(f"end_time must be >= start_time, got {self.end_time} < {self.start_time}")

    @property
    def duration(self) -> float:
        """Get section duration."""
        return self.end_time - self.start_time

    def contains_time(self, time: float) -> bool:
        """Check if time is within section."""
        return self.start_time <= time < self.end_time

    def get_next_section(self, params: Dict[str, Any]) -> Optional[str]:
        """Get next section based on params."""
        # Check branch links first
        for link in self.links:
            if link.is_branch and link.evaluate_condition(params):
                return link.target_section

        # Return default next section
        return self.next_section

    def copy(self) -> MontageSection:
        """Create a copy."""
        return MontageSection(
            name=self.name,
            start_time=self.start_time,
            end_time=self.end_time,
            loop_config=SectionLoopConfig(
                enabled=self.loop_config.enabled,
                loop_count=self.loop_config.loop_count,
                loop_start_offset=self.loop_config.loop_start_offset,
            ),
            next_section=self.next_section,
            links=[
                SectionLink(
                    target_section=link.target_section,
                    blend_time=link.blend_time,
                    is_branch=link.is_branch,
                    condition_param=link.condition_param,
                    condition_value=link.condition_value,
                )
                for link in self.links
            ],
        )


# =============================================================================
# SLOTS
# =============================================================================


@dataclass
class AnimSlot:
    """A slot for playing animations.

    Slots allow multiple animations to play on different bone groups
    simultaneously.

    Attributes:
        name: Slot name
        bone_filter: List of bone names to affect (empty = all bones)
        blend_weight: Weight for blending
        priority: Slot priority for conflict resolution
    """

    name: str
    bone_filter: List[str] = field(default_factory=list)
    blend_weight: float = 1.0
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Slot name cannot be empty")
        if self.blend_weight < 0 or self.blend_weight > 1:
            raise ValueError(f"blend_weight must be 0-1, got {self.blend_weight}")

    def affects_bone(self, bone_name: str) -> bool:
        """Check if slot affects a bone."""
        if not self.bone_filter:
            return True
        return bone_name in self.bone_filter


@dataclass
class SlotGroup:
    """A group of related slots.

    Attributes:
        name: Group name
        slots: Slots in this group
    """

    name: str
    slots: List[AnimSlot] = field(default_factory=list)

    def add_slot(self, slot: AnimSlot) -> bool:
        """Add a slot to the group."""
        if any(s.name == slot.name for s in self.slots):
            return False
        self.slots.append(slot)
        return True

    def remove_slot(self, name: str) -> bool:
        """Remove a slot from the group."""
        for i, slot in enumerate(self.slots):
            if slot.name == name:
                self.slots.pop(i)
                return True
        return False

    def get_slot(self, name: str) -> Optional[AnimSlot]:
        """Get slot by name."""
        for slot in self.slots:
            if slot.name == name:
                return slot
        return None


# =============================================================================
# ANIMATION MONTAGE
# =============================================================================


class AnimMontage:
    """An animation montage.

    Montages combine animation clips with sections for complex playback
    control including looping, branching, and blending.

    Attributes:
        name: Montage name
        duration: Total montage duration
        blend_settings: Blend settings
    """

    def __init__(
        self,
        name: str,
        animation_path: Optional[str] = None,
        duration: float = 0.0,
    ) -> None:
        if not name:
            raise ValueError("Montage name cannot be empty")

        self._name = name
        self._animation_path = animation_path
        self._duration = duration
        self._sections: Dict[str, MontageSection] = {}
        self._section_order: List[str] = []  # For display order
        self._slots: Dict[str, AnimSlot] = {}
        self._blend_settings = MontageBlendSettings()
        self._notifies: List[Dict[str, Any]] = []

        # Playback state
        self._current_section: Optional[str] = None
        self._current_time: float = 0.0
        self._current_loop: int = 0
        self._is_playing: bool = False

    @property
    def name(self) -> str:
        """Get montage name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set montage name."""
        if not value:
            raise ValueError("Montage name cannot be empty")
        self._name = value

    @property
    def animation_path(self) -> Optional[str]:
        """Get source animation path."""
        return self._animation_path

    @animation_path.setter
    def animation_path(self, value: Optional[str]) -> None:
        """Set source animation path."""
        self._animation_path = value

    @property
    def duration(self) -> float:
        """Get montage duration."""
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        """Set montage duration."""
        if value < 0:
            raise ValueError(f"Duration must be >= 0, got {value}")
        self._duration = value

    @property
    def blend_settings(self) -> MontageBlendSettings:
        """Get blend settings."""
        return self._blend_settings

    @property
    def sections(self) -> List[MontageSection]:
        """Get all sections in order."""
        return [self._sections[name] for name in self._section_order if name in self._sections]

    @property
    def section_count(self) -> int:
        """Get number of sections."""
        return len(self._sections)

    @property
    def slots(self) -> List[AnimSlot]:
        """Get all slots."""
        return list(self._slots.values())

    @property
    def current_section(self) -> Optional[str]:
        """Get current section name."""
        return self._current_section

    @property
    def current_time(self) -> float:
        """Get current time."""
        return self._current_time

    @property
    def is_playing(self) -> bool:
        """Check if montage is playing."""
        return self._is_playing

    # Section operations

    def add_section(self, section: MontageSection) -> bool:
        """Add a section."""
        if section.name in self._sections:
            return False
        if section.end_time > self._duration:
            self._duration = section.end_time

        self._sections[section.name] = section
        self._section_order.append(section.name)
        return True

    def remove_section(self, name: str) -> bool:
        """Remove a section."""
        if name not in self._sections:
            return False

        # Update links in other sections
        for section in self._sections.values():
            if section.next_section == name:
                section.next_section = None
            section.links = [l for l in section.links if l.target_section != name]

        del self._sections[name]
        self._section_order.remove(name)
        return True

    def get_section(self, name: str) -> Optional[MontageSection]:
        """Get section by name."""
        return self._sections.get(name)

    def get_section_at_time(self, time: float) -> Optional[MontageSection]:
        """Get section at time."""
        for section in self._sections.values():
            if section.contains_time(time):
                return section
        return None

    def rename_section(self, old_name: str, new_name: str) -> bool:
        """Rename a section."""
        if old_name not in self._sections:
            return False
        if new_name in self._sections:
            return False

        section = self._sections.pop(old_name)
        section.name = new_name
        self._sections[new_name] = section

        # Update order
        idx = self._section_order.index(old_name)
        self._section_order[idx] = new_name

        # Update links
        for s in self._sections.values():
            if s.next_section == old_name:
                s.next_section = new_name
            for link in s.links:
                if link.target_section == old_name:
                    link.target_section = new_name

        return True

    def set_section_link(
        self,
        from_section: str,
        to_section: Optional[str],
        blend_time: float = 0.25,
    ) -> bool:
        """Set default next section link."""
        section = self.get_section(from_section)
        if section is None:
            return False
        if to_section and to_section not in self._sections:
            return False

        section.next_section = to_section
        # Update blend time in existing non-branch link if present
        for link in section.links:
            if not link.is_branch and link.target_section == to_section:
                link.blend_time = blend_time
                return True

        return True

    def add_section_branch(
        self,
        from_section: str,
        to_section: str,
        condition_param: str,
        condition_value: Any,
        blend_time: float = 0.25,
    ) -> bool:
        """Add a conditional branch between sections."""
        section = self.get_section(from_section)
        if section is None:
            return False
        if to_section not in self._sections:
            return False

        link = SectionLink(
            target_section=to_section,
            blend_time=blend_time,
            is_branch=True,
            condition_param=condition_param,
            condition_value=condition_value,
        )
        section.links.append(link)
        return True

    def set_section_loop(
        self,
        section_name: str,
        enabled: bool = True,
        loop_count: int = -1,
        loop_start_offset: float = 0.0,
    ) -> bool:
        """Configure section looping."""
        section = self.get_section(section_name)
        if section is None:
            return False

        section.loop_config.enabled = enabled
        section.loop_config.loop_count = loop_count
        section.loop_config.loop_start_offset = loop_start_offset
        return True

    # Slot operations

    def add_slot(self, slot: AnimSlot) -> bool:
        """Add a slot."""
        if slot.name in self._slots:
            return False
        self._slots[slot.name] = slot
        return True

    def remove_slot(self, name: str) -> bool:
        """Remove a slot."""
        if name in self._slots:
            del self._slots[name]
            return True
        return False

    def get_slot(self, name: str) -> Optional[AnimSlot]:
        """Get slot by name."""
        return self._slots.get(name)

    # Playback

    def play(self, start_section: Optional[str] = None) -> None:
        """Start playback."""
        self._is_playing = True
        self._current_loop = 0

        if start_section and start_section in self._sections:
            section = self._sections[start_section]
            self._current_section = start_section
            self._current_time = section.start_time
        elif self._section_order:
            self._current_section = self._section_order[0]
            self._current_time = 0.0
        else:
            self._current_time = 0.0

    def stop(self) -> None:
        """Stop playback."""
        self._is_playing = False
        self._current_time = 0.0
        self._current_section = None
        self._current_loop = 0

    def pause(self) -> None:
        """Pause playback."""
        self._is_playing = False

    def resume(self) -> None:
        """Resume playback."""
        self._is_playing = True

    def jump_to_section(self, section_name: str) -> bool:
        """Jump to a section."""
        section = self.get_section(section_name)
        if section is None:
            return False

        self._current_section = section_name
        self._current_time = section.start_time
        self._current_loop = 0
        return True

    def update(self, dt: float, params: Optional[Dict[str, Any]] = None) -> List[str]:
        """Update montage playback.

        Args:
            dt: Delta time
            params: Parameters for branch conditions

        Returns:
            List of triggered notify names
        """
        if not self._is_playing:
            return []

        params = params or {}
        triggered_notifies: List[str] = []

        old_time = self._current_time
        self._current_time += dt

        # Check notifies
        for notify in self._notifies:
            if old_time <= notify["time"] < self._current_time:
                triggered_notifies.append(notify["name"])

        # Handle section transitions
        if self._current_section:
            section = self._sections.get(self._current_section)
            if section and self._current_time >= section.end_time:
                # Check for loop
                if section.loop_config.should_loop(self._current_loop):
                    self._current_time = section.start_time + section.loop_config.loop_start_offset
                    self._current_loop += 1
                else:
                    # Get next section
                    next_section = section.get_next_section(params)
                    if next_section and next_section in self._sections:
                        self._current_section = next_section
                        next_sect = self._sections[next_section]
                        self._current_time = next_sect.start_time
                        self._current_loop = 0
                    else:
                        # End of montage
                        self._is_playing = False

        # Check duration
        if self._current_time >= self._duration:
            self._is_playing = False
            self._current_time = self._duration

        return triggered_notifies

    def get_section_flow(self) -> List[Tuple[str, List[str]]]:
        """Get section flow graph for visualization."""
        flow = []
        for name in self._section_order:
            section = self._sections.get(name)
            if section:
                targets = []
                if section.next_section:
                    targets.append(section.next_section)
                for link in section.links:
                    if link.target_section not in targets:
                        targets.append(link.target_section)
                flow.append((name, targets))
        return flow


# =============================================================================
# MONTAGE PREVIEW
# =============================================================================


class MontagePreview:
    """Preview settings for montage visualization."""

    def __init__(self) -> None:
        self.show_sections = True
        self.show_notifies = True
        self.show_section_links = True
        self.show_timeline = True
        self.section_colors: Dict[str, Tuple[int, int, int]] = {}
        self.zoom_level = 1.0
        self.scroll_offset = 0.0

    def get_section_color(self, section_name: str) -> Tuple[int, int, int]:
        """Get color for a section."""
        if section_name in self.section_colors:
            return self.section_colors[section_name]
        # Generate default color based on name hash
        h = hash(section_name)
        return (
            100 + (h % 100),
            100 + ((h >> 8) % 100),
            100 + ((h >> 16) % 100),
        )

    def set_section_color(self, section_name: str, color: Tuple[int, int, int]) -> None:
        """Set color for a section."""
        self.section_colors[section_name] = color


# =============================================================================
# MONTAGE EDITOR
# =============================================================================


class MontageEditor:
    """Editor for animation montages.

    Provides functionality for creating and editing animation montages
    with sections, slots, and notifies.
    """

    def __init__(self) -> None:
        self._montage: Optional[AnimMontage] = None
        self._preview = MontagePreview()
        self._selected_section: Optional[str] = None
        self._selected_notify: int = -1
        self._on_change_callbacks: List[Callable[[], None]] = []

    @property
    def montage(self) -> Optional[AnimMontage]:
        """Get current montage."""
        return self._montage

    @property
    def preview(self) -> MontagePreview:
        """Get preview settings."""
        return self._preview

    @property
    def selected_section(self) -> Optional[str]:
        """Get selected section name."""
        return self._selected_section

    def create_new(self, name: str, animation_path: Optional[str] = None) -> AnimMontage:
        """Create a new montage."""
        self._montage = AnimMontage(name, animation_path)
        self._selected_section = None
        self._notify_change()
        return self._montage

    def load(self, montage: AnimMontage) -> None:
        """Load a montage for editing."""
        self._montage = montage
        self._selected_section = None
        self._notify_change()

    def clear(self) -> None:
        """Clear current montage."""
        self._montage = None
        self._selected_section = None
        self._notify_change()

    def select_section(self, name: Optional[str]) -> None:
        """Select a section."""
        if self._montage and (name is None or name in self._montage._sections):
            self._selected_section = name

    def add_section(
        self,
        name: str,
        start_time: float,
        end_time: float,
    ) -> Optional[MontageSection]:
        """Add a new section."""
        if self._montage is None:
            return None

        section = MontageSection(
            name=name,
            start_time=start_time,
            end_time=end_time,
        )

        if self._montage.add_section(section):
            self._notify_change()
            return section
        return None

    def remove_section(self, name: str) -> bool:
        """Remove a section."""
        if self._montage is None:
            return False

        if self._montage.remove_section(name):
            if self._selected_section == name:
                self._selected_section = None
            self._notify_change()
            return True
        return False

    def update_section_times(
        self,
        name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> bool:
        """Update section start/end times."""
        if self._montage is None:
            return False

        section = self._montage.get_section(name)
        if section is None:
            return False

        if start_time is not None:
            section.start_time = start_time
        if end_time is not None:
            section.end_time = end_time

        self._notify_change()
        return True

    def link_sections(
        self,
        from_section: str,
        to_section: Optional[str],
        blend_time: float = 0.25,
    ) -> bool:
        """Link two sections."""
        if self._montage is None:
            return False

        if self._montage.set_section_link(from_section, to_section, blend_time):
            self._notify_change()
            return True
        return False

    def add_branch(
        self,
        from_section: str,
        to_section: str,
        condition_param: str,
        condition_value: Any,
    ) -> bool:
        """Add a branch between sections."""
        if self._montage is None:
            return False

        if self._montage.add_section_branch(
            from_section, to_section, condition_param, condition_value
        ):
            self._notify_change()
            return True
        return False

    def set_loop(
        self,
        section_name: str,
        enabled: bool = True,
        loop_count: int = -1,
    ) -> bool:
        """Set section looping."""
        if self._montage is None:
            return False

        if self._montage.set_section_loop(section_name, enabled, loop_count):
            self._notify_change()
            return True
        return False

    def add_slot(self, name: str, bone_filter: Optional[List[str]] = None) -> bool:
        """Add a slot."""
        if self._montage is None:
            return False

        slot = AnimSlot(name=name, bone_filter=bone_filter or [])
        if self._montage.add_slot(slot):
            self._notify_change()
            return True
        return False

    def remove_slot(self, name: str) -> bool:
        """Remove a slot."""
        if self._montage is None:
            return False

        if self._montage.remove_slot(name):
            self._notify_change()
            return True
        return False

    def set_blend_settings(
        self,
        blend_in: Optional[float] = None,
        blend_out: Optional[float] = None,
        blend_mode: Optional[str] = None,
    ) -> None:
        """Update blend settings."""
        if self._montage is None:
            return

        if blend_in is not None:
            self._montage.blend_settings.blend_in = blend_in
        if blend_out is not None:
            self._montage.blend_settings.blend_out = blend_out
        if blend_mode is not None:
            self._montage.blend_settings.blend_mode = blend_mode

        self._notify_change()

    def validate(self) -> List[str]:
        """Validate montage for errors."""
        if self._montage is None:
            return ["No montage loaded"]

        errors = []

        # Check for sections
        if self._montage.section_count == 0:
            errors.append("Montage has no sections")

        # Check for unreachable sections
        reachable = set()
        if self._montage._section_order:
            to_visit = [self._montage._section_order[0]]
            while to_visit:
                name = to_visit.pop(0)
                if name in reachable:
                    continue
                reachable.add(name)

                section = self._montage.get_section(name)
                if section:
                    if section.next_section:
                        to_visit.append(section.next_section)
                    for link in section.links:
                        to_visit.append(link.target_section)

            unreachable = set(self._montage._sections.keys()) - reachable
            for name in unreachable:
                errors.append(f"Section '{name}' is unreachable")

        # Check for invalid links
        for section in self._montage.sections:
            if section.next_section and section.next_section not in self._montage._sections:
                errors.append(f"Section '{section.name}' links to unknown section '{section.next_section}'")
            for link in section.links:
                if link.target_section not in self._montage._sections:
                    errors.append(f"Section '{section.name}' has invalid branch to '{link.target_section}'")

        return errors

    def add_on_change(self, callback: Callable[[], None]) -> None:
        """Register change callback."""
        self._on_change_callbacks.append(callback)

    def remove_on_change(self, callback: Callable[[], None]) -> None:
        """Remove change callback."""
        if callback in self._on_change_callbacks:
            self._on_change_callbacks.remove(callback)

    def _notify_change(self) -> None:
        """Notify change callbacks."""
        for callback in self._on_change_callbacks:
            callback()


__all__ = [
    "MontageBlendSettings",
    "SectionLoopConfig",
    "SectionLink",
    "MontageSection",
    "AnimSlot",
    "SlotGroup",
    "AnimMontage",
    "MontagePreview",
    "MontageEditor",
]
