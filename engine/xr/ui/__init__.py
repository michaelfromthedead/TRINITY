"""XR UI module for immersive user interfaces.

This module provides XR-specific UI components for VR/AR/MR applications:
- World-space UI panels
- Head-locked UI elements
- Hand-attached interfaces
- Wrist-based UI (watch-style)
- XR buttons, sliders, and virtual keyboards

UI interaction modes:
- Ray: Laser pointer interaction from controllers
- Poke: Direct touch/poke interaction with hands
- Gaze: Eye tracking + dwell time activation

UI feedback types:
- Visual: Highlights, color changes, depth cues
- Haptic: Vibration feedback on controllers/hands
- Audio: Click sounds and feedback tones
"""

from __future__ import annotations

from .panel import (
    XRUIPanel,
    XRPanelType,
    XRInteractionMode,
    xr_ui_panel,
    UIInteractionManager,
)
from .button import (
    XRButton,
    XRButtonState,
    xr_button,
)
from .slider import (
    XRSlider,
    XRSliderOrientation,
    xr_slider,
)
from .keyboard import (
    VirtualKeyboard,
    KeyboardLayout,
    KeyType,
)
from .wrist_ui import (
    WristUI,
    WristUIPosition,
    WristMenuItem,
)


__all__ = [
    # Panel
    "XRUIPanel",
    "XRPanelType",
    "XRInteractionMode",
    "xr_ui_panel",
    "UIInteractionManager",
    # Button
    "XRButton",
    "XRButtonState",
    "xr_button",
    # Slider
    "XRSlider",
    "XRSliderOrientation",
    "xr_slider",
    # Keyboard
    "VirtualKeyboard",
    "KeyboardLayout",
    "KeyType",
    # Wrist UI
    "WristUI",
    "WristUIPosition",
    "WristMenuItem",
]
