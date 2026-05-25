"""
Screen Management Module.

Provides screen lifecycle management, navigation stack, and transitions
for the UI layer.
"""

from .screen import (
    Screen,
    ScreenParams,
    ScreenResult,
    ScreenState,
    LifecycleCallback,
    StateChangeCallback,
)

from .screen_stack import (
    ScreenStack,
    ScreenCache,
    ScreenFactory,
    StackOperation,
    StackEventCallback,
    HistoryEntry,
)

from .transitions import (
    # Enums
    TransitionDirection,
    Easing,
    # Types
    EasingFunction,
    TransformFunction,
    # Functions
    get_easing_function,
    # Interface
    ITransition,
    # Base class
    BaseTransition,
    # Transitions
    FadeTransition,
    SlideTransition,
    ZoomTransition,
    InstantTransition,
    CompositeTransition,
    CustomTransition,
    # Factory
    TransitionFactory,
    # Constants
    DEFAULT_SCREEN_WIDTH,
    DEFAULT_SCREEN_HEIGHT,
)


__all__ = [
    # Screen
    "Screen",
    "ScreenParams",
    "ScreenResult",
    "ScreenState",
    "LifecycleCallback",
    "StateChangeCallback",
    # Stack
    "ScreenStack",
    "ScreenCache",
    "ScreenFactory",
    "StackOperation",
    "StackEventCallback",
    "HistoryEntry",
    # Transitions
    "TransitionDirection",
    "Easing",
    "EasingFunction",
    "TransformFunction",
    "get_easing_function",
    "ITransition",
    "BaseTransition",
    "FadeTransition",
    "SlideTransition",
    "ZoomTransition",
    "InstantTransition",
    "CompositeTransition",
    "CustomTransition",
    "TransitionFactory",
    "DEFAULT_SCREEN_WIDTH",
    "DEFAULT_SCREEN_HEIGHT",
]
