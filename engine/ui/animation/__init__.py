"""
UI Animation System for the AI Game Engine.

This module provides a comprehensive animation system including:
- Tweening with easing functions
- Keyframe-based animations
- Animation triggers (state, event, property, data)
- Animator components for widgets

Example usage:
    # Simple tween
    tween = Tween(widget, "opacity", 1.0, 0.0, duration=0.5)
    tween.start()

    # Keyframe animation
    animation = KeyframeAnimation("fade_in")
    animation.add_track(KeyframeTrack(
        "opacity",
        [Keyframe(0.0, 0.0), Keyframe(1.0, 1.0, "ease_out")]
    ))

    # Animator with state machine
    animator = Animator(widget)
    animator.add_state("idle", idle_animation)
    animator.add_state("hover", hover_animation)
    animator.add_transition("idle", "hover", duration=0.2)

    # Triggers
    trigger = StateTrigger(WidgetState.HOVERED)
    trigger.on_activate(lambda: animator.transition_to("hover"))
"""

from engine.ui.animation.easing import (
    # Type
    EasingFunction,
    EasingType,
    # Linear
    linear,
    # Quad
    quad_in,
    quad_out,
    quad_in_out,
    # Cubic
    cubic_in,
    cubic_out,
    cubic_in_out,
    # Quart
    quart_in,
    quart_out,
    quart_in_out,
    # Quint
    quint_in,
    quint_out,
    quint_in_out,
    # Sine
    sine_in,
    sine_out,
    sine_in_out,
    # Expo
    expo_in,
    expo_out,
    expo_in_out,
    # Circ
    circ_in,
    circ_out,
    circ_in_out,
    # Elastic
    elastic_in,
    elastic_out,
    elastic_in_out,
    # Back
    back_in,
    back_out,
    back_in_out,
    # Bounce
    bounce_in,
    bounce_out,
    bounce_in_out,
    # Bezier
    CubicBezier,
    create_bezier,
    EASE,
    EASE_IN,
    EASE_OUT,
    EASE_IN_OUT,
    # Registry
    get_easing,
    # Utils
    clamp,
    lerp,
)

from engine.ui.animation.tween import (
    # Core
    Tween,
    TweenState,
    TweenConfig,
    LoopMode,
    # Sequences/Groups
    TweenSequence,
    TweenGroup,
    # Manager
    TweenManager,
    # Factory functions
    tween_to,
    tween_from,
    tween_by,
    # Callback types
    TweenCallback,
    TweenUpdateCallback,
    TweenValueCallback,
)

from engine.ui.animation.keyframe import (
    # Core types
    Keyframe,
    LoopMode as KeyframeLoopMode,
    # Track
    KeyframeTrack,
    # Animation
    KeyframeAnimation,
    # Manager
    KeyframeAnimationManager,
    # Factory functions
    create_keyframe_animation,
    create_property_track,
)

from engine.ui.animation.triggers import (
    # Base
    TriggerBase,
    TriggerState,
    # Concrete triggers
    StateTrigger,
    EventTrigger,
    PropertyTrigger,
    DataTrigger,
    MultiTrigger,
    # Logic
    TriggerLogic,
    # Widget states
    WidgetState,
    # Event types
    EventType,
    # Factory functions
    on_hover,
    on_press,
    on_focus,
    on_click,
    on_value_change,
    when_property,
    when_data,
)

from engine.ui.animation.animator import (
    # Core
    Animator,
    AnimatorState,
    # Animation state
    AnimationState,
    # Transition
    AnimationTransition,
    TransitionCondition,
    # Layer
    AnimationLayer,
    LayerBlendMode,
    # Manager
    AnimatorManager,
)

__all__ = [
    # ==========================================================================
    # EASING
    # ==========================================================================
    "EasingFunction",
    "EasingType",
    "linear",
    "quad_in",
    "quad_out",
    "quad_in_out",
    "cubic_in",
    "cubic_out",
    "cubic_in_out",
    "quart_in",
    "quart_out",
    "quart_in_out",
    "quint_in",
    "quint_out",
    "quint_in_out",
    "sine_in",
    "sine_out",
    "sine_in_out",
    "expo_in",
    "expo_out",
    "expo_in_out",
    "circ_in",
    "circ_out",
    "circ_in_out",
    "elastic_in",
    "elastic_out",
    "elastic_in_out",
    "back_in",
    "back_out",
    "back_in_out",
    "bounce_in",
    "bounce_out",
    "bounce_in_out",
    "CubicBezier",
    "create_bezier",
    "EASE",
    "EASE_IN",
    "EASE_OUT",
    "EASE_IN_OUT",
    "get_easing",
    "clamp",
    "lerp",
    # ==========================================================================
    # TWEEN
    # ==========================================================================
    "Tween",
    "TweenState",
    "TweenConfig",
    "LoopMode",
    "TweenSequence",
    "TweenGroup",
    "TweenManager",
    "tween_to",
    "tween_from",
    "tween_by",
    "TweenCallback",
    "TweenUpdateCallback",
    "TweenValueCallback",
    # ==========================================================================
    # KEYFRAME
    # ==========================================================================
    "Keyframe",
    "KeyframeLoopMode",
    "KeyframeTrack",
    "KeyframeAnimation",
    "KeyframeAnimationManager",
    "create_keyframe_animation",
    "create_property_track",
    # ==========================================================================
    # TRIGGERS
    # ==========================================================================
    "TriggerBase",
    "TriggerState",
    "StateTrigger",
    "EventTrigger",
    "PropertyTrigger",
    "DataTrigger",
    "MultiTrigger",
    "TriggerLogic",
    "WidgetState",
    "EventType",
    "on_hover",
    "on_press",
    "on_focus",
    "on_click",
    "on_value_change",
    "when_property",
    "when_data",
    # ==========================================================================
    # ANIMATOR
    # ==========================================================================
    "Animator",
    "AnimatorState",
    "AnimationState",
    "AnimationTransition",
    "TransitionCondition",
    "AnimationLayer",
    "LayerBlendMode",
    "AnimatorManager",
]
