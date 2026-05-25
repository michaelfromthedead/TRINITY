"""XR Interaction module for VR/AR interaction systems.

This module provides the core interaction system for XR applications,
supporting multiple interaction paradigms:

- **Direct Interaction**: Touch, grab, manipulate through physical contact
- **Indirect Interaction**: Ray casting, gaze-based, teleportation
- **Physics Interaction**: Physics-based grabbing, throwing, pushing

Key Components:
    XRInteractable: Base component for all interactable objects
    XRGrabbable: Grabbable objects with physics/kinematic attachment
    XRSocket: Snap socket system for attaching objects
    RayInteractor: Ray/laser pointer based interaction
    DirectInteractor: Touch/poke based interaction
    GazeInteractor: Eye/head gaze based interaction

Decorators:
    @xr_interactable: Mark a class as XR interactable
    @xr_grabbable: Mark a class as XR grabbable
    @xr_socket: Mark a class as XR socket
"""

# Base interactable types
from engine.xr.interaction.interactable import (
    InteractionState,
    InteractionType,
    InteractorType,
    InteractionEvent,
    InteractionHit,
    XRInteractable,
    InteractableManager,
    xr_interactable,
)

# Grabbable system
from engine.xr.interaction.grabbable import (
    GrabType,
    AttachmentMode,
    HandPoseMode,
    GrabAttachPoint,
    ThrowData,
    GrabState,
    XRGrabbable,
    xr_grabbable,
)

# Socket system
from engine.xr.interaction.socket import (
    SnapBehavior,
    EjectBehavior,
    SocketState,
    SocketAttachEvent,
    SocketDetachEvent,
    XRSocket,
    SocketManager,
    xr_socket,
)

# Ray interactor
from engine.xr.interaction.ray_interactor import (
    RayVisualMode,
    RayHitIndicator,
    RayConfig,
    RayState,
    RayCastResult,
    RayInteractor,
)

# Direct/poke interactor
from engine.xr.interaction.direct_interactor import (
    PokeMode,
    GrabDetection,
    DirectConfig,
    ContactPoint,
    DirectState,
    DirectInteractor,
    MultiPointDirectInteractor,
)

# Gaze interactor
from engine.xr.interaction.gaze_interactor import (
    GazeSource,
    ActivationMode,
    DwellIndicator,
    GazeConfig,
    EyeTrackingData,
    GazeState,
    GazeInteractor,
)

__all__ = [
    # interactable.py
    "InteractionState",
    "InteractionType",
    "InteractorType",
    "InteractionEvent",
    "InteractionHit",
    "XRInteractable",
    "InteractableManager",
    "xr_interactable",
    # grabbable.py
    "GrabType",
    "AttachmentMode",
    "HandPoseMode",
    "GrabAttachPoint",
    "ThrowData",
    "GrabState",
    "XRGrabbable",
    "xr_grabbable",
    # socket.py
    "SnapBehavior",
    "EjectBehavior",
    "SocketState",
    "SocketAttachEvent",
    "SocketDetachEvent",
    "XRSocket",
    "SocketManager",
    "xr_socket",
    # ray_interactor.py
    "RayVisualMode",
    "RayHitIndicator",
    "RayConfig",
    "RayState",
    "RayCastResult",
    "RayInteractor",
    # direct_interactor.py
    "PokeMode",
    "GrabDetection",
    "DirectConfig",
    "ContactPoint",
    "DirectState",
    "DirectInteractor",
    "MultiPointDirectInteractor",
    # gaze_interactor.py
    "GazeSource",
    "ActivationMode",
    "DwellIndicator",
    "GazeConfig",
    "EyeTrackingData",
    "GazeState",
    "GazeInteractor",
]
