"""Tests for XR module main __init__.py exports.

Tests verify that all public APIs are properly exported from the main
engine.xr module and can be imported correctly.
"""

import pytest
import importlib


class TestXRModuleImports:
    """Tests for XR module import structure."""

    def test_module_importable(self) -> None:
        """Test that the XR module can be imported."""
        import engine.xr as xr
        assert xr is not None

    def test_version_info(self) -> None:
        """Test version info is available."""
        import engine.xr
        assert hasattr(engine.xr, "__version__")
        assert isinstance(engine.xr.__version__, str)

    def test_all_exports_defined(self) -> None:
        """Test that __all__ is defined."""
        import engine.xr
        assert hasattr(engine.xr, "__all__")
        assert isinstance(engine.xr.__all__, list)
        assert len(engine.xr.__all__) > 0


class TestRuntimeExports:
    """Tests for runtime module exports."""

    def test_xr_runtime_exports(self) -> None:
        """Test XR runtime classes are exported and functional."""
        from engine.xr import (
            XRRuntime,
            XRRuntimeType,
            XRRuntimeState,
            XRRuntimeError,
            XRRuntimeNotAvailableError,
        )

        # Verify classes exist and are proper types
        assert XRRuntime is not None
        assert callable(XRRuntime), "XRRuntime should be callable (class)"

        # Verify XRRuntimeType enum has expected values
        assert hasattr(XRRuntimeType, 'OPENXR'), "XRRuntimeType should have OPENXR"
        assert hasattr(XRRuntimeType, 'STEAMVR'), "XRRuntimeType should have STEAMVR"

        # XRRuntimeState is a dataclass, verify it has expected fields
        state = XRRuntimeState()
        assert hasattr(state, 'session_state'), "XRRuntimeState should have session_state"

        # Verify error classes are proper exceptions
        assert issubclass(XRRuntimeError, Exception), "XRRuntimeError should be an Exception subclass"
        assert issubclass(XRRuntimeNotAvailableError, XRRuntimeError), "XRRuntimeNotAvailableError should extend XRRuntimeError"

    def test_runtime_functions(self) -> None:
        """Test runtime factory functions are exported."""
        from engine.xr import create_runtime, detect_available_runtimes

        assert callable(create_runtime)
        assert callable(detect_available_runtimes)

    def test_pose_and_view_exports(self) -> None:
        """Test Pose and ViewInfo are exported."""
        from engine.xr import Pose, ViewInfo

        assert Pose is not None
        assert ViewInfo is not None

    def test_session_exports(self) -> None:
        """Test session classes are exported."""
        from engine.xr import (
            XRSession,
            XRSessionConfig,
            XRSessionState,
            XRSessionMode,
            XRReferenceSpace,
            XRSessionStats,
            XRSessionError,
            InvalidStateTransitionError,
        )

        assert XRSession is not None
        assert XRSessionConfig is not None
        assert XRSessionState is not None
        assert XRSessionMode is not None
        assert XRReferenceSpace is not None
        assert XRSessionStats is not None
        assert XRSessionError is not None
        assert InvalidStateTransitionError is not None

    def test_capabilities_exports(self) -> None:
        """Test capability classes are exported."""
        from engine.xr import (
            XRCapabilities,
            XRFeature,
            DisplaySpecs,
            TrackingCapabilities,
            RenderingCapabilities,
            InputCapabilities,
            detect_capabilities,
            create_fallback_capabilities,
        )

        assert XRCapabilities is not None
        assert XRFeature is not None
        assert DisplaySpecs is not None
        assert TrackingCapabilities is not None
        assert RenderingCapabilities is not None
        assert InputCapabilities is not None
        assert callable(detect_capabilities)
        assert callable(create_fallback_capabilities)


class TestInputExports:
    """Tests for input module exports."""

    def test_hmd_exports(self) -> None:
        """Test HMD classes are exported and functional."""
        from engine.xr import (
            HeadMountedDisplay,
            HMDTrackingState,
            HMD_STATE_TRANSITIONS,
            HMDDisplayInfo,
            PredictionConfig,
        )

        # Verify classes exist and are callable
        assert HeadMountedDisplay is not None
        assert callable(HeadMountedDisplay), "HeadMountedDisplay should be callable"

        # Verify enum has expected states
        assert hasattr(HMDTrackingState, 'TRACKING'), "Should have TRACKING state"
        assert hasattr(HMDTrackingState, 'LIMITED'), "Should have LIMITED state"
        assert hasattr(HMDTrackingState, 'LOST'), "Should have LOST state"

        # Verify state transitions is a proper mapping
        assert isinstance(HMD_STATE_TRANSITIONS, dict), "HMD_STATE_TRANSITIONS should be a dict"
        assert len(HMD_STATE_TRANSITIONS) > 0, "HMD_STATE_TRANSITIONS should not be empty"

        # Verify dataclasses can be instantiated
        display_info = HMDDisplayInfo()
        assert hasattr(display_info, 'resolution_per_eye'), "HMDDisplayInfo should have resolution_per_eye"
        assert hasattr(display_info, 'ipd'), "HMDDisplayInfo should have ipd"

        pred_config = PredictionConfig()
        assert hasattr(pred_config, 'enabled'), "PredictionConfig should have enabled attribute"
        assert hasattr(pred_config, 'prediction_time_ms'), "PredictionConfig should have prediction_time_ms"

    def test_controller_exports(self) -> None:
        """Test controller classes are exported and functional."""
        from engine.xr import (
            XRController,
            XRHand,
            XRButton,
            XRControllerType,
            ControllerCapabilities,
            ButtonState,
        )

        # Verify classes exist and are callable
        assert XRController is not None
        assert callable(XRController), "XRController should be callable"

        # Verify XRHand enum has expected values
        assert hasattr(XRHand, 'LEFT'), "XRHand should have LEFT"
        assert hasattr(XRHand, 'RIGHT'), "XRHand should have RIGHT"

        # XRButton is a class, not enum - verify it's callable
        assert callable(XRButton), "XRButton should be callable"

        # ButtonState is a dataclass with button state fields
        button_state = ButtonState()
        assert hasattr(button_state, 'pressed'), "ButtonState should have pressed"
        assert hasattr(button_state, 'touched'), "ButtonState should have touched"
        assert hasattr(button_state, 'value'), "ButtonState should have value"

        # Verify capabilities dataclass
        caps = ControllerCapabilities()
        assert hasattr(caps, 'has_haptics'), "ControllerCapabilities should have has_haptics"

    def test_hand_tracking_exports(self) -> None:
        """Test hand tracking classes are exported and functional."""
        from engine.xr import (
            HandTrackingData,
            HandJoint,
            HAND_JOINT_COUNT,
            GestureType,
            JointData,
            GestureResult,
            GestureRecognizer,
            GestureEvent,
            HandTracker,
        )

        # Verify joint count constant
        assert HAND_JOINT_COUNT == 26, "Hand should have 26 joints"

        # Verify joint enum has expected values
        assert hasattr(HandJoint, 'WRIST'), "HandJoint should have WRIST"
        assert hasattr(HandJoint, 'INDEX_TIP'), "HandJoint should have INDEX_TIP"
        assert hasattr(HandJoint, 'THUMB_TIP'), "HandJoint should have THUMB_TIP"

        # Verify gesture enum has expected values
        assert hasattr(GestureType, 'PINCH'), "GestureType should have PINCH"
        assert hasattr(GestureType, 'FIST'), "GestureType should have FIST"
        assert hasattr(GestureType, 'OPEN_HAND'), "GestureType should have OPEN_HAND"

        # Verify classes are callable
        assert callable(HandTracker), "HandTracker should be callable"
        assert callable(GestureRecognizer), "GestureRecognizer should be callable"

        # Verify JointData dataclass structure
        joint_data = JointData()
        assert hasattr(joint_data, 'position'), "JointData should have position"
        assert hasattr(joint_data, 'orientation'), "JointData should have orientation"
        assert hasattr(joint_data, 'radius'), "JointData should have radius"

    def test_eye_tracking_exports(self) -> None:
        """Test eye tracking classes are exported and functional."""
        from engine.xr import (
            EyeTrackingData,
            EyeId,
            CalibrationState,
            GazeState,
            EyeData,
            FixationData,
            SaccadeData,
            BlinkData,
            FixationDetector,
            BlinkDetector,
            CalibrationPoint,
            EyeCalibration,
            EyeTracker,
        )

        # Verify EyeId enum has expected values
        assert hasattr(EyeId, 'LEFT'), "EyeId should have LEFT"
        assert hasattr(EyeId, 'RIGHT'), "EyeId should have RIGHT"
        assert hasattr(EyeId, 'COMBINED'), "EyeId should have COMBINED"

        # CalibrationState enum values
        assert hasattr(CalibrationState, 'NOT_STARTED'), "CalibrationState should have NOT_STARTED"
        assert hasattr(CalibrationState, 'COMPLETED'), "CalibrationState should have COMPLETED"

        # GazeState enum values
        assert hasattr(GazeState, 'FIXATION'), "GazeState should have FIXATION"
        assert hasattr(GazeState, 'SACCADE'), "GazeState should have SACCADE"

        # Verify classes are callable
        assert callable(EyeTracker), "EyeTracker should be callable"
        assert callable(FixationDetector), "FixationDetector should be callable"
        assert callable(BlinkDetector), "BlinkDetector should be callable"

        # Verify dataclasses can be instantiated
        eye_data = EyeData()
        assert hasattr(eye_data, 'gaze_direction'), "EyeData should have gaze_direction"

        fixation_data = FixationData()
        assert hasattr(fixation_data, 'position'), "FixationData should have position"


class TestRenderingExports:
    """Tests for rendering module exports."""

    def test_stereo_exports(self) -> None:
        """Test stereo rendering classes are exported and functional."""
        from engine.xr import (
            StereoRenderer,
            StereoMethod,
            ProjectionType,
            IPDMode,
            EyeIndex,
            ViewFrustum,
            EyeView,
            StereoConfig,
            StereoRenderTarget,
        )

        # Verify StereoMethod enum has expected values
        assert hasattr(StereoMethod, 'MULTI_VIEW'), "StereoMethod should have MULTI_VIEW"
        assert hasattr(StereoMethod, 'INSTANCED'), "StereoMethod should have INSTANCED"
        assert hasattr(StereoMethod, 'SEQUENTIAL'), "StereoMethod should have SEQUENTIAL"

        # Verify ProjectionType enum has expected values
        assert hasattr(ProjectionType, 'SYMMETRIC'), "ProjectionType should have SYMMETRIC"
        assert hasattr(ProjectionType, 'ASYMMETRIC'), "ProjectionType should have ASYMMETRIC"

        # Verify EyeIndex enum has expected values
        assert hasattr(EyeIndex, 'LEFT'), "EyeIndex should have LEFT"
        assert hasattr(EyeIndex, 'RIGHT'), "EyeIndex should have RIGHT"

        # Verify IPDMode enum has expected values
        assert hasattr(IPDMode, 'HARDWARE'), "IPDMode should have HARDWARE"
        assert hasattr(IPDMode, 'SOFTWARE'), "IPDMode should have SOFTWARE"

        # Verify classes are callable
        assert callable(StereoRenderer), "StereoRenderer should be callable"

        # Verify dataclasses can be instantiated
        config = StereoConfig()
        assert hasattr(config, 'ipd_meters'), "StereoConfig should have ipd_meters"
        assert hasattr(config, 'near_plane'), "StereoConfig should have near_plane"
        assert hasattr(config, 'far_plane'), "StereoConfig should have far_plane"

        frustum = ViewFrustum()
        assert hasattr(frustum, 'left'), "ViewFrustum should have left"
        assert hasattr(frustum, 'right'), "ViewFrustum should have right"
        assert hasattr(frustum, 'top'), "ViewFrustum should have top"
        assert hasattr(frustum, 'bottom'), "ViewFrustum should have bottom"


class TestInteractionExports:
    """Tests for interaction module exports."""

    def test_interactable_exports(self) -> None:
        """Test interactable classes are exported and functional."""
        from engine.xr import (
            XRInteractable,
            InteractionState,
            InteractionType,
            InteractorType,
            InteractionEvent,
            InteractionHit,
            xr_interactable,
        )

        # Verify InteractionState enum has expected values
        assert hasattr(InteractionState, 'IDLE'), "InteractionState should have IDLE"
        assert hasattr(InteractionState, 'HOVERED'), "InteractionState should have HOVERED"
        assert hasattr(InteractionState, 'SELECTED'), "InteractionState should have SELECTED"

        # Verify InteractionType enum has expected values
        assert hasattr(InteractionType, 'HOVER'), "InteractionType should have HOVER"
        assert hasattr(InteractionType, 'SELECT'), "InteractionType should have SELECT"
        assert hasattr(InteractionType, 'GRAB'), "InteractionType should have GRAB"

        # Verify InteractorType enum has expected values
        assert hasattr(InteractorType, 'RAY'), "InteractorType should have RAY"
        assert hasattr(InteractorType, 'DIRECT'), "InteractorType should have DIRECT"
        assert hasattr(InteractorType, 'GAZE'), "InteractorType should have GAZE"

        # Verify decorator is callable
        assert callable(xr_interactable), "xr_interactable should be callable"

        # Test decorator application
        @xr_interactable()
        class TestInteractable:
            pass

        assert hasattr(TestInteractable, '_xr_interactable'), "Decorator should add _xr_interactable attr"

    def test_grabbable_exports(self) -> None:
        """Test grabbable classes are exported and functional."""
        from engine.xr import (
            XRGrabbable,
            GrabType,
            GrabState,
            GrabAttachPoint,
            AttachmentMode,
            HandPoseMode,
            ThrowData,
            xr_grabbable,
        )

        # Verify GrabType enum has expected values
        assert hasattr(GrabType, 'DIRECT'), "GrabType should have DIRECT"
        assert hasattr(GrabType, 'RAY'), "GrabType should have RAY"
        assert hasattr(GrabType, 'SOCKET'), "GrabType should have SOCKET"

        # GrabState, AttachmentMode, HandPoseMode - verify they are proper types
        assert GrabState is not None, "GrabState should be exported"
        assert AttachmentMode is not None, "AttachmentMode should be exported"
        assert HandPoseMode is not None, "HandPoseMode should be exported"

        # ThrowData requires positional arguments - verify the class exists and is callable
        assert callable(ThrowData), "ThrowData should be callable"

        # GrabAttachPoint - verify structure
        attach_point = GrabAttachPoint()
        assert hasattr(attach_point, 'local_position'), "GrabAttachPoint should have local_position"
        assert hasattr(attach_point, 'local_rotation'), "GrabAttachPoint should have local_rotation"

        # Verify decorator is callable
        assert callable(xr_grabbable), "xr_grabbable should be callable"

    def test_socket_exports(self) -> None:
        """Test socket classes are exported."""
        from engine.xr import (
            XRSocket,
            SnapBehavior,
            EjectBehavior,
            SocketState,
            SocketAttachEvent,
            SocketDetachEvent,
            SocketManager,
            xr_socket,
        )

        assert XRSocket is not None
        assert SnapBehavior is not None
        assert EjectBehavior is not None
        assert SocketState is not None
        assert SocketAttachEvent is not None
        assert SocketDetachEvent is not None
        assert SocketManager is not None
        assert callable(xr_socket)


class TestSpatialExports:
    """Tests for spatial module exports."""

    def test_anchor_exports(self) -> None:
        """Test anchor classes are exported and functional."""
        from engine.xr import (
            SpatialAnchor,
            AnchorType,
            AnchorTrackingState,
            AnchorPersistenceState,
            AnchorPose,
            CloudAnchorConfig,
            spatial_anchor,
        )

        # Verify AnchorType enum has expected values
        assert hasattr(AnchorType, 'LOCAL'), "AnchorType should have LOCAL"
        assert hasattr(AnchorType, 'CLOUD'), "AnchorType should have CLOUD"

        # Verify AnchorTrackingState enum has expected values
        assert hasattr(AnchorTrackingState, 'NOT_TRACKING'), "AnchorTrackingState should have NOT_TRACKING"
        assert hasattr(AnchorTrackingState, 'TRACKING'), "AnchorTrackingState should have TRACKING"

        # AnchorPersistenceState - verify it exists
        assert AnchorPersistenceState is not None, "AnchorPersistenceState should be exported"

        # Verify AnchorPose and CloudAnchorConfig classes exist
        assert callable(AnchorPose), "AnchorPose should be callable"
        assert callable(CloudAnchorConfig), "CloudAnchorConfig should be callable"

        # Verify decorator is callable
        assert callable(spatial_anchor), "spatial_anchor should be callable"


class TestPlatformExports:
    """Tests for platform module exports."""

    def test_device_exports(self) -> None:
        """Test device classes are exported."""
        from engine.xr import (
            XRDevice,
            XRPlatformType,
            XRPlatformInfo,
            XRDeviceCapabilities,
        )

        assert XRDevice is not None
        assert XRPlatformType is not None
        assert XRPlatformInfo is not None
        assert XRDeviceCapabilities is not None

    def test_platform_exports(self) -> None:
        """Test platform classes are exported."""
        from engine.xr import (
            XRPlatform,
            OpenXRPlatform,
            SteamVRPlatform,
            MetaQuestPlatform,
            AppleVisionProPlatform,
            PSVR2Platform,
            detect_xr_platform,
            get_device_capabilities,
        )

        assert XRPlatform is not None
        assert OpenXRPlatform is not None
        assert SteamVRPlatform is not None
        assert MetaQuestPlatform is not None
        assert AppleVisionProPlatform is not None
        assert PSVR2Platform is not None
        assert callable(detect_xr_platform)
        assert callable(get_device_capabilities)

    def test_guardian_exports(self) -> None:
        """Test guardian classes are exported."""
        from engine.xr import (
            GuardianMode,
            BoundaryType,
            ProximityLevel,
            BoundaryVertex,
            PlayAreaBounds,
            GuardianConfig,
            ProximityInfo,
            GuardianSystem,
            OpenXRGuardian,
            SteamVRGuardian,
            QuestGuardian,
            create_guardian_system,
        )

        assert GuardianMode is not None
        assert BoundaryType is not None
        assert ProximityLevel is not None
        assert BoundaryVertex is not None
        assert PlayAreaBounds is not None
        assert GuardianConfig is not None
        assert ProximityInfo is not None
        assert GuardianSystem is not None
        assert OpenXRGuardian is not None
        assert SteamVRGuardian is not None
        assert QuestGuardian is not None
        assert callable(create_guardian_system)

    def test_social_exports(self) -> None:
        """Test social services classes are exported."""
        from engine.xr import (
            UserPresence,
            FriendRelationship,
            PartyState,
            InviteType,
            VoiceChatState,
            UserProfile,
            Friend,
            PartyMember,
            Party,
            Invite,
            VoiceChannel,
            SocialServices,
            MetaSocialServices,
            SteamSocialServices,
            PlayStationSocialServices,
            create_social_services,
        )

        assert UserPresence is not None
        assert FriendRelationship is not None
        assert PartyState is not None
        assert InviteType is not None
        assert VoiceChatState is not None
        assert UserProfile is not None
        assert Friend is not None
        assert PartyMember is not None
        assert Party is not None
        assert Invite is not None
        assert VoiceChannel is not None
        assert SocialServices is not None
        assert MetaSocialServices is not None
        assert SteamSocialServices is not None
        assert PlayStationSocialServices is not None
        assert callable(create_social_services)


class TestAvatarExports:
    """Tests for avatar module exports."""

    def test_avatar_exports(self) -> None:
        """Test avatar classes are exported."""
        from engine.xr import (
            XRAvatar,
            IKSolver,
            IKChain,
            IKTarget,
            AvatarHand,
            AvatarCalibration,
        )

        assert XRAvatar is not None
        assert IKSolver is not None
        assert IKChain is not None
        assert IKTarget is not None
        assert AvatarHand is not None
        assert AvatarCalibration is not None


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_guardian_system(self) -> None:
        """Test guardian system factory."""
        from engine.xr import create_guardian_system, OpenXRGuardian

        guardian = create_guardian_system("openxr")
        assert isinstance(guardian, OpenXRGuardian)

    def test_create_social_services(self) -> None:
        """Test social services factory."""
        from engine.xr import create_social_services, MetaSocialServices

        services = create_social_services("meta")
        assert isinstance(services, MetaSocialServices)

    def test_get_device_capabilities(self) -> None:
        """Test device capabilities lookup."""
        from engine.xr import get_device_capabilities, XRDevice, XRDeviceCapabilities

        caps = get_device_capabilities(XRDevice.META_QUEST_3)
        assert isinstance(caps, XRDeviceCapabilities)
        assert caps.supports_hand_tracking is True
