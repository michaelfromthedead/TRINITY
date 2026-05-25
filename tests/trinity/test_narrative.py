"""
Tests for Trinity Pattern - Tier 34: NARRATIVE Decorators
"""

import pytest

from trinity.decorators.narrative import conversation, dialogue, voice_over
from trinity.decorators.registry import Tier, registry


class TestDialogueDecorator:
    """Test @dialogue decorator."""

    def test_basic_application(self):
        """Test basic @dialogue application."""

        @dialogue(id="greeting", speaker="hero")
        class GreetingNode:
            pass

        assert hasattr(GreetingNode, "_dialogue")
        assert GreetingNode._dialogue is True
        assert GreetingNode._dialogue_id == "greeting"
        assert GreetingNode._dialogue_speaker == "hero"
        assert "dialogue" in GreetingNode._applied_decorators

    def test_with_none_speaker(self):
        """Test @dialogue with speaker=None."""

        @dialogue(id="narration", speaker=None)
        class NarrationNode:
            pass

        assert NarrationNode._dialogue is True
        assert NarrationNode._dialogue_id == "narration"
        assert NarrationNode._dialogue_speaker is None

    def test_registry_registration(self):
        """Test that @dialogue is registered in the registry."""
        spec = registry.get("dialogue")
        assert spec is not None
        assert spec.name == "dialogue"
        assert spec.tier == Tier.NARRATIVE
        assert "class" in spec.target_types

    def test_tags_created(self):
        """Test that @dialogue creates proper tags."""

        @dialogue(id="test", speaker="npc")
        class TestNode:
            pass

        assert hasattr(TestNode, "_tags")
        assert TestNode._tags.get("dialogue") is True
        assert TestNode._tags.get("dialogue_id") == "test"
        assert TestNode._tags.get("dialogue_speaker") == "npc"

    def test_validation_empty_id(self):
        """Test @dialogue validation rejects empty id."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @dialogue(id="")
            class BadNode:
                pass

    def test_validation_no_id(self):
        """Test @dialogue validation rejects missing id."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @dialogue()
            class BadNode:
                pass

    def test_composition_with_other_decorators(self):
        """Test @dialogue can be composed with other decorators."""
        from trinity.decorators.ecs_core import component

        @dialogue(id="test", speaker="hero")
        @component(name="DialogueComponent")
        class DialogueComponent:
            pass

        assert DialogueComponent._dialogue is True
        assert DialogueComponent._component is True
        assert "dialogue" in DialogueComponent._applied_decorators
        assert "component" in DialogueComponent._applied_decorators


class TestConversationDecorator:
    """Test @conversation decorator."""

    def test_basic_application(self):
        """Test basic @conversation application."""

        @conversation(id="quest_intro", start_node="greeting")
        class QuestConversation:
            pass

        assert hasattr(QuestConversation, "_conversation")
        assert QuestConversation._conversation is True
        assert QuestConversation._conversation_id == "quest_intro"
        assert QuestConversation._conversation_start_node == "greeting"
        assert "conversation" in QuestConversation._applied_decorators

    def test_registry_registration(self):
        """Test that @conversation is registered in the registry."""
        spec = registry.get("conversation")
        assert spec is not None
        assert spec.name == "conversation"
        assert spec.tier == Tier.NARRATIVE
        assert "class" in spec.target_types

    def test_tags_created(self):
        """Test that @conversation creates proper tags."""

        @conversation(id="test_conv", start_node="node1")
        class TestConv:
            pass

        assert hasattr(TestConv, "_tags")
        assert TestConv._tags.get("conversation") is True
        assert TestConv._tags.get("conversation_id") == "test_conv"
        assert TestConv._tags.get("conversation_start_node") == "node1"

    def test_validation_empty_id(self):
        """Test @conversation validation rejects empty id."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @conversation(id="", start_node="node1")
            class BadConv:
                pass

    def test_validation_empty_start_node(self):
        """Test @conversation validation rejects empty start_node."""
        with pytest.raises(ValueError, match="start_node must be a non-empty string"):

            @conversation(id="test", start_node="")
            class BadConv:
                pass

    def test_validation_missing_params(self):
        """Test @conversation validation rejects missing params."""
        with pytest.raises(ValueError, match="start_node must be a non-empty string"):

            @conversation(id="test")
            class BadConv:
                pass

        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @conversation(start_node="node1")
            class BadConv:
                pass

    def test_composition(self):
        """Test @conversation composition with other decorators."""
        from trinity.decorators.ecs_core import resource

        @conversation(id="test_conv", start_node="start")
        @resource
        class ConversationState:
            pass

        assert ConversationState._conversation is True
        assert ConversationState._resource is True


class TestVoiceOverDecorator:
    """Test @voice_over decorator."""

    def test_basic_application(self):
        """Test basic @voice_over application."""

        @voice_over(audio_asset="hero_greeting.wav")
        class HeroGreeting:
            pass

        assert hasattr(HeroGreeting, "_voice_over")
        assert HeroGreeting._voice_over is True
        assert HeroGreeting._voice_over_audio_asset == "hero_greeting.wav"
        assert HeroGreeting._voice_over_lip_sync is None
        assert "voice_over" in HeroGreeting._applied_decorators

    def test_with_lip_sync(self):
        """Test @voice_over with lip_sync."""

        @voice_over(audio_asset="dialogue.wav", lip_sync="lipsync_data.json")
        class DialogueWithLipSync:
            pass

        assert DialogueWithLipSync._voice_over is True
        assert DialogueWithLipSync._voice_over_audio_asset == "dialogue.wav"
        assert DialogueWithLipSync._voice_over_lip_sync == "lipsync_data.json"

    def test_registry_registration(self):
        """Test that @voice_over is registered in the registry."""
        spec = registry.get("voice_over")
        assert spec is not None
        assert spec.name == "voice_over"
        assert spec.tier == Tier.NARRATIVE
        assert "class" in spec.target_types

    def test_tags_created(self):
        """Test that @voice_over creates proper tags."""

        @voice_over(audio_asset="test.wav", lip_sync="sync.json")
        class TestVO:
            pass

        assert hasattr(TestVO, "_tags")
        assert TestVO._tags.get("voice_over") is True
        assert TestVO._tags.get("voice_over_audio_asset") == "test.wav"
        assert TestVO._tags.get("voice_over_lip_sync") == "sync.json"

    def test_validation_empty_audio_asset(self):
        """Test @voice_over validation rejects empty audio_asset."""
        with pytest.raises(ValueError, match="audio_asset must be a non-empty string"):

            @voice_over(audio_asset="")
            class BadVO:
                pass

    def test_validation_missing_audio_asset(self):
        """Test @voice_over validation rejects missing audio_asset."""
        with pytest.raises(ValueError, match="audio_asset must be a non-empty string"):

            @voice_over()
            class BadVO:
                pass

    def test_composition(self):
        """Test @voice_over composition with @dialogue."""

        @voice_over(audio_asset="greeting.wav")
        @dialogue(id="greeting", speaker="hero")
        class VoicedDialogue:
            pass

        assert VoicedDialogue._voice_over is True
        assert VoicedDialogue._dialogue is True
        assert "voice_over" in VoicedDialogue._applied_decorators
        assert "dialogue" in VoicedDialogue._applied_decorators


class TestNarrativeIntegration:
    """Test integration between narrative decorators."""

    def test_full_conversation_setup(self):
        """Test full conversation with multiple dialogue nodes."""

        @conversation(id="quest_dialogue", start_node="start")
        class QuestDialogue:
            pass

        @dialogue(id="start", speaker="hero")
        class StartNode:
            pass

        @dialogue(id="response", speaker="npc")
        @voice_over(audio_asset="npc_response.wav")
        class ResponseNode:
            pass

        assert QuestDialogue._conversation is True
        assert QuestDialogue._conversation_start_node == "start"
        assert StartNode._dialogue_id == "start"
        assert ResponseNode._dialogue_id == "response"
        assert ResponseNode._voice_over_audio_asset == "npc_response.wav"

    def test_registries(self):
        """Test that all narrative decorators use narrative registry."""

        @dialogue(id="d1", speaker="s1")
        class D1:
            pass

        @conversation(id="c1", start_node="n1")
        class C1:
            pass

        @voice_over(audio_asset="a1.wav")
        class V1:
            pass

        assert "narrative" in D1._registries
        assert "narrative" in C1._registries
        assert "narrative" in V1._registries

    def test_tier_ordering(self):
        """Test that narrative decorators are in correct tier."""
        specs = registry.by_tier(Tier.NARRATIVE)
        names = {spec.name for spec in specs}
        assert "dialogue" in names
        assert "conversation" in names
        assert "voice_over" in names
