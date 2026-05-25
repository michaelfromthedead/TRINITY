"""
Example usage of Trinity Pattern Tier 34 (NARRATIVE) and Tier 35 (CINEMATICS) decorators.

This demonstrates how to use dialogue, conversation, voice_over, cutscene, and camera_track
decorators for game narrative and cinematic sequences.
"""

from trinity.decorators.cinematics import camera_track, cutscene
from trinity.decorators.narrative import conversation, dialogue, voice_over


# ==============================================================================
# NARRATIVE EXAMPLES
# ==============================================================================


@dialogue(id="hero_greeting", speaker="hero")
class HeroGreeting:
    """A dialogue node for the hero's greeting."""

    text = "Hello, traveler! Welcome to our village."


@dialogue(id="npc_response", speaker="village_elder")
@voice_over(audio_asset="assets/audio/elder_response.wav", lip_sync="assets/lipsync/elder_response.json")
class ElderResponse:
    """Voiced dialogue from the village elder."""

    text = "We've been expecting you. The prophecy foretold your arrival."


@conversation(id="quest_intro", start_node="hero_greeting")
class QuestIntroConversation:
    """Main quest introduction conversation tree."""

    nodes = {
        "hero_greeting": HeroGreeting,
        "npc_response": ElderResponse,
    }


@dialogue(id="narration_intro", speaker=None)
class NarrationNode:
    """Narrator voice - no specific speaker."""

    text = "In a land far away, a hero begins their journey..."


# ==============================================================================
# CINEMATICS EXAMPLES
# ==============================================================================


@cutscene(id="game_intro", skippable=False, pause_gameplay=True)
class GameIntroCutscene:
    """Non-skippable intro cutscene that pauses gameplay."""

    duration = 45.0
    scenes = ["opening_shot", "title_card", "hero_awakens"]


@camera_track(blend_in=1.0, blend_out=1.0)
class SmoothCameraTrack:
    """Camera track with 1 second blend in/out."""

    path = "bezier_curve_path_1"
    duration = 10.0


@cutscene(id="boss_encounter", skippable=True, pause_gameplay=False)
@camera_track(blend_in=0.5, blend_out=2.0)
class BossEncounterCutscene:
    """Skippable boss encounter cutscene with custom camera blend times."""

    trigger = "boss_health_50_percent"


# ==============================================================================
# COMBINED NARRATIVE + CINEMATICS
# ==============================================================================


@camera_track(blend_in=0.75, blend_out=0.75)
@cutscene(id="quest_complete", skippable=True, pause_gameplay=True)
@voice_over(audio_asset="assets/audio/quest_complete.wav")
@dialogue(id="quest_complete_dialogue", speaker="quest_giver")
class QuestCompleteCutscene:
    """Full quest completion scene with dialogue, voice, cutscene, and camera."""

    text = "You have completed the quest! Your courage knows no bounds."
    reward_items = ["legendary_sword", "gold_coins_1000"]


# ==============================================================================
# INTROSPECTION
# ==============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("NARRATIVE & CINEMATICS DECORATOR EXAMPLES")
    print("=" * 70)

    # Check dialogue decorator
    print("\n1. HeroGreeting Dialogue:")
    print(f"   - Is dialogue: {HeroGreeting._dialogue}")
    print(f"   - Dialogue ID: {HeroGreeting._dialogue_id}")
    print(f"   - Speaker: {HeroGreeting._dialogue_speaker}")
    print(f"   - Registries: {HeroGreeting._registries}")

    # Check voice over
    print("\n2. ElderResponse with Voice Over:")
    print(f"   - Is dialogue: {ElderResponse._dialogue}")
    print(f"   - Has voice over: {ElderResponse._voice_over}")
    print(f"   - Audio asset: {ElderResponse._voice_over_audio_asset}")
    print(f"   - Lip sync: {ElderResponse._voice_over_lip_sync}")

    # Check conversation
    print("\n3. QuestIntroConversation:")
    print(f"   - Is conversation: {QuestIntroConversation._conversation}")
    print(f"   - Conversation ID: {QuestIntroConversation._conversation_id}")
    print(f"   - Start node: {QuestIntroConversation._conversation_start_node}")

    # Check cutscene
    print("\n4. GameIntroCutscene:")
    print(f"   - Is cutscene: {GameIntroCutscene._cutscene}")
    print(f"   - Cutscene ID: {GameIntroCutscene._cutscene_id}")
    print(f"   - Skippable: {GameIntroCutscene._cutscene_skippable}")
    print(f"   - Pause gameplay: {GameIntroCutscene._cutscene_pause_gameplay}")

    # Check camera track
    print("\n5. SmoothCameraTrack:")
    print(f"   - Is camera track: {SmoothCameraTrack._camera_track}")
    print(f"   - Blend in: {SmoothCameraTrack._camera_track_blend_in}s")
    print(f"   - Blend out: {SmoothCameraTrack._camera_track_blend_out}s")

    # Check combined decorators
    print("\n6. QuestCompleteCutscene (Combined):")
    print(f"   - Applied decorators: {QuestCompleteCutscene._applied_decorators}")
    print(f"   - Is cutscene: {QuestCompleteCutscene._cutscene}")
    print(f"   - Has camera track: {QuestCompleteCutscene._camera_track}")
    print(f"   - Has voice over: {QuestCompleteCutscene._voice_over}")
    print(f"   - Is dialogue: {QuestCompleteCutscene._dialogue}")

    # Check tags
    print("\n7. Tags on QuestCompleteCutscene:")
    for key, value in sorted(QuestCompleteCutscene._tags.items()):
        print(f"   - {key}: {value}")

    print("\n" + "=" * 70)
