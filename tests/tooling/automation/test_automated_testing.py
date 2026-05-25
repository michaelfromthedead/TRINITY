"""
Tests for automated_testing.py - Bot testing and playtest sessions.
"""

import time

import pytest

from engine.tooling.automation.automated_testing import (
    BotAction,
    BotActionType,
    BotBehavior,
    BotController,
    CombatBehavior,
    ExplorationBehavior,
    GameBot,
    PlaytestRecorder,
    PlaytestReporter,
    PlaytestSession,
    RandomWalkBehavior,
    ScriptedBehavior,
    create_bot,
    run_playtest,
)


class TestBotActionType:
    """Tests for BotActionType enum."""

    def test_action_types_exist(self):
        assert BotActionType.MOVE
        assert BotActionType.JUMP
        assert BotActionType.ATTACK
        assert BotActionType.INTERACT
        assert BotActionType.NAVIGATE_TO
        assert BotActionType.CUSTOM


class TestBotAction:
    """Tests for BotAction dataclass."""

    def test_create_action(self):
        action = BotAction(
            action_type=BotActionType.MOVE,
            parameters={"direction": (1, 0, 0)},
        )

        assert action.action_type == BotActionType.MOVE
        assert action.parameters["direction"] == (1, 0, 0)

    def test_move_factory(self):
        action = BotAction.move((1, 0, 0), duration=0.5)

        assert action.action_type == BotActionType.MOVE
        assert action.parameters["direction"] == (1, 0, 0)
        assert action.duration == 0.5

    def test_jump_factory(self):
        action = BotAction.jump()

        assert action.action_type == BotActionType.JUMP

    def test_attack_factory(self):
        action = BotAction.attack(target=123)

        assert action.action_type == BotActionType.ATTACK
        assert action.parameters["target"] == 123

    def test_navigate_to_factory(self):
        action = BotAction.navigate_to((10, 0, 10))

        assert action.action_type == BotActionType.NAVIGATE_TO
        assert action.parameters["position"] == (10, 0, 10)

    def test_wait_factory(self):
        action = BotAction.wait(1.0)

        assert action.action_type == BotActionType.WAIT
        assert action.duration == 1.0

    def test_custom_factory(self):
        action = BotAction.custom("special_attack", power=100)

        assert action.action_type == BotActionType.CUSTOM
        assert action.parameters["name"] == "special_attack"
        assert action.parameters["power"] == 100

    def test_to_dict(self):
        action = BotAction.move((1, 0, 0))
        data = action.to_dict()

        assert data["type"] == "MOVE"
        assert data["parameters"]["direction"] == (1, 0, 0)


class TestRandomWalkBehavior:
    """Tests for RandomWalkBehavior class."""

    def test_create_behavior(self):
        behavior = RandomWalkBehavior()
        assert behavior.name == "random_walk"

    def test_get_next_action(self):
        behavior = RandomWalkBehavior()
        bot = GameBot("TestBot", behavior)

        action = behavior.get_next_action(bot, {})

        assert action.action_type == BotActionType.MOVE


class TestExplorationBehavior:
    """Tests for ExplorationBehavior class."""

    def test_create_behavior(self):
        behavior = ExplorationBehavior()
        assert behavior.name == "exploration"

    def test_get_next_action(self):
        behavior = ExplorationBehavior()
        bot = GameBot("TestBot", behavior)

        world_state = {
            "position": (0, 0, 0),
            "world_bounds": ((-100, 100), (0, 10), (-100, 100)),
        }

        action = behavior.get_next_action(bot, world_state)
        assert action is not None


class TestCombatBehavior:
    """Tests for CombatBehavior class."""

    def test_create_behavior(self):
        behavior = CombatBehavior(aggression=0.8)
        assert behavior.name == "combat"
        assert behavior.aggression == 0.8

    def test_attack_when_enemy_nearby(self):
        behavior = CombatBehavior(aggression=1.0)  # Always attack
        bot = GameBot("TestBot", behavior)

        world_state = {
            "nearby_enemies": [{"id": 1, "position": (5, 0, 0)}],
        }

        action = behavior.get_next_action(bot, world_state)
        # Should attack or navigate to enemy
        assert action.action_type in (BotActionType.ATTACK, BotActionType.NAVIGATE_TO)


class TestScriptedBehavior:
    """Tests for ScriptedBehavior class."""

    def test_create_behavior(self):
        actions = [
            BotAction.move((1, 0, 0)),
            BotAction.jump(),
            BotAction.wait(1.0),
        ]
        behavior = ScriptedBehavior(actions)

        assert behavior.name == "scripted"

    def test_follows_script(self):
        actions = [
            BotAction.move((1, 0, 0)),
            BotAction.jump(),
        ]
        behavior = ScriptedBehavior(actions)
        bot = GameBot("TestBot", behavior)

        action1 = behavior.get_next_action(bot, {})
        action2 = behavior.get_next_action(bot, {})

        assert action1.action_type == BotActionType.MOVE
        assert action2.action_type == BotActionType.JUMP

    def test_loops_script(self):
        actions = [BotAction.jump()]
        behavior = ScriptedBehavior(actions)
        bot = GameBot("TestBot", behavior)

        # Should loop after reaching end
        action1 = behavior.get_next_action(bot, {})
        action2 = behavior.get_next_action(bot, {})

        assert action1.action_type == BotActionType.JUMP
        assert action2.action_type == BotActionType.JUMP


class TestGameBot:
    """Tests for GameBot class."""

    def test_create_bot(self):
        behavior = RandomWalkBehavior()
        bot = GameBot("TestBot", behavior)

        assert bot.name == "TestBot"
        assert bot.is_active is False

    def test_start_stop(self):
        behavior = RandomWalkBehavior()
        bot = GameBot("TestBot", behavior)

        bot.start()
        assert bot.is_active is True

        bot.stop()
        assert bot.is_active is False

    def test_update_returns_action(self):
        behavior = RandomWalkBehavior()
        bot = GameBot("TestBot", behavior)
        bot.start()

        action = bot.update({}, 0.016)

        assert action is not None

    def test_update_inactive_returns_none(self):
        behavior = RandomWalkBehavior()
        bot = GameBot("TestBot", behavior)

        action = bot.update({}, 0.016)

        assert action is None

    def test_record_kill(self):
        behavior = RandomWalkBehavior()
        bot = GameBot("TestBot", behavior)

        bot.record_kill()
        bot.record_kill()

        metrics = bot.get_metrics()
        assert metrics["enemies_killed"] == 2

    def test_record_death(self):
        behavior = RandomWalkBehavior()
        bot = GameBot("TestBot", behavior)

        bot.record_death()

        metrics = bot.get_metrics()
        assert metrics["deaths"] == 1

    def test_metrics_tracking(self):
        behavior = RandomWalkBehavior()
        bot = GameBot("TestBot", behavior)
        bot.start()

        bot.update({"position": (0, 0, 0)}, 0.016)
        bot.update({"position": (1, 0, 0)}, 0.016)

        metrics = bot.get_metrics()
        assert metrics["actions_count"] == 2
        assert metrics["time_active"] > 0


class TestBotController:
    """Tests for BotController class."""

    def test_create_controller(self):
        controller = BotController()
        assert controller is not None

    def test_create_bot(self):
        controller = BotController()
        bot = controller.create_bot("Bot1", behavior_type="random_walk")

        assert bot.name == "Bot1"

    def test_get_bot(self):
        controller = BotController()
        controller.create_bot("Bot1")

        bot = controller.get_bot("Bot1")
        assert bot is not None

        missing = controller.get_bot("Missing")
        assert missing is None

    def test_remove_bot(self):
        controller = BotController()
        controller.create_bot("Bot1")
        controller.remove_bot("Bot1")

        assert controller.get_bot("Bot1") is None

    def test_start_all(self):
        controller = BotController()
        controller.create_bot("Bot1")
        controller.create_bot("Bot2")

        controller.start_all()

        assert controller.get_bot("Bot1").is_active
        assert controller.get_bot("Bot2").is_active

    def test_stop_all(self):
        controller = BotController()
        controller.create_bot("Bot1")
        controller.create_bot("Bot2")

        controller.start_all()
        controller.stop_all()

        assert not controller.get_bot("Bot1").is_active
        assert not controller.get_bot("Bot2").is_active

    def test_update_all(self):
        controller = BotController()
        controller.create_bot("Bot1")
        controller.create_bot("Bot2")
        controller.start_all()

        actions = controller.update_all({}, 0.016)

        assert "Bot1" in actions
        assert "Bot2" in actions

    def test_get_all_metrics(self):
        controller = BotController()
        controller.create_bot("Bot1")
        controller.create_bot("Bot2")

        metrics = controller.get_all_metrics()

        assert "Bot1" in metrics
        assert "Bot2" in metrics

    def test_register_behavior(self):
        controller = BotController()

        class CustomBehavior(BotBehavior):
            name = "custom"

            def get_next_action(self, bot, world_state):
                return BotAction.wait(1.0)

        controller.register_behavior("custom", CustomBehavior)
        bot = controller.create_bot("Bot1", behavior_type="custom")

        assert isinstance(bot.behavior, CustomBehavior)


class TestPlaytestRecorder:
    """Tests for PlaytestRecorder class."""

    def test_create_recorder(self):
        recorder = PlaytestRecorder()
        assert recorder is not None

    def test_start_stop(self):
        recorder = PlaytestRecorder()
        recorder.start()
        recorder.stop()

    def test_record_event(self):
        recorder = PlaytestRecorder()
        recorder.start()

        recorder.record_event("test_event", "Bot1", data=123)

        events = recorder.get_events()
        assert len(events) == 1
        assert events[0].event_type == "test_event"

    def test_record_action(self):
        recorder = PlaytestRecorder()
        recorder.start()

        action = BotAction.jump()
        recorder.record_action("Bot1", action)

        events = recorder.get_events()
        assert len(events) == 1
        assert events[0].event_type == "action"

    def test_export(self, tmp_path):
        recorder = PlaytestRecorder()
        recorder.start()
        recorder.record_event("test", "Bot1")
        recorder.stop()

        output = tmp_path / "events.json"
        recorder.export(str(output))

        assert output.exists()


class TestPlaytestSession:
    """Tests for PlaytestSession class."""

    def test_create_session(self):
        session = PlaytestSession(
            name="TestSession",
            duration=60.0,
            bot_count=2,
        )

        assert session.name == "TestSession"
        assert session.bot_count == 2

    def test_setup(self):
        session = PlaytestSession(name="Test", bot_count=3)
        session.setup()

        assert session._controller is not None
        assert session._recorder is not None


class TestPlaytestReporter:
    """Tests for PlaytestReporter class."""

    def test_create_reporter(self):
        reporter = PlaytestReporter()
        assert reporter is not None

    def test_add_session(self):
        reporter = PlaytestReporter()
        reporter.add_session({"name": "Test", "duration": 60.0})

        assert len(reporter._sessions) == 1

    def test_generate_report(self):
        reporter = PlaytestReporter()
        reporter.add_session({
            "name": "Test",
            "duration": 60.0,
            "bot_count": 2,
            "event_count": 100,
            "metrics": {},
        })

        report = reporter.generate_report()

        assert "Test" in report
        assert "Duration" in report

    def test_export_json(self, tmp_path):
        reporter = PlaytestReporter()
        reporter.add_session({"name": "Test"})

        output = tmp_path / "report.json"
        reporter.export_json(str(output))

        assert output.exists()


class TestCreateBotFunction:
    """Tests for create_bot convenience function."""

    def test_create_with_string_behavior(self):
        bot = create_bot("TestBot", behavior="random_walk")
        assert bot.name == "TestBot"
        assert isinstance(bot.behavior, RandomWalkBehavior)

    def test_create_with_behavior_instance(self):
        behavior = CombatBehavior()
        bot = create_bot("TestBot", behavior=behavior)
        assert bot.behavior is behavior


class TestRunPlaytestFunction:
    """Tests for run_playtest convenience function."""

    def test_run_short_playtest(self):
        results = run_playtest(
            duration=0.1,
            bot_count=1,
            behavior="random_walk",
        )

        assert "duration" in results
        assert "metrics" in results
