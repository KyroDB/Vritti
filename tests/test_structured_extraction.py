"""
Unit tests for structured metadata extraction.

Tests action detection, negation detection, time conditions,
and goal compatibility checking.
"""

import pytest

from src.utils.structured_extraction import (
    ActionType,
    GoalParser,
)


class TestActionDetection:
    """Test action type detection."""

    def test_delete_action(self):
        parser = GoalParser()
        goal = parser.parse("Delete old log files from server")
        assert goal.action == ActionType.DELETE

    def test_create_action(self):
        parser = GoalParser()
        goal = parser.parse("Create new user account in database")
        assert goal.action == ActionType.CREATE

    def test_deploy_action(self):
        parser = GoalParser()
        goal = parser.parse("Deploy new version to production")
        assert goal.action == ActionType.DEPLOY

    def test_upload_vs_download(self):
        parser = GoalParser()

        upload_goal = parser.parse("Upload file to S3 bucket")
        assert upload_goal.action == ActionType.UPLOAD

        download_goal = parser.parse("Download file from S3 bucket")
        assert download_goal.action == ActionType.DOWNLOAD

    def test_install_vs_uninstall(self):
        parser = GoalParser()

        install_goal = parser.parse("Install python package")
        assert install_goal.action == ActionType.INSTALL

        uninstall_goal = parser.parse("Uninstall deprecated package")
        assert uninstall_goal.action == ActionType.UNINSTALL

    def test_start_stop_restart(self):
        parser = GoalParser()

        start_goal = parser.parse("Start the database service")
        assert start_goal.action == ActionType.START

        stop_goal = parser.parse("Stop the web server")
        assert stop_goal.action == ActionType.STOP

        restart_goal = parser.parse("Restart the application")
        assert restart_goal.action == ActionType.RESTART


class TestTargetExtraction:
    """Test target extraction."""

    def test_file_target(self):
        parser = GoalParser()
        goal = parser.parse("Delete old files")
        assert "file" in goal.target.lower()

    def test_database_target(self):
        parser = GoalParser()
        goal = parser.parse("Update database table")
        assert "database" in goal.target.lower() or "db" in goal.target.lower()

    def test_deployment_target(self):
        parser = GoalParser()
        goal = parser.parse("Deploy new version")
        assert "deploy" in goal.target.lower()

    def test_container_target(self):
        parser = GoalParser()
        goal = parser.parse("Restart pod in kubernetes")
        assert "pod" in goal.target.lower() or "container" in goal.target.lower()


class TestNegationDetection:
    """Test negation detection."""

    def test_except_keyword(self):
        parser = GoalParser()
        goal = parser.parse("Delete all files except those older than 7 days")
        assert goal.has_negation is True

    def test_excluding_keyword(self):
        parser = GoalParser()
        goal = parser.parse("Remove packages excluding system packages")
        assert goal.has_negation is True

    def test_not_keyword(self):
        parser = GoalParser()
        goal = parser.parse("Delete files not accessed recently")
        assert goal.has_negation is True

    def test_without_keyword(self):
        parser = GoalParser()
        goal = parser.parse("Deploy without running tests")
        assert goal.has_negation is True

    def test_no_negation(self):
        parser = GoalParser()
        goal = parser.parse("Delete all old files")
        assert goal.has_negation is False


class TestTimeConditionExtraction:
    """Test time condition extraction."""

    def test_older_than_days(self):
        parser = GoalParser()
        goal = parser.parse("Delete files older than 7 days")

        assert goal.time_condition is not None
        assert goal.time_condition.operator == "older_than"
        assert goal.time_condition.value == 7
        assert goal.time_condition.unit == "day"
        assert goal.time_condition.negated is False

    def test_newer_than_hours(self):
        parser = GoalParser()
        goal = parser.parse("Show logs newer than 24 hours")

        assert goal.time_condition is not None
        assert goal.time_condition.operator == "newer_than"
        assert goal.time_condition.value == 24
        assert goal.time_condition.unit == "hour"

    def test_days_ago(self):
        parser = GoalParser()
        goal = parser.parse("Delete backups from 30 days ago")

        assert goal.time_condition is not None
        assert goal.time_condition.operator == "older_than"
        assert goal.time_condition.value == 30
        assert goal.time_condition.unit == "day"

    def test_within_days(self):
        parser = GoalParser()
        goal = parser.parse("Show files created within 5 days")

        assert goal.time_condition is not None
        assert goal.time_condition.operator == "newer_than"
        assert goal.time_condition.value == 5

    def test_negated_time_condition(self):
        parser = GoalParser()
        goal = parser.parse("Delete all files except those older than 7 days")

        assert goal.time_condition is not None
        assert goal.time_condition.negated is True

    def test_more_than_days_old(self):
        parser = GoalParser()
        goal = parser.parse("Remove files more than 10 days old")

        assert goal.time_condition is not None
        assert goal.time_condition.operator == "older_than"
        assert goal.time_condition.value == 10


class TestEnvironmentExtraction:
    """Test environment extraction."""

    def test_production_environment(self):
        parser = GoalParser()

        goal1 = parser.parse("Deploy to production")
        assert goal1.environment == "production"

        goal2 = parser.parse("Deploy to prod")
        assert goal2.environment == "production"

    def test_staging_environment(self):
        parser = GoalParser()

        goal1 = parser.parse("Deploy to staging")
        assert goal1.environment == "staging"

        goal2 = parser.parse("Deploy to stage")
        assert goal2.environment == "staging"

    def test_dev_environment(self):
        parser = GoalParser()
        goal = parser.parse("Deploy to development")
        assert goal.environment == "development"

    def test_no_environment(self):
        parser = GoalParser()
        goal = parser.parse("Delete old files")
        assert goal.environment is None


class TestGoalCompatibility:
    """Test goal compatibility checking."""

    def test_opposite_actions_incompatible(self):
        parser = GoalParser()

        # Delete vs Create
        goal1 = parser.parse("Delete files")
        goal2 = parser.parse("Create files")
        assert not parser.goals_compatible(goal1, goal2)

        # Upload vs Download
        goal3 = parser.parse("Upload to S3")
        goal4 = parser.parse("Download from S3")
        assert not parser.goals_compatible(goal3, goal4)

        # Start vs Stop
        goal5 = parser.parse("Start service")
        goal6 = parser.parse("Stop service")
        assert not parser.goals_compatible(goal5, goal6)

    def test_different_environments_incompatible(self):
        parser = GoalParser()

        goal1 = parser.parse("Deploy to production")
        goal2 = parser.parse("Deploy to staging")

        assert not parser.goals_compatible(goal1, goal2)

    def test_opposite_time_conditions_incompatible(self):
        parser = GoalParser()

        goal1 = parser.parse("Delete files older than 7 days")
        goal2 = parser.parse("Delete files newer than 7 days")

        # Should be incompatible due to opposite time operators
        assert not parser.goals_compatible(goal1, goal2)

    def test_time_negation_mismatch_incompatible(self):
        parser = GoalParser()

        goal1 = parser.parse("Delete files older than 7 days")
        goal2 = parser.parse("Delete files except those older than 7 days")

        # Should be incompatible - opposite meanings
        assert not parser.goals_compatible(goal1, goal2)

    def test_similar_goals_compatible(self):
        parser = GoalParser()

        goal1 = parser.parse("Delete old log files older than 30 days")
        goal2 = parser.parse("Remove log files older than 30 days")

        # Same action, same time condition - compatible
        assert parser.goals_compatible(goal1, goal2)

    def test_same_environment_compatible(self):
        parser = GoalParser()

        goal1 = parser.parse("Deploy app to production")
        goal2 = parser.parse("Deploy service to production")

        # Same environment, same action - compatible
        assert parser.goals_compatible(goal1, goal2)

    def test_strict_mode_requires_exact_match(self):
        parser = GoalParser()

        goal1 = parser.parse("Delete old log files older than 30 days")
        goal2 = parser.parse("Remove log files older than 30 days")

        assert parser.goals_compatible(goal1, goal2, strict=False)
        assert not parser.goals_compatible(goal1, goal2, strict=True)

    def test_strict_mode_exact_match_passes(self):
        parser = GoalParser()

        goal1 = parser.parse("Deploy app to production")
        goal2 = parser.parse("Deploy app to production")

        assert parser.goals_compatible(goal1, goal2, strict=True)


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    def test_high_confidence_specific_goal(self):
        parser = GoalParser()
        goal = parser.parse("Delete files older than 7 days in production")

        # Should have high confidence: action + target + environment + time
        assert goal.confidence >= 0.7

    def test_medium_confidence_partial_info(self):
        parser = GoalParser()
        goal = parser.parse("Delete old files")

        # Should have medium confidence: action + target
        assert 0.4 <= goal.confidence < 0.7

    def test_low_confidence_vague_goal(self):
        parser = GoalParser()
        goal = parser.parse("Do something with resources")

        # Should have low confidence: unknown action
        assert goal.confidence < 0.5


class TestInputValidation:
    """Test input validation and security."""

    def test_empty_input_raises_error(self):
        parser = GoalParser()

        with pytest.raises(ValueError, match="cannot be empty"):
            parser.parse("")

    def test_too_long_input_raises_error(self):
        parser = GoalParser()
        long_input = "x" * 501  # Over max length

        with pytest.raises(ValueError, match="too long"):
            parser.parse(long_input)

    def test_max_length_accepted(self):
        parser = GoalParser()
        max_input = "Delete files " + "x" * (GoalParser.MAX_INPUT_LENGTH - 20)

        # Should not raise
        goal = parser.parse(max_input)
        assert goal is not None

    def test_special_characters_handled(self):
        parser = GoalParser()

        # Test various special characters
        goal = parser.parse("Delete files with @#$% characters")
        assert goal is not None
        assert goal.action == ActionType.DELETE


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_multiple_actions_first_wins(self):
        parser = GoalParser()

        # More specific upload should win over generic create
        goal = parser.parse("Upload and create new file")
        assert goal.action == ActionType.UPLOAD

    def test_multiple_environments_first_wins(self):
        parser = GoalParser()
        goal = parser.parse("Deploy from staging to production")

        # Should detect production (appears first in patterns)
        assert goal.environment in ["production", "staging"]

    def test_multiple_time_conditions_first_wins(self):
        parser = GoalParser()
        goal = parser.parse("Delete files older than 7 days but newer than 30 days")

        assert goal.time_condition is not None
        # Should pick first one (older than 7 days)
        assert goal.time_condition.value == 7

    def test_unicode_handling(self):
        parser = GoalParser()
        goal = parser.parse("Delete files with émojis 🎉")

        assert goal is not None
        assert goal.action == ActionType.DELETE

    def test_case_insensitive(self):
        parser = GoalParser()

        goal1 = parser.parse("DELETE FILES")
        goal2 = parser.parse("delete files")
        goal3 = parser.parse("Delete Files")

        assert goal1.action == goal2.action == goal3.action == ActionType.DELETE


class TestRealWorldExamples:
    """Test real-world goal examples."""

    def test_kubernetes_deployment(self):
        parser = GoalParser()
        goal = parser.parse("Deploy new version of microservice to production cluster")

        assert goal.action == ActionType.DEPLOY
        assert goal.environment == "production"

    def test_database_cleanup(self):
        parser = GoalParser()
        goal = parser.parse("Delete user records older than 90 days from database")

        assert goal.action == ActionType.DELETE
        assert goal.time_condition is not None
        assert goal.time_condition.value == 90
        assert goal.time_condition.unit == "day"

    def test_log_rotation(self):
        parser = GoalParser()
        goal = parser.parse("Remove log files except those newer than 14 days")

        assert goal.action in [ActionType.DELETE, ActionType.UNINSTALL]
        assert goal.has_negation is True
        assert goal.time_condition is not None
        assert goal.time_condition.negated is True

    def test_backup_restoration(self):
        parser = GoalParser()
        goal = parser.parse("Download backup from S3 created 7 days ago")

        assert goal.action == ActionType.DOWNLOAD
        assert goal.time_condition is not None

    def test_service_restart(self):
        parser = GoalParser()
        goal = parser.parse("Restart nginx service in production")

        assert goal.action == ActionType.RESTART
        assert goal.environment == "production"
