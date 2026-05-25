"""Tests for localization workflow."""

import pytest
from engine.tooling.localization.loc_workflow import (
    WorkflowState,
    WorkflowStep,
    TranslationTask,
    ValidationError,
    ValidationResult,
    LocalizationWorkflow,
)


class TestTranslationTask:
    """Tests for translation task."""

    def test_creation(self):
        """Test task creation."""
        task = TranslationTask(
            id="test:fr",
            string_key="test",
            source_text="Hello",
            target_language="fr",
        )
        assert task.id == "test:fr"
        assert task.state == WorkflowState.PENDING

    def test_complete(self):
        """Test completing task."""
        task = TranslationTask(
            id="test:fr",
            string_key="test",
            source_text="Hello",
            target_language="fr",
        )
        task.complete("Bonjour")
        assert task.state == WorkflowState.COMPLETED
        assert task.translated_text == "Bonjour"
        assert task.completed_at > 0


class TestValidationResult:
    """Tests for validation result."""

    def test_creation(self):
        """Test result creation."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult(is_valid=True)
        error = ValidationError(
            string_key="test",
            language="fr",
            error_type="empty",
            message="Empty translation",
            severity="error",
        )
        result.add_error(error)
        assert len(result.errors) == 1

    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult(is_valid=True)
        warning = ValidationError(
            string_key="test",
            language="fr",
            error_type="suspicious",
            message="Might be wrong",
            severity="warning",
        )
        result.add_error(warning)
        assert len(result.warnings) == 1


class TestLocalizationWorkflow:
    """Tests for localization workflow."""

    def setup_method(self):
        """Set up test workflow."""
        self.workflow = LocalizationWorkflow()

    def test_creation(self):
        """Test workflow creation."""
        assert self.workflow.current_step == WorkflowStep.EXTRACT
        assert self.workflow.state == WorkflowState.PENDING

    def test_set_target_languages(self):
        """Test setting target languages."""
        self.workflow.set_target_languages(["fr", "de", "es"])
        langs = self.workflow.get_target_languages()
        assert "fr" in langs
        assert "de" in langs
        assert "es" in langs

    # Extraction step tests
    def test_start_extraction(self):
        """Test starting extraction."""
        self.workflow.start_extraction()
        assert self.workflow.current_step == WorkflowStep.EXTRACT
        assert self.workflow.state == WorkflowState.IN_PROGRESS

    def test_add_extracted_string(self):
        """Test adding extracted strings."""
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "greeting")
        results = self.workflow.get_extraction_results()
        assert len(results) == 1
        assert results[0][0] == "hello"

    def test_complete_extraction(self):
        """Test completing extraction."""
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "")
        count = self.workflow.complete_extraction()
        assert count == 1
        assert self.workflow.state == WorkflowState.COMPLETED

    # Translation step tests
    def test_start_translation(self):
        """Test starting translation."""
        self.workflow.set_target_languages(["fr"])
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "")
        self.workflow.complete_extraction()

        self.workflow.start_translation()
        assert self.workflow.current_step == WorkflowStep.TRANSLATE

    def test_get_pending_tasks(self):
        """Test getting pending tasks."""
        self.workflow.set_target_languages(["fr", "de"])
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "")
        self.workflow.complete_extraction()
        self.workflow.start_translation()

        pending = self.workflow.get_pending_tasks()
        assert len(pending) == 2

    def test_get_pending_tasks_by_language(self):
        """Test getting pending tasks filtered by language."""
        self.workflow.set_target_languages(["fr", "de"])
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "")
        self.workflow.complete_extraction()
        self.workflow.start_translation()

        pending = self.workflow.get_pending_tasks("fr")
        assert len(pending) == 1
        assert pending[0].target_language == "fr"

    def test_submit_translation(self):
        """Test submitting translation."""
        self.workflow.set_target_languages(["fr"])
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "")
        self.workflow.complete_extraction()
        self.workflow.start_translation()

        task = self.workflow.get_pending_tasks()[0]
        result = self.workflow.submit_translation(task.id, "Bonjour")
        assert result
        assert task.state == WorkflowState.COMPLETED

    def test_translation_progress(self):
        """Test translation progress."""
        self.workflow.set_target_languages(["fr", "de"])
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "")
        self.workflow.add_extracted_string("bye", "Goodbye", "")
        self.workflow.complete_extraction()
        self.workflow.start_translation()

        progress = self.workflow.get_translation_progress()
        assert "fr" in progress
        assert "de" in progress
        assert progress["fr"] == (0, 2)

    def test_is_translation_complete(self):
        """Test translation completion check."""
        self.workflow.set_target_languages(["fr"])
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "")
        self.workflow.complete_extraction()
        self.workflow.start_translation()

        assert not self.workflow.is_translation_complete()

        task = self.workflow.get_pending_tasks()[0]
        self.workflow.submit_translation(task.id, "Bonjour")

        assert self.workflow.is_translation_complete()

    # Import step tests
    def test_start_import(self):
        """Test starting import."""
        self.workflow.start_import()
        assert self.workflow.current_step == WorkflowStep.IMPORT

    def test_import_translations(self):
        """Test importing translations."""
        self.workflow.set_target_languages(["fr"])
        self.workflow.start_import()

        data = {"hello": {"fr": "Bonjour"}}
        count = self.workflow.import_translations(data)
        assert count == 1

    # Validation step tests
    def test_start_validation(self):
        """Test starting validation."""
        self.workflow.start_validation()
        assert self.workflow.current_step == WorkflowStep.VALIDATE

    def test_validate_empty_translation(self):
        """Test validating empty translation."""
        errors = self.workflow.validate_single(
            "test", "Hello", "", "fr"
        )
        assert len(errors) > 0
        assert any(e.error_type == "empty_translation" for e in errors)

    def test_validate_placeholder_mismatch(self):
        """Test validating placeholder mismatch."""
        errors = self.workflow.validate_single(
            "test", "Hello {name}", "Bonjour", "fr"
        )
        assert len(errors) > 0
        assert any(e.error_type == "placeholder_mismatch" for e in errors)

    def test_validate_possibly_untranslated(self):
        """Test validating possibly untranslated."""
        errors = self.workflow.validate_single(
            "test", "Hello World", "Hello World", "fr"
        )
        assert len(errors) > 0
        assert any(e.error_type == "possibly_untranslated" for e in errors)

    def test_validate_all(self):
        """Test validating all translations."""
        self.workflow.set_target_languages(["fr"])
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "")
        self.workflow.complete_extraction()
        self.workflow.start_translation()

        task = self.workflow.get_pending_tasks()[0]
        self.workflow.submit_translation(task.id, "Bonjour")

        self.workflow.start_validation()
        result = self.workflow.validate_all()
        assert result.strings_checked == 1
        assert result.is_valid

    # Utility tests
    def test_reset(self):
        """Test resetting workflow."""
        self.workflow.set_target_languages(["fr"])
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "")

        self.workflow.reset()
        assert self.workflow.state == WorkflowState.PENDING
        assert len(self.workflow.get_extraction_results()) == 0

    def test_get_statistics(self):
        """Test getting statistics."""
        self.workflow.set_target_languages(["fr"])
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "")
        self.workflow.complete_extraction()
        self.workflow.start_translation()

        stats = self.workflow.get_statistics()
        assert stats["total_strings"] == 1
        assert stats["total_tasks"] == 1

    def test_export_for_external_translation(self):
        """Test exporting for external translation."""
        self.workflow.set_target_languages(["fr"])
        self.workflow.start_extraction()
        self.workflow.add_extracted_string("hello", "Hello", "greeting")
        self.workflow.complete_extraction()
        self.workflow.start_translation()

        export = self.workflow.export_for_external_translation("fr")
        assert export["language"] == "fr"
        assert len(export["strings"]) == 1

    def test_step_callback(self):
        """Test step completion callback."""
        completed_steps = []

        def on_complete(step):
            completed_steps.append(step)

        self.workflow.on_step_complete(on_complete)
        self.workflow.start_extraction()
        self.workflow.complete_extraction()

        assert WorkflowStep.EXTRACT in completed_steps

    def test_add_custom_validator(self):
        """Test adding custom validator."""
        def custom_validator(key, source, target):
            if "forbidden" in target.lower():
                return ValidationError(
                    string_key=key,
                    language="",
                    error_type="forbidden_word",
                    message="Contains forbidden word",
                    severity="error",
                )
            return None

        self.workflow.add_validator(custom_validator)

        errors = self.workflow.validate_single(
            "test", "Hello", "Forbidden content", "fr"
        )
        assert any(e.error_type == "forbidden_word" for e in errors)
