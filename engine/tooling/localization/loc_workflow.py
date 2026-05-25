"""
Localization workflow management.

Provides workflow stages: Extract, Translate, Import, Validate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, Callable
import time


class WorkflowState(Enum):
    """Workflow state."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class WorkflowStep(Enum):
    """Workflow steps in order."""
    EXTRACT = auto()
    TRANSLATE = auto()
    IMPORT = auto()
    VALIDATE = auto()


@dataclass(slots=True)
class TranslationTask:
    """
    A translation task in the workflow.
    """
    id: str
    string_key: str
    source_text: str
    target_language: str
    translated_text: str = ""
    state: WorkflowState = WorkflowState.PENDING
    assigned_to: str = ""
    priority: int = 0
    context: str = ""
    notes: str = ""
    created_at: float = 0.0
    completed_at: float = 0.0

    def complete(self, translation: str) -> None:
        """Mark task as completed with translation."""
        self.translated_text = translation
        self.state = WorkflowState.COMPLETED
        self.completed_at = time.time()


@dataclass(slots=True)
class ValidationError:
    """A validation error in a translation."""
    string_key: str
    language: str
    error_type: str
    message: str
    severity: str = "warning"  # "warning" or "error"
    auto_fixable: bool = False
    suggested_fix: str = ""


@dataclass(slots=True)
class ValidationResult:
    """
    Result of validating translations.
    """
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    strings_checked: int = 0
    strings_passed: int = 0

    def add_error(self, error: ValidationError) -> None:
        """Add a validation error."""
        if error.severity == "error":
            self.errors.append(error)
        else:
            self.warnings.append(error)


class LocalizationWorkflow:
    """
    Manages the localization workflow.

    Coordinates extraction, translation, import, and validation steps.
    """
    __slots__ = (
        "_tasks",
        "_current_step",
        "_state",
        "_target_languages",
        "_validators",
        "_on_step_complete",
        "_extraction_results",
    )

    def __init__(self):
        """Initialize workflow."""
        self._tasks: dict[str, TranslationTask] = {}
        self._current_step = WorkflowStep.EXTRACT
        self._state = WorkflowState.PENDING
        self._target_languages: list[str] = []
        self._validators: list[Callable[[str, str, str], Optional[ValidationError]]] = []
        self._on_step_complete: Optional[Callable[[WorkflowStep], None]] = None
        self._extraction_results: list[tuple[str, str, str]] = []  # (key, text, context)

        # Add default validators
        self._add_default_validators()

    def _add_default_validators(self) -> None:
        """Add default validation rules."""
        # Check for empty translations
        def check_empty(key: str, source: str, target: str) -> Optional[ValidationError]:
            if source and not target:
                return ValidationError(
                    string_key=key,
                    language="",
                    error_type="empty_translation",
                    message="Translation is empty",
                    severity="error",
                )
            return None

        # Check for placeholder consistency
        def check_placeholders(key: str, source: str, target: str) -> Optional[ValidationError]:
            import re
            source_placeholders = set(re.findall(r'\{(\w+)\}', source))
            target_placeholders = set(re.findall(r'\{(\w+)\}', target))

            if source_placeholders != target_placeholders:
                missing = source_placeholders - target_placeholders
                extra = target_placeholders - source_placeholders

                msg_parts = []
                if missing:
                    msg_parts.append(f"Missing: {missing}")
                if extra:
                    msg_parts.append(f"Extra: {extra}")

                return ValidationError(
                    string_key=key,
                    language="",
                    error_type="placeholder_mismatch",
                    message=f"Placeholder mismatch. {', '.join(msg_parts)}",
                    severity="error",
                )
            return None

        # Check for untranslated text
        def check_untranslated(key: str, source: str, target: str) -> Optional[ValidationError]:
            if source == target and len(source) > 3:  # Ignore short strings
                return ValidationError(
                    string_key=key,
                    language="",
                    error_type="possibly_untranslated",
                    message="Translation is identical to source (may be untranslated)",
                    severity="warning",
                )
            return None

        # Check for length limits
        def check_length(key: str, source: str, target: str) -> Optional[ValidationError]:
            # Translations shouldn't be drastically longer
            if len(target) > len(source) * 2 and len(source) > 10:
                return ValidationError(
                    string_key=key,
                    language="",
                    error_type="translation_too_long",
                    message=f"Translation is much longer than source ({len(target)} vs {len(source)} chars)",
                    severity="warning",
                )
            return None

        self._validators.extend([
            check_empty,
            check_placeholders,
            check_untranslated,
            check_length,
        ])

    @property
    def current_step(self) -> WorkflowStep:
        """Get current workflow step."""
        return self._current_step

    @property
    def state(self) -> WorkflowState:
        """Get workflow state."""
        return self._state

    def set_target_languages(self, languages: list[str]) -> None:
        """Set target languages for translation."""
        self._target_languages = languages.copy()

    def get_target_languages(self) -> list[str]:
        """Get target languages."""
        return self._target_languages.copy()

    def add_validator(
        self,
        validator: Callable[[str, str, str], Optional[ValidationError]]
    ) -> None:
        """
        Add a custom validator function.

        Validator takes (key, source_text, target_text) and returns
        ValidationError or None.
        """
        self._validators.append(validator)

    def on_step_complete(self, callback: Callable[[WorkflowStep], None]) -> None:
        """Set callback for step completion."""
        self._on_step_complete = callback

    # Step 1: Extract
    def start_extraction(self) -> None:
        """Start the extraction step."""
        self._current_step = WorkflowStep.EXTRACT
        self._state = WorkflowState.IN_PROGRESS
        self._extraction_results.clear()

    def add_extracted_string(self, key: str, text: str, context: str = "") -> None:
        """Add an extracted string."""
        self._extraction_results.append((key, text, context))

    def complete_extraction(self) -> int:
        """
        Complete extraction step.

        Returns:
            Number of strings extracted
        """
        count = len(self._extraction_results)
        self._state = WorkflowState.COMPLETED

        if self._on_step_complete:
            self._on_step_complete(WorkflowStep.EXTRACT)

        return count

    def get_extraction_results(self) -> list[tuple[str, str, str]]:
        """Get extraction results."""
        return self._extraction_results.copy()

    # Step 2: Translate
    def start_translation(self) -> None:
        """Start the translation step."""
        self._current_step = WorkflowStep.TRANSLATE
        self._state = WorkflowState.IN_PROGRESS

        # Create tasks for each string/language combination
        for key, text, context in self._extraction_results:
            for lang in self._target_languages:
                task_id = f"{key}:{lang}"
                task = TranslationTask(
                    id=task_id,
                    string_key=key,
                    source_text=text,
                    target_language=lang,
                    context=context,
                    created_at=time.time(),
                )
                self._tasks[task_id] = task

    def get_pending_tasks(self, language: Optional[str] = None) -> list[TranslationTask]:
        """Get pending translation tasks."""
        tasks = [t for t in self._tasks.values() if t.state == WorkflowState.PENDING]

        if language:
            tasks = [t for t in tasks if t.target_language == language]

        return sorted(tasks, key=lambda t: -t.priority)

    def get_task(self, task_id: str) -> Optional[TranslationTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def submit_translation(
        self,
        task_id: str,
        translation: str,
        translator: str = ""
    ) -> bool:
        """
        Submit a translation for a task.

        Args:
            task_id: Task ID
            translation: Translated text
            translator: Translator name

        Returns:
            True if submitted
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        task.complete(translation)
        if translator:
            task.assigned_to = translator

        return True

    def get_translation_progress(self) -> dict[str, tuple[int, int]]:
        """
        Get translation progress by language.

        Returns:
            Dict of language -> (completed, total)
        """
        progress: dict[str, tuple[int, int]] = {}

        for lang in self._target_languages:
            tasks = [t for t in self._tasks.values() if t.target_language == lang]
            completed = sum(1 for t in tasks if t.state == WorkflowState.COMPLETED)
            progress[lang] = (completed, len(tasks))

        return progress

    def is_translation_complete(self) -> bool:
        """Check if all translations are complete."""
        return all(t.state == WorkflowState.COMPLETED for t in self._tasks.values())

    def complete_translation(self) -> dict[str, dict[str, str]]:
        """
        Complete translation step.

        Returns:
            Dict of key -> {language -> translation}
        """
        self._state = WorkflowState.COMPLETED

        result: dict[str, dict[str, str]] = {}

        for task in self._tasks.values():
            if task.string_key not in result:
                result[task.string_key] = {}
            result[task.string_key][task.target_language] = task.translated_text

        if self._on_step_complete:
            self._on_step_complete(WorkflowStep.TRANSLATE)

        return result

    # Step 3: Import
    def start_import(self) -> None:
        """Start the import step."""
        self._current_step = WorkflowStep.IMPORT
        self._state = WorkflowState.IN_PROGRESS

    def import_translations(
        self,
        data: dict[str, dict[str, str]]
    ) -> int:
        """
        Import translations from external source.

        Args:
            data: Dict of key -> {language -> translation}

        Returns:
            Number of translations imported
        """
        count = 0

        for key, translations in data.items():
            for lang, text in translations.items():
                task_id = f"{key}:{lang}"
                if task_id in self._tasks:
                    self._tasks[task_id].translated_text = text
                    self._tasks[task_id].state = WorkflowState.COMPLETED
                    count += 1
                else:
                    # Create new task for imported translation
                    task = TranslationTask(
                        id=task_id,
                        string_key=key,
                        source_text="",  # Unknown source
                        target_language=lang,
                        translated_text=text,
                        state=WorkflowState.COMPLETED,
                        created_at=time.time(),
                        completed_at=time.time(),
                    )
                    self._tasks[task_id] = task
                    count += 1

        return count

    def complete_import(self) -> None:
        """Complete import step."""
        self._state = WorkflowState.COMPLETED

        if self._on_step_complete:
            self._on_step_complete(WorkflowStep.IMPORT)

    # Step 4: Validate
    def start_validation(self) -> None:
        """Start the validation step."""
        self._current_step = WorkflowStep.VALIDATE
        self._state = WorkflowState.IN_PROGRESS

    def validate_all(self) -> ValidationResult:
        """
        Validate all translations.

        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)

        # Group tasks by key to get source text
        source_texts: dict[str, str] = {}
        for key, text, _ in self._extraction_results:
            source_texts[key] = text

        for task in self._tasks.values():
            if task.state != WorkflowState.COMPLETED:
                continue

            result.strings_checked += 1
            source = source_texts.get(task.string_key, task.source_text)
            has_error = False

            for validator in self._validators:
                error = validator(task.string_key, source, task.translated_text)
                if error:
                    error.language = task.target_language
                    result.add_error(error)

                    if error.severity == "error":
                        has_error = True

            if not has_error:
                result.strings_passed += 1

        result.is_valid = len(result.errors) == 0
        return result

    def validate_single(
        self,
        key: str,
        source_text: str,
        target_text: str,
        language: str
    ) -> list[ValidationError]:
        """
        Validate a single translation.

        Returns:
            List of validation errors
        """
        errors = []

        for validator in self._validators:
            error = validator(key, source_text, target_text)
            if error:
                error.language = language
                errors.append(error)

        return errors

    def complete_validation(self) -> None:
        """Complete validation step."""
        self._state = WorkflowState.COMPLETED

        if self._on_step_complete:
            self._on_step_complete(WorkflowStep.VALIDATE)

    # Utility methods
    def reset(self) -> None:
        """Reset workflow to initial state."""
        self._tasks.clear()
        self._extraction_results.clear()
        self._current_step = WorkflowStep.EXTRACT
        self._state = WorkflowState.PENDING

    def get_statistics(self) -> dict[str, Any]:
        """Get workflow statistics."""
        total_tasks = len(self._tasks)
        completed = sum(1 for t in self._tasks.values() if t.state == WorkflowState.COMPLETED)
        pending = sum(1 for t in self._tasks.values() if t.state == WorkflowState.PENDING)

        return {
            "current_step": self._current_step.name,
            "state": self._state.name,
            "total_strings": len(self._extraction_results),
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "pending_tasks": pending,
            "target_languages": self._target_languages,
            "progress_percent": (completed / total_tasks * 100) if total_tasks > 0 else 0,
        }

    def export_for_external_translation(self, language: str) -> dict[str, Any]:
        """
        Export strings for external translation tool.

        Args:
            language: Target language

        Returns:
            Export data for translation
        """
        strings = []

        for key, text, context in self._extraction_results:
            task_id = f"{key}:{language}"
            task = self._tasks.get(task_id)

            strings.append({
                "key": key,
                "source": text,
                "target": task.translated_text if task else "",
                "context": context,
                "status": task.state.name if task else "PENDING",
            })

        return {
            "language": language,
            "strings": strings,
            "count": len(strings),
        }
