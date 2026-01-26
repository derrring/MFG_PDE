"""
Validation protocols and shared types.

This module defines the core types used across all validation modules:
- ValidationResult: Outcome of a validation check
- ValidationError: Exception for validation failures
- Validator protocol: Interface for custom validators

Issue #689: Validation module infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from mfg_pde.geometry.protocol import GeometryProtocol


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Must fix before proceeding
    WARNING = "warning"  # Proceed with caution
    INFO = "info"  # Informational only


@dataclass
class ValidationIssue:
    """A single validation issue found during checking."""

    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    location: str | None = None  # e.g., "m_initial", "diffusion[5,:]"
    suggestion: str | None = None  # How to fix

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}]"]
        if self.location:
            parts.append(f"({self.location})")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """
    Result of a validation check.

    Attributes:
        is_valid: True if validation passed (no errors)
        issues: List of validation issues found
        context: Additional context about the validation

    Usage:
        result = validate_m_initial(m_initial, geometry)
        if not result.is_valid:
            raise ValidationError(result)

        # Or collect all issues
        results = [
            validate_m_initial(...),
            validate_u_final(...),
        ]
        combined = ValidationResult.combine(results)
    """

    is_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    context: dict[str, object] = field(default_factory=dict)

    def add_error(
        self,
        message: str,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Add an error issue and mark result as invalid."""
        self.issues.append(
            ValidationIssue(
                message=message,
                severity=ValidationSeverity.ERROR,
                location=location,
                suggestion=suggestion,
            )
        )
        self.is_valid = False

    def add_warning(
        self,
        message: str,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Add a warning issue (does not invalidate result)."""
        self.issues.append(
            ValidationIssue(
                message=message,
                severity=ValidationSeverity.WARNING,
                location=location,
                suggestion=suggestion,
            )
        )

    def add_info(self, message: str, location: str | None = None) -> None:
        """Add an informational issue."""
        self.issues.append(
            ValidationIssue(
                message=message,
                severity=ValidationSeverity.INFO,
                location=location,
            )
        )

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @classmethod
    def combine(cls, results: list[ValidationResult]) -> ValidationResult:
        """Combine multiple validation results into one."""
        combined = cls()
        for result in results:
            combined.issues.extend(result.issues)
            combined.context.update(result.context)
            if not result.is_valid:
                combined.is_valid = False
        return combined

    @classmethod
    def ok(cls, context: dict[str, object] | None = None) -> ValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True, context=context or {})

    @classmethod
    def fail(
        cls,
        message: str,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> ValidationResult:
        """Create a failed validation result with a single error."""
        result = cls(is_valid=False)
        result.add_error(message, location, suggestion)
        return result

    def __str__(self) -> str:
        if self.is_valid:
            return "Validation passed"
        return "\n".join(str(issue) for issue in self.errors)


class ValidationError(Exception):
    """
    Exception raised when validation fails.

    Attributes:
        result: The ValidationResult that triggered this error
        message: Human-readable error message
    """

    def __init__(self, result: ValidationResult, message: str | None = None):
        self.result = result
        if message is None:
            message = str(result)
        super().__init__(message)

    @classmethod
    def from_message(
        cls,
        message: str,
        location: str | None = None,
        suggestion: str | None = None,
    ) -> ValidationError:
        """Create a ValidationError from a simple message."""
        result = ValidationResult.fail(message, location, suggestion)
        return cls(result, message)


@runtime_checkable
class Validator(Protocol):
    """
    Protocol for validator objects.

    Validators perform a specific validation check and return a ValidationResult.
    """

    def validate(self) -> ValidationResult:
        """Perform validation and return result."""
        ...


# Type aliases for common validation signatures
# These are for documentation/type hints only
if TYPE_CHECKING:
    ValidatorFunc = Callable[..., ValidationResult]
    ArrayValidator = Callable[["np.ndarray", str], ValidationResult]
    CallableValidator = Callable[[Callable, "GeometryProtocol"], ValidationResult]
