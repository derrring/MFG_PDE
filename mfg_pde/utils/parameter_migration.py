"""
Parameter Migration Utilities for MFG_PDE Package

This module provides utilities for migrating legacy parameter names to modern
equivalents while maintaining backward compatibility. It supports gradual
migration with clear deprecation warnings and automatic parameter translation.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ParameterMapping:
    """Mapping between legacy and modern parameter names."""

    old_name: str
    new_name: str
    deprecation_version: str
    removal_version: str
    transformation: Callable | None = None  # Optional value transformation
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = f"Renamed '{self.old_name}' to '{self.new_name}'"


@dataclass
class MigrationStats:
    """Statistics about parameter migration process."""

    total_parameters: int = 0
    migrated_parameters: int = 0
    warnings_issued: int = 0
    errors_encountered: int = 0
    migration_log: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ParameterMigrator:
    """Central system for managing parameter migrations across the codebase."""

    def __init__(self):
        self.mappings: list[ParameterMapping] = []
        self.migration_stats = MigrationStats()
        self._register_standard_mappings()

    def _register_standard_mappings(self) -> None:
        """Register all standard parameter mappings for MFG_PDE."""

        # Newton solver parameters
        self.add_mapping(
            old_name="NiterNewton",
            new_name="max_newton_iterations",
            deprecation_version="1.3.0",
            removal_version="2.0.0",
            description="Maximum number of Newton iterations",
        )

        self.add_mapping(
            old_name="l2errBoundNewton",
            new_name="newton_tolerance",
            deprecation_version="1.3.0",
            removal_version="2.0.0",
            description="Newton convergence tolerance",
        )

        # Picard iteration parameters
        self.add_mapping(
            old_name="Niter_max",
            new_name="max_picard_iterations",
            deprecation_version="1.3.0",
            removal_version="2.0.0",
            description="Maximum number of Picard iterations",
        )

        self.add_mapping(
            old_name="l2errBoundPicard",
            new_name="picard_tolerance",
            deprecation_version="1.3.0",
            removal_version="2.0.0",
            description="Picard convergence tolerance",
        )

        # Problem parameters
        self.add_mapping(
            old_name="coefCT",
            new_name="coupling_coefficient",
            deprecation_version="1.4.0",
            removal_version="2.0.0",
            description="Coupling strength between agents",
        )

        # Solver configuration parameters
        self.add_mapping(
            old_name="verbose_NewtonSolver",
            new_name="newton_verbose",
            deprecation_version="1.3.0",
            removal_version="2.0.0",
            description="Newton solver verbosity flag",
        )

        self.add_mapping(
            old_name="damping_NewtonSolver",
            new_name="newton_damping_factor",
            deprecation_version="1.3.0",
            removal_version="2.0.0",
            description="Newton solver damping factor",
        )

        # GFDM specific parameters
        self.add_mapping(
            old_name="taylorOrder",
            new_name="taylor_order",
            deprecation_version="1.4.0",
            removal_version="2.0.0",
            description="Taylor expansion order for GFDM",
        )

        # Boolean parameter transformations
        self.add_mapping(
            old_name="returnExtraInfo",
            new_name="return_structured",
            deprecation_version="1.3.0",
            removal_version="2.0.0",
            transformation=lambda x: x if isinstance(x, bool) else bool(x),
            description="Whether to return structured result objects",
        )

    def add_mapping(
        self,
        old_name: str,
        new_name: str,
        deprecation_version: str,
        removal_version: str,
        transformation: Callable | None = None,
        description: str = "",
    ) -> None:
        """
        Add a parameter mapping to the migration system.

        Args:
            old_name: Legacy parameter name
            new_name: Modern parameter name
            deprecation_version: Version when parameter was deprecated
            removal_version: Version when parameter will be removed
            transformation: Optional function to transform parameter value
            description: Description of the parameter purpose
        """
        mapping = ParameterMapping(
            old_name=old_name,
            new_name=new_name,
            deprecation_version=deprecation_version,
            removal_version=removal_version,
            transformation=transformation,
            description=description,
        )
        self.mappings.append(mapping)

    def migrate_parameters(self, kwargs: dict[str, Any], calling_function: str | None = None) -> dict[str, Any]:
        """
        Migrate legacy parameter names to modern equivalents.

        Args:
            kwargs: Dictionary of parameters to migrate
            calling_function: Name of calling function for better error messages

        Returns:
            Dictionary with migrated parameter names
        """
        if not kwargs:
            return kwargs

        migrated = kwargs.copy()
        self.migration_stats.total_parameters = len(kwargs)

        # Get calling function name if not provided
        if calling_function is None:
            frame = inspect.currentframe().f_back
            calling_function = frame.f_code.co_name if frame else "unknown"

        for mapping in self.mappings:
            if mapping.old_name in migrated:
                self._process_mapping(migrated, mapping, calling_function)

        return migrated

    def _process_mapping(self, kwargs: dict[str, Any], mapping: ParameterMapping, calling_function: str) -> None:
        """Process a single parameter mapping."""
        try:
            old_value = kwargs[mapping.old_name]

            # Issue deprecation warning
            warning_msg = (
                f"Parameter '{mapping.old_name}' is deprecated since v{mapping.deprecation_version}. "
                f"Use '{mapping.new_name}' instead. "
                f"Will be removed in v{mapping.removal_version}. "
                f"Called from: {calling_function}"
            )

            warnings.warn(warning_msg, DeprecationWarning, stacklevel=4)
            self.migration_stats.warnings_issued += 1
            self.migration_stats.migration_log.append(
                f"DEPRECATED: {mapping.old_name} → {mapping.new_name} in {calling_function}"
            )

            # Transform value if transformation function provided
            if mapping.transformation:
                try:
                    new_value = mapping.transformation(old_value)
                except Exception as e:
                    self.migration_stats.errors_encountered += 1
                    self.migration_stats.migration_log.append(
                        f"ERROR: Failed to transform {mapping.old_name}={old_value}: {e}"
                    )
                    new_value = old_value
            else:
                new_value = old_value

            # Migrate if new name not already specified
            if mapping.new_name not in kwargs:
                kwargs[mapping.new_name] = new_value
                self.migration_stats.migrated_parameters += 1
                self.migration_stats.migration_log.append(
                    f"MIGRATED: {mapping.old_name}={old_value} → {mapping.new_name}={new_value}"
                )
            else:
                # Warn if both old and new parameters specified
                warning_msg = (
                    f"Both '{mapping.old_name}' and '{mapping.new_name}' specified. "
                    f"Using '{mapping.new_name}' value and ignoring '{mapping.old_name}'"
                )
                warnings.warn(warning_msg, UserWarning, stacklevel=4)
                self.migration_stats.migration_log.append(
                    f"CONFLICT: Both {mapping.old_name} and {mapping.new_name} specified"
                )

            # Remove old parameter
            del kwargs[mapping.old_name]

        except Exception as e:
            self.migration_stats.errors_encountered += 1
            self.migration_stats.migration_log.append(f"ERROR: Failed to process mapping {mapping.old_name}: {e}")

    def get_migration_report(self) -> str:
        """
        Generate a comprehensive migration report.

        Returns:
            Formatted report string
        """
        stats = self.migration_stats

        report = f"""
Parameter Migration Report
=========================
Timestamp: {stats.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Total Parameters: {stats.total_parameters}
Migrated Parameters: {stats.migrated_parameters}
Warnings Issued: {stats.warnings_issued}
Errors Encountered: {stats.errors_encountered}

Migration Log:
"""

        for log_entry in stats.migration_log[-10:]:  # Show last 10 entries
            report += f"  {log_entry}\n"

        if len(stats.migration_log) > 10:
            report += f"  ... and {len(stats.migration_log) - 10} more entries\n"

        return report

    def get_deprecated_parameters(self) -> set[str]:
        """Get set of all deprecated parameter names."""
        return {mapping.old_name for mapping in self.mappings}

    def get_modern_parameters(self) -> set[str]:
        """Get set of all modern parameter names."""
        return {mapping.new_name for mapping in self.mappings}

    def check_parameters(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        """
        Check parameters for deprecated names without migrating.

        Args:
            kwargs: Parameters to check

        Returns:
            Tuple of (deprecated_found, modern_equivalents)
        """
        deprecated_found = []
        modern_equivalents = []

        deprecated_params = self.get_deprecated_parameters()

        for param_name in kwargs:
            if param_name in deprecated_params:
                deprecated_found.append(param_name)
                # Find the modern equivalent
                for mapping in self.mappings:
                    if mapping.old_name == param_name:
                        modern_equivalents.append(mapping.new_name)
                        break

        return deprecated_found, modern_equivalents


# Decorator for automatic parameter migration
def migrate_parameters(migrator: ParameterMigrator | None = None):
    """
    Decorator to automatically migrate parameters in function calls.

    Args:
        migrator: ParameterMigrator instance (uses global if None)

    Example:
        @migrate_parameters()
        def create_solver(**kwargs):
            # kwargs automatically migrated
            pass
    """
    if migrator is None:
        migrator = global_parameter_migrator

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Migrate parameters
            migrated_kwargs = migrator.migrate_parameters(kwargs, func.__name__)

            # Call original function with migrated parameters
            return func(*args, **migrated_kwargs)

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__

        return wrapper

    return decorator


# Global migrator instance
global_parameter_migrator = ParameterMigrator()


# Convenience functions
def migrate_kwargs(kwargs: dict[str, Any], calling_function: str | None = None) -> dict[str, Any]:
    """Convenience function to migrate parameters using global migrator."""
    return global_parameter_migrator.migrate_parameters(kwargs, calling_function)


def check_deprecated_usage(kwargs: dict[str, Any]) -> None:
    """Check for deprecated parameter usage and print warnings."""
    deprecated, modern = global_parameter_migrator.check_parameters(kwargs)

    if deprecated:
        print("\nWARNING: Deprecated parameters detected:")
        for old, new in zip(deprecated, modern, strict=False):
            print(f"   '{old}' → use '{new}' instead")
        print("   Consider updating your code to use modern parameter names.\n")


def get_parameter_migration_guide() -> str:
    """Generate a user-friendly parameter migration guide."""
    migrator = global_parameter_migrator

    guide = """
MFG_PDE Parameter Migration Guide
================================

The following parameter names have been modernized for clarity and consistency:

"""

    # Group by category
    categories = {
        "Newton Solver": [
            "NiterNewton",
            "l2errBoundNewton",
            "verbose_NewtonSolver",
            "damping_NewtonSolver",
        ],
        "Picard Iteration": ["Niter_max", "l2errBoundPicard"],
        "Problem Definition": ["coefCT"],
        "GFDM Method": ["taylorOrder"],
        "Return Options": ["returnExtraInfo"],
    }

    for category, old_names in categories.items():
        guide += f"\n{category}:\n"
        guide += "=" * len(category) + "\n"

        for mapping in migrator.mappings:
            if mapping.old_name in old_names:
                guide += f"  {mapping.old_name} → {mapping.new_name}\n"
                guide += f"    {mapping.description}\n"
                guide += f"    Deprecated: v{mapping.deprecation_version}, Removed: v{mapping.removal_version}\n\n"

    guide += """
Migration Strategy:
==================
1. Replace deprecated parameters with modern equivalents
2. Update any configuration files or scripts
3. Test functionality with new parameter names
4. Remove deprecated parameter usage before v2.0.0

For automatic migration, use the @migrate_parameters decorator:

    from mfg_pde.utils.parameter_migration import migrate_parameters

    @migrate_parameters()
    def your_function(**kwargs):
        # Parameters automatically migrated
        pass
"""

    return guide
