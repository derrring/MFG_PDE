"""
Backward compatibility layer for algorithm reorganization.

This module provides seamless backward compatibility for the old algorithm structure
while users transition to the new paradigm-based organization.
"""

from __future__ import annotations

import warnings
from typing import Any

# Legacy import mapping
_LEGACY_IMPORT_MAP = {
    "hjb_solvers": "numerical.hjb_solvers",
    "fp_solvers": "numerical.fp_solvers",
    "mfg_solvers": "numerical.mfg_solvers",
    "variational_solvers": "optimization.variational_methods",
    "neural_solvers": "neural.physics_informed",
}


def _issue_deprecation_warning(old_path: str, new_path: str) -> None:
    """Issue a deprecation warning for old import paths."""
    warnings.warn(
        f"Importing from '{old_path}' is deprecated and will be removed in v0.3.0. "
        f"Use 'mfg_pde.alg_new.{new_path}' instead. "
        f"See the Algorithm Reorganization Guide for migration details.",
        DeprecationWarning,
        stacklevel=3,
    )


def _import_from_new_structure(module_name: str) -> Any:
    """Import a module from the new structure with appropriate warnings."""
    if module_name not in _LEGACY_IMPORT_MAP:
        raise AttributeError(f"No legacy mapping found for '{module_name}'")

    new_path = _LEGACY_IMPORT_MAP[module_name]
    _issue_deprecation_warning(f"mfg_pde.alg.{module_name}", new_path)

    # Dynamic import from new structure
    import importlib

    try:
        return importlib.import_module(f"mfg_pde.alg_new.{new_path}")
    except ImportError as e:
        raise ImportError(
            f"Failed to import '{new_path}' from new algorithm structure. "
            f"This may indicate an incomplete migration. Original error: {e}"
        ) from e


class LegacyAlgorithmModule:
    """
    Module wrapper that provides backward compatibility for old algorithm imports.

    This allows code like:
        from mfg_pde.alg import hjb_solvers

    to continue working while issuing appropriate deprecation warnings.
    """

    def __getattr__(self, name: str) -> Any:
        """Handle attribute access for legacy imports."""
        if name in _LEGACY_IMPORT_MAP:
            return _import_from_new_structure(name)

        # Check if it's a direct solver import (for backwards compatibility)
        if name.endswith("Solver") or name.startswith("Base"):
            return self._try_import_solver(name)

        raise AttributeError(f"module 'mfg_pde.alg' has no attribute '{name}'")

    def _try_import_solver(self, solver_name: str) -> Any:
        """Try to import a solver from the old structure."""
        import importlib

        # Try importing from the original alg module
        try:
            original_module = importlib.import_module("mfg_pde.alg")
            if hasattr(original_module, solver_name):
                _issue_deprecation_warning(
                    f"mfg_pde.alg.{solver_name}",
                    "appropriate_new_path",  # This would be determined dynamically
                )
                return getattr(original_module, solver_name)
        except ImportError:
            pass

        raise AttributeError(f"Solver '{solver_name}' not found in legacy structure")


# Create an instance to use as the module
legacy_module = LegacyAlgorithmModule()


# Utility functions for smooth migration
def get_migration_guide() -> str:
    """Get a string with migration guidance."""
    return """
Algorithm Reorganization Migration Guide:

Old Structure → New Structure:
- mfg_pde.alg.hjb_solvers → mfg_pde.alg_new.numerical.hjb_solvers
- mfg_pde.alg.fp_solvers → mfg_pde.alg_new.numerical.fp_solvers
- mfg_pde.alg.mfg_solvers → mfg_pde.alg_new.numerical.mfg_solvers
- mfg_pde.alg.variational_solvers → mfg_pde.alg_new.optimization.variational_methods
- mfg_pde.alg.neural_solvers → mfg_pde.alg_new.neural.physics_informed

New Paradigms Available:
- mfg_pde.alg_new.optimization.optimal_transport
- mfg_pde.alg_new.neural.operator_learning
- mfg_pde.alg_new.reinforcement (coming soon)

For detailed migration instructions, see:
docs/development/ALGORITHM_REORGANIZATION_PLAN.md
"""


def check_deprecated_imports() -> list[str]:
    """
    Check the current Python session for deprecated algorithm imports.

    Returns:
        List of deprecated imports found
    """
    import sys

    deprecated = []

    for module_name in sys.modules:
        if module_name.startswith("mfg_pde.alg.") and not module_name.startswith("mfg_pde.alg_new."):
            # Check if it's one of the reorganized modules
            submodule = module_name.replace("mfg_pde.alg.", "")
            if submodule.split(".")[0] in _LEGACY_IMPORT_MAP:
                deprecated.append(module_name)

    return deprecated


__all__ = [
    "legacy_module",
    "get_migration_guide",
    "check_deprecated_imports",
]
