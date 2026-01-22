#!/usr/bin/env python3
"""
Migration script: Replace direct `import logging` with `mfg_logging`.

Issue #620 Phase 1.2: Migrate 36 files from direct logging to mfg_logging.

Migration pattern:
  Before:
    import logging
    logger = logging.getLogger(__name__)

  After:
    from mfg_pde.utils.mfg_logging import get_logger
    logger = get_logger(__name__)

Files using logging.DEBUG, logging.INFO etc. will keep both imports.

Author: Claude Code
Date: 2026-01-20
"""

from __future__ import annotations

import re
from pathlib import Path

# Files to migrate (excluding mfg_logging/logger.py which needs direct import)
FILES_TO_MIGRATE = [
    "mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py",
    "mfg_pde/alg/numerical/hjb_solvers/hjb_sl_interpolation.py",
    "mfg_pde/alg/numerical/hjb_solvers/base_hjb.py",
    "mfg_pde/alg/numerical/fp_solvers/fp_semi_lagrangian_adjoint.py",
    "mfg_pde/alg/numerical/fp_solvers/fp_semi_lagrangian.py",
    "mfg_pde/alg/numerical/coupling/mfg_residual.py",
    "mfg_pde/alg/numerical/coupling/newton_mfg_solver.py",
    "mfg_pde/alg/numerical/coupling/block_iterators.py",
    "mfg_pde/alg/neural/dgm/base_dgm.py",
    "mfg_pde/alg/neural/dgm/mfg_dgm_solver.py",
    "mfg_pde/alg/neural/dgm/sampling.py",
    "mfg_pde/alg/neural/dgm/variance_reduction.py",
    "mfg_pde/alg/neural/pinn_solvers/adaptive_training.py",
    "mfg_pde/alg/optimization/variational_solvers/variational_mfg_solver.py",
    "mfg_pde/alg/optimization/variational_solvers/primal_dual_solver.py",
    "mfg_pde/alg/optimization/variational_solvers/base_variational.py",
    "mfg_pde/alg/optimization/optimal_transport/wasserstein_solver.py",
    "mfg_pde/alg/optimization/optimal_transport/sinkhorn_solver.py",
    "mfg_pde/alg/reinforcement/core/base_mfrl.py",
    "mfg_pde/backends/__init__.py",
    "mfg_pde/config/omegaconf_manager.py",
    "mfg_pde/core/plugin_system.py",
    "mfg_pde/geometry/collocation.py",
    "mfg_pde/geometry/meshes/mesh_manager.py",
    "mfg_pde/solvers/variational.py",
    "mfg_pde/utils/acceleration/__init__.py",
    "mfg_pde/utils/convergence/convergence_monitors.py",
    "mfg_pde/utils/data/polars_integration.py",
    "mfg_pde/utils/numerical/nonlinear_solvers.py",
    "mfg_pde/utils/numerical/particle/sampling.py",
    "mfg_pde/utils/numerical/particle/mcmc.py",
    "mfg_pde/visualization/interactive_plots.py",
    "mfg_pde/visualization/mfg_analytics.py",
    # Complex files (workflow) - will review manually after
    "mfg_pde/workflow/workflow_manager.py",
    "mfg_pde/workflow/experiment_tracker.py",
    "mfg_pde/workflow/parameter_sweep.py",
]

# Patterns for logging constants that require keeping the import
LOGGING_CONSTANTS = re.compile(r"\blogging\.(DEBUG|INFO|WARNING|ERROR|CRITICAL|NOTSET)\b")
LOGGING_HANDLER = re.compile(r"\blogging\.(Handler|StreamHandler|FileHandler|NullHandler)\b")
LOGGING_OTHER = re.compile(r"\blogging\.(?!getLogger\b)\w+")


def needs_logging_import(content: str) -> bool:
    """Check if file uses logging module beyond getLogger."""
    # Remove the import line and logger creation to check remaining usage
    content_without_import = re.sub(r"^import logging\s*$", "", content, flags=re.MULTILINE)
    content_without_logger = re.sub(
        r"^logger\s*=\s*logging\.getLogger\(__name__\)\s*$", "", content_without_import, flags=re.MULTILINE
    )

    # Check for any remaining logging.X usage
    return bool(LOGGING_OTHER.search(content_without_logger))


def migrate_file(file_path: Path, dry_run: bool = False) -> dict:
    """
    Migrate a single file from import logging to mfg_logging.

    Returns dict with migration status and details.
    """
    result = {
        "path": str(file_path),
        "status": "unchanged",
        "needs_logging_import": False,
        "changes": [],
    }

    content = file_path.read_text()
    original_content = content

    # Check if file uses logging constants/handlers
    result["needs_logging_import"] = needs_logging_import(content)

    # Pattern 1: Replace `import logging` with mfg_logging import
    # If file still needs logging, keep it and add mfg_logging
    if result["needs_logging_import"]:
        # Keep import logging, add mfg_logging after
        new_import = "import logging\nfrom mfg_pde.utils.mfg_logging import get_logger"
        if "import logging\n" in content:
            content = content.replace("import logging\n", new_import + "\n", 1)
            result["changes"].append("Added mfg_logging import (kept logging for constants)")
    else:
        # Replace import logging entirely
        content = re.sub(
            r"^import logging\s*$",
            "from mfg_pde.utils.mfg_logging import get_logger",
            content,
            count=1,
            flags=re.MULTILINE,
        )
        if content != original_content:
            result["changes"].append("Replaced import logging with mfg_logging")

    # Pattern 2: Replace `logger = logging.getLogger(__name__)` with `logger = get_logger(__name__)`
    content = re.sub(
        r"^(\s*)logger\s*=\s*logging\.getLogger\(__name__\)\s*$",
        r"\1logger = get_logger(__name__)",
        content,
        flags=re.MULTILINE,
    )
    if "logger = get_logger(__name__)" in content and "logging.getLogger" not in content:
        result["changes"].append("Replaced logging.getLogger with get_logger")

    # Check if changes were made
    if content != original_content:
        result["status"] = "migrated"
        if not dry_run:
            file_path.write_text(content)

    return result


def main():
    """Run migration on all target files."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate logging imports to mfg_logging")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    print("=" * 70)
    print("LOGGING IMPORT MIGRATION")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Dry run: {args.dry_run}")
    print(f"Files to migrate: {len(FILES_TO_MIGRATE)}")
    print()

    migrated = 0
    unchanged = 0
    errors = 0
    needs_logging = []

    for rel_path in FILES_TO_MIGRATE:
        file_path = project_root / rel_path

        if not file_path.exists():
            print(f"  SKIP: {rel_path} (file not found)")
            errors += 1
            continue

        try:
            result = migrate_file(file_path, dry_run=args.dry_run)

            if result["status"] == "migrated":
                migrated += 1
                status = "MIGRATED" if not args.dry_run else "WOULD MIGRATE"
                print(f"  {status}: {rel_path}")
                if args.verbose:
                    for change in result["changes"]:
                        print(f"    - {change}")
                if result["needs_logging_import"]:
                    needs_logging.append(rel_path)
            else:
                unchanged += 1
                if args.verbose:
                    print(f"  UNCHANGED: {rel_path}")
        except Exception as e:
            print(f"  ERROR: {rel_path} - {e}")
            errors += 1

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Migrated: {migrated}")
    print(f"  Unchanged: {unchanged}")
    print(f"  Errors: {errors}")

    if needs_logging:
        print()
        print("Files that still need `import logging` (for constants/handlers):")
        for path in needs_logging:
            print(f"  - {path}")

    if args.dry_run:
        print()
        print("This was a dry run. No files were modified.")
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
