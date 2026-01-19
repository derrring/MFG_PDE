#!/usr/bin/env python3
"""
Audit deprecated symbols for removal readiness.

This script scans the codebase for @deprecated decorators and reports
which symbols are ready for removal based on:
1. All removal blockers cleared
2. Minimum age requirements met
3. No internal production usage

Usage:
    python scripts/audit_deprecated_symbols.py

    # Mark blockers as cleared for specific symbol
    python scripts/audit_deprecated_symbols.py --symbol old_function --cleared internal_usage equivalence_test

Output:
    - List of deprecated symbols
    - Removal readiness status for each
    - Actionable recommendations

Created: 2026-01-20 (Issue #616)
Reference: docs/development/DEPRECATION_LIFECYCLE_POLICY.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import ClassVar

# Add parent directory to path to import from scripts/
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

from mfg_pde.utils.deprecation import check_removal_readiness  # noqa: E402
from scripts.check_internal_deprecation import discover_deprecated_symbols  # noqa: E402


def format_blocker_list(blockers: list[str]) -> str:
    """Format blocker list for display."""
    if not blockers:
        return "None"
    return ", ".join(blockers)


def print_symbol_status(
    symbol_name: str,
    metadata: dict,
    completed_blockers: list[str],
    current_version: str = "v0.20.0",
) -> None:
    """Print detailed status for a deprecated symbol."""
    print(f"\n{'=' * 70}")
    print(f"Symbol: {symbol_name}")
    print(f"{'=' * 70}")
    print(f"Location: {metadata['location']}")
    print(f"Deprecated since: {metadata['since']}")
    print(f"Replacement: {metadata['replacement']}")
    if metadata.get("reason"):
        print(f"Reason: {metadata['reason']}")

    print(f"\nRemoval blockers: {format_blocker_list(metadata['removal_blockers'])}")
    print(f"Completed: {format_blocker_list(completed_blockers)}")

    # Check readiness (simplified - using mock function object)
    class MockDeprecatedObject:
        """Mock object to hold metadata for check_removal_readiness."""

        _deprecation_meta: ClassVar[dict] = {
            "since": metadata["since"],
            "removal_blockers": metadata["removal_blockers"],
            "replacement": metadata["replacement"],
            "reason": metadata.get("reason", ""),
            "symbol": symbol_name,
        }

    status = check_removal_readiness(
        MockDeprecatedObject,
        current_version=current_version,
        completed_blockers=completed_blockers,
    )

    print(f"\nRemoval readiness: {'✅ READY' if status['ready'] else '❌ NOT READY'}")

    if not status["ready"]:
        print("\nBlocking reasons:")
        for reason in status["blocking_reasons"]:
            print(f"  - {reason}")

        print("\nTo proceed with removal:")
        for blocker in status["remaining_blockers"]:
            if blocker == "internal_usage":
                print("  1. Run: python scripts/check_internal_deprecation.py")
                print("     Fix any violations found")
            elif blocker == "equivalence_test":
                print("  2. Add test: tests/unit/test_deprecation_equivalence.py")
                print(f"     Verify {symbol_name}(old) == new_api(new)")
            elif blocker == "migration_docs":
                print("  3. Update: docs/user/DEPRECATION_MODERNIZATION_GUIDE.md")
                print(f"     Document migration path for {symbol_name}")
    else:
        print("\n✅ All conditions met. Symbol can be removed in next major version.")
        print("\nRemoval checklist:")
        print("  1. Remove deprecated code")
        print("  2. Update CHANGELOG.md")
        print("  3. Run full test suite")
        print("  4. Update deprecation guide")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audit deprecated symbols for removal readiness")
    parser.add_argument(
        "--symbol",
        help="Specific symbol to audit (default: all)",
    )
    parser.add_argument(
        "--cleared",
        nargs="+",
        default=[],
        help="Blockers that have been cleared for this symbol",
    )
    parser.add_argument(
        "--current-version",
        default="v0.20.0",
        help="Current version for age calculation (default: v0.20.0)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Deprecated Symbol Audit")
    print("=" * 70)
    print()

    # Discover deprecated symbols
    repo_root = script_dir.parent
    src_path = repo_root / "mfg_pde"

    if not src_path.exists():
        print(f"❌ ERROR: Source directory not found: {src_path}", file=sys.stderr)
        return 2

    deprecated_registry = discover_deprecated_symbols(src_path)

    if not deprecated_registry:
        print("✅ No deprecated symbols found in codebase.")
        return 0

    print(f"Found {len(deprecated_registry)} deprecated symbol(s)\n")

    # If specific symbol requested
    if args.symbol:
        if args.symbol not in deprecated_registry:
            print(f"❌ ERROR: Symbol '{args.symbol}' not found", file=sys.stderr)
            print(f"Available symbols: {', '.join(deprecated_registry.keys())}")
            return 1

        symbol = deprecated_registry[args.symbol]
        print_symbol_status(
            args.symbol,
            {
                "location": symbol.location,
                "since": symbol.since,
                "removal_blockers": symbol.replacement.split("removal_blockers=")[1].split(")")[0].split(",")
                if "removal_blockers=" in symbol.replacement
                else ["internal_usage", "equivalence_test", "migration_docs"],
                "replacement": symbol.replacement,
                "reason": "",
            },
            completed_blockers=args.cleared,
            current_version=args.current_version,
        )
        return 0

    # Otherwise, show summary for all symbols
    print("Summary of all deprecated symbols:")
    print()

    for symbol_name, symbol in deprecated_registry.items():
        print(f"  • {symbol_name}")
        print(f"    Deprecated since: {symbol.since}")
        print(f"    Location: {symbol.location}")
        print(f"    Replacement: {symbol.replacement}")
        print()

    print(f"\nTotal: {len(deprecated_registry)} symbol(s)")
    print("\nFor detailed status of a specific symbol:")
    print(f"  python {Path(__file__).name} --symbol <symbol_name>")
    print("\nTo check removal readiness:")
    print(f"  python {Path(__file__).name} --symbol <symbol_name> --cleared internal_usage equivalence_test")

    return 0


if __name__ == "__main__":
    sys.exit(main())
