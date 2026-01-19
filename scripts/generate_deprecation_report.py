#!/usr/bin/env python3
"""
Generate deprecation status report for documentation.

Scans codebase for @deprecated decorators and generates a Markdown
table showing deprecation status for all symbols.

Usage:
    python scripts/generate_deprecation_report.py

    # Specify output file
    python scripts/generate_deprecation_report.py --output docs/DEPRECATION_STATUS.md

    # Mark blockers as cleared
    python scripts/generate_deprecation_report.py --cleared old_function:internal_usage,equivalence_test

Output:
    Markdown table with deprecation status for all symbols

Created: 2026-01-20
Reference: docs/development/DEPRECATION_LIFECYCLE_POLICY.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import ClassVar

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

from mfg_pde.utils.deprecation import check_removal_readiness  # noqa: E402
from scripts.check_internal_deprecation import discover_deprecated_symbols  # noqa: E402


def parse_cleared_blockers(cleared_str: str | None) -> dict[str, list[str]]:
    """
    Parse cleared blockers from command line.

    Format: "symbol1:blocker1,blocker2 symbol2:blocker3"

    Returns:
        Dict mapping symbol name to list of cleared blockers
    """
    if not cleared_str:
        return {}

    result = {}
    for item in cleared_str.split():
        if ":" not in item:
            continue
        symbol, blockers = item.split(":", 1)
        result[symbol] = blockers.split(",")

    return result


def format_blockers(blockers: list[str]) -> str:
    """Format blocker list for Markdown table."""
    if not blockers:
        return "-"
    return ", ".join(f"`{b}`" for b in blockers)


def generate_markdown_report(
    deprecated_registry: dict,
    cleared_blockers: dict[str, list[str]],
    current_version: str = "v0.20.0",
) -> str:
    """
    Generate Markdown deprecation status report.

    Args:
        deprecated_registry: Dict of deprecated symbols from discovery
        cleared_blockers: Dict mapping symbol to cleared blocker list
        current_version: Current version for age calculation

    Returns:
        Markdown formatted report
    """
    lines = [
        "# Deprecation Status Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Current Version**: {current_version}",
        f"**Total Deprecated Symbols**: {len(deprecated_registry)}",
        "",
        "---",
        "",
        "## Currently Deprecated Symbols",
        "",
        "| Symbol | Since | Location | Replacement | Blockers Remaining | Ready? |",
        "|:-------|:------|:---------|:------------|:-------------------|:-------|",
    ]

    # Track statistics
    ready_count = 0
    blocked_by_tests = []
    blocked_by_docs = []
    blocked_by_usage = []

    for symbol_name, symbol in sorted(deprecated_registry.items()):
        # Mock object for readiness check
        class MockDeprecatedObject:
            _deprecation_meta: ClassVar[dict] = {
                "since": symbol.since,
                "removal_blockers": ["internal_usage", "equivalence_test", "migration_docs"],
                "replacement": symbol.replacement,
                "reason": "",
                "symbol": symbol_name,
            }

        # Check readiness
        completed = cleared_blockers.get(symbol_name, [])
        status = check_removal_readiness(MockDeprecatedObject, current_version, completed_blockers=completed)

        # Format status
        ready_icon = "✅" if status["ready"] else "❌"
        remaining = format_blockers(status["remaining_blockers"])

        # Truncate replacement for table
        replacement = symbol.replacement
        if len(replacement) > 50:
            replacement = replacement[:47] + "..."

        # Add table row
        lines.append(
            f"| `{symbol_name}` | {symbol.since} | `{symbol.location}` | {replacement} | {remaining} | {ready_icon} |"
        )

        # Track statistics
        if status["ready"]:
            ready_count += 1
        else:
            if "internal_usage" in status["remaining_blockers"]:
                blocked_by_usage.append(symbol_name)
            if "equivalence_test" in status["remaining_blockers"]:
                blocked_by_tests.append(symbol_name)
            if "migration_docs" in status["remaining_blockers"]:
                blocked_by_docs.append(symbol_name)

    # Add summary section
    lines.extend(
        [
            "",
            "---",
            "",
            "## Removal Readiness Summary",
            "",
            f"- **Ready for removal**: {ready_count} symbol(s) ✅",
            f"- **Blocked by internal usage**: {len(blocked_by_usage)} symbol(s)",
            f"- **Blocked by missing tests**: {len(blocked_by_tests)} symbol(s)",
            f"- **Blocked by missing docs**: {len(blocked_by_docs)} symbol(s)",
            "",
        ]
    )

    # Add detailed blocker sections
    if ready_count > 0:
        lines.extend(
            [
                "### ✅ Ready for Removal",
                "",
                "These symbols can be safely removed in the next major version:",
                "",
            ]
        )
        for symbol_name, symbol in sorted(deprecated_registry.items()):
            completed = cleared_blockers.get(symbol_name, [])

            class MockDeprecatedObject:
                _deprecation_meta: ClassVar[dict] = {
                    "since": symbol.since,
                    "removal_blockers": ["internal_usage", "equivalence_test", "migration_docs"],
                    "replacement": symbol.replacement,
                    "reason": "",
                    "symbol": symbol_name,
                }

            status = check_removal_readiness(MockDeprecatedObject, current_version, completed_blockers=completed)
            if status["ready"]:
                lines.append(f"- `{symbol_name}` (deprecated since {symbol.since})")

        lines.append("")

    if blocked_by_usage:
        lines.extend(
            [
                "### ⚠️ Blocked by Internal Usage",
                "",
                "These symbols are still used in production code:",
                "",
            ]
        )
        for symbol in sorted(blocked_by_usage):
            lines.append(f"- `{symbol}` - Run `python scripts/check_internal_deprecation.py` to find usages")
        lines.append("")

    if blocked_by_tests:
        lines.extend(
            [
                "### 🧪 Blocked by Missing Equivalence Tests",
                "",
                "These symbols need equivalence tests in `tests/unit/test_deprecation_equivalence.py`:",
                "",
            ]
        )
        for symbol in sorted(blocked_by_tests):
            lines.append(f"- `{symbol}` - Add test verifying `{symbol}(old) == new_api(new)`")
        lines.append("")

    if blocked_by_docs:
        lines.extend(
            [
                "### 📚 Blocked by Missing Migration Docs",
                "",
                "These symbols need migration guide in `docs/user/DEPRECATION_MODERNIZATION_GUIDE.md`:",
                "",
            ]
        )
        for symbol in sorted(blocked_by_docs):
            lines.append(f"- `{symbol}` - Document migration path and examples")
        lines.append("")

    # Add footer
    lines.extend(
        [
            "---",
            "",
            "## How to Clear Blockers",
            "",
            "### Internal Usage",
            "```bash",
            "# Find all internal usages",
            "python scripts/check_internal_deprecation.py",
            "",
            "# Fix violations by updating to new API",
            "# Then mark blocker as cleared:",
            "python scripts/generate_deprecation_report.py --cleared symbol:internal_usage",
            "```",
            "",
            "### Equivalence Test",
            "```python",
            "# Add to tests/unit/test_deprecation_equivalence.py",
            "def test_old_symbol_equivalence():",
            '    """Verify old API gives same result as new API."""',
            "    result_old = old_function(args)",
            "    result_new = new_function(args)",
            "    assert result_old == result_new",
            "```",
            "",
            "### Migration Docs",
            "```markdown",
            "# Add to docs/user/DEPRECATION_MODERNIZATION_GUIDE.md",
            "## Function: old_function → new_function",
            "",
            "### Old (Deprecated)",
            "```python",
            "result = old_function(param=value)",
            "```",
            "",
            "### New (Preferred)",
            "```python",
            "result = new_function(param=value)",
            "```",
            "```",
            "",
            "---",
            "",
            "**Reference**: `docs/development/DEPRECATION_LIFECYCLE_POLICY.md`",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate deprecation status report")
    parser.add_argument(
        "--output",
        default="docs/DEPRECATION_STATUS.md",
        help="Output file path (default: docs/DEPRECATION_STATUS.md)",
    )
    parser.add_argument(
        "--cleared",
        help="Cleared blockers in format 'symbol1:blocker1,blocker2 symbol2:blocker3'",
    )
    parser.add_argument(
        "--current-version",
        default="v0.20.0",
        help="Current version (default: v0.20.0)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Deprecation Report Generator")
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
        print("   No report generated.")
        return 0

    # Parse cleared blockers
    cleared_blockers = parse_cleared_blockers(args.cleared)

    # Generate report
    report = generate_markdown_report(deprecated_registry, cleared_blockers, current_version=args.current_version)

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    print(f"✅ Report generated: {output_path}")
    print(f"   Total symbols: {len(deprecated_registry)}")
    print()
    print(f"View report: cat {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
