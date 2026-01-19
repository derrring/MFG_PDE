#!/usr/bin/env python3
"""
Check for internal usage of deprecated APIs in production code.

This script:
1. Scans mfg_pde/ for @deprecated decorators to build symbol registry
2. Scans production code for calls to deprecated symbols
3. Reports violations and exits with error code

Prevents Issue #616: "Factory not updated → production code got broken behavior"

Usage:
    python scripts/check_internal_deprecation.py

Exit Codes:
    0: No violations found
    1: Deprecated symbols used in production code
    2: Script error

Reference: docs/development/DEPRECATION_LIFECYCLE_POLICY.md
Created: 2026-01-20 (Issue #616)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import NamedTuple


class DeprecatedSymbol(NamedTuple):
    """Metadata for a deprecated symbol."""

    name: str
    location: str  # file:line where @deprecated was found
    since: str
    replacement: str
    removal_blockers: list[str] | None = None  # Conditions blocking removal
    reason: str | None = None  # Optional: explanation of why deprecated


class DeprecationDiscoveryVisitor(ast.NodeVisitor):
    """
    AST visitor to discover @deprecated decorators.

    Finds all functions/classes/methods decorated with @deprecated
    and extracts their metadata.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.deprecated_symbols: list[DeprecatedSymbol] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition to check for @deprecated decorator."""
        self._check_decorators(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self._check_decorators(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition to check for @deprecated decorator."""
        self._check_decorators(node)
        self.generic_visit(node)

    def _check_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        """Check if node has @deprecated decorator."""
        for decorator in node.decorator_list:
            # Check for @deprecated(...) pattern
            if isinstance(decorator, ast.Call):
                func_name = self._get_decorator_name(decorator.func)
                if func_name == "deprecated" or func_name == "deprecated_parameter":
                    # Extract metadata from decorator arguments
                    metadata = self._extract_metadata(decorator)
                    if metadata:
                        symbol = DeprecatedSymbol(
                            name=node.name,
                            location=f"{self.filename}:{node.lineno}",
                            **metadata,
                        )
                        self.deprecated_symbols.append(symbol)

    def _get_decorator_name(self, node: ast.expr) -> str | None:
        """Get decorator function name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _extract_metadata(self, call: ast.Call) -> dict[str, str | list[str]] | None:
        """Extract metadata from @deprecated decorator.

        Format: @deprecated(since="v1.0", replacement="new_api", removal_blockers=["internal_usage"])
        """
        metadata = {}

        for keyword in call.keywords:
            if keyword.arg in ["since", "replacement", "reason"]:
                if isinstance(keyword.value, ast.Constant):
                    metadata[keyword.arg] = keyword.value.value
            elif keyword.arg == "removal_blockers":
                # Extract list of blocker strings
                if isinstance(keyword.value, ast.List):
                    blockers = []
                    for elt in keyword.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            blockers.append(elt.value)
                    metadata["removal_blockers"] = blockers

        # Require: since, replacement, removal_blockers
        has_required = "since" in metadata and "replacement" in metadata and "removal_blockers" in metadata

        if has_required:
            return metadata
        return None


class DeprecationUsageVisitor(ast.NodeVisitor):
    """
    AST visitor to find calls to deprecated symbols.

    Scans code for function calls, attribute access, and class instantiation
    that use deprecated symbols.
    """

    def __init__(self, filename: str, deprecated_names: set[str]):
        self.filename = filename
        self.deprecated_names = deprecated_names
        self.violations: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for deprecated symbols."""
        # Direct function call: deprecated_func()
        if isinstance(node.func, ast.Name):
            if node.func.id in self.deprecated_names:
                self.violations.append(f"{self.filename}:{node.lineno} calls deprecated '{node.func.id}()'")

        # Attribute call: module.deprecated_func()
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.deprecated_names:
                self.violations.append(f"{self.filename}:{node.lineno} calls deprecated '{node.func.attr}()'")

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check name references (variable assignment, etc.)."""
        # Only flag if it's being used (Load context), not defined
        if isinstance(node.ctx, ast.Load):
            if node.id in self.deprecated_names:
                self.violations.append(f"{self.filename}:{node.lineno} uses deprecated symbol '{node.id}'")

        self.generic_visit(node)


def discover_deprecated_symbols(src_path: Path) -> dict[str, DeprecatedSymbol]:
    """
    Scan codebase for @deprecated decorators.

    Args:
        src_path: Root directory to scan (e.g., mfg_pde/)

    Returns:
        Dict mapping symbol name to DeprecatedSymbol metadata
    """
    print("🔍 Scanning for @deprecated decorators...")
    deprecated_registry: dict[str, DeprecatedSymbol] = {}

    for py_file in src_path.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            visitor = DeprecationDiscoveryVisitor(str(py_file))
            visitor.visit(tree)

            for symbol in visitor.deprecated_symbols:
                # Skip duplicates (e.g., multiple @deprecated_parameter on same function)
                if symbol.name not in deprecated_registry:
                    deprecated_registry[symbol.name] = symbol
                    # Format: Show removal blockers
                    if symbol.removal_blockers:
                        blockers_str = ", ".join(symbol.removal_blockers)
                        strategy = f"blockers=[{blockers_str}]"
                    else:
                        strategy = "no removal strategy"
                    print(f"   Found @deprecated: {symbol.name} at {symbol.location} ({strategy})")

        except (SyntaxError, UnicodeDecodeError):
            # Skip files with syntax errors or encoding issues
            pass

    print(f"   Total deprecated symbols: {len(deprecated_registry)}\n")
    return deprecated_registry


def check_production_code(
    src_path: Path,
    deprecated_names: set[str],
    exclude_files: set[str] | None = None,
) -> list[str]:
    """
    Check production code for usage of deprecated symbols.

    Args:
        src_path: Root directory to scan
        deprecated_names: Set of deprecated symbol names
        exclude_files: Optional set of files to skip (e.g., deprecation.py itself)

    Returns:
        List of violation messages
    """
    if not deprecated_names:
        return []

    print("🔍 Checking production code for deprecated symbol usage...")
    all_violations: list[str] = []

    exclude_files = exclude_files or set()

    for py_file in src_path.rglob("*.py"):
        # Skip test files
        if "test" in py_file.parts or py_file.name.startswith("test_"):
            continue

        # Skip excluded files (e.g., deprecation.py itself)
        if py_file.name in exclude_files:
            continue

        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            visitor = DeprecationUsageVisitor(str(py_file), deprecated_names)
            visitor.visit(tree)

            if visitor.violations:
                all_violations.extend(visitor.violations)

        except (SyntaxError, UnicodeDecodeError):
            pass

    return all_violations


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 = success, 1 = violations found, 2 = error)
    """
    print("=" * 70)
    print("Internal Deprecation Check")
    print("Enforcing: docs/development/DEPRECATION_LIFECYCLE_POLICY.md")
    print("=" * 70)
    print()

    # Determine source directory
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    src_path = repo_root / "mfg_pde"

    if not src_path.exists():
        print(f"❌ ERROR: Source directory not found: {src_path}", file=sys.stderr)
        return 2

    # Phase 1: Discover deprecated symbols
    deprecated_registry = discover_deprecated_symbols(src_path)

    if not deprecated_registry:
        print("✅ No @deprecated decorators found. Check passed.")
        return 0

    # Phase 2: Check production code
    deprecated_names = set(deprecated_registry.keys())
    violations = check_production_code(
        src_path,
        deprecated_names,
        exclude_files={"deprecation.py"},  # Exclude the decorator itself
    )

    # Report results
    print()
    if violations:
        print("❌ VIOLATIONS FOUND")
        print("=" * 70)
        for violation in violations:
            print(f"   {violation}")
        print()
        print("⛔ FAILURE: Production code calls deprecated functions.")
        print("   Per DEPRECATION_LIFECYCLE_POLICY.md, you must:")
        print("   1. Update all internal call sites to use new API")
        print("   2. Ensure deprecated code redirects to new API")
        print("   3. Add equivalence test verifying old = new")
        print()
        print(f"   Found {len(violations)} violation(s) across production code.")
        return 1
    else:
        print("✅ SUCCESS: No deprecated symbols used in production code.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
