#!/usr/bin/env python3
"""
Quick type checking and verification script for MFG_PDE development.
Provides rapid feedback on type safety improvements.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_quick_mypy(target_dir: str = "mfg_pde") -> tuple[bool, int, str]:
    """Run focused mypy check with minimal output."""
    print(f"🔍 Running mypy on {target_dir}...")

    start_time = time.time()
    result = subprocess.run(["mypy", target_dir, "--no-error-summary"], capture_output=True, text=True, timeout=60)
    duration = time.time() - start_time

    # Count errors
    error_count = len([line for line in result.stdout.split("\n") if "error:" in line])

    success = result.returncode == 0
    output = result.stdout + result.stderr

    print(f"⏱️  Completed in {duration:.1f}s")

    if success:
        print("✅ No type errors found!")
    else:
        print(f"⚠️  Found {error_count} type errors")

    return success, error_count, output


def run_ruff_check(target_dir: str = "mfg_pde") -> bool:
    """Run Ruff linting check."""
    print(f"🔍 Running Ruff check on {target_dir}...")

    result = subprocess.run(["ruff", "check", target_dir, "--quiet"], capture_output=True, text=True)

    success = result.returncode == 0
    if success:
        print("✅ No Ruff issues found!")
    else:
        issue_count = len(result.stdout.split("\n")) - 1
        print(f"⚠️  Found {issue_count} Ruff issues")
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)

    return success


def main():
    """Run quick type checking suite."""
    print("🚀 MFG_PDE Quick Type Check")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("mfg_pde").exists():
        print("❌ Error: Run from MFG_PDE root directory")
        return 1

    start_total = time.time()

    # Run mypy check
    mypy_success, error_count, mypy_output = run_quick_mypy()

    # Run Ruff check if available
    ruff_available = subprocess.run(["which", "ruff"], capture_output=True).returncode == 0
    if ruff_available:
        print()
        ruff_success = run_ruff_check()
    else:
        print("⚠️  Ruff not available, skipping lint check")
        ruff_success = True

    # Summary
    total_time = time.time() - start_total
    print(f"\n{'='*40}")
    print(f"📊 SUMMARY (completed in {total_time:.1f}s)")
    print(f"{'='*40}")

    if mypy_success and ruff_success:
        print("🎉 All checks passed!")
        return 0
    else:
        print("⚠️  Issues found - see output above")
        if not mypy_success:
            print(f"   • {error_count} mypy errors")
        if not ruff_success:
            print("   • Ruff linting issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
