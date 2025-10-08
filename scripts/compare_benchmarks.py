#!/usr/bin/env python3
"""
Benchmark Comparison Script for Performance Regression Testing

Compares pytest-benchmark results between PR and main branches to detect
performance regressions. Generates markdown reports and fails CI if
regressions exceed threshold.

Usage:
    python scripts/compare_benchmarks.py \\
        --pr-results .benchmarks/pr_results.json \\
        --main-results .benchmarks/main_results.json \\
        --threshold 15 \\
        --output .benchmarks/comparison.md
"""

import argparse
import json
import sys
from pathlib import Path


def load_benchmark_results(filepath: str | Path) -> dict:
    """Load pytest-benchmark JSON results."""
    with open(filepath) as f:
        return json.load(f)


def compare_benchmarks(
    pr_results: dict,
    main_results: dict,
    threshold_percent: float = 15.0,
) -> dict:
    """
    Compare benchmark results and identify regressions.

    Args:
        pr_results: Benchmark results from PR branch
        main_results: Benchmark results from main branch
        threshold_percent: Regression threshold as percentage (default: 15%)

    Returns:
        Dictionary containing comparison analysis
    """
    pr_benchmarks = {b["name"]: b for b in pr_results.get("benchmarks", [])}
    main_benchmarks = {b["name"]: b for b in main_results.get("benchmarks", [])}

    comparisons = []
    regressions = []
    improvements = []

    for name in pr_benchmarks:
        if name not in main_benchmarks:
            continue

        pr_bench = pr_benchmarks[name]
        main_bench = main_benchmarks[name]

        pr_time = pr_bench["stats"]["mean"]
        main_time = main_bench["stats"]["mean"]

        ratio = pr_time / main_time
        percent_change = (ratio - 1.0) * 100

        comparison = {
            "name": name,
            "pr_time": pr_time,
            "main_time": main_time,
            "ratio": ratio,
            "percent_change": percent_change,
            "pr_stddev": pr_bench["stats"]["stddev"],
            "main_stddev": main_bench["stats"]["stddev"],
        }

        comparisons.append(comparison)

        if percent_change > threshold_percent:
            regressions.append(comparison)
        elif percent_change < -threshold_percent:
            improvements.append(comparison)

    return {
        "comparisons": comparisons,
        "regressions": regressions,
        "improvements": improvements,
        "threshold": threshold_percent,
    }


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def generate_markdown_report(analysis: dict) -> str:
    """Generate markdown report from comparison analysis."""
    comparisons = analysis["comparisons"]
    regressions = analysis["regressions"]
    improvements = analysis["improvements"]
    threshold = analysis["threshold"]

    if not comparisons:
        return "⚠️ No comparable benchmarks found.\n"

    # Header
    report = f"**Regression threshold**: {threshold}% slower than main\n\n"

    # Regressions section
    if regressions:
        report += f"### ⚠️ Performance Regressions Detected ({len(regressions)})\n\n"
        report += "| Benchmark | Main | PR | Change |\n"
        report += "|-----------|------|----|---------|\n"
        for reg in regressions:
            report += f"| `{reg['name']}` | {format_time(reg['main_time'])} | {format_time(reg['pr_time'])} | 🔴 **+{reg['percent_change']:.1f}%** |\n"
        report += "\n"
    else:
        report += "### ✅ No Performance Regressions\n\n"
        report += f"All benchmarks are within {threshold}% of main branch performance.\n\n"

    # Improvements section
    if improvements:
        report += f"### 🚀 Performance Improvements ({len(improvements)})\n\n"
        report += "| Benchmark | Main | PR | Change |\n"
        report += "|-----------|------|----|---------|\n"
        for imp in improvements:
            report += f"| `{imp['name']}` | {format_time(imp['main_time'])} | {format_time(imp['pr_time'])} | 🟢 **{imp['percent_change']:.1f}%** |\n"
        report += "\n"

    # All results table
    report += "### 📊 Complete Benchmark Comparison\n\n"
    report += "| Benchmark | Main | PR | Change | Status |\n"
    report += "|-----------|------|----|---------|---------|\n"

    for comp in sorted(comparisons, key=lambda x: abs(x["percent_change"]), reverse=True):
        change = comp["percent_change"]
        if change > threshold:
            status = f"🔴 +{change:.1f}%"
        elif change < -threshold:
            status = f"🟢 {change:.1f}%"
        elif abs(change) < 2:
            status = f"⚪ {change:+.1f}%"
        else:
            status = f"🟡 {change:+.1f}%"

        report += f"| `{comp['name']}` | {format_time(comp['main_time'])} | {format_time(comp['pr_time'])} | {change:+.1f}% | {status} |\n"

    # Summary
    report += "\n---\n\n"
    report += f"**Summary**: {len(comparisons)} benchmarks compared, "
    report += f"{len(regressions)} regressions, {len(improvements)} improvements\n"

    return report


def main():
    parser = argparse.ArgumentParser(description="Compare pytest-benchmark results for performance regression testing")
    parser.add_argument(
        "--pr-results",
        required=True,
        help="Path to PR branch benchmark results JSON",
    )
    parser.add_argument(
        "--main-results",
        required=True,
        help="Path to main branch benchmark results JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Regression threshold percentage (default: 15)",
    )
    parser.add_argument(
        "--output",
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with error code if regressions detected",
    )

    args = parser.parse_args()

    # Load results
    try:
        pr_results = load_benchmark_results(args.pr_results)
        main_results = load_benchmark_results(args.main_results)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not load benchmark results: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in benchmark results: {e}")
        sys.exit(1)

    # Compare benchmarks
    analysis = compare_benchmarks(pr_results, main_results, args.threshold)

    # Generate report
    report = generate_markdown_report(analysis)

    # Output report
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"📄 Report written to: {args.output}")
    else:
        print(report)

    # Exit with error if regressions detected and flag set
    if args.fail_on_regression and analysis["regressions"]:
        print(f"\n❌ FAIL: {len(analysis['regressions'])} performance regression(s) detected")
        sys.exit(1)
    else:
        print("\n✅ PASS: Benchmark comparison complete")
        sys.exit(0)


if __name__ == "__main__":
    main()
