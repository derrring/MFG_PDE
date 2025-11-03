"""
Phase 3 API Overhead Measurement (Quick Version)
=================================================

Fast benchmark measuring just the API overhead without full solves.

Measures:
1. Configuration creation overhead (presets, builder)
2. MFGProblem construction time
3. Factory function overhead

Run time: <1 minute (vs full benchmark: ~10+ minutes)

Usage:
    python benchmarks/phase3_api_overhead.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from mfg_pde import ExampleMFGProblem
from mfg_pde.config import ConfigBuilder, presets


def measure_time(func, repeat: int = 100) -> tuple[float, float]:
    """
    Measure execution time with statistics.

    Args:
        func: Function to benchmark
        repeat: Number of repetitions

    Returns:
        (mean_time, std_time) in seconds
    """
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return float(np.mean(times)), float(np.std(times))


def main():
    """Run quick API overhead benchmarks."""
    print("=" * 70)
    print("Phase 3 API Overhead Benchmark (Quick Version)")
    print("=" * 70)

    results: dict[str, dict[str, Any]] = {}

    # 1. Configuration creation overhead
    print("\n1. Configuration Creation Overhead")
    print("-" * 70)

    def preset_fast():
        presets.fast_solver()

    def preset_accurate():
        presets.accurate_solver()

    def builder():
        ConfigBuilder().backend(backend_type="numpy").picard(max_iterations=50, tolerance=1e-6).build()

    mean_fast, std_fast = measure_time(preset_fast, repeat=1000)
    mean_accurate, std_accurate = measure_time(preset_accurate, repeat=1000)
    mean_builder, std_builder = measure_time(builder, repeat=1000)

    results["config_creation"] = {
        "preset_fast": {"mean": mean_fast, "std": std_fast},
        "preset_accurate": {"mean": mean_accurate, "std": std_accurate},
        "builder": {"mean": mean_builder, "std": std_builder},
    }

    print(f"  presets.fast_solver():     {mean_fast * 1e6:6.1f} ± {std_fast * 1e6:4.1f} µs")
    print(f"  presets.accurate_solver(): {mean_accurate * 1e6:6.1f} ± {std_accurate * 1e6:4.1f} µs")
    print(f"  ConfigBuilder().build():   {mean_builder * 1e6:6.1f} ± {std_builder * 1e6:4.1f} µs")

    # 2. MFGProblem construction
    print("\n2. MFGProblem Construction Overhead")
    print("-" * 70)

    def construct_problem():
        ExampleMFGProblem()

    mean, std = measure_time(construct_problem, repeat=1000)

    results["problem_construction"] = {"mean": mean, "std": std}

    print(f"  ExampleMFGProblem():       {mean * 1e6:6.1f} ± {std * 1e6:4.1f} µs")

    # 3. Factory function overhead
    print("\n3. Factory Function Overhead")
    print("-" * 70)

    # Preallocate to avoid construction overhead in timing
    problems = [ExampleMFGProblem() for _ in range(100)]
    configs = [presets.fast_solver() for _ in range(100)]

    # Use the correct factory API - need to pass problem and config to create_standard_solver
    from mfg_pde.factory import create_standard_solver

    def factory_call():
        idx = factory_call.counter % 100
        create_standard_solver(problems[idx], solver_type="fixed_point", custom_config=configs[idx])
        factory_call.counter += 1

    factory_call.counter = 0

    mean, std = measure_time(factory_call, repeat=100)

    results["factory_overhead"] = {"mean": mean, "std": std}

    print(f"  create_standard_solver():  {mean * 1000:6.2f} ± {std * 1000:4.2f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Phase 3 API Overhead")
    print("=" * 70)

    config_overhead = results["config_creation"]["preset_fast"]["mean"]
    problem_overhead = results["problem_construction"]["mean"]
    factory_overhead = results["factory_overhead"]["mean"]

    total_overhead_per_call = config_overhead + problem_overhead + factory_overhead

    print("\nPer solve_mfg() call overhead (typical case):")
    print(f"  Config creation:   {config_overhead * 1e6:6.1f} µs")
    print(f"  Problem creation:  {problem_overhead * 1e6:6.1f} µs")
    print(f"  Factory overhead:  {factory_overhead * 1000:6.2f} ms")
    print("  ----------------------------------------")
    print(f"  Total overhead:    {total_overhead_per_call * 1000:6.2f} ms")

    print("\nInterpretation:")
    print(f"  For a solve taking 1.0s:   overhead is {(total_overhead_per_call / 1.0) * 100:.2f}%")
    print(f"  For a solve taking 10.0s:  overhead is {(total_overhead_per_call / 10.0) * 100:.2f}%")
    print(f"  For a solve taking 100.0s: overhead is {(total_overhead_per_call / 100.0) * 100:.3f}%")

    if total_overhead_per_call < 0.01:  # <10ms
        print("\n✓ Phase 3 API overhead is negligible (<10ms)")
    elif total_overhead_per_call < 0.05:  # <50ms
        print("\n✓ Phase 3 API overhead is acceptable (<50ms)")
    else:
        print(f"\n⚠ Phase 3 API overhead may be noticeable (>{total_overhead_per_call * 1000:.0f}ms)")

    # Save results
    import json

    results_path = Path("benchmarks/results/phase3_api_overhead.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
