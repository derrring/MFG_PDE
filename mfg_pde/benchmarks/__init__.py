"""
Benchmarking tools for MFG_PDE performance evaluation.

This module provides comprehensive benchmarking capabilities for evaluating
the performance of different solvers, algorithms, and configurations across
various problem dimensions and sizes.
"""

from .highdim_benchmark_suite import (
    BenchmarkResult,
    BenchmarkSuite,
    HighDimMFGBenchmark,
    create_comprehensive_benchmark_suite,
    create_quick_benchmark_suite,
    run_standard_benchmarks,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "HighDimMFGBenchmark",
    "create_comprehensive_benchmark_suite",
    "create_quick_benchmark_suite",
    "run_standard_benchmarks",
]
