"""
Backend Factory for MFG_PDE

Factory methods for creating computational backends with appropriate configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..backends.base_backend import BaseBackend

from ..backends import create_backend, get_available_backends, get_backend_info
from ..core.mfg_problem import MFGProblem


class BackendFactory:
    """Factory for creating and managing computational backends."""

    @staticmethod
    def create_backend(backend_name: str = "auto", **kwargs: Any) -> BaseBackend:
        """
        Create a computational backend.

        Args:
            backend_name: Backend type ("numpy", "jax", or "auto")
            **kwargs: Backend-specific configuration

        Returns:
            Backend instance
        """
        return create_backend(backend_name, **kwargs)

    @staticmethod
    def create_optimal_backend(problem: MFGProblem, prefer_gpu: bool = True, precision: str = "float64") -> BaseBackend:
        """
        Create optimal backend based on problem characteristics.

        Args:
            problem: MFG problem instance
            prefer_gpu: Whether to prefer GPU acceleration
            precision: Numerical precision

        Returns:
            Optimal backend for the problem
        """
        available = get_available_backends()

        # Problem size analysis
        total_size = problem.Nx * problem.Nt
        is_large_problem = total_size > 100000

        # Backend selection logic
        if available.get("jax", False) and (is_large_problem or prefer_gpu):
            # Use JAX for large problems or when GPU is preferred
            device = "gpu" if prefer_gpu else "auto"
            return create_backend("jax", device=device, precision=precision)
        else:
            # Use NumPy for small problems or when JAX unavailable
            return create_backend("numpy", precision=precision)

    @staticmethod
    def benchmark_backends(problem: MFGProblem, backends: list | None = None) -> dict[str, Any]:
        """
        Benchmark available backends on a given problem.

        Args:
            problem: Test problem for benchmarking
            backends: List of backends to test (default: all available)

        Returns:
            Benchmark results dictionary
        """
        import time

        if backends is None:
            available = get_available_backends()
            backends = [name for name, avail in available.items() if avail]

        results = {}

        for backend_name in backends:
            try:
                # Create backend
                backend = create_backend(backend_name)

                # Simple benchmark: create arrays and do basic operations
                start_time = time.perf_counter()

                # Test array creation
                U = backend.zeros((problem.Nx,))
                M = backend.ones((problem.Nx,))
                x_grid = backend.linspace(problem.xmin, problem.xmax, problem.Nx)

                # Test basic operations
                for _ in range(10):
                    U_grad = backend.diff(U)
                    integral = backend.trapezoid(M, x=x_grid)
                    U = U + 0.01 * U_grad[:-1]  # Simple update

                end_time = time.perf_counter()

                results[backend_name] = {
                    "time": end_time - start_time,
                    "device_info": backend.get_device_info(),
                    "memory_info": backend.memory_usage(),
                    "status": "success",
                }

            except Exception as e:
                results[backend_name] = {
                    "time": float("inf"),
                    "error": str(e),
                    "status": "failed",
                }

        return results

    @staticmethod
    def get_backend_recommendations(problem: MFGProblem) -> dict[str, Any]:
        """
        Get backend recommendations based on problem characteristics.

        Args:
            problem: MFG problem to analyze

        Returns:
            Recommendations dictionary
        """
        total_size = problem.Nx * problem.Nt
        available = get_available_backends()

        recommendations: dict[str, Any] = {
            "problem_size": total_size,
            "size_category": ("small" if total_size < 10000 else "medium" if total_size < 100000 else "large"),
            "available_backends": available,
            "recommendations": [],
        }

        # Size-based recommendations
        if total_size < 1000:
            recommendations["recommendations"].append(
                {
                    "backend": "numpy",
                    "reason": "Small problem size, NumPy sufficient",
                    "priority": 1,
                }
            )
        elif total_size < 100000:
            if available.get("jax", False):
                recommendations["recommendations"].append(
                    {
                        "backend": "jax",
                        "reason": "Medium problem size, JAX provides good speedup",
                        "priority": 1,
                        "config": {"device": "auto", "jit_compile": True},
                    }
                )
            recommendations["recommendations"].append(
                {
                    "backend": "numpy",
                    "reason": "Reliable fallback for medium problems",
                    "priority": 2,
                }
            )
        else:  # Large problems
            if available.get("jax", False):
                recommendations["recommendations"].append(
                    {
                        "backend": "jax",
                        "reason": "Large problem size, GPU acceleration essential",
                        "priority": 1,
                        "config": {
                            "device": "gpu",
                            "jit_compile": True,
                            "precision": "float32",
                        },
                    }
                )
            recommendations["recommendations"].append(
                {
                    "backend": "numpy",
                    "reason": "May be too slow for large problems",
                    "priority": 3,
                    "warning": "Consider reducing problem size or using JAX",
                }
            )

        return recommendations


def create_backend_for_problem(problem: MFGProblem, backend: str = "auto", **kwargs: Any) -> BaseBackend:
    """
    Convenience function to create backend for a specific problem.

    Args:
        problem: MFG problem instance
        backend: Backend type or "auto" for automatic selection
        **kwargs: Backend configuration

    Returns:
        Configured backend instance
    """
    if backend == "auto":
        return BackendFactory.create_optimal_backend(problem, **kwargs)
    else:
        return BackendFactory.create_backend(backend, **kwargs)


def print_backend_info() -> None:
    """Print information about available backends."""
    info = get_backend_info()

    print(" MFG_PDE Backend Information")
    print("=" * 40)

    print("Available backends:")
    for name, available in info["available_backends"].items():
        status = "SUCCESS:" if available else "ERROR:"
        print(f"  {status} {name}")

    print(f"\nDefault backend: {info['default_backend']}")
    print(f"Registered backends: {', '.join(info['registered_backends'])}")

    if "jax_info" in info:
        jax_info = info["jax_info"]
        if "error" not in jax_info:
            print("\n JAX Information:")
            print(f"  Version: {jax_info['version']}")
            print(f"  Devices: {', '.join(jax_info['devices'])}")
            print(f"  Default device: {jax_info['default_device']}")
            print(f"  GPU available: {jax_info['has_gpu']}")
        else:
            print(f"\nWARNING:  JAX Error: {jax_info['error']}")


if __name__ == "__main__":
    # Demo backend information
    print_backend_info()

    # Create a test problem
    from ..core.mfg_problem import ExampleMFGProblem

    problem = ExampleMFGProblem(T=1.0, Nx=100, Nt=50)

    print("\n Backend Recommendations for Test Problem:")
    recommendations = BackendFactory.get_backend_recommendations(problem)

    print(f"Problem size: {recommendations['problem_size']} ({recommendations['size_category']})")
    print("Recommendations:")
    for i, rec in enumerate(recommendations["recommendations"], 1):
        print(f"  {i}. {rec['backend']} - {rec['reason']}")
        if "config" in rec:
            print(f"     Config: {rec['config']}")
        if "warning" in rec:
            print(f"     WARNING:  {rec['warning']}")
