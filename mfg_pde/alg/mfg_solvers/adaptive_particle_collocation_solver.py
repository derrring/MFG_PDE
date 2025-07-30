#!/usr/bin/env python3
"""
Adaptive Particle Collocation Solver

This solver demonstrates the decorator pattern for adaptive convergence.
It automatically detects particle methods and applies appropriate convergence criteria
without requiring separate enhanced solver classes.

The solver inherits from ParticleCollocationSolver and gains adaptive convergence
behavior through the @adaptive_convergence decorator.
"""

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ...utils.convergence import adaptive_convergence
from .particle_collocation_solver import ParticleCollocationSolver

if TYPE_CHECKING:
    from mfg_pde.geometry import BoundaryConditions
    from mfg_pde.core.mfg_problem import MFGProblem


@adaptive_convergence(
    classical_tol=1e-3,
    wasserstein_tol=1e-4,
    u_magnitude_tol=1e-3,
    u_stability_tol=1e-4,
    history_length=10,
    verbose=True,
)
class AdaptiveParticleCollocationSolver(ParticleCollocationSolver):
    """
    Particle Collocation Solver with automatic adaptive convergence.

    This solver automatically detects that it uses particle methods and applies
    advanced convergence criteria (Wasserstein distance + oscillation stabilization)
    instead of classical L2 error convergence.

    The @adaptive_convergence decorator handles all the convergence logic
    transparently - no changes needed to the core solving algorithm.

    Features:
    - Automatic particle method detection
    - Advanced convergence criteria for particle-based distributions
    - Classical fallback for non-particle components
    - Detailed convergence diagnostics
    - Backward compatibility with existing ParticleCollocationSolver
    """

    def __init__(
        self,
        problem: "MFGProblem",
        collocation_points: np.ndarray,
        num_particles: int = 5000,
        delta: float = 0.1,
        taylor_order: int = 2,
        weight_function: str = "wendland",
        max_newton_iterations: int = 30,
        newton_tolerance: float = 1e-4,
        kde_bandwidth: str = "scott",
        normalize_kde_output: bool = False,
        boundary_indices: Optional[np.ndarray] = None,
        boundary_conditions: Optional["BoundaryConditions"] = None,
        use_monotone_constraints: bool = False,
    ):
        """
        Initialize adaptive particle collocation solver.

        All parameters are identical to ParticleCollocationSolver.
        The adaptive convergence behavior is added automatically by the decorator.
        """
        super().__init__(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=num_particles,
            delta=delta,
            taylor_order=taylor_order,
            weight_function=weight_function,
            max_newton_iterations=max_newton_iterations,
            newton_tolerance=newton_tolerance,
            kde_bandwidth=kde_bandwidth,
            normalize_kde_output=normalize_kde_output,
            boundary_indices=boundary_indices,
            boundary_conditions=boundary_conditions,
            use_monotone_constraints=use_monotone_constraints,
        )

        # The decorator automatically:
        # 1. Detects particle methods (num_particles, kde_bandwidth, etc.)
        # 2. Sets up advanced convergence monitoring
        # 3. Wraps the solve method with adaptive criteria
        # 4. Provides detailed convergence diagnostics

    def get_convergence_mode(self) -> str:
        """
        Get the current convergence mode detected by the adaptive decorator.

        Returns:
            "particle_aware" for advanced criteria, "classical" for L2 error
        """
        if hasattr(self, "_adaptive_convergence_wrapper"):
            return self._adaptive_convergence_wrapper.get_convergence_mode()
        return "unknown"

    def get_detection_info(self) -> Dict[str, Any]:
        """
        Get information about particle method detection.

        Returns:
            Dictionary with detection details and confidence
        """
        if hasattr(self, "_adaptive_convergence_wrapper"):
            return self._adaptive_convergence_wrapper.get_detection_info()
        return {}

    def print_convergence_info(self):
        """Print detailed information about adaptive convergence behavior."""
        mode = self.get_convergence_mode()
        detection_info = self.get_detection_info()

        print("ADAPTIVE CONVERGENCE INFO")
        print("-" * 40)
        print(f"Convergence Mode: {mode.upper()}")

        if detection_info:
            print(f"Detection Confidence: {detection_info.get('confidence', 0):.1%}")

            components = detection_info.get("particle_components", [])
            if components:
                print("Evidence Found:")
                for comp in components[:5]:  # Show first 5
                    print(f"  - {comp}")
                if len(components) > 5:
                    print(f"  - ... and {len(components) - 5} more")

        if hasattr(self, "_adaptive_convergence_wrapper"):
            wrapper = self._adaptive_convergence_wrapper
            if wrapper._convergence_monitor:
                monitor = wrapper._convergence_monitor
                print("\nAdvanced Convergence Settings:")
                print(f"  Wasserstein tolerance: {monitor.wasserstein_tol}")
                print(f"  U magnitude tolerance: {monitor.u_magnitude_tol}")
                print(f"  Stability tolerance: {monitor.u_stability_tol}")
                print(
                    f"  History length: {monitor.oscillation_detector.history_length}"
                )


# Convenience function for creating adaptive solver
def create_adaptive_particle_solver(
    problem: "MFGProblem", collocation_points: np.ndarray, **kwargs
) -> AdaptiveParticleCollocationSolver:
    """
    Create adaptive particle collocation solver with optimized defaults.

    Args:
        problem: MFG problem instance
        collocation_points: Spatial collocation points
        **kwargs: Additional solver parameters

    Returns:
        Configured AdaptiveParticleCollocationSolver
    """
    defaults = {
        "num_particles": 5000,
        "delta": 0.4,
        "taylor_order": 2,
        "weight_function": "wendland",
        "use_monotone_constraints": True,
    }
    defaults.update(kwargs)

    return AdaptiveParticleCollocationSolver(problem, collocation_points, **defaults)


# Example of applying decorator to existing solver classes
@adaptive_convergence(verbose=False)  # Quiet mode for batch processing
class SilentAdaptiveParticleCollocationSolver(ParticleCollocationSolver):
    """
    Adaptive particle collocation solver with minimal output.
    Useful for batch processing or when detailed convergence info isn't needed.
    """

    pass


@adaptive_convergence(
    classical_tol=5e-4,  # Stricter classical tolerance
    wasserstein_tol=5e-5,  # Stricter Wasserstein tolerance
    u_magnitude_tol=5e-4,  # Stricter magnitude tolerance
    verbose=True,
)
class HighPrecisionAdaptiveParticleCollocationSolver(ParticleCollocationSolver):
    """
    Adaptive particle collocation solver with high precision convergence criteria.
    Suitable for problems requiring very accurate solutions.
    """

    pass


# Backward compatibility alias
import warnings


@adaptive_convergence(verbose=False)
class QuietAdaptiveParticleCollocationSolver(ParticleCollocationSolver):
    """
    Deprecated: Use SilentAdaptiveParticleCollocationSolver instead.

    This is a backward compatibility alias for the renamed class.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "QuietAdaptiveParticleCollocationSolver is deprecated. Use SilentAdaptiveParticleCollocationSolver instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
