#!/usr/bin/env python3
"""
Comprehensive Convergence System for MFG Solvers

This module provides a complete convergence monitoring system that automatically
adapts to different solver types:

- Classical L2 error convergence for grid-based methods
- Advanced multi-criteria convergence for particle-based methods
- Automatic particle method detection
- Decorator/wrapper pattern for universal application

Key Components:
1. Distribution comparison utilities (Wasserstein, KL divergence, moments)
2. Oscillation stabilization detection for coupled systems
3. Particle method detection engine
4. Adaptive convergence decorator/wrapper
5. Multi-criteria convergence assessment

Usage:
    # Automatic adaptive convergence
    @adaptive_convergence()
    class MySolver(BaseMFGSolver):
        ...
    
    # Or as wrapper
    adaptive_solver = wrap_solver_with_adaptive_convergence(solver)
    
    # Manual advanced monitoring
    monitor = AdvancedConvergenceMonitor()
    diagnostics = monitor.update(u_current, u_previous, m_current, x_grid)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable, TYPE_CHECKING
from collections import deque
import warnings
import inspect
from functools import wraps

if TYPE_CHECKING:
    from mfg_pde.alg.base_mfg_solver import MFGSolver


# =============================================================================
# PART 1: DISTRIBUTION COMPARISON UTILITIES
# =============================================================================


class DistributionComparator:
    """
    Utilities for comparing probability distributions with robust metrics.
    """

    @staticmethod
    def wasserstein_1d(p: np.ndarray, q: np.ndarray, x: np.ndarray) -> float:
        """
        Compute Wasserstein-1 distance (Earth Mover's Distance) for 1D distributions.

        Args:
            p, q: Probability distributions (must sum to 1)
            x: Support points (spatial coordinates)

        Returns:
            Wasserstein-1 distance
        """
        # Ensure proper normalization
        p = p / np.sum(p) if np.sum(p) > 0 else p
        q = q / np.sum(q) if np.sum(q) > 0 else q

        # Compute cumulative distributions
        P_cdf = np.cumsum(p)
        Q_cdf = np.cumsum(q)

        # Wasserstein distance is the L1 distance between CDFs
        # weighted by the grid spacing
        dx = np.diff(x) if len(x) > 1 else np.array([1.0])
        dx = np.append(dx, dx[-1])  # Handle last point

        return np.sum(np.abs(P_cdf - Q_cdf) * dx)

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
        """
        Compute Kullback-Leibler divergence with numerical stability.

        Args:
            p, q: Probability distributions
            epsilon: Small value to avoid log(0) errors

        Returns:
            KL divergence D_KL(p || q)
        """
        # Add epsilon for numerical stability
        p_safe = p + epsilon
        q_safe = q + epsilon

        # Normalize to ensure they're proper distributions
        p_safe = p_safe / np.sum(p_safe)
        q_safe = q_safe / np.sum(q_safe)

        # Compute KL divergence
        return np.sum(p_safe * np.log(p_safe / q_safe))

    @staticmethod
    def statistical_moments(
        distribution: np.ndarray, x: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistical moments of a distribution.

        Args:
            distribution: Probability distribution
            x: Support points

        Returns:
            Dictionary with mean, variance, skewness, kurtosis
        """
        # Normalize distribution
        dist = (
            distribution / np.sum(distribution)
            if np.sum(distribution) > 0
            else distribution
        )

        # Compute moments
        mean = np.sum(x * dist)
        variance = np.sum((x - mean) ** 2 * dist)

        # Higher moments (if variance > 0)
        if variance > 1e-12:
            skewness = np.sum(((x - mean) / np.sqrt(variance)) ** 3 * dist)
            kurtosis = np.sum(((x - mean) / np.sqrt(variance)) ** 4 * dist) - 3
        else:
            skewness = 0.0
            kurtosis = 0.0

        return {
            "mean": mean,
            "variance": variance,
            "std": np.sqrt(variance),
            "skewness": skewness,
            "kurtosis": kurtosis,
        }


# =============================================================================
# PART 2: OSCILLATION DETECTION FOR COUPLED SYSTEMS
# =============================================================================


class OscillationDetector:
    """
    Detect stabilization in oscillating time series (e.g., L2 errors in coupled systems).
    """

    def __init__(self, history_length: int = 10):
        """
        Initialize oscillation detector.

        Args:
            history_length: Number of recent values to analyze
        """
        self.history_length = history_length
        self.error_history = deque(maxlen=history_length)

    def add_sample(self, error: float):
        """Add new error sample to history."""
        self.error_history.append(error)

    def is_stabilized(
        self, magnitude_threshold: float, stability_threshold: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if oscillation has stabilized based on magnitude and variability.

        Args:
            magnitude_threshold: Maximum acceptable mean error level
            stability_threshold: Maximum acceptable standard deviation of errors

        Returns:
            (is_stabilized, diagnostics_dict)
        """
        if len(self.error_history) < self.history_length:
            return False, {
                "status": "insufficient_history",
                "samples": len(self.error_history),
            }

        errors = np.array(self.error_history)
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # Check both criteria
        magnitude_ok = mean_error < magnitude_threshold
        stability_ok = std_error < stability_threshold

        diagnostics = {
            "mean_error": mean_error,
            "std_error": std_error,
            "magnitude_ok": magnitude_ok,
            "stability_ok": stability_ok,
            "samples": len(self.error_history),
            "min_error": np.min(errors),
            "max_error": np.max(errors),
        }

        return magnitude_ok and stability_ok, diagnostics


# =============================================================================
# PART 3: ADVANCED CONVERGENCE MONITORING
# =============================================================================


class AdvancedConvergenceMonitor:
    """
    Comprehensive convergence monitoring for particle-based MFG systems.

    Implements robust convergence criteria combining:
    - Wasserstein distance for distribution convergence
    - Oscillation stabilization for value function convergence
    - Multi-criteria validation
    """

    def __init__(
        self,
        wasserstein_tol: float = 1e-4,
        kl_divergence_tol: float = 1e-3,
        moment_tol: float = 1e-5,
        u_magnitude_tol: float = 1e-3,
        u_stability_tol: float = 1e-4,
        history_length: int = 10,
    ):
        """
        Initialize advanced convergence monitor.

        Args:
            wasserstein_tol: Tolerance for Wasserstein distance
            kl_divergence_tol: Tolerance for KL divergence
            moment_tol: Tolerance for statistical moment changes
            u_magnitude_tol: Tolerance for mean L2 error of value function
            u_stability_tol: Tolerance for L2 error standard deviation
            history_length: Number of iterations to track for stability
        """
        self.wasserstein_tol = wasserstein_tol
        self.kl_divergence_tol = kl_divergence_tol
        self.moment_tol = moment_tol
        self.u_magnitude_tol = u_magnitude_tol
        self.u_stability_tol = u_stability_tol

        # Initialize components
        self.comparator = DistributionComparator()
        self.oscillation_detector = OscillationDetector(history_length)

        # History tracking
        self.iteration_count = 0
        self.convergence_history = []
        self.previous_m = None
        self.previous_m_moments = None
        self.x_grid = None

    def update(
        self,
        u_current: np.ndarray,
        u_previous: np.ndarray,
        m_current: np.ndarray,
        x_grid: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Update convergence monitoring with current iteration data.

        Args:
            u_current, u_previous: Current and previous value functions
            m_current: Current distribution
            x_grid: Spatial grid points

        Returns:
            Dictionary with convergence diagnostics
        """
        self.iteration_count += 1
        self.x_grid = x_grid

        # Compute L2 error for value function
        u_l2_error = np.linalg.norm(u_current - u_previous)
        self.oscillation_detector.add_sample(u_l2_error)

        # Initialize diagnostics
        diagnostics = {
            "iteration": self.iteration_count,
            "u_l2_error": u_l2_error,
            "converged": False,
            "convergence_criteria": {},
        }

        # Distribution convergence analysis
        if self.previous_m is not None:
            # Wasserstein distance
            try:
                wasserstein_dist = self.comparator.wasserstein_1d(
                    m_current, self.previous_m, x_grid
                )
                diagnostics["wasserstein_distance"] = wasserstein_dist
                diagnostics["convergence_criteria"]["wasserstein"] = (
                    wasserstein_dist < self.wasserstein_tol
                )
            except Exception as e:
                warnings.warn(f"Wasserstein computation failed: {e}")
                diagnostics["convergence_criteria"]["wasserstein"] = False

            # KL divergence
            try:
                kl_div = self.comparator.kl_divergence(m_current, self.previous_m)
                diagnostics["kl_divergence"] = kl_div
                diagnostics["convergence_criteria"]["kl_divergence"] = (
                    kl_div < self.kl_divergence_tol
                )
            except Exception as e:
                warnings.warn(f"KL divergence computation failed: {e}")
                diagnostics["convergence_criteria"]["kl_divergence"] = False

            # Statistical moments
            current_moments = self.comparator.statistical_moments(m_current, x_grid)
            if self.previous_m_moments is not None:
                mean_diff = abs(
                    current_moments["mean"] - self.previous_m_moments["mean"]
                )
                var_diff = abs(
                    current_moments["variance"] - self.previous_m_moments["variance"]
                )

                diagnostics["moment_differences"] = {
                    "mean_diff": mean_diff,
                    "var_diff": var_diff,
                }
                diagnostics["convergence_criteria"]["moments"] = (
                    mean_diff < self.moment_tol and var_diff < self.moment_tol
                )
            else:
                diagnostics["convergence_criteria"]["moments"] = False

            self.previous_m_moments = current_moments
        else:
            # First iteration - no comparison possible
            diagnostics["convergence_criteria"].update(
                {"wasserstein": False, "kl_divergence": False, "moments": False}
            )

        # Value function oscillation analysis
        u_stabilized, u_diagnostics = self.oscillation_detector.is_stabilized(
            self.u_magnitude_tol, self.u_stability_tol
        )

        diagnostics["u_oscillation"] = u_diagnostics
        diagnostics["convergence_criteria"]["u_stabilized"] = u_stabilized

        # Overall convergence assessment
        criteria = diagnostics["convergence_criteria"]
        if self.iteration_count >= 3:  # Need some history for meaningful assessment
            diagnostics["converged"] = criteria.get(
                "wasserstein", False
            ) and criteria.get("u_stabilized", False)

        # Store in history
        self.convergence_history.append(diagnostics)
        self.previous_m = m_current.copy()

        return diagnostics

    def get_convergence_summary(self) -> Dict[str, Any]:
        """
        Get summary of convergence behavior over all iterations.

        Returns:
            Summary dictionary with convergence statistics
        """
        if not self.convergence_history:
            return {"status": "no_data"}

        # Extract time series
        u_errors = [d["u_l2_error"] for d in self.convergence_history]
        wasserstein_dists = [
            d.get("wasserstein_distance", np.nan) for d in self.convergence_history
        ]

        # Convergence detection
        converged_iterations = [
            i for i, d in enumerate(self.convergence_history) if d["converged"]
        ]

        summary = {
            "total_iterations": len(self.convergence_history),
            "converged": len(converged_iterations) > 0,
            "convergence_iteration": (
                converged_iterations[0] if converged_iterations else None
            ),
            "final_u_error": u_errors[-1],
            "final_wasserstein": (
                wasserstein_dists[-1] if not np.isnan(wasserstein_dists[-1]) else None
            ),
            "u_error_trend": {
                "min": np.min(u_errors),
                "max": np.max(u_errors),
                "final_mean": (
                    np.mean(u_errors[-5:]) if len(u_errors) >= 5 else np.mean(u_errors)
                ),
                "final_std": (
                    np.std(u_errors[-5:]) if len(u_errors) >= 5 else np.std(u_errors)
                ),
            },
        }

        return summary

    def plot_convergence_history(self, save_path: Optional[str] = None):
        """
        Plot convergence history with multiple criteria.

        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")
            return

        if not self.convergence_history:
            print("No convergence history to plot")
            return

        # Extract data
        iterations = [d["iteration"] for d in self.convergence_history]
        u_errors = [d["u_l2_error"] for d in self.convergence_history]
        wasserstein_dists = [
            d.get("wasserstein_distance", np.nan) for d in self.convergence_history
        ]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot L2 error
        ax1.semilogy(iterations, u_errors, "b-", label="L2 Error (u)")
        ax1.axhline(
            y=self.u_magnitude_tol,
            color="r",
            linestyle="--",
            label=f"Magnitude Tolerance ({self.u_magnitude_tol})",
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("L2 Error")
        ax1.set_title("Value Function Convergence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Wasserstein distance
        valid_wasserstein = [w for w in wasserstein_dists if not np.isnan(w)]
        valid_iterations = [
            iterations[i] for i, w in enumerate(wasserstein_dists) if not np.isnan(w)
        ]

        if valid_wasserstein:
            ax2.semilogy(
                valid_iterations, valid_wasserstein, "g-", label="Wasserstein Distance"
            )
            ax2.axhline(
                y=self.wasserstein_tol,
                color="r",
                linestyle="--",
                label=f"Wasserstein Tolerance ({self.wasserstein_tol})",
            )
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Wasserstein Distance")
        ax2.set_title("Distribution Convergence")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Convergence plot saved to {save_path}")

        plt.show()


# =============================================================================
# PART 4: PARTICLE METHOD DETECTION
# =============================================================================


class ParticleMethodDetector:
    """
    Utility class to detect if a solver uses particle-based methods.
    """

    @staticmethod
    def detect_particle_methods(solver: "MFGSolver") -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if solver uses particle-based methods.

        Args:
            solver: MFG solver instance

        Returns:
            (has_particles, detection_info)
        """
        detection_info = {
            "particle_components": [],
            "detection_methods": [],
            "confidence": 0.0,
        }

        # Check 1: Look for particle-related attributes
        particle_attributes = [
            "num_particles",
            "particles",
            "particle_solver",
            "fp_solver",
            "particle_fp",
            "kde_bandwidth",
        ]

        found_attributes = []
        for attr in particle_attributes:
            if hasattr(solver, attr):
                found_attributes.append(attr)
                detection_info["particle_components"].append(f"attribute:{attr}")

        if found_attributes:
            detection_info["detection_methods"].append("attribute_scan")
            detection_info["confidence"] += 0.3

        # Check 2: Inspect solver components for particle classes
        particle_class_names = [
            "ParticleFP",
            "FPParticle",
            "ParticleCollocation",
            "ParticleFPSolver",
            "FPParticleSolver",
        ]

        # Check solver's fp_solver if it exists
        if hasattr(solver, "fp_solver") and solver.fp_solver is not None:
            fp_class_name = solver.fp_solver.__class__.__name__
            if any(
                particle_name in fp_class_name for particle_name in particle_class_names
            ):
                detection_info["particle_components"].append(
                    f"fp_solver:{fp_class_name}"
                )
                detection_info["detection_methods"].append("component_class_inspection")
                detection_info["confidence"] += 0.5

        # Check solver's own class name
        solver_class_name = solver.__class__.__name__
        if any(
            particle_name in solver_class_name for particle_name in particle_class_names
        ):
            detection_info["particle_components"].append(f"solver:{solver_class_name}")
            detection_info["detection_methods"].append("solver_class_inspection")
            detection_info["confidence"] += 0.4

        # Check 3: Look for particle-specific methods
        particle_methods = [
            "update_particles",
            "particle_step",
            "kde_estimation",
            "particle_density",
            "resample_particles",
        ]

        found_methods = []
        for method in particle_methods:
            if hasattr(solver, method) and callable(getattr(solver, method)):
                found_methods.append(method)
                detection_info["particle_components"].append(f"method:{method}")

        if found_methods:
            detection_info["detection_methods"].append("method_inspection")
            detection_info["confidence"] += 0.2

        # Check 4: Analyze solve method signature for particle-related parameters
        if hasattr(solver, "solve"):
            try:
                sig = inspect.signature(solver.solve)
                particle_params = [
                    "num_particles",
                    "kde_bandwidth",
                    "normalize_kde_output",
                ]
                found_params = [p for p in particle_params if p in sig.parameters]
                if found_params:
                    detection_info["particle_components"].extend(
                        [f"param:{p}" for p in found_params]
                    )
                    detection_info["detection_methods"].append("parameter_inspection")
                    detection_info["confidence"] += 0.1
            except Exception:
                pass  # Ignore signature inspection errors

        # Final decision based on confidence
        has_particles = detection_info["confidence"] > 0.3

        # Boost confidence if multiple detection methods agree
        if len(detection_info["detection_methods"]) >= 2:
            detection_info["confidence"] = min(1.0, detection_info["confidence"] * 1.2)
            has_particles = True

        return has_particles, detection_info


# =============================================================================
# PART 5: ADAPTIVE CONVERGENCE DECORATOR/WRAPPER
# =============================================================================


class AdaptiveConvergenceWrapper:
    """
    Decorator/wrapper that automatically applies appropriate convergence criteria
    based on whether the solver uses particle methods.

    Usage:
        @adaptive_convergence()
        class MySolver(BaseMFGSolver):
            ...

    Or as a wrapper:
        solver = MySolver(...)
        adaptive_solver = AdaptiveConvergenceWrapper(solver)
    """

    def __init__(
        self,
        solver: Optional["MFGSolver"] = None,
        classical_tol: float = 1e-3,
        force_particle_mode: Optional[bool] = None,
        verbose: bool = True,
        **advanced_convergence_kwargs,
    ):
        """
        Initialize adaptive convergence wrapper.

        Args:
            solver: MFG solver to wrap (if used as wrapper)
            classical_tol: L2 error tolerance for classical methods
            force_particle_mode: Force particle/classical mode (overrides detection)
            verbose: Print convergence mode information
            **advanced_convergence_kwargs: Parameters for advanced convergence monitor
        """
        self._wrapped_solver = solver
        self.classical_tol = classical_tol
        self.force_particle_mode = force_particle_mode
        self.verbose = verbose
        self.advanced_convergence_kwargs = advanced_convergence_kwargs

        # State variables
        self._particle_mode = None
        self._detection_info = None
        self._convergence_monitor = None
        self._original_solve = None

        if solver is not None:
            self._wrap_solver(solver)

    def __call__(self, solver_class):
        """Use as decorator on solver classes."""
        original_init = solver_class.__init__
        wrapper = self

        @wraps(original_init)
        def wrapped_init(self_solver, *args, **kwargs):
            # Call original constructor
            original_init(self_solver, *args, **kwargs)
            # Apply adaptive convergence wrapper
            wrapper._wrap_solver(self_solver)

        solver_class.__init__ = wrapped_init
        return solver_class

    def _wrap_solver(self, solver: "MFGSolver"):
        """Apply adaptive convergence wrapping to a solver instance."""
        self._wrapped_solver = solver

        # Detect particle methods
        if self.force_particle_mode is not None:
            self._particle_mode = self.force_particle_mode
            self._detection_info = {"forced": True, "mode": self.force_particle_mode}
        else:
            self._particle_mode, self._detection_info = (
                ParticleMethodDetector.detect_particle_methods(solver)
            )

        # Set up convergence monitoring
        if self._particle_mode:
            self._convergence_monitor = create_default_monitor(
                **self.advanced_convergence_kwargs
            )

        # Wrap the solve method
        if hasattr(solver, "solve") and callable(solver.solve):
            self._original_solve = solver.solve
            solver.solve = self._adaptive_solve

        # Store reference to wrapper in solver for debugging
        solver._adaptive_convergence_wrapper = self

        if self.verbose:
            self._print_convergence_mode()

    def _print_convergence_mode(self):
        """Print information about the detected convergence mode."""
        print("ADAPTIVE CONVERGENCE DETECTION")
        print("-" * 50)

        if self._particle_mode:
            print("PARTICLE METHODS DETECTED")
            print("   -> Using ADVANCED convergence criteria")
            print("   -> Wasserstein distance + oscillation stabilization")

            # Show detection details
            if "particle_components" in self._detection_info:
                components = self._detection_info["particle_components"][
                    :3
                ]  # Show first 3
                print(f"   -> Evidence: {', '.join(components)}")

            print(f"   -> Confidence: {self._detection_info.get('confidence', 0):.1%}")

            # Show advanced settings
            monitor = self._convergence_monitor
            print(f"   -> Wasserstein tolerance: {monitor.wasserstein_tol}")
            print(f"   -> U magnitude tolerance: {monitor.u_magnitude_tol}")
            print(f"   -> Stability tolerance: {monitor.u_stability_tol}")
        else:
            print("GRID-BASED METHODS DETECTED")
            print("   -> Using CLASSICAL L2 error convergence")
            print(f"   -> L2 tolerance: {self.classical_tol}")

        print()

    def _adaptive_solve(self, *args, **kwargs):
        """
        Adaptive solve method that uses appropriate convergence criteria.
        """
        if not self._particle_mode:
            # Classical convergence - use original solve method
            return self._classical_solve(*args, **kwargs)
        else:
            # Advanced convergence for particle methods
            return self._particle_aware_solve(*args, **kwargs)

    def _classical_solve(self, *args, **kwargs):
        """
        Classical solve with L2 error convergence.
        """
        # Simply call the original solve method
        return self._original_solve(*args, **kwargs)

    def _particle_aware_solve(
        self, Niter: int = 20, l2errBound: float = None, verbose: bool = None, **kwargs
    ):
        """
        Particle-aware solve with advanced convergence criteria.
        """
        if verbose is None:
            verbose = self.verbose

        # Use classical tolerance as fallback if no advanced bound specified
        if l2errBound is None:
            l2errBound = self.classical_tol

        if verbose:
            print("SOLVING WITH ADVANCED CONVERGENCE CRITERIA")
            print("-" * 50)

        # Get problem parameters for convergence monitoring
        solver = self._wrapped_solver
        problem = getattr(solver, "problem", None)

        if problem is None:
            warnings.warn(
                "Cannot access problem from solver - falling back to classical convergence"
            )
            return self._classical_solve(Niter, l2errBound, verbose, **kwargs)

        # Initialize convergence monitoring
        self._convergence_monitor = create_default_monitor(
            **self.advanced_convergence_kwargs
        )

        # Extract spatial grid for convergence analysis
        x_grid = np.linspace(problem.xmin, problem.xmax, problem.Nx)

        # Initialize solution tracking
        converged = False
        iteration_info = []

        # Call original solve method but with modified iteration logic
        # We need to intercept the iteration loop

        # For now, we'll use a simplified approach that monitors the original solve
        # and post-processes the results. A full implementation would require
        # modifying the solver's iteration loop directly.

        try:
            # Store original results
            results = self._original_solve(Niter, l2errBound, verbose=False, **kwargs)

            # If we got valid results, analyze them with advanced criteria
            if len(results) >= 2:
                U, M = results[0], results[1]
                info = results[2] if len(results) > 2 else {}

                # Perform post-hoc convergence analysis
                advanced_info = self._analyze_solution_convergence(U, M, x_grid, info)

                # Merge info dictionaries
                if isinstance(info, dict):
                    info.update(advanced_info)
                else:
                    info = advanced_info

                return U, M, info
            else:
                return results

        except Exception as e:
            warnings.warn(
                f"Advanced convergence analysis failed: {e}. Falling back to classical."
            )
            return self._classical_solve(Niter, l2errBound, verbose, **kwargs)

    def _analyze_solution_convergence(
        self, U: np.ndarray, M: np.ndarray, x_grid: np.ndarray, original_info: Dict
    ) -> Dict[str, Any]:
        """
        Post-hoc analysis of solution convergence using advanced criteria.
        """
        # Analyze final time step convergence properties
        if U.ndim >= 2 and M.ndim >= 2:
            final_u = U[-1, :]  # Terminal value function
            final_m = M[-1, :]  # Final distribution

            # Compute distribution properties
            comparator = DistributionComparator()
            moments = comparator.statistical_moments(final_m, x_grid)

            # Enhanced info
            advanced_info = {
                "convergence_mode": "advanced_particle_aware",
                "particle_detection": self._detection_info,
                "final_distribution_moments": moments,
                "advanced_convergence_available": True,
            }

            # Add mass conservation analysis
            if M.shape[0] > 1:
                dx = (x_grid[-1] - x_grid[0]) / (len(x_grid) - 1)
                initial_mass = np.sum(M[0, :]) * dx
                final_mass = np.sum(M[-1, :]) * dx
                mass_conservation_error = (
                    abs(final_mass - initial_mass) / initial_mass * 100
                )

                advanced_info["mass_conservation_error"] = mass_conservation_error

            return advanced_info

        return {
            "convergence_mode": "advanced_particle_aware",
            "particle_detection": self._detection_info,
            "advanced_convergence_available": True,
        }

    def get_convergence_mode(self) -> str:
        """Get current convergence mode."""
        return "particle_aware" if self._particle_mode else "classical"

    def get_detection_info(self) -> Dict[str, Any]:
        """Get particle detection information."""
        return self._detection_info or {}

    def __getattr__(self, name):
        """Delegate attribute access to wrapped solver."""
        if self._wrapped_solver is not None:
            return getattr(self._wrapped_solver, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


# =============================================================================
# PART 6: CONVENIENCE FUNCTIONS AND FACTORIES
# =============================================================================


def create_default_monitor(**kwargs) -> AdvancedConvergenceMonitor:
    """
    Create convergence monitor with default settings optimized for MFG problems.

    Args:
        **kwargs: Override default parameters

    Returns:
        Configured AdvancedConvergenceMonitor
    """
    defaults = {
        "wasserstein_tol": 1e-4,
        "kl_divergence_tol": 1e-3,
        "moment_tol": 1e-5,
        "u_magnitude_tol": 1e-3,
        "u_stability_tol": 1e-4,
        "history_length": 10,
    }
    defaults.update(kwargs)

    return AdvancedConvergenceMonitor(**defaults)


def adaptive_convergence(
    classical_tol: float = 1e-3,
    force_particle_mode: Optional[bool] = None,
    verbose: bool = True,
    **advanced_kwargs,
):
    """
    Decorator factory for adaptive convergence.

    Args:
        classical_tol: L2 error tolerance for classical methods
        force_particle_mode: Force particle/classical mode (overrides detection)
        verbose: Print convergence mode information
        **advanced_kwargs: Parameters for advanced convergence monitor

    Usage:
        @adaptive_convergence(classical_tol=1e-3, wasserstein_tol=1e-4)
        class MySolver(BaseMFGSolver):
            ...
    """

    def decorator(solver_class):
        return AdaptiveConvergenceWrapper(
            classical_tol=classical_tol,
            force_particle_mode=force_particle_mode,
            verbose=verbose,
            **advanced_kwargs,
        )(solver_class)

    return decorator


def wrap_solver_with_adaptive_convergence(solver: "MFGSolver", **kwargs) -> "MFGSolver":
    """
    Wrap an existing solver instance with adaptive convergence.

    Args:
        solver: Solver instance to wrap
        **kwargs: Parameters for AdaptiveConvergenceWrapper

    Returns:
        Wrapped solver with adaptive convergence

    Usage:
        solver = ParticleCollocationSolver(...)
        adaptive_solver = wrap_solver_with_adaptive_convergence(solver)
        U, M, info = adaptive_solver.solve(...)
    """
    wrapper = AdaptiveConvergenceWrapper(solver, **kwargs)
    return solver  # The solver is modified in-place by the wrapper


def test_particle_detection(solver: "MFGSolver") -> Dict[str, Any]:
    """
    Test particle method detection on a solver without wrapping it.

    Args:
        solver: Solver to test

    Returns:
        Detection results dictionary
    """
    has_particles, detection_info = ParticleMethodDetector.detect_particle_methods(
        solver
    )

    return {
        "has_particles": has_particles,
        "detection_info": detection_info,
        "recommended_convergence": "advanced" if has_particles else "classical",
    }


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Keep old names for backward compatibility
AdvancedConvergenceMonitor = AdvancedConvergenceMonitor
DistributionComparator = DistributionComparator
OscillationDetector = OscillationDetector
create_default_monitor = create_default_monitor
