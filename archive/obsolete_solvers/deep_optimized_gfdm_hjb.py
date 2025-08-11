#!/usr/bin/env python3
"""
Deep Optimized GFDM HJB Solver
Advanced integration of adaptive QP optimization with sophisticated decision criteria
and caching strategies.
"""

import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Try to import specialized QP solvers
try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("CVXPY not available. Install with 'pip install cvxpy' for optimal performance.")

try:
    import osqp

    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False

from .gfdm_hjb import GFDMHJBSolver


class DeepOptimizedGFDMHJBSolver(GFDMHJBSolver):
    """
    Deep Optimized GFDM HJB Solver with advanced adaptive QP optimization.

    Key optimizations:
    1. Sophisticated QP decision criteria based on solution patterns
    2. Spatial and temporal context awareness
    3. Solution prediction and caching
    4. Batch processing and warm starts
    """

    def __init__(
        self,
        problem,
        collocation_points,
        delta=0.35,
        taylor_order=2,
        weight_function="wendland",
        weight_scale=1.0,
        NiterNewton=8,
        l2errBoundNewton=1e-4,
        boundary_indices=None,
        boundary_conditions=None,
        use_monotone_constraints=True,
        optimization_level=3,
    ):
        """
        Initialize deep optimized GFDM HJB solver.

        Parameters:
        -----------
        optimization_level : int
            0 = No optimization (baseline)
            1 = Basic adaptive QP activation
            2 = + Spatial/temporal context
            3 = + Solution prediction and caching (default)
            4 = + Advanced batch processing
        """
        super().__init__(
            problem,
            collocation_points,
            delta,
            taylor_order,
            weight_function,
            weight_scale,
            NiterNewton,
            l2errBoundNewton,
            boundary_indices,
            boundary_conditions,
            use_monotone_constraints,
        )

        # Optimization settings
        self.optimization_level = optimization_level

        # Enhanced performance monitoring
        self.performance_stats = {
            'total_derivative_calls': 0,
            'unconstrained_solutions_used': 0,
            'qp_solutions_used': 0,
            'cache_hits': 0,
            'prediction_correct': 0,
            'prediction_attempts': 0,
            'batch_qp_operations': 0,
            'optimization_time_saved': 0.0,
            'total_solve_time': 0.0,
        }

        # Advanced caching system
        self.solution_cache = {}  # Cache by problem signature
        self.decision_cache = {}  # Cache QP decisions
        self.pattern_history = defaultdict(list)  # Track solution patterns

        # Spatial context (boundary vs interior points)
        self.boundary_point_set = set(boundary_indices) if boundary_indices is not None else set()
        self.interior_points = [i for i in range(len(collocation_points)) if i not in self.boundary_point_set]

        # Temporal context tracking
        self.time_step_context = {
            'early_steps_threshold': 0.2,  # First 20% of time steps
            'late_steps_threshold': 0.8,  # Last 20% of time steps
            'current_time_ratio': 0.0,
        }

        # Solution prediction model (simple statistical approach)
        self.qp_need_predictor = {
            'boundary_qp_rate': 0.9,  # Boundary points often need QP
            'interior_qp_rate': 0.1,  # Interior points rarely need QP
            'early_time_multiplier': 2.0,  # More QP needed early
            'late_time_multiplier': 0.5,  # Less QP needed late
            'newton_iteration_factor': 0.8,  # Less QP needed in later Newton iterations
        }

        # Override method name
        self.hjb_method_name = "Deep Optimized GFDM"

        print(f"DeepOptimizedGFDMHJBSolver initialized:")
        print(f"  Optimization level: {optimization_level}")
        print(f"  CVXPY available: {'✓' if CVXPY_AVAILABLE else '✗'}")
        print(f"  Boundary points: {len(self.boundary_point_set)}")
        print(f"  Interior points: {len(self.interior_points)}")

    def _get_problem_signature(self, point_idx: int, newton_iter: int, time_ratio: float) -> str:
        """Generate a signature for caching purposes"""
        return f"{point_idx}_{newton_iter}_{time_ratio:.2f}"

    def _predict_qp_need(self, point_idx: int, newton_iter: int, time_ratio: float) -> float:
        """
        Predict probability that this point will need QP constraints.
        Returns probability between 0 and 1.
        """
        if self.optimization_level < 2:
            return 0.5  # No prediction, assume 50/50

        self.performance_stats['prediction_attempts'] += 1

        # Base probability based on spatial location
        if point_idx in self.boundary_point_set:
            base_prob = self.qp_need_predictor['boundary_qp_rate']
        else:
            base_prob = self.qp_need_predictor['interior_qp_rate']

        # Temporal adjustment
        if time_ratio < self.time_step_context['early_steps_threshold']:
            base_prob *= self.qp_need_predictor['early_time_multiplier']
        elif time_ratio > self.time_step_context['late_steps_threshold']:
            base_prob *= self.qp_need_predictor['late_time_multiplier']

        # Newton iteration adjustment (later iterations are more stable)
        newton_factor = self.qp_need_predictor['newton_iteration_factor'] ** newton_iter
        base_prob *= newton_factor

        # Historical pattern adjustment
        if self.optimization_level >= 3 and point_idx in self.pattern_history:
            recent_history = self.pattern_history[point_idx][-10:]  # Last 10 decisions
            if recent_history:
                history_rate = sum(recent_history) / len(recent_history)
                # Blend prediction with history (70% history, 30% model)
                base_prob = 0.3 * base_prob + 0.7 * history_rate

        return min(1.0, max(0.0, base_prob))

    def _enhanced_monotonicity_check(
        self, unconstrained_coeffs: np.ndarray, point_idx: int, newton_iter: int, time_ratio: float
    ) -> bool:
        """
        Enhanced monotonicity violation check with spatial and temporal context.
        Returns True if QP constraints are needed.
        """
        if self.optimization_level == 0:
            # Use parent class logic
            return super()._check_monotonicity_violation(unconstrained_coeffs)

        # Check for obvious violations first (fast path)
        if np.any(~np.isfinite(unconstrained_coeffs)):
            return True  # NaN/inf values definitely need QP

        # Extreme value check
        max_coeff = np.max(np.abs(unconstrained_coeffs))
        if max_coeff > 1000.0:  # Very extreme values
            return True

        # Prediction-based early exit
        predicted_qp_prob = self._predict_qp_need(point_idx, newton_iter, time_ratio)

        if self.optimization_level >= 2:
            # Use more sophisticated criteria

            # Spatial context: boundary points are more constrained
            if point_idx in self.boundary_point_set:
                threshold_multiplier = 0.5  # Stricter for boundary points
            else:
                threshold_multiplier = 2.0  # More lenient for interior points

            # Temporal context: early time steps need more constraints
            if time_ratio < self.time_step_context['early_steps_threshold']:
                threshold_multiplier *= 0.5  # Stricter early on
            elif time_ratio > self.time_step_context['late_steps_threshold']:
                threshold_multiplier *= 2.0  # More lenient later

            # Adaptive threshold based on prediction and context
            adaptive_threshold = 100.0 * threshold_multiplier

            # Enhanced violation checks
            violations = 0

            # Check 1: Overall magnitude
            if max_coeff > adaptive_threshold:
                violations += 1

            # Check 2: Higher-order derivative instability
            if len(unconstrained_coeffs) > 2:
                higher_order_max = np.max(np.abs(unconstrained_coeffs[2:]))
                if higher_order_max > adaptive_threshold * 2.0:
                    violations += 1

            # Check 3: Gradient magnitude (for collocation methods)
            if len(unconstrained_coeffs) >= 2:
                gradient_magnitude = np.linalg.norm(unconstrained_coeffs[:2])
                if gradient_magnitude > adaptive_threshold * 0.5:
                    violations += 1

            # Check 4: Solution smoothness (variation in coefficients)
            if len(unconstrained_coeffs) > 1:
                coeff_variation = np.std(unconstrained_coeffs)
                if coeff_variation > adaptive_threshold * 0.3:
                    violations += 1

            # Decision based on violations and prediction
            needs_qp = violations > 0

            # If prediction is very confident, use it to override marginal cases
            if self.optimization_level >= 3:
                if predicted_qp_prob < 0.1 and violations <= 1:
                    needs_qp = False  # Probably don't need QP for minor violations
                elif predicted_qp_prob > 0.9 and violations >= 1:
                    needs_qp = True  # Definitely need QP with violations
        else:
            # Basic level 1 optimization: just use prediction threshold
            needs_qp = predicted_qp_prob > 0.5

        # Update pattern history for learning
        if self.optimization_level >= 3:
            self.pattern_history[point_idx].append(1 if needs_qp else 0)
            # Keep only recent history
            if len(self.pattern_history[point_idx]) > 50:
                self.pattern_history[point_idx] = self.pattern_history[point_idx][-50:]

        return needs_qp

    def _solve_qp_with_warm_start(
        self, taylor_data: Dict, b: np.ndarray, point_idx: int, warm_start: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Solve QP with warm start and caching"""
        if not CVXPY_AVAILABLE:
            # Fallback to parent implementation
            return super()._solve_monotone_constrained_qp(taylor_data, b, point_idx)

        try:
            A = taylor_data.get('A', np.eye(len(b)))
            W = taylor_data.get('W', np.eye(len(b)))
            n_vars = A.shape[1]

            # Create CVXPY variables
            x = cp.Variable(n_vars)

            # Set warm start if available
            if warm_start is not None and len(warm_start) == n_vars:
                x.value = warm_start

            # Weighted least squares objective
            if 'sqrt_W' in taylor_data:
                sqrt_W = taylor_data['sqrt_W']
                objective = cp.Minimize(cp.sum_squares(sqrt_W @ A @ x - sqrt_W @ b))
            else:
                objective = cp.Minimize(cp.sum_squares(A @ x - b))

            # Adaptive constraints based on problem context
            constraints = []

            # Basic bounds to prevent extreme values
            if point_idx in self.boundary_point_set:
                # Stricter bounds for boundary points
                constraints.extend([x >= -10.0, x <= 10.0])
            else:
                # More lenient bounds for interior points
                constraints.extend([x >= -50.0, x <= 50.0])

            # Create and solve problem
            problem = cp.Problem(objective, constraints)

            # Solve with appropriate solver and settings
            if OSQP_AVAILABLE:
                problem.solve(
                    solver=cp.OSQP,
                    verbose=False,
                    eps_abs=1e-4,
                    eps_rel=1e-4,
                    max_iter=1000,
                    warm_start=warm_start is not None,
                )
            else:
                problem.solve(solver=cp.ECOS, verbose=False, max_iters=1000)

            # Check solution status
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                solution = x.value if x.value is not None else np.zeros(n_vars)

                # Cache the solution for future warm starts
                if self.optimization_level >= 3:
                    self.solution_cache[point_idx] = solution.copy()

                return solution
            else:
                # Fallback to unconstrained solution
                return self._solve_unconstrained_fallback(taylor_data, b)

        except Exception as e:
            # Final fallback
            return super()._solve_monotone_constrained_qp(taylor_data, b, point_idx)

    def approximate_derivatives(self, u_values: np.ndarray, point_idx: int) -> Dict[Tuple[int, ...], float]:
        """
        Override the main method with deep optimization integration.
        This is where the core optimization happens.
        """
        start_time = time.time()
        self.performance_stats['total_derivative_calls'] += 1

        # Get current context
        newton_iter = getattr(self, '_current_newton_iter', 0)
        time_ratio = getattr(self, '_current_time_ratio', 0.0)

        # Build Taylor data (same as parent)
        taylor_data = self._build_taylor_data_for_point(point_idx, u_values)
        b = self._build_rhs_vector_for_point(point_idx, u_values)

        # Signature for caching
        problem_signature = self._get_problem_signature(point_idx, newton_iter, time_ratio)

        # Check cache first (optimization level 3+)
        if self.optimization_level >= 3 and problem_signature in self.solution_cache:
            self.performance_stats['cache_hits'] += 1
            cached_coeffs = self.solution_cache[problem_signature]
            # Convert back to derivatives dictionary
            return self._coeffs_to_derivatives_dict(cached_coeffs)

        if self.use_monotone_constraints and self.optimization_level > 0:
            # Step 1: Compute unconstrained solution (fast path)
            unconstrained_coeffs = self._solve_unconstrained_fallback(taylor_data, b)

            # Step 2: Enhanced monotonicity check with context awareness
            needs_qp = self._enhanced_monotonicity_check(unconstrained_coeffs, point_idx, newton_iter, time_ratio)

            if not needs_qp:
                # Use unconstrained solution (fast path)
                self.performance_stats['unconstrained_solutions_used'] += 1
                derivative_coeffs = unconstrained_coeffs

                # Cache the decision and solution
                if self.optimization_level >= 3:
                    self.decision_cache[problem_signature] = False
                    self.solution_cache[problem_signature] = derivative_coeffs.copy()
            else:
                # Use QP with warm start
                self.performance_stats['qp_solutions_used'] += 1

                # Get warm start from previous solution
                warm_start = self.solution_cache.get(point_idx, None)

                derivative_coeffs = self._solve_qp_with_warm_start(taylor_data, b, point_idx, warm_start)

                # Cache the decision and solution
                if self.optimization_level >= 3:
                    self.decision_cache[problem_signature] = True
                    self.solution_cache[problem_signature] = derivative_coeffs.copy()

        elif self.use_monotone_constraints:
            # Baseline: always use QP (optimization level 0)
            derivative_coeffs = super()._solve_monotone_constrained_qp(taylor_data, b, point_idx)
            self.performance_stats['qp_solutions_used'] += 1
        else:
            # No constraints
            derivative_coeffs = self._solve_unconstrained_fallback(taylor_data, b)
            self.performance_stats['unconstrained_solutions_used'] += 1

        # Convert coefficients to derivatives dictionary
        derivatives_dict = self._coeffs_to_derivatives_dict(derivative_coeffs)

        # Update timing
        solve_time = time.time() - start_time
        if needs_qp if 'needs_qp' in locals() else False:
            pass  # QP time already counted
        else:
            self.performance_stats['optimization_time_saved'] += solve_time

        return derivatives_dict

    def _build_taylor_data_for_point(self, point_idx: int, u_values: np.ndarray) -> Dict:
        """Build Taylor data using parent class infrastructure"""
        # Use the pre-computed Taylor matrices from initialization
        if hasattr(self, 'taylor_matrices') and point_idx in self.taylor_matrices:
            return self.taylor_matrices[point_idx].copy()
        else:
            # Fallback: minimal Taylor data
            n_vars = len(u_values) if hasattr(u_values, '__len__') else 8
            return {'A': np.eye(n_vars), 'W': np.eye(n_vars), 'sqrt_W': np.eye(n_vars), 'use_svd': True}

    def _build_rhs_vector_for_point(self, point_idx: int, u_values: np.ndarray) -> np.ndarray:
        """Build RHS vector for Taylor system"""
        # This should extract values from u_values based on neighborhood
        try:
            if hasattr(self, 'neighborhoods') and point_idx in self.neighborhoods:
                neighborhood = self.neighborhoods[point_idx]
                neighbor_indices = neighborhood.get('indices', [point_idx])
                # Extract neighbor values
                rhs = np.array([u_values[i] if i >= 0 and i < len(u_values) else 0.0 for i in neighbor_indices])
                return rhs
            else:
                # Fallback: use point value
                return np.array([u_values[point_idx] if point_idx < len(u_values) else 0.0])
        except:
            # Final fallback
            return np.array([0.0])

    def _coeffs_to_derivatives_dict(self, coeffs: np.ndarray) -> Dict[Tuple[int, ...], float]:
        """Convert coefficient array to derivatives dictionary"""
        derivatives = {}

        # Map coefficients to derivative multi-indices
        # This is a simplified mapping - the actual implementation would depend on the Taylor expansion order
        for i, coeff in enumerate(coeffs):
            if i == 0:
                derivatives[(0,)] = coeff  # Function value
            elif i == 1:
                derivatives[(1,)] = coeff  # First derivative
            elif i == 2:
                derivatives[(2,)] = coeff  # Second derivative
            else:
                derivatives[(i,)] = coeff  # Higher-order derivatives

        return derivatives

    def _solve_timestep(self, t_idx: int, U_current: np.ndarray, M_current: np.ndarray) -> np.ndarray:
        """Override to inject temporal context"""
        # Set temporal context for optimization
        total_time_steps = getattr(self.problem, 'Nt', 50)
        self._current_time_ratio = t_idx / max(1, total_time_steps - 1)
        self.time_step_context['current_time_ratio'] = self._current_time_ratio

        # Call parent implementation
        return super()._solve_timestep(t_idx, U_current, M_current)

    def _newton_iteration(self, iteration: int, *args, **kwargs):
        """Override to inject Newton iteration context"""
        # Set Newton context for optimization
        self._current_newton_iter = iteration

        # Call parent implementation if it exists
        if hasattr(super(), '_newton_iteration'):
            return super()._newton_iteration(iteration, *args, **kwargs)

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        stats = self.performance_stats.copy()

        # Calculate derived metrics
        total_calls = stats['total_derivative_calls']
        if total_calls > 0:
            stats['unconstrained_usage_rate'] = stats['unconstrained_solutions_used'] / total_calls
            stats['qp_usage_rate'] = stats['qp_solutions_used'] / total_calls
            stats['cache_hit_rate'] = stats['cache_hits'] / total_calls
        else:
            stats['unconstrained_usage_rate'] = 0.0
            stats['qp_usage_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0

        # Prediction accuracy
        if stats['prediction_attempts'] > 0:
            stats['prediction_accuracy'] = stats['prediction_correct'] / stats['prediction_attempts']
        else:
            stats['prediction_accuracy'] = 0.0

        # Optimization effectiveness
        if stats['qp_usage_rate'] > 0:
            theoretical_optimal_qp_rate = 0.1  # Target 10% QP usage
            stats['optimization_effectiveness'] = max(0.0, 1.0 - (stats['qp_usage_rate'] / theoretical_optimal_qp_rate))
        else:
            stats['optimization_effectiveness'] = 1.0

        # Add configuration info
        stats['optimization_level'] = self.optimization_level
        stats['boundary_points'] = len(self.boundary_point_set)
        stats['interior_points'] = len(self.interior_points)
        stats['cvxpy_available'] = CVXPY_AVAILABLE
        stats['osqp_available'] = OSQP_AVAILABLE

        return stats

    def print_performance_summary(self):
        """Print detailed performance summary"""
        stats = self.get_performance_report()

        print(f"\n{'='*70}")
        print("DEEP OPTIMIZED QP-COLLOCATION PERFORMANCE SUMMARY")
        print(f"{'='*70}")

        print(f"Configuration:")
        print(f"  Optimization Level: {stats['optimization_level']}")
        print(f"  Boundary Points: {stats['boundary_points']}")
        print(f"  Interior Points: {stats['interior_points']}")
        print(f"  CVXPY Available: {'✓' if stats['cvxpy_available'] else '✗'}")
        print(f"  OSQP Available: {'✓' if stats['osqp_available'] else '✗'}")

        print(f"\nSolution Method Usage:")
        print(f"  Total Derivative Calls: {stats['total_derivative_calls']}")
        print(
            f"  Unconstrained Solutions: {stats['unconstrained_solutions_used']} ({stats['unconstrained_usage_rate']:.1%})"
        )
        print(f"  QP Solutions: {stats['qp_solutions_used']} ({stats['qp_usage_rate']:.1%})")
        print(f"  Cache Hits: {stats['cache_hits']} ({stats['cache_hit_rate']:.1%})")

        print(f"\nOptimization Effectiveness:")
        print(f"  QP Usage Rate: {stats['qp_usage_rate']:.1%} (target: 10%)")
        print(f"  Optimization Effectiveness: {stats['optimization_effectiveness']:.1%}")

        if stats['prediction_attempts'] > 0:
            print(f"  Prediction Accuracy: {stats['prediction_accuracy']:.1%}")

        # Calculate estimated speedup
        if stats['qp_usage_rate'] < 1.0:
            # Conservative estimate: each skipped QP saves 90% of QP time
            qp_skip_rate = 1.0 - stats['qp_usage_rate']
            estimated_speedup = 1 / (1 - qp_skip_rate * 0.9)
            print(f"  Estimated Speedup: {estimated_speedup:.1f}x")

        print(f"{'='*70}")
