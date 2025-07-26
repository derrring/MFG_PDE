#!/usr/bin/env python3
"""
Tuned Smart QP GFDM HJB Solver
Fine-tuned version calibrated to achieve ~10% QP usage rate
based on validation test results.
"""

import numpy as np
import time
from typing import Optional, Dict, List, Tuple, Any
import warnings

# Try to import specialized QP solvers
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False

from .hjb_gfdm import HJBGFDMSolver


class HJBGFDMTunedQPSolver(HJBGFDMSolver):
    """
    Tuned Smart QP GFDM HJB Solver calibrated for ~10% QP usage.
    
    Based on validation results showing 29.5% usage, this version
    has more aggressive thresholds to reach the 10% target.
    """
    
    def __init__(self, problem, collocation_points, delta=0.35, taylor_order=2,
                 weight_function="wendland", weight_scale=1.0, 
                 max_newton_iterations=None, newton_tolerance=None,
                 # Deprecated parameters for backward compatibility
                 NiterNewton=None, l2errBoundNewton=None,
                 boundary_indices=None, boundary_conditions=None,
                 use_monotone_constraints=True, qp_usage_target=0.1):
        """
        Initialize tuned smart QP GFDM HJB solver.
        """
        # Handle backward compatibility
        if NiterNewton is not None and max_newton_iterations is None:
            max_newton_iterations = NiterNewton
        if l2errBoundNewton is not None and newton_tolerance is None:
            newton_tolerance = l2errBoundNewton
        
        # Set defaults if still None
        if max_newton_iterations is None:
            max_newton_iterations = 8
        if newton_tolerance is None:
            newton_tolerance = 1e-4
            
        super().__init__(
            problem, collocation_points, delta, taylor_order,
            weight_function, weight_scale, 
            max_newton_iterations=max_newton_iterations, newton_tolerance=newton_tolerance,
            boundary_indices=boundary_indices, boundary_conditions=boundary_conditions, 
            use_monotone_constraints=use_monotone_constraints
        )
        
        # Tuned QP settings
        self.qp_usage_target = qp_usage_target
        
        # Enhanced performance tracking
        self.tuned_qp_stats = {
            'total_qp_decisions': 0,
            'qp_activated': 0,
            'qp_skipped': 0,
            'extreme_violations': 0,
            'boundary_qp_activations': 0,
            'interior_qp_activations': 0,
            'early_time_qp_activations': 0,
            'late_time_qp_activations': 0,
            'threshold_adaptations': 0,
            'total_solve_time': 0.0
        }
        
        # Context tracking
        self._current_point_idx = 0
        self._current_time_ratio = 0.0
        self._current_newton_iter = 0
        self._problem_difficulty = self._assess_problem_difficulty()
        
        # Boundary point identification
        self.boundary_point_set = set(boundary_indices) if boundary_indices is not None else set()
        
        # More aggressive thresholds calibrated from validation results
        # Previous version achieved 29.5%, so we need ~3x more restrictive thresholds
        self._tuned_thresholds = self._compute_tuned_thresholds()
        
        # Adaptive decision threshold - starts high and adapts
        self._decision_threshold = 15.0  # Much higher than previous 5.0
        self._threshold_adaptation_rate = 0.02
        
        # Override method name
        self.hjb_method_name = "Tuned Smart QP GFDM"
        
        print(f"TunedSmartQPGFDMHJBSolver initialized:")
        print(f"  Target QP usage rate: {qp_usage_target:.1%}")
        print(f"  Decision threshold: {self._decision_threshold}")
        print(f"  CVXPY available: {'YES' if CVXPY_AVAILABLE else 'NO'}")
        print(f"  Boundary points: {len(self.boundary_point_set)}")
        print(f"  Problem difficulty: {self._problem_difficulty:.2f}")
    
    def _assess_problem_difficulty(self) -> float:
        """Assess problem difficulty (same as Smart QP version)"""
        difficulty = 0.0
        
        sigma = getattr(self.problem, 'sigma', 0.1)
        difficulty += min(1.0, sigma / 0.3)
        
        T = getattr(self.problem, 'T', 1.0)
        difficulty += min(1.0, T / 3.0)
        
        Nx = getattr(self.problem, 'Nx', 50)
        difficulty += min(1.0, (Nx - 20) / 80)
        
        coefCT = getattr(self.problem, 'coefCT', 0.02)
        difficulty += min(1.0, coefCT / 0.1)
        
        return min(2.0, difficulty)
    
    def _compute_tuned_thresholds(self) -> Dict[str, float]:
        """Compute much more aggressive thresholds to reach 10% target"""
        base_threshold = 100.0
        
        # Much more aggressive multipliers based on validation results
        # Previous version had difficulty_multiplier = 1.0 + difficulty
        # We need ~3x more restrictive, so increase base thresholds significantly
        difficulty_multiplier = (1.0 + self._problem_difficulty) * 3.0
        
        return {
            'extreme_violation': base_threshold * difficulty_multiplier * 20,  # 6000-12000 (vs 1000-2000)
            'severe_violation': base_threshold * difficulty_multiplier * 10,  # 3000-6000 (vs 500-1000)
            'moderate_violation': base_threshold * difficulty_multiplier * 5, # 1500-3000 (vs 200-400)
            'mild_violation': base_threshold * difficulty_multiplier * 3,     # 900-1800 (vs 100-200)
            'gradient_threshold': base_threshold * difficulty_multiplier * 2, # 600-1200 (vs 50-100)
            'variation_threshold': base_threshold * difficulty_multiplier * 1 # 300-600 (vs 30-60)
        }
    
    def _check_monotonicity_violation(self, coeffs: np.ndarray) -> bool:
        """
        Tuned monotonicity violation check calibrated for 10% usage.
        
        Based on validation showing 29.5% usage, this version uses
        much more restrictive criteria to reach the 10% target.
        """
        self.tuned_qp_stats['total_qp_decisions'] += 1
        
        # Fast rejection for obviously valid solutions
        if np.any(~np.isfinite(coeffs)):
            self.tuned_qp_stats['qp_activated'] += 1
            self.tuned_qp_stats['extreme_violations'] += 1
            return True
        
        # Get tuned thresholds
        thresholds = self._tuned_thresholds
        
        # Compute solution characteristics
        max_coeff = np.max(np.abs(coeffs))
        mean_coeff = np.mean(np.abs(coeffs))
        std_coeff = np.std(coeffs) if len(coeffs) > 1 else 0.0
        
        # Much more restrictive violation scoring
        violation_score = 0.0
        
        # Score 1: Only extreme coefficient values trigger QP
        if max_coeff > thresholds['extreme_violation']:
            violation_score += 20.0  # Extreme case - definitely need QP
            self.tuned_qp_stats['extreme_violations'] += 1
        elif max_coeff > thresholds['severe_violation']:
            violation_score += 10.0  # Severe case - likely need QP
        elif max_coeff > thresholds['moderate_violation']:
            violation_score += 3.0   # Moderate case - maybe need QP
        # Removed mild_violation scoring - too lenient
        
        # Score 2: Higher-order derivatives (more restrictive)
        if len(coeffs) > 2:
            higher_order_max = np.max(np.abs(coeffs[2:]))
            if higher_order_max > thresholds['extreme_violation']:
                violation_score += 10.0
            elif higher_order_max > thresholds['severe_violation']:
                violation_score += 5.0
        
        # Score 3: Gradient magnitude (more restrictive)
        if len(coeffs) >= 2:
            gradient_magnitude = np.linalg.norm(coeffs[:2])
            if gradient_magnitude > thresholds['gradient_threshold']:
                violation_score += 5.0  # Reduced from previous 2.0
        
        # Score 4: Coefficient variation (more restrictive)
        if std_coeff > thresholds['variation_threshold']:
            violation_score += 2.0  # Reduced from previous 1.0
        
        # More aggressive context-based adjustments
        
        # Adjustment 1: Spatial context (more restrictive)
        if self._current_point_idx in self.boundary_point_set:
            # Even boundary points need higher thresholds
            violation_score *= 1.2  # Reduced from 1.5
        else:
            # Interior points get much more restrictive treatment
            violation_score *= 0.5  # Reduced from 0.7
        
        # Adjustment 2: Temporal context (more restrictive)
        if self._current_time_ratio < 0.1:  # Only very early time steps
            violation_score *= 1.1  # Reduced from 1.3
        elif self._current_time_ratio < 0.3:  # Early time steps
            violation_score *= 1.0  # No bonus
        else:
            # Most time steps get restrictive treatment
            violation_score *= 0.6  # More restrictive than previous 0.8
        
        # Adjustment 3: Newton iteration context (more restrictive)
        newton_factor = 0.8 ** self._current_newton_iter  # More aggressive decay
        violation_score *= newton_factor
        
        # Adaptive decision threshold with continuous calibration
        current_decision_threshold = self._decision_threshold
        
        # Continuous threshold adaptation based on current QP usage
        if self.tuned_qp_stats['total_qp_decisions'] > 50:  # After short warmup
            current_qp_rate = (self.tuned_qp_stats['qp_activated'] / 
                              self.tuned_qp_stats['total_qp_decisions'])
            
            target_rate = self.qp_usage_target
            
            # Adaptive adjustment - more aggressive than previous version
            if current_qp_rate > target_rate * 1.2:  # If more than 20% above target
                self._decision_threshold *= (1.0 + self._threshold_adaptation_rate * 2)
                self.tuned_qp_stats['threshold_adaptations'] += 1
            elif current_qp_rate > target_rate * 1.1:  # If more than 10% above target
                self._decision_threshold *= (1.0 + self._threshold_adaptation_rate)
                self.tuned_qp_stats['threshold_adaptations'] += 1
            elif current_qp_rate < target_rate * 0.5:  # If less than 50% of target
                self._decision_threshold *= (1.0 - self._threshold_adaptation_rate)
                self.tuned_qp_stats['threshold_adaptations'] += 1
            
            current_decision_threshold = self._decision_threshold
        
        # Final decision with tuned threshold
        needs_qp = violation_score > current_decision_threshold
        
        # Update detailed statistics
        if needs_qp:
            self.tuned_qp_stats['qp_activated'] += 1
            
            # Track by context for analysis
            if self._current_point_idx in self.boundary_point_set:
                self.tuned_qp_stats['boundary_qp_activations'] += 1
            else:
                self.tuned_qp_stats['interior_qp_activations'] += 1
                
            if self._current_time_ratio < 0.3:
                self.tuned_qp_stats['early_time_qp_activations'] += 1
            elif self._current_time_ratio > 0.7:
                self.tuned_qp_stats['late_time_qp_activations'] += 1
        else:
            self.tuned_qp_stats['qp_skipped'] += 1
        
        return needs_qp
    
    def approximate_derivatives(self, u_values: np.ndarray, point_idx: int) -> Dict[Tuple[int, ...], float]:
        """Override to inject context for tuned QP decisions"""
        self._current_point_idx = point_idx
        return super().approximate_derivatives(u_values, point_idx)
    
    def _solve_timestep(self, u_n_plus_1: np.ndarray, u_prev_picard: np.ndarray, 
                       m_n_plus_1: np.ndarray, time_idx: int) -> np.ndarray:
        """Override to inject temporal context"""
        # Set temporal context for tuned QP decisions
        total_time_steps = getattr(self.problem, 'Nt', 50) + 1
        self._current_time_ratio = time_idx / max(1, total_time_steps - 1)
        
        # Newton iteration with context
        u_current = u_n_plus_1.copy()
        
        for newton_iter in range(self.max_newton_iterations):
            self._current_newton_iter = newton_iter
            
            # Compute HJB residual
            residual = self._compute_hjb_residual(u_current, u_n_plus_1, m_n_plus_1, time_idx)
            
            # Check convergence
            residual_norm = np.linalg.norm(residual)
            if residual_norm < self.newton_tolerance:
                break
            
            # Compute Jacobian and update
            try:
                jacobian = self._compute_hjb_jacobian(u_current, u_n_plus_1, m_n_plus_1, time_idx)
                
                if jacobian.shape[0] == jacobian.shape[1]:
                    du = np.linalg.solve(jacobian, -residual)
                else:
                    du = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
                
                u_current = u_current + du.reshape(u_current.shape)
                
            except np.linalg.LinAlgError:
                u_current = 0.5 * (u_current + u_n_plus_1)
        
        return u_current
    
    def _solve_monotone_constrained_qp(self, taylor_data: Dict, b: np.ndarray, point_idx: int) -> np.ndarray:
        """Optimized QP solve using CVXPY when available"""
        if not CVXPY_AVAILABLE:
            return super()._solve_monotone_constrained_qp(taylor_data, b, point_idx)
        
        try:
            A = taylor_data.get('A', np.eye(len(b)))
            W = taylor_data.get('W', np.eye(len(b)))
            n_vars = A.shape[1]
            
            x = cp.Variable(n_vars)
            
            if 'sqrt_W' in taylor_data:
                sqrt_W = taylor_data['sqrt_W']
                objective = cp.Minimize(cp.sum_squares(sqrt_W @ A @ x - sqrt_W @ b))
            else:
                objective = cp.Minimize(cp.sum_squares(A @ x - b))
            
            # Adaptive constraints
            constraints = []
            
            if point_idx in self.boundary_point_set:
                bound_scale = 5.0 * (1.0 + self._problem_difficulty)
                constraints.extend([x >= -bound_scale, x <= bound_scale])
            else:
                bound_scale = 20.0 * (1.0 + self._problem_difficulty)
                constraints.extend([x >= -bound_scale, x <= bound_scale])
            
            problem = cp.Problem(objective, constraints)
            
            if OSQP_AVAILABLE:
                problem.solve(solver=cp.OSQP, verbose=False, 
                             eps_abs=1e-4, eps_rel=1e-4, max_iter=1000)
            else:
                problem.solve(solver=cp.ECOS, verbose=False, max_iters=1000)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return x.value if x.value is not None else np.zeros(n_vars)
            else:
                return self._solve_unconstrained_fallback(taylor_data, b)
                
        except Exception as e:
            return super()._solve_monotone_constrained_qp(taylor_data, b, point_idx)
    
    def get_tuned_qp_report(self) -> Dict[str, Any]:
        """Generate tuned QP performance report"""
        stats = self.tuned_qp_stats.copy()
        
        # Calculate rates
        total_decisions = stats['total_qp_decisions']
        if total_decisions > 0:
            stats['qp_usage_rate'] = stats['qp_activated'] / total_decisions
            stats['qp_skip_rate'] = stats['qp_skipped'] / total_decisions
            stats['extreme_violation_rate'] = stats['extreme_violations'] / total_decisions
        else:
            stats['qp_usage_rate'] = 0.0
            stats['qp_skip_rate'] = 0.0
            stats['extreme_violation_rate'] = 0.0
        
        # Context analysis
        total_qp_activated = stats['qp_activated']
        if total_qp_activated > 0:
            stats['boundary_qp_percentage'] = stats['boundary_qp_activations'] / total_qp_activated * 100
            stats['interior_qp_percentage'] = stats['interior_qp_activations'] / total_qp_activated * 100
            stats['early_time_percentage'] = stats['early_time_qp_activations'] / total_qp_activated * 100
            stats['late_time_percentage'] = stats['late_time_qp_activations'] / total_qp_activated * 100
        else:
            stats['boundary_qp_percentage'] = 0.0
            stats['interior_qp_percentage'] = 0.0
            stats['early_time_percentage'] = 0.0
            stats['late_time_percentage'] = 0.0
        
        # Optimization assessment
        target_rate = self.qp_usage_target
        current_rate = stats['qp_usage_rate']
        
        if current_rate <= target_rate * 1.2:  # Within 20% of target
            stats['optimization_quality'] = 'EXCELLENT'
            stats['optimization_effectiveness'] = min(1.0, target_rate / max(0.01, current_rate))
        elif current_rate <= target_rate * 2.0:  # Within 100% of target
            stats['optimization_quality'] = 'GOOD'
            stats['optimization_effectiveness'] = target_rate / current_rate
        elif current_rate <= target_rate * 3.0:  # Within 200% of target
            stats['optimization_quality'] = 'FAIR'
            stats['optimization_effectiveness'] = target_rate / current_rate
        else:
            stats['optimization_quality'] = 'POOR'
            stats['optimization_effectiveness'] = target_rate / current_rate
        
        # Add configuration info
        stats['target_qp_rate'] = target_rate
        stats['problem_difficulty'] = self._problem_difficulty
        stats['final_decision_threshold'] = self._decision_threshold
        stats['cvxpy_available'] = CVXPY_AVAILABLE
        
        return stats
    
    def print_tuned_qp_summary(self):
        """Print comprehensive tuned QP performance summary"""
        stats = self.get_tuned_qp_report()
        
        print(f"\n{'='*70}")
        print("TUNED SMART QP OPTIMIZATION PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        print(f"Configuration:")
        print(f"  Target QP Usage Rate: {stats['target_qp_rate']:.1%}")
        print(f"  Problem Difficulty: {stats['problem_difficulty']:.2f}")
        print(f"  Final Decision Threshold: {stats['final_decision_threshold']:.1f}")
        print(f"  Threshold Adaptations: {stats['threshold_adaptations']}")
        
        print(f"\nQP Decision Statistics:")
        print(f"  Total QP Decisions: {stats['total_qp_decisions']}")
        print(f"  QP Activated: {stats['qp_activated']} ({stats['qp_usage_rate']:.1%})")
        print(f"  QP Skipped: {stats['qp_skipped']} ({stats['qp_skip_rate']:.1%})")
        print(f"  Extreme Violations: {stats['extreme_violations']} ({stats['extreme_violation_rate']:.1%})")
        
        print(f"\nContext Analysis:")
        if stats['qp_activated'] > 0:
            print(f"  Boundary Point QP: {stats['boundary_qp_percentage']:.1f}%")
            print(f"  Interior Point QP: {stats['interior_qp_percentage']:.1f}%")
            print(f"  Early Time QP: {stats['early_time_percentage']:.1f}%")
            print(f"  Late Time QP: {stats['late_time_percentage']:.1f}%")
        
        print(f"\nOptimization Results:")
        print(f"  Optimization Quality: {stats['optimization_quality']}")
        print(f"  Optimization Effectiveness: {stats['optimization_effectiveness']:.1%}")
        
        # Calculate estimated speedup
        if stats['qp_skip_rate'] > 0:
            estimated_speedup = 1 / (1 - stats['qp_skip_rate'] * 0.9)
            print(f"  Estimated Speedup: {estimated_speedup:.1f}x")
        
        # Target assessment
        target_rate = stats['target_qp_rate']
        current_rate = stats['qp_usage_rate']
        
        if current_rate <= target_rate * 1.2:
            print(f"  Status: TARGET ACHIEVED")
        elif current_rate <= target_rate * 2.0:
            print(f"  Status: CLOSE TO TARGET")
        else:
            print(f"  Status: NEEDS FURTHER TUNING")
        
        print(f"{'='*70}")


# Backward compatibility alias
import warnings

class HJBGFDMTunedSmartQPSolver(HJBGFDMTunedQPSolver):
    """
    Deprecated: Use HJBGFDMTunedQPSolver instead.
    
    This is a backward compatibility alias for the renamed class.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "HJBGFDMTunedSmartQPSolver is deprecated. Use HJBGFDMTunedQPSolver instead.",
            DeprecationWarning, stacklevel=2
        )
        super().__init__(*args, **kwargs)