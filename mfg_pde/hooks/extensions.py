"""
Algorithm Extension Points for MFG Solvers

This module provides hooks and mechanisms for extending and customizing
core algorithmic components while maintaining the clean solver interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import SolverHooks

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.types import SpatialTemporalState


class AlgorithmExtensionHook(SolverHooks):
    """
    Base class for algorithm extension hooks.

    This allows users to replace or modify specific algorithmic components
    without having to subclass the entire solver.

    Example:
        class CustomHJBHook(AlgorithmExtensionHook):
            def custom_hjb_step(self, u_current, m_current, problem, metadata):
                # Custom HJB solver implementation
                return modified_u

        solver.solve(problem, hooks=CustomHJBHook())
    """

    def __init__(self) -> None:
        self.extensions: dict[str, Callable] = {}

    def register_extension(self, extension_point: str, implementation: Callable) -> None:
        """Register a custom implementation for an extension point."""
        self.extensions[extension_point] = implementation

    def get_extension(self, extension_point: str) -> Callable | None:
        """Get custom implementation for an extension point."""
        return self.extensions.get(extension_point)

    def has_extension(self, extension_point: str) -> bool:
        """Check if custom implementation is registered."""
        return extension_point in self.extensions


class CustomHJBHook(AlgorithmExtensionHook):
    """
    Hook for customizing HJB equation solving step.

    Example:
        def my_hjb_solver(u_current, m_current, problem, x_grid, t_grid, dt, dx):
            # Custom HJB implementation
            u_new = custom_semi_lagrangian_step(u_current, m_current, problem, dt, dx)
            return u_new

        hjb_hook = CustomHJBHook(hjb_implementation=my_hjb_solver)
        result = solver.solve(problem, hooks=hjb_hook)
    """

    def __init__(self, hjb_implementation: Callable | None = None):
        super().__init__()
        if hjb_implementation:
            self.register_extension("hjb_step", hjb_implementation)

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        """Provide HJB customization before iteration begins."""
        # This would be called by a solver that supports HJB extensions


class CustomFPHook(AlgorithmExtensionHook):
    """
    Hook for customizing Fokker-Planck equation solving step.

    Example:
        def my_fp_solver(m_current, u_new, problem, x_grid, t_grid, dt, dx):
            # Custom FP implementation
            m_new = custom_upwind_step(m_current, u_new, problem, dt, dx)
            return m_new

        fp_hook = CustomFPHook(fp_implementation=my_fp_solver)
        result = solver.solve(problem, hooks=fp_hook)
    """

    def __init__(self, fp_implementation: Callable | None = None):
        super().__init__()
        if fp_implementation:
            self.register_extension("fp_step", fp_implementation)


class CustomConvergenceHook(AlgorithmExtensionHook):
    """
    Hook for customizing convergence checking algorithm.

    Example:
        def my_convergence_check(state, tolerance, history):
            # Custom convergence logic
            return state.residual < tolerance and len(history) > 10

        conv_hook = CustomConvergenceHook(convergence_implementation=my_convergence_check)
        result = solver.solve(problem, hooks=conv_hook)
    """

    def __init__(self, convergence_implementation: Callable | None = None):
        super().__init__()
        if convergence_implementation:
            self.register_extension("convergence_check", convergence_implementation)

    def on_convergence_check(self, state: SpatialTemporalState) -> bool | None:
        """Apply custom convergence check if available."""
        convergence_impl = self.get_extension("convergence_check")
        if convergence_impl:
            try:
                # Call custom convergence check
                # Note: This would need to be integrated with solver state
                return convergence_impl(state, getattr(self, "tolerance", 1e-6), getattr(self, "history", []))
            except Exception as e:
                print(f"Custom convergence check failed: {e}")
                return None
        return None


class PreprocessingHook(SolverHooks):
    """
    Hook for preprocessing solution state at each iteration.

    This allows users to apply transformations, filtering, or other
    preprocessing steps to the solution data.

    Example:
        def smooth_solution(state):
            # Apply smoothing to u and m
            from scipy.ndimage import gaussian_filter
            smoothed_u = gaussian_filter(state.u, sigma=0.5)
            smoothed_m = gaussian_filter(state.m, sigma=0.5)
            return state.copy_with_updates(u=smoothed_u, m=smoothed_m)

        preproc = PreprocessingHook(preprocessing_func=smooth_solution)
        result = solver.solve(problem, hooks=preproc)
    """

    def __init__(self, preprocessing_func: Callable | None = None):
        self.preprocessing_func = preprocessing_func

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        """Apply preprocessing to solution state."""
        if self.preprocessing_func:
            try:
                # Note: This would require solver integration to actually modify state
                self.preprocessing_func(state)
                # The solver would need to support state modification
                print(f"Preprocessing applied at iteration {state.iteration}")
            except Exception as e:
                print(f"Preprocessing failed: {e}")


class PostprocessingHook(SolverHooks):
    """
    Hook for postprocessing solution state after each iteration.

    Example:
        def enforce_constraints(state):
            # Ensure density is non-negative and normalized
            import numpy as np
            m_corrected = np.maximum(state.m, 0)
            # Normalize each time slice
            for t in range(m_corrected.shape[0]):
                total = np.trapezoid(m_corrected[t, :], axis=0)
                if total > 0:
                    m_corrected[t, :] /= total
            return state.copy_with_updates(m=m_corrected)

        postproc = PostprocessingHook(postprocessing_func=enforce_constraints)
        result = solver.solve(problem, hooks=postproc)
    """

    def __init__(self, postprocessing_func: Callable | None = None):
        self.postprocessing_func = postprocessing_func

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Apply postprocessing to solution state."""
        if self.postprocessing_func:
            try:
                self.postprocessing_func(state)
                print(f"Postprocessing applied at iteration {state.iteration}")
            except Exception as e:
                print(f"Postprocessing failed: {e}")
        return None


class CustomResidualHook(SolverHooks):
    """
    Hook for customizing residual calculation.

    Example:
        def my_residual(u, m, u_prev, m_prev):
            # Custom residual calculation
            import numpy as np
            u_residual = np.sqrt(np.mean((u - u_prev)**2))
            m_residual = np.sqrt(np.mean((m - m_prev)**2))
            return max(u_residual, m_residual)

        residual_hook = CustomResidualHook(residual_func=my_residual)
        result = solver.solve(problem, hooks=residual_hook)
    """

    def __init__(self, residual_func: Callable | None = None):
        self.residual_func = residual_func
        self.previous_state: SpatialTemporalState | None = None

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Calculate custom residual if function provided."""
        if self.residual_func and self.previous_state:
            try:
                custom_residual = self.residual_func(state.u, state.m, self.previous_state.u, self.previous_state.m)
                print(f"Custom residual at iteration {state.iteration}: {custom_residual:.6e}")
                # Note: In a full implementation, this would update the state's residual
            except Exception as e:
                print(f"Custom residual calculation failed: {e}")

        # Store current state for next iteration
        self.previous_state = state
        return None


class AdaptiveParameterHook(SolverHooks):
    """
    Hook for adaptive parameter adjustment during solving.

    This allows dynamic modification of solver parameters based on
    runtime conditions and solution progress.

    Example:
        def adapt_damping(iteration, residual, current_damping):
            # Increase damping if residual is increasing
            if iteration > 10 and residual > prev_residual:
                return min(current_damping * 1.1, 0.95)
            return current_damping

        adaptive = AdaptiveParameterHook()
        adaptive.add_parameter_rule('damping_factor', adapt_damping)
        result = solver.solve(problem, hooks=adaptive)
    """

    def __init__(self) -> None:
        self.parameter_rules: dict[str, Callable] = {}
        self.parameter_history: dict[str, list[float]] = {}
        self.residual_history: list[float] = []

    def add_parameter_rule(self, parameter_name: str, adaptation_func: Callable) -> None:
        """Add an adaptation rule for a parameter."""
        self.parameter_rules[parameter_name] = adaptation_func
        self.parameter_history[parameter_name] = []

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Apply parameter adaptation rules."""
        self.residual_history.append(state.residual)

        for param_name, adaptation_func in self.parameter_rules.items():
            try:
                current_value = self.parameter_history[param_name][-1] if self.parameter_history[param_name] else 1.0
                new_value = adaptation_func(state.iteration, state.residual, current_value, self.residual_history)
                self.parameter_history[param_name].append(new_value)
                print(f"Adapted {param_name}: {current_value:.6f} -> {new_value:.6f}")

                # Note: In a full implementation, this would actually update solver parameters

            except Exception as e:
                print(f"Parameter adaptation failed for {param_name}: {e}")

        return None


class MethodSwitchingHook(SolverHooks):
    """
    Hook for switching between different solution methods during solving.

    This allows automatic or manual switching between methods based on
    performance metrics or user-defined criteria.

    Example:
        switcher = MethodSwitchingHook()
        switcher.add_switch_rule(
            condition=lambda state: state.residual > 1e-2 and state.iteration > 50,
            new_method="newton",
            reason="Switch to Newton for better convergence"
        )
        result = solver.solve(problem, hooks=switcher)
    """

    def __init__(self) -> None:
        self.switch_rules: list[dict[str, Any]] = []
        self.current_method = "default"
        self.method_history: list[str] = []

    def add_switch_rule(
        self, condition: Callable[[SpatialTemporalState], bool], new_method: str, reason: str = "Method switch"
    ) -> None:
        """Add a rule for method switching."""
        self.switch_rules.append({"condition": condition, "new_method": new_method, "reason": reason})

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Check method switching conditions."""
        for rule in self.switch_rules:
            if rule["condition"](state) and self.current_method != rule["new_method"]:
                print(f"Switching method: {self.current_method} -> {rule['new_method']}")
                print(f"Reason: {rule['reason']}")

                self.current_method = rule["new_method"]
                self.method_history.append(self.current_method)

                # Note: In a full implementation, this would trigger actual method switching
                # This might require solver support for dynamic method switching

                return None  # Could return "restart" to restart with new method

        return None


class CustomInitializationHook(SolverHooks):
    """
    Hook for custom initialization of solution state.

    Example:
        def custom_init(problem, default_u, default_m):
            # Initialize with problem-specific guess
            u_guess = compute_analytical_approximation(problem)
            m_guess = compute_steady_state_density(problem)
            return u_guess, m_guess

        init_hook = CustomInitializationHook(initialization_func=custom_init)
        result = solver.solve(problem, hooks=init_hook)
    """

    def __init__(self, initialization_func: Callable | None = None):
        self.initialization_func = initialization_func

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """Apply custom initialization if provided."""
        if self.initialization_func:
            try:
                # Note: This would require solver integration to actually modify initial state
                print("Applying custom initialization")
                # custom_u, custom_m = self.initialization_func(problem, initial_state.u, initial_state.m)
                # The solver would need to support state modification
            except Exception as e:
                print(f"Custom initialization failed: {e}")
