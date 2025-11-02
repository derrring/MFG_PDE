#!/usr/bin/env python3
"""
High-Level MFG Solver Interface

Provides a simple, one-line interface for solving MFG problems with automatic
configuration and method selection.

Phase 3.3 Update (Phase 3.2 SolverConfig Integration)
------------------------------------------------------
This module now supports the new Phase 3.2 SolverConfig system:
- Accept SolverConfig instances or string preset names
- Support both new unified API and legacy method parameter
- Provide clear migration path

Example (New API - Recommended):
    >>> from mfg_pde import MFGProblem, solve_mfg
    >>> from mfg_pde.config import presets
    >>>
    >>> problem = MFGProblem(...)
    >>> result = solve_mfg(problem, config=presets.accurate_solver())

Example (Legacy API - Still Works):
    >>> from mfg_pde import ExampleMFGProblem, solve_mfg
    >>>
    >>> problem = ExampleMFGProblem()
    >>> result = solve_mfg(problem, method="accurate", resolution=50)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mfg_pde.config import SolverConfig
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.utils.solver_result import SolverResult

SolverMethod = Literal["auto", "fast", "accurate", "research"]


def solve_mfg(
    problem: MFGProblem,
    config: SolverConfig | str | None = None,
    method: SolverMethod | None = None,
    resolution: int | None = None,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> SolverResult:
    """
    Solve an MFG problem with automatic method selection and configuration.

    This is a high-level convenience function that combines solver creation
    and solving into a single call.

    Parameters
    ----------
    problem : MFGProblem
        MFG problem instance to solve
    config : SolverConfig, str, or None
        **NEW (Phase 3.2/3.3)**: Solver configuration
        - SolverConfig: Use provided configuration
        - str: Use named preset ("fast", "accurate", "research", "production")
        - None: Auto-select based on problem (equivalent to "fast")

        Examples:
            >>> from mfg_pde.config import presets
            >>> result = solve_mfg(problem, config=presets.accurate_solver())
            >>> result = solve_mfg(problem, config="accurate")
            >>> result = solve_mfg(problem, config="research")
    method : str, optional
        **DEPRECATED**: Use `config` parameter instead
        - "auto"/"fast": Fast hybrid solver
        - "accurate": High precision
        - "research": Comprehensive diagnostics

        Will be removed in v2.0.0. Use `config` parameter instead.
    resolution : int, optional
        Grid resolution or number of particles
        - 1D: Typically 50-200 grid points
        - 2D: Typically 30-100 points per dimension
        - If None, automatic selection based on problem dimension
    max_iterations : int, optional
        Maximum iterations for fixed-point solver
        If specified, overrides config setting
    tolerance : float, optional
        Convergence tolerance
        If specified, overrides config setting
    verbose : bool, default=True
        Print solver progress and diagnostics
    **kwargs : dict
        Additional solver-specific parameters
        - damping_factor: Damping parameter for fixed-point
        - use_anderson: Enable Anderson acceleration
        - backend: Computational backend
        - hjb_method: Override HJB solver method
        - fp_method: Override FP solver method

    Returns
    -------
    SolverResult
        Result with attributes:
        - U: Value function array
        - M: Density array
        - iterations: Number of iterations performed
        - error_history_U: Convergence history for U
        - error_history_M: Convergence history for M
        - converged: Whether convergence was achieved
        - execution_time: Total solve time in seconds
        - metadata: Additional solver-specific information

    Raises
    ------
    ValueError
        If problem is invalid or method/config is unknown
    RuntimeError
        If solver fails to converge

    Examples
    --------
    **New Unified API (Recommended)**:

    >>> from mfg_pde import MFGProblem, solve_mfg
    >>> from mfg_pde.config import presets
    >>>
    >>> # With preset object
    >>> problem = MFGProblem(...)
    >>> result = solve_mfg(problem, config=presets.accurate_solver())
    >>>
    >>> # With preset name
    >>> result = solve_mfg(problem, config="accurate")
    >>>
    >>> # With YAML config
    >>> from mfg_pde.config import load_solver_config
    >>> config = load_solver_config("experiment.yaml")
    >>> result = solve_mfg(problem, config=config)
    >>>
    >>> # With Builder API
    >>> from mfg_pde.config import ConfigBuilder
    >>> config = (
    ...     ConfigBuilder()
    ...     .solver_hjb(method="fdm", accuracy_order=2)
    ...     .solver_fp_particle(num_particles=5000)
    ...     .picard(max_iterations=100, tolerance=1e-6)
    ...     .build()
    ... )
    >>> result = solve_mfg(problem, config=config)

    **Legacy API (Deprecated but Still Works)**:

    >>> from mfg_pde import ExampleMFGProblem, solve_mfg
    >>>
    >>> # Simple usage with defaults
    >>> problem = ExampleMFGProblem()
    >>> result = solve_mfg(problem, method="fast")
    >>>
    >>> # High-accuracy solve
    >>> result = solve_mfg(
    ...     problem,
    ...     method="accurate",
    ...     resolution=100,
    ...     max_iterations=1000,
    ...     tolerance=1e-8
    ... )

    Notes
    -----
    - Automatic method selection (config=None) chooses fast hybrid solver
    - Resolution defaults: 1D: 100 points, 2D: 50x50 points, 3D: 30x30x30 points
    - For maximum control, use Phase 3.2 SolverConfig with Builder or YAML
    - The `method` parameter is deprecated and will be removed in v2.0.0
    """
    from mfg_pde.config import SolverConfig, presets
    from mfg_pde.factory import create_accurate_solver, create_research_solver, create_standard_solver

    # =========================================================================
    # Parameter Handling and Deprecation Warnings
    # =========================================================================

    # Handle deprecated method parameter
    if method is not None and config is None:
        warnings.warn(
            f"The 'method' parameter is deprecated. "
            f"Use 'config' parameter instead:\n"
            f"  Old: solve_mfg(problem, method='{method}')\n"
            f"  New: solve_mfg(problem, config='{method}')\n"
            f"The 'method' parameter will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        config = method

    # Default config
    if config is None:
        config = "fast"

    # =========================================================================
    # Config Resolution
    # =========================================================================

    # Convert string to SolverConfig
    if isinstance(config, str):
        preset_map = {
            "auto": "fast",
            "fast": "fast",
            "accurate": "accurate",
            "research": "research",
            "production": "production",
        }

        preset_name = preset_map.get(config, config)

        try:
            # Get preset function from presets module
            config = getattr(presets, f"{preset_name}_solver")()
        except AttributeError:
            valid_presets = list(preset_map.keys())
            raise ValueError(
                f"Unknown config preset: '{config}'\n\n"
                f"Valid presets are: {valid_presets}\n\n"
                f"Example:\n"
                f"  solve_mfg(problem, config='accurate')\n"
                f"  solve_mfg(problem, config=presets.accurate_solver())"
            ) from None

    # At this point, config should be a SolverConfig instance
    if not isinstance(config, SolverConfig):
        raise TypeError(
            f"config must be a SolverConfig instance, string preset name, or None. Got {type(config).__name__}"
        )

    # =========================================================================
    # Parameter Overrides
    # =========================================================================

    # Determine automatic resolution if not specified
    if resolution is None:
        dimension = getattr(problem, "dimension", getattr(problem, "d", 1))
        if dimension == 1:
            resolution = 100
        elif dimension == 2:
            resolution = 50
        else:  # 3D
            resolution = 30

    # Apply overrides to config if specified
    if max_iterations is not None or tolerance is not None:
        # Create modified config with overrides
        import copy

        config = copy.deepcopy(config)

        if max_iterations is not None and config.picard is not None:
            config.picard.max_iterations = max_iterations

        if tolerance is not None:
            if config.picard is not None:
                config.picard.tolerance = tolerance
            config.convergence_tolerance = tolerance

    # =========================================================================
    # Solver Creation and Execution
    # =========================================================================

    # Determine which factory to use based on config
    # For now, we'll use the existing factory functions
    # TODO: In future, replace with unified solver factory that accepts SolverConfig directly

    # Extract method hint from config for factory selection
    if hasattr(config, "metadata") and config.metadata:
        method_hint = config.metadata.get("preset_type", "fast")
    else:
        method_hint = "fast"  # Default

    # Select factory function based on method hint
    if method_hint in ["auto", "fast"]:
        solver_factory = create_standard_solver
    elif method_hint == "accurate":
        solver_factory = create_accurate_solver
    else:  # research
        solver_factory = create_research_solver

    # For research solver, create default hjb_solver and fp_solver if not provided
    if method_hint == "research" and "hjb_solver" not in kwargs and "fp_solver" not in kwargs:
        from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
        from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        kwargs["hjb_solver"] = HJBFDMSolver(problem=problem)
        kwargs["fp_solver"] = FPParticleSolver(problem=problem, num_particles=5000)

    # Convert backend string to backend object if provided
    if "backend" in kwargs and isinstance(kwargs["backend"], str):
        from mfg_pde.backends import create_backend

        kwargs["backend"] = create_backend(kwargs["backend"])

    # Create solver
    # Note: We pass custom_config for backward compatibility
    # In future, factories should accept SolverConfig directly
    try:
        solver = solver_factory(problem=problem, custom_config=config, **kwargs)
    except TypeError:
        # Fallback for factories that don't support custom_config yet
        solver = solver_factory(problem=problem, **kwargs)

    # Solve
    result = solver.solve(verbose=verbose)

    return result


# Public API
__all__ = ["solve_mfg"]
