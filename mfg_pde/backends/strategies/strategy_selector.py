"""
Intelligent strategy selection for particle-based FP solvers.

This module implements automatic selection of optimal computational strategy
based on backend capabilities, problem size, and runtime profiling.
"""

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from .particle_strategies import CPUParticleStrategy, GPUParticleStrategy, HybridParticleStrategy, ParticleStrategy

if TYPE_CHECKING:
    from mfg_pde.backends.base_backend import BaseBackend


class ProfilingMode(Enum):
    """
    Profiling and logging mode for strategy selection.

    Modes:
        DISABLED: No profiling or logging
        SILENT: Profile and log selection history, but don't print to console
        VERBOSE: Profile, log, and print selection decisions to console

    Examples:
        >>> selector = StrategySelector(profiling_mode=ProfilingMode.VERBOSE)
        >>> selector = StrategySelector(profiling_mode=ProfilingMode.SILENT)
        >>> selector = StrategySelector(profiling_mode=ProfilingMode.DISABLED)
    """

    DISABLED = auto()  # No profiling
    SILENT = auto()  # Profiling without console output
    VERBOSE = auto()  # Profiling with console output


class StrategySelector:
    """
    Intelligent strategy selection based on backend capabilities and problem size.

    This class implements the core logic for automatic strategy selection,
    enabling elegant separation of concerns between solver logic and
    backend-specific implementation details.

    Examples
    --------
    >>> from mfg_pde.backends.torch_backend import TorchBackend
    >>> selector = StrategySelector()
    >>> backend = TorchBackend(device="mps")
    >>> strategy = selector.select_strategy(
    ...     backend,
    ...     problem_size=(50000, 50, 50),
    ...     strategy_hint="auto"
    ... )
    >>> print(strategy.name)
    'gpu-mps'
    """

    def __init__(
        self,
        profiling_mode: ProfilingMode | str = ProfilingMode.SILENT,
    ):
        """
        Initialize strategy selector.

        Parameters
        ----------
        profiling_mode : ProfilingMode or str, default=ProfilingMode.SILENT
            Profiling mode: ProfilingMode.DISABLED, .SILENT, or .VERBOSE
            Can also pass strings: "disabled", "silent", "verbose"

        Examples
        --------
        >>> selector = StrategySelector(profiling_mode=ProfilingMode.VERBOSE)
        >>> selector = StrategySelector(profiling_mode="silent")
        """
        # Handle string mode
        if isinstance(profiling_mode, str):
            mode_map = {
                "disabled": ProfilingMode.DISABLED,
                "silent": ProfilingMode.SILENT,
                "verbose": ProfilingMode.VERBOSE,
            }
            profiling_mode = mode_map.get(profiling_mode.lower(), ProfilingMode.SILENT)

        self.profiling_mode = profiling_mode
        # Maintain internal boolean state for compatibility with existing code
        self.enable_profiling = profiling_mode != ProfilingMode.DISABLED
        self.verbose = profiling_mode == ProfilingMode.VERBOSE
        self.selection_history: list[dict] = []

    def select_strategy(
        self,
        backend: Optional["BaseBackend"],
        problem_size: tuple[int, int, int],
        strategy_hint: str | None = None,
    ) -> ParticleStrategy:
        """
        Select optimal strategy based on context.

        Parameters
        ----------
        backend : BaseBackend, optional
            Backend to use for GPU strategies (None for CPU-only)
        problem_size : tuple
            (num_particles, grid_size, time_steps)
        strategy_hint : str, optional
            User override: "cpu", "gpu", "hybrid", "auto" (default)

        Returns
        -------
        ParticleStrategy
            Optimal strategy for the given context

        Examples
        --------
        Automatic selection:
        >>> selector = StrategySelector()
        >>> strategy = selector.select_strategy(backend, (50000, 50, 50))

        Manual override:
        >>> strategy = selector.select_strategy(backend, (5000, 50, 50), strategy_hint="cpu")
        """
        # User override
        if strategy_hint in ("cpu", "gpu", "hybrid"):
            return self._create_strategy(strategy_hint, backend)

        # No backend available - must use CPU
        if backend is None:
            return CPUParticleStrategy()

        # Auto-selection based on capabilities and problem size
        return self._auto_select(backend, problem_size)

    def _auto_select(self, backend: "BaseBackend", problem_size: tuple[int, int, int]) -> ParticleStrategy:
        """
        Automatically select best strategy based on intelligent analysis.

        Selection logic:
        1. Check if backend supports GPU operations
        2. Estimate costs for available strategies
        3. Select minimum cost strategy
        4. Log decision if profiling enabled
        """
        import contextlib

        N, _Nx, _Nt = problem_size

        # Create candidate strategies
        strategies: dict[str, ParticleStrategy] = {
            "cpu": CPUParticleStrategy(),
        }

        # Add GPU strategy if backend supports parallel KDE
        if backend.has_capability("parallel_kde"):
            with contextlib.suppress(ValueError):
                strategies["gpu"] = GPUParticleStrategy(backend)

        # Add hybrid strategy for medium-sized problems on MPS
        if "gpu" in strategies:
            hints = backend.get_performance_hints()
            if hints["device_type"] == "mps" and 10000 <= N <= 100000:
                with contextlib.suppress(ValueError):
                    strategies["hybrid"] = HybridParticleStrategy(backend)

        # Estimate cost for each strategy
        costs = {name: strategy.estimate_cost(problem_size) for name, strategy in strategies.items()}

        # Select minimum cost strategy
        best_name = min(costs, key=costs.get)
        best_strategy = strategies[best_name]

        # Log selection
        if self.enable_profiling:
            self._log_selection(best_name, costs, problem_size, backend)

        return best_strategy

    def _create_strategy(self, name: str, backend: Optional["BaseBackend"]) -> ParticleStrategy:
        """
        Factory method for strategy creation with validation.

        Parameters
        ----------
        name : str
            Strategy name: "cpu", "gpu", or "hybrid"
        backend : BaseBackend, optional
            Backend for GPU/hybrid strategies

        Returns
        -------
        ParticleStrategy
            Created strategy instance

        Raises
        ------
        ValueError
            If strategy requires backend but None provided
        """
        if name == "cpu":
            return CPUParticleStrategy()
        elif name == "gpu":
            if backend is None:
                raise ValueError("GPU strategy requires backend (got None)")
            return GPUParticleStrategy(backend)
        elif name == "hybrid":
            if backend is None:
                raise ValueError("Hybrid strategy requires backend (got None)")
            return HybridParticleStrategy(backend)
        else:
            raise ValueError(f"Unknown strategy: {name}. Use 'cpu', 'gpu', or 'hybrid'")

    def _log_selection(self, selected: str, costs: dict, problem_size: tuple[int, int, int], backend: "BaseBackend"):
        """
        Log strategy selection for debugging and profiling.

        Parameters
        ----------
        selected : str
            Selected strategy name
        costs : dict
            Estimated costs for all strategies
        problem_size : tuple
            (num_particles, grid_size, time_steps)
        backend : BaseBackend
            Backend used for selection
        """
        N, Nx, Nt = problem_size
        hints = backend.get_performance_hints()

        selection_info = {
            "problem_size": {"N": N, "Nx": Nx, "Nt": Nt},
            "backend": backend.name,
            "device_type": hints["device_type"],
            "costs": costs,
            "selected": selected,
        }

        self.selection_history.append(selection_info)

        if self.verbose:
            print(f"[StrategySelector] Problem: N={N}, Nx={Nx}, Nt={Nt}")
            print(f"[StrategySelector] Backend: {backend.name} ({hints['device_type']})")
            print(f"[StrategySelector] Estimated costs: {costs}")
            print(f"[StrategySelector] Selected: {selected}")

    def get_selection_history(self) -> list[dict]:
        """
        Get history of strategy selections for analysis.

        Returns
        -------
        list of dict
            Selection history with problem sizes, costs, and decisions
        """
        return self.selection_history.copy()

    def clear_history(self):
        """Clear selection history."""
        self.selection_history.clear()


class AdaptiveStrategySelector(StrategySelector):
    """
    Adaptive strategy selector that learns from runtime performance.

    This advanced selector tracks actual execution times and adapts
    cost models based on observed performance. Future enhancement for
    production deployment.

    NOTE: This is a placeholder for future implementation.
    """

    def __init__(
        self,
        profiling_mode: ProfilingMode | str = ProfilingMode.SILENT,
    ):
        super().__init__(profiling_mode)
        self.performance_history: dict[str, list[float]] = {}

    def record_performance(self, strategy_name: str, problem_size: tuple, elapsed_time: float):
        """
        Record actual performance for learning.

        Parameters
        ----------
        strategy_name : str
            Strategy that was executed
        problem_size : tuple
            (num_particles, grid_size, time_steps)
        elapsed_time : float
            Actual execution time in seconds
        """
        key = f"{strategy_name}_{problem_size}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        self.performance_history[key].append(elapsed_time)

    def _select_from_history(self, backend: "BaseBackend", problem_size: tuple):
        """
        Select strategy based on historical performance data.

        NOTE: This is a placeholder for future machine learning implementation.
        Could implement:
        - Gaussian process regression for cost prediction
        - Thompson sampling for exploration/exploitation
        - Neural network for pattern recognition
        """
        # For now, fall back to standard selection
        return self._auto_select(backend, problem_size)
