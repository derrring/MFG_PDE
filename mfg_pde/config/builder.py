"""
Fluent builder API for solver configurations.

This module provides a convenient builder pattern for constructing
solver configurations programmatically.
"""

from __future__ import annotations

from typing import Any, Literal

from .core import SolverConfig


class ConfigBuilder:
    """
    Fluent builder for solver configurations.

    Provides a clean, discoverable API for constructing solver configurations
    programmatically with method chaining.

    Examples
    --------
    >>> # Basic usage
    >>> config = (
    ...     ConfigBuilder()
    ...     .solver_hjb(method="fdm", accuracy_order=2)
    ...     .solver_fp(method="particle", num_particles=5000)
    ...     .picard(max_iterations=50, tolerance=1e-6)
    ...     .backend("numpy")
    ...     .build()
    ... )

    >>> # With method-specific configs
    >>> config = (
    ...     ConfigBuilder()
    ...     .solver_hjb_gfdm(delta=0.15, stencil_size=25)
    ...     .solver_fp_particle(num_particles=10000, mode="collocation")
    ...     .build()
    ... )

    >>> # Complex configuration
    >>> config = (
    ...     ConfigBuilder()
    ...     .solver_hjb(
    ...         method="semi_lagrangian",
    ...         accuracy_order=3,
    ...         sl_config=SLConfig(interpolation_method="rbf", rk_order=4)
    ...     )
    ...     .solver_fp(method="fdm")
    ...     .picard(max_iterations=100, anderson_memory=5)
    ...     .backend("jax", device="gpu", precision="float64")
    ...     .logging(level="INFO", progress_bar=True)
    ...     .build()
    ... )
    """

    def __init__(self):
        """Initialize empty configuration builder."""
        self._hjb: dict[str, Any] = {}
        self._fp: dict[str, Any] = {}
        self._picard: dict[str, Any] = {}
        self._backend: dict[str, Any] = {}
        self._logging: dict[str, Any] = {}

    # HJB Solver Configuration Methods

    def solver_hjb(
        self,
        method: Literal["fdm", "gfdm", "semi_lagrangian", "weno"] = "fdm",
        accuracy_order: int = 2,
        boundary_conditions: Literal["dirichlet", "neumann", "periodic"] = "neumann",
        **kwargs,
    ) -> ConfigBuilder:
        """
        Configure HJB solver.

        Parameters
        ----------
        method : Literal["fdm", "gfdm", "semi_lagrangian", "weno"]
            HJB solver method
        accuracy_order : int
            Numerical accuracy order (1-5)
        boundary_conditions : Literal["dirichlet", "neumann", "periodic"]
            Boundary condition type
        **kwargs
            Additional HJB config parameters

        Returns
        -------
        ConfigBuilder
            Self for method chaining
        """
        self._hjb = {
            "method": method,
            "accuracy_order": accuracy_order,
            "boundary_conditions": boundary_conditions,
            **kwargs,
        }
        return self

    def solver_hjb_fdm(
        self,
        scheme: Literal["central", "upwind", "lax_friedrichs"] = "central",
        time_stepping: Literal["explicit", "implicit", "crank_nicolson"] = "implicit",
        accuracy_order: int = 2,
        **kwargs,
    ) -> ConfigBuilder:
        """Configure HJB FDM solver with FDM-specific parameters."""
        self._hjb = {
            "method": "fdm",
            "accuracy_order": accuracy_order,
            "fdm_config": {"scheme": scheme, "time_stepping": time_stepping},
            **kwargs,
        }
        return self

    def solver_hjb_gfdm(
        self,
        delta: float = 0.1,
        stencil_size: int = 20,
        qp_optimization_level: Literal["none", "auto", "always"] = "auto",
        **kwargs,
    ) -> ConfigBuilder:
        """Configure HJB GFDM solver with GFDM-specific parameters."""
        self._hjb = {
            "method": "gfdm",
            "gfdm_config": {
                "delta": delta,
                "stencil_size": stencil_size,
                "qp_optimization_level": qp_optimization_level,
            },
            **kwargs,
        }
        return self

    def solver_hjb_semi_lagrangian(
        self,
        interpolation_method: Literal["linear", "cubic", "rbf"] = "cubic",
        rk_order: Literal[1, 2, 3, 4] = 2,
        **kwargs,
    ) -> ConfigBuilder:
        """Configure HJB Semi-Lagrangian solver with SL-specific parameters."""
        self._hjb = {
            "method": "semi_lagrangian",
            "sl_config": {"interpolation_method": interpolation_method, "rk_order": rk_order},
            **kwargs,
        }
        return self

    def solver_hjb_weno(
        self, weno_order: Literal[3, 5, 7] = 5, flux_splitting: str = "lax_friedrichs", **kwargs
    ) -> ConfigBuilder:
        """Configure HJB WENO solver with WENO-specific parameters."""
        self._hjb = {
            "method": "weno",
            "weno_config": {"weno_order": weno_order, "flux_splitting": flux_splitting},
            **kwargs,
        }
        return self

    # FP Solver Configuration Methods

    def solver_fp(self, method: Literal["fdm", "particle", "network"] = "fdm", **kwargs) -> ConfigBuilder:
        """
        Configure Fokker-Planck solver.

        Parameters
        ----------
        method : Literal["fdm", "particle", "network"]
            FP solver method
        **kwargs
            Additional FP config parameters

        Returns
        -------
        ConfigBuilder
            Self for method chaining
        """
        self._fp = {"method": method, **kwargs}
        return self

    def solver_fp_fdm(
        self,
        scheme: Literal["upwind", "central", "lax_friedrichs"] = "upwind",
        time_stepping: Literal["explicit", "implicit"] = "implicit",
        **kwargs,
    ) -> ConfigBuilder:
        """Configure FP FDM solver with FDM-specific parameters."""
        self._fp = {
            "method": "fdm",
            "fdm_config": {"scheme": scheme, "time_stepping": time_stepping},
            **kwargs,
        }
        return self

    def solver_fp_particle(
        self,
        num_particles: int = 5000,
        kde_bandwidth: float | Literal["auto"] = "auto",
        normalization: Literal["none", "initial_only", "all"] = "initial_only",
        mode: Literal["hybrid", "collocation"] = "hybrid",
        **kwargs,
    ) -> ConfigBuilder:
        """Configure FP particle solver with particle-specific parameters."""
        self._fp = {
            "method": "particle",
            "particle_config": {
                "num_particles": num_particles,
                "kde_bandwidth": kde_bandwidth,
                "normalization": normalization,
                "mode": mode,
            },
            **kwargs,
        }
        return self

    def solver_fp_network(
        self,
        discretization_method: Literal["finite_volume", "finite_element"] = "finite_volume",
        **kwargs,
    ) -> ConfigBuilder:
        """Configure FP network solver with network-specific parameters."""
        self._fp = {
            "method": "network",
            "network_config": {"discretization_method": discretization_method},
            **kwargs,
        }
        return self

    # Picard Iteration Configuration

    def picard(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        anderson_memory: int = 0,
        verbose: bool = True,
    ) -> ConfigBuilder:
        """
        Configure Picard iteration.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        anderson_memory : int
            Anderson acceleration memory depth (0 = disabled)
        verbose : bool
            Print iteration progress

        Returns
        -------
        ConfigBuilder
            Self for method chaining
        """
        self._picard = {
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "anderson_memory": anderson_memory,
            "verbose": verbose,
        }
        return self

    # Backend Configuration

    def backend(
        self,
        backend_type: Literal["numpy", "jax", "pytorch"] = "numpy",
        device: Literal["cpu", "gpu", "auto"] = "cpu",
        precision: Literal["float32", "float64"] = "float64",
    ) -> ConfigBuilder:
        """
        Configure computational backend.

        Parameters
        ----------
        backend_type : Literal["numpy", "jax", "pytorch"]
            Backend type
        device : Literal["cpu", "gpu", "auto"]
            Compute device
        precision : Literal["float32", "float64"]
            Floating point precision

        Returns
        -------
        ConfigBuilder
            Self for method chaining
        """
        self._backend = {"type": backend_type, "device": device, "precision": precision}
        return self

    # Logging Configuration

    def logging(
        self,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
        progress_bar: bool = True,
        save_intermediate: bool = False,
        output_dir: str | None = None,
    ) -> ConfigBuilder:
        """
        Configure logging.

        Parameters
        ----------
        level : Literal["DEBUG", "INFO", "WARNING", "ERROR"]
            Logging level
        progress_bar : bool
            Show progress bar
        save_intermediate : bool
            Save intermediate results
        output_dir : str | None
            Directory for intermediate results

        Returns
        -------
        ConfigBuilder
            Self for method chaining
        """
        self._logging = {
            "level": level,
            "progress_bar": progress_bar,
            "save_intermediate": save_intermediate,
            "output_dir": output_dir,
        }
        return self

    # Build Method

    def build(self) -> SolverConfig:
        """
        Build the solver configuration.

        Returns
        -------
        SolverConfig
            Validated solver configuration

        Raises
        ------
        ValidationError
            If configuration is invalid
        """
        config_dict: dict[str, Any] = {}

        if self._hjb:
            config_dict["hjb"] = self._hjb
        if self._fp:
            config_dict["fp"] = self._fp
        if self._picard:
            config_dict["picard"] = self._picard
        if self._backend:
            config_dict["backend"] = self._backend
        if self._logging:
            config_dict["logging"] = self._logging

        return SolverConfig.model_validate(config_dict)

    def from_dict(self, data: dict[str, Any]) -> ConfigBuilder:
        """
        Populate builder from dictionary.

        Useful for converting kwargs to config.

        Parameters
        ----------
        data : dict
            Configuration dictionary

        Returns
        -------
        ConfigBuilder
            Self for method chaining
        """
        if "hjb" in data or "hjb_config" in data:
            self._hjb = data.get("hjb", data.get("hjb_config", {}))
        if "fp" in data or "fp_config" in data:
            self._fp = data.get("fp", data.get("fp_config", {}))
        if "picard" in data or "picard_config" in data:
            self._picard = data.get("picard", data.get("picard_config", {}))
        if "backend" in data or "backend_config" in data:
            self._backend = data.get("backend", data.get("backend_config", {}))
        if "logging" in data or "logging_config" in data:
            self._logging = data.get("logging", data.get("logging_config", {}))

        return self
