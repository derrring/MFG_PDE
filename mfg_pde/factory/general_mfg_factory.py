"""
Factory for creating general MFG problems from configurations and function specifications.

This module provides high-level interfaces for creating any MFG problem
through configuration files and programmatic function definitions.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.config.omegaconf_manager import OmegaConfManager
from mfg_pde.core.mfg_problem import MFGProblem, MFGProblemBuilder
from mfg_pde.geometry import BoundaryConditions
from mfg_pde.utils.logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class GeneralMFGFactory:
    """
    Factory for creating general MFG problems from various input formats.

    This factory can create any MFG problem from:
    - Python function definitions
    - Configuration files with function references
    - Inline lambda expressions in YAML
    - Symbolic mathematical expressions (future extension)
    """

    def __init__(self) -> None:
        """Initialize general MFG factory."""
        self.function_registry: dict[str, Callable[..., Any]] = {}

    def register_functions(self, functions: dict[str, Callable[..., Any]]) -> None:
        """Register named functions for use in configurations."""
        self.function_registry.update(functions)
        logger.info(f"Registered {len(functions)} functions: {list(functions.keys())}")

    def create_from_functions(
        self,
        hamiltonian_func: Callable[..., Any],
        hamiltonian_dm_func: Callable[..., Any],
        domain_config: dict[str, Any],
        time_config: dict[str, Any],
        solver_config: dict[str, Any] | None = None,
        **optional_components: Any,
    ) -> MFGProblem:
        """
        Create MFG problem from function objects.

        Args:
            hamiltonian_func: Hamiltonian function H(...)
            hamiltonian_dm_func: Hamiltonian derivative dH/dm(...)
            domain_config: Domain specification {"xmin": ..., "xmax": ..., "Nx": ...}
            time_config: Time specification {"T": ..., "Nt": ...}
            solver_config: Solver parameters {"sigma": ..., "coefCT": ...}
            **optional_components: Additional components (potential_func, etc.)
        """

        builder = (
            MFGProblemBuilder()
            .hamiltonian(hamiltonian_func, hamiltonian_dm_func)
            .domain(**domain_config)
            .time(**time_config)
        )

        if solver_config:
            builder.coefficients(**solver_config)

        # Add optional components
        for key, value in optional_components.items():
            if key == "potential_func" and value is not None:
                builder.potential(value)
            elif key == "initial_density_func" and value is not None:
                builder.initial_density(value)
            elif key == "final_value_func" and value is not None:
                builder.final_value(value)
            elif key == "boundary_conditions" and value is not None:
                if isinstance(value, dict):
                    builder.boundary_conditions(BoundaryConditions(**value))
                else:
                    builder.boundary_conditions(value)
            elif key == "jacobian_func" and value is not None:
                builder.jacobian(value)
            elif key == "coupling_func" and value is not None:
                builder.coupling(value)
            elif key == "parameters" and value is not None:
                builder.parameters(**value)
            elif key == "description" and value is not None:
                builder.description(value)

        return builder.build()

    def create_from_config_file(self, config_path: str) -> MFGProblem:
        """
        Create MFG problem from YAML configuration file.

        Expected format:
        ```yaml
        problem:
          description: "My custom MFG problem"
          type: custom

        functions:
          hamiltonian: "my_module.my_hamiltonian"
          hamiltonian_dm: "my_module.my_hamiltonian_dm"
          potential: "lambda x: 0.5 * x**2"
          initial_density: "lambda x: np.exp(-10*(x-0.5)**2)"

        domain:
          xmin: 0.0
          xmax: 1.0
          Nx: 101

        time:
          T: 1.0
          Nt: 101

        solver:
          sigma: 0.5
          coefCT: 1.0

        parameters:
          alpha: 2.0
          beta: 0.5
        ```
        """

        config_manager = OmegaConfManager()
        config = config_manager.load_config(config_path)

        return self.create_from_config_dict(dict(config))

    def create_from_config_dict(self, config: dict[str, Any]) -> MFGProblem:
        """Create MFG problem from configuration dictionary."""

        # Extract function specifications
        functions_config = config.get("functions", {})

        # Load required functions
        hamiltonian_func = self._load_function(functions_config.get("hamiltonian"))
        hamiltonian_dm_func = self._load_function(functions_config.get("hamiltonian_dm"))

        if hamiltonian_func is None:
            raise ValueError("hamiltonian function is required")
        if hamiltonian_dm_func is None:
            raise ValueError("hamiltonian_dm function is required")

        # Load optional functions
        optional_functions = {}
        for key in [
            "potential",
            "initial_density",
            "final_value",
            "jacobian",
            "coupling",
        ]:
            if key in functions_config:
                func = self._load_function(functions_config[key])
                if func is not None:
                    optional_functions[f"{key}_func"] = func

        # Extract configurations
        domain_config = config.get("domain", {"xmin": 0.0, "xmax": 1.0, "Nx": 51})
        time_config = config.get("time", {"T": 1.0, "Nt": 51})
        solver_config = config.get("solver", {"sigma": 1.0, "coefCT": 0.5})

        # Extract other components
        if "boundary_conditions" in config:
            optional_functions["boundary_conditions"] = config["boundary_conditions"]

        if "parameters" in config:
            optional_functions["parameters"] = config["parameters"]

        if "problem" in config:
            problem_info = config["problem"]
            if "description" in problem_info:
                optional_functions["description"] = problem_info["description"]

        return self.create_from_functions(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm_func,
            domain_config=domain_config,
            time_config=time_config,
            solver_config=solver_config,
            **optional_functions,
        )

    def _load_function(self, func_spec: str | None) -> Callable | None:
        """
        Load function from various specifications.

        Supports:
        - Module path: "my_module.my_function"
        - Lambda expressions: "lambda x: x**2"
        - Registry names: registered_function_name
        """

        if func_spec is None:
            return None

        func_spec = func_spec.strip()

        # Check if it's in the registry
        if func_spec in self.function_registry:
            return self.function_registry[func_spec]

        # Check if it's a lambda expression
        if func_spec.startswith("lambda"):
            try:
                # Create safe namespace for lambda evaluation
                safe_namespace = {
                    "np": np,
                    "numpy": np,
                    "exp": np.exp,
                    "sin": np.sin,
                    "cos": np.cos,
                    "sqrt": np.sqrt,
                    "abs": abs,
                    "max": max,
                    "min": min,
                    "pi": np.pi,
                    "e": np.e,
                }
                return eval(func_spec, safe_namespace)
            except Exception as e:
                logger.error(f"Failed to evaluate lambda function '{func_spec}': {e}")
                return None

        # Check if it's a module path
        if "." in func_spec:
            try:
                module_path, function_name = func_spec.rsplit(".", 1)
                module = importlib.import_module(module_path)
                return getattr(module, function_name)
            except Exception as e:
                logger.error(f"Failed to import function '{func_spec}': {e}")
                return None

        logger.warning(f"Could not resolve function specification: '{func_spec}'")
        return None

    def create_template_config(self, filename: str) -> None:
        """Create a template configuration file."""

        template = {
            "problem": {"description": "My custom MFG problem", "type": "custom"},
            "functions": {
                "hamiltonian": "lambda x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem: 0.5 * problem.coefCT * (problem.utils.npart(p_values.get('forward', 0))**2 + problem.utils.ppart(p_values.get('backward', 0))**2) - problem.f_potential[x_idx] - m_at_x**2",
                "hamiltonian_dm": "lambda x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem: -2.0 * m_at_x",
                "potential": "lambda x: 0.5 * x * (1 - x)",
                "initial_density": "lambda x: np.exp(-10 * (x - 0.3)**2)",
                "final_value": "lambda x: (x - 0.8)**2",
            },
            "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 51},
            "time": {"T": 1.0, "Nt": 51},
            "solver": {"sigma": 0.5, "coefCT": 1.0},
            "boundary_conditions": {"type": "periodic"},
            "parameters": {"alpha": 1.0, "beta": 0.5},
        }

        config_manager = OmegaConfManager()
        config_manager.save_config(template, filename)  # type: ignore[arg-type]
        logger.info(f"Created template configuration: {filename}")

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration for general MFG problem."""

        validation: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

        # Check required sections
        required_sections = ["functions", "domain", "time"]
        for section in required_sections:
            if section not in config:
                validation["valid"] = False
                validation["errors"].append(f"Missing required section: {section}")

        # Check required functions
        if "functions" in config:
            required_functions = ["hamiltonian", "hamiltonian_dm"]
            functions = config["functions"]
            for func in required_functions:
                if func not in functions:
                    validation["valid"] = False
                    validation["errors"].append(f"Missing required function: {func}")

        # Check domain parameters
        if "domain" in config:
            domain = config["domain"]
            required_domain = ["xmin", "xmax", "Nx"]
            for param in required_domain:
                if param not in domain:
                    validation["valid"] = False
                    validation["errors"].append(f"Missing domain parameter: {param}")

        # Check time parameters
        if "time" in config:
            time_config = config["time"]
            required_time = ["T", "Nt"]
            for param in required_time:
                if param not in time_config:
                    validation["valid"] = False
                    validation["errors"].append(f"Missing time parameter: {param}")

        return validation


# Global factory instance
_global_general_factory: GeneralMFGFactory | None = None


def get_general_factory() -> GeneralMFGFactory:
    """Get global general MFG factory instance."""
    global _global_general_factory
    if _global_general_factory is None:
        _global_general_factory = GeneralMFGFactory()
    return _global_general_factory


def create_general_mfg_problem(
    hamiltonian_func: Callable[..., Any], hamiltonian_dm_func: Callable[..., Any], **kwargs: Any
) -> MFGProblem:
    """Convenience function to create general MFG problem."""
    factory = get_general_factory()

    # Extract configurations
    domain_config = {
        "xmin": kwargs.pop("xmin", 0.0),
        "xmax": kwargs.pop("xmax", 1.0),
        "Nx": kwargs.pop("Nx", 51),
    }

    time_config = {"T": kwargs.pop("T", 1.0), "Nt": kwargs.pop("Nt", 51)}

    solver_config = {
        "sigma": kwargs.pop("sigma", 1.0),
        "coefCT": kwargs.pop("coefCT", 0.5),
    }

    return factory.create_from_functions(
        hamiltonian_func=hamiltonian_func,
        hamiltonian_dm_func=hamiltonian_dm_func,
        domain_config=domain_config,
        time_config=time_config,
        solver_config=solver_config,
        **kwargs,
    )
