"""
Advanced type system for MFG meta-programming.

This module provides a sophisticated type system that enables:
- Mathematical type checking at runtime
- Automatic solver specialization based on problem types
- Type-driven optimization and code generation
- Mathematical constraint validation through types
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

import numpy as np
from numpy.typing import NDArray


class MathematicalSpace(Enum):
    """Enumeration of mathematical spaces for type checking."""
    
    REAL_LINE = "R"
    UNIT_INTERVAL = "[0,1]"
    TORUS = "T"
    SIMPLEX = "Δ"
    PROBABILITY_MEASURES = "P"
    SOBOLEV_H1 = "H¹"
    SOBOLEV_H2 = "H²"
    CONTINUOUS = "C⁰"
    LIPSCHITZ = "Lip"


class NumericalMethod(Enum):
    """Enumeration of numerical methods for type-based dispatch."""
    
    FINITE_DIFFERENCE = "fdm"
    FINITE_ELEMENT = "fem"
    SPECTRAL = "spectral"
    PARTICLE = "particle"
    SEMI_LAGRANGIAN = "semi_lagrangian"
    VARIATIONAL = "variational"
    PRIMAL_DUAL = "primal_dual"


@dataclass
class MFGType:
    """
    Mathematical type descriptor for MFG problems.
    
    Encodes mathematical properties that affect numerical treatment:
    - Function spaces for state and control
    - Regularity requirements
    - Constraint structure
    - Optimization characteristics
    """
    
    state_space: MathematicalSpace
    control_space: MathematicalSpace
    density_space: MathematicalSpace
    time_horizon: Union[float, str]  # "finite" or "infinite"
    
    # Regularity properties
    hamiltonian_regularity: str = "C²"
    cost_regularity: str = "C¹"
    constraint_type: Optional[str] = None
    
    # Numerical properties
    preferred_methods: List[NumericalMethod] = None
    stability_constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferred_methods is None:
            self.preferred_methods = []
        if self.stability_constraints is None:
            self.stability_constraints = {}
    
    def is_compatible_with(self, method: NumericalMethod) -> bool:
        """Check if numerical method is compatible with this type."""
        if self.preferred_methods and method not in self.preferred_methods:
            return False
        
        # Additional compatibility checks
        if method == NumericalMethod.SPECTRAL and self.state_space != MathematicalSpace.TORUS:
            return False
        
        if method == NumericalMethod.PARTICLE and self.density_space != MathematicalSpace.PROBABILITY_MEASURES:
            return False
        
        return True
    
    def get_stability_constraint(self, parameter: str) -> Optional[Any]:
        """Get stability constraint for given parameter."""
        return self.stability_constraints.get(parameter)


T = TypeVar('T')
ProblemType = TypeVar('ProblemType')
SolverType = TypeVar('SolverType')


class TypedMFGProblem(Generic[T]):
    """
    MFG problem with mathematical type information.
    
    Extends standard MFG problems with type annotations that enable
    automatic solver selection and optimization.
    """
    
    def __init__(self, 
                 problem_data: Any,
                 mfg_type: MFGType,
                 type_parameter: Type[T] = None):
        self.problem_data = problem_data
        self.mfg_type = mfg_type
        self.type_parameter = type_parameter
        self._validated = False
    
    def validate_type_constraints(self) -> bool:
        """Validate that problem satisfies type constraints."""
        try:
            # Check function space compatibility
            self._validate_function_spaces()
            
            # Check regularity requirements
            self._validate_regularity()
            
            # Check numerical stability constraints
            self._validate_stability_constraints()
            
            self._validated = True
            return True
            
        except TypeError as e:
            print(f"Type validation failed: {e}")
            return False
    
    def _validate_function_spaces(self):
        """Validate function space requirements."""
        # Check if state space is appropriate for problem domain
        if hasattr(self.problem_data, 'xmin') and hasattr(self.problem_data, 'xmax'):
            if self.problem_data.xmin == 0 and self.problem_data.xmax == 1:
                if self.mfg_type.state_space not in [MathematicalSpace.UNIT_INTERVAL, MathematicalSpace.REAL_LINE]:
                    raise TypeError("State space incompatible with domain [0,1]")
    
    def _validate_regularity(self):
        """Validate regularity requirements."""
        # Check if Hamiltonian has required smoothness
        if hasattr(self.problem_data, 'hamiltonian'):
            # Would check actual function properties in practice
            pass
    
    def _validate_stability_constraints(self):
        """Validate numerical stability constraints."""
        constraints = self.mfg_type.stability_constraints
        
        if 'max_dt' in constraints:
            max_dt = constraints['max_dt']
            if hasattr(self.problem_data, 'dt') and self.problem_data.dt > max_dt:
                raise TypeError(f"Time step {self.problem_data.dt} exceeds stability limit {max_dt}")
    
    def get_compatible_solvers(self) -> List[Type]:
        """Get list of solver types compatible with this problem type."""
        compatible = []
        
        for method in NumericalMethod:
            if self.mfg_type.is_compatible_with(method):
                solver_class = SolverRegistry.get_solver_for_method(method)
                if solver_class:
                    compatible.append(solver_class)
        
        return compatible
    
    def specialize_for_backend(self, backend: str) -> 'TypedMFGProblem[T]':
        """Create specialized version for specific computational backend."""
        specialized_type = MFGType(
            state_space=self.mfg_type.state_space,
            control_space=self.mfg_type.control_space,
            density_space=self.mfg_type.density_space,
            time_horizon=self.mfg_type.time_horizon,
            hamiltonian_regularity=self.mfg_type.hamiltonian_regularity,
            cost_regularity=self.mfg_type.cost_regularity,
            constraint_type=self.mfg_type.constraint_type,
            preferred_methods=self._filter_methods_for_backend(backend),
            stability_constraints=self._adjust_constraints_for_backend(backend)
        )
        
        return TypedMFGProblem(self.problem_data, specialized_type, self.type_parameter)
    
    def _filter_methods_for_backend(self, backend: str) -> List[NumericalMethod]:
        """Filter numerical methods based on backend capabilities."""
        if backend == "jax":
            # JAX excels at certain methods
            return [m for m in self.mfg_type.preferred_methods 
                   if m in [NumericalMethod.SPECTRAL, NumericalMethod.VARIATIONAL]]
        elif backend == "numba":
            # Numba good for explicit methods
            return [m for m in self.mfg_type.preferred_methods
                   if m in [NumericalMethod.FINITE_DIFFERENCE, NumericalMethod.PARTICLE]]
        else:
            return self.mfg_type.preferred_methods
    
    def _adjust_constraints_for_backend(self, backend: str) -> Dict[str, Any]:
        """Adjust stability constraints for backend."""
        constraints = self.mfg_type.stability_constraints.copy()
        
        if backend == "jax":
            # JAX might allow larger time steps due to better precision
            if 'max_dt' in constraints:
                constraints['max_dt'] *= 1.5
        
        return constraints


class SolverMetaclass(type):
    """
    Metaclass for automatic solver registration and type checking.
    
    Automatically registers solvers with their compatible types
    and enables type-based dispatch.
    """
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Extract type information from class
        if hasattr(cls, 'compatible_types'):
            for mfg_type in cls.compatible_types:
                SolverRegistry.register_solver(cls, mfg_type)
        
        # Add type checking to solve method
        if hasattr(cls, 'solve'):
            original_solve = cls.solve
            cls.solve = mcs._add_type_checking(original_solve)
        
        return cls
    
    @staticmethod
    def _add_type_checking(solve_method):
        """Add runtime type checking to solve method."""
        def typed_solve(self, problem, *args, **kwargs):
            if isinstance(problem, TypedMFGProblem):
                if not problem.validate_type_constraints():
                    raise TypeError("Problem fails type validation")
                
                if not problem.mfg_type.is_compatible_with(self.method_type):
                    raise TypeError(f"Solver {type(self).__name__} incompatible with problem type")
            
            return solve_method(self, problem, *args, **kwargs)
        
        return typed_solve


class SolverRegistry:
    """Registry for type-based solver dispatch."""
    
    _solvers: Dict[MFGType, List[Type]] = {}
    _method_map: Dict[NumericalMethod, Type] = {}
    
    @classmethod
    def register_solver(cls, solver_class: Type, mfg_type: MFGType):
        """Register solver for given MFG type."""
        if mfg_type not in cls._solvers:
            cls._solvers[mfg_type] = []
        cls._solvers[mfg_type].append(solver_class)
        
        # Also register by method if available
        if hasattr(solver_class, 'method_type'):
            cls._method_map[solver_class.method_type] = solver_class
    
    @classmethod
    def get_solver_for_type(cls, mfg_type: MFGType) -> Optional[Type]:
        """Get best solver for given MFG type."""
        compatible_solvers = cls._solvers.get(mfg_type, [])
        if compatible_solvers:
            # Return first compatible solver (could implement ranking)
            return compatible_solvers[0]
        return None
    
    @classmethod
    def get_solver_for_method(cls, method: NumericalMethod) -> Optional[Type]:
        """Get solver implementing given numerical method."""
        return cls._method_map.get(method)
    
    @classmethod
    def find_compatible_solvers(cls, problem: TypedMFGProblem) -> List[Type]:
        """Find all solvers compatible with given typed problem."""
        compatible = []
        
        for solvers in cls._solvers.values():
            for solver_class in solvers:
                if hasattr(solver_class, 'method_type'):
                    if problem.mfg_type.is_compatible_with(solver_class.method_type):
                        compatible.append(solver_class)
        
        return compatible


class DynamicSolver(metaclass=SolverMetaclass):
    """
    Base class for dynamically typed MFG solvers.
    
    Uses metaclass to provide automatic type checking and registration.
    """
    
    method_type: NumericalMethod = None
    compatible_types: List[MFGType] = []
    
    def __init__(self, config=None):
        self.config = config
        self.problem_type: Optional[MFGType] = None
    
    def solve(self, problem, *args, **kwargs):
        """Solve method with automatic type checking (added by metaclass)."""
        raise NotImplementedError("Subclasses must implement solve method")
    
    def adapt_to_type(self, mfg_type: MFGType):
        """Adapt solver parameters to given MFG type."""
        self.problem_type = mfg_type
        
        # Adjust configuration based on type constraints
        if mfg_type.get_stability_constraint('max_dt'):
            max_dt = mfg_type.get_stability_constraint('max_dt')
            if hasattr(self.config, 'dt') and self.config.dt > max_dt:
                self.config.dt = max_dt * 0.9  # Safety margin


# Predefined MFG types for common problems

QUADRATIC_MFG_TYPE = MFGType(
    state_space=MathematicalSpace.REAL_LINE,
    control_space=MathematicalSpace.REAL_LINE,
    density_space=MathematicalSpace.PROBABILITY_MEASURES,
    time_horizon="finite",
    hamiltonian_regularity="C²",
    preferred_methods=[NumericalMethod.FINITE_DIFFERENCE, NumericalMethod.SPECTRAL],
    stability_constraints={"max_dt": 0.01}
)

CONGESTION_MFG_TYPE = MFGType(
    state_space=MathematicalSpace.UNIT_INTERVAL,
    control_space=MathematicalSpace.REAL_LINE,
    density_space=MathematicalSpace.PROBABILITY_MEASURES,
    time_horizon="finite",
    hamiltonian_regularity="C¹",
    constraint_type="congestion",
    preferred_methods=[NumericalMethod.VARIATIONAL, NumericalMethod.PRIMAL_DUAL],
    stability_constraints={"max_dt": 0.005, "min_regularization": 1e-6}
)

NETWORK_MFG_TYPE = MFGType(
    state_space=MathematicalSpace.SIMPLEX,
    control_space=MathematicalSpace.SIMPLEX,
    density_space=MathematicalSpace.PROBABILITY_MEASURES,
    time_horizon="finite",
    hamiltonian_regularity="C⁰",
    constraint_type="network",
    preferred_methods=[NumericalMethod.FINITE_DIFFERENCE, NumericalMethod.PARTICLE],
    stability_constraints={"max_dt": 0.02}
)

PERIODIC_MFG_TYPE = MFGType(
    state_space=MathematicalSpace.TORUS,
    control_space=MathematicalSpace.REAL_LINE,
    density_space=MathematicalSpace.PROBABILITY_MEASURES,
    time_horizon="finite",
    hamiltonian_regularity="C²",
    preferred_methods=[NumericalMethod.SPECTRAL, NumericalMethod.FINITE_DIFFERENCE],
    stability_constraints={"max_dt": 0.01}
)


# Type checking and conversion utilities

def infer_mfg_type(problem) -> MFGType:
    """Infer MFG type from problem characteristics."""
    # Simple inference based on problem attributes
    if hasattr(problem, 'xmin') and hasattr(problem, 'xmax'):
        if problem.xmin == 0 and problem.xmax == 1:
            state_space = MathematicalSpace.UNIT_INTERVAL
        elif hasattr(problem, 'periodic') and problem.periodic:
            state_space = MathematicalSpace.TORUS
        else:
            state_space = MathematicalSpace.REAL_LINE
    else:
        state_space = MathematicalSpace.REAL_LINE
    
    # Infer other properties
    control_space = MathematicalSpace.REAL_LINE
    density_space = MathematicalSpace.PROBABILITY_MEASURES
    time_horizon = "finite" if hasattr(problem, 'T') else "infinite"
    
    return MFGType(
        state_space=state_space,
        control_space=control_space,
        density_space=density_space,
        time_horizon=time_horizon
    )


def create_typed_problem(problem, mfg_type: MFGType = None) -> TypedMFGProblem:
    """Create typed MFG problem from standard problem."""
    if mfg_type is None:
        mfg_type = infer_mfg_type(problem)
    
    return TypedMFGProblem(problem, mfg_type)


def dispatch_solver(problem: TypedMFGProblem, 
                   method_preference: Optional[NumericalMethod] = None) -> DynamicSolver:
    """Automatically dispatch appropriate solver for typed problem."""
    if method_preference:
        solver_class = SolverRegistry.get_solver_for_method(method_preference)
        if solver_class and problem.mfg_type.is_compatible_with(method_preference):
            return solver_class()
    
    # Find best compatible solver
    compatible_solvers = SolverRegistry.find_compatible_solvers(problem)
    if compatible_solvers:
        # Could implement ranking/scoring here
        return compatible_solvers[0]()
    
    raise ValueError(f"No compatible solver found for problem type {problem.mfg_type}")


# Example usage
if __name__ == "__main__":
    # Create a typed problem
    from ..core.mfg_problem import MFGProblem
    
    problem = MFGProblem(xmin=0, xmax=1, T=1.0, Nx=100, Nt=50)
    typed_problem = create_typed_problem(problem, QUADRATIC_MFG_TYPE)
    
    print(f"Problem type: {typed_problem.mfg_type}")
    print(f"Validation passed: {typed_problem.validate_type_constraints()}")
    print(f"Compatible methods: {typed_problem.mfg_type.preferred_methods}")