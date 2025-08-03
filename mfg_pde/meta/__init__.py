"""
Meta-programming framework for MFG_PDE.

This module provides advanced meta-programming capabilities for automatically
generating, optimizing, and composing MFG solvers and mathematical formulations.

Key Features:
- Automatic solver generation from mathematical specifications
- Runtime code optimization and specialization
- Mathematical expression compilation
- Dynamic type system for numerical methods
"""

from .code_generation import (
    CodeGenerator,
    MFGSolverGenerator,
    generate_solver_class,
    generate_discretization,
)
from .mathematical_dsl import (
    MathematicalExpression,
    MFGSystemBuilder,
    HamiltonianBuilder,
    LagrangianBuilder,
)
from .optimization_meta import (
    OptimizationCompiler,
    JITSolverFactory,
    create_optimized_solver,
)
from .type_system import (
    MFGType,
    SolverMetaclass,
    DynamicSolver,
    TypedMFGProblem,
)

__all__ = [
    "CodeGenerator",
    "MFGSolverGenerator", 
    "generate_solver_class",
    "generate_discretization",
    "MathematicalExpression",
    "MFGSystemBuilder",
    "HamiltonianBuilder",
    "LagrangianBuilder",
    "OptimizationCompiler",
    "JITSolverFactory",
    "create_optimized_solver",
    "MFGType",
    "SolverMetaclass",
    "DynamicSolver",
    "TypedMFGProblem",
]