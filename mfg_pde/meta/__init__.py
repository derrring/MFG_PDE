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

from .code_generation import CodeGenerator, MFGSolverGenerator, generate_discretization, generate_solver_class
from .mathematical_dsl import HamiltonianBuilder, LagrangianBuilder, MathematicalExpression, MFGSystemBuilder
from .optimization_meta import JITSolverFactory, OptimizationCompiler, create_optimized_solver
from .type_system import DynamicSolver, MFGType, SolverMetaclass, TypedMFGProblem

__all__ = [
    "CodeGenerator",
    "DynamicSolver",
    "HamiltonianBuilder",
    "JITSolverFactory",
    "LagrangianBuilder",
    "MFGSolverGenerator",
    "MFGSystemBuilder",
    "MFGType",
    "MathematicalExpression",
    "OptimizationCompiler",
    "SolverMetaclass",
    "TypedMFGProblem",
    "create_optimized_solver",
    "generate_discretization",
    "generate_solver_class",
]
