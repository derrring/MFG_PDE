"""
Mathematical Notation Standards for MFG_PDE Package

This module defines the standard mathematical notation used throughout
the MFG_PDE package for consistency and clarity across all solvers,
documentation, and user interfaces.

The notation follows established conventions in the Mean Field Games
literature while providing clear mappings to computational variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class VariableType(Enum):
    """Classification of mathematical variables by their role."""

    SPATIAL = "spatial"  # Spatial coordinates and discretization
    TEMPORAL = "temporal"  # Time coordinates and discretization
    SOLUTION = "solution"  # Primary MFG solution variables
    PARAMETER = "parameter"  # Physical and numerical parameters
    COMPUTATIONAL = "computational"  # Solver and algorithm parameters


@dataclass
class NotationEntry:
    """Standard entry for mathematical notation documentation."""

    symbol: str  # Mathematical symbol (e.g., "u", "∇u")
    variable_name: str  # Code variable name (e.g., "U", "grad_u")
    description: str  # Mathematical meaning
    domain: str  # Mathematical domain (e.g., "[0,T] × [0,L]")
    units: str = ""  # Physical units when applicable
    variable_type: VariableType = VariableType.SOLUTION
    latex: str = ""  # LaTeX representation
    aliases: list[str] | None = None  # Alternative names or legacy names

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if not self.latex:
            self.latex = self.symbol


class MFGNotationRegistry:
    """Central registry for all mathematical notation used in MFG_PDE."""

    def __init__(self):
        self.entries: dict[str, NotationEntry] = {}
        self._register_standard_notation()

    def _register_standard_notation(self) -> None:
        """Register all standard mathematical notation for MFG systems."""

        # === Spatial and Temporal Variables ===
        self.register(
            NotationEntry(
                symbol="x",
                variable_name="x",
                description="Spatial coordinate",
                domain="[xmin, xmax]",
                units="length",
                variable_type=VariableType.SPATIAL,
                latex="x",
            )
        )

        self.register(
            NotationEntry(
                symbol="t",
                variable_name="t",
                description="Time coordinate",
                domain="[0, T]",
                units="time",
                variable_type=VariableType.TEMPORAL,
                latex="t",
            )
        )

        self.register(
            NotationEntry(
                symbol="Nx",
                variable_name="Nx",
                description="Number of spatial grid points",
                domain="ℕ⁺",
                variable_type=VariableType.SPATIAL,
                latex="N_x",
            )
        )

        self.register(
            NotationEntry(
                symbol="Nt",
                variable_name="Nt",
                description="Number of temporal grid points",
                domain="ℕ⁺",
                variable_type=VariableType.TEMPORAL,
                latex="N_t",
            )
        )

        self.register(
            NotationEntry(
                symbol="Δx",
                variable_name="Dx",
                description="Spatial grid spacing",
                domain="ℝ⁺",
                units="length",
                variable_type=VariableType.SPATIAL,
                latex="\\Delta x",
            )
        )

        self.register(
            NotationEntry(
                symbol="Δt",
                variable_name="Dt",
                description="Temporal grid spacing",
                domain="ℝ⁺",
                units="time",
                variable_type=VariableType.TEMPORAL,
                latex="\\Delta t",
            )
        )

        # === Primary MFG Solution Variables ===
        self.register(
            NotationEntry(
                symbol="u(t,x)",
                variable_name="U",
                description="Value function (Hamilton-Jacobi-Bellman solution)",
                domain="[0,T] × [xmin,xmax] → ℝ",
                variable_type=VariableType.SOLUTION,
                latex="u(t,x)",
                aliases=["value_function", "hjb_solution"],
            )
        )

        self.register(
            NotationEntry(
                symbol="m(t,x)",
                variable_name="M",
                description="Density function (Fokker-Planck solution)",
                domain="[0,T] × [xmin,xmax] → ℝ⁺",
                variable_type=VariableType.SOLUTION,
                latex="m(t,x)",
                aliases=["density_function", "fp_solution"],
            )
        )

        self.register(
            NotationEntry(
                symbol="∇u",
                variable_name="grad_U",
                description="Spatial gradient of value function",
                domain="[0,T] × [xmin,xmax] → ℝ",
                variable_type=VariableType.SOLUTION,
                latex="\\nabla u",
            )
        )

        # === Boundary and Initial Conditions ===
        self.register(
            NotationEntry(
                symbol="u(T,x)",
                variable_name="u_fin",
                description="Terminal condition for value function",
                domain="[xmin,xmax] → ℝ",
                variable_type=VariableType.SOLUTION,
                latex="u(T,x)",
                aliases=["final_condition", "terminal_u"],
            )
        )

        self.register(
            NotationEntry(
                symbol="m(0,x)",
                variable_name="m_init",
                description="Initial density distribution",
                domain="[xmin,xmax] → ℝ⁺",
                variable_type=VariableType.SOLUTION,
                latex="m(0,x)",
                aliases=["initial_density", "initial_m"],
            )
        )

        # === Physical Parameters ===
        self.register(
            NotationEntry(
                symbol="σ",
                variable_name="sigma",
                description="Diffusion coefficient in stochastic dynamics",
                domain="ℝ⁺",
                units="√(length²/time)",
                variable_type=VariableType.PARAMETER,
                latex="\\sigma",
            )
        )

        self.register(
            NotationEntry(
                symbol="λ",
                variable_name="coupling_coefficient",
                description="Coupling strength between agents",
                domain="ℝ⁺",
                variable_type=VariableType.PARAMETER,
                latex="\\lambda",
                aliases=["coefCT"],  # Legacy name
            )
        )

        # === Hamiltonian and Related Functions ===
        self.register(
            NotationEntry(
                symbol="H(x,p,m)",
                variable_name="H",
                description="Hamiltonian function",
                domain="[xmin,xmax] × ℝ × ℝ⁺ → ℝ",
                variable_type=VariableType.SOLUTION,
                latex="H(x,p,m)",
            )
        )

        self.register(
            NotationEntry(
                symbol="∂H/∂m",
                variable_name="dH_dm",
                description="Partial derivative of Hamiltonian with respect to density",
                domain="[xmin,xmax] × ℝ × ℝ⁺ → ℝ",
                variable_type=VariableType.SOLUTION,
                latex="\\frac{\\partial H}{\\partial m}",
            )
        )

        self.register(
            NotationEntry(
                symbol="V(x)",
                variable_name="f_potential",
                description="Potential function (external forcing)",
                domain="[xmin,xmax] → ℝ",
                variable_type=VariableType.PARAMETER,
                latex="V(x)",
                aliases=["potential", "external_potential"],
            )
        )

        # === Numerical Parameters ===
        self.register(
            NotationEntry(
                symbol="εₙ",
                variable_name="newton_tolerance",
                description="Newton method convergence tolerance",
                domain="ℝ⁺",
                variable_type=VariableType.COMPUTATIONAL,
                latex="\\varepsilon_N",
                aliases=["l2errBoundNewton"],
            )
        )

        self.register(
            NotationEntry(
                symbol="εₚ",
                variable_name="picard_tolerance",
                description="Picard iteration convergence tolerance",
                domain="ℝ⁺",
                variable_type=VariableType.COMPUTATIONAL,
                latex="\\varepsilon_P",
                aliases=["l2errBoundPicard"],
            )
        )

        self.register(
            NotationEntry(
                symbol="Nₙ",
                variable_name="max_newton_iterations",
                description="Maximum Newton iterations per time step",
                domain="ℕ⁺",
                variable_type=VariableType.COMPUTATIONAL,
                latex="N_N",
                aliases=["NiterNewton"],
            )
        )

        self.register(
            NotationEntry(
                symbol="Nₚ",
                variable_name="max_picard_iterations",
                description="Maximum Picard iterations for coupling",
                domain="ℕ⁺",
                variable_type=VariableType.COMPUTATIONAL,
                latex="N_P",
                aliases=["Niter_max"],
            )
        )

    def register(self, entry: NotationEntry) -> None:
        """Register a notation entry."""
        self.entries[entry.variable_name] = entry

        # Also register aliases
        if entry.aliases is not None:
            for alias in entry.aliases:
                if alias not in self.entries:
                    alias_entry = NotationEntry(
                        symbol=entry.symbol,
                        variable_name=alias,
                        description=f"Alias for {entry.variable_name}: {entry.description}",
                        domain=entry.domain,
                        units=entry.units,
                        variable_type=entry.variable_type,
                        latex=entry.latex,
                        aliases=[],
                    )
                    self.entries[alias] = alias_entry

    def get_entry(self, variable_name: str) -> NotationEntry | None:
        """Get notation entry by variable name."""
        return self.entries.get(variable_name)

    def get_symbol(self, variable_name: str) -> str:
        """Get mathematical symbol for variable name."""
        entry = self.get_entry(variable_name)
        return entry.symbol if entry else variable_name

    def get_latex(self, variable_name: str) -> str:
        """Get LaTeX representation for variable name."""
        entry = self.get_entry(variable_name)
        return entry.latex if entry else variable_name

    def get_description(self, variable_name: str) -> str:
        """Get description for variable name."""
        entry = self.get_entry(variable_name)
        return entry.description if entry else "Unknown variable"

    def get_variables_by_type(self, variable_type: VariableType) -> list[NotationEntry]:
        """Get all variables of a specific type."""
        return [entry for entry in self.entries.values() if entry.variable_type == variable_type and not entry.aliases]

    def generate_notation_guide(self) -> str:
        """Generate a comprehensive notation guide."""
        guide = """
Mathematical Notation Guide for MFG_PDE Package
==============================================

This guide documents the standard mathematical notation used throughout
the MFG_PDE package, providing clear mappings between mathematical symbols
and computational variable names.

"""

        # Group by variable type
        for var_type in VariableType:
            variables = self.get_variables_by_type(var_type)
            if not variables:
                continue

            guide += f"\n{var_type.value.title()} Variables\n"
            guide += "=" * (len(var_type.value) + 10) + "\n\n"

            for entry in sorted(variables, key=lambda x: x.variable_name):
                guide += f"**{entry.symbol}** (``{entry.variable_name}``)\n"
                guide += f"  {entry.description}\n"
                guide += f"  Domain: {entry.domain}\n"
                if entry.units:
                    guide += f"  Units: {entry.units}\n"
                if entry.aliases:
                    guide += f"  Legacy names: {', '.join(entry.aliases)}\n"
                guide += "\n"

        guide += """
Usage in Documentation
=====================

When writing docstrings or documentation, use both the mathematical
symbol and the variable name for clarity:

.. code-block:: python

    def solve_hjb(self, U: np.ndarray) -> np.ndarray:
        '''Solve Hamilton-Jacobi-Bellman equation for value function.

        Solves: ∂u/∂t + H(x, ∇u, m) = 0 with terminal condition u(T,x).

        Args:
            U: Value function u(t,x) array of shape (Nt+1, Nx+1)

        Returns:
            Updated value function array
        '''

Mathematical Equations in Documentation
=====================================

Use LaTeX notation with double backslashes for proper rendering:

- Value function: $u(t,x)$ → ``$u(t,x)$``
- Density function: $m(t,x)$ → ``$m(t,x)$``
- Hamiltonian: $H(x,\\nabla u, m)$ → ``$H(x,\\\\nabla u, m)$``
- HJB equation: $\\partial_t u + H(x, \\nabla u, m) = 0$

Consistency Rules
================

1. **Array Names**: Use uppercase for solution arrays (U, M)
2. **Parameters**: Use lowercase for scalar parameters (sigma, Dx, Dt)
3. **Functions**: Use lowercase for mathematical functions (H, dH_dm)
4. **Grid Variables**: Preserve mathematical convention (Nx, Nt, Dx, Dt)
5. **Legacy Support**: Maintain backward compatibility through aliases
"""

        return guide


# === Type Aliases for Consistency ===

# Standard array types with clear mathematical meaning
type SpatialArray = np.ndarray  # Shape: (Nx+1,) - spatial discretization
type TemporalArray = np.ndarray  # Shape: (Nt+1,) - temporal discretization
type SolutionArray = np.ndarray  # Shape: (Nt+1, Nx+1) - spatio-temporal solutions
type ParameterDict = dict[str, Any]  # Mathematical and numerical parameters
type ConfigDict = dict[str, Any]  # Solver configuration parameters

# Coordinate and grid types
type SpatialCoordinate = float  # Single spatial point x ∈ [xmin, xmax]
type TemporalCoordinate = float  # Single time point t ∈ [0, T]
type SpatialGrid = np.ndarray  # Full spatial grid array
type TemporalGrid = np.ndarray  # Full temporal grid array

# Mathematical function types
type HamiltonianFunction = float  # H(x, p, m) → ℝ
type GradientArray = np.ndarray  # ∇u computed on grid
type PotentialFunction = float  # V(x) → ℝ


# === Utility Functions ===


def validate_solution_arrays(U: SolutionArray, M: SolutionArray, Nx: int, Nt: int) -> bool:
    """
    Validate that solution arrays conform to mathematical conventions.

    Args:
        U: Value function u(t,x) array
        M: Density function m(t,x) array
        Nx: Number of spatial grid points
        Nt: Number of temporal grid points

    Returns:
        True if arrays are valid MFG solutions

    Raises:
        ValueError: If arrays don't match expected mathematical structure
    """
    expected_shape = (Nt + 1, Nx + 1)

    if U.shape != expected_shape:
        raise ValueError(f"Value function U shape {U.shape} != expected {expected_shape}")

    if M.shape != expected_shape:
        raise ValueError(f"Density function M shape {M.shape} != expected {expected_shape}")

    # Check mathematical constraints
    if np.any(M < 0):
        raise ValueError("Density function m(t,x) must be non-negative")

    if not np.all(np.isfinite(U)):
        raise ValueError("Value function u(t,x) must be finite")

    if not np.all(np.isfinite(M)):
        raise ValueError("Density function m(t,x) must be finite")

    return True


def format_mathematical_summary(U: SolutionArray, M: SolutionArray, problem_params: ParameterDict) -> str:
    """
    Generate a mathematical summary of MFG solution.

    Args:
        U: Value function u(t,x) solution
        M: Density function m(t,x) solution
        problem_params: Problem parameters (sigma, coupling_coefficient, etc.)

    Returns:
        Formatted mathematical summary string
    """
    Nt, Nx = U.shape[0] - 1, U.shape[1] - 1

    summary = f"""
Mathematical Solution Summary
============================

Problem Configuration:
  Spatial domain: x ∈ [0, {problem_params.get("Lx", 1.0)}]
  Temporal domain: t ∈ [0, {problem_params.get("T", 1.0)}]
  Grid resolution: {Nx} × {Nt} points
  Spatial step: Δx = {problem_params.get("Dx", 0.0):.6f}
  Temporal step: Δt = {problem_params.get("Dt", 0.0):.6f}

Physical Parameters:
  Diffusion coefficient: σ = {problem_params.get("sigma", 1.0):.3f}
  Coupling strength: λ = {problem_params.get("coupling_coefficient", 0.5):.3f}

Solution Statistics:
  Value function u(t,x):
    Range: [{np.min(U):.6f}, {np.max(U):.6f}]
    Mean: {np.mean(U):.6f}

  Density function m(t,x):
    Range: [{np.min(M):.6f}, {np.max(M):.6f}]
    Total mass: {np.sum(M) * problem_params.get("Dx", 1.0) * problem_params.get("Dt", 1.0):.6f}
    Conservation check: {"PASS" if abs(np.sum(M[0, :]) - np.sum(M[-1, :])) < 1e-6 else "FAIL"}
"""

    return summary


# Global notation registry instance
global_notation_registry = MFGNotationRegistry()


# Convenience functions for common operations
def get_mathematical_symbol(variable_name: str) -> str:
    """Get mathematical symbol for a variable name."""
    return global_notation_registry.get_symbol(variable_name)


def get_latex_notation(variable_name: str) -> str:
    """Get LaTeX notation for a variable name."""
    return global_notation_registry.get_latex(variable_name)


def generate_variable_documentation(variable_name: str) -> str:
    """Generate complete documentation for a variable."""
    entry = global_notation_registry.get_entry(variable_name)
    if not entry:
        return f"Variable '{variable_name}' not found in notation registry"

    doc = f"""
**{entry.symbol}** (``{entry.variable_name}``)

{entry.description}

- **Mathematical domain**: {entry.domain}
- **LaTeX notation**: ``{entry.latex}``
- **Variable type**: {entry.variable_type.value}
"""

    if entry.units:
        doc += f"- **Physical units**: {entry.units}\n"

    if entry.aliases:
        doc += f"- **Alternative names**: {', '.join(entry.aliases)}\n"

    return doc
