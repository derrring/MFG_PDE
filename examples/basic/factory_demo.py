#!/usr/bin/env python3
"""
Unified Problem Factory Demonstration (Phase 3.3)

Demonstrates the new unified factory system that supports all MFG problem types
with a consistent interface.

Key Features:
- Single create_mfg_problem() factory for all types
- Convenience factories: create_lq_problem(), create_crowd_problem(), etc.
- Dual-output support: unified MFGProblem or legacy specialized classes
- Backward compatibility with deprecation warnings

Run:
    python examples/basic/factory_demo.py
"""

import numpy as np

from mfg_pde.core import MFGComponents
from mfg_pde.factory import (
    create_lq_problem,
    create_mfg_problem,
    create_standard_problem,
    create_stochastic_problem,
)
from mfg_pde.geometry import BoundaryConditions, Domain1D
from mfg_pde.solve_mfg import solve_mfg


def demo_lq_problem():
    """Demonstrate Linear-Quadratic MFG factory."""
    print("\n" + "=" * 60)
    print("Demo 1: Linear-Quadratic MFG")
    print("=" * 60)

    # Create 1D domain
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=BoundaryConditions(type="periodic"))

    # Create LQ problem with factory
    print("\nUsing create_lq_problem():")
    print("  problem = create_lq_problem(")
    print("      geometry=domain,")
    print("      terminal_cost=lambda x: x**2,")
    print("      initial_density=lambda x: np.exp(-10*(x-0.5)**2),")
    print("      running_cost_control=0.5,")
    print("      running_cost_congestion=1.0")
    print("  )")

    problem = create_lq_problem(
        geometry=domain,
        terminal_cost=lambda x: x**2,
        initial_density=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        running_cost_control=0.5,
        running_cost_congestion=1.0,
    )

    print(f"\n✓ Created: {type(problem).__name__}")
    print(f"  Problem type: {problem.get_problem_type()}")
    print(f"  Has Hamiltonian: {problem.components.hamiltonian_func is not None}")
    print(f"  Has terminal cost: {problem.components.final_value_func is not None}")


def demo_standard_problem():
    """Demonstrate standard HJB-FP MFG factory."""
    print("\n" + "=" * 60)
    print("Demo 2: Standard HJB-FP MFG")
    print("=" * 60)

    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=BoundaryConditions(type="periodic"))

    # Define custom Hamiltonian
    def hamiltonian(x, p, m, t):
        """Custom Hamiltonian: H = 0.5*p^2 + V(x) + m"""
        V = 0.5 * x * (1 - x)  # Potential
        return 0.5 * p**2 + V + m

    def hamiltonian_dm(x, p, m, t):
        """dH/dm = 1"""
        return 1.0

    print("\nUsing create_standard_problem():")
    print("  problem = create_standard_problem(")
    print("      hamiltonian=H,")
    print("      hamiltonian_dm=dH_dm,")
    print("      terminal_cost=g,")
    print("      initial_density=rho_0,")
    print("      geometry=domain")
    print("  )")

    problem = create_standard_problem(
        hamiltonian=hamiltonian,
        hamiltonian_dm=hamiltonian_dm,
        terminal_cost=lambda x: (x - 0.7) ** 2,
        initial_density=lambda x: np.exp(-10 * (x - 0.3) ** 2),
        geometry=domain,
    )

    print(f"\n✓ Created: {type(problem).__name__}")
    print(f"  Problem type: {problem.get_problem_type()}")


def demo_main_factory():
    """Demonstrate main create_mfg_problem() factory."""
    print("\n" + "=" * 60)
    print("Demo 3: Main Factory with Components")
    print("=" * 60)

    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=BoundaryConditions(type="periodic"))

    # Create components explicitly
    print("\nUsing MFGComponents + create_mfg_problem():")
    print("  components = MFGComponents(")
    print("      hamiltonian_func=H,")
    print("      hamiltonian_dm_func=dH_dm,")
    print("      final_value_func=g,")
    print("      initial_density_func=rho_0,")
    print("      problem_type='standard'")
    print("  )")
    print("  problem = create_mfg_problem('standard', components, geometry=domain)")

    components = MFGComponents(
        hamiltonian_func=lambda x, p, m, t: 0.5 * p**2 + m,
        hamiltonian_dm_func=lambda x, p, m, t: 1.0,
        final_value_func=lambda x: x**2,
        initial_density_func=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        problem_type="standard",
    )

    problem = create_mfg_problem("standard", components, geometry=domain)

    print(f"\n✓ Created: {type(problem).__name__}")
    print(f"  Problem type: {problem.get_problem_type()}")
    print(f"  Components type: {components.problem_type}")


def demo_stochastic_problem():
    """Demonstrate stochastic MFG factory."""
    print("\n" + "=" * 60)
    print("Demo 4: Stochastic MFG with Common Noise")
    print("=" * 60)

    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=BoundaryConditions(type="periodic"))

    print("\nUsing create_stochastic_problem():")
    print("  problem = create_stochastic_problem(")
    print("      hamiltonian=H,")
    print("      hamiltonian_dm=dH_dm,")
    print("      terminal_cost=g,")
    print("      initial_density=rho_0,")
    print("      geometry=domain,")
    print("      noise_intensity=0.5,")
    print("      common_noise=lambda t: np.sin(2*np.pi*t)")
    print("  )")

    problem = create_stochastic_problem(
        hamiltonian=lambda x, p, m, t: 0.5 * p**2 + m,
        hamiltonian_dm=lambda x, p, m, t: 1.0,
        terminal_cost=lambda x: x**2,
        initial_density=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        geometry=domain,
        noise_intensity=0.5,
        common_noise=lambda t: np.sin(2 * np.pi * t),
    )

    print(f"\n✓ Created: {type(problem).__name__}")
    print(f"  Problem type: {problem.get_problem_type()}")
    print(f"  Noise intensity: {problem.components.noise_intensity}")
    print(f"  Has common noise: {problem.components.common_noise_func is not None}")


def demo_solve_integrated():
    """Demonstrate solving with unified factory + config."""
    print("\n" + "=" * 60)
    print("Demo 5: End-to-End Solve (Factory + Config)")
    print("=" * 60)

    # Create problem with factory
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=BoundaryConditions(type="periodic"))

    problem = create_lq_problem(
        geometry=domain,
        terminal_cost=lambda x: (x - 0.7) ** 2,
        initial_density=lambda x: np.exp(-10 * (x - 0.3) ** 2),
        running_cost_control=0.5,
        running_cost_congestion=1.0,
    )

    print("\n1. Problem created with factory")
    print(f"   Type: {type(problem).__name__}")

    # Solve with new config API
    print("\n2. Solving with new config API:")
    print("   result = solve_mfg(problem, config='accurate')")

    result = solve_mfg(problem, config="accurate", verbose=False)

    print("\n✓ Solved successfully!")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error: {result.error_history_U[-1]:.2e}")
    if result.execution_time:
        print(f"  Time: {result.execution_time:.3f}s")


def demo_backward_compatibility():
    """Show backward compatibility with legacy API."""
    print("\n" + "=" * 60)
    print("Demo 6: Backward Compatibility")
    print("=" * 60)

    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=BoundaryConditions(type="periodic"))

    print("\nNew unified API (default, use_unified=True):")
    problem_new = create_lq_problem(
        geometry=domain,
        terminal_cost=lambda x: x**2,
        initial_density=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        use_unified=True,  # Default
    )
    print(f"  Type: {type(problem_new).__name__}")
    print(f"  Problem type: {problem_new.get_problem_type()}")

    print("\nLegacy API (use_unified=False, deprecated):")
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        problem_old = create_lq_problem(
            geometry=domain,
            terminal_cost=lambda x: x**2,
            initial_density=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            use_unified=False,
        )

        if w:
            print(f"  ⚠️  Deprecation warning: {w[0].category.__name__}")

    print(f"  Type: {type(problem_old).__name__}")
    print("\n✓ Both APIs work, but new unified API is recommended")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Unified Problem Factory Demonstration")
    print("Phase 3.3: Complete MFG Problem Type Support")
    print("=" * 60)

    demo_lq_problem()
    demo_standard_problem()
    demo_main_factory()
    demo_stochastic_problem()
    demo_solve_integrated()
    demo_backward_compatibility()

    print("\n" + "=" * 60)
    print("All demonstrations complete")
    print("=" * 60)
    print("\nKey Features:")
    print("  1. Unified MFGProblem class for all types")
    print("  2. Convenience factories for common patterns")
    print("  3. Main create_mfg_problem() for custom types")
    print("  4. Backward compatibility maintained")
    print("  5. Integrates with Phase 3.2 SolverConfig")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
