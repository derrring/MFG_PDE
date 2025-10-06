"""
Command-line interface for MFG_PDE.

Provides convenient CLI tools for solving MFG problems, running benchmarks,
and validating the installation.
"""

import sys

import click


@click.group()
@click.version_option(version="1.5.0", prog_name="mfg_pde")
def main():
    """
    MFG_PDE: Mean Field Games PDE Solver

    A comprehensive framework for solving Mean Field Games with advanced
    numerical methods, reinforcement learning, and neural approaches.
    """


@main.command()
@click.option(
    "--problem", "-p", type=str, default="lq_mfg", help="Problem type to solve (lq_mfg, crowd_dynamics, etc.)"
)
@click.option("--nx", type=int, default=50, help="Number of spatial grid points")
@click.option("--nt", type=int, default=20, help="Number of temporal grid points")
@click.option("--solver", "-s", type=str, default="standard", help="Solver type (standard, fast, hybrid)")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file for results (PNG/PDF)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def solve(problem, nx, nt, solver, output, verbose):
    """
    Solve a Mean Field Games problem.

    Examples:
        mfg-solve --problem lq_mfg --nx 50 --nt 20
        mfg-solve -p crowd_dynamics --solver hybrid -v
        mfg-solve --problem lq_mfg --output result.png
    """
    try:
        from mfg_pde import ExampleMFGProblem
        from mfg_pde.factory import create_standard_solver

        if verbose:
            click.echo(f"Setting up {problem} problem...")
            click.echo(f"Grid: Nx={nx}, Nt={nt}")
            click.echo(f"Solver: {solver}")

        # Create problem
        mfg_problem = ExampleMFGProblem(Nx=nx, Nt=nt, T=1.0)

        # Create solver
        if solver == "standard":
            from mfg_pde.factory import create_standard_solver

            solver_obj = create_standard_solver(mfg_problem, "fixed_point")
        elif solver == "fast":
            from mfg_pde.factory import create_fast_solver

            solver_obj = create_fast_solver(mfg_problem, "fixed_point")
        elif solver == "hybrid":
            from mfg_pde.factory import create_standard_solver

            solver_obj = create_standard_solver(mfg_problem, "fixed_point")
        else:
            click.echo(f"Error: Unknown solver type '{solver}'", err=True)
            sys.exit(1)

        # Solve
        if verbose:
            click.echo("Solving MFG system...")

        result = solver_obj.solve()

        # Display results
        click.echo(f"\n{'='*50}")
        click.echo("Solution Summary")
        click.echo(f"{'='*50}")
        click.echo(f"Converged: {result.converged}")
        click.echo(f"Iterations: {result.iterations if hasattr(result, 'iterations') else 'N/A'}")

        if hasattr(result, "mass_conservation_error"):
            click.echo(f"Mass error: {result.mass_conservation_error:.2e}")

        # Save output if requested
        if output:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot value function
            ax1.imshow(result.U, aspect="auto", origin="lower")
            ax1.set_title("Value Function u(t,x)")
            ax1.set_xlabel("Space x")
            ax1.set_ylabel("Time t")

            # Plot density
            ax2.imshow(result.M, aspect="auto", origin="lower")
            ax2.set_title("Density m(t,x)")
            ax2.set_xlabel("Space x")
            ax2.set_ylabel("Time t")

            plt.tight_layout()
            plt.savefig(output, dpi=150, bbox_inches="tight")
            click.echo(f"\nSaved visualization to: {output}")

        click.echo(f"\n{'='*50}")
        click.echo("✓ Solution completed successfully!")

    except ImportError as e:
        click.echo(f"Error: Missing dependency - {e}", err=True)
        click.echo("Run: pip install mfg_pde[dev]", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during solving: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--quick", "-q", is_flag=True, help="Run quick validation (fast tests only)")
def validate(quick):
    """
    Validate MFG_PDE installation.

    Checks that all required components are properly installed and working.

    Examples:
        mfg-validate
        mfg-validate --quick
    """
    click.echo("Validating MFG_PDE installation...")
    click.echo(f"{'='*50}\n")

    all_ok = True

    # Check core imports
    checks = [
        ("Core package", "mfg_pde"),
        ("Factory", "mfg_pde.factory"),
        ("Solvers", "mfg_pde.alg.numerical"),
        ("Utils", "mfg_pde.utils"),
    ]

    for name, module in checks:
        try:
            __import__(module)
            click.echo(f"✓ {name}: OK")
        except ImportError as e:
            click.echo(f"✗ {name}: FAILED - {e}")
            all_ok = False

    # Check optional dependencies
    click.echo(f"\n{'='*50}")
    click.echo("Optional Dependencies:")
    click.echo(f"{'='*50}\n")

    optional = [
        ("JAX (GPU acceleration)", "jax"),
        ("PyTorch (Neural methods)", "torch"),
        ("Plotly (Interactive viz)", "plotly"),
        ("Click (CLI)", "click"),
    ]

    for name, module in optional:
        try:
            __import__(module)
            click.echo(f"✓ {name}: Available")
        except ImportError:
            click.echo(f"○ {name}: Not installed (optional)")

    # Run quick test if requested
    if not quick:
        click.echo(f"\n{'='*50}")
        click.echo("Running integration test...")
        click.echo(f"{'='*50}\n")

        try:
            from mfg_pde import ExampleMFGProblem
            from mfg_pde.factory import create_standard_solver

            problem = ExampleMFGProblem(Nx=10, Nt=5, T=0.5)
            solver = create_standard_solver(problem, "fixed_point")
            result = solver.solve()

            click.echo("✓ Integration test: PASSED")
            click.echo(f"  - Converged: {result.converged}")

        except Exception as e:
            click.echo(f"✗ Integration test: FAILED - {e}")
            all_ok = False

    # Final status
    click.echo(f"\n{'='*50}")
    if all_ok:
        click.echo("✓ Validation PASSED")
        click.echo(f"{'='*50}")
        return 0
    else:
        click.echo("✗ Validation FAILED")
        click.echo(f"{'='*50}")
        sys.exit(1)


@main.command()
@click.option("--solver", "-s", type=str, default="all", help="Solver to benchmark (all, numerical, neural, rl)")
@click.option("--size", type=click.Choice(["small", "medium", "large"]), default="small", help="Problem size")
def benchmark(solver, size):
    """
    Run performance benchmarks.

    Benchmarks different solver types on standard problems.

    Examples:
        mfg-benchmark
        mfg-benchmark --solver numerical --size medium
        mfg-benchmark -s neural
    """
    click.echo(f"Running benchmarks: solver={solver}, size={size}")
    click.echo(f"{'='*50}\n")

    # Define problem sizes
    sizes = {
        "small": (20, 10),
        "medium": (50, 20),
        "large": (100, 50),
    }

    nx, nt = sizes[size]

    try:
        import time

        from mfg_pde import ExampleMFGProblem
        from mfg_pde.factory import create_standard_solver

        problem = ExampleMFGProblem(Nx=nx, Nt=nt, T=1.0)

        click.echo(f"Problem: Nx={nx}, Nt={nt}")
        click.echo(f"DOF: {nx * nt}")
        click.echo(f"\n{'='*50}\n")

        # Benchmark standard solver
        click.echo("Testing standard solver...")
        solver_obj = create_standard_solver(problem, "fixed_point")

        start = time.time()
        result = solver_obj.solve()
        elapsed = time.time() - start

        click.echo(f"Time: {elapsed:.3f}s")
        click.echo(f"Converged: {result.converged}")
        if hasattr(result, "iterations"):
            click.echo(f"Iterations: {result.iterations}")

        click.echo(f"\n{'='*50}")
        click.echo("✓ Benchmark completed")

    except Exception as e:
        click.echo(f"Error during benchmark: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
