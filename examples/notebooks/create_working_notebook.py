#!/usr/bin/env python3
"""
Create a working Jupyter notebook for MFG demonstration.
"""

import numpy as np
import json
from pathlib import Path

def create_mfg_notebook():
    """Create a working MFG demonstration notebook."""
    
    # Create notebook structure
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MFG_PDE Framework Demonstration\n",
                    "\n",
                    "This notebook demonstrates the modern MFG_PDE framework with:\n",
                    "- Pydantic configuration and validation\n",
                    "- Professional logging system\n",
                    "- Array validation with physical constraints\n",
                    "- Interactive visualizations\n",
                    "\n",
                    "## Mathematical Framework\n",
                    "\n",
                    "We solve the Mean Field Game system:\n",
                    "\n",
                    "**Hamilton-Jacobi-Bellman equation:**\n",
                    "$$-\\frac{\\partial u}{\\partial t} + H(t,x,\\nabla u, m) = 0$$\n",
                    "\n",
                    "**Fokker-Planck equation:**\n",
                    "$$\\frac{\\partial m}{\\partial t} - \\nabla \\cdot (m \\nabla H_p) - \\frac{\\sigma^2}{2}\\Delta m = 0$$\n",
                    "\n",
                    "Where $u(t,x)$ is the value function and $m(t,x)$ is the agent density."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import required libraries\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from mpl_toolkits.mplot3d import Axes3D\n",
                    "\n",
                    "# MFG_PDE framework\n",
                    "from mfg_pde.core.mfg_problem import ExampleMFGProblem\n",
                    "from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver\n",
                    "from mfg_pde.config.array_validation import MFGGridConfig, MFGArrays\n",
                    "from mfg_pde.config.pydantic_config import create_research_config\n",
                    "\n",
                    "print(\"📦 MFG_PDE framework loaded successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Problem Setup with Pydantic Validation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create validated grid configuration\n",
                    "grid_config = MFGGridConfig(\n",
                    "    Nx=30,           # Spatial points\n",
                    "    Nt=15,           # Time points  \n",
                    "    xmin=0.0,\n",
                    "    xmax=1.0,\n",
                    "    T=0.3,           # Final time\n",
                    "    sigma=0.08       # Diffusion coefficient\n",
                    ")\n",
                    "\n",
                    "print(f\"Grid Configuration:\")\n",
                    "print(f\"  - Spatial points: {grid_config.Nx}\")\n",
                    "print(f\"  - Temporal points: {grid_config.Nt}\")\n",
                    "print(f\"  - Domain: [{grid_config.xmin}, {grid_config.xmax}]\")\n",
                    "print(f\"  - Time horizon: {grid_config.T}\")\n",
                    "print(f\"  - CFL number: {grid_config.cfl_number:.3f}\")\n",
                    "\n",
                    "# Check stability\n",
                    "if grid_config.cfl_number < 0.5:\n",
                    "    print(\"  ✅ CFL condition satisfied (stable)\")\n",
                    "else:\n",
                    "    print(\"  ⚠️ CFL condition may cause instability\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create MFG problem\n",
                    "problem = ExampleMFGProblem(\n",
                    "    xmin=grid_config.xmin,\n",
                    "    xmax=grid_config.xmax, \n",
                    "    T=grid_config.T,\n",
                    "    Nx=grid_config.Nx,\n",
                    "    Nt=grid_config.Nt,\n",
                    "    sigma=grid_config.sigma\n",
                    ")\n",
                    "\n",
                    "print(\"🎯 MFG problem created successfully!\")\n",
                    "print(f\"Problem parameters:\")\n",
                    "print(f\"  - σ (diffusion): {problem.sigma}\")\n",
                    "print(f\"  - Grid shape: {problem.Nx+1} × {problem.Nt+1}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Solver Configuration"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create research-grade configuration\n",
                    "config = create_research_config()\n",
                    "\n",
                    "print(\"📋 Research Configuration:\")\n",
                    "print(f\"  - Newton max iterations: {config.newton.max_iterations}\")\n",
                    "print(f\"  - Newton tolerance: {config.newton.tolerance}\")\n",
                    "print(f\"  - Picard max iterations: {config.picard.max_iterations}\")\n",
                    "print(f\"  - Picard tolerance: {config.picard.tolerance}\")\n",
                    "print(f\"  - Structured output: {config.return_structured}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Setup particle collocation solver\n",
                    "np.random.seed(123)  # Reproducible results\n",
                    "num_collocation = 20\n",
                    "collocation_points = np.random.uniform(0, 1, (num_collocation, 1))\n",
                    "\n",
                    "solver = ParticleCollocationSolver(\n",
                    "    problem=problem,\n",
                    "    collocation_points=collocation_points,\n",
                    "    num_particles=800,\n",
                    "    kde_bandwidth=0.06\n",
                    ")\n",
                    "\n",
                    "print(f\"🔬 Particle Collocation Solver configured:\")\n",
                    "print(f\"  - Particles: 800\")\n",
                    "print(f\"  - Collocation points: {num_collocation}\")\n",
                    "print(f\"  - KDE bandwidth: 0.06\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Solve the MFG System"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Solve the MFG system\n",
                    "import time\n",
                    "\n",
                    "print(\"🚀 Starting MFG solution...\")\n",
                    "start_time = time.time()\n",
                    "\n",
                    "result = solver.solve(\n",
                    "    max_iterations=10,\n",
                    "    tolerance=1e-3,\n",
                    "    verbose=False  # Set to True for detailed output\n",
                    ")\n",
                    "\n",
                    "solve_time = time.time() - start_time\n",
                    "U, M, info = result\n",
                    "\n",
                    "print(f\"✅ Solution completed in {solve_time:.2f} seconds\")\n",
                    "print(f\"Solution arrays shape: {U.shape}\")\n",
                    "if isinstance(info, dict) and 'converged' in info:\n",
                    "    print(f\"Convergence: {'Yes' if info['converged'] else 'No'}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Solution Validation with Pydantic"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Validate solution with Pydantic\n",
                    "print(\"🔍 Validating solution with Pydantic...\")\n",
                    "\n",
                    "try:\n",
                    "    arrays = MFGArrays(\n",
                    "        U_solution=U,\n",
                    "        M_solution=M, \n",
                    "        grid_config=grid_config\n",
                    "    )\n",
                    "    \n",
                    "    # Get comprehensive statistics\n",
                    "    stats = arrays.get_solution_statistics()\n",
                    "    \n",
                    "    print(\"✅ Validation passed!\")\n",
                    "    print(\"\\n📊 Solution Statistics:\")\n",
                    "    print(f\"  U (value function):\")\n",
                    "    print(f\"    - Range: [{stats['U']['min']:.3f}, {stats['U']['max']:.3f}]\")\n",
                    "    print(f\"    - Mean: {stats['U']['mean']:.3f}\")\n",
                    "    \n",
                    "    print(f\"  M (density):\")\n",
                    "    print(f\"    - Range: [{stats['M']['min']:.3f}, {stats['M']['max']:.3f}]\")\n",
                    "    print(f\"    - Mean: {stats['M']['mean']:.3f}\")\n",
                    "    \n",
                    "    print(f\"  Mass Conservation:\")\n",
                    "    print(f\"    - Initial mass: {stats['mass_conservation']['initial_mass']:.6f}\")\n",
                    "    print(f\"    - Final mass: {stats['mass_conservation']['final_mass']:.6f}\")\n",
                    "    print(f\"    - Mass drift: {stats['mass_conservation']['mass_drift']:.2e}\")\n",
                    "    \n",
                    "    mass_drift = abs(stats['mass_conservation']['mass_drift'])\n",
                    "    if mass_drift < 0.01:\n",
                    "        print(\"    ✅ Excellent mass conservation\")\n",
                    "    elif mass_drift < 0.05:\n",
                    "        print(\"    👍 Good mass conservation\")\n",
                    "    else:\n",
                    "        print(\"    ⚠️ Mass conservation needs attention\")\n",
                    "        \n",
                    "except Exception as e:\n",
                    "    print(f\"❌ Validation failed: {e}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Visualizations"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create grids for plotting\n",
                    "x_grid = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)\n",
                    "t_grid = np.linspace(0, problem.T, problem.Nt + 1)\n",
                    "X, T = np.meshgrid(x_grid, t_grid)\n",
                    "\n",
                    "print(f\"📊 Creating visualizations...\")\n",
                    "print(f\"Grid dimensions: {len(x_grid)} × {len(t_grid)}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3D Surface plots\n",
                    "fig = plt.figure(figsize=(15, 6))\n",
                    "\n",
                    "# Value function u(t,x)\n",
                    "ax1 = fig.add_subplot(1, 2, 1, projection='3d')\n",
                    "surf1 = ax1.plot_surface(T, X, U, cmap='viridis', alpha=0.8)\n",
                    "ax1.set_xlabel('Time t')\n",
                    "ax1.set_ylabel('Space x')\n",
                    "ax1.set_zlabel('u(t,x)')\n",
                    "ax1.set_title('Value Function u(t,x)')\n",
                    "\n",
                    "# Density m(t,x)\n",
                    "ax2 = fig.add_subplot(1, 2, 2, projection='3d')\n",
                    "surf2 = ax2.plot_surface(T, X, M, cmap='plasma', alpha=0.8)\n",
                    "ax2.set_xlabel('Time t')\n",
                    "ax2.set_ylabel('Space x')\n",
                    "ax2.set_zlabel('m(t,x)')\n",
                    "ax2.set_title('Density m(t,x)')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(\"✅ 3D surface plots displayed\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Contour plots\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
                    "\n",
                    "# Value function contours\n",
                    "contour1 = ax1.contourf(T, X, U, levels=20, cmap='viridis')\n",
                    "ax1.set_xlabel('Time t')\n",
                    "ax1.set_ylabel('Space x')\n",
                    "ax1.set_title('Value Function u(t,x) - Contours')\n",
                    "plt.colorbar(contour1, ax=ax1)\n",
                    "\n",
                    "# Density contours\n",
                    "contour2 = ax2.contourf(T, X, M, levels=20, cmap='plasma')\n",
                    "ax2.set_xlabel('Time t')\n",
                    "ax2.set_ylabel('Space x')\n",
                    "ax2.set_title('Density m(t,x) - Contours')\n",
                    "plt.colorbar(contour2, ax=ax2)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(\"✅ Contour plots displayed\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Time evolution plots\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
                    "\n",
                    "# Select time snapshots\n",
                    "time_indices = [0, len(t_grid)//3, 2*len(t_grid)//3, -1]\n",
                    "colors = ['blue', 'green', 'orange', 'red']\n",
                    "\n",
                    "# Value function evolution\n",
                    "for i, t_idx in enumerate(time_indices):\n",
                    "    ax1.plot(x_grid, U[t_idx], label=f't={t_grid[t_idx]:.2f}', \n",
                    "            color=colors[i], linewidth=2, alpha=0.8)\n",
                    "ax1.set_xlabel('Space x')\n",
                    "ax1.set_ylabel('u(t,x)')\n",
                    "ax1.set_title('Value Function Evolution')\n",
                    "ax1.legend()\n",
                    "ax1.grid(True, alpha=0.3)\n",
                    "\n",
                    "# Density evolution\n",
                    "for i, t_idx in enumerate(time_indices):\n",
                    "    ax2.plot(x_grid, M[t_idx], label=f't={t_grid[t_idx]:.2f}', \n",
                    "            color=colors[i], linewidth=2, alpha=0.8)\n",
                    "ax2.set_xlabel('Space x')\n",
                    "ax2.set_ylabel('m(t,x)')\n",
                    "ax2.set_title('Density Evolution')\n",
                    "ax2.legend()\n",
                    "ax2.grid(True, alpha=0.3)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(\"✅ Time evolution plots displayed\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Summary\n",
                    "\n",
                    "This notebook demonstrated the complete MFG_PDE framework:\n",
                    "\n",
                    "✅ **Pydantic Configuration**: Automatic validation and type checking  \n",
                    "✅ **Physical Constraints**: Mass conservation and CFL stability monitoring  \n",
                    "✅ **Modern Solvers**: Particle collocation with configurable parameters  \n",
                    "✅ **Professional Validation**: Comprehensive solution statistics  \n",
                    "✅ **Scientific Visualization**: Multiple plot types with proper notation  \n",
                    "\n",
                    "The framework provides a robust foundation for Mean Field Game research with:\n",
                    "- Type safety and automatic validation\n",
                    "- Physical constraint checking\n",
                    "- Professional logging and monitoring\n",
                    "- Interactive analysis capabilities\n",
                    "\n",
                    "### Mathematical Consistency\n",
                    "All notation follows the standard convention:\n",
                    "- Value function: $u(t,x)$ (time first)\n",
                    "- Density function: $m(t,x)$ (time first)\n",
                    "- Array indexing: `U[t_idx, x_idx]` matches mathematical convention"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebook
    output_dir = Path("./results/working_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    notebook_path = output_dir / "mfg_demonstration.ipynb"
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"📓 Notebook created: {notebook_path}")
    return notebook_path

if __name__ == "__main__":
    create_mfg_notebook()